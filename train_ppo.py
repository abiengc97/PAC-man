"""
PPO training for PAC-man using Stable-Baselines3.

Hyperparameters follow **NeymarL/Pacman-RL** (`/Pacman-RL/src/ppo.py`, `config.py`,
`manager.py`): π learning rate 1e-4, value head 10× faster (1e-3), γ=0.9, GAE λ=0.5,
entropy coeff 0.01, target KL 0.01 (early stop in TF when KL > 1.5×). The original
repo uses `config.controller.epsilon` (~0.5) as the PPO clip ratio; this script
defaults to **clip_range=0.2** (common in SB3) — pass `--clip-range 0.5` to match
the TF repo literally.

Protocol (one JSON line per message, newline-delimited):
  Python → C++:  {"action": N}         N = 0:UP 1:DOWN 2:LEFT 3:RIGHT
  Python → C++:  {"reset": true}
  C++    → Python: {"state":[...940 floats...], "reward":R, "done":B,
                    "lives":L, "score":S}

Usage:
  pip install stable-baselines3 gymnasium torch
  python train_ppo.py [--envs N] [--export-only]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE       = os.path.dirname(os.path.abspath(__file__))
PACMAN_BIN  = os.path.join(_HERE, "build", "pacman")

STATE_SIZE      = 940   # must match Game::RL_STATE_SIZE in C++
N_ACTIONS       = 4     # UP DOWN LEFT RIGHT
MAZE_ROWS       = 31
MAZE_COLS       = 28
MAZE_CELLS      = MAZE_ROWS * MAZE_COLS   # 868 — full maze at end of state vector
NON_SPATIAL     = STATE_SIZE - MAZE_CELLS  # 72 — scalars / ghost info at front


# ---------------------------------------------------------------------------
# Pacman-RL–aligned PPO hyperparameters (SB3 mapping)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PacmanRLPPOHparams:
    """Defaults aligned with Pacman-RL: PPOActor/PPOCritic + manager.py CLI defaults."""

    pi_lr: float = 1e-4           # TrainerConfig.lr
    value_lr_mult: float = 10.0   # v_lr=1e-3 vs pi_lr=1e-4 in ppo.py
    gamma: float = 0.9            # --gamma in manager.py
    gae_lambda: float = 0.5       # --lam (PPO GAE λ)
    clip_range: float = 0.2       # Pacman-RL uses epsilon≈0.5 as clip; 0.2 is safer in SB3
    ent_coef: float = 0.01        # pi_loss -= 0.01 * entropy in ppo.py
    target_kl: float = 0.01       # PPOActor.target_kl; SB3 stops epochs when KL explodes
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10            # TF side uses up to 80 inner steps; epoch count differs


# ---------------------------------------------------------------------------
# CNN Feature Extractor
# ---------------------------------------------------------------------------

class PacManCNNExtractor(BaseFeaturesExtractor):
    """
    Splits the flat state vector into:
      - non_spatial (72 floats): ghost positions, timers, scalars
      - maze (868 floats → 1×31×28): full live maze layout

    A small CNN processes the maze spatially; its output is concatenated
    with the non-spatial features and fed through a linear layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # CNN: processes maze as (batch, 1, 31, 28)
        # After MaxPool2d(2): 31→15, 28→14 → 7→7
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # → 32×31×28
            nn.Tanh(),
            nn.MaxPool2d(2),                               # → 32×15×14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → 64×15×14
            nn.Tanh(),
            nn.MaxPool2d(2),                               # → 64×7×7
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # → 64×7×7
            nn.Tanh(),
            nn.Flatten(),                                  # → 3136
        )

        cnn_out = 64 * 7 * 7  # 3136

        self.linear = nn.Sequential(
            nn.Linear(cnn_out + NON_SPATIAL, features_dim),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        non_spatial = observations[:, :NON_SPATIAL]
        maze_flat   = observations[:, NON_SPATIAL:]                  # (batch, 868)
        maze_2d     = maze_flat.view(-1, 1, MAZE_ROWS, MAZE_COLS) / 6.0  # normalise 0→1
        cnn_out     = self.cnn(maze_2d)
        return self.linear(torch.cat([non_spatial, cnn_out], dim=1))


def policy_kwargs_dict() -> dict:
    return dict(
        features_extractor_class=PacManCNNExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[256, 128],
        activation_fn=nn.Tanh,
    )


# ---------------------------------------------------------------------------
# Dual-LR callback: value network trains 10x faster than actor (from Pacman-RL)
# ---------------------------------------------------------------------------

class DualLRCallback(BaseCallback):
    """
    SB3 uses a single optimizer for the whole policy.  This callback splits it
    by injecting per-parameter-group learning rates after the first optimizer
    step, giving the value network a 10x higher LR than the actor — matching
    the NeymarL/Pacman-RL design (pi_lr=1e-4, v_lr=1e-3).
    """

    def __init__(self, policy_lr: float = 1e-4, value_lr_multiplier: float = 10.0,
                 verbose: int = 0):
        super().__init__(verbose)
        self.policy_lr = policy_lr
        self.value_lr = policy_lr * value_lr_multiplier
        self._patched = False

    def _on_training_start(self) -> None:
        self._patch_optimizer()

    def _patch_optimizer(self) -> None:
        if self._patched:
            return
        policy = self.model.policy
        opt = policy.optimizer

        # Separate params into value-net and everything else
        value_params, actor_params = [], []
        for name, param in policy.named_parameters():
            if "value_net" in name:
                value_params.append(param)
            else:
                actor_params.append(param)

        # Replace single param group with two groups at different LRs
        opt.param_groups.clear()
        opt.add_param_group({"params": actor_params, "lr": self.policy_lr})
        opt.add_param_group({"params": value_params, "lr": self.value_lr})
        self._patched = True
        if self.verbose:
            print(f"  [DualLR] actor_lr={self.policy_lr:.0e}  value_lr={self.value_lr:.0e}")

    def _on_step(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------

class PacManEnv(gym.Env):
    """Wraps the headless C++ Pac-Man game via stdin/stdout JSON."""

    metadata = {"render_modes": []}

    MAX_STEPS = 8_000  # ~133s of gameplay at 60fps; breaks looping strategies

    def __init__(self, level: int = 1):
        super().__init__()
        self.level = level
        self.observation_space = gym.spaces.Box(
            low=-2.0, high=10.0, shape=(STATE_SIZE,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self._proc: subprocess.Popen | None = None
        self._last_obs = np.zeros(STATE_SIZE, dtype=np.float32)
        self._step_count = 0

    # ------------------------------------------------------------------

    def _start_process(self) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                pass
        self._proc = subprocess.Popen(
            [PACMAN_BIN, "--headless", "--rl", f"--level={self.level}"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,   # captured so we can show on crash
            text=True,
            bufsize=1,
        )
        # Give the process a moment, then check it didn't immediately crash
        import time; time.sleep(0.05)
        if self._proc.poll() is not None:
            stderr_out = self._proc.stderr.read()
            raise RuntimeError(
                f"pacman exited immediately (code {self._proc.returncode}).\n"
                f"stderr: {stderr_out}\n"
                f"Did you rebuild?  cd build && make -j$(nproc)"
            )
        # C++ writes the initial state on startup
        data = self._read_step()
        self._last_obs = np.array(data["state"], dtype=np.float32)

    def _write(self, msg: str) -> None:
        assert self._proc and self._proc.stdin
        self._proc.stdin.write(msg + "\n")
        self._proc.stdin.flush()

    def _read_step(self) -> dict:
        assert self._proc and self._proc.stdout
        line = self._proc.stdout.readline()
        if not line:
            # Process died unexpectedly — return a terminal stub
            return {"state": self._last_obs.tolist(), "reward": -50.0,
                    "done": True, "lives": 0, "score": 0}
        return json.loads(line)

    # ------------------------------------------------------------------
    # gymnasium API

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        if self._proc is None:
            self._start_process()
        else:
            self._write('{"reset":true}')
            data = self._read_step()
            self._last_obs = np.array(data["state"], dtype=np.float32)
        return self._last_obs, {}

    def step(self, action: int):
        self._write(f'{{"action":{int(action)}}}')
        data = self._read_step()

        obs        = np.array(data["state"], dtype=np.float32)
        reward     = float(data["reward"])
        terminated = bool(data["done"])
        self._step_count += 1
        truncated  = self._step_count >= self.MAX_STEPS
        info       = {"lives": data["lives"], "score": data["score"]}

        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                pass
            self._proc = None


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

# (start_level, timesteps_on_this_level)
CURRICULUM = [
    (1,  2_000_000),
    (2,  2_000_000),
    (3,  2_000_000),
    (5,  3_000_000),
    (8,  3_000_000),
    (10, 4_000_000),
]


def make_env_fn(level: int):
    def _init():
        env = PacManEnv(level=level)
        env = Monitor(env)
        return env
    return _init


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    n_envs: int = 8,
    bc_init: str | None = None,
    hp: PacmanRLPPOHparams | None = None,
) -> PPO:
    if hp is None:
        hp = PacmanRLPPOHparams()

    print("\nPacman-RL–aligned PPO (SB3):", dataclasses.asdict(hp))

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("models",      exist_ok=True)

    pkw = policy_kwargs_dict()
    common = dict(
        learning_rate=hp.pi_lr,
        n_steps=hp.n_steps,
        batch_size=hp.batch_size,
        n_epochs=hp.n_epochs,
        gamma=hp.gamma,
        gae_lambda=hp.gae_lambda,
        clip_range=hp.clip_range,
        ent_coef=hp.ent_coef,
        target_kl=hp.target_kl,
        verbose=1,
        tensorboard_log="./tb_logs/",
    )

    # ---- Build initial env & model ----------------------------------------
    env = make_vec_env(make_env_fn(CURRICULUM[0][0]), n_envs=n_envs)

    if bc_init and os.path.isfile(bc_init):
        print(f"\nLoading BC pre-trained weights from {bc_init} ...")
        # Architecture must match the BC checkpoint; only override optimisation hparams.
        model = PPO.load(bc_init, env, **common)
        print("BC weights loaded — PPO will continue with Pacman-RL–aligned hparams.\n")
    else:
        model = PPO("MlpPolicy", env, policy_kwargs=pkw, **common)

    dual_lr = DualLRCallback(
        policy_lr=hp.pi_lr,
        value_lr_multiplier=hp.value_lr_mult,
        verbose=1,
    )

    # ---- Curriculum loop --------------------------------------------------
    for level, timesteps in CURRICULUM:
        print(f"\n{'='*55}")
        print(f"  Level {level:>2}  —  {timesteps:,} steps  ({n_envs} envs)")
        print(f"{'='*55}")

        env = make_vec_env(make_env_fn(level), n_envs=n_envs)
        model.set_env(env)

        checkpoint_cb = CheckpointCallback(
            save_freq=max(100_000 // n_envs, 1),
            save_path=f"checkpoints/level{level}/",
            name_prefix="ppo_pacman",
            verbose=0,
        )

        callbacks = [checkpoint_cb, dual_lr]

        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            reset_num_timesteps=False,   # keep global step counter
            tb_log_name=f"level{level}",
        )

        save_path = f"models/ppo_level{level}_final"
        model.save(save_path)
        print(f"  Saved → {save_path}.zip")
        env.close()

    return model


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(model: PPO, path: str = "pacman_policy.onnx") -> None:
    """Export the policy network to ONNX with the names expected by AIAgent.h."""

    class _PolicyWrapper(torch.nn.Module):
        """Exports CNN extractor + actor MLP (avoids Categorical distribution tracing issues)."""
        def __init__(self, policy):
            super().__init__()
            self.features_extractor = policy.features_extractor
            self.mlp = policy.mlp_extractor.policy_net
            self.action_net = policy.action_net

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            features = self.features_extractor(state)
            return self.action_net(self.mlp(features))

    wrapper = _PolicyWrapper(model.policy).cpu().eval()
    dummy   = torch.zeros(1, STATE_SIZE)  # CPU

    torch.onnx.export(
        wrapper, dummy, path,
        input_names=["state"],
        output_names=["action_logits"],
        dynamic_axes={"state": {0: "batch"}, "action_logits": {0: "batch"}},
        # Use a modern opset to avoid exporter attempting (and failing) to
        # down-convert ops (e.g., Gemm) to older opset versions.
        opset_version=18,
    )
    print(f"Exported ONNX model → {path}")
    print("Copy it to build/ and it will be loaded by AIAgent at runtime.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def hparams_from_args(args: argparse.Namespace) -> PacmanRLPPOHparams:
    """CLI overrides on top of Pacman-RL defaults."""
    return PacmanRLPPOHparams(
        pi_lr=args.lr,
        value_lr_mult=args.value_lr_mult,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        target_kl=args.target_kl,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
    )


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for PAC-man")
    parser.add_argument("--envs",        type=int,  default=8,
                        help="Number of parallel environments (default: 8)")
    parser.add_argument("--export-only", type=str,  default=None,
                        metavar="CHECKPOINT",
                        help="Skip training; export CHECKPOINT.zip to ONNX")
    parser.add_argument("--bc-init",    type=str,  default=None,
                        metavar="BC_ZIP",
                        help="BC pre-trained weights to load before PPO (from bc_pretrain.py)")
    d = PacmanRLPPOHparams()
    parser.add_argument("--lr", type=float, default=d.pi_lr,
                        help=f"Policy LR (Pacman-RL TrainerConfig.lr, default {d.pi_lr})")
    parser.add_argument("--value-lr-mult", type=float, default=d.value_lr_mult,
                        help=f"Value LR = lr × this (Pacman-RL π=1e-4 v=1e-3 → 10, default {d.value_lr_mult})")
    parser.add_argument("--gamma", type=float, default=d.gamma,
                        help=f"Discount (Pacman-RL default {d.gamma})")
    parser.add_argument("--gae-lambda", type=float, default=d.gae_lambda,
                        help=f"GAE λ (Pacman-RL --lam, default {d.gae_lambda})")
    parser.add_argument("--clip-range", type=float, default=d.clip_range,
                        help=f"PPO clip (TF repo uses ~0.5 as epsilon; SB3-safe default {d.clip_range})")
    parser.add_argument("--ent-coef", type=float, default=d.ent_coef,
                        help=f"Entropy bonus coefficient (Pacman-RL 0.01, default {d.ent_coef})")
    parser.add_argument("--target-kl", type=float, default=d.target_kl,
                        help=f"Target KL (Pacman-RL actor target_kl, default {d.target_kl})")
    parser.add_argument("--n-steps", type=int, default=d.n_steps, help="Steps per env per rollout")
    parser.add_argument("--batch-size", type=int, default=d.batch_size, help="Minibatch size for PPO")
    parser.add_argument("--n-epochs", type=int, default=d.n_epochs, help="Optimization epochs per rollout")
    args = parser.parse_args()

    if not os.path.isfile(PACMAN_BIN):
        print(f"ERROR: pacman binary not found at {PACMAN_BIN}")
        print("       Run:  cd build && cmake .. && make")
        raise SystemExit(1)

    if args.export_only:
        # Build a fresh model with the correct architecture, then load only policy weights.
        # PPO.load fails when the checkpoint has a different optimizer group count (DualLR).
        import zipfile, io
        env = make_vec_env(make_env_fn(1), n_envs=1)
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs_dict(), verbose=0)
        env.close()
        # Extract policy.pth from the zip and load only the policy state dict
        with zipfile.ZipFile(args.export_only, "r") as zf:
            with zf.open("policy.pth") as f:
                params = torch.load(io.BytesIO(f.read()), map_location="cpu")
        model.policy.load_state_dict(params, strict=False)
        model.policy.eval()
        export_onnx(model)
    else:
        hp = hparams_from_args(args)
        model = train(n_envs=args.envs, bc_init=args.bc_init, hp=hp)
        export_onnx(model)


if __name__ == "__main__":
    main()
