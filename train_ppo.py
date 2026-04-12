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
  C++    → Python: {"state":[...89 floats...], "reward":R, "done":B,
                    "lives":L, "score":S}

Usage:
  pip install stable-baselines3 gymnasium torch wandb
  wandb login                              # one-time auth
  python train_ppo.py [--envs N] [--wandb] [--wandb-project pacman-rl]
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
from stable_baselines3.common.vec_env import VecNormalize

# Optional experiment tracking (only used when enabled via CLI)
try:
    import wandb  # type: ignore
    from wandb.integration.sb3 import WandbCallback  # type: ignore
except Exception:  # pragma: no cover
    wandb = None
    WandbCallback = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE       = os.path.dirname(os.path.abspath(__file__))
PACMAN_BIN  = os.path.join(_HERE, "build", "pacman")

STATE_SIZE  = 89    # must match Game::RL_STATE_SIZE in C++
N_ACTIONS   = 4     # UP DOWN LEFT RIGHT
MAZE_ROWS   = 31    # exported for bc_pretrain.py
MAZE_COLS   = 28    # exported for bc_pretrain.py


# ---------------------------------------------------------------------------
# Pacman-RL–aligned PPO hyperparameters (SB3 mapping)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PacmanRLPPOHparams:
    """Defaults aligned with Pacman-RL: PPOActor/PPOCritic + manager.py CLI defaults."""

    pi_lr: float = 1e-4           # TrainerConfig.lr
    value_lr_mult: float = 10.0   # v_lr=1e-3 vs pi_lr=1e-4 in ppo.py
    gamma: float = 0.97           # raised from 0.9 — Pac-Man has long-horizon dependencies
                                  # (power-pellet payoff spans ~6 s; γ=0.9 discounts 20 steps to 12%)
    gae_lambda: float = 0.995     # raised from 0.5 — less biased advantage estimates
    clip_range: float = 0.2       # Pacman-RL uses epsilon≈0.5 as clip; 0.2 is safer in SB3
    ent_coef: float = 0.03        # raised from 0.01: higher entropy prevents early collapse
                                  # into wall-hugging loops before the full maze is explored
    target_kl: float = 0.05       # loosened from 0.01 — 0.01 caused frequent early-stop,
                                  # barely moving the policy per update
    n_steps: int = 4096           # raised from 2048 — 8 envs × 4096 = 32k samples/update,
                                  # more reliably captures complete episodes (MAX_STEPS=8000)
    batch_size: int = 256
    n_epochs: int = 10            # TF side uses up to 80 inner steps; epoch count differs


def policy_kwargs_dict() -> dict:
    # All 89 state features are already normalised to [0, 1].
    # A deeper MLP replaces the CNN — simpler, faster, and sufficient for
    # a fully-observable scalar state.
    return dict(
        net_arch=[256, 256, 128],
        activation_fn=nn.Tanh,
    )


# ---------------------------------------------------------------------------
# WandB game-metrics callback
# ---------------------------------------------------------------------------

class WandbGameMetricsCallback(BaseCallback):
    """Logs PAC-man game-specific episode stats to Weights & Biases.

    SB3's built-in WandbCallback only captures RL reward (ep_rew_mean) and
    episode length.  This callback additionally tracks:
      game/score_mean   — real Pac-Man score (not RL reward)
      game/score_max    — best episode score in this rollout window
      game/lives_mean   — average remaining lives when episode ended
      game/score_std    — spread of scores (useful to detect collapse)

    Scores / lives are read from info["score"] / info["lives"] which
    PacManEnv passes on every step; Monitor stores them per-episode via
    info["episode"]["score"] (we fall back to the step-level keys if absent).
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._ep_scores: list[float] = []
        self._ep_lives:  list[float] = []

    def _on_step(self) -> bool:
        if wandb is None or wandb.run is None:
            return True
        infos = self.locals.get("infos", [])
        for info in infos:
            # Monitor adds info["episode"] dict when an episode finishes.
            if "episode" in info:
                # Prefer score stored at step level (set just before done).
                score = info.get("score", info["episode"].get("score", 0))
                lives = info.get("lives", info["episode"].get("lives", 0))
                self._ep_scores.append(float(score))
                self._ep_lives.append(float(lives))
        return True

    def _on_rollout_end(self) -> None:
        if wandb is None or wandb.run is None or not self._ep_scores:
            return
        scores = np.array(self._ep_scores, dtype=np.float32)
        lives  = np.array(self._ep_lives,  dtype=np.float32)
        wandb.log({
            "game/score_mean": float(scores.mean()),
            "game/score_max":  float(scores.max()),
            "game/score_std":  float(scores.std()),
            "game/lives_mean": float(lives.mean()),
        }, step=self.num_timesteps)
        self._ep_scores.clear()
        self._ep_lives.clear()


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
        # Reset patched flag so optimizer is re-patched for each curriculum level's learn() call
        self._patched = False
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
            # All features are normalised to [0, 1] (coordinates, ratios, flags, timers).
            low=0.0, high=1.0, shape=(STATE_SIZE,), dtype=np.float32
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
            # Process died unexpectedly — clear proc so next reset() restarts it
            self._proc = None
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
    # (2,  2_000_000),
    # (3,  2_000_000),
    # (4,  2_000_000),   # bridging level — avoids hard jump from 3→5
    # (5,  3_000_000),
    # (6,  2_000_000),   # bridging level — avoids hard jump from 5→8
    # (7,  2_000_000),   # bridging level
    # (8,  3_000_000),
    # (9,  2_000_000),   # bridging level — avoids hard jump from 8→10
    # (10, 4_000_000),
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
    wandb_enabled: bool = False,
    wandb_project: str = "pacman-rl",
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    wandb_group: str | None = None,
    wandb_tags: list[str] | None = None,
) -> tuple[PPO, VecNormalize | None]:
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

    # ---- Weights & Biases (optional) ---------------------------------------
    if wandb_enabled:
        if wandb is None or WandbCallback is None:
            raise RuntimeError(
                "W&B requested but not installed. Install with:  pip install wandb"
            )

        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            group=wandb_group,
            tags=wandb_tags,
            config={
                "algo": "PPO",
                "n_envs": n_envs,
                "state_size": STATE_SIZE,
                "n_actions": N_ACTIONS,
                "curriculum": CURRICULUM,
                **dataclasses.asdict(hp),
            },
            sync_tensorboard=True,
            save_code=True,
        )

    # ---- Build initial env & model ----------------------------------------
    # Wrap with VecNormalize so the model's observation_space reflects normalised obs.
    # This env is closed immediately after construction; curriculum loop creates fresh envs.
    _raw_init = make_vec_env(make_env_fn(CURRICULUM[0][0]), n_envs=n_envs)
    _init_venv = VecNormalize(_raw_init, norm_obs=True, norm_reward=True, clip_obs=3.0)

    if bc_init and os.path.isfile(bc_init):
        print(f"\nLoading BC pre-trained weights from {bc_init} ...")
        # Architecture must match the BC checkpoint; only override optimisation hparams.
        model = PPO.load(bc_init, _init_venv, **common)
        print("BC weights loaded — PPO will continue with Pacman-RL–aligned hparams.\n")
    else:
        model = PPO("MlpPolicy", _init_venv, policy_kwargs=pkw, **common)

    _init_venv.venv.close()  # close subprocesses; VecNormalize object is no longer needed

    dual_lr = DualLRCallback(
        policy_lr=hp.pi_lr,
        value_lr_multiplier=hp.value_lr_mult,
        verbose=1,
    )

    # ---- Curriculum loop --------------------------------------------------
    # venv persists across levels so that VecNormalize running stats (obs_rms, ret_rms)
    # accumulate continuously rather than resetting with each new env.
    venv: VecNormalize | None = None

    for level, timesteps in CURRICULUM:
        if wandb_enabled and wandb is not None and wandb.run is not None:
            wandb.config.update({"curriculum_level": level}, allow_val_change=True)

        print(f"\n{'='*55}")
        print(f"  Level {level:>2}  —  {timesteps:,} steps  ({n_envs} envs)")
        print(f"{'='*55}")

        raw_env = make_vec_env(make_env_fn(level), n_envs=n_envs)
        if venv is None:
            # First level: create VecNormalize fresh (stats start from zero).
            venv = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        else:
            # Subsequent levels: transfer accumulated obs/reward running stats to new env
            # so normalisation remains consistent across the full curriculum.
            new_venv = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
            new_venv.obs_rms = venv.obs_rms
            new_venv.ret_rms = venv.ret_rms
            venv.venv.close()  # shut down old subprocesses
            venv = new_venv

        model.set_env(venv)

        checkpoint_cb = CheckpointCallback(
            save_freq=max(100_000 // n_envs, 1),
            save_path=f"checkpoints/level{level}/",
            name_prefix="ppo_pacman",
            verbose=0,
        )

        callbacks: list[BaseCallback] = [checkpoint_cb, dual_lr]
        if wandb_enabled and WandbCallback is not None:
            callbacks.append(
                WandbCallback(
                    gradient_save_freq=0,
                    model_save_path=f"wandb_models/level{level}",
                    verbose=0,
                )
            )
            callbacks.append(WandbGameMetricsCallback(verbose=0))

        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            reset_num_timesteps=False,   # keep global step counter
            tb_log_name=f"level{level}",
        )

        save_path = f"models/ppo_level{level}_final"
        model.save(save_path)
        venv.save(f"{save_path}_vecnorm.pkl")
        print(f"  Saved → {save_path}.zip  +  {save_path}_vecnorm.pkl")

    # Close the final level's subprocesses
    if venv is not None:
        venv.venv.close()

    if wandb_enabled and wandb is not None:
        wandb.finish()

    return model, venv


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(
    model: PPO,
    path: str = "pacman_policy.onnx",
    vec_normalize: VecNormalize | None = None,
) -> None:
    """Export the policy network to ONNX with the names expected by AIAgent.h.

    When *vec_normalize* is supplied its running obs mean/std are baked directly
    into the ONNX graph so the C++ runtime does not need a separate normalisation
    step — the model accepts raw observations and normalises internally.
    If *vec_normalize* is None the wrapper uses identity normalisation (mean=0, std=1).
    """

    if vec_normalize is not None:
        obs_mean = torch.tensor(vec_normalize.obs_rms.mean, dtype=torch.float32)
        obs_std  = torch.tensor(
            np.sqrt(vec_normalize.obs_rms.var + vec_normalize.epsilon), dtype=torch.float32
        )
    else:
        obs_mean = torch.zeros(STATE_SIZE, dtype=torch.float32)
        obs_std  = torch.ones(STATE_SIZE,  dtype=torch.float32)

    class _PolicyWrapper(torch.nn.Module):
        """Bakes VecNormalize stats + actor MLP into a single ONNX graph.

        Baking the normalisation constants in avoids Categorical distribution
        tracing issues and keeps the C++ inference path simple.
        """
        def __init__(self, policy, obs_mean: torch.Tensor, obs_std: torch.Tensor):
            super().__init__()
            self.features_extractor = policy.features_extractor
            self.mlp = policy.mlp_extractor.policy_net
            self.action_net = policy.action_net
            self.register_buffer("obs_mean", obs_mean)
            self.register_buffer("obs_std",  obs_std)

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            state = ((state - self.obs_mean) / self.obs_std).clamp(-10.0, 10.0)
            features = self.features_extractor(state)
            return self.action_net(self.mlp(features))

    wrapper = _PolicyWrapper(model.policy, obs_mean, obs_std).cpu().eval()
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
    norm_note = "with baked VecNormalize" if vec_normalize is not None else "without normalisation"
    print(f"Exported ONNX model ({norm_note}) → {path}")
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

    # W&B
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging (requires `pip install wandb`)")
    parser.add_argument("--wandb-project", type=str, default="pacman-rl",
                        help="W&B project name (default: pacman-rl)")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="W&B entity/team (default: none)")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B run name (default: auto)")
    parser.add_argument("--wandb-group", type=str, default=None,
                        help="W&B group (default: none)")
    parser.add_argument("--wandb-tags", type=str, default="",
                        help="Comma-separated W&B tags (default: none)")
    args = parser.parse_args()

    if not os.path.isfile(PACMAN_BIN):
        print(f"ERROR: pacman binary not found at {PACMAN_BIN}")
        print("       Run:  cd build && cmake .. && make")
        raise SystemExit(1)

    if args.export_only:
        # Build a fresh model with the correct architecture, then load only policy weights.
        # PPO.load fails when the checkpoint has a different optimizer group count (DualLR).
        import zipfile, io, pickle
        env = make_vec_env(make_env_fn(1), n_envs=1)
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs_dict(), verbose=0)
        env.close()
        # Extract policy.pth from the zip and load only the policy state dict
        with zipfile.ZipFile(args.export_only, "r") as zf:
            with zf.open("policy.pth") as f:
                params = torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=True)
        model.policy.load_state_dict(params, strict=False)
        model.policy.eval()
        # Try to load the companion VecNormalize stats saved alongside the checkpoint.
        vn: VecNormalize | None = None
        vecnorm_path = args.export_only.replace(".zip", "_vecnorm.pkl")
        if os.path.isfile(vecnorm_path):
            with open(vecnorm_path, "rb") as f:
                vn = pickle.load(f)
            print(f"Loaded VecNormalize stats from {vecnorm_path}")
        else:
            print(f"No VecNormalize stats found at {vecnorm_path} — exporting without normalisation")
        export_onnx(model, vec_normalize=vn)
    else:
        hp = hparams_from_args(args)
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()] if args.wandb_tags else None
        model, venv = train(
            n_envs=args.envs,
            bc_init=args.bc_init,
            hp=hp,
            wandb_enabled=args.wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name,
            wandb_group=args.wandb_group,
            wandb_tags=tags,
        )
        export_onnx(model, vec_normalize=venv)


if __name__ == "__main__":
    main()
