from __future__ import annotations

import json
import os
import pathlib
import subprocess
import time
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


_HERE = pathlib.Path(__file__).resolve().parent


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _default_pacman_binary() -> pathlib.Path:
    env_override = os.environ.get("PACMAN_BIN")
    if env_override:
        return pathlib.Path(env_override).expanduser().resolve()

    candidates = [
        _HERE / "PAC-man" / "build" / "pacman",
        _HERE / "build" / "pacman",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def resolve_pacman_binary(binary_path: Optional[str] = None) -> pathlib.Path:
    active_path = binary_path or os.environ.get("PACMAN_BIN") or PACMAN_BIN
    path = pathlib.Path(active_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            "Pac-Man binary not found at "
            f"{path}. Build it with: cd {_HERE / 'PAC-man'} && mkdir -p build && cd build && cmake .. && make -j$(nproc)"
        )
    if not os.access(path, os.X_OK):
        raise PermissionError(f"Pac-Man binary is not executable: {path}")
    return path


PACMAN_BIN = str(_default_pacman_binary())

STATE_SIZE = 954
N_ACTIONS = 4
MAZE_ROWS = 31
MAZE_COLS = 28
MAZE_CELLS = MAZE_ROWS * MAZE_COLS
NON_SPATIAL = STATE_SIZE - MAZE_CELLS
N_GHOSTS = 4
LOCAL_WIN = 7 * 7
GHOST_VEC_START = LOCAL_WIN
GHOST_FEAT_DIM = 4
ACTION_NAMES = ("Up", "Down", "Left", "Right")
MUZERO_FLAT_OBSERVATION_SHAPE = (1, 1, STATE_SIZE)
MUZERO_SPATIAL_CHANNELS = 29
MUZERO_SPATIAL_OBSERVATION_SHAPE = (MUZERO_SPATIAL_CHANNELS, MAZE_ROWS, MAZE_COLS)
MUZERO_OBSERVATION_SHAPE = MUZERO_SPATIAL_OBSERVATION_SHAPE
DEFAULT_START_LEVEL = int(os.environ.get("PACMAN_START_LEVEL", "1"))
DEFAULT_MAX_STEPS = int(os.environ.get("PACMAN_MAX_STEPS", "8000"))
VALID_LEVEL_SCHEDULES = {"fixed", "random_episode", "round_robin"}
VALID_OBSERVATION_ENCODINGS = {"flat", "spatial"}


def default_start_level() -> int:
    return int(os.environ.get("PACMAN_START_LEVEL", str(DEFAULT_START_LEVEL)))


def default_max_steps() -> int:
    return int(os.environ.get("PACMAN_MAX_STEPS", str(DEFAULT_MAX_STEPS)))


def normalize_observation_encoding(value: str) -> str:
    encoding = value.strip().lower()
    if encoding not in VALID_OBSERVATION_ENCODINGS:
        raise ValueError(
            f"Unsupported Pac-Man observation encoding '{value}'. "
            f"Expected one of {sorted(VALID_OBSERVATION_ENCODINGS)}."
        )
    return encoding


def default_observation_encoding() -> str:
    return normalize_observation_encoding(os.environ.get("PACMAN_OBSERVATION_ENCODING", "spatial"))


def muzero_observation_shape(observation_encoding: Optional[str] = None) -> tuple[int, int, int]:
    encoding = default_observation_encoding() if observation_encoding is None else normalize_observation_encoding(observation_encoding)
    return MUZERO_FLAT_OBSERVATION_SHAPE if encoding == "flat" else MUZERO_SPATIAL_OBSERVATION_SHAPE


def parse_level_list(value: Optional[str]) -> list[int]:
    if value is None:
        return []

    levels = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        level = int(part)
        if level < 1:
            raise ValueError(f"Pac-Man levels must be positive integers. Got {level}.")
        levels.append(level)
    return levels


def default_levels(start_level: Optional[int] = None) -> list[int]:
    configured = parse_level_list(os.environ.get("PACMAN_LEVELS"))
    if configured:
        return configured
    return [int(default_start_level() if start_level is None else start_level)]


def normalize_level_schedule(value: str) -> str:
    schedule = value.strip().lower().replace("-", "_")
    if schedule not in VALID_LEVEL_SCHEDULES:
        raise ValueError(
            f"Unsupported Pac-Man level schedule '{value}'. "
            f"Expected one of {sorted(VALID_LEVEL_SCHEDULES)}."
        )
    return schedule


def default_level_schedule(level_count: Optional[int] = None) -> str:
    configured = os.environ.get("PACMAN_LEVEL_SCHEDULE")
    if configured:
        return normalize_level_schedule(configured)
    return "random_episode" if (level_count or 0) > 1 else "fixed"


def default_level_switch_steps() -> Optional[int]:
    configured = os.environ.get("PACMAN_LEVEL_SWITCH_STEPS")
    if configured is None or configured == "":
        return None
    value = int(configured)
    if value <= 0:
        raise ValueError("PACMAN_LEVEL_SWITCH_STEPS must be a positive integer.")
    return value


def _grid_position_from_normalized(value: float, size: int) -> int:
    return int(np.clip(np.rint(value * size), 0, size - 1))


def observation_to_muzero_flat(observation: np.ndarray) -> np.ndarray:
    return np.asarray(observation, dtype=np.float32).reshape(MUZERO_FLAT_OBSERVATION_SHAPE)


def observation_to_muzero_spatial(observation: np.ndarray) -> np.ndarray:
    obs = np.asarray(observation, dtype=np.float32)
    if obs.shape != (STATE_SIZE,):
        raise ValueError(
            f"Expected Pac-Man observation shape {(STATE_SIZE,)} but received {obs.shape}."
        )

    ghost_pos = obs[49:57].reshape(N_GHOSTS, 2)
    ghost_dir = obs[57:65].reshape(N_GHOSTS, 2)
    pellet_frac = float(obs[65])
    combo = float(obs[66])
    fright_flag = obs[67:71]
    fright_tmr = obs[71:75]
    chase_mode = float(obs[75])
    mode_tmr = float(obs[76])
    pp_dir = obs[77:79]
    novelty = float(obs[79])
    pac_pos = obs[80:82]
    in_house = obs[82:86]
    maze_flat = obs[NON_SPATIAL:]

    tiles = maze_flat.reshape(MAZE_ROWS, MAZE_COLS)
    spatial = np.zeros(MUZERO_SPATIAL_OBSERVATION_SHAPE, dtype=np.float32)

    spatial[0] = (tiles == 0).astype(np.float32)
    spatial[1] = (tiles == 2).astype(np.float32)
    spatial[2] = (tiles == 3).astype(np.float32)
    spatial[3] = ((tiles == 4) | (tiles == 5)).astype(np.float32)
    spatial[4] = (tiles == 6).astype(np.float32)

    pac_row = _grid_position_from_normalized(float(pac_pos[0]), MAZE_ROWS)
    pac_col = _grid_position_from_normalized(float(pac_pos[1]), MAZE_COLS)
    spatial[5, pac_row, pac_col] = 1.0

    for ghost_index in range(N_GHOSTS):
        g_row = int(
            np.clip(
                np.rint(pac_row + float(ghost_pos[ghost_index, 0]) * 14.0),
                0,
                MAZE_ROWS - 1,
            )
        )
        g_col = int(
            np.clip(
                np.rint(pac_col + float(ghost_pos[ghost_index, 1]) * 14.0),
                0,
                MAZE_COLS - 1,
            )
        )
        active = 1.0 if float(in_house[ghost_index]) < 0.5 else 0.0
        mode_value = -1.0 if float(fright_flag[ghost_index]) >= 0.5 else 1.0
        spatial[6 + ghost_index, g_row, g_col] = active * mode_value
        spatial[10 + ghost_index, g_row, g_col] = float(fright_tmr[ghost_index]) * active
        spatial[14 + ghost_index, g_row, g_col] = float(ghost_dir[ghost_index, 0]) * active
        spatial[18 + ghost_index, g_row, g_col] = float(ghost_dir[ghost_index, 1]) * active

    spatial[22].fill(pellet_frac)
    spatial[23].fill(combo)
    spatial[24].fill(chase_mode)
    spatial[25].fill(mode_tmr)
    spatial[26].fill(float(pp_dir[0]))
    spatial[27].fill(float(pp_dir[1]))
    spatial[28, pac_row, pac_col] = novelty
    return spatial


def observation_to_muzero(
    observation: np.ndarray,
    observation_encoding: Optional[str] = None,
) -> np.ndarray:
    encoding = (
        default_observation_encoding()
        if observation_encoding is None
        else normalize_observation_encoding(observation_encoding)
    )
    if encoding == "flat":
        return observation_to_muzero_flat(observation)
    return observation_to_muzero_spatial(observation)


class PacManEnv(gym.Env):
    """Wrap the C++ Pac-Man RL process via stdin/stdout JSON messages."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        level: Optional[int] = None,
        levels: Optional[list[int]] = None,
        level_schedule: Optional[str] = None,
        level_switch_steps: Optional[int] = None,
        render: Optional[bool] = None,
        pacman_bin: Optional[str] = None,
        max_steps: Optional[int] = None,
    ):
        super().__init__()
        initial_level = int(default_start_level() if level is None else level)
        self.levels = [int(item) for item in (levels if levels is not None else default_levels(initial_level))]
        if not self.levels:
            raise ValueError("Pac-Man level list cannot be empty.")
        self.level_schedule = (
            default_level_schedule(len(self.levels))
            if level_schedule is None
            else normalize_level_schedule(level_schedule)
        )
        self.level_switch_steps = (
            default_level_switch_steps() if level_switch_steps is None else int(level_switch_steps)
        )
        if self.level_switch_steps is not None and self.level_switch_steps <= 0:
            raise ValueError("level_switch_steps must be a positive integer.")
        self._rng = np.random.default_rng()
        self._level_index = 0
        if initial_level in self.levels:
            self._level_index = self.levels.index(initial_level)
        self.level = self.levels[self._level_index]
        self.render_enabled = _env_flag("PACMAN_RENDER", False) if render is None else bool(render)
        self.max_steps = int(default_max_steps() if max_steps is None else max_steps)
        self.pacman_bin = str(resolve_pacman_binary(pacman_bin))
        self._proc: Optional[subprocess.Popen[str]] = None
        self._last_obs = np.zeros(STATE_SIZE, dtype=np.float32)
        self._step_count = 0
        self._steps_since_level_change = 0

        self.observation_space = gym.spaces.Box(
            low=-3.0, high=10.0, shape=(STATE_SIZE,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(N_ACTIONS)

    def _command(self) -> list[str]:
        command = [self.pacman_bin, "--rl", f"--level={self.level}"]
        command.append("--render" if self.render_enabled else "--headless")
        return command

    def _terminate_process(self) -> None:
        if self._proc is None:
            return
        try:
            for pipe in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
                if pipe is not None:
                    try:
                        pipe.close()
                    except Exception:
                        pass
            self._proc.terminate()
            self._proc.wait(timeout=2)
        except Exception:
            try:
                self._proc.kill()
                self._proc.wait(timeout=2)
            except Exception:
                pass
        finally:
            self._proc = None

    def _read_stderr(self) -> str:
        if not self._proc or not self._proc.stderr:
            return ""
        try:
            return self._proc.stderr.read()
        except Exception:
            return ""

    def _start_process(self) -> None:
        self._terminate_process()
        binary_path = pathlib.Path(self.pacman_bin).resolve()
        self._proc = subprocess.Popen(
            self._command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(binary_path.parent),
        )
        time.sleep(0.05)
        if self._proc.poll() is not None:
            stderr_out = self._proc.stderr.read() if self._proc.stderr else ""
            raise RuntimeError(
                f"Pac-Man exited immediately (code {self._proc.returncode}).\n"
                f"Command: {' '.join(self._command())}\n"
                f"stderr: {stderr_out}"
            )
        data = self._read_step()
        self._last_obs = self._validate_observation(data["state"])

    def _write(self, payload: Dict[str, Any]) -> None:
        if not self._proc or not self._proc.stdin:
            raise RuntimeError("Pac-Man process is not running.")
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"Pac-Man process exited with code {self._proc.returncode} before write.\n"
                f"stderr: {self._read_stderr()}"
            )
        try:
            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError(
                f"Pac-Man process exited with code {self._proc.poll()} while sending {payload}.\n"
                f"stderr: {self._read_stderr()}"
            ) from exc

    def _read_step(self) -> Dict[str, Any]:
        if not self._proc or not self._proc.stdout:
            raise RuntimeError("Pac-Man process is not running.")
        line = self._proc.stdout.readline()
        if not line:
            stderr_out = self._read_stderr()
            return {
                "state": self._last_obs.tolist(),
                "reward": -50.0,
                "done": True,
                "lives": 0,
                "score": 0,
                "stderr": stderr_out,
            }
        return json.loads(line)

    def _validate_observation(self, raw_state: Any) -> np.ndarray:
        observation = np.asarray(raw_state, dtype=np.float32)
        if observation.shape != (STATE_SIZE,):
            raise ValueError(
                f"Expected Pac-Man observation shape {(STATE_SIZE,)} but received {observation.shape}."
            )
        return observation

    def _random_level(self, exclude_current: bool = False) -> int:
        choices = self.levels
        if exclude_current and len(self.levels) > 1:
            choices = [level for level in self.levels if level != self.level]
        return int(choices[int(self._rng.integers(len(choices)))])

    def _advance_round_robin(self) -> int:
        self._level_index = (self._level_index + 1) % len(self.levels)
        return self.levels[self._level_index]

    def _select_level_for_reset(self) -> int:
        if len(self.levels) == 1 or self.level_schedule == "fixed":
            return self.level

        if self._proc is None:
            if self.level_schedule == "random_episode":
                return self._random_level()
            return self.level

        if self.level_switch_steps is not None and self._steps_since_level_change < self.level_switch_steps:
            return self.level

        if self.level_schedule == "random_episode":
            next_level = self._random_level(exclude_current=True)
        else:
            next_level = self._advance_round_robin()

        self._steps_since_level_change = 0
        return next_level

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        target_level = self._select_level_for_reset()
        level_changed = target_level != self.level
        self.level = target_level

        if self._proc is None or level_changed:
            self._start_process()
        else:
            self._write({"reset": True})
            data = self._read_step()
            self._last_obs = self._validate_observation(data["state"])
        return self._last_obs.copy(), {"level": self.level}

    def step(self, action: int):
        self._write({"action": int(action)})
        data = self._read_step()

        observation = self._validate_observation(data["state"])
        reward = float(data["reward"])
        terminated = bool(data["done"])
        self._step_count += 1
        self._steps_since_level_change += 1
        truncated = self._step_count >= self.max_steps
        info = {
            "level": self.level,
            "lives": int(data["lives"]),
            "score": int(data["score"]),
        }
        if "stderr" in data and data["stderr"]:
            info["stderr"] = data["stderr"]

        self._last_obs = observation
        return observation.copy(), reward, terminated, truncated, info

    def step_muzero(
        self,
        action: int,
        observation_encoding: Optional[str] = None,
    ) -> Tuple[np.ndarray, float, bool]:
        observation, reward, terminated, truncated, _ = self.step(action)
        return observation_to_muzero(
            observation,
            observation_encoding=observation_encoding,
        ), reward, terminated or truncated

    def close(self) -> None:
        self._terminate_process()

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass


CURRICULUM = [
    (1, 5_000_000),
    (2, 2_000_000),
    (3, 2_000_000),
    (4, 2_000_000),
    (5, 3_000_000),
    (6, 2_000_000),
    (7, 2_000_000),
    (8, 3_000_000),
    (9, 2_000_000),
    (10, 5_000_000),
]


def make_env_fn(level: int):
    def _init():
        env = PacManEnv(level=level)
        try:
            from stable_baselines3.common.monitor import Monitor
        except Exception as exc:  # pragma: no cover - PPO-only path
            raise ImportError(
                "stable-baselines3 is required for make_env_fn()/Monitor wrapping."
            ) from exc
        return Monitor(env)

    return _init
