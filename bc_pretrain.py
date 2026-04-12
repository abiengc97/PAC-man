"""
Behavioural Cloning pre-training from imitation log files.

Parses logs/imitation_*.jsonl, reconstructs the exact 948-float state vector
used during RL training, trains the CNN policy to mimic recorded actions, then
saves bc_pretrained.zip which train_ppo.py loads as the PPO starting point.

Usage:
    python3 bc_pretrain.py [--epochs 30] [--batch-size 512]
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from train_ppo import (
    PacManCNNExtractor, make_env_fn,
    STATE_SIZE, N_ACTIONS, MAZE_ROWS, MAZE_COLS, NON_SPATIAL,
)

# ------------------------------------------------------------------ constants
# TileType enum values (must match C++ exactly)
WALL        = 0
EMPTY       = 1
PELLET      = 2
POWER       = 3
GHOST_HOUSE = 4
GHOST_DOOR  = 5
TUNNEL      = 6

# Map maze-file characters → tile values
CHAR_MAP = {
    '#': WALL,  '.': PELLET, 'o': POWER,  ' ': EMPTY,
    'H': GHOST_HOUSE, 'D': GHOST_DOOR, 'T': TUNNEL,
    'P': EMPTY,         # Pac-Man spawn  → empty tile
    'B': EMPTY,         # Blinky spawn   → empty tile
    'I': GHOST_HOUSE,   # Inky spawn     → ghost house
    'N': GHOST_HOUSE,   # Pinky spawn    → ghost house
    'C': GHOST_HOUSE,   # Clyde spawn    → ghost house
    'E': EMPTY,         # ghost-house exit → empty
}

ACTION_MAP = {'up': 0, 'down': 1, 'left': 2, 'right': 3}

FRIGHTENED_DURATION = 360   # ~6 s at 60 fps


# ------------------------------------------------------------------ maze helpers

def load_maze(level_path: str) -> list[list[int]]:
    """Load a level file and return a (MAZE_ROWS × MAZE_COLS) tile grid."""
    grid = []
    with open(level_path) as f:
        for line in f:
            row = [CHAR_MAP.get(c, EMPTY) for c in line.rstrip('\n')]
            while len(row) < MAZE_COLS:
                row.append(WALL)
            grid.append(row[:MAZE_COLS])
    while len(grid) < MAZE_ROWS:
        grid.append([WALL] * MAZE_COLS)
    # Deep copy so callers can mutate freely
    return [r[:] for r in grid[:MAZE_ROWS]]


def count_total_pellets(grid: list[list[int]]) -> int:
    return sum(1 for row in grid for t in row if t in (PELLET, POWER))


# ------------------------------------------------------------------ state reconstruction

_DIR_VEC = {
    'up':    ( 0.0, -1.0),
    'down':  ( 0.0,  1.0),
    'left':  (-1.0,  0.0),
    'right': ( 1.0,  0.0),
    'none':  ( 0.0,  0.0),
}


def build_state(frame: dict, maze: list[list[int]],
                visit_count: dict, frightened_timers: list[int],
                total_pellets: int) -> list[float]:
    """Reconstruct the 948-float state vector from a single log frame."""
    pb      = frame['player_before']
    pr, pc  = pb['grid']['row'], pb['grid']['col']
    ghosts  = frame['ghosts_before']
    remaining  = frame['remaining_pellets_before']
    combo      = frame['ghost_eat_combo_before']
    is_chase   = frame['is_chase_mode_before']
    mode_timer = frame['ghost_mode_timer_before']
    power_pps  = frame['power_pellet_positions_before']

    state: list[float] = []

    # [0-48]  7×7 tile window centred on Pac-Man
    for dr in range(-3, 4):
        for dc in range(-3, 4):
            r, c = pr + dr, pc + dc
            if 0 <= r < MAZE_ROWS and 0 <= c < MAZE_COLS:
                state.append(float(maze[r][c]))
            else:
                state.append(float(WALL))

    # [49-56] per-ghost relative grid position (÷14)
    for g in ghosts:
        state.append((g['grid']['row'] - pr) / 14.0)
        state.append((g['grid']['col'] - pc) / 14.0)

    # [57-64] per-ghost direction unit vector (dx, dy)
    for g in ghosts:
        dx, dy = _DIR_VEC.get(g.get('direction', 'none'), (0.0, 0.0))
        state.append(dx)
        state.append(dy)

    # [65]    remaining pellets ratio
    state.append(remaining / max(1.0, total_pellets))

    # [66]    ghost eat combo ÷ 4
    state.append(combo / 4.0)

    # [67-70] per-ghost is_frightened flag
    for g in ghosts:
        state.append(1.0 if g['mode'] == 'frightened' else 0.0)

    # [71-74] per-ghost frightened timer (tracked from ate_power_pellet events)
    for t in frightened_timers:
        state.append(t / float(FRIGHTENED_DURATION))

    # [75]    global chase/scatter flag
    state.append(1.0 if is_chase else 0.0)

    # [76]    mode timer ÷ 1200
    state.append(mode_timer / 1200.0)

    # [77-78] direction to nearest power pellet (÷14); (0,0) if none left
    if power_pps:
        best_d = float('inf')
        pdx = pdy = 0.0
        for pp in power_pps:
            dr = pp['row'] - pr
            dc = pp['col'] - pc
            d  = abs(dr) + abs(dc)
            if d < best_d:
                best_d = d
                pdx = dc / 14.0
                pdy = dr / 14.0
        state.append(pdx)
        state.append(pdy)
    else:
        state.append(0.0)
        state.append(0.0)

    # [79]    visit novelty at current tile
    v = visit_count.get((pr, pc), 0)
    state.append(1.0 / (1 + v))
    visit_count[(pr, pc)] = v + 1

    # [80-81] Pac-Man absolute grid position, normalised (row/31, col/28)
    state.append(pr / float(MAZE_ROWS))
    state.append(pc / float(MAZE_COLS))

    # [82-85] per-ghost in_house flag (1 = in ghost house, 0 = on maze)
    for g in ghosts:
        state.append(1.0 if g.get('in_house', False) else 0.0)

    # [86-953] full 31×28 maze (row-major)
    for row in maze:
        for t in row:
            state.append(float(t))

    assert len(state) == STATE_SIZE, \
        f"State size mismatch: got {len(state)}, expected {STATE_SIZE}"
    return state


# ------------------------------------------------------------------ log parsing

def parse_logs(log_dir: str, level_path: str):
    """
    Walk all imitation_*.jsonl files, reconstruct state vectors, and collect
    (state, action) pairs.  Returns (np.float32 states, np.int64 actions).
    """
    all_states:  list[list[float]] = []
    all_actions: list[int]         = []

    log_files = sorted(glob.glob(os.path.join(log_dir, 'imitation_*.jsonl')))
    print(f"Found {len(log_files)} log file(s) in '{log_dir}'")

    for log_file in log_files:
        print(f"  Parsing {os.path.basename(log_file)} ...", end=' ', flush=True)

        # Per-session mutable state
        maze              = load_maze(level_path)
        total_pellets     = count_total_pellets(maze)
        visit_count: dict = {}
        frightened_timers = [0, 0, 0, 0]
        frames_added      = 0
        last_action: int | None = None   # carry-forward for 'none' frames

        with open(log_file) as f:
            for line in f:
                d = json.loads(line)

                if d['record_type'] != 'frame':
                    continue
                if d['state_before'] != 'playing':
                    continue

                action_str = d['input']['target_action']
                if action_str in ACTION_MAP:
                    action = ACTION_MAP[action_str]
                    last_action = action
                elif last_action is not None:
                    # 'none' frame — reuse last valid action rather than dropping it.
                    # Dropping biases the dataset toward always having a direction change.
                    action = last_action
                else:
                    continue   # very first frame has no prior action; skip

                # ---- build state from log data
                state  = build_state(d, maze, visit_count, frightened_timers, total_pellets)
                all_states.append(state)
                all_actions.append(action)
                frames_added += 1

                # ---- update maze: remove eaten tiles
                events = d['events']
                if events['ate_pellet'] or events['ate_power_pellet']:
                    er, ec = d['player_before']['grid']['row'], \
                             d['player_before']['grid']['col']
                    if 0 <= er < MAZE_ROWS and 0 <= ec < MAZE_COLS:
                        maze[er][ec] = EMPTY

                # ---- update frightened timers
                if events['ate_power_pellet']:
                    frightened_timers = [FRIGHTENED_DURATION] * 4
                else:
                    frightened_timers = [max(0, t - 1) for t in frightened_timers]

                # ---- reset per-life / per-level state
                if events['player_died']:
                    visit_count       = {}
                    frightened_timers = [0, 0, 0, 0]
                if events['level_cleared']:
                    maze              = load_maze(level_path)
                    visit_count       = {}
                    frightened_timers = [0, 0, 0, 0]

        print(f"{frames_added} frames")

    states_np  = np.array(all_states,  dtype=np.float32)
    actions_np = np.array(all_actions, dtype=np.int64)
    print(f"\nTotal dataset: {len(states_np)} frames")
    counts = np.bincount(actions_np, minlength=N_ACTIONS)
    for i, name in enumerate(['UP', 'DOWN', 'LEFT', 'RIGHT']):
        print(f"  {name:>5}: {counts[i]:6d}  ({counts[i]/len(actions_np)*100:.1f}%)")
    return states_np, actions_np


# ------------------------------------------------------------------ BC training

class _ActorWrapper(nn.Module):
    """Thin wrapper: features_extractor → policy MLP → action logits."""
    def __init__(self, policy):
        super().__init__()
        self.features_extractor = policy.features_extractor
        self.mlp                = policy.mlp_extractor.policy_net
        self.action_net         = policy.action_net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.action_net(self.mlp(self.features_extractor(x)))


def pretrain(log_dir: str   = 'logs',
             level_path: str = 'levels/level1.txt',
             out_path: str   = 'bc_pretrained',
             epochs: int     = 30,
             batch_size: int = 512,
             lr: float       = 1e-3) -> str:

    print("\n========== Behavioural Cloning Pre-training ==========")

    # 1. Dataset
    states, actions = parse_logs(log_dir, level_path)

    # 2. Build PPO model (just to get the network structure + weights)
    print("\nBuilding PPO policy structure ...")
    env   = make_vec_env(make_env_fn(1), n_envs=1)
    model = PPO(
        "MlpPolicy", env,
        policy_kwargs=dict(
            features_extractor_class=PacManCNNExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=[256, 128],
        ),
        verbose=0,
    )
    env.close()

    device = next(model.policy.parameters()).device
    actor  = _ActorWrapper(model.policy).to(device)
    opt    = torch.optim.Adam(actor.parameters(), lr=lr)
    crit   = nn.CrossEntropyLoss()

    # Train/val split (90/10) — saves best-val checkpoint, avoids overfitting
    n_total = len(states)
    idx = np.random.permutation(n_total)
    if n_total <= 1:
        train_idx, val_idx = idx, np.array([], dtype=np.int64)
    else:
        n_val = min(max(1, int(n_total * 0.1)), n_total - 1)
        val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_ds = TensorDataset(torch.from_numpy(states[train_idx]),
                             torch.from_numpy(actions[train_idx]))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=True)
    if len(val_idx) > 0:
        val_ds = TensorDataset(torch.from_numpy(states[val_idx]),
                               torch.from_numpy(actions[val_idx]))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=True)
    else:
        val_loader = None

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    # 3. Supervised training loop
    print(f"\nTraining for {epochs} epochs on {device} ({len(train_idx)} train / {len(val_idx)} val) ...")
    best_val_acc    = -1.0
    best_state_dict = None
    for epoch in range(1, epochs + 1):
        actor.train()
        total_loss = correct = total = 0

        for s_batch, a_batch in loader:
            s_batch = s_batch.to(device)
            a_batch = a_batch.to(device)

            logits = actor(s_batch)
            loss   = crit(logits, a_batch)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(a_batch)
            correct    += (logits.argmax(1) == a_batch).sum().item()
            total      += len(a_batch)

        scheduler.step()

        # Validation pass
        actor.eval()
        if val_loader is not None:
            val_correct = val_total = 0
            with torch.no_grad():
                for s_batch, a_batch in val_loader:
                    s_batch = s_batch.to(device)
                    a_batch = a_batch.to(device)
                    val_correct += (actor(s_batch).argmax(1) == a_batch).sum().item()
                    val_total   += len(a_batch)
            val_acc = val_correct / val_total
        else:
            val_acc = train_acc = correct / total

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc   = val_acc
            best_state_dict = {k: v.cpu().clone() for k, v in actor.state_dict().items()}

        train_acc = correct / total
        print(f"  epoch {epoch:3d}/{epochs}  "
              f"loss={total_loss/total:.4f}  train={train_acc:.3f}  val={val_acc:.3f}"
              + ("  ★" if is_best else ""))

    print(f"\nBest validation accuracy: {best_val_acc:.3f}  (restoring best weights)")
    actor.load_state_dict(best_state_dict)

    # 4. Save — the PPO model now carries BC-initialised actor weights.
    #    Value network weights remain random (BC only trains the actor).
    model.save(out_path)
    zip_path = out_path + '.zip'
    print(f"Saved → {zip_path}  (load with train_ppo.py --bc-init {zip_path})")
    return zip_path


# ------------------------------------------------------------------ entry point

def main():
    parser = argparse.ArgumentParser(
        description="BC pre-training for PAC-man PPO agent")
    parser.add_argument('--log-dir',    default='logs')
    parser.add_argument('--level',      default='levels/level1.txt')
    parser.add_argument('--out',        default='bc_pretrained')
    parser.add_argument('--epochs',     type=int,   default=30)
    parser.add_argument('--batch-size', type=int,   default=512)
    parser.add_argument('--lr',         type=float, default=1e-3)
    args = parser.parse_args()
    pretrain(args.log_dir, args.level, args.out,
             args.epochs, args.batch_size, args.lr)


if __name__ == '__main__':
    main()
