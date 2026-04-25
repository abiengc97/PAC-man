# Pac-Man Retro Port — C++ / SDL2

## Project Structure

```
pacman/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── Common.h        # Shared types, constants, enums
│   ├── Maze.h          # Maze loading and tile queries
│   ├── Player.h        # Pac-Man movement and input
│   ├── Ghost.h         # Ghost AI with 4 distinct behaviors
│   ├── AStar.h         # A* pathfinding
│   ├── Renderer.h      # SDL2 rendering wrapper
│   └── Game.h          # Game manager / main loop
├── src/
│   ├── main.cpp
│   ├── Game.cpp
│   ├── Maze.cpp
│   ├── Player.cpp
│   ├── Ghost.cpp
│   ├── AStar.cpp
│   └── Renderer.cpp
└── levels/
    └── level1.txt      # Classic Pac-Man maze layout
```

## Dependencies

- **C++17** compiler (GCC 7+, Clang 5+, MSVC 2017+)
- **SDL2** development libraries
- **CMake** 3.16+

### Install SDL2

**Ubuntu/Debian:**
```bash
sudo apt install libsdl2-dev
```

**macOS (Homebrew):**
```bash
brew install sdl2
```

**Windows (vcpkg):**
```bash
vcpkg install sdl2
```

## Build & Run

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
./pacman
```

## Controls

- **Arrow keys** or **WASD** — Move Pac-Man
- **P** — Pause / Resume
- **Enter** — Restart after Game Over
- **Escape** — Quit

## Gameplay Logging

The game now writes imitation-learning data to `logs/imitation_YYYYMMDD_HHMMSS.jsonl`
while you play.

Each line is a JSON object. The log includes:

- Per-frame keypress metadata plus a `target_action` label derived from buffered direction first, then current movement
- Pac-Man grid/pixel position, current direction, and buffered direction
- All ghost positions, directions, modes, and distance-to-player
- Power-pellet positions before/after the frame
- Score, lives, level, ghost mode schedule state, and pellet counts
- Outcome flags such as pellet eaten, power pellet eaten, bonus fruit eaten, ghost eaten, player death, and level clear

Bonus fruit now appears twice per level using the classic Pac-Man dot-count
thresholds and the original arcade fruit progression. `bonus_fruit_position`
is `null` when no fruit is active.

Ready-countdown frames are skipped. `ghost_mode_phase` advances only after a
full scatter/chase pair; `ghost_mode_step` advances on each scatter/chase
segment and is usually the more useful field to model.

## PPO Learning (Reinforcement Learning)

This project also supports training a Pac-Man agent with **PPO (Proximal Policy Optimization)** using **Stable-Baselines3** (`train_ppo.py`).

- **Environment**: `train_ppo.py` wraps the C++ game as a Gymnasium env by launching `build/pacman` in **headless RL mode** and exchanging one JSON message per step over stdin/stdout.
- **Observation**: a fixed-length vector of **89 floats** (normalized to \([0,1]\)) produced by the game (`state=[...89 floats...]`).
- **Actions**: 4 discrete actions: `0=UP, 1=DOWN, 2=LEFT, 3=RIGHT`.
- **Step loop**:
  - Python sends `{"action": N}` (or `{"reset": true}` to restart an episode)
  - C++ returns `{"state":[...], "reward":R, "done":B, "lives":L, "score":S}`
- **Training**: PPO collects rollouts from multiple parallel env processes (`--envs`, default 8), computes advantages with GAE, and performs clipped policy/value updates over minibatches (plus optional W&B/TensorBoard logging).
- **Export**: after training, the policy is exported to **ONNX** so it can be loaded for inference (`pacman_policy.onnx`).

### Train / Watch

```bash
# Train PPO (headless envs)
python3 train_ppo.py --envs 8

# Watch the trained agent play (renders the normal game window)
python3 ai_play.py
```

## Demos

> Place the GIFs at the paths below (or update the links if your filenames differ).

**Human (user) gameplay**

![Human gameplay](media/user.gif)

**AI (PPO-trained) gameplay**

![PPO-trained agent gameplay](media/ai_trained.gif)

## Level File Format

The maze is defined in `levels/level1.txt` using these characters:

| Char | Meaning          |
|------|------------------|
| `#`  | Wall             |
| `.`  | Pellet           |
| `o`  | Power Pellet     |
| ` `  | Empty space      |
| `H`  | Ghost House      |
| `D`  | Ghost Door       |
| `T`  | Tunnel           |
| `P`  | Pac-Man spawn    |
| `B`  | Blinky spawn     |
| `N`  | Pinky spawn      |
| `I`  | Inky spawn       |
| `C`  | Clyde spawn      |
| `E`  | Ghost house entry|

## Ghost AI

Each ghost has a unique targeting strategy in Chase mode:

- **Blinky (Red):** Directly targets Pac-Man's current tile
- **Pinky (Pink):** Targets 4 tiles ahead of Pac-Man
- **Inky (Cyan):** Vector calculation using Blinky's position
- **Clyde (Orange):** Targets Pac-Man when far (>8 tiles), retreats to corner when close

Ghosts cycle between Scatter and Chase modes on a timer,
and enter Frightened mode when Pac-Man eats a Power Pellet.

## Component Ownership

| Component              | Owner    |
|------------------------|----------|
| Maze / Map System      | Member 1 |
| Player Controller & Renderer | Member 2 |
| Ghost AI + A*          | Member 3 |
| Game Manager           | Shared   |
