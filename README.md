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
