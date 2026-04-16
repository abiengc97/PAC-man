#pragma once

#include <cstdint>
#include <utility>

// ============================================================
// Grid & Window Constants
// ============================================================
constexpr int TILE_SIZE    = 24;
constexpr int MAZE_COLS    = 28;
constexpr int MAZE_ROWS    = 31;
constexpr int SCREEN_W     = MAZE_COLS * TILE_SIZE;   // 672
constexpr int SCREEN_H     = (MAZE_ROWS + 3) * TILE_SIZE; // extra rows for score/lives HUD
constexpr int FPS          = 60;
constexpr int FRAME_DELAY  = 1000 / FPS;

// ============================================================
// Tile Types
// ============================================================
enum class TileType : uint8_t {
    WALL        = 0,
    EMPTY       = 1,
    PELLET      = 2,
    POWER       = 3,
    GHOST_HOUSE = 4,
    GHOST_DOOR  = 5,
    TUNNEL      = 6,
};

// ============================================================
// Directions
// ============================================================
enum class Direction {
    NONE  = -1,
    UP    = 0,
    DOWN  = 1,
    LEFT  = 2,
    RIGHT = 3,
};

// Direction deltas: {row_delta, col_delta}
constexpr std::pair<int,int> DIR_DELTA[] = {
    { -1,  0 },   // UP
    {  1,  0 },   // DOWN
    {  0, -1 },   // LEFT
    {  0,  1 },   // RIGHT
};

inline std::pair<int,int> dirDelta(Direction d) {
    if (d == Direction::NONE) return {0, 0};
    return DIR_DELTA[static_cast<int>(d)];
}

inline Direction oppositeDir(Direction d) {
    switch (d) {
        case Direction::UP:    return Direction::DOWN;
        case Direction::DOWN:  return Direction::UP;
        case Direction::LEFT:  return Direction::RIGHT;
        case Direction::RIGHT: return Direction::LEFT;
        default: return Direction::NONE;
    }
}

// ============================================================
// Ghost Identity
// ============================================================
enum class GhostID { BLINKY = 0, PINKY = 1, INKY = 2, CLYDE = 3 };

// ============================================================
// Ghost Mode
// ============================================================
enum class GhostMode {
    CHASE,
    SCATTER,
    FRIGHTENED,
    EATEN,
};

// ============================================================
// Game State
// ============================================================
enum class GameState {
    MENU,
    PLAYING,
    PAUSED,
    LEVEL_CLEAR,
    DYING,
    GAME_OVER,
};

// ============================================================
// Bonus Fruit
// ============================================================
enum class FruitType : uint8_t {
    CHERRY,
    STRAWBERRY,
    ORANGE,
    APPLE,
    MELON,
    GALAXIAN,
    BELL,
    KEY,
};

// ============================================================
// Simple 2D position
// ============================================================
struct GridPos {
    int row = 0;
    int col = 0;

    bool operator==(const GridPos& o) const { return row == o.row && col == o.col; }
    bool operator!=(const GridPos& o) const { return !(*this == o); }
};
