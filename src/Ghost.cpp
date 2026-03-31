#include "Ghost.h"
#include "Maze.h"
#include "Player.h"
#include "AStar.h"
#include <cstdlib>
#include <cmath>

Ghost::Ghost(GhostID id) : m_id(id) {
    // Stagger ghost house exit times (in ticks at 60fps)
    switch (id) {
        case GhostID::BLINKY: m_houseExitDelay = 0;   break; // immediately
        case GhostID::PINKY:  m_houseExitDelay = 120;  break; // 2 sec
        case GhostID::INKY:   m_houseExitDelay = 300;  break; // 5 sec
        case GhostID::CLYDE:  m_houseExitDelay = 480;  break; // 8 sec
    }
}

void Ghost::init(const Maze& maze) {
    m_spawnPos  = maze.getGhostSpawn(m_id);
    m_pos       = m_spawnPos;
    m_direction = Direction::LEFT;
    m_mode      = GhostMode::SCATTER;
    m_inHouse   = true;
    m_houseTimer = 0;
    m_frightenedTimer = 0;
    m_pixelX = m_pos.col * TILE_SIZE + TILE_SIZE / 2;
    m_pixelY = m_pos.row * TILE_SIZE + TILE_SIZE / 2;

    // Blinky starts outside the house
    if (m_id == GhostID::BLINKY) {
        m_pos = maze.getGhostHouseEntry();
        m_inHouse = false;
        m_pixelX = m_pos.col * TILE_SIZE + TILE_SIZE / 2;
        m_pixelY = m_pos.row * TILE_SIZE + TILE_SIZE / 2;
    }
}

void Ghost::reset(const Maze& maze) {
    init(maze);
}

void Ghost::update(const Maze& maze, const Player& pacman, const Ghost* blinky) {
    // Handle ghost house exit
    if (m_inHouse) {
        m_houseTimer++;
        if (m_houseTimer >= m_houseExitDelay) {
            // Move to house entry position
            m_pos = maze.getGhostHouseEntry();
            m_pixelX = m_pos.col * TILE_SIZE + TILE_SIZE / 2;
            m_pixelY = m_pos.row * TILE_SIZE + TILE_SIZE / 2;
            m_inHouse = false;
            m_direction = Direction::LEFT;
        }
        return; // don't move while in house
    }

    // Frightened timer countdown
    if (m_mode == GhostMode::FRIGHTENED) {
        m_frightenedTimer--;
        if (m_frightenedTimer <= 0) {
            m_mode = GhostMode::SCATTER; // revert to scatter after fright ends
        }
    }

    // Only make decisions at grid-aligned positions
    if (!isAlignedToGrid()) {
        // Continue moving in current direction
        auto [dr, dc] = dirDelta(m_direction);
        m_pixelX += dc * m_speed;
        m_pixelY += dr * m_speed;

        // Tunnel wrap even while between tiles (prevents ghosts going off-screen)
        if (m_pixelX < 0) {
            m_pixelX = maze.getCols() * TILE_SIZE - m_speed;
        } else if (m_pixelX >= maze.getCols() * TILE_SIZE) {
            m_pixelX = m_speed;
        }

        // Update grid position
        m_pos.row = m_pixelY / TILE_SIZE;
        m_pos.col = m_pixelX / TILE_SIZE;
        return;
    }

    // Choose direction based on mode
    Direction chosenDir = Direction::NONE;

    if (m_mode == GhostMode::FRIGHTENED) {
        chosenDir = chooseFrightenedDirection(maze);
    } else if (m_mode == GhostMode::EATEN) {
        // Head back to ghost house
        GridPos houseEntry = maze.getGhostHouseEntry();
        if (m_pos == houseEntry) {
            // Reached the house — respawn
            m_mode = GhostMode::SCATTER;
            m_pos = m_spawnPos;
            m_pixelX = m_pos.col * TILE_SIZE + TILE_SIZE / 2;
            m_pixelY = m_pos.row * TILE_SIZE + TILE_SIZE / 2;
            m_inHouse = true;
            m_houseTimer = m_houseExitDelay; // exit immediately after eaten return
            return;
        }
        chosenDir = AStar::getNextDirection(maze, m_pos, houseEntry);
    } else {
        // Chase or Scatter — use target tile
        GridPos target = computeTargetTile(maze, pacman, blinky);
        chosenDir = chooseBestDirection(maze, target);
    }

    if (chosenDir == Direction::NONE) {
        chosenDir = m_direction; // keep going if no better option
    }

    m_direction = chosenDir;

    // Move
    auto [dr, dc] = dirDelta(m_direction);
    m_pixelX += dc * m_speed;
    m_pixelY += dr * m_speed;

    // Tunnel wrap
    if (m_pixelX < 0) {
        m_pixelX = maze.getCols() * TILE_SIZE - m_speed;
    } else if (m_pixelX >= maze.getCols() * TILE_SIZE) {
        m_pixelX = m_speed;
    }

    // Update grid position
    m_pos.row = m_pixelY / TILE_SIZE;
    m_pos.col = m_pixelX / TILE_SIZE;
}

void Ghost::setMode(GhostMode mode) {
    if (m_mode == GhostMode::EATEN) return; // don't interrupt eaten mode

    if (mode == GhostMode::FRIGHTENED) {
        m_frightenedTimer = 360; // 6 seconds at 60fps
        reverseDirection();
    }
    m_mode = mode;
}

void Ghost::reverseDirection() {
    m_direction = oppositeDir(m_direction);
}

void Ghost::setEaten() {
    m_mode  = GhostMode::EATEN;
    m_speed = 4; // eyes move fast
}

// ============================================================
// Target Tile Computation
// ============================================================

GridPos Ghost::computeTargetTile(const Maze& maze, const Player& pacman, const Ghost* blinky) const {
    if (m_mode == GhostMode::SCATTER) {
        return maze.getGhostScatterTarget(m_id);
    }

    // Chase mode — each ghost has unique targeting
    switch (m_id) {
        case GhostID::BLINKY: return targetBlinky(pacman);
        case GhostID::PINKY:  return targetPinky(pacman);
        case GhostID::INKY:   return targetInky(pacman, blinky);
        case GhostID::CLYDE:  return targetClyde(maze, pacman);
    }
    return pacman.getPos();
}

GridPos Ghost::targetBlinky(const Player& pacman) const {
    // Blinky directly targets Pac-Man's current tile
    return pacman.getPos();
}

GridPos Ghost::targetPinky(const Player& pacman) const {
    // Pinky targets 4 tiles ahead of Pac-Man in his current direction
    auto [dr, dc] = dirDelta(pacman.getDirection());
    GridPos target = pacman.getPos();
    target.row += dr * 4;
    target.col += dc * 4;

    // Original bug: if Pac-Man faces UP, the target is also offset 4 left
    if (pacman.getDirection() == Direction::UP) {
        target.col -= 4;
    }
    return target;
}

GridPos Ghost::targetInky(const Player& pacman, const Ghost* blinky) const {
    // Inky: take tile 2 ahead of Pac-Man, then double the vector from Blinky to that tile
    if (!blinky) return pacman.getPos();

    auto [dr, dc] = dirDelta(pacman.getDirection());
    GridPos pivot = pacman.getPos();
    pivot.row += dr * 2;
    pivot.col += dc * 2;

    GridPos blinkyPos = blinky->getPos();
    GridPos target;
    target.row = pivot.row + (pivot.row - blinkyPos.row);
    target.col = pivot.col + (pivot.col - blinkyPos.col);
    return target;
}

GridPos Ghost::targetClyde(const Maze& maze, const Player& pacman) const {
    // Clyde: if distance to Pac-Man > 8 tiles, target Pac-Man. Otherwise scatter.
    int dist = AStar::heuristic(m_pos, pacman.getPos());
    if (dist > 8) {
        return pacman.getPos();
    }
    return maze.getGhostScatterTarget(m_id);
}

// ============================================================
// Direction Selection
// ============================================================

Direction Ghost::chooseFrightenedDirection(const Maze& maze) const {
    // In frightened mode, choose a random valid direction (but not reverse)
    Direction options[4] = {Direction::UP, Direction::DOWN, Direction::LEFT, Direction::RIGHT};
    Direction opposite = oppositeDir(m_direction);

    // Shuffle-ish: try random directions
    std::vector<Direction> valid;
    for (int i = 0; i < 4; i++) {
        if (options[i] != opposite && canMove(maze, options[i])) {
            valid.push_back(options[i]);
        }
    }

    if (valid.empty()) {
        // Dead end — must reverse
        if (canMove(maze, opposite)) return opposite;
        return m_direction;
    }

    return valid[std::rand() % valid.size()];
}

Direction Ghost::chooseBestDirection(const Maze& maze, GridPos target) const {
    // Classic Pac-Man ghost decision: at each intersection, pick the direction
    // that minimizes straight-line distance to the target tile.
    // Ghosts cannot reverse direction (except when mode changes).
    // Priority order for ties: UP, LEFT, DOWN, RIGHT

    Direction priorities[] = {Direction::UP, Direction::LEFT, Direction::DOWN, Direction::RIGHT};
    Direction opposite = oppositeDir(m_direction);

    Direction bestDir = Direction::NONE;
    int bestDist = 999999;

    for (auto dir : priorities) {
        if (dir == opposite) continue; // no reversing
        if (!canMove(maze, dir)) continue;

        auto [dr, dc] = dirDelta(dir);
        GridPos nextTile = {m_pos.row + dr, m_pos.col + dc};

        // Use squared Euclidean distance (like the original game)
        int dx = nextTile.col - target.col;
        int dy = nextTile.row - target.row;
        int dist = dx * dx + dy * dy;

        if (dist < bestDist) {
            bestDist = dist;
            bestDir = dir;
        }
    }

    if (bestDir == Direction::NONE) {
        // Only option is to reverse (dead end)
        if (canMove(maze, opposite)) return opposite;
    }

    return bestDir;
}

bool Ghost::canMove(const Maze& maze, Direction dir) const {
    auto [dr, dc] = dirDelta(dir);
    int nr = m_pos.row + dr;
    int nc = m_pos.col + dc;

    // Allow ghosts through ghost door when eaten (returning to house)
    if (m_mode == GhostMode::EATEN) {
        if (maze.isInBounds(nr, nc)) {
            TileType t = maze.getTile(nr, nc);
            return t != TileType::WALL;
        }
    }

    return maze.isWalkable(nr, nc);
}

bool Ghost::isAlignedToGrid() const {
    int expectedX = m_pos.col * TILE_SIZE + TILE_SIZE / 2;
    int expectedY = m_pos.row * TILE_SIZE + TILE_SIZE / 2;
    return (m_pixelX == expectedX && m_pixelY == expectedY);
}
