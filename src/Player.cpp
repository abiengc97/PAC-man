#include "Player.h"
#include "Maze.h"

Player::Player() {}

void Player::init(GridPos spawn) {
    m_pos       = spawn;
    m_direction = Direction::NONE;
    m_nextDir   = Direction::NONE;
    m_alive     = true;
    m_pixelX    = spawn.col * TILE_SIZE + TILE_SIZE / 2;
    m_pixelY    = spawn.row * TILE_SIZE + TILE_SIZE / 2;
    m_animFrame = 0;
    m_animTimer = 0;
}

void Player::handleInput(Direction inputDir) {
    m_nextDir = inputDir;
}

void Player::update(Maze& maze) {
    if (!m_alive) return;

    // Animation
    m_animTimer++;
    if (m_animTimer >= 4) {
        m_animTimer = 0;
        m_animFrame = (m_animFrame + 1) % 4;
    }

    // Try to apply buffered direction when aligned to grid
    if (isAlignedToGrid() && m_nextDir != Direction::NONE) {
        if (canMove(maze, m_nextDir)) {
            m_direction = m_nextDir;
            m_nextDir = Direction::NONE;
        }
    }

    // Continue in current direction
    if (m_direction == Direction::NONE) return;

    if (isAlignedToGrid() && !canMove(maze, m_direction)) {
        // Hit a wall, stop
        return;
    }

    // Move in current direction
    auto [dr, dc] = dirDelta(m_direction);
    m_pixelX += dc * m_speed;
    m_pixelY += dr * m_speed;

    // Update grid position based on pixel center
    int gridCol = m_pixelX / TILE_SIZE;
    int gridRow = m_pixelY / TILE_SIZE;

    // Tunnel wrap
    if (m_pixelX < 0) {
        m_pixelX = maze.getCols() * TILE_SIZE - m_speed;
        gridCol = maze.getCols() - 1;
    } else if (m_pixelX >= maze.getCols() * TILE_SIZE) {
        m_pixelX = m_speed;
        gridCol = 0;
    }

    m_pos = {gridRow, gridCol};
}

void Player::reset(GridPos spawn) {
    init(spawn);
}

void Player::die() {
    m_alive = false;
    m_direction = Direction::NONE;
}

void Player::respawn(GridPos spawn) {
    init(spawn);
}

bool Player::canMove(const Maze& maze, Direction dir) const {
    auto [dr, dc] = dirDelta(dir);
    int newRow = m_pos.row + dr;
    int newCol = m_pos.col + dc;
    return maze.isWalkable(newRow, newCol);
}

void Player::alignToGrid() {
    m_pixelX = m_pos.col * TILE_SIZE + TILE_SIZE / 2;
    m_pixelY = m_pos.row * TILE_SIZE + TILE_SIZE / 2;
}

bool Player::isAlignedToGrid() const {
    int expectedX = m_pos.col * TILE_SIZE + TILE_SIZE / 2;
    int expectedY = m_pos.row * TILE_SIZE + TILE_SIZE / 2;
    return (m_pixelX == expectedX && m_pixelY == expectedY);
}
