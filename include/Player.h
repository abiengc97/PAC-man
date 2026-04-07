#pragma once

#include "Common.h"

class Maze;

class Player {
public:
    Player();

    void init(GridPos spawn);
    void handleInput(Direction inputDir);
    void update(Maze& maze);
    void reset(GridPos spawn);

    // Getters
    GridPos   getPos()       const { return m_pos; }
    Direction getDirection()  const { return m_direction; }
    Direction getBufferedDirection() const { return m_nextDir; }
    bool      isAlive()      const { return m_alive; }
    int       getPixelX()    const { return m_pixelX; }
    int       getPixelY()    const { return m_pixelY; }
    int       getAnimFrame() const { return m_animFrame; }
    bool      isGridAligned() const { return isAlignedToGrid(); }

    // Called by Game when ghost collision happens
    void die();
    void respawn(GridPos spawn);

    // Movement speed (pixels per frame)
    void setSpeed(int speed) { m_speed = speed; }

private:
    GridPos   m_pos;
    Direction m_direction  = Direction::NONE;
    Direction m_nextDir    = Direction::NONE;  // buffered input
    bool      m_alive      = true;

    // Sub-tile pixel position for smooth movement
    int m_pixelX = 0;
    int m_pixelY = 0;
    int m_speed  = 2; // pixels per frame

    // Animation
    int m_animFrame  = 0;
    int m_animTimer  = 0;

    bool canMove(const Maze& maze, Direction dir) const;
    void alignToGrid();
    bool isAlignedToGrid() const;
};
