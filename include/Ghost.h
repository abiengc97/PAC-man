#pragma once

#include "Common.h"
#include <vector>

class Maze;
class Player;

class Ghost {
public:
    Ghost(GhostID id);

    void init(const Maze& maze);
    void update(const Maze& maze, const Player& pacman, const Ghost* blinky);
    void reset(const Maze& maze);

    // Mode control (called by Game Manager)
    void setMode(GhostMode mode);
    void reverseDirection();

    // Getters
    GhostID   getID()        const { return m_id; }
    GridPos   getPos()       const { return m_pos; }
    GhostMode getMode()      const { return m_mode; }
    Direction getDirection()  const { return m_direction; }
    int       getPixelX()    const { return m_pixelX; }
    int       getPixelY()    const { return m_pixelY; }
    bool      isInHouse()    const { return m_inHouse; }
    int       getFrightenedTimer() const { return m_frightenedTimer; }

    // For eaten ghost returning to house
    void setEaten();

private:
    GhostID   m_id;
    GridPos   m_pos;
    GridPos   m_spawnPos;
    Direction m_direction  = Direction::LEFT;
    GhostMode m_mode       = GhostMode::SCATTER;

    int m_pixelX = 0;
    int m_pixelY = 0;
    int m_speed  = 2;

    bool m_inHouse = true;
    int  m_houseTimer = 0;    // ticks before leaving ghost house
    int  m_frightenedTimer = 0;

    // Each ghost leaves the house at different times
    int  m_houseExitDelay = 0;

    // Target tile computation (the core AI)
    GridPos computeTargetTile(const Maze& maze, const Player& pacman, const Ghost* blinky) const;

    // Blinky: targets Pac-Man directly
    GridPos targetBlinky(const Player& pacman) const;

    // Pinky: targets 4 tiles ahead of Pac-Man
    GridPos targetPinky(const Player& pacman) const;

    // Inky: uses Blinky's position for a vector calculation
    GridPos targetInky(const Player& pacman, const Ghost* blinky) const;

    // Clyde: targets Pac-Man when far, scatter corner when close
    GridPos targetClyde(const Maze& maze, const Player& pacman) const;

    // Frightened mode: random valid direction at each intersection
    Direction chooseFrightenedDirection(const Maze& maze) const;

    // Choose best direction toward target (classic Pac-Man decision logic)
    Direction chooseBestDirection(const Maze& maze, GridPos target) const;

    bool canMove(const Maze& maze, Direction dir) const;
    bool isAlignedToGrid() const;
};
