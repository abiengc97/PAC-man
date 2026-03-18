#pragma once

#include "Common.h"
#include <string>
#include <vector>

class Maze {
public:
    Maze();

    // Load maze layout from a text file
    bool loadFromFile(const std::string& filepath);

    // Tile queries
    TileType getTile(int row, int col) const;
    bool     isWalkable(int row, int col) const;
    bool     isInBounds(int row, int col) const;

    // Pellet management
    void     eatPellet(int row, int col);
    int      getRemainingPellets() const;
    int      getTotalPellets() const;

    // Tunnel wrap
    GridPos  wrapPosition(int row, int col) const;

    // Reset pellets for a new level (keeps wall layout)
    void     resetPellets();

    // Getters
    int getRows() const { return m_rows; }
    int getCols() const { return m_cols; }

    // Spawn positions (parsed from level file)
    GridPos getPacManSpawn()   const { return m_pacmanSpawn; }
    GridPos getGhostSpawn(GhostID id) const;
    GridPos getGhostScatterTarget(GhostID id) const;
    GridPos getGhostHouseEntry() const { return m_ghostHouseEntry; }

    // Raw grid access for renderer
    const std::vector<std::vector<TileType>>& getGrid() const { return m_grid; }

private:
    int m_rows = 0;
    int m_cols = 0;
    std::vector<std::vector<TileType>> m_grid;
    std::vector<std::vector<TileType>> m_originalGrid; // for reset

    GridPos m_pacmanSpawn;
    GridPos m_ghostSpawns[4];
    GridPos m_ghostHouseEntry;

    int m_totalPellets    = 0;
    int m_remainingPellets = 0;

    void countPellets();
};
