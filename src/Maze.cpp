#include "Maze.h"
#include <fstream>
#include <iostream>
#include <sstream>

Maze::Maze() {}

bool Maze::loadFromFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Maze: failed to open " << filepath << std::endl;
        return false;
    }

    m_grid.clear();
    std::string line;
    int row = 0;

    while (std::getline(file, line)) {
        std::vector<TileType> gridRow;
        for (int col = 0; col < static_cast<int>(line.size()); ++col) {
            char c = line[col];
            TileType tile = TileType::EMPTY;

            switch (c) {
                case '#': tile = TileType::WALL;        break;
                case '.': tile = TileType::PELLET;      break;
                case 'o': tile = TileType::POWER;       break;
                case ' ': tile = TileType::EMPTY;       break;
                case 'H': tile = TileType::GHOST_HOUSE; break;
                case 'D': tile = TileType::GHOST_DOOR;  break;
                case 'T': tile = TileType::TUNNEL;      break;

                // Spawn markers — tile underneath is EMPTY
                case 'P': // Pac-Man spawn
                    m_pacmanSpawn = {row, col};
                    tile = TileType::EMPTY;
                    break;
                case 'B': // Blinky spawn
                    m_ghostSpawns[static_cast<int>(GhostID::BLINKY)] = {row, col};
                    tile = TileType::EMPTY;
                    break;
                case 'I': // Pinky spawn (pInky would clash, use I for Pinky? Let's use N)
                    m_ghostSpawns[static_cast<int>(GhostID::INKY)] = {row, col};
                    tile = TileType::GHOST_HOUSE;
                    break;
                case 'N': // piNky spawn
                    m_ghostSpawns[static_cast<int>(GhostID::PINKY)] = {row, col};
                    tile = TileType::GHOST_HOUSE;
                    break;
                case 'C': // Clyde spawn
                    m_ghostSpawns[static_cast<int>(GhostID::CLYDE)] = {row, col};
                    tile = TileType::GHOST_HOUSE;
                    break;
                case 'E': // Ghost house entry (above the door)
                    m_ghostHouseEntry = {row, col};
                    tile = TileType::EMPTY;
                    break;

                default:
                    tile = TileType::EMPTY;
                    break;
            }
            gridRow.push_back(tile);
        }
        m_grid.push_back(gridRow);
        ++row;
    }

    m_rows = static_cast<int>(m_grid.size());
    m_cols = m_rows > 0 ? static_cast<int>(m_grid[0].size()) : 0;

    // Store original layout for level reset
    m_originalGrid = m_grid;
    countPellets();

    std::cout << "Maze loaded: " << m_rows << " x " << m_cols
              << " (" << m_totalPellets << " pellets)" << std::endl;
    return true;
}

TileType Maze::getTile(int row, int col) const {
    if (!isInBounds(row, col)) return TileType::WALL;
    return m_grid[row][col];
}

bool Maze::isWalkable(int row, int col) const {
    if (!isInBounds(row, col)) {
        // Allow wrapping through tunnels at row boundaries
        if (row >= 0 && row < m_rows && (col < 0 || col >= m_cols)) {
            return true; // tunnel wrap
        }
        return false;
    }
    TileType t = m_grid[row][col];
    return t != TileType::WALL && t != TileType::GHOST_HOUSE && t != TileType::GHOST_DOOR;
}

bool Maze::isInBounds(int row, int col) const {
    return row >= 0 && row < m_rows && col >= 0 && col < m_cols;
}

void Maze::eatPellet(int row, int col) {
    if (!isInBounds(row, col)) return;
    TileType& t = m_grid[row][col];
    if (t == TileType::PELLET || t == TileType::POWER) {
        t = TileType::EMPTY;
        --m_remainingPellets;
    }
}

int Maze::getRemainingPellets() const {
    return m_remainingPellets;
}

int Maze::getTotalPellets() const {
    return m_totalPellets;
}

GridPos Maze::wrapPosition(int row, int col) const {
    GridPos p;
    p.row = row;
    if (col < 0) {
        p.col = m_cols - 1;
    } else if (col >= m_cols) {
        p.col = 0;
    } else {
        p.col = col;
    }
    return p;
}

void Maze::resetPellets() {
    m_grid = m_originalGrid;
    countPellets();
}

GridPos Maze::getGhostSpawn(GhostID id) const {
    return m_ghostSpawns[static_cast<int>(id)];
}

GridPos Maze::getGhostScatterTarget(GhostID id) const {
    // Classic Pac-Man scatter corners
    switch (id) {
        case GhostID::BLINKY: return {0, m_cols - 1};           // top-right
        case GhostID::PINKY:  return {0, 0};                     // top-left
        case GhostID::INKY:   return {m_rows - 1, m_cols - 1};  // bottom-right
        case GhostID::CLYDE:  return {m_rows - 1, 0};            // bottom-left
    }
    return {0, 0};
}

void Maze::countPellets() {
    m_totalPellets = 0;
    for (auto& row : m_grid) {
        for (auto& t : row) {
            if (t == TileType::PELLET || t == TileType::POWER)
                ++m_totalPellets;
        }
    }
    m_remainingPellets = m_totalPellets;
}
