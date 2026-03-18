#include "AStar.h"
#include "Maze.h"
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <functional>

namespace AStar {

// Hash for GridPos
struct GridPosHash {
    size_t operator()(const GridPos& p) const {
        return std::hash<int>()(p.row) ^ (std::hash<int>()(p.col) << 16);
    }
};

int heuristic(GridPos a, GridPos b) {
    return std::abs(a.row - b.row) + std::abs(a.col - b.col);
}

std::vector<GridPos> findPath(const Maze& maze, GridPos start, GridPos goal) {
    // Priority queue: (f_cost, GridPos)
    using Node = std::pair<int, GridPos>;
    auto cmp = [](const Node& a, const Node& b) { return a.first > b.first; };
    std::priority_queue<Node, std::vector<Node>, decltype(cmp)> openSet(cmp);

    std::unordered_map<GridPos, GridPos, GridPosHash> cameFrom;
    std::unordered_map<GridPos, int, GridPosHash> gScore;

    gScore[start] = 0;
    openSet.push({heuristic(start, goal), start});

    while (!openSet.empty()) {
        auto [fCost, current] = openSet.top();
        openSet.pop();

        if (current == goal) {
            // Reconstruct path
            std::vector<GridPos> path;
            GridPos node = goal;
            while (node != start) {
                path.push_back(node);
                node = cameFrom[node];
            }
            path.push_back(start);
            std::reverse(path.begin(), path.end());
            return path;
        }

        // Skip if we've already found a better path to this node
        if (gScore.count(current) && gScore[current] < fCost - heuristic(current, goal)) {
            continue;
        }

        // Explore neighbors in all 4 directions
        for (int d = 0; d < 4; ++d) {
            auto [dr, dc] = DIR_DELTA[d];
            int nr = current.row + dr;
            int nc = current.col + dc;

            // Handle tunnel wrap
            GridPos neighbor;
            if (nc < 0 || nc >= maze.getCols()) {
                neighbor = maze.wrapPosition(nr, nc);
            } else {
                neighbor = {nr, nc};
            }

            // Check walkability (ghosts can walk through ghost door when eaten)
            if (!maze.isInBounds(neighbor.row, neighbor.col)) continue;

            TileType tile = maze.getTile(neighbor.row, neighbor.col);
            if (tile == TileType::WALL) continue;

            int tentativeG = gScore[current] + 1;

            if (!gScore.count(neighbor) || tentativeG < gScore[neighbor]) {
                cameFrom[neighbor] = current;
                gScore[neighbor] = tentativeG;
                int f = tentativeG + heuristic(neighbor, goal);
                openSet.push({f, neighbor});
            }
        }
    }

    // No path found
    return {};
}

Direction getNextDirection(const Maze& maze, GridPos from, GridPos goal) {
    auto path = findPath(maze, from, goal);

    if (path.size() < 2) {
        return Direction::NONE;
    }

    GridPos next = path[1];
    int dr = next.row - from.row;
    int dc = next.col - from.col;

    // Handle tunnel wrap case
    if (dc > 1) return Direction::LEFT;   // wrapped right-to-left
    if (dc < -1) return Direction::RIGHT; // wrapped left-to-right

    if (dr == -1) return Direction::UP;
    if (dr ==  1) return Direction::DOWN;
    if (dc == -1) return Direction::LEFT;
    if (dc ==  1) return Direction::RIGHT;

    return Direction::NONE;
}

} // namespace AStar
