#pragma once

#include "Common.h"
#include <vector>

class Maze;

namespace AStar {

// Find shortest path from start to goal on the maze grid.
// Returns a vector of GridPos from start to goal (inclusive).
// Returns empty vector if no path exists.
std::vector<GridPos> findPath(const Maze& maze, GridPos start, GridPos goal);

// Get the next direction to move from 'from' toward 'goal'.
// This is the main function ghosts call each tick.
Direction getNextDirection(const Maze& maze, GridPos from, GridPos goal);

// Manhattan distance heuristic
int heuristic(GridPos a, GridPos b);

} // namespace AStar
