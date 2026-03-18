#pragma once

#include "Common.h"
#include "Maze.h"
#include "Player.h"
#include "Ghost.h"
#include "Renderer.h"
#include <array>

class Game {
public:
    Game();
    ~Game();

    bool init();
    void run();

private:
    // Core systems
    Renderer m_renderer;
    Maze     m_maze;
    Player   m_player;
    std::array<Ghost, 4> m_ghosts;

    // Game state
    GameState m_state    = GameState::MENU;
    bool      m_running  = true;
    int       m_score    = 0;
    int       m_lives    = 3;
    int       m_level    = 1;

    // Ghost mode timer (Chase/Scatter alternation)
    int  m_modeTimer     = 0;
    int  m_modePhase     = 0; // index into the scatter/chase schedule
    bool m_isChaseMode   = false;

    // Scatter/Chase duration schedule (in ticks at 60fps)
    // Original: 7s scatter, 20s chase, 7s scatter, 20s chase, 5s scatter, 20s chase, 5s scatter, then chase forever
    static constexpr int MODE_SCHEDULE[][2] = {
        {420, 1200},  // 7s scatter, 20s chase
        {420, 1200},  // 7s scatter, 20s chase
        {300, 1200},  // 5s scatter, 20s chase
        {300, -1},    // 5s scatter, then chase forever
    };
    static constexpr int MODE_SCHEDULE_SIZE = 4;

    // Ghost eat scoring
    int m_ghostEatCombo = 0; // resets per power pellet

    // Ready timer (brief pause at level start)
    int m_readyTimer = 0;
    static constexpr int READY_DURATION = 120; // 2 seconds

    // Main loop steps
    void processInput();
    void update();
    void render();

    // Game logic
    void checkCollisions();
    void updateGhostModes();
    void startLevel();
    void nextLevel();
    void playerDied();
    void eatPellet(TileType pelletType);
};
