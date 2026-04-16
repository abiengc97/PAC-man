#pragma once

#include "Common.h"
#include "Maze.h"
#include "Player.h"
#include "Ghost.h"
#include "Renderer.h"
#include <array>
#include <fstream>
#include <string>
#include <vector>

class Game {
public:
    explicit Game(bool headless = false, bool rlMode = false,
                  int startLevel = 1, bool rlRender = false);
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

    struct BonusFruitState {
        bool      active = false;
        FruitType type = FruitType::CHERRY;
        GridPos   pos;
        int       points = 0;
        int       timer = 0;
        int       spawnCount = 0;
    };

    BonusFruitState m_bonusFruit;

    // Ready timer (brief pause at level start)
    int m_readyTimer = 0;
    static constexpr int READY_DURATION = 120; // 2 seconds

    struct LoggedGhostState {
        GhostID   id = GhostID::BLINKY;
        GridPos   pos;
        int       pixelX = 0;
        int       pixelY = 0;
        Direction direction = Direction::NONE;
        GhostMode mode = GhostMode::SCATTER;
        bool      inHouse = true;
        int       manhattanDistanceToPlayer = 0;
    };

    struct FrameObservation {
        uint64_t tickMs = 0;
        int      frameNumber = 0;
        GameState state = GameState::MENU;
        int      score = 0;
        int      lives = 0;
        int      level = 0;
        int      remainingPellets = 0;
        int      ghostEatCombo = 0;
        int      modeTimer = 0;
        int      modePhase = 0;
        int      modeStep = 0;
        bool     isChaseMode = false;
        GridPos   playerPos;
        int       playerPixelX = 0;
        int       playerPixelY = 0;
        Direction playerDirection = Direction::NONE;
        Direction bufferedDirection = Direction::NONE;
        Direction targetAction = Direction::NONE;
        TileType  playerTile = TileType::EMPTY;
        bool      playerAlive = true;
        bool      playerAlignedToGrid = false;
        std::vector<GridPos> powerPelletPositions;
        bool      bonusFruitActive = false;
        GridPos   bonusFruitPos;
        FruitType bonusFruitType = FruitType::CHERRY;
        int       bonusFruitPoints = 0;
        std::array<LoggedGhostState, 4> ghosts;
    };

    struct FrameEvents {
        std::vector<std::string> keypresses;
        bool atePellet = false;
        bool atePowerPellet = false;
        bool ateBonusFruit = false;
        int  bonusFruitScore = 0;
        std::array<bool, 4> ghostsEaten{};
        bool playerDied = false;
        bool levelCleared = false;
        bool gameOver = false;
    };

    // Headless / RL mode
    bool      m_headless   = false;
    bool      m_rlMode     = false;
    bool      m_rlRender   = false;
    int       m_startLevel = 1;
    Direction m_rlAction   = Direction::NONE;

    // State layout (89 floats, all values normalised to [0, 1]):
    //   [0-1]   player row/30, col/27
    //   [2-9]   4× ghost row/30, col/27
    //   [10-13] 4× ghost is_frightened
    //   [14-17] 4× ghost frightened_timer / 360
    //   [18-21] 4× ghost is_in_house
    //   [22]    chase_mode
    //   [23]    mode_timer / 1200
    //   [24]    remaining_pellets / total_pellets
    //   [25]    ghost_eat_combo / 4
    //   [26-27] nearest pellet row/30, col/27
    //   [28-31] nearest pellet distance N/S/W/E (steps/30; 1.0 = wall/no pellet)
    //   [32-39] 4× power pellet row/30, col/27 (0,0 if consumed)
    //   [40-88] 7×7 local tile window / 7.0
    static constexpr int RL_STATE_SIZE = 89;

    void  runRL();
    std::array<float, RL_STATE_SIZE> buildStateVector() const;
    void  writeRLStep(const std::array<float, RL_STATE_SIZE>& state,
                      float reward, bool done) const;

    // Per-episode tile visit counts for exploration novelty reward (RL only)
    std::array<std::array<int, MAZE_COLS>, MAZE_ROWS> m_rlVisitCount{};

    bool        m_loggingEnabled = false;
    std::ofstream m_logFile;
    std::string m_logPath;
    int         m_loggedFrameCount = 0;
    FrameEvents m_frameEvents;

    // Main loop steps
    void processInput();
    void update();
    void render();

    // Game logic
    void checkCollisions(FrameEvents& events);
    void updateGhostModes();
    void startLevel();
    void nextLevel();
    void playerDied();
    void eatPellet(TileType pelletType);
    void resetBonusFruit();
    void updateBonusFruit(FrameEvents& events);
    GridPos findBonusFruitSpawn() const;

    // Data logging
    bool initLogging();
    void closeLogging();
    void resetFrameEvents();
    FrameObservation captureObservation() const;
    Direction getTargetAction() const;
    int getGhostModeStep() const;
    std::vector<GridPos> collectTilePositions(TileType tileType) const;
    void logFrame(const FrameObservation& observation, const FrameEvents& events);
};
