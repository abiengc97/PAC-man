#include "Game.h"
#include "Common.h"
#include <SDL2/SDL.h>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <climits>
#include <cstdlib>
#include <cmath>
#include <ctime>

namespace {

constexpr int LOG_FLUSH_INTERVAL = FPS;

const char* directionToString(Direction direction) {
    switch (direction) {
        case Direction::UP: return "up";
        case Direction::DOWN: return "down";
        case Direction::LEFT: return "left";
        case Direction::RIGHT: return "right";
        case Direction::NONE: return "none";
    }
    return "none";
}

const char* ghostModeToString(GhostMode mode) {
    switch (mode) {
        case GhostMode::CHASE: return "chase";
        case GhostMode::SCATTER: return "scatter";
        case GhostMode::FRIGHTENED: return "frightened";
        case GhostMode::EATEN: return "eaten";
    }
    return "scatter";
}

const char* ghostIdToString(GhostID id) {
    switch (id) {
        case GhostID::BLINKY: return "blinky";
        case GhostID::PINKY: return "pinky";
        case GhostID::INKY: return "inky";
        case GhostID::CLYDE: return "clyde";
    }
    return "unknown";
}

const char* tileTypeToString(TileType tile) {
    switch (tile) {
        case TileType::WALL: return "wall";
        case TileType::EMPTY: return "empty";
        case TileType::PELLET: return "pellet";
        case TileType::POWER: return "power_pellet";
        case TileType::GHOST_HOUSE: return "ghost_house";
        case TileType::GHOST_DOOR: return "ghost_door";
        case TileType::TUNNEL: return "tunnel";
    }
    return "empty";
}

const char* gameStateToString(GameState state) {
    switch (state) {
        case GameState::MENU: return "menu";
        case GameState::PLAYING: return "playing";
        case GameState::PAUSED: return "paused";
        case GameState::LEVEL_CLEAR: return "level_clear";
        case GameState::DYING: return "dying";
        case GameState::GAME_OVER: return "game_over";
    }
    return "menu";
}

std::string escapeJson(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());

    for (char ch : value) {
        switch (ch) {
            case '\\': escaped += "\\\\"; break;
            case '"': escaped += "\\\""; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped += ch; break;
        }
    }

    return escaped;
}

void writeGridPosJson(std::ostream& out, GridPos pos) {
    out << "{\"row\":" << pos.row << ",\"col\":" << pos.col << "}";
}

void writeGridPosListJson(std::ostream& out, const std::vector<GridPos>& positions) {
    out << "[";
    for (std::size_t i = 0; i < positions.size(); ++i) {
        if (i > 0) out << ",";
        writeGridPosJson(out, positions[i]);
    }
    out << "]";
}

int manhattanDistance(GridPos a, GridPos b) {
    return std::abs(a.row - b.row) + std::abs(a.col - b.col);
}

std::string timestampForFilename() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t nowTime = std::chrono::system_clock::to_time_t(now);

    std::tm localTime{};
#if defined(_WIN32)
    localtime_s(&localTime, &nowTime);
#else
    localtime_r(&nowTime, &localTime);
#endif

    std::ostringstream oss;
    oss << std::put_time(&localTime, "%Y%m%d_%H%M%S");
    return oss.str();
}

} // namespace

constexpr int Game::MODE_SCHEDULE[][2];

Game::Game(bool headless, bool rlMode, int startLevel, bool rlRender)
    : m_ghosts{Ghost(GhostID::BLINKY), Ghost(GhostID::PINKY),
               Ghost(GhostID::INKY), Ghost(GhostID::CLYDE)}
    , m_headless(headless)
    , m_rlMode(rlMode)
    , m_rlRender(rlRender)
    , m_startLevel(startLevel)
    , m_level(startLevel)
{
    std::srand(static_cast<unsigned>(std::time(nullptr)));
}

Game::~Game() {
    closeLogging();
}

bool Game::init() {
    if (!m_headless) {
        if (!m_renderer.init("Pac-Man", SCREEN_W, SCREEN_H)) {
            return false;
        }
    }

    startLevel();

    if (m_maze.getTotalPellets() == 0) {
        std::cerr << "Failed to load maze!" << std::endl;
        return false;
    }

    if (!m_rlMode) {
        initLogging();
    }
    return true;
}

void Game::startLevel() {
    std::string levelFile = "levels/level" + std::to_string(m_level) + ".txt";
    if (!m_maze.loadFromFile(levelFile)) {
        m_maze.loadFromFile("levels/level1.txt");
    }
    m_player.init(m_maze.getPacManSpawn());

    for (auto& ghost : m_ghosts) {
        ghost.init(m_maze);
    }

    m_modeTimer   = 0;
    m_modePhase   = 0;
    m_isChaseMode = false;
    m_ghostEatCombo = 0;
    m_readyTimer  = READY_DURATION;
    m_state       = GameState::PLAYING;
}

void Game::run() {
    if (m_rlMode) {
        runRL();
        return;
    }

    Uint32 frameStart;
    int frameTime;

    while (m_running) {
        frameStart = SDL_GetTicks();

        processInput();
        update();
        render();

        // Frame rate cap
        frameTime = SDL_GetTicks() - frameStart;
        if (FRAME_DELAY > frameTime) {
            SDL_Delay(FRAME_DELAY - frameTime);
        }
    }
}

void Game::processInput() {
    resetFrameEvents();

    if (m_rlMode) {
        if (m_rlAction != Direction::NONE) {
            m_player.handleInput(m_rlAction);
        }
        return;
    }

    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) {
            m_running = false;
            return;
        }

        if (event.type == SDL_KEYDOWN) {
            m_frameEvents.keypresses.push_back(SDL_GetKeyName(event.key.keysym.sym));

            switch (event.key.keysym.sym) {
                case SDLK_ESCAPE:
                    m_running = false;
                    break;

                case SDLK_p:
                    if (m_state == GameState::PLAYING)
                        m_state = GameState::PAUSED;
                    else if (m_state == GameState::PAUSED)
                        m_state = GameState::PLAYING;
                    break;

                case SDLK_RETURN:
                    if (m_state == GameState::GAME_OVER) {
                        // Restart
                        m_score = 0;
                        m_lives = 3;
                        m_level = 1;
                        startLevel();
                    }
                    break;

                case SDLK_UP:    case SDLK_w: m_player.handleInput(Direction::UP);    break;
                case SDLK_DOWN:  case SDLK_s: m_player.handleInput(Direction::DOWN);  break;
                case SDLK_LEFT:  case SDLK_a: m_player.handleInput(Direction::LEFT);  break;
                case SDLK_RIGHT: case SDLK_d: m_player.handleInput(Direction::RIGHT); break;
            }
        }
    }
}

void Game::update() {
    const bool shouldLogFrame = m_loggingEnabled &&
                                m_state == GameState::PLAYING &&
                                m_readyTimer == 0;
    FrameObservation observation;
    if (shouldLogFrame) {
        observation = captureObservation();
    }

    if (m_state != GameState::PLAYING) return;

    // Ready countdown at level start — skip entirely in RL mode for training speed
    if (m_readyTimer > 0) {
        if (m_rlMode) {
            m_readyTimer = 0;
        } else {
            m_readyTimer--;
            return;
        }
    }

    // Update ghost chase/scatter schedule
    updateGhostModes();

    // Update player
    m_player.update(m_maze);

    // Check if Pac-Man is on a pellet
    GridPos pPos = m_player.getPos();
    TileType tile = m_maze.getTile(pPos.row, pPos.col);
    if (tile == TileType::PELLET || tile == TileType::POWER) {
        m_frameEvents.atePellet = tile == TileType::PELLET;
        m_frameEvents.atePowerPellet = tile == TileType::POWER;
        eatPellet(tile);
    }

    // Update ghosts
    for (auto& ghost : m_ghosts) {
        // Pass blinky reference to inky for targeting
        const Ghost* blinky = &m_ghosts[static_cast<int>(GhostID::BLINKY)];
        ghost.update(m_maze, m_player, blinky);
    }

    // Check collisions
    checkCollisions(m_frameEvents);

    // Check level clear — log before nextLevel() resets maze/player/ghosts
    if (m_maze.getRemainingPellets() == 0) {
        m_frameEvents.levelCleared = true;
        m_state = GameState::LEVEL_CLEAR;
        if (shouldLogFrame) {
            logFrame(observation, m_frameEvents);
        }
        nextLevel();
        return;
    }

    if (shouldLogFrame) {
        logFrame(observation, m_frameEvents);
    }

    // Respawn after logging so player_after captures the death position
    if (m_frameEvents.playerDied && !m_frameEvents.gameOver) {
        m_player.respawn(m_maze.getPacManSpawn());
        for (auto& ghost : m_ghosts) {
            ghost.init(m_maze);
        }
        m_readyTimer  = READY_DURATION;
        m_modeTimer   = 0;
        m_modePhase   = 0;
        m_isChaseMode = false;
    }
}

void Game::render() {
    if (m_headless) return;
    m_renderer.clear();

    m_renderer.drawMaze(m_maze);
    m_renderer.drawPlayer(m_player);

    for (const auto& ghost : m_ghosts) {
        m_renderer.drawGhost(ghost);
    }

    m_renderer.drawHUD(m_score, m_lives, m_level);

    if (m_readyTimer > 0) {
        m_renderer.drawReady();
    }

    if (m_state == GameState::PAUSED) {
        m_renderer.drawPause();
    }

    if (m_state == GameState::GAME_OVER) {
        m_renderer.drawGameOver(m_score);
    }

    m_renderer.present();
}

void Game::checkCollisions(FrameEvents& events) {
    GridPos pPos = m_player.getPos();

    for (std::size_t i = 0; i < m_ghosts.size(); ++i) {
        auto& ghost = m_ghosts[i];
        if (ghost.isInHouse()) continue;
        if (ghost.getPos() != pPos) continue;

        switch (ghost.getMode()) {
            case GhostMode::FRIGHTENED:
                // Eat the ghost
                ghost.setEaten();
                events.ghostsEaten[i] = true;
                m_ghostEatCombo++;
                m_score += 200 * static_cast<int>(std::pow(2, m_ghostEatCombo - 1));
                break;

            case GhostMode::EATEN:
                // Ghost is already eaten, no collision
                break;

            default:
                // Ghost kills Pac-Man
                events.playerDied = true;
                playerDied();
                events.gameOver = m_state == GameState::GAME_OVER;
                return;
        }
    }
}

void Game::updateGhostModes() {
    m_modeTimer++;

    if (m_modePhase >= MODE_SCHEDULE_SIZE) return; // permanent chase

    int currentDuration;
    if (!m_isChaseMode) {
        currentDuration = MODE_SCHEDULE[m_modePhase][0]; // scatter duration
    } else {
        currentDuration = MODE_SCHEDULE[m_modePhase][1]; // chase duration
        if (currentDuration < 0) return; // permanent chase
    }

    if (m_modeTimer >= currentDuration) {
        m_modeTimer = 0;

        if (m_isChaseMode) {
            m_modePhase++; // advance to next scatter/chase pair
            m_isChaseMode = false;
        } else {
            m_isChaseMode = true;
        }

        // Apply mode to all ghosts
        GhostMode newMode = m_isChaseMode ? GhostMode::CHASE : GhostMode::SCATTER;
        for (auto& ghost : m_ghosts) {
            if (ghost.getMode() != GhostMode::FRIGHTENED &&
                ghost.getMode() != GhostMode::EATEN) {
                ghost.setMode(newMode);
            }
        }
    }
}

void Game::eatPellet(TileType pelletType) {
    GridPos pPos = m_player.getPos();
    m_maze.eatPellet(pPos.row, pPos.col);

    if (pelletType == TileType::PELLET) {
        m_score += 10;
    } else if (pelletType == TileType::POWER) {
        m_score += 50;
        m_ghostEatCombo = 0; // reset combo for new power pellet

        // Frighten all ghosts
        for (auto& ghost : m_ghosts) {
            ghost.setMode(GhostMode::FRIGHTENED);
        }
    }
}

void Game::nextLevel() {
    m_level++;
    m_rlVisitCount = {};   // new level = fresh exploration slate
    startLevel();
}

void Game::playerDied() {
    m_lives--;

    if (m_lives <= 0) {
        m_state = GameState::GAME_OVER;
        m_player.die();
    }
    // Respawn (when lives remain) is deferred to update() so the death
    // position is captured in player_after before the player teleports.
}

bool Game::initLogging() {
    try {
        std::filesystem::create_directories("../logs");
    } catch (const std::exception& ex) {
        std::cerr << "Failed to create logs directory: " << ex.what() << std::endl;
        m_loggingEnabled = false;
        return false;
    }

    m_logPath = "../logs/imitation_" + timestampForFilename() + ".jsonl";
    m_logFile.open(m_logPath, std::ios::out | std::ios::trunc);
    if (!m_logFile.is_open()) {
        std::cerr << "Failed to open gameplay log file: " << m_logPath << std::endl;
        m_loggingEnabled = false;
        return false;
    }

    m_loggingEnabled = true;
    m_loggedFrameCount = 0;

    m_logFile
        << "{\"record_type\":\"session_start\""
        << ",\"log_version\":1"
        << ",\"fps\":" << FPS
        << ",\"maze_rows\":" << m_maze.getRows()
        << ",\"maze_cols\":" << m_maze.getCols()
        << ",\"bonus_fruit_supported\":false"
        << ",\"power_pellets_logged_as_big_items\":true"
        << "}\n";

    std::cerr << "Gameplay logging enabled: " << m_logPath << std::endl;
    return true;
}

void Game::closeLogging() {
    if (!m_logFile.is_open()) return;

    m_logFile
        << "{\"record_type\":\"session_end\""
        << ",\"frames_logged\":" << m_loggedFrameCount
        << "}\n";
    m_logFile.close();
    m_loggingEnabled = false;
}

void Game::resetFrameEvents() {
    m_frameEvents = FrameEvents{};
}

Game::FrameObservation Game::captureObservation() const {
    FrameObservation observation;
    observation.tickMs = static_cast<uint64_t>(SDL_GetTicks());
    observation.frameNumber = m_loggedFrameCount;
    observation.state = m_state;
    observation.score = m_score;
    observation.lives = m_lives;
    observation.level = m_level;
    observation.remainingPellets = m_maze.getRemainingPellets();
    observation.ghostEatCombo = m_ghostEatCombo;
    observation.modeTimer = m_modeTimer;
    observation.modePhase = m_modePhase;
    observation.modeStep = getGhostModeStep();
    observation.isChaseMode = m_isChaseMode;
    observation.playerPos = m_player.getPos();
    observation.playerPixelX = m_player.getPixelX();
    observation.playerPixelY = m_player.getPixelY();
    observation.playerDirection = m_player.getDirection();
    observation.bufferedDirection = m_player.getBufferedDirection();
    observation.targetAction = getTargetAction();
    observation.playerTile = m_maze.getTile(observation.playerPos.row, observation.playerPos.col);
    observation.playerAlive = m_player.isAlive();
    observation.playerAlignedToGrid = m_player.isGridAligned();
    observation.powerPelletPositions = collectTilePositions(TileType::POWER);

    for (std::size_t i = 0; i < m_ghosts.size(); ++i) {
        const auto& ghost = m_ghosts[i];
        auto& loggedGhost = observation.ghosts[i];
        loggedGhost.id = ghost.getID();
        loggedGhost.pos = ghost.getPos();
        loggedGhost.pixelX = ghost.getPixelX();
        loggedGhost.pixelY = ghost.getPixelY();
        loggedGhost.direction = ghost.getDirection();
        loggedGhost.mode = ghost.getMode();
        loggedGhost.inHouse = ghost.isInHouse();
        loggedGhost.manhattanDistanceToPlayer = manhattanDistance(ghost.getPos(), observation.playerPos);
    }

    return observation;
}

Direction Game::getTargetAction() const {
    const Direction bufferedDirection = m_player.getBufferedDirection();
    if (bufferedDirection != Direction::NONE) {
        return bufferedDirection;
    }

    return m_player.getDirection();
}

int Game::getGhostModeStep() const {
    return m_modePhase * 2 + (m_isChaseMode ? 1 : 0);
}

std::vector<GridPos> Game::collectTilePositions(TileType tileType) const {
    std::vector<GridPos> positions;
    const auto& grid = m_maze.getGrid();

    for (int row = 0; row < static_cast<int>(grid.size()); ++row) {
        for (int col = 0; col < static_cast<int>(grid[row].size()); ++col) {
            if (grid[row][col] == tileType) {
                positions.push_back({row, col});
            }
        }
    }

    return positions;
}

void Game::logFrame(const FrameObservation& observation, const FrameEvents& events) {
    if (!m_loggingEnabled || !m_logFile.is_open()) return;

    m_logFile
        << "{\"record_type\":\"frame\""
        << ",\"frame\":" << observation.frameNumber
        << ",\"tick_ms\":" << observation.tickMs
        << ",\"state_before\":\"" << gameStateToString(observation.state) << "\""
        << ",\"state_after\":\"" << gameStateToString(m_state) << "\""
        << ",\"score_before\":" << observation.score
        << ",\"score_after\":" << m_score
        << ",\"lives_before\":" << observation.lives
        << ",\"lives_after\":" << m_lives
        << ",\"level_before\":" << observation.level
        << ",\"level_after\":" << m_level
        << ",\"remaining_pellets_before\":" << observation.remainingPellets
        << ",\"remaining_pellets_after\":" << m_maze.getRemainingPellets()
        << ",\"ghost_eat_combo_before\":" << observation.ghostEatCombo
        << ",\"ghost_eat_combo_after\":" << m_ghostEatCombo
        << ",\"ghost_mode_timer_before\":" << observation.modeTimer
        << ",\"ghost_mode_timer_after\":" << m_modeTimer
        << ",\"ghost_mode_phase_before\":" << observation.modePhase
        << ",\"ghost_mode_phase_after\":" << m_modePhase
        << ",\"ghost_mode_step_before\":" << observation.modeStep
        << ",\"ghost_mode_step_after\":" << getGhostModeStep()
        << ",\"is_chase_mode_before\":" << (observation.isChaseMode ? "true" : "false")
        << ",\"is_chase_mode_after\":" << (m_isChaseMode ? "true" : "false")
        << ",\"input\":{\"keypresses\":[";

    for (std::size_t i = 0; i < events.keypresses.size(); ++i) {
        if (i > 0) m_logFile << ",";
        m_logFile << "\"" << escapeJson(events.keypresses[i]) << "\"";
    }

    m_logFile
        << "],\"target_action\":\"" << directionToString(observation.targetAction)
        << "\",\"buffered_direction\":\"" << directionToString(observation.bufferedDirection)
        << "\",\"current_direction\":\"" << directionToString(observation.playerDirection)
        << "\"}"
        << ",\"player_before\":{\"grid\":";
    writeGridPosJson(m_logFile, observation.playerPos);
    m_logFile
        << ",\"pixel\":{\"x\":" << observation.playerPixelX << ",\"y\":" << observation.playerPixelY << "}"
        << ",\"direction\":\"" << directionToString(observation.playerDirection)
        << "\",\"buffered_direction\":\"" << directionToString(observation.bufferedDirection)
        << "\",\"tile\":\"" << tileTypeToString(observation.playerTile)
        << "\",\"alive\":" << (observation.playerAlive ? "true" : "false")
        << ",\"aligned_to_grid\":" << (observation.playerAlignedToGrid ? "true" : "false")
        << "}"
        << ",\"player_after\":{\"grid\":";
    writeGridPosJson(m_logFile, m_player.getPos());
    m_logFile
        << ",\"pixel\":{\"x\":" << m_player.getPixelX() << ",\"y\":" << m_player.getPixelY() << "}"
        << ",\"direction\":\"" << directionToString(m_player.getDirection())
        << "\",\"buffered_direction\":\"" << directionToString(m_player.getBufferedDirection())
        << "\",\"tile\":\"" << tileTypeToString(m_maze.getTile(m_player.getPos().row, m_player.getPos().col))
        << "\",\"alive\":" << (m_player.isAlive() ? "true" : "false")
        << ",\"aligned_to_grid\":" << (m_player.isGridAligned() ? "true" : "false")
        << "}"
        << ",\"power_pellet_positions_before\":";
    writeGridPosListJson(m_logFile, observation.powerPelletPositions);
    m_logFile
        << ",\"power_pellet_positions_after\":";
    writeGridPosListJson(m_logFile, collectTilePositions(TileType::POWER));
    m_logFile
        << ",\"bonus_fruit_position\":null"
        << ",\"ghosts_before\":[";

    for (std::size_t i = 0; i < observation.ghosts.size(); ++i) {
        if (i > 0) m_logFile << ",";
        const auto& ghost = observation.ghosts[i];
        m_logFile
            << "{\"id\":\"" << ghostIdToString(ghost.id)
            << "\",\"grid\":";
        writeGridPosJson(m_logFile, ghost.pos);
        m_logFile
            << ",\"pixel\":{\"x\":" << ghost.pixelX << ",\"y\":" << ghost.pixelY << "}"
            << ",\"direction\":\"" << directionToString(ghost.direction)
            << "\",\"mode\":\"" << ghostModeToString(ghost.mode)
            << "\",\"in_house\":" << (ghost.inHouse ? "true" : "false")
            << ",\"manhattan_distance_to_player\":" << ghost.manhattanDistanceToPlayer
            << "}";
    }

    m_logFile << "],\"ghosts_after\":[";
    for (std::size_t i = 0; i < m_ghosts.size(); ++i) {
        if (i > 0) m_logFile << ",";
        const auto& ghost = m_ghosts[i];
        m_logFile
            << "{\"id\":\"" << ghostIdToString(ghost.getID())
            << "\",\"grid\":";
        writeGridPosJson(m_logFile, ghost.getPos());
        m_logFile
            << ",\"pixel\":{\"x\":" << ghost.getPixelX() << ",\"y\":" << ghost.getPixelY() << "}"
            << ",\"direction\":\"" << directionToString(ghost.getDirection())
            << "\",\"mode\":\"" << ghostModeToString(ghost.getMode())
            << "\",\"in_house\":" << (ghost.isInHouse() ? "true" : "false")
            << ",\"manhattan_distance_to_player\":" << manhattanDistance(ghost.getPos(), m_player.getPos())
            << "}";
    }

    m_logFile
        << "],\"events\":{\"ate_pellet\":" << (events.atePellet ? "true" : "false")
        << ",\"ate_power_pellet\":" << (events.atePowerPellet ? "true" : "false")
        << ",\"player_died\":" << (events.playerDied ? "true" : "false")
        << ",\"level_cleared\":" << (events.levelCleared ? "true" : "false")
        << ",\"game_over\":" << (events.gameOver ? "true" : "false")
        << ",\"ghosts_eaten\":[";

    for (std::size_t i = 0; i < events.ghostsEaten.size(); ++i) {
        if (i > 0) m_logFile << ",";
        m_logFile << (events.ghostsEaten[i] ? "true" : "false");
    }

    m_logFile << "]}}\n";

    m_loggedFrameCount++;
    if (m_loggedFrameCount % LOG_FLUSH_INTERVAL == 0) {
        m_logFile.flush();
    }
}

// ============================================================
// RL interface
// ============================================================

void Game::runRL() {
    // Send initial state so Python can call reset() → first obs
    writeRLStep(buildStateVector(), 0.0f, false);

    Direction prevMoveDir = Direction::NONE; // tracks last actual movement direction
    bool prevHitWall     = false;           // was last frame a wall hit?
    int  wallHitStreak   = 0;               // consecutive wall-hit frames (for escalating stall penalty)

    // Repeat-state detection: circular buffer of last 8 grid positions.
    // If ≤3 unique tiles in the window the agent is looping — apply penalty.
    constexpr int REPEAT_WINDOW = 8;
    int recentRow[REPEAT_WINDOW]{};
    int recentCol[REPEAT_WINDOW]{};
    int recentIdx  = 0;
    int recentFill = 0;

    while (m_running) {
        std::string line;
        if (!std::getline(std::cin, line)) break;  // Python process closed

        // {"reset":true} → restart episode
        if (line.find("\"reset\"") != std::string::npos) {
            m_score = 0;
            m_lives = 3;
            m_level = m_startLevel;
            m_rlVisitCount = {};   // clear exploration counts each episode
            prevMoveDir    = Direction::NONE;  // prevent cross-episode reversal penalty
            prevHitWall    = false;            // prevent cross-episode wall-escape reward
            wallHitStreak  = 0;               // prevent cross-episode stall penalty
            recentIdx      = 0;
            recentFill     = 0;
            startLevel();
            writeRLStep(buildStateVector(), 0.0f, false);
            continue;
        }

        // {"action":N}  N = 0:UP  1:DOWN  2:LEFT  3:RIGHT
        int action = 0;
        const auto pos = line.find("\"action\"");
        if (pos != std::string::npos) {
            const auto colon = line.find(':', pos);
            if (colon != std::string::npos) {
                action = std::stoi(line.substr(colon + 1));
            }
        }
        m_rlAction = static_cast<Direction>(action);

        const int scoreBefore = m_score;
        const Uint32 frameStart = m_rlRender ? SDL_GetTicks() : 0;

        // --- Snapshot pixel state before update for reward shaping ---
        const int pxBefore = m_player.getPixelX();
        const int pyBefore = m_player.getPixelY();

        // Nearest dangerous ghost pixel distance (ignore frightened / eaten / in-house)
        auto dangerPixDist = [&](int px, int py) {
            int d = 9999;
            for (const auto& g : m_ghosts) {
                if (g.isInHouse()) continue;
                if (g.getMode() == GhostMode::FRIGHTENED) continue;
                if (g.getMode() == GhostMode::EATEN)      continue;
                int md = std::abs(g.getPixelX() - px) + std::abs(g.getPixelY() - py);
                d = std::min(d, md);
            }
            return d;
        };

        // Nearest pellet pixel distance (tile centre = col*TILE_SIZE + TILE_SIZE/2)
        auto pelletPixDist = [&](int px, int py) {
            int d = 999999;
            for (int r = 0; r < MAZE_ROWS; ++r)
                for (int c = 0; c < MAZE_COLS; ++c) {
                    TileType t = m_maze.getTile(r, c);
                    if (t == TileType::PELLET || t == TileType::POWER) {
                        int tx = c * TILE_SIZE + TILE_SIZE / 2;
                        int ty = r * TILE_SIZE + TILE_SIZE / 2;
                        d = std::min(d, std::abs(tx - px) + std::abs(ty - py));
                    }
                }
            return d;
        };

        const int pelletDistBefore = pelletPixDist(pxBefore, pyBefore);

        processInput();  // applies m_rlAction to player
        update();

        if (m_rlRender) {
            render();
            // Cap to ~60 fps so the human can actually watch
            const int elapsed = static_cast<int>(SDL_GetTicks() - frameStart);
            if (FRAME_DELAY > elapsed) SDL_Delay(FRAME_DELAY - elapsed);
        }

        // --- Post-update reward shaping ---
        const int pxAfter = m_player.getPixelX();
        const int pyAfter = m_player.getPixelY();

        // Wall hit: pixel position unchanged means player couldn't move
        const bool hitWall = (pxBefore == pxAfter && pyBefore == pyAfter);
        wallHitStreak = hitWall ? wallHitStreak + 1 : 0;

        const int pelletDistAfter = pelletPixDist(pxAfter, pyAfter);

        // ================================================================
        // Reward function — layered + potential-based shaping + mode-aware
        // ================================================================

        // --- 1. Base: score delta × 2  -----------------------------------
        // Pellet 10 pts → +2.0;  ghost combo 200/400/800/1600 → +40/+80/+160/+320.
        // Multiplying by 2 (vs old ×1) so pellet reward matches R_FOOD=+2 from the
        // design spec, and ghost-combo rewards stay proportionally large.
        const float scoreDelta = static_cast<float>(m_score - scoreBefore) / 10.0f;
        float reward = scoreDelta * 2.0f;

        // --- 2. Terminal rewards  ----------------------------------------
        // R_DIE = -500: death must be very painful so the agent prefers almost any
        //   non-dying action, including approaching ghosts only when safe.
        // R_WIN = +500: explicit, large signal to finish all pellets.
        if (m_frameEvents.playerDied)   reward -= 500.0f;
        if (m_frameEvents.levelCleared) reward += 500.0f;

        // All dense shaping below uses pre-/post-action pixel positions.
        // On a death frame update() immediately respawns Pac-Man, so pxAfter/pyAfter
        // reflect the spawn point — unrelated to the action taken.  Skip all shaping
        // on death to avoid corrupting credit assignment.
        if (!m_frameEvents.playerDied) {

        // --- 3. Per-step penalty  ----------------------------------------
        // -0.1/step: discourages looping and idling; makes earlier pellet/ghost
        // collection strictly more valuable than later.
        reward -= 0.1f;

        // --- 4. Wall-hit penalties  --------------------------------------
        // -1.0 per wall hit (direct, much clearer than step penalty alone).
        if (hitWall) reward -= 1.0f;

        // Escalating stall: after 8 consecutive wall frames add extra pressure
        // so the policy is forced to try a different direction. Capped at -2.0.
        if (wallHitStreak > 8)
            reward -= std::min(0.2f * static_cast<float>(wallHitStreak - 8), 2.0f);

        // Small reward for escaping a stall sequence.
        if (prevHitWall && !hitWall) reward += 0.3f;
        prevHitWall = hitWall;

        // --- 5. Repeat-state penalty  ------------------------------------
        // Maintain a rolling window of the last REPEAT_WINDOW grid positions.
        // If ≤3 unique tiles appear the agent is A-B-A-B cycling or spinning
        // in a tiny loop — penalise to break the habit.
        const GridPos pGrid = m_player.getPos();
        recentRow[recentIdx % REPEAT_WINDOW] = pGrid.row;
        recentCol[recentIdx % REPEAT_WINDOW] = pGrid.col;
        ++recentIdx;
        recentFill = std::min(recentFill + 1, REPEAT_WINDOW);
        if (recentFill == REPEAT_WINDOW) {
            int uniqueTiles = 0;
            for (int i = 0; i < REPEAT_WINDOW; ++i) {
                bool dup = false;
                for (int j = 0; j < i && !dup; ++j)
                    dup = (recentRow[i] == recentRow[j] && recentCol[i] == recentCol[j]);
                if (!dup) ++uniqueTiles;
            }
            if (uniqueTiles <= 3) reward -= 0.5f;
        }

        // --- 6. Direction-reversal penalty  ------------------------------
        // Suppress when danger is close (reversing is correct to escape ghosts)
        // and when the previous frame was a wall-hit (reversing out of a dead end
        // is correct).
        const bool isDangerClose = (dangerPixDist(pxAfter, pyAfter) < 3 * TILE_SIZE);
        if (!isDangerClose && !prevHitWall && prevMoveDir != Direction::NONE) {
            const bool reversed =
                (prevMoveDir == Direction::UP    && m_rlAction == Direction::DOWN)  ||
                (prevMoveDir == Direction::DOWN   && m_rlAction == Direction::UP)   ||
                (prevMoveDir == Direction::LEFT   && m_rlAction == Direction::RIGHT) ||
                (prevMoveDir == Direction::RIGHT  && m_rlAction == Direction::LEFT);
            if (reversed) reward -= 0.2f;
        }
        if (m_player.getDirection() != Direction::NONE)
            prevMoveDir = m_player.getDirection();

        // --- 7. Food-progress shaping  -----------------------------------
        // alpha * (d_food_prev − d_food_now) / TILE_SIZE
        // alpha = 0.5: moving one tile toward the nearest pellet → +0.5;
        // moving away → −0.5.  Continuous signal so the policy learns to
        // navigate toward targets without requiring the rare "eat" event.
        if (m_maze.getRemainingPellets() > 0 && pelletDistBefore < 999999) {
            reward += 0.5f * static_cast<float>(pelletDistBefore - pelletDistAfter)
                           / static_cast<float>(TILE_SIZE);
        }

        // --- 8. Mode-aware ghost shaping  --------------------------------
        // Normal mode   → exponential danger penalty  −β·exp(−d/τ)
        //   β = 1.5, τ = 2 tiles (48 px).  At ½-tile distance: ≈ −1.1/ghost;
        //   at 2 tiles: ≈ −0.55/ghost; at 4 tiles: ≈ −0.20/ghost.
        //   Smooth and extends to infinity — no hard radius needed.
        //
        // Frightened mode → pursuit shaping  +γ·(d_prev − d_now) / TILE_SIZE
        //   γ = 1.0: moving one tile toward a frightened ghost → +1.0.
        //   Mode SWITCH is critical: without it the agent keeps running away
        //   even after eating a power pellet.
        //
        // Power-pellet bonus: R_POWER = +8, +4 extra per ghost within 4 tiles
        //   at eat-time to reward strategic power-pellet usage.
        {
            constexpr float TAU  = 2.0f * TILE_SIZE;   // exponential decay constant
            constexpr float BETA = 1.5f;               // danger penalty scale
            constexpr float GAMMA_HUNT = 1.0f;         // pursuit shaping scale

            for (const auto& g : m_ghosts) {
                if (g.isInHouse()) continue;
                if (g.getMode() == GhostMode::EATEN) continue;

                const int gdAfterPx  = std::abs(g.getPixelX() - pxAfter)
                                     + std::abs(g.getPixelY() - pyAfter);

                if (g.getMode() == GhostMode::FRIGHTENED) {
                    // Pursuit shaping: reward getting closer to a frightened ghost.
                    const int gdBeforePx = std::abs(g.getPixelX() - pxBefore)
                                        + std::abs(g.getPixelY() - pyBefore);
                    reward += GAMMA_HUNT * static_cast<float>(gdBeforePx - gdAfterPx)
                                        / static_cast<float>(TILE_SIZE);
                } else {
                    // Exponential danger penalty.
                    reward -= BETA * std::exp(-static_cast<float>(gdAfterPx) / TAU);
                }
            }
        }

        // Power-pellet bonus: strategic tool to create a ghost-eating window.
        // R_POWER = +8 base, +4 per ghost within 4 tiles when pellet is eaten.
        // Intentionally smaller than ghost-eating rewards so the agent treats
        // power pellets as means to an end, not the goal itself.
        if (m_frameEvents.atePowerPellet) {
            int nearCount = 0;
            for (const auto& g : m_ghosts) {
                if (g.isInHouse()) continue;
                int d = std::abs(g.getPixelX() - pxAfter) + std::abs(g.getPixelY() - pyAfter);
                if (d < 4 * TILE_SIZE) ++nearCount;
            }
            reward += 8.0f + nearCount * 4.0f;
        }

        // --- 9. Exploration novelty  -------------------------------------
        // 1/√(1+v) pull toward unvisited tiles; decays slowly so the agent
        // keeps exploring the full maze rather than farming a safe pellet loop.
        {
            const int visits = m_rlVisitCount[pGrid.row][pGrid.col];
            reward += 0.05f / std::sqrt(static_cast<float>(1 + visits));
            m_rlVisitCount[pGrid.row][pGrid.col]++;
        }

        } // end if (!m_frameEvents.playerDied)

        // End the RL episode on life loss as well as true game termination.
        // PPO learns much more reliably when "death" is a terminal signal;
        // otherwise the negative death reward is followed by a respawn with
        // unrelated dynamics, which smears credit assignment across lives.
        const bool done = m_frameEvents.playerDied || m_frameEvents.gameOver || m_frameEvents.levelCleared;
        writeRLStep(buildStateVector(), reward, done);
    }
}

std::array<float, Game::RL_STATE_SIZE> Game::buildStateVector() const {
    std::array<float, RL_STATE_SIZE> state{};
    const GridPos pPos = m_player.getPos();
    int idx = 0;

    // [0-1]  player absolute position (normalised)
    state[idx++] = static_cast<float>(pPos.row) / 30.0f;
    state[idx++] = static_cast<float>(pPos.col) / 27.0f;

    // [2-9]  ghost absolute positions (row/30, col/27 each)
    for (const auto& ghost : m_ghosts) {
        const GridPos gPos = ghost.getPos();
        state[idx++] = static_cast<float>(gPos.row) / 30.0f;
        state[idx++] = static_cast<float>(gPos.col) / 27.0f;
    }

    // [10-13] per-ghost is_frightened
    for (const auto& ghost : m_ghosts) {
        state[idx++] = (ghost.getMode() == GhostMode::FRIGHTENED) ? 1.0f : 0.0f;
    }

    // [14-17] per-ghost frightened timer (classic max ≈ 360 ticks @ 60fps)
    constexpr float FRIGHTENED_TICKS = 360.0f;
    for (const auto& ghost : m_ghosts) {
        state[idx++] = static_cast<float>(ghost.getFrightenedTimer()) / FRIGHTENED_TICKS;
    }

    // [18-21] per-ghost is_in_house flag
    for (const auto& ghost : m_ghosts) {
        state[idx++] = ghost.isInHouse() ? 1.0f : 0.0f;
    }

    // [22] chase/scatter mode  [23] mode timer
    state[idx++] = m_isChaseMode ? 1.0f : 0.0f;
    state[idx++] = static_cast<float>(m_modeTimer) / 1200.0f;

    // [24] remaining pellets ratio  [25] ghost eat combo
    state[idx++] = static_cast<float>(m_maze.getRemainingPellets()) /
                   static_cast<float>(std::max(1, m_maze.getTotalPellets()));
    state[idx++] = static_cast<float>(m_ghostEatCombo) / 4.0f;

    // [26-27] nearest pellet absolute position (row/30, col/27)
    {
        int bestDist = INT_MAX;
        float npr = 0.0f, npc = 0.0f;
        for (int r = 0; r < MAZE_ROWS; ++r) {
            for (int c = 0; c < MAZE_COLS; ++c) {
                TileType t = m_maze.getTile(r, c);
                if (t == TileType::PELLET || t == TileType::POWER) {
                    int d = std::abs(r - pPos.row) + std::abs(c - pPos.col);
                    if (d < bestDist) { bestDist = d; npr = r / 30.0f; npc = c / 27.0f; }
                }
            }
        }
        state[idx++] = npr;
        state[idx++] = npc;
    }

    // [28-31] nearest pellet in each cardinal direction — scan along the corridor until
    //         a wall blocks the way.  Returns distance in tiles / 30; 1.0 = none found.
    {
        auto scanDir = [&](int dr, int dc) -> float {
            for (int step = 1; step <= 30; ++step) {
                const int r = pPos.row + dr * step;
                const int c = pPos.col + dc * step;
                if (r < 0 || r >= MAZE_ROWS || c < 0 || c >= MAZE_COLS) break;
                TileType t = m_maze.getTile(r, c);
                if (t == TileType::WALL) break;
                if (t == TileType::PELLET || t == TileType::POWER)
                    return static_cast<float>(step) / 30.0f;
            }
            return 1.0f;
        };
        state[idx++] = scanDir(-1,  0);  // North
        state[idx++] = scanDir( 1,  0);  // South
        state[idx++] = scanDir( 0, -1);  // West
        state[idx++] = scanDir( 0,  1);  // East
    }

    // [32-39] power pellet positions — scan row-major for up to 4; pad with 0.0 if eaten
    {
        int ppCount = 0;
        for (int r = 0; r < MAZE_ROWS && ppCount < 4; ++r) {
            for (int c = 0; c < MAZE_COLS && ppCount < 4; ++c) {
                if (m_maze.getTile(r, c) == TileType::POWER) {
                    state[idx++] = static_cast<float>(r) / 30.0f;
                    state[idx++] = static_cast<float>(c) / 27.0f;
                    ++ppCount;
                }
            }
        }
        while (ppCount < 4) { state[idx++] = 0.0f; state[idx++] = 0.0f; ++ppCount; }
    }

    // [40-88] 7×7 local tile window centred on Pac-Man (normalised by 7.0)
    constexpr int RADIUS = 3;
    for (int dr = -RADIUS; dr <= RADIUS; ++dr) {
        for (int dc = -RADIUS; dc <= RADIUS; ++dc) {
            const int r = pPos.row + dr;
            const int c = pPos.col + dc;
            state[idx++] = (r >= 0 && r < MAZE_ROWS && c >= 0 && c < MAZE_COLS)
                           ? static_cast<float>(m_maze.getTile(r, c)) / 7.0f
                           : 0.0f;
        }
    }
    // idx == 89

    return state;
}

void Game::writeRLStep(const std::array<float, RL_STATE_SIZE>& state,
                       float reward, bool done) const {
    std::cout << "{\"state\":[";
    for (std::size_t i = 0; i < state.size(); ++i) {
        if (i > 0) std::cout << ',';
        std::cout << state[i];
    }
    std::cout << "],\"reward\":" << reward
              << ",\"done\":"    << (done  ? "true" : "false")
              << ",\"lives\":"   << m_lives
              << ",\"score\":"   << m_score
              << "}\n";
    std::cout.flush();
}
