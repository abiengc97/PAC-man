#include "Game.h"
#include "Common.h"
#include <SDL2/SDL.h>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
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

Game::Game()
    : m_ghosts{Ghost(GhostID::BLINKY), Ghost(GhostID::PINKY),
               Ghost(GhostID::INKY), Ghost(GhostID::CLYDE)}
{
    std::srand(static_cast<unsigned>(std::time(nullptr)));
}

Game::~Game() {
    closeLogging();
}

bool Game::init() {
    if (!m_renderer.init("Pac-Man", SCREEN_W, SCREEN_H)) {
        return false;
    }

    startLevel();

    if (m_maze.getTotalPellets() == 0) {
        std::cerr << "Failed to load maze!" << std::endl;
        return false;
    }

    initLogging();
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
    const bool shouldLogFrame = m_loggingEnabled && m_state == GameState::PLAYING;
    FrameObservation observation;
    if (shouldLogFrame) {
        observation = captureObservation();
    }

    if (m_state != GameState::PLAYING) return;

    // Ready countdown at level start
    if (m_readyTimer > 0) {
        m_readyTimer--;
        if (shouldLogFrame) {
            logFrame(observation, m_frameEvents);
        }
        return;
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

    // Check level clear
    if (m_maze.getRemainingPellets() == 0) {
        m_frameEvents.levelCleared = true;
        m_state = GameState::LEVEL_CLEAR;
        nextLevel();
    }

    if (shouldLogFrame) {
        logFrame(observation, m_frameEvents);
    }
}

void Game::render() {
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
    startLevel();
}

void Game::playerDied() {
    m_lives--;

    if (m_lives <= 0) {
        m_state = GameState::GAME_OVER;
        m_player.die();
    } else {
        // Respawn player and ghosts, keep score and pellets
        m_player.respawn(m_maze.getPacManSpawn());
        for (auto& ghost : m_ghosts) {
            ghost.init(m_maze);
        }
        m_readyTimer = READY_DURATION;
        m_modeTimer  = 0;
        m_modePhase  = 0;
        m_isChaseMode = false;
    }
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

    std::cout << "Gameplay logging enabled: " << m_logPath << std::endl;
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
    observation.readyTimer = m_readyTimer;
    observation.remainingPellets = m_maze.getRemainingPellets();
    observation.ghostEatCombo = m_ghostEatCombo;
    observation.modeTimer = m_modeTimer;
    observation.modePhase = m_modePhase;
    observation.isChaseMode = m_isChaseMode;
    observation.playerPos = m_player.getPos();
    observation.playerPixelX = m_player.getPixelX();
    observation.playerPixelY = m_player.getPixelY();
    observation.playerDirection = m_player.getDirection();
    observation.bufferedDirection = m_player.getBufferedDirection();
    observation.effectiveAction = getEffectiveAction();
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

Direction Game::getEffectiveAction() const {
    const Direction bufferedDirection = m_player.getBufferedDirection();
    if (bufferedDirection != Direction::NONE) {
        return bufferedDirection;
    }

    return m_player.getDirection();
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
        << ",\"ready_timer_before\":" << observation.readyTimer
        << ",\"ready_timer_after\":" << m_readyTimer
        << ",\"remaining_pellets_before\":" << observation.remainingPellets
        << ",\"remaining_pellets_after\":" << m_maze.getRemainingPellets()
        << ",\"ghost_eat_combo_before\":" << observation.ghostEatCombo
        << ",\"ghost_eat_combo_after\":" << m_ghostEatCombo
        << ",\"ghost_mode_timer_before\":" << observation.modeTimer
        << ",\"ghost_mode_timer_after\":" << m_modeTimer
        << ",\"ghost_mode_phase_before\":" << observation.modePhase
        << ",\"ghost_mode_phase_after\":" << m_modePhase
        << ",\"is_chase_mode_before\":" << (observation.isChaseMode ? "true" : "false")
        << ",\"is_chase_mode_after\":" << (m_isChaseMode ? "true" : "false")
        << ",\"input\":{\"keypresses\":[";

    for (std::size_t i = 0; i < events.keypresses.size(); ++i) {
        if (i > 0) m_logFile << ",";
        m_logFile << "\"" << escapeJson(events.keypresses[i]) << "\"";
    }

    m_logFile
        << "],\"effective_action\":\"" << directionToString(observation.effectiveAction)
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
