#include "Game.h"
#include <SDL2/SDL.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>

constexpr int Game::MODE_SCHEDULE[][2];

Game::Game()
    : m_ghosts{Ghost(GhostID::BLINKY), Ghost(GhostID::PINKY),
               Ghost(GhostID::INKY), Ghost(GhostID::CLYDE)}
{
    std::srand(static_cast<unsigned>(std::time(nullptr)));
}

Game::~Game() {}

bool Game::init() {
    if (!m_renderer.init("Pac-Man", SCREEN_W, SCREEN_H)) {
        return false;
    }

    if (!m_maze.loadFromFile("levels/level1.txt")) {
        std::cerr << "Failed to load maze!" << std::endl;
        return false;
    }

    startLevel();
    return true;
}

void Game::startLevel() {
    m_maze.resetPellets();
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
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) {
            m_running = false;
            return;
        }

        if (event.type == SDL_KEYDOWN) {
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
    if (m_state != GameState::PLAYING) return;

    // Ready countdown at level start
    if (m_readyTimer > 0) {
        m_readyTimer--;
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
        eatPellet(tile);
    }

    // Update ghosts
    for (auto& ghost : m_ghosts) {
        // Pass blinky reference to inky for targeting
        const Ghost* blinky = &m_ghosts[static_cast<int>(GhostID::BLINKY)];
        ghost.update(m_maze, m_player, blinky);
    }

    // Check collisions
    checkCollisions();

    // Check level clear
    if (m_maze.getRemainingPellets() == 0) {
        m_state = GameState::LEVEL_CLEAR;
        nextLevel();
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

    if (m_state == GameState::GAME_OVER) {
        m_renderer.drawGameOver(m_score);
    }

    m_renderer.present();
}

void Game::checkCollisions() {
    GridPos pPos = m_player.getPos();

    for (auto& ghost : m_ghosts) {
        if (ghost.isInHouse()) continue;
        if (ghost.getPos() != pPos) continue;

        switch (ghost.getMode()) {
            case GhostMode::FRIGHTENED:
                // Eat the ghost
                ghost.setEaten();
                m_ghostEatCombo++;
                m_score += 200 * static_cast<int>(std::pow(2, m_ghostEatCombo - 1));
                break;

            case GhostMode::EATEN:
                // Ghost is already eaten, no collision
                break;

            default:
                // Ghost kills Pac-Man
                playerDied();
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
