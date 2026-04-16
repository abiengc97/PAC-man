#include "Renderer.h"
#include "Common.h"
#include "Maze.h"
#include "Player.h"
#include "Ghost.h"
#include <iostream>
#include <cmath>

// ============================================================
// Simple 5x7 bitmap font for score/text display
// Each char is 5 columns, 7 rows, packed into uint8_t[7]
// ============================================================
static const uint8_t FONT_0[] = {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E};
static const uint8_t FONT_1[] = {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E};
static const uint8_t FONT_2[] = {0x0E,0x11,0x01,0x06,0x08,0x10,0x1F};
static const uint8_t FONT_3[] = {0x0E,0x11,0x01,0x06,0x01,0x11,0x0E};
static const uint8_t FONT_4[] = {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02};
static const uint8_t FONT_5[] = {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E};
static const uint8_t FONT_6[] = {0x06,0x08,0x10,0x1E,0x11,0x11,0x0E};
static const uint8_t FONT_7[] = {0x1F,0x01,0x02,0x04,0x08,0x08,0x08};
static const uint8_t FONT_8[] = {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E};
static const uint8_t FONT_9[] = {0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C};

static const uint8_t FONT_A[] = {0x0E,0x11,0x11,0x1F,0x11,0x11,0x11};
static const uint8_t FONT_B[] = {0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E};
static const uint8_t FONT_C[] = {0x0E,0x11,0x10,0x10,0x10,0x11,0x0E};
static const uint8_t FONT_D[] = {0x1C,0x12,0x11,0x11,0x11,0x12,0x1C};
static const uint8_t FONT_E[] = {0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F};
static const uint8_t FONT_F[] = {0x1F,0x10,0x10,0x1E,0x10,0x10,0x10};
static const uint8_t FONT_G[] = {0x0E,0x11,0x10,0x17,0x11,0x11,0x0F};
static const uint8_t FONT_H[] = {0x11,0x11,0x11,0x1F,0x11,0x11,0x11};
static const uint8_t FONT_I[] = {0x0E,0x04,0x04,0x04,0x04,0x04,0x0E};
static const uint8_t FONT_J[] = {0x0F,0x02,0x02,0x02,0x02,0x12,0x0C};
static const uint8_t FONT_K[] = {0x11,0x12,0x14,0x18,0x14,0x12,0x11};
static const uint8_t FONT_L[] = {0x10,0x10,0x10,0x10,0x10,0x10,0x1F};
static const uint8_t FONT_M[] = {0x11,0x1B,0x15,0x15,0x11,0x11,0x11};
static const uint8_t FONT_N[] = {0x11,0x19,0x15,0x13,0x11,0x11,0x11};
static const uint8_t FONT_O[] = {0x0E,0x11,0x11,0x11,0x11,0x11,0x0E};
static const uint8_t FONT_P[] = {0x1E,0x11,0x11,0x1E,0x10,0x10,0x10};
static const uint8_t FONT_Q[] = {0x0E,0x11,0x11,0x11,0x15,0x12,0x0D};
static const uint8_t FONT_R[] = {0x1E,0x11,0x11,0x1E,0x14,0x12,0x11};
static const uint8_t FONT_S[] = {0x0E,0x11,0x10,0x0E,0x01,0x11,0x0E};
static const uint8_t FONT_T[] = {0x1F,0x04,0x04,0x04,0x04,0x04,0x04};
static const uint8_t FONT_U[] = {0x11,0x11,0x11,0x11,0x11,0x11,0x0E};
static const uint8_t FONT_V[] = {0x11,0x11,0x11,0x11,0x11,0x0A,0x04};
static const uint8_t FONT_W[] = {0x11,0x11,0x11,0x15,0x15,0x1B,0x11};
static const uint8_t FONT_X[] = {0x11,0x11,0x0A,0x04,0x0A,0x11,0x11};
static const uint8_t FONT_Y[] = {0x11,0x11,0x0A,0x04,0x04,0x04,0x04};
static const uint8_t FONT_Z[] = {0x1F,0x01,0x02,0x04,0x08,0x10,0x1F};
static const uint8_t FONT_SPACE[] = {0x00,0x00,0x00,0x00,0x00,0x00,0x00};
static const uint8_t FONT_COLON[] = {0x00,0x04,0x04,0x00,0x04,0x04,0x00};

static const uint8_t* getGlyph(char c) {
    switch(c) {
        case '0': return FONT_0; case '1': return FONT_1; case '2': return FONT_2;
        case '3': return FONT_3; case '4': return FONT_4; case '5': return FONT_5;
        case '6': return FONT_6; case '7': return FONT_7; case '8': return FONT_8;
        case '9': return FONT_9;
        case 'A': case 'a': return FONT_A;
        case 'B': case 'b': return FONT_B;
        case 'C': case 'c': return FONT_C;
        case 'D': case 'd': return FONT_D;
        case 'E': case 'e': return FONT_E;
        case 'F': case 'f': return FONT_F;
        case 'G': case 'g': return FONT_G;
        case 'H': case 'h': return FONT_H;
        case 'I': case 'i': return FONT_I;
        case 'J': case 'j': return FONT_J;
        case 'K': case 'k': return FONT_K;
        case 'L': case 'l': return FONT_L;
        case 'M': case 'm': return FONT_M;
        case 'N': case 'n': return FONT_N;
        case 'O': case 'o': return FONT_O;
        case 'P': case 'p': return FONT_P;
        case 'Q': case 'q': return FONT_Q;
        case 'R': case 'r': return FONT_R;
        case 'S': case 's': return FONT_S;
        case 'T': case 't': return FONT_T;
        case 'U': case 'u': return FONT_U;
        case 'V': case 'v': return FONT_V;
        case 'W': case 'w': return FONT_W;
        case 'X': case 'x': return FONT_X;
        case 'Y': case 'y': return FONT_Y;
        case 'Z': case 'z': return FONT_Z;
        case ':': return FONT_COLON;
        default:  return FONT_SPACE;
    }
}

// ============================================================
// Renderer Implementation
// ============================================================

Renderer::Renderer() {}

Renderer::~Renderer() {
    shutdown();
}

bool Renderer::init(const std::string& title, int width, int height) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL_Init failed: " << SDL_GetError() << std::endl;
        return false;
    }

    m_window = SDL_CreateWindow(
        title.c_str(),
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        width, height,
        SDL_WINDOW_SHOWN
    );
    if (!m_window) {
        std::cerr << "SDL_CreateWindow failed: " << SDL_GetError() << std::endl;
        return false;
    }

    m_renderer = SDL_CreateRenderer(m_window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!m_renderer) {
        std::cerr << "SDL_CreateRenderer failed: " << SDL_GetError() << std::endl;
        return false;
    }

    SDL_SetRenderDrawBlendMode(m_renderer, SDL_BLENDMODE_BLEND);

    return true;
}

void Renderer::shutdown() {
    if (m_renderer) { SDL_DestroyRenderer(m_renderer); m_renderer = nullptr; }
    if (m_window)   { SDL_DestroyWindow(m_window);     m_window   = nullptr; }
    SDL_Quit();
}

void Renderer::clear() {
    SDL_SetRenderDrawColor(m_renderer, 0, 0, 0, 255);
    SDL_RenderClear(m_renderer);
}

void Renderer::present() {
    SDL_RenderPresent(m_renderer);
}

void Renderer::drawMaze(const Maze& maze) {
    const auto& grid = maze.getGrid();

    for (int r = 0; r < maze.getRows(); ++r) {
        for (int c = 0; c < maze.getCols(); ++c) {
            int x = c * TILE_SIZE;
            int y = r * TILE_SIZE;

            switch (grid[r][c]) {
                case TileType::WALL:
                    drawRect(x, y, TILE_SIZE, TILE_SIZE, {33, 33, 222, 255}); // blue walls
                    // Inner darker rectangle for depth
                    drawRect(x+2, y+2, TILE_SIZE-4, TILE_SIZE-4, {20, 20, 170, 255});
                    break;

                case TileType::PELLET:
                    drawCircle(x + TILE_SIZE/2, y + TILE_SIZE/2, 2, {255, 183, 174, 255}); // small dot
                    break;

                case TileType::POWER:
                    drawCircle(x + TILE_SIZE/2, y + TILE_SIZE/2, 6, {255, 183, 174, 255}); // big dot
                    break;

                case TileType::GHOST_DOOR:
                    drawRect(x, y + TILE_SIZE/2 - 2, TILE_SIZE, 4, {255, 183, 255, 255}); // pink door
                    break;

                default:
                    break; // empty, tunnel, ghost house — black background
            }
        }
    }
}

void Renderer::drawPlayer(const Player& player) {
    if (!player.isAlive()) return;

    int cx = player.getPixelX();
    int cy = player.getPixelY();
    int radius = TILE_SIZE / 2 - 2;

    // Yellow for Pac-Man
    SDL_Color yellow = {255, 255, 0, 255};

    // Draw mouth based on direction and animation
    SDL_Color black = {0, 0, 0, 255};
    int frame = player.getAnimFrame();
    int mouthSize = (frame % 2 == 0) ? M_PI / 3 : 0;

    Direction dir = player.getDirection();
    if (dir == Direction::NONE) dir = Direction::RIGHT;

    // Draw Pac-Man in the right direction
    switch (dir) {
        case Direction::RIGHT:
            drawCircle(cx, cy, radius, yellow, 0, mouthSize);
            break;
        case Direction::LEFT:
            drawCircle(cx, cy, radius, yellow, M_PI, mouthSize);
            break;
        case Direction::UP:
            drawCircle(cx, cy, radius, yellow, 3 * M_PI / 2, mouthSize);
            break;
        case Direction::DOWN:
            drawCircle(cx, cy, radius, yellow, M_PI / 2, mouthSize);
            break;
        default: break;
    }
}

void Renderer::drawGhost(const Ghost& ghost) {
    if (ghost.isInHouse() && ghost.getMode() != GhostMode::EATEN) {
        // Still in house — draw dimmed
    }

    int cx = ghost.getPixelX();
    int cy = ghost.getPixelY();
    int radius = TILE_SIZE / 2 - 2;

    SDL_Color color;
    if (ghost.getMode() == GhostMode::FRIGHTENED) {
        color = {33, 33, 255, 255}; // dark blue when frightened
    } else if (ghost.getMode() == GhostMode::EATEN) {
        color = {200, 200, 200, 80}; // faint — just eyes
    } else {
        color = getGhostColor(ghost.getID());
    }

    // Ghost body: circle top + rectangle bottom
    drawCircle(cx, cy - 2, radius, color);
    drawRect(cx - radius, cy - 2, radius * 2 + 1, radius, color);

    // Wavy bottom edge
    for (int i = 0; i < 4; i++) {
        int bx = cx - radius + i * (radius * 2 / 3);
        drawRect(bx, cy + radius - 2, radius * 2 / 3 / 2, 3, color);
    }

    // Eyes
    SDL_Color white = {255, 255, 255, 255};
    SDL_Color eyeBlue = {33, 33, 222, 255};

    // Left eye
    drawCircle(cx - 4, cy - 3, 3, white);
    // Right eye
    drawCircle(cx + 4, cy - 3, 3, white);

    // Pupils — shift based on direction
    int pdx = 0, pdy = 0;
    switch (ghost.getDirection()) {
        case Direction::LEFT:  pdx = -1; break;
        case Direction::RIGHT: pdx =  1; break;
        case Direction::UP:    pdy = -1; break;
        case Direction::DOWN:  pdy =  1; break;
        default: break;
    }
    drawCircle(cx - 4 + pdx, cy - 3 + pdy, 1, eyeBlue);
    drawCircle(cx + 4 + pdx, cy - 3 + pdy, 1, eyeBlue);
}

void Renderer::drawBonusFruit(GridPos pos, FruitType type) {
    const int x = pos.col * TILE_SIZE;
    const int y = pos.row * TILE_SIZE;
    const int cx = x + TILE_SIZE / 2;
    const int cy = y + TILE_SIZE / 2 + 1;

    const SDL_Color cherryRed   = {222, 30, 68, 255};
    const SDL_Color berryRed    = {240, 55, 85, 255};
    const SDL_Color appleRed    = {210, 40, 40, 255};
    const SDL_Color orange      = {255, 153, 0, 255};
    const SDL_Color melonGreen  = {90, 200, 110, 255};
    const SDL_Color darkGreen   = {40, 150, 70, 255};
    const SDL_Color leafGreen   = {60, 190, 90, 255};
    const SDL_Color yellow      = {255, 220, 40, 255};
    const SDL_Color gold        = {255, 200, 40, 255};
    const SDL_Color blue        = {70, 180, 255, 255};
    const SDL_Color white       = {255, 255, 255, 255};
    const SDL_Color black       = {0, 0, 0, 255};

    auto stripe = [&](int dy, int width, SDL_Color color) {
        drawRect(cx - width / 2, cy + dy, width, 2, color);
    };

    switch (type) {
        case FruitType::CHERRY:
            drawCircle(cx - 4, cy + 2, 5, cherryRed);
            drawCircle(cx + 4, cy + 2, 5, cherryRed);
            drawRect(cx - 2, cy - 8, 2, 7, leafGreen);
            drawRect(cx + 2, cy - 8, 2, 7, leafGreen);
            drawRect(cx - 2, cy - 8, 6, 2, leafGreen);
            drawCircle(cx - 6, cy, 1, white);
            drawCircle(cx + 2, cy, 1, white);
            break;

        case FruitType::STRAWBERRY:
            drawRect(cx - 3, cy - 7, 6, 3, leafGreen);
            drawRect(cx - 5, cy - 4, 10, 3, leafGreen);
            stripe(-3, 8, berryRed);
            stripe(-1, 10, berryRed);
            stripe(1, 12, berryRed);
            stripe(3, 10, berryRed);
            stripe(5, 8, berryRed);
            drawRect(cx - 3, cy + 7, 6, 2, berryRed);
            drawCircle(cx - 3, cy - 1, 1, yellow);
            drawCircle(cx + 1, cy - 2, 1, yellow);
            drawCircle(cx - 1, cy + 2, 1, yellow);
            drawCircle(cx + 3, cy + 1, 1, yellow);
            drawCircle(cx - 4, cy + 4, 1, yellow);
            break;

        case FruitType::ORANGE:
            drawCircle(cx, cy + 1, 7, orange);
            drawRect(cx - 1, cy - 9, 2, 4, darkGreen);
            drawRect(cx + 1, cy - 8, 4, 2, leafGreen);
            drawCircle(cx - 3, cy - 1, 1, white);
            break;

        case FruitType::APPLE:
            drawCircle(cx - 3, cy + 1, 6, appleRed);
            drawCircle(cx + 3, cy + 1, 6, appleRed);
            drawRect(cx - 5, cy + 4, 10, 4, appleRed);
            drawRect(cx - 1, cy - 9, 2, 5, darkGreen);
            drawRect(cx + 1, cy - 8, 4, 2, leafGreen);
            drawCircle(cx - 4, cy - 1, 1, white);
            break;

        case FruitType::MELON:
            drawRect(cx - 1, cy - 8, 2, 4, darkGreen);
            drawCircle(cx, cy + 1, 8, melonGreen);
            stripe(-4, 10, darkGreen);
            stripe(0, 12, darkGreen);
            stripe(4, 10, darkGreen);
            drawCircle(cx - 3, cy - 1, 1, white);
            break;

        case FruitType::GALAXIAN:
            drawRect(cx - 7, cy + 3, 14, 3, blue);
            drawRect(cx - 5, cy, 10, 3, yellow);
            drawRect(cx - 3, cy - 3, 6, 3, white);
            drawRect(cx - 1, cy - 6, 2, 3, cherryRed);
            drawRect(cx - 9, cy + 1, 2, 2, cherryRed);
            drawRect(cx + 7, cy + 1, 2, 2, cherryRed);
            break;

        case FruitType::BELL:
            drawRect(cx - 1, cy - 9, 2, 4, white);
            stripe(-4, 8, yellow);
            stripe(-2, 10, yellow);
            stripe(0, 12, yellow);
            stripe(2, 12, yellow);
            drawRect(cx - 5, cy + 4, 10, 3, yellow);
            drawCircle(cx, cy + 6, 2, blue);
            drawRect(cx - 6, cy + 4, 2, 2, gold);
            drawRect(cx + 4, cy + 4, 2, 2, gold);
            break;

        case FruitType::KEY:
            drawCircle(cx - 4, cy - 1, 4, gold);
            drawCircle(cx - 4, cy - 1, 2, black);
            drawRect(cx - 1, cy - 2, 10, 3, gold);
            drawRect(cx + 5, cy - 2, 2, 7, gold);
            drawRect(cx + 8, cy - 2, 2, 4, gold);
            break;
    }
}

SDL_Color Renderer::getGhostColor(GhostID id) const {
    switch (id) {
        case GhostID::BLINKY: return {255, 0, 0, 255};       // red
        case GhostID::PINKY:  return {255, 184, 255, 255};   // pink
        case GhostID::INKY:   return {0, 255, 255, 255};     // cyan
        case GhostID::CLYDE:  return {255, 184, 82, 255};    // orange
    }
    return {255, 255, 255, 255};
}

void Renderer::drawHUD(int score, int lives, int level, FruitType currentFruit, bool bonusActive) {
    int hudY = MAZE_ROWS * TILE_SIZE + 4;
    SDL_Color white = {255, 255, 255, 255};
    SDL_Color yellow = {255, 255, 0, 255};

    // Score
    drawString(8, hudY, "SCORE:" + std::to_string(score), white, 2);

    // Lives (draw Pac-Man icons)
    for (int i = 0; i < lives; i++) {
        drawCircle(SCREEN_W - 30 - i * 28, hudY + 10, 8, yellow, 0, M_PI / 3);
    }

    // Level
    drawString(SCREEN_W / 2 - 40, hudY, "LV:" + std::to_string(level), white, 2);

    // Current level fruit preview
    drawString(SCREEN_W / 2 + 56, hudY, bonusActive ? "BONUS" : "FRUIT",
               bonusActive ? yellow : white, 2);
    drawBonusFruit({MAZE_ROWS, 20}, currentFruit);
}

void Renderer::drawPause(void) {
    int hudY = MAZE_ROWS * TILE_SIZE + 4;
    SDL_Color white = {255, 255, 255, 255};
    SDL_Color black = {0, 0, 0, 127};

    // Overlay
    drawRect(0, 0, SCREEN_W, SCREEN_H, black);
    drawString(SCREEN_W / 2 - 60, SCREEN_H / 2 - 20, "PAUSED", white, 3);

}

void Renderer::drawGameOver(int finalScore) {
    SDL_Color red = {255, 0, 0, 255};
    SDL_Color white = {255, 255, 255, 255};

    drawString(SCREEN_W / 2 - 60, SCREEN_H / 2 - 20, "GAME OVER", red, 3);
    drawString(SCREEN_W / 2 - 50, SCREEN_H / 2 + 20, "SCORE:" + std::to_string(finalScore), white, 2);
}

void Renderer::drawReady() {
    SDL_Color yellow = {255, 255, 0, 255};
    drawString(SCREEN_W / 2 - 36, SCREEN_H / 2, "READY", yellow, 3);
}

// ============================================================
// Drawing Helpers
// ============================================================

void Renderer::drawRect(int x, int y, int w, int h, SDL_Color color) {
    SDL_SetRenderDrawColor(m_renderer, color.r, color.g, color.b, color.a);
    SDL_Rect rect = {x, y, w, h};
    SDL_RenderFillRect(m_renderer, &rect);
}

void Renderer::drawCircle(int cx, int cy, int radius, SDL_Color color) {
    SDL_SetRenderDrawColor(m_renderer, color.r, color.g, color.b, color.a);
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            if (dx*dx + dy*dy <= radius*radius) {
                SDL_RenderDrawPoint(m_renderer, cx + dx, cy + dy);
            }
        }
    }
}

void Renderer::drawCircle(int cx, int cy, int radius, SDL_Color color, float mouthAngle, float mouthSize) {
    SDL_SetRenderDrawColor(m_renderer, color.r, color.g, color.b, color.a);
    float halfMouth = mouthSize / 2.0f;
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            if (dx*dx + dy*dy <= radius*radius) {
                if (dx == 0 && dy == 0) continue;
                float angle = atan2f((float)dy, (float)dx);
                float diff = atan2f(sinf(angle - mouthAngle), cosf(angle - mouthAngle));
                if (fabsf(diff) <= halfMouth) continue;
                SDL_RenderDrawPoint(m_renderer, cx + dx, cy + dy);
            }
        }
    }
}

void Renderer::drawChar(int x, int y, char c, SDL_Color color, int scale) {
    const uint8_t* glyph = getGlyph(c);
    SDL_SetRenderDrawColor(m_renderer, color.r, color.g, color.b, color.a);

    for (int row = 0; row < 7; row++) {
        for (int col = 0; col < 5; col++) {
            if (glyph[row] & (0x10 >> col)) {
                SDL_Rect pixel = {x + col * scale, y + row * scale, scale, scale};
                SDL_RenderFillRect(m_renderer, &pixel);
            }
        }
    }
}

void Renderer::drawString(int x, int y, const std::string& text, SDL_Color color, int scale) {
    int cx = x;
    for (char c : text) {
        drawChar(cx, y, c, color, scale);
        cx += 6 * scale; // 5 pixels + 1 gap
    }
}
