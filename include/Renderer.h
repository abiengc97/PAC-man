#pragma once

#include "Common.h"
#include <SDL2/SDL.h>
#include <string>

class Maze;
class Player;
class Ghost;

class Renderer {
public:
    Renderer();
    ~Renderer();

    bool init(const std::string& title, int width, int height);
    void shutdown();

    // Frame management
    void clear();
    void present();

    // Drawing functions
    void drawMaze(const Maze& maze);
    void drawPlayer(const Player& player);
    void drawGhost(const Ghost& ghost);
    void drawBonusFruit(GridPos pos, FruitType type);
    void drawTitleScreen(int selectedOption);
    void drawHUD(int score, int lives, int level, FruitType currentFruit, bool bonusActive);
    void drawPause(void);
    void drawGameOver(int finalScore);
    void drawReady();

    // Colors for ghost rendering (since no sprites yet)
    SDL_Color getGhostColor(GhostID id) const;

private:
    SDL_Window*   m_window   = nullptr;
    SDL_Renderer* m_renderer = nullptr;

    // Helper: draw a filled rectangle
    void drawRect(int x, int y, int w, int h, SDL_Color color);
    // Helper: draw a filled circle (for Pac-Man and ghosts)
    void drawCircle(int cx, int cy, int radius, SDL_Color color);
    void drawCircle(int cx, int cy, int radius, SDL_Color color, float mouthAngle, float mouthSize);
    // Helper: simple text rendering using rectangles (no SDL_ttf needed)
    void drawChar(int x, int y, char c, SDL_Color color, int scale = 1);
    void drawString(int x, int y, const std::string& text, SDL_Color color, int scale = 1);
    // Helper: draw a filled in triangle
    void drawTriangle(int x1, int y1, int x2, int y2, int x3, int y3, SDL_Color color);
};
