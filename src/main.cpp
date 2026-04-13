#include "Game.h"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    bool headless   = false;
    bool rlMode     = false;
    bool rlRender   = false;
    int  startLevel = 1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--headless") headless  = true;
        if (arg == "--rl")       rlMode    = true;
        if (arg == "--render")   rlRender  = true;
        if (arg.rfind("--level=", 0) == 0) {
            startLevel = std::stoi(arg.substr(8));
        }
    }

    // --render only meaningful with --rl; implies SDL must be initialised
    if (rlRender) headless = false;

    Game game(headless, rlMode, startLevel, rlRender);

    if (!game.init()) {
        std::cerr << "Failed to initialize game." << std::endl;
        return 1;
    }

    game.run();
    return 0;
}
