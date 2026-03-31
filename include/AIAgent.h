#pragma once

#include "Common.h"
#include "Maze.h"
#include "Player.h"
#include "Ghost.h"
#include <onnxruntime_cxx_api.h>
#include <array>
#include <vector>
#include <string>

// State vector fed into the ONNX model
// 49 tiles (7x7 around Pac-Man) + 8 ghost features + 2 scalars = 59 floats
constexpr int STATE_SIZE = 59;

class AIAgent {
public:
    AIAgent() = default;

    // Load the trained ONNX model. Returns false if file not found.
    bool load(const std::string& modelPath);

    bool isLoaded() const { return m_loaded; }

    // Build state vector and run inference → returns best Direction
    Direction getAction(const Maze& maze, const Player& player,
                        const std::array<Ghost, 4>& ghosts);

    // Build state vector only (used for recording training data)
    std::array<float, STATE_SIZE> buildState(const Maze& maze,
                                             const Player& player,
                                             const std::array<Ghost, 4>& ghosts);

private:
    bool m_loaded = false;

    Ort::Env            m_env{ORT_LOGGING_LEVEL_WARNING, "pacman"};
    Ort::SessionOptions m_sessionOptions;
    Ort::Session        m_session{nullptr};
    Ort::AllocatorWithDefaultOptions m_allocator;

    const char* m_inputName  = "state";
    const char* m_outputName = "action_logits";
};
