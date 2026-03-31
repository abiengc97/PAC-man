cmake_minimum_required(VERSION 3.16)
project(PacMan LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find SDL2
find_package(SDL2 REQUIRED)

# ONNX Runtime (optional; used only if AIAgent is present)
set(ORT_ROOT "${CMAKE_SOURCE_DIR}/onnxruntime" CACHE PATH "ONNX Runtime root directory")

# Sources
set(SOURCES
    src/main.cpp
    src/Game.cpp
    src/Maze.cpp
    src/Player.cpp
    src/Ghost.cpp
    src/AStar.cpp
    src/Renderer.cpp
)

add_executable(pacman ${SOURCES})

set(USE_AI_AGENT OFF)
if(EXISTS "${CMAKE_SOURCE_DIR}/src/AIAgent.cpp")
    set(USE_AI_AGENT ON)
    list(APPEND SOURCES src/AIAgent.cpp)
    target_sources(pacman PRIVATE src/AIAgent.cpp)
endif()

target_include_directories(pacman PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${SDL2_INCLUDE_DIRS}
)

# Link SDL2 always; link ONNX only if AIAgent exists and ONNX runtime lib exists.
target_link_libraries(pacman PRIVATE ${SDL2_LIBRARIES})
if(USE_AI_AGENT AND EXISTS "${ORT_ROOT}/lib/libonnxruntime.so")
    target_include_directories(pacman PRIVATE ${ORT_ROOT}/include)
    target_link_libraries(pacman PRIVATE ${ORT_ROOT}/lib/libonnxruntime.so)
endif()

# Copy level files and model to build directory
file(COPY ${CMAKE_SOURCE_DIR}/levels DESTINATION ${CMAKE_BINARY_DIR})
if(EXISTS "${CMAKE_SOURCE_DIR}/pacman_policy.onnx")
    file(COPY ${CMAKE_SOURCE_DIR}/pacman_policy.onnx DESTINATION ${CMAKE_BINARY_DIR})
endif()
