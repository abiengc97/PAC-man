cmake_minimum_required(VERSION 3.16)
project(PacMan LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find SDL2
find_package(SDL2 REQUIRED)

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

target_include_directories(pacman PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${SDL2_INCLUDE_DIRS}
)

target_link_libraries(pacman PRIVATE ${SDL2_LIBRARIES})

# Copy level files to build directory
file(COPY ${CMAKE_SOURCE_DIR}/levels DESTINATION ${CMAKE_BINARY_DIR})
