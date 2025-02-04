cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_COMPILER "g++")

set(CMAKE_CXX_FLAGS "-Wall -g -O3 -march=native -fopenmp -Wextra -std=c++17")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(beren3d LANGUAGES CXX)

set(LIBRARIES Solver)
set(INCLUDES solver main .)
set(BIN_PATH ${PROJECT_BINARY_DIR}/bin)

#set(PATH_TO_EIGEN /home/berendeev/bpi/Progs/eigen-3.4.0/)
set(PATH_TO_EIGEN /home/master/soft/eigen-3.4.0/)

include_directories(SYSTEM ${PATH_TO_EIGEN})
include_directories(${INCLUDES})

add_subdirectory(solver)
add_subdirectory(main)

