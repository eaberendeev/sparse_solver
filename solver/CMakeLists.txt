project(Solver LANGUAGES CXX)

get_filename_component(PARENT_DIR ../ ABSOLUTE)

set(SOURCE_LIB  solver.cpp)

add_library(Solver OBJECT ${SOURCE_LIB})
