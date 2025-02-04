#include <iostream>

#include "solver.h"

int main() {
    int Nx = 140;
    int Ny = 140;
    int Nz = 210;

    Solver solver(Nx, Ny, Nz);
    solver.solve_system();

    return 0;
}
