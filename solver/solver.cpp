#include "solver.h"
//#include "kernel.h"

void Solver::init(){
    dt = 1.5;
    dx = 0.5;
    int total_size = xSize*ySize*zSize;
    Mmat.resize(total_size * 3, total_size * 3);
    Imat.resize(total_size * 3, total_size * 3);
    curlE.resize(total_size * 3, total_size * 3);
    curlB.resize(total_size * 3, total_size * 3);

    stencil_Imat(Imat);
    stencil_curlE(curlE);
    stencil_curlB(curlB);

    Mmat = -0.25 * dt * dt * curlB * curlE;
    std::cout << "Initialization is done!\n";

}

void Solver::solve_system() {
    int N = 3*xSize*ySize*zSize;
    VectorXd En = VectorXd::Zero(N);
    VectorXd E = VectorXd::Random(N);
    VectorXd J = VectorXd::Random(N);

    VectorXd rhs = E - dt*J + Mmat*E;
    SparseMatrix A = Imat - Mmat;
    std::cout << "Prepare to solve done!\n";

    double time1 = omp_get_wtime();
    // my bicgtab solver
    solve_linear_system<BicgstabSolver<VectorXd>>(A, rhs, En, E);

    double time2 = omp_get_wtime();
    std::cout << "Ax-b error = " << (A * En - rhs).norm() << "\n";
    time2 = omp_get_wtime();
    // standart eigen solver
    solve_linear_system<bicgstab>(A, rhs, En, E);

    double time3 = omp_get_wtime();
    std::cout << "Ax-b error = " << (A * En - rhs).norm() << "\n";
    time3 = omp_get_wtime();

    auto matrix_op = [&](const VectorXd &v) {
        return (v - mv_product(Mmat, v)).eval();
    };
    // vaiant oof my bicgstab solver
    solve_linear_system<VectorXd, decltype(matrix_op)>(
        matrix_op, rhs, En, E);
    double time4 = omp_get_wtime();

    std::cout << "Ax-b error = " << (A * En - rhs).norm() << "\n";

    std::cout<< "Mysolver time = "<< (time2-time1) << "\n";
    std::cout << "Eigsolver time = " << (time3 - time2) << "\n";
    std::cout << "Mysolver2 time = " << (time4 - time3) << "\n";
}

void Solver::stencil_Imat(SparseMatrix &mat) {
    std::vector<Trip> trips;
    int totalSize = 3 * xSize * ySize * zSize;
    trips.reserve(totalSize);

    for (int i = 0; i < xSize; i++) {
        for (int j = 0; j < ySize; j++) {
            for (int k = 0; k < zSize; k++) {
                // i,j,k
                trips.push_back(Trip(vind(i, j, k, 0), vind(i, j, k, 0), 1.0));

                // i,j,k
                trips.push_back(Trip(vind(i, j, k, 1), vind(i, j, k, 1), 1.0));

                // i,j,k
                trips.push_back(Trip(vind(i, j, k, 2), vind(i, j, k, 2), 1.0));
            }
        }
    }
    std::cout << "Identity matrix. nnz: " << totalSize
              << ", rows: " << mat.rows() << ", cols: " << mat.cols() << "\n";
    mat.setFromTriplets(trips.begin(), trips.end());
}

void Solver::stencil_curlE(SparseMatrix &mat) {
    std::vector<Trip> trips;
    int totalSize = xSize * ySize * zSize * 12;
    trips.reserve(totalSize);

    for (int i = 0; i < xSize; i++) {
        for (int j = 0; j < ySize; j++) {
            for (int k = 0; k < zSize; k++) {
                const int ip = (i != xSize - 1) ? i + 1 : 3;
                const int jp = (j != ySize - 1) ? j + 1 : 3;
                const int kp = (k != zSize - 1) ? k + 1 : 3;

                const int vindx = vind(i, j, k, 0);
                const int vindy = vind(i, j, k, 1);
                const int vindz = vind(i, j, k, 2);

                // (x)[i,j+1/2,k+1/2]
                // ( Ez[i,j+1,k+1/2] - Ez[i,j,k+1/2] ) / dx
                double val = 1.0 / dx;
                trips.push_back(Trip(vindx, vind(i, jp, k, 2), val));
                trips.push_back(Trip(vindx, vind(i, j, k, 2), -val));
                // - ( Ey[i,j+1/2,k+1] - Ey[i,j+1/2,k] ) / dx
                val = -1.0 / dx;
                trips.push_back(Trip(vindx, vind(i, j, kp, 1), val));
                trips.push_back(Trip(vindx, vind(i, j, k, 1), -val));

                // (y)[i+1/2,j,k+1/2]
                // ( Ex[i+1/2,j,k+1] - Ex[i+1/2,j,k] ) / dx
                val = 1.0 / dx;
                trips.push_back(Trip(vindy, vind(i, j, kp, 0), val));
                trips.push_back(Trip(vindy, vind(i, j, k, 0), -val));
                // - ( Ez[i+1,j,k+1/2] - Ez[i,j,k+1/2] ) / dx
                val = -1.0 / dx;
                trips.push_back(Trip(vindy, vind(ip, j, k, 2), val));
                trips.push_back(Trip(vindy, vind(i, j, k, 2), -val));

                // (z)[i+1/2,j+1/2,k]
                // ( Ey[i+1,j+1/2,k] - Ey[i,j+1/2,k] ) / dx
                val = 1.0 / dx;
                trips.push_back(Trip(vindz, vind(ip, j, k, 1), val));
                trips.push_back(Trip(vindz, vind(i, j, k, 1), -val));
                // - ( Ex[i+1/2,j+1,k] - Ex[i+1/2,j,k] ) / dx
                val = -1.0 / dx;
                trips.push_back(Trip(vindz, vind(i, jp, k, 0), val));
                trips.push_back(Trip(vindz, vind(i, j, k, 0), -val));
            }
        }
    }

    std::cout << "rot_E matrix. nnz: " << totalSize << ", rows: " << mat.rows()
              << ", cols: " << mat.cols() << "\n";
    mat.setFromTriplets(trips.begin(), trips.end());
}

void Solver::stencil_curlB(SparseMatrix &mat) {
    std::vector<Trip> trips;
    int totalSize = xSize * ySize * zSize * 12;
    trips.reserve(totalSize);

    for (int i = 0; i < xSize; i++) {
        for (int j = 0; j < ySize; j++) {
            for (int k = 0; k < zSize; k++) {
                const int im = (i != 0) ? i - 1 : xSize - 4;
                const int jm = (j != 0) ? j - 1 : ySize - 4;
                const int km = (k != 0) ? k - 1 : zSize - 4;

                const int vindx = vind(i, j, k, 0);
                const int vindy = vind(i, j, k, 1);
                const int vindz = vind(i, j, k, 2);

                // (x)[i+1/2,j,k]
                // ( Bz[i+1/2,j+1/2,k] - Bz[i+1/2,j-1/2,k] ) / dx
                double val = 1.0 / dx;
                trips.push_back(Trip(vindx, vind(i, j, k, 2), val));
                trips.push_back(Trip(vindx, vind(i, jm, k, 2), -val));
                // - ( By[i+1/2,j,k+1/2] - By[i+1/2,j,k-1/2] ) / dx
                val = -1.0 / dx;
                trips.push_back(Trip(vindx, vind(i, j, k, 1), val));
                trips.push_back(Trip(vindx, vind(i, j, km, 1), -val));

                // (y)[i,j+1/2,k]
                // ( Bx[i,j+1/2,k+1/2] - Bx[i,j+1/2,k-1/2] ) / dx
                val = 1.0 / dx;
                trips.push_back(Trip(vindy, vind(i, j, k, 0), val));
                trips.push_back(Trip(vindy, vind(i, j, km, 0), -val));
                // -( Bz[i+1/2,j+1/2,k] - Bz[i-1/2,j+1/2,k] ) / dx
                val = -1.0 / dx;
                trips.push_back(Trip(vindy, vind(i, j, k, 2), val));
                trips.push_back(Trip(vindy, vind(im, j, k, 2), -val));

                // (z)[i,j,k+1/2]
                // ( By[i+1/2,j,k+1/2] - By[i-1/2,j,k+1/2] ) / dx
                val = 1.0 / dx;
                trips.push_back(Trip(vindz, vind(i, j, k, 1), val));
                trips.push_back(Trip(vindz, vind(im, j, k, 1), -val));
                // -( Bx[i,j+1/2,k+1/2] - Bx[i,j-1/2,k+1/2] ) / dx
                val = -1.0 / dx;
                trips.push_back(Trip(vindz, vind(i, j, k, 0), val));
                trips.push_back(Trip(vindz, vind(i, jm, k, 0), -val));
            }
        }
    }
    std::cout << "rot_B matrix. nnz: " << totalSize << ", rows: " << mat.rows()
              << ", cols: " << mat.cols() << "\n";
    mat.setFromTriplets(trips.begin(), trips.end());
}
