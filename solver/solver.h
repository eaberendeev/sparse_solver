#pragma once

#include "kernel.h"

#include <omp.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <iostream>
#include <type_traits>
#include <unsupported/Eigen/IterativeSolvers>
#include <vector>

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SparseMatrix;
typedef Eigen::GMRES<Eigen::SparseMatrix<double, Eigen::RowMajor>> gmres;
typedef Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor>> bicgstab;
typedef Eigen::VectorXd VectorXd;
typedef Eigen::Triplet<double> Trip;


struct Solver {
    Solver(int xSize_, int ySize_, int zSize_)
        : xSize{xSize_}, ySize{ySize_}, zSize{zSize_} {
        init();
    }
    void init();

    SparseMatrix Mmat;
    SparseMatrix Imat;
    SparseMatrix curlE;
    SparseMatrix curlB;

    inline int sind(int i, int j, int k) const {
        return i * ySize * zSize + j * zSize + k;
    };
    // index for 3D vector fields
    inline int vind(int i, int j, int k, int d, int nd = 3) const {
        return d + nd * (i * ySize * zSize + j * zSize + k);
    };

    ~Solver() {}

    void stencil_curlB(SparseMatrix& mat);
    void stencil_curlE(SparseMatrix& mat);
    void stencil_Imat(SparseMatrix& mat);

    void solve_system();

   private:
    int xSize;
    int ySize;
    int zSize;
    double dx;
    double dt;
};
