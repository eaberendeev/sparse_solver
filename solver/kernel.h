#pragma once

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

#define DEFAULT_MAX_ITERATIONS 300
#define DEFAULT_TOLERANCE      1.e-9

template <typename VectorType>
inline VectorType mv_product(const SparseMatrix &A, const VectorType &v) {
    int rows = A.rows();
    VectorType res = VectorType(rows);

    const double* val = A.valuePtr();
    const int* inner = A.innerIndexPtr();
    const int* outer = A.outerIndexPtr();
#pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        double sum = 0;
#pragma omp simd
	for (int j = outer[i]; j < outer[i+1]; ++j) {
	//__builtin_prefetch(&v[inner[j + 4]]); // Предзагрузка через 4 элемента
            sum += val[j] * v[inner[j]];
        }
        res[i] = sum;
    }
    return res;
}

template <typename VectorType>
inline VectorType mv_product(const SparseMatrix &A, double alpha, const SparseMatrix &B, double beta,  const VectorType &v) {
    int rows = A.rows();
    VectorType res = VectorType(rows);

    const double* val = A.valuePtr();
    const int* inner = A.innerIndexPtr();
    const int* outer = A.outerIndexPtr();
    const double* val2 = B.valuePtr();
    const int* inner2 = B.innerIndexPtr();
    const int* outer2 = B.outerIndexPtr();
#pragma omp parallel for simd schedule(dynamic, 8196)
    for (int i = 0; i < rows; ++i) {
        double sum = 0;
#pragma omp simd
        for (int j = outer[i]; j < outer[i+1]; ++j) {
            sum += alpha*val[j] * v[inner[j]];
        }
#pragma omp simd
        for (int j = outer2[i]; j < outer2[i+1]; ++j) {
            sum += beta*val2[j] * v[inner2[j]];
        }
        res[i] = sum;
    }
    return res;
}

template <typename VectorType, typename MatrixOp>
bool bicgstab_iteration(MatrixOp &&Spmv, const VectorType &rhs, VectorType &x,
                        const VectorType &diagonal, size_t &iters,
                        double &tol_error) {
    using std::abs;
    using std::sqrt;
    double tol = tol_error;
    int maxIters = iters;
    int n = x.size();

    VectorType r = rhs - Spmv(x);

    VectorType r0 = r;
    double r0_sqnorm = r0.squaredNorm();
    double rhs_sqnorm = rhs.squaredNorm();
    if (rhs_sqnorm == 0) {
        x.setZero();
        return true;
    }
    double rho = 1;
    double alpha = 1;
    double w = 1;
    VectorType v = VectorType::Zero(n), p = VectorType::Zero(n);
    VectorType y(n), z(n);
    VectorType s(n), t(n);
    double tol2 = tol * tol * rhs_sqnorm;
    double eps2 = Eigen::NumTraits<double>::epsilon() *
                  Eigen::NumTraits<double>::epsilon();
    int i = 0;
    int restarts = 0;
    double time1 = 0;
    double timeall = omp_get_wtime();

    while (r.squaredNorm() > tol2 && i < maxIters) {
        double rho_old = rho;
        rho = r0.dot(r);
        if (abs(rho) < eps2 * r0_sqnorm) {
            double time10 = omp_get_wtime();
            r = rhs - Spmv(x);
            time1 += omp_get_wtime() - time10;
            r0 = r;
            rho = r0_sqnorm = r.squaredNorm();
            if (restarts++ == 0)
                i = 0;
        }
        double beta = (rho / rho_old) * (alpha / w);

#pragma omp parallel for simd
        for (int i = 0; i < n; i++) {
            p(i) = r(i) + beta * (p(i) - w * v(i));
        }
#pragma omp parallel for simd 
        for (int i = 0; i < n; i++) {
            y(i) = p(i) / diagonal(i);
        }

        // p = r + beta * (p - w * v);
        // y = precond.solve(p);   // Применение предобуславливателя
        double time10 = omp_get_wtime();
        v = Spmv(y);
        time1 += omp_get_wtime() - time10;

        alpha = rho / r0.dot(v);

        // s = r - alpha * v;
        // z = precond.solve(s);   // Применение предобуславливателя

#pragma omp parallel for simd
        for (int i = 0; i < n; i++) {
            s(i) = r(i) - alpha * v(i);
        }
#pragma omp parallel for simd
        for (int i = 0; i < n; i++) {
            z(i) = s(i) / diagonal(i);
        }

        time10 = omp_get_wtime();
        t = Spmv(z);
        time1 += omp_get_wtime() - time10;

        double tmp = t.squaredNorm();
        if (tmp > 0)
            w = t.dot(s) / tmp;
        else
            w = 0;

            // x += alpha * y + w * z;
            // r = s - w * t;
#pragma omp parallel for simd 
        for (int i = 0; i < n; i++) {
            x(i) += alpha * y(i) + w * z(i);
            r(i) = s(i) - w * t(i);
        }
        ++i;
    }
    std::cout << "All bicgstab iteration time: " << omp_get_wtime() - timeall << std::endl;
    std::cout << "SpMv time in bicg iteration: " << time1 << std::endl;
    tol_error = sqrt(r.squaredNorm() / rhs_sqnorm);
    iters = i;
    return true;
}

template <typename VectorType>
class BicgstabSolverBase {
   public:
    BicgstabSolverBase()
        : max_iterations(DEFAULT_MAX_ITERATIONS),
          m_tolerance(DEFAULT_TOLERANCE),
          m_success(false) {}

    void setTolerance(double tolerance) { m_tolerance = tolerance; }
    void setMaxIterations(size_t max_iters) { max_iterations = max_iters; }
    bool info() const { return m_success; }
    double error() const { return m_error; }
    size_t iterations() const { return m_iterations; }

    virtual VectorType solveWithGuess(const VectorType &rhs, const VectorType &x0) = 0;

   protected:
    virtual ~BicgstabSolverBase() = default;

    VectorType m_diagonal;
    size_t max_iterations;
    size_t m_iterations;
    double m_tolerance;
    double m_error;
    bool m_success;
};

template <typename VectorType>
class BicgstabSolver : public BicgstabSolverBase<VectorType> {
   public:
    using Base = BicgstabSolverBase<VectorType>;
    using Base::m_diagonal;
    using Base::m_error;
    using Base::m_iterations;
    using Base::m_success;
    using Base::m_tolerance;
    using Base::max_iterations;
    BicgstabSolver(const SparseMatrix &A) : m_A(A) {
        initializePreconditioner(A.rows());
    }
    BicgstabSolver(const SparseMatrix &A, const VectorType &diagonal) : m_A(A) {
        computeDiagonalPreconditioner(diagonal);
    }
    VectorType solveWithGuess(const VectorType &rhs, const VectorType &x0) {
        VectorType x = x0;
        m_iterations = max_iterations;
        m_error = m_tolerance;
        auto matrix_op = [&](const VectorType &v) {
            return mv_product(m_A, v);
        };
        m_success = bicgstab_iteration(matrix_op, rhs, x, m_diagonal,
                                       m_iterations, m_error);
        return x;
    }

   private:
    void computeDiagonalPreconditioner(const Eigen::VectorXd &diag) {
        for (int i = 0; i < m_diagonal.size(); i++) {
            m_diagonal[i] = diag[i];
        }
    }
    void initializePreconditioner(int rows) {
        m_diagonal.resize(rows);
        std::fill(m_diagonal.begin(), m_diagonal.end(), 1.0);
    }
    const SparseMatrix &m_A;
};

template <typename VectorType>
class BicgstabSolver2 : public BicgstabSolverBase<VectorType> {
   public:
    using Base = BicgstabSolverBase<VectorType>;
    using Base::m_diagonal;
    using Base::m_error;
    using Base::m_iterations;
    using Base::m_success;
    using Base::m_tolerance;
    using Base::max_iterations;
    template <typename MatrixOp>
    BicgstabSolver2(MatrixOp &&op, int rows)
        : m_op(std::forward<MatrixOp>(op)) {
        initializePreconditioner(rows);
    }
    template <typename MatrixOp>
    BicgstabSolver2(MatrixOp &&op, const VectorType &diagonal)
        : m_op(std::forward<MatrixOp>(op)) {
        computeDiagonalPreconditioner(diagonal);
    }

    VectorType solveWithGuess(const VectorType &rhs, const VectorType &x0) {
        VectorType x = x0;
        m_iterations = max_iterations;
        m_error = m_tolerance;
        std::cout << m_iterations << " " << m_error << "\n";
        m_success =
            bicgstab_iteration(m_op, rhs, x, m_diagonal, m_iterations, m_error);
        std::cout << m_iterations << " " << m_error << "\n";

        return x;
    }

   private:
    void computeDiagonalPreconditioner(const Eigen::VectorXd &diag) {
        for (int i = 0; i < m_diagonal.size(); i++) {
            m_diagonal[i] = diag[i];
        }
    }
    void initializePreconditioner(int rows) {
        m_diagonal.resize(rows);
        std::fill(m_diagonal.begin(), m_diagonal.end(), 1.0);
    }

    std::function<VectorType(const VectorType &)> m_op;
};

template <typename SolverType, typename VectorType>
void solve_linear_system_impl(SolverType &solver, const VectorType &rhs,
                              VectorType &x, const VectorType &x0) {
    solver.setTolerance(DEFAULT_TOLERANCE);
    solver.setMaxIterations(DEFAULT_MAX_ITERATIONS);

    x = solver.solveWithGuess(rhs, x0);

    if (solver.iterations() >= DEFAULT_MAX_ITERATIONS) {
        std::cout << "Field solver failed!" << std::endl;
        std::cout << solver.error() << std::endl;
    }
    std::cout << "iterations: " << solver.iterations()
              << ", solver error: " << solver.error() << std::endl;
}

template <typename SolverType, typename VectorType>
void solve_linear_system(const SparseMatrix& A, const VectorType& rhs,
                     VectorType& x, const VectorType& x0) {
 SolverType solver(A);
 solve_linear_system_impl(solver, rhs, x, x0);
}

template <typename VectorType>
void solve_linear_system(const SparseMatrix &A,
                         const VectorType &diagonal, const VectorType &rhs,
                         VectorType &x, const VectorType &x0) {
    BicgstabSolver solver(A, diagonal);
    solve_linear_system_impl(solver, rhs, x, x0);
}

template <typename VectorType, typename MatrixOp>
void solve_linear_system(MatrixOp &op, const VectorType &rhs, VectorType &x,
                         const VectorType &x0) {
    BicgstabSolver2<VectorType> solver(op, rhs.size());
    solve_linear_system_impl(solver, rhs, x, x0);
}

template <typename VectorType, typename MatrixOp>
void solve_linear_system(MatrixOp &op, const VectorType &diagonal,
                         const VectorType &rhs, VectorType &x,
                         const VectorType &x0) {
    BicgstabSolver2<VectorType> solver(op, rhs.size(), diagonal);
    solve_linear_system_impl(solver, rhs, x, x0);
}
