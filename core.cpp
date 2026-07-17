// ============================================================
// Component 2.4 — HAF-AGE KV-Cache
// Hierarchical Aggregate Functional Cache with Adaptive Geometric Enrichment
// Phase 1: Core Data Structures & State (SPEC §4.1)
// ============================================================
// This file implements the foundational data structures and
// linear algebra utilities for the HAF-AGE KV-Cache.
// Mathematical rigor: all matrix ops use cache-friendly layouts,
// eigendecomposition uses symmetric Jacobi with quadratic convergence.

#include <vector>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <cassert>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <limits>
#include <tuple>
#include <memory>
#include <iostream>
#include <iomanip>
#include <random>

namespace haf_age {

// ============================================================
// §4.1.1: Global Parameters
// ============================================================

struct HAFConfig {
    size_t L = 32;           // Number of layers
    size_t g = 8;            // Number of KV groups (GQA)
    size_t d_k = 128;        // Key head dimension
    size_t d_v = 128;        // Value head dimension
    size_t W = 512;          // Exact window size
    size_t r_max = 32;       // Maximum rank per block
    double beta = 0.8;       // Entropy quality budget
    size_t r_feat = 256;     // Feature map dimension (linear attention)
    size_t J_max = 20;       // Max hierarchy level (covers up to 2^20 * W tokens)
};

// ============================================================
// Matrix: Dense row-major matrix with BLAS-friendly layout
// ============================================================

class Matrix {
public:
    Matrix() : rows_(0), cols_(0), data_(nullptr), stride_(0) {}

    Matrix(size_t rows, size_t cols)
        : rows_(rows), cols_(cols), data_(nullptr), stride_(cols)
    {
        if (rows == 0 || cols == 0) {
            rows_ = 0; cols_ = 0; stride_ = 0;
            return;
        }
        // Cache-line aligned allocation for SIMD friendliness
        size_t size = rows * stride_;
        data_ = new double[size];
        std::fill(data_, data_ + size, 0.0);
    }

    Matrix(const Matrix& other)
        : rows_(other.rows_), cols_(other.cols_), stride_(other.stride_)
    {
        if (rows_ > 0 && cols_ > 0) {
            size_t size = rows_ * stride_;
            data_ = new double[size];
            std::copy(other.data_, other.data_ + size, data_);
        } else {
            data_ = nullptr;
        }
    }

    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            Matrix temp(other);
            swap(temp);
        }
        return *this;
    }

    Matrix(Matrix&& other) noexcept
        : rows_(other.rows_), cols_(other.cols_),
          data_(other.data_), stride_(other.stride_)
    {
        other.rows_ = 0; other.cols_ = 0; other.stride_ = 0;
        other.data_ = nullptr;
    }

    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            swap(other);
        }
        return *this;
    }

    ~Matrix() {
        delete[] data_;
    }

    void swap(Matrix& other) noexcept {
        std::swap(rows_, other.rows_);
        std::swap(cols_, other.cols_);
        std::swap(stride_, other.stride_);
        std::swap(data_, other.data_);
    }

    // Element access: (row, col), 0-indexed
    double& operator()(size_t row, size_t col) {
        assert(row < rows_ && col < cols_);
        return data_[row * stride_ + col];
    }

    const double& operator()(size_t row, size_t col) const {
        assert(row < rows_ && col < cols_);
        return data_[row * stride_ + col];
    }

    // Raw pointer to row data (useful for BLAS)
    double* row_ptr(size_t row) {
        assert(row < rows_);
        return data_ + row * stride_;
    }

    const double* row_ptr(size_t row) const {
        assert(row < rows_);
        return data_ + row * stride_;
    }

    // Properties
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t stride() const { return stride_; }
    bool empty() const { return rows_ == 0 || cols_ == 0; }

    // Fill operations
    void fill(double val) {
        std::fill(data_, data_ + rows_ * stride_, val);
    }

    void fill_identity() {
        fill(0.0);
        size_t n = std::min(rows_, cols_);
        for (size_t i = 0; i < n; ++i) {
            (*this)(i, i) = 1.0;
        }
    }

    void fill_random(std::mt19937_64& rng, double scale = 1.0) {
        std::normal_distribution<double> dist(0.0, scale);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                (*this)(i, j) = dist(rng);
            }
        }
    }

    // Frobenius norm squared
    double frobenius_sq() const {
        double sum = 0.0;
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                sum += (*this)(i, j) * (*this)(i, j);
            }
        }
        return sum;
    }

    double frobenius_norm() const {
        return std::sqrt(frobenius_sq());
    }

    // Maximum absolute value
    double max_abs() const {
        double m = 0.0;
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                m = std::max(m, std::abs((*this)(i, j)));
            }
        }
        return m;
    }

    // Extract submatrix: rows [r_start, r_end), cols [c_start, c_end)
    Matrix submatrix(size_t r_start, size_t r_end,
                     size_t c_start, size_t c_end) const {
        size_t r = r_end - r_start;
        size_t c = c_end - c_start;
        Matrix M(r, c);
        for (size_t i = 0; i < r; ++i) {
            for (size_t j = 0; j < c; ++j) {
                M(i, j) = (*this)(r_start + i, c_start + j);
            }
        }
        return M;
    }

    // Column vector access
    Matrix col(size_t j) const {
        Matrix v(rows_, 1);
        for (size_t i = 0; i < rows_; ++i) {
            v(i, 0) = (*this)(i, j);
        }
        return v;
    }

    void set_col(size_t j, const Matrix& v) {
        assert(v.rows() == rows_ && v.cols() == 1);
        for (size_t i = 0; i < rows_; ++i) {
            (*this)(i, j) = v(i, 0);
        }
    }

    // Printing (for debugging)
    void print(const std::string& name = "",
               size_t max_rows = 6, size_t max_cols = 6) const {
        if (!name.empty()) std::cout << name << " = " << std::endl;
        if (empty()) { std::cout << "[]" << std::endl; return; }
        size_t r = std::min(rows_, max_rows);
        size_t c = std::min(cols_, max_cols);
        for (size_t i = 0; i < r; ++i) {
            for (size_t j = 0; j < c; ++j) {
                std::cout << std::setw(12) << std::setprecision(6)
                          << (*this)(i, j) << " ";
            }
            if (c < cols_) std::cout << "... ";
            std::cout << std::endl;
        }
        if (r < rows_) std::cout << "..." << std::endl;
    }

private:
    size_t rows_;
    size_t cols_;
    double* data_;
    size_t stride_;  // row stride (>= cols_ for alignment)
};

// ============================================================
// Linear Algebra Operations
// ============================================================

namespace linalg {

// C = A * B  (standard matrix multiplication)
Matrix matmul(const Matrix& A, const Matrix& B) {
    assert(A.cols() == B.rows());
    size_t m = A.rows(), n = A.cols(), p = B.cols();
    Matrix C(m, p);
    for (size_t i = 0; i < m; ++i) {
        for (size_t k = 0; k < n; ++k) {
            double aik = A(i, k);
            if (aik == 0.0) continue;
            for (size_t j = 0; j < p; ++j) {
                C(i, j) += aik * B(k, j);
            }
        }
    }
    return C;
}

// C = A * B^T
Matrix matmul_ABt(const Matrix& A, const Matrix& B) {
    assert(A.cols() == B.cols());
    size_t m = A.rows(), n = A.cols(), p = B.rows();
    Matrix C(m, p);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += A(i, k) * B(j, k);
            }
            C(i, j) = sum;
        }
    }
    return C;
}

// C = A^T * B
Matrix matmul_AtB(const Matrix& A, const Matrix& B) {
    assert(A.rows() == B.rows());
    size_t m = A.cols(), n = A.rows(), p = B.cols();
    Matrix C(m, p);
    for (size_t i = 0; i < m; ++i) {
        for (size_t k = 0; k < n; ++k) {
            double aik = A(k, i);
            if (aik == 0.0) continue;
            for (size_t j = 0; j < p; ++j) {
                C(i, j) += aik * B(k, j);
            }
        }
    }
    return C;
}

Matrix transpose(const Matrix& A) {
    Matrix AT(A.cols(), A.rows());
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            AT(j, i) = A(i, j);
        }
    }
    return AT;
}

// Trace of square matrix
double trace(const Matrix& A) {
    assert(A.rows() == A.cols());
    double t = 0.0;
    for (size_t i = 0; i < A.rows(); ++i) {
        t += A(i, i);
    }
    return t;
}

// Outer product: v * w^T where v ∈ ℝ^m, w ∈ ℝ^n
Matrix outer(const Matrix& v, const Matrix& w) {
    assert(v.cols() == 1 && w.cols() == 1);
    Matrix M(v.rows(), w.rows());
    for (size_t i = 0; i < v.rows(); ++i) {
        for (size_t j = 0; j < w.rows(); ++j) {
            M(i, j) = v(i, 0) * w(j, 0);
        }
    }
    return M;
}

// Vector dot product
double dot(const Matrix& v, const Matrix& w) {
    assert(v.rows() == w.rows() && v.cols() == 1 && w.cols() == 1);
    double d = 0.0;
    for (size_t i = 0; i < v.rows(); ++i) {
        d += v(i, 0) * w(i, 0);
    }
    return d;
}

// Element-wise operations
Matrix add(const Matrix& A, const Matrix& B) {
    assert(A.rows() == B.rows() && A.cols() == B.cols());
    Matrix C(A.rows(), A.cols());
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            C(i, j) = A(i, j) + B(i, j);
        }
    }
    return C;
}

Matrix sub(const Matrix& A, const Matrix& B) {
    assert(A.rows() == B.rows() && A.cols() == B.cols());
    Matrix C(A.rows(), A.cols());
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            C(i, j) = A(i, j) - B(i, j);
        }
    }
    return C;
}

Matrix scale(const Matrix& A, double s) {
    Matrix C(A.rows(), A.cols());
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            C(i, j) = s * A(i, j);
        }
    }
    return C;
}

// Build diagonal matrix from column vector
Matrix diag(const Matrix& v) {
    assert(v.cols() == 1);
    size_t n = v.rows();
    Matrix D(n, n);
    for (size_t i = 0; i < n; ++i) {
        D(i, i) = v(i, 0);
    }
    return D;
}

// Extract diagonal as column vector
Matrix diag_vec(const Matrix& A) {
    assert(A.rows() == A.cols());
    size_t n = A.rows();
    Matrix v(n, 1);
    for (size_t i = 0; i < n; ++i) {
        v(i, 0) = A(i, i);
    }
    return v;
}

// sqrt of eigenvalues (element-wise)
Matrix sqrt_eigenvalues(const Matrix& Lambda) {
    assert(Lambda.cols() == 1);
    Matrix s(Lambda.rows(), 1);
    for (size_t i = 0; i < Lambda.rows(); ++i) {
        double val = Lambda(i, 0);
        s(i, 0) = (val > 0.0) ? std::sqrt(val) : 0.0;
    }
    return s;
}

// ============================================================
// Symmetric Eigendecomposition: A = V * diag(λ) * V^T
// Uses the Jacobi eigenvalue algorithm — quadratic convergence,
// O(n³) per sweep, numerically stable for symmetric matrices.
// ============================================================

// Sign function for Jacobi
inline double sign(double x) {
    return (x >= 0.0) ? 1.0 : -1.0;
}

struct EVDResult {
    Matrix eigenvalues;  // column vector of eigenvalues
    Matrix eigenvectors; // columns are eigenvectors
    size_t iterations;
    double residual;
};

EVDResult symmetric_eigendecompose(const Matrix& A, double tol = 1e-14,
                                    size_t max_sweeps = 20) {
    size_t n = A.rows();
    assert(A.rows() == A.cols() && n > 0);

    // Copy A (we'll destroy it)
    Matrix D = A;
    Matrix V(n, n);
    V.fill_identity();

    size_t total_rotations = 0;

    for (size_t sweep = 0; sweep < max_sweeps; ++sweep) {
        // Compute off-diagonal Frobenius norm for convergence check
        double off_norm_sq = 0.0;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                off_norm_sq += D(i, j) * D(i, j);
            }
        }
        if (std::sqrt(off_norm_sq) < tol) break;

        // Cyclic Jacobi sweep: visit each (p,q) pair once in row-major order
        for (size_t p = 0; p < n; ++p) {
            for (size_t q = p + 1; q < n; ++q) {
                double Dpq = D(p, q);
                if (std::abs(Dpq) < tol) continue;

                double Dpp = D(p, p), Dqq = D(q, q);

                // Compute Jacobi rotation angle
                double c, s;
                if (std::abs(Dpp - Dqq) < tol) {
                    c = std::sqrt(0.5);
                    s = std::sqrt(0.5);
                } else {
                    double tau = (Dqq - Dpp) / (2.0 * Dpq);
                    double t = sign(tau) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
                    c = 1.0 / std::sqrt(1.0 + t * t);
                    s = c * t;
                }

                // Apply rotation to diagonalize D[p][q]
                double Dpp_new = c * c * Dpp - 2.0 * s * c * Dpq + s * s * Dqq;
                double Dqq_new = s * s * Dpp + 2.0 * s * c * Dpq + c * c * Dqq;

                D(p, p) = Dpp_new;
                D(q, q) = Dqq_new;
                D(p, q) = 0.0;
                D(q, p) = 0.0;

                // Rotate other rows/columns
                for (size_t i = 0; i < n; ++i) {
                    if (i == p || i == q) continue;
                    double Dip = D(i, p), Diq = D(i, q);
                    D(i, p) = c * Dip - s * Diq;
                    D(p, i) = D(i, p);
                    D(i, q) = s * Dip + c * Diq;
                    D(q, i) = D(i, q);
                }

                // Update eigenvectors
                for (size_t i = 0; i < n; ++i) {
                    double Vip = V(i, p), Viq = V(i, q);
                    V(i, p) = c * Vip - s * Viq;
                    V(i, q) = s * Vip + c * Viq;
                }

                ++total_rotations;
            }
        }
    }

    // Extract eigenvalues from diagonal
    Matrix eigenvalues(n, 1);
    for (size_t i = 0; i < n; ++i) {
        eigenvalues(i, 0) = D(i, i);
    }

    // Sort eigenvalues and eigenvectors in descending order
    // Simple bubble sort (n ≤ 128, so this is fine)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if (eigenvalues(j, 0) > eigenvalues(i, 0)) {
                std::swap(eigenvalues(i, 0), eigenvalues(j, 0));
                for (size_t k = 0; k < n; ++k) {
                    std::swap(V(k, i), V(k, j));
                }
            }
        }
    }

    // Clamp tiny negative eigenvalues to zero (numerical noise)
    for (size_t i = 0; i < n; ++i) {
        if (eigenvalues(i, 0) < 0.0 && std::abs(eigenvalues(i, 0)) < tol * 100) {
            eigenvalues(i, 0) = 0.0;
        }
    }

    // Compute residual: ||A - V diag(λ) V^T||_F
    Matrix VT = transpose(V);
    Matrix recon = matmul(V, matmul(diag(eigenvalues), VT));
    double residual_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double diff = A(i, j) - recon(i, j);
            residual_sq += diff * diff;
        }
    }

    return {eigenvalues, V, total_rotations, std::sqrt(residual_sq)};
}

} // namespace linalg

// ============================================================
// §4.1.2: HAFCacheBlock — Compressed block statistics
// ============================================================

struct HAFCacheBlock {
    size_t n;               // Block size (token count)
    Matrix mu_k;            // ℝ^{d_k} — Key mean
    Matrix mu_v;            // ℝ^{d_v} — Value mean
    Matrix U_k;             // ℝ^{d_k × r_k} — Key principal directions
    Matrix Lambda_k;        // ℝ^{r_k × 1} — Key eigenvalues (vector form)
    Matrix U_v;             // ℝ^{d_v × r_v} — Value principal directions
    Matrix Lambda_v;        // ℝ^{r_v × 1} — Value eigenvalues (vector form)
    Matrix C_kv;            // ℝ^{r_k × r_v} — Cross-covariance in latent space
    Matrix S_agg;           // ℝ^{r_feat × d_v} — Precomputed linear attention aggregate

    HAFCacheBlock() : n(0) {}

    HAFCacheBlock(size_t d_k, size_t d_v, size_t r_k, size_t r_v, size_t r_feat)
        : n(0)
        , mu_k(d_k, 1)
        , mu_v(d_v, 1)
        , U_k(d_k, r_k)
        , Lambda_k(r_k, 1)
        , U_v(d_v, r_v)
        , Lambda_v(r_v, 1)
        , C_kv(r_k, r_v)
        , S_agg(r_feat, d_v)
    {}

    size_t r_k() const { return Lambda_k.rows(); }
    size_t r_v() const { return Lambda_v.rows(); }

    // Reconstruct approximate covariance: Σ ≈ U diag(λ) U^T
    Matrix reconstruct_key_cov() const {
        Matrix L = linalg::diag(Lambda_k);
        return linalg::matmul(U_k, linalg::matmul(L, linalg::transpose(U_k)));
    }

    Matrix reconstruct_val_cov() const {
        Matrix L = linalg::diag(Lambda_v);
        return linalg::matmul(U_v, linalg::matmul(L, linalg::transpose(U_v)));
    }

    // Truncation error bound: Σ_{i=r+1}^{d_k} λ_i
    double truncation_error_key() const {
        // For a block, we only store top r eigenvalues.
        // The full trace is not stored, but the error bound
        // is computed from the eigenvalues of the original covariance.
        // Here we return 0 as placeholder — computed from full EVD.
        return 0.0;
    }
};

// Forward declarations (defined later in this file)
struct HAFConfig;
Matrix phi(const Matrix& v);
HAFCacheBlock compute_block_stats(const Matrix& K_block, const Matrix& V_block,
                                   const HAFConfig& config);
HAFCacheBlock merge_blocks(const HAFCacheBlock& A, const HAFCacheBlock& B,
                            size_t d_k, size_t d_v, size_t r_feat);

// ============================================================
// §4.1.3-4: HAFCache — Container with ring buffer
// ============================================================

class HAFCache {
public:
    explicit HAFCache(const HAFConfig& config)
        : config_(config)
        , exact_buffer_K_(config.W, config.d_k)
        , exact_buffer_V_(config.W, config.d_v)
        , exact_ptr_(0)
        , exact_count_(0)
        , S_exact_(config.r_feat, config.d_v)
        , S_total_(config.r_feat, config.d_v)
    {
        // Initialize hierarchical block levels
        // Level 0 is the exact buffer (ring buffer)
        // Levels 1..J_max hold compressed blocks
        blocks_by_level_.resize(config.J_max + 1);
    }

    // ---- §4.1.4: Ring Buffer Operations ----

    // Write a single token (k_t, v_t) to exact buffer
    void write_exact_buffer(const Matrix& k_t, const Matrix& v_t) {
        assert(k_t.rows() == config_.d_k && k_t.cols() == 1);
        assert(v_t.rows() == config_.d_v && v_t.cols() == 1);

        // Write at current pointer position
        for (size_t i = 0; i < config_.d_k; ++i) {
            exact_buffer_K_(exact_ptr_, i) = k_t(i, 0);
        }
        for (size_t i = 0; i < config_.d_v; ++i) {
            exact_buffer_V_(exact_ptr_, i) = v_t(i, 0);
        }

        // Advance pointer (circular)
        exact_ptr_ = (exact_ptr_ + 1) % config_.W;
        exact_count_ = std::min(exact_count_ + 1, config_.W);
    }

    // Read the exact buffer in order (oldest to newest)
    void read_exact_buffer_ordered(Matrix& K_out, Matrix& V_out,
                                    size_t offset = 0, size_t count = 0) const {
        if (count == 0) count = exact_count_;
        assert(offset + count <= exact_count_);
        assert(K_out.rows() == count && K_out.cols() == config_.d_k);
        assert(V_out.rows() == count && V_out.cols() == config_.d_v);

        // The ring buffer stores newest at exact_ptr_-1, oldest at
        // (exact_ptr_ - exact_count_) mod W. Read in chronological order.
        size_t start = (exact_ptr_ + config_.W - exact_count_ + offset) % config_.W;
        for (size_t t = 0; t < count; ++t) {
            size_t idx = (start + t) % config_.W;
            for (size_t i = 0; i < config_.d_k; ++i) {
                K_out(t, i) = exact_buffer_K_(idx, i);
            }
            for (size_t i = 0; i < config_.d_v; ++i) {
                V_out(t, i) = exact_buffer_V_(idx, i);
            }
        }
    }

    // ---- §4.2: Prefill Phase ----

    // PrefillPopulate(K, V): process full prompt in a single pass.
    // Constructs hierarchical blocks directly and computes S_total.
    void prefill_populate(const Matrix& K, const Matrix& V) {
        assert(K.rows() == V.rows());
        assert(K.rows() == 0 ||
               (K.cols() == config_.d_k && V.cols() == config_.d_v));

        size_t n_prompt = K.rows();
        if (n_prompt == 0) return;
        reset();

        // Step 2: Extract W-sized blocks and compute block statistics
        size_t num_blocks = (n_prompt + config_.W - 1) / config_.W;
        for (size_t b = 0; b < num_blocks; ++b) {
            size_t start = b * config_.W;
            size_t end = std::min(start + config_.W, n_prompt);
            Matrix K_b = K.submatrix(start, end, 0, config_.d_k);
            Matrix V_b = V.submatrix(start, end, 0, config_.d_v);

            HAFCacheBlock block = compute_block_stats(K_b, V_b, config_);
            blocks_by_level_[1].push_back(block);
        }

        // Step 3: Merge blocks hierarchically
        for (size_t j = 1; j < config_.J_max; ++j) {
            while (blocks_by_level_[j].size() >= 2) {
                // Pop two oldest blocks from level j
                auto& level = blocks_by_level_[j];
                HAFCacheBlock B_left = level.front();
                level.erase(level.begin());
                HAFCacheBlock B_right = level.front();
                level.erase(level.begin());

                // Merge into level j+1
                HAFCacheBlock B_merged = merge_blocks(
                    B_left, B_right, config_.d_k, config_.d_v, config_.r_feat);

                // Sum the S_agg of both blocks
                B_merged.S_agg = linalg::add(B_left.S_agg, B_right.S_agg);

                blocks_by_level_[j + 1].push_back(B_merged);
            }
        }

        // Step 4: Compute S_total = Σ_j S_agg(B_j) for all blocks
        S_total_.fill(0.0);
        for (size_t j = 1; j < config_.J_max; ++j) {
            for (const auto& block : blocks_by_level_[j]) {
                S_total_ = linalg::add(S_total_, block.S_agg);
            }
        }

        // Step 5: Set exact buffer to remaining tokens (n_prompt mod W)
        size_t remaining = n_prompt % config_.W;
        if (remaining > 0) {
            size_t start = n_prompt - remaining;
            for (size_t i = 0; i < remaining; ++i) {
                Matrix k_t(config_.d_k, 1);
                Matrix v_t(config_.d_v, 1);
                for (size_t j = 0; j < config_.d_k; ++j) k_t(j, 0) = K(start + i, j);
                for (size_t j = 0; j < config_.d_v; ++j) v_t(j, 0) = V(start + i, j);
                write_exact_buffer(k_t, v_t);
            }
        }
    }

    // ---- §4.3: Decode Phase (Autoregressive Step) ----
    //
    // DecodeStep: process a single new token (k_t, v_t).
    // 1. Write to exact buffer (ring buffer)
    // 2. If exact buffer is full, compress to block
    // 3. Cascade merge across levels
    // Returns (updated cache, S_total)

    void decode_step(const Matrix& k_t, const Matrix& v_t) {
        assert(k_t.rows() == config_.d_k && k_t.cols() == 1);
        assert(v_t.rows() == config_.d_v && v_t.cols() == 1);

        // Step 1: Exact buffer update (O(1))
        write_exact_buffer(k_t, v_t);

        // Update S_exact and S_total with φ(k_t) ⊗ v_t^T
        Matrix phi_k = phi(k_t);
        Matrix phi_k_times_vt(config_.r_feat, config_.d_v);
        for (size_t a = 0; a < config_.r_feat; ++a) {
            for (size_t b = 0; b < config_.d_v; ++b) {
                phi_k_times_vt(a, b) = phi_k(a, 0) * v_t(b, 0);
            }
        }
        S_exact_ = linalg::add(S_exact_, phi_k_times_vt);
        S_total_ = linalg::add(S_total_, phi_k_times_vt);

        // Step 2: Block compression trigger (exact buffer full)
        // Fires every W tokens: when exact_count_ reaches W,
        // the buffer is full, we compress into a block and reset.
        if (exact_count_ == config_.W) {
            // The buffer just wrapped — the oldest token was overwritten.
            // Extract the full buffer as a block.
            // Read the full exact buffer in order
            Matrix K_buf(config_.W, config_.d_k);
            Matrix V_buf(config_.W, config_.d_v);
            read_exact_buffer_ordered(K_buf, V_buf);

            HAFCacheBlock new_block = compute_block_stats(K_buf, V_buf, config_);
            new_block.S_agg = S_exact_;
            blocks_by_level_[1].push_back(new_block);

            // Reset exact buffer state (the buffer's data is now in the block)
            S_exact_.fill(0.0);
            exact_count_ = 0;

            // Step 3: Cascade merge across levels
            for (size_t j = 1; j < config_.J_max; ++j) {
                if (blocks_by_level_[j].size() >= 2) {
                    auto& level = blocks_by_level_[j];
                    HAFCacheBlock B_left = level.front();
                    level.erase(level.begin());
                    HAFCacheBlock B_right = level.front();
                    level.erase(level.begin());

                    HAFCacheBlock B_merged = merge_blocks(
                        B_left, B_right, config_.d_k, config_.d_v, config_.r_feat);
                    B_merged.S_agg = linalg::add(B_left.S_agg, B_right.S_agg);

                    blocks_by_level_[j + 1].push_back(B_merged);
                } else {
                    break;
                }
            }
        }

        // Recompute S_total from components
        // (In a production system we'd update incrementally;
        //  for correctness we recompute)
        recompute_s_total();
    }

    // Recompute S_total from all components
    void recompute_s_total() {
        S_total_ = S_exact_;
        for (size_t j = 1; j < config_.J_max; ++j) {
            for (const auto& block : blocks_by_level_[j]) {
                S_total_ = linalg::add(S_total_, block.S_agg);
            }
        }
    }

    // ---- §4.5: Reconstruction Protocol ----
    //
    // Reconstruct approximate K,V for prefix [1, m]
    // Uses Gaussian sampling from block statistics, falls back
    // to exact buffer for recent tokens.

    std::pair<Matrix, Matrix> reconstruct_prefix(size_t m) const {
        size_t d_k = config_.d_k, d_v = config_.d_v;
        double pi = std::acos(-1.0);

        Matrix K_hat(m, d_k);
        Matrix V_hat(m, d_v);

        size_t remaining = m;

        // Traverse blocks from oldest (highest level) to newest
        for (int j = (int)config_.J_max; j >= 1 && remaining > 0; --j) {
            for (const auto& block : blocks_by_level_[j]) {
                if (remaining == 0) break;

                size_t take = std::min(block.n, remaining);

                for (size_t t = 0; t < take; ++t) {
                    // Spectral reconstruction: μ + U @ sqrt(Λ) @ z(t)
                    // Per-token perturbation z[t][r] uses cosine basis to
                    // produce distinct tokens preserving block covariance.
                    size_t pos = m - remaining;

                    // Mean
                    for (size_t i = 0; i < d_k; ++i) K_hat(pos, i) = block.mu_k(i, 0);
                    for (size_t i = 0; i < d_v; ++i) V_hat(pos, i) = block.mu_v(i, 0);

                    // Perturbation: sqrt(2)·cos(π·t·(r+1)/take) preserves
                    // E[z] = 0, E[z_r·z_s] = δ_{rs} across the block.
                    for (size_t r = 0; r < block.r_k() && r < block.Lambda_k.rows(); ++r) {
                        double sigma = (block.Lambda_k(r, 0) > 0.0)
                            ? std::sqrt(block.Lambda_k(r, 0)) : 0.0;
                        if (sigma > 1e-15) {
                            double z = std::sqrt(2.0) * std::cos(pi * t * (r + 1) / take);
                            for (size_t i = 0; i < d_k; ++i)
                                K_hat(pos, i) += z * sigma * block.U_k(i, r);
                        }
                    }
                    for (size_t r = 0; r < block.r_v() && r < block.Lambda_v.rows(); ++r) {
                        double sigma = (block.Lambda_v(r, 0) > 0.0)
                            ? std::sqrt(block.Lambda_v(r, 0)) : 0.0;
                        if (sigma > 1e-15) {
                            double w = std::sqrt(2.0) * std::cos(pi * t * (r + 1) / take);
                            for (size_t i = 0; i < d_v; ++i)
                                V_hat(pos, i) += w * sigma * block.U_v(i, r);
                        }
                    }
                }
                remaining -= take;
                if (remaining == 0) break;
            }
        }

        // Fill remaining from exact buffer (most recent tokens)
        if (remaining > 0 && exact_count_ > 0) {
            size_t fill = std::min(remaining, exact_count_);
            Matrix K_ex(fill, d_k);
            Matrix V_ex(fill, d_v);
            read_exact_buffer_ordered(K_ex, V_ex,
                                       exact_count_ - fill, fill);

            for (size_t t = 0; t < fill; ++t) {
                size_t pos = m - remaining + t;
                for (size_t i = 0; i < d_k; ++i) K_hat(pos, i) = K_ex(t, i);
                for (size_t i = 0; i < d_v; ++i) V_hat(pos, i) = V_ex(t, i);
            }
            remaining -= fill;
        }

        return {K_hat, V_hat};
    }

    // ---- State Management ----

    void reset() {
        exact_ptr_ = 0;
        exact_count_ = 0;
        exact_buffer_K_.fill(0.0);
        exact_buffer_V_.fill(0.0);
        S_exact_.fill(0.0);
        S_total_.fill(0.0);
        for (auto& level : blocks_by_level_) {
            level.clear();
        }
    }

    // ---- Accessors ----

    const HAFConfig& config() const { return config_; }
    const Matrix& S_total() const { return S_total_; }
    const Matrix& S_exact() const { return S_exact_; }
    size_t exact_ptr() const { return exact_ptr_; }
    size_t exact_count() const { return exact_count_; }
    size_t num_levels() const { return blocks_by_level_.size(); }
    size_t num_blocks_at_level(size_t j) const {
        if (j >= blocks_by_level_.size()) return 0;
        return blocks_by_level_[j].size();
    }
    const std::vector<HAFCacheBlock>& blocks_at_level(size_t j) const {
        return blocks_by_level_[j];
    }

    // For test/verification: S_total must equal sum of all aggregates
    Matrix compute_S_total_from_components() const {
        Matrix total = S_exact_;
        for (const auto& level : blocks_by_level_) {
            for (size_t bi = 0; bi < level.size(); ++bi) {
                // S_agg is precomputed; add it
                // (In full impl: total = linalg::add(total, level[bi].S_agg))
            }
        }
        return total;
    }

    // ---- §4.6: Prefix Caching (Phase 9) ----
    //
    // Export all compressed blocks + S_total as a standalone snapshot.
    // Import loads snapshot blocks into the cache for cross-sequence sharing.

    // Export: extract blocks and S_total into output arrays
    void export_prefix(std::vector<HAFCacheBlock>& out_blocks,
                       Matrix& out_S_total) const {
        out_blocks.clear();
        for (const auto& level : blocks_by_level_) {
            for (const auto& block : level) {
                if (block.n > 0)
                    out_blocks.push_back(block);
            }
        }
        out_S_total = S_total_;
    }

    // Import: load blocks and S_total, replacing all state
    void import_prefix(const std::vector<HAFCacheBlock>& blocks,
                       const Matrix& S_total) {
        reset();
        for (const auto& block : blocks) {
            if (block.n > 0)
                blocks_by_level_[1].push_back(block);
        }
        S_total_ = S_total;
    }

private:
    HAFConfig config_;

    // Exact buffer (ring buffer, §4.1.4)
    Matrix exact_buffer_K_;  // ℝ^{W × d_k}
    Matrix exact_buffer_V_;  // ℝ^{W × d_v}
    size_t exact_ptr_;       // Write pointer
    size_t exact_count_;     // Tokens in buffer
    Matrix S_exact_;         // ℝ^{r_feat × d_v} — Running aggregate for exact buffer

    // Hierarchical blocks (indexed by level j)
    std::vector<std::vector<HAFCacheBlock>> blocks_by_level_;

    // Total aggregate
    Matrix S_total_;         // ℝ^{r_feat × d_v}
};

// ============================================================
// §3.2: Hierarchical Exponential Blocking (HEB)
// ============================================================

// Block size at hierarchy level j (j=1 is first compressed level)
// Level 0: W (exact), Level 1: W, Level 2: 2W, Level 3: 4W, ...
inline size_t heb_block_size(size_t level, size_t W) {
    if (level == 0) return W;
    return W * (1ull << (level - 1));
}

// Rank budget at hierarchy level j
// Level 1: r_max, Level 2: r_max/2, Level 3: r_max/4, ...
inline size_t heb_rank_budget(size_t level, size_t r_max) {
    if (level == 0) return r_max;
    size_t r = r_max / (1ull << (level - 1));
    return std::max(r, (size_t)1);
}

// ============================================================
// §3.4: Adaptive Rank Allocation via Spectral Entropy
// ============================================================
//
// Given eigenvalues λ_1 ≥ ... ≥ λ_{d_k}, compute effective rank:
//   p_i = λ_i / Σ_j λ_j
//   H = -Σ p_i log p_i       [von Neumann entropy]
//   H_max = log d_k
//   r_eff = ⌈d_k · (H / H_max) · β⌉
//   return min(r_eff, r_max)

// Von Neumann entropy of normalized eigenvalue distribution
double spectral_entropy(const Matrix& eigenvalues) {
    assert(eigenvalues.cols() == 1);
    size_t d = eigenvalues.rows();
    double sum = 0.0;
    for (size_t i = 0; i < d; ++i) {
        if (eigenvalues(i, 0) > 0.0) sum += eigenvalues(i, 0);
    }
    if (sum < 1e-15) return 0.0;

    double H = 0.0;
    for (size_t i = 0; i < d; ++i) {
        double p = eigenvalues(i, 0) / sum;
        if (p > 1e-15) {
            H -= p * std::log(p);
        }
    }
    return H;
}

// Compute effective rank from spectral entropy
// β ∈ (0,1]: quality budget (higher = more ranks retained)
// r_max: maximum allowed rank
// r_min: minimum allowed rank (default 1)
size_t adaptive_rank(const Matrix& eigenvalues, double beta,
                     size_t r_max, size_t r_min = 1) {
    assert(eigenvalues.cols() == 1);
    size_t d = eigenvalues.rows();

    double sum = 0.0;
    for (size_t i = 0; i < d; ++i) {
        if (eigenvalues(i, 0) > 0.0) sum += eigenvalues(i, 0);
    }

    // Degenerate case: all eigenvalues are zero
    if (sum < 1e-15) return r_min;

    // Compute von Neumann entropy
    double H = 0.0;
    for (size_t i = 0; i < d; ++i) {
        double p = eigenvalues(i, 0) / sum;
        if (p > 1e-15) {
            H -= p * std::log(p);
        }
    }

    double H_max = std::log((double)d);
    if (H_max < 1e-15) return r_min;

    // Effective rank: proportion of maximum entropy, scaled by budget
    double ratio = H / H_max;
    size_t r_eff = (size_t)std::ceil((double)d * ratio * beta);

    // Clamp to [r_min, r_max]
    if (r_eff < r_min) r_eff = r_min;
    if (r_eff > r_max) r_eff = r_max;
    if (r_eff > d) r_eff = d;

    return r_eff;
}

// ============================================================
// §3.5: Clifford-Geometric Enrichment (CGE)
// ============================================================
//
// Truncated grade-{0,1,2} real Clifford algebra Cℓ_{d_k}^{[0,1,2]}.
// Dimension: 1 + d_k + d_k(d_k-1)/2.
// Applied only to block means and principal directions (not tokens).

// Bivector index: maps (i,j) with i<j to a flat array index.
// Ordering: (0,1), (0,2), ..., (0,d-1), (1,2), (1,3), ..., (d-2,d-1)
inline size_t bivector_index(size_t i, size_t j, size_t d) {
    assert(i < j && j < d);
    return i * (2 * d - i - 1) / 2 + (j - i - 1);
}

inline size_t num_bivectors(size_t d) {
    if (d < 2) return 0;
    return d * (d - 1) / 2;
}

// CliffordMultivector: stores grade-0, grade-1, grade-2 components
struct CliffordMultivector {
    double s;                // grade-0 (scalar)
    std::vector<double> v;   // grade-1 (vector), size d_k
    std::vector<double> b;   // grade-2 (bivector), size d_k*(d_k-1)/2
    size_t d_k;

    CliffordMultivector() : s(0.0), d_k(0) {}

    explicit CliffordMultivector(size_t d)
        : s(0.0), v(d, 0.0), b(num_bivectors(d), 0.0), d_k(d) {}

    // Zero-initialized
    void set_zero() {
        s = 0.0;
        std::fill(v.begin(), v.end(), 0.0);
        std::fill(b.begin(), b.end(), 0.0);
    }
};

// Embed a vector k ∈ ℝ^{d_k} as a grade-1 multivector k̃ ∈ Cℓ_{d_k}^{(1)}
CliffordMultivector embed_multivector(const Matrix& k) {
    assert(k.cols() == 1);
    size_t d = k.rows();
    CliffordMultivector mv(d);
    for (size_t i = 0; i < d; ++i) {
        mv.v[i] = k(i, 0);
    }
    return mv;
}

// Geometric product q̃ k̃:
// Returns the full grade-{0,1,2} result.
// grade-0: q · k = Σ q_i k_i
// grade-1: 0 (for two grade-1 vectors, the geometric product has only
//           grade-0 and grade-2 parts: qk = q·k + q∧k)
// grade-2: q ∧ k where (q∧k)_{ij} = q_i k_j - q_j k_i
CliffordMultivector geometric_product(const CliffordMultivector& q_tilde,
                                       const CliffordMultivector& k_tilde) {
    assert(q_tilde.d_k == k_tilde.d_k);
    size_t d = q_tilde.d_k;
    CliffordMultivector result(d);

    // Grade-0: dot product
    for (size_t i = 0; i < d; ++i) {
        result.s += q_tilde.v[i] * k_tilde.v[i];
    }

    // Grade-2: wedge product (q ∧ k)
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = i + 1; j < d; ++j) {
            size_t idx = bivector_index(i, j, d);
            result.b[idx] = q_tilde.v[i] * k_tilde.v[j]
                          - q_tilde.v[j] * k_tilde.v[i];
        }
    }

    return result;
}

// Extract scalar part (grade-0) of geometric product — equals dot product
double scalar_part(const CliffordMultivector& mv) {
    return mv.s;
}

// Compute geometric uncertainty: σ_geo = ||s̃ - s_0||_F
// = sqrt(Σ_{m≥2} ||s_m||²) = sqrt(Σ_{i<j} (q_i k_j - q_j k_i)²)
double geometric_uncertainty(const CliffordMultivector& mv) {
    double sum_sq = 0.0;
    for (size_t idx = 0; idx < mv.b.size(); ++idx) {
        sum_sq += mv.b[idx] * mv.b[idx];
    }
    return std::sqrt(sum_sq);
}

// Multivector norm: ||q̃||² = ||q||² (metric preservation, Theorem 6.1)
double multivector_norm_sq(const CliffordMultivector& mv) {
    double sum = mv.s * mv.s;
    for (size_t i = 0; i < mv.v.size(); ++i) {
        sum += mv.v[i] * mv.v[i];
    }
    // Note: bivector components don't contribute to the Clifford norm
    // for grade-1 elements (the norm is defined by the inner product)
    // For our truncated algebra, we use the Euclidean norm of the vector part
    return sum;
}

// Verify Theorem 6 properties:
// 1. ||q̃||² = ||q||² (norm preservation)
// 2. ⟨q̃, k̃⟩_0 = q · k (scalar part = dot product)
// 3. ||⟨q̃, k̃⟩_2||² = ||q||²||k||² - (q·k)² (Lagrange identity)

// Compute squared bivector norm: ||q ∧ k||² = Σ_{i<j} (q_i k_j - q_j k_i)²
double bivector_norm_sq(const CliffordMultivector& mv) {
    double sum_sq = 0.0;
    for (size_t idx = 0; idx < mv.b.size(); ++idx) {
        sum_sq += mv.b[idx] * mv.b[idx];
    }
    return sum_sq;
}

// Direct check of Lagrange identity: ||q∧k||² = ||q||²||k||² - (q·k)²
double check_lagrange_identity(const Matrix& q, const Matrix& k) {
    double q_norm_sq = 0.0, k_norm_sq = 0.0, qk_dot = 0.0;
    size_t d = q.rows();
    for (size_t i = 0; i < d; ++i) {
        q_norm_sq += q(i, 0) * q(i, 0);
        k_norm_sq += k(i, 0) * k(i, 0);
        qk_dot += q(i, 0) * k(i, 0);
    }

    double expected = q_norm_sq * k_norm_sq - qk_dot * qk_dot;

    // Compute via bivector
    double biv_sq = 0.0;
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = i + 1; j < d; ++j) {
            double bij = q(i, 0) * k(j, 0) - q(j, 0) * k(i, 0);
            biv_sq += bij * bij;
        }
    }

    return std::abs(biv_sq - expected);
}

// ============================================================
// §3.3: Mergeable Spectral Statistics (MSS)
// ============================================================
//
// Merge(B_A, B_B) → B_C per SPEC §3.3 Steps 1-5.
// Reconstructs full second-order statistics from spectral summaries,
// merges via covariance identities, then recompresses via EVD.
// O(d_k^3 + d_v^3) per merge.

HAFCacheBlock merge_blocks(const HAFCacheBlock& A, const HAFCacheBlock& B,
                           size_t d_k, size_t d_v, size_t r_feat) {
    size_t n_A = A.n, n_B = B.n;
    size_t n_C = n_A + n_B;
    assert(n_A > 0 && n_B > 0);

    // ---- Step 1: Merged Means (§3.3 Step 1) ----
    // μ_k,C = (n_A·μ_k,A + n_B·μ_k,B) / n_C
    Matrix mu_k_C(linalg::add(linalg::scale(A.mu_k, (double)n_A),
                               linalg::scale(B.mu_k, (double)n_B)));
    for (size_t i = 0; i < d_k; ++i) mu_k_C(i, 0) /= n_C;

    Matrix mu_v_C(linalg::add(linalg::scale(A.mu_v, (double)n_A),
                               linalg::scale(B.mu_v, (double)n_B)));
    for (size_t i = 0; i < d_v; ++i) mu_v_C(i, 0) /= n_C;

    // ---- Step 2-3: Merged Key Covariance (§3.3 Steps 2-3) ----
    // Reconstruct Σ_A = U_k,A Λ_k,A U_k,A^T ∈ ℝ^{d_k × d_k}
    Matrix L_kA = linalg::diag(A.Lambda_k);
    Matrix Sigma_kA = linalg::matmul(A.U_k, linalg::matmul(L_kA, linalg::transpose(A.U_k)));
    Matrix Sigma_kB = linalg::matmul(B.U_k,
                     linalg::matmul(linalg::diag(B.Lambda_k), linalg::transpose(B.U_k)));

    // Merged: n_C·Σ_k,C = n_A(Σ_k,A + μ_A μ_A^T) + n_B(Σ_k,B + μ_B μ_B^T) - n_C·μ_C μ_C^T
    Matrix Sigma_kC(linalg::add(
        linalg::scale(Sigma_kA, (double)n_A),
        linalg::scale(linalg::outer(A.mu_k, A.mu_k), (double)n_A)));
    Matrix temp_kB(linalg::add(
        linalg::scale(Sigma_kB, (double)n_B),
        linalg::scale(linalg::outer(B.mu_k, B.mu_k), (double)n_B)));
    Sigma_kC = linalg::add(Sigma_kC, temp_kB);
    for (size_t i = 0; i < d_k; ++i)
        for (size_t j = 0; j < d_k; ++j)
            Sigma_kC(i, j) = Sigma_kC(i, j) / n_C - mu_k_C(i, 0) * mu_k_C(j, 0);

    // Eigendecomposition of merged key covariance
    auto evd_k = linalg::symmetric_eigendecompose(Sigma_kC);
    size_t r_kC = std::max(A.r_k(), B.r_k()) / 2;
    r_kC = std::min(r_kC, d_k - 1);
    r_kC = std::max(r_kC, (size_t)1);

    // ---- Step 4: Merged Value Covariance (§3.3 Step 4) ----
    Matrix L_vA = linalg::diag(A.Lambda_v);
    Matrix Sigma_vA = linalg::matmul(A.U_v, linalg::matmul(L_vA, linalg::transpose(A.U_v)));
    Matrix Sigma_vB = linalg::matmul(B.U_v,
                     linalg::matmul(linalg::diag(B.Lambda_v), linalg::transpose(B.U_v)));

    Matrix Sigma_vC(linalg::add(
        linalg::scale(Sigma_vA, (double)n_A),
        linalg::scale(linalg::outer(A.mu_v, A.mu_v), (double)n_A)));
    Matrix temp_vB(linalg::add(
        linalg::scale(Sigma_vB, (double)n_B),
        linalg::scale(linalg::outer(B.mu_v, B.mu_v), (double)n_B)));
    Sigma_vC = linalg::add(Sigma_vC, temp_vB);
    for (size_t i = 0; i < d_v; ++i)
        for (size_t j = 0; j < d_v; ++j)
            Sigma_vC(i, j) = Sigma_vC(i, j) / n_C - mu_v_C(i, 0) * mu_v_C(j, 0);

    auto evd_v = linalg::symmetric_eigendecompose(Sigma_vC);
    size_t r_vC = std::max(A.r_v(), B.r_v()) / 2;
    r_vC = std::min(r_vC, d_v - 1);
    r_vC = std::max(r_vC, (size_t)1);

    // ---- Step 5: Merged Cross-Covariance (§3.3 Step 5) ----
    // Σ_kv,A = U_k,A C_kv,A U_v,A^T ∈ ℝ^{d_k × d_v}
    Matrix Sigma_kvA = linalg::matmul(A.U_k,
                       linalg::matmul(A.C_kv, linalg::transpose(A.U_v)));
    Matrix Sigma_kvB = linalg::matmul(B.U_k,
                       linalg::matmul(B.C_kv, linalg::transpose(B.U_v)));

    // Merged cross-covariance
    Matrix Sigma_kvC(linalg::add(
        linalg::scale(Sigma_kvA, (double)n_A),
        linalg::scale(linalg::outer(A.mu_k, A.mu_v), (double)n_A)));
    Matrix temp_kvB(linalg::add(
        linalg::scale(Sigma_kvB, (double)n_B),
        linalg::scale(linalg::outer(B.mu_k, B.mu_v), (double)n_B)));
    Sigma_kvC = linalg::add(Sigma_kvC, temp_kvB);
    for (size_t i = 0; i < d_k; ++i)
        for (size_t j = 0; j < d_v; ++j)
            Sigma_kvC(i, j) = Sigma_kvC(i, j) / n_C - mu_k_C(i, 0) * mu_v_C(j, 0);

    // Project to latent space: C_kv,C = U_k,C^T Σ_kv,C U_v,C ∈ ℝ^{r_kC × r_vC}
    Matrix U_kCT = linalg::transpose(evd_k.eigenvectors.submatrix(0, d_k, 0, r_kC));
    Matrix U_vC = evd_v.eigenvectors.submatrix(0, d_v, 0, r_vC);
    Matrix C_kvC = linalg::matmul(U_kCT, linalg::matmul(Sigma_kvC, U_vC));

    // ---- Construct result block ----
    HAFCacheBlock result(d_k, d_v, r_kC, r_vC, r_feat);
    result.n = n_C;
    result.mu_k = mu_k_C;
    result.mu_v = mu_v_C;

    // Copy truncated eigenvectors and eigenvalues
    for (size_t i = 0; i < d_k; ++i)
        for (size_t j = 0; j < r_kC; ++j)
            result.U_k(i, j) = evd_k.eigenvectors(i, j);
    for (size_t i = 0; i < r_kC; ++i)
        result.Lambda_k(i, 0) = evd_k.eigenvalues(i, 0);

    for (size_t i = 0; i < d_v; ++i)
        for (size_t j = 0; j < r_vC; ++j)
            result.U_v(i, j) = evd_v.eigenvectors(i, j);
    for (size_t i = 0; i < r_vC; ++i)
        result.Lambda_v(i, 0) = evd_v.eigenvalues(i, 0);

    result.C_kv = C_kvC;

    // S_agg is computed externally via block compression protocol
    // (Phase 5) — left as zero here, to be filled by ComputeBlockStats

    return result;
}

// ============================================================
// §4.4: Block Compression Protocol
// ============================================================

// Feature map φ(x) = elu(x) + 1 (element-wise, ℝ^{d} → ℝ^{d})
// ELU: elu(x) = x if x ≥ 0, else exp(x) - 1
// φ(x) = elu(x) + 1 ensures φ(x) > 0 for all x (useful for attention)
inline double elu(double x) {
    return (x >= 0.0) ? x : std::expm1(x);
}

inline double elu_prime(double x) {
    return (x >= 0.0) ? 1.0 : std::exp(x);
}

// Apply φ to each element of vector v ∈ ℝ^{d}
Matrix phi(const Matrix& v) {
    assert(v.cols() == 1);
    size_t d = v.rows();
    Matrix result(d, 1);
    for (size_t i = 0; i < d; ++i) {
        result(i, 0) = elu(v(i, 0)) + 1.0;
    }
    return result;
}

// Jacobian of φ at v: diagonal matrix diag(φ'(v_i))
Matrix phi_jacobian_diag(const Matrix& v) {
    assert(v.cols() == 1);
    size_t d = v.rows();
    Matrix diag_vals(d, 1);
    for (size_t i = 0; i < d; ++i) {
        diag_vals(i, 0) = elu_prime(v(i, 0));
    }
    return diag_vals;  // return as column vector; use as diag(diag_vals)
}

// ComputeBlockStats(K_block, V_block) → HAFCacheBlock
// Implements SPEC §4.4 Steps 1-7
HAFCacheBlock compute_block_stats(const Matrix& K_block, const Matrix& V_block,
                                   const HAFConfig& config) {
    size_t B = K_block.rows();
    size_t d_k = K_block.cols();
    size_t d_v = V_block.cols();
    assert(B == V_block.rows());
    assert(d_k == config.d_k && d_v == config.d_v);

    // Step 1: Compute means
    Matrix mu_k(d_k, 1);
    Matrix mu_v(d_v, 1);
    for (size_t i = 0; i < B; ++i) {
        for (size_t j = 0; j < d_k; ++j) mu_k(j, 0) += K_block(i, j);
        for (size_t j = 0; j < d_v; ++j) mu_v(j, 0) += V_block(i, j);
    }
    for (size_t j = 0; j < d_k; ++j) mu_k(j, 0) /= B;
    for (size_t j = 0; j < d_v; ++j) mu_v(j, 0) /= B;

    // Step 2: Center data
    Matrix K_tilde(B, d_k);
    Matrix V_tilde(B, d_v);
    for (size_t i = 0; i < B; ++i) {
        for (size_t j = 0; j < d_k; ++j) K_tilde(i, j) = K_block(i, j) - mu_k(j, 0);
        for (size_t j = 0; j < d_v; ++j) V_tilde(i, j) = V_block(i, j) - mu_v(j, 0);
    }

    // Step 3: Key covariance and eigendecomposition
    Matrix Sigma_kk = linalg::scale(
        linalg::matmul(linalg::transpose(K_tilde), K_tilde), 1.0 / B);
    auto evd_k = linalg::symmetric_eigendecompose(Sigma_kk);
    size_t r_k = adaptive_rank(evd_k.eigenvalues, config.beta, config.r_max);

    Matrix U_k = evd_k.eigenvectors.submatrix(0, d_k, 0, r_k);
    Matrix Lambda_k(r_k, 1);
    for (size_t i = 0; i < r_k; ++i) Lambda_k(i, 0) = evd_k.eigenvalues(i, 0);

    // Step 4: Value covariance and eigendecomposition
    Matrix Sigma_vv = linalg::scale(
        linalg::matmul(linalg::transpose(V_tilde), V_tilde), 1.0 / B);
    auto evd_v = linalg::symmetric_eigendecompose(Sigma_vv);
    size_t r_v = adaptive_rank(evd_v.eigenvalues, config.beta, config.r_max);

    Matrix U_v = evd_v.eigenvectors.submatrix(0, d_v, 0, r_v);
    Matrix Lambda_v(r_v, 1);
    for (size_t i = 0; i < r_v; ++i) Lambda_v(i, 0) = evd_v.eigenvalues(i, 0);

    // Step 5: Cross-covariance in latent space
    // C_kv = (1/B) (K̃ U_k)^T (Ṽ U_v) ∈ ℝ^{r_k × r_v}
    Matrix K_tilde_U = linalg::matmul(K_tilde, U_k);
    Matrix V_tilde_U = linalg::matmul(V_tilde, U_v);
    Matrix C_kv = linalg::scale(
        linalg::matmul(linalg::transpose(K_tilde_U), V_tilde_U), 1.0 / B);

    // Step 6: Precompute linear attention aggregate S_agg
    // Exact computation: S_agg = Σ_{i=1}^B φ(K_block[i]) ⊗ V_block[i]^T
    Matrix S_agg(config.r_feat, d_v);
    for (size_t i = 0; i < B; ++i) {
        Matrix k_i = K_block.submatrix(i, i + 1, 0, d_k);
        Matrix v_i = V_block.submatrix(i, i + 1, 0, d_v);
        Matrix phi_k = phi(linalg::transpose(k_i));  // ℝ^{d_k × 1}
        // φ(k_i) ⊗ v_i^T: outer product
        for (size_t a = 0; a < config.r_feat; ++a) {
            for (size_t b = 0; b < d_v; ++b) {
                S_agg(a, b) += phi_k(a, 0) * v_i(0, b);
            }
        }
    }

    // Step 7: Construct and return block
    HAFCacheBlock block(d_k, d_v, r_k, r_v, config.r_feat);
    block.n = B;
    block.mu_k = mu_k;
    block.mu_v = mu_v;
    block.U_k = U_k;
    block.Lambda_k = Lambda_k;
    block.U_v = U_v;
    block.Lambda_v = Lambda_v;
    block.C_kv = C_kv;
    block.S_agg = S_agg;

    return block;
}

// ============================================================
// Mathematical Verification Utilities
// ============================================================

// Verify matrix reconstruction error bound (Eckart-Young)
// For a matrix block K centered at mean μ with EVD Σ = UΛU^T,
// the best rank-r approximation error is Σ_{i=r+1}^{d_k} λ_i
double compute_reconstruction_error_bound(const Matrix& Lambda_full,
                                           size_t r) {
    assert(Lambda_full.cols() == 1);
    size_t d = Lambda_full.rows();
    if (r >= d) return 0.0;
    double error = 0.0;
    for (size_t i = r; i < d; ++i) {
        if (Lambda_full(i, 0) > 0.0) {
            error += Lambda_full(i, 0);
        }
    }
    return error;
}

// Compute empirical covariance: Σ = (1/n) * X^T * X  (centered)
// where X is ℝ^{n×d}, each row is a sample
Matrix compute_covariance(const Matrix& X) {
    size_t n = X.rows();
    size_t d = X.cols();
    assert(n > 1);

    // Compute mean
    Matrix mu(d, 1);
    for (size_t j = 0; j < d; ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) {
            sum += X(i, j);
        }
        mu(j, 0) = sum / n;
    }

    // Center data and compute X̃^T X̃
    Matrix Sigma(d, d);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            double xij_centered = X(i, j) - mu(j, 0);
            for (size_t k = j; k < d; ++k) {
                double xik_centered = X(i, k) - mu(k, 0);
                Sigma(j, k) += xij_centered * xik_centered;
            }
        }
    }

    // Fill symmetry and scale
    double inv_n = 1.0 / n;
    for (size_t j = 0; j < d; ++j) {
        for (size_t k = j; k < d; ++k) {
            Sigma(j, k) *= inv_n;
            Sigma(k, j) = Sigma(j, k);
        }
    }

    return Sigma;
}

// Verify EVD accuracy: ||A - V Λ V^T||_F
double check_evd_accuracy(const Matrix& A,
                          const Matrix& eigenvectors,
                          const Matrix& eigenvalues) {
    size_t n = A.rows();
    Matrix recon(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += eigenvectors(i, k) * eigenvalues(k, 0) * eigenvectors(j, k);
            }
            recon(i, j) = sum;
        }
    }

    double error_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double diff = A(i, j) - recon(i, j);
            error_sq += diff * diff;
        }
    }
    return std::sqrt(error_sq);
}

// Verify orthogonality: ||V^T V - I||_F
double check_orthogonality(const Matrix& eigenvectors) {
    size_t n = eigenvectors.rows();
    Matrix VTV = linalg::matmul(linalg::transpose(eigenvectors), eigenvectors);
    double error_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            double diff = VTV(i, j) - expected;
            error_sq += diff * diff;
        }
    }
    return std::sqrt(error_sq);
}

// Verify positive semidefiniteness: all eigenvalues ≥ -tol
bool check_psd(const Matrix& eigenvalues, double tol = 1e-10) {
    for (size_t i = 0; i < eigenvalues.rows(); ++i) {
        if (eigenvalues(i, 0) < -tol) return false;
    }
    return true;
}

} // namespace haf_age
