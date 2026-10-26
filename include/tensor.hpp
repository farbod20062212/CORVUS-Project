#pragma once
// Matrix + matrix multiplication kernels (row-major, float).
// Kernels are register-blocked (4 rows at a time) so each loaded element is reused 4x,
// auto-vectorized by the compiler (-O3 -march=native) and parallelized with OpenMP.
#include <vector>
#include <cstddef>
#include <algorithm>
#include <cmath>

namespace nn {

struct Matrix {
    size_t rows = 0, cols = 0;
    std::vector<float> data;

    Matrix() = default;
    Matrix(size_t r, size_t c, float v = 0.f) : rows(r), cols(c), data(r * c, v) {}

    float& operator()(size_t r, size_t c) { return data[r * cols + c]; }
    float operator()(size_t r, size_t c) const { return data[r * cols + c]; }
    size_t size() const { return data.size(); }
    void resize(size_t r, size_t c) { rows = r; cols = c; data.assign(r * c, 0.f); }
    void zero() { std::fill(data.begin(), data.end(), 0.f); }
};

// C = A * B      (A: n x k, B: k x m)
inline void matmul(const Matrix& A, const Matrix& B, Matrix& C) {
    C.resize(A.rows, B.cols);
    const size_t n = A.rows, K = A.cols, M = B.cols;
    const long long nb = (long long)((n + 3) / 4);
    #pragma omp parallel for schedule(static)
    for (long long bi = 0; bi < nb; ++bi) {
        const size_t i0 = (size_t)bi * 4, rem = std::min<size_t>(4, n - i0);
        if (rem == 4) {
            float* c0 = &C.data[i0 * M]; float* c1 = c0 + M; float* c2 = c1 + M; float* c3 = c2 + M;
            const float* a0 = &A.data[i0 * K]; const float* a1 = a0 + K; const float* a2 = a1 + K; const float* a3 = a2 + K;
            for (size_t k = 0; k < K; ++k) {
                const float x0 = a0[k], x1 = a1[k], x2 = a2[k], x3 = a3[k];
                const float* b = &B.data[k * M];
                #pragma omp simd
                for (size_t j = 0; j < M; ++j) {
                    const float bj = b[j];
                    c0[j] += x0 * bj; c1[j] += x1 * bj; c2[j] += x2 * bj; c3[j] += x3 * bj;
                }
            }
        } else {
            for (size_t r = 0; r < rem; ++r) {
                float* c = &C.data[(i0 + r) * M]; const float* a = &A.data[(i0 + r) * K];
                for (size_t k = 0; k < K; ++k) {
                    const float x = a[k]; const float* b = &B.data[k * M];
                    #pragma omp simd
                    for (size_t j = 0; j < M; ++j) c[j] += x * b[j];
                }
            }
        }
    }
}

// C = A^T * B    (A: n x p, B: n x q) -> p x q
inline void matmulAtB(const Matrix& A, const Matrix& B, Matrix& C) {
    C.resize(A.cols, B.cols);
    const size_t n = A.rows, P = A.cols, Q = B.cols;
    const long long nb = (long long)((P + 3) / 4);
    #pragma omp parallel for schedule(static)
    for (long long bi = 0; bi < nb; ++bi) {
        const size_t p0 = (size_t)bi * 4, rem = std::min<size_t>(4, P - p0);
        if (rem == 4) {
            float* c0 = &C.data[p0 * Q]; float* c1 = c0 + Q; float* c2 = c1 + Q; float* c3 = c2 + Q;
            for (size_t r = 0; r < n; ++r) {
                const float* a = &A.data[r * P + p0];
                const float x0 = a[0], x1 = a[1], x2 = a[2], x3 = a[3];
                const float* b = &B.data[r * Q];
                #pragma omp simd
                for (size_t j = 0; j < Q; ++j) {
                    const float bj = b[j];
                    c0[j] += x0 * bj; c1[j] += x1 * bj; c2[j] += x2 * bj; c3[j] += x3 * bj;
                }
            }
        } else {
            for (size_t q = 0; q < rem; ++q) {
                float* c = &C.data[(p0 + q) * Q];
                for (size_t r = 0; r < n; ++r) {
                    const float x = A.data[r * P + p0 + q]; const float* b = &B.data[r * Q];
                    #pragma omp simd
                    for (size_t j = 0; j < Q; ++j) c[j] += x * b[j];
                }
            }
        }
    }
}

// C = A * B^T    (A: n x k, B: m x k) -> n x m
inline void matmulABt(const Matrix& A, const Matrix& B, Matrix& C) {
    C.resize(A.rows, B.rows);
    const size_t n = A.rows, K = A.cols, M = B.rows;
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)n; ++i) {
        const float* a = &A.data[(size_t)i * K]; float* c = &C.data[(size_t)i * M];
        size_t j = 0;
        for (; j + 4 <= M; j += 4) {
            const float* b0 = &B.data[j * K]; const float* b1 = b0 + K; const float* b2 = b1 + K; const float* b3 = b2 + K;
            float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
            #pragma omp simd reduction(+ : s0, s1, s2, s3)
            for (size_t k = 0; k < K; ++k) { const float x = a[k]; s0 += x * b0[k]; s1 += x * b1[k]; s2 += x * b2[k]; s3 += x * b3[k]; }
            c[j] = s0; c[j + 1] = s1; c[j + 2] = s2; c[j + 3] = s3;
        }
        for (; j < M; ++j) {
            const float* b = &B.data[j * K]; float s = 0.f;
            #pragma omp simd reduction(+ : s)
            for (size_t k = 0; k < K; ++k) s += a[k] * b[k];
            c[j] = s;
        }
    }
}

} // namespace nn
