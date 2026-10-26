#pragma once
#include "tensor.hpp"

namespace nn {

// Numerically stable row-wise softmax.
inline Matrix softmax(const Matrix& z) {
    Matrix p(z.rows, z.cols);
    for (size_t i = 0; i < z.rows; ++i) {
        float mx = z(i, 0);
        for (size_t j = 1; j < z.cols; ++j) mx = std::max(mx, z(i, j));
        float sum = 0.f;
        for (size_t j = 0; j < z.cols; ++j) { p(i, j) = std::exp(z(i, j) - mx); sum += p(i, j); }
        for (size_t j = 0; j < z.cols; ++j) p(i, j) /= sum;
    }
    return p;
}

class Loss {
public:
    virtual ~Loss() = default;
    virtual float forward(const Matrix& pred, const Matrix& target) = 0;
    virtual Matrix backward() = 0;
};

// Softmax + cross-entropy on raw logits, one-hot targets.
class SoftmaxCrossEntropy : public Loss {
    Matrix p_, t_;
public:
    float forward(const Matrix& logits, const Matrix& target) override {
        p_ = softmax(logits);
        t_ = target;
        double l = 0.0;
        for (size_t i = 0; i < p_.size(); ++i)
            if (t_.data[i] > 0.f) l -= t_.data[i] * std::log(p_.data[i] + 1e-12f);
        return (float)(l / (double)p_.rows);
    }
    Matrix backward() override {
        Matrix g = p_;
        const float inv = 1.f / (float)p_.rows;
        for (size_t i = 0; i < g.size(); ++i) g.data[i] = (g.data[i] - t_.data[i]) * inv;
        return g;
    }
};

class MSELoss : public Loss {
    Matrix p_, t_;
public:
    float forward(const Matrix& pred, const Matrix& target) override {
        p_ = pred; t_ = target;
        double l = 0.0;
        for (size_t i = 0; i < p_.size(); ++i) { double d = p_.data[i] - t_.data[i]; l += d * d; }
        return (float)(l / (double)p_.size());
    }
    Matrix backward() override {
        Matrix g = p_;
        const float k = 2.f / (float)p_.size();
        for (size_t i = 0; i < g.size(); ++i) g.data[i] = (p_.data[i] - t_.data[i]) * k;
        return g;
    }
};

} // namespace nn
