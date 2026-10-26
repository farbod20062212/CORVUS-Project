#pragma once
#include "layers.hpp"
#include <fstream>
#include <cstdint>

namespace nn {

class Optimizer {
public:
    float lr;
    explicit Optimizer(float lr_) : lr(lr_) {}
    virtual ~Optimizer() = default;
    virtual void step(std::vector<Param>& ps) = 0;
};

class SGD : public Optimizer {
    float momentum_, wd_;
    std::vector<std::vector<float>> vel_;
public:
    SGD(float lr, float momentum = 0.9f, float weightDecay = 0.f)
        : Optimizer(lr), momentum_(momentum), wd_(weightDecay) {}
    void step(std::vector<Param>& ps) override {
        if (vel_.size() != ps.size()) {
            vel_.assign(ps.size(), {});
            for (size_t i = 0; i < ps.size(); ++i) vel_[i].assign(ps[i].value->size(), 0.f);
        }
        for (size_t i = 0; i < ps.size(); ++i) {
            auto& w = ps[i].value->data; auto& g = ps[i].grad->data; auto& v = vel_[i];
            for (size_t k = 0; k < w.size(); ++k) {
                float grad = g[k] + (ps[i].decay ? wd_ * w[k] : 0.f);
                v[k] = momentum_ * v[k] - lr * grad;
                w[k] += v[k];
            }
        }
    }
};

// Adam with decoupled weight decay (AdamW).
class Adam : public Optimizer {
    float b1_, b2_, eps_, wd_;
    long long t_ = 0;
    std::vector<std::vector<float>> m_, v_;
public:
    Adam(float lr, float weightDecay = 0.f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
        : Optimizer(lr), b1_(b1), b2_(b2), eps_(eps), wd_(weightDecay) {}

    // Persist moments + step count so long training runs can be resumed exactly.
    void saveState(const std::string& path) const {
        std::ofstream o(path, std::ios::binary); if (!o) return;
        uint64_t n = m_.size(); o.write((const char*)&t_, sizeof(t_)); o.write((const char*)&n, sizeof(n));
        for (size_t i = 0; i < m_.size(); ++i) {
            uint64_t k = m_[i].size(); o.write((const char*)&k, sizeof(k));
            o.write((const char*)m_[i].data(), k * sizeof(float)); o.write((const char*)v_[i].data(), k * sizeof(float));
        }
    }
    bool loadState(const std::string& path, const std::vector<Param>& ps) {
        std::ifstream i(path, std::ios::binary); if (!i) return false;
        long long t; uint64_t n; i.read((char*)&t, sizeof(t)); i.read((char*)&n, sizeof(n));
        if (n != ps.size()) return false;
        std::vector<std::vector<float>> m(n), v(n);
        for (size_t k = 0; k < n; ++k) {
            uint64_t sz; i.read((char*)&sz, sizeof(sz));
            if (sz != ps[k].value->size()) return false;
            m[k].resize(sz); v[k].resize(sz);
            i.read((char*)m[k].data(), sz * sizeof(float)); i.read((char*)v[k].data(), sz * sizeof(float));
        }
        if (!i) return false;
        t_ = t; m_.swap(m); v_.swap(v); return true;
    }
    void step(std::vector<Param>& ps) override {
        if (m_.size() != ps.size()) {
            m_.assign(ps.size(), {}); v_.assign(ps.size(), {});
            for (size_t i = 0; i < ps.size(); ++i) {
                m_[i].assign(ps[i].value->size(), 0.f);
                v_[i].assign(ps[i].value->size(), 0.f);
            }
        }
        ++t_;
        const float c1 = 1.f - std::pow(b1_, (float)t_);
        const float c2 = 1.f - std::pow(b2_, (float)t_);
        for (size_t i = 0; i < ps.size(); ++i) {
            auto& w = ps[i].value->data; auto& g = ps[i].grad->data;
            for (size_t k = 0; k < w.size(); ++k) {
                m_[i][k] = b1_ * m_[i][k] + (1.f - b1_) * g[k];
                v_[i][k] = b2_ * v_[i][k] + (1.f - b2_) * g[k] * g[k];
                float mh = m_[i][k] / c1, vh = v_[i][k] / c2;
                if (ps[i].decay) w[k] -= lr * wd_ * w[k];
                w[k] -= lr * mh / (std::sqrt(vh) + eps_);
            }
        }
    }
};

} // namespace nn
