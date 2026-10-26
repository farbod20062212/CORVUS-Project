#pragma once
// Layers: Dense, Activation, Dropout.  Each implements forward + backward (manual backprop).
#include "tensor.hpp"
#include <random>
#include <memory>
#include <string>

namespace nn {

struct Param {
    Matrix* value;
    Matrix* grad;
    bool decay;   // apply weight decay (true for weights, false for biases)
};

class Layer {
public:
    virtual ~Layer() = default;
    virtual Matrix forward(const Matrix& x, bool training) = 0;
    virtual Matrix backward(const Matrix& grad) = 0;
    virtual std::vector<Param> params() { return {}; }
    // Non-trainable state that must be saved with the model (e.g. BatchNorm running statistics).
    virtual std::vector<Matrix*> buffers() { return {}; }
    virtual std::string name() const { return "Layer"; }
    virtual std::string describe() const { return name(); }
    size_t paramCount() { size_t n = 0; for (auto& p : params()) n += p.value->size(); return n; }
};

class Dense : public Layer {
    Matrix W, b, dW, db, x_;
public:
    Dense(size_t in, size_t out, float stddev, std::mt19937& rng)
        : W(in, out), b(1, out), dW(in, out), db(1, out) {
        std::normal_distribution<float> nd(0.f, stddev);
        for (auto& w : W.data) w = nd(rng);
    }
    Matrix forward(const Matrix& x, bool) override {
        x_ = x;
        Matrix y;
        matmul(x, W, y);
        for (size_t i = 0; i < y.rows; ++i)
            for (size_t j = 0; j < y.cols; ++j) y(i, j) += b(0, j);
        return y;
    }
    Matrix backward(const Matrix& g) override {
        matmulAtB(x_, g, dW);
        db.zero();
        for (size_t i = 0; i < g.rows; ++i)
            for (size_t j = 0; j < g.cols; ++j) db(0, j) += g(i, j);
        Matrix dx;
        matmulABt(g, W, dx);
        return dx;
    }
    std::vector<Param> params() override { return {{&W, &dW, true}, {&b, &db, false}}; }
    std::string name() const override { return "Dense"; }
    std::string describe() const override { return "Dense " + std::to_string(W.rows) + " -> " + std::to_string(W.cols); }
};

enum class Act { ReLU = 0, LeakyReLU = 1, Tanh = 2, Sigmoid = 3 };

class Activation : public Layer {
    Act type_;
    Matrix in_, out_;
public:
    explicit Activation(Act t) : type_(t) {}
    std::string name() const override { return "Activation"; }
    std::string describe() const override { static const char* n[] = {"ReLU", "LeakyReLU", "Tanh", "Sigmoid"}; return n[(int)type_]; }
    Matrix forward(const Matrix& x, bool) override {
        in_ = x;
        out_ = x;
        for (auto& v : out_.data) {
            switch (type_) {
                case Act::ReLU:      v = v > 0.f ? v : 0.f; break;
                case Act::LeakyReLU: v = v > 0.f ? v : 0.01f * v; break;
                case Act::Tanh:      v = std::tanh(v); break;
                case Act::Sigmoid:   v = 1.f / (1.f + std::exp(-v)); break;
            }
        }
        return out_;
    }
    Matrix backward(const Matrix& g) override {
        Matrix dx = g;
        for (size_t i = 0; i < dx.size(); ++i) {
            float x = in_.data[i], y = out_.data[i], d = 1.f;
            switch (type_) {
                case Act::ReLU:      d = x > 0.f ? 1.f : 0.f; break;
                case Act::LeakyReLU: d = x > 0.f ? 1.f : 0.01f; break;
                case Act::Tanh:      d = 1.f - y * y; break;
                case Act::Sigmoid:   d = y * (1.f - y); break;
            }
            dx.data[i] *= d;
        }
        return dx;
    }
};

class Dropout : public Layer {
    float p_;
    std::mt19937 rng_;
    Matrix mask_;
public:
    Dropout(float p, unsigned seed) : p_(p), rng_(seed) {}
    std::string name() const override { return "Dropout"; }
    std::string describe() const override { return "Dropout p=" + std::to_string(p_).substr(0, 4); }
    Matrix forward(const Matrix& x, bool training) override {
        if (!training || p_ <= 0.f) return x;
        std::bernoulli_distribution keep(1.0 - p_);
        mask_ = Matrix(x.rows, x.cols);
        Matrix y = x;
        const float scale = 1.f / (1.f - p_);
        for (size_t i = 0; i < y.size(); ++i) {
            mask_.data[i] = keep(rng_) ? scale : 0.f;
            y.data[i] *= mask_.data[i];
        }
        return y;
    }
    Matrix backward(const Matrix& g) override {
        if (mask_.size() != g.size()) return g;
        Matrix dx = g;
        for (size_t i = 0; i < dx.size(); ++i) dx.data[i] *= mask_.data[i];
        return dx;
    }
};


// ---------------------------------------------------------------------------
// Conv2D: input/output are flattened (batch x C*H*W), layout CHW. Uses im2col + matmul.
// ---------------------------------------------------------------------------
class Conv2D : public Layer {
    int inC_, H_, W_, outC_, k_, s_, p_, oh_, ow_;
    Matrix Wt_, b_, dW_, db_, cols_;
    size_t batch_ = 0;
public:
    Conv2D(int inC, int H, int W, int outC, int k, int stride, int pad, std::mt19937& rng)
        : inC_(inC), H_(H), W_(W), outC_(outC), k_(k), s_(stride), p_(pad) {
        oh_ = (H + 2 * pad - k) / stride + 1;
        ow_ = (W + 2 * pad - k) / stride + 1;
        const size_t K = (size_t)inC * k * k;
        Wt_ = Matrix(K, outC); dW_ = Matrix(K, outC);
        b_ = Matrix(1, outC);  db_ = Matrix(1, outC);
        std::normal_distribution<float> nd(0.f, std::sqrt(2.f / (float)K));
        for (auto& w : Wt_.data) w = nd(rng);
    }
    int outH() const { return oh_; }
    int outW() const { return ow_; }
    std::string name() const override { return "Conv2D"; }
    std::string describe() const override { return "Conv2D " + std::to_string(inC_) + "->" + std::to_string(outC_) + " k" + std::to_string(k_) + " s" + std::to_string(s_) + " out " + std::to_string(oh_) + "x" + std::to_string(ow_); }

    Matrix forward(const Matrix& x, bool) override {
        const long long B = (long long)x.rows; batch_ = (size_t)B;
        const size_t K = (size_t)inC_ * k_ * k_, P = (size_t)oh_ * ow_;
        cols_.resize((size_t)B * P, K);
        #pragma omp parallel for schedule(static)
        for (long long bi = 0; bi < B; ++bi) {
            const float* xin = &x.data[(size_t)bi * inC_ * H_ * W_];
            for (int oy = 0; oy < oh_; ++oy)
                for (int ox = 0; ox < ow_; ++ox) {
                    float* row = &cols_.data[((size_t)bi * P + (size_t)oy * ow_ + ox) * K];
                    size_t idx = 0;
                    for (int c = 0; c < inC_; ++c)
                        for (int ky = 0; ky < k_; ++ky)
                            for (int kx = 0; kx < k_; ++kx) {
                                int iy = oy * s_ - p_ + ky, ix = ox * s_ - p_ + kx;
                                row[idx++] = (iy >= 0 && iy < H_ && ix >= 0 && ix < W_)
                                                 ? xin[((size_t)c * H_ + iy) * W_ + ix] : 0.f;
                            }
                }
        }
        Matrix yp; matmul(cols_, Wt_, yp);                 // (B*P) x outC
        Matrix y((size_t)B, (size_t)outC_ * P);
        #pragma omp parallel for schedule(static)
        for (long long bi = 0; bi < B; ++bi)
            for (size_t pos = 0; pos < P; ++pos)
                for (int c = 0; c < outC_; ++c)
                    y.data[(size_t)bi * outC_ * P + (size_t)c * P + pos] =
                        yp.data[((size_t)bi * P + pos) * outC_ + c] + b_.data[c];
        return y;
    }

    Matrix backward(const Matrix& g) override {
        const long long B = (long long)batch_;
        const size_t K = (size_t)inC_ * k_ * k_, P = (size_t)oh_ * ow_;
        Matrix gp((size_t)B * P, outC_);
        #pragma omp parallel for schedule(static)
        for (long long bi = 0; bi < B; ++bi)
            for (size_t pos = 0; pos < P; ++pos)
                for (int c = 0; c < outC_; ++c)
                    gp.data[((size_t)bi * P + pos) * outC_ + c] =
                        g.data[(size_t)bi * outC_ * P + (size_t)c * P + pos];
        matmulAtB(cols_, gp, dW_);                          // K x outC
        db_.zero();
        for (size_t r = 0; r < gp.rows; ++r)
            for (int c = 0; c < outC_; ++c) db_.data[c] += gp.data[r * outC_ + c];
        Matrix dcols; matmulABt(gp, Wt_, dcols);            // (B*P) x K
        Matrix dx((size_t)B, (size_t)inC_ * H_ * W_);
        #pragma omp parallel for schedule(static)
        for (long long bi = 0; bi < B; ++bi) {
            float* out = &dx.data[(size_t)bi * inC_ * H_ * W_];
            for (int oy = 0; oy < oh_; ++oy)
                for (int ox = 0; ox < ow_; ++ox) {
                    const float* row = &dcols.data[((size_t)bi * P + (size_t)oy * ow_ + ox) * K];
                    size_t idx = 0;
                    for (int c = 0; c < inC_; ++c)
                        for (int ky = 0; ky < k_; ++ky)
                            for (int kx = 0; kx < k_; ++kx, ++idx) {
                                int iy = oy * s_ - p_ + ky, ix = ox * s_ - p_ + kx;
                                if (iy >= 0 && iy < H_ && ix >= 0 && ix < W_)
                                    out[((size_t)c * H_ + iy) * W_ + ix] += row[idx];
                            }
                }
        }
        return dx;
    }
    std::vector<Param> params() override { return {{&Wt_, &dW_, true}, {&b_, &db_, false}}; }
};

// ---------------------------------------------------------------------------
// MaxPool2D (k x k, stride k)
// ---------------------------------------------------------------------------
class MaxPool2D : public Layer {
    int C_, H_, W_, k_, oh_, ow_;
    std::vector<int> arg_;
    size_t batch_ = 0;
public:
    MaxPool2D(int C, int H, int W, int k) : C_(C), H_(H), W_(W), k_(k), oh_(H / k), ow_(W / k) {}
    std::string name() const override { return "MaxPool2D"; }
    std::string describe() const override { return "MaxPool2D " + std::to_string(k_) + "x" + std::to_string(k_) + " out " + std::to_string(oh_) + "x" + std::to_string(ow_); }
    Matrix forward(const Matrix& x, bool) override {
        const long long B = (long long)x.rows; batch_ = (size_t)B;
        const size_t O = (size_t)C_ * oh_ * ow_;
        Matrix y((size_t)B, O);
        arg_.assign((size_t)B * O, 0);
        #pragma omp parallel for schedule(static)
        for (long long bi = 0; bi < B; ++bi) {
            const float* xin = &x.data[(size_t)bi * C_ * H_ * W_];
            size_t o = 0;
            for (int c = 0; c < C_; ++c)
                for (int oy = 0; oy < oh_; ++oy)
                    for (int ox = 0; ox < ow_; ++ox, ++o) {
                        float best = -1e30f; int bidx = 0;
                        for (int ky = 0; ky < k_; ++ky)
                            for (int kx = 0; kx < k_; ++kx) {
                                int idx = (c * H_ + oy * k_ + ky) * W_ + ox * k_ + kx;
                                if (xin[idx] > best) { best = xin[idx]; bidx = idx; }
                            }
                        y.data[(size_t)bi * O + o] = best;
                        arg_[(size_t)bi * O + o] = bidx;
                    }
        }
        return y;
    }
    Matrix backward(const Matrix& g) override {
        const size_t O = (size_t)C_ * oh_ * ow_, CHW = (size_t)C_ * H_ * W_;
        Matrix dx(batch_, CHW);
        for (size_t bi = 0; bi < batch_; ++bi)
            for (size_t o = 0; o < O; ++o)
                dx.data[bi * CHW + arg_[bi * O + o]] += g.data[bi * O + o];
        return dx;
    }
};

// ---------------------------------------------------------------------------
// Embedding with mean pooling (fastText-style). Input: batch x len of token ids
// stored as floats (0 = padding). Output: batch x dim.
// ---------------------------------------------------------------------------
class Embedding : public Layer {
    int V_, D_;
    Matrix E_, dE_, ids_;
    std::vector<int> cnt_;
public:
    Embedding(int vocabWithPad, int dim, std::mt19937& rng) : V_(vocabWithPad), D_(dim), E_(vocabWithPad, dim), dE_(vocabWithPad, dim) {
        std::uniform_real_distribution<float> u(-0.1f, 0.1f);
        for (auto& w : E_.data) w = u(rng);
        for (int j = 0; j < D_; ++j) E_(0, j) = 0.f;
    }
    Matrix forward(const Matrix& x, bool) override {
        ids_ = x; cnt_.assign(x.rows, 0);
        Matrix y(x.rows, D_);
        for (size_t b = 0; b < x.rows; ++b) {
            int n = 0;
            for (size_t t = 0; t < x.cols; ++t) {
                int id = (int)x(b, t);
                if (id <= 0 || id >= V_) continue;
                const float* e = &E_.data[(size_t)id * D_];
                for (int j = 0; j < D_; ++j) y(b, j) += e[j];
                ++n;
            }
            cnt_[b] = n;
            if (n > 0) for (int j = 0; j < D_; ++j) y(b, j) /= (float)n;
        }
        return y;
    }
    Matrix backward(const Matrix& g) override {
        dE_.zero();
        for (size_t b = 0; b < ids_.rows; ++b) {
            if (cnt_[b] == 0) continue;
            const float inv = 1.f / (float)cnt_[b];
            for (size_t t = 0; t < ids_.cols; ++t) {
                int id = (int)ids_(b, t);
                if (id <= 0 || id >= V_) continue;
                float* d = &dE_.data[(size_t)id * D_];
                for (int j = 0; j < D_; ++j) d[j] += g(b, j) * inv;
            }
        }
        return Matrix(ids_.rows, ids_.cols);   // no gradient flows to token ids
    }
    std::vector<Param> params() override { return {{&E_, &dE_, false}}; }
    std::string name() const override { return "Embedding"; }
    std::string describe() const override { return "Embedding vocab " + std::to_string(V_ - 1) + " dim " + std::to_string(D_) + " (mean-pool)"; }
};


// ---------------------------------------------------------------------------
// BatchNorm2D: per-channel normalization over (batch x H x W). Layout CHW, rows = batch.
// Training uses batch statistics (and updates running stats); eval uses the running stats.
// backward() assumes the preceding forward() was a training pass.
// ---------------------------------------------------------------------------
class BatchNorm2D : public Layer {
    int C_, HW_; float eps_ = 1e-5f, mom_ = 0.1f;
    Matrix g_, b_, dg_, db_, runMean_, runVar_, xhat_;
    std::vector<float> rstd_;
    size_t B_ = 0;
public:
    BatchNorm2D(int C, int HW) : C_(C), HW_(HW), g_(1, C, 1.f), b_(1, C), dg_(1, C), db_(1, C), runMean_(1, C), runVar_(1, C, 1.f) {}

    Matrix forward(const Matrix& x, bool training) override {
        const size_t B = x.rows, CHW = (size_t)C_ * HW_; B_ = B;
        Matrix y(B, CHW);
        if (training) {
            xhat_.resize(B, CHW); rstd_.assign(C_, 0.f);
            const double N = (double)B * HW_;
            #pragma omp parallel for schedule(static)
            for (int c = 0; c < C_; ++c) {
                double m = 0, v = 0;
                for (size_t b = 0; b < B; ++b) { const float* p = &x.data[b * CHW + (size_t)c * HW_]; for (int i = 0; i < HW_; ++i) m += p[i]; }
                m /= N;
                for (size_t b = 0; b < B; ++b) { const float* p = &x.data[b * CHW + (size_t)c * HW_]; for (int i = 0; i < HW_; ++i) { double t = p[i] - m; v += t * t; } }
                v /= N;
                const float rs = 1.f / std::sqrt((float)v + eps_); rstd_[c] = rs;
                for (size_t b = 0; b < B; ++b) {
                    const float* p = &x.data[b * CHW + (size_t)c * HW_]; float* h = &xhat_.data[b * CHW + (size_t)c * HW_]; float* o = &y.data[b * CHW + (size_t)c * HW_];
                    for (int i = 0; i < HW_; ++i) { h[i] = (p[i] - (float)m) * rs; o[i] = g_.data[c] * h[i] + b_.data[c]; }
                }
                const double unb = N > 1 ? v * N / (N - 1) : v;
                runMean_.data[c] = (1.f - mom_) * runMean_.data[c] + mom_ * (float)m;
                runVar_.data[c] = (1.f - mom_) * runVar_.data[c] + mom_ * (float)unb;
            }
        } else {
            #pragma omp parallel for schedule(static)
            for (int c = 0; c < C_; ++c) {
                const float sc = g_.data[c] / std::sqrt(runVar_.data[c] + eps_), sh = b_.data[c] - sc * runMean_.data[c];
                for (size_t b = 0; b < B; ++b) {
                    const float* p = &x.data[b * CHW + (size_t)c * HW_]; float* o = &y.data[b * CHW + (size_t)c * HW_];
                    for (int i = 0; i < HW_; ++i) o[i] = sc * p[i] + sh;
                }
            }
        }
        return y;
    }
    Matrix backward(const Matrix& g) override {
        const size_t B = B_, CHW = (size_t)C_ * HW_; const double N = (double)B * HW_;
        Matrix dx(B, CHW);
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < C_; ++c) {
            double s1 = 0, s2 = 0, sdg = 0, sdb = 0;
            for (size_t b = 0; b < B; ++b) {
                const float* dy = &g.data[b * CHW + (size_t)c * HW_]; const float* xh = &xhat_.data[b * CHW + (size_t)c * HW_];
                for (int i = 0; i < HW_; ++i) { double dxh = dy[i] * g_.data[c]; s1 += dxh; s2 += dxh * xh[i]; sdg += dy[i] * xh[i]; sdb += dy[i]; }
            }
            const double inv = rstd_[c] / N;
            for (size_t b = 0; b < B; ++b) {
                const float* dy = &g.data[b * CHW + (size_t)c * HW_]; const float* xh = &xhat_.data[b * CHW + (size_t)c * HW_]; float* o = &dx.data[b * CHW + (size_t)c * HW_];
                for (int i = 0; i < HW_; ++i) o[i] = (float)(inv * (N * dy[i] * g_.data[c] - s1 - xh[i] * s2));
            }
            dg_.data[c] = (float)sdg; db_.data[c] = (float)sdb;
        }
        return dx;
    }
    std::vector<Param> params() override { return {{&g_, &dg_, false}, {&b_, &db_, false}}; }
    std::vector<Matrix*> buffers() override { return {&runMean_, &runVar_}; }
    std::string name() const override { return "BatchNorm2D"; }
    std::string describe() const override { return "BatchNorm2D " + std::to_string(C_) + " channels"; }
};

// ---------------------------------------------------------------------------
// GlobalAvgPool: (batch x C*HW) -> (batch x C). Makes the classifier head independent of image size.
// ---------------------------------------------------------------------------
class GlobalAvgPool : public Layer {
    int C_, HW_; size_t B_ = 0;
public:
    GlobalAvgPool(int C, int HW) : C_(C), HW_(HW) {}
    Matrix forward(const Matrix& x, bool) override {
        B_ = x.rows; Matrix y(B_, C_);
        for (size_t b = 0; b < B_; ++b)
            for (int c = 0; c < C_; ++c) {
                const float* p = &x.data[b * (size_t)C_ * HW_ + (size_t)c * HW_]; double s = 0;
                for (int i = 0; i < HW_; ++i) s += p[i];
                y(b, c) = (float)(s / HW_);
            }
        return y;
    }
    Matrix backward(const Matrix& g) override {
        Matrix dx(B_, (size_t)C_ * HW_); const float inv = 1.f / (float)HW_;
        for (size_t b = 0; b < B_; ++b)
            for (int c = 0; c < C_; ++c) {
                float* p = &dx.data[b * (size_t)C_ * HW_ + (size_t)c * HW_]; const float v = g(b, c) * inv;
                for (int i = 0; i < HW_; ++i) p[i] = v;
            }
        return dx;
    }
    std::string name() const override { return "GlobalAvgPool"; }
    std::string describe() const override { return "GlobalAvgPool " + std::to_string(C_) + " channels"; }
};

// ---------------------------------------------------------------------------
// ResidualBlock (composite layer): y = act( F(x) + S(x) ),  F = conv-[bn]-act-conv-[bn],
// S = identity, or 1x1 conv-[bn] when the shape changes. Owns its child layers (composition).
// ---------------------------------------------------------------------------
class ResidualBlock : public Layer {
    std::vector<std::unique_ptr<Layer>> main_, short_;
    Activation post_;
    int inC_, outC_, stride_;
public:
    ResidualBlock(int inC, int H, int W, int outC, int stride, Act act, bool bn, std::mt19937& rng)
        : post_(act), inC_(inC), outC_(outC), stride_(stride) {
        const int oh = (H + 2 - 3) / stride + 1, ow = (W + 2 - 3) / stride + 1;
        main_.push_back(std::make_unique<Conv2D>(inC, H, W, outC, 3, stride, 1, rng));
        if (bn) main_.push_back(std::make_unique<BatchNorm2D>(outC, oh * ow));
        main_.push_back(std::make_unique<Activation>(act));
        main_.push_back(std::make_unique<Conv2D>(outC, oh, ow, outC, 3, 1, 1, rng));
        if (bn) main_.push_back(std::make_unique<BatchNorm2D>(outC, oh * ow));
        if (stride != 1 || inC != outC) {
            short_.push_back(std::make_unique<Conv2D>(inC, H, W, outC, 1, stride, 0, rng));
            if (bn) short_.push_back(std::make_unique<BatchNorm2D>(outC, oh * ow));
        }
    }
    Matrix forward(const Matrix& x, bool training) override {
        Matrix a = x;
        for (auto& l : main_) a = l->forward(a, training);
        if (short_.empty()) { for (size_t i = 0; i < a.size(); ++i) a.data[i] += x.data[i]; }
        else { Matrix s = x; for (auto& l : short_) s = l->forward(s, training); for (size_t i = 0; i < a.size(); ++i) a.data[i] += s.data[i]; }
        return post_.forward(a, training);
    }
    Matrix backward(const Matrix& gy) override {
        Matrix g = post_.backward(gy);
        Matrix gm = g; for (auto it = main_.rbegin(); it != main_.rend(); ++it) gm = (*it)->backward(gm);
        if (short_.empty()) { for (size_t i = 0; i < gm.size(); ++i) gm.data[i] += g.data[i]; return gm; }
        Matrix gs = g; for (auto it = short_.rbegin(); it != short_.rend(); ++it) gs = (*it)->backward(gs);
        for (size_t i = 0; i < gm.size(); ++i) gm.data[i] += gs.data[i];
        return gm;
    }
    std::vector<Param> params() override {
        std::vector<Param> ps;
        for (auto& l : main_) for (auto& p : l->params()) ps.push_back(p);
        for (auto& l : short_) for (auto& p : l->params()) ps.push_back(p);
        return ps;
    }
    std::vector<Matrix*> buffers() override {
        std::vector<Matrix*> bs;
        for (auto& l : main_) for (auto* b : l->buffers()) bs.push_back(b);
        for (auto& l : short_) for (auto* b : l->buffers()) bs.push_back(b);
        return bs;
    }
    std::string name() const override { return "ResidualBlock"; }
    std::string describe() const override {
        return "ResidualBlock " + std::to_string(inC_) + "->" + std::to_string(outC_) + " stride " + std::to_string(stride_) +
               (short_.empty() ? " (identity skip)" : " (1x1 conv skip)");
    }
};

} // namespace nn
