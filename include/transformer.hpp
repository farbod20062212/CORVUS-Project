#pragma once
// GPT-style decoder-only Transformer with hand-written forward + backward,
// plus a KV-cache generator for fast inference.
#include "model.hpp"
#include "bpe.hpp"
#include <random>

namespace nn {

struct GPTConfig {
    int vocab = 0, ctx = 128, dim = 128, layers = 4, heads = 4;
    float dropout = 0.1f;
};

struct Tn {   // trainable tensor: value + gradient
    Matrix w, g; bool decay = false;
    void init(size_t r, size_t c, bool d) { w = Matrix(r, c); g = Matrix(r, c); decay = d; }
};
struct Block { Tn ln1g, ln1b, wqkv, bqkv, wo, bo, ln2g, ln2b, w1, b1, w2, b2; };
struct LayerCache {
    Matrix xin, xhat1, h1, qkv, att, mid, xhat2, h2, f1, g;
    std::vector<float> rstd1, rstd2, P, m1, m2;
};

// ------------- numeric helpers -------------
namespace tf {

inline float gelu(float x) { const float c = 0.7978845608f; return 0.5f * x * (1.f + std::tanh(c * (x + 0.044715f * x * x * x))); }
inline float geluGrad(float x) {
    const float c = 0.7978845608f; float u = c * (x + 0.044715f * x * x * x), t = std::tanh(u);
    return 0.5f * (1.f + t) + 0.5f * x * (1.f - t * t) * c * (1.f + 3.f * 0.044715f * x * x);
}

inline void linear(const Matrix& X, const Matrix& W, const Matrix& b, Matrix& Y) {
    matmul(X, W, Y);
    const long long n = (long long)Y.rows;
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i) for (size_t j = 0; j < Y.cols; ++j) Y.data[i * Y.cols + j] += b.data[j];
}
inline void linearBackward(const Matrix& X, const Matrix& W, const Matrix& dY, Matrix& dW, Matrix& db, Matrix* dX) {
    matmulAtB(X, dY, dW);
    db.resize(1, dY.cols);
    for (size_t i = 0; i < dY.rows; ++i) for (size_t j = 0; j < dY.cols; ++j) db.data[j] += dY.data[i * dY.cols + j];
    if (dX) matmulABt(dY, W, *dX);
}

inline void lnForward(const Matrix& x, const Matrix& g, const Matrix& b, Matrix& y, Matrix& xhat, std::vector<float>& rstd) {
    const size_t N = x.rows, d = x.cols;
    y.resize(N, d); xhat.resize(N, d); rstd.assign(N, 0.f);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)N; ++i) {
        const float* r = &x.data[i * d]; double m = 0, v = 0;
        for (size_t j = 0; j < d; ++j) m += r[j];
        m /= d;
        for (size_t j = 0; j < d; ++j) { double t = r[j] - m; v += t * t; }
        float rs = 1.f / std::sqrt((float)(v / d) + 1e-5f); rstd[i] = rs;
        for (size_t j = 0; j < d; ++j) {
            float h = (float)(r[j] - m) * rs; xhat.data[i * d + j] = h; y.data[i * d + j] = h * g.data[j] + b.data[j];
        }
    }
}
inline void lnBackward(const Matrix& dy, const Matrix& xhat, const std::vector<float>& rstd, const Matrix& g,
                       Matrix& dx, Matrix& dg, Matrix& db) {
    const size_t N = dy.rows, d = dy.cols;
    dx.resize(N, d); dg.resize(1, d); db.resize(1, d);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)N; ++i) {
        double s1 = 0, s2 = 0;
        for (size_t j = 0; j < d; ++j) { float dh = dy.data[i * d + j] * g.data[j]; s1 += dh; s2 += dh * xhat.data[i * d + j]; }
        for (size_t j = 0; j < d; ++j) {
            float dh = dy.data[i * d + j] * g.data[j];
            dx.data[i * d + j] = rstd[i] * (dh - (float)(s1 / d) - xhat.data[i * d + j] * (float)(s2 / d));
        }
    }
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < d; ++j) { dg.data[j] += dy.data[i * d + j] * xhat.data[i * d + j]; db.data[j] += dy.data[i * d + j]; }
}

inline void dropFwd(Matrix& x, std::vector<float>& mask, float p, std::mt19937& rng, bool train) {
    if (!train || p <= 0.f) { mask.clear(); return; }
    mask.resize(x.size()); std::bernoulli_distribution keep(1.0 - p); const float sc = 1.f / (1.f - p);
    for (size_t i = 0; i < x.size(); ++i) { mask[i] = keep(rng) ? sc : 0.f; x.data[i] *= mask[i]; }
}
inline void dropBwd(Matrix& g, const std::vector<float>& mask) {
    if (mask.empty()) return;
    for (size_t i = 0; i < g.size(); ++i) g.data[i] *= mask[i];
}

// qkv: (B*T) x 3d  -> out (B*T) x d, P: B*H*T*T attention probabilities (causal)
inline void attnForward(const Matrix& qkv, int B, int T, int H, int dh, std::vector<float>& P, Matrix& out) {
    const int d = H * dh; const float scale = 1.f / std::sqrt((float)dh);
    out.resize((size_t)B * T, d); P.assign((size_t)B * H * T * T, 0.f);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < B; ++b)
        for (int h = 0; h < H; ++h) {
            float* Pbh = &P[((size_t)b * H + h) * T * T];
            for (int i = 0; i < T; ++i) {
                const float* q = &qkv.data[((size_t)b * T + i) * 3 * d + h * dh];
                float* row = Pbh + (size_t)i * T; float mx = -1e30f;
                for (int j = 0; j <= i; ++j) {
                    const float* k = &qkv.data[((size_t)b * T + j) * 3 * d + d + h * dh];
                    float s = 0.f;
                    #pragma omp simd reduction(+ : s)
                    for (int t = 0; t < dh; ++t) s += q[t] * k[t];
                    s *= scale; row[j] = s; mx = std::max(mx, s);
                }
                float sum = 0.f;
                for (int j = 0; j <= i; ++j) { row[j] = std::exp(row[j] - mx); sum += row[j]; }
                const float inv = 1.f / sum;
                float* o = &out.data[((size_t)b * T + i) * d + h * dh];
                for (int t = 0; t < dh; ++t) o[t] = 0.f;
                for (int j = 0; j <= i; ++j) {
                    row[j] *= inv;
                    const float* v = &qkv.data[((size_t)b * T + j) * 3 * d + 2 * d + h * dh];
                    for (int t = 0; t < dh; ++t) o[t] += row[j] * v[t];
                }
            }
        }
}
inline void attnBackward(const Matrix& qkv, const std::vector<float>& P, const Matrix& dout, int B, int T, int H, int dh, Matrix& dqkv) {
    const int d = H * dh; const float scale = 1.f / std::sqrt((float)dh);
    dqkv.resize((size_t)B * T, 3 * d);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < B; ++b)
        for (int h = 0; h < H; ++h) {
            const float* Pbh = &P[((size_t)b * H + h) * T * T];
            std::vector<float> dP(T);
            for (int i = 0; i < T; ++i) {
                const float* doi = &dout.data[((size_t)b * T + i) * d + h * dh];
                const float* row = Pbh + (size_t)i * T;
                float sum = 0.f;
                for (int j = 0; j <= i; ++j) {
                    const float* v = &qkv.data[((size_t)b * T + j) * 3 * d + 2 * d + h * dh];
                    float s = 0.f;
                    #pragma omp simd reduction(+ : s)
                    for (int t = 0; t < dh; ++t) s += doi[t] * v[t];
                    dP[j] = s; sum += row[j] * s;
                    float* dv = &dqkv.data[((size_t)b * T + j) * 3 * d + 2 * d + h * dh];
                    for (int t = 0; t < dh; ++t) dv[t] += row[j] * doi[t];
                }
                const float* q = &qkv.data[((size_t)b * T + i) * 3 * d + h * dh];
                float* dq = &dqkv.data[((size_t)b * T + i) * 3 * d + h * dh];
                for (int j = 0; j <= i; ++j) {
                    float ds = row[j] * (dP[j] - sum) * scale;
                    const float* k = &qkv.data[((size_t)b * T + j) * 3 * d + d + h * dh];
                    float* dk = &dqkv.data[((size_t)b * T + j) * 3 * d + d + h * dh];
                    for (int t = 0; t < dh; ++t) { dq[t] += ds * k[t]; dk[t] += ds * q[t]; }
                }
            }
        }
}

} // namespace tf

// ------------------------------- GPT -------------------------------
class GPT {
public:
    GPTConfig cfg;
    Tn tok, pos, lnfg, lnfb;
    std::vector<Block> blocks;
    BPE bpe;

    std::vector<LayerCache> cache;
    Matrix hf, xhatf; std::vector<float> rstdf, embMask;
    std::mt19937 rng{1234};

    void init(const GPTConfig& c, unsigned seed) {
        cfg = c; rng.seed(seed);
        if (cfg.dim % cfg.heads) throw std::runtime_error("dim must be divisible by heads");
        std::mt19937 r(seed); std::normal_distribution<float> nd(0.f, 0.02f);
        const size_t d = cfg.dim, ff = 4 * d;
        auto randn = [&](Tn& t, float scale) { for (auto& v : t.w.data) v = nd(r) * scale; };
        tok.init(cfg.vocab, d, false); randn(tok, 1.f);
        pos.init(cfg.ctx, d, false);   randn(pos, 1.f);
        lnfg.init(1, d, false); lnfb.init(1, d, false); std::fill(lnfg.w.data.begin(), lnfg.w.data.end(), 1.f);
        blocks.assign(cfg.layers, Block());
        const float resid = 1.f / std::sqrt(2.f * cfg.layers);
        for (auto& b : blocks) {
            b.ln1g.init(1, d, false); b.ln1b.init(1, d, false); b.ln2g.init(1, d, false); b.ln2b.init(1, d, false);
            std::fill(b.ln1g.w.data.begin(), b.ln1g.w.data.end(), 1.f); std::fill(b.ln2g.w.data.begin(), b.ln2g.w.data.end(), 1.f);
            b.wqkv.init(d, 3 * d, true); b.bqkv.init(1, 3 * d, false); randn(b.wqkv, 1.f);
            b.wo.init(d, d, true); b.bo.init(1, d, false); randn(b.wo, resid);
            b.w1.init(d, ff, true); b.b1.init(1, ff, false); randn(b.w1, 1.f);
            b.w2.init(ff, d, true); b.b2.init(1, d, false); randn(b.w2, resid);
        }
    }

    std::vector<Param> params() {
        std::vector<Param> ps;
        auto add = [&](Tn& t) { ps.push_back({&t.w, &t.g, t.decay}); };
        add(tok); add(pos); add(lnfg); add(lnfb);
        for (auto& b : blocks) { add(b.ln1g); add(b.ln1b); add(b.wqkv); add(b.bqkv); add(b.wo); add(b.bo);
                                 add(b.ln2g); add(b.ln2b); add(b.w1); add(b.b1); add(b.w2); add(b.b2); }
        return ps;
    }
    size_t paramCount() { size_t n = 0; for (auto& p : params()) n += p.value->size(); return n; }

    // Forward (+ optional backward). inputs/targets/mask have B*T entries. Returns mean masked cross-entropy.
    float step(const int* inputs, const int* targets, const float* mask, int B, int T, bool train, bool backward) {
        using namespace tf;
        const int d = cfg.dim, H = cfg.heads, dh = d / H, V = cfg.vocab, L = cfg.layers;
        const size_t N = (size_t)B * T;
        if (T > cfg.ctx) throw std::runtime_error("sequence longer than ctx");
        cache.resize(L);
        Matrix x(N, d);
        for (size_t n = 0; n < N; ++n) {
            const float* te = &tok.w.data[(size_t)inputs[n] * d]; const float* pe = &pos.w.data[(size_t)(n % T) * d];
            for (int j = 0; j < d; ++j) x.data[n * d + j] = te[j] + pe[j];
        }
        dropFwd(x, embMask, cfg.dropout, rng, train);
        for (int l = 0; l < L; ++l) {
            Block& B_ = blocks[l]; LayerCache& c = cache[l];
            c.xin = x;
            lnForward(x, B_.ln1g.w, B_.ln1b.w, c.h1, c.xhat1, c.rstd1);
            linear(c.h1, B_.wqkv.w, B_.bqkv.w, c.qkv);
            attnForward(c.qkv, B, T, H, dh, c.P, c.att);
            Matrix proj; linear(c.att, B_.wo.w, B_.bo.w, proj);
            dropFwd(proj, c.m1, cfg.dropout, rng, train);
            c.mid = x; for (size_t i = 0; i < c.mid.size(); ++i) c.mid.data[i] += proj.data[i];
            lnForward(c.mid, B_.ln2g.w, B_.ln2b.w, c.h2, c.xhat2, c.rstd2);
            linear(c.h2, B_.w1.w, B_.b1.w, c.f1);
            c.g = c.f1; for (auto& v : c.g.data) v = gelu(v);
            Matrix f2; linear(c.g, B_.w2.w, B_.b2.w, f2);
            dropFwd(f2, c.m2, cfg.dropout, rng, train);
            x = c.mid; for (size_t i = 0; i < x.size(); ++i) x.data[i] += f2.data[i];
        }
        lnForward(x, lnfg.w, lnfb.w, hf, xhatf, rstdf);
        Matrix lg; matmulABt(hf, tok.w, lg);                // N x V (tied embeddings)

        double wsum = 0; for (size_t n = 0; n < N; ++n) wsum += mask[n];
        if (wsum <= 0) return 0.f;
        std::vector<double> rowLoss(N, 0.0);
        #pragma omp parallel for schedule(static)
        for (long long n = 0; n < (long long)N; ++n) {
            float* r = &lg.data[(size_t)n * V]; float mx = r[0];
            for (int v = 1; v < V; ++v) mx = std::max(mx, r[v]);
            double sum = 0; for (int v = 0; v < V; ++v) { r[v] = std::exp(r[v] - mx); sum += r[v]; }
            const float inv = (float)(1.0 / sum); for (int v = 0; v < V; ++v) r[v] *= inv;
            if (mask[n] > 0.f) rowLoss[n] = -(double)mask[n] * std::log((double)r[targets[n]] + 1e-20);
            if (backward) {
                const float w = (float)(mask[n] / wsum);
                for (int v = 0; v < V; ++v) r[v] *= w;
                r[targets[n]] -= w;
            }
        }
        double total = 0; for (double v : rowLoss) total += v;
        const float loss = (float)(total / wsum);
        if (!backward) return loss;

        // ---- backward ----
        Matrix dhf; matmul(lg, tok.w, dhf);                 // N x d
        matmulAtB(lg, hf, tok.g);                           // V x d (head contribution)
        Matrix dx; lnBackward(dhf, xhatf, rstdf, lnfg.w, dx, lnfg.g, lnfb.g);
        for (int l = L - 1; l >= 0; --l) {
            Block& B_ = blocks[l]; LayerCache& c = cache[l];
            Matrix df2 = dx; dropBwd(df2, c.m2);
            Matrix dg, df1, dh2;
            linearBackward(c.g, B_.w2.w, df2, B_.w2.g, B_.b2.g, &dg);
            df1 = dg; for (size_t i = 0; i < df1.size(); ++i) df1.data[i] *= geluGrad(c.f1.data[i]);
            linearBackward(c.h2, B_.w1.w, df1, B_.w1.g, B_.b1.g, &dh2);
            Matrix dmidLn; lnBackward(dh2, c.xhat2, c.rstd2, B_.ln2g.w, dmidLn, B_.ln2g.g, B_.ln2b.g);
            Matrix dmid = dx; for (size_t i = 0; i < dmid.size(); ++i) dmid.data[i] += dmidLn.data[i];
            Matrix dproj = dmid; dropBwd(dproj, c.m1);
            Matrix datt, dqkv, dh1;
            linearBackward(c.att, B_.wo.w, dproj, B_.wo.g, B_.bo.g, &datt);
            attnBackward(c.qkv, c.P, datt, B, T, H, dh, dqkv);
            linearBackward(c.h1, B_.wqkv.w, dqkv, B_.wqkv.g, B_.bqkv.g, &dh1);
            Matrix dxLn; lnBackward(dh1, c.xhat1, c.rstd1, B_.ln1g.w, dxLn, B_.ln1g.g, B_.ln1b.g);
            dx = dmid; for (size_t i = 0; i < dx.size(); ++i) dx.data[i] += dxLn.data[i];
        }
        dropBwd(dx, embMask);
        pos.g.resize(cfg.ctx, d);
        for (size_t n = 0; n < N; ++n) {
            float* gt = &tok.g.data[(size_t)inputs[n] * d]; float* gp = &pos.g.data[(size_t)(n % T) * d];
            const float* r = &dx.data[n * d];
            for (int j = 0; j < d; ++j) { gt[j] += r[j]; gp[j] += r[j]; }
        }
        return loss;
    }
};

// ------------------------- checkpoint I/O -------------------------
inline void saveLM(const std::string& path, GPT& m) {
    std::ofstream o(path, std::ios::binary);
    if (!o) throw std::runtime_error("cannot write " + path);
    o.write("GPT1", 4);
    io::wr(o, m.cfg.vocab); io::wr(o, m.cfg.ctx); io::wr(o, m.cfg.dim); io::wr(o, m.cfg.layers);
    io::wr(o, m.cfg.heads); io::wr(o, m.cfg.dropout);
    m.bpe.save(o);
    for (auto& p : m.params()) io::wrVec(o, p.value->data);
}
inline void loadLM(const std::string& path, GPT& m) {
    std::ifstream i(path, std::ios::binary);
    if (!i) throw std::runtime_error(cannotOpenMsg("model file", path));
    char mg[4]; i.read(mg, 4);
    if (std::memcmp(mg, "GPT1", 4) != 0) throw std::runtime_error("not a GPT1 language-model file");
    GPTConfig c;
    io::rd(i, c.vocab); io::rd(i, c.ctx); io::rd(i, c.dim); io::rd(i, c.layers); io::rd(i, c.heads); io::rd(i, c.dropout);
    m.bpe.load(i);
    m.init(c, 1);
    for (auto& p : m.params()) io::rdVec(i, p.value->data);
    if (!i) throw std::runtime_error("model file truncated/corrupt");
}

// ------------------------- fast inference (KV cache) -------------------------
class Generator {
    GPT& m; std::vector<Matrix> K, Vc; std::vector<float> logits;
    static void vecmat(const float* x, int in, const Matrix& W, const Matrix& b, float* y) {
        const int out = (int)W.cols;
        for (int j = 0; j < out; ++j) y[j] = b.data[j];
        for (int k = 0; k < in; ++k) { const float a = x[k]; const float* w = &W.data[(size_t)k * out]; for (int j = 0; j < out; ++j) y[j] += a * w[j]; }
    }
    static void ln(const float* x, int d, const Matrix& g, const Matrix& b, float* y) {
        double mean = 0, var = 0; for (int j = 0; j < d; ++j) mean += x[j]; mean /= d;
        for (int j = 0; j < d; ++j) { double t = x[j] - mean; var += t * t; }
        float rs = 1.f / std::sqrt((float)(var / d) + 1e-5f);
        for (int j = 0; j < d; ++j) y[j] = (float)(x[j] - mean) * rs * g.data[j] + b.data[j];
    }
public:
    int pos = 0;
    explicit Generator(GPT& model) : m(model) {
        K.assign(m.cfg.layers, Matrix(m.cfg.ctx, m.cfg.dim)); Vc = K; logits.assign(m.cfg.vocab, 0.f);
    }
    void reset() { pos = 0; }
    const std::vector<float>& step(int token) {
        const int d = m.cfg.dim, H = m.cfg.heads, dh = d / H, ff = 4 * d, V = m.cfg.vocab;
        if (pos >= m.cfg.ctx) throw std::runtime_error("context full");
        std::vector<float> x(d), h(d), qkv(3 * d), att(d), proj(d), f1(ff), f2(d);
        for (int j = 0; j < d; ++j) x[j] = m.tok.w(token, j) + m.pos.w(pos, j);
        const float scale = 1.f / std::sqrt((float)dh);
        std::vector<float> sc(pos + 1);
        for (int l = 0; l < m.cfg.layers; ++l) {
            Block& B = m.blocks[l];
            ln(x.data(), d, B.ln1g.w, B.ln1b.w, h.data());
            vecmat(h.data(), d, B.wqkv.w, B.bqkv.w, qkv.data());
            std::copy(qkv.begin() + d, qkv.begin() + 2 * d, &K[l].data[(size_t)pos * d]);
            std::copy(qkv.begin() + 2 * d, qkv.begin() + 3 * d, &Vc[l].data[(size_t)pos * d]);
            std::fill(att.begin(), att.end(), 0.f);
            for (int hh = 0; hh < H; ++hh) {
                const float* q = &qkv[hh * dh]; float mx = -1e30f;
                for (int j = 0; j <= pos; ++j) {
                    const float* k = &K[l].data[(size_t)j * d + hh * dh];
                    float s = 0.f; for (int t = 0; t < dh; ++t) s += q[t] * k[t];
                    sc[j] = s * scale; mx = std::max(mx, sc[j]);
                }
                float sum = 0.f; for (int j = 0; j <= pos; ++j) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
                for (int j = 0; j <= pos; ++j) {
                    const float p = sc[j] / sum; const float* v = &Vc[l].data[(size_t)j * d + hh * dh];
                    for (int t = 0; t < dh; ++t) att[hh * dh + t] += p * v[t];
                }
            }
            vecmat(att.data(), d, B.wo.w, B.bo.w, proj.data());
            for (int j = 0; j < d; ++j) x[j] += proj[j];
            ln(x.data(), d, B.ln2g.w, B.ln2b.w, h.data());
            vecmat(h.data(), d, B.w1.w, B.b1.w, f1.data());
            for (auto& v : f1) v = tf::gelu(v);
            vecmat(f1.data(), ff, B.w2.w, B.b2.w, f2.data());
            for (int j = 0; j < d; ++j) x[j] += f2[j];
        }
        ln(x.data(), d, m.lnfg.w, m.lnfb.w, h.data());
        for (int v = 0; v < V; ++v) { const float* e = &m.tok.w.data[(size_t)v * d]; float s = 0.f; for (int j = 0; j < d; ++j) s += h[j] * e[j]; logits[v] = s; }
        ++pos;
        return logits;
    }
};

// ------------------------- scaling tools -------------------------
// Function-preserving growth: more layers (new blocks start as identity: wo = w2 = 0)
// and a longer context window (new positions get fresh embeddings). Width (dim/heads) cannot grow.
inline void growLM(GPT& m, int newLayers, int newCtx, unsigned seed) {
    if (newLayers < m.cfg.layers || newCtx < m.cfg.ctx) throw std::runtime_error("grow can only increase --layers / --ctx");
    GPTConfig c = m.cfg; c.layers = newLayers; c.ctx = newCtx;
    GPT g; g.init(c, seed); g.bpe = m.bpe;
    g.tok.w = m.tok.w; g.lnfg.w = m.lnfg.w; g.lnfb.w = m.lnfb.w;
    std::copy(m.pos.w.data.begin(), m.pos.w.data.end(), g.pos.w.data.begin());   // rows 0..oldCtx-1
    for (int l = 0; l < m.cfg.layers; ++l) {
        Block& s = m.blocks[l]; Block& d = g.blocks[l];
        d.ln1g.w = s.ln1g.w; d.ln1b.w = s.ln1b.w; d.wqkv.w = s.wqkv.w; d.bqkv.w = s.bqkv.w; d.wo.w = s.wo.w; d.bo.w = s.bo.w;
        d.ln2g.w = s.ln2g.w; d.ln2b.w = s.ln2b.w; d.w1.w = s.w1.w; d.b1.w = s.b1.w; d.w2.w = s.w2.w; d.b2.w = s.b2.w;
    }
    for (int l = m.cfg.layers; l < newLayers; ++l) { g.blocks[l].wo.w.zero(); g.blocks[l].w2.w.zero(); }
    m = std::move(g);
}

struct LMEstimate { double params, flopsPerToken, trainBytes, minTokens; };
inline LMEstimate estimateLM(const GPTConfig& c, int batch) {
    const double d = c.dim, L = c.layers, V = c.vocab, T = c.ctx;
    LMEstimate e;
    e.params = V * d + T * d + L * (12.0 * d * d + 13.0 * d) + 2 * d;
    e.flopsPerToken = 6.0 * e.params + 6.0 * L * T * d;                 // matmuls + attention
    const double N = (double)batch * T;
    const double act = L * (18.0 * N * d + (double)batch * c.heads * T * T) + 2.0 * N * V;
    e.trainBytes = 4.0 * (4.0 * e.params + act);                        // weights+grads+Adam(2) + activations
    e.minTokens = 20.0 * e.params;                                      // rule of thumb (compute-optimal)
    return e;
}

} // namespace nn
