#pragma once
// Sequential models (MLP / CNN / Text) built by polymorphic builders, + binary save/load.
#include "layers.hpp"
#include "dataset.hpp"
#include "loss.hpp"
#include <cstdint>
#include <cstring>

namespace nn {

enum class Kind { MLP = 0, CNN = 1, Text = 2 };
enum class Arch { Plain = 0, ResNet = 1 };   // CNN family

struct ModelConfig {
    Kind kind = Kind::MLP;
    int inputDim = 0, outputDim = 0;
    std::vector<int> hidden{64, 32};
    Act act = Act::ReLU;
    float dropout = 0.f;
    Task task = Task::Classification;
    unsigned seed = 42;
    // CNN
    int imgC = 1, imgH = 0, imgW = 0;
    std::vector<int> conv;            // Plain: channels of each conv3x3+act+pool block | ResNet: width of each stage
    Arch arch = Arch::Plain;
    bool batchNorm = false;           // BatchNorm after every convolution
    bool globalPool = false;          // global average pooling instead of flatten (always on for ResNet)
    int blocks = 1;                   // ResNet: residual blocks per stage
    int fit = 0;                      // FitMode used to bring images to imgH x imgW (stored so predict matches training)
    // Text
    int vocab = 0, embed = 64, maxLen = 0, ngrams = 1;
};

// ------------------------------------------------------------------------------------------
// Builders (Template Method): every model = feature extractor (kind-specific) + shared classifier head.
// Adding a new model family means adding one subclass; nothing else changes.
// ------------------------------------------------------------------------------------------
using LayerList = std::vector<std::unique_ptr<Layer>>;

struct BuildCtx {
    std::mt19937 rng; unsigned dseed; LayerList& layers;
    BuildCtx(unsigned seed, LayerList& l) : rng(seed), dseed(seed + 100), layers(l) {}
    template <class L, class... A> L& add(A&&... a) { layers.push_back(std::make_unique<L>(std::forward<A>(a)...)); return static_cast<L&>(*layers.back()); }
};

class ModelBuilder {
public:
    virtual ~ModelBuilder() = default;
    virtual std::string name() const = 0;
    void build(const ModelConfig& c, LayerList& layers) const {
        BuildCtx ctx(c.seed, layers);
        const int features = buildFeatureExtractor(c, ctx);
        buildHead(c, features, ctx);
    }
    static const ModelBuilder& forKind(Kind k);
protected:
    virtual int buildFeatureExtractor(const ModelConfig& c, BuildCtx& ctx) const = 0;   // returns feature count
    static void buildHead(const ModelConfig& c, int prev, BuildCtx& ctx) {
        for (int h : c.hidden) {
            const float stddev = (c.act == Act::ReLU || c.act == Act::LeakyReLU) ? std::sqrt(2.f / prev) : std::sqrt(1.f / prev);   // He / Xavier
            ctx.add<Dense>((size_t)prev, (size_t)h, stddev, ctx.rng);
            ctx.add<Activation>(c.act);
            if (c.dropout > 0.f) ctx.add<Dropout>(c.dropout, ctx.dseed++);
            prev = h;
        }
        ctx.add<Dense>((size_t)prev, (size_t)c.outputDim, std::sqrt(1.f / prev), ctx.rng);
    }
};

class MLPBuilder : public ModelBuilder {
public:
    std::string name() const override { return "MLP"; }
protected:
    int buildFeatureExtractor(const ModelConfig& c, BuildCtx&) const override { return c.inputDim; }
};

class CNNBuilder : public ModelBuilder {
public:
    std::string name() const override { return "CNN"; }
protected:
    int buildFeatureExtractor(const ModelConfig& c, BuildCtx& ctx) const override {
        int C = c.imgC, H = c.imgH, W = c.imgW;
        if (c.arch == Arch::ResNet) {                       // stem -> residual stages -> global average pool
            const int w0 = c.conv.empty() ? 16 : c.conv[0];
            ctx.add<Conv2D>(C, H, W, w0, 3, 1, 1, ctx.rng);
            if (c.batchNorm) ctx.add<BatchNorm2D>(w0, H * W);
            ctx.add<Activation>(c.act);
            C = w0;
            for (size_t s = 0; s < c.conv.size(); ++s)
                for (int b = 0; b < std::max(1, c.blocks); ++b) {
                    const int stride = (s > 0 && b == 0) ? 2 : 1;
                    ctx.add<ResidualBlock>(C, H, W, c.conv[s], stride, c.act, c.batchNorm, ctx.rng);
                    if (stride == 2) { H = (H - 1) / 2 + 1; W = (W - 1) / 2 + 1; }
                    C = c.conv[s];
                }
            ctx.add<GlobalAvgPool>(C, H * W);
            return C;
        }
        for (int ch : c.conv) {                             // plain: conv -> [bn] -> act -> pool
            ctx.add<Conv2D>(C, H, W, ch, 3, 1, 1, ctx.rng);
            if (c.batchNorm) ctx.add<BatchNorm2D>(ch, H * W);
            ctx.add<Activation>(c.act);
            C = ch;
            if (H >= 2 && W >= 2) { ctx.add<MaxPool2D>(C, H, W, 2); H /= 2; W /= 2; }
        }
        if (c.globalPool) { ctx.add<GlobalAvgPool>(C, H * W); return C; }
        return C * H * W;
    }
};

class TextBuilder : public ModelBuilder {
public:
    std::string name() const override { return "Text"; }
protected:
    int buildFeatureExtractor(const ModelConfig& c, BuildCtx& ctx) const override {
        ctx.add<Embedding>(c.vocab + 1, c.embed, ctx.rng);
        if (c.dropout > 0.f) ctx.add<Dropout>(c.dropout, ctx.dseed++);
        return c.embed;
    }
};

inline const ModelBuilder& ModelBuilder::forKind(Kind k) {
    static const MLPBuilder mlp; static const CNNBuilder cnn; static const TextBuilder text;
    switch (k) { case Kind::CNN: return cnn; case Kind::Text: return text; default: return mlp; }
}

class Model {
public:
    ModelConfig cfg;
    LayerList layers;

    Model() = default;
    explicit Model(const ModelConfig& c) { build(c); }

    void build(const ModelConfig& c) {
        cfg = c; layers.clear();
        ModelBuilder::forKind(cfg.kind).build(cfg, layers);
    }

    Matrix forward(const Matrix& x, bool training) {
        Matrix h = x;
        for (auto& l : layers) h = l->forward(h, training);
        return h;
    }
    void backward(const Matrix& grad) {
        Matrix g = grad;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) g = (*it)->backward(g);
    }
    std::vector<Param> params() {
        std::vector<Param> ps;
        for (auto& l : layers) for (auto& p : l->params()) ps.push_back(p);
        return ps;
    }
    std::vector<Matrix*> buffers() {   // BatchNorm running statistics etc.
        std::vector<Matrix*> bs;
        for (auto& l : layers) for (auto* b : l->buffers()) bs.push_back(b);
        return bs;
    }
    size_t paramCount() {
        size_t n = 0; for (auto& p : params()) n += p.value->size(); return n;
    }
    std::string summary() {
        std::string s = ModelBuilder::forKind(cfg.kind).name() + " model\n";
        char buf[160]; size_t total = 0;
        for (size_t i = 0; i < layers.size(); ++i) {
            const size_t n = layers[i]->paramCount(); total += n;
            std::snprintf(buf, sizeof buf, "  [%2zu] %-52s %9zu params\n", i, layers[i]->describe().c_str(), n); s += buf;
        }
        std::snprintf(buf, sizeof buf, "  total: %zu parameters\n", total); return s + buf;
    }
    std::vector<std::vector<float>> snapshot() {
        std::vector<std::vector<float>> s;
        for (auto& p : params()) s.push_back(p.value->data);
        for (auto* b : buffers()) s.push_back(b->data);
        return s;
    }
    void restore(const std::vector<std::vector<float>>& s) {
        auto ps = params(); size_t i = 0;
        for (; i < ps.size(); ++i) ps[i].value->data = s[i];
        for (auto* b : buffers()) b->data = s[i++];
    }
};

// ---------- binary I/O ----------
namespace io {
template <class T> void wr(std::ostream& o, const T& v) { o.write(reinterpret_cast<const char*>(&v), sizeof(T)); }
template <class T> void rd(std::istream& i, T& v) { i.read(reinterpret_cast<char*>(&v), sizeof(T)); }
inline void wrStr(std::ostream& o, const std::string& s) { uint32_t n = (uint32_t)s.size(); wr(o, n); o.write(s.data(), n); }
inline void rdStr(std::istream& i, std::string& s) { uint32_t n; rd(i, n); s.resize(n); i.read(&s[0], n); }
inline void wrVec(std::ostream& o, const std::vector<float>& v) { uint64_t n = v.size(); wr(o, n); o.write(reinterpret_cast<const char*>(v.data()), n * sizeof(float)); }
inline void rdVec(std::istream& i, std::vector<float>& v) { uint64_t n; rd(i, n); v.resize(n); i.read(reinterpret_cast<char*>(v.data()), n * sizeof(float)); }
inline void wrInts(std::ostream& o, const std::vector<int>& v) { uint32_t n = (uint32_t)v.size(); wr(o, n); for (int x : v) wr(o, x); }
inline void rdInts(std::istream& i, std::vector<int>& v) { uint32_t n; rd(i, n); v.resize(n); for (auto& x : v) rd(i, x); }
inline void wrStrs(std::ostream& o, const std::vector<std::string>& v) { uint32_t n = (uint32_t)v.size(); wr(o, n); for (auto& s : v) wrStr(o, s); }
inline void rdStrs(std::istream& i, std::vector<std::string>& v) { uint32_t n; rd(i, n); v.resize(n); for (auto& s : v) rdStr(i, s); }
}

// Format MNN3 (current). MNN2 files from the previous version are still readable.
inline void saveModel(const std::string& path, Model& m, const Preprocessor& prep) {
    std::ofstream o(path, std::ios::binary);
    if (!o) throw std::runtime_error("cannot write " + path);
    const ModelConfig& c = m.cfg;
    o.write("MNN3", 4);
    io::wr(o, (int)c.kind); io::wr(o, c.inputDim); io::wr(o, c.outputDim);
    io::wrInts(o, c.hidden);
    io::wr(o, (int)c.act); io::wr(o, c.dropout); io::wr(o, (int)c.task); io::wr(o, c.seed);
    io::wr(o, c.imgC); io::wr(o, c.imgH); io::wr(o, c.imgW); io::wrInts(o, c.conv);
    io::wr(o, c.vocab); io::wr(o, c.embed); io::wr(o, c.maxLen); io::wr(o, c.ngrams);
    io::wr(o, (int)c.arch); io::wr(o, (int)c.batchNorm); io::wr(o, (int)c.globalPool); io::wr(o, c.blocks); io::wr(o, c.fit);
    io::wr(o, prep.normMode); io::wr(o, prep.chans); io::wrVec(o, prep.mean); io::wrVec(o, prep.stdv);
    io::wr(o, prep.yMean); io::wr(o, prep.yStd);
    io::wrStrs(o, prep.classes); io::wrStrs(o, prep.vocab);
    for (auto& p : m.params()) io::wrVec(o, p.value->data);
    for (auto* b : m.buffers()) io::wrVec(o, b->data);
}

inline void loadModel(const std::string& path, Model& m, Preprocessor& prep) {
    std::ifstream i(path, std::ios::binary);
    if (!i) throw std::runtime_error(cannotOpenMsg("model file", path));
    char magic[4]; i.read(magic, 4);
    const bool v3 = std::memcmp(magic, "MNN3", 4) == 0, v2 = std::memcmp(magic, "MNN2", 4) == 0;
    if (!v3 && !v2) throw std::runtime_error("not a minnn model file (expected MNN2/MNN3)");
    ModelConfig c; int kind, act, task;
    io::rd(i, kind); io::rd(i, c.inputDim); io::rd(i, c.outputDim);
    io::rdInts(i, c.hidden);
    io::rd(i, act); io::rd(i, c.dropout); io::rd(i, task); io::rd(i, c.seed);
    io::rd(i, c.imgC); io::rd(i, c.imgH); io::rd(i, c.imgW); io::rdInts(i, c.conv);
    io::rd(i, c.vocab); io::rd(i, c.embed); io::rd(i, c.maxLen); io::rd(i, c.ngrams);
    if (v3) { int arch, bn, gp; io::rd(i, arch); io::rd(i, bn); io::rd(i, gp); io::rd(i, c.blocks); io::rd(i, c.fit); c.arch = (Arch)arch; c.batchNorm = bn != 0; c.globalPool = gp != 0; }
    c.kind = (Kind)kind; c.act = (Act)act; c.task = (Task)task;
    io::rd(i, prep.normMode); if (v3) io::rd(i, prep.chans); else prep.chans = 1;
    io::rdVec(i, prep.mean); io::rdVec(i, prep.stdv);
    io::rd(i, prep.yMean); io::rd(i, prep.yStd);
    io::rdStrs(i, prep.classes); io::rdStrs(i, prep.vocab);
    m.build(c);
    for (auto& p : m.params()) io::rdVec(i, p.value->data);
    if (v3) for (auto* b : m.buffers()) io::rdVec(i, b->data);
    if (!i) throw std::runtime_error("model file truncated/corrupt");
}

} // namespace nn
