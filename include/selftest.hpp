#pragma once
// `minnn selftest` - verifies every component: kernels, backprop (finite differences), tokenizer,
// KV-cache inference vs training forward, save/load, model growth, optimizer resume, CSV/text parsing.
#include "lm.hpp"
#include "trainer.hpp"
#include "command.hpp"
#include "image.hpp"
#include <sstream>

namespace nn {
namespace selftest {

struct Ctx { int total = 0, failed = 0; };
inline void report(Ctx& c, const char* name, bool ok, const std::string& detail = "") {
    ++c.total; if (!ok) ++c.failed;
    std::printf("  [%s] %-46s %s\n", ok ? "PASS" : "FAIL", name, detail.c_str());
}
inline std::string fmt(const char* f, double v) { char b[64]; std::snprintf(b, sizeof b, f, v); return b; }

inline Matrix randMat(size_t r, size_t c, std::mt19937& g, float s = 1.f) {
    Matrix m(r, c); std::normal_distribution<float> nd(0.f, s); for (auto& v : m.data) v = nd(g); return m;
}

// ---- 1. matmul kernels vs naive double-precision reference ----
inline void testMatmul(Ctx& c) {
    std::mt19937 g(1); double worst = 0;
    const size_t dims[][3] = {{1, 1, 1}, {5, 7, 3}, {13, 17, 9}, {64, 33, 70}, {3, 130, 4}};
    for (auto& d : dims) {
        size_t n = d[0], k = d[1], m = d[2];
        Matrix A = randMat(n, k, g), B = randMat(k, m, g), C, R(n, m);
        matmul(A, B, C);
        for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < m; ++j) { double s = 0; for (size_t t = 0; t < k; ++t) s += (double)A(i, t) * B(t, j); R(i, j) = (float)s; }
        for (size_t i = 0; i < C.size(); ++i) worst = std::max(worst, (double)std::fabs(C.data[i] - R.data[i]));
        Matrix At = randMat(n, k, g), Bt = randMat(n, m, g), C2, R2(k, m);              // At^T * Bt
        matmulAtB(At, Bt, C2);
        for (size_t i = 0; i < k; ++i) for (size_t j = 0; j < m; ++j) { double s = 0; for (size_t t = 0; t < n; ++t) s += (double)At(t, i) * Bt(t, j); R2(i, j) = (float)s; }
        for (size_t i = 0; i < C2.size(); ++i) worst = std::max(worst, (double)std::fabs(C2.data[i] - R2.data[i]));
        Matrix Ab = randMat(n, k, g), Bb = randMat(m, k, g), C3, R3(n, m);              // Ab * Bb^T
        matmulABt(Ab, Bb, C3);
        for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < m; ++j) { double s = 0; for (size_t t = 0; t < k; ++t) s += (double)Ab(i, t) * Bb(j, t); R3(i, j) = (float)s; }
        for (size_t i = 0; i < C3.size(); ++i) worst = std::max(worst, (double)std::fabs(C3.data[i] - R3.data[i]));
    }
    report(c, "matmul / matmulAtB / matmulABt vs reference", worst < 1e-3, "max abs err " + fmt("%.1e", worst));
}

// ---- 2. finite-difference gradient checks ----
// A sample counts only if numeric estimates at 3 step sizes agree with each other (otherwise the loss has a
// kink there, e.g. a max-pool tie flips, and finite differences are meaningless). When they agree, the analytic
// gradient must match. Returns worst |ana-num| / tolerance (pass if < 1) and how many samples were usable.
inline bool nearTol(double a, double b, double at = 2e-3, double rt = 0.05) { return std::fabs(a - b) <= at + rt * std::max(std::fabs(a), std::fabs(b)); }
inline double gradCheckModel(Model& m, const Matrix& X, const Matrix& Y, int& used, int& total, bool training = false, double absTol = 2e-3, double relTol = 0.05) {
    SoftmaxCrossEntropy loss;
    loss.forward(m.forward(X, training), Y); m.backward(loss.backward());
    auto ps = m.params(); std::mt19937 r(3); double worst = 0; used = total = 0;
    const float epss[3] = {1e-2f, 3e-3f, 1e-3f};
    for (auto& p : ps) for (int k = 0; k < 5; ++k) {
        size_t i = r() % p.value->size(); float o = p.value->data[i], ana = p.grad->data[i]; double num[3];
        for (int e = 0; e < 3; ++e) {
            SoftmaxCrossEntropy l2;
            p.value->data[i] = o + epss[e]; float lp = l2.forward(m.forward(X, training), Y);
            p.value->data[i] = o - epss[e]; float lm = l2.forward(m.forward(X, training), Y);
            p.value->data[i] = o; num[e] = (lp - lm) / (2 * epss[e]);
        }
        ++total;
        if (!(nearTol(num[0], num[1], absTol, relTol) && nearTol(num[1], num[2], absTol, relTol) && nearTol(num[0], num[2], absTol, relTol))) continue;   // kink: skip
        ++used;
        double mid = num[1], tol = absTol + relTol * std::max(std::fabs((double)ana), std::fabs(mid));
        worst = std::max(worst, std::fabs(ana - mid) / tol);
    }
    return worst;
}

// MaxPool2D alone, with distinct values (gap 0.1) and eps = 0.02 < gap/2, so no argmax can flip: exact check.
inline double maxPoolCheck() {
    std::mt19937 g(4); MaxPool2D pool(2, 6, 6, 2);
    const size_t B = 3, D = 2 * 6 * 6; Matrix X(B, D);
    for (size_t b = 0; b < B; ++b) { std::vector<float> v(D); std::iota(v.begin(), v.end(), 0.f); std::shuffle(v.begin(), v.end(), g); for (size_t j = 0; j < D; ++j) X(b, j) = 0.1f * v[j]; }
    Matrix Gup = randMat(B, 2 * 3 * 3, g);
    auto lossFn = [&]() { Matrix y = pool.forward(X, false); double s = 0; for (size_t i = 0; i < y.size(); ++i) s += (double)Gup.data[i] * y.data[i]; return s; };
    pool.forward(X, true); Matrix dx = pool.backward(Gup); double worst = 0; const float eps = 0.02f;
    for (size_t i = 0; i < X.size(); ++i) {
        float o = X.data[i]; X.data[i] = o + eps; double lp = lossFn(); X.data[i] = o - eps; double lm = lossFn(); X.data[i] = o;
        worst = std::max(worst, std::fabs(dx.data[i] - (lp - lm) / (2 * eps)));
    }
    return worst;
}

// Conv2D alone with loss = 0.5*sum(y^2): exactly quadratic, so central differences are exact.
inline double conv2dCheck() {
    std::mt19937 g(11); double worst = 0; const float eps = 1e-2f;
    for (int stride : {1, 2}) for (int pad : {0, 1}) {
        Conv2D conv(2, 7, 6, 3, 3, stride, pad, g); Matrix X = randMat(3, 2 * 7 * 6, g);
        auto lossFn = [&]() { Matrix y = conv.forward(X, false); double s = 0; for (float v : y.data) s += 0.5 * v * v; return s; };
        Matrix y = conv.forward(X, true); Matrix dx = conv.backward(y);
        for (auto& p : conv.params()) for (int k = 0; k < 12; ++k) {
            size_t i = g() % p.value->size(); float o = p.value->data[i], ana = p.grad->data[i];
            p.value->data[i] = o + eps; double lp = lossFn(); p.value->data[i] = o - eps; double lm = lossFn(); p.value->data[i] = o;
            worst = std::max(worst, std::fabs(ana - (lp - lm) / (2 * eps)) / (std::fabs(ana) + 1e-2));
        }
        for (int k = 0; k < 20; ++k) {
            size_t i = g() % X.size(); float o = X.data[i], ana = dx.data[i];
            X.data[i] = o + eps; double lp = lossFn(); X.data[i] = o - eps; double lm = lossFn(); X.data[i] = o;
            worst = std::max(worst, std::fabs(ana - (lp - lm) / (2 * eps)) / (std::fabs(ana) + 1e-2));
        }
    }
    return worst;
}

inline void testLayerGrads(Ctx& c) {
    std::mt19937 g(2);
    auto onehot = [&](size_t n, int k) { Matrix Y(n, k); for (size_t i = 0; i < n; ++i) Y(i, g() % k) = 1.f; return Y; };
    auto detail = [](double w, int used, int total) { return fmt("%.2f", w) + " x tol, " + std::to_string(used) + "/" + std::to_string(total) + " samples usable"; };
    int used, total;
    {   ModelConfig cf; cf.kind = Kind::MLP; cf.inputDim = 5; cf.outputDim = 3; cf.hidden = {7, 6}; cf.act = Act::Tanh;
        Model m(cf); Matrix X = randMat(4, 5, g); Matrix Y = onehot(4, 3);
        double e = gradCheckModel(m, X, Y, used, total);
        report(c, "backprop: Dense + Activation", e < 1.0 && used >= total * 0.7, detail(e, used, total)); }
    {   double e = conv2dCheck();
        report(c, "backprop: Conv2D (stride 1/2, pad 0/1) exact", e < 2e-3, "rel err " + fmt("%.1e", e)); }
    {   double e = maxPoolCheck();
        report(c, "backprop: MaxPool2D exact", e < 1e-3, "abs err " + fmt("%.1e", e)); }
    {   ModelConfig cf; cf.kind = Kind::CNN; cf.imgC = 2; cf.imgH = 6; cf.imgW = 6; cf.inputDim = 72; cf.outputDim = 3;
        cf.conv = {3, 4}; cf.hidden = {5}; cf.act = Act::Tanh;
        Model m(cf); Matrix X = randMat(3, 72, g); Matrix Y = onehot(3, 3);
        double e = gradCheckModel(m, X, Y, used, total);
        report(c, "backprop: Conv2D + MaxPool2D + Dense (full)", e < 1.0 && used >= total * 0.7, detail(e, used, total)); }
    {   ModelConfig cf; cf.kind = Kind::Text; cf.vocab = 10; cf.embed = 6; cf.maxLen = 5; cf.inputDim = 5; cf.outputDim = 3;
        cf.hidden = {4}; cf.act = Act::Tanh;
        Model m(cf); Matrix X(4, 5); for (auto& v : X.data) v = (float)(g() % 11);
        Matrix Y = onehot(4, 3);
        double e = gradCheckModel(m, X, Y, used, total);
        report(c, "backprop: Embedding + Dense", e < 1.0 && used >= total * 0.7, detail(e, used, total)); }
}

// ---- 2b. image processing (OOP module) ----
inline Image rampX(int C, int H, int W) {      // value = x / (W-1) in every row/channel
    Image im(C, H, W); for (int c = 0; c < C; ++c) for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) im.at(c, y, x) = (float)x / (W - 1);
    return im;
}
inline void testImageProc(Ctx& c) {
    std::mt19937 g(5);
    {   Image k(3, 9, 7, 0.37f); double dev = 0;
        for (auto hw : std::vector<std::pair<int, int>>{{3, 2}, {20, 15}, {9, 7}, {4, 13}, {1, 1}}) { Image r = k.resized(hw.first, hw.second); for (float v : r.px) dev = std::max(dev, (double)std::fabs(v - 0.37f)); }
        report(c, "image: resize keeps a constant image constant", dev < 1e-5, "max dev " + fmt("%.1e", dev)); }
    {   Image a(1, 4, 4); for (int i = 0; i < 16; ++i) a.px[i] = i / 16.f;
        Image r = a.resized(2, 2); double dev = 0;
        for (int y = 0; y < 2; ++y) for (int x = 0; x < 2; ++x) { double m = (a.at(0, 2*y, 2*x) + a.at(0, 2*y, 2*x+1) + a.at(0, 2*y+1, 2*x) + a.at(0, 2*y+1, 2*x+1)) / 4.0; dev = std::max(dev, std::fabs(r.at(0, y, x) - m)); }
        report(c, "image: 2x downscale = exact block average", dev < 1e-6, "max dev " + fmt("%.1e", dev)); }
    {   Image cb(1, 16, 16); for (int y = 0; y < 16; ++y) for (int x = 0; x < 16; ++x) cb.at(0, y, x) = (x + y) % 2;
        Image r = cb.resized(4, 4); double dev = 0; for (float v : r.px) dev = std::max(dev, (double)std::fabs(v - 0.5f));
        report(c, "image: downscale is anti-aliased (checkerboard -> 0.5)", dev < 1e-5, "max dev " + fmt("%.1e", dev)); }
    {   Image a = rampX(2, 5, 6); Image r = a.resized(5, 6); bool same = r.px == a.px;
        report(c, "image: resize to same size is identity", same); }
    {   // columns 0-2 and 9-11 are bright borders, the middle is 0.5
        Image src(1, 6, 12, 0.5f); for (int y = 0; y < 6; ++y) for (int x = 0; x < 12; ++x) if (x < 3 || x >= 9) src.at(0, y, x) = 1.f;
        Image st = src.fitted(8, 8, FitMode::Stretch), cr = src.fitted(8, 8, FitMode::Crop), pd = src.fitted(8, 8, FitMode::Pad);
        double cdev = 0; for (float v : cr.px) cdev = std::max(cdev, (double)std::fabs(v - 0.5f));
        const float mean = src.channelMeans()[0];
        bool shapes = st.H == 8 && st.W == 8 && cr.H == 8 && cr.W == 8 && pd.H == 8 && pd.W == 8;
        bool ok = shapes && cdev < 1e-5 && st.at(0, 3, 0) > 0.9f && std::fabs(pd.at(0, 0, 0) - mean) < 1e-6 && std::fabs(pd.at(0, 7, 7) - mean) < 1e-6 && pd.at(0, 3, 0) > 0.9f;
        report(c, "image: fit modes stretch / crop / pad", ok, "crop dev " + fmt("%.1e", cdev)); }
    {   Image a = rampX(2, 4, 5); report(c, "image: flip twice = identity", a.flippedH().flippedH().px == a.px); }
}

inline void testTransforms(Ctx& c) {
    std::mt19937 g(6);
    {   Image a = rampX(2, 5, 6); Image b = a; RandomHFlip(0.f).apply(b, g);
        Image d = a; RandomHFlip(1.f).apply(d, g); RandomHFlip(1.f).apply(d, g);
        report(c, "transform: HFlip p=0 no-op, p=1 twice = identity", b.px == a.px && d.px == a.px); }
    {   Image a = rampX(1, 9, 9); Image b = a; RandomAffine(0.f, 1.f, 1.f, 0.f).apply(b, g);
        Image z = a; RandomAffine(0.f, 2.f, 2.f, 0.f).apply(z, g);         // fixed zoom 2x, no rotation / shift
        const double e = std::max({std::fabs(z.at(0, 4, 0) - 0.25), std::fabs(z.at(0, 4, 8) - 0.75), std::fabs(z.at(0, 4, 4) - 0.5)});
        report(c, "transform: Affine identity exact, zoom 2x correct", b.px == a.px && e < 1e-5, "zoom err " + fmt("%.1e", e)); }
    {   Image a(3, 12, 12); std::uniform_real_distribution<float> u(0.f, 1.f); for (auto& v : a.px) v = u(g);
        auto build = [&]() { Compose p; p.add(std::make_unique<RandomAffine>(20.f, 0.8f, 1.2f, 0.1f)); p.add(std::make_unique<ColorJitter>(0.2f, 0.2f)); p.add(std::make_unique<GaussianNoise>(0.05f)); p.add(std::make_unique<Cutout>(0.4f, 1.f)); return p; };
        Compose p = build(); Image x1 = a, x2 = a, x3 = a; std::mt19937 r1(11), r2(11), r3(12);
        p.apply(x1, r1); p.apply(x2, r2); p.apply(x3, r3);
        bool range = true; for (float v : x1.px) range = range && v >= 0.f && v <= 1.f && std::isfinite(v);
        report(c, "transform: pipeline reproducible per seed, stays in [0,1]", x1.px == x2.px && x1.px != x3.px && x1.px != a.px && range, p.describe().substr(0, 40) + "..."); }
    {   Matrix X(40, 2 * 6 * 6); std::uniform_real_distribution<float> u(0.f, 1.f); for (size_t i = 0; i < X.size(); ++i) X.data[i] = (i % 72 < 36 ? 0.3f : 0.7f) * u(g) + 0.1f * (i % 72 < 36);
        const Matrix raw = X; Preprocessor pr; pr.fitChannels(X, 2); pr.apply(X);
        double worstMean = 0, worstStd = 0;
        for (int ch = 0; ch < 2; ++ch) { double m = 0, q = 0; for (size_t i = 0; i < X.rows; ++i) for (int k = 0; k < 36; ++k) m += X.data[i * 72 + ch * 36 + k]; m /= X.rows * 36;
            for (size_t i = 0; i < X.rows; ++i) { for (int k = 0; k < 36; ++k) { double t = X.data[i * 72 + ch * 36 + k] - m; q += t * t; } }
            q = std::sqrt(q / (X.rows * 36));
            worstMean = std::max(worstMean, std::fabs(m)); worstStd = std::max(worstStd, std::fabs(q - 1)); }
        Matrix back = X; for (size_t i = 0; i < back.rows; ++i) pr.normalizeRow(&back.data[i * 72], 72, true);
        double rt = 0; for (size_t i = 0; i < back.size(); ++i) rt = std::max(rt, (double)std::fabs(back.data[i] - raw.data[i]));
        report(c, "normalizer: per-channel mean 0 / std 1, exact inverse", worstMean < 1e-4 && worstStd < 1e-4 && rt < 1e-5, "round-trip err " + fmt("%.1e", rt)); }
    {   Matrix X(6, 2 * 5 * 5); std::uniform_real_distribution<float> u(0.f, 1.f); for (auto& v : X.data) v = u(g);
        Preprocessor pr; pr.fitChannels(X, 2); pr.apply(X);
        ImageAugmenter none(2, 5, 5), flip(2, 5, 5); flip.pipeline().add(std::make_unique<RandomHFlip>(1.f));
        Matrix A = X, Bm = X; none.apply(A, pr, g); flip.apply(Bm, pr, g);
        double e0 = 0, e1 = 0;
        for (size_t i = 0; i < X.rows; ++i) { Image orig = Image::fromRow(&X.data[i * 50], 2, 5, 5), fl = orig.flippedH();
            for (size_t k = 0; k < 50; ++k) { e0 = std::max(e0, (double)std::fabs(A.data[i * 50 + k] - X.data[i * 50 + k])); e1 = std::max(e1, (double)std::fabs(Bm.data[i * 50 + k] - fl.px[k])); } }
        report(c, "augmenter: de-normalize -> augment -> re-normalize", e0 == 0 && e1 < 1e-4, "flip err " + fmt("%.1e", e1)); }
}

// ---- 2c. BatchNorm / ResNet / builders / MNN3 format ----
inline void testBatchNormResNet(Ctx& c) {
    std::mt19937 g(8);
    {   BatchNorm2D bn(3, 16); Matrix x = randMat(5, 48, g, 2.f);
        for (size_t b = 0; b < 5; ++b) for (int ch = 0; ch < 3; ++ch) for (int k = 0; k < 16; ++k) x.data[b * 48 + ch * 16 + k] += 3.f * (ch + 1);
        Matrix y = bn.forward(x, true); double wm = 0, wv = 0;
        for (int ch = 0; ch < 3; ++ch) { double m = 0, q = 0; for (size_t b = 0; b < 5; ++b) for (int k = 0; k < 16; ++k) m += y.data[b * 48 + ch * 16 + k]; m /= 80;
            for (size_t b = 0; b < 5; ++b) { for (int k = 0; k < 16; ++k) { double t = y.data[b * 48 + ch * 16 + k] - m; q += t * t; } }
            q /= 80; wm = std::max(wm, std::fabs(m)); wv = std::max(wv, std::fabs(q - 1)); }
        for (int it = 0; it < 80; ++it) bn.forward(x, true);                         // let running statistics converge
        Matrix ye = bn.forward(x, false), yt = bn.forward(x, true); double d = 0; for (size_t i = 0; i < ye.size(); ++i) d = std::max(d, (double)std::fabs(ye.data[i] - yt.data[i]));
        report(c, "BatchNorm2D: batch stats 0/1, eval uses running stats", wm < 1e-4 && wv < 1e-3 && d < 0.05, "train/eval gap " + fmt("%.3f", d)); }
    {   ModelConfig cf; cf.kind = Kind::CNN; cf.arch = Arch::ResNet; cf.batchNorm = true; cf.globalPool = true; cf.imgC = 2; cf.imgH = 6; cf.imgW = 6; cf.inputDim = 72;
        cf.outputDim = 3; cf.conv = {3, 4}; cf.blocks = 1; cf.hidden = {}; cf.act = Act::Tanh;
        Model m(cf); Matrix X = randMat(4, 72, g); Matrix Y(4, 3); for (size_t i = 0; i < 4; ++i) Y(i, g() % 3) = 1.f;
        int used, total; double e = gradCheckModel(m, X, Y, used, total, true, 2e-4, 0.03);   // tight: gradients here are ~1e-2
        report(c, "backprop: ResNet (BatchNorm + residual + GAP)", e < 1.0 && used >= total * 0.9, fmt("%.2f", e) + " x tol, " + std::to_string(used) + "/" + std::to_string(total) + " samples usable"); }
    {   ModelConfig cf; cf.kind = Kind::CNN; cf.arch = Arch::ResNet; cf.batchNorm = true; cf.imgC = 3; cf.imgH = 8; cf.imgW = 8; cf.inputDim = 192; cf.outputDim = 4;
        cf.conv = {4, 8, 16}; cf.blocks = 2; cf.hidden = {6}; Model m(cf);
        Matrix out = m.forward(randMat(2, 192, g), false);
        size_t sum = 0; for (auto& l : m.layers) sum += l->paramCount();
        const std::string sm = m.summary();
        bool ok = out.rows == 2 && out.cols == 4 && sum == m.paramCount() && sm.find("ResidualBlock") != std::string::npos && sm.find("GlobalAvgPool") != std::string::npos && sm.find("BatchNorm2D") != std::string::npos;
        report(c, "builders: ResNet shape flow + summary()", ok, std::to_string(m.paramCount()) + " params"); }
    {   ModelConfig cf; cf.kind = Kind::CNN; cf.arch = Arch::ResNet; cf.batchNorm = true; cf.imgC = 1; cf.imgH = 6; cf.imgW = 6; cf.inputDim = 36; cf.outputDim = 2; cf.conv = {3}; cf.hidden = {};
        Model m(cf); Matrix X = randMat(5, 36, g, 3.f); auto snap = m.snapshot(); for (int i = 0; i < 6; ++i) m.forward(X, true);   // buffers drift
        bool drifted = m.snapshot() != snap; m.restore(snap); bool back = m.snapshot() == snap;
        report(c, "model snapshot/restore includes BatchNorm statistics", drifted && back); }
    {   bool ok = ModelBuilder::forKind(Kind::MLP).name() == "MLP" && ModelBuilder::forKind(Kind::CNN).name() == "CNN" && ModelBuilder::forKind(Kind::Text).name() == "Text";
        report(c, "builders: factory returns the right subclass", ok); }
}

inline void testModelFiles(Ctx& c) {
    std::mt19937 g(9); const std::string dir = std::filesystem::temp_directory_path().string();
    {   ModelConfig cf; cf.kind = Kind::CNN; cf.arch = Arch::ResNet; cf.batchNorm = true; cf.globalPool = true; cf.imgC = 2; cf.imgH = 6; cf.imgW = 6; cf.inputDim = 72;
        cf.outputDim = 3; cf.conv = {3, 4}; cf.blocks = 2; cf.hidden = {5}; cf.fit = (int)FitMode::Pad; cf.act = Act::LeakyReLU; Model a(cf);
        Matrix X = randMat(6, 72, g, 2.f); for (int i = 0; i < 12; ++i) a.forward(X, true);      // move running statistics away from init
        Preprocessor p; p.classes = {"x", "y", "z"}; p.normMode = 3; p.chans = 2; p.mean = {0.1f, 0.2f}; p.stdv = {1.5f, 2.5f};
        std::string path = dir + "/t_v3.mnn"; saveModel(path, a, p);
        Model b; Preprocessor q; loadModel(path, b, q);
        Matrix ya = a.forward(X, false), yb = b.forward(X, false); bool same = ya.data == yb.data;
        bool cfgOk = b.cfg.arch == Arch::ResNet && b.cfg.batchNorm && b.cfg.blocks == 2 && b.cfg.fit == (int)FitMode::Pad && q.normMode == 3 && q.chans == 2 && q.mean == p.mean && q.stdv == p.stdv;
        std::filesystem::remove(path);
        report(c, "save/load MNN3: ResNet + BN running stats + fit + per-channel norm", same && cfgOk); }
    {   ModelConfig cf; cf.kind = Kind::CNN; cf.imgC = 1; cf.imgH = 6; cf.imgW = 6; cf.inputDim = 36; cf.outputDim = 3; cf.conv = {3}; cf.hidden = {5}; Model a(cf);
        Preprocessor p; p.classes = {"a", "b", "c"}; p.normMode = 2; p.mean = {0.3f}; p.stdv = {1.5f};
        std::string path = dir + "/t_v2.mnn"; { std::ofstream o(path, std::ios::binary); const ModelConfig& k = a.cfg; o.write("MNN2", 4);   // byte-exact old writer
            io::wr(o, (int)k.kind); io::wr(o, k.inputDim); io::wr(o, k.outputDim); io::wrInts(o, k.hidden); io::wr(o, (int)k.act); io::wr(o, k.dropout); io::wr(o, (int)k.task); io::wr(o, k.seed);
            io::wr(o, k.imgC); io::wr(o, k.imgH); io::wr(o, k.imgW); io::wrInts(o, k.conv); io::wr(o, k.vocab); io::wr(o, k.embed); io::wr(o, k.maxLen); io::wr(o, k.ngrams);
            io::wr(o, p.normMode); io::wrVec(o, p.mean); io::wrVec(o, p.stdv); io::wr(o, p.yMean); io::wr(o, p.yStd); io::wrStrs(o, p.classes); io::wrStrs(o, p.vocab);
            for (auto& pr : a.params()) io::wrVec(o, pr.value->data); }
        Model b; Preprocessor q; loadModel(path, b, q); Matrix X = randMat(3, 36, g);
        bool ok = a.forward(X, false).data == b.forward(X, false).data && b.cfg.arch == Arch::Plain && !b.cfg.batchNorm && q.chans == 1 && q.normMode == 2;
        std::filesystem::remove(path);
        report(c, "save/load: old MNN2 files still load (compatibility)", ok); }
}

// ---- 2d. real image files end to end: folder loader, fit, per-channel norm, augmenter, BatchNorm CNN ----
inline void testImageEndToEnd(Ctx& c) {
    namespace fs = std::filesystem; std::mt19937 g(10);
    const fs::path root = fs::temp_directory_path() / "minnn_selftest_imgs"; fs::remove_all(root);
    for (const char* cls : {"top", "bottom"}) fs::create_directories(root / cls);
    std::uniform_int_distribution<int> noise(0, 40);
    for (int k = 0; k < 24; ++k) for (const char* cls : {"top", "bottom"}) {       // non-square 16x24 PGM files, bright block in the top / bottom half
        const int H = 16, W = 24; std::vector<unsigned char> px(H * W);
        const int y0 = std::string(cls) == "top" ? 1 : 9, x0 = (int)(g() % 8) + 4;
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) { bool in = y >= y0 && y < y0 + 6 && x >= x0 && x < x0 + 8; px[y * W + x] = (unsigned char)(noise(g) + (in ? 190 : 0)); }
        std::ofstream o(root / cls / ("img" + std::to_string(k) + ".pgm"), std::ios::binary); o << "P5\n" << W << " " << H << "\n255\n"; o.write((const char*)px.data(), px.size());
    }
    Preprocessor prep; Dataset all = loadImageFolder(root.string(), 1, 8, 8, prep, FitMode::Crop);
    bool loaded = all.size() == 48 && all.X.cols == 64 && prep.classes.size() == 2 && prep.classes[0] == "bottom";
    Dataset tr, va; trainValSplit(all, 0.25f, 3, tr, va); prep.fitChannels(tr.X, 1); prep.apply(tr.X); prep.apply(va.X);
    ModelConfig cf; cf.kind = Kind::CNN; cf.imgC = 1; cf.imgH = 8; cf.imgW = 8; cf.inputDim = 64; cf.outputDim = 2; cf.conv = {4, 8}; cf.hidden = {8}; cf.batchNorm = true; cf.act = Act::ReLU;
    Model m(cf); ImageAugmenter aug(1, 8, 8); aug.pipeline().add(std::make_unique<RandomAffine>(8.f, 0.95f, 1.05f, 0.05f)); aug.pipeline().add(std::make_unique<ColorJitter>(0.1f, 0.1f));
    Adam opt(5e-3f); auto ps = m.params(); SoftmaxCrossEntropy loss; std::vector<size_t> idx(tr.size()); std::iota(idx.begin(), idx.end(), 0); std::mt19937 rr(1);
    for (int ep = 0; ep < 40; ++ep) { std::shuffle(idx.begin(), idx.end(), rr);
        for (size_t b = 0; b < idx.size(); b += 12) { size_t e = std::min(idx.size(), b + 12); Matrix X = makeBatchX(tr, idx, b, e); aug.apply(X, prep, rr);
            Matrix Y = makeBatchTargets(tr, idx, b, e, Task::Classification, 2, prep); loss.forward(m.forward(X, true), Y); m.backward(loss.backward()); opt.step(ps); } }
    std::vector<size_t> vi(va.size()); std::iota(vi.begin(), vi.end(), 0); Matrix out = m.forward(makeBatchX(va, vi, 0, vi.size()), false); size_t right = 0;
    for (size_t i = 0; i < out.rows; ++i) right += (out(i, 0) > out(i, 1)) == (va.cls[i] == 0);
    const double acc = (double)right / out.rows; fs::remove_all(root);
    report(c, "images end-to-end: PGM folder -> crop -> BN-CNN + augmentation", loaded && acc >= 0.9, "val accuracy " + fmt("%.2f", acc)); }

// ---- 2e. object-oriented layer: Trainer + callbacks, command registry, image sources ----
struct SpyCallback : TrainCallback {
    std::vector<int> epochs; int stopAt; bool began = false, ended = false;
    explicit SpyCallback(int stop) : stopAt(stop) {}
    void onTrainBegin(Model&) override { began = true; }
    bool onEpochEnd(Model&, const EpochStats& s) override { epochs.push_back(s.epoch); return s.epoch < stopAt; }
    void onTrainEnd(Model&) override { ended = true; }
};
inline Dataset makeBlobs(std::mt19937& g, size_t n) {
    Dataset d; d.X = Matrix(n, 2); std::normal_distribution<float> nd(0.f, 0.5f);
    for (size_t i = 0; i < n; ++i) { int k = (int)(i % 2); d.X(i, 0) = (k ? 1.5f : -1.5f) + nd(g); d.X(i, 1) = (k ? -1.f : 1.f) + nd(g); d.cls.push_back(k); }
    return d;
}
inline ModelConfig blobConfig() { ModelConfig cf; cf.kind = Kind::MLP; cf.inputDim = 2; cf.outputDim = 2; cf.hidden = {8}; cf.act = Act::Tanh; return cf; }
inline void writeBE32(std::ostream& o, uint32_t v) { unsigned char b[4] = {(unsigned char)(v >> 24), (unsigned char)(v >> 16), (unsigned char)(v >> 8), (unsigned char)v}; o.write((const char*)b, 4); }

inline void testOOP(Ctx& c) {
    namespace fs = std::filesystem; std::mt19937 g(21);
    Dataset train = makeBlobs(g, 160), val = makeBlobs(g, 60); Preprocessor prep;

    {   // callbacks are called in order, every epoch, and can stop the run
        Model m(blobConfig()); TrainOptions o; o.epochs = 20; o.batch = 16; o.lr = 1e-2f; o.verbose = false;
        Trainer t(m, prep, o); auto spy = std::make_shared<SpyCallback>(3); t.addCallback(spy); t.fit(train, val);
        report(c, "Trainer: callbacks see every epoch and can stop training", spy->began && spy->ended && spy->epochs == std::vector<int>({1, 2, 3}),
               "epochs seen: " + std::to_string(spy->epochs.size())); }
    {   // fit() wrapper == Trainer, and the model really learns
        TrainOptions o; o.epochs = 30; o.batch = 16; o.lr = 1e-2f; o.verbose = false; o.patience = 0;
        Model m1(blobConfig()), m2(blobConfig());
        fit(m1, train, val, prep, o); Trainer(m2, prep, o).fit(train, val);
        double diff = 0; auto p1 = m1.params(), p2 = m2.params();
        for (size_t i = 0; i < p1.size(); ++i) for (size_t j = 0; j < p1[i].value->size(); ++j) diff = std::max(diff, (double)std::fabs(p1[i].value->data[j] - p2[i].value->data[j]));
        EvalResult r = evaluate(m1, val, prep);
        report(c, "Trainer: fit() wrapper identical; learns 2 blobs", diff == 0.0 && r.metric >= 0.95, "val acc " + fmt("%.2f", r.metric) + ", weight diff " + fmt("%.0e", diff)); }
    {   // EarlyStopping: patience counting, reset on improvement, restore of the best weights
        Model m(blobConfig()); EarlyStopping es(2, false); es.onTrainBegin(m);
        EpochStats st; st.valLoss = 1.f; es.onEpochEnd(m, st); auto W1 = m.snapshot();
        for (auto& p : m.params()) for (auto& v : p.value->data) v += 1.f;              // "training made the model worse"
        st.valLoss = 2.f; bool c1 = es.onEpochEnd(m, st);
        st.valLoss = 3.f; bool c2 = es.onEpochEnd(m, st);
        es.onTrainEnd(m);
        bool restored = m.snapshot() == W1;
        EarlyStopping e2(2, false); e2.onTrainBegin(m); bool k = true;
        for (float v : {1.f, 2.f, 0.5f, 0.6f}) { st.valLoss = v; k = e2.onEpochEnd(m, st) && k; }      // improvement at 0.5 resets the counter
        st.valLoss = 0.7f; bool stops = !e2.onEpochEnd(m, st);
        EarlyStopping e0(0, false); e0.onTrainBegin(m); bool never = true;
        for (int i = 0; i < 50; ++i) { st.valLoss = 1.f + i; never = never && e0.onEpochEnd(m, st); }  // patience 0 = never stops
        report(c, "EarlyStopping: patience, counter reset, restores best weights", c1 && !c2 && restored && k && stops && never && es.bestLoss() == 1.f); }
    {   // CheckpointSaver writes only when validation loss improves, and the file loads back identically
        const std::string path = (fs::temp_directory_path() / "minnn_selftest_ckpt.bin").string(); fs::remove(path);
        Model m(blobConfig()); CheckpointSaver cs(path, prep, false); EpochStats st;
        st.valLoss = 1.f; cs.onEpochEnd(m, st); bool w1 = fs::exists(path); fs::remove(path);
        st.valLoss = 2.f; cs.onEpochEnd(m, st); bool skipped = !fs::exists(path);
        st.valLoss = 0.5f; cs.onEpochEnd(m, st); bool w2 = fs::exists(path);
        Model back; Preprocessor pp; loadModel(path, back, pp); fs::remove(path);
        report(c, "CheckpointSaver: saves only on improvement, loads back", w1 && skipped && w2 && back.snapshot() == m.snapshot()); }

    {   // command registry
        CommandRegistry reg; std::string seen;
        reg.add(std::make_unique<FnCommand>("echo", "e", [&](const Args& a) { seen = a.count("x") ? a.at("x") : "?"; return 7; }));
        reg.add(std::make_unique<FnCommand>("plain", "p", [&](const Args&) { seen = "plain"; return 3; }, false));
        auto call = [&](std::vector<std::string> v) { std::vector<char*> av; for (auto& x : v) av.push_back(&x[0]); return reg.dispatch((int)av.size(), av.data()); };
        int r1 = call({"minnn", "echo", "--x", "hello", "--flag"}); std::string s1 = seen;
        int r2 = call({"minnn", "plain", "stray-token"});                       // takesOptions()==false: tokens are not parsed
        int r3 = call({"minnn", "nope"}), r4 = call({"minnn"});
        bool badThrows = false; try { call({"minnn", "echo", "oops"}); } catch (const std::runtime_error&) { badThrows = true; }
        bool dupThrows = false; try { reg.add(std::make_unique<FnCommand>("echo", "again", [](const Args&) { return 0; })); } catch (const std::runtime_error&) { dupThrows = true; }
        std::vector<std::string> av2 = {"minnn", "t", "--a", "--b", "v"}; std::vector<char*> pv; for (auto& x : av2) pv.push_back(&x[0]);
        Args pa = parseArgs((int)pv.size(), pv.data(), 2);
        bool listed = reg.listing().find("minnn echo") != std::string::npos && reg.listing().find("minnn plain") != std::string::npos;
        report(c, "CommandRegistry: dispatch, option parsing, errors", r1 == 7 && s1 == "hello" && r2 == 3 && r3 == -1 && r4 == -1 && badThrows && dupThrows && pa["a"] == "1" && pa["b"] == "v" && listed); }

    {   // ImageSource strategy classes used through the base interface
        const fs::path root = fs::temp_directory_path() / "minnn_selftest_src"; fs::remove_all(root);
        for (const char* cls : {"a", "b"}) fs::create_directories(root / cls);
        for (int k = 0; k < 4; ++k) for (const char* cls : {"a", "b"}) {
            std::vector<unsigned char> px(100); for (auto& v : px) v = (unsigned char)(g() % 256);
            std::ofstream o(root / cls / ("i" + std::to_string(k) + ".pgm"), std::ios::binary); o << "P5\n10 10\n255\n"; o.write((const char*)px.data(), px.size());
        }
        FolderImageSource fsrc(root.string(), 1, 6, 6, FitMode::Stretch); ImageSource& base = fsrc;
        Preprocessor pa, pb; Dataset d1 = base.load(pa); Dataset d2 = loadImageFolder(root.string(), 1, 6, 6, pb, FitMode::Stretch);
        bool same = d1.X.data == d2.X.data && d1.cls == d2.cls && pa.classes == pb.classes && pa.classes.size() == 2 && d1.size() == 8;
        bool meta = base.channels() == 1 && base.height() == 6 && base.width() == 6 && base.describe().find(root.string()) != std::string::npos;
        fs::remove_all(root);
        const std::string ip = (fs::temp_directory_path() / "minnn_t_img.idx").string(), lp = (fs::temp_directory_path() / "minnn_t_lbl.idx").string();
        { std::ofstream fi(ip, std::ios::binary), fl(lp, std::ios::binary); writeBE32(fi, 0x803); writeBE32(fi, 5); writeBE32(fi, 4); writeBE32(fi, 3);
          for (int i = 0; i < 5 * 12; ++i) fi.put((char)(i == 0 ? 255 : i % 100));
          writeBE32(fl, 0x801); writeBE32(fl, 5); for (int l : {0, 2, 1, 2, 0}) fl.put((char)l); }
        IdxImageSource isrc(ip, lp, 3); ImageSource& ib = isrc; Preprocessor pi; Dataset di = ib.load(pi);
        bool idxOk = di.size() == 3 && ib.height() == 4 && ib.width() == 3 && ib.channels() == 1 && di.X.cols == 12 && std::fabs(di.X(0, 0) - 1.f) < 1e-6f &&
                     di.cls == std::vector<int>({0, 2, 1}) && pi.classes.size() == 3;
        fs::remove(ip); fs::remove(lp);
        report(c, "ImageSource: Folder == loader function, IDX size/limit/labels", same && meta && idxOk); }
}

// ---- 2f. text files as written on Windows (BOM + CRLF) and helpful "cannot open" errors ----
inline void testDataFiles(Ctx& c) {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "minnn_selftest_txt"; fs::remove_all(root); fs::create_directories(root / "cls_a");
    const std::string raw = "\xEF\xBB\xBF" "سلام دنیا\r\nخط دوم\r\n", want = "سلام دنیا\nخط دوم\n";
    auto put = [&](const fs::path& p) { std::ofstream o(p, std::ios::binary); o << raw; };
    put(root / "a.txt"); put(root / "cls_a" / "d1.txt");

    auto one = collectTexts((root / "a.txt").string());
    auto all = collectTexts(root.string());
    TextData td = loadTextFolder(root.string());
    std::string lone = "a\rb\r\nc"; normalizeText(lone);
    report(c, "text files: BOM and CRLF removed (single file, folder, class folders)",
           one.size() == 1 && one[0] == want && all.size() == 2 && all[0] == want && all[1] == want &&
           td.texts.size() == 1 && td.texts[0] == want && td.labels[0] == "cls_a" && lone == "a\nb\nc");

    auto msgOf = [](auto fn) { try { fn(); } catch (const std::exception& e) { return std::string(e.what()); } return std::string(); };
    const std::string miss = (root / "nope.txt").string();
    std::string m1 = msgOf([&] { collectTexts(miss); }), m2 = msgOf([&] { readCSVRecords(miss); }), m3 = msgOf([&] { readCSV(miss); });
    auto good = [&](const std::string& m) { return m.find(miss) != std::string::npos && m.find("current working directory") != std::string::npos; };
    report(c, "missing data file: error names the path and the working directory", good(m1) && good(m2) && good(m3));
    fs::remove_all(root);
}

// ---- 2g. CSV columns by header name, and a CSV as the source of raw training text ----
inline void testCsvColumns(Ctx& c) {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "minnn_selftest_csv"; fs::remove_all(dir); fs::create_directories(dir);
    const std::string csv = (dir / "reviews.csv").string();
    { std::ofstream o(csv, std::ios::binary);
      o << "\xEF\xBB\xBF" << "id,\"Comment\",Sentiment\r\n1,\"hello, world\",pos\r\n2,\"multi\nline \"\"quoted\"\"\",neg\r\n3,,neg\r\n"; }

    bool names = columnIndex(csv, "comment", -9) == 1 && columnIndex(csv, " SENTIMENT ", -9) == 2 && columnIndex(csv, "id", -9) == 0;
    bool numbers = columnIndex(csv, "2", -9) == 2 && columnIndex(csv, "-1", -9) == -1 && columnIndex(csv, "", 7) == 7;
    std::string err; try { columnIndex(csv, "missing", 0); } catch (const std::exception& e) { err = e.what(); }
    bool helpful = err.find("'missing'") != std::string::npos && err.find("Comment | Sentiment") != std::string::npos;
    auto hdr = readCSVHeader(csv);
    bool header = hdr.size() == 3 && hdr[1] == "Comment" && hdr[0] == "id";

    TextData td = loadTextCSV(csv, columnIndex(csv, "comment", 0), columnIndex(csv, "sentiment", -1), true);
    bool rows = td.texts.size() == 3 && td.texts[0] == "hello, world" && td.texts[1] == "multi\nline \"quoted\"" && td.labels[1] == "neg";

    auto docs = collectTexts(csv, "comment", true);       // a CSV as raw text: one document, one row per line, empty cells skipped
    bool asText = docs.size() == 1 && docs[0] == "hello, world\nmulti\nline \"quoted\"\n";
    report(c, "CSV columns by name or number; --text accepts a CSV",
           names && numbers && helpful && header && rows && asText);
    fs::remove_all(dir);
}

// ---- 2g. data problems come with an explanation (empty labels, missing numbers) ----
inline void testDataProblems(Ctx& c) {
    namespace fs = std::filesystem;
    bool emptyLabels = countEmptyLabels({"pos", "", "neg", "  ", "pos"}) == 2 && countEmptyLabels({"a", "b"}) == 0;
    std::string miss = badNumberMsg(7, 2, ""), spaces = badNumberMsg(7, 2, "  "), text = badNumberMsg(7, 2, "abc");
    bool msgs = miss.find("row 7 col 2") != std::string::npos && miss.find("missing") != std::string::npos &&
                spaces.find("missing") != std::string::npos && text.find("non-numeric feature 'abc'") != std::string::npos &&
                text.find("missing") == std::string::npos;
    report(c, "data problems: empty labels counted, missing vs non-numeric cells explained", emptyLabels && msgs);

    {   // mapClasses must actually WARN about empty labels (and stay silent for clean labels)
        std::ostringstream captured; std::ostream* saved = warnStream(); warnStream() = &captured;
        Preprocessor pr1, pr2;
        auto cls = mapClasses({"pos", "", "neg", "pos"}, pr1); std::string withEmpty = captured.str();
        captured.str(""); mapClasses({"pos", "neg", "pos"}, pr2); std::string clean = captured.str();
        warnStream() = saved;
        report(c, "mapClasses: warns about empty labels, silent for clean labels",
               withEmpty.find("1 of 4") != std::string::npos && withEmpty.find("EMPTY label") != std::string::npos &&
               withEmpty.find("IS NOT NULL") != std::string::npos && clean.empty() && cls.size() == 4 && pr1.classes.size() == 3);
    }

    const fs::path p = fs::temp_directory_path() / "minnn_selftest_missing.csv";
    { std::ofstream o(p, std::ios::binary); o << "a,b,label\n1,2,x\n3,,y\n5,6,x\n"; }       // second row has an empty feature cell
    std::string what; Preprocessor prep;
    try { loadLabeled(p.string(), -1, Task::Classification, prep); } catch (const std::exception& e) { what = e.what(); }
    fs::remove(p);
    report(c, "tabular CSV with an empty numeric cell: error names row/col and says it is missing",
           what.find("row 2 col 1") != std::string::npos && what.find("missing") != std::string::npos, what.substr(0, 60));
}

// ---- 2h. CSV separators: comma, semicolon (Excel in many regions) and TAB ----
inline void testCsvDelimiters(Ctx& c) {
    namespace fs = std::filesystem;
    const std::string bom = "\xEF\xBB\xBF";
    bool det = detectDelimiter("a,b,c") == ',' && detectDelimiter("a;b;c") == ';' && detectDelimiter("a\tb\tc") == '\t' &&
               detectDelimiter(bom + "text;label\n1;2\n") == ';' && detectDelimiter("\"x;y;z\",b\n") == ',' &&
               detectDelimiter("\n\n  \na;b\n") == ';' && detectDelimiter("a,b;c") == ',' && detectDelimiter("nothing here") == ',' &&
               detectDelimiter("") == ',';
    report(c, "CSV separator detection (comma / semicolon / TAB / quoted / BOM / blank lines)", det);

    auto rows = parseCSVText(bom + "text;label\r\n\"a;b\";pos\r\nc;neg\r\n");
    report(c, "CSV parser: semicolon file, delimiter inside quotes kept in the field",
           rows.size() == 3 && rows[0] == std::vector<std::string>({"text", "label"}) && rows[1][0] == "a;b" && rows[1][1] == "pos" && rows[2][1] == "neg");

    const fs::path root = fs::temp_directory_path() / "minnn_selftest_delim"; fs::create_directories(root);
    auto put = [&](const char* name, const std::string& body) { fs::path p = root / name; std::ofstream o(p, std::ios::binary); o << body; return p.string(); };
    Preprocessor p1, p2, p3;
    Dataset d1 = loadLabeled(put("semi.csv", bom + "a;b;label\r\n1;2;x\r\n3;4;y\r\n5;6;x\r\n"), -1, Task::Classification, p1);
    Dataset d2 = loadLabeled(put("tab.tsv", "a\tb\tlabel\n1\t2\tx\n3\t4\ty\n"), -1, Task::Classification, p2);
    Dataset d3 = loadLabeled(put("nohdr.csv", bom + "1;2;x\n3;4;y\n"), -1, Task::Classification, p3);      // BOM + numeric first row, no header
    report(c, "tabular loader: semicolon+BOM, TAB, and headerless semicolon files",
           d1.size() == 3 && d1.X.cols == 2 && d1.X(1, 1) == 4.f && p1.classes.size() == 2 &&
           d2.size() == 2 && d2.X.cols == 2 && d3.size() == 2 && d3.X(0, 0) == 1.f);

    auto err = [&](auto fn) { try { fn(); } catch (const std::exception& e) { return std::string(e.what()); } return std::string(); };
    std::string e1 = err([&] { Preprocessor q; loadLabeled(put("one.csv", "a|b\n1|2\n"), -1, Task::Classification, q); });
    std::string e2 = err([&] { loadTextCSV(put("pipe.csv", "text|label\nhello|pos\n"), 0, -1, true); });
    TextData td = loadTextCSV(put("semi_text.csv", "text;label\n\"خوب; عالی\";pos\n"), 0, -1, true);
    report(c, "unsplittable files are refused with a clear message, not trained on",
           e1.find("only 1 column") != std::string::npos && e2.find("same column") != std::string::npos &&
           td.texts.size() == 1 && td.texts[0] == "خوب; عالی" && td.labels[0] == "pos");

    std::string h1 = badNumberMsg(1, 0, "0,5"), h2 = badNumberMsg(1, 0, "abc"), h3 = badNumberMsg(1, 0, "1,2,3");
    report(c, "decimal comma gets a hint; other bad cells do not",
           h1.find("decimal comma") != std::string::npos && h2.find("decimal comma") == std::string::npos && h3.find("decimal comma") == std::string::npos);
    fs::remove_all(root);
}

inline GPT tinyGPT(unsigned seed = 5, int layers = 2, int ctx = 16) {
    GPT m; GPTConfig c; c.vocab = 30; c.ctx = ctx; c.dim = 16; c.layers = layers; c.heads = 2; c.dropout = 0.f; m.init(c, seed); return m;
}

// ---- 3. tokenizer ----
inline void testBPE(Ctx& c) {
    BPE b; std::unordered_map<std::string, int64_t> counts;
    std::string corpus = "سلام دنیا! لینوکس یک سیستم عامل است. hello world, this is linux kernel code in C++ 123";
    for (int i = 0; i < 20; ++i) BPE::countChunks(corpus, counts);
    b.train(counts, 60);
    bool rt = true;
    for (std::string t : {corpus, std::string("سلام، دنیا! ي ك"), std::string("a  b\n\nc\t"), std::string(""), std::string("🙂 emoji ✓")})
        rt = rt && (b.decode(b.encode(t)) == t);
    std::string bin; for (int i = 0; i < 256; ++i) bin += (char)i;
    rt = rt && (b.decode(b.encode(bin)) == bin);
    report(c, "BPE encode/decode round-trip (any bytes)", rt);
    auto sp = b.encode("<|user|>hi<|bot|>ok<|eot|>");
    report(c, "BPE special tokens", sp.front() == BPE::USER && sp.back() == BPE::EOT &&
           std::count(sp.begin(), sp.end(), (int)BPE::BOT) == 1);
    auto ids = b.encode(corpus);
    report(c, "BPE compresses text", ids.size() < corpus.size() * 0.7, fmt("%.0f", (double)corpus.size()) + " bytes -> " + fmt("%.0f", (double)ids.size()) + " tokens");
    std::stringstream ss; b.save(ss); BPE b2; b2.load(ss);
    report(c, "BPE save/load identical", b2.encode(corpus) == ids);
}

// ---- 4. KV-cache generator == training forward ----
inline void testKVCache(Ctx& c) {
    GPT m = tinyGPT(); std::mt19937 g(4); const int T = 11;
    std::vector<int> tokens(T + 1); for (auto& t : tokens) t = g() % 30;
    std::vector<float> mk(T, 1.f);
    float full = m.step(tokens.data(), tokens.data() + 1, mk.data(), 1, T, false, false);
    Generator gen(m); double sum = 0;
    for (int t = 0; t < T; ++t) {
        auto lg = gen.step(tokens[t]); float mx = *std::max_element(lg.begin(), lg.end()); double z = 0;
        for (float v : lg) z += std::exp(v - mx);
        sum += -(lg[tokens[t + 1]] - mx - std::log(z));
    }
    double diff = std::fabs(full - sum / T);
    report(c, "KV-cache inference == training forward", diff < 1e-4, "loss diff " + fmt("%.1e", diff));
}

// ---- 5. save / load round-trips ----
inline void testSaveLoad(Ctx& c) {
    namespace fs = std::filesystem; std::string dir = fs::temp_directory_path().string();
    std::mt19937 g(6); bool ok = true;
    {   ModelConfig cf; cf.kind = Kind::CNN; cf.imgC = 1; cf.imgH = 8; cf.imgW = 8; cf.inputDim = 64; cf.outputDim = 3; cf.conv = {4}; cf.hidden = {6};
        Model a(cf); Preprocessor p; p.classes = {"a", "b", "c"}; p.normMode = 2; p.mean = {0.3f}; p.stdv = {1.5f};
        saveModel(dir + "/t_cnn.bin", a, p); Model b; Preprocessor q; loadModel(dir + "/t_cnn.bin", b, q);
        Matrix X = randMat(2, 64, g); Matrix o1 = a.forward(X, false), o2 = b.forward(X, false);
        for (size_t i = 0; i < o1.size(); ++i) ok = ok && o1.data[i] == o2.data[i];
        ok = ok && q.classes == p.classes && q.normMode == 2 && q.mean[0] == 0.3f; }
    {   ModelConfig cf; cf.kind = Kind::Text; cf.vocab = 5; cf.embed = 4; cf.maxLen = 3; cf.inputDim = 3; cf.outputDim = 2; cf.hidden = {3};
        Model a(cf); Preprocessor p; p.vocab = {"x", "y", "z", "w", "v"}; saveModel(dir + "/t_txt.bin", a, p);
        Model b; Preprocessor q; loadModel(dir + "/t_txt.bin", b, q); Matrix X(1, 3); X.data = {1, 2, 0};
        ok = ok && q.vocab == p.vocab && a.forward(X, false).data == b.forward(X, false).data; }
    report(c, "save/load: CNN + Text classifiers", ok);
    GPT m = tinyGPT(); m.bpe.rebuild(); saveLM(dir + "/t_lm.bin", m); GPT n; loadLM(dir + "/t_lm.bin", n);
    std::vector<int> in(8), tg(8); std::vector<float> mk(8, 1.f); for (int i = 0; i < 8; ++i) { in[i] = g() % 30; tg[i] = g() % 30; }
    float l1 = m.step(in.data(), tg.data(), mk.data(), 1, 8, false, false), l2 = n.step(in.data(), tg.data(), mk.data(), 1, 8, false, false);
    report(c, "save/load: language model", l1 == l2 && n.paramCount() == m.paramCount());
    for (auto f : {"t_cnn.bin", "t_txt.bin", "t_lm.bin"}) fs::remove(dir + "/" + f);
}

// ---- 6. model growth is function-preserving ----
inline void testGrow(Ctx& c) {
    GPT m = tinyGPT(5, 2, 16); std::mt19937 g(8); std::vector<int> in(8), tg(8); std::vector<float> mk(8, 1.f);
    for (int i = 0; i < 8; ++i) { in[i] = g() % 30; tg[i] = g() % 30; }
    float before = m.step(in.data(), tg.data(), mk.data(), 1, 8, false, false);
    growLM(m, 5, 32, 9);
    float after = m.step(in.data(), tg.data(), mk.data(), 1, 8, false, false);
    report(c, "grow: +3 layers, 2x context keeps outputs", std::fabs(before - after) < 1e-5 && m.cfg.layers == 5 && m.cfg.ctx == 32,
           "loss " + fmt("%.5f", before) + " -> " + fmt("%.5f", after));
}

// ---- 7. optimizer state resume ----
inline void testAdamResume(Ctx& c) {
    Matrix w1(1, 4, 1.f), g1(1, 4), w2(1, 4, 1.f), g2(1, 4); std::vector<Param> p1 = {{&w1, &g1, true}}, p2 = {{&w2, &g2, true}};
    Adam a(0.1f, 0.01f), b(0.1f, 0.01f);
    for (int s = 0; s < 3; ++s) { for (int i = 0; i < 4; ++i) g1.data[i] = 0.5f * (i + 1) * (s + 1); a.step(p1); }
    std::string path = std::filesystem::temp_directory_path().string() + "/t_adam.opt";
    a.saveState(path); w2 = w1; bool loaded = b.loadState(path, p2);
    for (int i = 0; i < 4; ++i) { g1.data[i] = g2.data[i] = 0.3f * i - 0.2f; }
    a.step(p1); b.step(p2);
    bool same = loaded; for (int i = 0; i < 4; ++i) same = same && w1.data[i] == w2.data[i];
    std::filesystem::remove(path);
    report(c, "Adam state save/load -> identical updates", same);
}

// ---- 8. CSV + tokenizer for Persian ----
inline void testParsing(Ctx& c) {
    std::string path = std::filesystem::temp_directory_path().string() + "/t_csv.csv";
    { std::ofstream f(path, std::ios::binary); f << "\xEF\xBB\xBF" "text,label\r\n\"line one\nline two, with comma\",pos\r\n\"say \"\"hi\"\"\",neg\r\n"; }
    auto rows = readCSVRecords(path); std::filesystem::remove(path);
    report(c, "CSV: quotes, commas, newlines, BOM, CRLF",
           rows.size() == 3 && rows[1][0] == "line one\nline two, with comma" && rows[2][0] == "say \"hi\"" && rows[0][0] == "text");
    auto t = tokenize("سلام، دنیا! \xD9\x8A \xD9\x83 Hello,World");
    std::vector<std::string> want = {"سلام", "دنیا", "\xDB\x8C", "\xDA\xA9", "hello", "world"};
    report(c, "tokenizer: Persian punctuation + ي/ك normalization", t == want);
    std::vector<float> lg(300, 0.f); lg[43] = 5.f; SampleOpts so; so.temp = 0.f; so.rep = 1.5f; BPE b; b.initPieces(); b.pieces.resize(300);
    std::mt19937 g(1);
    report(c, "sampling: no penalty on single bytes ('+'), bans <|user|>", sampleToken(lg, so, {43}, g, b) == 43);
}

// ---- 9. language model actually learns ----
inline void testLMLearns(Ctx& c) {
    GPT m; GPTConfig cf; cf.vocab = 12; cf.ctx = 8; cf.dim = 24; cf.layers = 2; cf.heads = 2; cf.dropout = 0.f; m.init(cf, 3);
    Adam opt(5e-3f, 0.f, 0.9f, 0.95f); auto ps = m.params(); const int B = 4, T = 8;
    std::vector<int> in(B * T), tg(B * T); std::vector<float> mk(B * T, 1.f); float first = 0, last = 0;
    for (int it = 0; it < 150; ++it) {
        for (int b = 0; b < B; ++b) for (int t = 0; t < T; ++t) { in[b * T + t] = (b * 3 + t) % 12; tg[b * T + t] = (b * 3 + t + 1) % 12; }
        float l = m.step(in.data(), tg.data(), mk.data(), B, T, true, true); opt.step(ps);
        if (it == 0) { first = l; }
        last = l;
    }
    report(c, "GPT overfits a cyclic sequence (loss falls)", last < 0.3f && last < first * 0.2f, fmt("%.2f", first) + " -> " + fmt("%.3f", last));
}

} // namespace selftest

namespace selftest {
// Runs one test function; an exception becomes a reported FAIL instead of aborting the whole self-test.
inline void guarded(Ctx& c, const char* name, void (*fn)(Ctx&)) {
    try { fn(c); }
    catch (const std::exception& e) { report(c, name, false, std::string("threw: ") + e.what()); }
}
}

inline int runSelfTest() {
    selftest::Ctx c;
    std::printf("minnn self-test\n");
    selftest::guarded(c, "testMatmul", selftest::testMatmul);
    selftest::guarded(c, "testLayerGrads", selftest::testLayerGrads);
    selftest::guarded(c, "testImageProc", selftest::testImageProc);
    selftest::guarded(c, "testTransforms", selftest::testTransforms);
    selftest::guarded(c, "testBatchNormResNet", selftest::testBatchNormResNet);
    selftest::guarded(c, "testModelFiles", selftest::testModelFiles);
    selftest::guarded(c, "testImageEndToEnd", selftest::testImageEndToEnd);
    selftest::guarded(c, "testOOP", selftest::testOOP);
    selftest::guarded(c, "testDataFiles", selftest::testDataFiles);
    selftest::guarded(c, "testDataProblems", selftest::testDataProblems);
    selftest::guarded(c, "testCsvDelimiters", selftest::testCsvDelimiters);
    selftest::guarded(c, "testCsvColumns", selftest::testCsvColumns);
    { int rc = cmdGradCheck(); selftest::report(c, "backprop: full GPT (attention, LayerNorm, GELU)", rc == 0); }
    selftest::guarded(c, "testBPE", selftest::testBPE);
    selftest::guarded(c, "testKVCache", selftest::testKVCache);
    selftest::guarded(c, "testSaveLoad", selftest::testSaveLoad);
    selftest::guarded(c, "testGrow", selftest::testGrow);
    selftest::guarded(c, "testAdamResume", selftest::testAdamResume);
    selftest::guarded(c, "testParsing", selftest::testParsing);
    selftest::guarded(c, "testLMLearns", selftest::testLMLearns);
    std::printf("\n%d/%d checks passed%s\n", c.total - c.failed, c.total, c.failed ? "  <-- FAILURES" : "");
    return c.failed ? 1 : 0;
}

} // namespace nn
