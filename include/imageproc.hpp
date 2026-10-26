#pragma once
// Object-oriented image processing:
//   Image            - value class: CHW float image in [0,1] with resize / crop / pad / flip.
//   ImageTransform   - abstract base for augmentations (RandomAffine, ColorJitter, ...), combined by Compose.
//   ImageAugmenter   - applies a Compose pipeline to batch rows (de-normalize -> augment -> re-normalize).
#include "dataset.hpp"
#include "third_party/stb_image.h"
#include <memory>
#include <random>
#include <cmath>
#include <string>

namespace nn {

enum class FitMode { Stretch = 0, Crop = 1, Pad = 2 };
inline FitMode parseFit(const std::string& s) {
    if (s == "stretch") return FitMode::Stretch;
    if (s == "crop") return FitMode::Crop;
    if (s == "pad") return FitMode::Pad;
    throw std::runtime_error("--fit must be stretch | crop | pad");
}

namespace detail {
// 1-D resampling weights: area (box) averaging when shrinking (anti-aliased), linear interpolation otherwise.
struct AxisWeights { std::vector<int> start, count; std::vector<size_t> off; std::vector<float> w; };
inline AxisWeights makeAxis(int n, int m) {
    AxisWeights a; a.start.resize(m); a.count.resize(m); a.off.resize(m);
    const double s = (double)n / m;
    for (int j = 0; j < m; ++j) {
        a.off[j] = a.w.size();
        if (s > 1.0) {
            const double lo = j * s, hi = (j + 1) * s;
            const int i0 = (int)std::floor(lo), i1 = std::min(n, (int)std::ceil(hi));
            double tot = 0;
            for (int i = i0; i < i1; ++i) tot += std::min(hi, i + 1.0) - std::max(lo, (double)i);
            for (int i = i0; i < i1; ++i) a.w.push_back((float)((std::min(hi, i + 1.0) - std::max(lo, (double)i)) / tot));
            a.start[j] = i0; a.count[j] = i1 - i0;
        } else {
            const double c = (j + 0.5) * s - 0.5;
            int i0 = (int)std::floor(c); const double f = c - i0;
            if (i0 < 0) { a.start[j] = 0; a.count[j] = 1; a.w.push_back(1.f); }
            else if (i0 >= n - 1) { a.start[j] = n - 1; a.count[j] = 1; a.w.push_back(1.f); }
            else { a.start[j] = i0; a.count[j] = 2; a.w.push_back((float)(1.0 - f)); a.w.push_back((float)f); }
        }
    }
    return a;
}
} // namespace detail

class Image {
public:
    int C = 0, H = 0, W = 0;
    std::vector<float> px;

    Image() = default;
    Image(int c, int h, int w, float v = 0.f) : C(c), H(h), W(w), px((size_t)c * h * w, v) {}

    float& at(int c, int y, int x) { return px[((size_t)c * H + y) * W + x]; }
    float at(int c, int y, int x) const { return px[((size_t)c * H + y) * W + x]; }
    size_t size() const { return px.size(); }
    bool empty() const { return px.empty(); }

    // Bilinear sample at fractional coordinates; pixels outside the image read as `fill`.
    float sample(int c, float y, float x, float fill) const {
        const int y0 = (int)std::floor(y), x0 = (int)std::floor(x);
        const float fy = y - y0, fx = x - x0;
        auto get = [&](int yy, int xx) { return (yy < 0 || yy >= H || xx < 0 || xx >= W) ? fill : at(c, yy, xx); };
        return (1 - fy) * ((1 - fx) * get(y0, x0) + fx * get(y0, x0 + 1)) + fy * ((1 - fx) * get(y0 + 1, x0) + fx * get(y0 + 1, x0 + 1));
    }
    std::vector<float> channelMeans() const {
        std::vector<float> m(C, 0.f); const size_t per = (size_t)H * W;
        for (int c = 0; c < C; ++c) { double s = 0; for (size_t i = 0; i < per; ++i) s += px[c * per + i]; m[c] = per ? (float)(s / per) : 0.f; }
        return m;
    }

    static bool load(const std::string& path, int channels, Image& out) {
        int w, h, n;
        unsigned char* d = stbi_load(path.c_str(), &w, &h, &n, channels);
        if (!d) return false;
        out = Image(channels, h, w);
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) for (int c = 0; c < channels; ++c)
            out.at(c, y, x) = d[((size_t)y * w + x) * channels + c] / 255.f;
        stbi_image_free(d);
        return true;
    }
    static Image fromRow(const float* r, int C, int H, int W) { Image im(C, H, W); std::copy(r, r + im.size(), im.px.begin()); return im; }
    void toRow(float* r) const { std::copy(px.begin(), px.end(), r); }

    Image resized(int nh, int nw) const {
        if (nh == H && nw == W) return *this;
        const auto ax = detail::makeAxis(W, nw), ay = detail::makeAxis(H, nh);
        Image tmp(C, H, nw), out(C, nh, nw);
        for (int c = 0; c < C; ++c)
            for (int y = 0; y < H; ++y)
                for (int j = 0; j < nw; ++j) {
                    float s = 0.f; for (int k = 0; k < ax.count[j]; ++k) s += ax.w[ax.off[j] + k] * at(c, y, ax.start[j] + k);
                    tmp.at(c, y, j) = s;
                }
        for (int c = 0; c < C; ++c)
            for (int i = 0; i < nh; ++i)
                for (int x = 0; x < nw; ++x) {
                    float s = 0.f; for (int k = 0; k < ay.count[i]; ++k) s += ay.w[ay.off[i] + k] * tmp.at(c, ay.start[i] + k, x);
                    out.at(c, i, x) = s;
                }
        return out;
    }
    Image cropped(int y0, int x0, int h, int w) const {
        Image out(C, h, w);
        for (int c = 0; c < C; ++c) for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) out.at(c, y, x) = at(c, y0 + y, x0 + x);
        return out;
    }
    Image flippedH() const {
        Image out(C, H, W);
        for (int c = 0; c < C; ++c) for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) out.at(c, y, x) = at(c, y, W - 1 - x);
        return out;
    }
    // Brings any image to exactly nh x nw. Stretch distorts the aspect ratio; Crop keeps the centered region
    // with the target aspect; Pad keeps everything and fills the border with the image's channel means.
    Image fitted(int nh, int nw, FitMode mode) const {
        if (mode == FitMode::Stretch) return resized(nh, nw);
        if (mode == FitMode::Crop) {
            const double a = (double)nw / nh;
            if ((double)W / H > a) { int cw = std::max(1, (int)std::lround(H * a)); return cropped(0, (W - cw) / 2, H, cw).resized(nh, nw); }
            int chh = std::max(1, (int)std::lround(W / a)); return cropped((H - chh) / 2, 0, chh, W).resized(nh, nw);
        }
        const double sc = std::min((double)nw / W, (double)nh / H);
        const int h2 = std::max(1, std::min(nh, (int)std::lround(H * sc))), w2 = std::max(1, std::min(nw, (int)std::lround(W * sc)));
        Image small = resized(h2, w2), out(C, nh, nw);
        const auto m = channelMeans();
        for (int c = 0; c < C; ++c) std::fill(out.px.begin() + (size_t)c * nh * nw, out.px.begin() + (size_t)(c + 1) * nh * nw, m[c]);
        const int oy = (nh - h2) / 2, ox = (nw - w2) / 2;
        for (int c = 0; c < C; ++c) for (int y = 0; y < h2; ++y) for (int x = 0; x < w2; ++x) out.at(c, oy + y, ox + x) = small.at(c, y, x);
        return out;
    }
};

// ------------------------------- augmentations -------------------------------
class ImageTransform {
public:
    virtual ~ImageTransform() = default;
    virtual void apply(Image& img, std::mt19937& rng) const = 0;
    virtual std::string describe() const = 0;
protected:
    static float uni(std::mt19937& r, float a, float b) { return a == b ? a : std::uniform_real_distribution<float>(a, b)(r); }
};

class RandomHFlip : public ImageTransform {
    float p_;
public:
    explicit RandomHFlip(float p = 0.5f) : p_(p) {}
    void apply(Image& img, std::mt19937& r) const override { if (uni(r, 0.f, 1.f) < p_) img = img.flippedH(); }
    std::string describe() const override { return "hflip p=" + std::to_string(p_).substr(0, 4); }
};
class RandomVFlip : public ImageTransform {
    float p_;
public:
    explicit RandomVFlip(float p = 0.5f) : p_(p) {}
    void apply(Image& img, std::mt19937& r) const override {
        if (uni(r, 0.f, 1.f) >= p_) return;
        Image out(img.C, img.H, img.W);
        for (int c = 0; c < img.C; ++c) for (int y = 0; y < img.H; ++y) for (int x = 0; x < img.W; ++x) out.at(c, y, x) = img.at(c, img.H - 1 - y, x);
        img = std::move(out);
    }
    std::string describe() const override { return "vflip p=" + std::to_string(p_).substr(0, 4); }
};

// Rotation + zoom + translation in ONE bilinear resampling pass (no repeated blurring).
// Pixels that come from outside the image are filled with the channel means.
class RandomAffine : public ImageTransform {
    float rot_, zmin_, zmax_, shift_;
public:
    RandomAffine(float maxRotDeg, float zoomMin, float zoomMax, float maxShiftFrac)
        : rot_(maxRotDeg), zmin_(zoomMin), zmax_(zoomMax), shift_(maxShiftFrac) {}
    void apply(Image& img, std::mt19937& r) const override {
        const float th = uni(r, -rot_, rot_) * 3.14159265f / 180.f, z = uni(r, zmin_, zmax_);
        const float tx = uni(r, -shift_, shift_) * img.W, ty = uni(r, -shift_, shift_) * img.H;
        if (th == 0.f && z == 1.f && tx == 0.f && ty == 0.f) return;
        const Image src = img; const auto fill = src.channelMeans();
        const float cs = std::cos(th), sn = std::sin(th), cx = (img.W - 1) / 2.f, cy = (img.H - 1) / 2.f;
        for (int y = 0; y < img.H; ++y)
            for (int x = 0; x < img.W; ++x) {
                const float dx = x - cx - tx, dy = y - cy - ty;
                const float sx = (cs * dx + sn * dy) / z + cx, sy = (-sn * dx + cs * dy) / z + cy;
                for (int c = 0; c < img.C; ++c) img.at(c, y, x) = src.sample(c, sy, sx, fill[c]);
            }
    }
    std::string describe() const override {
        char b[96]; std::snprintf(b, sizeof b, "affine rot<=%.0fdeg zoom %.2f..%.2f shift<=%.0f%%", rot_, zmin_, zmax_, shift_ * 100.f); return b;
    }
};

class ColorJitter : public ImageTransform {
    float b_, c_;
public:
    ColorJitter(float brightness, float contrast) : b_(brightness), c_(contrast) {}
    void apply(Image& img, std::mt19937& r) const override {
        const float bf = uni(r, -b_, b_), cf = 1.f + uni(r, -c_, c_);
        double m = 0; for (float v : img.px) m += v; m /= std::max<size_t>(1, img.size());
        for (auto& v : img.px) v = std::min(1.f, std::max(0.f, ((v - (float)m) * cf + (float)m) + bf));
    }
    std::string describe() const override { char b[64]; std::snprintf(b, sizeof b, "jitter brightness +-%.2f contrast +-%.2f", b_, c_); return b; }
};

class GaussianNoise : public ImageTransform {
    float sigma_;
public:
    explicit GaussianNoise(float sigma) : sigma_(sigma) {}
    void apply(Image& img, std::mt19937& r) const override {
        std::normal_distribution<float> nd(0.f, sigma_);
        for (auto& v : img.px) v = std::min(1.f, std::max(0.f, v + nd(r)));
    }
    std::string describe() const override { char b[48]; std::snprintf(b, sizeof b, "noise sigma %.3f", sigma_); return b; }
};

// Cutout / random erasing: hides a random square (helps against over-reliance on a single image region).
class Cutout : public ImageTransform {
    float frac_, p_;
public:
    Cutout(float sizeFrac, float p = 0.5f) : frac_(sizeFrac), p_(p) {}
    void apply(Image& img, std::mt19937& r) const override {
        if (uni(r, 0.f, 1.f) >= p_) return;
        const int side = std::max(1, (int)std::lround(frac_ * std::min(img.H, img.W)));
        const int y0 = (int)uni(r, 0.f, (float)(img.H - side) + 0.999f), x0 = (int)uni(r, 0.f, (float)(img.W - side) + 0.999f);
        const auto m = img.channelMeans();
        for (int c = 0; c < img.C; ++c) for (int y = y0; y < std::min(img.H, y0 + side); ++y) for (int x = x0; x < std::min(img.W, x0 + side); ++x) img.at(c, y, x) = m[c];
    }
    std::string describe() const override { char b[48]; std::snprintf(b, sizeof b, "cutout %.0f%%", frac_ * 100.f); return b; }
};

class Compose {
    std::vector<std::unique_ptr<ImageTransform>> ts_;
public:
    Compose& add(std::unique_ptr<ImageTransform> t) { ts_.push_back(std::move(t)); return *this; }
    size_t size() const { return ts_.size(); }
    void apply(Image& img, std::mt19937& r) const { for (auto& t : ts_) t->apply(img, r); }
    std::string describe() const { std::string s; for (auto& t : ts_) s += (s.empty() ? "" : " + ") + t->describe(); return s; }
};

// Applies a pipeline to rows of a normalized batch: de-normalize -> augment in [0,1] pixel space -> re-normalize.
// Each row gets its own RNG seeded serially, so results are reproducible and the loop is parallel.
class ImageAugmenter {
    int C_, H_, W_; Compose pipe_;
public:
    ImageAugmenter(int C, int H, int W) : C_(C), H_(H), W_(W) {}
    Compose& pipeline() { return pipe_; }
    bool empty() const { return pipe_.size() == 0; }
    std::string describe() const { return pipe_.describe(); }
    void apply(Matrix& X, const Preprocessor& prep, std::mt19937& rng) const {
        if (empty()) return;
        std::vector<unsigned> seeds(X.rows); for (auto& s : seeds) s = (unsigned)rng();
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < (long long)X.rows; ++i) {
            float* row = &X.data[(size_t)i * X.cols];
            Image img = Image::fromRow(row, C_, H_, W_);
            prep.normalizeRow(img.px.data(), img.size(), true);
            std::mt19937 r(seeds[i]); pipe_.apply(img, r);
            for (auto& v : img.px) v = std::min(1.f, std::max(0.f, v));
            prep.normalizeRow(img.px.data(), img.size(), false);
            img.toRow(row);
        }
    }
};

// Builds the pipeline from CLI options. --augment is a sensible preset; every option can also be set/overridden.
inline std::shared_ptr<ImageAugmenter> buildAugmenter(const std::map<std::string, std::string>& a, int C, int H, int W) {
    auto has = [&](const char* k) { return a.count(k) > 0; };
    auto num = [&](const char* k, float d) { return has(k) ? std::stof(a.at(k)) : d; };
    const bool preset = has("augment");
    float rot = num("rotate", preset ? 10.f : 0.f), shift = num("shift", preset ? 0.1f : 0.f), zmin = 1.f, zmax = 1.f;
    if (preset) { zmin = 0.9f; zmax = 1.1f; }
    if (has("zoom")) { std::string z = a.at("zoom"); size_t c = z.find(','); zmin = std::stof(z.substr(0, c)); zmax = c == std::string::npos ? zmin : std::stof(z.substr(c + 1)); }
    const float jit = num("jitter", preset ? 0.15f : 0.f), noise = num("noise", 0.f), cut = num("cutout", 0.f);
    auto aug = std::make_shared<ImageAugmenter>(C, H, W);
    if (has("hflip")) aug->pipeline().add(std::make_unique<RandomHFlip>(0.5f));
    if (has("vflip")) aug->pipeline().add(std::make_unique<RandomVFlip>(0.5f));
    if (rot > 0.f || shift > 0.f || zmin != 1.f || zmax != 1.f) aug->pipeline().add(std::make_unique<RandomAffine>(rot, zmin, zmax, shift));
    if (jit > 0.f) aug->pipeline().add(std::make_unique<ColorJitter>(jit, jit));
    if (noise > 0.f) aug->pipeline().add(std::make_unique<GaussianNoise>(noise));
    if (cut > 0.f) aug->pipeline().add(std::make_unique<Cutout>(cut, 0.5f));
    return aug->empty() ? nullptr : aug;
}

} // namespace nn
