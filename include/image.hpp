#pragma once
// Image loading: class-per-folder datasets (png/jpg/bmp/tga/gif/pgm/ppm) and MNIST IDX files.
#include "dataset.hpp"
#include "imageproc.hpp"
#include <filesystem>
#include <cstdint>

namespace nn {

namespace fs = std::filesystem;

// Decode + fit to C x H x W floats in [0,1] (CHW layout). Returns false on failure.
inline bool loadImageCHW(const std::string& path, int C, int H, int W, std::vector<float>& out, FitMode fit = FitMode::Stretch) {
    Image img;
    if (!Image::load(path, C, img)) return false;
    out = img.fitted(H, W, fit).px;
    return true;
}

inline bool isImageFile(const fs::path& p) {
    std::string e = p.extension().string();
    for (auto& ch : e) ch = (char)std::tolower((unsigned char)ch);
    return e == ".png" || e == ".jpg" || e == ".jpeg" || e == ".bmp" || e == ".tga" ||
           e == ".gif" || e == ".pgm" || e == ".ppm";
}

// root/<class_name>/*.png ...
inline Dataset loadImageFolder(const std::string& root, int C, int H, int W, Preprocessor& prep, FitMode fit = FitMode::Stretch) {
    if (!fs::is_directory(root)) throw std::runtime_error(cannotOpenMsg("image folder", root));
    std::vector<std::string> classNames;
    for (auto& e : fs::directory_iterator(root)) if (e.is_directory()) classNames.push_back(e.path().filename().string());
    std::sort(classNames.begin(), classNames.end());
    if (classNames.size() < 2) throw std::runtime_error("need at least 2 class sub-folders in " + root);
    prep.classes = classNames;

    std::vector<std::string> files; std::vector<int> labels;
    for (size_t k = 0; k < classNames.size(); ++k) {
        std::vector<std::string> fl;
        for (auto& e : fs::directory_iterator(fs::path(root) / classNames[k]))
            if (e.is_regular_file() && isImageFile(e.path())) fl.push_back(e.path().string());
        std::sort(fl.begin(), fl.end());
        for (auto& f : fl) { files.push_back(f); labels.push_back((int)k); }
    }
    if (files.empty()) throw std::runtime_error("no images found");

    const size_t dim = (size_t)C * H * W;
    Matrix X(files.size(), dim);
    std::vector<char> ok(files.size(), 0);
    #pragma omp parallel for schedule(dynamic, 8)
    for (long long i = 0; i < (long long)files.size(); ++i) {
        std::vector<float> img;
        if (loadImageCHW(files[i], C, H, W, img, fit)) {
            std::copy(img.begin(), img.end(), &X.data[(size_t)i * dim]);
            ok[i] = 1;
        }
    }
    Dataset ds; ds.X = Matrix(0, dim);
    size_t good = 0, bad = 0;
    for (char c : ok) (c ? good : bad)++;
    if (bad) std::printf("warning: %zu unreadable image(s) skipped\n", bad);
    ds.X = Matrix(good, dim);
    size_t r = 0;
    for (size_t i = 0; i < files.size(); ++i) {
        if (!ok[i]) continue;
        std::copy(&X.data[i * dim], &X.data[i * dim] + dim, &ds.X.data[r * dim]);
        ds.cls.push_back(labels[i]); ++r;
    }
    return ds;
}

// ---- MNIST-style IDX files ----
inline uint32_t be32(std::istream& f) {
    unsigned char b[4]; f.read(reinterpret_cast<char*>(b), 4);
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | ((uint32_t)b[2] << 8) | b[3];
}
inline Dataset loadIDX(const std::string& imgPath, const std::string& lblPath, int& H, int& W,
                       Preprocessor& prep, size_t limit = 0) {
    std::ifstream fi(imgPath, std::ios::binary), fl(lblPath, std::ios::binary);
    if (!fi) throw std::runtime_error(cannotOpenMsg("IDX image file", imgPath));
    if (!fl) throw std::runtime_error(cannotOpenMsg("IDX label file", lblPath));
    if (be32(fi) != 0x00000803) throw std::runtime_error("bad IDX image magic");
    if (be32(fl) != 0x00000801) throw std::runtime_error("bad IDX label magic");
    size_t n = be32(fi); H = (int)be32(fi); W = (int)be32(fi);
    size_t nl = be32(fl);
    if (n != nl) throw std::runtime_error("image/label count mismatch");
    if (limit && limit < n) n = limit;
    const size_t dim = (size_t)H * W;
    Dataset ds; ds.X = Matrix(n, dim);
    std::vector<unsigned char> buf(dim);
    int maxLabel = 0;
    for (size_t i = 0; i < n; ++i) {
        fi.read(reinterpret_cast<char*>(buf.data()), dim);
        for (size_t j = 0; j < dim; ++j) ds.X.data[i * dim + j] = buf[j] / 255.f;
        unsigned char l; fl.read(reinterpret_cast<char*>(&l), 1);
        ds.cls.push_back(l); maxLabel = std::max(maxLabel, (int)l);
    }
    if (!fi || !fl) throw std::runtime_error("IDX file truncated");
    prep.classes.clear();
    for (int k = 0; k <= maxLabel; ++k) prep.classes.push_back(std::to_string(k));
    return ds;
}


// ---------------------------------------------------------------------------
// Where labelled images come from, as objects (strategy pattern). The training code only talks to ImageSource,
// so a new kind of source (e.g. a CSV of file paths) is one new subclass and no change elsewhere.
// ---------------------------------------------------------------------------
class ImageSource {
public:
    virtual ~ImageSource() = default;
    virtual Dataset load(Preprocessor& prep) = 0;     // reads the data and fills prep.classes
    virtual int channels() const = 0;
    virtual int height() const = 0;                   // for file formats that carry their own size (IDX) valid after load()
    virtual int width() const = 0;
    virtual std::string describe() const = 0;
};

// root/<class_name>/*.png ...  resized to C x H x W with the chosen FitMode.
class FolderImageSource : public ImageSource {
    std::string root_; int C_, H_, W_; FitMode fit_;
public:
    FolderImageSource(std::string root, int C, int H, int W, FitMode fit = FitMode::Stretch)
        : root_(std::move(root)), C_(C), H_(H), W_(W), fit_(fit) {}
    Dataset load(Preprocessor& prep) override { return loadImageFolder(root_, C_, H_, W_, prep, fit_); }
    int channels() const override { return C_; }
    int height() const override { return H_; }
    int width() const override { return W_; }
    std::string describe() const override {
        return "folder " + root_ + " -> " + std::to_string(C_) + "x" + std::to_string(H_) + "x" + std::to_string(W_);
    }
};

// MNIST-style IDX files (always 1 channel; the size is read from the file).
class IdxImageSource : public ImageSource {
    std::string images_, labels_; size_t limit_; int H_ = 0, W_ = 0;
public:
    IdxImageSource(std::string images, std::string labels, size_t limit = 0)
        : images_(std::move(images)), labels_(std::move(labels)), limit_(limit) {}
    Dataset load(Preprocessor& prep) override { return loadIDX(images_, labels_, H_, W_, prep, limit_); }
    int channels() const override { return 1; }
    int height() const override { return H_; }
    int width() const override { return W_; }
    std::string describe() const override { return "IDX " + images_; }
};

} // namespace nn
