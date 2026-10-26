#pragma once
// CSV loading, splitting, standardization, batching.
#include "tensor.hpp"
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <numeric>
#include <random>
#include <stdexcept>
#include <cstdlib>
#include <filesystem>
#include <iostream>

namespace nn {

enum class Task { Classification = 0, Regression = 1 };

struct Dataset {
    Matrix X;                 // n x features
    std::vector<int> cls;     // class ids (classification)
    std::vector<float> y;     // targets (regression)
    size_t size() const { return X.rows; }
};

// Everything needed at inference time (saved inside the model file).
struct Preprocessor {
    int normMode = 0;                   // 0 = none, 1 = per-feature, 2 = one global mean/std, 3 = per-channel (images)
    int chans = 1;                      // channel count for mode 3
    std::vector<float> mean, stdv;
    float yMean = 0.f, yStd = 1.f;      // regression target standardization
    std::vector<std::string> classes;   // class names (classification)
    std::vector<std::string> vocab;     // text models: token id = index + 1

    void fit(const Matrix& X, bool global = false) {
        if (global) {
            double m = 0; for (float v : X.data) m += v;
            m /= (double)X.size();
            double q = 0; for (float v : X.data) q += (v - m) * (v - m);
            q = std::sqrt(q / (double)X.size());
            mean.assign(1, (float)m); stdv.assign(1, q < 1e-8 ? 1.f : (float)q);
            normMode = 2; return;
        }
        normMode = 1;
        mean.assign(X.cols, 0.f); stdv.assign(X.cols, 1.f);
        for (size_t j = 0; j < X.cols; ++j) {
            double m = 0; for (size_t i = 0; i < X.rows; ++i) m += X(i, j);
            m /= (double)X.rows;
            double v = 0; for (size_t i = 0; i < X.rows; ++i) { double d = X(i, j) - m; v += d * d; }
            v = std::sqrt(v / (double)X.rows);
            mean[j] = (float)m; stdv[j] = v < 1e-8 ? 1.f : (float)v;
        }
    }
    // Per-channel mean/std for images stored as CHW rows (RGB photos have very different channel statistics).
    void fitChannels(const Matrix& X, int C) {
        normMode = 3; chans = C;
        const size_t per = X.cols / (size_t)C;
        mean.assign(C, 0.f); stdv.assign(C, 1.f);
        for (int c = 0; c < C; ++c) {
            double m = 0, q = 0; const double n = (double)X.rows * per;
            for (size_t i = 0; i < X.rows; ++i) { const float* p = &X.data[i * X.cols + c * per]; for (size_t k = 0; k < per; ++k) m += p[k]; }
            m /= n;
            for (size_t i = 0; i < X.rows; ++i) { const float* p = &X.data[i * X.cols + c * per]; for (size_t k = 0; k < per; ++k) q += (p[k] - m) * (p[k] - m); }
            q = std::sqrt(q / n);
            mean[c] = (float)m; stdv[c] = q < 1e-8 ? 1.f : (float)q;
        }
    }
    // Normalizes (or, with inverse=true, un-normalizes) one feature row of n values in place.
    void normalizeRow(float* r, size_t n, bool inverse = false) const {
        if (normMode == 1) {
            for (size_t j = 0; j < n; ++j) r[j] = inverse ? r[j] * stdv[j] + mean[j] : (r[j] - mean[j]) / stdv[j];
        } else if (normMode == 2) {
            for (size_t j = 0; j < n; ++j) r[j] = inverse ? r[j] * stdv[0] + mean[0] : (r[j] - mean[0]) / stdv[0];
        } else if (normMode == 3) {
            const size_t per = n / (size_t)chans;
            for (int c = 0; c < chans; ++c)
                for (size_t k = 0; k < per; ++k) { float& v = r[c * per + k]; v = inverse ? v * stdv[c] + mean[c] : (v - mean[c]) / stdv[c]; }
        }
    }
    void apply(Matrix& X) const { for (size_t i = 0; i < X.rows; ++i) normalizeRow(&X.data[i * X.cols], X.cols); }
};

// Text files written on Windows: strip a UTF-8 BOM and turn CRLF / lone CR into LF, so the model never sees "\r".
inline void normalizeText(std::string& s) {
    if (s.size() >= 3 && (unsigned char)s[0] == 0xEF && (unsigned char)s[1] == 0xBB && (unsigned char)s[2] == 0xBF) s.erase(0, 3);
    std::string out; out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\r') { out += '\n'; if (i + 1 < s.size() && s[i + 1] == '\n') ++i; }
        else out += s[i];
    }
    s.swap(out);
}

// "cannot open X" that also says where the program is looking (relative paths start at the working directory).
inline std::string cannotOpenMsg(const std::string& what, const std::string& path) {
    std::string cwd; try { cwd = std::filesystem::current_path().string(); } catch (...) { cwd = "?"; }
    return "cannot open " + what + ": " + path + "\n  current working directory: " + cwd +
           "\n  -> use a full path (e.g. C:\\data\\x.txt or /home/me/x.txt) or run from the folder that contains it";
}

inline std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

// Error text for a cell that cannot be read as a number; an empty cell is a *missing value*, not "text".
inline std::string badNumberMsg(size_t row, size_t col, const std::string& cell) {
    std::string where = "row " + std::to_string(row) + " col " + std::to_string(col);
    if (trim(cell).empty())
        return where + ": missing (empty) value. Remove that row or fill it in (SQL: WHERE col IS NOT NULL, or COALESCE(col, 0)).";
    bool decimalComma = cell.find(',') != std::string::npos && cell.find(',') == cell.rfind(',');
    for (char ch : cell) if (!(std::isdigit((unsigned char)ch) || ch == ',' || ch == '-' || ch == '+')) decimalComma = false;
    return where + ": non-numeric feature '" + cell + "'" +
           (decimalComma ? "  (looks like a decimal comma - write numbers with '.' as the decimal separator)" : "");
}

// Warnings go to this stream (std::cerr by default; the self-test points it at a string to check the wording).
inline std::ostream*& warnStream() { static std::ostream* p = &std::cerr; return p; }

inline size_t countEmptyLabels(const std::vector<std::string>& labels) {
    return (size_t)std::count_if(labels.begin(), labels.end(), [](const std::string& l) { return trim(l).empty(); });
}
// Which character separates the columns? Excel in many regions writes ';' (or tab) instead of ','.
// Looks at the first non-empty line only, ignoring anything inside quotes; the most frequent of , ; TAB wins (ties: comma).
inline char detectDelimiter(const std::string& text) {
    static const char cand[3] = {',', ';', '\t'};
    size_t counts[3] = {0, 0, 0}; bool q = false, content = false;
    size_t i = (text.size() >= 3 && (unsigned char)text[0] == 0xEF && (unsigned char)text[1] == 0xBB && (unsigned char)text[2] == 0xBF) ? 3 : 0;
    for (; i < text.size(); ++i) {
        const char c = text[i];
        if (c == '"') { q = !q; content = true; }
        else if (!q && (c == '\n' || c == '\r')) {
            if (content) break;                                   // end of the first real line
            counts[0] = counts[1] = counts[2] = 0;                // a blank / whitespace-only line: forget it
        } else if (!q) {
            for (int k = 0; k < 3; ++k) if (c == cand[k]) ++counts[k];
            if (c != ' ' && c != '\t') content = true;            // blanks and indentation are not content
        }
    }
    int best = 0; for (int k = 1; k < 3; ++k) if (counts[k] > counts[best]) best = k;
    return cand[best];
}

inline std::vector<std::string> splitCSV(const std::string& line, char delim = ',') {
    std::vector<std::string> out; std::string cur; bool q = false;
    for (char c : line) {
        if (c == '"') q = !q;
        else if (c == delim && !q) { out.push_back(trim(cur)); cur.clear(); }
        else cur += c;
    }
    out.push_back(trim(cur));
    return out;
}
inline bool parseFloat(const std::string& s, float& v) {
    if (s.empty()) return false;
    char* end = nullptr; v = std::strtof(s.c_str(), &end);
    return end != s.c_str() && *end == '\0';
}

// Reads rows of fields. Auto-detects (and skips) a header row.
inline std::vector<std::vector<std::string>> readCSV(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error(cannotOpenMsg("file", path));
    std::vector<std::vector<std::string>> rows;
    std::string line; bool first = true; char delim = ',';
    while (std::getline(f, line)) {
        if (trim(line).empty()) continue;
        if (first) {
            if (line.size() >= 3 && (unsigned char)line[0] == 0xEF && (unsigned char)line[1] == 0xBB && (unsigned char)line[2] == 0xBF) line.erase(0, 3);
            delim = detectDelimiter(line);
        }
        auto fields = splitCSV(line, delim);
        if (first) {
            first = false;
            bool allNum = true; float tmp;
            for (auto& s : fields) if (!parseFloat(s, tmp)) allNum = false;
            // header if any field is non-numeric AND every field is non-numeric-looking-or-text
            bool anyNum = false;
            for (auto& s : fields) if (parseFloat(s, tmp)) anyNum = true;
            if (!allNum && !anyNum) continue;   // pure text row => header
        }
        rows.push_back(std::move(fields));
    }
    return rows;
}

// Map label strings to class ids (sorted numerically if all numeric, else alphabetically).
inline std::vector<int> mapClasses(const std::vector<std::string>& labels, Preprocessor& prep) {
    if (size_t n = countEmptyLabels(labels))   // NULL / "" in the label column would otherwise become a silent extra class ""
        *warnStream() << "warning: " << n << " of " << labels.size() << " rows have an EMPTY label and will form their own class \"\".\n"
                      << "         Remove them (SQL: WHERE label IS NOT NULL AND label != '') or fill in the label.\n";
    std::set<std::string> uniq(labels.begin(), labels.end());
    std::vector<std::string> names(uniq.begin(), uniq.end());
    bool numeric = true; float tmp;
    for (auto& s : names) if (!parseFloat(s, tmp)) numeric = false;
    if (numeric) std::sort(names.begin(), names.end(), [](const std::string& a, const std::string& b) {
        return std::strtof(a.c_str(), nullptr) < std::strtof(b.c_str(), nullptr); });
    prep.classes = names;
    std::map<std::string, int> id;
    for (size_t k = 0; k < names.size(); ++k) id[names[k]] = (int)k;
    std::vector<int> out;
    for (auto& s : labels) out.push_back(id[s]);
    return out;
}

// Load labeled data. labelCol < 0 counts from the end (-1 = last column).
inline Dataset loadLabeled(const std::string& path, int labelCol, Task task,
                           Preprocessor& prep) {
    auto rows = readCSV(path);
    if (rows.empty()) throw std::runtime_error("empty dataset");
    int ncol = (int)rows[0].size();
    if (ncol < 2) throw std::runtime_error("only " + std::to_string(ncol) + " column found in " + path + ": need feature columns plus a label.\n"
                                           "  The columns must be separated by a comma, a semicolon or a TAB (found none of them in the first line).");
    if (labelCol < 0) labelCol += ncol;
    if (labelCol < 0 || labelCol >= ncol) throw std::runtime_error("bad --label-col");

    Dataset ds;
    ds.X = Matrix(rows.size(), ncol - 1);
    std::vector<std::string> labels(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        if ((int)rows[i].size() != ncol)
            throw std::runtime_error("row " + std::to_string(i + 1) + ": wrong column count");
        int c = 0;
        for (int j = 0; j < ncol; ++j) {
            if (j == labelCol) { labels[i] = rows[i][j]; continue; }
            float v;
            if (!parseFloat(rows[i][j], v))
                throw std::runtime_error(badNumberMsg(i + 1, j, rows[i][j]));
            ds.X(i, c++) = v;
        }
    }
    if (task == Task::Classification) {
        ds.cls = mapClasses(labels, prep);
    } else {
        for (size_t i = 0; i < labels.size(); ++i) {
            float v; if (!parseFloat(labels[i], v)) throw std::runtime_error("non-numeric target at row " + std::to_string(i + 1));
            ds.y.push_back(v);
        }
    }
    return ds;
}

// Load features only (for prediction). skipCol: column to ignore (or -999 for none).
inline Matrix loadFeatures(const std::string& path, int skipCol, bool hasLabelToSkip) {
    auto rows = readCSV(path);
    if (rows.empty()) throw std::runtime_error("empty file");
    int ncol = (int)rows[0].size();
    if (hasLabelToSkip && skipCol < 0) skipCol += ncol;
    Matrix X(rows.size(), ncol - (hasLabelToSkip ? 1 : 0));
    for (size_t i = 0; i < rows.size(); ++i) {
        int c = 0;
        for (int j = 0; j < ncol; ++j) {
            if (hasLabelToSkip && j == skipCol) continue;
            float v;
            if (!parseFloat(rows[i][j], v)) throw std::runtime_error("non-numeric value at row " + std::to_string(i + 1));
            X(i, c++) = v;
        }
    }
    return X;
}

inline Dataset subset(const Dataset& d, const std::vector<size_t>& idx) {
    Dataset s; s.X = Matrix(idx.size(), d.X.cols);
    for (size_t k = 0; k < idx.size(); ++k) {
        std::copy(&d.X.data[idx[k] * d.X.cols], &d.X.data[idx[k] * d.X.cols] + d.X.cols, &s.X.data[k * d.X.cols]);
        if (!d.cls.empty()) s.cls.push_back(d.cls[idx[k]]);
        if (!d.y.empty()) s.y.push_back(d.y[idx[k]]);
    }
    return s;
}

inline void trainValSplit(const Dataset& all, float valFrac, unsigned seed, Dataset& train, Dataset& val) {
    std::vector<size_t> idx(all.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);
    size_t nv = (size_t)(valFrac * all.size());
    std::vector<size_t> vi(idx.begin(), idx.begin() + nv), ti(idx.begin() + nv, idx.end());
    train = subset(all, ti); val = subset(all, vi);
}

// Build target matrix for rows [idx...]
inline Matrix makeBatchTargets(const Dataset& d, const std::vector<size_t>& idx, size_t b, size_t e,
                               Task task, size_t nClasses, const Preprocessor& prep) {
    const size_t n = e - b;
    if (task == Task::Classification) {
        Matrix Y(n, nClasses);
        for (size_t k = 0; k < n; ++k) Y(k, d.cls[idx[b + k]]) = 1.f;
        return Y;
    }
    Matrix Y(n, 1);
    for (size_t k = 0; k < n; ++k) Y(k, 0) = (d.y[idx[b + k]] - prep.yMean) / prep.yStd;
    return Y;
}

inline Matrix makeBatchX(const Dataset& d, const std::vector<size_t>& idx, size_t b, size_t e) {
    Matrix X(e - b, d.X.cols);
    for (size_t k = 0; k < e - b; ++k)
        std::copy(&d.X.data[idx[b + k] * d.X.cols], &d.X.data[idx[b + k] * d.X.cols] + d.X.cols, &X.data[k * d.X.cols]);
    return X;
}


inline void splitIndices(size_t n, float valFrac, unsigned seed, std::vector<size_t>& tr, std::vector<size_t>& va) {
    std::vector<size_t> idx(n); std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(seed); std::shuffle(idx.begin(), idx.end(), rng);
    size_t nv = (size_t)(valFrac * n);
    va.assign(idx.begin(), idx.begin() + nv); tr.assign(idx.begin() + nv, idx.end());
}

} // namespace nn
