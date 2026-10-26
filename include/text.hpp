#pragma once
// Text pipeline: robust CSV reader (quotes/newlines), UTF-8 aware tokenizer (Persian friendly),
// vocabulary, n-grams, encoding to token-id matrices.
#include "dataset.hpp"
#include <unordered_map>
#include <filesystem>

namespace nn {

struct TextData {
    std::vector<std::string> texts, labels;
};

// Full RFC-4180-ish CSV parser: supports "quoted, fields", "" escapes and newlines inside quotes.
inline std::vector<std::vector<std::string>> parseCSVText(std::string s) {
    if (s.size() >= 3 && (unsigned char)s[0] == 0xEF && (unsigned char)s[1] == 0xBB && (unsigned char)s[2] == 0xBF) s.erase(0, 3);
    const char delim = detectDelimiter(s);
    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> rec; std::string field; bool inQ = false;
    auto endRecord = [&]() {
        rec.push_back(field); field.clear();
        if (!(rec.size() == 1 && trim(rec[0]).empty())) rows.push_back(rec);
        rec.clear();
    };
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (inQ) {
            if (c == '"') { if (i + 1 < s.size() && s[i + 1] == '"') { field += '"'; ++i; } else inQ = false; }
            else field += c;
        } else if (c == '"' && field.empty()) inQ = true;
        else if (c == delim) { rec.push_back(field); field.clear(); }
        else if (c == '\n' || c == '\r') { if (c == '\r' && i + 1 < s.size() && s[i + 1] == '\n') ++i; endRecord(); }
        else field += c;
    }
    if (!field.empty() || !rec.empty()) endRecord();
    return rows;
}

inline std::vector<std::vector<std::string>> readCSVRecords(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(cannotOpenMsg("file", path));
    std::stringstream ss; ss << f.rdbuf();
    return parseCSVText(ss.str());
}

// Only the first record (the header row); does not read the whole file.
inline std::vector<std::string> readCSVHeader(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(cannotOpenMsg("file", path));
    std::string rec; bool q = false; char ch;
    while (f.get(ch)) {
        if (ch == '"') q = !q;
        if ((ch == '\n' || ch == '\r') && !q) { if (!trim(rec).empty()) break; rec.clear(); continue; }   // skip leading blank lines
        rec += ch;
    }
    auto rows = parseCSVText(rec);
    return rows.empty() ? std::vector<std::string>() : rows[0];
}

// Column given as a number ("0", "-1" = last) or by its header name ("comment", case-insensitive).
inline int columnIndex(const std::string& path, const std::string& spec, int fallback) {
    const std::string sp = trim(spec);
    if (sp.empty()) return fallback;
    try { size_t pos = 0; int v = std::stoi(sp, &pos); if (pos == sp.size()) return v; } catch (...) {}
    auto header = readCSVHeader(path);
    auto lower = [](std::string x) { for (auto& ch : x) ch = (char)std::tolower((unsigned char)ch); return x; };
    for (size_t i = 0; i < header.size(); ++i) if (lower(trim(header[i])) == lower(sp)) return (int)i;
    std::string cols; for (size_t i = 0; i < header.size(); ++i) cols += (i ? " | " : "") + trim(header[i]);
    throw std::runtime_error("column '" + sp + "' not found in " + path + "\n  columns in the first row: " + cols);
}

inline TextData loadTextCSV(const std::string& path, int textCol, int labelCol, bool header) {
    auto rows = readCSVRecords(path);
    if (header && !rows.empty()) rows.erase(rows.begin());
    if (rows.empty()) throw std::runtime_error("no rows in " + path);
    TextData td;
    for (size_t i = 0; i < rows.size(); ++i) {
        int n = (int)rows[i].size();
        int tc = textCol < 0 ? textCol + n : textCol, lc = labelCol < 0 ? labelCol + n : labelCol;
        if (tc < 0 || tc >= n || lc < 0 || lc >= n) throw std::runtime_error("bad column index at record " + std::to_string(i + 1));
        if (tc == lc) throw std::runtime_error("the text column and the label column are the same column (the row has " + std::to_string(n) +
                                               " column(s)).\n  The columns must be separated by a comma, a semicolon or a TAB.");
        td.texts.push_back(rows[i][tc]); td.labels.push_back(trim(rows[i][lc]));
    }
    return td;
}

// root/<class_name>/*.txt  (one document per file)
inline TextData loadTextFolder(const std::string& root) {
    namespace fs = std::filesystem;
    if (!fs::is_directory(root)) throw std::runtime_error(cannotOpenMsg("folder", root));
    std::vector<std::string> names;
    for (auto& e : fs::directory_iterator(root)) if (e.is_directory()) names.push_back(e.path().filename().string());
    std::sort(names.begin(), names.end());
    TextData td;
    for (auto& n : names)
        for (auto& e : fs::directory_iterator(fs::path(root) / n)) {
            if (!e.is_regular_file() || e.path().extension() != ".txt") continue;
            std::ifstream f(e.path(), std::ios::binary); std::stringstream ss; ss << f.rdbuf();
            std::string txt = ss.str(); normalizeText(txt);
            td.texts.push_back(std::move(txt)); td.labels.push_back(n);
        }
    if (td.texts.empty()) throw std::runtime_error("no .txt files found in class folders of " + root);
    return td;
}

// UTF-8 tokenizer: lowercases ASCII, splits on punctuation (incl. Persian ، ؛ ؟ « » and general punctuation),
// keeps ZWNJ inside words, normalizes Arabic ي/ك to Persian ی/ک and drops tatweel/diacritics.
inline std::vector<std::string> tokenize(const std::string& s) {
    std::vector<std::string> out; std::string cur;
    auto flush = [&]() { if (!cur.empty()) { out.push_back(cur); cur.clear(); } };
    const size_t n = s.size();
    for (size_t i = 0; i < n;) {
        unsigned char c = (unsigned char)s[i];
        if (c < 0x80) {
            if (std::isalnum(c) || c == '\'') cur += (char)std::tolower(c); else flush();
            ++i;
        } else if (c == 0xD8 && i + 1 < n) {
            unsigned char d = (unsigned char)s[i + 1];
            if (d == 0x8C || d == 0x9B || d == 0x9F) flush();          // ، ؛ ؟
            else { cur += (char)c; cur += (char)d; }
            i += 2;
        } else if (c == 0xD9 && i + 1 < n) {
            unsigned char d = (unsigned char)s[i + 1];
            if (d == 0x8A) { cur += "\xDB\x8C"; }                      // ي -> ی
            else if (d == 0x83) { cur += "\xDA\xA9"; }                 // ك -> ک
            else if (d == 0x80 || (d >= 0x8B && d <= 0x92)) {}         // tatweel + harakat
            else { cur += (char)c; cur += (char)d; }
            i += 2;
        } else if (c == 0xC2 && i + 1 < n) {
            unsigned char d = (unsigned char)s[i + 1];
            if (d == 0xAB || d == 0xBB || d == 0xA0) flush(); else { cur += (char)c; cur += (char)d; }
            i += 2;
        } else if (c == 0xE2 && i + 2 < n) {
            unsigned char d = (unsigned char)s[i + 1], e = (unsigned char)s[i + 2];
            bool zw = (d == 0x80 && (e == 0x8C || e == 0x8D));         // ZWNJ / ZWJ stay inside words
            if (zw) { cur += (char)c; cur += (char)d; cur += (char)e; }
            else if (d == 0x80 || d == 0x81) flush();                  // general punctuation
            else { cur += (char)c; cur += (char)d; cur += (char)e; }
            i += 3;
        } else { cur += (char)c; ++i; }
    }
    flush();
    return out;
}

// Unigrams (+ bigrams if ngrams >= 2), capped by maxLen tokens of the original text.
inline std::vector<std::string> featureTokens(const std::string& text, int maxLen, int ngrams) {
    auto t = tokenize(text);
    if ((int)t.size() > maxLen) t.resize(maxLen);
    std::vector<std::string> f = t;
    if (ngrams >= 2) for (size_t i = 0; i + 1 < t.size(); ++i) f.push_back(t[i] + " " + t[i + 1]);
    return f;
}

inline std::vector<std::string> buildVocab(const std::vector<std::string>& texts, const std::vector<size_t>& idx,
                                           int maxLen, int ngrams, size_t maxVocab, int minCount) {
    std::unordered_map<std::string, int> cnt;
    for (size_t i : idx) for (auto& t : featureTokens(texts[i], maxLen, ngrams)) ++cnt[t];
    std::vector<std::pair<std::string, int>> v;
    for (auto& kv : cnt) if (kv.second >= minCount) v.push_back(kv);
    std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second != b.second ? a.second > b.second : a.first < b.first; });
    if (v.size() > maxVocab) v.resize(maxVocab);
    std::vector<std::string> vocab; for (auto& p : v) vocab.push_back(p.first);
    return vocab;
}

inline size_t textWidth(int maxLen, int ngrams) { return (size_t)maxLen * (ngrams >= 2 ? 2 : 1); }

// Encode one document to token ids (1-based; 0 = padding). Unknown tokens are dropped.
inline void encodeText(const std::string& text, const std::unordered_map<std::string, int>& vmap,
                       int maxLen, int ngrams, float* row) {
    const size_t width = textWidth(maxLen, ngrams);
    std::fill(row, row + width, 0.f);
    size_t k = 0;
    for (auto& t : featureTokens(text, maxLen, ngrams)) {
        auto it = vmap.find(t);
        if (it != vmap.end() && k < width) row[k++] = (float)it->second;
    }
}

inline std::unordered_map<std::string, int> vocabMap(const std::vector<std::string>& vocab) {
    std::unordered_map<std::string, int> m;
    for (size_t i = 0; i < vocab.size(); ++i) m[vocab[i]] = (int)i + 1;
    return m;
}

} // namespace nn
