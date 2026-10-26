#pragma once
// Byte-level BPE tokenizer (handles any UTF-8 text: Persian, English, emoji...).
// ids: 0..255 raw bytes | 256 <|eot|> | 257 <|user|> | 258 <|bot|> | 259.. merges
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <climits>
#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace nn {

class BPE {
public:
    static constexpr int EOT = 256, USER = 257, BOT = 258, BASE = 259;
    std::vector<std::pair<int, int>> merges;     // merge k produces token BASE+k
    std::vector<std::string> pieces;             // bytes of each token
    std::unordered_map<uint64_t, int> rank;      // (a,b) -> merge index
    mutable std::unordered_map<std::string, std::vector<int>> cache;

    int vocabSize() const { return BASE + (int)merges.size(); }
    static uint64_t key(int a, int b) { return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b; }

    void initPieces() {
        pieces.assign(BASE, "");
        for (int b = 0; b < 256; ++b) pieces[b] = std::string(1, (char)b);
    }
    void rebuild() {
        initPieces(); rank.clear(); cache.clear();
        for (size_t k = 0; k < merges.size(); ++k) {
            rank[key(merges[k].first, merges[k].second)] = (int)k;
            pieces.push_back(pieces[merges[k].first] + pieces[merges[k].second]);
        }
    }

    static bool isWordByte(unsigned char c) { return c >= 0x80 || std::isalnum(c) || c == '_'; }

    // Pre-tokenization: [space]+word-run, or single other byte.
    static std::vector<std::string> chunks(const std::string& s) {
        std::vector<std::string> out; size_t i = 0, n = s.size();
        while (i < n) {
            size_t st = i;
            if (s[i] == ' ' && i + 1 < n && isWordByte((unsigned char)s[i + 1])) ++i;
            if (isWordByte((unsigned char)s[i])) { while (i < n && isWordByte((unsigned char)s[i])) ++i; }
            else ++i;
            out.push_back(s.substr(st, i - st));
        }
        return out;
    }

    static void countChunks(const std::string& text, std::unordered_map<std::string, int64_t>& counts) {
        for (auto& c : chunks(text)) ++counts[c];
    }

    void train(const std::unordered_map<std::string, int64_t>& counts, int numMerges) {
        merges.clear();
        struct W { std::vector<int> sym; int64_t cnt; };
        std::vector<W> words; words.reserve(counts.size());
        for (auto& kv : counts) {
            W w; w.cnt = kv.second;
            for (unsigned char c : kv.first) w.sym.push_back(c);
            if (w.sym.size() > 1) words.push_back(std::move(w));
        }
        for (int m = 0; m < numMerges; ++m) {
            std::unordered_map<uint64_t, int64_t> pc;
            for (auto& w : words)
                for (size_t k = 0; k + 1 < w.sym.size(); ++k) pc[key(w.sym[k], w.sym[k + 1])] += w.cnt;
            uint64_t bestKey = 0; int64_t best = 0;
            for (auto& kv : pc)
                if (kv.second > best || (kv.second == best && kv.first < bestKey)) { best = kv.second; bestKey = kv.first; }
            if (best < 2) break;
            int a = (int)(bestKey >> 32), b = (int)(bestKey & 0xffffffffu), id = BASE + m;
            merges.push_back({a, b});
            for (auto& w : words) {
                std::vector<int> ns; ns.reserve(w.sym.size());
                for (size_t k = 0; k < w.sym.size();) {
                    if (k + 1 < w.sym.size() && w.sym[k] == a && w.sym[k + 1] == b) { ns.push_back(id); k += 2; }
                    else ns.push_back(w.sym[k++]);
                }
                w.sym.swap(ns);
            }
        }
        rebuild();
    }

    const std::vector<int>& encodeChunk(const std::string& ch) const {
        auto it = cache.find(ch);
        if (it != cache.end()) return it->second;
        std::vector<int> sym; for (unsigned char c : ch) sym.push_back(c);
        while (sym.size() > 1) {
            int bestRank = INT_MAX;
            for (size_t k = 0; k + 1 < sym.size(); ++k) {
                auto r = rank.find(key(sym[k], sym[k + 1]));
                if (r != rank.end() && r->second < bestRank) bestRank = r->second;
            }
            if (bestRank == INT_MAX) break;
            int a = merges[bestRank].first, b = merges[bestRank].second;
            std::vector<int> ns;
            for (size_t k = 0; k < sym.size();) {
                if (k + 1 < sym.size() && sym[k] == a && sym[k + 1] == b) { ns.push_back(BASE + bestRank); k += 2; }
                else ns.push_back(sym[k++]);
            }
            sym.swap(ns);
        }
        return cache.emplace(ch, std::move(sym)).first->second;
    }

    // Literal special markers inside text ("<|user|>", "<|bot|>", "<|eot|>") become special ids.
    std::vector<int> encode(const std::string& text) const {
        std::vector<int> out; std::string seg;
        auto flush = [&]() {
            if (seg.empty()) return;
            for (auto& c : chunks(seg)) { auto& e = encodeChunk(c); out.insert(out.end(), e.begin(), e.end()); }
            seg.clear();
        };
        for (size_t i = 0; i < text.size();) {
            if (text[i] == '<') {
                if (text.compare(i, 7, "<|eot|>") == 0) { flush(); out.push_back(EOT); i += 7; continue; }
                if (text.compare(i, 8, "<|user|>") == 0) { flush(); out.push_back(USER); i += 8; continue; }
                if (text.compare(i, 7, "<|bot|>") == 0) { flush(); out.push_back(BOT); i += 7; continue; }
            }
            seg += text[i++];
        }
        flush();
        return out;
    }

    std::string decodeToken(int id) const { return (id >= 0 && id < (int)pieces.size()) ? pieces[id] : ""; }
    std::string decode(const std::vector<int>& ids) const {
        std::string s; for (int t : ids) s += decodeToken(t); return s;
    }

    void save(std::ostream& o) const {
        uint32_t n = (uint32_t)merges.size(); o.write((const char*)&n, 4);
        for (auto& m : merges) { o.write((const char*)&m.first, 4); o.write((const char*)&m.second, 4); }
    }
    void load(std::istream& i) {
        uint32_t n; i.read((char*)&n, 4); merges.resize(n);
        for (auto& m : merges) { i.read((char*)&m.first, 4); i.read((char*)&m.second, 4); }
        rebuild();
    }
};

} // namespace nn
