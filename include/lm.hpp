#pragma once
// Language-model commands: train-lm / chat / generate / gradcheck-lm
#include "transformer.hpp"
#include "optimizer.hpp"
#include "text.hpp"
#include <chrono>
#include <iostream>
#include <filesystem>
#include <csignal>

namespace nn {

using LArgs = std::map<std::string, std::string>;
inline std::string aget(const LArgs& a, const std::string& k, const std::string& d) {
    auto it = a.find(k); return it == a.end() ? d : it->second;
}

struct Stream {
    std::vector<int> ids; std::vector<float> mask;
    void add(const std::vector<int>& t, float m) { ids.insert(ids.end(), t.begin(), t.end()); mask.insert(mask.end(), t.size(), m); }
    void add1(int t, float m) { ids.push_back(t); mask.push_back(m); }
};

inline std::string readWholeFile(const std::string& p) {
    std::ifstream f(p, std::ios::binary); if (!f) throw std::runtime_error(cannotOpenMsg("file", p));
    std::stringstream ss; ss << f.rdbuf(); std::string s = ss.str(); normalizeText(s); return s;
}

// path = a .txt file, a folder of .txt files, or a .csv whose column `textCol` (number or header name) holds the documents.
inline std::vector<std::string> collectTexts(const std::string& path, const std::string& textCol = "0", bool header = true) {
    namespace fs = std::filesystem;
    std::vector<std::string> out;
    if (!fs::exists(path)) throw std::runtime_error(cannotOpenMsg("text file or folder", path));
    {   std::string ext = fs::path(path).extension().string(); for (auto& ch : ext) ch = (char)std::tolower((unsigned char)ch);
        if (fs::is_regular_file(path) && ext == ".csv") {
            auto rows = readCSVRecords(path);
            if (header && !rows.empty()) rows.erase(rows.begin());
            const int col = columnIndex(path, textCol, 0);
            std::string joined;
            for (auto& r : rows) { if ((int)r.size() <= col || col < 0) continue; std::string t = trim(r[col]); if (!t.empty()) joined += t + "\n"; }
            if (joined.empty()) throw std::runtime_error("no text found in column '" + textCol + "' of " + path);
            normalizeText(joined); out.push_back(std::move(joined)); return out;
        } }
    if (fs::is_directory(path)) {
        std::vector<std::string> files;
        for (auto& e : fs::recursive_directory_iterator(path)) if (e.is_regular_file() && e.path().extension() == ".txt") files.push_back(e.path().string());
        std::sort(files.begin(), files.end());
        for (auto& f : files) out.push_back(readWholeFile(f));
    } else out.push_back(readWholeFile(path));
    return out;
}

// ---------------- sampling ----------------
struct SampleOpts { float temp = 0.8f; int topk = 40; float topp = 0.95f; float rep = 1.0f; };

inline int sampleToken(std::vector<float> lg, const SampleOpts& o, const std::vector<int>& recent, std::mt19937& rng, const BPE& bpe) {
    lg[BPE::USER] = lg[BPE::BOT] = -1e30f;
    // Repetition penalty only for multi-byte pieces: never punish single bytes / punctuation like '+' or UTF-8 halves.
    if (o.rep > 1.f) for (int t : recent) if (bpe.pieces[t].size() >= 3) lg[t] = lg[t] > 0 ? lg[t] / o.rep : lg[t] * o.rep;
    if (o.temp <= 0.f) return (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
    std::vector<int> idx(lg.size()); std::iota(idx.begin(), idx.end(), 0);
    size_t k = (o.topk > 0 && (size_t)o.topk < idx.size()) ? (size_t)o.topk : idx.size();
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    idx.resize(k);
    std::vector<double> p(k); double mx = lg[idx[0]], sum = 0;
    for (size_t i = 0; i < k; ++i) { p[i] = std::exp((lg[idx[i]] - mx) / o.temp); sum += p[i]; }
    for (auto& v : p) v /= sum;
    size_t cut = k; double acc = 0;
    for (size_t i = 0; i < k; ++i) { acc += p[i]; if (acc >= o.topp) { cut = i + 1; break; } }
    std::discrete_distribution<int> dist(p.begin(), p.begin() + cut);
    return idx[dist(rng)];
}

// Keeps a sliding context window on top of the KV-cache generator.
class Session {
    GPT& m; Generator g; std::vector<int> win;
public:
    explicit Session(GPT& model) : m(model), g(model) {}
    void reset() { win.clear(); g.reset(); }
    const std::vector<float>& feed(int t) {
        if (g.pos >= m.cfg.ctx) {
            size_t keep = (size_t)m.cfg.ctx / 2;
            std::vector<int> tail(win.end() - std::min(keep, win.size()), win.end());
            auto it = std::find(tail.begin(), tail.end(), (int)BPE::USER);
            if (it != tail.end() && tail.end() - it >= 8) tail.erase(tail.begin(), it);
            win = tail; g.reset();
            for (int x : win) g.step(x);
        }
        win.push_back(t);
        return g.step(t);
    }
};

// Generates until EOT / maxNew. Writes decoded bytes to stdout as they are produced. Returns generated ids.
inline std::vector<int> generateText(GPT& m, Session& s, const std::vector<int>& prompt, int maxNew, const SampleOpts& so,
                                     std::mt19937& rng, bool stream, std::vector<int>& history) {
    std::vector<float> lg;
    for (int t : prompt) { lg = s.feed(t); history.push_back(t); }
    std::vector<int> out;
    for (int i = 0; i < maxNew; ++i) {
        std::vector<int> recent(history.end() - std::min<size_t>(64, history.size()), history.end());
        int t = sampleToken(lg, so, recent, rng, m.bpe);
        if (t == BPE::EOT) break;
        out.push_back(t); history.push_back(t);
        if (stream) { std::cout << m.bpe.decodeToken(t) << std::flush; }
        lg = s.feed(t);
    }
    return out;
}

inline SampleOpts sampleOptsFrom(const LArgs& a) {
    SampleOpts so;
    so.temp = std::stof(aget(a, "temp", "0.8")); so.topk = std::stoi(aget(a, "top-k", "40"));
    so.topp = std::stof(aget(a, "top-p", "0.95")); so.rep = std::stof(aget(a, "rep-penalty", "1.0"));
    return so;
}


// ---------------- model-size presets ----------------
struct Preset { const char* name; int ctx, dim, layers, heads; float lr; };
inline const std::vector<Preset>& presets() {
    static const std::vector<Preset> p = {
        {"tiny",   64,   64,  2,  4, 1e-3f},   // ~0.1M  - smoke tests
        {"small", 128,  128,  4,  4, 6e-4f},   // ~1M    - default, small datasets
        {"medium",256,  256,  6,  8, 4e-4f},   // ~5M
        {"large", 512,  512,  8,  8, 3e-4f},   // ~25M
        {"xl",   1024,  768, 12, 12, 2.5e-4f}, // ~90M
    };
    return p;
}
// Fills ctx/dim/layers/heads/lr from --preset unless given explicitly.
inline LArgs applyPreset(LArgs a) {
    if (!a.count("preset")) return a;
    for (auto& p : presets()) if (a["preset"] == p.name) {
        auto def = [&](const char* k, const std::string& v) { if (!a.count(k)) a[k] = v; };
        def("ctx", std::to_string(p.ctx)); def("dim", std::to_string(p.dim));
        def("layers", std::to_string(p.layers)); def("heads", std::to_string(p.heads));
        def("lr", std::to_string(p.lr));
        // training-size defaults per preset: merges (vocab), batch, accum, steps, dropout, warmup
        static const std::map<std::string, std::vector<std::string>> X = {
            {"tiny",   {"300",  "16", "1", "1000",  "0.1",  "50"}},
            {"small",  {"1000", "16", "1", "3000",  "0.1",  "100"}},
            {"medium", {"3000", "16", "1", "10000", "0.1",  "200"}},
            {"large",  {"6000", "8",  "4", "30000", "0.05", "500"}},
            {"xl",     {"8000", "4",  "8", "60000", "0.0",  "1000"}},
        };
        const char* keys[] = {"merges", "batch", "accum", "steps", "dropout", "warmup"};
        const auto& v = X.at(p.name);
        for (int i = 0; i < 6; ++i) def(keys[i], v[i]);
        return a;
    }
    throw std::runtime_error("unknown --preset (tiny|small|medium|large|xl)");
}

inline int cmdInfo(const LArgs& a0) {
    if (a0.count("model")) {   // tabular / image / text-classifier files (MNN2)
        std::ifstream f(a0.at("model"), std::ios::binary); char mg[4] = {0, 0, 0, 0}; f.read(mg, 4); f.close();
        if (std::memcmp(mg, "MNN2", 4) == 0 || std::memcmp(mg, "MNN3", 4) == 0) {
            Model cm; Preprocessor pp; loadModel(a0.at("model"), cm, pp);
            std::printf("%s", cm.summary().c_str());
            std::printf("input %d | outputs %d | classes: %zu\n", cm.cfg.inputDim, cm.cfg.outputDim, pp.classes.size());
            return 0;
        }
    }
    LArgs a = applyPreset(a0); GPTConfig c; int batch = std::stoi(aget(a, "batch", "16"));
    if (a.count("model")) { GPT m; loadLM(a.at("model"), m); c = m.cfg; }
    else {
        c.vocab = std::stoi(aget(a, "vocab", std::to_string(259 + std::stoi(aget(a, "merges", "1000"))))); c.ctx = std::stoi(aget(a, "ctx", "128")); c.dim = std::stoi(aget(a, "dim", "128"));
        c.layers = std::stoi(aget(a, "layers", "4")); c.heads = std::stoi(aget(a, "heads", "4"));
    }
    LMEstimate e = estimateLM(c, batch);
    std::printf("config: vocab %d | ctx %d | dim %d | layers %d | heads %d\n", c.vocab, c.ctx, c.dim, c.layers, c.heads);
    std::printf("parameters        : %.2f M\n", e.params / 1e6);
    std::printf("training RAM      : ~%.2f GB at batch %d (weights+grads+Adam+activations)\n", e.trainBytes / 1e9, batch);
    std::printf("compute / token   : ~%.1f MFLOP (training)\n", e.flopsPerToken / 1e6);
    std::printf("data needed       : ~%.0f M tokens for a well-trained model (20 tokens per parameter)\n", e.minTokens / 1e6);
    for (double gf : {20.0, 100.0}) {
        double tps = gf * 1e9 / e.flopsPerToken;
        std::printf("at %3.0f GFLOPS     : ~%.0f tokens/s -> %.1f h for that many tokens\n", gf, tps, e.minTokens / tps / 3600.0);
    }
    return 0;
}

inline int cmdGrow(const LArgs& a) {
    GPT m; loadLM(aget(a, "model", "lm.bin"), m);
    int L = std::stoi(aget(a, "layers", std::to_string(m.cfg.layers))), C = std::stoi(aget(a, "ctx", std::to_string(m.cfg.ctx)));
    std::printf("growing: layers %d -> %d, ctx %d -> %d (%.2fM params)\n", m.cfg.layers, L, m.cfg.ctx, C, m.paramCount() / 1e6);
    growLM(m, L, C, (unsigned)std::stoul(aget(a, "seed", "7")));
    std::string out = aget(a, "out", "lm_grown.bin"); saveLM(out, m);
    std::printf("saved %s (%.2fM params). Continue training it with: train-lm --resume %s ...\n", out.c_str(), m.paramCount() / 1e6, out.c_str());
    return 0;
}

// Prints the training-cost estimate (uses estimateLM from transformer.hpp).
inline void printEstimate(const GPTConfig& c, int B, int accum) {
    LMEstimate e = estimateLM(c, B);
    const double gb = 1024.0 * 1024.0 * 1024.0, bytes = e.trainBytes + (accum > 1 ? 4.0 * e.params : 0.0);
    std::printf("model: %.2fM params | est. training RAM %.2f GB | ~%.2f GFLOP per optimizer step\n",
                e.params / 1e6, bytes / gb, e.flopsPerToken * (double)B * c.ctx * std::max(1, accum) / 1e9);
    if (bytes / gb > 12.0) std::printf("WARNING: this will not fit comfortably in 16 GB RAM - lower --batch/--ctx/--dim.\n");
}

inline volatile std::sig_atomic_t g_stopFlag = 0;
inline void onSigInt(int) { g_stopFlag = 1; }

// ---------------- commands ----------------
inline int cmdGenerate(const LArgs& a) {
    GPT m; loadLM(aget(a, "model", "lm.bin"), m);
    std::mt19937 rng((unsigned)std::stoul(aget(a, "seed", "0")) + (unsigned)std::chrono::steady_clock::now().time_since_epoch().count());
    Session s(m); std::vector<int> hist;
    std::string prompt = aget(a, "prompt", "");
    std::vector<int> p = m.bpe.encode(prompt);
    int maxNew = std::stoi(aget(a, "max-new", "200"));
    if ((int)p.size() > m.cfg.ctx - 8) p.erase(p.begin(), p.end() - (m.cfg.ctx - 8));
    std::cout << prompt << std::flush;
    generateText(m, s, p, maxNew, sampleOptsFrom(a), rng, true, hist);
    std::cout << "\n";
    return 0;
}

inline int cmdChat(const LArgs& a) {
    GPT m; loadLM(aget(a, "model", "lm.bin"), m);
    std::mt19937 rng((unsigned)std::chrono::steady_clock::now().time_since_epoch().count());
    SampleOpts so = sampleOptsFrom(a);
    int maxNew = std::min(std::stoi(aget(a, "max-new", "150")), m.cfg.ctx - 16);
    Session s(m); std::vector<int> hist;
    std::cout << "minnn chat  (/reset = new conversation, /quit = exit)\n";
    std::string line;
    while (true) {
        std::cout << "\nyou> " << std::flush;
        if (!std::getline(std::cin, line)) break;
        line = trim(line);
        if (line.empty()) continue;
        if (line == "/quit") break;
        if (line == "/reset") { s.reset(); hist.clear(); std::cout << "(conversation cleared)\n"; continue; }
        std::vector<int> p = {BPE::USER};
        auto e = m.bpe.encode(line); p.insert(p.end(), e.begin(), e.end()); p.push_back(BPE::BOT);
        std::cout << "bot> " << std::flush;
        auto reply = generateText(m, s, p, maxNew, so, rng, true, hist);
        std::cout << "\n";
        s.feed(BPE::EOT); hist.push_back(BPE::EOT);
    }
    return 0;
}

inline int cmdTrainLM(const LArgs& a0) {
    namespace ch = std::chrono;
    const LArgs a = applyPreset(a0);
    const unsigned seed = (unsigned)std::stoul(aget(a, "seed", "42"));
    std::mt19937 rng(seed);

    // ---- gather data ----
    std::vector<std::string> texts;
    if (a.count("text")) texts = collectTexts(a.at("text"), aget(a, "text-col", "0"), aget(a, "header", "1") != "0");
    struct Pair { std::string p, r; };
    std::vector<Pair> pairs;
    if (a.count("chat")) {
        auto rows = readCSVRecords(a.at("chat"));
        if (aget(a, "header", "1") != "0" && !rows.empty()) rows.erase(rows.begin());
        int pc = columnIndex(a.at("chat"), aget(a, "prompt-col", "0"), 0), rc = columnIndex(a.at("chat"), aget(a, "response-col", "1"), 1);
        for (auto& r : rows) {
            if ((int)r.size() <= std::max(pc, rc)) continue;
            std::string p = trim(r[pc]), s = trim(r[rc]);
            if (!p.empty() && !s.empty()) pairs.push_back({p, s});
        }
        if (pairs.empty()) throw std::runtime_error("no prompt/response pairs found: check --prompt-col / --response-col and that the columns are separated by a comma, semicolon or TAB");
    }
    if (texts.empty() && pairs.empty()) throw std::runtime_error("give --text file/dir and/or --chat pairs.csv");

    // ---- model + tokenizer ----
    GPT m;
    if (a.count("resume")) {
        loadLM(a.at("resume"), m);
        std::printf("resumed from %s\n", a.at("resume").c_str());
    } else {
        std::unordered_map<std::string, int64_t> counts;
        for (auto& t : texts) BPE::countChunks(t, counts);
        for (auto& p : pairs) { BPE::countChunks(p.p, counts); BPE::countChunks(p.r, counts); }
        m.bpe.train(counts, std::stoi(aget(a, "merges", "1000")));
        GPTConfig c;
        c.vocab = m.bpe.vocabSize(); c.ctx = std::stoi(aget(a, "ctx", "128")); c.dim = std::stoi(aget(a, "dim", "128"));
        c.layers = std::stoi(aget(a, "layers", "4")); c.heads = std::stoi(aget(a, "heads", "4"));
        c.dropout = std::stof(aget(a, "dropout", "0.1"));
        m.init(c, seed);
    }
    if (a.count("dropout") && a.count("resume")) m.cfg.dropout = std::stof(a.at("dropout"));
    const int T = m.cfg.ctx;

    // ---- encode ----
    const float valFrac = std::stof(aget(a, "val", "0.05"));
    const bool onPrompt = a.count("train-on-prompt") > 0;
    Stream tr, va;
    for (auto& t : texts) {
        auto ids = m.bpe.encode(t);
        size_t nv = (size_t)(ids.size() * valFrac), cut = ids.size() - nv;
        tr.add(std::vector<int>(ids.begin(), ids.begin() + cut), 1.f); tr.add1(BPE::EOT, 1.f);
        if (nv) { va.add(std::vector<int>(ids.begin() + cut, ids.end()), 1.f); va.add1(BPE::EOT, 1.f); }
    }
    std::shuffle(pairs.begin(), pairs.end(), rng);
    size_t nvp = (size_t)(pairs.size() * valFrac);
    for (size_t i = 0; i < pairs.size(); ++i) {
        Stream& dst = i < nvp ? va : tr;
        dst.add1(BPE::USER, onPrompt ? 1.f : 0.f); dst.add(m.bpe.encode(pairs[i].p), onPrompt ? 1.f : 0.f);
        dst.add1(BPE::BOT, onPrompt ? 1.f : 0.f);  dst.add(m.bpe.encode(pairs[i].r), 1.f); dst.add1(BPE::EOT, 1.f);
    }
    std::printf("vocab %d | train tokens %zu | val tokens %zu | model %.2fM params | ctx %d dim %d layers %d heads %d\n",
                m.cfg.vocab, tr.ids.size(), va.ids.size(), m.paramCount() / 1e6, T, m.cfg.dim, m.cfg.layers, m.cfg.heads);
    printEstimate(m.cfg, std::stoi(aget(a, "batch", "16")), std::stoi(aget(a, "accum", "1")));
    if ((int)tr.ids.size() < T + 2) throw std::runtime_error("not enough training data for this --ctx (lower --ctx or add data)");

    // ---- training ----
    const int steps = std::stoi(aget(a, "steps", "2000")), Bsz = std::stoi(aget(a, "batch", "16"));
    const int accum = std::max(1, std::stoi(aget(a, "accum", "1")));      // gradient accumulation (bigger effective batch)
    const int warmup = std::stoi(aget(a, "warmup", "100")), evalEvery = std::stoi(aget(a, "eval-every", "200"));
    const float lrMax = std::stof(aget(a, "lr", "6e-4")), clip = std::stof(aget(a, "clip", "1.0"));
    const std::string out = aget(a, "out", "lm.bin");
    Adam opt(lrMax, std::stof(aget(a, "wd", "0.01")), 0.9f, 0.95f);
    auto ps = m.params();
    bool optResumed = a.count("resume") && opt.loadState(a.at("resume") + ".opt", ps);
    if (optResumed) std::printf("restored optimizer state from %s.opt\n", a.at("resume").c_str());
    std::vector<std::vector<float>> acc;
    if (accum > 1) for (auto& p : ps) acc.emplace_back(p.value->size(), 0.f);

    std::vector<int> in(Bsz * T), tg(Bsz * T); std::vector<float> mk(Bsz * T);
    auto makeBatch = [&](const Stream& s, std::mt19937& r) {
        std::uniform_int_distribution<size_t> pick(0, s.ids.size() - T - 2);
        for (int b = 0; b < Bsz; ++b) {
            size_t st = 0; float msum = 0;
            for (int tries = 0; tries < 30 && msum <= 0; ++tries) {
                st = pick(r); msum = 0; for (int t = 0; t < T; ++t) msum += s.mask[st + 1 + t];
            }
            for (int t = 0; t < T; ++t) { in[b * T + t] = s.ids[st + t]; tg[b * T + t] = s.ids[st + 1 + t]; mk[b * T + t] = s.mask[st + 1 + t]; }
        }
    };
    auto evaluate = [&]() {
        const Stream& s = va.ids.size() > (size_t)T + 2 ? va : tr;
        std::mt19937 r(99); double tot = 0; const int nb = 8;
        for (int i = 0; i < nb; ++i) { makeBatch(s, r); tot += m.step(in.data(), tg.data(), mk.data(), Bsz, T, false, false); }
        return (float)(tot / nb);
    };

    g_stopFlag = 0; std::signal(SIGINT, onSigInt);
    std::printf("effective batch: %d x %d x %d = %d tokens/step | Ctrl+C saves %s.last and stops\n",
                Bsz, accum, T, Bsz * accum * T, out.c_str());

    float best = 1e30f; auto t0 = ch::steady_clock::now(); double run = 0; int runN = 0;
    for (int it = 1; it <= steps; ++it) {
        if (g_stopFlag) { saveLM(out + ".last", m); opt.saveState(out + ".last.opt"); std::printf("\ninterrupted at step %d -> saved %s.last (+ .opt); continue with --resume %s.last\n", it - 1, out.c_str(), out.c_str()); break; }
        float lr = it <= warmup ? lrMax * it / warmup
                                : lrMax * (0.1f + 0.9f * 0.5f * (1.f + std::cos(3.14159265f * (it - warmup) / std::max(1, steps - warmup))));
        opt.lr = lr;
        float loss = 0.f;
        for (int k = 0; k < accum; ++k) {
            makeBatch(tr, rng);
            loss += m.step(in.data(), tg.data(), mk.data(), Bsz, T, true, true) / accum;
            if (accum > 1) for (size_t i = 0; i < ps.size(); ++i) { auto& g = ps[i].grad->data; for (size_t j = 0; j < g.size(); ++j) acc[i][j] += g[j]; }
        }
        if (accum > 1) for (size_t i = 0; i < ps.size(); ++i) { auto& g = ps[i].grad->data; for (size_t j = 0; j < g.size(); ++j) { g[j] = acc[i][j] / accum; acc[i][j] = 0.f; } }
        double sq = 0; for (auto& p : ps) for (float g : p.grad->data) sq += (double)g * g;
        float norm = (float)std::sqrt(sq);
        if (clip > 0 && norm > clip) { float sc = clip / norm; for (auto& p : ps) for (auto& g : p.grad->data) g *= sc; }
        opt.step(ps);
        run += loss; ++runN;
        if (it % 20 == 0 || it == 1) {
            double secs = ch::duration<double>(ch::steady_clock::now() - t0).count();
            std::printf("step %5d/%d | loss %.3f | lr %.2e | %.0f tok/s\n", it, steps, run / runN, lr, (double)it * Bsz * accum * T / secs);
            std::fflush(stdout); run = 0; runN = 0;
        }
        if (it % evalEvery == 0 || it == steps) {
            float v = evaluate();
            std::printf(">>> eval @ step %d | val loss %.3f | perplexity %.1f%s\n", it, v, std::exp(v), v < best ? "  (saved)" : "");
            if (v < best) { best = v; saveLM(out, m); opt.saveState(out + ".opt"); }
        }
    }
    std::printf("done. best val loss %.3f -> %s\n", best, out.c_str());

    if (!std::filesystem::exists(out)) return 0;
    // quick sample from the best checkpoint
    GPT best_m; loadLM(out, best_m); Session s(best_m); std::vector<int> hist; std::mt19937 gr(7);
    std::vector<int> prompt = pairs.empty() ? std::vector<int>(tr.ids.begin(), tr.ids.begin() + std::min<size_t>(8, tr.ids.size()))
                                            : std::vector<int>{BPE::USER};
    if (!pairs.empty()) { auto e = best_m.bpe.encode(pairs.back().p); prompt.insert(prompt.end(), e.begin(), e.end()); prompt.push_back(BPE::BOT); }
    std::cout << "--- sample ---\n" << best_m.bpe.decode(prompt) << std::flush;
    generateText(best_m, s, prompt, 80, SampleOpts(), gr, true, hist);
    std::cout << "\n";
    return 0;
}

// Finite-difference check of the hand-written GPT backward pass.
// The default init (std 0.02) makes attention nearly uniform, so Q/K gradients are tiny and a bug there would hide
// below float noise. We therefore rescale every weight matrix (x10) so each path carries a real gradient, then
// require |analytic - numeric| <= 1e-4 + 3% of the larger value for 8 random entries of EVERY tensor.
inline int cmdGradCheck() {
    GPT m; GPTConfig c; c.vocab = 24; c.ctx = 8; c.dim = 16; c.layers = 2; c.heads = 2; c.dropout = 0.f;
    m.init(c, 5);
    auto ps = m.params();
    for (auto& p : ps) if (p.value->rows > 1) for (auto& v : p.value->data) v *= 10.f;
    const int B = 2, T = 8; std::vector<int> in(B * T), tg(B * T); std::vector<float> mk(B * T);
    std::mt19937 r(1);
    for (int i = 0; i < B * T; ++i) { in[i] = r() % c.vocab; tg[i] = r() % c.vocab; mk[i] = (r() % 4) ? 1.f : 0.f; }
    m.step(in.data(), tg.data(), mk.data(), B, T, false, true);
    double worst = 0; const float eps = 3e-3f; size_t checked = 0, nonzero = 0;
    for (auto& p : ps) {
        for (int k = 0; k < 8; ++k) {
            size_t i = r() % p.value->size(); float orig = p.value->data[i], ana = p.grad->data[i];
            p.value->data[i] = orig + eps; float lp = m.step(in.data(), tg.data(), mk.data(), B, T, false, false);
            p.value->data[i] = orig - eps; float lm = m.step(in.data(), tg.data(), mk.data(), B, T, false, false);
            p.value->data[i] = orig;
            double num = (lp - lm) / (2.0 * eps);
            double tol = 1e-4 + 0.03 * std::max(std::fabs((double)ana), std::fabs(num));
            worst = std::max(worst, std::fabs(ana - num) / tol); ++checked;
            if (std::fabs(ana) > 1e-3) ++nonzero;
        }
    }
    std::printf("gradient check: %zu entries (%zu with |grad| > 1e-3), worst error = %.2f x tolerance -> %s\n",
                checked, nonzero, worst, worst < 1.0 ? "OK" : "FAIL");
    return worst < 1.0 ? 0 : 1;
}

} // namespace nn
