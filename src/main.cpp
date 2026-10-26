#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX            // keep windows.h from defining min()/max() macros
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif
#include "trainer.hpp"
#include "image.hpp"
#include "text.hpp"
#include "lm.hpp"
#include "selftest.hpp"
#include "command.hpp"
#include <iostream>
#include <iomanip>

using namespace nn;

static std::string get(const Args& a, const std::string& k, const std::string& d) {
    auto it = a.find(k); return it == a.end() ? d : it->second;
}
static std::vector<int> parseList(const std::string& s) {
    std::vector<int> v; std::stringstream ss(s); std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(std::stoi(t));
    return v;
}
static Act parseAct(const std::string& s) {
    return s == "tanh" ? Act::Tanh : s == "sigmoid" ? Act::Sigmoid : s == "leaky" ? Act::LeakyReLU : Act::ReLU;
}

static CommandRegistry& registry();   // defined below; the single list of commands

static void usage() {
    std::cout << "minnn - a from-scratch neural network in C++17  (tabular / images / text / language model)\n\nCOMMANDS\n"
              << registry().listing() << "\n";
    std::cout <<
R"(COMMON OPTIONS
  --epochs N  --batch N  --lr X  --lr-decay X  --wd X  --opt adam|sgd
  --act relu|leaky|tanh|sigmoid  --dropout P  --hidden 128,64
  --val X (0.2)  --patience N  --seed N  --out FILE (model.bin)

TABULAR   (--label-col -1 | name  --task classification|regression)
  CSV: numeric feature columns + 1 label column. Header auto-detected.

IMAGES    folder layout:  data/<class_name>/*.png|jpg|bmp|...
  --size 28 | 64x64     resize target (default 28)      --channels 1|3   grayscale / color
  --fit stretch|crop|pad  how to handle other aspect ratios (crop = center crop, pad = letterbox)
  --arch plain|resnet   plain: conv blocks + maxpool  |  resnet: residual stages + global avg pool (needs depth:
                        on a small benchmark a 2-stage ResNet scored far below the plain CNN; try --blocks 2 --conv 8,16,32)
  --conv 16,32,64       channels per block / per ResNet stage      --blocks N  residual blocks per stage
  --bn                  BatchNorm after every conv (for deeper nets; no gain measured on a 2-conv CNN)   --gap  global avg pool
  augmentation:  --augment (preset: shift/rotate/zoom/jitter)   or pick:  --hflip --vflip --rotate DEG
                 --zoom 0.9,1.1 --shift 0.1 --jitter 0.2 --noise 0.02 --cutout 0.3
  --idx-images F --idx-labels F [--limit N]   MNIST-format files
  predict: --data image.png | folder/   [--top 3] [--tta]  (tta = also look at the mirrored image)

TEXT      CSV with header (text,label) or folder layout data/<class>/*.txt
  --text-col 0|name --label-col -1|name --header 1      (columns: number, or header name)
  --max-len 200  --max-vocab 20000  --min-count 2  --ngrams 1|2  --embed 64
  predict: --text "some sentence"  |  --data lines.txt (one text per line)

LANGUAGE MODEL (GPT-style Transformer, byte-level BPE, works with Persian/English/any UTF-8)
  train-lm data:   --chat pairs.csv [--prompt-col 0|name --response-col 1|name --header 1 --train-on-prompt]
                   --text file.txt | folder/ | docs.csv [--text-col name]    (a .csv: one document per row)
                   --text file.txt|folder/     (can be combined with --chat)
  size:            --preset tiny|small|medium|large|xl   or  --ctx --dim --layers --heads --merges --dropout
  training:        --steps N --batch N --accum K (effective batch = batch*K) --lr X --warmup N
                   --eval-every N --clip X --wd X --out lm.bin --resume lm.bin (continue / fine-tune)
  chat/generate:   --temp 0.8 --top-k 40 --top-p 0.95 --rep-penalty 1.0 --max-new 150
  scaling:         info (estimate) -> train small -> grow (more layers / longer ctx) -> train-lm --resume
  --checkpoint          (train / train-image / train-text) write the best model to --out after every improvement
  Ctrl+C while training saves <out>.last (+ .opt); continue with --resume <out>.last
  tip: set OMP_NUM_THREADS=<cores> to control CPU threads.
)";
}

// ---------------- shared helpers ----------------
static TrainOptions trainOpts(const Args& a, int epochs, size_t batch, float lr, int patience) {
    TrainOptions o;
    o.epochs = std::stoi(get(a, "epochs", std::to_string(epochs)));
    o.batch = (size_t)std::stoul(get(a, "batch", std::to_string(batch)));
    o.lr = std::stof(get(a, "lr", std::to_string(lr)));
    o.lrDecay = std::stof(get(a, "lr-decay", "1.0"));
    o.weightDecay = std::stof(get(a, "wd", "0"));
    o.useAdam = get(a, "opt", "adam") != "sgd";
    o.patience = std::stoi(get(a, "patience", std::to_string(patience)));
    o.seed = (unsigned)std::stoul(get(a, "seed", "42"));
    return o;
}

static void runFit(Model& model, const Dataset& train, const Dataset& val, const Preprocessor& prep,
                   const Args& a, const TrainOptions& o) {
    std::printf("model: %zu parameters | train %zu | val %zu\n", model.paramCount(), train.size(), val.size());
    const std::string out = get(a, "out", "model.bin");
    Trainer trainer(model, prep, o);
    if (a.count("checkpoint")) trainer.addCallback(std::make_shared<CheckpointSaver>(out, prep));   // keep best model on disk while training
    trainer.fit(train, val);
    EvalResult tr = evaluate(model, train, prep), va = evaluate(model, val, prep);
    const char* mn = model.cfg.task == Task::Classification ? "accuracy" : "RMSE";
    std::printf("final | train %s %.4f | val %s %.4f\n", mn, tr.metric, mn, va.metric);
    saveModel(out, model, prep);
    std::printf("saved model -> %s\n", out.c_str());
}

static void printTop(const Matrix& probs, size_t row, const Preprocessor& prep, int topk) {
    std::vector<size_t> idx(probs.cols); std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t x, size_t y) { return probs(row, x) > probs(row, y); });
    for (int k = 0; k < topk && k < (int)idx.size(); ++k)
        std::cout << (k ? " | " : "") << prep.classes[idx[k]] << " " << std::fixed << std::setprecision(1)
                  << 100.f * probs(row, idx[k]) << "%";
    std::cout << "\n";
}

// ---------------- tabular ----------------
static int cmdTrain(const Args& a) {
    Task task = get(a, "task", "classification") == "regression" ? Task::Regression : Task::Classification;
    Preprocessor prep;
    Dataset all = loadLabeled(get(a, "data", ""), columnIndex(get(a, "data", ""), get(a, "label-col", "-1"), -1), task, prep);
    std::printf("loaded %zu samples, %zu features", all.size(), all.X.cols);
    if (task == Task::Classification) std::printf(", %zu classes", prep.classes.size());
    std::printf("\n");

    unsigned seed = (unsigned)std::stoul(get(a, "seed", "42"));
    Dataset train, val;
    trainValSplit(all, std::stof(get(a, "val", "0.2")), seed, train, val);
    prep.fit(train.X);
    prep.apply(train.X); prep.apply(val.X);
    if (task == Task::Regression) {
        double m = 0; for (float v : train.y) m += v; m /= train.y.size();
        double s = 0; for (float v : train.y) s += (v - m) * (v - m);
        s = std::sqrt(s / train.y.size());
        prep.yMean = (float)m; prep.yStd = s < 1e-8 ? 1.f : (float)s;
    }
    ModelConfig cfg;
    cfg.kind = Kind::MLP;
    cfg.inputDim = (int)all.X.cols;
    cfg.outputDim = task == Task::Classification ? (int)prep.classes.size() : 1;
    cfg.hidden = parseList(get(a, "hidden", "64,32"));
    cfg.act = parseAct(get(a, "act", "relu"));
    cfg.dropout = std::stof(get(a, "dropout", "0"));
    cfg.task = task; cfg.seed = seed;
    Model model(cfg);
    runFit(model, train, val, prep, a, trainOpts(a, 100, 32, 1e-3f, 20));
    return 0;
}

// ---------------- images ----------------
static void parseSize(const std::string& s, int& H, int& W) {
    size_t x = s.find('x');
    if (x == std::string::npos) H = W = std::stoi(s);
    else { H = std::stoi(s.substr(0, x)); W = std::stoi(s.substr(x + 1)); }
}

// Image classifier training, as an object: each phase is a small method that can be read / tested on its own.
class TrainImageCommand : public Command {
    Preprocessor prep_;
    FitMode fit_ = FitMode::Stretch;
    int C_ = 1, H_ = 0, W_ = 0;

    std::unique_ptr<ImageSource> makeSource(const Args& a) {
        fit_ = parseFit(get(a, "fit", "stretch"));
        if (a.count("idx-images"))
            return std::make_unique<IdxImageSource>(a.at("idx-images"), get(a, "idx-labels", ""), (size_t)std::stoul(get(a, "limit", "0")));
        C_ = std::stoi(get(a, "channels", "1"));
        parseSize(get(a, "size", "28"), H_, W_);
        return std::make_unique<FolderImageSource>(get(a, "data", ""), C_, H_, W_, fit_);
    }
    ModelConfig makeConfig(const Args& a, unsigned seed) const {
        ModelConfig cfg;
        cfg.kind = Kind::CNN; cfg.imgC = C_; cfg.imgH = H_; cfg.imgW = W_;
        cfg.inputDim = C_ * H_ * W_; cfg.outputDim = (int)prep_.classes.size();
        cfg.arch = get(a, "arch", "plain") == "resnet" ? Arch::ResNet : Arch::Plain;
        cfg.conv = parseList(get(a, "conv", cfg.arch == Arch::ResNet ? "16,32,64" : "16,32"));
        cfg.hidden = parseList(get(a, "hidden", cfg.arch == Arch::ResNet ? "" : "128"));
        cfg.batchNorm = a.count("bn") > 0 || cfg.arch == Arch::ResNet;
        cfg.globalPool = a.count("gap") > 0 || cfg.arch == Arch::ResNet;
        cfg.blocks = std::stoi(get(a, "blocks", "1"));
        cfg.fit = (int)fit_;
        cfg.act = parseAct(get(a, "act", "relu"));
        cfg.dropout = std::stof(get(a, "dropout", "0"));
        cfg.task = Task::Classification; cfg.seed = seed;
        return cfg;
    }
public:
    std::string name() const override { return "train-image"; }
    std::string summary() const override { return "--data folder/ | --idx-images F --idx-labels F      image classifier (CNN / ResNet)"; }
    int run(const Args& a) override {
        auto source = makeSource(a);
        std::printf("loading images: %s (fit: %s) ...\n", source->describe().c_str(), get(a, "fit", "stretch").c_str());
        Dataset all = source->load(prep_);
        C_ = source->channels(); H_ = source->height(); W_ = source->width();     // IDX files carry their own size
        std::printf("loaded %zu images (%dx%dx%d), %zu classes\n", all.size(), C_, H_, W_, prep_.classes.size());

        const unsigned seed = (unsigned)std::stoul(get(a, "seed", "42"));
        Dataset train, val;
        trainValSplit(all, std::stof(get(a, "val", "0.2")), seed, train, val);
        prep_.fitChannels(train.X, C_);                 // per-channel mean/std from the training split only
        prep_.apply(train.X); prep_.apply(val.X);

        Model model(makeConfig(a, seed));
        std::printf("%s", model.summary().c_str());
        TrainOptions o = trainOpts(a, 30, 64, 1e-3f, 8);
        o.augmenter = buildAugmenter(a, C_, H_, W_);
        if (o.augmenter) std::printf("augmentation: %s\n", o.augmenter->describe().c_str());
        runFit(model, train, val, prep_, a, o);
        return 0;
    }
};

// ---------------- text ----------------
static int cmdTrainText(const Args& a) {
    std::string path = get(a, "data", "");
    TextData td = std::filesystem::is_directory(path)
        ? loadTextFolder(path)
        : loadTextCSV(path, columnIndex(path, get(a, "text-col", "0"), 0), columnIndex(path, get(a, "label-col", "-1"), -1),
                      get(a, "header", "1") != "0");
    Preprocessor prep;
    std::vector<int> cls = mapClasses(td.labels, prep);
    std::printf("loaded %zu documents, %zu classes\n", td.texts.size(), prep.classes.size());

    const int maxLen = std::stoi(get(a, "max-len", "200")), ngrams = std::stoi(get(a, "ngrams", "1"));
    unsigned seed = (unsigned)std::stoul(get(a, "seed", "42"));
    std::vector<size_t> ti, vi;
    splitIndices(td.texts.size(), std::stof(get(a, "val", "0.2")), seed, ti, vi);
    prep.vocab = buildVocab(td.texts, ti, maxLen, ngrams, (size_t)std::stoul(get(a, "max-vocab", "20000")),
                            std::stoi(get(a, "min-count", "2")));
    std::printf("vocabulary: %zu tokens\n", prep.vocab.size());
    if (prep.vocab.empty()) throw std::runtime_error("empty vocabulary (try --min-count 1)");

    auto vm = vocabMap(prep.vocab);
    const size_t width = textWidth(maxLen, ngrams);
    Dataset all; all.X = Matrix(td.texts.size(), width); all.cls = cls;
    for (size_t i = 0; i < td.texts.size(); ++i)
        encodeText(td.texts[i], vm, maxLen, ngrams, &all.X.data[i * width]);
    Dataset train = subset(all, ti), val = subset(all, vi);

    ModelConfig cfg;
    cfg.kind = Kind::Text; cfg.inputDim = (int)width; cfg.outputDim = (int)prep.classes.size();
    cfg.vocab = (int)prep.vocab.size(); cfg.embed = std::stoi(get(a, "embed", "64"));
    cfg.maxLen = maxLen; cfg.ngrams = ngrams;
    cfg.hidden = parseList(get(a, "hidden", "64"));
    cfg.act = parseAct(get(a, "act", "relu"));
    cfg.dropout = std::stof(get(a, "dropout", "0"));
    cfg.task = Task::Classification; cfg.seed = seed;
    Model model(cfg);
    runFit(model, train, val, prep, a, trainOpts(a, 60, 32, 5e-3f, 10));
    return 0;
}

// ---------------- predict ----------------
static int cmdPredict(const Args& a) {
    Model model; Preprocessor prep;
    loadModel(get(a, "model", "model.bin"), model, prep);
    const ModelConfig& c = model.cfg;
    const int topk = std::stoi(get(a, "top", "1"));
    std::cout << std::fixed << std::setprecision(4);

    if (c.kind == Kind::MLP) {
        bool hasLabel = a.count("label-col") > 0;
        Matrix X = loadFeatures(get(a, "data", ""), hasLabel ? columnIndex(get(a, "data", ""), a.at("label-col"), 0) : 0, hasLabel);
        if ((int)X.cols != c.inputDim) throw std::runtime_error("feature count mismatch: model expects " + std::to_string(c.inputDim));
        prep.apply(X);
        Matrix out = model.forward(X, false);
        if (c.task == Task::Classification) {
            Matrix p = softmax(out);
            for (size_t i = 0; i < p.rows; ++i) printTop(p, i, prep, topk);
        } else {
            for (size_t i = 0; i < out.rows; ++i) std::cout << out(i, 0) * prep.yStd + prep.yMean << "\n";
        }
    } else if (c.kind == Kind::CNN) {
        std::string path = get(a, "data", "");
        std::vector<std::string> files;
        if (std::filesystem::is_directory(path)) {
            for (auto& e : std::filesystem::recursive_directory_iterator(path))
                if (e.is_regular_file() && isImageFile(e.path())) files.push_back(e.path().string());
            std::sort(files.begin(), files.end());
        } else files.push_back(path);
        const size_t dim = (size_t)c.imgC * c.imgH * c.imgW;
        const bool tta = a.count("tta") > 0;           // test-time augmentation: average original + mirrored image
        for (auto& f : files) {
            Image img;
            if (!Image::load(f, c.imgC, img)) { std::cout << f << ": cannot read\n"; continue; }
            Image fit = img.fitted(c.imgH, c.imgW, (FitMode)c.fit);
            Matrix X(tta ? 2 : 1, dim);
            fit.toRow(&X.data[0]);
            if (tta) fit.flippedH().toRow(&X.data[dim]);
            prep.apply(X);
            Matrix p = softmax(model.forward(X, false));
            if (tta) for (size_t j = 0; j < p.cols; ++j) p(0, j) = 0.5f * (p(0, j) + p(1, j));
            std::cout << f << ": "; printTop(p, 0, prep, topk);
        }
    } else {
        std::vector<std::string> texts;
        if (a.count("text")) texts.push_back(a.at("text"));
        else {
            std::ifstream f(get(a, "data", "")); std::string line;
            if (!f) throw std::runtime_error("use --text \"...\" or --data file.txt");
            while (std::getline(f, line)) if (!trim(line).empty()) texts.push_back(line);
        }
        auto vm = vocabMap(prep.vocab);
        const size_t width = textWidth(c.maxLen, c.ngrams);
        Matrix X(texts.size(), width);
        for (size_t i = 0; i < texts.size(); ++i) encodeText(texts[i], vm, c.maxLen, c.ngrams, &X.data[i * width]);
        Matrix p = softmax(model.forward(X, false));
        for (size_t i = 0; i < texts.size(); ++i) {
            std::string t = texts[i].size() > 50 ? texts[i].substr(0, 50) + "..." : texts[i];
            std::cout << t << "  ->  "; printTop(p, i, prep, topk);
        }
    }
    return 0;
}

// ---------------- demo ----------------
static void writeSpiral(const std::string& path) {
    std::ofstream f(path);
    std::mt19937 rng(7); std::normal_distribution<float> nz(0.f, 0.15f);
    f << "x,y,label\n";
    const int N = 300, K = 3;
    for (int k = 0; k < K; ++k)
        for (int i = 0; i < N; ++i) {
            float r = (float)i / N, t = k * 4.f + 4.f * r + nz(rng);
            f << r * std::sin(t) << "," << r * std::cos(t) << "," << "class" << k << "\n";
        }
}

static CommandRegistry& registry() {
    static CommandRegistry reg;
    static bool built = false;
    if (built) return reg;
    built = true;
    reg.add(std::make_unique<FnCommand>("demo", "train on a generated 3-class spiral (quick smoke test)",
        [](const Args&) {
            writeSpiral("spiral.csv");
            std::cout << "wrote spiral.csv (3-class spiral), training...\n";
            return cmdTrain({{"data", "spiral.csv"}, {"hidden", "64,64"}, {"epochs", "150"}, {"act", "tanh"}, {"out", "spiral_model.bin"}});
        }, false));
    reg.add(std::make_unique<FnCommand>("train", "--data data.csv                    tabular CSV (classification / regression)", cmdTrain));
    reg.add(std::make_unique<TrainImageCommand>());
    reg.add(std::make_unique<FnCommand>("train-text", "--data file.csv | folder/        text classifier", cmdTrainText));
    reg.add(std::make_unique<FnCommand>("predict", "--model model.bin ...             works for every model type", cmdPredict));
    reg.add(std::make_unique<FnCommand>("train-lm", "--chat pairs.csv | --text file_or_dir   language model (GPT) from scratch", cmdTrainLM));
    reg.add(std::make_unique<FnCommand>("chat", "--model lm.bin                    talk to your language model", cmdChat));
    reg.add(std::make_unique<FnCommand>("generate", "--model lm.bin --prompt \"...\"     continue a text", cmdGenerate));
    reg.add(std::make_unique<FnCommand>("info", "--preset large | --model m.bin   params / RAM / time estimate", cmdInfo));
    reg.add(std::make_unique<FnCommand>("grow", "--model lm.bin --layers 12 --ctx 512   small -> larger model", cmdGrow));
    reg.add(std::make_unique<FnCommand>("selftest", "verify every component", [](const Args&) { return runSelfTest(); }, false));
    reg.add(std::make_unique<FnCommand>("gradcheck-lm", "finite-difference check of the GPT backward pass", [](const Args&) { return cmdGradCheck(); }, false));
    return reg;
}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);   // Persian / UTF-8 text in the console (chat, generate)
    SetConsoleCP(CP_UTF8);
#endif
    try {
        int rc = registry().dispatch(argc, argv);
        if (rc >= 0) return rc;
        usage(); return 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n"; return 1;
    }
}
