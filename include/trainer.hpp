#pragma once
#include "model.hpp"
#include "optimizer.hpp"
#include "imageproc.hpp"
#include <chrono>
#include <cstdio>

namespace nn {

struct TrainOptions {
    int epochs = 100;
    size_t batch = 32;
    float lr = 1e-3f;
    float lrDecay = 1.0f;       // multiply lr by this every epoch
    float weightDecay = 0.f;
    bool useAdam = true;
    int patience = 20;          // early stopping (0 = off)
    unsigned seed = 1;
    std::shared_ptr<ImageAugmenter> augmenter;   // image models: augmentation pipeline (null = none)
    bool verbose = true;        // print progress lines
};

struct EvalResult { float loss = 0.f; float metric = 0.f; };   // metric: accuracy or RMSE (original units)

inline EvalResult evaluate(Model& m, const Dataset& d, const Preprocessor& prep) {
    EvalResult r;
    if (d.size() == 0) return r;
    std::vector<size_t> idx(d.size()); std::iota(idx.begin(), idx.end(), 0);
    const Task task = m.cfg.task; const size_t C = (size_t)m.cfg.outputDim;
    std::unique_ptr<Loss> loss;
    if (task == Task::Classification) loss = std::make_unique<SoftmaxCrossEntropy>();
    else loss = std::make_unique<MSELoss>();
    double totLoss = 0, totMetric = 0;
    for (size_t b = 0; b < d.size(); b += 1024) {
        size_t e = std::min(d.size(), b + 1024);
        Matrix X = makeBatchX(d, idx, b, e);
        Matrix Y = makeBatchTargets(d, idx, b, e, task, C, prep);
        Matrix out = m.forward(X, false);
        totLoss += loss->forward(out, Y) * (double)(e - b);
        for (size_t i = 0; i < out.rows; ++i) {
            if (task == Task::Classification) {
                size_t best = 0;
                for (size_t j = 1; j < out.cols; ++j) if (out(i, j) > out(i, best)) best = j;
                if ((int)best == d.cls[b + i]) totMetric += 1.0;
            } else {
                double diff = (out(i, 0) - Y(i, 0)) * prep.yStd;
                totMetric += diff * diff;
            }
        }
    }
    r.loss = (float)(totLoss / d.size());
    r.metric = task == Task::Classification ? (float)(totMetric / d.size()) : (float)std::sqrt(totMetric / d.size());
    return r;
}

// ---------------------------------------------------------------------------
// Training as objects.  Trainer runs the loop; behaviour is plugged in through TrainCallback observers
// (progress logging, early stopping, checkpointing, or your own).  Callbacks run in registration order.
// ---------------------------------------------------------------------------
struct EpochStats {
    int epoch = 0, totalEpochs = 0;
    float trainLoss = 0.f, valLoss = 0.f, valMetric = 0.f, lr = 0.f;
    double seconds = 0.0;
    bool classification = true;
};

class TrainCallback {
public:
    virtual ~TrainCallback() = default;
    virtual void onTrainBegin(Model&) {}
    virtual bool onEpochEnd(Model&, const EpochStats&) { return true; }   // return false to stop training
    virtual void onTrainEnd(Model&) {}
};

// Prints epoch 1, every 5th epoch and the last one.
class ProgressLogger : public TrainCallback {
public:
    bool onEpochEnd(Model&, const EpochStats& s) override {
        if (s.epoch == 1 || s.epoch % 5 == 0 || s.epoch == s.totalEpochs)
            std::printf("epoch %4d | train loss %.4f | val loss %.4f | val %s %.4f | lr %.5f | %.1fs\n",
                        s.epoch, s.trainLoss, s.valLoss, s.classification ? "acc" : "rmse", s.valMetric, s.lr, s.seconds);
        return true;
    }
};

// Remembers the weights (incl. BatchNorm statistics) of the best validation loss, stops after `patience`
// epochs without improvement (0 = never stop) and restores the best weights when training ends.
class EarlyStopping : public TrainCallback {
    int patience_, bad_ = 0; bool verbose_;
    float best_ = 1e30f; std::vector<std::vector<float>> snap_;
public:
    explicit EarlyStopping(int patience, bool verbose = true) : patience_(patience), verbose_(verbose) {}
    float bestLoss() const { return best_; }
    void onTrainBegin(Model& m) override { best_ = 1e30f; bad_ = 0; snap_ = m.snapshot(); }
    bool onEpochEnd(Model& m, const EpochStats& s) override {
        if (s.valLoss < best_ - 1e-6f) { best_ = s.valLoss; snap_ = m.snapshot(); bad_ = 0; return true; }
        if (patience_ > 0 && ++bad_ >= patience_) {
            if (verbose_) std::printf("early stopping at epoch %d (best val loss %.4f)\n", s.epoch, best_);
            return false;
        }
        return true;
    }
    void onTrainEnd(Model& m) override { m.restore(snap_); }
};

// Writes the model file whenever the validation loss improves, so a crash / Ctrl+C keeps the best model.
class CheckpointSaver : public TrainCallback {
    std::string path_; const Preprocessor& prep_; float best_ = 1e30f; bool verbose_;
public:
    CheckpointSaver(std::string path, const Preprocessor& prep, bool verbose = true) : path_(std::move(path)), prep_(prep), verbose_(verbose) {}
    bool onEpochEnd(Model& m, const EpochStats& s) override {
        if (s.valLoss < best_ - 1e-6f) {
            best_ = s.valLoss; saveModel(path_, m, prep_);
            if (verbose_) std::printf("  checkpoint saved (epoch %d, val loss %.4f)\n", s.epoch, best_);
        }
        return true;
    }
};

class Trainer {
    Model& model_; const Preprocessor& prep_; TrainOptions opt_;
    std::vector<std::shared_ptr<TrainCallback>> callbacks_;
    std::shared_ptr<EarlyStopping> early_;

    std::unique_ptr<Loss> makeLoss() const {
        if (model_.cfg.task == Task::Classification) return std::make_unique<SoftmaxCrossEntropy>();
        return std::make_unique<MSELoss>();
    }
    std::unique_ptr<Optimizer> makeOptimizer() const {
        if (opt_.useAdam) return std::make_unique<Adam>(opt_.lr, opt_.weightDecay);
        return std::make_unique<SGD>(opt_.lr, 0.9f, opt_.weightDecay);
    }
public:
    // Installs the standard behaviour first: progress logging (if verbose) and early stopping / best-weights restore.
    Trainer(Model& m, const Preprocessor& prep, TrainOptions o) : model_(m), prep_(prep), opt_(std::move(o)) {
        if (opt_.verbose) callbacks_.push_back(std::make_shared<ProgressLogger>());
        early_ = std::make_shared<EarlyStopping>(opt_.patience, opt_.verbose);
        callbacks_.push_back(early_);
    }
    Trainer& addCallback(std::shared_ptr<TrainCallback> cb) { callbacks_.push_back(std::move(cb)); return *this; }
    float bestValLoss() const { return early_->bestLoss(); }

    void fit(const Dataset& train, const Dataset& val) {
        const Task task = model_.cfg.task; const size_t C = (size_t)model_.cfg.outputDim;
        auto loss = makeLoss(); auto optim = makeOptimizer();
        auto params = model_.params();
        std::vector<size_t> idx(train.size()); std::iota(idx.begin(), idx.end(), 0);
        std::mt19937 rng(opt_.seed);
        const auto t0 = std::chrono::steady_clock::now();
        for (auto& cb : callbacks_) cb->onTrainBegin(model_);

        for (int ep = 1; ep <= opt_.epochs; ++ep) {
            std::shuffle(idx.begin(), idx.end(), rng);
            double sum = 0;
            for (size_t b = 0; b < train.size(); b += opt_.batch) {
                size_t e = std::min(train.size(), b + opt_.batch);
                Matrix X = makeBatchX(train, idx, b, e);
                if (opt_.augmenter && model_.cfg.kind == Kind::CNN) opt_.augmenter->apply(X, prep_, rng);
                Matrix Y = makeBatchTargets(train, idx, b, e, task, C, prep_);
                Matrix out = model_.forward(X, true);
                sum += loss->forward(out, Y) * (double)(e - b);
                model_.backward(loss->backward());
                optim->step(params);
            }
            EvalResult v = val.size() ? evaluate(model_, val, prep_) : evaluate(model_, train, prep_);
            EpochStats st; st.epoch = ep; st.totalEpochs = opt_.epochs; st.trainLoss = (float)(sum / train.size());
            st.valLoss = v.loss; st.valMetric = v.metric; st.lr = optim->lr; st.classification = task == Task::Classification;
            st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            bool keepGoing = true;
            for (auto& cb : callbacks_) keepGoing = cb->onEpochEnd(model_, st) && keepGoing;   // every observer sees the epoch
            if (!keepGoing) break;
            optim->lr *= opt_.lrDecay;
        }
        for (auto& cb : callbacks_) cb->onTrainEnd(model_);
    }
};

// Convenience wrapper with the original signature.
inline void fit(Model& m, const Dataset& train, const Dataset& val, const Preprocessor& prep, const TrainOptions& o) {
    Trainer(m, prep, o).fit(train, val);
}

} // namespace nn
