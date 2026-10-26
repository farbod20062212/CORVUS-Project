#pragma once
// Command-line framework: every CLI command is an object registered in a CommandRegistry.
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <stdexcept>
#include <algorithm>

namespace nn {

using Args = std::map<std::string, std::string>;

// "--key value" and bare "--flag" (-> "1"). Throws on anything that is not an option.
inline Args parseArgs(int argc, char** argv, int from) {
    Args a;
    for (int i = from; i < argc; ++i) {
        std::string k = argv[i];
        if (k.rfind("--", 0) != 0) throw std::runtime_error("unexpected argument: " + k);
        k = k.substr(2);
        if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) a[k] = argv[++i];
        else a[k] = "1";
    }
    return a;
}

class Command {
public:
    virtual ~Command() = default;
    virtual std::string name() const = 0;
    virtual std::string summary() const = 0;          // one line shown in the usage text
    virtual bool takesOptions() const { return true; } // false: do not parse --options (demo, selftest)
    virtual int run(const Args& args) = 0;
};

// Adapter: wraps a plain function as a Command (for simple commands that need no state of their own).
class FnCommand : public Command {
    std::string name_, summary_; std::function<int(const Args&)> fn_; bool options_;
public:
    FnCommand(std::string name, std::string summary, std::function<int(const Args&)> fn, bool takesOptions = true)
        : name_(std::move(name)), summary_(std::move(summary)), fn_(std::move(fn)), options_(takesOptions) {}
    std::string name() const override { return name_; }
    std::string summary() const override { return summary_; }
    bool takesOptions() const override { return options_; }
    int run(const Args& a) override { return fn_(a); }
};

class CommandRegistry {
    std::vector<std::unique_ptr<Command>> cmds_;
public:
    CommandRegistry& add(std::unique_ptr<Command> c) {
        if (find(c->name())) throw std::runtime_error("duplicate command: " + c->name());
        cmds_.push_back(std::move(c)); return *this;
    }
    Command* find(const std::string& name) const {
        for (auto& c : cmds_) if (c->name() == name) return c.get();
        return nullptr;
    }
    const std::vector<std::unique_ptr<Command>>& all() const { return cmds_; }

    // Returns the command's exit code, or -1 when the command name is unknown.
    int dispatch(int argc, char** argv) const {
        if (argc < 2) return -1;
        Command* c = find(argv[1]);
        if (!c) return -1;
        Args args;
        if (c->takesOptions()) args = parseArgs(argc, argv, 2);
        return c->run(args);
    }
    std::string listing() const {
        size_t w = 0; for (auto& c : cmds_) w = std::max(w, c->name().size());
        std::string out;
        for (auto& c : cmds_) out += "  minnn " + c->name() + std::string(w - c->name().size() + 2, ' ') + c->summary() + "\n";
        return out;
    }
};

} // namespace nn
