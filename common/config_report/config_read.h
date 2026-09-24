/*
 * common/config_report/config_read.h — rc::cfg::Reader, the one way an agent reads its config.
 *
 * This is the half that knows about ConfigLoader; config_report.h stays stdlib-only so it builds
 * under common/run_tests.sh's bare g++ and links from controller/tools/. Every read through a Reader
 * registers the key with its CODE DEFAULT, its effective value, its type and a one-line description
 * written at the read site — which is the whole point: the description cannot drift from the code,
 * and it exists for keys the config file never mentions (375 of 2033 fleet-wide).
 *
 * It is a near-drop-in for the three dialects in the tree:
 *   A  getf/geti/gets/getb(k, def)                 ->  cfg.f(k, def, "what")     (12 agents)
 *   B  ConfigLoaderUtils::load_optional<T,L>(cl,k,tgt) -> cfg.opt<T,L>(k, tgt, "what")   (room/retina/robot)
 *   C  genericworker load_optional / load_optional_cast -> the same opt()          (controller)
 *   E  exists(k) ? get<T>(k) : def  in a member-init list -> cfg.i(k, def, "what")  (presence, 14 agents)
 *
 * opt() takes the code default OUT OF THE TARGET before assigning. That is not a trick: in dialects
 * B/C/E the default already lives in the struct's member initialiser, which is the right place, and
 * the wrapper reads it for free.
 *
 * It also absorbs controller/generated/genericworker.h:86's report_config_miss — the SHADOWED
 * (a dotted key written under a [Section] header) and TYPE-MISMATCH diagnostics that exactly one of
 * twenty agents had. Both now land in the table as Origin::Shadowed / Origin::TypeError rather than
 * only on stderr, because a key that is PRESENT and still not in force is the failure that looks
 * most like success.
 */

#pragma once

#include <optional>
#include <type_traits>
#include <string>
#include <string_view>
#include <vector>

#include <ConfigLoader/ConfigLoader.h>

#include "config_report.h"

namespace rc::cfg
{

namespace detail
{
template <class T>
std::string str(const T& v)
{
    if constexpr (std::is_same_v<T, bool>) return v ? "true" : "false";
    else if constexpr (std::is_same_v<T, std::string>) return v;
    else if constexpr (std::is_floating_point_v<T>) return std::format("{:.6g}", v);
    else if constexpr (std::is_arithmetic_v<T>) return std::format("{}", v);
    else
    {
        std::string out = "[";
        for (const auto& e : v) out += (out.size() > 1 ? "," : "") + str(e);
        return out + "]";
    }
}

template <class T>
constexpr std::string_view type_name()
{
    if constexpr (std::is_same_v<T, bool>) return "bool";
    else if constexpr (std::is_same_v<T, std::string>) return "str";
    else if constexpr (std::is_floating_point_v<T>) return "float";
    else if constexpr (std::is_integral_v<T>) return "int";
    else return "list";
}

// The value as the loader holds it, for the unread rows — those have no read site to type them.
//
// ⚠ It probes the typed get<T> in turn rather than calling ConfigLoader's variant-returning
// `ConfigTypes get(const std::string&) const`: that overload is DECLARED at ConfigLoader.h:84 and
// DEFINED NOWHERE in robocomp_core, so every caller of it is a link error waiting to happen.
// Order matters — ConfigLoader.tpp widens int to double on purpose (so `Period = 25` satisfies a
// double key), which means get<double> succeeds on an int and would mistype every integer here.
inline std::string probe_value(const ConfigLoader& cl, const std::string& k)
{
    const auto attempt = [&]<class T>(std::type_identity<T>) -> std::optional<std::string>
    {
        try { return str(cl.get<T>(k)); } catch (...) { return std::nullopt; }
    };
    if (auto v = attempt(std::type_identity<bool>{})) return *v;
    if (auto v = attempt(std::type_identity<int>{})) return *v;
    if (auto v = attempt(std::type_identity<double>{})) return *v;
    if (auto v = attempt(std::type_identity<std::string>{})) return *v;
    if (auto v = attempt(std::type_identity<std::vector<int>>{})) return *v;
    if (auto v = attempt(std::type_identity<std::vector<double>>{})) return *v;
    if (auto v = attempt(std::type_identity<std::vector<std::string>>{})) return *v;
    if (auto v = attempt(std::type_identity<std::vector<bool>>{})) return *v;
    return "(unreadable)";
}
}  // namespace detail

class Reader
{
public:
    Reader(const ConfigLoader& cl, std::string_view agent, Registry& reg = registry())
        : cl_(&cl), reg_(&reg)
    {
        reg.set_agent(agent);
    }

    // Reads and registers NOTHING. agent_presence_monitor.cpp:697 loads a PEER's config file into a
    // second ConfigLoader; without this, that peer's keys land in THIS agent's table and the whole
    // instrument starts lying about which process is running what. Offline tools want it too.
    static Reader scratch(const ConfigLoader& cl) { return Reader(cl, nullptr); }

    // ── dialect A: the default arrives as an argument, the value comes back ──────────────────────
    float f(std::string_view k, float def, std::string_view what, Opts o = {}) const
    { return get<float, double>(k, def, what, o); }
    double d(std::string_view k, double def, std::string_view what, Opts o = {}) const
    { return get<double, double>(k, def, what, o); }
    int i(std::string_view k, int def, std::string_view what, Opts o = {}) const
    { return get<int, int>(k, def, what, o); }
    bool b(std::string_view k, bool def, std::string_view what, Opts o = {}) const
    { return get<bool, bool>(k, def, what, o); }
    std::string s(std::string_view k, std::string def, std::string_view what, Opts o = {}) const
    { return get<std::string, std::string>(k, std::move(def), what, o); }

    // ⚠ ConfigLoader THROWS on an empty array, so an absent key is how a config says "leave it" —
    // `key = []` is not a way to say nothing (memory: configloader-empty-array-throws).
    template <class T, class LoadT = T>
    std::vector<T> v(std::string_view k, std::vector<T> def, std::string_view what, Opts o = {}) const
    { return get<std::vector<T>, std::vector<LoadT>>(k, std::move(def), what, o); }

    // ── dialect B/C/E: in-out target; the CODE DEFAULT is read out of the target ─────────────────
    template <class Target, class LoadT = Target>
    void opt(std::string_view k, Target& target, std::string_view what, Opts o = {}) const
    { target = get<Target, LoadT>(k, target, what, o); }

    // The genericworker dialect: load_optional_cast<ConfigType>(key, value) states the type to READ
    // and leaves the target's type to deduction. opt<Target,LoadT> wants them the other way round,
    // and the target's type is not recoverable from the call text - so this mirrors the old spelling
    // exactly and the migration stays a one-for-one substitution instead of a typing exercise.
    template <class LoadT, class Target>
    void opt_cast(std::string_view k, Target& target, std::string_view what, Opts o = {}) const
    { target = get<Target, LoadT>(k, target, what, o); }

    template <class Target, class LoadT = Target>
    void req(std::string_view k, Target& target, std::string_view what, Opts o = {}) const
    {
        if (not cl_->exists(k_(k)))
        {
            std::print("[cfg] ★ REQUIRED key '{}' is missing: {}\n", k, what);
            throw std::runtime_error("missing required config key: " + std::string(k));
        }
        opt<Target, LoadT>(k, target, what, o);
    }

    // ── presence, honestly reported ──────────────────────────────────────────────────────────────
    //
    // This is what deletes room_concept's sentinel workarounds (room_config.cpp:450-585): a NaN for
    // floats, and for int/bool the key read TWICE from opposite seeds so that only agreement proves
    // presence. All of that existed because load_optional returns void. `overlay_source` also marks
    // the key as PARSED, which is the half the apply_* side pairs with to catch the CmdNoiseRot
    // defect — a value read, printed, and applied by nothing.
    template <class T, class LoadT = T>
    [[nodiscard]] std::optional<T> maybe(std::string_view k, std::string_view what,
                                         std::string_view overlay_source = {}, Opts o = {}) const
    {
        if (not reg_) return read_raw<T, LoadT>(k);
        const bool present = cl_->exists(k_(k));
        const auto got = present ? read_raw<T, LoadT>(k) : std::nullopt;
        Record r;
        r.key = k;
        r.type = detail::type_name<T>();
        r.code_default = "(none)";
        r.effective = got ? detail::str(*got) : "(absent)";
        r.description = what;
        r.origin = got ? Origin::File : Origin::Default;
        r.kind = o.kind;
        r.mut = o.mut;
        r.overlay_source = overlay_source;
        reg_->note(std::move(r));
        if (got and not overlay_source.empty()) reg_->note_parsed(k);
        return got;
    }

    // Every key the file holds, as data — so config_report.h never has to include ConfigLoader.
    [[nodiscard]] std::vector<FileKey> file_keys() const
    {
        std::vector<FileKey> out;
        for (const auto key : cl_->getKeys())
        {
            std::string k{key};
            out.push_back({k, detail::probe_value(*cl_, k)});
        }
        return out;
    }

    Published publish(const std::string& dump_path = "etc/config_effective.csv") const
    { return reg_->publish(file_keys(), dump_path); }

    [[nodiscard]] const ConfigLoader& loader() const { return *cl_; }

private:
    explicit Reader(const ConfigLoader& cl, std::nullptr_t) : cl_(&cl), reg_(nullptr) {}

    static std::string k_(std::string_view k) { return std::string(k); }

    template <class T, class LoadT>
    std::optional<T> read_raw(std::string_view k) const
    {
        try { return static_cast<T>(cl_->get<LoadT>(k_(k))); }
        catch (...) { return std::nullopt; }
    }

    // A key that is ABSENT is the supported way to say "use the default" and stays quiet. A key that
    // is PRESENT but unusable does NOT: it looks configured and is not. Those are the two cases
    // report_config_miss separated, and they keep that separation here.
    template <class Target, class LoadT>
    Target get(std::string_view k, Target def, std::string_view what, Opts o) const
    {
        const std::string key = k_(k);
        Target value = def;
        Origin origin = Origin::Default;

        if (cl_->exists(key))
        {
            try
            {
                if constexpr (std::is_same_v<Target, LoadT>) value = cl_->get<LoadT>(key);
                else value = static_cast<Target>(cl_->get<LoadT>(key));
                origin = Origin::File;
            }
            catch (const std::exception& e)
            {
                origin = Origin::TypeError;
                std::print("[cfg] ★ key '{}' EXISTS but could not be read as {} — the built-in "
                           "default {} is in force. {}\n", k, detail::type_name<Target>(),
                           detail::str(def), e.what());
            }
        }
        else if (const auto shadow = shadowing_key(key); not shadow.empty())
        {
            origin = Origin::Shadowed;
            std::print("[cfg] ★ key '{}' NOT FOUND, but the file defines '{}' — a dotted key written "
                       "below a [Section] header is namespaced under it. The built-in default is in "
                       "force; move it above the first [Section] or give it its own header.\n",
                       k, shadow);
        }

        if (reg_)
        {
            Record r;
            r.key = key;
            r.type = detail::type_name<Target>();
            r.code_default = detail::str(def);
            r.effective = detail::str(value);
            r.description = what;
            r.origin = origin;
            r.kind = o.kind;
            r.mut = o.mut;
            reg_->note(std::move(r));
        }
        return value;
    }

    // "<Section>.<key>" when the agent asked for "<key>" — the shadowing case, from genericworker.h:86.
    [[nodiscard]] std::string shadowing_key(const std::string& key) const
    {
        for (const auto candidate : cl_->getKeys())
        {
            const std::string full{candidate};
            if (full.size() > key.size() + 1 and full.ends_with(key)
                and full[full.size() - key.size() - 1] == '.')
                return full;
        }
        return {};
    }

    const ConfigLoader* cl_;
    Registry* reg_;
};

}  // namespace rc::cfg
