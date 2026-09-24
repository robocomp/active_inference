/*
 * common/config_report/config_report.h — what this agent is ACTUALLY running, with provenance.
 *
 * WHY THIS EXISTS. Reading an agent's etc/config.toml does not tell you what it will do. Measured
 * across the fleet 2026-09-18: 375 of 2033 keys the code reads (18%) are absent from the agent's own
 * config file and run the code default with nothing on the page to say so — and 8 keys sit in a
 * config file that no source reads. Commit b6c40b4 put it exactly right: "Absent is not off, but it
 * reads as off." P3Bot ran six older solver defaults that way for months, and a table of key=value
 * would have printed all six and looked healthy.
 *
 * So the load-bearing column here is NOT the value, it is the PROVENANCE: code default / file /
 * which overlay. And the description lives at the READ SITE, not in the file: door_concept's config
 * carries four chair-shaped `Tracker.BirthSeat*` keys whose comment was rewritten to sound door-ish
 * and which door_concept reads nowhere — file prose can lie, a description next to the read cannot.
 *
 * STDLIB ONLY, ON PURPOSE. common/run_tests.sh compiles each *_test.cpp with a bare g++ — no toml++,
 * no $ROBOCOMP/classes. Everything that touches ConfigLoader lives in config_read.h instead, so this
 * half stays testable in seconds and linkable from controller/tools/, which builds without Qt/DSR/Ice.
 * publish() therefore takes the file's keys as DATA rather than reaching for a loader.
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <format>
#include <fstream>
#include <map>
#include <mutex>
#include <print>
#include <string>
#include <string_view>
#include <vector>

#include <sys/ioctl.h>
#include <unistd.h>

#include "../diag_log/rotating_csv.h"   // rc::diag::open_rotating — keeps the previous runs

namespace rc::cfg
{

// Where the effective value came from. Default/File/Overlay/Manifest answer "who won"; Shadowed and
// TypeError are the two ways a key can be PRESENT and still not take effect, which is the failure
// that looks most like success.
enum class Origin : std::uint8_t
{
    Default,     // no file key at all — the CODE default is in force. The b6c40b4 case.
    File,        // the file's value, at the read site's own key
    Overlay,     // a later [Platform.x] / [Scenario.y] overlay changed it
    Manifest,    // concept_manifest world fact won (see concept_manifest::resolve)
    Runtime,     // changed after startup; the tabled value is the STARTUP one
    Shadowed,    // absent here, but the file defines <Section>.<key> — default in force
    TypeError,   // present, wrong variant arm, get<> threw — default in force
    Unread,      // in the file, no read site asked for it. Produced by the sweep, not by a read.
};

// Experiment is not cosmetic: an A/B arm sitting at its DEFAULT is still a fact about the run, so it
// prints when nothing else at its default does. Diagnostic is excluded from the fingerprint — turning
// a CSV on must not change which arm a run is.
enum class Kind : std::uint8_t { Permanent, Experiment, Diagnostic, Deprecated };

// A self_tuner knob's startup value is not what ran, so it must never enter the fingerprint.
enum class Mutability : std::uint8_t { Startup, RuntimeTuned };

struct Opts
{
    Kind kind = Kind::Permanent;
    Mutability mut = Mutability::Startup;
};
inline constexpr Opts permanent{};
inline constexpr Opts experiment{Kind::Experiment};
inline constexpr Opts diagnostic{Kind::Diagnostic};
inline constexpr Opts deprecated{Kind::Deprecated};
inline constexpr Opts runtime_tuned{Kind::Permanent, Mutability::RuntimeTuned};

struct Record
{
    std::string key, type, code_default, effective, description, overlay_source;
    Origin origin = Origin::Default;
    Kind kind = Kind::Permanent;
    Mutability mut = Mutability::Startup;
    int reads = 0;              // >1 ⇒ two sites read the same key; worth knowing, not an error
    bool parsed = false;        // an overlay key whose value was successfully parsed
    bool apply_seen = false;    // an apply_* function acknowledged this key's destination
};

struct Published
{
    std::string fingerprint;    // 8 hex over the arm-relevant effective values
    std::string arm;            // the Experiment keys, human-readable
    int undocumented = 0;
    int unread = 0;
    bool sweep_armed = false;
};

constexpr std::string_view to_string(Origin o)
{
    switch (o)
    {
        case Origin::Default:   return "default";
        case Origin::File:      return "file";
        case Origin::Overlay:   return "overlay";
        case Origin::Manifest:  return "manifest";
        case Origin::Runtime:   return "runtime";
        case Origin::Shadowed:  return "shadowed";
        case Origin::TypeError: return "typeerr";
        case Origin::Unread:    return "unread";
    }
    return "?";
}

constexpr std::string_view to_string(Kind k)
{
    switch (k)
    {
        case Kind::Permanent:  return "permanent";
        case Kind::Experiment: return "experiment";
        case Kind::Diagnostic: return "diagnostic";
        case Kind::Deprecated: return "deprecated";
    }
    return "?";
}

// One file key as the loader saw it. Passed IN to publish() so this header never includes ConfigLoader.
struct FileKey
{
    std::string key, value;
};

class Registry
{
public:
    // ── filled by the read sites ──────────────────────────────────────────────────────────────
    void note(Record r)
    {
        const std::lock_guard lk(m_);
        auto& e = recs_[r.key];
        const int seen = e.reads;
        e = std::move(r);
        e.reads = seen + 1;
    }

    // ── filled afterwards, by whoever changes or consumes a value ─────────────────────────────

    // Called UNCONDITIONALLY by every apply_* set() lambda, whether or not the value changed.
    // Without that, an overlay value that happens to EQUAL its destination is indistinguishable from
    // one that was never applied — and "parsed but applied by nothing" is precisely the CmdNoiseRot
    // defect, so a false positive here would discredit the only check that can see it.
    void note_overlay_seen(std::string_view dest_key)
    {
        const std::lock_guard lk(m_);
        if (auto it = recs_.find(std::string(dest_key)); it != recs_.end())
            it->second.apply_seen = true;
        else
            pending_apply_seen_.emplace_back(dest_key);
    }

    void note_overlay(std::string_view dest_key, std::string_view value, std::string_view source)
    {
        const std::lock_guard lk(m_);
        auto& e = recs_[std::string(dest_key)];
        if (e.key.empty()) e.key = dest_key;
        e.effective = value;
        e.origin = Origin::Overlay;
        e.overlay_source = source;
        e.apply_seen = true;
    }

    // An overlay key that was PARSED (its own [Platform.x].Key read site). Marks the parse half of
    // the parsed/applied pair; the apply half arrives via note_overlay_seen on the DESTINATION key.
    void note_parsed(std::string_view overlay_key)
    {
        const std::lock_guard lk(m_);
        recs_[std::string(overlay_key)].parsed = true;
    }

    // A RuntimeTuned key is excluded from the fingerprint, so it must be DECLARED that way at its
    // read site. Discovering it here instead would quietly move the arm identity of a run that had
    // already published one — the instrument changing its own answer after the fact. Say so.
    void note_runtime(std::string_view key, std::string_view value)
    {
        const std::lock_guard lk(m_);
        auto& e = recs_[std::string(key)];
        if (e.mut != Mutability::RuntimeTuned)
            std::print("[cfg] ★ '{}' changed at runtime but its read site did not declare "
                       "rc::cfg::runtime_tuned — the startup table would assert a value that is "
                       "already stale. Declare it, or stop tuning it.\n", key);
        e.effective = value;
        e.origin = Origin::Runtime;
        e.mut = Mutability::RuntimeTuned;
    }

    // For consumers that read by ENUMERATION rather than by name (the Ice loop in generated/main.cpp).
    // Strictly better than exempting the prefix: it marks the keys actually forwarded, so a key that
    // fell into the catch(...) stays unmarked and surfaces — and forwarding that did NOT happen is
    // exactly the months-inert [omnirobot.ThreadPool] bug a prefix exemption would have hidden.
    void mark_consumed(std::string_view key, std::string_view by)
    {
        const std::lock_guard lk(m_);
        consumed_[std::string(key)] = by;
    }

    void exempt_prefix(std::string_view prefix, std::string_view why)
    {
        const std::lock_guard lk(m_);
        exempt_.emplace_back(std::string(prefix), std::string(why));
    }

    // A CLAIM: every key this agent reads goes through a Reader. It ARMS the unread sweep; until it
    // is made, publish() reports INCONCLUSIVE rather than an empty list. check_registry_complete.sh
    // is what stops the claim from rotting into a lie.
    void declare_complete(std::string_view agent)
    {
        const std::lock_guard lk(m_);
        agent_ = agent;
        complete_ = true;
    }

    void set_agent(std::string_view agent)
    {
        const std::lock_guard lk(m_);
        if (agent_.empty()) agent_ = agent;
    }

    [[nodiscard]] std::size_t size() const { const std::lock_guard lk(m_); return recs_.size(); }
    [[nodiscard]] bool has(std::string_view k) const
    {
        const std::lock_guard lk(m_);
        return recs_.contains(std::string(k));
    }
    [[nodiscard]] Record record(std::string_view k) const
    {
        const std::lock_guard lk(m_);
        const auto it = recs_.find(std::string(k));
        return it == recs_.end() ? Record{} : it->second;
    }

    // ── the one print point ───────────────────────────────────────────────────────────────────
    Published publish(const std::vector<FileKey>& file_keys, const std::string& dump_path);

    [[nodiscard]] std::string fingerprint() const;
    [[nodiscard]] std::string arm() const;

    // An instrument that can silently fail to exist is worse than none: say so at exit. Armed only
    // on the process-wide registry() — a scratch Registry (a tool, a test) must not write anything.
    void arm_exit_flush() { const std::lock_guard lk(m_); exit_flush_ = true; }
    ~Registry();

private:
    friend std::string run_stamp();
    void flush_locked(const std::vector<FileKey>& file_keys, const std::string& dump_path,
                      const Published& p, std::string_view stage) const;

    mutable std::mutex m_;
    std::map<std::string, Record> recs_;                        // ordered ⇒ a stable fingerprint
    std::map<std::string, std::string> consumed_;
    std::vector<std::pair<std::string, std::string>> exempt_;
    std::vector<std::string> pending_apply_seen_;
    std::string agent_, last_dump_, last_fp_, last_arm_;
    bool complete_ = false;
    bool published_ = false;
    bool exit_flush_ = false;
};

inline Registry& registry()
{
    // Registry holds a mutex, so it is neither copyable nor movable: arm it in place, once.
    static Registry r;
    [[maybe_unused]] static const bool armed = (r.arm_exit_flush(), true);
    return r;
}

// The seven prefixes consumed by GENERATED code. One call, so no agent invents its own list; an
// eighth is a design conversation, not a local edit. Every one of them is PRINTED with the number of
// keys it absorbed, so an exemption that swallows too much is visible rather than silent.
inline void exempt_generated_prefixes(Registry& reg = registry())
{
    reg.exempt_prefix("Agent.",     "generated/genericworker.cpp: agent name/id and the viewer flags");
    reg.exempt_prefix("Period.",    "generated/main.cpp: the compute/emergency periods");
    reg.exempt_prefix("Proxies.",   "generated Ice proxy wiring");
    reg.exempt_prefix("Endpoints.", "generated Ice endpoint wiring");
    reg.exempt_prefix("Ice.",       "forwarded to the Ice communicator by enumeration over getKeys()");
    reg.exempt_prefix("Owns.",      "owned-nodes bookkeeping");
    reg.exempt_prefix("Component",  "[Component.Debug] Verbose, read by generated code");
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
namespace detail
{
// ─── terminal styling ───────────────────────────────────────────────────────────────────────────
//
// Colour ONLY when stdout is a terminal. These agents are launched under journald, tee'd into log
// files and scraped by other tools; ANSI escapes there are noise that also breaks every grep written
// against the output. NO_COLOR (the de-facto standard) and TERM=dumb are honoured for the same
// reason. When colour is off the table still draws — the box rules carry the shape on their own.
inline bool colour_enabled()
{
    static const bool on = []
    {
        if (std::getenv("NO_COLOR") != nullptr) return false;
        const char* term = std::getenv("TERM");
        if (term != nullptr and std::string_view(term) == "dumb") return false;
        return ::isatty(STDOUT_FILENO) == 1;
    }();
    return on;
}

inline std::string paint(std::string_view sgr, std::string_view text)
{
    if (not colour_enabled()) return std::string(text);
    return std::format("\x1b[{}m{}\x1b[0m", sgr, text);
}

// How WIDE a string looks, not how many bytes it is. The box rules and the ★/↳/▌ glyphs are
// multi-byte UTF-8, so std::format's {:<N} — which pads by BYTES — would misalign every row that
// contains one. Counting non-continuation bytes is exact for everything this table prints (no
// double-width CJK), and it is the difference between a table and a staircase.
inline std::size_t vis_len(std::string_view s)
{
    std::size_t n = 0;
    for (const unsigned char c : s) if ((c & 0xC0) != 0x80) ++n;
    return n;
}

inline std::string pad(std::string_view s, std::size_t w)
{
    const auto len = vis_len(s);
    return std::string(s) + std::string(len < w ? w - len : 0, ' ');
}

inline std::string clip(std::string_view s, std::size_t w)
{
    if (vis_len(s) <= w) return std::string(s);
    std::string out;
    std::size_t n = 0;
    for (std::size_t i = 0; i < s.size(); ++i)
    {
        if ((static_cast<unsigned char>(s[i]) & 0xC0) != 0x80)
        {
            if (n + 1 > w - 1) break;
            ++n;
        }
        out += s[i];
    }
    return out + "…";
}

inline std::string rule(std::size_t n, std::string_view glyph = "─")
{
    std::string out;
    for (std::size_t i = 0; i < n; ++i) out += glyph;
    return out;
}

// Terminal width, clamped to something a table can live in. 100 when there is no terminal, so a log
// file gets a stable, diff-friendly shape rather than one that depends on who launched the agent.
inline std::size_t term_width()
{
    static const std::size_t w = []() -> std::size_t
    {
        if (not colour_enabled()) return 100;
        ::winsize ws{};
        if (::ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 and ws.ws_col > 0)
            return std::clamp<std::size_t>(ws.ws_col, 80, 150);
        return 100;
    }();
    return w;
}

// Each provenance gets its own colour, because provenance is the column that matters: a value that
// differs from its default is interesting, but WHERE it came from is what the table exists to say.
struct Look { std::string_view sgr, badge; };
inline Look look_of(Origin o)
{
    switch (o)
    {
        case Origin::Default:   return {"33",    "DEFAULT"};   // yellow  — the file is SILENT
        case Origin::File:      return {"32",    "file"};      // green   — the file said so
        case Origin::Overlay:   return {"36;1",  "OVERLAY"};   // cyan    — an overlay won
        case Origin::Manifest:  return {"34;1",  "manifest"};  // blue    — a world fact won
        case Origin::Runtime:   return {"35",    "runtime"};   // magenta — tuned after startup
        case Origin::Shadowed:  return {"33;1",  "SHADOWED"};  // bright  — present but namespaced away
        case Origin::TypeError: return {"31;1",  "TYPE ERR"};  // red     — present and unusable
        case Origin::Unread:    return {"31",    "unread"};
    }
    return {"0", "?"};
}

inline std::string fnv1a8(std::string_view s)
{
    std::uint64_t h = 1469598103934665603ull;
    for (const unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    return std::format("{:08x}", static_cast<std::uint32_t>(h ^ (h >> 32)));
}

inline std::string now_iso()
{
    const auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
    localtime_r(&t, &tm);
    return std::format("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}", tm.tm_year + 1900, tm.tm_mon + 1,
                       tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
}

// CSV cell: quote and double any embedded quote. Descriptions are free text written by humans.
inline std::string csv(std::string_view s)
{
    std::string out = "\"";
    for (const char c : s) { if (c == '"') out += '"'; out += (c == '\n' or c == '\r') ? ' ' : c; }
    return out + '"';
}
}  // namespace detail

inline std::string Registry::fingerprint() const
{
    std::string blob;
    for (const auto& [k, r] : recs_)
        if (r.kind != Kind::Diagnostic and r.mut == Mutability::Startup and r.origin != Origin::Unread)
            blob += k + '=' + r.effective + '\n';
    return detail::fnv1a8(blob);
}

inline std::string Registry::arm() const
{
    std::string out;
    for (const auto& [k, r] : recs_)
        if (r.kind == Kind::Experiment)
            out += (out.empty() ? "" : ",") + k + '=' + r.effective;
    return out.empty() ? "none" : out;
}

inline Published Registry::publish(const std::vector<FileKey>& file_keys, const std::string& dump_path)
{
    std::unique_lock lk(m_);

    for (const auto& k : pending_apply_seen_)
        if (auto it = recs_.find(k); it != recs_.end()) it->second.apply_seen = true;
    pending_apply_seen_.clear();

    // ── the unread sweep ──────────────────────────────────────────────────────────────────────
    std::vector<FileKey> unread;
    std::map<std::string, int> absorbed;
    for (const auto& [prefix, why] : exempt_) absorbed[prefix] = 0;
    for (const auto& fk : file_keys)
    {
        if (recs_.contains(fk.key) or consumed_.contains(fk.key)) continue;
        const auto hit = std::ranges::find_if(exempt_, [&](const auto& e)
                                              { return fk.key.starts_with(e.first); });
        if (hit != exempt_.end()) { ++absorbed[hit->first]; continue; }
        unread.push_back(fk);
    }

    Published p;
    p.fingerprint = fingerprint();
    p.arm = arm();
    p.sweep_armed = complete_;
    p.unread = static_cast<int>(unread.size());
    for (const auto& [k, r] : recs_)
        if (r.description.empty() and r.origin != Origin::Unread) ++p.undocumented;

    // ── console: a TABLE of the deltas only ────────────────────────────────────────────────────
    //
    // Most of the keys in the file restate their own default; printing those is the flood this is
    // meant to replace. A row appears when the value DIFFERS from the code default, when an overlay
    // moved it, when it is present-but-not-in-force (shadowed / type error), or when it is an A/B
    // arm — an arm sitting at its default is still a fact about the run.
    using detail::paint; using detail::pad; using detail::clip; using detail::rule;
    const auto label = agent_.empty() ? std::string("agent") : agent_;
    const std::size_t W = detail::term_width();
    const auto dim  = [](std::string_view t) { return paint("2", t); };
    const auto bold = [](std::string_view t) { return paint("1", t); };

    // ── header bar ──
    {
        const auto title = std::format("  {}  {}  ", "⚙", label);
        const auto right = std::format("  fp {}  ·  {} keys  ", p.fingerprint, recs_.size());
        const auto fill  = W > detail::vis_len(title) + detail::vis_len(right)
                         ? W - detail::vis_len(title) - detail::vis_len(right) : 1;
        std::print("\n{}\n", paint("44;97;1", title + std::string(fill, ' ') + right));
    }

    // ── column layout, sized to the terminal: the description gets whatever is left ──
    // The value column gets the room it needs before the key column does: a truncated VALUE is a
    // table that cannot answer its own question, while a truncated key is still recognisable.
    constexpr std::size_t kOrigin = 10, kKind = 5, kType = 6, kArrow = 5;
    const std::size_t kVal = W >= 120 ? 24 : (W >= 100 ? 20 : 14);
    const std::size_t kDef = kVal;
    const std::size_t fixed = 2 + kOrigin + kKind + 1 + kType + kVal + kArrow + kDef;
    const std::size_t kKey  = W > fixed + 20 ? std::min<std::size_t>(42, W - fixed - 2) : 20;

    std::print("{}\n", dim(std::format("  {}{}{}{}{}{}",
               pad("ORIGIN", kOrigin), pad("A/B", kKind), pad("KEY", kKey + 1),
               pad("TYPE", kType), pad("EFFECTIVE", kVal + kArrow), "DEFAULT")));
    std::print("{}\n", dim("  " + rule(W - 2)));

    int deltas = 0, overlays = 0;
    for (const auto& [k, r] : recs_)
    {
        const bool changed = r.effective != r.code_default;
        const bool notable = r.origin == Origin::Overlay or r.origin == Origin::Shadowed
                             or r.origin == Origin::TypeError or r.kind == Kind::Experiment
                             or r.kind == Kind::Deprecated;
        if (not changed and not notable) continue;
        ++deltas;
        if (r.origin == Origin::Overlay) ++overlays;

        const auto lk = detail::look_of(r.origin);
        // A coloured left bar makes provenance readable at a glance, before any word is parsed.
        const auto bar = paint(lk.sgr, "▌");
        const auto kind_chip = r.kind == Kind::Experiment  ? paint("35;1", pad("A/B", kKind))
                             : r.kind == Kind::Diagnostic  ? dim(pad("diag", kKind))
                             : r.kind == Kind::Deprecated  ? paint("31", pad("dep", kKind))
                                                           : pad("", kKind);
        std::print("{} {}{}{} {}{}{}{}\n",
                   bar,
                   paint(lk.sgr, pad(lk.badge, kOrigin)),
                   kind_chip,
                   bold(pad(clip(k, kKey), kKey + 1)),
                   dim(pad(r.type, kType)),
                   paint(changed ? "1" : "2", pad(clip(r.effective, kVal), kVal)),
                   // An UNCHANGED row (an A/B arm sitting at its default) prints "· at default"
                   // rather than "false <- false", which says the same thing twice and reads like a
                   // diff that is not one.
                   changed ? dim("  ←  ") + dim(clip(r.code_default, kDef))
                           : dim("  ·  at default"),
                   r.overlay_source.empty() ? "" : paint("36", "  [" + r.overlay_source + "]"));

        if (not r.description.empty())
            std::print("{}\n", dim(std::format("   {}  {}", "↳",
                                               clip(r.description, W > 8 ? W - 8 : 40))));
    }
    if (deltas == 0)
        std::print("{}\n", dim("  (every value is its code default)"));

    // ── the A/B arm strip: what THIS run is, at a glance ──
    {
        std::string strip;
        for (const auto& [k, r] : recs_)
            if (r.kind == Kind::Experiment)
            {
                const bool on = r.effective != "false" and r.effective != "0" and not r.effective.empty();
                const auto dot = on ? paint("32;1", "●") : paint("90", "○");
                auto name = k.substr(k.find('.') == std::string::npos ? 0 : k.find('.') + 1);
                strip += dot + " " + (on ? name : dim(name)) + "  ";
            }
        if (not strip.empty())
        {
            // Wrapped, because thirteen arms on one line is a paragraph, not a strip - and the point
            // of this row is that the run's identity is readable in one glance.
            std::print("\n  {}", paint("35;1", "A/B ARMS "));
            std::size_t col = 11;
            for (const auto& [k, r] : recs_)
            {
                if (r.kind != Kind::Experiment) continue;
                const bool on = r.effective != "false" and r.effective != "0" and not r.effective.empty();
                auto name = k.substr(k.find('.') == std::string::npos ? 0 : k.find('.') + 1);
                const auto cell = std::format("{} {}  ", on ? "●" : "○", name);
                if (col + detail::vis_len(cell) > W - 2) { std::print("\n{}", std::string(11, ' ')); col = 11; }
                std::print("{}", on ? paint("32;1", cell) : paint("90", cell));
                col += detail::vis_len(cell);
            }
            std::print("\n");
        }
    }

    // ── parsed-but-applied-by-nothing: the CmdNoiseRot class. NOT unread — it WAS read ──
    std::vector<std::string> inert;
    for (const auto& [k, r] : recs_)
        if (r.parsed and not r.apply_seen) inert.push_back(k);
    if (not inert.empty())
    {
        std::print("\n{}\n", paint("31;1", std::format("  ★ {} overlay key(s) PARSED and applied "
                                                       "by NOTHING:", inert.size())));
        for (const auto& k : inert) std::print("{}\n", paint("31", "      " + k));
        std::print("{}\n", dim("      the CmdNoiseRot / [*.ThreadPool] shape: read, printed, and it "
                               "reaches nothing. Wire it into an apply_*, or delete the parse."));
    }

    // ── the unread sweep ──
    std::print("{}\n", dim("  " + rule(W - 2)));
    if (not complete_)
    {
        // Never an empty unread list: that reads as a clean bill of health, and this agent has not
        // earned one. State the raw unaccounted count as a fact, not an accusation.
        std::print("  {} {}\n", paint("33;1", "⚠ UNREAD SWEEP INCONCLUSIVE"),
                   dim(std::format("— {} has not declared its registry complete", label)));
        std::print("{}\n", dim(std::format("      {} file key(s) unaccounted for, expected while "
                                           "migration is partial. Run", unread.size())));
        std::print("{}\n", dim(std::format("      common/config_report/check_registry_complete.sh {}",
                                           label)));
    }
    else if (unread.empty())
        std::print("  {} {}\n", paint("32;1", "✓ SWEEP ARMED"),
                   dim(std::format("— 0 of {} file keys unread", file_keys.size())));
    else
    {
        std::print("  {} {}\n", paint("31;1", std::format("✗ {} KEY(S) READ BY NOTHING",
                                                          unread.size())),
                   dim("— in the file, reaching no code (sweep ARMED)"));
        for (const auto& fk : unread)
            std::print("      {} {}\n", paint("31", pad(clip(fk.key, 48), 49)),
                       dim("= " + clip(fk.value, 20) + "   ← delete it, or wire it up"));
    }

    // ── footer counters ──
    {
        std::string ex;
        for (const auto& [prefix, why] : exempt_) ex += std::format("{}*[{}] ", prefix, absorbed[prefix]);
        const auto arms = std::ranges::count_if(recs_, [](const auto& kv)
                                               { return kv.second.kind == Kind::Experiment; });
        const auto chip = [&](std::string_view sgr, auto n, std::string_view what)
        { return paint(sgr, std::format("{}", n)) + dim(std::format(" {}", what)); };
        std::print("  {}   {}   {}   {}   {}\n",
                   chip("1",    recs_.size(),    "read"),
                   chip("36;1", deltas,          "differ"),
                   chip("36",   overlays,        "by overlay"),
                   chip("35;1", arms,            "A/B"),
                   chip(p.undocumented == 0 ? "32" : "33;1", p.undocumented, "undocumented"));
        std::print("{}\n", dim(std::format("  full table → {}   ·   exempt {}",
                                           dump_path, ex.empty() ? "none" : ex)));
        std::print("\n");
    }

    flush_locked(file_keys, dump_path, p, "post-overlay");
    published_ = true;
    last_dump_ = dump_path;
    last_fp_ = p.fingerprint;
    last_arm_ = p.arm;
    return p;
}

inline void Registry::flush_locked(const std::vector<FileKey>& file_keys, const std::string& dump_path,
                                   const Published& p, std::string_view stage) const
{
    std::ofstream out;
    if (not rc::diag::open_rotating(out, dump_path))   // imbues the classic locale itself
    {
        std::print("[cfg] ★ could not write {} — this run has no effective-config record\n", dump_path);
        return;
    }
    std::print(out, "# agent={} started={} fingerprint={} arm={}\n", agent_, detail::now_iso(),
               p.fingerprint, p.arm);
    std::print(out, "# registry_complete={} stage={} keys_read={} keys_in_file={} unread={} undocumented={}\n",
               complete_ ? 1 : 0, stage, recs_.size(), file_keys.size(), p.unread, p.undocumented);
    std::string ex;
    for (const auto& [prefix, why] : exempt_) ex += prefix + "*|";
    std::print(out, "# exempt={}\n", ex);
    std::print(out, "key,type,origin,kind,mut,code_default,effective,overlay_source,reads,description\n");
    for (const auto& [k, r] : recs_)
        std::print(out, "{},{},{},{},{},{},{},{},{},{}\n", detail::csv(k), detail::csv(r.type),
                   to_string(r.origin), to_string(r.kind),
                   r.mut == Mutability::Startup ? "startup" : "runtime",
                   detail::csv(r.code_default), detail::csv(r.effective), detail::csv(r.overlay_source),
                   r.reads, detail::csv(r.description));
    // The unread keys belong in the record too — they are the sweep's finding, not an absence.
    for (const auto& fk : file_keys)
        if (not recs_.contains(fk.key) and not consumed_.contains(fk.key)
            and std::ranges::none_of(exempt_, [&](const auto& e) { return fk.key.starts_with(e.first); }))
            std::print(out, "{},,unread,,,,{},,0,\n", detail::csv(fk.key), detail::csv(fk.value));
}

// room_concept's resolve_overlays_from_graph() can std::exit(EXIT_FAILURE) on a missing or unknown
// scenario_name, i.e. BEFORE publish(). A refusal to start should still leave a record of what the
// agent believed, and a run that never published should say so rather than look like a clean one.
inline Registry::~Registry()
{
    if (not exit_flush_ or published_ or recs_.empty()) return;
    std::print("[cfg] ★ publish() was never called — this run has no effective-config record.\n"
               "[cfg]   Dumping {} key(s) as they stood, marked stage=pre-overlay.\n", recs_.size());
    Published p;
    p.fingerprint = fingerprint();
    p.arm = arm();
    flush_locked({}, last_dump_.empty() ? std::string("etc/config_effective.csv") : last_dump_,
                 p, "pre-overlay");
}

// The arm must be written into the run's OWN metrics rows: config is read once at startup, so a
// file's mtime does not say which run used it. Prepend this to any other diagnostic CSV's header.
inline std::string run_stamp()
{
    auto& r = registry();
    const std::lock_guard lk(r.m_);
    return std::format("# cfg_fp={} arm={}\n", r.last_fp_.empty() ? "unpublished" : r.last_fp_,
                       r.last_arm_.empty() ? "unknown" : r.last_arm_);
}

}  // namespace rc::cfg
