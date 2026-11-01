#pragma once
/*
 * instance_identity.h — an agent's instance numbers (chair_<N>) and object_uid, never issued twice (2026-11-01).
 *
 * WHY: the agents used to rebuild their name high-water mark from the LIVE graph at start-up. A chair that died
 *   before a restart is not in the graph, so its number was handed to the next birth: one name, two objects.
 *   Measured 2026-08-11 in chair_concept: "chair_3" died at (-2.16,-4.43) and the name went to a chair born at
 *   (-1.61,-2.80) — in the logs, one chair teleporting 1.8 m. Everything keyed on the name (phantom log, existence
 *   trace, episodic memory, given names, the long-term object memory) splices two objects together.
 *   Spec: docs/superpowers/specs/2026-11-01-long-term-object-memory-design.md §7, §12.
 *
 * RULE: the next number is max(persisted counter, highest number in the live graph) + 1, and it is WRITTEN TO DISK
 *   BEFORE IT IS RETURNED. A crash between the write and the node insert leaves a gap, never a reuse.
 *   A birth never fails here: a failed write is logged loudly and the in-memory counter carries on.
 *   A missing or corrupt file never restarts from zero — the live graph still bounds it from below.
 *
 * FILE (one per agent, in its working directory, e.g. chair_concept/etc/instance_identity.csv), two ASCII lines:
 *     agent_id <uint>
 *     last <int>
 *   Written through a temp file in the same directory + rename, with the classic locale. Read with from_chars
 *   (es_ES machines: see CLAUDE.md). Delete the file to restart numbering (the live graph still applies).
 *
 * object_uid = (agent_id << 40) | N — the same N as the name, unique across agents with no coordination.
 * Header-only, no DSR, no Qt. Main thread (one counter per agent; not synchronised).
 */

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <locale>
#include <print>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>

namespace rc::identity
{

enum class LoadStatus { ok, missing, corrupt, agent_mismatch };

inline std::uint64_t make_uid(std::uint32_t agent_id, int n)
{
    return (static_cast<std::uint64_t>(agent_id) << 40) | static_cast<std::uint64_t>(n);
}

namespace detail
{
// The whole of `s`, after trimming spaces/tabs/CR at either end, must be a non-negative integer; "5b", "7,0", "" and
// "-3" are not. Trimming matters: a CRLF file or a hand edit with a trailing space is not corruption.
inline bool parse_count(std::string_view s, int& out)
{
    while (not s.empty() and (s.front() == ' ' or s.front() == '\t' or s.front() == '\r')) s.remove_prefix(1);
    while (not s.empty() and (s.back() == ' ' or s.back() == '\t' or s.back() == '\r')) s.remove_suffix(1);
    if (s.empty()) return false;
    int v = 0;
    const auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc{} or p != s.data() + s.size() or v < 0) return false;
    out = v;
    return true;
}
} // namespace detail

// Highest N over names "<prefix><N>" (the prefix includes the '_'); names that do not parse are ignored; 0 if none.
inline int max_suffix(std::span<const std::string> names, std::string_view prefix)
{
    int best = 0;
    for (const auto& name : names)
        if (int n = 0; name.starts_with(prefix) and detail::parse_count(std::string_view(name).substr(prefix.size()), n))
            best = std::max(best, n);
    return best;
}

class InstanceCounter
{
public:
    static InstanceCounter open(std::filesystem::path file, std::uint32_t agent_id)
    {
        InstanceCounter c(std::move(file), agent_id);
        c.load();
        return c;
    }

    LoadStatus status() const { return status_; }
    int last() const { return last_; }

    int issue(int live_max)
    {
        last_ = std::max(last_, live_max) + 1;
        save();
        return last_;
    }

private:
    InstanceCounter(std::filesystem::path file, std::uint32_t agent_id) : file_(std::move(file)), agent_id_(agent_id) {}

    void load()
    {
        std::error_code ec;
        if (not std::filesystem::exists(file_, ec))
        {
            status_ = LoadStatus::missing;
            std::println(stderr, "[identity] {}: missing — numbering continues from the live graph", file_.string());
            return;
        }
        std::ifstream f(file_);
        std::string line, reason;
        bool have_agent = false, have_last = false;
        int agent = 0, last = 0;
        while (std::getline(f, line))
        {
            const auto sp = line.find(' ');
            const std::string_view key = std::string_view(line).substr(0, sp);
            const std::string_view val = sp == std::string::npos ? std::string_view{} : std::string_view(line).substr(sp + 1);
            if (key == "agent_id") have_agent = detail::parse_count(val, agent);
            else if (key == "last") have_last = detail::parse_count(val, last);
        }
        // A readable `last` is kept even when the rest is damaged: under max(...) a real counter can never be wrong,
        // and dropping it would let the live graph re-issue every dead instance's number.
        if (have_last) last_ = last;
        if (not have_agent or not have_last)
        {
            status_ = LoadStatus::corrupt;
            std::println(stderr, "[identity] ⚠ {}: corrupt ({}) — numbering continues from max(live graph, {})",
                         file_.string(), not have_agent ? "no readable agent_id" : "no readable last", last_);
            return;
        }
        if (static_cast<std::uint32_t>(agent) != agent_id_)
        {
            status_ = LoadStatus::agent_mismatch;
            std::println(stderr, "[identity] {}: written by agent {}, I am {} — keeping the larger counter ({})",
                         file_.string(), agent, agent_id_, last_);
            return;
        }
        status_ = LoadStatus::ok;
    }

    void save() const
    {
        const auto tmp = std::filesystem::path(file_.string() + ".tmp");
        {
            std::ofstream f(tmp, std::ios::trunc);
            f.imbue(std::locale::classic());
            f << "agent_id " << agent_id_ << '\n' << "last " << last_ << '\n';
            f.flush();
            if (not f)
            {
                std::println(stderr, "[identity] ⚠ cannot write {} — {} issued from memory only; it may be re-issued "
                                     "after a restart", tmp.string(), last_);
                return;
            }
        }
        std::error_code ec;
        std::filesystem::rename(tmp, file_, ec);
        if (ec)
            std::println(stderr, "[identity] ⚠ cannot rename {} -> {}: {} — {} issued from memory only",
                         tmp.string(), file_.string(), ec.message(), last_);
    }

    std::filesystem::path file_;
    std::uint32_t agent_id_;
    int last_ = 0;
    LoadStatus status_ = LoadStatus::missing;
};

} // namespace rc::identity
