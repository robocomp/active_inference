#pragma once
/*
 * em_answers.h — the chat's episodic-memory tools as pure text (2026-10-17): what the robot did TODAY (missions, their
 * steps, the affordance claims under them, with outcomes) and the all-time outcome statistics, formatted for an 8B
 * model from an rc::em::Store (common/episodic/em_query.h). No DSR (tested in tests/test_em.cpp); ChatSession reads the
 * Store from episodic_agent's domain-4 graph with rc::em::load_store and passes it here.
 *
 * AVAILABILITY: std::nullopt = chat_agent has not joined domain 4 (episodic_agent is not running — the join is lazy,
 * specificworker.cpp). A joined graph holding nothing at all (no record, no statistic) is answered the same way: the
 * memory has nothing to say, and "nothing happened today" would be a claim it cannot back.
 *
 * TODAY (finding 11): the graph keeps a day's records until the first commitment of the next day rolls them over, so
 * after midnight it can still hold yesterday's. The "today" tools (episodes, episode, last_episode, the STATE line) keep
 * only records with day_of(t0) == today; `today` = "YYYYMMDD", empty = the current local day.
 * ALL TIME (finding 3): the stats nodes hold only rolled-over days, so outcome_stats folds every closed record still in
 * the graph into a copy of them before answering (fold is pure) — "all time" then really includes today.
 * TIMES are local wall-clock HH:MM (the record's t0/t1 are ms since the epoch). The tools accept "HH:MM" (today, local
 * time) or epoch milliseconds for since/until (since inclusive, until exclusive, on t0 — em_query.h).
 */

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "../../common/episodic/em_consolidate.h"
#include "../../common/episodic/em_query.h"

namespace rc::chat
{

inline constexpr std::string_view kEmUnavailable = "episodic memory is not available";

/// episodes() lists at most this many (the newest); the rest are counted, and the model is told how to narrow.
/// A presentation cap for the 8B model's context, not a model threshold.
inline constexpr std::size_t kEmMaxListed = 25;
/// episode(): structural events listed per record (the rest counted).
inline constexpr std::size_t kEmMaxEvents = 8;

namespace em_detail
{
using namespace rc::em;

inline bool unavailable(const std::optional<Store>& st) { return not st or (st->today.empty() and st->stats.empty()); }

/// "YYYYMMDD" of `today`, or of now when empty.
inline std::string today_or_now(std::string_view today)
{
    if (not today.empty()) return std::string(today);
    using namespace std::chrono;
    return day_of(std::uint64_t(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count()));
}

/// The store with only today's records (stats untouched).
inline Store todays(const Store& st, std::string_view today)
{
    const std::string d = today_or_now(today);
    Store out{.today = {}, .stats = st.stats};
    for (const Record& r : st.today)
        if (day_of(r.t0) == d) out.today.push_back(r);
    return out;
}

/// The stats nodes plus every CLOSED record still in the graph (not yet rolled over), folded into a copy.
inline std::vector<Stats> all_time(const Store& st)
{
    std::map<StatsKey, Stats> by_key;
    for (const Stats& s : st.stats) by_key[StatsKey{s.key_level, s.key_kind, s.key_author, s.key_identity}] = s;
    for (const Record& r : st.today)
        if (r.t1) fold(by_key[stats_key(r)], r);
    std::vector<Stats> out;
    for (auto& [k, s] : by_key) out.push_back(std::move(s));
    return out;
}

/// "missions asked by a person" / "missions on its own" / "<level> <kind>".
inline std::string stats_label(const Stats& s)
{
    if (s.key_level == "mission")
        return s.key_author == "self" ? "missions on its own" : s.key_author == "human" ? "missions asked by a person"
                                                                                       : "missions";
    return s.key_level.empty() ? s.key_kind : s.key_level + " " + s.key_kind;
}

inline std::chrono::zoned_time<std::chrono::milliseconds> local(std::uint64_t t_ms)
{
    using namespace std::chrono;
    return zoned_time{current_zone(), sys_time<milliseconds>{milliseconds{std::int64_t(t_ms)}}};
}

/// "HH:MM", local time.
inline std::string hhmm(std::uint64_t t_ms)
{
    using namespace std::chrono;
    const auto lt = local(t_ms).get_local_time();
    const hh_mm_ss hms{floor<minutes>(lt - floor<days>(lt))};
    return std::format("{:02}:{:02}", hms.hours().count(), hms.minutes().count());
}

/// "YYYY-MM-DD HH:MM", local time.
inline std::string date_hhmm(std::uint64_t t_ms)
{
    using namespace std::chrono;
    const year_month_day ymd{floor<days>(local(t_ms).get_local_time())};
    return std::format("{:04}-{:02}-{:02} {}", int(ymd.year()), unsigned(ymd.month()), unsigned(ymd.day()), hhmm(t_ms));
}

/// One record on one line: "<id> <HH:MM>-<HH:MM|now> <what>[ (part of <parent>)]: <outcome>".
inline std::string line(const Record& r)
{
    std::string what;
    switch (r.level)
    {
        case Level::Mission:
            // A self mission (em_builder.h open_self, ruling R10) has kind "mission" and no action: just "mission".
            what = not r.action_json.empty() ? "mission \"" + rc::em::detail::summary(r.action_json) + "\""
                 : "mission" + (r.kind.empty() or r.kind == "mission" ? std::string{} : " " + r.kind)
                             + (r.target.empty() ? std::string{} : " " + r.target);
            if (not r.room.empty()) what += " in " + r.room;
            what += r.author == Author::Human ? ", asked by a person" + (r.source.empty() ? "" : " (" + r.source + ")")
                                              : std::string(", on its own");
            break;
        case Level::Step:
            what = "step " + r.kind + (r.target.empty() ? "" : " " + r.target);
            if (r.attempt > 1) what += std::format(", attempt {}", r.attempt);
            break;
        case Level::Claim:
            what = "claim " + r.kind + (r.target.empty() ? "" : " " + r.target);
            if (r.promised and r.delivered)
                what += std::format(", promised {:.2f} delivered {:.2f} nats", *r.promised, *r.delivered);
            else if (r.promised) what += std::format(", promised {:.2f} nats", *r.promised);
            break;
    }
    if (not r.parent.empty()) what += " (part of " + r.parent + ")";
    const std::string outcome = not r.outcome.empty() ? r.outcome : r.t1 ? std::string("no outcome") : std::string("still open");
    return std::format("{} {}-{} {}: {}", r.id, hhmm(r.t0), r.t1 ? hhmm(*r.t1) : std::string("now"), what, outcome);
}

inline bool is_belief_key(std::string_view k)
{
    return std::ranges::find(kFactAttrs, k) != kFactAttrs.end();
}

/// "<node>: <attr> <pre> -> <post>, …" per node, beliefs only (kFactAttrs; pose and covariance are left out — they
/// answer nothing a person asks). `only` non-empty = that node only. Empty when no belief was read.
inline std::vector<std::string> fact_lines(const Record& r, std::string_view only = {})
{
    std::map<std::string, std::map<std::string, std::pair<std::optional<double>, std::optional<double>>>> by_node;
    for (const Fact& f : r.pre)
        for (const auto& [k, v] : f.values)
            if (is_belief_key(k) and (only.empty() or f.node == only)) by_node[f.node][k].first = v;
    for (const Fact& f : r.post)
        for (const auto& [k, v] : f.values)
            if (is_belief_key(k) and (only.empty() or f.node == only)) by_node[f.node][k].second = v;
    std::vector<std::string> out;
    for (const auto& [node, attrs] : by_node)
    {
        std::string s = node + ":";
        bool first = true;
        for (const auto& [k, pp] : attrs)
        {
            s += (first ? " " : ", ") + k + " ";
            first = false;
            if (pp.first and pp.second) s += std::format("{:.2f} -> {:.2f}", *pp.first, *pp.second);
            else if (pp.first) s += std::format("{:.2f} before", *pp.first);
            else s += std::format("{:.2f} after", *pp.second);
        }
        out.push_back(std::move(s));
    }
    return out;
}

/// The memory identity (ltsm id) of a live node name, from today's records; else `target` itself if the statistics
/// are keyed by it; else nullopt.
inline std::optional<std::string> identity_of(const Store& st, std::string_view target)
{
    for (const Record& r : st.today)
    {
        if (r.target == target and not r.target_ltsm.empty()) return r.target_ltsm;
        for (const auto* fs : {&r.pre, &r.post})
            for (const Fact& f : *fs)
                if (f.node == target and not f.ltsm_id.empty()) return f.ltsm_id;
    }
    if (std::ranges::any_of(st.stats, [&](const Stats& s) { return s.key_identity == target; }))
        return std::string(target);
    return std::nullopt;
}

inline std::string stats_line(const Stats& s, std::string_view label)
{
    std::vector<std::pair<std::string, int>> oc(s.outcomes.begin(), s.outcomes.end());
    std::ranges::stable_sort(oc, [](const auto& a, const auto& b) { return a.second > b.second; });
    std::string outs;
    for (const auto& [k, n] : oc) outs += (outs.empty() ? "" : ", ") + std::format("{} {}", k, n);
    std::string out = std::format("{} (all time): {} time{}", label, s.n, s.n == 1 ? "" : "s");
    if (not outs.empty()) out += ": " + outs;
    if (s.n_dur > 0)   // lost / interrupted records are counted but not timed (em_consolidate.h)
    {
        out += std::format("; mean duration {:.1f} s", s.dur_mean_s);
        if (s.n_dur > 1) out += std::format(" (sd {:.1f} s)", std::sqrt(s.dur_m2 / double(s.n_dur - 1)));
    }
    if (s.n_gain > 0)
        out += std::format("; information promised {:.2f}, delivered {:.2f} nats on average over {}",
                           s.promised_sum / s.n_gain, s.delivered_sum / s.n_gain, s.n_gain);
    if (s.first_t0 != 0) out += "; first " + date_hhmm(s.first_t0);
    if (s.last_t0 != 0) out += (s.first_t0 != 0 ? ", last " : "; last ") + date_hhmm(s.last_t0);
    return out;
}
}   // namespace em_detail

/// A since/until argument: "HH:MM" (today, local time) or epoch milliseconds. nullopt = not understood.
[[nodiscard]] inline std::optional<std::uint64_t> parse_em_time(std::string_view s)
{
    const auto num = [](std::string_view p) -> std::optional<std::uint64_t>
    {
        std::uint64_t v = 0;
        const auto [end, ec] = std::from_chars(p.data(), p.data() + p.size(), v);
        if (p.empty() or ec != std::errc{} or end != p.data() + p.size()) return std::nullopt;
        return v;
    };
    const auto colon = s.find(':');
    if (colon == std::string_view::npos) return num(s);
    const auto h = num(s.substr(0, colon)), m = num(s.substr(colon + 1));
    if (not h or not m or *h > 23 or *m > 59 or s.size() - colon - 1 != 2) return std::nullopt;
    using namespace std::chrono;
    const auto* z = current_zone();
    const auto today = floor<days>(z->to_local(system_clock::now()));
    return std::uint64_t(duration_cast<milliseconds>(
        z->to_sys(today + hours{*h} + minutes{*m}, choose::earliest).time_since_epoch()).count());
}

/// episodes(since?, until?, author?, kind?, target?): today's records, by start time. With neither kind nor target
/// (finding 13) only the MISSIONS are listed — the question is usually "what did you do" — with a line saying how to
/// get the steps and claims (kind=<op>, or episode(id) for one mission's tree).
[[nodiscard]] inline std::string answer_episodes(const std::optional<rc::em::Store>& st, std::optional<std::uint64_t> since,
                                                 std::optional<std::uint64_t> until, std::string_view author,
                                                 std::string_view kind, std::string_view target,
                                                 std::string_view today = {})
{
    using namespace em_detail;
    if (unavailable(st)) return std::string(kEmUnavailable);
    std::optional<Author> au;
    if (author == "human") au = Author::Human;
    else if (author == "self") au = Author::Self;
    else if (not author.empty()) return "author must be 'human' or 'self' (or empty for both)";
    const bool missions_only = kind.empty() and target.empty();
    auto v = rc::em::episodes(todays(*st, today), since, until, au, kind, target);
    if (missions_only) std::erase_if(v, [](const Record& r) { return r.level != Level::Mission; });
    if (v.empty()) return "no episode recorded today matches that";
    std::string out = missions_only
        ? std::format("{} mission{} today (steps and claims: call episodes with kind=<op>, e.g. kind=cross, or "
                      "episode(id) for one mission's steps and claims):\n", v.size(), v.size() == 1 ? "" : "s")
        : std::format("{} episode{} today:\n", v.size(), v.size() == 1 ? "" : "s");
    const std::size_t skip = v.size() > kEmMaxListed ? v.size() - kEmMaxListed : 0;
    if (skip > 0)
        out += std::format("({} earlier ones not shown; narrow with since/until/kind/target)\n", skip);
    for (std::size_t i = skip; i < v.size(); ++i) out += "- " + line(v[i]) + "\n";
    return out;
}

/// episode(id): the record and everything under it, indented by depth, with the beliefs read before/after and the
/// structural events seen while it ran.
[[nodiscard]] inline std::string answer_episode(const std::optional<rc::em::Store>& st, std::string_view id,
                                                std::string_view today = {})
{
    using namespace em_detail;
    if (unavailable(st)) return std::string(kEmUnavailable);
    const auto v = rc::em::episode(todays(*st, today), id);
    if (v.empty()) return std::format("no episode with id '{}' today", id);
    std::map<std::string, int, std::less<>> depth;
    std::string out;
    for (const Record& r : v)
    {
        const auto p = depth.find(r.parent);
        const int d = r.id == v.front().id or p == depth.end() ? 0 : p->second + 1;
        depth[r.id] = d;
        const std::string ind(std::size_t(2 * d), ' ');
        out += ind + line(r) + "\n";
        for (const auto& f : fact_lines(r)) out += ind + "  " + f + "\n";
        if (not r.events.empty())
        {
            std::string ev;
            for (std::size_t i = 0; i < r.events.size() and i < kEmMaxEvents; ++i)
                ev += (ev.empty() ? "" : "; ") + std::format("+{} s {} {}", r.events[i].dt_ms / 1000, r.events[i].kind,
                                                            r.events[i].node);
            if (r.events.size() > kEmMaxEvents) ev += std::format("; {} more", r.events.size() - kEmMaxEvents);
            out += ind + "  events: " + ev + "\n";
        }
    }
    return out;
}

/// outcome_stats(kind, target?): all-time counts per outcome, duration and information gain, today included. With a
/// target, the rows keyed to its memory identity; without one (or kind ""), the per-kind rows. A step op and a claim
/// kind of the same name are separate rows ("step cross", "claim cross"); missions are split by author.
[[nodiscard]] inline std::string answer_outcome_stats(const std::optional<rc::em::Store>& st, std::string_view kind,
                                                      std::string_view target)
{
    using namespace em_detail;
    if (unavailable(st)) return std::string(kEmUnavailable);
    const std::vector<Stats> rows = all_time(*st);
    const auto per_kind = [&]
    {
        std::string out;
        for (const Stats& s : rows)
            if ((kind.empty() or s.key_kind == kind) and s.key_identity.empty()) out += stats_line(s, stats_label(s)) + "\n";
        return out;
    };
    const std::string what = kind.empty() ? std::string("any kind") : std::string(kind);
    if (target.empty())
    {
        const std::string out = per_kind();
        return out.empty() ? "no statistics for " + what : out;
    }
    const auto ident = identity_of(*st, target);
    std::string out;
    if (ident)
        for (const Stats& s : rows)
            if ((kind.empty() or s.key_kind == kind) and s.key_identity == *ident)
                out += stats_line(s, stats_label(s) + " " + std::string(target)) + "\n";
    if (not out.empty()) return out;
    const std::string fallback = per_kind();
    std::string head = std::format("no statistics for {} on {}{}", what, target,
                                   ident ? std::string{} : std::string(" (no memory identity for it)"));
    return fallback.empty() ? head : head + "; for all " + what + ":\n" + fallback;
}

/// last_episode(target): the newest record today aimed at `target` or that read a belief of it.
[[nodiscard]] inline std::string answer_last_episode(const std::optional<rc::em::Store>& st, std::string_view target,
                                                     std::string_view today = {})
{
    using namespace em_detail;
    if (unavailable(st)) return std::string(kEmUnavailable);
    const auto r = rc::em::last(todays(*st, today), target);
    if (not r) return std::format("nothing recorded today about '{}'", target);
    std::string out = "last about " + std::string(target) + ": " + line(*r) + "\n";
    for (const auto& f : fact_lines(*r, target)) out += "  " + f + "\n";
    return out;
}

/// The STATE's "today: …" line (em_query.h today_line); nullopt — the line omitted — whenever the tools would say
/// "not available" (no EM graph joined, or joined but empty, e.g. just after the join before its import lands), so
/// the STATE and the tools never disagree.
[[nodiscard]] inline std::optional<std::string> today_digest_line(const std::optional<rc::em::Store>& st,
                                                                std::string_view today = {})
{
    if (em_detail::unavailable(st)) return std::nullopt;
    return "today: " + rc::em::today_line(em_detail::todays(*st, today));
}

}   // namespace rc::chat
