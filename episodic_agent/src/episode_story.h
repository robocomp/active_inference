#pragma once
/*
 * episode_story.h — a mission told as a story, for a PERSON (2026-10-18): the "View data" popup in the em window
 * (episode_text.h). The chat tools keep their compact one-line-per-record form (em_answers.h), which is written for
 * the 8B model; this one is written for whoever is watching the robot: how the mission started, what happened in
 * order, how it ended, in words.
 *
 *   header      what the mission was, who asked, START (time, room) and END (time, duration, the outcome explained)
 *   timeline    one line per moment, wall-clock time and seconds since the start: the steps and actions (claims)
 *               starting and finishing, what the robot believed about the target before and after, and what changed
 *               in the world meanwhile (things noticed or forgotten, actions offered or withdrawn, room entered).
 *               A record's events are copied onto its ancestors (em_builder.h), so the same event is shown once.
 *
 * Clicking a step or claim tells its whole mission and marks that record's lines with "»". Pure over a Store; the
 * outcome and event words are em_builder.h's and the controller journal's. An unknown word is shown as it is.
 */

#include "../../common/episodic/em_query.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace rc::em::story
{

/// More timeline lines than this are cut (the rest counted): a long wander can collect hundreds of events.
inline constexpr std::size_t kMaxLines = 60;
/// A node born or dropped more often than this in one mission is told once, as flickering.
inline constexpr int kFlicker = 4;

namespace detail
{

inline std::string hhmmss(std::uint64_t t_ms)
{
    using namespace std::chrono;
    const auto lt = zoned_time{current_zone(), sys_time<milliseconds>{milliseconds{std::int64_t(t_ms)}}}.get_local_time();
    const hh_mm_ss hms{floor<seconds>(lt - floor<days>(lt))};
    return std::format("{:02}:{:02}:{:02}", hms.hours().count(), hms.minutes().count(), hms.seconds().count());
}

inline std::string duration(std::uint64_t ms)
{
    const auto s = std::int64_t(std::llround(double(ms) / 1000.0));
    if (s < 60) return std::format("{} s", s);
    if (s < 3600) return std::format("{} min {:02} s", s / 60, s % 60);
    return std::format("{} h {:02} min", s / 3600, (s % 3600) / 60);
}

/// An affordance node name in words: "aff_door_5_open" → "open door_5", "afford_room" → "explore the room".
inline std::string action_words(std::string_view aff)
{
    if (aff == "afford_room") return "explore the room";
    if (not aff.starts_with("aff_")) return std::string(aff);
    const std::string_view rest = aff.substr(4);
    static constexpr std::array verbs{"open", "cross", "approach", "inspect", "close", "reach", "look", "pass"};
    if (const auto u = rest.rfind('_'); u != std::string_view::npos)
        for (const std::string_view v : verbs)
            if (rest.substr(u + 1) == v) return std::string(v) + " " + std::string(rest.substr(0, u));
    return "look at " + std::string(rest);
}

/// A claim or step as an action: "open door_5", "explore the room" (afford_room has no target).
inline std::string doing(const Record& r)
{
    if (r.kind == "afford_room" or (r.kind == "room" and r.target.empty())) return "explore the room";
    return r.target.empty() ? r.kind : r.kind + " " + r.target;
}

inline std::string ing(std::string_view verb)
{
    if (verb == "stop") return "stopping";
    if (verb.ends_with('e')) return std::string(verb.substr(0, verb.size() - 1)) + "ing";
    return std::string(verb) + "ing";
}

/// "opening door_5" for the finishing line.
inline std::string doing_ing(const Record& r)
{
    const std::string d = doing(r);
    const auto sp = d.find(' ');
    return sp == std::string::npos ? ing(d) : ing(d.substr(0, sp)) + d.substr(sp);
}

/// What an outcome word means, for a mission / step / claim. `brief` = a few words, for the day's list.
inline std::string explain(const Record& r, bool brief = false)
{
    const std::string& o = r.outcome;
    const bool self = r.author == Author::Self and r.level == Level::Mission;
    if (o.empty()) return r.t1 ? "no outcome reported" : "still running";
    if (o == "completed") return brief ? "done" : "done, the order was carried out";
    if (o == "satisfied" or o == "succeeded") return "succeeded";
    if (o == "superseded")
        return brief ? (self ? "interrupted by an order" : "replaced by another order")
             : self  ? "interrupted, because a person gave an order (or the drive mode changed)"
                     : "replaced by another order";
    if (o == "room_changed")
        return brief ? "entered another room" : "it entered another room, where a new episode on its own began";
    if (o == "preempted") return brief ? "cut short" : "cut short, the robot switched to something else first";
    if (o == "lost")
        return brief ? "no end reported"
                     : "end unknown, it never reported one (the controller was restarted or stopped, or the report was missed)";
    if (o == "interrupted_by_em_shutdown")
        return brief ? "memory shut down" : "end unknown, the memory was shut down while it was running";
    if (o == "cancelled" or o == "aborted") return "cancelled";
    if (o == "unreachable") return "gave up, the target could not be reached";
    if (o.starts_with("failed")) return o.size() > 7 ? "failed (" + o.substr(7) + ")" : "failed";
    return o;
}

/// "door_5 open 0% → 100%, reachable 68% → 90%"; pre-only or post-only values read "was"/"now". Beliefs only.
inline std::vector<std::string> beliefs(const std::vector<Fact>& pre, const std::vector<Fact>& post, bool show_pre,
                                        bool show_post)
{
    static const std::map<std::string, std::string, std::less<>> word{
        {"door_open_prob", "open"}, {"door_reach_prob", "reachable"}, {"obj_exist_prob", "really there"},
        {"passage_passable", "passable"}};
    std::map<std::string, std::map<std::string, std::pair<std::optional<double>, std::optional<double>>>> by;
    if (show_pre)
        for (const Fact& f : pre)
            for (const auto& [k, v] : f.values)
                if (word.contains(k)) by[f.node][k].first = v;
    if (show_post)
        for (const Fact& f : post)
            for (const auto& [k, v] : f.values)
                if (word.contains(k)) by[f.node][k].second = v;
    const auto pct = [](double p) { return std::format("{:.0f}%", 100.0 * std::clamp(p, 0.0, 1.0)); };
    std::vector<std::string> out;
    for (const auto& [node, attrs] : by)
    {
        std::string s = node;
        bool first = true;
        for (const auto& [k, pp] : attrs)
        {
            s += (first ? " " : ", ") + word.find(k)->second + " ";
            first = false;
            if (pp.first and pp.second)
                s += pct(*pp.first) == pct(*pp.second) ? "still " + pct(*pp.first) : pct(*pp.first) + " → " + pct(*pp.second);
            else s += pct(pp.first ? *pp.first : *pp.second);
        }
        out.push_back(std::move(s));
    }
    return out;
}

struct Line
{
    std::uint64_t t;
    int order;                  // within one instant: starts before events before ends
    std::string text;
    std::vector<std::string> notes;
    bool marked = false;
};

/// "chair_2" before "chair_10".
struct Natural
{
    using is_transparent = void;
    bool operator()(std::string_view a, std::string_view b) const
    {
        const auto split = [](std::string_view s)
        {
            std::size_t i = s.size();
            while (i > 0 and s[i - 1] >= '0' and s[i - 1] <= '9') --i;
            std::uint64_t n = 0;
            std::from_chars(s.data() + i, s.data() + s.size(), n);
            return std::pair{s.substr(0, i), i < s.size() ? n : 0};
        };
        const auto [pa, na] = split(a);
        const auto [pb, nb] = split(b);
        return std::tie(pa, na, a) < std::tie(pb, nb, b);
    }
};

/// Whether an event matters to this mission: rooms (entered, discovered, explore-the-room), and the nodes the mission
/// acted on (its steps' and claims' targets) with their affordances ("aff_door_5", "aff_door_5_open"). The rest — the
/// detector's passing sightings, a chair born and forgotten every two seconds — would drown the story.
inline bool relevant(const Event& e, const std::set<std::string, std::less<>>& targets)
{
    if (e.kind == "current_moved" or e.node.starts_with("room_") or e.node == "afford_room") return true;
    if ((e.kind == "born" or e.kind == "died") and e.node.starts_with("door_")) return true;   // the way between rooms
    for (const std::string& t : targets)
        if (e.node == t or e.node == "aff_" + t or e.node.starts_with("aff_" + t + "_")) return true;
    return false;
}

/// The relevant events of every record in the tree, once each, grouped per (instant, kind) into one line. The names
/// of the other things born or forgotten meanwhile go to `others`.
inline std::vector<Line> event_lines(const std::vector<Record>& tree, std::set<std::string, Natural>& others)
{
    std::set<std::string, std::less<>> targets;
    for (const Record& r : tree)
        if (not r.target.empty()) targets.insert(r.target);
    // Once each, in time order.
    std::set<std::tuple<std::uint64_t, std::string, std::string>> unique;
    for (const Record& r : tree)
        for (const Event& e : r.events)
            unique.emplace(r.t0 + std::uint64_t(std::max<std::int64_t>(0, e.dt_ms)), e.kind, e.node);
    // A node born and dropped again and again (a detector losing a door it sees edge-on) is one line, not hundreds.
    std::map<std::string, std::tuple<int, std::uint64_t, std::uint64_t>, std::less<>> churn;   // count, first, last
    for (const auto& [t, kind, node] : unique)
        if (kind == "born" or kind == "died")
        {
            auto& [n, first, last] = churn.try_emplace(node, 0, t, t).first->second;
            ++n;
            last = t;
        }
    std::map<std::pair<std::uint64_t, std::string>, std::vector<std::string>> grouped;
    std::set<std::string, std::less<>> told;         // flickering nodes already given their line
    std::vector<Line> out;
    for (const auto& [t, kind, node] : unique)
    {
        const Event e{.dt_ms = 0, .kind = kind, .node = node};
        const auto c = churn.find(node);
        const bool flickers = c != churn.end() and std::get<0>(c->second) > kFlicker;
        if (not relevant(e, targets))
        {
            if (kind == "born" or kind == "died") others.insert(node);
            continue;
        }
        if (flickers and (kind == "born" or kind == "died"))
        {
            if (told.insert(node).second)
                out.push_back(Line{.t = t, .order = 1,
                                   .text = std::format("{} kept appearing and dropping out of its world model ({} times, "
                                                       "until {}): the perception could not hold on to it",
                                                       node, std::get<0>(c->second), hhmmss(std::get<2>(c->second))),
                                   .notes = {}});
            continue;
        }
        grouped[{t, kind}].push_back(node);
    }
    const auto list = [](const std::vector<std::string>& v, bool as_actions)
    {
        std::string s;
        for (std::size_t i = 0; i < v.size(); ++i)
            s += (i == 0 ? "" : i + 1 == v.size() ? " and " : ", ") + (as_actions ? action_words(v[i]) : v[i]);
        return s;
    };
    for (const auto& [key, nodes] : grouped)
    {
        const auto& [t, kind] = key;
        std::string text;
        if (kind == "born") text = "appeared in its world model: " + list(nodes, false);
        else if (kind == "died") text = "dropped from its world model: " + list(nodes, false);
        else if (kind == "current_moved") text = "the robot is now in " + list(nodes, false);
        else if (kind == "offered") text = "it could now: " + list(nodes, true);
        else if (kind == "withdrawn") text = "no longer offered: " + list(nodes, true);
        else text = kind + ": " + list(nodes, false);
        out.push_back(Line{.t = t, .order = 1, .text = std::move(text), .notes = {}});
    }
    return out;
}

inline std::string title(const Record& m)
{
    if (m.author == Author::Self)
        return "ACTING ON ITS OWN" + (m.room.empty() ? std::string{} : " in " + m.room) + "   [" + m.id + "]";
    const std::string what = m.action_json.empty() ? doing(m) : rc::em::detail::summary(m.action_json);
    return "MISSION \"" + what + "\"   [" + m.id + "]";
}

inline std::string who(const Record& m)
{
    if (m.author == Author::Self) return "Nobody asked: the robot chose what to do by itself.";
    static const std::map<std::string, std::string, std::less<>> via{
        {"chat", "through the chat"}, {"remote", "from the remote control"}, {"panel", "from the mission panel"}};
    const auto v = via.find(m.source);
    return "Asked by a person" + (v != via.end() ? " " + v->second : m.source.empty() ? "" : " (" + m.source + ")")
         + ".";
}

}   // namespace detail

/// The mission holding record `id` (itself, or the ancestor whose parent is ""), told as a story. `id` not in `st`
/// → nullopt.
[[nodiscard]] inline std::optional<std::string> tell(const Store& st, std::string_view id)
{
    using namespace detail;
    const auto find = [&](std::string_view x) { return std::ranges::find(st.today, x, &Record::id); };
    auto it = find(id);
    if (it == st.today.end()) return std::nullopt;
    for (int guard = 0; not it->parent.empty() and guard < 8; ++guard)
        if (const auto p = find(it->parent); p != st.today.end()) it = p;
        else break;
    const Record& m = *it;
    const std::vector<Record> tree = rc::em::episode(st, m.id);   // m first, then its descendants by t0

    std::vector<Line> lines;
    // The mission itself.
    {
        Line l{.t = m.t0, .order = 0, .text = {}, .notes = {}, .marked = false};
        if (m.author == Author::Self) l.text = "began acting on its own" + (m.room.empty() ? std::string{} : " in " + m.room);
        else l.text = "received the order \"" + (m.action_json.empty() ? doing(m) : rc::em::detail::summary(m.action_json))
                    + "\"" + (m.room.empty() ? std::string{} : " while in " + m.room);
        lines.push_back(std::move(l));
    }
    std::map<std::string, int> claims_per_parent;
    for (const Record& r : tree)
    {
        if (r.id == m.id) continue;
        const bool mark = r.id == id;
        Line start{.t = r.t0, .order = 0, .text = {}, .notes = beliefs(r.pre, {}, true, false), .marked = mark};
        if (r.level == Level::Step)
        {
            start.text = (r.attempt > 1 ? std::format("tried again (attempt {}) to ", r.attempt) : std::string("set out to "))
                       + doing(r);
            if (not start.notes.empty()) start.notes.insert(start.notes.begin(), "what it believed then:");
        }
        else
        {
            start.text = "decided to " + doing(r);
            if (r.promised) start.text += std::format(", expecting to learn {:.2f} nats", *r.promised);
            if (not start.notes.empty()) start.notes.insert(start.notes.begin(), "what it believed then:");
            ++claims_per_parent[r.parent];
        }
        lines.push_back(std::move(start));
        if (not r.t1) continue;
        if (r.level == Level::Step and r.t1 == m.t1 and r.outcome == m.outcome) continue;   // the mission's end says it
        Line end{.t = *r.t1, .order = 2, .text = {}, .notes = beliefs(r.pre, r.post, true, true), .marked = mark};
        std::erase_if(end.notes, [](const std::string& s) { return s.find("→") == std::string::npos; });   // changes only
        end.text = "finished " + doing_ing(r) + " — " + explain(r);
        if (r.level == Level::Claim and r.delivered)
            end.text += *r.delivered > 0.005 ? std::format(" (learned {:.2f} nats)", *r.delivered)
                                             : std::string(" (learned nothing new)");
        if (not end.notes.empty()) end.notes.insert(end.notes.begin(), "what changed:");
        lines.push_back(std::move(end));
    }
    std::set<std::string, Natural> others;
    for (Line& l : event_lines(tree, others)) lines.push_back(std::move(l));
    if (m.t1)
        lines.push_back(Line{.t = *m.t1, .order = 3,
                             .text = (m.author == Author::Self ? "stopped acting on its own — " : "the mission ended — ")
                                   + explain(m),
                             .notes = {}, .marked = false});
    std::ranges::stable_sort(lines, [](const Line& a, const Line& b) { return std::tie(a.t, a.order) < std::tie(b.t, b.order); });

    std::string out = title(m) + "\n" + who(m) + "\n\n";
    out += "START   " + hhmmss(m.t0) + (m.room.empty() ? std::string{} : ", in " + m.room) + "\n";
    if (m.t1) out += "END     " + hhmmss(*m.t1) + ", " + duration(*m.t1 - m.t0) + " later — " + explain(m) + "\n";
    else out += "END     not yet — still running\n";
    const auto n_steps = std::ranges::count(tree, Level::Step, &Record::level);
    const auto n_claims = std::ranges::count(tree, Level::Claim, &Record::level);
    out += std::format("        {} step{}, {} action{} chosen along the way\n", n_steps, n_steps == 1 ? "" : "s",
                       n_claims, n_claims == 1 ? "" : "s");
    if (id != m.id) out += "        lines marked » belong to " + std::string(id) + ", the record you clicked\n";
    out += "\nWHAT HAPPENED\n";
    const std::size_t shown = std::min(lines.size(), kMaxLines);
    for (std::size_t i = 0; i < shown; ++i)
    {
        const Line& l = lines[i];
        const std::string rel = std::format("+{}", duration(l.t >= m.t0 ? l.t - m.t0 : 0));
        out += std::format("{} {}  {:>12}   {}\n", l.marked ? "»" : " ", hhmmss(l.t), rel, l.text);
        for (const std::string& n : l.notes) out += std::format("{:28}{}\n", "", (n.ends_with(':') ? "" : "  ") + n);
    }
    if (lines.size() > shown) out += std::format("  … {} more moments not shown\n", lines.size() - shown);
    if (not others.empty())
    {
        out += std::format("\nMEANWHILE, NOT RELATED TO THE MISSION\n  {} other thing{} came and went in the world model: ",
                           others.size(), others.size() == 1 ? "" : "s");
        std::size_t i = 0;
        for (const std::string& o : others)
        {
            if (i == 12) { out += std::format(", … ({} more)", others.size() - 12); break; }
            out += (i++ == 0 ? "" : ", ") + o;
        }
        out += "\n";
    }
    return out;
}

/// The day's missions, one plain line each: "11:15:36  49 s  "cross door_5", asked by a person — completed: …".
[[nodiscard]] inline std::string day_list(const Store& st, std::string_view day)
{
    using namespace detail;
    std::string out;
    int n = 0;
    for (const Record& r : st.today)
    {
        if (r.level != Level::Mission or day_of(r.t0) != day) continue;
        ++n;
        const std::string what = r.author == Author::Self
            ? "on its own" + (r.room.empty() ? std::string{} : " in " + r.room)
            : "\"" + (r.action_json.empty() ? doing(r) : rc::em::detail::summary(r.action_json)) + "\", asked by a person";
        out += std::format("  {}  {:>11}   {} — {}   [{}]\n", hhmmss(r.t0), r.t1 ? duration(*r.t1 - r.t0) : "running",
                           what, explain(r, true), r.id);
    }
    if (n == 0) return {};
    return std::format("{} mission{} on {}-{}-{} (click one for its story):\n", n, n == 1 ? "" : "s", day.substr(0, 4),
                       day.substr(4, 2), day.substr(6, 2)) + out;
}

}   // namespace rc::em::story
