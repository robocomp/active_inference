#pragma once
/*
 * episode_story.h — a mission told for a PERSON (2026-10-18): the "View data" popup in the em window (episode_text.h).
 * The chat tools keep their compact form (em_answers.h), written for the 8B model. EM exists to store the CONSEQUENCES
 * of applying affordances, the material of an abstract world model, so every applied affordance is told as a
 * transition: before → action → after.
 *
 *   header        what the mission was, who asked, START (time, room) and END (time, duration, the outcome explained)
 *   timeline      wall-clock time and time since the start. Each step and applied affordance (claim) is one block:
 *                 what was on offer when it was chosen, what it expected to learn, the state before (room, beliefs),
 *                 and its consequences: beliefs that moved, room change, affordances that became possible or stopped
 *                 being possible, things that appeared or were dropped (net, within its window). Between them, the
 *                 world events not inside any action's window.
 *   consequences  the same transitions with each instance replaced by its type ("open <door> ⇒ then possible:
 *                 cross <door>"): what an abstract world model learns from the mission. An action whose end was never
 *                 seen (lost, EM shutdown) is left out: nothing can be credited to it.
 *
 * The post-state is what EM read at the close (em_builder.h close()), so a consequence that settles after it (a door
 * belief updated a cycle later, the room switch after a crossing) is not in the record and is not shown.
 *
 * Clicking a step or claim tells its whole mission and marks that record's lines with "»". Pure over a Store; the
 * outcome and event words are em_builder.h's and the controller journal's. An unknown word is shown as it is.
 */

#include "../../common/episodic/em_query.h"

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include <algorithm>
#include <cctype>
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
                                        bool show_post, double min_change = 0.0)
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
                s += pct(*pp.first) == pct(*pp.second) or std::abs(*pp.second - *pp.first) < min_change
                         ? "still " + pct(*pp.first) : pct(*pp.first) + " → " + pct(*pp.second);
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

/// The tree's events once each, on absolute time, in time order; and which nodes flicker (born or dropped more often
/// than kFlicker — a detector losing a door it sees edge-on).
struct Events
{
    std::set<std::tuple<std::uint64_t, std::string, std::string>> all;   // t, kind, node
    std::map<std::string, std::tuple<int, std::uint64_t, std::uint64_t>, std::less<>> churn;   // count, first, last
    std::set<std::string, std::less<>> targets;
    bool flickers(std::string_view node) const
    {
        const auto c = churn.find(node);
        return c != churn.end() and std::get<0>(c->second) > kFlicker;
    }
};

inline Events collect(const std::vector<Record>& tree)
{
    Events ev;
    for (const Record& r : tree)
    {
        if (not r.target.empty()) ev.targets.insert(r.target);
        for (const Event& e : r.events)
            ev.all.emplace(r.t0 + std::uint64_t(std::max<std::int64_t>(0, e.dt_ms)), e.kind, e.node);
    }
    for (const auto& [t, kind, node] : ev.all)
        if (kind == "born" or kind == "died")
        {
            auto& [n, first, last] = ev.churn.try_emplace(node, 0, t, t).first->second;
            ++n;
            last = t;
        }
    return ev;
}

inline std::string list(const std::vector<std::string>& v, bool as_actions = false)
{
    std::string s;
    for (std::size_t i = 0; i < v.size(); ++i)
        s += (i == 0 ? "" : i + 1 == v.size() ? " and " : ", ") + (as_actions ? action_words(v[i]) : v[i]);
    return s;
}

/// What applying an affordance changed: its beliefs before → after, and the NET changes in the world seen while it ran
/// (to kGraceMs after its end — an affordance offered as a consequence is published a cycle after the claim closes).
/// An action offered and withdrawn again inside the window is no consequence; a flickering node is never one.
struct Effect
{
    std::string room_before, room_after;
    std::vector<std::string> beliefs;                       // "door_1 open 0% → 89%", changes only
    std::vector<std::string> became_possible, no_longer_possible, appeared, dropped;
};

inline constexpr std::uint64_t kGraceMs = 500;
/// A belief moving less than this (a probability) is not counted as a consequence: estimator jitter.
inline constexpr double kMinChange = 0.05;

inline bool in_window(const Record& r, std::uint64_t t) { return r.t1 and t >= r.t0 and t <= *r.t1 + kGraceMs; }

inline Effect effect(const Record& r, const Events& ev, const std::vector<Record>& tree)
{
    Effect fx{.room_before = r.room, .room_after = r.room, .beliefs = beliefs(r.pre, r.post, true, true, kMinChange),
              .became_possible = {}, .no_longer_possible = {}, .appeared = {}, .dropped = {}};
    std::erase_if(fx.beliefs, [](const std::string& b) { return b.find("→") == std::string::npos; });
    std::map<std::string, int, Natural> offer, exist;      // net count per node
    for (const auto& [t, kind, node] : ev.all)
    {
        if (not in_window(r, t) or ev.flickers(node)) continue;
        if (not relevant(Event{.dt_ms = 0, .kind = kind, .node = node}, ev.targets)) continue;
        if (kind == "current_moved") fx.room_after = node;
        else if (kind == "offered") ++offer[node];
        else if (kind == "withdrawn") --offer[node];
        else if (kind == "born") ++exist[node];
        else if (kind == "died") --exist[node];
    }
    // The action's own affordance coming and going is the selector's bookkeeping, not a consequence of applying it; and
    // a step's means (the affordances its claims applied) are not its effects.
    std::erase_if(offer, [&](const auto& kv)
    {
        const std::string a = action_words(kv.first);
        return a == doing(r) or std::ranges::any_of(tree, [&](const Record& c) { return c.parent == r.id and doing(c) == a; });
    });
    for (const auto& [n, d] : offer)
        if (d > 0) fx.became_possible.push_back(n);
        else if (d < 0) fx.no_longer_possible.push_back(n);
    for (const auto& [n, d] : exist)
        if (d > 0) fx.appeared.push_back(n);
        else if (d < 0) fx.dropped.push_back(n);
    return fx;
}

/// The effect as note lines under the action ("" when nothing was observed).
inline std::vector<std::string> effect_notes(const Effect& fx)
{
    std::vector<std::string> out;
    if (fx.room_after != fx.room_before) out.push_back("  robot moved: " + fx.room_before + " → " + fx.room_after);
    for (const std::string& b : fx.beliefs) out.push_back("  belief: " + b);
    if (not fx.became_possible.empty()) out.push_back("  became possible: " + list(fx.became_possible, true));
    if (not fx.no_longer_possible.empty()) out.push_back("  no longer possible: " + list(fx.no_longer_possible, true));
    if (not fx.appeared.empty()) out.push_back("  appeared: " + list(fx.appeared));
    if (not fx.dropped.empty()) out.push_back("  dropped: " + list(fx.dropped));
    return out;
}

/// Every instance name "word_N" in `s` replaced by its type "<word>": the abstract form.
inline std::string abstracted(std::string_view s)
{
    std::string out, tok;
    const auto flush = [&]
    {
        std::size_t i = tok.size();
        while (i > 0 and std::isdigit(static_cast<unsigned char>(tok[i - 1]))) --i;
        out += (i < tok.size() and i > 1 and tok[i - 1] == '_') ? "<" + tok.substr(0, i - 1) + ">" : tok;
        tok.clear();
    };
    for (const char c : s)
        if (std::isalnum(static_cast<unsigned char>(c)) or c == '_') tok += c;
        else { flush(); out += c; }
    flush();
    return out;
}

/// The affordances the selector had on offer when it chose this claim ("open door_1 (gain 0.30), …"); empty if none.
inline std::string alternatives(const Record& r)
{
    const QJsonDocument d = QJsonDocument::fromJson(QByteArray::fromStdString(r.alternatives_json));
    if (not d.isArray()) return {};
    std::string out;
    int n = 0;
    for (const QJsonValue& v : d.array())
    {
        const QJsonObject o = v.toObject();
        const std::string name = o["node_name"].toString().toStdString();
        if (name.empty()) continue;
        if (++n > 4) { out += ", …"; break; }
        out += (out.empty() ? "" : ", ") + action_words(name);
        if (o["gain"].isDouble()) out += std::format(" (gain {:.2f})", o["gain"].toDouble());
    }
    return out;
}

/// The relevant events as timeline lines, grouped per (instant, kind); a flickering node is one line. Events inside a
/// closed CLAIM's window are left out: they are told as that action's consequences. The names of the unrelated things
/// born or dropped go to `others`.
inline std::vector<Line> event_lines(const std::vector<Record>& tree, const Events& ev, std::set<std::string, Natural>& others)
{
    std::map<std::pair<std::uint64_t, std::string>, std::vector<std::string>> grouped;
    std::set<std::string, std::less<>> told;         // flickering nodes already given their line
    std::vector<Line> out;
    for (const auto& [t, kind, node] : ev.all)
    {
        if (not relevant(Event{.dt_ms = 0, .kind = kind, .node = node}, ev.targets))
        {
            if (kind == "born" or kind == "died") others.insert(node);
            continue;
        }
        if (ev.flickers(node) and (kind == "born" or kind == "died"))
        {
            const auto& [n, first, last] = ev.churn.find(node)->second;
            if (told.insert(node).second)
                out.push_back(Line{.t = t, .order = 1,
                                   .text = std::format("{} kept appearing and dropping out of its world model ({} times, "
                                                       "until {}): the perception could not hold on to it",
                                                       node, n, hhmmss(last)),
                                   .notes = {}});
            continue;
        }
        if (std::ranges::any_of(tree, [&](const Record& r) { return r.level == Level::Claim and in_window(r, t); }))
            continue;
        grouped[{t, kind}].push_back(node);
    }
    for (const auto& [key, nodes] : grouped)
    {
        const auto& [t, kind] = key;
        std::string text;
        if (kind == "born") text = "appeared in its world model: " + list(nodes);
        else if (kind == "died") text = "dropped from its world model: " + list(nodes);
        else if (kind == "current_moved") text = "the robot is now in " + list(nodes);
        else if (kind == "offered") text = "it could now: " + list(nodes, true);
        else if (kind == "withdrawn") text = "no longer offered: " + list(nodes, true);
        else text = kind + ": " + list(nodes);
        out.push_back(Line{.t = t, .order = 1, .text = std::move(text), .notes = {}});
    }
    return out;
}

/// Nothing can be credited to an action whose end was never seen.
inline bool end_unknown(const Record& r) { return r.outcome == "lost" or r.outcome == "interrupted_by_em_shutdown"; }

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
    const Events ev = collect(tree);
    std::vector<std::string> abstract_lines;
    int unknown_ends = 0;
    for (const Record& r : tree)
    {
        if (r.id == m.id) continue;
        const bool claim = r.level == Level::Claim;
        Line l{.t = r.t0, .order = 0, .text = {}, .notes = {}, .marked = r.id == id};
        l.text = (claim ? std::string("APPLIED ")
                        : r.attempt > 1 ? std::format("STEP (attempt {}) ", r.attempt) : std::string("STEP "))
               + doing(r) + (r.t1 ? "  — " + explain(r) + ", " + duration(*r.t1 - r.t0) : std::string("  — still running"));
        if (claim)
        {
            if (const std::string alt = alternatives(r); not alt.empty()) l.notes.push_back("chosen among: " + alt);
            if (r.promised)
                l.notes.push_back(r.delivered
                    ? std::format("expected to learn {:.2f} nats, learned {:.2f}", *r.promised, *r.delivered)
                    : std::format("expected to learn {:.2f} nats", *r.promised));
        }
        std::string before = r.room.empty() ? std::string{} : "in " + r.room;
        for (const std::string& x : beliefs(r.pre, {}, true, false)) before += (before.empty() ? "" : " · ") + x;
        if (not before.empty()) l.notes.push_back("before: " + before);
        if (r.t1 and end_unknown(r))
        {
            l.notes.push_back("after: unknown — its end was never seen, so nothing is credited to it");
            ++unknown_ends;
        }
        else if (r.t1)
        {
            const Effect fx = effect(r, ev, tree);
            const std::vector<std::string> notes = effect_notes(fx);
            l.notes.push_back(notes.empty() ? "consequences: none observed" : "consequences:");
            l.notes.insert(l.notes.end(), notes.begin(), notes.end());
            std::vector<std::string> fxs;
            if (fx.room_after != fx.room_before) fxs.push_back("the robot is in another room");
            for (const std::string& b : fx.beliefs) fxs.push_back(abstracted(b));
            if (not fx.became_possible.empty()) fxs.push_back("then possible: " + abstracted(list(fx.became_possible, true)));
            if (not fx.no_longer_possible.empty()) fxs.push_back("no longer: " + abstracted(list(fx.no_longer_possible, true)));
            if (not fx.appeared.empty()) fxs.push_back("appeared: " + abstracted(list(fx.appeared)));
            std::string joined;
            for (const std::string& f : fxs) joined += (joined.empty() ? "" : "; ") + f;
            abstract_lines.push_back(std::format("{:<20} {:<16} ⇒ {}", abstracted(doing(r)), explain(r, true),
                                                 joined.empty() ? std::string("no observed effect") : joined));
        }
        lines.push_back(std::move(l));
    }
    std::set<std::string, Natural> others;
    for (Line& l : event_lines(tree, ev, others)) lines.push_back(std::move(l));
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
        for (const std::string& n : l.notes) out += std::format("{:28}{}\n", "", n);
    }
    if (lines.size() > shown) out += std::format("  … {} more moments not shown\n", lines.size() - shown);
    out += "\nCONSEQUENCES OF ITS ACTIONS (each instance replaced by its type: what an abstract world model learns)\n";
    if (abstract_lines.empty()) out += "  no action with a known end in this mission\n";
    for (const std::string& a : abstract_lines) out += "  " + a + "\n";
    if (unknown_ends > 0)
        out += std::format("  ({} action{} left out: its end was never seen)\n", unknown_ends, unknown_ends == 1 ? "" : "s");
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
