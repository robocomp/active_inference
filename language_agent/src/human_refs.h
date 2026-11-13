#pragma once
/*
 * human_refs.h — what "that" and "my chair" refer to, for the human at the chat (spec 2026-11-11-human-model,
 * "Resolving that"). Pure: no Qt, no DSR. Nothing here DECIDES: it ranks candidates and the caller always reads the
 * choice back to the person (or asks which) before anything is stored.
 */

#include <algorithm>
#include <cctype>
#include <cmath>
#include <expected>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "object_facts.h"

namespace rc::chat
{

struct RefCandidate { std::string name; std::uint64_t uid = 0; double weight = 0; bool live = false; };

/// "that": the person sees through the robot's camera, so the evidence is (1) being in view, weighted by how centred
/// (cos of the bearing; the |bearing| < half_fov test is the camera's geometry, not a tuning cutoff), (2) being named
/// in the last turns (`recent_names`, newest first: 1, 1/2, 1/3 …), (3) being the last approach target (1). Each source
/// scores in [0, 1] and the scores add. Highest first.
inline std::vector<RefCandidate> that_candidates(const std::vector<ObjectFact>& live, std::span<const std::string> recent_names,
                                                 const std::optional<std::string>& last_target, double half_fov_deg)
{
    std::vector<RefCandidate> out;
    for (const auto& o : live)
    {
        if (is_human_node(o.name)) continue;
        double w = 0;
        if (o.in_current_room and o.bearing_deg.has_value() and std::abs(*o.bearing_deg) < half_fov_deg)
            w += std::cos(*o.bearing_deg * std::numbers::pi / 180.0);
        for (std::size_t i = 0; i < recent_names.size(); ++i)
            if (recent_names[i] == o.name) { w += 1.0 / static_cast<double>(i + 1); break; }
        if (last_target and *last_target == o.name) w += 1.0;
        if (w > 0) out.push_back({o.name, o.uid, w, true});
    }
    std::ranges::stable_sort(out, [](const RefCandidate& a, const RefCandidate& b) { return a.weight > b.weight; });
    return out;
}

/// "my chair" (cls "chair"; "" = "mine", any class): the human's owned objects, perceived ones first (weight 1), then
/// the remembered ones (weight 1/2). Empty = they own none of that class.
inline std::vector<RefCandidate> my_candidates(std::string_view cls, const HumanFacts& h, const std::vector<ObjectFact>& live,
                                               const std::vector<RememberedObject>& remembered)
{
    const std::string want = normalize_subtype(std::string(cls));
    std::vector<RefCandidate> out;
    auto seen = [&](std::uint64_t uid) { return std::ranges::any_of(out, [&](const RefCandidate& c) { return c.uid == uid; }); };
    for (const auto& o : live)
        if (o.uid != 0 and h.owned_uids.contains(o.uid) and (want.empty() or o.subtype == want) and not seen(o.uid))
            out.push_back({o.name, o.uid, 1.0, true});
    for (const auto& r : remembered)
        if (r.uid != 0 and h.owned_uids.contains(r.uid) and (want.empty() or r.subtype == want) and not seen(r.uid))
            out.push_back({r.name, r.uid, 0.5, false});
    return out;
}

/// The class asked by "my chair" / "my chairs" ("chair"), or "" for "mine"; nullopt when the words do not say "my".
inline std::optional<std::string> my_class(std::string w)
{
    std::ranges::transform(w, w.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    while (not w.empty() and (w.back() == ' ' or w.back() == '?' or w.back() == '.' or w.back() == '!')) w.pop_back();
    while (not w.empty() and w.front() == ' ') w.erase(w.begin());
    if (w.starts_with("the ")) w = w.substr(4);
    if (w == "mine" or w == "my things" or w == "my stuff") return std::string{};
    if (w.starts_with("my ")) return normalize_subtype(w.substr(3));
    if (w.starts_with("mi ") or w.starts_with("mis ")) return normalize_subtype(w.substr(w.find(' ') + 1));   // es
    return std::nullopt;
}

/// "go to my chair" (the WHOLE message, as remembered_goto's go_to_object_name): "my chair", normalised to English
/// ("ve a mi silla" → "my chair"), so the fast path can resolve it. nullopt for anything else.
inline std::optional<std::string> go_to_mine(std::string_view text)
{
    std::string s(text);
    std::ranges::transform(s, s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    while (not s.empty() and (s.back() == ' ' or s.back() == '.' or s.back() == '!' or s.back() == '?')) s.pop_back();
    while (not s.empty() and s.front() == ' ') s.erase(s.begin());
    for (const std::string_view h : {"go to ", "go back to ", "vuelve a ", "ve a "})
    {
        if (not s.starts_with(h)) continue;
        const std::string rest = s.substr(h.size());
        if (std::ranges::count(rest, ' ') != 1) return std::nullopt;   // "my chair": one owner word + one class word
        const auto cls = my_class(rest);
        if (not cls.has_value() or cls->empty()) return std::nullopt;
        return "my " + *cls;
    }
    return std::nullopt;
}

// ── After resolution an owned object is followed by its UID: chair_N names are re-issued across restarts (spec call 5),
//    so a name looked up again can land on a different object that now carries it. ──

/// The ONE owned object of the class (cls "" = any): an error sentence when none, or more than one (ask which).
inline std::expected<RefCandidate, std::string> resolve_one_mine(std::string_view cls, const HumanFacts& h,
                                                                 const std::vector<ObjectFact>& live,
                                                                 const std::vector<RememberedObject>& remembered)
{
    const std::string kind = cls.empty() ? std::string("object") : normalize_subtype(std::string(cls));
    const auto c = my_candidates(cls, h, live, remembered);
    if (c.empty()) return std::unexpected("I don't know which " + kind + " is theirs yet (nothing of that kind is linked to "
                                          "the person in my long-term memory)");
    if (c.size() > 1)
    {
        std::string names;
        for (const auto& x : c) names += (names.empty() ? "" : ", ") + x.name;
        return std::unexpected("the person owns more than one " + kind + " (" + names + "): ask them which one");
    }
    return c.front();
}

/// The live name to drive to for an owned object, or why not: it must be perceived, by its uid, in this room.
inline std::expected<std::string, std::string> mine_live_target(const RefCandidate& c, const std::vector<ObjectFact>& live)
{
    if (const auto f = std::ranges::find(live, c.uid, &ObjectFact::uid); c.uid != 0 and f != live.end())
    {
        if (f->in_current_room) return f->name;
        return std::unexpected(f->name + " is theirs but it is in " + (f->room.empty() ? std::string("another room") : f->room) +
                               ", not this room: going there is a mission (plan_mission)");
    }
    if (std::ranges::any_of(live, [&](const ObjectFact& o) { return o.name == c.name; }))
        return std::unexpected(c.name + " is theirs but is not perceived now, and its old name " + c.name +
                               " now belongs to another object: I cannot go to theirs by that name");
    return std::unexpected(c.name + " is theirs but not perceived now; it is only remembered. Going there is a mission: "
                           "tell the person to say \"go to " + c.name + "\"");
}

/// The remembered object's name for the go-and-look card, when its name is not now worn by a live object.
inline std::optional<std::string> mine_goto_name(const RefCandidate& c, const std::vector<ObjectFact>& live)
{
    if (c.live or std::ranges::any_of(live, [&](const ObjectFact& o) { return o.name == c.name; })) return std::nullopt;
    return c.name;
}

/// where_is / find_objects for "my chair(s)": every owned one, each found by its uid (live, else remembered).
inline std::string answer_where_mine(std::string_view cls, const HumanFacts& h, const std::vector<ObjectFact>& live,
                                     const Remembered& mem, const std::string& room, bool* from_memory = nullptr)
{
    if (from_memory) *from_memory = false;
    const auto mine = my_candidates(cls, h, live, mem.objects);
    if (mine.empty())
        return std::format("I don't know which {} is theirs yet: nothing of that kind is linked to the person in my "
                           "long-term memory.", cls.empty() ? std::string("object") : normalize_subtype(std::string(cls)));
    std::string out;
    for (const auto& m : mine)
        if (m.live)
            out += answer_where_is(live, m.name, room) + "\n";   // live names are unique on the graph
        else if (const auto r = std::ranges::find(mem.objects, m.uid, &RememberedObject::uid); r != mem.objects.end())
        {
            out += remembered_block({*r}, mem.now_ms);
            if (from_memory) *from_memory = true;
        }
    return out;
}

}   // namespace rc::chat
