#pragma once
/*
 * object_facts.h — what the chat may say about OBJECTS, as plain data and pure text (no Qt, no DSR; tested in
 * tests/test_objects.cpp). WorldDigest fills ObjectFact from the graph; everything here only formats and filters.
 *
 * WHERE OBJECTS COME FROM (survey 2026-10-12). Each concept agent publishes its own beliefs, so what exists in the
 * graph depends on which agents are running:
 *   chair_concept, table_concept, bottle_concept, cabinet_concept, refrigerator_concept, hood_concept
 *       → node type "object", name "<class>_<N>", object_subtype = class ("chair", "table", "bottle", "cabinet",
 *         "refrigerator", "hood"), RT edge from the ROOM node (the room it was born in), width_m / depth_m /
 *         height_m (metres; a round table/bottle publishes its diameter as both width and depth), mesh_color_rgb
 *         (DISPLAY-ONLY chromaticity tint — not a colour name; not used here). No existence probability is
 *         published on the node (obj_exist_prob is registered but no producer writes it).
 *   human_concept  → node type "person" (person_N), RT from the room, no size.          Listed as subtype "person".
 *   residual_concept → node type "obstacle": unexplained clutter, no class.             Counted, never listed.
 *   door_concept   → type "object" named door_*: doors have their own section.          Excluded here.
 *   affordances (aff_*), intentions, metaconcepts (recall_*, kitchen), rooms, walls       Excluded.
 *
 * Only the CURRENT room is perceived: objects hanging from another room node are not listed in the digest (the
 * where_is tool still says which room they belong to).
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <numbers>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace rc::chat
{

struct ObjectFact
{
    std::uint64_t id = 0;
    std::string name;      ///< e.g. chair_1
    std::string subtype;   ///< object_subtype, else the name's prefix
    std::string room;      ///< the room it hangs from ("" = none resolvable)
    bool in_current_room = false;
    std::optional<double> dist_m{};      ///< from the robot (body frame), when the transform resolves
    std::optional<double> bearing_deg{}; ///< 0 = ahead, + = left
    std::optional<float> width_m{}, depth_m{}, height_m{};
    std::string approach_sides{};        ///< its approach affordance's sides with labels as seen now (approach.h sides_line)
    std::string shape{};                 ///< obj_shape ("round" | "square" for a table top), "" = not stated
    std::optional<float> shape_prob{};   ///< obj_shape_prob: P(shape), 0.5 = undecided
};

/// Body frame: x right, y FORWARD. Bearing 0 = straight ahead, + = to the left. (Doors use the same words.)
inline std::string bearing_words(double deg)
{
    const double a = std::abs(deg);
    if (a <= 20.0) return "ahead";
    if (a >= 150.0) return "behind";
    const char* side = deg > 0 ? "left" : "right";
    if (a <= 60.0) return std::format("ahead-{}", side);
    if (a <= 120.0) return std::format("to the {}", side);
    return std::format("behind-{}", side);
}

/// The robot's measured velocity, robot frame (robot_concept writes it on the robot node from odometry): advance
/// + forward, angular + left (CCW, rad/s). Numbers, not a moving/still verdict — any cutoff would be a threshold;
/// the reading's age is said so a stale one is not mistaken for the present.
inline std::string motion_line(float advance_mps, float side_mps, float angular_rps, std::uint64_t age_ms)
{
    const double turn_deg = std::abs(angular_rps) * 180.0 / std::numbers::pi;
    const std::string turn = angular_rps == 0.f ? "not turning"
                           : std::format("turning {} {:.0f} deg/s", angular_rps > 0 ? "left" : "right", turn_deg);
    return std::format("robot motion now (odometry, {} ms old): {} {:.2f} m/s, sideways {:.2f} m/s, {}", age_ms,
                       advance_mps < 0 ? "backward" : "forward", std::abs(advance_mps), std::abs(side_mps), turn);
}

/// "table_1 (table) 2.1 m, +35 deg ahead-left, 1.20x0.70x0.75 m, square top (p 0.93)"
inline std::string object_line(const ObjectFact& o, bool with_subtype = true)
{
    std::string s = o.name;
    if (with_subtype and not o.subtype.empty() and not o.name.starts_with(o.subtype)) s += " (" + o.subtype + ")";
    if (o.dist_m.has_value() and o.bearing_deg.has_value())
        s += std::format(" {:.1f} m, {:+.0f} deg {}", *o.dist_m, *o.bearing_deg, bearing_words(*o.bearing_deg));
    else s += " position unknown";
    if (o.width_m.has_value() and o.depth_m.has_value() and o.height_m.has_value())
        s += std::format(", {:.2f}x{:.2f}x{:.2f} m", *o.width_m, *o.depth_m, *o.height_m);
    if (not o.approach_sides.empty()) s += ", approach sides: " + o.approach_sides;
    if (not o.shape.empty())
    {
        s += ", " + o.shape + (o.subtype == "table" ? " top" : " shape");
        if (o.shape_prob.has_value()) s += std::format(" (p {:.2f})", *o.shape_prob);
    }
    return s;
}

/// Nearest first; objects without a position last, by name.
inline void sort_by_distance(std::vector<ObjectFact>& v)
{
    std::ranges::sort(v, [](const ObjectFact& a, const ObjectFact& b)
    {
        if (a.dist_m.has_value() != b.dist_m.has_value()) return a.dist_m.has_value();
        if (a.dist_m.has_value() and *a.dist_m != *b.dist_m) return *a.dist_m < *b.dist_m;
        return a.name < b.name;
    });
}

/// The STATE's OBJECTS section: counts per subtype in the current room + the n nearest. Kept small on purpose.
inline std::string objects_section(std::vector<ObjectFact> all, int n_nearest, int n_obstacles)
{
    std::vector<ObjectFact> here;
    for (auto& o : all)
        if (o.in_current_room) here.push_back(std::move(o));
    sort_by_distance(here);
    std::map<std::string, int> counts;
    for (const auto& o : here) ++counts[o.subtype];
    std::string s = "objects perceived in the current room (only this room is perceived):";
    if (here.empty()) s += " none\n";
    else
    {
        std::string c;
        for (const auto& [k, n] : counts) c += std::format("{}{} {}", c.empty() ? " " : ", ", n, k);
        s += c + "\n";
        const int n = std::min<int>(n_nearest, static_cast<int>(here.size()));
        s += std::format("  nearest {}{}:\n", n, n < static_cast<int>(here.size()) ? " (find_objects lists the rest)" : "");
        for (int i = 0; i < n; ++i) s += "  - " + object_line(here[i]) + "\n";
    }
    if (n_obstacles > 0) s += std::format("unidentified obstacles (clutter no object model explains): {}\n", n_obstacles);
    return s;
}

/// Lower-case, singular, English class name from what a person or the model might type. Unknown words pass through
/// (lower-cased, plural 's' dropped) — an unknown class simply matches nothing.
inline std::string normalize_subtype(std::string w)
{
    std::ranges::transform(w, w.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    while (not w.empty() and (w.back() == ' ' or w.back() == '?' or w.back() == '.')) w.pop_back();
    while (not w.empty() and w.front() == ' ') w.erase(w.begin());
    static const std::map<std::string, std::string> syn{
        {"fridge", "refrigerator"}, {"fridges", "refrigerator"}, {"refrigerators", "refrigerator"},
        {"nevera", "refrigerator"}, {"frigorifico", "refrigerator"}, {"frigorífico", "refrigerator"},
        {"cupboard", "cabinet"}, {"cupboards", "cabinet"}, {"armario", "cabinet"},
        {"silla", "chair"}, {"sillas", "chair"}, {"mesa", "table"}, {"mesas", "table"},
        {"botella", "bottle"}, {"botellas", "bottle"}, {"campana", "hood"}, {"extractor", "hood"},
        {"people", "person"}, {"persons", "person"}, {"human", "person"}, {"humans", "person"},
        {"persona", "person"}, {"personas", "person"}, {"man", "person"}, {"woman", "person"},
        {"object", ""}, {"objects", ""}, {"thing", ""}, {"things", ""}, {"all", ""}, {"any", ""},
    };
    if (const auto it = syn.find(w); it != syn.end()) return it->second;
    if (w.size() > 3 and w.ends_with("s")) w.pop_back();
    return w;
}

/// find_objects(subtype?) — the current room's objects of that class (all classes when empty), nearest first.
/// An unknown or absent class is an empty list with a sentence the model can repeat, never an error.
inline std::string answer_find_objects(std::vector<ObjectFact> all, const std::string& subtype_raw,
                                       const std::string& room)
{
    const std::string want = normalize_subtype(subtype_raw);
    std::vector<ObjectFact> hit;
    for (auto& o : all)
        if (o.in_current_room and (want.empty() or o.subtype == want)) hit.push_back(std::move(o));
    sort_by_distance(hit);
    const std::string what = want.empty() ? std::string("object") : want;
    if (hit.empty())
        return std::format("No {} perceived in the current room ({}). This does not mean there is none: it may be "
                           "unseen, or the agent that recognises it may not be running.",
                           what, room.empty() ? std::string("unknown") : room);
    std::string s = std::format("{} {}{} perceived in {}:\n", hit.size(), what, hit.size() == 1 ? "" : "s",
                                room.empty() ? std::string("the current room") : room);
    for (const auto& o : hit) s += "- " + object_line(o) + "\n";
    return s;
}

/// where_is(name) — by exact node name, else by class (every object of it, any room, nearest first).
inline std::string answer_where_is(std::vector<ObjectFact> all, const std::string& name_raw, const std::string& room)
{
    std::string name = name_raw;
    std::ranges::transform(name, name.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    while (not name.empty() and name.back() == ' ') name.pop_back();
    if (name.starts_with("the ")) name = name.substr(4);
    const auto describe = [&](const ObjectFact& o)
    {
        if (o.in_current_room) return object_line(o) + std::format(" — in the current room ({})", room);
        return o.name + (o.room.empty() ? std::string(" — room unknown")
                                        : " — in " + o.room + ", not the room the robot is in (not perceived from here)");
    };
    for (const auto& o : all)
        if (o.name == name) return describe(o);
    const std::string want = normalize_subtype(name);
    std::vector<ObjectFact> hit;
    for (auto& o : all)
        if (not want.empty() and o.subtype == want) hit.push_back(std::move(o));
    if (hit.empty())
        return std::format("'{}' is not perceived. The robot has not seen it, or the agent that recognises it is not "
                           "running; that does not mean it is not there.", name_raw);
    std::ranges::stable_partition(hit, [](const ObjectFact& o) { return o.in_current_room; });
    sort_by_distance(hit);
    std::string s;
    for (const auto& o : hit) s += "- " + describe(o) + "\n";
    return s;
}

/// Births and removals between two snapshots of the current room's objects (by node id). A change of room is not a
/// change of the world: the caller resets the baseline instead of calling this.
struct ObjectChanges
{
    std::vector<ObjectFact> born;
    std::vector<std::string> gone;
    [[nodiscard]] bool empty() const { return born.empty() and gone.empty(); }
};

inline ObjectChanges diff_objects(const std::map<std::uint64_t, ObjectFact>& before,
                                  const std::map<std::uint64_t, ObjectFact>& after)
{
    ObjectChanges c;
    for (const auto& [id, o] : after)
        if (not before.contains(id)) c.born.push_back(o);
    for (const auto& [id, o] : before)
        if (not after.contains(id)) c.gone.push_back(o.name);
    sort_by_distance(c.born);
    std::ranges::sort(c.gone);
    return c;
}

/// "new: chair_2 (2.1 m, ahead-left); gone: bottle_1" — "" when nothing changed.
inline std::string changes_line(const ObjectChanges& c)
{
    std::string s;
    if (not c.born.empty())
    {
        s += "new: ";
        for (std::size_t i = 0; i < c.born.size(); ++i)
        {
            const auto& o = c.born[i];
            s += (i ? ", " : "") + o.name;
            if (o.dist_m.has_value() and o.bearing_deg.has_value())
                s += std::format(" ({:.1f} m, {})", *o.dist_m, bearing_words(*o.bearing_deg));
        }
    }
    if (not c.gone.empty())
    {
        s += s.empty() ? "gone: " : "; gone: ";
        for (std::size_t i = 0; i < c.gone.size(); ++i) s += (i ? ", " : "") + c.gone[i];
    }
    return s;
}

}   // namespace rc::chat
