/*
 * door_identity_table.h — the room-keyed door identity table (etc/door_identities.csv) and the RESTORE decision.
 *
 * The table maps, per room, a door NAME to its aperture PLACE in that room's frame. Since 2026-10-20 a row may also
 * carry what is needed to RECREATE the door when the room is re-entered from memory: its aperture yaw and size, the
 * existence belief it had, and its leaf memory (hinge point + opening direction in the room frame, the hinge/swing
 * log-odds, phi and its sigma). A row without the pose (an old 4-column row, or one written at birth before the door
 * was fitted) is an identity only: it can give a re-detected door its name back, but it cannot restore a door.
 *
 * WHY RESTORE (2026-10-20, tourP2): room_1 → room_2 → room_1. ltsm recalls room_1 (same name, same frame, remembered
 * outline) but door_concept recreated only the door just crossed; room_1's other door (door_1) was gone until the
 * tracker saw it again, and "cross door_1" failed with "not in the graph". The knowledge was not lost — it is restored
 * here as a PRIOR, and the normal existence channel then confirms it (seen) or removes it (absent where visible).
 *
 * Pure (Eigen + std): tests/test_door_identity_table.cpp. Parsing uses std::from_chars (locale-free; CLAUDE.md) and
 * writing imbues the classic locale, so the file has decimal POINTS whatever LC_NUMERIC the Qt process runs under.
 */
#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <chrono>
#include <utility>
#include <cmath>
#include <istream>
#include <locale>
#include <numbers>
#include <optional>
#include <ostream>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rc::door
{

/// The aperture's pose and size in the room frame (the door node's RT is (xy, 0) yaw, base on the floor).
struct RestorePose
{
    float yaw = 0.f, width = 0.f, height = 0.f, depth = 0.f;
};

/// The leaf, as the room-frame hint DoorFitter's belief init consumes (leaf_side_hint), plus the phi filter.
struct LeafMemory
{
    Eigen::Vector2f hinge_xy = Eigen::Vector2f::Zero();   ///< room frame
    Eigen::Vector2f open_dir = Eigen::Vector2f::Zero();   ///< room frame: the side the leaf opens to
    float hinge_llr = 0.f, swing_llr = 0.f, phi = 0.f, phi_sigma = 0.f;
};

struct IdentityRow
{
    std::string     room;
    std::string     name;
    Eigen::Vector2f xy = Eigen::Vector2f::Zero();   ///< aperture centre, THAT room's frame
    std::optional<RestorePose> pose;                 ///< absent ⇒ identity only, not restorable
    std::optional<float>       exist_logodds;        ///< the existence belief when written; absent ⇒ unknown
    std::optional<LeafMemory>  leaf;
    /// Task D-C: wall-clock ms (system clock, epoch) at which the door was last DETECTED — the age of `exist_logodds`.
    /// Absent ⇒ unknown age (a 17-column row): the remembered belief is not trusted beyond its identity (see
    /// restore_existence_prior).
    std::optional<std::int64_t> seen_wall_ms;
    /// Task D-C round 2 (tour3C): the aperture centre's σ along its wall when written (m) — how well the remembered PLACE was
    /// known. Absent ⇒ unknown (a row written before the column existed): the restored belief keeps the birth seed's σ.
    std::optional<float> place_sd;
    /// Door-existence model (Task 7): the place table's memory. All three are OPTIONAL columns (a row of 19 or fewer
    /// columns loads with all three absent). `ledger` is ViewLedger::serialise() (one comma-free cell, parsed by the
    /// place table, not here), `visit` the visit counter, `removed_ms` the wall-clock ms at which the phantom was removed.
    std::optional<std::string>  ledger;
    std::optional<int>          visit;
    std::optional<std::int64_t> removed_ms;
};

inline constexpr std::string_view kIdentityHeader =
    "room,name,ap_x,ap_y,yaw,width,height,depth,exist_logodds,hinge_x,hinge_y,open_x,open_y,hinge_llr,swing_llr,phi,phi_sigma,seen_ms,place_sd,ledger,visit,removed_ms";
inline constexpr std::size_t kIdentityColsOld  = 4;
inline constexpr std::size_t kIdentityCols17   = 17;   ///< 2026-10-20 .. 10-21: no seen_ms column (age unknown)
inline constexpr std::size_t kIdentityCols18   = 18;   ///< 2026-10-21 .. 10-22: no place_sd column (place precision unknown)
inline constexpr std::size_t kIdentityCols19   = 19;   ///< up to Task 7: no ledger / visit / removed_ms
inline constexpr std::size_t kIdentityColsFull = 22;

/// A transform read before the room existed resolves to |t| ~ 1e13 m (try35): not a place.
inline constexpr float kIdentityMaxPlaceM = 1.0e3f;

struct ParsedIdentityTable
{
    std::vector<IdentityRow> rows;
    bool unkeyed_format = false;   ///< the pre-2026-10-13 (name,ap_x,ap_y) table: refused whole
    int  malformed      = 0;       ///< rows dropped (bad arity, non-door name, unparsable or non-finite place)
    std::uint64_t next_door_index = 0;   ///< the persisted name counter (`#next_door,N` line); 0 = absent (older file)
};

// ── DOOR NAMES ARE OPAQUE (2026-10-29, door-identity-design part 1) ─────────────────────────────────────────────────
// A door node's name is `door_<N>` with N drawn from ONE monotonic counter that is persisted with this table and never
// goes back: a name is never handed to a second door, in this run or a later one. The name records nothing — not birth
// order, not which hole, not which room: identity across crossings is the PLACE (ghosts, the (room, place) rows), and
// the passage identity is ltsm's. A door that is a door we already know (a ghost re-acquired, an identity row matched by
// place, a restored row, the crossed door carried across) asks for its remembered name; it gets it when no live node
// holds it, otherwise a fresh counter name — and nothing in door_concept depends on which of the two it got.
// (Was: the LOWEST free number, reserved names, and three rename paths that re-made a live node under a remembered
//  name — a name then recorded birth order and timing, drifted, and was reused for a different hole.)

/// The counter line appended after the rows: `#next_door,<N>`. A pre-counter reader drops it as one malformed row.
inline constexpr std::string_view kNextDoorTag = "#next_door";

/// N of a `door_<N>` name (N ≥ 1, digits only); nullopt for anything else.
inline std::optional<std::uint64_t> door_name_index(std::string_view name)
{
    if (not name.starts_with("door_")) return std::nullopt;
    const std::string_view digits = name.substr(5);
    std::uint64_t v = 0;
    if (digits.empty()) return std::nullopt;
    const auto [p, ec] = std::from_chars(digits.data(), digits.data() + digits.size(), v);
    if (ec != std::errc{} or p != digits.data() + digits.size() or v == 0) return std::nullopt;
    return v;
}

/// Raise `next` past every `door_<N>` in `names` (a counter can only move forward).
template <class Names>
void observe_door_names(std::uint64_t& next, const Names& names)
{
    next = std::max<std::uint64_t>(next, 1);
    for (const auto& nm : names)
        if (const auto n = door_name_index(nm); n.has_value()) next = std::max(next, *n + 1);
}

/// The counter a run starts from: the persisted one, raised past every name the table remembers (a file written before
/// the counter existed, or edited by hand, cannot make the counter hand out a remembered name).
inline std::uint64_t next_door_index_floor(const ParsedIdentityTable& t)
{
    std::uint64_t next = std::max<std::uint64_t>(t.next_door_index, 1);
    for (const auto& r : t.rows)
        if (const auto n = door_name_index(r.name); n.has_value()) next = std::max(next, *n + 1);
    return next;
}

/// The name a new door node gets. `preferred_name` (the remembered name of a door we know) when no node in
/// `graph_door_names` (every live `door_*` node) wears it; otherwise `door_<next++>`. `next` is first raised past every
/// live name, so a fresh name can never collide with a node of an earlier run or another writer.
inline std::string allocate_door_name(std::string_view preferred_name, std::span<const std::string> graph_door_names,
                                      std::uint64_t& next)
{
    observe_door_names(next, graph_door_names);
    const bool preferred_free = not preferred_name.empty()
                                and std::ranges::find(graph_door_names, preferred_name) == graph_door_names.end();
    if (preferred_free)
    {
        // A remembered name in use again: the counter must stay past it too.
        if (const auto n = door_name_index(preferred_name); n.has_value()) next = std::max(next, *n + 1);
        return std::string(preferred_name);
    }
    return "door_" + std::to_string(next++);
}

namespace detail
{
inline bool num(std::string_view v, float& out)
{
    while (not v.empty() and (v.back() == '\r' or v.back() == ' ')) v.remove_suffix(1);
    if (v.empty()) return false;
    const auto [p, ec] = std::from_chars(v.data(), v.data() + v.size(), out);
    return ec == std::errc{} and p == v.data() + v.size() and std::isfinite(out);
}
inline bool num_i64(std::string_view v, std::int64_t& out)
{
    while (not v.empty() and (v.back() == '\r' or v.back() == ' ')) v.remove_suffix(1);
    if (v.empty()) return false;
    const auto [p, ec] = std::from_chars(v.data(), v.data() + v.size(), out);
    return ec == std::errc{} and p == v.data() + v.size();
}
inline std::vector<std::string_view> split(std::string_view line)
{
    std::vector<std::string_view> c;
    for (std::size_t p = 0;;)
    {
        const auto q = line.find(',', p);
        c.emplace_back(line.substr(p, q == std::string_view::npos ? std::string_view::npos : q - p));
        if (q == std::string_view::npos) break;
        p = q + 1;
    }
    return c;
}
}   // namespace detail

inline ParsedIdentityTable parse_identity_table(std::istream& in)
{
    ParsedIdentityTable out;
    std::string line;
    if (not std::getline(in, line)) return out;   // empty file
    // A pre-2026-10-13 table (name,ap_x,ap_y) has no room column: its coordinates are in whichever room the agent sat
    // in when each row was written, so they cannot be trusted against any one room.
    if (not line.starts_with("room,")) { out.unkeyed_format = true; return out; }
    while (std::getline(in, line))
    {
        if (line.empty() or line == "\r") continue;
        const auto c = detail::split(line);
        if (c.size() == 2 and c[0] == kNextDoorTag)
        {
            if (std::int64_t n = 0; detail::num_i64(c[1], n) and n > 0)
                out.next_door_index = static_cast<std::uint64_t>(n);
            else
                ++out.malformed;
            continue;
        }
        float x = 0.f, y = 0.f;
        if ((c.size() != kIdentityColsOld and c.size() != kIdentityCols17 and c.size() != kIdentityCols18
             and c.size() != kIdentityCols19 and c.size() != kIdentityColsFull) or c[0].empty()
            or not c[1].starts_with("door_") or not detail::num(c[2], x) or not detail::num(c[3], y)
            or std::hypot(x, y) > kIdentityMaxPlaceM)
        {
            ++out.malformed;
            continue;
        }
        IdentityRow r{std::string(c[0]), std::string(c[1]), Eigen::Vector2f(x, y), {}, {}, {}, {}};
        if (c.size() >= kIdentityCols17)
        {
            // Each group is all-or-nothing: half a pose is no pose.
            if (RestorePose p; detail::num(c[4], p.yaw) and detail::num(c[5], p.width) and detail::num(c[6], p.height)
                               and detail::num(c[7], p.depth) and p.width > 0.f and p.height > 0.f and p.depth > 0.f)
                r.pose = p;
            if (float L = 0.f; detail::num(c[8], L)) r.exist_logodds = L;
            LeafMemory m;
            float hx, hy, ox, oy;
            if (detail::num(c[9], hx) and detail::num(c[10], hy) and detail::num(c[11], ox) and detail::num(c[12], oy)
                and detail::num(c[13], m.hinge_llr) and detail::num(c[14], m.swing_llr) and detail::num(c[15], m.phi)
                and detail::num(c[16], m.phi_sigma))
            {
                m.hinge_xy = {hx, hy};
                m.open_dir = {ox, oy};
                r.leaf = m;
            }
            if (std::int64_t t = 0; c.size() >= kIdentityCols18 and detail::num_i64(c[17], t) and t > 0)
                r.seen_wall_ms = t;
            if (float sd = 0.f; c.size() >= kIdentityCols19 and detail::num(c[18], sd) and sd > 0.f)
                r.place_sd = sd;
            if (c.size() == kIdentityColsFull)
            {
                if (not c[19].empty()) r.ledger = std::string(c[19]);
                if (std::int64_t v = 0; detail::num_i64(c[20], v) and v >= 0 and v <= 2147483647) r.visit = static_cast<int>(v);
                if (std::int64_t t = 0; detail::num_i64(c[21], t) and t > 0) r.removed_ms = t;
            }
        }
        out.rows.push_back(std::move(r));
    }
    return out;
}

inline void write_identity_table(std::ostream& f, std::span<const IdentityRow> rows, std::uint64_t next_door_index = 0)
{
    f.imbue(std::locale::classic());   // never emit a comma decimal separator
    f << kIdentityHeader << '\n';
    const auto opt = [&](bool has, float v) { if (has) f << v; };
    for (const auto& i : rows)
    {
        f << i.room << ',' << i.name << ',' << i.xy.x() << ',' << i.xy.y();
        const bool hp = i.pose.has_value(), hl = i.leaf.has_value();
        const bool he = i.exist_logodds.has_value() and std::isfinite(*i.exist_logodds);
        f << ','; opt(hp, hp ? i.pose->yaw    : 0.f);
        f << ','; opt(hp, hp ? i.pose->width  : 0.f);
        f << ','; opt(hp, hp ? i.pose->height : 0.f);
        f << ','; opt(hp, hp ? i.pose->depth  : 0.f);
        f << ','; opt(he, he ? *i.exist_logodds : 0.f);
        f << ','; opt(hl, hl ? i.leaf->hinge_xy.x() : 0.f);
        f << ','; opt(hl, hl ? i.leaf->hinge_xy.y() : 0.f);
        f << ','; opt(hl, hl ? i.leaf->open_dir.x() : 0.f);
        f << ','; opt(hl, hl ? i.leaf->open_dir.y() : 0.f);
        f << ','; opt(hl, hl ? i.leaf->hinge_llr : 0.f);
        f << ','; opt(hl, hl ? i.leaf->swing_llr : 0.f);
        f << ','; opt(hl, hl ? i.leaf->phi : 0.f);
        f << ','; opt(hl, hl ? i.leaf->phi_sigma : 0.f);
        f << ',';
        if (i.seen_wall_ms.has_value() and *i.seen_wall_ms > 0) f << *i.seen_wall_ms;
        f << ',';
        if (i.place_sd.has_value() and std::isfinite(*i.place_sd) and *i.place_sd > 0.f) f << *i.place_sd;
        f << ',';
        if (i.ledger.has_value() and i.ledger->find_first_of(",\r\n") == std::string::npos) f << *i.ledger;   // a cell must stay a cell
        f << ',';
        if (i.visit.has_value()) f << *i.visit;
        f << ',';
        if (i.removed_ms.has_value() and *i.removed_ms > 0) f << *i.removed_ms;
        f << '\n';
    }
    if (next_door_index > 0) f << kNextDoorTag << ',' << next_door_index << '\n';
}

// ── THE RESTORE DECISION ─────────────────────────────────────────────────────────────────────────────────────────

/// A door node live in the graph at the moment of the re-entry: its name, and its aperture in the ENTERED room's
/// frame when that resolves (a door HELD in the room left behind may not resolve; its name still blocks).
struct LiveDoor
{
    std::string                    name;
    std::optional<Eigen::Vector2f> xy_in_room;
};

/// Values are logged (RestoreSkip a=); NameLive is RETIRED (2026-10-29: a live name no longer blocks a restore — the
/// restored door takes a fresh name) and kept only so PlaceLive stays 2.
enum class SkipReason { NoPose, NameLive, PlaceLive };

struct RestoreDecision
{
    struct Skip { std::string name; SkipReason reason; };
    std::vector<IdentityRow> restore;
    std::vector<Skip>        skipped;
    bool not_recalled = false;   ///< the room is fresh (not a recall): nothing is restored, whatever its name
};

/// Which of `room`'s remembered doors to recreate on entering it.
///  · Only when the room IS a recall (`recalled`): a fresh room sharing a name with old rows restores nothing.
///  · A row without a pose cannot be recreated (old 4-column / birth row) — skipped, the caller logs it.
///  · A door already live is not duplicated: by PLACE — a live door whose aperture lies within the same-hole distance
///    (reacquire radius + the remembered width, as note_identity uses). NOT by name (2026-10-29): names are opaque, a
///    row whose name a live node wears is still restored when its hole is free, under a fresh name (allocate_door_name).
inline RestoreDecision plan_restore(std::span<const IdentityRow> rows, std::string_view room, bool recalled,
                                    std::span<const LiveDoor> live, float reacquire_radius_m)
{
    RestoreDecision d;
    if (not recalled) { d.not_recalled = true; return d; }
    for (const auto& r : rows)
    {
        if (r.room != room) continue;
        if (not r.pose.has_value()) { d.skipped.push_back({r.name, SkipReason::NoPose}); continue; }
        const float same_hole = std::max(0.f, reacquire_radius_m) + std::max(0.f, r.pose->width);
        const bool place_live = std::ranges::any_of(live, [&](const LiveDoor& l)
        { return l.xy_in_room.has_value() and l.xy_in_room->allFinite() and (*l.xy_in_room - r.xy).norm() < same_hole; });
        if (place_live) d.skipped.push_back({r.name, SkipReason::PlaceLive});
        else            d.restore.push_back(r);
    }
    return d;
}

/// Upsert BY PLACE WITHIN `room`: ONE row per hole, under its REMEMBERED (oldest) name.
/// The target is the oldest row of this room that has the same name or lies on the same hole (`same_hole_m` = reacquire
/// radius + the opening's width); later duplicates of it are dropped. Rows are updated IN PLACE, so table order is age.
/// ★THE OLDEST NAME WINS (tourP9, 2026-10-21; supersedes the 2026-10-18 "the live door's name wins"). room_1's main door
///  kept turning into `door_4`: a duplicate birth on door_1's hole (tourP6: the restored door_1 sat at (0,0), the tracker
///  re-detected the real hole and could not reuse the live name) wrote ITS row over door_1's, and every later run
///  re-acquired the hole as door_4. A hole's name is a fact about the building, learnt once; a duplicate does not rename it.
/// ★THE PLACE IS A POSTERIOR, NOT THE LAST WRITE (tourP8 → tourP9). door_2's row moved 0.56 → 0.35 m along its wall: at
///  shutdown the door's fit had degraded (FE 1.60 vs 0.06, existence L = −4, P 0.02) and its estimate replaced a good
///  memory; tourP9's look then aimed 0.2–0.4 m off the hole. The writer is THIS hole's door with probability `weight`
///  (its P(exists)); otherwise the memory stands — so the remembered place moves to the posterior mean
///  xy + weight·(xy_new − xy). A credible door updates its row fully, a dying one barely moves it.
inline bool upsert_row(std::vector<IdentityRow>& rows, const std::string& room, IdentityRow row, float same_hole_m,
                       float weight = 1.0f)
{
    if (room.empty() or not row.name.starts_with("door_") or not row.xy.allFinite() or row.xy.norm() > kIdentityMaxPlaceM)
        return false;
    const auto same = [&](const IdentityRow& i, const Eigen::Vector2f& at, const std::string& nm)
    { return i.room == room and (i.name == nm or (i.xy - at).norm() < same_hole_m); };
    auto it = std::ranges::find_if(rows, [&](const IdentityRow& i) { return same(i, row.xy, row.name); });
    if (it == rows.end())
    {
        row.room = room;
        rows.push_back(std::move(row));
        return true;
    }
    const float a = std::isfinite(weight) ? std::clamp(weight, 0.f, 1.f) : 0.f;
    const auto mix = [a](float o, float n) { return o + a * (n - o); };
    const auto mix_angle = [a](float o, float n) { return o + a * std::remainder(n - o, 2.0f * std::numbers::pi_v<float>); };
    IdentityRow& t = *it;
    const std::string incoming = row.name;
    t.xy = t.xy + a * (row.xy - t.xy);
    // the place's precision moves with the place (Task D-C round 2): a mixed place is as well known as the mix of the two
    if (row.place_sd.has_value())
        t.place_sd = t.place_sd.has_value() ? mix(*t.place_sd, *row.place_sd) : *row.place_sd;
    if (row.pose.has_value())
    {
        if (t.pose.has_value())
            t.pose = RestorePose{mix_angle(t.pose->yaw, row.pose->yaw), mix(t.pose->width, row.pose->width),
                                 mix(t.pose->height, row.pose->height), mix(t.pose->depth, row.pose->depth)};
        else
            t.pose = row.pose;
    }
    if (row.exist_logodds.has_value())
    {
        t.exist_logodds = row.exist_logodds;   // the belief as it is now: it is already a posterior
        t.seen_wall_ms  = row.seen_wall_ms;    // …and its age goes with it (Task D-C)
    }
    if (row.leaf.has_value())
    {
        if (t.leaf.has_value())
        {
            LeafMemory m;
            m.hinge_xy  = t.leaf->hinge_xy + a * (row.leaf->hinge_xy - t.leaf->hinge_xy);
            const Eigen::Vector2f od = t.leaf->open_dir + a * (row.leaf->open_dir - t.leaf->open_dir);
            m.open_dir  = od.norm() > 1e-6f ? Eigen::Vector2f(od.normalized()) : row.leaf->open_dir;
            m.hinge_llr = mix(t.leaf->hinge_llr, row.leaf->hinge_llr);
            m.swing_llr = mix(t.leaf->swing_llr, row.leaf->swing_llr);
            m.phi       = mix(t.leaf->phi, row.leaf->phi);
            m.phi_sigma = mix(t.leaf->phi_sigma, row.leaf->phi_sigma);
            t.leaf = m;
        }
        else
            t.leaf = row.leaf;
    }
    // ONE row per hole: drop every later row of this room that is the incoming name or lies on the target's hole.
    const IdentityRow keep = t;
    const auto first = std::distance(rows.begin(), it);
    for (auto j = rows.begin() + first + 1; j != rows.end();)
        j = (same(*j, keep.xy, keep.name) or (j->room == room and j->name == incoming)) ? rows.erase(j) : j + 1;
    return true;
}

/// A room entered FRESH (not a recall) starts its restorable memory over: the pose, leaf and existence of its rows are
/// dropped, the NAME and PLACE kept (still an identity: a re-detected door gets its name back). etc/door_identities.csv outlives ltsm's memory, so without this a later recall of the same room name
/// could restore an OLDER run's doors as confident phantoms. Returns how many rows lost their restore data.
inline int strip_restorable(std::vector<IdentityRow>& rows, std::string_view room)
{
    int n = 0;
    for (auto& r : rows)
        if (r.room == room and (r.pose.has_value() or r.leaf.has_value() or r.exist_logodds.has_value()))
        {
            r.pose.reset();
            r.leaf.reset();
            r.exist_logodds.reset();
            r.seen_wall_ms.reset();
            r.place_sd.reset();
            ++n;
        }
    return n;
}

/// Room recognition swap (spec 2026-10-22): a room entered FRESH while ltsm had not yet recognised it is PROVISIONAL.
/// When ltsm then posts a LATE recall (recall_late) for the room entered, that room is a recalled entry (its doors
/// are restored) and the provisional room just left is dead: room names are never reused, so its identity rows go
/// (drop_room), not merely stripped to identity. An early recall, or none, drops nothing.
struct RoomEntryPlan { bool recalled = false; bool drop_left = false; };
// `startup_announced` (warm start, 2026-11-01): the announcement is ltsm's START-UP recall of the room the robot starts in
// (common/room_transition/startup_recall.h) — the first room is then a remembered one and its doors are restored.
inline RoomEntryPlan plan_room_entry(bool first_resolution, bool recall_announced, bool recall_late,
                                     std::string_view left_room, std::string_view entered_room, bool startup_announced = false)
{
    RoomEntryPlan p;
    p.recalled = recall_announced and (not first_resolution or startup_announced);
    p.drop_left = p.recalled and recall_late and not left_room.empty() and left_room != entered_room;
    return p;
}

/// Room-transition protocol, case 2 (C2.8): ltsm's recognition proposal for the room the robot is IN, as door_concept sees it.
/// room_concept ACCEPTS by renaming the proposal's parent P to K IN PLACE (same node id) and writing state="accepted" on
/// proposal_<K>; it REJECTS by writing state="rejected" only.
struct ProposalView { std::string name; std::uint64_t parent = 0; std::string state; };

/// What following `current` means this cycle. Identity is the NODE ID: the same id under a new name is a RENAME, never a
/// room change (nothing is released or held), and no `current` edge at all is "no information", not a release.
enum class FollowKind { NoInformation, Same, First, Change };
[[nodiscard]] inline FollowKind room_follow_kind(std::uint64_t followed_id, std::optional<std::uint64_t> current)
{
    if (not current.has_value()) return FollowKind::NoInformation;
    if (followed_id == *current) return FollowKind::Same;
    return followed_id == 0 ? FollowKind::First : FollowKind::Change;
}

struct AdoptionPlan { bool restore = false; bool drop_old = false; bool reseat = false; };

/// Every name THIS agent has held beliefs under, per room node id. The frame change P -> K applies to beliefs formed under a
/// name other than K, whichever of {rename, "accepted"} arrives first (room_concept renames FIRST and writes "accepted"
/// LAST, so the rename is normally seen a cycle earlier). First sight already named K ⇒ no other name ⇒ nothing to move.
class RoomNameHistory
{
public:
    void observe(std::uint64_t room_id, const std::string& name) { names_[room_id].insert(name); }
    [[nodiscard]] std::vector<std::string> others(std::uint64_t room_id, std::string_view name) const
    {
        std::vector<std::string> out;
        if (const auto it = names_.find(room_id); it != names_.end())
            for (const auto& n : it->second) if (n != name) out.push_back(n);
        return out;
    }
private:
    std::unordered_map<std::uint64_t, std::set<std::string>> names_;
};

/// Does `id` belong to `room_id`? Walk the parent chain (door -> wall -> floor -> room, or door -> room), the same chain
/// resolve_wall relies on, stopping at the root (parent 0) or at a node already visited (a malformed cyclic graph).
template <class ParentOf>
[[nodiscard]] bool descends_from(std::uint64_t id, std::uint64_t room_id, ParentOf parent_of)
{
    std::set<std::uint64_t> visited;
    while (id != 0 and visited.insert(id).second)
    {
        id = parent_of(id);
        if (id == room_id) return true;
    }
    return false;
}

/// An ACCEPTED proposal for the current room (parent == its id, named for the name it now wears): that room is RECALLED
/// under K, once (`already_handled`). `other_names` = the names this agent held beliefs under for this room id that are not
/// K (RoomNameHistory::others): their identity rows go (names are never reused) and the beliefs are re-seated by the frame
/// change. Rejected / pending: nothing — the room stays as entered (FRESH). Merging the live doors into K's remembered ones
/// is plan_restore's place test (PlaceLive).
[[nodiscard]] inline AdoptionPlan plan_adoption(std::span<const ProposalView> proposals, std::uint64_t room_id,
                                                std::string_view room_name, std::span<const std::string> other_names,
                                                bool already_handled)
{
    AdoptionPlan p;
    if (already_handled or room_id == 0 or room_name.empty()) return p;
    const std::string want = "proposal_" + std::string(room_name);
    p.restore = std::ranges::any_of(proposals, [&](const ProposalView& v)
    { return v.parent == room_id and v.name == want and v.state == "accepted"; });
    p.drop_old = p.restore and not other_names.empty();
    p.reseat = p.drop_old;
    return p;
}

/// Room-transition protocol, case C1.4: a room ltsm ANNOUNCED (recall_<room>) is a REMEMBERED room, not a new one. Its
/// doors come back from the identity table (plan_restore, the crossed door's face included: it is a row of that room, and
/// a face already live at its place is skipped by place), so no entry mirror is invented for it. Only a proto-room nobody
/// announced is new, and gets the mirror of the door it was entered through.
/// Which room nodes were ANNOUNCED, as a fact about the room's identity that persists: the verdict is taken ONCE, when a
/// room node is first seen, from whether its recall_<name> metaconcept is live THEN (room_concept builds a recalled room by
/// reading that very node, so it is live while the room is born), and kept under the room's node id until that node is gone.
/// It does not depend on the recall still existing later (ltsm retires it), nor on a name latch cleared at a room switch,
/// nor on a stale latch for a room never entered (another room is another id).
class AnnouncedRooms
{
public:
    /// First sight of `room_id` fixes the verdict; later calls ignore `recall_live_now`.
    void observe(std::uint64_t room_id, bool recall_live_now) { verdict_.emplace(room_id, recall_live_now); }
    [[nodiscard]] bool announced(std::uint64_t room_id) const
    { const auto it = verdict_.find(room_id); return it != verdict_.end() and it->second; }
    [[nodiscard]] bool seen(std::uint64_t room_id) const { return verdict_.contains(room_id); }
    /// Forget rooms whose node is gone.
    template <class Pred> void prune(Pred room_exists) { std::erase_if(verdict_, [&](const auto& kv) { return not room_exists(kv.first); }); }
    [[nodiscard]] std::size_t size() const { return verdict_.size(); }
private:
    std::unordered_map<std::uint64_t, bool> verdict_;
};

[[nodiscard]] inline bool entry_mirror_wanted(bool room_is_proto, bool recall_announced)
{
    return room_is_proto and not recall_announced;
}

/// Delete every identity row of `room`; returns the dropped door names (for the decision log).
inline std::vector<std::string> drop_room(std::vector<IdentityRow>& rows, std::string_view room)
{
    std::vector<std::string> names;
    std::erase_if(rows, [&](const IdentityRow& r) { if (r.room != room) return false; names.push_back(r.name); return true; });
    return names;
}

/// The existence prior a restored door starts from: the remembered belief when one was written, else `default_L`
/// (the agent passes Existence.BirthLogodds = 1.0 nats, P = 0.73 — the same moderate prior a fresh birth gets).
/// ★A DIFFERENT CONVENTION FROM GHOST RE-ACQUISITION, on purpose (controller ruling 2026-10-20). A ghost is a door
/// REMOVED for absence that a new detection brings back: its shape is remembered but its existence restarts from
/// BirthLogodds, because the last thing known about it was evidence AGAINST it. A restored door was alive when its room
/// was left — the last thing known is the belief it then held — so that belief is the prior, not a fresh birth's.
///
/// ★Task D-C (rtD1i door_5): A REMEMBERED BELIEF DECAYS TOWARD THE BIRTH PRIOR WITH THE TIME SINCE THE DOOR WAS LAST SEEN.
///  door_5 came back three times at L = −4 (P 0.02) — already below the removal bound — because one bad visit (a shut door
///  read as open, then refuted by the image) wrote −4 and the memory held it as a frozen posterior. A memory is evidence
///  with an AGE: with probability 1 − e^{−h·Δt} the scene the evidence was about has changed since (the door state was
///  redrawn from the prior), so P = e^{−h·Δt}·P_mem + (1 − e^{−h·Δt})·P_birth — the standard change-point (hazard) mixture,
///  in probability, not in log-odds.
///  THE HAZARD'S SOURCE — A BORROWED RATE, stated as such (review I-2): h = kLeafMoveHyperShape / kLeafMoveHyperTimeS =
///  1/600 s, the leaf channel's move-hazard HYPERPRIOR (door_leaf_channel.h: one uncommanded leaf move per 10-min run
///  length). It is not a measured rate of change of a door's existence — this agent has no such model. It is borrowed
///  because the existence channels are conditioned on the leaf (silhouette, contour and frame are rendered at φ), so a
///  verdict is only as current as the leaf state it was formed under. Kept as a copy here so this header stays pure (no
///  door_belief.h); tests/test_door_identity_table.cpp static_asserts it equals the leaf channel's ratio.
///  Δt unknown (a 17-column row, or no stamp) ⇒ the memory's age is unknown ⇒ survival 0 ⇒ the birth prior: the row is
///  still an identity (name + place), but carries no existence belief.
inline constexpr double kRememberedExistenceHazardPerS = 1.0 / 600.0;   // BORROWED: kLeafMoveHyperShape / kLeafMoveHyperTimeS
inline float restore_existence_prior(const IdentityRow& r, float default_L, std::int64_t now_wall_ms)
{
    if (not r.exist_logodds.has_value() or not std::isfinite(*r.exist_logodds))
        return default_L;
    if (not r.seen_wall_ms.has_value())
        return default_L;
    const double dt_s = std::max<double>(0.0, static_cast<double>(now_wall_ms - *r.seen_wall_ms) * 1e-3);
    const double surv = std::exp(-kRememberedExistenceHazardPerS * dt_s);
    const double p_mem = 1.0 / (1.0 + std::exp(-static_cast<double>(*r.exist_logodds)));
    const double p_birth = 1.0 / (1.0 + std::exp(-static_cast<double>(default_L)));
    const double p = surv * p_mem + (1.0 - surv) * p_birth;
    return static_cast<float>(std::log(p / (1.0 - p)));
}

/// THE REMEMBERED LEAF AGES (2026-11-01, warm13: DOOR_2 remembered open at 120 deg from a run 5517 s earlier, restored at its
/// old σ; the door was shut, P(passable) stayed 1.00 while the robot stared at it, and nothing asked to open it). The angle
/// is a STATE: with probability e^{-hΔt} (the same hazard as above — here it is the leaf channel's own move hazard, its
/// proper source) the leaf is still where it was seen, else it was moved since and is anywhere in [0, phi_max] (uniform).
/// The mixture, moment-matched to the (phi, σ) the belief is seeded with. Hinge and swing side are STRUCTURE: not aged.
/// No stamp ⇒ the age is unknown ⇒ the uniform prior.
[[nodiscard]] inline std::pair<float, float> restore_leaf_phi(float phi, float sigma, std::optional<std::int64_t> seen_wall_ms,
                                                              std::int64_t now_wall_ms, float phi_max)
{
    const double surv = seen_wall_ms.has_value()
        ? std::exp(-kRememberedExistenceHazardPerS * std::max<double>(0.0, double(now_wall_ms - *seen_wall_ms) * 1e-3)) : 0.0;
    const double pm = std::max(1e-3, double(phi_max));
    const double u_mean = 0.5 * pm, u_m2 = pm * pm / 3.0;   // uniform on [0, pm]: E[phi], E[phi^2]
    const double mean = surv * double(phi) + (1.0 - surv) * u_mean;
    const double m2 = surv * (double(sigma) * sigma + double(phi) * phi) + (1.0 - surv) * u_m2;
    return {static_cast<float>(mean), static_cast<float>(std::sqrt(std::max(0.0, m2 - mean * mean)))};
}

/// (Task D-C review I-1) THE EXISTENCE A ROW REMEMBERS IS THE EVIDENCE, NOT THE WALL-BAND PRIOR. The core adds the band's
/// log-prior to L every cycle as a REPLACED term (DoorFitState::room_prior_term); a restored instance starts with that term
/// at 0 and the core adds it again, so writing the full L counted the prior twice on every round trip. The row stores
/// L − room_prior_term; the band term re-enters once, from the restored room's polygon.
template <class Inst>
[[nodiscard]] float remembered_logodds(const Inst& inst)
{
    return inst.existence.logodds() - inst.room_prior_term;
}

/// A door with fewer matched frames than this and no fresh detection is not fitted this cycle (process_door_node). A door
/// that is KNOWN without being seen now — the crossed door carried across, a door restored from memory — is given this
/// many so it is fitted (its belief initialised from its node), which is also what lets the existence channel judge it
/// and what lets its pragmatic affordances (approach / open / cross, which need a fitted aperture) be offered.
inline constexpr int kFitWithoutFreshDataFrames = 5;
inline bool fitted_this_cycle(bool has_fresh_data, int matched_frames)
{
    return has_fresh_data or matched_frames >= kFitWithoutFreshDataFrames;
}

/// Seed the instance of a RESTORED door from its row. ★Writes the pose too (tourP6): the node-added path may already
/// have built the instance the moment the node was inserted — BEFORE its RT edge existed — reading its place as (0,0);
/// the restore must not trust that read (nor skip seeding because the instance "already exists"), it must state it.
/// `Inst` is rc::DoorInstance (templated so the harness needs no DSR).
template <class Inst>
void seed_restored_instance(Inst& inst, const IdentityRow& r, float prior_L, float max_L,
                            std::chrono::steady_clock::time_point now)
{
    auto s = inst.model.state();
    s.cx = s.ap_cx = r.xy.x();
    s.cy = s.ap_cy = r.xy.y();
    s.cz = 0.f;
    if (r.pose.has_value())
    {
        s.yaw = s.ap_yaw = r.pose->yaw;
        s.w = r.pose->width; s.h = r.pose->height; s.thickness = r.pose->depth;
    }
    s.phi = 0.f;
    inst.model.set_state(s);
    inst.existence.set_max(max_L);
    inst.existence.set(prior_L);
    inst.existence_seeded = true;
    // Task D-C: the removal decision weighs what the data do SINCE this entry (DoorFitState::existence_entry_L), and the
    // door's age carries on from the row's stamp (a restored door not seen this visit keeps aging).
    inst.existence_entry_L = inst.existence.logodds();
    inst.existence_since_entry = 0.0f;
    // Task D-C round 2 (tour3C): the remembered place's precision seeds the belief's along-wall σ (DoorFit's lazy init).
    inst.remembered_place_sd = r.place_sd;
    inst.last_detection_wall_ms = r.seen_wall_ms.value_or(0);
    inst.matched_frames   = std::max(inst.matched_frames, kFitWithoutFreshDataFrames);
    inst.seed_wh_from_box = true;
    if (r.leaf.has_value())
    {
        inst.leaf_side_hint   = std::make_pair(r.leaf->hinge_xy, r.leaf->open_dir);
        inst.phi_est          = r.leaf->phi;
        inst.phi_sigma        = r.leaf->phi_sigma;
        inst.hinge_llr        = r.leaf->hinge_llr;
        inst.swing_llr        = r.leaf->swing_llr;
        inst.phi_last_t       = now;
        inst.phi_from_command = false;
    }
}

inline const char* to_string(SkipReason s)
{
    switch (s)
    {
        case SkipReason::NoPose:    return "no remembered pose (identity-only row)";
        case SkipReason::NameLive:  return "a door of that name is already live (retired reason)";
        case SkipReason::PlaceLive: return "a live door already holds that hole";
    }
    return "?";
}

}   // namespace rc::door
