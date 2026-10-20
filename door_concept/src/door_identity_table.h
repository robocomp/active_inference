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
#include <cmath>
#include <istream>
#include <locale>
#include <optional>
#include <ostream>
#include <span>
#include <string>
#include <string_view>
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
};

inline constexpr std::string_view kIdentityHeader =
    "room,name,ap_x,ap_y,yaw,width,height,depth,exist_logodds,hinge_x,hinge_y,open_x,open_y,hinge_llr,swing_llr,phi,phi_sigma";
inline constexpr std::size_t kIdentityColsOld  = 4;
inline constexpr std::size_t kIdentityColsFull = 17;

/// A transform read before the room existed resolves to |t| ~ 1e13 m (try35): not a place.
inline constexpr float kIdentityMaxPlaceM = 1.0e3f;

struct ParsedIdentityTable
{
    std::vector<IdentityRow> rows;
    bool unkeyed_format = false;   ///< the pre-2026-10-13 (name,ap_x,ap_y) table: refused whole
    int  malformed      = 0;       ///< rows dropped (bad arity, non-door name, unparsable or non-finite place)
};

namespace detail
{
inline bool num(std::string_view v, float& out)
{
    while (not v.empty() and (v.back() == '\r' or v.back() == ' ')) v.remove_suffix(1);
    if (v.empty()) return false;
    const auto [p, ec] = std::from_chars(v.data(), v.data() + v.size(), out);
    return ec == std::errc{} and p == v.data() + v.size() and std::isfinite(out);
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
        float x = 0.f, y = 0.f;
        if ((c.size() != kIdentityColsOld and c.size() != kIdentityColsFull) or c[0].empty()
            or not c[1].starts_with("door_") or not detail::num(c[2], x) or not detail::num(c[3], y)
            or std::hypot(x, y) > kIdentityMaxPlaceM)
        {
            ++out.malformed;
            continue;
        }
        IdentityRow r{std::string(c[0]), std::string(c[1]), Eigen::Vector2f(x, y), {}, {}, {}};
        if (c.size() == kIdentityColsFull)
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
        }
        out.rows.push_back(std::move(r));
    }
    return out;
}

inline void write_identity_table(std::ostream& f, std::span<const IdentityRow> rows)
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
        f << '\n';
    }
}

// ── THE RESTORE DECISION ─────────────────────────────────────────────────────────────────────────────────────────

/// A door node live in the graph at the moment of the re-entry: its name, and its aperture in the ENTERED room's
/// frame when that resolves (a door HELD in the room left behind may not resolve; its name still blocks).
struct LiveDoor
{
    std::string                    name;
    std::optional<Eigen::Vector2f> xy_in_room;
};

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
///  · A door already live is not duplicated: by NAME (one name per doorway; covers the crossed door carried across,
///    including one still HELD in the room left behind for the late carry), or by PLACE — a live door whose aperture
///    lies within the same-hole distance (reacquire radius + the remembered width, as note_identity uses).
inline RestoreDecision plan_restore(std::span<const IdentityRow> rows, std::string_view room, bool recalled,
                                    std::span<const LiveDoor> live, float reacquire_radius_m)
{
    RestoreDecision d;
    if (not recalled) { d.not_recalled = true; return d; }
    for (const auto& r : rows)
    {
        if (r.room != room) continue;
        if (not r.pose.has_value()) { d.skipped.push_back({r.name, SkipReason::NoPose}); continue; }
        bool name_live = false, place_live = false;
        const float same_hole = std::max(0.f, reacquire_radius_m) + std::max(0.f, r.pose->width);
        for (const auto& l : live)
        {
            name_live  = name_live or l.name == r.name;
            place_live = place_live or (l.xy_in_room.has_value() and l.xy_in_room->allFinite()
                                        and (*l.xy_in_room - r.xy).norm() < same_hole);
        }
        // A name restored twice in one table (should not happen: upsert by name) would make two nodes; the first wins.
        for (const auto& k : d.restore) name_live = name_live or k.name == r.name;
        if (name_live)       d.skipped.push_back({r.name, SkipReason::NameLive});
        else if (place_live) d.skipped.push_back({r.name, SkipReason::PlaceLive});
        else                 d.restore.push_back(r);
    }
    return d;
}

/// The existence prior a restored door starts from: the remembered belief when one was written, else `default_L`
/// (the agent passes Existence.BirthLogodds = 1.0 nats, P = 0.73 — the same moderate prior a fresh birth gets).
inline float restore_existence_prior(const IdentityRow& r, float default_L)
{
    return r.exist_logodds.has_value() and std::isfinite(*r.exist_logodds) ? *r.exist_logodds : default_L;
}

inline const char* to_string(SkipReason s)
{
    switch (s)
    {
        case SkipReason::NoPose:    return "no remembered pose (identity-only row)";
        case SkipReason::NameLive:  return "a door of that name is already live";
        case SkipReason::PlaceLive: return "a live door already holds that hole";
    }
    return "?";
}

}   // namespace rc::door
