#pragma once
/*
 * can_see.h — "can you see X (from Y)?" and "where should you stand to see X?", answered from the PERCEPTION CARDS
 * (common/perception_card): the concept agents' own detection models, evaluated by the one shared evaluator the
 * planners and the dashboard use. PURE: Eigen + the evaluator, no DSR, no Qt (tested in tests/test_can_see.cpp).
 *
 * perception_world.h fills a SeeScene from the graph (main thread); everything here only resolves names, builds the
 * evaluator's World and words the Result.
 *
 * Wording rules (task 12 brief):
 *   · the channel and the limit are named, with the occluder's name when there is one;
 *   · P(notice) and P(recognise) are given separately whenever they differ at the two printed decimals;
 *   · no card for the concept ⇒ "I have no detector for that";
 *   · ricoh-only channels are never claimed (the evaluator does not model the ricoh);
 *   · a card still on its "prior" calibration says so ("my detector for fridges isn't calibrated yet").
 * The adjectives for a probability (likely / unlikely …) are presentation, not a decision: the numbers are always
 * printed next to them and no behaviour hangs on the bands.
 */

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <limits>
#include <cmath>
#include <expected>
#include <format>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "perception_card/evaluator.h"

namespace rc::chat
{

struct SeeDoor
{
    std::string name;
    Eigen::Vector2f xy{0, 0};              ///< centre, room frame
    float width_m = 0;                     ///< 0 = unknown (no gap is cut)
    std::optional<float> phi_rad;          ///< door_phi_rad; nullopt (or < 0 = "not measured") when unknown
    std::optional<float> p_open;           ///< door_open_prob, the fallback when phi is not measured
};

struct SeeObject
{
    std::string name, subtype;             ///< subtype = object_subtype, the CLASS (chair, table, refrigerator ...)
    std::string shape;                     ///< obj_shape ("round" | "square" for a table): the card's subtype key
    float x = 0, y = 0, yaw = 0;           ///< room frame
    std::optional<float> z_origin;         ///< room -> object RT translation z (nullopt: not read)
    bool origin_is_centre = false;         ///< the node origin is the object's CENTRE (bottle), not its base
    std::optional<float> w, d, h;          ///< published sizes (metres)
    std::optional<Eigen::Vector2f> standoff;   ///< where the robot stands to look at / approach it
};

/// The robot's body clearance: rc::card::where_to_stand's kRobotRadiusM (evaluator.h, function-local there).
inline constexpr float kRobotRadiusM = 0.30f;

/// The base height of an object from its RT translation z. Every concept agent's node origin is the object's BASE
/// (table / refrigerator / chair *_scene_graph.cpp: z = 0; cabinet: its carcass z0; hood: its body's underside), except
/// the bottle, whose origin is the cylinder CENTRE (bottle_scene_graph.cpp:293 publishes s.cz = table_top + h/2).
[[nodiscard]] inline float object_base_z(const SeeObject& o, float height)
{
    const float z = o.z_origin.value_or(0.0f);
    return o.origin_is_centre ? z - 0.5f * height : z;
}

/// Everything the evaluator needs from the graph, as plain data.
struct SeeScene
{
    std::string room;
    std::vector<Eigen::Vector2f> polygon;
    std::vector<SeeDoor> doors;
    std::vector<SeeObject> objects;        ///< current room only
    std::optional<Eigen::Vector2f> robot_xy;
    float robot_yaw = 0;                   ///< the BODY x axis' room bearing; Shadow faces body +y, i.e. robot_yaw + pi/2
    std::optional<rc::nbv::Sensor> zed;
    float zed_mount_z = 0;
    std::optional<rc::card::LidarRig> helios, bpearl;
    float room_height = 2.5f;              ///< room_height of the room node: the wall slab top for the renderers
    bool room_height_known = false;        ///< false: 2.5 m is an assumption, and the answers say so
};

/// A door's possible states as (weight, half-gap) pairs; the weights sum to 1.
///   phi measured (>= 0)        one state: a leaf swung by phi covers w·cos(phi) of the doorway along the wall, so the
///                              clear span seen square-on is w·(1 - cos phi): flush = no gap, 90 deg = the whole width.
///   phi not measured, P(open)  two states, the door's own: shut (1 - p) and swung open to pi/2 (p).
///   both unmeasured            uniform phi on [0, pi/2], the 8-node Gauss-Legendre rule the evaluator uses for an
///                              articulated target (evaluator.h kGLx / kGLw, repeated here: they are function-local there).
///   no width                   the width is unknown: the door is treated as OPEN (never cut as a shut wall), at the
///                              nominal width below, and the caller notes it. A modelling assumption, flagged.
struct DoorState { float weight, half_gap; };
inline constexpr float kNominalDoorWidthM = 0.9f;   // only for a door that publishes no width_m
inline constexpr float kSentinelFree = 0.0f;        // door_phi_rad / door_open_prob are -1 when NOT MEASURED (>= 0 is a reading)

[[nodiscard]] inline std::vector<DoorState> door_states(const SeeDoor& d)
{
    constexpr float kHalfPi = std::numbers::pi_v<float> / 2.0f;
    if (d.width_m <= 0) return {{1.0f, 0.5f * kNominalDoorWidthM}};
    const float half_w = 0.5f * d.width_m;
    if (d.phi_rad.has_value() and *d.phi_rad >= kSentinelFree)
        return {{1.0f, half_w * (1.0f - std::cos(std::min(*d.phi_rad, std::numbers::pi_v<float>)))}};
    if (d.p_open.has_value() and *d.p_open >= kSentinelFree)
    {
        const float p = std::clamp(*d.p_open, 0.0f, 1.0f);
        return {{1.0f - p, 0.0f}, {p, half_w}};
    }
    constexpr std::array<float, 4> x{0.1834346425f, 0.5255324099f, 0.7966664774f, 0.9602898565f};
    constexpr std::array<float, 4> w{0.3626837834f, 0.3137066459f, 0.2223810345f, 0.1012285363f};
    std::vector<DoorState> out;
    for (const float sgn : {-1.0f, 1.0f})
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            const float phi = 0.5f * kHalfPi * (1.0f + sgn * x[i]);
            out.push_back({0.5f * w[i], half_w * (1.0f - std::cos(phi))});
        }
    return out;
}

[[nodiscard]] inline bool door_uncertain(const SeeDoor& d) { return door_states(d).size() > 1; }

/// A door at its MOST PROBABLE state (ruling R12), never an average slit: visibility is not linear in the aperture, so
/// the mean half-gap of an uncertain door is a state the door is never in. phi measured -> its gap; P(open) measured ->
/// shut when P(open) < 0.5, else open at pi/2; both unmeasured -> shut (a door is shut until something says otherwise);
/// no width -> open at the nominal width (as door_states).
[[nodiscard]] inline float door_probable_half_gap(const SeeDoor& d)
{
    if (d.width_m <= 0 or not door_uncertain(d)) return door_states(d).front().half_gap;
    if (d.p_open.has_value() and *d.p_open >= kSentinelFree and *d.p_open >= 0.5f) return 0.5f * d.width_m;
    return 0.0f;
}

/// The note naming that assumption for one door ("" when there is nothing to assume).
[[nodiscard]] inline std::string door_probable_note(const SeeDoor& d)
{
    if (d.width_m <= 0)
        return std::format("assumption: {}: width unknown, taken as open at a nominal {:.1f} m", d.name, kNominalDoorWidthM);
    if (not door_uncertain(d)) return {};
    if (d.p_open.has_value() and *d.p_open >= kSentinelFree)
        return std::format("assumption: {}: leaf angle unmeasured, taken at its most probable state, {} (P open {:.2f})", d.name,
                           *d.p_open >= 0.5f ? "open" : "shut", *d.p_open);
    return std::format("assumption: {}: leaf angle and P(open) unmeasured, taken as shut", d.name);
}

/// The evaluator's World, without `exclude` (the target itself must not occlude itself). `half_gaps` (one per door)
/// overrides each door's gap; null = every door at its most probable state.
/// Obstacles stand at their REAL height: base from the RT z (object_base_z), top = base + height_m. With no height_m
/// published the obstacle's height is unknown (z1 <= z0), which nbv reads as infinitely tall: the conservative side.
[[nodiscard]] inline rc::card::World world_for(const SeeScene& s, const std::string& exclude,
                                               const std::vector<float>* half_gaps = nullptr)
{
    rc::card::World w;
    w.polygon = s.polygon;
    for (std::size_t i = 0; i < s.doors.size(); ++i)
        if (const float hg = half_gaps != nullptr ? (*half_gaps)[i] : door_probable_half_gap(s.doors[i]); hg > 0)
            w.gaps.push_back({s.doors[i].xy, hg});
    for (const auto& o : s.objects)
    {
        if (o.name == exclude or not o.w.has_value() or not o.d.has_value()) continue;   // no size ⇒ not an occluder
        rc::nbv::Obstacle ob;
        ob.cx = o.x; ob.cy = o.y; ob.w = *o.w; ob.h = *o.d; ob.yaw = o.yaw;
        ob.z0 = o.h ? object_base_z(o, *o.h) : o.z_origin.value_or(0.0f);
        ob.z1 = o.h ? ob.z0 + *o.h : ob.z0;
        w.obstacles.emplace_back(o.name, ob);
    }
    w.zed = s.zed;
    w.zed_mount_z = s.zed_mount_z;
    w.room_height = s.room_height;
    w.room_height_known = s.room_height_known;
    w.helios = s.helios;
    w.bpearl = s.bpearl;
    return w;
}

struct Target
{
    rc::card::Hypothesis h;
    std::string label;      ///< "fridge_1" or the concept
    std::string name;       ///< the instance name ("" for a bare hypothesis)
};

/// A LiDAR's rig from its model and its pose in the ROBOT BASE frame (`body`). The evaluator composes
/// robot(vp.xy, vp.yaw) * mount itself, so the mount must NOT carry the robot's own pose (perception_world.h reads it
/// body <- sensor, the frame canvas_lidar.cpp uses).
[[nodiscard]] inline rc::card::LidarRig lidar_rig_from_body_mount(const rc::media::SensorModel& m, const Eigen::Isometry3f& body_T_sensor)
{
    return rc::card::LidarRig{m, body_T_sensor};
}

namespace see_detail
{
/// The unit direction of the polygon edge nearest the door (the wall it is cut in), walked in the polygon's order.
inline Eigen::Vector2f door_wall_dir(const SeeScene& s, const SeeDoor& d)
{
    Eigen::Vector2f dir(1, 0);
    float best = std::numeric_limits<float>::max();
    for (std::size_t i = 0; i < s.polygon.size(); ++i)
    {
        const Eigen::Vector2f p = s.polygon[i], q = s.polygon[(i + 1) % s.polygon.size()];
        const Eigen::Vector2f e = q - p;
        const float len2 = e.squaredNorm();
        if (len2 < 1e-9f) continue;
        const float t = std::clamp((d.xy - p).dot(e) / len2, 0.0f, 1.0f);
        const float dist = (d.xy - (p + t * e)).norm();
        if (dist < best) { best = dist; dir = e.normalized(); }
    }
    return dir;
}

/// Where the robot looks "from door_X": the door centre lies ON the polygon edge, where inside/outside is a coin toss
/// (half the doors were refused as "outside the room"), so step inward along the wall normal, toward the current room,
/// by the robot's body radius: the closest the robot's centre can be to that wall. Inward = the interior side of the
/// edge, from the polygon's orientation (left of each edge for a counter-clockwise polygon).
inline Eigen::Vector2f door_viewpoint(const SeeScene& s, const SeeDoor& d)
{
    if (s.polygon.size() < 3) return d.xy;
    float area2 = 0.0f;
    for (std::size_t i = 0; i < s.polygon.size(); ++i)
    {
        const Eigen::Vector2f& p = s.polygon[i];
        const Eigen::Vector2f& q = s.polygon[(i + 1) % s.polygon.size()];
        area2 += p.x() * q.y() - q.x() * p.y();
    }
    const Eigen::Vector2f dir = door_wall_dir(s, d);
    const Eigen::Vector2f left(-dir.y(), dir.x());
    return d.xy + (area2 >= 0.0f ? left : Eigen::Vector2f(-left)) * kRobotRadiusM;
}

/// Where the door's span lies: centre ± w/2 along the nearest polygon edge. Does the sight segment a->b cross it?
inline bool door_on_sight_line(const SeeScene& s, const SeeDoor& d, const Eigen::Vector2f& a, const Eigen::Vector2f& b)
{
    const Eigen::Vector2f dir = door_wall_dir(s, d);
    const float hw = 0.5f * (d.width_m > 0 ? d.width_m : kNominalDoorWidthM);
    const Eigen::Vector2f c = d.xy - dir * hw, e = d.xy + dir * hw;
    const auto cross = [](const Eigen::Vector2f& u, const Eigen::Vector2f& v) { return u.x() * v.y() - u.y() * v.x(); };
    const Eigen::Vector2f r = b - a, q = e - c;
    const float den = cross(r, q);
    if (std::abs(den) < 1e-9f) return false;
    const float t = cross(c - a, q) / den, u = cross(c - a, r) / den;
    return t >= 0 and t <= 1 and u >= 0 and u <= 1;
}
}   // namespace see_detail

/// evaluate() with the doors' uncertainty marginalised (ruling R10): the doors whose span the sight line viewpoint ->
/// target crosses and whose state is uncertain (at most 2, nearest the viewpoint first) are enumerated and the results
/// mixed linearly (p_notice, p_recognise); every other door sits at its most probable state (off the sight line it
/// cannot change the answer, and R12 forbids an average slit). limit / occluder / channels come from the heaviest
/// component. "Assumed" notes name only the doors actually mixed and the width-unknown ones.
[[nodiscard]] inline rc::card::Result evaluate_scene(const rc::card::Card& card, const SeeScene& s, const Target& tg,
                                                     const rc::card::Viewpoint& vp)
{
    constexpr std::size_t kMaxMixedDoors = 2;
    const Eigen::Vector2f goal(tg.h.x, tg.h.y);
    std::vector<std::size_t> relevant;
    std::size_t dropped = 0;
    {
        std::vector<std::size_t> cand;
        for (std::size_t i = 0; i < s.doors.size(); ++i)
            if (door_uncertain(s.doors[i]) and see_detail::door_on_sight_line(s, s.doors[i], vp.xy, goal)) cand.push_back(i);
        std::ranges::sort(cand, {}, [&](std::size_t i) { return (s.doors[i].xy - vp.xy).norm(); });
        dropped = cand.size() > kMaxMixedDoors ? cand.size() - kMaxMixedDoors : 0;
        cand.resize(std::min(cand.size(), kMaxMixedDoors));
        relevant = std::move(cand);
    }
    std::vector<float> gaps(s.doors.size());
    for (std::size_t i = 0; i < s.doors.size(); ++i) gaps[i] = door_probable_half_gap(s.doors[i]);

    std::vector<std::vector<DoorState>> opts;
    for (const auto i : relevant) opts.push_back(door_states(s.doors[i]));
    rc::card::Result mixed, heaviest;
    float best_w = -1.0f;
    std::vector<std::size_t> idx(relevant.size(), 0);
    for (bool more = true; more;)
    {
        float wgt = 1.0f;
        for (std::size_t k = 0; k < relevant.size(); ++k)
        {
            gaps[relevant[k]] = opts[k][idx[k]].half_gap;
            wgt *= opts[k][idx[k]].weight;
        }
        const auto world = world_for(s, tg.name, &gaps);
        auto r = rc::card::evaluate(card, world, tg.h, vp);
        mixed.p_notice += wgt * r.p_notice;
        mixed.p_recognise += wgt * r.p_recognise;
        if (wgt > best_w) { best_w = wgt; heaviest = std::move(r); }
        more = false;
        for (std::size_t k = relevant.size(); k-- > 0;)
            if (++idx[k] < opts[k].size()) { more = true; break; }
            else idx[k] = 0;
    }
    mixed.channels = std::move(heaviest.channels);
    mixed.limit = heaviest.limit; mixed.limit_gain = heaviest.limit_gain; mixed.occluder = heaviest.occluder;
    mixed.notes = std::move(heaviest.notes);
    for (const auto& d : s.doors)
        if (d.width_m <= 0)
            mixed.notes.push_back(std::format("assumption: {}: width unknown, taken as open at a nominal {:.1f} m", d.name, kNominalDoorWidthM));
    for (const auto i : relevant)
    {
        const auto& d = s.doors[i];
        if (d.width_m <= 0) continue;            // already named above
        mixed.notes.push_back(d.p_open.has_value() and *d.p_open >= kSentinelFree
                                  ? std::format("assumption: {}: leaf angle unmeasured, mixed over open (P {:.2f}) and shut", d.name, *d.p_open)
                                  : std::format("assumption: {}: leaf angle and P(open) unmeasured, averaged over a uniform opening", d.name));
    }
    if (dropped > 0)
        mixed.notes.push_back(std::format("assumption: {} further uncertain door(s) on the sight line taken at their most probable state", dropped));
    return mixed;
}

// ── names ──
[[nodiscard]] inline std::string lower_ascii(std::string s)
{
    std::ranges::transform(s, s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

/// "fridge" and "refrigerator" are one class; the cards and the graph say refrigerator.
[[nodiscard]] inline std::string canonical_concept(std::string c)
{
    c = lower_ascii(std::move(c));
    while (not c.empty() and c.front() == ' ') c.erase(c.begin());
    while (not c.empty() and c.back() == ' ') c.pop_back();
    if (c == "fridge") return "refrigerator";
    // Spanish names for the classes the cards cover (the person may ask in Spanish; the cards say English).
    for (const auto& [es, en] : std::initializer_list<std::pair<std::string_view, std::string_view>>{
             {"silla", "chair"}, {"sillas", "chair"}, {"nevera", "refrigerator"}, {"frigor\u00edfico", "refrigerator"},
             {"frigorifico", "refrigerator"}, {"mesa", "table"}, {"mesas", "table"}, {"puerta", "door"}, {"puertas", "door"}})
        if (c == es) return std::string(en);
    if (c.size() > 2 and c.ends_with('s') and not c.ends_with("ss") and not c.contains('_')) c.pop_back();   // plural
    return c;
}

/// `what` = the concept (required), `where` = an instance name or "" . A concept with exactly one instance in the
/// room is that instance; with several the person is asked; with none there is nothing to place.
[[nodiscard]] inline std::expected<Target, std::string> resolve_target(const SeeScene& s, const std::string& what,
                                                                         const std::string& where,
                                                                         const std::optional<rc::card::Card>& card)
{
    const std::string wn = lower_ascii(where);
    const std::string cls = canonical_concept(what.empty() ? where : what);
    const SeeObject* pick = nullptr;
    if (not wn.empty())
    {
        const auto it = std::ranges::find_if(s.objects, [&](const SeeObject& o) { return lower_ascii(o.name) == wn; });
        if (it != s.objects.end()) pick = &*it;
    }
    if (pick == nullptr)
    {
        std::vector<const SeeObject*> same;
        for (const auto& o : s.objects)
            if (lower_ascii(o.subtype) == cls or lower_ascii(o.name) == cls) same.push_back(&o);
        if (same.size() > 1)
        {
            std::string names;
            for (const auto* o : same) names += (names.empty() ? "" : ", ") + o->name;
            return std::unexpected(std::format("ASK: which {} do you mean ({})?", cls, names));
        }
        if (same.empty())
            return std::unexpected(std::format("I have not perceived any {} in this room, so I cannot say what it "
                                               "would take to see one.", cls));
        pick = same.front();
    }
    Target t;
    t.name = pick->name;
    t.label = pick->name;
    t.h.concept_ = canonical_concept(pick->subtype.empty() ? cls : pick->subtype);
    // The card's subtype key for this instance: its obj_shape (table: round | square), else its object_subtype, when
    // the card has an entry under that name; else "" (the card's default entry).
    t.h.subtype = "";
    if (card)
        for (const std::string* k : {&pick->shape, &pick->subtype})
            if (not k->empty() and card->subtypes.contains(*k)) { t.h.subtype = *k; break; }
    t.h.x = pick->x; t.h.y = pick->y; t.h.yaw = pick->yaw;
    // The object where it IS, at its real height: base from the RT z (object_base_z: bottle origin = its centre), top =
    // base + height_m. Without a published height the card's own span is kept (its length, and its base when the pose
    // gives none) -- so a hood keeps its hanging span and a bottle on a table stays on the table.
    if (pick->w.has_value() or pick->d.has_value() or pick->h.has_value() or pick->z_origin.has_value())
    {
        rc::card::GeometryPrior g = card ? card->geometry : rc::card::GeometryPrior{};
        if (pick->w) g.w = *pick->w;
        if (pick->d) g.d = *pick->d;
        const float span = pick->h ? *pick->h : (g.z1 > g.z0 ? g.z1 - g.z0 : g.h);
        g.h = span;
        if (pick->z_origin) g.z0 = object_base_z(*pick, span);
        g.z1 = g.z0 + span;
        t.h.size = g;
    }
    return t;
}

/// "" = from where the robot is, at the heading it HAS (ruling R11: body yaw + pi/2, Shadow faces body +y); a door gives
/// a point one robot radius inside the room from its centre (see_detail::door_viewpoint); an object its standoff (else
/// its centre); else ASK. A named viewpoint takes best heading (yaw unset): the robot would arrive facing the target.
[[nodiscard]] inline std::expected<rc::card::Viewpoint, std::string> resolve_from(const SeeScene& s, const std::string& from)
{
    const std::string f = lower_ascii(from);
    if (f.empty() or f == "here" or f == "me" or f == "the robot")
    {
        if (not s.robot_xy.has_value()) return std::unexpected("I do not know where I am in this room right now.");
        return rc::card::Viewpoint{*s.robot_xy, s.robot_yaw + std::numbers::pi_v<float> / 2.0f};
    }
    for (const auto& d : s.doors)
        if (lower_ascii(d.name) == f) return rc::card::Viewpoint{see_detail::door_viewpoint(s, d), std::nullopt};
    for (const auto& o : s.objects)
        if (lower_ascii(o.name) == f) return rc::card::Viewpoint{o.standoff.value_or(Eigen::Vector2f(o.x, o.y)), std::nullopt};
    return std::unexpected("ASK: which place do you mean?");
}

// ── wording ──
namespace see_detail
{
inline std::string sensor_words(const std::string& s)
{
    if (s == "zed") return "my camera (the zed)";
    if (s == "helios") return "the helios lidar";
    if (s == "bpearl") return "the bpearl lidar";
    return s;
}

inline std::string limit_words(rc::card::Limit l, const std::string& occluder)
{
    using L = rc::card::Limit;
    switch (l)
    {
        case L::None:         return "";
        case L::OutOfView:    return "it is outside the field of view";
        case L::BehindWall:   return "a wall is in the way";
        case L::Occluded:     return occluder.empty() ? "something is in the way" : "it is hidden behind " + occluder;
        case L::TooSmall:     return "it is too small at that distance";
        case L::TooClose:     return "it is too close for the sensor";
        case L::EdgeOn:       return "it is seen edge-on";
        case L::BetweenRings: return "it falls between the lidar's beams";
        case L::BelowFan:     return "it is below the lidar's beams";
        case L::AboveFan:     return "it is above the lidar's beams";
        case L::DeadZone:     return "it is in the lidar's dead zone";
        case L::NotModelled:  return "I have no model of how I would sense it";
        case L::Inside:       return "that spot is not somewhere I can stand";
        case L::Range:        return "it is out of the sensor's range";
    }
    return "";
}

inline std::string p2(float p) { return std::format("{:.2f}", std::clamp(p, 0.0f, 1.0f)); }

inline std::string likelihood(float p)
{
    if (p >= 0.85f) return "very likely";
    if (p >= 0.6f) return "probably";
    if (p >= 0.35f) return "maybe";
    if (p >= 0.1f) return "unlikely";
    return "almost certainly not";
}

inline std::string plural(const std::string& cls) { return cls.ends_with('s') ? cls : cls + "s"; }

/// The channel that carries the answer: the highest-p channel the evaluator really ran (never the ricoh, never a
/// NotModelled one).
inline const rc::card::ChannelResult* lead_channel(const rc::card::Result& r)
{
    const rc::card::ChannelResult* best = nullptr;
    for (const auto& c : r.channels)
    {
        if (c.sensor == "ricoh" or c.limit == rc::card::Limit::NotModelled) continue;
        if (best == nullptr or c.p > best->p) best = &c;
    }
    return best;
}

/// Why an Inside result is infeasible, as the evaluator said it: the feasibility note evaluate() wrote ("viewpoint
/// outside the room polygon", "viewpoint inside X", "target inside a wall"), not whatever note happens to come first.
inline std::string inside_reason(const rc::card::Result& r)
{
    for (const auto& n : r.notes)
        if (n.starts_with("viewpoint outside") or n.starts_with("viewpoint inside") or n.starts_with("target inside")) return n;
    return limit_words(r.limit, r.occluder);
}

inline std::string assumed_words(const std::vector<std::string>& notes)
{
    std::string assumed;
    for (const auto& n : notes)
        if (n.starts_with("assumption: ")) assumed += (assumed.empty() ? "" : "; ") + n.substr(12);
    return assumed.empty() ? std::string{} : " Assumed: " + assumed + ".";
}

inline std::string calibration_note(const rc::card::Card& c)
{
    return c.calibration == "prior"
               ? std::format(" (My detector for {} isn't calibrated yet, so treat the numbers as rough.)", plural(c.concept_))
               : std::string{};
}
}   // namespace see_detail

/// No card for the concept: either no agent has published ANY card (the files were not found: say where we looked) or
/// the cards exist and none is for this concept.
[[nodiscard]] inline std::string answer_no_card(bool any_cards, const std::string& root)
{
    return any_cards ? "I have no detector for that."
                     : std::format("I couldn't find any perception cards (looked in {}).", root);
}

/// The answer to "can you see <target> from <from>?". `card` nullopt = no agent published a card for the concept.
/// `turned` (from here only): the same look at best heading; when its printed numbers differ and it is better, the
/// answer adds what turning to face the target would give (ruling R11).
[[nodiscard]] inline std::string answer_can_see(const std::optional<rc::card::Card>& card, const std::string& target,
                                                const std::string& from_label, const rc::card::Result& r,
                                                const rc::card::Result* turned = nullptr)
{
    using namespace see_detail;
    if (not card.has_value()) return "I have no detector for that.";
    const std::string at = from_label.empty() ? "from here" : "from " + from_label;
    if (r.limit == rc::card::Limit::Inside)
    {
        const std::string why = inside_reason(r);
        return std::format("I cannot judge {} {}: {}.{}", target, at, why, calibration_note(*card));
    }
    const auto* lead = lead_channel(r);
    if (lead == nullptr)
        return std::format("I have no model of how my sensors would pick up {} {}.{}", target, at, calibration_note(*card));

    std::string out;
    if (p2(r.p_notice) != p2(r.p_recognise))
        out = std::format("{} I would notice that something is there {} (P {}), but recognising it as {} is {} (P {}).",
                          "Partly.", at, p2(r.p_notice), target, likelihood(r.p_recognise), p2(r.p_recognise));
    else
        out = std::format("{}, I {} see {} {} (P {}).", r.p_recognise >= 0.5f ? "Yes" : "Probably not",
                          r.p_recognise >= 0.5f ? "can" : "cannot", target, at, p2(r.p_recognise));
    out += std::format(" The channel that decides it is {}", sensor_words(lead->sensor));
    // The limit of the lead channel itself (r.limit is the best identity channel's — usually the same one).
    const auto lim = lead->limit != rc::card::Limit::None ? lead->limit : r.limit;
    const std::string occ = lead->occluder.empty() ? r.occluder : lead->occluder;
    if (lim != rc::card::Limit::None) out += std::format("; what limits it: {}", limit_words(lim, occ));
    out += ".";
    if (turned != nullptr and turned->limit != rc::card::Limit::Inside and turned->p_recognise > r.p_recognise
        and (p2(turned->p_recognise) != p2(r.p_recognise) or p2(turned->p_notice) != p2(r.p_notice)))
        out += p2(turned->p_notice) != p2(turned->p_recognise)
                   ? std::format(" If I turn to face it, P recognise {} (P notice {}).", p2(turned->p_recognise), p2(turned->p_notice))
                   : std::format(" If I turn to face it, P {}.", p2(turned->p_recognise));
    out += assumed_words(r.notes);
    out += calibration_note(*card);
    return out;
}

/// The whole "can you see <target> from <from>?" path: the viewpoint, the evaluation (doors marginalised), and for
/// "from here" a second look at best heading for the "if I turn" clause. Returns the reply text.
[[nodiscard]] inline std::string can_see_reply(const rc::card::Card& card, const SeeScene& s, const Target& tg, const std::string& from)
{
    const auto vp = resolve_from(s, from);
    if (not vp.has_value()) return vp.error();
    const auto r = evaluate_scene(card, s, tg, *vp);
    if (not vp->yaw.has_value()) return answer_can_see(card, tg.label, from, r);
    const auto best = evaluate_scene(card, s, tg, rc::card::Viewpoint{vp->xy, std::nullopt});
    return answer_can_see(card, tg.label, from, r, &best);
}

/// A named thing to say a place relative to.
struct Landmark { std::string name; Eigen::Vector2f xy; };

[[nodiscard]] inline std::vector<Landmark> landmarks_of(const SeeScene& s)
{
    std::vector<Landmark> out;
    for (const auto& d : s.doors) out.push_back({d.name, d.xy});
    for (const auto& o : s.objects) out.push_back({o.name, {o.x, o.y}});
    return out;
}

/// "1.2 m from door_1, facing fridge_1"; the nearest landmark that is not the target itself.
[[nodiscard]] inline std::string place_words(const Eigen::Vector2f& xy, const std::vector<Landmark>& lm, const std::string& target)
{
    const Landmark* best = nullptr;
    float bd = 0;
    for (const auto& l : lm)
    {
        if (l.name == target) continue;
        const float d = (l.xy - xy).norm();
        if (best == nullptr or d < bd) { best = &l; bd = d; }
    }
    return best == nullptr ? std::format("at ({:.1f}, {:.1f}) in the room", xy.x(), xy.y())
                           : std::format("{:.1f} m from {}", bd, best->name);
}

/// The answer to "where should you stand to see <target>?": the best stand, then the runners-up. `notes` = the
/// assumptions the stand search made (each uncertain door's most probable state).
[[nodiscard]] inline std::string answer_where_to_stand(const std::optional<rc::card::Card>& card, const std::string& target,
                                                       const std::vector<rc::card::Stand>& stands,
                                                       const std::vector<Landmark>& lm,
                                                       const std::vector<std::string>& notes = {})
{
    using namespace see_detail;
    if (not card.has_value()) return "I have no detector for that.";
    if (stands.empty())
        return std::format("I cannot find a spot in this room from which I could see {}.{}", target, calibration_note(*card));
    const auto& top = stands.front();
    std::string out = std::format("The best place to see {} is {}, looking at it (P recognise {}, P notice {})", target,
                                  place_words(top.xy, lm, target), p2(top.r.p_recognise), p2(top.r.p_notice));
    if (const auto* lead = lead_channel(top.r); lead != nullptr)
    {
        out += std::format(", mostly through {}", sensor_words(lead->sensor));
        if (lead->limit != rc::card::Limit::None)
            out += std::format(" (what still limits it there: {})", limit_words(lead->limit, lead->occluder));
    }
    out += ".";
    if (stands.size() > 1)
    {
        out += " Other spots: ";
        for (std::size_t i = 1; i < stands.size(); ++i)
            out += std::format("{}{} (P {})", i > 1 ? "; " : "", place_words(stands[i].xy, lm, target), p2(stands[i].r.p_recognise));
        out += ".";
    }
    out += assumed_words(notes);
    out += calibration_note(*card);
    return out;
}

/// The whole "where should you stand to see <target>?" path (ruling R12): every door at its most probable state, the
/// shared where_to_stand search, and the assumption named.
[[nodiscard]] inline std::string where_to_stand_reply(const rc::card::Card& card, const SeeScene& s, const Target& tg, int k = 3)
{
    std::vector<float> gaps;
    std::vector<std::string> notes;
    for (const auto& d : s.doors)
    {
        gaps.push_back(door_probable_half_gap(d));
        if (auto n = door_probable_note(d); not n.empty()) notes.push_back(std::move(n));
    }
    const auto stands = rc::card::where_to_stand(card, world_for(s, tg.name, &gaps), tg.h, k);
    return answer_where_to_stand(card, tg.label, stands, landmarks_of(s), notes);
}

}   // namespace rc::chat
