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
    std::string name, subtype;
    float x = 0, y = 0, yaw = 0;           ///< room frame
    std::optional<float> w, d, h;          ///< published sizes (metres)
    std::optional<Eigen::Vector2f> standoff;   ///< where the robot stands to look at / approach it
};

/// Everything the evaluator needs from the graph, as plain data.
struct SeeScene
{
    std::string room;
    std::vector<Eigen::Vector2f> polygon;
    std::vector<SeeDoor> doors;
    std::vector<SeeObject> objects;        ///< current room only
    std::optional<Eigen::Vector2f> robot_xy;
    float robot_yaw = 0;
    std::optional<rc::nbv::Sensor> zed;
    float zed_mount_z = 0;
    std::optional<rc::card::LidarRig> helios, bpearl;
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

/// The expected half-gap (the door "at its expected state").
[[nodiscard]] inline float door_half_gap(const SeeDoor& d)
{
    float g = 0;
    for (const auto& st : door_states(d)) g += st.weight * st.half_gap;
    return g;
}

/// The evaluator's World, without `exclude` (the target itself must not occlude itself). `half_gaps` (one per door)
/// overrides each door's gap; null = every door at its expected state.
[[nodiscard]] inline rc::card::World world_for(const SeeScene& s, const std::string& exclude,
                                               const std::vector<float>* half_gaps = nullptr)
{
    rc::card::World w;
    w.polygon = s.polygon;
    for (std::size_t i = 0; i < s.doors.size(); ++i)
        if (const float hg = half_gaps != nullptr ? (*half_gaps)[i] : door_half_gap(s.doors[i]); hg > 0)
            w.gaps.push_back({s.doors[i].xy, hg});
    for (const auto& o : s.objects)
    {
        if (o.name == exclude or not o.w.has_value() or not o.d.has_value()) continue;   // no size ⇒ not an occluder
        rc::nbv::Obstacle ob;
        ob.cx = o.x; ob.cy = o.y; ob.w = *o.w; ob.h = *o.d; ob.yaw = o.yaw;
        ob.z0 = 0.0f; ob.z1 = o.h.value_or(0.0f);
        w.obstacles.emplace_back(o.name, ob);
    }
    w.zed = s.zed;
    w.zed_mount_z = s.zed_mount_z;
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
/// Where the door's span lies: centre ± w/2 along the nearest polygon edge. Does the sight segment a->b cross it?
inline bool door_on_sight_line(const SeeScene& s, const SeeDoor& d, const Eigen::Vector2f& a, const Eigen::Vector2f& b)
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
/// mixed linearly (p_notice, p_recognise); every other door sits at its expected state. limit / occluder / channels
/// come from the heaviest component.
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
    for (std::size_t i = 0; i < s.doors.size(); ++i) gaps[i] = door_half_gap(s.doors[i]);

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
    {
        if (d.width_m <= 0)
            mixed.notes.push_back(std::format("assumption: {}: width unknown, taken as open at a nominal {:.1f} m", d.name, kNominalDoorWidthM));
        else if (door_uncertain(d))
            mixed.notes.push_back(d.p_open.has_value() and *d.p_open >= kSentinelFree
                                      ? std::format("assumption: {}: leaf angle unmeasured, mixed over open (P {:.2f}) and shut", d.name, *d.p_open)
                                      : std::format("assumption: {}: leaf angle and P(open) unmeasured, averaged over a uniform opening", d.name));
    }
    if (dropped > 0)
        mixed.notes.push_back(std::format("assumption: {} further uncertain door(s) on the sight line taken at their expected state", dropped));
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
    t.h.subtype = "";
    t.h.x = pick->x; t.h.y = pick->y; t.h.yaw = pick->yaw;
    if (pick->w.has_value() or pick->d.has_value() or pick->h.has_value())
    {
        rc::card::GeometryPrior g = card ? card->geometry : rc::card::GeometryPrior{};
        if (pick->w) g.w = *pick->w;
        if (pick->d) g.d = *pick->d;
        if (pick->h) g.h = *pick->h;
        g.z0 = 0.0f; g.z1 = g.h;
        t.h.size = g;
    }
    return t;
}

/// "" = from where the robot is; a door gives its centre; an object its standoff (else its centre); else ASK.
[[nodiscard]] inline std::expected<rc::card::Viewpoint, std::string> resolve_from(const SeeScene& s, const std::string& from)
{
    const std::string f = lower_ascii(from);
    if (f.empty() or f == "here" or f == "me" or f == "the robot")
    {
        if (not s.robot_xy.has_value()) return std::unexpected("I do not know where I am in this room right now.");
        return rc::card::Viewpoint{*s.robot_xy, std::nullopt};
    }
    for (const auto& d : s.doors)
        if (lower_ascii(d.name) == f) return rc::card::Viewpoint{d.xy, std::nullopt};
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
[[nodiscard]] inline std::string answer_can_see(const std::optional<rc::card::Card>& card, const std::string& target,
                                                const std::string& from_label, const rc::card::Result& r)
{
    using namespace see_detail;
    if (not card.has_value()) return "I have no detector for that.";
    const std::string at = from_label.empty() ? "from here" : "from " + from_label;
    if (r.limit == rc::card::Limit::Inside)
    {
        const std::string why = r.notes.empty() ? limit_words(r.limit, r.occluder) : r.notes.front();
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
    std::string assumed;
    for (const auto& n : r.notes)
        if (n.starts_with("assumption: ")) assumed += (assumed.empty() ? "" : "; ") + n.substr(12);
    if (not assumed.empty()) out += " Assumed: " + assumed + ".";
    out += calibration_note(*card);
    return out;
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

/// The answer to "where should you stand to see <target>?": the best stand, then the runners-up.
[[nodiscard]] inline std::string answer_where_to_stand(const std::optional<rc::card::Card>& card, const std::string& target,
                                                       const std::vector<rc::card::Stand>& stands,
                                                       const std::vector<Landmark>& lm)
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
    out += calibration_note(*card);
    return out;
}

}   // namespace rc::chat
