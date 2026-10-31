// evaluator.h -- the shared perception-card evaluator: (card, world facts, HYPOTHETICAL object, viewpoint) ->
// P(detect) and the factor that limits it. Task 3 = the camera half; Task 4 adds the LiDAR half (LidarRig and the
// helios/bpearl fields are declared here for it and unused until then).
//
// Geometry is NOT reimplemented: it is common/nbv/viewpoint_score.h (rc::nbv), the same code the planners call, so
// the answer a planner gets and the answer the dashboard shows are one function. Per level (spec §2):
//   coverage     = visible_fraction (frustum ^ walls ^ named obstacles)
//   countable    x the envelope, expected_p_detect over predicted_fill_axes / predicted_fill_sigma
//   articulated  x the card's configuration terms at phi, or marginalised over [0, phi_max] when phi is unset (leaf_obliquity; resolvability / kappa need a MEASUREMENT a
//                hypothetical object does not have, so they stay at their neutral 1)
//   animate      x Gauss-Hermite over the position, sigma^2 = sigma0^2 + k * seconds_unseen (term_motion)
// Pure: Eigen + the headers above, no DSR, no Qt. Frames: the ROOM frame, metres, yaw in radians (as rc::nbv).
#pragma once

#include "nbv/viewpoint_score.h"
#include "media_transport/sensor_model.h"   // rc::media::SensorModel -- DDS-free (factored out of media_transport.h)
#include "perception_card/card.h"
#include "perception_card/terms.h"
#include "detectability/blind_spot_map.h"   // the existing loader/query of the learned per-viewpoint correction

#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <numbers>
#include <span>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rc::card
{
// mount = body_T_sensor, in the robot BODY frame (Shadow: x right, y FORWARD, z up; ROBOT_GEOMETRY.md). evaluate_lidar
// places it in the room as Rz(heading - pi/2) * mount at the viewpoint, heading being the room bearing of body +y.
struct LidarRig { rc::media::SensorModel model; Eigen::Isometry3f mount; };

struct World
{
    std::vector<Eigen::Vector2f>                              polygon;     // empty = no walls known
    std::vector<rc::nbv::WallGap>                             gaps;
    std::vector<std::pair<std::string, rc::nbv::Obstacle>>    obstacles;  // NAMED, so an occluder can be reported
    std::optional<rc::nbv::Sensor>                            zed;
    float                                                     zed_mount_z = 0.0f;
    std::optional<LidarRig>                                   helios, bpearl;   // Task 4
};

struct Hypothesis
{
    std::string concept_, subtype;       // "concept" is a C++20 keyword (as in Card)
    float x = 0, y = 0, yaw = 0;
    std::optional<GeometryPrior> size;   // else the card's geometry
    std::optional<float> phi;            // articulation angle
    float seconds_unseen = 0;
};

struct Viewpoint { Eigen::Vector2f xy{0, 0}; std::optional<float> yaw; };   // nullopt = best heading: face the target centre

enum class Limit { None, OutOfView, BehindWall, Occluded, TooSmall, TooClose, EdgeOn, BetweenRings, BelowFan,
                   AboveFan, DeadZone, NotModelled, Inside, Range };

struct ChannelResult
{
    std::string sensor;
    float p = 0;
    Limit limit = Limit::None;           // the factor whose removal raises P most; None only when no factor would help at all
    float limit_gain = 0;                // P with that factor removed (set to 1) minus P: the consumer decides whether to show it
    std::string occluder;                // name of the obstacle whose removal raises visibility most (empty if none does)
    std::vector<std::string> notes;      // e.g. "walls unknown"
};

struct Result
{
    float p_notice = 0, p_recognise = 0;
    std::vector<ChannelResult> channels;   // EMPTY for an infeasible viewpoint/target (limit Inside): no sensor was asked
    Limit limit = Limit::None;           // of the best Identity-capable channel (highest p)
    float limit_gain = 0;                // that channel's limit_gain
    std::string occluder;
    std::vector<std::string> notes;
};

namespace detail
{
// Wall thickness for a card that declares no level-1 settings (Card::walls unset, an old card): the value the
// evaluator used before cards carried their own. A declared card uses its own wall_thickness_m.
inline constexpr float kLegacyWallThicknessM = 0.12f;
// A declared thickness of 0 is a LINE (rc::occlusion::walls_block tests polygon edges as lines). wall_obstacles needs
// a positive depth, so a line is a 1 mm box -- the representation door_look.h already uses -- not a model knob.
inline constexpr float kLineWallM = 1e-3f;

struct Obstacles { std::vector<rc::nbv::Obstacle> walls, objs, all; };

inline float wall_thickness_for(const Card& card)
{
    return card.walls ? std::max(kLineWallM, card.walls->wall_thickness_m) : kLegacyWallThicknessM;
}

// Distance from p to the oriented footprint rectangle of t (0 inside it).
inline float point_rect_dist(const Eigen::Vector2f& p, const rc::nbv::Target& t)
{
    const Eigen::Vector2f d = p - t.centre();
    const float lx = std::max(0.0f, std::abs(d.dot(t.axis_x())) - 0.5f * t.w);
    const float ly = std::max(0.0f, std::abs(d.dot(t.axis_y())) - 0.5f * t.h);
    return std::hypot(lx, ly);
}
inline float point_segment_dist(const Eigen::Vector2f& p, const Eigen::Vector2f& a, const Eigen::Vector2f& b)
{
    const Eigen::Vector2f ab = b - a;
    const float l2 = ab.squaredNorm();
    const float s = l2 > 1e-12f ? std::clamp((p - a).dot(ab) / l2, 0.0f, 1.0f) : 0.0f;
    return (p - (a + s * ab)).norm();
}
inline bool segments_cross(const Eigen::Vector2f& a, const Eigen::Vector2f& b, const Eigen::Vector2f& c, const Eigen::Vector2f& d)
{
    const auto cr = [](const Eigen::Vector2f& p, const Eigen::Vector2f& q) { return p.x() * q.y() - p.y() * q.x(); };
    const float d1 = cr(b - a, c - a), d2 = cr(b - a, d - a), d3 = cr(d - c, a - c), d4 = cr(d - c, b - c);
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0));
}
// Distance from the wall segment ab to the footprint of t: 0 if it touches or crosses it.
inline float segment_rect_dist(const Eigen::Vector2f& a, const Eigen::Vector2f& b, const rc::nbv::Target& t)
{
    std::array<Eigen::Vector2f, 4> c;
    for (int i = 0; i < 4; ++i)
        c[i] = t.centre() + ((i == 1 or i == 2) ? 0.5f : -0.5f) * t.w * t.axis_x() + (i >= 2 ? 0.5f : -0.5f) * t.h * t.axis_y();
    float d = std::min(point_rect_dist(a, t), point_rect_dist(b, t));
    for (int i = 0; i < 4; ++i)
    {
        if (segments_cross(a, b, c[i], c[(i + 1) % 4])) return 0.0f;
        d = std::min(d, point_segment_dist(c[i], a, b));
    }
    return d;
}

// Is this wall run the target's OWN wall under the card's level-1 rule (so it never occludes the target)?
//  · radius form (own_wall_skip_m > 0): rc::occlusion::walls_block skips a segment closer than the skip to the SAMPLE
//    it tests; the agents sample the object's surface, and the samples nearest a wall are the ones it could hide, so
//    the honest summary of "per sample" is the distance from the segment to the FOOTPRINT (its nearest point).
//  · plane form (own_wall_plane_m >= 0, door): both ends within +-own_wall_plane_m of the target's face plane, as
//    door_wall_edges does with the polygon vertices. A run is a sub-segment of its polygon edge, so an edge inside the
//    band gives runs inside it; only an edge slanting out of the band while a gap-cut piece of it stays inside would
//    differ, and that piece sits at the gap itself.
inline bool own_wall(const rc::nbv::Obstacle& run, const WallOcclusion& wo, const rc::nbv::Target& t)
{
    const Eigen::Vector2f u(std::cos(run.yaw), std::sin(run.yaw));
    const Eigen::Vector2f a = Eigen::Vector2f(run.cx, run.cy) - 0.5f * run.w * u, b = Eigen::Vector2f(run.cx, run.cy) + 0.5f * run.w * u;
    if (wo.own_wall_plane_m >= 0.0f)
    {
        const Eigen::Vector2f n = t.axis_y();
        return std::abs(n.dot(a - t.centre())) <= wo.own_wall_plane_m and std::abs(n.dot(b - t.centre())) <= wo.own_wall_plane_m;
    }
    return wo.own_wall_skip_m > 0.0f and segment_rect_dist(a, b, t) < wo.own_wall_skip_m;
}

// Walls (door gaps cut out, at the card's thickness), the named obstacles, and both together. Shared by every
// channel. With a target, the target's OWN walls (card level 1) are left out: they never occlude it. Without one
// (feasibility: "is the target inside a wall?") every wall stays.
inline Obstacles build_obstacles(const World& world, const Card& card, const rc::nbv::Target* target = nullptr)
{
    Obstacles o;
    o.walls = rc::nbv::wall_obstacles(world.polygon, wall_thickness_for(card), world.gaps);
    if (target and card.walls)
        std::erase_if(o.walls, [&](const rc::nbv::Obstacle& w) { return own_wall(w, *card.walls, *target); });
    o.all = o.walls;
    for (const auto& [name, ob] : world.obstacles) { o.objs.push_back(ob); o.all.push_back(ob); }
    return o;
}

// Leave-one-out: the named obstacle whose removal raises visible_fraction the most ("" if none does).
// `v_all` is the visible fraction with everything present.
inline std::string name_occluder(const World& world, const rc::nbv::Target& t, const Eigen::Vector2f& from, float yaw,
                                 const rc::nbv::Sensor& sensor, float v_all, const Obstacles& ob)
{
    std::string name;
    float best_gain = 0.0f;
    for (std::size_t k = 0; k < world.obstacles.size(); ++k)
    {
        std::vector<rc::nbv::Obstacle> rest = ob.walls;
        for (std::size_t m = 0; m < world.obstacles.size(); ++m)
            if (m != k) rest.push_back(world.obstacles[m].second);
        const float gain = rc::nbv::visible_fraction(t, from, yaw, sensor, rest) - v_all;
        if (gain > best_gain) { best_gain = gain; name = world.obstacles[k].first; }
    }
    return name;
}
}  // namespace detail

namespace detail
{
// The blind-spot map a card points at, loaded through rc::detect::BlindSpotMap::load and cached per (path, mtime)
// so where_to_stand's ~100 evaluations read the file once. nullptr (and *err set) when it cannot be read or is empty.
inline std::shared_ptr<const rc::detect::BlindSpotMap> blind_spot_map_for(const std::string& path, std::string* err)
{
    static std::mutex mu;
    static std::map<std::string, std::pair<std::filesystem::file_time_type, std::shared_ptr<const rc::detect::BlindSpotMap>>> cache;
    std::error_code ec;
    const auto mt = std::filesystem::last_write_time(path, ec);
    if (ec) { *err = "blind-spot map unreadable: " + path; return nullptr; }
    const std::scoped_lock lk(mu);
    auto& slot = cache[path];
    if (not slot.second or slot.first != mt)
    {
        auto m = std::make_shared<rc::detect::BlindSpotMap>();
        slot = {mt, m->load(path) > 0 ? std::shared_ptr<const rc::detect::BlindSpotMap>(m) : nullptr};
    }
    if (not slot.second) *err = "blind-spot map empty or unreadable: " + path;
    return slot.second;
}
}  // namespace detail

inline ChannelResult evaluate_camera(const Card& card, const World& world, const Hypothesis& h, const Viewpoint& vp)
{
    namespace nbv = rc::nbv;
    ChannelResult out;
    out.sensor = "zed";

    const auto ch = std::ranges::find_if(card.channels, [](const Channel& c) { return c.sensor == "zed"; });
    if (ch == card.channels.end() or not ch->modelled or not world.zed)
    {
        out.limit = Limit::NotModelled;
        out.notes.push_back(not world.zed ? "no zed in the world" : "card does not model the zed channel");
        return out;
    }
    if (world.polygon.size() < 3) out.notes.push_back("walls unknown");
    else if (not card.walls) out.notes.push_back("card declares no wall-occlusion settings: 0.12 m walls, no own-wall skip");

    const SubtypeParams& sp = params_for(card, h.subtype);
    nbv::Sensor sensor = *world.zed;
    sensor.env = sp.env;
    if (world.zed_mount_z > 1e-3f) sensor.height_m = world.zed_mount_z;

    const GeometryPrior g = h.size.value_or(card.geometry);
    nbv::Target t;
    t.cx = h.x; t.cy = h.y; t.yaw = h.yaw; t.w = g.w; t.h = g.d; t.z0 = g.z0; t.z1 = g.z1;

    // The envelope's arity (R8): axes = 1 is the agents' single-fill call p_detect(fill = max axis, visible, env), with
    // no fill-sigma marginalisation (table / refrigerator / hood do none); axes = 2 is the two-axis marginalised path.
    // A card that lists range_conf and no envelope (table with existence_absence_envelope = false) has no fill model.
    const auto env_it = std::ranges::find_if(sp.terms, [](const TermSpec& s) { return s.name == "envelope"; });
    const bool has_range_conf = std::ranges::any_of(sp.terms, [](const TermSpec& s) { return s.name == "range_conf"; });
    const bool use_env = env_it != sp.terms.end() or not has_range_conf;
    const int axes = env_it != sp.terms.end() and env_it->params.contains("axes") and env_it->params.at("axes") < 1.5f ? 1 : 2;
    const auto env_val = [&](float fmax, float fmin, float sig, float v)
    {
        if (not use_env) return std::clamp(v, 0.0f, 1.0f);
        if (axes == 1) return rc::detect::p_detect(fmax, v, sensor.env);
        return nbv::expected_p_detect(fmax, fmin, sig, v, sensor.env);
    };
    // Instance terms, marginalised at their prior for a hypothetical object (no score history).
    const auto has_inst = [&](std::string_view n) { return std::ranges::find(card.instance_terms, n) != card.instance_terms.end(); };
    const float f_inst = has_inst("p_clear") and card.instance_prior.contains("p_clear")
                             ? std::clamp(card.instance_prior.at("p_clear"), 0.0f, 1.0f) : 1.0f;
    std::shared_ptr<const rc::detect::BlindSpotMap> bsm;
    if (card.calibration == "blind_spot_map" and not card.blind_spot_map_path.empty())
    {
        std::string err;
        bsm = detail::blind_spot_map_for(card.blind_spot_map_path, &err);
        if (not bsm) out.notes.push_back(err + " (skipped)");
        else out.notes.push_back("limit analysis ignores the blind-spot map");
    }

    const Eigen::Vector2f from = vp.xy;
    const Eigen::Vector2f to_obj = t.centre() - from;
    const float yaw = vp.yaw ? *vp.yaw : (to_obj.norm() > 1e-6f ? std::atan2(to_obj.y(), to_obj.x()) : 0.0f);

    // Occluder lists: walls (with the door gaps cut out, the target's own walls left out) and the named obstacles.
    const detail::Obstacles ob = detail::build_obstacles(world, card, &t);
    const auto& walls = ob.walls; const auto& objs = ob.objs; const auto& all = ob.all;

    if (nbv::stands_inside(from, objs))
    {
        out.limit = Limit::Inside; out.limit_gain = 1.0f; out.p = 0.0f;
        return out;
    }

    // Card factors beyond the envelope. leaf_obliquity needs phi; the others are neutral for a hypothetical.
    const bool articulated = card.level == Level::Articulated;
    const bool mass = card.level == Level::Mass;
    // phi unset on an articulated card: the agent has not measured the hinge angle, so the leaf is neither closed nor
    // open -- marginalise the obliquity over the agent's own prior, uniform on [0, phi_max] (door: Openable.PhiMaxRad,
    // carried as leaf_obliquity's `phi_max` param; absent = pi/2). Only leaf_obliquity depends on phi, so the
    // marginal is a mean of that one factor. Gauss-Legendre with 8 nodes: the integrand is |cos(phi - bearing)|,
    // smooth except for one kink where the leaf turns edge-on to the ray; 8 nodes put that error well under 1e-2 in P
    // (the card's other numbers are good to ~0.05) for 8 cheap evaluations of one scalar term, the visibility
    // geometry being computed once. The result is the mean, so P(phi unset) lies between its extremes by construction.
    constexpr std::array<float, 4> kGLx{0.1834346425f, 0.5255324099f, 0.7966664774f, 0.9602898565f};
    constexpr std::array<float, 4> kGLw{0.3626837834f, 0.3137066459f, 0.2223810345f, 0.1012285363f};
    float phi_max = std::numbers::pi_v<float> / 2.0f;
    if (const auto lo = std::ranges::find_if(sp.terms, [](const TermSpec& s) { return s.name == "leaf_obliquity"; });
        lo != sp.terms.end() and lo->params.contains("phi_max"))
        phi_max = lo->params.at("phi_max");
    const bool marginal_phi = articulated and not h.phi;
    if (marginal_phi)
        out.notes.push_back("phi unknown: marginalised over uniform [0, " + std::to_string(phi_max) + "] rad (8-node Gauss-Legendre)");
    // central: the agents' central_frac = clamp(1 - |centroid offset from the principal point| / half-frame). With a
    // given heading the HORIZONTAL offset is the projected bearing of the target centre; with best heading
    // (vp.yaw unset) the camera faces the centre, the offset is 0 and central stays 1. The VERTICAL part of the
    // agent's centroid depends on which surfaces it samples (a fridge its faces, a table its top), which a card does
    // not carry, so it is not modelled: central is an upper bound in that direction.
    const auto central_of = [&](const nbv::Target& tt) -> float
    {
        if (not vp.yaw) return 1.0f;
        const Eigen::Vector2f r = tt.centre() - from;
        const Eigen::Vector2f fwd(std::cos(yaw), std::sin(yaw)), right(std::sin(yaw), -std::cos(yaw));
        const float depth = r.dot(fwd);
        if (depth <= 1e-3f) return 0.0f;                   // centre at / behind the image plane
        return std::clamp(1.0f - std::abs(r.dot(right) / depth) / sensor.tan_half_h(), 0.0f, 1.0f);
    };
    const auto card_factors = [&](const nbv::Target& tt) -> std::pair<float, float>   // {obliquity, rest}
    {
        TermInputs in;
        in.range = (tt.centre() - from).norm();
        in.seconds_unseen = h.seconds_unseen;
        in.central = central_of(tt);
        const Eigen::Vector2f r = tt.centre() - from;
        const auto set_phi = [&](TermInputs& ti, float phi)
        {
            ti.phi = phi;
            const Eigen::Vector2f n = std::cos(phi) * tt.axis_y() + std::sin(phi) * tt.axis_x();   // leaf normal at phi
            ti.obliquity = r.norm() > 1e-6f ? n.dot(r / r.norm()) : 1.0f;
        };
        set_phi(in, h.phi.value_or(0.0f));                 // once, for every term of the loop
        float f_obl = 1.0f, f_rest = 1.0f;
        for (const TermSpec& s : sp.terms)
        {
            if (s.name == "envelope" or s.name == "in_fov" or s.name == "coverage" or s.name == "motion")
                continue;                                  // carried by the geometry above / by the GH below
            if (s.name == "leaf_obliquity")
            {
                if (not articulated) continue;
                if (not marginal_phi) { f_obl *= eval_term(s, in); continue; }
                float acc = 0.0f;                          // mean over [0, phi_max]: weights sum to 2 on [-1, 1]
                TermInputs q = in;
                for (std::size_t i = 0; i < kGLx.size(); ++i)
                    for (const float sgn : {-1.0f, 1.0f})
                    {
                        set_phi(q, 0.5f * phi_max * (1.0f + sgn * kGLx[i]));
                        acc += 0.5f * kGLw[i] * eval_term(s, q);
                    }
                f_obl *= acc;
            }
            else f_rest *= eval_term(s, in);
        }
        return {f_obl, f_rest * f_inst};
    };

    const auto p_at = [&](const nbv::Target& tt)
    {
        const float v = nbv::visible_fraction(tt, from, yaw, sensor, all);
        const auto [fo, fr] = card_factors(tt);
        if (mass) return v * fo * fr;                      // spec §2: a mass is COVERAGE only, no detector envelope
        const auto [fmax, fmin] = nbv::predicted_fill_axes(tt, from, sensor, yaw);
        const float sg = nbv::predicted_fill_sigma(tt, from, sensor);
        if (not bsm) return env_val(fmax, fmin, sg, v) * fo * fr;
        // The learned correction is per VIEWPOINT, so it applies to a hypothesis too: same view coordinates, prior
        // (without visibility) and re-multiplication by visibility as table_existence.cpp does at the query site.
        const Eigen::Vector2f d = from - tt.centre();
        const float horiz = d.norm(), dz = sensor.height_m - tt.z1;
        rc::detect::Viewpoint view;
        view.az = rc::detect::fold_bearing(std::atan2(d.y(), d.x()) - tt.yaw, h.subtype == "round");
        view.el = std::atan2(dz, horiz);
        view.lr = std::log(std::max(0.05f, std::hypot(horiz, dz)));
        const float prior = std::clamp(env_val(fmax, fmin, sg, 1.0f) * fo * fr, 0.0f, 1.0f);
        return bsm->query(0, view, prior).p * v;
    };

    // ── P: the animate level marginalises the position over sigma^2 = sigma0^2 + k * seconds_unseen ────────────
    float sigma = 0.0f;
    if (card.level == Level::Animate)
    {
        const auto m = std::ranges::find_if(sp.terms, [](const TermSpec& s) { return s.name == "motion"; });
        if (m == sp.terms.end()) out.notes.push_back("animate card has no motion term: no position spread");
        else
        {
            TermInputs in; in.seconds_unseen = h.seconds_unseen;
            sigma = eval_term(*m, in);
        }
    }
    if (sigma > 1e-4f)   // numerical guard only: below 0.1 mm the 25-node quadrature collapses onto one point
    {
        float acc = 0.0f;
        for (std::size_t i = 0; i < nbv::detail::kGHx.size(); ++i)
            for (std::size_t j = 0; j < nbv::detail::kGHx.size(); ++j)
            {
                nbv::Target tt = t;
                tt.cx += sigma * nbv::detail::kGHx[i];
                tt.cy += sigma * nbv::detail::kGHx[j];
                acc += nbv::detail::kGHw[i] * nbv::detail::kGHw[j] * p_at(tt);
            }
        out.p = std::clamp(acc, 0.0f, 1.0f);
    }
    else out.p = std::clamp(p_at(t), 0.0f, 1.0f);
    if (not std::isfinite(out.p)) out.p = 0.0f;       // never hand a NaN to a planner

    // ── the limiting factor: the one whose REMOVAL would raise P the most (leave-one-out, as for the occluder) ───
    // Evaluated at the nominal position. Order-free, unlike peeling factors one by one (which hands everything to
    // whichever is peeled first: a target behind a wall would read "too small" because its envelope was peeled first).
    const auto [fmax, fmin] = nbv::predicted_fill_axes(t, from, sensor, yaw);
    const float sig_fill = nbv::predicted_fill_sigma(t, from, sensor);
    nbv::Sensor wide = sensor;  wide.hfov_rad = 3.0f; wide.vfov_rad = 3.0f;       // "in view" everywhere
    const float v_all     = nbv::visible_fraction(t, from, yaw, sensor, all);
    const float v_no_fov  = nbv::visible_fraction(t, from, 0.0f, wide, all);
    const float v_no_wall = nbv::visible_fraction(t, from, yaw, sensor, objs);
    const float v_no_obj  = nbv::visible_fraction(t, from, yaw, sensor, walls);
    const auto [f_obl, f_rest] = card_factors(t);
    const auto env_p = [&](float v) { return mass ? v : env_val(fmax, fmin, sig_fill, v); };
    const float now = env_p(v_all) * f_obl * f_rest;

    Limit env_label = fmax > nbv::framing_fill(sensor) ? Limit::TooClose : Limit::TooSmall;
    std::array<std::pair<float, Limit>, 6> gains{{
        {v_all * f_obl * f_rest - now,                 env_label},            // a perfect envelope
        {env_p(v_no_fov) * f_obl * f_rest - now,       Limit::OutOfView},
        {env_p(v_no_wall) * f_obl * f_rest - now,      Limit::BehindWall},
        {env_p(v_no_obj) * f_obl * f_rest - now,       Limit::Occluded},
        {env_p(v_all) * f_rest - now,                  Limit::EdgeOn},        // f_obl -> 1
        {env_p(v_all) * f_obl - now,                   Limit::Range}}};      // the other card terms (range_conf)
    if (not mass and not std::isfinite(fmax))   // a corner at / behind the image plane: standing on it, or looking away
    {
        const float bear = std::abs(std::remainder(std::atan2(to_obj.y(), to_obj.x()) - yaw, 2.0f * std::numbers::pi_v<float>));
        const bool ahead = to_obj.norm() < 1e-3f or bear <= 0.5f * sensor.hfov_rad;   // standing on it counts as too close
        gains = {{{1.0f, ahead ? Limit::TooClose : Limit::OutOfView}}};
    }
    const auto best = std::ranges::max_element(gains, {}, &std::pair<float, Limit>::first);
    out.limit = best->first > 0.0f ? best->second : Limit::None;
    out.limit_gain = best->first > 0.0f ? best->first : 0.0f;

    // ── occluder by leave-one-out: the obstacle whose removal raises visible_fraction the most ───────────────────
    if (v_no_obj - v_all > 0.0f)       // some obstacle costs visibility: name the worst, whatever the label says
        out.occluder = detail::name_occluder(world, t, from, yaw, sensor, v_all, ob);
    return out;
}
// ── LiDAR half (spec §4b) ───────────────────────────────────────────────────────────────────────────────────
// A ring is a cone about the device z axis; `mount` (room frame) carries it into the room, so an INVERTED sensor
// (Helios, Bpearl) simply has room elevation = -device elevation: nothing is hard-coded, the rotation does it.
// lambda = sum over crossing rings of (columns across the target) x visible_fraction, x range_factor(r);
// P = 1 - exp(-lambda q_return). A ring is a line, so "does it cross the z span" is geometry, not a gate.
namespace detail
{
inline constexpr float kDeg2Rad = std::numbers::pi_v<float> / 180.0f;

// Device ring elevations [deg]: the model's table, else `rings` spread uniformly over fov_start..fov_end.
inline std::vector<float> ring_elevations_deg(const rc::media::SensorModel& m)
{
    if (not m.ring_elev_deg.empty()) return m.ring_elev_deg;
    std::vector<float> e;
    const int n = m.rings.value_or(0);
    if (n < 1 or not m.fov_start_deg or not m.fov_end_deg) return e;
    for (int i = 0; i < n; ++i)
        e.push_back(n == 1 ? 0.5f * (*m.fov_start_deg + *m.fov_end_deg)
                           : *m.fov_start_deg + (*m.fov_end_deg - *m.fov_start_deg) * static_cast<float>(i) / static_cast<float>(n - 1));
    return e;
}

// Elevation [rad] in the ROOM frame of the ring at device elevation `e_deg`, at the device azimuth whose room
// bearing is closest to `bearing` (1 degree sampling: exact for a pure flip, a 0.5 degree error otherwise).
inline float room_elevation(const Eigen::Isometry3f& mount, float e_deg, float bearing)
{
    const float e = e_deg * kDeg2Rad, ce = std::cos(e), se = std::sin(e);
    float best = 1e9f, el = 0.0f;
    for (int a = 0; a < 360; ++a)
    {
        const float az = static_cast<float>(a) * kDeg2Rad;
        const Eigen::Vector3f d = mount.linear() * Eigen::Vector3f(ce * std::cos(az), ce * std::sin(az), se);
        const float err = std::abs(std::remainder(std::atan2(d.y(), d.x()) - bearing, 2.0f * std::numbers::pi_v<float>));
        if (err < best) { best = err; el = std::asin(std::clamp(d.z(), -1.0f, 1.0f)); }
    }
    return el;
}

inline float norm_cdf(float x) { return 0.5f * std::erfc(-x / std::numbers::sqrt2_v<float>); }

// Soft at both ends of the usable range: a smooth dead zone and a smooth far limit, no hard cutoff.
inline float range_factor(float r, float rmin, float rmax, float sigma)
{
    return norm_cdf((r - rmin) / sigma) * norm_cdf((rmax - r) / sigma);
}

struct LidarGeom
{
    Eigen::Vector2f origin{0, 0};
    float oz = 0, r = 0, bearing = 0, ang_width = 0, az_step = 0, rmin = 0, rmax = 0, sigma_r = 0.02f;
    int n_cross = 0;
    float min_el = 0, max_el = 0;      // room elevations of the fan at this bearing [rad]
    bool ok = false;                   // false: the model cannot be evaluated (no rings / no azimuth step)
};

inline LidarGeom lidar_geom(const LidarRig& rig, const rc::nbv::Target& t, const Eigen::Vector2f& from)
{
    LidarGeom g;
    const auto& m = rig.model;
    g.origin = from; g.oz = rig.mount.translation().z();
    const Eigen::Vector2f rel = t.centre() - from;
    g.r = rel.norm();
    g.bearing = g.r > 1e-6f ? std::atan2(rel.y(), rel.x()) : 0.0f;
    g.rmin = m.range_min_m.value_or(0.0f);
    g.rmax = m.range_max_m.value_or(1e9f);
    g.sigma_r = std::max(m.range_sigma_floor_m.value_or(0.02f), 1e-3f);   // 1 mm: keeps Phi finite, not a knob on P
    g.az_step = m.azimuth_step_deg.value_or(0.0f) * kDeg2Rad;

    // Angular width of the footprint's silhouette seen from `from`; a sensor inside the footprint sees all round.
    const Eigen::Vector2f ex = t.axis_x(), ey = t.axis_y();
    float lo = 0.0f, hi = 0.0f;
    bool inside = std::abs((from - t.centre()).dot(ex)) <= 0.5f * t.w and std::abs((from - t.centre()).dot(ey)) <= 0.5f * t.h;
    for (int i = 0; i < 4 and not inside; ++i)
    {
        const Eigen::Vector2f c = t.centre() + ((i & 1) ? 0.5f : -0.5f) * t.w * ex + ((i & 2) ? 0.5f : -0.5f) * t.h * ey;
        const float a = std::remainder(std::atan2(c.y() - from.y(), c.x() - from.x()) - g.bearing, 2.0f * std::numbers::pi_v<float>);
        lo = std::min(lo, a); hi = std::max(hi, a);
    }
    g.ang_width = inside ? 2.0f * std::numbers::pi_v<float> : hi - lo;

    const auto elevs = ring_elevations_deg(m);
    if (elevs.empty() or g.az_step <= 0.0f) return g;
    g.ok = true;
    g.min_el = 1e9f; g.max_el = -1e9f;
    for (float e : elevs)
    {
        const float el = room_elevation(rig.mount, e, g.bearing);
        g.min_el = std::min(g.min_el, el); g.max_el = std::max(g.max_el, el);
        const float zr = g.oz + g.r * std::tan(std::clamp(el, -1.5707f, 1.5707f));
        if (zr >= t.z0 and zr <= t.z1) ++g.n_cross;
    }
    return g;
}

// lambda0 = expected returns BEFORE the range factor, for a given visible fraction; `ring_fix` = "one ring
// crosses it at least".
//
// ★ THE RANGE FACTOR IS COMMON-MODE, SO IT MULTIPLIES P, NOT lambda. Every return from one target shares the same
// range r: whether r lies in the dead zone / past the far limit is ONE event for the whole target, not an
// independent coin per return. Folding Phi into lambda lets a large column count outvote it (a 2 cm target
// 0.05 m inside a 0.1 m dead zone got p = 0.93); as a factor on P the shared event cannot be averaged away by
// the number of returns (same reasoning as the common-mode marginalisation in CLAUDE.md).
inline float lidar_lambda(const LidarGeom& g, float v, bool ring_fix = false)
{
    if (not g.ok) return 0.0f;
    const float cols = g.ang_width / g.az_step;
    const float rings = static_cast<float>(ring_fix ? std::max(g.n_cross, 1) : g.n_cross);
    return rings * cols * v;
}
inline rc::nbv::Sensor lidar_sight(float oz)
{
    // An all-round sight line for visible_fraction: ~172 deg either side of the heading (the clamp is 3 rad),
    // a vertical channel on so heights of obstacles and target count, at the sensor's height. The ring geometry
    // decides what is in the fan; this only asks "is the line of sight clear".
    rc::nbv::Sensor s;
    s.hfov_rad = 3.0f; s.vfov_rad = 2.9f; s.height_m = oz;
    return s;
}
}  // namespace detail

// Expected number of returns on `t` from a LiDAR whose origin is `from` (xy; z and orientation from rig.mount).
inline float expected_returns(const LidarRig& rig, const rc::nbv::Target& t, const Eigen::Vector2f& from,
                              std::span<const rc::nbv::Obstacle> obstacles)
{
    const detail::LidarGeom g = detail::lidar_geom(rig, t, from);
    const float v = rc::nbv::visible_fraction(t, from, g.bearing, detail::lidar_sight(g.oz), obstacles);
    return detail::lidar_lambda(g, v);        // lambda0: the range factor is common-mode and lives on P
}

// room_T_sensor for a robot at vp facing the target (see evaluate_lidar below for the convention).
inline LidarRig rig_in_room(const LidarRig& body_rig, const Viewpoint& vp, const Hypothesis& h)
{
    const Eigen::Vector2f to_t = Eigen::Vector2f(h.x, h.y) - vp.xy;
    const float heading = vp.yaw ? *vp.yaw : (to_t.norm() > 1e-6f ? std::atan2(to_t.y(), to_t.x()) : 0.0f);
    Eigen::Isometry3f robot = Eigen::Isometry3f::Identity();
    robot.linear() = Eigen::AngleAxisf(heading - std::numbers::pi_v<float> / 2.0f, Eigen::Vector3f::UnitZ()).toRotationMatrix();
    robot.translation() = Eigen::Vector3f(vp.xy.x(), vp.xy.y(), 0.0f);
    LidarRig rig = body_rig;
    rig.mount = robot * body_rig.mount;
    return rig;
}

// `sensor` = "helios" | "bpearl". The rig's mount is body_T_sensor (BODY frame). The robot stands at vp.xy with heading
// vp.yaw (the room bearing it faces, the camera's optical heading as in rc::nbv) or, unset, the bearing to the target
// centre -- the same best heading the camera takes. Shadow's forward is body +y, so the body frame is the heading
// rotated by -pi/2: room_T_sensor = [Rz(heading - pi/2), (vp.xy, 0)] * body_T_sensor.
inline ChannelResult evaluate_lidar(const Card& card, const World& world, const Hypothesis& h, const Viewpoint& vp,
                                    std::string_view sensor)
{
    namespace nbv = rc::nbv;
    ChannelResult out;
    out.sensor = std::string(sensor);
    const std::optional<LidarRig>& rig_opt = sensor == "bpearl" ? world.bpearl : world.helios;
    const auto ch = std::ranges::find_if(card.channels, [&](const Channel& c) { return c.sensor == sensor; });
    if (ch == card.channels.end() or not ch->modelled or not rig_opt)
    {
        out.limit = Limit::NotModelled;
        out.notes.push_back(not rig_opt ? "no " + out.sensor + " in the world" : "card does not model the " + out.sensor + " channel");
        return out;                                  // left out of the combination by the caller
    }
    if (world.polygon.size() < 3) out.notes.push_back("walls unknown");

    // The rig as it sits for this viewpoint: robot at vp.xy facing `heading`, the mount's own xy a lever arm.
    const LidarRig rig = rig_in_room(*rig_opt, vp, h);
    const Eigen::Vector2f from = rig.mount.translation().head<2>();
    if (not rig.model.range_sigma_floor_m) out.notes.push_back("range sigma 0.02 m assumed (model has none)");
    if (rig.model.ring_elev_deg.empty()) out.notes.push_back("rings spaced uniformly over the fov (no ring table)");

    const SubtypeParams& sp = params_for(card, h.subtype);
    const GeometryPrior gp = h.size.value_or(card.geometry);
    nbv::Target t;
    t.cx = h.x; t.cy = h.y; t.yaw = h.yaw; t.w = gp.w; t.h = gp.d; t.z0 = gp.z0; t.z1 = gp.z1;

    const detail::Obstacles ob = detail::build_obstacles(world, card, &t);
    const auto& walls = ob.walls; const auto& objs = ob.objs; const auto& all = ob.all;

    if (nbv::stands_inside(from, objs))
    {
        out.limit = Limit::Inside; out.limit_gain = 1.0f; out.p = 0.0f;
        return out;
    }

    const detail::LidarGeom g = detail::lidar_geom(rig, t, from);
    if (not g.ok)
    {
        out.limit = Limit::NotModelled; out.notes.push_back("lidar model lacks rings or azimuth step");
        return out;
    }
    const nbv::Sensor sight = detail::lidar_sight(g.oz);
    const float q = sp.q_return;
    const float rf = detail::range_factor(g.r, g.rmin, g.rmax, g.sigma_r);   // common-mode: multiplies P
    const auto p_of = [&](float lam, float range = -1.0f)
    { return (range < 0.0f ? rf : range) * (1.0f - std::exp(-std::max(lam, 0.0f) * q)); };

    const float v_all     = nbv::visible_fraction(t, from, g.bearing, sight, all);
    const float v_no_wall = nbv::visible_fraction(t, from, g.bearing, sight, objs);
    const float v_no_obj  = nbv::visible_fraction(t, from, g.bearing, sight, walls);
    out.p = p_of(detail::lidar_lambda(g, v_all));
    if (not std::isfinite(out.p)) out.p = 0.0f;

    // Leave-one-out: the factor whose removal raises P the most.
    Limit ring_label = Limit::BetweenRings;
    if (g.n_cross == 0)
    {
        const float a0 = std::atan2(t.z0 - g.oz, g.r), a1 = std::atan2(t.z1 - g.oz, g.r);
        ring_label = a1 < g.min_el ? Limit::BelowFan : a0 > g.max_el ? Limit::AboveFan : Limit::BetweenRings;
    }
    const Limit range_label = g.r < g.rmin ? Limit::DeadZone : Limit::Range;
    const std::array<std::pair<float, Limit>, 4> gains{{
        {p_of(detail::lidar_lambda(g, v_all, true)) - out.p,   ring_label},
        {p_of(detail::lidar_lambda(g, v_all), 1.0f)      - out.p,   range_label},
        {p_of(detail::lidar_lambda(g, v_no_wall))         - out.p,   Limit::BehindWall},
        {p_of(detail::lidar_lambda(g, v_no_obj))          - out.p,   Limit::Occluded}}};
    const auto best = std::ranges::max_element(gains, {}, &std::pair<float, Limit>::first);
    out.limit = best->first > 0.0f ? best->second : Limit::None;
    out.limit_gain = best->first > 0.0f ? best->first : 0.0f;
    // No single removal helps when TWO factors each zero P on their own: a target beyond the far limit (or in the dead
    // zone) that no ring crosses either -- fixing the rings leaves the range at 0, fixing the range leaves no returns.
    // Leave-two-out then: name the pair's range side, the one that holds whatever the height (a ring crossing could be
    // bought by a different target height at this distance, the range could not).
    if (out.limit == Limit::None)
    {
        const float both = p_of(detail::lidar_lambda(g, v_all, true), 1.0f) - out.p;
        if (both > 0.0f) { out.limit = (g.r < g.rmin or g.r > g.rmax) ? range_label : ring_label; out.limit_gain = both; }
    }

    if (v_no_obj - v_all > 0.0f)
        out.occluder = detail::name_occluder(world, t, from, g.bearing, sight, v_all, ob);
    return out;
}

// ── combining the channels (spec §5) ─────────────────────────────────────────────────────────────────────────
// Independent looks: P = 1 - prod(1 - p_c). Notice = ANY channel says "something is there" (an identifying look also
// notices); recognise = the ones that can say "it is THIS" (Identity or Both). A NotModelled channel is not a zero
// look, it is no look: it is left out of both products.
inline Result evaluate(const Card& card, const World& world, const Hypothesis& h, const Viewpoint& vp)
{
    namespace nbv = rc::nbv;
    Result out;

    // Feasibility first: an impossible viewpoint or target gets P = 0 whatever the sensors would say.
    const detail::Obstacles ob = detail::build_obstacles(world, card);   // every wall: "is the target inside one?"
    const auto zero = [&](std::string occluder, std::string note)
    {
        out.p_notice = out.p_recognise = 0.0f;
        out.limit = Limit::Inside; out.limit_gain = 1.0f;
        out.occluder = std::move(occluder);
        out.notes.push_back(std::move(note));
        return out;
    };
    if (not nbv::is_reachable(vp.xy, world.polygon)) return zero("", "viewpoint outside the room polygon");
    for (const auto& [name, o] : world.obstacles)
        if (nbv::stands_inside(vp.xy, std::span<const nbv::Obstacle>(&o, 1))) return zero(name, "viewpoint inside " + name);
    if (nbv::stands_inside({h.x, h.y}, ob.walls)) return zero("wall", "target inside a wall");

    // Recognising implies noticing: P(notice) is the noisy-OR over EVERY counted channel (the union of the Occupancy,
    // Identity and Both ones, each once), so P(notice) >= P(recognise) by construction. Without any Both channel it is
    // exactly 1 - (1 - p_occ)(1 - p_rec) with p_occ, p_rec the noisy-ORs over Occupancy and Identity; a Both channel
    // enters once, not twice (it is one look).
    float miss_notice = 1.0f, miss_recognise = 1.0f, best_id = -1.0f;
    for (const Channel& c : card.channels)
    {
        ChannelResult r;
        if (c.sensor == "zed") r = evaluate_camera(card, world, h, vp);
        else if (c.sensor == "helios" or c.sensor == "bpearl") r = evaluate_lidar(card, world, h, vp, c.sensor);
        else { r.sensor = c.sensor; r.limit = Limit::NotModelled; r.notes.push_back("no evaluator for " + c.sensor); }
        const bool finite = std::isfinite(r.p);
        if (not finite and r.limit != Limit::NotModelled)
            out.notes.push_back(r.sensor + ": non-finite p, channel excluded");
        const bool counts = r.limit != Limit::NotModelled and c.modelled and finite;
        if (counts)
        {
            const float p = std::clamp(r.p, 0.0f, 1.0f);
            miss_notice *= 1.0f - p;
            if (c.role == Role::Identity or c.role == Role::Both)
            {
                miss_recognise *= 1.0f - p;
                if (p > best_id) { best_id = p; out.limit = r.limit; out.limit_gain = r.limit_gain; out.occluder = r.occluder; }
            }
        }
        out.notes.insert(out.notes.end(), r.notes.begin(), r.notes.end());
        out.channels.push_back(std::move(r));
    }
    out.p_notice = 1.0f - miss_notice;
    out.p_recognise = 1.0f - miss_recognise;
    if (best_id < 0.0f) out.limit = Limit::NotModelled;     // no modelled channel can identify it
    return out;
}

struct Stand { Eigen::Vector2f xy; float yaw; Result r; };

// Viewpoints that maximise P(recognise): 36 bearings round the target x 3 standoffs inside the camera's
// standoff_band (near edge, peak, far edge). The band is the SAME scan the planners use. Stands that are inside an
// obstacle or outside the room are dropped (feasibility, as in rc::nbv), the rest are scored by evaluate() with
// the heading on the target, and the best k returned.
inline std::vector<Stand> where_to_stand(const Card& card, const World& world, const Hypothesis& h, int k = 3)
{
    namespace nbv = rc::nbv;
    constexpr int kBearings = 36;
    constexpr float kRobotRadiusM = 0.30f;     // body clearance, a property of the robot (as the planners pass it)
    constexpr float kBandCapM = 8.0f;          // the band scan's own range (standoff_band max_m)
    const SubtypeParams& sp = params_for(card, h.subtype);
    const GeometryPrior g = h.size.value_or(card.geometry);
    nbv::Target t;
    t.cx = h.x; t.cy = h.y; t.yaw = h.yaw; t.w = g.w; t.h = g.d; t.z0 = g.z0; t.z1 = g.z1;
    nbv::Sensor sensor = world.zed.value_or(nbv::Sensor{});
    sensor.env = sp.env;
    if (world.zed_mount_z > 1e-3f) sensor.height_m = world.zed_mount_z;
    const float floor_m = kRobotRadiusM + 0.5f * std::max(g.w, g.d);

    const detail::Obstacles ob = detail::build_obstacles(world, card);
    std::vector<Stand> stands;
    for (int i = 0; i < kBearings; ++i)
    {
        const float a = 2.0f * std::numbers::pi_v<float> * static_cast<float>(i) / kBearings;
        const Eigen::Vector2f dir(std::cos(a), std::sin(a));
        const nbv::StandoffBand band = nbv::standoff_band(t, sensor, t.centre(), dir, floor_m);
        for (const float d : {band.near_m, band.best, std::min(band.far_m, kBandCapM)})
        {
            const Eigen::Vector2f xy = t.centre() + dir * d;
            if (nbv::stands_inside(xy, ob.objs, kRobotRadiusM) or not nbv::is_reachable(xy, world.polygon, kRobotRadiusM))
                continue;
            const Eigen::Vector2f to = t.centre() - xy;
            Stand s{xy, std::atan2(to.y(), to.x()), {}};
            s.r = evaluate(card, world, h, Viewpoint{xy, s.yaw});
            stands.push_back(std::move(s));
        }
    }
    const std::size_t n = std::min<std::size_t>(stands.size(), static_cast<std::size_t>(std::max(k, 0)));
    std::ranges::partial_sort(stands, stands.begin() + n, {}, [](const Stand& s) { return -s.r.p_recognise; });
    stands.resize(n);
    return stands;
}
}  // namespace rc::card
