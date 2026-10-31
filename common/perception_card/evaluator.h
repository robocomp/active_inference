// evaluator.h -- the shared perception-card evaluator: (card, world facts, HYPOTHETICAL object, viewpoint) ->
// P(detect) and the factor that limits it. Task 3 = the camera half; Task 4 adds the LiDAR half (LidarRig and the
// helios/bpearl fields are declared here for it and unused until then).
//
// Geometry is NOT reimplemented: it is common/nbv/viewpoint_score.h (rc::nbv), the same code the planners call, so
// the answer a planner gets and the answer the dashboard shows are one function. Per level (spec §2):
//   coverage     = visible_fraction (frustum ^ walls ^ named obstacles)
//   countable    x the envelope, expected_p_detect over predicted_fill_axes / predicted_fill_sigma
//   articulated  x the card's configuration terms at phi (leaf_obliquity; resolvability / kappa need a MEASUREMENT a
//                hypothetical object does not have, so they stay at their neutral 1)
//   animate      x Gauss-Hermite over the position, sigma^2 = sigma0^2 + k * seconds_unseen (term_motion)
// Pure: Eigen + the headers above, no DSR, no Qt. Frames: the ROOM frame, metres, yaw in radians (as rc::nbv).
#pragma once

#include "nbv/viewpoint_score.h"
#include "media_transport/sensor_model.h"   // rc::media::SensorModel -- DDS-free (factored out of media_transport.h)
#include "perception_card/card.h"
#include "perception_card/terms.h"

#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <span>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rc::card
{
struct LidarRig { rc::media::SensorModel model; Eigen::Isometry3f mount; };   // mount in the room frame (Task 4)

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
    std::vector<ChannelResult> channels;
    Limit limit = Limit::None;
    std::string occluder;
    std::vector<std::string> notes;
};

namespace detail
{
// Wall thickness handed to wall_obstacles. A modelling constant of the room, not a tuning knob on P.
inline constexpr float kWallThicknessM = 0.12f;

struct Obstacles { std::vector<rc::nbv::Obstacle> walls, objs, all; };

// Walls (door gaps cut out), the named obstacles, and both together. Shared by every channel.
inline Obstacles build_obstacles(const World& world)
{
    Obstacles o;
    o.walls = rc::nbv::wall_obstacles(world.polygon, kWallThicknessM, world.gaps);
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

    const SubtypeParams& sp = params_for(card, h.subtype);
    nbv::Sensor sensor = *world.zed;
    sensor.env = sp.env;
    if (world.zed_mount_z > 1e-3f) sensor.height_m = world.zed_mount_z;

    const GeometryPrior g = h.size.value_or(card.geometry);
    nbv::Target t;
    t.cx = h.x; t.cy = h.y; t.yaw = h.yaw; t.w = g.w; t.h = g.d; t.z0 = g.z0; t.z1 = g.z1;

    const Eigen::Vector2f from = vp.xy;
    const Eigen::Vector2f to_obj = t.centre() - from;
    const float yaw = vp.yaw ? *vp.yaw : (to_obj.norm() > 1e-6f ? std::atan2(to_obj.y(), to_obj.x()) : 0.0f);

    // Occluder lists: walls (with the door gaps cut out) and the named obstacles.
    const detail::Obstacles ob = detail::build_obstacles(world);
    const auto& walls = ob.walls; const auto& objs = ob.objs; const auto& all = ob.all;

    if (nbv::stands_inside(from, objs))
    {
        out.limit = Limit::Inside; out.limit_gain = 1.0f; out.p = 0.0f;
        return out;
    }

    // Card factors beyond the envelope. leaf_obliquity needs phi; the others are neutral for a hypothetical.
    const bool articulated = card.level == Level::Articulated;
    const bool mass = card.level == Level::Mass;
    if (articulated and not h.phi) out.notes.push_back("phi unknown: leaf taken closed");
    const float phi = h.phi.value_or(0.0f);
    const auto card_factors = [&](const nbv::Target& tt) -> std::pair<float, float>   // {obliquity, rest}
    {
        TermInputs in;
        in.range = (tt.centre() - from).norm();
        in.phi = phi;
        in.seconds_unseen = h.seconds_unseen;
        const Eigen::Vector2f n = std::cos(phi) * tt.axis_y() + std::sin(phi) * tt.axis_x();   // leaf normal at phi
        const Eigen::Vector2f r = tt.centre() - from;
        in.obliquity = r.norm() > 1e-6f ? n.dot(r / r.norm()) : 1.0f;
        float f_obl = 1.0f, f_rest = 1.0f;
        for (const TermSpec& s : sp.terms)
        {
            if (s.name == "envelope" or s.name == "in_fov" or s.name == "central" or s.name == "coverage" or s.name == "motion")
                continue;                                  // carried by the geometry above / by the GH below
            if (s.name == "leaf_obliquity") { if (articulated) f_obl *= eval_term(s, in); }
            else f_rest *= eval_term(s, in);
        }
        return {f_obl, f_rest};
    };

    const auto p_at = [&](const nbv::Target& tt)
    {
        const float v = nbv::visible_fraction(tt, from, yaw, sensor, all);
        const auto [fo, fr] = card_factors(tt);
        if (mass) return v * fo * fr;                      // spec §2: a mass is COVERAGE only, no detector envelope
        const auto [fmax, fmin] = nbv::predicted_fill_axes(tt, from, sensor, yaw);
        return nbv::expected_p_detect(fmax, fmin, nbv::predicted_fill_sigma(tt, from, sensor), v, sensor.env) * fo * fr;
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
    const auto env_p = [&](float v) { return mass ? v : nbv::expected_p_detect(fmax, fmin, sig_fill, v, sensor.env); };
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

// `sensor` = "helios" | "bpearl". The rig's mount.xy is a lever arm from the robot (at vp.xy, heading vp.yaw or 0);
// z and the orientation come from the mount as given.
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

    // The rig as it sits for this viewpoint: robot at vp.xy / vp.yaw, the mount's own xy a lever arm.
    LidarRig rig = *rig_opt;
    {
        Eigen::Isometry3f robot = Eigen::Isometry3f::Identity();
        robot.linear() = Eigen::AngleAxisf(vp.yaw.value_or(0.0f), Eigen::Vector3f::UnitZ()).toRotationMatrix();
        robot.translation() = Eigen::Vector3f(vp.xy.x(), vp.xy.y(), 0.0f);
        rig.mount = robot * rig_opt->mount;
    }
    const Eigen::Vector2f from = rig.mount.translation().head<2>();
    if (not rig.model.range_sigma_floor_m) out.notes.push_back("range sigma 0.02 m assumed (model has none)");
    if (rig.model.ring_elev_deg.empty()) out.notes.push_back("rings spaced uniformly over the fov (no ring table)");

    const SubtypeParams& sp = params_for(card, h.subtype);
    const GeometryPrior gp = h.size.value_or(card.geometry);
    nbv::Target t;
    t.cx = h.x; t.cy = h.y; t.yaw = h.yaw; t.w = gp.w; t.h = gp.d; t.z0 = gp.z0; t.z1 = gp.z1;

    const detail::Obstacles ob = detail::build_obstacles(world);
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

    if (v_no_obj - v_all > 0.0f)
        out.occluder = detail::name_occluder(world, t, from, g.bearing, sight, v_all, ob);
    return out;
}
}  // namespace rc::card
