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
#include <optional>
#include <string>
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
    const std::vector<nbv::Obstacle> walls = nbv::wall_obstacles(world.polygon, detail::kWallThicknessM, world.gaps);
    std::vector<nbv::Obstacle> objs, all = walls;
    for (const auto& [name, o] : world.obstacles) { objs.push_back(o); all.push_back(o); }

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
    {
        float best_gain = 0.0f;
        for (std::size_t k = 0; k < world.obstacles.size(); ++k)
        {
            std::vector<nbv::Obstacle> rest = walls;
            for (std::size_t m = 0; m < world.obstacles.size(); ++m)
                if (m != k) rest.push_back(world.obstacles[m].second);
            const float gain = nbv::visible_fraction(t, from, yaw, sensor, rest) - v_all;
            if (gain > best_gain) { best_gain = gain; out.occluder = world.obstacles[k].first; }
        }
    }
    return out;
}
}  // namespace rc::card
