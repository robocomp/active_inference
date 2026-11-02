#pragma once
/*
 * chained_action.h — EACH STEP ENDS WHERE THE NEXT ONE WANTS TO BEGIN (spec docs/superpowers/specs/2026-11-02-chained-actions-design.md).
 *
 * Live runs warm29-31: every step was chosen as if it were the last, so the robot left a door facing away from its next goal,
 * often beside a wall. The mission now looks ONE step ahead: when step k starts it resolves step k+1's first standpoint (an
 * `open`/`cross` door's approach vantage, in a frame the controller can express) and hands it to the door passage as a NextGoal.
 * Two things then follow, both Active-Inference choices over the two-step sequence rather than rules:
 *
 *   1. THE CROSS'S END is a REGION — the axis points past the door from "wholly past the leaf's swept arc" out to door_concept's
 *      published cross target — and its point minimises the cost of the NEXT step from there, in SECONDS (travel at v_max plus
 *      the turn at w_max): the one unit in which distance and heading trade without a weight to tune. Its heading is the
 *      direction of the route's first segment to the next goal.
 *   2. A `close` IS EXECUTED ON THE WAY (DoorPassage::start_close(door, chained)): the claim is published while the robot
 *      follows its route to the next goal, and door_concept serves it the moment its own precondition P(clear of the swing)
 *      holds. The robot never stops for it; the step ends when the door reads shut (1aeac589).
 *
 * The leaf's swept extent is the EXPECTED reach  s_min = base + P(leaf on the entered side) · leaf_len  from the door's
 * `door_leaf_into_room_prob` — a continuous term (0 -> the body only, 1 -> the whole leaf), no switch. Absent attribute =
 * no information: the region collapses to the published cross target (today's behaviour).
 *
 * Pure Eigen/STL — no DSR, no Qt (tests/test_chained_action.cpp).
 */

#include <algorithm>
#include <cmath>
#include <functional>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "../../common/mission/mission_ir.h"

namespace rc::chain
{

/// The next step's first standpoint, in the frame the passage drives in (the current room's).
struct NextGoal
{
    std::string door;                           ///< the door the next step is about ("" = a plain point)
    Eigen::Vector2f p = Eigen::Vector2f::Zero();
    float yaw = 0.f;                            ///< the heading the next step wants there (facing its door)
    std::string label;                          ///< "k -> k+1" for the [chain] log line
    bool operator==(const NextGoal&) const = default;
};

[[nodiscard]] inline float wrap_pi(float a)
{
    a = std::fmod(a + std::numbers::pi_v<float>, 2.f * std::numbers::pi_v<float>);
    if (a < 0.f) a += 2.f * std::numbers::pi_v<float>;
    return a - std::numbers::pi_v<float>;
}

/// The cost of the next step from a point: path length (m) and the direction (unit) of the route's first segment. The planner's
/// answer where it can give one; the straight line otherwise.
struct RouteFrom { float length_m = 0.f; Eigen::Vector2f first_dir = Eigen::Vector2f::UnitX(); };
using RouteFn = std::function<std::optional<RouteFrom>(const Eigen::Vector2f&)>;

struct EndRegion
{
    Eigen::Vector2f centre = Eigen::Vector2f::Zero();   ///< the aperture centre
    Eigen::Vector2f n = Eigen::Vector2f::UnitX();       ///< unit normal INTO the room the robot leaves (axis_from convention)
    float s_min = 0.f;                                  ///< nearest standoff past the aperture (m)
    float s_max = 0.f;                                  ///< the published cross target's standoff (m)
    float step_m = 0.05f;
    float v_max = 0.35f, w_max = 1.0f;                  ///< the base's limits: they turn metres and radians into seconds
};

struct EndChoice
{
    float s = 0.f;                       ///< standoff past the aperture
    Eigen::Vector2f p = Eigen::Vector2f::Zero();
    float yaw = 0.f;                     ///< the route's first segment toward the next goal
    float cost_s = 0.f;                  ///< cost-to-go of the next step from p (s)
    float turn_deg = 0.f;                ///< the turn from the travel heading (through the door) to yaw
    bool moved = false;                  ///< differs from the published standoff
};

/// s_min from the door's leaf belief: base + P(leaf on the entered side) · leaf_len. nullopt P = unknown -> s_max (no freedom).
[[nodiscard]] inline float swept_standoff(float base_m, std::optional<float> p_leaf_into_entered_room, float leaf_len_m, float s_max)
{
    if (not p_leaf_into_entered_room.has_value() or not std::isfinite(*p_leaf_into_entered_room)) return s_max;
    return std::min(s_max, base_m + std::clamp(*p_leaf_into_entered_room, 0.f, 1.f) * std::max(0.f, leaf_len_m));
}

/// x* = argmin over the region of  C_route(x -> V) + C_turn(travel heading -> first segment), both in seconds.
/// The point is on the axis (the hop through the aperture is a straight line, door_route.h); `route` may be empty (straight line).
[[nodiscard]] inline std::optional<EndChoice> choose_end(const EndRegion& r, const NextGoal& goal, const RouteFn& route = {})
{
    if (not goal.p.allFinite() or not r.centre.allFinite() or r.step_m <= 0.f or r.v_max <= 0.f or r.w_max <= 0.f) return std::nullopt;
    const float travel = std::atan2(-r.n.y(), -r.n.x());   // through the door = against the normal
    const float lo = std::min(r.s_min, r.s_max), hi = r.s_max;
    const int n = static_cast<int>(std::floor((hi - lo) / r.step_m + 1e-4f));
    std::optional<EndChoice> best;
    std::vector<float> standoffs;
    for (int k = 0; k <= n; ++k) standoffs.push_back(lo + r.step_m * static_cast<float>(k));
    if (std::abs(standoffs.back() - hi) > 1e-4f) standoffs.push_back(hi);   // the published end itself is always a candidate
    for (const float s : standoffs)
    {
        const Eigen::Vector2f p = r.centre - s * r.n;
        RouteFrom rf;
        if (const auto q = route ? route(p) : std::nullopt; q.has_value() and std::isfinite(q->length_m) and q->first_dir.allFinite()
                                                            and q->first_dir.norm() > 1e-6f)
            rf = {q->length_m, q->first_dir.normalized()};
        else
        {
            const Eigen::Vector2f d = goal.p - p;
            if (d.norm() < 1e-6f) rf = {0.f, Eigen::Vector2f(std::cos(travel), std::sin(travel))};
            else rf = {d.norm(), d.normalized()};
        }
        const float yaw = std::atan2(rf.first_dir.y(), rf.first_dir.x());
        const float turn = std::abs(wrap_pi(yaw - travel));
        const float cost = rf.length_m / r.v_max + turn / r.w_max;
        if (not best.has_value() or cost < best->cost_s - 1e-6f)
            best = EndChoice{.s = s, .p = p, .yaw = yaw, .cost_s = cost, .turn_deg = turn * 180.f / std::numbers::pi_v<float>,
                             .moved = std::abs(s - hi) > 1e-4f};
    }
    return best;
}

/// The step AFTER the leaf `id` in a mission's step tree (document order, descending into seq / repeat bodies), skipping the
/// ops in `skip` (a `close` runs on the way, it has no standpoint). nullptr = none / not resolvable (then: old behaviour).
[[nodiscard]] inline const rc::mission::Step* next_step(const std::vector<rc::mission::Step>& steps, const std::string& id,
                                                        const std::vector<std::string>& skip = {"close"})
{
    std::vector<const rc::mission::Step*> flat;
    const auto walk = [&](auto&& self, const std::vector<rc::mission::Step>& list) -> void
    {
        for (const auto& s : list)
        {
            if ((s.op == "seq" or s.op == "repeat") and not s.steps.empty()) self(self, s.steps);
            else flat.push_back(&s);
        }
    };
    walk(walk, steps);
    bool seen = false;
    for (const auto* s : flat)
    {
        if (seen and std::ranges::find(skip, s->op) == skip.end()) return s;
        if (s->id == id) seen = true;
    }
    return nullptr;
}

/// The step right after `id` whatever it is (no skipping): what a cross's NEXT step is, for deciding "close behind comes next".
[[nodiscard]] inline const rc::mission::Step* following_step(const std::vector<rc::mission::Step>& steps, const std::string& id)
{
    return next_step(steps, id, {});
}

}   // namespace rc::chain
