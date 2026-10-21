// field_extent.h — Task 3f (b) (tour3A leg 1): WHERE the local policy's cost-to-go lives, and what it says about the cells the
// planner's map does not cover.
//
// ── WHY ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
// tour3A: after the room switch the doorway goal line (2.4 m past the door) lay OUTSIDE the newborn room_2's planner grid
// (its first layout was a 2.1 m-deep rectangle; the grid ended at y 2.71, the goal points sat at y 2.85–3.00). The field was
// built over the planner's extent only, CostToGo dropped every off-grid goal point, and the field was +inf everywhere: the
// policy could only Release, for 9 minutes. An all-inf field is a modelling failure, not information about the world.
//
// ── THE MODEL ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
// (1) The field's grid is the UNION of the planner's extent, the goal points and the robot, each padded by the body's
//     circumscribed radius plus a field cell (so a goal point is never off the grid it seeds, and the doorway corridor that
//     joins the door to its goal line — inside the convex box of robot and goal — is on it too).
// (2) A cell OUTSIDE the planner's extent is UNSEEN, not "outside the world": the planner's raster says nothing about it.
//     · Its distance to the nearest KNOWN obstacle (extended_distance) = the box border's EDT value plus the way out to it —
//       an UPPER bound on the true clearance (an unseen obstacle beyond the box can only make it smaller), continuous with the
//       EDT inside: the planner's border cells are its outside-room margin, so a cell beside the box is as close to a wall
//       as the box lets it be, never a free strip hugging a wall. The price below, and the contact model's returns, carry
//       what the bound cannot.
//     · Its PRICE is the honest prior for unseen space — residual_risk.h's surprisal −log Pr(θ < ½) at a SYMMETRIC prior with
//       no evidence (residual's Jeffreys Beta(½,½); any symmetric Beta gives Pr(θ < ½) = ½): ln 2 nats per planner cell,
//       summed over the cells the body covers there (the A*'s own per-cell convention, GridPlanner::body_risk_at), turned
//       into metres per metre by the same residual_risk_weight. Not free (0) — that would invent seen free space; not a
//       wall (+inf) — nothing was seen there.
//     The robot's contact model (returns) still judges every motion into it; the price only orders the routes.
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <numbers>
#include <span>
#include <Eigen/Dense>

#include "cost_to_go.h"

namespace rc::policy
{
// The planner's raster as a box in the room frame: [lo, hi] = the outer edges of its cells.
struct PlannerExtent
{
    Eigen::Vector2f lo = Eigen::Vector2f::Zero(), hi = Eigen::Vector2f::Zero();
    [[nodiscard]] bool valid() const { return (hi.array() > lo.array()).all(); }
    [[nodiscard]] bool contains(const Eigen::Vector2f& p) const
    {
        return valid() and (p.array() >= lo.array()).all() and (p.array() <= hi.array()).all();
    }
};

// (1) the field's grid: cell centres at origin + (i,j)·cell, covering extent ∪ goal points ∪ robot (each padded by `pad`).
[[nodiscard]] inline CostToGo::Grid field_grid(const PlannerExtent& ext, std::span<const Eigen::Vector2f> goal_points,
                                               const Eigen::Vector2f& robot, float pad, float cell)
{
    Eigen::Vector2f lo = ext.valid() ? ext.lo : robot, hi = ext.valid() ? ext.hi : robot;
    auto grow = [&](const Eigen::Vector2f& p)
    {
        if (not p.allFinite()) return;
        lo = lo.cwiseMin(p - Eigen::Vector2f::Constant(pad));
        hi = hi.cwiseMax(p + Eigen::Vector2f::Constant(pad));
    };
    grow(robot);
    for (const auto& q : goal_points) grow(q);
    const Eigen::Vector2f span = hi - lo;
    return CostToGo::Grid{lo, cell, static_cast<int>(std::ceil(span.x() / cell)) + 1, static_cast<int>(std::ceil(span.y() / cell)) + 1};
}

// (2) the distance to the nearest known obstacle, extended past the planner's box: inside it the planner's EDT; outside, the
// EDT at the nearest point of the box plus the way out to p — an UPPER bound on the true clearance (nothing is known beyond).
[[nodiscard]] inline float extended_distance(const PlannerExtent& ext, const Eigen::Vector2f& p,
                                             const std::function<float(const Eigen::Vector2f&)>& distance_in)
{
    if (not ext.valid() or ext.contains(p)) return distance_in(p);
    const Eigen::Vector2f q = p.cwiseMax(ext.lo).cwiseMin(ext.hi);
    return distance_in(q) + (p - q).norm();
}

// (2) the price of standing on unseen space, as a cost-to-go DENSITY (metres per metre): ln 2 nats per planner cell under
// the body (its area / cell²) — −log Pr(θ < ½) = ln 2 at any symmetric no-evidence Beta (residual's Jeffreys Beta(½,½)) —
// times the weight that turns the planner's nats into metres.
[[nodiscard]] inline float unseen_prior_density(float body_area_m2, float planner_cell_m, float risk_weight)
{
    if (not (body_area_m2 > 0.f) or not (planner_cell_m > 0.f)) return 0.f;
    return std::max(0.f, risk_weight) * std::numbers::ln2_v<float> * body_area_m2 / (planner_cell_m * planner_cell_m);
}
}  // namespace rc::policy
