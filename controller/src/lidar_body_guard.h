#pragma once
/*
 * lidar_body_guard.h — a physical safety net that does NOT trust the anchor (Task 14, 2026-10-19).
 *
 * try74 leg 2: the published door anchor was 0.18–0.23 m off laterally at a claimed σ_lat of 0.016 m (≈12σ). The anchor
 * guards (controller_anchor_servo.h) trust the anchor, so they let the servo drive the body's front-left corner into the
 * REAL east jamb. The controller's live LiDAR field could not stop it either, for two reasons:
 *   1. its self-filter (TrajectoryController::read_lidar_points_robot) deletes every return within
 *      max(0.40, circumscribed + 0.08) = 0.405 m of the centre, plus a box |x| < 0.32, −0.24 < y < 0.42 — so a jamb at
 *      contact range (≤ 0.33 m) is not in the field at all;
 *   2. its ceiling (ControllerSession::anchor_clearance_speed_limit) is floored at 0.05 m/s forward.
 *
 * This header reads the returns themselves:
 *
 * SELF-FILTER = THE BODY. A return is the robot seeing itself only if it lies inside the robot's footprint polygon grown
 * by the LiDAR's range noise σ_r (signed distance < σ_r). ⚑ DECISION LEVEL, flagged: a return is either kept or deleted;
 * the level is the sensor's own advertised 1-σ range noise (lidar3d_dds [SensorModel] RangeSigmaFloorM: helios 0.02,
 * bpearl 0.03 m), not a tuned slack. What the old disc protected against — the body's own parts (wheels at 0.246–0.27 m,
 * the r ≈ 0 return population, ROBOT_GEOMETRY.md) — is all INSIDE the footprint (the hull already bounds the wheels by
 * their circumscribing sphere), so it stays filtered; the 0.08 m of disc slack beyond the body was a disc-robot
 * approximation (commit 7339252e: robot_radius + 0.08, "beyond the body, a return is the world, not us").
 *
 * Each kept return is an obstacle of the same kind the anchor guards use (rc::AnchorJamb): mean = the return (robot frame),
 * cov = σ_r² I, radius 0, P(there) = 1 — a measurement, egocentric, so no pose error moves it.
 *
 * TRANSLATION along unit u (p ⟂ u), on the swept body path. The body's corridor is |q·p| ≤ support(±p) of the RAW body;
 * a return counts with P_in = w((support(side) − |ℓ|)/σ_p) (w = anchor_contact_weight: a return the mean puts on the
 * corridor edge is a certain graze). Its gap to contact is d = q·u − f(ℓ), f = the body's LEADING profile at lateral
 * offset ℓ (clamped to the corridor, so a return just outside it is measured against the corner). It is CLOSING with
 * P_c = w(d/σ_u): ahead of the leading face ⇒ certain; beside or behind the body (d ≪ 0) ⇒ 0. The stopping law then
 * allows v_block = Φ(g/σ_u)·sqrt(2·a_dec·max(0, g)), g = d − margin (the safety margin as a STANDOFF along the motion),
 * and the ceiling is v_cap − P_in·P_c·(v_cap − v_block). NO FLOOR: a closing return at contact gives 0.
 *   Why not anchor_jamb_speed_limit on these points: that law grows the corridor by the margin and counts any point ahead
 *   of the CENTRE as closing — the slack an UNCERTAIN anchor needs. A LiDAR return beside the body inside the margin band
 *   would then read as blocking forward motion, and every real jamb return in a 0.9 m door is in that band. Lateral
 *   margin is pose uncertainty; a return in the robot frame has none beyond σ_r.
 *
 * ROTATION: anchor_obstacles_rot_limit on the same returns — the grown body's corners swept onto each return, the turn
 * capped by the stopping law on the remaining angle, unfloored; a return already inside the grown body caps only the turn
 * that closes on it.
 *
 * Pure Eigen/STL (pulls controller_anchor_servo.h). Tested in tests/test_lidar_body_guard.cpp.
 */

#include "controller_anchor_servo.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <unordered_map>
#include <vector>
#include <Eigen/Dense>

namespace rc
{

// Is this robot-frame return the robot seeing itself? Inside the raw footprint grown by the range noise.
[[nodiscard]] inline bool lidar_return_is_self(std::span<const Eigen::Vector2f> body, float sigma_r, const Eigen::Vector2f& q)
{
    return anchor_signed_distance(body, q) < sigma_r;
}

// The OLD controller self-filter (TrajectoryController::read_lidar_points_robot before Task 14), kept as a pure function
// only so its losses can be counted against the footprint filter (task-14-report.md). Not used to drive.
[[nodiscard]] inline bool lidar_return_dropped_by_disc(const Eigen::Vector2f& q, float circumscribed)
{
    const float radius = std::max(0.40f, circumscribed + 0.08f);
    if (q.norm() < radius) return true;
    return std::abs(q.x()) < 0.32f and q.y() > -0.24f and q.y() < 0.42f;
}

struct LidarGuardReturns
{
    std::vector<AnchorJamb> obstacles;   // kept returns within reach, one per σ-cell
    int n_self = 0;                      // deleted as the body
    int n_far = 0;                       // beyond reach: cannot limit any motion this cycle
};

// Robot-frame returns → guard obstacles. `reach` = the radius beyond which no return can cap a motion this cycle: the
// grown body's reach plus the braking distance from v_cap (ARITHMETIC: there the stopping law gives ≥ v_cap and the sweep
// cannot reach). Returns are merged one per σ_r cell (the nearest to the centre) — a discretisation below the sensor's
// own noise, so it changes no answer by more than σ_r, and it bounds the per-cycle cost.
[[nodiscard]] inline LidarGuardReturns lidar_guard_returns(std::span<const Eigen::Vector2f> points_robot,
                                                           std::span<const Eigen::Vector2f> body, float sigma_r, float reach)
{
    LidarGuardReturns out;
    const float s = std::max(sigma_r, 1e-3f);
    const Eigen::Matrix2f cov = (s * s) * Eigen::Matrix2f::Identity();
    std::unordered_map<std::int64_t, std::size_t> cell;
    for (const auto& q : points_robot)
    {
        if (not q.allFinite()) continue;
        if (q.norm() > reach) { ++out.n_far; continue; }
        if (lidar_return_is_self(body, sigma_r, q)) { ++out.n_self; continue; }
        const auto ci = static_cast<std::int64_t>(std::floor(q.x() / s)), cj = static_cast<std::int64_t>(std::floor(q.y() / s));
        const std::int64_t key = (ci << 32) ^ (cj & 0xffffffff);
        if (const auto it = cell.find(key); it != cell.end())
        {
            if (q.norm() < out.obstacles[it->second].mean.norm()) out.obstacles[it->second].mean = q;
            continue;
        }
        cell.emplace(key, out.obstacles.size());
        out.obstacles.push_back(AnchorJamb{q, cov, 0.f, 1.f});
    }
    return out;
}

// The body's leading profile along u at lateral offset l (p = u rotated +90°): max of x·u over the polygon's chord x·p = l.
// l outside the polygon's lateral extent ⇒ −∞.
[[nodiscard]] inline float lidar_leading_profile(std::span<const Eigen::Vector2f> body, const Eigen::Vector2f& u,
                                                 const Eigen::Vector2f& p, float l)
{
    float best = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < body.size(); ++i)
    {
        const Eigen::Vector2f a = body[i], b = body[(i + 1) % body.size()];
        const float la = a.dot(p), lb = b.dot(p), aa = a.dot(u), ab = b.dot(u);
        if (l < std::min(la, lb) or l > std::max(la, lb)) continue;
        const float t = lb != la ? (l - la) / (lb - la) : 0.f;
        best = std::max(best, aa + t * (ab - aa));
        if (lb == la) best = std::max(best, std::max(aa, ab));
    }
    return best;
}

// The translation ceiling along u (robot frame, any direction: a reverse is u = (0, −1)) from the returns. Unfloored.
[[nodiscard]] inline float lidar_return_speed_limit(std::span<const AnchorJamb> returns, const Eigen::Vector2f& u,
                                                    std::span<const Eigen::Vector2f> body, float margin, float a_dec,
                                                    float v_cap)
{
    if (not (u.squaredNorm() > 0.f) or body.size() < 3 or returns.empty()) return v_cap;
    const Eigen::Vector2f un = u.normalized();
    const Eigen::Vector2f p(-un.y(), un.x());
    const float s_plus = anchor_body_support(body, 0.f, p), s_minus = anchor_body_support(body, 0.f, -p);
    float v = v_cap;
    for (const auto& j : returns)
    {
        const float lat = j.mean.dot(p), along = j.mean.dot(un);
        const float s_u = anchor_axis_sigma(j.cov, un), s_p = anchor_axis_sigma(j.cov, p);
        const float edge = lat >= 0.f ? s_plus : s_minus;
        const float p_in = anchor_contact_weight((edge - std::abs(lat)) / s_p);
        const float f = lidar_leading_profile(body, un, p, std::clamp(lat, -s_minus, s_plus));
        if (not std::isfinite(f)) continue;
        const float d = along - f;
        const float p_closing = anchor_contact_weight(d / s_u);
        const float g = d - margin;
        const float v_block = std::min(v_cap, anchor_std_normal_cdf(g / s_u) * std::sqrt(2.f * std::max(0.f, a_dec) * std::max(0.f, g)));
        v = std::min(v, v_cap - j.p * p_in * p_closing * (v_cap - v_block));
    }
    return std::max(0.f, v);
}

// The rotation ceiling for a turn in the direction of `rot` from the returns: the corner sweep, unfloored.
[[nodiscard]] inline float lidar_return_rot_limit(std::span<const AnchorJamb> returns, float rot,
                                                  std::span<const Eigen::Vector2f> body, float margin, float a_dec, float w_cap)
{
    return anchor_obstacles_rot_limit(returns, rot, body, margin, a_dec, w_cap);
}

}  // namespace rc
