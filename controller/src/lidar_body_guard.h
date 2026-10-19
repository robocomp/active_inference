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
 * SELF-FILTER = THE BODY. A return is the robot seeing itself only if it lies STRICTLY INSIDE the robot's footprint
 * polygon (signed distance < 0). ⚑ DECISION LEVEL, flagged: a return is either kept or deleted, and the level is the hull
 * itself — no slack. What the old disc protected against — the body's own parts (wheels at 0.246–0.27 m, the r ≈ 0 return
 * population, ROBOT_GEOMETRY.md) — is all INSIDE the footprint (the hull already bounds the wheels by their circumscribing
 * sphere), so it stays filtered; the 0.08 m of disc slack beyond the body was a disc-robot approximation (commit 7339252e:
 * robot_radius + 0.08, "beyond the body, a return is the world, not us").
 * ★ Fix round 1 (review): the first version deleted the σ_r band OUTSIDE the hull as well. That made the guard
 *   NON-MONOTONE — a jamb closing to within σ_r of the hull (lag, overshoot, the 2–4 cm pose-lag shift of a room-frame
 *   buffer read on the control pose) vanished, and the cap went back to v_cap in the last centimetres: try74's failure
 *   again. A return in that band is now kept and is CONTACT for the motion that closes on it (cap 0). The price: any part
 *   of the robot the hull misses (a cable, a mount) reads as contact — the census (lidar_guard_census) counts that band
 *   (n_band) at start-up and on every cross so such a part is SEEN; σ_r must not be widened to hide it.
 *
 * Each kept return is an obstacle of the same kind the anchor guards use (rc::AnchorJamb): mean = the return (robot frame),
 * cov = σ_r² I, radius 0, P(there) = 1 — a measurement, egocentric, so no pose error moves it.
 *
 * TRANSLATION along unit u (p ⟂ u), on the swept body path. The body's corridor is |q·p| ≤ support(±p) of the RAW body;
 * a return counts with P_in = w((support(side) − |ℓ|)/σ_p) (w = anchor_contact_weight: a return the mean puts on the
 * corridor edge is a certain graze). Its gap to contact is d = q·u − f(ℓ), f = the body's LEADING profile at lateral
 * offset ℓ (clamped to the corridor, so a return just outside it is measured against the corner). It is CLOSING with
 * P_c = w(d/σ_u): ahead of the leading face ⇒ certain; beside or behind the body (d ≪ 0) ⇒ 0. The stopping law then
 * allows v_block = Φ(g/σ_u)·sqrt(2·a_dec·max(0, g)), g = d − max(margin, σ_r) (the safety margin as a STANDOFF along the
 * motion, never inside the noise band: a return within σ_r of the hull is contact),
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
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>
#include <Eigen/Dense>

namespace rc
{

// Is this robot-frame return the robot seeing itself? Strictly inside the raw footprint (see the header).
[[nodiscard]] inline bool lidar_return_is_self(std::span<const Eigen::Vector2f> body, const Eigen::Vector2f& q)
{
    return anchor_signed_distance(body, q) < 0.f;
}

// ★ Fix round 3 (try77): THE SELF DECISION IS MADE AT CAPTURE, IN THE RIGID ROBOT FRAME. The controller's LiDAR buffer is
// ROOM-frame (each plane placed with room←lidar at its own stamp) and the reader re-expresses it on the CURRENT pose. For a
// return off the WORLD that is exact (the point is room-fixed, so the current pose is the right one to see it from). For a
// return off the ROBOT ITSELF it is wrong by the motion since capture — |ω|·r·lag + |v|·lag, ~2.4 cm on a wheel face at
// 1 rad/s and 80 ms — enough to carry a self return out of the hull into the contact band. The robot's own returns are
// fixed in the robot frame, and robot←lidar is a RIGID mount, so at capture the test is exact with no lag term at all:
// the obstacle tracker's buffer converter calls this on robot←lidar·p and drops the robot's returns before the cloud ever
// becomes room-frame. `r_circ` = the hull's circumscribed radius: a cheap reject for the ~all points far from the body.
[[nodiscard]] inline bool lidar_capture_is_self(std::span<const Eigen::Vector2f> body, float r_circ, const Eigen::Vector2f& q_robot)
{
    if (body.size() < 3 or q_robot.squaredNorm() > r_circ * r_circ) return false;
    return anchor_point_in_convex(body, q_robot);
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
    int n_self = 0;                      // deleted as the body (strictly inside the hull)
    int n_band = 0;                      // kept, within σ_r OUTSIDE the hull: contact — or a body part the hull misses
    int n_far = 0;                       // beyond reach: cannot limit any motion this cycle
    // Stage counts (try76 fix): offered → finite → within reach → outside the hull. A census of an EMPTY near field is
    // otherwise indistinguishable from an empty cloud.
    int n_read = 0, n_nonfinite = 0;
    // The nearest finite return NOT inside the hull, at ANY range (robot frame, by distance from the centre), so a census
    // can say where the world is when nothing is within reach.
    std::optional<Eigen::Vector2f> nearest_any;
    // Fix round 3: the band returns themselves (robot frame), so their place on the body can be logged — a body part the
    // hull misses sits at a FIXED body position; a lag artefact moves with the motion.
    std::vector<Eigen::Vector2f> band_points;
};

// Robot-frame returns → guard obstacles. `reach` = the radius beyond which no return can cap a motion this cycle: the
// grown body's reach plus the braking distance from v_cap (ARITHMETIC: there the stopping law gives ≥ v_cap and the sweep
// cannot reach). Returns are merged one per σ_r cell (the nearest to the centre) — a discretisation below the sensor's
// own noise, so it changes no answer by more than σ_r, and it bounds the per-cycle cost. `next(i)` yields the i-th return
// in the robot frame (or nullopt to skip it), so a caller streams its own cloud without an intermediate copy.
template <class Next>
[[nodiscard]] inline LidarGuardReturns lidar_guard_returns_from(std::size_t n, Next&& next,
                                                                std::span<const Eigen::Vector2f> body, float sigma_r, float reach)
{
    LidarGuardReturns out;
    const float s = std::max(sigma_r, 1e-3f);
    const Eigen::Matrix2f cov = (s * s) * Eigen::Matrix2f::Identity();
    std::unordered_map<std::int64_t, std::size_t> cell;
    for (std::size_t i = 0; i < n; ++i)
    {
        ++out.n_read;
        const std::optional<Eigen::Vector2f> qo = next(i);
        if (not qo.has_value() or not qo->allFinite()) { ++out.n_nonfinite; continue; }
        const Eigen::Vector2f q = *qo;
        const auto note_nearest = [&] { if (not out.nearest_any.has_value() or q.squaredNorm() < out.nearest_any->squaredNorm()) out.nearest_any = q; };
        if (q.norm() > reach) { ++out.n_far; note_nearest(); continue; }
        const float sd = anchor_signed_distance(body, q);
        if (sd < 0.f) { ++out.n_self; continue; }
        note_nearest();
        if (sd < sigma_r) { ++out.n_band; out.band_points.push_back(q); }
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

// The session's entry point: a ROOM-frame cloud (the obstacle tracker's lidar buffer: z-banded, room frame) and the robot's
// room pose as `room_T_robot` (ControllerRobotPose::as_transform — robot frame x right, y forward), streamed into the robot
// frame with no intermediate copy. tests/test_lidar_body_guard.cpp drives it through the real buffer (lidar_guard_buffer.h).
[[nodiscard]] inline LidarGuardReturns lidar_guard_returns_room(std::span<const float> xs_room, std::span<const float> ys_room,
                                                                const Eigen::Affine2f& room_T_robot,
                                                                std::span<const Eigen::Vector2f> body, float sigma_r, float reach)
{
    const Eigen::Affine2f robot_from_room = room_T_robot.inverse();
    return lidar_guard_returns_from(std::min(xs_room.size(), ys_room.size()),
                                    [&](std::size_t i) -> std::optional<Eigen::Vector2f>
                                    {
                                        if (not std::isfinite(xs_room[i]) or not std::isfinite(ys_room[i])) return std::nullopt;
                                        return robot_from_room * Eigen::Vector2f(xs_room[i], ys_room[i]);
                                    },
                                    body, sigma_r, reach);
}

[[nodiscard]] inline LidarGuardReturns lidar_guard_returns(std::span<const Eigen::Vector2f> points_robot,
                                                           std::span<const Eigen::Vector2f> body, float sigma_r, float reach)
{
    return lidar_guard_returns_from(points_robot.size(),
                                    [&](std::size_t i) { return std::optional<Eigen::Vector2f>(points_robot[i]); },
                                    body, sigma_r, reach);
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
    // σ_r is each return's own covariance (isotropic: cov = σ_r² I), so it is read back from there.
    if (not (u.squaredNorm() > 0.f) or body.size() < 3 or returns.empty()) return v_cap;
    const Eigen::Vector2f un = u.normalized();
    const Eigen::Vector2f p(-un.y(), un.x());
    const float s_plus = anchor_body_support(body, 0.f, p), s_minus = anchor_body_support(body, 0.f, -p);
    float v = v_cap;
    for (const auto& j : returns)
    {
        const float lat = j.mean.dot(p), along = j.mean.dot(un);
        const float s_u = anchor_axis_sigma(j.cov, un), s_p = anchor_axis_sigma(j.cov, p);
        const float sigma_contact = std::sqrt(std::max(0.f, 0.5f * j.cov.trace()));
        const float edge = lat >= 0.f ? s_plus : s_minus;
        const float p_in = anchor_contact_weight((edge - std::abs(lat)) / s_p);
        const float f = lidar_leading_profile(body, un, p, std::clamp(lat, -s_minus, s_plus));
        if (not std::isfinite(f)) continue;
        const float d = along - f;
        const float p_closing = anchor_contact_weight(d / s_u);
        // The standoff: the safety margin, and never less than σ_r — a return in the noise band outside the hull is CONTACT
        // for the motion that closes on it (fix round 1: monotone to the last centimetre, whatever the margin).
        const float g = d - std::max(margin, sigma_contact);
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

// ── THE RETURNS' DEADLOCK AND ITS WAY OUT (fix round 1, review item 2; the LiDAR twin of anchor_jamb_guard's fix C) ─────
// A return inside the band the caps act on (the hull grown by margin + σ_r) closes forward AND both turns when it pins a
// corner — the base then stands still against a real obstacle. The way out is the motion that OPENS clearance, judged on
// the same returns: candidates reverse / advance (/ ±side on a holonomic base) / turn either way, each scored by the drop
// of the total band penetration B = Σ max(0, (margin + σ_r) − sd_i) per metre of body-point travel (a turn by its arm,
// the worst return's radius); the largest drop taken. A SUM, not the min: a jamb face lying alongside the body is not
// opened by reversing, and with the min that face would veto the reverse that frees the pinned corner. Speed: the
// deficit closed on the approach loop's time constant, v = v_max·tanh(pen/(v_max·τ_s)), pen = the worst penetration.
// The result then goes through the returns' own caps, so it cannot close on another return. Nothing opens ⇒ nullopt.
[[nodiscard]] inline float lidar_band_penetration(std::span<const AnchorJamb> returns, std::span<const Eigen::Vector2f> body,
                                                  float band, float side, float adv, float dth, float* worst = nullptr)
{
    float sum = 0.f, w = 0.f;
    for (const auto& j : returns)
    {
        const float pen = band - anchor_signed_distance(body, anchor_point_after(j.mean, side, adv, dth));
        if (pen > 0.f) { sum += pen; w = std::max(w, pen); }
    }
    if (worst != nullptr) *worst = w;
    return sum;
}

// ★ Fix round 4 (try79: parked 300 s alongside room_2's east wall, 5 cm off its side). The first version scored single
//   motions over 1 mm and capped the winner with the SAFETY MARGIN — and a differential base parallel to a wall has no
//   single motion that opens it: advance/reverse leave the side gap unchanged, and either turn swings one end of the side
//   (or a corner) toward the wall, which the margin-capped turn ceiling refuses. Two changes, both of them the model's:
//   1. TWO-SEGMENT PRIMITIVES at the deficit's own scale — a turn, a translation, or a turn THEN a translation — so
//      "turn away a little, then drive out" is a candidate. Scale: the deficit D = the worst band penetration (≥ σ_r); a
//      translation moves the body D, a turn moves its farthest point D (θ = D / r_max). Scored by the drop of the total
//      band penetration at the END, and REJECTED if any return touches the hull (sd ≤ 0) along the way (sampled every
//      σ_r/2 of body-point travel — ARITHMETIC, a discretisation below the noise).
//   2. The backoff is capped by the returns with the CONTACT standoff only (margin 0 ⇒ max(0, σ_r)): its job is to give
//      the margin back, so it may move inside the margin band as long as it does not close on contact.
//   Receding horizon: only the first segment is commanded, and the next cycle chooses again.
struct BackoffSeg { float side = 0.f, adv = 0.f, rot = 0.f; };

[[nodiscard]] inline std::optional<std::vector<Eigen::Vector2f>> lidar_backoff_rollout(std::span<const AnchorJamb> returns,
                                                                                        std::span<const Eigen::Vector2f> body,
                                                                                        std::span<const BackoffSeg> segs,
                                                                                        float r_max, float step)
{
    std::vector<Eigen::Vector2f> pts;
    pts.reserve(returns.size());
    for (const auto& j : returns) pts.push_back(j.mean);
    for (const auto& sg : segs)
    {
        const float travel = std::hypot(sg.side, sg.adv) + std::abs(sg.rot) * r_max;
        const int n = std::max(1, int(std::ceil(travel / std::max(step, 1e-4f))));
        for (int k = 0; k < n; ++k)
        {
            for (auto& q : pts)
            {
                q = anchor_point_after(q, sg.side / float(n), sg.adv / float(n), sg.rot / float(n));
                if (anchor_signed_distance(body, q) <= 0.f) return std::nullopt;   // touches the body: rejected
            }
        }
    }
    return pts;
}

[[nodiscard]] inline std::optional<AnchorCommand> lidar_return_backoff(std::span<const AnchorJamb> returns,
                                                                       std::span<const Eigen::Vector2f> body, float margin,
                                                                       float sigma_r, float a_dec, float v_max, float w_max,
                                                                       bool holonomic)
{
    if (returns.empty() or body.size() < 3) return std::nullopt;
    const float band = margin + sigma_r;
    float pen = 0.f;
    const float b0 = lidar_band_penetration(returns, body, band, 0.f, 0.f, 0.f, &pen);
    if (not (b0 > 0.f)) return std::nullopt;
    float r_max = 1e-3f;
    for (const auto& v : body) r_max = std::max(r_max, v.norm());
    const float D = std::max(pen, sigma_r);
    const float th = D / r_max;
    std::vector<std::vector<BackoffSeg>> cand;
    for (const float sgn : {1.f, -1.f})
    {
        cand.push_back({BackoffSeg{0.f, sgn * D, 0.f}});
        cand.push_back({BackoffSeg{0.f, 0.f, sgn * th}});
        for (const float sa : {1.f, -1.f}) cand.push_back({BackoffSeg{0.f, 0.f, sgn * th}, BackoffSeg{0.f, sa * D, 0.f}});
        if (holonomic) cand.push_back({BackoffSeg{sgn * D, 0.f, 0.f}});
    }
    const auto band_sum = [&](const std::vector<Eigen::Vector2f>& pts)
    {
        float s = 0.f;
        for (const auto& q : pts) s += std::max(0.f, band - anchor_signed_distance(body, q));
        return s;
    };
    float best_gain = 0.f;
    const std::vector<BackoffSeg>* best = nullptr;
    for (const auto& c : cand)
        if (const auto end = lidar_backoff_rollout(returns, body, c, r_max, 0.5f * sigma_r); end.has_value())
            if (const float gain = b0 - band_sum(*end); gain > best_gain) { best_gain = gain; best = &c; }
    if (best == nullptr) return std::nullopt;
    const BackoffSeg& first = best->front();
    const float v = v_max * std::tanh(D / (v_max * kAnchorTauS));
    AnchorCommand c;
    if (first.rot != 0.f) c.rot = std::clamp(std::copysign(v / r_max, first.rot), -w_max, w_max);
    else
    {
        const float len = std::hypot(first.side, first.adv);
        c.side = v * first.side / len;
        c.adv = v * first.adv / len;
    }
    // Through the returns' own caps, at the CONTACT standoff (margin 0: the backoff may use the margin band, not contact).
    if (const float lin = std::hypot(c.adv, c.side); lin > 0.f)
    {
        const float vl = lidar_return_speed_limit(returns, Eigen::Vector2f(c.side / lin, c.adv / lin), body, 0.f, a_dec, v_max);
        if (lin > vl) { c.adv *= vl / lin; c.side *= vl / lin; }
    }
    if (c.rot != 0.f)
        c.rot = std::copysign(std::min(std::abs(c.rot), lidar_return_rot_limit(returns, c.rot, body, 0.f, a_dec, w_max)), c.rot);
    if (c.adv == 0.f and c.side == 0.f and c.rot == 0.f) return std::nullopt;
    return c;
}

// ── THE GUARD APPLIED TO A COMMAND (robot-frame convention: side = +x right, rot CCW positive) ─────────────────────────
struct LidarCapResult { bool bound_lin = false, bound_rot = false; float v_lim = 0.f, w_lim = 0.f; };
inline LidarCapResult lidar_cap_command(std::span<const AnchorJamb> returns, std::span<const Eigen::Vector2f> body,
                                        float margin, float a_dec, float v_max, float w_max,
                                        float& adv, float& side, float& rot)
{
    LidarCapResult r{.v_lim = v_max, .w_lim = w_max};
    if (const float lin = std::hypot(adv, side); lin > 0.f)
    {
        r.v_lim = lidar_return_speed_limit(returns, Eigen::Vector2f(side / lin, adv / lin), body, margin, a_dec, v_max);
        if (lin > r.v_lim) { r.bound_lin = true; adv *= r.v_lim / lin; side *= r.v_lim / lin; }
    }
    if (rot != 0.f)
    {
        r.w_lim = lidar_return_rot_limit(returns, rot, body, margin, a_dec, w_max);
        if (std::abs(rot) > r.w_lim) { r.bound_rot = true; rot = std::copysign(r.w_lim, rot); }
    }
    return r;
}

// ── THE CENSUS (review item 4): is anything the LiDAR sees beyond the hull that the hull should hold? ────────────────
struct LidarGuardCensus
{
    int n_read = 0, n_nonfinite = 0, n_far = 0, n_self = 0, n_band = 0, n_kept = 0;
    float nearest_any_sd = std::numeric_limits<float>::infinity();   // the nearest return at any range, to the hull (m)
    float nearest_any_bearing_deg = 0.f;
    float nearest_sd = std::numeric_limits<float>::infinity();   // the nearest kept return's distance to the hull (m)
    float nearest_bearing_deg = 0.f;                             // its bearing, 0 = forward, + = right
    float v_fwd = 0.f, v_back = 0.f, w_ccw = 0.f, w_cw = 0.f;    // the caps the returns set right now
};
[[nodiscard]] inline LidarGuardCensus lidar_guard_census(const LidarGuardReturns& r, std::span<const Eigen::Vector2f> body,
                                                         float margin, float a_dec, float v_max, float w_max)
{
    LidarGuardCensus c{.n_read = r.n_read, .n_nonfinite = r.n_nonfinite, .n_far = r.n_far, .n_self = r.n_self,
                       .n_band = r.n_band, .n_kept = int(r.obstacles.size())};
    if (r.nearest_any.has_value())
    {
        c.nearest_any_sd = anchor_signed_distance(body, *r.nearest_any);
        c.nearest_any_bearing_deg = std::atan2(r.nearest_any->x(), r.nearest_any->y()) * 57.29578f;
    }
    for (const auto& j : r.obstacles)
        if (const float sd = anchor_signed_distance(body, j.mean); sd < c.nearest_sd)
        {
            c.nearest_sd = sd;
            c.nearest_bearing_deg = std::atan2(j.mean.x(), j.mean.y()) * 57.29578f;
        }
    c.v_fwd = lidar_return_speed_limit(r.obstacles, Eigen::Vector2f(0.f, 1.f), body, margin, a_dec, v_max);
    c.v_back = lidar_return_speed_limit(r.obstacles, Eigen::Vector2f(0.f, -1.f), body, margin, a_dec, v_max);
    c.w_ccw = lidar_return_rot_limit(r.obstacles, +1.f, body, margin, a_dec, w_max);
    c.w_cw = lidar_return_rot_limit(r.obstacles, -1.f, body, margin, a_dec, w_max);
    return c;
}

}  // namespace rc
