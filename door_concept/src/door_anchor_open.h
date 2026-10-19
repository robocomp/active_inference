/*
 * door_anchor_open.h — Task 11 (try69 leg 2): what the passage anchor's own rays say about whether its door is OPEN,
 * and which pose the controller's not-seen wait-turn may aim at.
 *
 * ─── FIX 2: THE ANCHOR'S RAYS ARE LOCAL EVIDENCE ABOUT THE LEAF ─────────────────────────────────────────────────────
 * The user's idea: "the door model can predict that the lidar will pass through, check it and use it as local evidence".
 * While a passage anchor exists, every measured cycle hands update_door_anchor a sweep that crosses the opening, and
 * the opening's posterior says exactly where each beam meets its plane. Under SHUT the leaf closes the span, so a beam
 * crossing inside it ends AT the plane (range noise, the leaf's thickness and the pose's own depth spread); under OPEN
 * it flies on into the next room. This file scores the two hypotheses as LIKELIHOODS, per beam, with the same
 * density phi_field_loglik writes (DoorBelief, door_belief.cpp: a surface hit, an unexpected short return, a miss
 * beyond), and returns the frame's log-likelihood ratio  Λ = log p(rays | open) − log p(rays | shut).
 *   · A beam that never reaches the plane (an occluder, a floor return in front of it) has the same density under both
 *     and contributes 0. A beam crossing the WALL beside the opening is the shut leaf's density under both: its span
 *     membership m = P(|u| < w/2) under the posterior's lateral spread mixes the two continuously, no span gate.
 *   · NOT counts. The Beta(1,1) count posterior AnchorEvidence::p_open "manufactures confidence" (fractional counts,
 *     MEMORY: fractional-counts-manufacture-confidence): its mean says nothing about how decisive the beams were.
 *   · TEMPERED, twice. (1) All beams share ONE pose error: the posterior's depth spread is a common mode of intraclass
 *     correlation ρ = σ_c² / (σ_c² + σ_r²), so n_inf discriminating beams carry n_inf / (1 + (n_inf − 1) ρ)
 *     independent observations (Kish; the same design effect the leaf channel uses). (2) By p_view: a frame whose
 *     beams are not on this doorway says nothing about its leaf — a power likelihood, as the pose update itself is.
 *
 * FUSION (the ruling): the door's own P(open) π is the prior; the anchor's latest Λ, while fresh, is a likelihood:
 *     P(open | everything) = σ(logit π + Λ).
 * It is added, not substituted (never an override), and the anchor's frame is the LATEST one only, so a frame's beams
 * are never counted twice by the anchor channel. A truly shut door gives Λ ≪ 0, so the shut-door guarantees hold
 * (test_door_anchor_open). ⚠ The leaf channel (estimate_phi) reads the same bpearl sweep through its own selection; the
 * ruling accepts that overlap — the anchor's frame is posed on the OPENING the robot is servoing on, the leaf's on the
 * room-frame aperture, and try69 is exactly the case where those two disagree.
 *
 * ─── FIX 4: THE BORN-STALE WAIT POSE IS NEVER THE UNPUBLISHED POSTERIOR ──────────────────────────────────────────────
 * When the anchor is not seen, door_concept still writes a pose (born stale) for the controller's wait-turn to look
 * at. That used to be the live, UNPUBLISHED posterior — the one p_view had just refused to publish — which during a
 * wedge follows the odometry 81–89° off (try66 L2 replay), so the robot turned hard at the jamb toward nothing. It is
 * now the last SEEN (published) pose carried forward by odometry, the same seen_pose the predicted rule publishes;
 * before anything is seen in this passage, the pose the anchor was initialised with (from the room), carried by the
 * same odometry. wait_pose() is that choice, as a pure function, so a test can check the posterior is never chosen.
 *
 * Header-only; no DSR / Qt.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Dense>

#include "door_anchor_update.h"   // anchor_detail::to_anchor / Ray2, robot_to_anchor_axes (read only)

namespace rc::door
{

struct AnchorOpenEvidence
{
    double llr    = 0.0;   // Λ, tempered: log p(rays | open) − log p(rays | shut)
    double raw    = 0.0;   // Σ per-beam log ratio, untempered (diagnostic)
    double n_inf  = 0.0;   // beams that discriminate open from shut (soft count, diagnostic)
    double beta   = 1.0;   // the common-mode power applied (before p_view)
    int    n_rays = 0;
};

// The densities are phi_field_loglik's (door_belief.cpp log_density) with its FieldNuisance defaults: P(short of the
// model) 0.10, P(beyond it) 0.03, range noise 5 cm, an 8 m cap. Not re-tuned here: one beam model for the door.
inline AnchorOpenEvidence anchor_open_llr(const rc::interaction::RelPose& post, float width, float thickness,
                                          const Eigen::Vector3f& lidar_origin, const std::vector<Eigen::Vector3f>& rays,
                                          float p_view)
{
    AnchorOpenEvidence out;
    constexpr double w_u = 0.10, w_b = 0.03, w_s = 1.0 - w_u - w_b, sig_r = 0.05, cap = 8.0;
    const double pv = std::clamp(double(p_view), 0.0, 1.0);
    if (rays.empty() or width <= 0.f or pv <= 0.0)
        return out;
    const Eigen::Vector3d x = post.x.cast<double>();
    const Eigen::Matrix3d Pa = robot_to_anchor_axes(post.P, post.x.z()).cast<double>();   // (along, across, yaw)
    const double var_lat = std::max(0.0, Pa(0, 0)), var_dep = std::max(0.0, Pa(1, 1)), var_yaw = std::max(0.0, Pa(2, 2));
    const double t2 = double(thickness) * double(thickness) / 12.0;   // a leaf somewhere across the wall's depth
    const double half_w = 0.5 * double(width);
    const Eigen::Vector2d o = lidar_origin.head<2>().cast<double>();
    std::vector<double> lr;
    lr.reserve(rays.size());
    for (const auto& p : rays)
    {
        anchor_detail::Ray2 r{o, p.head<2>().cast<double>()};
        const double rho = (r.e - r.o).norm();
        if (rho < 1e-3) continue;
        const anchor_detail::UD a = anchor_detail::to_anchor(r, x);
        if (not (a.d_o < 0.0) or not (a.d > a.d_o)) continue;   // starts beyond the plane, or travels away from it
        const double t_ap = rho * (-a.d_o) / (a.d - a.d_o);    // range along the ray to the plane
        if (not (t_ap > 0.0 and t_ap < cap)) continue;
        const double r_m = std::min(rho, cap - 1e-3);
        // The posterior's spread at THIS crossing: lateral offset + the yaw lever (the depth to the origin), and the
        // depth offset + the yaw lever along the wall.
        const double s_u = std::sqrt(var_lat + a.d_o * a.d_o * var_yaw + sig_r * sig_r);
        const double s_d = std::sqrt(sig_r * sig_r + t2 + var_dep + a.u * a.u * var_yaw);
        const double m = 0.5 * std::erfc(-(half_w - std::abs(a.u)) / (std::sqrt(2.0) * s_u));   // P(crossing in the span)
        // SHUT (and the wall beside the opening): a surface at t_ap. OPEN: nothing until the next room.
        const double z = (r_m - t_ap) / s_d;
        double p_s = w_s * std::exp(-0.5 * z * z) / (s_d * std::sqrt(2.0 * M_PI));
        p_s += r_m < t_ap ? w_u / t_ap : w_b / std::max(1e-3, cap - t_ap);
        const double p_o = r_m >= t_ap ? (1.0 - w_u) / std::max(1e-3, cap - t_ap) : w_u / t_ap;
        const double l = std::log(m * p_o + (1.0 - m) * p_s) - std::log(p_s);
        if (std::isfinite(l)) lr.push_back(l);
    }
    out.n_rays = int(lr.size());
    if (lr.empty())
        return out;
    double n_inf = 0.0, raw = 0.0;
    for (const double l : lr) { raw += l; n_inf += 1.0 - std::exp(-std::abs(l)); }
    // The common mode: the posterior's depth spread is shared by every beam (one pose), against the beam's own noise.
    const double sc2 = var_dep;
    const double rho = sc2 / (sc2 + sig_r * sig_r);
    out.raw   = raw;
    out.n_inf = n_inf;
    out.beta  = 1.0 / (1.0 + std::max(0.0, n_inf - 1.0) * rho);
    out.llr   = pv * out.beta * raw;
    return out;
}

// P(open) fused: the door's own posterior π as the prior, the anchor's latest frame Λ as a likelihood. π at exactly 0
// or 1 is a certainty no evidence can move, and stays (never an override).
[[nodiscard]] inline float fuse_open(float pi, double llr)
{
    const double p = std::clamp(double(pi), 0.0, 1.0);
    if (p <= 0.0 or p >= 1.0 or not std::isfinite(llr)) return float(p);
    const double l = std::log(p) - std::log1p(-p) + llr;
    return float(l >= 0 ? 1.0 / (1.0 + std::exp(-l)) : std::exp(l) / (1.0 + std::exp(l)));
}

// FIX 4: the pose the born-stale (not-seen) write hands the controller's wait-turn. Never the unpublished posterior.
[[nodiscard]] inline const rc::interaction::RelPose& wait_pose(bool seen_once, const rc::interaction::RelPose& seen_pose,
                                                               const rc::interaction::RelPose& init_pose)
{
    return seen_once ? seen_pose : init_pose;
}

}  // namespace rc::door
