/*
 * door_anchor_open.h — Task 11 (try69 leg 2): what the passage anchor's own beams say about the LEAF ANGLE, and which
 * pose the controller's not-seen wait-turn may aim at.
 *
 * ─── FIX 2 (fix round 1): THE ANCHOR'S BEAMS ARE EVIDENCE ABOUT φ, NOT ABOUT "SHUT OR NOT" ─────────────────────────
 * The user's idea: "the door model can predict that the lidar will pass through, check it and use it as local evidence".
 * While a passage anchor exists, every measured cycle hands update_door_anchor a sweep that crosses the opening, and
 * the opening's posterior says where each beam meets its plane. The first version scored a leaf AT the plane (φ = 0)
 * against NOTHING beyond it and fused that ratio into P(passable). But P(passable) is P(clear span ≥ body), and a leaf
 * swung away by 30° returns beams just beyond the plane, which "nothing beyond" explains far better than "a leaf at the
 * plane": the review measured Λ +8.4 at 30° (clear span 0.13 m) and fused P 0.989 — an AJAR, impassable door read open.
 * The cure is to ask the beams the question P(passable) is about. anchor_phi_loglik scores them under the SWUNG LEAF
 * at every angle of door_concept's own φ grid — for the believed hinge and the other jamb — and DoorFitter::estimate_phi
 * ADDS that curve to its own φ likelihood (DoorInstance::anchor_phi_*), consumed once. The passability marginalisation
 * is then passable_prob's / passable_logit's, over the joint posterior; nothing is fused into P(passable) directly.
 *   · Per beam the density is phi_field_loglik's (DoorBelief, door_belief.cpp log_density): a surface hit (range noise
 *     ⊕ the leaf's thickness ⊕ the posterior's depth spread), P(short of it) 0.10, P(beyond it) 0.03, an 8 m cap; with
 *     no surface on its path the beam flies on (P(short) still 0.10 before the plane).
 *   · The first surface: the leaf segment at φ (hinge (hinge_u, 0), direction (dir_u cos φ, swing_y sin φ), length
 *     len — LeafPrior's convention, anchor coordinates), or the wall where the beam crosses the plane outside the span.
 *     The span membership m = Φ((w/2 − |u|)/σ_u) under the posterior's lateral + yaw spread mixes the two
 *     continuously: p = m·p(leaf only) + (1 − m)·p(wall or leaf, whichever first).
 *   · TEMPERED like the leaf channel: n_inf beams that discriminate between angles (Σ 1 − e^{−spread}) sharing ONE pose
 *     error of intraclass correlation ρ = σ_dep² / (σ_dep² + σ_r²) carry n_inf / (1 + (n_inf − 1) ρ) observations
 *     (Kish); and by p_view, the frame's P(this doorway in view) — a power likelihood, as the pose update itself is.
 *
 * ─── FIX 4: THE BORN-STALE WAIT POSE IS NEVER THE UNPUBLISHED POSTERIOR ──────────────────────────────────────────────
 * When the anchor is not seen, door_concept still writes a pose (born stale) for the controller's wait-turn to look
 * at. That used to be the live, UNPUBLISHED posterior — the one p_view had just refused to publish — which during a
 * wedge follows the odometry 81–89° off (try66 L2 replay), so the robot turned hard at the jamb toward nothing. It is
 * now the last SEEN (published) pose carried forward by odometry, the same seen_pose the predicted rule publishes;
 * before anything is seen in this passage, the pose the anchor was initialised with (from the room), carried by the
 * same odometry. wait_pose() is that choice and carry_wait_poses() the exclusive carry, as pure functions.
 *
 * Header-only; no DSR / Qt.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Dense>

#include "door_anchor_update.h"   // LeafPrior, robot_to_anchor_axes, interaction::predict (read only)

namespace rc::door
{

struct AnchorPhiCurve
{
    std::vector<float> ll_this, ll_other;   // tempered log-likelihood on the φ grid, the SWING marginalised; joint max 0 (empty = nothing)
    double n_inf = 0.0, beta = 1.0;         // discriminating beams; the common-mode power (before p_view)
    int    n_rays = 0;
    // Task 12 fix D: this frame's evidence about the SWING SIDE, log p(rays | believed swing) − log p(rays | flipped), each
    // marginalised over both hinges (½ each) and over φ under the caller's prior on it (LeafPrior phi / phi_sigma; uniform
    // when unknown) — on the same tempered scale as the curves. 0 when nothing discriminates (a shut leaf is the same leaf
    // whichever way it would swing). The door accumulates it as it does the hinge's (DoorInstance::swing_llr).
    double swing_llr = 0.0;
};

// `leaf` gives the believed hinge/swing in ANCHOR coordinates (the anchor caller's LeafPrior); `other` is the same
// leaf on the other jamb (hinge_u, dir_u negated). Grid: phi_min + (phi_max − phi_min)·k/(nstep − 1), door_concept's.
inline AnchorPhiCurve anchor_phi_loglik(const rc::interaction::RelPose& post, float width, float thickness,
                                        const LeafPrior& leaf, const Eigen::Vector3f& lidar_origin,
                                        const std::vector<Eigen::Vector3f>& rays, float p_view,
                                        float phi_min, float phi_max, int nstep)
{
    AnchorPhiCurve out;
    constexpr double w_u = 0.10, w_b = 0.03, w_s = 1.0 - w_u - w_b, sig_r = 0.05, cap = 8.0;
    const double pv = std::clamp(double(p_view), 0.0, 1.0);
    if (rays.empty() or width <= 0.f or pv <= 0.0 or nstep < 2 or not leaf.known)
        return out;
    const double c = std::cos(double(post.x.z())), s = std::sin(double(post.x.z()));
    const Eigen::Vector2d t0 = post.x.head<2>().cast<double>();
    auto inv = [&](const Eigen::Vector2d& p) { const Eigen::Vector2d q = p - t0; return Eigen::Vector2d(c * q.x() + s * q.y(), -s * q.x() + c * q.y()); };
    const Eigen::Matrix3d Pa = robot_to_anchor_axes(post.P, post.x.z()).cast<double>();   // (along, across, yaw)
    const double var_lat = std::max(0.0, Pa(0, 0)), var_dep = std::max(0.0, Pa(1, 1)), var_yaw = std::max(0.0, Pa(2, 2));
    const double t2 = double(thickness) * double(thickness) / 12.0;
    const double half_w = 0.5 * double(width);
    const double len = leaf.len > 0.f ? double(leaf.len) : double(width);
    const Eigen::Vector2d oa = inv(lidar_origin.head<2>().cast<double>());

    // The leaf direction per (hypothesis, angle). ★Task 12 fix D: FOUR hypotheses, hinge × swing, exactly as the pose update
    // marginalises them (update_door_anchor's hinge × swing blocks at p_swing ½). With the swing a config pin (Openable.SwingDir,
    // or a side inherited from the other room) and only the believed swing scored, a door AJAR 20–60° and swung AWAY from a
    // believed TOWARD leaf was explained by the believed side's leaf near its own far end (φ locked at 73.7°) and read
    // P(passable) ≈ 0.76 (the Task 11 re-review probe): an impassable door looked passable. Index hs = 2·hinge + swing.
    const int H = 4, K = nstep;
    std::vector<Eigen::Vector2d> hinge(H), dir(H * K);
    for (int hs = 0; hs < H; ++hs)
    {
        const double hsg = hs / 2 == 0 ? 1.0 : -1.0, ssg = hs % 2 == 0 ? 1.0 : -1.0;
        const double hu = hsg * double(leaf.hinge_u), du = hsg * double(leaf.dir_u), sy = ssg * double(leaf.swing_y);
        hinge[hs] = Eigen::Vector2d(hu, 0.0);
        for (int k = 0; k < K; ++k)
        {
            const double phi = double(phi_min) + (double(phi_max) - double(phi_min)) * k / (K - 1);
            dir[hs * K + k] = Eigen::Vector2d(du * std::cos(phi), sy * std::sin(phi));
        }
    }
    const auto density = [&](double t_surf, double t_ref, double r, double s_d)
    {
        double p;
        if (t_surf > 0.0 and t_surf < cap)
        {
            const double z = (r - t_surf) / s_d;
            p = w_s * std::exp(-0.5 * z * z) / (s_d * std::sqrt(2.0 * M_PI));
            p += r < t_surf ? w_u / t_surf : w_b / std::max(1e-3, cap - t_surf);
        }
        else if (t_ref > 0.0 and t_ref < cap)
            p = r >= t_ref ? (1.0 - w_u) / std::max(1e-3, cap - t_ref) : w_u / t_ref;
        else
            p = 1.0 / cap;
        return std::log(std::max(1e-300, p));
    };

    std::vector<double> ll(std::size_t(H * K), 0.0);
    double n_inf = 0.0;
    std::vector<double> lr(std::size_t(H * K));
    for (const auto& p3 : rays)
    {
        const Eigen::Vector2d ea = inv(p3.head<2>().cast<double>());
        const Eigen::Vector2d v = ea - oa;
        const double rho = v.norm();
        if (rho < 1e-3) continue;
        const Eigen::Vector2d u = v / rho;
        const double r = std::min(rho, cap - 1e-3);
        double t_plane = -1.0, xc = 1e9;
        if (std::abs(u.y()) > 1e-9)
            if (const double t = -oa.y() / u.y(); t > 0.0) { t_plane = t; xc = oa.x() + t * u.x(); }
        const double s_u = std::sqrt(var_lat + oa.y() * oa.y() * var_yaw + sig_r * sig_r);
        const double s_d = std::sqrt(sig_r * sig_r + t2 + var_dep + (t_plane > 0.0 ? xc * xc * var_yaw : 0.0));
        const double m = t_plane > 0.0 ? 0.5 * std::erfc(-(half_w - std::abs(xc)) / (std::sqrt(2.0) * s_u)) : 0.0;
        const double t_wall = t_plane;   // where a beam outside the span meets the wall
        double vmin = 1e300, vmax = -1e300;
        for (int h = 0; h < H; ++h)
            for (int k = 0; k < K; ++k)
            {
                // ray oa + t u vs segment hinge + q dir, q ∈ [0, len]
                const Eigen::Vector2d& d = dir[std::size_t(h * K + k)];
                const double den = u.x() * (-d.y()) - u.y() * (-d.x());
                double t_leaf = -1.0;
                if (std::abs(den) > 1e-12)
                {
                    const Eigen::Vector2d w = hinge[std::size_t(h)] - oa;
                    const double t = (w.x() * (-d.y()) - w.y() * (-d.x())) / den;
                    const double q = (u.x() * w.y() - u.y() * w.x()) / den;
                    if (t > 0.0 and q >= 0.0 and q <= len) t_leaf = t;
                }
                const double l_in  = density(t_leaf, t_plane, r, s_d);
                const double t_out = (t_leaf > 0.0 and (t_wall <= 0.0 or t_leaf < t_wall)) ? t_leaf : t_wall;
                const double l_out = density(t_out, t_plane, r, s_d);
                const double mx = std::max(l_in, l_out);
                const double l = mx + std::log(m * std::exp(l_in - mx) + (1.0 - m) * std::exp(l_out - mx));
                lr[std::size_t(h * K + k)] = l;
                if (h == 0) { vmin = std::min(vmin, l); vmax = std::max(vmax, l); }   // the believed leaf's spread (as the leaf channel)
            }
        if (not std::isfinite(vmax - vmin)) continue;
        for (std::size_t j = 0; j < lr.size(); ++j) ll[j] += lr[j];
        n_inf += 1.0 - std::exp(-(vmax - vmin));
        ++out.n_rays;
    }
    if (out.n_rays == 0)
        return out;
    const double rho_c = var_dep / (var_dep + sig_r * sig_r);
    out.n_inf = n_inf;
    out.beta  = 1.0 / (1.0 + std::max(0.0, n_inf - 1.0) * rho_c);
    const double g = pv * out.beta;
    double mx = -std::numeric_limits<double>::infinity();
    for (double& x : ll) { x *= g; mx = std::max(mx, x); }
    // Per hinge, the swing marginalised at ½ each: log ½(e^a + e^b), shifted so the joint max stays 0.
    const auto lae = [](double a, double b) { const double m = std::max(a, b); return m + std::log(std::exp(a - m) + std::exp(b - m)); };
    std::vector<double> lh(std::size_t(2 * K));
    double mh = -std::numeric_limits<double>::infinity();
    for (int h = 0; h < 2; ++h)
        for (int k = 0; k < K; ++k)
        {
            const double v = lae(ll[std::size_t((2 * h) * K + k)], ll[std::size_t((2 * h + 1) * K + k)]) - std::log(2.0);
            lh[std::size_t(h * K + k)] = v;
            mh = std::max(mh, v);
        }
    out.ll_this.resize(std::size_t(K));
    out.ll_other.resize(std::size_t(K));
    for (int k = 0; k < K; ++k)
    {
        out.ll_this[std::size_t(k)]  = float(lh[std::size_t(k)] - mh);
        out.ll_other[std::size_t(k)] = float(lh[std::size_t(K + k)] - mh);
    }
    // The swing's evidence: each swing marginalised over both hinges (½) and φ under the caller's prior.
    double e_b = -std::numeric_limits<double>::infinity(), e_o = e_b;
    const double sp = std::max(1e-4, double(leaf.phi_sigma));
    for (int k = 0; k < K; ++k)
    {
        const double phi = double(phi_min) + (double(phi_max) - double(phi_min)) * k / (K - 1);
        const double d = (phi - double(leaf.phi)) / sp;
        const double lw = leaf.phi_known ? -0.5 * d * d : 0.0;
        for (int h = 0; h < 2; ++h)
        {
            e_b = lae(e_b, ll[std::size_t((2 * h) * K + k)] + lw);
            e_o = lae(e_o, ll[std::size_t((2 * h + 1) * K + k)] + lw);
        }
    }
    out.swing_llr = std::isfinite(e_b - e_o) ? e_b - e_o : 0.0;
    return out;
}

// Task 12 fix round 1 (I-2): a retry's room fusion, weighted by w = P(the robot is OUT of the doorway band). The belief is
// broadened by w·σ_model² (in robot axes: model_cov) and fused with the room initialisation as K = w·P_b (w·P_b + R)⁻¹ — the
// room's information scaled by w, continuous; inside the band (w → 0) the belief stands untouched. Returns the new mean and P.
[[nodiscard]] inline rc::interaction::RelPose retry_fusion(const rc::interaction::RelPose& belief, const rc::interaction::RelPose& room,
                                                           const Eigen::Matrix3f& model_cov, float w_out, Eigen::Vector3f* innov_out = nullptr)
{
    const float w = std::clamp(w_out, 0.f, 1.f);
    const Eigen::Matrix3f Pb = belief.P + w * model_cov;
    const Eigen::Matrix3f K = w * Pb * (w * Pb + room.P).inverse();
    Eigen::Vector3f innov = room.x - belief.x;
    innov.z() = rc::interaction::wrap_pi(innov.z());
    if (innov_out != nullptr) *innov_out = innov;
    rc::interaction::RelPose out = belief;
    out.x += K * innov;
    out.x.z() = rc::interaction::wrap_pi(out.x.z());
    const Eigen::Matrix3f Pn = (Eigen::Matrix3f::Identity() - K) * Pb;
    out.P = 0.5f * (Pn + Pn.transpose());
    return out;
}

// Task 12 fix round 1 (I-3): a frame is evidence about the LEAF (its φ curve, the hinge and swing log-odds) only when its pose is
// the one being PUBLISHED as seen — p_view at or above the publication level. An aliased pose scores the beams against a wrong
// opening; the hinge and swing log-odds accumulate with no forgetting, so one aliased stretch could flip and latch them.
[[nodiscard]] inline bool anchor_frame_is_leaf_evidence(float p_view, float publish_level) { return p_view >= publish_level; }

// FIX 4: the pose the born-stale (not-seen) write hands the controller's wait-turn. Never the unpublished posterior.
[[nodiscard]] inline const rc::interaction::RelPose& wait_pose(bool seen_once, const rc::interaction::RelPose& seen_pose,
                                                               const rc::interaction::RelPose& init_pose)
{
    return seen_once ? seen_pose : init_pose;
}
// …and the odometry carry behind it, EXCLUSIVE: before the first sighting the room initialisation is carried (seen_pose
// means nothing yet); from it on, seen_pose is carried and the init pose is frozen (it is no longer the wait pose).
inline void carry_wait_poses(bool seen_once, rc::interaction::RelPose& seen_pose, rc::interaction::RelPose& init_pose,
                             const rc::rt::Twist& twist, const Eigen::Vector3f& var, float dt)
{
    if (seen_once) seen_pose = rc::interaction::predict(seen_pose, twist, var, dt);
    else           init_pose = rc::interaction::predict(init_pose, twist, var, dt);
}

}  // namespace rc::door
