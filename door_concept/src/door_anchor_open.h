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

// The weight retry_fusion takes: P(out of the band) = 1 − P_in(belief)·P_in(room) — refuse the room only where both put the
// robot in the doorway (regression fix after try73: a wrong belief alone must not veto the room that disagrees with it).
[[nodiscard]] inline float retry_weight_out_of_band(const rc::interaction::RelPose& belief, const rc::interaction::RelPose& room,
                                                    float half_depth)
{
    return 1.f - anchor_doorway_band(belief, half_depth) * anchor_doorway_band(room, half_depth);
}

// ─── Task 17 (try78 leg 2): WHEN THE BELIEF AND THE ROOM DISAGREE FAR BEYOND THEIR COVARIANCE, THE BELIEF IS THE SUSPECT ─
// try78: the anchor was created ~0.5 m off and converged to σ_lat 0.02 m there; every retry's room fusion was refused because
// the wrong belief put the robot inside the doorway band (weight 0.01), and the room (0.77 m and 0.97 m away) never pulled it
// back. The disagreement ν = room − belief (yaw wrapped) is scored under two hypotheses:
//   belief RIGHT: ν ~ N(0, S),                    S = P_belief + P_room (the room init's P0: belief + localisation + σ_model);
//   belief LOST : ν ~ Student-t_ν(0, S + Σ_L),    heavy-tailed AND broad — a lost belief has no scale of its own; Σ_L is the
//                 scale of the aliases a lost anchor is seen in: one opening width along the wall and across it (try75/78:
//                 0.4–0.75 m), a quarter turn in yaw (try70: 74–86°). (Without Σ_L a same-scale t is MORE likely than the
//                 Gaussian at ν = 0 in 3-D, and an agreeing belief would be re-initialised.)
// P(lost | ν) is their posterior ratio under the prior mass ⚑ kAnchorLostPrior (flagged, LeafPrior::p_not_as_believed's
// convention) and ⚑ ν = 1 (Cauchy, the heaviest standard tail: "no scale", not a tuned dof).
inline constexpr double kAnchorLostPrior = 0.2;
inline constexpr double kAnchorLostDof   = 1.0;
// room_llr (Task D-B): the room channel alone, log p(ν | lost) − log p(ν | right) — the per-cycle filter weights it by time.
struct LostEvidence { double d2 = 0.0; float p_lost = 0.f; double view_llr = 0.0; double room_llr = 0.0; };

// ─── Task 17 add-on: THE BELIEF'S OWN VISIBILITY AS EVIDENCE THAT IT IS LOST ──────────────────────────────────────────────
// try78: the wrong anchor stood at the doorway for 60 s with p_view 0.13–0.18 — what its own model says a WRONG pose looks
// like. Every measured frame (the belief predicts the doorway's neighbourhood in the sweep) is scored under the two hypotheses:
//   right: p_view ~ ½·U(0,1) + ½·Beta(2,1)   (density 1 + 2p … ½ uniform: a right pose can be refused by clutter, a leaf)
//   lost : p_view ~ ½·U(0,1) + ½·Beta(1,2)   (density 1 + 2(1 − p))
// so the frame's log-ratio log[(1 + 2(1 − p)) / (1 + 2p)] is bounded (±log 3): no frame is decisive alone. Consecutive frames
// share the view, so the evidence is TIME-weighted: Δllr = log-ratio · dt / τ, τ = ⚑ kAnchorPredictMs — the flagged horizon a
// view is trusted over, here the time over which views decorrelate. Accumulated for the passage (a seen frame subtracts),
// reset with the belief (creation, re-init, retry fusion). Continuous; no counter, no limit.
[[nodiscard]] inline double view_lost_llr_step(float p_view, double dt_s, double tau_s)
{
    if (not std::isfinite(p_view) or not (dt_s > 0.0) or not (tau_s > 0.0)) return 0.0;
    const double p = std::clamp(double(p_view), 0.0, 1.0);
    return std::log((1.0 + 2.0 * (1.0 - p)) / (1.0 + 2.0 * p)) * dt_s / tau_s;
}

[[nodiscard]] inline LostEvidence belief_lost(const rc::interaction::RelPose& belief, const rc::interaction::RelPose& room,
                                              float width, double view_llr = 0.0)
{
    LostEvidence o;
    // Task D-B: the right/lost likelihood ratio is alias_log_ratio (door_anchor_update.h) — the same model the update uses to
    // test its own fit against its prior.
    o.room_llr = alias_log_ratio((room.x - belief.x).cast<double>(), (belief.P + room.P).cast<double>(), double(width), &o.d2);
    // The room disagreement and the belief's own visibility are independent evidence: their log-ratios add.
    o.view_llr = std::isfinite(view_llr) ? view_llr : 0.0;
    const double a = std::log(kAnchorLostPrior) + o.room_llr + o.view_llr, b = std::log(1.0 - kAnchorLostPrior);
    o.p_lost = float(1.0 / (1.0 + std::exp(b - a)));
    return o;
}
// The retry's posterior as the two-hypothesis mixture: with P(lost) the belief carries nothing and the room initialisation
// IS the answer; otherwise the room fusion at the doorway-band weight (retry_fusion). Moment-matched — continuous in P(lost),
// no gate: w_reinit = max(band weight, P(lost)) is the mass on the room's side, as ruled.
[[nodiscard]] inline rc::interaction::RelPose retry_reinit(const rc::interaction::RelPose& belief, const rc::interaction::RelPose& room,
                                                           const Eigen::Matrix3f& model_cov, float w_band, float p_lost)
{
    const auto fused = retry_fusion(belief, room, model_cov, w_band);
    const float p = std::clamp(p_lost, 0.f, 1.f);
    rc::interaction::RelPose out = fused;
    Eigen::Vector3f d = room.x - fused.x;
    d.z() = rc::interaction::wrap_pi(d.z());
    out.x = fused.x + p * d;
    out.x.z() = rc::interaction::wrap_pi(out.x.z());
    const Eigen::Vector3f df = fused.x - out.x, dr = (fused.x + d) - out.x;
    out.P = (1.f - p) * (fused.P + df * df.transpose()) + p * (room.P + dr * dr.transpose());
    out.P = 0.5f * (out.P + out.P.transpose());
    return out;
}

// ─── Task D-B (rtD1h leg 5): THE LOST CHECK ON EVERY CYCLE OF A CLAIMED CROSS — a two-mode filter ───────────────────────
// rtD1h leg 5: hung correctly from the room, the anchor's FIRST update put it on an alias 1.4 m / 84° off and the beams held
// it; belief_lost (Mahalanobis² 88 ⇒ P(lost) 1.00) ran only at the controller's retry, 46 s later. (The jump itself is the
// update's — the update now tests its own fit against its prior, alias_log_ratio in door_anchor_update.h; this is the
// check that an alias that gets in anyway cannot survive.)
// WHAT THE BEAMS TRACK is a latent with two modes — `right` (the opening) or `lost` (an alias: the beams then say nothing
// about the opening, whose pose is the room initialisation's, σ_model-wide):
//   TRANSITIONS  happen at UPDATES, not with time: a tracker can be captured by an alias only where its prior reaches one.
//                Per update, h = P(the prior puts the opening outside its own basin) = 1 − Π_axes P(|z| < ½ a_i / σ_i), σ the
//                prior's marginal σ in anchor axes and a the ALIAS SCALE belief_lost already uses (Σ_L: one opening width
//                along and across the wall, a quarter turn in yaw); ⚑ the basin = half that scale (a definition). The chain
//                is symmetric (a lost tracker re-acquires the opening on the same terms). rtD1h's hang prior (0.31, 0.20,
//                0.10) gives h ≈ 0.18 (test 16d); a converged tracker (σ 2 cm, 1°) gives h ≈ 0 — so a seen, correct anchor's P(lost)
//                cannot creep up with time and a parked one changes nothing (lost_hazard).
//   EVIDENCE     (i) the belief's own visibility, per MEASURED frame, weighted by how NEW the view is: (1 − ρ) with
//                ρ = e^(−Δ/ℓ), Δ how far the doorway moved in the robot frame since the last measured frame and ℓ half the
//                opening — the view-decorrelation transport_bias uses. A parked robot re-reading one view adds nothing.
//                (ii) the room disagreement (belief_lost's room channel) as a SNAPSHOT: the current innovation scored ONCE on
//                top of the filtered mode, never fed back into the recursion — it is one comparison of one belief with one
//                room initialisation, however many cycles look at it.
//   MIXING       (IMM) before each update the tracker's prior is mixed with the room initialisation at w = P(lost then |
//                right now) = p·h / (p·h + (1 − p)(1 − h)), p the last posterior: 0 for a converged tracker, → 1 as P(lost)
//                → 1 while the prior can still reach the opening.
//   OUTPUT       published = the two modes combined, moment-matched — (1 − p)·tracker + p·room init (retry_reinit with no band
//                fusion). Its P(door in view) is combined the same way (anchor_view_mixture): a pose that is mostly the
//                room's is not "seen" because the TRACKER sees something.
// Every re-init resets the mode (q = 0: the tracker IS the room initialisation then); the retry's one-shot fusion is unchanged.
// ★Fix round 2: the state is DOUBLE LOG-ODDS throughout and never passes through a probability — a float q rounded to
//  exactly 1 after ~17 nats (a few seconds of unseeing views) and logit(1) = +∞ latched P(lost) = 1 for ever; the mirror
//  underflow at ~−103 silenced the check. −∞ is reachable only as the birth state (q = 0: the tracker IS the room
//  initialisation) and is left by the first update's hazard.
struct AnchorLostMode
{
    double q_logodds = -std::numeric_limits<double>::infinity();   // the filtered mode WITHOUT the room snapshot (log-odds)
    double p_logodds = -std::numeric_limits<double>::infinity();   // the posterior: filtered mode + this cycle's room snapshot
    float  p = 0.f;                                                 // ...as a probability, for publication / the mixture weights
    Eigen::Vector3f last_view = Eigen::Vector3f::Constant(std::numeric_limits<float>::quiet_NaN());   // pose at the last measured frame
    [[nodiscard]] double q() const;
};
inline double logit_of(double p) { return p <= 0.0 ? -std::numeric_limits<double>::infinity() : p >= 1.0 ? std::numeric_limits<double>::infinity() : std::log(p / (1.0 - p)); }
inline double sigmoid_of(double l) { return std::isfinite(l) ? 1.0 / (1.0 + std::exp(-l)) : (l > 0 ? 1.0 : 0.0); }
inline double AnchorLostMode::q() const { return sigmoid_of(q_logodds); }
// One symmetric transition of a two-state chain with switching probability h, IN LOG-ODDS:
//   odds⁻ = (q(1 − h) + (1 − q)h) / ((1 − q)(1 − h) + q h), evaluated on the side of 0 the state is on so that ±∞ stay exact.
inline double lost_transition_logodds(double L, double h)
{
    if (not (h > 0.0)) return L;
    if (h >= 1.0) return -L;
    const double lh = std::log(h), l1h = std::log1p(-h);
    const auto lae = [](double a, double b) { if (a < b) std::swap(a, b); return a == -std::numeric_limits<double>::infinity() ? a : a + std::log1p(std::exp(b - a)); };
    if (L <= 0.0) return lae(L + l1h, lh) - lae(l1h, L + lh);
    return lae(l1h, -L + lh) - lae(-L + l1h, lh);
}

// The per-update capture hazard h of a prior (see TRANSITIONS): alias_hazard (door_anchor_update.h), the same one the update
// weighs its own fit with.
[[nodiscard]] inline double lost_hazard(const rc::interaction::RelPose& prior, float width)
{
    return alias_hazard(prior.P, prior.x.z(), width);
}

// Step 1, before the update: the transition at this update and the IMM mixing of the tracker's prior with the room
// initialisation (when there is one this cycle). The bias factor's share of P scales with the tracker's weight
// (P − L Lᵀ ⪰ 0 is kept: P_new ⪰ (1 − w)·P_old). Returns w.
inline float lost_mode_predict(AnchorLostMode& m, rc::interaction::RelPose& tracker, AnchorBias* bias,
                               const rc::interaction::RelPose* room, float width)
{
    const double h = lost_hazard(tracker, width);
    m.q_logodds = lost_transition_logodds(m.q_logodds, h);   // the filtered mode, one transition
    if (room == nullptr) return 0.f;
    // w = p·h / (p·h + (1 − p)(1 − h)) = σ(logit p + logit h), in log-odds
    const double w = h > 0.0 ? sigmoid_of(m.p_logodds + std::log(h) - std::log1p(-h)) : 0.0;
    tracker = retry_reinit(tracker, *room, Eigen::Matrix3f::Zero(), 0.f, float(w));
    if (bias != nullptr) bias->L *= float(std::sqrt(1.0 - w));
    return float(w);
}

// Step 2, after a MEASURED update: the frame's visibility evidence on the filtered mode, weighted by the view's novelty.
inline double lost_mode_view(AnchorLostMode& m, const rc::interaction::RelPose& tracker, float p_view, float width)
{
    double novelty = 1.0;
    if (m.last_view.allFinite())
    {
        const float ell = 0.5f * std::max(width, 1e-3f);
        // Δ: how far the doorway moved in the robot frame (a turn moves it by its lever) — transport_bias's measure.
        novelty = -std::expm1(-double((tracker.x.head<2>() - m.last_view.head<2>()).norm()) / double(ell));
    }
    m.last_view = tracker.x;
    if (not std::isfinite(p_view)) return 0.0;
    const double pv = std::clamp(double(p_view), 0.0, 1.0);
    const double step = novelty * std::log((1.0 + 2.0 * (1.0 - pv)) / (1.0 + 2.0 * pv));
    if (std::isfinite(m.q_logodds)) m.q_logodds += step;   // q = 0 or 1 exactly: evidence cannot move a certainty
    return step;
}

// Step 3, every cycle with a room initialisation: the posterior = the filtered mode + the room SNAPSHOT (once, not recursed).
inline LostEvidence lost_mode_posterior(AnchorLostMode& m, const rc::interaction::RelPose& tracker, const rc::interaction::RelPose& room,
                                        float width)
{
    LostEvidence e = belief_lost(tracker, room, width);
    m.p_logodds = m.q_logodds + (std::isfinite(e.room_llr) ? e.room_llr : 0.0);
    m.p = float(sigmoid_of(m.p_logodds));
    e.p_lost = m.p;
    return e;
}

// The OUTPUT: both modes combined, moment-matched — what is published as the anchor.
[[nodiscard]] inline rc::interaction::RelPose lost_mode_output(const rc::interaction::RelPose& tracker, const rc::interaction::RelPose& room,
                                                               float p_lost)
{
    return retry_reinit(tracker, room, Eigen::Matrix3f::Zero(), 0.f, p_lost);
}
// ...and its P(door in view): the mixture's — the tracker's view where the tracker is right, the room init's where it is lost.
[[nodiscard]] inline float anchor_view_mixture(float p_view_tracker, float p_view_room, float p_lost)
{
    const float p = std::clamp(p_lost, 0.f, 1.f);
    return (1.f - p) * p_view_tracker + p * p_view_room;
}

// Task 12 fix round 1 (I-3): a frame is evidence about the LEAF (its φ curve, the hinge and swing log-odds) only when its pose is
// the one being PUBLISHED as seen — p_view at or above the publication level. An aliased pose scores the beams against a wrong
// opening; the hinge and swing log-odds accumulate with no forgetting, so one aliased stretch could flip and latch them.
[[nodiscard]] inline bool anchor_frame_is_leaf_evidence(float p_view, float publish_level) { return p_view >= publish_level; }

// ─── Task 21 (try89 leg 3): NO ANCHOR SURVIVES ITS PASSAGE, AND EVERY ANCHOR IS BORN ON THE ROBOT'S NEAR SIDE ──────────────
// try89: after leg 2's cross went through, the anchor was KEPT past the room switch (crossing_through, Fix E) and nothing ever
// ended it — REMOVE (b) needs the consumer's `executing` edge to something else, and the controller wrote none for the
// close-less "passage DONE" and its later claims. Leg 3's approach saw the door again, the leftover anchor RESUMED publishing
// (crossing_through counts as a claimed cross), the controller drove on it before the cross was claimed, and the cross claim
// then re-initialised it KEEPING leg 2's side (s +1.78, 176°: the normal flipped), so "through" fired 2 m short of the door.
//
// (1) After the room switch (crossing_through) the passage is over as soon as EITHER this door is asked anything again (any
//     of its approach/open/cross/close claimed: the consumer moved on — a legacy consumer writes no edge, but its claim is
//     visible on our own affordance) OR the predicted publication has run out (the same kAnchorPredictMs bound that stops
//     publishing it: the anchor is no longer driven on). No new level: both are existing events.
[[nodiscard]] inline bool passage_over_after_through(bool crossing_through, bool door_step_claimed, bool predict_window_spent)
{
    return crossing_through and (door_step_claimed or predict_window_spent);
}
// (2) Which side of the opening the robot is on, along the aperture's across_u (+1 / −1): the side the CROSS LEAVES (the
//     latched interior side, door_concept's cross_side_latched — it holds mid-doorway, where the robot's own position is
//     ambiguous), else the robot's position (unlatched: 0). The anchor's +y then points from that side through the opening,
//     so the robot is at s < 0 — at creation, at every re-init and at every retry; never inherited from an older belief.
[[nodiscard]] inline float anchor_near_side(float cross_side_latched, float robot_along_n)
{
    if (cross_side_latched != 0.0f and std::isfinite(cross_side_latched)) return cross_side_latched < 0.0f ? -1.0f : +1.0f;
    return robot_along_n > 0.0f ? +1.0f : -1.0f;
}
// The opening frame's axes in the room: x along the wall, y the normal pointing AWAY from the near side (right-handed), and
// the sign of x against wall_u (the anchor's axis_sign).
struct OpeningAxes { Eigen::Vector2f ux, uy; float sign = 1.0f; };
[[nodiscard]] inline OpeningAxes anchor_opening_axes(const Eigen::Vector2f& wall_u, const Eigen::Vector2f& across_u, float near_side)
{
    OpeningAxes a{wall_u.normalized(), across_u.normalized(), 1.0f};
    if (near_side > 0.0f) { a.ux = -a.ux; a.uy = -a.uy; a.sign = -1.0f; }   // rotate by π: the near side on −y
    return a;
}

// ─── Task 19 (user decision): WHEN THE PASSAGE ANCHOR IS CREATED ─────────────────────────────────────────────────────────
// On the CROSS claim only. The cross is offered only while door_concept believes the door passable (open), so the anchor is
// born on an open door and is never updated against a shut one or a swinging leaf; approach and open run on the room-frame
// standpoint. An approach claim creates nothing. (`approach_claimed` is taken only to say so explicitly.)
[[nodiscard]] inline bool anchor_creation_claim(bool cross_claimed, bool /*approach_claimed*/) { return cross_claimed; }

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
