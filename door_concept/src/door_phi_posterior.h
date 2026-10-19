/*
 * door_phi_posterior.h — the leaf-angle (phi) channel as PURE functions: which rays interrogate a doorway,
 * which nuisances the field likelihood marginalises, and how one frame's likelihood updates the carried
 * belief. DoorFitter::estimate_phi is a thin wrapper around these (plus the command prior and the clock),
 * so tests/test_door_phi_stability.cpp exercises the SAME code the agent runs.
 *
 * ─── WHAT THIS FIXED (2026-10-13, live try36-try40) ──────────────────────────────────────────────────
 * After crossing DOOR_1, the door carried into room_2 (leaf ~95 deg, opening OUT of room_2, i.e. beside
 * the doorway on the far side of the wall) had door_open_prob flip 1.00 <-> 0.00 cycle to cycle while
 * phi_est sat at 80-100 deg — 223 open/cross alternations in try38. The logged curve was a DELTA every
 * frame (dbg_phi_cv = 4.90 = sqrt(24): all the weight on one of 25 angles) and phi_sigma alternated
 * 0.00 / 2.09 rad. Three faults compounded:
 *   1. the aperture the likelihood scored was 0.23-0.40 m along the wall from the real doorway (the
 *      actuator's "nearest by place" on every room_2 request), and its slide nuisance spanned only
 *      +-6 cm — so NO hypothesis explained the frame and the least-bad angle depended on the viewpoint;
 *   2. the per-ray log densities were summed as if independent, turning that misfit into hundreds of
 *      nats — a delta the persistence prior could not resist;
 *   3. the posterior was formed in LINEAR float: a delta landing where the prior had underflowed made
 *      every weight exactly 0, the code read that as "no curve" and reset phi_sigma to the full range —
 *      so the NEXT frame had no prior at all and re-decided the angle from scratch;
 *   4. the carried prior was centred on a RATE-LIMITED phi_est, not on the posterior it summarised: the
 *      clamp moved the estimate at most swing_rate*dt per cycle while the posterior's width shrank to a
 *      few degrees, so the next prior sat tens of degrees (many sigma) from where the evidence had put
 *      the leaf, and the two fought — harness (c): a first look at an open door read P = 1.00, 0.10, 1.00.
 * The cures are model terms, not gates: the slide is marginalised over +-w/2 (DoorBelief::
 * phi_field_loglik), the joint likelihood is tempered by the frame's effective number of independent
 * discriminating rays (ibid.), and the posterior is formed in the LOG domain so the prior can never be
 * lost to underflow (phi_posterior_step below). The belief that persists is the posterior's (mean, sigma);
 * the hinge's kinematics live where they belong, in the predict step's width (phi_prior_sigma), not as a
 * clamp on the mean.
 *
 * Header-only, no DSR / Qt / OpenCV.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "door_belief.h"
#include "door_geometry.h"

namespace rc::door
{

// ─── the nuisances the field likelihood marginalises, for THIS aperture ──────────────────────────────
// Depth keeps its measured span (DoorBelief::FieldNuisance). The along-wall SLIDE spans +-w/2 — a doorway
// misplaced by more than half its own width is a different doorway — at the depth grid's own step, so the
// two nuisance axes are resolved equally (and the common mode they leave behind is derived from that step,
// see DoorBelief::phi_field_loglik).
inline rc::DoorBelief::FieldNuisance phi_field_nuisance(const Aperture& ap)
{
    rc::DoorBelief::FieldNuisance nz;
    const float step = nz.depth_n > 1 ? (nz.depth_hi - nz.depth_lo) / static_cast<float>(nz.depth_n - 1) : 0.05f;
    const float half = 0.5f * std::max(0.0f, ap.w);
    const int   k    = static_cast<int>(std::ceil(half / std::max(1e-3f, step)));
    nz.lat_lo = -static_cast<float>(k) * step;
    nz.lat_hi = +static_cast<float>(k) * step;
    nz.lat_n  = 2 * k + 1;
    return nz;
}

// ─── rays that interrogate this doorway ───────────────────────────────────────────────────────────────
// Unchanged from DoorFitter::estimate_phi (see the comments there for the slab and the jamb band), except
// that the lateral band is widened by the slide the likelihood will consider: a hypothesis that places the
// doorway `lat_half` along the wall must be scored on the rays that pass through THAT doorway too.
struct DoorwayRays
{
    rc::ai::LidarRays rays;
    int off_plane = 0;   // returns more than 10 cm off the wall plane (diagnostic only)
};
inline DoorwayRays select_doorway_rays(const Aperture& ap, const Eigen::Vector3f& origin,
                                       const std::vector<Eigen::Vector3f>& pts, float lat_half,
                                       float precision)
{
    DoorwayRays out;
    auto& rays = out.rays;
    const Eigen::Vector2f c = ap.centre_xy();
    const float reach = ap.w + 0.15f;                       // swept radius + thickness/noise band
    const float z_lo  = ap.floor_z + 0.05f, z_hi = ap.floor_z + ap.h;
    rays.origin = origin;
    rays.endpoints.reserve(512);
    const Eigen::Vector2f o2 = origin.head<2>();
    for (const auto& p : pts)
    {
        const Eigen::Vector2f d = p.head<2>() - o2;
        const float len2 = d.squaredNorm();
        if (len2 < 1e-6f) continue;
        const float t = std::clamp((c - o2).dot(d) / len2, 0.0f, 1.0f);
        const Eigen::Vector2f closest = o2 + t * d;
        const float lateral = std::abs(ap.wall_u.dot(closest - c));
        const float perp    = std::abs(ap.across_u().dot(closest - c));
        if (lateral > 0.5f * ap.w + std::max(0.0f, lat_half) + 0.40f or perp > reach) continue;
        const bool stopped_near = (p.head<2>() - c).norm() <= reach;
        if (stopped_near and (p.z() < z_lo or p.z() > z_hi)) continue;   // floor / above the lintel
        if (std::abs(ap.across_u().dot(p.head<2>() - c)) > 0.10f) ++out.off_plane;
        // ★ A RAY THAT NEVER REACHED THE DOORWAY TESTED NOTHING ABOUT ITS LEAF (2026-10-18, user: "for an open door
        //   the LiDAR has to go through, and vice versa"). OPEN is shown by returns BEYOND the opening, SHUT by returns
        //   ON the leaf plane; a return short of the plane hit something in front of the door. Kept in, such returns
        //   let a leaf swung toward the robot "explain" the doorway recess's side walls: live try59, the robot 3.5 m
        //   away saw 0 returns on the shut leaf and ~190 short ones on the hinge side, and phi went 0.00 → 1.69 rad
        //   (σ 0.045) on a CLOSED door — no ray had reached the opening to say it was blocked. Kept: returns that reach
        //   the plane, within 3σ of range noise plus half the wall's depth.
        const float s_o = ap.across_u().dot(o2 - c), s_p = ap.across_u().dot(p.head<2>() - c);
        const float tol = 3.0f * 0.05f + 0.5f * std::max(0.0f, ap.thickness);
        if (s_o != 0.0f and std::copysign(1.0f, s_o) * s_p > tol) continue;   // ended on the robot's side, short of it
        rays.endpoints.push_back(p);
    }
    rays.precision   = std::max(1e-3f, precision);
    rays.max_range_m = 8.0f;
    rays.robust_c_m  = 0.05f;   // the beam model's RANGE NOISE sigma (see estimate_phi)
    return out;
}

// ─── the predict step: how wide the carried belief is by now ─────────────────────────────────────────
// Same model as before (compounded with the swing the leaf could have made in dt), factored out so the
// harness runs it. dt = +inf ⇒ first look ⇒ no prior.
inline float phi_prior_sigma(float prev_sigma, float swing_rate, float dt, float step_rad, float max_rad)
{
    if (not std::isfinite(dt))
        return max_rad;
    const float g = swing_rate * dt;
    return std::clamp(std::sqrt(prev_sigma * prev_sigma + g * g), step_rad, max_rad);
}

// ─── the update step: one frame's likelihood x the carried prior ─────────────────────────────────────
struct PhiPosteriorStep
{
    std::vector<std::pair<float, float>> curve;   // (phi, posterior weight), max weight == 1
    float mean  = 0.0f;
    float sigma = 0.0f;
    bool  measured = false;
    float cv = 0.0f;           // spread of the (normalised) likelihood across angles; 0 = flat
    float w_lik_best = 0.0f;   // normalised likelihood x prior at the posterior's best angle (dbg_w_sdf)
};

// `loglik` is (phi, log-likelihood up to a constant) on the SAME grid, or empty for "not measured".
// ★THE POSTERIOR IS FORMED IN THE LOG DOMAIN. It used to be float(s_n * prior) with s_n = L/sum(L): a
// likelihood delta at an angle the prior had underflowed made EVERY weight exactly zero, which the caller
// read as "no curve" and answered by resetting sigma to the whole range — the persistence prior was thrown
// away exactly when the frame disagreed with it most, and the next frame decided the angle alone. In logs
// the max term is exp(0) = 1, so the posterior is always a proper curve: the prior can lose an argument,
// but it can never be deleted by rounding.
// ★A FLAT LIKELIHOOD IS NOT A MEASUREMENT (unchanged): `measured` is the likelihood's coefficient of
// variation across the hypotheses, and an unmeasured frame leaves the prior alone.
inline PhiPosteriorStep phi_posterior_step(const std::vector<std::pair<float, float>>& loglik,
                                           float phi_min, float phi_max, int nstep,
                                           float prior_mean, float prior_sigma)
{
    PhiPosteriorStep out;
    if (nstep < 2)
        return out;
    const std::size_t N = static_cast<std::size_t>(nstep);
    std::vector<double> lik(N, 0.0);
    double lsum = 0.0;
    if (loglik.size() == N)
    {
        double lmax = -std::numeric_limits<double>::infinity();
        for (const auto& [phi, l] : loglik) if (std::isfinite(l)) lmax = std::max(lmax, static_cast<double>(l));
        if (std::isfinite(lmax))
            for (std::size_t k = 0; k < N; ++k)
            {
                const double l = loglik[k].second;
                lik[k] = std::isfinite(l) ? std::exp(l - lmax) : 0.0;
                lsum += lik[k];
            }
    }
    // ⚠ A tenth of the mean (coefficient of variation) is where a shape is barely distinguishable from
    // uniform. It gates only the MEASURED FLAG, i.e. whether ignorance accumulates — never the estimate.
    if (lsum > 1e-12)
    {
        const double mu = lsum / static_cast<double>(N);
        double v = 0.0;
        for (const double x : lik) v += (x - mu) * (x - mu);
        out.cv = static_cast<float>(std::sqrt(v / static_cast<double>(N)) / mu);
    }
    out.measured = out.cv > 0.10f;

    const float sp = std::max(1e-4f, prior_sigma);
    std::vector<double> lp(N);
    double lpmax = -std::numeric_limits<double>::infinity();
    for (std::size_t k = 0; k < N; ++k)
    {
        const float  phi = phi_min + (phi_max - phi_min) * static_cast<float>(k) / static_cast<float>(nstep - 1);
        const double d   = (phi - prior_mean) / sp;
        const double ll  = (out.measured and lik[k] > 0.0) ? std::log(lik[k])
                         : (out.measured ? -std::numeric_limits<double>::infinity() : 0.0);
        lp[k] = ll - 0.5 * d * d;
        lpmax = std::max(lpmax, lp[k]);
    }
    if (not std::isfinite(lpmax))
    {
        // Every hypothesis the likelihood allows has zero likelihood — cannot happen from a finite loglik,
        // but if it does the prior stands alone rather than nothing.
        for (std::size_t k = 0; k < N; ++k)
        {
            const float phi = phi_min + (phi_max - phi_min) * static_cast<float>(k) / static_cast<float>(nstep - 1);
            const double d  = (phi - prior_mean) / sp;
            lp[k] = -0.5 * d * d;
        }
        lpmax = *std::ranges::max_element(lp);
        out.measured = false;
    }
    out.curve.reserve(N);
    double num = 0.0, den = 0.0, best = -1.0;
    for (std::size_t k = 0; k < N; ++k)
    {
        const float  phi = phi_min + (phi_max - phi_min) * static_cast<float>(k) / static_cast<float>(nstep - 1);
        const double w   = std::exp(lp[k] - lpmax);
        out.curve.emplace_back(phi, static_cast<float>(w));
        num += w * phi; den += w;
        if (w > best)
        {
            best = w;
            const double d = (phi - prior_mean) / sp;
            out.w_lik_best = static_cast<float>((lsum > 1e-12 ? lik[k] / lsum : 0.0) * std::exp(-0.5 * d * d));
        }
    }
    out.mean = static_cast<float>(num / den);   // den >= 1: the max term is exp(0)
    double var = 0.0;
    for (const auto& [phi, w] : out.curve) var += w * (phi - out.mean) * (phi - out.mean);
    out.sigma = static_cast<float>(std::sqrt(std::max(0.0, var / den)));
    return out;
}

// ─── THE HINGE IS A HIDDEN VARIABLE OF THE MODEL, NOT A CONFIG FACT (Task 11, live try69) ───────────────────────
// Which jamb carries the leaf was a config pin (Openable.HingeSide = Near) that a mirror then inherited as geometry.
// DOOR_1's real hinge is on the OTHER jamb. With one hinge in the model an open leaf has only two ways to be
// explained: over-swing it (room_1, try69: phi 2.05 rad for a leaf at 1.57) or slide the aperture one width along the
// wall so its believed hinge lands on the real one (room_2: the aperture slid 0.78 m, σ_s 0.60 → 0.03, while phi read
// 90°). Once the robot stood in front of the REAL opening, the slid aperture covered solid wall, a shut leaf explained
// it perfectly, phi fell to 0.01 rad and P(passable) to 0 for a door standing open — the cross was withdrawn.
// So the hinge h ∈ {believed, other} is marginalised. It is STATIC (a door does not re-hang itself), so its belief
// accumulates frame by frame with no process noise, as a log-odds:
//     llr ← llr + log Σ_k L_b(φ_k) N(φ_k; μ, σ) − log Σ_k L_o(φ_k) N(φ_k; μ, σ)
// (each hinge's marginal evidence under the SAME carried φ prior), and the φ curve the posterior step sees is the
// hinge MIXTURE  log[ P(b) L_b(φ) + P(o) L_o(φ) ]. A shut leaf is identical under both hinges (door_geometry.h,
// test (e)), so a closed door moves the log-odds by exactly 0 — the hinge is learned only while the leaf is open,
// which is the only time it matters.
// `loglik_this` / `loglik_other` are FieldResult::loglik / loglik_other: one grid, one common scale.
struct PhiHingeMarginal
{
    std::vector<std::pair<float, float>> loglik;   // the hinge-marginal φ log-likelihood (for phi_posterior_step)
    float hinge_llr = 0.0f;                        // posterior log P(believed hinge) / P(other)
};
inline PhiHingeMarginal phi_hinge_marginal(const std::vector<std::pair<float, float>>& loglik_this,
                                           const std::vector<std::pair<float, float>>& loglik_other,
                                           float hinge_llr, float prior_mean, float prior_sigma)
{
    PhiHingeMarginal out;
    out.hinge_llr = hinge_llr;
    if (loglik_other.size() != loglik_this.size() or loglik_this.empty())
    {
        out.loglik = loglik_this;   // one hypothesis scored: nothing to marginalise, nothing learned about the hinge
        return out;
    }
    constexpr double ninf = -std::numeric_limits<double>::infinity();
    const auto lae = [](double a, double b)
    {
        if (a == -std::numeric_limits<double>::infinity()) return b;
        if (b == -std::numeric_limits<double>::infinity()) return a;
        const double m = std::max(a, b);
        return m + std::log1p(std::exp(-std::abs(a - b)));
    };
    const auto fin = [](float v) { return std::isfinite(v) ? static_cast<double>(v) : -std::numeric_limits<double>::infinity(); };
    const double l = hinge_llr;
    const double lp_b = l >= 0 ? -std::log1p(std::exp(-l)) : l - std::log1p(std::exp(l));    // log σ(llr)
    const double lp_o = l >= 0 ? -l - std::log1p(std::exp(-l)) : -std::log1p(std::exp(l));   // log σ(−llr)
    const double sp = std::max(1e-4, static_cast<double>(prior_sigma));
    double e_b = ninf, e_o = ninf;
    out.loglik.reserve(loglik_this.size());
    for (std::size_t k = 0; k < loglik_this.size(); ++k)
    {
        const double a = fin(loglik_this[k].second), b = fin(loglik_other[k].second);
        const double d = (static_cast<double>(loglik_this[k].first) - prior_mean) / sp;
        e_b = lae(e_b, a - 0.5 * d * d);
        e_o = lae(e_o, b - 0.5 * d * d);
        const double mix = lae(a + lp_b, b + lp_o);
        out.loglik.emplace_back(loglik_this[k].first, std::isfinite(mix) ? static_cast<float>(mix)
                                                                         : -std::numeric_limits<float>::infinity());
    }
    if (std::isfinite(e_b) and std::isfinite(e_o))
        out.hinge_llr = static_cast<float>(l + e_b - e_o);
    return out;
}

// The geometry everyone else reads (the aperture fit's leaf SDF, the published leaf pose, the anchor's leaf prior) is
// ONE leaf, so it hangs from the more probable hinge. ⚑ P = 1/2 is the MAP of a binary posterior, a decision level and
// not a tuned one; the log-odds is negated with the flip, so the belief itself is unchanged — only which hypothesis is
// called "believed". Returns true when it flipped.
inline bool adopt_map_hinge(rc::DoorBelief& b, float& hinge_llr)
{
    if (not (hinge_llr < 0.0f))
        return false;
    const auto h = b.params().leaf.hinge;
    b.set_leaf_hinge(h == HingeSide::Near ? HingeSide::Far : HingeSide::Near);
    hinge_llr = -hinge_llr;
    return true;
}

}  // namespace rc::door
