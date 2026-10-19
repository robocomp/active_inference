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
#include <array>
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
        // ★ A RETURN SHORT OF THE DOORWAY PLANE IS KEPT (try97, 2026-10-19). From 2026-10-18 (01dda32d) such returns were
        //   dropped: on a shut DOOR_1 the corridor wall beside the jamb (try59, ~190 returns) was "explained" by a leaf
        //   swung toward the robot (phi 1.69 rad), because the field had no other structure to put them on. But a leaf
        //   swung toward the sensor returns EXACTLY such rays, and so does nothing else: dropping them made every leaf on
        //   the sensor's side unobservable — never supported, only refuted by rays that happened to fly past it — so the
        //   swing was inferred AWAY from whoever looked. try97: from room_1 (the side DOOR_1 opens into) the swing went to
        //   room_2 with +2309 nats and phi 110°; carried across, room_2 then held a leaf on ITS side that its own
        //   selection could not see, parked on the line of sight from the hinge (73°, 49° from the parked pose) for a
        //   leaf at 90°. A selection that depends on the hypothesis is not a likelihood. The corridor wall is now
        //   modelled for what it is (FieldNuisance::room_walls, room_walls_beside below), under every hypothesis.
        rays.endpoints.push_back(p);
    }
    rays.precision   = std::max(1e-3f, precision);
    rays.max_range_m = 8.0f;
    rays.robust_c_m  = 0.05f;   // the beam model's RANGE NOISE sigma (see estimate_phi)
    return out;
}

// ─── THE ROOM'S OTHER WALLS, as the field's background structure (try97) ─────────────────────────────────────────────
// The room polygon's edges as vertical wall segments for DoorBelief::FieldNuisance::room_walls, LESS the door's own wall:
// that one is already the field's plane-with-a-hole, and solid it would stop every ray through the opening. An edge is
// the door's wall when both its ends lie within the field's own depth support [depth_lo, depth_hi] of the doorway plane
// — the band in which the model already says the door's wall can be. No new level: a wall perpendicular to the door
// (a corridor side) starts in that band and leaves it, so it stays.
inline std::vector<std::array<Eigen::Vector2f, 2>> room_walls_beside(const Aperture& ap,
                                                                     const std::vector<Eigen::Vector2f>& polygon,
                                                                     float depth_lo, float depth_hi)
{
    std::vector<std::array<Eigen::Vector2f, 2>> out;
    const std::size_t n = polygon.size();
    if (n < 3) return out;
    const Eigen::Vector2f nrm = ap.across_u();
    const auto in_band = [&](const Eigen::Vector2f& p)
    { const float d = nrm.dot(p - ap.wall_O); return d >= depth_lo and d <= depth_hi; };
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        const Eigen::Vector2f& a = polygon[i];
        const Eigen::Vector2f& b = polygon[(i + 1) % n];
        if (in_band(a) and in_band(b)) continue;   // the door's own wall
        out.push_back({a, b});
    }
    return out;
}

// ─── THE φ GRID, ONE DEFINITION (Task 11 fix round 1) ─────────────────────────────────────────────────────────────────
// DoorFitter::estimate_phi and the passage anchor's φ curve (door_anchor_open.h) must score the same hypotheses: the
// range [0, max(0.1, phi_max)] and the step count derived from the declared resolution, as estimate_phi always did.
struct PhiGrid { float lo = 0.0f, hi = 0.1f; int n = 5; };
inline PhiGrid phi_grid(float phi_max_rad, float phi_step_rad)
{
    PhiGrid g;
    g.hi = std::max(0.1f, phi_max_rad);
    g.n  = std::clamp(static_cast<int>(std::lround(g.hi / std::max(0.01f, phi_step_rad))) + 1, 5, 64);
    return g;
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
    std::vector<std::pair<float, double>> log_curve;   // (phi, log posterior weight), max 0 — never underflows (passable_logit)
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
        out.log_curve.emplace_back(phi, lp[k] - lpmax);
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

// ─── Task 20 (try87): THE SWING IS A HIDDEN VARIABLE TOO — the (hinge × swing) marginal ─────────────────────────────────
// The swing side was a config pin (Openable.SwingDir), learned only from the passage anchor's beams (Task 12 fix D) — and
// the anchor now lives only during the cross (Task 19), where try87 leg 1 still left it at log-odds +0.0. DOOR_1 opens into
// room_1; the pin put it into room_2, and the room_2 mirror inherited that. From room_2 the open leaf stands on the FAR side
// of the wall, which a one-swing model can only explain as a leaf over-swung behind the wall (2.0 rad) or — once the robot
// faced the doorway and more beams reached the leaf — a half-shut leaf on the near side: φ 2.00 → 1.04 rad in two frames,
// σ 0.04, clear span 0.50 m < 0.70 m of body, P(passable) 0 for a door standing open. door_concept then offered `open` for
// a door that was open; the provider accepted every OPEN (a no-op), perception never read it open, and the open claim timed
// out 22 times (60 s each).
// So the leaf is marginalised over the four (hinge, swing) hypotheses the frame scored, under a mean-field prior
// P(h)·P(s) = σ(±hinge_llr)·σ(±swing_llr). Both are STATIC, so each log-odds accumulates with no process noise:
//     w_hs  = log P(h) + log P(s) + log Σ_k L_hs(φ_k) N(φ_k; μ, σ)
//     hinge_llr ← log Σ_s exp w_(believed,s) − log Σ_s exp w_(other,s)      (over the swings scored for BOTH hinges)
//     swing_llr ← log Σ_h exp w_(h,believed) − log Σ_h exp w_(h,other)      (over the hinges scored for BOTH swings)
// and the φ curve is the mixture log Σ_hs P(h)P(s) L_hs(φ). A shut leaf is the same under every hypothesis (φ = 0 lies in
// the wall plane whatever the hinge or swing), so a closed door teaches nothing about either — as the hinge alone.
// With only the two hinge curves scored this is exactly phi_hinge_marginal (the swing prior cancels).
struct PhiLeafMarginal
{
    std::vector<std::pair<float, float>> loglik;   // the (hinge × swing)-marginal φ log-likelihood (for phi_posterior_step)
    float hinge_llr = 0.0f;                        // posterior log P(believed hinge) / P(other)
    float swing_llr = 0.0f;                        // posterior log P(believed swing) / P(other)
};
// `bb` believed hinge+swing (FieldResult::loglik), `ob` other hinge (loglik_other), `bs` other swing (loglik_swing), `os`
// both flipped (loglik_both). An empty curve = that hypothesis was not scored this frame: it takes no part.
inline PhiLeafMarginal phi_leaf_marginal(const std::vector<std::pair<float, float>>& bb,
                                         const std::vector<std::pair<float, float>>& ob,
                                         const std::vector<std::pair<float, float>>& bs,
                                         const std::vector<std::pair<float, float>>& os,
                                         float hinge_llr, float swing_llr, float prior_mean, float prior_sigma)
{
    PhiLeafMarginal out;
    out.hinge_llr = hinge_llr;
    out.swing_llr = swing_llr;
    if (bb.empty())
        return out;
    constexpr double ninf = -std::numeric_limits<double>::infinity();
    const auto lae = [](double a, double b)
    {
        if (a == -std::numeric_limits<double>::infinity()) return b;
        if (b == -std::numeric_limits<double>::infinity()) return a;
        const double m = std::max(a, b);
        return m + std::log1p(std::exp(-std::abs(a - b)));
    };
    const auto fin = [](float v) { return std::isfinite(v) ? static_cast<double>(v) : -std::numeric_limits<double>::infinity(); };
    const auto log_sig = [](double l) { return l >= 0 ? -std::log1p(std::exp(-l)) : l - std::log1p(std::exp(l)); };   // log σ(l)
    const double hl = hinge_llr, sl = swing_llr;
    const double lp_h[2] = {log_sig(hl), log_sig(-hl)};   // believed, other
    const double lp_s[2] = {log_sig(sl), log_sig(-sl)};
    // [hinge][swing]
    const std::vector<std::pair<float, float>>* c[2][2] = {{&bb, &bs}, {&ob, &os}};
    bool have[2][2];
    for (int h = 0; h < 2; ++h)
        for (int s = 0; s < 2; ++s)
            have[h][s] = c[h][s]->size() == bb.size();
    // The prior over the hypotheses that were scored, renormalised (an unscored one is not "impossible", it is absent).
    double lz = ninf;
    for (int h = 0; h < 2; ++h)
        for (int s = 0; s < 2; ++s)
            if (have[h][s]) lz = lae(lz, lp_h[h] + lp_s[s]);
    const double sp = std::max(1e-4, static_cast<double>(prior_sigma));
    double e[2][2] = {{ninf, ninf}, {ninf, ninf}};
    out.loglik.reserve(bb.size());
    for (std::size_t k = 0; k < bb.size(); ++k)
    {
        const double d = (static_cast<double>(bb[k].first) - prior_mean) / sp;
        double mix = ninf;
        for (int h = 0; h < 2; ++h)
            for (int s = 0; s < 2; ++s)
                if (have[h][s])
                {
                    const double a = fin((*c[h][s])[k].second);
                    e[h][s] = lae(e[h][s], a - 0.5 * d * d);
                    mix = lae(mix, a + lp_h[h] + lp_s[s] - lz);
                }
        out.loglik.emplace_back(bb[k].first, std::isfinite(mix) ? static_cast<float>(mix) : -std::numeric_limits<float>::infinity());
    }
    // Posterior log weight of each scored hypothesis (the common normaliser cancels in every ratio below).
    double w[2][2];
    for (int h = 0; h < 2; ++h)
        for (int s = 0; s < 2; ++s)
            w[h][s] = have[h][s] ? lp_h[h] + lp_s[s] + e[h][s] : ninf;
    // Hinge: over the swings scored under BOTH hinges, so the ratio compares like with like.
    {
        double num = ninf, den = ninf;
        for (int s = 0; s < 2; ++s)
            if (have[0][s] and have[1][s]) { num = lae(num, w[0][s]); den = lae(den, w[1][s]); }
        if (std::isfinite(num) and std::isfinite(den)) out.hinge_llr = static_cast<float>(num - den);
    }
    {
        double num = ninf, den = ninf;
        for (int h = 0; h < 2; ++h)
            if (have[h][0] and have[h][1]) { num = lae(num, w[h][0]); den = lae(den, w[h][1]); }
        if (std::isfinite(num) and std::isfinite(den)) out.swing_llr = static_cast<float>(num - den);
    }
    return out;
}

// The swing the geometry swings to is the MAP one; ⚑ P = 1/2, the MAP of a binary posterior (a decision level, as the
// hinge's). The log-odds is negated with the flip, so the belief itself is unchanged. Returns true when it flipped.
inline bool adopt_map_swing(rc::DoorBelief& b, float& swing_llr)
{
    if (not (swing_llr < 0.0f))
        return false;
    b.set_leaf_swing(-b.params().leaf.swing);
    swing_llr = -swing_llr;
    return true;
}

// The geometry everyone else reads (the aperture fit's leaf SDF, the published leaf pose, the anchor's leaf prior) is
// ONE leaf, so it hangs from the more probable hinge. ⚑ P = 1/2 is the MAP of a binary posterior, a decision level and
// not a tuned one; the log-odds is negated with the flip, so the belief itself is unchanged — only which hypothesis is
// called "believed". Returns true when it flipped.
// ─── Task 15: THE APERTURE'S PLACE ALONG THE WALL, FROM THE WALL'S OWN HOLE ─────────────────────────────────────────────
// try75 leg 1: with the leaf open 90° (hinge west, toward the robot) the leaf channel's φ decayed toward shut (1.7 → 0.3 rad
// over 20 s on a flat tempered likelihood) and the mask fit — which places the aperture ONLY through the leaf panel's
// SDF at the believed φ and hinge — slid s by one door width (−0.12 → −0.87 m) onto solid wall; the LiDAR's own MAP slide
// (FieldResult::lat_map) said +0.15…+0.30 m the whole time, and nothing listened. A wrong leaf (φ or hinge) and an aperture
// slid by a width explain the door mask equally well; the WALL does not: beams that pass through the true opening cross a
// slid aperture's solid wall. So the slide's marginal likelihood from the field (FieldResult::lat_loglik: over every leaf
// angle and, when scored, both hinges, and the depth — the leaf marginalised, not conditioned on its MAP) enters the
// aperture's s as an observation: moment-matched to N(mean, var) on its grid, the grid's own quantisation (step²/12) added,
// and fused into the belief (DoorBelief::fuse_slide). A flat curve (a shut leaf hides the hole) gives var ≈ support²/12:
// weak, by construction. No level: the curve's own spread is the observation's σ.
struct SlideObservation { float mean = 0.f, var = -1.f; };
[[nodiscard]] inline SlideObservation slide_observation(const std::vector<std::pair<float, double>>& lat_loglik)
{
    SlideObservation o;
    if (lat_loglik.size() < 2) return o;
    double mx = -std::numeric_limits<double>::infinity();
    for (const auto& [x, l] : lat_loglik) if (std::isfinite(l)) mx = std::max(mx, l);
    if (not std::isfinite(mx)) return o;
    double sw = 0.0, sx = 0.0, sxx = 0.0;
    for (const auto& [x, l] : lat_loglik)
    {
        if (not std::isfinite(l)) continue;
        const double w = std::exp(l - mx);
        sw += w; sx += w * x; sxx += w * double(x) * double(x);
    }
    if (not (sw > 0.0)) return o;
    const double m = sx / sw, v = std::max(0.0, sxx / sw - m * m);
    const double step = double(lat_loglik[1].first - lat_loglik[0].first);
    o.mean = float(m);
    o.var  = float(v + step * step / 12.0);
    return o;
}

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
