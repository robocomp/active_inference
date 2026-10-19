/*
 * door_opening_edges.h — WHERE ALONG ITS WALL A DOOR IS, from the edges of the opening the LiDAR sees, registered to the
 * wall the same scan sees, with the registration error carried as a PERSISTENT nuisance (2026-10-19, try96).
 *
 * ─── WHAT THIS FIXED (live try96, legs 2-3) ──────────────────────────────────────────────────────────────────────────────
 * door_1 slid 1.1 m along its wall, onto solid wall, then read phi 0 / P(open) 0 for a door standing open. The mechanism,
 * from the run's own files (door-slide-report.md has the numbers):
 *   1. room_1's localisation broke for ~30 s (gt_error.csv t≈274-303 s): heading 10-20° wrong, position 0.3-0.6 m, while
 *      its reported σ_θ was ~0.9°. At 4.6 m from the door that misregisters the LiDAR cloud in the room frame by ~1.4 m
 *      along the wall: the hole APPEARED 1.4 m west.
 *   2. The only LiDAR writer of the aperture's along-wall offset s was Task 15's fuse_slide: every cycle (~10 Hz) the
 *      field's slide marginal was fused as an INDEPENDENT Gaussian measurement in the room frame. Hundreds of frames that
 *      share ONE pose error were counted as hundreds of observations, so s followed the misregistered hole (σ_s stayed
 *      ~5 mm the whole time: the mask frames' process noise re-opened it, the slide frames collapsed it again).
 *   3. Once s sat > w/2 from the true opening, the slide nuisance (±w/2 about the CURRENT s) could no longer see the real
 *      hole; the slid aperture covered solid wall, a shut leaf explained it, and phi went to 0 — after the pose recovered.
 * A door is static; an apparent jump of its opening is far more often the ROBOT's registration than the door. Three model
 * terms, no gate:
 *   (a) REGISTRATION TO THE WALL. The scan's rotation ε and across-wall offset δ relative to the model wall are estimated
 *       from the wall's own returns either side of the opening (a 1-D rotation search + total-least-squares refit, as
 *       controller/src/doorway_gap.h refits its wall line) and the sweep is un-rotated about the sensor and shifted back
 *       before ANY doorway likelihood is computed. That removes the heading part of a misregistration exactly (it is
 *       what made the hole appear 0.9 of the 1.4 m away in try96) and the across part, which the field's ±0.2 m depth
 *       nuisance could not reach (try96: 0.3-0.45 m).
 *   (b) THE OPENING'S EDGES. The along-wall likelihood of s comes from the through-span: the contiguous stretch of the
 *       registered wall line the rays pass THROUGH, bounded by returns ON it (a jamb, the wall, a leaf at the plane). The
 *       aperture contains everything that passed through, so s ∈ [a_right − w, a_left] (soft-edged by each edge's own
 *       precision: the gap between the last blocked and first through bin, and the range noise). A full-width span pins s;
 *       an ajar door's narrow slit constrains it only loosely; a SHUT door shows no span and says nothing about s. No leaf
 *       hypothesis enters, so a leaf or wall segment that "fits at many offsets" (the mirror/other-hinge hypotheses the
 *       field marginal integrates over uniformly) cannot move the aperture. Which span is this door: probabilistic data
 *       association against a uniform clutter density over the window, with P(the door shows a span) = E[1 − cos φ] (the
 *       clear fraction of the aperture along the wall, under the leaf posterior).
 *   (c) THE REGISTRATION OFFSET PERSISTS. What the wall cannot register — the robot's own position error ALONG the wall —
 *       is shared by every frame of a stretch. It is a state b in the filter, observed together with s (z = s + b), a
 *       Gauss-Markov process: stationary variance σ0² = the pose chain's along-wall variance + the static range term
 *       (DoorModel.AI2RangeNoiseLatPerM · range, the same common mode the mask frames carry), correlation time
 *       kRegistrationTauS, and process noise grown by the frame-to-frame change of the VISIBLE (across-wall) registration
 *       offset — position error is isotropic, so when the across component jumps, the hidden along component jumps as
 *       much. A static door with a settled s then sends an apparent shift into b; a door born at a wrong offset (σ_s ≫ σ0)
 *       moves onto the measured opening in a few frames.
 *
 * ⚑ MODEL CONSTANTS AND DISCRETISATIONS, flagged (CLAUDE.md):
 *   kRegYawMaxRad = 0.4 rad   the rotation search range: the scale of the gross registration errors seen live (try96 20°,
 *                             try73 21°) — the controller's kGapGrossYawRad. A search support, not a level.
 *   kRegYawStepRad = 0.5°     the search grid; the TLS refit then resolves the angle continuously.
 *   kRegDepthMaxM = 0.6 m     the across-wall search range (try96 needed 0.45 m).
 *   kRegSigmaRM = 0.03 m      the wall-return band (range noise), for the support count and the inlier refit.
 *   kRegWindowM = 1.0 m       the wall stretch beside the opening used to register and to look for spans — the
 *                             controller's kGapWallFitFarM ("within one wall").
 *   kRegMinInliers = 8        fewer wall returns than this and the frame is NOT registered (identity) — a line from a
 *                             handful of points is not better than the pose (kGapWallFitMinPts).
 *   kEdgeBinM = 0.02 m        along-wall bin of the span scan (below the beam spacing at the ranges that matter).
 *   kRegistrationTauS = 5 s   correlation time of the room localiser's error. try96's bad stretch lasted ~30 s, i.e. ~6
 *                             independent draws, against the hundreds of frames the old fusion counted.
 *   clutter density 1/L       one spurious opening expected per search window (an uninformative choice).
 *
 * Header-only, pure Eigen + std (no DSR / Qt), so tests/test_door_slide.cpp and tests/test_door_phi_stability.cpp run
 * the SAME code DoorFitter::estimate_phi runs.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <vector>

#include <Eigen/Dense>

#include "door_belief.h"
#include "door_geometry.h"

namespace rc::door
{

inline constexpr float kRegYawMaxRad     = 0.4f;
inline constexpr float kRegYawStepRad    = 0.0087266f;   // 0.5°
inline constexpr float kRegDepthMaxM     = 0.6f;
inline constexpr float kRegSigmaRM       = 0.03f;
inline constexpr float kRegWindowM       = 1.0f;
inline constexpr int   kRegMinInliers    = 8;
inline constexpr float kEdgeBinM         = 0.02f;
inline constexpr float kRegistrationTauS = 5.0f;

// ─── (a) registration of the scan to the model wall ──────────────────────────────────────────────────────────────────────
// eps: the scan's rotation about the SENSOR relative to the model wall (rad, positive = from u toward n); depth: the
// scan's wall line's across-wall offset from the model line at the aperture centre, AFTER the rotation is undone (m).
struct WallRegistration
{
    bool  ok = false;
    float eps = 0.f, eps_var = 0.f;
    float depth = 0.f, depth_var = 0.f;
    int   n_inliers = 0;
};

namespace detail
{
inline Eigen::Vector2f rot2(const Eigen::Vector2f& v, float a)
{
    const float c = std::cos(a), s = std::sin(a);
    return {c * v.x() - s * v.y(), s * v.x() + c * v.y()};
}
}   // namespace detail

// The wall returns are those at the door's height band within kRegWindowM either side of the aperture (the hole has
// none, a leaf at 90° and a perpendicular corner wall make lines the rotation search cannot align with the wall).
[[nodiscard]] inline WallRegistration register_to_wall(const Aperture& ap, const Eigen::Vector3f& origin,
                                                       const std::vector<Eigen::Vector3f>& pts)
{
    WallRegistration out;
    const Eigen::Vector2f u = ap.wall_u, n = ap.across_u(), c = ap.centre_xy(), o = origin.head<2>();
    const float half_len = 0.5f * ap.w + kRegWindowM;
    // Candidates: anything that could be the wall under a rotation up to kRegYawMaxRad about the sensor.
    const float reach = (c - o).norm() + half_len;
    const float across_max = kRegDepthMaxM + reach * std::sin(kRegYawMaxRad);
    const float z_lo = ap.floor_z + 0.05f, z_hi = ap.floor_z + ap.h;
    std::vector<Eigen::Vector2f> cand;   // sensor-relative xy
    cand.reserve(1024);
    for (const auto& p : pts)
    {
        if (p.z() < z_lo or p.z() > z_hi) continue;
        const Eigen::Vector2f q = p.head<2>();
        const float a = u.dot(q - c), r = n.dot(q - c);
        if (std::abs(a) > half_len + reach * std::sin(kRegYawMaxRad) or std::abs(r) > across_max) continue;
        cand.push_back(q - o);
    }
    if (static_cast<int>(cand.size()) < kRegMinInliers)
        return out;
    // Across-wall coordinate of a sensor-relative point q after undoing a rotation eps: n·(o + R(−eps) q − c).
    const float r_o = n.dot(o - c);
    const int   nb  = static_cast<int>(std::ceil(2.f * kRegDepthMaxM / kRegSigmaRM));
    std::vector<int> hist(static_cast<std::size_t>(nb));
    int best_support = -1; float best_eps = 0.f, best_depth = 0.f;
    const int ne = static_cast<int>(std::lround(kRegYawMaxRad / kRegYawStepRad));
    // From 0 outward, strictly greater support wins: on a tie the smaller rotation stands.
    for (int k = 0; k <= 2 * ne; ++k)
    {
        const int   j   = (k % 2 == 0) ? k / 2 : -(k + 1) / 2;
        const float eps = static_cast<float>(j) * kRegYawStepRad;
        std::ranges::fill(hist, 0);
        const float ce = std::cos(-eps), se = std::sin(-eps);
        for (const auto& q : cand)
        {
            const Eigen::Vector2f qr(ce * q.x() - se * q.y(), se * q.x() + ce * q.y());
            const float a = u.dot(qr) + u.dot(o - c);
            if (std::abs(a) > half_len) continue;
            const float r = n.dot(qr) + r_o;
            const int b = static_cast<int>(std::floor((r + kRegDepthMaxM) / kRegSigmaRM));
            if (b >= 0 and b < nb) ++hist[static_cast<std::size_t>(b)];
        }
        for (int b = 0; b < nb; ++b)
        {
            const int s = hist[static_cast<std::size_t>(b)] + (b > 0 ? hist[static_cast<std::size_t>(b - 1)] : 0)
                        + (b + 1 < nb ? hist[static_cast<std::size_t>(b + 1)] : 0);
            if (s > best_support)
            { best_support = s; best_eps = eps; best_depth = -kRegDepthMaxM + (static_cast<float>(b) + 0.5f) * kRegSigmaRM; }
        }
    }
    // Total-least-squares refit on the inliers of the winner: the slope is the residual rotation, the intercept the depth.
    const auto refit = [&](float eps, float depth, float band, double& sa, double& sr, double& saa, double& sar, int& m)
    {
        sa = sr = saa = sar = 0.0; m = 0;
        for (const auto& q : cand)
        {
            const Eigen::Vector2f qr = detail::rot2(q, -eps);
            const float a = u.dot(qr) + u.dot(o - c);
            if (std::abs(a) > half_len) continue;
            const float r = n.dot(qr) + r_o;
            if (std::abs(r - depth) > band) continue;
            sa += a; sr += r; saa += double(a) * a; sar += double(a) * r; ++m;
        }
    };
    float eps = best_eps, depth = best_depth;
    double sa, sr, saa, sar; int m = 0;
    for (int it = 0; it < 2; ++it)
    {
        refit(eps, depth, 2.f * kRegSigmaRM, sa, sr, saa, sar, m);
        if (m < kRegMinInliers) return out;
        const double ma = sa / m, mr = sr / m, vaa = saa / m - ma * ma, var_ = sar / m - ma * mr;
        if (not (vaa > 1e-6)) return out;
        const double slope = var_ / vaa;
        eps   += static_cast<float>(std::atan(slope));   // the line still leans by atan(slope) from u toward n
        depth  = static_cast<float>(mr - slope * ma);    // its across offset at the aperture centre (a = 0)
    }
    // Final moments at the refined rotation.
    refit(eps, depth, 2.f * kRegSigmaRM, sa, sr, saa, sar, m);
    if (m < kRegMinInliers) return out;
    const double ma = sa / m, vaa = std::max(1e-6, saa / m - ma * ma);
    const double s2 = double(kRegSigmaRM) * kRegSigmaRM;
    out.ok        = true;
    out.eps       = eps;
    out.depth     = static_cast<float>(sr / m);
    out.eps_var   = static_cast<float>(s2 / (m * vaa));
    out.depth_var = static_cast<float>(s2 / m + ma * ma * out.eps_var);
    out.n_inliers = m;
    return out;
}

// Undo the registration: un-rotate every point about the sensor by eps, then shift it (and the sensor) back across the
// wall by depth. Returns the registered sensor origin.
inline Eigen::Vector3f apply_registration(const WallRegistration& r, const Aperture& ap, const Eigen::Vector3f& origin,
                                          std::vector<Eigen::Vector3f>& pts)
{
    if (not r.ok) return origin;
    const Eigen::Vector2f o = origin.head<2>(), dn = r.depth * ap.across_u();
    const float ce = std::cos(-r.eps), se = std::sin(-r.eps);
    for (auto& p : pts)
    {
        const Eigen::Vector2f q = p.head<2>() - o;
        const Eigen::Vector2f qr(ce * q.x() - se * q.y(), se * q.x() + ce * q.y());
        p.head<2>() = o + qr - dn;
    }
    Eigen::Vector3f oo = origin;
    oo.head<2>() -= dn;
    return oo;
}

// ─── (b) the through-spans along the registered wall line ────────────────────────────────────────────────────────────────
// a_left / a_right: the span's edges in the wall frame (along u from wall_O, the same coordinate as the belief's s), each
// with its own variance (the unresolved gap between the last blocked and the first through bin, uniform, + range noise).
struct OpeningRun
{
    float a_left = 0.f, a_right = 0.f;
    float var_left = 0.f, var_right = 0.f;
    int   n_through = 0;
};

// A ray (sensor → return) that crosses the wall line at the doorway's height is THROUGH if its return lies beyond the
// line by more than the wall's half-thickness plus 3σ of range noise, BLOCKED if it lies on it within that, and UNKNOWN if
// it stopped short (something in front of the wall — a leaf on the robot's side, a person). The tolerance is
// select_doorway_rays' own (3·0.05 + T/2). Runs of through-bins (unknown bins bridged) bounded by blocked bins on BOTH
// sides are the openings; an empty or one-sided result is no measurement.
[[nodiscard]] inline std::vector<OpeningRun> measure_openings(const Aperture& ap, const Eigen::Vector3f& origin,
                                                              const std::vector<Eigen::Vector3f>& pts)
{
    std::vector<OpeningRun> out;
    const Eigen::Vector2f u = ap.wall_u, n = ap.across_u(), O = ap.wall_O, o = origin.head<2>();
    const float s_o = n.dot(o - O);
    const float tol = 3.0f * 0.05f + 0.5f * std::max(0.0f, ap.thickness);
    if (std::abs(s_o) <= tol) return out;   // the sensor is in the wall's slab: no ray crosses the line usefully
    const float side = s_o > 0.f ? 1.f : -1.f;   // the robot's side; "beyond" is −side
    const float a_lo = ap.s - kRegWindowM, a_hi = ap.s + ap.w + kRegWindowM;
    const int nb = static_cast<int>(std::ceil((a_hi - a_lo) / kEdgeBinM));
    if (nb < 3) return out;
    std::vector<int> nT(static_cast<std::size_t>(nb), 0), nB(static_cast<std::size_t>(nb), 0);
    const float z_lo = ap.floor_z + 0.05f, z_hi = ap.floor_z + ap.h;
    for (const auto& p : pts)
    {
        const Eigen::Vector3f d = p - origin;
        const float dn = n.dot(d.head<2>());
        if (side * dn >= 0.f) continue;                 // the ray does not head toward the line
        const float t = -s_o / dn;                      // fraction of the ray at which it meets the line
        const float beyond = -side * (s_o + dn);        // the return's distance past the line (+ = through)
        if (t >= 1.f and beyond < -tol) continue;   // stopped in front of the line: something else (unknown)
        // Where the ray meets the line (the return itself when it stopped on it): the doorway's height band only.
        const Eigen::Vector3f X = t >= 1.f ? p : Eigen::Vector3f(origin + t * d);
        if (X.z() < z_lo or X.z() > z_hi) continue;
        // A return ON the line is the wall, a jamb or a leaf only if IT is at that height: a downward beam that crossed
        // the line at 0.1 m and hit the next room's floor a few cm beyond it is the floor, which blocks nothing.
        if (beyond <= tol and (p.z() < z_lo or p.z() > z_hi)) continue;
        const float a = u.dot(X.head<2>() - O);
        const int b = static_cast<int>(std::floor((a - a_lo) / kEdgeBinM));
        if (b < 0 or b >= nb) continue;
        if (beyond > tol) ++nT[static_cast<std::size_t>(b)];
        else              ++nB[static_cast<std::size_t>(b)];
    }
    const auto centre = [&](int b) { return a_lo + (static_cast<float>(b) + 0.5f) * kEdgeBinM; };
    // Along the wall a ray's crossing is exact in angle; what is unresolved is the gap between the last blocked and the
    // first through bin (uniform, gap²/12) plus each of those two bins' own quantisation.
    const float sr2 = kEdgeBinM * kEdgeBinM / 6.f;
    int last_blocked = -1;          // the last bin with a blocked return
    int first_through = -1, last_through = -1, n_through = 0;
    for (int b = 0; b < nb; ++b)
    {
        const bool T = nT[static_cast<std::size_t>(b)] > 0, B = nB[static_cast<std::size_t>(b)] > 0;
        if (B)
        {
            // A blocked bin closes the open run (if any) — and, when it also holds through rays, IS the edge.
            if (first_through >= 0 and last_blocked >= 0)
            {
                OpeningRun r;
                const float lg = static_cast<float>(first_through - last_blocked) * kEdgeBinM;
                const float rg = static_cast<float>(b - last_through) * kEdgeBinM;
                r.a_left    = 0.5f * (centre(last_blocked) + centre(first_through));
                r.a_right   = 0.5f * (centre(last_through) + centre(b));
                r.var_left  = lg * lg / 12.f + sr2;
                r.var_right = rg * rg / 12.f + sr2;
                r.n_through = n_through;
                out.push_back(r);
            }
            last_blocked = b;
            first_through = last_through = -1; n_through = 0;
            if (T) { first_through = last_through = b; n_through = nT[static_cast<std::size_t>(b)]; }
            continue;
        }
        if (T)
        {
            if (first_through < 0) first_through = b;
            last_through = b;
            n_through += nT[static_cast<std::size_t>(b)];
        }
    }
    return out;
}

// P(the door shows a through-span) = the expected clear fraction of its aperture along the wall, 1 − cos φ, under the leaf
// posterior (phi, weight) — 1 wide open, 0 shut. An empty curve (no leaf belief) counts as an unknown leaf: ½.
[[nodiscard]] inline float clear_fraction(const std::vector<std::pair<float, float>>& curve)
{
    double sw = 0.0, sc = 0.0;
    for (const auto& [phi, w] : curve)
        if (std::isfinite(w) and w > 0.f) { sw += w; sc += w * (1.0 - std::cos(static_cast<double>(phi))); }
    return sw > 0.0 ? static_cast<float>(std::clamp(sc / sw, 0.0, 1.0)) : 0.5f;
}

// ─── (c) the persistent registration offset and the update ───────────────────────────────────────────────────────────────
struct AlongWallState
{
    bool  init = false;
    float b = 0.f, Pb = 0.f;                    // the registration offset along the wall (m) and its variance
    Eigen::Vector3f c = Eigen::Vector3f::Zero();   // its cross-covariance with the belief's [s, w, h]
    bool  have_depth = false;
    float last_depth = 0.f;                     // the last frame's visible (across-wall) registration offset
};

struct AlongWallResult
{
    int   n_runs = 0;
    float z = std::numeric_limits<float>::quiet_NaN();   // the chosen span's measurement of s
    float R = 0.f, nu = 0.f, S = 0.f, beta = 0.f, p_detect = 0.f, sigma0_sq = 0.f, Q_b = 0.f;
    bool  updated = false;
};

// One cycle: predict the registration offset, then — if a span was measured — a probabilistic-data-association update of
// [s, w, h, b] with the chosen span. `sigma0_sq` is the frame's stationary along-wall registration variance (pose chain +
// range), `dt` the time since the last call, `p_detect` clear_fraction() of the leaf posterior.
inline AlongWallResult along_wall_update(rc::DoorBelief& belief, AlongWallState& st, const std::vector<OpeningRun>& runs,
                                         const WallRegistration& reg, float p_detect, float sigma0_sq, float range_m,
                                         float dt_s)
{
    AlongWallResult res;
    res.n_runs = static_cast<int>(runs.size());
    res.p_detect = std::clamp(p_detect, 0.f, 1.f);
    res.sigma0_sq = std::max(0.f, sigma0_sq);
    // ── predict b: Gauss-Markov toward 0 with stationary variance σ0², plus the change of the visible registration ──
    if (not st.init)
    {
        st.init = true; st.b = 0.f; st.Pb = res.sigma0_sq; st.c.setZero();
    }
    else
    {
        const float a = std::isfinite(dt_s) and dt_s > 0.f ? std::exp(-dt_s / kRegistrationTauS) : 1.f;
        float q = (1.f - a * a) * res.sigma0_sq;
        if (reg.ok and st.have_depth)
        {
            const float dd = reg.depth - a * st.last_depth;
            q += dd * dd;   // isotropic position error: the hidden along component moved as much as the visible one
        }
        st.b *= a; st.c *= a;
        st.Pb = a * a * st.Pb + q;
        res.Q_b = q;
    }
    if (reg.ok) { st.have_depth = true; st.last_depth = reg.depth; }
    if (runs.empty())
        return res;
    // ── each span as a measurement of s: the aperture contains it, s ∈ [a_right − w, a_left], soft-edged ──
    const auto& S3 = belief.covariance();
    const float s = belief.state().s, w = belief.state().w;
    const float var_w = std::max(0.f, S3(1, 1));
    // The residual rotation's lever arm moves the span along the wall by range·δε.
    const float r_rot = reg.ok ? range_m * range_m * reg.eps_var : 0.f;
    // Cauchy-Schwarz: the mask update re-shapes Σ without b, so keep the joint matrix a covariance.
    for (int i = 0; i < 3; ++i)
    {
        const float lim = std::sqrt(std::max(0.f, S3(i, i) * st.Pb));
        st.c(i) = std::clamp(st.c(i), -lim, lim);
    }
    const float S_prior = S3(0, 0) + 2.f * st.c(0) + st.Pb;
    const float L = w + 2.f * kRegWindowM;
    const float clutter = 1.f / std::max(1e-3f, L);
    float best_e = -1.f, sum_e = 0.f;
    float best_z = 0.f, best_R = 0.f, best_S = 0.f;
    for (const auto& r : runs)
    {
        const float lo = r.a_right - w, hi = r.a_left;
        const float slack = hi - lo;   // w − span width: 0 for a full-width span, ≈ w for a slit
        const float z = 0.5f * (lo + hi);
        const float R = slack * slack / 12.f + 0.25f * (r.var_left + r.var_right + var_w) + r_rot;
        const float Sj = S_prior + R;
        const float nu = z - (s + st.b);
        const float g = std::exp(-0.5f * nu * nu / Sj) / std::sqrt(2.f * std::numbers::pi_v<float> * Sj);
        const float e = res.p_detect * g;
        sum_e += e;
        if (e > best_e) { best_e = e; best_z = z; best_R = R; best_S = Sj; }
    }
    const float denom = clutter * (1.f - res.p_detect) + sum_e;
    res.beta = denom > 0.f ? best_e / denom : 0.f;
    res.z = best_z; res.R = best_R; res.S = best_S; res.nu = best_z - (s + st.b);
    if (res.beta > 0.f)
    {
        belief.fuse_along_wall(res.nu, best_R, res.beta, st.b, st.Pb, st.c);
        res.updated = true;
    }
    return res;
}

// The registered sweep and its sensor origin for one door, plus the registration itself (estimate_phi's first step).
struct RegisteredSweep
{
    WallRegistration reg;
    std::vector<Eigen::Vector3f> pts;
    Eigen::Vector3f origin = Eigen::Vector3f::Zero();
};
[[nodiscard]] inline RegisteredSweep register_sweep(const Aperture& ap, const Eigen::Vector3f& origin,
                                                    const std::vector<Eigen::Vector3f>& pts)
{
    RegisteredSweep out;
    out.reg = register_to_wall(ap, origin, pts);
    out.pts = pts;
    out.origin = apply_registration(out.reg, ap, origin, out.pts);
    return out;
}

}  // namespace rc::door
