/*
 * refrigerator_belief.cpp  —  AI2 refrigerator belief implementation (single floor-anchored box)
 *
 * The refrigerator-specific model hooks behind RefrigeratorBelief: ONE box SDF (floor-anchored, z∈[0,H]),
 * the soft [box, clutter] mixture responsibilities + the clutter-inclusive free energy, the finite-difference
 * (slope-clamped) Jacobian, the optional YOLO-independent LiDAR range factor, and self_test(). All Bayesian
 * bookkeeping (predict / GN-MAP / Woodbury) lives in common/ai_belief/recursive_laplace.h; this file feeds it.
 *
 * Deliberately minimal vs table_belief: no legs, no top slab, no w↔h symmetry fold / orientation-mode
 * accumulator, no footprint-moment / coverage / free-space / depth-tilt machinery.
 */

#include "refrigerator_belief.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <utility>

namespace rc
{

// ─── SDF primitive (scalar; room→local handled inline) ───────────────────────────────────────────

namespace
{
// Box SDF given the point's distances-to-face along each axis (|local| − half_extent).
float box_sdf(float dx, float dy, float dz)
{
    const float ox = std::max(dx, 0.0f), oy = std::max(dy, 0.0f), oz = std::max(dz, 0.0f);
    const float outside = std::sqrt(ox * ox + oy * oy + oz * oz);
    const float inside  = std::min(std::max(dx, std::max(dy, dz)), 0.0f);
    return outside + inside;
}
}  // namespace

// Signed distance to the solid, floor-anchored box: centre (cx,cy,H/2), half extents (w/2,h/2,H/2), yaw.
float RefrigeratorBelief::sdf_box(const Eigen::Vector3f& p, const RefrigeratorBeliefState& s) const
{
    const float c = std::cos(-s.yaw), sn = std::sin(-s.yaw);
    const float px = p.x() - s.cx, py = p.y() - s.cy;
    const float lx = px * c - py * sn;
    const float ly = px * sn + py * c;
    const float half_H = 0.5f * s.H;
    return box_sdf(std::abs(lx) - 0.5f * s.w, std::abs(ly) - 0.5f * s.h, std::abs(p.z() - half_H) - half_H);
}

float RefrigeratorBelief::sdf_prim(const Eigen::Vector3f& p, const RefrigeratorBeliefState& s, int) const
{
    return sdf_box(p, s);   // prim == 0 (the only primitive)
}

// ─── Mixture responsibilities ────────────────────────────────────────────────────────────────────

// Un-normalised mixture components u[0]=box, u[1]=clutter and their sum = the marginal likelihood numerator
// p(point|model) ∝ Σ_k π_k N(d_k; 0, R). The FREE ENERGY reads −log(sum); the clutter term keeps a far/misfit
// point at a large real penalty (−log clutter likelihood) rather than silently zeroing the energy.
float RefrigeratorBelief::mixture_unnormalized(const Eigen::Vector3f& p, const RefrigeratorBeliefState& s,
                                               float R, std::array<float, 2>& u) const
{
    const float eps    = std::clamp(params_.clutter_frac, 0.0f, 0.99f);
    const float inv2R  = 0.5f / std::max(1e-9f, R);
    const float d      = sdf_box(p, s);
    u[0] = (1.0f - eps) * std::exp(-d * d * inv2R);
    const float cs = params_.clutter_scale_m;
    u[1] = eps * std::exp(-cs * cs * inv2R);
    return u[0] + u[1];
}

std::array<float, 2> RefrigeratorBelief::responsibilities(const Eigen::Vector3f& p,
                                                          const RefrigeratorBeliefState& s, float R) const
{
    std::array<float, 2> u{};
    const float sum = mixture_unnormalized(p, s, R, u);
    if (sum <= 0.0f) { u = {0.0f, 1.0f}; return u; }
    u[0] /= sum; u[1] /= sum;
    return u;
}

float RefrigeratorBelief::mean_energy(const std::vector<Eigen::Vector3f>& pts,
                                      const RefrigeratorBeliefState& s, float R) const
{
    if (pts.empty()) return 0.0f;
    double acc = 0.0;
    std::array<float, 2> u{};
    for (const auto& p : pts)
    {
        const float sum = mixture_unnormalized(p, s, R, u);
        acc += -std::log(std::max(1e-30f, sum));
    }
    return static_cast<float>(acc / static_cast<double>(pts.size()));
}

// ─── Jacobian (central finite difference, slope-clamped) ───────────────────────────────────────────

Eigen::Matrix<float, 6, 1> RefrigeratorBelief::sdf_jacobian(const Eigen::Vector3f& p,
                                                            const RefrigeratorBeliefState& s, int prim) const
{
    Eigen::Matrix<float, 6, 1> J;
    const Eigen::Matrix<float, 6, 1> base = s.vec();
    const float e = params_.fd_eps;
    const float g = params_.jac_slope_clamp;
    for (int j = 0; j < 6; ++j)
    {
        Eigen::Matrix<float, 6, 1> vp = base, vm = base;
        vp(j) += e; vm(j) -= e;
        const float slope = (sdf_prim(p, RefrigeratorBeliefState::from_vec(vp), prim) -
                             sdf_prim(p, RefrigeratorBeliefState::from_vec(vm), prim)) / (2.0f * e);
        // CLAMP the slope: a non-smooth SDF seam can spike a per-DOF slope and blow up the information.
        J(j) = std::clamp(slope, -g, g);
    }
    return J;
}

// ─── Constraints / canonical form ──────────────────────────────────────────────────────────────────

// Per-GN-iteration bounds: floor each extent so the box can't invert or collapse, and wrap yaw. Physical
// floors (w,h,H ≥ 0.10 m), not tuned gates — a degenerate zero/negative extent has no gradient.
void RefrigeratorBelief::apply_constraints(RefrigeratorBeliefState& s) const
{
    s.w = std::max(s.w, 0.10f);
    s.h = std::max(s.h, 0.10f);
    s.H = std::max(s.H, 0.10f);
    s.yaw = std::remainder(s.yaw, 2.0f * static_cast<float>(M_PI));   // wrap to (−π, π]
}

// ─── Engine hooks: prior cov · process noise (F = I, static) · common-mode ──────────────────────────

Eigen::Matrix<float, 6, 1> RefrigeratorBelief::prior_cov_diag() const
{
    const float pp = params_.prior_pos_std       * params_.prior_pos_std;
    const float hh = params_.prior_height_std    * params_.prior_height_std;     // H: BROAD (data-driven)
    const float wd = params_.prior_footprint_std * params_.prior_footprint_std;  // w,h(depth): TIGHT
    const float yy = params_.prior_yaw_std       * params_.prior_yaw_std;
    return (Eigen::Matrix<float, 6, 1>() << pp, pp, hh, wd, wd, yy).finished();   // [cx,cy,H,w,h,yaw]
}

Eigen::Matrix<float, 6, 1> RefrigeratorBelief::process_noise_diag() const
{
    const float qm = params_.process_std_m   * params_.process_std_m;
    const float qy = params_.process_std_yaw * params_.process_std_yaw;
    return (Eigen::Matrix<float, 6, 1>() << qm, qm, qm, qm, qm, qy).finished();
}

// Inverse of the per-frame common-mode covariance Σc (diagonal): position (cx,cy) = config floor + pose-
// chain/range/motion; size (H,w,h) = config std + range/motion; yaw = config std + range/grazing/motion.
// Marginalising this SHARED error (Woodbury in the engine) makes the frame's information SATURATE at Σc.
Eigen::Matrix<float, 6, 1> RefrigeratorBelief::common_mode_inv_diag(const RefrigeratorFrame& frame) const
{
    const float p2 = params_.common_mode_pos_std  * params_.common_mode_pos_std;
    const float s2 = params_.common_mode_size_std * params_.common_mode_size_std;
    const float y2 = params_.common_mode_yaw_std  * params_.common_mode_yaw_std;
    const float cs = std::max(0.0f, frame.chain_cov_size);
    const float inv_x = 1.0f / std::max(1e-9f, p2 + std::max(0.0f, frame.chain_cov_xx));
    const float inv_y = 1.0f / std::max(1e-9f, p2 + std::max(0.0f, frame.chain_cov_yy));
    const float inv_s = 1.0f / std::max(1e-9f, s2 + cs);
    const float inv_yaw = 1.0f / std::max(1e-9f, y2 + std::max(0.0f, frame.chain_cov_yaw));
    return (Eigen::Matrix<float, 6, 1>() << inv_x, inv_y, inv_s, inv_s, inv_s, inv_yaw).finished();
}

// ─── Back face + wall-flush / wall-parallel structural factor (ported from cabinet_concept) ──────────

// Centre of the BACK face: (cx,cy) + (h/2)·outward, where outward = the depth (local-Y) axis chosen to
// point AWAY from the room interior. The box's local-Y axis (the depth h axis) is (−sin yaw, cos yaw); the
// back face is the one whose outward normal aims away from the interior — that is the face put against a wall.
Eigen::Vector2f RefrigeratorBelief::back_centre(const RefrigeratorBeliefState& s) const
{
    const Eigen::Vector2f c(s.cx, s.cy);
    const Eigen::Vector2f depth_axis(-std::sin(s.yaw), std::cos(s.yaw));   // local +Y (the depth h axis)
    float sgn = 1.0f;
    if (has_room_interior_)
        sgn = (depth_axis.dot(c - room_interior_) >= 0.0f) ? 1.0f : -1.0f;   // outward = away from interior
    return c + sgn * 0.5f * s.h * depth_axis;
}

// Posterior weight of the "flush against this wall" component (shared by the flush + parallel terms). With a
// Gaussian on the back-face-to-wall gap, the flush component's posterior weight is exp(−(gap/reach)²), so
// marginalising the discrete {flush, free-standing} component multiplies the factor precision by exactly it —
// a continuous covariance, not a proximity gate. Fades to ~0 for a genuine mid-room fridge (depth stays wide).
float RefrigeratorBelief::flush_weight(const RefrigeratorBeliefState& s, const RefrigeratorFrame& f) const
{
    if (not f.wall.ok) return 0.0f;
    const float gap = (back_centre(s) - f.wall.p).dot(f.wall.n);
    const float rel = gap / std::max(1e-3f, params_.wall_reach_m);
    return std::exp(-rel * rel);
}

// Drive the BACK-face centre onto the nearest wall line (gap→0) and the back face PARALLEL to the wall
// (width-axis · wall-normal → 0). Both precisions are scaled by the flush mixture weight, so an island
// feels ~nothing. Finite-difference Jacobians (slope-clamped), mirroring cabinet_concept::accumulate_wall.
void RefrigeratorBelief::accumulate_wall(const RefrigeratorBeliefState& s, const RefrigeratorFrame& f,
                                         Eigen::Matrix<float, 6, 6>& Id, Eigen::Matrix<float, 6, 1>& bd) const
{
    dbg_wall_gap_ = 0.0f; dbg_wall_lambda_ = 0.0f;
    if (not f.wall.ok or params_.wall_precision <= 0.0f) return;

    const auto gap_of = [&](const RefrigeratorBeliefState& st)
    { return (back_centre(st) - f.wall.p).dot(f.wall.n); };
    const auto par_of = [&](const RefrigeratorBeliefState& st)   // width axis (local +X) · wall normal
    { return Eigen::Vector2f(std::cos(st.yaw), std::sin(st.yaw)).dot(f.wall.n); };

    const float gap = gap_of(s);
    dbg_wall_gap_ = gap;

    const float wgt = flush_weight(s, f);
    const float lam_flush = wgt / (1.0f / params_.wall_precision + f.wall.sigma_m * f.wall.sigma_m);
    dbg_wall_lambda_ = lam_flush;
    if (not(lam_flush > 1e-6f)) return;

    const float eps = params_.fd_eps;
    const float g   = params_.jac_slope_clamp;
    const Eigen::Matrix<float, 6, 1> v = s.vec();
    Eigen::Matrix<float, 6, 1> Jg, Jp;
    for (int k = 0; k < 6; ++k)
    {
        Eigen::Matrix<float, 6, 1> vp = v, vm = v;
        vp(k) += eps; vm(k) -= eps;
        const auto sp = RefrigeratorBeliefState::from_vec(vp), sm = RefrigeratorBeliefState::from_vec(vm);
        Jg(k) = std::clamp((gap_of(sp) - gap_of(sm)) / (2.0f * eps), -g, g);
        Jp(k) = std::clamp((par_of(sp) - par_of(sm)) / (2.0f * eps), -g, g);
    }
    Id.noalias() += lam_flush * (Jg * Jg.transpose());
    bd.noalias() += -lam_flush * Jg * gap;

    if (params_.wall_parallel_precision > 0.0f)
    {
        const float lam_par = wgt * params_.wall_parallel_precision;   // same mixture weight
        const float rp = par_of(s);
        Id.noalias() += lam_par * (Jp * Jp.transpose());
        bd.noalias() += -lam_par * Jp * rp;
    }
}

// ─── Extra GN factors: wall-flush + footprint prior + YOLO-independent LiDAR first-hit range ─────────

void RefrigeratorBelief::accumulate_extra(const RefrigeratorBeliefState& s, const RefrigeratorFrame& f,
                                          Eigen::Matrix<float, 6, 6>& Id, Eigen::Matrix<float, 6, 1>& bd) const
{
    // (1) Wall-flush + wall-parallel structural term (supplies the missing back-face/depth information).
    accumulate_wall(s, f, Id, bd);

    // (2) FOOTPRINT prior: a TIGHT Gaussian pulling w → prior_footprint_m and h(depth) → prior_footprint_m
    //     (a standard fridge footprint ≈ 0.60×0.60 is very consistent), precision λ_wd = 1/prior_footprint_std².
    //     H (index 2) is anchored separately just below (2a) — the top-of-box float needs a static anchor too.
    if (params_.prior_footprint_std > 0.0f)
    {
        const float lam_wd = 1.0f / (params_.prior_footprint_std * params_.prior_footprint_std);
        Id(3, 3) += lam_wd;  bd(3) += lam_wd * (params_.prior_footprint_m - s.w);   // w   (index 3)

        // Depth (h, idx 4) is IDENTIFIABLE only when points fall on BOTH depth faces (the +ly AND the −ly face) —
        // that pair is what separates the extent h from the centre cy. A front-only view (even one that also grazes
        // a SIDE face — side points sit BETWEEN the depth faces, |ly|≪h/2, and are excluded by the band) puts points
        // on ONE depth face only; its many points still spuriously drag h to the clamp (the size common-mode caps σ,
        // not the mean). So the footprint prior on depth must PREVAIL absent two-face evidence: grow its precision by
        // depth_unobs_precision·(1 − two_sided), two_sided = 2·min(n₊,n₋)/(n₊+n₋) ∈ [0 one-face … 1 balanced]. A
        // covariance keyed on the right physical covariate (CLAUDE.md), not a gate; relaxes to data-driven the moment
        // the back face is genuinely observed. Robust to side/front clutter — only points AT a depth face count.
        float lam_h = lam_wd;
        if (params_.depth_unobs_precision > 0.0f and f.points.size() > 8)
        {
            const float c = std::cos(-s.yaw), sn = std::sin(-s.yaw);
            const float halfw = 0.5f * s.w, delta = params_.depth_obs_band_m;
            int nplus = 0, nminus = 0;
            for (const auto& p : f.points)
            {
                const float px = p.x() - s.cx, py = p.y() - s.cy;
                const float lx = px * c - py * sn;
                if (std::abs(lx) > halfw + delta) continue;         // width-spanning core only (drop side clutter)
                if (p.z() < 0.0f or p.z() > s.H + delta) continue;
                const float ly = px * sn + py * c;
                // Clearly FORWARD-of / BEHIND-centre by a FIXED margin δ (NOT keyed to the current half-depth). A
                // point straddling the centre (|ly| ≤ δ) counts as NEITHER — so a COLLAPSED depth slab (all points
                // near ly≈0) cannot masquerade as "both faces seen" and switch the boost OFF. That circularity was
                // re-collapsing h once it dipped. Two-sided evidence now means genuine points on BOTH sides, apart.
                if (ly >  delta)      ++nplus;
                else if (ly < -delta) ++nminus;
            }
            const int tot = nplus + nminus;
            const float two_sided = (tot > 0) ? (2.0f * std::min(nplus, nminus) / static_cast<float>(tot)) : 0.0f;
            lam_h += params_.depth_unobs_precision * (1.0f - two_sided);
        }
        Id(4, 4) += lam_h;  bd(4) += lam_h * (params_.prior_footprint_m - s.h);     // h≡depth (index 4)
    }

    // (2a) HEIGHT anchor (H, index 2): the box TOP is unconstrained-from-above — points at the real fridge top
    //      (z ≈ 1.9 m) sit on a SIDE face of any TALLER box, so they never pull H down, and prior_cov_diag's broad
    //      hh is only the recursive-filter DRIFT allowance, NOT a static anchor. Without this term H random-walks
    //      UP (observed: 2.37 m for a 1.9 m cloud) with no restoring force. This is a static Gaussian anchor toward
    //      prior_height_m with precision 1/prior_height_std² — the DATA still sets the LOWER bound (points force
    //      H ≥ cloud-top), the anchor only prevents float ABOVE it. Broad enough to let a genuinely tall/short
    //      fridge move when the cloud top demands it, firm enough to stop the runaway.
    if (params_.prior_height_std > 0.0f)
    {
        const float lam_H = 1.0f / (params_.prior_height_std * params_.prior_height_std);
        Id(2, 2) += lam_H;  bd(2) += lam_H * (params_.prior_height_m - s.H);        // H (index 2)
    }

    // (2d) TOP TRACKS THE OBSERVED ROBUST CLOUD TOP. The floor-anchored box top is a "free" upper boundary — empty
    //      surface above the cloud is unpenalised, so H ratchets up on the over-segmented JUNK TAIL above the fridge
    //      (a mask whose real top is ~1.93 m still has a few % of points running to ~2.3 m → H floated to 2.37). A
    //      soft one-sided cap loses to that tail (its force grows AS H drops and more junk is exposed). Instead pin H
    //      FIRMLY to z_top_obs (the p97 top, which sits BELOW the junk tail): a two-sided anchor whose force competes
    //      directly with the tail. The data lower-bound (front points force H ≥ real top) keeps it from under-fitting.
    //      z_top_obs is data-derived (the observed top), so a firm precision is honest, not a magic height number.
    if (f.z_top_obs > 0.0f and params_.top_no_float_precision > 0.0f)
    {
        const float lam_tf = params_.top_no_float_precision;
        const float target = f.z_top_obs + params_.top_no_float_margin_m;
        Id(2, 2) += lam_tf;  bd(2) += lam_tf * (target - s.H);   // pull H → observed robust top (both directions)
    }

    // (2b) SHORT-HEIGHT prior (the "a 70 cm fridge is improbable" term): a ONE-SIDED Gaussian factor that acts
    //      ONLY when H < plaus_height_min, pulling H up toward it. Its precision GROWS as H falls further below
    //      (λ_h = gain·deficit/soft), so a mildly-short fridge is nudged gently and a 70 cm cloud is resisted
    //      hard. A plausibly-tall fridge (deficit ≤ 0) feels NOTHING. Continuous covariance, not a clamp — the
    //      effect is that fitting a short cloud as a fridge fights this prior → worse data fit → lower plausibility.
    if (params_.plaus_height_prior_gain > 0.0f and params_.plaus_height_soft > 1e-4f)
    {
        const float deficit = params_.plaus_height_min - s.H;   // >0 ⇒ implausibly short
        if (deficit > 0.0f)
        {
            const float lam_h = params_.plaus_height_prior_gain * (deficit / params_.plaus_height_soft);
            Id(2, 2) += lam_h;  bd(2) += lam_h * deficit;   // residual = (height_min − H) = deficit ⇒ pull H up (idx 2)
        }
    }

    // (2c) WALL NO-CROSS (one-sided): the flush term (accumulate_wall) is TWO-SIDED — it drives the back-face gap
    //      → 0 but does NOT resist the box being fit CROSSING the wall (back extending PAST it into the wall/
    //      exterior), and its flush weight exp(−(gap/reach)²) has decayed to ~0 once the back is well past, so it
    //      cannot pull it back. This ONE-SIDED factor (same form as the short-height prior above) resists
    //      penetration only: with gap = (back_centre − wall.p)·(inward normal) (>0 inside the room, <0 crossed),
    //      residual = min(0, gap − margin) is driven to 0 with a precision that GROWS as the back penetrates
    //      further (λ = wall_no_cross_precision · how-far-past). Flush (gap≈0) feels ~nothing; a wall is a hard
    //      boundary so it is strong. Continuous (soft), no clamp. Inert when f.wall.ok==false.
    if (f.wall.ok and params_.wall_no_cross_precision > 0.0f)
    {
        const auto gap_of = [&](const RefrigeratorBeliefState& st)
        { return (back_centre(st) - f.wall.p).dot(f.wall.n); };   // >0 interior side, <0 crossed the wall
        const float gap      = gap_of(s);
        const float residual = std::min(0.0f, gap - params_.wall_no_cross_margin_m);   // <0 ⇒ crossing (or within margin)
        if (residual < 0.0f)
        {
            const float how_far_past = -residual;                                       // m past the boundary (>0)
            const float lam_nc = params_.wall_no_cross_precision * how_far_past;         // precision grows with penetration
            const float eps = params_.fd_eps;
            const float g   = params_.jac_slope_clamp;
            const Eigen::Matrix<float, 6, 1> v = s.vec();
            Eigen::Matrix<float, 6, 1> Jg;
            for (int k = 0; k < 6; ++k)
            {
                Eigen::Matrix<float, 6, 1> vp = v, vm = v;
                vp(k) += eps; vm(k) -= eps;
                Jg(k) = std::clamp((gap_of(RefrigeratorBeliefState::from_vec(vp))
                                  - gap_of(RefrigeratorBeliefState::from_vec(vm))) / (2.0f * eps), -g, g);
            }
            Id.noalias() += lam_nc * (Jg * Jg.transpose());
            bd.noalias() += -lam_nc * Jg * residual;   // residual<0 ⇒ push the gap UP (back out of the wall)
        }
    }

    // (3) YOLO-independent LiDAR first-hit range factor, once per device ray-set (each own origin).
    rc::ai::accumulate_lidar_rays<6>(*this, s, f.lidar, Id, bd);
    for (const auto& lr : f.lidar_extra)   // extra per-device ray-sets (e.g. low bpearl) — each own origin
        rc::ai::accumulate_lidar_rays<6>(*this, s, lr, Id, bd);
}

// ─── Footprint second-moment (geometry helper for the optional birth seed) ──────────────────────────

FootprintMoment RefrigeratorBelief::footprint_moment(const std::vector<Eigen::Vector3f>& pts,
                                                     float z_lo, float z_hi)
{
    FootprintMoment fm;
    double sx = 0, sy = 0; int n = 0;
    for (const auto& p : pts)
        if (p.z() >= z_lo and p.z() <= z_hi) { sx += p.x(); sy += p.y(); ++n; }
    if (n < 8) return fm;
    const double cx = sx / n, cy = sy / n;
    double xx = 0, yy = 0, xy = 0;
    for (const auto& p : pts)
        if (p.z() >= z_lo and p.z() <= z_hi)
        {
            const double dx = p.x() - cx, dy = p.y() - cy;
            xx += dx * dx; yy += dy * dy; xy += dx * dy;
        }
    xx /= n; yy /= n; xy /= n;
    // 2×2 symmetric inertia eigen-decomposition (closed form).
    const double tr = xx + yy, det = xx * yy - xy * xy;
    const double disc = std::sqrt(std::max(0.0, 0.25 * tr * tr - det));
    const double l1 = 0.5 * tr + disc, l2 = 0.5 * tr - disc;   // l1 ≥ l2
    fm.ok = true; fm.n = n;
    fm.cx = static_cast<float>(cx); fm.cy = static_cast<float>(cy);
    fm.ext_major = static_cast<float>(std::sqrt(std::max(0.0, 12.0 * l1)));   // full extent of uniform rect
    fm.ext_minor = static_cast<float>(std::sqrt(std::max(0.0, 12.0 * l2)));
    fm.phi = static_cast<float>(0.5 * std::atan2(2.0 * xy, xx - yy));
    return fm;
}

// ─── FRONT (door) yaw resolver — sequential Bayes over the discrete door-facing modes ────────────

// Which discrete yaw offsets the footprint leaves open (see the header): a clearly rectangular footprint is only
// 180°-ambiguous; a near-square one leaves all four modes.
std::array<bool, 4> RefrigeratorBelief::allowed_modes() const
{
    const float sum_wh = state_.w + state_.h;
    const bool rect = sum_wh > 1e-4f and std::abs(state_.w - state_.h) / sum_wh > kRectAspect;
    return {true, not rect, true, not rect};
}

// Adopt door mode k: rotate yaw + (for a 90°/270° swap) relabel w↔h and swap the matching Σ / prior rows+cols.
void RefrigeratorBelief::apply_mode_rotation(int k)
{
    constexpr float kHalfPi = 0.5f * static_cast<float>(M_PI);
    const auto wrap = [](float a) { return std::remainder(a, 2.0f * static_cast<float>(M_PI)); };
    state_.yaw = wrap(state_.yaw + static_cast<float>(k) * kHalfPi);
    if (k == 1 or k == 3)
    {
        std::swap(state_.w, state_.h);                 // width (idx 3) ↔ depth (idx 4) relabel
        Sigma_.row(3).swap(Sigma_.row(4));
        Sigma_.col(3).swap(Sigma_.col(4));
        std::swap(prior_mean_(3), prior_mean_(4));
    }
}

// Fold one appearance FrontCue into the door-mode accumulator and adopt the argmax mode. See the header.
bool RefrigeratorBelief::resolve_front(const FrontCue& cue, float evidence_weight)
{
    constexpr float kHalfPi    = 0.5f * static_cast<float>(M_PI);
    // Clamp each accumulated mode-evidence to ±kFrontClamp so a long confident run cannot drive the accumulator
    // arbitrarily deep and make the belief unable to RECANT when the fridge is physically turned. ≈6 nats ≈ a
    // 400:1 odds — decisive but recoverable. Mirrors ChairBelief's ±6 flip clamp.
    constexpr float kFrontClamp = 6.0f;
    const float w = std::clamp(evidence_weight, 0.0f, 1.0f) * std::clamp(cue.confidence, 0.0f, 1.0f);
    if (w <= 1e-4f)
        return false;   // no confident evidence this cue → leave the accumulator (and the belief) untouched

    const auto wrap = [](float a) { return std::remainder(a, 2.0f * static_cast<float>(M_PI)); };
    const Eigen::Vector2f cue_dir(std::cos(cue.bearing_rad), std::sin(cue.bearing_rad));
    const std::array<bool, 4> allowed = allowed_modes();

    // Accumulate evidence for every ALLOWED candidate: the alignment of that mode's door normal (local −Y rotated
    // by the candidate yaw ψ_k) with the observed door bearing. Door local −Y ⇒ room dir R(ψ)·(0,−1)=(sinψ,−cosψ).
    for (int k = 0; k < 4; ++k)
    {
        if (not allowed[k])
            continue;
        const float psi = wrap(state_.yaw + static_cast<float>(k) * kHalfPi);
        const Eigen::Vector2f front_dir(std::sin(psi), -std::cos(psi));
        front_acc_[k] = std::clamp(front_acc_[k] + w * front_dir.dot(cue_dir), -kFrontClamp, kFrontClamp);
    }

    int kbest = 0;
    for (int k = 1; k < 4; ++k)
        if (allowed[k] and front_acc_[k] > front_acc_[kbest]) kbest = k;
    if (kbest == 0)
        return false;   // current believed door direction still best-explained by the accumulated appearance

    apply_mode_rotation(kbest);
    const std::array<float, 4> old = front_acc_;    // re-baseline so the new current mode (old kbest) is index 0
    for (int m = 0; m < 4; ++m)
        front_acc_[m] = std::clamp(old[(m + kbest) % 4] - old[kbest], -kFrontClamp, kFrontClamp);
    return true;
}

// Posterior over the 4 front modes = softmax(front_acc_) with disallowed modes zeroed.
std::array<float, 4> RefrigeratorBelief::front_posterior() const
{
    const std::array<bool, 4> allowed = allowed_modes();
    std::array<float, 4> p{};
    float mx = -std::numeric_limits<float>::infinity();
    for (int k = 0; k < 4; ++k) if (allowed[k]) mx = std::max(mx, front_acc_[k]);
    float sum = 0.0f;
    for (int k = 0; k < 4; ++k) { p[k] = allowed[k] ? std::exp(front_acc_[k] - mx) : 0.0f; sum += p[k]; }
    if (sum <= 0.0f) { for (int k = 0; k < 4; ++k) p[k] = allowed[k] ? 0.5f : 0.0f; return p; }
    for (float& v : p) v /= sum;
    return p;
}

float RefrigeratorBelief::front_confidence() const
{
    const auto p = front_posterior();
    return *std::max_element(p.begin(), p.end());
}

int RefrigeratorBelief::front_mode() const
{
    const auto p = front_posterior();
    return static_cast<int>(std::distance(p.begin(), std::max_element(p.begin(), p.end())));
}

// REPORTED yaw variance: within-mode Σ(5,5) + Var_k[Δ_k] under the front posterior (Δ_k = wrapped k·π/2).
float RefrigeratorBelief::yaw_marginal_var() const
{
    constexpr float kHalfPi = 0.5f * static_cast<float>(M_PI);
    const auto wrap = [](float a) { return std::remainder(a, 2.0f * static_cast<float>(M_PI)); };
    const auto p = front_posterior();
    float mean = 0.0f, msq = 0.0f;
    for (int k = 0; k < 4; ++k)
    {
        const float d = wrap(static_cast<float>(k) * kHalfPi);   // ∈ (−π, π]
        mean += p[k] * d; msq += p[k] * d * d;
    }
    return Sigma_(5, 5) + std::max(0.0f, msq - mean * mean);
}

Eigen::Matrix<float, 6, 6> RefrigeratorBelief::covariance_reported() const
{
    Eigen::Matrix<float, 6, 6> S = Sigma_;
    S(5, 5) = yaw_marginal_var();
    return S;
}

// ─── "Is this really a fridge?" plausibility + soft singleton (mis-detection filter) ────────────────

// Continuous fridge-plausibility ∈ (0,1] = product of soft factors. See the header. No hard threshold: the
// mis-detection (elongated / short / poor-fit) is simply poorly explained by the fridge shape priors → low score.
float RefrigeratorBelief::fridge_plausibility(const RefrigeratorBeliefState& s, float fe,
                                              const RefrigeratorBeliefParams& p)
{
    const float sum_wh   = std::max(1e-4f, s.w + s.h);
    const float aspect   = std::abs(s.w - s.h) / sum_wh;                       // 0 square → 1 elongated
    const float as       = std::max(1e-4f, p.plaus_aspect_scale);
    const float aspect_ok = std::exp(-(aspect / as) * (aspect / as));

    const float ss       = std::max(1e-4f, p.plaus_size_scale);
    const float dw       = s.w - p.prior_footprint_m, dh = s.h - p.prior_footprint_m;
    const float size_ok  = std::exp(-(dw * dw + dh * dh) / (2.0f * ss * ss));   // footprint near 0.60×0.60

    const float hsoft    = std::max(1e-4f, p.plaus_height_soft);
    const float height_ok = 1.0f / (1.0f + std::exp(-(s.H - p.plaus_height_min) / hsoft));  // logistic: tall→1

    // The fit_ok term (absolute mean_energy vs a guessed FeRef) was DROPPED — FE varies strongly with view /
    // point-count, so an absolute reference is unreliable and was REJECTING REAL FRIDGES. The SHAPE factors are
    // the robust fridge-vs-not discriminators: on the FITTED box the footprint prior pins w,h≈0.60 (aspect_ok /
    // size_ok ≈1 for a genuine fridge), while an elongated / short mis-detection fails aspect and/or height.
    // `fe` stays in the signature (call sites unchanged) but is no longer used.
    (void) fe;
    return std::clamp(aspect_ok * size_ok * height_ok, 0.0f, 1.0f);
}

// Birth-candidate plausibility — HEIGHT ONLY. ★A partial mask (e.g. a front-only ZED view) is a VERTICAL
// face, so its 2-D footprint projects to a thin line → always "elongated"/tiny → aspect & size are MEANINGLESS
// at birth and would reject a real fridge (observed: refrigerator_dets=1 but births=0). The mask's z-EXTENT is
// reliable from any viewpoint: a fridge is TALL (~1.5 m+), the mis-detected cabinet is SHORT (~0.7 m). So gate
// birth on height alone; the footprint (square 0.6×0.6) is settled later by the fit + the footprint prior.
// Too few points ⇒ neutral 0.5 (don't punish a sparse first glimpse).
float RefrigeratorBelief::candidate_plausibility(const std::vector<Eigen::Vector3f>& pts,
                                                 const RefrigeratorBeliefParams& p)
{
    if (pts.size() < 12) return 0.5f;
    float zmin = std::numeric_limits<float>::max(), zmax = -std::numeric_limits<float>::max();
    for (const auto& q : pts) { zmin = std::min(zmin, q.z()); zmax = std::max(zmax, q.z()); }
    const float H     = std::max(0.10f, zmax - zmin);                 // rough height = mask z-range
    const float hsoft = std::max(1e-4f, p.plaus_height_soft);
    return 1.0f / (1.0f + std::exp(-(H - p.plaus_height_min) / hsoft));   // tall → ~1 (birth), short → ~0
}

// Soft singleton inhibition deltas (see the header). Bounded per-instance existence-logodds increments:
// stronger-plausibility fridges inhibit weaker ones; a fridge's own accumulated plausibility supports it.
std::vector<float> RefrigeratorBelief::singleton_existence_deltas(const std::vector<float>& plaus_evidence,
                                                                  const std::vector<float>& p_exists,
                                                                  float gain, float inhibition, float clamp)
{
    const std::size_t n = plaus_evidence.size();
    std::vector<float> out(n, 0.0f);
    const float cl = std::max(1e-3f, clamp);
    for (std::size_t i = 0; i < n; ++i)
    {
        const float shape_i = gain * std::tanh(plaus_evidence[i] / cl);        // + support / − decay by shape
        float penalty_i = 0.0f;
        for (std::size_t j = 0; j < n; ++j)
            if (j != i and plaus_evidence[j] > plaus_evidence[i])              // only STRONGER fridges inhibit
                penalty_i += (j < p_exists.size() ? p_exists[j] : 0.5f);
        out[i] = shape_i - inhibition * penalty_i;
    }
    return out;
}

// ─── Verification ────────────────────────────────────────────────────────────────────────────────

bool RefrigeratorBelief::self_test()
{
    std::mt19937 rng(12345);
    std::normal_distribution<float> noise(0.0f, 0.01f);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f), U01(0.0f, 1.0f);

    RefrigeratorBeliefParams P;
    const RefrigeratorBeliefState gt{0.20f, -0.30f, 1.75f, 0.70f, 0.60f, 0.30f};   // ground truth (w≠h)

    const float c = std::cos(gt.yaw), sn = std::sin(gt.yaw);
    const auto to_world = [&](float lx, float ly, float lz) -> Eigen::Vector3f
    { return {gt.cx + c * lx - sn * ly, gt.cy + sn * lx + c * ly, lz}; };

    std::vector<Eigen::Vector3f> pts;
    // Four vertical side faces (local x=±w/2 and y=±h/2), sampled over the full height.
    for (int i = 0; i < 1500; ++i)
    {
        const float u = U01(rng), z = U01(rng) * gt.H;
        const int face = i & 3;
        float lx, ly;
        if (face == 0) { lx =  0.5f * gt.w; ly = (u - 0.5f) * gt.h; }
        else if (face == 1) { lx = -0.5f * gt.w; ly = (u - 0.5f) * gt.h; }
        else if (face == 2) { ly =  0.5f * gt.h; lx = (u - 0.5f) * gt.w; }
        else                { ly = -0.5f * gt.h; lx = (u - 0.5f) * gt.w; }
        pts.push_back(to_world(lx + noise(rng), ly + noise(rng), z + noise(rng)));
    }
    // Top face (z = H).
    for (int i = 0; i < 400; ++i)
        pts.push_back(to_world(U(rng) * 0.5f * gt.w, U(rng) * 0.5f * gt.h, gt.H + noise(rng)));
    // Clutter (floor / off-box).
    for (int i = 0; i < 200; ++i)
        pts.push_back({gt.cx + U(rng) * 1.5f, gt.cy + U(rng) * 1.5f, U01(rng) * 0.05f});

    // Seed off-truth (centroid-ish), then run the recursive filter on the same frame several times.
    RefrigeratorBeliefState s0{0.0f, 0.0f, 1.4f, 0.5f, 0.5f, 0.15f};
    Eigen::Vector3f sum = Eigen::Vector3f::Zero(); float zmax = 0.0f;
    for (const auto& p : pts) { sum += p; zmax = std::max(zmax, p.z()); }
    s0.cx = sum.x() / pts.size(); s0.cy = sum.y() / pts.size(); s0.H = zmax;

    RefrigeratorBelief b(s0, P);
    RefrigeratorFrame frame; frame.points = pts;
    for (int it = 0; it < 40; ++it) b.update(frame);

    const auto& s = b.state();
    const auto wrap_pi = [](float a) { return std::remainder(a, static_cast<float>(M_PI)); };  // box: yaw mod π
    const float dyaw = std::abs(wrap_pi(s.yaw - gt.yaw));

    const bool ok =
        std::abs(s.cx - gt.cx) < 0.05f and
        std::abs(s.cy - gt.cy) < 0.05f and
        std::abs(s.H  - gt.H)  < 0.08f and
        std::abs(s.w  - gt.w)  < 0.10f and
        std::abs(s.h  - gt.h)  < 0.10f and
        dyaw < 0.10f;

    std::printf("RefrigeratorBelief::self_test [full-cloud] %s  cx=%.3f cy=%.3f H=%.3f w=%.3f h=%.3f yaw=%.3f (gt %.3f)\n",
                ok ? "PASS" : "FAIL", s.cx, s.cy, s.H, s.w, s.h, s.yaw, gt.yaw);

    // ── (a) WALL-FLUSH under a PARTIAL (front-only) view ─────────────────────────────────────────────
    // A fridge back-against a wall along y=0 (interior on +y). Its back face is at y=0, centre at cy=+h/2.
    // We observe ONLY the FRONT face (the back is occluded by the wall, the tall top is unseen from the
    // floor) — so depth `h` is unconstrained by points. The seed has the front face on the data but a too-deep
    // box whose back is buried 0.25 m INSIDE the wall. The wall-flush factor (+ footprint prior) must shrink
    // depth to ≈0.60 and pull the back face flush to y=0, WITHOUT any point ever touching the back face.
    const auto front_only_cloud = [&](const RefrigeratorBeliefState& g)
    {
        std::vector<Eigen::Vector3f> cl;
        const float cy = std::cos(g.yaw), sy = std::sin(g.yaw);
        const auto W = [&](float lx, float ly, float lz) -> Eigen::Vector3f
        { return {g.cx + cy * lx - sy * ly, g.cy + sy * lx + cy * ly, lz}; };
        for (int i = 0; i < 1600; ++i)                       // FRONT face only: local y = +h/2
        {
            const float lx = (U01(rng) - 0.5f) * g.w, z = U01(rng) * g.H;
            cl.push_back(W(lx + noise(rng), 0.5f * g.h + noise(rng), z + noise(rng)));
        }
        return cl;
    };

    const RefrigeratorBeliefState gtw{0.50f, 0.30f, 1.70f, 0.60f, 0.60f, 0.0f};   // back face flush at y=0
    RefrigeratorFrame fw;
    fw.points = front_only_cloud(gtw);
    fw.R.assign(fw.points.size(), P.sigma_base_m * P.sigma_base_m);
    fw.wall.ok = true; fw.wall.p = {0.0f, 0.0f}; fw.wall.n = {0.0f, 1.0f}; fw.wall.sigma_m = 0.02f;

    // Seed: front face correct (cy+h/2 = 0.60) but depth too big ⇒ back face 0.25 m behind the wall.
    RefrigeratorBeliefState sw{0.50f, 0.175f, 1.70f, 0.60f, 0.85f, 0.0f};
    RefrigeratorBelief bw(sw, P);
    bw.set_room_interior({0.5f, 5.0f});                      // interior on +y ⇒ back face is the −y (wall) face
    for (int it = 0; it < 60; ++it) bw.update(fw);
    const auto& w = bw.state();
    const float back_y   = bw.back_centre(w).y();            // should be ≈0 (flush)
    const bool ok_wall =
        std::abs(back_y)   < 0.04f and                       // back face within 4 cm of the wall
        std::abs(w.h - 0.60f) < 0.10f and                    // depth held at the standard footprint
        std::abs(w.w - 0.60f) < 0.10f;
    std::printf("RefrigeratorBelief::self_test [wall-flush partial] %s  cy=%.3f h=%.3f w=%.3f back_gap=%.3f λ=%.1f\n",
                ok_wall ? "PASS" : "FAIL", w.cy, w.h, w.w, back_y, bw.last_wall_lambda());

    // ── (b) FREE-STANDING: a far wall must NOT drag the fridge ────────────────────────────────────────
    // Same front-only cloud, but the nearest wall is 5 m away. The flush mixture weight → ~0, so accumulate_wall
    // is inert: the fridge must stay put (cy≈0.30 from the front face + footprint prior), NOT slide toward the
    // wall to make its back flush at y=−5.
    RefrigeratorFrame ff = fw;
    ff.wall.p = {0.0f, -5.0f};                               // wall 5 m behind (out of reach)
    RefrigeratorBeliefState sf{0.50f, 0.30f, 1.70f, 0.60f, 0.60f, 0.0f};
    RefrigeratorBelief bf(sf, P);
    bf.set_room_interior({0.5f, 5.0f});
    for (int it = 0; it < 60; ++it) bf.update(ff);
    const auto& fr = bf.state();
    const bool ok_free =
        std::abs(fr.cy - 0.30f) < 0.10f and                  // not dragged toward the far wall
        std::abs(fr.h - 0.60f) < 0.12f and
        bf.last_wall_lambda() < 1.0f;                        // flush precision essentially off
    std::printf("RefrigeratorBelief::self_test [free-standing]     %s  cy=%.3f h=%.3f λ=%.3f\n",
                ok_free ? "PASS" : "FAIL", fr.cy, fr.h, bf.last_wall_lambda());

    // ── (b′) WALL NO-CROSS: the fridge must not penetrate the wall behind it ────────────────────────────
    // Wall along y=0 (interior on +y, inward normal +y). A FRONT-only cloud whose front face sits at y=+0.20 with
    // the standard 0.60 m depth prior places the BACK face at y≈−0.40 — buried 0.40 m INSIDE the wall. That far
    // past, the flush mixture weight exp(−(gap/reach)²) has decayed to ~0, so the two-sided flush factor can no
    // longer pull it out: WITHOUT the one-sided no-cross term the box stays CROSSED (back≈−0.40). WITH it (precision
    // grows with penetration) the back is driven back to the interior side (gap ≥ ~0). As with the short-height
    // prior above, the test engages a FIRM precision to demonstrate the mechanism cleanly; the shipped default
    // (2000) is gentler (it strongly reduces, but does not fully null, a deep forced penetration).
    const RefrigeratorBeliefState gtc{0.50f, -0.10f, 1.70f, 0.60f, 0.60f, 0.0f};   // front face at y=+0.20
    RefrigeratorFrame fc;
    fc.points = front_only_cloud(gtc);
    fc.R.assign(fc.points.size(), P.sigma_base_m * P.sigma_base_m);
    fc.wall.ok = true; fc.wall.p = {0.0f, 0.0f}; fc.wall.n = {0.0f, 1.0f}; fc.wall.sigma_m = 0.02f;
    const RefrigeratorBeliefState sc{0.50f, -0.10f, 1.70f, 0.60f, 0.60f, 0.0f};    // seed already crossing (back at −0.40)

    RefrigeratorBeliefParams Poff = P; Poff.wall_no_cross_precision = 0.0f;         // constraint OFF (baseline)
    RefrigeratorBelief boff(sc, Poff); boff.set_room_interior({0.5f, 5.0f});
    for (int it = 0; it < 80; ++it) boff.update(fc);
    const float back_off = boff.back_centre(boff.state()).y();                      // expect strongly negative (crossed)

    RefrigeratorBeliefParams Pon = P; Pon.wall_no_cross_precision = 100000.0f;      // firm, to demonstrate the null
    RefrigeratorBelief bon(sc, Pon); bon.set_room_interior({0.5f, 5.0f});
    for (int it = 0; it < 80; ++it) bon.update(fc);
    const float back_on = bon.back_centre(bon.state()).y();                         // expect ≥ ~0 (does not cross)

    // A fridge genuinely IN FRONT of the wall (back at y≈+0.30) must be UNAFFECTED (no-cross residual = 0 there).
    const RefrigeratorBeliefState gtf{0.50f, 0.60f, 1.70f, 0.60f, 0.60f, 0.0f};     // front face at y=+0.90, back at +0.30
    RefrigeratorFrame fcf; fcf.points = front_only_cloud(gtf);
    fcf.R.assign(fcf.points.size(), P.sigma_base_m * P.sigma_base_m);
    fcf.wall.ok = true; fcf.wall.p = {0.0f, 0.0f}; fcf.wall.n = {0.0f, 1.0f}; fcf.wall.sigma_m = 0.02f;
    RefrigeratorBelief bfront(gtf, Pon); bfront.set_room_interior({0.5f, 5.0f});
    for (int it = 0; it < 80; ++it) bfront.update(fcf);
    const float back_front = bfront.back_centre(bfront.state()).y();                // expect ≈ +0.30 (untouched)

    const bool ok_nocross =
        back_off   < -0.15f and      // WITHOUT the constraint: the back penetrates the wall
        back_on    > -0.05f and      // WITH it: the back does not cross (gap ≥ ~0)
        back_front >  0.15f;         // a genuinely-in-front fridge is unaffected
    std::printf("RefrigeratorBelief::self_test [wall no-cross]      %s  back_off=%.3f back_on=%.3f back_front=%.3f\n",
                ok_nocross ? "PASS" : "FAIL", back_off, back_on, back_front);

    // ── (c) FRONT (door) yaw resolver ──────────────────────────────────────────────────────────────
    // A SQUARE fridge (w==h ⇒ all 4 door modes allowed) initially believed with its door facing +X (yaw=0 ⇒
    // door local −Y faces bearing atan2(−cos0,sin0)=atan2(−1,0)=−π/2). Feed a run of confident cues that the
    // door actually faces +Y (bearing +π/2). The resolver must rotate the belief so its door normal ends up
    // pointing at +π/2, and a converged belief must NOT be flipped by one low-confidence contradictory cue.
    constexpr float kHalfPi = 0.5f * static_cast<float>(M_PI);
    const auto wrap2pi = [](float a) { return std::remainder(a, 2.0f * static_cast<float>(M_PI)); };
    const auto door_bearing = [&](const RefrigeratorBelief& b)   // room bearing the believed door (local −Y) faces
    { const float y = b.state().yaw; return std::atan2(-std::cos(y), std::sin(y)); };

    RefrigeratorBeliefState sq{0.0f, 0.0f, 1.70f, 0.60f, 0.60f, 0.0f};   // square footprint, door believed at −π/2
    RefrigeratorBelief bd(sq, P);
    const float sigma_yaw_undecided = std::sqrt(bd.yaw_marginal_var());   // (c) wide when the accumulator is ~0
    const FrontCue cue_pos_y{ static_cast<float>(kHalfPi), 0.9f };        // door faces +Y, high confidence
    for (int it = 0; it < 30; ++it) bd.resolve_front(cue_pos_y, 1.0f);
    const float dyaw_front = std::abs(wrap2pi(door_bearing(bd) - kHalfPi));
    const float sigma_yaw_settled = std::sqrt(bd.yaw_marginal_var());     // (c) tightens once evidence accrues

    // A single LOW-confidence cue for the OPPOSITE door direction must not flip the now-settled belief.
    const float yaw_before_bad = bd.state().yaw;
    const FrontCue cue_bad{ static_cast<float>(-kHalfPi), 0.05f };
    const bool flipped_on_bad = bd.resolve_front(cue_bad, 0.1f);
    const bool ok_front =
        dyaw_front < 0.05f and                        // door now faces the observed bearing (+Y)
        not flipped_on_bad and                        // a weak contradictory cue does not flip a settled door
        std::abs(wrap2pi(bd.state().yaw - yaw_before_bad)) < 1e-4f and
        sigma_yaw_undecided > 1.0f and                // (c) undecided ⇒ honest wide σ_yaw (>~57°): the discrete
                                                      //     door-mode entropy dominates the reported variance
        sigma_yaw_settled   < 0.7f and                // (c) resolved ⇒ discrete entropy collapses to ≈Σ(5,5) only
        sigma_yaw_settled   < 0.5f * sigma_yaw_undecided;
    std::printf("RefrigeratorBelief::self_test [front-resolver]     %s  door_bearing=%.3f (gt %.3f) σyaw %.2f→%.2f p=%.2f\n",
                ok_front ? "PASS" : "FAIL", door_bearing(bd), kHalfPi,
                sigma_yaw_undecided, sigma_yaw_settled, bd.front_confidence());

    // ── (d) FRIDGE PLAUSIBILITY: proper fridge scores ~1, elongated/short scores ~0 ─────────────────────
    const RefrigeratorBeliefState proper{0.0f, 0.0f, 1.75f, 0.60f, 0.60f, 0.0f};   // square + tall
    const RefrigeratorBeliefState wrongish{0.0f, 0.0f, 0.70f, 1.20f, 0.40f, 0.0f}; // elongated + short (the mis-detection)
    const float pl_proper = fridge_plausibility(proper,   P.plaus_fe_ref, P);       // low FE ⇒ fit_ok = 1
    const float pl_wrong  = fridge_plausibility(wrongish, P.plaus_fe_ref, P);
    const bool ok_plaus = pl_proper > 0.8f and pl_wrong < 0.05f;
    std::printf("RefrigeratorBelief::self_test [plausibility]       %s  proper=%.3f wrongish=%.3f\n",
                ok_plaus ? "PASS" : "FAIL", pl_proper, pl_wrong);

    // ── (e) plaus_evidence sign convergence (bounded accumulator) + single-bad-frame robustness ─────────
    // Mirror the fitter's accumulator: evidence += (plausibility − 0.5), clamped to ±PlausClamp (here 8).
    constexpr float kClamp = 8.0f;
    const auto accumulate = [&](const RefrigeratorBeliefState& st, int frames, float ev0) {
        float ev = ev0;
        const float pl = fridge_plausibility(st, P.plaus_fe_ref, P);
        for (int i = 0; i < frames; ++i) ev = std::clamp(ev + (pl - 0.5f), -kClamp, kClamp);
        return ev;
    };
    const float ev_proper = accumulate(proper,   40, 0.0f);   // proper frames → strongly positive
    const float ev_wrong  = accumulate(wrongish, 40, 0.0f);   // short/elongated frames → strongly negative
    // One bad frame must NOT flip a settled-positive fridge (bounded memory recants only over MANY frames).
    const float ev_settled = accumulate(proper, 40, 0.0f);
    const float pl_wrong_frame = fridge_plausibility(wrongish, P.plaus_fe_ref, P);
    const float ev_after_bad   = std::clamp(ev_settled + (pl_wrong_frame - 0.5f), -kClamp, kClamp);
    const bool ok_evidence = ev_proper > 2.0f and ev_wrong < -2.0f and ev_after_bad > 0.0f;
    std::printf("RefrigeratorBelief::self_test [plaus-evidence]     %s  ev_proper=%.2f ev_wrong=%.2f ev_after_bad=%.2f\n",
                ok_evidence ? "PASS" : "FAIL", ev_proper, ev_wrong, ev_after_bad);

    // ── (f) SOFT SINGLETON: a strong (proper) + a weak (elongated) fridge → weak decays, strong survives ─
    // Two instances: i=0 strong (plaus_evidence>0), i=1 weak (plaus_evidence<0), both currently P(exists)≈0.9.
    std::vector<float> pe{ 6.0f, -3.0f }, px{ 0.9f, 0.9f };
    float L0 = 0.0f, L1 = 0.0f;   // existence log-odds, folded each cycle by the singleton+shape deltas
    for (int c = 0; c < 40; ++c)
    {
        const auto d = singleton_existence_deltas(pe, px, /*gain=*/1.5f, /*inhibition=*/1.0f, kClamp);
        L0 = std::clamp(L0 + d[0], -4.0f, 4.0f);
        L1 = std::clamp(L1 + d[1], -4.0f, 4.0f);
        px[0] = 1.0f / (1.0f + std::exp(-L0)); px[1] = 1.0f / (1.0f + std::exp(-L1));
    }
    // removal boundary (existence_removal_prob≈0.12 ⇒ L < log(0.12/0.88) ≈ −1.99).
    const float remove_L = std::log(0.12f / 0.88f);
    const bool ok_singleton = L1 < remove_L and L0 > 0.0f and pe[0] > 0.0f;
    std::printf("RefrigeratorBelief::self_test [singleton]          %s  L_strong=%.2f L_weak=%.2f (removeL=%.2f)\n",
                ok_singleton ? "PASS" : "FAIL", L0, L1, remove_L);

    // ── (g) SHORT-HEIGHT PRIOR: fitting a 70 cm cloud yields a HIGHER FE than a 1.7 m cloud (same footprint) ─
    // The one-sided height prior pulls H up toward plaus_height_min, so a genuinely-short cloud fits worse (its
    // observed top face becomes interior → misfit). A clean top face resists a WEAK prior, so this test engages
    // the prior firmly (Ph.plaus_height_prior_gain well past the point where it lifts H to the plausible range)
    // to demonstrate the mechanism; the shipped DEFAULT gain is gentler (firming, not overriding) — the primary
    // short-fridge rejector is the plausibility height_ok term, which reads the fitted H directly.
    RefrigeratorBeliefParams Ph = P;
    Ph.plaus_height_prior_gain = 20000.0f;   // firm enough to lift a short cloud's H to the plausible range
    std::uniform_real_distribution<float> Uh(-0.5f, 0.5f);
    const auto box_cloud = [&](float Hgt) {
        std::vector<Eigen::Vector3f> cl;
        for (int i = 0; i < 1600; ++i)
        {
            const float u = U01(rng), z = U01(rng) * Hgt; const int face = i & 3;
            float lx, ly;
            if (face == 0) { lx =  0.30f; ly = (u - 0.5f) * 0.60f; }
            else if (face == 1) { lx = -0.30f; ly = (u - 0.5f) * 0.60f; }
            else if (face == 2) { ly =  0.30f; lx = (u - 0.5f) * 0.60f; }
            else                { ly = -0.30f; lx = (u - 0.5f) * 0.60f; }
            cl.push_back({lx + noise(rng), ly + noise(rng), z + noise(rng)});
        }
        for (int i = 0; i < 400; ++i) cl.push_back({Uh(rng) * 0.60f, Uh(rng) * 0.60f, Hgt + noise(rng)});  // top
        return cl;
    };
    const auto fit_fe = [&](float Hgt) {
        RefrigeratorBeliefState s0{0.0f, 0.0f, Hgt, 0.60f, 0.60f, 0.0f};
        RefrigeratorBelief bb(s0, Ph);
        RefrigeratorFrame fr; fr.points = box_cloud(Hgt);
        fr.R.assign(fr.points.size(), Ph.sigma_base_m * Ph.sigma_base_m);
        for (int it = 0; it < 30; ++it) bb.update(fr);
        return bb.mean_energy(fr.points, bb.state(), Ph.sigma_base_m * Ph.sigma_base_m);
    };
    const float fe_short = fit_fe(0.70f);   // short: prior resists ⇒ misfit ⇒ higher FE
    const float fe_tall  = fit_fe(1.70f);   // tall: prior inactive ⇒ clean fit ⇒ lower FE
    const bool ok_height = fe_short > fe_tall;
    std::printf("RefrigeratorBelief::self_test [short-height prior] %s  FE(0.70m)=%.3f > FE(1.70m)=%.3f\n",
                ok_height ? "PASS" : "FAIL", fe_short, fe_tall);

    return ok and ok_wall and ok_free and ok_nocross and ok_front and
           ok_plaus and ok_evidence and ok_singleton and ok_height;
}

}  // namespace rc
