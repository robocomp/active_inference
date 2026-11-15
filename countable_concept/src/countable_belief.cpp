/*
 * countable_belief.cpp  —  AI2 bed belief implementation (single floor-anchored box)
 *
 * The bed-specific model hooks behind CountableBelief: ONE box SDF (floor-anchored, z∈[0,H]),
 * the soft [box, clutter] mixture responsibilities + the clutter-inclusive free energy, the finite-difference
 * (slope-clamped) Jacobian, the optional YOLO-independent LiDAR range factor, and self_test(). All Bayesian
 * bookkeeping (predict / GN-MAP / Woodbury) lives in common/ai_belief/recursive_laplace.h; this file feeds it.
 *
 * Deliberately minimal vs table_belief: no legs, no top slab, no w↔h symmetry fold / orientation-mode
 * accumulator, no footprint-moment / coverage / free-space / depth-tilt machinery.
 */

#include "countable_belief.h"

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
float CountableBelief::sdf_box(const Eigen::Vector3f& p, const CountableBeliefState& s) const
{
    const float c = std::cos(-s.yaw), sn = std::sin(-s.yaw);
    const float px = p.x() - s.cx, py = p.y() - s.cy;
    const float lx = px * c - py * sn;
    const float ly = px * sn + py * c;
    const float half_H = 0.5f * s.H;
    return box_sdf(std::abs(lx) - 0.5f * s.w, std::abs(ly) - 0.5f * s.h, std::abs(p.z() - half_H) - half_H);
}

// The headboard panel: local x ∈ [−w/2, w/2], y ∈ [h/2 − t, h/2] (the back, +Y), z ∈ [0, headboard_h_m].
float CountableBelief::sdf_headboard(const Eigen::Vector3f& p, const CountableBeliefState& s) const
{
    if (params_.headboard_h_m <= 0.0f) return std::numeric_limits<float>::infinity();
    const float c = std::cos(-s.yaw), sn = std::sin(-s.yaw);
    const float px = p.x() - s.cx, py = p.y() - s.cy;
    const float lx = px * c - py * sn;
    const float ly = px * sn + py * c;
    const float t  = std::min(params_.headboard_t_m, s.h);
    const float yc = 0.5f * s.h - 0.5f * t;
    const float hz = 0.5f * params_.headboard_h_m;
    return box_sdf(std::abs(lx) - 0.5f * s.w, std::abs(ly - yc) - 0.5f * t, std::abs(p.z() - hz) - hz);
}

float CountableBelief::sdf_prim(const Eigen::Vector3f& p, const CountableBeliefState& s, int) const
{
    return sdf_compound(p, s);   // prim == 0: box ∪ headboard, ONE primitive (the engine's mixture is unchanged)
}

// ─── Mixture responsibilities ────────────────────────────────────────────────────────────────────

// Un-normalised mixture components u[0]=box, u[1]=clutter, u[2]=wall, and their sum = the marginal likelihood
// numerator p(point|model) ∝ Σ_k π_k N(d_k; 0, R). The FREE ENERGY reads −log(sum); the clutter term keeps a
// far/misfit point at a large real penalty (−log clutter likelihood) rather than silently zeroing the energy.
//
// The WALL component is the room's already-inferred surface entering as a COMPETING EXPLANATION. Its distance
// is to the vertical plane through the nearest room-polygon edge (the wall spans all z, so only the horizontal
// offset matters), widened by the polygon/localization slack. Consequences, both automatic:
//   • a point on the wall gets u[2] ≫ u[0] ⇒ its BOX responsibility collapses ⇒ zero GN pull on the geometry,
//     so a wall mislabel can no longer drag the fitted box;
//   • the marginal likelihood is already high there WITHOUT a box ⇒ box_log_evidence_gain ≈ 0 ⇒ the bed
//     hypothesis earns no birth/existence evidence from wall points.
// A real bed is unaffected: its VISIBLE faces stand a depth off the wall plane, so their points keep u[0].
// Only the back face (which the camera never sees) is ambiguous, which is honest.
float CountableBelief::mixture_unnormalized(const Eigen::Vector3f& p, const CountableBeliefState& s,
                                               float R, std::array<float, 3>& u) const
{
    const float eps    = std::clamp(params_.clutter_frac, 0.0f, 0.99f);
    const float pw     = wall_component_frac();
    const float inv2R  = 0.5f / std::max(1e-9f, R);
    const float d      = sdf_compound(p, s);
    u[0] = (1.0f - eps - pw) * std::exp(-d * d * inv2R);
    const float cs = params_.clutter_scale_m;
    u[1] = eps * std::exp(-cs * cs * inv2R);
    u[2] = wall_component(p, R, pw);
    return u[0] + u[1] + u[2];
}

// Prior weight actually available to the wall component this frame: 0 unless a room polygon was pushed AND the
// feature is on, and never so large that box+clutter lose their mass.
float CountableBelief::wall_component_frac() const
{
    if (not point_wall_.ok) return 0.0f;
    const float eps = std::clamp(params_.clutter_frac, 0.0f, 0.99f);
    return std::clamp(params_.wall_explain_frac, 0.0f, std::max(0.0f, 0.99f - eps));
}

// π_wall · N(distance-to-wall-plane; 0, R + σ_wall²), ONE-SIDED. Horizontal offset only — a wall is a vertical
// plane. dw > 0 is the room interior: the wall explains a point less and less as it stands further in front.
// dw ≤ 0 is ON or BEYOND the wall, i.e. inside the masonry or outside the room altogether — there the exterior
// explains the point FULLY (flat at π_wall, no decay), because "beyond the room envelope" is a place the room
// model already accounts for and no bed of ours can occupy. That one-sidedness is what stops a detection
// landing outside the layout from ever looking like unexplained evidence for a new bed.
float CountableBelief::wall_component(const Eigen::Vector3f& p, float R, float pw) const
{
    if (pw <= 0.0f) return 0.0f;
    const float sw  = params_.wall_explain_sigma_m + point_wall_.sigma_m;
    const float Rw  = std::max(1e-9f, R + sw * sw);
    const float dw  = (p.head<2>() - point_wall_.p).dot(point_wall_.n);   // signed: >0 into the room
    const float din = std::max(0.0f, dw);                                 // outside/on the wall ⇒ fully explained
    return pw * std::exp(-din * din * 0.5f / Rw);
}

std::array<float, 3> CountableBelief::responsibilities(const Eigen::Vector3f& p,
                                                          const CountableBeliefState& s, float R) const
{
    std::array<float, 3> u{};
    const float sum = mixture_unnormalized(p, s, R, u);
    if (sum <= 0.0f) { u = {0.0f, 1.0f, 0.0f}; return u; }
    u[0] /= sum; u[1] /= sum; u[2] /= sum;
    return u;
}

float CountableBelief::mean_energy(const std::vector<Eigen::Vector3f>& pts,
                                      const CountableBeliefState& s, float R) const
{
    if (pts.empty()) return 0.0f;
    double acc = 0.0, wacc = 0.0;
    std::array<float, 3> u{};
    for (const auto& p : pts)
    {
        const float sum = mixture_unnormalized(p, s, R, u);
        acc  += -std::log(std::max(1e-30f, sum));
        wacc += (sum > 0.0f) ? (u[2] / sum) : 0.0;
    }
    dbg_wall_resp_ = static_cast<float>(wacc / static_cast<double>(pts.size()));
    return static_cast<float>(acc / static_cast<double>(pts.size()));
}

// The SAME mixture with the box component deleted: how well the scene explains these points with no bed in
// it. Independent of the box state by construction, hence no State argument. Paired with mean_energy() this
// gives the log Bayes factor for "a bed is here" (box_log_evidence_gain).
float CountableBelief::mean_energy_no_box(const std::vector<Eigen::Vector3f>& pts, float R) const
{
    if (pts.empty()) return 0.0f;
    const float eps = std::clamp(params_.clutter_frac, 0.0f, 0.99f);
    const float pw  = wall_component_frac();
    const float cs  = params_.clutter_scale_m;
    // Renormalise the surviving components so the comparison is between two proper models, not between a
    // 3-component mixture and a deficient 2-component one (that alone would fake a gain for every point).
    const float z   = std::max(1e-6f, eps + pw);
    const float inv2R = 0.5f / std::max(1e-9f, R);
    const float uc  = (eps / z) * std::exp(-cs * cs * inv2R);
    double acc = 0.0;
    for (const auto& p : pts)
        acc += -std::log(std::max(1e-30f, uc + wall_component(p, R, pw / z)));
    return static_cast<float>(acc / static_cast<double>(pts.size()));
}

// ─── Jacobian (central finite difference, slope-clamped) ───────────────────────────────────────────

Eigen::Matrix<float, 6, 1> CountableBelief::sdf_jacobian(const Eigen::Vector3f& p,
                                                            const CountableBeliefState& s, int prim) const
{
    Eigen::Matrix<float, 6, 1> J;
    const Eigen::Matrix<float, 6, 1> base = s.vec();
    const float e = params_.fd_eps;
    const float g = params_.jac_slope_clamp;
    for (int j = 0; j < 6; ++j)
    {
        Eigen::Matrix<float, 6, 1> vp = base, vm = base;
        vp(j) += e; vm(j) -= e;
        const float slope = (sdf_prim(p, CountableBeliefState::from_vec(vp), prim) -
                             sdf_prim(p, CountableBeliefState::from_vec(vm), prim)) / (2.0f * e);
        // CLAMP the slope: a non-smooth SDF seam can spike a per-DOF slope and blow up the information.
        J(j) = std::clamp(slope, -g, g);
    }
    return J;
}

// ─── Constraints / canonical form ──────────────────────────────────────────────────────────────────

// Per-GN-iteration bounds: floor each extent so the box can't invert or collapse, and wrap yaw. Physical
// floors (w,h,H ≥ 0.10 m), not tuned gates — a degenerate zero/negative extent has no gradient.
void CountableBelief::apply_constraints(CountableBeliefState& s) const
{
    s.w = std::max(s.w, 0.10f);
    s.h = std::max(s.h, 0.10f);
    s.H = std::max(s.H, 0.10f);
    s.yaw = std::remainder(s.yaw, 2.0f * static_cast<float>(M_PI));   // wrap to (−π, π]
}

// ─── Engine hooks: prior cov · process noise (F = I, static) · common-mode ──────────────────────────

Eigen::Matrix<float, 6, 1> CountableBelief::prior_cov_diag() const
{
    const float pp = params_.prior_pos_std       * params_.prior_pos_std;
    const float hh = params_.prior_height_std    * params_.prior_height_std;     // H: BROAD (data-driven)
    const float wd = params_.prior_footprint_std * params_.prior_footprint_std;  // w,h(depth): TIGHT
    const float yy = params_.prior_yaw_std       * params_.prior_yaw_std;
    return (Eigen::Matrix<float, 6, 1>() << pp, pp, hh, wd, wd, yy).finished();   // [cx,cy,H,w,h,yaw]
}

Eigen::Matrix<float, 6, 1> CountableBelief::process_noise_diag() const
{
    const float qm = params_.process_std_m   * params_.process_std_m;
    const float qy = params_.process_std_yaw * params_.process_std_yaw;
    const Eigen::Matrix<float, 6, 1> q0 =
        (Eigen::Matrix<float, 6, 1>() << qm, qm, qm, qm, qm, qy).finished();
    // Q = exp(ω) once volatility is inferred. Until the first measured cycle ω is unset and we return the
    // configured constant, so a belief that never updates behaves exactly as before.
    if (not params_.volatility_infer or not omega_init_)
        return q0;
    Eigen::Matrix<float, 6, 1> q;
    for (int i = 0; i < 6; ++i) q(i) = std::exp(omega_(i));
    return q;
}

// ─── Inferred volatility: the HISTORY stage (MODEL_HISTORY.md §3) ────────────────────────────────
//
// Generative statement: between frames the state takes a step θ_t = θ_{t−1} + w, w ~ N(0, Q), and Q is not
// known — it is a property of THIS object (a bed bolted to a wall vs a chair someone keeps moving). Infer
// ω = log Q by gradient ascent on the log-evidence of the observed step, regularised by a hyperprior:
//
//     ω_i ← ω_i + lr · [ ½·( δθ_i²·e^{−ω_i} − 1 )  −  (ω_i − ω₀_i)/σ_ω² ]
//
// The bracket is exactly d/dω log N(δθ_i; 0, e^{ω_i}) plus the hyperprior pull. Consistent observation makes
// δθ² small relative to e^ω, the first term goes negative, ω falls, and BOTH failure modes of a constant Q
// are repaired at once: retention τ = Σ/e^ω grows, and the precision floor (e^ω/i)^¼ drops. A bed that is
// actually shoved produces a large δθ², the term flips positive, ω climbs and the model adapts within
// seconds — stiffness is never a lock, and the (ω−ω₀)/σ_ω² term guarantees recoverability structurally
// rather than by a clamp.
//
// ★DEVIATION FROM THE DOC, deliberate: MODEL_HISTORY.md §3 writes the surprise as δθᵀΣ⁻¹δθ (dimensionless)
// while multiplying it by e^{−ω} (units 1/θ²). That is dimensionally inconsistent — the quantity whose
// variance is being inferred is the STEP itself, so the per-DOF squared step δθ_i² (m², rad²) is what pairs
// with e^{−ω_i}. With the group reduced to one DOF, k/2 becomes the ½ above. Same behaviour, correct units.
void CountableBelief::update_volatility(const Eigen::Matrix<float, 6, 1>& dtheta)
{
    if (not params_.volatility_infer)
        return;
    if (not omega_init_)   // seed at the configured constant: experience starts from the species prior
    {
        const float qm = params_.process_std_m   * params_.process_std_m;
        const float qy = params_.process_std_yaw * params_.process_std_yaw;
        for (int i = 0; i < 6; ++i) omega0_(i) = std::log(std::max(1e-12f, (i == 5) ? qy : qm));
        omega_ = omega0_;
        vol_n_ = 0.0f;
        for (int i = 0; i < 6; ++i) vol_s_(i) = 0.0f;
        omega_init_ = true;
    }
    // ★CONJUGATE running estimate, NOT a gradient step. The gradient form
    //       ω ← ω + lr·[½(δθ²·e^{−ω} − 1) − (ω−ω₀)/σ²]
    // is UNBOUNDED ABOVE: e^{−ω} ≈ 40000 at the configured Q, so a single 0.5 m correction gives a gradient of
    // 5000 and one lr=0.02 step moves ω by +100 — past the ~88 where float exp() overflows. Live consequence
    // (etc/ai2_log.csv 20:27): a birth at 0.48 m produced ω_yaw = 300.6, so Q = exp(300) = inf, Σ went inf then
    // 0 through the inverse, the Mahalanobis association gate collapsed, NO mask could associate ever again
    // (npts = 0 in 943 of 946 cycles), and the instance froze forever at its garbage birth state
    // (w = 0.126, yaw = −102°) — a self-sealing zombie. See [[inferred-volatility-history-stage]].
    //
    // The variance of a Gaussian step has a conjugate posterior whose mean is a WEIGHTED AVERAGE of the prior
    // and the observed squared steps, so Q can never leave [min, max] of what has actually been seen:
    //
    //     n ← λ·n + 1 ,  S ← λ·S + δθ² ,  Q = (n₀·Q₀ + S) / (n₀ + n)
    //
    // λ < 1 gives a finite memory, which is what lets ω RISE again when the object genuinely moves; n₀ is the
    // hyperprior's strength in pseudo-observations, doing the job the (ω−ω₀)/σ² pull-back did. No learning
    // rate, no exponent to overflow, and recoverability is structural exactly as before.
    constexpr float kLambda = 0.995f;   // ~200-observation memory: long enough to earn stiffness, short enough to recant
    const float n0 = 1.0f / std::max(1e-3f, params_.volatility_sigma * params_.volatility_sigma);
    vol_n_ = kLambda * vol_n_ + 1.0f;
    for (int i = 0; i < 6; ++i)
    {
        const float d2 = dtheta(i) * dtheta(i);
        if (not std::isfinite(d2)) continue;                       // never let a bad frame poison the estimate
        vol_s_(i) = kLambda * vol_s_(i) + d2;
        const float q0 = std::exp(omega0_(i));
        const float q  = (n0 * q0 + vol_s_(i)) / (n0 + vol_n_);    // bounded by construction
        omega_(i) = std::clamp(std::log(std::max(1e-12f, q)), params_.volatility_omega_min, 4.0f);
    }
}

// τ = Σ_ii / Q_ii frames: how many predict-only frames double this DOF's variance — "how long it remembers".
Eigen::Matrix<float, 6, 1> CountableBelief::retention_frames() const
{
    const Eigen::Matrix<float, 6, 1> q = process_noise_diag();
    Eigen::Matrix<float, 6, 1> tau;
    for (int i = 0; i < 6; ++i) tau(i) = (q(i) > 1e-20f) ? Sigma_(i, i) / q(i) : 0.0f;
    return tau;
}

// Inverse of the per-frame common-mode covariance Σc (diagonal): position (cx,cy) = config floor + pose-
// chain/range/motion; size (H,w,h) = config std + range/motion; yaw = config std + range/grazing/motion.
// Marginalising this SHARED error (Woodbury in the engine) makes the frame's information SATURATE at Σc.
Eigen::Matrix<float, 6, 1> CountableBelief::common_mode_inv_diag(const CountableFrame& frame) const
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
Eigen::Vector2f CountableBelief::back_centre(const CountableBeliefState& s) const
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
// a continuous covariance, not a proximity gate. Fades to ~0 for a genuine mid-room bed (depth stays wide).
// Two-sided fraction for one footprint axis: an extent is IDENTIFIABLE only when the cloud spans it.
// Lifted out of accumulate_extra 2026-08-13 so the WALL factor can ask the same question — asking it twice,
// in two places, is how the two drift apart.
float CountableBelief::two_sided_along(const CountableBeliefState& s, const CountableFrame& f, bool along_x) const
{
            if (params_.depth_unobs_precision <= 0.0f or f.points.size() <= 8)
                return 1.0f;                       // feature off / too few points ⇒ claim nothing, add no boost
            return two_sided_fraction(s, f, along_x);
}

float CountableBelief::two_sided_fraction(const CountableBeliefState& s, const CountableFrame& f, bool along_x) const
{
            const float c = std::cos(-s.yaw), sn = std::sin(-s.yaw);
            const float delta = params_.depth_obs_band_m;
            const float half_cross = 0.5f * (along_x ? s.h : s.w);   // the OTHER axis: the spanning core
            int nplus = 0, nminus = 0;
            for (const auto& q : f.points)
            {
                const float px = q.x() - s.cx, py = q.y() - s.cy;
                const float lx = px * c - py * sn, ly = px * sn + py * c;
                const float along = along_x ? lx : ly;               // the axis being tested
                const float cross = along_x ? ly : lx;               // must lie within the spanning core
                if (std::abs(cross) > half_cross + delta) continue;  // drop clutter off the side of this face pair
                if (q.z() < 0.0f or q.z() > s.H + delta) continue;
                // Clearly on one side or the other by a FIXED margin δ (NOT keyed to the current half-extent). A
                // point straddling the centre counts as NEITHER, so a COLLAPSED slab cannot masquerade as
                // "both faces seen" and switch the boost off — that circularity re-collapsed the extent.
                if (along >  delta)      ++nplus;
                else if (along < -delta) ++nminus;
            }
            const int tot = nplus + nminus;
            return (tot > 0) ? (2.0f * std::min(nplus, nminus) / static_cast<float>(tot)) : 0.0f;
        }

// flush_weight MARGINALISED over whether the data can discriminate {flush, free-standing} at all — see
// hood_concept, where this was found. With the depth axis unobserved the gap cannot tell them apart, so the
// posterior falls back to the CLASS prior π_flush.
//
// ★A COUNTABLE'S π_flush IS 0, DELIBERATELY. Its own manifest says "the mixture weight decays to 0 for a
// genuine mid-room bed" — a bed is COMMONLY against a wall, not DEFINITIONALLY, and asserting
// otherwise would pin a free-standing one to the nearest wall. Only a class that is wall-mounted by
// definition (a range hood) declares 1.0. Default 0 ⇒ this is exactly flush_weight() and nothing changes.
float CountableBelief::flush_posterior(const CountableBeliefState& s, const CountableFrame& f) const
{
    if (not f.wall.ok) return 0.0f;
    const float two_sided = two_sided_along(s, f, false);
    return two_sided * flush_weight(s, f)
         + (1.0f - two_sided) * std::clamp(params_.wall_flush_prior, 0.0f, 1.0f);
}

float CountableBelief::flush_weight(const CountableBeliefState& s, const CountableFrame& f) const
{
    if (not f.wall.ok) return 0.0f;
    // ★THE GAP IS UNCERTAIN TOO (2026-11-14). exp(−(gap/reach)²) treats the gap as known exactly, so a young
    // belief whose back face sits 0.5 m off the wall — because its POSE is still uncertain by that much — got
    // weight ~0 and was never drawn to the wall it stands against. The flush hypothesis predicts gap = 0 up to
    // the reach; what it is compared with is the belief's gap, whose variance is J Σ Jᵀ. Adding the two is the
    // marginal likelihood of the flush component: wide while the pose is uncertain, the bare reach once settled.
    const auto gap_of = [&](const CountableBeliefState& st) { return (back_centre(st) - f.wall.p).dot(f.wall.n); };
    const float gap = gap_of(s);
    Eigen::Matrix<float, 6, 1> J;
    const Eigen::Matrix<float, 6, 1> v = s.vec();
    for (int k = 0; k < 6; ++k)
    {
        Eigen::Matrix<float, 6, 1> vp = v, vm = v;
        vp(k) += params_.fd_eps; vm(k) -= params_.fd_eps;
        J(k) = (gap_of(CountableBeliefState::from_vec(vp)) - gap_of(CountableBeliefState::from_vec(vm))) / (2.0f * params_.fd_eps);
    }
    const float var_gap = std::max(0.0f, J.dot(Sigma_ * J));
    const float reach   = std::max(1e-3f, params_.wall_reach_m);
    return std::exp(-gap * gap / (reach * reach + 2.0f * var_gap));
}

// Drive the BACK-face centre onto the nearest wall line (gap→0) and the back face PARALLEL to the wall
// (width-axis · wall-normal → 0). Both precisions are scaled by the flush mixture weight, so an island
// feels ~nothing. Finite-difference Jacobians (slope-clamped), mirroring cabinet_concept::accumulate_wall.
void CountableBelief::accumulate_wall(const CountableBeliefState& s, const CountableFrame& f,
                                         Eigen::Matrix<float, 6, 6>& Id, Eigen::Matrix<float, 6, 1>& bd) const
{
    dbg_wall_gap_ = 0.0f; dbg_wall_lambda_ = 0.0f;
    if (not f.wall.ok or params_.wall_precision <= 0.0f) return;

    const auto gap_of = [&](const CountableBeliefState& st)
    { return (back_centre(st) - f.wall.p).dot(f.wall.n); };
    const auto par_of = [&](const CountableBeliefState& st)   // width axis (local +X) · wall normal
    { return Eigen::Vector2f(std::cos(st.yaw), std::sin(st.yaw)).dot(f.wall.n); };

    const float gap = gap_of(s);
    dbg_wall_gap_ = gap;

    // ★WHEN THE DEPTH CANNOT BE SEEN AND A WALL CAN, THE WALL DETERMINES IT. A front-only view leaves the
    // centre and the depth jointly unidentifiable — the data fixes only the FRONT face, one equation in two
    // unknowns — and the response was to pin the depth to its prior at depth_unobs_precision. For a bed
    // that prior is a good 0.60 so the harm is small, but the better second equation is already available:
    // a flush object's BACK FACE IS ON THE WALL and the room model knows where the wall is. Found on
    // hood_concept, where the pinned depth was a nominal guess and the excess went straight through the wall.
    const float depth_unobs = 1.0f - two_sided_along(s, f, false);
    const float wgt = flush_posterior(s, f);
    const float lam_base = 1.0f / (1.0f / params_.wall_precision + f.wall.sigma_m * f.wall.sigma_m);
    const float lam_flush = wgt * (lam_base + params_.depth_unobs_precision * depth_unobs);
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
        const auto sp = CountableBeliefState::from_vec(vp), sm = CountableBeliefState::from_vec(vm);
        Jg(k) = std::clamp((gap_of(sp) - gap_of(sm)) / (2.0f * eps), -g, g);
        Jp(k) = std::clamp((par_of(sp) - par_of(sm)) / (2.0f * eps), -g, g);
    }
    Id.noalias() += lam_flush * (Jg * Jg.transpose());
    bd.noalias() += -lam_flush * Jg * gap;

    if (params_.wall_parallel_precision > 0.0f)
    {
        // ★★AN OBJECT ADJACENT TO A WALL IS ALIGNED TO IT, WHATEVER POSE THE MASKS CAME FROM. The yaw was
        // decided by a flat constant of 200 against thousands of mask points seen obliquely, and the points
        // won — hood_concept sat 9 degrees off the wall it is bolted to, drifting. That is not a tuning
        // problem, it is the wrong quantity in the precision: a single oblique face gives a poor surface
        // normal, the WALL's direction is known far better, and how much better is not a number anyone
        // needs to choose. A segment of length L localised to sigma_m has angular uncertainty ~ sigma_m/L,
        // and the room model already carries both — a 3 m wall known to 2 cm is known to 0.0066 rad, i.e.
        // lambda ~22800 against the constant's 200. DERIVED, not picked.
        //
        // The residual is sin(misalignment), so a precision on it is a precision on the angle. Still scaled
        // by the flush posterior, which is the point: ADJACENCY is what earns the alignment, and a
        // free-standing object (a mid-room bed, a kitchen island) has wgt -> 0 and keeps its data-driven
        // yaw. Both 180-degree solutions satisfy it, so the front/back resolution is untouched.
        const float sigma_theta = (f.wall.length_m > 0.1f) ? f.wall.sigma_m / f.wall.length_m : 0.0f;
        const float lam_geom    = (sigma_theta > 1e-4f) ? 1.0f / (sigma_theta * sigma_theta)
                                                        : params_.wall_parallel_precision;
        const float lam_par = wgt * lam_geom;   // same mixture weight
        const float rp = par_of(s);
        Id.noalias() += lam_par * (Jp * Jp.transpose());
        bd.noalias() += -lam_par * Jp * rp;
    }
}

// ─── Extra GN factors: wall-flush + footprint prior + YOLO-independent LiDAR first-hit range ─────────

void CountableBelief::accumulate_extra(const CountableBeliefState& s, const CountableFrame& f,
                                          Eigen::Matrix<float, 6, 6>& Id, Eigen::Matrix<float, 6, 1>& bd) const
{
    // (1) Wall-flush + wall-parallel structural term (supplies the missing back-face/depth information).
    //     It is the class's FURNITURE prior ("a bed stands against a wall", π_flush), not data: in the likelihood-only
    //     size summary (size_prior_off_) it contributes nothing, or it re-adds the depth pin through lam_flush
    //     (depth_unobs_precision·depth_unobs) and the π_flush fallback of flush_posterior, and the summary's depth
    //     becomes the front-to-wall distance (final review C2, 2026-11-14).
    // A LiDAR-only frame (f.lidar_only) skips every Gaussian class anchor — flush/parallel, room axis, (2) footprint, (2a)
    // height: the predicted belief already contains them, and re-adding them each frame counts the prior as a measurement
    // (spec lidar-measures §7b, C1 = A). The one-sided constraints (2b), (2c) and the rays stay: zero unless violated.
    if (not size_prior_off_ and not f.lidar_only)
        accumulate_wall(s, f, Id, bd);

    // ROOM-AXIS ALIGNMENT (furniture prior): furniture is aligned with the room's axes whether or not it touches
    // a wall. ¼·sin(4Δ) vanishes at every multiple of 90°, so the four yaws a box can take on the room grid are
    // equally good and the front/back (and width/depth) decision stays with the data and the clearance prior.
    if (f.room_axis_ok and params_.room_axis_std > 0.0f and not f.lidar_only)
    {
        const float dlt = s.yaw - f.room_axis;
        const float r   = 0.25f * std::sin(4.0f * dlt);
        const float dr  = std::cos(4.0f * dlt);                 // ∂r/∂yaw
        const float lam = 1.0f / (params_.room_axis_std * params_.room_axis_std);
        Id(5, 5) += lam * dr * dr;
        bd(5)    += -lam * dr * r;
    }

    // (2) FOOTPRINT prior: a TIGHT Gaussian pulling w → prior_footprint_m and h(depth) → prior_footprint_m
    //     (a standard bed footprint ≈ 0.60×0.60 is very consistent), precision λ_wd = 1/prior_footprint_std².
    //     H (index 2) is anchored separately just below (2a) — the top-of-box float needs a static anchor too.
    if (params_.prior_footprint_std > 0.0f and not size_prior_off_ and not f.lidar_only)
    {
        const float lam_wd = 1.0f / (params_.prior_footprint_std * params_.prior_footprint_std);

        // An extent is IDENTIFIABLE only when the cloud spans it — points on BOTH opposing faces, which is what
        // separates the extent from the centre. A one-face view puts points on a single face and its many points
        // still drag that extent outward (the size common-mode caps σ, not the mean), so the footprint prior must
        // PREVAIL absent two-face evidence: grow its precision by depth_unobs_precision·(1 − two_sided).
        //
        // ★This is now computed PER FOOTPRINT AXIS. It used to be hard-wired to the depth axis (h/local-Y) only,
        // which silently assumed the robot always approaches the FRONT. Approach from the SIDE and the roles swap:
        // the ±Y faces are the ones being seen, so h reads as two-sided and its boost correctly switches off —
        // while w, now the unobserved axis, had NO protection at all and inflated freely. Live: the robot came in
        // from the side and w grew to 1.414 m against a 0.60 prior, which the identity LLR then scored at −15.5,
        // and the bed filter removed a real bed for being the wrong shape. Which axis is unobservable is a
        // property of the VIEWPOINT, not of the model's axis labelling. See [[bed-unobserved-axis-prior]].
        const auto two_sided_along = [&](bool along_x) -> float
        { return this->two_sided_along(s, f, along_x); };
        // ★THE PRECISION IS CONSERVED AND MOVES ONLY AS FAR AS THE WALL TAKES IT (see accumulate_wall).
        // Handing it over whenever a wall merely EXISTS would be a bug here: a mid-room bed has a small
        // flush posterior, so the depth pin would be removed without the wall replacing it and the depth
        // would float on lam_wd alone. It moves in proportion to that posterior — continuously, and to
        // nothing at all when the bed is genuinely free-standing.
        const float depth_unobs = 1.0f - this->two_sided_along(s, f, false);
        const float w_flush     = flush_posterior(s, f);              // 0 when no wall is staged
        const float lam_w = lam_wd + params_.depth_unobs_precision * (1.0f - two_sided_along(true));
        const float lam_h = lam_wd + params_.depth_unobs_precision * depth_unobs * (1.0f - w_flush);
        Id(3, 3) += lam_w;  bd(3) += lam_w * (params_.prior_footprint_m - s.w);   // w      (index 3)
        Id(4, 4) += lam_h;  bd(4) += lam_h * (params_.depth_mean() - s.h);   // h≡depth(index 4)
    }

    // (2a) HEIGHT anchor (H, index 2): the box TOP is unconstrained-from-above — points at the real bed top
    //      (z ≈ 1.9 m) sit on a SIDE face of any TALLER box, so they never pull H down, and prior_cov_diag's broad
    //      hh is only the recursive-filter DRIFT allowance, NOT a static anchor. Without this term H random-walks
    //      UP (observed: 2.37 m for a 1.9 m cloud) with no restoring force. This is a static Gaussian anchor toward
    //      prior_height_m with precision 1/prior_height_std² — the DATA still sets the LOWER bound (points force
    //      H ≥ cloud-top), the anchor only prevents float ABOVE it. Broad enough to let a genuinely tall/short
    //      bed move when the cloud top demands it, firm enough to stop the runaway.
    if (params_.prior_height_std > 0.0f and not size_prior_off_ and not f.lidar_only)
    {
        const float lam_H = 1.0f / (params_.prior_height_std * params_.prior_height_std);
        Id(2, 2) += lam_H;  bd(2) += lam_H * (params_.prior_height_m - s.H);        // H (index 2)
    }

    // (2d) TOP TRACKS THE OBSERVED ROBUST CLOUD TOP. The floor-anchored box top is a "free" upper boundary — empty
    //      surface above the cloud is unpenalised, so H ratchets up on the over-segmented JUNK TAIL above the bed
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

    // (2b) SHORT-HEIGHT prior (the "a 70 cm bed is improbable" term): a ONE-SIDED Gaussian factor that acts
    //      ONLY when H < plaus_height_min, pulling H up toward it. Its precision GROWS as H falls further below
    //      (λ_h = gain·deficit/soft), so a mildly-short bed is nudged gently and a 70 cm cloud is resisted
    //      hard. A plausibly-tall bed (deficit ≤ 0) feels NOTHING. Continuous covariance, not a clamp — the
    //      effect is that fitting a short cloud as a bed fights this prior → worse data fit → lower plausibility.
    //      A CLASS prior like (2)/(2a), so it is off in the likelihood-only size summary too (size_prior_off_).
    if (params_.plaus_height_prior_gain > 0.0f and params_.plaus_height_soft > 1e-4f and not size_prior_off_)
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
        const auto gap_of = [&](const CountableBeliefState& st)
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
                Jg(k) = std::clamp((gap_of(CountableBeliefState::from_vec(vp))
                                  - gap_of(CountableBeliefState::from_vec(vm))) / (2.0f * eps), -g, g);
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

FootprintMoment CountableBelief::footprint_moment(const std::vector<Eigen::Vector3f>& pts,
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
std::array<bool, 4> CountableBelief::allowed_modes() const
{
    const float sum_wh = state_.w + state_.h;
    const bool rect = sum_wh > 1e-4f and std::abs(state_.w - state_.h) / sum_wh > kRectAspect;
    return {true, not rect, true, not rect};
}

// ─── Door CLEARANCE prior over the 4 modes (a door cannot open into a wall) ──────────────────────
//
// A bed stands with its BACK against a wall, so its door faces the room — it physically cannot face
// the wall it is attached to. That is geometry the agent already knows (the same wall the flush factor uses),
// and it is the ONLY thing that can resolve a 180° error: rotating the box half a turn swaps front↔back but
// moves nothing the camera can see. The appearance cue is structurally blind here — detect_front needs ≥2
// camera-facing faces and never scores the back one, so for a wall-flush bed viewed head-on it emits
// nothing at all. See [[bed-door-clearance-prior]].
//
// Per candidate mode k: how much its door direction points INTO the room, measured against the wall's inward
// normal, scaled by how strongly we believe this bed is flush against that wall in the first place:
//
//     logprior_k = gain · flush_weight · (front_dir_k · wall_inward_normal)
//
// (front_dir·n) ∈ [−1,1] is +1 for a door facing straight into the room and −1 for one facing into the wall.
// flush_weight is the EXISTING {flush, free-standing} mixture weight, so a genuine mid-room bed finds no
// wall in range, the weight decays to 0, and the prior vanishes on its own — continuous, no proximity gate.
//
// ★This is a PRIOR, not evidence: it is recomputed from current geometry wherever the mode is scored and is
// never accumulated into front_acc_. Adding it per cycle would be the same static-evidence repetition that has
// bitten this codebase four times now (see [[existence-policy-unification]]); as a prior it cannot saturate,
// and it silently follows the belief if the fit's yaw or the room polygon changes.
std::array<float, 4> CountableBelief::door_clearance_logprior() const
{
    std::array<float, 4> lp{0.0f, 0.0f, 0.0f, 0.0f};
    if (not point_wall_.ok or params_.door_clearance_gain <= 0.0f)
        return lp;                                   // no room polygon ⇒ nothing to say about clearance
    const float gap  = (back_centre(state_) - point_wall_.p).dot(point_wall_.n);
    const float rel  = gap / std::max(1e-3f, params_.wall_reach_m);
    const float flush = std::exp(-rel * rel);        // same mixture weight the wall-flush factor uses
    constexpr float kHalfPi = 0.5f * static_cast<float>(M_PI);
    const auto wrap = [](float a) { return std::remainder(a, 2.0f * static_cast<float>(M_PI)); };
    for (int k = 0; k < 4; ++k)
    {
        const float psi = wrap(state_.yaw + static_cast<float>(k) * kHalfPi);
        const Eigen::Vector2f front_dir(std::sin(psi), -std::cos(psi));   // door = local −Y
        lp[k] = params_.door_clearance_gain * flush * front_dir.dot(point_wall_.n);
    }
    return lp;
}

// Adopt whichever door mode the accumulated appearance evidence PLUS the clearance prior favours, with no new
// appearance cue required. Called every cycle so the geometry alone can correct a bed whose door faces its
// own wall — the case detect_front can never report on. Returns true if the belief was rotated.
bool CountableBelief::resolve_front_geometric()
{
    const auto lp = door_clearance_logprior();
    const std::array<bool, 4> allowed = allowed_modes();
    int kbest = 0;
    float best = front_acc_[0] + lp[0];
    for (int k = 1; k < 4; ++k)
        if (allowed[k] and front_acc_[k] + lp[k] > best) { best = front_acc_[k] + lp[k]; kbest = k; }
    if (kbest == 0)
        return false;
    apply_mode_rotation(kbest);
    const std::array<float, 4> old = front_acc_;    // re-baseline so the adopted mode becomes index 0
    for (int m = 0; m < 4; ++m)
        front_acc_[m] = std::clamp(old[(m + kbest) % 4] - old[kbest], -6.0f, 6.0f);
    return true;
}

// Adopt door mode k: rotate yaw + (for a 90°/270° swap) relabel w↔h and swap the matching Σ / prior rows+cols.
void CountableBelief::apply_mode_rotation(int k)
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
        size_lik_.mean.tail<2>().reverseInPlace();                       // the data summary relabels with the state
        size_lik_.info.row(1).swap(size_lik_.info.row(2));
        size_lik_.info.col(1).swap(size_lik_.info.col(2));
    }
}

// ─── Likelihood-only size summary ────────────────────────────────────────────────────────────────────────────────────
// The posterior holds the class size prior (accumulate_extra 2 and 2a), added to the data information EVERY frame. To score
// the shape against the class without counting that prior twice, redo the data accumulation at the converged state with
// those two blocks off, saturate it with this frame's common-mode (as ai::update does), marginalise the pose out by a Schur
// complement, and fold the (H, w, h) frame summary into a parallel recursion that forgets exactly as the belief does.
void CountableBelief::update_size_likelihood(const CountableFrame& frame)
{
    using Mat6 = Eigen::Matrix<float, 6, 6>;
    using Vec6 = Eigen::Matrix<float, 6, 1>;
    using Mat3d = Eigen::Matrix3d;
    using Vec3d = Eigen::Vector3d;
    using Mat6d = Eigen::Matrix<double, 6, 6>;
    using Vec6d = Eigen::Matrix<double, 6, 1>;
    const CountableBeliefState s = state_;
    const float sig2 = sigma2();

    // DOUBLE throughout: the raw size information is ~4e5 and the Schur remainder along an unobserved depth ~0.2, so float
    // cancellation would invent 5-14 information units per frame (and an indefinite J). Only the stored summary is float.
    Mat6d Id = Mat6d::Zero(); Vec6d bd = Vec6d::Zero();
    for (std::size_t i = 0; i < frame.points.size(); ++i)          // the same accumulation as ai::update's lambda
    {
        const float R = (i < frame.R.size() and frame.R[i] > 0.0f) ? frame.R[i] : sig2;
        const auto  r = responsibilities(frame.points[i], s, R);
        const float w = r[0] / R;
        if (w < 1e-6f) continue;
        const double d = sdf_prim(frame.points[i], s, 0);
        const Vec6d  J = sdf_jacobian(frame.points[i], s, 0).cast<double>();
        Id.noalias() += double(w) * (J * J.transpose());
        bd.noalias() += -double(w) * J * d;
    }
    {   // the extras (data or structure, small) in float, size prior blocks off; RAII restores the switch
        Mat6 Ie = Mat6::Zero(); Vec6 be = Vec6::Zero();
        struct Off { bool& f; explicit Off(bool& x) : f(x) { f = true; } ~Off() { f = false; } } off(size_prior_off_);
        accumulate_extra(s, frame, Ie, be);
        Id += Ie.cast<double>(); bd += be.cast<double>();
    }

    const Vec6 sc = common_mode_inv_diag(frame);

    // pseudo-inverse on eigenvalues > 1e-6 · max: an unobserved direction stays unobserved instead of exploding
    const auto pinv = [](const Mat3d& M) -> Mat3d
    {
        const Eigen::SelfAdjointEigenSolver<Mat3d> es(0.5 * (M + M.transpose()));
        const double top = es.eigenvalues().cwiseAbs().maxCoeff();
        Vec3d inv = Vec3d::Zero();
        for (int k = 0; k < 3; ++k) if (es.eigenvalues()(k) > 1e-6 * top and es.eigenvalues()(k) > 0.0) inv(k) = 1.0 / es.eigenvalues()(k);
        return es.eigenvectors() * inv.asDiagonal() * es.eigenvectors().transpose();
    };

    // ORDER MATTERS: marginalise the pose out of the RAW data information FIRST, then apply the common-mode cap in the
    // 3-D size subspace. Saturating the 6x6 first (as ai::update does for the posterior) caps EVERY direction the data
    // touches at ~Scinv (Id is 1e4..1e6 against Scinv ~2.5e3), flattens the pose-size cross terms to ~0, and the Schur
    // then removes nothing: a depth the view cannot see would keep as much information as one it can. Marginalising
    // first leaves a rank-deficient J exactly where the geometry is degenerate (a front face fixes the face, not the
    // depth behind it). No regulariser on the pose block (any pose precision taken from Sigma_ already contains the
    // class prior through the cy-h coupling): pseudo-inverse instead. Select (H, w, h) = (2, 3, 4); pose = (cx, cy, yaw).
    constexpr int S[3] = {2, 3, 4}, P[3] = {0, 1, 5};
    Mat3d Iss, Isp, Ipp; Vec3d bs, bp;
    for (int a = 0; a < 3; ++a)
    {
        bs(a) = bd(S[a]); bp(a) = bd(P[a]);
        for (int b = 0; b < 3; ++b) { Iss(a, b) = Id(S[a], S[b]); Isp(a, b) = Id(S[a], P[b]); Ipp(a, b) = Id(P[a], P[b]); }
    }
    const Mat3d Ppinv = pinv(Ipp);
    Mat3d J = Iss - Isp * Ppinv * Isp.transpose();
    J = 0.5 * (J + J.transpose());
    Vec3d g = bs - Isp * Ppinv * bp;
    {   // the common-mode cap, in the size subspace (ai::saturate's Woodbury form, in double)
        Mat3d Sc = Mat3d::Zero(); for (int a = 0; a < 3; ++a) Sc(a, a) = sc(S[a]);
        const Mat3d ScM = Sc * (J + Sc).inverse();
        Mat3d Je = Sc - ScM * Sc;
        g = ScM * g;
        J = 0.5 * (Je + Je.transpose());
    }

    const Vec3d theta_s(s.H, s.w, s.h);
    const Vec3d mean_frame = theta_s + pinv(J) * g;

    // ★AN EXTENT THE CLOUD DOES NOT SPAN CARRIES NO INFORMATION (final review C1, 2026-11-14). The Schur complement above
    // removes the pose-size degeneracy only in exact geometry. The FD, slope-clamped SDF Jacobians at a linearisation
    // point whose depth came from the class prior leave a residual information on that depth even noise-free (a front
    // face seen alone gave ~230 /m² on h, and the summary's depth then FOLLOWED the prior depth: 1.0/2.1/3.0 → 0.94/2.04/
    // 2.94), so the prior came back in through the linearisation point. The model's own identifiability statistic says
    // how much of each footprint extent the cloud spans (the fraction of points clearly on both sides of the centre,
    // the same statistic the fit's depth-observability block uses): scale the frame's information along each body axis
    // by it — w (local x) by t_x, h (local y) by t_y, H untouched, cross terms by the geometric mean (D J D with
    // D = diag(1, √t_x, √t_y)). The frame MEAN stays the unscaled data solve: only the confidence in it is reduced, so
    // a half-seen extent counts half, and nothing is gated. Cost if wrong: a genuinely seen depth is slightly
    // under-counted on partial views.
    const double t_x = two_sided_fraction(s, frame, true);    // in [0, 1] by construction
    const double t_y = two_sided_fraction(s, frame, false);
    const Vec3d D(1.0, std::sqrt(t_x), std::sqrt(t_y));
    J = D.asDiagonal() * J * D.asDiagonal();

    // a non-finite frame (degenerate Schur, NaN point) is skipped: it must never poison the stored summary
    if (not (J.allFinite() and mean_frame.allFinite()))
        return;

    // fold the frame into the recursion (the forgetting is predict_size_likelihood, run before this on every cycle).
    // J is PSD in exact arithmetic; drop numerically negative eigenvalues so rounding cannot subtract information.
    {
        const Eigen::SelfAdjointEigenSolver<Mat3d> es(J);
        const Vec3d ev = es.eigenvalues().cwiseMax(0.0);
        J = es.eigenvectors() * ev.asDiagonal() * es.eigenvectors().transpose();
    }
    Mat3d info = size_lik_.info.cast<double>();
    const Vec3d eta = info * size_lik_.mean.cast<double>() + J * mean_frame;
    info += J;
    size_lik_.info = info.cast<float>();
    size_lik_.mean = (pinv(info) * eta).cast<float>();
}

// Forget exactly as the belief does: info <- (info^-1 + Q)^-1 with Q = q_scale*diag(Q_s), in the inverse-free form
// (I + info*Q)^-1 * info. A null direction stays null (a pseudo-inverse round trip would turn "no information" into 1/q),
// and Q = 0 leaves the information untouched (no 1/Q anywhere). Double inside; float stored.
void CountableBelief::predict_size_likelihood(float q_scale)
{
    if (size_lik_.info.norm() == 0.0f or not (q_scale > 0.0f)) return;
    const Eigen::Matrix<float, 6, 1> q = process_noise_diag();
    Eigen::Matrix3d Q = Eigen::Matrix3d::Zero();
    for (int a = 0; a < 3; ++a) Q(a, a) = double(q_scale) * double(q(2 + a));
    const Eigen::Matrix3d info = size_lik_.info.cast<double>();
    const Eigen::Matrix3d out  = (Eigen::Matrix3d::Identity() + info * Q).inverse() * info;
    size_lik_.info = (0.5 * (out + out.transpose())).cast<float>();
}

// Fold one appearance FrontCue into the door-mode accumulator and adopt the argmax mode. See the header.
bool CountableBelief::resolve_front(const FrontCue& cue, float evidence_weight)
{
    constexpr float kHalfPi    = 0.5f * static_cast<float>(M_PI);
    // Clamp each accumulated mode-evidence to ±kFrontClamp so a long confident run cannot drive the accumulator
    // arbitrarily deep and make the belief unable to RECANT when the bed is physically turned. ≈6 nats ≈ a
    // 400:1 odds — decisive but recoverable. Mirrors ChairBelief's ±6 flip clamp.
    constexpr float kFrontClamp = 6.0f;
    const float w = std::clamp(evidence_weight, 0.0f, 1.0f) * std::clamp(cue.confidence, 0.0f, 1.0f);
    if (w <= 1e-4f)
        return false;   // no confident evidence this cue → leave the accumulator (and the belief) untouched

    const auto wrap = [](float a) { return std::remainder(a, 2.0f * static_cast<float>(M_PI)); };
    const Eigen::Vector2f cue_dir(std::cos(cue.bearing_rad), std::sin(cue.bearing_rad));

    // ★NOVELTY: charge the OBSERVER-bearing bin, so a repeated look from the same place cannot keep voting.
    // The same discipline ChairBelief::flip_acc_ already applies.
    //
    // ★★AND AN UNKNOWN VIEWPOINT FAILS CLOSED. This used to read `if (isfinite(view_bearing)) {...}` with the
    // NaN case falling through at FULL weight — labelled "keeps the legacy behaviour rather than silently
    // changing it", which is exactly backwards: it is the one branch where we CANNOT tell a fresh view from
    // the same view repeated, and it handed out the discovery rate. That is fail-OPEN, the same shape as the
    // gates invariant II condemns, and it bites precisely when the pose chain is unavailable — i.e. when
    // registration is least trustworthy.
    //
    // With the viewpoint unknown the honest charge is the one we would make for a REPEAT, so we attribute the
    // look to the bin we have already spent the most on. That also makes the branch self-limiting: successive
    // unknown-viewpoint cues deplete that same bin, novelty decays toward 0, and a broken pose chain can no
    // longer feed the accumulator indefinitely. Known and unknown now share one code path — only the choice
    // of BIN differs, which is the only thing the two cases actually disagree about.
    constexpr int B = kFrontViewBins;
    int bin = 0;
    if (std::isfinite(cue.view_bearing_rad))
    {
        const float span = 2.0f * static_cast<float>(M_PI) / static_cast<float>(B);
        const float wrapped = std::atan2(std::sin(cue.view_bearing_rad), std::cos(cue.view_bearing_rad));
        bin = std::clamp(static_cast<int>(std::floor((wrapped + static_cast<float>(M_PI)) / span)), 0, B - 1);
    }
    else
        bin = static_cast<int>(std::distance(front_view_spent_.begin(),
                               std::max_element(front_view_spent_.begin(), front_view_spent_.end())));
    const float budget    = std::max(1e-3f, front_view_budget_);
    const float remaining = std::max(0.0f, budget - front_view_spent_[bin]);
    const float novelty   = remaining / (remaining + budget);
    const float w_eff     = w * novelty;
    front_view_spent_[bin] += w_eff;   // charge the bin what this frame actually contributed
    const std::array<bool, 4> allowed = allowed_modes();

    // Accumulate evidence for EVERY candidate: the alignment of that mode's door normal (local −Y rotated by the
    // candidate yaw ψ_k) with the observed door bearing. Door local −Y ⇒ room dir R(ψ)·(0,−1)=(sinψ,−cosψ).
    //
    // ★Accumulate even for modes that are not currently ADOPTABLE. `allowed` reflects the footprint's present
    // rectangularity, which changes as the fit converges — a bed births elongated (live: |w−h|/(w+h)=0.47 at
    // cyc26, so only {0,180} were allowed) and squares up later. `continue`-ing here THREW AWAY every observation
    // of the true door while the box happened to be elongated, so by the time all four modes became adoptable the
    // evidence for the right one had never been recorded. Adoption is still restricted below; only the bookkeeping
    // is unconditional. See [[bed-door-mode-lock]].
    for (int k = 0; k < 4; ++k)
    {
        const float psi = wrap(state_.yaw + static_cast<float>(k) * kHalfPi);
        const Eigen::Vector2f front_dir(std::sin(psi), -std::cos(psi));
        front_acc_[k] += w_eff * front_dir.dot(cue_dir);
    }

    // ★Re-baseline on the INCUMBENT, then clamp. The accumulator is a log-odds RATIO against the currently held
    // mode, so what must stay bounded is the DIFFERENCE between modes, not each mode's absolute score.
    // Previously every entry was clamped independently to ±kFrontClamp: a static view re-adding the same w each
    // frame drove the incumbent front_acc_[0] to +6, and since a challenger is bounded by the SAME clamp and must
    // be STRICTLY greater to win, 6.0 > 6.0 is false ⇒ the mode could NEVER change again. Live: front_conf pinned
    // at 0.996 (= softmax of exactly [+6,0,−6,0]) for thousands of cycles while the cue said "door is 90° from
    // where you think", and the bed stayed a quarter-turn off. Same defect family as
    // [[chair-flip-acc-repetition-defect]]. With the incumbent held at 0, a challenger can always overtake it,
    // and recanting — the reason the clamp exists — actually works.
    const float base = front_acc_[0];
    for (float& a : front_acc_)
        a = std::clamp(a - base, -kFrontClamp, kFrontClamp);

    // Score = accumulated APPEARANCE evidence + the geometric CLEARANCE prior (a door cannot open into the wall
    // this bed is attached to). Same currency (nats), so they simply add; the prior is recomputed here rather
    // than accumulated, so it never saturates.
    const auto lp = door_clearance_logprior();
    int kbest = 0;
    float best = front_acc_[0] + lp[0];
    for (int k = 1; k < 4; ++k)
        if (allowed[k] and front_acc_[k] + lp[k] > best) { best = front_acc_[k] + lp[k]; kbest = k; }
    if (kbest == 0)
        return false;   // current believed door direction still best-explained by appearance + clearance

    apply_mode_rotation(kbest);
    const std::array<float, 4> old = front_acc_;    // re-baseline so the new current mode (old kbest) is index 0
    for (int m = 0; m < 4; ++m)
        front_acc_[m] = std::clamp(old[(m + kbest) % 4] - old[kbest], -kFrontClamp, kFrontClamp);
    return true;
}

// Posterior over the 4 front modes = softmax(front_acc_) with disallowed modes zeroed.
std::array<float, 4> CountableBelief::front_posterior() const
{
    const std::array<bool, 4> allowed = allowed_modes();
    const auto lp = door_clearance_logprior();   // the SAME score resolve_front decides on (evidence + prior)
    std::array<float, 4> s{};
    for (int k = 0; k < 4; ++k) s[k] = front_acc_[k] + lp[k];
    std::array<float, 4> p{};
    float mx = -std::numeric_limits<float>::infinity();
    for (int k = 0; k < 4; ++k) if (allowed[k]) mx = std::max(mx, s[k]);
    float sum = 0.0f;
    for (int k = 0; k < 4; ++k) { p[k] = allowed[k] ? std::exp(s[k] - mx) : 0.0f; sum += p[k]; }
    if (sum <= 0.0f) { for (int k = 0; k < 4; ++k) p[k] = allowed[k] ? 0.5f : 0.0f; return p; }
    for (float& v : p) v /= sum;
    return p;
}

float CountableBelief::front_confidence() const
{
    const auto p = front_posterior();
    return *std::max_element(p.begin(), p.end());
}

int CountableBelief::front_mode() const
{
    const auto p = front_posterior();
    return static_cast<int>(std::distance(p.begin(), std::max_element(p.begin(), p.end())));
}

// REPORTED yaw variance: within-mode Σ(5,5) + Var_k[Δ_k] under the front posterior (Δ_k = wrapped k·π/2).
float CountableBelief::yaw_marginal_var() const
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

Eigen::Matrix<float, 6, 6> CountableBelief::covariance_reported() const
{
    Eigen::Matrix<float, 6, 6> S = Sigma_;
    S(5, 5) = yaw_marginal_var();
    return S;
}

// ─── "Is this really a bed?" plausibility + soft singleton (mis-detection filter) ────────────────

// Continuous bed-plausibility ∈ (0,1] = product of soft factors. See the header. No hard threshold: the
// mis-detection (elongated / short / poor-fit) is simply poorly explained by the bed shape priors → low score.
float CountableBelief::countable_plausibility(const CountableBeliefState& s, float fe,
                                              const CountableBeliefParams& p)
{
    const float sum_wh   = std::max(1e-4f, s.w + s.h);
    const float aspect   = std::abs(std::abs(s.w - s.h) / sum_wh
                                   - std::abs(p.prior_footprint_m - p.depth_mean())
                                     / std::max(1e-4f, p.prior_footprint_m + p.depth_mean()));   // 0 = the class's own aspect
    const float as       = std::max(1e-4f, p.plaus_aspect_scale);
    const float aspect_ok = std::exp(-(aspect / as) * (aspect / as));

    const float ss       = std::max(1e-4f, p.plaus_size_scale);
    const float dw       = s.w - p.prior_footprint_m, dh = s.h - p.depth_mean();
    const float size_ok  = std::exp(-(dw * dw + dh * dh) / (2.0f * ss * ss));   // footprint near 0.60×0.60

    const float hsoft    = std::max(1e-4f, p.plaus_height_soft);
    const float height_ok = 1.0f / (1.0f + std::exp(-(s.H - p.plaus_height_min) / hsoft));  // logistic: tall→1

    // The fit_ok term (absolute mean_energy vs a guessed FeRef) was DROPPED — FE varies strongly with view /
    // point-count, so an absolute reference is unreliable and was REJECTING REAL BEDS. The SHAPE factors are
    // the robust bed-vs-not discriminators: on the FITTED box the footprint prior pins w,h≈0.60 (aspect_ok /
    // size_ok ≈1 for a genuine bed), while an elongated / short mis-detection fails aspect and/or height.
    // `fe` stays in the signature (call sites unchanged) but is no longer used.
    (void) fe;
    return std::clamp(aspect_ok * size_ok * height_ok, 0.0f, 1.0f);
}

// Birth-candidate plausibility = P(this cloud is a <class>) on the SAME scale as the class evidence of the fit:
//   logit(π_cls) + Λ_height, with Λ_height = birth_height_log_evidence(robust z-extent of the cloud).
// π_cls is the manifest's label reliability; Λ_height scores the cloud's height as a density under "this class"
// (a mixture over the mattress..headboard range, plus an outlier fraction for mask bleed) against "some other
// furniture" — two-sided, so a 1.9 m blackboard cloud is as improbable as a too-short one. ★A partial mask (a
// front-only ZED view) is a VERTICAL face, so its 2-D footprint is a thin line and aspect/size are MEANINGLESS at
// birth; the z-extent is reliable from any viewpoint, so only height enters here. The footprint is settled later
// by the fit (class_log_evidence). Fewer than 12 points ⇒ the bare prior π_cls (a sparse first glimpse is
// neither rewarded nor punished).
float CountableBelief::candidate_plausibility(const std::vector<Eigen::Vector3f>& pts,
                                                 const CountableBeliefParams& p)
{
    if (pts.size() < 12) return p.class_p_label_true;
    using namespace rc::countable;
    return class_probability(logit(p.class_p_label_true) +
                             birth_height_log_evidence(robust_z_extent(pts), p.class_shape_prior(), p.birth_height_noise_m,
                                                       p.birth_height_outlier_frac, p.class_alt_std_m));
}

// ─── Verification ────────────────────────────────────────────────────────────────────────────────

bool CountableBelief::self_test()
{
    std::mt19937 rng(12345);
    std::normal_distribution<float> noise(0.0f, 0.01f);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f), U01(0.0f, 1.0f);

    CountableBeliefParams P;
    const CountableBeliefState gt{0.20f, -0.30f, 1.75f, 0.70f, 0.60f, 0.30f};   // ground truth (w≠h)

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
    CountableBeliefState s0{0.0f, 0.0f, 1.4f, 0.5f, 0.5f, 0.15f};
    Eigen::Vector3f sum = Eigen::Vector3f::Zero(); float zmax = 0.0f;
    for (const auto& p : pts) { sum += p; zmax = std::max(zmax, p.z()); }
    s0.cx = sum.x() / pts.size(); s0.cy = sum.y() / pts.size(); s0.H = zmax;

    CountableBelief b(s0, P);
    CountableFrame frame; frame.points = pts;
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

    std::printf("CountableBelief::self_test [full-cloud] %s  cx=%.3f cy=%.3f H=%.3f w=%.3f h=%.3f yaw=%.3f (gt %.3f)\n",
                ok ? "PASS" : "FAIL", s.cx, s.cy, s.H, s.w, s.h, s.yaw, gt.yaw);

    // ── (a) WALL-FLUSH under a PARTIAL (front-only) view ─────────────────────────────────────────────
    // A bed back-against a wall along y=0 (interior on +y). Its back face is at y=0, centre at cy=+h/2.
    // We observe ONLY the FRONT face (the back is occluded by the wall, the tall top is unseen from the
    // floor) — so depth `h` is unconstrained by points. The seed has the front face on the data but a too-deep
    // box whose back is buried 0.25 m INSIDE the wall. The wall-flush factor (+ footprint prior) must shrink
    // depth to ≈0.60 and pull the back face flush to y=0, WITHOUT any point ever touching the back face.
    const auto front_only_cloud = [&](const CountableBeliefState& g)
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

    const CountableBeliefState gtw{0.50f, 0.30f, 1.70f, 0.60f, 0.60f, 0.0f};   // back face flush at y=0
    CountableFrame fw;
    fw.points = front_only_cloud(gtw);
    fw.R.assign(fw.points.size(), P.sigma_base_m * P.sigma_base_m);
    fw.wall.ok = true; fw.wall.p = {0.0f, 0.0f}; fw.wall.n = {0.0f, 1.0f}; fw.wall.sigma_m = 0.02f;

    // Seed: front face correct (cy+h/2 = 0.60) but depth too big ⇒ back face 0.25 m behind the wall.
    CountableBeliefState sw{0.50f, 0.175f, 1.70f, 0.60f, 0.85f, 0.0f};
    CountableBelief bw(sw, P);
    bw.set_room_interior({0.5f, 5.0f});                      // interior on +y ⇒ back face is the −y (wall) face
    for (int it = 0; it < 60; ++it) bw.update(fw);
    const auto& w = bw.state();
    const float back_y   = bw.back_centre(w).y();            // should be ≈0 (flush)
    const bool ok_wall =
        std::abs(back_y)   < 0.04f and                       // back face within 4 cm of the wall
        std::abs(w.h - 0.60f) < 0.10f and                    // depth held at the standard footprint
        std::abs(w.w - 0.60f) < 0.10f;
    std::printf("CountableBelief::self_test [wall-flush partial] %s  cy=%.3f h=%.3f w=%.3f back_gap=%.3f λ=%.1f\n",
                ok_wall ? "PASS" : "FAIL", w.cy, w.h, w.w, back_y, bw.last_wall_lambda());

    // ── (b) FREE-STANDING: a far wall must NOT drag the bed ────────────────────────────────────────
    // Same front-only cloud, but the nearest wall is 5 m away. The flush mixture weight → ~0, so accumulate_wall
    // is inert: the bed must stay put (cy≈0.30 from the front face + footprint prior), NOT slide toward the
    // wall to make its back flush at y=−5.
    CountableFrame ff = fw;
    ff.wall.p = {0.0f, -5.0f};                               // wall 5 m behind (out of reach)
    CountableBeliefState sf{0.50f, 0.30f, 1.70f, 0.60f, 0.60f, 0.0f};
    CountableBelief bf(sf, P);
    bf.set_room_interior({0.5f, 5.0f});
    for (int it = 0; it < 60; ++it) bf.update(ff);
    const auto& fr = bf.state();
    const bool ok_free =
        std::abs(fr.cy - 0.30f) < 0.10f and                  // not dragged toward the far wall
        std::abs(fr.h - 0.60f) < 0.12f and
        bf.last_wall_lambda() < 1.0f;                        // flush precision essentially off
    std::printf("CountableBelief::self_test [free-standing]     %s  cy=%.3f h=%.3f λ=%.3f\n",
                ok_free ? "PASS" : "FAIL", fr.cy, fr.h, bf.last_wall_lambda());

    // ── (b′) WALL NO-CROSS: the bed must not penetrate the wall behind it ────────────────────────────
    // Wall along y=0 (interior on +y, inward normal +y). A FRONT-only cloud whose front face sits at y=+0.20 with
    // the standard 0.60 m depth prior places the BACK face at y≈−0.40 — buried 0.40 m INSIDE the wall. That far
    // past, the flush mixture weight exp(−(gap/reach)²) has decayed to ~0, so the two-sided flush factor can no
    // longer pull it out: WITHOUT the one-sided no-cross term the box stays CROSSED (back≈−0.40). WITH it (precision
    // grows with penetration) the back is driven back to the interior side (gap ≥ ~0). As with the short-height
    // prior above, the test engages a FIRM precision to demonstrate the mechanism cleanly; the shipped default
    // (2000) is gentler (it strongly reduces, but does not fully null, a deep forced penetration).
    const CountableBeliefState gtc{0.50f, -0.10f, 1.70f, 0.60f, 0.60f, 0.0f};   // front face at y=+0.20
    CountableFrame fc;
    fc.points = front_only_cloud(gtc);
    fc.R.assign(fc.points.size(), P.sigma_base_m * P.sigma_base_m);
    fc.wall.ok = true; fc.wall.p = {0.0f, 0.0f}; fc.wall.n = {0.0f, 1.0f}; fc.wall.sigma_m = 0.02f;
    const CountableBeliefState sc{0.50f, -0.10f, 1.70f, 0.60f, 0.60f, 0.0f};    // seed already crossing (back at −0.40)

    CountableBeliefParams Poff = P; Poff.wall_no_cross_precision = 0.0f;         // constraint OFF (baseline)
    CountableBelief boff(sc, Poff); boff.set_room_interior({0.5f, 5.0f});
    for (int it = 0; it < 80; ++it) boff.update(fc);
    const float back_off = boff.back_centre(boff.state()).y();                      // expect strongly negative (crossed)

    CountableBeliefParams Pon = P; Pon.wall_no_cross_precision = 100000.0f;      // firm, to demonstrate the null
    CountableBelief bon(sc, Pon); bon.set_room_interior({0.5f, 5.0f});
    for (int it = 0; it < 80; ++it) bon.update(fc);
    const float back_on = bon.back_centre(bon.state()).y();                         // expect ≥ ~0 (does not cross)

    // A bed genuinely IN FRONT of the wall (back at y≈+0.30) must be UNAFFECTED (no-cross residual = 0 there).
    const CountableBeliefState gtf{0.50f, 0.60f, 1.70f, 0.60f, 0.60f, 0.0f};     // front face at y=+0.90, back at +0.30
    CountableFrame fcf; fcf.points = front_only_cloud(gtf);
    fcf.R.assign(fcf.points.size(), P.sigma_base_m * P.sigma_base_m);
    fcf.wall.ok = true; fcf.wall.p = {0.0f, 0.0f}; fcf.wall.n = {0.0f, 1.0f}; fcf.wall.sigma_m = 0.02f;
    CountableBelief bfront(gtf, Pon); bfront.set_room_interior({0.5f, 5.0f});
    for (int it = 0; it < 80; ++it) bfront.update(fcf);
    const float back_front = bfront.back_centre(bfront.state()).y();                // expect ≈ +0.30 (untouched)

    const bool ok_nocross =
        back_off   < -0.15f and      // WITHOUT the constraint: the back penetrates the wall
        back_on    > -0.05f and      // WITH it: the back does not cross (gap ≥ ~0)
        back_front >  0.15f;         // a genuinely-in-front bed is unaffected
    std::printf("CountableBelief::self_test [wall no-cross]      %s  back_off=%.3f back_on=%.3f back_front=%.3f\n",
                ok_nocross ? "PASS" : "FAIL", back_off, back_on, back_front);

    // ── (c) FRONT (door) yaw resolver ──────────────────────────────────────────────────────────────
    // A SQUARE bed (w==h ⇒ all 4 door modes allowed) initially believed with its door facing +X (yaw=0 ⇒
    // door local −Y faces bearing atan2(−cos0,sin0)=atan2(−1,0)=−π/2). Feed a run of confident cues that the
    // door actually faces +Y (bearing +π/2). The resolver must rotate the belief so its door normal ends up
    // pointing at +π/2, and a converged belief must NOT be flipped by one low-confidence contradictory cue.
    constexpr float kHalfPi = 0.5f * static_cast<float>(M_PI);
    const auto wrap2pi = [](float a) { return std::remainder(a, 2.0f * static_cast<float>(M_PI)); };
    const auto door_bearing = [&](const CountableBelief& b)   // room bearing the believed door (local −Y) faces
    { const float y = b.state().yaw; return std::atan2(-std::cos(y), std::sin(y)); };

    CountableBeliefState sq{0.0f, 0.0f, 1.70f, 0.60f, 0.60f, 0.0f};   // square footprint, door believed at −π/2
    CountableBelief bd(sq, P);
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
    std::printf("CountableBelief::self_test [front-resolver]     %s  door_bearing=%.3f (gt %.3f) σyaw %.2f→%.2f p=%.2f\n",
                ok_front ? "PASS" : "FAIL", door_bearing(bd), kHalfPi,
                sigma_yaw_undecided, sigma_yaw_settled, bd.front_confidence());

    // (d)-(f), the shape LLR / its accumulator / the singleton deltas, are gone: the class factor is class_log_evidence
    // + class_factor_step (countable_class_evidence.h), tested by class_evidence_test and class_factor_test.

    // ── (g) SHORT-HEIGHT PRIOR: fitting a 70 cm cloud yields a HIGHER FE than a 1.7 m cloud (same footprint) ─
    // The one-sided height prior pulls H up toward plaus_height_min, so a genuinely-short cloud fits worse (its
    // observed top face becomes interior → misfit). A clean top face resists a WEAK prior, so this test engages
    // the prior firmly (Ph.plaus_height_prior_gain well past the point where it lifts H to the plausible range)
    // to demonstrate the mechanism; the shipped DEFAULT gain is gentler (firming, not overriding) — the primary
    // short-bed rejector is the plausibility height_ok term, which reads the fitted H directly.
    CountableBeliefParams Ph = P;
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
        CountableBeliefState s0{0.0f, 0.0f, Hgt, 0.60f, 0.60f, 0.0f};
        CountableBelief bb(s0, Ph);
        CountableFrame fr; fr.points = box_cloud(Hgt);
        fr.R.assign(fr.points.size(), Ph.sigma_base_m * Ph.sigma_base_m);
        for (int it = 0; it < 30; ++it) bb.update(fr);
        return bb.mean_energy(fr.points, bb.state(), Ph.sigma_base_m * Ph.sigma_base_m);
    };
    const float fe_short = fit_fe(0.70f);   // short: prior resists ⇒ misfit ⇒ higher FE
    const float fe_tall  = fit_fe(1.70f);   // tall: prior inactive ⇒ clean fit ⇒ lower FE
    const bool ok_height = fe_short > fe_tall;
    std::printf("CountableBelief::self_test [short-height prior] %s  FE(0.70m)=%.3f > FE(1.70m)=%.3f\n",
                ok_height ? "PASS" : "FAIL", fe_short, fe_tall);

    // ── (i) WALL EXPLAINING-AWAY: a wall patch must earn ~no evidence for a bed; a real bed must ────
    // The discriminating case from countable_1.png: YOLO puts a "bed" box on a flat piece of WALL. Both
    // clouds are fitted equally well by a box (a wall slab IS a box face), so free energy alone cannot separate
    // them — the separator is the log Bayes factor vs a scene that already contains the wall.
    {
        CountableBeliefParams Pw = P;
        Pw.wall_explain_frac = 0.25f; Pw.wall_explain_sigma_m = 0.05f;
        CountableBelief bw(CountableBeliefState{0.0f, 0.0f, 1.90f, 0.60f, 0.60f, 0.0f}, Pw);
        // Wall along y at x = 0, inward normal +x (room interior at x>0).
        CountableFrame fw;
        fw.wall.ok = true; fw.wall.p = {0.0f, 0.0f}; fw.wall.n = {1.0f, 0.0f}; fw.wall.sigma_m = 0.02f;

        // (1) A WALL PATCH: a flat vertical sheet lying IN the wall plane (x≈0).
        std::vector<Eigen::Vector3f> wall_pts;
        for (int i = 0; i < 600; ++i)
            wall_pts.push_back({noise(rng), U(rng) * 0.4f, U01(rng) * 1.9f});
        // (2) A REAL COUNTABLE standing against that wall: visible FRONT face 0.60 m into the room (x≈0.60).
        std::vector<Eigen::Vector3f> countable_pts;
        for (int i = 0; i < 600; ++i)
            countable_pts.push_back({0.60f + noise(rng), U(rng) * 0.3f, U01(rng) * 1.9f});

        const float Rq = bw.sigma2();
        // Box hypotheses fitted to each cloud (both are legitimate box fits — that is the whole difficulty).
        const CountableBeliefState box_on_wall  {0.0f, 0.0f, 1.90f, 0.60f, 0.60f, 0.0f};   // centre in the wall
        const CountableBeliefState box_on_floor {0.30f, 0.0f, 1.90f, 0.60f, 0.60f, 0.0f};  // front face at x=0.60
        bw.set_frame_wall_for_test(fw.wall);
        const float gain_wall   = bw.box_log_evidence_gain(wall_pts,   box_on_wall,  Rq);
        const float gain_countable = bw.box_log_evidence_gain(countable_pts, box_on_floor, Rq);
        // last_wall_resp() is a side effect of scoring, so score each cloud then read it.
        bw.mean_energy(wall_pts, box_on_wall, Rq);
        const float resp_wall = bw.last_wall_resp();
        bw.mean_energy(countable_pts, box_on_floor, Rq);
        const float resp_countable = bw.last_wall_resp();
        // The wall patch must gain far less than the real bed, and its points must be owned by the wall.
        const bool ok_explain = gain_countable > 1.0f and gain_wall < 0.25f * gain_countable and
                                resp_wall > 0.5f and resp_countable < 0.05f;
        std::printf("CountableBelief::self_test [wall explain-away] %s  gain wall=%+.2f obj=%+.2f | wall-resp %.2f vs %.2f\n",
                    ok_explain ? "PASS" : "FAIL", gain_wall, gain_countable, resp_wall, resp_countable);

    // ── (j) DOOR-MODE LOCK: a saturated incumbent must still be overtakeable ──────────────────────────
    // Replays the live failure (etc/ai2_log.csv): hundreds of frames of one static view drive the incumbent's
    // accumulator to the clamp; then the robot moves and sees the door on a DIFFERENT face. With the old
    // independent per-mode clamp the challenger was bounded by the same ±6 and had to be STRICTLY greater, so
    // 6.0 > 6.0 failed and the mode could never change — front_conf sat at 0.996 (= softmax[+6,0,−6,0]) for
    // thousands of cycles while the cue said "the door is 90° from where you think it is".
    bool ok_lock = false;
    {
        CountableBelief bl(CountableBeliefState{0.0f, 0.0f, 1.90f, 0.60f, 0.60f, 0.0f}, P);
        // Phase 1: 300 identical frames confirming the CURRENT mode (door on local −Y ⇒ bearing = yaw − π/2).
        for (int i = 0; i < 300; ++i)
            bl.resolve_front(FrontCue{bl.state().yaw - kHalfPi, 0.6f}, 1.0f);
        const float conf_locked = bl.front_confidence();
        const float yaw_before  = bl.state().yaw;
        // Phase 2: the robot moves and now sees the door on local +X (bearing = yaw). The belief MUST turn.
        bool flipped = false;
        for (int i = 0; i < 60 and not flipped; ++i)
            flipped = bl.resolve_front(FrontCue{bl.state().yaw, 0.6f}, 1.0f);
        const float turned = std::abs(std::remainder(bl.state().yaw - yaw_before, 2.0f * static_cast<float>(M_PI)));
        ok_lock = conf_locked > 0.9f and flipped and std::abs(turned - kHalfPi) < 0.05f;
        std::printf("CountableBelief::self_test [door-mode lock]    %s  saturated conf=%.3f → %s, turned %.1f°\n",
                    ok_lock ? "PASS" : "FAIL", conf_locked,
                    flipped ? "RECANTED" : "STUCK (locked)", turned * 57.29578f);
    }

    // ── (k) INFERRED VOLATILITY (MODEL_HISTORY.md §3) — the two properties any fix must have ─────────
    // The doc's own acceptance test: a STATIC object must show ω falling and its unobserved σ no longer
    // growing, while a MOVED object must still adapt within seconds. A Σ-clamp passes only the first; an
    // evidence-count stiffener only the second. Making Q itself inferred is what moves both.
    bool ok_vol = false;
    {
        CountableBeliefParams Pv = P;
        Pv.volatility_infer = true;
        Pv.volatility_lr = 0.05f;

        // Drive ω directly with the same update the filter applies, so the test exercises the real code.
        const auto run = [&](float step_m, int frames) {
            CountableBelief b(CountableBeliefState{0.0f, 0.0f, 1.90f, 0.60f, 0.60f, 0.0f}, Pv);
            Eigen::Matrix<float, 6, 1> d = Eigen::Matrix<float, 6, 1>::Zero();
            for (int i = 0; i < frames; ++i) { d(3) = (i % 2 ? step_m : -step_m); b.update_volatility_for_test(d); }
            return b;
        };
        // (1) STATIC: the belief barely moves (0.2 mm jitter on w) for 600 measured frames.
        CountableBelief b_static = run(0.0002f, 600);
        // (2) MOVED: the bed is shoved — 4 cm steps.
        CountableBelief b_moved  = run(0.04f, 600);

        const float w0   = std::log(Pv.process_std_m * Pv.process_std_m);   // ω₀ for a length DOF
        const float ws   = b_static.log_volatility()(3);
        const float wm   = b_moved.log_volatility()(3);
        // Retention: frames to double σ_w², τ = Σ/Q. Report the RATIO against the constant-Q baseline.
        const float tau_ratio_static = std::exp(w0 - ws);
        const float tau_ratio_moved  = std::exp(w0 - wm);
        ok_vol = ws < w0 - 1.0f and wm > w0 + 1.0f and tau_ratio_static > 3.0f and tau_ratio_moved < 0.5f;
        std::printf("CountableBelief::self_test [volatility]        %s  omega0=%.2f | static->%.2f (retention x%.1f) | moved->%.2f (retention x%.2f)\n",
                    ok_vol ? "PASS" : "FAIL", w0, ws, tau_ratio_static, wm, tau_ratio_moved);
    }

        return ok and ok_wall and ok_free and ok_nocross and ok_front and
               ok_height and ok_explain and ok_lock and ok_vol;
    }
}

}  // namespace rc
