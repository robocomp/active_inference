/*
 * cabinet_belief.cpp — the cabinet-RUN generative model (see cabinet_belief.h for the rationale).
 */

#include "cabinet_belief.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <print>

namespace rc
{

namespace
{
constexpr float kMinExtent = 0.05f;   // m — a run/carcass thinner than this is not a physical cabinet
constexpr float kJClamp    = 4.0f;    // finite-difference slope clamp (dimensionless d(sdf)/d(param))

inline float wrap_pi(float a)
{
    while (a >   std::numbers::pi_v<float>) a -= 2.0f * std::numbers::pi_v<float>;
    while (a <= -std::numbers::pi_v<float>) a += 2.0f * std::numbers::pi_v<float>;
    return a;
}
}  // namespace

// ─── Geometry: one oriented box ──────────────────────────────────────────────────────────────────

// Exact oriented-box SDF. Local frame: x along the run's long axis, y along its front normal, z up.
float CabinetBelief::sdf_box(const Eigen::Vector3f& p, const CabinetBeliefState& s) const
{
    const Eigen::Vector2f u = s.axis(), n = s.normal();
    const Eigen::Vector2f dxy(p.x() - s.cx, p.y() - s.cy);
    const Eigen::Vector3f q(dxy.dot(u), dxy.dot(n), p.z() - 0.5f * (s.z0 + s.z1));
    const Eigen::Vector3f half(0.5f * std::max(kMinExtent, s.L),
                               0.5f * std::max(kMinExtent, s.d),
                               0.5f * std::max(kMinExtent, s.height()));
    const Eigen::Vector3f e = q.cwiseAbs() - half;
    const float outside = e.cwiseMax(0.0f).norm();
    const float inside  = std::min(e.maxCoeff(), 0.0f);
    return outside + inside;
}

float CabinetBelief::sdf_prim(const Eigen::Vector3f& p, const CabinetBeliefState& s, int) const
{
    return sdf_box(p, s);   // a run has exactly one primitive (no legs — contrast the table)
}

Eigen::Matrix<float, 7, 1> CabinetBelief::sdf_jacobian(const Eigen::Vector3f& p,
                                                       const CabinetBeliefState& s, int prim) const
{
    Eigen::Matrix<float, 7, 1> J;
    const Eigen::Matrix<float, 7, 1> v = s.vec();
    const float eps = params_.fd_eps;
    for (int k = 0; k < 7; ++k)
    {
        Eigen::Matrix<float, 7, 1> vp = v, vm = v;
        vp(k) += eps; vm(k) -= eps;
        const float dp = sdf_prim(p, CabinetBeliefState::from_vec(vp), prim);
        const float dm = sdf_prim(p, CabinetBeliefState::from_vec(vm), prim);
        J(k) = std::clamp((dp - dm) / (2.0f * eps), -kJClamp, kJClamp);
    }
    return J;
}

// ─── Mixture: [box, clutter] ─────────────────────────────────────────────────────────────────────

float CabinetBelief::mixture_unnormalized(const Eigen::Vector3f& p, const CabinetBeliefState& s,
                                          float R, std::array<float, 2>& u) const
{
    const float eps  = std::clamp(params_.clutter_frac, 0.0f, 0.99f);
    const float dist = sdf_box(p, s);
    const float Rc   = std::max(1e-9f, R);
    const float inv_sqrt = 1.0f / std::sqrt(2.0f * std::numbers::pi_v<float> * Rc);
    u[0] = (1.0f - eps) * inv_sqrt * std::exp(-0.5f * dist * dist / Rc);
    u[1] = eps / std::max(1e-6f, params_.clutter_scale_m);   // uniform density over the clutter scale
    return u[0] + u[1];
}

std::array<float, 2> CabinetBelief::responsibilities(const Eigen::Vector3f& p,
                                                     const CabinetBeliefState& s, float R) const
{
    std::array<float, 2> u{};
    const float sum = mixture_unnormalized(p, s, R, u);
    if (not(sum > 0.0f) or not std::isfinite(sum)) return {0.0f, 1.0f};
    return {u[0] / sum, u[1] / sum};
}

// TRUE −log mixture MARGINAL likelihood (the clutter component INCLUDED). This is the quantity that
// rises on a bad fit; a responsibility-weighted surface energy reads ≈0 on a bad fit and silently
// breaks any hypothesis comparison. resolve_tier() compares two models with it, so it must be the
// honest marginal. (See the belief invariants in CONCEPT_AGENT_RECIPE.md.)
float CabinetBelief::mean_energy(const std::vector<Eigen::Vector3f>& pts,
                                 const CabinetBeliefState& s, float R) const
{
    if (pts.empty()) return 0.0f;
    double acc = 0.0;
    std::array<float, 2> u{};
    for (const auto& p : pts)
        acc += -std::log(std::max(1e-30f, mixture_unnormalized(p, s, R, u)));
    return static_cast<float>(acc / static_cast<double>(pts.size()));
}

// ─── Constraints and the C2v yaw fold ────────────────────────────────────────────────────────────

void CabinetBelief::apply_constraints(CabinetBeliefState& s) const
{
    s.L   = std::max(kMinExtent, s.L);
    s.d   = std::max(kMinExtent, s.d);
    s.yaw = wrap_pi(s.yaw);
    if (s.z1 < s.z0 + kMinExtent)                    // keep a physical, positively-oriented height
    {
        const float mid = 0.5f * (s.z0 + s.z1);
        s.z0 = mid - 0.5f * kMinExtent;
        s.z1 = mid + 0.5f * kMinExtent;
    }
    s.z0 = std::max(0.0f, s.z0);                     // a carcass does not sink through the floor
}

// A box is C2v: yaw and yaw+pi produce an identical SDF, so the raw MAP can sit on either. Fold
// deterministically by requiring the FRONT normal to point into the room. This is a geometric
// disambiguation against a known reference (the room polygon centroid), not a threshold — and it is
// what makes `d` mean "front face → back face" rather than an arbitrary sign.
void CabinetBelief::canonicalize(CabinetBeliefState& s) const
{
    s.yaw = wrap_pi(s.yaw);
    if (not has_room_interior_) return;
    const Eigen::Vector2f to_room = room_interior_ - s.centre();
    if (to_room.squaredNorm() < 1e-8f) return;
    if (s.normal().dot(to_room) < 0.0f)              // front normal points away from the room ⇒ flip
        s.yaw = wrap_pi(s.yaw + std::numbers::pi_v<float>);
}

// ─── Transition / prior / common-mode diagonals ──────────────────────────────────────────────────

Eigen::Matrix<float, 7, 1> CabinetBelief::process_noise_diag() const
{
    const float pm = params_.process_std_m * params_.process_std_m;
    const float py = params_.process_std_yaw * params_.process_std_yaw;
    Eigen::Matrix<float, 7, 1> q;
    q << pm, pm, py, pm, pm, pm, pm;   // cx cy yaw L d z0 z1
    return q;
}

// The tier's carcass priors enter HERE (and through the birth seed), not as a per-frame factor: in a
// recursive filter the prior is folded into Σ once, so re-applying it every frame would multiply its
// strength by the frame count. A tight prior_cov entry means "the data must work hard to move this",
// which is exactly the standard-dimension prior a kitchen supplies.
Eigen::Matrix<float, 7, 1> CabinetBelief::prior_cov_diag() const
{
    const auto& tp = tier_prior();
    Eigen::Matrix<float, 7, 1> p;
    p << params_.prior_pos_std * params_.prior_pos_std,
         params_.prior_pos_std * params_.prior_pos_std,
         params_.prior_yaw_std * params_.prior_yaw_std,
         params_.prior_L_std   * params_.prior_L_std,
         tp.d_std  * tp.d_std,
         tp.z0_std * tp.z0_std,
         tp.z1_std * tp.z1_std;
    return p;
}

Eigen::Matrix<float, 7, 1> CabinetBelief::prior_mean_vec() const
{
    const auto& tp = tier_prior();
    Eigen::Matrix<float, 7, 1> m;
    m << 0.0f, 0.0f, 0.0f, params_.prior_L_mean, tp.d_mean, tp.z0_mean, tp.z1_mean;
    return m;
}

Eigen::Matrix<float, 7, 1> CabinetBelief::common_mode_inv_diag(const CabinetFrame& f) const
{
    const float pos  = params_.common_mode_pos_std  * params_.common_mode_pos_std;
    const float size = params_.common_mode_size_std * params_.common_mode_size_std
                     + std::max(0.0f, f.chain_cov_size);
    const float yaw  = params_.common_mode_yaw_std  * params_.common_mode_yaw_std
                     + std::max(0.0f, f.chain_cov_yaw);
    const float sx = pos + std::max(0.0f, f.chain_cov_xx);
    const float sy = pos + std::max(0.0f, f.chain_cov_yy);
    Eigen::Matrix<float, 7, 1> c;
    c << 1.0f / std::max(1e-9f, sx),   1.0f / std::max(1e-9f, sy),
         1.0f / std::max(1e-9f, yaw),  1.0f / std::max(1e-9f, size),
         1.0f / std::max(1e-9f, size), 1.0f / std::max(1e-9f, size),
         1.0f / std::max(1e-9f, size);
    return c;
}

// ─── The wall-flush structural factor ────────────────────────────────────────────────────────────

// A cabinet run's BACK face is never observed, so the pair (position-along-normal, depth) has a null
// direction: sliding the centre back by δ while growing d by 2δ leaves the observed front face
// unchanged. The wall breaks it — a run is built flush against it.
//
// The precision is NOT a proximity gate. Model the run as a 2-component mixture {flush against this
// wall, free-standing}; with a Gaussian on the gap, the posterior weight of the flush component is
// exp(−(gap/reach)²), and marginalising the discrete component multiplies the factor's precision by
// exactly that weight. So the term fades smoothly to nothing for a genuine island (the apartamento
// kitchen has one: Worktop3 sits 2.17 m / 1.18 m off both walls) and `d` stays honestly wide there.
float CabinetBelief::flush_weight(const CabinetBeliefState& s, const CabinetFrame& f) const
{
    if (not f.wall.ok) return 0.0f;
    const float gap = (s.back_centre() - f.wall.p).dot(f.wall.n);
    const float rel = gap / std::max(1e-3f, params_.wall_reach_m);
    return std::exp(-rel * rel);
}

void CabinetBelief::accumulate_wall(const CabinetBeliefState& s, const CabinetFrame& f,
                                    Eigen::Matrix<float, 7, 7>& Id, Eigen::Matrix<float, 7, 1>& bd) const
{
    dbg_wall_gap_ = 0.0f; dbg_wall_lambda_ = 0.0f;
    if (not f.wall.ok or params_.wall_precision <= 0.0f) return;

    const auto gap_of = [&](const CabinetBeliefState& st)
    { return (st.back_centre() - f.wall.p).dot(f.wall.n); };
    const auto par_of = [&](const CabinetBeliefState& st)
    { return st.axis().dot(f.wall.n); };            // 0 ⇔ run axis parallel to the wall

    const float gap = gap_of(s);
    dbg_wall_gap_ = gap;

    // Posterior weight of the "flush against this wall" component (shared definition), and the wall's own
    // uncertainty. accumulate_extent uses the SAME weight for the wall-segment domain terms.
    const float wgt = flush_weight(s, f);
    const float lam_flush = wgt / (1.0f / params_.wall_precision + f.wall.sigma_m * f.wall.sigma_m);
    dbg_wall_lambda_ = lam_flush;
    if (not(lam_flush > 1e-6f)) return;

    const float eps = params_.fd_eps;
    const Eigen::Matrix<float, 7, 1> v = s.vec();
    Eigen::Matrix<float, 7, 1> Jg, Jp;
    for (int k = 0; k < 7; ++k)
    {
        Eigen::Matrix<float, 7, 1> vp = v, vm = v;
        vp(k) += eps; vm(k) -= eps;
        const auto sp = CabinetBeliefState::from_vec(vp), sm = CabinetBeliefState::from_vec(vm);
        Jg(k) = std::clamp((gap_of(sp) - gap_of(sm)) / (2.0f * eps), -kJClamp, kJClamp);
        Jp(k) = std::clamp((par_of(sp) - par_of(sm)) / (2.0f * eps), -kJClamp, kJClamp);
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

// ─── Along-axis CONTAINMENT (the factor that lets a run GROW) ────────────────────────────────────

// The per-point mixture cedes any point past the current end cap to clutter, so it structurally
// cannot lengthen a run (see the header note). The global statistic that can is the along-axis SPAN
// of the points compatible with the run's cross-section.
//
// It is folded in as a CENSORED likelihood: an observed span is a LOWER BOUND on the true extent
// (occlusion/foreshortening only ever shorten it), so each end contributes a HINGE residual —
// non-zero only while the box fails to contain the observed cloud, and identically zero once it
// does. That one-sidedness is the measurement model, not a tuning choice, and it is why the factor
// cannot be used to shrink a run: retracting the extent is the free-space/vacate channel's job.
//
// Both ends are constrained SEPARATELY (rather than fitting L to the span) so the same factor moves
// the centre as well as the length — a run first seen from one end grows toward the other.
void CabinetBelief::accumulate_extent(const CabinetBeliefState& s, const CabinetFrame& f,
                                      Eigen::Matrix<float, 7, 7>& Id, Eigen::Matrix<float, 7, 1>& bd) const
{
    dbg_span_obs_ = 0.0f; dbg_span_pts_ = 0;
    if (params_.extent_precision <= 0.0f or f.points.empty()) return;

    // Cross-section compatibility: a SOFT weight on how far the point sits outside the run's depth
    // slab and height band (zero distance ⇒ weight 1, falling off over the observation sigma). Not a
    // gate — a point marginally outside still votes, just less.
    const Eigen::Vector2f u = s.axis(), n = s.normal();
    const float sig  = std::max(1e-3f, params_.sigma_base_m);
    const float halfd = 0.5f * std::max(kMinExtent, s.d);
    const float zc = 0.5f * (s.z0 + s.z1), halfh = 0.5f * std::max(kMinExtent, s.height());

    std::vector<std::pair<float, float>> sw;   // (along-axis coord, weight)
    sw.reserve(f.points.size());
    double wsum = 0.0;
    for (const auto& p : f.points)
    {
        const Eigen::Vector2f dxy(p.x() - s.cx, p.y() - s.cy);
        const float lat  = std::max(0.0f, std::abs(dxy.dot(n)) - halfd);
        const float vert = std::max(0.0f, std::abs(p.z() - zc) - halfh);
        const float w = std::exp(-0.5f * (lat * lat + vert * vert) / (sig * sig));
        if (w < 1e-3f) continue;
        sw.emplace_back(dxy.dot(u), w);
        wsum += w;
    }
    if (sw.size() < 8 or wsum <= 0.0) return;
    dbg_span_pts_ = static_cast<int>(sw.size());

    // Robust span ends via weighted order statistics (a quantile, so a single stray return cannot
    // stretch the run — the same role the Cauchy scale plays in the LiDAR factor).
    std::sort(sw.begin(), sw.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    const double q = std::clamp(params_.extent_quantile, 0.0f, 0.45f) * wsum;
    const auto pick = [&](bool from_low)
    {
        double acc = 0.0;
        if (from_low) { for (const auto& e : sw) { acc += e.second; if (acc >= q) return e.first; } }
        else          { for (auto it = sw.rbegin(); it != sw.rend(); ++it) { acc += it->second; if (acc >= q) return it->first; } }
        return from_low ? sw.front().first : sw.back().first;
    };
    const float s_lo = pick(true), s_hi = pick(false);
    dbg_span_obs_ = s_hi - s_lo;

    const Eigen::Vector2f c(s.cx, s.cy);

    // ── Wall-segment domain (corner-aware growth) ────────────────────────────────────────────────
    // A run built flush against a wall physically ENDS at that wall segment's corners. nearest_wall
    // supplies them (collinear-merged). Two one-sided effects, both scaled by the flush mixture weight so
    // they vanish continuously for an island / when no polygon is known (no gate, no `if (near_wall)`):
    //   (1) CENSOR the observed span at the corners — points past a corner belong to the perpendicular
    //       run, not this one, so they must not be evidence FOR its length (the corrected conditioning of
    //       the censored lower-bound invariant). This PREVENTS growing across the corner.
    //   (2) A symmetric INWARD hinge that fires when a box END already sticks out past its corner, pulling
    //       it back. This RETRACTS an over-grown run (the only channel that can — the space beyond the
    //       corner is occupied by the other run, so free-space evidence never arrives there).
    float s_lo_eff = s_lo, s_hi_eff = s_hi;
    const float wgt = flush_weight(s, f);
    const bool  seg = f.wall.has_segment and wgt > 1e-3f;
    Eigen::Vector2f P_lo_seg = Eigen::Vector2f::Zero(), P_hi_seg = Eigen::Vector2f::Zero();
    if (seg)
    {
        const float t_a = (f.wall.a - c).dot(u), t_b = (f.wall.b - c).dot(u);
        const float t_lo_w = std::min(t_a, t_b), t_hi_w = std::max(t_a, t_b);
        P_lo_seg = (t_a <= t_b) ? f.wall.a : f.wall.b;   // fixed room corners on the −u / +u ends
        P_hi_seg = (t_a <= t_b) ? f.wall.b : f.wall.a;
        s_hi_eff = s_hi - wgt * std::max(0.0f, s_hi - t_hi_w);
        s_lo_eff = s_lo + wgt * std::max(0.0f, t_lo_w - s_lo);
    }

    // Re-express the (censored) extreme observations as FIXED room-frame points so the residual is a pure
    // function of the state (the along-axis coordinate depends on cx,cy,yaw and must not be frozen in it).
    const Eigen::Vector2f p_lo = c + u * s_lo_eff, p_hi = c + u * s_hi_eff;

    // GROW hinges: how far each (censored) observed end sticks OUT of the box along the axis.
    const auto r_lo_of = [&](const CabinetBeliefState& st)
    { return std::max(0.0f, (st.centre() - p_lo).dot(st.axis()) - 0.5f * st.L); };
    const auto r_hi_of = [&](const CabinetBeliefState& st)
    { return std::max(0.0f, (p_hi - st.centre()).dot(st.axis()) - 0.5f * st.L); };
    // RETRACT hinges: how far each box END extends PAST its wall corner (silent inside the segment).
    const auto q_lo_of = [&](const CabinetBeliefState& st)
    { return seg ? std::max(0.0f, (P_lo_seg - st.centre()).dot(st.axis()) + 0.5f * st.L) : 0.0f; };
    const auto q_hi_of = [&](const CabinetBeliefState& st)
    { return seg ? std::max(0.0f, (st.centre() - P_hi_seg).dot(st.axis()) + 0.5f * st.L) : 0.0f; };

    const float r_lo = r_lo_of(s), r_hi = r_hi_of(s);
    const float q_lo = q_lo_of(s), q_hi = q_hi_of(s);
    // Instrumentation: is the wall-segment domain actually binding? (corner projections + retract residuals)
    dbg_seg_active_ = seg ? 1 : 0;
    if (seg) { const float t_a = (f.wall.a - c).dot(u), t_b = (f.wall.b - c).dot(u);
               dbg_seg_tlo_ = std::min(t_a, t_b); dbg_seg_thi_ = std::max(t_a, t_b); }
    dbg_seg_shi_ = s_hi; dbg_seg_qlo_ = q_lo; dbg_seg_qhi_ = q_hi;
    if (r_lo <= 0.0f and r_hi <= 0.0f and q_lo <= 0.0f and q_hi <= 0.0f) return;   // fully contained ⇒ silent

    const float lam = params_.extent_precision;
    // Retract precision folds the wall's own position uncertainty (like accumulate_wall's lam_flush).
    const float lam_seg = seg ? lam * wgt / (1.0f + f.wall.sigma_m * f.wall.sigma_m * lam) : 0.0f;
    const float eps = params_.fd_eps;
    const Eigen::Matrix<float, 7, 1> v = s.vec();
    Eigen::Matrix<float, 7, 1> Jlo, Jhi, Qlo, Qhi;
    for (int k = 0; k < 7; ++k)
    {
        Eigen::Matrix<float, 7, 1> vp = v, vm = v;
        vp(k) += eps; vm(k) -= eps;
        const auto sp = CabinetBeliefState::from_vec(vp), sm = CabinetBeliefState::from_vec(vm);
        Jlo(k) = std::clamp((r_lo_of(sp) - r_lo_of(sm)) / (2.0f * eps), -kJClamp, kJClamp);
        Jhi(k) = std::clamp((r_hi_of(sp) - r_hi_of(sm)) / (2.0f * eps), -kJClamp, kJClamp);
        Qlo(k) = std::clamp((q_lo_of(sp) - q_lo_of(sm)) / (2.0f * eps), -kJClamp, kJClamp);
        Qhi(k) = std::clamp((q_hi_of(sp) - q_hi_of(sm)) / (2.0f * eps), -kJClamp, kJClamp);
    }
    if (r_lo > 0.0f) { Id.noalias() += lam     * (Jlo * Jlo.transpose()); bd.noalias() += -lam     * Jlo * r_lo; }
    if (r_hi > 0.0f) { Id.noalias() += lam     * (Jhi * Jhi.transpose()); bd.noalias() += -lam     * Jhi * r_hi; }
    if (q_lo > 0.0f) { Id.noalias() += lam_seg * (Qlo * Qlo.transpose()); bd.noalias() += -lam_seg * Qlo * q_lo; }
    if (q_hi > 0.0f) { Id.noalias() += lam_seg * (Qhi * Qhi.transpose()); bd.noalias() += -lam_seg * Qhi * q_hi; }
}

// ─── Room-axis (Manhattan) yaw alignment ──────────────────────────────────────────────────────────
//
// Pull yaw to the NEAREST room axis (a multiple of π/2 in the room frame — a kitchen run is built
// parallel to a wall). Unlike accumulate_wall's parallel term, this needs no detected wall and no
// flush weight, so it also aligns a peninsula/island that only touches a wall on one short side, a
// merged/mis-positioned run, or a run whose wall isn't currently observed. The residual is the signed
// angular error to that axis (in [-π/4, π/4]); it is a pure yaw term (∂r/∂yaw = 1, all other partials
// zero), so no finite differences are needed. Silent once aligned (r→0): it biases orientation without
// fighting a correctly-aligned run, and — being a room-frame axis — it is invariant to canonicalize's
// 180° C2v flip. Per-frame like the other structural factors; predict-step process noise on yaw bounds
// the information it accumulates.
void CabinetBelief::accumulate_axis_alignment(const CabinetBeliefState& s,
                                              Eigen::Matrix<float, 7, 7>& Id, Eigen::Matrix<float, 7, 1>& bd) const
{
    dbg_axis_resid_ = 0.0f;
    if (params_.room_axis_precision <= 0.0f) return;
    constexpr float kQuarter = std::numbers::pi_v<float> / 2.0f;
    const float nearest = std::round(s.yaw / kQuarter) * kQuarter;   // nearest room axis
    const float r = wrap_pi(s.yaw - nearest);                        // signed error ∈ [-π/4, π/4]
    dbg_axis_resid_ = r;
    // Capture range: outside it, release the pull so a genuinely oblique object is left alone.
    // ≤0 or ≥π/4 means "always on" (every yaw is within π/4 of some axis).
    if (params_.room_axis_capture_rad > 0.0f and std::abs(r) > params_.room_axis_capture_rad) return;
    const float lam = params_.room_axis_precision;
    Id(2, 2) += lam;              // yaw is state index 2 [cx,cy,yaw,L,d,z0,z1]
    bd(2)    += -lam * r;
}

// ─── Free-space / VACATE: the occlusion-aware UPPER bound that closes accumulate_extent ──────────────
//
// A LiDAR beam that TRAVERSES the carcass box and lands BEYOND its far face proves that crossing is empty.
// Its midpoint is a witness point inside the footprint; driving the 2-D footprint SDF→0 there RETREATS the
// nearest face just past it (shrink-only). p_through = Φ((1−t_far)·len/σ_surf) is the mixture posterior of
// the "beam exited the far face" component vs "beam returned from the surface" — continuous, no gate, and
// occlusion-aware BY GEOMETRY: a short return (occluder) has t_far>1 ⇒ p_through→0 ⇒ no witness past it, so
// a legitimately occluded extension is never carved (it just ages). The full solid carcass [z0,z1] is the
// carve volume (no legs, no top-slab z-gate — a cabinet is solid). Correlated beams saturate through the
// engine's common-mode Woodbury, same as the range factor. Ported from table_belief's vacate term.
void CabinetBelief::accumulate_freespace(const CabinetBeliefState& s, const CabinetFrame& f,
                                         Eigen::Matrix<float, 7, 7>& Id, Eigen::Matrix<float, 7, 1>& bd) const
{
    dbg_vacate_beams_ = 0;
    if (params_.free_space_precision <= 0.0f or f.lidar_freespace.endpoints.empty())
        return;

    const rc::ai::LidarRays& rays = f.lidar_freespace;
    const float cyaw = std::cos(-s.yaw), syaw = std::sin(-s.yaw);
    const float hL = 0.5f * std::max(kMinExtent, s.L);
    const float hd = 0.5f * std::max(kMinExtent, s.d);
    const float lo[3] = {-hL, -hd, s.z0};      // local carcass box: full footprint, SOLID z-band [z0,z1]
    const float hi[3] = { hL,  hd, s.z1};
    const float sigma_surf = std::sqrt(params_.sigma_base_m * params_.sigma_base_m
                                     + params_.common_mode_pos_std * params_.common_mode_pos_std);
    const auto phi = [](float x) { return 0.5f * std::erfc(-x * 0.70710678f); };   // Φ(x)

    const Eigen::Vector3f& O = rays.origin;
    const float oxr = O.x() - s.cx, oyr = O.y() - s.cy;
    const Eigen::Vector3f Ol(oxr * cyaw - oyr * syaw, oxr * syaw + oyr * cyaw, O.z());   // sensor in local frame
    for (const auto& ep : rays.endpoints)
    {
        const float exr = ep.x() - s.cx, eyr = ep.y() - s.cy;
        const Eigen::Vector3f El(exr * cyaw - eyr * syaw, exr * syaw + eyr * cyaw, ep.z());
        const Eigen::Vector3f dloc = El - Ol;                       // local ray (t=1 at the endpoint)
        const float len = dloc.norm();
        if (len < 1e-4f) continue;

        float t_near = 0.0f, t_far = std::numeric_limits<float>::max();
        bool miss = false;
        for (int a = 0; a < 3 and not miss; ++a)
        {
            if (std::abs(dloc(a)) < 1e-6f) { if (Ol(a) < lo[a] or Ol(a) > hi[a]) miss = true; }
            else
            {
                float t1 = (lo[a] - Ol(a)) / dloc(a), t2 = (hi[a] - Ol(a)) / dloc(a);
                if (t1 > t2) std::swap(t1, t2);
                t_near = std::max(t_near, t1);
                t_far  = std::min(t_far,  t2);
            }
        }
        if (miss or t_near > t_far or t_far < 0.0f or t_near >= 1.0f) continue;

        const float p_through = phi((1.0f - t_far) / (sigma_surf / len));   // endpoint beyond far face ⇒ empty
        if (p_through < 1e-3f) continue;

        const float t_mid = 0.5f * (t_near + t_far);
        const Eigen::Vector3f p_free = O + t_mid * (ep - O);       // WORLD witness point (certified empty)

        // 2-D footprint SDF (L×d rectangle in the yaw frame, z ignored) — the 3-D box SDF is degenerate on
        // the thin faces (∂/∂L=∂/∂d=0), so use the lateral SDF to recover the L/d/cx/cy gradient. e<0 (witness
        // INSIDE the footprint) ⇒ driving e→0 retreats the nearest face just past it = shrink-only.
        const auto sdf_fp = [&](const CabinetBeliefState& st) -> float
        {
            const float cc = std::cos(-st.yaw), s2 = std::sin(-st.yaw);
            const float lx = (p_free.x() - st.cx) * cc - (p_free.y() - st.cy) * s2;
            const float ly = (p_free.x() - st.cx) * s2 + (p_free.y() - st.cy) * cc;
            const float dx = std::abs(lx) - 0.5f * st.L, dy = std::abs(ly) - 0.5f * st.d;
            const float ox = std::max(dx, 0.0f), oy = std::max(dy, 0.0f);
            return std::sqrt(ox * ox + oy * oy) + std::min(std::max(dx, dy), 0.0f);   // 2-D box SDF
        };
        const float e = sdf_fp(s);
        if (e >= 0.0f) continue;                                   // must be inside the footprint to shrink
        Eigen::Matrix<float, 7, 1> J;
        const Eigen::Matrix<float, 7, 1> base = s.vec();
        const float fde = params_.fd_eps;
        for (int j = 0; j < 7; ++j)
        {
            Eigen::Matrix<float, 7, 1> vp = base, vm = base; vp(j) += fde; vm(j) -= fde;
            J(j) = (sdf_fp(CabinetBeliefState::from_vec(vp)) - sdf_fp(CabinetBeliefState::from_vec(vm))) / (2.0f * fde);
        }
        const float w = params_.free_space_precision * p_through;
        Id.noalias() += w * (J * J.transpose());
        bd.noalias() += -w * J * e;
        ++dbg_vacate_beams_;
    }
}

void CabinetBelief::accumulate_extra(const CabinetBeliefState& s, const CabinetFrame& f,
                                     Eigen::Matrix<float, 7, 7>& Id, Eigen::Matrix<float, 7, 1>& bd) const
{
    accumulate_wall(s, f, Id, bd);
    accumulate_extent(s, f, Id, bd);
    accumulate_freespace(s, f, Id, bd);   // occlusion-aware UPPER bound closing the one-sided extent
    accumulate_axis_alignment(s, Id, bd);

    // YOLO-independent LiDAR first-hit range factor, once per device ray-set (each keeps its OWN
    // origin so the sphere-trace stays occlusion-aware per device — merging would lose that).
    dbg_lidar_rays_ = 0;
    if (f.lidar.precision > 0.0f and not f.lidar.endpoints.empty())
    {
        ai::accumulate_lidar_rays<7>(*this, s, f.lidar, Id, bd);
        dbg_lidar_rays_ += static_cast<int>(f.lidar.endpoints.size());
    }
    for (const auto& set : f.lidar_extra)
        if (set.precision > 0.0f and not set.endpoints.empty())
        {
            ai::accumulate_lidar_rays<7>(*this, s, set, Id, bd);
            dbg_lidar_rays_ += static_cast<int>(set.endpoints.size());
        }
}

// ─── Discrete TIER mode ──────────────────────────────────────────────────────────────────────────

float CabinetBelief::tier_posterior() const
{
    return 1.0f / (1.0f + std::exp(tier_evidence_));   // p(alternative tier)
}

// REPORTED covariance: Σ_ with the discrete TIER-mode entropy folded into the DOFs the tier actually
// loads — d (4), z0 (5), z1 (6). For each, add p(1−p)Δ² where p = tier_posterior() and Δ is the gap
// between the two tiers' prior means for that DOF (base vs wall). Undecided tier (p≈½) → an honest wide
// σ that straddles both carcasses; resolved (p→0) → collapses to Σ_. Diagonal-only (mirrors table/chair).
Eigen::Matrix<float, 7, 7> CabinetBelief::covariance_reported() const
{
    Eigen::Matrix<float, 7, 7> S = Sigma_;
    const float p  = tier_posterior();
    const float pe = std::max(0.0f, p * (1.0f - p));   // discrete-mode entropy weight, p∈[0,1] ⇒ ≥0
    const CabinetTierPrior& b = params_.base_tier;
    const CabinetTierPrior& w = params_.wall_tier;
    const float dd  = std::abs(b.d_mean  - w.d_mean);   // Δ: tier separation for each loaded DOF
    const float dz0 = std::abs(b.z0_mean - w.z0_mean);
    const float dz1 = std::abs(b.z1_mean - w.z1_mean);
    S(4, 4) += pe * dd  * dd;
    S(5, 5) += pe * dz0 * dz0;
    S(6, 6) += pe * dz1 * dz1;
    return S;
}

// Sequential Bayesian comparison of {current tier, other tier}. A Laplace filter cannot represent the
// genuinely BIMODAL prior on (z0,z1) that "base unit OR wall unit" is, so the discrete part rides
// outside the Gaussian — the same construction the table uses for its w<->h mode. Adopting a mode
// re-seeds the (d,z0,z1) block toward the new tier's carcass prior and re-inflates its Σ block,
// mirroring the table's swap of both the state and the corresponding rows/cols of Σ.
bool CabinetBelief::resolve_tier(const std::vector<Eigen::Vector3f>& pts, float R, float evidence_weight)
{
    if (pts.empty() or evidence_weight <= 0.0f) return false;

    const CabinetTier other = (tier_ == CabinetTier::Base) ? CabinetTier::Wall : CabinetTier::Base;
    const CabinetTierPrior& op = (other == CabinetTier::Base) ? params_.base_tier : params_.wall_tier;

    CabinetBeliefState alt = state_;      // the same run, re-seeded to the other tier's carcass
    alt.d = op.d_mean; alt.z0 = op.z0_mean; alt.z1 = op.z1_mean;

    const float e_now = mean_energy(pts, state_, R);
    const float e_alt = mean_energy(pts, alt,    R);

    // Positive evidence favours the CURRENT mode. Bounded accumulator: an honest forgetting factor so
    // a long agreeing history cannot make the mode unswitchable if the world changes.
    tier_evidence_ = std::clamp(tier_evidence_ + evidence_weight * (e_alt - e_now),
                                -params_.tier_evidence_cap, params_.tier_evidence_cap);
    if (tier_evidence_ >= 0.0f) return false;             // MAP over the mode: boundary at zero

    tier_ = other;
    tier_evidence_ = 0.0f;                                 // restart the comparison from the new mode
    state_.d = op.d_mean; state_.z0 = op.z0_mean; state_.z1 = op.z1_mean;
    const auto pc = prior_cov_diag();                      // re-inflate the carcass block's Σ
    for (int k = 4; k <= 6; ++k)
    {
        Sigma_.row(k).setZero(); Sigma_.col(k).setZero();
        Sigma_(k, k) = pc(k);
    }
    return true;
}

// ─── Birth seed: the run's own principal axis + extents ──────────────────────────────────────────

// The 2-D inertia tensor of the cloud is a sufficient statistic of the filled-rectangle model, so its
// principal axis IS the run's axis and the equivalent-uniform-rectangle extents (sqrt(12*lambda)) are
// (L, d). Seeding from it starts the box near the data; a small axis-aligned seed would route the real
// run to clutter and never grow into it. `aniso` reports how well-determined the axis is — a nearly
// square cloud has no meaningful principal direction and the caller should leave yaw to other evidence.
RunSeed CabinetBelief::seed_from_points(const std::vector<Eigen::Vector3f>& pts, bool room_axis_snap)
{
    RunSeed s;
    if (pts.size() < 8) return s;
    s.n = static_cast<int>(pts.size());

    double mx = 0.0, my = 0.0;
    for (const auto& p : pts) { mx += p.x(); my += p.y(); }
    mx /= pts.size(); my /= pts.size();
    s.cx = static_cast<float>(mx); s.cy = static_cast<float>(my);

    double sxx = 0.0, syy = 0.0, sxy = 0.0;
    for (const auto& p : pts)
    {
        const double dx = p.x() - mx, dy = p.y() - my;
        sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
    }
    sxx /= pts.size(); syy /= pts.size(); sxy /= pts.size();

    if (room_axis_snap)
    {
        // Room-axis seed: a kitchen run lies on a ROOM axis, so pick the axis (X or Y) the cloud spreads
        // along MOST as the run's long axis — never the PCA diagonal. For an L-shaped corner mask the raw
        // PCA axis is diagonal and its major extent spans BOTH walls, which births one oblique box the
        // grow-only extent factor can never retract; the marginal room-axis variance instead measures the
        // dominant arm alone (the perpendicular arm contributes only its narrow off-axis position spread).
        const bool along_x = sxx >= syy;
        s.yaw = along_x ? 0.0f : static_cast<float>(std::numbers::pi_v<double> * 0.5);
        s.L = static_cast<float>(std::sqrt(std::max(0.0, 12.0 * (along_x ? sxx : syy))));
        s.d = static_cast<float>(std::sqrt(std::max(0.0, 12.0 * (along_x ? syy : sxx))));
        // aniso = how much more the cloud spreads along the dominant room axis than the other. Low for a
        // near-symmetric L ⇒ the caller leaves yaw to the wall/Manhattan factors rather than guessing an arm.
        const double sum2 = sxx + syy;
        s.aniso = (sum2 > 1e-8) ? static_cast<float>(std::abs(sxx - syy) / sum2) : 0.0f;
    }
    else
    {
        // Principal axis + eigenvalues of the 2x2 covariance.
        const double tr = sxx + syy, det = sxx * syy - sxy * sxy;
        const double disc = std::sqrt(std::max(0.0, 0.25 * tr * tr - det));
        const double l1 = 0.5 * tr + disc, l2 = std::max(0.0, 0.5 * tr - disc);
        s.yaw = static_cast<float>(0.5 * std::atan2(2.0 * sxy, sxx - syy));
        s.L = static_cast<float>(std::sqrt(std::max(0.0, 12.0 * l1)));   // equivalent uniform rectangle
        s.d = static_cast<float>(std::sqrt(std::max(0.0, 12.0 * l2)));
        const float sum = s.L + s.d;
        s.aniso = (sum > 1e-4f) ? (s.L - s.d) / sum : 0.0f;
    }

    // Robust vertical band (2% / 98%) — the carcass top/bottom, immune to a few stray returns.
    std::vector<float> zs; zs.reserve(pts.size());
    for (const auto& p : pts) zs.push_back(p.z());
    std::sort(zs.begin(), zs.end());
    const std::size_t lo = static_cast<std::size_t>(0.02f * (zs.size() - 1));
    const std::size_t hi = static_cast<std::size_t>(0.98f * (zs.size() - 1));
    s.z0 = zs[lo]; s.z1 = zs[hi];

    s.ok = true;
    return s;
}

// ─── Verification ────────────────────────────────────────────────────────────────────────────────

bool CabinetBelief::self_test()
{
    bool ok = true;
    const auto check = [&](bool c, const char* what)
    { if (not c) { std::print("[cabinet_belief::self_test] FAIL: {}\n", what); ok = false; } };

    // Ground truth: a 3.0 m base run, 0.60 m deep, 0.90 m tall, along +x, centred at (0, 0.7).
    // Its back face is at y = 0.4 and there is a wall there with inward normal +y.
    CabinetBeliefState gt;
    gt.cx = 0.0f; gt.cy = 0.7f; gt.yaw = 0.0f; gt.L = 3.0f; gt.d = 0.6f; gt.z0 = 0.0f; gt.z1 = 0.9f;

    const CabinetBeliefParams pr;
    const CabinetBelief ref(gt, pr);
    const float R = pr.sigma_base_m * pr.sigma_base_m;

    // ── SDF sanity ────────────────────────────────────────────────────────────────────────────
    check(std::abs(ref.sdf_box({0.0f, 0.7f, 0.45f}, gt) + 0.30f) < 1e-3f, "centre SDF = -half-depth");
    check(std::abs(ref.sdf_box({0.0f, 1.0f, 0.45f}, gt)) < 1e-3f,         "front face SDF = 0");
    check(ref.sdf_box({0.0f, 1.5f, 0.45f}, gt) > 0.45f,                   "outside SDF positive");

    // ── Sample the OBSERVABLE faces only: front, top and the two ends. The back face is never
    //    seen (it is against the wall) — that is the whole identifiability problem.
    std::vector<Eigen::Vector3f> pts;
    for (int i = 0; i <= 60; ++i)
    {
        const float t = -1.5f + 3.0f * static_cast<float>(i) / 60.0f;
        for (int j = 0; j <= 9; ++j)
            pts.emplace_back(t, 1.0f, 0.9f * static_cast<float>(j) / 9.0f);   // front face (y = cy + d/2)
        pts.emplace_back(t, 0.7f, 0.9f);                                      // top face
    }
    for (int j = 0; j <= 9; ++j)                                              // the two end caps
    {
        const float z = 0.9f * static_cast<float>(j) / 9.0f;
        pts.emplace_back(-1.5f, 0.85f, z);
        pts.emplace_back( 1.5f, 0.85f, z);
    }

    const auto fit_from = [&](bool with_wall, const CabinetBeliefState& seed)
    {
        CabinetBelief b(seed, pr);
        b.set_room_interior({0.0f, 5.0f});                    // the room is on the +y side
        for (int it = 0; it < 25; ++it)
        {
            CabinetFrame f;
            f.points = pts;
            if (with_wall) { f.wall.ok = true; f.wall.p = {0.0f, 0.4f}; f.wall.n = {0.0f, 1.0f}; }
            b.update(f);
        }
        return b;
    };

    // Seed with a DELIBERATELY WRONG depth (0.35 vs the true 0.60) and a short, offset, yawed run,
    // so the test measures recovery rather than a lucky initialisation.
    CabinetBeliefState seed;
    seed.cx = 0.15f; seed.cy = 0.85f; seed.yaw = 0.08f; seed.L = 1.2f;
    seed.d = 0.35f; seed.z0 = 0.0f; seed.z1 = 0.9f;

    // ── (a) With the wall the whole run is recovered, INCLUDING the never-observed depth ───────
    const auto bw = fit_from(true, seed);
    const auto sw = bw.state();
    check(std::abs(sw.cx - gt.cx) < 0.15f, "wall fit recovers cx");
    check(std::abs(sw.cy - gt.cy) < 0.15f, "wall fit recovers cy");
    check(std::abs(wrap_pi(sw.yaw - gt.yaw)) < 0.15f, "wall fit recovers yaw");
    check(std::abs(sw.L - gt.L) < 0.30f, "wall fit grows L to the observed span");
    check(std::abs(sw.d - gt.d) < 0.10f, "wall fit recovers depth d from a WRONG seed");
    check(std::abs(sw.z1 - gt.z1) < 0.12f, "wall fit recovers top height");

    // ── (b) Without a wall, (centre-along-normal, depth) is genuinely degenerate. The correct
    //    behaviour is NOT to recover d — it is to fit everything that IS observed and leave the
    //    unobservable combination where the prior put it, with an honestly wider σ. Concretely the
    //    FRONT FACE (cy + d/2) must still land on the data while cy and d individually do not.
    //    Any implementation that "recovers" d here would be manufacturing information.
    const auto bn = fit_from(false, seed);
    const auto sn = bn.state();
    const float front_obs = gt.cy + 0.5f * gt.d;            // = 1.0 m, where the points actually are
    check(std::abs((sn.cy + 0.5f * sn.d) - front_obs) < 0.05f,
          "free-standing fit still nails the OBSERVED front face");
    check(std::abs(sn.d - gt.d) > 0.10f,
          "free-standing fit does NOT fake knowledge of the unobserved depth");
    const float sd_wall = std::sqrt(std::max(0.0f, bw.covariance()(4, 4)));
    const float sd_free = std::sqrt(std::max(0.0f, bn.covariance()(4, 4)));
    check(sd_free > 1.25f * sd_wall, "depth σ is wider without a wall (degeneracy stays honest)");
    check(std::isfinite(sn.d) and sn.d > kMinExtent, "free-standing depth stays physical");

    // ── (b2) THE ISLAND. The apartamento kitchen contains a genuine free-standing run (Worktop3,
    //    2.17 m / 1.18 m off both walls). With a wall present but far outside its reach, the flush
    //    factor must fade to nothing ON ITS OWN — the mixture weight exp(−(gap/reach)²) does it, so
    //    there is no `if (near_wall)` anywhere. The island must fit like the no-wall case.
    {
        CabinetBelief b(seed, pr);
        b.set_room_interior({0.0f, 5.0f});
        for (int it = 0; it < 25; ++it)
        {
            CabinetFrame f;
            f.points = pts;
            f.wall.ok = true; f.wall.p = {0.0f, -1.5f}; f.wall.n = {0.0f, 1.0f};   // ~1.9 m behind
            b.update(f);
        }
        check(b.last_wall_lambda() < 1e-3f * pr.wall_precision, "a distant wall's flush precision decays to ~0");
        check(std::abs(b.state().d - bn.state().d) < 0.05f, "island fits like the no-wall case");
    }

    // ── (c) The free energy must RISE on a deliberately wrong fit (true mixture marginal) ──────
    CabinetBeliefState bad = gt; bad.cx += 1.2f; bad.yaw += 0.9f;
    check(ref.mean_energy(pts, bad, R) > ref.mean_energy(pts, gt, R), "free energy rises on a wrong fit");

    // ── (d) canonicalize() folds the 180° C2v ambiguity toward the room ────────────────────────
    {
        CabinetBelief b(gt, pr);
        b.set_room_interior({0.0f, 5.0f});
        CabinetBeliefState flipped = gt;
        flipped.yaw = wrap_pi(gt.yaw + std::numbers::pi_v<float>);
        b.canonicalize(flipped);
        check(flipped.normal().y() > 0.0f, "canonicalize points the front normal into the room");
    }

    // ── (e) The tier mode prefers the tier whose carcass actually explains the cloud ───────────
    {
        CabinetBeliefState wall_seed = gt;                    // same run, but seeded as a WALL unit
        wall_seed.d  = pr.wall_tier.d_mean;
        wall_seed.z0 = pr.wall_tier.z0_mean; wall_seed.z1 = pr.wall_tier.z1_mean;
        CabinetBelief b(wall_seed, pr);
        b.set_tier(CabinetTier::Wall);
        bool switched = false;
        for (int it = 0; it < 12 and not switched; ++it)
            switched = b.resolve_tier(pts, R, 1.0f);
        check(switched and b.tier() == CabinetTier::Base, "tier mode switches to Base for a floor-level cloud");
    }

    // ── (e′) covariance_reported() folds the tier-mode entropy into (d,z0,z1) ──────────────────
    {
        CabinetBelief bu(gt, pr);                             // fresh ⇒ tier_evidence_=0 ⇒ p=½ (undecided)
        const auto cov = bu.covariance();
        const auto rep = bu.covariance_reported();
        check(rep(5, 5) > cov(5, 5) and rep(6, 6) > cov(6, 6),
              "covariance_reported inflates σ_z0/σ_z1 while the tier is undecided (p≈½)");
        // Accumulate agreeing base-cloud frames (no switch) ⇒ tier_evidence_ grows ⇒ p→0 ⇒ collapse to Σ_.
        for (int it = 0; it < 40; ++it) bu.resolve_tier(pts, R, 1.0f);
        const auto rep2 = bu.covariance_reported();
        check(rep2(5, 5) < rep(5, 5) and std::abs(rep2(5, 5) - bu.covariance()(5, 5)) < 1e-4f,
              "covariance_reported collapses to Σ_ on z once the tier resolves (p→0)");
    }

    // ── (f) THE PENINSULA. A U-/L-kitchen aisle runs parallel to a wall but touches it on only one
    //    SHORT side, so the flush/parallel wall factor is absent along its length and cannot hold its
    //    orientation. The room-axis (Manhattan) prior must keep it parallel to a room axis on its own.
    //    Ground truth here is a run along the room +y axis (yaw = ±π/2); seed it OBLIQUE with NO wall
    //    and require it to settle on an axis.
    {
        std::vector<Eigen::Vector3f> pts_y;                    // same run as gt, rotated onto +y
        for (int i = 0; i <= 60; ++i)
        {
            const float t = -1.5f + 3.0f * static_cast<float>(i) / 60.0f;
            for (int j = 0; j <= 9; ++j)
                pts_y.emplace_back(0.3f, t, 0.9f * static_cast<float>(j) / 9.0f);   // front face (x = d/2)
            pts_y.emplace_back(0.0f, t, 0.9f);                                      // top face
        }
        for (int j = 0; j <= 9; ++j)                                               // the two end caps
        {
            const float z = 0.9f * static_cast<float>(j) / 9.0f;
            pts_y.emplace_back(0.15f, -1.5f, z);
            pts_y.emplace_back(0.15f,  1.5f, z);
        }
        CabinetBeliefState pseed;
        pseed.cx = 0.2f; pseed.cy = 0.1f; pseed.yaw = std::numbers::pi_v<float> / 2.0f - 0.35f;  // ~20° oblique
        pseed.L = 1.2f;  pseed.d = 0.4f; pseed.z0 = 0.0f; pseed.z1 = 0.9f;
        CabinetBelief b(pseed, pr);
        b.set_room_interior({5.0f, 0.0f});                    // room on the +x side ⇒ front normal → +x
        for (int it = 0; it < 25; ++it) { CabinetFrame f; f.points = pts_y; b.update(f); }   // NO wall
        const float yaw = b.state().yaw;
        constexpr float kQ = std::numbers::pi_v<float> / 2.0f;
        const float axis_resid = std::abs(wrap_pi(yaw - std::round(yaw / kQ) * kQ));
        check(axis_resid < 0.10f, "peninsula (no wall) settles on a room axis via the Manhattan prior");
        check(std::abs(std::cos(yaw)) < 0.20f, "peninsula aligns to the CORRECT axis (+y, from the cloud)");
    }

    // ── (g) THE CORNER MASK. One SAM mask that wraps a corner gives an L-shaped cloud: a long arm along
    //    +x and a shorter arm along +y meeting at the origin. The raw PCA seed diagonalises it into ONE
    //    oblique, over-long box straddling both walls; the room-axis seed must instead pick the dominant
    //    arm's axis and its extent alone, leaving the other arm to fall out as residual.
    {
        std::vector<Eigen::Vector3f> Lc;
        for (int i = 0; i <= 60; ++i)                          // arm A: long, along +x
        {
            const float x = 3.0f * static_cast<float>(i) / 60.0f;
            for (int w = -1; w <= 1; ++w) Lc.emplace_back(x, 0.12f * static_cast<float>(w), 0.45f);
        }
        for (int j = 0; j <= 30; ++j)                          // arm B: shorter, along +y
        {
            const float y = 1.5f * static_cast<float>(j) / 30.0f;
            for (int w = -1; w <= 1; ++w) Lc.emplace_back(0.12f * static_cast<float>(w), y, 0.45f);
        }
        const RunSeed raw  = seed_from_points(Lc, /*room_axis_snap=*/false);
        const RunSeed snap = seed_from_points(Lc, /*room_axis_snap=*/true);
        constexpr float kQ = std::numbers::pi_v<float> / 2.0f;
        const float raw_axis  = std::abs(wrap_pi(raw.yaw  - std::round(raw.yaw  / kQ) * kQ));
        const float snap_axis = std::abs(wrap_pi(snap.yaw - std::round(snap.yaw / kQ) * kQ));
        check(raw_axis > 0.15f,  "PCA seed of an L-mask IS oblique (the bug the room-axis seed prevents)");
        check(snap_axis < 0.02f, "room-axis seed of an L-mask lands on a room axis");
        check(std::abs(std::cos(snap.yaw)) > 0.98f, "room-axis seed picks the dominant (+x) arm");
        check(snap.L < raw.L, "room-axis seed length is the arm, not the longer diagonal span");
    }

    // ── (h) WALL-SEGMENT DOMAIN: corner-aware growth prevention + retract + island invariance ──────
    //    A corner mask hands arm A (a run along +x, flush to a wall at y=−0.3, spanning x∈[0,3], corner at
    //    x=3) the perpendicular arm B (along +y at x≈3). The segment terms must (h1) stop arm A growing
    //    past x=3 into arm B, (h2) retract an already-overgrown end back to the corner, and (h3) stay
    //    fully inert when the wall is far (no manufactured length on an island).
    {
        std::vector<Eigen::Vector3f> armA_pts, Lc;
        for (int i = 0; i <= 60; ++i)
        {
            const float x = 3.0f * static_cast<float>(i) / 60.0f;
            for (int j = 0; j <= 9; ++j) armA_pts.emplace_back(x, 0.3f, 0.9f * static_cast<float>(j) / 9.0f);  // front face y=+0.3
            armA_pts.emplace_back(x, 0.0f, 0.9f);                                                              // top
        }
        Lc = armA_pts;
        for (int j = 0; j <= 30; ++j)                                       // arm B (perpendicular, at the corner)
        {
            const float y = 0.2f + 1.3f * static_cast<float>(j) / 30.0f;
            for (int k = 0; k <= 9; ++k) Lc.emplace_back(3.0f, y, 0.9f * static_cast<float>(k) / 9.0f);
        }
        const auto set_wall = [](CabinetFrame& fr, float wall_y)
        {
            fr.wall.ok = true; fr.wall.p = {0.0f, wall_y}; fr.wall.n = {0.0f, 1.0f};
            fr.wall.has_segment = true; fr.wall.a = {-0.5f, wall_y}; fr.wall.b = {3.0f, wall_y};   // +x corner at x=3
        };
        CabinetBeliefState base; base.cy = 0.0f; base.yaw = 0.0f; base.d = 0.6f; base.z0 = 0.0f; base.z1 = 0.9f;

        // (h1) corner absorption PREVENTED: seed arm A short; feed the full L-cloud; must not grow past x=3.
        { CabinetBeliefState sd = base; sd.cx = 1.4f; sd.L = 2.4f;
          CabinetBelief b(sd, pr); b.set_room_interior({1.5f, 5.0f});
          for (int it = 0; it < 60; ++it) { CabinetFrame f; f.points = Lc; set_wall(f, -0.3f); b.update(f); }
          const auto st = b.state();
          check(st.cx + 0.5f * st.L < 3.0f + 0.10f, "corner: +x end is censored at the wall corner (no growth into arm B)");
          check(std::abs(wrap_pi(st.yaw)) < 0.08f,   "corner: run stays axis-aligned (arm B not absorbed → no tilt)"); }

        // (h2) RETRACT: seed OVERGROWN past the corner; arm-A points only; the inward hinge pulls it back.
        { CabinetBeliefState sd = base; sd.cx = 2.2f; sd.L = 4.8f;
          CabinetBelief b(sd, pr); b.set_room_interior({1.5f, 5.0f});
          for (int it = 0; it < 80; ++it) { CabinetFrame f; f.points = armA_pts; set_wall(f, -0.3f); b.update(f); }
          check(b.state().cx + 0.5f * b.state().L < 3.0f + 0.15f, "retract: overgrown +x end is pulled back to the corner"); }

        // (h3) ISLAND INVARIANCE: same overgrown seed but the wall is 2 m away ⇒ wgt≈0 ⇒ NO retract.
        { CabinetBeliefState sd = base; sd.cx = 2.2f; sd.L = 4.8f;
          CabinetBelief b(sd, pr); b.set_room_interior({1.5f, 5.0f});
          for (int it = 0; it < 80; ++it) { CabinetFrame f; f.points = armA_pts; set_wall(f, -2.3f); b.update(f); }
          check(b.state().cx + 0.5f * b.state().L > 3.5f, "island: a far wall does NOT retract the run (no manufactured length)"); }
    }

    if (ok) std::print("[cabinet_belief::self_test] all checks passed\n");
    return ok;
}

}  // namespace rc
