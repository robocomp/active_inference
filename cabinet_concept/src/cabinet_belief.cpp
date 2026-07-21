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

    // Posterior weight of the "flush against this wall" component, and the wall's own uncertainty.
    const float rel = gap / std::max(1e-3f, params_.wall_reach_m);
    const float wgt = std::exp(-rel * rel);
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

    // Re-express the two extreme observations as FIXED room-frame points, so the residual is a pure
    // function of the state (the along-axis coordinate itself depends on cx,cy,yaw and must not be
    // frozen inside the residual).
    const Eigen::Vector2f c(s.cx, s.cy);
    const Eigen::Vector2f p_lo = c + u * s_lo, p_hi = c + u * s_hi;

    // Hinge residuals: how far each observed end sticks OUT of the box along the axis.
    const auto r_lo_of = [&](const CabinetBeliefState& st)
    { return std::max(0.0f, (st.centre() - p_lo).dot(st.axis()) - 0.5f * st.L); };
    const auto r_hi_of = [&](const CabinetBeliefState& st)
    { return std::max(0.0f, (p_hi - st.centre()).dot(st.axis()) - 0.5f * st.L); };

    const float r_lo = r_lo_of(s), r_hi = r_hi_of(s);
    if (r_lo <= 0.0f and r_hi <= 0.0f) return;      // already contained ⇒ the factor is silent

    const float lam = params_.extent_precision;
    const float eps = params_.fd_eps;
    const Eigen::Matrix<float, 7, 1> v = s.vec();
    Eigen::Matrix<float, 7, 1> Jlo, Jhi;
    for (int k = 0; k < 7; ++k)
    {
        Eigen::Matrix<float, 7, 1> vp = v, vm = v;
        vp(k) += eps; vm(k) -= eps;
        const auto sp = CabinetBeliefState::from_vec(vp), sm = CabinetBeliefState::from_vec(vm);
        Jlo(k) = std::clamp((r_lo_of(sp) - r_lo_of(sm)) / (2.0f * eps), -kJClamp, kJClamp);
        Jhi(k) = std::clamp((r_hi_of(sp) - r_hi_of(sm)) / (2.0f * eps), -kJClamp, kJClamp);
    }
    if (r_lo > 0.0f) { Id.noalias() += lam * (Jlo * Jlo.transpose()); bd.noalias() += -lam * Jlo * r_lo; }
    if (r_hi > 0.0f) { Id.noalias() += lam * (Jhi * Jhi.transpose()); bd.noalias() += -lam * Jhi * r_hi; }
}

void CabinetBelief::accumulate_extra(const CabinetBeliefState& s, const CabinetFrame& f,
                                     Eigen::Matrix<float, 7, 7>& Id, Eigen::Matrix<float, 7, 1>& bd) const
{
    accumulate_wall(s, f, Id, bd);
    accumulate_extent(s, f, Id, bd);

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
RunSeed CabinetBelief::seed_from_points(const std::vector<Eigen::Vector3f>& pts)
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

    // Principal axis + eigenvalues of the 2x2 covariance.
    const double tr = sxx + syy, det = sxx * syy - sxy * sxy;
    const double disc = std::sqrt(std::max(0.0, 0.25 * tr * tr - det));
    const double l1 = 0.5 * tr + disc, l2 = std::max(0.0, 0.5 * tr - disc);
    s.yaw = static_cast<float>(0.5 * std::atan2(2.0 * sxy, sxx - syy));
    s.L = static_cast<float>(std::sqrt(std::max(0.0, 12.0 * l1)));   // equivalent uniform rectangle
    s.d = static_cast<float>(std::sqrt(std::max(0.0, 12.0 * l2)));
    const float sum = s.L + s.d;
    s.aniso = (sum > 1e-4f) ? (s.L - s.d) / sum : 0.0f;

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

    if (ok) std::print("[cabinet_belief::self_test] all checks passed\n");
    return ok;
}

}  // namespace rc
