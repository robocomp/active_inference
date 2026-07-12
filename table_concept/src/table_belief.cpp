/*
 * table_belief.cpp  —  AI2 table belief implementation (see TABLE.md)
 *
 * The table-specific model hooks behind TableBelief: the compound SDF (top slab + 4 derived legs), the soft
 * mixture responsibilities + true (clutter-inclusive) free energy, the finite-difference Jacobian, the extra
 * GN factors (LiDAR range · coverage traction · free-space vacate), the footprint second-moment fusion, the
 * canonical symmetry fold and orientation-mode resolution, and self_test(). All Bayesian bookkeeping
 * (predict / GN-MAP / Woodbury) lives in common/ai_belief/recursive_laplace.h; this file only feeds it.
 */

#include "table_belief.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>

namespace rc
{

// ─── SDF primitives (scalar, room→local handled by caller) ───────────────────────────────────────

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

// Vertical cylinder SDF: dl = (local point − cylinder centre), radius r, half-height hh.
float cylinder_sdf(float dlx, float dly, float dlz, float r, float hh)
{
    const float d_radial = std::sqrt(dlx * dlx + dly * dly) - r;
    const float d_vert   = std::abs(dlz) - hh;
    const float outside  = std::sqrt(std::max(d_radial, 0.0f) * std::max(d_radial, 0.0f) +
                                     std::max(d_vert, 0.0f)   * std::max(d_vert, 0.0f));
    const float inside   = std::min(std::max(d_radial, d_vert), 0.0f);
    return outside + inside;
}
}  // namespace

Eigen::Vector2f TableBelief::leg_center_local(const TableBeliefState& s, int k) const
{
    static const std::array<std::array<float, 2>, 4> sg = {{ {1, 1}, {-1, 1}, {-1, -1}, {1, -1} }};
    const float lx = sg[k][0] * (0.5f * s.w - params_.leg_radius);
    const float ly = sg[k][1] * (0.5f * s.h - params_.leg_radius);
    return {lx, ly};
}

float TableBelief::sdf_top(const Eigen::Vector3f& p, const TableBeliefState& s) const
{
    const float c = std::cos(-s.yaw), sn = std::sin(-s.yaw);
    const float px = p.x() - s.cx, py = p.y() - s.cy;
    const float lx = px * c - py * sn;
    const float ly = px * sn + py * c;
    const float half_t = 0.5f * params_.top_thickness;
    const float top_cz = s.H - half_t;
    return box_sdf(std::abs(lx) - 0.5f * s.w, std::abs(ly) - 0.5f * s.h, std::abs(p.z() - top_cz) - half_t);
}

float TableBelief::sdf_leg(const Eigen::Vector3f& p, const TableBeliefState& s, int k) const
{
    const float c = std::cos(-s.yaw), sn = std::sin(-s.yaw);
    const float px = p.x() - s.cx, py = p.y() - s.cy;
    const float lx = px * c - py * sn;
    const float ly = px * sn + py * c;
    const auto  ctr = leg_center_local(s, k);
    const float hh  = leg_half_height(s);
    return cylinder_sdf(lx - ctr.x(), ly - ctr.y(), p.z() - hh, params_.leg_radius, hh);
}

float TableBelief::sdf_prim(const Eigen::Vector3f& p, const TableBeliefState& s, int prim) const
{
    return prim == 0 ? sdf_top(p, s) : sdf_leg(p, s, prim - 1);
}

float TableBelief::sdf_compound(const Eigen::Vector3f& p, const TableBeliefState& s) const
{
    float m = sdf_top(p, s);
    for (int k = 0; k < 4; ++k) m = std::min(m, sdf_leg(p, s, k));
    return m;
}

// ─── Mixture responsibilities ────────────────────────────────────────────────────────────────────

// UN-normalised mixture components u[0..5] (top, 4 legs, clutter) and their sum = the marginal likelihood
// numerator p(point | model) ∝ Σ_k π_k N(d_k; 0, R). responsibilities() normalises this; the FREE ENERGY reads
// −log(sum) — the clutter term (u[5]) makes a far/misfit point cost −log(clutter likelihood), a large real
// penalty, so the energy actually reflects misfit (it is NOT silently zeroed by clutter attribution).
float TableBelief::mixture_unnormalized(const Eigen::Vector3f& p, const TableBeliefState& s, float R,
                                        std::array<float, 6>& u) const
{
    const float eps    = std::clamp(params_.clutter_frac, 0.0f, 0.99f);
    const float pi_surf = (1.0f - eps) / 5.0f;
    const float inv2R  = 0.5f / std::max(1e-9f, R);

    // Height-based attribution. The leg cylinders' lateral surface exists only over z∈[0, H−top_thickness];
    // the top slab only over z∈[H−top_thickness, H]. The two meet at the join plane, so a tabletop point near
    // a corner (z≈H, laterally close to an inset corner leg) is otherwise mis-attributed to that leg, and GN
    // shrinks w,h to slide the legs under it → systematic under-size. Gate each component's mixing weight by a
    // smooth vertical compatibility about the join plane (band = slab half-thickness — physical, not tuned).
    const float z_join = s.H - params_.top_thickness;       // leg top == slab bottom
    const float band   = std::max(0.5f * params_.top_thickness, 1e-3f);
    const float leg_z  = 1.0f / (1.0f + std::exp((p.z() - z_join) / band));  // →1 below join, →0 above (tabletop)
    const float top_z  = 1.0f - leg_z;                                       // →1 above join (tabletop), →0 below

    u[0] = pi_surf * top_z * std::exp(-sdf_top(p, s) * sdf_top(p, s) * inv2R);
    for (int k = 0; k < 4; ++k)
    {
        const float d = sdf_leg(p, s, k);
        u[1 + k] = pi_surf * leg_z * std::exp(-d * d * inv2R);
    }
    // Clutter = uniform; modelled as a Gaussian floor at clutter_scale so a point further than that from
    // EVERY surface is explained by clutter rather than dragging the model.
    const float cs = params_.clutter_scale_m;
    u[5] = eps * std::exp(-cs * cs * inv2R);

    float sum = 0.0f;
    for (float v : u) sum += v;
    return sum;
}

std::array<float, 6> TableBelief::responsibilities(const Eigen::Vector3f& p, const TableBeliefState& s, float R) const
{
    std::array<float, 6> u{};
    const float sum = mixture_unnormalized(p, s, R, u);
    // EM holds responsibilities fixed within a GN iteration, so this needs no Jacobian change.
    if (sum <= 0.0f) { u = {}; u[5] = 1.0f; return u; }
    for (float& v : u) v /= sum;
    return u;
}

// ─── Jacobian (central finite difference) ────────────────────────────────────────────────────────

Eigen::Matrix<float, 6, 1> TableBelief::sdf_jacobian(const Eigen::Vector3f& p, const TableBeliefState& s, int prim) const
{
    Eigen::Matrix<float, 6, 1> J;
    Eigen::Matrix<float, 6, 1> base = s.vec();
    const float e = params_.fd_eps;
    for (int j = 0; j < 6; ++j)
    {
        Eigen::Matrix<float, 6, 1> vp = base, vm = base;
        vp(j) += e; vm(j) -= e;
        J(j) = (sdf_prim(p, TableBeliefState::from_vec(vp), prim) -
                sdf_prim(p, TableBeliefState::from_vec(vm), prim)) / (2.0f * e);
    }
    return J;
}

// ─── Constraints / canonical form ────────────────────────────────────────────────────────────────

// Per-GN-iteration bounds: floor each extent so the box can't invert or collapse, and wrap yaw.
// THRESHOLDS (physical floors, not tuned gates): w,h ≥ 0.10 m and H ≥ top_thickness+0.05 m keep the SDF
// well-posed (a degenerate zero/negative extent has no gradient and would trap the GN step). yaw wraps to
// (−π, π] for continuity. These clamp geometry only; they never gate evidence.
void TableBelief::apply_constraints(TableBeliefState& s) const
{
    s.w = std::max(s.w, 0.10f);
    s.h = std::max(s.h, 0.10f);
    s.H = std::max(s.H, params_.top_thickness + 0.05f);
    s.yaw = std::remainder(s.yaw, 2.0f * static_cast<float>(M_PI));   // wrap to (−π, π]
}

// ─── Engine hooks: prior cov · process noise (F = I, static) · common-mode ───────────────────────

Eigen::Matrix<float, 6, 1> TableBelief::prior_cov_diag() const
{
    const float ss = params_.prior_size_std * params_.prior_size_std;   // H, w, h
    return (Eigen::Matrix<float, 6, 1>() << 0.30f * 0.30f, 0.30f * 0.30f, ss, ss, ss, 0.60f * 0.60f).finished();
}

Eigen::Matrix<float, 6, 1> TableBelief::process_noise_diag() const
{
    const float qm = params_.process_std_m * params_.process_std_m;     // rigid + static ⇒ small
    const float qy = params_.process_std_yaw * params_.process_std_yaw;
    return (Eigen::Matrix<float, 6, 1>() << qm, qm, qm, qm, qm, qy).finished();
}

// Inverse of the per-frame common-mode covariance Σ_c (diagonal): position (cx,cy) = config floor + pose-
// chain + range variance; size (H,w,h) = config std; yaw = config std + range. Marginalising this SHARED
// error (via Woodbury in the engine) makes the frame's information SATURATE at Σ_c regardless of point
// count — N correlated points cannot collapse σ.
Eigen::Matrix<float, 6, 1> TableBelief::common_mode_inv_diag(const TableFrame& frame) const
{
    const float p2 = params_.common_mode_pos_std  * params_.common_mode_pos_std;
    const float s2 = params_.common_mode_size_std * params_.common_mode_size_std;
    const float y2 = params_.common_mode_yaw_std  * params_.common_mode_yaw_std;
    const float cs = frame.chain_cov_size;   // range-driven SIZE variance: distant frame → less size authority
    return (Eigen::Matrix<float, 6, 1>() <<
            1.0f / std::max(1e-9f, p2 + frame.chain_cov_xx),
            1.0f / std::max(1e-9f, p2 + frame.chain_cov_yy),
            1.0f / std::max(1e-9f, s2 + cs), 1.0f / std::max(1e-9f, s2 + cs), 1.0f / std::max(1e-9f, s2 + cs),
            1.0f / std::max(1e-9f, y2 + frame.chain_cov_yaw)).finished();   // size + yaw caps grow with view range
}

// ─── Footprint second-moment statistic ───────────────────────────────────────────────────────────

// 2D footprint second-moment of the top-band points (the yaw/extent statistic used by the moment factor).
//
// Centroid + inertia tensor of the (x,y) projection, then the closed-form 2×2 symmetric eigendecomposition:
// eigenvalues λ₁≥λ₂ give the equivalent uniform-rectangle FULL extents √(12·λ), and the major eigenvector
// angle gives the principal-axis orientation. Model-free (only the z-band selects the plane) — so a mask
// bigger/yawed than the current box is measured for what it is.
FootprintMoment TableBelief::footprint_moment(const std::vector<Eigen::Vector3f>& pts, float z_lo, float z_hi)
{
    FootprintMoment m;
    double sx = 0.0, sy = 0.0;
    int n = 0;
    for (const auto& p : pts)
        if (p.z() >= z_lo and p.z() <= z_hi) { sx += p.x(); sy += p.y(); ++n; }
    if (n < 16) return m;                                    // too few plane points for a stable moment
    const double mx = sx / n, my = sy / n;
    double cxx = 0.0, cyy = 0.0, cxy = 0.0;
    for (const auto& p : pts)
        if (p.z() >= z_lo and p.z() <= z_hi)
        { const double dx = p.x() - mx, dy = p.y() - my; cxx += dx * dx; cyy += dy * dy; cxy += dx * dy; }
    cxx /= n; cyy /= n; cxy /= n;
    const double tr = cxx + cyy, det = cxx * cyy - cxy * cxy;
    const double disc = std::sqrt(std::max(0.0, 0.25 * tr * tr - det));
    const double l1 = 0.5 * tr + disc, l2 = std::max(0.0, 0.5 * tr - disc);
    m.ok = true;
    m.n = n;
    m.cx = static_cast<float>(mx);
    m.cy = static_cast<float>(my);
    m.ext_major = static_cast<float>(std::sqrt(std::max(0.0, 12.0 * l1)));   // full extent of uniform rect
    m.ext_minor = static_cast<float>(std::sqrt(std::max(0.0, 12.0 * l2)));
    m.phi = 0.5f * std::atan2(2.0f * static_cast<float>(cxy),
                              static_cast<float>(cxx - cyy));                 // major-axis angle ∈ (−π/2, π/2]
    return m;
}

// ─── Extra GN factors: LiDAR range · coverage traction · free-space vacate ───────────────────────

// Extra per-frame evidence folded into the GN normal equations (engine calls it inside the loop): the
// YOLO-independent LiDAR range factor, then the coverage and free-space terms. Accumulates into (Id, bd).
//
// LiDAR: sphere-traces THIS belief's own SDF (n_prims=5, sdf_prim/sdf_jacobian), so the SAME shared call the
// bottle uses drops in unchanged at N=6. A no-op when f.lidar.precision==0. Because all returns of one sweep
// share the sensor-pose error they are correlated, but they inform the SAME (Id,bd) the depth points do, so
// the engine's common-mode Woodbury saturation de-correlates them too — no separate handling needed here.
void TableBelief::accumulate_extra(const TableBeliefState& s, const TableFrame& f,
                                   Eigen::Matrix<float, 6, 6>& Id, Eigen::Matrix<float, 6, 1>& bd) const
{
    rc::ai::accumulate_lidar_rays<6>(*this, s, f.lidar, Id, bd);
    for (const auto& lr : f.lidar_extra)   // extra per-device ray-sets (e.g. low bpearl) — each occlusion-aware, own origin
        rc::ai::accumulate_lidar_rays<6>(*this, s, lr, Id, bd);
    dbg_coverage_pts_ = 0;   // monitor counters — reset per accumulate; the final (converged) call is what's read
    dbg_vacate_beams_ = 0;
    dbg_moment_pts_   = 0;

    // Coverage / traction (table_1.png): reclaim on-plane mask points the mixture ceded to CLUTTER as
    // GROW-ONLY extent evidence. For each point OUTSIDE the top slab (sdf_top > 0) and near the tabletop plane,
    // add the standard top-slab SDF pull weighted by (clutter responsibility)·Cauchy(outside dist), so a model
    // that under-covers a large mask is pulled out to explain it. One-sided (never shrinks — that is the
    // free-space carve's job); self-bounded (sdf→0 when covered); off-plane points (legs/floor/contamination)
    // are gated out. 0 ⇒ OFF.
    if (params_.coverage_precision > 0.0f and not f.points.empty())
    {
        const float eps     = std::clamp(params_.clutter_frac, 0.0f, 0.99f);
        const float pi_surf = (1.0f - eps) / 5.0f;
        const float ratio0  = (eps > 1e-6f) ? pi_surf / eps : 1e6f;         // surface-vs-clutter prior ratio
        const float cs2     = params_.clutter_scale_m * params_.clutter_scale_m;
        const float R       = std::max(1e-6f, params_.sigma_base_m * params_.sigma_base_m);
        const float inv2R   = 0.5f / R;
        const float cc2     = params_.coverage_robust_c_m * params_.coverage_robust_c_m;
        const float z_join  = s.H - params_.top_thickness;
        const float band    = std::max(0.5f * params_.top_thickness, 1e-3f);
        for (const auto& p : f.points)
        {
            const float e = sdf_top(p, s);
            if (e <= 0.0f) continue;                                        // inside/on the slab → normal term fits it
            const float top_z = 1.0f - 1.0f / (1.0f + std::exp((p.z() - z_join) / band));  // →1 on the tabletop plane
            if (top_z < 0.05f) continue;                                    // off-plane (legs/floor/contamination)
            // clutter responsibility of this on-plane point (legs ≈0 on-plane): →1 beyond clutter_scale.
            const float clut = 1.0f / (1.0f + ratio0 * std::exp((cs2 - e * e) * inv2R));
            const float w = params_.coverage_precision * top_z * clut / (1.0f + e * e / cc2);
            if (w < 1e-9f) continue;
            const Eigen::Matrix<float, 6, 1> J = sdf_jacobian(p, s, 0);     // top slab (prim 0)
            Id.noalias() += w * (J * J.transpose());
            bd.noalias() += -w * J * e;
            ++dbg_coverage_pts_;
        }
    }

    // Free-space / VACATE (EXISTENCE_BELIEF_PLAN.md Step 4): the shrink-only counter-force that BOUNDS the
    // grow-only coverage term above. A LiDAR beam that traverses the top-slab volume and continues BEYOND it
    // (endpoint past the far face) proves that crossing is empty; its midpoint lies INSIDE the footprint, so a
    // GLS pull driving the FOOTPRINT SDF→0 there RETREATS the nearest face past the empty point = shrink.
    // A beam that RETURNS inside the slab (occupancy) has its far face BEYOND the endpoint ⇒ p_through
    // ≈ 0 ⇒ no vacate, so a real tabletop is untouched. z-gated to the SOLID top band [H−t, H] (beams passing
    // under the tabletop between the legs are excluded) ⇒ no hollow ambiguity, so no hollow guard is needed.
    // Correlated beams saturate through the engine's common-mode Woodbury, same as the range factor. Occupy-
    // where-masked (coverage) + vacate-where-through (this) settle the extent where camera and LiDAR AGREE.
    if (params_.free_space_precision > 0.0f and not f.lidar.endpoints.empty())
    {
        const float cyaw = std::cos(-s.yaw), syaw = std::sin(-s.yaw);
        const float hw = 0.5f * s.w, hd = 0.5f * s.h;
        const float lo[3] = {-hw, -hd, s.H - params_.top_thickness};   // local top-slab box (solid tabletop band)
        const float hi[3] = { hw,  hd, s.H};
        const float sigma_surf = std::sqrt(params_.sigma_base_m * params_.sigma_base_m
                                         + params_.common_mode_pos_std * params_.common_mode_pos_std);
        const auto  phi = [](float x) { return 0.5f * std::erfc(-x * 0.70710678f); };   // Φ(x)
        const Eigen::Vector3f& O = f.lidar.origin;
        const float oxr = O.x() - s.cx, oyr = O.y() - s.cy;
        const Eigen::Vector3f Ol(oxr * cyaw - oyr * syaw, oxr * syaw + oyr * cyaw, O.z());   // sensor in local frame
        for (const auto& ep : f.lidar.endpoints)
        {
            const float exr = ep.x() - s.cx, eyr = ep.y() - s.cy;
            const Eigen::Vector3f El(exr * cyaw - eyr * syaw, exr * syaw + eyr * cyaw, ep.z());
            const Eigen::Vector3f d = El - Ol;                            // local ray (t=1 at the endpoint)
            const float len = d.norm();
            if (len < 1e-4f) continue;

            // Ray vs local top-slab box → normalised [t_near, t_far] (endpoint at t=1). Standard 3-axis slab.
            float t_near = 0.0f, t_far = std::numeric_limits<float>::max();
            bool miss = false;
            for (int a = 0; a < 3 and not miss; ++a)
            {
                if (std::abs(d(a)) < 1e-6f) { if (Ol(a) < lo[a] or Ol(a) > hi[a]) miss = true; }
                else
                {
                    float t1 = (lo[a] - Ol(a)) / d(a), t2 = (hi[a] - Ol(a)) / d(a);
                    if (t1 > t2) std::swap(t1, t2);
                    t_near = std::max(t_near, t1);
                    t_far  = std::min(t_far,  t2);
                }
            }
            if (miss or t_near > t_far or t_far < 0.0f or t_near >= 1.0f) continue;

            // Passed-through soft weight: endpoint beyond the far face (else it returned inside ⇒ occupancy).
            const float p_through = phi((1.0f - t_far) / (sigma_surf / len));
            if (p_through < 1e-3f) continue;

            const float t_mid = 0.5f * (t_near + t_far);                  // box-centre along the ray (empty)
            const Eigen::Vector3f p_free = O + t_mid * (ep - O);          // WORLD empty point

            // Residual = the LATERAL (2D footprint) SDF, NOT the full 3D sdf_top: for a point just under the
            // tabletop the thin THICKNESS face dominates the 3D SDF (∂/∂w = ∂/∂h = 0), so a 3D residual cannot
            // move the extent. The footprint SDF (signed distance to the w×h rectangle in the yaw frame, z
            // ignored) restores the w/h/cx/cy gradient. e < 0 (empty point INSIDE the footprint) ⇒ driving e→0
            // retreats the NEAREST face just past it = shrink-only. Central-difference Jacobian over the state.
            const auto sdf_fp = [&](const TableBeliefState& st) -> float
            {
                const float cc = std::cos(-st.yaw), s2 = std::sin(-st.yaw);
                const float lx = (p_free.x() - st.cx) * cc - (p_free.y() - st.cy) * s2;
                const float ly = (p_free.x() - st.cx) * s2 + (p_free.y() - st.cy) * cc;
                const float dx = std::abs(lx) - 0.5f * st.w, dy = std::abs(ly) - 0.5f * st.h;
                const float ox = std::max(dx, 0.0f), oy = std::max(dy, 0.0f);
                return std::sqrt(ox * ox + oy * oy) + std::min(std::max(dx, dy), 0.0f);   // 2D box SDF
            };
            const float e = sdf_fp(s);
            if (e >= 0.0f) continue;                                      // must be inside the footprint to shrink
            Eigen::Matrix<float, 6, 1> J;
            const Eigen::Matrix<float, 6, 1> base = s.vec();
            const float fde = params_.fd_eps;
            for (int j = 0; j < 6; ++j)
            {
                Eigen::Matrix<float, 6, 1> vp = base, vm = base; vp(j) += fde; vm(j) -= fde;
                J(j) = (sdf_fp(TableBeliefState::from_vec(vp)) - sdf_fp(TableBeliefState::from_vec(vm))) / (2.0f * fde);
            }
            const float w = params_.free_space_precision * p_through;
            Id.noalias() += w * (J * J.transpose());
            bd.noalias() += -w * J * e;
            ++dbg_vacate_beams_;
        }
    }
}

// ─── Footprint-moment measurement fusion ─────────────────────────────────────────────────────────

// Fuse the footprint second-moment as a SEPARATE 3-DOF (w,h,yaw) Kalman update with its OWN covariance,
// deliberately OUTSIDE the per-point common-mode saturation. No-op when precision==0.
// (See TableBeliefParams::footprint_moment_precision.)
//
// Rationale: the per-point range common-mode caps a single boundary point's authority (its deprojected
// position error grows with range and is shared across the mask). But the moment is a GLOBAL fit over
// thousands of points; its principal-axis angle and eigen-extents are direction/scale estimates in which the
// shared radial error largely cancels, so they stay informative at range where the per-point channel is
// frozen. This is exactly the DOF the per-point mixture cannot supply on a flat top (tables_5.png stuck yaw).
// Standard information/Kalman update on the full 6×6 Σ. yaw variance ∝ 1/anisotropy² → a near-square footprint
// carries no orientation info (K→0; resolve_orientation owns that flip). NOT applied to cx,cy (a partial
// view's centroid is biased; the per-point channel already fixes position).
void TableBelief::apply_footprint_moment(const TableFrame& frame)
{
    dbg_moment_pts_ = 0;
    dbg_moment_aniso_ = 0.0f; dbg_moment_r_yaw_ = 0.0f; dbg_moment_dyaw_ = 0.0f;   // DIAGNOSTIC: reset per cycle
    dbg_moment_ext_major_ = 0.0f; dbg_moment_ext_minor_ = 0.0f;
    if (params_.footprint_moment_precision <= 0.0f or frame.points.empty()) return;
    const float z_hi = state_.H + 3.0f * params_.sigma_base_m;
    const float z_lo = state_.H - params_.top_thickness - 3.0f * params_.sigma_base_m;   // tabletop band (excl. legs)
    const FootprintMoment mom = footprint_moment(frame.points, z_lo, z_hi);
    if (not mom.ok) return;
    dbg_moment_pts_ = mom.n;

    const auto near_pi = [](float a, float ref)                       // a mapped to nearest ref mod π (box π-sym)
    { return ref + std::remainder(a - ref, static_cast<float>(M_PI)); };
    // Two symmetry-consistent (major,minor)→(w,h) assignments; pick the one nearest the CURRENT (w,h) so the
    // moment REINFORCES the present labelling and never itself triggers a 90° flip (resolve_orientation owns that).
    const float yawA  = near_pi(mom.phi, state_.yaw);
    const float yawB  = near_pi(mom.phi + 0.5f * static_cast<float>(M_PI), state_.yaw);
    const auto  sq     = [](float x) { return x * x; };
    const bool  A = sq(mom.ext_major - state_.w) + sq(mom.ext_minor - state_.h)
                  <= sq(mom.ext_minor - state_.w) + sq(mom.ext_major - state_.h);
    const float mw   = A ? mom.ext_major : mom.ext_minor;
    const float mh   = A ? mom.ext_minor : mom.ext_major;
    const float myaw = A ? yawA : yawB;

    const float sum   = mom.ext_major + mom.ext_minor;
    const float aniso = (sum > 1e-4f) ? (mom.ext_major - mom.ext_minor) / sum : 0.0f;
    const float Pm    = params_.footprint_moment_precision;
    // Per-frame measurement variance. 1/Pm is the STATIC noise FLOOR; frame.moment_extra_var adds the SHARED
    // per-frame error (ego-motion corruption + a gentle range term the fitter authors). Without it a fixed 1/Pm
    // is absurdly confident (≈2 cm @Pm=2000) and the belief SNAPS to each frame's noisy footprint (the live
    // yaw/extent wander) instead of accumulating. A moving robot / far view inflates it → the moment backs off.
    // COMPLETENESS backoff: how much of the believed footprint this frame's moment actually spans. A partial /
    // foreshortened view yields a fragmentary footprint whose 2D inertia is spuriously anisotropic — high aniso,
    // hence a razor-sharp r_yaw below, but pointing the WRONG way. Grow the moment measurement variance as the
    // observed area falls below the believed area so the sliver can no longer rotate/reshape a converged table.
    // completeness≥1 (full view or a legitimately larger mask) ⇒ no inflation. See TableBeliefParams. 0 = OFF.
    float compl_extra = 0.0f;
    if (params_.footprint_moment_completeness_gain > 0.0f)
    {
        const float obs_area = mom.ext_major * mom.ext_minor;
        const float bel_area = state_.w * state_.h;
        const float completeness = (bel_area > 1e-4f and obs_area > 0.0f) ? obs_area / bel_area : 1.0f;
        const float c  = std::clamp(completeness, params_.footprint_moment_min_completeness, 1.0f);
        const float cs = params_.footprint_moment_completeness_gain * (1.0f / c - 1.0f);   // 0 at completeness≥1
        compl_extra = cs * cs;
    }
    const float r_base = 1.0f / Pm + std::max(0.0f, frame.moment_extra_var) + compl_extra;
    float r_w   = r_base;                                             // measurement variance (m²)
    float r_h   = r_base;
    const float r_yaw = r_base / std::max(aniso * aniso, 1e-3f);      // rad²; near-square → huge → no yaw pull
    // DIAGNOSTIC (rogue-mask yaw attribution): record this frame's footprint anisotropy, the yaw variance the
    // moment actually used, the yaw move it requested, and the observed extents. Read by the fitter's [yaw-jump].
    dbg_moment_aniso_ = aniso; dbg_moment_r_yaw_ = r_yaw;
    dbg_moment_dyaw_  = std::remainder(myaw - state_.yaw, 2.0f * static_cast<float>(M_PI));
    dbg_moment_ext_major_ = mom.ext_major; dbg_moment_ext_minor_ = mom.ext_minor;

    // A mask is only ever a LOWER BOUND on the table extent. Occlusion, foreshortening (a side view of the
    // tabletop, whose far edge is grazing/self-occluded), FoV clipping and YOLO under-segmentation can all make
    // the observed footprint SHORTER than the table — but NOTHING makes it longer. So the moment may only GROW
    // the extent, never SHRINK it: a partial view can no longer collapse a converged table (the side-view shrink).
    // This needs no completeness signal (trunc/in-view missed the foreshortening case anyway). Legitimate shrink —
    // a genuinely short table, or an over-grown model — is the job of the OCCLUSION-AWARE free-space / vacate
    // channel (FreeSpacePrecision: a beam that passes THROUGH proves empty; a beam that returns short is just
    // occluded and does not vacate), which is where "absence" belongs. The extent grows to the largest footprint
    // seen across the orbit (the best view = the true extent) and holds. Over-grow is bounded by the step-guard.
    // THRESHOLD (the file's one hard switch): grow-only guard. kDrop is a huge measurement variance that
    // effectively removes the w or h row (K→0: no mean move, no Σ reduction). It fires when the moment would
    // SHRINK an extent, because a mask is only ever a LOWER BOUND (see the grow-only paragraph above) — a
    // measured shrink is an untrustworthy partial view, so drop it. Legitimate shrink is the vacate channel's job.
    constexpr float kDrop = 1e12f;
    if (mw < state_.w) r_w = kDrop;      // moment would SHRINK w ⇒ partial view → drop the row
    if (mh < state_.h) r_h = kDrop;      // moment would SHRINK h ⇒ partial view → drop the row

    Eigen::Matrix<float, 3, 6> H = Eigen::Matrix<float, 3, 6>::Zero();
    H(0, 3) = 1.0f; H(1, 4) = 1.0f; H(2, 5) = 1.0f;                   // measures w, h, yaw
    const Eigen::Vector3f y(mw - state_.w, mh - state_.h,
                            std::remainder(myaw - state_.yaw, 2.0f * static_cast<float>(M_PI)));
    const Eigen::Matrix3f Rm = Eigen::Vector3f(r_w, r_h, r_yaw).asDiagonal();
    const Eigen::Matrix3f S  = H * Sigma_ * H.transpose() + Rm;
    const Eigen::Matrix<float, 6, 3> K = Sigma_ * H.transpose() * S.inverse();
    Eigen::Matrix<float, 6, 1> xv = state_.vec() + K * y;
    Sigma_ = (Eigen::Matrix<float, 6, 6>::Identity() - K * H) * Sigma_;
    Sigma_ = 0.5f * (Sigma_ + Sigma_.transpose());                   // keep symmetric/SPD
    TableBeliefState s = TableBeliefState::from_vec(xv);
    apply_constraints(s);
    state_ = s;
}

// ─── Canonical symmetry fold ─────────────────────────────────────────────────────────────────────

// Fold the state onto the box-symmetry representative CLOSEST to the predicted mean (CONTINUITY fold,
// replaces the w≥h sign fold; TABLE.md §5).
//
// The box SDF is invariant under the exact symmetry group {e, r_π, swap∘r_{π/2}, swap∘r_{−π/2}} — four
// representations of the SAME table. The old fold chose by sign(w−h), a NOISE process when w≈h, which
// converted extent jitter into 90° yaw SNAPS. Instead pick the representative CLOSEST (prior Mahalanobis under
// Σ_pred) to the predicted mean, i.e. the MAP over the covering sheets: the chart follows continuity, never
// the sign of a near-tie. Since the GN result stays near the prediction, the identity representative wins in
// the common case (no fold on noise). The genuine cross-class flip (near-square ambiguity) is owned by
// resolve_orientation(), not here.
void TableBelief::canonicalize(TableBeliefState& s) const
{
    constexpr float kHalfPi = 0.5f * static_cast<float>(M_PI);
    const auto wrap = [](float a) { return std::remainder(a, 2.0f * static_cast<float>(M_PI)); };

    // The four exact-symmetry representatives of s (cx,cy,H unchanged; only w,h,yaw transform).
    const std::array<TableBeliefState, 4> reps = {{
        { s.cx, s.cy, s.H, s.w, s.h, wrap(s.yaw)            },   // e
        { s.cx, s.cy, s.H, s.h, s.w, wrap(s.yaw + kHalfPi)  },   // swap ∘ r_{+π/2}
        { s.cx, s.cy, s.H, s.w, s.h, wrap(s.yaw + 2*kHalfPi)},   // r_π
        { s.cx, s.cy, s.H, s.h, s.w, wrap(s.yaw - kHalfPi)  },   // swap ∘ r_{−π/2}
    }};

    // CONTINUITY fold with a FIXED yaw scale (flag-gated). The Σ-weighted fold below weights the yaw difference
    // by Σ_pred⁻¹, which vanishes exactly when the view is edge-on (σ_yaw huge) — so a 90°/180° box-symmetry snap
    // costs ≈nothing and the reported yaw oscillates ±90/±180 between equivalent representatives of the SAME box
    // (the grazing-view flips seen in the CSV; dims constant for r_π, swapped for w↔h). A fixed yaw scale keeps
    // continuity effective independent of σ_yaw: a genuine slow rotation stays cheap, a symmetry snap does not.
    // resolve_orientation still owns the evidence-based cross-class flip; this only fixes which representative is
    // REPORTED for continuity. (Set TableModel.OrientationContinuityFold=true to A/B against the Σ-weighted fold.)
    if (params_.orientation_continuity_fold)
    {
        constexpr float kYawScale2 = 0.5f * 0.5f;   // rad²: a slow genuine rotation is cheap; a 90/180 snap is not
        const float sz2 = std::max(1e-6f, params_.prior_size_std * params_.prior_size_std);
        const auto cont = [&](const TableBeliefState& r) -> float
        {
            const float dy = wrap(r.yaw - prior_mean_(5));
            const float dw = r.w - prior_mean_(3), dh = r.h - prior_mean_(4);
            return dy * dy / kYawScale2 + (dw * dw + dh * dh) / sz2;
        };
        int best = 0; float best_c = cont(reps[0]);
        for (int i = 1; i < 4; ++i)
            if (const float c = cont(reps[i]); c < best_c) { best_c = c; best = i; }
        s = reps[best];
        return;
    }

    const Eigen::Matrix<float, 6, 6> Pinv = Sigma_.inverse();   // Σ_pred⁻¹ (Σ_ still holds the predicted cov here)
    const auto mahalanobis = [&](const TableBeliefState& r) -> float
    {
        Eigen::Matrix<float, 6, 1> d = r.vec() - prior_mean_;
        d(5) = wrap(r.yaw - prior_mean_(5));   // yaw difference must wrap
        return d.dot(Pinv * d);
    };

    int best = 0; float best_m = mahalanobis(reps[0]);
    for (int i = 1; i < 4; ++i)
        if (const float m = mahalanobis(reps[i]); m < best_m) { best_m = m; best = i; }
    s = reps[best];
}

// ─── Free energy & orientation-mode resolution ───────────────────────────────────────────────────

// TRUE free energy: mean per-point negative-log marginal likelihood of the mixture, −log Σ_k π_k N(d_k;0,R),
// INCLUDING the clutter component (so F rises with misfit).
//
// The old readout summed only the surface prims weighted by responsibility, so a misfit point (routed to
// clutter, r_surface≈0) contributed ~0 and a badly-fit model read FE≈0 — blind to the very errors that matter.
// Here a point far from every surface falls to the clutter floor u[5]=ε·exp(−cs²/2R), so −log(sum) is large:
// the energy now rises with misfit, so mode comparison (e_swap−e_now) and any F readout are meaningful. The
// constant (2πR)^{−3/2} normaliser is dropped — it cancels in every comparison and in Δ over frames.
float TableBelief::mean_energy(const std::vector<Eigen::Vector3f>& pts, const TableBeliefState& s, float R) const
{
    if (pts.empty()) return 0.0f;
    std::array<float, 6> u{};
    double e = 0.0;
    for (const auto& p : pts)
    {
        const float sum = mixture_unnormalized(p, s, R, u);
        e += -std::log(std::max(sum, 1e-12f));
    }
    return static_cast<float>(e / static_cast<double>(pts.size()));
}

// One sequential-Bayesian step over the two orientation modes — current [(w,h,ψ)] vs the w↔h swap [(h,w,ψ)]
// (≡ 90° rotation, the near-square ambiguity) — adopting whichever has the lower accumulated energy.
//
// Each frame adds the log-evidence FOR the current mode ∝ (E_swap − E_now): >0 when current fits better, ≈0
// when the footprint is near-square (the two fit alike), <0 when the swap fits better. Accumulation gives the
// belief MEMORY — one partial/biased frame can't flip it (what stops the per-frame snapping). Boundary = zero
// accumulated evidence (the MAP over the mode); NO tuned threshold. Mirrors chair's resolve_orientation
// (180°) with the table's 90° / w↔h symmetry.
bool TableBelief::resolve_orientation(const std::vector<Eigen::Vector3f>& pts, float R, float evidence_weight)
{
    if (pts.empty()) return false;
    TableBeliefState swap = state_;
    std::swap(swap.w, swap.h);   // same yaw; (h,w,ψ) is the competing class (≡ (w,h,ψ+π/2))
    const float e_now  = mean_energy(pts, state_, R);
    const float e_swap = mean_energy(pts, swap,   R);

    // Sequential Bayesian mode evidence on the TRUE free energy (e_swap−e_now is now a real, correctly-signed
    // log-likelihood-ratio, not the noise the broken FE produced — so NO leak / discriminability / hysteresis
    // band-aids are needed; a near-square footprint gives e_swap≈e_now → the accumulator rests near 0 at the
    // honest ambiguity). evidence_weight ∈ [0,1] is the continuous ego-motion reliability (a smeared frame
    // barely votes). Bounded so it stays a finite memory that a genuine aspect change can still overturn.
    flip_evidence_ += std::clamp(evidence_weight, 0.0f, 1.0f) * (e_swap - e_now);
    flip_evidence_ = std::clamp(flip_evidence_, -6.0f, 6.0f);
    if (flip_evidence_ < 0.0f)   // the swapped mode has the lower accumulated free energy
    {
        std::swap(state_.w, state_.h);
        // Carry Σ into the new labelling: swap the w,h rows AND columns (DOF indices 3,4).
        Sigma_.row(3).swap(Sigma_.row(4));
        Sigma_.col(3).swap(Sigma_.col(4));
        flip_evidence_ = -flip_evidence_;   // relabel: the accumulated evidence now backs the new current mode
        return true;
    }
    return false;
}

// p of the ALTERNATIVE (swapped) mode from the accumulated log-evidence: σ(−flip_evidence_).
float TableBelief::mode_posterior() const
{
    return 1.0f / (1.0f + std::exp(flip_evidence_));
}

// REPORTED yaw variance: the within-mode Σ(5,5) plus the discrete-mode entropy p(1−p)(π/2)². Near-square →
// p≈½ → adds ~(π/2)²/4 (σ≈45°, honest); once evidence resolves the mode → p→0 → collapses to Σ(5,5).
float TableBelief::yaw_marginal_var() const
{
    const float p = mode_posterior();
    constexpr float kHalfPi = 0.5f * static_cast<float>(M_PI);
    return Sigma_(5, 5) + p * (1.0f - p) * kHalfPi * kHalfPi;
}

Eigen::Matrix<float, 6, 6> TableBelief::covariance_reported() const
{
    Eigen::Matrix<float, 6, 6> S = Sigma_;
    S(5, 5) = yaw_marginal_var();
    return S;
}

// ─── Self-test ───────────────────────────────────────────────────────────────────────────────────

// Isolated Eigen unit test: recovers a synthetic table pose + exercises each factor. Prints PASS/FAIL.
bool TableBelief::self_test()
{
    std::mt19937 rng(12345);
    std::normal_distribution<float> noise(0.0f, 0.01f);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f), U01(0.0f, 1.0f);

    TableBeliefParams P;
    const TableBeliefState gt{0.20f, -0.30f, 0.74f, 1.50f, 1.00f, 0.30f};   // ground truth (w>h, canonical)

    const float c = std::cos(gt.yaw), sn = std::sin(gt.yaw);
    const auto to_world = [&](float lx, float ly, float lz) -> Eigen::Vector3f
    { return {gt.cx + c * lx - sn * ly, gt.cy + sn * lx + c * ly, lz}; };

    std::vector<Eigen::Vector3f> pts;
    // Top surface
    for (int i = 0; i < 1200; ++i)
    {
        const float lx = U(rng) * 0.5f * gt.w, ly = U(rng) * 0.5f * gt.h;
        pts.push_back(to_world(lx, ly, gt.H + noise(rng)));
    }
    // Legs (4 cylinders)
    for (int k = 0; k < 4; ++k)
    {
        const float sx = (k == 0 || k == 3) ? 1.f : -1.f, sy = (k < 2) ? 1.f : -1.f;
        const float cxk = sx * (0.5f * gt.w - P.leg_radius), cyk = sy * (0.5f * gt.h - P.leg_radius);
        for (int i = 0; i < 200; ++i)
        {
            const float ang = U(rng) * static_cast<float>(M_PI);
            const float z = U01(rng) * (gt.H - P.top_thickness);
            pts.push_back(to_world(cxk + P.leg_radius * std::cos(ang) + noise(rng),
                                   cyk + P.leg_radius * std::sin(ang) + noise(rng), z));
        }
    }
    // Clutter (floor / off-table)
    std::vector<Eigen::Vector3f> clutter;
    for (int i = 0; i < 150; ++i)
        clutter.push_back({gt.cx + U(rng) * 1.5f, gt.cy + U(rng) * 1.5f, U01(rng) * 0.05f});
    std::vector<Eigen::Vector3f> all = pts;
    all.insert(all.end(), clutter.begin(), clutter.end());

    bool ok = true;
    auto check = [&](bool cond, const char* msg) { if (!cond) { ok = false; std::printf("  FAIL: %s\n", msg); } };

    // (a) SDF correctness at known points
    {
        TableBelief b(gt, P);
        const float s_center = b.sdf_compound(to_world(0, 0, gt.H - 0.5f * P.top_thickness), gt);
        check(s_center < -1e-3f, "top-slab centre SDF should be negative (inside)");
        const float s_surf = b.sdf_top(to_world(0.2f * gt.w, 0.1f * gt.h, gt.H), gt);
        check(std::abs(s_surf) < 1e-2f, "on-surface SDF should be ~0");
        const float s_far = b.sdf_compound({gt.cx + 3.0f, gt.cy, 1.5f}, gt);
        check(s_far > 1.0f, "far point SDF should be large positive");
    }

    // (b) Jacobian: coarse FD (used in solver) vs fine FD — must agree
    {
        TableBelief b(gt, P);
        const Eigen::Vector3f q = to_world(0.45f * gt.w, 0.0f, gt.H);   // near a top edge → nonzero gradient
        const auto Jc = b.sdf_jacobian(q, gt, 0);
        TableBeliefParams Pf = P; Pf.fd_eps = 1e-4f;
        TableBelief bf(gt, Pf);
        const auto Jf = bf.sdf_jacobian(q, gt, 0);
        check((Jc - Jf).norm() < 1e-2f, "SDF Jacobian (coarse vs fine FD) disagree");
    }

    // (c) Fit recovery from a perturbed init. The engine's Schur-consistent mean step (b_eff) applies the
    // proper Bayesian gain rather than the old per-frame ML refit, so cold-start convergence is slower but
    // unbiased (verified: |err| → mm by ~30 frames, exact by ~60); the aggressive 8-frame recovery the old
    // full-gradient step gave is gone (that was what chased per-frame shared bias into the position wander).
    TableBelief belief(TableBeliefState{0.0f, 0.0f, 0.75f, 1.0f, 0.6f, 0.0f}, P);
    TableFrame frame; frame.points = all;
    float e = 0.0f;
    for (int it = 0; it < 40; ++it) e = belief.update(frame);
    const auto& s = belief.state();
    std::printf("  recovered: cx=%.3f cy=%.3f H=%.3f w=%.3f h=%.3f yaw=%.3f  (energy=%.4f)\n",
                s.cx, s.cy, s.H, s.w, s.h, s.yaw, e);
    std::printf("  truth:     cx=%.3f cy=%.3f H=%.3f w=%.3f h=%.3f yaw=%.3f\n",
                gt.cx, gt.cy, gt.H, gt.w, gt.h, gt.yaw);
    check(std::abs(s.w - gt.w)   < 0.06f, "w not recovered");
    check(std::abs(s.h - gt.h)   < 0.06f, "h not recovered");
    check(std::abs(s.H - gt.H)   < 0.03f, "H not recovered");
    check(std::abs(s.cx - gt.cx) < 0.03f, "cx not recovered");
    check(std::abs(s.cy - gt.cy) < 0.03f, "cy not recovered");
    check(std::abs(std::remainder(s.yaw - gt.yaw, static_cast<float>(M_PI))) < 0.06f, "yaw not recovered");

    // (d) Clutter responsibility: a far floor point is mostly clutter
    {
        const auto r = belief.responsibilities({gt.cx + 1.2f, gt.cy + 1.2f, 0.02f}, belief.state(),
                                               P.sigma_base_m * P.sigma_base_m);
        std::printf("  clutter-point responsibilities: top=%.2f legs=%.2f clutter=%.2f\n",
                    r[0], r[1] + r[2] + r[3] + r[4], r[5]);
        check(r[5] > 0.5f, "far floor point should be mostly clutter");
    }

    // (e) Σ finite & SPD
    {
        const Eigen::Matrix<float, 6, 6>& S = belief.covariance();
        check(S.allFinite(), "Σ not finite");
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 6, 6>> es(S);
        check(es.eigenvalues().minCoeff() > 0.0f, "Σ not SPD");
        std::printf("  Σ diag (std, m/rad): cx=%.3f cy=%.3f H=%.3f w=%.3f h=%.3f yaw=%.3f\n",
                    std::sqrt(S(0,0)), std::sqrt(S(1,1)), std::sqrt(S(2,2)),
                    std::sqrt(S(3,3)), std::sqrt(S(4,4)), std::sqrt(S(5,5)));
    }

    // (f) NBV: the D-optimal gain ½·ln det(I+Σ·ΔI) picks the face perpendicular to the most-uncertain
    // extent — observe the +x face when w is uncertain, the +y face when h is uncertain.
    {
        const auto& bs = belief.state();
        const float cc = std::cos(bs.yaw), ssn = std::sin(bs.yaw);
        // sample the slab side band (z∈[H−t, H]) of the +x (axis 0) or +y (axis 1) face → clean w/h signal
        auto face_pts = [&](int axis) {
            std::vector<Eigen::Vector3f> fp;
            for (int m = 0; m < 10; ++m)
            {
                const float t = -1.0f + 2.0f * m / 9.0f;
                const float lx = (axis == 0) ? 0.5f * bs.w : t * 0.5f * bs.w;
                const float ly = (axis == 0) ? t * 0.5f * bs.h : 0.5f * bs.h;
                for (int k = 0; k < 4; ++k)
                {
                    const float z = bs.H - P.top_thickness * (1.0f - static_cast<float>(k) / 3.0f);
                    fp.push_back({bs.cx + cc * lx - ssn * ly, bs.cy + ssn * lx + cc * ly, z});
                }
            }
            return fp;
        };
        const float Rn = P.sigma_base_m * P.sigma_base_m;
        const auto dIx = belief.predicted_information(face_pts(0), Rn);
        const auto dIy = belief.predicted_information(face_pts(1), Rn);
        const auto gain = [](const Eigen::Matrix<float, 6, 6>& S, const Eigen::Matrix<float, 6, 6>& dI)
        { return 0.5f * std::log(std::max(1e-9f, (Eigen::Matrix<float, 6, 6>::Identity() + S * dI).determinant())); };
        Eigen::Matrix<float, 6, 6> Sw = Eigen::Matrix<float, 6, 6>::Identity() * 1e-4f; Sw(3, 3) = 0.25f;  // w uncertain
        Eigen::Matrix<float, 6, 6> Sh = Eigen::Matrix<float, 6, 6>::Identity() * 1e-4f; Sh(4, 4) = 0.25f;  // h uncertain
        std::printf("  NBV gains: w-unc(+x=%.3f +y=%.3f)  h-unc(+x=%.3f +y=%.3f)\n",
                    gain(Sw, dIx), gain(Sw, dIy), gain(Sh, dIx), gain(Sh, dIy));
        check(gain(Sw, dIx) > gain(Sw, dIy), "NBV: w uncertain → +x face should win");
        check(gain(Sh, dIy) > gain(Sh, dIx), "NBV: h uncertain → +y face should win");
    }

    // (h) LiDAR first-hit range factor corrects the MASK-EROSION extent UNDER-SIZE — the reason this channel
    //     was ported to the table. A YOLO mask whose boundary sits INSIDE the true rim yields a top cloud that
    //     is uniformly shrunk, so a mask-only fit under-sizes the extent. LiDAR returns on the TRUE front rim
    //     pin it in metric range (residual measured ALONG the ray), pushing the face back out. The camera is
    //     on the −y side, so the y-extent h is the corrected DOF. Verify: with LiDAR, h recovers toward truth
    //     and clearly beats mask-only. Exercises the N=6 accumulate_lidar_rays path (accumulate_extra hook).
    {
        const float Rb    = P.sigma_base_m * P.sigma_base_m;
        const float erode = 0.90f;   // mask boundary ~5% inside the true rim each side (silhouette erosion)
        std::vector<Eigen::Vector3f> eroded_top;
        for (int i = 0; i < 1200; ++i)
        {
            const float lx = U(rng) * 0.5f * gt.w * erode, ly = U(rng) * 0.5f * gt.h * erode;
            eroded_top.push_back(to_world(lx, ly, gt.H + noise(rng)));
        }
        // Sensor on the −y side; LiDAR returns on the TRUE front rim (slab front vertical face at ly=−h/2).
        const Eigen::Vector3f o3 = to_world(0.0f, -0.5f * gt.h - 1.0f, gt.H - 0.2f);
        std::vector<Eigen::Vector3f> rays;
        for (int i = 0; i < 60; ++i)
        {
            const float lx = U(rng) * 0.5f * gt.w;                          // across the TRUE front-face width
            const float z  = gt.H - U01(rng) * P.top_thickness;            // within the slab band (guaranteed hit)
            rays.push_back(to_world(lx, -0.5f * gt.h + noise(rng), z + noise(rng)));   // TRUE front face
        }
        const TableBeliefState init{gt.cx, gt.cy, gt.H, gt.w, gt.h, gt.yaw};   // start AT truth ⇒ only erosion pulls
        auto fit = [&](bool with_lidar) {
            TableBelief b(init, P);
            TableFrame fr; fr.points = eroded_top; fr.R.assign(eroded_top.size(), Rb);
            if (with_lidar) { fr.lidar.origin = o3; fr.lidar.endpoints = rays; fr.lidar.precision = 2500.0f; }
            for (int it = 0; it < 12; ++it) b.update(fr);
            return b.state().h;
        };
        const float h_no  = fit(false);
        const float h_lid = fit(true);
        std::printf("  mask-erosion extent(h): mask-only=%.3f  with-lidar=%.3f  (truth=%.3f)\n", h_no, h_lid, gt.h);
        check(h_no < gt.h - 0.02f, "mask-only should UNDER-size h (erosion)");
        check(std::abs(h_lid - gt.h) < 0.6f * std::abs(h_no - gt.h), "lidar should clearly beat mask-only extent");
    }

    // (i) Coverage / traction (table_1.png): an UNDER-covering model whose mask extends well beyond it must be
    //     PULLED OUT to cover the on-plane points (the clutter-escape fix), grow-only and self-bounded, while
    //     OFF-plane contamination is ignored.
    {
        const float Rb = P.sigma_base_m * P.sigma_base_m;
        // Full-extent tabletop cloud (points out to the TRUE rim), + off-plane contamination far to the side.
        std::vector<Eigen::Vector3f> plane;
        for (int i = 0; i < 1500; ++i)
        {
            const float lx = U(rng) * 0.5f * gt.w, ly = U(rng) * 0.5f * gt.h;
            plane.push_back(to_world(lx, ly, gt.H + noise(rng)));
        }
        std::vector<Eigen::Vector3f> contam = plane;               // same cloud + off-plane blob 0.6 m out in +x
        for (int i = 0; i < 300; ++i)
            contam.push_back(to_world(0.5f * gt.w + 0.6f + U(rng) * 0.1f, U(rng) * 0.3f, 0.30f + noise(rng)));
        // OFFSET + under-covering start (the table_1.png geometry): the model sits to one side and small, so the
        // far side of the mask is well beyond clutter_scale → escapes to clutter → the normal fit STALLS in a few
        // frames. Coverage reclaims those points and pulls the model out AND over. Budget-limited (3 frames).
        const TableBeliefState init{gt.cx + 0.30f, gt.cy, gt.H, 0.55f * gt.w, 0.55f * gt.h, gt.yaw};
        auto fit = [&](float cov, const std::vector<Eigen::Vector3f>& pts) {
            TableBeliefParams pp = P; pp.coverage_precision = cov;
            TableBelief b(init, pp);
            TableFrame fr; fr.points = pts; fr.R.assign(pts.size(), Rb);
            for (int it = 0; it < 3; ++it) b.update(fr);
            return b.state();
        };
        const auto s_no  = fit(0.0f, plane);
        const auto s_cov = fit(500.0f, plane);
        const auto s_con = fit(500.0f, contam);      // off-plane blob must NOT extra-grow the slab
        std::printf("  coverage 3-frame w: no=%.3f cov=%.3f contam=%.3f (truth %.3f); |cx-gt| no=%.3f cov=%.3f\n",
                    s_no.w, s_cov.w, s_con.w, gt.w, std::abs(s_no.cx - gt.cx), std::abs(s_cov.cx - gt.cx));
        check(s_cov.w > s_no.w + 0.05f,        "coverage should grow w toward the mask extent faster (budget-limited)");
        check(std::abs(s_cov.cx - gt.cx) < std::abs(s_no.cx - gt.cx), "coverage should recenter toward the unexplained side");
        check(std::abs(s_con.w - s_cov.w) < 0.06f, "off-plane contamination must NOT extra-grow the slab (on-plane gate)");
    }

    // (j) Free-space / VACATE (EXISTENCE_BELIEF_PLAN.md Step 4): the shrink-only counter-force that BOUNDS
    //     coverage. An OVER-SEGMENTED mask (YOLO bleeding past the true rim / a coverage runaway) holds the
    //     slab too WIDE; LiDAR beams that pass THROUGH the phantom over-extension (no real tabletop there → they
    //     exit the far face) pull the extent back IN. Beams that RETURN at a rim (occupancy) leave the model be.
    {
        const float Rb   = P.sigma_base_m * P.sigma_base_m;
        const float over = 1.40f;   // model/mask over-extended to 1.4× the true half-width on BOTH x sides
        std::vector<Eigen::Vector3f> mask;
        for (int i = 0; i < 1400; ++i)   // dense TRUE tabletop
        {
            const float lx = U(rng) * 0.5f * gt.w, ly = U(rng) * 0.5f * gt.h;
            mask.push_back(to_world(lx, ly, gt.H + noise(rng)));
        }
        for (int i = 0; i < 300; ++i)    // sparse PHANTOM over-segmentation strips beyond the true ±x rim
        {
            const float side = (i % 2 == 0) ? 1.0f : -1.0f;
            const float lx = side * (0.5f * gt.w + U01(rng) * (over - 1.0f) * 0.5f * gt.w);
            mask.push_back(to_world(lx, U(rng) * 0.5f * gt.h, gt.H + noise(rng)));
        }
        // One sensor FAR on the +y side (x≈0) so the near-parallel −y beams through the ±x phantom columns barely
        // drift in x (they stay OUTSIDE the true rim, never over the real tabletop) and EXIT the far −y face.
        const Eigen::Vector3f o_fs = to_world(0.0f, 50.0f, gt.H - 0.015f);
        std::vector<Eigen::Vector3f> through;
        for (int i = 0; i < 160; ++i)
        {
            const float side = (i % 2 == 0) ? 1.0f : -1.0f;
            const float lx = side * (0.5f * gt.w + (0.15f + U01(rng) * 0.70f) * (over - 1.0f) * 0.5f * gt.w);
            through.push_back(to_world(lx, -0.5f * gt.h - 1.0f, gt.H - 0.015f + noise(rng)));
        }
        const TableBeliefState init{gt.cx, gt.cy, gt.H, over * gt.w, gt.h, gt.yaw};   // start OVER-wide
        auto fit = [&](float fs) {
            TableBeliefParams pp = P; pp.free_space_precision = fs;
            TableBelief b(init, pp);
            TableFrame fr; fr.points = mask; fr.R.assign(mask.size(), Rb);
            if (fs > 0.0f) { fr.lidar.origin = o_fs; fr.lidar.endpoints = through; }
            for (int it = 0; it < 6; ++it) b.update(fr);
            return b.state();
        };
        const auto s_no = fit(0.0f);
        const auto s_fs = fit(800.0f);
        std::printf("  free-space vacate w: no=%.3f fs=%.3f (over-mask=%.3f truth=%.3f) cx no=%.3f fs=%.3f\n",
                    s_no.w, s_fs.w, over * gt.w, gt.w, s_no.cx, s_fs.cx);
        check(s_fs.w < s_no.w - 0.03f,  "free-space should SHRINK the over-segmented slab (vacate)");
        check(s_fs.w > gt.w - 0.10f,    "free-space must not collapse the slab below the true extent (occupancy holds)");
    }

    // (k) Footprint second-moment factor (tables_5.png "stuck yaw"): the LIVE failure — a TOP-FACE-ONLY cloud
    //     (no legs; ZED sees the tabletop, legs self-occluded) at RANGE (~3 m → the common-mode caps per-frame
    //     size/yaw authority: "geometry freezes afar"), from a small axis-aligned birth, in a realistic dwell
    //     (~15 frames). The per-point mixture is degenerate for yaw/extent on a flat top and the range cap
    //     starves what little signal the rim gives → w and yaw barely move. The moment factor measures (w,h,yaw)
    //     globally and saturates each DOF to its (range-limited) ceiling → recovers both, clearly beating baseline.
    {
        const float Rb = P.sigma_base_m * P.sigma_base_m;
        std::vector<Eigen::Vector3f> top;                 // dense filled tabletop, NO legs, NO clutter
        for (int i = 0; i < 2000; ++i)
        {
            const float lx = U(rng) * 0.5f * gt.w, ly = U(rng) * 0.5f * gt.h;
            top.push_back(to_world(lx, ly, gt.H + noise(rng)));
        }
        const float rng_m   = 3.0f;                       // live view range
        const float cc_yaw  = (0.03f * rng_m) * (0.03f * rng_m);   // AI2RangeNoiseYawPerM · range, squared
        const float cc_size = (0.08f * rng_m) * (0.08f * rng_m);   // AI2RangeNoiseSizePerM · range, squared
        const TableBeliefState init{gt.cx, gt.cy, gt.H, 0.60f, 0.40f, 0.0f};   // small, axis-aligned birth
        auto fit = [&](float mom_prec) {
            TableBeliefParams pp = P; pp.footprint_moment_precision = mom_prec;
            TableBelief b(init, pp);
            TableFrame fr; fr.points = top; fr.R.assign(top.size(), Rb);
            fr.chain_cov_yaw = cc_yaw; fr.chain_cov_size = cc_size;   // the range freeze the live fit fights
            fr.moment_extra_var = cc_size;   // realistic per-frame moment variance (range term) — must still recover
            for (int it = 0; it < 40; ++it) b.update(fr);
            return b.state();
        };
        const auto s_off = fit(0.0f);
        const auto s_on  = fit(2000.0f);
        const float dyaw_off = std::abs(std::remainder(s_off.yaw - gt.yaw, static_cast<float>(M_PI)));
        const float dyaw_on  = std::abs(std::remainder(s_on.yaw  - gt.yaw, static_cast<float>(M_PI)));
        std::printf("  moment factor @3m: OFF(w=%.2f h=%.2f yaw=%.3f dyaw=%.3f)  ON(w=%.2f h=%.2f yaw=%.3f dyaw=%.3f)  (gt w=%.2f h=%.2f yaw=%.3f)\n",
                    s_off.w, s_off.h, s_off.yaw, dyaw_off, s_on.w, s_on.h, s_on.yaw, dyaw_on, gt.w, gt.h, gt.yaw);
        check(dyaw_off > 0.10f,                "top-only mixture at range SHOULD lag in yaw (baseline)");
        check(dyaw_on  < 0.05f,                "moment factor should recover yaw from a top-only cloud at range");
        check(std::abs(s_on.w - gt.w) < 0.20f, "moment factor should recover w");
        check(std::abs(s_on.h - gt.h) < 0.20f, "moment factor should recover h");
        check(dyaw_on < 0.4f * dyaw_off,       "moment factor should clearly beat the mixture-only baseline in yaw");
    }

    // (l) Footprint moment is UNCONDITIONALLY GROW-ONLY: a mask is only ever a LOWER BOUND on the extent (a
    //     partial / trimmed / foreshortened / under-segmented view can only shorten the observed footprint,
    //     never lengthen it), so the moment factor may only GROW w/h, never shrink — no completeness/trunc
    //     signal is consulted. A PARTIAL view must HOLD a converged table; a LARGER mask must GROW it. Shrink
    //     is delegated to the occlusion-aware vacate channel, not the mask.
    {
        rng.seed(4242);   // deterministic: this block's result must not depend on prior blocks' RNG consumption
        const float Rb = P.sigma_base_m * P.sigma_base_m;
        auto top = [&](float w, float h, float keep) {                // keep: clip |lx| ≤ 0.5·w·keep (border trim)
            std::vector<Eigen::Vector3f> pts;
            for (int i = 0; i < 2000; ++i)
            { const float lx = U(rng) * 0.5f * w, ly = U(rng) * 0.5f * h;
              if (std::abs(lx) > 0.5f * w * keep) continue;
              pts.push_back(to_world(lx, ly, gt.H + noise(rng))); }
            return pts;
        };
        TableBeliefParams pm = P; pm.footprint_moment_precision = 2000.0f;
        auto fit = [&](float ext_w, float keep) {
            TableBelief b(TableBeliefState{gt.cx, gt.cy, gt.H, gt.w, gt.h, gt.yaw}, pm);   // converged at truth
            for (int it = 0; it < 45; ++it)
            { auto pts = top(ext_w, gt.h, keep);
              TableFrame fr; fr.points = pts; fr.R.assign(pts.size(), Rb);
              fr.chain_cov_yaw = (0.03f * 3) * (0.03f * 3); fr.chain_cov_size = (0.08f * 3) * (0.08f * 3);
              fr.moment_extra_var = (0.03f * 3) * (0.03f * 3);
              b.update(fr); }
            return b.state().w;
        };
        const float w_trim = fit(gt.w, 0.55f);  // border-trimmed partial view ⇒ must HOLD ~gt.w
        const float w_part = fit(gt.w, 0.55f);  // SIDE-VIEW partial view ⇒ HOLD
        const float w_grow = fit(2.0f, 1.0f);   // a LARGER full mask ⇒ must GROW toward 2.0
        std::printf("  moment grow-only: trim-partial w=%.2f side-partial w=%.2f (gt=%.2f, must hold)  larger-mask w=%.2f (must grow >gt)\n",
                    w_trim, w_part, gt.w, w_grow);
        check(w_trim > gt.w - 0.12f,   "border-trimmed partial mask must NOT shrink a converged table");
        check(w_part > gt.w - 0.12f,   "SIDE-VIEW partial mask (in-view, trunc=0) must NOT shrink — the live failure");
        check(w_grow > gt.w + 0.10f,   "a genuinely LARGER mask must still GROW the extent");
    }

    // (m) COMPLETENESS backoff: a converged table hit by a FRAGMENTARY footprint (partial/foreshortened view)
    //     whose sliver principal-axis is rotated OFF the believed yaw must NOT rotate the belief when the
    //     completeness gain is ON, whereas with the gain OFF the moment snaps yaw toward the sliver (the live
    //     "table rotates when going away / returning from a detour" failure — ai2_log.csv completeness≈0.01).
    {
        rng.seed(9090);
        const float Rb = P.sigma_base_m * P.sigma_base_m;
        // Thin sliver ~30° off the table's yaw: elongated along dir(gt.yaw+off), a few cm wide → high aniso,
        // completeness ≈ (major·minor)/(w·h) ≪ 1. This is what a foreshortened partial view of the top looks like.
        // The sliver is kept fully INSIDE the believed top rectangle so the per-point SDF mixture has ~zero yaw
        // gradient (interior top points: ∂sdf/∂yaw≈0) — exactly the live regime where the per-point channel is
        // inert (dyaw_points≈0) and ONLY the global moment statistic can rotate the box. Verified: with the moment
        // OFF this sliver moves yaw by 0.000 rad; the whole snap below is the moment's doing.
        const float off = 0.52f;                                   // ~30° offset from the converged yaw
        const float dc = std::cos(gt.yaw + off), ds = std::sin(gt.yaw + off);
        auto sliver = [&]() {
            std::vector<Eigen::Vector3f> pts;
            for (int i = 0; i < 1500; ++i)
            { const float t = U(rng) * 0.30f * gt.w;               // major dir, kept inside the box
              const float n = U(rng) * 0.03f;                      // ~3 cm minor → sliver, interior
              pts.push_back({gt.cx + dc * t - ds * n, gt.cy + ds * t + dc * n, gt.H + noise(rng)}); }
            return pts;
        };
        auto fit_yaw = [&](float compl_gain) {
            TableBeliefParams pm = P; pm.footprint_moment_precision = 2000.0f;
            pm.footprint_moment_completeness_gain = compl_gain;
            TableBelief b(TableBeliefState{gt.cx, gt.cy, gt.H, gt.w, gt.h, gt.yaw}, pm);   // converged at truth
            for (int it = 0; it < 30; ++it)
            { auto pts = sliver();
              TableFrame fr; fr.points = pts; fr.R.assign(pts.size(), Rb);
              fr.chain_cov_yaw = (0.03f * 3) * (0.03f * 3); fr.chain_cov_size = (0.08f * 3) * (0.08f * 3);
              fr.moment_extra_var = (0.03f * 3) * (0.03f * 3);
              b.update(fr); }
            return std::abs(std::remainder(b.state().yaw - gt.yaw, static_cast<float>(M_PI)));
        };
        const float dyaw_off = fit_yaw(0.0f);   // gain OFF ⇒ sliver snaps yaw toward its (wrong) principal axis
        const float dyaw_on  = fit_yaw(0.5f);   // gain ON  ⇒ fragmentary footprint can no longer rotate the box
        std::printf("  completeness backoff: dyaw OFF=%.3f rad  ON=%.3f rad (sliver ~%.2f rad off yaw)\n",
                    dyaw_off, dyaw_on, off);
        check(dyaw_off > 0.15f,            "sliver SHOULD snap yaw with completeness backoff OFF (baseline)");
        check(dyaw_on  < 0.5f * dyaw_off,  "completeness backoff must hold yaw against a fragmentary footprint");
        check(dyaw_on  < 0.10f,            "completeness backoff should keep yaw near the converged truth");
    }

    std::printf("TableBelief::self_test %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace rc
