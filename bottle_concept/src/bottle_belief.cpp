/*
 * bottle_belief.cpp  —  AI2 bottle belief (mirrors table_belief.cpp for a single vertical cylinder)
 */

#include "bottle_belief.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

namespace rc
{

// ─── SDF primitive (scalar, room frame) ─────────────────────────────────────────

namespace
{
// Vertical cylinder SDF: dl = (room point − cylinder centre), radius r, half-height hh.
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

float BottleBelief::sdf_cylinder(const Eigen::Vector3f& p, const BottleBeliefState& s) const
{
    return cylinder_sdf(p.x() - s.cx, p.y() - s.cy, p.z() - s.cz, s.radius, 0.5f * s.height);
}

float BottleBelief::sdf_prim(const Eigen::Vector3f& p, const BottleBeliefState& s, int /*prim*/) const
{
    return sdf_cylinder(p, s);   // single primitive
}

// ─── Mixture responsibilities ───────────────────────────────────────────────────

std::array<float, 2> BottleBelief::mixture_unnorm(const Eigen::Vector3f& p, const BottleBeliefState& s, float R) const
{
    const float eps     = std::clamp(params_.clutter_frac, 0.0f, 0.99f);
    const float pi_surf = 1.0f - eps;
    const float inv2R   = 0.5f / std::max(1e-9f, R);

    std::array<float, 2> u{};
    const float d = sdf_cylinder(p, s);
    u[0] = pi_surf * std::exp(-d * d * inv2R);
    // Clutter = uniform; modelled as a Gaussian floor at clutter_scale so a point further than that from
    // the surface is explained by clutter rather than dragging the model.
    const float cs = params_.clutter_scale_m;
    u[1] = eps * std::exp(-cs * cs * inv2R);
    return u;   // UNNORMALIZED
}

std::array<float, 2> BottleBelief::responsibilities(const Eigen::Vector3f& p, const BottleBeliefState& s, float R) const
{
    std::array<float, 2> u = mixture_unnorm(p, s, R);
    const float sum = u[0] + u[1];
    if (sum <= 0.0f) { u = {0.0f, 1.0f}; return u; }
    u[0] /= sum; u[1] /= sum;
    return u;
}

// Mean per-point −log marginal (clutter included): F rises with misfit. The common Gaussian normaliser
// cancels in every Δ/comparison, so the raw unnormalised sum suffices.
float BottleBelief::mixture_nll(const std::vector<Eigen::Vector3f>& pts, const BottleBeliefState& s, float R) const
{
    if (pts.empty()) return 0.0f;
    double nll = 0.0;
    for (const auto& p : pts)
    {
        const auto u = mixture_unnorm(p, s, R);
        nll += -std::log(std::max(1e-30, static_cast<double>(u[0]) + static_cast<double>(u[1])));
    }
    return static_cast<float>(nll / static_cast<double>(pts.size()));
}

float BottleBelief::clutter_fraction(const std::vector<Eigen::Vector3f>& pts, float R) const
{
    if (pts.empty()) return 0.0f;
    double c = 0.0;
    for (const auto& p : pts) c += responsibilities(p, state_, R)[1];   // index 1 = clutter
    return static_cast<float>(c / static_cast<double>(pts.size()));
}

// ─── Jacobian (central finite difference) ──────────────────────────────────────

Eigen::Matrix<float, 5, 1> BottleBelief::sdf_jacobian(const Eigen::Vector3f& p, const BottleBeliefState& s, int prim) const
{
    Eigen::Matrix<float, 5, 1> J;
    const Eigen::Matrix<float, 5, 1> base = s.vec();
    const float e = params_.fd_eps;
    for (int j = 0; j < 5; ++j)
    {
        Eigen::Matrix<float, 5, 1> vp = base, vm = base;
        vp(j) += e; vm(j) -= e;
        J(j) = (sdf_prim(p, BottleBeliefState::from_vec(vp), prim) -
                sdf_prim(p, BottleBeliefState::from_vec(vm), prim)) / (2.0f * e);
    }
    return J;
}

// ─── Constraints / canonical form ───────────────────────────────────────────────

void BottleBelief::apply_constraints(BottleBeliefState& s) const
{
    s.radius = std::max(s.radius, params_.min_radius);
    s.height = std::max(s.height, params_.min_height);
}

// ─── Occluding-contour silhouette factor (the radius-pinning term) ──────────────
// For each mask edge ray (origin = camera centre, direction d in the horizontal plane), the tangency
// condition is: perpendicular distance from the cylinder axis (cx,cy) to the ray equals the radius.
//   g    = d × (axis − cam) = d.x·(cy−oy) − d.y·(cx−ox)      (signed perpendicular distance, d unit)
//   r    = |g| − radius                                       (tangent residual)
//   ∂|g|/∂cx = sign(g)·(−d.y),  ∂|g|/∂cy = sign(g)·d.x,  ∂r/∂radius = −1
// Each edge ray is an INDEPENDENT tangent measurement at precision sil_precision — the SAME scale as the
// depth term's 1/R (config MaskPrecision=500 vs depth 1/σ²=2500). Do NOT divide by the ray count: that
// made the whole silhouette ~1/N as strong as a single depth point and the depth mass buried it (radius
// blew to 0.15 m live). The shared-mask correlation across rays is de-correlated by the common-mode
// Woodbury saturation in the engine, exactly as it is for the correlated depth points.
void BottleBelief::accumulate_extra(const BottleBeliefState& s, const BottleFrame& f,
                                    Eigen::Matrix<float, 5, 5>& Id, Eigen::Matrix<float, 5, 1>& bd) const
{
    // Occluding-contour silhouette tangent term (skips its own block if disabled; the LiDAR channel below
    // is independent and must still run).
    if (f.sil_precision > 0.0f and not f.sil_dirs.empty())
    {
        const float w = f.sil_precision;   // per-ray precision (NOT ÷N)
        for (const auto& draw : f.sil_dirs)
        {
            const float dn = draw.norm();
            if (dn < 1e-6f) continue;
            const Eigen::Vector2f d = draw / dn;                 // unit ray direction (horizontal)
            const float g   = d.x() * (s.cy - f.sil_cam_xy.y()) - d.y() * (s.cx - f.sil_cam_xy.x());
            const float sgn = (g >= 0.0f) ? 1.0f : -1.0f;
            const float r   = std::abs(g) - s.radius;            // tangent residual
            Eigen::Matrix<float, 5, 1> J = Eigen::Matrix<float, 5, 1>::Zero();
            J(0) = sgn * (-d.y());   // ∂/∂cx
            J(1) = sgn * ( d.x());   // ∂/∂cy
            J(3) = -1.0f;            // ∂/∂radius
            Id.noalias() += w * (J * J.transpose());
            bd.noalias() += -w * J * r;
        }
    }

    // Second, YOLO-independent evidence channel: LiDAR first-hit range. Sphere-traces THIS belief's own SDF,
    // so the exact same call drops into table/chair accumulate_extra unchanged. No-op if f.lidar.precision==0.
    rc::ai::accumulate_lidar_rays<5>(*this, s, f.lidar, Id, bd);
}

// ─── Engine hooks: prior cov, process noise (F = I, static), common-mode ─────────

Eigen::Matrix<float, 5, 1> BottleBelief::prior_cov_diag() const
{
    const float pp = params_.prior_pos_std  * params_.prior_pos_std;    // cx, cy, cz
    const float ss = params_.prior_size_std * params_.prior_size_std;   // radius, height
    return (Eigen::Matrix<float, 5, 1>() << pp, pp, pp, ss, ss).finished();
}

Eigen::Matrix<float, 5, 1> BottleBelief::process_noise_diag() const
{
    // Position: marginalise the two-component transition mixture over P(being carried). VARIANCES mix, not
    // stds — variance is what is additive. p_mover = 0 (nothing can have moved it) reproduces the previous
    // behaviour exactly, so this is a strict generalisation.
    const float q_static = params_.process_std_m       * params_.process_std_m;
    const float q_moved  = params_.process_std_moved_m * params_.process_std_moved_m;
    const float qp = (1.0f - mover_p_) * q_static + mover_p_ * q_moved;        // cx,cy,cz
    const float qs = params_.process_std_size_m * params_.process_std_size_m;  // radius,height (rigid → sticky)
    return (Eigen::Matrix<float, 5, 1>() << qp, qp, qp, qs, qs).finished();
}

// Inverse of the per-frame common-mode covariance Σ_c (diagonal): position (cx,cy,cz) = config floor +
// pose-chain variance; size (radius,height) = config std. Marginalising this SHARED error (via Woodbury in
// the engine) makes the frame's information SATURATE at Σ_c regardless of point count — N correlated points
// cannot collapse σ.
Eigen::Matrix<float, 5, 1> BottleBelief::common_mode_inv_diag(const BottleFrame& frame) const
{
    // Range de-rating inflates the shared-error covariance (pos AND size) so a distant frame's information
    // SATURATES lower — the frame can barely move the mean → far objects hold, close ones refine.
    const float sc = std::max(1.0f, frame.precision_scale);
    const float p2 = params_.common_mode_pos_std  * params_.common_mode_pos_std  * sc;
    const float s2 = params_.common_mode_size_std * params_.common_mode_size_std * sc;
    // Size (radius,height) shared error: config floor (range-derated) + ego-motion smear (frame.chain_cov_size).
    // Marginalising it caps how far a MOVING frame can grow/shrink the bottle — confirm, don't reshape.
    const float inv_size = 1.0f / std::max(1e-9f, s2 + frame.chain_cov_size);
    return (Eigen::Matrix<float, 5, 1>() <<
            1.0f / std::max(1e-9f, p2 + frame.chain_cov_xx),
            1.0f / std::max(1e-9f, p2 + frame.chain_cov_yy),
            1.0f / std::max(1e-9f, p2),
            inv_size, inv_size).finished();
}

// ─── Self-test ───────────────────────────────────────────────────────────────────

bool BottleBelief::self_test()
{
    std::mt19937 rng(12345);
    std::normal_distribution<float> noise(0.0f, 0.005f);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f), U01(0.0f, 1.0f);

    BottleBeliefParams P;
    const BottleBeliefState gt{0.20f, -0.30f, 0.85f, 0.045f, 0.22f};   // ground truth

    std::vector<Eigen::Vector3f> pts;
    // Full lateral surface (2π arc, full height span) — defines radius + the vertical extent.
    for (int i = 0; i < 1600; ++i)
    {
        const float a  = 2.0f * static_cast<float>(M_PI) * U01(rng);
        const float z  = gt.cz + 0.5f * gt.height * U(rng);
        pts.push_back({gt.cx + gt.radius * std::cos(a) + noise(rng),
                       gt.cy + gt.radius * std::sin(a) + noise(rng), z + noise(rng)});
    }
    for (int cap = 0; cap < 2; ++cap)   // top + bottom caps → both z faces are constrained
    {
        const float zc = gt.cz + (cap == 0 ? 0.5f : -0.5f) * gt.height;
        for (int i = 0; i < 150; ++i)
        {
            const float rr = gt.radius * std::sqrt(U01(rng)), a = 2.0f * static_cast<float>(M_PI) * U01(rng);
            pts.push_back({gt.cx + rr * std::cos(a), gt.cy + rr * std::sin(a), zc + noise(rng)});
        }
    }
    // Clutter on the floor (far below the cylinder, off to the sides) — unambiguously not the bottle.
    std::vector<Eigen::Vector3f> all = pts;
    for (int i = 0; i < 120; ++i)
        all.push_back({gt.cx + U(rng) * 0.8f, gt.cy + U(rng) * 0.8f, 0.02f * U01(rng)});

    bool ok = true;
    auto check = [&](bool cond, const char* msg) { if (!cond) { ok = false; std::printf("  FAIL: %s\n", msg); } };

    // (a) SDF correctness at known points
    {
        BottleBelief b(gt, P);
        check(b.sdf_cylinder({gt.cx, gt.cy, gt.cz}, gt) < -1e-3f, "axis-centre SDF should be negative (inside)");
        check(std::abs(b.sdf_cylinder({gt.cx + gt.radius, gt.cy, gt.cz}, gt)) < 1e-2f, "on-surface SDF should be ~0");
        check(b.sdf_cylinder({gt.cx + 1.0f, gt.cy, gt.cz}, gt) > 0.5f, "far point SDF should be large positive");
    }

    // (b) Jacobian: coarse FD (used in solver) vs fine FD — must agree
    {
        BottleBelief b(gt, P);
        const Eigen::Vector3f q = {gt.cx + gt.radius, gt.cy, gt.cz};   // on the lateral surface → nonzero gradient
        const auto Jc = b.sdf_jacobian(q, gt, 0);
        BottleBeliefParams Pf = P; Pf.fd_eps = 1e-4f;
        BottleBelief bf(gt, Pf);
        const auto Jf = bf.sdf_jacobian(q, gt, 0);
        check((Jc - Jf).norm() < 1e-2f, "SDF Jacobian (coarse vs fine FD) disagree");
    }

    // (c) Fit recovery from a perturbed init
    BottleBelief belief(BottleBeliefState{0.10f, -0.20f, 0.80f, 0.035f, 0.20f}, P);
    BottleFrame frame; frame.points = all;
    float e = 0.0f;
    for (int it = 0; it < 10; ++it) e = belief.update(frame);
    const auto& s = belief.state();
    std::printf("  recovered: cx=%.3f cy=%.3f cz=%.3f r=%.3f h=%.3f  (energy=%.4f)\n",
                s.cx, s.cy, s.cz, s.radius, s.height, e);
    std::printf("  truth:     cx=%.3f cy=%.3f cz=%.3f r=%.3f h=%.3f\n",
                gt.cx, gt.cy, gt.cz, gt.radius, gt.height);
    check(std::abs(s.cx - gt.cx)         < 0.02f, "cx not recovered");
    check(std::abs(s.cy - gt.cy)         < 0.02f, "cy not recovered");
    check(std::abs(s.radius - gt.radius) < 0.015f, "radius not recovered");
    check(std::abs(s.height - gt.height) < 0.04f, "height not recovered");

    // (d) Clutter responsibility: a far floor point is mostly clutter
    {
        const auto r = belief.responsibilities({gt.cx + 0.6f, gt.cy + 0.6f, gt.cz - 0.2f}, belief.state(),
                                               P.sigma_base_m * P.sigma_base_m);
        std::printf("  clutter-point responsibilities: cylinder=%.2f clutter=%.2f\n", r[0], r[1]);
        check(r[1] > 0.5f, "far floor point should be mostly clutter");
    }

    // (d2) mixture_nll RISES with misfit (the honest F). A drifted cylinder (all points → clutter) must score
    // a HIGHER nll AND a HIGHER clutter_fraction than the recovered fit — the property the divergence sentinel
    // and the convergence gate now rely on (replaces the surface-only energy==0 sentinel).
    {
        const float Rr = P.sigma_base_m * P.sigma_base_m;
        const float nll_good = belief.mixture_nll(all, belief.state(), Rr);
        const float clut_good = belief.clutter_fraction(all, Rr);
        BottleBeliefState drift = belief.state(); drift.cx += 0.8f; drift.cy += 0.8f;   // pull the model off the cloud
        const float nll_bad  = belief.mixture_nll(all, drift, Rr);
        BottleBelief drifted(drift, P);
        const float clut_bad = drifted.clutter_fraction(all, Rr);
        std::printf("  mixture_nll: good=%.2f drifted=%.2f | clutter_frac good=%.2f drifted=%.2f\n",
                    nll_good, nll_bad, clut_good, clut_bad);
        check(nll_bad > nll_good + 1.0f, "mixture_nll does not rise on a drifted (misfit) model");
        check(clut_bad > 0.9f and clut_good < 0.5f, "clutter_fraction does not separate a good fit from a drifted one");
    }

    // (e) Σ finite & SPD
    {
        const Eigen::Matrix<float, 5, 5>& S = belief.covariance();
        check(S.allFinite(), "Σ not finite");
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 5, 5>> es(S);
        check(es.eigenvalues().minCoeff() > 0.0f, "Σ not SPD");
        std::printf("  Σ diag (std, m): cx=%.4f cy=%.4f cz=%.4f r=%.4f h=%.4f\n",
                    std::sqrt(S(0,0)), std::sqrt(S(1,1)), std::sqrt(S(2,2)), std::sqrt(S(3,3)), std::sqrt(S(4,4)));
    }

    // (f) DEPTH DEGENERACY + silhouette: a one-sided (front-arc-only) cloud leaves radius weakly
    //     constrained — moving radius and depth together keeps the SDF ~0 — so the fit INFLATES radius
    //     (the "grows in diameter" symptom). The occluding-contour tangent rays lift exactly that
    //     degeneracy. Verify: with the silhouette, radius is recovered and beats the no-silhouette fit.
    {
        const Eigen::Vector2f cam(gt.cx, gt.cy - 0.8f);        // camera on the −y side
        std::vector<Eigen::Vector3f> arc;
        for (int i = 0; i < 900; ++i)
        {
            const float a = 1.5f * static_cast<float>(M_PI) + 0.40f * U(rng);   // narrow ~±23° front arc (faces cam)
            const float z = gt.cz + 0.5f * gt.height * U(rng);
            arc.push_back({gt.cx + gt.radius * std::cos(a) + noise(rng),
                           gt.cy + gt.radius * std::sin(a) + noise(rng), z + noise(rng)});
        }
        // Two occluding-contour tangent rays from the camera to the true cylinder.
        const float D = (Eigen::Vector2f(gt.cx, gt.cy) - cam).norm();
        const float alpha = std::asin(std::clamp(gt.radius / D, 0.0f, 1.0f));
        const Eigen::Vector2f u = (Eigen::Vector2f(gt.cx, gt.cy) - cam).normalized();
        const auto rot = [](const Eigen::Vector2f& v, float t) {
            return Eigen::Vector2f(v.x() * std::cos(t) - v.y() * std::sin(t), v.x() * std::sin(t) + v.y() * std::cos(t)); };

        const BottleBeliefState init{gt.cx, gt.cy + 0.03f, gt.cz, 2.0f * gt.radius, gt.height};   // inflated radius seed
        auto fit = [&](bool with_sil) {
            BottleBelief b(init, P);
            BottleFrame fr; fr.points = arc;
            if (with_sil) { fr.sil_cam_xy = cam; fr.sil_dirs = {rot(u, alpha), rot(u, -alpha)}; fr.sil_precision = 3000.0f; }
            for (int it = 0; it < 12; ++it) b.update(fr);
            return b.state().radius;
        };
        const float r_no  = fit(false);
        const float r_sil = fit(true);
        std::printf("  front-arc radius: no-silhouette=%.3f  with-silhouette=%.3f  (truth=%.3f)\n", r_no, r_sil, gt.radius);
        check(std::abs(r_sil - gt.radius) < 0.010f, "silhouette should recover radius on a one-sided cloud");
        check(std::abs(r_sil - gt.radius) < std::abs(r_no - gt.radius), "silhouette should beat the no-silhouette radius");
    }

    // (g) LiDAR first-hit range factor lifts the DEPTH degeneracy: the same front-arc cloud that lets the
    //     cylinder slide along the viewing axis (the plain SDF pushes only along the surface NORMAL) is
    //     pinned in RANGE by a handful of LiDAR returns, because the ray residual t*−ρ_obs is measured ALONG
    //     the ray. Camera on −y ⇒ depth is the cy DOF. Verify: with LiDAR, cy is recovered and beats no-LiDAR.
    {
        const Eigen::Vector3f o3(gt.cx, gt.cy - 0.8f, gt.cz);   // sensor on the −y side, looking +y (depth = y)
        std::vector<Eigen::Vector3f> arc, rays;
        for (int i = 0; i < 400; ++i)
        {
            const float a = 1.5f * static_cast<float>(M_PI) + 0.40f * U(rng);   // ~±23° front arc facing sensor
            const float z = gt.cz + 0.5f * gt.height * U(rng);
            const Eigen::Vector3f p(gt.cx + gt.radius * std::cos(a) + noise(rng),
                                    gt.cy + gt.radius * std::sin(a) + noise(rng), z + noise(rng));
            arc.push_back(p);
            if (i % 3 == 0) rays.push_back(p);                 // ~1/3 of the returns also arrive as LiDAR
        }
        const BottleBeliefState init{gt.cx, gt.cy - 0.05f, gt.cz, 1.6f * gt.radius, gt.height};  // parked + inflated
        auto fit = [&](bool with_lidar) {
            BottleBelief b(init, P);
            BottleFrame fr; fr.points = arc;
            if (with_lidar) { fr.lidar.origin = o3; fr.lidar.endpoints = rays; fr.lidar.precision = 2500.0f; }
            for (int it = 0; it < 12; ++it) b.update(fr);
            return b.state().cy;
        };
        const float cy_no  = fit(false);
        const float cy_lid = fit(true);
        std::printf("  front-arc depth(cy): no-lidar=%.3f  with-lidar=%.3f  (truth=%.3f)\n", cy_no, cy_lid, gt.cy);
        check(std::abs(cy_lid - gt.cy) < 0.020f, "lidar should recover depth (cy) on a one-sided cloud");
        check(std::abs(cy_lid - gt.cy) < 0.6f * std::abs(cy_no - gt.cy), "lidar should clearly beat no-lidar depth");
    }

    // (h) EGO-MOTION → SIZE COMMON-MODE (the anti-RESHAPE fixation term). The same full-surface cloud from an
    //     INFLATED cylinder (radius 1.5×) drags radius toward the data when the frame is STILL (chain_cov_size=0);
    //     when the frame carries a large ego-motion size common-mode, the Woodbury cap freezes the size DOF so the
    //     moving frame CONFIRMS but barely reshapes. Verify: |Δradius| still ≫ |Δradius| moving. Position (cx,cy)
    //     is left free here so this isolates the size channel (chain_cov_xx/yy = 0 in both).
    {
        const BottleBeliefState init{gt.cx, gt.cy, gt.cz, 1.5f * gt.radius, gt.height};   // inflated radius seed
        std::vector<Eigen::Vector3f> surf;
        for (int i = 0; i < 1600; ++i)
        {
            const float a = 2.0f * static_cast<float>(M_PI) * U01(rng);
            const float z = gt.cz + 0.5f * gt.height * U(rng);
            surf.push_back({gt.cx + gt.radius * std::cos(a) + noise(rng),
                            gt.cy + gt.radius * std::sin(a) + noise(rng), z + noise(rng)});
        }
        auto dr = [&](float cc_size) {
            BottleBelief b(init, P);
            BottleFrame fr; fr.points = surf; fr.chain_cov_size = cc_size;
            for (int it = 0; it < 6; ++it) b.update(fr);
            return b.state().radius - init.radius;   // signed radius move toward the (smaller) true surface
        };
        const float d_still  = dr(0.0f);
        const float d_moving = dr(1.0f);   // (0.20 gain · ~5 m/s)² ≈ 1 m² — a strongly-moving frame
        std::printf("  ego-motion size cap: Δradius still=%.4f moving=%.4f (truth Δ=%.4f)\n",
                    d_still, d_moving, gt.radius - init.radius);
        check(std::abs(d_moving) < 0.5f * std::abs(d_still),
              "ego-motion size common-mode should freeze radius on a moving frame");
    }

    std::printf("BottleBelief::self_test %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace rc
