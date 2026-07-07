/*
 * residual_belief.cpp  —  AI2 residual (null-concept) belief: an oriented FOOTPRINT BOX fit to a LiDAR
 * cluster's floor projection. The simplest concept belief — a single 2D box primitive + clutter, no
 * silhouette / lidar-ray / accumulate_extra. See residual_belief.h.
 */

#include "residual_belief.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <utility>

namespace rc
{

// ─── SDF primitive (scalar, room frame, 2D footprint — ignores z) ────────────────

namespace
{
// Oriented-box footprint SDF. dx,dy = (room point − box centre) in world XY; the box's local x-axis points
// along `yaw`, with full extents (w,d). Rotate the world offset into the box frame by R(−yaw), then the
// standard axis-aligned box SDF (Inigo Quilez).
float box_sdf_2d(float dx, float dy, float yaw, float w, float d)
{
    const float c = std::cos(yaw), s = std::sin(yaw);
    const float lx =  c * dx + s * dy;   // R(−yaw) · offset
    const float ly = -s * dx + c * dy;
    const float qx = std::abs(lx) - 0.5f * w;
    const float qy = std::abs(ly) - 0.5f * d;
    const float outside = std::sqrt(std::max(qx, 0.0f) * std::max(qx, 0.0f) +
                                    std::max(qy, 0.0f) * std::max(qy, 0.0f));
    const float inside  = std::min(std::max(qx, qy), 0.0f);
    return outside + inside;
}
}  // namespace

float ResidualBelief::sdf_box(const Eigen::Vector3f& p, const ResidualBeliefState& s) const
{
    return box_sdf_2d(p.x() - s.cx, p.y() - s.cy, s.yaw, s.w, s.d);
}

float ResidualBelief::sdf_prim(const Eigen::Vector3f& p, const ResidualBeliefState& s, int /*prim*/) const
{
    return sdf_box(p, s);   // single primitive
}

// ─── Mixture responsibilities ───────────────────────────────────────────────────

std::array<float, 2> ResidualBelief::responsibilities(const Eigen::Vector3f& p, const ResidualBeliefState& s, float R) const
{
    const float eps     = std::clamp(params_.clutter_frac, 0.0f, 0.99f);
    const float pi_surf = 1.0f - eps;
    const float inv2R   = 0.5f / std::max(1e-9f, R);

    std::array<float, 2> u{};
    const float dist = sdf_box(p, s);
    u[0] = pi_surf * std::exp(-dist * dist * inv2R);
    // Clutter = uniform; modelled as a Gaussian floor at clutter_scale so a point further than that from the
    // box edge is explained by clutter rather than dragging the model.
    const float cs = params_.clutter_scale_m;
    u[1] = eps * std::exp(-cs * cs * inv2R);

    const float sum = u[0] + u[1];
    if (sum <= 0.0f) { u = {0.0f, 1.0f}; return u; }
    u[0] /= sum; u[1] /= sum;
    return u;
}

// ─── Jacobian (central finite difference; per-DOF slope clamped) ────────────────
// The box SDF is non-smooth at its corners/edge seams; a point sitting on a seam can spike a raw FD slope.
// Clamp each component to ±gmax so one pathological point can't blow the frame information (recipe rule).
Eigen::Matrix<float, 5, 1> ResidualBelief::sdf_jacobian(const Eigen::Vector3f& p, const ResidualBeliefState& s, int prim) const
{
    constexpr float gmax = 10.0f;
    Eigen::Matrix<float, 5, 1> J;
    const Eigen::Matrix<float, 5, 1> base = s.vec();
    const float e = params_.fd_eps;
    for (int j = 0; j < 5; ++j)
    {
        Eigen::Matrix<float, 5, 1> vp = base, vm = base;
        vp(j) += e; vm(j) -= e;
        const float g = (sdf_prim(p, ResidualBeliefState::from_vec(vp), prim) -
                         sdf_prim(p, ResidualBeliefState::from_vec(vm), prim)) / (2.0f * e);
        J(j) = std::clamp(g, -gmax, gmax);
    }
    return J;
}

// ─── Constraints / canonical form ───────────────────────────────────────────────

void ResidualBelief::apply_constraints(ResidualBeliefState& s) const
{
    s.w = std::clamp(s.w, params_.min_size, params_.max_size);
    s.d = std::clamp(s.d, params_.min_size, params_.max_size);
}

// Box symmetry fold (as table): (w,d,yaw) ≡ (d,w,yaw±90°) and ≡ (w,d,yaw±180°). Canonical representative:
// enforce w ≥ d and wrap yaw into (−π/2, π/2]. Removes the discrete ambiguity so a filtered estimate is
// continuous frame-to-frame (no yaw flips / w↔d swaps under noise).
void ResidualBelief::canonicalize(ResidualBeliefState& s) const
{
    if (s.w < s.d) { std::swap(s.w, s.d); s.yaw += 0.5f * static_cast<float>(M_PI); }
    while (s.yaw >   0.5f * static_cast<float>(M_PI)) s.yaw -= static_cast<float>(M_PI);
    while (s.yaw <= -0.5f * static_cast<float>(M_PI)) s.yaw += static_cast<float>(M_PI);
}

// ─── Engine hooks: prior cov, process noise (F = I, static), common-mode ─────────

Eigen::Matrix<float, 5, 1> ResidualBelief::prior_cov_diag() const
{
    const float pp = params_.prior_pos_std  * params_.prior_pos_std;    // cx, cy
    const float yy = params_.prior_yaw_std  * params_.prior_yaw_std;    // yaw
    const float ss = params_.prior_size_std * params_.prior_size_std;   // w, d
    return (Eigen::Matrix<float, 5, 1>() << pp, pp, yy, ss, ss).finished();
}

Eigen::Matrix<float, 5, 1> ResidualBelief::process_noise_diag() const
{
    const float qp = params_.process_std_pos_m  * params_.process_std_pos_m;    // cx, cy (tight)
    const float qy = params_.process_std_yaw    * params_.process_std_yaw;      // yaw
    const float qs = params_.process_std_size_m * params_.process_std_size_m;   // w, d (GENEROUS — fission)
    return (Eigen::Matrix<float, 5, 1>() << qp, qp, qy, qs, qs).finished();
}

// Inverse of the per-frame common-mode covariance Σ_c (diagonal): position (cx,cy) = config floor +
// pose-chain variance; yaw and size = config stds. Marginalising this SHARED error (via Woodbury in the
// engine) makes the frame's information SATURATE at Σ_c regardless of point count — N correlated points
// cannot collapse σ.
Eigen::Matrix<float, 5, 1> ResidualBelief::common_mode_inv_diag(const ResidualFrame& frame) const
{
    const float p2 = params_.common_mode_pos_std  * params_.common_mode_pos_std;
    const float y2 = params_.common_mode_yaw_std  * params_.common_mode_yaw_std;
    const float s2 = params_.common_mode_size_std * params_.common_mode_size_std;
    return (Eigen::Matrix<float, 5, 1>() <<
            1.0f / std::max(1e-9f, p2 + frame.chain_cov_xx),
            1.0f / std::max(1e-9f, p2 + frame.chain_cov_yy),
            1.0f / y2,
            1.0f / s2, 1.0f / s2).finished();
}

// ─── Self-test ───────────────────────────────────────────────────────────────────

bool ResidualBelief::self_test()
{
    std::mt19937 rng(12345);
    std::normal_distribution<float> noise(0.0f, 0.006f);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f), U01(0.0f, 1.0f);

    const auto ang_diff = [](float a, float b) {
        float d = a - b;
        while (d >  static_cast<float>(M_PI)) d -= 2.0f * static_cast<float>(M_PI);
        while (d < -static_cast<float>(M_PI)) d += 2.0f * static_cast<float>(M_PI);
        return std::abs(d);
    };

    ResidualBeliefParams P;
    const ResidualBeliefState gt{0.50f, -0.30f, 0.50f, 0.90f, 0.40f};   // elongated (w>d), yaw identifiable

    // Sample the box PERIMETER (all four edges, or only the near −d/2 face) at random z (SDF ignores z).
    const auto edge_cloud = [&](const ResidualBeliefState& s, int n, bool near_edge_only) {
        std::vector<Eigen::Vector3f> pts;
        const float c = std::cos(s.yaw), sn = std::sin(s.yaw);
        for (int i = 0; i < n; ++i)
        {
            float lx, ly;
            const int edge = near_edge_only ? 3 : (i % 4);   // 3 = local −d/2 face
            const float t = U(rng) * 0.5f;
            switch (edge)
            {
                case 0: lx =  0.5f * s.w; ly = t * s.d; break;   // right
                case 1: lx = -0.5f * s.w; ly = t * s.d; break;   // left
                case 2: lx = t * s.w; ly =  0.5f * s.d; break;   // top
                default: lx = t * s.w; ly = -0.5f * s.d; break;  // bottom (near face for a −y sensor)
            }
            const float wx = s.cx + (c * lx - sn * ly) + noise(rng);
            const float wy = s.cy + (sn * lx + c * ly) + noise(rng);
            pts.push_back({wx, wy, 0.4f + 0.4f * U01(rng)});
        }
        return pts;
    };

    bool ok = true;
    auto check = [&](bool cond, const char* msg) { if (!cond) { ok = false; std::printf("  FAIL: %s\n", msg); } };

    // (a) SDF correctness at known points
    {
        ResidualBelief b(gt, P);
        check(b.sdf_box({gt.cx, gt.cy, 0.5f}, gt) < -1e-3f, "centre SDF should be negative (inside)");
        const float c = std::cos(gt.yaw), s = std::sin(gt.yaw);
        const Eigen::Vector3f on{gt.cx + c * 0.5f * gt.w, gt.cy + s * 0.5f * gt.w, 0.5f};   // +x local face
        check(std::abs(b.sdf_box(on, gt)) < 1e-2f, "on-face SDF should be ~0");
        check(b.sdf_box({gt.cx + 3.0f, gt.cy, 0.5f}, gt) > 1.0f, "far point SDF should be large positive");
    }

    // (b) Jacobian: coarse FD (used in solver) vs fine FD — must agree
    {
        ResidualBelief b(gt, P);
        const float c = std::cos(gt.yaw), s = std::sin(gt.yaw);
        const Eigen::Vector3f q{gt.cx + c * 0.5f * gt.w + 0.03f, gt.cy + s * 0.5f * gt.w, 0.5f};
        const auto Jc = b.sdf_jacobian(q, gt, 0);
        ResidualBeliefParams Pf = P; Pf.fd_eps = 2e-4f;
        ResidualBelief bf(gt, Pf);
        const auto Jf = bf.sdf_jacobian(q, gt, 0);
        check((Jc - Jf).norm() < 2e-2f, "SDF Jacobian (coarse vs fine FD) disagree");
    }

    // (c) Fit recovery from a perturbed init (full perimeter + clutter)
    std::vector<Eigen::Vector3f> all = edge_cloud(gt, 1200, false);
    for (int i = 0; i < 150; ++i)   // clutter far off to the sides
        all.push_back({gt.cx + U(rng) * 2.0f, gt.cy + U(rng) * 2.0f, 0.4f * U01(rng)});

    ResidualBelief belief(ResidualBeliefState{0.40f, -0.20f, 0.30f, 0.70f, 0.50f}, P);
    ResidualFrame frame; frame.points = all;
    float e = 0.0f;
    for (int it = 0; it < 12; ++it) e = belief.update(frame);
    const auto& s = belief.state();
    std::printf("  recovered: cx=%.3f cy=%.3f yaw=%.3f w=%.3f d=%.3f  (energy=%.4f)\n",
                s.cx, s.cy, s.yaw, s.w, s.d, e);
    std::printf("  truth:     cx=%.3f cy=%.3f yaw=%.3f w=%.3f d=%.3f\n", gt.cx, gt.cy, gt.yaw, gt.w, gt.d);
    check(std::abs(s.cx - gt.cx) < 0.03f, "cx not recovered");
    check(std::abs(s.cy - gt.cy) < 0.03f, "cy not recovered");
    check(ang_diff(s.yaw, gt.yaw) < 0.12f, "yaw not recovered");
    check(std::abs(s.w - gt.w) < 0.06f, "w not recovered");
    check(std::abs(s.d - gt.d) < 0.06f, "d not recovered");

    // (d) Clutter responsibility: a far floor point is mostly clutter
    {
        const auto r = belief.responsibilities({gt.cx + 1.5f, gt.cy + 1.5f, 0.1f}, belief.state(),
                                               P.sigma_base_m * P.sigma_base_m);
        std::printf("  clutter-point responsibilities: box=%.2f clutter=%.2f\n", r[0], r[1]);
        check(r[1] > 0.5f, "far floor point should be mostly clutter");
    }

    // (e) Σ finite & SPD
    {
        const Eigen::Matrix<float, 5, 5>& S = belief.covariance();
        check(S.allFinite(), "Σ not finite");
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 5, 5>> es(S);
        check(es.eigenvalues().minCoeff() > 0.0f, "Σ not SPD");
        std::printf("  Σ diag (std): cx=%.4f cy=%.4f yaw=%.4f w=%.4f d=%.4f\n",
                    std::sqrt(S(0,0)), std::sqrt(S(1,1)), std::sqrt(S(2,2)), std::sqrt(S(3,3)), std::sqrt(S(4,4)));
    }

    // (f) OCCLUSION HONESTY (the residual analog of bottle's silhouette test — but here we ASSERT the
    //     uncertainty rather than fix it). A one-sided view (only the near −y face of an axis-aligned box)
    //     cannot separate the position along the view axis from the unseen depth: sliding cy and growing d
    //     keeps the near edge — and the SDF — unchanged. So the fit MUST leave a WIDE Σ along that axis.
    //     This is the feature: the planner inflates more where the obstacle's depth is a guess. No factor
    //     pins it (unlike bottle); the honest wide Σ is the correct behaviour. We assert on σ_cy (position
    //     along the view axis) — the swap-invariant signal (unlike σ_d, which the w↔d canonical fold can
    //     mix once the degenerate depth grows past the width).
    {
        const ResidualBeliefState aa{0.0f, 0.0f, 0.0f, 0.80f, 0.50f};   // axis-aligned; near face at y=−d/2

        ResidualBelief bfull(ResidualBeliefState{0.05f, 0.05f, 0.0f, 0.6f, 0.6f}, P);
        ResidualFrame ff; ff.points = edge_cloud(aa, 1000, false);
        for (int it = 0; it < 12; ++it) bfull.update(ff);
        const float scy_full = std::sqrt(bfull.covariance()(1, 1));   // σ_cy, full view

        ResidualBelief bone(ResidualBeliefState{0.05f, 0.05f, 0.0f, 0.6f, 0.6f}, P);
        ResidualFrame fo; fo.points = edge_cloud(aa, 1000, true);
        for (int it = 0; it < 12; ++it) bone.update(fo);
        const float scy_one = std::sqrt(bone.covariance()(1, 1));     // σ_cy, one-sided (view axis)

        std::printf("  σ_cy (view axis): full-view=%.4f  one-sided=%.4f  (ratio=%.1f×, should be WIDE)\n",
                    scy_full, scy_one, scy_one / std::max(1e-6f, scy_full));
        // Principled assertion = the RATIO (occlusion must widen the view-axis Σ). The absolute magnitude is
        // deliberately capped by the common-mode saturation, so we don't pin a magic number — only the
        // direction/size of the effect (a full view constrains the depth axis; a one-sided view must not).
        check(scy_one > 0.04f, "one-sided view should leave the view-axis Σ wide (honest occlusion)");
        check(scy_one > 2.5f * scy_full, "one-sided view-axis Σ should be much wider than a full view");
    }

    std::printf("ResidualBelief::self_test %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace rc
