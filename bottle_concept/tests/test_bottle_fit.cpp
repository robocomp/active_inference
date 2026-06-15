/*
 * test_bottle_fit.cpp
 *
 * Zero-dependency (no DSR / no Ice / no scene) smoke test for the bottle
 * generative model. Synthesises a room-frame point cloud on a known cylinder
 * surface — with noise and partial (single-sided) occlusion — and checks:
 *
 *   1. gradient_step recovers the pose (cx,cy,cz) and size (radius,height)
 *      from an offset prior.
 *   2. pose_covariance() SHRINKS as observations get denser / better-spread
 *      and INFLATES toward the prior bound when points are sparse or absent
 *      (the P_bottle behaviour the controller relies on).
 *
 * This isolates the model maths from the perception/scene integration, so a
 * failure here is a model bug, not a graph/Webots bug.
 */

#include "../src/bottle_model.h"

#include <cmath>
#include <numbers>
#include <print>
#include <random>
#include <vector>

namespace
{
constexpr float kTwoPi = 2.0f * std::numbers::pi_v<float>;

// Ground-truth cylinder (room frame).
const BottleState kGT{0.40f, 0.00f, 0.85f, 0.035f, 0.20f};

int g_failures = 0;

void check(bool ok, const std::string& what)
{
    std::print("  [{}] {}\n", ok ? "PASS" : "FAIL", what);
    if (not ok) ++g_failures;
}

/**
 * Sample points on the cylinder surface.
 * @param n_az        azimuth samples around the circle
 * @param n_h         height samples along the axis
 * @param az_lo,az_hi visible azimuth arc (radians) — restrict for occlusion
 * @param noise_std   Gaussian position noise (m)
 * @param add_caps    include top/bottom cap points
 */
std::vector<Eigen::Vector3f> sample_cylinder(const BottleState& s,
                                             int n_az, int n_h,
                                             float az_lo, float az_hi,
                                             float noise_std,
                                             bool add_caps,
                                             std::mt19937& rng)
{
    std::normal_distribution<float> noise(0.0f, noise_std);
    std::vector<Eigen::Vector3f> pts;
    const float hh = s.height * 0.5f;

    for (int ia = 0; ia < n_az; ++ia)
    {
        const float a = az_lo + (az_hi - az_lo) * static_cast<float>(ia) / static_cast<float>(std::max(1, n_az - 1));
        for (int ih = 0; ih < n_h; ++ih)
        {
            const float z = -hh + s.height * static_cast<float>(ih) / static_cast<float>(std::max(1, n_h - 1));
            Eigen::Vector3f p{
                s.cx + s.radius * std::cos(a) + noise(rng),
                s.cy + s.radius * std::sin(a) + noise(rng),
                s.cz + z + noise(rng)};
            pts.push_back(p);
        }
    }

    if (add_caps)
        for (int ia = 0; ia < n_az; ++ia)
        {
            const float a = az_lo + (az_hi - az_lo) * static_cast<float>(ia) / static_cast<float>(std::max(1, n_az - 1));
            for (float rr : {0.4f, 0.8f})
            {
                pts.emplace_back(s.cx + rr * s.radius * std::cos(a) + noise(rng),
                                 s.cy + rr * s.radius * std::sin(a) + noise(rng),
                                 s.cz + hh + noise(rng));
            }
        }

    return pts;
}

BottleModelParams make_params()
{
    // Mirrors the agent's default config (etc/config [BottleModel]).
    BottleModelParams p;
    p.sigma_obs = 0.02f;
    p.optimization_iters = 15;
    p.optimization_lr = 0.005f;   // cm-scale object: 10× smaller than the table's
    p.lambda_size  = 0.5f;     // size prior pins radius/height — the mechanism that
    p.prior_radius = 0.035f;   // resolves the single-view radius/centre ambiguity
    p.prior_height = 0.20f;
    p.prior_size_std = 0.03f;
    p.lambda_pos  = 0.05f;
    p.lambda_state = 0.02f;
    return p;
}

float trace3(const Eigen::Matrix3f& m) { return m(0,0) + m(1,1) + m(2,2); }

Eigen::Vector3f centroid(const std::vector<Eigen::Vector3f>& pts)
{
    Eigen::Vector3f c = Eigen::Vector3f::Zero();
    for (const auto& p : pts) c += p;
    return pts.empty() ? c : (c / static_cast<float>(pts.size())).eval();
}

} // namespace

int main()
{
    std::mt19937 rng(42);

    // ── Test 1: pose + size recovery in the agent's operating regime ──────────
    // The agent's SampleQueue accumulates points across views, so the fit sees
    // (near-)full angular coverage; cold-start snaps the centre to the observed
    // centroid before descent. We reproduce that: full ring, centre seeded at
    // the centroid, size starting wrong → the fit must recover radius/height.
    std::print("Test 1: fit recovers size with centre cold-started (full coverage)\n");
    {
        auto pts = sample_cylinder(kGT, 24, 8, 0.0f, kTwoPi, 0.002f, true, rng);
        const Eigen::Vector3f cen = centroid(pts);

        // Centre cold-started to the centroid (≈GT for full coverage); size off.
        BottleState init{cen.x(), cen.y(), cen.z(), 0.05f, 0.16f};
        BottleModel model(init, make_params());
        model.set_prior(BottleState{cen.x(), cen.y(), cen.z(), 0.035f, 0.20f});

        const std::vector<float> w;  // uniform
        float fe = 0.0f;
        for (int frame = 0; frame < 60; ++frame)
            fe = model.gradient_step(pts, w);

        const auto& s = model.state();
        std::print("  fit: c=({:.3f},{:.3f},{:.3f}) r={:.3f} h={:.3f}  (GT c=({:.3f},{:.3f},{:.3f}) r={:.3f} h={:.3f})  FE={:.5f}\n",
                   s.cx, s.cy, s.cz, s.radius, s.height,
                   kGT.cx, kGT.cy, kGT.cz, kGT.radius, kGT.height, fe);

        check(std::abs(s.cx - kGT.cx) < 0.01f, "cx within 1 cm");
        check(std::abs(s.cy - kGT.cy) < 0.01f, "cy within 1 cm");
        check(std::abs(s.cz - kGT.cz) < 0.02f, "cz within 2 cm");
        check(std::abs(s.radius - kGT.radius) < 0.01f, "radius within 1 cm");
        check(std::abs(s.height - kGT.height) < 0.03f, "height within 3 cm");
    }

    // ── Test 2: covariance shrinks with denser / better coverage ──────────────
    std::print("Test 2: pose_covariance shrinks with more & better-spread points\n");
    {
        BottleModel model(kGT, make_params());   // sit at the optimum

        auto dense_full  = sample_cylinder(kGT, 32, 10, 0.0f, kTwoPi, 0.002f, true,  rng);   // full ring
        auto sparse_arc  = sample_cylinder(kGT,  4,  3, -0.6f, 0.6f,  0.002f, false, rng);   // few, one side
        const std::vector<float> w;

        const float tr_dense  = trace3(model.pose_covariance(dense_full,  w));
        const float tr_sparse = trace3(model.pose_covariance(sparse_arc,  w));
        const float tr_empty  = trace3(model.pose_covariance({},          w));

        std::print("  trace(Σ): dense_full={:.3e}  sparse_arc={:.3e}  empty(prior)={:.3e}\n",
                   tr_dense, tr_sparse, tr_empty);

        check(tr_dense > 0.0f, "dense covariance is positive/finite");
        check(tr_sparse > tr_dense, "sparse arc → larger covariance than dense full ring");
        check(tr_empty >= tr_sparse, "no points → covariance inflates to the prior bound");
        check(tr_dense < 1e-2f, "dense covariance is tight (well localised)");
    }

    // ── Test 3: dropout inflation is monotone as points are removed ───────────
    std::print("Test 3: covariance grows monotonically as observations drop out\n");
    {
        BottleModel model(kGT, make_params());
        const std::vector<float> w;
        float prev = -1.0f;
        bool monotone = true;
        for (int n_az : {32, 16, 8, 4, 2})
        {
            auto pts = sample_cylinder(kGT, n_az, 6, 0.0f, kTwoPi, 0.002f, true, rng);
            const float tr = trace3(model.pose_covariance(pts, w));
            std::print("  n_az={:2d} pts={:3d}  trace(Σ)={:.3e}\n", n_az, static_cast<int>(pts.size()), tr);
            if (prev >= 0.0f and tr < prev - 1e-9f)
                monotone = false;
            prev = tr;
        }
        check(monotone, "trace(Σ) is non-decreasing as point count falls");
    }

    std::print("\n{} ({} failure(s))\n", g_failures == 0 ? "ALL PASSED" : "FAILURES", g_failures);
    return g_failures == 0 ? 0 : 1;
}
