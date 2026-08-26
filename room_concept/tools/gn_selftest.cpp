/*
 *  gn_selftest.cpp — offline validation of the Gauss-Newton/LM backend (room_gn_solver).
 *
 *  Answers the two questions the shadow log cannot answer until the agent runs, and that no amount
 *  of reading the code settles:
 *    1. Are the analytic Jacobians right? (central finite differences of the SAME loss, per factor
 *       family in isolation and then all together)
 *    2. Does the solver recover a known pose from a perturbed start, and in how many iterations?
 *
 *  Synthetic room + synthetic scan, so the ground truth is known exactly. Build:
 *      make -C build gn_selftest && ./bin/gn_selftest
 */
#include <cmath>
#include <cstdio>
#include <deque>
#include <chrono>
#include <random>
#include <vector>

#include "room_concept.h"
#include "room_gn_solver.h"
#include "room_model.h"
#include "image_edge_types.h"
#include "image_edge_ops.h"
#include "image_edge_accumulate.h"
#include "image_edge_source.h"

using rc::RoomConcept;

namespace
{
    constexpr float kHalfW = 3.0f;    // room is 6 x 4 m, centred on the origin
    constexpr float kHalfL = 2.0f;

    std::vector<Eigen::Vector2f> room_polygon()
    {
        return {{-kHalfW, -kHalfL}, {kHalfW, -kHalfL}, {kHalfW, kHalfL}, {-kHalfW, kHalfL}};
    }

    /// Ray-cast a rectangle from `pose` and return the hits in the ROBOT frame, as the lidar would.
    torch::Tensor synth_scan(const Eigen::Vector3f& pose, int n, float noise_sigma, std::mt19937& rng)
    {
        std::normal_distribution<float> noise(0.f, noise_sigma);
        auto t = torch::zeros({n, 3}, torch::kFloat32);
        auto a = t.accessor<float, 2>();
        for (int i = 0; i < n; ++i)
        {
            const float bearing = -static_cast<float>(M_PI) + 2.f * static_cast<float>(M_PI) * i / n;
            const float world_dir = pose.z() + bearing;
            const float dx = std::cos(world_dir), dy = std::sin(world_dir);

            // Distance to the rectangle along the ray (slab method, from a point inside).
            float best = std::numeric_limits<float>::max();
            if (std::abs(dx) > 1e-6f)
            {
                for (const float wall : {-kHalfW, kHalfW})
                {
                    const float s = (wall - pose.x()) / dx;
                    if (s > 1e-4f and std::abs(pose.y() + s * dy) <= kHalfL + 1e-3f) best = std::min(best, s);
                }
            }
            if (std::abs(dy) > 1e-6f)
            {
                for (const float wall : {-kHalfL, kHalfL})
                {
                    const float s = (wall - pose.y()) / dy;
                    if (s > 1e-4f and std::abs(pose.x() + s * dx) <= kHalfW + 1e-3f) best = std::min(best, s);
                }
            }
            const float r = best + noise(rng);
            a[i][0] = r * std::cos(bearing);   // robot frame
            a[i][1] = r * std::sin(bearing);
            a[i][2] = 0.f;
        }
        return t;
    }

    torch::Tensor pose_tensor(const Eigen::Vector3f& p)
    {
        return torch::tensor({p.x(), p.y(), p.z()},
                             torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
    }

    torch::Tensor mat3(const Eigen::Matrix3f& m)
    {
        auto t = torch::zeros({3, 3}, torch::kFloat32);
        auto a = t.accessor<float, 2>();
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) a[r][c] = m(r, c);
        return t;
    }

    int failures = 0;
    void check(const char* what, bool ok, const std::string& detail)
    {
        std::printf("  %-46s %s   %s\n", what, ok ? "PASS" : "FAIL", detail.c_str());
        if (not ok) ++failures;
    }
} // namespace

int main()
{
    torch::set_num_threads(1);
    std::mt19937 rng(12345);

    rc::Model model;
    model.init_from_polygon(room_polygon(), 0.f, 0.f, 0.f, 2.4f);

    RoomConcept::Params params;
    params.rfe_obs_sigma        = 0.05f;
    params.rfe_huber_delta      = 0.15f;
    params.sdf_current_slot_only = true;
    params.enable_corner_tracking = true;
    params.corner_precision_gain = 1.0f;
    params.corner_huber_sigma    = 3.0f;
    params.corner_max_slots      = 5;
    params.far_points_weight     = true;      // exercise the shared weighting
    params.incidence_angle_weight = true;
    params.object_anchor.enable  = true;
    params.object_anchor.weight  = 1.0f;
    params.object_anchor.huber_delta = 3.0f;
    params.object_anchor_max_slots = 3;

    // ---- A three-slot window along a short trajectory -----------------------------------------
    const std::vector<Eigen::Vector3f> truth = {
        {0.20f, -0.50f, 0.30f}, {0.35f, -0.42f, 0.38f}, {0.52f, -0.31f, 0.44f}};

    std::deque<RoomConcept::WindowSlot> window;
    for (size_t i = 0; i < truth.size(); ++i)
    {
        RoomConcept::WindowSlot slot;
        slot.pose = pose_tensor(truth[i]);
        slot.lidar_points = synth_scan(truth[i], 400, 0.01f, rng);
        slot.odometry_delta = (i == 0) ? Eigen::Vector3f::Zero()
                                       : Eigen::Vector3f(truth[i] - truth[i - 1]);
        slot.motion_cov = Eigen::Vector3f(4e-4f, 4e-4f, 1e-4f).asDiagonal();
        slot.odom_delta_tensor = torch::tensor(
            {slot.odometry_delta.x(), slot.odometry_delta.y(), slot.odometry_delta.z()}, torch::kFloat32);
        slot.motion_prec_tensor = mat3(slot.motion_cov.inverse());

        // One corner landmark (yaw-free) and one object anchor (with yaw), both consistent with truth.
        const Eigen::Vector2f corner_world(kHalfW, kHalfL);
        const float c = std::cos(truth[i].z()), s = std::sin(truth[i].z());
        const Eigen::Vector2f dw = corner_world - truth[i].head<2>();
        RoomConcept::WindowSlot::CornerObs co;
        co.model_corner_world = corner_world;
        co.detected_robot = {c * dw.x() + s * dw.y(), -s * dw.x() + c * dw.y()};
        co.information = Eigen::Matrix2f::Identity() * 25.f;
        slot.corner_obs.push_back(co);
        slot.rebuild_corner_batch(torch::kCPU);

        const Eigen::Vector3f obj_world(-1.2f, 0.8f, 0.9f);
        const Eigen::Vector2f dwo = obj_world.head<2>() - truth[i].head<2>();
        rc::ObjectAnchorObs oa;
        oa.pose_world = obj_world;
        oa.obs_robot = { c * dwo.x() + s * dwo.y(),
                        -s * dwo.x() + c * dwo.y(),
                         obj_world.z() - truth[i].z()};
        oa.information = Eigen::Vector3f(16.f, 16.f, 4.f).asDiagonal();
        oa.has_orientation = true;
        slot.object_anchors.push_back(oa);

        window.push_back(std::move(slot));
    }

    RoomConcept::BoundaryPrior bp;
    bp.valid = true;
    bp.mu = truth[0];
    bp.precision = Eigen::Vector3f(100.f, 100.f, 400.f).asDiagonal();

    rc::gn::Input in;
    in.model = &model;
    in.params = &params;
    in.window = &window;
    in.boundary_prior = &bp;
    in.boundary_weight = 1.0f;
    in.device = torch::kCPU;

    // ---- 1. Jacobian: each factor family alone, then all together -----------------------------
    std::printf("\nJacobian vs central differences (max relative error over all 9 state components)\n");
    std::vector<Eigen::Vector3f> at = truth;
    at[2] += Eigen::Vector3f(0.03f, -0.02f, 0.015f);   // off the optimum, where the gradient is informative

    // `corrupt` blows the landmark residuals up past the Huber knee. Without it every landmark sits
    // in the quadratic branch and the SATURATED branch of the robust weight is never tested — which is
    // exactly how a 2x error there survived the first version of this file and had to be found in a
    // live shadow log instead.
    struct Case { const char* name; bool sdf, motion, corner, object, boundary; float corrupt; };
    for (const Case& cs : std::vector<Case>{
            {"SDF only",                    true,  false, false, false, false, 0.f},
            {"motion only",                 false, true,  false, false, false, 0.f},
            {"corner only",                 false, false, true,  false, false, 0.f},
            {"corner only, SATURATED",      false, false, true,  false, false, 1.5f},
            {"object anchor only",          false, false, false, true,  false, 0.f},
            {"object anchor only, SATURATED", false, false, false, true, false, 1.5f},
            {"boundary only",               false, false, false, false, true , 0.f},
            {"all factors",                 true,  true,  true,  true,  true , 0.f},
            {"all factors, SATURATED",      true,  true,  true,  true,  true , 1.5f}})
    {
        RoomConcept::Params p = params;
        p.enable_corner_tracking = cs.corner;
        p.object_anchor.enable   = cs.object;
        // Disabling SDF/motion is done by emptying what the factor reads.
        std::deque<RoomConcept::WindowSlot> w = window;
        if (not cs.sdf)
            for (auto& s : w) s.lidar_points = torch::Tensor{};
        if (not cs.motion)
            for (auto& s : w) s.motion_prec_tensor = mat3(Eigen::Matrix3f::Zero());
        if (cs.corrupt > 0.f)
            for (auto& s : w)
            {
                for (auto& co : s.corner_obs) co.detected_robot.x() += cs.corrupt;
                s.rebuild_corner_batch(torch::kCPU);
                for (auto& oa : s.object_anchors) oa.obs_robot.x() += cs.corrupt;
            }
        RoomConcept::BoundaryPrior b = bp;
        b.valid = cs.boundary;

        rc::gn::Input i2 = in;
        i2.params = &p;
        i2.window = &w;
        i2.boundary_prior = &b;
        const float err = rc::gn::gradient_check(i2, at);
        char buf[128];
        std::snprintf(buf, sizeof buf, "rel err = %.3e", err);
        check(cs.name, std::isfinite(err) and err < 1e-2f, buf);
    }

    // ---- 2. Convergence from a perturbed start ------------------------------------------------
    // ---- 1b. The check itself degenerates AT the optimum -------------------------------------
    // Relative error needs a non-degenerate reference. At a converged pose the true gradient is ~0,
    // so both the analytic b and its finite difference are float noise and their ratio is arbitrary.
    // This is not a solver property, it is a property of the TEST, and it is why the shadow log's
    // grad_relerr must be sampled slightly OFF the optimum rather than at it.
    std::printf("\nWhere the gradient check is valid (same Jacobians, three sample points)\n");
    for (const auto& [name, off] : std::vector<std::pair<const char*, Eigen::Vector3f>>{
            {"at the optimum (DEGENERATE by construction)", {0.f, 0.f, 0.f}},
            {"optimum + 5 mm / 2 mrad (smooth basin)",      {0.005f, 0.005f, 0.002f}},
            {"optimum + 15 cm / 60 mrad (far)",             {0.15f, 0.15f, 0.06f}}})
    {
        std::vector<Eigen::Vector3f> p = truth;
        for (auto& q : p) q += off;
        const float err = rc::gn::gradient_check(in, p);
        char buf[128];
        std::snprintf(buf, sizeof buf, "rel err = %.3e", err);
        // Only the middle sample is expected to be a valid verdict on the Jacobian.
        const bool expect_clean = off.norm() > 1e-6f and off.norm() < 0.1f;
        check(name, (not expect_clean) or err < 1e-2f, buf);
    }

    std::printf("\nRecovery from a perturbed start (LM, max 10 iterations)\n");
    for (const auto& pert : std::vector<Eigen::Vector3f>{
            {0.02f, 0.02f, 0.01f}, {0.10f, -0.08f, 0.05f}, {0.25f, 0.20f, 0.12f}})
    {
        std::vector<Eigen::Vector3f> poses = truth;
        for (auto& p : poses) p += pert;

        rc::gn::Options opts;
        const float loss0 = rc::gn::evaluate(in, poses);
        const auto t0 = std::chrono::high_resolution_clock::now();
        const auto r = rc::gn::solve(in, poses, opts);
        const float solve_ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();

        float worst_xy = 0.f, worst_th = 0.f;
        for (size_t i = 0; i < poses.size(); ++i)
        {
            worst_xy = std::max(worst_xy, (poses[i].head<2>() - truth[i].head<2>()).norm());
            worst_th = std::max(worst_th, std::abs(poses[i].z() - truth[i].z()));
        }
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "start %.2fm/%.3frad -> %.1fmm/%.4frad, loss %.4f->%.4f, %d it (%d rej), %.2f ms",
                      pert.head<2>().norm(), pert.z(), worst_xy * 1000.f, worst_th,
                      loss0, r.loss, r.iterations, r.rejected, solve_ms);
        check("converged and improved", r.ok and r.loss < loss0 and worst_xy < 0.05f, buf);
    }

    // =================================================================================================
    //  SE(2) PREINTEGRATION (se2_preintegration.h)
    //
    //  The covariance is the whole point of that file, and a covariance is the one quantity a
    //  convergence test cannot check: a wrong Σ still converges, it just converges to a moved minimum
    //  (the same failure mode the SDF 2x scaling had, which is why this harness exists at all). So the
    //  reference here is MONTE CARLO — draw the per-step noise the model claims, integrate the nominal
    //  arithmetic with it, and compare the empirical spread of Δ against the propagated Σ. That
    //  validates A, B, Q and the midpoint 0.5 factor in one shot, against no analytic re-derivation.
    // =================================================================================================
    std::printf("\nSE(2) preintegration\n");
    {
        struct Step { float v_lat, v_long, omega, dt; };

        // A turning, translating interval: the regime where the cross terms exist at all. A straight
        // leg would pass with A = I and prove nothing.
        const auto make_traj = [](float dt, float total_s)
        {
            std::vector<Step> traj;
            const int n = static_cast<int>(std::lround(total_s / dt));
            for (int i = 0; i < n; ++i)
                traj.push_back({0.05f, 0.45f, 0.8f, dt});
            return traj;
        };

        // The nominal integration, with optional per-step perturbations. Mirrors Integrator::add()'s
        // mean lines exactly — deliberately re-written rather than reused, so a bug in the recursion
        // cannot hide by being present on both sides of the comparison.
        const auto integrate = [](const std::vector<Step>& traj, float theta0,
                                  float scale_v, float scale_omega,
                                  const std::vector<Eigen::Vector3f>* pert)
        {
            Eigen::Vector3f d = Eigen::Vector3f::Zero();
            float th = theta0;
            for (size_t i = 0; i < traj.size(); ++i)
            {
                const auto& s = traj[i];
                float dx_local = s.v_lat  * (1.f + scale_v) * s.dt;
                float dy_local = s.v_long * (1.f + scale_v) * s.dt;
                float dtheta   = s.omega  * (1.f + scale_omega) * s.dt;
                if (pert != nullptr)
                {
                    dx_local += (*pert)[i][0];   // additive on the INCREMENT, which is what Q·dt models
                    dy_local += (*pert)[i][1];
                    dtheta   += (*pert)[i][2];
                }
                const float tm = th + 0.5f * dtheta;
                d[0] += dx_local * std::cos(tm) - dy_local * std::sin(tm);
                d[1] += dx_local * std::sin(tm) + dy_local * std::cos(tm);
                d[2] += dtheta;
                th += dtheta;
            }
            return d;
        };

        const float theta0 = 0.6f;
        const auto traj = make_traj(0.01f, 0.5f);      // 50 samples at 100 Hz over half a second

        // ---- 1. Σ against Monte Carlo -------------------------------------------------------------
        rc::preint::NoiseModel qn;                     // random part only, so covariance() == cov
        qn.sigma_v_lat = qn.sigma_v_long = 0.02f;
        qn.sigma_omega = 0.05f;
        qn.scale_v = qn.scale_omega = 0.f;
        // The Monte Carlo below draws its perturbations as sigma·√dt — the FREE-DRIFT model — so these
        // legs must be run against that model, not against the ZUPT-conditioned one. Leaving the
        // default on would have quietly reduced iv.cov by ~8% at this trajectory's speed and turned a
        // sharp agreement test into a loose one. The ZUPT gets its own section further down.
        qn.zupt_enabled = false;

        rc::preint::Integrator ig(theta0);
        ig.set_noise(qn);
        for (const auto& s : traj) ig.add(s.v_lat, s.v_long, s.omega, s.dt);
        const auto iv = ig.result();

        const Eigen::Vector3f mean_nominal = integrate(traj, theta0, 0.f, 0.f, nullptr);
        check("mean matches the legacy integration",
              (iv.delta - mean_nominal).norm() < 1e-5f,
              "d = [" + std::to_string(iv.delta[0]) + ", " + std::to_string(iv.delta[1]) + ", "
                      + std::to_string(iv.delta[2]) + "]");

        constexpr int kTrials = 40000;
        Eigen::Matrix3f mc = Eigen::Matrix3f::Zero();
        Eigen::Vector3f mc_mean = Eigen::Vector3f::Zero();
        std::vector<Eigen::Vector3f> samples;
        samples.reserve(kTrials);
        {
            std::mt19937 mrng(999);
            std::normal_distribution<float> g(0.f, 1.f);
            std::vector<Eigen::Vector3f> pert(traj.size());
            for (int t = 0; t < kTrials; ++t)
            {
                for (size_t i = 0; i < traj.size(); ++i)
                {
                    const float sd = std::sqrt(traj[i].dt);
                    pert[i] = Eigen::Vector3f(qn.sigma_v_lat  * sd * g(mrng),
                                              qn.sigma_v_long * sd * g(mrng),
                                              qn.sigma_omega  * sd * g(mrng));
                }
                const Eigen::Vector3f d = integrate(traj, theta0, 0.f, 0.f, &pert);
                samples.push_back(d);
                mc_mean += d;
            }
            mc_mean /= static_cast<float>(kTrials);
            for (const auto& d : samples)
            {
                const Eigen::Vector3f e = d - mc_mean;
                mc += e * e.transpose();
            }
            mc /= static_cast<float>(kTrials - 1);
        }
        const float cov_rel = (mc - iv.cov).norm() / std::max(iv.cov.norm(), 1e-12f);
        {
            char buf[256];
            std::snprintf(buf, sizeof buf,
                          "rel Frobenius %.4f  (sig_x %.4f vs %.4f, sig_th %.4f vs %.4f, rho_yth %+.3f vs %+.3f)",
                          cov_rel, std::sqrt(iv.cov(0, 0)), std::sqrt(mc(0, 0)),
                          std::sqrt(iv.cov(2, 2)), std::sqrt(mc(2, 2)),
                          iv.cov(1, 2) / std::sqrt(iv.cov(1, 1) * iv.cov(2, 2)),
                          mc(1, 2) / std::sqrt(mc(1, 1) * mc(2, 2)));
            check("covariance recursion vs 40k Monte Carlo", cov_rel < 0.10f, buf);
        }
        // The cross terms are the reason this file exists: assert they are actually PRESENT, or the
        // test above could be passed by a diagonal that happens to have the right diagonal.
        {
            const float rho = std::abs(iv.cov(1, 2)) / std::sqrt(iv.cov(1, 1) * iv.cov(2, 2));
            char buf[128];
            std::snprintf(buf, sizeof buf, "|rho(y,theta)| = %.3f — a diagonal model asserts 0", rho);
            check("covariance is NOT diagonal", rho > 0.05f, buf);
        }

        // ---- 2. Scale Jacobians against central differences ---------------------------------------
        {
            constexpr float eps = 1e-3f;
            const Eigen::Vector3f fd_omega =
                (integrate(traj, theta0, 0.f, eps, nullptr) - integrate(traj, theta0, 0.f, -eps, nullptr))
                / (2.f * eps);
            const Eigen::Vector3f fd_v =
                (integrate(traj, theta0, eps, 0.f, nullptr) - integrate(traj, theta0, -eps, 0.f, nullptr))
                / (2.f * eps);
            const float e_omega = (fd_omega - iv.g_omega).norm() / std::max(fd_omega.norm(), 1e-9f);
            const float e_v     = (fd_v     - iv.g_v).norm()     / std::max(fd_v.norm(),     1e-9f);
            char buf[192];
            std::snprintf(buf, sizeof buf, "rel err omega %.2e, v %.2e", e_omega, e_v);
            check("scale Jacobians vs central differences", e_omega < 1e-3f and e_v < 1e-3f, buf);
        }

        // ---- 3. chain() must equal one continuous integration -------------------------------------
        // This is the property the legacy `stride_cov_accum_ += cov` does NOT have.
        {
            const size_t half = traj.size() / 2;
            std::vector<Step> a(traj.begin(), traj.begin() + static_cast<long>(half));
            std::vector<Step> b(traj.begin() + static_cast<long>(half), traj.end());

            rc::preint::Integrator ia(theta0);
            ia.set_noise(qn);
            for (const auto& s : a) ia.add(s.v_lat, s.v_long, s.omega, s.dt);

            rc::preint::Integrator ib(ia.heading());     // b starts where a ends
            ib.set_noise(qn);
            for (const auto& s : b) ib.add(s.v_lat, s.v_long, s.omega, s.dt);

            const auto chained = rc::preint::chain(ia.result(), ib.result());
            const float d_err = (chained.delta - iv.delta).norm();
            const float c_err = (chained.cov - iv.cov).norm() / std::max(iv.cov.norm(), 1e-12f);
            const float g_err = (chained.g_omega - iv.g_omega).norm() / std::max(iv.g_omega.norm(), 1e-9f);

            // What the legacy additive form would have said, for scale.
            const Eigen::Matrix3f naive = ia.result().cov + ib.result().cov;
            const float naive_err = (naive - iv.cov).norm() / std::max(iv.cov.norm(), 1e-12f);

            char buf[224];
            std::snprintf(buf, sizeof buf,
                          "delta %.2e, cov %.2e, g %.2e  (naive sum: cov %.2e)",
                          d_err, c_err, g_err, naive_err);
            check("chain() == one continuous integration",
                  d_err < 1e-5f and c_err < 1e-4f and g_err < 1e-4f, buf);
        }

        // ---- 4. Rate invariance -------------------------------------------------------------------
        // The same physical interval sampled at 20 Hz and at 100 Hz must give the same covariance.
        // The legacy form cannot: its constants are added once per FRAME, so five times the frame rate
        // is five times the asserted noise. This is the 08-09 units defect as a property test.
        {
            const auto make_iv = [&](float dt)
            {
                const auto tr = make_traj(dt, 0.5f);
                rc::preint::Integrator g(theta0);
                g.set_noise(qn);
                for (const auto& s : tr) g.add(s.v_lat, s.v_long, s.omega, s.dt);
                return g.result();
            };
            const auto slow = make_iv(0.05f);      // 20 Hz
            const auto fast = make_iv(0.01f);      // 100 Hz
            const float rel = (slow.cov - fast.cov).norm() / std::max(fast.cov.norm(), 1e-12f);
            const float gr  = (slow.g_omega - fast.g_omega).norm() / std::max(fast.g_omega.norm(), 1e-9f);
            char buf[192];
            std::snprintf(buf, sizeof buf,
                          "sigma_th 20Hz %.5f vs 100Hz %.5f, cov rel %.4f, g rel %.4f",
                          std::sqrt(slow.cov(2, 2)), std::sqrt(fast.cov(2, 2)), rel, gr);
            check("covariance is invariant to sample rate", rel < 0.02f and gr < 0.02f, buf);
        }

        // ---- 5. ZUPT: the rest hypothesis as a factor ---------------------------------------------
        // Measured 2026-08-18: parked, the free-drift model let the motion prior's sigma reach 82 cm
        // over a 294 s stationary stretch, so it stopped constraining anything. These legs pin the
        // three properties that make the fix a FACTOR rather than a clamp: it bites at rest, it fades
        // out on its own in motion with no threshold, and it does not mistake a pivot for rest.
        {
            const auto run = [&](float v_lat, float v_long, float omega, float dt, float total_s,
                                 bool zupt)
            {
                rc::preint::NoiseModel q = qn;
                q.zupt_enabled = zupt;
                rc::preint::Integrator g(theta0);
                g.set_noise(q);
                const int n = static_cast<int>(std::lround(total_s / dt));
                for (int i = 0; i < n; ++i) g.add(v_lat, v_long, omega, dt);
                return g.result();
            };
            char buf[224];

            // (a) Parked for five minutes: bounded, and orders below free drift.
            const auto park_free = run(0.f, 0.f, 0.f, 0.008f, 300.f, false);
            const auto park_zupt = run(0.f, 0.f, 0.f, 0.008f, 300.f, true);
            const float sx_free = std::sqrt(park_free.cov(0, 0));
            const float sx_zupt = std::sqrt(park_zupt.cov(0, 0));
            const float st_zupt = std::sqrt(park_zupt.cov(2, 2));
            // ★Assert the MECHANISM, not a magic size. Parked, every sample's velocity variance
            // collapses to the ZUPT's own, so the interval has a closed form. With the rest hypothesis
            // carried as a DENSITY (2026-08-26) that closed form is sigma = density·√T — note there is
            // no dt in it, which IS the property being asserted. The old per-sample form gave
            // sigma_rest·√(dt·T), and that stray √dt was the bug: publishing faster tightened the
            // parked prior with no physical change. Checking the closed form pins the algebra and
            // stays correct when the density is re-measured on another platform — an absolute bound
            // would just re-encode today's number, which is exactly how the first version of this test
            // came to enshrine a sigma_rest that was 10x too tight.
            // ★ The EXACT closed form, not the R << P approximation. Parked, both variances are
            // densities over dt, so P = a/dt and R = b/dt and the scalar update returns their
            // HARMONIC combination — sigma = sqrt(a·b·T/(a+b)) — with no dt in it. The previous
            // version asserted sigma_rest·sqrt(dt·T), which is that expression's limit for b << a,
            // and it held only while the rest constant was 5.9x too tight for this robot. With the
            // measured density it is no longer a floor far below the model, it is COMPARABLE to it,
            // and the approximation is off by 20%. Using the exact form keeps the test honest under
            // any future re-measurement instead of encoding one platform's regime.
            const float a = qn.sigma_v_lat * qn.sigma_v_lat;
            const float b = rc::preint::NoiseModel{}.zupt_density_v
                          * rc::preint::NoiseModel{}.zupt_density_v;
            const float predicted = std::sqrt(a * b * 300.f / (a + b));
            const float rel_err   = std::abs(sx_zupt - predicted) / predicted;
            std::snprintf(buf, sizeof buf,
                          "300 s parked: sigma_x %.3f m free-drift -> %.4f m with ZUPT "
                          "(closed form %.4f m, err %.1f%%), sigma_th %.4f rad",
                          sx_free, sx_zupt, predicted, rel_err * 100.f, st_zupt);
            // The second clause asserts only that the rest hypothesis TIGHTENS the interval. How much
            // is not a free parameter — it is set by the ratio of the two measured densities, and
            // demanding a fixed factor would silently re-impose a rest noise nobody measured.
            check("ZUPT bounds the parked interval to sqrt(a*b*T/(a+b))",
                  rel_err < 0.05f and sx_zupt < sx_free, buf);

            // (a2) RATE INVARIANCE — the test that would have caught the bug this replaced. The same
            // 300 s parked interval, integrated at three sample rates spanning 8x, must give the SAME
            // covariance: the rest hypothesis contributes information per unit of TIME, so how many
            // samples happened to land in the interval cannot matter. Under the old per-sample form
            // these three differed by sqrt(8) = 2.83x, and nothing in the suite objected.
            {
                const float s_4  = std::sqrt(run(0.f, 0.f, 0.f, 0.004f, 300.f, true).cov(0, 0));
                const float s_16 = std::sqrt(run(0.f, 0.f, 0.f, 0.016f, 300.f, true).cov(0, 0));
                const float s_32 = std::sqrt(run(0.f, 0.f, 0.f, 0.032f, 300.f, true).cov(0, 0));
                const float spread = (std::max({s_4, s_16, s_32}) - std::min({s_4, s_16, s_32}))
                                   / std::max(s_16, 1e-12f);
                std::snprintf(buf, sizeof buf,
                              "300 s parked at 4/16/32 ms: sigma_x %.5f / %.5f / %.5f m (spread %.2f%%)",
                              s_4, s_16, s_32, spread * 100.f);
                check("the parked ZUPT interval does not depend on the sample rate",
                      spread < 0.02f, buf);
            }

            // (b) Fades out with speed, on its own. No threshold: the gap closes because the rest
            // hypothesis' own variance grows with the observed motion.
            const auto slow_f = run(0.05f, 0.45f, 0.8f, 0.01f, 0.5f, false);
            const auto slow_z = run(0.05f, 0.45f, 0.8f, 0.01f, 0.5f, true);
            const auto fast_f = run(0.05f, 2.00f, 0.8f, 0.01f, 0.5f, false);
            const auto fast_z = run(0.05f, 2.00f, 0.8f, 0.01f, 0.5f, true);
            const float gap_slow = (slow_f.cov - slow_z.cov).norm() / slow_f.cov.norm();
            const float gap_fast = (fast_f.cov - fast_z.cov).norm() / fast_f.cov.norm();
            std::snprintf(buf, sizeof buf, "relative cov gap: %.1f%% at 0.45 m/s -> %.1f%% at 2.0 m/s",
                          gap_slow * 100.f, gap_fast * 100.f);
            check("ZUPT fades out as the robot moves", gap_fast < gap_slow and gap_fast < 0.02f, buf);

            // (c) A pivot is NOT rest. v_lat = v_long = 0 while spinning, so a per-axis test would
            // hand the pivot the parked translation noise; the lever arm is what prevents it.
            const auto pivot = run(0.f, 0.f, 0.8f, 0.008f, 10.f, true);
            const auto still = run(0.f, 0.f, 0.0f, 0.008f, 10.f, true);
            const float sx_pivot = std::sqrt(pivot.cov(0, 0));
            const float sx_still = std::sqrt(still.cov(0, 0));
            // ★ Assert that the LEVER ARM is what separates them, not a magnitude. The separation
            // factor is set by the lever against the rest density, so it moves whenever either is
            // re-measured — it was 5x when the rest noise was 5.9x too tight and is 1.3x now that it
            // has been measured, because a base whose wheels are genuinely that noisy simply cannot
            // distinguish a slow pivot from rest as sharply. Demanding the old factor would be
            // demanding the old wrong constant back. What must remain TRUE is the mechanism: with the
            // lever arm removed, a pivot becomes indistinguishable from rest, and that is the failure
            // this term exists to prevent.
            rc::preint::NoiseModel q_nolever = qn;
            q_nolever.zupt_enabled = true;
            // m_v = speed + lever·|omega|, so the lever is what lets a PURE ROTATION register as
            // motion in the translation channel. Setting it to ~0 removes exactly that coupling and
            // nothing else: the pivot then presents v_lat = v_long = 0 and is read as rest, which is
            // the failure this term exists to prevent. (The code floors it at 1e-3 to keep m_w finite.)
            q_nolever.zupt_lever_m = 1e-3f;
            rc::preint::Integrator g_nl(theta0);
            g_nl.set_noise(q_nolever);
            for (int i = 0; i < 1250; ++i) g_nl.add(0.f, 0.f, 0.8f, 0.008f);
            const float sx_pivot_nolever = std::sqrt(g_nl.result().cov(0, 0));
            std::snprintf(buf, sizeof buf,
                          "10 s: sigma_x %.4f m pivoting vs %.4f m still (%.2fx); with the lever "
                          "removed the pivot falls to %.4f m (%.2fx)",
                          sx_pivot, sx_still, sx_pivot / std::max(sx_still, 1e-9f),
                          sx_pivot_nolever, sx_pivot_nolever / std::max(sx_still, 1e-9f));
            // Without the lever a pivot collapses onto the parked answer; with it, it is strictly
            // looser. Both clauses are about the MECHANISM and neither pins a magnitude that would
            // move when the rest density is re-measured on another robot.
            check("the lever arm is what stops a pivot being read as rest",
                  sx_pivot > 1.15f * sx_pivot_nolever
                  and std::abs(sx_pivot_nolever - sx_still) < 0.05f * sx_still, buf);

            // (d) Continuous in speed — the property that makes this a model term and not a switch.
            // ★A RELATIVE step size cannot test this: P⁺ rises quadratically off a tiny rest floor, so
            // adjacent samples legitimately differ by hundreds of percent near v = 0 while the curve is
            // perfectly smooth. The discriminator that actually separates a smooth rise from a hidden
            // threshold is GRID REFINEMENT — a jump discontinuity keeps its size however fine the sweep,
            // a continuous function's largest step shrinks in proportion to the spacing.
            {
                const auto max_step_frac = [&](int n)
                {
                    float prev = -1.f, worst = 0.f, lo = 1e30f, hi = -1e30f;
                    bool monotonic = true;
                    for (int i = 0; i <= n; ++i)
                    {
                        const float v = 0.30f * static_cast<float>(i) / static_cast<float>(n);
                        const float c = run(0.f, v, 0.f, 0.008f, 1.f, true).cov(1, 1);
                        lo = std::min(lo, c); hi = std::max(hi, c);
                        if (prev >= 0.f)
                        {
                            if (c < prev * 0.999f) monotonic = false;
                            worst = std::max(worst, c - prev);
                        }
                        prev = c;
                    }
                    return std::pair<float, bool>{worst / std::max(hi - lo, 1e-30f), monotonic};
                };
                const auto [step_coarse, mono_c] = max_step_frac(60);
                const auto [step_fine,   mono_f] = max_step_frac(240);
                const float shrink = step_coarse / std::max(step_fine, 1e-30f);
                std::snprintf(buf, sizeof buf,
                              "largest step %.4f of range at 60 pts -> %.4f at 240 pts (%.1fx smaller), monotonic %s",
                              step_coarse, step_fine, shrink, (mono_c and mono_f) ? "yes" : "NO");
                check("ZUPT is continuous in speed (no hidden threshold)",
                      mono_c and mono_f and shrink > 3.f, buf);
            }
        }
    }


    // ═══ RGB edge alignment (ImageEdge) ═══════════════════════════════════════════════════════
    // Four questions, in the order in which getting one wrong makes the next meaningless:
    //   1. does the reduced CameraModel reproduce the projection, and its analytic Jacobian match?
    //   2. does the sub-pixel edge estimator hit its own claimed precision? (RMS/sigma ~ 1)
    //   3. does the common-mode cap SATURATE, and stay positive semi-definite?
    //   4. does the pose Jacobian of the actual residual match central differences?
    std::printf("\n-- image edge: camera model + jacobian ------------------------------------------\n");
    {
        char buf[256];
        rc::CameraModel cam;
        cam.kind = rc::CameraModel::Kind::Pinhole;
        cam.fx = 448.f; cam.fy = 448.f; cam.width = 1280.f; cam.height = 720.f;
        cam.cx = 640.f; cam.cy = 360.f; cam.valid = true;

        // Analytic P vs central difference of project_with_model itself.
        double worst = 0.0;
        const double pts[][3] = {{0.3,3.0,-1.0},{-1.2,2.0,0.4},{0.0,5.0,-1.08},{2.0,4.0,1.1}};
        for (const auto& q : pts)
        {
            const Eigen::Vector3d p(q[0], q[1], q[2]);
            Eigen::Matrix<double,2,3> Pa, Pn;
            if (not rc::img::project_jacobian_model(cam, p, Pa)) continue;
            const double h = 1e-6;
            for (int j = 0; j < 3; ++j)
            {
                Eigen::Vector3d pp = p, pm = p; pp[j] += h; pm[j] -= h;
                Eigen::Vector2d a, b;
                rc::img::project_with_model(cam, pp, a);
                rc::img::project_with_model(cam, pm, b);
                Pn(0,j) = (a.x()-b.x())/(2*h); Pn(1,j) = (a.y()-b.y())/(2*h);
            }
            worst = std::max(worst, (Pa-Pn).cwiseAbs().maxCoeff()
                                    / std::max(1.0, Pn.cwiseAbs().maxCoeff()));
        }
        std::snprintf(buf, sizeof buf, "max rel err %.2e", worst);
        check("pinhole dP/dp_cam matches finite diff", worst < 1e-5, buf);

        // The 360 branch must round-trip project -> ray -> project. Its azimuth convention is
        // RECOVERED from project() rather than assumed, so this guards the recovery too.
        rc::CameraModel eq;
        eq.kind = rc::CameraModel::Kind::Equirect;
        eq.width = 1920.f; eq.height = 960.f; eq.azimuth_sign = 1.f; eq.azimuth_offset = 0.f;
        eq.valid = true;
        Eigen::Vector2d uv0, uv90;
        rc::img::project_with_model(eq, Eigen::Vector3d(0,1,0), uv0);
        rc::img::project_with_model(eq, Eigen::Vector3d(1,0,0), uv90);
        std::snprintf(buf, sizeof buf, "az0 u=%.1f (want 960), az90 u=%.1f (want 1440)", uv0.x(), uv90.x());
        check("equirect azimuth mapping", std::abs(uv0.x()-960.)<1e-6 and std::abs(uv90.x()-1440.)<1e-6, buf);
    }

    std::printf("\n-- image edge: common-mode saturation -------------------------------------------\n");
    {
        char buf[256];
        // The acceptance test for the Woodbury cap, and the reason it exists: information must
        // SATURATE as samples are added to ONE segment, and H must stay positive semi-definite.
        // Run well past any realistic segment length -- the float32 version of this code went
        // INDEFINITE at N=1000 (trace -1.0e4), which is a negative variance downstream.
        std::mt19937 r2(99);
        std::normal_distribution<float> nz(0.f, 1.f);
        float tr_small = 0.f, tr_big = 0.f, min_eig = 0.f, raw_big = 0.f;
        for (int N : {10, 2000})
        {
            rc::ImageEdgeSegment seg;
            for (int k = 0; k < N; ++k)
            {
                rc::ImageEdgeSample smp;
                const float d = 2.f + 3.f * k / std::max(1, N - 1);
                smp.sigma_px = 0.4f; smp.pi_vis = 1.f; smp.search_L = 20.f;
                smp.h(0) = 0.0035f * 448.f;         // pitch: identical for every sample
                smp.h(1) = 0.010f * 448.f / d;      // height: varies along the line
                smp.h(2) = 0.0035f * 44.8f;
                smp.h(3) = 0.f;
                seg.samples.push_back(smp);
            }
            const auto acc = rc::img::accumulate_segment(seg,
                [&](std::size_t k, Eigen::Matrix<float,1,3>& J) -> float
                {
                    const float d = 2.f + 3.f * k / std::max<std::size_t>(1, seg.samples.size()-1);
                    J << 448.f / d, 0.2f, 30.f;
                    return 0.35f * nz(r2);
                },
                [](std::size_t) { return 0.0f; });
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(acc.H);
            if (N == 10) tr_small = acc.trace_eff;
            else { tr_big = acc.trace_eff; min_eig = es.eigenvalues().minCoeff(); raw_big = acc.trace_raw; }
        }
        const float growth = tr_big / std::max(1e-9f, tr_small);
        std::snprintf(buf, sizeof buf, "trace 10 -> 2000 samples: %.0f -> %.0f (%.2fx) while raw = %.3g",
                      tr_small, tr_big, growth, raw_big);
        check("information SATURATES with sample count", growth > 1.0f and growth < 3.0f, buf);
        std::snprintf(buf, sizeof buf, "min eigenvalue %.3e vs trace %.3e", min_eig, tr_big);
        check("capped H stays positive semi-definite", min_eig > -1e-4f * std::max(1.f, tr_big), buf);
    }

    std::printf("\n-- image edge: sub-pixel estimator + pose jacobian ------------------------------\n");
    {
        char buf[256];
        // A synthetic room rendered into a synthetic pinhole camera. The shading steps at the
        // projected wall-wall corners are ANTI-ALIASED, so the true edge sits at a known fractional
        // column; a hard `x >= u` would quantise the truth to the pixel grid and inject a ~0.29 px
        // error that has nothing to do with the estimator (that mistake made the CRB look 11x
        // optimistic when it was in fact correct to 9%).
        rc::CameraModel cam;
        cam.kind = rc::CameraModel::Kind::Pinhole;
        cam.fx = 448.f; cam.fy = 448.f; cam.width = 1280.f; cam.height = 720.f;
        cam.cx = 640.f; cam.cy = 360.f; cam.valid = true;
        const Eigen::Matrix3f Rc = Eigen::Matrix3f::Identity();
        const Eigen::Vector3f tc(0.f, 0.f, -1.08f);       // camera 1.08 m up (P3Bot ZED mount)
        const auto poly = room_polygon();
        const float room_h = 2.4f;
        const Eigen::Vector3f pose_t(0.20f, -0.30f, 0.15f);

        auto to_cam = [&](const Eigen::Vector3f& pr, const Eigen::Vector3f& po)
        {
            const float c = std::cos(po.z()), sn = std::sin(po.z());
            const Eigen::Vector3f e(pr.x()-po.x(), pr.y()-po.y(), pr.z());
            return Eigen::Vector3f(Rc * Eigen::Vector3f(c*e.x()+sn*e.y(), -sn*e.x()+c*e.y(), e.z()) + tc);
        };

        const int W = 1280, H = 720;
        std::vector<float> ucorner;
        for (const auto& v : poly)
        {
            Eigen::Vector2d uv;
            if (rc::img::project_with_model(cam, to_cam({v.x(), v.y(), 1.2f}, pose_t).cast<double>(), uv)
                and uv.x() > 2 and uv.x() < W-3) ucorner.push_back(static_cast<float>(uv.x()));
        }
        std::sort(ucorner.begin(), ucorner.end());
        auto shade = [&](double x) { int b = 0; for (float u : ucorner) if (x >= u) ++b; return 70.0 + 55.0*(b%3); };
        std::vector<std::uint8_t> img(static_cast<std::size_t>(W)*H);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
            {
                double acc = 0; const int S = 16;
                for (int k = 0; k < S; ++k) acc += shade(x - 0.5 + (k + 0.5)/S);
                img[static_cast<std::size_t>(y)*W + x] = static_cast<std::uint8_t>(std::lround(acc/S));
            }
        std::mt19937 r3(5);
        std::normal_distribution<float> pn(0.f, 1.5f);
        for (auto& px : img) px = static_cast<std::uint8_t>(std::clamp<int>(px + std::lround(pn(r3)), 0, 255));

        rc::GrayFrame frame;
        frame.gray = img; frame.width = W; frame.height = H; frame.stamp = 1; frame.valid = true;
        frame.sigma_i = rc::img::estimate_noise_sigma_immerkaer(frame.gray.data(), W, H);
        std::snprintf(buf, sizeof buf, "measured %.3f, rendered 1.500", frame.sigma_i);
        check("Immerkaer noise sigma recovers the truth", std::abs(frame.sigma_i - 1.5f) < 0.25f, buf);

        rc::ImageEdgeSource src;
        rc::ImageEdgeSource::Config ic;
        ic.use_wall_corners = true; ic.use_floor_junction = false;
        ic.room_height = room_h; ic.sample_spacing_m = 0.10f;
        src.set_config(ic); src.set_room_polygon(poly);
        rc::ImageEdgeSource::Stats st;
        const auto obs = src.extract(frame, cam, Rc, tc, pose_t,
                                     Eigen::Matrix3f::Identity()*1e-4f,
                                     Eigen::Vector3f::Zero(), 0, &st);

        auto resid_rms = [&](const Eigen::Vector3f& po)
        {
            double s2 = 0; int n = 0;
            for (const auto& sg : obs.segments) for (const auto& sm : sg.samples)
            {
                Eigen::Vector2d uv;
                if (not rc::img::project_with_model(cam, to_cam(sm.p_room, po).cast<double>(), uv)) continue;
                const double r = sm.n_hat.x()*(uv.x()-sm.uv_meas.x()) + sm.n_hat.y()*(uv.y()-sm.uv_meas.y());
                s2 += r*r; ++n;
            }
            return n ? std::sqrt(s2/n) : -1.0;
        };
        const double rms = resid_rms(pose_t);
        const double ratio = rms / std::max(1e-6f, st.med_sigma_px);
        std::snprintf(buf, sizeof buf, "%d samples, RMS %.4f px vs claimed sigma %.4f px (ratio %.2f)",
                      st.n_searched, rms, st.med_sigma_px, ratio);
        check("sub-pixel edge meets its CLAIMED precision", st.n_searched > 8 and ratio > 0.4 and ratio < 2.5, buf);

        // The residual must actually respond to pose. A term that does not is the circularity bug.
        const double rms_off = resid_rms(pose_t + Eigen::Vector3f(0.f, 0.f, 0.0087f));  // +0.5 deg
        std::snprintf(buf, sizeof buf, "RMS %.3f px at truth -> %.3f px at +0.5 deg yaw", rms, rms_off);
        check("residual RESPONDS to a pose perturbation", rms_off > 5.0 * rms, buf);

        // Pose Jacobian vs central differences, in DOUBLE (a 1e-6 probe of a float pose is float noise).
        double jworst = 0.0; int jn = 0;
        for (const auto& sg : obs.segments) for (const auto& sm : sg.samples)
        {
            auto to_cam_d = [&](const Eigen::Vector3d& po)
            {
                const double c = std::cos(po.z()), sn = std::sin(po.z());
                const Eigen::Vector3d e(sm.p_room.x()-po.x(), sm.p_room.y()-po.y(), (double)sm.p_room.z());
                return Eigen::Vector3d(Rc.cast<double>()
                       * Eigen::Vector3d(c*e.x()+sn*e.y(), -sn*e.x()+c*e.y(), e.z()) + tc.cast<double>());
            };
            Eigen::Matrix<double,1,3> Jn;
            bool ok = true;
            for (int j = 0; j < 3 and ok; ++j)
            {
                Eigen::Vector3d pp = pose_t.cast<double>(), pm = pose_t.cast<double>();
                const double h = 1e-6; pp[j] += h; pm[j] -= h;
                Eigen::Vector2d a, b;
                ok = rc::img::project_with_model(cam, to_cam_d(pp), a)
                 and rc::img::project_with_model(cam, to_cam_d(pm), b);
                if (ok) Jn(0,j) = (sm.n_hat.x()*(a.x()-b.x()) + sm.n_hat.y()*(a.y()-b.y())) / (2*h);
            }
            if (not ok) continue;
            Eigen::Matrix<double,2,3> P;
            if (not rc::img::project_jacobian_model(cam, to_cam_d(pose_t.cast<double>()), P)) continue;
            const double c = std::cos((double)pose_t.z()), sn = std::sin((double)pose_t.z());
            Eigen::Matrix3d Rm; Rm << c, sn, 0, -sn, c, 0, 0, 0, 1;
            const Eigen::Vector3d e(sm.p_room.x()-pose_t.x(), sm.p_room.y()-pose_t.y(), (double)sm.p_room.z());
            const Eigen::Vector3d prb = Rm * e;
            Eigen::Matrix3d Jx;
            Jx.col(0) = Rc.cast<double>() * (-Rm.col(0));
            Jx.col(1) = Rc.cast<double>() * (-Rm.col(1));
            Jx.col(2) = Rc.cast<double>() * Eigen::Vector3d(prb.y(), -prb.x(), 0.0);
            const Eigen::Matrix<double,1,3> Ja = sm.n_hat.transpose().cast<double>() * P * Jx;
            jworst = std::max(jworst, (Ja-Jn).cwiseAbs().maxCoeff()
                                      / std::max(1.0, Jn.cwiseAbs().maxCoeff()));
            ++jn;
        }
        std::snprintf(buf, sizeof buf, "max rel err %.2e over %d samples", jworst, jn);
        check("pose Jacobian d r / d x matches finite diff", jn > 8 and jworst < 1e-4, buf);
    }

    std::printf("\n-- image edge: panorama seam ----------------------------------------------------\n");
    {
        char buf[256];
        // The Ricoh's column axis is CYCLIC: column W-1 and column 0 are neighbours on one continuous
        // sphere. A wall corner is near-vertical, so its search normal is near-HORIZONTAL and the
        // whole normal search runs along u — which means a corner near the seam has its window
        // straddle the cut. If the sampler does not wrap, half that window returns nothing and the
        // peak is taken from whichever half stayed in bounds. That is a BIASED MATCH reported with a
        // finite sigma, not a missing sample, so nothing downstream can tell it apart from a real
        // edge. This test puts the truth on the far side of the seam on purpose.
        constexpr int W = 1920, H = 240;
        const float u_true = 1915.6f;                 // 4.4 px BELOW the cut
        const float u_pred = 3.0f;                    // ... predicted 7.4 px above it, across the seam
        const float s_true = -7.4f;                   // u_pred + s_true == u_true (mod W)
        std::vector<std::uint8_t> img(static_cast<std::size_t>(W) * H);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
            {
                // Signed distance to the edge, ON THE CIRCLE.
                float d = static_cast<float>(x) - u_true;
                while (d >  0.5f * W) d -= W;
                while (d <= -0.5f * W) d += W;
                // Gaussian-blurred step (b = 1 px): |dI/du| is then a smooth peak at d = 0, which is
                // what the parabolic vertex is entitled to assume.
                const float f = 0.5f * (1.f + std::erf(d / std::sqrt(2.0f)));
                img[static_cast<std::size_t>(y) * W + x] =
                    static_cast<std::uint8_t>(std::lround(40.f + 170.f * f));
            }

        // The same search image_edge_source.cpp runs, at both settings of the one thing under test.
        const Eigen::Vector2f n_hat(1.f, 0.f);
        const auto search = [&](int wrap_u, bool& found) -> float
        {
            constexpr int steps = 12;                 // reaches across the seam from u_pred
            std::vector<float> prof;
            float best_mag = 0.f, best_s = 0.f;
            int   best_i = 0;
            for (int i = -steps; i <= steps; ++i)
            {
                float gd = 0.f;
                if (not rc::img::dir_derivative(img.data(), W, H, u_pred + static_cast<float>(i),
                                                0.5f * H, n_hat, gd, wrap_u))
                { prof.push_back(0.f); continue; }
                const float m = std::fabs(gd);
                prof.push_back(m);
                if (m > best_mag) { best_mag = m; best_s = static_cast<float>(i);
                                    best_i = static_cast<int>(prof.size()) - 1; }
            }
            found = best_mag > 1.f;
            if (best_i > 0 and best_i + 1 < static_cast<int>(prof.size()))
                best_s += rc::img::parabolic_vertex(prof[best_i-1], prof[best_i], prof[best_i+1]);
            return best_s;
        };

        bool found_w = false, found_n = false;
        const float s_wrap = search(W, found_w);
        const float s_none = search(0, found_n);

        std::snprintf(buf, sizeof buf, "recovered s = %.3f px, truth %.3f px (err %.3f)",
                      s_wrap, s_true, std::fabs(s_wrap - s_true));
        check("seam-straddling search finds the edge WITH wrap",
              found_w and std::fabs(s_wrap - s_true) < 0.15f, buf);

        // The negative control. It must FAIL, and it must be recorded that it fails, because this is
        // the defect the wrap exists to remove -- if this line ever starts passing on its own, the
        // sampler changed underneath and the positive test above stopped proving anything.
        std::snprintf(buf, sizeof buf, "no-wrap gives s = %.3f px (found=%d) vs truth %.3f",
                      s_none, static_cast<int>(found_n), s_true);
        check("... and does NOT find it without wrap (negative control)",
              not found_w ? false : (not found_n or std::fabs(s_none - s_true) > 2.0f), buf);

        // Away from the seam the two must be INDISTINGUISHABLE: wrapping may not change any answer
        // it was not introduced to change. A pinhole frame keeps sampling exactly as it did.
        const float u_mid = 960.0f;
        const auto search_at = [&](float u0, int wrap_u) -> float
        {
            constexpr int steps = 12;
            std::vector<float> prof;
            float best_mag = 0.f, best_s = 0.f; int best_i = 0;
            for (int i = -steps; i <= steps; ++i)
            {
                float gd = 0.f;
                if (not rc::img::dir_derivative(img.data(), W, H, u0 + static_cast<float>(i),
                                                0.5f * H, n_hat, gd, wrap_u))
                { prof.push_back(0.f); continue; }
                const float m = std::fabs(gd);
                prof.push_back(m);
                if (m > best_mag) { best_mag = m; best_s = static_cast<float>(i);
                                    best_i = static_cast<int>(prof.size()) - 1; }
            }
            if (best_i > 0 and best_i + 1 < static_cast<int>(prof.size()))
                best_s += rc::img::parabolic_vertex(prof[best_i-1], prof[best_i], prof[best_i+1]);
            return best_s;
        };
        const float d_mid = std::fabs(search_at(u_mid, W) - search_at(u_mid, 0));
        std::snprintf(buf, sizeof buf, "|wrap - nowrap| = %.2e px at u = %.0f", d_mid, u_mid);
        check("wrapping changes NOTHING away from the seam", d_mid < 1e-6f, buf);
    }

    std::printf("\n%s (%d failure%s)\n\n", failures == 0 ? "ALL PASS" : "FAILURES",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
