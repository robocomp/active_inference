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
            // collapses to the ZUPT's own, so the interval has a closed form: sigma = sigma_rest·√(dt·T).
            // Checking against that pins the algebra and stays correct when sigma_rest is re-measured on
            // another platform — an absolute bound would just re-encode today's number, which is exactly
            // how the first version of this test came to enshrine a sigma_rest that was 10x too tight.
            const float predicted = rc::preint::NoiseModel{}.zupt_sigma_v * std::sqrt(0.008f * 300.f);
            const float rel_err   = std::abs(sx_zupt - predicted) / predicted;
            std::snprintf(buf, sizeof buf,
                          "300 s parked: sigma_x %.3f m free-drift -> %.4f m with ZUPT "
                          "(closed form %.4f m, err %.1f%%), sigma_th %.4f rad",
                          sx_free, sx_zupt, predicted, rel_err * 100.f, st_zupt);
            check("ZUPT bounds the parked interval to sigma_rest*sqrt(dt*T)",
                  rel_err < 0.05f and sx_zupt < 0.2f * sx_free, buf);

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
            std::snprintf(buf, sizeof buf, "10 s: sigma_x %.4f m pivoting vs %.4f m truly still (%.0fx)",
                          sx_pivot, sx_still, sx_pivot / std::max(sx_still, 1e-9f));
            // The factor of separation is set by the lever arm against sigma_rest, so it moves whenever
            // either is re-measured; what must hold is that a pivot is treated as MOTION, decisively.
            check("a pivot is not mistaken for rest", sx_pivot > 5.f * sx_still, buf);

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

    std::printf("\n%s (%d failure%s)\n\n", failures == 0 ? "ALL PASS" : "FAILURES",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
