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

    struct Case { const char* name; bool sdf, motion, corner, object, boundary; };
    for (const Case& cs : std::vector<Case>{
            {"SDF only",            true,  false, false, false, false},
            {"motion only",         false, true,  false, false, false},
            {"corner only",         false, false, true,  false, false},
            {"object anchor only",  false, false, false, true,  false},
            {"boundary only",       false, false, false, false, true },
            {"all factors",         true,  true,  true,  true,  true }})
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

    std::printf("\n%s (%d failure%s)\n\n", failures == 0 ? "ALL PASS" : "FAILURES",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
