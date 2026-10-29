// RoomConcept, split out of room_concept.cpp (refactor step 2, docs/refactor_proposal_2026-10-13.md).
// Prediction: motion calibrator feed, surprise/noise adaptation, forward prediction, velocity integration.

#include "room_concept.h"
#include "../../common/status_stream/status_sink.h"
#include <pthread.h>   // pthread_setname_np: name the worker so a per-thread CPU sample attributes itself
#include "pointcloud_center_estimator.h"
#include <numeric>
#include <set>
#include "room_gn_solver.h"
#include "room_obs_weights.h"
#include "room_gn_solver.h"
#include "image_edge_accumulate.h"
#include "image_edge_ops.h"
#include <algorithm>
#include <ranges>
#include <fstream>
#include <filesystem>
#include <limits>
#include <locale>
#include <ctime>
#include <print>
#include <sstream>
#include <sys/stat.h>
#include <QDebug>

namespace rc
{

    // Innovation-based adaptive covariance — see Params::adaptive_cov_enabled.
    // The innovation is the difference between two estimates of the SAME pose (SDF-optimised vs
    // odometry-predicted), so its running second moment is a sample of the estimator's own error and a
    // lower bound the published covariance must not undercut. Taking the max with the Laplace value
    // leaves good frames alone and lifts sigma only where the estimator is demonstrably disagreeing
    // with itself, which is what keeps the mean (and therefore the consumer's speed governor) intact.
    // Hand the optimizer's own correction to the slow parameter learner. Called on BOTH return
    // paths: the early-exit path contributes the RAMP (correction is identically zero there) and the
    // optimized path contributes the correction that ends it. Feeding only the latter would show the
    // learner an error with no record of the motion that produced it.
    void RoomConcept::feed_motion_calibrator(UpdateResult &res)
    {
        // Params are assigned directly onto .params with no init hook, so bind on first use.
        if (not motion_calib_.configured())
        {
            auto mc = params.motion_calib;
            mc.fit_sigma = params.sigma_sdf;   // the heading R's expected misfit is the localiser's own noise
            motion_calib_.configure(mc);
        }
        if (motion_calib_.enabled())
        {
            const float th = std::atan2(res.robot_pose.linear()(1, 0), res.robot_pose.linear()(0, 0));
            const float ex = res.robot_pose.translation().x() - res.pred_x;
            const float ey = res.robot_pose.translation().y() - res.pred_y;
            // FORWARD is th + 90 deg on this robot (see the frame note in motion_calibration.h):
            // u_fwd = (-sin th, cos th), u_lat = (cos th, sin th). Measured, not assumed.
            const float c = std::cos(th), sn = std::sin(th);
            const float r_forward = -ex * sn + ey * c;
            const float r_lateral =  ex * c  + ey * sn;
            float r_theta = th - res.pred_theta;
            while (r_theta >  static_cast<float>(M_PI)) r_theta -= 2.f * static_cast<float>(M_PI);
            while (r_theta < -static_cast<float>(M_PI)) r_theta += 2.f * static_cast<float>(M_PI);

            // The prediction increment in the SAME frame, and the posterior variance that says how
            // much this correction should be believed.
            const float pos_var = res.covariance.rows() > 1
                ? 0.5f * (res.covariance(0, 0) + res.covariance(1, 1)) : 0.f;
            const float th_var = res.covariance.rows() > 2 ? res.covariance(2, 2) : 0.f;

            res.calib_pos_var = pos_var;
            // ★ THE POLISH IS A CORRECTOR TOO. r_forward/r_lateral/r_theta already carry it without
            // any extra plumbing — they are res.robot_pose (read AFTER the polish moved it) minus
            // pred_* (captured BEFORE) — but they are only ACCUMULATED when `corrected` is true, and
            // that was "the optimizer ran". On the ~99% of cycles the optimizer skips, the polish is
            // now the thing correcting the pose, and its corrections were being discarded.
            // The variances handed over stay the localiser's own: a polish step is bounded by the
            // motion prior, so it cannot contribute a correction the prior called implausible.
            // ★ REVERTED 2026-08-26: the polish is NOT a corrector the calibrator may learn from.
            // Its steps remove NOISE-driven drift, and the estimator has no way to know that — it
            // explains the accumulated correction as a systematic scale. Live, with the weights
            // finally real, that drove the raw solve far outside its priors (the forward scale and
            // the wheel mismatch both many sigmas out) while `informed` stayed false. The `informed`
            // gate is the only reason none of it reached the wheels.
            // The original trigger is viable again anyway: the optimizer now fires on ~40% of cycles,
            // not 0.3%, so learning from full windowed solves is no longer starved. Those corrections
            // are dominated by systematic model error accumulated over a ramp, which is the quantity
            // a scale parameter is entitled to explain.
            // ⚠ The MOTION-based episode close stays — that part is independent and sound.
            const bool corrected_this_cycle = res.iterations_used > 0;
            // ★ THE COVARIATES ARE WHAT THE PREDICTION USED, so they carry the rest gain (2026-10-04).
            // With rest-on-prediction the increment is scaled by P(moving) before it enters the
            // prediction; a parameter acts on the prediction only through what got in. Unscaled, a gyro
            // bias was regressed over every parked second the rest filter had (correctly) withheld from
            // the prediction -- 85% of cycles on arm 7's gyro-all leg -- and its estimate was diluted.
            // Measured there: the prediction's own per-cycle bias was +0.00045 rad/s on rest cycles and
            // +0.00339 while moving; regressing on t_gyro x gain gives +0.00343.
            const float g_tr = zupt_pred_gain_tr_, g_ro = zupt_pred_gain_ro_;   // 1 when the block did not scale
            auto hc = res.heading_cov;
            hc.th_gyro *= g_ro; hc.t_gyro *= g_ro; hc.th_wheel *= g_ro; hc.fwd_wheel *= g_tr;
            motion_calib_.observe(res.dy_local * g_tr, res.dx_local * g_tr,
                                  (res.imu_dtheta + res.wheel_dtheta) * g_ro,
                                  r_forward, r_lateral, r_theta,
                                  pos_var, th_var, corrected_this_cycle, res.sdf_mse,
                                  last_cycle_dt_s_, hc);   // wall time: the intake judges speed with it
        }
        res.calib_value = motion_calib_.last_solve().value;
        res.calib_sigma = motion_calib_.last_solve().sigma;
        res.calib_b_omega = motion_calib_.estimated_omega_bias();
        {
            const auto &r = motion_calib_.last_solve();
            // Bit p = parameter p informed. The low four bits keep their old meaning (k_v, eps_yaw,
            // k_omega, b_omega); k_lat, dk_wheel and k_omega_w follow in enum order.
            res.calib_informed = 0;
            for (int p = 0; p < rc::calib::P_COUNT; ++p)
                if (r.informed[p]) res.calib_informed |= (1 << p);
            res.calib_condition = r.condition;
        }
        res.calib_sigma_b_omega = motion_calib_.last_solve().sigma[rc::calib::P_B_OMEGA];
        res.calib_sigma_yaw = motion_calib_.yaw_sigma();
        res.calib_sigma_k_v = motion_calib_.k_v_sigma();
        res.calib_sigma_k_w = motion_calib_.k_w_sigma();
        // ★ The ESTIMATE, not the applied value: in open-loop mode (MotionCalibApply=false) the
        // applied accessors return nominal by design, and logging those would blind the very arm
        // the mode exists to run. Identical to the applied values whenever apply is true.
        res.calib_k_v = motion_calib_.estimated_forward_scale();
        res.calib_k_w = motion_calib_.estimated_omega_scale();
        res.calib_yaw = motion_calib_.estimated_yaw_offset();
        res.calib_episodes = motion_calib_.episodes();
        res.calib_applied = 0;
        for (int i = 0; i < rc::calib::P_COUNT; ++i)
            if (motion_calib_.acting(i)) res.calib_applied |= 1 << i;
        res.calib_carried = motion_calib_.carried();
        res.calib_dropped = motion_calib_.dropped();
    }

    void RoomConcept::score_surprise(UpdateResult& res, bool corrected)
    {
        res.surprise = {};
        fill_noise_vc(res);
        if (not corrected) return;
        const Eigen::Matrix3f P_post = res.covariance;
        if (not sur_init_)
        {   // the first correction has no pose prior to be scored against; it only seeds the recursion
            if (P_post.allFinite()) { sur_P_pred_ = sur_P_prev_ = P_post; reset_noise_stretch(); sur_init_ = true; }
            sur_d_ = sur_phi_ = sur_T_ = 0.f;
            return;
        }
        const float est_th = std::atan2(res.robot_pose.linear()(1, 0), res.robot_pose.linear()(0, 0));
        const Eigen::Vector3f c{res.robot_pose.translation().x() - res.pred_x,
                                res.robot_pose.translation().y() - res.pred_y,
                                std::remainder(est_th - res.pred_theta, 2.f * float(M_PI))};
        res.surprise = rc::surprise::kl_gauss(c, sur_P_pred_, P_post);
        // Per axis in the body frame: forward = (-sin th, cos th), lateral = (cos th, sin th) on this robot.
        const Eigen::Vector2f uf(-std::sin(est_th), std::cos(est_th)), ul(std::cos(est_th), std::sin(est_th));
        // ★ The per-axis posterior is the SOLVER's, from BEFORE apply_adaptive_covariance. That floor sets
        //   P(i,i) = max(P, EMA v², v²): a posterior that is a function of this very correction, which voids
        //   E[c²] = P_pred - P_post (Fable 2026-10-04: ~200/874 corrections were floor-bound and carried
        //   87-97% of sum c², poisoning kappa). current_covariance is the filter state the floor never
        //   touches (res.covariance = current_covariance, then the floor edits only res). The PREDICTION
        //   side stays on the published value: it is what prev_cov -> predict_step actually propagated.
        const Eigen::Matrix3f P_solver = current_covariance.allFinite() ? current_covariance : P_post;
        res.surprise.floor_bound = (P_post.diagonal() - P_solver.diagonal()).maxCoeff() > 0.f;
        const Eigen::Matrix2f Pp = sur_P_pred_.topLeftCorner<2, 2>(), Pq = P_solver.topLeftCorner<2, 2>();
        res.surprise.c_fwd = uf.dot(c.head<2>());  res.surprise.c_lat = ul.dot(c.head<2>());  res.surprise.c_th = c[2];
        res.surprise.pp_fwd = uf.dot(Pp * uf);     res.surprise.pp_lat = ul.dot(Pp * ul);     res.surprise.pp_th = sur_P_pred_(2, 2);
        res.surprise.pq_fwd = uf.dot(Pq * uf);     res.surprise.pq_lat = ul.dot(Pq * ul);     res.surprise.pq_th = P_solver(2, 2);
        res.surprise.open_cycles = sur_open_cycles_;
        // The motion-noise components are learnt from the SCAN-TO-SCAN innovation every cycle
        // (observe_innovation, motion_noise_innov.h), not from this correction: the window posterior is not
        // one Kalman update, so (c, P_pred, P_post) cannot give the gain (Fable window memo 2026-10-05).
        fill_noise_vc(res);
        // ── the systematic drift (drift_monitor.h): c against the stretch's own odometry motion ──
        drift_mon_.observe(Eigen::Vector3d(res.surprise.c_fwd, res.surprise.c_lat, res.surprise.c_th),
                           Eigen::Vector3d(std::max(res.surprise.pp_fwd - res.surprise.pq_fwd, 0.f),
                                           std::max(res.surprise.pp_lat - res.surprise.pq_lat, 0.f),
                                           std::max(res.surprise.pp_th  - res.surprise.pq_th,  0.f)),
                           sur_d_, sur_phi_, sur_T_, sur_open_cycles_);
        {
            Eigen::Matrix3d b, se; drift_mon_.solve(b, se);
            for (int a = 0; a < 3; ++a) for (int j = 0; j < 3; ++j)
            { res.surprise.drift_b[3 * a + j] = float(b(a, j)); res.surprise.drift_se[3 * a + j] = float(se(a, j)); }
            res.surprise.drift_n = int(drift_mon_.stretches());
        }
        sur_d_ = sur_phi_ = sur_T_ = 0.f;
        sur_open_cycles_ = 0;
        reset_noise_stretch();
        // ★ Re-seed from the SOLVER's posterior, not the floored one (Fable 2026-10-05 window memo §1).
        //   apply_adaptive_covariance sets P(i,i) = max(P, EMA v², v²) with v THIS correction, so seeding
        //   from it made the next P_pred >= c_prev² by construction: measured pp(n)/pq(n-1) median 3.07,
        //   p90 9.7 on heading after a floor-bound correction (vs 1.65 unfloored). The comment above about
        //   prev_cov -> predict_step concerns a path the window never reads while CovarianceFromSolver.
        const Eigen::Matrix3f &seed = P_solver.allFinite() ? P_solver : P_post;
        if (seed.allFinite()) { sur_P_pred_ = sur_P_prev_ = seed; }
    }

    void RoomConcept::reset_noise_stretch()
    {
        for (auto &u : sur_U_) u.setZero();
        sur_QK_.setZero();
        sur_mixed_ = false;
    }

    void RoomConcept::fill_noise_vc(UpdateResult &res) const
    {
        const auto &np = params.odom_preint_noise;
        res.surprise.vc = {np.k_long, np.k_lat, np.k_lat_turn, np.k_th_turn, np.k_t_trans, np.k_t_rot,
                           float(noise_innov_.s(0)), float(noise_innov_.s(1)), float(noise_innov_.s(2))};
        res.surprise.vc_trained = not sur_mixed_ and sur_open_cycles_ > 0;
    }

    void RoomConcept::apply_adaptive_covariance(UpdateResult& res)
    {
        if (not params.adaptive_cov_enabled)
            return;
        const float lambda = std::clamp(params.adaptive_cov_lambda, 1e-4f, 1.0f);
        for (int i = 0; i < 3; ++i)
        {
            const float v = res.innovation[i];
            if (not std::isfinite(v))
                continue;
            innov_m2_[i] = (1.0f - lambda) * innov_m2_[i] + lambda * v * v;
            // BOTH terms, and the instantaneous one is the load-bearing half. With the EMA alone this
            // measured a ratio of 0.788 — WORSE than the flat baseline of 1.059, and inverted: the
            // largest-innovation frames reported sigma 0.0427 while the smallest reported 0.0542. Two
            // reasons. The EMA only clears the ~46.6 mm Laplace floor if the innovation RMS is
            // SUSTAINED above it, which a 12 mm median never does; and being an average it rises one
            // time-constant AFTER the frames that caused it, so on the frame that actually disagreed it
            // has not moved yet. |v| on this frame is a direct observation that two estimates of this
            // same pose differ by that much, so the uncertainty is at least that — not circular, just
            // refusing to claim precision the estimator has already contradicted.
            if (std::isfinite(res.covariance(i, i)))
                res.covariance(i, i) = std::max({res.covariance(i, i), innov_m2_[i], v * v});
        }
    }

    Eigen::Affine2f RoomConcept::predict_pose_forward(const Eigen::Affine2f& pose,
                                                      float adv, float side, float rot, float dt) const
    {
        if (dt <= 0.f)
            return pose;
        const float theta = std::atan2(pose.linear()(1, 0), pose.linear()(0, 0));
        const float dx_local = adv * dt;
        const float dy_local = side * dt;
        const float dtheta = rot * dt;                  // velocity is CCW+; use directly
        const float theta_mid = theta + 0.5f * dtheta;  // midpoint integration (reduces bias)
        Eigen::Affine2f out = Eigen::Affine2f::Identity();
        out.translation() = pose.translation()
            + Eigen::Vector2f(dx_local * std::cos(theta_mid) - dy_local * std::sin(theta_mid),
                              dx_local * std::sin(theta_mid) + dy_local * std::cos(theta_mid));
        out.linear() = Eigen::Rotation2Df(theta + dtheta).toRotationMatrix();
        return out;
    }

     Eigen::Vector3f RoomConcept::integrate_velocity_over_window(
                const Eigen::Affine2f& robot_pose,
                const std::vector<VelocityCommand> &velocity_history,
                const std::int64_t &t_start_ms,
                const std::int64_t &t_end_ms,
                rc::preint::Interval *preint_out)
    {
        Eigen::Vector3f total_delta = Eigen::Vector3f::Zero();

        float running_theta = std::atan2(robot_pose.linear()(1,0), robot_pose.linear()(0,0));

        // See the note in integrate_odometry_over_window: observer only, never touches total_delta.
        rc::preint::Integrator preint(running_theta);
        preint.set_noise(params.cmd_preint_noise);

        // Integrate over all velocity commands in [t_start_ms, t_end_ms] using source/recv epoch-ms.
        for (size_t i = 0; i < velocity_history.size(); ++i) {
            const auto& vcmd = velocity_history[i];

            // Get time window for this command
            const std::int64_t cmd_start_ms = vcmd.effective_ts_ms();
            const std::int64_t cmd_end_ms = (i + 1 < velocity_history.size())
                           ? velocity_history[i + 1].effective_ts_ms()
                           : t_end_ms;

            if (cmd_start_ms <= 0 || cmd_end_ms <= 0 || cmd_end_ms <= cmd_start_ms)
                continue;

            // Clip to [t_start_ms, t_end_ms]
            if (cmd_end_ms < t_start_ms) continue;
            if (cmd_start_ms > t_end_ms) break;

            const std::int64_t effective_start_ms = std::max(cmd_start_ms, t_start_ms);
            const std::int64_t effective_end_ms = std::min(cmd_end_ms, t_end_ms);

            const float dt = static_cast<float>(effective_end_ms - effective_start_ms) * 0.001f;
            if (dt <= 0) continue;

            // Integrate this segment
            const float dx_local = (vcmd.adv_x * dt);
            const float dy_local = (vcmd.adv_y * dt);

            const float dtheta = vcmd.rot * dt;   // velocity buffer is CCW+; use directly

            // Transform to global frame using MIDPOINT theta (reduces integration bias)
            const float theta_mid = running_theta + 0.5f * dtheta;
            total_delta[0] += dx_local * std::cos(theta_mid) - dy_local * std::sin(theta_mid);
            total_delta[1] += dx_local * std::sin(theta_mid) + dy_local * std::cos(theta_mid);
            total_delta[2] += dtheta;

            if (preint_out != nullptr)
                preint.add(vcmd.adv_x, vcmd.adv_y, vcmd.rot, dt);

            // Update running theta for next segment
            running_theta += dtheta;
        }

        if (preint_out != nullptr)
            *preint_out = preint.result();

        return total_delta;
    }

    RoomConcept::PredictionState RoomConcept::predict_step(
                                std::shared_ptr<Model> &room,
                                const OdometryPrior &odometry_prior,
                                bool is_localized)
    {
        // Only the localized [x, y, θ] form is ever used (the [w, h, x, y, θ] mapping form had no caller
        // when torch went, 2026-10-17, and was dropped with it).
        (void)is_localized;
        PredictionState prediction;

        // Get current pose for Jacobian computation
        if (not room->robot_prev_pose.has_value())
        {
            // Fallback: simple additive noise
            prediction.propagated_cov = room->prev_cov
                + params.cmd_noise_trans * params.cmd_noise_trans * Eigen::Matrix3f::Identity();
            return prediction;
        }

        const auto& robot_prev_pose = room->robot_prev_pose.value();
        const float theta = std::atan2(robot_prev_pose.linear()(1, 0), robot_prev_pose.linear()(0, 0));

        // Transform global delta back to robot frame for noise computation
        const float cos_t = std::cos(theta);
        const float sin_t = std::sin(theta);
        const float dx_global = odometry_prior.delta_pose[0];
        const float dy_global = odometry_prior.delta_pose[1];
        // Inverse rotation: robot_frame = R^T * global_frame
        const float dx_local = dx_global * cos_t + dy_global * sin_t;
        const float dy_local = -dx_global * sin_t + dy_global * cos_t;

        // ===== MOTION MODEL JACOBIAN =====
        // ∂x/∂θ = -dx_local·sin(θ) - dy_local·cos(θ);  ∂y/∂θ = dx_local·cos(θ) - dy_local·sin(θ)
        Eigen::Matrix3f F = Eigen::Matrix3f::Identity();
        F(0, 2) = -dx_local * sin_t - dy_local * cos_t;
        F(1, 2) =  dx_local * cos_t - dy_local * sin_t;

        // ===== EKF PREDICTION =====
        // P_pred = F * P_prev * F^T + Q
        prediction.propagated_cov = F * room->prev_cov * F.transpose() + odometry_prior.covariance_eigen;
        prediction.have_propagated = true;
        return prediction;
    }

} // namespace rc
