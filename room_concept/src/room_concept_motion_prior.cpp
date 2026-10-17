// RoomConcept, split out of room_concept.cpp (refactor step 2, docs/refactor_proposal_2026-10-13.md).
// Motion prior: source selection, odometry/command priors, motion covariance, fusion, velocity-adaptive weights.

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

    Eigen::Vector3f RoomConcept::compute_velocity_adaptive_weights(const OdometryPrior& odometry_prior)
    {
        /**
         * Compute velocity-adaptive precision weights for [x, y, theta].
         *
         * Based on the current velocity profile:
         * - If rotating (high angular, low linear): boost theta weight, reduce x,y
         * - If moving straight (high linear, low angular): boost x,y, reduce theta
         * - If stationary: use base weights (uniform)
         *
         * The weights scale gradients during optimization, making the system
         * more responsive to parameters expected to change based on motion.
         */
        if (!params.velocity_adaptive_weights || !odometry_prior.valid)
        {
            return Eigen::Vector3f::Ones();
        }

        // Get velocities from odometry prior (dt is in milliseconds, convert to seconds)
        const float dt_sec = std::max(odometry_prior.dt / 1000.0f, 0.001f);
        const float linear_speed = odometry_prior.delta_pose.head<2>().norm() / dt_sec;
        const float angular_speed = std::abs(odometry_prior.delta_pose[2]) / dt_sec;

        // Determine motion profile
        const bool is_rotating = angular_speed > params.angular_velocity_threshold;
        const bool is_translating = linear_speed > params.linear_velocity_threshold;

        float w_x, w_y, w_theta;

        if (is_rotating && !is_translating)
        {
            // Pure rotation: emphasize theta, de-emphasize x, y
            w_x = params.weight_reduction_factor;
            w_y = params.weight_reduction_factor;
            w_theta = params.weight_boost_factor;
        }
        else if (is_translating && !is_rotating)
        {
            // Pure translation: emphasize x, y based on direction
            // Get velocity direction in robot frame
            const float vx = odometry_prior.delta_pose[0] / std::max(odometry_prior.dt, 0.001f);
            const float vy = odometry_prior.delta_pose[1] / std::max(odometry_prior.dt, 0.001f);

            if (std::abs(vy) > std::abs(vx))
            {
                // Mostly forward/backward motion - emphasize y (forward axis)
                w_x = 1.0f;
                w_y = params.weight_boost_factor;
            }
            else
            {
                // Mostly lateral motion - emphasize x
                w_x = params.weight_boost_factor;
                w_y = 1.0f;
            }
            w_theta = params.weight_reduction_factor;
        }
        else if (is_rotating && is_translating)
        {
            // Combined motion: moderate boost for all
            w_x = params.combined_motion_weight;
            w_y = params.combined_motion_weight;
            w_theta = params.combined_motion_weight;
        }
        else
        {
            // Stationary: use base weights
            w_x = 1.0f;
            w_y = 1.0f;
            w_theta = 1.0f;
        }

        Eigen::Vector3f new_weights(w_x, w_y, w_theta);

        // Smooth transition using exponential moving average
        const float alpha = params.weight_smoothing_alpha;
        current_velocity_weights_ = (1.0f - alpha) * current_velocity_weights_ + alpha * new_weights;

        return current_velocity_weights_;
    }

    // =========================================================================
    //  update() helper methods
    // =========================================================================

    std::string_view RoomConcept::motion_prior_source_name(MotionPriorSource source)
    {
        switch (source)
        {
            case MotionPriorSource::None: return "none";
            case MotionPriorSource::Command: return "command";
            case MotionPriorSource::Measured: return "measured";
            case MotionPriorSource::Fused: return "fused";
            case MotionPriorSource::FallbackZero: return "fallback_zero";
        }
        return "unknown";
    }

    RoomConcept::MotionPriorSelection RoomConcept::build_motion_prior_selection(
        const std::vector<VelocityCommand>& velocity_history,
        const std::vector<OdometryReading>& odometry_history,
        const std::pair<std::vector<Eigen::Vector3f>, std::int64_t>& lidar)
    {
        MotionPriorSelection selection;
        auto state = model_->get_state();
        selection.predicted_pos = Eigen::Vector2f(state[2], state[3]);
        selection.predicted_theta = state[4];

        selection.command_prior = compute_odometry_prior(velocity_history, lidar);
        selection.measured_prior = compute_measured_odometry_prior(odometry_history, lidar);
        last_measured_prior_ = selection.measured_prior;
        last_command_prior_fresh_ = selection.command_prior.fresh;
        last_measured_prior_fresh_ = selection.measured_prior.fresh;
        last_cmd_cov_ = selection.command_prior.covariance_eigen;

        const auto wrap_angle = [](float angle)
        {
            while (angle > static_cast<float>(M_PI)) angle -= 2.f * static_cast<float>(M_PI);
            while (angle < -static_cast<float>(M_PI)) angle += 2.f * static_cast<float>(M_PI);
            return angle;
        };

        const Eigen::Vector2f base_pos = last_update_result.ok
            ? last_update_result.robot_pose.translation()
            : Eigen::Vector2f(state[2], state[3]);
        const float base_theta = last_update_result.ok
            ? std::atan2(last_update_result.robot_pose.linear()(1, 0), last_update_result.robot_pose.linear()(0, 0))
            : state[4];

        const auto predicted_mean_from_prior = [&base_pos, base_theta, &wrap_angle](const OdometryPrior &prior)
        {
            Eigen::Vector3f predicted = Eigen::Vector3f::Zero();
            predicted.head<2>() = base_pos + prior.delta_pose.head<2>();
            predicted[2] = wrap_angle(base_theta + prior.delta_pose[2]);
            return predicted;
        };

        const auto set_model_prediction = [this, &selection](const Eigen::Vector2f &pos,
                                                             float theta,
                                                             const Eigen::Matrix3f &precision)
        {
            selection.predicted_pos = pos;
            selection.predicted_theta = theta;
            selection.prediction_precision = precision;

            model_->robot_pos = pos;
            model_->robot_theta = theta;
            model_->set_prediction(pos, theta, precision);
        };

        // The command channel enters the prediction only when the config flag allows it
        // (RoomConcept.UseCommandVelocityPrior, see Params::use_command_velocity_prior). The prior
        // itself was computed above regardless — compute_odometry_prior() also advances
        // last_lidar_timestamp, and the cmd_* CSV columns stay honest about what the channel WOULD
        // have said. Gating here, not at the source, is what keeps those two facts true.
        const bool command_usable = params.use_command_velocity_prior
            and selection.command_prior.valid and selection.command_prior.fresh;

        if (command_usable
            && selection.measured_prior.valid && selection.measured_prior.fresh
            && last_update_result.ok)
        {
            const Eigen::Vector3f pred_cmd = predicted_mean_from_prior(selection.command_prior);
            const Eigen::Vector3f pred_meas = predicted_mean_from_prior(selection.measured_prior);
            const auto [fused_mean, fused_precision] = fuse_priors(
                pred_cmd, selection.command_prior.covariance_eigen,
                pred_meas, selection.measured_prior.covariance_eigen);

            selection.source = MotionPriorSource::Fused;
            selection.selected_prior.valid = true;
            selection.selected_prior.fresh = true;
            selection.selected_prior.delta_pose.head<2>() = fused_mean.head<2>() - base_pos;
            selection.selected_prior.delta_pose[2] = wrap_angle(fused_mean[2] - base_theta);
            selection.selected_prior.dt = std::max(selection.command_prior.dt, selection.measured_prior.dt);
            selection.selected_prior.covariance_eigen = fused_precision.inverse();
            set_model_prediction(fused_mean.head<2>(), fused_mean[2], fused_precision);
        }
        else if (selection.measured_prior.valid && selection.measured_prior.fresh && last_update_result.ok)
        {
            selection.source = MotionPriorSource::Measured;
            selection.selected_prior = selection.measured_prior;
            const Eigen::Vector3f predicted = predicted_mean_from_prior(selection.selected_prior);
            set_model_prediction(predicted.head<2>(), predicted[2], selection.selected_prior.covariance_eigen.inverse());
        }
        else if (command_usable && last_update_result.ok)
        {
            selection.source = MotionPriorSource::Command;
            selection.selected_prior = selection.command_prior;
            const Eigen::Vector3f predicted = predicted_mean_from_prior(selection.selected_prior);
            set_model_prediction(predicted.head<2>(), predicted[2], selection.selected_prior.covariance_eigen.inverse());
        }
        else
        {
            selection.source = MotionPriorSource::FallbackZero;
            selection.selected_prior.valid = true;
            selection.selected_prior.fresh = false;
            selection.selected_prior.is_measured = false;
            selection.selected_prior.delta_pose = Eigen::Vector3f::Zero();
            selection.selected_prior.dt = std::max(selection.command_prior.dt, selection.measured_prior.dt);
            selection.selected_prior.covariance_eigen = compute_motion_covariance(selection.selected_prior, false);
            set_model_prediction(selection.predicted_pos,
                                 selection.predicted_theta,
                                 selection.selected_prior.covariance_eigen.inverse());
        }

        // Preintegration: the strided window CHAINS intervals, so the selected prior needs one no
        // matter which branch produced it. Measured/Command carry the real interval through the
        // wholesale copy above (random covariance and scale Jacobians kept separate, which is what
        // chain() needs to transport). The Fused branch cannot: fuse_priors() has already combined
        // both channels' random and correlated parts into one matrix and they are not separable
        // afterwards, so the whole thing enters as a random part with zero scale Jacobians.
        // ⚠ Consequence, stated rather than hidden: across a STRIDED interval the fused branch then
        // treats the scale error of successive frames as independent draws and under-counts it (√M
        // instead of M). It only bites with window_stride_enabled (off by default), and the real cure
        // is the scale STATE, which removes the term from the covariance entirely — see the
        // SCALE-AS-A-STATE note in se2_preintegration.h.
        if (params.motion_preintegration and not selection.selected_prior.has_preint)
        {
            rc::preint::Interval synth;
            synth.delta   = selection.selected_prior.delta_pose;
            synth.cov     = selection.selected_prior.covariance_eigen;
            // ⚠ The TIMING is a property of the interval, not of either channel, so it must be carried
            // across from a real interval rather than invented. An earlier version set samples = 1 and
            // left duration_s at 0; chaining those then accumulated a count that meant nothing and a
            // duration that stayed zero for ever — preint_n read 1..8 while preint_T read 0 on 2254 of
            // 2404 rows. A diagnostic that disagrees with itself is worse than no diagnostic, because
            // the disagreement is only visible if someone cross-tabulates the two columns.
            if (selection.measured_prior.has_preint)
            {
                synth.samples    = selection.measured_prior.preint.samples;
                synth.duration_s = selection.measured_prior.preint.duration_s;
            }
            else if (selection.command_prior.has_preint)
            {
                synth.samples    = selection.command_prior.preint.samples;
                synth.duration_s = selection.command_prior.preint.duration_s;
            }
            else
            {
                synth.samples    = 1;
                synth.duration_s = selection.selected_prior.dt * 1e-3f;   // dt is milliseconds
            }
            selection.selected_prior.preint = synth;
            selection.selected_prior.has_preint = true;
        }

        // ── THE REST HYPOTHESIS, WHERE IT CAN ACTUALLY REACH THE POSE ────────────────────────────
        // Applied to the MEAN, here, at prior selection — not inside the solver and not inside the
        // preintegrator.
        //
        // ★ WHY HERE AND NOWHERE ELSE. The covariance-shaping form never touches the mean by design.
        // The zero-delta factor form does touch it, but factors are only evaluated when Gauss-Newton
        // RUNS, and it runs on ~1% of cycles: on the other 99% the published pose is the raw
        // prediction x_prev (+) Delta and nothing constrains it at all. Measured with the factor on:
        // the per-cycle step was 3.54 mm against 3.49 mm before — unchanged, because the factor never
        // saw those cycles. The prediction is what gets published, so the prediction is where a
        // hypothesis about the prediction has to act.
        // The preintegrator's invariant survives: it still never writes its own mean. This is the
        // prior SELECTOR combining two priors, which is what it already does for the command channel.
        //
        // ★ IT IS A MIXTURE RESPONSIBILITY, not a gate — the same device rc::img::responsibility uses
        // to decide whether an image sample is an inlier. Two hypotheses for this interval: at rest,
        // in which case Delta is pure odometry noise, N(Delta; 0, sigma_odom^2); or moving, for which
        // any displacement up to what the base can do in T is about equally likely, a uniform of width
        // 2*v_max*T. The published increment is Delta scaled by the posterior probability of MOVING.
        // sigma_odom is the odometry's OWN noise over the interval, d_rest^2 * T — a density times a
        // time, so this is rate-invariant like everything else here.
        // At rest Delta sits at ~1 sigma, the rest hypothesis is far more likely than a uniform, and
        // the increment collapses. At 0.25 m/s over 50 ms Delta is ~3.7 sigma, the rest hypothesis is
        // e^-6.8 down, and the increment passes through untouched. No threshold: the crossover falls
        // where the two densities meet and moves with the odometry's own noise.
        // A cycle the rest block does not scale used the increment whole; never carry the last cycle's gain.
        zupt_pred_gain_tr_ = zupt_pred_gain_ro_ = 1.f;
        if (params.zupt_on_prediction and selection.selected_prior.valid)
        {
            auto& d = selection.selected_prior.delta_pose;
            const float T = std::max(1e-3f, selection.selected_prior.dt * 1e-3f);
            const auto& nm = params.odom_preint_noise;
            const float L     = std::max(nm.zupt_lever_m, 1e-3f);
            float w_tr = 1.f, w_ro = 1.f;
            if (params.zupt_pred_learn_rest)
            {
                // ── IS THE BODY AT REST? A velocity state, per body-frame channel ─────────────────────
                // See rest_density_learner.h for the three versions that failed and why this one is built
                // as it is: a velocity Kalman filter per channel (forward, lateral, rotation) with smooth/
                // jump process noise, the odometry noise learnt from its own innovations (so slow motion
                // cannot inflate it), and P(moving) by Savage-Dickey on v = 0. Body frame, because a
                // differential base translates only along its heading: one channel carries the motion.
                if (not rest_chan_init_)
                {
                    rc::preint::RestMotionChannel::Params pr; pr.v0 = 1.15f;   // rotation: uniform on +-2 rad/s
                    rest_rot_.set_params(pr);
                    rest_chan_init_ = true;
                }
                // the configured |translation| density spread over two axes; only ever a one-cycle prior
                rest_fwd_.set_prior(nm.zupt_density_v / std::sqrt(2.f));
                rest_lat_.set_prior(nm.zupt_density_v / std::sqrt(2.f));
                rest_rot_.set_prior(nm.zupt_density_omega);
                const float c = std::cos(base_theta), s = std::sin(base_theta);
                const float bx =  c * d[0] + s * d[1];        // the increment in the body frame
                const float by = -s * d[0] + c * d[1];
                const float pf = rest_fwd_.step(bx, T), pl = rest_lat_.step(by, T), pr = rest_rot_.step(d[2], T);
                w_tr = 1.f - (1.f - pf) * (1.f - pl);
                w_ro = pr;
            }
            else
            {
                // LEGACY: the per-cycle mixture with the borrowed PreintZuptDensity* -- FAILED the registered
                // slow-approach test (2026-10-02); kept reproducible. Coupled, so a pivot is not read as rest.
                const float m_tr = d.head<2>().norm() + L * std::abs(d[2]);
                const float m_ro = std::abs(d[2]) + d.head<2>().norm() / L;
                w_tr = rc::preint::mixture_p_moving(m_tr, nm.zupt_density_v,     T, params.zupt_pred_v_max * T);
                w_ro = rc::preint::mixture_p_moving(m_ro, nm.zupt_density_omega, T, params.zupt_pred_w_max * T);
            }
            zupt_pred_gain_tr_ = w_tr; zupt_pred_gain_ro_ = w_ro;   // for the viewer / debug row
            d.head<2>() *= w_tr;
            d[2]        *= w_ro;
            // ★ AND RE-ISSUE THE PREDICTION FROM THE SCALED INCREMENT. set_model_prediction() above
            // already wrote base (+) UNSCALED Delta into the model, and that is the pose an early exit
            // publishes -- so until 2026-10-01 this block scaled a copy nothing downstream read for the
            // pose, and the rest hypothesis never reached the published mean it exists to act on.
            // Measured on the first live run with it on: the selected increment fell 30x (46 -> 1.6 um
            // per cycle) while the published pose still stepped 81 um per cycle and wandered 72 mm in
            // 11.5 min truly still. The fallback source predicts no motion, so there is nothing to redo.
            if (selection.source != MotionPriorSource::FallbackZero)
                set_model_prediction(base_pos + d.head<2>(), wrap_angle(base_theta + d[2]),
                                     selection.prediction_precision);
        }

        last_selected_prior_ = selection.selected_prior;
        {   // mirror for the viewer, same lock as the ingress debug
            std::lock_guard lock(motion_ingress_debug_mutex_);
            predictor_delta_ = selection.selected_prior.delta_pose;
        }
        last_motion_prior_source_ = selection.source;
        return selection;
    }

    RoomConcept::OdometryPrior RoomConcept::compute_odometry_prior(
             const std::vector<VelocityCommand>& velocity_history,
             const std::pair<std::vector<Eigen::Vector3f>, std::int64_t> &lidar)
    {
         OdometryPrior prior;
         prior.valid = false;
            prior.fresh = false;
            prior.is_measured = false;
         const auto &[points, lidar_timestamp] = lidar;

         if (last_lidar_timestamp == 0)
         {
             last_lidar_timestamp = lidar_timestamp;
             return prior;
         }

         // Calculate dt
         const auto dt = lidar_timestamp - last_lidar_timestamp;
         if (dt <= 0)
        {
             last_lidar_timestamp = lidar_timestamp;
             return prior;
         }
         prior.dt = dt;

        const auto has_fresh_command = [&velocity_history, this, lidar_timestamp]() -> bool
        {
            if (velocity_history.empty() || !last_update_result.ok)
                return false;

            for (size_t i = 0; i < velocity_history.size(); ++i)
            {
                const auto segment_start = velocity_history[i].effective_ts_ms();
                const auto segment_end = (i + 1 < velocity_history.size())
                    ? velocity_history[i + 1].effective_ts_ms()
                    : lidar_timestamp;
                if (segment_start <= 0 || segment_end <= segment_start)
                    continue;

                const auto effective_start = std::max(segment_start, last_lidar_timestamp);
                const auto effective_end = std::min(segment_end, lidar_timestamp);
                if (effective_end > effective_start)
                    return true;
            }
            return false;
        };
        prior.fresh = has_fresh_command();
        if (!velocity_history.empty())
            prior.velocity_cmd = velocity_history.back();

        rc::preint::Interval interval;
        if (!velocity_history.empty() && last_update_result.ok)
            prior.delta_pose = integrate_velocity_over_window(last_update_result.robot_pose,
                                                              velocity_history,
                                                        last_lidar_timestamp,
                                                        lidar_timestamp,
                                                        params.motion_preintegration ? &interval : nullptr);
        else
            // If no history or no valid previous pose, assume STATIONARY (Zero motion)
            // This protects us when sitting still!
            prior.delta_pose = Eigen::Vector3f::Zero();

        prior.valid = true; // ALWAYS valid now


        // Compute covariance — propagated over the interval's segments, or the legacy asserted
        // diagonal. See the same branch in compute_measured_odometry_prior().
        Eigen::Matrix3f cov_eigen;
        if (params.motion_preintegration and interval.samples > 0)
        {
            prior.preint = interval;
            prior.has_preint = true;
            cov_eigen = interval.covariance();
        }
        else
            cov_eigen = compute_motion_covariance(prior);
        prior.covariance_eigen = cov_eigen;

        last_lidar_timestamp = lidar_timestamp;
        return prior;
    }

    Eigen::Matrix3f RoomConcept::compute_motion_covariance(const OdometryPrior &odometry_prior,
                                                             bool is_measured_odometry)
    {
        // ★ The EMA learner that used to override these was DELETED on 2026-08-26. It was A/B'd and
        // rejected in 2026-08: it learned slip_k ~= 0.81 against a true 0.036 rad increment, collapsed
        // the measured prior's weight to 0.04 during fast turns so the prior fell back on the command
        // channel, took early exit during rotation from 91.6% to 68.5%, and was the only one of four
        // configurations to LOSE THE TRACK. The defect was the residual it learned from — a
        // post-optimisation window-pair residual carrying the optimiser's own correction — which is
        // code, not a simulation artefact. The batch estimator (MotionCalibEnabled) is the learner
        // that replaced it, and it learns from a different quantity.
        const float noise_trans = is_measured_odometry
                                ? params.odom_noise_trans * params.odom_noise_scale
                                : params.cmd_noise_trans;
        const float noise_rot   = is_measured_odometry ? params.odom_noise_rot   * params.odom_noise_scale : params.cmd_noise_rot;
        const float noise_base  = is_measured_odometry ? params.odom_noise_base  * params.odom_noise_scale : params.cmd_noise_base;

        // Apply learned bias: subtract systematic odometry drift from the effective delta.
        // This makes the motion factor mean correct; the covariance still covers residual noise.
        const Eigen::Vector3f effective_delta = odometry_prior.delta_pose;

        float motion_magnitude = std::sqrt(
            effective_delta[0] * effective_delta[0] +
            effective_delta[1] * effective_delta[1]
        );

        // Uncertainty grows with distance; when stationary use tight constraint
        float base_uncertainty;
        if (motion_magnitude < params.stationary_motion_threshold) {
            base_uncertainty = params.stationary_motion_threshold;
        } else {
            base_uncertainty = noise_base;
        }

        float position_std = base_uncertainty + noise_trans * motion_magnitude;
        float rotation_std = params.rotation_noise_base + noise_rot * std::abs(effective_delta[2]);

        // Rotation-position coupling: rotation creates position uncertainty
        // (pivot wobble, wheel slip, lever arm effects)
        float rot_induced_pos = params.rotation_position_coupling * std::abs(effective_delta[2]);
        position_std = std::sqrt(position_std * position_std + rot_induced_pos * rot_induced_pos);

        // Encoder angular slip model (measured odometry only). The static param is authoritative:
        // the EMA path that used to override it here is deleted, and the batch estimator corrects the
        // MEAN through k_omega rather than re-writing this covariance term.
        const float effective_slip_k = params.encoder_rot_slip_k;
        if (is_measured_odometry && effective_slip_k > 0.f)
        {
            // UNITS (fixed 08-09). rotation_std is the std of the rotation INCREMENT over this
            // window, in rad — see its two other terms, rotation_noise_base (rad) and
            // odom_noise_rot * |delta_theta| (a fraction of an angle). This term used to be
            // k * (|delta_theta| / dt), i.e. k * ANGULAR SPEED in rad/s, folded in as if it were
            // rad. That inflates sigma by 1/dt ~ 16x (variance ~260x) at the 62 ms update interval:
            // measured 08-09 at omega=1.09 rad/s it claimed sigma = 0.055 rad against a true
            // increment of 0.068 rad — 81% of the whole rotation as one sigma, for an encoder whose
            // demonstrated error is ~7%. The measured prior was therefore drowned out at ANY k (the
            // fusion gave it weight 0.004 at k=0.25 and still only 0.05 at k=0.05), the motion prior
            // collapsed onto the COMMAND channel, and the command under-reports rotation by 25% — so
            // the SDF had to push heading further into the turn on 171/171 frames.
            //
            // A slip coefficient is a FRACTION of the rotation that slipped, so it multiplies the
            // increment, not the rate. This is k * omega * dt, written directly as k * |delta_theta|
            // — dimensionless k, and now invariant to the update rate (the old form made the right k
            // depend on how fast the localizer happened to be running).
            const float slip_std = effective_slip_k * std::abs(effective_delta[2]);
            rotation_std = std::sqrt(rotation_std * rotation_std + slip_std * slip_std);
        }

        Eigen::Matrix3f cov = Eigen::Matrix3f::Identity();
        cov(0, 0) = position_std * position_std;
        cov(1, 1) = position_std * position_std;
        cov(2, 2) = rotation_std * rotation_std;

        return cov;
    }

    RoomConcept::OdometryPrior RoomConcept::compute_measured_odometry_prior(
             const std::vector<OdometryReading>& odometry_history,
             const std::pair<std::vector<Eigen::Vector3f>, std::int64_t> &lidar)
    {
        OdometryPrior prior;
        prior.valid = false;
        prior.fresh = false;
        prior.is_measured = true;
        const auto& [points, lidar_timestamp] = lidar;

        const int64_t prev_ts = last_update_result.timestamp_ms;
        if (prev_ts == 0 || odometry_history.empty() || !last_update_result.ok)
            return prior;
        const auto dt = lidar_timestamp - prev_ts;
        if (dt <= 0)
            return prior;
        prior.dt = static_cast<float>(dt);

        const auto has_fresh_measurement = [&odometry_history, prev_ts, lidar_timestamp]() -> bool
        {
            if (odometry_history.empty())
                return false;

            for (size_t i = 0; i < odometry_history.size(); ++i)
            {
                const auto segment_start = odometry_history[i].effective_ts_ms();
                const auto segment_end = (i + 1 < odometry_history.size())
                    ? odometry_history[i + 1].effective_ts_ms()
                    : lidar_timestamp;
                if (segment_start <= 0 || segment_end <= segment_start)
                    continue;

                const auto effective_start = std::max(segment_start, prev_ts);
                const auto effective_end = std::min(segment_end, lidar_timestamp);
                if (effective_end > effective_start)
                    return true;
            }
            return false;
        };
        prior.fresh = has_fresh_measurement();

        rc::preint::Interval interval;
        // The IMU history is the fast channel the heading comes from; the clock map puts the lidar
        // sweep bounds into the same clock the rates are measured in. Both are optional -- absent,
        // this degrades exactly to the previous wheel-only, wall-clock behaviour.
        std::vector<ImuReading> imu_history;
        if (run_ctx_.imu_buffer != nullptr)
            imu_history = run_ctx_.imu_buffer->get_snapshot<0>();
        prior.delta_pose = integrate_odometry_over_window(
            last_update_result.robot_pose,
            odometry_history,
            prev_ts,
            lidar_timestamp,
            params.motion_preintegration ? &interval : nullptr,
            imu_history.empty() ? nullptr : &imu_history,
            run_ctx_.sim_clock);

        prior.valid = true;

        // Covariance for this interval. Either PROPAGATED through the interval's samples
        // (se2_preintegration.h — full 3×3 with the cross terms a rotating heading actually produces)
        // or the legacy asserted diagonal. The mean above is identical in both cases.
        Eigen::Matrix3f cov_eigen;
        if (params.motion_preintegration and interval.samples > 0)
        {
            prior.preint = interval;
            prior.has_preint = true;
            cov_eigen = interval.covariance();

            // ── ONE-SHOT PROOF OF LIFE, on the first interval that carries real motion ──────────────
            // A config key that quietly fails to bind is a recurring failure mode here (HierPrecEeDthetaMin
            // was dead twice over; mask_trunc_frac was dead code) and a covariance change is invisible in
            // every outcome metric until it has already shifted the pose. So print both matrices side by
            // side, once, and let the log say which one is in force rather than the config file implying it.
            if (not preint_announced_ and interval.delta.head<2>().norm() > 0.005f)
            {
                preint_announced_ = true;
                const Eigen::Matrix3f legacy = compute_motion_covariance(prior, true);
                qInfo() << "[Preint] MotionPreintegration ACTIVE — motion covariance is PROPAGATED, not asserted.";
                qInfo().nospace()
                    << "  interval: " << interval.samples << " samples over "
                    << interval.duration_s * 1000.f << " ms, |dp|=" << interval.delta.head<2>().norm()
                    << " m, dth=" << interval.delta[2] << " rad";
                qInfo().nospace()
                    << "  sigma_x  " << std::sqrt(cov_eigen(0, 0)) << " (legacy " << std::sqrt(legacy(0, 0))
                    << ")   sigma_y " << std::sqrt(cov_eigen(1, 1)) << " (legacy " << std::sqrt(legacy(1, 1))
                    << ")   sigma_th " << std::sqrt(cov_eigen(2, 2)) << " (legacy " << std::sqrt(legacy(2, 2)) << ")";
                qInfo().nospace()
                    << "  cross terms the legacy diagonal cannot express: rho(x,y)="
                    << cov_eigen(0, 1) / std::sqrt(cov_eigen(0, 0) * cov_eigen(1, 1))
                    << " rho(y,th)=" << cov_eigen(1, 2) / std::sqrt(cov_eigen(1, 1) * cov_eigen(2, 2))
                    << " rho(x,th)=" << cov_eigen(0, 2) / std::sqrt(cov_eigen(0, 0) * cov_eigen(2, 2));
            }
        }
        else
            cov_eigen = compute_motion_covariance(prior, true);
        prior.covariance_eigen = cov_eigen;

        return prior;
    }

    std::pair<Eigen::Vector3f, Eigen::Matrix3f> RoomConcept::fuse_priors(
        const Eigen::Vector3f &pred_cmd, const Eigen::Matrix3f &cov_cmd,
        const Eigen::Vector3f &pred_odom, const Eigen::Matrix3f &cov_odom)
    {
        // Bayesian fusion of two Gaussian priors:
        //   Σ_fused⁻¹ = Σ_cmd⁻¹ + Σ_odom⁻¹
        //   μ_fused = Σ_fused * (Σ_cmd⁻¹ * μ_cmd + Σ_odom⁻¹ * μ_odom)
        constexpr float reg = 1e-6f;
        const Eigen::Matrix3f prec_cmd = (cov_cmd + reg * Eigen::Matrix3f::Identity()).inverse();
        const Eigen::Matrix3f prec_odom = (cov_odom + reg * Eigen::Matrix3f::Identity()).inverse();

        const Eigen::Matrix3f fused_precision = prec_cmd + prec_odom;
        const Eigen::Matrix3f fused_cov = fused_precision.inverse();

        // Handle angle wrapping: compute angular difference carefully
        Eigen::Vector3f pred_odom_adj = pred_odom;
        float angle_diff = pred_odom[2] - pred_cmd[2];
        while (angle_diff > M_PI) angle_diff -= 2.0f * M_PI;
        while (angle_diff < -M_PI) angle_diff += 2.0f * M_PI;
        pred_odom_adj[2] = pred_cmd[2] + angle_diff;  // Unwrap relative to cmd

        const Eigen::Vector3f fused_mean = fused_cov * (prec_cmd * pred_cmd + prec_odom * pred_odom_adj);

        // Normalize fused angle
        float theta = fused_mean[2];
        while (theta > M_PI) theta -= 2.0f * M_PI;
        while (theta < -M_PI) theta += 2.0f * M_PI;

        Eigen::Vector3f result = fused_mean;
        result[2] = theta;

        return {result, fused_precision};
    }

} // namespace rc
