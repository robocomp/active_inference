// RoomConcept, split out of room_concept.cpp (refactor step 2, docs/refactor_proposal_2026-10-13.md).
// Prediction early exit and the innovation monitor.

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

    void RoomConcept::observe_innovation(UpdateResult &res)
    {
        if (not params.motion_noise_innov) return;
        // The belief state, not a threshold on the data: while searching / relocalising the scan-only pose
        // is not of the room the odometry is moving through.
        const bool scan_ok = innov_ok_ and not searching() and not grid_search_active_.load(std::memory_order_relaxed);
        auto &np = params.odom_preint_noise;
        const bool fed = scan_ok and cyc_U_ok_ and innov_prev_ok_ and innov_prev_cycle_ + 1 == innov_cycle_
                         and np.motion_proportional;
        // ── THE ODOMETRY STEP, LAID OUT AT THE SCAN-ONLY HEADING (2026-10-08, fix B) ─────────────────────────
        // cyc_odom_ is the predictor's step in WORLD axes, rotated with the PREDICTOR's heading. While moving that
        // heading ran +0.1..+0.3 deg ahead of the scan-only heading (published heading vs GT +0.29 deg, scan-only
        // +0.07 deg; run 2026-10-08 18-03), and compared against scan-only poses the difference appeared as a
        // lateral component eps * d_fwd -- read by the kinematic mount factor as a FALSE helios yaw (+0.56 deg in
        // a null run). Recover the body increment with the predictor's own mid-cycle heading, then lay it out at
        // the scan-only mid-cycle heading, so z_n - z_{n-1} and the step live in the same frame.
        Eigen::Vector3f odom_w = cyc_odom_;
        {
            const float th_p = res.pred_theta - 0.5f * cyc_odom_[2];                 // predictor, mid-cycle
            const float th_z = innov_z_[2]   - 0.5f * cyc_odom_[2];                  // scan-only, mid-cycle
            const float sp = std::sin(th_p), cp = std::cos(th_p), sz = std::sin(th_z), cz = std::cos(th_z);
            const float b_fwd = -sp * cyc_odom_[0] + cp * cyc_odom_[1];             // body forward = (-sin, cos)
            const float b_lat =  cp * cyc_odom_[0] + sp * cyc_odom_[1];             // body lateral = ( cos, sin)
            if (std::isfinite(res.pred_theta))
                odom_w = Eigen::Vector3f(-sz * b_fwd + cz * b_lat, cz * b_fwd + sz * b_lat, cyc_odom_[2]);
        }
        if (fed)
        {
            if (not noise_innov_init_)
            {   // priors = the configured coefficients, as loaded
                rc::preint::MotionNoiseInnov::Params vp;
                vp.memory = params.motion_noise_memory;
                vp.k0 = {np.k_long, np.k_lat, np.k_lat_turn, np.k_th_turn, np.k_t_trans, np.k_t_rot};
                noise_innov_.set_params(vp);
                noise_innov_init_ = true;
            }
            Eigen::Vector3f delta = innov_z_ - innov_z_prev_ - odom_w;
            delta[2] = std::remainder(delta[2], 2.f * float(M_PI));
            const float th = innov_z_[2] - 0.5f * cyc_odom_[2];
            const float sn = std::sin(th), cs = std::cos(th);
            Eigen::Matrix3f T;   // columns: body forward (-sin, cos), lateral (cos, sin), heading — in world
            T << -sn, cs, 0.f,
                  cs, sn, 0.f,
                 0.f, 0.f, 1.f;
            const Eigen::Vector3f odom_body(T.col(0).dot(odom_w), T.col(1).dot(odom_w), odom_w[2]);
            // P(the body moved this cycle): the rest channel's own belief (RestMotionChannel), either motion
            const float p_move = 1.f - (1.f - std::clamp(zupt_pred_gain_tr_, 0.f, 1.f)) * (1.f - std::clamp(zupt_pred_gain_ro_, 0.f, 1.f));
            noise_innov_.observe(delta, T, cyc_U_, innov_R_ + innov_R_prev_, odom_body, p_move, cyc_dt_s_);
            if (params.motion_noise_learn)
            {   // the coefficients the NEXT prediction is built with
                np.k_long     = float(noise_innov_.k(0)); np.k_lat     = float(noise_innov_.k(1));
                np.k_lat_turn = float(noise_innov_.k(2)); np.k_th_turn = float(noise_innov_.k(3));
                np.k_t_trans  = float(noise_innov_.k(4)); np.k_t_rot   = float(noise_innov_.k(5));
            }
            // ── the replay log: everything the estimator saw, so it can be re-fitted offline ──
            if (not innov_csv_.is_open())
            {
                std::filesystem::create_directories("tmp/noise_innov");
                char stamp[32]; const std::time_t now = std::time(nullptr);
                std::strftime(stamp, sizeof stamp, "%Y-%m-%d_%H-%M-%S", std::localtime(&now));
                innov_csv_.open(std::string("tmp/noise_innov/innov_") + stamp + ".csv");
                innov_csv_.imbue(std::locale::classic());
                innov_csv_ << "ts_ms,d_fwd,d_lat,d_th,odom_fwd,odom_lat,odom_th,searching";
                for (const char *ax : {"fwd", "lat", "th"})
                {
                    for (int j = 0; j < rc::preint::MotionNoiseInnov::NP; ++j) innov_csv_ << ",x" << j << '_' << ax;
                    innov_csv_ << ",y_" << ax << ",m_" << ax;
                }
                innov_csv_ << ",k_long,k_lat,k_lat_turn,k_th_turn,k_t_trans,k_t_rot,s_fwd,s_lat,s_th,q_fwd,q_lat,q_th"
                              ",z_x,z_y,z_th,r_out_fwd,r_out_lat,r_out_th,p_move,dt_s,mv_fwd,mv_lat,mv_th,z_iters\n";
            }
            innov_csv_ << res.timestamp_ms << ',' << T.col(0).dot(delta) << ',' << T.col(1).dot(delta) << ',' << delta[2]
                       << ',' << odom_body[0] << ',' << odom_body[1] << ',' << odom_body[2] << ",0";
            for (int a = 0; a < 3; ++a)
            {
                const auto &x = noise_innov_.last_x(a);
                for (int j = 0; j < rc::preint::MotionNoiseInnov::NP; ++j) innov_csv_ << ',' << x(j);
                innov_csv_ << ',' << noise_innov_.last_y(a) << ',' << noise_innov_.last_m(a);
            }
            for (int j = 0; j < 6; ++j) innov_csv_ << ',' << noise_innov_.k(j);
            for (int a = 0; a < 3; ++a) innov_csv_ << ',' << noise_innov_.s(a);
            for (int a = 0; a < 3; ++a) innov_csv_ << ',' << noise_innov_.q(a);
            // the scan-only pose itself (the bias-field structure function is built from z over path length)
            // and each axis's outlier responsibility
            innov_csv_ << ',' << innov_z_[0] << ',' << innov_z_[1] << ',' << innov_z_[2];
            for (int a = 0; a < 3; ++a) innov_csv_ << ',' << noise_innov_.last_r(a);
            innov_csv_ << ',' << p_move << ',' << cyc_dt_s_;
            for (int a = 0; a < 3; ++a) innov_csv_ << ',' << noise_innov_.mv(a);
            innov_csv_ << ',' << last_scan_iters_;
            innov_csv_ << '\n';
        }
        // ── the slow POSE-FIELD bias (pose_field_bias.h): learnt along unbroken chains of scan-only poses, and
        //    its covariance added to the PUBLISHED pose when asked (the solver's own covariance never sees it) ──
        if (params.pose_field_bias and scan_ok)
        {
            if (not field_bias_init_)
            {
                rc::PoseFieldBias::Params fp; fp.sigma0 = params.pose_field_sigma0;
                field_bias_.set_params(fp);
                field_bias_init_ = true;
            }
            // Only the part that ACCUMULATES along a path (k_j, scale), not the while-moving step: its source is
            // ambiguous and, summed as odometry, it swallowed the whole bias signal live (2026-10-07 11-14: sigma_b
            // 0 for the entire tour; replay without it: 47 mm, heading 18.5x -> 3.8x). Without it the estimate can
            // only err LARGE (22-20 replay: 71 vs ~45 mm needed), never collapse to zero.
            const float odo_var_tr = fed ? float(noise_innov_.odom_var_accum(0) + noise_innov_.odom_var_accum(1)) : 0.f;
            field_bias_.observe(innov_z_, innov_R_, odom_w, odo_var_tr, 0.f, fed);
            const Eigen::Matrix3f Sb = field_bias_.covariance(innov_R_);
            res.surprise.pf_sigma_b = float(field_bias_.sigma_b());          // global (log, outlier scale)
            res.surprise.pf_sigma_b_loc = float(field_bias_.sigma_b_local()); // at this place: what Sb carries
            res.surprise.pf_pi_out  = float(field_bias_.pi_elsewhere());
            res.surprise.pf_len     = float(field_bias_.length());
            res.surprise.pf_xx      = 0.5f * (Sb(0, 0) + Sb(1, 1));
            res.surprise.pf_tt      = Sb(2, 2);
            if (params.pose_field_publish and Sb.allFinite()) res.covariance += Sb;
        }
        innov_prev_ok_ = scan_ok;
        if (scan_ok) { innov_z_prev_ = innov_z_; innov_R_prev_ = innov_R_; innov_prev_cycle_ = innov_cycle_; }
    }

    std::optional<RoomConcept::UpdateResult> RoomConcept::try_prediction_early_exit(
        const ScanPoints& scan_pts,
        const Eigen::Vector3f& slot_odom_delta,
        const OdometryPrior& odometry_prior,
        std::int64_t lidar_timestamp_ms,
        bool measure_only)
    {
        // No fast_rotation block here: the SDF quality check below already decides whether
        // the predicted pose (including theta) is accurate enough to skip Adam.
        // If measured odometry is available and accurate, the theta prediction will be good,
        // mean_sdf_pred will be low, and early exit will fire correctly — even during turns.
        // If the prediction is poor (large rotation error), mean_sdf_pred will be high and
        // Adam will run.  An explicit angular-velocity gate was previously needed because
        // slot_odom_delta used command velocity (inconsistent with the fused prediction).
        // Now that slot_odom_delta uses measured odometry, the prediction is self-consistent
        // and the SDF gate alone is sufficient.
        // Fresh each frame: NaN unless we actually reach the SDF evaluation below. This lets update()
        // report NaN (⇒ not plotted) for frames where the gate never ran, rather than a stale value.
        last_early_exit_metric_ = std::numeric_limits<float>::quiet_NaN();
        last_pred_sdf_median_   = std::numeric_limits<float>::quiet_NaN();
        sdf_polished_this_cycle_ = false;
        // ── The accumulator resets HERE, in the function that owns it ─────────────────────────────
        // Returning nullopt from anywhere in this function means the caller is about to optimise, so
        // the pose is about to be re-determined and "uncertainty accumulated since the last solve"
        // starts again at zero.
        // ★ It lives here because the two previous attempts to reset it elsewhere both went wrong:
        // once seeded from current_covariance, whose xy block is still the 1.0 placeholder, and once
        // anchored on `res.covariance = current_covariance` — a line that appears THREE times, so it
        // landed in the manual-reset settle branch and never ran at all. Both produced 0% early exit,
        // and neither was visible from the flag that switched the feature on. State that belongs to
        // one decision belongs in the function that makes it.
        if (forced_solve_last_cycle_) { unopt_pos_var_ = 0.f; forced_solve_last_cycle_ = false; }
        // WHICH condition sent this frame to the optimiser. Four call sites returned the same
        // nullopt, so "the optimiser ran" was all the record there was — and a gate cannot be shown
        // to earn its cost, or to be redundant, from a number that does not distinguish it from the
        // other three. ee_* are reset by the periodic report.
        const auto give_up_to_optimizer = [this](EeReason why) -> std::optional<UpdateResult>
        { forced_solve_last_cycle_ = true; ++ee_forced_[static_cast<int>(why)]; return std::nullopt; };
        // ── THE PREDICTION STEP OF THE POSE COVARIANCE ────────────────────────────────────────────
        // ★ current_covariance was ONLY ever assigned on the optimized path, and the optimizer runs on
        // ~0.2% of cycles. So on every other cycle it carried whatever the last solve left, or its
        // Identity*0.1 initialiser, and two consumers were reading that placeholder as a measurement:
        // the calibrator's episode weight (measured at pos_var = 1.000000 m^2 exactly — sigma of one
        // METRE — which made 2994 episodes weigh 1 while 6 optimized ones weighed 1269, so 9408
        // episodes or ~2.4 km of driving would have been needed to reach `informed`), and now the
        // polish's own regulariser.
        // A pose that runs open-loop gets LESS certain with every cycle, and saying so is the whole
        // point: this is the ordinary Kalman prediction step, with the motion prior's covariance as
        // the process noise it already computes for the motion factor.
        // ⚠ GATED ON THE POLISH 2026-08-26, after this was left running with the polish OFF and the
        // TRACK WAS LOST: sdf_mse 0.026 -> 0.4611, early exit down to 21%, cov_tt peaking at 7.1 rad^2.
        // The SHRINK half of this recursion lives inside the polish block, so with the polish
        // disabled only the GROWTH half ran — and current_covariance is not a private diagnostic, the
        // early-exit path publishes it as res.covariance. An inflating covariance loosens everything
        // that depends on it and the window went with it.
        // A one-sided recursion is not a recursion: growth and shrink ship together or not at all.
        if (params.sdf_polish_enabled
            and odometry_prior.valid and odometry_prior.covariance_eigen.allFinite())
        {
            const Eigen::Matrix3f P_grown = current_covariance + odometry_prior.covariance_eigen;
            if (P_grown.allFinite() and not measure_only) current_covariance = P_grown;   // measure_only: no state change
        }

        if (!params.prediction_early_exit ||
            !last_update_result.ok ||
            !odometry_prior.valid ||
            tracking_step_count_ <= params.min_tracking_steps)
            return give_up_to_optimizer(EeReason::NotTracking);

        const auto& newest = window_mgr_.newest();
        const Eigen::Vector3f pose0 = newest.pose;   // the PREDICTION; a polish below replaces newest.pose
        const Eigen::VectorXf sdf_pred = model_->sdf_at_pose(scan_pts, pose0);
        // ── THE GATE'S OWN STATISTIC GETS THE DOOR WEIGHTS TOO ──────────────────────────────────
        // This is a MEAN, and a mean is what outliers own. The reported sdf_mse is a median and was
        // already weighted, so it fell to 7 mm the moment the door filter came on — while this number
        // stayed at 0.2 and the gate never opened, giving zero early exits with the robot standing
        // still. A few hundred returns three metres beyond an open doorway do that to a mean and
        // barely touch a median; weighting one and not the other made the two disagree about the same
        // scan. Same weights, same points, both statistics.
        // sum(w*|sdf|) / sum(w): with no open door the weights are undefined and this is exactly the
        // plain mean it replaces.
        float mean_sdf_pred = sdf_pred.size() > 0 ? sdf_pred.cwiseAbs().mean()
                                                  : std::numeric_limits<float>::quiet_NaN();
        if (const auto dw = door_point_weights(scan_pts); dw.size() > 0 and dw.size() == sdf_pred.size())
        {
            const Eigen::VectorXf a = sdf_pred.cwiseAbs();
            const Eigen::VectorXf& w = dw;
            const float wsum = w.sum();
            if (wsum > 1e-6f)
            {
                // ⚠ BOTH VALUES ARE KEPT, because this is a DECISION variable and a gate cannot be
                // audited by the statistic it already truncated (the assoc_chi2 lesson). If the
                // aperture geometry or p_open is ever wrong, the discount would blind the gate in
                // that direction and a genuinely bad pose would pass as good; the only way to see
                // that is to have both numbers side by side. ee_metric_unweighted_ is recorded
                // beside the one the gate uses, so "would it have decided differently?" is answered
                // from the log rather than argued.
                ee_metric_unweighted_ = mean_sdf_pred;
                mean_sdf_pred = a.cwiseProduct(w).sum() / wsum;
                ee_door_discount_ = 1.f - wsum / static_cast<float>(w.size());
            }
        }
        else { ee_metric_unweighted_ = mean_sdf_pred; ee_door_discount_ = 0.f; }
        // Record the decision variable whether or not we early-exit — on an Adam frame this is the
        // value that TRIGGERED optimization (it exceeded the trust threshold). update() reads it back.
        last_early_exit_metric_ = mean_sdf_pred;
        // Same points, same pose, median instead of mean — see last_pred_sdf_median_. ★ AND THE SAME DOOR WEIGHTS
        // (Task R1b): this median was the one statistic left unweighted, and it is half of the lost belief's s
        // (max(sdf_mse, pred_sdf_median)). Live rtD1g2 recall 1: pred 2.52–2.71 m for 600 ms after the hand-over
        // while the door-weighted post-fit median read 0.08–0.11 m — the returns seen back through the door just
        // used, counted against the room the robot had entered. P(lost) 0.501 and a search followed.
        last_pred_sdf_median_ = weighted_median_abs_sdf(sdf_pred, door_point_weights(scan_pts));
        // MEASURE ONLY: the diagnostics above are the point; the decision below is not ours to take while
        // the layout is still being estimated (there is no fixed map to trust the prediction against).
        if (measure_only) return std::nullopt;
        // The scan-only pose AT THE PREDICTION, for the noise learner (motion_noise_innov.h). Taken
        // before either decision below, so it exists on early-exit and solve cycles alike; a solve
        // re-takes it at the converged pose.
        if (params.motion_noise_innov)
            scan_only_pose(scan_pts, pose0.x(), pose0.y(), pose0.z());

        // Widen the SDF trust threshold when the robot is rotating.
        // A theta error ε at room scale R produces SDF displacement ~R*ε.
        // The base threshold (sigma_sdf * trust_factor ≈ 7.5 cm) is too tight during rotation:
        // even a 0.02 rad odometry error at 5 m gives ~10 cm — larger than the base threshold.
        // We add rotation_sdf_coupling * |delta_theta| to compensate for this geometric effect.
        const float rot_boost = params.rotation_sdf_coupling * std::abs(odometry_prior.delta_pose[2]);
        const float prediction_trust_threshold = params.sigma_sdf * params.prediction_trust_factor + rot_boost;
        if (mean_sdf_pred >= prediction_trust_threshold)
            return give_up_to_optimizer(EeReason::SdfResidual);

        // ── AND THE OTHER HALF OF THE STOPPING CRITERION: HOW UNCERTAIN HAS THE POSE BECOME? ──────
        // The residual test above asks "is the fit good?". It cannot ask "could the pose be wrong in
        // a direction this residual does not see?" — and a scalar mean |SDF| genuinely cannot see
        // one. Sliding along a wall barely changes that wall's point distances; the constraint comes
        // from the perpendicular geometry, so drift down an under-constrained direction is nearly
        // invisible to it. MEASURED: parked with ground truth motionless for 3963 cycles, the pose
        // wandered 210 mm in x and 193 in y while this gate passed on all but 12 of them. The
        // criterion was not wrong about the residual; it was answering a different question.
        //
        // So accumulate how much the pose COULD have moved since the last real solve — the motion
        // prior's own covariance, summed — and verify once that reaches the same distance the
        // residual test already tolerates. Past that point "the residual is small" stops being
        // evidence the pose is right, because the pose is no longer determined to that accuracy.
        //
        // ★ NO NEW CONSTANT. The bound IS prediction_trust_threshold, the tolerance already in force
        // one line above: once the pose's own uncertainty reaches what the gate is willing to forgive
        // in residual, forgiving it further is unfounded. The two halves therefore move together —
        // widen the gate for rotation and this widens with it.
        // ★ NOT PUBLISHED AND NOT FED TO THE SOLVER. It is a decision variable only, which is the
        // difference from the covariance recursion reverted earlier today: that one inflated a value
        // consumers read, and it lost the track. This cannot reach anything but this `if`.
        // ★ THE INCREMENT MUST BE AN INCREMENT. covariance_eigen is the PRIOR's covariance, and it
        // carries a fixed floor — StationaryMotionThreshold, 2 cm, which exists so a parked robot's
        // motion factor cannot claim absurd precision. Accumulating that treats a FLOOR as a
        // per-cycle growth: nine cycles of a 2 cm floor "reach" 6 cm without the robot having moved
        // at all, and if the floor is larger it reaches it in one. Measured: 0% early exit with the
        // residual sitting at 0.007, far inside its own threshold — the gate was being forced by an
        // accumulator counting a constant.
        // The preintegrated interval covariance IS the increment, by construction: it is what the
        // sensors say accumulated over THIS interval, with no floor in it. If preintegration is off
        // there is no honest increment available, so nothing accumulates and this half of the
        // criterion simply does not participate.
        if (odometry_prior.valid and odometry_prior.has_preint)
        {
            const Eigen::Matrix3f dP = odometry_prior.preint.covariance();
            if (dP.allFinite())
            {
                unopt_pos_var_ += 0.5f * (dP(0, 0) + dP(1, 1));
                if (std::sqrt(std::max(0.f, unopt_pos_var_)) >= prediction_trust_threshold)
                    return give_up_to_optimizer(EeReason::UnoptDrift);   // uncertain enough that the map should be consulted
            }
        }

        // ── Corner-consistency gate ─────────────────────────────────────────────────────────────
        // The SDF gate above cannot see a 180° flip (a rot180 pose is SDF-ambiguous), so it would let a
        // flipped-but-SDF-good prediction early-exit and never give the corners a chance to veto it. Here
        // we validate the SAME predicted pose against the corner factor: if the worst corner's whitened
        // residual m=√(rᵀΛ_det r) exceeds the tolerance, the prediction is inconsistent with the corners
        // (the flip signature) → reject the early-exit so Adam runs and the corner factor pulls it back.
        if (params.corner_early_exit_check and corner_veto_fires())   // body: corner_veto_fires()
            return give_up_to_optimizer(EeReason::CornerVeto);

        // NOTE: the object-anchor early-exit GATE was reverted (2026-07-12) to A/B whether it was the
        // source of the localization instability. The anchor is now FACTOR-ONLY: it refines the pose
        // inside SDF-triggered optimizations but no longer FORCES the optimizer to run on its own.
        // (object_anchor_early_exit_sigma is left in the config but unused while this is out.)

        prediction_early_exits_++;
        ++ee_taken_;
        // Report the split, not just the rate. "99% early exit" is the same number whether the SDF
        // is doing all the work or the corner veto is quietly forcing a third of the solves.
        report_early_exit_split();
        last_t_adam_ms_ = 0.f;
        last_t_cov_ms_ = 0.f;
        last_t_breakdown_ms_ = 0.f;

        // ── ONE DAMPED SDF STEP, on the cycle that is about to publish without the optimizer ──────
        // This is where the drift lives. The gate's verdict is "good enough not to need a full
        // solve", and it has been implemented as "do nothing" — so on ~99% of cycles the published
        // pose is dead reckoning with no absolute reference at all, and it random-walks: measured
        // 3.49 mm per cycle, 210 mm over 3963 parked cycles with ground truth motionless.
        //
        // ★ SHRINKING THE STEP CANNOT FIX THIS, which is why the two earlier attempts did not. A
        // random walk with a smaller step is still a random walk: halve it and the drift only grows
        // more slowly, still without bound, still as sqrt(N). What removes drift is an ABSOLUTE
        // reference applied every cycle, which turns the walk into a mean-reverting process. The map
        // is that reference and it is already being evaluated here for the gate.
        //
        // ★ HOW FAR IT MAY MOVE IS SET BY THE MOTION PRIOR, not by a step size. The steepest-descent
        // step for a scalar residual is exactly L/|g|^2 — no constant — and it is then scaled to lie
        // inside the prior's own 1-sigma ellipsoid: delta' Lambda delta <= 1. So the correction may
        // go anywhere the prior considers a plausible amount of motion for this interval and no
        // further. It cannot overrule a confident prediction, and it can undo a wander the prior
        // admits could not have happened. Nothing here is tuned; both bounds come from quantities the
        // estimator already states.
        sdf_polish_step(scan_pts, pose0, odometry_prior);

        const Eigen::Vector3f p_acc = window_mgr_.newest().pose;
        const float x = p_acc[0], y = p_acc[1], phi = p_acc[2];

        UpdateResult res = build_early_exit_result(scan_pts, sdf_pred, mean_sdf_pred, lidar_timestamp_ms,
                                                   x, y, phi);

        write_early_exit_debug_row(res, odometry_prior, lidar_timestamp_ms, mean_sdf_pred, x, y, phi);


        return res;
    }

    bool RoomConcept::corner_veto_fires() const
    {
            const auto& slot = window_mgr_.newest();
            if (!slot.corner_obs.empty())
            {
                const float px = slot.pose[0], py = slot.pose[1], pth = slot.pose[2];
                const float ct = std::cos(pth), st = std::sin(pth);
                int n_bad = 0;
                for (const auto& obs : slot.corner_obs)
                {
                    const float dx = obs.model_corner_world.x() - px;
                    const float dy = obs.model_corner_world.y() - py;
                    // Predicted observation z_hat = R(-θ)·(c_world - t), residual = detected - z_hat.
                    const float rx = obs.detected_robot.x() - ( ct * dx + st * dy);
                    const float ry = obs.detected_robot.y() - (-st * dx + ct * dy);
                    const Eigen::Vector2f r(rx, ry);
                    const float maha_sq = params.corner_precision_gain * r.dot(obs.information * r);
                    if (std::sqrt(std::max(0.f, maha_sq)) > params.corner_early_exit_sigma)
                        ++n_bad;
                }
                // CONSENSUS: force Adam only when enough corners disagree — a lone outlier is outvoted by
                // the corroborating majority, but a rot180 flip (ALL corners disagree) still trips it.
                return n_bad >= params.corner_early_exit_min_bad;
            }
        return false;
    }

    void RoomConcept::report_early_exit_split()
    {
        {
            const std::int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                                         std::chrono::steady_clock::now().time_since_epoch()).count();
            if (ee_report_last_ms_ == 0) ee_report_last_ms_ = now;
            else if (now - ee_report_last_ms_ > 10000)
            {
                ee_report_last_ms_ = now;
                const int f = ee_forced_[0] + ee_forced_[1] + ee_forced_[2] + ee_forced_[3];
                qInfo().nospace()
                    << "[earlyexit] taken " << ee_taken_ << " / forced " << f
                    << "  (not_tracking " << ee_forced_[0] << ", sdf_residual " << ee_forced_[1]
                    << ", unopt_drift " << ee_forced_[2] << ", CORNER_VETO " << ee_forced_[3] << ")";
                ee_taken_ = 0; ee_forced_ = {};
            }
        }
    }

    void RoomConcept::sdf_polish_step(const ScanPoints& scan_pts, const Eigen::Vector3f& pose0,
                                      const OdometryPrior& odometry_prior)
    {
        const auto t_polish0 = std::chrono::steady_clock::now();   // per-stage timing (optimizer_timing.csv)
        if (params.sdf_polish_enabled and odometry_prior.valid)
        {
            // ★ A GAUSS-NEWTON STEP, not steepest descent. The scalar form — step L/|g|^2 on the
            // MEAN residual — was tried first and left a shiver: it is the step that would zero a
            // LINEAR residual in one go, the residual is not linear, so it over-shoots and corrects
            // back every cycle. Measured, that showed as a growth exponent of 0.167 instead of 0 and
            // as heading barely improving at all, because translation dominates the mean's gradient
            // and the yaw direction is averaged away inside a single scalar.
            // Using the per-point Jacobian fixes both: yaw gets its own column instead of a share of
            // one number, and the curvature the scalar form ignores is what stops the over-shoot.
            // ★ This is ONE ITERATION of the optimizer's own SdfFactor, against the newest slot only
            // — same query, same observation weights, same IRLS Huber, same J = [gx, gy, -gx*qy +
            // gy*qx]. Sharing room_obs_weights.h is the point: a polish weighting its points
            // differently from the solver would be a second estimator that can disagree with the
            // first, which is the shape of bug this codebase has paid for before.
            // ★ AND THE PRIOR IS THE DAMPING. delta = -(H_sdf + Lambda_prior)^-1 b. At the prediction
            // the prior's own residual is zero, so it contributes to H alone and acts as a
            // Levenberg term with physical units: the step is limited by how much motion the prior
            // says was plausible for this interval. No lambda to tune, and the units are metres and
            // radians rather than a bare number.
            {
                Eigen::Matrix3f H;
                Eigen::Vector3f bb;
                if (scan_linearize(scan_pts, pose0.x(), pose0.y(), pose0.z(), H, bb))
                {
                    // ★ THE REGULARISER IS THE POSE'S OWN ACCUMULATED UNCERTAINTY, not one
                    // interval's motion prior. This step corrects error built up over HUNDREDS of
                    // free-running cycles, so "how far could the robot have moved in the last 50 ms"
                    // is the wrong question — "how uncertain has this pose become since the map last
                    // constrained it" is the right one.
                    // Measured with the interval prior here: parked, the ZUPT tightens it to ~4.7 mm
                    // so Lambda ~ 45000 against H_sdf ~ 20 — two thousand times stiffer, the step is
                    // crushed to nothing, and the growth exponent went back to 0.522, a pure random
                    // walk. The ZUPT's covariance-tightening was suppressing the very correction that
                    // fixes what it makes the estimator confidently wrong about.
                    const Eigen::Matrix3f Lam = current_covariance.inverse();
                    const Eigen::Matrix3f A = H + Lam;
                    Eigen::Vector3f d = -A.ldlt().solve(bb);
                    if (d.allFinite())
                    {
                        window_mgr_.newest().pose = pose0 + d;
                        last_sdf_polish_mm_ = d.head<2>().norm() * 1000.f;
                        sdf_polished_this_cycle_ = true;
                        // ── AND THE POSTERIOR COVARIANCE OF THE STEP WE JUST TOOK ────────────────
                        // A = H_sdf + Lambda is the information after this correction, so A^-1 is
                        // the covariance. Free: it is already factorised for the solve above.
                        const Eigen::Matrix3f P_post = A.inverse();
                        if (P_post.allFinite() and P_post(0, 0) > 0.f and P_post(2, 2) > 0.f)
                            current_covariance = P_post;
                    }
                }
            }
        }
        last_t_polish_ms_ = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t_polish0).count();
    }

    RoomConcept::UpdateResult RoomConcept::build_early_exit_result(const ScanPoints& scan_pts,
                const Eigen::VectorXf& sdf_pred, float mean_sdf_pred, std::int64_t lidar_timestamp_ms,
                float x, float y, float phi)
    {
        UpdateResult res;
        res.ok = true;
        res.final_loss = mean_sdf_pred;
        // sdf_mse is a MEDIAN absolute residual on every path (2026-08-13). It used to be set to
        // mean_sdf_pred here while the optimized path (compute_sdf_median_abs) set a median, so the
        // same field carried two different estimators of the same sigma — a ~15% step depending only
        // on which branch produced the frame, and >98% of frames come through this one. Consumers
        // (DSR.StableSdfMseMax, SymmetryGoodFitMse, EpistemicController.Sdf{Safe,Danger}, the two
        // Boundary*QualityThresholds and boundary_weight_now) were recalibrated by kMedianOverMeanAbs
        // when this changed. The mean is still what the early-exit GATE tests, and it is reported
        // unchanged as early_exit_metric below — that field's name says what it holds.
        res.sdf_mse = weighted_median_abs_sdf(sdf_pred, door_point_weights(scan_pts));
        res.pred_sdf_median = res.sdf_mse;   // on this path they are the same quantity
        res.early_exit_metric = mean_sdf_pred;   // the value that PASSED the threshold (optimizer skipped)
        res.early_exit_metric_unweighted = ee_metric_unweighted_;   // what it would have been with the door in
        res.door_discount = ee_door_discount_;
        res.sdf_polished = sdf_polished_this_cycle_;   // the calibrator counts this as a correction
        // (No res.covariance assignment here: this path already publishes current_covariance further
        //  down. Adding a second one was redundant — and it was how the growth step below reached
        //  every consumer.)
        // Heading attribution must be set on BOTH return paths. This one carries >98% of frames, and
        // it is the interesting one: between corrections the prediction runs open-loop, so these are
        // the only cycles where an accumulating channel error is visible before the optimizer hides it.
        res.pred_x = last_pred_pos_.x();
        res.slot_appended = params.window_stride_enabled and not rec_stride_replace_;
        res.pred_y = last_pred_pos_.y();
        res.pred_theta = last_pred_theta_;
        res.dx_local = cyc_dx_local_;
        res.dy_local = cyc_dy_local_;
        res.imu_dvx = cyc_imu_dvx_; res.imu_dvy = cyc_imu_dvy_;
        res.imu_dpx = cyc_imu_dpx_; res.imu_dpy = cyc_imu_dpy_;
        res.wheel_dvx = cyc_wheel_dvx_; res.wheel_dvy = cyc_wheel_dvy_;
        res.imu_lin_segs = cyc_imu_lin_segs_;
        res.imu_dtheta          = cyc_imu_dtheta_;
        res.wheel_dtheta        = cyc_wheel_dtheta_;
        res.heading_cov         = cyc_heading_cov_;
        res.gyro_weight         = cyc_heading_dt_ > 0.0
                                ? static_cast<float>(cyc_gyro_w_dt_ / cyc_heading_dt_) : -1.f;
        res.heading_diag        = cyc_hdiag_;
        res.heading_diag.nl_c0 = static_cast<float>(wheel_noise_.coeffs()[0]);
        res.heading_diag.nl_c1 = static_cast<float>(wheel_noise_.coeffs()[1]);
        res.heading_diag.nl_c2 = static_cast<float>(wheel_noise_.coeffs()[2]);
        res.heading_diag.nl_samples = wheel_noise_.samples();
        res.heading_diag.rest_gain_tr = params.zupt_on_prediction ? zupt_pred_gain_tr_ : -1.f;
        res.heading_diag.rest_gain_ro = params.zupt_on_prediction ? zupt_pred_gain_ro_ : -1.f;
        res.heading_diag.rest_dens_v  = rest_fwd_.density() * std::sqrt(2.f);   // as a |translation| density
        res.heading_diag.rest_dens_w  = rest_rot_.density();
        if (params.heading_fusion and cyc_hdiag_.total_dt > 0.f)
            res.heading_diag.dens_w = static_cast<float>(cyc_dens_w_dt_ / cyc_hdiag_.total_dt);
        if (params.heading_fusion and cyc_hdiag_.gyro_dt > 0.f)
            res.heading_diag.dens_g = static_cast<float>(cyc_dens_g_dt_ / cyc_hdiag_.gyro_dt);
        res.wheel_shadow_dtheta = cyc_wheel_shadow_dtheta_;
        res.imu_segs            = cyc_imu_segs_;
        res.wheel_segs          = cyc_wheel_segs_;
        // MUST come after the fields above: it reads dy_local/dx_local/imu_dtheta as the covariates
        // H. Called earlier it sees zeros, H -> 0, and the learner silently never learns anything.
        feed_motion_calibrator(res);
        res.iterations_used = 0;
        res.state << 2.f * model_->half_extents.x(), 2.f * model_->half_extents.y(), x, y, phi;

        Eigen::Affine2f pose_aff = Eigen::Affine2f::Identity();
        pose_aff.translation() = Eigen::Vector2f{x, y};
        pose_aff.linear() = Eigen::Rotation2Df(phi).toRotationMatrix();
        res.robot_pose = pose_aff;
        res.covariance = current_covariance;

        if (model_->has_prediction)
        {
            res.innovation[0] = x - model_->predicted_pos.x();
            res.innovation[1] = y - model_->predicted_pos.y();
            float p_theta = model_->predicted_theta;
            res.innovation[2] = phi - p_theta;
            while (res.innovation[2] > M_PI) res.innovation[2] -= 2.0f * M_PI;
            while (res.innovation[2] < -M_PI) res.innovation[2] += 2.0f * M_PI;
            res.innovation_norm = std::sqrt(res.innovation[0]*res.innovation[0] +
                                            res.innovation[1]*res.innovation[1]);
            apply_adaptive_covariance(res);
        }
        // Only a polished early exit moved the estimate; otherwise the posterior IS the prediction (KL 0).
        score_surprise(res, res.sdf_polished);
        res.timestamp_ms = lidar_timestamp_ms;   // before observe_innovation: its log row is keyed on it
        observe_innovation(res);

        model_->robot_pos = {x, y};
        model_->robot_theta = phi;
        model_->robot_prev_pose = res.robot_pose;
        model_->has_prediction = false;

        res.timestamp_ms = lidar_timestamp_ms;
        last_update_result = res;
        prev_sdf_mse_ = res.sdf_mse;   // track for boundary quality gate next frame
        last_lidar_timestamp = lidar_timestamp_ms;
        return res;
    }

    void RoomConcept::write_early_exit_debug_row(const UpdateResult& res, const OdometryPrior& odometry_prior,
                std::int64_t lidar_timestamp_ms, float mean_sdf_pred, float x, float y, float phi)
    {
        if (debug_log_.is_open())
        {
            const auto wall_now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            const float lr_eff = params.learning_rate_pos /
                std::sqrt(static_cast<float>(std::max(1, (int)window_mgr_.size())));
            float meas_cov_xx = 0.f, meas_cov_tt = 0.f;
            if (last_measured_prior_.valid)
            {
                meas_cov_xx = last_measured_prior_.covariance_eigen(0, 0);
                meas_cov_tt = last_measured_prior_.covariance_eigen(2, 2);
            }
            const float sel_cov_xx = last_selected_prior_.covariance_eigen(0, 0);
            const float sel_cov_tt = last_selected_prior_.covariance_eigen(2, 2);
            const auto motion_ingress = get_motion_ingress_debug();
            debug_log_
                << lidar_timestamp_ms
                << ',' << wall_now_ms
                << ',' << odometry_prior.dt
                << ',' << 0           // n_lidar not available here
                // vel_adv_y, vel_rot, odom_adv, odom_rot — these were hardcoded to 0 on this path,
                // so every early-exit frame looked STATIONARY in the log. Since early exit is ~98%
                // of frames, any "rate vs angular speed" question asked of this file silently saw
                // only the frames where the gate had already failed, and the four velocity columns
                // were unusable for classifying anything. The ingress copies below are the very
                // values the Adam path reports (record_*_ingress is handed the same normalized
                // numbers that go into velocity_history / odometry_history), so both paths now agree.
                << ',' << motion_ingress.command_adv_normalized
                << ',' << motion_ingress.command_rot_normalized
                << ',' << motion_ingress.odom_adv_normalized
                << ',' << motion_ingress.odom_rot_normalized
                << ',' << motion_ingress.command_source
                << ',' << motion_ingress.command_adv_raw
                << ',' << motion_ingress.command_adv_normalized
                << ',' << motion_ingress.command_rot_raw
                << ',' << motion_ingress.command_rot_normalized
                << ',' << motion_ingress.command_ts_ms
                << ',' << motion_ingress.odom_source
                << ',' << motion_ingress.odom_adv_raw
                << ',' << motion_ingress.odom_adv_normalized
                << ',' << motion_ingress.odom_rot_raw
                << ',' << motion_ingress.odom_rot_normalized
                << ',' << motion_ingress.odom_ts_ms
                << ',' << (int)odometry_prior.valid
                << ',' << (int)odometry_prior.fresh
                << ',' << odometry_prior.delta_pose[0]
                << ',' << odometry_prior.delta_pose[1]
                << ',' << odometry_prior.delta_pose[2]
                << ',' << last_cmd_cov_(0,0) << ',' << last_cmd_cov_(2,2)
                << ',' << (int)last_measured_prior_.valid
                << ',' << (int)last_measured_prior_.fresh
                << ',' << last_measured_prior_.delta_pose[0]
                << ',' << last_measured_prior_.delta_pose[1]
                << ',' << last_measured_prior_.delta_pose[2]
                << ',' << meas_cov_xx << ',' << meas_cov_tt
                << ',' << (int)last_selected_prior_.valid
                << ',' << (int)last_selected_prior_.fresh
                << ',' << motion_prior_source_name(last_motion_prior_source_)
                << ',' << last_selected_prior_.delta_pose[0]
                << ',' << last_selected_prior_.delta_pose[1]
                << ',' << last_selected_prior_.delta_pose[2]
                << ',' << sel_cov_xx << ',' << sel_cov_tt
                << ',' << x << ',' << y << ',' << phi
                // slot_mcov: was hardcoded 0 here because the local is out of scope on this path, so
                // the two columns read zero on 93% of rows (1690 of 1822 measured) and any analysis of
                // the motion covariance silently ran on the 7% that took the Adam path — which is
                // conditioned on the prediction having already FAILED, i.e. the least representative
                // frames in the log. Same defect as n_lidar two lines up. The value is now mirrored in
                // last_slot_motion_cov_ at the slot build, which both writers can reach.
                << ',' << last_slot_motion_cov_(0, 0) << ',' << last_slot_motion_cov_(2, 2)
                << ',' << 1               // early_exit = 1
                << ',' << 0               // iters
                << ',' << mean_sdf_pred
                << ',' << lr_eff
                << ',' << x << ',' << y << ',' << phi
                << ',' << res.innovation[0] << ',' << res.innovation[1] << ',' << res.innovation[2]
                << ',' << res.innovation_norm
                << ',' << res.sdf_mse
                << ',' << res.covariance(0,0) << ',' << res.covariance(2,2)
                << ',' << 0               // cond_num
                << ',' << (int)window_mgr_.size()
                << ',' << tracking_step_count_
                << ',' << 0.f << ',' << 0.f << ',' << 0.f << ',' << 0.f  // loss_boundary/obs/motion/corner
                << ',' << 0.f   // loss_object  <-- was MISSING: this path emitted 91 fields against
                                // the Adam path's 92, so every early-exit row was shifted by one from
                                // loss_init onward and any reader keyed on the header silently
                                // mis-parsed or dropped them. Since early exit is ~98% of frames, that
                                // meant per-frame analysis saw ONLY the frames where the prediction had
                                // already failed the gate — a badly biased sample of exactly the thing
                                // one wants to measure.
                << ',' << 0.f   // loss_init
                << ',' << std::chrono::duration<float, std::milli>(
                               std::chrono::high_resolution_clock::now() - t_update_start_).count()
                << ',' << 0.f << ',' << 0.f << ',' << 0.f  // t_adam, t_cov, t_breakdown
                << ',' << "na"   // slot_poses_pre (no Adam ran)
                << ',' << "na"   // slot_poses_post
                ;
            (void)0;

            // Per-slot sdf_mse and boundary prior are still meaningful on early exit
            {
                std::string s;
                for (size_t i = 0; i < window_mgr_.window.size(); ++i)
                {
                    if (i > 0) s += '|';
                    std::ostringstream ss;
                    ss << std::fixed << std::setprecision(4)
                       << window_mgr_.window[i].sdf_mse_final;
                    s += ss.str();
                }
                debug_log_ << ',' << (s.empty() ? "na" : s);
            }
            {
                const auto& bp = window_mgr_.boundary_prior;
                debug_log_ << ',' << (int)bp.valid
                           << ',' << bp.mu[0]
                           << ',' << bp.mu[1]
                           << ',' << bp.mu[2];
            }
            debug_log_ << ',' << 0.f   // lbfgs_grad_norm (no optimization ran)
                       << ',' << "na"  // loss_curve
                       << ",," 
                       << ',' << learned_odom_bias_.x()
                       << ',' << learned_odom_bias_.y()
                       << ',' << learned_odom_bias_.z();
            debug_log_ << ',' << res.early_exit_metric
                       << ',' << lost_logodds_
                       << ',' << rivals_.size()
                       << ',' << (grid_search_active_.load(std::memory_order_relaxed) ? 1 : 0);
            write_debug_tail();
            debug_log_ << '\n';
            debug_log_.flush();
        }
    }

    // Moved out of room_concept.h (Task R1b, file-length lint); the doc comments stay with the declarations.
    float RoomConcept::median_abs_sdf(const Eigen::VectorXf &sdf_vals)
    {
        if (sdf_vals.size() == 0) return std::numeric_limits<float>::quiet_NaN();
        std::vector<float> a(static_cast<std::size_t>(sdf_vals.size()));
        for (Eigen::Index i = 0; i < sdf_vals.size(); ++i) a[static_cast<std::size_t>(i)] = std::abs(sdf_vals(i));
        const auto mid = a.begin() + static_cast<std::ptrdiff_t>((a.size() - 1) / 2);
        std::nth_element(a.begin(), mid, a.end());
        return *mid;
    }

    float RoomConcept::weighted_median_abs_sdf(const Eigen::VectorXf &sdf_vals, const Eigen::VectorXf &weights)
    {
        if (weights.size() == 0 or weights.size() != sdf_vals.size())
            return median_abs_sdf(sdf_vals);
        std::vector<std::pair<float, float>> v;
        v.reserve(static_cast<size_t>(sdf_vals.size()));
        double total = 0.0;
        for (Eigen::Index i = 0; i < sdf_vals.size(); ++i)
        {
            const float aa = std::abs(sdf_vals(i)), wa = weights(i);
            if (wa > 0.f and std::isfinite(aa)) { v.emplace_back(aa, wa); total += wa; }
        }
        if (v.empty() or total <= 0.0) return median_abs_sdf(sdf_vals);   // every return discounted
        std::sort(v.begin(), v.end(), [](const auto &x, const auto &y) { return x.first < y.first; });
        double acc = 0.0;
        for (const auto &[val, wt] : v)
        {
            acc += wt;
            if (acc >= 0.5 * total) return val;
        }
        return v.back().first;
    }

} // namespace rc
