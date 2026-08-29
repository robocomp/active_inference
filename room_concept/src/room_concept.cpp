#include "room_concept.h"
#include "pointcloud_center_estimator.h"
#include "room_gn_solver.h"
#include "room_obs_weights.h"
#include "room_gn_solver.h"
#include "image_edge_accumulate.h"
#include "image_edge_ops.h"
#include <algorithm>
#include <ranges>
#include <fstream>
#include <limits>
#include <locale>
#include <print>
#include <sstream>
#include <sys/stat.h>
#include <QDebug>

namespace rc
{
    namespace
    {
        constexpr float kObsWeightEps = 1e-6f;

        /// Conversion between the two absolute-residual estimators this class used to mix.
        /// For a half-normal residual of scale sigma, E|x| = sqrt(2/pi)*sigma = 0.798*sigma while
        /// median|x| = 0.674*sigma. Until 2026-08-13 UpdateResult::sdf_mse was a MEAN on the
        /// early-exit path and a MEDIAN on the optimized path; it is now a median everywhere, so a
        /// threshold that was calibrated against the old mean must be multiplied by this factor to
        /// keep its meaning. Applied once to each affected config value (see config.toml)
        /// and once in boundary_weight_now().
        /// ⚠ PROVISIONAL. 0.845 is the ratio for a HALF-NORMAL residual, and this residual is not
        /// half-normal: roughly two thirds of the floor is systematic map mismatch (furniture,
        /// doorway gaps, wall thickness the polygon omits) rather than zero-mean noise. A systematic
        /// component drives the mean and the median TOWARDS each other — for a pure offset the ratio
        /// is 1 — so 0.845 is expected to OVER-correct, leaving the rescaled thresholds slightly too
        /// tight. Do not reason about the shape of the distribution; measure the ratio:
        ///
        ///     python3 tools/measure_sdf_ratio.py tmp/sdf_localizer/log_<newest>.csv
        ///
        /// On an early-exit frame the log now carries both statistics of the SAME residuals at the
        /// SAME pose (sdf_mse = median, early_exit_metric = mean), so their ratio IS this constant.
        /// The tool prints the measured value and re-derives all six thresholds from their original
        /// mean-calibrated numbers. Replace the value here and in config.toml together.
        constexpr float kMedianOverMeanAbs = 0.674f / 0.798f;   // = 0.845, pending measurement
    }

    // NOTE: these two are declared in room_obs_weights.h (external linkage) rather than living in the
    // anonymous namespace, so room_gn_solver builds its analytic Jacobian from the SAME weights this
    // loss uses. Bodies unchanged.
        torch::Tensor build_observation_weights(const Model& model,
                                                const RoomConcept::Params& params,
                                                const torch::Tensor& points_robot,
                                                const torch::Tensor& pose_theta,
                                                const Model::SdfQueryResult& query)
        {
            auto weights = torch::ones({points_robot.size(0)}, points_robot.options());
            bool any_weighting = false;

            if (points_robot.size(0) <= 1)
                return weights;

            if (params.far_points_weight)
            {
                auto pts_xy = points_robot.index(
                    {torch::indexing::Slice(), torch::indexing::Slice(0, 2)});
                auto dists = torch::norm(pts_xy, 2, /*dim=*/1);
                auto dists_alpha = torch::pow(dists, params.far_points_exponent);
                auto range_weights = dists_alpha / (dists_alpha.mean() + kObsWeightEps);
                range_weights = range_weights.clamp_min(params.far_points_min_weight);
                weights = weights * range_weights;
                any_weighting = true;
            }

            if (params.incidence_angle_weight && query.closest_normals.defined() && query.closest_normals.numel() > 0)
            {
                auto pts_xy = points_robot.index(
                    {torch::indexing::Slice(), torch::indexing::Slice(0, 2)});
                auto ray_norms = torch::norm(pts_xy, 2, /*dim=*/1).clamp_min(kObsWeightEps);
                auto ray_dirs_robot = pts_xy / ray_norms.unsqueeze(1);

                auto theta = pose_theta.to(points_robot.device());
                const auto c = torch::cos(theta).squeeze();
                const auto s = torch::sin(theta).squeeze();
                auto rot = torch::stack({
                    torch::stack({c, -s}),
                    torch::stack({s, c})
                });
                auto ray_dirs_room = torch::matmul(ray_dirs_robot, rot.transpose(0, 1));

                auto normals = query.closest_normals;
                auto normal_norms = torch::norm(normals, 2, /*dim=*/1).clamp_min(kObsWeightEps);
                auto normals_unit = normals / normal_norms.unsqueeze(1);

                auto incidence = torch::abs(torch::sum(ray_dirs_room * normals_unit, /*dim=*/1));
                auto incidence_weights = torch::pow(incidence.clamp_min(kObsWeightEps),
                                                    params.incidence_angle_exponent);
                incidence_weights = incidence_weights.clamp_min(params.incidence_angle_min_weight);
                weights = weights * incidence_weights;
                any_weighting = true;
            }

            if (!any_weighting)
                return weights;

            return (weights / (weights.mean() + kObsWeightEps)).detach();
        }

        torch::Tensor compute_observation_loss_from_query(const Model& model,
                                  const RoomConcept::Params& params,
                                                          const torch::Tensor& points_robot,
                                                          const torch::Tensor& pose_theta,
                                                          const Model::SdfQueryResult& query)
        {
            if (!query.sdf.defined() || query.sdf.size(0) == 0)
                return torch::zeros({}, points_robot.options());

            const float inv_var = 1.0f / (params.rfe_obs_sigma * params.rfe_obs_sigma);
            auto per_point = torch::nn::functional::huber_loss(
                query.sdf,
                torch::zeros_like(query.sdf),
                torch::nn::functional::HuberLossFuncOptions()
                    .reduction(torch::kNone).delta(params.rfe_huber_delta));

            auto weights = build_observation_weights(model, params, points_robot, pose_theta, query);
            return 0.5f * inv_var * (per_point * weights).mean();
        }

    namespace
    {
        torch::Tensor compute_observation_loss(const Model& model,
                                               const RoomConcept::Params& params,
                                               const torch::Tensor& points_robot,
                                               const torch::Tensor& pose_xy,
                                               const torch::Tensor& pose_theta)
        {
            const auto query = model.sdf_query_at_pose(points_robot, pose_xy, pose_theta);
            return compute_observation_loss_from_query(model, params, points_robot, pose_theta, query);
        }
    }

    // =====================================================================
    //  Threading: start / stop / run / get_last_result / push_command
    // =====================================================================

    // Static initialization: limit PyTorch threads to avoid CPU overload. These run before config
    // loads, so they are conservative defaults; start() re-applies the config intra-op count once
    // params are live. (set_num_interop_threads must be called before the inter-op pool is first
    // used, so it stays here and is NOT runtime-tunable.)
    static bool torch_threads_initialized = []() {
        torch::set_num_threads(2);
        torch::set_num_interop_threads(1);
        return true;
    }();

    RoomConcept::~RoomConcept()
    {
        stop();
    }

    void RoomConcept::start()
    {
        if (loc_running_.load() || loc_thread_.joinable()) return;
        stop_requested_ = false;
        commands_pending_.store(false);
        {
            std::lock_guard lock(wake_mutex_);
            latest_notified_lidar_ts_ = std::numeric_limits<std::int64_t>::min();
        }

        // Apply the configured intra-op thread count now that params are loaded (the static
        // initializer used a conservative default before config was available). set_num_threads is
        // safe to call at runtime; this is the main lever for room_concept's CPU-core footprint.
        if (params.torch_num_threads > 0)
            torch::set_num_threads(params.torch_num_threads);

        // If CUDA is requested, initialize the CUDA context HERE on the calling thread
        // (main/Qt thread) before spawning the localization thread.
        // libtorch requires that CUDA be initialized on the thread that will own the
        // context, or at minimum that the CUDA dispatcher has been activated before
        // worker threads try to create CUDA tensors.  Without this, is_available()
        // returns true (driver found) but CUDA kernels fail to register.
        if (params.use_cuda && torch::cuda::is_available())
        {
            try {
                // Warm-up: create and discard a tiny CUDA tensor to trigger CUDA init
                auto tmp = torch::zeros({1}, torch::TensorOptions().device(torch::kCUDA));
                (void)tmp;
            } catch (const std::exception& e) {
                qWarning() << "CUDA init failed:" << e.what() << "falling back to CPU.";
                params.use_cuda = false;
            }
        }

        // Announce the non-default motion-prior configuration, so a run whose predictions came from
        // the encoder alone says so in its own log rather than only in a config file nobody re-reads.
        if (not params.use_command_velocity_prior)
            qInfo() << "[RoomConcept] UseCommandVelocityPrior = false: the commanded (joystick/controller)"
                    << "velocity is computed and logged but EXCLUDED from the motion prior;"
                    << "motion_prior_source will read measured / fallback_zero, never fused / command.";

        loc_running_ = true;
        loc_thread_ = std::thread(&RoomConcept::run, this);
    }

    void RoomConcept::stop()
    {
        stop_requested_ = true;
        wake_cv_.notify_all();
        if (loc_thread_.joinable())
        {
            if (loc_thread_.get_id() == std::this_thread::get_id())
            {
                qWarning() << "RoomConcept::stop called from localization thread; detaching to avoid self-join deadlock";
                loc_thread_.detach();
            }
            else
                loc_thread_.join();
        }
        loc_running_ = false;
    }

    void RoomConcept::init_debug_log()
    {
        if (!params.debug_log_enabled) return;

        ::mkdir("tmp", 0755);
        ::mkdir("tmp/sdf_localizer", 0755);

        // Build timestamped filename: log_YYYY-MM-DD_HH-MM-SS.csv
        {
            const auto now = std::chrono::system_clock::now();
            const std::time_t tt = std::chrono::system_clock::to_time_t(now);
            std::tm tm_local{};
            localtime_r(&tt, &tm_local);
            char ts_buf[32];
            std::strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%d_%H-%M-%S", &tm_local);
            debug_log_path_ = std::string("tmp/sdf_localizer/log_") + ts_buf + ".csv";
        }
        debug_log_.open(debug_log_path_, std::ios::out | std::ios::trunc);
        if (!debug_log_.is_open())
        {
            qWarning() << "Debug log could not be opened:" << QString::fromStdString(debug_log_path_);
            return;
        }

        // CSV header — one row per update() call
        debug_log_
            << "ts_ms"
            << ",wall_ms"
            << ",dt_ms"
            << ",n_lidar"
            << ",vel_adv_y"
            << ",vel_rot"
            << ",odom_adv"
            << ",odom_rot"
            << ",cmd_ingress_source,cmd_adv_raw,cmd_adv_norm,cmd_rot_raw,cmd_rot_norm,cmd_ingress_ts"
            << ",odom_ingress_source,odom_adv_raw,odom_adv_norm,odom_rot_raw,odom_rot_norm,odom_ingress_ts"
            << ",cmd_valid"
            << ",cmd_fresh"
            << ",cmd_dx,cmd_dy,cmd_dth"
            << ",cmd_cov_xx,cmd_cov_tt"
            << ",meas_valid"
            << ",meas_fresh"
            << ",meas_dx,meas_dy,meas_dth"
            << ",meas_cov_xx,meas_cov_tt"
            << ",sel_valid,sel_fresh,sel_source"
            << ",sel_dx,sel_dy,sel_dth"
            << ",sel_cov_xx,sel_cov_tt"
            << ",pred_x,pred_y,pred_theta"
            << ",slot_mcov_xx,slot_mcov_tt"
            << ",early_exit"
            << ",iters"
            << ",final_loss"
            << ",lr_eff"
            << ",res_x,res_y,res_theta"
            << ",innov_x,innov_y,innov_theta"
            << ",innov_norm"
            << ",sdf_mse"
            << ",cov_xx,cov_tt"
            << ",cond_num"
            << ",window_size"
            << ",tracking_steps"
            << ",loss_boundary"
            << ",loss_obs"
            << ",loss_motion"
            << ",loss_corner"
            << ",loss_object"
            << ",loss_init"
            << ",t_update_ms"
            << ",t_adam_ms"
            << ",t_cov_ms"
            << ",t_breakdown_ms"
            // ---- appended for loss/recovery analysis; APPEND ONLY, never insert ----
            // early_exit_metric was absent and had to be proxied through final_loss, which conflates
            // two different quantities across the two paths. recovery_bad/cooldown make the trigger's
            // internal state visible so an episode can be read forwards instead of inferred backwards
            // from window_size resets. search_active marks the frames a search was live on.
            //
            // ORDER FIXED 08-09: this group used to be declared HERE, before slot_poses_*, while both
            // writers emit it LAST. Every column from slot_poses_pre onward was therefore mislabelled
            // (slot-pose strings landed in the column called early_exit_metric). Columns 0..73, up to
            // t_breakdown_ms, were always correct. The header is what moved rather than the writers,
            // deliberately: the row layout below is what every log already on disk was written with,
            // so a reader keyed on this corrected header now reads the old files correctly too.
            << ",slot_poses_pre"
            << ",slot_poses_post"
            << ",slot_sdf_mse"
            << ",bp_valid,bp_x,bp_y,bp_theta"
            << ",lbfgs_grad_norm"
            << ",loss_curve"
            << ",ml_slip_k"
            << ",ml_odom_noise_trans"
            << ",ml_bias_x,ml_bias_y,ml_bias_theta"
            << ",early_exit_metric"
            << ",recovery_bad,recovery_cooldown,search_active"
            // ★ APPENDED 08-16, AT THE END, ON BOTH WRITERS TOGETHER. This file's history is the
            // reason for that discipline: a group once declared mid-header while both writers emitted
            // it LAST mislabelled every column from slot_poses_pre onward. New columns go at the tail
            // and nowhere else, and every one below is written on BOTH paths — a column that exists on
            // only one is worse than no column, because it silently restricts the analysis to the
            // frames that happened to take that path (which is how n_lidar came to read 0 on 98% of
            // rows). Old logs simply lack these fields; a reader keyed on the header handles that.
            //
            // preint_*   how many odometry samples the motion factor summarised and over how long —
            //            the direct witness of whether striding is chaining intervals as intended.
            // slot_mcov_ off-diagonals of the motion covariance. The whole argument for propagating it
            //   xy/xt/yt is that a heading error rotates subsequent translation; these are the terms
            //            that carries, and until now only the diagonal was logged, so the change's
            //            central claim was the one thing the log could not show.
            << ",preint_n,preint_T"
            << ",belief_age_s,belief_decay,aff_outcome,aff_completions,tgt_x,tgt_y,pub_tx,pub_ty,pub_ok"
            << ",slot_mcov_xy,slot_mcov_xt,slot_mcov_yt"
            // imu_cover fraction of the interval's segments whose heading came from the GYRO rather
            //            than the wheels. -1 when no segments were integrated.
            << ",imu_cover"
            // pub_cov_*  what the RT edge actually carried, AFTER the worker's kinematic clamp folded
            //            its un-applied residual in. cov_xx above is the PRE-clamp value, so these two
            //            differing is the clamp's contribution made visible instead of inferred.
            // clamp_hit  1 when the clamp fired on this frame, so its rate is measured not asserted.
            // wg_ratio   wheel/gyro heading ratio over IMU-covered segments; NaN until enough rotation
            //            has accumulated for the quotient to mean anything.
            << ",pub_cov_xx,pub_cov_tt,clamp_hit,wg_ratio"
            << "\n";
        debug_log_.flush();
        qInfo() << "Debug log writing to" << QString::fromStdString(debug_log_path_);

        // Loss/recovery episode log — one row per search, independent of debug_log_enabled because it
        // is low-volume (66 rows in a 708k-frame run) and it is the only record of what recovery did.
        {
            recovery_log_path_ = debug_log_path_;
            const auto slash = recovery_log_path_.find_last_of('/');
            const std::string dir = (slash == std::string::npos) ? std::string()
                                                                 : recovery_log_path_.substr(0, slash + 1);
            const std::string stem = (slash == std::string::npos) ? recovery_log_path_
                                                                  : recovery_log_path_.substr(slash + 1);
            recovery_log_path_ = dir + "recovery_" + (stem.rfind("log_", 0) == 0 ? stem.substr(4) : stem);
            recovery_log_.open(recovery_log_path_, std::ios::out | std::ios::trunc);
            if (recovery_log_.is_open())
            {
                recovery_log_ << "ts_ms,trigger,n_lidar,"
                                 "inc_x,inc_y,inc_theta,inc_loss,good_thr,"
                                 "stage,best_x,best_y,best_theta,best_loss,"
                                 "topk_best,topk_worst,ess,cov_xx,cov_yy,cov_tt,"
                                 "n_evals,jump_m,jump_rad,success,duration_ms,beta,n_points\n";
                recovery_log_.flush();
                qInfo() << "Recovery episode log writing to" << QString::fromStdString(recovery_log_path_);
            }
        }
    }

    std::optional<RoomConcept::UpdateResult> RoomConcept::get_last_result() const
    {
        std::lock_guard lock(result_mutex_);
        return last_result_;
    }

    Eigen::Matrix<float,5,1> RoomConcept::get_loc_state() const
    {
        std::lock_guard lock(result_mutex_);
        if (last_result_.has_value() && last_result_->ok)
            return last_result_->state;
        return Eigen::Matrix<float,5,1>::Zero();
    }

    void RoomConcept::record_command_ingress(const std::string& source,
                                             float raw_adv,
                                             float raw_rot,
                                             float normalized_adv,
                                             float normalized_rot,
                                             std::int64_t ts_ms)
    {
        std::lock_guard lock(motion_ingress_debug_mutex_);
        motion_ingress_debug_.command_source = source;
        motion_ingress_debug_.command_adv_raw = raw_adv;
        motion_ingress_debug_.command_adv_normalized = normalized_adv;
        motion_ingress_debug_.command_rot_raw = raw_rot;
        motion_ingress_debug_.command_rot_normalized = normalized_rot;
        motion_ingress_debug_.command_ts_ms = ts_ms;
    }

    void RoomConcept::record_odometry_ingress(const std::string& source,
                                              float raw_adv,
                                              float raw_rot,
                                              float normalized_adv,
                                              float normalized_rot,
                                              std::int64_t ts_ms)
    {
        std::lock_guard lock(motion_ingress_debug_mutex_);
        motion_ingress_debug_.odom_source = source;
        motion_ingress_debug_.odom_adv_raw = raw_adv;
        motion_ingress_debug_.odom_adv_normalized = normalized_adv;
        motion_ingress_debug_.odom_rot_raw = raw_rot;
        motion_ingress_debug_.odom_rot_normalized = normalized_rot;
        motion_ingress_debug_.odom_ts_ms = ts_ms;
    }

    RoomConcept::MotionIngressDebug RoomConcept::get_motion_ingress_debug() const
    {
        std::lock_guard lock(motion_ingress_debug_mutex_);
        return motion_ingress_debug_;
    }

    Eigen::Vector3f RoomConcept::get_predictor_delta() const
    {
        std::lock_guard lock(motion_ingress_debug_mutex_);
        return predictor_delta_;
    }

    void RoomConcept::push_command(Command cmd)
    {
        {
            std::lock_guard lock(cmd_mutex_);
            pending_commands_.push_back(std::move(cmd));
        }
        commands_pending_.store(true);
        wake_cv_.notify_one();
    }

    void RoomConcept::notify_new_lidar(std::int64_t lidar_timestamp_ms)
    {
        {
            std::lock_guard lock(wake_mutex_);
            if (lidar_timestamp_ms <= latest_notified_lidar_ts_)
                return;
            latest_notified_lidar_ts_ = lidar_timestamp_ms;
        }
        wake_cv_.notify_one();
    }

    void RoomConcept::configure_room_from_polygon(const std::vector<Eigen::Vector2f>& polygon_vertices)
    {
        init_use_polygon_ = true;
        init_polygon_vertices_ = polygon_vertices;
    }

    void RoomConcept::configure_room_from_rect(float width, float length)
    {
        init_use_polygon_ = false;
        init_room_width_ = width;
        init_room_length_ = length;
    }

    void RoomConcept::set_seed_pose_file(const std::string& pose_file_path)
    {
        seed_pose_file_path_ = pose_file_path;
    }

    float RoomConcept::estimate_orientation_from_points(const std::vector<Eigen::Vector3f>& pts) const
    {
        if (pts.size() < 2)
            return 0.f;

        Eigen::Vector2f mean = Eigen::Vector2f::Zero();
        for (const auto& p : pts)
            mean += p.head<2>();
        mean /= static_cast<float>(pts.size());

        Eigen::Matrix2f cov = Eigen::Matrix2f::Zero();
        for (const auto& p : pts)
        {
            const Eigen::Vector2f d = p.head<2>() - mean;
            cov += d * d.transpose();
        }
        cov /= std::max<std::size_t>(1, pts.size() - 1);

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> es(cov);
        if (es.info() != Eigen::Success)
            return 0.f;

        const Eigen::Vector2f principal = es.eigenvectors().col(1);
        return std::atan2(principal.y(), principal.x());
    }

    void RoomConcept::resolve_initial_yaw_ambiguity(const std::vector<Eigen::Vector3f>& lidar_points, float prior_phi)
    {
        constexpr float kYawTieTolerance = 0.02f;
        if (model_ == nullptr || lidar_points.empty())
            return;

        auto wrap_angle = [](float a)
        {
            while (a > M_PI) a -= 2.f * M_PI;
            while (a < -M_PI) a += 2.f * M_PI;
            return a;
        };
        auto ang_dist = [&](float a, float b)
        {
            return std::abs(wrap_angle(a - b));
        };

        const auto state = get_current_state();
        const float x = state[2];
        const float y = state[3];
        const float theta = wrap_angle(state[4]);

        set_robot_pose(x, y, theta, false);
        const float sdf_theta = evaluate_pose_fit(lidar_points);

        const float theta_pi = wrap_angle(theta + static_cast<float>(M_PI));
        set_robot_pose(x, y, theta_pi, false);
        const float sdf_theta_pi = evaluate_pose_fit(lidar_points);

        const bool similar_fit = std::isfinite(sdf_theta) && std::isfinite(sdf_theta_pi)
                                 && std::abs(sdf_theta - sdf_theta_pi) <= kYawTieTolerance;
        const float d_theta = ang_dist(theta, prior_phi);
        const float d_theta_pi = ang_dist(theta_pi, prior_phi);

        if ((sdf_theta_pi < sdf_theta) || (similar_fit && d_theta_pi < d_theta))
            set_robot_pose(x, y, theta_pi, false);
        else
            set_robot_pose(x, y, theta, false);
    }

    bool RoomConcept::bootstrap_initialization_from_lidar()
    {
        if (is_initialized())
            return true;
        if (run_ctx_.high_lidar_buffer == nullptr)
            return false;

        Eigen::Vector2f init_xy = Eigen::Vector2f::Zero();
        float init_phi = 0.f;
        bool have_saved_pose = false;

        // Try saved pose first — if available we can initialize the model immediately
        // without needing a lidar scan (lidar is only required for pose estimation).
        if (!seed_pose_file_path_.empty())
        {
            std::ifstream in(seed_pose_file_path_);
            float sx = 0.f, sy = 0.f, st = 0.f;
            if (in.is_open() && (in >> sx >> sy >> st))
            {
                init_xy = Eigen::Vector2f(sx, sy);
                init_phi = st;
                have_saved_pose = true;
            }
            else
            {
                qWarning() << "RoomConcept bootstrap: seed pose not loaded from"
                           << QString::fromStdString(seed_pose_file_path_)
                           << "(file missing or parse error)";
            }
        }

        // Grab a sweep if one is already buffered. Without a saved pose we CANNOT proceed without it
        // (the initial position is estimated from the cloud); with one we can seed immediately, but we
        // still want the points — they are what tells us whether the seed is any good (below).
        std::vector<Eigen::Vector3f> pts;
        {
            const auto& [lidar_from_buffer] = run_ctx_.high_lidar_buffer->read_last();
            if (lidar_from_buffer.has_value())
                pts = lidar_from_buffer->first;
        }

        if (!have_saved_pose)
        {
            if (pts.empty())
                return false;

            PointcloudCenterEstimator estimator;
            Eigen::Vector2d room_center_in_robot = Eigen::Vector2d::Zero();
            if (const auto obb = estimator.estimate_obb(pts); obb.has_value())
            {
                room_center_in_robot = obb->center;
                init_phi = static_cast<float>(obb->rotation);
            }
            else if (const auto c = estimator.estimate(pts); c.has_value())
            {
                room_center_in_robot = c.value();
                init_phi = estimate_orientation_from_points(pts);
            }

            Eigen::Rotation2Df R(init_phi);
            init_xy = -(R * room_center_in_robot.cast<float>());
        }

        if (init_use_polygon_ && init_polygon_vertices_.size() >= 3)
        {
            set_polygon_room(init_polygon_vertices_);
            set_robot_pose(init_xy.x(), init_xy.y(), init_phi, false);
        }
        else
        {
            set_initial_state(init_room_width_, init_room_length_, init_xy.x(), init_xy.y(), init_phi);
        }

        bool used_grid_search = false;
        if (!have_saved_pose)
        {
            search_trigger_ = "seed";
            used_grid_search = grid_search_initial_pose(pts, 0.5f, static_cast<float>(M_PI_4));
            if (!used_grid_search)
                qWarning() << "RoomConcept bootstrap: grid search failed. Keeping estimator-based initialization.";
        }
        else
        {
            // The saved pose is a PRIOR, not certainty: the robot may have been moved while the agent
            // was down, the file may predate a layout/room-frame change, or it may simply be stale.
            // Check that it still explains the scan and fall through to the global search if it does
            // not. Deferred when no sweep has arrived yet (the seeded model is published meanwhile so
            // the viewer/planner have geometry) — validate_seed_pose runs on the first one instead.
            if (pts.empty())
                pending_seed_validation_ = true;
            else
                used_grid_search = !validate_seed_pose(pts);
        }

        // Needs points; on the deferred-validation path there are none yet, and the saved yaw is
        // carried as-is until the first sweep resolves it.
        if (!used_grid_search && !pts.empty())
            resolve_initial_yaw_ambiguity(pts, init_phi);

        return true;
    }

    /// Accept-or-relocalize check for a seeded pose. Returns true when the pose stands, false when it
    /// was rejected and the global grid search was run in its place.
    ///
    /// The bar is params.recovery_loss_threshold — deliberately the SAME "this fit means we are lost"
    /// number the runtime RecoveryManager uses, rather than a new knob. Units match: both are metres
    /// (median |SDF| in both, since the units were unified), and both sit far above a converged fit and far
    /// below a real mislocalization.
    bool RoomConcept::validate_seed_pose(const std::vector<Eigen::Vector3f>& pts)
    {
        pending_seed_validation_ = false;
        if (pts.empty() || model_ == nullptr)
            return true;

        const float seed_err = evaluate_pose_fit(pts);
        if (std::isfinite(seed_err) && seed_err <= params.recovery_loss_threshold)
        {
            qInfo() << "[RoomConcept] Seed pose accepted: mean|sdf| =" << seed_err << "m (bar"
                    << params.recovery_loss_threshold << "m).";
            return true;
        }

        qWarning() << "[RoomConcept] Seed pose REJECTED: mean|sdf| =" << seed_err << "m exceeds"
                   << params.recovery_loss_threshold << "m — the saved pose does not explain the scan."
                   << "Running grid search...";
        search_trigger_ = "seed_validation";
        if (!grid_search_initial_pose(pts, 0.5f, static_cast<float>(M_PI_4)))
            qWarning() << "[RoomConcept] Grid search after seed rejection did not reach a good fit;"
                       << "keeping its best candidate. Recovery will retry if it stays bad.";
        window_mgr_.clear(); reset_stride_state();
        symmetry_check_counter_ = 0;
        return false;
    }

    void RoomConcept::run()
    {
        init_debug_log();
        rerun_frame_counter_ = 0;

        if (params.rerun_enabled)
        {
            RerunLogger::Config cfg;
            cfg.enabled = true;
            cfg.host = params.rerun_host;
            cfg.port = params.rerun_port;
            cfg.sdf_every_n = params.rerun_sdf_every_n;
            cfg.sdf_resolution = params.rerun_sdf_resolution;
            cfg.max_queue = params.rerun_max_queue;
            rerun_logger_.init(cfg);
        }

        auto wait_period = std::chrono::milliseconds(0);
        constexpr auto kMinWait = std::chrono::milliseconds(2);
        int same_frame_count = 0;
        std::int64_t last_lidar_ts_processed = std::numeric_limits<std::int64_t>::min();

        while (!stop_requested_.load())
        {
            float update_ms = 0.f;

            // ===== 1. DRAIN PENDING COMMANDS =====
            {
                std::vector<Command> cmds;
                {
                    std::lock_guard lock(cmd_mutex_);
                    cmds.swap(pending_commands_);
                }
                commands_pending_.store(false);
                for (auto& cmd : cmds)
                {
                    std::visit([this](auto&& arg)
                    {
                        using T = std::decay_t<decltype(arg)>;
                        if constexpr (std::is_same_v<T, CmdSetPolygon>)
                            set_polygon_room(arg.vertices);
                        else if constexpr (std::is_same_v<T, CmdSetPose>)
                            set_robot_pose(arg.x, arg.y, arg.theta);
                        else if constexpr (std::is_same_v<T, CmdGridSearch>)
                        {
                            search_trigger_ = "manual";
                            grid_search_initial_pose(arg.lidar_points, arg.grid_res, arg.angle_res);
                        }
                    }, cmd);
                }
            }

            // ===== 2. CHECK INITIALIZATION =====
            if (!is_initialized())
            {
                bootstrap_initialization_from_lidar();
                if (!is_initialized())
                {
                    std::unique_lock lock(wake_mutex_);
                    wake_cv_.wait(lock, [this]()
                    {
                        return stop_requested_.load()
                            || commands_pending_.load()
                            || latest_notified_lidar_ts_ != std::numeric_limits<std::int64_t>::min();
                    });
                }
                continue;
            }

            {
                std::unique_lock lock(wake_mutex_);
                wake_cv_.wait(lock, [this, last_lidar_ts_processed]()
                {
                    return stop_requested_.load()
                        || commands_pending_.load()
                        || latest_notified_lidar_ts_ > last_lidar_ts_processed;
                });
            }
            if (stop_requested_.load())
                break;
            if (commands_pending_.load())
                continue;
            
            // ===== 3. READ LIDAR DATA =====
            LidarData lidar_high;
            if (run_ctx_.high_lidar_buffer!= nullptr)
            {
                const auto& [lidar_from_buffer] = run_ctx_.high_lidar_buffer->read_last();
                if (lidar_from_buffer.has_value())
                {
                    lidar_high = lidar_from_buffer.value();
                }
                else
                {        
                    wait_period = std::chrono::milliseconds(0);
                    continue;
                }
            }
           
            // ===== 5. SNAPSHOT VELOCITY & ODOMETRY HISTORY =====
            auto vel_snap  = run_ctx_.velocity_buffer->get_snapshot<0>();
            auto odom_snap = run_ctx_.odometry_buffer->get_snapshot<0>();

            // ===== 6. RUN LOCALIZATION UPDATE =====
            const bool has_new_lidar_frame = lidar_high.second > last_lidar_ts_processed;
            if (!has_new_lidar_frame)
            {
                same_frame_count++;
                wait_period = std::chrono::milliseconds(0);
            }
            else
            {
                same_frame_count = 0;
                last_lidar_ts_processed = lidar_high.second;
            }

            if (has_new_lidar_frame)
            {
                // Deferred seed validation: bootstrap seeded the model from the saved pose before any
                // sweep was available, so this is the first chance to ask whether that pose actually
                // explains the room. Must run BEFORE update() — otherwise the optimizer spends this
                // frame refining a pose we are about to throw away.
                if (pending_seed_validation_)
                    validate_seed_pose(lidar_high.first);

                const auto t_update_start_ = std::chrono::high_resolution_clock::now();
                const auto res = update(lidar_high, vel_snap, odom_snap);
                update_ms = std::chrono::duration<float, std::milli>(
                    std::chrono::high_resolution_clock::now() - t_update_start_).count();

                // Optimizer-timing telemetry for the UI readout (early-exit ⇔ iterations_used <= 0).
                opt_total_count_.fetch_add(1, std::memory_order_relaxed);
                if (res.iterations_used <= 0)
                    opt_earlyexit_count_.fetch_add(1, std::memory_order_relaxed);
                opt_update_us_sum_.fetch_add(static_cast<std::uint64_t>(update_ms * 1000.0f),
                                             std::memory_order_relaxed);

                // Track-establishment confidence for the symmetry-flip threshold: grow on a good fit,
                // DECAY (not reset) on a bad one so a momentary tight-turn degradation doesn't collapse
                // the inertia that protects a long-correct orientation.
                if (res.sdf_mse < params.symmetry_good_fit_mse)
                    good_fit_streak_ = std::min(good_fit_streak_ + 1, params.symmetry_confidence_cap);
                else
                    good_fit_streak_ = std::max(0, good_fit_streak_ - params.symmetry_confidence_decay);

                // Lean per-update timing CSV (loc-thread only): distribution + CUDA warmup curve.
                if (params.optimizer_timing_csv)
                {
                    if (!opt_csv_open_attempted_)
                    {
                        opt_csv_open_attempted_ = true;
                        opt_csv_.open("etc/optimizer_timing.csv", std::ios::out | std::ios::trunc);
                        if (opt_csv_.is_open())
                            opt_csv_ << "# device=" << (get_device().is_cuda() ? "CUDA" : "CPU") << "\n"
                                     << "ts_ms,update_ms,iters,early_exit,window_size,t_adam_ms,t_cov_ms,sdf_mse,cov_tt\n";
                    }
                    if (opt_csv_.is_open())
                    {
                        opt_csv_ << res.timestamp_ms << ',' << update_ms << ',' << res.iterations_used
                                 << ',' << (res.iterations_used <= 0 ? 1 : 0) << ',' << window_mgr_.size()
                                 << ',' << last_t_adam_ms_ << ',' << last_t_cov_ms_ << ',' << res.sdf_mse
                                 << ',' << res.covariance(2, 2) << '\n';
                        opt_csv_.flush();
                    }
                }

            if (params.rerun_enabled)
            {
                RerunFrame rf;
                rf.ts_ms = res.timestamp_ms;
                rf.x = res.state[2];
                rf.y = res.state[3];
                rf.theta = res.state[4];
                rf.innov_x = res.innovation[0];
                rf.innov_y = res.innovation[1];
                rf.innov_theta = res.innovation[2];
                rf.pred_x = rf.x - rf.innov_x;
                rf.pred_y = rf.y - rf.innov_y;
                rf.pred_theta = rf.theta - rf.innov_theta;
                rf.early_exit = (res.iterations_used <= 0);
                rf.window_size = static_cast<int>(window_mgr_.size());
                rf.iters = res.iterations_used;

                rf.loss_init = last_loss_init_;
                rf.final_loss = res.final_loss;
                rf.loss_boundary = last_loss_breakdown_.boundary;
                rf.loss_obs = last_loss_breakdown_.obs;
                rf.loss_motion = last_loss_breakdown_.motion;
                rf.loss_corner = last_loss_breakdown_.corner;
                rf.loss_object = last_loss_breakdown_.object;

                rf.sdf_mse = res.sdf_mse;
                rf.innov_norm = res.innovation_norm;
                rf.cov_xx = res.covariance(0, 0);
                rf.cov_xy = res.covariance(0, 1);
                rf.cov_yy = res.covariance(1, 1);
                rf.cov_tt = res.covariance(2, 2);
                rf.cond_num = res.condition_number;

                rf.t_update_ms = update_ms;
                rf.t_adam_ms = last_t_adam_ms_;
                rf.t_cov_ms = last_t_cov_ms_;
                rf.t_breakdown_ms = last_t_breakdown_ms_;

                // The EMA learner these carried was deleted 2026-08-26; the fields stay at their
                // "no value" sentinel so the bridge keeps drawing NaN rather than a stale number.
                rf.learned_slip_k           = -1.f;
                rf.learned_odom_noise_trans = -1.f;
                rf.learned_bias_x           = 0.f;
                rf.learned_bias_y           = 0.f;
                rf.learned_bias_theta       = 0.f;
                rf.motion_learn_frames      = 0;

                // Send real room polygon once so the bridge can draw corner-to-corner contour.
                if (!rerun_room_polygon_sent_ && model_ != nullptr && model_->use_polygon && model_->polygon_vertices.defined())
                {
                    auto poly_cpu = model_->polygon_vertices.to(torch::kCPU);
                    if (poly_cpu.dim() == 2 && poly_cpu.size(1) == 2 && poly_cpu.size(0) >= 3)
                    {
                        auto acc = poly_cpu.accessor<float, 2>();
                        rf.has_room_polygon = true;
                        rf.room_polygon.reserve(static_cast<size_t>(poly_cpu.size(0)));
                        for (int i = 0; i < poly_cpu.size(0); ++i)
                            rf.room_polygon.push_back({acc[i][0], acc[i][1]});
                        rerun_room_polygon_sent_ = true;
                    }
                }

                rf.lidar_points.reserve(res.lidar_scan.size());
                for (const auto &p : res.lidar_scan)
                    rf.lidar_points.push_back({p.x(), p.y(), p.z()});

                // Optional SDF grid every N frames.
                rerun_frame_counter_++;
                const int sdf_every_n = std::max(0, params.rerun_sdf_every_n);
                const bool send_sdf = (sdf_every_n > 0) && (rerun_frame_counter_ % sdf_every_n == 0) && (model_ != nullptr);
                if (send_sdf)
                {
                    const int res_grid = std::max(8, params.rerun_sdf_resolution);
                    const float half_w = 0.5f * std::max(0.1f, res.state[0]);
                    const float half_h = 0.5f * std::max(0.1f, res.state[1]);
                    const float span = 2.f * std::max(half_w, half_h);
                    const float cell = span / static_cast<float>(std::max(1, res_grid - 1));
                    const float ox = -0.5f * span;
                    const float oy = -0.5f * span;

                    std::vector<Eigen::Vector3f> grid_robot;
                    grid_robot.reserve(static_cast<size_t>(res_grid) * static_cast<size_t>(res_grid));

                    const float x = res.state[2];
                    const float y = res.state[3];
                    const float th = res.state[4];
                    const float c = std::cos(th);
                    const float s = std::sin(th);

                    for (int iy = 0; iy < res_grid; ++iy)
                    {
                        const float wy = oy + static_cast<float>(iy) * cell;
                        for (int ix = 0; ix < res_grid; ++ix)
                        {
                            const float wx = ox + static_cast<float>(ix) * cell;
                            const float dx = wx - x;
                            const float dy = wy - y;
                            const float rx = c * dx + s * dy;
                            const float ry = -s * dx + c * dy;
                            grid_robot.emplace_back(rx, ry, 0.f);
                        }
                    }

                    torch::NoGradGuard no_grad;
                    auto grid_t = points_to_tensor_xyz(grid_robot, get_device());
                    auto sdf_t = model_->sdf(grid_t).to(torch::kCPU);
                    auto acc = sdf_t.accessor<float, 1>();

                    rf.has_sdf_grid = true;
                    rf.sdf_w = res_grid;
                    rf.sdf_h = res_grid;
                    rf.sdf_origin_x = ox;
                    rf.sdf_origin_y = oy;
                    rf.sdf_cell_size = cell;
                    rf.sdf_values.resize(static_cast<size_t>(res_grid) * static_cast<size_t>(res_grid));
                    for (size_t i = 0; i < rf.sdf_values.size(); ++i)
                        rf.sdf_values[i] = acc[i];
                }

                rerun_logger_.log_frame(std::move(rf));
            }

                // ===== 7. PUBLISH RESULT =====
                {
                    std::lock_guard lock(result_mutex_);
                    last_result_ = res;
                }
                if (res.ok && !loc_initialized_.load())
                    loc_initialized_ = true;

                // Wake the owner to publish this fresh correction to the DSR graph IMMEDIATELY rather
                // than waiting for the next compute() tick (the callback marshals the graph write to the
                // main thread via a Qt::QueuedConnection). Fired OUTSIDE result_mutex_ so we never hold
                // the lock across the hop. Removes ~one compute-period of lidar→RT latency.
                if (res.ok && on_result_ready_)
                    on_result_ready_();

                // ===== 8. RECOVERY DETECTION =====
            // Relocalization (this, the map-trust reloc below, and the symmetry check) stays armed for
            // the WHOLE run, not just until the room node stabilizes. There used to be a
            // set_relocalization_enabled() hook to disarm it once stable; it never had a call site, so
            // always-on is the behaviour that has actually run and been validated, and it is the one we
            // want: a robot that is picked up and moved after the room is stable still needs to recover,
            // and this is the backstop for a bad startup seed that validate_seed_pose let through.
            {
                // res.sdf_mse is ALREADY median |SDF| in metres — see its declaration. It used to be
                // sqrt()'d here, which is the square root of a length: dimensionally meaningless, and
                // it made the effective trigger 0.2025 m while the number in the config read 0.45.
                // RecoveryLossThreshold is now that same 0.2 m directly.
                //
                // Judge on the WORST of the prediction error and the post-fit residual. The post-fit
                // residual ALONE was the bug the operator hit: after the layout is repositioned, the
                // prediction is chronically wrong (the plotted `pred |SDF|` sits ~0.5, far above the
                // opt threshold) but Adam re-solves it from scratch every frame, so the post-fit number
                // looks healthy and the bad-frame counter resets — the search never fires, even though
                // the belief plainly does not explain the world. early_exit_metric is exactly the curve
                // shown in the UI, so what the operator sees and what arms recovery are now the same
                // signal.
                //
                // Turn safety: a hard turn legitimately spikes the prediction error. Two things absorb
                // that — the metric is only compared against a threshold well above normal turn
                // transients, and RecoveryConsecutiveCount (10) requires it to persist. If turns start
                // triggering, that count is the dial, not the threshold.
                // Both arguments are MEDIAN |SDF| (2026-08-13). Watching the worse of the post-fit
                // residual and the prediction error is deliberate and is kept; what changed is that
                // the second one is now res.pred_sdf_median rather than res.early_exit_metric, which
                // is a MEAN. RecoveryLossThreshold is a median bar — it is the same constant that
                // gates compute_seed_error, which returns a median — so feeding it a mean made
                // recovery fire on a signal running some 15-20% high, i.e. earlier than the
                // threshold says. Comparing like with like restores the documented meaning.
                const float avg_sdf_err = std::max(res.sdf_mse, res.pred_sdf_median);
                if (recovery_.check(avg_sdf_err, res.iterations_used,
                                    params.recovery_loss_threshold, params.recovery_consecutive_count))
                {
                    qWarning() << "[LocThread] Recovery triggered after" << recovery_.consecutive_bad_frames
                               << "bad frames. avg_sdf_err=" << avg_sdf_err << "m"
                               << "Running grid search...";
                    const auto& pts = lidar_high.first;
                    search_trigger_ = "recovery";
                    grid_search_initial_pose(pts, 0.5f, static_cast<float>(M_PI_4));
                    window_mgr_.clear(); reset_stride_state();
                    recovery_.on_recovery_done(params.recovery_cooldown_frames);
                    symmetry_check_counter_ = 0;
                }
            }

                // ===== 8b. FE-NATIVE RELOCALIZATION (map-trust collapse) =========
            // Runs the SAME hierarchical grid search as recovery above, but triggered by the higher-level
            // map-trust belief exp(u_b_) collapsing (the map no longer explains the robot) rather than a raw
            // sdf threshold. u_b_ was just updated this frame (optimized path) or nudged (early-exit rotation),
            // so it is current here. Coexists with recovery_ (independent backstop). See HIERARCHICAL_PRECISION.md.
            if (params.hier_prec_reloc_enabled && params.hier_prec_boundary_enabled)
            {
                if (map_trust_reloc_cooldown_ > 0)
                    --map_trust_reloc_cooldown_;
                else
                {
                    const float map_trust = std::exp(u_b_);
                    map_trust_low_streak_ = (map_trust < params.hier_prec_reloc_floor)
                                          ? map_trust_low_streak_ + 1 : 0;
                    if (map_trust_low_streak_ >= params.hier_prec_reloc_consecutive)
                    {
                        qWarning() << "[reloc] map_trust collapsed exp(u_b)=" << map_trust << "for"
                                   << map_trust_low_streak_ << "frames — running grid search...";
                        log_hier_prec_row("reloc", 0.f, 0.f, 0.f, /*reloc_fired=*/true);
                        const auto& pts = lidar_high.first;
                        search_trigger_ = "map_trust";
                        grid_search_initial_pose(pts, 0.5f, static_cast<float>(M_PI_4));
                        window_mgr_.clear(); reset_stride_state();
                        u_b_init_ = false;   // reseed u_b_ to g(v) on the next boundary update
                        map_trust_v_ = 0.f;  // fresh full trust after relocating
                        map_trust_low_streak_ = 0;
                        map_trust_reloc_cooldown_ = params.hier_prec_reloc_cooldown_frames;
                        symmetry_check_counter_ = 0;
                    }
                }
            }

                // ===== 9. PERIODIC SYMMETRY CHECK ================================
            // Uses res.sdf_mse as reference (already computed this frame).
            // Tests all four pose symmetries that a polygonal room may have:
            //   rot180  : (-x, -y,  θ+π)   — 180° rotation
            //   refl_y  : (-x,  y,  π−θ)   — Y-axis reflection  ← most common failure
            //   refl_x  : ( x, -y,   −θ)   — X-axis reflection
            //   rot180_y: (-x,  y,  θ+π)   — combined rot+refl (same as rot180 ∘ refl_y)
            if (params.symmetry_check_interval > 0
                && res.iterations_used > 0)
            {
                ++symmetry_check_counter_;
                if (symmetry_check_counter_ >= params.symmetry_check_interval)
                {
                    symmetry_check_counter_ = 0;
                    const auto cur  = model_->get_state();
                    const float cx  = cur[2], cy = cur[3], cth = cur[4];
                    const auto& pts = lidar_high.first;

                    // Subsample (reuse grid-search budget)
                    std::vector<Eigen::Vector3f> sample;
                    const int stride = std::max(1,
                        static_cast<int>(pts.size()) / params.grid_search_max_samples);
                    sample.reserve(pts.size() / stride + 1);
                    for (size_t i = 0; i < pts.size(); i += stride)
                        sample.push_back(pts[i]);
                    const torch::Tensor pts_t = points_to_tensor_xyz(sample, get_device());

                    const float loss_cur = res.sdf_mse;

                    // Evaluate all symmetry candidates. MUST use the same reduction as loss_cur above
                    // (median |SDF|, metres): this was mean(sdf²) in m² while loss_cur was already in
                    // metres, so every candidate scored ~14x "better" than the current pose no matter
                    // how correct that pose was, and `advantage` came out positive on every check —
                    // the test was biased toward flipping until good_fit_streak_ grew large enough for
                    // the confidence gain to outrun the bogus evidence.
                    auto eval_at = [&](float nx, float ny, float nth) -> float {
                        torch::NoGradGuard ng;
                        auto xy = torch::tensor({nx, ny},
                            torch::TensorOptions().dtype(torch::kFloat32).device(get_device()));
                        auto th = torch::tensor({nth},
                            torch::TensorOptions().dtype(torch::kFloat32).device(get_device()));
                        return median_abs_sdf(model_->sdf_at_pose(pts_t, xy, th));
                    };

                    struct Candidate { const char* name; float x, y, theta, loss; };
                    std::array<Candidate,4> cands = {{
                        {"rot180",   -cx,  -cy,  cth + static_cast<float>(M_PI),  0.f},
                        {"refl_y",   -cx,   cy,  static_cast<float>(M_PI) - cth,  0.f},
                        {"refl_x",    cx,  -cy,  -cth,                             0.f},
                        {"rot180_y", -cx,   cy,  cth + static_cast<float>(M_PI),  0.f},
                    }};
                    for (auto& c : cands)
                        c.loss = eval_at(c.x, c.y, c.theta);

                    // Pick the best candidate
                    const auto* best = &cands[0];
                    for (const auto& c : cands)
                        if (c.loss < best->loss) best = &c;

                    // Per-check advantage of the best symmetric candidate over the current pose, net of
                    // the base margin. Positive ⇒ the flip looks better THIS check.
                    const float advantage = (loss_cur - best->loss) - params.symmetry_flip_min_improvement;
                    // Leaky accumulation: sustained advantage builds; a momentary tight-turn blip decays
                    // (leak<1), and a check where the current pose is better actively drains it.
                    symmetry_flip_evidence_ = std::max(0.f,
                        params.symmetry_evidence_leak * symmetry_flip_evidence_ + advantage);
                    // Threshold grows with how long the current orientation has been established → a
                    // long-correct track demands far more sustained evidence to flip than a fresh one.
                    const float thresh = params.symmetry_flip_evidence_thresh
                        * (1.f + params.symmetry_confidence_gain
                                 * static_cast<float>(std::min(good_fit_streak_, params.symmetry_confidence_cap)));
                    const bool flip_triggered = symmetry_flip_evidence_ > thresh;

                    // Trial CSV: one row per check (not just on a flip) so a flip event can be
                    // traced back through the evidence/threshold trajectory that led to it.
                    // Independent of debug_log_enabled — cheap, gated by symmetry_debug_csv.
                    if (params.symmetry_debug_csv)
                    {
                        if (!symmetry_csv_open_attempted_)
                        {
                            symmetry_csv_open_attempted_ = true;
                            ::mkdir("tmp", 0755);
                            ::mkdir("tmp/sdf_localizer", 0755);
                            const auto now = std::chrono::system_clock::now();
                            const std::time_t tt = std::chrono::system_clock::to_time_t(now);
                            std::tm tm_local{};
                            localtime_r(&tt, &tm_local);
                            char ts_buf[32];
                            std::strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%d_%H-%M-%S", &tm_local);
                            const std::string path = std::string("tmp/sdf_localizer/symmetry_check_") + ts_buf + ".csv";
                            symmetry_csv_.open(path, std::ios::out | std::ios::trunc);
                            if (symmetry_csv_.is_open())
                                symmetry_csv_ << "ts_ms,cx,cy,cth,loss_cur,best_name,best_loss,"
                                                 "loss_rot180,loss_refl_y,loss_refl_x,loss_rot180_y,"
                                                 "advantage,evidence,thresh,good_fit_streak,flip_triggered\n";
                            else
                                qWarning() << "Symmetry debug CSV could not be opened:" << QString::fromStdString(path);
                        }
                        if (symmetry_csv_.is_open())
                        {
                            symmetry_csv_ << res.timestamp_ms
                                          << ',' << cx << ',' << cy << ',' << cth
                                          << ',' << loss_cur
                                          << ',' << best->name << ',' << best->loss
                                          << ',' << cands[0].loss << ',' << cands[1].loss
                                          << ',' << cands[2].loss << ',' << cands[3].loss
                                          << ',' << advantage
                                          << ',' << symmetry_flip_evidence_
                                          << ',' << thresh
                                          << ',' << good_fit_streak_
                                          << ',' << (int)flip_triggered
                                          << '\n';
                            symmetry_csv_.flush();
                        }
                    }

                    if (flip_triggered)
                    {
                        qWarning() << "[SymmetryCheck]" << best->name << "flip — evidence"
                                   << symmetry_flip_evidence_ << "> thresh" << thresh
                                   << "(streak" << good_fit_streak_ << ")";
                        set_robot_pose(best->x, best->y, best->theta);
                        recovery_.reset();
                        window_mgr_.clear(); reset_stride_state();
                        symmetry_flip_evidence_ = 0.f;
                        good_fit_streak_ = 0;   // fresh orientation — re-establish confidence from scratch
                    }
                }
            }

                wait_period = std::chrono::milliseconds(0);

                // ===== DIFFERENTIAL TEST: RFE vs single-step vs prediction-only =====
            // Compares SDF accuracy AND pose jitter (temporal consistency).
            // Single-step will typically achieve lower SDF (it's unconstrained), but
            // RFE should show lower jitter (the window regularises across time).
            // Enable via params.differential_test_enabled.
            if (params.differential_test_enabled && res.ok)
            {
                const bool adam_ran = res.iterations_used > 0;
                const bool sample_early_exit = !adam_ran && (diff_test_.early_exit_seen++ % 50 == 0);

                if (adam_ran || sample_early_exit)
                {
                    // Recover (or use) the predicted pose
                    float pred_x, pred_y, pred_theta;
                    if (adam_ran)
                    {
                        pred_x     = res.state[2] - res.innovation[0];
                        pred_y     = res.state[3] - res.innovation[1];
                        pred_theta = res.state[4] - res.innovation[2];
                    }
                    else
                    {
                        pred_x     = res.state[2];
                        pred_y     = res.state[3];
                        pred_theta = res.state[4];
                    }

                    const auto& pts = lidar_high.first;
                    auto pts_tensor = points_to_tensor_xyz(pts, get_device());

                    // 1. Prediction-only SDF
                    float pred_sdf;
                    {
                        torch::NoGradGuard no_grad;
                        auto xy = torch::tensor({pred_x, pred_y},
                            torch::TensorOptions().dtype(torch::kFloat32).device(get_device()));
                        auto th = torch::tensor({pred_theta},
                            torch::TensorOptions().dtype(torch::kFloat32).device(get_device()));
                        auto sdf_vals = model_->sdf_at_pose(pts_tensor, xy, th);
                        pred_sdf = torch::median(torch::abs(sdf_vals)).item<float>();
                    }

                    // 2. Single-step Adam (no window, no motion factors) — returns pose + SDF
                    const float rfe_sdf = res.sdf_mse;
                    const float rfe_x = res.state[2], rfe_y = res.state[3], rfe_theta = res.state[4];

                    auto single_pose = torch::tensor({pred_x, pred_y, pred_theta},
                        torch::TensorOptions().dtype(torch::kFloat32).device(get_device())).requires_grad_(true);
                    {
                        torch::optim::Adam opt({torch::optim::OptimizerParamGroup({single_pose},
                            std::make_unique<torch::optim::AdamOptions>(params.learning_rate_pos))});
                        for (int i = 0; i < params.num_iterations; ++i)
                        {
                            opt.zero_grad();
                            auto xy = single_pose.index({torch::indexing::Slice(0, 2)});
                            auto th = single_pose.index({torch::indexing::Slice(2, 3)});
                            auto loss = compute_observation_loss(*model_, params, pts_tensor, xy, th);
                            loss.backward();
                            opt.step();
                        }
                    }
                    float single_sdf;
                    {
                        torch::NoGradGuard no_grad;
                        auto xy = single_pose.index({torch::indexing::Slice(0, 2)});
                        auto th = single_pose.index({torch::indexing::Slice(2, 3)});
                        auto sdf_vals = model_->sdf_at_pose(pts_tensor, xy, th);
                        single_sdf = torch::median(torch::abs(sdf_vals)).item<float>();
                    }
                    auto sp_cpu = single_pose.detach().to(torch::kCPU);
                    auto sp = sp_cpu.accessor<float, 1>();
                    const float s_x = sp[0], s_y = sp[1], s_theta = sp[2];

                    // --- Correction jitter (how far each method moved from prediction) ---
                    const float rfe_corr = std::sqrt((rfe_x - pred_x) * (rfe_x - pred_x) +
                                                     (rfe_y - pred_y) * (rfe_y - pred_y));
                    const float single_corr = std::sqrt((s_x - pred_x) * (s_x - pred_x) +
                                                        (s_y - pred_y) * (s_y - pred_y));
                    auto wrap = [](float a) { while (a > M_PI) a -= 2*M_PI; while (a < -M_PI) a += 2*M_PI; return a; };
                    const float rfe_tcorr = std::abs(wrap(rfe_theta - pred_theta));
                    const float single_tcorr = std::abs(wrap(s_theta - pred_theta));

                    diff_test_.rfe_jitter_sum += rfe_corr;
                    diff_test_.single_jitter_sum += single_corr;
                    diff_test_.rfe_theta_jitter_sum += rfe_tcorr;
                    diff_test_.single_theta_jitter_sum += single_tcorr;

                    // --- Correction consistency (how stable corrections are across frames) ---
                    // correction = optimised - predicted (for each method)
                    const float rfe_cx = rfe_x - pred_x, rfe_cy = rfe_y - pred_y;
                    const float rfe_ctheta = wrap(rfe_theta - pred_theta);
                    const float single_cx = s_x - pred_x, single_cy = s_y - pred_y;
                    const float single_ctheta = wrap(s_theta - pred_theta);

                    if (diff_test_.has_prev)
                    {
                        // RFE correction change
                        const float drx = rfe_cx - diff_test_.prev_rfe_cx;
                        const float dry = rfe_cy - diff_test_.prev_rfe_cy;
                        diff_test_.rfe_corr_consistency_sum += std::sqrt(drx*drx + dry*dry);
                        diff_test_.rfe_theta_consistency_sum += std::abs(wrap(rfe_ctheta - diff_test_.prev_rfe_ctheta));

                        // Single-step correction change
                        const float dsx = single_cx - diff_test_.prev_single_cx;
                        const float dsy = single_cy - diff_test_.prev_single_cy;
                        diff_test_.single_corr_consistency_sum += std::sqrt(dsx*dsx + dsy*dsy);
                        diff_test_.single_theta_consistency_sum += std::abs(wrap(single_ctheta - diff_test_.prev_single_ctheta));
                    }
                    diff_test_.prev_rfe_cx = rfe_cx; diff_test_.prev_rfe_cy = rfe_cy; diff_test_.prev_rfe_ctheta = rfe_ctheta;
                    diff_test_.prev_single_cx = single_cx; diff_test_.prev_single_cy = single_cy; diff_test_.prev_single_ctheta = single_ctheta;
                    diff_test_.has_prev = true;

                    // --- Accumulate SDF stats ---
                    diff_test_.pred_sdf_sum   += pred_sdf;
                    diff_test_.single_sdf_sum += single_sdf;
                    diff_test_.rfe_sdf_sum    += rfe_sdf;
                    diff_test_.count++;
                    if (adam_ran) diff_test_.adam_frames++;
                    if (rfe_sdf < single_sdf - 1e-5f)      diff_test_.rfe_wins++;
                    else if (single_sdf < rfe_sdf - 1e-5f)  diff_test_.single_wins++;

                    // Periodic report every 100 frames
                    if (diff_test_.count % 100 == 0 && diff_test_.count > 0)
                    {
                        const int n = diff_test_.count;
                        std::cout << "\n===== DIFFERENTIAL TEST (" << n << " frames, "
                                  << diff_test_.adam_frames << " Adam + "
                                  << (n - diff_test_.adam_frames) << " early-exit) =====\n"
                                  << "  --- SDF accuracy (lower = better fit to room) ---\n"
                                  << "  Prediction-only  avg SDF: " << (diff_test_.pred_sdf_sum / n) << " m\n"
                                  << "  Single-step Adam avg SDF: " << (diff_test_.single_sdf_sum / n) << " m\n"
                                  << "  Full RFE (W=" << params.rfe_window_size << ")   avg SDF: "
                                  << (diff_test_.rfe_sdf_sum / n) << " m\n"
                                  << "  SDF wins — RFE: " << diff_test_.rfe_wins
                                  << "  Single: " << diff_test_.single_wins << "\n"
                                  << "  --- Correction jitter (lower = more stable) ---\n"
                                  << "  RFE    avg pos correction: " << (diff_test_.rfe_jitter_sum / n * 1000) << " mm"
                                  << "  avg θ correction: " << (diff_test_.rfe_theta_jitter_sum / n * 180 / M_PI) << " deg\n"
                                  << "  Single avg pos correction: " << (diff_test_.single_jitter_sum / n * 1000) << " mm"
                                  << "  avg θ correction: " << (diff_test_.single_theta_jitter_sum / n * 180 / M_PI) << " deg\n";
                        if (n > 1)
                        {
                            const int np = n - 1;
                            std::cout << "  --- Correction consistency (lower = smoother) ---\n"
                                      << "  RFE    avg Δcorr: " << (diff_test_.rfe_corr_consistency_sum / np * 1000) << " mm/frame"
                                      << "  avg Δθcorr: " << (diff_test_.rfe_theta_consistency_sum / np * 180 / M_PI) << " deg/frame\n"
                                      << "  Single avg Δcorr: " << (diff_test_.single_corr_consistency_sum / np * 1000) << " mm/frame"
                                      << "  avg Δθcorr: " << (diff_test_.single_theta_consistency_sum / np * 180 / M_PI) << " deg/frame\n";
                        }
                        std::cout << "===========================================\n" << std::endl;
                    }
                }
            }

            // End of heavy localization update path (executed only for new lidar frames).
            }
        }

        rerun_logger_.stop();
        loc_running_ = false;
    }

    float RoomConcept::find_best_initial_orientation(const std::vector<Eigen::Vector3f>& lidar_points,
                                                        float x, float y, float base_phi)
    {
        if (model_ == nullptr || lidar_points.empty())
            return base_phi;

        // Test 4 orientations: base, +90°, +180°, +270° (covers symmetries)
        const std::vector<float> angle_offsets = {0.0f, M_PI_2, M_PI, 3.0f * M_PI_2};

        // Also test mirrored positions (x, y) and (-x, y) with all rotations
        // This handles axis-mirroring ambiguity
        const std::vector<std::pair<float, float>> position_variants = {
            {x, y},      // Original
            {-x, y},     // Mirror X
            {x, -y},     // Mirror Y
            {-x, -y}     // Mirror both
        };

        // Subsample points for faster evaluation
        std::vector<Eigen::Vector3f> sample_points;
        const int max_samples = params.orientation_search_max_samples;
        const int stride = std::max(1, static_cast<int>(lidar_points.size()) / max_samples);
        for (size_t i = 0; i < lidar_points.size(); i += stride)
            sample_points.push_back(lidar_points[i]);

        const torch::Tensor points_tensor = points_to_tensor_xyz(sample_points);

        float best_phi = base_phi;
        float best_x = x;
        float best_y = y;
        float best_loss = std::numeric_limits<float>::infinity();

        // Test all combinations of position variants and angle offsets
        for (const auto& [test_x, test_y] : position_variants)
        {
            for (float offset : angle_offsets)
            {
                float test_phi = base_phi + offset;
                // Normalize to [-pi, pi]
                while (test_phi > M_PI) test_phi -= 2.0f * M_PI;
                while (test_phi < -M_PI) test_phi += 2.0f * M_PI;

                // Temporarily set the pose
                model_->robot_pos.data().copy_(torch::tensor({test_x, test_y},
                    torch::TensorOptions().device(get_device())));
                model_->robot_theta.data().copy_(torch::tensor({test_phi},
                    torch::TensorOptions().device(get_device())));

                // Evaluate SDF loss (without gradient)
                torch::NoGradGuard no_grad;
                const auto sdf_vals = model_->sdf(points_tensor);
                const float loss = torch::mean(torch::square(sdf_vals)).item<float>();

                if (loss < best_loss)
                {
                    best_loss = loss;
                    best_phi = test_phi;
                    best_x = test_x;
                    best_y = test_y;
                }
            }
        }

        // Set the best position found
        model_->robot_pos.data().copy_(torch::tensor({best_x, best_y},
            torch::TensorOptions().device(get_device())));

        return best_phi;
    }

    void RoomConcept::write_search_episode(const SearchEpisode& e)
    {
        if (not recovery_log_.is_open())
            return;
        recovery_log_ << e.ts_ms << ',' << e.trigger << ',' << e.n_lidar
                      << ',' << e.incumbent_x << ',' << e.incumbent_y << ',' << e.incumbent_theta
                      << ',' << e.incumbent_loss << ',' << e.good_thr
                      << ',' << e.stage
                      << ',' << e.best_x << ',' << e.best_y << ',' << e.best_theta << ',' << e.best_loss
                      << ',' << e.topk_best << ',' << e.topk_worst << ',' << e.ess
                      << ',' << e.cov_xx << ',' << e.cov_yy << ',' << e.cov_tt
                      << ',' << e.n_evals << ',' << e.jump_m << ',' << e.jump_rad
                      << ',' << (e.success ? 1 : 0) << ',' << e.duration_ms
                      << ',' << e.beta << ',' << e.n_points << '\n';
        recovery_log_.flush();
    }

    bool RoomConcept::grid_search_initial_pose(const std::vector<Eigen::Vector3f>& lidar_points,
                                                  float grid_resolution,
                                                  float /*angle_resolution*/)
    {
        if (model_ == nullptr || lidar_points.empty())
            return false;

        // One row per call, written wherever this returns — see RoomConcept::SearchEpisode for why the
        // per-frame log was not enough (episodes had to be inferred from window_size resets).
        SearchEpisode ep;
        ep.ts_ms = last_update_result.timestamp_ms;
        ep.n_lidar = static_cast<int>(lidar_points.size());
        ep.trigger = search_trigger_;   // set by the caller just before invoking the search
        const auto ep_t0 = std::chrono::steady_clock::now();
        struct EpisodeWriter
        {
            RoomConcept* self; SearchEpisode* e; const std::chrono::steady_clock::time_point* t0;
            ~EpisodeWriter()
            {
                e->duration_ms = std::chrono::duration<float, std::milli>(
                                     std::chrono::steady_clock::now() - *t0).count();
                self->write_search_episode(*e);
            }
        } ep_writer{this, &ep, &ep_t0};

        // Mark the search live for the UI. RAII because this function returns from four places
        // (Stage 0's flip short-circuit, Stage 1's coarse hit, the Stage 2 tail); a manual clear
        // would eventually be forgotten at one of them and pin the indicator on "SEARCHING".
        struct SearchFlag
        {
            std::atomic<bool>& active;
            std::atomic<std::int64_t>& end_ms;
            explicit SearchFlag(std::atomic<bool>& a, std::atomic<std::int64_t>& e) : active(a), end_ms(e)
            { active.store(true, std::memory_order_relaxed); }
            ~SearchFlag()
            {
                end_ms.store(std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count(),
                    std::memory_order_relaxed);
                active.store(false, std::memory_order_relaxed);
            }
        } search_flag(grid_search_active_, grid_search_end_ms_);

        // ── Helpers ───────────────────────────────────────────────────────────

        // Fit at a pose, without permanently touching the model. Same reduction as everything else
        // (median |SDF|, METRES) so this score is directly comparable to res.sdf_mse and therefore to
        // the recovery trigger. It was mean(sdf²) in m², which is why good_thr=1.0 — read as "1 m²" —
        // accepted anything under ~1 m of error and let Stage 0 declare victory on the pose it was
        // handed, while recovery considered that same pose lost at 0.2 m.
        auto eval_loss = [&](const torch::Tensor& pts, float x, float y, float theta) -> float
        {
            model_->robot_pos.data().copy_(
                torch::tensor({x, y}, torch::TensorOptions().device(get_device())));
            model_->robot_theta.data().copy_(
                torch::tensor({theta}, torch::TensorOptions().device(get_device())));
            torch::NoGradGuard no_grad;
            ++ep.n_evals;
            return median_abs_sdf(model_->sdf(pts));
        };

        // Commit best pose and reset all tracking state.
        // Reassign with new leaf tensors (requires_grad=true) so the next
        // optimizer iteration starts clean — avoids version-counter corruption
        // that .data().copy_() causes when NoGradGuard is not active.
        // Softmax-weighted moment match over the poses the search actually evaluated.
        // The loss is median |SDF| in METRES, so the natural temperature is the SDF observation noise:
        // poses that fit within sensor noise of the winner are genuine rivals and must carry weight;
        // poses much worse than that are not. Angles are wrapped about the winner before averaging.
        //
        // This exists because committing a search result with a FIXED covariance is the search's real
        // failure mode. In a corridor the along-axis direction is a flat valley of near-equal loss, so
        // the winner is an arbitrary point drawn from that valley — and it was being published with
        // sigma = 0.32 m in every direction regardless, i.e. asserted as a confident fix. Measured over
        // 708k frames: recovery fires in tight BURSTS (11 events inside ~350 frames), which is what
        // "commit an ambiguous pose as certain, fail again immediately, search again" looks like.
        // Moment-matching turns the ambiguity into what it actually is — a large variance along the
        // unresolved direction — which both stops the false confidence and lets consumers back off.
        auto moment_match = [&](const std::vector<Eigen::Vector4f>& evals, float best_loss,
                                float ref_theta) -> Eigen::Matrix3f
        {
            // TEMPERATURE. The weight must express how DISTINGUISHABLE two candidate poses are, so
            // the scale is the sampling noise of the statistic being compared — median |SDF| over N
            // points — NOT the per-point sensor noise. For half-normal residuals of scale sigma the
            // median's standard error is ~0.79*sigma/sqrt(N); two poses closer than that in loss are
            // not separated by the evidence, two much further apart are.
            //
            // This was sigma_sdf itself (0.15 m), i.e. the per-point noise with the sqrt(N) left out —
            // about 12x too flat at N=150. Measured consequence over 33 real episodes: ESS p50 563
            // (hundreds of poses carrying real weight) and a committed sigma_x of 1.27 m / sigma_theta
            // of 84 deg. Those numbers describe the temperature, not the room.
            //
            // NOTE this is an UPPER bound on the indistinguishability scale: every candidate is scored
            // against the SAME point set, so the comparison is paired and most of the sampling noise
            // cancels. Erring high leaves the committed covariance conservative — it over-reports
            // uncertainty rather than under-reporting it, which is the safe direction for a consumer.
            const float n_eff = std::max(1.f, static_cast<float>(params.grid_search_max_samples));
            const float beta  = std::max(1e-4f, 0.79f * params.sigma_sdf / std::sqrt(n_eff));
            ep.beta = beta; ep.n_points = static_cast<int>(n_eff);
            // Effective sample size of the softmax weights, ESS = (sum w)^2 / sum w^2. ~1 => one pose
            // explains the scan; large => a flat valley of rivals, so the winner is an arbitrary draw
            // and "success" is not the same thing as "resolved". Recorded per episode.
            float w2sum = 0.f;
            float wsum = 0.f;
            Eigen::Vector3f mean = Eigen::Vector3f::Zero();
            for (const auto& e : evals)
            {
                const float w = std::exp(-(e[3] - best_loss) / beta);
                if (not std::isfinite(w) or w < 1e-6f) continue;
                const float dth = std::remainder(e[2] - ref_theta, 2.f * static_cast<float>(M_PI));
                mean += w * Eigen::Vector3f(e[0], e[1], dth);
                wsum += w; w2sum += w * w;
            }
            if (wsum < 1e-9f)
                return Eigen::Matrix3f::Identity() * 0.1f;   // degenerate: fall back to the old constant
            mean /= wsum;
            Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
            for (const auto& e : evals)
            {
                const float w = std::exp(-(e[3] - best_loss) / beta);
                if (not std::isfinite(w) or w < 1e-6f) continue;
                const float dth = std::remainder(e[2] - ref_theta, 2.f * static_cast<float>(M_PI));
                const Eigen::Vector3f d = Eigen::Vector3f(e[0], e[1], dth) - mean;
                cov += w * d * d.transpose();
            }
            cov /= wsum;
            // Floor the diagonal: a single surviving candidate gives an exactly-zero spread, which is a
            // statement of infinite confidence from one sample. Use the search's own resolution as the
            // smallest honest claim.
            const float pos_floor = 0.05f * 0.05f;
            const float rot_floor = (2.f * static_cast<float>(M_PI) / 180.f) * (2.f * static_cast<float>(M_PI) / 180.f);
            cov(0,0) = std::max(cov(0,0), pos_floor);
            cov(1,1) = std::max(cov(1,1), pos_floor);
            cov(2,2) = std::max(cov(2,2), rot_floor);
            ep.ess    = (w2sum > 0.f) ? (wsum * wsum / w2sum) : 1.f;
            ep.cov_xx = cov(0,0); ep.cov_yy = cov(1,1); ep.cov_tt = cov(2,2);
            return cov;
        };

        auto commit_pose_cov = [&](float x, float y, float theta, const Eigen::Matrix3f& cov)
        {
            model_->robot_pos = torch::tensor(
                {x, y},
                torch::TensorOptions().dtype(torch::kFloat32)
                    .device(get_device()).requires_grad(true));
            model_->robot_theta = torch::tensor(
                {theta},
                torch::TensorOptions().dtype(torch::kFloat32)
                    .device(get_device()).requires_grad(true));
            smoothed_pose_              = Eigen::Vector3f(x, y, theta);
            has_smoothed_pose_          = true;
            needs_orientation_search_   = false;
            tracking_step_count_        = 0;
            current_covariance          = cov;
        };
        // Legacy entry point: the fixed I*0.1 is retained ONLY for the paths that have no candidate set
        // to match against, and is documented as a guess rather than a measurement.
        auto commit_pose = [&](float x, float y, float theta)
        { commit_pose_cov(x, y, theta, Eigen::Matrix3f::Identity() * 0.1f); };

        // ── Room bounds ───────────────────────────────────────────────────────
        float min_x, max_x, min_y, max_y;
        if (model_->use_polygon && model_->polygon_vertices.defined())
        {
            auto verts_cpu = model_->polygon_vertices.to(torch::kCPU);
            auto acc = verts_cpu.accessor<float, 2>();
            min_x = max_x = acc[0][0];
            min_y = max_y = acc[0][1];
            for (int i = 1; i < verts_cpu.size(0); i++)
            {
                min_x = std::min(min_x, acc[i][0]);
                max_x = std::max(max_x, acc[i][0]);
                min_y = std::min(min_y, acc[i][1]);
                max_y = std::max(max_y, acc[i][1]);
            }
        }
        else
        {
            auto he_cpu = model_->half_extents.to(torch::kCPU);
            float hw = he_cpu[0].item<float>();
            float hh = he_cpu[1].item<float>();
            min_x = -hw; max_x = hw;
            min_y = -hh; max_y = hh;
        }
        const float margin = params.grid_search_wall_margin;
        min_x += margin; max_x -= margin;
        min_y += margin; max_y -= margin;

        // ── Subsampled point tensors ───────────────────────────────────────────
        // Coarse tensor (half budget) for Stages 0 and 1; fine (full) for Stage 2.
        auto make_tensor = [&](int max_pts) -> torch::Tensor {
            std::vector<Eigen::Vector3f> sample;
            const int stride = std::max(1, static_cast<int>(lidar_points.size()) / max_pts);
            sample.reserve(static_cast<size_t>(lidar_points.size() / stride + 1));
            for (size_t i = 0; i < lidar_points.size(); i += stride)
                sample.push_back(lidar_points[i]);
            return points_to_tensor_xyz(sample, get_device());
        };
        const torch::Tensor pts_coarse = make_tensor(params.grid_search_max_samples / 2);
        const torch::Tensor pts_fine   = make_tensor(params.grid_search_max_samples);

        // Success bar, DERIVED from the recovery trigger rather than set independently. The invariant
        // is that a search which reports success must leave a pose recovery will not immediately call
        // lost again — otherwise the two fight forever: recovery fires, the search "succeeds" without
        // improving anything, the counter clears, the cooldown expires and it fires again. That is
        // exactly what a standalone good_thr produced. Keeping it a FRACTION of the trigger makes the
        // relationship un-driftable; grid_search_good_factor only chooses how much margin.
        const float good_thr = params.grid_search_good_factor * params.recovery_loss_threshold;
        ep.good_thr = good_thr;

        // Best pose seen so far, seeded from the pose we were handed (Stage 0 fills it in). Carried
        // through every stage so the final commit is never a REGRESSION — the coarse 1 m grid does not
        // evaluate the current pose, so without an incumbent a search over an already-decent pose could
        // hand back a worse grid point.
        float incumbent_x = 0.f, incumbent_y = 0.f, incumbent_theta = 0.f;
        float incumbent_loss = std::numeric_limits<float>::infinity();

        // ══ STAGE 0: FULL ROTATION SWEEP at the current position ══════════════
        // Ordered by what actually goes wrong. A localizer that loses itself has almost always lost its
        // ORIENTATION — the 180° mirror of a near-symmetric room, or yaw bled away during a fast turn —
        // while the position stays roughly right, because translation is observable from wall distances
        // every frame whereas a flip is SDF-ambiguous and can persist silently. A large teleport is the
        // rare case, so paying for a full-room x/y lattice before trying rotation is backwards: it is
        // the expensive search for the unlikely fault.
        //
        // So: sweep yaw over the whole circle at the CURRENT x/y, then refine around the winner. This
        // subsumes the old four-cardinal-flip test (180/90/270 are simply members of the sweep) and
        // additionally catches the small-to-moderate yaw errors it could not see at all — the drift
        // that leaves the pose "nearly right" and the residual stubbornly high.
        float s0_best_theta = 0.f;
        {
            const auto cur  = model_->get_state();
            const float cx  = cur[2];
            const float cy  = cur[3];
            const float cth = cur[4];

            const float coarse_dth = static_cast<float>(M_PI) / 15.f;   // 12° — 30 evals over 360°
            std::vector<Eigen::Vector4f> sweep;   // (x, y, theta, loss), for the moment match below
            float best_loss  = std::numeric_limits<float>::infinity();
            float best_theta = cth;
            for (int i = 0; i < 30; ++i)
            {
                const float theta = cth + static_cast<float>(i) * coarse_dth;
                const float loss  = eval_loss(pts_coarse, cx, cy, theta);
                sweep.emplace_back(cx, cy, theta, loss);
                if (loss < best_loss) { best_loss = loss; best_theta = theta; }
            }
            // Refine the winner at 2° over ±12°, so a flip is recovered to better than the angular
            // resolution the coarse sweep could express.
            const float fine_dth = 2.f * static_cast<float>(M_PI) / 180.f;
            for (int i = -6; i <= 6; ++i)
            {
                const float theta = best_theta + static_cast<float>(i) * fine_dth;
                const float loss  = eval_loss(pts_fine, cx, cy, theta);
                sweep.emplace_back(cx, cy, theta, loss);
                if (loss < best_loss) { best_loss = loss; best_theta = theta; }
            }

            // The incumbent is the pose we were HANDED, always — so no later stage can return something
            // worse than the caller's own estimate.
            incumbent_x = cx; incumbent_y = cy; incumbent_theta = cth;
            incumbent_loss = eval_loss(pts_fine, cx, cy, cth);
            ep.incumbent_x = cx; ep.incumbent_y = cy; ep.incumbent_theta = cth;
            ep.incumbent_loss = incumbent_loss;

            s0_best_theta = best_theta;
            // Commit only on a real improvement that also clears the bar. Requiring the IMPROVEMENT is
            // what stops the search concluding on the pose it was given: if rotation cannot help, this
            // must fall through to the translational stages rather than report success unchanged.
            if (best_loss < good_thr and best_loss < incumbent_loss)
            {
                // Only THETA was searched here, so only theta's spread is evidence — a 360° yaw sweep
                // says nothing about position, so keep the x/y uncertainty we already had rather than
                // inventing one. In a near-symmetric room this reports a large sigma_theta instead of
                // silently picking one of two rival basins.
                Eigen::Matrix3f cov0 = current_covariance;
                cov0(2, 2) = moment_match(sweep, best_loss, best_theta)(2, 2);
                cov0(0, 2) = cov0(2, 0) = cov0(1, 2) = cov0(2, 1) = 0.f;
                ep.stage = 0; ep.best_x = cx; ep.best_y = cy; ep.best_theta = best_theta;
                ep.best_loss = best_loss; ep.success = true;
                ep.jump_rad = std::abs(std::remainder(best_theta - cth, 2.f * static_cast<float>(M_PI)));
                commit_pose_cov(cx, cy, best_theta, cov0);
                return true;
            }
            if (best_loss < incumbent_loss)
            {
                incumbent_theta = best_theta;
                incumbent_loss  = best_loss;
            }
        }

        // ══ STAGE 0b: LOCAL translation × rotation around the current pose ════
        // Rotation alone did not explain the scan. Before searching the whole room, try a small
        // neighbourhood — a modest position error combined with a yaw error is far more likely than a
        // teleport, and this covers it for ~1/8 the cost of the global lattice.
        {
            const float span = 1.0f, step = 0.25f;                          // ±1 m at 25 cm
            const float dth_span = 30.f * static_cast<float>(M_PI) / 180.f; // ±30°
            const float dth_step = 10.f * static_cast<float>(M_PI) / 180.f; // at 10°
            float best_x = incumbent_x, best_y = incumbent_y, best_theta = incumbent_theta;
            float best_loss = incumbent_loss;
            for (float dx = -span; dx <= span + 1e-4f; dx += step)
                for (float dy = -span; dy <= span + 1e-4f; dy += step)
                {
                    const float rx = incumbent_x + dx, ry = incumbent_y + dy;
                    if (rx < min_x or rx > max_x or ry < min_y or ry > max_y) continue;
                    for (float dth = -dth_span; dth <= dth_span + 1e-4f; dth += dth_step)
                    {
                        const float th   = s0_best_theta + dth;
                        const float loss = eval_loss(pts_coarse, rx, ry, th);
                        if (loss < best_loss)
                        { best_loss = loss; best_x = rx; best_y = ry; best_theta = th; }
                    }
                }
            if (best_loss < good_thr and best_loss < incumbent_loss)
            {
                ep.stage = 1; ep.best_x = best_x; ep.best_y = best_y; ep.best_theta = best_theta;
                ep.best_loss = best_loss; ep.success = true;
                ep.jump_m = std::hypot(best_x - ep.incumbent_x, best_y - ep.incumbent_y);
                ep.jump_rad = std::abs(std::remainder(best_theta - ep.incumbent_theta, 2.f * static_cast<float>(M_PI)));
                commit_pose(best_x, best_y, best_theta);
                return true;
            }
            if (best_loss < incumbent_loss)
            {
                incumbent_x = best_x; incumbent_y = best_y;
                incumbent_theta = best_theta; incumbent_loss = best_loss;
            }
        }

        // ══ STAGE 1: Coarse GLOBAL grid — last resort (≥1 m step, 90° angles) ══
        // Only reached when neither a pure rotation nor a ±1 m neighbourhood explains the scan, i.e.
        // the genuine kidnapping case. Evaluate the whole room at low resolution, keep the top-K.
        constexpr int TOP_K = 8;
        const float   coarse_step  = std::max(grid_resolution, 1.0f);
        const float   coarse_angle = static_cast<float>(M_PI_2);   // 90°

        struct Candidate { float x, y, theta, loss; };
        std::vector<Candidate> candidates;

        {
            std::vector<float> angles;
            for (float a = -static_cast<float>(M_PI); a < static_cast<float>(M_PI); a += coarse_angle)
                angles.push_back(a);

            int total = 0;
            for (float x = min_x; x <= max_x; x += coarse_step)
                for (float y = min_y; y <= max_y; y += coarse_step)
                    for (float theta : angles)
                    {
                        candidates.push_back({x, y, theta, eval_loss(pts_coarse, x, y, theta)});
                        ++total;
                    }

            std::sort(candidates.begin(), candidates.end(),
                      [](const Candidate& a, const Candidate& b){ return a.loss < b.loss; });
            if (static_cast<int>(candidates.size()) > TOP_K)
                candidates.resize(TOP_K);

            // Only stop here if the COARSE grid already clears the bar outright. On a 1 m/90° lattice
            // that is rare by construction, which is the point: the usual path is to fall through to
            // the fine refinement rather than commit a lattice point as if it were a solution.
            ep.topk_best  = candidates.front().loss;
            ep.topk_worst = candidates.back().loss;
            if (candidates.front().loss < good_thr)
            {
                const auto& b = candidates.front();
                ep.stage = 2; ep.best_x = b.x; ep.best_y = b.y; ep.best_theta = b.theta;
                ep.best_loss = b.loss; ep.success = true;
                ep.jump_m = std::hypot(b.x - ep.incumbent_x, b.y - ep.incumbent_y);
                ep.jump_rad = std::abs(std::remainder(b.theta - ep.incumbent_theta, 2.f * static_cast<float>(M_PI)));
                commit_pose(b.x, b.y, b.theta);
                return true;
            }
        }

        // ══ STAGE 2: Fine refinement around each top-K candidate ══════════════
        // Search a neighbourhood of ±coarse_step with step = coarse_step/3
        // and ±coarse_angle with step = coarse_angle/3 (≈30°), using the full
        // lidar budget.
        const float fine_pos   = coarse_step  / 3.f;
        const float fine_angle = coarse_angle / 3.f;

        // Start from the incumbent (the pose we were handed) when it still beats every coarse
        // candidate, so refinement can only improve on it — never trade a good pose for a lattice point.
        float best_x     = candidates.front().x;
        float best_y     = candidates.front().y;
        float best_theta = candidates.front().theta;
        float best_loss  = candidates.front().loss;
        std::vector<Eigen::Vector4f> refined;   // every pose Stage 2 evaluates, for the moment match
        if (incumbent_loss < best_loss)
        {
            best_x = incumbent_x; best_y = incumbent_y;
            best_theta = incumbent_theta; best_loss = incumbent_loss;
        }
        int   total2     = 0;

        for (const auto& cand : candidates)
        {
            for (float dx = -coarse_step; dx <= coarse_step + 1e-4f; dx += fine_pos)
            {
                const float rx = cand.x + dx;
                if (rx < min_x || rx > max_x) continue;
                for (float dy = -coarse_step; dy <= coarse_step + 1e-4f; dy += fine_pos)
                {
                    const float ry = cand.y + dy;
                    if (ry < min_y || ry > max_y) continue;
                    for (float da = -coarse_angle; da <= coarse_angle + 1e-4f; da += fine_angle)
                    {
                        const float loss = eval_loss(pts_fine, rx, ry, cand.theta + da);
                        refined.emplace_back(rx, ry, cand.theta + da, loss);
                        if (loss < best_loss)
                        {
                            best_loss  = loss;
                            best_x     = rx;
                            best_y     = ry;
                            best_theta = cand.theta + da;
                        }
                        ++total2;
                    }
                }
            }
        }

        // Commit with the SPREAD of the refined candidate set, not a constant. A decisive winner gives
        // a tight covariance; a flat valley — the corridor's along-axis direction, or a near-symmetric
        // room — gives a large one along exactly the unresolved DOF. That is the honest answer, and it
        // is what stops the caller treating an ambiguous result as a fix and thrashing on it.
        ep.stage = 3; ep.best_x = best_x; ep.best_y = best_y; ep.best_theta = best_theta;
        ep.best_loss = best_loss; ep.success = best_loss < good_thr;
        ep.jump_m = std::hypot(best_x - ep.incumbent_x, best_y - ep.incumbent_y);
        ep.jump_rad = std::abs(std::remainder(best_theta - ep.incumbent_theta, 2.f * static_cast<float>(M_PI)));
        commit_pose_cov(best_x, best_y, best_theta, moment_match(refined, best_loss, best_theta));
        return best_loss < good_thr;
    }

    void RoomConcept::set_initial_state(float width, float length, float x, float y, float phi)
    {
        model_ = std::make_shared<Model>();
        model_->set_device(get_device());  // Set device before init
        model_->init_from_state(width, length, x, y, phi, params.wall_height);
        needs_orientation_search_ = true;  // Will search for best orientation on first update
        has_smoothed_pose_ = false;  // Reset smoothing
        tracking_step_count_ = 0;  // Reset early exit tracking
        prediction_early_exits_ = 0;
        current_velocity_weights_ = Eigen::Vector3f::Ones();  // Reset velocity weights
        prev_sdf_mse_ = 0.f;  // Reset boundary quality gate
        u_b_init_ = false;    // reseed hierarchical boundary log-precision to g(v) after a reset
        window_mgr_.clear(); reset_stride_state();
        rerun_room_polygon_sent_ = false;

    }


    void RoomConcept::set_polygon_room(const std::vector<Eigen::Vector2f>& polygon_vertices)
    {
        if (polygon_vertices.size() < 3)
        {
            std::cerr << "set_polygon_room: Need at least 3 vertices" << std::endl;
            return;
        }

        // Vertices are in room frame (where user clicked on the viewer)
        // Keep current robot pose if we have one, otherwise start at origin
        float init_x = 0.0f;
        float init_y = 0.0f;
        float init_phi = 0.0f;

        if (model_ != nullptr and model_->has_state())
        {
            // Preserve current robot pose. has_state(), not just non-null: the previous model may
            // have been allocated and never filled, and get_state() would throw here.
            const auto state = model_->get_state();
            init_x = state[2];
            init_y = state[3];
            init_phi = state[4];
        }

        model_ = std::make_shared<Model>();
        model_->set_device(get_device());  // Set device before init
        model_->init_from_polygon(polygon_vertices, init_x, init_y, init_phi, params.wall_height);

        // Reset state but keep covariance reasonable
        last_lidar_timestamp = 0;
        last_update_result = UpdateResult{};
        current_covariance = Eigen::Matrix3f::Identity() * 0.1f;
        needs_orientation_search_ = true;  // Will search for best orientation on first update
        has_smoothed_pose_ = false;  // Reset smoothing
        tracking_step_count_ = 0;  // Reset early exit tracking
        prediction_early_exits_ = 0;
        current_velocity_weights_ = Eigen::Vector3f::Ones();  // Reset velocity weights
        prev_sdf_mse_ = 0.f;  // Reset boundary quality gate
        u_b_init_ = false;    // reseed hierarchical boundary log-precision to g(v) after a reset
        window_mgr_.clear(); reset_stride_state();
        rerun_room_polygon_sent_ = false;

        // Initialize corner detector with model polygon + graded-covariance tuning from config.
        {
            auto& cp = corner_detector_.params();
            cp.wall_band      = params.corner_wall_band;
            cp.base_sigma     = params.corner_base_sigma;
            cp.orient_tau_deg = params.corner_orient_tau_deg;
            cp.merge_chi2         = params.corner_merge_chi2;
            cp.merge_prior_sigma  = params.corner_merge_prior_sigma;
            cp.min_wall_map_sigmas  = params.corner_min_wall_map_sigmas;
            cp.min_yield_map_sigmas = params.corner_min_yield_map_sigmas;
            cp.yield_leak           = params.corner_yield_leak;
            cp.yield_warmup         = params.corner_yield_warmup;
            cp.yield_release_factor = params.corner_yield_release_factor;
        }
        corner_detector_.set_model_corners(polygon_vertices);

        // Report the landmark exclusion once, with the geometry that justified it — a silent drop of
        // model corners is exactly the kind of thing that later reads as "the detector is broken".
        if (const auto& dropped = corner_detector_.short_wall_dropped_indices(); !dropped.empty())
        {
            const int N = static_cast<int>(polygon_vertices.size());
            QStringList detail;
            for (const int i : dropped)
            {
                const float w_in  = (polygon_vertices[i] - polygon_vertices[(i + N - 1) % N]).norm();
                const float w_out = (polygon_vertices[(i + 1) % N] - polygon_vertices[i]).norm();
                detail << QString("v%1(%2/%3m)").arg(i).arg(w_in, 0, 'f', 3).arg(w_out, 0, 'f', 3);
            }
            qInfo().noquote()
                << "[corners]" << dropped.size() << "of" << N
                << "polygon vertices refused LANDMARK status: adjacent walls shorter than"
                << params.corner_min_wall_map_sigmas * corner_detector_.params().map_sigma
                << "m, which the traced layout cannot assert. They remain in the polygon and the SDF."
                << "\n           " << detail.join(' ');
        }

        // A fresh polygon invalidates every per-vertex tally (indices may not even mean the same thing).
        corner_vertex_stats_.clear();
        corner_stats_frames_ = 0;
    }

    void RoomConcept::set_robot_pose(float x, float y, float theta, bool manual_reset)
    {
        if (model_ == nullptr)
        {
            qWarning() << "Cannot set robot pose: model not initialized";
            return;
        }

        // Directly update robot pose tensors
        model_->robot_pos = torch::tensor({x, y}, torch::TensorOptions().dtype(torch::kFloat32).device(model_->device_).requires_grad(true));
        model_->robot_theta = torch::tensor({theta}, torch::TensorOptions().dtype(torch::kFloat32).device(model_->device_).requires_grad(true));

        // Reset smoothed pose to new position
        smoothed_pose_ = Eigen::Vector3f(x, y, theta);
        has_smoothed_pose_ = true;

        // Reset covariance to reasonable value (we're uncertain after manual placement)
        current_covariance = Eigen::Matrix3f::Identity() * 0.1f;

        // Reset tracking counters and prediction state
        tracking_step_count_ = 0;
        needs_orientation_search_ = false;  // User explicitly set orientation
        last_lidar_timestamp = 0;  // Force fresh start

        // Optional skip optimization for a few frames to let manual reset settle.
        manual_reset_frames_ = manual_reset ? params.manual_reset_skip_frames : 0;

        // Clear any previous prediction
        model_->has_prediction = false;
        model_->robot_prev_pose = std::nullopt;

        // Clear sliding window (stale poses are invalid after manual reset)
        window_mgr_.clear(); reset_stride_state();

        // Reset last update result with new pose
        last_update_result = UpdateResult{};
        last_update_result.robot_pose.translation() = Eigen::Vector2f(x, y);
        last_update_result.robot_pose.linear() = Eigen::Rotation2Df(theta).toRotationMatrix();
        last_update_result.ok = true;
    }

    float RoomConcept::evaluate_pose_fit(const std::vector<Eigen::Vector3f>& lidar_points,
                                                  int max_samples) const
    {
        if (model_ == nullptr || lidar_points.empty())
            return std::numeric_limits<float>::infinity();

        std::vector<Eigen::Vector3f> sampled;
        const int cap = std::max(1, max_samples);
        if (static_cast<int>(lidar_points.size()) > cap)
        {
            const int stride = static_cast<int>(lidar_points.size()) / cap;
            sampled.reserve(cap);
            for (size_t i = 0; i < lidar_points.size(); i += stride)
                sampled.push_back(lidar_points[i]);
        }
        else
        {
            sampled = lidar_points;
        }

        torch::NoGradGuard no_grad;
        const auto points_tensor = points_to_tensor_xyz(sampled, get_device());
        // Median, not mean: this result is compared against recovery_loss_threshold (validate_seed_pose)
        // and against itself across yaw hypotheses, and every other fit number in this class is a
        // median. Mean |SDF| runs high wherever furniture the polygon does not model is in view, which
        // made the seed check stricter than the bar it was measured against.
        return median_abs_sdf(model_->sdf(points_tensor));
    }

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

    RoomConcept::UpdateResult RoomConcept::update(
                const LidarData &lidar,
                const std::vector<VelocityCommand> &velocity_history,
                const std::vector<OdometryReading> &odometry_history)
    {
        t_update_start_ = std::chrono::high_resolution_clock::now();
        UpdateResult res;
        if (lidar.first.empty() || model_ == nullptr)
            return res;

        res.lidar_scan = lidar.first;   // store scan for synchronized visualization

        // ===== MANUAL RESET: skip optimization during settle period =====
        if (manual_reset_frames_ > 0)
        {
            manual_reset_frames_--;
            const auto state = model_->get_state();
            res.ok = true;
            res.state = state;
            res.robot_pose.translation() = Eigen::Vector2f(state[2], state[3]);
            res.robot_pose.linear() = Eigen::Rotation2Df(state[4]).toRotationMatrix();
            res.covariance = current_covariance;
            res.timestamp_ms = lidar.second;
            last_update_result = res;
            last_lidar_timestamp = lidar.second;
            return res;
        }

        // ===== ORIENTATION SEARCH ON FIRST UPDATE =====
        if (needs_orientation_search_)
        {
            const auto state = model_->get_state();
            const float best_phi = find_best_initial_orientation(lidar.first, state[2], state[3], state[4]);
            model_->robot_theta.data().copy_(torch::tensor({best_phi},
                torch::TensorOptions().device(get_device())));
            needs_orientation_search_ = false;
        }

        // ===== PREDICTION =====
        const auto motion_prior_selection = build_motion_prior_selection(velocity_history, odometry_history, lidar);
        const auto &selected_prior = motion_prior_selection.selected_prior;
        const PredictionState prediction = predict_step(model_, selected_prior, true);

        if (prediction.have_propagated && prediction.propagated_cov.defined())
        {
            auto cov_cpu = prediction.propagated_cov.to(torch::kCPU);
            auto cov_acc = cov_cpu.accessor<float, 2>();
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    current_covariance(i, j) = cov_acc[i][j];
        }

        const auto pred_pos = motion_prior_selection.predicted_pos;
        const auto pred_theta = motion_prior_selection.predicted_theta;
        last_pred_pos_ = pred_pos;      // kept for UpdateResult on BOTH return paths
        last_pred_theta_ = pred_theta;

        // ===== BUILD NEW WINDOW SLOT =====
        const auto& all_points = lidar.first;
        std::vector<Eigen::Vector3f> sampled_points;
        if (static_cast<int>(all_points.size()) > params.max_lidar_points)
        {
            const int stride = static_cast<int>(all_points.size()) / params.max_lidar_points;
            sampled_points.reserve(params.max_lidar_points);
            for (size_t i = 0; i < all_points.size(); i += stride)
                sampled_points.push_back(all_points[i]);
        }
        else
            sampled_points = all_points;

        const torch::Tensor points_tensor = points_to_tensor_xyz(sampled_points, get_device());
        tracking_step_count_++;

        // Motion constraint for the new slot.
        // Prefer measured odometry (encoder/IMU) when available — it is more accurate than
        // the commanded velocity, especially during rotation where wheel slip can differ from
        // the commanded angular rate.  The initial pose (pred_pos, pred_theta) was already
        // computed from the fused estimate inside apply_dual_prior_fusion, so using the same
        // measured delta here removes the command/measured inconsistency that previously caused
        // Adam to fight the motion factor during turns.
        Eigen::Vector3f slot_odom_delta = Eigen::Vector3f::Zero();
        Eigen::Matrix3f slot_motion_cov = Eigen::Matrix3f::Identity() * params.default_slot_motion_cov;
        if (selected_prior.valid)
        {
            slot_odom_delta = selected_prior.delta_pose;
            slot_motion_cov = selected_prior.covariance_eigen;
        }

        // ---- Strided window: decide ADMIT vs REPLACE, and span the motion factor accordingly -------
        // See Params::window_stride_enabled. The per-frame delta above describes ONE inter-frame gap.
        // While we are replacing the newest slot instead of appending, the motion factor that slot
        // carries must describe the whole interval back to the last ADMITTED slot — otherwise the
        // window's oldest-to-newest chain is constrained by a fraction of the motion that actually
        // happened, which under-constrains exactly the DOF this change exists to fix.
        bool stride_replace = false;
        if (params.window_stride_enabled)
        {
            stride_delta_accum_ += slot_odom_delta;       // global-frame increments, additive
            stride_cov_accum_   += slot_motion_cov;       // independent increments

            // Preintegrated form of the same accumulation. `+=` on the covariance drops the transport
            // term — an error in the heading accumulated so far rotates ALL the translation that
            // follows — and with window_min_turn_rad = 0.15 rad that term is not small. chain() is the
            // same recursion the per-sample loop uses, applied at frame granularity.
            if (params.motion_preintegration and selected_prior.has_preint)
                stride_preint_accum_ = rc::preint::chain(stride_preint_accum_, selected_prior.preint);

            if (stride_has_admitted_ and window_mgr_.size() > 1)
            {
                const float travel = std::hypot(pred_pos.x() - stride_last_admitted_[0],
                                                pred_pos.y() - stride_last_admitted_[1]);
                const float turn = std::abs(std::remainder(pred_theta - stride_last_admitted_[2],
                                                           2.f * static_cast<float>(M_PI)));
                stride_replace = (travel < params.window_min_travel_m
                                  and turn < params.window_min_turn_rad);
            }
            // Either way the newest slot spans back to the last admitted one.
            slot_odom_delta = stride_delta_accum_;
            slot_motion_cov = stride_cov_accum_;
            // covariance() is called exactly HERE, once, on the interval the motion factor will
            // actually carry — never per frame (see the warning on Interval::covariance()).
            if (params.motion_preintegration and stride_preint_accum_.samples > 0)
                slot_motion_cov = stride_preint_accum_.covariance();
        }

        // Mirror what this slot's motion factor actually got, for BOTH debug-log writers (the
        // early-exit one is in another function and can only see members).
        last_slot_motion_cov_ = slot_motion_cov;
        if (params.motion_preintegration and params.window_stride_enabled and stride_preint_accum_.samples > 0)
        {
            last_preint_samples_    = stride_preint_accum_.samples;
            last_preint_duration_s_ = stride_preint_accum_.duration_s;
        }
        else if (params.motion_preintegration and selected_prior.has_preint)
        {
            last_preint_samples_    = selected_prior.preint.samples;
            last_preint_duration_s_ = selected_prior.preint.duration_s;
        }
        else { last_preint_samples_ = 0; last_preint_duration_s_ = 0.f; }

        WindowSlot new_slot;
        new_slot.pose = torch::tensor({pred_pos.x(), pred_pos.y(), pred_theta},
            torch::TensorOptions().dtype(torch::kFloat32).device(get_device()).requires_grad(true));
        new_slot.lidar_points = points_tensor;
        new_slot.odometry_delta = slot_odom_delta;
        new_slot.motion_cov = slot_motion_cov;
        new_slot.timestamp_ms = lidar.second;
        // ── THE CONNECTION, missing until 2026-08-26 ──────────────────────────────────────────────
        // pump_image_edges() has been extracting contours and calling set_image_edges() every tick,
        // and NOTHING read them: `image_edges()` had zero call sites and `new_slot.image_edges` was
        // never assigned, so Slot::image_edges was empty for every slot ever built. Every consumer
        // guards on `obs.empty()` / `image_edges.empty()` and so all of them silently did nothing —
        // the shadow monitor, the driving factor, and both loss terms. `drive = true` would have
        // been inert too. A feature that is enabled, extracting, and disconnected looks exactly like
        // a feature that is working and has nothing to say.
        if (params.image_edge.enable)
            new_slot.image_edges = take_image_edges();

        // Pre-cache tensors used every Adam iteration.
        // Always build on CPU first (accessor<> requires CPU), then move to device.
        new_slot.odom_delta_tensor = torch::tensor(
            {slot_odom_delta[0], slot_odom_delta[1], slot_odom_delta[2]},
            torch::kFloat32).to(get_device());
        {
            Eigen::Matrix3f prec = slot_motion_cov.inverse();
            auto prec_cpu = torch::zeros({3, 3}, torch::kFloat32);
            auto acc = prec_cpu.accessor<float, 2>();
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    acc[r][c] = prec(r, c);
            new_slot.motion_prec_tensor = prec_cpu.to(get_device());
        }
        // ── The rest hypothesis, as a constraint on the STATE ────────────────────────────────────
        // A zero-delta motion factor between this slot and the last: "you did not move", with the
        // interval's own precision. The solver then weighs it against "you moved by Delta" and the
        // MEAN goes where the evidence says — which the covariance-shaping form could never do,
        // because the preintegrator never writes the mean.
        if (params.zupt_as_factor and selected_prior.has_preint)
        {
            const Eigen::Matrix3f R = selected_prior.preint.zupt_covariance();
            if (R(0, 0) > 0.f and R(2, 2) > 0.f)
            {
                const Eigen::Matrix3f prec_z = R.inverse();
                auto z_cpu = torch::zeros({3, 3}, torch::kFloat32);
                auto acc_z = z_cpu.accessor<float, 2>();
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 3; c++) acc_z[r][c] = prec_z(r, c);
                new_slot.zupt_prec_tensor = z_cpu.to(get_device());
            }
        }

        bool window_slid = false;
        if (stride_replace)
        {
            // Not enough motion to be worth a slot: overwrite the newest so the current frame is still
            // represented (and still optimized) without consuming the window's span. Nothing is
            // marginalized and nothing is dropped, so the older slots keep their separation.
            window_mgr_.newest() = std::move(new_slot);
        }
        else
        {
            // FEJ+Schur: marginalize the dropping slot BEFORE append() pops it, while both x₀ (front)
            // and its Markov blanket x₁ (window[1]) are still live at their converged linearization
            // points. append() then just pops (fej_schur=true suppresses the legacy mu re-anchoring),
            // and the post-optimization recompute_boundary_prior() below is skipped.
            if (params.boundary_fej_schur)
                window_mgr_.marginalize_oldest(*model_, params, get_device());

            window_slid = window_mgr_.append(std::move(new_slot), params.rfe_window_size,
                                             params.boundary_mu_quality_threshold,
                                             params.boundary_fej_schur);
            if (params.window_stride_enabled)
            {
                stride_last_admitted_ = Eigen::Vector3f(pred_pos.x(), pred_pos.y(), pred_theta);
                stride_has_admitted_  = true;
                stride_delta_accum_.setZero();
                stride_cov_accum_.setZero();
                stride_preint_accum_ = rc::preint::Interval{};
            }
        }
        window_mgr_.subsample_old_slots(params.rfe_max_lidar_per_old_slot);

        // ===== CORNER DETECTION (optional, controlled by EnableCornerTracking) =====
        if (params.enable_corner_tracking && !init_polygon_vertices_.empty())
        {
            auto newest_cpu = window_mgr_.newest().pose.detach().to(torch::kCPU);
            auto pa = newest_cpu.accessor<float, 1>();
            const float cx = pa[0], cy = pa[1], cth = pa[2];

            // current_covariance feeds the association gate: a poorly-localized robot associates
            // permissively, a sharply-localized one refuses a neighbouring corner outright.
            auto det = corner_detector_.detect(all_points, cx, cy, cth, current_covariance);
            res.corners_in_fov = det.corners_in_fov;
            res.corner_matches = det.matches;

            // Per-model-corner attribution → etc/corner_stats.csv. This is what decides whether a
            // given pillar earns its landmark status or should follow the trace artefacts out of
            // set_model_corners; the aggregate rej_* counters cannot say WHICH corner is at fault.
            if (params.corner_stats_csv)
                accumulate_corner_stats(det);

            // Acceptance-rate diagnostic removed from the terminal (2026-07-25). DetectionResult still
            // CARRIES every number it printed (corners_in_fov/detected/accepted, rej_*, soft_orient,
            // convex agree, merged_coincident, model_dup_dropped) — re-enabling is a print, not a re-derivation.
            // The residual-distribution line and the per-corner AMBIGUOUS dump that lived here were
            // one-off instrumentation for setting map_sigma from data (2026-07-20) and are removed now
            // that it is set. DetectionResult still CARRIES every one of those numbers — resid_mean/max,
            // resid_chi2_mean, corners_with_rival, runnerup_chi2_mean, min_assoc_prob, and per-match
            // assoc_prob / assoc_chi2_val / runnerup_chi2 — so re-enabling is a print, not a re-derivation.
            // (They were also mis-scoped: only the print above was inside the %20 throttle, so those two
            // fired EVERY frame.)

            // Store corner observations in the newest slot for the RFE loss
            auto& newest_slot = window_mgr_.newest();
            newest_slot.corner_obs.clear();
            for (const auto& m : det.matches)
            {
                if (m.model_index < 0 || m.model_index >= static_cast<int>(init_polygon_vertices_.size()))
                    continue;
                // Retired by the information-yield rule: still detected, still drawn (as retired),
                // but it must not vote. This is the ONLY place retirement touches inference.
                if (m.suppressed)
                    continue;
                // Contract check at the OPTIMIZER BOUNDARY (independent of the detector's own check —
                // this is the line the Hessian is actually built from). A non-finite observation, or a
                // zero-precision one, must never enter: it constrains nothing yet still participates in
                // the Hessian assembly, which is how min_ev → 0 (cond_num sentinel 1e8) produced NaN
                // losses and a NaN pose on 2026-07-21. Dropping it degrades to SDF-only, which is the
                // intended cascade.
                if (not m.detected.allFinite() or not m.information.allFinite())
                    continue;
                if (m.information.cwiseAbs().maxCoeff() <= 1e-9f)
                    continue;
                WindowSlot::CornerObs obs;
                obs.model_corner_world = init_polygon_vertices_[m.model_index];
                obs.detected_robot = m.detected;
                obs.information = m.information;   // graded Λ_det (robot frame) — used anisotropically by the loss
                newest_slot.corner_obs.push_back(obs);
            }
            // Stack the pose-independent corner constants ONCE so the optimizer closure evaluates the
            // corner factor as batched ops instead of O(corners) tiny per-iteration tensor allocations.
            newest_slot.rebuild_corner_batch(get_device());
        }

        // Store validated object anchors (from the graph, set on the main thread) in the newest
        // slot so the RFE loss can add them as SE(2) pose-landmark factors.
        if (params.object_anchor.enable)
        {
            std::scoped_lock lk(object_anchors_mutex_);
            window_mgr_.newest().object_anchors = latest_object_anchors_;
        }

        // ===== EARLY EXIT CHECK =====
        if (auto early = try_prediction_early_exit(points_tensor, slot_odom_delta, selected_prior, lidar.second))
        {
            // Store quality for future boundary prior gate (early exit = good pose).
            window_mgr_.newest().sdf_mse_final = early->sdf_mse;
            // Motion-model adaptation runs here too so prediction-mode frames count.
            service_calibration();
            early->corners_in_fov = res.corners_in_fov;
            early->corner_matches = std::move(res.corner_matches);
            early->lidar_scan = std::move(res.lidar_scan);
            // The MOUNT monitor runs on early-exit cycles too — pass 1 only, no second solve. This
            // is where ~99.75% of cycles end, so leaving it below the return starved the instrument
            // exactly in proportion to how good the prediction had become.
            if (params.image_edge.enable and params.image_edge_shadow and not params.image_edge.drive)
                run_image_edge_shadow(read_window_poses(), boundary_weight_now(), lidar.second,
                                      /*probe_pose=*/false);
            return *early;
        }

        // ===== OPTIMISATION =====
        {
            // Capture slot poses before optimisation for debug log (pose persistence check)
            if (params.debug_log_enabled)
            {
                std::string s;
                for (size_t i = 0; i < window_mgr_.window.size(); ++i)
                {
                    if (i > 0) s += '|';
                    auto cpu = window_mgr_.window[i].pose.detach().to(torch::kCPU);
                    auto a = cpu.accessor<float, 1>();
                    std::ostringstream ss;
                    ss << std::fixed << std::setprecision(4)
                       << a[0] << ';' << a[1] << ';' << a[2];  // ';' avoids CSV column shift
                    s += ss.str();
                }
                slot_poses_pre_ = s.empty() ? "na" : s;
            }

            // The GN shadow must start from the state the authoritative backend started from, so
            // snapshot it BEFORE the solve. Free (a handful of float copies) when shadowing is off.
            const bool gn_shadow_this_frame = params.gn_shadow and params.optimizer_type != "GN";
            std::vector<Eigen::Vector3f> poses_before;
            if (gn_shadow_this_frame) poses_before = read_window_poses();

            const auto t0 = std::chrono::high_resolution_clock::now();
            auto [last_loss, iterations] =
                  (params.optimizer_type == "GN")    ? run_gn_loop(selected_prior)
                : (params.optimizer_type == "LBFGS") ? run_lbfgs_loop(selected_prior)
                                                     : run_adam_loop(selected_prior);
            last_t_adam_ms_ = std::chrono::duration<float, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();

            if (gn_shadow_this_frame)
                run_gn_shadow(poses_before, read_window_poses(), last_loss, iterations,
                              last_t_adam_ms_, lidar.second);

            // RGB edge shadow: evaluated + logged, pose untouched. Inert unless enable && shadow,
            // and skipped entirely once the term is DRIVING (there would be nothing to compare to).
            if (params.image_edge.enable and params.image_edge_shadow and not params.image_edge.drive)
                run_image_edge_shadow(read_window_poses(), boundary_weight_now(), lidar.second,
                                      /*probe_pose=*/true);
            res.final_loss     = last_loss;
            res.iterations_used = iterations;

            // Capture slot poses after Adam for debug log
            if (params.debug_log_enabled)
            {
                std::string s;
                for (size_t i = 0; i < window_mgr_.window.size(); ++i)
                {
                    if (i > 0) s += '|';
                    auto cpu = window_mgr_.window[i].pose.detach().to(torch::kCPU);
                    auto a = cpu.accessor<float, 1>();
                    std::ostringstream ss;
                    ss << std::fixed << std::setprecision(4)
                       << a[0] << ';' << a[1] << ';' << a[2];  // ';' avoids CSV column shift
                    s += ss.str();
                }
                slot_poses_post_ = s.empty() ? "na" : s;
            }
        }

        // ===== FE TERM BREAKDOWN (diagnostic log, no gradient needed) =====
        // Compute only every 5 frames to avoid re-running the full forward pass just for logging.
        // This saves ~4 ms per Adam run (≈5% of spike duration) at the cost of slightly stale
        // per-term breakdown values in the CSV.
        if (params.debug_log_enabled && (tracking_step_count_ % 5 == 0)) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            torch::NoGradGuard no_grad;
            last_loss_breakdown_ = window_mgr_.compute_rfe_loss_breakdown(*model_, params, get_device());
            last_t_breakdown_ms_ = std::chrono::duration<float, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
        } else {
            last_t_breakdown_ms_ = 0.f;
        }

        // ===== COVARIANCE UPDATE =====
        {
            const auto t0 = std::chrono::high_resolution_clock::now();
            auto [covariance, condition_number] = compute_posterior_covariance(points_tensor);
            res.covariance = covariance;
            res.condition_number = condition_number;
            hess_pre_adaptive_ = res.covariance;   // before apply_adaptive_covariance floors it
            last_t_cov_ms_ = std::chrono::duration<float, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
        }

        // ===== EXTRACT RESULT FROM NEWEST SLOT =====
        res.ok = true;

        {
            auto newest_cpu = window_mgr_.newest().pose.detach().to(torch::kCPU);
            auto p_acc = newest_cpu.accessor<float, 1>();
            float x = p_acc[0], y = p_acc[1], phi = p_acc[2];

            // ── DIVERGENCE RECOVERY: evidence is a cascade, never a cliff ────────────────────────
            // Corners are OPTIONAL evidence: none matched simply means the SDF term drives alone, and
            // that already happens naturally (loss_corner = 0). The failure that must never propagate
            // is the SDF optimization itself going non-finite — observed 2026-07-21: a singular Hessian
            // (cond_num hit its 1e8 sentinel) produced NaN losses, NaN was copied into the window pose
            // here, and from then on EVERY frame was NaN: the robot vanished from the canvas, the RT
            // edge published NaN to every peer, and corner detection silently reported in_fov=32 /
            // fewpoints=32 because NaN fails every comparison. Nothing recovered on its own.
            //
            // So: if the optimizer's output is not finite, drop one rung down the evidence ladder and
            // dead-reckon on odometry instead. The window slot is rewritten with the fallback so the
            // NaN cannot persist into the next frame's linearization point, and the covariance is
            // inflated because a dead-reckoned pose genuinely IS less certain — downstream consumers
            // then see the uncertainty grow rather than a confident lie.
            if (not std::isfinite(x) or not std::isfinite(y) or not std::isfinite(phi))
            {
                const bool have_odom = model_->has_prediction
                                       and std::isfinite(model_->predicted_pos[0].item<float>())
                                       and std::isfinite(model_->predicted_pos[1].item<float>())
                                       and std::isfinite(model_->predicted_theta[0].item<float>());
                if (have_odom)   // rung 2: odometry dead-reckoning
                {
                    x   = model_->predicted_pos[0].item<float>();
                    y   = model_->predicted_pos[1].item<float>();
                    phi = model_->predicted_theta[0].item<float>();
                }
                else if (last_good_pose_valid_)   // rung 3: hold the last pose we trusted
                {
                    x = last_good_pose_[0]; y = last_good_pose_[1]; phi = last_good_pose_[2];
                }
                else { x = 0.f; y = 0.f; phi = 0.f; }

                // Overwrite the poisoned slot so the next optimization starts from a finite point.
                window_mgr_.newest().pose.data().copy_(torch::tensor({x, y, phi},
                    torch::TensorOptions().device(window_mgr_.newest().pose.device())));
                // Dead reckoning ⇒ uncertainty grows. Keep it finite and clearly worse than a fix.
                if (not current_covariance.allFinite())
                    current_covariance = Eigen::Matrix3f::Identity() * 0.5f;
                else
                    current_covariance *= 4.0f;
                res.diverged = true;
                std::println("[SAFETY] SDF optimization produced a NON-FINITE pose — falling back to {} "
                             "(x={:.3f} y={:.3f} th={:.3f}); covariance inflated, corners bypassed",
                             have_odom ? "ODOMETRY" : (last_good_pose_valid_ ? "LAST GOOD POSE" : "ORIGIN"),
                             x, y, phi);
            }
            else
            {
                last_good_pose_ = Eigen::Vector3f{x, y, phi};
                last_good_pose_valid_ = true;
            }

            while (phi > M_PI) phi -= 2.0f * M_PI;
            while (phi < -M_PI) phi += 2.0f * M_PI;

            model_->robot_pos.data().copy_(torch::tensor({x, y},
                torch::TensorOptions().device(get_device())));
            model_->robot_theta.data().copy_(torch::tensor({phi},
                torch::TensorOptions().device(get_device())));

            res.sdf_mse = compute_sdf_median_abs(points_tensor, *model_);

            // Store localization quality so future frames can quality-gate the boundary prior
            // (Solutions B & C): when this slot becomes the oldest it carries its own sdf_mse.
            window_mgr_.newest().sdf_mse_final = res.sdf_mse;

            // ===== ONLINE MOTION MODEL ADAPTATION =====
            // Runs here so the newest slot's sdf_mse_final is valid for the quality gate.
            service_calibration();

            // Build state vector without calling get_state() (avoids 3 GPU→CPU transfers)
            {
                auto ext_cpu = model_->half_extents.to(torch::kCPU);
                auto ext = ext_cpu.accessor<float, 1>();
                res.state << 2.f * ext[0], 2.f * ext[1], x, y, phi;
            }

            // ── FLIP INSTRUMENTATION ──────────────────────────────────────────────────────────────
            // Detect a ~180° yaw jump vs the last optimized pose and dump WHY the optimizer chose it:
            //  • SDF at old vs new pose — Δ≈0 ⇒ the walls are INDIFFERENT (symmetric room), so the SDF
            //    can't prevent the flip; new≪old ⇒ the flip genuinely fits the walls better (old was wrong).
            //  • the FE terms at the settled pose — is corner/object DISAGREEING with the flip (high) but
            //    outweighed, or also indifferent (low)?  That says whether anything can break the tie.
            if (flip_prev_valid_)
            {
                const float dth = std::remainder(phi - flip_prev_th_, 2.0f * static_cast<float>(M_PI));
                if (std::abs(dth) > 2.0f)   // ~115°+ between consecutive optimized poses = a flip
                {
                    torch::NoGradGuard ng;
                    const auto opt = torch::TensorOptions().device(get_device());
                    auto sdf_at = [&](float px, float py, float pth) {
                        auto xy = torch::tensor({px, py}, opt);
                        auto th = torch::tensor({pth}, opt);
                        return torch::mean(torch::abs(model_->sdf_at_pose(points_tensor, xy, th))).item<float>();
                    };
                    const float sdf_old = sdf_at(flip_prev_x_, flip_prev_y_, flip_prev_th_);
                    const float sdf_new = sdf_at(x, y, phi);
                    // FRESH anchor whitened distance (σ) at old vs new pose: does the anchor SEE the flip?
                    // huge new / small old ⇒ the anchor discriminates strongly (should have vetoed it);
                    // both ~0 ⇒ the anchor isn't engaging (empty slot / wrong pin).
                    float anc_old = 0.f, anc_new = 0.f;
                    int   nanc = static_cast<int>(window_mgr_.newest().object_anchors.size());
                    for (const auto& a : window_mgr_.newest().object_anchors)
                    {
                        auto whit = [&](float px, float py, float pth) {
                            const float c = std::cos(pth), s = std::sin(pth);
                            const float dx = a.pose_world.x() - px, dy = a.pose_world.y() - py;
                            const float rx = a.obs_robot.x() - ( c * dx + s * dy);
                            const float ry = a.obs_robot.y() - (-s * dx + c * dy);
                            return std::sqrt(std::max(0.f,
                                rx * (a.information(0,0)*rx + a.information(0,1)*ry)
                              + ry * (a.information(1,0)*rx + a.information(1,1)*ry)));
                        };
                        anc_old = std::max(anc_old, whit(flip_prev_x_, flip_prev_y_, flip_prev_th_));
                        anc_new = std::max(anc_new, whit(x, y, phi));
                    }
                    // FRESH FE breakdown at the settled (flipped) pose (not the stale 5-frame cache).
                    const auto bd = window_mgr_.compute_rfe_loss_breakdown(*model_, params, get_device());
                    std::print("[FLIP] dyaw={:+.0f}deg dxy=({:+.2f},{:+.2f}) | SDF old={:.3f} new={:.3f} | "
                               "anchorSD old={:.1f} new={:.1f} (n={}) | FE obs={:.2f} corner={:.2f} object={:.2f} | iters={}\n",
                               dth * 57.2958f, x - flip_prev_x_, y - flip_prev_y_, sdf_old, sdf_new,
                               anc_old, anc_new, nanc, bd.obs, bd.corner, bd.object, res.iterations_used);

                    // Also log one row per flip to a dedicated CSV (independent of DebugLog).
                    if (!flip_csv_open_attempted_)
                    {
                        flip_csv_open_attempted_ = true;
                        ::mkdir("tmp", 0755);
                        ::mkdir("tmp/sdf_localizer", 0755);
                        const std::time_t tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
                        std::tm tm_local{}; localtime_r(&tt, &tm_local);
                        char ts_buf[32]; std::strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%d_%H-%M-%S", &tm_local);
                        flip_csv_.open(std::string("tmp/sdf_localizer/flips_") + ts_buf + ".csv",
                                       std::ios::out | std::ios::trunc);
                        if (flip_csv_.is_open())
                            flip_csv_ << "ts_ms,dyaw_deg,dx,dy,new_x,new_y,new_th,sdf_old,sdf_new,"
                                         "anchorSD_old,anchorSD_new,n_anchors,fe_obs,fe_corner,fe_object,iters\n";
                    }
                    if (flip_csv_.is_open())
                    {
                        flip_csv_ << res.timestamp_ms << ',' << dth * 57.2958f << ','
                                  << (x - flip_prev_x_) << ',' << (y - flip_prev_y_) << ','
                                  << x << ',' << y << ',' << phi << ','
                                  << sdf_old << ',' << sdf_new << ','
                                  << anc_old << ',' << anc_new << ',' << nanc << ','
                                  << bd.obs << ',' << bd.corner << ',' << bd.object << ','
                                  << res.iterations_used << '\n';
                        flip_csv_.flush();
                    }
                }
            }
            flip_prev_x_ = x; flip_prev_y_ = y; flip_prev_th_ = phi; flip_prev_valid_ = true;

            Eigen::Affine2f pose = Eigen::Affine2f::Identity();
            pose.translation() = Eigen::Vector2f{x, y};
            pose.linear() = Eigen::Rotation2Df(phi).toRotationMatrix();
            res.robot_pose = pose;

            if (model_->has_prediction)
            {
                res.innovation[0] = x - model_->predicted_pos[0].item<float>();
                res.innovation[1] = y - model_->predicted_pos[1].item<float>();
                float pred_th = model_->predicted_theta[0].item<float>();
                res.innovation[2] = phi - pred_th;
                while (res.innovation[2] > M_PI) res.innovation[2] -= 2.0f * M_PI;
                while (res.innovation[2] < -M_PI) res.innovation[2] += 2.0f * M_PI;
                res.innovation_norm = std::sqrt(res.innovation[0]*res.innovation[0] +
                                                res.innovation[1]*res.innovation[1]);
                apply_adaptive_covariance(res);
            }
            // AFTER the adaptive floor, so `pub` is what actually leaves this agent. Logging it at
            // the covariance update instead recorded the recursion and called it the published value.
            log_hessian_check(res);
        }

        // ===== FINALIZE =====
        // Legacy boundary prior is recomputed post-optimization from the surviving oldest slot.
        // FEJ+Schur already folded the dropped slot into a frozen prior BEFORE the slide (above),
        // so skip this re-anchoring entirely when the flag is on.
        if (window_slid && window_mgr_.size() > 0 && not params.boundary_fej_schur)
            window_mgr_.recompute_boundary_prior(*model_, params, get_device());

        model_->robot_prev_pose = res.robot_pose;

        {
            auto cov_cpu = torch::zeros({3, 3}, torch::kFloat32);
            auto cov_acc = cov_cpu.accessor<float, 2>();
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    cov_acc[i][j] = res.covariance(i, j);
            model_->prev_cov = cov_cpu.to(get_device());
        }

        model_->has_prediction = false;
        res.timestamp_ms = lidar.second;
        last_update_result = res;
        prev_sdf_mse_ = res.sdf_mse;   // track for boundary quality gate next frame
        // Hierarchical boundary precision (HIERARCHICAL_PRECISION.md): infer u_b_ (and the slow map_trust
        // state) from this frame's converged boundary residual, for NEXT frame's boundary_weight. Optimized
        // path only — the early-exit path skips optimization, so there is no fresh boundary evidence there.
        update_boundary_hyperprecision(res.covariance);

        // ===== DEBUG LOG =====
        if (debug_log_.is_open())
        {
            const auto wall_now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();

            const float vel_adv_y = velocity_history.empty() ? 0.f : velocity_history.back().adv_y;
            const float vel_rot   = velocity_history.empty() ? 0.f : velocity_history.back().rot;
            const float odom_adv  = odometry_history.empty() ? 0.f : odometry_history.back().adv;
            const float odom_rot  = odometry_history.empty() ? 0.f : odometry_history.back().rot;

            const float cmd_cov_xx = last_cmd_cov_(0,0);
            const float cmd_cov_tt = last_cmd_cov_(2,2);
            float meas_cov_xx = 0.f, meas_cov_tt = 0.f;
            if (last_measured_prior_.valid && last_measured_prior_.covariance.defined())
            {
                auto mc = last_measured_prior_.covariance.to(torch::kCPU);
                auto ma = mc.accessor<float, 2>();
                meas_cov_xx = ma[0][0];
                meas_cov_tt = ma[2][2];
            }
            const float sel_cov_xx = last_selected_prior_.covariance_eigen(0, 0);
            const float sel_cov_tt = last_selected_prior_.covariance_eigen(2, 2);

            std::string losses_str;
            for (size_t ai = 0; ai < last_adam_losses_.size(); ++ai)
            {
                if (ai > 0) losses_str += '|';
                std::ostringstream ss; ss << last_adam_losses_[ai];
                losses_str += ss.str();
            }
            if (losses_str.empty()) losses_str = "na";

            const float lr_eff = params.learning_rate_pos /
                std::sqrt(static_cast<float>(std::max(1, (int)window_mgr_.size())));
            const auto motion_ingress = get_motion_ingress_debug();

            debug_log_
                << lidar.second
                << ',' << wall_now_ms
                << ',' << selected_prior.dt
                << ',' << sampled_points.size()
                << ',' << vel_adv_y
                << ',' << vel_rot
                << ',' << odom_adv
                << ',' << odom_rot
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
                << ',' << (int)motion_prior_selection.command_prior.valid
                << ',' << (int)motion_prior_selection.command_prior.fresh
                << ',' << motion_prior_selection.command_prior.delta_pose[0]
                << ',' << motion_prior_selection.command_prior.delta_pose[1]
                << ',' << motion_prior_selection.command_prior.delta_pose[2]
                << ',' << cmd_cov_xx
                << ',' << cmd_cov_tt
                << ',' << (int)last_measured_prior_.valid
                << ',' << (int)last_measured_prior_.fresh
                << ',' << last_measured_prior_.delta_pose[0]
                << ',' << last_measured_prior_.delta_pose[1]
                << ',' << last_measured_prior_.delta_pose[2]
                << ',' << meas_cov_xx
                << ',' << meas_cov_tt
                << ',' << (int)last_selected_prior_.valid
                << ',' << (int)last_selected_prior_.fresh
                << ',' << motion_prior_source_name(last_motion_prior_source_)
                << ',' << last_selected_prior_.delta_pose[0]
                << ',' << last_selected_prior_.delta_pose[1]
                << ',' << last_selected_prior_.delta_pose[2]
                << ',' << sel_cov_xx
                << ',' << sel_cov_tt
                << ',' << pred_pos.x()
                << ',' << pred_pos.y()
                << ',' << pred_theta
                << ',' << slot_motion_cov(0,0)
                << ',' << slot_motion_cov(2,2)
                << ',' << 0                          // early_exit = 0 (Adam ran)
                << ',' << res.iterations_used
                << ',' << res.final_loss
                << ',' << lr_eff
                << ',' << res.state[2]
                << ',' << res.state[3]
                << ',' << res.state[4]
                << ',' << res.innovation[0]
                << ',' << res.innovation[1]
                << ',' << res.innovation[2]
                << ',' << res.innovation_norm
                << ',' << res.sdf_mse
                << ',' << res.covariance(0,0)
                << ',' << res.covariance(2,2)
                << ',' << res.condition_number
                << ',' << (int)window_mgr_.size()
                << ',' << tracking_step_count_
                << ',' << last_loss_breakdown_.boundary
                << ',' << last_loss_breakdown_.obs
                << ',' << last_loss_breakdown_.motion
                << ',' << last_loss_breakdown_.corner
                << ',' << last_loss_breakdown_.object
                << ',' << last_loss_init_
                << ',' << std::chrono::duration<float, std::milli>(
                               std::chrono::high_resolution_clock::now() - t_update_start_).count()
                << ',' << last_t_adam_ms_
                << ',' << last_t_cov_ms_
                << ',' << last_t_breakdown_ms_;

            // Per-slot pose persistence columns
            debug_log_ << ',' << slot_poses_pre_
                       << ',' << slot_poses_post_;

            // Per-slot sdf_mse_final (built here so newest slot's value is already set)
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

            // Boundary prior anchor
            {
                const auto& bp = window_mgr_.boundary_prior;
                debug_log_ << ',' << (int)bp.valid
                           << ',' << bp.mu[0]
                           << ',' << bp.mu[1]
                           << ',' << bp.mu[2];
            }

            debug_log_ << ',' << last_lbfgs_grad_norm_
                       << ',' << losses_str
                       << ",," 
                       << ',' << learned_odom_bias_.x()
                       << ',' << learned_odom_bias_.y()
                       << ',' << learned_odom_bias_.z();
            // last_early_exit_metric_, NOT res.early_exit_metric: res only receives it AFTER this
            // block (the assignment below), so this column logged the value from before the gate ran
            // — uniformly stale — and could never show the predicted-pose SDF that TRIGGERED this
            // optimization. That number is the margin by which the prediction missed the early-exit
            // threshold, i.e. the only direct measure of how close a rotating frame came to skipping
            // Adam. The early-exit path at the bottom of try_prediction_early_exit was already
            // correct (it assigns res.early_exit_metric before logging).
            debug_log_ << ',' << last_early_exit_metric_
                       << ',' << recovery_.consecutive_bad_frames
                       << ',' << recovery_.cooldown
                       << ',' << (grid_search_active_.load(std::memory_order_relaxed) ? 1 : 0);
            write_debug_tail();

            debug_log_ << '\n';
            debug_log_.flush();
        }

        // Adam path: expose the predicted-pose SDF that triggered this optimization (NaN if the
        // early-exit gate never evaluated it this frame — e.g. warmup / no odometry).
        res.early_exit_metric = last_early_exit_metric_;
        res.pred_sdf_median   = last_pred_sdf_median_;
        res.pred_x = last_pred_pos_.x();
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
        res.wheel_shadow_dtheta = cyc_wheel_shadow_dtheta_;
        res.imu_segs            = cyc_imu_segs_;
        res.wheel_segs          = cyc_wheel_segs_;
        // MUST come after the fields above: it reads dy_local/dx_local/imu_dtheta as the covariates
        // H. Called earlier it sees zeros, H -> 0, and the learner silently never learns anything.
        feed_motion_calibrator(res);
        return res;
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

        const auto matrix_to_tensor = [this](const Eigen::Matrix3f &matrix)
        {
            auto tensor = torch::zeros({3, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(get_device()));
            auto tensor_cpu = tensor.to(torch::kCPU);
            auto acc = tensor_cpu.accessor<float, 2>();
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 3; ++col)
                    acc[row][col] = matrix(row, col);
            return tensor_cpu.to(get_device());
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

            model_->robot_pos.data().copy_(torch::tensor({pos.x(), pos.y()},
                torch::TensorOptions().device(get_device())));
            model_->robot_theta.data().copy_(torch::tensor({theta},
                torch::TensorOptions().device(get_device())));
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
            selection.selected_prior.covariance = matrix_to_tensor(selection.selected_prior.covariance_eigen);
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
            selection.selected_prior.covariance = matrix_to_tensor(selection.selected_prior.covariance_eigen);
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
        if (params.zupt_on_prediction and selection.selected_prior.valid)
        {
            auto& d = selection.selected_prior.delta_pose;
            const float T = std::max(1e-3f, selection.selected_prior.dt * 1e-3f);
            const auto& nm = params.odom_preint_noise;
            const float sig_p = std::sqrt(nm.zupt_density_v * nm.zupt_density_v * T);      // m
            const float sig_r = std::sqrt(nm.zupt_density_omega * nm.zupt_density_omega * T);
            const float L     = std::max(nm.zupt_lever_m, 1e-3f);
            // Coupled, so a pivot cannot be read as rest: a robot turning on the spot has |dp| ~ 0
            // and must still be recognised as moving.
            const float m_tr = d.head<2>().norm() + L * std::abs(d[2]);
            const float m_ro = std::abs(d[2]) + d.head<2>().norm() / L;
            const auto moving_p = [](float m, float sigma, float span) -> float
            {
                if (not (sigma > 0.f) or not (span > 0.f)) return 1.f;
                const float rest   = std::exp(-0.5f * m * m / (sigma * sigma))
                                   / std::sqrt(2.f * static_cast<float>(M_PI)) / sigma;
                const float moving = 1.f / (2.f * span);
                const float den = rest + moving;
                return den > 0.f ? moving / den : 1.f;
            };
            // Span of the "moving" uniform: what this base could plausibly have done in T.
            const float w_tr = moving_p(m_tr, sig_p, params.zupt_pred_v_max * T);
            const float w_ro = moving_p(m_ro, sig_r, params.zupt_pred_w_max * T);
            zupt_pred_gain_tr_ = w_tr; zupt_pred_gain_ro_ = w_ro;   // for the viewer / debug row
            d.head<2>() *= w_tr;
            d[2]        *= w_ro;
        }

        last_selected_prior_ = selection.selected_prior;
        {   // mirror for the viewer, same lock as the ingress debug
            std::lock_guard lock(motion_ingress_debug_mutex_);
            predictor_delta_ = selection.selected_prior.delta_pose;
        }
        last_motion_prior_source_ = selection.source;
        return selection;
    }

    std::optional<RoomConcept::UpdateResult> RoomConcept::try_prediction_early_exit(
        const torch::Tensor& points_tensor,
        const Eigen::Vector3f& slot_odom_delta,
        const OdometryPrior& odometry_prior,
        std::int64_t lidar_timestamp_ms)
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
        const auto give_up_to_optimizer = [this]() -> std::optional<UpdateResult>
        { forced_solve_last_cycle_ = true; return std::nullopt; };
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
            if (P_grown.allFinite()) current_covariance = P_grown;
        }

        if (!params.prediction_early_exit ||
            !last_update_result.ok ||
            !odometry_prior.valid ||
            tracking_step_count_ <= params.min_tracking_steps)
            return give_up_to_optimizer();

        torch::NoGradGuard no_grad;
        const auto& newest = window_mgr_.newest();
        auto pose_xy = newest.pose.index({torch::indexing::Slice(0, 2)});
        auto pose_th = newest.pose.index({torch::indexing::Slice(2, 3)});
        const auto sdf_pred = model_->sdf_at_pose(points_tensor, pose_xy, pose_th);
        const float mean_sdf_pred = torch::mean(torch::abs(sdf_pred)).item<float>();
        // Record the decision variable whether or not we early-exit — on an Adam frame this is the
        // value that TRIGGERED optimization (it exceeded the trust threshold). update() reads it back.
        last_early_exit_metric_ = mean_sdf_pred;
        // Same points, same pose, median instead of mean — see last_pred_sdf_median_.
        last_pred_sdf_median_ = torch::median(torch::abs(sdf_pred)).item<float>();

        // Widen the SDF trust threshold when the robot is rotating.
        // A theta error ε at room scale R produces SDF displacement ~R*ε.
        // The base threshold (sigma_sdf * trust_factor ≈ 7.5 cm) is too tight during rotation:
        // even a 0.02 rad odometry error at 5 m gives ~10 cm — larger than the base threshold.
        // We add rotation_sdf_coupling * |delta_theta| to compensate for this geometric effect.
        const float rot_boost = params.rotation_sdf_coupling * std::abs(odometry_prior.delta_pose[2]);
        const float prediction_trust_threshold = params.sigma_sdf * params.prediction_trust_factor + rot_boost;
        if (mean_sdf_pred >= prediction_trust_threshold)
            return give_up_to_optimizer();

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
                    return give_up_to_optimizer();      // uncertain enough that the map should be consulted
            }
        }

        // ── Corner-consistency gate ─────────────────────────────────────────────────────────────
        // The SDF gate above cannot see a 180° flip (a rot180 pose is SDF-ambiguous), so it would let a
        // flipped-but-SDF-good prediction early-exit and never give the corners a chance to veto it. Here
        // we validate the SAME predicted pose against the corner factor: if the worst corner's whitened
        // residual m=√(rᵀΛ_det r) exceeds the tolerance, the prediction is inconsistent with the corners
        // (the flip signature) → reject the early-exit so Adam runs and the corner factor pulls it back.
        if (params.corner_early_exit_check)
        {
            const auto& slot = window_mgr_.newest();
            if (!slot.corner_obs.empty())
            {
                auto pose_cpu0 = slot.pose.detach().to(torch::kCPU);
                auto pa = pose_cpu0.accessor<float, 1>();
                const float px = pa[0], py = pa[1], pth = pa[2];
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
                if (n_bad >= params.corner_early_exit_min_bad)
                    return give_up_to_optimizer();
            }
        }

        // NOTE: the object-anchor early-exit GATE was reverted (2026-07-12) to A/B whether it was the
        // source of the localization instability. The anchor is now FACTOR-ONLY: it refines the pose
        // inside SDF-triggered optimizations but no longer FORCES the optimizer to run on its own.
        // (object_anchor_early_exit_sigma is left in the config but unused while this is out.)

        prediction_early_exits_++;
        last_t_adam_ms_ = 0.f;
        last_t_cov_ms_ = 0.f;
        last_t_breakdown_ms_ = 0.f;

        // Rotation early-exit gap: this frame skips the optimizer, so update_boundary_hyperprecision won't
        // run. If we're turning hard, feed the predicted-pose residual into map-trust so a degrading
        // rotation can't silently evade the collapse trigger. No-op unless the reloc feature is enabled.
        nudge_map_trust_early_exit(mean_sdf_pred, odometry_prior.delta_pose[2]);

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
                const auto q = model_->sdf_query_at_pose(points_tensor, pose_xy, pose_th, true);
                if (q.sdf.defined() and q.sdf.size(0) > 0 and q.grad.defined())
                {
                    const auto w = build_observation_weights(*model_, params, points_tensor,
                                                             pose_th, q);
                    const auto d_cpu = q.sdf.detach().to(torch::kCPU).contiguous();
                    const auto g_cpu = q.grad.detach().to(torch::kCPU).contiguous();
                    const auto w_cpu = w.detach().to(torch::kCPU).contiguous();
                    const auto p_cpu = points_tensor.detach().to(torch::kCPU).contiguous();
                    const auto da = d_cpu.accessor<float, 1>();
                    const auto ga = g_cpu.accessor<float, 2>();
                    const auto wa = w_cpu.accessor<float, 1>();
                    const auto pa = p_cpu.accessor<float, 2>();
                    const int n = static_cast<int>(d_cpu.size(0));
                    const float inv_var = 1.0f / (params.rfe_obs_sigma * params.rfe_obs_sigma);
                    const float delta_h = params.rfe_huber_delta;
                    const float th_now = last_pred_theta_;
                    const float c = std::cos(th_now), sn = std::sin(th_now);
                    Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
                    Eigen::Vector3f bb = Eigen::Vector3f::Zero();
                    for (int i = 0; i < n; ++i)
                    {
                        const float dv = da[i];
                        if (not std::isfinite(dv)) continue;
                        const float ad = std::abs(dv);
                        const float u = (ad <= delta_h or ad < 1e-9f) ? 1.0f : delta_h / ad;
                        const float a = 0.5f * inv_var * wa[i] * u / static_cast<float>(n);
                        const float px = pa[i][0], py = pa[i][1];
                        const float qx = c * px - sn * py, qy = sn * px + c * py;
                        Eigen::Vector3f J;
                        J << ga[i][0], ga[i][1], -ga[i][0] * qy + ga[i][1] * qx;
                        H.noalias()  += a * J * J.transpose();
                        bb.noalias() += (a * dv) * J;
                    }
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
                        auto upd = torch::zeros({3}, torch::kFloat32);
                        auto ua = upd.accessor<float, 1>();
                        ua[0] = d.x(); ua[1] = d.y(); ua[2] = d.z();
                        window_mgr_.newest().pose = (newest.pose.detach() + upd.to(get_device()))
                                                        .detach();
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

        auto pose_cpu = window_mgr_.newest().pose.detach().to(torch::kCPU);
        auto p_acc = pose_cpu.accessor<float, 1>();
        const float x = p_acc[0], y = p_acc[1], phi = p_acc[2];

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
        res.sdf_mse = torch::median(torch::abs(sdf_pred)).item<float>();
        res.pred_sdf_median = res.sdf_mse;   // on this path they are the same quantity
        res.early_exit_metric = mean_sdf_pred;   // the value that PASSED the threshold (optimizer skipped)
        res.sdf_polished = sdf_polished_this_cycle_;   // the calibrator counts this as a correction
        // (No res.covariance assignment here: this path already publishes current_covariance further
        //  down. Adding a second one was redundant — and it was how the growth step below reached
        //  every consumer.)
        // Heading attribution must be set on BOTH return paths. This one carries >98% of frames, and
        // it is the interesting one: between corrections the prediction runs open-loop, so these are
        // the only cycles where an accumulating channel error is visible before the optimizer hides it.
        res.pred_x = last_pred_pos_.x();
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
        res.wheel_shadow_dtheta = cyc_wheel_shadow_dtheta_;
        res.imu_segs            = cyc_imu_segs_;
        res.wheel_segs          = cyc_wheel_segs_;
        // MUST come after the fields above: it reads dy_local/dx_local/imu_dtheta as the covariates
        // H. Called earlier it sees zeros, H -> 0, and the learner silently never learns anything.
        feed_motion_calibrator(res);
        res.iterations_used = 0;
        {
            auto ext_cpu = model_->half_extents.to(torch::kCPU);
            auto ext = ext_cpu.accessor<float, 1>();
            res.state << 2.f * ext[0], 2.f * ext[1], x, y, phi;
        }

        Eigen::Affine2f pose_aff = Eigen::Affine2f::Identity();
        pose_aff.translation() = Eigen::Vector2f{x, y};
        pose_aff.linear() = Eigen::Rotation2Df(phi).toRotationMatrix();
        res.robot_pose = pose_aff;
        res.covariance = current_covariance;

        if (model_->has_prediction)
        {
            res.innovation[0] = x - model_->predicted_pos[0].item<float>();
            res.innovation[1] = y - model_->predicted_pos[1].item<float>();
            float p_theta = model_->predicted_theta[0].item<float>();
            res.innovation[2] = phi - p_theta;
            while (res.innovation[2] > M_PI) res.innovation[2] -= 2.0f * M_PI;
            while (res.innovation[2] < -M_PI) res.innovation[2] += 2.0f * M_PI;
            res.innovation_norm = std::sqrt(res.innovation[0]*res.innovation[0] +
                                            res.innovation[1]*res.innovation[1]);
            apply_adaptive_covariance(res);
        }

        model_->robot_pos.data().copy_(torch::tensor({x, y},
            torch::TensorOptions().device(get_device())));
        model_->robot_theta.data().copy_(torch::tensor({phi},
            torch::TensorOptions().device(get_device())));
        model_->robot_prev_pose = res.robot_pose;
        model_->has_prediction = false;

        res.timestamp_ms = lidar_timestamp_ms;
        last_update_result = res;
        prev_sdf_mse_ = res.sdf_mse;   // track for boundary quality gate next frame
        last_lidar_timestamp = lidar_timestamp_ms;

        // Debug log for early-exit frames
        if (debug_log_.is_open())
        {
            const auto wall_now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            const float lr_eff = params.learning_rate_pos /
                std::sqrt(static_cast<float>(std::max(1, (int)window_mgr_.size())));
            float meas_cov_xx = 0.f, meas_cov_tt = 0.f;
            if (last_measured_prior_.valid && last_measured_prior_.covariance.defined())
            {
                auto mc = last_measured_prior_.covariance.to(torch::kCPU);
                auto ma = mc.accessor<float, 2>();
                meas_cov_xx = ma[0][0]; meas_cov_tt = ma[2][2];
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
                       << ',' << recovery_.consecutive_bad_frames
                       << ',' << recovery_.cooldown
                       << ',' << (grid_search_active_.load(std::memory_order_relaxed) ? 1 : 0);
            write_debug_tail();
            debug_log_ << '\n';
            debug_log_.flush();
        }

        return res;
    }

    // Hierarchical boundary precision (HIERARCHICAL_PRECISION.md, Stage 1 + Option A).
    // Treat the boundary-prior precision scale as a random variable π=exp(u_b_) with a log-precision
    // hyperprior u ~ N(g(v), σ_u²), g(v)=u0+g_gain·v, and a slow in-process hyper-state v (map_trust)
    // with its own prior v ~ N(0, σ_v²). One fast VB step on u_b_ and one slow step on v per converged
    // frame. Well-posed only because μ,Λ_b are FEJ-frozen (fixed linearization point) — see the FEJ+Schur
    // marginalization. No-op unless enabled ⇒ zero effect on the legacy quality-gate path.
    void RoomConcept::update_boundary_hyperprecision(const Eigen::Matrix3f &sigma_x)
    {
        if (not params.hier_prec_boundary_enabled) return;

        const auto &bp = window_mgr_.boundary_prior;
        // Same guard as the boundary loss term: it only contributes when the prior is valid AND the
        // oldest slot is not the current one (window.size() > 1). Otherwise there is no residual to learn from.
        if (not bp.valid or window_mgr_.size() <= 1) return;

        // Oldest surviving pose x̂ (post-optimization) — the state the boundary prior anchors.
        const auto &oldest = window_mgr_.window.front().pose;
        const auto x_cpu = oldest.detach().to(torch::kCPU).contiguous();
        const auto xa = x_cpu.accessor<float, 1>();
        Eigen::Vector3f d(xa[0] - bp.mu[0], xa[1] - bp.mu[1], xa[2] - bp.mu[2]);
        d[2] = std::atan2(std::sin(d[2]), std::cos(d[2]));   // wrap the angle residual

        // Expected boundary residual ⟨r_b⟩ = Δᵀ Λ_b Δ + tr(Λ_b Σ_x): the point residual plus the
        // pose-uncertainty discount (a shaky posterior should not over-drive the precision down).
        const float quad  = d.dot(bp.precision * d);
        const float trace = (bp.precision * sigma_x).trace();
        const float r_b   = quad + std::max(0.0f, trace);

        constexpr float d_dim = 3.0f;                        // boundary factor dimensionality
        if (not u_b_init_)                                   // seed to the top-down prediction g(v)
        {
            u_b_ = params.hier_prec_u0 + params.hier_prec_g_gain * map_trust_v_;
            u_b_init_ = true;
        }
        const float g_v = params.hier_prec_u0 + params.hier_prec_g_gain * map_trust_v_;

        // Fast VB step on u=log π:  ∂F/∂u = ½ eᵘ r_b − d/2 + (u − g(v))/σ_u²  (minimize ⇒ u -= lr·∂F/∂u).
        const float grad_u = 0.5f * std::exp(u_b_) * r_b - 0.5f * d_dim
                             + (u_b_ - g_v) / params.hier_prec_sigma_u2;
        u_b_ -= params.hier_prec_lr_u * grad_u;
        // Pure numeric guard on exp(u_b_) (≈[2e-9, 5e8]); NOT a behavioural gate — the operating band is
        // a few units around g(v). Prevents inf/denorm if a pathological residual ever appears.
        u_b_ = std::clamp(u_b_, -20.0f, 20.0f);

        // Slow VB step on the map_trust hyper-state v (Option A):
        //   F_v = (u − g(v))²/(2σ_u²) + v²/(2σ_v²),  ∂F_v/∂v = −g_gain·(u − g(v))/σ_u² + v/σ_v².
        const float grad_v = -params.hier_prec_g_gain * (u_b_ - g_v) / params.hier_prec_sigma_u2
                             + map_trust_v_ / params.hier_prec_sigma_v2;
        map_trust_v_ -= params.hier_prec_lr_v * grad_v;

        // A/B trace: boundary_weight is exp(u_b_) AFTER this update (i.e. what next frame will apply).
        log_hier_prec_row("opt", r_b, quad, std::max(0.0f, trace), /*reloc_fired=*/false);
    }

    // Fold one detection frame into the per-model-corner tallies. Cheap (a handful of map lookups per
    // frame); the expensive part is the rewrite below, which is throttled.
    void RoomConcept::accumulate_corner_stats(const CornerDetector::DetectionResult& det)
    {
        for (const int idx : det.in_fov_indices)
            corner_vertex_stats_[idx].in_fov++;
        for (const int idx : det.occluded_indices)
            corner_vertex_stats_[idx].occluded++;

        for (const auto& m : det.matches)
        {
            auto& s = corner_vertex_stats_[m.model_index];
            s.accepted++;
            s.last_yield  = m.yield;
            s.retired_now = m.suppressed;
            if (m.suppressed) s.suppressed++;
            s.assoc_prob_sum += m.assoc_prob;
            s.resid_sum      += (m.detected - m.predicted).norm();
            // Λ_det is 2×2 symmetric and ALREADY scaled by assoc_prob — i.e. exactly the precision the
            // loss sees. Its smallest eigenvalue is the number that matters: near zero means the two
            // wall fits were near-parallel and the corner constrains nothing along the bisector, which
            // is the pillar failure mode described at detect()'s per-corner band clamp.
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> es(m.information);
            if (es.info() == Eigen::Success)
            {
                s.lambda_min_sum += es.eigenvalues()(0);
                s.lambda_max_sum += es.eigenvalues()(1);
            }
            // A runner-up only exists when another model corner was inside the gate; the sentinel
            // means "no rival", and averaging it in is what made the old rival statistic meaningless.
            if (m.runnerup_chi2 < 1e8f)
            {
                s.rival_n++;
                s.runnerup_chi2_sum += m.runnerup_chi2;
            }
        }

        if (++corner_stats_frames_ % kCornerStatsRewritePeriod == 0)
            write_corner_stats_csv();
    }

    // Rewrite etc/corner_stats.csv from scratch — one row per model corner, current totals. Truncating
    // rather than appending keeps it directly readable (sort by accept_rate or lambda_min) without an
    // aggregation step. Columns chosen to answer one question per pair:
    //   accept_rate    — does this corner ever pay off, or is it pure cost?
    //   assoc_prob     — is it aliasing with a neighbour? (≪1 ⇒ the PDA is already muting it)
    //   runnerup_chi2  — how close was the rival? (near assoc χ² ⇒ coin flip ⇒ pose jumps)
    //   lambda_min     — does it constrain anything, or only one direction? (→0 ⇒ no information)
    //   resid_mean     — is the map honest here?
    void RoomConcept::write_corner_stats_csv()
    {
        // Reopen with trunc rather than rewinding: field widths vary between rewrites (a shrinking
        // average writes fewer characters), so seeking to 0 would leave the tail of the previous,
        // longer row behind. Throttled to every kCornerStatsRewritePeriod frames, so this is free.
        corner_stats_csv_.close();
        corner_stats_csv_.open("etc/corner_stats.csv", std::ios::out | std::ios::trunc);
        if (!corner_stats_csv_.is_open())
            return;

        corner_stats_csv_ << "# frames=" << corner_stats_frames_
                          << " wall_in/wall_out are the adjacent polygon edge lengths (m)\n"
                          << "vertex,wall_in,wall_out,in_fov,occluded,accepted,accept_rate,"
                             "assoc_prob,rival_n,runnerup_chi2,lambda_min,lambda_max,resid_mean,"
                             "yield,retired_now,suppressed,suppressed_frac\n";

        const auto& poly = init_polygon_vertices_;
        const int N = static_cast<int>(poly.size());
        for (const auto& [idx, s] : corner_vertex_stats_)
        {
            float w_in = 0.f, w_out = 0.f;
            if (N >= 3 && idx >= 0 && idx < N)
            {
                w_in  = (poly[idx] - poly[(idx + N - 1) % N]).norm();
                w_out = (poly[(idx + 1) % N] - poly[idx]).norm();
            }
            const double acc  = s.accepted > 0 ? static_cast<double>(s.accepted) : 0.0;
            const double rate = s.in_fov > 0 ? acc / s.in_fov : 0.0;
            const auto avg = [&](double sum) { return s.accepted > 0 ? sum / acc : 0.0; };
            corner_stats_csv_ << idx << ',' << w_in << ',' << w_out << ','
                              << s.in_fov << ',' << s.occluded << ',' << s.accepted << ',' << rate << ','
                              << avg(s.assoc_prob_sum) << ',' << s.rival_n << ','
                              << (s.rival_n > 0 ? s.runnerup_chi2_sum / s.rival_n : 0.0) << ','
                              << avg(s.lambda_min_sum) << ',' << avg(s.lambda_max_sum) << ','
                              << avg(s.resid_sum) << ',' << s.last_yield << ','
                              << (s.retired_now ? 1 : 0) << ',' << s.suppressed << ','
                              << (s.accepted > 0 ? s.suppressed / acc : 0.0) << '\n';
        }
        corner_stats_csv_.flush();
    }

    // Append one row to etc/hier_prec.csv (lazy-opened, truncate). Loc-thread only, no lock — same
    // discipline as opt_csv_. src labels the origin: "opt" (optimized boundary update), "ee"
    // (early-exit rotation surrogate), "reloc" (a map-trust relocalization just fired).
    void RoomConcept::log_hier_prec_row(const char *src, float r_b, float quad, float trace, bool reloc_fired)
    {
        if (!hier_prec_csv_open_attempted_)
        {
            hier_prec_csv_open_attempted_ = true;
            hier_prec_csv_.open("etc/hier_prec.csv", std::ios::out | std::ios::trunc);
            if (hier_prec_csv_.is_open())
                hier_prec_csv_ << "ts_ms,src,r_b,quad,trace,u_b,boundary_weight,map_trust_v,window_size,reloc_fired\n";
        }
        if (hier_prec_csv_.is_open())
        {
            hier_prec_csv_ << last_update_result.timestamp_ms << ',' << src << ',' << r_b << ',' << quad << ','
                           << trace << ',' << u_b_ << ',' << std::exp(u_b_) << ','
                           << map_trust_v_ << ',' << window_mgr_.size() << ',' << (reloc_fired ? 1 : 0) << '\n';
            hier_prec_csv_.flush();
        }
    }

    // Rotation early-exit gap closer — see the method's header doc. Fast-only step on u_b_ using a
    // surrogate residual so a degrading rotation that keeps early-exiting can still collapse map-trust.
    void RoomConcept::nudge_map_trust_early_exit(float mean_sdf_pred, float dtheta)
    {
        if (not params.hier_prec_boundary_enabled or not params.hier_prec_reloc_enabled) return;
        if (std::abs(dtheta) < params.hier_prec_ee_dtheta_min) return;      // only in the rotation gap
        if (params.sigma_sdf <= 0.f) return;

        // Surrogate residual: no boundary factor is evaluated on early-exit, so use the whitened
        // predicted-pose residual as evidence that the map's explanatory precision should drop.
        const float w    = mean_sdf_pred / params.sigma_sdf;
        const float r_ee = w * w;

        if (not u_b_init_) { u_b_ = params.hier_prec_u0 + params.hier_prec_g_gain * map_trust_v_; u_b_init_ = true; }
        const float g_v   = params.hier_prec_u0 + params.hier_prec_g_gain * map_trust_v_;
        constexpr float d_dim = 3.0f;
        const float grad_u = 0.5f * std::exp(u_b_) * r_ee - 0.5f * d_dim
                             + (u_b_ - g_v) / params.hier_prec_sigma_u2;
        u_b_ -= params.hier_prec_lr_u * grad_u;
        u_b_ = std::clamp(u_b_, -20.0f, 20.0f);
        // Fast-only: leave the slow map_trust_v_ to the optimized path (real boundary evidence).
        log_hier_prec_row("ee", r_ee, r_ee, 0.0f, /*reloc_fired=*/false);
    }

    // =========================================================================
    //  Backend-shared helpers
    // =========================================================================
    float RoomConcept::boundary_weight_now() const
    {
        if (params.hier_prec_boundary_enabled)
            return std::exp(u_b_);
        if (params.rfe_boundary_quality_gate and prev_sdf_mse_ > 1e-6f)
        {
            // prev_sdf_mse_ became a MEDIAN absolute residual on 2026-08-13 (it was a mean on the
            // early-exit path, which is >98% of frames). A median of a half-normal is 0.674*sigma
            // against the mean's 0.798*sigma, so the denominator shrank by kMedianOverMeanAbs and
            // this weight would otherwise have risen by 1/0.845 = 1.18 across the board. The factor
            // restores the calibration; it is applied here rather than to sigma_sdf because
            // sigma_sdf also sets the early-exit gate, which still tests the mean and must not move.
            const float sigma2 = params.sigma_sdf * params.sigma_sdf;
            return std::min(1.0f, kMedianOverMeanAbs * sigma2 / prev_sdf_mse_);
        }
        return 1.0f;
    }

    std::vector<Eigen::Vector3f> RoomConcept::read_window_poses() const
    {
        std::vector<Eigen::Vector3f> out;
        out.reserve(window_mgr_.window.size());
        for (const auto& slot : window_mgr_.window)
        {
            const auto cpu = slot.pose.detach().to(torch::kCPU);
            const auto a = cpu.accessor<float, 1>();
            out.emplace_back(a[0], a[1], a[2]);
        }
        return out;
    }

    void RoomConcept::write_window_poses(const std::vector<Eigen::Vector3f>& poses)
    {
        if (poses.size() != window_mgr_.window.size()) return;
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < poses.size(); ++i)
            window_mgr_.window[i].pose.data().copy_(
                torch::tensor({poses[i].x(), poses[i].y(), poses[i].z()},
                              torch::TensorOptions().dtype(torch::kFloat32).device(get_device())));
    }

    // =========================================================================
    //  Levenberg-Marquardt backend (analytic Jacobians — see room_gn_solver.h)
    // =========================================================================
// ── The published sigma, measured against what the window's own equations say ────────────────────
// Four precisions for the same newest pose:
//
//   marg  = H_nn − H_no H_oo⁻¹ H_on   the window's marginal. What the solve actually knows.
//   block = H_nn                      the same without the complement, i.e. pretending every other
//                                     window pose is known EXACTLY. Always the more confident.
//   rec   = current_covariance⁻¹      the filter's own posterior, before the adaptive floor.
//   pub   = res.covariance⁻¹          what leaves this agent, after apply_adaptive_covariance.
//
// ★ MEASURED 2026-08-29, first 79 optimised frames, and it corrected the guess that prompted it.
//   The guess was that sigma would prove OVER-confident because the filter has no prediction step.
//   It has one: predict_step() writes propagated_cov into current_covariance on every optimised
//   cycle (the polish-gated growth further down is the EARLY-EXIT path's, a different branch). And
//   the reading came out the other way — sigma_theta published 0.042 rad against the window's own
//   0.0034, i.e. 12x LOOSER, with x and y 7-8x. The filter is conservative, not over-confident, and
//   the ratio does NOT drift with the gap since the last solve (corr +0.05 over log gap), which is
//   what a working prediction step looks like.
// ★ block/marg came out at 0.79 median (0.51 at p10, on short windows): dropping the cross-terms
//   would claim ~21% tighter than the window supports, and up to 2x. That is the answer to "why
//   Schur-complement at all" in numbers.
// ★ WHAT IS STILL OPEN: whether 7-12x is the RIGHT amount of conservatism. It is set by the process
//   noise in propagated_cov, and sigma_pub varies far less than sigma_marg does (CV 0.16 vs 0.33 in
//   x), so the published number is carrying much more motion prior than fit quality. That matters
//   downstream — the speed governor and every concept agent's precision read it.
void RoomConcept::log_hessian_check(const UpdateResult& res)
{
    if (not params.hessian_check or not last_marg_ok_) return;
    const auto sig = [](const Eigen::Matrix3f& prec, int i) -> double
    {
        const Eigen::Matrix3f c = prec.inverse();
        return (c.allFinite() and c(i, i) > 0.f) ? std::sqrt(static_cast<double>(c(i, i))) : -1.0;
    };
    const auto sd = [](const Eigen::Matrix3f& c, int i) -> double
    {
        return (c.allFinite() and c(i, i) > 0.f) ? std::sqrt(static_cast<double>(c(i, i))) : -1.0;
    };
    static bool header = false;
    std::ofstream f("etc/hessian_check.csv", header ? std::ios::app : std::ios::trunc);
    if (not f) return;
    f.imbue(std::locale::classic());          // es_ES would write commas; see CLAUDE.md
    if (not header)
    {
        header = true;
        f << "t_ms,slots,dof_marginalised,"
             "sx_marg,sy_marg,st_marg,sx_block,sy_block,st_block,"
             "sx_rec,sy_rec,st_rec,sx_pub,sy_pub,st_pub,"
             "ratio_x,ratio_y,ratio_t,block_over_marg_t,ms_since_last_opt\n";
    }
    const double sm[3] = {sig(last_marg_prec_, 0),  sig(last_marg_prec_, 1),  sig(last_marg_prec_, 2)};
    const double sb[3] = {sig(last_block_prec_, 0), sig(last_block_prec_, 1), sig(last_block_prec_, 2)};
    const double sr[3] = {sd(hess_pre_adaptive_, 0), sd(hess_pre_adaptive_, 1), sd(hess_pre_adaptive_, 2)};
    const double sp[3] = {sd(res.covariance, 0),     sd(res.covariance, 1),     sd(res.covariance, 2)};
    const auto ratio = [](double a, double b) { return (a > 0.0 and b > 0.0) ? a / b : -1.0; };
    f << std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch()).count()
      << ',' << last_gn_window_slots_ << ',' << last_marg_dof_;
    for (int i = 0; i < 3; ++i) f << ',' << sm[i];
    for (int i = 0; i < 3; ++i) f << ',' << sb[i];
    for (int i = 0; i < 3; ++i) f << ',' << sr[i];
    for (int i = 0; i < 3; ++i) f << ',' << sp[i];
    // marg / pub > 1 means the window is LESS certain than what we publish — the over-confidence.
    for (int i = 0; i < 3; ++i) f << ',' << ratio(sm[i], sp[i]);
    const std::int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch()).count();
    f << ',' << ratio(sb[2], sm[2]) << ','
      << (hess_prev_opt_ms_ == 0 ? -1 : now_ms - hess_prev_opt_ms_) << '\n';
    hess_prev_opt_ms_ = now_ms;
    ++hess_check_rows_;

    if (now_ms - hess_check_last_log_ms_ > 5000)
    {
        hess_check_last_log_ms_ = now_ms;
        qInfo().nospace()
            << "[hess] sigma_theta  window(marg) " << sm[2] << "  window(block) " << sb[2]
            << "  filter " << sr[2] << "  published " << sp[2] << "  | marg/pub " << ratio(sm[2], sp[2])
            << " block/marg " << ratio(sb[2], sm[2])
            << " | slots " << last_gn_window_slots_ << " dof_marg " << last_marg_dof_
            << " rows " << hess_check_rows_;
    }
}

    std::pair<float, int> RoomConcept::run_gn_loop(const OdometryPrior& odometry_prior)
    {
        // velocity_adaptive_weights is deliberately not applied here: it preconditions Adam's
        // gradient, and a curvature-correct step does not want a preconditioner. Same minimum.
        gn::Input in;
        in.model           = model_.get();
        in.params          = &params;
        in.window          = &window_mgr_.window;
        in.boundary_prior  = &window_mgr_.boundary_prior;
        in.boundary_weight = boundary_weight_now();
        in.device          = get_device();

        gn::Options opts;
        opts.max_iters    = params.gn_max_iters;
        opts.lambda_init  = params.gn_lambda_init;
        opts.step_tol     = params.gn_step_tol;
        opts.loss_rel_tol = params.gn_loss_rel_tol;

        // ── Private landmark variables ──────────────────────────────────────────────────────────────
        // One per distinct object anchor in the window. Born from the producing agent's belief (mean +
        // Σ_o as the prior), then refined by room's observations alone and NEVER written back to the
        // graph. The prior is spent on the birth frame: Σ_o came through the robot pose, so re-applying
        // it every frame would keep feeding the same evidence back and shrink both estimates without
        // new information (the incest that makes two agents confidently agree on the wrong thing).
        std::vector<gn::Landmark> landmarks;
        if (params.object_anchor_optimize_landmark and params.object_anchor.enable)
        {
            std::scoped_lock lk(object_anchors_mutex_);
            for (const auto& slot : window_mgr_.window)
                for (const auto& a : slot.object_anchors)
                {
                    if (std::ranges::any_of(landmarks, [&](const auto& l) { return l.id == a.node_id; }))
                        continue;
                    gn::Landmark lm;
                    lm.id = a.node_id;
                    auto it = landmark_estimates_.find(a.node_id);
                    if (it == landmark_estimates_.end())
                    {
                        const Eigen::Matrix2f sig = a.map_cov.topLeftCorner<2, 2>();
                        if (not sig.allFinite() or sig(0, 0) <= 0.f or sig(1, 1) <= 0.f)
                            continue;                       // no usable birth prior yet
                        lm.p                 = a.pose_world.head<2>();
                        lm.prior_mean        = lm.p;
                        lm.prior_information = sig.inverse();
                        lm.information       = lm.prior_information;
                        lm.has_prior         = true;        // birth frame only
                        qInfo() << "[room][landmark] born" << QString::fromStdString(a.type)
                                << "#" << static_cast<qulonglong>(a.node_id)
                                << "at (" << lm.p.x() << "," << lm.p.y() << ") prior σ="
                                << std::sqrt(std::max(sig(0, 0), sig(1, 1))) * 1000.f << "mm";
                    }
                    else
                    {
                        lm.p           = it->second.p;
                        lm.information = it->second.information;
                        lm.has_prior   = false;
                    }
                    landmarks.push_back(lm);
                }
            in.landmarks = landmarks.empty() ? nullptr : &landmarks;
        }

        auto poses = read_window_poses();
        const auto r = gn::solve(in, poses, opts);
        if (not r.ok)
        {
            // Singular system / non-finite loss: fall back rather than hand the caller a NaN pose.
            // The window is untouched at this point, so Adam starts from the same state GN did.
            qWarning() << "[gn] solve failed (loss" << r.loss_init << ") — falling back to Adam";
            return run_adam_loop(odometry_prior);
        }

        write_window_poses(poses);
        if (in.landmarks != nullptr)
        {
            std::scoped_lock lk(object_anchors_mutex_);
            for (const auto& lm : landmarks)
                landmark_estimates_[lm.id] = {lm.p, lm.information};
        }
        // ── What this window says about the newest pose, both ways ──────────────────────────────
        // Taken HERE because this is the only place the assembled Input and the converged poses
        // exist together; recomputing either elsewhere would be comparing two different problems.
        // One extra linearize (~0.3 ms) on an OPTIMISED frame only, and those are ~0.7% of frames.
        if (params.hessian_check)
        {
            const auto nm = gn::newest_pose_marginal(in, poses);
            last_marg_ok_         = nm.ok;
            last_marg_prec_       = nm.marginal;
            last_block_prec_      = nm.block;
            last_marg_dof_        = nm.n_marginalised;
            last_gn_window_slots_ = static_cast<int>(poses.size());
        }
        else last_marg_ok_ = false;
        last_lbfgs_grad_norm_ = r.grad_norm;
        last_adam_losses_.clear();
        last_adam_losses_.push_back(r.loss_init);
        last_adam_losses_.push_back(r.loss);
        last_loss_init_ = r.loss_init;
        return {r.loss, r.iterations};
    }


    // ── RGB edge shadow + systematic-residual monitor ────────────────────────────────────────────
    void RoomConcept::run_image_edge_shadow(const std::vector<Eigen::Vector3f>& poses_after,
                                            float boundary_weight, std::int64_t timestamp_ms,
                                            bool probe_pose)
    {
        // ── Health, so "no CSV" is diagnosable instead of ambiguous ───────────────────────────────
        // Three different failures land in the same silence: the camera never bound, it bound but
        // extracts nothing, or rows are being written and nobody looked. Count them apart and say so
        // once every 5 s. Without this the only evidence is an absent file, which is consistent with
        // all three.
        ++imgedge_calls_;
        if (window_mgr_.window.empty()) { ++imgedge_no_window_; return; }
        const auto& slot = window_mgr_.window.back();
        const auto& obs  = slot.image_edges;
        if (obs.empty())       { ++imgedge_no_obs_; imgedge_health(timestamp_ms); return; }
        if (not obs.cam.valid) { ++imgedge_no_cam_; imgedge_health(timestamp_ms); return; }

        last_mount_fy_ = obs.cam.fy;
        const Eigen::Vector3f pose = poses_after.back();
        // Where the robot was while these samples were taken. Logged beside the fit because the
        // question the cumulative form could not answer is whether the estimate moves WITH position.
        mnt_pose_x_ = pose.x(); mnt_pose_y_ = pose.y(); mnt_pose_th_ = pose.z();
        const float cth = std::cos(pose.z()), sth = std::sin(pose.z());
        Eigen::Matrix3f Rm; Rm << cth, sth, 0.f, -sth, cth, 0.f, 0.f, 0.f, 1.f;

        // ── Pass 1: the term's own numbers at the authority's pose ───────────────────────────────
        float trace_raw = 0.f, trace_eff = 0.f, sum_gamma = 0.f, chi2 = 0.f, loss_img = 0.f;
        int   n_used = 0;
        std::vector<float> abs_r;
        // Systematic-residual monitor: fit  r_k ~ b_const + b_invd * (fy / d_k).
        // Range SEPARATES the two mount errors, exactly as rotation-vs-time separates gyro scale
        // from gyro bias in calibration_estimator.h: a PITCH error is constant in pixels
        // (delta_psi = b_const / fy) while a HEIGHT error scales as fy/d (delta_h = b_invd). A
        // persistent, pose-independent, significant b is miscalibration; a fluctuating one is noise.
        double S11 = 0, S12 = 0, S22 = 0, Sy1 = 0, Sy2 = 0, Syy = 0, Sw = 0;

        for (const auto& seg : obs.segments)
        {
            const auto acc = rc::img::accumulate_segment(seg,
                [&](std::size_t k, Eigen::Matrix<float, 1, 3>& J) -> float
                {
                    const auto& smp = seg.samples[k];
                    const Eigen::Vector3f e(smp.p_room.x() - pose.x(), smp.p_room.y() - pose.y(), smp.p_room.z());
                    const Eigen::Vector3f p_robot = Rm * e;
                    const Eigen::Vector3f p_cam   = obs.cam_R_robot * p_robot + obs.cam_t_robot;
                    Eigen::Vector2d uv;
                    if (not rc::img::project_with_model(obs.cam, p_cam.cast<double>(), uv))
                        return std::numeric_limits<float>::quiet_NaN();
                    double du = uv.x() - static_cast<double>(smp.uv_meas.x());
                    if (obs.cam.kind != CameraModel::Kind::Pinhole)
                    {
                        while (du >  0.5 * obs.cam.width) du -= obs.cam.width;
                        while (du <= -0.5 * obs.cam.width) du += obs.cam.width;
                    }
                    const double dv = uv.y() - static_cast<double>(smp.uv_meas.y());
                    Eigen::Matrix<double, 2, 3> P;
                    if (not rc::img::project_jacobian_model(obs.cam, p_cam.cast<double>(), P))
                        return std::numeric_limits<float>::quiet_NaN();
                    Eigen::Matrix3f Jx;
                    Jx.col(0) = obs.cam_R_robot * (-Rm.col(0));
                    Jx.col(1) = obs.cam_R_robot * (-Rm.col(1));
                    Jx.col(2) = obs.cam_R_robot * Eigen::Vector3f(p_robot.y(), -p_robot.x(), 0.f);
                    J = smp.n_hat.transpose() * (P.cast<float>() * Jx);
                    const float r = smp.n_hat.x() * static_cast<float>(du)
                                  + smp.n_hat.y() * static_cast<float>(dv);
                    // ── The inlier weight, computed ONCE and shared by both monitors ──────────
                    // ★ SAME WEIGHT THE FACTOR USES: gamma/sigma^2, not 1/sigma^2. gamma is the
                    // mixture responsibility — this sample's posterior probability of being an
                    // inlier, an inlier Gaussian against a UNIFORM over the window actually
                    // searched. A bad match therefore removes itself by its own evidence, which is
                    // the only outlier handling here and needs no threshold. Ignoring it let
                    // mismatches dominate a 4-sample fit: r_rms ran 11.9 px against a stated sigma
                    // of 0.05.
                    // ★ THE INLIER VARIANCE MUST INCLUDE THE PREDICTION'S OWN SPREAD, not just the
                    // edge-localisation sigma. smp.h holds each nuisance's sensitivity ALREADY
                    // multiplied by that nuisance's prior sigma (px per unit-variance nuisance), so
                    // h.squaredNorm() IS the predicted px^2 variance contributed by an unknown
                    // mount pitch, height, boresight yaw and image/lidar offset. Widening by it is
                    // what makes "a residual consistent with a plausible mount error" an INLIER —
                    // exactly the sample that carries mount information. Weighting by sigma_px
                    // alone rejected everything: at r ~ 12 px against sigma_px ~ 0.05,
                    // exp(-0.5*(r/sigma)^2) underflows, gamma is 0 for every sample, and the pooled
                    // fit starved silently. An estimator that admits nothing and an estimator that
                    // has nothing to admit look identical from outside, which is why the sample
                    // counts are logged.
                    const float  s2  = smp.sigma_px * smp.sigma_px + smp.h.squaredNorm();
                    const float  gam = s2 > 0.f
                                     ? rc::img::responsibility(r, s2, smp.pi_vis, smp.search_L)
                                     : 0.f;
                    const double w   = s2 > 0.f ? static_cast<double>(gam) / static_cast<double>(s2)
                                                : 0.0;

                    // ── Monitor 2: RIGID IMAGE TRANSLATION,  r_k ~ tx*nx_k + ty*ny_k ─────────
                    // Why this exists beside the fy/d fit below. The residual is SCALAR and taken
                    // along each contour's OWN normal, so a rigid displacement of the prediction —
                    // principal point, boresight yaw, a small pitch — enters sample k as
                    // t . n_hat_k: positive on some contours, negative on others, averaging toward
                    // zero. The CONSTANT column cannot represent it. So "b_const ~ 0" was never
                    // evidence that the camera points where we think it does; it is what a rigid
                    // pointing error looks like in a basis that cannot see one.
                    // Measured parked (2026-08-28, 82 windows): b_const -0.15 +/- 0.11 px and
                    // b_invd -0.0036 +/- 0.0012 m together account for ~0.7 px of a 6.11 px r_rms.
                    // This is the basis in which the missing 5.4 px would show up if it is pointing.
                    // ★ ALL CONTOUR CLASSES, unlike the fit below. What conditions this fit is the
                    // spread of contour ORIENTATIONS, not of ranges, and that is why it works
                    // PARKED where the fy/d fit cannot (rho 0.9864, cond 146, unmoved across 82
                    // windows because the pose spanned 3 cm). Mixing floor junctions (near-vertical
                    // normals) with wall corners (near-horizontal) makes the columns orthogonal,
                    // but it is NOT required: simulated at the real geometry, junctions alone give
                    // cond 1.1, because a contour's normal already turns by ~0.15 rad along it.
                    // Taking every class is for sample count and coverage, not conditioning.
                    if (gam > 1e-6f)
                    {
                        const double nx = smp.n_hat.x(), ny = smp.n_hat.y();
                        mnt_T11_ += w * nx * nx;  mnt_T12_ += w * nx * ny;  mnt_T22_ += w * ny * ny;
                        mnt_Tx_  += w * nx * r;   mnt_Ty_  += w * ny * r;   mnt_Tyy_ += w * r * r;
                        ++mnt_tn_;

                        // ── Monitor 3: the four mount nuisances as PARAMETERS (see room_concept.h) ──
                        // Same samples, same weight, same residual. The only new thing is keeping h
                        // instead of collapsing it to h.squaredNorm().
                        // ★ head<4> ON PURPOSE: column [4] is THIS CONTOUR's map position, a
                        //   per-segment nuisance, not a mount parameter. Fitting it globally would
                        //   average six different walls' offsets into one meaningless number and
                        //   contaminate the four that are genuinely global.
                        const Eigen::Vector4d hd = smp.h.head<4>().cast<double>();
                        mnt_H_.noalias() += w * hd * hd.transpose();
                        mnt_b_.noalias() += w * hd * static_cast<double>(r);
                        ++mnt_hn_;
                    }

                    // ── Monitor 1: FLOOR-JUNCTION samples only ───────────────────────────────
                    // The fy/d signature is what makes pitch and height separable, and a vertical
                    // corner does not carry it.
                    if (seg.class_id == ContourClass::FloorWall and smp.sigma_px > 0.f)
                    {
                        const float d = p_cam.norm();
                        if (d > 0.3f)
                        {
                            if (not (gam > 1e-6f)) return r;
                            const double x2 = obs.cam.fy / d;
                            S11 += w;        S12 += w * x2;   S22 += w * x2 * x2;
                            Sy1 += w * r;    Sy2 += w * x2 * r; Syy += w * r * r; Sw += w;
                            // ── POOLED across the whole run ───────────────────────────────────
                            // The mount is STATIC, so the per-frame fit is the wrong unit: it asks
                            // four samples spanning almost no depth to separate a constant from a
                            // 1/d term, which is near-degenerate, and the answer wandered a full
                            // degree between quarters of a run. Pooling turns range diversity from a
                            // within-image accident into something the TRAJECTORY supplies for free.
                            mnt_S11_ += w;       mnt_S12_ += w * x2;    mnt_S22_ += w * x2 * x2;
                            mnt_Sy1_ += w * r;   mnt_Sy2_ += w * x2 * r; mnt_Syy_ += w * r * r;
                            ++mnt_n_;
                        }
                    }
                    abs_r.push_back(std::fabs(r));
                    return r;
                },
                [](std::size_t) { return 0.0f; });
            trace_raw += acc.trace_raw;
            trace_eff += acc.trace_eff;
            sum_gamma += acc.sum_gamma;
            chi2      += acc.chi2;
            loss_img  += acc.loss;
            n_used    += acc.n_used;
        }
        if (n_used == 0) return;

        float r_rms = 0.f;
        for (float a : abs_r) r_rms += a * a;
        r_rms = std::sqrt(r_rms / static_cast<float>(std::max<std::size_t>(1, abs_r.size())));

        // Solve the 2x2 weighted normal equations for the monitor.
        float b_const = 0.f, b_invd = 0.f, se_const = 0.f, se_invd = 0.f;
        {
            const double det = S11 * S22 - S12 * S12;
            if (std::abs(det) > 1e-12 and Sw > 0)
            {
                b_const = static_cast<float>(( S22 * Sy1 - S12 * Sy2) / det);
                b_invd  = static_cast<float>((-S12 * Sy1 + S11 * Sy2) / det);
                se_const = static_cast<float>(std::sqrt(std::max(0.0,  S22 / det)));
                se_invd  = static_cast<float>(std::sqrt(std::max(0.0,  S11 / det)));
            }
        }

        // ── Pass 2: who won? Solve twice from the SAME start, exactly the run_gn_shadow pattern.
        //    Shadow mode must not be able to change the published pose, so the window poses are
        //    restored afterwards by the caller (poses_after is what the authority produced).
        float dx = 0.f, dy = 0.f, dth = 0.f;
        if (probe_pose)
        {
            rc::gn::Input in;
            in.model = model_.get();
            in.params = &params;
            in.window = &window_mgr_.window;
            in.boundary_prior = &window_mgr_.boundary_prior;
            in.boundary_weight = boundary_weight;
            in.device = get_device();

            Params p_on  = params;  p_on.image_edge.enable = true;  p_on.image_edge.drive = true;
            rc::gn::Input in_on = in; in_on.params = &p_on;
            auto poses_on = poses_after;
            rc::gn::Options opts;
            const auto r_on = rc::gn::solve(in_on, poses_on, opts);
            if (r_on.ok)
            {
                const auto& a = poses_after.back();
                const auto& b = poses_on.back();
                dx = b.x() - a.x(); dy = b.y() - a.y();
                dth = std::atan2(std::sin(b.z() - a.z()), std::cos(b.z() - a.z()));
            }
        }

        if (not image_edge_csv_.is_open())
        {
            image_edge_csv_.open(params.image_edge_csv, std::ios::out | std::ios::trunc);
            if (image_edge_csv_.is_open())
            {
                image_edge_csv_.imbue(std::locale::classic());   // CLAUDE.md: never a comma decimal
                image_edge_csv_
                    // does it carry information?
                    << "ts_ms,frame_stamp,dt_img_lidar_ms,n_segments,n_used,sum_gamma,sigma_i,"
                       "r_rms_px,chi2_per_dof,trace_raw,trace_eff,info_ratio,loss_img,"
                    // who won?
                       "dpose_valid,dpose_x,dpose_y,dpose_th,"
                    // is the mount calibrated?
                       "bias_const_px,se_const_px,bias_invd_m,se_invd_m,implied_dpitch_rad,implied_dheight_m\n";
            }
        }
        {   // Publish for the viewer BEFORE the CSV branch, so the plot does not depend on logging
            // being enabled — an instrument that only works when another instrument is on is how a
            // channel comes to look dead.
            std::scoped_lock lk(image_edge_stats_mutex_);
            image_edge_stats_.valid        = true;
            image_edge_stats_.chi2_per_dof = chi2 / std::max(1.f, sum_gamma);
            image_edge_stats_.r_rms_px     = r_rms;
            image_edge_stats_.loss_img     = loss_img;
            image_edge_stats_.n_used       = n_used;
            image_edge_stats_.n_segments   = static_cast<int>(obs.segments.size());
            image_edge_stats_.ts_ms        = timestamp_ms;
        }

        if (image_edge_csv_.is_open())
        {
            const float chi2_dof = chi2 / std::max(1.f, sum_gamma);
            image_edge_csv_
                << timestamp_ms << ',' << obs.frame_stamp << ',' << obs.dt_to_slot_ms
                << ',' << obs.segments.size() << ',' << n_used << ',' << sum_gamma
                << ',' << obs.sigma_i << ',' << r_rms << ',' << chi2_dof
                << ',' << trace_raw << ',' << trace_eff
                << ',' << (trace_raw / std::max(1e-9f, trace_eff))
                << ',' << loss_img
                << ',' << (probe_pose ? 1 : 0) << ',' << dx << ',' << dy << ',' << dth
                << ',' << b_const << ',' << se_const << ',' << b_invd << ',' << se_invd
                << ',' << (obs.cam.fy > 0.f ? b_const / obs.cam.fy : 0.f) << ',' << b_invd
                << '\n';
            image_edge_csv_.flush();
            ++imgedge_rows_;
        }
        log_triple_points(obs, timestamp_ms, pose);
        imgedge_health(timestamp_ms);
        mount_pooled_solve(timestamp_ms);
    }

    /// One row per detected triple point per frame. Diagnostic only — nothing consumes these yet.
    ///
    /// ★ THE CROSS-CHECK THAT MAKES THIS WORTH LOGGING SEPARATELY: du and dv here are the SAME
    ///   displacement the rigid-shift monitor reports as tx and ty, arrived at by a completely
    ///   different route — two per-segment line offsets intersected, rather than a two-parameter
    ///   regression over every sample in the frame. If the detector is right the medians agree. If
    ///   they disagree the detector is wrong, and that is worth knowing BEFORE anything is built on
    ///   top of it.
    void RoomConcept::log_triple_points(const ImageEdgeObs& obs, std::int64_t timestamp_ms,
                                        const Eigen::Vector3f& pose)
    {
        ++triple_frames_;
        if (obs.triple_points.empty()) return;
        if (not triple_csv_.is_open())
        {
            triple_csv_.open("etc/image_edge_triple.csv", std::ios::out | std::ios::trunc);
            if (triple_csv_.is_open())
            {
                triple_csv_.imbue(std::locale::classic());   // CLAUDE.md: never a comma decimal
                triple_csv_ << "ts_ms,vertex,at_ceiling,u_pred,v_pred,u_meas,v_meas,du,dv,"
                               "suu,svv,suv,cond,n_corner,n_floor,"
                               // depth_raw as published; pred_fwd and pred_range are what the MODEL
                               // says at this pose. depth_raw ~= pred_fwd means the value is the
                               // forward coordinate (assumed); depth_raw ~= pred_range, with the
                               // excess growing toward the image edge, means it is range along the
                               // ray and xyz_from_pixel_depth needs the other formula.
                               "depth_raw,pred_fwd,pred_range,range_m,range_sigma,depth_dt_ms,"
                               "pose_x,pose_y,pose_theta\n";
            }
        }
        if (not triple_csv_.is_open()) return;
        // What the MODEL says this corner's distance is, in both conventions, so the logged
        // depth_raw can be compared against each. Camera Y is forward (CameraAPI::ray_from_pixel).
        const float cth = std::cos(pose.z()), sth = std::sin(pose.z());
        Eigen::Matrix3f Rm; Rm << cth, sth, 0.f, -sth, cth, 0.f, 0.f, 0.f, 1.f;
        const auto to_cam = [&](const Eigen::Vector3f& p_room)
        {
            const Eigen::Vector3f e(p_room.x() - pose.x(), p_room.y() - pose.y(), p_room.z());
            return Eigen::Vector3f(obs.cam_R_robot * (Rm * e) + obs.cam_t_robot);
        };
        const auto pred_fwd   = [&](const TriplePoint& t) { return to_cam(t.p_room).y(); };
        const auto pred_range = [&](const TriplePoint& t) { return to_cam(t.p_room).norm(); };
        for (const auto& t : obs.triple_points)
        {
            triple_csv_ << timestamp_ms << ',' << t.vertex << ','
                        << (t.from == ContourClass::WallCeiling ? 1 : 0) << ','
                        << t.uv_pred.x() << ',' << t.uv_pred.y() << ','
                        << t.uv_meas.x() << ',' << t.uv_meas.y() << ','
                        << (t.uv_meas.x() - t.uv_pred.x()) << ','
                        << (t.uv_meas.y() - t.uv_pred.y()) << ','
                        << t.cov_uv(0, 0) << ',' << t.cov_uv(1, 1) << ',' << t.cov_uv(0, 1) << ','
                        << t.cond << ',' << t.n_corner << ',' << t.n_floor << ','
                        << t.depth_raw << ',' << pred_fwd(t) << ',' << pred_range(t) << ','
                        << t.range_m << ',' << t.range_sigma << ','
                        << (obs.depth_stamp_ms ? obs.depth_stamp_ms - timestamp_ms : 0) << ','
                        << pose.x() << ',' << pose.y() << ',' << pose.z() << '\n';
            ++triple_rows_;
            if (t.from == ContourClass::WallCeiling) ++triple_ceil_; else ++triple_floor_;
        }
        triple_csv_.flush();

        // ── THE TRIPLE-POINT POSE FACTOR, IN SHADOW ──────────────────────────────────────────────
        // What it would be if it entered the loss: a 2-D projection residual per corner,
        //     r = uv_meas - project(pose, p_room),     W = cov_uv^-1
        // with the pose Jacobian the source already derives for its scalar samples, kept in full
        // rather than contracted onto a contour normal (a triple point has no aperture problem).
        //
        // ★ SELF-CONTAINED BY DESIGN. It forms its own 3x3 normal equations here instead of going
        //   through the GN solver, so there is no code path by which this can move the published
        //   pose — `drive` is not a flag that has to be respected, the influence does not exist.
        //   Building it this way first is the point: the gate for letting it drive is its own
        //   chi2/dof, and that has to be measured before the plumbing that could act on it exists.
        //
        // ★ WHAT THIS CANNOT SHOW. Corners sharing a floor segment are correlated (the detector
        //   pairs each vertex with whichever adjacent floor line carries more weight, so two corners
        //   of one wall can share it), and this treats them as independent. With a handful per frame
        //   and a covariance measured CONSERVATIVE by 36x in u / 2.3x in v against fixed-pose
        //   repeatability, the error is bounded and in the safe direction — but chi2 below 1 must be
        //   read as "conservative sigmas", not "better than the noise floor".
        Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
        Eigen::Vector3d g = Eigen::Vector3d::Zero();
        double chi2 = 0.0; int n_used = 0;
        const float cth2 = std::cos(pose.z()), sth2 = std::sin(pose.z());
        Eigen::Matrix3f Rm2; Rm2 << cth2, sth2, 0.f, -sth2, cth2, 0.f, 0.f, 0.f, 1.f;
        for (const auto& t : obs.triple_points)
        {
            const Eigen::Vector3f e(t.p_room.x() - pose.x(), t.p_room.y() - pose.y(), t.p_room.z());
            const Eigen::Vector3f p_robot = Rm2 * e;
            const Eigen::Vector3f p_cam   = obs.cam_R_robot * p_robot + obs.cam_t_robot;
            Eigen::Vector2d uvp;
            if (not rc::img::project_with_model(obs.cam, p_cam.cast<double>(), uvp)) continue;
            Eigen::Matrix<double, 2, 3> P;
            if (not rc::img::project_jacobian_model(obs.cam, p_cam.cast<double>(), P)) continue;
            // d(p_cam)/d(pose) — identical to the source's Jx, kept as 2x3 instead of contracted.
            Eigen::Matrix3f Jx;
            Jx.col(0) = obs.cam_R_robot * (-Rm2.col(0));
            Jx.col(1) = obs.cam_R_robot * (-Rm2.col(1));
            Jx.col(2) = obs.cam_R_robot * Eigen::Vector3f(p_robot.y(), -p_robot.x(), 0.f);
            const Eigen::Matrix<double, 2, 3> J = P * Jx.cast<double>();
            const Eigen::Matrix2d C = t.cov_uv.cast<double>();
            if (not (C.determinant() > 1e-12)) continue;
            const Eigen::Matrix2d W = C.inverse();
            // Wrap-safe in u: on the Ricoh's equirectangular model the column axis is cyclic, and a
            // corner sitting on the seam would otherwise contribute a residual of nearly a full
            // image width. Identity on a pinhole.
            const Eigen::Vector2d r(
                rc::img::du_wrapped(static_cast<double>(t.uv_meas.x()) - uvp.x(), obs.cam),
                static_cast<double>(t.uv_meas.y()) - uvp.y());
            H.noalias() += J.transpose() * W * J;
            g.noalias() += J.transpose() * W * r;
            chi2 += r.dot(W * r);
            ++n_used;
        }
        if (n_used == 0) return;
        ++tps_frames_;
        const double dof = std::max(1.0, 2.0 * n_used - 3.0);
        // ★ TWO CHI-SQUARES, AND ONLY THE SECOND IS THE GATE.
        //   `chi2` is evaluated at the LOCALISER's pose, so it measures how much this factor
        //   disagrees with the incumbent — useful, but it conflates a wrong pose with wrong sigmas
        //   and cannot answer "is the covariance honest?". `chi2_post` is what is LEFT after the
        //   three pose DOF have absorbed everything they can: chi2 - g' H^-1 g. That is the part no
        //   pose could explain, so it is the one that must approach 1 before this may drive.
        //   Verified in simulation: with a deliberately wrong pose the pre-fit value runs into the
        //   thousands while the recovered dpose is correct to ~1.5 mm, which is exactly the
        //   confusion this split removes.
        const double c2d = chi2 / dof;
        // ★ A SINGLE CORNER IS NOT A WASTED FRAME. A factor CONTRIBUTES information; it never has to
        //   be invertible on its own, and in a multi-modal graph which landmarks are visible changes
        //   from frame to frame by design. One corner is a rank-2 constraint on three DOF, and the
        //   SDF and LiDAR terms supply the rest — that is what the graph is for.
        //   The n>=2 gate below belongs to this DIAGNOSTIC, which has to invert to report a dpose.
        //   Writing 0 for those frames (as this did) reports a real contribution as an absence, and
        //   dragged every dpose median to exactly 0.000 over 31.5% of frames. NaN now, never 0.
        const double NAd = std::numeric_limits<double>::quiet_NaN();
        Eigen::Vector3d dp = Eigen::Vector3d::Constant(NAd);
        double cond = NAd, chi2_post = NAd, c2d_post = NAd;
        // The information this frame CONTRIBUTES, which is defined whatever the rank: reported so a
        // one-corner frame can be seen doing its job instead of looking like a hole in the data.
        const double info_trace = H.trace();
        if (n_used >= 2)
        {
            // ★ CORRELATION-NORMALISED, and the raw form is a trap this file already warns about
            //   for the mount fits: H mixes METRES and RADIANS, so its raw condition number is
            //   unit-dependent and says nothing about geometry. Measured on a real two-corner view
            //   the raw number reads 418 while the unit-free one reads 97 — and the posterior is
            //   5.6 mm / 2.4 mm / 0.083 deg, i.e. excellent. The raw number made a good factor look
            //   degenerate. What the normalised form shows is STRUCTURE, not weakness: corr(x,theta)
            //   ~ 0.98, the ordinary bearing-only ambiguity when the visible corners sit ahead and
            //   clustered, with y determined independently.
            const Eigen::Matrix3d Cv = H.inverse();
            if (Cv.allFinite())
            {
                Eigen::Matrix3d Rc = Eigen::Matrix3d::Identity();
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        Rc(i, j) = Cv(i, j) / std::sqrt(std::max(1e-300, Cv(i, i) * Cv(j, j)));
                const Eigen::Vector3d ev =
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(Rc).eigenvalues();
                cond = ev(2) / std::max(1e-12, ev(0));
            }
            // A tiny ridge only so a single visible corner cannot produce a NaN. It is NOT a prior:
            // if the geometry cannot determine a pose component, `cond` is what says so.
            dp = (H + 1e-9 * Eigen::Matrix3d::Identity()).ldlt().solve(g);
            chi2_post = std::max(0.0, chi2 - g.dot(dp));
            c2d_post  = chi2_post / dof;
        }
        if (std::isfinite(c2d_post)) { tps_chi2_sum_ += c2d_post; ++tps_chi2_n_; }
        if (not triple_pose_csv_.is_open())
        {
            triple_pose_csv_.open("etc/image_edge_triple_pose.csv", std::ios::out | std::ios::trunc);
            if (triple_pose_csv_.is_open())
            {
                triple_pose_csv_.imbue(std::locale::classic());
                triple_pose_csv_ << "ts_ms,n_corners,chi2,dof,chi2_per_dof,"
                                    "chi2_post,chi2_post_per_dof,info_trace,"
                                    "dpose_x,dpose_y,dpose_th,cond_H,pose_x,pose_y,pose_theta\n";
            }
        }
        if (triple_pose_csv_.is_open())
        {
            triple_pose_csv_ << timestamp_ms << ',' << n_used << ',' << chi2 << ',' << dof << ','
                             << c2d << ',' << chi2_post << ',' << c2d_post << ','
                             << info_trace << ','
                             << dp(0) << ',' << dp(1) << ',' << dp(2) << ','
                             << cond << ',' << pose.x() << ',' << pose.y() << ',' << pose.z() << '\n';
            triple_pose_csv_.flush();
        }
    }

    /// One weighted least squares per WINDOW — the accumulators reset after every solve — with the
    /// robot's pose logged beside it.
    ///
    /// ★ IT WAS CUMULATIVE, AND THAT HID THE ANSWER. Pooling over the whole run produced an estimate
    ///   that wandered from −0.087° to +0.565° while its own standard error shrank to ±0.0044°, and
    ///   crossed zero on the way — 149x its claimed uncertainty, for a quantity that is bolted to the
    ///   robot and cannot change at all. The formal error assumes each sample is an independent draw,
    ///   and they are not: every sample taken against one wall shares whatever is wrong with THAT
    ///   wall — its position in the map, the floor height beneath it, the residual pose error there.
    ///   Two hundred samples along one wall are close to one measurement repeated, which is exactly
    ///   the "N samples are not N measurements" correction image_edge_accumulate.h applies WITHIN a
    ///   frame; pooling reintroduced it ACROSS frames where nothing was correcting it.
    ///
    /// ★ A per-window row can be pooled back into the cumulative answer offline; the cumulative row
    ///   can never be taken apart. So this form strictly dominates, and it makes the between-window
    ///   scatter — the honest uncertainty — visible instead of hidden.
    ///
    /// What to do with the output: plot dpitch_deg against pose_x/pose_y. If it tracks which wall is
    /// in view, the wandering is MAP error and the mount cannot be measured this way until the map is
    /// fixed. If the windows agree once pose is accounted for, the mount number survives.
    ///
    /// ★ THE STANDARD ERRORS ARE INFLATED BY THE OBSERVED SCATTER, and that is not a fudge. The
    ///   per-frame monitor reported se_const_px = 0.013 px while consecutive frames disagreed by
    ///   tens of pixels: it was describing the spread WITHIN one frame's four samples and knew
    ///   nothing about the frames disagreeing. chi2/dof measures exactly that discrepancy — how much
    ///   larger the residuals are than the sigmas claim — so scaling the parameter sigma by
    ///   sqrt(chi2/dof) reports the precision the data actually supports. When the residual model is
    ///   right chi2/dof -> 1 and the inflation vanishes on its own.
    void RoomConcept::mount_pooled_solve(std::int64_t timestamp_ms)
    {
        if (mnt_win_start_ms_ == 0) { mnt_win_start_ms_ = timestamp_ms; return; }
        const std::int64_t win_ms = timestamp_ms - mnt_win_start_ms_;
        if (win_ms < 5000) return;
        // Both conditions, and NEITHER resets on its own: a window short of samples keeps
        // accumulating rather than being solved thin and reported as if it were a measurement. The
        // window length is therefore variable and is logged, so a row can never be misread as
        // covering 5 s when it covered thirty.
        // ★ EITHER monitor having enough samples opens the window. The two draw on DIFFERENT
        // populations — the fy/d fit on floor junctions only, the translation fit on every class —
        // so gating both on the first would silence the translation fit in exactly the
        // corners-only configuration (useFloorJunction = false) where it is the only one of the two
        // still able to answer.
        if (mnt_n_ < 200 and mnt_tn_ < 200) return;

        // ── Monitor 1: b_const + b_invd * (fy/d), floor junctions ────────────────────────────────
        // ★ NaN, not 0, when the window could not be solved. A 0 here would read as "measured no
        // pitch error" when it means "never asked", and that conflation has cost this codebase real
        // time before. NaN survives the CSV round trip and cannot be averaged by accident.
        const double NA = std::numeric_limits<double>::quiet_NaN();
        double rho = NA, cond = NA, b_const = NA, b_invd = NA, c2d = NA, se_const = NA, se_invd = NA;
        double dpitch_deg = NA, se_pitch_deg = NA;
        const double det  = mnt_S11_ * mnt_S22_ - mnt_S12_ * mnt_S12_;
        const bool   m_ok = mnt_n_ >= 200 and det > 0.0 and std::isfinite(det);
        if (m_ok)
        {

        // ── COULD THIS WINDOW SEPARATE PITCH FROM HEIGHT AT ALL? ─────────────────────────────────
        // The two parameters are told apart ONLY by how their covariates scale with range: pitch is
        // constant in pixels, height goes as fy/d. If every sample in the window sat at the same
        // distance, x2 = fy/d would be constant, S12 would equal S11*x2 exactly, and the normal
        // matrix would be singular — the fit would still return two numbers, and they would be
        // meaningless, trading off along the degenerate direction. That is what was happening: the
        // per-window pitch estimate wandered -0.087..+0.565 deg while its own standard error shrank
        // to +/-0.0044, and the worst rows were exactly the sparse, narrow-range ones.
        //
        // ★ CORRELATION-NORMALISED, following calibration_estimator.h, and for the same reason it
        // records: on the RAW matrix the units span orders of magnitude regardless of geometry, and
        // the raw number once ranked a separable window as WORSE than a deliberately collinear one.
        // Normalising by the diagonal leaves a 2x2 with 1s on it and the correlation off it, whose
        // eigenvalues are 1 +/- |rho|. So for this fit the whole diagnostic is one number:
        //     rho  = S12 / sqrt(S11*S22)        how collinear the two covariates were
        //     cond = (1+|rho|) / (1-|rho|)
        // rho -> 1 means the window saw one range and cannot answer; it is a direct readout of the
        // RANGE DIVERSITY the trajectory happened to supply.
        // ★ It is REPORTED, not gated. "The estimate wanders" and "this window was never able to
        // answer" are different facts, and a filter that silently dropped the second would leave the
        // first looking like noise.
            rho  = mnt_S12_ / std::sqrt(std::max(1e-300, mnt_S11_ * mnt_S22_));
            const double arho = std::min(std::abs(rho), 1.0 - 1e-12);
            cond = (1.0 + arho) / (1.0 - arho);
            b_const = ( mnt_S22_ * mnt_Sy1_ - mnt_S12_ * mnt_Sy2_) / det;
            b_invd  = (-mnt_S12_ * mnt_Sy1_ + mnt_S11_ * mnt_Sy2_) / det;
            // Weighted residual sum of squares of the FITTED model, so chi2/dof is a statement about
            // the part the two mount parameters could not explain.
            const double chi2 = std::max(0.0, mnt_Syy_ - (b_const * mnt_Sy1_ + b_invd * mnt_Sy2_));
            const double dof  = std::max(1.0, static_cast<double>(mnt_n_) - 2.0);
            c2d  = chi2 / dof;
            const double infl = std::sqrt(std::max(1.0, c2d));
            se_const = std::sqrt(mnt_S22_ / det) * infl;
            se_invd  = std::sqrt(mnt_S11_ / det) * infl;
            const double fy = last_mount_fy_ > 0.f ? last_mount_fy_ : 1.f;
            dpitch_deg   = (b_const / fy) * 180.0 / M_PI;
            se_pitch_deg = (se_const / fy) * 180.0 / M_PI;
        }

        // ── Monitor 2: tx*nx + ty*ny — the rigid image displacement of the PREDICTION ────────────
        // Sign: r = n_hat . (uv_pred - uv_meas), so a positive tx means the projected room sits to
        // the RIGHT of the edge actually found, and positive ty means it sits BELOW.
        // Interpretation is deliberately NOT unique and must not be reported as if it were: an
        // image translation is produced identically by a principal-point offset, a boresight yaw
        // (tx ~ fx*dyaw) and a mount pitch (ty ~ fy*dpitch). What it settles is the question the
        // fy/d fit cannot even pose — whether a rigid pointing error is present AT ALL.
        double tx = NA, ty = NA, se_tx = NA, se_ty = NA, c2d_t = NA, rho_t = NA, cond_t = NA;
        const double det_t = mnt_T11_ * mnt_T22_ - mnt_T12_ * mnt_T12_;
        const bool   t_ok  = mnt_tn_ >= 200 and det_t > 0.0 and std::isfinite(det_t);
        if (t_ok)
        {
            // Same correlation-normalised conditioning as monitor 1, and here it reads the SPREAD
            // OF CONTOUR ORIENTATIONS rather than of ranges: rho_t -> +/-1 means every normal in
            // the window pointed the same way (all floor junctions, or all corners) and the two
            // components only trade off. Reported, never gated.
            rho_t = mnt_T12_ / std::sqrt(std::max(1e-300, mnt_T11_ * mnt_T22_));
            const double ar = std::min(std::abs(rho_t), 1.0 - 1e-12);
            cond_t = (1.0 + ar) / (1.0 - ar);
            tx = ( mnt_T22_ * mnt_Tx_ - mnt_T12_ * mnt_Ty_) / det_t;
            ty = (-mnt_T12_ * mnt_Tx_ + mnt_T11_ * mnt_Ty_) / det_t;
            const double chi2_t = std::max(0.0, mnt_Tyy_ - (tx * mnt_Tx_ + ty * mnt_Ty_));
            c2d_t = chi2_t / std::max(1.0, static_cast<double>(mnt_tn_) - 2.0);
            const double infl_t = std::sqrt(std::max(1.0, c2d_t));
            se_tx = std::sqrt(mnt_T22_ / det_t) * infl_t;
            se_ty = std::sqrt(mnt_T11_ / det_t) * infl_t;
        }

        // ── Monitor 3 / STAGE 1: the four mount nuisances solved as parameters ───────────────────
        // ★ SIGN. The residual is r = n_hat.(uv_pred(0) - uv_meas) and h = d(n_hat.uv_pred)/d(nuisance).
        //   The measurement is the projection under the TRUE mount, so n_hat.uv_meas =
        //   n_hat.uv_pred(0) + h*x_true, giving r = -h*x_true. A least-squares fit of r on h therefore
        //   returns MINUS the parameter: the physical value, and the correction to apply, is -x.
        //   Self-checking: mountYawCorrection is already applied, so p_yaw must now read ~0. If it
        //   reads about twice the applied correction instead, this sign is inverted.
        // ★ The prior is the IDENTITY because h carries sigma_i (see the header note).
        Eigen::Vector4d mp = Eigen::Vector4d::Constant(NA), msig = Eigen::Vector4d::Constant(NA);
        double c2d_h = NA, cond_h = NA, rho_h = 0.0;
        int    informed_mask = 0, rho_i = 0, rho_j = 1;
        const bool h_ok = mnt_hn_ >= 200;
        if (h_ok)
        {
            const Eigen::Matrix4d A = mnt_H_ + Eigen::Matrix4d::Identity();
            const Eigen::Matrix4d C = A.inverse();
            if (C.allFinite())
            {
                const Eigen::Vector4d x = C * mnt_b_;
                // mnt_Tyy_ is the same weighted sum of squares over the same samples.
                const double chi2_h = std::max(0.0, mnt_Tyy_ - x.dot(mnt_b_));
                c2d_h = chi2_h / std::max(1.0, static_cast<double>(mnt_hn_) - 4.0);
                const double infl_h = std::sqrt(std::max(1.0, c2d_h));
                mp = -x;
                for (int i = 0; i < 4; ++i)
                {
                    msig(i) = std::sqrt(std::max(0.0, C(i, i))) * infl_h;
                    // `informed` = the data shrank this parameter's uncertainty below 0.9x its prior.
                    // Per-parameter, never global: three of these can be answered while the fourth is
                    // pure prior, and a global flag would licence acting on the one that is not.
                    if (msig(i) < 0.9) informed_mask |= (1 << i);
                }
                // ── COULD THESE FOUR BE TOLD APART AT ALL? ───────────────────────────────────
                // ★ `informed` PER PARAMETER CANNOT SEE A DEGENERATE PAIR. A strong prior shrinks
                //   the posterior sigma even along a direction the data never constrained, so a
                //   parameter can read INFORMED while only its COMBINATION with another is
                //   determined. Pitch and height are separated solely by the fy/d covariate — the
                //   same two columns whose 2-parameter fit sits at rho 0.9864 / cond 146 standing
                //   still. Reporting them individually without this would repeat, in a 4x4, exactly
                //   the error the fy/d monitor already made.
                // Correlation-normalised, per calibration_estimator.h: on the raw matrix the units
                // span orders of magnitude regardless of geometry, and the raw number once ranked a
                // separable window as WORSE than a deliberately collinear one.
                Eigen::Matrix4d R = Eigen::Matrix4d::Identity();
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 4; ++j)
                        R(i, j) = C(i, j) / std::sqrt(std::max(1e-300, C(i, i) * C(j, j)));
                const Eigen::Vector4d ev = Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d>(R)
                                               .eigenvalues();
                cond_h = ev(3) / std::max(1e-12, ev(0));
                for (int i = 0; i < 4; ++i)              // worst pair, named rather than implied
                    for (int j = i + 1; j < 4; ++j)
                        if (std::abs(R(i, j)) > std::abs(rho_h))
                        { rho_h = R(i, j); rho_i = i; rho_j = j; }
                mnt_p_sum_  += mp;
                mnt_p_sum2_ += mp.cwiseProduct(mp);
                ++mnt_p_n_;
            }
        }

        if (not mount_csv_.is_open())
        {
            mount_csv_.open("etc/image_edge_mount.csv", std::ios::out | std::ios::trunc);
            if (mount_csv_.is_open())
            {
                mount_csv_.imbue(std::locale::classic());   // CLAUDE.md: never a comma decimal
                mount_csv_ << "ts_ms,window_ms,n_samples,chi2_per_dof,b_const_px,se_const_px,"
                              "b_invd_m,se_invd_m,dpitch_deg,se_pitch_deg,"
                              "pose_x,pose_y,pose_theta,rho,cond,"
                              "t_n,tx_px,se_tx_px,ty_px,se_ty_px,chi2_t,rho_t,cond_t,"
                              "h_n,p_pitch_rad,p_height_m,p_yaw_rad,p_dt,"
                              "s_pitch,s_height,s_yaw,s_dt,chi2_h,informed,cond_h,rho_h\n";
            }
        }
        if (mount_csv_.is_open())
        {
            mount_csv_ << timestamp_ms << ',' << win_ms << ',' << mnt_n_ << ',' << c2d << ','
                       << b_const << ',' << se_const << ',' << b_invd << ',' << se_invd << ','
                       << dpitch_deg << ',' << se_pitch_deg << ','
                       << mnt_pose_x_ << ',' << mnt_pose_y_ << ',' << mnt_pose_th_ << ','
                       << rho << ',' << cond << ','
                       << mnt_tn_ << ',' << tx << ',' << se_tx << ',' << ty << ',' << se_ty << ','
                       << c2d_t << ',' << rho_t << ',' << cond_t << ','
                       // Parameters written in PHYSICAL units; sigmas stay in units of the prior,
                       // where 1.0 means "the data said nothing" and <0.9 is `informed`.
                       << mnt_hn_ << ','
                       << mp(0) * params.image_edge.mount_pitch_sigma  << ','
                       << mp(1) * params.image_edge.mount_height_sigma << ','
                       << mp(2) * params.image_edge.mount_yaw_sigma    << ',' << mp(3) << ','
                       << msig(0) << ',' << msig(1) << ',' << msig(2) << ',' << msig(3) << ','
                       << c2d_h << ',' << informed_mask << ',' << cond_h << ',' << rho_h << '\n';
            mount_csv_.flush();
        }
        // Between-window scatter, which is the uncertainty that turned out to matter. Kept as a
        // running mean and sum of squares so the log line can say how much the WINDOWS disagree
        // beside how much each window claims to know — the two differed by 149x in the pooled form.
        ++mnt_wins_;
        if (m_ok) { mnt_pitch_sum_ += dpitch_deg; mnt_pitch_sum2_ += dpitch_deg * dpitch_deg; ++mnt_pitch_n_; }
        // Report the WITHIN-window error beside the BETWEEN-window scatter. If the second is much
        // larger than the first, the samples are not independent and the first is meaningless — say
        // so on the line rather than leaving a reader to discover it from a table of rows.
        double spread = 0.0;
        if (mnt_pitch_n_ >= 2)
        {
            const double m = mnt_pitch_sum_ / mnt_pitch_n_;
            spread = std::sqrt(std::max(0.0, mnt_pitch_sum2_ / mnt_pitch_n_ - m * m));
        }
        qInfo().nospace().noquote() << "[mount] window " << mnt_wins_ << " (" << win_ms << " ms, "
                          << mnt_n_ << " samples) | pitch "
                          << QString::number(dpitch_deg, 'f', 4) << " +/- "
                          << QString::number(se_pitch_deg, 'f', 4) << " deg within"
                          << (mnt_pitch_n_ >= 2
                              ? QString(", %1 deg BETWEEN windows (%2x)")
                                    .arg(spread, 0, 'f', 4)
                                    .arg(spread / std::max(1e-9, se_pitch_deg), 0, 'f', 0)
                              : QString())
                          << " | height " << QString::number(b_invd, 'f', 4)
                          << " | chi2/dof " << QString::number(c2d, 'f', 2)
                          << " | rho " << QString::number(rho, 'f', 4)
                          << " cond " << QString::number(cond, 'f', 0)
                          << (m_ok and cond > 50.0
                                  ? "  <- DEGENERATE: one range in view, pitch and height are not"
                                    " separable here and these two numbers only trade off"
                                  : m_ok ? "" : "  <- NOT SOLVED (too few floor-junction samples)")
                          << " | pose " << QString::number(mnt_pose_x_, 'f', 2) << ","
                          << QString::number(mnt_pose_y_, 'f', 2)
                          << (m_ok and c2d > 4.0
                                  ? "  <- residual model is WRONG; the estimate inherits it" : "");
        // Monitor 2 on its OWN line. It answers a different question from a different population,
        // and folding it into the line above would invite reading one window's two fits as one
        // estimate of one thing.
        if (t_ok)
            qInfo().nospace().noquote() << "[mount/shift] window " << mnt_wins_ << " (" << mnt_tn_
                              << " samples, all classes) | tx "
                              << QString::number(tx, 'f', 3) << " +/- "
                              << QString::number(se_tx, 'f', 3) << " px | ty "
                              << QString::number(ty, 'f', 3) << " +/- "
                              << QString::number(se_ty, 'f', 3) << " px | |t| "
                              << QString::number(std::hypot(tx, ty), 'f', 3) << " px"
                              << " | chi2/dof " << QString::number(c2d_t, 'f', 2)
                              << " | rho " << QString::number(rho_t, 'f', 4)
                              << " cond " << QString::number(cond_t, 'f', 1)
                              << (cond_t > 50.0
                                      ? "  <- one contour ORIENTATION in view; tx and ty are not"
                                        " separable here and only trade off"
                                      : "")
                              << (std::hypot(tx, ty) > 3.0 * std::max(se_tx, se_ty)
                                      ? "  <- SIGNIFICANT rigid displacement: pointing or principal"
                                        " point, NOT distinguishable from each other by this fit"
                                      : "  <- no rigid displacement resolved");
        else
            qInfo().nospace().noquote() << "[mount/shift] window " << mnt_wins_ << ": only " << mnt_tn_
                              << " weighted samples (need 200) — NOT solved, which is not the same"
                                 " as 'no displacement found'";

        // ── Monitor 3 line: the calibration itself ───────────────────────────────────────────────
        if (h_ok and std::isfinite(mp(0)))
        {
            const double sig[4] = {params.image_edge.mount_pitch_sigma,
                                   params.image_edge.mount_height_sigma,
                                   params.image_edge.mount_yaw_sigma, 1.0};
            const char*  nm[4]  = {"pitch", "height", "yaw", "dt"};
            const char*  un[4]  = {"deg", "m", "deg", "x"};
            QString body;
            for (int i = 0; i < 4; ++i)
            {
                double v = mp(i) * sig[i];
                if (i == 0 or i == 2) v *= 180.0 / M_PI;          // report angles in degrees
                // Between-window scatter is the honest uncertainty on any parameter that cannot be
                // separated within a window (yaw vs heading); the within-window sigma is not.
                double spread = 0.0;
                if (mnt_p_n_ >= 2)
                {
                    const double m = mnt_p_sum_(i) / mnt_p_n_;
                    spread = std::sqrt(std::max(0.0, mnt_p_sum2_(i) / mnt_p_n_ - m * m)) * sig[i];
                    if (i == 0 or i == 2) spread *= 180.0 / M_PI;
                }
                body += QString(" | %1 %2 %3 (%4 prior sig%5)")
                            .arg(nm[i]).arg(v, 0, 'f', i == 1 ? 4 : 4).arg(un[i])
                            .arg(msig(i), 0, 'f', 3)
                            .arg((informed_mask >> i) & 1 ? ", INFORMED" : "");
                if (mnt_p_n_ >= 2)
                    body += QString(" [%1 between windows]").arg(spread, 0, 'f', 4);
            }
            qInfo().nospace().noquote()
                << "[mount/calib] window " << mnt_wins_ << " (" << mnt_hn_ << " samples)"
                << body << " | chi2/dof " << QString::number(c2d_h, 'f', 2)
                << " | cond " << QString::number(cond_h, 'f', 1)
                << " (worst pair " << nm[rho_i] << "/" << nm[rho_j] << " rho "
                << QString::number(rho_h, 'f', 4) << ")"
                << (cond_h > 50.0
                        ? "   <- DEGENERATE: that pair trades off; only their COMBINATION is"
                          " determined, and a per-parameter INFORMED flag cannot see this"
                        : "")
                << (informed_mask == 0
                        ? "   <- nothing INFORMED yet: every posterior is still its prior"
                        : "");
        }

        // ── RESET: every row is an independent window ────────────────────────────────────────────
        mnt_S11_ = mnt_S12_ = mnt_S22_ = mnt_Sy1_ = mnt_Sy2_ = mnt_Syy_ = 0.0;
        mnt_T11_ = mnt_T12_ = mnt_T22_ = mnt_Tx_  = mnt_Ty_  = mnt_Tyy_ = 0.0;
        mnt_H_.setZero(); mnt_b_.setZero(); mnt_hn_ = 0;
        mnt_n_ = 0;
        mnt_tn_ = 0;
        mnt_win_start_ms_ = timestamp_ms;
    }

    /// One line every 5 s naming which of the three silences we are in.
    void RoomConcept::imgedge_health(std::int64_t timestamp_ms)
    {
        if (imgedge_health_last_ms_ == 0) { imgedge_health_last_ms_ = timestamp_ms; return; }
        if (timestamp_ms - imgedge_health_last_ms_ < 5000) return;
        qInfo().nospace().noquote()
            << "[triple] " << triple_rows_ << " points (" << triple_floor_ << " floor, "
            << triple_ceil_ << " ceiling) over " << triple_frames_ << " frames ("
            << QString::number(triple_frames_ ? double(triple_rows_) / triple_frames_ : 0.0, 'f', 2)
            << "/frame)"
            << (triple_rows_ == 0 and triple_frames_ > 0
                    ? "  <- frames ARE arriving and no vertex resolved: check that wall corners AND"
                      " floor junctions are both enabled, they are intersected in pairs"
                    : "")
            << (tps_chi2_n_ > 0
                    ? QString(" | POSE FACTOR (shadow, cannot drive): POST-fit chi2/dof %1 over"
                              " %2 frames — this is the gate, and it must approach 1")
                          .arg(tps_chi2_sum_ / tps_chi2_n_, 0, 'f', 2).arg(tps_frames_)
                    : QString());
        qInfo().nospace() << "[imgedge] " << imgedge_rows_ << " rows / " << imgedge_calls_
                          << " calls | no_window=" << imgedge_no_window_
                          << " no_obs=" << imgedge_no_obs_ << " (bound, nothing extracted)"
                          << " no_cam=" << imgedge_no_cam_ << " (no valid intrinsics/extrinsic)";
        imgedge_health_last_ms_ = timestamp_ms;
        imgedge_rows_ = imgedge_calls_ = imgedge_no_window_ = imgedge_no_obs_ = imgedge_no_cam_ = 0;
    }

    void RoomConcept::run_gn_shadow(const std::vector<Eigen::Vector3f>& poses_before,
                                    const std::vector<Eigen::Vector3f>& poses_after,
                                    float authority_loss, int authority_iters, float authority_ms,
                                    std::int64_t timestamp_ms)
    {
        if (poses_before.size() != window_mgr_.window.size() or poses_before.empty()) return;

        gn::Input in;
        in.model           = model_.get();
        in.params          = &params;
        in.window          = &window_mgr_.window;
        in.boundary_prior  = &window_mgr_.boundary_prior;
        in.boundary_weight = boundary_weight_now();
        in.device          = get_device();

        gn::Options opts;
        opts.max_iters    = params.gn_max_iters;
        opts.lambda_init  = params.gn_lambda_init;
        opts.step_tol     = params.gn_step_tol;
        opts.loss_rel_tol = params.gn_loss_rel_tol;

        // Both backends start from the SAME state — that is the whole point of the comparison.
        auto poses_gn = poses_before;
        const auto t0 = std::chrono::high_resolution_clock::now();
        const auto r = gn::solve(in, poses_gn, opts);
        const float gn_ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();

        // Cross-scoring: each backend's answer measured on BOTH objectives. If the GN objective and
        // compute_rfe_loss ever disagree about which pose is better, these four numbers say so.
        // A RELATIVE gradient check is only meaningful in a band, and both obvious sample points miss it:
        //   at the converged pose  — the true gradient is ~0, so the analytic b and its finite difference
        //                            are both float noise and their ratio is arbitrary (measured 0.78
        //                            live, and reproduced in gn_selftest, on Jacobians that are correct);
        //   at the pre-solve pose  — can be far out, where the polygon SDF is piecewise: a ±1e-3 probe
        //                            re-assigns points to a different wall segment and the central
        //                            difference is no longer a fair reference (5.9e-2 in gn_selftest).
        // The valid probe is just OFF the optimum: a real gradient, still inside the smooth basin.
        // grad_relerr keeps the pre-solve sample for continuity; grad_relerr_basin is the column that
        // actually indicts the Jacobian.
        static constexpr float kProbeXY  = 0.005f;   // 5 mm
        static constexpr float kProbeYaw = 0.002f;   // 2 mrad
        const float grad_relerr = params.gn_grad_check
            ? gn::gradient_check(in, poses_before)
            : std::numeric_limits<float>::quiet_NaN();
        float grad_relerr_basin = std::numeric_limits<float>::quiet_NaN();
        if (params.gn_grad_check and r.ok)
        {
            auto probe = poses_gn;
            for (auto& p : probe) p += Eigen::Vector3f(kProbeXY, kProbeXY, kProbeYaw);
            grad_relerr_basin = gn::gradient_check(in, probe);
        }
        const float gn_obj_at_authority = gn::evaluate(in, poses_after);
        const float gn_obj_at_gn        = r.ok ? gn::evaluate(in, poses_gn)
                                               : std::numeric_limits<float>::quiet_NaN();
        float torch_obj_at_gn = std::numeric_limits<float>::quiet_NaN();
        if (r.ok)
        {
            torch::NoGradGuard no_grad;
            write_window_poses(poses_gn);
            torch_obj_at_gn = window_mgr_.compute_rfe_loss(*model_, params, get_device(),
                                                            in.boundary_weight).item<float>();
        }
        write_window_poses(poses_after);   // the authority always keeps the pose
        // Measured AFTER the restore, at the authority's own poses. loss_auth is NOT a substitute:
        // Adam reports the loss from before its final step, so only this column is comparable with
        // gn_obj_at_auth. The pair (torch_obj_at_auth, gn_obj_at_auth) at one set of poses is what
        // proves the two backends are minimising the same objective.
        float torch_obj_at_auth = std::numeric_limits<float>::quiet_NaN();
        {
            torch::NoGradGuard no_grad;
            torch_obj_at_auth = window_mgr_.compute_rfe_loss(*model_, params, get_device(),
                                                              in.boundary_weight).item<float>();
        }

        const size_t last = poses_after.size() - 1;
        const float dx  = r.ok ? poses_gn[last].x() - poses_after[last].x() : 0.f;
        const float dy  = r.ok ? poses_gn[last].y() - poses_after[last].y() : 0.f;
        const float dth_raw = poses_gn[last].z() - poses_after[last].z();
        const float dth = r.ok ? std::atan2(std::sin(dth_raw), std::cos(dth_raw)) : 0.f;
        float max_dxy = 0.f;
        if (r.ok)
            for (size_t i = 0; i < poses_after.size(); ++i)
                max_dxy = std::max(max_dxy, (poses_gn[i].head<2>() - poses_after[i].head<2>()).norm());

        if (not gn_shadow_csv_.is_open())
        {
            gn_shadow_csv_.open(params.gn_shadow_csv_path, std::ios::out | std::ios::trunc);
            if (gn_shadow_csv_.is_open())
            {
                gn_shadow_csv_.imbue(std::locale::classic());   // CLAUDE.md: never emit a comma decimal
                gn_shadow_csv_ << "ts_ms,window_size,ok,"
                                  "iters_auth,loss_auth,ms_auth,"
                                  "iters_gn,rejected_gn,loss_gn,ms_gn,lambda_gn,grad_norm_gn,step_gn,"
                                  "dx,dy,dth,max_dxy,"
                                  "torch_obj_at_auth,gn_obj_at_auth,torch_obj_at_gn,gn_obj_at_gn,loss_init_gn,"
                                  "grad_relerr,grad_relerr_basin\n";
            }
        }
        if (gn_shadow_csv_.is_open())
        {
            gn_shadow_csv_ << timestamp_ms
                << ',' << window_mgr_.size()
                << ',' << (r.ok ? 1 : 0)
                << ',' << authority_iters << ',' << authority_loss << ',' << authority_ms
                << ',' << r.iterations << ',' << r.rejected << ',' << r.loss << ',' << gn_ms
                << ',' << r.lambda_final << ',' << r.grad_norm << ',' << r.step_norm
                << ',' << dx << ',' << dy << ',' << dth << ',' << max_dxy
                << ',' << torch_obj_at_auth << ',' << gn_obj_at_authority
                << ',' << torch_obj_at_gn << ',' << gn_obj_at_gn
                << ',' << r.loss_init << ',' << grad_relerr << ',' << grad_relerr_basin << '\n';
            gn_shadow_csv_.flush();
        }
    }

    std::pair<float, int> RoomConcept::run_adam_loop(const OdometryPrior& odometry_prior)
    {
        auto window_params = window_mgr_.collect_params();

        const int ws = static_cast<int>(window_mgr_.size());
        const float ws_scale = 1.0f / std::sqrt(static_cast<float>(std::max(1, ws)));
        const float lr = params.learning_rate_pos * ws_scale;
        torch::optim::Adam optimizer(
            {torch::optim::OptimizerParamGroup(window_params,
                std::make_unique<torch::optim::AdamOptions>(lr))});

        const Eigen::Vector3f velocity_weights = params.velocity_adaptive_weights
            ? compute_velocity_adaptive_weights(odometry_prior)
            : Eigen::Vector3f::Ones();

        // ===== Boundary quality gate =====
        // Scale the boundary prior by how trustworthy the previous frame's pose was.
        // w = min(1, sigma_sdf² / sdf_mse_prev)
        // Good prev pose (sdf_mse_prev ≈ 0) → w≈1 (strong prior, normal behaviour).
        // Bad  prev pose (sdf_mse_prev >> sigma_sdf) → w→0 (prior suppressed, ADAM free to recover).
        // Boundary precision scale. Legacy: quality gate min(1, σ_sdf²/sdf_mse_prev). Hierarchical
        // (HIERARCHICAL_PRECISION.md): inferred π=exp(u_b_) from the previous frame's boundary residual,
        // predicted top-down by the map_trust hyper-state. Mutually exclusive; hierarchical wins when on.
        const float boundary_weight = boundary_weight_now();

        float last_loss = std::numeric_limits<float>::infinity();
        float prev_loss = std::numeric_limits<float>::infinity();
        int iterations = 0;

        last_adam_losses_.clear();
        last_adam_losses_.reserve(params.num_iterations);
        last_loss_init_ = 0.f;

        for (int i = 0; i < params.num_iterations; ++i)
        {
            optimizer.zero_grad();

            const torch::Tensor loss = window_mgr_.compute_rfe_loss(*model_, params, get_device(),
                                                                      boundary_weight);

            // Record initial loss (before any parameter update) for convergence diagnostics
            if (i == 0)
                last_loss_init_ = loss.item<float>();

            loss.backward();

            if (params.velocity_adaptive_weights)
            {
                torch::NoGradGuard no_grad;
                auto& newest_pose = window_mgr_.newest().pose;
                if (newest_pose.grad().defined())
                {
                    newest_pose.mutable_grad().index({0}) *= velocity_weights[0];
                    newest_pose.mutable_grad().index({1}) *= velocity_weights[1];
                    newest_pose.mutable_grad().index({2}) *= velocity_weights[2];
                }
            }

            optimizer.step();

            prev_loss = last_loss;
            last_loss = loss.item<float>();
            last_adam_losses_.push_back(last_loss);
            iterations = i + 1;

            if (last_loss < params.min_loss_threshold)
                break;
            if (i > params.convergence_min_iters &&
                std::abs(prev_loss - last_loss) < params.convergence_relative_tol * prev_loss)
                break;
        }

        last_lbfgs_grad_norm_ = 0.f;
        return {last_loss, iterations};
    }

    // =========================================================================
    //  L-BFGS optimisation loop
    // =========================================================================
    // Replaces Adam when params.optimizer_type == "LBFGS".
    //
    // A single optimizer.step(closure) call executes up to params.num_iterations
    // Newton steps internally, each with a strong-Wolfe line search that finds
    // the right step size automatically — no need to tune learning_rate_pos.
    //
    // velocity_adaptive_weights are applied inside the closure (after backward)
    // so L-BFGS sees the scaled gradients consistently across all evaluations.
    // This is equivalent to optimising in a velocity-weighted parameter space.
    //
    // iter_count tracks function evaluations (closure calls), not Newton steps.
    // With strong_wolfe, each Newton step typically calls the closure 2-4 times.
    // =========================================================================
    std::pair<float, int> RoomConcept::run_lbfgs_loop(const OdometryPrior& odometry_prior)
    {
        auto window_params = window_mgr_.collect_params();

        torch::optim::LBFGS optimizer(
            window_params,
            torch::optim::LBFGSOptions(static_cast<double>(params.lbfgs_lr))
                .max_iter(params.num_iterations)
                .history_size(static_cast<int64_t>(params.lbfgs_history_size))
                .line_search_fn(std::string("strong_wolfe"))
                .tolerance_grad(params.lbfgs_tolerance_grad)
                .tolerance_change(params.lbfgs_tolerance_change));

        const Eigen::Vector3f velocity_weights = compute_velocity_adaptive_weights(odometry_prior);

        // Boundary precision scale. Legacy: quality gate min(1, σ_sdf²/sdf_mse_prev). Hierarchical
        // (HIERARCHICAL_PRECISION.md): inferred π=exp(u_b_) from the previous frame's boundary residual,
        // predicted top-down by the map_trust hyper-state. Mutually exclusive; hierarchical wins when on.
        const float boundary_weight = boundary_weight_now();

        float last_loss = std::numeric_limits<float>::infinity();
        int iter_count = 0;

        last_adam_losses_.clear();
        last_adam_losses_.reserve(params.num_iterations);
        last_loss_init_ = 0.f;

        auto closure = [&]() -> torch::Tensor {
            optimizer.zero_grad();

            const torch::Tensor loss = window_mgr_.compute_rfe_loss(
                *model_, params, get_device(), boundary_weight);
            loss.backward();

            if (params.velocity_adaptive_weights)
            {
                torch::NoGradGuard no_grad;
                auto& newest_pose = window_mgr_.newest().pose;
                if (newest_pose.grad().defined())
                {
                    newest_pose.mutable_grad().index({0}) *= velocity_weights[0];
                    newest_pose.mutable_grad().index({1}) *= velocity_weights[1];
                    newest_pose.mutable_grad().index({2}) *= velocity_weights[2];
                }
            }

            const float lv = loss.item<float>();
            if (iter_count == 0) last_loss_init_ = lv;
            last_loss = lv;
            last_adam_losses_.push_back(lv);
            ++iter_count;

            return loss;
        };

        optimizer.step(closure);

        // Gradient infinity norm: measures how far we are from the KKT condition.
        // Near zero → converged; large → stopped early (max_iter or tolerance_change hit first).
        last_lbfgs_grad_norm_ = 0.f;
        {
            torch::NoGradGuard ng;
            for (auto& p : window_params)
                if (p.grad().defined())
                    last_lbfgs_grad_norm_ = std::max(last_lbfgs_grad_norm_,
                        p.grad().abs().max().item<float>());
        }

        return {last_loss, iter_count};
    }

    std::pair<Eigen::Matrix3f, float> RoomConcept::compute_posterior_covariance(
        const torch::Tensor& points_tensor)
    {
        try {
            auto& newest = window_mgr_.newest();
            auto pose_for_hess = newest.pose.clone().detach().requires_grad_(true);
            auto pose_xy = pose_for_hess.index({torch::indexing::Slice(0, 2)});
            auto pose_theta_h = pose_for_hess.index({torch::indexing::Slice(2, 3)});

            torch::Tensor likelihood_loss =
                compute_observation_loss(*model_, params, points_tensor, pose_xy, pose_theta_h);

            // The RGB edge term is a LIKELIHOOD, so once it drives the pose it must also inform the
            // reported covariance — otherwise the agent would act on evidence it does not admit to
            // having, and sigma_theta (the pre-registered success criterion, and what feeds the
            // controller's speed governor and every concept agent's precision) could never move.
            // ★ Gated on DRIVE, not merely enable: shadow mode must not change any published number.
            if (params.image_edge.enable and params.image_edge.drive and not newest.image_edges.empty())
                likelihood_loss = likelihood_loss + ImageEdgeFactor::loss(
                    newest.image_edges, pose_xy, pose_theta_h, params.image_edge, get_device());

            Eigen::Matrix3f H_likelihood = autograd_hessian_3x3(likelihood_loss, pose_for_hess);

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(H_likelihood);
            Eigen::Vector3f evals = eig.eigenvalues().cwiseMax(params.eigenvalue_clamp_posterior);
            H_likelihood = eig.eigenvectors() * evals.asDiagonal() * eig.eigenvectors().transpose();

            Eigen::Matrix3f prior_precision = current_covariance.inverse();
            const float lambda = params.covariance_regularization;
            Eigen::Matrix3f posterior_precision = prior_precision + H_likelihood
                                                 + lambda * Eigen::Matrix3f::Identity();

            Eigen::Matrix3f new_cov = posterior_precision.inverse();

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(posterior_precision);
            const auto eigenvalues = solver.eigenvalues();
            const float max_ev = eigenvalues.maxCoeff();
            const float min_ev = eigenvalues.minCoeff();
            const float cond = (min_ev > 1e-8f) ? (max_ev / min_ev) : 1e8f;

            if (new_cov.allFinite() && new_cov.determinant() > params.covariance_det_min
                && cond < params.condition_number_max)
            {
                current_covariance = new_cov;
                return {new_cov, cond};
            }
            return {current_covariance, cond};
        } catch (const std::exception &e) {
            std::cerr << "RFE covariance update failed: " << e.what() << std::endl;
            return {current_covariance, -1.0f};
        }
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
        prior.covariance = torch::zeros({3, 3},
                              torch::TensorOptions().dtype(torch::kFloat32).device(get_device()));
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                prior.covariance[r][c] = cov_eigen(r, c);

        last_lidar_timestamp = lidar_timestamp;
        return prior;
    }

    /// Calibration bookkeeping, run every cycle: apply a pending reset, restore the window once,
    /// drain any closed pivots into the estimator, and keep the window persisted.
    ///
    /// ★ UNGATED, and that matters. All of this used to live inside adapt_motion_model(), which was
    /// gated by LearnMotionModel — a flag switched off in 2026-08 to disable a DIFFERENT learner (the
    /// EMA slip-k path, A/B'd and rejected, now deleted). One switch was therefore turning off two
    /// unrelated things, and the batch estimator's whole bookkeeping went with it: closures reached
    /// nothing, the state file was never written, and nothing said so.
    void RoomConcept::service_calibration()
    {
        // ── Reset, if the calibration window asked for one ────────────────────────────────────────
        if (calib_reset_pending_.exchange(false))
        {
            motion_calib_.reset_state(params.calib_state_file);
            qInfo() << "[calib] RESET: window emptied and" 
                    << QString::fromStdString(params.calib_state_file)
                    << "deleted. Every parameter is back at its prior and reports NOT informed, which"
                       " is the honest state of a robot that has just been told to un-learn.";
        }

        // ── Restore the window once, then keep it saved ───────────────────────────────────────────
        // Evidence, not parameters: the solve after a restart is the solve it would have been had the
        // run never stopped, with the prior still at zero and nothing ratcheting.
        if (not calib_state_loaded_)
        {
            calib_state_loaded_ = true;
            if (const std::size_t n = motion_calib_.load_state(params.calib_state_file); n > 0)
                qInfo().nospace() << "[calib] restored " << n << " measurements from "
                                  << QString::fromStdString(params.calib_state_file)
                                  << " (" << motion_calib_.closures() << " closed pivot(s)) — the "
                                     "window resumes, the priors do not move";
        }

        // Drain any closed pivots first, on THIS thread, before the episode path runs. A closure is
        // the strongest measurement of the rotation model this robot can make — no map, no survey,
        // no localiser in the number — and until 2026-08-26 it reached nothing: the pivot measured
        // the scale to +/-0.18%, logged it, and the estimator that PRICES the manoeuvre never heard.
        // So the offer's value never fell and the robot pivoted for ever, unable to extinguish its
        // own worth. Feeding it back is what closes that loop.
        {
            std::vector<ClosureObs> pending;
            {
                std::scoped_lock lk(pending_closures_mutex_);
                pending.swap(pending_closures_);
            }
            for (const auto& c : pending)
            {
                motion_calib_.observe_closure(c.truth_rad, c.turned_rad, c.rate_rad_s, c.sigma_s);
                // Say whether the closure actually TAUGHT k_omega, not just that it arrived. A
                // parameter's value cannot distinguish "measured" from "left where the prior put
                // it", and the accessors now apply it only once it is taught — so this is the line
                // that says whether the pivot changed the robot's behaviour or only its logs.
                qInfo().nospace() << "[calib] CLOSURE -> batch estimator | truth "
                                  << QString::number(c.truth_rad * 180.0 / M_PI, 'f', 1) << " deg, odometry "
                                  << QString::number(c.turned_rad * 180.0 / M_PI, 'f', 1) << " deg, rate "
                                  << QString::number(c.rate_rad_s, 'f', 3) << " rad/s, sigma "
                                  << QString::number(c.sigma_s * 100.0, 'f', 3) << "% | k_omega now "
                                  << QString::number((motion_calib_.omega_scale() - 1.f) * 100.f, 'f', 4)
                                  << "% +/- " << QString::number(motion_calib_.k_w_sigma() * 100.f, 'f', 4)
                                  << "% over " << motion_calib_.closures() << " closure(s), "
                                  << (motion_calib_.taught(rc::calib::P_K_OMEGA)
                                          ? "TAUGHT — the scale is now applied"
                                          : "still UNTAUGHT — nominal is applied, the value is only reported");
            }
            // A closure is minutes of robot time; persist immediately rather than waiting for the
            // periodic save, so a crash between here and then cannot throw it away.
            if (not pending.empty())
                motion_calib_.save_state(params.calib_state_file);
        }

        // Periodic save of the window. Every 30 s: often enough that a crash costs little, rare
        // enough to be invisible next to a solve.
        {
            const std::int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            if (calib_state_last_save_ms_ == 0) calib_state_last_save_ms_ = now_ms;
            else if (now_ms - calib_state_last_save_ms_ >= 30000)
            {
                motion_calib_.save_state(params.calib_state_file);
                calib_state_last_save_ms_ = now_ms;
            }
        }

    }

    void RoomConcept::write_debug_tail()
    {
        if (not debug_log_.is_open())
            return;
        debug_log_ << ',' << last_preint_samples_
                   << ',' << last_preint_duration_s_
                   << ',' << last_belief_age_s_
                   << ',' << last_belief_decay_
                   << ',' << last_aff_outcome_
                   << ',' << last_aff_completions_
                   << ',' << last_tgt_x_
                   << ',' << last_tgt_y_
                   << ',' << last_pub_tx_
                   << ',' << last_pub_ty_
                   << ',' << last_pub_ok_
                   << ',' << last_slot_motion_cov_(0, 1)
                   << ',' << last_slot_motion_cov_(0, 2)
                   << ',' << last_slot_motion_cov_(1, 2)
                   << ',' << last_imu_cover_
                   << ',' << last_pub_cov_xx_
                   << ',' << last_pub_cov_tt_
                   << ',' << (last_clamp_hit_ ? 1 : 0)
                   << ',' << last_wheel_gyro_ratio_;
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

    RoomConcept::OptTiming RoomConcept::take_optimizer_timing()
    {
        const auto n  = opt_total_count_.exchange(0, std::memory_order_relaxed);
        const auto ee = opt_earlyexit_count_.exchange(0, std::memory_order_relaxed);
        const auto us = opt_update_us_sum_.exchange(0, std::memory_order_relaxed);
        OptTiming t;
        t.count         = n;
        t.early_exits   = ee;
        t.avg_update_ms = (n > 0) ? (static_cast<double>(us) / 1000.0 / n) : 0.0;
        return t;
    }

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
        if (not motion_calib_.configured()) motion_calib_.configure(params.motion_calib);
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
            motion_calib_.observe(res.dy_local, res.dx_local, res.imu_dtheta + res.wheel_dtheta,
                                  r_forward, r_lateral, r_theta,
                                  pos_var, th_var, corrected_this_cycle, res.sdf_mse,
                                  last_cycle_dt_s_);
        }
        res.calib_value = motion_calib_.last_solve().value;
        res.calib_sigma = motion_calib_.last_solve().sigma;
        res.calib_b_omega = motion_calib_.omega_bias();
        {
            const auto &r = motion_calib_.last_solve();
            res.calib_informed = (r.informed[rc::calib::P_K_V]     ? 1 : 0)
                               | (r.informed[rc::calib::P_EPS_YAW] ? 2 : 0)
                               | (r.informed[rc::calib::P_K_OMEGA] ? 4 : 0)
                               | (r.informed[rc::calib::P_B_OMEGA] ? 8 : 0);
            res.calib_condition = r.condition;
        }
        res.calib_sigma_b_omega = motion_calib_.last_solve().sigma[rc::calib::P_B_OMEGA];
        res.calib_sigma_yaw = motion_calib_.yaw_sigma();
        res.calib_sigma_k_v = motion_calib_.k_v_sigma();
        res.calib_sigma_k_w = motion_calib_.k_w_sigma();
        res.calib_k_v = motion_calib_.forward_scale();
        res.calib_k_w = motion_calib_.omega_scale();
        res.calib_yaw = motion_calib_.yaw_offset();
        res.calib_episodes = motion_calib_.episodes();
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
        int dim = is_localized ? 3 : 5;  // 3 for localized [x,y,theta], 5 for full state [w,h,x,y,theta]
        PredictionState prediction;
        const auto device = get_device();

        // Ensure prev_cov is properly initialized
        if (!room->prev_cov.defined() || room->prev_cov.numel() == 0)
        {
            room->prev_cov = 0.1f * torch::eye(dim, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        }

        // Get current pose for Jacobian computation
        if (not room->robot_prev_pose.has_value())
        {
            // Fallback: simple additive noise
            prediction.propagated_cov = room->prev_cov + params.cmd_noise_trans * params.cmd_noise_trans *
                torch::eye(dim, torch::TensorOptions().dtype(torch::kFloat32).device(device));
            return prediction;
        }

        auto robot_prev_pose = room->robot_prev_pose.value();
        const float theta = std::atan2(robot_prev_pose.linear()(1, 0), robot_prev_pose.linear()(0, 0));

        // Transform global delta back to robot frame for noise computation
        float cos_t = std::cos(theta);
        float sin_t = std::sin(theta);

        float dx_global = odometry_prior.delta_pose[0];
        float dy_global = odometry_prior.delta_pose[1];

        // Inverse rotation: robot_frame = R^T * global_frame
        float dx_local = dx_global * cos_t + dy_global * sin_t;
        float dy_local = -dx_global * sin_t + dy_global * cos_t;

        // ===== MOTION MODEL JACOBIAN =====
        // State: [x, y, theta] (for localized) or [w, h, x, y, theta] (for mapping)

        // Reuse pre-allocated F (identity + off-diag) and Q (zeroed + filled)
        if (predict_alloc_dim_ != dim)
        {
            predict_F_ = torch::eye(dim, torch::TensorOptions().dtype(torch::kFloat32).device(device));
            predict_Q_ = torch::zeros({dim, dim}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
            predict_alloc_dim_ = dim;
        }
        // Reset F off-diagonal and Q to zero for this frame
        auto F = predict_F_;
        auto Q = predict_Q_;
        // Reset the variable elements
        if (is_localized) {
            // ∂x/∂θ = -dx_local·sin(θ) - dy_local·cos(θ)
            // ∂y/∂θ =  dx_local·cos(θ) - dy_local·sin(θ)
            F[0][2] = -dx_local * sin_t - dy_local * cos_t;
            F[1][2] =  dx_local * cos_t - dy_local * sin_t;
        } else {
            F[2][4] = -dx_local * sin_t - dy_local * cos_t;
            F[3][4] =  dx_local * cos_t - dy_local * sin_t;
        }

        Q.zero_();
        const auto prior_cov = odometry_prior.covariance.defined()
            ? odometry_prior.covariance.to(device)
            : torch::eye(3, torch::TensorOptions().dtype(torch::kFloat32).device(device));

        if (is_localized)
            Q.index_put_({torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)}, prior_cov);
        else
            Q.index_put_({torch::indexing::Slice(2, 5), torch::indexing::Slice(2, 5)}, prior_cov);

        // ===== EKF PREDICTION =====
        // P_pred = F * P_prev * F^T + Q
        torch::Tensor propagated = torch::matmul(torch::matmul(F, room->prev_cov), F.t()) + Q;

        prediction.propagated_cov = propagated;
        prediction.have_propagated = true;
        return prediction;
    }

    Eigen::Vector3f RoomConcept::integrate_odometry_over_window(
                const Eigen::Affine2f& robot_pose,
                const std::vector<OdometryReading> &odometry_history,
                const int64_t &t_start_ms,
                const int64_t &t_end_ms,
                rc::preint::Interval *preint_out,
                const std::vector<ImuReading> *imu_history,
                const SimClockMap *sim_clock)
    {
        Eigen::Vector3f total_delta = Eigen::Vector3f::Zero();
        float running_theta = std::atan2(robot_pose.linear()(1,0), robot_pose.linear()(0,0));

        // Preintegration runs on the SAME segments, seeded at the SAME heading. It is a pure observer:
        // it never writes total_delta, so the mean cannot change depending on whether it is asked for.
        rc::preint::Integrator preint(running_theta);
        preint.set_noise(params.odom_preint_noise);

        // ── ONE CLOCK, and it is the producer's ────────────────────────────────────────────────────
        // The two ends of this interval come from different places. The BOUNDS are lidar sweep stamps,
        // which are local wall-clock. The RATES are in whatever clock the producer measured them in --
        // and a simulator measures per SIMULATION second while running behind real time, so
        // integrating its rates over wall intervals under-counts by exactly the sim/wall ratio (~6%
        // here; the mirror image of the bug webots-bridge commit 5968a65 fixed from the other side).
        //
        // Converting the two BOUNDS is the cheap direction: everything below then works in one clock,
        // and because this function returns a DISPLACEMENT rather than a rate, no caller has to know.
        // On real hardware the map is identity and none of this costs anything.
        const bool use_sim_clock = sim_clock != nullptr and sim_clock->valid()
                                   and not odometry_history.empty()
                                   and odometry_history.front().simulated;
        const auto to_clock = [&](std::int64_t wall_ms) -> std::int64_t
        { return use_sim_clock ? sim_clock->to_sim(wall_ms) : wall_ms; };
        const auto stamp_of = [&](const OdometryReading &o) -> std::int64_t
        { return use_sim_clock ? o.integration_ts_ms() : o.effective_ts_ms(); };

        const std::int64_t win_start_ms = to_clock(t_start_ms);
        const std::int64_t win_end_ms   = to_clock(t_end_ms);
        // The window's own duration, kept for the calibrator: elapsed time is the only covariate that
        // separates a gyro bias from a gyro scale.
        last_cycle_dt_s_ = static_cast<float>(win_end_ms - win_start_ms) * 1e-3f;

        // ── Heading change from the gyro, over an arbitrary sub-interval ───────────────────────────
        // Yaw is the channel wheel odometry gets worst: a differential base turns by scrubbing its
        // wheels sideways, so it over-reports rotation -- measured 8.2% on this robot, against
        // translation exact to 0.1%. The gyro measures the body rate directly and carries no such
        // error. It also runs ~10x faster, so this integrates the rate as it actually varied instead
        // of holding one 10 Hz `rot` sample flat across a whole sweep interval.
        //
        // Returns false unless the IMU brackets the WHOLE segment; partial coverage is deliberately
        // refused, because a Δθ covering part of a segment silently under-rotates, which is worse
        // than a consistent wheel estimate.
        const auto imu_stamp = [&](const ImuReading &s) -> std::int64_t
        { return use_sim_clock ? s.integration_ts_ms() : s.effective_ts_ms(); };
        // ── Linear channel from the accelerometer ──────────────────────────────────────────────────
        // Returns the velocity CHANGE over [seg_start, seg_end] in the body frame, and the
        // displacement that change contributes within the same interval (the double integration).
        //
        // Both are INCREMENTS confined to one interval and are never chained. That distinction is the
        // whole design: over ~50 ms the accelerometer estimates dv ~15x more precisely than the
        // wheels (0.1-0.4 mm/s of noise against their 67 mm/s per sample), but an accelerometer
        // cannot observe velocity itself, only its change, and chaining the same 0.5 deg tilt error
        // reaches 0.086 m/s after 1 s and 5.1 m/s after 60 s. So this supplies dv; the wheels supply
        // the absolute v that dv is a change TO.
        //
        // dp here is the correction to the constant-velocity assumption inside the interval
        // (0.5*a*T^2), not a position estimate. At 50 ms it is sub-millimetre and its only job is to
        // stop a hard acceleration being modelled as if the velocity had been constant throughout.
        //
        // GRAVITY IS NOT REMOVED. The samples carry it, and the horizontal components are honest
        // horizontal acceleration only insofar as the mount is level; the residual is a slowly
        // varying bias, which is why this is logged and cross-checked before it is ever fused.
        const auto imu_dvel = [&](std::int64_t seg_start, std::int64_t seg_end,
                                  float &dvx_out, float &dvy_out,
                                  float &dpx_out, float &dpy_out) -> bool
        {
            if (imu_history == nullptr or imu_history->size() < 2) return false;
            if (imu_stamp(imu_history->front()) > seg_start or imu_stamp(imu_history->back()) < seg_end)
                return false;                       // partial coverage refused, as for imu_dtheta
            double vx = 0, vy = 0, px = 0, py = 0;
            bool any = false;
            for (size_t k = 0; k + 1 < imu_history->size(); ++k)
            {
                const auto &a0 = (*imu_history)[k];
                const auto &a1 = (*imu_history)[k + 1];
                const std::int64_t t0 = std::max(imu_stamp(a0), seg_start);
                const std::int64_t t1 = std::min(imu_stamp(a1), seg_end);
                if (t1 <= t0) continue;
                const double h = static_cast<double>(t1 - t0) * 1e-3;
                // Trapezoid on acceleration: the sample rate is ~5x the interval, so the ramp between
                // samples is real information and a zero-order hold would systematically lag it.
                const double ax = 0.5 * (a0.acc_x + a1.acc_x);
                const double ay = 0.5 * (a0.acc_y + a1.acc_y);
                px += vx * h + 0.5 * ax * h * h;    // v already accumulated within THIS interval only
                py += vy * h + 0.5 * ay * h * h;
                vx += ax * h;
                vy += ay * h;
                any = true;
            }
            if (not any) return false;
            dvx_out = static_cast<float>(vx); dvy_out = static_cast<float>(vy);
            dpx_out = static_cast<float>(px); dpy_out = static_cast<float>(py);
            return true;
        };

        // Mean per-sample variance a channel reported over [seg_start, seg_end], converted to the
        // NOISE DENSITY the preintegrator wants: sigma = sqrt(var_sample * dt_sample). Returns <0
        // when the producer said "unknown" (negative variance) or the segment is not bracketed, and
        // the caller then falls back to the asserted constant. See the conversion note in
        // se2_preintegration.h: handing it the raw per-sample variance over-states the noise by
        // dt_sample/dt, a factor of five at 100 Hz across a 50 ms segment.
        const auto imu_sigma = [&](std::int64_t seg_start, std::int64_t seg_end,
                                   bool want_gyro) -> float
        {
            if (imu_history == nullptr or imu_history->size() < 2) return -1.f;
            double var_sum = 0.0; int n = 0;
            std::int64_t first = 0, last = 0;
            for (const auto &sm : *imu_history)
            {
                const std::int64_t st = imu_stamp(sm);
                if (st < seg_start or st > seg_end) continue;
                const float v = want_gyro ? sm.gyro_var : sm.acc_var;
                if (not (v >= 0.f)) return -1.f;      // producer does not know -> use the model
                var_sum += v; ++n;
                if (first == 0) first = st;
                last = st;
            }
            if (n < 2 or last <= first) return -1.f;
            const double dt_sample = static_cast<double>(last - first) * 1e-3 / (n - 1);
            return static_cast<float>(std::sqrt(var_sum / n * dt_sample));
        };

        // ── The WHEELS' own stated noise, as a density ─────────────────────────────────────────────
        // Same conversion as imu_sigma, and it exists for the same reason: the producer states a
        // PER-SAMPLE variance at ITS OWN rate (FullPoseEuler::velCov, forwarded onto the robot node by
        // robot_concept), while the preintegrator integrates densities. sigma = sqrt(var * dt_sample).
        //
        // dt_sample is MEASURED from the stamps, never configured. The bridge's [FullPose]
        // PublishPeriod went 100 ms -> 20 ms on 2026-08-25, and a hard-coded period would have made
        // the density 5x too large the moment it changed -- silently, since a variance that is merely
        // wrong still looks like a number. Measuring it means the rate can move again and this line
        // stays correct; it is also the only way a FASTER stream can tighten anything, because a
        // density is invariant to sample rate and only a smaller per-sample variance or a shorter
        // dt_sample moves it.
        //
        // Estimated across the whole history rather than per pair, so one late sample cannot inflate
        // the period for the segment it lands in.
        double odom_dt_sample = -1.0;
        if (odometry_history.size() >= 2)
        {
            const std::int64_t first = stamp_of(odometry_history.front());
            const std::int64_t last  = stamp_of(odometry_history.back());
            if (last > first)
                odom_dt_sample = static_cast<double>(last - first) * 1e-3
                               / static_cast<double>(odometry_history.size() - 1);
        }
        // Returns <0 for "use the model": the channel is off, the producer said "unknown" (negative
        // variance), or the period could not be measured. A zero variance is REFUSED for the same
        // reason: it would claim infinite confidence in a channel, and no wheel is that good.
        const auto wheel_sigma = [&](float var_sample) -> float
        {
            if (not params.odom_variance_injection) return -1.f;
            if (not (var_sample > 0.f) or odom_dt_sample <= 0.0) return -1.f;
            return static_cast<float>(std::sqrt(static_cast<double>(var_sample) * odom_dt_sample));
        };
        // Two independent noise contributions to the same channel add in quadrature. Either being
        // "unstated" (<0) leaves the other alone, and both unstated falls back to the model.
        const auto quad = [](float a, float b) -> float
        {
            if (a < 0.f) return b;
            if (b < 0.f) return a;
            return std::sqrt(a * a + b * b);
        };

        const auto imu_dtheta = [&](std::int64_t seg_start, std::int64_t seg_end, float &dtheta_out) -> bool
        {
            if (imu_history == nullptr or imu_history->size() < 2) return false;
            if (imu_stamp(imu_history->front()) > seg_start or imu_stamp(imu_history->back()) < seg_end)
                return false;
            double acc = 0.0;
            bool any = false;
            for (size_t k = 0; k + 1 < imu_history->size(); ++k)
            {
                const std::int64_t a = imu_stamp((*imu_history)[k]);
                const std::int64_t b = imu_stamp((*imu_history)[k + 1]);
                if (b <= a) continue;
                const std::int64_t s = std::max(a, seg_start);
                const std::int64_t e = std::min(b, seg_end);
                if (e <= s) continue;
                // Trapezoid across the sample pair, so a rate ramping through the segment is not
                // biased by holding the leading sample flat.
                const double w0 = (*imu_history)[k].gyro_z, w1 = (*imu_history)[k + 1].gyro_z;
                const double span = static_cast<double>(b - a);
                const double f0 = static_cast<double>(s - a) / span;
                const double f1 = static_cast<double>(e - a) / span;
                const double wm = 0.5 * ((w0 + (w1 - w0) * f0) + (w0 + (w1 - w0) * f1));
                acc += wm * static_cast<double>(e - s) * 0.001;
                any = true;
            }
            if (not any) return false;
            dtheta_out = static_cast<float>(acc);
            return true;
        };
        int imu_segments = 0, wheel_segments = 0;
        int odom_var_segments = 0;   // segments whose noise the WHEELS themselves stated
        cyc_imu_dtheta_ = cyc_wheel_dtheta_ = cyc_wheel_shadow_dtheta_ = 0.f;
        cyc_dx_local_ = cyc_dy_local_ = 0.f;
        cyc_imu_dvx_ = cyc_imu_dvy_ = cyc_imu_dpx_ = cyc_imu_dpy_ = 0.f;
        cyc_wheel_dvx_ = cyc_wheel_dvy_ = 0.f; cyc_imu_lin_segs_ = 0;
        cyc_imu_segs_ = cyc_wheel_segs_ = 0;

        // Integrate over all odometry readings in [win_start_ms, win_end_ms], on the clock chosen above.
        for (size_t i = 0; i < odometry_history.size(); ++i)
        {
            const auto& odom = odometry_history[i];

            // Get time window for this reading
            const std::int64_t cmd_start_ms = stamp_of(odom);
            const std::int64_t cmd_end_ms = (i + 1 < odometry_history.size())
                           ? stamp_of(odometry_history[i + 1])
                           : win_end_ms;

            if (cmd_start_ms <= 0 || cmd_end_ms <= 0 || cmd_end_ms <= cmd_start_ms)
                continue;

            // Clip to [win_start_ms, win_end_ms]
            if (cmd_end_ms < win_start_ms) continue;
            if (cmd_start_ms > win_end_ms) break;

            const std::int64_t effective_start_ms = std::max(cmd_start_ms, win_start_ms);
            const std::int64_t effective_end_ms = std::min(cmd_end_ms, win_end_ms);

            const float dt = static_cast<float>(effective_end_ms - effective_start_ms) * 0.001f;
            if (dt <= 0) continue;

            // Odometry velocities are in robot frame: adv=forward(Y), side=lateral(X), rot=angular
            // Learned scale from motion_calib_ (1.0 until it has seen anything, and exactly 1.0
            // while the feature is off, so this line is a no-op in the default build).
            // Forward and lateral carry SEPARATE scales. They used to share one, which was harmless
            // only because lateral travel is usually ~0; on a mecanum the lateral channel is the one
            // roller slip corrupts and it is physically a different error.
            const float k_v   = motion_calib_.forward_scale();
            const float k_lat = motion_calib_.lateral_scale();
            const float dx_local = odom.side * dt * k_lat;  // lateral (X in robot frame)
            const float dy_local = odom.adv  * dt * k_v;    // forward (Y in robot frame)
            cyc_dx_local_ += dx_local;
            cyc_dy_local_ += dy_local;
            // The wheels' own velocity CHANGE across this segment, so the two channels are compared
            // as like for like: an accelerometer gives dv, never v. A disagreement here is the first
            // thing translation has ever had that can see wheel slip -- during slip the wheels report
            // a dv that did not happen while the accelerometer sees the truth. Heading has had this
            // cross-check since the gyro went in; translation has had NO second opinion at all.
            if (i + 1 < odometry_history.size())
            {
                const auto &nx = odometry_history[i + 1];
                cyc_wheel_dvx_ += nx.side - odom.side;
                cyc_wheel_dvy_ += nx.adv  - odom.adv;
            }
            float imu_dpx_seg = 0.f, imu_dpy_seg = 0.f;
            if (float dvx = 0.f, dvy = 0.f, dpx = 0.f, dpy = 0.f;
                imu_dvel(effective_start_ms, effective_end_ms, dvx, dvy, dpx, dpy))
            {
                cyc_imu_dvx_ += dvx; cyc_imu_dvy_ += dvy;
                cyc_imu_dpx_ += dpx; cyc_imu_dpy_ += dpy;
                ++cyc_imu_lin_segs_;
                imu_dpx_seg = dpx; imu_dpy_seg = dpy;
            }
            // THE INJECTION, linear half. Body-frame axes line up one-for-one now that the producer
            // rotates the device frame: dx_local is lateral (+X), dy_local forward (+Y), and so are
            // dpx/dpy. Refused unless the IMU brackets the WHOLE segment, exactly as for dtheta -- a
            // partial integral silently under-reports rather than degrading gracefully.
            const float dx_use = dx_local + (params.imu_linear_injection ? imu_dpx_seg : 0.f);
            const float dy_use = dy_local + (params.imu_linear_injection ? imu_dpy_seg : 0.f);

            // THE INJECTION. Heading change from the gyro when it brackets this segment, otherwise
            // the wheel-derived rate. Translation stays on the wheels either way -- an accelerometer
            // cannot supply it without a drifting double integration, and the wheels are already
            // exact there.
            const float k_w = motion_calib_.omega_scale();
            // A BIAS is subtracted per unit TIME, a scale multiplies the RATE. That difference is the
            // only thing separating the two, and it is why the joint solve can find both at once.
            const float b_w = motion_calib_.omega_bias();
            // Per-wheel mismatch: unequal effective radii make a commanded straight line curve, so
            // it adds heading in proportion to DISTANCE driven, not to rotation or to time.
            const float dk_wheel = motion_calib_.wheel_mismatch();
            const float curve = dk_wheel * dy_local;
            float dtheta = odom.rot * dt * k_w - b_w * dt + curve;
            float rot_eff = dt > 0.f ? dtheta / dt : (odom.rot * k_w - b_w);
            // Which sensor's noise describes rot_eff below. The channel that SUPPLIED the mean is the
            // one whose stated variance applies to it; crediting the gyro's noise to a wheel-derived
            // heading would describe a measurement that was never made.
            bool heading_from_imu = false;
            if (float dth_imu = 0.f; imu_dtheta(effective_start_ms, effective_end_ms, dth_imu))
            {
                // Keep BOTH on the covered segments: their ratio is how much heading the gyro is
                // taking out of the wheel estimate, which is the whole point of the injection and the
                // one number that says it is doing something rather than merely running.
                wheel_dtheta_sum_ += dtheta;
                imu_dtheta_sum_   += dth_imu;
                cyc_wheel_shadow_dtheta_ += dtheta;   // what the wheels said, before the override
                cyc_imu_dtheta_          += dth_imu;  // what actually entered the prior
                ++cyc_imu_segs_;
                // The learned scale applies to whichever channel supplies the heading, and the gyro
                // supplies ~99% of it -- applying it only to the wheel branch would leave it inert.
                dtheta = dth_imu * k_w - b_w * dt + curve;
                rot_eff = dtheta / dt;               // the mean rate the gyro actually saw
                heading_from_imu = true;
                ++imu_segments;
            }
            else
            {
                cyc_wheel_dtheta_ += dtheta;          // wheel value that entered the prior unmodified
                ++cyc_wheel_segs_;
                ++wheel_segments;
            }

            // Transform to global frame using MIDPOINT theta (reduces integration bias)
            // The learned yaw offset rotates the body->world mapping: it absorbs a mount or
            // body-axis misalignment, which shows up as travel drifting off the fitted heading.
            const float theta_mid = running_theta + 0.5f * dtheta + motion_calib_.yaw_offset();
            total_delta[0] += dx_use * std::cos(theta_mid) - dy_use * std::sin(theta_mid);
            total_delta[1] += dx_use * std::sin(theta_mid) + dy_use * std::cos(theta_mid);
            total_delta[2] += dtheta;

            // Preintegration sees the SAME rate the mean used, or its covariance would describe a
            // trajectory that was not integrated.
            if (preint_out != nullptr)
            {
                // Densities from what the producers actually stated this segment, falling back to
                // the asserted constants per channel. The gyro's covers the heading it supplied; the
                // accelerometer's covers the translation only when its correction is actually being
                // used, because otherwise the displacement came from the wheels and it is the wheels'
                // noise that describes it.
                // Yaw: the gyro's density when the gyro supplied the heading, the wheels' own when
                // they did. imu_sigma already returns <0 unless the IMU brackets the segment, so the
                // two never both apply.
                const float sig_om = heading_from_imu
                                   ? imu_sigma(effective_start_ms, effective_end_ms, true)
                                   : wheel_sigma(odom.var_rot);
                // Translation always comes from the wheels. When the accelerometer's within-segment
                // correction is enabled it is ADDED to that displacement, so its noise is an extra
                // independent contribution to the same channel, not a replacement for the wheels'.
                const float sig_acc = params.imu_linear_injection
                                    ? imu_sigma(effective_start_ms, effective_end_ms, false) : -1.f;
                const float sig_lat  = quad(wheel_sigma(odom.var_side), sig_acc);
                const float sig_long = quad(wheel_sigma(odom.var_adv),  sig_acc);
                // The accelerometer's within-segment correction enters as an EFFECTIVE mean velocity,
                // so the preintegrator's constant-velocity step reproduces the corrected displacement
                // without changing its structure -- and the correction is then covered by Q and by
                // the transport term A, which it would not be if it were added to the mean afterwards.
                const float v_lat_eff  = odom.side + (params.imu_linear_injection ? imu_dpx_seg / dt : 0.f);
                const float v_long_eff = odom.adv  + (params.imu_linear_injection ? imu_dpy_seg / dt : 0.f);
                preint.add(v_lat_eff, v_long_eff, rot_eff, dt, sig_lat, sig_long, sig_om);
                if (sig_lat >= 0.f or sig_long >= 0.f or (not heading_from_imu and sig_om >= 0.f))
                    ++odom_var_segments;
            }

            running_theta += dtheta;
        }

        // ── Proof of life, then periodic health ────────────────────────────────────────────────────
        // A channel that silently fails to bind is a recurring failure mode here, and an unused or
        // degrading IMU is invisible in every outcome metric until it has already cost accuracy. The
        // one-shot says it bound; the periodic line says it is STILL bound, how much of each interval
        // the IMU actually covers, and how much heading it is removing from the wheel estimate.
        imu_seg_used_  += imu_segments;
        imu_seg_total_ += imu_segments + wheel_segments;
        imu_stats_sim_clock_ = use_sim_clock;

        // Proof of life for the wheel-variance channel. It must be printed from the segment counter
        // and not from "the flag is on", because every way this can fail -- producer silent, producer
        // says "unknown", period unmeasurable, attribute never registered -- ends in the same silent
        // fallback to the constants. A flag that is on and a channel that is working are different
        // claims, and only the second one is worth logging.
        if (not odom_var_announced_ and odom_var_segments > 0)
        {
            odom_var_announced_ = true;
            const auto& o = odometry_history.back();
            const float s_lat  = wheel_sigma(o.var_side);
            const float s_long = wheel_sigma(o.var_adv);
            const float s_om   = wheel_sigma(o.var_rot);
            qInfo().nospace() << "[OdomVar] wheels' stated variance ACTIVE | dt_sample="
                              << QString::number(odom_dt_sample * 1e3, 'f', 1) << " ms"
                              << " | var(adv,side,rot)=" << o.var_adv << "," << o.var_side << "," << o.var_rot
                              << " -> sigma(lat,long,om)="
                              << QString::number(s_lat,  'g', 3) << ","
                              << QString::number(s_long, 'g', 3) << ","
                              << QString::number(s_om,   'g', 3)
                              << " vs model " << params.odom_preint_noise.sigma_v_lat << ","
                              << params.odom_preint_noise.sigma_v_long << ","
                              << params.odom_preint_noise.sigma_omega
                              << " | " << odom_var_segments << "/" << (imu_segments + wheel_segments)
                              << " segments";
        }

        if (not imu_injection_announced_ and imu_segments > 0)
        {
            imu_injection_announced_ = true;
            qInfo().nospace() << "[ImuInject] gyro heading ACTIVE | clock="
                              << (use_sim_clock ? "SIM (producer)" : "WALL — sim map NOT bound")
                              << " | " << imu_segments << "/" << (imu_segments + wheel_segments)
                              << " segments from IMU";
        }

        // Per-WINDOW coverage, mirrored for the debug log. The 5 s stats above reset themselves, so
        // they cannot be sampled per row — and without a coverage column the rotation channel is
        // uninterpretable: a scale that fails to recover looks identical whether the estimator is
        // wrong or the injected channel simply was not the one in use. That ambiguity already cost
        // one full recovery run.
        last_imu_cover_ = (imu_segments + wheel_segments) > 0
                        ? static_cast<float>(imu_segments) / static_cast<float>(imu_segments + wheel_segments)
                        : -1.f;

        if (imu_stats_last_log_ms_ == 0)
            imu_stats_last_log_ms_ = t_end_ms;
        else if (t_end_ms - imu_stats_last_log_ms_ >= 5000 and imu_seg_total_ > 0)
        {
            const double cover = 100.0 * imu_seg_used_ / imu_seg_total_;
            // Ratio of what the wheels would have said to what the gyro said, over the same segments.
            // Expect ~1.05-1.08 on this robot: a differential base over-reports rotation because it
            // turns by scrubbing. A ratio pinned at 1.000 means the gyro is agreeing suspiciously
            // exactly -- more likely the same source twice than two sensors agreeing.
            // ★ THE GUARD MUST BE A MEANINGFUL ROTATION, NOT A NON-ZERO ONE. At 1e-4 rad this printed
            // a ratio for a robot standing still: observed live, "wheel/gyro=-9.6824 over -0.001 rad"
            // — one milliradian of true rotation, so the quotient was 0/0 and the number was noise
            // wearing four decimal places. It is worse now that the wheel channel carries injected
            // noise: over a 5 s window a parked robot accumulates sigma_w*sqrt(T) = 0.010*sqrt(5) =
            // 0.022 rad of random walk against ~0 of signal, so a parked window CANNOT produce a
            // meaningful ratio however long it runs. 0.5 rad is roughly 30 degrees — enough that the
            // scrubbing error being measured is well clear of the noise floor.
            constexpr double kMinRotForRatio = 0.5;   // rad
            const bool ratio_valid = std::abs(imu_dtheta_sum_) > kMinRotForRatio;
            const double ratio = ratio_valid ? wheel_dtheta_sum_ / imu_dtheta_sum_
                                             : std::numeric_limits<double>::quiet_NaN();
            last_wheel_gyro_ratio_ = static_cast<float>(ratio);   // NaN while the guard is unmet
            qInfo().nospace() << "[ImuInject] clock="
                              << (imu_stats_sim_clock_ ? "SIM" : "WALL(unbound)")
                              << " coverage=" << QString::number(cover, 'f', 1) << "%"
                              << " (" << imu_seg_used_ << "/" << imu_seg_total_ << " seg)"
                              << " dtheta wheel/gyro="
                              << (ratio_valid ? QString::number(ratio, 'f', 4)
                                              : QString("n/a (needs >%1 rad of turning)")
                                                    .arg(kMinRotForRatio, 0, 'f', 1))
                              << " over " << QString::number(imu_dtheta_sum_, 'f', 3) << " rad";
            imu_stats_last_log_ms_ = t_end_ms;
            imu_seg_used_ = imu_seg_total_ = 0;
            imu_dtheta_sum_ = wheel_dtheta_sum_ = 0.0;
        }

        if (preint_out != nullptr)
            *preint_out = preint.result();

        return total_delta;
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
        // Full 3×3, not just the diagonal: this tensor feeds the meas_cov_* diagnostic columns, and a
        // silently truncated copy would make the logged covariance disagree with the one in use.
        prior.covariance = torch::zeros({3, 3},
                              torch::TensorOptions().dtype(torch::kFloat32).device(get_device()));
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                prior.covariance[r][c] = cov_eigen(r, c);

        return prior;
    }

    std::pair<Eigen::Vector3f, Eigen::Matrix3f> RoomConcept::fuse_priors(
        const Eigen::Vector3f &pred_cmd, const Eigen::Matrix3f &cov_cmd,
        const Eigen::Vector3f &pred_odom, const Eigen::Matrix3f &cov_odom) const
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

    // =====================================================================
    //  Differential test: shadow single-step Adam evaluator
    // =====================================================================
    float RoomConcept::shadow_single_step_adam(const torch::Tensor& points_tensor,
                                               float pred_x, float pred_y, float pred_theta) const
    {
        const auto device = get_device();

        // Create an isolated pose tensor starting from the predicted pose
        auto pose = torch::tensor({pred_x, pred_y, pred_theta},
            torch::TensorOptions().dtype(torch::kFloat32).device(device)).requires_grad_(true);

        torch::optim::Adam optimizer(
            {torch::optim::OptimizerParamGroup({pose},
                std::make_unique<torch::optim::AdamOptions>(params.learning_rate_pos))});

        // Run the same number of iterations as the main loop
        for (int i = 0; i < params.num_iterations; ++i)
        {
            optimizer.zero_grad();
            auto xy = pose.index({torch::indexing::Slice(0, 2)});
            auto th = pose.index({torch::indexing::Slice(2, 3)});
            auto loss = compute_observation_loss(*model_, params, points_tensor, xy, th);
            loss.backward();
            optimizer.step();
        }

        // Evaluate SDF at optimised pose (same metric as main pipeline)
        torch::NoGradGuard no_grad;
        auto xy = pose.index({torch::indexing::Slice(0, 2)});
        auto th = pose.index({torch::indexing::Slice(2, 3)});
        auto sdf_vals = model_->sdf_at_pose(points_tensor, xy, th);
        return torch::median(torch::abs(sdf_vals)).item<float>();
    }

    // =====================================================================
    //  Exact 3×3 Hessian via double-backprop
    //  H_ij = ∂²L/∂p_i∂p_j computed by differentiating through the gradient
    // =====================================================================
    Eigen::Matrix3f RoomConcept::autograd_hessian_3x3(const torch::Tensor& loss,
                                                       const torch::Tensor& param)
    {
        // First-order gradient with graph retained for second pass
        auto grad = torch::autograd::grad({loss}, {param},
            /*grad_outputs=*/{}, /*retain_graph=*/true, /*create_graph=*/true)[0];

        Eigen::Matrix3f H;
        for (int i = 0; i < 3; i++)
        {
            // Differentiate grad[i] w.r.t. param → row i of the Hessian
            auto gi = grad.index({i});
            auto row = torch::autograd::grad({gi}, {param},
                /*grad_outputs=*/{}, /*retain_graph=*/(i < 2), /*create_graph=*/false)[0];
            auto row_cpu = row.to(torch::kCPU);
            auto acc = row_cpu.accessor<float, 1>();
            for (int j = 0; j < 3; j++)
                H(i, j) = acc[j];
        }
        // Symmetrise to absorb any floating-point asymmetry
        H = 0.5f * (H + H.transpose());
        return H;
    }

    // =====================================================================
    //  WindowManager methods
    // =====================================================================

    bool RoomConcept::WindowManager::append(WindowSlot slot, int max_window_size,
                                             float mu_quality_threshold, bool fej_schur)
    {
        bool slid = false;
        if (static_cast<int>(window.size()) >= max_window_size)
        {
            // FEJ+Schur path: marginalize_oldest() (called before this) already folded the dropping
            // slot into the boundary prior at a FROZEN linearization point. Touching mu here would be
            // exactly the non-FEJ re-anchoring we are removing — so just pop.
            if (not fej_schur)
            {
                auto pose_cpu = window.front().pose.detach().to(torch::kCPU);
                auto pose_acc = pose_cpu.accessor<float, 1>();

                // Solution C: only update boundary mu if the dropped slot had acceptable
                // localization quality. If the slot was confused (displacement, obstacle),
                // keep the previous mu so the prior continues anchoring to the last good pose.
                const bool slot_is_good = (window.front().sdf_mse_final < mu_quality_threshold)
                                          || !boundary_prior.valid;
                if (slot_is_good)
                    boundary_prior.mu = Eigen::Vector3f(pose_acc[0], pose_acc[1], pose_acc[2]);
            }

            window.pop_front();
            slid = true;
        }
        window.push_back(std::move(slot));
        return slid;
    }

    void RoomConcept::WindowManager::subsample_old_slots(int max_pts_per_slot)
    {
        if (window.size() <= 1 || max_pts_per_slot <= 0) return;
        for (size_t i = 0; i < window.size() - 1; i++)
        {
            auto& slot = window[i];
            if (slot.subsampled) continue;   // already done — skip
            const int64_t n_pts = slot.lidar_points.size(0);
            if (n_pts > max_pts_per_slot)
            {
                const int64_t stride = n_pts / max_pts_per_slot;
                auto indices = torch::arange(0, n_pts, stride,
                    torch::TensorOptions().dtype(torch::kLong).device(slot.lidar_points.device()));
                slot.lidar_points = slot.lidar_points.index_select(0, indices).contiguous();
            }
            slot.subsampled = true;
        }
    }

    std::vector<torch::Tensor> RoomConcept::WindowManager::collect_params() const
    {
        std::vector<torch::Tensor> p;
        p.reserve(window.size());
        for (const auto& slot : window)
            p.push_back(slot.pose);
        return p;
    }

    torch::Tensor RoomConcept::WindowManager::compute_rfe_loss(
        const Model& model, const Params& params, torch::Device device,
        float boundary_weight) const
    {
        if (window.empty())
            return torch::tensor(0.0f, torch::TensorOptions().device(device));

        torch::Tensor total_loss = torch::tensor(0.0f, torch::TensorOptions().device(device));

        // --- 1. Boundary prior on oldest surviving state (Eq. 28) ---
        // With W=1 the oldest slot IS the current slot: the prior would anchor the current
        // pose to the previous frame's post-ADAM estimate, creating an error integrator that
        // drives sawtooth drift even when the robot is static.  Only apply when W > 1.
        if (boundary_prior.valid && window.size() > 1)
        {
            const auto& oldest_pose = window.front().pose;
            const auto mu = torch::tensor(
                {boundary_prior.mu[0], boundary_prior.mu[1], boundary_prior.mu[2]},
                torch::TensorOptions().dtype(torch::kFloat32).device(device));

            auto prec_data = torch::zeros({3, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    prec_data[i][j] = boundary_prior.precision(i, j);

            // Δ = wrap(x_front − mu). For FEJ+Schur, mu is the FROZEN linearization point; for the
            // legacy path, mu is the dropped-slot pose. Same quadratic core either way.
            auto raw_diff = oldest_pose - mu;
            auto angle_diff = raw_diff.index({2});
            auto wrapped_diff = torch::cat({raw_diff.index({torch::indexing::Slice(0, 2)}),
                                             torch::atan2(torch::sin(angle_diff), torch::cos(angle_diff)).unsqueeze(0)});
            auto diff = wrapped_diff.unsqueeze(1);
            // Both paths use the same pure-quadratic form; FEJ+Schur differs only in how mu/precision
            // are computed (marginalize_oldest → frozen marginal mode + Schur precision).
            auto boundary_loss = boundary_weight *
                0.5f * torch::matmul(diff.t(), torch::matmul(prec_data, diff)).squeeze();
            total_loss = total_loss + boundary_loss;
        }

        // --- 2. Observation factors: SDF likelihood at each timestep ---
        // When sdf_current_slot_only is set, skip SDF evaluation for old slots;
        // they still contribute via motion factors and corner factors.
        for (const auto& slot : window)
        {
            if (params.sdf_current_slot_only && &slot != &window.back())
                continue;
            auto pose_xy = slot.pose.index({torch::indexing::Slice(0, 2)});
            auto pose_theta = slot.pose.index({torch::indexing::Slice(2, 3)});

            const auto query = model.sdf_query_at_pose(slot.lidar_points, pose_xy, pose_theta);
            const auto slot_obs_loss =
                compute_observation_loss_from_query(model, params, slot.lidar_points, pose_theta, query);
            total_loss = total_loss + slot_obs_loss;
        }

        // --- 3. Motion factors between consecutive slots ---
        for (size_t i = 1; i < window.size(); i++)
        {
            const auto& curr = window[i];
            const auto& prev = window[i - 1];

            auto pose_delta = curr.pose - prev.pose;
            auto raw_residual = pose_delta - curr.odom_delta_tensor;

            auto angle_res = raw_residual.index({2});
            auto wrapped_angle = torch::atan2(torch::sin(angle_res), torch::cos(angle_res));
            auto residual = torch::cat({raw_residual.index({torch::indexing::Slice(0, 2)}),
                                        wrapped_angle.unsqueeze(0)});

            auto res_col = residual.unsqueeze(1);
            auto motion_loss = 0.5f * torch::matmul(res_col.t(), torch::matmul(curr.motion_prec_tensor, res_col)).squeeze();
            total_loss = total_loss + motion_loss;
        }

        // --- 4. Corner observation factors (newest slots only, Huber-saturated) ---
        if (params.enable_corner_tracking)
        {
        const float corner_gain  = params.corner_precision_gain;
        const float corner_huber = params.corner_huber_sigma;   // whitened (σ) units
        const int corner_start = std::max(0, static_cast<int>(window.size()) - params.corner_max_slots);
        for (size_t si = corner_start; si < window.size(); si++)
        {
            const auto& slot = window[si];
            if (!slot.corner_cw.defined() || slot.corner_cw.size(0) == 0) continue;

            auto pose_xy    = slot.pose.index({torch::indexing::Slice(0, 2)});  // [2]
            auto pose_theta = slot.pose.index({2});
            auto cos_th = torch::cos(pose_theta);
            auto sin_th = torch::sin(pose_theta);

            // Batched predicted observations: z_hat_i = R(-θ) · (c_world_i - t), for all corners at once.
            auto dw = slot.corner_cw - pose_xy;                    // [N,2]
            auto dwx = dw.index({torch::indexing::Slice(), 0});   // [N]
            auto dwy = dw.index({torch::indexing::Slice(), 1});   // [N]
            auto pred_x = cos_th * dwx + sin_th * dwy;            // [N]
            auto pred_y = -sin_th * dwx + cos_th * dwy;          // [N]
            auto predicted = torch::stack({pred_x, pred_y}, 1);   // [N,2]

            auto residual = slot.corner_detected - predicted;     // [N,2] robot frame

            // Anisotropic graded precision Λ_det (robot frame): rᵀΛr downweights the residual
            // component along ill-determined (shallow/marginal) directions automatically.
            // Batched quadratic form: maha_i = r_iᵀ Λ_i r_i via bmm.
            auto Lr = torch::bmm(slot.corner_information, residual.unsqueeze(2)).squeeze(2);  // [N,2]
            auto maha_sq = corner_gain * (residual * Lr).sum(1);                              // [N]
            // Huber saturation in whitened space (m = √(rᵀΛr) is in σ units).
            auto m = torch::sqrt(maha_sq + 1e-8f);                                            // [N]
            auto huber_weight = torch::where(m <= corner_huber,
                torch::ones_like(m), corner_huber / m);                                      // [N]
            total_loss = total_loss + (0.5f * huber_weight * maha_sq).sum();
        }
        } // enable_corner_tracking

        // --- 5. Object-anchor factors (validated modelled objects as SE(2) pose landmarks) ---
        if (params.object_anchor.enable)
        {
            const int obj_start = std::max(0, static_cast<int>(window.size()) - params.object_anchor_max_slots);
            for (size_t si = obj_start; si < window.size(); si++)
            {
                const auto& slot = window[si];
                if (slot.object_anchors.empty()) continue;
                auto pose_xy    = slot.pose.index({torch::indexing::Slice(0, 2)});
                auto pose_theta = slot.pose.index({torch::indexing::Slice(2, 3)});
                total_loss = total_loss + ObjectAnchorFactor::loss(
                    slot.object_anchors, pose_xy, pose_theta, params.object_anchor, device);
            }
        }

        // --- 6. RGB structural-contour factors (image gradient vs projected model contours) ---
        // MUST mirror rc::gn::ImageEdgeFactorGn term for term. Two things break if it does not:
        // GnShadow stops measuring solver agreement and starts measuring objective divergence, and
        // — the one that matters more — compute_posterior_covariance() takes an AUTOGRAD Hessian of
        // this function, so a term missing here is a term missing from the reported sigma, which is
        // the pre-registered success criterion for the whole feature.
        // Gated on enable AND drive, the SAME condition rc::gn::build_factors uses. If the two gates
        // differed, then in shadow mode the torch objective would carry the term and the GN objective
        // would not, and GnShadow's cross-scoring would report a backend disagreement that is really
        // just two different objectives.
        if (params.image_edge.enable and params.image_edge.drive)
        {
            const int img_start = std::max(0, static_cast<int>(window.size()) - std::max(1, params.image_edge_max_slots));
            for (size_t si = img_start; si < window.size(); si++)
            {
                const auto& slot = window[si];
                if (slot.image_edges.empty()) continue;
                auto pose_xy    = slot.pose.index({torch::indexing::Slice(0, 2)});
                auto pose_theta = slot.pose.index({torch::indexing::Slice(2, 3)});
                total_loss = total_loss + ImageEdgeFactor::loss(
                    slot.image_edges, pose_xy, pose_theta, params.image_edge, device);
            }
        }

        return total_loss;
    }

    RoomConcept::WindowManager::LossBreakdown
    RoomConcept::WindowManager::compute_rfe_loss_breakdown(
        const Model& model, const Params& params, torch::Device device) const
    {
        LossBreakdown bd;
        if (window.empty()) return bd;

        // 1. Boundary prior
        if (boundary_prior.valid) {
            const auto& oldest_pose = window.front().pose;
            auto mu = torch::tensor(
                {boundary_prior.mu[0], boundary_prior.mu[1], boundary_prior.mu[2]},
                torch::TensorOptions().dtype(torch::kFloat32).device(device));
            auto prec_data = torch::zeros({3, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    prec_data[i][j] = boundary_prior.precision(i, j);
            auto raw_diff = oldest_pose.detach() - mu;
            auto angle_diff = raw_diff.index({2});
            auto wrapped_diff = torch::cat({raw_diff.index({torch::indexing::Slice(0, 2)}),
                torch::atan2(torch::sin(angle_diff), torch::cos(angle_diff)).unsqueeze(0)});
            auto diff = wrapped_diff.unsqueeze(1);
            bd.boundary = (0.5f * torch::matmul(diff.t(), torch::matmul(prec_data, diff)).squeeze()).item<float>();
        }

        // 2. Observation factors
        float obs_acc = 0.f;
        for (const auto& slot : window) {
            if (params.sdf_current_slot_only && &slot != &window.back())
                continue;
            auto pose_xy    = slot.pose.detach().index({torch::indexing::Slice(0, 2)});
            auto pose_theta = slot.pose.detach().index({torch::indexing::Slice(2, 3)});
            const auto query = model.sdf_query_at_pose(slot.lidar_points, pose_xy, pose_theta);
            const float slot_loss = compute_observation_loss_from_query(
                model, params, slot.lidar_points, pose_theta, query).item<float>();
            obs_acc += slot_loss;
        }
        bd.obs = obs_acc;

        // 3. Motion factors
        float mot_acc = 0.f;
        for (size_t i = 1; i < window.size(); i++) {
            const auto& curr = window[i];
            const auto& prev = window[i - 1];
            auto pose_delta  = curr.pose.detach() - prev.pose.detach();
            auto raw_residual = pose_delta - curr.odom_delta_tensor;
            auto angle_res   = raw_residual.index({2});
            auto wrapped_angle = torch::atan2(torch::sin(angle_res), torch::cos(angle_res));
            auto residual = torch::cat({raw_residual.index({torch::indexing::Slice(0, 2)}),
                                        wrapped_angle.unsqueeze(0)});
            auto res_col = residual.unsqueeze(1);
            mot_acc += (0.5f * torch::matmul(res_col.t(),
                torch::matmul(curr.motion_prec_tensor, res_col)).squeeze()).item<float>();
        }
        bd.motion = mot_acc;

        // 4. Corner factors
        if (params.enable_corner_tracking) {
            const float corner_gain  = params.corner_precision_gain;
            const float corner_huber = params.corner_huber_sigma;
            const int corner_start = std::max(0, static_cast<int>(window.size()) - params.corner_max_slots);
            float cor_acc = 0.f;
            for (size_t si = corner_start; si < window.size(); si++) {
                const auto& slot = window[si];
                if (slot.corner_obs.empty()) continue;
                auto pose_xy    = slot.pose.detach().index({torch::indexing::Slice(0, 2)});
                auto pose_theta = slot.pose.detach().index({2});
                auto cos_th = torch::cos(pose_theta);
                auto sin_th = torch::sin(pose_theta);
                for (const auto& obs : slot.corner_obs) {
                    auto c_w = torch::tensor({obs.model_corner_world.x(), obs.model_corner_world.y()},
                        torch::TensorOptions().dtype(torch::kFloat32).device(device));
                    auto dw = c_w - pose_xy;
                    auto pred_x = cos_th * dw.index({0}) + sin_th * dw.index({1});
                    auto pred_y = -sin_th * dw.index({0}) + cos_th * dw.index({1});
                    auto predicted = torch::stack({pred_x, pred_y});
                    auto detected  = torch::tensor({obs.detected_robot.x(), obs.detected_robot.y()},
                        torch::TensorOptions().dtype(torch::kFloat32).device(device));
                    auto residual  = detected - predicted;
                    auto Lambda = torch::tensor(
                        {obs.information(0,0), obs.information(0,1), obs.information(1,0), obs.information(1,1)},
                        torch::TensorOptions().dtype(torch::kFloat32).device(device)).reshape({2,2});
                    auto maha_sq = corner_gain * torch::matmul(residual.unsqueeze(0),
                                       torch::matmul(Lambda, residual.unsqueeze(1))).squeeze();
                    auto m = torch::sqrt(maha_sq + 1e-8f);
                    auto hw = torch::where(m <= corner_huber, torch::ones_like(m), corner_huber / m);
                    cor_acc += (0.5f * hw * maha_sq).item<float>();
                }
            }
            bd.corner = cor_acc;
        }

        // 5. Object-anchor factors
        if (params.object_anchor.enable) {
            const int obj_start = std::max(0, static_cast<int>(window.size()) - params.object_anchor_max_slots);
            float obj_acc = 0.f;
            for (size_t si = obj_start; si < window.size(); si++) {
                const auto& slot = window[si];
                if (slot.object_anchors.empty()) continue;
                auto pose_xy    = slot.pose.detach().index({torch::indexing::Slice(0, 2)});
                auto pose_theta = slot.pose.detach().index({torch::indexing::Slice(2, 3)});
                obj_acc += ObjectAnchorFactor::loss(
                    slot.object_anchors, pose_xy, pose_theta, params.object_anchor, device).item<float>();
            }
            bd.object = obj_acc;
        }

        // 6. RGB structural contours — same order and same terms as compute_rfe_loss.
        if (params.image_edge.enable and params.image_edge.drive) {
            const int img_start = std::max(0, static_cast<int>(window.size()) - std::max(1, params.image_edge_max_slots));
            float img_acc = 0.f;
            for (size_t si = img_start; si < window.size(); si++) {
                const auto& slot = window[si];
                if (slot.image_edges.empty()) continue;
                auto pose_xy    = slot.pose.detach().index({torch::indexing::Slice(0, 2)});
                auto pose_theta = slot.pose.detach().index({torch::indexing::Slice(2, 3)});
                img_acc += ImageEdgeFactor::loss(
                    slot.image_edges, pose_xy, pose_theta, params.image_edge, device).item<float>();
            }
            bd.image = img_acc;
        }

        return bd;
    }

    void RoomConcept::WindowManager::recompute_boundary_prior(
        const Model& model, const Params& params, torch::Device device)
    {
        if (window.empty())
            return;

        const auto& oldest = window.front();

        // ── mu update (Solution C already applied in append()) ────────────────
        // By this point boundary_prior.mu was already conditionally updated when the
        // slot was dropped from the window. We only need to update it here for the
        // "recompute after recovery / reset" path where append() was not called.
        // Unconditional write is safe: recompute is only called when window_slid==true
        // and the slot that slid out already had its quality checked in append().
        // (No additional guard needed here — the pose in mu was set correctly.)

        // ── Hessian computation (Solution B) ─────────────────────────────────
        // If the oldest slot's scan was of poor quality (sdf_mse_final exceeds the
        // threshold), the H_obs term would encode a high-confidence direction toward
        // a contaminated pose. In that case we use only the kinematic (motion) factor
        // for the precision matrix, which is always trustworthy.
        const bool use_obs_hessian =
            (oldest.sdf_mse_final < params.boundary_hessian_quality_threshold);

        Eigen::Matrix3f H = Eigen::Matrix3f::Zero();

        if (use_obs_hessian)
        {
            // ── Full path: H_obs + H_motion + H_prior (original behaviour) ───
            auto oldest_pose_for_hess = oldest.pose.clone().detach().requires_grad_(true);
            auto pose_xy    = oldest_pose_for_hess.index({torch::indexing::Slice(0, 2)});
            auto pose_theta = oldest_pose_for_hess.index({torch::indexing::Slice(2, 3)});

            auto loss = compute_observation_loss(model, params, oldest.lidar_points, pose_xy, pose_theta);

            if (window.size() > 1)
            {
                const auto& next = window[1];
                auto next_pose = next.pose.detach();
                auto delta = next_pose - oldest_pose_for_hess;
                auto raw_res = delta - next.odom_delta_tensor;
                auto angle_r = raw_res.index({2});
                auto residual = torch::cat({raw_res.index({torch::indexing::Slice(0, 2)}),
                                            torch::atan2(torch::sin(angle_r), torch::cos(angle_r)).unsqueeze(0)});
                auto res_col = residual.unsqueeze(1);
                loss = loss + 0.5f * torch::matmul(res_col.t(),
                                                    torch::matmul(next.motion_prec_tensor, res_col)).squeeze();
            }

            if (boundary_prior.valid)
            {
                auto mu_t = torch::tensor(
                    {boundary_prior.mu[0], boundary_prior.mu[1], boundary_prior.mu[2]},
                    torch::TensorOptions().dtype(torch::kFloat32).device(device));
                auto diff  = (oldest_pose_for_hess - mu_t).unsqueeze(1);
                auto prec_t = torch::zeros({3, 3},
                    torch::TensorOptions().dtype(torch::kFloat32).device(device));
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 3; c++)
                        prec_t[r][c] = boundary_prior.precision(r, c);
                loss = loss + 0.5f * torch::matmul(diff.t(),
                                                    torch::matmul(prec_t, diff)).squeeze();
            }

            H = RoomConcept::autograd_hessian_3x3(loss, oldest_pose_for_hess);
        }
        else
        {
            // ── Degraded path: kinematic precision only ───────────────────────
            // H_obs is not included because the scan was contaminated (obstacle,
            // displacement confusion). Use only the motion-factor precision from the
            // next slot, which reflects purely the odometry model uncertainty.
            // This gives a conservative, direction-agnostic anchor.
            if (window.size() > 1)
            {
                auto prec_cpu = window[1].motion_prec_tensor.to(torch::kCPU);
                auto acc = prec_cpu.accessor<float, 2>();
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 3; c++)
                        H(r, c) = acc[r][c];
            }
            else
            {
                // No motion link available: fall back to a weak isotropic prior.
                H = Eigen::Matrix3f::Identity() * params.eigenvalue_clamp_boundary;
            }
        }

        // ── Eigenvalue clamping (floor + ceiling) ─────────────────────────────
        // Floor prevents degenerate (zero-precision) directions.
        // Ceiling (eigenvalue_clamp_boundary_max) prevents over-confident priors
        // regardless of cause — acts as a safety net for both paths.
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(H);
        Eigen::Vector3f evals = eig.eigenvalues()
            .cwiseMax(params.eigenvalue_clamp_boundary)
            .cwiseMin(params.eigenvalue_clamp_boundary_max);
        H = eig.eigenvectors() * evals.asDiagonal() * eig.eigenvectors().transpose();

        boundary_prior.precision = H;
        boundary_prior.valid = true;
    }

    // =====================================================================
    //  FEJ + Schur marginalization of the dropping slot (boundary_fej_schur)
    //
    //  Called BEFORE append() pops, with x₀ = window.front() (dropping) and
    //  x₁ = window[1] (its Markov blanket via the motion factor). Forms the exact
    //  Schur complement of x₀ over x₁ from the factors touching x₀ — the previous
    //  marginal prior on x₀, x₀'s (soft quality-weighted) obs factor, and the
    //  x₀↔x₁ motion factor — and stores a First-Estimates-Jacobian-frozen prior on x₁.
    //
    //     Λ₀₀ = Λ_prev + w₀·H_obs(x₀*) + Ω     Λ₀₁ = Λ₁₀ = −Ω     Λ₁₁ = Ω
    //     g₀  = Λ_prev·Δ_prev + g_prev + w₀·g_obs − Ω·r          g₁ = +Ω·r
    //     Λ_marg = Ω − Ω Λ₀₀⁻¹ Ω               g_marg = Ω·r + Ω Λ₀₀⁻¹ g₀
    //
    //  All Jacobians are evaluated at the CURRENT (converged-from-previous-frame)
    //  estimates x₀*, x₁* and then frozen — that fixed linearization point is what
    //  prevents the "erroneous information gain" ratchet (loss_boundary 0→559).
    // =====================================================================
    void RoomConcept::WindowManager::marginalize_oldest(
        const Model& model, const Params& params, torch::Device device)
    {
        // Only meaningful when a slide is imminent (window full) and there IS a blanket slot.
        if (static_cast<int>(window.size()) < params.rfe_window_size || window.size() < 2)
            return;

        const auto& x0slot = window.front();   // dropping
        const auto& x1slot = window[1];        // Markov blanket (motion-linked survivor)

        auto to_eig3 = [](const torch::Tensor& t) {
            auto c = t.detach().to(torch::kCPU).contiguous();
            auto a = c.accessor<float, 1>();
            return Eigen::Vector3f(a[0], a[1], a[2]);
        };
        auto to_eig33 = [](const torch::Tensor& t) {
            auto c = t.detach().to(torch::kCPU).contiguous();
            auto a = c.accessor<float, 2>();
            Eigen::Matrix3f M;
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) M(i, j) = a[i][j];
            return M;
        };
        auto wrap = [](float a) { return std::atan2(std::sin(a), std::cos(a)); };

        const Eigen::Vector3f x0 = to_eig3(x0slot.pose);   // FEJ linearization points
        const Eigen::Vector3f x1 = to_eig3(x1slot.pose);

        // ── Motion factor x₀↔x₁ (precision Ω, residual r), linearized at (x₀*, x₁*) ──
        // odom_delta_tensor of x₁ is the odometry delta from x₀ to x₁; J₀ = −I, J₁ = +I.
        const Eigen::Matrix3f Omega = to_eig33(x1slot.motion_prec_tensor);
        const Eigen::Vector3f odom = to_eig3(x1slot.odom_delta_tensor);
        Eigen::Vector3f r = (x1 - x0) - odom;
        r[2] = wrap(r[2]);

        // ── Dropped slot's obs factor: H_obs, g_obs at x₀*, softly quality-weighted ──
        // w₀ = 1/(1+(sdf/σ_q)²): a poorly-fit dropped slot injects weak, high-covariance
        // information that cannot anchor the survivor (continuous replacement of the hard gate).
        Eigen::Matrix3f H_obs = Eigen::Matrix3f::Zero();
        Eigen::Vector3f g_obs = Eigen::Vector3f::Zero();
        {
            const float sq = std::max(1e-6f, params.boundary_quality_sigma);
            const float ratio = x0slot.sdf_mse_final / sq;
            const float w0 = 1.0f / (1.0f + ratio * ratio);

            auto pose_h = x0slot.pose.clone().detach().requires_grad_(true);
            auto pose_xy    = pose_h.index({torch::indexing::Slice(0, 2)});
            auto pose_theta = pose_h.index({torch::indexing::Slice(2, 3)});
            auto obs = compute_observation_loss(model, params, x0slot.lidar_points, pose_xy, pose_theta);

            // First-order gradient (graph retained/created so the Hessian pass can reuse it).
            auto grad = torch::autograd::grad({obs}, {pose_h}, {}, /*retain*/true, /*create*/true)[0];
            g_obs = w0 * to_eig3(grad);
            for (int i = 0; i < 3; ++i)
            {
                auto row = torch::autograd::grad({grad.index({i})}, {pose_h}, {},
                                                 /*retain*/(i < 2), /*create*/false)[0];
                auto rc = row.to(torch::kCPU); auto ra = rc.accessor<float, 1>();
                for (int j = 0; j < 3; ++j) H_obs(i, j) = w0 * ra[j];
            }
            H_obs = 0.5f * (H_obs + H_obs.transpose().eval());
        }

        // ── Assemble the joint linear system over (x₀, x₁) from factors touching x₀ ──
        Eigen::Matrix3f L00 = H_obs + Omega;     // + Λ_prev below
        Eigen::Matrix3f L01 = -Omega;            // L10 = L01ᵀ = −Ω (symmetric)
        Eigen::Matrix3f L11 = Omega;
        Eigen::Vector3f g0  = g_obs - Omega * r; // motion: J₀ᵀΩr = −Ωr
        Eigen::Vector3f g1  =         Omega * r; // motion: J₁ᵀΩr = +Ωr

        // Previous marginal prior on x₀ (FEJ form 0.5·ΔᵀΛΔ + gᵀΔ, Δ = wrap(x₀* − mu_prev)).
        if (boundary_prior.valid)
        {
            Eigen::Vector3f dprev = x0 - boundary_prior.mu;
            dprev[2] = wrap(dprev[2]);
            L00 += boundary_prior.precision;
            g0  += boundary_prior.precision * dprev + boundary_prior.grad;
        }

        // ── Schur complement: eliminate x₀ ──
        const Eigen::Matrix3f L00inv =
            L00.ldlt().solve(Eigen::Matrix3f::Identity());
        Eigen::Matrix3f Lmarg = L11 - L01.transpose() * L00inv * L01;  // Ω − ΩL₀₀⁻¹Ω
        Eigen::Vector3f gmarg = g1 - L01.transpose() * L00inv * g0;    // Ωr + ΩL₀₀⁻¹g₀

        // ── Convert to (mean, precision) form BEFORE clamping ──
        // The marginal's mode is  mean = x₁* − Λ_marg⁻¹·g_marg  (≈ x₀*+odom, the odometry-propagated
        // anchor). We store the prior as a pure quadratic 0.5·(x−mean)ᵀΛ(x−mean) rather than a frozen
        // point + linear term, because the eigenvalue clamp below scales Λ: clamping Λ while KEEPING
        // g_marg would move the effective mode by Λ_clamped⁻¹·g_marg (with Ω~1e4 that is a ~0.2 m
        // phantom anchor that drags the window off the walls — the bug seen 2026-07-16). Computing the
        // mean from the UNCLAMPED Λ_marg fixes the mode; clamping then only softens the confidence.
        const Eigen::Vector3f mean = x1 - Lmarg.ldlt().solve(gmarg);

        // Eigenvalue clamp (floor + ceiling) for numerical safety, as in the legacy path.
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(Lmarg);
        Eigen::Vector3f evals = eig.eigenvalues()
            .cwiseMax(params.eigenvalue_clamp_boundary)
            .cwiseMin(params.eigenvalue_clamp_boundary_max);
        Lmarg = eig.eigenvectors() * evals.asDiagonal() * eig.eigenvectors().transpose();

        Eigen::Vector3f mu = mean;
        mu[2] = wrap(mu[2]);
        if (not Lmarg.allFinite() or not mu.allFinite())
            return;   // keep the previous prior rather than poison with NaNs

        boundary_prior.mu        = mu;                       // FROZEN marginal mode (FEJ anchor)
        boundary_prior.precision = Lmarg;
        boundary_prior.grad      = Eigen::Vector3f::Zero();  // mean form ⇒ no separate linear term
        boundary_prior.valid     = true;
    }

} // namespace rc
