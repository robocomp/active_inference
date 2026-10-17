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

    // =====================================================================
    //  Threading: start / stop / run / get_last_result / push_command
    // =====================================================================

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

        // Announce the non-default motion-prior configuration, so a run whose predictions came from
        // the encoder alone says so in its own log rather than only in a config file nobody re-reads.
        if (not params.use_command_velocity_prior)
            qInfo() << "[RoomConcept] UseCommandVelocityPrior = false: the commanded (joystick/controller)"
                    << "velocity is computed and logged but EXCLUDED from the motion prior;"
                    << "motion_prior_source will read measured / fallback_zero, never fused / command.";

        loc_running_ = true;
        loc_thread_ = std::thread([this]
        {
            pthread_setname_np(pthread_self(), "localizer");
            run();
        });
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

    void RoomConcept::run()
    {
        init_recovery_log();
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
                auto res = update(lidar_high, vel_snap, odom_snap);
                // Stamped HERE, before this result is stored: a relocalisation in section 8 below advances
                // the epoch AFTER this frame is published, so the jump — and the new epoch — ride on the
                // NEXT result, the first one that actually carries the new pose. Seed validation above runs
                // before update(), so its jump is already in this result and so is its epoch.
                res.reloc_epoch = reloc_epoch_.load(std::memory_order_relaxed);
                update_ms = std::chrono::duration<float, std::milli>(
                    std::chrono::high_resolution_clock::now() - t_update_start_).count();

                // Optimizer-timing telemetry for the UI readout (early-exit ⇔ iterations_used <= 0).
                opt_total_count_.fetch_add(1, std::memory_order_relaxed);
                if (res.iterations_used <= 0)
                    opt_earlyexit_count_.fetch_add(1, std::memory_order_relaxed);
                opt_update_us_sum_.fetch_add(static_cast<std::uint64_t>(update_ms * 1000.0f),
                                             std::memory_order_relaxed);

                // Lean per-update timing CSV (loc-thread only): distribution + CUDA warmup curve.
                if (params.optimizer_timing_csv)
                {
                    if (!opt_csv_open_attempted_)
                    {
                        opt_csv_open_attempted_ = true;
                        opt_csv_.open("etc/optimizer_timing.csv", std::ios::out | std::ios::trunc);
                        if (opt_csv_.is_open())
                            opt_csv_ << "# device=CPU\n"
                                     << "ts_ms,update_ms,iters,early_exit,window_size,t_adam_ms,t_cov_ms,sdf_mse,cov_tt,t_slam_ms,t_wall_ms,t_polish_ms\n";
                    }
                    if (opt_csv_.is_open())
                    {
                        opt_csv_ << res.timestamp_ms << ',' << update_ms << ',' << res.iterations_used
                                 << ',' << (res.iterations_used <= 0 ? 1 : 0) << ',' << window_mgr_.size()
                                 << ',' << last_t_adam_ms_ << ',' << last_t_cov_ms_ << ',' << res.sdf_mse
                                 << ',' << res.covariance(2, 2) << ',' << last_t_slam_ms_ << ',' << last_t_wall_ms_
                                 << ',' << last_t_polish_ms_ << '\n';
                        opt_csv_.flush();
                    }
                    last_t_slam_ms_ = last_t_wall_ms_ = last_t_polish_ms_ = 0.f;   // 0 next row = stage did not run
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
                if (!rerun_room_polygon_sent_ && model_ != nullptr && model_->use_polygon)
                {
                    const auto& poly = model_->polygon_vertices;
                    if (poly.size() >= 3)
                    {
                        rf.has_room_polygon = true;
                        rf.room_polygon.reserve(poly.size());
                        for (const auto& v : poly)
                            rf.room_polygon.push_back({v.x(), v.y()});
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

                    const Eigen::VectorXf acc = model_->sdf(grid_robot);

                    rf.has_sdf_grid = true;
                    rf.sdf_w = res_grid;
                    rf.sdf_h = res_grid;
                    rf.sdf_origin_x = ox;
                    rf.sdf_origin_y = oy;
                    rf.sdf_cell_size = cell;
                    rf.sdf_values.resize(static_cast<size_t>(res_grid) * static_cast<size_t>(res_grid));
                    for (size_t i = 0; i < rf.sdf_values.size(); ++i)
                        rf.sdf_values[i] = acc(static_cast<Eigen::Index>(i));
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
                // than waiting for the next compute() tick. The callback runs ON THIS (localiser) THREAD —
                // it reaches scene_graph_->update() directly, no Qt marshal — so any scene-graph state the
                // GUI thread also reads must be published under a lock (see pinned_landmarks()). Fired
                // OUTSIDE result_mutex_. Removes ~one compute-period of lidar→RT latency.
                if (res.ok && on_result_ready_)
                    on_result_ready_();

                // ===== 8. RELOCALISATION: lost belief + rival modes (see relocalisation_step) =====
                relocalisation_step(res, lidar_high.first);

                wait_period = std::chrono::milliseconds(0);


            // End of heavy localization update path (executed only for new lidar frames).
            }
        }

        rerun_logger_.stop();
        loc_running_ = false;
    }

    void RoomConcept::set_initial_state(float width, float length, float x, float y, float phi)
    {
        model_ = std::make_shared<Model>();
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


    void RoomConcept::configure_room_estimate()
    {
        init_use_polygon_ = false;
        init_polygon_vertices_.clear();
        plain_slam_seed_ = true;   // a fresh room to learn: the scan map starts again from the current pose
        wall_map_ = wallmap::WallMap{};
        wall_map_.params = params.wall_map;
        box_channel_.configure(params.box);
        if (params.box.enabled) box_channel_.open_csv("etc/box_layout.csv");
        if (params.box.enabled)
            qInfo().noquote() << QString("[room][boxes] SIDECAR PUBLISHING: the delimiting polygon comes "
                                         "from the box layout, not the wall map. every=%1 frames, cell=%2 m, "
                                         "band=[%3,%4] m")
                                     .arg(params.box.every_frames).arg(params.box.cell, 0, 'f', 2)
                                     .arg(params.box.z_min, 0, 'f', 2).arg(params.box.z_max, 0, 'f', 2);
        wall_map_.params.obs_sigma   = params.rfe_obs_sigma;     // the wall factor's σ and knee ARE the SDF's
        wall_map_.params.huber_delta = params.rfe_huber_delta;
        map_ready_ = false;
        wall_reanchored_ = false;
        last_wall_status_.clear();
        std::scoped_lock lk(wall_map_mutex_);
        derived_polygon_.clear();
    }

    void RoomConcept::begin_room_estimation(std::string_view why)
    {
        // Idempotent: the crossing evidence that calls this keeps arriving, and a second reset would
        // throw away walls already learnt in the new space.
        if (estimating() and not wall_frozen_)
            return;

        const bool was_given = not estimating();
        // ★ OUT OF A ROOM THIS RUN LEARNT (2026-10-11, door crossing): its map layout was ADOPTED as the given room and
        //   plain SLAM kept running as its shadow. Both flags are released here — while map_layout_adopted_ stays set,
        //   the old room's layout keeps overwriting the derived polygon every frame and the new room is never learnt.
        //   The shadow's scan map is KEPT: it is already registered through the doorway, so the pose that plain SLAM
        //   takes back over is continuous with the one the SDF localiser was publishing (they differ by the shadow's
        //   disagreement, centimetres), and the map simply grows into the new room.
        const bool keep_slam_map = map_layout_adopted_;
        map_layout_adopted_ = false;
        {
            std::scoped_lock lk(wall_map_mutex_);
            map_layout_.clear();
        }
        runtime_estimate_.store(true, std::memory_order_relaxed);
        wall_frozen_ = false;          // layout_state() is Localizing while this is set, whatever the mode
        configure_room_estimate();     // fresh WallMap, map_ready_ = false, derived polygon gone
        if (keep_slam_map) plain_slam_seed_ = false;

        // ⚠ NOT RESET HERE, and deliberately: the camera-calibration channels were handed the OLD room's
        // polygon at startup (calib_->set_room_polygon in specificworker_startup) and this class has no
        // handle on them. They will keep projecting the apartment until the new polygon closes. Harmless
        // for a topology run; wire it before trusting a camera-derived number taken past a doorway.
        qInfo().noquote()
            << QString("[room][layout] LOCALIZING -> SEARCHING (%1). %2 The map-guided checks — recovery, "
                       "relocalisation, symmetry, prediction early-exit — are now off because there is no "
                       "map to judge against, and stay off until the new polygon closes and is re-anchored.")
                   .arg(QString::fromStdString(std::string(why)))
                   .arg(was_given ? "The given layout is no longer fitted."
                                  : "The learnt layout has been un-frozen.");
    }

    bool RoomConcept::take_pending_reanchor(Eigen::Affine2f& T)
    {
        if (not has_pending_reanchor_)
            return false;
        T = pending_reanchor_;
        pending_reanchor_     = Eigen::Affine2f::Identity();
        has_pending_reanchor_ = false;
        return true;
    }

    void RoomConcept::set_robot_pose(float x, float y, float theta, bool manual_reset)
    {
        if (model_ == nullptr)
        {
            qWarning() << "Cannot set robot pose: model not initialized";
            return;
        }

        // Directly update the model pose
        model_->set_pose(Eigen::Vector3f(x, y, theta));

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

} // namespace rc
