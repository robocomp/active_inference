/*
 *    Copyright (C) 2026 by YOUR NAME HERE
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    RoboComp is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "specificworker.h"
#include "camera_visualizer.h"
#include <print>
#include <random>
#include <fstream>
#include <QDir>
#include <QFileInfo>
#include <QVBoxLayout>

///////////////////////////////////////////////////////////////////////////////
SpecificWorker::SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check)
    : GenericWorker(configLoader, tprx)
{
    this->startup_check_flag = startup_check;
    if (this->startup_check_flag)
    {
        this->startup_check();
    }
    else
    {
#ifdef HIBERNATION_ENABLED
        hibernationChecker.start(500);
#endif
        statemachine.setChildMode(QState::ExclusiveStates);
        statemachine.start();
        auto error = statemachine.errorString();
        if (error.length() > 0) { qWarning() << error; throw error; }
    }
}

///////////////////////////////////////////////////////////////////////////////
SpecificWorker::~SpecificWorker()
{
    stop_lidar_thread = true;
    if (read_lidar_th.joinable())
        read_lidar_th.join();
    save_robot_pose_once();
    room_concept_.stop();
    std::cout << "Destroying SpecificWorker" << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::initialize()
{
    std::cout << "initialize worker" << std::endl;
    GenericWorker::initialize();

    QObject::connect(QCoreApplication::instance(), &QCoreApplication::aboutToQuit,
                     this, &SpecificWorker::save_robot_pose_once, Qt::UniqueConnection);

    // ── RoomConcept params ─────────────────────────────────────────────────
    params.PREDICTION_EARLY_EXIT = configLoader.get<bool>("RoomConcept.PredictionEarlyExit");
    room_concept_.params.num_iterations             = configLoader.get<int>("RoomConcept.NumIterations");
    room_concept_.params.rfe_window_size            = configLoader.get<int>("RoomConcept.WindowSize");
    room_concept_.params.max_lidar_points           = configLoader.get<int>("RoomConcept.MaxLidarPoints");
    room_concept_.params.rfe_max_lidar_per_old_slot = configLoader.get<int>("RoomConcept.MaxLidarOldSlot");
    room_concept_.params.recovery_loss_threshold    = static_cast<float>(configLoader.get<double>("RoomConcept.RecoveryLossThreshold"));
    room_concept_.params.recovery_consecutive_count = configLoader.get<int>("RoomConcept.RecoveryConsecutiveCount");
    try { room_concept_.params.odom_noise_scale = static_cast<float>(configLoader.get<double>("RoomConcept.OdomNoiseScale")); } catch (...) {}
    try { room_concept_.params.differential_test_enabled = configLoader.get<bool>("RoomConcept.DifferentialTest"); } catch (...) {}
    try { room_concept_.params.sdf_current_slot_only = configLoader.get<bool>("RoomConcept.SdfCurrentSlotOnly"); } catch (...) {}
    try { params.OptimizerType = configLoader.get<std::string>("RoomConcept.OptimizerType");
          room_concept_.params.optimizer_type = params.OptimizerType; } catch (...) {}

    try { room_concept_.params.sigma_sdf               = static_cast<float>(configLoader.get<double>("RoomConcept.SigmaSdf")); } catch (...) {}
    try { room_concept_.params.prediction_trust_factor  = static_cast<float>(configLoader.get<double>("RoomConcept.PredictionTrustFactor")); } catch (...) {}
    try { room_concept_.params.min_tracking_steps       = configLoader.get<int>("RoomConcept.MinTrackingSteps"); } catch (...) {}
    try { room_concept_.params.rotation_sdf_coupling    = static_cast<float>(configLoader.get<double>("RoomConcept.RotationSdfCoupling")); } catch (...) {}

    try { room_concept_.params.lbfgs_lr               = static_cast<float>(configLoader.get<double>("RoomConcept.LbfgsLr")); } catch (...) {}
    try { room_concept_.params.lbfgs_history_size     = configLoader.get<int>("RoomConcept.LbfgsHistorySize"); } catch (...) {}
    try { room_concept_.params.lbfgs_tolerance_grad   = configLoader.get<double>("RoomConcept.LbfgsToleranceGrad"); } catch (...) {}
    try { room_concept_.params.lbfgs_tolerance_change = configLoader.get<double>("RoomConcept.LbfgsToleranceChange"); } catch (...) {}

    try { room_concept_.params.learning_rate_pos        = static_cast<float>(configLoader.get<double>("RoomConcept.LearningRatePos")); } catch (...) {}
    try { room_concept_.params.rfe_obs_sigma            = static_cast<float>(configLoader.get<double>("RoomConcept.ObsSigma")); } catch (...) {}
    try { room_concept_.params.rfe_huber_delta          = static_cast<float>(configLoader.get<double>("RoomConcept.HuberDelta")); } catch (...) {}
    try { room_concept_.params.convergence_relative_tol = static_cast<float>(configLoader.get<double>("RoomConcept.ConvergenceRelTol")); } catch (...) {}
    try { room_concept_.params.convergence_min_iters    = configLoader.get<int>("RoomConcept.ConvergenceMinIters"); } catch (...) {}

    try { room_concept_.params.rfe_boundary_quality_gate          = configLoader.get<bool>("RoomConcept.BoundaryQualityGate"); } catch (...) {}
    try { room_concept_.params.boundary_hessian_quality_threshold = static_cast<float>(configLoader.get<double>("RoomConcept.BoundaryHessianQualityThreshold")); } catch (...) {}
    try { room_concept_.params.boundary_mu_quality_threshold      = static_cast<float>(configLoader.get<double>("RoomConcept.BoundaryMuQualityThreshold")); } catch (...) {}
    try { room_concept_.params.eigenvalue_clamp_boundary_max      = static_cast<float>(configLoader.get<double>("RoomConcept.EigenvalueClampBoundaryMax")); } catch (...) {}

    try { room_concept_.params.recovery_cooldown_frames = configLoader.get<int>("RoomConcept.RecoveryCooldownFrames"); } catch (...) {}

    try { room_concept_.params.velocity_adaptive_weights  = configLoader.get<bool>("RoomConcept.VelocityAdaptiveWeights"); } catch (...) {}
    try { room_concept_.params.linear_velocity_threshold  = static_cast<float>(configLoader.get<double>("RoomConcept.LinearVelocityThreshold")); } catch (...) {}
    try { room_concept_.params.angular_velocity_threshold = static_cast<float>(configLoader.get<double>("RoomConcept.AngularVelocityThreshold")); } catch (...) {}
    try { room_concept_.params.weight_boost_factor        = static_cast<float>(configLoader.get<double>("RoomConcept.WeightBoostFactor")); } catch (...) {}
    try { room_concept_.params.weight_reduction_factor    = static_cast<float>(configLoader.get<double>("RoomConcept.WeightReductionFactor")); } catch (...) {}
    try { room_concept_.params.weight_smoothing_alpha     = static_cast<float>(configLoader.get<double>("RoomConcept.WeightSmoothingAlpha")); } catch (...) {}

    try { room_concept_.params.cmd_noise_trans             = static_cast<float>(configLoader.get<double>("RoomConcept.CmdNoiseTrans")); } catch (...) {}
    try { room_concept_.params.cmd_noise_rot               = static_cast<float>(configLoader.get<double>("RoomConcept.CmdNoiseRot")); } catch (...) {}
    try { room_concept_.params.cmd_noise_base              = static_cast<float>(configLoader.get<double>("RoomConcept.CmdNoiseBase")); } catch (...) {}
    try { room_concept_.params.odom_noise_trans            = static_cast<float>(configLoader.get<double>("RoomConcept.OdomNoiseTrans")); } catch (...) {}
    try { room_concept_.params.odom_noise_rot              = static_cast<float>(configLoader.get<double>("RoomConcept.OdomNoiseRot")); } catch (...) {}
    try { room_concept_.params.odom_noise_base             = static_cast<float>(configLoader.get<double>("RoomConcept.OdomNoiseBase")); } catch (...) {}
    try { room_concept_.params.encoder_rot_slip_k          = static_cast<float>(configLoader.get<double>("RoomConcept.EncoderRotSlipK")); } catch (...) {}
    try { room_concept_.params.stationary_motion_threshold = static_cast<float>(configLoader.get<double>("RoomConcept.StationaryMotionThreshold")); } catch (...) {}

    try { room_concept_.params.learn_motion_model      = configLoader.get<bool>("RoomConcept.LearnMotionModel"); } catch (...) {}
    try { room_concept_.params.motion_learn_alpha      = static_cast<float>(configLoader.get<double>("RoomConcept.MotionLearnAlpha")); } catch (...) {}
    try { room_concept_.params.motion_learn_beta       = static_cast<float>(configLoader.get<double>("RoomConcept.MotionLearnBeta")); } catch (...) {}
    try { room_concept_.params.motion_learn_min_omega  = static_cast<float>(configLoader.get<double>("RoomConcept.MotionLearnMinOmega")); } catch (...) {}
    try { room_concept_.params.motion_learn_min_trans  = static_cast<float>(configLoader.get<double>("RoomConcept.MotionLearnMinTrans")); } catch (...) {}
    try { room_concept_.params.motion_learn_min_frames         = configLoader.get<int>("RoomConcept.MotionLearnMinFrames"); } catch (...) {}
    try { room_concept_.params.motion_learn_quality_threshold  = static_cast<float>(configLoader.get<double>("RoomConcept.MotionLearnQualityThreshold")); } catch (...) {}

    try { room_concept_.params.enable_corner_tracking = configLoader.get<bool>("RoomConcept.EnableCornerTracking"); } catch (...) {}
    try { room_concept_.params.far_points_weight      = configLoader.get<bool>("RoomConcept.FarPointsWeight"); } catch (...) {}
    try { room_concept_.params.far_points_exponent    = static_cast<float>(configLoader.get<double>("RoomConcept.FarPointsExponent")); } catch (...) {}
    try { room_concept_.params.far_points_min_weight  = static_cast<float>(configLoader.get<double>("RoomConcept.FarPointsMinWeight")); } catch (...) {}
    try { room_concept_.params.use_cuda               = configLoader.get<bool>("RoomConcept.UseCuda"); } catch (...) {}
    try { room_concept_.params.debug_log_enabled      = configLoader.get<bool>("RoomConcept.DebugLog"); } catch (...) {}

    try { room_concept_.params.rerun_enabled        = configLoader.get<bool>("RoomConcept.RerunEnabled"); } catch (...) {}
    try { room_concept_.params.rerun_host           = configLoader.get<std::string>("RoomConcept.RerunHost"); } catch (...) {}
    try { room_concept_.params.rerun_port           = configLoader.get<int>("RoomConcept.RerunPort"); } catch (...) {}
    try { room_concept_.params.rerun_sdf_every_n    = configLoader.get<int>("RoomConcept.RerunSdfEveryN"); } catch (...) {}
    try { room_concept_.params.rerun_sdf_resolution = configLoader.get<int>("RoomConcept.RerunSdfResolution"); } catch (...) {}
    try { room_concept_.params.rerun_max_queue      = configLoader.get<int>("RoomConcept.RerunMaxQueue"); } catch (...) {}

    // ── DSR stabilization thresholds ──────────────────────────────────────
    try { params.STABLE_FRAMES_REQUIRED = configLoader.get<int>("DSR.StableFramesRequired"); } catch (...) {}
    try { params.STABLE_SDF_MSE_MAX     = static_cast<float>(configLoader.get<double>("DSR.StableSdfMseMax")); } catch (...) {}
    try { params.STABLE_COV_TT_MAX      = static_cast<float>(configLoader.get<double>("DSR.StableCovTtMax")); } catch (...) {}

    // ── EpistemicController params ─────────────────────────────────────────
    auto& ec = epistemic_controller_.params;
    auto& ep = epistemic_controller_.epistemic_planner().params;
    try { ec.num_arc_curvatures    = configLoader.get<int>("EpistemicController.NumArcCurvatures"); } catch (...) {}
    try { ec.horizon_steps         = configLoader.get<int>("EpistemicController.HorizonSteps"); } catch (...) {}
    try { ec.dt                    = static_cast<float>(configLoader.get<double>("EpistemicController.Dt")); } catch (...) {}
    try { ec.max_adv_speed         = static_cast<float>(configLoader.get<double>("EpistemicController.MaxAdvSpeed")); } catch (...) {}
    try { ec.max_rot_speed         = static_cast<float>(configLoader.get<double>("EpistemicController.MaxRotSpeed")); } catch (...) {}
    try { ec.w_epistemic           = static_cast<float>(configLoader.get<double>("EpistemicController.WEpistemic")); } catch (...) {}
    try { ec.w_pragmatic           = static_cast<float>(configLoader.get<double>("EpistemicController.WPragmatic")); } catch (...) {}
    try { ec.w_heading             = static_cast<float>(configLoader.get<double>("EpistemicController.WHeading")); } catch (...) {}
    try { ec.w_boundary            = static_cast<float>(configLoader.get<double>("EpistemicController.WBoundary")); } catch (...) {}
    try { ec.k_rot                 = static_cast<float>(configLoader.get<double>("EpistemicController.KRot")); } catch (...) {}
    try { ec.gaussian_sigma        = static_cast<float>(configLoader.get<double>("EpistemicController.GaussianSigma")); } catch (...) {}
    try { ec.speed_horizon_s       = static_cast<float>(configLoader.get<double>("EpistemicController.SpeedHorizonS")); } catch (...) {}
    try { ec.obstacle_radius       = static_cast<float>(configLoader.get<double>("EpistemicController.ObstacleRadius")); } catch (...) {}
    try { ec.obstacle_k            = static_cast<float>(configLoader.get<double>("EpistemicController.ObstacleK")); } catch (...) {}
    try { ec.obstacle_step_cap     = static_cast<float>(configLoader.get<double>("EpistemicController.ObstacleStepCap")); } catch (...) {}
    try { ec.w_obstacle            = static_cast<float>(configLoader.get<double>("EpistemicController.WObstacle")); } catch (...) {}
    try { ec.wall_filter_margin    = static_cast<float>(configLoader.get<double>("EpistemicController.WallFilterMargin")); } catch (...) {}
    try { ec.bandwidth_coupling    = static_cast<float>(configLoader.get<double>("EpistemicController.BandwidthCoupling")); } catch (...) {}
    try { ec.sdf_safe              = static_cast<float>(configLoader.get<double>("EpistemicController.SdfSafe")); } catch (...) {}
    try { ec.sdf_danger            = static_cast<float>(configLoader.get<double>("EpistemicController.SdfDanger")); } catch (...) {}
    try { ec.governor_alpha_min    = static_cast<float>(configLoader.get<double>("EpistemicController.GovernorAlphaMin")); } catch (...) {}
    try { ec.fim_corner_sigma      = static_cast<float>(configLoader.get<double>("EpistemicController.FimCornerSigma")); } catch (...) {}
    try { ec.fim_max_range         = static_cast<float>(configLoader.get<double>("EpistemicController.FimMaxRange")); } catch (...) {}
    try { ep.grid_resolution       = static_cast<float>(configLoader.get<double>("EpistemicController.GridResolution")); } catch (...) {}
    try { ep.min_distance          = static_cast<float>(configLoader.get<double>("EpistemicController.MinDistance")); } catch (...) {}
    try { ep.max_candidates        = configLoader.get<int>("EpistemicController.MaxCandidates"); } catch (...) {}
    try { ep.target_wall_margin    = static_cast<float>(configLoader.get<double>("EpistemicController.TargetWallMargin")); } catch (...) {}
    try { ep.angular_dominance_ratio = static_cast<float>(configLoader.get<double>("EpistemicController.AngularDominanceRatio")); } catch (...) {}
    try { ep.w_exploration         = static_cast<float>(configLoader.get<double>("EpistemicController.WExploration")); } catch (...) {}
    try { ep.ior_cell_size         = static_cast<float>(configLoader.get<double>("EpistemicController.IorCellSize")); } catch (...) {}
    try { ep.ior_decay_time        = static_cast<float>(configLoader.get<double>("EpistemicController.IorDecayTime")); } catch (...) {}
    try { ep.w_ior                 = static_cast<float>(configLoader.get<double>("EpistemicController.WIor")); } catch (...) {}
    try { ep.fim_corner_sigma      = static_cast<float>(configLoader.get<double>("EpistemicController.FimCornerSigma")); } catch (...) {}
    try { ep.fim_max_range         = static_cast<float>(configLoader.get<double>("EpistemicController.FimMaxRange")); } catch (...) {}
    try { ep.arrival_distance      = static_cast<float>(configLoader.get<double>("EpistemicController.ArrivalDistance")); } catch (...) {}
    try { ep.dwell_time            = static_cast<float>(configLoader.get<double>("EpistemicController.DwellTime")); } catch (...) {}
    try { max_lin_accel_           = static_cast<float>(configLoader.get<double>("EpistemicController.MaxLinAccel")); } catch (...) {}
    try { max_rot_accel_           = static_cast<float>(configLoader.get<double>("EpistemicController.MaxRotAccel")); } catch (...) {}

    // ── Start lidar reader thread ──────────────────────────────────────────
    read_lidar_th = std::thread(&SpecificWorker::read_lidar, this);
    qInfo() << __FUNCTION__ << "Started lidar reader";

    // ── Wire RoomConcept run context ───────────────────────────────────────
    rc::RoomConcept::RunContext run_ctx;
    run_ctx.sensor_buffer   = &lidar_buffer;
    run_ctx.velocity_buffer = &velocity_buffer_;
    run_ctx.odometry_buffer = &odometry_buffer_;
    room_concept_.set_run_context(run_ctx);
    room_concept_.params.prediction_early_exit = params.PREDICTION_EARLY_EXIT;
    rfe_saved_window_size_ = room_concept_.params.rfe_window_size;

    initialize_room_model_from_svg();
    const std::string pose_path = pose_file_path();
    room_concept_.set_seed_pose_file(pose_path);
    std::cout << "Pose seed file: " << pose_path << "\n";

    // custom widget
    //graph_viewers[agent_name]->add_custom_widget_to_dock("room concept", &custom_widget);
    qInfo() << __FUNCTION__ << "Adding custom widget to dock" << agent_name.c_str();
    graph_viewers.at("")->add_custom_widget_to_dock("layout", &custom_widget);
    viewer_2d_ = std::make_unique<rc::Viewer2D>(custom_widget.frame, params.GRID_MAX_DIM, true);
    viewer_2d_->show();
    viewer_2d_->add_robot(params.ROBOT_WIDTH, params.ROBOT_LENGTH, 0.f, 0.f, QColor("blue"));

    // Free-Energy time series in the lower frame of the custom widget.
    if (custom_widget.frame_series->layout() == nullptr)
    {
        auto* series_layout = new QVBoxLayout(custom_widget.frame_series);
        series_layout->setContentsMargins(2, 2, 2, 2);
        series_layout->setSpacing(2);
        custom_widget.frame_series->setLayout(series_layout);
    }
    ts_plot_fe_ = new rc::TimeSeriesPlot(custom_widget.frame_series);
    ts_plot_fe_->set_visible_window(60.f);
    ts_plot_fe_->add_series("free_energy", QColor(255, 170, 0), 1.8f, 0);
    ts_plot_fe_->add_series("cov_det_scaled", QColor(0, 190, 255), 1.6f, 0);
    custom_widget.frame_series->layout()->addWidget(ts_plot_fe_);
    
    // Load room polygon for visualizations
    std::vector<Eigen::Vector2f> room_polygon_for_viz;
    if (room_initialized_from_svg_polygon_)
    {
         room_polygon_for_viz = rc::SvgRoomLoader::load_polygon_points(

            "beta_layout.svg", "room_contour", false, true);
         if (room_polygon_for_viz.size() >= 3)
             viewer_2d_->draw_room_polygon(room_polygon_for_viz, false);
    }

    // Camera visualizer
    camera_viz_ = std::make_unique<rc::CameraVisualizer>(G, room_polygon_for_viz, nullptr);
    connect(custom_widget.btn_camera_viz, &QPushButton::clicked, this, &SpecificWorker::slot_show_camera_visualization);

    // ── DSR: resolve existing graph node IDs ──────────────────────────────
    check_init_graph_is_valid();

    // RT_API
    rt_api = G->get_rt_api();

    // ── Connect DSR signals ────────────────────────────────────────────────
    // connect(G.get(), &DSR::DSRGraph::update_node_signal,      this, &SpecificWorker::modify_node_slot);
    // connect(G.get(), &DSR::DSRGraph::update_edge_signal,      this, &SpecificWorker::modify_edge_slot);
    // connect(G.get(), &DSR::DSRGraph::update_node_attr_signal, this, &SpecificWorker::modify_node_attrs_slot);
    // connect(G.get(), &DSR::DSRGraph::update_edge_attr_signal, this, &SpecificWorker::modify_edge_attrs_slot);
    // connect(G.get(), &DSR::DSRGraph::del_edge_signal,         this, &SpecificWorker::del_edge_slot);
    // connect(G.get(), &DSR::DSRGraph::del_node_signal,         this, &SpecificWorker::del_node_slot);

    room_concept_.start();

    // ── Wire mouse-driven pose reset ───────────────────────────────────────
    connect(viewer_2d_.get(), &rc::Viewer2D::robot_moved,
            this, [this](QPointF p){ slot_mouse_translate(p); });
    connect(viewer_2d_.get(), &rc::Viewer2D::robot_rotate,
            this, [this](QPointF p){ slot_mouse_rotate(p); });
}

///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::compute()
{
    const auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    // const auto& [robot_pose_gt_, lidar_data_, obstacle_data_] = lidar_buffer.read(timestamp);
    // if (!lidar_data_.has_value())
    // { qWarning() << "No lidar data"; return; }

    const auto loc_res  = room_concept_.get_last_result();
    const bool have_loc = loc_res.has_value() && loc_res->ok;

    const Eigen::Affine2f pose_for_draw = best_available_pose(loc_res, have_loc);

    // ── Update 2-D viewer ─────────────────────────────────────────────────
    const Eigen::Affine2f loc_pose = have_loc ? loc_res->robot_pose : pose_for_draw;
    const bool use_loc = have_loc && !loc_res->lidar_scan.empty();
    //const std::vector<Eigen::Vector3f>& draw_points =
    //    use_loc ? loc_res->lidar_scan : lidar_data_->first;
    
    if(use_loc)    
        viewer_2d_->update_frame({
            .lidar_points     = loc_res->lidar_scan,
            .display_pose     = pose_for_draw,
            .covariance       = have_loc ? loc_res->covariance : Eigen::Matrix3f::Identity(),
            .max_lidar_points = params.MAX_LIDAR_DRAW_POINTS,
            .have_loc         = have_loc,
            .is_initialized   = room_concept_.is_initialized(),
            .has_room_polygon = room_initialized_from_svg_polygon_,
            .room_width       = have_loc ? loc_res->state[0] : 0.f,
            .room_length      = have_loc ? loc_res->state[1] : 0.f,
            .loc_pose         = loc_pose,
            .use_loc_pose     = use_loc,
        });

    if (have_loc && !loc_res->corner_matches.empty())
        viewer_2d_->draw_corners(loc_res->corner_matches, pose_for_draw);
    else
        viewer_2d_->draw_corners({}, pose_for_draw);

    // ── DSR graph update ───────────────────────────────────────────────────
    if (have_loc)
        update_dsr(*loc_res);

    update_ui(loc_res, pose_for_draw);
    fps_counter_.print("[Compute]", 3000);
}

///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::update_dsr(const rc::RoomConcept::UpdateResult& res)
{
    const float sdf_mse = res.sdf_mse;
    const float cov_tt  = (res.covariance.rows() > 2 && res.covariance.cols() > 2)
                          ? res.covariance(2, 2) : 1.f;
    const bool stable   = (res.iterations_used == 0)
                          && sdf_mse < params.STABLE_SDF_MSE_MAX
                          && cov_tt  < params.STABLE_COV_TT_MAX;

    qInfo() << "Localization stable:" << stable
            << "| sdf_mse:" << sdf_mse << "(" << params.STABLE_SDF_MSE_MAX << ")"
            << "| cov_tt:" << cov_tt << "(" << params.STABLE_COV_TT_MAX << ")"
            << "| iterations_used:" << res.iterations_used;

    if (!room_node_created_)
    {
        stable_frames_ = stable ? stable_frames_ + 1 : 0;
        if (stable_frames_ >= params.STABLE_FRAMES_REQUIRED)
            dsr_create_room_and_reparent(res);
        else
            dsr_update_pose(res);   // world->robot RT while waiting for stable room creation
    }
    else
    {
        dsr_update_pose(res);       // robot->room RT once room node exists
    }
}

///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::dsr_update_pose(const rc::RoomConcept::UpdateResult& res)
{
    if (!G || !rt_api) return;

    const Eigen::Matrix2f R = res.robot_pose.linear();
    const Eigen::Vector2f t = res.robot_pose.translation();
    const float theta_room_to_robot = std::atan2(R(1, 0), R(0, 0));

    // Convert room->robot estimate into robot->room when the room is a child of the robot.
    const Eigen::Vector2f t_robot_to_room = -(R.transpose() * t);
    const float theta_robot_to_room = -theta_room_to_robot;

    const uint64_t parent_id = room_node_created_ ? dsr_robot_id_ : dsr_world_id_;
    const uint64_t child_id  = room_node_created_ ? dsr_room_id_  : dsr_robot_id_;

    auto parent_opt = G->get_node(parent_id);
    if (!parent_opt.has_value()) return;

    const float x = room_node_created_ ? t_robot_to_room.x() : t.x();
    const float y = room_node_created_ ? t_robot_to_room.y() : t.y();
    const float theta = room_node_created_ ? theta_robot_to_room : theta_room_to_robot;

    rt_api->insert_or_assign_edge_RT(parent_opt.value(), child_id,
                                     {x, y, 0.f},
                                     {0.f, 0.f, theta});

    Eigen::Matrix3f cov_se2 = Eigen::Matrix3f::Identity();
    if (res.covariance.rows() >= 3 && res.covariance.cols() >= 3)
        cov_se2 = res.covariance.topLeftCorner<3, 3>();

    // If we invert pose (robot->room), propagate covariance through the inverse map.
    if (room_node_created_)
    {
        const float c = std::cos(theta_room_to_robot);
        const float s = std::sin(theta_room_to_robot);
        Eigen::Matrix3f J = Eigen::Matrix3f::Zero();
        J(0, 0) = -c;
        J(0, 1) = -s;
        J(0, 2) =  s * t.x() - c * t.y();
        J(1, 0) =  s;
        J(1, 1) = -c;
        J(1, 2) =  c * t.x() + s * t.y();
        J(2, 2) = -1.f;
        cov_se2 = J * cov_se2 * J.transpose();
    }

    std::vector<float> cov_flat(36, 0.f);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            cov_flat[r * 6 + c] = cov_se2(r, c);

    auto edge_rt = G->get_edge(parent_id, child_id, "RT");
    if (!edge_rt.has_value())
    {
        qWarning() << "dsr_update_pose: edge RT not found after insert_or_assign_edge_RT";
        return;
    }
    G->add_or_modify_attrib_local<rt_se2_covariance_att>(edge_rt.value(), cov_flat);
    G->insert_or_assign_edge(edge_rt.value());

    // Verify what was actually written to DSR using rt_api
    auto rt_edge = DSR::RT_API::get_edge_RT(parent_opt.value(), child_id);
    if (rt_edge.has_value())
    {
        auto rtmat = rt_api->get_edge_RT_as_rtmat(rt_edge.value());
        if (rtmat.has_value())
        {
            // Verification output disabled.
            // std::print("[dsr_update_pose VERIFY via rt_api] Written RT edge: trans=({:.4f},{:.4f}) rot_angle={:.4f} rad ({:.2f} deg)\n",
            //            trans.x(), trans.y(), rot_angle, rot_angle * 180.0 / M_PI);
        }
        else
        {
            // std::print("[dsr_update_pose VERIFY] Could not convert Edge to RTMat\n");
        }
    }
    else
    {
        // std::print("[dsr_update_pose VERIFY] Could not retrieve RT edge via rt_api\n");
    }
}

///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::dsr_create_room_and_reparent(const rc::RoomConcept::UpdateResult& res)
{
    if (!G) return;

    const auto room_polygon = room_concept_.nominal_room_polygon();
    std::vector<float> polygon_x;
    std::vector<float> polygon_y;
    polygon_x.reserve(room_polygon.size());
    polygon_y.reserve(room_polygon.size());
    for (const auto& vertex : room_polygon)
    {
        polygon_x.push_back(vertex.x());
        polygon_y.push_back(vertex.y());
    }

    if (const auto room_nodes = G->get_nodes_by_type("room"); !room_nodes.empty())
    {
        dsr_room_id_ = room_nodes.front().id();
        room_node_created_ = true;
        stable_frames_ = 0;
        qInfo() << "DSR: reusing existing room node id=" << dsr_room_id_;
        dsr_update_pose(res);
        return;
    }

    DSR::Node room_node = DSR::Node::create<room_node_type>("room");
    room_node.attrs()[delimiting_polygon_x_str.data()] = DSR::Attribute{polygon_x, 0, 0};
    room_node.attrs()[delimiting_polygon_y_str.data()] = DSR::Attribute{polygon_y, 0, 0};
    room_node.attrs()[room_height_str.data()] = DSR::Attribute{params.room_height, 0, 0};
    const auto room_id_opt = G->insert_node(room_node);
    if (!room_id_opt.has_value())
    {
        qWarning() << "DSR: failed to create room node";
        return;
    }

    dsr_room_id_ = room_id_opt.value();
    room_node_created_ = true;
    stable_frames_ = 0;
    qInfo() << "DSR: created room node id=" << dsr_room_id_ << "hanging from robot id=" << dsr_robot_id_;

    dsr_update_pose(res);
}

///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::check_init_graph_is_valid()
{
    if (!G) { qWarning() << "dsr_init_graph: DSR graph not available"; return; }

    // Resolve the root/world node by type (name may vary, e.g. "root", "world")
    const auto root_nodes = G->get_nodes_by_type("root");
    if (!root_nodes.empty())
    {
        dsr_world_id_ = root_nodes.front().id();
        qInfo() << "DSR: found root node id=" << dsr_world_id_
                << "name=" << root_nodes.front().name().c_str();
    }
    else { qWarning() << "dsr_init_graph: no 'root' type node found in graph"; return; }

    // Resolve the robot node by type
    const auto robot_nodes = G->get_nodes_by_type("robot");
    if (!robot_nodes.empty())
    {
        dsr_robot_id_ = robot_nodes.front().id();
        qInfo() << "DSR: found robot node id=" << dsr_robot_id_
                << "name=" << robot_nodes.front().name().c_str();
    }
    else { qWarning() << "dsr_init_graph: no 'robot' type node found in graph"; return; }
}

///////////////////////////////////////////////////////////////////////////////
Eigen::Affine2f SpecificWorker::best_available_pose(
    const std::optional<rc::RoomConcept::UpdateResult>& loc_res, bool have_loc) const
{
    if (have_loc)
        return loc_res->robot_pose;
    if (room_concept_.is_initialized())
    {
        const auto s = room_concept_.get_current_state();
        Eigen::Affine2f p = Eigen::Affine2f::Identity();
        p.translation() = Eigen::Vector2f(s[2], s[3]);
        p.linear() = Eigen::Rotation2Df(s[4]).toRotationMatrix();
        return p;
    }
    return Eigen::Affine2f::Identity();
}

///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::update_ui(const std::optional<rc::RoomConcept::UpdateResult>& loc_res,
                               const Eigen::Affine2f& /*pose_for_draw*/)
{
    if (!loc_res.has_value()) return;
    if (ts_plot_sdf_) ts_plot_sdf_->add_point("sdf_mse", loc_res->sdf_mse);
    if (ts_plot_fe_)
    {
        ts_plot_fe_->add_point("free_energy", loc_res->final_loss);

        const float det_cov = std::max(1e-12f, std::abs(loc_res->covariance.determinant()));
        float det_scaled = -std::log10(det_cov) / 10.f;  // map ~[1..1e-10] to [0..1]
        if (det_scaled < 0.f) det_scaled = 0.f;
        if (det_scaled > 1.f) det_scaled = 1.f;
        ts_plot_fe_->add_point("cov_det_scaled", det_scaled);
    }
}

///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::read_lidar()
{
    FPSCounter lidar_fps;
    auto wait_period = std::chrono::milliseconds(getPeriod("Compute"));
    while (!stop_lidar_thread)
    {
        const auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        try
        {
            RoboCompLidar3D::TData data_high;
            try
            {
                data_high = lidar3d_proxy->getLidarData(
                    "", 0.f, static_cast<float>(M_PI) * 2.f, params.LIDAR_LOW_DECIMATION_FACTOR);
            }
            catch (const Ice::Exception& e)
            { qWarning() << "[read_lidar] getLidarData failed:" << e.what(); std::terminate(); }

            std::vector<Eigen::Vector3f> points_high;
            std::vector<Eigen::Vector2f> obstacle_points;
            points_high.reserve(data_high.points.size());
            obstacle_points.reserve(data_high.points.size());

            const float min_h_mm   = params.LIDAR_HIGH_MIN_HEIGHT   * 1000.f;
            const float floor_h_mm = params.LIDAR_HIGH_FLOOR_HEIGHT * 1000.f;
            const float robot_h_mm = params.ROBOT_HEIGHT             * 1000.f;
            constexpr float mm2m   = 0.001f;

            for (const auto& p : data_high.points)
            {
                const bool is_high     = p.z > min_h_mm;
                const bool is_obstacle = p.z > floor_h_mm && p.z < robot_h_mm;
                if (is_high || is_obstacle)
                {
                    const float mx = p.x * mm2m;
                    const float my = p.y * mm2m;
                    if (is_high)     points_high.emplace_back(mx, my, p.z * mm2m);
                    if (is_obstacle) obstacle_points.emplace_back(mx, my);
                }
            }

            lidar_buffer.put<1>(
                std::make_pair(std::move(points_high), static_cast<std::int64_t>(data_high.timestamp)),
                timestamp);
            lidar_buffer.put<2>(std::move(obstacle_points), timestamp);

            const long p_ms = static_cast<long>(data_high.period);
            if (wait_period > std::chrono::milliseconds(p_ms + 2)) --wait_period;
            else if (wait_period < std::chrono::milliseconds(p_ms - 2)) ++wait_period;

            lidar_fps.print("[LidarThread]", 2000);
            std::this_thread::sleep_for(wait_period);
        }
        catch (const Ice::Exception& e)
        { qWarning() << "[read_lidar] Ice exception:" << e.what(); }
    }
}

///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::initialize_room_model_from_svg()
{
    const auto room_polygon = rc::SvgRoomLoader::load_polygon_points(
        "beta_layout.svg", "room_contour", false, true);
    if (room_polygon.size() >= 3)
    {
        room_concept_.configure_room_from_polygon(room_polygon);
        room_initialized_from_svg_polygon_ = true;
        qInfo() << "Configured RoomConcept from SVG polygon, vertices:" << room_polygon.size();
        return;
    }
    room_concept_.configure_room_from_rect(params.GRID_MAX_DIM.width(), params.GRID_MAX_DIM.height());
    room_initialized_from_svg_polygon_ = false;
    qWarning() << "SVG polygon not loaded; using rectangular fallback.";
}

///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::save_robot_pose_on_exit() const
{
    Eigen::Vector3f pose = Eigen::Vector3f::Zero();
    if (const auto loc = room_concept_.get_last_result(); loc.has_value() && loc->ok)
    {
        pose[0] = loc->state[2]; pose[1] = loc->state[3]; pose[2] = loc->state[4];
    }
    else if (room_concept_.is_initialized())
    {
        const auto state = room_concept_.get_current_state();
        pose[0] = state[2]; pose[1] = state[3]; pose[2] = state[4];
    }
    else return;

    const QString qpath = QString::fromStdString(pose_file_path());
    QDir().mkpath(QFileInfo(qpath).absolutePath());
    std::ofstream out(qpath.toStdString(), std::ios::trunc);
    if (!out.is_open()) { qWarning() << "Cannot open pose file:" << qpath; return; }
    out << pose[0] << ' ' << pose[1] << ' ' << pose[2] << '\n';
    std::cout << "Saved robot pose: " << pose[0] << " " << pose[1] << " " << pose[2] << "\n";
}

void SpecificWorker::save_robot_pose_once()
{
    if (pose_saved_.exchange(true)) return;
    save_robot_pose_on_exit();
}

std::string SpecificWorker::pose_file_path() const
{
    auto find_etc_upwards = [](const QString& start) -> QString {
        QDir dir(start);
        for (int depth = 0; depth < 8; ++depth)
        {
            const QString etc_dir = dir.absoluteFilePath("etc");
            if (QDir(etc_dir).exists()) return etc_dir;
            if (!dir.cdUp()) break;
        }
        return {};
    };
    const QString from_app = find_etc_upwards(QCoreApplication::applicationDirPath());
    if (!from_app.isEmpty()) return (from_app + "/last_robot_pose.txt").toStdString();
    const QString from_cwd = find_etc_upwards(QDir::currentPath());
    if (!from_cwd.isEmpty()) return (from_cwd + "/last_robot_pose.txt").toStdString();
    return (QDir(QCoreApplication::applicationDirPath() + "/../etc").absolutePath()
            + "/last_robot_pose.txt").toStdString();
}

///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::navigate_to_target(const std::optional<rc::RoomConcept::UpdateResult>& loc_res,
                                        const std::optional<rc::ObstacleData>& obstacles)
{
    if (!loc_res.has_value() || !loc_res->ok) return;

    epistemic_controller_.set_robot_state(loc_res->robot_pose, loc_res->covariance);
    epistemic_controller_.set_localization_quality(loc_res->sdf_mse);
    if (obstacles.has_value())
        epistemic_controller_.set_lidar_obstacles(*obstacles);

    auto plan = epistemic_controller_.plan();
    if (!plan.has_value()) return;
    auto cmd = plan->command;

    const auto now = std::chrono::steady_clock::now();
    const float dt = std::chrono::duration<float>(now - prev_cmd_time_).count();
    prev_cmd_time_ = now;

    auto ramp = [](float target, float current, float max_accel, float dt_s) -> float {
        const float max_delta = max_accel * dt_s;
        return std::clamp(target, current - max_delta, current + max_delta);
    };
    cmd.adv_x = ramp(cmd.adv_x, prev_cmd_.adv_x, max_lin_accel_, dt);
    cmd.rot   = ramp(cmd.rot,   prev_cmd_.rot,   max_rot_accel_, dt);
    prev_cmd_ = cmd;

    try { omnirobot_proxy->setSpeedBase(cmd.adv_x * 1000.f, 0.f, cmd.rot); }
    catch (const Ice::Exception& e)
    { qWarning() << "[navigate_to_target] setSpeedBase failed:" << e.what(); }
}

///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::slot_mouse_translate(QPointF scene_pos)
{
    // Shift+Left: move robot to clicked position, keep current heading.
    // Use push_command (thread-safe queue) — never call set_robot_pose() directly
    // from the GUI thread while the localization thread may be mid-backward().
    const auto state = room_concept_.get_current_state();
    const float theta = state[4];
    room_concept_.push_command(rc::RoomConcept::CmdSetPose{
        static_cast<float>(scene_pos.x()),
        static_cast<float>(scene_pos.y()),
        theta});
    qInfo() << "[mouse] Translate robot to" << scene_pos.x() << scene_pos.y();
}

///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::slot_mouse_rotate(QPointF scene_pos)
{
    // Ctrl+Left: rotate robot to face the clicked point, keep current position.
    const auto state = room_concept_.get_current_state();
    const float rx    = state[2];
    const float ry    = state[3];
    const float theta = std::atan2(static_cast<float>(scene_pos.y()) - ry,
                                   static_cast<float>(scene_pos.x()) - rx);
    room_concept_.push_command(rc::RoomConcept::CmdSetPose{rx, ry, theta});
    qInfo() << "[mouse] Rotate robot toward" << scene_pos.x() << scene_pos.y()
            << "-> theta" << qRadiansToDegrees(theta) << "deg";
}

///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::slot_show_camera_visualization()
{
    if (camera_viz_)
    {
        camera_viz_->update_frame();
        camera_viz_->show();
        camera_viz_->raise();
        camera_viz_->activateWindow();
    }
}

///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::emergency()
{
    std::cout << "Emergency worker" << std::endl;
}

void SpecificWorker::restore()
{
    std::cout << "Restore worker" << std::endl;
}

int SpecificWorker::startup_check()
{
    std::cout << "Startup check" << std::endl;
    QTimer::singleShot(200, QCoreApplication::instance(), SLOT(quit()));
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
/// ICE INTERFACE CALLBACKS
///////////////////////////////////////////////////////////////////////////////

void SpecificWorker::JoystickAdapter_sendData(RoboCompJoystickAdapter::TData data)
{
    rc::VelocityCommand cmd;
    for (const auto& axis : data.axes)
    {
        if      (axis.name == "rotate")  cmd.rot   = axis.value;
        else if (axis.name == "advance") cmd.adv_y = axis.value / 1000.0f;
        else if (axis.name == "side")    cmd.adv_x = 0.0f;
    }
    cmd.timestamp  = std::chrono::high_resolution_clock::now();
    cmd.recv_ts_ms = static_cast<std::int64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    velocity_buffer_.put<0>(std::move(cmd), static_cast<std::uint64_t>(cmd.recv_ts_ms));
}

void SpecificWorker::FullPoseEstimationPub_newFullPose(RoboCompFullPoseEstimation::FullPoseEuler pose)
{
    static std::mt19937 gen{std::random_device{}()};
    const float nf = params.ODOMETRY_NOISE_FACTOR;

    auto add_noise = [&](float value) -> float {
        if (nf <= 0.f || value == 0.f) return value;
        std::normal_distribution<float> dist(0.f, std::abs(value) * nf);
        return value + dist(gen);
    };

    rc::OdometryReading odom;
    // Webots reports adv with opposite sign of our body Y+ (forward) axis.
    // Flipping one in-plane axis flips the right-hand-rule yaw sign as well,
    // so pose.rot must also be negated to match the math-CCW convention used
    // downstream (SDF rotation, Eigen::Rotation2Df, EKF Jacobian, DSR writer).
    odom.adv          = add_noise(-pose.adv);
    odom.side         = add_noise( pose.side);
    odom.rot          = add_noise(-pose.rot);
    odom.source_ts_ms = pose.timestamp;
    odom.recv_ts_ms   = static_cast<std::int64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    odom.timestamp    = std::chrono::high_resolution_clock::time_point(
        std::chrono::milliseconds(pose.timestamp));
    odometry_buffer_.put<0>(std::move(odom), static_cast<std::uint64_t>(odom.recv_ts_ms));
}
