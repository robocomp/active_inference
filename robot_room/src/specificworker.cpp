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
    graph_viewers[agent_name]->add_custom_widget_to_dock("room concept", &custom_widget);
    //widget_2d = qobject_cast<DSR::QScene2dViewer*> (viewer->get_widget(opts::scene));

    viewer_2d_ = std::make_unique<rc::Viewer2D>(centralWidget(), params.GRID_MAX_DIM, true);
    // viewer_2d_->show();
    // viewer_2d_->add_robot(params.ROBOT_WIDTH, params.ROBOT_LENGTH, 0.f, 0.f, QColor("blue"));
    // if (room_initialized_from_svg_polygon_)
    // {
    //     const auto room_polygon = rc::SvgRoomLoader::load_polygon_points(
    //         "beta_layout.svg", "room_contour", false, true);
    //     if (room_polygon.size() >= 3)
    //         viewer_2d_->draw_room_polygon(room_polygon, false);
    // }

    // ── DSR: create world + robot nodes ───────────────────────────────────
    //dsr_init_graph();

    // ── Connect DSR signals ────────────────────────────────────────────────
    connect(G.get(), &DSR::DSRGraph::update_node_signal,      this, &SpecificWorker::modify_node_slot);
    connect(G.get(), &DSR::DSRGraph::update_edge_signal,      this, &SpecificWorker::modify_edge_slot);
    connect(G.get(), &DSR::DSRGraph::update_node_attr_signal, this, &SpecificWorker::modify_node_attrs_slot);
    connect(G.get(), &DSR::DSRGraph::update_edge_attr_signal, this, &SpecificWorker::modify_edge_attrs_slot);
    connect(G.get(), &DSR::DSRGraph::del_edge_signal,         this, &SpecificWorker::del_edge_slot);
    connect(G.get(), &DSR::DSRGraph::del_node_signal,         this, &SpecificWorker::del_node_slot);

    room_concept_.start();
}

///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::compute()
{
    const auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    const auto& [robot_pose_gt_, lidar_data_, obstacle_data_] = lidar_buffer.read(timestamp);
    if (!lidar_data_.has_value())
    { qWarning() << "No lidar data"; return; }

    const auto loc_res  = room_concept_.get_last_result();
    const bool have_loc = loc_res.has_value() && loc_res->ok;
    const Eigen::Affine2f pose_for_draw = best_available_pose(loc_res, have_loc);

    // ── Update 2-D viewer ─────────────────────────────────────────────────
    // const Eigen::Affine2f loc_pose = have_loc ? loc_res->robot_pose : pose_for_draw;
    // const bool use_loc = have_loc && !loc_res->lidar_scan.empty();
    // const std::vector<Eigen::Vector3f>& draw_points =
    //     use_loc ? loc_res->lidar_scan : lidar_data_->first;

    // viewer_2d_->update_frame({
    //     .lidar_points     = draw_points,
    //     .display_pose     = pose_for_draw,
    //     .max_lidar_points = params.MAX_LIDAR_DRAW_POINTS,
    //     .have_loc         = have_loc,
    //     .is_initialized   = room_concept_.is_initialized(),
    //     .has_room_polygon = room_initialized_from_svg_polygon_,
    //     .room_width       = have_loc ? loc_res->state[0] : 0.f,
    //     .room_length      = have_loc ? loc_res->state[1] : 0.f,
    //     .loc_pose         = loc_pose,
    //     .use_loc_pose     = use_loc,
    // });

    // if (have_loc && !loc_res->corner_matches.empty())
    //     viewer_2d_->draw_corners(loc_res->corner_matches, pose_for_draw);
    // else
    //     viewer_2d_->draw_corners({}, pose_for_draw);

    // ── DSR graph update ───────────────────────────────────────────────────
    // if (have_loc)
    // {
    //     const float sdf_mse = loc_res->sdf_mse;
    //     const float cov_tt  = (loc_res->covariance.rows() > 2 && loc_res->covariance.cols() > 2)
    //                           ? loc_res->covariance(2, 2) : 1.f;
    //     const bool stable   = (loc_res->iterations_used == 0)  // prediction early exit → pose is stable
    //                           && sdf_mse < params.STABLE_SDF_MSE_MAX
    //                           && cov_tt  < params.STABLE_COV_TT_MAX;

    //     if (!room_node_created_)
    //     {
    //         stable_frames_ = stable ? stable_frames_ + 1 : 0;
    //         if (stable_frames_ >= params.STABLE_FRAMES_REQUIRED)
    //             dsr_create_room_and_reparent(*loc_res);
    //         else
    //             dsr_update_pose(*loc_res);   // world→robot RT
    //     }
    //     else
    //     {
    //         dsr_update_pose(*loc_res);       // room→robot RT
    //     }
    // }

    //update_ui(loc_res, pose_for_draw);
    fps_counter_.print("[Compute]", 2000);
}

///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::dsr_init_graph()
{
    if (!G) { qWarning() << "dsr_init_graph: DSR graph not available"; return; }
    const uint32_t agent_id = G->get_agent_id();

    // Create or retrieve 'world' root node
    if (auto wopt = G->get_node("world"); wopt.has_value())
    {
        dsr_world_id_ = wopt->id();
    }
    else
    {
        DSR::Node world_node;
        world_node.name("world");
        world_node.type("world");
        world_node.agent_id(agent_id);
        G->add_or_modify_attrib_local<level_att>(world_node, 0);
        G->add_or_modify_attrib_local<pos_x_att>(world_node, 0.f);
        G->add_or_modify_attrib_local<pos_y_att>(world_node, 0.f);
        auto id = G->insert_node(world_node);
        if (!id.has_value()) { qWarning() << "dsr_init_graph: failed to insert world node"; return; }
        dsr_world_id_ = id.value();
        qInfo() << "DSR: created world node id=" << dsr_world_id_;
    }

    // Create or retrieve 'robot' node
    if (auto ropt = G->get_node("robot"); ropt.has_value())
    {
        dsr_robot_id_ = ropt->id();
    }
    else
    {
        DSR::Node robot_node;
        robot_node.name("robot");
        robot_node.type("robot");
        robot_node.agent_id(agent_id);
        G->add_or_modify_attrib_local<level_att>(robot_node, 1);
        G->add_or_modify_attrib_local<pos_x_att>(robot_node, 100.f);
        G->add_or_modify_attrib_local<pos_y_att>(robot_node, 0.f);
        auto id = G->insert_node(robot_node);
        if (!id.has_value()) { qWarning() << "dsr_init_graph: failed to insert robot node"; return; }
        dsr_robot_id_ = id.value();
        qInfo() << "DSR: created robot node id=" << dsr_robot_id_;
    }

    // Create identity RT edge world→robot
    if (dsr_world_id_ && dsr_robot_id_)
    {
        auto rt    = G->get_rt_api();
        auto wopt  = G->get_node(dsr_world_id_);
        if (wopt.has_value())
            rt->insert_or_assign_edge_RT(*wopt, dsr_robot_id_, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f});
    }
}

///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::dsr_update_pose(const rc::RoomConcept::UpdateResult& res)
{
    if (!G) return;

    const float x     = res.robot_pose.translation().x();
    const float y     = res.robot_pose.translation().y();
    const float theta = std::atan2(res.robot_pose.linear()(1, 0), res.robot_pose.linear()(0, 0));

    const uint64_t parent_id = room_node_created_ ? dsr_room_id_ : dsr_world_id_;
    auto parent_opt = G->get_node(parent_id);
    if (!parent_opt.has_value()) return;

    auto rt = G->get_rt_api();
    rt->insert_or_assign_edge_RT(*parent_opt, dsr_robot_id_,
                                 {x * 1000.f, y * 1000.f, 0.f},   // DSR uses mm
                                 {0.f, 0.f, theta});

    // Store localization covariance on the RT edge as custom attributes
    const float cov_xx = (res.covariance.rows() > 0 && res.covariance.cols() > 0)
                         ? res.covariance(0, 0) : 0.f;
    const float cov_tt = (res.covariance.rows() > 2 && res.covariance.cols() > 2)
                         ? res.covariance(2, 2) : 0.f;

    auto edge_opt = DSR::RT_API::get_edge_RT(*parent_opt, dsr_robot_id_);
    if (edge_opt.has_value())
    {
        G->runtime_checked_add_or_modify_attrib_local(*edge_opt, "loc_cov_xx", cov_xx);
        G->runtime_checked_add_or_modify_attrib_local(*edge_opt, "loc_cov_tt", cov_tt);
        G->insert_or_assign_edge(*edge_opt);
    }
}

///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::dsr_create_room_and_reparent(const rc::RoomConcept::UpdateResult& res)
{
    if (!G) return;
    const uint32_t agent_id = G->get_agent_id();

    // Create room node
    DSR::Node room_node;
    room_node.name("room");
    room_node.type("room");
    room_node.agent_id(agent_id);
    G->add_or_modify_attrib_local<level_att>(room_node, 1);
    G->add_or_modify_attrib_local<pos_x_att>(room_node, -200.f);
    G->add_or_modify_attrib_local<pos_y_att>(room_node, 0.f);
    auto room_id_opt = G->insert_node(room_node);
    if (!room_id_opt.has_value()) { qWarning() << "DSR: failed to insert room node"; return; }
    dsr_room_id_ = room_id_opt.value();

    // world→room identity RT
    auto wopt = G->get_node(dsr_world_id_);
    if (wopt.has_value())
    {
        auto rt = G->get_rt_api();
        rt->insert_or_assign_edge_RT(*wopt, dsr_room_id_, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f});
    }

    // Delete world→robot, then create room→robot
    G->delete_edge(dsr_world_id_, dsr_robot_id_, "RT");
    room_node_created_ = true;

    // Update robot node level
    if (auto ropt = G->get_node(dsr_robot_id_); ropt.has_value())
    {
        G->add_or_modify_attrib_local<level_att>(*ropt, 2);
        G->update_node(*ropt);
    }

    dsr_update_pose(res);
    qInfo() << "DSR: stabilized — room node created, robot re-parented under room.";
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
        const std::string key = (params.OptimizerType == "LBFGS") ? "fe_lbfgs" : "fe_adam";
        ts_plot_fe_->add_point(key, loc_res->final_loss);
        ts_plot_fe_->add_point("fe_pred", loc_res->final_loss);
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
    odom.adv          = add_noise(-pose.adv);
    odom.side         = add_noise(pose.side);
    odom.rot          = add_noise(pose.rot);
    odom.source_ts_ms = pose.timestamp;
    odom.recv_ts_ms   = static_cast<std::int64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    odom.timestamp    = std::chrono::high_resolution_clock::time_point(
        std::chrono::milliseconds(pose.timestamp));
    odometry_buffer_.put<0>(std::move(odom), static_cast<std::uint64_t>(odom.recv_ts_ms));
}
