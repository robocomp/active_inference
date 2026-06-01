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

#include "component_logging.h"
#include <algorithm>
#include <print>
#include <random>
#include <stdexcept>
#include <fstream>
#include <unordered_set>
#include <QDir>
#include <QFileInfo>
#include <QVBoxLayout>

#include <variant>

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
        const int period = configLoader.get<int>("Period.Compute");

        states["Waiting"] = std::make_unique<GRAFCETStep>("Waiting", period,
            std::bind(&SpecificWorker::waiting_loop, this),
            std::bind(&SpecificWorker::waiting_enter, this));
        states["Operating"] = std::make_unique<GRAFCETStep>("Operating", period,
            std::bind(&SpecificWorker::operating_loop, this),
            std::bind(&SpecificWorker::operating_enter, this));
        states["Degraded"] = std::make_unique<GRAFCETStep>("Degraded", period,
            std::bind(&SpecificWorker::degraded_loop, this),
            std::bind(&SpecificWorker::degraded_enter, this));

        states["Compute"]->addTransition(states["Compute"].get(), SIGNAL(entered()), states["Waiting"].get());
        states["Waiting"]->addTransition(this, SIGNAL(presenceReady()), states["Operating"].get());
        states["Operating"]->addTransition(this, SIGNAL(presenceLost()), states["Degraded"].get());
        states["Degraded"]->addTransition(states["Degraded"].get(), SIGNAL(entered()), states["Waiting"].get());

        statemachine.addState(states["Waiting"].get());
        statemachine.addState(states["Operating"].get());
        statemachine.addState(states["Degraded"].get());

        statemachine.setChildMode(QState::ExclusiveStates);
        statemachine.start();
        auto error = statemachine.errorString();
        if (error.length() > 0) { qWarning() << error; throw error; }
    }
}

///////////////////////////////////////////////////////////////////////////////
SpecificWorker::~SpecificWorker()
{
    save_window_settings();
    save_robot_pose_once();
    room_concept_.stop();
}

///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::initialize()
{
    GenericWorker::initialize();

    QObject::connect(QCoreApplication::instance(), &QCoreApplication::aboutToQuit,
                     this, &SpecificWorker::save_robot_pose_once, Qt::UniqueConnection);
    QObject::connect(QCoreApplication::instance(), &QCoreApplication::aboutToQuit,
                     this, &SpecificWorker::cleanup_owned_nodes, Qt::UniqueConnection);

    // ── RoomConcept params ─────────────────────────────────────────────────
    rc::ConfigLoaderUtils::load_required<bool>(configLoader, "RoomConcept.PredictionEarlyExit", params.PREDICTION_EARLY_EXIT);
    rc::ConfigLoaderUtils::load_required<int>(configLoader, "RoomConcept.NumIterations", room_concept_.params.num_iterations);
    rc::ConfigLoaderUtils::load_required<int>(configLoader, "RoomConcept.WindowSize", room_concept_.params.rfe_window_size);
    rc::ConfigLoaderUtils::load_required<int>(configLoader, "RoomConcept.MaxLidarPoints", room_concept_.params.max_lidar_points);
    rc::ConfigLoaderUtils::load_required<int>(configLoader, "RoomConcept.MaxLidarOldSlot", room_concept_.params.rfe_max_lidar_per_old_slot);
    rc::ConfigLoaderUtils::load_required<float, double>(configLoader, "RoomConcept.RecoveryLossThreshold", room_concept_.params.recovery_loss_threshold);
    rc::ConfigLoaderUtils::load_required<int>(configLoader, "RoomConcept.RecoveryConsecutiveCount", room_concept_.params.recovery_consecutive_count);

    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.OdometryNoiseFactor", params.ODOMETRY_NOISE_FACTOR);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.OdomNoiseScale", room_concept_.params.odom_noise_scale);
    rc::ConfigLoaderUtils::load_optional<bool>(configLoader, "RoomConcept.DifferentialTest", room_concept_.params.differential_test_enabled);
    rc::ConfigLoaderUtils::load_optional<bool>(configLoader, "RoomConcept.SdfCurrentSlotOnly", room_concept_.params.sdf_current_slot_only);
    rc::ConfigLoaderUtils::load_optional_apply<std::string>(configLoader, "RoomConcept.OptimizerType", [&](const std::string& optimizer_type)
    {
        params.OptimizerType = optimizer_type;
        room_concept_.params.optimizer_type = optimizer_type;
    });

    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.SigmaSdf", room_concept_.params.sigma_sdf);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.PredictionTrustFactor", room_concept_.params.prediction_trust_factor);
    rc::ConfigLoaderUtils::load_optional<int>(configLoader, "RoomConcept.MinTrackingSteps", room_concept_.params.min_tracking_steps);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.RotationSdfCoupling", room_concept_.params.rotation_sdf_coupling);

    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.LbfgsLr", room_concept_.params.lbfgs_lr);
    rc::ConfigLoaderUtils::load_optional<int>(configLoader, "RoomConcept.LbfgsHistorySize", room_concept_.params.lbfgs_history_size);
    rc::ConfigLoaderUtils::load_optional<double>(configLoader, "RoomConcept.LbfgsToleranceGrad", room_concept_.params.lbfgs_tolerance_grad);
    rc::ConfigLoaderUtils::load_optional<double>(configLoader, "RoomConcept.LbfgsToleranceChange", room_concept_.params.lbfgs_tolerance_change);

    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.LearningRatePos", room_concept_.params.learning_rate_pos);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.ObsSigma", room_concept_.params.rfe_obs_sigma);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.HuberDelta", room_concept_.params.rfe_huber_delta);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.ConvergenceRelTol", room_concept_.params.convergence_relative_tol);
    rc::ConfigLoaderUtils::load_optional<int>(configLoader, "RoomConcept.ConvergenceMinIters", room_concept_.params.convergence_min_iters);

    rc::ConfigLoaderUtils::load_optional<bool>(configLoader, "RoomConcept.BoundaryQualityGate", room_concept_.params.rfe_boundary_quality_gate);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.BoundaryHessianQualityThreshold", room_concept_.params.boundary_hessian_quality_threshold);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.BoundaryMuQualityThreshold", room_concept_.params.boundary_mu_quality_threshold);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.EigenvalueClampBoundaryMax", room_concept_.params.eigenvalue_clamp_boundary_max);

    rc::ConfigLoaderUtils::load_optional<int>(configLoader, "RoomConcept.RecoveryCooldownFrames", room_concept_.params.recovery_cooldown_frames);

    rc::ConfigLoaderUtils::load_optional<bool>(configLoader, "RoomConcept.VelocityAdaptiveWeights", room_concept_.params.velocity_adaptive_weights);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.LinearVelocityThreshold", room_concept_.params.linear_velocity_threshold);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.AngularVelocityThreshold", room_concept_.params.angular_velocity_threshold);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.WeightBoostFactor", room_concept_.params.weight_boost_factor);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.WeightReductionFactor", room_concept_.params.weight_reduction_factor);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.WeightSmoothingAlpha", room_concept_.params.weight_smoothing_alpha);

    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.CmdNoiseTrans", room_concept_.params.cmd_noise_trans);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.CmdNoiseRot", room_concept_.params.cmd_noise_rot);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.CmdNoiseBase", room_concept_.params.cmd_noise_base);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.OdomNoiseTrans", room_concept_.params.odom_noise_trans);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.OdomNoiseRot", room_concept_.params.odom_noise_rot);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.OdomNoiseBase", room_concept_.params.odom_noise_base);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.EncoderRotSlipK", room_concept_.params.encoder_rot_slip_k);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.StationaryMotionThreshold", room_concept_.params.stationary_motion_threshold);

    rc::ConfigLoaderUtils::load_optional<bool>(configLoader, "RoomConcept.LearnMotionModel", room_concept_.params.learn_motion_model);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.MotionLearnAlpha", room_concept_.params.motion_learn_alpha);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.MotionLearnBeta", room_concept_.params.motion_learn_beta);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.MotionLearnMinOmega", room_concept_.params.motion_learn_min_omega);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.MotionLearnMinTrans", room_concept_.params.motion_learn_min_trans);
    rc::ConfigLoaderUtils::load_optional<int>(configLoader, "RoomConcept.MotionLearnMinFrames", room_concept_.params.motion_learn_min_frames);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.MotionLearnQualityThreshold", room_concept_.params.motion_learn_quality_threshold);

    rc::ConfigLoaderUtils::load_optional<bool>(configLoader, "RoomConcept.EnableCornerTracking", room_concept_.params.enable_corner_tracking);
    rc::ConfigLoaderUtils::load_optional<bool>(configLoader, "RoomConcept.FarPointsWeight", room_concept_.params.far_points_weight);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.FarPointsExponent", room_concept_.params.far_points_exponent);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.FarPointsMinWeight", room_concept_.params.far_points_min_weight);
    rc::ConfigLoaderUtils::load_optional<bool>(configLoader, "RoomConcept.IncidenceAngleWeight", room_concept_.params.incidence_angle_weight);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.IncidenceAngleExponent", room_concept_.params.incidence_angle_exponent);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "RoomConcept.IncidenceAngleMinWeight", room_concept_.params.incidence_angle_min_weight);
    rc::ConfigLoaderUtils::load_optional<bool>(configLoader, "RoomConcept.UseCuda", room_concept_.params.use_cuda);
    rc::ConfigLoaderUtils::load_optional<bool>(configLoader, "RoomConcept.DebugLog", room_concept_.params.debug_log_enabled);

    rc::ConfigLoaderUtils::load_optional<bool>(configLoader, "RoomConcept.RerunEnabled", room_concept_.params.rerun_enabled);
    rc::ConfigLoaderUtils::load_optional<std::string>(configLoader, "RoomConcept.RerunHost", room_concept_.params.rerun_host);
    rc::ConfigLoaderUtils::load_optional<int>(configLoader, "RoomConcept.RerunPort", room_concept_.params.rerun_port);
    rc::ConfigLoaderUtils::load_optional<int>(configLoader, "RoomConcept.RerunSdfEveryN", room_concept_.params.rerun_sdf_every_n);
    rc::ConfigLoaderUtils::load_optional<int>(configLoader, "RoomConcept.RerunSdfResolution", room_concept_.params.rerun_sdf_resolution);
    rc::ConfigLoaderUtils::load_optional<int>(configLoader, "RoomConcept.RerunMaxQueue", room_concept_.params.rerun_max_queue);

    // ── DSR stabilization thresholds ──────────────────────────────────────
    rc::ConfigLoaderUtils::load_optional<int>(configLoader, "DSR.StableFramesRequired", params.STABLE_FRAMES_REQUIRED);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "DSR.StableSdfMseMax", params.STABLE_SDF_MSE_MAX);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "DSR.StableCovTtMax", params.STABLE_COV_TT_MAX);
    rc::ConfigLoaderUtils::load_optional<bool>(configLoader, "DSR.BootstrapTableEnabled", params.BOOTSTRAP_TABLE_ENABLED);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "DSR.BootstrapTableX", params.BOOTSTRAP_TABLE_X);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "DSR.BootstrapTableY", params.BOOTSTRAP_TABLE_Y);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "DSR.BootstrapTableYaw", params.BOOTSTRAP_TABLE_YAW);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "DSR.BootstrapTableWidth", params.BOOTSTRAP_TABLE_WIDTH);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "DSR.BootstrapTableDepth", params.BOOTSTRAP_TABLE_DEPTH);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "DSR.BootstrapTableHeight", params.BOOTSTRAP_TABLE_HEIGHT);

    // ── EpistemicController params ─────────────────────────────────────────
    auto& ec = epistemic_controller_.params;
    auto& ep = epistemic_controller_.epistemic_planner().params;
    rc::ConfigLoaderUtils::load_optional<int>(configLoader, "EpistemicController.NumArcCurvatures", ec.num_arc_curvatures);
    rc::ConfigLoaderUtils::load_optional<int>(configLoader, "EpistemicController.HorizonSteps", ec.horizon_steps);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.Dt", ec.dt);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.MaxAdvSpeed", ec.max_adv_speed);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.MaxRotSpeed", ec.max_rot_speed);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.WEpistemic", ec.w_epistemic);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.WPragmatic", ec.w_pragmatic);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.WHeading", ec.w_heading);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.WBoundary", ec.w_boundary);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.KRot", ec.k_rot);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.GaussianSigma", ec.gaussian_sigma);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.SpeedHorizonS", ec.speed_horizon_s);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.ObstacleRadius", ec.obstacle_radius);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.ObstacleK", ec.obstacle_k);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.ObstacleStepCap", ec.obstacle_step_cap);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.WObstacle", ec.w_obstacle);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.WallFilterMargin", ec.wall_filter_margin);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.BandwidthCoupling", ec.bandwidth_coupling);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.SdfSafe", ec.sdf_safe);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.SdfDanger", ec.sdf_danger);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.GovernorAlphaMin", ec.governor_alpha_min);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.FimCornerSigma", ec.fim_corner_sigma);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.FimMaxRange", ec.fim_max_range);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.GridResolution", ep.grid_resolution);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.MinDistance", ep.min_distance);
    rc::ConfigLoaderUtils::load_optional<int>(configLoader, "EpistemicController.MaxCandidates", ep.max_candidates);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.TargetWallMargin", ep.target_wall_margin);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.AngularDominanceRatio", ep.angular_dominance_ratio);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.WExploration", ep.w_exploration);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.IorCellSize", ep.ior_cell_size);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.IorDecayTime", ep.ior_decay_time);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.WIor", ep.w_ior);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.WPathInterest", ep.w_path_interest);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.FimCornerSigma", ep.fim_corner_sigma);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.FimMaxRange", ep.fim_max_range);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.ArrivalDistance", ep.arrival_distance);
    rc::ConfigLoaderUtils::load_optional<float, double>(configLoader, "EpistemicController.DwellTime", ep.dwell_time);
    epistemic_controller_.set_robot_footprint(params.ROBOT_WIDTH, params.ROBOT_LENGTH);
    
    // ── Wire RoomConcept run context ───────────────────────────────────────
    rc::RoomConcept::RunContext run_ctx;
    run_ctx.high_lidar_buffer = &high_lidar_buffer_;
    run_ctx.velocity_buffer = &velocity_buffer_;
    run_ctx.odometry_buffer = &odometry_buffer_;
    room_concept_.set_run_context(run_ctx);
    room_concept_.params.prediction_early_exit = params.PREDICTION_EARLY_EXIT;

    initialize_room_model_from_svg();
    const std::string pose_path = pose_file_path();
    room_concept_.set_seed_pose_file(pose_path);

    auto default_viewer = find_graph_viewer("");
    if (!default_viewer)
        throw std::runtime_error("SpecificWorker requires a default DSR viewer. Enable at least one Agent viewer flag for the default graph.");

    default_viewer->add_custom_widget_to_dock("layout", &custom_widget);
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
    connect(custom_widget.btn_lidar_points_viz, &QPushButton::toggled, this, &SpecificWorker::slot_toggle_lidar_points_display);
    viewer_2d_->set_lidar_points_visible(custom_widget.btn_lidar_points_viz->isChecked());

    // ── DSR: resolve existing graph node IDs ──────────────────────────────
    check_init_graph_is_valid();

    // Ensure a clean startup: if a stale room node exists from previous runs,
    // remove it so the room is recreated only after localization is stable.
    cleanup_room_graph_nodes();

    // RT_API
    rt_api = G->get_rt_api();

    // ── Connect DSR signals ────────────────────────────────────────────────
    connect(G.get(), &DSR::DSRGraph::update_node_signal,      this, &SpecificWorker::modify_node_slot);
    // connect(G.get(), &DSR::DSRGraph::update_edge_signal,      this, &SpecificWorker::modify_edge_slot);
    connect(G.get(), &DSR::DSRGraph::update_node_attr_signal, this, &SpecificWorker::modify_node_attrs_slot);
    // connect(G.get(), &DSR::DSRGraph::update_edge_attr_signal, this, &SpecificWorker::modify_edge_attrs_slot);
    // connect(G.get(), &DSR::DSRGraph::del_edge_signal,         this, &SpecificWorker::del_edge_slot);
    // connect(G.get(), &DSR::DSRGraph::del_node_signal,         this, &SpecificWorker::del_node_slot);

    room_concept_.start();

    // ── Presence coordinator ────────────────────────────────────────────────
    presence_coordinator_.configure(configLoader, G, static_cast<std::uint32_t>(agent_id));
    presence_coordinator_.set_transition_hooks({
        .request_presence_ready = [this]() { emit presenceReady(); },
        .request_presence_lost  = [this]() { emit presenceLost(); },
    });
    presence_coordinator_.set_peer_hooks({
        .on_peer_restarted = [](std::uint32_t id)
        {
            qInfo() << "[Presence] peer" << id << "restarted";
        },
        .on_optional_peer_lost = [this](const std::string &name, std::uint32_t id)
        {
            on_optional_peer_lost(name, id);
        },
        .on_optional_peer_ready = [this](const std::string &name, std::uint32_t id)
        {
            on_optional_peer_ready(name, id);
        },
    });
    presence_coordinator_.set_lifecycle_hooks({
        .on_waiting_enter = [this]()
        {
            qInfo() << "[SM] -> Waiting";
            const auto missing = presence_coordinator_.missing_required_names();
            if (!missing.empty())
            {
                QString m;
                for (const auto &label : missing)
                    m += " " + QString::fromStdString(label);
                qInfo() << "  missing:" << m;
            }
        },
        .on_operating_enter = [this]()
        {
            qInfo() << "[SM] -> Operating: all required constraints satisfied";
            room_concept_.stop();
            room_concept_.start();
        },
        .on_operating_loop = [this]()
        {
            compute();
            if (auto v = find_graph_viewer(""); v)
                v->set_external_fps(states.at("Operating")->getActualFps());
        },
        .on_degraded_enter = [this]()
        {
            qInfo() << "[SM] -> Degraded: required peer lost. Cleaning up and exiting.";
            room_concept_.stop();
            cleanup_owned_nodes();
            QTimer::singleShot(500, QCoreApplication::instance(), SLOT(quit()));
        },
    });
    presence_coordinator_.start();

    // ── Wire mouse-driven pose reset ───────────────────────────────────────
    connect(viewer_2d_.get(), &rc::Viewer2D::robot_moved,
            this, [this](QPointF p){ slot_mouse_translate(p); });
    connect(viewer_2d_.get(), &rc::Viewer2D::robot_rotate,
            this, [this](QPointF p){ slot_mouse_rotate(p); });

        restore_window_settings();
}

///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::compute()
{
    affordance_manager_.monitor_execution(G);

    const auto loc_res  = room_concept_.get_last_result();
    const bool have_loc = loc_res.has_value() && loc_res->ok;

    const Eigen::Affine2f pose_for_draw = best_available_pose(loc_res, have_loc);

    // ── Update 2-D viewer ─────────────────────────────────────────────────
    const Eigen::Affine2f loc_pose = have_loc ? loc_res->robot_pose : pose_for_draw;
    const bool use_loc = have_loc && !loc_res->lidar_scan.empty();

    std::vector<Eigen::Vector3f> lidar_for_canvas;
    if (use_loc)
        lidar_for_canvas = loc_res->lidar_scan;
    else
    {
        const auto& [lidar_from_buffer] = high_lidar_buffer_.read_last();
        if (lidar_from_buffer.has_value())
            lidar_for_canvas = lidar_from_buffer->first;
    }

    viewer_2d_->update_frame({
        .lidar_points     = lidar_for_canvas,
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

    update_epistemic_overlay();

    if (have_loc && !loc_res->corner_matches.empty())
        viewer_2d_->draw_corners(loc_res->corner_matches, pose_for_draw);
    else
        viewer_2d_->draw_corners({}, pose_for_draw);

    // ── DSR graph update (only on fresh localization frames) ──────────────
    if (have_loc && loc_res->timestamp_ms > 0 && loc_res->timestamp_ms != last_dsr_published_ts_ms_)
    {
        update_dsr(*loc_res);
        last_dsr_published_ts_ms_ = loc_res->timestamp_ms;
    }

    update_ui(loc_res);
    fps_counter_.print("[Compute]", 3000);
}

///////////////////////////////////////////////////////////////////////////////
std::optional<rc::LidarData> SpecificWorker::read_lidar_from_graph() const
{
    // Lidar comes in meters
    if (!G)
        return std::nullopt;

    std::optional<DSR::Node> lidar_node = G->get_node("lidar3d");
    if (!lidar_node.has_value())
        lidar_node = G->get_node("lidar3D");
    if (!lidar_node.has_value() && !params.LIDAR_NAME.empty())
        lidar_node = G->get_node(params.LIDAR_NAME);
    if (!lidar_node.has_value())
    {
        const auto laser_nodes = G->get_nodes_by_type("laser");
        if (!laser_nodes.empty())
            lidar_node = laser_nodes.front();
    }

    if (!lidar_node.has_value())
    {
        std::print("[RoomConcept] Lidar node not found. Tried: 'lidar3d', 'lidar3D', '{}', type 'laser'\n",
                       params.LIDAR_NAME);
        return std::nullopt;
    }
 
    const auto lx = G->get_attrib_by_name<laser_X_att>(lidar_node.value());
    const auto ly = G->get_attrib_by_name<laser_Y_att>(lidar_node.value());
    const auto lz = G->get_attrib_by_name<laser_Z_att>(lidar_node.value());
    const auto laser_ts = G->get_attrib_by_name<laser_timestamp_att>(lidar_node.value());
    if (!lx.has_value() || !ly.has_value() || !lz.has_value())
    {
        std::print("[RoomConcept] Missing laser_X/Y/Z attributes on node '{}'\n", lidar_node->name());
        return std::nullopt;
    }
    
    const auto &xs = lx.value().get();
    const auto &ys = ly.value().get();
    const auto &zs = lz.value().get();
    const std::size_t npts = std::min({xs.size(), ys.size(), zs.size()});

    // Points are in lidar3D sensor frame; bring them into robot frame using
    // the RT edge robot→lidar3D (T_rl: p_robot = T_rl * p_lidar).
    std::optional<Eigen::Affine3d> T_rl;
    if (rt_api)
    {
        auto pose = rt_api->get_RT_pose_from_parent(lidar_node.value());
        if (pose.has_value() && !pose->isApprox(Eigen::Affine3d::Identity()))
            T_rl = pose.value();
    }

    const float to_m = 1.f;

    std::vector<Eigen::Vector3f> points_high;
    points_high.reserve(npts);
    const float min_h_m = params.LIDAR_HIGH_MIN_HEIGHT;
    int infinite_count = 0;

    for (std::size_t i = 0; i < npts; ++i)
    {
        const float x_raw = xs[i];
        const float y_raw = ys[i];
        const float z_raw = zs[i];
        if (!std::isfinite(x_raw) || !std::isfinite(y_raw) || !std::isfinite(z_raw))
        {
            infinite_count++;
            continue;
        }

        Eigen::Vector3f p(x_raw * to_m, y_raw * to_m, z_raw * to_m);
        if (T_rl.has_value())
            p = (T_rl.value() * p.cast<double>()).cast<float>();

        if (p.z() > min_h_m)
            points_high.emplace_back(p);
    }

    const auto now_ts = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    std::int64_t source_ts = static_cast<std::int64_t>(laser_ts.value_or(static_cast<uint64_t>(now_ts)));

    // Keep RoomConcept frame gate moving even if graph timestamp is stale/repeated.
    static std::int64_t last_source_ts = -1;
    if (source_ts <= last_source_ts)
        source_ts = std::max(last_source_ts + 1, now_ts);
    if (source_ts <= last_source_ts)
        source_ts = last_source_ts + 1;
    last_source_ts = source_ts;

    return rc::LidarData{std::move(points_high), source_ts};
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
        dsr_update_affordance(res); // publish epistemic target affordance
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

    const float x     = room_node_created_ ? t_robot_to_room.x() : t.x();
    const float y     = room_node_created_ ? t_robot_to_room.y() : t.y();
    const float theta = room_node_created_ ? theta_robot_to_room : theta_room_to_robot;

    // ── Covariance (SE2 3×3 packed into 6×6 flat row-major) ───────────────
    Eigen::Matrix3f cov_se2 = Eigen::Matrix3f::Identity();
    if (res.covariance.rows() >= 3 && res.covariance.cols() >= 3)
        cov_se2 = res.covariance.topLeftCorner<3, 3>();

    if (room_node_created_)
    {
        const float c = std::cos(theta_room_to_robot);
        const float s = std::sin(theta_room_to_robot);
        Eigen::Matrix3f J = Eigen::Matrix3f::Zero();
        J(0, 0) = -c;  J(0, 1) = -s;  J(0, 2) =  s * t.x() - c * t.y();
        J(1, 0) =  s;  J(1, 1) = -c;  J(1, 2) =  c * t.x() + s * t.y();
        J(2, 2) = -1.f;
        cov_se2 = J * cov_se2 * J.transpose();
    }

    std::vector<float> cov_flat(36, 0.f);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            cov_flat[r * 6 + c] = cov_se2(r, c);

    // ── All attributes written in one shot via normal API ───────────────────
    auto edge_rt = G->get_edge(parent_id, child_id, "RT");
    if (!edge_rt.has_value())
    {
        rt_api->insert_or_assign_edge_RT(parent_opt.value(), child_id,
                                         {x, y, 0.f},
                                         {0.f, 0.f, theta});
        edge_rt = G->get_edge(parent_id, child_id, "RT");
        if (!edge_rt.has_value())
        {
            qWarning() << "dsr_update_pose: failed to create RT edge"
                       << "parent_id=" << parent_id
                       << "child_id=" << child_id;
            return;
        }
    }

    G->add_or_modify_attrib_local<rt_translation_att>(
        edge_rt.value(), std::vector<float>{x, y, 0.f});
    G->add_or_modify_attrib_local<rt_rotation_euler_xyz_att>(
        edge_rt.value(), std::vector<float>{0.f, 0.f, theta});
    G->add_or_modify_attrib_local<rt_covariance_att>(edge_rt.value(), cov_flat);
    G->add_or_modify_attrib_local<rt_translation_velocity_att>(
        edge_rt.value(), std::vector<float>{last_robot_adv_speed_, last_robot_side_speed_, 0.f});
    G->add_or_modify_attrib_local<rt_rotation_euler_xyz_velocity_att>(
        edge_rt.value(), std::vector<float>{0.f, 0.f, last_robot_rot_speed_});
    G->insert_or_assign_edge(edge_rt.value());
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
        dsr_update_pose(res);
        return;
    }

    DSR::Node room_node = DSR::Node::create<room_node_type>("room");
    room_node.attrs()[delimiting_polygon_x_str.data()] = DSR::Attribute{polygon_x, 0, 0};
    room_node.attrs()[delimiting_polygon_y_str.data()] = DSR::Attribute{polygon_y, 0, 0};
    room_node.attrs()[room_height_str.data()] = DSR::Attribute{params.room_height, 0, 0};
    // TODO: change to add_or_modify_attrib_local once available

    const auto room_id_opt = G->insert_node(room_node);
    if (!room_id_opt.has_value())
    {
        qWarning() << "DSR: failed to create room node";
        return;
    }

    dsr_room_id_ = room_id_opt.value();
    room_node_created_ = true;
    stable_frames_ = 0;
    trigger_graph_layout_twopi();

    dsr_update_pose(res);
    dsr_create_wall_nodes();

    // Seed the epistemic planner with room geometry so it can generate candidates.
    // room_polygon is already computed at the top of this function.
    if (!room_polygon.empty())
    {
        Eigen::Vector2f pmin = room_polygon.front();
        Eigen::Vector2f pmax = room_polygon.front();
        for (const auto& v : room_polygon)
        {
            pmin = pmin.cwiseMin(v);
            pmax = pmax.cwiseMax(v);
        }
        epistemic_controller_.set_room_bounds(pmin, pmax);
        epistemic_controller_.set_room_polygon(room_polygon);
    }
}

///////////////////////////////////////////////////////////////////////////////
/// Publish the epistemic planner's best navigation target as an "affordance"
/// node hanging from the room node.  The node is created on the first call and
/// its attributes + RT edge are refreshed every DSR update cycle.
///
/// Attributes written to the node:
///   epistemic_target_x_m     — target X in room frame (m)
///   epistemic_target_y_m     — target Y in room frame (m)
///   epistemic_target_yaw_rad — desired robot heading at that point (rad)
///   epistemic_gain           — planner score (FIM × IoR × distance)
///   epistemic_pending        — true: target not yet reached by any agent
///   active                   — false when published by this agent; sibling
///                              controller sets it true while executing it
///
/// The room→affordance relation is expressed with an edge of type "has_intention".
void SpecificWorker::dsr_update_affordance(const rc::RoomConcept::UpdateResult& res)
{
    if (!G || !room_node_created_) return;

    auto& planner = epistemic_controller_.epistemic_planner();

    // Always update robot state so mark_and_refresh uses the correct position.
    epistemic_controller_.set_robot_state(res.robot_pose, res.covariance);

    if (affordance_manager_.consume_completion_event())
    {
        planner.clear_target();
        planner.mark_and_refresh();   // keep path trail live in viewer
        return;
    }

    if (affordance_manager_.is_executing(G))
    {
        planner.mark_and_refresh();   // stamp path + refresh IoR overlay during navigation
        return;
    }

    // Refresh obstacle exclusion zones from DSR graph before selecting the target.
    update_planner_obstacle_footprints();

    // Ask the planner for the current best target (handles dwell / arrival internally)
    const auto target_opt = planner.update_target();
    if (!target_opt.has_value()) return;

    const float tx   = target_opt->position.x();
    const float ty   = target_opt->position.y();
    const float gain = target_opt->score;

    // Heading: face toward room centre so the robot maximises wall/corner visibility
    const float cx  = (planner.room_min().x() + planner.room_max().x()) * 0.5f;
    const float cy  = (planner.room_min().y() + planner.room_max().y()) * 0.5f;
    const float yaw = std::atan2(cy - ty, cx - tx);

    affordance_manager_.publish_target(
        G,
        dsr_room_id_,
        tx,
        ty,
        yaw,
        gain,
        [this]() { trigger_graph_layout_twopi(); },
        [this]() { trigger_graph_layout_twopi(); });
}

///////////////////////////////////////////////////////////////////////////////
/// Query the DSR graph for all "object" and "obstacle" type nodes that are
/// direct children of the room node, read their RT pose (position + yaw in
/// room frame) and width_m / depth_m attributes, and pass them to the
/// epistemic planner so that candidate targets that fall inside or too close
/// to any such footprint are excluded from target selection.
void SpecificWorker::update_planner_obstacle_footprints()
{
    if (!G || !rt_api || !room_node_created_) return;

    std::vector<rc::EpistemicPlanner::ObstacleFootprint> footprints;

    auto collect = [&](const std::string& node_type)
    {
        for (const auto& node : G->get_nodes_by_type(node_type))
        {
            const auto w_opt = G->get_attrib_by_name<width_m_att>(node);
            const auto d_opt = G->get_attrib_by_name<depth_m_att>(node);
            if (!w_opt.has_value() || !d_opt.has_value()) continue;
            const float half_w = w_opt.value() * 0.5f;
            const float half_d = d_opt.value() * 0.5f;
            if (half_w <= 0.f || half_d <= 0.f) continue;

            const auto rt_opt = rt_api->get_RT_pose_from_parent(node);
            if (!rt_opt.has_value()) continue;

            const Eigen::Vector3d t = rt_opt->translation();
            const float yaw = static_cast<float>(
                std::atan2(rt_opt->linear()(1, 0), rt_opt->linear()(0, 0)));

            footprints.push_back({
                .center = {static_cast<float>(t.x()), static_cast<float>(t.y())},
                .half_w = half_w,
                .half_d = half_d,
                .yaw    = yaw
            });
        }
    };

    collect("object");
    collect("obstacle");

    epistemic_controller_.epistemic_planner().set_obstacle_footprints(std::move(footprints));
}

///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::load_robot_body_dimensions_from_graph()
{
    if (!G)
        return;

    std::optional<DSR::Node> body_node = std::nullopt;

    const auto body_nodes = G->get_nodes_by_type("body");
    if (!body_nodes.empty())
        body_node = body_nodes.front();
    else
        body_node = G->get_node("body");

    if (!body_node.has_value())
    {
        qCWarning(logGraph) << "dsr_init_graph: no 'body' node found; keeping default robot dimensions"
                            << params.ROBOT_WIDTH << params.ROBOT_LENGTH << params.ROBOT_HEIGHT;
        return;
    }

    dsr_body_id_ = body_node->id();

    if (const auto width_value = G->get_attrib_by_name<width_m_att>(body_node.value()); width_value.has_value())
        params.ROBOT_WIDTH = width_value.value();
    if (const auto depth_value = G->get_attrib_by_name<depth_m_att>(body_node.value()); depth_value.has_value())
        params.ROBOT_LENGTH = depth_value.value();
    if (const auto height_value = G->get_attrib_by_name<height_m_att>(body_node.value()); height_value.has_value())
        params.ROBOT_HEIGHT = height_value.value();

    epistemic_controller_.set_robot_footprint(params.ROBOT_WIDTH, params.ROBOT_LENGTH);

    qCInfo(logGraph) << "Robot dimensions from body node: width depth height ="
                     << params.ROBOT_WIDTH << params.ROBOT_LENGTH << params.ROBOT_HEIGHT;
}

///////////////////////////////////////////////////////////////////////////////
/// Create one DSR node of type "wall" per polygon edge (static, set once) and
/// one "floor" node at the room-frame origin, all hanging from the room node.
///
/// Convention (internal polygon frame, which is CW in screen/Y-down space):
///   X+ = along wall (walk direction from P_i to P_{i+1})
///   Y+ = outward from room (left-hand perp of direction = (-dy, dx) for CW)
///   Yaw stored in RT edge = atan2(dir.y, dir.x)
///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::dsr_create_wall_nodes()
{
    if (!G || !rt_api) return;

    auto room_node_opt = G->get_node(dsr_room_id_);
    if (!room_node_opt.has_value()) { qWarning() << "dsr_create_wall_nodes: room node missing"; return; }

    // Guard — idempotent: if wall nodes already exist under this room, skip.
    if (!G->get_nodes_by_type("wall").empty())
        return;

    const auto polygon = room_concept_.nominal_room_polygon();
    const int n = static_cast<int>(polygon.size());
    if (n < 3) { qWarning() << "dsr_create_wall_nodes: polygon has fewer than 3 vertices"; return; }

    const float half_h = params.room_height * 0.5f;

    // ── Walls ────────────────────────────────────────────────────────────────
    for (int i = 0; i < n; ++i)
    {
        const Eigen::Vector2f& p0 = polygon[i];
        const Eigen::Vector2f& p1 = polygon[(i + 1) % n];
        const float L = (p1 - p0).norm();
        if (L < 0.1f)
        {
            qWarning() << "dsr_create_wall_nodes: skipping degenerate wall" << i << "(length" << L << "m)";
            continue;
        }

        const Eigen::Vector2f dir = (p1 - p0) / L;
        const float yaw = std::atan2(dir.y(), dir.x());
        const Eigen::Vector2f mid = (p0 + p1) * 0.5f;

        DSR::Node wall_node = DSR::Node::create<wall_node_type>("wall_" + std::to_string(i));
        G->add_or_modify_attrib_local<width_m_att>(wall_node, L);
        G->add_or_modify_attrib_local<height_m_att>(wall_node, params.room_height);
        G->add_or_modify_attrib_local<parent_att>(wall_node, dsr_room_id_);
        G->add_or_modify_attrib_local<level_att>(wall_node, 4);

        const auto wall_id = G->insert_node(wall_node);
        if (!wall_id.has_value())
        {
            qWarning() << "dsr_create_wall_nodes: failed to insert wall_" + QString::number(i);
            continue;
        }

        rt_api->insert_or_assign_edge_RT(room_node_opt.value(),
                                         wall_id.value(),
                                         {mid.x(), mid.y(), half_h},
                                         {0.f, 0.f, yaw});
    }

    // ── Floor ─────────────────────────────────────────────────────────────────
    // Purely semantic parent for floor-attached objects; placed at room origin.
    DSR::Node floor_node = DSR::Node::create<floor_node_type>("floor");
    G->add_or_modify_attrib_local<parent_att>(floor_node, dsr_room_id_);
    G->add_or_modify_attrib_local<level_att>(floor_node, 4);

    const auto floor_id = G->insert_node(floor_node);
    if (!floor_id.has_value())
        qWarning() << "dsr_create_wall_nodes: failed to insert floor node";
    else
        rt_api->insert_or_assign_edge_RT(room_node_opt.value(),
                                         floor_id.value(),
                                         {0.f, 0.f, 0.f},
                                         {0.f, 0.f, 0.f});

    trigger_graph_layout_twopi();
}

///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::cleanup_room_graph_nodes()
{
    if (!G) return;
    // Delete affordance nodes hanging from room via "has_intention" edges,
    // then delete the room nodes themselves.
    for (const auto& room_node : G->get_nodes_by_type("room"))
    {
        for (const auto& edge : G->get_node_edges_by_type(room_node, "has_intention"))
            if (G->get_node(edge.to()).has_value())
                G->delete_node(edge.to());
        G->delete_node(room_node);
    }
    // Delete wall and floor nodes owned by this agent.
    for (const auto& n : G->get_nodes_by_type("wall"))
        G->delete_node(n);
    if (auto n = G->get_node("floor"); n.has_value())
        G->delete_node(n.value());
    // Fallback: delete the affordance node by its known name in case it is orphaned.
    if (auto n = G->get_node("afford"); n.has_value())
        G->delete_node(n.value());
    room_node_created_ = false;
    dsr_room_id_ = 0;
    affordance_manager_.reset();
    stable_frames_ = 0;
}

void SpecificWorker::check_init_graph_is_valid()
{
    if (!G) { qCWarning(logGraph) << "dsr_init_graph: DSR graph not available"; return; }

    // Resolve the root/world node by type (name may vary, e.g. "root", "world")
    const auto root_nodes = G->get_nodes_by_type("root");
    if (!root_nodes.empty())
    {
        dsr_world_id_ = root_nodes.front().id();
    }
    else { qCWarning(logGraph) << "dsr_init_graph: no 'root' type node found in graph"; return; }

    // Resolve the robot node by type
    const auto robot_nodes = G->get_nodes_by_type("robot");
    if (!robot_nodes.empty())
    {
        dsr_robot_id_ = robot_nodes.front().id();
    }
    else { qCWarning(logGraph) << "dsr_init_graph: no 'robot' type node found in graph"; return; }

    load_robot_body_dimensions_from_graph();
}

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
void SpecificWorker::update_epistemic_overlay()
{
    // Epistemic score grid heatmap overlay (drawn behind lidar/robot by z-order).
    const auto& planner = epistemic_controller_.epistemic_planner();
    const auto& cell_scores = planner.cell_scores();
    std::vector<std::pair<Eigen::Vector2f, float>> score_cells;
    score_cells.reserve(cell_scores.size());
    for (const auto& cell : cell_scores)
        score_cells.emplace_back(cell.center, cell.score);
    viewer_2d_->draw_score_grid(score_cells, planner.cell_size());

    // IoR inhibition overlay: warm red fades out as visited cells recover
    const auto& ior = planner.ior_cells();
    std::vector<std::pair<Eigen::Vector2f, float>> ior_cells;
    ior_cells.reserve(ior.size());
    for (const auto& cell : ior)
        ior_cells.emplace_back(cell.center, cell.freshness);
    viewer_2d_->draw_ior_grid(ior_cells, planner.cell_size());

    const auto& current_target = planner.current_target();
    if (current_target.has_value() && !current_target->rotate_in_place)
    {
        viewer_2d_->draw_selected_grid_cell(current_target->position, planner.cell_size());
        viewer_2d_->update_target_marker(current_target->position.x(),
                                         current_target->position.y(),
                                         true);
    }
    else
    {
        viewer_2d_->draw_selected_grid_cell(std::nullopt, planner.cell_size());
        viewer_2d_->update_target_marker(0.f, 0.f, false);
    }
}

///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::update_ui(const std::optional<rc::RoomConcept::UpdateResult>& loc_res)
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

void SpecificWorker::initialize_room_model_from_svg()
{
    const auto room_polygon = rc::SvgRoomLoader::load_polygon_points(
        "beta_layout.svg", "room_contour", false, true);
    if (room_polygon.size() >= 3)
    {
        room_concept_.configure_room_from_polygon(room_polygon);
        room_initialized_from_svg_polygon_ = true;
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
/// SLOTS from GUI and DSR signals
///////////////////////////////////////////////////////////////////////////////

void SpecificWorker::modify_node_slot(std::uint64_t id, const std::string &type)
{
    if (!G)
        return;

    if (type != "laser")
        return;

    const auto node_opt = G->get_node(id);
    if (!node_opt.has_value())
        return;

    const auto& lidar3D = node_opt.value();
    const std::string& name = lidar3D.name();
    if (name != params.LIDAR_NAME)
        return;

    const auto lx = G->get_attrib_by_name<laser_X_att>(lidar3D);
    const auto ly = G->get_attrib_by_name<laser_Y_att>(lidar3D);
    const auto lz = G->get_attrib_by_name<laser_Z_att>(lidar3D);
    const auto laser_ts = G->get_attrib_by_name<laser_timestamp_att>(lidar3D);
    if (!lx.has_value() || !ly.has_value() || !lz.has_value() || !laser_ts.has_value())
    { 
        std::print("[modify_node_slot] Node '{}' missing laser_X/Y/Z attributes\n", name);
        return;
    }
    else
    {
        const auto &xs = lx.value().get();
        const auto &ys = ly.value().get();
        const auto &zs = lz.value().get();
        const std::size_t npts = std::min({xs.size(), ys.size(), zs.size()});

        std::vector<Eigen::Vector3f> points_high;
        points_high.reserve(npts);
        const float min_h_m = params.LIDAR_HIGH_MIN_HEIGHT;

        for (std::size_t i = 0; i < npts; ++i)
        {
            const float x_raw = xs[i];
            const float y_raw = ys[i];
            const float z_raw = zs[i];
            if (!std::isfinite(x_raw) || !std::isfinite(y_raw) || !std::isfinite(z_raw))
            {
                continue;
            }

            if (z_raw > min_h_m)
                points_high.emplace_back(x_raw, y_raw, z_raw);
        }
        const std::uint64_t ts = static_cast<std::uint64_t>(std::max<std::int64_t>(0, static_cast<std::int64_t>(laser_ts.value_or(0))));
        rc::LidarData lidar_data{std::move(points_high), static_cast<std::int64_t>(laser_ts.value())};

        high_lidar_buffer_.put<0>(std::move(lidar_data), ts);
        room_concept_.notify_new_lidar(static_cast<std::int64_t>(ts));
    }
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
void SpecificWorker::slot_toggle_lidar_points_display(bool checked)
{
    if (viewer_2d_)
        viewer_2d_->set_lidar_points_visible(checked);
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

/// @brief ///////////DSR callback triggered when a node is modified. We check if it's the robot node and if the current speed attributes have been updated, then we read them and push them to the odometry buffer with some optional noise added.

void SpecificWorker::modify_node_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names)
{
    if (!G || id == 0)
        return;

    if (dsr_robot_id_ != 0 && id != dsr_robot_id_)
        return;

    const auto touches_any = [&att_names](std::initializer_list<const char*> names)
    {
        return std::ranges::any_of(names, [&att_names](const char* name)
        {
            return std::find(att_names.begin(), att_names.end(), name) != att_names.end();
        });
    };

    const bool touches_current_speed = touches_any({
        "robot_current_advance_speed",
        "robot_current_side_speed",
        "robot_current_angular_speed",
        "robot_current_speed_timestamp"
    });
    const bool touches_ref_speed = touches_any({
        "robot_ref_adv_speed",
        "robot_ref_side_speed",
        "robot_ref_rot_speed",
        "robot_ref_speed_timestamp"
    });

    if (not touches_current_speed and not touches_ref_speed)
        return;

    const auto node_opt = G->get_node(id);
    if (!node_opt.has_value())
        return;

    const auto &robot_node = node_opt.value();

    if (touches_current_speed)
    {
        if (auto adv_value = G->get_attrib_by_name<robot_current_advance_speed_att>(robot_node); adv_value.has_value())
        {
            if (auto side_value = G->get_attrib_by_name<robot_current_side_speed_att>(robot_node); side_value.has_value())
            {
                if (auto rot_value = G->get_attrib_by_name<robot_current_angular_speed_att>(robot_node); rot_value.has_value())
                {
                    if (auto ts_value = G->get_attrib_by_name<robot_current_speed_timestamp_att>(robot_node); ts_value.has_value())
                    {
                        const auto source_ts = static_cast<std::uint64_t>(ts_value.value());
                        if (source_ts > 0 and source_ts > last_robot_current_speed_timestamp_)
                        {
                            static std::mt19937 gen{std::random_device{}()};
                            const float nf = params.ODOMETRY_NOISE_FACTOR;

                            auto add_noise = [&](float value) -> float {
                                if (nf <= 0.f || value == 0.f) return value;
                                std::normal_distribution<float> dist(0.f, std::abs(value) * nf);
                                return value + dist(gen);
                            };

                            rc::OdometryReading odom;
                            odom.adv = add_noise(adv_value.value());
                            odom.side = add_noise(side_value.value());
                            odom.rot = add_noise(rot_value.value());
                            odom.source_ts_ms = static_cast<std::int64_t>(source_ts);
                            odom.recv_ts_ms = static_cast<std::int64_t>(
                                std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::system_clock::now().time_since_epoch()).count());
                            odom.timestamp = std::chrono::high_resolution_clock::time_point(
                                std::chrono::milliseconds(source_ts));

                            room_concept_.record_odometry_ingress("dsr_current_speed",
                                                                  adv_value.value(),
                                                                  rot_value.value(),
                                                                  odom.adv,
                                                                  odom.rot,
                                                                  odom.source_ts_ms);
                            odometry_buffer_.put<0>(std::move(odom), static_cast<std::uint64_t>(odom.recv_ts_ms));
                            last_robot_current_speed_timestamp_ = source_ts;
                            last_robot_adv_speed_  = adv_value.value();
                            last_robot_side_speed_ = side_value.value();
                            last_robot_rot_speed_  = rot_value.value();
                        }
                    }
                }
            }
        }
    }

    if (touches_ref_speed)
    {
        if (auto adv_value = G->get_attrib_by_name<robot_ref_adv_speed_att>(robot_node); adv_value.has_value())
        {
            if (auto side_value = G->get_attrib_by_name<robot_ref_side_speed_att>(robot_node); side_value.has_value())
            {
                if (auto rot_value = G->get_attrib_by_name<robot_ref_rot_speed_att>(robot_node); rot_value.has_value())
                {
                    if (auto ts_value = G->get_attrib_by_name<robot_ref_speed_timestamp_att>(robot_node); ts_value.has_value())
                    {
                        const auto source_ts = static_cast<std::uint64_t>(ts_value.value());
                        if (source_ts > 0 and source_ts > last_robot_ref_speed_timestamp_)
                        {
                            rc::VelocityCommand cmd;
                            cmd.adv_y = adv_value.value();
                            cmd.adv_x = side_value.value();
                            cmd.rot = rot_value.value();
                            cmd.source_ts_ms = static_cast<std::int64_t>(source_ts);
                            cmd.recv_ts_ms = static_cast<std::int64_t>(
                                std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::system_clock::now().time_since_epoch()).count());
                            cmd.timestamp = std::chrono::high_resolution_clock::time_point(
                                std::chrono::milliseconds(source_ts));

                            room_concept_.record_command_ingress("dsr_ref",
                                                                 adv_value.value(),
                                                                 rot_value.value(),
                                                                 cmd.adv_y,
                                                                 cmd.rot,
                                                                 cmd.source_ts_ms);
                            velocity_buffer_.put<0>(std::move(cmd), source_ts);
                            last_robot_ref_speed_timestamp_ = source_ts;
                        }
                    }
                }
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
/// ICE INTERFACE CALLBACKS
///////////////////////////////////////////////////////////////////////////////

void SpecificWorker::JoystickAdapter_sendData(RoboCompJoystickAdapter::TData data)
{
    rc::VelocityCommand cmd;
    float raw_adv_y = 0.f;
    float raw_rot = 0.f;
    for (const auto& axis : data.axes)
    {
        if      (axis.name == "rotate")
        {
            raw_rot = axis.value;
            cmd.rot = axis.value;
        }
        else if (axis.name == "advance")
        {
            raw_adv_y = axis.value / 1000.0f;
            cmd.adv_y = raw_adv_y;
        }
        else if (axis.name == "side")    cmd.adv_x = 0.0f;
    }
    cmd.timestamp  = std::chrono::high_resolution_clock::now();
    cmd.recv_ts_ms = static_cast<std::int64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    room_concept_.record_command_ingress("joystick",
                                         raw_adv_y,
                                         raw_rot,
                                         cmd.adv_y,
                                         cmd.rot,
                                         cmd.recv_ts_ms);
    velocity_buffer_.put<0>(std::move(cmd), static_cast<std::uint64_t>(cmd.recv_ts_ms));
}

