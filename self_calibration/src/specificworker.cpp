#include <algorithm>
#include "specificworker.h"
#include <chrono>

#include "../../common/agent_exit/terminal_exit.h"        // rc::agent::terminal_exit (SHARED)
#include "../../common/config_report/config_read.h"       // rc::cfg::Reader (SHARED)

#include <print>

#include <QCoreApplication>
#include <QTimer>

#include <dsr/api/dsr_api.h>

SpecificWorker::SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check)
    : GenericWorker(configLoader, tprx)
{
    this->startup_check_flag = startup_check;
    if (this->startup_check_flag)
    {
        this->startup_check();
        return;
    }

    // Agent-presence state machine: Waiting -> Operating -> Degraded (as bottle_concept).
    int period = 0;
    rc::cfg::Reader(configLoader, "self_calibration").req("Period.Compute", period,
            "GRAFCET step period (ms) for every state of this agent's state machine");

    {
        const rc::cfg::Reader cfg(configLoader, "self_calibration");
        inputs_cfg_.camera_rate_hz   = cfg.d("Inputs.camera_rate_hz", inputs_cfg_.camera_rate_hz,
                "zed rgb/depth and ricoh hand-over rate (Hz): a COMPUTE budget, frames beyond it are drained unconverted; <=0 disables cameras");
        inputs_cfg_.stall_lidar_ms   = cfg.i("Inputs.stall_lidar_ms", inputs_cfg_.stall_lidar_ms, "LiDAR stream declared stalled after this silence (ms)");
        inputs_cfg_.stall_camera_ms  = cfg.i("Inputs.stall_camera_ms", inputs_cfg_.stall_camera_ms, "camera stream declared stalled after this silence (ms)");
        inputs_cfg_.stall_imu_ms     = cfg.i("Inputs.stall_imu_ms", inputs_cfg_.stall_imu_ms, "IMU stream declared stalled after this silence (ms)");
        inputs_cfg_.log_period_ms    = cfg.i("Inputs.log_period_ms", inputs_cfg_.log_period_ms, "stream-rate log period (ms)");
    }

    {
        const rc::cfg::Reader cfg(configLoader, "self_calibration");
        auto& h = harvester_cfg_;
        h.keyframe_spacing_s = cfg.d("Harvester.keyframe_spacing_s", h.keyframe_spacing_s, "a Helios sweep becomes a keyframe at most this often (s): a COMPUTE budget, the posterior does not depend on it");
        h.ring_span_s        = cfg.d("Harvester.ring_span_s", h.ring_span_s, "time the keyframe ring holds before pairs are committed (s): compute budget");
        h.max_keyframes      = std::size_t(cfg.i("Harvester.max_keyframes", int(h.max_keyframes), "hard bound on the keyframe ring"));
        h.wheel_rot_sigma_rel   = cfg.d("Harvester.wheel_rot_sigma_rel", h.wheel_rot_sigma_rel, "relative sigma of the wheel rotation (Shadow over-reports ~41 %)");
        h.wheel_rot_sigma_floor = cfg.d("Harvester.wheel_rot_sigma_floor", h.wheel_rot_sigma_floor, "wheel rotation noise floor (rad/sqrt(s))");
        h.gyro_sigma_rel        = cfg.d("Harvester.gyro_sigma_rel", h.gyro_sigma_rel, "relative sigma of the integrated gyro rotation");
        h.gyro_sigma_floor      = cfg.d("Harvester.gyro_sigma_floor", h.gyro_sigma_floor, "gyro rotation noise floor (rad/sqrt(s))");
        h.reg_sigma_m        = cfg.d("Harvester.reg_sigma_m", h.reg_sigma_m, "scan-registration noise floor per pair row (m)");
        h.odo_sigma_per_m    = cfg.d("Harvester.odo_sigma_per_m", h.odo_sigma_per_m, "odometry error per metre travelled between the two keyframes (m/m)");
        h.odo_sigma_per_rad  = cfg.d("Harvester.odo_sigma_per_rad", h.odo_sigma_per_rad, "odometry translation error per radian rotated between the two keyframes (m/rad)");
    }

    {
        const rc::cfg::Reader cfg(configLoader, "self_calibration");
        zed_floor_enabled_ = cfg.b("ZedFloor.enabled", zed_floor_enabled_, "run the ZED floor-plane factor (estimate + log only)");
        zed_floor_cfg_.rate_hz = cfg.d("ZedFloor.rate_hz", zed_floor_cfg_.rate_hz, "depth frames fed per second (Hz): a COMPUTE budget");
        zed_floor_cfg_.stride = cfg.i("ZedFloor.stride", zed_floor_cfg_.stride, "every stride-th depth pixel in u and v: a COMPUTE budget");
        zed_floor_log_period_ms_ = cfg.i("ZedFloor.log_period_ms", zed_floor_log_period_ms_, "summary line + CSV row period (ms)");
    }

    {
        const rc::cfg::Reader cfg(configLoader, "self_calibration");
        mount_factors_enabled_ = cfg.b("MountFactors.enabled", mount_factors_enabled_,
                "run the floor + vertical LiDAR mount factors (estimate + log only)");
        mount_cfg_.rate_hz = cfg.d("MountFactors.rate_hz", mount_cfg_.rate_hz,
                "sweeps fed per second per LiDAR (Hz): a COMPUTE budget");
        mount_cfg_.pose_sigma_xy = cfg.d("MountFactors.pose_sigma_xy", mount_cfg_.pose_sigma_xy,
                "m, WIDE per-sweep pose nuisance prior of the vertical factor (never the localiser's marginal)");
        mount_cfg_.pose_sigma_yaw_deg = cfg.d("MountFactors.pose_sigma_yaw_deg", mount_cfg_.pose_sigma_yaw_deg,
                "deg, WIDE per-sweep pose-yaw nuisance prior of the vertical factor");
        mount_log_period_ms_ = cfg.i("MountFactors.log_period_ms", mount_log_period_ms_,
                "summary line + mounts CSV row period (ms)");
    }

    {
        const rc::cfg::Reader cfg(configLoader, "self_calibration");
        helios_enabled_ = cfg.b("HeliosMount.enabled", helios_enabled_, "register the harvester's keyframe pairs and estimate the Helios planar mount (estimate + log only)");
        helios_cfg_.band_min_m = cfg.d("HeliosMount.band_min_m", helios_cfg_.band_min_m, "lower height of the 2-D wall band (m); room_concept's Media.lidar_high_min_height");
        helios_cfg_.band_max_m = cfg.d("HeliosMount.band_max_m", helios_cfg_.band_max_m, "upper height of the 2-D wall band (m); room_concept's Media.lidar_high_max_height");
        helios_cfg_.gyro_bias_sigma = cfg.d("HeliosMount.gyro_bias_sigma", helios_cfg_.gyro_bias_sigma, "rad/s, prior on the common gyro bias (marginalised nuisance)");
        helios_cfg_.gyro_scale_sigma = cfg.d("HeliosMount.gyro_scale_sigma", helios_cfg_.gyro_scale_sigma, "fractional prior on the common gyro scale (marginalised nuisance)");
        helios_log_period_ms_ = cfg.i("HeliosMount.log_period_ms", helios_log_period_ms_, "summary line + one mount CSV row period (ms)");
    }

    dashboard_show_ = rc::cfg::Reader(configLoader, "self_calibration").b("Dashboard.show", true,
            "open the live trajectory-fragment dashboard window (display only; false = no window and no cost)");

    states["Waiting"] = std::make_unique<GRAFCETStep>("Waiting", period,
        std::bind(&SpecificWorker::waiting_loop, this), std::bind(&SpecificWorker::waiting_enter, this));
    states["Operating"] = std::make_unique<GRAFCETStep>("Operating", period,
        std::bind(&SpecificWorker::operating_loop, this), std::bind(&SpecificWorker::operating_enter, this));
    states["Degraded"] = std::make_unique<GRAFCETStep>("Degraded", period,
        std::bind(&SpecificWorker::degraded_loop, this), std::bind(&SpecificWorker::degraded_enter, this));

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
    if (error.length() > 0)
    {
        qWarning() << error;
        throw error;
    }
}

SpecificWorker::~SpecificWorker()
{
    request_shutdown();
    save_dashboard_geometry();
    delete dashboard_window_;   // top-level, parentless: owned here
    std::print("self_calibration: SpecificWorker destroyed.\n");
}

void SpecificWorker::request_shutdown()
{
    if (shutting_down_.exchange(true))
        return;
    inputs_.reset();   // subscribers + InnerEigenAPI go down first, from the main thread, while G is alive
    if (G)
        disconnect(G.get(), nullptr, this, nullptr);   // sever graph callbacks before any teardown
    cleanup_owned_nodes();
}

void SpecificWorker::terminal_shutdown()
{
    static std::atomic<bool> terminating{false};
    if (terminating.exchange(true))
        return;
    rc::agent::terminal_exit([this] { request_shutdown(); }, [this] { if (G) G->reset(); });
}

void SpecificWorker::initialize()
{
    std::print("self_calibration: initialize()\n");
    GenericWorker::initialize();   // creates the DSR graph and, with it, this agent's own node "<name> <id>"

    if (not G)
    {
        qWarning() << "self_calibration: DSR graph not available in initialize()";
        return;
    }

    // Nothing here reads pixels out of the graph (media plane only, later tasks).
    G->set_ignored_attributes<cam_rgb_att, cam_depth_att, laser_X_att, laser_Y_att, laser_Z_att>();

    // The whole presence protocol is SHARED (common/concept_presence): debounced Degraded, one-shot post-sync
    // sweep on first Operating. This agent has NO primary input stream yet, so the input is "always live".
    presence_protocol_.wire(
        presence_coordinator_, &statemachine, this, configLoader, G, static_cast<std::uint32_t>(agent_id),
        {
            .shutting_down      = [this] { return shutting_down_.load(); },
            .pump_primary_input = {},
            .primary_age_ms     = [] { return std::int64_t{-1}; },
            .primary_live       = [] { return true; },
            .primary_stalled    = [](std::int64_t*) { return false; },
            .emit_ready         = [this] { emit presenceReady(); },
            .emit_lost          = [this] { emit presenceLost(); },
            .compute            = [this] { compute(); },
            .terminal_shutdown  = [this] { terminal_shutdown(); },
            .on_first_operating = [this] { remove_stale_nodes(); },
            .on_optional_peer_lost  = [](const std::string& name, std::uint32_t)
                                      { qInfo() << "[Presence] optional peer lost:" << QString::fromStdString(name); },
            .on_optional_peer_ready = [](const std::string& name, std::uint32_t)
                                      { qInfo() << "[Presence] optional peer ready:" << QString::fromStdString(name); },
        });

    // SIGTERM / Ctrl+C (sigwatch -> quit -> aboutToQuit) goes through the crash-free terminal shutdown.
    QObject::connect(QCoreApplication::instance(), &QCoreApplication::aboutToQuit,
                     this, &SpecificWorker::terminal_shutdown, Qt::UniqueConnection);

    build_dashboard();
    remove_stale_nodes();
    std::print("self_calibration: initialize() done, agent node '{} {}' present\n", agent_name, agent_id);
}

void SpecificWorker::compute()
{
    if (shutting_down_.load() or not G)
        return;
    // Created here (an Operating tick, main thread), never at t=0 from initialize(): the graph is loaded and the
    // sensor nodes have synced by now (CLAUDE.md, media plane).
    if (not inputs_)
        inputs_ = std::make_unique<sc::SensorInputs>(G, inputs_cfg_);
    inputs_->poll();
    if (not harvester_)
        harvester_ = std::make_unique<sc::calib::TrajectoryHarvester>(harvester_cfg_);
    // Odometry first (piecewise-constant over source_ts_ms deltas, inside the harvester), then the sweeps.
    for (const auto& o : inputs_->take_odometry())
    {
        harvester_->add_odometry(o.source_ts_ms, o.adv, o.side, o.rot);
        odo_stamps_.push_back(o.source_ts_ms);
        if (odo_stamps_.size() > 400) odo_stamps_.pop_front();
    }
    for (const auto& l : inputs_->take_lidar())
    {
        feed_mount_factors(l);
        if (l.sensor == "helios")
        {
            const bool stalled = inputs_->stalled("helios") or inputs_->any_stalled();
            const bool captured = harvester_->add_keyframe(l.stamp_ms, stalled);
            if (captured and helios_enabled_ and l.robot_from_sensor.has_value())
            {
                if (not helios_stage_)
                    helios_stage_ = std::make_unique<sc::HeliosMountStage>(helios_cfg_, harvester_cfg_);
                helios_stage_->store_keyframe(l.stamp_ms, l.points_robot());
                note_helios_clock(l.stamp_ms);
            }
            feed_dashboard_keyframe(captured, stalled);
            if (dash_model_)
                dash_model_->note_lidar_stamp(l.stamp_ms, std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count());
        }
    }
    for (const auto& m : inputs_->take_imu())
        harvester_->add_gyro(m.stamp_ms, m.gyro.z());   // yaw rate about the IMU z axis (assumed up: pin with the mount)
    for (const auto& im : inputs_->take_images())       // only the ZED depth is consumed (floor plane); other cameras are dropped
        feed_zed_floor(im);
    // Phase 4.4 attaches scan registration to these pairs; for now the agent only reports them (<= once / 10 s).
    const auto pairs = harvester_->take_pairs();
    npairs_logged_ += long(pairs.size());
    if (helios_stage_)
    {
        for (const auto& p : pairs)
            helios_stage_->add_pair(p);
        if (const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count(); now - helios_log_ms_ >= helios_log_period_ms_)
        {
            helios_log_ms_ = now;
            helios_stage_->log_state(now);
            qInfo().noquote() << QString::fromStdString(helios_stage_->summary());
        }
    }
    feed_dashboard_pairs(pairs);
    if (const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count(); now - harvester_log_ms_ >= 10000)
    {
        harvester_log_ms_ = now;
        const auto e = harvester_->close_episode();
        qInfo().noquote() << QString("[Harvester] keyframes %1 (stalled-dropped %2)  pairs %3 (total %4)  info(dx,dy,psi)=(%5, %6, %7)")
                .arg(e.keyframes).arg(e.dropped_stalled).arg(e.pairs).arg(npairs_logged_)
                .arg(e.info(0, 0), 0, 'g', 3).arg(e.info(1, 1), 0, 'g', 3).arg(e.info(2, 2), 0, 'g', 3);
    }
    refresh_dashboard();
    fps.print("self_calibration compute", 3000);
}

// Clock check (plan 4.4 step 5): the Helios stamp and the odometry source stamp must share one clock. Once, over the
// first 50 keyframes: median and spread of (keyframe stamp - nearest odometry stamp). Reported, never corrected here.
void SpecificWorker::note_helios_clock(std::int64_t keyframe_stamp_ms)
{
    if (helios_clock_logged_ or odo_stamps_.empty())
        return;
    const auto it = std::ranges::lower_bound(odo_stamps_, keyframe_stamp_ms);   // stamps are monotone
    std::int64_t best = odo_stamps_.back();
    if (it != odo_stamps_.end())
        best = (it != odo_stamps_.begin() and keyframe_stamp_ms - *(it - 1) < *it - keyframe_stamp_ms) ? *(it - 1) : *it;
    helios_stage_->note_clock_offset_ms(double(keyframe_stamp_ms - best));
    if (helios_stage_->clock_samples() >= 50)
    {
        helios_clock_logged_ = true;
        qInfo().noquote() << QString::fromStdString(helios_stage_->clock_summary());
    }
}

// Floor + vertical LiDAR mount factors (plan Phase 3). A stalled stream's sweeps are not used (sensor_inputs
// contract); a sweep without a resolvable robot-frame mount is skipped (nothing to put the points in).
void SpecificWorker::feed_mount_factors(const sc::LidarIn& l)
{
    if (not mount_factors_enabled_ or inputs_->stalled(l.sensor) or not l.robot_from_sensor.has_value())
        return;
    if (not mount_stage_)
        mount_stage_ = std::make_unique<sc::LidarMountStage>(mount_cfg_);
    const auto& poly = inputs_->polygon();
    // room_concept's factor used the MEASURED ceiling only (0 = no ceiling rows); a configured height has sigma 0.
    const float ceiling = poly.ceiling_sigma_m > 0.f ? poly.ceiling_m : 0.f;
    mount_stage_->observe({.sensor = l.sensor, .stamp_ms = l.stamp_ms, .points_robot = l.points_robot(),
                           .origin_robot = l.robot_from_sensor->translation(), .room_T_robot = l.room_T_robot},
                          poly.vertices, ceiling);
    if (const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count(); now - mount_log_ms_ >= mount_log_period_ms_)
    {
        mount_log_ms_ = now;
        mount_stage_->log_state(now);
        qInfo().noquote() << QString::fromStdString(mount_stage_->summary());
    }
}

// ZED floor plane (plan 4.1). Only the zed depth stream; a stalled stream's frames and frames without a resolved
// robot-frame mount or intrinsics are not used.
void SpecificWorker::feed_zed_floor(const sc::ImageIn& im)
{
    if (not zed_floor_enabled_ or im.sensor != "zed" or im.stream != "depth" or inputs_->stalled("zed/depth")
        or not im.robot_from_sensor.has_value())
        return;
    if (not zed_floor_stage_)
        zed_floor_stage_ = std::make_unique<sc::ZedFloorStage>(zed_floor_cfg_);
    zed_floor_stage_->observe(im, im.intrinsics, im.room_T_robot);
    if (const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count(); now - zed_floor_log_ms_ >= zed_floor_log_period_ms_)
    {
        zed_floor_log_ms_ = now;
        zed_floor_stage_->log_state(now);
        qInfo().noquote() << QString::fromStdString(zed_floor_stage_->summary());
    }
}

void SpecificWorker::emergency()
{
    fps.print("Emergency worker", 3000);
}

void SpecificWorker::restore()
{
    std::print("self_calibration: restore()\n");
}

int SpecificWorker::startup_check()
{
    std::print("self_calibration: startup_check()\n");
    QTimer::singleShot(200, QCoreApplication::instance(), SLOT(quit()));
    return 0;
}

// ─── Presence state-machine step hooks (delegated to the coordinator) ──────────
void SpecificWorker::waiting_enter()   { presence_coordinator_.waiting_enter(); }
void SpecificWorker::waiting_loop()    { presence_coordinator_.waiting_loop(); }
void SpecificWorker::operating_enter() { presence_coordinator_.operating_enter(); }
void SpecificWorker::operating_loop()  { presence_coordinator_.operating_loop(); }
void SpecificWorker::degraded_enter()  { presence_coordinator_.degraded_enter(); }
void SpecificWorker::degraded_loop()   { presence_coordinator_.degraded_loop(); }

// The only node this agent owns today is its own agent node ("self_calibration 27"), which the DSR graph
// constructor (re)creates and cleanup_owned_nodes() deletes through [Owns]. This sweep reaps anything ELSE
// carrying the agent's name left by a crashed run (future proposal nodes). The live agent node is excluded.
void SpecificWorker::remove_stale_nodes()
{
    if (not G)
        return;
    std::vector<std::uint64_t> stale;
    for (const auto& node : G->get_nodes_by_type("metaconcept"))
        if (node.name().starts_with("self_calibration"))
            stale.push_back(node.id());
    for (const auto id : stale)
        G->delete_node(id);
    if (not stale.empty())
        std::print("self_calibration: swept {} stale node(s)\n", stale.size());
}

void SpecificWorker::cleanup_owned_nodes()
{
    if (owned_nodes_cleaned_)
        return;
    owned_nodes_cleaned_ = true;
    remove_stale_nodes();                          // while the monitor is still up, before [Owns] deletion
    presence_coordinator_.cleanup_owned_nodes();   // stops the monitor FIRST, then deletes the [Owns] nodes
}
