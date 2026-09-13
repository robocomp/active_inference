// ── STARTUP AND SHUTDOWN ─────────────────────────────────────────────────────────────────────────
// SpecificWorker::initialize() and request_shutdown(), in their own translation unit. Same class:
// this IS the coordination the worker exists for — it builds the collaborators, hands each the few
// things it borrows, wires the presence policy and starts the threads — and it is ~440 lines of it,
// which is why it does not belong in the same file as the per-tick loop.
//
// ⚠ ORDER IS LOAD-BEARING HERE, in three places that nothing in the surrounding code shows:
//   · every collaborator is built BEFORE the viewer and borrows viewer_raw_slot_, not the viewer;
//   · CalibChannels::configure() builds its calibration channels AFTER the driving extractor,
//     because they copy its config — the other order segfaulted on the first start;
//   · the DSR signal connects come after the collaborators exist, so a slot can never fire into a
//     half-built worker. They are QUEUED, never DirectConnection (CLAUDE.md).
// Each phase also times itself into tmp/startup_timing.csv and pumps the event loop, because all of
// this runs on the GUI thread and every millisecond here is a millisecond the window cannot paint.

#include "specificworker.h"

#include "calib_channels.h"
#include "ground_truth_log.h"
#include "mount_calibrator.h"
#include "pose_publisher.h"

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_rt_api.h>
#include <dsr/core/types/type_checking/dsr_attr_name.h>

#include <QCoreApplication>
#include <QDateTime>
#include <QDebug>
#include <QElapsedTimer>
#include <QString>

#include <filesystem>
#include <fstream>
#include <locale>

void SpecificWorker::request_shutdown()
{
    if (shutting_down_.exchange(true))
        return;

    save_window_settings();
    if (viewer_)
        viewer_->persist_window_state();   // _Exit below skips ~RoomViewer, so save its state now
    save_robot_pose_once();

    // Drop the lidar media subscriber BEFORE tearing down RoomConcept (pump() calls
    // room_concept_.notify_new_lidar) and while G is still alive.
    if (calib_) calib_->stop();   // drop the RGB readers before the graph goes
    imu_ingestor_.reset();   // stop the IMU reader before the graph goes
    lidar_ingestor_.reset();

    room_concept_.stop();
    cleanup_owned_nodes();

    // Cleanly remove THIS agent's DDS participant + entities so peers free agent id 5 IMMEDIATELY.
    // A bare _Exit skips ~DSRGraph (which does exactly this), so the dead id-5 participant lingers in
    // live peers' discovery until the FastDDS liveliness lease expires — a quick reopen then hits
    // "There is already an agent connected with the id: 5". DSRGraph::reset() runs
    // remove_participant_and_entities() directly, WITHOUT touching the Ice communicator, so it doesn't
    // trip the static-destruction abort below. Mirrors bottle_concept's known-good shutdown. This must
    // run even if other shared_ptr copies of G survive — reset() is an explicit method, not refcount-
    // driven, so we don't need to chase down every holder (scene_graph_/camera_viz_/viewer_2d_).
    if (G)
    {
        try { G->reset(); }
        catch (...) { /* best-effort: exiting regardless */ }
    }

    // Crash-free terminal exit. After our cleanup (state saved, self agent node deleted, participant
    // removed, peers notified) we hard-exit instead of returning into Ice::Application's communicator
    // teardown + C++ static destruction. Those run with UNDEFINED cross-TU order: a global/DDS holder
    // copies a graph Node (e.g. type "mind") AFTER the node-type registry static is destroyed, so
    // Node::type() throws "<type> is not a valid node type" -> std::terminate/abort on every exit.
    // _Exit skips all of that; the OS reclaims memory/sockets/threads. Only reached on a real shutdown
    // (shutting_down_ latched above). Brief pause lets the removal deltas + participant departure reach
    // peers first.
    std::cout.flush();
    std::cerr.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    std::_Exit(EXIT_SUCCESS);
}

///////////////////////////////////////////////////////////////////////////////
void SpecificWorker::initialize()
{
    GenericWorker::initialize();

    // ── STARTUP PHASES: TIMED, AND THE WINDOW STAYS ALIVE THROUGH THEM ───────────────────────────
    // initialize() runs on the GUI THREAD, so every millisecond spent in it is a millisecond the
    // window cannot paint, resize or respond — the agent looks hung from the moment it appears until
    // the first estimate lands. Two things here, and they are deliberately together:
    //   · each phase's cost is appended to tmp/startup_timing.csv, so "which part is slow" stops
    //     being a guess. Nothing in this agent measured startup at all; compute_timing.csv only
    //     begins once compute() is already ticking.
    //   · processEvents() is pumped at each boundary, so the window paints and stays responsive
    //     across the slow phases instead of after them. Safe at these points: the DSR update signals
    //     are not connected until later in this function, so no graph slot can re-enter here, and the
    //     compute timer has not started.
    QElapsedTimer init_timer, phase_timer;
    init_timer.start();
    phase_timer.start();
    {   // truncate once, then never hold the stream open — see the phase() lambda
        std::ofstream reset("tmp/startup_timing.csv", std::ios::out | std::ios::trunc);
        if (reset.is_open()) { reset.imbue(std::locale::classic()); reset << "phase,ms,cumulative_ms\n"; }
    }
    const auto phase = [&](const char* name)
    {
        const auto ms = phase_timer.restart();
        // ⚠ OPEN-APPEND-CLOSE per row, because RoomViewer's constructor appends its own sub-phases to
        // this same file while this function is still running. Holding a truncating stream open across
        // that made TWO writers with independent file offsets: this one's next write landed at its own
        // offset and overwrote the viewer's rows — which silently destroyed the two rows that mattered
        // (viewer:custom_widget and viewer:viewer_2d, the latter being the 16-second one) and left a
        // half-overwritten line in the file. An instrument that erases its own most important reading
        // is worse than no instrument.
        std::ofstream f("tmp/startup_timing.csv", std::ios::out | std::ios::app);
        if (f.is_open())
        {
            f.imbue(std::locale::classic());
            f << name << ',' << ms << ',' << init_timer.elapsed() << '\n';
        }
        if (ms > 200)
            qInfo().noquote() << QString("[startup] %1 took %2 ms (cumulative %3 ms) — the window is "
                                         "blocked for the duration of any phase on this thread")
                                     .arg(name).arg(ms).arg(init_timer.elapsed());
        QCoreApplication::processEvents();
    };
    phase("generic_worker");

    // Ignore payload attributes in local graph updates to avoid unnecessary copying and processing of potentially large data
    G->set_ignored_attributes<cam_rgb_att, cam_depth_att>();
    qInfo() << "Ignoring DSR RGBD payload attributes cam_rgb/cam_depth in local graph updates";

    QObject::connect(QCoreApplication::instance(), &QCoreApplication::aboutToQuit,
                     this, &SpecificWorker::request_shutdown, Qt::UniqueConnection);

    // ── Load all config (agent + RoomConcept + EpistemicController params) ──
    rc::load_room_config(configLoader, params, room_concept_, epistemic_controller_);
    // The mount calibrator borrows the viewer SLOT, not the viewer: it is constructed here, long
    // before RoomViewer exists, and reads through the pointer whenever it needs to display.
    mount_ = std::make_unique<rc::MountCalibrator>(G, params, &viewer_raw_slot_);
    // ⚠ REVIEW 2026-09-13 (three reviewers, independently): rt_api_ IS NULL HERE. It is assigned a
    //   few lines below, so PosePublisher::rt_api_ is nullptr for the life of the process and is also
    //   blind to the HISTORY_SIZE = 25 tuning applied after that. INERT TODAY — pose_publisher.cpp
    //   never dereferences the member; it publishes through scene_graph_, which gets its own RT_API
    //   correctly after the assignment. Left as found because this review does not change behaviour.
    //   The hazard is that the constructor signature ASSERTS the dependency is satisfied: the first
    //   line in that class to use its injected RT API segfaults. Fix by dropping the parameter or by
    //   constructing below the assignment — do not fix by "just adding a null check".
    pose_pub_ = std::make_unique<rc::PosePublisher>(G, rt_api_.get(), params, room_concept_,
                                                    &viewer_raw_slot_, shutting_down_);
    // ⚠ REVIEW 2026-09-13: this lambda dereferences gt_log_, which is constructed on the NEXT line.
    //   Safe as written — nothing can publish in between (the localiser thread and the graph slots
    //   both start much later) — but it is the same shape as the set_scene_graph-inside-a-lambda bug
    //   that made this agent publish nothing all evening. If either statement ever moves, this is a
    //   null deref on the first corrected pose.
    pose_pub_->set_on_corrected_published([this](const rc::RoomConcept::UpdateResult& res) { gt_log_->log_ground_truth(res); });
    calib_ = std::make_unique<rc::CalibChannels>(G, params, room_concept_, *mount_, &viewer_raw_slot_);
    gt_log_ = std::make_unique<rc::GroundTruthLog>(G, room_concept_, shutting_down_);
    phase("load_config");

    // ── Collaborators (constructor injection; worker owns rt_api + shared params) ──
    rt_api_ = G->get_rt_api();
    // 2026-09-09: widen the RT timestamped history from the library default (5 blocks) so consumers
    // querying a real acquisition timestamp (camera/lidar, PTP-anchored) more than ~50ms old do not
    // silently clamp to the newest/oldest retained sample (RT_API::bracketing_blocks never
    // extrapolates -- see dsr_rt_api.cpp). publish_predicted_tick writes this SAME edge at the IMU
    // rate (~100 Hz measured, [ImuInject] coverage log), so N slots cover roughly N*10ms of history.
    // 25 slots -> ~250ms, chosen as a settled production value (was bumped to 150 = ~1.5s only for
    // the one-off ricoh_omni_dds time_offset test, since reverted -- see debug_pose_sync_sensores.md).
    // Process-local: rt_api_ is this worker's own instance (DSRGraph::get_rt_api() is a factory,
    // never shared -- see dsr_api.h), and this agent is the sole writer of the robot<->room RT edge,
    // so this is the only place that needs it.
    rt_api_->HISTORY_SIZE = 25;
    scene_graph_ = std::make_unique<rc::RoomSceneGraph>(
        G, rt_api_.get(), params, room_concept_, epistemic_controller_,
        [this] { trigger_graph_layout_twopi(); });
    // ⚠ THE PUBLISHER CANNOT PUBLISH WITHOUT THIS, AND IT MUST BE A STATEMENT, NOT PART OF THE LAMBDA
    // ABOVE. PosePublisher is built with the config, before the scene graph exists, so the scene graph
    // is handed over here. Written once inside that relayout lambda by mistake, it ran only when a
    // graph relayout fired — never in practice — so scene_graph_ stayed null, every publish returned
    // early, and the agent ran with NO pose published at all: no RT edge, no pose trace, no
    // ground-truth rows, and stable_frames_ stuck at 0/30 for ever with all three of its terms
    // passing. The guard that made a null non-fatal is what turned a crash into that silence.
    pose_pub_->set_scene_graph(scene_graph_.get());
    lidar_ingestor_ = std::make_unique<rc::LidarIngestor>(G, room_concept_, params);
    phase("ingestors");
    // No source switch any more: the producer no longer writes the imu_* attributes, so a "dsr"
    // option would select a path with nothing on it — a dead config of exactly the kind this
    // codebase keeps rediscovering. The escape hatch is git, not a flag that cannot work.
    imu_ingestor_ = std::make_unique<rc::ImuIngestor>(G, imu_buffer_, sim_clock_);
    // FIX 2026-09-04: publish a predicted pose on every IMU sample -- see publish_predicted_tick().
    // Runs on the imu-ingest thread; the callback itself takes publish_mutex_ before touching anything
    // shared with maybe_publish_corrected_pose().
    imu_ingestor_->set_on_new_sample([this](std::int64_t imu_ts_ms) { pose_pub_->publish_predicted_tick(imu_ts_ms); });
    // The RGB channels configure themselves — the 56 lines that stood here are CalibChannels::configure().
    calib_->configure();

    // ── Wire RoomConcept run context ───────────────────────────────────────
    rc::RoomConcept::RunContext run_ctx;
    run_ctx.high_lidar_buffer = &lidar_ingestor_->buffer();
    // The ceiling the LiDAR measured travels with the model, not through a second wire: the scene
    // graph publishes it on the room node and the image-edge module projects the contour at it.
    // ⚠ NOT HERE ANY MORE. This ran in initialize(), reading an atomic that starts at 0 before the
    //   LiDAR thread has ever produced a scan, so RoomConcept held 0 for the life of the process and
    //   every consumer took its `> 1.5f` fallback: the room node's room_height was never published
    //   from the measurement, and the camera's wall-ceiling corners sat at the STATED 3.0 m for ever.
    //   Measured 2026-09-12: pz read exactly 3.0000 on all 16607 ceiling rows, and the camera's own
    //   floor-vs-ceiling disagreement put the real ceiling ~14 cm lower. The copy now happens per
    //   cycle in compute(), where the measurement can actually exist.
    run_ctx.velocity_buffer = &velocity_buffer_;
    run_ctx.odometry_buffer = &odometry_buffer_;
    run_ctx.imu_buffer      = &imu_buffer_;
    run_ctx.sim_clock       = &sim_clock_;
    room_concept_.set_run_context(run_ctx);
    room_concept_.params.prediction_early_exit = params.PREDICTION_EARLY_EXIT;

    // The scenario overlay CHOOSES the layout file, so it has to be resolved before the SVG is
    // read. It used to be applied at the end of check_init_graph_is_valid() (below), which is
    // after this line — the axis was inert and the log still said "applied".
    scene_graph_->resolve_overlays_from_graph();
    if (not scene_graph_->overlays_resolved() and not params.scenario_overlays.empty())
    {
        // The graph could not name the robot yet, so the scenario is unknown — and the very next
        // line would read whatever layout the shared section happens to hold. Stopping is the
        // only honest option: a wrong floor plan reads downstream as a localiser fault.
        qCritical() << "[room] REFUSING TO START: no type-\"robot\" node in the graph when the"
                    << "layout must be chosen. Start robot_concept first.";
        std::exit(EXIT_FAILURE);
    }
    // The other half of the overlays: values whose home is the localiser or the planner, not the
    // config struct. Same sections, same log line — split only because RoomConfig does not own those
    // objects. ★ This is where CmdNoise* finally lands: the parse existed, the application did not.
    {
        QStringList l;
        for (const auto& c : params.apply_platform_to(scene_graph_->overlay_robot_name(),
                                                      room_concept_, epistemic_controller_))
            l << QString::fromStdString(c);
        for (const auto& c : params.apply_scenario_to(scene_graph_->overlay_scenario_name(),
                                                      epistemic_controller_))
            l << QString::fromStdString(c);
        if (not l.isEmpty())
            qInfo() << "[room] overlay (localiser/planner params) applied:" << l.join(", ");
    }
    // The compute period is fixed into the GRAFCET steps at construction, before the graph can name
    // the robot — so a per-robot Period is applied here instead, by re-setting the steps.
    if (const auto it = params.platform_overlays.find(scene_graph_->overlay_robot_name());
        it != params.platform_overlays.end() and it->second.period_compute.has_value())
    {
        const int per = *it->second.period_compute;
        for (const auto& name : {"Waiting", "Operating", "Degraded", "Emergency", "Initialize", "Compute"})
            if (const auto st = states.find(name); st != states.end() and st->second)
                st->second->setPeriod(per);
        qInfo() << "[room] platform overlay: compute period set to" << per << "ms";
    }

    if (room_concept_.estimating())
    {
        // RoomShape.MapMode = estimate: no layout is loaded. The room is learnt from the LiDAR and
        // published once its polygon closes; the viewer draws the walls as they are born.
        room_concept_.configure_room_estimate();
        calib_->set_room_polygon({}, Eigen::Vector2f::Zero());
        room_initialized_from_svg_polygon_ = false;
        qInfo() << "[room] RoomShape.MapMode = estimate: no layout loaded; the room will be learnt from the LiDAR"
                << "(first pose = origin until the polygon closes and is re-anchored).";
    }
    else
        initialize_room_model_from_svg();
    const std::string pose_path = pose_file_path();
    phase("room_model");
    room_concept_.set_seed_pose_file(pose_path);
    phase("seed_pose");

    // The DSR graph viewer is OPTIONAL now: the layout GUI lives in its own top-level window
    // (see RoomViewer), so the agent runs with Agent.graph=false (no DSRViewer created). When a
    // viewer flag IS enabled we still use it for the graph relayout; every access is null-guarded.
    if (!find_graph_viewer(""))
        qInfo() << "[room] No DSR viewer (Agent.graph=false); layout window runs standalone.";

    // Room polygon for visualizations (viewer outline + camera-projection overlay). Reuse the one
    // the localizer got — re-loading the SVG here would skip the recentring and draw the outline in
    // the un-shifted frame.
    const std::vector<Eigen::Vector2f>& room_polygon_for_viz = calib_->room_polygon();

    // GUI / visualization (2-D viewer, FE plot, camera-projection window + RGB media plane).
    viewer_ = std::make_unique<rc::RoomViewer>(
        G, params, room_polygon_for_viz,
        room_initialized_from_svg_polygon_, room_concept_, epistemic_controller_);

    // AFTER the viewer exists. Registering this beside the ingestor setup (where the camera calib
    // is otherwise configured) put it before make_unique, so the `if (viewer_)` was false and the
    // handler was never installed — a Reset that silently cleared only half the evidence.
    phase("viewer_window");   // the window EXISTS from here; everything after this is visible hang time
    viewer_raw_slot_ = viewer_.get();   // collaborators built before the viewer read through this
    viewer_->set_camera_reset_handler([this]
    {
        mount_->reset_evidence();
        qInfo() << "[camcal] evidence cleared and etc/camera_calib.txt deleted";
    });

    if (auto* w = viewer_->widget())
    {
        connect(w->btn_camera_viz, &QPushButton::clicked, this, [this] { viewer_->show_camera(); });
        connect(w->btn_ricoh_viz,  &QPushButton::clicked, this, [this] { viewer_->show_ricoh(); });
        connect(w->btn_calib_viz, &QPushButton::clicked, this, [this] { viewer_->show_calibration(); });
        connect(w->btn_lidar_points_viz, &QPushButton::toggled, this, [this](bool on) { viewer_->toggle_lidar_points(on); });
        if (auto* v = viewer_->viewer())
            v->set_lidar_points_visible(w->btn_lidar_points_viz->isChecked());
    }

    // ── DSR scene-graph writer: resolve graph ids + body dims ──────────────
    scene_graph_->check_init_graph_is_valid();

    // Ensure a clean startup: if a stale room node exists from previous runs,
    // remove it so the room is recreated only after localization is stable.
    // In PreserveBootstrapRoom (static-room) mode, skip this: the room/table are a
    // pre-seeded static prior in the bootstrap graph and must NOT be deleted.
    if (params.PRESERVE_BOOTSTRAP_ROOM)
        qInfo() << "[room] PreserveBootstrapRoom=true: skipping start cleanup; adopting pre-seeded room/table as a static prior.";
    else
        scene_graph_->cleanup_room_graph_nodes();

    // ── Connect DSR signals ────────────────────────────────────────────────
    // NEVER Qt::DirectConnection on DSR update signals: it runs the slot on the raw
    // FastDDS reader thread and corrupts the heap under peer churn (smashed AgentInfo
    // heartbeat). modify_node_slot is empty now (LiDAR is media-only), so we simply do
    // NOT connect update_node_signal. The attrs slot below uses the default (Queued)
    // connection — it runs on the main thread and only reads velocity/odometry attrs.
    connect(G.get(), &DSR::DSRGraph::update_node_attr_signal, this, &SpecificWorker::modify_node_attrs_slot);
    // connect(G.get(), &DSR::DSRGraph::update_edge_attr_signal, this, &SpecificWorker::modify_edge_attrs_slot);
    // connect(G.get(), &DSR::DSRGraph::del_edge_signal,         this, &SpecificWorker::del_edge_slot);
    // connect(G.get(), &DSR::DSRGraph::del_node_signal,         this, &SpecificWorker::del_node_slot);

    // Publish corrections the INSTANT the localizer produces them, on the LOCALIZER thread itself.
    // FIX 2026-09-03: was marshalled to the main thread via Qt::QueuedConnection under the belief
    // that DSR writes must happen there. DSR's own API (dsr_api.h) guards the graph with an internal
    // std::shared_mutex and is written to be called from any thread/process concurrently -- the
    // marshal was a local caution, not a real requirement. Removed here to stop this publish waiting
    // behind whatever the main thread's Qt event loop is doing that tick (viewer redraw, image-edge
    // pumping, camera projection) before external consumers (controller, retina, ...) can see it.
    // compute()'s own redundant call to maybe_publish_corrected_pose() is REMOVED below (see FIX
    // 2026-09-03 there) -- keeping both would race on this callback's now-direct thread against
    // compute()'s main-thread one, hitting the same non-atomic dedup state and the same ofstream
    // members inside dsr_update_calibration/dsr_update_affordance without a lock.
    room_concept_.set_on_result_ready([this]() { pose_pub_->maybe_publish_corrected_pose(); });

    // LiDAR is pumped synchronously from compute() (no ingest thread); just start the localizer.
    room_concept_.start();
    phase("localiser_thread");

    // ── Presence coordinator ────────────────────────────────────────────────
    presence_coordinator_.configure(configLoader, G, static_cast<std::uint32_t>(agent_id));
    // Colour this agent's node in the graph view by its live health: the coordinator already
    // publishes the presence lifecycle; this adds the external FSM axis (Initialize/Compute/
    // Emergency/Restore). Generic discovery via objectName(), so genericworker regeneration
    // cannot break it.
    presence_coordinator_.attach_state_machine(&statemachine);
    AgentPresenceCoordinator::Policy presence_policy;
    presence_policy.set_local_ready_false_on_waiting_enter = false;
    presence_policy.set_local_ready_true_on_operating_enter = false;
    presence_policy.set_local_ready_false_on_degraded_enter = false;
    presence_coordinator_.set_policy(presence_policy);
    presence_coordinator_.set_transition_hooks({
        // Required peers being up is necessary but NOT sufficient: without the LiDAR media plane the
        // localizer has no evidence and can never stabilize, so advancing to Operating would just look
        // like a silent hang. Hold in Waiting until the stream is advertised; on_waiting_loop re-checks
        // every tick and advances the moment it appears (same shape as the retina's room gate).
        // Declines SILENTLY when the stream is missing — this fires on every presence event, so
        // logging here would spam. on_waiting_loop owns the (throttled) "why are we still waiting" line.
        .request_presence_ready = [this]()
        {
            if (lidar_stream_ready())
                emit presenceReady();
        },
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
            QTimer::singleShot(0, this, [this]() { presence_coordinator_.set_local_ready(false); });
            const auto missing = presence_coordinator_.missing_required_names();
            if (!missing.empty())
            {
                QString m;
                for (const auto &label : missing)
                    m += " " + QString::fromStdString(label);
                qInfo() << "  missing:" << m;
            }
        },
        .on_waiting_loop = [this]()
        {
            if (shutting_down_)
                return;
            const bool peers_ready = presence_coordinator_.all_required_ready();
            std::string why;
            const bool lidar_ready = lidar_stream_ready(&why);
            if (peers_ready and lidar_ready)
            {
                std::println("[SM] Waiting: peers ready and LiDAR stream '{}' advertised -> Operating", why);
                emit presenceReady();
                return;
            }
            // Say WHY we are stuck, on a throttle. This is the line that was missing: a room agent
            // with no lidar producer used to sit in Waiting printing nothing at all.
            const auto now = QDateTime::currentMSecsSinceEpoch();
            if (params.LIDAR_WAIT_LOG_PERIOD_MS > 0 and
                now - last_wait_log_ms_ >= params.LIDAR_WAIT_LOG_PERIOD_MS)
            {
                last_wait_log_ms_ = now;
                std::string missing;
                for (const auto& label : presence_coordinator_.missing_required_names())
                    missing += " " + label;
                std::println("[SM] Waiting — peers{}{} | lidar: {}",
                             peers_ready ? " OK" : " MISSING:",
                             peers_ready ? std::string{} : missing,
                             lidar_ready ? ("OK (" + why + ")") : why);
            }
        },
        .on_operating_enter = [this]()
        {
            operating_since_ms_   = QDateTime::currentMSecsSinceEpoch();
            lidar_stall_reported_ = false;
            qInfo() << "[SM] -> Operating: all required constraints satisfied";
            QTimer::singleShot(0, this, [this]() { presence_coordinator_.set_local_ready(true); });
            if (!room_concept_.is_running())
            {
                qWarning() << "[SM] Operating enter: RoomConcept thread was not running, starting it";
                room_concept_.start();
            }
            // Start the dedicated LiDAR ingest thread ONLY now (post graph-join): it reads the DSR graph
            // (subscriber discovery + inner_eigen transform), which is unsafe during the join. Idempotent.
            if (lidar_ingestor_)
                lidar_ingestor_->start();
                if (imu_ingestor_) imu_ingestor_->start();
                // Same rule as the LiDAR thread: only post graph-join, because bring-up reads the graph.
                if (calib_) calib_->start();
        },
        .on_operating_loop = [this]()
        {
            const auto run_operating_tick = [this]()
            {
                operating_compute_queued_.store(false, std::memory_order_release);
                // Stream-stall guard: acting on a dead LiDAR means integrating stale evidence, which
                // CLAUDE.md forbids. Drop back to Waiting (via Degraded, whose only transition is
                // ->Waiting) so the gate above can re-admit us when the producer returns.
                if (std::int64_t age = 0; not lidar_stall_reported_ and lidar_stream_stalled(&age))
                {
                    lidar_stall_reported_ = true;
                    degraded_from_lidar_  = true;
                    std::println("[SM] Operating -> Waiting: LiDAR stream STALLED ({}) — "
                                 "not localizing on stale evidence",
                                 age < 0 ? std::string("no sweep ever arrived")
                                         : std::format("last sweep {} ms ago", age));
                    emit presenceLost();
                    return;
                }
                compute();
                if (auto v = find_graph_viewer(""); v)
                    v->set_external_fps(states.at("Operating")->getActualFps());
            };

            if (QThread::currentThread() == this->thread())
            {
                run_operating_tick();
                return;
            }

            if (!operating_compute_queued_.exchange(true, std::memory_order_acq_rel))
                QMetaObject::invokeMethod(this, run_operating_tick, Qt::QueuedConnection);
        },
        .on_degraded_enter = [this]()
        {
            if (shutting_down_)
                return;
            // DEBOUNCE — do NOT tear down on entry. A transient required-peer flap (startup
            // handshake, brief DSR node churn, a peer restarting) momentarily fires presenceLost and
            // then recovers; the old code deleted room's OWN agent node + stopped RoomConcept here,
            // then "recovered" to Operating in a half-broken state and exited anyway. Wait a grace
            // period and shut down only if a required peer is STILL genuinely missing.
            constexpr int REQUIRED_LOSS_GRACE_MS = 3000;
            // A LiDAR stall also routes through here (Degraded is the only way back to Waiting), but it
            // is NOT a peer loss — say so, otherwise the log blames the wrong thing. The grace timer
            // below then finds all peers present and correctly declines to shut down.
            if (degraded_from_lidar_)
            {
                degraded_from_lidar_ = false;
                qInfo() << "[SM] -> Degraded (LiDAR stall, peers intact) — passing through to Waiting";
            }
            else
                qInfo() << "[SM] -> Degraded: required peer lost —" << REQUIRED_LOSS_GRACE_MS << "ms grace before shutdown";
            QTimer::singleShot(REQUIRED_LOSS_GRACE_MS, this, [this]()
            {
                if (shutting_down_)
                    return;
                if (presence_coordinator_.all_required_ready())
                {
                    qInfo() << "[SM] required peers recovered during grace — staying alive";
                    return;
                }
                qWarning() << "[SM] required peer still missing after grace — shutting down cleanly";
                request_shutdown();   // does cleanup + crash-free _Exit (terminal)
            });
        },
    });
    presence_coordinator_.start();

    // ── Wire mouse-driven pose reset ───────────────────────────────────────
    if (auto* v = viewer_->viewer())
    {
        connect(v, &rc::Viewer2D::robot_moved,  this, [this](QPointF p){ viewer_->on_robot_moved(p); });
        connect(v, &rc::Viewer2D::robot_rotate, this, [this](QPointF p){ viewer_->on_robot_rotated(p); });
    }


    restore_window_settings();
}

///////////////////////////////////////////////////////////////////////////////
// Compute the RT publish rate over a ~1 s window and show it in the custom widget (corrected +
// predicted blocks/s). Counters accumulate every compute tick; the readout refreshes once per second.
// The widget write is done only on the GUI thread (Qt requirement); counters still reset so the rate
// stays a true 1 s average.
// The pose publishing that lived here — corrected at LiDAR rate, predicted at IMU rate, the kinematic
// clamp, the jump log and the shared trace, 521 lines across two threads — is now rc::PosePublisher
// (src/pose_publisher.{h,cpp}).

// The ground-truth grading that lived here — including the self-checking heading-convention test — is
// now rc::GroundTruthLog (src/ground_truth_log.{h,cpp}).

