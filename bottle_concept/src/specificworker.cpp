/*
 *    Copyright (C) 2026 by RoboComp CORTEX Team
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

/**
 * SpecificWorker — bottle_concept agent: RoboComp lifecycle + presence protocol +
 * orchestration only. compute() wires the collaborators into the per-cycle pipeline:
 *   mask_ingestor_ (read masks) → tracker_ (birth/associate/death of bottle nodes) →
 *   fitter_ (per-bottle free-energy fit + write-back) → evaluator_ (validation drivers).
 */

#include "specificworker.h"
#include "../../common/nbv/graph_obstacles.h"   // rc::nbv::collect_graph_obstacles — shared, DSR-side
#include <cstdlib>   // std::_Exit for the crash-free terminal shutdown
#include <thread>    // std::this_thread::sleep_for — let DDS flush before _Exit
#include <chrono>

#include "bottle_dof.h"   // rc::kBottleDofs — names/units for the BeliefInspector rows

#include <algorithm>
#include <array>
#include <cmath>
#include <format>    // stall-transition log formatting (std::println on cout, survives Verbose=false)
#include <limits>
#include <numbers>
#include <print>
#include <sstream>

#include <QCoreApplication>
#include <QTimer>
#include <QSettings>   // persist the standalone dashboard window geometry
#include <QByteArray>
#include <QDateTime>   // wall-clock ms for the primary-input stream gate (operating_since_ms_, stall grace)

#include "../../common/bearing_confirm/bearing_confirm.h"   // Part C: RGB-360 bearing → live-instance confirm

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

    cfg_ = rc::load_bottle_config(configLoader);

    // AI2 belief self-test (isolated Eigen unit test) — runs once at startup so a broken generative
    // model / engine wiring (incl. the silhouette radius factor) is caught before the live loop.
    rc::BottleBelief::self_test();

#ifdef HIBERNATION_ENABLED
    hibernationChecker.start(500);
#endif

    // Agent-presence state machine: Waiting → Operating → Degraded.
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

    // Compute → Waiting on start
    states["Compute"]->addTransition(states["Compute"].get(), SIGNAL(entered()), states["Waiting"].get());
    // Waiting → Operating when all required peers are ready
    states["Waiting"]->addTransition(this, SIGNAL(presenceReady()), states["Operating"].get());
    // Operating → Degraded when a required peer is lost
    states["Operating"]->addTransition(this, SIGNAL(presenceLost()), states["Degraded"].get());
    // Degraded → Waiting immediately (self-kill scheduled inside degraded_enter)
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
    std::print("bottle_concept: SpecificWorker destroyed.\n");
}

void SpecificWorker::request_shutdown()
{
    if (shutting_down_.exchange(true))
        return;

    // Persist window geometry + dock layout (splitter positions) while the QMainWindow is still intact.
    // terminal_shutdown() → _Exit() bypasses Qt's aboutToQuit save hook, so we must save here explicitly
    // (mirrors table_concept).
    save_window_settings();
    save_dashboard_geometry();   // the standalone dashboard is not in `windows`, so save it explicitly

    // Sever graph callbacks BEFORE any teardown. On Ctrl+C a del_node delta can be
    // delivered from a DSR/DDS internal thread and invoke del_node_slot (instances_.erase)
    // on this already-destructing object — the exit segfault. Dropping inner_eigen_ here
    // (while G is still fully alive) also unsubscribes its internal graph signals cleanly.
    if (G)
        disconnect(G.get(), nullptr, this, nullptr);
    // Drop the LiDAR media subscriber BEFORE tearing down the graph/inner_eigen it reads (it holds a raw
    // inner_eigen_ pointer + a DDS reader) — mirrors room_concept's teardown ordering.
    lidar_ingestor_.reset();
    inner_eigen_.reset();

    cleanup_owned_nodes();
}

void SpecificWorker::terminal_shutdown()
{
    static std::atomic<bool> terminating{false};
    if (terminating.exchange(true))
        return;   // _Exit is coming; never run this twice

    // 1) Sever our graph callbacks, delete our owned DSR nodes (publishes del-deltas) and notify
    //    peers. Idempotent (shutting_down_ guard).
    request_shutdown();

    // 2) Cleanly remove THIS agent's DDS participant and entities from the shared graph. This is the
    //    crucial step that a bare _Exit skips: without it, peers (voxelizer) keep seeing our
    //    half-deleted 'bottle_*' node (node present, room->bottle RT edge gone) and SEGV walking the
    //    RT tree, and a fast restart hits "agent id 10 already connected". DSRGraph::reset() runs
    //    remove_participant_and_entities() (the clean "Publisher unmatched" path) WITHOUT touching
    //    the Ice communicator, so it does not trip the IceUtil::Mutex teardown abort.
    if (G)
    {
        try { G->reset(); }
        catch (...) { /* best-effort: we are exiting regardless */ }
    }

    // 3) Give the DDS writers a brief window to actually transmit the entity removals + participant
    //    departure to peers before the process vanishes, so no peer is left reading a stale node.
    std::cout.flush();
    std::cerr.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    // 4) Hard-exit, skipping C++ static destruction and the Ice::Application communicator teardown
    //    that races an Ice worker thread against IceUtil::Mutex destruction (ThreadSyscallException
    //    EINVAL) — bottle is the only agent with an active OUTGOING Ice client proxy, so the only one
    //    that hits it. State is persisted, graph presence cleanly removed; the OS reclaims the rest.
    std::_Exit(EXIT_SUCCESS);
}

// Persist/restore the standalone dashboard window's geometry. The generated save_window_settings()
// only covers the QMainWindow(s) in `windows`; our extracted top-level widget is separate, so we
// carry its own QSettings entry (mirrors room_concept's RoomViewer).
void SpecificWorker::restore_dashboard_geometry()
{
    if (not dashboard_window_)
        return;
    QSettings settings(QStringLiteral("RoboComp"), QStringLiteral("bottle_concept"));
    const QByteArray geom = settings.value(QStringLiteral("DashboardWindow_geometry")).toByteArray();
    if (not geom.isEmpty())
        dashboard_window_->restoreGeometry(geom);
    else
        dashboard_window_->resize(1180, 900);
}

void SpecificWorker::save_dashboard_geometry() const
{
    if (not dashboard_window_)
        return;
    QSettings settings(QStringLiteral("RoboComp"), QStringLiteral("bottle_concept"));
    settings.setValue(QStringLiteral("DashboardWindow_geometry"), dashboard_window_->saveGeometry());
    settings.sync();
}

void SpecificWorker::initialize()
{
    std::print("bottle_concept: initialize()\n");

    // Shadow-mode birth/death record (CONCEPT_AGENT_LIFECYCLE.md §4.2). Recording only — see
    // log_phantom_event(). Truncating: one file per run.
    phantom_log_.open("etc/bottle_phantom_events.csv");
    GenericWorker::initialize();

    if (not G)
    {
        qWarning() << "bottle_concept: DSR graph not available in initialize()";
        return;
    }

    // Agent-presence protocol wiring.
    presence_coordinator_.configure(configLoader, G, static_cast<std::uint32_t>(agent_id));
    // Colour this agent's node in the graph view by its live health: the coordinator already
    // publishes the presence lifecycle; this adds the external FSM axis (Initialize/Compute/
    // Emergency/Restore). Generic discovery via objectName(), so genericworker regeneration
    // cannot break it.
    presence_coordinator_.attach_state_machine(&statemachine);
    presence_coordinator_.set_transition_hooks({
        // Peers-ready is necessary but NOT sufficient: also require the masks producer to be LIVE (a fresh
        // frame within the timeout — not merely a persisting `masks` node, which would re-admit into an
        // instant re-stall). Declines silently — this fires on every presence event; on_waiting_loop pumps
        // the ingest and re-polls until the stream is actually producing.
        .request_presence_ready = [this]() { if (masks_stream_live()) emit presenceReady(); },
        .request_presence_lost  = [this]() { emit presenceLost(); },
    });
    presence_coordinator_.set_peer_hooks({
        .on_peer_restarted = [](std::uint32_t id)
        {
            qInfo() << "[Presence] peer" << id << "restarted";
        },
        .on_optional_peer_lost = [this](const std::string& name, std::uint32_t id)
        {
            on_optional_peer_lost(name, id);
        },
        .on_optional_peer_ready = [this](const std::string& name, std::uint32_t id)
        {
            on_optional_peer_ready(name, id);
        },
    });
    presence_coordinator_.set_lifecycle_hooks({
        .on_waiting_enter = [this]()
        {
            const auto missing = presence_coordinator_.missing_required_names();
            if (missing.empty())
                qInfo("[SM] -> Waiting");
            else
            {
                QString m;
                for (const auto& label : missing)
                    m += " " + QString::fromStdString(label);
                qInfo() << "[SM] -> Waiting (missing:" << m.trimmed() << ")";
            }
        },
        .on_waiting_loop = [this]()
        {
            if (shutting_down_)
                return;
            // Pump the masks ingest WHILE Waiting: bottle polls a graph node (no free-running ingest thread
            // like room's LiDAR), so producer liveness (ms_since_last_frame) only advances if we refresh
            // here too. Without it the agent could neither detect the producer coming back nor avoid the
            // re-admit→instant-re-stall flap. It also lets admission key on real freshness, not node-exists.
            if (mask_ingestor_)
                mask_ingestor_->refresh();
            const bool peers_ready = presence_coordinator_.all_required_ready();
            const bool masks_live  = masks_stream_live();
            if (peers_ready and masks_live)   // promote only when peers AND a live masks producer are both up
            {
                emit presenceReady();
                return;
            }
            const auto now = QDateTime::currentMSecsSinceEpoch();
            if (now - last_wait_log_ms_ >= 2000)   // throttle the "why still Waiting" line
            {
                last_wait_log_ms_ = now;
                const auto age = mask_ingestor_ ? mask_ingestor_->ms_since_last_frame() : -1;
                std::println("[SM] Waiting — peers {} | masks {} (age {} ms)",
                             peers_ready ? "OK" : "MISSING", masks_live ? "LIVE" : "stale",
                             age < 0 ? std::string("none") : std::to_string(age));
            }
        },
        .on_operating_enter = [this]()
        {
            // Primary-input stream-gate resets: baseline for the cold-start stall grace + re-arm the one-shot.
            operating_since_ms_   = QDateTime::currentMSecsSinceEpoch();
            masks_stall_reported_ = false;
            qInfo("[SM] -> Operating: all required peers present");
            // One-time startup sweep: remove leftover affordance nodes from a PREVIOUS run so a fresh
            // create doesn't collide and get a DSR-generated name. Guarded — on a RE-entry to Operating
            // (after a transient required-peer flap → Degraded → recover) the affordances in the graph
            // are THIS run's live ones, and wiping them every bounce makes them flicker.
            if (not startup_affordance_sweep_done_)
            {
                startup_affordance_sweep_done_ = true;
                remove_stale_affordance_nodes();
            }
        },
        .on_operating_loop = [this]()
        {
            // Primary-input stream gate: a dead masks producer means acting on stale evidence. Demote out
            // of Operating (Operating→Degraded→Waiting) rather than re-integrating frozen frames; the gate
            // re-admits when the producer returns. Belief Σ-aging (a different, belief axis) is untouched.
            if (std::int64_t age = 0; not masks_stall_reported_ and masks_stream_stalled(&age))
            {
                masks_stall_reported_ = true;
                degraded_from_masks_  = true;
                std::println("[SM] Operating -> Waiting: masks stream STALLED ({}) — not integrating stale evidence",
                             age < 0 ? std::string("no frame ever arrived")
                                     : std::format("last frame {} ms ago", age));
                emit presenceLost();
                return;
            }
            compute();
        },
        .on_degraded_enter = [this]()
        {
            if (shutting_down_)
                return;
            // A mask-stream stall (peers intact) routes through Degraded too — but it is RECOVERABLE, not a
            // shutdown cause. Flag it so the grace timer below finds all peers present and declines to exit;
            // the FSM has already bounced Degraded→Waiting, where the admission gate holds until masks return.
            if (degraded_from_masks_)
            {
                degraded_from_masks_ = false;
                qInfo("[SM] -> Degraded (masks stall, peers intact) — passing through to Waiting, re-admit on producer return");
            }
            else
                // DEBOUNCE — do NOT cleanup/exit on entry. A transient required-peer flap (startup
                // handshake, brief DSR node churn, a peer restarting) fires presenceLost momentarily and
                // then recovers; tearing down here deleted our own node and disconnected the graph, then
                // the agent "recovered" into a broken half-shutdown state and later aborted. Instead wait
                // a grace period and only shut down if a required peer is STILL genuinely missing.
                qInfo("[SM] -> Degraded: required peer lost — %d ms grace before shutdown",
                      REQUIRED_LOSS_GRACE_MS);
            QTimer::singleShot(REQUIRED_LOSS_GRACE_MS, this, [this]()
            {
                if (shutting_down_)
                    return;
                if (presence_coordinator_.all_required_ready())
                {
                    qInfo("[SM] required peers recovered during grace — staying alive");
                    return;
                }
                qWarning("[SM] required peer still missing after grace — shutting down cleanly");
                terminal_shutdown();
            });
        },
    });
    presence_coordinator_.start();

    // Route SIGTERM/Ctrl+C (sigwatch -> a.quit() -> aboutToQuit) through the crash-free terminal
    // shutdown so the intentional exit never hits the Ice teardown abort either.
    QObject::connect(QCoreApplication::instance(), &QCoreApplication::aboutToQuit,
                     this, &SpecificWorker::terminal_shutdown, Qt::UniqueConnection);

    rt_api_ = G->get_rt_api();
    inner_eigen_ = G->get_inner_eigen_api();
    gaussian_api_ = std::make_unique<DSR::InnerGaussianAPI>(G.get());
    mask_ingestor_ = std::make_unique<rc::MaskIngestor>(G);
    scene_graph_ = std::make_unique<rc::BottleSceneGraph>(G, rt_api_.get(), inner_eigen_.get(), cfg_,
                                                          [this] { trigger_graph_layout_twopi(); });

    // Part B: consume camera-frame masks + transform through the probabilistic chain. MaskIngestor
    // re-frames the support points (src→fit frame, capture-stamp pinned); the scene-graph adds the
    // localization/chain covariance to the published RT edge via InnerGaussianAPI.
    if (cfg_.masks_use_camera_frame)
        mask_ingestor_->enable_frame_transform(inner_eigen_.get(), cfg_.masks_source_frame, cfg_.masks_target_frame);
    scene_graph_->set_chain_cov_source(gaussian_api_.get(), cfg_.masks_source_frame,
                                       cfg_.rt_cov_add_chain and cfg_.masks_use_camera_frame);

    connect(G.get(), &DSR::DSRGraph::del_node_signal, this, &SpecificWorker::del_node_slot);

    // Remove any "bottle*" cylinder nodes left behind by a previous (crashed) run
    // so this agent always starts from a clean slate.
    remove_owned_bottle_nodes();

    const auto rooms = G->get_nodes_by_type("room");
    if (not rooms.empty())
        room_node_id_ = rooms.front().id();
    else
        qWarning() << "bottle_concept: no room node found at startup";

    // Validation harness (no-op unless an Eval.*/Scene.* flag is set). Table-top lookup injected as a
    // callback so the evaluator stays decoupled from the scene-graph layer.
    evaluator_ = std::make_unique<rc::BottleEvaluator>(
        cfg_, webots2robocomp_proxy, inner_eigen_.get(),
        [this](float bx, float by) { return scene_graph_->find_table_top(bx, by); });

    // Hidden-face next-best-view planner (epistemic affordance).
    epistemic_planner_ = rc::EpistemicPlanner(cfg_.epistemic_obs_distance, cfg_.epistemic_view_info);
    // ONE detector envelope: the far-side viewpoint is the argmax of the same model absence is weighted by,
    // and the published gain is multiplied by P(detect) there — so a hidden-face look the detector could not
    // fire from stops bidding for the drive AROUND the bottle.
    epistemic_planner_.set_detector_envelope(rc::detect::DetectorEnvelope{});
    epistemic_planner_.set_robot_radius(0.30f);   // Shadow's footprint radius
    // ★The camera model is read PER CYCLE at the compute site (rc::nbv::sensor_from_graph),
    // NOT once here: the zed intrinsics are published by robot_concept when frames start
    // arriving, so reading them in initialize() races the producer. Losing that race leaves
    // vfov = 0, which silently collapses the fill model to horizontal-only — the exact bug
    // rc::nbv exists to fix, and it drives the robot nose-to-nose with tall objects.

    // Active-inference fit core (pure belief). Owns the instance map; READS via scene_graph_ but the
    // worker (process_bottle_node) owns the write-back + eval — see the canonical concept-agent loop.
    fitter_ = std::make_unique<rc::BottleFitter>(
        G, inner_eigen_.get(), cfg_,
        mask_ingestor_.get(), scene_graph_.get());

    // YOLO-independent LiDAR range channel: lidar3D media-plane consumer that stages each cycle's sweep in the
    // room frame for the fitter. Dormant (no DDS participant) unless BottleModel.LidarPrecision > 0. Subscriber
    // is brought up lazily on the main thread from compute()::pump(), never here (graph may still be joining).
    lidar_ingestor_ = std::make_unique<rc::BottleLidarIngestor>(G, inner_eigen_.get(), cfg_);

    // ── Live "Bottle Inference" dashboard — its OWN top-level window ───────────────────────────────
    // Extracted from the DSR graph dock (add_custom_widget_to_dock) into a standalone window so it shows
    // even with Agent.graph=false. Mirrors room_concept/kinova_controller and table_concept. TimeSeriesPlot
    // is a plain QWidget (no QOpenGL backing store), safe as a top-level. NOT WA_DeleteOnClose: closing
    // must only HIDE it, or the ts_*_plot_ pointers publish_bottle_diagnostics uses would dangle. A
    // QApplication always exists (generated/main.cpp). Fed each cycle in publish_bottle_diagnostics.
    {
        custom_widget_ = new Custom_widget("Bottle — Free Energy, Surprise, Belief Uncertainty & Residuals");

        auto* series_layout = new QVBoxLayout(custom_widget_->frame_series);
        series_layout->setContentsMargins(0, 0, 0, 0);
        custom_widget_->frame_series->setLayout(series_layout);

        const auto add_plot = [&](rc::TimeSeriesPlot*& plot)
        {
            plot = new rc::TimeSeriesPlot(custom_widget_->frame_series);
            plot->set_visible_window(60.f);
            series_layout->addWidget(plot, 1);
        };
        add_plot(ts_plot_);            // free energy + its baseline
        add_plot(ts_surprise_plot_);   // FE surprise (attention signal) — own panel, much smaller scale
        add_plot(ts_cov_plot_);        // U(Σ) belief uncertainty
        add_plot(ts_res_plot_);        // residual (unexplained) points

        // BELIEF INSPECTOR — the panel that replaced the Σ[cx,cy] trace. A time-series of two variances
        // showed a slice of the belief; this shows ALL of it: every state DOF with its posterior σ and Σ as
        // a correlation heatmap, which is where the structure lives (the bottle's radius is depth-degenerate
        // and its correlation with cz is exactly what the trace could not show). Stretch 2.
        belief_inspector_ = new rc::BeliefInspector(QStringLiteral("belief inspector"),
                                                    custom_widget_->frame_series);
        series_layout->addWidget(belief_inspector_, 2);

        // GenericWorker::initialize() may have started compute() already, so some instances can exist.
        for (auto& [_, inst] : fitter_->instances())
            publish_bottle_diagnostics(inst, inst.prev_free_energy);
    }

    // ── Section 1: evidence-pipeline counter strip ────────────────────────────
    evidence_monitor_ = new rc::EvidenceMonitor(QStringLiteral("bottle_concept — evidence monitor"));

    // ── Combined window: counters (top) over the plots + belief inspector (bottom) ──
    // Identical three-section structure to every other concept agent. Only HIDDEN on close, never deleted
    // (compute() keeps the raw child pointers).
    dashboard_window_ = new QWidget;
    dashboard_window_->setWindowTitle(QStringLiteral("bottle_concept — dashboard"));
    auto* outer = new QVBoxLayout(dashboard_window_);
    outer->setContentsMargins(0, 0, 0, 0);
    // No splitter: the counter strip is two lines of text with nothing to resize, so it simply takes
    // its natural height and the plots + inspector get everything else.
    outer->addWidget(evidence_monitor_, 0);   // section 1 — natural height
    outer->addWidget(custom_widget_, 1);      // sections 2 + 3 — all remaining space

    restore_dashboard_geometry();
    dashboard_window_->show();
}

namespace { constexpr int PLACE_SETTLE_CYCLES = 30; }   // ~settle time after a start-placement move

// SHADOW-MODE birth/death recorder — CONCEPT_AGENT_LIFECYCLE.md §4.2, theory in MODEL_HISTORY.md §4.
// RECORDS ONLY; it can never alter a birth or a removal. NOTE: bottle has NO existence channel (tracker death_frames only) — p_detect is
    // unavailable, so every bottle death is UNATTRIBUTABLE and must not be learned from.
void SpecificWorker::log_phantom_event(std::string_view event, std::uint64_t id, std::string_view name,
                                       float x, float y, const rc::BottleInstance* inst, std::string_view note)
{
    if (not phantom_log_.is_open())
        return;
    rc::history::PhantomEvent e;
    e.event = event; e.id = id; e.name = name; e.x = x; e.y = y; e.note = note;
    // Observer pose → view bearing: the classifier failure is VIEWPOINT-dependent, so the eventual p_FA field
    // is keyed on (world cell × bearing), never place alone.
    if (inner_eigen_)
        if (const auto rtb = inner_eigen_->get_transformation_matrix("room", "body", 0); rtb.has_value())
        {
            const auto& Tm = rtb.value();
            e.robot_x = static_cast<float>(Tm(0, 3));
            e.robot_y = static_cast<float>(Tm(1, 3));
            e.robot_yaw = std::atan2(static_cast<float>(Tm(1, 0)), static_cast<float>(Tm(0, 0)));
            e.view_bearing = std::atan2(e.robot_y - y, e.robot_x - x);
            e.range_m = std::hypot(e.robot_x - x, e.robot_y - y);
        }
    if (inst)
    {
        e.age_cycles    = inst->processed_cycles;
        e.p_detect      = 0.0f;
        e.central_frac  = 0.0f;
        e.in_fov_frac   = (0 > 0) ? 1.0f : 0.0f;
        e.exist_logodds = 0.0f;
    }
    phantom_log_.write(e);
}

void SpecificWorker::compute()
{
    if (not G or not rt_api_)
        return;

    if (room_node_id_ == 0)
    {
        const auto rooms = G->get_nodes_by_type("room");
        if (rooms.empty()) return;
        room_node_id_ = rooms.front().id();
    }

    // One-shot: place the bottle on its arm-side spot BEFORE any fit, then let the scene settle so
    // the voxelizer captures it there before the tracker births the node — a node created from a
    // pre-move camera frame would lock the XY ownership gate at the old pose.
    if (cfg_.place_on_start and not evaluator_->place_done())
    {
        evaluator_->place_bottle_on_start();
        return;
    }
    if (cfg_.place_on_start and place_settle_ < PLACE_SETTLE_CYCLES)
    {
        ++place_settle_;
        return;
    }

    // Evidence-pipeline per-cycle counters (the *_cum fields persist). Producers below add to these; the
    // snapshot is pushed at the end of the cycle.
    ev_g_.births = ev_g_.merges = ev_g_.removals = 0;

    mask_ingestor_->refresh();

    // Stage this cycle's LiDAR sweep (room frame) for the fitter's range factor. clear-then-set so the factor
    // only contributes on cycles with a genuinely fresh sweep (never re-uses a stale one). No-op when off.
    fitter_->clear_lidar_sweep();
    if (lidar_ingestor_ and lidar_ingestor_->pump())
        fitter_->set_lidar_sweep(lidar_ingestor_->sweep_room(), lidar_ingestor_->origin_room());

    run_instance_tracker();   // data-driven birth/associate/death (the only instance-lifecycle path)

    // Robot/camera ego-motion (room frame) for this cycle — the "be-still-to-update" signal. Computed ONCE on
    // the main thread (ts=0 pose diff) before the per-instance loop; run_inference's confirm_only gate reads it.
    fitter_->update_ego_motion();

    // Bottle instances are DSR `object` nodes named "bottle_*" (migrated from type "cylinder").
    for (const auto& node : G->get_nodes_by_type("object"))
        if (node.name().starts_with("bottle"))
            process_bottle_node(node);

    // Validation drivers (Webots) also teleport the bottle, so they are mutually exclusive with the
    // arm-side start placement — skip them when place_on_start owns the bottle pose.
    if (not cfg_.place_on_start)
    {
        // Static-restart takes precedence over the continuous sweep.
        if (cfg_.static_pose_test)
            evaluator_->place_static_test_pose();
        else
            evaluator_->step_move_experiment(fitter_->instances());
    }

    // ── Dashboard: counters + belief inspector, one throttled push ──
    ev_g_.instances    = static_cast<int>(fitter_->instances().size());
    ev_g_.mask_stale   = not mask_ingestor_->packet().valid;
    ev_g_.sweep_points = lidar_ingestor_ ? static_cast<int>(lidar_ingestor_->sweep_room().size()) : 0;
    {
        const auto now = std::chrono::steady_clock::now();
        if (last_compute_tp_.time_since_epoch().count() != 0)
        {
            const float dt = std::chrono::duration<float>(now - last_compute_tp_).count();
            if (dt > 1e-4f)
            {
                const float hz = 1.0f / dt;
                ev_g_.compute_hz = ev_g_.compute_hz > 0.0f ? 0.9f * ev_g_.compute_hz + 0.1f * hz : hz;
            }
        }
        last_compute_tp_ = now;
    }
    refresh_evidence_monitor();   // throttled inside; no-ops when the dashboard was not built
    fps_counter_.print("[bottle_concept Compute]");
}

// Canonical per-node orchestration (mirrors table_concept::process_table_node): the fitter runs the
// pure belief (ensure_instance → observe → run_inference, no DSR writes); the worker owns the DSR
// write-back (scene_graph_->step_write_model) and the eval log.
namespace {
// Two bottles cannot share physical space. Footprint = the circle (cx,cy,radius) in the room plane;
// overlap area as a fraction of the SMALLER circle (1.0 = one circle fully inside the other) lets the
// merge operator collapse two instances fitted to the same bottle.
float circle_overlap_ratio(const rc::BottleState& A, const rc::BottleState& B)
{
    const float ra = std::max(A.radius, 1e-4f), rb = std::max(B.radius, 1e-4f);
    const float d  = std::hypot(A.cx - B.cx, A.cy - B.cy);
    if (d >= ra + rb)           return 0.0f;          // disjoint
    if (d <= std::abs(ra - rb)) return 1.0f;          // smaller fully inside the larger
    const float d1 = (d * d + ra * ra - rb * rb) / (2.0f * d);   // a-centre → radical line
    const float d2 = d - d1;
    const float seg_a = ra * ra * std::acos(std::clamp(d1 / ra, -1.0f, 1.0f))
                        - d1 * std::sqrt(std::max(0.0f, ra * ra - d1 * d1));
    const float seg_b = rb * rb * std::acos(std::clamp(d2 / rb, -1.0f, 1.0f))
                        - d2 * std::sqrt(std::max(0.0f, rb * rb - d2 * d2));
    const float amin  = std::numbers::pi_v<float> * std::min(ra, rb) * std::min(ra, rb);
    return amin > 1e-9f ? std::clamp((seg_a + seg_b) / amin, 0.0f, 1.0f) : 0.0f;
}
}  // namespace

// Collapse instances whose circle footprints overlap (same physical bottle fitted twice): keep the one
// with more integrated fresh evidence, retire the other (affordance + node). Runs before tracking so a
// duplicate is gone before it is fed a mask. Mirrors table_concept::merge_overlapping_instances.
void SpecificWorker::merge_overlapping_instances()
{
    if (cfg_.tracker_merge_overlap <= 0.0f)
        return;
    auto& insts = fitter_->instances();
    if (insts.size() < 2)
        return;

    std::vector<std::uint64_t> ids;
    ids.reserve(insts.size());
    for (auto& [id, _] : insts) ids.push_back(id);

    std::unordered_set<std::uint64_t> removed;
    for (std::size_t i = 0; i < ids.size(); ++i)
    {
        if (removed.count(ids[i])) continue;
        for (std::size_t j = i + 1; j < ids.size(); ++j)
        {
            if (removed.count(ids[j])) continue;
            const auto ia = insts.find(ids[i]), ib = insts.find(ids[j]);
            if (ia == insts.end() or ib == insts.end()) continue;

            const float ratio = circle_overlap_ratio(ia->second.model.state(), ib->second.model.state());
            if (ratio < cfg_.tracker_merge_overlap) continue;

            const bool keep_i = ia->second.matched_frames >= ib->second.matched_frames;
            const std::uint64_t keep = keep_i ? ids[i] : ids[j];
            const std::uint64_t drop = keep_i ? ids[j] : ids[i];
            std::print("bottle_concept: [tracker] MERGE id={} into id={} (circle overlap {:.2f})\n",
                       drop, keep, ratio);
            if (auto it = insts.find(drop); it != insts.end())
                it->second.affordance.remove();
            fitter_->forget_node(drop);
            G->delete_node(drop);
            removed.insert(drop);
            if (drop == ids[i]) break;   // this i is gone; advance to the next i
        }
    }
}

void SpecificWorker::retire_diverged_instances()
{
    if (cfg_.diverged_retire_frames <= 0)
        return;
    auto& insts = fitter_->instances();
    std::vector<std::uint64_t> doomed;
    for (auto& [id, inst] : insts)
        if (inst.frames_diverged >= cfg_.diverged_retire_frames)
            doomed.push_back(id);
    for (const std::uint64_t id : doomed)
    {
        std::print("bottle_concept: [tracker] RETIRE-DIVERGED id={} (unexplained: clutter>{:.0f}% for {} frames)\n",
                   id, 100.0f * cfg_.clutter_diverge_frac, cfg_.diverged_retire_frames);
        // Affordance FIRST (while the instance/id still exists), then the C++ instance, then the DSR node —
        // same ordering as a tracker DEATH so aff_<bottle> is never orphaned.
        if (auto it = insts.find(id); it != insts.end())
        {
            // Shadow-mode death record (§4.2). bottle retires on DIVERGENCE (clutter), not on sensor
            // absence, so p_detect is unavailable — these deaths are UNATTRIBUTABLE by construction and
            // the note records why, so the analysis does not mistake them for classifier phantoms.
            log_phantom_event("DEATH", id, it->second.node_name,
                              it->second.model.state().cx, it->second.model.state().cy, &it->second,
                              "retire-diverged (no existence channel)");
            it->second.affordance.remove();
        }
        fitter_->forget_node(id);
        G->delete_node(id);
    }
}

void SpecificWorker::run_instance_tracker()
{
    merge_overlapping_instances();   // enforce physical exclusion before associating/birthing this cycle
    retire_diverged_instances();     // drop diverged models before they are re-associated/re-fed a mask

    rc::TrackerParams tp;
    tp.gate_mahalanobis = cfg_.tracker_gate_mahalanobis;
    tp.gate_fallback_m  = cfg_.tracker_gate_fallback_m;
    tp.birth_frames     = cfg_.tracker_birth_frames;
    tp.death_frames     = cfg_.tracker_death_frames;
    tp.birth_min_sep_m  = cfg_.tracker_birth_min_sep_m;
    tp.detection_noise_m = cfg_.tracker_detection_noise_m;
    tp.nll_cost         = cfg_.tracker_nll_cost;
    tracker_.set_params(tp);

    // Tracks ← live instances: centre from the fit, XY cov from the AI2 belief's position Σ. The gate is
    // the Mahalanobis innovation S = P + R²I; the belief Σ inflates on stale (look-away) predicts, so the
    // gate widens across a dropout and the re-acquired detection associates instead of spawning a rebirth.
    std::vector<rc::TrackView> tracks;
    tracks.reserve(fitter_->instances().size());
    for (auto& [id, inst] : fitter_->instances())
    {
        rc::TrackView t;
        t.id = id;
        const auto& s = inst.model.state();
        t.xy = {s.cx, s.cy};
        if (inst.ai2_initialized)
        {
            t.cov = inst.ai2_belief.covariance().block<2, 2>(0, 0);
            t.has_cov = true;
        }
        t.expected_visible = inst.expected_visible;   // negative-info death gate: persist out-of-FoV
        tracks.push_back(t);
        inst.assigned_mask_idx = -1;   // cleared; re-set below only if associated this cycle
    }

    // Detections ← this frame's "bottle" mask slices (carry the slice index for the assignment).
    std::vector<rc::DetectionView> dets;
    const auto& pkt = mask_ingestor_->packet();
    if (pkt.valid)
        for (int i = 0; i < static_cast<int>(pkt.slices.size()); ++i)
            if (pkt.slices[i].label == "bottle" and pkt.slices[i].has_depth)   // skip ricoh bearing-only slices (Part B)
                dets.push_back({Eigen::Vector2f(pkt.slices[i].centroid.x(), pkt.slices[i].centroid.y()), i});

    // Part C (confirm): a ricoh no-depth "bottle" bearing that lines up (in azimuth from the robot) with a
    // live instance is evidence it is STILL THERE even when the zed missed it → HOLD its death-miss this
    // cycle (set expected_visible=false, exactly like being out of the zed frustum). No fit, no birth.
    if (cfg_.bearing_confirm_enabled and pkt.valid and not tracks.empty())
    {
        std::vector<rc::BearingDetectionView> bearings;
        for (int i = 0; i < static_cast<int>(pkt.slices.size()); ++i)
            if (pkt.slices[i].label == "bottle" and not pkt.slices[i].has_depth)
                bearings.push_back({pkt.slices[i].azimuth_room_rad, i});

        if (not bearings.empty())
        {
            // Robot room-frame XY (≈ the zed origin in room, matching how the bearing was produced).
            Eigen::Vector2f robot_xy(0.f, 0.f);
            if (const auto p = inner_eigen_->transform("room", Eigen::Vector3d::Zero(), "zed"); p.has_value())
                robot_xy = {static_cast<float>(p->x()), static_cast<float>(p->y())};

            for (const auto& c : rc::confirm_tracks_by_bearing(tracks, bearings, robot_xy,
                                                               cfg_.bearing_confirm_gate_rad))
            {
                tracks[c.track_index].expected_visible = false;   // hold the death-miss (peripheral glance)
                std::print("[bearing] confirm bottle id={} slice={} innov={:.1f}deg (holds death-miss)\n",
                           c.track_id, c.slice_index, c.innovation_rad * 180.0f / std::numbers::pi_v<float>);
            }
        }
    }

    const auto res = tracker_.update(tracks, dets);

    // Diagnostic: how many bottle detections vs instances, and what the tracker decided. Throttled,
    // but always logged on a birth/death. Reveals "only 1 slice" (upstream) vs "2 slices, no birth".
    static int dbg = 0;
    const int n_assigned = static_cast<int>(std::count_if(res.assignment.begin(), res.assignment.end(),
                                                          [](int a){ return a >= 0; }));

    // Evidence-pipeline counters (section 1 of the dashboard). Same fields every concept agent fills.
    ev_g_.mask_frame_id = pkt.valid ? pkt.frame_id : -1;
    ev_g_.total_slices  = pkt.valid ? static_cast<int>(pkt.slices.size()) : 0;
    ev_g_.class_dets    = static_cast<int>(dets.size());
    ev_g_.assigned      = n_assigned;
    ev_g_.discarded     = static_cast<int>(dets.size()) - n_assigned;
    ev_g_.births       += static_cast<int>(res.births.size());
    ev_g_.births_cum   += static_cast<long>(res.births.size());
    ev_g_.removals     += static_cast<int>(res.deaths.size());
    ev_g_.removals_cum += static_cast<long>(res.deaths.size());
    if (++dbg % 30 == 0 or not res.births.empty() or not res.deaths.empty())
    {
        std::print("[tracker] instances={} bottle_dets={} assigned={} unassigned={} births={} deaths={}\n",
                   tracks.size(), dets.size(), n_assigned,
                   static_cast<int>(dets.size()) - n_assigned, res.births.size(), res.deaths.size());
        for (const auto& d : dets)
            std::print("[tracker]   det slice={} xy=({:.2f},{:.2f})\n", d.slice_index, d.xy.x(), d.xy.y());
    }

    // DEATH: retire unsupported instances (never one that was matched this cycle).
    for (const std::uint64_t id : res.deaths)
    {
        std::print("bottle_concept: [tracker] DEATH id={} (unsupported {} frames)\n", id, cfg_.tracker_death_frames);
        // Delete this bottle's affordance node FIRST, while the instance (and its id) still exists.
        // forget_node() only erases the C++ instance; deleting the cylinder below would otherwise
        // orphan aff_<bottle> (parent gone), and the close-time sweep keys on a live cylinder parent.
        if (auto it = fitter_->instances().find(id); it != fitter_->instances().end())
            it->second.affordance.remove();
        fitter_->forget_node(id);
        G->delete_node(id);
    }

    // ASSOCIATE: route each matched detection's mask slice to its instance (read in observe()).
    for (int d = 0; d < static_cast<int>(dets.size()); ++d)
        if (res.assignment[d] >= 0)
        {
            const std::uint64_t id = tracks[res.assignment[d]].id;
            if (auto it = fitter_->instances().find(id); it != fitter_->instances().end())
                it->second.assigned_mask_idx = dets[d].slice_index;
        }

    // BIRTH: spawn an instance from each promoted (persistently-unexplained) detection, seeding the
    // fitter with the detection XY so the model starts AT the bottle (not the same-cycle RT-read default).
    for (const int d : res.births)
    {
        const Eigen::Vector3f& c = pkt.slices[dets[d].slice_index].centroid;
        const auto new_id = scene_graph_->create_instance_from_detection(c, room_node_id_);
        if (new_id != 0)
        {
            fitter_->note_birth(new_id, Eigen::Vector2f(c.x(), c.y()));
            // Shadow-mode birth record (CONCEPT_AGENT_LIFECYCLE.md §4.2): place + viewpoint that
            // produced it, so a phantom that dies young is attributable to both.
            log_phantom_event("BIRTH", new_id, "", c.x(), c.y(), nullptr, "");
        }
    }
}

void SpecificWorker::process_bottle_node(const DSR::Node& node)
{
    fitter_->ensure_instance(node, room_node_id_);
    auto& inst = fitter_->instances().at(node.id());
    ++inst.processed_cycles;

    const auto observation = fitter_->observe(inst, node);
    if (not observation.has_fresh_data and inst.matched_frames < 5)
        return;

    const float free_energy = fitter_->run_inference(inst, observation);

    if (auto node_opt = G->get_node(node.id()); node_opt.has_value())
        scene_graph_->step_write_model(inst, node_opt.value(), free_energy);

    // Epistemic capability: publish/refresh the hidden-face affordance for the controller.
    step_epistemic(inst);

    // Hold the residual count so the dashboard trace keeps its last real value between masks.
    if (observation.has_fresh_data)
        inst.dbg_resid_pts = static_cast<int>(observation.residual_pts.size());

    // Live dashboard (after step_epistemic so last_epistemic_gain is current).
    publish_bottle_diagnostics(inst, free_energy);

    // Eval logs every compute cycle, independent of the graph-write change-gate inside step_write_model.
    evaluator_->log_eval(inst, free_energy);

    inst.prev_free_energy = free_energy;
}

// Feed the live dashboard. Series are registered lazily & idempotently here (instances can be created
// via the graph-signal path before the plots exist). Sampled every cycle so a flat trace = a settled
// belief between fresh masks.
void SpecificWorker::publish_bottle_diagnostics(rc::BottleInstance& inst, float free_energy)
{
    if (not ts_plot_)
        return;   // no graph viewer / dashboard this run

    // U(Σ): the belief's scalar uncertainty = Σ of the posterior stds over position + size — the same
    // definition the other concept agents plot (cf. rc::geom::belief_uncertainty).
    const auto sigma_of = [&](int j) -> float {
        return std::sqrt(std::max(0.0f, inst.ai2_belief.covariance()(j, j)));
    };
    const float u_sigma = inst.ai2_initialized
                        ? sigma_of(0) + sigma_of(1) + sigma_of(3) + sigma_of(4)   // cx,cy,radius,height
                        : 0.0f;

    ts_plot_->add_series(inst.node_name + "_fe", QColor(255, 170, 0), 1.1f);
    ts_plot_->add_point (inst.node_name + "_fe", free_energy);
    // FE BASELINE on the SAME panel (same units): the FE lifting ABOVE the grey baseline IS the surprise,
    // shown visually. The smoothed gap gets its own panel (it lives on a much smaller scale).
    ts_plot_->add_series(inst.node_name + "_base", QColor(140, 140, 140), 0.9f);
    if (ts_surprise_plot_)
        ts_surprise_plot_->add_series(inst.node_name + "_surprise", QColor(255, 60, 60), 1.3f);
    if (inst.fe_baseline >= 0.0f)   // skip the uninitialised (-1) baseline before the first fit
    {
        ts_plot_->add_point(inst.node_name + "_base", inst.fe_baseline);
        if (ts_surprise_plot_)
            ts_surprise_plot_->add_point(inst.node_name + "_surprise", inst.fe_surprise);
    }
    if (ts_cov_plot_)
    {
        ts_cov_plot_->add_series(inst.node_name + "_cov", QColor(0, 190, 255), 1.1f);
        ts_cov_plot_->add_point (inst.node_name + "_cov", u_sigma);
    }
    if (ts_res_plot_)
    {
        // dbg_resid_pts HOLDS its last fresh-frame value, so the line doesn't crash to 0 between masks.
        ts_res_plot_->add_series(inst.node_name + "_res", QColor(170, 80, 255), 1.1f);
        ts_res_plot_->add_point (inst.node_name + "_res", static_cast<float>(inst.dbg_resid_pts));
    }

    // (The dimensions (r,h) and posterior-σ traces that used to live here are gone: the BeliefInspector
    // below shows every DOF's value AND its σ live, next to the whole correlation structure.)
}

// ─── Belief inspector ────────────────────────────────────────────────────────
//
// Build the per-instance BELIEF snapshot (state, Σ, scalar gauges) and push it to the bottom panel,
// throttled to ~5 Hz. Deliberately NOT hung off publish_bottle_diagnostics: that runs per-node inside the
// publish loop and early-returns on a missing plot, whereas the inspector wants one pass over ALL live
// instances. Main-thread only.
// Section 1: push this cycle's evidence-pipeline counters, then (same tick, same data) the belief
// inspector — so the two sections can never show different cycles. Throttled to ~5 Hz.
void SpecificWorker::refresh_evidence_monitor()
{
    if (not evidence_monitor_)
        return;
    const auto tick = std::chrono::steady_clock::now();
    if (last_monitor_tp_.time_since_epoch().count() != 0
        and std::chrono::duration<float>(tick - last_monitor_tp_).count() < 0.2f)
        return;
    last_monitor_tp_ = tick;

    evidence_monitor_->update_view(ev_g_);
    refresh_belief_inspector();
}

void SpecificWorker::refresh_belief_inspector()
{
    if (not belief_inspector_)
        return;
    const auto now = std::chrono::steady_clock::now();

    std::vector<rc::BeliefCard> cards;
    cards.reserve(fitter_->instances().size());
    for (const auto& [id, inst] : fitter_->instances())
    {
        rc::BeliefCard c;
        c.node = inst.node_name;

        // A bottle is a yaw-symmetric cylinder: no orientation DOF and no discrete modes, so Σ is the plain
        // posterior (there is no covariance_reported() to fold a mode entropy into).
        const auto& S = inst.ai2_belief.covariance();
        const auto& s = inst.ai2_belief.state();
        const std::array<float, rc::BottleBelief::N> v = {s.cx, s.cy, s.cz, s.radius, s.height};
        for (int j = 0; j < rc::BottleBelief::N; ++j)
            c.dofs.push_back({rc::kBottleDofs[j].name, rc::kBottleDofs[j].unit, v[j],
                              std::sqrt(std::max(0.0f, S(j, j))), rc::kBottleDofs[j].sigma_star});

        // Row-major copy, filled explicitly: Eigen stores column-major, and while Σ is symmetric today an
        // implicit .data() copy would silently transpose if that ever stopped being true.
        constexpr int N = rc::BottleBelief::N;
        c.cov.resize(N * N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                c.cov[i * N + j] = S(i, j);

        // prev_free_energy is seeded to FLT_MAX, not 0 — pass NaN before the first fit so the gauge reads
        // "-" instead of a nonsense 3.4e38.
        c.s.fe          = inst.prev_free_energy < std::numeric_limits<float>::max()
                        ? inst.prev_free_energy
                        : std::numeric_limits<float>::quiet_NaN();
        c.s.fe_baseline = inst.fe_baseline;
        c.s.fe_surprise = inst.fe_surprise;
        // The bottle carries no existence log-odds (it relies on the tracker's negative-information death),
        // so logodds/p_exists stay NaN and the card prints "-" rather than a fabricated probability.
        c.s.age_s       = inst.last_belief_touch.time_since_epoch().count() == 0
                        ? -1.0f
                        : std::chrono::duration<float>(now - inst.last_belief_touch).count();
        c.s.since_det   = inst.frames_since_detection;
        c.s.initialized = inst.ai2_initialized;
        cards.push_back(std::move(c));
    }
    belief_inspector_->update_view(cards);
}

// Publish/refresh the "go see the hidden face" affordance (mirrors table_concept::step_epistemic). The
// node persists and re-offers every cycle; a low ΔH is published as-is so the controller's EFE
// selection won't pick a well-seen bottle (belief→knowledge governor without deleting the node).
void SpecificWorker::step_epistemic(rc::BottleInstance& inst)
{
    if (inst.epistemic_cooldown > 0)
        --inst.epistemic_cooldown;

    // Controller-completion hold: when the controller completes (active=false, pending=false), keep the
    // node but suppress its gain for a cooldown so it isn't immediately re-claimed before the belief settles.
    if (const auto aid = inst.affordance.node_id(); aid != 0)
        if (auto an = G->get_node(aid); an.has_value())
        {
            const bool a = G->get_attrib_by_name<active_att>(an.value()).value_or(false);
            const bool p = G->get_attrib_by_name<epistemic_pending_att>(an.value()).value_or(true);
            if (not a and not p and inst.epistemic_cooldown == 0)
            {
                inst.epistemic_cooldown = cfg_.epistemic_cooldown_cycles;
                std::print("[{}] controller completed affordance → hold {} cycles (node kept, gain suppressed)\n",
                           inst.node_name, cfg_.epistemic_cooldown_cycles);
            }
        }

    // ZED origin in the room frame — defines which arc of the bottle is hidden from the camera.
    Eigen::Vector2f camera_xy(std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::quiet_NaN());
    if (inner_eigen_)
        if (const auto c = inner_eigen_->transform("room", Mat::Vector3d(0.0, 0.0, 0.0), "zed", 0);
            c.has_value())
            camera_xy = Eigen::Vector2f(static_cast<float>(c->x()), static_cast<float>(c->y()));

    if (not inst.ai2_initialized)
        return;   // belief not yet seeded (no fresh frame) → no NBV this cycle
    auto prop = epistemic_planner_.compute(inst.ai2_belief, camera_xy, cfg_.ai2_sigma_base_m,
                                           rc::nbv::sensor_from_graph(*G, inner_eigen_.get()),
                                           rc::nbv::collect_graph_obstacles(*G, inner_eigen_.get(), inst.node_id));
    if (not prop.valid or not prop.is_finite())
        return;   // no camera pose / degenerate ray this cycle → leave the existing affordance untouched

    if (inst.epistemic_cooldown > 0)
        prop.epistemic_gain = 0.0f;

    inst.last_epistemic_gain = prop.epistemic_gain;   // expose to the dashboard

    const auto affordance_node_before = inst.affordance.node_id();
    inst.affordance.update(prop);
    if (affordance_node_before == 0 and inst.affordance.node_id() != 0)
        trigger_graph_layout_twopi();
    inst.epistemic_pending = true;

    log_epistemic_csv(inst, prop, camera_xy);   // gated CSV: ΔH + viewpoint + affordance state (fresh here)
}

// Optional gated CSV of the epistemic/affordance evolution (no-op unless Epistemic.CsvPath is set).
// One row per cycle with a valid proposal: the published ΔH (post-cooldown suppression), the far-side
// target, the camera + bottle positions that define the hidden side, and the affordance protocol state.
void SpecificWorker::log_epistemic_csv(const rc::BottleInstance& inst,
                                       const rc::EpistemicProposal& prop,
                                       const Eigen::Vector2f& camera_xy)
{
    if (cfg_.epistemic_csv_path.empty())
        return;

    if (not epistemic_csv_.is_open())
    {
        epistemic_csv_.open(cfg_.epistemic_csv_path, std::ios::out | std::ios::trunc);
        if (not epistemic_csv_.is_open())
        {
            std::print("bottle_concept: [epistemic] cannot open CSV '{}'\n", cfg_.epistemic_csv_path);
            cfg_.epistemic_csv_path.clear();   // disable further attempts
            return;
        }
        epistemic_csv_ << "cycle,node,gain,pending,cooldown,aff_state,aff_node,"
                          "target_x,target_y,target_yaw,cam_x,cam_y,bottle_cx,bottle_cy,radius,sigma_r_mm\n";
    }

    const auto& s = inst.model.state();
    const float sigma_r = inst.ai2_initialized
                              ? 1000.0f * std::sqrt(std::max(0.0f, inst.ai2_belief.covariance()(3, 3))) : -1.0f;   // radius (mm)
    epistemic_csv_ << inst.processed_cycles << ',' << inst.node_name << ','
                   << prop.epistemic_gain << ',' << (inst.epistemic_pending ? 1 : 0) << ','
                   << inst.epistemic_cooldown << ','
                   << rc::BottleAffordance::state_name(inst.affordance.state()) << ','
                   << inst.affordance.node_id() << ','
                   << prop.epistemic_target_x_m << ',' << prop.epistemic_target_y_m << ','
                   << prop.epistemic_target_yaw_rad << ','
                   << camera_xy.x() << ',' << camera_xy.y() << ','
                   << s.cx << ',' << s.cy << ',' << s.radius << ',' << sigma_r << '\n';
    epistemic_csv_.flush();   // flush each row so a plot can tail the file during a live run
}

void SpecificWorker::del_node_slot(std::uint64_t id)
{
    if (fitter_)
    {
        // If the deleted node was an instance's affordance node (controller satisfied / external delete),
        // reset its state machine so it re-creates on the next epistemic cycle.
        for (auto& [_, inst] : fitter_->instances())
            inst.affordance.on_node_deleted(id);
        fitter_->forget_node(id);
    }
    // A node left the graph — re-run the twopi layout so the view stays coherent.
    trigger_graph_layout_twopi();
}

void SpecificWorker::modify_node_attrs_slot(std::uint64_t id, const std::vector<std::string>&)
{
    // Track controller-owned protocol transitions on each instance's affordance node (active/pending).
    if (fitter_)
        for (auto& [_, inst] : fitter_->instances())
            inst.affordance.on_node_modified(id);
}

void SpecificWorker::emergency()
{
    std::print("bottle_concept: emergency()\n");
}

void SpecificWorker::restore()
{
    std::print("bottle_concept: restore()\n");
}

int SpecificWorker::startup_check()
{
    std::print("bottle_concept: startup_check()\n");
    QTimer::singleShot(200, QCoreApplication::instance(), SLOT(quit()));
    return 0;
}
