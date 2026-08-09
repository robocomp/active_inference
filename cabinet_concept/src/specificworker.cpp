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
 * specificworker.cpp — cabinet_concept agent orchestration.
 *
 * Per compute() cycle: ingest the ZED YOLO "cabinet" masks, associate them to cabinet instances with the shared
 * InstanceTracker (birth / associate / merge; ricoh 360 detections are bearing-only and only raise attention),
 * run one AI2 recursive-Laplace belief update (CabinetBelief) per assigned slice via process_cabinet_node, write
 * the fitted pose+geometry back to DSR (RT edge + dims + mesh + covariance), and emit epistemic action
 * proposals when a cabinet stays under-observed. Also feeds the standalone belief dashboard + evidence monitor
 * windows. The fit core is rc::CabinetFitter, perception rc::MaskIngestor, DSR I/O rc::CabinetSceneGraph. See
 * CABINET.md for the belief/fit core.
 */

#include "specificworker.h"
#include "../../common/nbv/graph_obstacles.h"   // rc::nbv::collect_graph_obstacles — shared, DSR-side
#include "cabinet_geometry.h"   // rc::geom pure footprint/uncertainty helpers

#include <print>
#include <format>    // stall-transition log formatting (std::println on cout, survives Verbose=false)
#include <cstdlib>   // std::_Exit — crash-free terminal shutdown
#include <thread>    // brief DDS flush before _Exit
#include <chrono>
#include <iostream>  // std::cout/cerr flush
#include <QSettings>   // persist the standalone dashboard window geometry
#include <QByteArray>
#include <QDateTime>   // wall-clock ms for the primary-input stream gate (operating_since_ms_, stall grace)

#include <algorithm>
#include <array>
#include <cmath>
#include <unordered_set>
#include <vector>

// DSR attribute name tags — generated from dsr_attr_name.h
#include <dsr/api/dsr_api.h>


// ─── Constructor / Destructor ────────────────────────────────────────────────────────────────────

SpecificWorker::SpecificWorker(const ConfigLoader& configLoader,
                               TuplePrx tprx,
                               bool startup_check)
    : GenericWorker(configLoader, tprx),
      startup_check_flag(startup_check)
{
    if (startup_check_flag)
    {
        this->startup_check();
        return;
    }

    load_config(configLoader);

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

    const auto err = statemachine.errorString();
    if (err.length() > 0)
    {
        qWarning() << err;
        throw err;
    }
}

SpecificWorker::~SpecificWorker()
{
    request_shutdown();
    std::print("cabinet_concept: SpecificWorker destroyed.\n");
}

void SpecificWorker::request_shutdown()
{
    if (shutting_down_.exchange(true))
        return;

    save_window_settings();
    save_dashboard_geometry();   // the standalone dashboard is not in `windows`, so save it explicitly
    save_strip_geometry();       // …nor is the compact belief strip

    cleanup_owned_nodes();

    // Drop the LiDAR media subscriber BEFORE tearing down the graph/inner_eigen it reads (it holds a raw
    // pointer to both). Mirrors bottle_concept.
    lidar_ingestor_.reset();

    // Drop the InnerEigenAPI now (the fitter only holds a raw pointer and is null-guarded): letting it
    // destruct later with the rest of the object can fault inside DSR. Mirrors bottle_concept.
    inner_eigen_.reset();
}

void SpecificWorker::terminal_shutdown()
{
    static std::atomic<bool> terminating{false};
    if (terminating.exchange(true))
        return;   // _Exit is coming; never run this twice

    // Crash-free terminal exit (matches bottle_concept). After our cleanup (owned
    // nodes deleted, peers notified) hard-exit instead of returning into the Ice communicator teardown +
    // C++ static destruction, which run with undefined cross-TU order and abort (a global/DDS holder
    // copies a graph Node after the node-type registry static is gone → Node::type() throws).
    request_shutdown();
    if (G)
    {
        try { G->reset(); }   // clean DDS participant/entity removal without touching the Ice communicator
        catch (...) { /* best-effort: we are exiting regardless */ }
    }
    std::cout.flush();
    std::cerr.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));   // let the del-deltas reach peers
    std::_Exit(EXIT_SUCCESS);
}

// ─── Initialisation ──────────────────────────────────────────────────────────────────────────────

void SpecificWorker::initialize()
{
    std::print("cabinet_concept: initialize()\n");
    GenericWorker::initialize();
 
    // Ignore payload attributes in local graph updates to avoid unnecessary copying and processing of potentially large data
    G->set_ignored_attributes<cam_rgb_att, cam_depth_att, laser_X_att, laser_Y_att, laser_Z_att>();
    qInfo() << "Ignoring DSR RGBD payload attributes cam_rgb/cam_depth in local graph updates";


    if (not G)
    {
        qWarning() << "cabinet_concept: DSR graph not available in initialize()";
        return;
    }

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
            const auto missing = presence_coordinator_.missing_required_names();
            if (missing.empty())
                qInfo("[SM] -> Waiting");
            else
            {
                QString m;
                for (const auto &label : missing)
                    m += " " + QString::fromStdString(label);
                qInfo() << "[SM] -> Waiting (missing:" << m.trimmed() << ")";
            }
        },
        .on_waiting_loop = [this]()
        {
            if (shutting_down_)
                return;
            // Pump the masks ingest WHILE Waiting: cabinet polls a graph node (no free-running ingest thread
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
            // One-time startup sweep: remove leftover affordance nodes from a PREVIOUS run (e.g. a crash
            // that skipped cleanup) so a fresh create doesn't collide and get a DSR-generated name.
            // Guarded — on a RE-entry to Operating (transient required-peer flap → Degraded → recover)
            // the affordances in the graph are THIS run's live ones; wiping them every bounce flickers.
            if (not startup_affordance_sweep_done_)
            {
                startup_affordance_sweep_done_ = true;
                remove_stale_affordance_nodes();
                // ALSO purge stale "cabinet*" box nodes now that the graph has fully SYNCED from peers.
                // The initialize() sweep can run BEFORE those nodes arrive from the persistent DSR server,
                // so a leftover from a crashed / SIGKILLed previous run would survive it. Safe here:
                // on_operating_enter fires before the first compute(), so this run has created none of its
                // own cabinets yet. Guarded (same flag) so a Degraded→Operating bounce never wipes live ones.
                remove_owned_cabinet_nodes();
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
                // Debounce: a transient required-peer flap (startup handshake, brief node churn) fires
                // presenceLost momentarily and then recovers; tearing down here would kill the agent on a
                // blip. Wait a grace period and only shut down if a required peer is STILL missing.
                qInfo("[SM] -> Degraded: required peer lost — %d ms grace before shutdown", REQUIRED_LOSS_GRACE_MS);
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

    QObject::connect(QCoreApplication::instance(), &QCoreApplication::aboutToQuit,
                     this, &SpecificWorker::terminal_shutdown, Qt::UniqueConnection);

    rt_api_ = G->get_rt_api();
    inner_eigen_ = G->get_inner_eigen_api();
    mask_ingestor_ = std::make_unique<rc::MaskIngestor>(G);
    scene_graph_ = std::make_unique<rc::CabinetSceneGraph>(
        G, rt_api_.get(), cfg_, [this] { trigger_graph_layout_twopi(); });

    // Subscribe to graph signals
    // ★NOT update_node_attr_signal. It fires for EVERY attribute change on EVERY node in the shared graph —
    // every RT pose write from robot_concept, every other agent's per-cycle attribute writes — and each
    // emission copies a std::vector<std::string> on the DDS reader thread and queues an event to this thread.
    // Under a churn burst (a peer restarting is enough) the queue drains slower than it fills, and the main
    // thread then does nothing but service slots: the timer-driven compute() is starved (its log stops
    // growing), and — the symptom that costs the most — Ctrl-C dies, because generated/main.cpp routes SIGINT
    // through a QSocketNotifier serviced by this SAME event loop. The agent can then only be killed with -9,
    // which cannot be caught, so every node it owns LEAKS into the shared graph.
    // Measured 2026-08-07 on table_concept (identical subscription): main thread pegged at 100% of a core,
    // ai2_log.csv frozen, Ctrl-C inert, right after a voxelizer restart. CLAUDE.md already states the rule
    // this violated: if you don't need a signal, don't connect it at all (bottle_concept connects none).
    // The two things the slot did are now POLLED once per cycle in poll_affordance_protocol().
    connect(G.get(), &DSR::DSRGraph::update_node_signal,
            this, &SpecificWorker::modify_node_slot);
    connect(G.get(), &DSR::DSRGraph::del_node_signal,
            this, &SpecificWorker::del_node_slot);

    // Remove any "cabinet*" nodes left behind by a previous (crashed) run so this agent always starts
    // from a clean slate and never adopts a stale/drifted node (the instance tracker re-births them
    // data-driven from masks).
    remove_owned_cabinet_nodes();

    // Resolve room node
    const auto rooms = G->get_nodes_by_type("room");
    if (not rooms.empty())
        room_node_id_ = rooms.front().id();
    else
        qWarning() << "cabinet_concept: no room node found at startup";

    // Active-inference fit core. Owns the instance map; collaborates with the ingestor + scene graph.
    fitter_ = std::make_unique<rc::CabinetFitter>(
        G, inner_eigen_.get(), cfg_, mask_ingestor_.get(), scene_graph_.get());
    existence_ = std::make_unique<rc::CabinetExistence>(G, cfg_);   // evidence-based removal (existence log-odds)

    // Part B: localization/chain covariance on the published RT edge (mirrors bottle_concept).
    gaussian_api_ = std::make_unique<DSR::InnerGaussianAPI>(G.get());
    fitter_->set_chain_cov_source(gaussian_api_.get(), "zed");
    // Object-anchor observation z_o for room_concept's landmark factor (expressed in the localizer base frame).
    fitter_->set_object_observation(cfg_.publish_object_obs, cfg_.object_obs_frame);

    // YOLO-independent LiDAR range channel: lidar3D media-plane consumer that stages each cycle's sweep in the
    // room frame for the fitter's range factor. Dormant (no DDS participant) unless CabinetModel.LidarPrecision
    // > 0. Subscriber is brought up lazily on the compute/main thread once the lidar3D node + descriptor exist.
    lidar_ingestor_ = std::make_unique<rc::CabinetLidarIngestor>(G, inner_eigen_.get(), cfg_);

    // Build rc::EpistemicPlanner (info-gain scoring only; stand-off distance is the sole parameter).
    epistemic_planner_ = rc::EpistemicPlanner(cfg_.obs_distance);
    // ONE detector envelope: the viewpoint the planner asks for is the argmax of the same model absence
    // should be weighted by. Replaces the old kMinimumStandOffM literal in the planner.
    epistemic_planner_.set_detector_envelope(rc::detect::DetectorEnvelope{});
    epistemic_planner_.set_robot_radius(0.30f);   // Shadow's footprint radius

    // The camera's REAL geometry, read once (intrinsics and the zed mount are both static). BOTH FoVs: a
    // WALL-TIER cabinet sits high, so the VERTICAL axis is the binding one and the horizontal-only model this
    // replaced was blind to it. Height from inner_eigen room->zed, NOT the proto's body-relative z: the room
    // floor datum is offset from the body origin, and the box z-span is room-frame. ts==0 on the main thread.
    // ★The camera model is read PER CYCLE at the compute site (rc::nbv::sensor_from_graph),
    // NOT once here: the zed intrinsics are published by robot_concept when frames start
    // arriving, so reading them in initialize() races the producer. Losing that race leaves
    // vfov = 0, which silently collapses the fill model to horizontal-only — the exact bug
    // rc::nbv exists to fix, and it drives the robot nose-to-nose with tall objects.

    // Stale affordance nodes are swept on entering Operating (presence hook) and on shutdown — see
    // remove_stale_affordance_nodes(), keyed on the parent object type (robust to node-name renames).

    // Standalone Qt dashboard + evidence-monitor windows (belief plots + per-instance snapshot).
    // Shadow-mode birth/death record (CONCEPT_AGENT_LIFECYCLE.md §4.2). Recording only — see
    // log_phantom_event(). Truncating: one file per run.
    phantom_log_.open("etc/cabinet_phantom_events.csv");

    build_dashboard();
}

// ─── Main compute loop ───────────────────────────────────────────────────────────────────────────

// SHADOW-MODE birth/death recorder — CONCEPT_AGENT_LIFECYCLE.md §4.2, theory in MODEL_HISTORY.md §4.
// RECORDS ONLY; it can never alter a birth or a removal. The attribution fields captured at death
// (p_detect / in-FoV / central) are what tell a genuine classifier phantom from one of OUR removal defects —
// a death with LOW p_detect means the log is recording a removal bug, and must not be learned from.
void SpecificWorker::log_phantom_event(std::string_view event, std::uint64_t id, std::string_view name,
                                       float x, float y, const rc::CabinetInstance* inst, std::string_view note)
{
    if (not phantom_log_.is_open())
        return;
    rc::history::PhantomEvent e;
    e.event = event; e.id = id; e.name = name; e.x = x; e.y = y; e.note = note;
    // Observer pose → view bearing. The classifier failure is VIEWPOINT-dependent, so the eventual p_FA field
    // is keyed on (world cell × bearing); a place-only key would suppress a genuine object placed there.
    if (inner_eigen_)
        if (const auto rtb = inner_eigen_->get_transformation_matrix("room", "body", 0); rtb.has_value())
        {
            const auto& Tm = rtb.value();
            e.robot_x = static_cast<float>(Tm(0, 3));
            e.robot_y = static_cast<float>(Tm(1, 3));
            e.robot_yaw = std::atan2(static_cast<float>(Tm(1, 0)), static_cast<float>(Tm(0, 0)));
            e.view_bearing = std::atan2(e.robot_y - y, e.robot_x - x);   // instance → camera, room frame
            e.range_m = std::hypot(e.robot_x - x, e.robot_y - y);
        }
    if (inst)   // death: carry the existence state that says whether this was a CONFIDENT kill
    {
        e.age_cycles    = inst->processed_cycles;
        e.p_detect      = inst->dbg_ex_pdetect;
        e.central_frac  = inst->dbg_ex_central;
        e.in_fov_frac   = (inst->dbg_ex_sil_ndet > 0) ? 1.0f : 0.0f;
        e.exist_logodds = inst->existence.logodds();
    }
    phantom_log_.write(e);
}

void SpecificWorker::compute()
{
    if (not G or not rt_api_)
        return;

    // Refresh room node id if not yet found
    if (room_node_id_ == 0)
    {
        const auto rooms = G->get_nodes_by_type("room");
        if (rooms.empty()) return;
        room_node_id_ = rooms.front().id();
    }

    // Controller-owned affordance flags (claim / completion / epistemic_pending). Polled here rather than
    // pushed by update_node_attr_signal — see the connect block in initialize().
    poll_affordance_protocol();


    // Push the room geometry (wall polygon + interior centroid) into the fitter each cycle: the wall-flush
    // factor needs the polygon and canonicalize needs the interior. Cheap, and room_concept may refine it.
    refresh_room_geometry();

    const bool fresh_masks = mask_ingestor_->refresh();

    // A cabinet run cannot be L-shaped: split any L-corner 'cabinet' mask into its two perpendicular arms
    // (in place) BEFORE the tracker, so it births/fits two clean single-arm runs instead of one tilted box.
    split_lshaped_cabinet_masks();

    // EvidenceMonitor per-cycle counters (cumulative *_cum fields persist across cycles). The producers below
    // (tracker / merge / removal) add to these; the snapshot is pushed at the end of the cycle.
    ev_g_.births = ev_g_.merges = ev_g_.removals = 0;
    ev_g_.mask_stale = not fresh_masks;

    // Snapshot the residual (surprise) field ONCE per cycle before the tracker, so fused birth (run_instance_
    // tracker) and the logging probe (compute() tail) share one read. No-op unless a probe/fusion flag is on.
    if (cfg_.birth_fusion or cfg_.birth_surprise_probe) read_residual_field();

    bool fresh_sweep = false;
    // Stage 2 kitchen model: the (wall,tier) cells own the WallRunBeliefs; identity IS the cell, so there is no
    // birth/associate/merge and the disasters (crossings, 10 cm, ceiling boxes) are unrepresentable. When on it
    // fully replaces the classic tracker/fitter/residual-birth pipeline below.
    if (cfg_.kitchen_model) { run_kitchen_model(); }
    else {

    run_instance_tracker();   // data-driven birth / associate / merge (the only instance-lifecycle path)

    // Stage this cycle's LiDAR sweep (room frame) for the fitter's range factor. clear-then-set so the factor
    // never consumes a stale sweep; pump() is main-thread (reads the graph) + dormant while the feature is off.
    fitter_->clear_lidar_sweep();
    if (lidar_ingestor_)
    {
        lidar_ingestor_->pump();   // pumps BOTH planes (helios primary; bpearl if LidarBpearlPrecision>0)
        if (lidar_ingestor_->helios_fresh())
        {
            fitter_->set_lidar_sweep(lidar_ingestor_->sweep_room(), lidar_ingestor_->origin_room());
            fresh_sweep = true;
        }
        if (lidar_ingestor_->bpearl_fresh())   // low LiDAR as a SEPARATE ray-set (own origin) — sees the legs
        {
            fitter_->set_lidar_sweep_bpearl(lidar_ingestor_->sweep_bpearl_room(), lidar_ingestor_->origin_bpearl_room());
            fresh_sweep = true;
        }
    }

    // `object` nodes named "cabinet_*" are ours (see CabinetSceneGraph: cortex has no `cabinet` type;
    // the class lives in the `object_subtype` attr).
    std::vector<DSR::Node> cabinet_nodes;
    for (const auto& n : G->get_nodes_by_type("object"))
        if (n.name().starts_with("cabinet"))
            cabinet_nodes.push_back(n);
    for (const auto& node : cabinet_nodes)
        process_cabinet_node(node);

    // Residual-driven birth: after the fits (residuals current), cluster the pooled unexplained points and
    // mature a coherent unmodeled arm (e.g. an L-corner's perpendicular arm) into its own axis-aligned run.
    birth_from_residual();

    // Ricoh 360 = peripheral attention: associate ricoh detections to cabinets BY DIRECTION (after the ZED fits,
    // so cabinet positions are current); an unassigned bearing becomes a "seek a ZED view here" attention target.
    process_ricoh_bearings();

    // Evidence-based removal: each existence channel integrates on its OWN sensor cadence (silhouette/mask on a
    // fresh mask frame, LiDAR carve on a fresh sweep) — a camera-only cycle still accrues absence, a LiDAR-only
    // cycle still carves free space. After the fits so footprints are current. OFF unless enabled.
    if (cfg_.existence_removal_enabled)
        existence_->update_and_remove(*fitter_, lidar_ingestor_.get(), fresh_masks, fresh_sweep, ev_g_,
            [this](std::uint64_t id, const rc::CabinetInstance& inst)
            {   // shadow-mode death record (§4.2) — p_detect here says whether this was a CONFIDENT
                // disconfirmation (a real phantom) or a weak one (more likely our own removal bug)
                log_phantom_event("DEATH", id, inst.node_name,
                                  inst.model.state().cx, inst.model.state().cy, &inst,
                                  std::format("L {:.2f}", inst.existence.logodds()));
            });

    }   // end classic instance pipeline (skipped when cfg_.kitchen_model)

    // ── Evidence monitor: global counters + throttled snapshot push ──
    ev_g_.instances    = cfg_.kitchen_model ? static_cast<int>(kitchen_mgr_.active_boxes().size())
                                            : static_cast<int>(fitter_->instances().size());
    ev_g_.sweep_points = (lidar_ingestor_ and fresh_sweep) ? static_cast<int>(lidar_ingestor_->sweep_room().size()) : 0;
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
    refresh_evidence_monitor();
    prune_dead_series();   // drop timeseries lines for cabinets that were removed from the graph
    log_birth_surprise();  // EXPERIMENTAL read-only: residual grid → birth-surprise regions (cfg-gated, off by default)
    fps_counter_.print("[cabinet_concept Compute]");
}

// EXPERIMENTAL, read-only. Reinterpret residual_concept's `grid` node (under room) as an unexplained-occupancy
// SURPRISE field and cluster its confidently-unexplained cells into candidate regions (birth_surprise_probe.h).
// Log, per cycle, the regions NOT already covered by a believed cabinet alongside the tracker's actual birth count,
// so we can see whether surprise flags a real new cabinet cleanly (and stays quiet on phantoms) BEFORE letting it
// drive the lifecycle. Never writes the graph; never births. See [[cabinet-birth-surprise-probe]].
// Snapshot residual_concept's `grid` node (published under room ~2 Hz) into residual_field_ — the dense
// P(occupied ∧ ¬explained) surprise field. Read-only; never mutates the graph. Called once at the compute() head
// so BOTH the fused-birth path (run_instance_tracker) and the logging probe (log_birth_surprise) share one read.
bool SpecificWorker::read_residual_field()
{
    residual_field_ = rc::GridField{};   // reset (empty ⇒ invalid ⇒ consumers no-op)
    if (room_node_id_ == 0) return false;
    const auto gopt = G->get_node("residual");   // node renamed "grid"→"residual" (type stays "grid")
    if (not gopt.has_value()) return false;
    const auto& gnode = gopt.value();
    const auto pa = G->get_attrib_by_name<grid_occupancy_prob_att>(gnode);
    const auto ma = G->get_attrib_by_name<grid_field_meta_att>(gnode);
    if (not (pa.has_value() and ma.has_value())) return false;
    const auto& M = ma.value().get();
    if (M.size() < 5) return false;
    residual_field_.prob = pa.value().get();   // snapshot copy (small, ~2 Hz)
    if (const auto va = G->get_attrib_by_name<grid_occupancy_var_att>(gnode); va.has_value())
        residual_field_.var = va.value().get();
    residual_field_.xmin = M[0]; residual_field_.ymin = M[1]; residual_field_.cell = M[2];
    residual_field_.width = static_cast<int>(M[3]); residual_field_.height = static_cast<int>(M[4]);
    return residual_field_.valid();
}

void SpecificWorker::log_birth_surprise()
{
    if (not cfg_.birth_surprise_probe or not residual_field_.valid()) return;
    const rc::GridField& gf = residual_field_;   // read at the compute() head by read_residual_field()

    // Believed cabinet footprints (room frame) — a region under one is already explained, NOT a birth.
    std::vector<rc::FootprintBox> cabinets;
    for (const auto& [id, inst] : fitter_->instances())
        if (inst.ai2_initialized)
        { const auto& s = inst.ai2_belief.state(); cabinets.push_back({s.cx, s.cy, s.L, s.d, s.yaw}); }

    const auto cands = rc::BirthSurpriseProbe::scan(gf, cabinets);
    const long cyc = ++birth_surprise_cycle_;   // advances only on cycles where the grid field was actually read
    int n_birth = 0;                       // uncovered high-surprise regions = birth candidates
    for (const auto& c : cands) if (not c.covered_by_cabinet) ++n_birth;

    // CSV: one row per region per cycle (covered flag distinguishes birth candidates from explained mass; the
    // latter should be ~0 if residual_concept's concept-subtraction is working — a free sanity check).
    if (not birth_surprise_csv_.is_open())
    {
        birth_surprise_csv_.open("etc/birth_surprise.csv", std::ios::out | std::ios::trunc);
        if (birth_surprise_csv_.is_open())
            birth_surprise_csv_ << "cycle,region,cx,cy,cells,mass,ext_x,ext_y,mean_p,mean_var,covered,"
                                << "n_cabinets,tracker_births,instances\n";
    }
    if (birth_surprise_csv_.is_open())
    {
        int r = 0;
        for (const auto& c : cands)
            birth_surprise_csv_ << cyc << ',' << r++ << ',' << c.cx << ',' << c.cy << ',' << c.cells << ','
                                << c.mass << ',' << c.ext_x << ',' << c.ext_y << ',' << c.mean_p << ',' << c.mean_var
                                << ',' << (c.covered_by_cabinet ? 1 : 0) << ',' << cabinets.size() << ','
                                << ev_g_.births << ',' << fitter_->instances().size() << '\n';
        birth_surprise_csv_.flush();
    }

    // ── FUSION readout: residual surprise MASS under each YOLO "cabinet" detection (birth_fusion.csv). The measured
    //    quantity: does a real detection land on high unexplained-occupancy (→ corroborated → birth fast/confident)
    //    while a flicker/phantom detection lands on ~0? covered = the detection sits inside an already-believed
    //    cabinet footprint (associate, not birth). This is the signal that would let residual GATE/accelerate birth.
    if (not birth_fusion_csv_.is_open())
    {
        birth_fusion_csv_.open("etc/birth_fusion.csv", std::ios::out | std::ios::trunc);
        if (birth_fusion_csv_.is_open())
            birth_fusion_csv_ << "cycle,det,det_x,det_y,mass_r05,mass_r03,near_dist,near_mass,covered,"
                              << "n_cabinets,tracker_births,instances\n";
    }
    if (birth_fusion_csv_.is_open())
    {
        int di = 0;
        for (const auto& d : last_cabinet_dets_xy_)
        {
            const float m05 = rc::BirthSurpriseProbe::residual_mass_near(gf, d.x(), d.y(), 0.50f);
            const float m03 = rc::BirthSurpriseProbe::residual_mass_near(gf, d.x(), d.y(), 0.30f);
            float nd = 1e9f, nm = 0.f;                      // nearest region to this detection
            for (const auto& c : cands)
            { const float dd = std::hypot(c.cx - d.x(), c.cy - d.y()); if (dd < nd) { nd = dd; nm = c.mass; } }
            bool covered = false;
            for (const auto& t : cabinets)
            { const float cc = std::cos(t.yaw), ss = std::sin(t.yaw), dx = d.x() - t.cx, dy = d.y() - t.cy;
              if (std::abs(cc*dx + ss*dy) <= 0.5f*t.w + 0.30f and std::abs(-ss*dx + cc*dy) <= 0.5f*t.h + 0.30f)
                  { covered = true; break; } }
            birth_fusion_csv_ << cyc << ',' << di++ << ',' << d.x() << ',' << d.y() << ',' << m05 << ',' << m03 << ','
                              << (nd > 1e8f ? -1.f : nd) << ',' << nm << ',' << (covered ? 1 : 0) << ','
                              << cabinets.size() << ',' << ev_g_.births << ',' << fitter_->instances().size() << '\n';
            if (ev_g_.births > 0 and not covered)          // a NEW cabinet just born — print its corroboration
                std::print("[birth-fusion] BIRTH det@({:.2f},{:.2f}) residual mass_r05={:.1f} mass_r03={:.1f} "
                           "near_region_mass={:.1f} dist={:.2f}\n", d.x(), d.y(), m05, m03, nm, nd);
        }
        birth_fusion_csv_.flush();
    }

    // Console: throttled (every ~20 cycles) OR whenever the tracker actually births this cycle — so the surprise
    // state at the birth instant is always printed for correlation.
    if (n_birth > 0 and (ev_g_.births > 0 or (birth_surprise_log_ctr_++ % 20) == 0))
    {
        const rc::BirthCandidate* top = nullptr;   // strongest UNcovered region
        for (const auto& c : cands) if (not c.covered_by_cabinet) { top = &c; break; }   // cands sorted by mass
        if (top)
            std::print("[birth-surprise] uncovered={} cabinets={} tracker_births={} | top: ({:.2f},{:.2f}) "
                       "mass={:.1f} cells={} ext={:.2f}x{:.2f} mean_p={:.2f} var={:.3f}\n",
                       n_birth, cabinets.size(), ev_g_.births, top->cx, top->cy, top->mass, top->cells,
                       top->ext_x, top->ext_y, top->mean_p, top->mean_var);
    }
}


// ─── Per-node processing + publish ───────────────────────────────────────────────────────────────

// Process one "cabinet" DSR node this cycle: ensure its instance exists, then fuse each assigned ZED slice
// (one belief update per slice, gated to a fresh mask frame) or age the belief when no mask arrived, and
// hand the result to publish_cabinet_cycle. run_instance_tracker has already associated this cycle's slices.
void SpecificWorker::process_cabinet_node(const DSR::Node& node)
{
    const bool created = fitter_->ensure_instance(node, room_node_id_);
    auto& inst = fitter_->instances().at(node.id());

    if (created)
    {
        // (Per-instance time-series are registered idempotently by publish_cabinet_diagnostics each cycle.)
        // Canvas position — viewer randomizes pos_x/pos_y if absent.
        if (not G->get_attrib_by_name<pos_x_att>(node).has_value())
        {
            auto n_mut = node;
            float rpx = 200.f, rpy = 200.f;
            if (room_node_id_ != 0)
                if (const auto rn = G->get_node(room_node_id_); rn.has_value())
                {
                    rpx = G->get_attrib_by_name<pos_x_att>(rn.value()).value_or(200.f);
                    rpy = G->get_attrib_by_name<pos_y_att>(rn.value()).value_or(200.f);
                }
            G->add_or_modify_attrib_local<pos_x_att>(n_mut, rpx + 150.f);
            G->add_or_modify_attrib_local<pos_y_att>(n_mut, rpy +  50.f);
            G->update_node(n_mut);
        }
    }

    ++inst.processed_cycles;
    // Detection-aliveness ages once per cycle; observe_slice() resets it to 0 when a slice is assigned.
    if (inst.frames_since_detection < 1000000)
        ++inst.frames_since_detection;

    float free_energy = 0.0f;
    CabinetObservation last_obs;
    bool updated = false;

    // Multi-slice fusion: one belief update per assigned ZED slice of this cabinet. Sequential Bayesian updates
    // = joint likelihood for the recursive filter, and each slice keeps its OWN R and common-mode (they don't
    // share a registration error) — which concatenating points into one frame could not express.
    // ONLY on a fresh mask FRAME for this instance: the tracker assigns every cycle (birth/association
    // continuity), but re-fitting the SAME packet each cycle would overcount evidence — gate on a new frame_id.
    const auto& pkt = mask_ingestor_->packet();
    // Freshness gate. mask_frame_id is a PUBLISH counter — it advances on every republish even when the source
    // camera is FROZEN/paused, so gating on it alone re-integrates the SAME capture as independent evidence and
    // the belief RATCHETS: w/h/yaw rotate + reshape with the robot AND scene stationary (ai2_log.csv symptom —
    // npts pinned identical, flip_ev saturated, yaw walking ~90°). The producer also ships mask_timestamp_ms =
    // the CAPTURE stamp of the source RGBD frame; require THAT to advance too. A stale capture stamp ⇒ no new
    // sensor information ⇒ fall through to AGE the belief (predict-only, Σ grows), never re-integrate the same
    // frame. timestamp_ms==0 (older producer with no capture stamp) ⇒ fall back to the frame_id gate alone.
    const bool fresh_frame   = pkt.valid and pkt.frame_id > inst.last_masks_frame_seen;
    const bool fresh_capture = pkt.timestamp_ms == 0 or pkt.timestamp_ms > inst.last_mask_timestamp_ms;
    if (fresh_frame and fresh_capture)
    {
        for (const int idx : inst.assigned_mask_idxs)
        {
            auto obs = fitter_->observe_slice(inst, idx);
            if (not obs.has_fresh_data)
                continue;
            free_energy = fitter_->run_inference(inst, obs);
            last_obs = std::move(obs);
            updated = true;
        }
        inst.last_masks_frame_seen = pkt.frame_id;
    }

    if (not updated)
    {
        // No mask associated this cycle: legacy node-attrib path, else a stale observation → age the belief.
        last_obs = fitter_->observe(inst, node);
        if (not last_obs.has_fresh_data and inst.matched_frames < 5)
            return;
        free_energy = fitter_->run_inference(inst, last_obs);
    }

    publish_cabinet_cycle(inst, node, last_obs, free_energy);
}


// Publish one fitted cabinet: persist belief→DSR, assess convergence, then push diagnostics + intentions.
// Each step short-circuits if its node lookup fails (the node may have been removed mid-cycle).
void SpecificWorker::publish_cabinet_cycle(rc::CabinetInstance& inst,
                                         const DSR::Node& node,
                                         const CabinetObservation& observation,
                                         float free_energy)
{
    const auto node_id = node.id();
    if (not scene_graph_->persist_cabinet_belief(inst, node_id, room_node_id_, free_energy))
        return;
    if (not assess_cabinet_state(inst, node_id, free_energy))
        return;
    publish_cabinet_diagnostics(inst, observation, free_energy);
    publish_cabinet_intentions(inst, node_id, observation, free_energy);
}

// Run the convergence/stability step for one cabinet; re-resolves the node by id (false if it is gone).
bool SpecificWorker::assess_cabinet_state(rc::CabinetInstance& inst, uint64_t node_id, float free_energy)
{
    auto node_opt = G->get_node(node_id);
    if (not node_opt.has_value())
        return false;

    step_convergence(inst, node_opt.value(), free_energy);
    return true;
}

// Feed this instance's time-series to the standalone dashboard: FE + baseline, FE-surprise, U(Σ) covariance,
// residual count, inferred dims (w,h), and size posterior σ. Series are added idempotently every cycle.
void SpecificWorker::publish_cabinet_diagnostics(const rc::CabinetInstance& inst,
                                               const CabinetObservation& observation,
                                               float free_energy)
{

    // Register lazily & idempotently HERE (same thread/object that adds points). The instance is
    // often created via the graph-signal path before the plots exist (created=false in the compute
    // loop), so the old `if (created)` registration never fired and every point was dropped.
    if (ts_plot_)
    {
        ts_plot_->add_series(inst.node_name + "_fe", QColor(255, 170, 0), 1.1f);
        ts_plot_->add_point (inst.node_name + "_fe", free_energy);
        // FE BASELINE stays on the FE panel (same units): the FE line lifting ABOVE the gray baseline IS the
        // surprise, shown here visually. The SURPRISE (smoothed gap) goes on its OWN panel (ts_surprise_plot_)
        // because it lives on a much smaller scale — the active-perception attention signal.
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
            ts_cov_plot_->add_point (inst.node_name + "_cov", rc::geom::belief_uncertainty(inst));
        }
        if (ts_res_plot_)
        {
            ts_res_plot_->add_series(inst.node_name + "_res", QColor(170, 80, 255), 1.1f);
            // Plot inst.dbg_resid_pts (the SAME value the EvidenceMonitor counter shows), which HOLDS its last
            // value between masks — so the line matches the counter and doesn't vanish. The old code sampled
            // observation.residual_pts only on fresh frames, so on a non-fresh publish it added nothing → no line.
            ts_res_plot_->add_point(inst.node_name + "_res", static_cast<float>(inst.dbg_resid_pts));
        }
        // (The inferred-dimensions trace that used to live here is gone: the BeliefInspector below shows
        // every DOF's value AND its σ live, so a separate w/h trace was showing the same thing twice.)
        // (The size-posterior σ_w/σ_h trace that used to live here is gone: the BeliefInspector panel now
        // shows σ for EVERY DOF, next to its target σ* and the whole correlation structure.)
    }

    if (fitter_->should_log(inst))
        std::print("[{}] series: FE={:.4f} U(Σ)={:.3f} res={}\n",
                   inst.node_name,
                   free_energy,
                   rc::geom::belief_uncertainty(inst),
                   observation.residual_pts.size());
}

void SpecificWorker::publish_cabinet_intentions(rc::CabinetInstance& inst,
                                              uint64_t node_id,
                                              const CabinetObservation& observation,
                                              float free_energy)
{
    // step_epistemic now owns the propose-vs-withdraw decision via the planner's ΔH (it withdraws
    // when the expected information gain falls below threshold), so it runs unconditionally rather
    // than being gated on the legacy coverage-deficit proxy.
    if (auto node_opt = G->get_node(node_id); node_opt.has_value())
        step_epistemic(inst, node_opt.value());
}

// ─── Initialisation helpers ──────────────────────────────────────────────────────────────────────

void SpecificWorker::load_config(const ConfigLoader& cfg)
{
    cfg_ = rc::load_cabinet_config(cfg);
}


// ─── Per-cycle steps ─────────────────────────────────────────────────────────────────────────────

// Convergence latch for one cabinet: measure how far the accepted state moved this cycle and, once it holds
// still for K_stable cycles, publish model_stable + the model_uncertainty_att readout (posterior std sum).
void SpecificWorker::step_convergence(rc::CabinetInstance& inst,
                                       DSR::Node& node,
                                       float free_energy)
{
    // Convergence on STATE stability, not |ΔFE|: the free energy keeps jittering with queue
    // churn / point-count even when the fitted geometry is settled, so it never latched. Track
    // how much the accepted state moved between cycles instead.
    const auto& s = inst.model.state();
    const auto& p = inst.prev_conv_state;
    const float state_delta = inst.has_prev_conv_state
        ? (std::abs(s.cx - p.cx) + std::abs(s.cy - p.cy) + std::abs(s.L - p.L) + std::abs(s.d - p.d) +
           std::abs(s.z0 - p.z0) + std::abs(s.z1 - p.z1) + std::abs(s.yaw - p.yaw))
        : std::numeric_limits<float>::max();
    inst.prev_conv_state = s;
    inst.has_prev_conv_state = true;
    if (state_delta < cfg_.state_eps)
    {
        inst.frames_converged = std::min(inst.frames_converged + 1, cfg_.K_stable);
    }
    else
    {
        inst.frames_converged = 0;
        inst.model_stable     = false;
    }

    // Model uncertainty for model_uncertainty_att: sum of the belief's per-DOF posterior stds over
    // position + size (m), from the AI2 covariance Σ over [cx,cy,H,w,h,yaw]. Shrinks as the robot
    // gathers viewpoints — the AI2-native replacement for the old queue face-coverage deficit.
    const float model_uncertainty = rc::geom::belief_uncertainty(inst);
    G->add_or_modify_attrib_local<model_uncertainty_att>(node, model_uncertainty);

    if (fitter_->should_log(inst))
        std::print("[{}] convergence: Δstate={:.4f} stable={}/{} U(Σ)={:.3f}m\n",
                   inst.node_name, state_delta, inst.frames_converged, cfg_.K_stable, model_uncertainty);

    if (inst.frames_converged >= cfg_.K_stable)
    {
        if (not inst.model_stable)
        {
            inst.model_stable = true;
            G->add_or_modify_attrib_local<model_stable_att>(node, true);
            G->update_node(node);
            std::print("cabinet_concept: node '{}' STABLE (F={:.4f})\n",
                       inst.node_name, free_energy);
        }
    }
    else
    {
        if (inst.model_stable)
        {
            G->add_or_modify_attrib_local<model_stable_att>(node, false);
            G->update_node(node);
        }
        inst.model_stable = false;
    }
}

// Publish/refresh the epistemic next-best-view affordance for one cabinet from its belief Σ (D-optimal NBV),
// with a post-completion cooldown that suppresses the published gain so a just-finished cabinet isn't re-claimed.
void SpecificWorker::step_epistemic(rc::CabinetInstance& inst, DSR::Node& node)
{
    if (inst.epistemic_cooldown > 0)
        --inst.epistemic_cooldown;

    // Controller-completion hold (anti-churn): the cabinet affordance completes on a weak detection
    // (contract goal conf≥0.20), which fires almost instantly — before ΔH has decayed. Start a short
    // cooldown so we don't re-offer a just-completed cabinet while its gain is still high. We do NOT
    // delete the node (that is what made the affordance vanish from the graph): it stays and keeps
    // refreshing; we only suppress its published gain during the hold so the controller's EFE selection
    // won't immediately re-claim it.
    if (const auto aid = inst.affordance.node_id(); aid != 0)
        if (auto an = G->get_node(aid); an.has_value())
        {
            const bool a = G->get_attrib_by_name<active_att>(an.value()).value_or(false);
            const bool p = G->get_attrib_by_name<epistemic_pending_att>(an.value()).value_or(true);
            if (not a and not p and inst.epistemic_cooldown == 0)   // just completed by the controller
            {
                inst.epistemic_cooldown = cfg_.epistemic_cooldown_cycles;
                std::print("[{}] controller completed affordance → hold {} cycles (node kept, gain suppressed)\n",
                           inst.node_name, cfg_.epistemic_cooldown_cycles);
            }
        }

    // Σ-based D-optimal NBV from the belief. Skip until the belief has seen its first frame (else Σ is
    // the broad prior and the proposal is moot).
    if (not inst.ai2_initialized)
        return;
    rc::EpistemicProposal prop =
        epistemic_planner_.compute(inst.ai2_belief, cfg_.ai2_range_noise_lat_per_m, cfg_.ai2_sigma_base_m,
                                   rc::nbv::sensor_from_graph(*G, inner_eigen_.get()), cfg_.verbose_log,
                                   rc::nbv::collect_graph_obstacles(*G, inner_eigen_.get(), inst.node_id),
                                   // The reachable region — kills the through-the-wall faces that the
                                   // direction-blind gain cannot tell apart. Empty until room_concept publishes,
                                   // and is_reachable then imposes no constraint (the pre-existing behaviour).
                                   fitter_->room_polygon());
    if (not prop.valid or not prop.is_finite())
        return;   // degenerate (non-finite) fit — leave the existing affordance node as-is, retry next cycle

    // Belief→knowledge governor WITHOUT deleting the node: keep publishing the affordance every cycle
    // with its TRUE expected information gain ΔH (nats). A low gain is published as-is so the controller's
    // grounded EFE selection simply doesn't pick it (cost outweighs the small epistemic value), and it
    // re-arms automatically when the belief degrades and ΔH climbs again — no satisfy-latch to get stuck
    // in, no node churn. During the post-completion hold the gain is forced to 0 so a just-finished cabinet
    // isn't re-claimed before its belief has settled.
    if (inst.epistemic_cooldown > 0)
        prop.epistemic_gain = 0.0f;

    // Verification pull (active inference): a cabinet whose predicted absence could NOT be resolved from recent
    // views (wants_verification — far/peripheral/edge-on "I don't see it") does NOT get deleted; it gets a strong
    // epistemic gain so the controller drives to a good ZED viewpoint to CONFIRM-or-remove it. Absence never
    // deletes a cabinet the robot hasn't properly looked at — it sends the robot to look. Overrides the cooldown
    // (a "might be gone" alarm is not anti-chatter-suppressible). Clears once a verifying view resolves it.
    if (inst.wants_verification)
        prop.epistemic_gain = std::max(prop.epistemic_gain, cfg_.existence_verify_gain);

    // Write attributes to the cabinet node (read by legacy consumers)
    scene_graph_->write_epistemic_proposal(node, prop);
    // Publish / refresh dedicated affordance node (persists; update_node refreshes target+gain)
    const auto affordance_node_before = inst.affordance.node_id();
    inst.dbg_nbv_gain = prop.epistemic_gain;   // the value actually published
    inst.affordance.update(prop);
    if (affordance_node_before == 0 and inst.affordance.node_id() != 0)
        trigger_graph_layout_twopi();
    inst.epistemic_pending = true;
}

// ─── DSR helpers ─────────────────────────────────────────────────────────────────────────────────

// Re-run the graph viewer's twopi layout now and again once queued, so it also settles after the pending
// node/edge update signals are processed. No-op when Agent.graph is off (no viewer widget).
void SpecificWorker::trigger_graph_layout_twopi()
{
    const auto it = graph_viewers.find("");
    if (it == graph_viewers.end() || !it->second)
        return;

    QWidget* graph_widget = it->second->get_widget(DSR::DSRViewer::view::graph);
    auto* graph_viewer = qobject_cast<DSR::GraphViewer*>(graph_widget);
    if (!graph_viewer)
        return;

    // Run now and once queued, so layout also happens after pending node/edge
    // update signals are processed by the viewer.
    graph_viewer->compute_layout("twopi");
    QMetaObject::invokeMethod(graph_viewer,
                              [graph_viewer]() { graph_viewer->compute_layout("twopi"); },
                              Qt::QueuedConnection);
}
// ─── DSR signal slots (QUEUED — never DirectConnection; see CLAUDE.md) ───────────────────────────

void SpecificWorker::modify_node_slot(std::uint64_t /*id*/, const std::string& /*type*/)
{
    // Deliberately does NOT create instances. insert_node() emits update_node_signal SYNCHRONOUSLY on the
    // main thread (same-thread Auto → Direct), so creating the instance here runs INSIDE the tracker's
    // create_instance_from_detection → before its note_birth() seed is stored → the model is born at the
    // 0,0 RT-read default and frozen there (the tracker then never associates and re-births forever).
    // Instance creation is owned by the compute() loop (process_cabinet_node → ensure_instance), which runs
    // AFTER note_birth() each cycle, so the birth seed is in place. The loop also covers externally-created
    // cabinet nodes (every cycle), so nothing is lost by not creating here.
}

// Poll the controller-owned protocol flags once per cycle. This REPLACES the update_node_attr_signal
// subscription (see the connect block in initialize() for the starvation it caused). Both things it did are
// pure reads of the CURRENT graph state, so polling gives the same answer — a controller claim or completion
// simply lands on the next cycle instead of within the emitting DDS callback.
//
// Cost is bounded by OUR instance count (typically 1–3), not by the graph's global write rate, which is the
// whole point: the old path did work proportional to what every other agent was writing.
void SpecificWorker::poll_affordance_protocol()
{
    if (not fitter_)
        return;   // may be called before the fit core exists — the guard every graph slot here needs

    for (auto& [cabinet_id, inst] : fitter_->instances())
    {
        // Affordance state machine: idle→pending→executing→satisfied, driven by the controller-owned
        // active/pending flags on the affordance node. on_node_modified() re-reads them itself.
        if (const auto aid = inst.affordance.node_id(); aid != 0)
            inst.affordance.on_node_modified(aid);

        // Mission controller clearing epistemic_pending on the object node itself.
        if (auto node_opt = G->get_node(cabinet_id); node_opt.has_value())
        {
            const auto v = G->get_attrib_by_name<epistemic_pending_att>(node_opt.value());
            if (v.has_value() and not v.value())
                inst.epistemic_pending = false;
        }
    }
}
void SpecificWorker::del_node_slot(std::uint64_t id)
{
    // The del_node signal is connected BEFORE fitter_ is constructed, and initialize()'s startup
    // stale-node sweep (remove_owned_cabinet_nodes) deletes leftover "cabinet_*" nodes — firing this slot
    // while fitter_ is still null. Guard it (a leftover node has no live instance to notify anyway).
    if (not fitter_)
        return;
    // Notify affordance in case its own DSR node was deleted externally
    for (auto& [cabinet_id, inst] : fitter_->instances())
        if (inst.affordance.node_id() == id)
            inst.affordance.on_node_deleted(id);

    if (fitter_->instances().count(id))
    {
        std::print("cabinet_concept: node {} removed from DSR, destroying instance\n", id);
        fitter_->instances().erase(id);
    }
}

// ─── Lifecycle stubs ─────────────────────────────────────────────────────────────────────────────

void SpecificWorker::emergency()
{
    std::print("cabinet_concept: emergency()\n");
}

void SpecificWorker::restore()
{
    std::print("cabinet_concept: restore()\n");
}

int SpecificWorker::startup_check()
{
    std::print("cabinet_concept: startup_check()\n");
    return 0;
}




