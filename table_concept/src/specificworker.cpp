/*
 *    Copyright (C) 2026 by RoboComp CORTEX Team
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or *    (at your option) any later version.
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
 * specificworker.cpp — table_concept agent orchestration.
 *
 * Per compute() cycle: ingest the ZED YOLO "table" masks, associate them to table instances with the shared
 * InstanceTracker (birth / associate / merge; ricoh 360 detections are bearing-only and only raise attention),
 * run one AI2 recursive-Laplace belief update (TableBelief) per assigned slice via process_table_node, write
 * the fitted pose+geometry back to DSR (RT edge + dims + mesh + covariance), and emit epistemic action
 * proposals when a table stays under-observed. Also feeds the standalone belief dashboard + evidence monitor
 * windows. The fit core is rc::TableFitter, perception rc::MaskIngestor, DSR I/O rc::TableSceneGraph. See
 * TABLE.md for the belief/fit core.
 */

#include "specificworker.h"

#include "../../common/dashboard/belief_series.h"   // rc::dash::publish_belief_series (SHARED)

#include "../../common/birth_surprise/residual_field_reader.h"   // rc::read_residual_field (SHARED)
#include "../../common/exclusion/exclusion.h"   // rc::exclusion:: (SHARED)
#include "../../common/nbv/graph_obstacles.h"   // rc::nbv::collect_graph_obstacles — shared, DSR-side
#include "table_geometry.h"   // rc::geom pure footprint/uncertainty helpers

#include <locale>
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
    std::print("table_concept: SpecificWorker destroyed.\n");
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
    std::print("table_concept: initialize()\n");
    GenericWorker::initialize();
 
    // Ignore payload attributes in local graph updates to avoid unnecessary copying and processing of potentially large data
    G->set_ignored_attributes<cam_rgb_att, cam_depth_att, laser_X_att, laser_Y_att, laser_Z_att>();
    qInfo() << "Ignoring DSR RGBD payload attributes cam_rgb/cam_depth in local graph updates";


    if (not G)
    {
        qWarning() << "table_concept: DSR graph not available in initialize()";
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
            // Pump the masks ingest WHILE Waiting: table polls a graph node (no free-running ingest thread
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
    // A/B ONLY (Masks.legacy_room_frame): read the producer's pre-flip ROOM-frame array instead of
    // transforming the camera-frame one here. Announce it — a silent A/B knob is how a benchmark ends up
    // measuring the wrong build. Normal operation is false; see table_config.h.
    if (cfg_.masks_legacy_room_frame)
    {
        mask_ingestor_->set_legacy_room_frame(true);
        std::println("[table_concept] Masks.legacy_room_frame=true — A/B baseline: reading the retina's "
                     "ROOM-frame mask array, NOT transforming camera points here");
    }
    scene_graph_ = std::make_unique<rc::TableSceneGraph>(
        G, rt_api_.get(), cfg_, [this] { trigger_graph_layout_twopi(); });

    // Subscribe to graph signals.
    //
    // ★NOT update_node_attr_signal. It fires for EVERY attribute change on EVERY node in the shared graph —
    // every RT pose write from robot_concept, every other agent's per-cycle attribute writes — and each
    // emission copies a std::vector<std::string> on the DDS reader thread and queues an event to this thread.
    // Under a churn burst (a peer restarting is enough) the queue drains slower than it fills, and the main
    // thread then does nothing but service slots: the timer-driven compute() is starved (ai2_log stops
    // growing), and — the symptom that costs the most — Ctrl-C dies, because generated/main.cpp routes SIGINT
    // through a QSocketNotifier serviced by this SAME event loop. The agent can then only be killed with -9,
    // which cannot be caught, so every node it owns LEAKS into the shared graph.
    // Measured on THIS agent 2026-08-07: main thread pegged at 100% of a core (301 jiffies/3 s), ai2_log.csv
    // frozen, Ctrl-C inert, right after a retina restart. residual_concept did the same on 08-06 under the
    // same trigger. CLAUDE.md already states the rule this violated: if you don't need a signal, don't connect
    // it at all (bottle_concept connects none).
    // The two things the slot did are now POLLED once per cycle in poll_affordance_protocol() — a controller
    // claim does not need sub-cycle latency, so nothing is lost and the firehose is gone.
    connect(G.get(), &DSR::DSRGraph::update_node_signal,
            this, &SpecificWorker::modify_node_slot);
    connect(G.get(), &DSR::DSRGraph::del_node_signal,
            this, &SpecificWorker::del_node_slot);

    // Remove any "table*" nodes left behind by a previous (crashed) run so this agent always starts
    // from a clean slate and never adopts a stale/drifted node (the instance tracker re-births them
    // data-driven from masks).
    remove_owned_table_nodes();

    // Resolve room node
    const auto rooms = G->get_nodes_by_type("room");
    if (not rooms.empty())
        room_node_id_ = rooms.front().id();
    else
        qWarning() << "table_concept: no room node found at startup";

    // Active-inference fit core. Owns the instance map; collaborates with the ingestor + scene graph.
    fitter_ = std::make_unique<rc::TableFitter>(
        G, inner_eigen_.get(), cfg_, mask_ingestor_.get(), scene_graph_.get());
    existence_ = std::make_unique<rc::TableExistence>(G, cfg_);   // evidence-based removal (existence log-odds)

    // Part B: localization/chain covariance on the published RT edge (mirrors bottle_concept).
    gaussian_api_ = std::make_unique<DSR::InnerGaussianAPI>(G.get());
    fitter_->set_chain_cov_source(gaussian_api_.get(), "zed");
    // Object-anchor observation z_o for room_concept's landmark factor (expressed in the localizer base frame).
    fitter_->set_object_observation(cfg_.publish_object_obs, cfg_.object_obs_frame);

    // YOLO-independent LiDAR range channel: lidar3D media-plane consumer that stages each cycle's sweep in the
    // room frame for the fitter's range factor. Dormant (no DDS participant) unless TableModel.LidarPrecision
    // > 0. Subscriber is brought up lazily on the compute/main thread once the lidar3D node + descriptor exist.
    lidar_ingestor_ = std::make_unique<rc::ConceptLidarIngestor>(G, inner_eigen_.get(),
        [this] { return rc::LidarGates{cfg_.lidar_precision, cfg_.free_space_precision, cfg_.lidar_bpearl_precision}; });

    // Build rc::EpistemicPlanner (info-gain scoring only; stand-off distance is the sole parameter).
    epistemic_planner_ = rc::EpistemicPlanner(cfg_.obs_distance);
    // ONE detector envelope, both directions: the viewpoint the planner asks for is the argmax of the same
    // model absence should be weighted by. Now config-driven (TableModel.DetectMinFill/MaxFill/Soft); the
    // defaults are still the fleet prior, so nothing changes until etc/config.toml sets them.
    const rc::detect::DetectorEnvelope det_env{cfg_.detect_min_fill, cfg_.detect_max_fill, cfg_.detect_soft};
    epistemic_planner_.set_detector_envelope(det_env);
    existence_->set_detector_envelope(det_env);   // …and REMOVAL weights absence by the very same model, which
                                                  // is what "both directions" above has always meant. Until
                                                  // 2026-08-06 only the planner used it and removal kept a
                                                  // one-sided range curve that deleted the real table at 0.46 m.

    // The camera's REAL geometry, read once (intrinsics and the zed mount are both static). BOTH FoVs: for a
    // low, wide tabletop viewed from a camera at ~1 m the VERTICAL axis is the one that binds, and the
    // horizontal-only model this replaced could not see it. Height is the zed optical centre above the room
    // floor — ts==0 on the main thread, which is the only safe way to use that cache (CLAUDE.md).
    // ★The camera model is read PER CYCLE at the compute site (rc::nbv::sensor_from_graph),
    // NOT once here: the zed intrinsics are published by robot_concept when frames start
    // arriving, so reading them in initialize() races the producer. Losing that race leaves
    // vfov = 0, which silently collapses the fill model to horizontal-only — the exact bug
    // rc::nbv exists to fix, and it drives the robot nose-to-nose with tall objects.

    // Stale affordance nodes are swept on entering Operating (presence hook) and on shutdown — see
    // remove_stale_affordance_nodes(), keyed on the parent object type (robust to node-name renames).

    // Shadow-mode birth/death record (CONCEPT_AGENT_LIFECYCLE.md §4.2). Recording only — see
    // log_phantom_event(). Truncating, like the sibling *_events.csv writers: one file per run.
    phantom_log_.open("etc/table_phantom_events.csv");

    // Standalone Qt dashboard + evidence-monitor windows (belief plots + per-instance snapshot).
    build_dashboard();
}

// ─── Main compute loop ───────────────────────────────────────────────────────────────────────────

// Publish the room's wall polygon to the fitter → projection, where it becomes the silhouette channel's
// LINE-OF-SIGHT test. Without it "predicted visible" was decided by the camera frustum plus occlusion by
// whatever OTHER objects YOLO happened to segment — and a wall is not a YOLO class, so a wall could never
// occlude anything: from the next room a table still projects into the image, lands on wall pixels carrying
// no table mask, and every sample votes ABSENCE at full strength until the instance is deleted.
// Identical bug, identical fix and identical polygon source as refrigerator_concept and door_concept; the
// crossing test itself is the shared rc::occlusion::walls_block(). Polygon is in ROOM-frame METRES (verified
// against both sibling consumers, which use it unscaled). Cheap: a few dozen floats, re-read per cycle so a
// re-localised or re-fitted room takes effect immediately. No room node / no attribute ⇒ silently inactive,
// which is exactly the pre-existing behaviour.
void SpecificWorker::refresh_room_geometry()
{
    if (not G or room_node_id_ == 0 or not fitter_) return;
    // LATCH. The polygon is authored ONCE and never edited: room_scene_graph.cpp writes it at room-node
    // creation and otherwise only backfills it if missing (guarded by `not has_poly`). Re-localisation moves
    // the room FRAME, not these vertices. Re-reading it every cycle cost a full G->get_node(), which returns
    // the Node BY VALUE and deep-copies both its attribute map and its m_fano edge map — and the room node is
    // the RT parent of the robot and every concept instance, so that is on the order of hundreds of
    // allocations per call, ~10×/s, under the graph's shared_mutex where it contends with the FastDDS reader
    // threads. All to re-read six floats that cannot have changed.
    // Re-arms when the room node id changes, which is the only way a different polygon can appear.
    if (polygon_room_id_ == room_node_id_) return;
    const auto room = G->get_node(room_node_id_);
    if (not room.has_value()) return;
    const auto px = G->get_attrib_by_name<delimiting_polygon_x_att>(room.value());
    const auto py = G->get_attrib_by_name<delimiting_polygon_y_att>(room.value());
    if (not px.has_value() or not py.has_value()) return;
    const auto& xs = px->get(); const auto& ys = py->get();
    const std::size_t n = std::min(xs.size(), ys.size());
    if (n < 3) return;                       // degenerate ⇒ leave the test inactive rather than half-armed
    std::vector<Eigen::Vector2f> poly;
    poly.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        poly.emplace_back(xs[i], ys[i]);
    room_polygon_ = poly;                       // keep a copy for the NBV's reachability test
    fitter_->set_room_polygon(std::move(poly));
    polygon_room_id_ = room_node_id_;   // latched; re-reads only if the room node is replaced
    std::print("table_concept: [room] wall polygon loaded ({} verts) — silhouette line-of-sight ARMED\n", n);
}

// SHADOW-MODE birth/death recorder — CONCEPT_AGENT_LIFECYCLE.md §4.2, theory in MODEL_HISTORY.md §4.
// RECORDS ONLY. Nothing here feeds back into a belief, a birth or a removal, and it must stay that way until
// the removal path is validated: a birth→death event is evidence of EITHER a classifier false positive OR a
// removal false positive, and the attribution fields below (p_detect / in_fov / central / fixated) are what
// tells the two apart. Deaths clustering where p_detect was LOW mean the log is recording our own removal
// bugs, not YOLO's errors — which is exactly the check that gates arming the p_FA field.
void SpecificWorker::log_phantom_event(std::string_view event, std::uint64_t id, std::string_view name,
                                       float x, float y, const rc::TableInstance* inst, std::string_view note)
{
    if (not phantom_log_.is_open())
        return;
    rc::history::PhantomEvent e;
    e.event = event; e.id = id; e.name = name; e.x = x; e.y = y; e.note = note;
    // Observer pose → view bearing. The classifier failure is VIEWPOINT-dependent (a radiator only reads as a
    // chair from certain angles), so the eventual false-alarm field is keyed on (world cell × bearing); a
    // place-only key would suppress a genuine object placed there from every direction.
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
    if (inst)   // death: carry the existence-channel state that decides whether this was a CONFIDENT kill
    {
        e.age_cycles    = inst->processed_cycles;
        e.p_detect      = inst->dbg_ex_pdetect;
        e.central_frac  = inst->dbg_ex_central;
        // The REAL fraction (detectable / attempted), not a 0-or-1 stand-in: the attribution question is
        // "how much of the object could the sensor actually have seen", which a bare probed/not-probed flag
        // cannot answer — and a flag in a column documented as a fraction reads as data while carrying none.
        e.in_fov_frac   = (inst->dbg_ex_sil_ntotal > 0)
                        ? static_cast<float>(inst->dbg_ex_sil_ndet) / inst->dbg_ex_sil_ntotal : 0.0f;
        e.fixated       = inst->dbg_fixated ? 1 : 0;
        e.exist_logodds = inst->existence.logodds();
    }
    phantom_log_.write(e);
}

void SpecificWorker::compute()
{
    // ★ONE graph walk per cycle for the SHARED mutual-exclusion rule: who else claims room space.
    // Feeds BOTH the birth filter (a candidate on somebody else's object accrues no evidence) and
    // the existence occupancy discount. Main thread — collect_graph_obstacles uses ts==0 (CLAUDE.md).
    if (G) foreign_claims_ = rc::exclusion::foreign_claims(*G, inner_eigen_.get(), "table");
    if (existence_) existence_->set_foreign_claims(&foreign_claims_);

    if (not G or not rt_api_)
        return;

    // Refresh room node id if not yet found (see refresh_room_geometry below for the walls)
    if (room_node_id_ == 0)
    {
        const auto rooms = G->get_nodes_by_type("room");
        if (rooms.empty()) return;
        room_node_id_ = rooms.front().id();
    }

    refresh_room_geometry();   // room walls → the silhouette line-of-sight test (see below)

    // Controller-owned affordance flags (claim / completion / epistemic_pending). Polled here rather than
    // pushed by update_node_attr_signal — see the connect block in initialize() for why that subscription
    // could starve this very loop.
    poll_affordance_protocol();

    const bool fresh_masks = mask_ingestor_->refresh();

    // EvidenceMonitor per-cycle counters (cumulative *_cum fields persist across cycles). The producers below
    // (tracker / merge / removal) add to these; the snapshot is pushed at the end of the cycle.
    ev_g_.births = ev_g_.merges = ev_g_.removals = 0;
    ev_g_.mask_stale = not fresh_masks;

    // Snapshot the residual (surprise) field ONCE per cycle before the tracker, so fused birth (run_instance_
    // tracker) and the logging probe (compute() tail) share one read. No-op unless a probe/fusion flag is on.
    if (cfg_.birth_fusion or cfg_.birth_surprise_probe) read_residual_field();

    run_instance_tracker();   // data-driven birth / associate / merge (the only instance-lifecycle path)

    // Stage this cycle's LiDAR sweep (room frame) for the fitter's range factor. clear-then-set so the factor
    // never consumes a stale sweep; pump() is main-thread (reads the graph) + dormant while the feature is off.
    fitter_->clear_lidar_sweep();
    bool fresh_sweep = false;
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

    const auto table_nodes = G->get_nodes_by_type("object");
    for (const auto& node : table_nodes)
        if (node.name().starts_with("table"))
            process_table_node(node);

    // Ricoh 360 = peripheral attention: associate ricoh detections to tables BY DIRECTION (after the ZED fits,
    // so table positions are current); an unassigned bearing becomes a "seek a ZED view here" attention target.
    process_ricoh_bearings();

    // Evidence-based removal: each existence channel integrates on its OWN sensor cadence (silhouette/mask on a
    // fresh mask frame, LiDAR carve on a fresh sweep) — a camera-only cycle still accrues absence, a LiDAR-only
    // cycle still carves free space. After the fits so footprints are current. OFF unless enabled.
    if (cfg_.existence_removal_enabled)
        existence_->update_and_remove(*fitter_, lidar_ingestor_.get(), fresh_masks, fresh_sweep, ev_g_,
            [this](std::uint64_t id, const rc::TableInstance& inst)
            {   // shadow-mode death record (§4.2) — carries the p_detect/fixated state that says whether this
                // was a CONFIDENT disconfirmation (a real phantom) or a weak one (likely our own removal bug)
                // note is for what the SCHEMA cannot express; the log-odds already has its own column
                // (PhantomEvent::exist_logodds), so formatting it in here would duplicate it at lower
                // precision and leave the analysis guessing which is authoritative.
                log_phantom_event("DEATH", id, inst.node_name,
                                  inst.ai2_belief.state().cx, inst.ai2_belief.state().cy, &inst, "");
            });

    // ── Evidence monitor: global counters + throttled snapshot push ──
    ev_g_.instances    = static_cast<int>(fitter_->instances().size());
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
    prune_dead_series();   // drop timeseries lines for tables that were removed from the graph
    log_birth_surprise();  // EXPERIMENTAL read-only: residual grid → birth-surprise regions (cfg-gated, off by default)
    fps_counter_.print("[table_concept Compute]");
}

// EXPERIMENTAL, read-only. Reinterpret residual_concept's `grid` node (under room) as an unexplained-occupancy
// SURPRISE field and cluster its confidently-unexplained cells into candidate regions (birth_surprise_probe.h).
// Log, per cycle, the regions NOT already covered by a believed table alongside the tracker's actual birth count,
// so we can see whether surprise flags a real new table cleanly (and stays quiet on phantoms) BEFORE letting it
// drive the lifecycle. Never writes the graph; never births. See [[table-birth-surprise-probe]].
// Snapshot residual_concept's `grid` node (published under room ~2 Hz) into residual_field_ — the dense
// P(occupied ∧ ¬explained) surprise field. Read-only; never mutates the graph. Called once at the compute() head
// so BOTH the fused-birth path (run_instance_tracker) and the logging probe (log_birth_surprise) share one read.
bool SpecificWorker::read_residual_field()
{
    return rc::read_residual_field(*G, room_node_id_, residual_field_);
}

void SpecificWorker::log_birth_surprise()
{
    if (not cfg_.birth_surprise_probe or not residual_field_.valid()) return;
    const rc::GridField& gf = residual_field_;   // read at the compute() head by read_residual_field()

    // Believed table footprints (room frame) — a region under one is already explained, NOT a birth.
    std::vector<rc::FootprintBox> tables;
    for (const auto& [id, inst] : fitter_->instances())
        if (inst.ai2_initialized)
        { const auto& s = inst.ai2_belief.state(); tables.push_back({s.cx, s.cy, s.w, s.h, s.yaw}); }

    const auto cands = rc::BirthSurpriseProbe::scan(gf, tables);
    const long cyc = ++birth_surprise_cycle_;   // advances only on cycles where the grid field was actually read
    int n_birth = 0;                       // uncovered high-surprise regions = birth candidates
    for (const auto& c : cands) if (not c.covered_by_concept) ++n_birth;

    // CSV: one row per region per cycle (covered flag distinguishes birth candidates from explained mass; the
    // latter should be ~0 if residual_concept's concept-subtraction is working — a free sanity check).
    if (not birth_surprise_csv_.is_open())
    {
        birth_surprise_csv_.open("etc/birth_surprise.csv", std::ios::out | std::ios::trunc);
        birth_surprise_csv_.imbue(std::locale::classic());   // ★Qt imbues the GLOBAL locale, so operator<< inserts THOUSANDS
                                            // SEPARATORS into integers (pkt_ts 1785763853131 → "1,785,763,853,131"),
                                            // splitting one CSV field into five. Field counts then vary per row and
                                            // the whole log is unreadable by column name — every value past the
                                            // first big integer is shifted, which silently invalidates any analysis.
                                            // Pin "C" so the log is machine-readable regardless of the UI locale.
        if (birth_surprise_csv_.is_open())
            birth_surprise_csv_ << "cycle,region,cx,cy,cells,mass,ext_x,ext_y,mean_p,mean_var,covered,"
                                << "n_tables,tracker_births,instances\n";
    }
    if (birth_surprise_csv_.is_open())
    {
        int r = 0;
        for (const auto& c : cands)
            birth_surprise_csv_ << cyc << ',' << r++ << ',' << c.cx << ',' << c.cy << ',' << c.cells << ','
                                << c.mass << ',' << c.ext_x << ',' << c.ext_y << ',' << c.mean_p << ',' << c.mean_var
                                << ',' << (c.covered_by_concept ? 1 : 0) << ',' << tables.size() << ','
                                << ev_g_.births << ',' << fitter_->instances().size() << '\n';
        birth_surprise_csv_.flush();
    }

    // ── FUSION readout: residual surprise MASS under each YOLO "table" detection (birth_fusion.csv). The measured
    //    quantity: does a real detection land on high unexplained-occupancy (→ corroborated → birth fast/confident)
    //    while a flicker/phantom detection lands on ~0? covered = the detection sits inside an already-believed
    //    table footprint (associate, not birth). This is the signal that would let residual GATE/accelerate birth.
    if (not birth_fusion_csv_.is_open())
    {
        birth_fusion_csv_.open("etc/birth_fusion.csv", std::ios::out | std::ios::trunc);
        birth_fusion_csv_.imbue(std::locale::classic());   // ★Qt imbues the GLOBAL locale, so operator<< inserts THOUSANDS
                                            // SEPARATORS into integers (pkt_ts 1785763853131 → "1,785,763,853,131"),
                                            // splitting one CSV field into five. Field counts then vary per row and
                                            // the whole log is unreadable by column name — every value past the
                                            // first big integer is shifted, which silently invalidates any analysis.
                                            // Pin "C" so the log is machine-readable regardless of the UI locale.
        if (birth_fusion_csv_.is_open())
            birth_fusion_csv_ << "cycle,det,det_x,det_y,mass_r05,mass_r03,near_dist,near_mass,covered,"
                              << "n_tables,tracker_births,instances\n";
    }
    if (birth_fusion_csv_.is_open())
    {
        // DIAGNOSTIC block only — the 0.50/0.30 m probe radii and the 0.30 m footprint margin below feed the
        // birth_fusion.csv columns for offline analysis of residual-vs-birth fusion. None of these values touch
        // the live belief or any birth decision, so they are hardcoded (a config key would imply they matter at
        // runtime). If residual-gated birth is ever wired in, promote them then.
        int di = 0;
        for (const auto& d : last_table_dets_xy_)
        {
            const float m05 = rc::BirthSurpriseProbe::residual_mass_near(gf, d.x(), d.y(), 0.50f);
            const float m03 = rc::BirthSurpriseProbe::residual_mass_near(gf, d.x(), d.y(), 0.30f);
            float nd = 1e9f, nm = 0.f;                      // nearest region to this detection
            for (const auto& c : cands)
            { const float dd = std::hypot(c.cx - d.x(), c.cy - d.y()); if (dd < nd) { nd = dd; nm = c.mass; } }
            bool covered = false;
            for (const auto& t : tables)
            { const float cc = std::cos(t.yaw), ss = std::sin(t.yaw), dx = d.x() - t.cx, dy = d.y() - t.cy;
              if (std::abs(cc*dx + ss*dy) <= 0.5f*t.w + 0.30f and std::abs(-ss*dx + cc*dy) <= 0.5f*t.h + 0.30f)
                  { covered = true; break; } }
            birth_fusion_csv_ << cyc << ',' << di++ << ',' << d.x() << ',' << d.y() << ',' << m05 << ',' << m03 << ','
                              << (nd > 1e8f ? -1.f : nd) << ',' << nm << ',' << (covered ? 1 : 0) << ','
                              << tables.size() << ',' << ev_g_.births << ',' << fitter_->instances().size() << '\n';
            if (ev_g_.births > 0 and not covered)          // a NEW table just born — print its corroboration
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
        for (const auto& c : cands) if (not c.covered_by_concept) { top = &c; break; }   // cands sorted by mass
        if (top)
            std::print("[birth-surprise] uncovered={} tables={} tracker_births={} | top: ({:.2f},{:.2f}) "
                       "mass={:.1f} cells={} ext={:.2f}x{:.2f} mean_p={:.2f} var={:.3f}\n",
                       n_birth, tables.size(), ev_g_.births, top->cx, top->cy, top->mass, top->cells,
                       top->ext_x, top->ext_y, top->mean_p, top->mean_var);
    }
}


// ─── Per-node processing + publish ───────────────────────────────────────────────────────────────

// Process one "table" DSR node this cycle: ensure its instance exists, then fuse each assigned ZED slice
// (one belief update per slice, gated to a fresh mask frame) or age the belief when no mask arrived, and // hand the result to publish_table_cycle. run_instance_tracker has already associated this cycle's slices.
void SpecificWorker::process_table_node(const DSR::Node& node)
{
    const bool created = fitter_->ensure_instance(node, room_node_id_);
    // ensure_instance() may bail (bad RT read, missing room), and its bool return does NOT report that —
    // so look the instance up defensively rather than .at()-ing into a possibly-absent key on the compute path.
    const auto it = fitter_->instances().find(node.id());
    if (it == fitter_->instances().end())
        return;
    auto& inst = it->second;

    if (created)
    {
        // (Per-instance time-series are registered idempotently by publish_table_diagnostics each cycle.)
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

    // Appearance drift (DISPLAY only). Runs every cycle whether or not this table was seen: an unobserved
    // object's colour genuinely becomes less certain as the lighting and viewpoint move on, and without
    // this the belief saturates on the first confident view and can never be corrected.
    {
        const auto tp_now = std::chrono::steady_clock::now();
        if (inst.last_appearance_tp.time_since_epoch().count() != 0)
            inst.appearance.inflate_for_age(
                std::chrono::duration<float>(tp_now - inst.last_appearance_tp).count());
        inst.last_appearance_tp = tp_now;
    }

    float free_energy = 0.0f;
    TableObservation last_obs;
    bool updated = false;

    // Multi-slice fusion: one belief update per assigned ZED slice of this table. Sequential Bayesian updates
    // = joint likelihood for the recursive filter, and each slice keeps its OWN R and common-mode (they don't
    // share a registration error) — which concatenating points into one frame could not express.
    // ONLY on a fresh mask FRAME for this instance: the tracker assigns every cycle (birth/association
    // continuity), but re-fitting the SAME packet each cycle would overcount evidence — gate on a new frame_id.
    const auto& pkt = mask_ingestor_->packet();
    // Freshness gate. mask_frame_id is a PUBLISH counter — it advances on every republish even when the source
    // camera is FROZEN/paused, so gating on it alone re-integrates the SAME capture as independent evidence and // the belief RATCHETS: w/h/yaw rotate + reshape with the robot AND scene stationary (ai2_log.csv symptom —
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
        if (not last_obs.has_fresh_data and inst.matched_frames < cfg_.matched_frames_before_aging)
            return;
        free_energy = fitter_->run_inference(inst, last_obs);
    }

    publish_table_cycle(inst, node, last_obs, free_energy);
}


// Publish one fitted table: persist belief→DSR, assess convergence, then push diagnostics + intentions.
// Each step short-circuits if its node lookup fails (the node may have been removed mid-cycle).
void SpecificWorker::publish_table_cycle(rc::TableInstance& inst,
                                         const DSR::Node& node,
                                         const TableObservation& observation,
                                         float free_energy)
{
    const auto node_id = node.id();
    if (not scene_graph_->persist_table_belief(inst, node_id, room_node_id_, free_energy))
        return;
    if (not assess_table_state(inst, node_id, free_energy))
        return;
    publish_table_diagnostics(inst, observation, free_energy);
    publish_table_intentions(inst, node_id, observation, free_energy);
}

// Run the convergence/stability step for one table; re-resolves the node by id (false if it is gone).
bool SpecificWorker::assess_table_state(rc::TableInstance& inst, uint64_t node_id, float free_energy)
{
    auto node_opt = G->get_node(node_id);
    if (not node_opt.has_value())
        return false;

    step_convergence(inst, node_opt.value(), free_energy);
    return true;
}

// Feed this instance's time-series to the standalone dashboard: FE + baseline, FE-surprise, U(Σ) covariance,
// residual count, inferred dims (w,h), and size posterior σ. Series are added idempotently every cycle.
void SpecificWorker::publish_table_diagnostics(const rc::TableInstance& inst,
                                               const TableObservation& observation,
                                               float free_energy)
{
    rc::dash::publish_belief_series({ts_plot_, ts_surprise_plot_, ts_cov_plot_, ts_res_plot_},
                                   {.node = inst.node_name,
                                     .free_energy  = free_energy,
                                     .fe_baseline  = inst.fe_baseline,
                                     .fe_surprise  = inst.fe_surprise,
                                     .uncertainty  = rc::geom::belief_uncertainty(inst),
                                     .residual_pts = static_cast<float>(inst.dbg_resid_pts)});

    if (fitter_->should_log(inst))
        std::print("[{}] series: FE={:.4f} U(Σ)={:.3f} res={}\n",
                   inst.node_name, free_energy, rc::geom::belief_uncertainty(inst), observation.residual_pts.size());
}

void SpecificWorker::publish_table_intentions(rc::TableInstance& inst,
                                              uint64_t node_id,
                                              const TableObservation& observation,
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
    cfg_ = rc::load_table_config(cfg);
}



// Every OTHER object the graph knows about, as robot-inflated oriented footprints, so the NBV never proposes a
// viewpoint standing on the furniture or looking through it. Reads `object` and `box` nodes, skips this table,
// and inflates each footprint by the robot radius so a point-in-rectangle test at the viewpoint is equivalent
// to a footprint overlap. A node with no footprint attrs, or one inner_eigen cannot locate, is SKIPPED rather
// than guessed at. Mirrors refrigerator_concept.
std::vector<rc::EpistemicPlanner::Obstacle> SpecificWorker::collect_viewpoint_obstacles(std::uint64_t self_id) const
{
    // One shared implementation (common/nbv/graph_obstacles.h) — this used to be a local copy that read the
    // deprecated obj_width/obj_depth and pre-inflated by the robot radius. Both bugs are documented there.
    return rc::nbv::collect_graph_obstacles(*G, inner_eigen_.get(), self_id);
}

// ─── Per-cycle steps ─────────────────────────────────────────────────────────────────────────────

// Convergence latch for one table: measure how far the accepted state moved this cycle and, once it holds
// still for K_stable cycles, publish model_stable + the model_uncertainty_att readout (posterior std sum).
void SpecificWorker::step_convergence(rc::TableInstance& inst,
                                       DSR::Node& node,
                                       float free_energy)
{
    // Convergence on STATE stability, not |ΔFE|: the free energy keeps jittering with queue
    // churn / point-count even when the fitted geometry is settled, so it never latched. Track
    // how much the accepted state moved between cycles instead.
    const auto& s = inst.model.state();
    const auto& p = inst.prev_conv_state;
    const float state_delta = inst.has_prev_conv_state
        ? (std::abs(s.cx - p.cx) + std::abs(s.cy - p.cy) + std::abs(s.w - p.w) + std::abs(s.h - p.h) +
           std::abs(s.table_height - p.table_height) + std::abs(s.yaw - p.yaw) + std::abs(s.leg_inset - p.leg_inset))
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
            std::print("table_concept: node '{}' STABLE (F={:.4f})\n",
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

// Publish/refresh the epistemic next-best-view affordance for one table from its belief Σ (D-optimal NBV),
// with a post-completion cooldown that suppresses the published gain so a just-finished table isn't re-claimed.
void SpecificWorker::step_epistemic(rc::TableInstance& inst, DSR::Node& node)
{
    if (inst.epistemic_cooldown > 0)
        --inst.epistemic_cooldown;

    // Controller-completion hold (anti-churn): the table affordance completes on a weak detection
    // (contract goal conf≥0.20), which fires almost instantly — before ΔH has decayed. Start a short
    // cooldown so we don't re-offer a just-completed table while its gain is still high. We do NOT
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
    {
        inst.affordance.hold_offered();   // no proposal is not a withdrawal — see hold_offered()
        return;
    }
    rc::EpistemicProposal prop =
        epistemic_planner_.compute(inst.ai2_belief, cfg_.ai2_range_noise_lat_per_m, cfg_.ai2_sigma_base_m,
                                   rc::nbv::sensor_from_graph(*G, inner_eigen_.get()),
                                   collect_viewpoint_obstacles(inst.node_id),
                                   // The reachable region — kills the through-the-wall faces the
                                   // direction-blind gain cannot tell apart. Empty until room_concept
                                   // publishes; is_reachable then imposes no constraint (prior behaviour).
                                   room_polygon_);
    if (not prop.valid or not prop.is_finite())
    {
        // Degenerate fit, or the camera model is still incomplete — retry next cycle. But do NOT leave
        // the affordance retired while we retry: that is the state the controller reads as "the
        // producer no longer wants this look", and it never recovers on its own.
        inst.affordance.hold_offered();
        return;
    }

    // Mirror the proposal onto the instance so ai2_log records what was actually proposed THIS cycle. The
    // affordance node carries the FROZEN pose while the controller owns a claim, so neither the graph nor
    // stdout could answer "what is it proposing now?" after the fact.
    inst.dbg_nbv_standoff = prop.chosen_standoff_m;
    inst.dbg_nbv_gain     = prop.epistemic_gain;
    inst.dbg_nbv_target_x = prop.epistemic_target_x_m;
    inst.dbg_nbv_target_y = prop.epistemic_target_y_m;
    inst.dbg_nbv_pdetect  = prop.chosen_p_detect;
    inst.dbg_nbv_fill     = prop.chosen_fill;
    inst.dbg_nbv_vfov     = prop.sensor_vfov_rad;

    // Belief→knowledge governor WITHOUT deleting the node: keep publishing the affordance every cycle
    // with its TRUE expected information gain ΔH (nats). A low gain is published as-is so the controller's
    // grounded EFE selection simply doesn't pick it (cost outweighs the small epistemic value), and it
    // re-arms automatically when the belief degrades and ΔH climbs again — no satisfy-latch to get stuck
    // in, no node churn. During the post-completion hold the gain is forced to 0 so a just-finished table
    // isn't re-claimed before its belief has settled.
    if (inst.epistemic_cooldown > 0)
        prop.epistemic_gain = 0.0f;

    // Verification pull (active inference): a table whose predicted absence could NOT be resolved from recent
    // views (wants_verification — far/peripheral/edge-on "I don't see it") does NOT get deleted; it gets a strong
    // epistemic gain so the controller drives to a good ZED viewpoint to CONFIRM-or-remove it. Absence never
    // deletes a table the robot hasn't properly looked at — it sends the robot to look. Overrides the cooldown
    // (a "might be gone" alarm is not anti-chatter-suppressible). Clears once a verifying view resolves it.
    if (inst.wants_verification)
        prop.epistemic_gain = std::max(prop.epistemic_gain, cfg_.existence_verify_gain);

    // Write attributes to the table node (read by legacy consumers)
    scene_graph_->write_epistemic_proposal(node, prop);
    // Publish / refresh dedicated affordance node (persists; update_node refreshes target+gain)
    const auto affordance_node_before = inst.affordance.node_id();
    // Planner internals stay in EpistemicProposal; the producer takes the shared eleven-field view.
    rc::AffordanceTarget tgt;
    tgt.x_m     = prop.epistemic_target_x_m;
    tgt.y_m     = prop.epistemic_target_y_m;
    tgt.yaw_rad = prop.epistemic_target_yaw_rad;
    tgt.gain    = prop.epistemic_gain;
    tgt.valid   = prop.valid;
    tgt.face_gains.assign(prop.face_gains.begin(), prop.face_gains.end());
    tgt.sigma_star.assign(prop.sigma_star.begin(), prop.sigma_star.end());
    tgt.standoff_min_m = prop.standoff_min_m;
    tgt.standoff_max_m = prop.standoff_max_m;
    tgt.framing_fill   = prop.framing_fill;
    inst.affordance.update(tgt);
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
    if (it == graph_viewers.end() or not it->second)
        return;

    QWidget* graph_widget = it->second->get_widget(DSR::DSRViewer::view::graph);
    auto* graph_viewer = qobject_cast<DSR::GraphViewer*>(graph_widget);
    if (not graph_viewer)
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
    // Instance creation is owned by the compute() loop (process_table_node → ensure_instance), which runs
    // AFTER note_birth() each cycle, so the birth seed is in place. The loop also covers externally-created
    // table nodes (every cycle), so nothing is lost by not creating here.
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
        return;   // may be called before the fit core exists — same guard the slots carry

    for (auto& [table_id, inst] : fitter_->instances())
    {
        // Affordance state machine: idle→pending→executing→satisfied, driven by the controller-owned
        // active/pending flags on the affordance node. on_node_modified() re-reads them itself, so handing it
        // the id every cycle is exactly what the signal used to do.
        if (const auto aid = inst.affordance.node_id(); aid != 0)
            inst.affordance.on_node_modified(aid);

        // Mission controller clearing epistemic_pending on the table node itself. The signal carried the
        // changed-attribute list so it could skip the read; polling just reads the flag, which is the same
        // graph lookup the slot did once it decided to look.
        if (auto node_opt = G->get_node(table_id); node_opt.has_value())
        {
            const auto v = G->get_attrib_by_name<epistemic_pending_att>(node_opt.value());
            if (v.has_value() and not v.value())
                inst.epistemic_pending = false;
        }
    }
}

void SpecificWorker::del_node_slot(std::uint64_t id)
{
    // ★A graph slot must be safe at ANY time, including before the fit core exists. The signals are connected
    // in initialize() BEFORE `fitter_` is constructed, and the very next statement — the startup stale-sweep
    // remove_owned_table_nodes() — calls G->delete_node() on the MAIN thread. Emitter and receiver on the same
    // thread makes the Auto connection DIRECT, so this slot ran synchronously with `fitter_` still null and
    // dereferencing it SIGSEGV'd. Only reproducible when a previous run actually left "table*" nodes behind,
    // which is why it lay dormant: a clean start sweeps nothing and never enters this slot.
    // (cleanup_owned_nodes() already guards with `if (G and fitter_)` — same idiom, this slot just lacked it.)
    if (not fitter_)
        return;
    // Notify affordance in case its own DSR node was deleted externally
    for (auto& [table_id, inst] : fitter_->instances())
        if (inst.affordance.node_id() == id)
            inst.affordance.on_node_deleted(id);

    if (fitter_->instances().count(id))
    {
        std::print("table_concept: node {} removed from DSR, destroying instance\n", id);
        fitter_->instances().erase(id);
    }
}

// ─── Lifecycle stubs ─────────────────────────────────────────────────────────────────────────────

void SpecificWorker::emergency()
{
    std::print("table_concept: emergency()\n");
}

void SpecificWorker::restore()
{
    std::print("table_concept: restore()\n");
}

int SpecificWorker::startup_check()
{
    std::print("table_concept: startup_check()\n");
    return 0;
}




