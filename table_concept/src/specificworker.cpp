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

#include "../../common/diag_log/rotating_csv.h"   // keep the previous run instead of wiping it

#include "../../common/obj/convergence.h"   // rc::converge::step (SHARED)

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
#include <QDateTime>   // wall-clock ms for the primary-input stream gate (stall grace baseline)

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

    // ★SEVER THE GRAPH CALLBACKS BEFORE ANY TEARDOWN. This agent connects del_node_signal to del_node_slot,
    // and cleanup_owned_nodes() below calls G->delete_node() — from the MAIN thread, where emitter and
    // receiver share a thread and Qt's Auto connection therefore resolves to DIRECT. So the slot re-enters
    // synchronously while the object is being torn down. bottle_concept severed the callbacks here and
    // wrote down the symptom it was fixing ("a del_node delta ... on this already-destructing object — the
    // exit segfault"); the other six kept the connection live. Dropping inner_eigen_ afterwards (while G is
    // still fully alive) also unsubscribes its internal graph signals cleanly.
    if (G)
        disconnect(G.get(), nullptr, this, nullptr);

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

    // SHARED (common/agent_exit) — the reasoning for every step, including why G->reset() cannot be skipped,
    // now lives with the code instead of in seven copies of the same comment block.
    rc::agent::terminal_exit([this] { request_shutdown(); },
                             [this] { if (G) G->reset(); });
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

    // ★THE WHOLE PRESENCE PROTOCOL IS SHARED (common/concept_presence). The 134 lines that used to sit here
    // were byte-identical across hood/refrigerator/table and differed elsewhere only in comment wording — while
    // hiding two agents that had SILENTLY DROPPED a rule (chair and door swept their own live affordances on
    // every Degraded bounce). The four rules it encodes, and the gate state the transitions own
    // (operating_since_ms / stall_reported / degraded_from_input / first_operating_done), now live with the code
    // that enforces them; only the seams below are this agent's.
    presence_protocol_.wire(
        presence_coordinator_, &statemachine, this, configLoader, G, static_cast<std::uint32_t>(agent_id),
        {
            .shutting_down      = [this] { return shutting_down_.load(); },
            // This agent POLLS a graph node for its primary input (no free-running reader thread), so the
            // ingest must be pumped while Waiting or liveness never advances.
            .pump_primary_input = [this] { if (mask_ingestor_) mask_ingestor_->refresh(); },
            .primary_age_ms     = [this] { return mask_ingestor_ ? mask_ingestor_->ms_since_last_frame() : -1; },
            .primary_live       = [this] { return masks_stream_live(); },
            .primary_stalled    = [this](std::int64_t *age) { return masks_stream_stalled(age); },
            .emit_ready         = [this] { emit presenceReady(); },
            .emit_lost          = [this] { emit presenceLost(); },
            .compute            = [this] { compute(); },
            .terminal_shutdown  = [this] { terminal_shutdown(); },
            // One-shot, POST-SYNC: leftovers from a crashed previous run. The initialize() sweep can run before
            // those nodes arrive from the persistent DSR server, and this fires before the first compute().
            .on_first_operating = [this] { remove_stale_affordance_nodes(); },
            .on_optional_peer_lost  = [this](const std::string &name, std::uint32_t id)
                                      { on_optional_peer_lost(name, id); },
            .on_optional_peer_ready = [this](const std::string &name, std::uint32_t id)
                                      { on_optional_peer_ready(name, id); },
        });

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
    // Observer pose → view bearing. SHARED (common/phantom_log/observer_pose.h): the classifier failure is
    // VIEWPOINT-dependent, so the false-alarm field is keyed on (world cell × bearing), never place alone.
    rc::history::note_observer(e, inner_eigen_.get(), x, y);
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
    if (fitter_) fitter_->set_foreign_claims(&foreign_claims_);   // the FIT judges the same geometry

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

    // The detector's truth table. BEFORE removal on purpose: an instance killed this cycle must still
    // leave the row that records the look which killed it — that row is the whole point of the file.
    log_detect_probe();

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

    // Believed table footprints (room frame) — a region under one is already explained, NOT a birth. This is
    // the one genuinely per-object step: a table's two footprint extents. Everything after it — both CSVs,
    // the fusion probe, the throttled console line — is SHARED (common/birth_surprise/birth_surprise_log.h).
    std::vector<rc::FootprintBox> footprints;
    for (const auto& [id, inst] : fitter_->instances())
        if (inst.ai2_initialized)
        { const auto& s = inst.ai2_belief.state(); footprints.push_back({s.cx, s.cy, s.w, s.h, s.yaw}); }

    birth_surprise_log_.write(residual_field_, footprints, last_table_dets_xy_,
                              ev_g_.births, fitter_->instances().size());
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

    // The convergence RULE (counter, verdict, graph writes) is shared — common/obj/convergence.h. What stays
    // here is the one per-object part: which DOFs enter state_delta, above.
    const float model_uncertainty = rc::geom::belief_uncertainty(inst);
    const bool stable_edge = rc::converge::step(*G, node, state_delta, model_uncertainty,
                                                inst.frames_converged, inst.model_stable,
                                                {cfg_.state_eps, cfg_.K_stable});
    if (fitter_->should_log(inst))
        std::print("[{}] convergence: Δstate={:.4f} stable={}/{} U(Σ)={:.3f}m\n",
                   inst.node_name, state_delta, inst.frames_converged, cfg_.K_stable, model_uncertainty);

    if (stable_edge and inst.model_stable)
        std::print("table_concept: node '{}' STABLE (F={:.4f})\n", inst.node_name, free_energy);
}

// Publish/refresh the epistemic next-best-view affordance for one table from its belief Σ (D-optimal NBV),
// with a post-completion cooldown that suppresses the published gain so a just-finished table isn't re-claimed.
void SpecificWorker::step_epistemic(rc::TableInstance& inst, DSR::Node& node)
{
    // The cycle around the planner call is SHARED (common/epistemic_step): the cooldown, the
    // controller-completion hold, the gain suppression and verification floor, the affordance refresh, and —
    // the reason it is shared rather than merely deduplicated — the single bail path that ALWAYS calls
    // hold_offered(). Only the planner call and what this agent records are table's.
    rc::epistemic::StepHooks<rc::EpistemicProposal> hooks;

    hooks.compute = [&]() -> std::optional<rc::EpistemicProposal>
    {
        // Σ-based D-optimal NBV from the belief. Skip until the belief has seen its first frame (else Σ is
        // the broad prior and the proposal is moot).
        if (not inst.ai2_initialized)
            return std::nullopt;
        rc::EpistemicProposal prop =
            epistemic_planner_.compute(inst.ai2_belief, cfg_.ai2_range_noise_lat_per_m, cfg_.ai2_sigma_base_m,
                                       rc::nbv::sensor_from_graph(*G, inner_eigen_.get()),
                                       collect_viewpoint_obstacles(inst.node_id),
                                       // The reachable region — kills the through-the-wall faces the
                                       // direction-blind gain cannot tell apart. Empty until room_concept
                                       // publishes; is_reachable then imposes no constraint (prior behaviour).
                                       room_polygon_);
        // Degenerate fit, or the camera model is still incomplete — retry next cycle.
        if (not prop.valid or not prop.is_finite())
            return std::nullopt;
        return prop;
    };

    hooks.record = [&](const rc::EpistemicProposal& published, float raw_gain)
    {
        // Mirror the proposal onto the instance so ai2_log records what was actually proposed THIS cycle. The
        // affordance node carries the FROZEN pose while the controller owns a claim, so neither the graph nor
        // stdout could answer "what is it proposing now?" after the fact.
        // ⚠table logs the RAW gain (pre-suppression) where bottle/cabinet/chair log the published one — see
        // the divergence note in common/epistemic_step. Behaviour preserved here; resolve it deliberately.
        inst.dbg_nbv_standoff = published.chosen_standoff_m;
        inst.dbg_nbv_gain_raw = raw_gain;                     // what the belief asked for
        inst.dbg_nbv_gain_pub = published.epistemic_gain;     // what the controller will see
        inst.dbg_nbv_target_x = published.epistemic_target_x_m;
        inst.dbg_nbv_target_y = published.epistemic_target_y_m;
        inst.dbg_nbv_pdetect  = published.chosen_p_detect;
        inst.dbg_nbv_fill     = published.chosen_fill;
        inst.dbg_nbv_vfov     = published.sensor_vfov_rad;
        // Write attributes to the table node (read by legacy consumers)
        scene_graph_->write_epistemic_proposal(node, published);
    };

    hooks.on_affordance_created = [this] { trigger_graph_layout_twopi(); };

    rc::epistemic::step(inst, *G,
                        {.cooldown_cycles   = cfg_.epistemic_cooldown_cycles,
                         // Verification pull: absence never deletes a table the robot hasn't properly looked
                         // at — it sends the robot to look.
                         .verify_gain_floor = inst.wants_verification ? cfg_.existence_verify_gain : 0.0f},
                        hooks);
}

// ─── DSR helpers ─────────────────────────────────────────────────────────────────────────────────

// Re-run the graph viewer's twopi layout now and again once queued, so it also settles after the pending
// node/edge update signals are processed. No-op when Agent.graph is off (no viewer widget).
void SpecificWorker::trigger_graph_layout_twopi()
{
    // SHARED (common/graph_layout) — pure viewer plumbing. See the header for why the layout runs twice.
    rc::gui::trigger_layout_twopi(graph_viewers);
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
    if (not fitter_ or not G)
        return;   // may be called before the fit core exists — the guard every graph reader here needs
    // SHARED (common/object_affordance): see rc::poll_protocol for why this is POLLED and not driven by
    // update_node_attr_signal, and for what happened to the one agent that had neither.
    rc::poll_protocol(fitter_->instances(), *G);
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




