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
 * specificworker.cpp — hood_concept agent orchestration.
 *
 * Per compute() cycle: ingest the ZED YOLO "hood" masks, associate them to hood instances with the shared
 * InstanceTracker (birth / associate / merge; ricoh 360 detections are bearing-only and only raise attention),
 * run one AI2 recursive-Laplace belief update (HoodBelief) per assigned slice via process_hood_node, write
 * the fitted pose+geometry back to DSR (RT edge + dims + mesh + covariance), and emit epistemic action
 * proposals when a hood stays under-observed. Also feeds the standalone belief dashboard + evidence monitor
 * windows. The fit core is rc::HoodFitter, perception rc::MaskIngestor, DSR I/O rc::HoodSceneGraph. See
 * HOOD.md for the belief/fit core.
 */

#include "specificworker.h"

#include "../../common/diag_log/rotating_csv.h"   // keep the previous run instead of wiping it

#include "../../common/obj/convergence.h"   // rc::converge::step (SHARED)

#include "../../common/dashboard/belief_series.h"   // rc::dash::publish_belief_series (SHARED)

#include "../../common/birth_surprise/residual_field_reader.h"   // rc::read_residual_field (SHARED)
#include "../../common/exclusion/exclusion.h"   // rc::exclusion:: (SHARED)
#include "hood_geometry.h"   // rc::geom pure footprint/uncertainty helpers

#include <locale>
#include <print>
#include <format>    // stall-transition log formatting (std::println on cout, survives Verbose=false)
#include <cstdlib>   // std::_Exit — crash-free terminal shutdown
#include <thread>    // brief DDS flush before _Exit
#include <chrono>
#include <iostream>  // std::cout/cerr flush
#include <QSettings>   // persist the standalone dashboard window geometry
#include <QByteArray>
#include <QDateTime>   // wall-clock ms for the primary-input stream gate (presence_protocol_.operating_since_ms(), stall grace)

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
    std::print("hood_concept: SpecificWorker destroyed.\n");
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
    save_dashboard_geometry();
    save_strip_geometry();       // …nor is the compact belief strip   // the standalone dashboard is not in `windows`, so save it explicitly

    if (p_fa_field_.size() > 0 and p_fa_field_.save(p_fa_path_))
        std::print("hood_concept: p_FA field saved ({} cells)\n", p_fa_field_.size());

    cleanup_owned_nodes();

    // Drop the LiDAR + RGB media subscribers BEFORE tearing down the graph/inner_eigen they read (each holds a
    // raw pointer). Mirrors bottle_concept.
    lidar_ingestor_.reset();
    rgb_ingestor_.reset();

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
    std::print("hood_concept: initialize()\n");
    // The learnt p_FA field outlives the run. Absent file ⇒ empty field ⇒ estimate() returns the caller's
    // prior for every cell ⇒ behaviour identical to before it existed.
    if (p_fa_field_.load(p_fa_path_))
        std::print("hood_concept: p_FA field loaded ({} cells)\n", p_fa_field_.size());
    GenericWorker::initialize();

    // One-shot belief self-check (pure Eigen; recovers a synthetic fridge pose+size from a box+clutter cloud,
    // plus the appearance FRONT door-mode resolver + reported-σ_yaw entropy).
    rc::HoodBelief::self_test();
    // Fitter self-check (motion_magnitude robustness + the "be-still-to-update" confirm_only gate).
    rc::HoodFitter::self_test();
    // Appearance door-ness metric self-check (OpenCV; a vertical-lined patch must out-score a plain one).
    rc::HoodProjection::self_test();
    // Birth-burst store self-check (cell dedup, cap, expiry, take-erases, local-consistency δ).
    rc::BirthFragment::self_test();
 
    // Ignore payload attributes in local graph updates to avoid unnecessary copying and processing of potentially large data
    G->set_ignored_attributes<cam_rgb_att, cam_depth_att, laser_X_att, laser_Y_att, laser_Z_att>();
    qInfo() << "Ignoring DSR RGBD payload attributes cam_rgb/cam_depth in local graph updates";


    if (not G)
    {
        qWarning() << "hood_concept: DSR graph not available in initialize()";
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
    scene_graph_ = std::make_unique<rc::HoodSceneGraph>(
        G, rt_api_.get(), cfg_, [this] { trigger_graph_layout_twopi(); });

    // Subscribe to graph signals.
    //
    // ★NOT update_node_attr_signal. It fires for EVERY attribute change on EVERY node in the shared graph —
    // every RT pose write from robot_concept, every other agent's per-cycle attribute writes — and each
    // emission copies a std::vector<std::string> on the DDS reader thread and queues an event to this thread.
    // Under a churn burst (a peer restarting is enough) the queue drains slower than it fills, and the main
    // thread then does nothing but service slots: the timer-driven compute() is starved (ai2_log stops
    // growing), and — the symptom that costs the most — Ctrl-C dies, because generated/main.cpp routes SIGINT
    // through a QSocketNotifier serviced by this SAME event loop. The agent then can only be killed with -9,
    // which cannot be caught, so every node it owns LEAKS into the shared graph.
    // Measured 2026-08-07 on table_concept (identical subscription): main thread pegged at 100% of a core,
    // ai2_log.csv frozen, Ctrl-C inert, following a retina restart. residual_concept did the same on 08-06
    // under the same trigger. CLAUDE.md already states the rule this violated: if you don't need a signal,
    // don't connect it at all (bottle_concept connects none).
    // The two things the slot did are now POLLED once per cycle in poll_affordance_protocol() — a controller
    // claim does not need sub-cycle latency, so nothing is lost and the firehose is gone.
    connect(G.get(), &DSR::DSRGraph::update_node_signal,
            this, &SpecificWorker::modify_node_slot);
    connect(G.get(), &DSR::DSRGraph::del_node_signal,
            this, &SpecificWorker::del_node_slot);

    // Remove any "hood*" nodes left behind by a previous (crashed) run so this agent always starts
    // from a clean slate and never adopts a stale/drifted node (the instance tracker re-births them
    // data-driven from masks).
    remove_owned_hood_nodes();

    // Resolve room node
    const auto rooms = G->get_nodes_by_type("room");
    if (not rooms.empty())
        room_node_id_ = rooms.front().id();
    else
        qWarning() << "hood_concept: no room node found at startup";

    // Active-inference fit core. Owns the instance map; collaborates with the ingestor + scene graph.
    fitter_ = std::make_unique<rc::HoodFitter>(
        G, inner_eigen_.get(), cfg_, mask_ingestor_.get(), scene_graph_.get());
    existence_ = std::make_unique<rc::HoodExistence>(G, cfg_);
    existence_->set_p_fa_field(&p_fa_field_);   // consume the learnt field (see hood_existence.h)   // evidence-based removal (existence log-odds)

    // Part B: localization/chain covariance on the published RT edge (mirrors bottle_concept).
    gaussian_api_ = std::make_unique<DSR::InnerGaussianAPI>(G.get());
    fitter_->set_chain_cov_source(gaussian_api_.get(), "zed");
    // Object-anchor observation z_o for room_concept's landmark factor (expressed in the localizer base frame).
    fitter_->set_object_observation(cfg_.publish_object_obs, cfg_.object_obs_frame);

    // YOLO-independent LiDAR range channel: lidar3D media-plane consumer that stages each cycle's sweep in the
    // room frame for the fitter's range factor. Dormant (no DDS participant) unless HoodModel.LidarPrecision
    // > 0. Subscriber is brought up lazily on the compute/main thread once the lidar3D node + descriptor exist.
    lidar_ingestor_ = std::make_unique<rc::ConceptLidarIngestor>(G, inner_eigen_.get(),
        [this] { return rc::LidarGates{cfg_.lidar_precision, cfg_.free_space_precision, cfg_.lidar_bpearl_precision}; });

    // ZED RGB media-plane consumer for appearance-based FRONT (door) detection. Dormant (no DDS participant)
    // unless HoodConcept.FrontDetectEnabled. Subscriber comes up lazily on the compute/main thread once
    // the "zed" node + media descriptor exist (media-plane consumer pattern).
    rgb_ingestor_ = std::make_unique<rc::HoodRgbIngestor>(G, cfg_);

    // rc::EpistemicPlanner owns the BELIEF half of the NBV (Σ, ΔI, the adequacy gap); the SENSOR half — where
    // the detector can actually fire — comes from the camera model it is handed each cycle in step_epistemic().
    // ONE detector envelope, both directions: the viewpoint the planner asks for is the argmax of the same
    // model the removal channel uses to decide how much a missing mask is worth.
    const rc::detect::DetectorEnvelope det_env{cfg_.detect_min_fill, cfg_.detect_max_fill, cfg_.detect_soft};
    epistemic_planner_.set_detector_envelope(det_env);
    epistemic_planner_.set_vertical_extent(cfg_.vertical_extent_m);   // a hood HANGS — no floor anchor
    if (existence_) existence_->set_detector_envelope(det_env);

    // Stale affordance nodes are swept on entering Operating (presence hook) and on shutdown — see
    // remove_stale_affordance_nodes(), keyed on the parent object type (robust to node-name renames).

    // Standalone Qt dashboard + evidence-monitor windows (belief plots + per-instance snapshot).
    // Shadow-mode birth/death record (CONCEPT_AGENT_LIFECYCLE.md §4.2). Recording only — see
    // log_phantom_event(). Truncating: one file per run.
    phantom_log_.open("etc/hood_phantom_events.csv");

    build_dashboard();
}

// ─── Main compute loop ───────────────────────────────────────────────────────────────────────────

void SpecificWorker::refresh_room_geometry()
{
    if (not G or room_node_id_ == 0) return;
    const auto room = G->get_node(room_node_id_);
    if (not room.has_value()) return;
    const auto px = G->get_attrib_by_name<delimiting_polygon_x_att>(room.value());
    const auto py = G->get_attrib_by_name<delimiting_polygon_y_att>(room.value());
    if (not px.has_value() or not py.has_value()) return;
    const auto& xs = px->get(); const auto& ys = py->get();
    const std::size_t n = std::min(xs.size(), ys.size());
    if (n < 2) return;

    std::vector<Eigen::Vector2f> poly;
    poly.reserve(n);
    Eigen::Vector2f centroid = Eigen::Vector2f::Zero();
    for (std::size_t i = 0; i < n; ++i)
    {
        poly.emplace_back(xs[i], ys[i]);
        centroid += poly.back();
    }
    centroid /= static_cast<float>(n);
    fitter_->set_room_geometry(centroid, std::move(poly));
}

// SHADOW-MODE birth/death recorder — CONCEPT_AGENT_LIFECYCLE.md §4.2, theory in MODEL_HISTORY.md §4.
// RECORDS ONLY; it can never alter a birth or a removal. The attribution fields captured at death
// (p_detect / in-FoV / central) are what tell a genuine classifier phantom from one of OUR removal defects —
// a death with LOW p_detect means the log is recording a removal bug, and must not be learned from.
void SpecificWorker::log_phantom_event(std::string_view event, std::uint64_t id, std::string_view name,
                                       float x, float y, const rc::HoodInstance* inst, std::string_view note)
{
    if (not phantom_log_.is_open())
        return;
    rc::history::PhantomEvent e;
    e.event = event; e.id = id; e.name = name; e.x = x; e.y = y; e.note = note;
    // Observer pose → view bearing. The classifier failure is VIEWPOINT-dependent, so the eventual p_FA field
    // is keyed on (world cell × bearing); a place-only key would suppress a genuine object placed there.
    // Observer pose → view bearing. SHARED (common/phantom_log/observer_pose.h): the classifier failure is
    // VIEWPOINT-dependent, so the false-alarm field is keyed on (world cell × bearing), never place alone.
    rc::history::note_observer(e, inner_eigen_.get(), x, y);
    if (inst)   // death: carry the existence state that says whether this was a CONFIDENT kill
    {
        e.age_cycles    = inst->processed_cycles;
        e.p_detect      = inst->dbg_ex_pdetect;
        e.central_frac  = inst->dbg_ex_central;
        e.in_fov_frac   = (inst->dbg_ex_sil_ntotal > 0)   // ★a FRACTION, not a probed/not-probed flag:
                        ? static_cast<float>(inst->dbg_ex_sil_ndet) / inst->dbg_ex_sil_ntotal : 0.0f;
        e.exist_logodds = inst->existence.logodds();
        // Was it ever confirmed from a SECOND direction? If so this death is a departure, not a
        // hallucination, and phantom_feed refuses to teach the p_FA field from it.
        e.ever_verified = inst->ever_verified ? 1 : 0;
    }
    phantom_log_.write(e);
    // ★THE SAME RECORD, NOW ALSO LEARNING. A confident denial retro-labels the detections at this
    // (cell x bearing) as clutter; a death whose killing look could not resolve the object weighs 0 and
    // teaches nothing. See common/view_field/phantom_feed.h for why that is a weight and not a threshold.
    rc::field::note_phantom_event(p_fa_field_, "hood", e);
}

void SpecificWorker::compute()
{
    // ★ONE graph walk per cycle for the SHARED mutual-exclusion rule: who else claims room space.
    // Feeds BOTH the birth filter (a candidate on somebody else's object accrues no evidence) and
    // the existence occupancy discount. Main thread — collect_graph_obstacles uses ts==0 (CLAUDE.md).
    if (G) foreign_claims_ = rc::exclusion::foreign_claims(*G, inner_eigen_.get(), "hood");
    if (existence_) existence_->set_foreign_claims(&foreign_claims_);
    if (fitter_) fitter_->set_foreign_claims(&foreign_claims_);   // the FIT judges the same geometry

    if (not G or not rt_api_)
        return;

    // Refresh room node id if not yet found
    if (room_node_id_ == 0)
    {
        const auto rooms = G->get_nodes_by_type("room");
        if (rooms.empty()) return;
        room_node_id_ = rooms.front().id();
    }

    // Push the room polygon + interior centroid into the fitter (wall-flush factor). Cheap; room_concept refines it.
    refresh_room_geometry();

    // Controller-owned affordance flags (claim / completion / epistemic_pending). Polled here rather than
    // pushed by update_node_attr_signal — see the connect block in initialize() for why that subscription
    // could starve this very loop.
    poll_affordance_protocol();

    const bool fresh_masks = mask_ingestor_->refresh();

    // EvidenceMonitor per-cycle counters (cumulative *_cum fields persist across cycles). The producers below
    // (tracker / merge / removal) add to these; the snapshot is pushed at the end of the cycle.
    ev_g_.births = ev_g_.merges = ev_g_.removals = ev_g_.births_refused = 0;
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

    // Stage this cycle's newest ZED RGB frame for the fitter's appearance FRONT (door) detection. clear-then-set
    // so a stalled RGB stream never re-integrates a stale appearance; pump() (main-thread) decodes into a
    // deep-copied BGR frame and is dormant while FrontDetectEnabled is off.
    fitter_->clear_rgb_frame();
    if (rgb_ingestor_)
    {
        rgb_ingestor_->pump();
        if (rgb_ingestor_->fresh())
            fitter_->set_rgb_frame(rgb_ingestor_->frame(), rgb_ingestor_->stamp_ms());
    }

    // Robot/camera ego-motion (transform chain) → the "be-still-to-update" confirm-only gate. Once per cycle,
    // BEFORE the instance loop so every run_inference this cycle reads the current robot speed (matches chair).
    fitter_->update_ego_motion();

    // Generic DSR type "object"; the class filter is the "hood_" name prefix + object_subtype attr.
    const auto object_nodes = G->get_nodes_by_type("object");
    for (const auto& node : object_nodes)
        if (node.name().starts_with("hood"))
            process_hood_node(node);

    // Ricoh 360 = peripheral attention: associate ricoh detections to hoods BY DIRECTION (after the ZED fits,
    // so hood positions are current); an unassigned bearing becomes a "seek a ZED view here" attention target.
    process_ricoh_bearings();

    // The detector's truth table. BEFORE removal on purpose: an instance killed this cycle must still
    // leave the row that records the look which killed it — that row is the whole point of the file.
    log_detect_probe();

    // Evidence-based removal: each existence channel integrates on its OWN sensor cadence (silhouette/mask on a
    // fresh mask frame, LiDAR carve on a fresh sweep) — a camera-only cycle still accrues absence, a LiDAR-only
    // cycle still carves free space. After the fits so footprints are current. OFF unless enabled.
    if (cfg_.existence_removal_enabled)
        existence_->update_and_remove(*fitter_, lidar_ingestor_.get(), fresh_masks, fresh_sweep, ev_g_,
            [this](std::uint64_t id, const rc::HoodInstance& inst)
            {   // shadow-mode death record (§4.2) — p_detect here says whether this was a CONFIDENT
                // disconfirmation (a real phantom) or a weak one (more likely our own removal bug)
                log_phantom_event("DEATH", id, inst.node_name,
                                  inst.model.state().cx, inst.model.state().cy, &inst,
                                  std::format("L {:.2f}", inst.existence.logodds()));
            });

    // "Is this really a fridge?" soft SINGLETON + plausibility→existence decay → retire mis-detections. The
    // worker sees ALL instances, so mutual inhibition + the removal decision live here (not in the fitter). It
    // reuses each instance's existence log-odds + a dedicated debounce, so it works whether or not the
    // sensor-existence channel above is enabled. Continuous/bounded; a genuine second fridge survives.
    apply_fridge_filter();

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
    prune_dead_series();   // drop timeseries lines for hoods that were removed from the graph
    log_birth_surprise();  // EXPERIMENTAL read-only: residual grid → birth-surprise regions (cfg-gated, off by default)

    fps_counter_.print("[hood_concept Compute]");   // std::cout heartbeat: Period/Fps/cpu%/mem (every ~1s)
}

// EXPERIMENTAL, read-only. Reinterpret residual_concept's `grid` node (under room) as an unexplained-occupancy
// SURPRISE field and cluster its confidently-unexplained cells into candidate regions (birth_surprise_probe.h).
// Log, per cycle, the regions NOT already covered by a believed hood alongside the tracker's actual birth count,
// so we can see whether surprise flags a real new hood cleanly (and stays quiet on phantoms) BEFORE letting it
// drive the lifecycle. Never writes the graph; never births. See [[hood-birth-surprise-probe]].
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

    // Believed hood footprints (room frame) — a region under one is already explained, NOT a birth.
    std::vector<rc::FootprintBox> hoods;
    for (const auto& [id, inst] : fitter_->instances())
        if (inst.ai2_initialized)
        { const auto& s = inst.ai2_belief.state(); hoods.push_back({s.cx, s.cy, s.w, s.h, s.yaw}); }

    const auto cands = rc::BirthSurpriseProbe::scan(gf, hoods);
    const long cyc = ++birth_surprise_cycle_;   // advances only on cycles where the grid field was actually read
    int n_birth = 0;                       // uncovered high-surprise regions = birth candidates
    for (const auto& c : cands) if (not c.covered_by_concept) ++n_birth;

    // CSV: one row per region per cycle (covered flag distinguishes birth candidates from explained mass; the
    // latter should be ~0 if residual_concept's concept-subtraction is working — a free sanity check).
    if (not birth_surprise_csv_.is_open())
    {
        rc::diag::open_rotating(birth_surprise_csv_, "etc/birth_surprise.csv");
        birth_surprise_csv_.imbue(std::locale::classic());   // ★Qt imbues the global locale, which inserts THOUSANDS SEPARATORS
                                            // into integers (pkt_ts 1785763853131 -> "1,785,763,853,131"), splitting
                                            // one CSV field into five and making the whole log unparseable by column.
                                            // Pin "C" so the log is machine-readable regardless of the UI locale.
        if (birth_surprise_csv_.is_open())
            birth_surprise_csv_ << "cycle,region,cx,cy,cells,mass,ext_x,ext_y,mean_p,mean_var,covered,"
                                << "n_hoods,tracker_births,instances\n";
    }
    if (birth_surprise_csv_.is_open())
    {
        int r = 0;
        for (const auto& c : cands)
            birth_surprise_csv_ << cyc << ',' << r++ << ',' << c.cx << ',' << c.cy << ',' << c.cells << ','
                                << c.mass << ',' << c.ext_x << ',' << c.ext_y << ',' << c.mean_p << ',' << c.mean_var
                                << ',' << (c.covered_by_concept ? 1 : 0) << ',' << hoods.size() << ','
                                << ev_g_.births << ',' << fitter_->instances().size() << '\n';
        birth_surprise_csv_.flush();
    }

    // ── FUSION readout: residual surprise MASS under each YOLO "hood" detection (birth_fusion.csv). The measured
    //    quantity: does a real detection land on high unexplained-occupancy (→ corroborated → birth fast/confident)
    //    while a flicker/phantom detection lands on ~0? covered = the detection sits inside an already-believed
    //    hood footprint (associate, not birth). This is the signal that would let residual GATE/accelerate birth.
    if (not birth_fusion_csv_.is_open())
    {
        rc::diag::open_rotating(birth_fusion_csv_, "etc/birth_fusion.csv");
        birth_fusion_csv_.imbue(std::locale::classic());   // ★Qt imbues the global locale, which inserts THOUSANDS SEPARATORS
                                            // into integers (pkt_ts 1785763853131 -> "1,785,763,853,131"), splitting
                                            // one CSV field into five and making the whole log unparseable by column.
                                            // Pin "C" so the log is machine-readable regardless of the UI locale.
        if (birth_fusion_csv_.is_open())
            birth_fusion_csv_ << "cycle,det,det_x,det_y,mass_r05,mass_r03,near_dist,near_mass,covered,"
                              << "n_hoods,tracker_births,instances\n";
    }
    if (birth_fusion_csv_.is_open())
    {
        // DIAGNOSTIC block only — the 0.50/0.30 m probe radii and the 0.30 m footprint margin below feed the
        // birth_fusion.csv columns for offline analysis of residual-vs-birth fusion. None of these values touch
        // the live belief or any birth decision, so they are hardcoded (a config key would imply they matter at
        // runtime). If residual-gated birth is ever wired in, promote them then.
        int di = 0;
        for (const auto& d : last_hood_dets_xy_)
        {
            const float m05 = rc::BirthSurpriseProbe::residual_mass_near(gf, d.x(), d.y(), 0.50f);
            const float m03 = rc::BirthSurpriseProbe::residual_mass_near(gf, d.x(), d.y(), 0.30f);
            float nd = 1e9f, nm = 0.f;                      // nearest region to this detection
            for (const auto& c : cands)
            { const float dd = std::hypot(c.cx - d.x(), c.cy - d.y()); if (dd < nd) { nd = dd; nm = c.mass; } }
            bool covered = false;
            for (const auto& t : hoods)
            { const float cc = std::cos(t.yaw), ss = std::sin(t.yaw), dx = d.x() - t.cx, dy = d.y() - t.cy;
              if (std::abs(cc*dx + ss*dy) <= 0.5f*t.w + 0.30f and std::abs(-ss*dx + cc*dy) <= 0.5f*t.h + 0.30f)
                  { covered = true; break; } }
            birth_fusion_csv_ << cyc << ',' << di++ << ',' << d.x() << ',' << d.y() << ',' << m05 << ',' << m03 << ','
                              << (nd > 1e8f ? -1.f : nd) << ',' << nm << ',' << (covered ? 1 : 0) << ','
                              << hoods.size() << ',' << ev_g_.births << ',' << fitter_->instances().size() << '\n';
            if (ev_g_.births > 0 and not covered)          // a NEW hood just born — print its corroboration
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
            std::print("[birth-surprise] uncovered={} hoods={} tracker_births={} | top: ({:.2f},{:.2f}) "
                       "mass={:.1f} cells={} ext={:.2f}x{:.2f} mean_p={:.2f} var={:.3f}\n",
                       n_birth, hoods.size(), ev_g_.births, top->cx, top->cy, top->mass, top->cells,
                       top->ext_x, top->ext_y, top->mean_p, top->mean_var);
    }
}


// ─── Per-node processing + publish ───────────────────────────────────────────────────────────────

// Process one "hood" DSR node this cycle: ensure its instance exists, then fuse each assigned ZED slice
// (one belief update per slice, gated to a fresh mask frame) or age the belief when no mask arrived, and // hand the result to publish_hood_cycle. run_instance_tracker has already associated this cycle's slices.
void SpecificWorker::process_hood_node(const DSR::Node& node)
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
        // (Per-instance time-series are registered idempotently by publish_hood_diagnostics each cycle.)
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
    HoodObservation last_obs;
    bool updated = false;

    // Multi-slice fusion: one belief update per assigned ZED slice of this hood. Sequential Bayesian updates
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

    publish_hood_cycle(inst, node, last_obs, free_energy);
}


// Publish one fitted hood: persist belief→DSR, assess convergence, then push diagnostics + intentions.
// Each step short-circuits if its node lookup fails (the node may have been removed mid-cycle).
void SpecificWorker::publish_hood_cycle(rc::HoodInstance& inst,
                                         const DSR::Node& node,
                                         const HoodObservation& observation,
                                         float free_energy)
{
    const auto node_id = node.id();
    if (not scene_graph_->persist_hood_belief(inst, node_id, room_node_id_, free_energy))
        return;
    if (not assess_hood_state(inst, node_id, free_energy))
        return;
    publish_hood_diagnostics(inst, observation, free_energy);
    publish_hood_intentions(inst, node_id, observation, free_energy);
}

// Run the convergence/stability step for one hood; re-resolves the node by id (false if it is gone).
bool SpecificWorker::assess_hood_state(rc::HoodInstance& inst, uint64_t node_id, float free_energy)
{
    auto node_opt = G->get_node(node_id);
    if (not node_opt.has_value())
        return false;

    step_convergence(inst, node_opt.value(), free_energy);
    return true;
}

// Feed this instance's time-series to the standalone dashboard: FE + baseline, FE-surprise, U(Σ) covariance,
// residual count, inferred dims (w,h), and size posterior σ. Series are added idempotently every cycle.
void SpecificWorker::publish_hood_diagnostics(const rc::HoodInstance& inst,
                                               const HoodObservation& observation,
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

void SpecificWorker::publish_hood_intentions(rc::HoodInstance& inst,
                                              uint64_t node_id,
                                              const HoodObservation& observation,
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
    cfg_ = rc::load_hood_config(cfg);
    // Declarative-priors experiment: report whether common/concept_manifest/hood.concept.toml can
    // reproduce the priors this config just produced. Read-only — the manifest is not authoritative yet.
    rc::verify_hood_manifest(cfg_, "../common/concept_manifest/hood.concept.toml");
}



// The ZED as the NBV's forward model: both FoVs from the REAL intrinsics, and the optical centre's height above
// the FLOOR datum — the same datum the hood box's z ∈ [0, H] is expressed in, which is what makes the
// vertical framing term mean anything.
//
// ★The VERTICAL channel is the whole point. `roi_fill` is max(Δcol/W, Δrow/H) and for a 1.7 m hood seen
// from a camera at ~0.95 m the row term dominates by ~4×: the box overflows the frame vertically at 1.2 m,
// where it still fills only a quarter of it horizontally. Supplying hfov alone (what this did) hides the axis
// that actually truncates the mask, and the planner then proposes a pose where no mask can form at all.
// A missing zed node / camera API / room→zed transform leaves the corresponding field at its default, which
// disables the vertical channel rather than inventing a mount height.
rc::nbv::Sensor SpecificWorker::zed_sensor_model() const
{
    rc::nbv::Sensor s;
    const auto zed = G->get_node("zed");
    if (not zed.has_value())
        return s;
    if (auto cam = G->get_camera_api(zed.value()); cam)
    {
        const float fx = cam->get_focal_x(), fy = cam->get_focal_y();
        const float W  = static_cast<float>(cam->get_width()), H = static_cast<float>(cam->get_height());
        if (fx > 0.0f and W > 0.0f) s.hfov_rad = 2.0f * std::atan(0.5f * W / fx);
        if (fy > 0.0f and H > 0.0f) s.vfov_rad = 2.0f * std::atan(0.5f * H / fy);
    }
    // Mount height in ROOM frame (ts=0, main thread — the InnerEigen cache rule in CLAUDE.md). Not the proto's
    // body-relative 0.945: the room's floor datum is offset from the body origin, and the box heights are in
    // room frame, so mixing the two would bias the binding axis by that offset.
    if (inner_eigen_)
        if (const auto rtz = inner_eigen_->get_transformation_matrix("room", "zed", 0); rtz.has_value())
            s.height_m = static_cast<float>(rtz.value()(2, 3));
    return s;
}

// ─── Per-cycle steps ─────────────────────────────────────────────────────────────────────────────

// Convergence latch for one hood: measure how far the accepted state moved this cycle and, once it holds
// still for K_stable cycles, publish model_stable + the model_uncertainty_att readout (posterior std sum).
void SpecificWorker::step_convergence(rc::HoodInstance& inst,
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
           std::abs(s.z_top - p.z_top) + std::abs(s.yaw - p.yaw))
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
        std::print("hood_concept: node '{}' STABLE (F={:.4f})\n", inst.node_name, free_energy);
}

// Publish/refresh the epistemic next-best-view affordance for one hood from its belief Σ (D-optimal NBV),
// with a post-completion cooldown that suppresses the published gain so a just-finished hood isn't re-claimed.
void SpecificWorker::step_epistemic(rc::HoodInstance& inst, DSR::Node& node)
{
    // The cycle around the planner call is SHARED (common/epistemic_step): the cooldown, the
    // controller-completion hold, the gain suppression and verification floor, the affordance refresh, and —
    // the reason it is shared rather than merely deduplicated — the single bail path that ALWAYS calls
    // hold_offered(). Only the planner call and what this agent records are hood's.
    rc::epistemic::StepHooks<rc::EpistemicProposal> hooks;

    hooks.compute = [&]() -> std::optional<rc::EpistemicProposal>
    {
        // Σ-based D-optimal NBV from the belief. Skip until the belief has seen its first frame (else Σ is
        // the broad prior and the proposal is moot).
        if (not inst.ai2_initialized)
            return std::nullopt;
        rc::EpistemicProposal prop =
            epistemic_planner_.compute(inst.ai2_belief, cfg_.ai2_range_noise_lat_per_m, cfg_.ai2_sigma_base_m,
                                       zed_sensor_model(),
                                       rc::nbv::collect_graph_obstacles(*G, inner_eigen_.get(), inst.node_id),
                                       // The reachable region. refresh_room_geometry() already loads this
                                       // polygon for the fit's wall factor; the NBV never saw it, so every
                                       // viewpoint — including the ones behind the wall the hood is pushed
                                       // against — read as reachable. Empty until room_concept publishes, and
                                       // is_reachable then imposes no constraint (the pre-existing behaviour).
                                       fitter_->room_polygon());
        // Degenerate fit, or no sensor model yet — retry next cycle.
        if (not prop.valid or not prop.is_finite())
            return std::nullopt;
        return prop;
    };

    hooks.record = [&](const rc::EpistemicProposal& published, float raw_gain)
    {
        // Mirror the proposal onto the instance so ai2_log records what was actually proposed. Without this
        // the only record was stdout, and the affordance node carries the FROZEN pose during an executing
        // claim — so "the target is too close" was unanswerable after the fact from either source.
        // ⚠hood logs the RAW gain (pre-suppression) where bottle/cabinet/chair log the published one — see
        // the divergence note in common/epistemic_step. Behaviour preserved here; resolve it deliberately.
        inst.dbg_nbv_standoff = published.chosen_standoff_m;
        inst.dbg_nbv_gain_raw = raw_gain;                     // what the belief asked for
        inst.dbg_nbv_gain_pub = published.epistemic_gain;     // what the controller will see
        inst.dbg_nbv_target_x = published.epistemic_target_x_m;
        inst.dbg_nbv_target_y = published.epistemic_target_y_m;
        inst.dbg_nbv_pdetect  = published.chosen_p_detect;
        inst.dbg_nbv_vfov     = published.sensor_vfov_rad;
        // Write attributes to the hood node (read by legacy consumers)
        scene_graph_->write_epistemic_proposal(node, published);
    };

    hooks.on_affordance_created = [this] { trigger_graph_layout_twopi(); };

    rc::epistemic::step(inst, *G,
                        {.cooldown_cycles   = cfg_.epistemic_cooldown_cycles,
                         // Verification pull: absence never deletes a hood the robot hasn't properly looked
                         // at — it sends the robot to look.
                         .verify_gain_floor = inst.wants_verification ? cfg_.existence_verify_gain : 0.0f},
                        hooks);
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
    // Instance creation is owned by the compute() loop (process_hood_node → ensure_instance), which runs
    // AFTER note_birth() each cycle, so the birth seed is in place. The loop also covers externally-created
    // hood nodes (every cycle), so nothing is lost by not creating here.
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
    // ★A graph slot must be safe at ANY time, including before the fit core exists. del_node_signal is
    // connected in initialize() BEFORE `fitter_` is constructed, and the startup stale-sweep
    // remove_owned_hood_nodes() then calls G->delete_node() on the MAIN thread — same-thread emit ⇒
    // Auto resolves to DIRECT ⇒ this slot runs synchronously with `fitter_` still null. Dormant unless a
    // previous run actually left "hood*" nodes behind, which is why it had not been seen; it SIGSEGV'd
    // table_concept exactly this way on 2026-08-06. cabinet_concept guards its del_node_slot for the same
    // reason (and missed its attrs slot).
    if (not fitter_)
        return;
    // Notify affordance in case its own DSR node was deleted externally
    for (auto& [hood_id, inst] : fitter_->instances())
        if (inst.affordance.node_id() == id)
            inst.affordance.on_node_deleted(id);

    if (fitter_->instances().count(id))
    {
        std::print("hood_concept: node {} removed from DSR, destroying instance\n", id);
        fitter_->instances().erase(id);
    }
}

// ─── Lifecycle stubs ─────────────────────────────────────────────────────────────────────────────

void SpecificWorker::emergency()
{
    std::print("hood_concept: emergency()\n");
}

void SpecificWorker::restore()
{
    std::print("hood_concept: restore()\n");
}

int SpecificWorker::startup_check()
{
    std::print("hood_concept: startup_check()\n");
    return 0;
}




