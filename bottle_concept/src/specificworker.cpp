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

#include "../../common/config_report/config_read.h"   // rc::cfg::Reader (SHARED)
#include "../../common/room_resolve/room_resolve.h"   // rc::room::current_room (proto-aware, deterministic)

#include "../../common/diag_log/rotating_csv.h"   // keep the previous run instead of wiping it


#include "../../common/dashboard/belief_series.h"   // rc::dash::publish_belief_series (SHARED)

#include "../../common/peripheral_channel/peripheral_channel.h"   // THE shared ricoh path
#include "../../common/exclusion/exclusion.h"   // rc::exclusion — the SHARED no-two-objects rule
#include "../../common/exclusion/exclusion.h"   // rc::exclusion:: (SHARED)
#include "../../common/footprint/footprint.h"   // rc::geom:: (SHARED)
#include "../../common/track/merge_instances.h"   // rc::track::merge_overlapping — the SHARED merge sweep
#include "../../common/nbv/graph_obstacles.h"   // rc::nbv::collect_graph_obstacles — shared, DSR-side
#include <cstdlib>   // std::_Exit for the crash-free terminal shutdown
#include <thread>    // std::this_thread::sleep_for — let DDS flush before _Exit
#include <chrono>


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
#include <QDateTime>   // wall-clock ms for the primary-input stream gate (presence_protocol_.operating_since_ms(), stall grace)

#include "../../common/instance_tracker/birth_evidence.h"   // rc::birth:: the shared CREATE policy

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
    // Registered rather than read raw, so the period appears in the startup table like every
    // other key. req() keeps the throw-if-missing behaviour: a period is not something to
    // default silently.
    int period = 0;
    rc::cfg::Reader(configLoader, "bottle_concept").req("Period.Compute", period,
            "GRAFCET step period (ms) for every state of this agent's state machine");

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
    save_strip_geometry();       // …nor is the compact belief strip

    // Sever graph callbacks BEFORE any teardown. On Ctrl+C a del_node delta can be
    // delivered from a DSR/DDS internal thread and invoke del_node_slot (instances_.erase)
    // on this already-destructing object — the exit segfault. Dropping inner_eigen_ here
    // (while G is still fully alive) also unsubscribes its internal graph signals cleanly.
    if (G)
        disconnect(G.get(), nullptr, this, nullptr);
    // Drop the LiDAR media subscriber BEFORE tearing down the graph/inner_eigen it reads (it holds a raw
    // inner_eigen_ pointer + a DDS reader) — mirrors room_concept's teardown ordering.
    lidar_ingestor_.reset();
    rgb_ingestor_.reset();
    depth_ingestor_.reset();
    inner_eigen_.reset();

    cleanup_owned_nodes();
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

    // Ignore payload attributes in local graph updates: bottle reads pixels off the MEDIA PLANE, never out of
    // the graph, so every cam_rgb/cam_depth/laser_* blob a peer publishes was being deserialised and stored
    // here for nothing. ★Six of the seven agents did this and bottle did not — see the CRDT dot-cloud growth
    // note in CLAUDE.md for what unused attribute traffic costs. Purely a filter on what we ACCEPT; it
    // changes nothing this agent reads.
    G->set_ignored_attributes<cam_rgb_att, cam_depth_att, laser_X_att, laser_Y_att, laser_Z_att>();

    // Agent-presence protocol wiring.
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

    if (const auto room = rc::room::current_room(*G); room.has_value())
        room_node_id_ = *room;
    else
        qWarning() << "bottle_concept: no room node found at startup";

    // Validation harness (no-op unless an Eval.*/Scene.* flag is set). Table-top lookup injected as a
    // callback so the evaluator stays decoupled from the scene-graph layer.
    evaluator_ = std::make_unique<rc::BottleEvaluator>(
        cfg_, G, webots2robocomp_proxy, inner_eigen_.get(),
        [this](float bx, float by) { return scene_graph_->find_table_top(bx, by); });

    // Hidden-face next-best-view planner (epistemic affordance).
    epistemic_planner_ = rc::EpistemicPlanner(cfg_.epistemic_obs_distance, cfg_.epistemic_view_info);
    // ONE detector envelope: the far-side viewpoint is the argmax of the same model absence is weighted by,
    // and the published gain is multiplied by P(detect) there — so a hidden-face look the detector could not
    // fire from stops bidding for the drive AROUND the bottle.
    epistemic_planner_.set_detector_envelope(
        rc::detect::DetectorEnvelope{cfg_.detect_min_fill, cfg_.detect_max_fill, cfg_.detect_soft});
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
    // ZED RGB + DEPTH media-plane consumers for the classifier-free contour existence channel. Dormant (no DDS
    // participant) unless BottleConcept.ContourCheckEnabled. Subscribers come up lazily on the compute/main
    // thread once the "zed" node + media descriptor exist (media-plane consumer pattern).
    camera_planes_wanted_ = cfg_.contour_check_enabled;
    rgb_ingestor_   = std::make_unique<rc::RgbIngestor>(G, &camera_planes_wanted_, "zed");
    depth_ingestor_ = std::make_unique<rc::DepthIngestor>(G, &camera_planes_wanted_, "zed");

    lidar_ingestor_ = std::make_unique<rc::ConceptLidarIngestor>(G, inner_eigen_.get(),
        // Bottle declares only the helios range gate: its belief has no free-space carve and no bpearl
        // factor, so those two planes stay dormant rather than staging points nothing would read. The
        // capability is in the shared ingestor the day BottleBelief grows either factor.
        [this] { return rc::LidarGates{cfg_.lidar_precision, 0.0f, 0.0f}; });
    existence_      = std::make_unique<rc::BottleExistence>(cfg_);

    // Live dashboard: the belief strip + the drill-down behind its "details" button. SHARED construction
    // (common/dashboard/concept_dashboard.h) — see specificworker_dashboard.cpp.
    build_dashboard();

    // ── WHAT THIS AGENT IS ACTUALLY RUNNING ────────────────────────────────────────────────────
    //
    // ★PUBLISHED AT THE END OF initialize(), NOT WHERE THE CONFIG IS PARSED. This agent's own keys
    // are settled much earlier, but the SHARED presence unit reads its sixteen [Presence.*]/[Owns.*]
    // keys when the coordinator is configured, further down this same function. Publishing before
    // that armed the unread sweep on a registry those sixteen had not reached yet, and named every
    // one of them "in the file, read by nothing" - confident false positives from the one check whose
    // whole value is that it does not cry wolf. The rule is general: publish when the LAST reader has
    // run, which is the end of startup, not the end of parsing.
    //
    // Prints only the DELTAS - values differing from the code default, plus every A/B arm even at its
    // default - and writes the full table to etc/config_effective.csv, this run's own record of which
    // arm it was. Config is read once at startup, so a file's mtime never says which run used it.
    //
    // declare_complete() CLAIMS that every config key this agent reads goes through a Reader, and it
    // ARMS the unread sweep; check_registry_complete.sh bottle_concept is the grep that keeps the claim
    // honest. Re-run it whenever a config read is added.
    rc::cfg::exempt_generated_prefixes();
    rc::cfg::registry().declare_complete("bottle_concept");
    rc::cfg::Reader(configLoader, "bottle_concept").publish("etc/config_effective.csv");
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
    // Observer pose → view bearing. SHARED (common/phantom_log/observer_pose.h): the classifier failure is
    // VIEWPOINT-dependent, so the false-alarm field is keyed on (world cell × bearing), never place alone.
    rc::history::note_observer(e, *G, inner_eigen_.get(), x, y);
    if (inst)
    {
        // ★These are REAL now. Before the existence channel every bottle death was UNATTRIBUTABLE by
        // construction (retired on a miss counter, with no p_detect to say whether the miss meant anything),
        // so the phantom analysis had to discard them. A death now records the geometry that justified it.
        e.age_cycles    = inst->processed_cycles;
        e.p_detect      = inst->dbg_ex_p_detect;
        e.central_frac  = inst->dbg_ex_vis;
        e.in_fov_frac   = inst->dbg_ex_vis;
        e.exist_logodds = inst->exist_logodds;
    }
    phantom_log_.write(e);
}

// Load the room's delimiting polygon (a trusted NOMINAL model authored by room_concept, never fitted).
// Mirrors refrigerator_concept::refresh_room_geometry. Leaves the polygon empty until room_concept
// publishes, which is exactly the "impose no constraint" case rc::nbv::is_reachable expects.
void SpecificWorker::refresh_room_polygon()
{
    if (not G or room_node_id_ == 0) return;
    const auto room = G->get_node(room_node_id_);
    if (not room.has_value()) return;
    const auto px = G->get_attrib_by_name<delimiting_polygon_x_att>(room.value());
    const auto py = G->get_attrib_by_name<delimiting_polygon_y_att>(room.value());
    if (not px.has_value() or not py.has_value()) return;
    const auto& xs = px->get(); const auto& ys = py->get();
    const std::size_t n = std::min(xs.size(), ys.size());
    if (n < 3) return;                       // degenerate ⇒ leave it empty rather than half-armed
    std::vector<Eigen::Vector2f> poly; poly.reserve(n);
    for (std::size_t i = 0; i < n; ++i) poly.emplace_back(xs[i], ys[i]);
    room_polygon_ = std::move(poly);
}

void SpecificWorker::compute()
{
    // ★ONE graph walk per cycle for the SHARED mutual-exclusion rule: who else claims room space.
    // Feeds BOTH the birth filter (a candidate on somebody else's object accrues no evidence) and
    // the existence occupancy discount. Main thread — collect_graph_obstacles uses ts==0 (CLAUDE.md).
    if (G) foreign_claims_ = rc::exclusion::foreign_claims(*G, inner_eigen_.get(), "bottle");
    if (existence_) existence_->set_foreign_claims(&foreign_claims_);
    if (fitter_) fitter_->set_foreign_claims(&foreign_claims_);   // the FIT judges the same geometry

    refresh_room_polygon();   // NBV reachability needs the room; empty until room_concept publishes
    if (not G or not rt_api_)
        return;

    if (room_node_id_ == 0)
    {
        const auto room = rc::room::current_room(*G);
        if (not room.has_value()) return;
        room_node_id_ = *room;
    }

    // Controller-owned affordance flags (claim / completion / epistemic_pending). Polled rather than pushed
    // by update_node_attr_signal, which fires for every attribute change on every node in the shared graph.
    poll_affordance_protocol();

    // One-shot: place the bottle on its arm-side spot BEFORE any fit, then let the scene settle so
    // the retina captures it there before the tracker births the node — a node created from a
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

    const bool fresh_masks = mask_ingestor_->refresh();

    // Stage this cycle's LiDAR sweep (room frame) for the fitter's range factor. clear-then-set so the factor
    // only contributes on cycles with a genuinely fresh sweep (never re-uses a stale one). No-op when off.
    fitter_->clear_lidar_sweep();
    bool fresh_sweep = false;
    if (lidar_ingestor_ and lidar_ingestor_->pump())
    {
        fitter_->set_lidar_sweep(lidar_ingestor_->sweep_room(), lidar_ingestor_->origin_room());
        fresh_sweep = true;
    }

    // The camera planes for the contour channel. pump() deep-copies the frame (thread-boundary-safe) and
    // decodes on THIS thread; the existence step only borrows them for the call.
    if (rgb_ingestor_)   rgb_ingestor_->pump();
    if (depth_ingestor_) depth_ingestor_->pump();

    run_instance_tracker();   // data-driven birth/associate/death (the only instance-lifecycle path)

    // Robot/camera ego-motion (room frame) for this cycle — the "be-still-to-update" signal. Computed ONCE on
    // the main thread (ts=0 pose diff) before the per-instance loop; run_inference's confirm_only gate reads it.
    fitter_->update_ego_motion();

    // Bottle instances are DSR `object` nodes named "bottle_*" (migrated from type "cylinder").
    for (const auto& node : G->get_nodes_by_type("object"))
        if (node.name().starts_with("bottle"))
            process_bottle_node(node);

    // EXISTENCE: integrate both evidence channels and retire the bottles whose volume is demonstrably empty.
    // Runs AFTER the per-instance fits so it projects this cycle's geometry, and it is the ONLY removal path
    // for absence (Tracker.DeathFrames is disabled while it is on) — one authority, one decision, on L.
    if (existence_)
    {
        rc::BottleExistence::Inputs in;
        in.G            = G.get();
        in.inner_eigen  = inner_eigen_.get();
        in.fresh_masks  = fresh_masks;
        in.masks_stamp_ms = mask_ingestor_->packet().timestamp_ms;
        in.sweep        = (fresh_sweep and lidar_ingestor_) ? &lidar_ingestor_->sweep_room() : nullptr;
        in.origin       = lidar_ingestor_ ? lidar_ingestor_->origin_room() : Eigen::Vector3f::Zero();
        in.room_polygon = &room_polygon_;
        // Camera clock for the contour channel: the contour is scored against THOSE pixels and pinned to
        // their stamp, so re-scoring a frame already scored would charge one observation twice.
        in.frames_fresh = (rgb_ingestor_ and rgb_ingestor_->fresh()) or (depth_ingestor_ and depth_ingestor_->fresh());
        in.rgb_frame    = rgb_ingestor_   ? &rgb_ingestor_->frame()   : nullptr;
        in.depth_frame  = depth_ingestor_ ? &depth_ingestor_->frame() : nullptr;
        in.frame_stamp_ms = (rgb_ingestor_ and rgb_ingestor_->stamp_ms() > 0)
                          ? rgb_ingestor_->stamp_ms()
                          : (depth_ingestor_ ? depth_ingestor_->stamp_ms() : 0);
        existence_->update_and_remove(*fitter_, in,
            [this](std::uint64_t id, rc::BottleInstance& inst)
            {
                // Same teardown ORDER as a tracker DEATH: the affordance first (while the instance and its id
                // still exist), then the C++ instance, then the DSR node — otherwise aff_<bottle> is orphaned.
                log_phantom_event("DEATH", id, inst.node_name,
                                  inst.model.state().cx, inst.model.state().cy, &inst, "existence");
                inst.affordance.remove();
                fitter_->forget_node(id);
                G->delete_node(id);
                ++ev_g_.removals;
            });
    }

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


// Collapse instances whose circle footprints overlap (same physical bottle fitted twice): keep the one
// with more integrated fresh evidence, retire the other (affordance + node). Runs before tracking so a
// duplicate is gone before it is fed a mask. Mirrors table_concept::merge_overlapping_instances.
// Retire one instance: drop its affordance node, forget it in the fitter, delete its graph node. The single
// teardown path shared by every lifecycle exit (merge / death), keeping the affordance+fitter+graph invariant.
// ★Named to match the other four agents, which already had it — this was the same three lines written inline.
void SpecificWorker::retire_instance(std::uint64_t id)
{
    if (auto it = fitter_->instances().find(id); it != fitter_->instances().end())
        it->second.affordance.remove();
    fitter_->forget_node(id);
    G->delete_node(id);
}


void SpecificWorker::merge_overlapping_instances()
{
    if (cfg_.tracker_merge_overlap <= 0.0f)
        return;

    // The sweep is SHARED (common/track); only the two per-object questions stay here.
    // ★The counter comes from the sweep now, and bottle is where it mattered most: this agent RESET
    // ev_g_.merges every cycle and never incremented it, so the dashboard's `merges=%d/%ld` read 0/0 for
    // the life of the agent while MERGE lines scrolled past on stdout. chair and door had the same hole.
    rc::track::merge_overlapping(
        fitter_->instances(), ev_g_,
        [&](const auto& a, const auto& b) -> std::optional<float>
        {
            const auto& sa = a.model.state();
            const auto& sb = b.model.state();
            // Two bottles cannot share physical space. SHARED exact two-circle lens: a cylinder's
            // footprint is a CIRCLE, and its bounding square would overestimate the area by 4/pi and
            // make the answer depend on a yaw a cylinder does not have.
            const float ratio = rc::geom::overlap_ratio(rc::geom::Circle{sa.cx, sa.cy, sa.radius},
                                                        rc::geom::Circle{sb.cx, sb.cy, sb.radius});
            return ratio >= cfg_.tracker_merge_overlap ? std::optional{ratio} : std::nullopt;
        },
        [](const auto& in) { return in.matched_frames; },      // keep the more-observed instance
        [&](std::uint64_t keep, std::uint64_t drop, auto&, const auto&, float ratio)
        {
            std::print("bottle_concept: [tracker] MERGE id={} into id={} (circle overlap {:.2f})\n",
                       drop, keep, ratio);
            retire_instance(drop);
        });
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
    // ★Invariant 5: removal is a decision on L, never a miss counter. With the existence channel on, the
    // tracker's death counter is DISABLED so there is exactly one removal authority — the same thing
    // table_concept and refrigerator_concept do (`death_frames = INT_MAX`). Turning the channel off in
    // config restores the counter exactly, which is what makes the two A/B-able.
    tp.death_frames     = cfg_.existence_enabled ? std::numeric_limits<int>::max()
                                                 : cfg_.tracker_death_frames;
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

    // ★BIRTH EVIDENCE — the shared CREATE policy (common/instance_tracker/birth_evidence.h). This agent fed
    // the tracker `birth_evidence = 1.0` on EVERY compute cycle, so `birth_frames` counted cycles rather than
    // observations: at ~10 Hz compute against a ~9.5 Hz mask stream one mask frame was counted several times,
    // and "N frames" became well under a second of a single unchanging view — which is how a YOLO false
    // positive on a wall panel becomes furniture. Three rules, none of them a threshold: an OBSERVATION not a
    // cycle; birth admitted by the UPDATE rule (frame_admissible — a frame the fit would refuse may not create
    // an object); and an admissible observation still worth only its reliability (confidence x range).
    const bool birth_new_obs = pkt.valid and static_cast<long>(pkt.frame_id) > last_birth_mask_frame_;
    if (birth_new_obs)
        last_birth_mask_frame_ = static_cast<long>(pkt.frame_id);
    const rc::birth::Detectability birth_detect{0.50f, 2.5f, 2.0f};

    if (pkt.valid)
        for (int i = 0; i < static_cast<int>(pkt.slices.size()); ++i)
            // ★★ONLY THE FRONT RGB-D CAMERA MAY CREATE OR UPDATE AN OBJECT. `has_depth` is NOT that
            // question: once the producer began depth-filling ricoh masks from reprojected LiDAR it
            // publishes them as full 3D slices with has_depth = 1, so a 360° detection from BEHIND the
            // robot passed every guard written as `if (has_depth)`. Reported live on bottle_concept —
            // moving and cloning with the robot facing away, 3 m off. mask_source says which camera,
            // unambiguously, and the retina has been publishing it all along. A ricoh slice may
            // still CONFIRM a live instance (common/peripheral_channel) or raise a proto-object to go and look
            // at; it may not move one. See MaskIngestor::MaskSlice::may_fit_geometry.
            if (pkt.slices[i].label == "bottle" and pkt.slices[i].may_fit_geometry())
            {
                const auto& sl = pkt.slices[i];
                rc::DetectionView dv;
                dv.xy = Eigen::Vector2f(sl.centroid.x(), sl.centroid.y());
                dv.slice_index = i;
                // One admissible, reliable observation — never a cycle. See birth_evidence.h.
                dv.birth_evidence = rc::birth::evidence({sl.confidence, sl.range}, birth_detect,
                                                        birth_new_obs, fitter_->frame_admissible(sl));

                // ★MUTUAL EXCLUSION — no two objects occupy the same space (SHARED, common/exclusion).
                // A continuous support multiplied into the birth evidence exactly like the others above, so a
                // candidate condensing onto ANOTHER CONCEPT's object never accrues enough to mature: it is not
                // vetoed, it is unsupported. Every agent already refused to fit two of its OWN instances to one
                // object; none ever asked what a different concept had claimed, which is how a refrigerator was
                // created on top of door_3 (16 cm apart, same width, same yaw) and then could not die.
                if (not foreign_claims_.empty())
                {
                    const rc::exclusion::Claim* who = nullptr;
                    // ★A BOTTLE STANDS ON SOMETHING, ALWAYS. Its footprint sits inside a table's or a
                    // worktop's for its whole life, so with no vertical band every bottle candidate in the
                    // room reads as 100% claimed and a bottle on a table could never be born at all. The band
                    // is the slice centroid ± half the prior height, which clears the surface it rests on.
                    const float cand_hz = 0.5f * cfg_.prior_height;
                    const float unclaimed = rc::exclusion::p_unclaimed(
                        {dv.xy.x(), dv.xy.y(), 2.0f * cfg_.prior_radius, 2.0f * cfg_.prior_radius, 0.0f}, foreign_claims_, &who,
                        sl.centroid.z() - cand_hz, sl.centroid.z() + cand_hz);
                    dv.birth_evidence *= unclaimed;
                    if (unclaimed < 0.99f)
                        std::print("[bottle] birth cand CLAIMED by '{}' ({:.0f}%): birth_ev x{:.2f}\n",
                                   who ? who->node : "?", 100.0f * (1.0f - unclaimed), unclaimed);
                }
                dets.push_back(dv);
            }

    // Part C (confirm): a ricoh no-depth "bottle" bearing that lines up (in azimuth from the robot) with a
    // live instance is evidence it is STILL THERE even when the zed missed it → HOLD its death-miss this
    // cycle (set expected_visible=false, exactly like being out of the zed frustum). No fit, no birth.
    if (cfg_.bearing_confirm_enabled and pkt.valid and not tracks.empty())
    {
        // Association is common/peripheral_channel now — one path for all seven agents. gather()
        // takes BOTH slice kinds (bearing-only and LiDAR-depth-filled): which arrives depends on where
        // the LiDAR swept, not on anything this agent decided, so it must not change behaviour. What
        // stays local is the ACTION on a confirm, which for bottle is holding the death-miss.
        Eigen::Vector2f robot_xy(0.f, 0.f);
        if (const auto p = inner_eigen_->transform(rc::room::current_room_frame(*G), Eigen::Vector3d::Zero(), "zed"); p.has_value())
            robot_xy = {static_cast<float>(p->x()), static_cast<float>(p->y())};

        const auto dets_p = rc::peripheral::gather(pkt, "bottle", robot_xy);
        if (not dets_p.empty())
        {
            std::vector<rc::peripheral::TrackRef> trefs;
            trefs.reserve(tracks.size());
            for (std::size_t ti = 0; ti < tracks.size(); ++ti)
                trefs.push_back({tracks[ti].id, tracks[ti].xy, cfg_.prior_radius});

            rc::peripheral::Params pp;
            pp.angular_margin_rad = cfg_.bearing_confirm_gate_rad;
            for (const auto& c : rc::peripheral::associate(trefs, dets_p, robot_xy, pp).confirms)
                for (std::size_t ti = 0; ti < tracks.size(); ++ti)
                    if (tracks[ti].id == c.track_id)
                    {
                        tracks[ti].expected_visible = false;   // hold the death-miss (peripheral glance)
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
            // ★SENIORITY IS OBSERVED AT BIRTH, not inferred later (common/exclusion). Birth is the only
            // moment at which "who was here first" is actually seen: resolving it on a later cycle would
            // have a real object, standing legitimately beside a real neighbour, wake up after a RESTART,
            // find the neighbour present, and declare ITSELF the junior. Anything not created this run
            // stays senior by default — the rule can fail to catch a collision, never invent one.
            if (auto it = fitter_->instances().find(new_id); it != fitter_->instances().end())
                it->second.exclusion.resolve_at_birth({c.x(), c.y(), 2.0f * cfg_.prior_radius, 2.0f * cfg_.prior_radius, 0.0f},
                                                      foreign_claims_,
                                                      c.z() - 0.5f * cfg_.prior_height, c.z() + 0.5f * cfg_.prior_height);
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
    // U(Σ): the belief's scalar uncertainty = Σ of the posterior stds over position + size. Computed here
    // rather than shared because a bottle's DOFs are its own (cx, cy, radius, height) — the reduction to a
    // number is the per-agent half; drawing it is not. cf. rc::geom::belief_uncertainty in the box concepts.
    const auto sigma_of = [&](int j) -> float {
        return std::sqrt(std::max(0.0f, inst.ai2_belief.covariance()(j, j)));
    };
    const float u_sigma = inst.ai2_initialized
                        ? sigma_of(0) + sigma_of(1) + sigma_of(3) + sigma_of(4)   // cx,cy,radius,height
                        : 0.0f;

    rc::dash::publish_belief_series({ts_plot_, ts_surprise_plot_, ts_cov_plot_, ts_res_plot_},
                                   {.node = inst.node_name,
                                    .free_energy  = free_energy,
                                    .fe_baseline  = inst.fe_baseline,
                                    .fe_surprise  = inst.fe_surprise,
                                    .uncertainty  = u_sigma,
                                    .residual_pts = static_cast<float>(inst.dbg_resid_pts)});
}

// Publish/refresh the "go see the hidden face" affordance (mirrors table_concept::step_epistemic). The
// node persists and re-offers every cycle; a low ΔH is published as-is so the controller's EFE
// selection won't pick a well-seen bottle (belief→knowledge governor without deleting the node).
void SpecificWorker::step_epistemic(rc::BottleInstance& inst)
{
    // The ZED origin in the room frame — bottle's planner takes the camera position explicitly (its NBV is a
    // hidden-face argument about which side of the cylinder has been seen), so it is computed here and
    // captured, not derived inside the planner.
    Eigen::Vector2f camera_xy(std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::quiet_NaN());
    if (inner_eigen_)
        if (const auto c = inner_eigen_->transform(rc::room::current_room_frame(*G), Mat::Vector3d(0.0, 0.0, 0.0), "zed", 0);
            c.has_value())
            camera_xy = Eigen::Vector2f(static_cast<float>(c->x()), static_cast<float>(c->y()));

    // The cycle around the planner call is SHARED (common/epistemic_step). ★For bottle the important part of
    // that move is the bail path: this function used to `return` bare on BOTH refusals, leaving a
    // just-completed affordance Completed for ever — the exact failure ObjectAffordance::hold_offered()
    // was written for and documents. The shared step has one bail path and it always holds the offer.
    // ★Bottle also has no object-node write: it publishes only the affordance, so `record` mirrors and stops.
    rc::epistemic::StepHooks<rc::EpistemicProposal> hooks;

    hooks.compute = [&]() -> std::optional<rc::EpistemicProposal>
    {
        if (not inst.ai2_initialized)
            return std::nullopt;   // belief not yet seeded (no fresh frame) → no NBV this cycle
        auto prop = epistemic_planner_.compute(inst.ai2_belief, camera_xy, cfg_.ai2_sigma_base_m,
                                               rc::nbv::sensor_from_graph(*G, inner_eigen_.get()),
                                               rc::nbv::collect_graph_obstacles(*G, inner_eigen_.get(),
                                                                                inst.node_id),
                                               room_polygon_);
        if (not prop.valid or not prop.is_finite())
            return std::nullopt;   // no camera pose / degenerate ray this cycle
        return prop;
    };

    hooks.record = [&](const rc::EpistemicProposal& published, float raw_gain)
    {
        inst.last_epistemic_gain = published.epistemic_gain;   // expose to the dashboard
        inst.dbg_nbv_gain_raw    = raw_gain;                   // what the belief asked for
        inst.dbg_nbv_gain_pub    = published.epistemic_gain;   // what the controller will see
    };

    hooks.on_affordance_created = [this] { trigger_graph_layout_twopi(); };

    // Gated CSV: ΔH + viewpoint + affordance state. Runs AFTER the affordance is refreshed, so the protocol
    // state it records is this cycle's, not last cycle's.
    hooks.on_published = [&](const rc::EpistemicProposal& published)
    { log_epistemic_csv(inst, published, camera_xy); };

    rc::epistemic::step(inst, *G, {.cooldown_cycles = cfg_.epistemic_cooldown_cycles}, hooks);
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
        rc::diag::open_rotating(epistemic_csv_, cfg_.epistemic_csv_path);
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
                   << rc::ObjectAffordance::state_name(inst.affordance.state()) << ','
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

// Poll the controller-owned protocol flags once per cycle. This REPLACES the update_node_attr_signal
// subscription: that signal fires for EVERY attribute change on EVERY node in the shared graph, and the
// agents that once subscribed to it could starve their own compute loop on the firehose.
//
// ★BOTTLE HAD NEITHER. Its modify_node_attrs_slot() was written, declared and never connected to anything —
// dead code that LOOKED like the mechanism — so ObjectAffordance::on_node_modified() was never called and
// bottle's affordance state machine never advanced past what bottle itself set. The controller taking the
// claim (Offered→Executing) and completing it (→Completed) was invisible to this agent: `affordance.state()`
// was stale for the life of every instance, and the epistemic CSV's `aff_state` column recorded that stale
// value. The graph-side reads (the completion hold in step_epistemic, hold_offered) go straight to the node
// and were unaffected, which is exactly why this stayed invisible.
void SpecificWorker::poll_affordance_protocol()
{
    if (not fitter_ or not G)
        return;   // may be called before the fit core exists — the guard every graph reader here needs
    // SHARED (common/object_affordance): see rc::poll_protocol for why this is POLLED and not driven by
    // update_node_attr_signal, and for what happened to the one agent that had neither.
    rc::poll_protocol(fitter_->instances(), *G);
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
