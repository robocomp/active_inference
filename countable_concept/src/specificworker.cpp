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
 * specificworker.cpp — countable_concept agent orchestration.
 *
 * Per compute() cycle: ingest the ZED YOLO "bed" masks, associate them to bed instances with the shared
 * InstanceTracker (birth / associate / merge; ricoh 360 detections are bearing-only and only raise attention),
 * run one AI2 recursive-Laplace belief update (CountableBelief) per assigned slice via process_countable_node, write
 * the fitted pose+geometry back to DSR (RT edge + dims + mesh + covariance), and emit epistemic action
 * proposals when a bed stays under-observed. Also feeds the standalone belief dashboard + evidence monitor
 * windows. The fit core is rc::CountableFitter, perception rc::MaskIngestor, DSR I/O rc::CountableSceneGraph. See
 * COUNTABLE.md for the belief/fit core.
 */

#include "specificworker.h"
#include "countable_gain_query.h"   // P1: external gain queries
#include "countable_card.h"   // make_card(): this agent's perception card
#include "../../common/perception_card/card_io.h"   // write_card_file
#include <filesystem>

#include "../../common/config_report/config_read.h"   // rc::cfg::Reader (SHARED)
#include "../../common/room_resolve/room_resolve.h"   // rc::room::current_room (proto-aware, deterministic)
#include "../../common/room_resolve/room_follow.h"        // rc::room::read_outline / drop_left_behind
#include "../../common/room_resolve/room_containment.h"   // rc::room::outline_usable

#include "../../common/diag_log/rotating_csv.h"   // keep the previous run instead of wiping it

#include "../../common/obj/convergence.h"   // rc::converge::step (SHARED)

#include "../../common/dashboard/belief_series.h"   // rc::dash::publish_belief_series (SHARED)

#include "../../common/birth_surprise/residual_field_reader.h"   // rc::read_residual_field (SHARED)
#include "countable_geometry.h"   // rc::geom pure footprint/uncertainty helpers

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

    // Registered rather than read raw, so the period appears in the startup table like every
    // other key. req() keeps the throw-if-missing behaviour: a period is not something to
    // default silently.
    int period = 0;
    rc::cfg::Reader(configLoader, cfg_.agent_name).req("Period.Compute", period,
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
    std::print("{}: SpecificWorker destroyed.\n", log_name());
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

    cleanup_owned_nodes();

    // Drop the LiDAR + RGB media subscribers BEFORE tearing down the graph/inner_eigen they read (each holds a
    // raw pointer). Mirrors bottle_concept.
    lidar_ingestor_.reset();
    rgb_ingestor_.reset();
    depth_ingestor_.reset();

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
    std::print("{}: initialize()\n", log_name());
    GenericWorker::initialize();

    // One-shot belief self-check (pure Eigen; recovers a synthetic bed pose+size from a box+clutter cloud,
    // plus the appearance FRONT door-mode resolver + reported-σ_yaw entropy).
    rc::CountableBelief::self_test();
    // Fitter self-check (motion_magnitude robustness + the "be-still-to-update" confirm_only gate).
    rc::CountableFitter::self_test();
    // Appearance door-ness metric self-check (OpenCV; a vertical-lined patch must out-score a plain one).
    rc::CountableProjection::self_test();
    // Birth-burst store self-check (cell dedup, cap, expiry, take-erases, local-consistency δ).
    rc::BirthFragment::self_test();
 
    // Ignore payload attributes in local graph updates to avoid unnecessary copying and processing of potentially large data
    G->set_ignored_attributes<cam_rgb_att, cam_depth_att, laser_X_att, laser_Y_att, laser_Z_att>();
    qInfo() << "Ignoring DSR RGBD payload attributes cam_rgb/cam_depth in local graph updates";


    if (not G)
    {
        qWarning() << QString::fromStdString(log_name()) << ": DSR graph not available in initialize()";
        return;
    }

    // ★THE WHOLE PRESENCE PROTOCOL IS SHARED (common/concept_presence). The 134 lines that used to sit here
    // were byte-identical across hood/bed/table and differed elsewhere only in comment wording — while
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
    scene_graph_ = std::make_unique<rc::CountableSceneGraph>(
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

    // Remove any "bed*" nodes left behind by a previous (crashed) run so this agent always starts
    // from a clean slate and never adopts a stale/drifted node (the instance tracker re-births them
    // data-driven from masks).
    remove_owned_countable_nodes();

    // Resolve room node
    if (const auto room = rc::room::current_room(*G); room.has_value())
        room_node_id_ = *room;
    else
        qWarning() << QString::fromStdString(log_name()) << ": no room node found at startup";

    // Active-inference fit core. Owns the instance map; collaborates with the ingestor + scene graph.
    fitter_ = std::make_unique<rc::CountableFitter>(
        G, inner_eigen_.get(), cfg_, mask_ingestor_.get(), scene_graph_.get());
    existence_ = std::make_unique<rc::CountableExistence>(G, cfg_);   // evidence-based removal (existence log-odds)

    // Part B: localization/chain covariance on the published RT edge (mirrors bottle_concept).
    gaussian_api_ = std::make_unique<DSR::InnerGaussianAPI>(G.get());
    fitter_->set_chain_cov_source(gaussian_api_.get(), "zed");
    // Object-anchor observation z_o for room_concept's landmark factor (expressed in the localizer base frame).
    fitter_->set_timing(&timing_);
    fitter_->set_object_observation(cfg_.publish_object_obs, cfg_.object_obs_frame);

    // YOLO-independent LiDAR range channel: lidar3D media-plane consumer that stages each cycle's sweep in the
    // room frame for the fitter's range factor. Dormant (no DDS participant) unless Model.LidarPrecision
    // > 0. Subscriber is brought up lazily on the compute/main thread once the lidar3D node + descriptor exist.
    lidar_ingestor_ = std::make_unique<rc::ConceptLidarIngestor>(G, inner_eigen_.get(),
        [this] { return rc::LidarGates{cfg_.lidar_precision, cfg_.free_space_precision, cfg_.lidar_bpearl_precision}; });

    // ZED RGB media-plane consumer for appearance-based FRONT (door) detection. Dormant (no DDS participant)
    // unless Concept.FrontDetectEnabled. Subscriber comes up lazily on the compute/main thread once
    // the "zed" node + media descriptor exist (media-plane consumer pattern).
    camera_planes_wanted_ = cfg_.front_detect_enabled or cfg_.contour_check_enabled;
    rgb_ingestor_   = std::make_unique<rc::RgbIngestor>(G, &camera_planes_wanted_, "zed");
    // The DEPTH plane, for the metric half of the classifier-free contour channel: "is there a surface at
    // the distance I predict, with space behind its edge?" — the question that separates a bed from a
    // bed-shaped region of a wall, which no amount of RGB gradient can.
    depth_ingestor_ = std::make_unique<rc::DepthIngestor>(G, &camera_planes_wanted_, "zed");

    // rc::EpistemicPlanner owns the BELIEF half of the NBV (Σ, ΔI, the adequacy gap); the SENSOR half — where
    // the detector can actually fire — comes from the camera model it is handed each cycle in step_epistemic().
    // ONE detector envelope, both directions: the viewpoint the planner asks for is the argmax of the same
    // model the removal channel uses to decide how much a missing mask is worth.
    const rc::detect::DetectorEnvelope det_env{cfg_.detect_min_fill, cfg_.detect_max_fill, cfg_.detect_soft};
    epistemic_planner_.set_detector_envelope(det_env);
    epistemic_planner_.set_agent_name(cfg_.agent_name);
    if (existence_) existence_->set_detector_envelope(det_env);

    // Perception card: what this agent's P(detect) really multiplies, from the config it runs with. The agent's
    // directory is the cwd (the fleet scripts start every agent with `cd $B/$a`), so etc/perception_card.json lands next to etc/config.toml.
    rc::card::write_card_file(std::filesystem::current_path(), rc::make_card(cfg_));
    presence_coordinator_.publish_card_json(rc::card::graph_json(rc::make_card(cfg_)));

    // Stale affordance nodes are swept on entering Operating (presence hook) and on shutdown — see
    // remove_stale_affordance_nodes(), keyed on the parent object type (robust to node-name renames).

    // Standalone Qt dashboard + evidence-monitor windows (belief plots + per-instance snapshot).
    // Shadow-mode birth/death record (CONCEPT_AGENT_LIFECYCLE.md §4.2). Recording only — see
    // log_phantom_event(). Truncating: one file per run.
    phantom_log_.open("etc/" + cfg_.concept_class + "_phantom_events.csv");

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
    // ARMS the unread sweep; check_registry_complete.sh countable_concept is the grep that keeps the claim
    // honest. Re-run it whenever a config read is added.
    rc::cfg::exempt_generated_prefixes();
    rc::cfg::registry().declare_complete(cfg_.agent_name);
    rc::cfg::Reader(configLoader, cfg_.agent_name).publish("etc/config_effective.csv");
}

// ─── Main compute loop ───────────────────────────────────────────────────────────────────────────

// The WORKING room's outline (shared reader) → the fitter (wall-flush factor, wall line-of-sight) and the birth rule.
// ★Always that room's own: cleared when it has none, so births WAIT instead of using the room just left.
void SpecificWorker::refresh_room_geometry()
{
    if (not G or room_node_id_ == 0) return;
    auto poly = rc::room::read_outline(*G, room_node_id_);
    if (not rc::room::outline_usable(poly)) poly.clear();
    Eigen::Vector2f centroid = Eigen::Vector2f::Zero();
    float xmin = 1e9f, xmax = -1e9f, ymin = 1e9f, ymax = -1e9f;
    for (const auto& v : poly)
    {
        centroid += v;
        xmin = std::min(xmin, v.x()); xmax = std::max(xmax, v.x());
        ymin = std::min(ymin, v.y()); ymax = std::max(ymax, v.y());
    }
    if (not poly.empty()) centroid /= static_cast<float>(poly.size());
    if (room_node_id_ != outline_room_ or poly.size() != outline_n_)
    {
        if (poly.empty())
            std::print("{}: [room] room {} has NO usable outline (none yet, or still a proto) — births WAIT\n", log_name(), room_node_id_);
        else
            std::print("{}: [room] room {} outline {} verts, x∈[{:.2f},{:.2f}] y∈[{:.2f},{:.2f}]\n", log_name(),
                       room_node_id_, poly.size(), xmin, xmax, ymin, ymax);
        outline_room_ = room_node_id_;
        outline_n_    = poly.size();
    }
    fitter_->set_room_geometry(centroid, std::move(poly), room_node_id_);
}

// SHADOW-MODE birth/death recorder — CONCEPT_AGENT_LIFECYCLE.md §4.2, theory in MODEL_HISTORY.md §4.
// RECORDS ONLY; it can never alter a birth or a removal. The attribution fields captured at death
// (p_detect / in-FoV / central) are what tell a genuine classifier phantom from one of OUR removal defects —
// a death with LOW p_detect means the log is recording a removal bug, and must not be learned from.
void SpecificWorker::log_phantom_event(std::string_view event, std::uint64_t id, std::string_view name,
                                       float x, float y, const rc::CountableInstance* inst, std::string_view note)
{
    if (not phantom_log_.is_open())
        return;
    rc::history::PhantomEvent e;
    e.event = event; e.id = id; e.name = name; e.x = x; e.y = y; e.note = note;
    // Observer pose → view bearing. The classifier failure is VIEWPOINT-dependent, so the eventual p_FA field
    // is keyed on (world cell × bearing); a place-only key would suppress a genuine object placed there.
    // Observer pose → view bearing. SHARED (common/phantom_log/observer_pose.h): the classifier failure is
    // VIEWPOINT-dependent, so the false-alarm field is keyed on (world cell × bearing), never place alone.
    rc::history::note_observer(e, *G, inner_eigen_.get(), x, y);
    if (inst)   // death: carry the existence state that says whether this was a CONFIDENT kill
    {
        e.age_cycles    = inst->processed_cycles;
        e.p_detect      = inst->dbg_ex_pdetect;
        e.central_frac  = inst->dbg_ex_central;
        e.in_fov_frac   = (inst->dbg_ex_sil_ntotal > 0)   // ★a FRACTION, not a probed/not-probed flag:
                        ? static_cast<float>(inst->dbg_ex_sil_ndet) / inst->dbg_ex_sil_ntotal : 0.0f;
        e.exist_logodds = rc::countable::logit(inst->p_exists());   // the PRODUCT (occupancy x class)
    }
    phantom_log_.write(e);
}

// [timing] One row per tick → timing_log.csv, in the directory of Model.AI2CsvPath (ai2_log.csv); "etc/" if unset.
void SpecificWorker::write_timing_row()
{
    if (not timing_csv_.is_open())
    {
        if (timing_csv_failed_) return;
        const std::filesystem::path ai2{cfg_.ai2_csv_path};
        const auto dir = ai2.has_parent_path() ? ai2.parent_path() : std::filesystem::path{"etc"};
        if (not rc::diag::open_rotating(timing_csv_, (dir / "timing_log.csv").string(), rc::TickTiming::header()))
        {
            timing_csv_failed_ = true;
            return;
        }
        timing_csv_.imbue(std::locale::classic());
    }
    const auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count();
    timing_.write_row(timing_csv_, ts);
}

void SpecificWorker::compute()
{
    if (not G or not rt_api_)
        return;

    // The room this agent works in follows the robot, every cycle (was: latched at the first room). No room yet: wait.
    room_node_id_ = rc::room::follow_current_room(room_node_id_, rc::room::current_room(*G));
    if (room_node_id_ == 0) return;

    timing_.reset();
    const rc::RtMemoTickScope rt_memo_scope(rt_memo_);   // one consistent pose per tick; off again on every return
    rc::g_tick_timing = &timing_;   // sub-stage timers + rt_calls in projection/fitter/existence
    const auto tick_t0 = std::chrono::steady_clock::now();

    // Push the room polygon + interior centroid into the fitter (wall-flush factor). Cheap; room_concept refines it.
    refresh_room_geometry();

    // Controller-owned affordance flags (claim / completion / epistemic_pending). Polled here rather than
    // pushed by update_node_attr_signal — see the connect block in initialize() for why that subscription
    // could starve this very loop.
    poll_affordance_protocol();
    // K2b: approaches whose object was deleted from outside (del_node_slot) go now, on the main thread.
    for (auto& a : retired_approaches_) a.remove();
    retired_approaches_.clear();

    bool fresh_masks = false;
    {
        const rc::StageTimer t(&timing_.mask_refresh);   // [timing] mask refresh
        fresh_masks = mask_ingestor_->refresh();
    }

    // EvidenceMonitor per-cycle counters (cumulative *_cum fields persist across cycles). The producers below
    // (tracker / merge / removal) add to these; the snapshot is pushed at the end of the cycle.
    ev_g_.births = ev_g_.merges = ev_g_.removals = ev_g_.births_refused = 0;
    ev_g_.mask_stale = not fresh_masks;

    // Snapshot the residual (surprise) field ONCE per cycle before the tracker, so fused birth (run_instance_
    // tracker) and the logging probe (compute() tail) share one read. No-op unless a probe/fusion flag is on.
    if (cfg_.birth_fusion or cfg_.birth_surprise_probe) read_residual_field();
    read_occupied_cells();   // G6: residual's latched cells, for the standpoint check (re-read only when the publisher wrote)

    {
        const rc::StageTimer t(&timing_.tracker);   // [timing] tracker (birth/associate/merge)
        run_instance_tracker();   // data-driven birth / associate / merge (the only instance-lifecycle path)
    }
    if (cfg_.thumbnails_enabled) update_thumbnails();   // K2b: the best crop retina offered for each instance, per viewing angle

    // Stage this cycle's LiDAR sweep (room frame) for the fitter's range factor. clear-then-set so the factor
    // never consumes a stale sweep; pump() is main-thread (reads the graph) + dormant while the feature is off.
    fitter_->clear_lidar_sweep();
    bool fresh_sweep = false;
    // ★ONE graph walk per cycle for the SHARED mutual-exclusion rule: who else claims room space right now.
    // Feeds BOTH the birth filter (an unmatured candidate on somebody else's object accrues no evidence) and
    // the existence occupancy discount (a junior instance stops counting its senior's returns). Main thread —
    // collect_graph_obstacles uses a ts==0 transform (CLAUDE.md).
    {
        const rc::StageTimer t(&timing_.foreign_claims);   // [timing] foreign-claims graph walk
        foreign_claims_ = rc::exclusion::foreign_claims(*G, inner_eigen_.get(), cfg_.concept_class);
    }
    if (existence_) existence_->set_foreign_claims(&foreign_claims_);
    if (fitter_) fitter_->set_foreign_claims(&foreign_claims_);   // the FIT judges the same geometry
    if (fitter_)
    {   // G3: the other objects' published beliefs, the persistent fit's foreign density (inert with the flag off)
        const rc::StageTimer t(&timing_.foreign_claims);
        fitter_->refresh_neighbours();
    }

    if (lidar_ingestor_)
    {
        const rc::StageTimer t(&timing_.lidar_pump);   // [timing] LiDAR pump + stage
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
    {
    const rc::StageTimer t(&timing_.rgb_stage);   // [timing] RGB (+depth) pump + stage
    if (rgb_ingestor_)
    {
        rgb_ingestor_->pump();
        if (rgb_ingestor_->fresh())
            fitter_->set_rgb_frame(rgb_ingestor_->frame(), rgb_ingestor_->stamp_ms());
    }
    if (depth_ingestor_)
        depth_ingestor_->pump();
    }

    // Robot/camera ego-motion (transform chain) → the "be-still-to-update" confirm-only gate. Once per cycle,
    // BEFORE the instance loop so every run_inference this cycle reads the current robot speed (matches chair).
    {
        const rc::StageTimer t(&timing_.ego_motion);   // [timing] ego-motion
        fitter_->update_ego_motion();
    }

    // Generic DSR type "object"; the class filter is the "countable_" name prefix + object_subtype attr.
    // The robot left a bed's room: fitting it here would corrupt the pose ltom remembers; ltom offers it back.
    rc::room::drop_left_behind(*G, cfg_.concept_class + "_", room_node_id_, [this](std::uint64_t id)
    {
        if (auto it = fitter_->instances().find(id); it != fitter_->instances().end())
            teardown_consumers(it->second);
        fitter_->forget_node(id);
    });
    const auto object_nodes = G->get_nodes_by_type("object");
    for (const auto& node : object_nodes)
        if (node.name().starts_with(cfg_.concept_class + "_"))   // "bed_": a bedroom_* node is not a bed
        {
            const rc::StageTimer t(&timing_.inst_total);   // [timing] per-instance total (summed)
            process_countable_node(node);
        }
    serve_gain_queries();   // P1: the A/B/A visit's promise at a scripted standpoint (after the fits: this tick's beliefs)

    // Ricoh 360 = peripheral attention: associate ricoh detections to beds BY DIRECTION (after the ZED fits,
    // so bed positions are current); an unassigned bearing becomes a "seek a ZED view here" attention target.
    {
        const rc::StageTimer t(&timing_.ricoh);   // [timing] ricoh bearings
        process_ricoh_bearings();
    }

    // The detector's truth table. BEFORE removal on purpose: an instance killed this cycle must still
    // leave the row that records the look which killed it — that row is the whole point of the file.
    log_detect_probe();

    // Evidence-based removal: each existence channel integrates on its OWN sensor cadence (silhouette/mask on a
    // fresh mask frame, LiDAR carve on a fresh sweep) — a camera-only cycle still accrues absence, a LiDAR-only
    // cycle still carves free space. After the fits so footprints are current. OFF unless enabled.
    if (cfg_.existence_removal_enabled)
    {
        // The camera planes for the classifier-free contour channel. Borrowed for the duration of the call
        // and read on THIS thread only — a cv::Mat copy is a refcounted shallow handle, so one crossing a
        // thread boundary would have to be deep-copied (CLAUDE.md). `fresh` is the RGB clock: the contour is
        // scored against those pixels and pinned to their stamp, so re-scoring a frame already scored would
        // charge the same observation twice.
        const bool cam_fresh = (rgb_ingestor_ and rgb_ingestor_->fresh())
                            or (depth_ingestor_ and depth_ingestor_->fresh());
        const cv::Mat* rgb_f   = rgb_ingestor_   ? &rgb_ingestor_->frame()   : nullptr;
        const cv::Mat* depth_f = depth_ingestor_ ? &depth_ingestor_->frame() : nullptr;
        const std::uint64_t cam_stamp = (rgb_ingestor_ and rgb_ingestor_->stamp_ms() > 0)
                                      ? rgb_ingestor_->stamp_ms()
                                      : (depth_ingestor_ ? depth_ingestor_->stamp_ms() : 0);
        existence_->set_camera_frames(rgb_f, depth_f, cam_stamp, cam_fresh);
        const rc::StageTimer t_ex(&timing_.existence);   // [timing] existence evaluation (+removal)
        existence_->update_and_remove(*fitter_, lidar_ingestor_.get(), fresh_masks, fresh_sweep, ev_g_,
            [this](std::uint64_t id, const rc::CountableInstance& inst)
            {   // shadow-mode death record (§4.2) — p_detect here says whether this was a CONFIDENT
                // disconfirmation (a real phantom) or a weak one (more likely our own removal bug)
                log_phantom_event("DEATH", id, inst.node_name,
                                  inst.model.state().cx, inst.model.state().cy, &inst,
                                  std::format("L {:.2f}", inst.existence.logodds()));
            });
    }

    // K2b changed object: ghosts at remembered spots, the identity link of objects born new (after the fits and the removals,
    // so the live instances are this tick's).
    if (cfg_.changed_object_enabled) update_identity_links();

    // The CLASS factor of existence (existence = occupancy x class): soft SINGLETON + shape evidence → retire mis-detections. The
    // worker sees ALL instances, so mutual inhibition + the removal decision live here (not in the fitter). It
    // owns each instance's class log-odds + a dedicated debounce, so it works whether or not the
    // sensor-existence channel above is enabled. Continuous/bounded; a genuine second bed survives.
    {
        const rc::StageTimer t(&timing_.class_factor);   // [timing] class factor
        apply_class_factor();
    }

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
    {
        const rc::StageTimer t(&timing_.dashboard);   // [timing] evidence monitor + dashboard series upkeep
        refresh_evidence_monitor();
        prune_dead_series();   // drop timeseries lines for beds that were removed from the graph
        log_birth_surprise();  // EXPERIMENTAL read-only: residual grid → birth-surprise regions (cfg-gated, off by default)
    }

    timing_.n_inst = static_cast<int>(fitter_->instances().size());
    timing_.tick = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tick_t0).count();
    write_timing_row();

    fps_counter_.print("[" + cfg_.agent_name + " Compute]");   // std::cout heartbeat: Period/Fps/cpu%/mem (every ~1s)
}

// EXPERIMENTAL, read-only. Reinterpret residual_concept's `grid` node (under room) as an unexplained-occupancy
// SURPRISE field and cluster its confidently-unexplained cells into candidate regions (birth_surprise_probe.h).
// Log, per cycle, the regions NOT already covered by a believed bed alongside the tracker's actual birth count,
// so we can see whether surprise flags a real new bed cleanly (and stays quiet on phantoms) BEFORE letting it
// drive the lifecycle. Never writes the graph; never births. See [[bed-birth-surprise-probe]].
// G6: residual_concept's latched occupied cells (grid_occupied_cells + grid_cell_size) priced exactly as the controller prices
// them: through the published belief field (grid_occupancy_prob / _var / _meta, via rc::read_residual_field) and the controller's
// own residual_risk::risks_from_field (+inf = hard, finite = soft nats) — the input of the shared standpoint oracle (K3). Keyed on the publish stamps of the cells and the field, like the
// controller's obstacle tracker: the publisher writes ~2 Hz and unchanged stamps mean what we hold is still the truth. A vanished
// node empties it (no belief, no veto); a missing field makes every cell hard, as there. Main thread.
void SpecificWorker::read_occupied_cells()
{
    const auto g = G->get_node("residual");
    if (not g.has_value()) { occupied_cells_ = {}; occupied_stamp_cells_ = occupied_stamp_prob_ = 0; return; }
    const auto stamp_cells = G->get_attrib_timestamp<grid_occupied_cells_att>(g.value()).value_or(0);
    const auto stamp_prob  = G->get_attrib_timestamp<grid_occupancy_prob_att>(g.value()).value_or(0);
    if (stamp_cells != 0 and stamp_cells == occupied_stamp_cells_ and stamp_prob == occupied_stamp_prob_) return;
    occupied_stamp_cells_ = stamp_cells;
    occupied_stamp_prob_  = stamp_prob;
    occupied_cells_ = {};
    const auto cs = G->get_attrib_by_name<grid_cell_size_att>(g.value());
    const auto cells = G->get_attrib_by_name<grid_occupied_cells_att>(g.value());
    if (not (cs.has_value() and cells.has_value())) return;
    rc::GridField field;
    rc::residual_risk::BeliefField bf;   // stays invalid (every cell hard) when the field cannot be read
    if (rc::read_residual_field(*G, room_node_id_, field))
    {
        bf.prob = field.prob; bf.var = field.var;
        bf.xmin = field.xmin; bf.ymin = field.ymin; bf.cell = field.cell;
        bf.width = field.width; bf.height = field.height;
    }
    occupied_cells_ = rc::standpoint::cells_from_xyz(cells.value().get(), cs.value(), bf);
}

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

    // Believed bed footprints (room frame) — a region under one is already explained, NOT a birth. This is
    // the one genuinely per-object step: a bed's two footprint extents. Everything after it — both CSVs,
    // the fusion probe, the throttled console line — is SHARED (common/birth_surprise/birth_surprise_log.h).
    std::vector<rc::FootprintBox> footprints;
    for (const auto& [id, inst] : fitter_->instances())
        if (inst.ai2_initialized)
        { const auto& s = inst.ai2_belief.state(); footprints.push_back({s.cx, s.cy, s.w, s.h, s.yaw}); }

    birth_surprise_log_.write(residual_field_, footprints, last_countable_dets_xy_,
                              ev_g_.births, fitter_->instances().size());
}


// ─── Per-node processing + publish ───────────────────────────────────────────────────────────────

// Process one "bed" DSR node this cycle: ensure its instance exists, then fuse each assigned ZED slice
// (one belief update per slice, gated to a fresh mask frame) or age the belief when no mask arrived, and // hand the result to publish_countable_cycle. run_instance_tracker has already associated this cycle's slices.
void SpecificWorker::process_countable_node(const DSR::Node& node)
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
        // (Per-instance time-series are registered idempotently by publish_countable_diagnostics each cycle.)
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
    CountableObservation last_obs;
    bool updated = false;
    bool cam_data = false;   // [timing] this instance had camera points this tick
    // [timing] classify this instance's tick: camera data / LiDAR rays only / nothing (runs on every exit path)
    struct TimingClassify
    {
        rc::TickTiming& t; const bool& cam; bool ray;
        ~TimingClassify() { if (cam) ++t.n_cam; else if (ray) ++t.n_ray; else ++t.n_none; }
    } timing_classify{timing_, cam_data, fitter_->has_lidar_sweep()};

    // Multi-slice fusion: one belief update per assigned ZED slice of this bed. Sequential Bayesian updates
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
    // Colour drift (K2b, Concept.AppearanceEnabled; DISPLAY + names only): every cycle, seen or not, on wall-clock time —
    // an unobserved object's colour genuinely becomes less certain as lighting and viewpoint move on (as in the legacy chair).
    if (cfg_.appearance_enabled)
    {
        const auto tp_now = std::chrono::steady_clock::now();
        if (inst.last_appearance_tp.time_since_epoch().count() != 0)
            inst.appearance.inflate_for_age(std::chrono::duration<float>(tp_now - inst.last_appearance_tp).count());
        inst.last_appearance_tp = tp_now;
    }
    if (fresh_frame and fresh_capture)
    {
        // The FRONT camera's verdict on this mask frame (K2b): a ZED slice assigned to this instance. Once per new frame.
        inst.front_seen = std::ranges::any_of(inst.assigned_mask_idxs, [&](int idx)
            { return idx >= 0 and idx < static_cast<int>(pkt.slices.size()) and pkt.slices[idx].is_zed(); });
        if (inst.front_seen) inst.front_seen_frame = pkt.frame_id;
        // Colour: both channels of every assigned slice; a slice with no colour cells (ricoh) is a no-op inside observe().
        if (cfg_.appearance_enabled)
            for (const int idx : inst.assigned_mask_idxs)
                if (idx >= 0 and idx < static_cast<int>(pkt.slices.size()))
                {
                    const auto& sl = pkt.slices[idx];
                    inst.appearance.observe(sl.color_chroma, sl.color_var, sl.color_neff, sl.color_lum, sl.color_lum_var, sl.color_lum_n);
                }
        for (const int idx : inst.assigned_mask_idxs)
        {
            auto obs = fitter_->observe_slice(inst, idx);
            if (not obs.has_fresh_data)
                continue;
            free_energy = fitter_->run_inference(inst, obs);
            if (obs.has_fresh_data) cam_data = true;
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
        cam_data = last_obs.has_fresh_data;
    }

    publish_countable_cycle(inst, node, last_obs, free_energy);
}


// Publish one fitted bed: persist belief→DSR, assess convergence, then push diagnostics + intentions.
// Each step short-circuits if its node lookup fails (the node may have been removed mid-cycle).
void SpecificWorker::publish_countable_cycle(rc::CountableInstance& inst,
                                         const DSR::Node& node,
                                         const CountableObservation& observation,
                                         float free_energy)
{
    const rc::StageTimer t_pub(&timing_.publish);   // [timing] graph write + diagnostics + intentions
    const auto node_id = node.id();
    if (not scene_graph_->persist_countable_belief(inst, node_id, room_node_id_, free_energy))
        return;
    if (not assess_countable_state(inst, node_id, free_energy))
        return;
    publish_countable_diagnostics(inst, observation, free_energy);
    publish_countable_intentions(inst, node_id, observation, free_energy);
}

// Run the convergence/stability step for one bed; re-resolves the node by id (false if it is gone).
bool SpecificWorker::assess_countable_state(rc::CountableInstance& inst, uint64_t node_id, float free_energy)
{
    auto node_opt = G->get_node(node_id);
    if (not node_opt.has_value())
        return false;

    step_convergence(inst, node_opt.value(), free_energy);
    return true;
}

// Feed this instance's time-series to the standalone dashboard: FE + baseline, FE-surprise, U(Σ) covariance,
// residual count, inferred dims (w,h), and size posterior σ. Series are added idempotently every cycle.
void SpecificWorker::publish_countable_diagnostics(const rc::CountableInstance& inst,
                                               const CountableObservation& observation,
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

void SpecificWorker::publish_countable_intentions(rc::CountableInstance& inst,
                                              uint64_t node_id,
                                              const CountableObservation& observation,
                                              float free_energy)
{
    // step_epistemic now owns the propose-vs-withdraw decision via the planner's ΔH (it withdraws
    // when the expected information gain falls below threshold), so it runs unconditionally rather
    // than being gated on the legacy coverage-deficit proxy.
    // Concept.EpistemicOfferEnabled = false (K2b; the legacy chair's AuRo C3(b) comparison arm): no viewpoint offer; the belief
    // still updates from whatever the robot happens to see, and the approach is still offered.
    if (cfg_.epistemic_offer_enabled)
        if (auto node_opt = G->get_node(node_id); node_opt.has_value())
            step_epistemic(inst, node_opt.value());
    if (cfg_.approach_enabled)
        step_approach(inst);
}

// ─── Initialisation helpers ──────────────────────────────────────────────────────────────────────

void SpecificWorker::load_config(const ConfigLoader& cfg)
{
    cfg_ = rc::load_countable_config(cfg);
    // Declarative-priors experiment: report whether common/concept_manifest/bed.concept.toml can
    // reproduce the priors this config just produced. Read-only — the manifest is not authoritative yet.
    rc::verify_countable_manifest(cfg_, rc::countable_manifest_path(cfg_.concept_class));
}



// The ZED as the NBV's forward model: both FoVs from the REAL intrinsics, and the optical centre's height above
// the FLOOR datum — the same datum the bed box's z ∈ [0, H] is expressed in, which is what makes the
// vertical framing term mean anything.
//
// ★The VERTICAL channel is the whole point. `roi_fill` is max(Δcol/W, Δrow/H) and for a 1.7 m bed seen
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
    {
        if (const auto rtz = rc::cached_rt(inner_eigen_, rc::room::current_room_frame(*G), "zed", 0); rtz.has_value())
            s.height_m = static_cast<float>(rtz.value()(2, 3));
    }
    return s;
}

// ─── Per-cycle steps ─────────────────────────────────────────────────────────────────────────────

// Convergence latch for one bed: measure how far the accepted state moved this cycle and, once it holds
// still for K_stable cycles, publish model_stable + the model_uncertainty_att readout (posterior std sum).
void SpecificWorker::step_convergence(rc::CountableInstance& inst,
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
           std::abs(s.countable_height - p.countable_height) + std::abs(s.yaw - p.yaw) + std::abs(s.leg_inset - p.leg_inset))
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
        std::print("{}: node '{}' STABLE (F={:.4f})\n", log_name(), inst.node_name, free_energy);
}

// One instance's offer from the planner (Σ-based D-optimal NBV; the engine's predicted gain on the persistent path).
std::optional<rc::EpistemicProposal> SpecificWorker::compute_offer(rc::CountableInstance& inst)
{
    const OfferInputs in = offer_inputs(inst);
    rc::EpistemicProposal prop =
        epistemic_planner_.compute(inst.ai2_belief, cfg_.ai2_range_noise_lat_per_m, cfg_.ai2_sigma_base_m,
                                   in.sensor, in.obstacles,
                                   // The reachable region. refresh_room_geometry() already loads this
                                   // polygon for the fit's wall factor; the NBV never saw it, so every
                                   // viewpoint — including the ones behind the wall the bed is pushed
                                   // against — read as reachable. Empty until room_concept publishes, and
                                   // is_reachable then imposes no constraint (the pre-existing behaviour).
                                   fitter_->room_polygon(),
                                   // G5: the Helios the offer's look stages rays with (persistent path; else none)
                                   in.lidar,
                                   // G6: residual's latched occupied cells (the controller reads the same attribute)
                                   &occupied_cells_);
    // Degenerate fit, or no sensor model yet — retry next cycle.
    if (not prop.valid or not prop.is_finite())
        return std::nullopt;
    return prop;
}

SpecificWorker::OfferInputs SpecificWorker::offer_inputs(const rc::CountableInstance& inst)
{
    return {zed_sensor_model(), rc::nbv::collect_graph_obstacles(*G, inner_eigen_.get(), inst.node_id), fitter_->offer_lidar(inst)};
}

void SpecificWorker::serve_gain_queries()
{
    const std::filesystem::path req = fitter_->gain_query_request_path();
    if (req.empty()) return;
    std::error_code ec;
    const auto mt = std::filesystem::last_write_time(req, ec);
    if (ec or mt == gain_query_mtime_) return;          // no file, or nothing new since the last read
    std::ifstream f(req);
    const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    // a half-written last line is re-read next tick: remember the mtime only once the text is whole
    if (rc::gain_query_text_complete(text)) gain_query_mtime_ = mt;
    for (const rc::GainQuery& q : rc::new_gain_queries(text, gain_query_last_id_))
    {
        gain_query_last_id_ = std::max(gain_query_last_id_, q.id);
        for (auto& [id, inst] : fitter_->instances())
        {
            if (q.node != "*" and q.node != inst.node_name) continue;
            if (not inst.ai2_initialized or not inst.ai2_belief.params().persistent) continue;   // no engine offer to ask
            const OfferInputs in = offer_inputs(inst);
            const float g = epistemic_planner_.gain_at(inst.ai2_belief, {q.x, q.y}, q.yaw, in.sensor, in.obstacles, in.lidar);
            std::optional<rc::CountableFitter::OfferCheck> chk;
            if (const auto& h = inst.offer_cache.held; h.has_value() and inst.offer_cache.serial == inst.ai2_belief.gn_update_serial())
                chk = rc::CountableFitter::OfferCheck{
                    inst.offer_cache.serial, h->epistemic_target_x_m, h->epistemic_target_y_m, h->epistemic_target_yaw_rad,
                    h->promised_nats,
                    epistemic_planner_.gain_at(inst.ai2_belief, {h->epistemic_target_x_m, h->epistemic_target_y_m},
                                               h->epistemic_target_yaw_rad, in.sensor, in.obstacles, in.lidar)};
            fitter_->log_gain_query(inst, q.id, q.x, q.y, q.yaw, g, chk);
            std::print("{}: [gain-query {}] {} at ({:.3f}, {:.3f}, {:.3f}) -> {:.4f} nats (serial {})\n", log_name(), q.id,
                       inst.node_name, q.x, q.y, q.yaw, g, inst.ai2_belief.gn_update_serial());
        }
    }
}

// Publish/refresh the epistemic next-best-view affordance for one bed from its belief Σ (D-optimal NBV),
// with a post-completion cooldown that suppresses the published gain so a just-finished bed isn't re-claimed.
void SpecificWorker::step_epistemic(rc::CountableInstance& inst, DSR::Node& node)
{
    // The cycle around the planner call is SHARED (common/epistemic_step): the cooldown, the
    // controller-completion hold, the gain suppression and verification floor, the affordance refresh, and —
    // the reason it is shared rather than merely deduplicated — the single bail path that ALWAYS calls
    // hold_offered(). Only the planner call and what this agent records are bed's.
    rc::epistemic::StepHooks<rc::EpistemicProposal> hooks;

    hooks.compute = [&]() -> std::optional<rc::EpistemicProposal>
    {
        // Σ-based D-optimal NBV from the belief. Skip until the belief has seen its first frame (else Σ is
        // the broad prior and the proposal is moot).
        if (not inst.ai2_initialized)
            return std::nullopt;
        // Persistent path: once per engine update serial and standing-world version (the residual's latched cells and their
        // field), held through a cooldown (common/epistemic_step/offer_cache.h). The legacy belief has no update serial, so
        // it computes every tick as before.
        if (inst.ai2_belief.params().persistent)
            return inst.offer_cache.get(inst.ai2_belief.gn_update_serial(),
                                        occupied_stamp_cells_ * 1000003ull ^ occupied_stamp_prob_,
                                        inst.epistemic_cooldown > 0, [&] { return compute_offer(inst); });
        return compute_offer(inst);
    };

    hooks.record = [&](const rc::EpistemicProposal& published, float /*final gain: see nbv_gain_raw below*/)
    {
        // Mirror the proposal onto the instance so ai2_log records what was actually proposed. Without this
        // the only record was stdout, and the affordance node carries the FROZEN pose during an executing
        // claim — so "the target is too close" was unanswerable after the fact from either source.
        // nbv_gain_raw = the PROMISED nats (the best face's raw ΔH before `useful`, the suppression, the floor and p_stand).
        // step()'s raw_gain is the planner's FINAL gain, already × useful ≡ 0 here ("no consumer, no bid"), so it logged 0 on
        // every row (engine1 diagnosis item 4; plan rev 2 F1.1).
        inst.dbg_nbv_standoff = published.chosen_standoff_m;
        inst.dbg_nbv_gain_raw = published.promised_nats;      // what the belief asked for
        inst.dbg_nbv_gain_pub = published.epistemic_gain;     // what the controller will see
        inst.dbg_nbv_target_x = published.epistemic_target_x_m;
        inst.dbg_nbv_target_y = published.epistemic_target_y_m;
        inst.dbg_nbv_pdetect  = published.chosen_p_detect;
        inst.dbg_nbv_vfov     = published.sensor_vfov_rad;
        // G5: the promise, beside the delivery the fitter logs per update (persistent path; offers.csv)
        fitter_->log_offer(inst, published.epistemic_target_x_m, published.epistemic_target_y_m,
                           published.epistemic_target_yaw_rad, published.promised_nats);
        // Write attributes to the bed node (read by legacy consumers)
        scene_graph_->write_epistemic_proposal(node, published);
    };

    hooks.on_affordance_created = [this] { trigger_graph_layout_twopi(); };

    rc::epistemic::step(inst, *G,
                        {.cooldown_cycles   = cfg_.epistemic_cooldown_cycles,
                         // Verification pull: absence never deletes a bed the robot hasn't properly looked
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
    // Instance creation is owned by the compute() loop (process_countable_node → ensure_instance), which runs
    // AFTER note_birth() each cycle, so the birth seed is in place. The loop also covers externally-created
    // bed nodes (every cycle), so nothing is lost by not creating here.
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
    // remove_owned_countable_nodes() then calls G->delete_node() on the MAIN thread — same-thread emit ⇒
    // Auto resolves to DIRECT ⇒ this slot runs synchronously with `fitter_` still null. Dormant unless a
    // previous run actually left "bed*" nodes behind, which is why it had not been seen; it SIGSEGV'd
    // table_concept exactly this way on 2026-08-06. cabinet_concept guards its del_node_slot for the same
    // reason (and missed its attrs slot).
    if (not fitter_)
        return;
    // Notify affordance in case its own DSR node was deleted externally
    for (auto& [countable_id, inst] : fitter_->instances())
    {
        if (inst.affordance.node_id() == id)
            inst.affordance.on_node_deleted(id);
        inst.approach.on_node_deleted(id);   // K2b: its own approach node deleted from outside
    }

    if (fitter_->instances().count(id))
    {
        std::print("{}: node {} removed from DSR, destroying instance\n", log_name(), id);
        // K2b: its approach node would stay on offer for an object that no longer exists; compute() deletes it on the main
        // thread next tick (as the legacy chair does)
        retired_approaches_.push_back(std::move(fitter_->instances().at(id).approach));
        fitter_->instances().erase(id);
    }
}

// ─── Lifecycle stubs ─────────────────────────────────────────────────────────────────────────────

void SpecificWorker::emergency()
{
    std::print("{}: emergency()\n", log_name());
}

void SpecificWorker::restore()
{
    std::print("{}: restore()\n", log_name());
}

int SpecificWorker::startup_check()
{
    std::print("{}: startup_check()\n", log_name());
    return 0;
}




