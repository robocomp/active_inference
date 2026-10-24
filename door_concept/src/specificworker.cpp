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
 * SpecificWorker — door_concept agent
 *
 * Implements the Active Inference loop described in ../CONCEPT_AGENT_RECIPE.md:
 *
 *  ① Read sensing attributes from DSR door nodes
 *  ② Update the historical sample queue with fresh near-surface candidates
 *  ③ Run gradient-descent steps on the 7-param generative model (SDF + FE)
 *  ④ Write updated model parameters back to DSR (RT edge + geometry attrs)
 *  ⑤ Check convergence and set model_stable_att
 *  ⑥ Compute epistemic action proposals (viewpoint → mission-controller)
 *  ⑦ Detect divergence and set request_full_sample_att
 */

#include "specificworker.h"

#include "../../common/config_report/config_read.h"   // rc::cfg::Reader (SHARED)
#include "../../common/room_resolve/room_resolve.h"   // rc::room:: current_room / is_proto / is_proto_mirror

#include "../../common/detectability/detectability.h"   // rc::detect — the YOLO inverse model
#include "../../common/contour_edge/contour_edge_check.h"   // rc::edges — classifier-free check
#include "../../common/contour_edge/contour_depth_check.h"  // rc::edges — its metric half
#include "../../common/detect_probe/detect_probe.h"   // rc::probe — the detector's truth table (SHARED)

#include "../../common/diag_log/rotating_csv.h"   // keep the previous run instead of wiping it

#include "../../common/dashboard/belief_certainty.h"   // rc::dash::fill_certainty (SHARED)

#include "../../common/obj/convergence.h"   // rc::converge::step (SHARED)

#include "../../common/dashboard/belief_series.h"   // rc::dash::publish_belief_series (SHARED)

#include "../../common/birth_surprise/residual_field_reader.h"   // rc::read_residual_field (SHARED)

#include "../../common/peripheral_channel/peripheral_channel.h"   // THE shared ricoh path
#include "../../common/exclusion/exclusion.h"   // rc::exclusion — the SHARED no-two-objects rule
#include "../../common/exclusion/exclusion.h"   // rc::exclusion:: (SHARED)

#include "../../common/footprint/footprint.h"   // rc::geom:: Footprint / overlap_ratio (SHARED)
#include "../../common/track/merge_instances.h"   // rc::track::merge_overlapping — the SHARED merge sweep
#include "../../common/instance_tracker/birth_evidence.h"   // rc::birth:: the shared CREATE policy
#include "../../common/nbv/graph_obstacles.h"   // rc::nbv::collect_graph_obstacles — shared, DSR-side
#include "door_dof.h"   // rc::kDoorDofs — names/units for the BeliefInspector rows
#include "door_leaf_visibility.h"   // rc::door::crossing_logodds (Task D2)
#include "../../common/graph_provenance/creation_stamp.h"   // rc::provenance::now_ms — the recorded cycle's wall clock

#include <limits>   // numeric_limits<int>::max — the disabled tracker death counter
#include <QTimer>
#include <QSettings>   // persist the standalone dashboard window geometry
#include <QByteArray>
#include <QFont>
#include <QFontMetrics>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QComboBox>
#include <QVBoxLayout>
#include <QDateTime>   // wall-clock ms for the primary-input (masks) stream gate
#include <filesystem>
#include <print>
#include <cstdlib>   // std::_Exit — crash-free terminal shutdown
#include <thread>    // brief DDS flush before _Exit
#include <chrono>
#include <fstream>   // per-cycle detection-slice diagnostic CSV
#include <charconv>  // locale-independent parsing of etc/door_identities.csv (CLAUDE.md)
#include <locale>    // classic-locale writers
#include <format>
#include <iostream>  // std::cout/cerr flush

#include <algorithm>
#include <cmath>
#include <sstream>
#include <array>
#include <vector>
#include <unordered_set>

// DSR attribute name tags — generated from dsr_attr_name.h
#include <dsr/api/dsr_api.h>

namespace {

// Scalar model-uncertainty readout for model_uncertainty_att / the dashboard: the sum of the belief's
// per-DOF posterior stds from the AI2 covariance Σ over the wall-frame [s,w,h] (along-wall offset + panel
// width + height). Shrinks as the robot gathers viewpoints (mostly s, as the door is localised).
float belief_uncertainty(const rc::DoorInstance& inst)
{
    if (not inst.ai2_initialized)
        return 0.0f;
    const auto& S = inst.ai2_belief.covariance();
    const auto sd = [&](int i) { return std::sqrt(std::max(0.0f, S(i, i))); };
    return sd(0) + sd(1) + sd(2);
}

// (footprint_of / footprint_overlap_ratio / point_in_quad — the merge and birth geometry — moved with the tracker
// slice into the DoorCore: src/core/door_core.cpp.)

// Tracker lifecycle event log (etc/door_events.csv) — makes birth/merge/prune/suppress visible so the
// "create then remove" churn can be diagnosed from a file, not stdout. seq gives ordering.
rc::door::DoorDecisionLog* g_decisions = nullptr;   // the worker's decision log while enabled (set in initialize())

void log_tracker_event(const char* ev, std::uint64_t id, float x, float y, const std::string& note,
                       const std::string& name = {})
{
    // The same lifecycle events into the binary decision log (decision_log.h), with the position as raw doubles.
    if (g_decisions != nullptr and g_decisions->enabled())
    {
        using K = rc::door::DoorDecisionKind;
        static const std::unordered_map<std::string_view, K> kinds{
            {"BIRTH", K::Birth}, {"SUPPRESS", K::Suppress}, {"MERGE", K::Merge}, {"REMOVE", K::Remove},
            {"REACQUIRE", K::Reacquire}, {"RELEASE", K::Release}, {"RESTORE", K::Restore}};
        if (const auto k = kinds.find(ev); k != kinds.end())
            g_decisions->record(k->second, id, name, x, y, 0, 0, note);
    }
    static std::ofstream f = [] { std::ofstream o; rc::diag::open_rotating(o, "etc/door_events.csv");
                                   o << "seq,event,id,x,y,note\n"; return o; }();
    static int seq = 0;
    f << seq++ << ',' << ev << ',' << id << ',' << x << ',' << y << ',' << note << '\n';
    f.flush();
}

}  // namespace


// ─── Constructor / Destructor ─────────────────────────────────────────────────

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

    // Startup sanity check of the wall-frame [s,w,h] belief AND the aperture/leaf geometry (mirrors
    // bottle_concept). Non-fatal — but say so loudly rather than discarding the result, because a FAIL here
    // means every SDF, silhouette and projected ROI downstream is computed against the wrong panel.
    if (not rc::DoorBelief::self_test())
        std::print("door_concept: *** DoorBelief::self_test FAILED — panel geometry is NOT trustworthy ***\n");

    load_config(configLoader);

    // Registered rather than read raw, so the period appears in the startup table like every
    // other key. req() keeps the throw-if-missing behaviour: a period is not something to
    // default silently.
    int period = 0;
    rc::cfg::Reader(configLoader, "door_concept").req("Period.Compute", period,
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
    std::print("door_concept: SpecificWorker destroyed.\n");
}

void SpecificWorker::request_shutdown()
{
    if (shutting_down_.exchange(true))
        return;
    const auto self_scope = input_snapshot_.self_scope();

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

    // Flush the diagnostics' tails (up to 63 buffered rows each): exit is std::_Exit and this worker is never
    // deleted, so no ofstream destructor runs. Same thread as compute() (Qt timer), so no write can race this.
    anchor_frames_.close();
    g_decisions = nullptr;
    decision_log_.close();
    input_snapshot_.close();   // flush the input tape (records are whole on disk already; this closes the file)
    if (anchor_log_.is_open()) anchor_log_.close();

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

// Persist/restore the standalone dashboard window's geometry. The generated save_window_settings()
// only covers the QMainWindow(s) in `windows`; our extracted top-level widget is separate, so we
// carry its own QSettings entry (mirrors room_concept's RoomViewer).
void SpecificWorker::restore_dashboard_geometry()
{
    if (not dashboard_window_)
        return;
    QSettings settings(QStringLiteral("RoboComp"), QStringLiteral("door_concept"));
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
    QSettings settings(QStringLiteral("RoboComp"), QStringLiteral("door_concept"));
    settings.setValue(QStringLiteral("DashboardWindow_geometry"), dashboard_window_->saveGeometry());
    settings.sync();
}

// ─── Compact belief strip ────────────────────────────────────────────────────────────────────────

// One row per instance: the certainty channel (adequacy gap in nats, or ½·ln|Σ| when this agent
// publishes no σ*), p(existence), the FE surprise, and the node's birth stamp. The widget owns the
// history — this only pushes the current instant, on the same ~5 Hz tick as the panels above.
void SpecificWorker::refresh_belief_strip()
{
    // The loop, the REPORTED covariance, the creation-stamp birth time and update_view() are SHARED
    // (common/dashboard/belief_strip.h). What stays is what only this agent knows: where p_exists
    // comes from and which DOF table describes the belief.
    rc::dash::fill_strip(belief_strip_, fitter_->instances(), *G,
        [](rc::BeliefStripRow& r, const auto& inst)
        {
            if (inst.existence_seeded) r.p_exists = inst.existence.p_exists();
            rc::dash::fill_certainty(r, inst.ai2_belief.covariance_reported(), rc::kDoorDofs);
        });
}

void SpecificWorker::restore_strip_geometry()
{
    if (not strip_window_)
        return;
    QSettings settings(QStringLiteral("RoboComp"), QStringLiteral("door_concept"));
    const QByteArray geom = settings.value(QStringLiteral("BeliefStripWindow_geometry")).toByteArray();
    if (not geom.isEmpty())
        strip_window_->restoreGeometry(geom);
    else
        strip_window_->resize(520, 210);   // small ON PURPOSE — meant to sit in a corner, always open
}

void SpecificWorker::save_strip_geometry() const
{
    if (not strip_window_)
        return;
    QSettings settings(QStringLiteral("RoboComp"), QStringLiteral("door_concept"));
    settings.setValue(QStringLiteral("BeliefStripWindow_geometry"), strip_window_->saveGeometry());
    settings.sync();
}


// ─── Initialisation ──────────────────────────────────────────────────────────

void SpecificWorker::initialize()
{
    std::print("door_concept: initialize()\n");

    // Shadow-mode birth/death record (CONCEPT_AGENT_LIFECYCLE.md §4.2). Recording only — see
    // log_phantom_event(). Truncating: one file per run.
    phantom_log_.open("etc/door_phantom_events.csv");
    if (cfg_.decision_log)
    {
        if (decision_log_.open("etc/door_decisions.bin"))
            g_decisions = &decision_log_;
        else
            std::print("door_concept: could NOT open etc/door_decisions.bin — the decision log is off this run\n");
    }
    // Task 8 fix 3: the interaction anchor's measured cycles, truncated here (one file per run; replay_anchor_frames).
    // Final review I-4: a development instrument, DoorConcept.AnchorFrameRecord (default off).
    if (cfg_.anchor_frame_record and not anchor_frames_.open("etc/anchor_frames.bin"))
        std::print("door_concept: [anchor] could NOT open etc/anchor_frames.bin — the anchor's frame record is off this run\n");
    GenericWorker::initialize();
 
    // Ignore payload attributes in local graph updates to avoid unnecessary copying and processing of potentially large data
    G->set_ignored_attributes<cam_rgb_att, cam_depth_att, laser_X_att, laser_Y_att, laser_Z_att>();
    qInfo() << "Ignoring DSR RGBD payload attributes cam_rgb/cam_depth in local graph updates";


    if (not G)
    {
        qWarning() << "door_concept: DSR graph not available in initialize()";
        return;
    }
    // Input recording: latch from here (the startup sweep and the identity load below are its first events); the
    // file itself is opened at the end of initialize(), once the effective config has been published.
    if (cfg_.record_inputs)
        input_snapshot_.arm(static_cast<std::uint32_t>(agent_id));
    const auto self_scope = input_snapshot_.self_scope();   // initialize()'s deletes are ours

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
            .on_first_operating = [this] {
                const auto self_scope = input_snapshot_.self_scope();
                const auto before = input_snapshot_.armed() ? own_node_ids() : std::vector<std::uint64_t>{};
                remove_stale_affordance_nodes();
                // Anchors a SIGKILLed previous run left hanging from the robot (they may sync in after initialize()).
                if (G) rc::interaction::sweep_stale(*G, agent_id);
                if (input_snapshot_.armed())
                {
                    const auto after = own_node_ids();
                    std::vector<std::uint64_t> removed;
                    std::ranges::set_difference(before, after, std::back_inserter(removed));
                    input_snapshot_.on_owned_swept("first-Operating sweep", std::move(removed));
                }
            },
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
    scene_graph_ = std::make_unique<rc::DoorSceneGraph>(
        G, rt_api_.get(), cfg_, [this] { trigger_graph_layout_twopi(); });

    // Remove any "door*" nodes left behind by a previous (crashed) run so this agent always starts
    // from a clean slate and never adopts a stale/drifted node (the instance tracker re-births them
    // data-driven from masks). Runs BEFORE the graph signals are connected: delete_node() fires
    // del_node_signal synchronously, and del_node_slot dereferences fitter_ — which does not exist yet.
    // Connecting after this sweep (and after fitter_ is built) both avoids that null-deref SIGSEGV and
    // skips a pointless self-notification for our own cleanup deletions.
    const auto own_before_sweep = input_snapshot_.armed() ? own_node_ids() : std::vector<std::uint64_t>{};
    remove_owned_door_nodes();
    // …and any passage anchor (`door_N@robot`) of ours a crashed run left under the robot, with the links our
    // affordances still hold to it. Again once on the first Operating entry (post-sync), like the affordances.
    rc::interaction::sweep_stale(*G, agent_id);
    if (input_snapshot_.armed())
    {
        const auto after = own_node_ids();
        std::vector<std::uint64_t> removed;
        std::ranges::set_difference(own_before_sweep, after, std::back_inserter(removed));
        input_snapshot_.on_owned_swept("startup stale-sweep", std::move(removed));
    }

    // Resolve room node
    if (const auto room = rc::room::current_room(*G); room.has_value())
        room_node_id_ = *room;
    else
        qWarning() << "door_concept: no room node found at startup";

    // The DSR-free decision core (src/core/door_core.h). It reads ONLY the DoorCycleInput, so the snapshot builds one
    // every cycle from here on (recording stays behind DoorConcept.RecordInputs). Its decisions reach the DDEC log and
    // etc/door_events.csv through log_tracker_event — the same writer, at the same point in the cycle, as before.
    // ★BUILT BEFORE THE FITTER: the fit lives in the core (Task 4b), and the DoorFitter (the instance map + the
    // existence probes that still need cortex) builds its instances through it and shares its room polygon.
    input_snapshot_.enable_for_core(static_cast<std::uint32_t>(agent_id));
    core_ = std::make_unique<rc::door::core::DoorCore>(rc::door::core::DoorCoreConfig{cfg_});

    // The DSR-bound side of the fit. Owns the instance map; collaborates with the ingestor + scene graph.
    fitter_ = std::make_unique<rc::DoorFitter>(
        G, inner_eigen_.get(), cfg_, mask_ingestor_.get(), scene_graph_.get(), core_->fit());
    // (the existence probes' central-image box, Existence.CentralRegionFrac, is read by the DoorCore from the config)
    {
        rc::door::core::DoorCoreSinks sinks;
        sinks.decision = [](const rc::door::DoorDecision& d)
        {
            using K = rc::door::DoorDecisionKind;
            const auto k = static_cast<K>(d.kind);
            // The per-door leaf-angle row goes to the decision log only (it never had a door_events.csv line); the core
            // stamps it with the same cycle / t_ms the log's own clock carries.
            if (k == K::Phi or k == K::Existence)   // (the per-door existence row: no door_events.csv line either)
            {
                if (g_decisions != nullptr and g_decisions->enabled()) g_decisions->append(d);
                return;
            }
            const char* ev = k == K::Birth ? "BIRTH" : k == K::Suppress ? "SUPPRESS" : k == K::Merge ? "MERGE"
                           : k == K::Reacquire ? "REACQUIRE" : k == K::Remove ? "REMOVE" : k == K::Release ? "RELEASE"
                           : k == K::Restore ? "RESTORE" : "OTHER";
            log_tracker_event(ev, d.door_id, static_cast<float>(d.a), static_cast<float>(d.b), d.note, d.name);
        };
        sinks.diag = [ai2_path = cfg_.ai2_csv_path](const rc::door::core::DiagRow& r)
        {
            if (r.stream == "stdout")
                std::print("{}", r.line);
            else if (r.stream == "door_dets_log.csv")
            {
                // DIAGNOSTIC (merged-vs-single mask), one row per "door" slice per tracker call — rows formatted by
                // the core; the file truncated once per process launch, as before.
                static std::ofstream dcsv = []
                {
                    std::ofstream f; rc::diag::open_rotating(f, "etc/door_dets_log.csv");
                    f << "cycle,n_door_slices,slice_idx,npts,conf,cx,cy,range,trunc_frac,motion_var,"
                         // mask_h: slice height SPAN (m). size_w: candidate weight, UNAPPLIED.
                         // ★mask_z0 / mask_z1: the ABSOLUTE height band, which the span alone cannot give. It
                         // decides what a split door means: pieces at 0.0-0.92 and 0.9-2.2 TILE one panel and
                         // should be merged for association; two pieces both starting at 0.0 are competing
                         // claims about the same lower half, which would mean the segmenter can only ever see
                         // the bottom — and then downgrading them is right rather than merging them.
                         "mask_h,size_w,mask_z0,mask_z1\n";
                    return f;
                }();
                if (dcsv) { dcsv << r.line; dcsv.flush(); }
            }
            else if (r.stream == "door_existence_log.csv")
            {
                // The existence readout (throttled 1-in-60 + forced rows), rows formatted by the core; the file opened
                // (rotating) on the first row with its header, as the old function-static stream did.
                static std::ofstream ex_csv = []{ std::ofstream f; rc::diag::open_rotating(f, "etc/door_existence_log.csv");
                                                  f << rc::door::core::kExistenceCsvHeader; return f; }();
                if (ex_csv) { ex_csv << r.line; ex_csv.flush(); }
            }
            else if (r.stream == "ai2_log.csv" or r.stream == "ai2_log_phi.csv")
            {
                // The fit's two per-cycle logs (DoorModel.AI2CsvPath and its _phi twin): rows formatted by the core,
                // the files opened (rotating) on the first row with their header, as DoorFitter::log_*_csv did. A file
                // that cannot be opened stays closed and its rows are dropped (the old code cleared the path).
                struct Log { std::ofstream f; bool tried = false; };
                static Log ai2, phi;
                const bool is_phi = r.stream == "ai2_log_phi.csv";
                Log& L = is_phi ? phi : ai2;
                if (not L.tried)
                {
                    L.tried = true;
                    const std::string& base = ai2_path;
                    rc::diag::open_rotating(L.f, is_phi ? rc::door::core::phi_csv_path(base) : base);
                    if (L.f.is_open()) L.f << (is_phi ? rc::door::core::kPhiCsvHeader : rc::door::core::kAi2CsvHeader);
                }
                if (L.f.is_open()) { L.f << r.line; L.f.flush(); }
            }
        };
        // The input recorder's image ROIs: which frames the existence checks read this cycle and where (cropped at the
        // end of the cycle). The frames and their stamps are the ingestors' — the ones stage_images handed the core.
        sinks.image_use = [this](const rc::door::core::ImageUseNote& u)
        {
            if (not input_snapshot_.in_cycle()) return;
            input_snapshot_.note_images(u.rgb_used and rgb_ingestor_ ? &rgb_ingestor_->frame() : nullptr,
                                        rgb_ingestor_ ? rgb_ingestor_->stamp_ms() : 0, u.frame_ref,
                                        u.depth_used and depth_ingestor_ ? &depth_ingestor_->frame() : nullptr,
                                        depth_ingestor_ ? depth_ingestor_->stamp_ms() : 0,
                                        rgb_ingestor_ ? rgb_ingestor_->frame().cols : 0,
                                        rgb_ingestor_ ? rgb_ingestor_->frame().rows : 0);
        };
        sinks.image_roi = [this](const rc::door::core::ImageRoiNote& r)
        {
            if (not input_snapshot_.in_cycle()) return;
            if (r.poly.empty()) input_snapshot_.note_roi(r.point, r.extra_px, r.rgb, r.depth);
            else                input_snapshot_.note_roi(r.poly, r.extra_px, r.rgb, r.depth);
        };
        core_->connect(&core_world_, std::move(sinks));
    }

    // Subscribe to graph signals ONLY now that fitter_ (which every slot dereferences) exists, and after
    // the startup stale-sweep above. Cross-thread Auto resolves to Queued (never add DirectConnection).
    // ★NOT update_node_attr_signal. It fires for EVERY attribute change on EVERY node in the shared graph —
    // every RT pose write from robot_concept, every other agent's per-cycle attribute writes — and each
    // emission copies a std::vector<std::string> on the DDS reader thread and queues an event to this thread.
    // Under a churn burst (a peer restarting is enough) the queue drains slower than it fills, and the main
    // thread then does nothing but service slots: the timer-driven compute() is starved (its log stops
    // growing), and — the symptom that costs the most — Ctrl-C dies, because generated/main.cpp routes SIGINT
    // through a QSocketNotifier serviced by this SAME event loop. The agent can then only be killed with -9,
    // which cannot be caught, so every node it owns LEAKS into the shared graph.
    // Measured 2026-08-07 on table_concept (identical subscription): main thread pegged at 100% of a core,
    // ai2_log.csv frozen, Ctrl-C inert, right after a retina restart. CLAUDE.md already states the rule
    // this violated: if you don't need a signal, don't connect it at all (bottle_concept connects none).
    // The two things the slot did are now POLLED once per cycle in poll_affordance_protocol().
    connect(G.get(), &DSR::DSRGraph::update_node_signal,
            this, &SpecificWorker::modify_node_slot);
    connect(G.get(), &DSR::DSRGraph::del_node_signal,
            this, &SpecificWorker::del_node_slot);

    // (Part B, the localization/chain covariance on the published RT edge, is computed by the DoorCore's fit from the RT
    //  chain the input snapshot records with each mask packet — door_fit.h chain_point_cov — gated by RtCovAddChain.)

    // ★CREATED HERE, ON THE MAIN THREAD, AFTER THE GRAPH IS UP — the media-plane consumer rule. The
    // ingestor itself discovers the sensor node lazily inside pump() and stays dormant (no DDS
    // participant at all) while its precision gate is 0, so this costs nothing when switched off.
    lidar_ingestor_ = std::make_unique<rc::ConceptLidarIngestor>(G, inner_eigen_.get(),
        [this] { return rc::LidarGates{0.0f, 0.0f, cfg_.lidar_bpearl_precision}; });

    // Build rc::EpistemicPlanner (Σ-based D-optimal NBV) with the configured stand-off.
    epistemic_planner_ = rc::EpistemicPlanner(cfg_.obs_distance);
    // ONE detector envelope: the viewpoint the planner asks for is the argmax of the same model absence
    // should be weighted by. Replaces cfg_.min_standoff_m (DoorConcept.MinStandOffM), a hand-picked
    // stand-in for the near shoulder of exactly this curve.
    epistemic_planner_.set_detector_envelope(
        rc::detect::DetectorEnvelope{cfg_.detect_min_fill, cfg_.detect_max_fill, cfg_.detect_soft});
    epistemic_planner_.set_robot_radius(0.30f);   // Shadow's footprint radius

    // ZED RGB consumer for the contour check. Lazily brought up inside pump() once the descriptor exists
    // (media-plane consumer pattern); dormant while DoorConcept.RgbContourCheck is false.
    rgb_ingestor_ = std::make_unique<rc::RgbIngestor>(G, &cfg_.rgb_contour_check, "zed");
    depth_ingestor_ = std::make_unique<rc::DepthIngestor>(G, &cfg_.rgb_contour_check, "zed");

    // DoorControl client. The endpoint is config, not compiled in: the provider is whatever can move a
    // door here — a Webots supervisor today, home automation or a person tomorrow.
    door_actuator_.configure(cfg_.door_control_endpoint);
    // Input recording (spec §6.6): the identity table + learned registration exactly as loaded, and the room entered.
    input_snapshot_.on_identity_load("etc/door_identities.csv", "etc/door_world_registration.csv",
        room_node_id_ != 0 ? std::optional<std::uint64_t>{room_node_id_} : std::nullopt,
        room_node_id_ != 0 ? rc::room::name_of(*G, room_node_id_) : std::string{});
    door_registration_.load("etc/door_world_registration.csv");   // solved per request, per room

    // The camera's REAL geometry, read once (intrinsics and the zed mount are both static). BOTH FoVs: a door
    // is ~2 m tall, so the VERTICAL axis binds well before the horizontal one, and the horizontal-only model
    // this replaced was blind to it. Height from inner_eigen room->zed, NOT a body-relative constant: the room
    // floor datum is offset from the body origin and the z-span is room-frame. ts==0 on the main thread.
    // ★The camera model is read PER CYCLE at the compute site (rc::nbv::sensor_from_graph),
    // NOT once here: the zed intrinsics are published by robot_concept when frames start
    // arriving, so reading them in initialize() races the producer. Losing that race leaves
    // vfov = 0, which silently collapses the fill model to horizontal-only — the exact bug
    // rc::nbv exists to fix, and it drives the robot nose-to-nose with tall objects.

    // Stale affordance nodes are swept on entering Operating (presence hook) and on shutdown — see
    // remove_stale_affordance_nodes(), keyed on the parent object type (robust to node-name renames).

    // Which door is which — remembered from earlier runs. Without this the numbering falls out of the order
    // the robot happens to sweep the room, so the same door was door_1 one run and door_2 the next.
    load_identities();
    if (room_node_id_ != 0)
        enter_room_memory(room_node_id_, true);   // the first room is a fresh entry: older runs' poses can't come back

    // ── Time-series dashboard — its OWN top-level window ──────────────────────
    // Extracted from the DSR graph dock (add_custom_widget_to_dock) into a standalone window so it shows
    // even with Agent.graph=false (this agent runs graph=false → no DSRViewer). Mirrors room_concept,
    // kinova_controller, table_concept and bottle_concept. TimeSeriesPlot is a plain QWidget (no QOpenGL
    // backing store), safe as a top-level. NOT WA_DeleteOnClose: closing must only HIDE it, or the
    // ts_*_plot_ pointers the compute() feed uses would dangle. A QApplication always exists (generated/main.cpp).
    {
        custom_widget_ = new Custom_widget("Door — Free Energy, Surprise, Belief Uncertainty & Residuals");

        // Create plot inside frame_series
        auto* series_layout = new QVBoxLayout(custom_widget_->frame_series);
        series_layout->setContentsMargins(0, 0, 0, 0);
        custom_widget_->frame_series->setLayout(series_layout);

        ts_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
        ts_plot_->set_visible_window(60.f);
        series_layout->addWidget(ts_plot_, 1);

        // FE SURPRISE (attention signal) on its own panel — it lives on a much smaller scale (~0–1) than
        // the FE, so it needs the full panel height to be readable. Spikes when a door moves, decays as
        // the fit re-converges.
        ts_surprise_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
        ts_surprise_plot_->set_visible_window(60.f);
        series_layout->addWidget(ts_surprise_plot_, 1);

        ts_cov_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
        ts_cov_plot_->set_visible_window(60.f);
        series_layout->addWidget(ts_cov_plot_, 1);

        ts_res_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
        ts_res_plot_->set_visible_window(60.f);
        series_layout->addWidget(ts_res_plot_, 1);


        // BELIEF INSPECTOR — the panel that replaced the pose-σ trace. A time-series of two variances showed
        // a slice of the belief; this shows ALL of it: every state DOF with its posterior σ, Σ as a
        // correlation heatmap (where the structure lives), and the 4-mode yaw posterior. Stretch 2.
        belief_inspector_ = new rc::BeliefInspector(QStringLiteral("belief inspector"),
                                                    custom_widget_->frame_series);
        series_layout->addWidget(belief_inspector_, 2);

        // GenericWorker::initialize() may have already started compute(), so
        // some instances can exist before the plots are constructed.
        for (const auto& [_, inst] : fitter_->instances())
        {
            ts_plot_->add_series(inst.node_name + "_fe", QColor(255, 170, 0), 1.1f);
            ts_cov_plot_->add_series(inst.node_name + "_cov", QColor(0, 190, 255), 1.1f);
            ts_res_plot_->add_series(inst.node_name + "_res", QColor(170, 80, 255), 1.1f);
        }
    }

    // ── Section 1: evidence-pipeline counter strip ────────────────────────────
    evidence_monitor_ = new rc::EvidenceMonitor(QStringLiteral("door_concept — evidence monitor"));

    // ── Combined window: counters (top) over the plots + belief inspector (bottom) ──
    // Identical three-section structure to every other concept agent. Only HIDDEN on close, never deleted
    // (compute() keeps the raw child pointers).
    dashboard_window_ = new QWidget;
    dashboard_window_->setWindowTitle(QStringLiteral("door_concept — dashboard"));
    auto* outer = new QVBoxLayout(dashboard_window_);
    outer->setContentsMargins(0, 0, 0, 0);
    // No splitter: the counter strip is two lines of text with nothing to resize, so it simply takes
    // its natural height and the plots + inspector get everything else.
    outer->addWidget(evidence_monitor_, 0);   // section 1 — natural height
    outer->addWidget(custom_widget_, 1);      // sections 2 + 3 — all remaining space

    // NOT shown at startup: the strip below is the standing display and this is its drill-down,
    // opened by the strip's "details ▸" button. Geometry is still restored, so the first click
    // puts it back where you left it. (Want it up from the start? add `->show()`.)
    restore_dashboard_geometry();

    // ── Compact belief strip — a SEPARATE, small top-level window ──────────────────────────────────
    // Not another panel inside the big window: the point is a display small enough to keep in a corner
    // while the 1180×900 dashboard stays closed until something looks wrong. One row per instance, each
    // row a 60 s trace of the certainty channel + p(existence) + FE surprise.
    strip_window_ = new QWidget;
    strip_window_->setWindowTitle(QStringLiteral("doors — beliefs"));
    auto* strip_layout = new QVBoxLayout(strip_window_);
    strip_layout->setContentsMargins(0, 0, 0, 0);
    belief_strip_ = new rc::BeliefStrip(QStringLiteral("doors"), strip_window_);
    belief_strip_->set_visible_window(60.f);
    strip_layout->addWidget(belief_strip_, 1);

    // "details ▸" — reveal the big dashboard on demand. A lambda connect needs no Q_OBJECT on either
    // side, so the moc-free widget pattern is preserved; strip_window_ is the context object, so the
    // connection dies with the window. show() alone is not enough for a minimised or buried window.
    {
        auto* bar = new QHBoxLayout;
        bar->setContentsMargins(4, 0, 4, 3);
        bar->addStretch(1);
        auto* details = new QPushButton(QStringLiteral("details ▸"), strip_window_);
        details->setToolTip(QStringLiteral("show / hide the full dashboard: evidence counters, FE/surprise/Σ "
                                           "time series, and the per-DOF belief inspector"));
        QFont bf = details->font(); bf.setPointSizeF(bf.pointSizeF() - 1.0); details->setFont(bf);
        details->setFixedHeight(QFontMetrics(bf).height() + 8);
        QObject::connect(details, &QPushButton::clicked, strip_window_, [this, details]()
        {
            if (not dashboard_window_) return;
            // TOGGLE: the button the dashboard came out of is the button it goes back into. A drill-down
            // that can only be OPENED is one that stays open — the 1180×900 window sits on top of
            // everything and the only way back to a clear screen is to hunt it down in the window list and
            // minimise it by hand (which is exactly where it was found, WM_STATE=Iconic).
            // ★A MINIMISED window counts as PUT AWAY, not as up: otherwise the first click after
            // minimising would "hide" an invisible window and it would take two clicks to see it again.
            const bool up = dashboard_window_->isVisible() and not dashboard_window_->isMinimized();
            if (up)
                dashboard_window_->hide();
            else
            {
                dashboard_window_->show();
                dashboard_window_->setWindowState((dashboard_window_->windowState() & ~Qt::WindowMinimized)
                                                  | Qt::WindowActive);
                dashboard_window_->raise();
                dashboard_window_->activateWindow();
            }
            // The label says what the NEXT click does. It can go stale if the window is closed from its
            // own title bar; the state is re-read above on every click, so the behaviour never is.
            details->setText(up ? QStringLiteral("details ▸") : QStringLiteral("◂ hide"));
        });
        bar->addWidget(details, 0);

        // ── ASK SOMEBODY TO OPEN / CLOSE A DOOR (RoboCompDoorControl) ──────────────────────────────
        // ★NOT "drive the simulator". The provider may be a Webots supervisor, home automation, or a
        // person asked over TTS — the robot asks for a door to be moved and does not know or care who
        // does it. On the real robot there is no Webots, so the dependency has to be a CAPABILITY.
        //
        // ★The button acts on the NEAREST believed door, and its label says which one. Acting on "the
        // door" when several are alive would silently pick one; naming it makes a wrong pick visible
        // before the request goes out rather than after something opens across the flat.
        //
        // ★Enabled state follows getCapabilities(), NOT peer presence. The Webots provider replies
        // canActuate=false whenever its [DoorControl] Doors list is absent — the honest answer for a
        // world whose doors are scenery — and a button that is enabled because a peer exists and then
        // always fails teaches the operator to distrust the button. The tooltip carries the reason.
        // ★WHICH PROVIDER DOOR. The interface offers two resolution paths: quote an advertised id, or
        // let the provider match by PLACE. Place-matching needs our frame and the provider's world frame
        // registered to each other, and here they are NOT — the room frame is room-local, so door_3 goes
        // out at (-373, 4440) mm against DOOR_0 at (-3583, 7166) mm, 4 m apart, and the provider
        // correctly answers UnknownDoor. That is a real gap and this combo does not close it; it routes
        // around it for the case where a human is pressing the button and knows which door they mean.
        // "(by place)" stays as the first entry so the registration path can be exercised deliberately.
        auto* door_pick = new QComboBox(strip_window_);
        door_pick->setToolTip(QStringLiteral("which door the PROVIDER should move: an advertised id, or "
                                             "'(by place)' to let it match on the pose we send"));
        door_pick->addItem(QStringLiteral("(by place)"));
        door_actuation_pick_ = door_pick;

        auto* open_btn  = new QPushButton(QStringLiteral("open"),  strip_window_);
        auto* close_btn = new QPushButton(QStringLiteral("close"), strip_window_);
        auto* act_state = new QLabel(QStringLiteral("—"), strip_window_);
        auto* phi_label = new QLabel(QStringLiteral("φ —"), strip_window_);
        phi_label->setToolTip(QStringLiteral("leaf opening angle: 0° = flush in the aperture, 90° = fully "
                                             "open. 'pinned' means it is a MODEL CONSTANT, not something "
                                             "measured — M0 does not fit phi, so the silhouette always "
                                             "predicts a shut door."));
        { QFont f = phi_label->font(); f.setPointSizeF(f.pointSizeF() - 1.0); phi_label->setFont(f); }
        door_phi_label_ = phi_label;
        for (auto* b : {open_btn, close_btn})
        {
            QFont f = b->font(); f.setPointSizeF(f.pointSizeF() - 1.0); b->setFont(f);
            b->setFixedHeight(QFontMetrics(f).height() + 8);
            b->setEnabled(false);
        }
        { QFont f = act_state->font(); f.setPointSizeF(f.pointSizeF() - 1.0); act_state->setFont(f); }
        act_state->setToolTip(QStringLiteral("state of the last door request"));
        door_act_open_btn_ = open_btn; door_act_close_btn_ = close_btn; door_act_state_ = act_state;

        // ★THE BUTTON AND THE `open` AFFORDANCE SHARE ONE REQUEST PATH (request_door_actuation). They
        // used to be one path and one lambda because there was only the button; the moment an affordance
        // could also ask, the frame resolution, the by-place refusal, the registration donation and the
        // phi anticipation all had to exist exactly once, or the autonomous path would quietly differ
        // from the one anybody had ever tested.
        const auto fire = [this](bool open)
        {
            const auto self_scope = input_snapshot_.self_scope();
            // Input recording: the press itself is the event (the core redoes the target resolution next cycle).
            if (input_snapshot_.armed())
                input_snapshot_.on_ui_request(open, door_actuation_pick_ and door_actuation_pick_->currentIndex() > 0
                                                        ? door_actuation_pick_->currentText().toStdString() : std::string{});
            const auto target = nearest_door_for_actuation();
            if (not target.has_value())
            {
                if (door_act_state_) door_act_state_->setText(QStringLiteral("no door believed"));
                return;
            }
            rc::DoorInstance* inst = nullptr;
            for (auto& [id, candidate] : fitter_->instances())
                if (candidate.node_name == target->name)
                { inst = &candidate; break; }
            if (inst == nullptr)
            {
                if (door_act_state_) door_act_state_->setText(QStringLiteral("door vanished"));
                return;
            }
            std::string provider_id;
            if (door_actuation_pick_ and door_actuation_pick_->currentIndex() > 0)
                provider_id = door_actuation_pick_->currentText().toStdString();

            std::string why;
            const bool sent = request_door_actuation(*inst, open, provider_id,
                                                      open ? "robot needs to pass through"
                                                           : "robot has passed through", why);
            if (door_act_state_)
                door_act_state_->setText(sent and door_pending_.has_value()
                    ? QString::fromStdString(inst->node_name + ": " + door_pending_->state)
                    : QString::fromStdString(why.empty() ? "no provider" : why));
        };
        QObject::connect(open_btn,  &QPushButton::clicked, strip_window_, [fire]{ fire(true);  });
        QObject::connect(close_btn, &QPushButton::clicked, strip_window_, [fire]{ fire(false); });

        bar->addWidget(door_pick, 0);
        bar->addWidget(open_btn, 0);
        bar->addWidget(close_btn, 0);
        bar->addWidget(phi_label, 0);
        bar->addWidget(act_state, 0);
        strip_layout->addLayout(bar, 0);
    }

    restore_strip_geometry();
    strip_window_->show();

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
    // ARMS the unread sweep; check_registry_complete.sh door_concept is the grep that keeps the claim
    // honest. Re-run it whenever a config read is added.
    rc::cfg::exempt_generated_prefixes();
    rc::cfg::registry().declare_complete("door_concept");
    rc::cfg::Reader(configLoader, "door_concept").publish("etc/config_effective.csv");
    open_input_recording();
}

// DoorConcept.RecordInputs: open the DCIN tape (rotating the previous run's) with this run's effective config as the
// header's configuration snapshot (spec §3.1 "configuration snapshot (once per file)").
void SpecificWorker::open_input_recording()
{
    if (not input_snapshot_.armed())
        return;
    rc::door::core::DoorRecordingHeader h;
    h.agent_id = static_cast<std::uint64_t>(agent_id);
    h.agent_name = "door_concept";
    {
        std::ifstream f("etc/config_effective.csv", std::ios::binary);
        std::ostringstream ss;
        if (f) ss << f.rdbuf();
        h.config_text = ss.str();
    }
    h.start_steady_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch()).count();
    h.start_wall_ms = static_cast<std::int64_t>(rc::provenance::now_ms());
    h.note = "door_concept live recording; config_text = etc/config_effective.csv";
    const auto max_bytes = static_cast<std::uint64_t>(std::max(0, cfg_.record_max_mb)) << 20;
    if (input_snapshot_.open(cfg_.record_inputs_path, h, max_bytes))
        std::print("door_concept: [record] recording every cycle's inputs to {} (cap {} MB)\n",
                   cfg_.record_inputs_path, cfg_.record_max_mb);
    else
        std::print("door_concept: [record] could NOT open {} — input recording is off this run\n", cfg_.record_inputs_path);
}

// Our own object/affordance node ids (sorted), for the sweep diffs the recording logs as OwnedNodesSwept.
std::vector<std::uint64_t> SpecificWorker::own_node_ids() const
{
    std::vector<std::uint64_t> ids;
    if (not G) return ids;
    for (const char* type : {"object", "affordance"})
        for (const auto& n : G->get_nodes_by_type(type))
            if (rc::door::core::created_by(n.id(), static_cast<std::uint32_t>(agent_id)))
                ids.push_back(n.id());
    std::ranges::sort(ids);
    return ids;
}

// ─── The ActuatorPort (ruling: an injected SYNCHRONOUS port) ─────────────────────────────────────────────────────
// Every DoorActuator call goes through one of these three, so each call and its reply land in the recorded input's
// transcript (in a cycle → that cycle; from a strip-button callback → the next). Behaviour is the bare call's.
namespace
{
rc::door::core::ActuatorPending to_core(const rc::DoorActuator::Pending& p)
{ return {p.id, p.door_name, p.opening, p.state, p.settled}; }

rc::door::core::ActuatorCaps to_core(const rc::DoorActuator::Caps& c)
{
    rc::door::core::ActuatorCaps o;
    o.available = c.available; o.continuous_angle = c.continuous_angle; o.requires_human = c.requires_human;
    o.typical_latency_s = c.typical_latency_s; o.n_doors = c.n_doors; o.note = c.note;
    for (const auto& d : c.doors) o.doors.push_back({d.id, d.xy_mm, d.angle});
    return o;
}
}  // namespace

const rc::DoorActuator::Caps& SpecificWorker::actuator_refresh_caps()
{
    const auto& c = door_actuator_.refresh_caps();
    if (input_snapshot_.armed())
    {
        rc::door::core::ActuatorCall call;
        call.kind = rc::door::core::ActuatorCallKind::RefreshCaps;
        call.caps = to_core(c);
        input_snapshot_.on_actuator_call(std::move(call));
    }
    return c;
}

std::optional<rc::DoorActuator::Pending> SpecificWorker::actuator_request(const std::string& door_name,
    const Eigen::Vector2f& xy_m, float yaw, float width_m, bool open, const Eigen::Matrix4d& world_T_room,
    const std::string& purpose, const std::string& provider_id)
{
    auto reply = door_actuator_.request(door_name, xy_m, yaw, width_m, open, world_T_room, purpose, provider_id);
    if (input_snapshot_.armed())
    {
        rc::door::core::ActuatorCall call;
        call.kind = rc::door::core::ActuatorCallKind::Request;
        call.door_name = door_name; call.xy_m = xy_m; call.yaw = yaw; call.width_m = width_m; call.open = open;
        call.world_T_room = world_T_room; call.purpose = purpose; call.provider_id = provider_id;
        if (reply.has_value()) call.pending = to_core(reply.value());
        input_snapshot_.on_actuator_call(std::move(call));
    }
    return reply;
}

void SpecificWorker::actuator_poll(rc::DoorActuator::Pending& p)
{
    std::optional<rc::door::core::ActuatorPending> in;
    if (input_snapshot_.armed()) in = to_core(p);
    door_actuator_.poll(p);
    if (in.has_value())
    {
        rc::door::core::ActuatorCall call;
        call.kind = rc::door::core::ActuatorCallKind::Poll;
        call.poll_in = std::move(in);
        call.pending = to_core(p);
        input_snapshot_.on_actuator_call(std::move(call));
    }
}

// ─── Door actuation ──────────────────────────────────────────────────────────
//
// Which door a button press means. NEAREST TO THE ROBOT, and its name is put on the button, because
// "the door" is ambiguous the moment two are alive and a silent pick can open one across the flat.
std::optional<SpecificWorker::ActuationTarget> SpecificWorker::nearest_door_for_actuation() const
{
    if (not fitter_ or not inner_eigen_)
        return std::nullopt;
    const auto rtb = inner_eigen_->get_transformation_matrix(rc::room::current_room_frame(*G), "body", 0);
    if (not rtb.has_value())
        return std::nullopt;
    const Eigen::Vector2f robot(static_cast<float>(rtb.value()(0, 3)),
                                static_cast<float>(rtb.value()(1, 3)));
    std::optional<ActuationTarget> best;
    for (const auto& [id, inst] : fitter_->instances())
    {
        if (inst.is_bearing_hypothesis)
            continue;   // a proto-object is a place to go and look, not a door to ask about
        const auto& st = inst.model.state();
        ActuationTarget t;
        t.name    = inst.node_name;
        // Task 11 fix round 1: the APERTURE, not the leaf — the leaf's centre jumps by up to a width when the hinge
        // flips (and moves w/2 with every swing); "which door is nearest" must not.
        t.xy      = Eigen::Vector2f(st.ap_cx, st.ap_cy);
        t.yaw     = st.ap_yaw;
        t.width_m = st.w;
        t.range_m = (t.xy - robot).norm();
        // ★THE LEAF ANGLE, and whether it is a MEASUREMENT. In M0 phi is PINNED at 0 — it is not fitted
        // from anything, so a UI reading "0.0°" would be an assertion the robot has never checked. That
        // pin is also why a door that really opens reads as absent: the silhouette keeps predicting a
        // flush leaf. Showing the state and its provenance together is the difference between "the door
        // is shut" and "we have assumed the door is shut".
        t.phi = st.phi;                 // DoorState::phi — the belief's leaf angle (LeafPose has no phi)
        t.phi_fitted = true;    // M1: estimated per cycle from silhouette support (DoorFitter::estimate_phi)
        if (not best.has_value() or t.range_m < best->range_m)
            best = std::move(t);
    }
    return best;
}

// Keep the buttons honest: enabled only when the provider says it can actually move a door, labelled
// with the door they would act on, and carrying the provider's own reason in the tooltip when they are
// not. Also polls any outstanding request. Called from compute(), cheap.
void SpecificWorker::refresh_door_actuation_ui()
{
    if (not door_act_open_btn_)
        return;
    // Capabilities are re-read occasionally, not per cycle: it is a remote call, and a provider that
    // comes up later must still be picked up without restarting this agent.
    static int tick = 0;
    if (++tick % 120 == 0 or door_actuator_.caps().note == "not connected yet")
        actuator_refresh_caps();
    const auto& c = door_actuator_.caps();

    // Keep the picker in step with what the provider advertises, without disturbing a live selection.
    if (door_actuation_pick_ and door_actuation_pick_->count() != static_cast<int>(c.doors.size()) + 1)
    {
        const QString keep = door_actuation_pick_->currentText();
        door_actuation_pick_->clear();
        door_actuation_pick_->addItem(QStringLiteral("(by place)"));
        for (const auto& d : c.doors)
            door_actuation_pick_->addItem(QString::fromStdString(d.id),
                                          QPointF(d.xy_mm.x(), d.xy_mm.y()));
        const int back = door_actuation_pick_->findText(keep);
        door_actuation_pick_->setCurrentIndex(back >= 0 ? back : 0);
    }

    const auto target = nearest_door_for_actuation();
    const bool on = c.available and target.has_value();
    door_act_open_btn_->setEnabled(on);
    door_act_close_btn_->setEnabled(on);
    if (door_phi_label_)
        door_phi_label_->setText(target.has_value()
            ? QString::fromStdString(std::format("φ {:.1f}° {}", target->phi * 180.0f / static_cast<float>(M_PI),
                                                 target->phi_fitted ? "(fitted)" : "(pinned)"))
            : QStringLiteral("φ —"));

    const QString who = target.has_value()
        ? QString::fromStdString(std::format("{} ({:.1f} m)", target->name, target->range_m))
        : QStringLiteral("no door believed");
    const QString tip = QString::fromStdString(c.note) + "\n" + who;
    door_act_open_btn_->setToolTip(QStringLiteral("ask a provider to OPEN — ") + tip);
    door_act_close_btn_->setToolTip(QStringLiteral("ask a provider to CLOSE — ") + tip);
    door_act_open_btn_->setText(target.has_value()
        ? QString::fromStdString("open " + target->name) : QStringLiteral("open"));
    door_act_close_btn_->setText(target.has_value()
        ? QString::fromStdString("close " + target->name) : QStringLiteral("close"));

    if (door_pending_.has_value() and not door_pending_->settled)
    {
        actuator_poll(*door_pending_);
        if (door_act_state_)
            door_act_state_->setText(QString::fromStdString(door_pending_->door_name + ": "
                                                            + door_pending_->state));
    }
    else if (door_act_state_ and not door_pending_.has_value())
        door_act_state_->setText(QString::fromStdString(c.available ? "ready" : c.note));
}

// ─── Belief inspector ────────────────────────────────────────────────────────

// Build the per-instance BELIEF snapshot (pose state, Σ, the 4-mode yaw posterior, scalar gauges) and push
// it to the bottom panel, throttled to ~5 Hz (a full card rebuild every compute cycle would waste the GUI
// thread). The door has no EvidenceMonitor to share a tick with, so the gate lives here. Main-thread.
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
    refresh_belief_strip();   // same tick, same instance pass — the views can never disagree
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

        // Σ over the wall-frame [s,w,h]: along-wall offset + panel width + panel height.
        const auto  S = inst.ai2_belief.covariance_reported();
        const auto& s = inst.ai2_belief.state();
        const std::array<float, rc::DoorBelief::kReported> v = {s.s, s.w, s.h};
        for (int j = 0; j < rc::DoorBelief::kReported; ++j)
            c.dofs.push_back({rc::kDoorDofs[j].name, rc::kDoorDofs[j].unit, v[j],
                              std::sqrt(std::max(0.0f, S(j, j))), rc::kDoorDofs[j].sigma_star});

        // Row-major copy, filled explicitly: Eigen stores column-major, and while Σ is symmetric today an
        // implicit .data() copy would silently transpose if that ever stopped being true.
        constexpr int N = rc::DoorBelief::kReported;   // the reported [s, w, h], as S
        c.cov.resize(N * N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                c.cov[i * N + j] = S(i, j);

        // ── THE HINGE, AS ITS OWN ROW ────────────────────────────────────────────────────────────
        // phi is not in Σ — it is the articulated coordinate, minimised in its own branch with the
        // aperture held — so it gets a DOF row with sigma 0 and no σ*, which is the honest rendering of
        // "estimated, but we do not yet publish a covariance for it". ⚠That zero is a PLACEHOLDER, not a
        // claim of certainty; when the hinge branch grows a posterior width this is where it goes.
        c.dofs.push_back({"phi", "rad", inst.phi_est, 0.0f, -1.0f});

        // ── WHICH CHANNEL CHOSE THE ANGLE, as a footer bar group ─────────────────────────────────
        // ★THE NUMBER THAT ACTUALLY DIAGNOSES THIS AGENT NOW. The leaf angle is decided by a mixture of
        // three likelihoods — LiDAR surface (SDF), semantic mask overlap, RGB contour — and which of them
        // is deciding is invisible from the angle alone. It is exactly what was needed and missing
        // tonight: a phi of 17.8 deg on a door standing open reads identically whether the SDF found the
        // leaf and was outvoted, or was silent and the mask walked the estimate back to flush. Those two
        // want opposite fixes, and one of them (a summed free energy making the SDF curve a delta, so its
        // weight was either ~0.9 or exactly 0) took a CSV column to see at all.
        // Drawn as a discrete posterior because that is what it is: the normalised contribution each
        // channel made to the winning hypothesis. All three at 0 ⇒ nothing measured the angle this cycle.
        {
            const float wsum = inst.dbg_w_sdf + inst.dbg_w_mask + inst.dbg_w_edge;
            if (wsum > 1e-9f)
            {
                c.modes.push_back({"phi evidence", "lidar", inst.dbg_w_sdf  / wsum});
                c.modes.push_back({"phi evidence", "mask",  inst.dbg_w_mask / wsum});
                c.modes.push_back({"phi evidence", "edge",  inst.dbg_w_edge / wsum});
            }
        }

        // No discrete orientation ambiguity: the containing wall fixes the door's yaw.

        // ★`model` is the card's free-text tag; the hinge branch's evidence COUNT goes here because a
        // free energy computed over zero points is not a small free energy, it is not a measurement —
        // and the gauge strip has no way to say that on its own.
        c.model = inst.dbg_leaf_pts > 0
                ? std::format("hinge: {} lidar pts", inst.dbg_leaf_pts)
                : std::string("hinge: NO lidar pts");
        c.s.fe          = inst.dbg_energy;
        c.s.fe_baseline = inst.fe_baseline;
        c.s.fe_surprise = inst.fe_surprise;
        // Un-seeded until the existence channel's first visit; leave the card's NaN so it prints "-" rather
        // than a fake 0.5 probability.
        if (inst.existence_seeded)
        {
            c.s.logodds  = inst.existence.logodds();
            c.s.p_exists = inst.existence.p_exists();
        }
        c.s.remove_streak = static_cast<int>(inst.existence_debounce.streak);
        c.s.age_s       = inst.last_belief_touch.time_since_epoch().count() == 0
                        ? -1.0f
                        : std::chrono::duration<float>(now - inst.last_belief_touch).count();
        c.s.since_det   = inst.frames_since_detection;
        c.s.initialized = inst.ai2_initialized;
        cards.push_back(std::move(c));
    }
    belief_inspector_->update_view(cards);
}

// ─── Detector truth table (rc::probe) ────────────────────────────────────────────────────────────
//
// One row per cycle per live door: WHERE the robot stood, HOW the door projected, and what the
// detector did. door had no detectability block at all, which is why the question "is this door dying
// because the detector cannot see it, or because absence is over-charged" had no data behind it.
//
// ★WHY THE POSE IS THE POINT. Absence is integrated as if every frame were an independent trial. It is
// not: measured in retina on 2026-09-08, while the robot is stopped only 5.1% of frames carry a NEW
// image (99.6% while moving), so a parked robot does not merely repeat a viewpoint — it re-integrates
// the SAME PIXELS thousands of times. Without (rx,ry,rtheta) that cannot be seen in the file, and any
// envelope fitted over these rows without de-duplicating by viewpoint will be confident about copies.
void SpecificWorker::log_detect_probe()
{
    if (cfg_.detect_probe_csv_path.empty() or not fitter_ or not mask_ingestor_)
        return;
    if (not rc::probe::ensure_open(detect_probe_csv_, cfg_.detect_probe_csv_path))
        return;

    // Pose is REQUIRED, not optional-with-a-fallback: a row whose viewpoint silently defaulted to the
    // origin would cluster with every other failed lookup and invent a viewpoint that never existed.
    if (not inner_eigen_)
        return;
    const auto rtb = inner_eigen_->get_transformation_matrix(rc::room::current_room_frame(*G), "body", 0);
    if (not rtb.has_value())
        return;
    const auto& Tb = rtb.value();
    const float rx = static_cast<float>(Tb(0, 3)), ry = static_cast<float>(Tb(1, 3));
    const float rtheta = static_cast<float>(std::atan2(Tb(1, 0), Tb(0, 0)));
    float cam_z = static_cast<float>(Tb(2, 3));
    if (const auto rtz = inner_eigen_->get_transformation_matrix(rc::room::current_room_frame(*G), "zed", 0); rtz.has_value())
        cam_z = static_cast<float>(rtz.value()(2, 3));

    const auto& pkt = mask_ingestor_->packet();
    int n_all = 0, n_my = 0;
    if (pkt.valid)
        for (const auto& sl : pkt.slices)
        {
            if (not sl.is_zed()) continue;   // the ZED is the detector this file is about
            ++n_all;
            if (sl.label == "door") ++n_my;
        }

    for (const auto& [id, inst] : fitter_->instances())
    {
        const auto& st = inst.model.state();
        rc::probe::Sample row;
        row.cycle    = inst.processed_cycles;
        row.node     = inst.node_name;
        row.stamp_ms = inst.last_mask_timestamp_ms;
        row.rx = rx; row.ry = ry; row.rtheta = rtheta; row.cam_z = cam_z;
        row.ocx = st.cx; row.ocy = st.cy; row.ocz = st.cz;
        row.range_m       = inst.last_range;
        row.roi_fill      = inst.roi_fill;
        row.roi_fill_h    = inst.roi_fill_h;
        row.roi_fill_v    = inst.roi_fill_v;
        row.roi_valid     = inst.roi_valid;
        row.obliquity_cos = inst.dbg_obliquity_cos;
        row.trunc_frac    = inst.last_trunc_frac;
        // door's p_detect is the REAL thing — it has a silhouette channel, so this is the number the
        // removal actually used, not a stand-in.
        row.p_detect      = inst.dbg_sil_pdetect;
        row.p_exists_prior = inst.existence.p_exists();
        // ★THE LABEL IS PER-CYCLE, and it is not detection_alive. That flag LATCHES (measured on hood:
        // 1 in 9432/9432 rows, still 1 at frames_since_det=6172), so as an outcome it is degenerate and
        // fit_envelope correctly refuses it. "A mask arrived THIS cycle" is the trial.
        row.fired = (inst.frames_since_detection == 0);
        row.conf  = row.fired ? inst.last_mask_confidence : 0.0f;
        row.n_my = n_my; row.n_all = n_all;
        // What ELSE the detector called the thing at this place. A door named `wall` is the detector
        // AGREEING that something is there and misnaming it — the worst case to charge as absence.
        row.sibling = rc::probe::nearest_other_label(pkt, "door", {st.cx, st.cy});
        rc::probe::append(detect_probe_csv_, row);
    }
}

// ─── Main compute loop ───────────────────────────────────────────────────────

// SHADOW-MODE birth/death recorder — CONCEPT_AGENT_LIFECYCLE.md §4.2, theory in MODEL_HISTORY.md §4.
// RECORDS ONLY; it can never alter a birth or a removal. NOTE: door has a silhouette channel, so p_detect is the real thing.
void SpecificWorker::log_phantom_event(std::string_view event, std::uint64_t id, std::string_view name,
                                       float x, float y, const rc::DoorInstance* inst, std::string_view note)
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
        e.age_cycles    = inst->processed_cycles;
        e.p_detect      = inst->dbg_sil_pdetect;
        e.central_frac  = inst->dbg_sil_central;
        e.in_fov_frac   = (inst->dbg_sil_ntotal > 0)   // ★a FRACTION: the denominator was
                        ? static_cast<float>(inst->dbg_sil_ndet) / inst->dbg_sil_ntotal : 0.0f;   // already here, unused

        e.exist_logodds = inst->existence.logodds();
    }
    phantom_log_.write(e);
}

void SpecificWorker::compute()
{
    // Input recording: every delete this cycle causes is OURS (input_latch.h — what tells it from another agent's).
    const auto self_scope = input_snapshot_.self_scope();
    // (The SHARED mutual-exclusion rule's claim set — who else claims room space — is the DoorCore's, computed ONCE per
    //  cycle from the snapshot below and read by both its birth filter and its fit. This walk used to be a second copy.)

    if (not G or not rt_api_)
        return;

    // The decision log's clock: one tick per compute() that got past the graph guard (the cycle the replay will step).
    // ONE steady read feeds both the DDEC stamp and the recorded input (DoorCycleInput::t_ms() == the DDEC stamp).
    const std::int64_t cycle_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::steady_clock::now().time_since_epoch()).count();
    decision_log_.set_clock(++dec_cycle_, cycle_ns / 1'000'000);
    // Input recording (DoorConcept.RecordInputs, default off): the graph snapshot NOW, the media parts where the code
    // below consumes them, and the record written when this cycle ends (any return path) — after the actuator calls.
    input_snapshot_.begin_cycle(*G, inner_eigen_.get(), rt_api_.get(), dec_cycle_, cycle_ns,
                                static_cast<std::int64_t>(rc::provenance::now_ms()), mask_ingestor_.get());
    struct EndRecordedCycle { rc::door::InputSnapshot& s; ~EndRecordedCycle() { s.end_cycle(); } } end_recorded_cycle{input_snapshot_};
    // The fit's clock and camera for this cycle, before anything below can build an instance (a restore, a carry).
    if (core_) core_->set_cycle_views(input_snapshot_.current());

    // Which room are we in? Follows ltsm_agent's `current` edge and LETS GO of a room we have left.
    // Must run before anything below reads room_node_id_ — births, the containment prior and every RT
    // write are all parented by it.
    step_room_following();
    if (room_node_id_ == 0)
    {
        // ★ NO ROOM, BUT A PASSAGE IN FLIGHT (mid room-switch): the anchor's prediction and its robot-frame LiDAR /
        //   mask update need no room, and an edge left unwritten for 1 s is read as a dead owner — the controller
        //   would fall back to the room route in the doorway. Keep it alive; only the re-init waits for a room.
        if (passage_anchor_.has_value())
        {
            if (lidar_ingestor_)   // the room-frame sweep fails here; the robot-frame one does not
            {
                lidar_ingestor_->pump();
                // ★After EVERY pump: pump() reports room freshness only; the robot-frame sweep can be fresh alone.
                input_snapshot_.capture_bpearl(*G, inner_eigen_.get(), *lidar_ingestor_);
            }
            if (mask_ingestor_)
            {
                mask_ingestor_->refresh();
                input_snapshot_.capture_masks(*G, inner_eigen_.get(), rt_api_.get(), *mask_ingestor_);
            }
            step_passage_anchor();
        }
        // ★The core's media views must not fall behind the ingestors on a cycle its decision step skips: the packet
        //   staged here is "the last packet" the next cycle's tracker reads (core/sensing.h).
        if (core_) core_->observe_media(input_snapshot_.current());
        return;
    }

    // Hang a mirror of the crossed door from every proto-room (and drop mirrors whose room or source is gone).
    step_entry_mirrors();

    // Controller-owned affordance flags (claim / completion / epistemic_pending). Polled here rather than
    // pushed by update_node_attr_signal — see the connect block in initialize().
    poll_affordance_protocol();


    // Evidence-pipeline per-cycle counters (the *_cum fields persist). Producers below add to these; the
    // snapshot is pushed at the end of the cycle.
    ev_g_.births = ev_g_.merges = ev_g_.removals = 0;

    refresh_room_geometry();  // room-containment pose prior (cheap; the polygon is a nominal model)
    // Low LiDAR sweep in the ROOM frame — the hinge branch's evidence. Main thread (reads the graph).
    // (The fit reads it from the recorded sweep — core/sensing.h LeafSweep — no longer from the ingestor's storage.)
    if (lidar_ingestor_)
    {
        lidar_ingestor_->pump();
        // after EVERY pump: its return omits the robot-frame plane (make_bpearl_sweep)
        input_snapshot_.capture_bpearl(*G, inner_eigen_.get(), *lidar_ingestor_);
    }

    mask_ingestor_->refresh();
    input_snapshot_.capture_masks(*G, inner_eigen_.get(), rt_api_.get(), *mask_ingestor_);
    // The DoorCore on this cycle's input: ego-motion, merge, associate, [the existence update — old code, through
    // CoreWorld::after_association], birth — the only instance-lifecycle path — then the FIT of every door
    // (CoreWorld::fit_targets / begin_fit / after_fit = the old door loop and process_door_node) and the Phi rows.
    run_tracker_slice();

    // ── The three PRAGMATIC affordances: approach / open / cross ──────────────────────────────────
    // ★DELIBERATELY OUTSIDE the loop above. process_door_node() bails early on a stale or young
    // instance, and a cycle that cannot compute a precondition must SAY so rather than be skipped:
    // rc::pragmatic holds a standing offer on an unmeasured cycle and withdraws it only on a MEASURED
    // one, which is only possible if it is called on every cycle. Skipping instead would leave the
    // consumer ranking a price nobody refreshed — the frozen-gain defect the epistemic path already
    // paid for once (see common/epistemic_step's header).
    if (cfg_.pragmatic.enabled)
        for (auto& [door_id, inst] : fitter_->instances())
            step_pragmatic_affordances(inst);
    for (const auto& [door_id, inst] : fitter_->instances())
        remember_door(inst);
    step_remembered_looks();
    // The passage door hung from the robot while its approach/cross executes — AFTER the pragmatic step, so the
    // claims it reads were polled this cycle (specificworker_anchor.cpp).
    step_passage_anchor();

    // Overall compute()-cycle rate: counts every cycle, prints "Epoch time = …ms. Fps = N" once a
    // second (FPSCounter::print is throttled and self-counting).
    // ── Dashboard: counters + belief inspector, one throttled push ──
    ev_g_.instances  = static_cast<int>(fitter_->instances().size());
    ev_g_.mask_stale = not mask_ingestor_->packet().valid;
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

    fps_counter_.print("[door_concept Compute]");
}

// Retire one instance: drop its affordance node, forget it in the fitter, delete its graph node. The single
// teardown path shared by every lifecycle exit (merge / death), keeping the affordance+fitter+graph invariant.
// ★Named to match the other four agents, which already had it — this was the same three lines written inline.
void SpecificWorker::retire_instance(std::uint64_t id)
{
    if (auto it = fitter_->instances().find(id); it != fitter_->instances().end())
    {
        log_instance_teardown(it->second, "retire");
        it->second.affordance.remove();
        // ★ALL FOUR, or the pragmatic ones leak. Deleting the door node drops its RT edge but NOT its
        // affordance children, and rc::owned::remove_stale_affordances only runs at startup and shutdown
        // — so between those an orphaned `aff_door_1_open` would sit in the shared graph, offered, with
        // nothing alive behind it. Four agents already shipped exactly this (TIER B).
        it->second.aff_approach.remove();
        it->second.aff_open.remove();
        it->second.aff_cross.remove();
        it->second.aff_close.remove();
    }
    fitter_->forget_node(id);
    G->delete_node(id);
}


// ─── The tracker slice: DoorCore (src/core/door_core.h) on this cycle's input ───────────────────────────────────
// The decisions are the core's; what stays here is the graph side of them (CoreWorld below) and where the core's
// decisions and diagnostic rows go. ★Behaviour-preserving move (Task 4 of the door-core replay plan): the core reads
// the SAME values the old run_instance_tracker read, but from the input snapshot (taken at the top of the cycle) —
// the foreign claims, the camera pose behind the ego-motion and its clock. Those three are cycle-top reads now,
// where they used to be mid-cycle ones (task-4-report.md names each).
void SpecificWorker::run_tracker_slice()
{
    core_->step(input_snapshot_.current());
}

std::uint64_t SpecificWorker::CoreWorld::room_id() const { return w_.room_node_id_; }

std::vector<Eigen::Vector2f> SpecificWorker::CoreWorld::room_polygon() const
{ return w_.fitter_ ? w_.fitter_->room_polygon() : std::vector<Eigen::Vector2f>{}; }

std::vector<std::pair<std::uint64_t, rc::DoorFitState*>> SpecificWorker::CoreWorld::fit_states() const
{
    std::vector<std::pair<std::uint64_t, rc::DoorFitState*>> out;
    if (not w_.fitter_) return out;
    out.reserve(w_.fitter_->instances().size());
    for (auto& [id, inst] : w_.fitter_->instances())   // ★the map's own order: association ties and Phi rows follow it
        out.emplace_back(id, &inst);
    return out;
}

// compute()'s old door loop, its graph half: the doors to fit, in graph order. Doors are generic `object` nodes named
// "door_*" (schema migration); filter by name prefix.
std::vector<std::uint64_t> SpecificWorker::CoreWorld::fit_targets()
{
    fit_nodes_.clear();
    std::vector<std::uint64_t> ids;
    for (const auto& node : w_.G->get_nodes_by_type("object"))
        if (node.name().starts_with("door") and not rc::room::is_proto_mirror(*w_.G, node)   // a mirror is not a belief
            and not rc::interaction::is_anchor(*w_.G, node)    // nor is the passage door's robot-relative copy
            and not w_.held_doors_.contains(node.id()))        // nor a door HELD in the room left behind (release_room)
        {
            ids.push_back(node.id());
            fit_nodes_.insert_or_assign(node.id(), node);
        }
    return ids;
}

// process_door_node's graph half, before the fit: the instance (ensure_instance), the dashboard series and the canvas.
rc::DoorFitState* SpecificWorker::CoreWorld::begin_fit(std::uint64_t id)
{
    const auto nit = fit_nodes_.find(id);
    if (nit == fit_nodes_.end()) return nullptr;
    const DSR::Node& node = nit->second;
    const bool created = w_.fitter_->ensure_instance(node, w_.room_node_id_);
    auto& inst = w_.fitter_->instances().at(node.id());

    if (created)
    {
        // Register per-instance time-series (Qt dashboards stay in the worker).
        if (w_.ts_plot_)
        {
            w_.ts_plot_->add_series(inst.node_name + "_fe",  QColor(255, 170,   0), 1.1f);
            if (w_.ts_cov_plot_) w_.ts_cov_plot_->add_series(inst.node_name + "_cov", QColor(  0, 190, 255), 1.1f);
            if (w_.ts_res_plot_) w_.ts_res_plot_->add_series(inst.node_name + "_res", QColor(170,  80, 255), 1.1f);
        }
        w_.place_door_on_canvas(node, inst);
    }

    // The RT parent is resolved lazily — a door is born hanging from the room and moves under a wall_*
    // node as soon as one is near enough. Re-place it when that happens, or the canvas keeps drawing it
    // where its FIRST parent was.
    w_.place_door_on_canvas(node, inst);
    return &inst;
}

// process_door_node's graph half, after a fit that ran: persist the belief, convergence, the epistemic step.
void SpecificWorker::CoreWorld::after_fit(std::uint64_t id, const rc::door::core::DoorObservation& obs, float free_energy)
{
    const auto nit = fit_nodes_.find(id);
    auto it = w_.fitter_->instances().find(id);
    if (nit == fit_nodes_.end() or it == w_.fitter_->instances().end()) return;
    w_.publish_door_cycle(it->second, nit->second, obs, free_energy);
}

std::optional<rc::door::core::GhostMatch> SpecificWorker::CoreWorld::match_ghost(const Eigen::Vector2f& xy) const
{
    const DoorGhost* g = w_.match_ghost(xy);
    if (g == nullptr) return std::nullopt;
    return rc::door::core::GhostMatch{g->name, g->xy, g->lived_cycles, static_cast<std::size_t>(g - w_.ghosts_.data())};
}

std::optional<rc::door::core::IdentityMatch> SpecificWorker::CoreWorld::match_identity(const Eigen::Vector2f& xy) const
{
    const DoorIdentity* i = w_.match_identity(xy);
    if (i == nullptr) return std::nullopt;
    return rc::door::core::IdentityMatch{i->name, i->xy};
}

std::vector<std::string> SpecificWorker::CoreWorld::reserved_names() const { return w_.reserved_names(); }

void SpecificWorker::CoreWorld::retire(std::uint64_t id) { w_.retire_instance(id); }

void SpecificWorker::CoreWorld::set_assigned_mask(std::uint64_t id, int slice)
{
    if (auto it = w_.fitter_->instances().find(id); it != w_.fitter_->instances().end())
        it->second.assigned_mask_idx = slice;
}

// What ran between the association and the births in the old run_instance_tracker — kept in place, in order.
void SpecificWorker::CoreWorld::after_association(const rc::door::core::TrackerCycleInfo& info)
{
    // Evidence-pipeline counters (section 1 of the dashboard). Same fields every concept agent fills.
    auto& ev = w_.ev_g_;
    ev.mask_frame_id = info.mask_frame_id;
    ev.total_slices  = info.total_slices;
    ev.class_dets    = info.class_dets;
    ev.assigned      = info.assigned;
    ev.discarded     = info.class_dets - info.assigned;
    ev.births       += info.tracker_births;
    ev.births_cum   += static_cast<long>(info.tracker_births);
    ev.removals     += info.tracker_deaths;
    ev.removals_cum += static_cast<long>(info.tracker_deaths);
    ev.merges       += info.merges;       // the merge sweep's counter (common/track/merge_instances.h's rule)
    ev.merges_cum   += info.merges;
    // The remembered-door look reads this frame's detections as "seen there" (door_look.h, step_remembered_looks).
    w_.look_new_frame_ = info.new_frame;
    w_.look_dets_xy_ = info.dets_xy;

    // EXISTENCE BELIEF: continuous log-odds removal — the principled replacement for the wall-clock
    // stillbirth prune below. The prune could not kill a MATURE phantom (processed_cycles≥maturity → "furniture"
    // immunity, purely age-based) and its binary streak reset on ANY assignment, so a phantom fed a trickle of
    // clutter (door_3: 8 points at 7.4 m) never reached patience AND aged into permanence. The existence belief
    // fixes both: it integrates SUPPORT-MASS-WEIGHTED evidence (8 pts where ~130 are expected → strong negative)
    // once per sensor frame, with no age immunity — a real door stays only by continuing to be explained.
    // ★NO FALLBACK, deliberately — the five agents that retired their prune have none either.
    // OFF means NO removal, which is an honest nothing; the alternative was a fallback this
    // file's own comment documents as broken (see the note above).
    // ★BEFORE the update, deliberately: the row must carry the belief as it stood BEFORE the outcome
    // recorded on that same row was integrated. Weighting a trial by the posterior the trial produced
    // is circular, and the one-cycle offset is also what confused the August audit until the column
    // was renamed `ex_p_prior`. It also means an instance removed THIS cycle still leaves the row
    // recording the look that removed it.
    // Refresh the graded posterior ONCE per cycle, before any door samples it: it is a property of the
    // frame, not of a door, and re-reading it per instance would let two doors in the same cycle be
    // judged against different frames.
    const bool semantic_ok = w_.semantic_field_.refresh(w_.G.get(), rc::door_semantic_class_id());
    w_.input_snapshot_.capture_semantic(w_.semantic_field_, semantic_ok);
    // One RGB frame per cycle, deep-copied out of the loaned SHM view by the ingestor.
    if (w_.rgb_ingestor_)
        w_.rgb_ingestor_->pump();
    if (w_.depth_ingestor_)
        w_.depth_ingestor_->pump();
    w_.refresh_door_actuation_ui();

    w_.log_detect_probe();
    // (The existence update — Existence.Enabled — runs next, inside the DoorCore: it reads the semantic plane captured
    //  above from the input and the frames from stage_images below.)
}

// The frames the core's existence checks score, after the pump above: the ingestors' own (BGR, decoded depth).
void SpecificWorker::CoreWorld::stage_images(rc::door::core::ImageSource& images)
{
    static const cv::Mat empty;
    images.stage_live(w_.rgb_ingestor_ ? w_.rgb_ingestor_->frame() : empty,
                      w_.depth_ingestor_ ? w_.depth_ingestor_->frame() : empty);
}

// An existence REMOVE, the graph half that ran BEFORE the old REMOVE row: remember the ghost (which forgets the stored
// identity) and log the affordance withdrawals.
void SpecificWorker::CoreWorld::before_remove(std::uint64_t id)
{
    if (const auto it = w_.fitter_->instances().find(id); it != w_.fitter_->instances().end())
    {
        w_.remember_ghost(it->second);
        w_.log_instance_teardown(it->second, "existence remove");
    }
}

// …and AFTER it: the shadow-mode death record, the affordance nodes, the fit state, the node.
void SpecificWorker::CoreWorld::remove_door(std::uint64_t id, float L)
{
    if (const auto it = w_.fitter_->instances().find(id); it != w_.fitter_->instances().end())
    {
        // Shadow-mode death record (§4.2), taken BEFORE teardown while the state that justified
        // the kill is readable. Low p_detect here ⇒ our removal bug, not a classifier phantom.
        w_.log_phantom_event("DEATH", id, it->second.node_name,
                             it->second.model.state().cx, it->second.model.state().cy, &it->second,
                             std::format("L {:.2f}", L));
        it->second.affordance.remove();
        it->second.aff_approach.remove();   // and the three pragmatic siblings — see retire_instance
        it->second.aff_open.remove();
        it->second.aff_cross.remove();
        it->second.aff_close.remove();
    }
    w_.fitter_->forget_node(id);
    w_.G->delete_node(id);
}

std::uint64_t SpecificWorker::CoreWorld::create_door(const rc::door::core::BirthRequest& r)
{
    const Eigen::Vector3f& c = r.centroid;
    const auto new_id = w_.scene_graph_->create_instance_from_detection(c, r.room_id, r.preferred_name, r.reserved);
    if (new_id != 0)
    {
        // (The birth seed — the room-frame XY the newborn's model starts at — is noted by the DoorCore itself.)
        // ★SENIORITY IS OBSERVED AT BIRTH, not inferred later (common/exclusion). Birth is the only
        // moment at which "who was here first" is actually seen: resolving it on a later cycle would
        // have a real object, standing legitimately beside a real neighbour, wake up after a RESTART,
        // find the neighbour present, and declare ITSELF the junior. Anything not created this run
        // stays senior by default — the rule can fail to catch a collision, never invent one.
        // ⚠CORRECTION (Task 4b, replaying the fit): this lookup is NOT inert. Task 4 reported the instance as built
        //  only in adopt_birth, after it; but the insert above fired modify_node_slot synchronously (same thread), so
        //  ensure_instance has ALREADY built it and resolve_at_birth runs — against the DoorCore's claim set, the SAME
        //  set the worker's own walk produced (that walk is gone). Behaviour unchanged; on tourR1/R2 the set is empty.
        if (auto it = w_.fitter_->instances().find(new_id); it != w_.fitter_->instances().end())
            it->second.exclusion.resolve_at_birth({c.x(), c.y(), w_.cfg_.door_prior_w_m, 0.10f /*leaf thickness*/, 0.0f},
                                                  w_.core_->claims(), 0.0f, w_.cfg_.door_prior_h_m);
        // Shadow-mode birth record (CONCEPT_AGENT_LIFECYCLE.md §4.2): place + viewpoint that
        // produced it, so a phantom that dies young is attributable to both.
        w_.log_phantom_event("BIRTH", new_id, "", c.x(), c.y(), nullptr, "");
    }
    return new_id;
}

void SpecificWorker::CoreWorld::reacquire(std::uint64_t new_id, const rc::door::core::GhostMatch& g)
{
    if (g.handle < w_.ghosts_.size())
        w_.core_->fit().note_reacquire(new_id, w_.ghosts_[g.handle].belief);
}

void SpecificWorker::CoreWorld::forget_ghost(const Eigen::Vector2f& xy) { w_.forget_ghost(xy); }

std::optional<std::string> SpecificWorker::CoreWorld::adopt_birth(std::uint64_t new_id, int slice,
                                                                  const Eigen::Vector3f& c)
{
    if (const auto nopt = w_.G->get_node(new_id); nopt.has_value())
    {
        w_.fitter_->ensure_instance(nopt.value(), w_.room_node_id_);
        if (auto it = w_.fitter_->instances().find(new_id); it != w_.fitter_->instances().end())
            it->second.assigned_mask_idx = slice;
        // Write the identity down NOW, at the birth place. Waiting for removal or shutdown loses it
        // to a SIGKILL — and the fleet gets SIGKILLed (see the attr-signal starvation note); the
        // aperture is refined by the later upserts anyway.
        w_.note_identity(nopt->name(), Eigen::Vector2f(c.x(), c.y()));
    }
    if (const auto it = w_.fitter_->instances().find(new_id); it != w_.fitter_->instances().end())
        return it->second.node_name;
    return std::nullopt;
}

// Snapshot residual_concept's published residual field. Mirrors table_concept::read_residual_field — same
// node, same attribute trio — so this is one more consumer of an established channel. The field is already
// EXPLAINED-COLLAPSED by the producer (cells a modelled object accounts for are attenuated toward zero), so
// "first cell above p_occupied along a ray" means "first thing out there nothing explains" with no extra
// filtering here. Returns false (and leaves the field empty ⇒ invalid ⇒ the cascade skips this rung) whenever
// residual_concept is not up yet, which is the normal state during startup.
bool SpecificWorker::read_residual_field()
{
    return rc::read_residual_field(*G, room_node_id_, residual_field_);
}

// Load the room's delimiting polygon (a trusted NOMINAL model authored by room_concept, never fitted) into the
// fitter so it can impose the room-containment pose prior. Mirrors cabinet_concept::refresh_room_geometry.
void SpecificWorker::refresh_room_geometry()
{
    static int miss = 0;                          // throttle the "polygon still missing" diagnostic
    if (not G or room_node_id_ == 0) return;
    auto room = G->get_node(room_node_id_);
    if (not room.has_value())
    {
        // Latched room id went stale (room_concept recreated the room on relocalization). Re-resolve so the
        // containment prior recovers instead of silently reading a dead node forever.
        const auto room_opt = rc::room::current_room(*G);
        if (not room_opt.has_value()) { room_node_id_ = 0; return; }
        room_node_id_ = *room_opt;
        room = G->get_node(room_node_id_);
        if (not room.has_value()) return;
    }
    const auto px = G->get_attrib_by_name<delimiting_polygon_x_att>(room.value());
    const auto py = G->get_attrib_by_name<delimiting_polygon_y_att>(room.value());
    if (not px.has_value() or not py.has_value())
    {
        if (++miss % 120 == 1)
            std::print("door_concept: [room-prior] room node {} has NO delimiting_polygon attribute yet "
                       "(containment prior INACTIVE — out-of-room doors cannot be removed)\n", room_node_id_);
        return;
    }
    const auto& xs = px->get(); const auto& ys = py->get();
    const std::size_t n = std::min(xs.size(), ys.size());
    if (n < 3)
    {
        if (++miss % 120 == 1)
            std::print("door_concept: [room-prior] delimiting_polygon present but degenerate (n={})\n", n);
        return;
    }
    std::vector<Eigen::Vector2f> poly; poly.reserve(n);
    Eigen::Vector2f centroid = Eigen::Vector2f::Zero();
    float xmin = 1e9f, xmax = -1e9f, ymin = 1e9f, ymax = -1e9f;
    for (std::size_t i = 0; i < n; ++i)
    {
        poly.emplace_back(xs[i], ys[i]); centroid += poly.back();
        xmin = std::min(xmin, xs[i]); xmax = std::max(xmax, xs[i]);
        ymin = std::min(ymin, ys[i]); ymax = std::max(ymax, ys[i]);
    }
    centroid /= static_cast<float>(n);
    const bool first = not fitter_->has_room_polygon();
    fitter_->set_room_geometry(centroid, std::move(poly));
    if (first)
        std::print("door_concept: [room-prior] room polygon LOADED — {} verts, x∈[{:.2f},{:.2f}] y∈[{:.2f},{:.2f}] "
                   "(containment prior ACTIVE)\n", n, xmin, xmax, ymin, ymax);
}

// ─── Identity re-acquisition (ghost registry) ───────────────────────────────────────────────────

// Remember a door about to be deleted, so the same physical door coming back resumes this identity + geometry.
void SpecificWorker::remember_ghost(const rc::DoorInstance& inst)
{
    if (cfg_.exist_reacquire_radius_m <= 0.0f or not inst.ai2_initialized or inst.is_bearing_hypothesis)
        return;
    // Keyed on the APERTURE: the identity of a door is its hole in the wall, so a door that flickers out
    // while its leaf happens to be open must still be re-acquired at the same place.
    const auto& s = inst.model.state();
    const Eigen::Vector2f ap(s.ap_cx, s.ap_cy);
    // A GHOST IS A PLACE, NOT A NAME. Superseding by name was the identity defect: `door_1` died at the north
    // door, its number was recycled to a different door across the room, and when THAT one died this line
    // erased the north door's ghost — so the real door, reborn 0.50 m from where it died (well inside
    // ReacquireRadiusM), had nothing to match and came back as door_3. Only a ghost of the SAME HOLE may be
    // replaced; that is the one this removal actually supersedes. (Name recycling is now blocked upstream in
    // create_instance_from_detection, so two ghosts cannot normally share a name either.)
    std::erase_if(ghosts_, [&](const DoorGhost& g)
                  { return (g.xy - ap).norm() < cfg_.exist_reacquire_radius_m; });
    ghosts_.push_back({inst.node_name, ap, inst.ai2_belief, inst.processed_cycles});
    // ★ ABSENCE FORGETS THE NAME ACROSS RUNS (2026-10-18, user). This runs on a REMOVE, i.e. the existence belief
    //   was driven down by looks that should have seen the door and did not. The RAM ghost above still lets a
    //   real door that merely flickered out come back under its name within this run; but writing the name to
    //   disk made every phantom permanent — the table kept door_2/3/4/6 in room_1 run after run, each reserving
    //   a name and a place. A door re-acquired from its ghost writes itself back (note_identity at re-birth).
    forget_identity(inst.node_name, ap, s.w);
    // Bounded, most-recent-first: this is a short-lived flicker memory, not a permanent map.
    while (static_cast<int>(ghosts_.size()) > std::max(1, cfg_.exist_ghost_max))
        ghosts_.erase(ghosts_.begin());
}

// The names still spoken for by a removed-but-remembered door — this run's ghosts AND the identities
// persisted from earlier runs. Handed to the birth path so a freed number is never given to a DIFFERENT
// physical door while its owner can still come back and claim it.
std::vector<std::string> SpecificWorker::reserved_names() const
{
    std::vector<std::string> names;
    names.reserve(ghosts_.size() + identities_.size());
    for (const auto& g : ghosts_)      names.push_back(g.name);
    for (const auto& i : identities_)  names.push_back(i.name);
    return names;
}

// ─── Identity across runs (etc/door_identities.csv) ─────────────────────────────────────────────
//
// Ghosts are RAM: they cannot survive the process, and cleanup_owned_nodes deletes every door node on the
// way out, so at the next launch nothing in the graph says which door was which. Names then fell out of
// BIRTH ORDER — the order the robot happens to sweep the room — and the same physical door was door_1 one
// run and door_2 the next. This table is the persistent half of the same idea as remember_ghost: keyed on
// the APERTURE (the hole in the wall, which is what a door IS), carrying the name and nothing else.
//
// Locale: read with std::from_chars, write through the classic locale — CLAUDE.md. These machines run
// es_ES, where strtof would silently truncate "4.61" to 4 and put every door on the wrong side of a wall.

std::string SpecificWorker::current_room_name() const
{
    if (room_node_id_ == 0) return {};
    if (const auto n = G->get_node(room_node_id_); n.has_value())
        room_name_cache_ = {room_node_id_, n->name()};
    // ltsm evicts the room left behind BEFORE release_room runs; its doors are still written under its name.
    return room_name_cache_.first == room_node_id_ ? room_name_cache_.second : std::string{};
}

void SpecificWorker::load_identities()
{
    std::ifstream f("etc/door_identities.csv");
    if (not f)
        return;
    identities_.clear();
    // Parsing is rc::door::parse_identity_table (door_identity_table.h): std::from_chars, both the 4-column rows of the
    // 2026-10-13 table (identity only) and the 17-column ones that carry the pose to restore.
    auto parsed = rc::door::parse_identity_table(f);
    if (parsed.unkeyed_format)
    {
        decision_log_.record(rc::door::DoorDecisionKind::IdentityLoad, 0, {}, 0, 0, 1.0);
        std::print("door_concept: [identity] etc/door_identities.csv has the old un-keyed format — ignored "
                   "(rewritten room-keyed on the next save)\n");
        return;
    }
    identities_ = std::move(parsed.rows);
    decision_log_.record(rc::door::DoorDecisionKind::IdentityLoad, 0, {}, double(identities_.size()), double(parsed.malformed));
    for (const auto& i : identities_)   // spec 6.6: the table content at load is an explicit event
        decision_log_.record(rc::door::DoorDecisionKind::IdentityLoad, 0, i.name, i.xy.x(), i.xy.y(), i.pose.has_value() ? 1.0 : 0.0, 0, i.room);
    if (not identities_.empty())
    {
        std::print("door_concept: [identity] loaded {} remembered door face(s) ({} dropped as malformed):",
                   identities_.size(), parsed.malformed);
        for (const auto& i : identities_)
            std::print(" {}:{}@({:.2f},{:.2f}){}", i.room, i.name, i.xy.x(), i.xy.y(), i.pose.has_value() ? "" : "[id-only]");
        std::print("\n");
    }
}

void SpecificWorker::save_identities() const
{
    // DIAG-ROTATE: exempt — this is PERSISTED STATE, not a diagnostics log. note_identity() rewrites the
    // whole table on every identity change, so rotating here would leave one stamped copy per door seen.
    std::ofstream f("etc/door_identities.csv", std::ios::trunc);
    if (not f)
        return;
    rc::door::write_identity_table(f, identities_);   // imbues the classic locale: never a comma decimal separator
}

// Upsert BY PLACE WITHIN THE CURRENT ROOM: one hole in this room's walls carries one name. The same name may hold a
// row in another room — that is the doorway's other face, not a conflict.
void SpecificWorker::note_identity(const std::string& name, const Eigen::Vector2f& ap, float width_m)
{
    DoorIdentity row;
    row.name = name;
    row.xy   = ap;
    // A birth: the door exists with the birth prior's probability (Existence.BirthLogodds).
    upsert_identity(std::move(row), width_m, 1.0f / (1.0f + std::exp(-cfg_.exist_birth_logodds)));
}

float SpecificWorker::identity_weight(const rc::DoorInstance& inst) const
{
    return inst.existence_seeded ? inst.existence.p_exists() : 1.0f / (1.0f + std::exp(-cfg_.exist_birth_logodds));
}

void SpecificWorker::upsert_identity(DoorIdentity row, float width_m, float weight)
{
    const std::string room = current_room_name();
    if (cfg_.exist_reacquire_radius_m <= 0.0f)
        return;
    // ★ ONE DOORWAY, ONE NAME (2026-10-18, user). A row whose centre lies within this opening's width plus the
    //   re-acquire radius describes the SAME hole — the openings overlap once their centres are closer than the
    //   width, and the radius is the place uncertainty this agent already uses for "same door". Matching on the
    //   radius alone kept room_1's main doorway under two names 1.07 m apart (door_1 @ −1.65 from an aperture fitted
    //   on the far jamb, door_5 @ −0.58), and the next run picked whichever was closer: door_5.
    //   ⚠ SUPERSEDED 2026-10-21 (tourP9): the OLDEST name of the hole wins, and the place moves by the writer's P(exists)
    //   — see rc::door::upsert_row. width_m = 0 (a birth, before the width is fitted) keeps the old radius-only test.
    const float same_hole = cfg_.exist_reacquire_radius_m + std::max(0.0f, std::isfinite(width_m) ? width_m : 0.0f);
    const std::string dec_name = row.name;
    const Eigen::Vector2f dec_xy = row.xy;
    if (rc::door::upsert_row(identities_, room, std::move(row), same_hole, weight))
    {
        decision_log_.record(rc::door::DoorDecisionKind::IdentityUpsert, 0, dec_name, dec_xy.x(), dec_xy.y(), width_m, weight, room);
        save_identities();
    }
}

// What a FITTED door leaves behind to be restored from: its aperture pose and size (room frame), its existence belief,
// and its leaf memory — the same quantities the crossed door carries across (EntryMirror), here in the room frame.
SpecificWorker::DoorIdentity SpecificWorker::identity_of(const rc::DoorInstance& inst) const
{
    const auto& s = inst.model.state();
    DoorIdentity row;
    row.name = inst.node_name;
    row.xy   = Eigen::Vector2f(s.ap_cx, s.ap_cy);
    if (std::isfinite(s.ap_yaw) and s.w > 0.0f and s.h > 0.0f and s.thickness > 0.0f)
        row.pose = rc::door::RestorePose{s.ap_yaw, s.w, s.h, s.thickness};
    if (inst.existence_seeded)
        row.exist_logodds = rc::door::remembered_logodds(inst);   // review I-1: the evidence, without the band prior
    if (inst.last_detection_wall_ms > 0)
        row.seen_wall_ms = inst.last_detection_wall_ms;   // Task D-C: the remembered belief's age
    if (inst.ai2_initialized)
    {
        // Task D-C round 2: how well the PLACE is known — the aperture centre s + w/2 along its wall.
        const auto S = inst.ai2_belief.covariance();
        const float v = S(0, 0) + S(0, 1) + 0.25f * S(1, 1);
        if (std::isfinite(v) and v > 0.0f) row.place_sd = std::sqrt(v);
    }
    const Eigen::Vector2f open = inst.leaf.swing * inst.aperture.across_u();
    if (inst.leaf_pose.hinge_xy.allFinite() and open.allFinite() and std::isfinite(inst.phi_est)
        and std::isfinite(inst.phi_sigma) and std::isfinite(inst.hinge_llr) and std::isfinite(inst.swing_llr))
        row.leaf = rc::door::LeafMemory{inst.leaf_pose.hinge_xy, open, inst.hinge_llr, inst.swing_llr,
                                        inst.phi_est, inst.phi_sigma};
    return row;
}

// The persistent row(s) of a door removed for absence: its name, or any row on the same hole (see note_identity).
void SpecificWorker::forget_identity(const std::string& name, const Eigen::Vector2f& ap, float width_m)
{
    const std::string room = current_room_name();
    if (room.empty())
        return;
    const float same_hole = cfg_.exist_reacquire_radius_m + std::max(0.0f, std::isfinite(width_m) ? width_m : 0.0f);
    const auto n = std::erase_if(identities_, [&](const DoorIdentity& i)
                                 { return i.room == room and (i.name == name or (ap.allFinite() and (i.xy - ap).norm() < same_hole)); });
    if (n > 0)
    {
        std::print("door_concept: [identity] {} removed for absence — {} remembered row(s) forgotten in {}\n", name, n, room);
        decision_log_.record(rc::door::DoorDecisionKind::IdentityForget, 0, name, ap.x(), ap.y(), width_m, double(n), room);
        save_identities();
    }
}

const SpecificWorker::DoorIdentity* SpecificWorker::match_identity(const Eigen::Vector2f& xy) const
{
    const std::string room = current_room_name();
    if (cfg_.exist_reacquire_radius_m <= 0.0f or room.empty())
        return nullptr;
    const DoorIdentity* best = nullptr;
    float best_d = cfg_.exist_reacquire_radius_m;
    for (const auto& i : identities_)
        if (i.room != room) continue;
        else if (const float d = (i.xy - xy).norm(); d < best_d) { best_d = d; best = &i; }
    return best;
}

// Shutdown: the doors still alive have never been remembered (remember_ghost only fires on removal), and
// they are about to be deleted from the graph. Write them down or the next run starts blind.
void SpecificWorker::remember_live_identities()
{
    if (not fitter_)
        return;
    for (const auto& [id, inst] : fitter_->instances())
    {
        if (not inst.ai2_initialized or inst.is_bearing_hypothesis)
            continue;
        upsert_identity(identity_of(inst), inst.model.state().w, identity_weight(inst));
    }
}

const SpecificWorker::DoorGhost* SpecificWorker::match_ghost(const Eigen::Vector2f& xy) const
{
    if (cfg_.exist_reacquire_radius_m <= 0.0f)
        return nullptr;
    const DoorGhost* best = nullptr;
    float best_d = cfg_.exist_reacquire_radius_m;
    for (const auto& g : ghosts_)
        if (const float d = (g.xy - xy).norm(); d < best_d) { best_d = d; best = &g; }
    return best;
}

// ─── NBV decision monitor (etc/door_nbv_log.csv) ────────────────────────────────────────────────
//
// "The robot is still being sent into the door" has THREE possible authors — the four-face plan, the pose we
// publish, or the controller's own standpoint repair downstream — and no artefact could tell them apart. This
// row is the producer's full confession: where the door is, where we told the robot to stand, how far apart
// those are, and the per-face evidence behind the choice. If `dist_m` here is healthy (~the stand-off) and the
// robot still ends up at the leaf, the defect is downstream of this agent and the CSV proves it.
//
// One row per PUBLISHED proposal (i.e. per affordance refresh), so it lines up 1:1 with what the controller
// can see. Written through the classic locale — see CLAUDE.md: these machines run es_ES, and an ofstream that
// picks up a comma separator would corrupt its own columns on read-back.
void SpecificWorker::log_nbv_decision(const rc::DoorInstance& inst, const rc::nbv::Plan& plan,
                                      const rc::EpistemicProposal& prop)
{
    static std::ofstream csv = []
    {
        std::ofstream f; rc::diag::open_rotating(f, "etc/door_nbv_log.csv");
        f.imbue(std::locale::classic());
        f << "cycle,node,valid,door_cx,door_cy,door_yaw,door_w,ap_cx,ap_cy,"
             "face,standoff_m,band_min_m,band_max_m,tgt_x,tgt_y,tgt_yaw,dist_to_door_m,tgt_in_room,gain,"
             "vis_px,vis_nx,vis_py,vis_ny,reach_px,reach_nx,reach_py,reach_ny,"
             "pdet_px,pdet_nx,pdet_py,pdet_ny,exp_px,exp_nx,exp_py,exp_ny,n_obstacles\n";
        return f;
    }();
    if (not csv)
        return;

    const auto& ms = inst.model.state();
    // THE number to read first: how far the published standpoint is from the door the robot must photograph.
    // A value near 0 means we are driving it into the leaf; it should sit inside [band_min, band_max].
    const float dist = std::hypot(prop.epistemic_target_x_m - ms.cx, prop.epistemic_target_y_m - ms.cy);
    const bool in_room = fitter_->has_room_polygon()
                       ? fitter_->point_in_room(Eigen::Vector2f(prop.epistemic_target_x_m,
                                                                prop.epistemic_target_y_m), 0.0f)
                       : true;
    static const char* fn[4] = {"+x", "-x", "+y", "-y"};
    csv << inst.processed_cycles << ',' << inst.node_name << ',' << (plan.valid ? 1 : 0) << ','
        << ms.cx << ',' << ms.cy << ',' << ms.yaw << ',' << ms.w << ',' << ms.ap_cx << ',' << ms.ap_cy << ','
        << (plan.valid ? fn[plan.best_face] : "-") << ',' << plan.best_standoff_m << ','
        << plan.standoff_min_m << ',' << plan.standoff_max_m << ','
        << prop.epistemic_target_x_m << ',' << prop.epistemic_target_y_m << ','
        << prop.epistemic_target_yaw_rad << ',' << dist << ',' << (in_room ? 1 : 0) << ','
        << prop.epistemic_gain;
    for (int i = 0; i < 4; ++i) csv << ',' << plan.face_visible[i];
    for (int i = 0; i < 4; ++i) csv << ',' << (plan.face_reachable[i] ? 1 : 0);
    for (int i = 0; i < 4; ++i) csv << ',' << plan.face_p_detect[i];
    for (int i = 0; i < 4; ++i) csv << ',' << plan.face_gains[i];
    csv << ',' << nbv_obstacle_count_ << '\n';
    csv.flush();

    // Loud, once, if we ever publish a standpoint ON the door: that is the user-visible "crash into it"
    // symptom, and it must never pass silently just because a CSV recorded it.
    if (plan.valid and dist < plan.standoff_min_m * 0.5f)
    {
        static int shouted = 0;
        if (shouted++ < 5)
            std::print("door_concept: [NBV] ★PUBLISHING A STANDPOINT ON THE DOOR — {} target=({:.2f},{:.2f}) is "
                       "{:.2f} m from the leaf, band=[{:.2f},{:.2f}] face={} in_room={}\n",
                       inst.node_name, prop.epistemic_target_x_m, prop.epistemic_target_y_m, dist,
                       plan.standoff_min_m, plan.standoff_max_m, fn[plan.best_face], in_room ? 1 : 0);
    }
}

// Consume the ghost(s) at a place a door has just been re-acquired at. By PLACE, for the same reason
// remember_ghost supersedes by place: a name can outlive its owner and be worn by a ghost somewhere else
// entirely, and erasing that one would throw away an identity nobody claimed.
void SpecificWorker::forget_ghost(const Eigen::Vector2f& xy)
{
    std::erase_if(ghosts_, [&](const DoorGhost& g)
                  { return (g.xy - xy).norm() < cfg_.exist_reacquire_radius_m; });
}

// Continuous existence belief on the SHARED rc::exist channel (the one table/chair use): fold one sensor
// frame of pixel-level silhouette evidence into each instance's log-odds L and remove the doors whose
// predicted silhouette is demonstrably empty. See DoorInstance::existence for the model.
//
// The discipline, and what each part of it fixes (live run 2026-07-29, etc/door_existence_log.csv):
//   OCCUPANCY confirms · ABSENCE removes · OCCLUSION and OUT-OF-FoV **HOLD**.
//   · HOLD is structural, not a special case: rc::exist HOLDs whenever n_detectable==0, so a door behind the
//     robot contributes nothing. The old scheme took pd = max(zed_pd, range_detectability) where the second
//     term had NO bearing test — a door squarely behind the camera scored pd≈0.25–0.47 and was charged
//     absence every frame until it died (door_1: L 4 → −3.01 over ~180 frames while roi=0), then re-born as
//     door_3 at the same spot. That whole term is gone.
//   · Occlusion shrinks the detectable footprint continuously (occluded samples simply do not vote) instead
//     of the old `continue`, which granted indefinite immunity — door_2 sat frozen at L=2.29 for 1204 frames
//     because its line of sight happened to be blocked.
//   · Detectability is now ONE physical covariate, the silhouette's subtended image area (n_cells), which
//     collapses for a far door AND for an edge-on one. The old separate obliquity ramp returned exactly 0
//     below cos=0.20, making a grazing phantom permanently unjudgeable.
// Runs from run_instance_tracker (association already resolved, so assigned_mask_idx is this cycle's).
// (update_existence_beliefs moved into the DoorCore — src/core/door_existence.cpp, Task 5 of the door-core replay plan.
//  Its graph half is CoreWorld::before_remove / remove_door below; its frames come from CoreWorld::stage_images.)

///////////////////////////////////////////////////////////////
// (process_door_node moved: its fit into DoorCore::fit_doors (src/core/door_core.cpp), its graph half into
//  CoreWorld::begin_fit / after_fit above.)

void SpecificWorker::publish_door_cycle(rc::DoorInstance& inst,
                                         const DSR::Node& node,
                                         const DoorObservation& observation,
                                         float free_energy)
{
    const auto node_id = node.id();
    if (not scene_graph_->persist_door_belief(inst, node_id, room_node_id_, free_energy))
        return;
    if (not assess_door_state(inst, node_id, free_energy))
        return;
    publish_door_diagnostics(inst, observation, free_energy);
    publish_door_intentions(inst, node_id, observation, free_energy);
}

bool SpecificWorker::assess_door_state(rc::DoorInstance& inst, uint64_t node_id, float free_energy)
{
    auto node_opt = G->get_node(node_id);
    if (not node_opt.has_value())
        return false;

    step_convergence(inst, node_opt.value(), free_energy);
    return true;
}

void SpecificWorker::publish_door_diagnostics(const rc::DoorInstance& inst,
                                               const DoorObservation& observation,
                                               float free_energy)
{
    rc::dash::publish_belief_series({ts_plot_, ts_surprise_plot_, ts_cov_plot_, ts_res_plot_},
                                   {.node = inst.node_name,
                                     .free_energy  = free_energy,
                                     .fe_baseline  = inst.fe_baseline,
                                     .fe_surprise  = inst.fe_surprise,
                                     .uncertainty  = belief_uncertainty(inst),
                                     .residual_pts = static_cast<float>(inst.dbg_resid_pts)});

    if (fitter_->should_log(inst))
        std::print("[{}] series: FE={:.4f} U(Σ)={:.3f} res={}\n",
                   inst.node_name, free_energy, belief_uncertainty(inst), observation.residual_pts.size());
}

void SpecificWorker::publish_door_intentions(rc::DoorInstance& inst,
                                              uint64_t node_id,
                                              const DoorObservation& observation,
                                              float free_energy)
{
    // step_epistemic now owns the propose-vs-withdraw decision via the planner's ΔH (it withdraws
    // when the expected information gain falls below threshold), so it runs unconditionally rather
    // than being gated on the legacy coverage-deficit proxy.
    if (auto node_opt = G->get_node(node_id); node_opt.has_value())
    {
        step_epistemic(inst, node_opt.value());
        log_epistemic_transition(inst);
    }
}

// ─── Initialisation helpers ──────────────────────────────────────────────────

void SpecificWorker::load_config(const ConfigLoader& cfg)
{
    cfg_ = rc::load_door_config(cfg);
}


// ─── Per-cycle steps ─────────────────────────────────────────────────────────


void SpecificWorker::step_convergence(rc::DoorInstance& inst,
                                       DSR::Node& node,
                                       float free_energy)
{
    // Convergence on STATE stability, not |ΔFE|: the free energy keeps jittering with queue
    // churn / point-count even when the fitted geometry is settled, so it never latched. Track
    // how much the accepted state moved between cycles instead.
    const auto& s = inst.model.state();
    const auto& p = inst.prev_conv_state;
    // phi is in the sum so a door SWINGING cannot read as "converged" (its centre and yaw move together
    // in a way the other terms alone under-report). Contributes exactly 0 while phi is pinned.
    const float state_delta = inst.has_prev_conv_state
        ? (std::abs(s.cx - p.cx) + std::abs(s.cy - p.cy) + std::abs(s.w - p.w) +
           std::abs(s.h - p.h) + std::abs(s.yaw - p.yaw) + std::abs(s.phi - p.phi))
        : std::numeric_limits<float>::max();
    inst.prev_conv_state = s;
    inst.has_prev_conv_state = true;

    // The convergence RULE (counter, verdict, graph writes) is shared — common/obj/convergence.h. What stays
    // here is the one per-object part: which DOFs enter state_delta, above.
    const float model_uncertainty = belief_uncertainty(inst);
    const bool stable_edge = rc::converge::step(*G, node, state_delta, model_uncertainty,
                                                inst.frames_converged, inst.model_stable,
                                                {cfg_.state_eps, cfg_.K_stable});
    if (fitter_->should_log(inst))
    {
        const auto& cs = inst.model.state();
        std::print("[{}] convergence: pos=({:.2f},{:.2f}) yaw={:.2f} Δstate={:.4f} stable={}/{} U(Σ)={:.3f}m\n",
                   inst.node_name, cs.cx, cs.cy, cs.yaw, state_delta, inst.frames_converged, cfg_.K_stable,
                   model_uncertainty);
    }

    if (stable_edge and inst.model_stable)
        std::print("door_concept: node '{}' STABLE (F={:.4f})\n", inst.node_name, free_energy);
}

void SpecificWorker::step_epistemic(rc::DoorInstance& inst, DSR::Node& node)
{
    // Peripheral hypothesis (Part C-birth) — GLANCE or GO-AND-CHECK, decided by whether it has a RANGE.
    //
    //   · bearing only  → ORIENT: rotate in place down the ray. There is no trustworthy (x,y) to drive to,
    //                     so proposing one would be inventing a destination. The broad along-ray Σ already
    //                     makes the gain high, and one glance usually collapses it.
    //   · with a range  → the normal Servo path: the hypothesis has a real (x,y), so the planner's viewpoint
    //                     is meaningful and the affordance becomes "GO THERE and check". This is the rung the
    //                     range cascade unlocks (door_bearing_range.h).
    //
    // Note this reads `hypothesis_range_known`, not the σ, deliberately: how MUCH the range is trusted is
    // already carried by Σ into the planner's gain and stand-off, so a second σ test here would double-count.
    const bool orient = inst.is_bearing_hypothesis and not inst.hypothesis_range_known;

    // The cycle around the planner call is SHARED (common/epistemic_step). ★For door the important part of
    // that move is the FIRST bail path: the not-yet-initialised belief used to `return` bare, leaving a
    // just-completed affordance Completed for ever. Only the is_finite path held the offer; now both do,
    // because the shared step has exactly one bail path.
    rc::epistemic::StepHooks<rc::EpistemicProposal> hooks;

    hooks.compute = [&]() -> std::optional<rc::EpistemicProposal>
    {
        // Σ-based D-optimal NBV from the belief. Skip until the belief has seen its first frame (else Σ is
        // the broad prior and the proposal is moot).
        if (not inst.ai2_initialized)
            return std::nullopt;
        // Obstacles = the other fitted objects PLUS THE WALLS. The walls are what makes an edge-on view of a
        // door impossible, and without them the planner sent the affordance to a point ON the wall: the leaf's
        // ±x faces have normals lying in the wall plane, so their viewpoints sit several metres along it (live
        // 2026-08-07: `face=-x d=3.67m`). Furniture alone could never say that — `collect_graph_obstacles`
        // reads only `object`/`box` nodes. With the wall present, a grazing sightline has to traverse metres
        // of it and `visible_fraction` collapses continuously, so "view the door from a cone about its normal"
        // is a CONSEQUENCE of the geometry rather than an angular rule anyone had to write down.
        //
        // Every known aperture is punched out of the run, this door's included — otherwise the wall would
        // block the honest head-on views too, since the leaf sits inside the wall's own footprint. Other doors
        // are gaps as well: you really can see through a doorway across the room.
        auto obstacles = rc::nbv::collect_graph_obstacles(*G, inner_eigen_.get(), inst.node_id);
        if (fitter_->has_room_polygon())
        {
            std::vector<rc::nbv::WallGap> gaps;
            gaps.reserve(fitter_->instances().size());
            for (const auto& [oid, other] : fitter_->instances())
            {
                if (other.is_bearing_hypothesis or not other.ai2_initialized)
                    continue;   // no trustworthy aperture yet — punching a hole here would invent a sightline
                const auto& os = other.model.state();
                gaps.push_back({Eigen::Vector2f(os.ap_cx, os.ap_cy), 0.5f * os.w});
            }
            // Wall depth = the depth of material this aperture is cut through, i.e. the leaf's own fitted
            // thickness. A derived physical quantity, not a new knob — and occlusion by a plane is insensitive
            // to it anyway (a ray either crosses the plane or runs along it).
            const auto walls = rc::nbv::wall_obstacles(fitter_->room_polygon(),
                                                       std::max(1e-3f, inst.model.state().thickness), gaps);
            obstacles.insert(obstacles.end(), walls.begin(), walls.end());
        }
        nbv_obstacle_count_ = static_cast<int>(obstacles.size());   // 0 walls ⇒ the room polygon never arrived

        rc::nbv::Plan nbv_plan;
        // The existence question rides in the same gain (nats): high for a door only the 360 has reported, ~0 once the ZED settled it.
        const rc::ExistenceLook exist_look{inst.existence.p_exists(), cfg_.exist_detection_prob, cfg_.exist_clutter_prob};
        rc::EpistemicProposal prop =
            epistemic_planner_.compute(inst.ai2_belief, cfg_.ai2_range_noise_lat_per_m, cfg_.ai2_sigma_base_m,
                                       rc::nbv::sensor_from_graph(*G, inner_eigen_.get()), obstacles,
                                       fitter_->room_polygon(), &nbv_plan, &exist_look);
        log_nbv_decision(inst, nbv_plan, prop);   // logs REFUSED proposals too — keep it before the check
        // Degenerate (non-finite) fit — retry next cycle.
        if (not prop.valid or not prop.is_finite())
            return std::nullopt;
        if (orient)
            prop.epistemic_target_yaw_rad = inst.hypothesis_azimuth;
        return prop;
    };

    // Write attributes to the door node (read by legacy consumers)
    hooks.record = [&](const rc::EpistemicProposal& published, float /*raw_gain*/)
    { scene_graph_->write_epistemic_proposal(node, published); };

    hooks.on_affordance_created = [this] { trigger_graph_layout_twopi(); };

    rc::epistemic::step(inst, *G,
                        {.cooldown_cycles = cfg_.epistemic_cooldown_cycles, .orient_mode = orient},
                        hooks);
}


// ═══ THE THREE PRAGMATIC AFFORDANCES: approach / open / cross ════════════════════════════════════
//
// See door_pragmatics.h for the model and common/pragmatic_affordance for the protocol. What lives here
// is the wiring: measure the preconditions, publish them on the door node so the contracts have
// something to hold a predicate against, offer or withdraw each affordance, and — the one side effect
// this agent has on the world — ask a provider to move the leaf when the consumer claims `open`.

// What the agent believes about INTERACTING with one door, this cycle. Every field carries its own
// `*_known`: "could not compute" is a different fact from "computed and it is false", and conflating
// the two is how a standing offer gets withdrawn by a missing input.
rc::door::InteractionState SpecificWorker::compute_interaction_state(const rc::DoorInstance& inst) const
{
    rc::door::InteractionState st;
    const auto& pg = cfg_.pragmatic;
    // A bearing-only hypothesis has no located aperture — there is nothing to approach, open or cross
    // yet, only a direction to look in. That is the epistemic affordance's job, and it already has it.
    if (not inst.ai2_initialized or inst.is_bearing_hypothesis)
        return st;

    const auto& ap = inst.aperture;
    st.phi_rad       = inst.phi_est;
    st.phi_known     = true;
    // Task L: the span the logs and the cross precondition quote is the EXPECTATION over the whole leaf posterior (the same
    // curve P(passable) marginalises), not the span at its mean angle: a posterior split between a shut and an open leaf has
    // a mean angle at which neither is true. Without a curve (a mirror's memory) it falls back to the mean angle.
    st.clear_width_m = inst.phi_curve_log.empty() ? rc::door::clear_width(ap.w, inst.phi_est)
                                                  : rc::door::clear_width_expected(inst.phi_curve_log, ap.w);

    // ── P(passable) ──────────────────────────────────────────────────────────────────────────────
    // MARGINALISED over the leaf-angle posterior, never read off its argmax: measured 2026-09-10 the
    // phi support curve is nearly flat (peak 0.007–0.245), so its argmax is noise, and a parameter the
    // data does not identify must not be allowed to decide whether an affordance exists.
    const float needed = pg.robot_passage_width_m + pg.passage_margin_m;
    // ★Task E: …AND OVER THE APERTURE WIDTH'S OWN (reported, honest) σ: an estimate the mask disagrees with lowers P(passable)
    //   (and, in the approach offer below, the predicted P(complete)) continuously instead of reading exactly yes or no.
    const float sigma_w_rep = std::sqrt(std::max(0.0f, inst.ai2_belief.covariance_reported()(1, 1)));
    // ★ Task 11 fix round 1: IN LOG SPACE (rc::door::passable_logit over the posterior's log weights). The linear float
    //   sum underflowed every passable angle of a collapsed curve to exactly 0 — an absorbing "certainly shut" that no
    //   later evidence could move. The log-odds is kept beside p_open; p_open = σ(it).
    if (const auto l = rc::door::passable_logit(inst.phi_curve_log, ap.w, needed, sigma_w_rep); l.has_value())
    {
        st.p_open_logit = *l;
        st.p_open = static_cast<float>(rc::door::sigmoid(*l));
        st.p_open_known = true;
        st.p_open_from_curve = true;
    }
    else if (const auto p = rc::door::passable_prob(inst.phi_curve, ap.w, needed); p.has_value())
    {
        st.p_open = *p;   // a curve carried without its log weights (a mirror's memory): the linear form
        st.p_open_logit = std::log(std::max(1e-300, double(*p))) - std::log(std::max(1e-300, 1.0 - double(*p)));
        st.p_open_known = true;
        st.p_open_from_curve = true;
    }
    else if (pg.phi_sigma_rad > 0.0f and not inst.phi_from_command)
    {
        // ★NOT REACHED WHILE phi_est IS OUR OWN COMMAND. See DoorInstance::phi_from_command: with no
        // curve AND an armed anticipation, marginalising over N(phi_est, sigma) would report the door
        // wide open because we ASKED for it to be, satisfy the `open` contract's own completion
        // predicate, and return Satisfied for a leaf that may not have moved. p_open_known then stays
        // false, write_interaction_state publishes the neutral 0, and `open` can only complete on a real
        // observation — or time out, which is the honest outcome when nobody ever saw the door open.
        // ★NO CURVE THIS CYCLE IS THE NORMAL CASE AT THE MOMENT IT MATTERS, which is why this branch is
        // not a fallback but part of the design. estimate_phi() clears phi_curve and refills it only from
        // a fresh mask — and this agent's own config records that the ADE20K posterior over a plainly
        // visible closed door collapses 0.995 → 0.048 as the robot CLOSES on it. So the door stops being
        // segmented exactly when the robot is in the actuation zone: read off the curve alone, `open`
        // would vanish at the only range from which it can be taken.
        // What stands in is the BELIEF about the angle, which is maintained across cycles: phi_est is a
        // rate-limited posterior mean carrying estimate_phi's own persistence prior. Its width is not
        // tracked, so it is supplied (phi_sigma_rad) and the SAME passability question is asked under a
        // discretised Gaussian about phi_est. Still a marginalisation, still no angle threshold — the
        // difference from the branch above is only where the weights come from, and the published
        // door_open_prob_from_curve flag says which so the two can never be confused in a log.
        std::vector<std::pair<float, float>> prior_curve;
        prior_curve.reserve(17);
        for (int i = -8; i <= 8; ++i)
        {
            const float z   = 0.25f * static_cast<float>(i);            // ±2σ in quarter-σ steps
            const float phi = std::clamp(inst.phi_est + z * pg.phi_sigma_rad,
                                          0.0f, static_cast<float>(M_PI) * 0.5f);
            prior_curve.emplace_back(phi, std::exp(-0.5f * z * z));
        }
        if (const auto p = rc::door::passable_prob(prior_curve, ap.w, needed); p.has_value())
        {
            st.p_open = *p;
            st.p_open_logit = std::log(std::max(1e-300, double(*p))) - std::log(std::max(1e-300, 1.0 - double(*p)));
            st.p_open_known = true;
            st.p_open_from_curve = false;
        }
    }

    // ── P(the robot is in the actuation zone), and which SIDE it is on ───────────────────────────
    if (not inner_eigen_)
        return st;
    const auto rtb = inner_eigen_->get_transformation_matrix(rc::room::current_room_frame(*G), "body", 0);
    if (not rtb.has_value())
        return st;   // ALWAYS check the optional (CLAUDE.md); no pose ⇒ nothing below is knowable
    const Eigen::Vector2f robot(static_cast<float>(rtb.value()(0, 3)),
                                 static_cast<float>(rtb.value()(1, 3)));
    const Eigen::Vector2f centre = ap.centre_xy();
    const Eigen::Vector2f normal = ap.across_u();
    const Eigen::Vector2f delta  = robot - centre;

    st.robot_side    = normal.dot(delta);      // signed distance to the aperture PLANE
    st.robot_range_m = delta.norm();

    // σ of that range: the door's own along-wall uncertainty projected onto the radial direction, plus
    // the robot↔door pose-chain uncertainty. The floor is what keeps this a probability rather than a
    // step function — with σ == 0, P(d ≤ R) is exactly the distance threshold this design avoids.
    const Eigen::Vector2f radial = st.robot_range_m > 1e-4f ? Eigen::Vector2f(delta / st.robot_range_m)
                                                             : normal;
    const float sigma_s  = std::sqrt(std::max(0.0f, inst.ai2_belief.covariance()(0, 0)));  // θ = [s,w,h]
    const float var_door = std::pow(radial.dot(ap.wall_u) * sigma_s, 2.0f);
    const float var_pose = radial.x() * radial.x() * inst.chain_cov_xx
                         + radial.y() * radial.y() * inst.chain_cov_yy;
    const float sigma    = std::sqrt(var_door
                                     + std::max(var_pose, pg.pose_sigma_floor_m * pg.pose_sigma_floor_m));
    // ★THE ACTUATION ZONE IS A BAND. Near edge = the leaf's own swept arc: a door hinged on the aperture
    // edge reaches a full aperture-width out from the wall plane at 90 degrees, and since `LeafState::swing`
    // is NEVER FITTED (door_fitter sets it from config and it stays there) we cannot claim it opens away
    // from us. A robot inside that arc is struck by the door it just asked to be opened. Far edge = beyond
    // it a request is not unambiguously about THIS door. Both edges are physical; the old disc modelled
    // only the far one, so `open` stayed on offer right up against the leaf — failing toward the danger.
    const float d_min = rc::door::safe_standoff(ap.w, pg);
    const float d_max = std::max(pg.actuation_reach_m, d_min + 0.30f);   // a band, never empty
    st.safe_standoff_m = d_min;
    // Stand in the MIDDLE of the actuation band: furthest from both failure modes (inside the leaf's arc on
    // one side, too far to be about this door on the other) and so the most tolerant of the pose error that
    // put us there. ★NOT ApproachStandoffM when that is inside the arc — a configured distance cannot know
    // how wide this particular door is.
    st.approach_standoff_m = 0.5f * (d_min + d_max);
    // ★WHICH SIDE is a REGION question, not a half-plane one (live try25/28: the robot, inside room_1 but
    // beyond the plane of door_24's partition, was sent to (1.84,2.78) — outside the room). Stand on the side
    // whose stand-off is in the same region of the room polygon as the robot; see rc::door::approach_side.
    st.approach_side = rc::door::approach_side(ap, robot, st.robot_side, st.approach_standoff_m,
                                               fitter_->room_polygon());
    // ★AT THE DOOR INCLUDES THE VIEWING ANGLE (tourP8): range alone put the robot "in band" 58° off the normal, 0.69 m from
    //  the wall, from where the doorway could not be seen. The opening must be in front: P(w·|cos θ| ≥ body) under the same σ.
    const float cos_t = rc::door::facing_cos(delta.x(), delta.y(), normal.x(), normal.y());
    st.p_facing   = rc::door::prob_facing_opening(cos_t, ap.w, needed, sigma);
    st.facing_deg = std::acos(std::clamp(std::abs(cos_t), 0.0f, 1.0f)) * 57.2958f;
    st.p_band        = rc::door::prob_in_band(st.robot_range_m, d_min, d_max, sigma) * st.p_facing;
    // Marginalise the near edge over the door's state (see InteractionState::p_reach). Without a leaf-angle
    // posterior this cycle the door is treated as one that may need swinging: the band alone, as before.
    const float p_open_now = st.p_open_known ? std::clamp(st.p_open, 0.0f, 1.0f) : 0.0f;
    st.p_within      = rc::door::prob_within(st.robot_range_m, d_max, sigma) * st.p_facing;
    st.p_reach       = p_open_now * st.p_within
                     + (1.0f - p_open_now) * st.p_band;
    st.p_reach_known = true;
    st.range_sigma_m = sigma;

    // ── crossing progress, signed against the side latched when the claim was made ───────────────
    // "Crossed" is a CHANGE of side, not a position, so it is only defined while a claim is outstanding.
    // Outside one it stays 0 and unknown, and write_interaction_state publishes the neutral 0.
    if (std::isfinite(inst.cross_side0))
    {
        const float sgn = inst.cross_side0 < 0.0f ? -1.0f : +1.0f;
        st.crossing_progress = std::clamp(-sgn * st.robot_side
                                          / std::max(1e-3f, pg.cross_clearance_m), -1.0f, 1.0f);
        st.crossing_known = true;
    }
    return st;
}

// Decision log: the instance is being torn down — every affordance still offered is withdrawn WITH it (the node goes with
// the door), and the per-door transition memory is dropped so it cannot grow for ever.
void SpecificWorker::log_instance_teardown(const rc::DoorInstance& inst, const char* why)
{
    if (decision_log_.enabled())
    {
        using K = rc::door::DoorDecisionKind;
        const std::pair<const char*, const rc::pragmatic::PragmaticAffordance*> pr[] = {
            {"approach", &inst.aff_approach}, {"open", &inst.aff_open}, {"cross", &inst.aff_cross}, {"close", &inst.aff_close}};
        for (const auto& [kind, aff] : pr)
            if (aff->is_offered())
                decision_log_.record(K::AffWithdrawn, inst.node_id, inst.node_name, aff->epoch(), aff->last_p(), 0, 0,
                                     std::string(kind) + ":" + why);
        if (inst.affordance.node_id() != 0)
            decision_log_.record(K::AffWithdrawn, inst.node_id, inst.node_name, 0, 0, 0, 0, std::string("epistemic:") + why);
    }
    for (const char* k : {"approach", "open", "cross", "close", "epistemic"})
        aff_seen_.erase({inst.node_id, k});
}

// Decision log: the epistemic aff_door_N (ObjectAffordance) offered / withdrawn, seen after step_epistemic.
void SpecificWorker::log_epistemic_transition(const rc::DoorInstance& inst)
{
    if (not decision_log_.enabled())
        return;
    auto& seen = aff_seen_[{inst.node_id, "epistemic"}];
    const bool offered = inst.affordance.node_id() != 0;
    if (offered != seen.offered)
        decision_log_.record(offered ? rc::door::DoorDecisionKind::AffOffered : rc::door::DoorDecisionKind::AffWithdrawn,
                             inst.node_id, inst.node_name, 0, 0, 0, 0, "epistemic");
    seen.offered = offered;
}

// Decision log: an affordance's state TRANSITIONS, once each, as seen right after the cycle's update(). OFFERED/WITHDRAWN
// follow is_offered() (the node exists), EXECUTING follows claimed() (the consumer took it). a = the epoch on our node,
// b = P(precondition) last measured, c = P(claimed epoch) unused (0). Pure observation — decides nothing.
void SpecificWorker::log_affordance_transitions(const rc::DoorInstance& inst, const char* kind,
                                                const rc::pragmatic::PragmaticAffordance& aff)
{
    if (not decision_log_.enabled())
        return;
    auto& seen = aff_seen_[{inst.node_id, kind}];
    using K = rc::door::DoorDecisionKind;
    const bool offered = aff.is_offered(), claimed = aff.claimed();
    if (offered and not seen.offered)
        decision_log_.record(K::AffOffered, inst.node_id, inst.node_name, aff.epoch(), aff.last_p(), 0, 0, kind);
    if (claimed and not seen.claimed)
        decision_log_.record(K::AffExecuting, inst.node_id, inst.node_name, aff.epoch(), aff.last_p(), 0, 0, kind);
    if (not offered and seen.offered)
        decision_log_.record(K::AffWithdrawn, inst.node_id, inst.node_name, aff.epoch(), aff.last_p(), 0, 0, kind);
    seen = {offered, claimed};
}

// consume_completion() + the decision-log row (a = epoch, b = P(precondition), c = the outcome code).
bool SpecificWorker::consume_completion_logged(const rc::DoorInstance& inst, const char* kind,
                                               rc::pragmatic::PragmaticAffordance& aff)
{
    if (not aff.consume_completion())
        return false;
    decision_log_.record(rc::door::DoorDecisionKind::AffCompleted, inst.node_id, inst.node_name, aff.epoch(), aff.last_p(),
                       static_cast<double>(static_cast<int>(aff.outcome())), 0, kind);
    return true;
}

void SpecificWorker::step_pragmatic_affordances(rc::DoorInstance& inst)
{
    if (not G or not scene_graph_)
        return;
    auto node_opt = G->get_node(inst.node_id);
    if (not node_opt.has_value())
        return;   // the door node is gone this cycle; del_node_slot will reset the affordances

    const auto& pg = cfg_.pragmatic;
    if (not inst.pragmatic_inited)
    {
        const rc::pragmatic::PragmaticAffordance::Policy pol{.offer_prob    = pg.offer_prob,
                                                              .withdraw_prob = pg.withdraw_prob,
                                                              .stable_cycles = pg.stable_cycles};
        inst.aff_approach.init(G, inst.node_id, inst.node_name, "approach", pol);
        inst.aff_open    .init(G, inst.node_id, inst.node_name, "open",     pol);
        inst.aff_cross   .init(G, inst.node_id, inst.node_name, "cross",    pol);
        inst.aff_close   .init(G, inst.node_id, inst.node_name, "close",    pol);
        inst.pragmatic_inited = true;
    }

    // Measure once, publish once, decide on exactly what was published. The dashboard, the log and the
    // three offers therefore cannot disagree about which cycle they are describing.
    inst.interaction = compute_interaction_state(inst);
    const auto& st = inst.interaction;
    // Task 11 fix round 1 (cost): the leaf channel scores the second jamb every cycle only while the hinge matters.
    inst.hinge_question_live = inst.aff_approach.claimed() or inst.aff_open.claimed() or inst.aff_cross.claimed()
                               or inst.aff_close.claimed()
                               or (passage_anchor_.has_value()
                                   and (passage_anchor_->door_id == inst.node_id or passage_anchor_->door_name == inst.node_name));
    scene_graph_->write_interaction_state(node_opt.value(), st);
    G->update_node(node_opt.value());

    // The public assertion about this doorway, decided on the SAME measurement the offers below use.
    step_transitable_edge(inst, st);

    const bool located = st.phi_known and st.p_reach_known;   // a real aperture AND a robot pose

    // A node appearing or disappearing changes the graph's shape, so the viewer's layout has to be
    // recomputed — the same reason the epistemic step has an on_affordance_created hook.
    const bool had_any = inst.aff_approach.is_offered() or inst.aff_open.is_offered()
                      or inst.aff_cross.is_offered() or inst.aff_close.is_offered();

    // ── approach ─────────────────────────────────────────────────────────────────────────────────
    // ★UNCONDITIONAL (p = 1) for a located door: a door can always be walked up to. What falls away as
    // the robot arrives is its VALUE, not its availability — approaching a door you are already at is
    // worth nothing, and saying so as a value rather than as a withdrawal is what stops the
    // complete→re-offer→complete ping-pong a preconditioned version would produce at the doorway. Its
    // completion predicate is door_reach_prob. For a door believed SHUT that is the band, i.e. `open`'s own
    // precondition, so "approach succeeded" and "open became possible" are one measurement; for a door
    // believed OPEN the leaf's arc no longer matters and anywhere within reach counts (InteractionState::p_reach).
    {
        rc::pragmatic::Offer o;
        o.known = located;
        // ★D-A: the best standpoint in the actuation band on the robot's side (compute_interaction_state), given the
        // standpoints the controller has handed back — P(the contract completes)·P(inside the room)·P(not refused),
        // door_approach_pose.h. That score IS the offer's probability (fix round I-1): a band the hand-backs have used up
        // falls below WithdrawProb and the offer policy withdraws it, re-offering as the evidence decays. It does not
        // depend on where the robot is, so it cannot ping-pong at the doorway (what o.p = 1 used to guard). A changed pose
        // bumps the epoch by itself.
        const std::int64_t now_ms = static_cast<std::int64_t>(rc::provenance::now_ms());
        inst.approach_memory.prune(now_ms, pg.approach_refusal_memory_s);
        rc::door::ApproachProblem apb;
        apb.side = st.approach_side;
        apb.d_min = st.safe_standoff_m;
        apb.d_max = std::max(pg.actuation_reach_m, st.safe_standoff_m + 0.30f);
        // Fix round 2: the pose and the offer's AVAILABILITY at the NOMINAL σ (the pose floor) — independent of the live σ,
        // so no availability cliff and no argmax staircase churning the epoch; the LIVE σ AT ARRIVAL (the localiser's pose
        // covariance at the standpoint: translation + yaw × standoff — not the stale far-range mask chain) scales the VALUE.
        apb.sigma = pg.pose_sigma_floor_m;
        apb.sigma_arrival = rc::door::arrival_sigma(robot_pose_cov_in_room().value_or(Eigen::Matrix3f::Zero()),
                                                    0.5f * (apb.d_min + apb.d_max), pg.pose_sigma_floor_m);
        apb.needed_w = pg.robot_passage_width_m + pg.passage_margin_m;
        apb.sigma_w = std::sqrt(std::max(0.0f, inst.ai2_belief.covariance_reported()(1, 1)));   // Task E: P(complete) marginalised over the width's reported σ
        apb.memory_s = pg.approach_refusal_memory_s;
        apb.footprint_r = 0.5f * pg.robot_passage_width_m + pg.passage_margin_m;
        apb.room = G ? rc::room::current_room_frame(*G) : std::string{};
        apb.polygon = fitter_->room_polygon();
        const auto choice = rc::door::choose_approach_pose(inst.aperture, apb, inst.approach_memory, now_ms);
        const auto free_choice = rc::door::choose_approach_pose(inst.aperture, apb, rc::door::ApproachMemory{}, now_ms);
        const auto& pose = choice.pose;
        // The offer's probability: the share of the band's serviceability the hand-backs have left — the best score now
        // over the best score with no hand-back remembered. 1 with no hand-back (as before), → 0 as the band is used up.
        o.p   = rc::door::approach_offer_prob(choice, free_choice);
        o.x_m = pose.x; o.y_m = pose.y; o.yaw_rad = pose.yaw;
        inst.approach_offered_xy = {pose.x, pose.y};
        inst.approach_offered_valid = true;
        inst.approach_room = apb.room;
        inst.approach_side = apb.side;
        // M-3: the pose the consumer CLAIMED (what it is driving to, after its own repairs) is the one a hand-back refers to.
        if (inst.aff_approach.claimed() and inst.aff_approach.claim().has_value())
        {
            const auto& cl = inst.aff_approach.claim().value();
            if (not inst.approach_claimed_valid or cl.epoch != inst.approach_claimed_epoch)   // a NEW claim replaces the old
                inst.approach_claimed_epoch = cl.epoch;
            inst.approach_claimed_xy = {cl.x, cl.y};
            inst.approach_claimed_valid = true;
        }
        else if (inst.approach_claimed_valid and inst.aff_approach.epoch() != inst.approach_claimed_epoch)
            inst.approach_claimed_valid = false;   // the claimed proposal was replaced and nobody claims the new one
        // VALUE (nats): the preference for being at the door (DoorAffordance.ValueApproach) times the chance the approach can
        // COMPLETE — p_complete_live: from where the localisation will put the robot AND under the door belief's REPORTED width σ
        // (apb.sigma_w), so a poor estimate lowers it instead of withdrawing the offer. ★Task E: no longer × (1 − P(reach)).
        // That factor made the value vanish on arrival ("approaching a door you are already at is worth nothing") to stop a
        // complete→re-offer ping-pong in the OPEN contest — pragmatic offers are not in the open contest any more (they are
        // selected only when a passage names them) — and in a passage it would let the door's epistemic look (a few tenths of
        // a nat) out-bid an approach that is about to complete. The value is the goal's, not the distance's; the controller
        // charges distance itself (G = λ·dist − value, the selector's own score).
        o.value    = pg.value_approach * choice.p_complete_live;
        o.contract = rc::door::approach_contract(pg);
        o.why      = std::format("P(complete at arrival σ {:.2f}) {:.2f}; P(served)={:.2f} = P(complete) {:.2f} × P(inside) {:.2f} × P(not refused) {:.2f} at standoff {:.2f} "
                                 "lateral {:+.2f} ({} hand-back(s) remembered); robot P(reach)={:.2f} = P(open) {:.2f} mix of "
                                 "P(band) {:.2f} ⇒ value {:.3f} nats; range {:.2f} m "
                                 "in band [{:.2f},{:.2f}] σ {:.2f}, w {:.2f}", apb.sigma_arrival, choice.p_complete_live, choice.score, choice.p_complete, choice.p_inside,
                                 1.0f - choice.p_refused, choice.standoff, choice.lateral, inst.approach_memory.refused.size(),
                                 st.p_reach, st.p_open_known ? st.p_open : 0.0f, st.p_band, o.value, st.robot_range_m,
                                 st.safe_standoff_m, std::max(pg.actuation_reach_m, st.safe_standoff_m + 0.30f),
                                 st.range_sigma_m, inst.aperture.w);
        inst.aff_approach.update(o);
        log_affordance_transitions(inst, "approach", inst.aff_approach);
    }

    // ── open ─────────────────────────────────────────────────────────────────────────────────────
    // Possible when the robot is believed to be IN the actuation zone AND the aperture is believed NOT
    // passable. Both halves are needed: offering "open" from across the room is a request the consumer
    // would have to chain by itself, and offering it for a door already standing open is asking for a
    // no-op. The product is the joint probability under the two independent beliefs — no gate on either.
    {
        rc::pragmatic::Offer o;
        o.known = located and st.p_open_known and not cfg_.door_control_endpoint.empty();
        // The BAND, not the marginal p_reach: this is the step that swings the leaf, so the robot must be
        // outside its arc whatever the door is believed to be.
        o.p     = st.p_band * std::clamp(1.0f - st.p_open, 0.0f, 1.0f);
        if (inst.actuation_refused)
        {
            // ★A REFUSAL IS INFORMATION, and this kind is permanent: UnknownDoor / NotActuable mean never
            // ask again. Keeping the affordance on the wire would advertise an action that can only fail,
            // and an impossible offer with a plausible price is worse than silence.
            o.p   = 0.0f;
            o.why = "the provider cannot actuate this door: " + inst.actuation_note;
        }
        else
            o.why = std::format("P(in band)={:.2f} (incl. P(facing)={:.2f} at {:.0f} deg off the normal) × P(shut)={:.2f}{}",
                                 st.p_band, st.p_facing, st.facing_deg,
                                 1.0f - st.p_open, st.p_open_from_curve ? "" : " (phi from the belief, no curve)");
        const auto pose = rc::door::front_pose(inst.aperture, st.approach_side, st.approach_standoff_m);
        // Orient policy: the (x,y) is informational — the consumer does not navigate, it turns to face
        // the leaf and waits for the world to change. The YAW is the part that acts.
        o.x_m = pose.x; o.y_m = pose.y; o.yaw_rad = pose.yaw;
        o.value    = pg.value_open;
        o.contract = rc::door::open_contract(pg);
        inst.aff_open.update(o);
        log_affordance_transitions(inst, "open", inst.aff_open);
    }

    // ── cross ────────────────────────────────────────────────────────────────────────────────────
    // Possible exactly while the aperture is believed passable for THIS robot's body. The target is on
    // the far side and the heading carries the robot through; completion is the change of side.
    {
        rc::pragmatic::Offer o;
        o.known = located and st.p_open_known;
        // try102: passable AND at the door (rc::door::cross_offer_prob) — never from across the room.
        o.p     = rc::door::cross_offer_prob(st.p_open, st.p_within);
        if (not inst.aff_cross.claimed() or inst.cross_side_latched == 0.0f)
            inst.cross_side_latched = rc::door::interior_side(inst.aperture, st.approach_standoff_m,
                                                              fitter_->room_polygon(), st.approach_side);   // leave THIS room
        // ★ PAST THE LEAF, NOT INTO IT (2026-10-13). A leaf swung to the FAR side stands up to its own width deep
        //   beside the doorway; a through-pose at the fixed CrossStandoffM (1.2 m) on the centre line put the
        //   robot's footprint against it — live try36: `standpoint (0.03,-1.05) occupied` by the open DOOR_1 leaf,
        //   cross unreachable, robot stuck in room_2. The leaf's own pose says how deep it reaches beyond the
        //   aperture plane: stand past its tip by half the body + the swing clearance (never nearer than configured).
        // ★THE SWEPT RADIUS, NOT THE ESTIMATED TIP (2026-10-13). The swing SIDE is never fitted (config SwingDir,
        //   one value for every face) and phi flips; try43: room_2's face of DOOR_1 modelled its leaf in room_2, so the
        //   far-side depth read 0, the target sat 1.27 m out — inside the real leaf at 104° (`standpoint (0.04,-1.27)
        //   occupied`, 6361 returns), and the robot stalled IN the doorway. What the kinematics guarantee whatever the
        //   side and the angle: the leaf never reaches past its own width from the aperture plane.
        const float cross_standoff = std::max(pg.cross_standoff_m,
                                              2.0f * inst.leaf_pose.half_w + 0.5f * pg.robot_passage_width_m
                                              + pg.swing_clearance_m);
        inst.cross_standoff_m = cross_standoff;
        const auto pose = rc::door::through_pose(inst.aperture, inst.cross_side_latched, cross_standoff);
        o.x_m = pose.x; o.y_m = pose.y; o.yaw_rad = pose.yaw;
        o.value    = pg.value_cross;
        o.contract = rc::door::cross_contract(pg);
        o.why      = std::format("P(passable)={:.2f} (log-odds {:+.1f}) — clear span {:.2f} m at phi={:.2f} rad vs {:.2f} m of body"
                                 " × P(at the door)={:.2f} (P(facing) {:.2f} at {:.0f} deg; range {:.2f} m ≤ {:.2f}, σ {:.2f})",
                                 st.p_open, st.p_open_logit, st.clear_width_m, st.phi_rad,
                                 pg.robot_passage_width_m + pg.passage_margin_m, st.p_within, st.p_facing, st.facing_deg,
                                 st.robot_range_m,
                                 std::max(pg.actuation_reach_m, st.safe_standoff_m + 0.30f), st.range_sigma_m);
        inst.aff_cross.update(o);
        log_affordance_transitions(inst, "cross", inst.aff_cross);
    }

    // ── close ────────────────────────────────────────────────────────────────────────────────────
    // Possible while the door is believed OPEN and the robot believed CLEAR of the leaf's swing — beyond
    // safe_standoff (the swept arc + the body, the near edge of the actuation band): closing a door onto the
    // robot is the one request this must never make. No far bound: the building closes it, not the robot. The
    // product is the joint probability, no gate. Pragmatic, so never selected by default: the controller's
    // passage asks for it once the robot is through (controller_door_passage.h).
    {
        rc::pragmatic::Offer o;
        o.known = located and st.p_open_known and not cfg_.door_control_endpoint.empty();
        const float p_clear = 1.0f - rc::door::prob_within(st.robot_range_m, st.safe_standoff_m, st.range_sigma_m);
        o.p     = st.p_open * p_clear;
        if (inst.actuation_refused)
        {
            o.p   = 0.0f;   // never-ask-again from the provider: see `open`
            o.why = "the provider cannot actuate this door: " + inst.actuation_note;
        }
        else
            o.why = std::format("P(open)={:.2f} × P(clear of the swing)={:.2f} (range {:.2f} m vs swept {:.2f} m, σ {:.2f})",
                                st.p_open, p_clear, st.robot_range_m, st.safe_standoff_m, st.range_sigma_m);
        // Orient policy, as `open`: the robot does not navigate, it faces the leaf where it stands and waits.
        const auto pose = rc::door::front_pose(inst.aperture, st.approach_side,
                                               std::max(st.robot_range_m, st.safe_standoff_m));
        o.x_m = pose.x; o.y_m = pose.y; o.yaw_rad = pose.yaw;
        o.value    = pg.value_open;
        o.contract = rc::door::close_contract(pg);
        inst.aff_close.update(o);
        log_affordance_transitions(inst, "close", inst.aff_close);
    }

    if (const bool has_any = inst.aff_approach.is_offered() or inst.aff_open.is_offered()
                          or inst.aff_cross.is_offered() or inst.aff_close.is_offered(); has_any != had_any)
        trigger_graph_layout_twopi();

    // ── the side effects of a claim ──────────────────────────────────────────────────────────────
    // ★ONE-SHOTS, taken from the affordance rather than recomputed from the flags. `open` has a real
    // effect in the world; firing it per cycle while the claim stands would queue a request per cycle.
    // ★A CLAIM IS SERVED, NOT CONSUMED-AND-FORGOTTEN (2026-10-13). consume_claim() is a one-shot; if the claim
    // could not be acted on the cycle it landed, the request used to be lost for good and the claim timed out
    // (try43 leg 1: two timeouts, the operator had to open DOOR_1 by hand). It stays unserved until it is
    // either acted on or the consumer lets it go.
    if (inst.aff_open.consume_claim())
        inst.open_claim_unserved = true;
    if (inst.open_claim_unserved and not inst.aff_open.claimed())
        inst.open_claim_unserved = false;
    if (inst.open_claim_unserved)
    {
        // ★PROTOCOL INVARIANT 2: ACT ON WHAT THE CONSUMER IS EXECUTING, NOT ON OUR LATEST PUBLICATION.
        // A stale epoch (executing_epoch < epistemic_target_epoch) was refused outright because we could not
        // tell which doorway the robot committed to. But the epoch is only a PROXY for that question, and for
        // `open` it never resolves: the controller holds the epoch it committed to by design, while we bump
        // ours every time the front pose refines by 0.1 mm — so the claim stays stale for its whole life.
        // Ask the question itself: is the pose the consumer is actually driving to within actuation reach of
        // THIS aperture? Then the request is about this door whatever the epoch, and the provider door is
        // chosen by this door's own place (request_door_actuation), not by the robot's pose.
        // ★THE CLAIM NAMES THIS DOOR ALREADY: it is a claim on THIS instance's `open` node. The distance test that
        //   stood here (claimed pose within ActuationReachM of the aperture, else "NOT sending a door request") was a
        //   second proxy for the same question, and it answered it wrong whenever the controller's standpoint repair
        //   moved the claimed pose back: try100 leg 1 claimed (-0.69,2.42), 2.2 m from the aperture, the request was
        //   withheld for the whole 12 s claim, the open timed out and the mission failed door_wont_open — the door
        //   was opened only on the next claim. Whether the robot is close enough to pass is `cross`'s business.
        if (not pg.autonomous_actuation)
        {
            inst.open_claim_unserved = false;
            inst.actuation_note = "autonomous actuation is disabled (DoorAffordance.AutonomousActuation)";
            std::print("door_concept: [open] {} was CLAIMED but autonomous actuation is off — "
                       "the request must come from the strip button\n", inst.node_name);
        }
        else
        {
            inst.open_claim_unserved = false;   // served: one request per claim, whatever the provider says
            std::string why;
            // No provider id from here: the request path CHOOSES the provider door per door (nearest by
            // place); pg.actuation_provider_id is only its fallback when no frame resolves.
            if (request_door_actuation(inst, /*open=*/true, /*provider_id=*/std::string{},
                                        "robot needs to pass through", why))
                std::print("door_concept: [open] {} claimed → asked the provider to OPEN it "
                           "(request {} of this run)\n", inst.node_name, inst.actuation_requests);
            else
            {
                inst.actuation_note = why;
                std::print("door_concept: [open] {} claimed but the request was refused LOCALLY: {}\n",
                           inst.node_name, why);
            }
        }
    }
    // `close`, served like `open` (one request per claim). No stale-epoch question: the claimed pose is where the
    // robot already stands, beside THIS door.
    if (inst.aff_close.consume_claim())
        inst.close_claim_unserved = true;
    if (inst.close_claim_unserved and not inst.aff_close.claimed())
        inst.close_claim_unserved = false;
    if (inst.close_claim_unserved)
    {
        inst.close_claim_unserved = false;
        if (not pg.autonomous_actuation)
        {
            inst.actuation_note = "autonomous actuation is disabled (DoorAffordance.AutonomousActuation)";
            std::print("door_concept: [close] {} was CLAIMED but autonomous actuation is off\n", inst.node_name);
        }
        else if (std::string why; request_door_actuation(inst, /*open=*/false, /*provider_id=*/std::string{},
                                                          "the robot has gone through; closing the door behind it", why))
            std::print("door_concept: [close] {} claimed → asked the provider to CLOSE it (request {} of this run)\n",
                       inst.node_name, inst.actuation_requests);
        else
        {
            inst.actuation_note = why;
            std::print("door_concept: [close] {} claimed but the request was refused LOCALLY: {}\n", inst.node_name, why);
        }
    }
    if (consume_completion_logged(inst, "close", inst.aff_close))
        std::print("door_concept: [close] {} finished, outcome = {} (P(passable) is now {:.2f})\n", inst.node_name,
                   rc::affordance::to_string(inst.aff_close.outcome()), st.p_open);
    if (consume_completion_logged(inst, "open", inst.aff_open))
        std::print("door_concept: [open] {} finished, outcome = {} (P(passable) is now {:.2f}{})\n",
                   inst.node_name, rc::affordance::to_string(inst.aff_open.outcome()),
                   st.p_open, st.p_open_known ? "" : ", unmeasured");

    if (inst.aff_cross.consume_claim())
    {
        // ★LATCH THE SIDE, ONCE, HERE. door_crossing_progress is signed against it, so re-latching per
        // cycle would make the measure its own input and read 0 for ever — the crossing could never
        // complete and the contract would always time out.
        //
        // ★AND LATCH IT AGAINST THE CONSUMER'S OWN TARGET WHERE WE HAVE IT (protocol invariant 2: the
        // producer measures progress against executing_target_*, NEVER against its own latest
        // publication). The through-pose we published says which side we EXPECT the robot to end up on;
        // the claim says which side it is actually driving to, after its own repairs. If the consumer
        // repaired our target to the opposite side of the aperture — which it may legitimately do when
        // ours is unreachable — then progress signed against our expectation would count the crossing
        // backwards and the contract could only ever time out. With no edge (a pre-rollout consumer) we
        // fall back to the robot's measured side, which is what this did before.
        inst.cross_side0 = st.robot_side;
        if (const auto& c = inst.aff_cross.claim(); c.has_value())
        {
            const Eigen::Vector2f n = inst.aperture.across_u();
            const Eigen::Vector2f centre = inst.aperture.centre_xy();
            const float target_side = n.dot(Eigen::Vector2f(c->x, c->y) - centre);
            // The robot must END on the far side of the plane from where it starts. If the consumer's
            // committed target is on the SAME side as the robot, this claim is not a crossing at all.
            if (target_side * st.robot_side > 0.0f)
                std::print("door_concept: [cross] ⚠{} claimed, but the consumer is driving to "
                           "({:.2f},{:.2f}), which is on the robot's OWN side of the aperture "
                           "(side {:+.2f} vs robot {:+.2f}) — this claim cannot complete as a crossing\n",
                           inst.node_name, c->x, c->y, target_side, st.robot_side);
        }
        std::print("door_concept: [cross] {} claimed from side {:+.2f} m — through-pose ({:.2f},{:.2f})\n",
                   inst.node_name, inst.cross_side0,
                   rc::door::through_pose(inst.aperture, inst.cross_side_latched, inst.cross_standoff_m).x,
                   rc::door::through_pose(inst.aperture, inst.cross_side_latched, inst.cross_standoff_m).y);
    }
    if (consume_completion_logged(inst, "cross", inst.aff_cross))
    {
        std::print("door_concept: [cross] {} finished, outcome = {} (progress {:+.2f})\n",
                   inst.node_name, rc::affordance::to_string(inst.aff_cross.outcome()),
                   st.crossing_progress);
        inst.cross_side0 = std::numeric_limits<float>::quiet_NaN();
    }
    if (consume_completion_logged(inst, "approach", inst.aff_approach))
    {
        const auto outcome = inst.aff_approach.outcome();
        std::print("door_concept: [approach] {} finished, outcome = {} (P(reach)={:.2f})\n",
                   inst.node_name, rc::affordance::to_string(outcome), st.p_reach);
        // ★D-A: a hand-back names a standpoint the robot could not use — remember it as decaying evidence; the next offer
        // re-solves the pose around it (door_approach_pose.h) instead of re-arming the identical one.
        // M-4: OutsideRoom too — a standpoint the consumer could not place in its room is a standpoint it cannot use.
        // M-3: the CLAIMED pose (the consumer's, after its repairs), else the one offered.
        if ((outcome == rc::affordance::Outcome::Unreachable or outcome == rc::affordance::Outcome::Infeasible
             or outcome == rc::affordance::Outcome::OutsideRoom)
            and (inst.approach_claimed_valid or inst.approach_offered_valid))
        {
            const Eigen::Vector2f xy = inst.approach_claimed_valid ? inst.approach_claimed_xy : inst.approach_offered_xy;
            inst.approach_memory.refuse(inst.aperture, inst.approach_side, inst.approach_room, xy,
                                        static_cast<std::int64_t>(rc::provenance::now_ms()));
            std::print("door_concept: [approach] {} {} standpoint ({:.2f},{:.2f}) handed back — remembered as blocked "
                       "(decays on {:.0f} s); the next offer is re-solved around it\n", inst.node_name,
                       inst.approach_claimed_valid ? "claimed" : "offered", xy.x(), xy.y(),
                       cfg_.pragmatic.approach_refusal_memory_s);
        }
        inst.approach_claimed_valid = false;   // a completion ends that claim
    }
}

// ─── ONE request path: ask a provider to move a door ─────────────────────────────────────────────
//
// Called by the strip buttons AND by the `open` affordance when the consumer claims it. Everything that
// can go wrong about frames, identity and anticipation is decided here, once.
//
// ★NOTHING IN HERE EVER CONCLUDES THAT A DOOR IS OPEN. `Delivered` means the provider did what it was
// asked, not that the world changed; that stays a perceptual question, answered by door_open_prob from
// the image. Wiring a protocol event into a belief is the mistake the affordance work spent a month
// removing.
// The single request path, wrapped so the decision log sees EVERY request, including the ones refused locally.
bool SpecificWorker::request_door_actuation(rc::DoorInstance& inst, bool open, const std::string& provider_id,
                                             const std::string& purpose, std::string& why)
{
    const bool sent = request_door_actuation_impl(inst, open, provider_id, purpose, why);
    decision_log_.record(rc::door::DoorDecisionKind::ActuatorRequest, inst.node_id, inst.node_name, open ? 1.0 : 0.0, sent ? 1.0 : 0.0,
                         double(inst.actuation_requests), 0,
                         purpose + "|" + (sent ? inst.actuation_note : why));
    return sent;
}

bool SpecificWorker::request_door_actuation_impl(rc::DoorInstance& inst, bool open,
                                                  const std::string& provider_id,
                                                  const std::string& purpose, std::string& why)
{
    why.clear();
    if (cfg_.door_control_endpoint.empty())
    { why = "no DoorControl endpoint configured"; return false; }

    // ★THE APERTURE, NOT THE LEAF. What a provider is asked to move is identified by the DOORWAY — the
    // static hole in the wall — and the leaf is the part that moves. At phi == 0 the two coincide
    // exactly, so this is identical to the previous behaviour today, and it stops the quoted pose
    // wandering by w/2 the moment a leaf actually swings, which is precisely when we ask again.
    const auto& ap        = inst.aperture;
    const Eigen::Vector2f xy = ap.centre_xy();
    const float yaw       = ap.yaw();
    const float width_m   = ap.w;

    // ── world←room: WHERE IS OUR DOOR IN THE PROVIDER'S FRAME ─────────────────────────────────────────
    // ★ROOT IS NOT A WORLD FRAME (root-caused 2026-10-12). The RT tree is root→Shadow (static identity)
    // and Shadow→room_k (room_concept writes T_robot←room on that edge), so get_transformation_matrix(
    // "root", room) is the ROBOT's pose inverse — a frame that moves with the robot. That is the "6.45 m
    // and ~90° out" measured on door_3, and it is why a world pose quoted through root wandered by metres.
    // It is no longer used.
    // ★AND THE LEARNED REGISTRATION WAS POISONED. Every request that named the CONFIGURED id
    // (ActuationProviderId = DOOR_1) recorded "this door ↔ DOOR_1" as an operator correspondence —
    // including requests for the OTHER door — and pairs from different rooms' frames were fitted together.
    // Live logs show 2-pair fits with 1.07-1.47 m residual being used anyway; door_19 went out at
    // (-5268, 10327) mm against DOOR_1's true (-4769, 9380).
    // Sources, best first:
    //   1. the simulator's witness: world←body from robot_gt_* (sim only; absent on the real robot) ∘
    //      body←room from the graph. Error = the localiser's error, centimetres, in ANY room.
    //   2. the learned registration, but only pairs recorded IN THIS ROOM by an OPERATOR pick.
    const std::string room_name = rc::room::current_room_frame(*G);
    Eigen::Matrix4d world_T_room = Eigen::Matrix4d::Identity();
    bool have_tf = false;
    std::string tf_source;
    if (inner_eigen_)
        if (const auto robots = G->get_nodes_by_type("robot"); not robots.empty())
        {
            const auto gx = G->get_attrib_by_name<robot_gt_x_att>(robots.front());
            const auto gy = G->get_attrib_by_name<robot_gt_y_att>(robots.front());
            const auto ga = G->get_attrib_by_name<robot_gt_angle_att>(robots.front());
            if (gx.has_value() and gy.has_value() and ga.has_value())
                if (const auto rtb = inner_eigen_->get_transformation_matrix(room_name, "body", 0); rtb.has_value())
                {
                    world_T_room = rc::world_T_body_from_sim_gt(gx.value(), gy.value(), ga.value())
                                 * rtb.value().matrix().inverse();
                    have_tf = true;
                    tf_source = std::format("sim witness: robot_gt ({:.3f}, {:.3f}) m ∘ body←{}", gx.value(),
                                            gy.value(), room_name);
                }
        }
    if (not have_tf)
        if (const auto& f = door_registration_.solve(room_name); f.valid)
            if (const auto reg = door_registration_.world_T_room(); reg.has_value())
            {
                world_T_room = *reg;
                have_tf = true;
                tf_source = std::format("learned registration of {} ({} pair(s), residual {:.3f} m{})", room_name,
                                        f.n_pairs, f.residual_m, f.provisional ? ", PROVISIONAL" : "");
            }

    // ── WHICH PROVIDER DOOR: operator pick > nearest by place > configured fallback ──────────────────
    // ★A CONFIGURED ID NO LONGER OVERRIDES EVERY DOOR. It used to be passed as if an operator had picked
    // it, so every door this agent opened — including the second door of room 1 — was sent to DOOR_1.
    // Now the provider door is CHOSEN PER DOOR, by world position, and the choice is logged with its
    // distance. The configured id is consulted only when there is no frame to match in at all.
    const bool operator_pick = not provider_id.empty();
    std::string chosen_id = provider_id;
    std::string id_reason = operator_pick ? "operator pick (strip)" : "";
    const Eigen::Vector2f ours_world = (world_T_room * Eigen::Vector4d(xy.x(), xy.y(), 0.0, 1.0))
                                           .head<2>().cast<float>();
    if (not operator_pick and have_tf)
    {
        if (door_actuator_.caps().doors.empty())
            actuator_refresh_caps();   // the strip UI normally refreshes these; a headless run may not have
        if (const auto m = rc::nearest_provider_door(door_actuator_.caps().doors, ours_world); m.has_value())
        {
            if (m->dist_m > cfg_.pragmatic.actuation_match_max_m)
            {
                why = std::format("nearest advertised door {} is {:.2f} m from {} at world ({:.2f}, {:.2f}) m "
                                  "(> DoorAffordance.ActuationMatchMaxM {:.2f}) — not asking to open some other "
                                  "door [{}]", m->id, m->dist_m, inst.node_name, ours_world.x(), ours_world.y(),
                                  cfg_.pragmatic.actuation_match_max_m, tf_source);
                std::println("door_concept: [actuator] REFUSED locally: {}", why);
                return false;
            }
            chosen_id = m->id;
            id_reason = m->runner_up_m >= 0.0f
                ? std::format("nearest by place: {:.2f} m (runner-up {:.2f} m)", m->dist_m, m->runner_up_m)
                : std::format("nearest by place: {:.2f} m (only advertised door)", m->dist_m);
        }
        else
            id_reason = "provider advertises no ids — it resolves by the pose sent";
    }
    if (chosen_id.empty() and not have_tf)
    {
        if (cfg_.pragmatic.actuation_provider_id.empty())
        {
            why = std::format("no world←{} transform (no sim witness, no registration for this room) and no "
                              "DoorAffordance.ActuationProviderId fallback — pick a provider door in the strip",
                              room_name);
            return false;
        }
        chosen_id = cfg_.pragmatic.actuation_provider_id;
        id_reason = "⚠CONFIGURED FALLBACK (DoorAffordance.ActuationProviderId): no frame to match by place";
    }
    std::println("door_concept: [actuator] {} {} -> provider door '{}' — {}; our door at world ({:.3f}, {:.3f}) m "
                 "[{}]", open ? "OPEN" : "CLOSE", inst.node_name, chosen_id.empty() ? "(by place)" : chosen_id,
                 id_reason, ours_world.x(), ours_world.y(), have_tf ? tf_source : "no transform");

    door_pending_ = actuator_request(inst.node_name, xy, yaw, width_m, open, world_T_room,
                                     purpose, chosen_id);
    if (not door_pending_.has_value())
    { why = "no provider"; return false; }
    ++inst.actuation_requests;
    inst.actuation_note = door_pending_->state;

    // ★A REFUSAL IS INFORMATION, AND THE KINDS DIFFER (door_actuator.h). UnknownDoor / NotActuable mean
    // never ask again, and the `open` affordance reads this to stop offering an action that can only
    // fail. TemporarilyBusy and Declined are NOT latched: "later" and "a person said no this time" are
    // different beliefs from "this door cannot be actuated", and collapsing them into one would retire a
    // perfectly good affordance on a transient.
    if (door_pending_->settled)
    {
        const auto& st = door_pending_->state;
        // ★The state text is DoorActuator's refusal_text ("refused: unknown door" / "not actuable"); the
        // CamelCase enum names never appear in it, so this latch could not fire until 2026-10-12.
        if (st.find("unknown door") != std::string::npos or st.find("not actuable") != std::string::npos)
        {
            inst.actuation_refused = true;
            std::print("door_concept: [actuator] {} is NOT ACTUABLE by this provider ({}) — "
                       "the `open` affordance will stop being offered for it\n", inst.node_name, st);
        }
    }

    // ★A SUCCESSFUL BY-NAME REQUEST IS A LABELLED CORRESPONDENCE, and it is free. The operator has just
    // asserted "the door I believe here is the one you call DOOR_1" — a pair of poses in the two frames,
    // produced as a side effect of pressing a button. Two such pairs pin the rigid transform outright,
    // which is what makes the autonomous (by place) path possible at all: a robot deciding on its own to
    // open a door has nobody to pick a name from a list.
    // ★ONLY AN OPERATOR PICK IS A CORRESPONDENCE. A configured fallback id, or one this agent chose itself,
    // asserts nothing new — recording those is exactly how "door_21 ↔ DOOR_0" AND "door_21 ↔ DOOR_1" both
    // ended up in the file. And the pair is a fact about THIS room's frame only (Pair::room).
    if (operator_pick and not door_pending_->settled)
        for (const auto& d : door_actuator_.caps().doors)
            if (d.id == provider_id)
            {
                // The pair carries BOTH yaws, so a one-pair fit can recover rotation and the result is
                // identical after a restart instead of silently assuming none.
                door_registration_.add({xy, d.xy_mm / 1000.0f, inst.node_name, d.id, yaw, d.angle, room_name},
                                       "etc/door_world_registration.csv");
                const auto& f = door_registration_.solve(room_name);
                std::println("door_concept: [registration] {} pair(s) -> theta {:+.3f} rad "
                             "t ({:+.3f}, {:+.3f}) m, residual {:.3f} m{}",
                             f.n_pairs, f.theta, f.t.x(), f.t.y(), f.residual_m,
                             f.provisional ? "  ⚠PROVISIONAL (one pair: rotation from yaws)" : "");
                break;
            }

    // ★ARM THE ANTICIPATION. The request just issued is a prediction about the world: from now until the
    // swing completes, the leaf angle is a known function of time. Handing that to the estimator is what
    // stops the agent being surprised by its own action — with phi pinned, the predicted silhouette
    // stayed in the doorway the leaf had just left, all 420 samples went dark, and the existence channel
    // deleted the door BECAUSE it obeyed. The command enters as a PRIOR on phi, never as a licence to
    // suppress evidence: the estimate is still scored against the image every cycle and can refute it.
    if (not door_pending_->settled)
    {
        inst.phi_cmd_from   = inst.phi_est;
        inst.phi_cmd_to     = open ? static_cast<float>(M_PI) / 2.0f : 0.0f;
        inst.phi_cmd_rate   = cfg_.pragmatic.actuation_swing_rate;   // MUST match the bridge; see the config
        inst.phi_cmd_t0     = std::chrono::steady_clock::now();
        inst.phi_cmd_active = true;
        std::println("door_concept: [phi] anticipating {} swing {:.2f} -> {:.2f} rad at {:.2f} rad/s",
                     inst.node_name, inst.phi_cmd_from, inst.phi_cmd_to, inst.phi_cmd_rate);
    }
    return true;
}


// ═══ WHICH ROOM ARE WE IN — follow `current`, and let go of the room we leave ════════════════════

std::optional<std::uint64_t> SpecificWorker::resolve_current_room() const
{
    if (not G) return std::nullopt;
    // `current` is ltsm_agent's ONE output into the live graph: robot --[current]--> room. Read it by
    // EDGE TYPE rather than by walking from a robot node we would have to identify by name — the robot
    // node is "Shadow" on one platform and something else on the next, and a name is a second source of
    // truth that drifts. The destination's TYPE is the check that matters.
    std::optional<std::uint64_t> found;
    int n_room_edges = 0;
    for (const auto& e : G->get_edges_by_type("current"))
    {
        const auto dst = G->get_node(e.to());
        if (not dst.has_value() or dst.value().type() != "room")
            continue;                       // a `current` edge to something that is not a room is not ours to read
        ++n_room_edges;
        found = e.to();
    }
    // ★TWO `current` EDGES IS AMBIGUOUS, AND GUESSING IS WORSE THAN WAITING. It would mean the owner is
    // mid-write or two writers exist; either way picking one at random could let go of a room we are
    // still in. Report nothing, which the caller treats as "no information", not as a release.
    if (n_room_edges > 1)
    {
        static int warned = 0;
        if (warned++ % 120 == 0)
            std::print("door_concept: [room] {} `current` edges point at rooms — ambiguous, not following "
                       "any of them until exactly one remains\n", n_room_edges);
        return std::nullopt;
    }
    return found;
}

void SpecificWorker::step_room_following()
{
    if (not G) return;
    latch_recall_names();   // BEFORE the switch below: ltsm may retire the recall the moment the room is live again
    const auto current = resolve_current_room();

    // ★ABSENCE OF A `current` EDGE IS NOT A RELEASE, and this is the whole backward-compatibility
    // contract. Nothing in today's fleet publishes that edge, so `current` is nullopt everywhere and this
    // function must behave exactly as the old latch did — otherwise shipping it would make every agent
    // let go of every room on the first cycle. "Nobody has told me where I am" and "I have been told I am
    // somewhere else" are different facts and only the second one releases anything.
    if (not current.has_value())
    {
        if (room_node_id_ == 0)                       // the legacy first-resolution path, unchanged
        {
            // ★Not rooms.front(): with a proto-room in the graph that is a coin toss (unordered_set).
            if (const auto room = rc::room::current_room(*G); room.has_value())
            {
                room_node_id_ = *room;
                decision_log_.record(rc::door::DoorDecisionKind::RoomSwitch, room_node_id_, {}, 0.0, 0.0, 0.0, 0.0, "first resolution");
                enter_room_memory(room_node_id_, true);
            }
        }
        return;
    }

    if (room_node_id_ == *current)
    {
        step_pending_restore();   // a recalled room's doors, once its polygon is loaded
        // The crossed door was HELD at the switch and has been deleted with its room since: carry it across now.
        if (carry_pending_room_ == room_node_id_ and not lost_doors_.empty())
        {
            carry_pending_room_ = 0;
            std::optional<EntryMirror> adopted_from;
            std::uint64_t adopt = 0;
            if (auto rec = recreate_lost_door(room_node_id_); rec.has_value())
            {
                adopt = rec->first;
                adopted_from = std::move(rec->second);
            }
            lost_doors_.clear();
            install_carried_door(room_node_id_, adopt, adopted_from);
        }
        return;                                        // already following it; the common case
    }

    if (room_node_id_ == 0)
    {
        room_node_id_ = *current;
        std::print("door_concept: [room] following `current` → room node {}\n", room_node_id_);
        decision_log_.record(rc::door::DoorDecisionKind::RoomSwitch, room_node_id_, {}, 0.0, 0.0, 0.0, 0.0, "follow current");
        enter_room_memory(room_node_id_, true);
        return;
    }

    // We are anchored to one room and told we are in another. LET GO.
    release_room(room_node_id_, *current);
}

// ═══ ENTRY MIRRORS — the door a proto-room was entered through ════════════════════════════════════

void SpecificWorker::step_entry_mirrors()
{
    (void)current_room_name();   // keep the room-name cache warm: the room may be evicted before release_room
    if (not G or not fitter_ or not scene_graph_ or not inner_eigen_)
        return;

    // 1. Retire mirrors whose proto-room, source instance or own node is gone. A proto-room that was
    //    PROMOTED (edge removed) also ends its mirror: promotion is when the door becomes a real instance there.
    for (auto it = entry_mirrors_.begin(); it != entry_mirrors_.end();)
    {
        const auto& [room_id, mir] = *it;
        // ★ THE ROOM, NOT ITS `proto` EDGE (2026-10-11). `proto` is room_concept's "layout provisional" and may go
        //   before ltsm moves `current`; the mirror must survive until then, when release_room ADOPTS it as this
        //   room's door. Retiring it on promotion deleted the second face of ltsm's passage (live).
        const bool room_ok   = G->get_node(room_id).has_value();
        const bool source_ok = fitter_->instances().contains(mir.source_id) and G->get_node(mir.source_id).has_value();
        const bool mirror_ok = G->get_node(mir.node_id).has_value();
        // ★ A SOURCE THAT DIES WITH ITS ROOM DOES NOT END THE MIRROR (2026-10-12, eviction ON). ltsm moves `current`
        //   to the new room and then deletes the old one, door included; if that deletion reaches this cycle before
        //   the `current` move does, retiring here would delete the door release_room is about to adopt.
        const bool source_node_gone = not G->get_node(mir.source_id).has_value();
        if (room_ok and mirror_ok and (source_ok or source_node_gone))
        {
            ++it;
            continue;
        }
        if (mirror_ok)
            G->delete_node(mir.node_id);
        std::print("door_concept: [proto] entry mirror {} retired (proto-room {} {}, source {} {}, node {})\n",
                   mir.node_id, room_id, room_ok ? "ok" : "gone/promoted",
                   mir.source_id, source_ok ? "ok" : "gone", mirror_ok ? "deleted" : "already gone");
        it = entry_mirrors_.erase(it);
    }

    // 2. Birth: one mirror per proto-room, from the instance whose aperture CONTAINS the proto-room origin.
    //    room_concept puts that origin at the centre of the crossed aperture, read from THIS agent's door node,
    //    so for the right door the origin sits on its aperture segment and for every other door it does not.
    //    The containment test is geometric identity, not a tuned gate: along the aperture within the door's own
    //    half-width, across it within the same half-width (the fit may have refined the door since the
    //    proto-room froze its frame).
    for (const auto& room : G->get_nodes_by_type("room"))
    {
        if (not rc::room::is_proto(*G, room.id()) or entry_mirrors_.contains(room.id()))
            continue;
        // The room this agent is IN needs no mirror: its doors are its own instances (release_room adopted the
        // entry mirror). Without this the adopted door — whose aperture holds the proto origin by construction —
        // would be mirrored into its own room. A proto stays proto until room_concept promotes its LAYOUT.
        if (room.id() == room_node_id_)
            continue;

        std::optional<DSR::Node> best;
        Mat::RTMat best_T;
        double best_d = std::numeric_limits<double>::max();
        for (const auto& [door_id, inst] : fitter_->instances())
        {
            const auto dn = G->get_node(door_id);
            if (not dn.has_value() or rc::room::is_proto_mirror(*G, dn.value()))
                continue;
            // proto <- door, through the robot (both rooms hang from it). ts==0 on the main-thread instance.
            const auto T = inner_eigen_->get_transformation_matrix(room.name(), dn->name(), 0);
            if (not T.has_value())
                continue;
            // The proto origin in the DOOR frame: x along the aperture, y across it.
            const Eigen::Vector3d o = T->inverse().translation();
            const double half_w = 0.5 * G->get_attrib_by_name<width_m_att>(dn.value()).value_or(cfg_.door_prior_w_m);
            if (std::abs(o.x()) > half_w or std::abs(o.y()) > half_w)
                continue;
            if (const double d = o.head<2>().norm(); d < best_d)
            {
                best_d = d;
                best = dn;
                best_T = T.value();
            }
        }
        if (not best.has_value())
        {
            static int miss = 0;
            if (miss++ % 60 == 0)
                std::print("door_concept: [proto] proto-room '{}' id={} has no door of ours on its origin yet "
                           "(RT not written, or the crossed door is not an instance)\n", room.name(), room.id());
            continue;
        }

        const Eigen::Matrix3d R = best_T.rotation();
        // R = Rx(a)Ry(b)Rz(c) — cortex's rt_rotation_euler_xyz. Closed form, NOT Eigen::eulerAngles (see
        // room_concept/src/mount_calibrator.cpp for the decomposition it silently picks otherwise).
        const Eigen::Vector3f euler(static_cast<float>(std::atan2(-R(1, 2), R(2, 2))),
                                    static_cast<float>(std::asin(std::clamp(R(0, 2), -1.0, 1.0))),
                                    static_cast<float>(std::atan2(-R(0, 1), R(0, 0))));
        const Eigen::Vector3f t = best_T.translation().cast<float>();
        const auto names = reserved_names();
        if (const auto id = scene_graph_->create_entry_mirror(room, best.value(), t, euler, names); id != 0)
        {
            EntryMirror em{id, best->id()};
            em.name = best->name();   // the doorway's name: the mirror takes it once the source has left the graph
            const auto room_now = G->get_node(room_node_id_);
            const auto src_it = fitter_->instances().find(best->id());
            if (room_now.has_value() and src_it != fitter_->instances().end() and src_it->second.ai2_initialized)
                if (const auto Ts = inner_eigen_->get_transformation_matrix(room_now->name(), best->name(), 0); Ts.has_value())
                {
                    // room <- door, as a 2-D rigid map; its inverse takes the source's room-frame leaf into the door frame.
                    const Eigen::Matrix2f R = Ts->rotation().topLeftCorner<2, 2>().cast<float>();
                    const Eigen::Vector2f o = Ts->translation().head<2>().cast<float>();
                    const auto& si = src_it->second;
                    em.has_leaf    = true;
                    em.hinge_local = R.transpose() * (si.leaf_pose.hinge_xy - o);
                    em.open_local  = R.transpose() * (si.leaf.swing * si.aperture.across_u());
                    em.phi_est     = si.phi_est;
                    em.phi_sigma   = si.phi_sigma;
                    em.hinge_llr   = si.hinge_llr;
                    em.swing_llr   = si.swing_llr;
                    em.phi_curve   = si.phi_curve;
                    em.matched_frames = si.matched_frames;
                }
            if (src_it != fitter_->instances().end() and src_it->second.existence_seeded)
                em.existence_L = src_it->second.existence.logodds();
            entry_mirrors_[room.id()] = std::move(em);
        }
    }

    // 3. Refresh the interaction channel from the source — the one belief about this door.
    //    Re-read the mirror node every time: update_node writes the WHOLE copy, so a stale copy would erase
    //    whatever anybody else added since.
    // (Task D2: the source's existence L is refreshed here too, while the source lives — ltsm deletes it with its room.)
    for (auto& [room_id, mir] : entry_mirrors_)
        if (const auto si = fitter_->instances().find(mir.source_id); si != fitter_->instances().end() and si->second.existence_seeded)
            mir.existence_L = si->second.existence.logodds();
    for (const auto& [room_id, mir] : entry_mirrors_)
    {
        const auto src = G->get_node(mir.source_id);
        auto dst = G->get_node(mir.node_id);
        if (not src.has_value() or not dst.has_value())
            continue;
        bool changed = false;
        if (const auto v = G->get_attrib_by_name<door_open_prob_att>(src.value()); v.has_value()
            and G->get_attrib_by_name<door_open_prob_att>(dst.value()) != v)
        {
            G->add_or_modify_attrib_local<door_open_prob_att>(dst.value(), v.value());
            changed = true;
        }
        if (const auto v = G->get_attrib_by_name<door_phi_rad_att>(src.value()); v.has_value()
            and G->get_attrib_by_name<door_phi_rad_att>(dst.value()) != v)
        {
            G->add_or_modify_attrib_local<door_phi_rad_att>(dst.value(), v.value());
            changed = true;
        }
        if (changed)
            G->update_node(dst.value());
    }
}

void SpecificWorker::release_room(std::uint64_t old_room, std::uint64_t new_room)
{
    decision_log_.record(rc::door::DoorDecisionKind::RoomSwitch, new_room, {}, double(old_room),
                       fitter_ ? double(fitter_->instances().size()) : 0.0);
    std::print("door_concept: [room] LET GO — `current` moved {} → {}. Releasing {} door(s) that belong "
               "to the room we have left; the new room's doors will be re-earned from scratch.\n",
               old_room, new_room, fitter_ ? fitter_->instances().size() : 0u);

    // Our own nodes, through our own teardown path — which drops the epistemic affordance AND the three
    // pragmatic ones, then the door node (whose RT edge goes with it). Collected first: retire_instance
    // erases from the map we would otherwise be iterating.
    // ★ THE ROOM LEFT BEHIND MAY STILL BE LIVE (2026-10-11, door crossing with ltsm eviction OFF): ltsm moves `current`
    //   and keeps the old room in the working memory. Its doors are then still true and still somebody's evidence —
    //   the crossed one is the counterpart in ltsm's passage node — so they are HELD: this agent stops fitting them
    //   (drops the instance and its four affordances) but leaves the door node and its RT under the old room's wall.
    //   They go with this agent's owned nodes on exit, or with the room if ltsm ever evicts it.
    // ★ A ROOM SWITCH DURING THE CROSS IS THE CROSS'S TERMINAL TRANSITION (fix round 1, controller ruling). The robot
    //   has gone through: ltsm moves `current` before door_concept could complete the cross, and the door released
    //   below takes its cross affordance with it. Removed HERE, first, so the affordance's link goes before its node.
    //   During approach/open the anchor is instead relinked at the end of this function.
    // ★ Task 12 Fix E (user, 10-18: "keep going inside so it can localise in the room"): the room switch is NOT the end
    //   of the cross any more. The robot is ~0.3 m past the plane when room_concept moves `current`; the controller keeps
    //   servoing the anchor until it is THROUGH (s ≥ AnchorCrossStandoffM along the normal). The anchor hangs from the
    //   robot, so the switch does not move it: it is KEPT, relinked below to the door re-identified in the new room, and
    //   published as for a claimed cross (crossing_through). It is removed by REMOVE (b) once the consumer executes
    //   something else (close-behind, the next affordance), or on exit.
    if (passage_anchor_.has_value() and passage_anchor_->cross_was_claimed)
    {
        passage_anchor_->crossing_through = true;
        std::print("door_concept: [anchor] {}@robot KEPT through the room switch — the cross goes on until the robot is through\n",
                   passage_anchor_->door_name);
    }
    const bool old_room_lives = G->get_node(old_room).has_value();
    // ★ AND THE CROSSED DOOR IS ALREADY KNOWN IN THE NEW ROOM: its entry mirror (step_entry_mirrors) sits under the
    //   new room at the exact transform of the source door. It becomes this room's door instance, so a detection of
    //   that door from inside the new room associates with it instead of being born as a duplicate (live: door_26).
    std::uint64_t adopt_mirror = 0;
    // ★ …AND IT KEEPS WHAT WE KNEW ABOUT ITS LEAF (2026-10-12). The robot just went through it, so the door is
    //   open, and from the new room an open door is the one thing door_concept cannot re-detect: the leaf is beside
    //   the doorway, there is no shut panel to fit. Live try16: the mirror sat at door_open_prob = unmeasured for the
    //   whole stay in room 2. try17: fitted from scratch, its leaf swung the config's way — INTO room 2 — when the
    //   real one opens OUT of it, so the LiDAR settled phi at 46° and P(passable) at 0. What carries over is captured
    //   when the mirror is made (EntryMirror), because ltsm deletes the source with its room before we get here.
    std::optional<EntryMirror> adopted_from;
    if (const auto it = entry_mirrors_.find(new_room); it != entry_mirrors_.end())
    {
        adopt_mirror = it->second.node_id;
        adopted_from = it->second;
        entry_mirrors_.erase(it);
    }
    // ★ NO MIRROR ⇒ RECREATE THE DOOR WE CAME THROUGH (2026-10-12). A remembered room is re-entered already promoted,
    //   so no proto ever existed for step_entry_mirrors to put a mirror under, and the crossed door was deleted with
    //   the room left behind before we got here (ltsm evicts first). Live try24: re-entered room_1 had NO door, and
    //   "cross door" had nothing to go through. The door the robot just crossed is the lost one nearest to it.
    if (adopt_mirror == 0)
        if (auto rec = recreate_lost_door(new_room); rec.has_value())
        {
            adopt_mirror = rec->first;
            adopted_from = std::move(rec->second);
        }
    lost_doors_.clear();
    // ★ …OR THE CROSSED DOOR IS STILL LIVE IN THE ROOM LEFT BEHIND (try106 leg 3). ltsm moves `current` and evicts the
    //   old room in two steps, and which one reaches us first is a race: when the move wins, the crossed door is not
    //   lost yet, so there is nothing to recreate here. It is HELD below, and when ltsm deletes it the carry happens
    //   then (step_room_following). Without this, room_1 was re-entered with no door_1 and "cross door_1" failed.
    carry_pending_room_ = (adopt_mirror == 0 and old_room_lives) ? new_room : 0;
    decision_log_.record(rc::door::DoorDecisionKind::CarryPending, new_room, {}, double(carry_pending_room_ != 0),
                       old_room_lives ? 1.0 : 0.0, double(adopt_mirror));
    // ★ OUR OWN MIRROR OF IT, NOW. Both rooms are live at this instant, so each held door's pose in the new room is a
    //   direct chain; after the switch the door is no longer fitted and its memo is never refreshed again.
    if (carry_pending_room_ != 0 and inner_eigen_ and fitter_)
        if (const auto room_n = G->get_node(new_room); room_n.has_value())
            for (const auto& [id, inst] : fitter_->instances())
                if (const auto m = door_memo_.find(id); m != door_memo_.end())
                    if (const auto Tx = inner_eigen_->get_transformation_matrix(room_n->name(), m->second.name, 0);
                        Tx.has_value() and Tx->matrix().allFinite())
                        m->second.T_in_room[new_room] = Eigen::Affine3d(Tx->matrix());
    // ★ ONE NAME PER DOORWAY (2026-10-13). The mirror was born under a fresh number because its source still held the
    //   doorway's name (DSR names are unique). Once the source is gone — ltsm evicts it with the room left behind —
    //   the mirror is re-made under that name, so the same hole is door_8 from both rooms, across every crossing, and
    //   "cross door_8" means the same thing on the way back. A source still live (eviction off) keeps the fresh name.
    if (adopt_mirror != 0 and adopted_from.has_value())
        if (const auto renamed = adopt_doorway_name(adopt_mirror, new_room, adopted_from->name); renamed != 0)
            adopt_mirror = renamed;
    remember_live_identities();   // the room we are leaving, under ITS name: room_node_id_ has not moved yet
    if (fitter_)
    {
        std::vector<std::uint64_t> leaving;
        leaving.reserve(fitter_->instances().size());
        for (const auto& [id, inst] : fitter_->instances())
        {
            // The door carried across (recreated just above) is the NEW room's door, not one of the room left: its
            // node-added slot already gave it an instance, and retiring it here deleted the node we had just made
            // (live try27: recreated, then gone; a phantom born mid-room took its name).
            if (id == adopt_mirror) continue;
            leaving.push_back(id);
            // ★A door being released is NOT a door that stopped existing. Log it as a release so the
            // phantom/death record is not polluted with removals that carry no evidence about the world:
            // the existence belief that justified this door is still perfectly valid, we are simply no
            // longer the agent looking at that room.
            log_tracker_event("RELEASE", id, inst.model.state().cx, inst.model.state().cy,
                              std::format("room {} -> {}", old_room, new_room), inst.node_name);
        }
        for (const auto id : leaving)
        {
            if (not old_room_lives) { retire_instance(id); continue; }
            if (auto it = fitter_->instances().find(id); it != fitter_->instances().end())
            {
                log_instance_teardown(it->second, "release room");
                it->second.affordance.remove();
                it->second.aff_approach.remove();
                it->second.aff_open.remove();
                it->second.aff_cross.remove();
                it->second.aff_close.remove();
            }
            fitter_->forget_node(id);
            held_doors_.insert(id);
        }
        if (old_room_lives)
            std::print("door_concept: [room] the room left behind is still live — {} door node(s) HELD there, not "
                       "deleted\n", leaving.size());
    }

    // ★GHOSTS ARE ROOM-LOCAL AND MUST NOT CROSS. A ghost is "a door was removed at this (x,y), let it
    // take its old name if it comes back there" — in ROOM coordinates. Carried into a different room's
    // frame, the same (x,y) is a different physical place, and the first door detected near it would
    // silently inherit another room's identity.
    ghosts_.clear();

    // The identity table is room-keyed since 2026-10-13: the released room's doors were written under ITS name above
    // (before room_node_id_ moves), and the new room's rows are matched only against the new room.

    room_node_id_ = new_room;
    // ★DROP THE OLD POLYGON BEFORE PICKING UP THE NEW ONE. The containment prior is the path that can
    // remove a door without any sensor evidence, so a single cycle evaluated against the walls of the
    // room we have just left is a cycle in which every door in the new room is "outside the room".
    // Emptying it first makes has_room_polygon() false, which disables the prior rather than running it
    // on the wrong walls — the fail-safe direction.
    fitter_->set_room_geometry(Eigen::Vector2f::Zero(), {});
    refresh_room_geometry();         // and pick the new one up in the same cycle
    install_carried_door(new_room, adopt_mirror, adopted_from);
    // AFTER the carry: the crossed door is live by now (or still HELD under its name for the late carry), so the
    // restore below cannot duplicate it.
    enter_room_memory(new_room, false);
    step_pending_restore();          // now, if the recalled outline is already loaded; else on a later cycle
}

// ─── RESTORE ON RE-ENTRY ──────────────────────────────────────────────────────────────────────────────────────────

void SpecificWorker::latch_recall_names()
{
    // ltsm_agent's room_recall.h: `recall_<room>` metaconcept, object_subtype "room_recall", RT child of a live door.
    for (const auto& n : G->get_nodes_by_type("metaconcept"))
        if (n.name().starts_with("recall_")
            and G->get_attrib_by_name<object_subtype_att>(n).value_or(std::string{}) == "room_recall")
            if (const auto room = n.name().substr(7); not room.empty() and recall_names_seen_.insert(room).second)
                std::print("door_concept: [restore] ltsm announces remembered '{}' behind a door — its doors will be "
                           "restored if the robot enters it\n", room);
}

void SpecificWorker::enter_room_memory(std::uint64_t room_id, bool first_resolution)
{
    pending_restore_room_ = 0;
    const auto room = G->get_node(room_id);
    const bool recalled = room.has_value() and not first_resolution and recall_names_seen_.contains(room->name());
    recall_names_seen_.clear();      // a recall is evidence about ONE crossing only
    if (not room.has_value())
        return;
    if (recalled)
    {
        pending_restore_room_ = room_id;
        return;
    }
    // ★FRESH ENTRY (controller ruling 2026-10-20): the table outlives ltsm's memory, so rows of a room of this name from
    //  an older run must not be restorable later as confident phantoms. Names and places stay (identity only).
    std::vector<std::string> strippable;   // decision log: which rows lose their restorable data
    for (const auto& i : identities_)
        if (i.room == room->name() and (i.pose.has_value() or i.leaf.has_value() or i.exist_logodds.has_value()))
            strippable.push_back(i.name);
    if (const int n = rc::door::strip_restorable(identities_, room->name()); n > 0)
    {
        save_identities();
        for (const auto& nm : strippable)
            decision_log_.record(rc::door::DoorDecisionKind::IdentityStrip, 0, nm, 0, 0, 0, 0, room->name());
        std::print("door_concept: [restore] '{}' entered FRESH ({}) — {} remembered row(s) reduced to identity only; "
                   "only doors fitted from now on can be restored on a later recall\n",
                   room->name(), first_resolution ? "first room resolved" : "no ltsm recall announced it", n);
    }
    else
        std::print("door_concept: [restore] '{}' entered FRESH ({}) — nothing restored; its doors are earned from evidence\n",
                   room->name(), first_resolution ? "first room resolved" : "no ltsm recall announced it");
}

void SpecificWorker::step_pending_restore()
{
    if (pending_restore_room_ == 0 or pending_restore_room_ != room_node_id_ or not fitter_)
        return;
    if (not fitter_->has_room_polygon())
    {
        refresh_room_geometry();
        if (not fitter_->has_room_polygon())
            return;   // the recalled outline has not reached us yet: wait (the door's belief must find its wall)
    }
    const auto room = std::exchange(pending_restore_room_, 0);
    restore_remembered_doors(room);
}

void SpecificWorker::restore_remembered_doors(std::uint64_t new_room)
{
    if (not G or not fitter_ or not scene_graph_)
        return;
    const auto room = G->get_node(new_room);
    if (not room.has_value())
        return;
    const bool recalled = true;   // only reached through enter_room_memory's recall branch

    // Every door node live in the graph, with its aperture in the ENTERED room's frame when that resolves (ts==0 on the
    // main-thread instance; always check the optional). A door held in the room left behind still blocks by name.
    std::vector<rc::door::LiveDoor> live;
    for (const auto& n : G->get_nodes_by_type("object"))
    {
        if (not n.name().starts_with("door_")) continue;
        rc::door::LiveDoor l{n.name(), std::nullopt};
        if (inner_eigen_)
            if (const auto T = inner_eigen_->get_transformation_matrix(room->name(), n.name(), 0);
                T.has_value() and T->matrix().allFinite() and T->translation().head<2>().norm() < rc::door::kIdentityMaxPlaceM)
                l.xy_in_room = T->translation().head<2>().cast<float>();
        live.push_back(std::move(l));
    }
    const auto d = rc::door::plan_restore(identities_, room->name(), recalled, live, cfg_.exist_reacquire_radius_m);
    for (const auto& k : d.skipped)
        decision_log_.record(rc::door::DoorDecisionKind::RestoreSkip, 0, k.name, double(static_cast<int>(k.reason)), 0, 0, 0, room->name());
    for (const auto& k : d.skipped)
        std::print("door_concept: [restore] '{}' in '{}' NOT restored: {}\n", k.name, room->name(), rc::door::to_string(k.reason));

    for (const auto& r : d.restore)
    {
        const auto names = reserved_names();
        // Task D-C: the remembered belief, decayed toward BirthLogodds by the time since the door was last seen.
        const float prior_L = rc::door::restore_existence_prior(r, cfg_.exist_birth_logodds, static_cast<std::int64_t>(rc::provenance::now_ms()));
        // Door node origin = aperture centre on the floor, yaw = the aperture's (write_rt_pose's room-frame form).
        const auto id = scene_graph_->create_door_copy(room.value(), r.pose->width, r.pose->depth, r.pose->height,
                                                       std::nullopt, r.leaf.has_value() ? std::optional<float>(r.leaf->phi) : std::nullopt,
                                                       Eigen::Vector3f(r.xy.x(), r.xy.y(), 0.0f),
                                                       Eigen::Vector3f(0.0f, 0.0f, r.pose->yaw), names,
                                                       std::format("'{}' remembered in {}", r.name, room->name()), r.name);
        if (id == 0)
        {
            std::print("door_concept: [restore] '{}' could not be recreated in '{}' (insert failed)\n", r.name, room->name());
            continue;
        }
        const auto n = G->get_node(id);
        if (not n.has_value())
            continue;
        // ★REBUILD, AS THE CARRIED DOOR DOES (tourP6): the node-added path built an instance at insert time, before
        //  the RT edge existed, so it read (0,0); ensure_instance then answered "already exists" and the old code SKIPPED
        //  the seeding — door_1 sat unfitted at (0,0) with no approach/cross for 5 minutes, and the tracker birthed a
        //  duplicate (door_4) on its real hole. Forget that instance, rebuild it now that the RT is written, and seed it.
        if (fitter_->instances().contains(id))
            fitter_->forget_node(id);
        fitter_->ensure_instance(n.value(), room_node_id_);
        const auto iit = fitter_->instances().find(id);
        if (iit == fitter_->instances().end())
            continue;
        auto& inst = iit->second;
        // ★THE EXISTENCE PRIOR IS THE MEMORY. Not a fresh birth's prior and not certainty either: the belief the door had
        //  when the room was left (or, for a row written without one, Existence.BirthLogodds). The channel takes it from
        //  here — seen ⇒ confirmed, absent where it should be visible (P(detect), wall line of sight) ⇒ removed.
        //  ⚠ Deliberately NOT the ghost convention (re-acquisition restarts at BirthLogodds): a ghost was REMOVED for
        //  absence, a restored door was ALIVE when left — see rc::door::restore_existence_prior.
        // Fitted without a fresh detection (as the carried door is), at the remembered pose and size, with its leaf memory.
        rc::door::seed_restored_instance(inst, r, prior_L, cfg_.exist_max_logodds, std::chrono::steady_clock::now());
        log_tracker_event("RESTORE", id, r.xy.x(), r.xy.y(), std::format("remembered in {}", room->name()), n->name());
        if (n->name() != r.name)
            decision_log_.record(rc::door::DoorDecisionKind::RestoreRename, id, r.name, 0, 0, 0, 0, n->name());
        std::print("door_concept: [restore] '{}' RESTORED in remembered '{}' at ({:.2f},{:.2f}) yaw={:.2f} w={:.2f} h={:.2f} — "
                   "existence prior L={:.2f} (P={:.2f}, {}), leaf memory {}\n",
                   n->name(), room->name(), r.xy.x(), r.xy.y(), r.pose->yaw, r.pose->width, r.pose->height, prior_L,
                   inst.existence.p_exists(),
                   not r.exist_logodds.has_value() ? "default BirthLogodds"
                   : r.seen_wall_ms.has_value() ? std::format("remembered L={:.2f}, seen {:.0f} s ago, decayed toward BirthLogodds",
                                                              *r.exist_logodds, 1e-3 * double(static_cast<std::int64_t>(rc::provenance::now_ms()) - *r.seen_wall_ms))
                                                : std::string("remembered with no age: BirthLogodds"),
                   r.leaf.has_value() ? "yes" : "no");
    }
}

// ─── LOOK AT A REMEMBERED DOOR (door_look.h) ─────────────────────────────────────────────────────────────────────

void SpecificWorker::drop_looks(const char* why)
{
    for (auto& [nm, e] : looks_)
        if (e.aff.node_id() != 0)
        {
            e.aff.remove();
            decision_log_.record(rc::door::DoorDecisionKind::LookDrop, 0, nm, 0, 0, 0, 0, why);
            std::print("door_concept: [look] aff_door_look_{} withdrawn — {}\n", nm, why);
        }
    looks_.clear();
}

void SpecificWorker::step_remembered_looks()
{
    if (not G or not fitter_ or room_node_id_ == 0)
        return;
    const std::string room = current_room_name();
    if (room_node_id_ != looks_room_)
    {
        drop_looks("the robot is in another room now");
        looks_room_ = room_node_id_;
    }
    if (room.empty() or not fitter_->has_room_polygon())
    {
        for (auto& [nm, e] : looks_) e.aff.hold_offered();   // no polygon this cycle is not a withdrawal
        return;
    }
    const auto& poly = fitter_->room_polygon();

    // The doors that ARE instantiated (this room's instances), and the doorways their walls have open.
    std::vector<rc::door::LiveDoor> live;
    std::vector<rc::nbv::WallGap> gaps;
    for (const auto& [id, inst] : fitter_->instances())
    {
        const auto& s = inst.model.state();
        live.push_back({inst.node_name, Eigen::Vector2f(s.ap_cx, s.ap_cy)});
        if (inst.ai2_initialized and not inst.is_bearing_hypothesis)
            gaps.push_back({Eigen::Vector2f(s.ap_cx, s.ap_cy), 0.5f * s.w});
    }
    std::vector<std::string> ruled;
    for (const auto& [nm, e] : looks_) if (e.ruled_out) ruled.push_back(nm);
    const auto offer = rc::door::look::offer_rows(identities_, room, live, ruled, cfg_.exist_reacquire_radius_m);

    // Withdraw what is no longer offered: re-born (the common, good ending). Ruled-out entries stay, inert, as the mark.
    for (auto it = looks_.begin(); it != looks_.end();)
    {
        if (it->second.ruled_out or std::ranges::any_of(offer, [&](const auto& r) { return r.name == it->first; }))
        { ++it; continue; }
        if (it->second.aff.node_id() != 0)
        {
            it->second.aff.remove();
            decision_log_.record(rc::door::DoorDecisionKind::LookDrop, 0, it->first, 0, 0, 0, 0, "re-born at its hole");
            std::print("door_concept: [look] aff_door_look_{} withdrawn — the door is live again (re-born at its hole)\n",
                       it->first);
        }
        it = looks_.erase(it);
    }
    if (offer.empty())
        return;

    rc::exist::RemovalPolicy pol;
    pol.logodds_max   = cfg_.exist_max_logodds;
    pol.removal_prob  = cfg_.exist_removal_prob;
    pol.remove_frames = static_cast<float>(cfg_.exist_remove_frames);
    const rc::door::look::Params prm{.robot_radius_m = 0.5f * cfg_.pragmatic.robot_passage_width_m,
                                     .clutter_prob   = cfg_.exist_clutter_prob,
                                     .detection_prob = cfg_.exist_detection_prob,
                                     .prior_w_m      = cfg_.door_prior_w_m,
                                     .prior_h_m      = cfg_.door_prior_h_m,
                                     .thickness_m    = cfg_.door_thickness_m};
    const auto sensor    = rc::nbv::sensor_from_graph(*G, inner_eigen_.get());
    const auto furniture = rc::nbv::collect_graph_obstacles(*G, inner_eigen_.get(), 0);
    const auto cam       = fitter_->camera_in_room();
    const auto room_node = G->get_node(room_node_id_);
    if (not room_node.has_value())
        return;

    for (const auto& r : offer)
    {
        auto it = looks_.find(r.name);
        if (it == looks_.end())
        {
            // The prior: what memory says (the row's existence when written; BirthLogodds for an identity-only row).
            const float L0 = rc::door::restore_existence_prior(r, cfg_.exist_birth_logodds, static_cast<std::int64_t>(rc::provenance::now_ms()));
            it = looks_.emplace(r.name, LookEntry{rc::door::look::Evidence(1.0f / (1.0f + std::exp(-L0)), pol), {}, false}).first;
            it->second.aff.init(G, room_node_id_, "door_look_" + r.name, "door");
            std::print("door_concept: [look] '{}' remembered in '{}' at ({:.2f},{:.2f}) is not in the graph — offering a "
                       "look (prior P(exists)={:.2f})\n", r.name, room, r.xy.x(), r.xy.y(), it->second.ev.p_exists());
        }
        auto& e = it->second;
        if (const auto aid = e.aff.node_id(); aid != 0)
            e.aff.on_node_modified(aid);   // the controller's protocol flags, polled (see rc::poll_protocol)

        // ── Absence / presence evidence from THIS mask frame's view of the place ──
        if (look_new_frame_ and cam.has_value())
        {
            const Eigen::Vector2f cxy(static_cast<float>(cam->coeff(0, 3)), static_cast<float>(cam->coeff(1, 3)));
            const float cyaw = static_cast<float>(std::atan2(cam->coeff(1, 1), cam->coeff(0, 1)));   // zed +y = forward
            const float pd = rc::door::look::p_detect_from(r, cxy, cyaw, poly, sensor, furniture, gaps, prm);
            const float same_hole = cfg_.exist_reacquire_radius_m + (r.pose.has_value() ? r.pose->width : 0.0f);
            const bool seen = std::ranges::any_of(look_dets_xy_, [&](const Eigen::Vector2f& d) { return (d - r.xy).norm() < same_hole; });
            if (e.ev.observe(pd, seen, prm.detection_prob, prm.clutter_prob,
                             rc::door::look::view_of(r, poly, cxy, prm.prior_w_m), pol))
            {
                e.ruled_out = true;
                e.aff.remove();
                decision_log_.record(rc::door::DoorDecisionKind::LookDrop, 0, r.name, e.ev.p_exists(), 0, 0, 0, "ruled out");
                std::print("door_concept: [look] '{}' RULED OUT this visit — absent where it should have been visible "
                           "(P(exists)={:.3f}); not offered again until the room is re-entered\n", r.name, e.ev.p_exists());
                continue;
            }
        }

        // ── The offer: the standpoint of highest expected information gain about this door's existence ──
        const auto plan = rc::door::look::plan(r, e.ev.p_exists(), poly, sensor, furniture, gaps, prm);
        if (not plan.valid)
        {
            e.aff.hold_offered();   // could not compute a look this cycle (camera model, polygon): not a withdrawal
            continue;
        }
        rc::AffordanceTarget tgt;
        tgt.x_m = plan.pos.x(); tgt.y_m = plan.pos.y(); tgt.yaw_rad = plan.yaw;
        tgt.gain = plan.eig;
        tgt.valid = true;
        tgt.standoff_min_m = plan.band_near_m;
        tgt.standoff_max_m = plan.band_far_m;
        tgt.framing_fill   = rc::nbv::framing_fill(sensor);
        const bool fresh = e.aff.node_id() == 0;
        e.aff.update(tgt);
        if (fresh and e.aff.node_id() != 0)
            decision_log_.record(rc::door::DoorDecisionKind::LookOffer, 0, r.name, plan.pos.x(), plan.pos.y(), plan.eig, plan.yaw);
        if (fresh and e.aff.node_id() != 0)
            std::print("door_concept: [look] aff_door_look_{} OFFERED at ({:.2f},{:.2f}) yaw={:.2f} — EIG {:.3f} nats "
                       "(p_detect {:.2f}, visible {:.2f})\n", r.name, plan.pos.x(), plan.pos.y(), plan.yaw, plan.eig,
                       plan.p_detect, plan.visible_frac);
    }
}

// The door carried across a room switch becomes the new room's door: its instance rebuilt against the room we are in,
// the leaf it remembers from the crossing, and the passage anchor relinked to it. Called by release_room, and LATE by
// step_room_following when the crossed door was still live in the room left behind at the switch (carry_pending_room_).
void SpecificWorker::install_carried_door(std::uint64_t new_room, std::uint64_t adopt_mirror,
                                          const std::optional<EntryMirror>& adopted_from)
{
    decision_log_.record(rc::door::DoorDecisionKind::CarryInstall, adopt_mirror,
                       adopted_from.has_value() ? adopted_from->name : std::string{}, double(new_room),
                       adopted_from.has_value() and adopted_from->has_leaf ? 1.0 : 0.0);
    // A door recreated just above was inserted while room_node_id_ still named the room LEFT: our own node-added slot
    // already built an instance for it against that (deleted) room, which reads its RT as (0,0) and makes the
    // adoption below see "already exists" and skip (live try25: no leaf memory, the door lost, a phantom picked
    // instead). Rebuild it against the room we are in.
    if (adopt_mirror != 0 and fitter_->instances().contains(adopt_mirror))
        fitter_->forget_node(adopt_mirror);
    if (adopt_mirror != 0)
        if (const auto n = G->get_node(adopt_mirror); n.has_value() and fitter_->ensure_instance(n.value(), room_node_id_))
        {
            std::print("door_concept: [room] the entry mirror {} is now this room's door (the crossed door, seen from "
                       "inside)\n", n->name());
            const auto room_n = G->get_node(new_room);
            if (adopted_from.has_value() and adopted_from->has_leaf and room_n.has_value())
                if (auto it = fitter_->instances().find(adopt_mirror); it != fitter_->instances().end())
                    if (const auto Tm = inner_eigen_->get_transformation_matrix(room_n->name(), n->name(), 0); Tm.has_value())
                    {
                        auto& dst = it->second;
                        const auto& em = *adopted_from;
                        const Eigen::Matrix2f R = Tm->rotation().topLeftCorner<2, 2>().cast<float>();
                        const Eigen::Vector2f o = Tm->translation().head<2>().cast<float>();
                        dst.leaf_side_hint = std::make_pair(Eigen::Vector2f(R * em.hinge_local + o),
                                                            Eigen::Vector2f(R * em.open_local));
                        dst.phi_est   = em.phi_est;
                        dst.phi_sigma = em.phi_sigma;
                        dst.hinge_llr = em.hinge_llr;   // the hinge hint is the source's MAP jamb; this is how sure it was
                        dst.swing_llr = em.swing_llr;   // Task 13: the swing hint is the source's MAP side; likewise
                        dst.phi_curve = em.phi_curve;
                        dst.phi_last_t = std::chrono::steady_clock::now();
                        // The robot's body just went through the aperture: an observation that it was passable, not
                        // merely our own request.
                        dst.phi_from_command = false;
                        // ★ AND IT IS A CONFIRMED DOOR, so it is FITTED. process_door_node skips an instance with no
                        //   fresh detection and fewer than 5 matched frames — from this side an open door gives neither.
                        dst.matched_frames = em.matched_frames;
                        std::print("door_concept: [room] {} remembers its leaf from the crossing: phi={:.0f} deg "
                                   "sigma={:.0f} deg, curve {} pts\n", n->name(), dst.phi_est * 57.2958f,
                                   dst.phi_sigma * 57.2958f, dst.phi_curve.size());
                    }
            // ★Task D2 — THE BELIEF CROSSES WITH THE DOOR, AND SO DOES THE CROSSING (gold12: door_1 was re-made at BirthLogodds 1.0,
            //   forgetting its L = +1.83 and the fact that the robot's body had just passed through it, and died 5 s later).
            //   L = the source's L (BirthLogodds when the source was never seeded: the crossing is evidence independent of it) + the
            //   crossing as an observation: ln(Existence.CrossPassablePrior / Existence.CrossLeak) nats (door_leaf_visibility.h). The
            //   door's own door_open_prob is NOT read: the crossing may have raised it (circular). Marked seeded so the existence slice
            //   does not re-seed it.
            if (adopted_from.has_value())
                if (auto it = fitter_->instances().find(adopt_mirror); it != fitter_->instances().end())
                {
                    auto& dst = it->second;
                    const float L_src = adopted_from->existence_L.value_or(cfg_.exist_birth_logodds);
                    const float dLx = rc::door::crossing_logodds(cfg_.exist_cross_passable_prior, cfg_.exist_cross_leak);
                    dst.existence.set(L_src + dLx);
                    dst.existence_seeded      = true;
                    dst.existence_entry_L     = dst.existence.logodds();
                    dst.existence_since_entry = 0.0f;
                    std::print("door_concept: [room] {} keeps its existence across the crossing: L {:.2f}{} + crossing {:.2f} "
                               "(prior {:.2f}, leak {:.2f}) -> {:.2f}\n", n->name(), L_src,
                               adopted_from->existence_L.has_value() ? "" : " (birth seed)", dLx, cfg_.exist_cross_passable_prior,
                               cfg_.exist_cross_leak, dst.existence.logodds());
                }
        }
    // ★ THE PASSAGE ANCHOR IS NOT A DOOR OF EITHER ROOM. It hangs from the robot, so nothing above touched it; the
    //   one thing the room switch owes it (during approach/open, and since Task 12 Fix E during the cross as well) is its
    //   ORIGINAL: when the door carried across is the anchored one (same source node, or the
    //   same doorway name — one name per doorway), the anchor now names that door.
    if (passage_anchor_.has_value() and adopt_mirror != 0 and adopted_from.has_value()
        and (adopted_from->source_id == passage_anchor_->door_id or adopted_from->name == passage_anchor_->door_name))
        relink_passage_anchor(adopt_mirror, "room switch: the crossed door, re-identified in the new room");
}


// Put the door on the 2-D graph canvas NEXT TO ITS ACTUAL RT PARENT, and keep it there when the parent
// changes. ★The canvas is a picture of the structure, so it must be anchored to the same node the
// structure is: this used to place every door relative to the ROOM while the RT edge came from a WALL,
// which drew the door floating beside the room with its single edge disappearing off-screen toward a
// wall somewhere else in the layout. It reads as "the door is not connected to the room" — and the only
// way to find out otherwise is to trace the RT chain by hand, which nobody should have to do to trust a
// diagram. No-op unless the anchor actually changed, so it costs one attribute read per cycle.
void SpecificWorker::place_door_on_canvas(const DSR::Node& node, rc::DoorInstance& inst)
{
    if (not G) return;
    // The RT parent, as the graph states it — not as we remember it.
    std::uint64_t anchor = 0;
    for (const auto& e : G->get_edges_to_id(node.id()))
        if (e.type() == "RT") { anchor = e.from(); break; }
    if (anchor == 0)
        anchor = room_node_id_;            // pre-wall: a door is born hanging from the room
    if (anchor == 0 or anchor == inst.canvas_anchor_id)
        return;

    float px = 200.f, py = 200.f;
    if (const auto an = G->get_node(anchor); an.has_value())
    {
        px = G->get_attrib_by_name<pos_x_att>(an.value()).value_or(200.f);
        py = G->get_attrib_by_name<pos_y_att>(an.value()).value_or(200.f);
    }
    auto n_mut = node;
    G->add_or_modify_attrib_local<pos_x_att>(n_mut, px + 90.f);
    G->add_or_modify_attrib_local<pos_y_att>(n_mut, py + 60.f);
    G->update_node(n_mut);
    inst.canvas_anchor_id = anchor;
    trigger_graph_layout_twopi();
}


// ─── door --[transitable]--> door ────────────────────────────────────────────────────────────────
//
// A SELF-EDGE on the door's own node, present exactly while this agent believes the robot can get
// through. It exists so a consumer can ask the GRAPH whether a doorway is passable instead of fetching
// the door's geometry, the leaf angle and the robot's own width and re-deriving the answer — which is
// three transforms and a model each, and three chances to derive it differently from us.
//
// ★WHY A SELF-EDGE AND NOT AN ATTRIBUTE. An attribute would ride on `update_node`, which is a WHOLE-NODE
// REPLACE on both sync engines: any writer refreshing this node from a stale copy silently reverts — or
// destroys — a flag another writer had just set. That is the same reasoning that put the affordance
// protocol's execution claim on an edge rather than on the affordance node, and it applies here for the
// same mechanical reason. An edge is its own object with its own key, so asserting and retracting it
// cannot race the door's per-cycle geometry republish.
//
// ★"transitable", NOT "open". The belief is P(clear span >= THIS ROBOT'S passage width), so the edge says
// "this body fits", not "the leaf has moved". A door ajar by 20 cm is open and not transitable. Naming it
// after the quantity actually computed is what stops a consumer reading it as a claim about the mechanism.
//
// ★UNMEASURED HOLDS. p_open_known == false means nothing measured the aperture this cycle — a different
// fact from "measured, and it is shut", and the two must not share a code path. Neither streak advances
// and the edge is left exactly as it is. Today's sentinel work is the same lesson from the other end:
// door_open_prob publishes -1 for NOT MEASURED precisely because 0 is a real answer ("believed shut"),
// and an absence encoded as a legitimate value is indistinguishable from that value.
void SpecificWorker::step_transitable_edge(rc::DoorInstance& inst, const rc::door::InteractionState& st)
{
    if (not G)
        return;
    const auto& pg = cfg_.pragmatic;

    // HOLD on an unmeasured cycle — see the header comment. Not a gate on the belief: a gate on whether
    // there IS one this cycle.
    if (not st.p_open_known)
        return;

    // Inside the Schmitt band NEITHER streak grows, so a probability hovering at the boundary holds
    // whatever was last decided rather than flapping a public edge at the compute rate.
    // ★NO ASSERTION ON PREDICTION ALONE (tourP8 D5): the streak toward asserting advances only on a cycle a leaf /
    //  aperture likelihood contributed to (phi_measured); the command anticipation is a prior, not evidence.
    rc::door::TransitableStreaks sk{inst.transitable_streak, inst.not_transitable_streak};
    const auto dec = rc::door::transitable_step(sk, inst.transitable_asserted, st.p_open, inst.phi_measured,
                                                pg.transitable_prob, pg.not_transitable_prob,
                                                pg.transitable_stable_cycles);
    inst.transitable_streak     = sk.yes;
    inst.not_transitable_streak = sk.no;

    if (dec.assert_now)
    {
        auto edge = DSR::Edge::create<transitable_edge_type>(inst.node_id, inst.node_id);
        if (G->insert_or_assign_edge(edge))
        {
            inst.transitable_asserted = true;
            std::print("door_concept: [transitable] {} ASSERTED — P(passable)={:.3f} held {} measured "
                       "cycles (clear span {:.2f} m at phi={:.1f} deg vs {:.2f} m of body)\n",
                       inst.node_name, st.p_open, inst.transitable_streak, st.clear_width_m,
                       st.phi_rad * 180.0f / static_cast<float>(M_PI),
                       pg.robot_passage_width_m + pg.passage_margin_m);
        }
        else
            std::print("door_concept: [transitable] {} could NOT assert the edge (node {} missing?)\n",
                       inst.node_name, inst.node_id);
    }
    else if (dec.retract_now)
    {
        G->delete_edge(inst.node_id, inst.node_id, "transitable");
        inst.transitable_asserted = false;
        std::print("door_concept: [transitable] {} RETRACTED — P(passable)={:.3f} held {} measured "
                   "cycles below {:.2f} (clear span {:.2f} m at phi={:.1f} deg)\n",
                   inst.node_name, st.p_open, inst.not_transitable_streak, pg.not_transitable_prob,
                   st.clear_width_m, st.phi_rad * 180.0f / static_cast<float>(M_PI));
    }
}

// ─── DSR helpers ─────────────────────────────────────────────────────────────


void SpecificWorker::trigger_graph_layout_twopi()
{
    // SHARED (common/graph_layout) — pure viewer plumbing. See the header for why the layout runs twice.
    rc::gui::trigger_layout_twopi(graph_viewers);
}
// ─── DSR signal slots ────────────────────────────────────────────────────────

void SpecificWorker::modify_node_slot(std::uint64_t id, const std::string& type)
{
    // Doors are generic `object` nodes named "door_*" (schema migration).
    if (type != "object")
        return;

    const auto node_opt = G->get_node(id);
    if (not node_opt.has_value())
        return;
    if (not node_opt.value().name().starts_with("door"))
        return;
    if (rc::room::is_proto_mirror(*G, node_opt.value()))
        return;   // an entry mirror under a proto-room — never a fitted instance (see step_entry_mirrors)
    if (rc::interaction::is_anchor(*G, node_opt.value()))
        return;   // `door_N@robot`: the passage door's robot-relative copy, never a fitted instance
    if (held_doors_.contains(id))
        return;   // a door of the room left behind: HELD there, not this room's (try106: re-adopted against the wrong room)

    input_snapshot_.on_door_upserted(*G, inner_eigen_.get(), node_opt.value());   // recorded if FOREIGN (latch)
    fitter_->ensure_instance(node_opt.value(), room_node_id_);
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
    input_snapshot_.on_node_deleted(id);   // recorded if FOREIGN (latch); delivered in the next cycle's input
    for (auto& [nm, e] : looks_)
        if (e.aff.node_id() == id)
        {
            e.aff.on_node_deleted(id);
            decision_log_.record(rc::door::DoorDecisionKind::LookDrop, 0, nm, 0, 0, 0, 0, "node deleted by another agent");
        }
    // Notify affordance in case its own DSR node was deleted externally
    for (auto& [door_id, inst] : fitter_->instances())
    {
        if (inst.affordance.node_id() == id)
            inst.affordance.on_node_deleted(id);
        // The three pragmatic siblings, same reason: a node deleted by anybody else (the startup sweep,
        // an operator, the LTSM let-go rule) must leave this agent believing it is absent, or update()
        // keeps refreshing an id that is gone and the affordance is never re-offered.
        inst.aff_approach.on_node_deleted(id);
        inst.aff_open.on_node_deleted(id);
        inst.aff_cross.on_node_deleted(id);
        inst.aff_close.on_node_deleted(id);
        // Decision log: an affordance node deleted by somebody else is a withdrawal the producer now believes in.
        log_affordance_transitions(inst, "approach", inst.aff_approach);
        log_affordance_transitions(inst, "open", inst.aff_open);
        log_affordance_transitions(inst, "cross", inst.aff_cross);
        log_affordance_transitions(inst, "close", inst.aff_close);
        log_epistemic_transition(inst);
    }

    if (auto it = fitter_->instances().find(id); it != fitter_->instances().end())
    {
        // ★ THE DOOR'S AFFORDANCES GO WITH IT. They are not RT children (nobody else's sweep reaches them), so a door
        //   deleted by another agent — ltsm evicting the room it was in (2026-10-12, live: aff_door_19, _approach,
        //   _cross left floating after room_1 was removed) — left them in the graph with nothing alive behind them.
        log_instance_teardown(it->second, "node deleted by another agent");
        it->second.affordance.remove();
        it->second.aff_approach.remove();
        it->second.aff_open.remove();
        it->second.aff_cross.remove();
        it->second.aff_close.remove();
        std::print("door_concept: node {} removed from DSR, destroying instance and its affordances\n", id);
        // ★WRITE DOWN WHAT IT WAS, NOW (tourP4, 2026-10-20). An instance still in the map here was deleted by SOMEBODY
        //  ELSE — our own removals (retire_instance) forget the instance before deleting the node. The usual somebody is
        //  ltsm evicting the room we are leaving, and it does so BEFORE `current` reaches us, so release_room's
        //  remember_live_identities found 0 instances and the fitted doors' pose rows were never written: on re-entry
        //  every row of room_1 was identity-only and nothing could be restored. room_node_id_ is still that room here
        //  (or its name is cached), so the row lands under the right room.
        if (it->second.ai2_initialized and not it->second.is_bearing_hypothesis)
            upsert_identity(identity_of(it->second), it->second.model.state().w, identity_weight(it->second));
        fitter_->instances().erase(it);
        if (const auto m = door_memo_.find(id); m != door_memo_.end())
        {
            lost_doors_.push_back(std::move(m->second));
            door_memo_.erase(m);
        }
    }
    // A door HELD in the room left behind, deleted now with that room: lost, and carried across by step_room_following
    // if it is the one the robot just went through.
    if (held_doors_.erase(id) > 0)
        if (const auto m = door_memo_.find(id); m != door_memo_.end())
        {
            lost_doors_.push_back(std::move(m->second));
            door_memo_.erase(m);
        }
}

// ─── Lifecycle stubs ─────────────────────────────────────────────────────────

void SpecificWorker::emergency()
{
    std::print("door_concept: emergency()\n");
}

void SpecificWorker::restore()
{
    std::print("door_concept: restore()\n");
}

int SpecificWorker::startup_check()
{
    std::print("door_concept: startup_check()\n");
    return 0;
}





// ─── Doors carried across a room change ─────────────────────────────────────────────────────────────────────
void SpecificWorker::remember_door(const rc::DoorInstance& inst)
{
    if (not G or not inner_eigen_ or not inst.ai2_initialized or inst.is_bearing_hypothesis) return;
    const auto node = G->get_node(inst.node_id);
    const auto room = G->get_node(room_node_id_);
    const auto roots = G->get_nodes_by_type("root");
    if (not node.has_value() or not room.has_value() or roots.empty()) return;
    const auto Tr = inner_eigen_->get_transformation_matrix(room->name(), node->name(), 0);
    const auto Tw = inner_eigen_->get_transformation_matrix(roots.front().name(), node->name(), 0);
    if (not Tr.has_value() or not Tw.has_value()) return;
    EntryMirror em;
    const Eigen::Matrix2f R = Tr->rotation().topLeftCorner<2, 2>().cast<float>();
    const Eigen::Vector2f o = Tr->translation().head<2>().cast<float>();
    em.has_leaf       = true;
    em.hinge_local    = R.transpose() * (inst.leaf_pose.hinge_xy - o);
    em.open_local     = R.transpose() * (inst.leaf.swing * inst.aperture.across_u());
    em.phi_est        = inst.phi_est;
    em.phi_sigma      = inst.phi_sigma;
    em.hinge_llr      = inst.hinge_llr;
    em.swing_llr      = inst.swing_llr;
    em.phi_curve      = inst.phi_curve;
    em.matched_frames = inst.matched_frames;
    if (inst.existence_seeded) em.existence_L = inst.existence.logodds();
    em.T_root_door    = Eigen::Affine3d(Tw->matrix());
    em.width  = G->get_attrib_by_name<width_m_att>(node.value()).value_or(cfg_.door_prior_w_m);
    em.depth  = G->get_attrib_by_name<depth_m_att>(node.value()).value_or(cfg_.door_thickness_m);
    em.height = G->get_attrib_by_name<height_m_att>(node.value()).value_or(cfg_.door_prior_h_m);
    if (const auto v = G->get_attrib_by_name<door_open_prob_att>(node.value()); v.has_value()) em.open_prob = v.value();
    if (const auto v = G->get_attrib_by_name<door_phi_rad_att>(node.value()); v.has_value())   em.phi_rad = v.value();
    em.name = node->name();
    for (const auto& r : G->get_nodes_by_type("room"))
        if (r.id() != room_node_id_)
            if (const auto Tx = inner_eigen_->get_transformation_matrix(r.name(), node->name(), 0);
                Tx.has_value() and Tx->matrix().allFinite())
                em.T_in_room[r.id()] = Eigen::Affine3d(Tx->matrix());
    door_memo_[inst.node_id] = std::move(em);
}

std::uint64_t SpecificWorker::adopt_doorway_name(std::uint64_t id, std::uint64_t room_id, const std::string& name)
{
    if (name.empty() or not scene_graph_ or not inner_eigen_) return 0;
    const auto node = G->get_node(id);
    const auto room = G->get_node(room_id);
    if (not node.has_value() or not room.has_value() or node->name() == name) return 0;
    if (G->get_node(name).has_value())
    {
        std::print("door_concept: [identity] {} keeps its own name: '{}' is still a live node (the room left behind "
                   "is still in the working memory)\n", node->name(), name);
        return 0;
    }
    const auto T = inner_eigen_->get_transformation_matrix(room->name(), node->name(), 0);
    if (not T.has_value() or not T->matrix().allFinite()) return 0;
    const Eigen::Matrix3d R = T->rotation();
    const Eigen::Vector3f euler(static_cast<float>(std::atan2(-R(1, 2), R(2, 2))),
                                static_cast<float>(std::asin(std::clamp(R(0, 2), -1.0, 1.0))),
                                static_cast<float>(std::atan2(-R(0, 1), R(0, 0))));
    const auto op = G->get_attrib_by_name<door_open_prob_att>(node.value());
    const auto ph = G->get_attrib_by_name<door_phi_rad_att>(node.value());
    const auto names = reserved_names();
    const auto nid = scene_graph_->create_door_copy(
        room.value(),
        G->get_attrib_by_name<width_m_att> (node.value()).value_or(cfg_.door_prior_w_m),
        G->get_attrib_by_name<depth_m_att> (node.value()).value_or(cfg_.door_thickness_m),
        G->get_attrib_by_name<height_m_att>(node.value()).value_or(cfg_.door_prior_h_m),
        op.has_value() ? std::optional<float>(op.value()) : std::nullopt,
        ph.has_value() ? std::optional<float>(ph.value()) : std::nullopt,
        T->translation().cast<float>(), euler, names, std::format("'{}' id={} (renamed)", node->name(), id), name);
    if (nid == 0) return 0;
    if (fitter_ and fitter_->instances().contains(id)) fitter_->forget_node(id);
    G->delete_node(id);
    std::print("door_concept: [identity] {} is the doorway '{}' seen from '{}' — re-made under that name (id {})\n",
               node->name(), name, room->name(), nid);
    return nid;
}

std::optional<std::pair<std::uint64_t, SpecificWorker::EntryMirror>> SpecificWorker::recreate_lost_door(std::uint64_t new_room)
{
    if (lost_doors_.empty() or not inner_eigen_ or not scene_graph_) return std::nullopt;
    const auto room = G->get_node(new_room);
    const auto roots = G->get_nodes_by_type("root");
    if (not room.has_value() or roots.empty()) return std::nullopt;
    const auto Trb = inner_eigen_->get_transformation_matrix(roots.front().name(), "body", 0);
    const auto Tnr = inner_eigen_->get_transformation_matrix(room->name(), roots.front().name(), 0);
    if (not Trb.has_value() or not Tnr.has_value()) return std::nullopt;
    const Eigen::Vector2d robot = Trb->translation().head<2>();
    const auto best = std::ranges::min_element(lost_doors_, {}, [&](const EntryMirror& m)
                                               { return (m.T_root_door.translation().head<2>() - robot).norm(); });
    // Prefer the pose measured directly in this room while the door still lived; the root route is stale by however
    // far the robot drove between the crossing and the eviction.
    const auto direct = best->T_in_room.find(new_room);
    const Eigen::Affine3d T = direct != best->T_in_room.end() ? direct->second
                                                              : Eigen::Affine3d(Tnr->matrix()) * best->T_root_door;
    std::print("door_concept: [room] recreating '{}' from its pose {} \n", best->name,
               direct != best->T_in_room.end() ? "measured in this room before it was deleted" : "via root (fallback)");
    // A room just born has no valid robot↔room link yet (room_concept retires it at a birth): its chain can resolve to
    // garbage (live try35: t = (3.7e13, 5.2e12) m). A door recreated there would be wrong for ever — refuse it; the
    // next room change, or the tracker, will bring the door back from evidence.
    if (not T.matrix().allFinite() or T.translation().head<2>().norm() > 1.0e3)
    {
        std::print("door_concept: [room] NOT recreating '{}' in '{}': the room<-root transform is not valid yet "
                   "(|t| = {:.3g} m)\n", best->name, room->name(), T.translation().head<2>().norm());
        return std::nullopt;
    }
    const Eigen::Matrix3d R = T.rotation();
    const Eigen::Vector3f euler(static_cast<float>(std::atan2(-R(1, 2), R(2, 2))),
                                static_cast<float>(std::asin(std::clamp(R(0, 2), -1.0, 1.0))),
                                static_cast<float>(std::atan2(-R(0, 1), R(0, 0))));
    const auto names = reserved_names();
    const auto id = scene_graph_->create_door_copy(room.value(), best->width, best->depth, best->height, best->open_prob,
                                                   best->phi_rad, T.translation().cast<float>(), euler, names,
                                                   std::format("'{}' (deleted with the room left behind)", best->name),
                                                   best->name);
    if (id == 0) return std::nullopt;
    std::print("door_concept: [room] recreated the door just crossed ('{}', {:.2f} m from the robot) in '{}'\n",
               best->name, (best->T_root_door.translation().head<2>() - robot).norm(), room->name());
    return std::make_pair(id, *best);
}
