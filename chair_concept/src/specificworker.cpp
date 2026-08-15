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
 * SpecificWorker — chair_concept agent
 *
 * Implements the Active Inference loop described in ../CONCEPT_AGENT_RECIPE.md:
 *
 *  ① Read sensing attributes from DSR chair nodes
 *  ② Update the historical sample queue with fresh near-surface candidates
 *  ③ Run gradient-descent steps on the 7-param generative model (SDF + FE)
 *  ④ Write updated model parameters back to DSR (RT edge + geometry attrs)
 *  ⑤ Check convergence and set model_stable_att
 *  ⑥ Compute epistemic action proposals (viewpoint → mission-controller)
 *  ⑦ Detect divergence and set request_full_sample_att
 */

#include "specificworker.h"
#include "../../common/exclusion/exclusion.h"   // rc::exclusion — the SHARED no-two-objects rule
#include "../../common/exclusion/exclusion.h"   // rc::exclusion:: (SHARED)
#include "../../common/footprint/footprint.h"   // rc::geom:: (SHARED)
#include "../../common/instance_tracker/birth_evidence.h"   // rc::birth:: the shared CREATE policy
#include <limits>   // numeric_limits<int>::max — the disabled tracker death counter
#include <QFontMetrics>
#include <QHBoxLayout>
#include <QPushButton>
#include "chair_dof.h"   // rc::kChairDofs — names/units for the BeliefInspector rows
#include "../../common/nbv/graph_obstacles.h"   // rc::nbv::collect_graph_obstacles — shared, DSR-side

#include <QTimer>
#include <QSettings>   // persist the standalone dashboard window geometry
#include <QByteArray>
#include <QDateTime>   // wall-clock ms for the primary-input (masks) stream gate
#include <filesystem>
#include <print>
#include <cstdlib>   // std::_Exit — crash-free terminal shutdown
#include <thread>    // brief DDS flush before _Exit
#include <chrono>
#include <fstream>   // per-cycle detection-slice diagnostic CSV
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
// per-DOF posterior stds of the POSE, from the AI2 covariance Σ over [cx,cy,yaw] (pose-only belief; size
// is a fixed template). Shrinks as the robot gathers viewpoints (mostly yaw, via the backrest).
float belief_uncertainty(const rc::ChairInstance& inst)
{
    if (not inst.ai2_initialized)
        return 0.0f;
    const auto& S = inst.ai2_belief.covariance();
    const auto sd = [&](int i) { return std::sqrt(std::max(0.0f, S(i, i))); };
    return sd(0) + sd(1) + sd(2);
}

// Tracker lifecycle event log (etc/chair_events.csv) — makes birth/merge/prune/suppress visible so the
// "create then remove" churn can be diagnosed from a file, not stdout. seq gives ordering.
void log_tracker_event(const char* ev, std::uint64_t id, float x, float y, const std::string& note)
{
    static std::ofstream f = [] { std::ofstream o("etc/chair_events.csv", std::ios::trunc);
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
    std::print("chair_concept: SpecificWorker destroyed.\n");
}

void SpecificWorker::request_shutdown()
{
    if (shutting_down_.exchange(true))
        return;

    save_window_settings();
    save_dashboard_geometry();   // the standalone dashboard is not in `windows`, so save it explicitly
    save_strip_geometry();       // …nor is the compact belief strip

    cleanup_owned_nodes();

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

// Persist/restore the standalone dashboard window's geometry. The generated save_window_settings()
// only covers the QMainWindow(s) in `windows`; our extracted top-level widget is separate, so we
// carry its own QSettings entry (mirrors room_concept's RoomViewer).
void SpecificWorker::restore_dashboard_geometry()
{
    if (not dashboard_window_)
        return;
    QSettings settings(QStringLiteral("RoboComp"), QStringLiteral("chair_concept"));
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
    QSettings settings(QStringLiteral("RoboComp"), QStringLiteral("chair_concept"));
    settings.setValue(QStringLiteral("DashboardWindow_geometry"), dashboard_window_->saveGeometry());
    settings.sync();
}

// ─── Initialisation ──────────────────────────────────────────────────────────

void SpecificWorker::initialize()
{
    std::print("chair_concept: initialize()\n");
    GenericWorker::initialize();
 
    // Ignore payload attributes in local graph updates to avoid unnecessary copying and processing of potentially large data
    G->set_ignored_attributes<cam_rgb_att, cam_depth_att, laser_X_att, laser_Y_att, laser_Z_att>();
    qInfo() << "Ignoring DSR RGBD payload attributes cam_rgb/cam_depth in local graph updates";


    if (not G)
    {
        qWarning() << "chair_concept: DSR graph not available in initialize()";
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
            // Pump the masks ingest WHILE Waiting: chair polls a graph node (no free-running ingest thread
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
            // Stale-node sweep on (re)entering Operating: remove any leftover affordance nodes from a
            // previous run (e.g. after a crash that skipped cleanup) so a fresh create doesn't collide
            // and get a DSR-generated name. Keyed on the parent object type, not the node name.
            remove_stale_affordance_nodes();
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
    scene_graph_ = std::make_unique<rc::ChairSceneGraph>(
        G, rt_api_.get(), cfg_, [this] { trigger_graph_layout_twopi(); });

    // Remove any "chair*" nodes left behind by a previous (crashed) run so this agent always starts
    // from a clean slate and never adopts a stale/drifted node (the instance tracker re-births them
    // data-driven from masks). Runs BEFORE the graph signals are connected: delete_node() fires
    // del_node_signal synchronously, and del_node_slot dereferences fitter_ — which does not exist yet.
    // Connecting after this sweep (and after fitter_ is built) both avoids that null-deref SIGSEGV and
    // skips a pointless self-notification for our own cleanup deletions.
    remove_owned_chair_nodes();

    // Shadow-mode birth/death record (CONCEPT_AGENT_LIFECYCLE.md §4.2). Recording only — see
    // log_phantom_event(). Truncating, like the sibling chair_events.csv: one file per run.
    phantom_log_.open("etc/chair_phantom_events.csv");

    // Resolve room node
    const auto rooms = G->get_nodes_by_type("room");
    if (not rooms.empty())
        room_node_id_ = rooms.front().id();
    else
        qWarning() << "chair_concept: no room node found at startup";

    // Active-inference fit core. Owns the instance map; collaborates with the ingestor + scene graph.
    fitter_ = std::make_unique<rc::ChairFitter>(
        G, inner_eigen_.get(), cfg_, mask_ingestor_.get(), scene_graph_.get());

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
    // ai2_log.csv frozen, Ctrl-C inert, right after a voxelizer restart. CLAUDE.md already states the rule
    // this violated: if you don't need a signal, don't connect it at all (bottle_concept connects none).
    // The two things the slot did are now POLLED once per cycle in poll_affordance_protocol().
    connect(G.get(), &DSR::DSRGraph::update_node_signal,
            this, &SpecificWorker::modify_node_slot);
    connect(G.get(), &DSR::DSRGraph::del_node_signal,
            this, &SpecificWorker::del_node_slot);

    // Part B: localization/chain covariance on the published RT edge (mirrors bottle/table).
    gaussian_api_ = std::make_unique<DSR::InnerGaussianAPI>(G.get());
    fitter_->set_chain_cov_source(gaussian_api_.get(), "zed", cfg_.rt_cov_add_chain);

    // Build rc::EpistemicPlanner (Σ-based D-optimal NBV) with the configured stand-off.
    epistemic_planner_ = rc::EpistemicPlanner(cfg_.obs_distance);
    // ONE detector envelope: the viewpoint the planner asks for is the argmax of the same model absence
    // should be weighted by. This REPLACES cfg_.min_standoff_m (ChairConcept.MinStandOffM), which was a
    // hand-picked stand-in for the near shoulder of exactly this curve.
    epistemic_planner_.set_detector_envelope(
        rc::detect::DetectorEnvelope{cfg_.detect_min_fill, cfg_.detect_max_fill, cfg_.detect_soft});
    epistemic_planner_.set_robot_radius(0.30f);   // Shadow's footprint radius

    // The camera's REAL geometry, read once (intrinsics and the zed mount are both static). BOTH FoVs: for a
    // chair the backrest makes the VERTICAL extent the binding axis at close range, and the horizontal-only
    // model this replaced was blind to it. ts==0 on the main thread — the only safe use of that cache.
    // ★The camera model is read PER CYCLE at the compute site (rc::nbv::sensor_from_graph),
    // NOT once here: the zed intrinsics are published by robot_concept when frames start
    // arriving, so reading them in initialize() races the producer. Losing that race leaves
    // vfov = 0, which silently collapses the fill model to horizontal-only — the exact bug
    // rc::nbv exists to fix, and it drives the robot nose-to-nose with tall objects.

    // Stale affordance nodes are swept on entering Operating (presence hook) and on shutdown — see
    // remove_stale_affordance_nodes(), keyed on the parent object type (robust to node-name renames).

    // ── Time-series dashboard — its OWN top-level window ──────────────────────
    // Extracted from the DSR graph dock (add_custom_widget_to_dock) into a standalone window so it shows
    // even with Agent.graph=false (this agent runs graph=false → no DSRViewer). Mirrors room_concept,
    // kinova_controller, table_concept and bottle_concept. TimeSeriesPlot is a plain QWidget (no QOpenGL
    // backing store), safe as a top-level. NOT WA_DeleteOnClose: closing must only HIDE it, or the
    // ts_*_plot_ pointers the compute() feed uses would dangle. A QApplication always exists (generated/main.cpp).
    {
        custom_widget_ = new Custom_widget("Chair — Free Energy, Surprise, Belief Uncertainty & Residuals");

        // Create plot inside frame_series
        auto* series_layout = new QVBoxLayout(custom_widget_->frame_series);
        series_layout->setContentsMargins(0, 0, 0, 0);
        custom_widget_->frame_series->setLayout(series_layout);

        ts_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
        ts_plot_->set_visible_window(60.f);
        series_layout->addWidget(ts_plot_, 1);

        // FE SURPRISE (attention signal) on its own panel — it lives on a much smaller scale (~0–1) than
        // the FE, so it needs the full panel height to be readable. Spikes when a chair moves, decays as
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
    evidence_monitor_ = new rc::EvidenceMonitor(QStringLiteral("chair_concept — evidence monitor"));

    // ── Combined window: counters (top) over the plots + belief inspector (bottom) ──
    // Identical three-section structure to every other concept agent. Only HIDDEN on close, never deleted
    // (compute() keeps the raw child pointers).
    dashboard_window_ = new QWidget;
    dashboard_window_->setWindowTitle(QStringLiteral("chair_concept — dashboard"));
    auto* outer = new QVBoxLayout(dashboard_window_);
    outer->setContentsMargins(0, 0, 0, 0);
    // No splitter: the counter strip is two lines of text with nothing to resize, so it simply takes
    // its natural height and the plots + inspector get everything else.
    outer->addWidget(evidence_monitor_, 0);   // section 1 — natural height
    outer->addWidget(custom_widget_, 1);      // sections 2 + 3 — all remaining space

    // NOT shown at startup: the compact strip below is the standing display, and this window is the
    // drill-down you ask for with its "details \u25B8" button. Geometry is still restored here, so the first
    // click puts it back exactly where you last left it.
    restore_dashboard_geometry();
    // EXPLICITLY hidden: restoreGeometry() also restores the window STATE, so relying on
    // "we never called show()" is fragile. The drill-down must start down.
    if (dashboard_window_) dashboard_window_->hide();

    // ── Compact belief strip — a SEPARATE, small top-level window ──────────────────────────────────
    strip_window_ = new QWidget;
    strip_window_->setWindowTitle(QStringLiteral("chair_concept \u2014 beliefs"));
    auto* strip_layout = new QVBoxLayout(strip_window_);
    strip_layout->setContentsMargins(0, 0, 0, 0);
    belief_strip_ = new rc::BeliefStrip(QStringLiteral("chairs"), strip_window_);
    belief_strip_->set_visible_window(60.f);
    strip_layout->addWidget(belief_strip_, 1);
    {
        auto* bar = new QHBoxLayout;
        bar->setContentsMargins(4, 0, 4, 3);
        bar->addStretch(1);
        auto* details = new QPushButton(QStringLiteral("details \u25B8"), strip_window_);
        details->setToolTip(QStringLiteral("show / hide the full dashboard: evidence counters, FE/surprise/\u03A3 "
                                           "time series, and the per-DOF belief inspector"));
        QFont bf = details->font(); bf.setPointSizeF(bf.pointSizeF() - 1.0); details->setFont(bf);
        details->setFixedHeight(QFontMetrics(bf).height() + 8);
        QObject::connect(details, &QPushButton::clicked, strip_window_, [this, details]()
        {
            if (not dashboard_window_) return;
            // TOGGLE: the button the dashboard came out of is the button it goes back into. A drill-down
            // that can only be OPENED is one that stays open — the 1180x900 window sits on top of
            // everything and the only way back to a clear screen is to hunt it down in the window list
            // and minimise it by hand.
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
        strip_layout->addLayout(bar, 0);
    }
    restore_strip_geometry();
    strip_window_->show();
}

// One row per chair: the adequacy gap (nats still missing before Σ meets the consumer's σ*), p(exists),
// the FE surprise and the node's birth stamp. The widget owns the history — this pushes the current
// instant only, on the same throttled tick as the two panels above.
void SpecificWorker::refresh_belief_strip()
{
    if (not belief_strip_)
        return;   // headless: nothing was built

    std::vector<rc::BeliefStripRow> rows;
    rows.reserve(fitter_->instances().size());
    for (const auto& [id, inst] : fitter_->instances())
    {
        rc::BeliefStripRow r;
        r.node        = inst.node_name;
        r.surprise    = inst.fe_surprise;
        r.initialized = inst.ai2_initialized;
        // exist_logodds is NaN until the existence channel initialises — leave p_exists NaN in that case
        // (the widget reads NaN as "this agent has no existence belief yet") rather than inventing 0.5.
        if (std::isfinite(inst.exist_logodds))
            r.p_exists = 1.0f / (1.0f + std::exp(-inst.exist_logodds));

        // The same REPORTED covariance the inspector and the NBV planner use (its yaw entry carries the
        // 4-mode orientation entropy), so the strip cannot disagree with either.
        const auto S = inst.ai2_belief.covariance_reported();
        r.gap_nats = rc::any_sigma_star(rc::kChairDofs)
                   ? rc::adequacy_gap_nats(rc::kChairDofs, [&](std::size_t j) { return S(j, j); })
                   : -1.0f;

        // Fallback channel: ½·ln det Σ via the Cholesky (Σ log L_ii), not log(det()) — a covariance with
        // centimetre σ has a determinant where a direct determinant is numerical noise.
        const auto llt = S.llt();
        if (llt.info() == Eigen::Success)
            r.logdet_nats = llt.matrixL().toDenseMatrix().diagonal().array().log().sum();

        if (const auto n = G->get_node(inst.node_id); n.has_value())
            r.birth_ms = G->get_attrib_by_name<timestamp_creation_att>(n.value()).value_or(0);

        rows.push_back(std::move(r));
    }
    belief_strip_->update_view(rows);
}

void SpecificWorker::restore_strip_geometry()
{
    if (not strip_window_)
        return;
    QSettings settings(QStringLiteral("RoboComp"), QStringLiteral("chair_concept"));
    const QByteArray geom = settings.value(QStringLiteral("BeliefStripWindow_geometry")).toByteArray();
    if (not geom.isEmpty())
    {
        strip_window_->restoreGeometry(geom);
        // ★A restored geometry carries the window STATE too. If the strip was ever maximised its saved
        // state brings it back filling the screen — indistinguishable, to the user, from the big dashboard
        // having opened itself. The strip is meant to sit in a corner, so refuse those states and cap the
        // size; the position is still honoured.
        strip_window_->setWindowState(strip_window_->windowState()
                                      & ~(Qt::WindowMaximized | Qt::WindowFullScreen));
        const QSize sz = strip_window_->size();
        strip_window_->resize(std::min(sz.width(), 900), std::min(sz.height(), 420));
    }
    else
        strip_window_->resize(520, 210);   // small ON PURPOSE — it sits in a corner, always open
}

void SpecificWorker::save_strip_geometry() const
{
    if (not strip_window_)
        return;
    QSettings settings(QStringLiteral("RoboComp"), QStringLiteral("chair_concept"));
    settings.setValue(QStringLiteral("BeliefStripWindow_geometry"), strip_window_->saveGeometry());
    settings.sync();
}

// ─── Belief inspector ────────────────────────────────────────────────────────

// Build the per-instance BELIEF snapshot (pose state, Σ, the 4-mode yaw posterior, scalar gauges) and push
// it to the bottom panel, throttled to ~5 Hz (a full card rebuild every compute cycle would waste the GUI
// thread). The chair has no EvidenceMonitor to share a tick with, so the gate lives here. Main-thread.
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

        // REPORTED covariance: σ_yaw carries the 4-mode orientation entropy, so an unresolved chair shows
        // the honest orientation uncertainty rather than the within-mode one.
        const auto  S = inst.ai2_belief.covariance_reported();
        const auto& s = inst.ai2_belief.state();
        const std::array<float, rc::ChairBelief::N> v = {s.cx, s.cy, s.yaw};
        for (int j = 0; j < rc::ChairBelief::N; ++j)
            c.dofs.push_back({rc::kChairDofs[j].name, rc::kChairDofs[j].unit, v[j],
                              std::sqrt(std::max(0.0f, S(j, j))), rc::kChairDofs[j].sigma_star});

        // Row-major copy, filled explicitly: Eigen stores column-major, and while Σ is symmetric today an
        // implicit .data() copy would silently transpose if that ever stopped being true.
        constexpr int N = rc::ChairBelief::N;
        c.cov.resize(N * N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                c.cov[i * N + j] = S(i, j);

        // The chair's discrete ambiguity is its 4-fold yaw symmetry (front/back/side of a squarish seat).
        const auto pm = inst.ai2_belief.mode_posterior();
        static const char* kYawLabels[4] = {"m0", "m0+90", "m0+180", "m0-90"};
        for (int k = 0; k < 4; ++k)
            c.modes.push_back({"yaw", kYawLabels[k], pm[k]});

        c.s.fe          = inst.dbg_energy;
        c.s.fe_baseline = inst.fe_baseline;
        c.s.fe_surprise = inst.fe_surprise;
        // exist_logodds is NaN until the existence channel initialises; pass it straight through so the
        // card prints "-" rather than a fake 0.5 probability.
        c.s.logodds     = inst.exist_logodds;
        if (std::isfinite(inst.exist_logodds))
            c.s.p_exists = 1.0f / (1.0f + std::exp(-inst.exist_logodds));
        c.s.age_s       = inst.last_belief_touch.time_since_epoch().count() == 0
                        ? -1.0f
                        : std::chrono::duration<float>(now - inst.last_belief_touch).count();
        c.s.since_det   = inst.frames_since_detection;
        c.s.initialized = inst.ai2_initialized;
        cards.push_back(std::move(c));
    }
    belief_inspector_->update_view(cards);
}

// ─── Main compute loop ───────────────────────────────────────────────────────

void SpecificWorker::compute()
{
    // ★ONE graph walk per cycle for the SHARED mutual-exclusion rule: who else claims room space.
    // Feeds BOTH the birth filter (a candidate on somebody else's object accrues no evidence) and
    // the existence occupancy discount. Main thread — collect_graph_obstacles uses ts==0 (CLAUDE.md).
    if (G) foreign_claims_ = rc::exclusion::foreign_claims(*G, inner_eigen_.get(), "chair");

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


    // Evidence-pipeline per-cycle counters (the *_cum fields persist). Producers below add to these; the
    // snapshot is pushed at the end of the cycle.
    ev_g_.births = ev_g_.merges = ev_g_.removals = 0;

    refresh_room_geometry();  // room-containment pose prior (cheap; the polygon is a nominal model)
    fitter_->update_ego_motion();   // robot/camera speed → "be-still-to-update" gate (once per cycle)
    mask_ingestor_->refresh();
    run_instance_tracker();   // data-driven birth/associate/death + merge (the only instance-lifecycle path)

    // Chairs are generic `object` nodes named "chair_*" (schema migration); filter by name prefix.
    const auto chair_nodes = G->get_nodes_by_type("object");
    for (const auto& node : chair_nodes)
        if (node.name().starts_with("chair"))
            process_chair_node(node);

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

    fps_counter_.print("[chair_concept Compute]");
}

// Collapse instances whose seat footprints overlap (same physical chair fitted twice): keep the one with
// more integrated fresh evidence, retire the other (affordance + node). Runs before tracking so a
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

            const auto& sa = ia->second.model.state();
            const auto& sb = ib->second.model.state();
            // SHARED exact polygon clip (common/footprint) — this was a byte-level duplicate of it.
            const float ratio = rc::geom::overlap_ratio({sa.cx, sa.cy, sa.seat_w, sa.seat_d, sa.yaw},
                                                        {sb.cx, sb.cy, sb.seat_w, sb.seat_d, sb.yaw});
            if (ratio < cfg_.tracker_merge_overlap) continue;

            const bool keep_i = ia->second.matched_frames >= ib->second.matched_frames;
            const std::uint64_t keep = keep_i ? ids[i] : ids[j];
            const std::uint64_t drop = keep_i ? ids[j] : ids[i];
            std::print("chair_concept: [tracker] MERGE id={} into id={} (footprint overlap {:.2f})\n",
                       drop, keep, ratio);
            log_tracker_event("MERGE", drop, ia->second.model.state().cx, ia->second.model.state().cy,
                              std::format("into {} overlap {:.2f}", keep, ratio));
            if (auto it = insts.find(drop); it != insts.end())
                it->second.affordance.remove();
            fitter_->forget_node(drop);
            G->delete_node(drop);
            removed.insert(drop);
            if (drop == ids[i]) break;   // this i is gone; advance to the next i
        }
    }
}

// Data-driven multi-instance lifecycle (mirrors table_concept). Chairs are persistent furniture, so
// death is OFF by default (removed only by MERGE); birth_min_sep is wide. The only instance-lifecycle path.
void SpecificWorker::run_instance_tracker()
{
    merge_overlapping_instances();   // enforce physical exclusion before associating/birthing this cycle

    rc::TrackerParams tp;
    tp.gate_mahalanobis = cfg_.tracker_gate_mahalanobis;
    tp.gate_fallback_m  = cfg_.tracker_gate_fallback_m;
    tp.detection_noise_m = cfg_.tracker_detection_noise_m;
    tp.birth_frames     = cfg_.tracker_birth_frames;
    // ★Invariant 5: removal is a Bayesian decision on the existence log-odds, NEVER a miss counter. An
    //    armed death counter beside a live existence channel is a SECOND removal authority, and it is the
    //    one that carries no evidence and leaves no attributable record — a phantom analysis cannot tell a
    //    reasoned removal from a timeout. Tying it to the existence flag makes the two mutually exclusive
    //    by construction, and keeps them A/B-able: turn the channel off and the counter comes back exactly.
    //    chair removes on `exist_logodds < exist_remove_logodds` (Existence.Enabled, default true).
    tp.death_frames     = cfg_.exist_enabled ? std::numeric_limits<int>::max()
                                            : cfg_.tracker_death_frames;
    tp.birth_min_sep_m  = cfg_.tracker_birth_min_sep_m;
    tp.nll_cost         = cfg_.tracker_nll_cost;
    tracker_.set_params(tp);

    // Tracks ← live instances: centre from the fit, XY cov from the belief's position covariance Σ.
    std::vector<rc::TrackView> tracks;
    tracks.reserve(fitter_->instances().size());
    for (auto& [id, inst] : fitter_->instances())
    {
        rc::TrackView t;
        t.id = id;
        const auto& s = inst.model.state();
        t.xy = {s.cx, s.cy};
        // Negative-information for the tracker's death path (if DeathEnabled): a miss counts only when the
        // chair projects into the camera FoV. Out-of-view → HOLD (matches the prune gate above). roi_valid
        // is a cycle stale (set in run_inference) but that's fine for a frustum test.
        t.expected_visible = inst.roi_valid;
        if (inst.ai2_initialized)
        {
            // Gate association on the belief's position cov (+ chain) → Mahalanobis S = P + R²I.
            const auto& S = inst.ai2_belief.covariance();
            t.cov = Eigen::Matrix2f::Zero();
            t.cov(0, 0) = S(0, 0) + inst.chain_cov_xx;
            t.cov(1, 1) = S(1, 1) + inst.chain_cov_yy;
            t.has_cov = true;
        }
        tracks.push_back(t);
        inst.assigned_mask_idx = -1;   // cleared; re-set below only if associated this cycle
    }

    // Detections ← this frame's "chair" mask slices (carry the slice index for the assignment).
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
    const rc::birth::Detectability birth_detect{cfg_.ai2_periph_ref, 2.5f, 2.0f};

    if (pkt.valid)
        for (int i = 0; i < static_cast<int>(pkt.slices.size()); ++i)
        {
            const auto& sl = pkt.slices[i];
                // ★★ONLY THE FRONT RGB-D CAMERA MAY CREATE OR UPDATE AN OBJECT. `has_depth` is NOT that
                // question: once the producer began depth-filling ricoh masks from reprojected LiDAR it
                // publishes them as full 3D slices with has_depth = 1, so a 360° detection from BEHIND the
                // robot passed every guard written as `if (has_depth)`. Reported live on bottle_concept —
                // moving and cloning with the robot facing away, 3 m off. mask_source says which camera,
                // unambiguously, and the voxelizer has been publishing it all along. A ricoh slice may
                // still CONFIRM a live instance (bearing_confirm) or raise a proto-object to go and look
                // at; it may not move one. See MaskIngestor::MaskSlice::may_fit_geometry.
            if (sl.label != "chair" or sl.support_end <= sl.support_begin
                or not sl.may_fit_geometry()) continue;
            rc::DetectionView dv;
            dv.xy = Eigen::Vector2f(sl.centroid.x(), sl.centroid.y());
            dv.slice_index = i;
            // ZED-only BIRTH: only a ZED slice (per-pixel depth ⇒ depth_var==0) may SPAWN a chair. A ricoh
            // LiDAR-reprojected-depth slice (depth_var>0) has unreliable depth/extent, so it may ASSOCIATE to /
            // confirm an existing chair but must NOT birth a phantom — this is exactly what created chair_3 (a
            // wrong ricoh detection at 8.8 m, outside the room). A confident-ricoh escape hatch is OFF by default.
            const bool is_zed = sl.depth_var == 0.0f;
            dv.birthable = is_zed or (cfg_.ricoh_birth_enabled
                                      and sl.confidence >= cfg_.ricoh_birth_conf
                                      and sl.depth_var  <= cfg_.ricoh_birth_max_var);
            dv.support = sl.support_end - sl.support_begin;   // for the fragment test below
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
                const float unclaimed = rc::exclusion::p_unclaimed(
                    {dv.xy.x(), dv.xy.y(), cfg_.tracker_birth_seat_w, cfg_.tracker_birth_seat_d, 0.0f}, foreign_claims_, &who);
                dv.birth_evidence *= unclaimed;
                if (unclaimed < 0.99f)
                    std::print("[chair] birth cand CLAIMED by '{}' ({:.0f}%): birth_ev x{:.2f}\n",
                               who ? who->node : "?", 100.0f * (1.0f - unclaimed), unclaimed);
            }
            dets.push_back(dv);
        }

    // ── A FRAGMENT IS NOT AN OBJECT ───────────────────────────────────────────────────────────────
    // YOLO splits one chair into two masks routinely — a backrest and a seat, when a table occludes the
    // middle. Each fragment then births its own instance, and NOTHING downstream can undo it: both
    // instances win a distinct mask every cycle, so the existence channel correctly holds both at L = +4
    // (it is not a phantom in the evidence sense — it has real, separate support), and the merge cannot
    // fire because two 0.5 m footprints 0.68 m apart do not overlap AT ALL. Measured live 2026-08-11:
    // chair_3 carried a median of 156 support points against chair_2's 2485, six per cent of its
    // neighbour, and sat at L = +4 indefinitely.
    //
    // ★THE BIRTH GATE CANNOT CATCH THIS AND NO VALUE OF IT COULD. At the moment of that birth the two
    // centroids were 0.75 m apart against a 0.70 m separation gate — legal by five centimetres — and they
    // only settled to 0.68 m afterwards, by which time births are no longer checked. Raising the gate to
    // 0.8 m would forbid genuinely adjacent dining chairs, which is a worse error.
    //
    // The discriminator is SIZE, and it is available for free: both masks arrive in the SAME FRAME at
    // essentially the same range, so their support counts are directly comparable with no range
    // normalisation. A detection carrying a small fraction of a much larger same-label mask, close enough
    // to be part of it, is a piece of that mask.
    if (cfg_.birth_fragment_frac > 0.0f)
        for (auto& d : dets)
        {
            if (not d.birthable) continue;
            for (const auto& big : dets)
            {
                if (&big == &d or big.support <= 0) continue;
                if (static_cast<float>(d.support) >= cfg_.birth_fragment_frac * static_cast<float>(big.support))
                    continue;                                   // comparable size ⇒ its own object
                if ((d.xy - big.xy).norm() > cfg_.birth_fragment_reach_m) continue;   // too far to be a piece
                d.birthable = false;
                std::print("chair_concept: [birth] FRAGMENT suppressed: {} pts at {:.2f} m from a {}-pt mask "
                           "({:.0f}% of it) — a piece, not a chair\n",
                           d.support, (d.xy - big.xy).norm(), big.support,
                           100.0f * static_cast<float>(d.support) / static_cast<float>(big.support));
                break;
            }
        }

    // DIAGNOSTIC (merged-vs-single mask): one CSV row per "chair" slice per cycle — count, size, centroid,
    // range. If a cluttered scene collapses to ONE big slice (npts ≫ a clean single chair) the extra chairs
    // never get born because no separate detection ever arrives — the failure is upstream, not in birth. File
    // truncated once per process launch.
    {
        static std::ofstream dcsv = []
        {
            std::ofstream f("etc/chair_dets_log.csv", std::ios::trunc);
            f << "cycle,n_chair_slices,slice_idx,npts,conf,cx,cy,range,trunc_frac,motion_var\n";
            return f;
        }();
        static int dcyc = 0;
        ++dcyc;
        if (dcsv)
        {
            if (dets.empty())
                dcsv << dcyc << ",0,-1,0,0,0,0,0,0,0\n";
            for (const auto& d : dets)
            {
                const auto& sl = pkt.slices[d.slice_index];
                const std::size_t n = (sl.support_end > sl.support_begin) ? (sl.support_end - sl.support_begin) : 0;
                dcsv << dcyc << ',' << dets.size() << ',' << d.slice_index << ',' << n << ',' << sl.confidence
                     << ',' << sl.centroid.x() << ',' << sl.centroid.y() << ',' << sl.range << ','
                     << sl.trunc_frac << ',' << sl.motion_var << '\n';
            }
            dcsv.flush();
        }
    }

    const auto res = tracker_.update(tracks, dets);

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
        std::print("[tracker] instances={} chair_dets={} assigned={} unassigned={} births={} deaths={}\n",
                   tracks.size(), dets.size(), n_assigned,
                   static_cast<int>(dets.size()) - n_assigned, res.births.size(), res.deaths.size());

    // DEATH: OFF by default — a chair is persistent furniture; long occlusion is not absence. Enable
    // Tracker.DeathEnabled to restore miss-timer retirement.
    if (cfg_.tracker_death_enabled)
        for (const std::uint64_t id : res.deaths)
        {
            std::print("chair_concept: [tracker] DEATH id={} (unobserved {} frames)\n", id, cfg_.tracker_death_frames);
            if (auto it = fitter_->instances().find(id); it != fitter_->instances().end())
                it->second.affordance.remove();
            fitter_->forget_node(id);
            G->delete_node(id);
        }

    // ASSOCIATE: route each detection's mask slice to its instance (read in observe()).
    {
        // §3.1 (Fable, PERCEPTION_ASSOCIATION_PLAN.md): associate by MODEL EVIDENCE, not centroid. The
        // teleport was: a merged mask's centroid lands MIDWAY between two chairs → falls in the wrong
        // track's Mahalanobis gate → greedy flip. Instead score each (initialised instance × chair slice)
        // by the instance's belief NLL on the slice's POINTS — a merged mask fits NO single-chair model, and
        // each instance keeps the slice that best matches ITS geometry. Greedy lowest-nll, 1-to-1, no gate.
        // ★The evidence is `association_nll` = the mixture NLL, NOT `mean_energy`: mean_energy weights the SDF
        // by responsibility, so a FAR slice (all points → clutter) scored ~0 = a PERFECT match → a distant
        // instance mis-claimed the 3rd chair's slice and SUPPRESSED its birth (chair "never seen"). The
        // mixture NLL counts the clutter cost, so a far/clutter'd slice scores HIGH → unclaimed → it births.
        struct EvPair { float e; std::uint64_t id; int slice; };
        std::vector<EvPair> pairs;
        const float R = cfg_.ai2_sigma_base_m * cfg_.ai2_sigma_base_m;
        const float rn2 = cfg_.tracker_detection_noise_m * cfg_.tracker_detection_noise_m;
        for (auto& [id, inst] : fitter_->instances())
        {
            if (not inst.ai2_initialized) continue;
            const auto& bs = inst.ai2_belief.state();
            const auto& BS = inst.ai2_belief.covariance();
            const float sxx = BS(0, 0) + inst.chain_cov_xx + rn2;   // innovation cov diag S = P + R²I
            const float syy = BS(1, 1) + inst.chain_cov_yy + rn2;
            for (const auto& d : dets)
            {
                // POSITION GATE (mirrors the tracker's S=P+R²I Mahalanobis): an instance may claim a slice
                // ONLY if the slice centroid is within its gate. Without it the evidence greedy assigns EVERY
                // instance some slice, so an instance whose own chair is occluded this frame claims a FAR
                // chair's slice (high nll, but the least-bad available) → suppresses THAT chair's birth AND
                // teleports the instance (the "3rd chair never seen" + wrong-pose bug). Gate by position,
                // rank by shape-evidence = the correct combination.
                const float ex = d.xy.x() - bs.cx, ey = d.xy.y() - bs.cy;
                const float m2 = ex * ex / std::max(1e-9f, sxx) + ey * ey / std::max(1e-9f, syy);
                if (m2 > cfg_.tracker_gate_mahalanobis) continue;   // outside the gate → not claimable
                const auto& sl = pkt.slices[d.slice_index];
                const std::size_t b = std::min<std::size_t>(sl.support_begin, pkt.support_points.size());
                const std::size_t e = std::min<std::size_t>(sl.support_end,   pkt.support_points.size());
                if (e <= b) continue;
                const std::vector<Eigen::Vector3f> pts(pkt.support_points.begin() + b, pkt.support_points.begin() + e);
                pairs.push_back({inst.ai2_belief.association_nll(pts, R), id, d.slice_index});
            }
        }
        std::sort(pairs.begin(), pairs.end(), [](const EvPair& a, const EvPair& b) { return a.e < b.e; });
        std::unordered_set<std::uint64_t> used_inst; std::unordered_set<int> used_slice;
        for (const auto& p : pairs)
        {
            if (used_inst.count(p.id) or used_slice.count(p.slice)) continue;
            fitter_->instances().at(p.id).assigned_mask_idx = p.slice;
            used_inst.insert(p.id); used_slice.insert(p.slice);
        }

        // Pass 2 — INITIALISATION: a freshly-born instance has no belief yet (Pass 1 skips it), so it has no
        // way to earn its first mask. Assign it the nearest UNUSED slice within the metric fallback gate of
        // its BIRTH position, so it initialises AT the chair it was born from — NOT the nearest chair (that
        // was the teleport: a far-born instance grabbed a near chair's mask when its own was occluded, then
        // merged, so the far chair never persisted). If no slice sits at its birth spot this frame it stays
        // unassigned (frozen) — no teleport.
        const float fb2 = cfg_.tracker_gate_fallback_m * cfg_.tracker_gate_fallback_m;
        for (auto& [id, inst] : fitter_->instances())
        {
            if (inst.ai2_initialized or used_inst.count(id)) continue;
            const auto& ms = inst.model.state();
            int best = -1; float best_r2 = fb2;
            for (const auto& d : dets)
            {
                if (used_slice.count(d.slice_index)) continue;
                const float ex = d.xy.x() - ms.cx, ey = d.xy.y() - ms.cy, r2 = ex * ex + ey * ey;
                if (r2 < best_r2) { best_r2 = r2; best = d.slice_index; }
            }
            if (best >= 0) { inst.assigned_mask_idx = best; used_slice.insert(best); used_inst.insert(id); }
        }
    }

    // EXISTENCE BELIEF: continuous log-odds removal — the principled replacement for the wall-clock
    // stillbirth prune below. The prune could not kill a MATURE phantom (processed_cycles≥maturity → "furniture"
    // immunity, purely age-based) and its binary streak reset on ANY assignment, so a phantom fed a trickle of
    // clutter (chair_3: 8 points at 7.4 m) never reached patience AND aged into permanence. The existence belief
    // fixes both: it integrates SUPPORT-MASS-WEIGHTED evidence (8 pts where ~130 are expected → strong negative)
    // once per sensor frame, with no age immunity — a real chair stays only by continuing to be explained.
    if (cfg_.exist_enabled)
        update_existence_beliefs();
    else if (cfg_.tracker_prune_enabled)
    {
        std::vector<std::uint64_t> stillborn;
        for (auto& [id, inst] : fitter_->instances())
        {
            if (inst.assigned_mask_idx >= 0) { inst.unassigned_streak = 0; continue; }
            // Negative-information: an unassigned cycle is evidence of a PHANTOM only when the chair SHOULD
            // be seen — its model projects into the camera FoV (roi_valid) yet no mask associated. When the
            // chair is out of view (robot looked away) the streak is HELD, so a real chair glimpsed once and
            // left behind persists as furniture instead of being pruned within seconds (the flicker). A true
            // phantom sits at a detected location → projects in-frame → still accrues the streak and is pruned.
            if (not inst.roi_valid) continue;
            ++inst.unassigned_streak;
            const bool young = inst.processed_cycles < cfg_.tracker_prune_maturity_cycles;
            if (young and inst.unassigned_streak >= cfg_.tracker_prune_patience)
                stillborn.push_back(id);
        }
        for (const std::uint64_t id : stillborn)
        {
            const auto it = fitter_->instances().find(id);
            std::print("chair_concept: [tracker] PRUNE stillborn id={} (unassigned {} cycles, age {} < maturity {})\n",
                       id, it != fitter_->instances().end() ? it->second.unassigned_streak : 0,
                       it != fitter_->instances().end() ? it->second.processed_cycles : 0,
                       cfg_.tracker_prune_maturity_cycles);
            if (it != fitter_->instances().end())
                log_tracker_event("PRUNE", id, it->second.model.state().cx, it->second.model.state().cy,
                                  std::format("unassigned {} age {}", it->second.unassigned_streak, it->second.processed_cycles));
            if (it != fitter_->instances().end())
                it->second.affordance.remove();
            fitter_->forget_node(id);
            G->delete_node(id);
        }
    }

    // BIRTH: spawn an instance from each promoted (persistently-unexplained) detection, seeding the
    // fitter with the detection XY so the model starts AT the chair (not the 0,0 RT-read default).
    for (const int d : res.births)
    {
        const int slice = dets[d].slice_index;
        // §3.1: under AI2 the assignment is by belief evidence, not the tracker's centroid — so a slice an
        // existing instance already claimed by mean_energy must NOT also spawn a phantom (the tracker's
        // centroid birth and the evidence assignment can disagree). Full birth-validity (P(v=clean) — don't
        // birth from a merged/contaminated mask at all) is §2, still to come.
        const Eigen::Vector3f& c = pkt.slices[slice].centroid;

        // Room-containment pose prior at BIRTH: never spawn a chair outside the walls (a mislocalized frame while
        // the robot is lost drops a detection beyond a wall). Zero prior mass outside the room → suppress.
        if (cfg_.exist_room_prior and fitter_->has_room_polygon()
            and not fitter_->point_in_room(Eigen::Vector2f(c.x(), c.y()), cfg_.exist_room_margin_m))
        {
            std::print("chair_concept: [tracker] BIRTH SUPPRESSED slice={} at ({:.2f},{:.2f}) — OUTSIDE room\n",
                       slice, c.x(), c.y());
            log_tracker_event("SUPPRESS", 0, c.x(), c.y(), "outside room");
            continue;
        }
        {
            std::uint64_t claimer = 0;
            for (auto& [id, inst] : fitter_->instances())
                if (inst.assigned_mask_idx == slice) { claimer = id; break; }
            if (claimer != 0)
            {
                // DIAGNOSTIC (missing-chair): a persistently-detected cluster that never instantiates is
                // usually a birth SUPPRESSED here — a distant instance's belief mis-claimed this slice by
                // mean_energy, stealing it from birth (and likely teleporting toward it). Surface which
                // instance claimed it and how far it sits, so the failure is visible, not silent.
                float dist = -1.0f;
                if (auto it = fitter_->instances().find(claimer); it != fitter_->instances().end())
                {
                    const auto& st = it->second.model.state();
                    dist = std::hypot(st.cx - c.x(), st.cy - c.y());
                }
                std::print("chair_concept: [tracker] BIRTH SUPPRESSED slice={} at ({:.2f},{:.2f}) — claimed by "
                           "id={} ({:.2f} m away)\n", slice, c.x(), c.y(), claimer, dist);
                log_tracker_event("SUPPRESS", claimer, c.x(), c.y(), std::format("claimer {:.2f}m", dist));
                continue;
            }
        }
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
                it->second.exclusion.resolve_at_birth({c.x(), c.y(), cfg_.tracker_birth_seat_w, cfg_.tracker_birth_seat_d, 0.0f},
                                                      foreign_claims_);
            // Materialise the ChairInstance NOW, at birth — do not wait for the freshly inserted DSR node
            // to surface in get_nodes_by_type (it is not reliably visible the same cycle). Birth was
            // decoupled across three async steps (create node → ensure_instance when the node appears →
            // belief init when a mask associates), so a born instance was never registered as a track:
            // the tracker kept seeing its detection unexplained and RE-BIRTHED it every cycle (the
            // 403-birth / 401-merge churn, duplicates collapsing at 0,0). Creating the instance here
            // registers it as a track immediately and hands it its birth slice so observe()/run_inference
            // initialise the belief AT the detection centroid this cycle — birth, track, and init are now atomic.
            if (const auto nopt = G->get_node(new_id); nopt.has_value())
            {
                fitter_->ensure_instance(nopt.value(), room_node_id_);
                if (auto it = fitter_->instances().find(new_id); it != fitter_->instances().end())
                    it->second.assigned_mask_idx = slice;
            }
            log_tracker_event("BIRTH", new_id, c.x(), c.y(), "");
            // Shadow-mode birth record (§4.2): a phantom is a birth that dies young from a confident view,
            // so the birth half captures the place AND the viewpoint that produced it.
            log_phantom_event("BIRTH", new_id, "", c.x(), c.y(), nullptr, "");
        }
    }

    // ── Part C-birth: NEW-object hypotheses from unmatched 360 bearings ──────────────────────────────
    // A peripheral "chair" bearing (a no-depth 360 slice) that lines up with no live chair and PERSISTS
    // births a broad-Σ hypothesis: mean placed on the ray at a nominal range, Σ huge along the ray (range
    // unknown) and tight across it (bearing known). The hypothesis authors an Orient affordance (rotate to
    // look); a depth mask then collapses Σ, or it dies unobserved. Default OFF (Bearing.BirthEnabled).
    if (cfg_.bearing_birth_enabled and pkt.valid)
    {
        std::vector<rc::BearingDetectionView> bearings;
        for (int i = 0; i < static_cast<int>(pkt.slices.size()); ++i)
            if (pkt.slices[i].label == "chair" and not pkt.slices[i].has_depth)
                bearings.push_back({pkt.slices[i].azimuth_room_rad, i});

        if (not bearings.empty())
        {
            Eigen::Vector2f robot_xy(0.f, 0.f);
            if (const auto p = inner_eigen_->transform("room", Eigen::Vector3d::Zero(), "zed"); p.has_value())
                robot_xy = {static_cast<float>(p->x()), static_cast<float>(p->y())};

            // A bearing that lines up with a live chair is "explained" (confirmation, not new); the rest go
            // to the gap-tolerant stager, and a persistent unmatched bearing promotes to a hypothesis birth.
            const auto confirmed = rc::confirm_tracks_by_bearing(tracks, bearings, robot_xy, cfg_.bearing_confirm_gate_rad);
            std::vector<char> matched(bearings.size(), 0);
            for (const auto& cf : confirmed)
                for (int b = 0; b < static_cast<int>(bearings.size()); ++b)
                    if (bearings[b].slice_index == cf.slice_index) matched[b] = 1;
            std::vector<float> unmatched;
            for (int b = 0; b < static_cast<int>(bearings.size()); ++b)
                if (not matched[b]) unmatched.push_back(bearings[b].azimuth_room_rad);

            bearing_stager_.set_params(cfg_.bearing_birth_frames, cfg_.bearing_match_rad, cfg_.bearing_max_miss);
            for (const float az : bearing_stager_.update(unmatched))
            {
                // Anti-dup: skip if a live chair already sits near the nominal point on this ray.
                const Eigen::Vector2f p_nom = robot_xy + cfg_.bearing_nominal_range_m * Eigen::Vector2f(std::cos(az), std::sin(az));
                bool near_existing = false;
                for (const auto& t : tracks)
                    if ((t.xy - p_nom).norm() < cfg_.tracker_birth_min_sep_m) { near_existing = true; break; }
                if (near_existing) continue;

                const Eigen::Vector3f c_room(p_nom.x(), p_nom.y(), cfg_.ai2_floor_z);
                const auto new_id = scene_graph_->create_instance_from_detection(c_room, room_node_id_);
                if (new_id == 0) continue;
                if (const auto nopt = G->get_node(new_id); nopt.has_value())
                {
                    fitter_->ensure_instance(nopt.value(), room_node_id_);
                    if (auto it = fitter_->instances().find(new_id); it != fitter_->instances().end())
                        fitter_->seed_bearing_hypothesis(it->second, robot_xy, az, cfg_.bearing_nominal_range_m,
                                                         cfg_.bearing_along_std_m, cfg_.bearing_across_std_m,
                                                         cfg_.bearing_yaw_std_rad);
                }
                std::print("chair_concept: [bearing] BIRTH hypothesis id={} az={:.0f}deg (nominal {:.1f}m on ray)\n",
                           new_id, az * 180.0f / 3.14159265f, cfg_.bearing_nominal_range_m);
                log_tracker_event("BEARING_BIRTH", new_id, p_nom.x(), p_nom.y(), "");
            }
        }
    }
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
        const auto rooms = G->get_nodes_by_type("room");
        if (rooms.empty()) { room_node_id_ = 0; return; }
        room_node_id_ = rooms.front().id();
        room = G->get_node(room_node_id_);
        if (not room.has_value()) return;
    }
    const auto px = G->get_attrib_by_name<delimiting_polygon_x_att>(room.value());
    const auto py = G->get_attrib_by_name<delimiting_polygon_y_att>(room.value());
    if (not px.has_value() or not py.has_value())
    {
        if (++miss % 120 == 1)
            std::print("chair_concept: [room-prior] room node {} has NO delimiting_polygon attribute yet "
                       "(containment prior INACTIVE — out-of-room chairs cannot be removed)\n", room_node_id_);
        return;
    }
    const auto& xs = px->get(); const auto& ys = py->get();
    const std::size_t n = std::min(xs.size(), ys.size());
    if (n < 3)
    {
        if (++miss % 120 == 1)
            std::print("chair_concept: [room-prior] delimiting_polygon present but degenerate (n={})\n", n);
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
        std::print("chair_concept: [room-prior] room polygon LOADED — {} verts, x∈[{:.2f},{:.2f}] y∈[{:.2f},{:.2f}] "
                   "(containment prior ACTIVE)\n", n, xmin, xmax, ymin, ymax);
}

// Continuous existence belief: fold one sensor frame of evidence into each instance's log-odds L and remove
// any whose L crosses the floor. See the ChairInstance::exist_logodds comment for the model. Runs from
// run_instance_tracker (association already resolved, so assigned_mask_idx is this cycle's assignment).
// SHADOW-MODE birth/death recorder — CONCEPT_AGENT_LIFECYCLE.md §4.2, theory in MODEL_HISTORY.md §4.
// RECORDS ONLY; it can never alter a birth or a removal. chair_concept is the agent this matters most in:
// the "part of a radiator reads as a chair" failure is a CHAIR phenomenon, so this log is what actually tests
// whether phantom deaths cluster in (world cell × view bearing).
// The attribution fields decide whether a death was a CONFIDENT disconfirmation (a real phantom) or a weak
// one (more likely one of our own removal defects). chair has no silhouette channel, so P(detect) comes from
// ChairFitter::zed_detectability() — the same quantity that gates its vacate evidence.
void SpecificWorker::log_phantom_event(std::string_view event, std::uint64_t id, std::string_view name,
                                       float x, float y, const rc::ChairInstance* inst, std::string_view note)
{
    if (not phantom_log_.is_open())
        return;
    rc::history::PhantomEvent e;
    e.event = event; e.id = id; e.name = name; e.x = x; e.y = y; e.note = note;
    // Observer pose → view bearing. The classifier failure is VIEWPOINT-dependent (a radiator only reads as a
    // chair from certain angles), so the eventual p_FA field is keyed on (world cell × bearing); a place-only
    // key would suppress a genuine chair placed there from every direction.
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
    if (inst)   // death: carry the state that says whether this was a CONFIDENT kill
    {
        e.age_cycles    = inst->processed_cycles;
        e.p_detect      = fitter_ ? fitter_->zed_detectability(*inst) : 0.0f;
        e.in_fov_frac   = inst->roi_valid ? 1.0f : 0.0f;
        e.central_frac  = std::clamp(1.0f - inst->last_centroid_radius, 0.0f, 1.0f);
        e.fixated       = inst->dbg_fixated ? 1 : 0;
        e.exist_logodds = std::isnan(inst->exist_logodds) ? 0.0f : inst->exist_logodds;
    }
    phantom_log_.write(e);
}

void SpecificWorker::update_existence_beliefs()
{
    // Integrate at the SENSOR rate, not the compute rate: only when a new mask frame arrived. Otherwise a fast
    // compute loop would decay a briefly-occluded real chair away between two sensor frames.
    const auto& pkt = mask_ingestor_->packet();
    const bool sensor_fresh = pkt.valid and static_cast<int>(pkt.frame_id) != exist_last_mask_frame_;
    if (not sensor_fresh)
        return;
    exist_last_mask_frame_ = static_cast<int>(pkt.frame_id);

    if (exist_support_scale_ <= 0.0f)                 // lazy-seed the expected-support scale C = E[npts·range²]
        exist_support_scale_ = cfg_.exist_expected_support_c;

    const float g   = cfg_.exist_evidence_gain;
    const float ar  = cfg_.exist_adequacy_ref;
    const float cap = cfg_.exist_adequacy_cap;

    // "ZED removes": every mask frame is ZED-cadenced (upload_masks runs per ZED RGBD frame, ricoh slices appended),
    // so a fresh frame ALREADY means "ZED looked this cycle" — including a frame with ZERO ZED detections, which is
    // exactly ZED staring at empty space and finding nothing (the strongest vacate evidence). Do NOT gate on
    // "ZED produced a detection": that would refuse to remove a phantom precisely when ZED confirms it is gone.
    // ★ONE REMOVAL DECISION, THE SHARED ONE (rc::exist::decide_removal). chair was the last agent deciding
    // by hand — a bare `L < RemoveLogodds` with NO debounce, the only agent in the fleet able to delete an
    // instance on a single frame. The boundary is preserved EXACTLY: RemoveLogodds = -3.0 is the same test
    // as should_remove(p) with p = 1/(1+e^3) = 0.04743, so nothing about WHEN a chair is condemned changes.
    // What changes is that the decision must now be sustained, in the fleet's unit (IDEAL OBSERVATIONS,
    // Sum p_vis) rather than cycles, and that a condemned-but-unexecutable instance reports itself.
    //
    // ★MEASURED FIRST, on chair's own etc/chair_existence_log.csv, because a debounce chosen blind is how
    // this family of bugs starts. Two facts decided it:
    //   · on the cycles the chair was NOT detected, zed_pd (its p_vis) has median 0.000 and p90 0.000 —
    //     the SENSOR channel is very nearly inert for removal (chair_1/chair_2 sat pinned at L=+4.00 for
    //     1554 rows each with zed_pd median 0.000). Counting LOOKS therefore holds those chairs rather
    //     than deleting them, which is correct: a camera that cannot resolve the chair has not seen it go.
    //   · the ROOM PRIOR is the channel that actually fires, and it is visibility-INDEPENDENT by
    //     construction, drawing OutOfRoomGain = 1.5 nats every frame — so +4 to -3 is under FIVE frames.
    //     chair_3 reached L = -2.70 and recovered: it came within 0.30 nats of being deleted by a
    //     localization wobble. That path is where the debounce has to bite, and it does: a prior passes
    //     p_vis = 1.0, so RemoveFrames = 15 means 15 frames of sustained "outside the room".
    rc::exist::RemovalPolicy policy;
    policy.logodds_max   = cfg_.exist_max_logodds;
    policy.removal_prob  = 1.0f / (1.0f + std::exp(-cfg_.exist_remove_logodds));   // exact same boundary
    policy.remove_frames = static_cast<float>(cfg_.exist_remove_frames);

    std::vector<std::uint64_t> to_remove;
    for (auto& [id, inst] : fitter_->instances())
    {
        if (std::isnan(inst.exist_logodds))           // seed on first visit (fresh birth OR adopted graph node)
        {
            inst.existence.set_max(cfg_.exist_max_logodds);
            inst.existence.set(cfg_.exist_birth_logodds);
            inst.exist_logodds = inst.existence.logodds();
        }

        // ROOM-CONTAINMENT POSE PRIOR (runs BEFORE the frustum gate, so it reaches a chair a localization glitch
        // put OUTSIDE the walls / behind a wall where the sensor can never vacate it): P(chair outside) ≈ 0, so
        // an out-of-room centre draws a STRONG negative every frame regardless of visibility → removed in a few.
        if (cfg_.exist_room_prior and fitter_->has_room_polygon() and not inst.is_bearing_hypothesis)
        {
            const auto& ms = inst.model.state();
            if (not fitter_->point_in_room(Eigen::Vector2f(ms.cx, ms.cy), cfg_.exist_room_margin_m))
            {
                // A containment violation is fully resolvable — the polygon is a trusted nominal model —
                // so p_vis = 1 and the whole ratio applies. Through the shared policy like every other
                // channel, so it inherits the clamp and the correlation handling.
                inst.existence.integrate(1.0f, -cfg_.exist_out_of_room_gain);
                inst.exist_logodds = inst.existence.logodds();
                // A containment violation is fully resolvable — the polygon is a trusted nominal model — so
                // this cycle is worth one IDEAL observation and the debounce advances at full weight. Same
                // treatment as door_concept's two prior channels: the difference from a sensor channel is
                // argued, and expressed in the shared unit rather than by skipping the shared decision.
                if (rc::exist::decide_removal(inst.existence, inst.existence_debounce, policy, 1.0f).remove)
                    to_remove.push_back(id);
                continue;   // outside the room → no sensor evidence can rescue it; skip the normal channels
            }
        }

        // HOLD (no evidence) unless the instance is a depth chair in the camera frustum with a real belief.
        // roi_valid is one compute-cycle stale (fine for a frustum test); a bearing-only hypothesis carries no
        // depth (existence unjudgeable from support mass); an un-initialised newborn hasn't had a chance yet.
        if (inst.is_bearing_hypothesis or not inst.roi_valid or not inst.ai2_initialized)
            continue;

        // TWO evidence channels for whether a chair really occupies this in-frustum spot. Each sets the
        // pair the shared policy wants: p_vis (could this look have resolved it?) and the log-ratio.
        float llr;                 // the p_vis-weighted product, kept for the CSV/diagnostics
        float p_vis  = 1.0f;       // P(this probe could have resolved the chair | it exists)
        float ratio  = 0.0f;       // log[ P(outcome|exists) / P(outcome|¬exists) ], unweighted
        const bool won = inst.assigned_mask_idx >= 0 and inst.assigned_mask_idx < static_cast<int>(pkt.slices.size());
        if (won)
        {
            // WON a mask → how much of the EXPECTED chair silhouette does the model actually EXPLAIN?
            //   explanation = (support / expected_at_range) × (1 − clutter_frac).
            // A real chair explains a good fraction → POSITIVE. Two failure modes both explain ≈nothing → NEGATIVE:
            // a far-too-SPARSE won mask (chair_3's ~8 points), OR a big but ~ALL-CLUTTER blob (chair_3/chair_5's
            // 5994/1126 pts at clutter≈0.99 — support present, but the chair model fits none of it). Folding
            // clutter into the SIGN is the fix for the "won a garbage blob → scored neutral → never removed" gap.
            const auto& sl = pkt.slices[inst.assigned_mask_idx];
            const int npts = (sl.support_end > sl.support_begin) ? static_cast<int>(sl.support_end - sl.support_begin) : 0;
            const float range    = std::max(0.5f, inst.last_range);
            const float expected = exist_support_scale_ / (range * range);
            const float adequacy = std::clamp(static_cast<float>(npts) / std::max(1.0f, expected), 0.0f, cap);
            const float explained = adequacy * std::clamp(1.0f - inst.last_clutter_frac, 0.0f, 1.0f);
            llr = (explained >= ar) ? g * (explained - ar) / std::max(1e-3f, cap - ar)
                                    : -g * (ar - explained) / std::max(1e-3f, ar);
            // Be-still invariant, CONTINUOUS: an unreliable frame (moving AND off-axis → smeared/high-clutter) only
            // CONFIRMS — scale its NEGATIVE evidence by the frame reliability ∈ [0,1] so it can hold/raise existence
            // (positive kept full) but not argue the chair away. Reliability→1 for a still or well-centred frame.
            // ★RICOH CONFIRMS, NEVER REMOVES: a won ricoh slice (depth_var>0, or bearing-only) has unreliable
            // depth/clutter → it may CONFIRM existence (its win already reset frames_since_detection) but must not
            // produce NEGATIVE evidence. Only a ZED win (depth_var==0) may argue a chair down.
            const bool won_zed = sl.has_depth and sl.depth_var == 0.0f;
            ratio = llr;                               // the unweighted evidence this mask carries
            if (llr < 0.0f)
            {
                // A refuting look is only worth what the frame could resolve — and a ricoh win may never
                // refute at all (unreliable depth/clutter), which is p_vis = 0, i.e. a HOLD.
                p_vis = won_zed ? fitter_->frame_reliability(inst) : 0.0f;
                llr   = ratio * p_vis;
            }
        }
        else
        {
            // WON NOTHING while in the frustum, UNOCCLUDED, on a ZED-active frame → absence evidence, but weighted
            // TWO ways: (1) CONFIDENCE ramps with frames_since_detection (freshness-as-precision, anti-death-spiral
            // — a chair that just lost the slice or is briefly hidden barely moves and recovers on its next win);
            // (2) ZED EXPECTED-DETECTABILITY pd ∈ [0,1] — absence only removes to the degree ZED would RELIABLY
            // have detected a present chair (falls off toward the image edge + with range). A far/peripheral chair
            // that only ricoh can see has pd≈0 → it HOLDs (maintained by ricoh confirmations) and is removed only
            // once a clean, close, centred ZED look comes up empty. This is "ZED removes".
            const float conf = (cfg_.exist_vacate_confident_frames > 0)
                ? std::clamp(static_cast<float>(inst.frames_since_detection)
                             / static_cast<float>(cfg_.exist_vacate_confident_frames), 0.0f, 1.0f)
                : 0.0f;
            // pd FLOOR (clear line of sight): occlusion is already HELD above, so here the LoS is clear. Even a
            // peripheral chair (low pd) then vacates at ≥ the floor rate, so a glitch-stranded phantom the robot
            // never centres still dies over time; conf gates on staleness so a recently-seen chair is untouched.
            const float pd = std::max(fitter_->zed_detectability(inst), cfg_.exist_zed_clear_los_floor);
            // ★OCCLUSION SCALES ABSENCE, IT NO LONGER SKIPS THE CYCLE. This used to `continue` on a boolean
            // los_occluded(), which is a HOLD with no way out: a phantom chair born inside the dining set is
            // permanently "occluded" by the very furniture it overlaps, so its absence was never charged and
            // L froze short of the removal floor. Measured live: chair_3 pinned at L = -1.94958 for 300+
            // cycles with won=0, since_det=1814 and occluded=1 — while zed_pd said 0.53, i.e. the two
            // visibility models flatly disagreed and the pessimistic one won by being a gate.
            //
            // Genuine occlusion still protects a real chair: strength → 1 leaves ~no absence evidence, which
            // is the behaviour the old branch was after. What changes is that a PARTIALLY or SPURIOUSLY
            // occluded phantom keeps vacating, slowly, instead of living for ever.
            const float occ = cfg_.exist_occlusion_check ? fitter_->los_occlusion(inst) : 0.0f;
            // p_vis is precisely "how well could this look have resolved a chair that IS here":
            // staleness confidence x ZED detectability x how much line of sight is left.
            p_vis = std::clamp(conf * pd * std::clamp(1.0f - occ, 0.0f, 1.0f), 0.0f, 1.0f);
            ratio = -g;                                // absence, at full strength, before visibility
            llr   = ratio * p_vis;
        }

        // ★THROUGH THE SHARED POLICY. `llr` above is already the p_vis-weighted product (each branch
        // scales by conf / zed_pd / (1-occlusion) / frame_reliability), so p_vis is handed in separately
        // and the ratio recovered — the class then owns the clamp, the frame-correlation decorrelation
        // and the removal boundary, identically to refrigerator/table/cabinet/door.
        inst.existence.integrate(p_vis, ratio);
        inst.exist_logodds = inst.existence.logodds();

        // p_vis is EXACTLY what one look was worth (staleness confidence x ZED detectability x remaining
        // line of sight), which is the debounce's unit — so it is passed straight through. A cycle that
        // could not have resolved the chair advances nothing and correctly HOLDS.
        const auto verdict = rc::exist::decide_removal(inst.existence, inst.existence_debounce, policy, p_vis);
        if (verdict.remove)
            to_remove.push_back(id);
        if (verdict.stalled)
            std::print("chair_concept: {}\n", rc::exist::stall_note(inst.node_name, inst.existence,
                                                                    inst.existence_debounce, verdict));
    }

    // Throttled existence readout so a "why is this phantom still here?" case is diagnosable from the log.
    static int ex_dbg = 0;
    if (++ex_dbg % 60 == 0)
        for (const auto& [id, inst] : fitter_->instances())
            if (not inst.is_bearing_hypothesis)
            {
                const auto& ms = inst.model.state();
                const bool has_poly = fitter_->has_room_polygon();
                const bool inroom = (not has_poly)
                                    or fitter_->point_in_room(Eigen::Vector2f(ms.cx, ms.cy), cfg_.exist_room_margin_m);
                const float occluded = (inst.roi_valid and inst.assigned_mask_idx < 0) ? fitter_->los_occlusion(inst) : 0.0f;
                const float zed_pd = fitter_->zed_detectability(inst);   // ZED expected-detectability that gates vacate
                std::print("chair_concept: [existence] {} L={:.2f} pos=({:.2f},{:.2f}) inroom={} roomprior={} roi={} won={} since_det={} occluded={} zed_pd={:.2f}\n",
                           inst.node_name, inst.exist_logodds, ms.cx, ms.cy, inroom ? 1 : 0,
                           has_poly ? 1 : 0, inst.roi_valid ? 1 : 0,
                           inst.assigned_mask_idx >= 0 ? 1 : 0, inst.frames_since_detection, occluded, zed_pd);
                // Same diagnostic to a CSV (you read those) — roomprior=0 ⇒ polygon NOT loaded; inroom=0 ⇒ outside walls;
                // zed_pd=0 ⇒ ZED can't reliably see it (far/edge) so absence won't remove it (ricoh-only-visible).
                static std::ofstream ex_csv = []{ std::ofstream f("etc/chair_existence_log.csv", std::ios::trunc);
                    f << "cycle,node,L,cx,cy,inroom,roomprior_loaded,roi,won,since_det,occluded,zed_pd\n"; return f; }();
                if (ex_csv)
                {
                    ex_csv << ex_dbg << ',' << inst.node_name << ',' << inst.exist_logodds << ',' << ms.cx << ',' << ms.cy
                           << ',' << (inroom ? 1 : 0) << ',' << (has_poly ? 1 : 0) << ',' << (inst.roi_valid ? 1 : 0)
                           << ',' << (inst.assigned_mask_idx >= 0 ? 1 : 0) << ',' << inst.frames_since_detection
                           << ',' << occluded << ',' << zed_pd << '\n';
                    ex_csv.flush();
                }
            }

    for (const std::uint64_t id : to_remove)
    {
        const auto it = fitter_->instances().find(id);
        const float L = (it != fitter_->instances().end()) ? it->second.exist_logodds : 0.0f;
        std::print("chair_concept: [existence] REMOVE id={} (log-odds {:.2f} < {:.2f}; unexplained/vacated in view)\n",
                   id, L, cfg_.exist_remove_logodds);
        if (it != fitter_->instances().end())
        {
            log_tracker_event("REMOVE", id, it->second.model.state().cx, it->second.model.state().cy,
                              std::format("logodds {:.2f}", L));
            // Shadow-mode death record (§4.2), taken BEFORE teardown while the state that justified the kill
            // is still readable. A low p_detect here means this was OUR removal bug, not a YOLO phantom.
            log_phantom_event("DEATH", id, it->second.node_name,
                              it->second.model.state().cx, it->second.model.state().cy, &it->second,
                              std::format("L {:.2f}", L));
            it->second.affordance.remove();
        }
        fitter_->forget_node(id);
        G->delete_node(id);
    }
}

///////////////////////////////////////////////////////////////
void SpecificWorker::process_chair_node(const DSR::Node& node)
{
    const bool created = fitter_->ensure_instance(node, room_node_id_);
    auto& inst = fitter_->instances().at(node.id());

    if (created)
    {
        // Register per-instance time-series (Qt dashboards stay in the worker).
        if (ts_plot_)
        {
            ts_plot_->add_series(inst.node_name + "_fe",  QColor(255, 170,   0), 1.1f);
            if (ts_cov_plot_) ts_cov_plot_->add_series(inst.node_name + "_cov", QColor(  0, 190, 255), 1.1f);
            if (ts_res_plot_) ts_res_plot_->add_series(inst.node_name + "_res", QColor(170,  80, 255), 1.1f);
        }
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

    // Refresh the level-2 arrangement prior BEFORE inference: run_inference both fuses it as a GN
    // factor (accumulate_extra) and consults it when picking the discrete yaw mode.
    refresh_rig_yaw_prior(inst, node);

    const auto observation = fitter_->observe(inst, node);

    // Stale check: skip heavy update if data hasn't moved for too long
    if (not observation.has_fresh_data and inst.matched_frames < 5)
        return;

    const float free_energy = fitter_->run_inference(inst, observation);
    publish_chair_cycle(inst, node, observation, free_energy);
}


// ─── Level-2 arrangement prior (ring_metaconcept → this chair) ───────────────────────────────────
//
// Read the incoming non-RT `group_member` edge and hand its message to the belief. The rig is the
// SOLE writer of that edge; this agent only reads it. (The message deliberately does not live on our
// own node: update_node resets attributes absent from the submitted copy, so our own writes would
// intermittently delete a prior parked there — see the note in cortex's dsr_attr_name.h.)
//
// Three ways the prior goes inert, all of which reduce the belief to its pre-rig behaviour exactly:
// no edge, κ ≤ 0 (the rig's own "ignore me" signal, written when a member leaves a set), or a stale
// message (the rig died without cleaning up — a prior nobody is maintaining must not steer a chair).
void SpecificWorker::refresh_rig_yaw_prior(rc::ChairInstance& inst, const DSR::Node& node)
{
    inst.rig_edge_found = false;
    inst.rig_kappa      = 0.0f;
    if (not cfg_.rig_yaw_prior_enabled)
        return;
    inst.ai2_belief.clear_rig_yaw_prior();

    for (const auto& e : G->get_edges_to_id(node.id()))
    {
        if (e.type() != "group_member")
            continue;

        const auto kappa = G->get_attrib_by_name<rig_yaw_kappa_att>(e);
        const auto yaw   = G->get_attrib_by_name<rig_yaw_prior_att>(e);
        if (not kappa.has_value() or not yaw.has_value() or kappa.value() <= 0.0f)
            continue;

        // Staleness: a rig that stopped publishing must stop steering. 0 disables the check.
        if (cfg_.rig_prior_stale_ms > 0)
            if (const auto stamp = G->get_attrib_by_name<rig_stamp_ms_att>(e); stamp.has_value())
            {
                const auto now = static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count());
                if (now > stamp.value() and (now - stamp.value()) > static_cast<std::uint64_t>(cfg_.rig_prior_stale_ms))
                    continue;
            }

        // The consumer-side cap is a second bound on top of the belief's own structural one
        // (ChairBelief::kRigModeShare·kFlipClamp): config can only ever make the prior WEAKER.
        const float k = std::min(kappa.value(), cfg_.rig_yaw_kappa_max);
        inst.ai2_belief.set_rig_yaw_prior(yaw.value(), k);
        inst.rig_edge_found = true;
        inst.rig_kappa      = k;
        inst.rig_prior_yaw  = yaw.value();
        return;   // one rig per member
    }
}

void SpecificWorker::publish_chair_cycle(rc::ChairInstance& inst,
                                         const DSR::Node& node,
                                         const ChairObservation& observation,
                                         float free_energy)
{
    const auto node_id = node.id();
    if (not scene_graph_->persist_chair_belief(inst, node_id, room_node_id_, free_energy))
        return;
    if (not assess_chair_state(inst, node_id, free_energy))
        return;
    publish_chair_diagnostics(inst, observation, free_energy);
    publish_chair_intentions(inst, node_id, observation, free_energy);
}

bool SpecificWorker::assess_chair_state(rc::ChairInstance& inst, uint64_t node_id, float free_energy)
{
    auto node_opt = G->get_node(node_id);
    if (not node_opt.has_value())
        return false;

    step_convergence(inst, node_opt.value(), free_energy);
    return true;
}

void SpecificWorker::publish_chair_diagnostics(const rc::ChairInstance& inst,
                                               const ChairObservation& observation,
                                               float free_energy)
{

    // Register lazily & idempotently HERE (same thread/object that adds points). The instance is
    // often created via the graph-signal path before the plots exist (created=false in the compute
    // loop), so the old `if (created)` registration never fired and every point was dropped.
    if (ts_plot_)
    {
        ts_plot_->add_series(inst.node_name + "_fe", QColor(255, 170, 0), 1.1f);
        ts_plot_->add_point (inst.node_name + "_fe", free_energy);
        // FE BASELINE on the SAME panel (same units): the FE lifting ABOVE the grey baseline IS the
        // surprise, shown visually. The smoothed gap goes on its own panel (much smaller scale).
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
            ts_cov_plot_->add_point (inst.node_name + "_cov", belief_uncertainty(inst));
        }
        if (ts_res_plot_)
        {
            ts_res_plot_->add_series(inst.node_name + "_res", QColor(170, 80, 255), 1.1f);
            // Residual points only exist on FRESH-mask frames; plotting 0 on every idle cycle made the
            // series crash thousands→0 between masks. Only sample on fresh frames so the line holds
            // the last real value between detections (a meaningful per-mask residual-count trend).
            if (observation.has_fresh_data)
                ts_res_plot_->add_point (inst.node_name + "_res", static_cast<float>(observation.residual_pts.size()));
        }
        // (The inferred-dimensions trace that used to live here is gone: the BeliefInspector below shows
        // every DOF's value AND its σ live, so a separate trace was showing the same thing twice.)
        // (The pose-σ trace that used to live here is gone: the BeliefInspector panel now shows σ for EVERY
        // DOF, next to the whole correlation structure and the yaw-mode posterior.)
    }

    if (fitter_->should_log(inst))
        std::print("[{}] series: FE={:.4f} U(Σ)={:.3f} res={}\n",
                   inst.node_name,
                   free_energy,
                   belief_uncertainty(inst),
                   observation.residual_pts.size());
}

void SpecificWorker::publish_chair_intentions(rc::ChairInstance& inst,
                                              uint64_t node_id,
                                              const ChairObservation& observation,
                                              float free_energy)
{
    // step_epistemic now owns the propose-vs-withdraw decision via the planner's ΔH (it withdraws
    // when the expected information gain falls below threshold), so it runs unconditionally rather
    // than being gated on the legacy coverage-deficit proxy.
    if (auto node_opt = G->get_node(node_id); node_opt.has_value())
        step_epistemic(inst, node_opt.value());
}

// ─── Initialisation helpers ──────────────────────────────────────────────────

void SpecificWorker::load_config(const ConfigLoader& cfg)
{
    cfg_ = rc::load_chair_config(cfg);
}


// ─── Per-cycle steps ─────────────────────────────────────────────────────────


void SpecificWorker::step_convergence(rc::ChairInstance& inst,
                                       DSR::Node& node,
                                       float free_energy)
{
    // Convergence on STATE stability, not |ΔFE|: the free energy keeps jittering with queue
    // churn / point-count even when the fitted geometry is settled, so it never latched. Track
    // how much the accepted state moved between cycles instead.
    const auto& s = inst.model.state();
    const auto& p = inst.prev_conv_state;
    const float state_delta = inst.has_prev_conv_state
        ? (std::abs(s.cx - p.cx) + std::abs(s.cy - p.cy) + std::abs(s.seat_w - p.seat_w) + std::abs(s.seat_d - p.seat_d) +
           std::abs(s.seat_h - p.seat_h) + std::abs(s.back_h - p.back_h) + std::abs(s.yaw - p.yaw))
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
    // position + size (m), from the AI2 covariance Σ. Shrinks as the robot gathers viewpoints — the
    // AI2-native replacement for the old queue face-coverage deficit.
    const float model_uncertainty = belief_uncertainty(inst);
    G->add_or_modify_attrib_local<model_uncertainty_att>(node, model_uncertainty);

    if (fitter_->should_log(inst))
    {
        const auto& cs = inst.model.state();
        std::print("[{}] convergence: pos=({:.2f},{:.2f}) yaw={:.2f} Δstate={:.4f} stable={}/{} U(Σ)={:.3f}m\n",
                   inst.node_name, cs.cx, cs.cy, cs.yaw, state_delta, inst.frames_converged, cfg_.K_stable,
                   model_uncertainty);
    }

    if (inst.frames_converged >= cfg_.K_stable)
    {
        if (not inst.model_stable)
        {
            inst.model_stable = true;
            G->add_or_modify_attrib_local<model_stable_att>(node, true);
            G->update_node(node);
            std::print("chair_concept: node '{}' STABLE (F={:.4f})\n",
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

void SpecificWorker::step_epistemic(rc::ChairInstance& inst, DSR::Node& node)
{
    if (inst.epistemic_cooldown > 0)
        --inst.epistemic_cooldown;

    // Controller-completion hold (anti-churn): the chair affordance completes on a weak detection
    // (contract goal conf≥0.20), which fires almost instantly — before ΔH has decayed. Start a short
    // cooldown so we don't re-offer a just-completed chair while its gain is still high. We do NOT
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
                                   rc::nbv::sensor_from_graph(*G, inner_eigen_.get()),
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
    // in, no node churn. During the post-completion hold the gain is forced to 0 so a just-finished chair
    // isn't re-claimed before its belief has settled.
    if (inst.epistemic_cooldown > 0)
        prop.epistemic_gain = 0.0f;

    // Bearing-only hypothesis (Part C-birth): author an ORIENT affordance whose target yaw IS the bearing,
    // so the controller rotates to look down the ray (the broad along-ray Σ already makes prop's gain high).
    const bool orient = inst.is_bearing_hypothesis;
    if (orient)
        prop.epistemic_target_yaw_rad = inst.hypothesis_azimuth;

    // Write attributes to the chair node (read by legacy consumers)
    scene_graph_->write_epistemic_proposal(node, prop);
    // Publish / refresh dedicated affordance node (persists; update_node refreshes target+gain)
    const auto affordance_node_before = inst.affordance.node_id();
    inst.dbg_nbv_gain = prop.epistemic_gain;   // the value actually published
    // Planner internals stay in EpistemicProposal; the producer takes the shared eleven-field view.
    rc::AffordanceTarget tgt;
    tgt.x_m     = prop.epistemic_target_x_m;
    tgt.y_m     = prop.epistemic_target_y_m;
    tgt.yaw_rad = prop.epistemic_target_yaw_rad;
    tgt.gain    = prop.epistemic_gain;
    tgt.valid   = prop.valid;
    inst.affordance.update(tgt, orient);
    if (affordance_node_before == 0 and inst.affordance.node_id() != 0)
        trigger_graph_layout_twopi();
    inst.epistemic_pending = true;
}

// ─── DSR helpers ─────────────────────────────────────────────────────────────


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
// ─── DSR signal slots ────────────────────────────────────────────────────────

void SpecificWorker::modify_node_slot(std::uint64_t id, const std::string& type)
{
    // Chairs are generic `object` nodes named "chair_*" (schema migration).
    if (type != "object")
        return;

    const auto node_opt = G->get_node(id);
    if (not node_opt.has_value())
        return;
    if (not node_opt.value().name().starts_with("chair"))
        return;

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
    if (not fitter_)
        return;   // may be called before the fit core exists — the guard every graph slot here needs

    for (auto& [chair_id, inst] : fitter_->instances())
    {
        // Affordance state machine: idle→pending→executing→satisfied, driven by the controller-owned
        // active/pending flags on the affordance node. on_node_modified() re-reads them itself.
        if (const auto aid = inst.affordance.node_id(); aid != 0)
            inst.affordance.on_node_modified(aid);

        // Mission controller clearing epistemic_pending on the object node itself.
        if (auto node_opt = G->get_node(chair_id); node_opt.has_value())
        {
            const auto v = G->get_attrib_by_name<epistemic_pending_att>(node_opt.value());
            if (v.has_value() and not v.value())
                inst.epistemic_pending = false;
        }
    }
}
void SpecificWorker::del_node_slot(std::uint64_t id)
{
    // Notify affordance in case its own DSR node was deleted externally
    for (auto& [chair_id, inst] : fitter_->instances())
        if (inst.affordance.node_id() == id)
            inst.affordance.on_node_deleted(id);

    if (fitter_->instances().count(id))
    {
        std::print("chair_concept: node {} removed from DSR, destroying instance\n", id);
        fitter_->instances().erase(id);
    }
}

// ─── Lifecycle stubs ─────────────────────────────────────────────────────────

void SpecificWorker::emergency()
{
    std::print("chair_concept: emergency()\n");
}

void SpecificWorker::restore()
{
    std::print("chair_concept: restore()\n");
}

int SpecificWorker::startup_check()
{
    std::print("chair_concept: startup_check()\n");
    return 0;
}




