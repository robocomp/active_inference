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
#include "door_dof.h"   // rc::kDoorDofs — names/units for the BeliefInspector rows

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

// Two doors cannot share physical space. Footprint = the oriented panel rectangle (cx,cy,w,thickness,yaw)
// in the room plane; these helpers compute the overlap area so the merge operator can collapse duplicate
// instances. Corners CCW (local order (-,-),(+,-),(+,+),(-,+)). Mirrors table_concept.
std::array<Eigen::Vector2f, 4> footprint_corners(const rc::DoorState& s)
{
    const float c = std::cos(s.yaw), sn = std::sin(s.yaw);
    const Eigen::Vector2f ex(c, sn), ey(-sn, c), ctr(s.cx, s.cy);
    const float hw = 0.5f * s.w, hh = 0.5f * s.thickness;
    return { ctr - hw * ex - hh * ey, ctr + hw * ex - hh * ey,
             ctr + hw * ex + hh * ey, ctr - hw * ex + hh * ey };
}

float poly_area(const std::vector<Eigen::Vector2f>& p)
{
    if (p.size() < 3) return 0.0f;
    float a = 0.0f;
    for (std::size_t i = 0, n = p.size(); i < n; ++i)
    {
        const auto& u = p[i]; const auto& v = p[(i + 1) % n];
        a += u.x() * v.y() - v.x() * u.y();
    }
    return 0.5f * std::abs(a);
}

// Sutherland–Hodgman: clip the subject polygon against the convex CCW clip rectangle.
std::vector<Eigen::Vector2f> clip_poly(std::vector<Eigen::Vector2f> subj,
                                       const std::array<Eigen::Vector2f, 4>& clip)
{
    for (int e = 0; e < 4 and not subj.empty(); ++e)
    {
        const Eigen::Vector2f a = clip[e], b = clip[(e + 1) % 4], d1 = b - a;
        const auto inside = [&](const Eigen::Vector2f& p)
        { return d1.x() * (p.y() - a.y()) - d1.y() * (p.x() - a.x()) >= 0.0f; };
        std::vector<Eigen::Vector2f> out;
        for (std::size_t i = 0, n = subj.size(); i < n; ++i)
        {
            const Eigen::Vector2f cur = subj[i], prv = subj[(i + n - 1) % n];
            const bool ci = inside(cur), pi = inside(prv);
            const auto isect = [&]() -> Eigen::Vector2f
            {
                const Eigen::Vector2f d2 = cur - prv;
                const float den = d2.x() * d1.y() - d2.y() * d1.x();
                const float t = std::abs(den) < 1e-12f ? 0.0f
                    : ((a.x() - prv.x()) * d1.y() - (a.y() - prv.y()) * d1.x()) / den;
                return prv + t * d2;
            };
            if (ci) { if (not pi) out.push_back(isect()); out.push_back(cur); }
            else if (pi) out.push_back(isect());
        }
        subj.swap(out);
    }
    return subj;
}

// Overlap area as a fraction of the SMALLER footprint (1.0 = one seat fully inside the other).
float footprint_overlap_ratio(const rc::DoorState& a, const rc::DoorState& b)
{
    const auto ca = footprint_corners(a), cb = footprint_corners(b);
    const auto inter = clip_poly(std::vector<Eigen::Vector2f>(ca.begin(), ca.end()), cb);
    const float ai = poly_area(inter);
    const float amin = std::min(poly_area({ca.begin(), ca.end()}), poly_area({cb.begin(), cb.end()}));
    return amin > 1e-6f ? ai / amin : 0.0f;
}

// Tracker lifecycle event log (etc/door_events.csv) — makes birth/merge/prune/suppress visible so the
// "create then remove" churn can be diagnosed from a file, not stdout. seq gives ordering.
void log_tracker_event(const char* ev, std::uint64_t id, float x, float y, const std::string& note)
{
    static std::ofstream f = [] { std::ofstream o("etc/door_events.csv", std::ios::trunc);
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

    rc::DoorBelief::self_test();   // startup sanity check of the wall-frame [s,w,h] belief (mirrors bottle_concept)

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
    std::print("door_concept: SpecificWorker destroyed.\n");
}

void SpecificWorker::request_shutdown()
{
    if (shutting_down_.exchange(true))
        return;

    save_window_settings();
    save_dashboard_geometry();   // the standalone dashboard is not in `windows`, so save it explicitly

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

// ─── Initialisation ──────────────────────────────────────────────────────────

void SpecificWorker::initialize()
{
    std::print("door_concept: initialize()\n");
    GenericWorker::initialize();
 
    // Ignore payload attributes in local graph updates to avoid unnecessary copying and processing of potentially large data
    G->set_ignored_attributes<cam_rgb_att, cam_depth_att, laser_X_att, laser_Y_att, laser_Z_att>();
    qInfo() << "Ignoring DSR RGBD payload attributes cam_rgb/cam_depth in local graph updates";


    if (not G)
    {
        qWarning() << "door_concept: DSR graph not available in initialize()";
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
            // Pump the masks ingest WHILE Waiting: door polls a graph node (no free-running ingest thread
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
    scene_graph_ = std::make_unique<rc::DoorSceneGraph>(
        G, rt_api_.get(), cfg_, [this] { trigger_graph_layout_twopi(); });

    // Remove any "door*" nodes left behind by a previous (crashed) run so this agent always starts
    // from a clean slate and never adopts a stale/drifted node (the instance tracker re-births them
    // data-driven from masks). Runs BEFORE the graph signals are connected: delete_node() fires
    // del_node_signal synchronously, and del_node_slot dereferences fitter_ — which does not exist yet.
    // Connecting after this sweep (and after fitter_ is built) both avoids that null-deref SIGSEGV and
    // skips a pointless self-notification for our own cleanup deletions.
    remove_owned_door_nodes();

    // Resolve room node
    const auto rooms = G->get_nodes_by_type("room");
    if (not rooms.empty())
        room_node_id_ = rooms.front().id();
    else
        qWarning() << "door_concept: no room node found at startup";

    // Active-inference fit core. Owns the instance map; collaborates with the ingestor + scene graph.
    fitter_ = std::make_unique<rc::DoorFitter>(
        G, inner_eigen_.get(), cfg_, mask_ingestor_.get(), scene_graph_.get());
    fitter_->set_central_region_frac(cfg_.exist_central_region_frac);   // existence: "is the robot looking AT it"

    // Subscribe to graph signals ONLY now that fitter_ (which every slot dereferences) exists, and after
    // the startup stale-sweep above. Cross-thread Auto resolves to Queued (never add DirectConnection).
    connect(G.get(), &DSR::DSRGraph::update_node_signal,
            this, &SpecificWorker::modify_node_slot);
    connect(G.get(), &DSR::DSRGraph::update_node_attr_signal,
            this, &SpecificWorker::modify_node_attrs_slot);
    connect(G.get(), &DSR::DSRGraph::del_node_signal,
            this, &SpecificWorker::del_node_slot);

    // Part B: localization/chain covariance on the published RT edge (mirrors bottle/table).
    gaussian_api_ = std::make_unique<DSR::InnerGaussianAPI>(G.get());
    fitter_->set_chain_cov_source(gaussian_api_.get(), "zed", cfg_.rt_cov_add_chain);

    // Build rc::EpistemicPlanner (Σ-based D-optimal NBV) with the configured stand-off.
    epistemic_planner_ = rc::EpistemicPlanner(cfg_.obs_distance);
    epistemic_planner_.set_min_standoff(cfg_.min_standoff_m);   // push viewpoints further out (YOLO needs a gap)

    // Stale affordance nodes are swept on entering Operating (presence hook) and on shutdown — see
    // remove_stale_affordance_nodes(), keyed on the parent object type (robust to node-name renames).

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

    restore_dashboard_geometry();
    dashboard_window_->show();
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
        const std::array<float, rc::DoorBelief::N> v = {s.s, s.w, s.h};
        for (int j = 0; j < rc::DoorBelief::N; ++j)
            c.dofs.push_back({rc::kDoorDofs[j].name, rc::kDoorDofs[j].unit, v[j],
                              std::sqrt(std::max(0.0f, S(j, j))), rc::kDoorDofs[j].sigma_star});

        // Row-major copy, filled explicitly: Eigen stores column-major, and while Σ is symmetric today an
        // implicit .data() copy would silently transpose if that ever stopped being true.
        constexpr int N = rc::DoorBelief::N;
        c.cov.resize(N * N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                c.cov[i * N + j] = S(i, j);

        // No discrete orientation ambiguity: the containing wall fixes the door's yaw (c.modes stays empty).

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
        c.s.remove_streak = inst.existence_remove_streak;
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
    if (not G or not rt_api_)
        return;

    // Refresh room node id if not yet found
    if (room_node_id_ == 0)
    {
        const auto rooms = G->get_nodes_by_type("room");
        if (rooms.empty()) return;
        room_node_id_ = rooms.front().id();
    }

    // Evidence-pipeline per-cycle counters (the *_cum fields persist). Producers below add to these; the
    // snapshot is pushed at the end of the cycle.
    ev_g_.births = ev_g_.merges = ev_g_.removals = 0;

    refresh_room_geometry();  // room-containment pose prior (cheap; the polygon is a nominal model)
    fitter_->update_ego_motion();   // robot/camera speed → "be-still-to-update" gate (once per cycle)
    mask_ingestor_->refresh();
    run_instance_tracker();   // data-driven birth/associate/death + merge (the only instance-lifecycle path)

    // Doors are generic `object` nodes named "door_*" (schema migration); filter by name prefix.
    const auto door_nodes = G->get_nodes_by_type("object");
    for (const auto& node : door_nodes)
        if (node.name().starts_with("door"))
            process_door_node(node);

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

// Collapse instances whose seat footprints overlap (same physical door fitted twice): keep the one with
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

            const float ratio = footprint_overlap_ratio(ia->second.model.state(), ib->second.model.state());
            if (ratio < cfg_.tracker_merge_overlap) continue;

            const bool keep_i = ia->second.matched_frames >= ib->second.matched_frames;
            const std::uint64_t keep = keep_i ? ids[i] : ids[j];
            const std::uint64_t drop = keep_i ? ids[j] : ids[i];
            std::print("door_concept: [tracker] MERGE id={} into id={} (footprint overlap {:.2f})\n",
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

// Data-driven multi-instance lifecycle (mirrors table_concept). Doors are persistent furniture, so
// death is OFF by default (removed only by MERGE); birth_min_sep is wide. The only instance-lifecycle path.
void SpecificWorker::run_instance_tracker()
{
    merge_overlapping_instances();   // enforce physical exclusion before associating/birthing this cycle

    rc::TrackerParams tp;
    tp.gate_mahalanobis = cfg_.tracker_gate_mahalanobis;
    tp.gate_fallback_m  = cfg_.tracker_gate_fallback_m;
    tp.detection_noise_m = cfg_.tracker_detection_noise_m;
    tp.birth_frames     = cfg_.tracker_birth_frames;
    tp.death_frames     = cfg_.tracker_death_frames;
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
        // door projects into the camera FoV. Out-of-view → HOLD (matches the prune gate above). roi_valid
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

    // Detections ← this frame's "door" mask slices (carry the slice index for the assignment).
    std::vector<rc::DetectionView> dets;
    const auto& pkt = mask_ingestor_->packet();
    if (pkt.valid)
        for (int i = 0; i < static_cast<int>(pkt.slices.size()); ++i)
        {
            const auto& sl = pkt.slices[i];
            if (sl.label != "door" or sl.support_end <= sl.support_begin) continue;
            rc::DetectionView dv;
            dv.xy = Eigen::Vector2f(sl.centroid.x(), sl.centroid.y());
            dv.slice_index = i;
            // ZED-only BIRTH: only a ZED slice (per-pixel depth ⇒ depth_var==0) may SPAWN a door. A ricoh
            // LiDAR-reprojected-depth slice (depth_var>0) has unreliable depth/extent, so it may ASSOCIATE to /
            // confirm an existing door but must NOT birth a phantom — this is exactly what created door_3 (a
            // wrong ricoh detection at 8.8 m, outside the room). A confident-ricoh escape hatch is OFF by default.
            const bool is_zed = sl.depth_var == 0.0f;
            dv.birthable = is_zed or (cfg_.ricoh_birth_enabled
                                      and sl.confidence >= cfg_.ricoh_birth_conf
                                      and sl.depth_var  <= cfg_.ricoh_birth_max_var);
            dets.push_back(dv);
        }

    // DIAGNOSTIC (merged-vs-single mask): one CSV row per "door" slice per cycle — count, size, centroid,
    // range. If a cluttered scene collapses to ONE big slice (npts ≫ a clean single door) the extra doors
    // never get born because no separate detection ever arrives — the failure is upstream, not in birth. File
    // truncated once per process launch.
    {
        static std::ofstream dcsv = []
        {
            std::ofstream f("etc/door_dets_log.csv", std::ios::trunc);
            f << "cycle,n_door_slices,slice_idx,npts,conf,cx,cy,range,trunc_frac,motion_var\n";
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
        std::print("[tracker] instances={} door_dets={} assigned={} unassigned={} births={} deaths={}\n",
                   tracks.size(), dets.size(), n_assigned,
                   static_cast<int>(dets.size()) - n_assigned, res.births.size(), res.deaths.size());

    // DEATH: OFF by default — a door is persistent furniture; long occlusion is not absence. Enable
    // Tracker.DeathEnabled to restore miss-timer retirement.
    if (cfg_.tracker_death_enabled)
        for (const std::uint64_t id : res.deaths)
        {
            std::print("door_concept: [tracker] DEATH id={} (unobserved {} frames)\n", id, cfg_.tracker_death_frames);
            if (auto it = fitter_->instances().find(id); it != fitter_->instances().end())
                it->second.affordance.remove();
            fitter_->forget_node(id);
            G->delete_node(id);
        }

    // ASSOCIATE: route each detection's mask slice to its instance (read in observe()).
    {
        // §3.1 (Fable, PERCEPTION_ASSOCIATION_PLAN.md): associate by MODEL EVIDENCE, not centroid. The
        // teleport was: a merged mask's centroid lands MIDWAY between two doors → falls in the wrong
        // track's Mahalanobis gate → greedy flip. Instead score each (initialised instance × door slice)
        // by the instance's belief NLL on the slice's POINTS — a merged mask fits NO single-door model, and
        // each instance keeps the slice that best matches ITS geometry. Greedy lowest-nll, 1-to-1, no gate.
        // ★The evidence is `association_nll` = the mixture NLL, NOT `mean_energy`: mean_energy weights the SDF
        // by responsibility, so a FAR slice (all points → clutter) scored ~0 = a PERFECT match → a distant
        // instance mis-claimed the 3rd door's slice and SUPPRESSED its birth (door "never seen"). The
        // mixture NLL counts the clutter cost, so a far/clutter'd slice scores HIGH → unclaimed → it births.
        struct EvPair { float e; std::uint64_t id; int slice; };
        std::vector<EvPair> pairs;
        const float R = cfg_.ai2_sigma_base_m * cfg_.ai2_sigma_base_m;
        const float rn2 = cfg_.tracker_detection_noise_m * cfg_.tracker_detection_noise_m;
        for (auto& [id, inst] : fitter_->instances())
        {
            if (not inst.ai2_initialized) continue;
            const Eigen::Vector2f bc = inst.ai2_belief.center_xy();   // room-frame panel centre
            const auto& BS = inst.ai2_belief.covariance();
            // Σ(0,0)=σ_s² is the along-wall position variance; use it isotropically for the position gate.
            const float pos_var = BS(0, 0);
            const float sxx = pos_var + inst.chain_cov_xx + rn2;   // innovation cov diag S = P + R²I
            const float syy = pos_var + inst.chain_cov_yy + rn2;
            for (const auto& d : dets)
            {
                // POSITION GATE (mirrors the tracker's S=P+R²I Mahalanobis): an instance may claim a slice
                // ONLY if the slice centroid is within its gate. Without it the evidence greedy assigns EVERY
                // instance some slice, so an instance whose own door is occluded this frame claims a FAR
                // door's slice (high nll, but the least-bad available) → suppresses THAT door's birth AND
                // teleports the instance (the "3rd door never seen" + wrong-pose bug). Gate by position,
                // rank by shape-evidence = the correct combination.
                const float ex = d.xy.x() - bc.x(), ey = d.xy.y() - bc.y();
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
        // its BIRTH position, so it initialises AT the door it was born from — NOT the nearest door (that
        // was the teleport: a far-born instance grabbed a near door's mask when its own was occluded, then
        // merged, so the far door never persisted). If no slice sits at its birth spot this frame it stays
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
    // clutter (door_3: 8 points at 7.4 m) never reached patience AND aged into permanence. The existence belief
    // fixes both: it integrates SUPPORT-MASS-WEIGHTED evidence (8 pts where ~130 are expected → strong negative)
    // once per sensor frame, with no age immunity — a real door stays only by continuing to be explained.
    if (cfg_.exist_enabled)
        update_existence_beliefs();
    else if (cfg_.tracker_prune_enabled)
    {
        std::vector<std::uint64_t> stillborn;
        for (auto& [id, inst] : fitter_->instances())
        {
            if (inst.assigned_mask_idx >= 0) { inst.unassigned_streak = 0; continue; }
            // Negative-information: an unassigned cycle is evidence of a PHANTOM only when the door SHOULD
            // be seen — its model projects into the camera FoV (roi_valid) yet no mask associated. When the
            // door is out of view (robot looked away) the streak is HELD, so a real door glimpsed once and
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
            std::print("door_concept: [tracker] PRUNE stillborn id={} (unassigned {} cycles, age {} < maturity {})\n",
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
    // fitter with the detection XY so the model starts AT the door (not the 0,0 RT-read default).
    for (const int d : res.births)
    {
        const int slice = dets[d].slice_index;
        // §3.1: under AI2 the assignment is by belief evidence, not the tracker's centroid — so a slice an
        // existing instance already claimed by mean_energy must NOT also spawn a phantom (the tracker's
        // centroid birth and the evidence assignment can disagree). Full birth-validity (P(v=clean) — don't
        // birth from a merged/contaminated mask at all) is §2, still to come.
        const Eigen::Vector3f& c = pkt.slices[slice].centroid;

        // Room-containment pose prior at BIRTH: never spawn a door outside the walls (a mislocalized frame while
        // the robot is lost drops a detection beyond a wall). Zero prior mass outside the room → suppress.
        if (cfg_.exist_room_prior and fitter_->has_room_polygon()
            and not fitter_->point_in_room(Eigen::Vector2f(c.x(), c.y()), cfg_.exist_room_margin_m))
        {
            std::print("door_concept: [tracker] BIRTH SUPPRESSED slice={} at ({:.2f},{:.2f}) — OUTSIDE room\n",
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
                // DIAGNOSTIC (missing-door): a persistently-detected cluster that never instantiates is
                // usually a birth SUPPRESSED here — a distant instance's belief mis-claimed this slice by
                // mean_energy, stealing it from birth (and likely teleporting toward it). Surface which
                // instance claimed it and how far it sits, so the failure is visible, not silent.
                float dist = -1.0f;
                if (auto it = fitter_->instances().find(claimer); it != fitter_->instances().end())
                {
                    const auto& st = it->second.model.state();
                    dist = std::hypot(st.cx - c.x(), st.cy - c.y());
                }
                std::print("door_concept: [tracker] BIRTH SUPPRESSED slice={} at ({:.2f},{:.2f}) — claimed by "
                           "id={} ({:.2f} m away)\n", slice, c.x(), c.y(), claimer, dist);
                log_tracker_event("SUPPRESS", claimer, c.x(), c.y(), std::format("claimer {:.2f}m", dist));
                continue;
            }
        }
        // RE-ACQUISITION: is this detection a door that was removed and has come back at the same place? If so
        // it is not a new object — it takes its old name and resumes the belief it had converged to, instead of
        // becoming door_N+1 with template priors. (Live run: one real door burned three identities this way.)
        const rc::DoorBelief* revive = nullptr;
        std::string preferred_name;
        if (const DoorGhost* g = match_ghost(Eigen::Vector2f(c.x(), c.y())); g != nullptr)
        {
            preferred_name = g->name;
            revive = &g->belief;
            std::print("door_concept: [tracker] RE-ACQUIRE '{}' at ({:.2f},{:.2f}) — {:.2f} m from where it was "
                       "removed after {} cycles; resuming its belief\n",
                       g->name, c.x(), c.y(), (g->xy - Eigen::Vector2f(c.x(), c.y())).norm(), g->lived_cycles);
        }
        const auto new_id = scene_graph_->create_instance_from_detection(c, room_node_id_, preferred_name);
        if (new_id != 0)
        {
            fitter_->note_birth(new_id, Eigen::Vector2f(c.x(), c.y()));
            if (revive != nullptr)
            {
                fitter_->note_reacquire(new_id, *revive);
                log_tracker_event("REACQUIRE", new_id, c.x(), c.y(), preferred_name);
                forget_ghost(preferred_name);   // consumed — the door is alive again
            }
            // Materialise the DoorInstance NOW, at birth — do not wait for the freshly inserted DSR node
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
        }
    }

    // ── Part C-birth: NEW-object hypotheses from unmatched 360 bearings ──────────────────────────────
    // A peripheral "door" bearing (a no-depth 360 slice) that lines up with no live door and PERSISTS
    // births a broad-Σ hypothesis: mean placed on the ray at a nominal range, Σ huge along the ray (range
    // unknown) and tight across it (bearing known). The hypothesis authors an Orient affordance (rotate to
    // look); a depth mask then collapses Σ, or it dies unobserved. Default OFF (Bearing.BirthEnabled).
    if (cfg_.bearing_birth_enabled and pkt.valid)
    {
        std::vector<rc::BearingDetectionView> bearings;
        for (int i = 0; i < static_cast<int>(pkt.slices.size()); ++i)
            if (pkt.slices[i].label == "door" and not pkt.slices[i].has_depth)
                bearings.push_back({pkt.slices[i].azimuth_room_rad, i});

        if (not bearings.empty())
        {
            Eigen::Vector2f robot_xy(0.f, 0.f);
            if (const auto p = inner_eigen_->transform("room", Eigen::Vector3d::Zero(), "zed"); p.has_value())
                robot_xy = {static_cast<float>(p->x()), static_cast<float>(p->y())};

            // A bearing that lines up with a live door is "explained" (confirmation, not new); the rest go
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
                // Anti-dup: skip if a live door already sits near the nominal point on this ray.
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
                std::print("door_concept: [bearing] BIRTH hypothesis id={} az={:.0f}deg (nominal {:.1f}m on ray)\n",
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
    const auto& s = inst.model.state();
    std::erase_if(ghosts_, [&](const DoorGhost& g) { return g.name == inst.node_name; });
    ghosts_.push_back({inst.node_name, Eigen::Vector2f(s.cx, s.cy), inst.ai2_belief, inst.processed_cycles});
    // Bounded, most-recent-first: this is a short-lived flicker memory, not a permanent map.
    while (static_cast<int>(ghosts_.size()) > std::max(1, cfg_.exist_ghost_max))
        ghosts_.erase(ghosts_.begin());
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

void SpecificWorker::forget_ghost(const std::string& name)
{
    std::erase_if(ghosts_, [&](const DoorGhost& g) { return g.name == name; });
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
void SpecificWorker::update_existence_beliefs()
{
    // Integrate at the SENSOR rate, not the compute rate: only when a new mask frame arrived. Otherwise a fast
    // compute loop would decay a briefly-occluded real door away between two sensor frames.
    const auto& pkt = mask_ingestor_->packet();
    const bool sensor_fresh = pkt.valid and static_cast<int>(pkt.frame_id) != exist_last_mask_frame_;
    if (not sensor_fresh)
        return;
    exist_last_mask_frame_ = static_cast<int>(pkt.frame_id);

    // The silhouette channel needs the producer's raw 2D mask pixels (mask_pixels_xy — an OPTIONAL, newer
    // producer channel). Without them EVERY door reports n_detectable==0 and therefore HOLDs forever, which
    // looks exactly like "phantoms are never removed". Say so once rather than failing silently.
    if (pkt.mask_pixels.empty())
    {
        static bool warned = false;
        if (not warned)
        {
            warned = true;
            std::print("door_concept: [existence] WARNING — the masks node carries no mask_pixels_xy; the "
                       "silhouette channel has no evidence, so NO door can be confirmed or removed. Check the "
                       "voxelizer's mask-pixel publish.\n");
        }
        return;
    }

    // Physical sensor rates (interpretable detection/clutter probabilities), shared with table/chair.
    rc::exist::SensorModel sm;
    sm.sensor_sigma_m = cfg_.exist_sensor_sigma_m;
    sm.detection_prob = cfg_.exist_detection_prob;
    sm.clutter_prob   = cfg_.exist_clutter_prob;

    std::vector<std::uint64_t> to_remove;
    for (auto& [id, inst] : fitter_->instances())
    {
        inst.existence.set_max(cfg_.exist_max_logodds);
        if (not inst.existence_seeded)               // seed on first visit (fresh birth OR adopted graph node)
        {
            inst.existence.set(cfg_.exist_birth_logodds);
            inst.existence_seeded = true;
        }

        // ROOM-CONTAINMENT POSE PRIOR (runs BEFORE the sensor channel, so it reaches a door a localization glitch
        // put OUTSIDE the walls / behind a wall where the sensor can never vacate it): P(door outside) ≈ 0, so
        // an out-of-room centre draws a STRONG negative every frame regardless of visibility → removed in a few.
        // This is a genuine PRIOR over pose, not a detectability heuristic, so it survives the port unchanged.
        if (cfg_.exist_room_prior and fitter_->has_room_polygon() and not inst.is_bearing_hypothesis)
        {
            const auto& ms = inst.model.state();
            if (not fitter_->point_in_room(Eigen::Vector2f(ms.cx, ms.cy), cfg_.exist_room_margin_m))
            {
                inst.existence.set(inst.existence.logodds() - cfg_.exist_out_of_room_gain);
                if (inst.existence.should_remove(cfg_.exist_removal_prob)) ++inst.existence_remove_streak;
                else                                                       inst.existence_remove_streak = 0;
                if (inst.existence_remove_streak >= cfg_.exist_remove_frames)
                    to_remove.push_back(id);
                continue;   // outside the room → no sensor evidence can rescue it; skip the normal channel
            }
        }

        // A bearing-only hypothesis has no depth (existence unjudgeable) and an un-initialised newborn has no
        // silhouette to project — both HOLD.
        if (inst.is_bearing_hypothesis or not inst.ai2_initialized)
            continue;

        // ── SILHOUETTE CHANNEL (the only one that may remove a door) ──────────────────────────────────
        const auto sil = fitter_->compute_silhouette_existence(inst);
        inst.dbg_sil_occ = sil.e_occ;     inst.dbg_sil_free  = sil.e_free;
        inst.dbg_sil_ndet = sil.n_detectable; inst.dbg_sil_ntotal = sil.n_total;
        inst.dbg_sil_noccl = sil.n_occluded;  inst.dbg_sil_ncells = sil.n_cells;
        inst.dbg_sil_central = sil.central_frac(); inst.dbg_sil_resolv = sil.resolvability();
        if (sil.n_detectable == 0)
        {
            // NOT PROBED this frame — behind the robot, out of the frustum, or fully occluded. rc::exist HOLDs.
            // This branch is the whole fix for "the door disappears when the robot turns around": there is no
            // pd to fall back on, because a sensor that did not look produces no evidence either way.
            inst.dbg_sil_pdetect = 0.0f; inst.dbg_sil_free_eff = 0.0f;
            continue;
        }

        // OCCUPANCY, weighted by how much of its mask the door MODEL actually explains. last_clutter_frac is the
        // mixture's clutter responsibility — the same per-point posterior the free energy is built from — so a
        // blob the door model fits none of confirms nothing. This is the missing term that let door_2 (FE 1.33
        // vs 0.17 for the real door, clutter 0.38 → 0.98) climb to L=2.29 and then sit there: the old channel
        // scored point MASS, never fit QUALITY. Stale between wins, but e_occ is only non-zero on a frame whose
        // door mask overlaps this silhouette, i.e. one where the instance is being detected.
        const float q_explain = std::clamp(1.0f - inst.last_clutter_frac, 0.0f, 1.0f);
        const float e_occ = sil.e_occ * q_explain;

        // ABSENCE. Suppressed entirely when the door was DETECTED this frame by any sensor (it is not gone —
        // a ZED false-negative, or a door only the ricoh resolved, must never vote itself away). This is the
        // honest form of "ricoh confirms, never removes": the old code enforced it on the positive side while
        // the negative side removed through range_detectability, which is precisely the ricoh's own reach.
        const bool observed = inst.frames_since_detection == 0;
        const float raw_free = observed ? 0.0f : sil.e_free;
        // P(detect | present, geometry): could this view have resolved a door that IS there?
        //   resolvability — is the panel big enough in the image to segment (range × foreshortening),
        //   in_fov_frac  — how much of it the real frustum + occluders actually left visible,
        //   central_frac — whether the robot is LOOKING at it rather than clipping the frustum edge.
        // Predicted-but-absent counts toward REMOVAL only in proportion to p_detect; the remainder is epistemic
        // surprise ("I cannot resolve this from here"), which should send the robot to look, never delete.
        const float p_detect = sil.resolvability() * sil.in_fov_frac() * sil.central_frac();
        const float sfree = raw_free * p_detect;
        inst.dbg_sil_pdetect = p_detect; inst.dbg_sil_free_eff = sfree;

        inst.existence.integrate(rc::exist::mask_evidence(e_occ, sfree, sil.n_detectable, sm));

        // Debounce on consecutive EVIDENCE cycles (not wall-clock), so a transient hiccup cannot delete a real
        // door, and removal always reflects sustained agreement across frames.
        if (inst.existence.should_remove(cfg_.exist_removal_prob)) ++inst.existence_remove_streak;
        else                                                       inst.existence_remove_streak = 0;
        if (inst.existence_remove_streak >= cfg_.exist_remove_frames)
            to_remove.push_back(id);
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
                const float oblq = fitter_->door_view_obliquity(inst);   // diagnostic only (see door_fitter.h)
                std::print("door_concept: [existence] {} L={:.2f} p={:.2f} pos=({:.2f},{:.2f}) inroom={} roomprior={} "
                           "won={} since_det={} | sil occ={:.0f} free={:.0f} free_eff={:.1f} ndet={}/{} occl={} "
                           "resolv={:.2f} central={:.2f} pdet={:.2f} oblq={:.2f} strk={}\n",
                           inst.node_name, inst.existence.logodds(), inst.existence.p_exists(), ms.cx, ms.cy,
                           inroom ? 1 : 0, has_poly ? 1 : 0,
                           inst.assigned_mask_idx >= 0 ? 1 : 0, inst.frames_since_detection,
                           inst.dbg_sil_occ, inst.dbg_sil_free, inst.dbg_sil_free_eff,
                           inst.dbg_sil_ndet, inst.dbg_sil_ntotal, inst.dbg_sil_noccl,
                           inst.dbg_sil_resolv, inst.dbg_sil_central, inst.dbg_sil_pdetect, oblq,
                           inst.existence_remove_streak);
                // Same diagnostic to a CSV. How to read it: ndet=0 ⇒ the door was NOT looked at (out of frustum
                // or fully occluded) ⇒ HOLD, L must not move — that is the "robot turned around" case. A real
                // door in view shows occ≫free. A phantom in a resolving view (pdet high) shows free≫occ and a
                // rising strk. free_eff≪free means "seen, but this view could not resolve it" ⇒ go verify, not delete.
                static std::ofstream ex_csv = []{ std::ofstream f("etc/door_existence_log.csv", std::ios::trunc);
                    f << "cycle,node,L,p_exists,cx,cy,inroom,roomprior_loaded,won,since_det,"
                         "sil_occ,sil_free,sil_free_eff,n_detectable,n_total,n_occluded,"
                         "resolvability,central_frac,p_detect,oblq,remove_streak\n"; return f; }();
                if (ex_csv)
                {
                    ex_csv << ex_dbg << ',' << inst.node_name << ',' << inst.existence.logodds() << ','
                           << inst.existence.p_exists() << ',' << ms.cx << ',' << ms.cy
                           << ',' << (inroom ? 1 : 0) << ',' << (has_poly ? 1 : 0)
                           << ',' << (inst.assigned_mask_idx >= 0 ? 1 : 0) << ',' << inst.frames_since_detection
                           << ',' << inst.dbg_sil_occ << ',' << inst.dbg_sil_free << ',' << inst.dbg_sil_free_eff
                           << ',' << inst.dbg_sil_ndet << ',' << inst.dbg_sil_ntotal << ',' << inst.dbg_sil_noccl
                           << ',' << inst.dbg_sil_resolv << ',' << inst.dbg_sil_central << ',' << inst.dbg_sil_pdetect
                           << ',' << oblq << ',' << inst.existence_remove_streak << '\n';
                    ex_csv.flush();
                }
            }

    for (const std::uint64_t id : to_remove)
    {
        const auto it = fitter_->instances().find(id);
        const float L = (it != fitter_->instances().end()) ? it->second.existence.logodds() : 0.0f;
        std::print("door_concept: [existence] REMOVE id={} (L={:.2f}, p={:.3f} < {:.3f} for {} evidence cycles)\n",
                   id, L, 1.0f / (1.0f + std::exp(-L)), cfg_.exist_removal_prob, cfg_.exist_remove_frames);
        if (it != fitter_->instances().end())
        {
            // Retain the identity before deleting the node: a door that comes back at the same place is the SAME
            // door, and must resume its accumulated belief instead of being re-born as door_N+1 (see remember_ghost).
            remember_ghost(it->second);
            log_tracker_event("REMOVE", id, it->second.model.state().cx, it->second.model.state().cy,
                              std::format("logodds {:.2f}", L));
            it->second.affordance.remove();
        }
        fitter_->forget_node(id);
        G->delete_node(id);
    }
}

///////////////////////////////////////////////////////////////
void SpecificWorker::process_door_node(const DSR::Node& node)
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
    const auto observation = fitter_->observe(inst, node);

    // Stale check: skip heavy update if data hasn't moved for too long
    if (not observation.has_fresh_data and inst.matched_frames < 5)
        return;

    const float free_energy = fitter_->run_inference(inst, observation);
    publish_door_cycle(inst, node, observation, free_energy);
}


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

void SpecificWorker::publish_door_intentions(rc::DoorInstance& inst,
                                              uint64_t node_id,
                                              const DoorObservation& observation,
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
    const float state_delta = inst.has_prev_conv_state
        ? (std::abs(s.cx - p.cx) + std::abs(s.cy - p.cy) + std::abs(s.w - p.w) +
           std::abs(s.h - p.h) + std::abs(s.yaw - p.yaw))
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
            std::print("door_concept: node '{}' STABLE (F={:.4f})\n",
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

void SpecificWorker::step_epistemic(rc::DoorInstance& inst, DSR::Node& node)
{
    if (inst.epistemic_cooldown > 0)
        --inst.epistemic_cooldown;

    // Controller-completion hold (anti-churn): the door affordance completes on a weak detection
    // (contract goal conf≥0.20), which fires almost instantly — before ΔH has decayed. Start a short
    // cooldown so we don't re-offer a just-completed door while its gain is still high. We do NOT
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
        epistemic_planner_.compute(inst.ai2_belief, cfg_.ai2_range_noise_lat_per_m, cfg_.ai2_sigma_base_m);
    if (not prop.valid or not prop.is_finite())
        return;   // degenerate (non-finite) fit — leave the existing affordance node as-is, retry next cycle

    // Belief→knowledge governor WITHOUT deleting the node: keep publishing the affordance every cycle
    // with its TRUE expected information gain ΔH (nats). A low gain is published as-is so the controller's
    // grounded EFE selection simply doesn't pick it (cost outweighs the small epistemic value), and it
    // re-arms automatically when the belief degrades and ΔH climbs again — no satisfy-latch to get stuck
    // in, no node churn. During the post-completion hold the gain is forced to 0 so a just-finished door
    // isn't re-claimed before its belief has settled.
    if (inst.epistemic_cooldown > 0)
        prop.epistemic_gain = 0.0f;

    // Bearing-only hypothesis (Part C-birth): author an ORIENT affordance whose target yaw IS the bearing,
    // so the controller rotates to look down the ray (the broad along-ray Σ already makes prop's gain high).
    const bool orient = inst.is_bearing_hypothesis;
    if (orient)
        prop.epistemic_target_yaw_rad = inst.hypothesis_azimuth;

    // Write attributes to the door node (read by legacy consumers)
    scene_graph_->write_epistemic_proposal(node, prop);
    // Publish / refresh dedicated affordance node (persists; update_node refreshes target+gain)
    const auto affordance_node_before = inst.affordance.node_id();
    inst.affordance.update(prop, orient);
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
    // Doors are generic `object` nodes named "door_*" (schema migration).
    if (type != "object")
        return;

    const auto node_opt = G->get_node(id);
    if (not node_opt.has_value())
        return;
    if (not node_opt.value().name().starts_with("door"))
        return;

    fitter_->ensure_instance(node_opt.value(), room_node_id_);
}

void SpecificWorker::modify_node_attrs_slot(std::uint64_t id,
                                             const std::vector<std::string>& att_names)
{
    // Delegate to the affordance state machine for any instance whose affordance
    // node was modified (controller claim/completion updates active/pending)
    for (auto& [door_id, inst] : fitter_->instances())
        if (inst.affordance.node_id() == id)
            inst.affordance.on_node_modified(id);

    // React to mission-controller clearing epistemic_pending on the door node itself
    if (fitter_->instances().count(id))
    {
        const bool pending_cleared = std::any_of(att_names.begin(), att_names.end(),
            [](const std::string& s) { return s == "epistemic_pending"; });

        if (pending_cleared)
        {
            auto node_opt = G->get_node(id);
            if (node_opt.has_value())
            {
                const auto v = G->get_attrib_by_name<epistemic_pending_att>(node_opt.value());
                if (v.has_value() and not v.value())
                    fitter_->instances().at(id).epistemic_pending = false;
            }
        }
    }
}

void SpecificWorker::del_node_slot(std::uint64_t id)
{
    // Notify affordance in case its own DSR node was deleted externally
    for (auto& [door_id, inst] : fitter_->instances())
        if (inst.affordance.node_id() == id)
            inst.affordance.on_node_deleted(id);

    if (fitter_->instances().count(id))
    {
        std::print("door_concept: node {} removed from DSR, destroying instance\n", id);
        fitter_->instances().erase(id);
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




