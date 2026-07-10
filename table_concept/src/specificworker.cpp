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
 * SpecificWorker — table_concept agent
 *
 * Implements the Active Inference loop described in TABLE_CONCEPT.md §11.2:
 *
 *  ① Read sensing attributes from DSR table nodes
 *  ② Update the historical sample queue with fresh near-surface candidates
 *  ③ Run gradient-descent steps on the 7-param generative model (SDF + FE)
 *  ④ Write updated model parameters back to DSR (RT edge + geometry attrs)
 *  ⑤ Check convergence and set model_stable_att
 *  ⑥ Compute epistemic action proposals (viewpoint → mission-controller)
 *  ⑦ Detect divergence and set request_full_sample_att
 */

#include "specificworker.h"

#include <filesystem>
#include <print>
#include <cstdlib>   // std::_Exit — crash-free terminal shutdown
#include <thread>    // brief DDS flush before _Exit
#include <chrono>
#include <iostream>  // std::cout/cerr flush
#include <QSettings>   // persist the standalone dashboard window geometry
#include <QByteArray>

#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>
#include <unordered_set>
#include <vector>

// DSR attribute name tags — generated from dsr_attr_name.h
#include <dsr/api/dsr_api.h>

namespace {

// Scalar model-uncertainty readout for model_uncertainty_att / the dashboard: the sum of the belief's
// per-DOF posterior stds over position (cx,cy) + size (w,h), in metres, from the AI2 covariance Σ over
// [cx,cy,H,w,h,yaw]. Shrinks as the robot gathers viewpoints. 0 before the belief is seeded.
float belief_uncertainty(const rc::TableInstance& inst)
{
    if (not inst.ai2_initialized)
        return 0.0f;
    const auto& S = inst.ai2_belief.covariance();
    const auto sd = [&](int i) { return std::sqrt(std::max(0.0f, S(i, i))); };
    return sd(0) + sd(1) + sd(3) + sd(4);
}

// Two tables cannot share physical space. The footprint is the oriented rectangle (cx,cy,w,h,yaw) in the
// room plane; these helpers compute the area of overlap between two footprints so the merge operator can
// collapse duplicate instances. Corners are returned CCW (local order (-,-),(+,-),(+,+),(-,+)).
std::array<Eigen::Vector2f, 4> footprint_corners(const rc::TableState& s)
{
    const float c = std::cos(s.yaw), sn = std::sin(s.yaw);
    const Eigen::Vector2f ex(c, sn), ey(-sn, c), ctr(s.cx, s.cy);
    const float hw = 0.5f * s.w, hh = 0.5f * s.h;
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

// Overlap area as a fraction of the SMALLER footprint (1.0 = one table fully inside the other).
float footprint_overlap_ratio(const rc::TableState& a, const rc::TableState& b)
{
    const auto ca = footprint_corners(a), cb = footprint_corners(b);
    const auto inter = clip_poly(std::vector<Eigen::Vector2f>(ca.begin(), ca.end()), cb);
    const float ai = poly_area(inter);
    const float amin = std::min(poly_area({ca.begin(), ca.end()}), poly_area({cb.begin(), cb.end()}));
    return amin > 1e-6f ? ai / amin : 0.0f;
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
    std::print("table_concept: SpecificWorker destroyed.\n");
}

void SpecificWorker::request_shutdown()
{
    if (shutting_down_.exchange(true))
        return;

    save_window_settings();
    save_dashboard_geometry();   // the standalone dashboard is not in `windows`, so save it explicitly

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

// Persist/restore the standalone dashboard window's geometry. The generated save_window_settings()
// only covers the QMainWindow(s) in `windows`; our extracted top-level widget is separate, so we
// carry its own QSettings entry (mirrors room_concept's RoomViewer).
void SpecificWorker::restore_dashboard_geometry()
{
    if (not custom_widget_)
        return;
    QSettings settings(QStringLiteral("RoboComp"), QStringLiteral("table_concept"));
    const QByteArray geom = settings.value(QStringLiteral("DashboardWindow_geometry")).toByteArray();
    if (not geom.isEmpty())
        custom_widget_->restoreGeometry(geom);
    else
        custom_widget_->resize(560, 620);
}

void SpecificWorker::save_dashboard_geometry() const
{
    if (not custom_widget_)
        return;
    QSettings settings(QStringLiteral("RoboComp"), QStringLiteral("table_concept"));
    settings.setValue(QStringLiteral("DashboardWindow_geometry"), custom_widget_->saveGeometry());
    settings.sync();
}

// ─── Initialisation ──────────────────────────────────────────────────────────

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
    presence_coordinator_.set_transition_hooks({
        .request_presence_ready = [this]() { emit presenceReady(); },
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
        .on_operating_enter = [this]()
        {
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
            compute();
        },
        .on_degraded_enter = [this]()
        {
            if (shutting_down_)
                return;
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
    scene_graph_ = std::make_unique<rc::TableSceneGraph>(
        G, rt_api_.get(), cfg_, [this] { trigger_graph_layout_twopi(); });

    // Subscribe to graph signals
    connect(G.get(), &DSR::DSRGraph::update_node_signal,
            this, &SpecificWorker::modify_node_slot);
    connect(G.get(), &DSR::DSRGraph::update_node_attr_signal,
            this, &SpecificWorker::modify_node_attrs_slot);
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

    // Part B: localization/chain covariance on the published RT edge (mirrors bottle_concept).
    gaussian_api_ = std::make_unique<DSR::InnerGaussianAPI>(G.get());
    fitter_->set_chain_cov_source(gaussian_api_.get(), "zed", cfg_.rt_cov_add_chain);

    // YOLO-independent LiDAR range channel: lidar3D media-plane consumer that stages each cycle's sweep in the
    // room frame for the fitter's range factor. Dormant (no DDS participant) unless TableModel.LidarPrecision
    // > 0. Subscriber is brought up lazily on the compute/main thread once the lidar3D node + descriptor exist.
    lidar_ingestor_ = std::make_unique<rc::TableLidarIngestor>(G, inner_eigen_.get(), cfg_);

    // Build rc::EpistemicPlanner (info-gain scoring only; stand-off distance is the sole parameter).
    epistemic_planner_ = rc::EpistemicPlanner(cfg_.obs_distance);

    // Stale affordance nodes are swept on entering Operating (presence hook) and on shutdown — see
    // remove_stale_affordance_nodes(), keyed on the parent object type (robust to node-name renames).

    // ── Time-series dashboard — its OWN top-level window ──────────────────────
    // Extracted from the DSR graph dock (add_custom_widget_to_dock) into a standalone window, so the
    // dashboard shows even with Agent.graph=false (no DSRViewer created). Mirrors room_concept and
    // kinova_controller. The TimeSeriesPlot is a plain QWidget (no QOpenGL backing store), safe as a
    // top-level. NOT WA_DeleteOnClose: closing must only HIDE it, or the ts_*_plot_ pointers the
    // compute() feed uses would dangle. A QApplication always exists (generated/main.cpp).
    {
        custom_widget_ = new Custom_widget("Table Model — Free Energy, Coverage Deficit, Residuals & Dimensions (w,h)");
        custom_widget_->setWindowTitle(QStringLiteral("table_concept — belief dashboard"));
        restore_dashboard_geometry();
        custom_widget_->show();

        // Create plot inside frame_series
        auto* series_layout = new QVBoxLayout(custom_widget_->frame_series);
        series_layout->setContentsMargins(0, 0, 0, 0);
        custom_widget_->frame_series->setLayout(series_layout);

        ts_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
        ts_plot_->set_visible_window(60.f);
        series_layout->addWidget(ts_plot_);

        // FE SURPRISE (attention signal) on its own panel — it lives on a much smaller scale (~0–1) than the FE
        // (~2–8), so it needs the full panel height to be readable. Spikes when a table moves, decays as the fit
        // re-converges. See the belief/fitter plumbing (inst.fe_surprise).
        ts_surprise_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
        ts_surprise_plot_->set_visible_window(60.f);
        series_layout->addWidget(ts_surprise_plot_);

        ts_cov_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
        ts_cov_plot_->set_visible_window(60.f);
        series_layout->addWidget(ts_cov_plot_);

        ts_res_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
        ts_res_plot_->set_visible_window(60.f);
        series_layout->addWidget(ts_res_plot_);

        // Inferred table dimensions (w, h) — the size DOFs the stabiliser targets. Watch these to
        // confirm the belief has stopped jittering between fresh masks (flat = stable).
        ts_state_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
        ts_state_plot_->set_visible_window(60.f);
        series_layout->addWidget(ts_state_plot_);

        // (D) Counter-evidence accumulator S_w/S_h (the CUSUM gate is always on). Watch S charge on a
        // surprise and decay back (glitch absorbed) vs climb to the barrier (a real change unlocks).
        {
            ts_ce_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
            ts_ce_plot_->set_visible_window(60.f);
            series_layout->addWidget(ts_ce_plot_);
        }

        // GenericWorker::initialize() may have already started compute(), so
        // some instances can exist before the plots are constructed.
        for (const auto& [_, inst] : fitter_->instances())
        {
            ts_plot_->add_series(inst.node_name + "_fe", QColor(255, 170, 0), 1.1f);
            ts_cov_plot_->add_series(inst.node_name + "_cov", QColor(0, 190, 255), 1.1f);
            ts_res_plot_->add_series(inst.node_name + "_res", QColor(170, 80, 255), 1.1f);
            ts_state_plot_->add_series(inst.node_name + "_w", QColor(255, 90, 90), 1.1f);
            ts_state_plot_->add_series(inst.node_name + "_h", QColor(90, 200, 90), 1.1f);
            if (ts_ce_plot_)
            {
                ts_ce_plot_->add_series(inst.node_name + "_ceW", QColor(255, 90, 90), 1.1f);
                ts_ce_plot_->add_series(inst.node_name + "_ceH", QColor(90, 200, 90), 1.1f);
            }
        }
    }

    // ── Evidence-consuming monitor — its OWN top-level window (per-instance snapshot + global counters) ──
    // Same lifecycle discipline as the dashboard above: plain QWidget (no QOpenGL), built on the main thread,
    // updated from compute() (which is the GUI thread here). Only HIDDEN on close, never deleted (compute()
    // keeps a raw pointer). Shows even with Agent.graph=false.
    evidence_monitor_ = new rc::EvidenceMonitor(QStringLiteral("table_concept — evidence monitor"));
    evidence_monitor_->show();
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

    const bool fresh_masks = mask_ingestor_->refresh();

    // EvidenceMonitor per-cycle counters (cumulative *_cum fields persist across cycles). The producers below
    // (tracker / merge / removal) add to these; the snapshot is pushed at the end of the cycle.
    ev_g_.births = ev_g_.merges = ev_g_.removals = 0;
    ev_g_.mask_stale = not fresh_masks;

    run_instance_tracker();   // data-driven birth / associate / merge (the only instance-lifecycle path)

    // Stage this cycle's LiDAR sweep (room frame) for the fitter's range factor. clear-then-set so the factor
    // never consumes a stale sweep; pump() is main-thread (reads the graph) + dormant while the feature is off.
    fitter_->clear_lidar_sweep();
    bool fresh_sweep = false;
    if (lidar_ingestor_ and lidar_ingestor_->pump())
    {
        fitter_->set_lidar_sweep(lidar_ingestor_->sweep_room(), lidar_ingestor_->origin_room());
        fresh_sweep = true;
    }

    const auto table_nodes = G->get_nodes_by_type("table");
    for (const auto& node : table_nodes)
        process_table_node(node);

    // Ricoh 360 = peripheral attention: associate ricoh detections to tables BY DIRECTION (after the ZED fits,
    // so table positions are current); an unassigned bearing becomes a "seek a ZED view here" attention target.
    process_ricoh_bearings();

    // Evidence-based removal: each existence channel integrates on its OWN sensor cadence (silhouette/mask on a
    // fresh mask frame, LiDAR carve on a fresh sweep) — a camera-only cycle still accrues absence, a LiDAR-only
    // cycle still carves free space. After the fits so footprints are current. OFF unless enabled.
    if (cfg_.existence_removal_enabled)
        update_existence_and_remove(fresh_masks, fresh_sweep);

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
}

// Remove timeseries series for tables no longer present in the graph, so a removed table's lines don't persist
// (and a table reborn under the same name gets fresh series via the idempotent add_series in the feed). Runs
// every compute cycle; cheap (a set diff over the few live instances).
void SpecificWorker::prune_dead_series()
{
    if (not ts_plot_) return;
    std::unordered_set<std::string> live;
    for (const auto& [id, inst] : fitter_->instances()) live.insert(inst.node_name);
    for (auto it = ts_known_tables_.begin(); it != ts_known_tables_.end();)
    {
        if (live.contains(*it)) { ++it; continue; }
        const std::string& n = *it;
        ts_plot_->remove_series(n + "_fe");   ts_plot_->remove_series(n + "_base");
        if (ts_surprise_plot_) ts_surprise_plot_->remove_series(n + "_surprise");
        if (ts_cov_plot_)      ts_cov_plot_->remove_series(n + "_cov");
        if (ts_res_plot_)      ts_res_plot_->remove_series(n + "_res");
        if (ts_state_plot_)  { ts_state_plot_->remove_series(n + "_w");  ts_state_plot_->remove_series(n + "_h"); }
        if (ts_ce_plot_)     { ts_ce_plot_->remove_series(n + "_sW");    ts_ce_plot_->remove_series(n + "_sH"); }
        it = ts_known_tables_.erase(it);
    }
    for (const auto& [id, inst] : fitter_->instances()) ts_known_tables_.insert(inst.node_name);
}

// Ricoh 360 as PERIPHERAL ATTENTION. A ricoh detection's DIRECTION (bearing from the robot) is reliable even
// when its centroid/range is biased, so we use ONLY that: associate a ricoh bearing to an existing table if the
// bearing falls within that table's angular extent (its circumscribed radius over its range — a physical size,
// not a tuned gate). A ricoh bearing that NO table explains is UNASSIGNED → an attention target ("there is
// something table-like in that direction that the ZED hasn't confirmed; seek a ZED view there"). Never births,
// never fits — that is the ZED's job. This is step 1–3 of the peripheral→saccade→foveal design; the controller
// consuming the target (turn the ZED to the bearing) is step 4.
void SpecificWorker::process_ricoh_bearings()
{
    ricoh_attention_targets_.clear();
    if (not cfg_.use_ricoh_slices or not inner_eigen_)   // flag now gates ONLY the attention path (ricoh never fits)
        return;
    const auto& pkt = mask_ingestor_->packet();
    if (not pkt.valid)
        return;
    const auto rtb = inner_eigen_->get_transformation_matrix("room", "body", 0);   // robot pose in room
    if (not rtb.has_value())
        return;
    const auto& Tm = rtb.value();
    const float bx = static_cast<float>(Tm(0, 3)), by = static_cast<float>(Tm(1, 3));   // body XY in room
    const auto wrap = [](float a) { return std::remainder(a, 2.0f * static_cast<float>(M_PI)); };

    for (const auto& sl : pkt.slices)
    {
        if (sl.label != "table" or sl.depth_var <= 0.0f)          // ricoh (depth-carrying 360) detections only
            continue;
        if (sl.confidence < cfg_.ricoh_birth_conf)                // only reasonably confident peripheral blobs
            continue;

        // Robust position from the mask's 3D points: component-wise MEDIAN (resists the see-through/outlier tail
        // that biases the centroid). Foreground-gated upstream, so the median sits near the tabletop surface.
        // Falls back to the centroid if too few points. Used for the ATTENTION target's bearing + rough range
        // ONLY — never the fit (a partial-view range is biased, so it decides WHERE TO LOOK / WHICH table, not
        // where the table IS; the ZED measures that). Bearing is reliable; range is indicative.
        float mx = sl.centroid.x(), my = sl.centroid.y();
        const std::size_t b = std::min(sl.support_begin, pkt.support_points.size());
        const std::size_t e = std::min(sl.support_end,   pkt.support_points.size());
        if (e > b + 8)
        {
            std::vector<float> xs, ys; xs.reserve(e - b); ys.reserve(e - b);
            for (std::size_t i = b; i < e; ++i) { xs.push_back(pkt.support_points[i].x()); ys.push_back(pkt.support_points[i].y()); }
            const std::size_t k = xs.size() / 2;
            std::nth_element(xs.begin(), xs.begin() + k, xs.end()); mx = xs[k];
            std::nth_element(ys.begin(), ys.begin() + k, ys.end()); my = ys[k];
        }
        const float br  = std::atan2(my - by, mx - bx);                            // bearing to the robust point
        const float rng = std::sqrt((mx - bx) * (mx - bx) + (my - by) * (my - by));// rough range (biased, indicative)

        bool assigned = false;
        for (const auto& [id, inst] : fitter_->instances())
        {
            const auto& s = inst.model.state();
            const float dx = s.cx - bx, dy = s.cy - by;
            const float dist = std::sqrt(dx * dx + dy * dy);
            if (dist < 1e-3f) { assigned = true; break; }
            const float tb   = std::atan2(dy, dx);                                  // bearing to the table
            const float rad  = 0.5f * std::sqrt(s.w * s.w + s.h * s.h);             // circumscribed footprint radius
            const float tol  = std::atan2(rad, dist) + 0.05f;                       // TIGHT: table angular half-size
            const float band = rad + 1.0f;                                          // GENEROUS: ricoh range is rough
            // Assign only if the detection matches this table in BOTH direction AND (rough) range — so a new,
            // unconfirmed table hiding along the SAME bearing as a known one (different range) stays UNASSIGNED
            // and still raises attention, instead of collapsing onto the known table (bearing-only would miss it).
            if (std::abs(wrap(br - tb)) < tol and std::abs(rng - dist) < band) { assigned = true; break; }
        }
        if (not assigned)
            ricoh_attention_targets_.push_back({br, rng, sl.confidence, {mx, my}});
    }

    ev_g_.ricoh_attention = static_cast<int>(ricoh_attention_targets_.size());
    static int rb_dbg = 0;
    if (not ricoh_attention_targets_.empty() and (rb_dbg++ % 10 == 0))
        for (const auto& t : ricoh_attention_targets_)
            std::print("[ricoh-attention] UNASSIGNED table bearing={:+.0f}° range≈{:.1f}m conf={:.2f} → seek ZED view\n",
                       t.bearing_rad * 180.0f / static_cast<float>(M_PI), t.range_m, t.confidence);
}

// Build the per-instance snapshot from live instance state and push it to the monitor at ~5 Hz (a full
// QTableWidget rebuild every compute cycle would waste the GUI thread). All reads are main-thread.
void SpecificWorker::refresh_evidence_monitor()
{
    if (not evidence_monitor_)
        return;
    const auto now = std::chrono::steady_clock::now();
    if (last_monitor_tp_.time_since_epoch().count() != 0
        and std::chrono::duration<float>(now - last_monitor_tp_).count() < 0.2f)
        return;
    last_monitor_tp_ = now;

    std::vector<rc::EvidenceRow> rows;
    rows.reserve(fitter_->instances().size());
    for (const auto& [id, inst] : fitter_->instances())
    {
        rc::EvidenceRow x;
        x.node       = inst.node_name;
        x.conf       = inst.last_mask_confidence;
        x.since_det  = inst.frames_since_detection;
        x.n_zed      = inst.dbg_n_zed_slices;
        x.n_ricoh    = inst.dbg_n_ricoh_slices;
        x.cand       = inst.dbg_cand_pts;
        x.resid      = inst.dbg_resid_pts;
        x.gated      = inst.dbg_gated;
        x.energy     = inst.dbg_energy;
        x.R          = inst.dbg_R;
        x.lidar_rays = inst.dbg_lidar_rays;
        x.lidar_raw  = inst.dbg_lidar_raw;
        x.lidar_resid= inst.dbg_lidar_resid_m;
        x.cov_ang    = inst.dbg_lidar_cov_ang;
        x.vacate     = inst.ai2_belief.last_vacate_beams();
        x.coverage   = inst.ai2_belief.last_coverage_pts();
        x.L          = inst.existence.logodds();
        x.p          = inst.existence.p_exists();
        x.ex_locc    = inst.dbg_ex_lidar_occ;  x.ex_lfree = inst.dbg_ex_lidar_free;  x.ex_ln    = inst.dbg_ex_lidar_n;
        x.ex_lfree_eff = inst.dbg_ex_lidar_free_eff;
        x.ex_socc    = inst.dbg_ex_sil_occ;    x.ex_sfree = inst.dbg_ex_sil_free;    x.ex_sndet = inst.dbg_ex_sil_ndet;
        x.ex_sfree_eff = inst.dbg_ex_sil_free_eff;
        x.streak     = inst.existence_remove_streak;
        const auto& s = inst.ai2_belief.state();
        x.w = s.w; x.h = s.h; x.H = s.H;
        rows.push_back(std::move(x));
    }
    evidence_monitor_->update_view(ev_g_, rows);
}

// Carve the LiDAR sweep against every table footprint and integrate the per-instance existence log-odds, then
// remove the tables whose volume is demonstrably empty (P(occupied) < ExistenceRemovalProb). The shared
// occupancy carve (common/existence_belief.h) supplies occ/free evidence with common-mode saturation + the
// n_reached HOLD gate; the hollow guard suppresses interior free-evidence while a table is actively observed.
void SpecificWorker::update_existence_and_remove(bool fresh_masks, bool fresh_sweep)
{
    // Decoupled cadence: the SILHOUETTE/mask channel integrates on a fresh MASK frame (camera clock), the LiDAR
    // free-space carve on a fresh SWEEP (LiDAR clock). Each accrues its evidence independently, so a camera-only
    // cycle still votes absence and a LiDAR-only cycle still carves free space. The removal debounce advances
    // only on an EVIDENCE cycle (when a channel actually integrated), so mixed sensor rates don't bias it.
    const std::vector<Eigen::Vector3f>* sweep = nullptr;
    Eigen::Vector3f origin = Eigen::Vector3f::Zero();
    if (fresh_sweep and lidar_ingestor_)
    {
        const auto& s = lidar_ingestor_->sweep_room();
        if (not s.empty()) { sweep = &s; origin = lidar_ingestor_->origin_room(); }
    }
    if (not fresh_masks and sweep == nullptr)   // no new evidence on either channel this cycle
        return;

    rc::exist::SensorModel sm;
    sm.sensor_sigma_m = cfg_.existence_sensor_sigma_m;
    sm.detection_prob = cfg_.existence_detection_prob;
    sm.clutter_prob   = cfg_.existence_clutter_prob;

    // Absence-confidence vs range: a miss at long range is weak evidence of removal (the sensor likely could
    // not resolve the object). c = (range_ref/range)^power capped at 1 → 1 up close, →0 far. Scales the FREE
    // (absence) half only; occupancy stays fully informative. power 0 disables (constant 1).
    const auto absence_range_conf = [&](float range_m) -> float
    {
        const float rr = cfg_.existence_absence_range_ref_m;
        if (rr <= 0.0f or cfg_.existence_absence_range_power <= 0.0f) return 1.0f;
        return std::min(1.0f, std::pow(rr / std::max(range_m, 1e-3f), cfg_.existence_absence_range_power));
    };

    std::vector<std::uint64_t> doomed;
    for (auto& [id, inst] : fitter_->instances())
    {
        if (not inst.ai2_initialized) continue;
        inst.existence.set_max(cfg_.existence_logodds_max);
        const auto& bs = inst.ai2_belief.state();
        const auto& S  = inst.ai2_belief.covariance();
        const float surf_sigma = std::sqrt(std::max(0.0f, 0.5f * (S(0, 0) + S(1, 1))));   // footprint position σ
        const bool observed = inst.frames_since_detection == 0;                      // fresh mask this cycle
        bool integrated = false;

        // SILHOUETTE / MASK channel (pixel-level) — CAMERA clock: project the tabletop silhouette and compare,
        // per predicted-VISIBLE pixel, against the current YOLO foreground. Lit by a "table" mask ⇒ occupancy
        // (holds L up); lit by NOTHING ⇒ predicted-but-absent (the "gone" signal, present EVEN WITH NO YOLO MASK);
        // lit by a NON-table mask ⇒ occluded ⇒ HOLD (never false absence behind a nearer object). n_detectable==0
        // (out of FoV / fully occluded) ⇒ mask_evidence HOLDs.
        if (fresh_masks)
        {
            const auto sil = fitter_->compute_silhouette_existence(inst);
            inst.dbg_ex_sil_occ = sil.e_occ; inst.dbg_ex_sil_free = sil.e_free; inst.dbg_ex_sil_ndet = sil.n_detectable;
            if (sil.n_detectable > 0)
            {
                // Observed-guard (mirrors the LiDAR hollow guard): if the table was DETECTED by any sensor this
                // frame it is not gone, so suppress silhouette ABSENCE — a ZED false-negative, or a table seen
                // only by the ricoh whose silhouette the ZED-based check can't confirm, must never vote it away.
                // Occupancy always counts. (The ricoh-projected silhouette, once the producer ships 360 mask
                // pixels, will let absence be judged in the ricoh view directly instead of relying on this guard.)
                float sfree = observed ? 0.0f : sil.e_free;
                // Absence is evidence of removal only in proportion to how much of the table SHOULD have been
                // seen from here: in_fov_frac = n_detectable / n_total folds the real camera FRUSTUM (out-of-FoV
                // samples don't count) AND the occlusion HOLD together, and range_conf adds the angular-size /
                // detectability drop. A ricoh-only table (barely in the ZED frustum) → in_fov_frac ≈ 0 → HOLD.
                sfree *= absence_range_conf(sil.mean_range_m) * sil.in_fov_frac();
                inst.dbg_ex_sil_free_eff = sfree;   // effective absence after range/occlusion + observed-guard
                inst.existence.integrate(rc::exist::mask_evidence(sil.e_occ, sfree, sil.n_detectable, sm));
                integrated = true;
            }
        }

        // LiDAR occupancy carve of the TOP-SLAB z-band [H−t, H] — LiDAR clock (the slab spans the full footprint
        // there, so a beam hits rim/top = occupancy or passes through = gone — no hollow ambiguity between legs).
        if (sweep)
        {
            rc::exist::Evidence ev = rc::exist::carve_box(origin, *sweep, bs.cx, bs.cy, bs.yaw, bs.w, bs.h,
                                                          bs.H - rc::TableModel::TOP_THICKNESS, bs.H, surf_sigma, sm);
            inst.dbg_ex_lidar_occ = ev.e_occ; inst.dbg_ex_lidar_free = ev.e_free; inst.dbg_ex_lidar_n = ev.n_reached;
            if (ev.n_reached > 0)                                                    // 0 ⇒ not probed ⇒ HOLD
            {
                // Degrade the free/absence half by LiDAR range: far away the beams are sparse and the thin
                // top-slab subtends few of them, so a pass-through is weak evidence the tabletop is gone.
                const float lidar_range = (origin - Eigen::Vector3f(bs.cx, bs.cy, bs.H)).norm();
                ev.e_free *= absence_range_conf(lidar_range);
                // OCCUPANCY-ONLY by default: a solid-slab model vs a thin real tabletop makes LiDAR "free"
                // unreliable (beams under the top surface read as gone), so it must not drive removal — only
                // the camera silhouette does. Suppress free unless it was observed (hollow guard) OR the
                // ExistenceLidarAbsence override is on. Occupancy still counts, holding L up.
                const bool suppress_free = observed or not cfg_.existence_lidar_absence;
                inst.dbg_ex_lidar_free_eff = suppress_free ? 0.0f : ev.e_free;
                ev.log_odds_delta = rc::exist::hollow_guarded_delta(ev, suppress_free, sm);
                inst.existence.integrate(ev);
                integrated = true;
            }
        }

        // Debounce advances ONLY on an evidence cycle, so it counts sustained EVIDENCE agreement (not wall-clock
        // cycles) regardless of the two sensors' rates. A transient hiccup still can't delete a real table.
        if (integrated)
        {
            if (inst.existence.should_remove(cfg_.existence_removal_prob)) ++inst.existence_remove_streak;
            else                                                          inst.existence_remove_streak = 0;

            if (fitter_->should_log(inst))
                std::print("[{}] [existence] L={:.2f} p={:.2f} | lidar occ={:.1f} free={:.1f} n={} | sil occ={:.0f} free={:.0f} ndet={} | {} streak={}\n",
                           inst.node_name, inst.existence.logodds(), inst.existence.p_exists(),
                           inst.dbg_ex_lidar_occ, inst.dbg_ex_lidar_free, inst.dbg_ex_lidar_n,
                           inst.dbg_ex_sil_occ, inst.dbg_ex_sil_free, inst.dbg_ex_sil_ndet,
                           observed ? "obs" : "-", inst.existence_remove_streak);
        }
        if (inst.existence_remove_streak >= cfg_.existence_remove_frames)
            doomed.push_back(id);
    }
    ev_g_.removals     += static_cast<int>(doomed.size());   // EvidenceMonitor counters
    ev_g_.removals_cum += static_cast<long>(doomed.size());
    for (const std::uint64_t id : doomed)
    {
        std::print("table_concept: [existence] removing table id={} — free-space evidence (empty volume)\n", id);
        if (auto it = fitter_->instances().find(id); it != fitter_->instances().end())
            it->second.affordance.remove();
        fitter_->forget_node(id);
        G->delete_node(id);
    }
}

// Data-driven multi-instance lifecycle (mirrors bottle_concept::run_instance_tracker). Tables are large
// static furniture, so birth_min_sep is wide, death is off, and overlaps merge. The only path that
// creates/associates table instances (the prior-scaffold + greedy nearest-mask were removed in Stage 2).
// Collapse instances whose footprints overlap (same physical table fitted twice): keep the one with more
// integrated fresh evidence, retire the other (affordance + node). Runs before tracking so a duplicate is
// gone before it is fed a mask. v1 keeps-best; precision-weighted DOF pooling is a later refinement.
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

            // Keep the more-observed instance (more integrated fresh frames); retire the other.
            const bool keep_i = ia->second.matched_frames >= ib->second.matched_frames;
            const std::uint64_t keep = keep_i ? ids[i] : ids[j];
            const std::uint64_t drop = keep_i ? ids[j] : ids[i];
            std::print("table_concept: [tracker] MERGE id={} into id={} (footprint overlap {:.2f})\n",
                       drop, keep, ratio);
            ++ev_g_.merges; ++ev_g_.merges_cum;   // EvidenceMonitor counter
            if (auto it = insts.find(drop); it != insts.end())
                it->second.affordance.remove();
            fitter_->forget_node(drop);
            G->delete_node(drop);
            removed.insert(drop);
            if (drop == ids[i]) break;   // this i is gone; advance to the next i
        }
    }
}

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
    tp.multi_det_per_track = true;   // fuse ZED + ricoh-360 slices of the same table (one update per slice)
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
        if (inst.ai2_initialized)
        {
            // Gate on the belief's position covariance (+ localization chain) so association uses the
            // Mahalanobis innovation S = P + R²I, not the Euclidean fallback (matters for multi-instance).
            const auto& S = inst.ai2_belief.covariance();   // Σ over [cx,cy,H,w,h,yaw]
            t.cov = Eigen::Matrix2f::Zero();
            t.cov(0, 0) = S(0, 0) + inst.chain_cov_xx;
            t.cov(1, 1) = S(1, 1) + inst.chain_cov_yy;
            t.has_cov = true;
        }
        tracks.push_back(t);
        inst.assigned_mask_idxs.clear();   // cleared; re-filled below with every slice associated this cycle
    }

    // Detections ← the current "table" mask slices (carry the slice index for the assignment). Built EVERY
    // cycle from the packet (even a stale one): the tracker needs continuous detections so its birth CANDIDATES
    // mature over birth_frames consecutive cycles and so association stays live — gating this on mask freshness
    // wiped the candidates every stale cycle and NOTHING ever birthed. Re-fitting a stale mask is prevented
    // separately, in process_table_node (per-instance frame_id gate), NOT here.
    // Each det's BIRTHABLE flag encodes trust: a dense ZED slice (depth_var==0) may spawn a table; a ricoh-360
    // / LiDAR-depth slice (depth_var>0) may only REFINE unless it is confident AND its depth is well-scored
    // (conf ≥ RicohBirthConf and depth_var ≤ RicohBirthMaxVar) — otherwise a sparse peripheral blob would
    // birth phantoms across the scene. All slices still ASSOCIATE regardless (that is the fusion).
    std::vector<rc::DetectionView> dets;
    const auto& pkt = mask_ingestor_->packet();
    if (pkt.valid)
        for (int i = 0; i < static_cast<int>(pkt.slices.size()); ++i)
        {
            const auto& sl = pkt.slices[i];
            if (sl.label != "table" or sl.support_end <= sl.support_begin)
                continue;
            if (sl.depth_var > 0.0f)   // RICOH is BEARING-ONLY: never births/associates/fits — the tracker sees
                continue;              // ZED masks only. Ricoh drives the attention path (process_ricoh_bearings).
            rc::DetectionView dv;
            dv.xy = {sl.centroid.x(), sl.centroid.y()};
            dv.slice_index = i;
            dv.birthable = true;       // ZED only reaches here → always birthable
            dets.push_back(dv);
        }

    const auto res = tracker_.update(tracks, dets);

    // Diagnostic (throttled, plus on any birth/death): reveals "1 slice" (upstream) vs "N slices, no birth".
    static int dbg = 0;
    const int n_assigned = static_cast<int>(std::count_if(res.assignment.begin(), res.assignment.end(),
                                                          [](int a){ return a >= 0; }));

    // EvidenceMonitor mask/tracker counters (merges are added in merge_overlapping_instances above).
    ev_g_.mask_frame_id = pkt.valid ? pkt.frame_id : -1;
    ev_g_.total_slices  = pkt.valid ? static_cast<int>(pkt.slices.size()) : 0;
    ev_g_.table_dets    = static_cast<int>(dets.size());
    ev_g_.assigned      = n_assigned;
    ev_g_.discarded     = static_cast<int>(dets.size()) - n_assigned;
    ev_g_.births       += static_cast<int>(res.births.size());
    ev_g_.births_cum   += static_cast<long>(res.births.size());
    if (++dbg % 30 == 0 or not res.births.empty() or not res.deaths.empty())
    {
        std::print("[tracker] instances={} table_dets={} assigned={} unassigned={} births={} deaths={}\n",
                   tracks.size(), dets.size(), n_assigned,
                   static_cast<int>(dets.size()) - n_assigned, res.births.size(), res.deaths.size());
        for (const auto& t : tracks)
            std::print("[tracker]   track id={} xy=({:.2f},{:.2f}) has_cov={}\n",
                       t.id, t.xy.x(), t.xy.y(), t.has_cov);
        for (const auto& d : dets)
            std::print("[tracker]   det slice={} xy=({:.2f},{:.2f})\n", d.slice_index, d.xy.x(), d.xy.y());
    }

    // DEATH: OFF by default — a table is rigid, persistent furniture, so a long occlusion (no mask for
    // many frames) is NOT absence and must not retire it. The ONLY way a table is removed is the MERGE
    // operator (two tables can't share space). Enable Tracker.DeathEnabled to restore miss-timer retirement.
    if (cfg_.tracker_death_enabled)
        for (const std::uint64_t id : res.deaths)
        {
            std::print("table_concept: [tracker] DEATH id={} (unobserved {} frames)\n", id, cfg_.tracker_death_frames);
            if (auto it = fitter_->instances().find(id); it != fitter_->instances().end())
                it->second.affordance.remove();
            fitter_->forget_node(id);
            G->delete_node(id);
        }

    // ASSOCIATE: route every matched detection's mask slice to its instance (read in observe_slice()). With
    // multi_det_per_track a track may collect SEVERAL slices (ZED + ricoh) → fused as sequential updates.
    for (int d = 0; d < static_cast<int>(dets.size()); ++d)
        if (res.assignment[d] >= 0)
        {
            const std::uint64_t id = tracks[res.assignment[d]].id;
            if (auto it = fitter_->instances().find(id); it != fitter_->instances().end())
                it->second.assigned_mask_idxs.push_back(dets[d].slice_index);
        }

    // BIRTH: spawn an instance from each promoted (persistently-unexplained) detection, seeding the
    // fitter with the detection XY so the model starts AT the table (not the 0,0 RT-read default).
    for (const int d : res.births)
    {
        const Eigen::Vector3f& c = pkt.slices[dets[d].slice_index].centroid;
        const auto new_id = scene_graph_->create_instance_from_detection(c, room_node_id_);
        if (new_id != 0)
            fitter_->note_birth(new_id, Eigen::Vector2f(c.x(), c.y()));
    }

    // SEAM-SPLIT instrumentation: per instance, split the slices fused this cycle into ZED (depth_var==0) vs
    // ricoh (depth_var>0). Two ricoh slices on one track ⇒ a 360 strip-seam split (a wide table straddling a
    // strip boundary → two partial detections that didn't merge). Log their centroids + separation so it can
    // be confirmed behind the robot, and count it globally (surfaced in the EvidenceMonitor header).
    int seam_candidates = 0;
    for (auto& [id, inst] : fitter_->instances())
    {
        inst.dbg_n_zed_slices = inst.dbg_n_ricoh_slices = 0;
        std::vector<Eigen::Vector3f> ricoh_centroids;
        for (const int idx : inst.assigned_mask_idxs)
            if (idx >= 0 and idx < static_cast<int>(pkt.slices.size()))
            {
                if (pkt.slices[idx].depth_var > 0.0f)
                { ++inst.dbg_n_ricoh_slices; ricoh_centroids.push_back(pkt.slices[idx].centroid); }
                else
                    ++inst.dbg_n_zed_slices;
            }
        if (inst.dbg_n_ricoh_slices >= 2)
        {
            ++seam_candidates;
            float max_sep = 0.0f;
            for (std::size_t a = 0; a < ricoh_centroids.size(); ++a)
                for (std::size_t b = a + 1; b < ricoh_centroids.size(); ++b)
                    max_sep = std::max(max_sep, (ricoh_centroids[a] - ricoh_centroids[b]).norm());
            std::print("[seam] {} fused {} ricoh slices (+{} zed) — SEAM-SPLIT candidate, max centroid sep {:.2f}m",
                       inst.node_name, inst.dbg_n_ricoh_slices, inst.dbg_n_zed_slices, max_sep);
            for (const auto& rc : ricoh_centroids)
                std::print("  c=({:.2f},{:.2f},{:.2f})", rc.x(), rc.y(), rc.z());
            std::print("\n");
        }
    }
    ev_g_.seam_splits      = seam_candidates;
    ev_g_.seam_splits_cum += seam_candidates;
}

///////////////////////////////////////////////////////////////
void SpecificWorker::process_table_node(const DSR::Node& node)
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
    // Detection-aliveness ages once per cycle; observe_slice() resets it to 0 when a slice is assigned.
    if (inst.frames_since_detection < 1000000)
        ++inst.frames_since_detection;

    float free_energy = 0.0f;
    TableObservation last_obs;
    bool updated = false;

    // Multi-sensor fusion (Q2b): one belief update per assigned slice (e.g. ZED + ricoh-360). Sequential
    // Bayesian updates = joint likelihood for the recursive filter, and each sensor keeps its OWN R and
    // common-mode (they don't share a registration error) — which concatenating points into one frame could
    // not express. The dense ZED slice and the sparse ricoh slice therefore both inform the SAME table.
    // ONLY on a fresh mask FRAME for this instance: the tracker assigns every cycle (birth/association
    // continuity), but re-fitting the SAME packet each cycle would overcount evidence — gate on a new frame_id.
    const auto& pkt = mask_ingestor_->packet();
    if (pkt.valid and pkt.frame_id > inst.last_masks_frame_seen)
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

    publish_table_cycle(inst, node, last_obs, free_energy);
}


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

bool SpecificWorker::assess_table_state(rc::TableInstance& inst, uint64_t node_id, float free_energy)
{
    auto node_opt = G->get_node(node_id);
    if (not node_opt.has_value())
        return false;

    step_convergence(inst, node_opt.value(), free_energy);
    return true;
}

void SpecificWorker::publish_table_diagnostics(const rc::TableInstance& inst,
                                               const TableObservation& observation,
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
            ts_cov_plot_->add_point (inst.node_name + "_cov", belief_uncertainty(inst));
        }
        if (ts_res_plot_)
        {
            ts_res_plot_->add_series(inst.node_name + "_res", QColor(170, 80, 255), 1.1f);
            // Plot inst.dbg_resid_pts (the SAME value the EvidenceMonitor counter shows), which HOLDS its last
            // value between masks — so the line matches the counter and doesn't vanish. The old code sampled
            // observation.residual_pts only on fresh frames, so on a non-fresh publish it added nothing → no line.
            ts_res_plot_->add_point(inst.node_name + "_res", static_cast<float>(inst.dbg_resid_pts));
        }
        if (ts_state_plot_)
        {
            // Sample EVERY cycle (not just fresh frames): the whole point is to see whether the accepted
            // dimensions hold flat between masks (stabiliser working) or drift/jitter on stale data.
            const auto& s = inst.model.state();
            ts_state_plot_->add_series(inst.node_name + "_w", QColor(255, 90, 90), 1.1f);
            ts_state_plot_->add_series(inst.node_name + "_h", QColor(90, 200, 90), 1.1f);
            ts_state_plot_->add_point (inst.node_name + "_w", s.w);
            ts_state_plot_->add_point (inst.node_name + "_h", s.h);
        }
        if (ts_ce_plot_ and inst.ai2_initialized)
        {
            // Belief posterior std for the size DOFs (Σ index 3=w, 4=h), in mm — the size-uncertainty trace.
            const auto& S = inst.ai2_belief.covariance();
            ts_ce_plot_->add_series(inst.node_name + "_sW", QColor(255, 90, 90), 1.1f);
            ts_ce_plot_->add_series(inst.node_name + "_sH", QColor(90, 200, 90), 1.1f);
            ts_ce_plot_->add_point (inst.node_name + "_sW", 1000.f * std::sqrt(std::max(0.f, S(3, 3))));
            ts_ce_plot_->add_point (inst.node_name + "_sH", 1000.f * std::sqrt(std::max(0.f, S(4, 4))));
        }
    }

    if (fitter_->should_log(inst))
        std::print("[{}] series: FE={:.4f} U(Σ)={:.3f} res={}\n",
                   inst.node_name,
                   free_energy,
                   belief_uncertainty(inst),
                   observation.residual_pts.size());
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

// ─── Initialisation helpers ──────────────────────────────────────────────────

void SpecificWorker::load_config(const ConfigLoader& cfg)
{
    cfg_ = rc::load_table_config(cfg);
}


// ─── Per-cycle steps ─────────────────────────────────────────────────────────


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
    const float model_uncertainty = belief_uncertainty(inst);
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
        return;
    rc::EpistemicProposal prop =
        epistemic_planner_.compute(inst.ai2_belief, cfg_.ai2_range_noise_lat_per_m, cfg_.ai2_sigma_base_m);
    if (not prop.valid or not prop.is_finite())
        return;   // degenerate (non-finite) fit — leave the existing affordance node as-is, retry next cycle

    // Belief→knowledge governor WITHOUT deleting the node: keep publishing the affordance every cycle
    // with its TRUE expected information gain ΔH (nats). A low gain is published as-is so the controller's
    // grounded EFE selection simply doesn't pick it (cost outweighs the small epistemic value), and it
    // re-arms automatically when the belief degrades and ΔH climbs again — no satisfy-latch to get stuck
    // in, no node churn. During the post-completion hold the gain is forced to 0 so a just-finished table
    // isn't re-claimed before its belief has settled.
    if (inst.epistemic_cooldown > 0)
        prop.epistemic_gain = 0.0f;

    // Write attributes to the table node (read by legacy consumers)
    scene_graph_->write_epistemic_proposal(node, prop);
    // Publish / refresh dedicated affordance node (persists; update_node refreshes target+gain)
    const auto affordance_node_before = inst.affordance.node_id();
    inst.affordance.update(prop);
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

void SpecificWorker::modify_node_attrs_slot(std::uint64_t id,
                                             const std::vector<std::string>& att_names)
{
    // Delegate to the affordance state machine for any instance whose affordance
    // node was modified (controller claim/completion updates active/pending)
    for (auto& [table_id, inst] : fitter_->instances())
        if (inst.affordance.node_id() == id)
            inst.affordance.on_node_modified(id);

    // React to mission-controller clearing epistemic_pending on the table node itself
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
    for (auto& [table_id, inst] : fitter_->instances())
        if (inst.affordance.node_id() == id)
            inst.affordance.on_node_deleted(id);

    if (fitter_->instances().count(id))
    {
        std::print("table_concept: node {} removed from DSR, destroying instance\n", id);
        fitter_->instances().erase(id);
    }
}

// ─── Lifecycle stubs ─────────────────────────────────────────────────────────

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




