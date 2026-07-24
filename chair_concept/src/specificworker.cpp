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

// Two chairs cannot share physical space. Footprint = the oriented seat rectangle (cx,cy,seat_w,seat_d,yaw)
// in the room plane; these helpers compute the overlap area so the merge operator can collapse duplicate
// instances. Corners CCW (local order (-,-),(+,-),(+,+),(-,+)). Mirrors table_concept.
std::array<Eigen::Vector2f, 4> footprint_corners(const rc::ChairState& s)
{
    const float c = std::cos(s.yaw), sn = std::sin(s.yaw);
    const Eigen::Vector2f ex(c, sn), ey(-sn, c), ctr(s.cx, s.cy);
    const float hw = 0.5f * s.seat_w, hh = 0.5f * s.seat_d;
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
float footprint_overlap_ratio(const rc::ChairState& a, const rc::ChairState& b)
{
    const auto ca = footprint_corners(a), cb = footprint_corners(b);
    const auto inter = clip_poly(std::vector<Eigen::Vector2f>(ca.begin(), ca.end()), cb);
    const float ai = poly_area(inter);
    const float amin = std::min(poly_area({ca.begin(), ca.end()}), poly_area({cb.begin(), cb.end()}));
    return amin > 1e-6f ? ai / amin : 0.0f;
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
    if (not custom_widget_)
        return;
    QSettings settings(QStringLiteral("RoboComp"), QStringLiteral("chair_concept"));
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
    QSettings settings(QStringLiteral("RoboComp"), QStringLiteral("chair_concept"));
    settings.setValue(QStringLiteral("DashboardWindow_geometry"), custom_widget_->saveGeometry());
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
        custom_widget_ = new Custom_widget("Chair Model — Free Energy, Coverage Deficit, Residuals & Dimensions (w,h)");
        custom_widget_->setWindowTitle(QStringLiteral("chair_concept — belief dashboard"));
        restore_dashboard_geometry();
        custom_widget_->show();

        // Create plot inside frame_series
        auto* series_layout = new QVBoxLayout(custom_widget_->frame_series);
        series_layout->setContentsMargins(0, 0, 0, 0);
        custom_widget_->frame_series->setLayout(series_layout);

        ts_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
        ts_plot_->set_visible_window(60.f);
        series_layout->addWidget(ts_plot_);

        ts_cov_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
        ts_cov_plot_->set_visible_window(60.f);
        series_layout->addWidget(ts_cov_plot_);

        ts_res_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
        ts_res_plot_->set_visible_window(60.f);
        series_layout->addWidget(ts_res_plot_);

        // Inferred chair dimensions (w, h) — the size DOFs the stabiliser targets. Watch these to
        // confirm the belief has stopped jittering between fresh masks (flat = stable).
        ts_state_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
        ts_state_plot_->set_visible_window(60.f);
        series_layout->addWidget(ts_state_plot_);

        // Belief size posterior std σ_w/σ_h (mm) — watch the size uncertainty shrink as views accumulate.
        ts_ce_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
        ts_ce_plot_->set_visible_window(60.f);
        series_layout->addWidget(ts_ce_plot_);

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
                ts_ce_plot_->add_series(inst.node_name + "_sW", QColor(255, 90, 90), 1.1f);
                ts_ce_plot_->add_series(inst.node_name + "_sH", QColor(90, 200, 90), 1.1f);
            }
        }
    }
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

    refresh_room_geometry();  // room-containment pose prior (cheap; the polygon is a nominal model)
    fitter_->update_ego_motion();   // robot/camera speed → "be-still-to-update" gate (once per cycle)
    mask_ingestor_->refresh();
    run_instance_tracker();   // data-driven birth/associate/death + merge (the only instance-lifecycle path)

    const auto chair_nodes = G->get_nodes_by_type("chair");
    for (const auto& node : chair_nodes)
        process_chair_node(node);

    // Overall compute()-cycle rate: counts every cycle, prints "Epoch time = …ms. Fps = N" once a
    // second (FPSCounter::print is throttled and self-counting).
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

            const float ratio = footprint_overlap_ratio(ia->second.model.state(), ib->second.model.state());
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
    if (pkt.valid)
        for (int i = 0; i < static_cast<int>(pkt.slices.size()); ++i)
            if (pkt.slices[i].label == "chair" and pkt.slices[i].support_end > pkt.slices[i].support_begin)
                dets.push_back({Eigen::Vector2f(pkt.slices[i].centroid.x(), pkt.slices[i].centroid.y()), i});

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
    if (not G or room_node_id_ == 0) return;
    const auto room = G->get_node(room_node_id_);
    if (not room.has_value()) return;
    const auto px = G->get_attrib_by_name<delimiting_polygon_x_att>(room.value());
    const auto py = G->get_attrib_by_name<delimiting_polygon_y_att>(room.value());
    if (not px.has_value() or not py.has_value()) return;
    const auto& xs = px->get(); const auto& ys = py->get();
    const std::size_t n = std::min(xs.size(), ys.size());
    if (n < 3) return;
    std::vector<Eigen::Vector2f> poly; poly.reserve(n);
    Eigen::Vector2f centroid = Eigen::Vector2f::Zero();
    for (std::size_t i = 0; i < n; ++i) { poly.emplace_back(xs[i], ys[i]); centroid += poly.back(); }
    centroid /= static_cast<float>(n);
    fitter_->set_room_geometry(centroid, std::move(poly));
}

// Continuous existence belief: fold one sensor frame of evidence into each instance's log-odds L and remove
// any whose L crosses the floor. See the ChairInstance::exist_logodds comment for the model. Runs from
// run_instance_tracker (association already resolved, so assigned_mask_idx is this cycle's assignment).
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

    std::vector<std::uint64_t> to_remove;
    for (auto& [id, inst] : fitter_->instances())
    {
        if (std::isnan(inst.exist_logodds))           // seed on first visit (fresh birth OR adopted graph node)
            inst.exist_logodds = cfg_.exist_birth_logodds;

        // ROOM-CONTAINMENT POSE PRIOR (runs BEFORE the frustum gate, so it reaches a chair a localization glitch
        // put OUTSIDE the walls / behind a wall where the sensor can never vacate it): P(chair outside) ≈ 0, so
        // an out-of-room centre draws a STRONG negative every frame regardless of visibility → removed in a few.
        if (cfg_.exist_room_prior and fitter_->has_room_polygon() and not inst.is_bearing_hypothesis)
        {
            const auto& ms = inst.model.state();
            if (not fitter_->point_in_room(Eigen::Vector2f(ms.cx, ms.cy), cfg_.exist_room_margin_m))
            {
                inst.exist_logodds = std::clamp(inst.exist_logodds - cfg_.exist_out_of_room_gain,
                                                cfg_.exist_remove_logodds - 1.0f, cfg_.exist_max_logodds);
                if (inst.exist_logodds < cfg_.exist_remove_logodds)
                    to_remove.push_back(id);
                continue;   // outside the room → no sensor evidence can rescue it; skip the normal channels
            }
        }

        // HOLD (no evidence) unless the instance is a depth chair in the camera frustum with a real belief.
        // roi_valid is one compute-cycle stale (fine for a frustum test); a bearing-only hypothesis carries no
        // depth (existence unjudgeable from support mass); an un-initialised newborn hasn't had a chance yet.
        if (inst.is_bearing_hypothesis or not inst.roi_valid or not inst.ai2_initialized)
            continue;

        // TWO evidence channels for whether a chair really occupies this in-frustum spot.
        float llr;
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
            // Be-still invariant: a MOVING frame only CONFIRMS existence — its smeared/high-clutter mask must not
            // count as NEGATIVE evidence. Winning the slice already reset frames_since_detection (no vacate); floor
            // the log-odds delta at 0 so a moving pass can hold/raise existence but never argue the chair away.
            if (fitter_->confirm_only(inst))
                llr = std::max(0.0f, llr);
        }
        else if (cfg_.exist_occlusion_check and fitter_->los_occluded(inst))
            // WON NOTHING but the line of sight is BLOCKED by a closer object (another chair, the table, a
            // person …): absence of a mask is EXPECTED, not evidence the chair is gone → HOLD, never vacate.
            continue;
        else
        {
            // WON NOTHING while in the frustum, UNOCCLUDED, on a live sensor frame → absence evidence, but its
            // CONFIDENCE RAMPS with how long the instance has gone unexplained (freshness-as-precision, NOT a
            // hard gate): a chair that just lost the 1-to-1 slice or is briefly hidden (small frames_since_
            // detection) is barely touched and recovers on its next win — this ramp prevents the death-spiral;
            // a spot left unexplained-and-in-clear-view for seconds (a real phantom) accrues the FULL negative.
            const float conf = (cfg_.exist_vacate_confident_frames > 0)
                ? std::clamp(static_cast<float>(inst.frames_since_detection)
                             / static_cast<float>(cfg_.exist_vacate_confident_frames), 0.0f, 1.0f)
                : 0.0f;
            llr = -g * conf;
        }

        inst.exist_logodds = std::clamp(inst.exist_logodds + llr,
                                        cfg_.exist_remove_logodds - 1.0f, cfg_.exist_max_logodds);

        if (inst.exist_logodds < cfg_.exist_remove_logodds)
            to_remove.push_back(id);
    }

    // Throttled existence readout so a "why is this phantom still here?" case is diagnosable from the log.
    static int ex_dbg = 0;
    if (++ex_dbg % 60 == 0)
        for (const auto& [id, inst] : fitter_->instances())
            if (not inst.is_bearing_hypothesis)
                std::print("chair_concept: [existence] {} L={:.2f} roi={} won={} since_det={} occluded={}\n",
                           inst.node_name, inst.exist_logodds, inst.roi_valid ? 1 : 0,
                           inst.assigned_mask_idx >= 0 ? 1 : 0, inst.frames_since_detection,
                           (inst.roi_valid and inst.assigned_mask_idx < 0 and fitter_->los_occluded(inst)) ? 1 : 0);

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
    const auto observation = fitter_->observe(inst, node);

    // Stale check: skip heavy update if data hasn't moved for too long
    if (not observation.has_fresh_data and inst.matched_frames < 5)
        return;

    const float free_energy = fitter_->run_inference(inst, observation);
    publish_chair_cycle(inst, node, observation, free_energy);
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
        if (ts_state_plot_)
        {
            // Sample EVERY cycle (not just fresh frames): the whole point is to see whether the accepted
            // dimensions hold flat between masks (stabiliser working) or drift/jitter on stale data.
            const auto& s = inst.model.state();
            ts_state_plot_->add_series(inst.node_name + "_w", QColor(255, 90, 90), 1.1f);
            ts_state_plot_->add_series(inst.node_name + "_h", QColor(90, 200, 90), 1.1f);
            ts_state_plot_->add_point (inst.node_name + "_w", s.seat_w);
            ts_state_plot_->add_point (inst.node_name + "_h", s.seat_d);
        }
        if (ts_ce_plot_ and inst.ai2_initialized)
        {
            // Pose posterior std (pose-only belief): position (Σ 0=cx, mm) and yaw (Σ 2, mrad).
            const auto& S = inst.ai2_belief.covariance();
            ts_ce_plot_->add_series(inst.node_name + "_pos", QColor(255, 90, 90), 1.1f);
            ts_ce_plot_->add_series(inst.node_name + "_yaw", QColor(90, 200, 90), 1.1f);
            ts_ce_plot_->add_point (inst.node_name + "_pos", 1000.f * std::sqrt(std::max(0.f, S(0, 0))));
            ts_ce_plot_->add_point (inst.node_name + "_yaw", 1000.f * std::sqrt(std::max(0.f, S(2, 2))));
        }
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
        epistemic_planner_.compute(inst.ai2_belief, cfg_.ai2_range_noise_lat_per_m, cfg_.ai2_sigma_base_m);
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
    if (type != "chair")
        return;

    const auto node_opt = G->get_node(id);
    if (not node_opt.has_value())
        return;

    fitter_->ensure_instance(node_opt.value(), room_node_id_);
}

void SpecificWorker::modify_node_attrs_slot(std::uint64_t id,
                                             const std::vector<std::string>& att_names)
{
    // Delegate to the affordance state machine for any instance whose affordance
    // node was modified (controller claim/completion updates active/pending)
    for (auto& [chair_id, inst] : fitter_->instances())
        if (inst.affordance.node_id() == id)
            inst.affordance.on_node_modified(id);

    // React to mission-controller clearing epistemic_pending on the chair node itself
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




