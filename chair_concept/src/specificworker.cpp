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
 * Implements the Active Inference loop described in TABLE_CONCEPT.md §11.2:
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
#include <filesystem>
#include <print>
#include <cstdlib>   // std::_Exit — crash-free terminal shutdown
#include <thread>    // brief DDS flush before _Exit
#include <chrono>
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
            // Stale-node sweep on (re)entering Operating: remove any leftover affordance nodes from a
            // previous run (e.g. after a crash that skipped cleanup) so a fresh create doesn't collide
            // and get a DSR-generated name. Keyed on the parent object type, not the node name.
            remove_stale_affordance_nodes();
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
    scene_graph_ = std::make_unique<rc::ChairSceneGraph>(
        G, rt_api_.get(), cfg_, [this] { trigger_graph_layout_twopi(); });

    // Subscribe to graph signals
    connect(G.get(), &DSR::DSRGraph::update_node_signal,
            this, &SpecificWorker::modify_node_slot);
    connect(G.get(), &DSR::DSRGraph::update_node_attr_signal,
            this, &SpecificWorker::modify_node_attrs_slot);
    connect(G.get(), &DSR::DSRGraph::del_node_signal,
            this, &SpecificWorker::del_node_slot);

    // Remove any "chair*" nodes left behind by a previous (crashed) run so this agent always starts
    // from a clean slate and never adopts a stale/drifted node (mirrors bottle's startup sweep;
    // scaffold_missing_chair_nodes re-creates them from priors).
    remove_owned_chair_nodes();

    // Resolve room node
    const auto rooms = G->get_nodes_by_type("room");
    if (not rooms.empty())
        room_node_id_ = rooms.front().id();
    else
        qWarning() << "chair_concept: no room node found at startup";

    // Setup prior store
    prior_store_ = std::make_unique<rc::PriorStore>(priors_path_);
    priors_cache_ = prior_store_->load_priors();

    // Active-inference fit core. Owns the instance map; collaborates with the ingestor + scene graph.
    fitter_ = std::make_unique<rc::ChairFitter>(
        G, inner_eigen_.get(), cfg_, priors_cache_, mask_ingestor_.get(), scene_graph_.get());

    // Part B: localization/chain covariance on the published RT edge (mirrors bottle/table).
    gaussian_api_ = std::make_unique<DSR::InnerGaussianAPI>(G.get());
    fitter_->set_chain_cov_source(gaussian_api_.get(), "zed", cfg_.rt_cov_add_chain);

    // Missing chair nodes are scaffolded lazily from priors only after masks
    // provide some chair evidence in the current scene.

    // Build rc::EpistemicPlanner with configured parameters
    epistemic_planner_ = rc::EpistemicPlanner(cfg_.delta_min, cfg_.gain_threshold, cfg_.obs_distance,
                                              cfg_.epistemic_use_info_gain, cfg_.epistemic_min_info_gain);

    // Stale affordance nodes are swept on entering Operating (presence hook) and on shutdown — see
    // remove_stale_affordance_nodes(), keyed on the parent object type (robust to node-name renames).

    // ── Time-series widget ──────────────────────────────────────────────────
    if (not graph_viewers.empty())
    {
        custom_widget_ = new Custom_widget("Chair Model — Free Energy, Coverage Deficit, Residuals & Dimensions (w,h)");
        // Dock into the main DSR graph window (tab alongside the graph/tree views). The TimeSeriesPlot
        // is a plain QWidget (no QOpenGLWidget backing store), so it doesn't hit the shared-backing-store
        // repaint crash that pushed the heavy Voxel3D GL view into its own window — docking it here is
        // safe and keeps the dashboard together with the graph it annotates.
        graph_viewers.at("")->add_custom_widget_to_dock("Chair Inference", custom_widget_);

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

        // (D) Counter-evidence accumulator S_w/S_h — only when the gate is enabled. Watch S charge on a
        // surprise and decay back (glitch absorbed) vs climb to the barrier (a real change unlocks).
        if (cfg_.counter_evidence_gate)
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

        // GenericWorker::initialize() scheduled restore_window_settings() BEFORE this custom dock was
        // added, and add_custom_widget_to_dock() force-tabifies/raises it into a default slot. Re-restore
        // the saved layout now that the dock exists — deferred so it runs after the dock is fully realised
        // (and after the base's restore) — so the user's saved relative position of the "Chair Inference"
        // dock actually sticks. The dock carries an objectName, so QMainWindow::restoreState keys on it.
        QTimer::singleShot(0, this, [this]() { restore_window_settings(); });
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

    mask_ingestor_->refresh();
    if (cfg_.tracker_enabled)
        run_instance_tracker();   // data-driven birth/associate/death + merge
    else
        scene_graph_->scaffold_missing_chair_nodes(priors_cache_, mask_ingestor_->packet(), room_node_id_);

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
// death is OFF by default (removed only by MERGE); birth_min_sep is wide. Replaces the prior-scaffold +
// greedy nearest-mask when cfg_.tracker_enabled.
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

    // Tracks ← live instances: centre from the fit, XY cov from the stabiliser posterior precision
    // (fisher_info_raw[0]=cx, [1]=cy in the 8-DOF [cx,cy,cz,yaw,seat_w,seat_d,seat_h,back_h] layout).
    std::vector<rc::TrackView> tracks;
    tracks.reserve(fitter_->instances().size());
    for (auto& [id, inst] : fitter_->instances())
    {
        rc::TrackView t;
        t.id = id;
        const auto& s = inst.model.state();
        t.xy = {s.cx, s.cy};
        const float rx = inst.stab.fisher_info_raw[0], ry = inst.stab.fisher_info_raw[1];
        if (rx > 1e-6f and ry > 1e-6f)
        {
            t.cov = Eigen::Matrix2f::Zero();
            t.cov(0, 0) = 1.0f / rx;
            t.cov(1, 1) = 1.0f / ry;
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

    // ASSOCIATE: route each matched detection's mask slice to its instance (read in observe()).
    for (int d = 0; d < static_cast<int>(dets.size()); ++d)
        if (res.assignment[d] >= 0)
        {
            const std::uint64_t id = tracks[res.assignment[d]].id;
            if (auto it = fitter_->instances().find(id); it != fitter_->instances().end())
                it->second.assigned_mask_idx = dets[d].slice_index;
        }

    // STILLBIRTH PRUNE: a phantom born from association churn (it stole a detection the real chair then
    // re-won) starves once the real instance reclaims its mask. A young instance (still inside its probation
    // window) that the tracker leaves unassigned for a sustained streak is such a phantom. A real chair
    // locks on and matches essentially every cycle → its streak resets and never reaches patience → it
    // survives probation and becomes permanent furniture (immune, like DeathEnabled=false). The streak is
    // updated for ALL instances here so a mature chair's history stays correct even though it can't be pruned.
    if (cfg_.tracker_prune_enabled)
    {
        std::vector<std::uint64_t> stillborn;
        for (auto& [id, inst] : fitter_->instances())
        {
            if (inst.assigned_mask_idx >= 0) { inst.unassigned_streak = 0; continue; }
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
                it->second.affordance.remove();
            fitter_->forget_node(id);
            G->delete_node(id);
        }
    }

    // BIRTH: spawn an instance from each promoted (persistently-unexplained) detection, seeding the
    // fitter with the detection XY so the model starts AT the chair (not the 0,0 RT-read default).
    for (const int d : res.births)
    {
        const Eigen::Vector3f& c = pkt.slices[dets[d].slice_index].centroid;
        const auto new_id = scene_graph_->create_instance_from_detection(c, room_node_id_);
        if (new_id != 0)
            fitter_->note_birth(new_id, Eigen::Vector2f(c.x(), c.y()));
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
    inst.prev_free_energy = free_energy;
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
            ts_cov_plot_->add_point (inst.node_name + "_cov", inst.last_coverage_deficit);
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
        if (ts_ce_plot_)
        {
            // (D) Counter-evidence accumulator for the size DOFs (index 4=seat_w, 5=seat_d). Held between
            // fresh frames (only updated in accept on a measurement), so it reads as a charge/decay trace.
            ts_ce_plot_->add_series(inst.node_name + "_ceW", QColor(255, 90, 90), 1.1f);
            ts_ce_plot_->add_series(inst.node_name + "_ceH", QColor(90, 200, 90), 1.1f);
            ts_ce_plot_->add_point (inst.node_name + "_ceW", inst.stab.counter_evidence[4]);
            ts_ce_plot_->add_point (inst.node_name + "_ceH", inst.stab.counter_evidence[5]);
        }
    }

    if (fitter_->should_log(inst))
        std::print("[{}] series: FE={:.4f} cov={:.1f} res={}\n",
                   inst.node_name,
                   free_energy,
                   inst.last_coverage_deficit,
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

    if (observation.has_fresh_data)
    {
        auto node_opt = G->get_node(node_id);
        if (node_opt.has_value())
            step_refresh_check(inst, node_opt.value(), free_energy, observation.explanation_ratio);
    }
}

// ─── Initialisation helpers ──────────────────────────────────────────────────

void SpecificWorker::load_config(const ConfigLoader& cfg)
{
    cfg_ = rc::load_chair_config(cfg);
    priors_path_ = cfg_.priors_path;           // mirrored into the existing members for now
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

    // Check all four vertical faces are covered (for affordance/uncertainty reporting only)
    const auto coverage = inst.queue.face_coverage(inst.model);

    // Compute per-face coverage deficit for model_uncertainty_att
    float total_deficit = 0.0f;
    for (int i = 0; i < 4; ++i)
        total_deficit += std::max(0.0f, cfg_.delta_min - coverage[i]);
    inst.last_coverage_deficit = total_deficit;   // expose to plot
    G->add_or_modify_attrib_local<model_uncertainty_att>(node, total_deficit);

    // ↑ Top-down: generative model prediction vs. coverage evidence
    if (fitter_->should_log(inst))
        std::print("[{}] coverage: +x={:.1f} -x={:.1f} +y={:.1f} -y={:.1f}  "
                   "stable={}/{} U={:.1f}\n",
                   inst.node_name,
                   coverage[0], coverage[1], coverage[2], coverage[3],
                   inst.frames_converged, cfg_.K_stable,
                   total_deficit);

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

    auto prop = epistemic_planner_.compute(inst.model, inst.queue, inst.stab.fisher_info_raw);
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

    // Write attributes to the chair node (read by legacy consumers)
    scene_graph_->write_epistemic_proposal(node, prop);
    // Publish / refresh dedicated affordance node (persists; update_node refreshes target+gain)
    const auto affordance_node_before = inst.affordance.node_id();
    inst.affordance.update(prop);
    if (affordance_node_before == 0 and inst.affordance.node_id() != 0)
        trigger_graph_layout_twopi();
    inst.epistemic_pending = true;
}

void SpecificWorker::step_refresh_check(rc::ChairInstance& inst,
                                         DSR::Node& node,
                                         float free_energy,
                                         float explanation_ratio)
{
    if (free_energy > inst.prev_free_energy)
        ++inst.frames_rising;
    else
        inst.frames_rising = 0;

    if (inst.frames_rising >= cfg_.M_diverge and explanation_ratio < cfg_.explanation_ratio_thresh)
    {
        G->add_or_modify_attrib_local<request_full_sample_att>(node, true);
        G->update_node(node);
        inst.frames_rising   = 0;
        inst.queue.clear();
        inst.matched_frames  = 0;
        std::print("chair_concept: divergence detected for '{}' — requesting full resample\n",
                   inst.node_name);
    }
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




