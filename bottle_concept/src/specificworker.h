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
 * bottle_concept — Active Inference agent for bottle instance detection.
 *
 * Reads YOLO masks (the "masks" DSR node written by the voxelizer), selects the
 * slices labelled "bottle", and fits a vertical-cylinder generative model
 * (5-param state + cylinder SDF) by free-energy minimisation. The fitted pose
 * AND its Laplace-curvature covariance (P_bottle) are written on the room→bottle
 * RT edge, so the kinova_controller can read the bottle pose with an explicit
 * uncertainty that drives its closed→open-loop look-up rate.
 *
 * Bottle nodes use the DSR `cylinder` node type (geometrically exact: radius +
 * height) named "bottle_N". This is a focused port of table_concept: the mask
 * reading, RT machinery and FE loop are kept; tracks, the epistemic planner, the
 * table affordance, the warm-start policy, the Qt widgets and the presence
 * protocol are dropped for the MVP.
 */

#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#include <atomic>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <genericworker.h>
#include <Eigen/Dense>

#include <dsr/api/dsr_rt_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/api/dsr_inner_gaussian_api.h>   // Part B: chain covariance propagation

#include "../../common/agent_presence_coordinator/agent_presence_coordinator.h"
#include "bottle_instance.h"    // rc::BottleInstance
#include "epistemic_planner.h"  // rc::EpistemicPlanner (hidden-face next-best-view)
#include "bottle_config.h"       // rc::BottleConfig + rc::load_bottle_config
#include "bottle_evaluator.h"   // rc::BottleEvaluator (validation harness)
#include "../../common/mask_ingestor/mask_ingestor.h"  // rc::MaskIngestor (masks reading)
#include "bottle_scene_graph.h" // rc::BottleSceneGraph (DSR node/RT I/O)
#include "bottle_fitter.h"      // rc::BottleFitter (active-inference fit core)
#include "bottle_lidar_ingestor.h"  // rc::BottleLidarIngestor (lidar3D media plane → room-frame sweep)
#include "../../common/instance_tracker/instance_tracker.h"   // rc::InstanceTracker (birth/associate/death)
#include "../../common/dashboard/custom_widget.h"      // Custom_widget (dockable dashboard host)
#include "../../common/dashboard/timeseries_plot.h"    // rc::TimeSeriesPlot

// ─── (rc::BottleInstance / rc::BottleConfig moved to bottle_instance.h / bottle_config.h) ─


// ─── SpecificWorker ──────────────────────────────────────────────────────────

class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:
    SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check);
    ~SpecificWorker();

public slots:
    void initialize();
    void compute();
    void emergency();
    void restore();
    int  startup_check();

    void modify_node_slot(std::uint64_t, const std::string& type){};
    void modify_node_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names);
    void modify_edge_slot(std::uint64_t from, std::uint64_t to, const std::string& type){};
    void modify_edge_attrs_slot(std::uint64_t from, std::uint64_t to,
                                const std::string& type, const std::vector<std::string>& att_names){};
    void del_edge_slot(std::uint64_t from, std::uint64_t to, const std::string& edge_tag){};
    void del_node_slot(std::uint64_t from);

private:
    // (BottleObservation moved to bottle_fitter.h; MaskSlice/MasksPacket to mask_ingestor.h)

    // ── Presence protocol ──────────────────────────────────────────────────────
    void waiting_enter();
    void waiting_loop();
    void operating_enter();
    void operating_loop();
    void degraded_enter();
    void degraded_loop();
    void cleanup_owned_nodes();
    void remove_stale_affordance_nodes();   // sweep affordances parented to a cylinder (start + exit)
    void request_shutdown();
    // Terminal, crash-free exit: runs request_shutdown() then std::_Exit() to bypass the fragile
    // Ice communicator/static teardown that aborts (IceUtil::Mutex EINVAL). Single exit point for
    // both the confirmed-degraded path and SIGTERM/Ctrl-C (aboutToQuit).
    void terminal_shutdown();
    // Grace before a required-peer loss is treated as terminal: a transient presence flap (startup
    // handshake, brief node churn) must NOT kill the agent or destroy its graph state.
    static constexpr int REQUIRED_LOSS_GRACE_MS = 3000;
    void on_optional_peer_lost(const std::string& name, std::uint32_t id);
    void on_optional_peer_ready(const std::string& name, std::uint32_t id);
    // Delete every "bottle*" cylinder node this agent owns (startup sweep + teardown).
    void remove_owned_bottle_nodes();

    // Per-node orchestration (canonical concept-agent loop): the fitter runs the pure belief
    // (ensure_instance → observe → run_inference); the worker persists it via scene_graph_ and logs eval.
    void process_bottle_node(const DSR::Node& node);

    // Epistemic capability: publish/refresh the hidden-face affordance for this bottle. Keeps the node
    // alive and re-offered; a low ΔH is published as-is so the controller's EFE selection won't pick it.
    void step_epistemic(rc::BottleInstance& inst);

    // Live "Bottle Inference" dashboard: feed FE / dimensions / posterior σ / epistemic ΔH each cycle.
    void publish_bottle_diagnostics(rc::BottleInstance& inst, float free_energy);

    // Optional gated CSV of the epistemic/affordance evolution (debug/monitor). No-op unless
    // cfg_.epistemic_csv_path is set. Written from step_epistemic where ΔH + the viewpoint are fresh.
    void log_epistemic_csv(const rc::BottleInstance& inst,
                           const rc::EpistemicProposal& prop,
                           const Eigen::Vector2f& camera_xy);

    // ── Members ──────────────────────────────────────────────────────────────
    bool startup_check_flag = false;
    bool owned_nodes_cleaned_ = false;
    bool startup_affordance_sweep_done_ = false;   // the stale-affordance sweep is a one-time startup
                                                   // cleanup; re-running it on every Operating re-entry
                                                   // (after a transient Degraded flap) would wipe THIS
                                                   // run's live affordances → graph flicker.
    std::atomic<bool> shutting_down_{false};
    AgentPresenceCoordinator presence_coordinator_;

    rc::BottleConfig                                 cfg_;
    rc::EpistemicPlanner                             epistemic_planner_;   // hidden-face next-best-view

    // Live belief dashboard — its OWN top-level window (extracted from the DSR graph dock so it shows
    // independently of Agent.graph; mirrors room_concept/kinova_controller). Geometry persisted via QSettings.
    Custom_widget*      custom_widget_ = nullptr;
    rc::TimeSeriesPlot* ts_fe_plot_    = nullptr;   // free energy
    rc::TimeSeriesPlot* ts_dim_plot_   = nullptr;   // radius, height (m)
    rc::TimeSeriesPlot* ts_sigma_plot_ = nullptr;   // posterior σ(radius,height) (mm)
    rc::TimeSeriesPlot* ts_ce_plot_    = nullptr;   // CUSUM/SPRT counter-evidence (radius, height)
    void restore_dashboard_geometry();
    void save_dashboard_geometry() const;

    std::ofstream epistemic_csv_;   // optional per-cycle epistemic/affordance log (Epistemic.CsvPath)

    std::unique_ptr<DSR::RT_API> rt_api_;
    std::unique_ptr<DSR::InnerEigenAPI> inner_eigen_;
    std::unique_ptr<DSR::InnerGaussianAPI> gaussian_api_;   // Part B: source→target chain covariance
    uint64_t                     room_node_id_ = 0;

    // Collaborators (constructed in initialize(), after G + the DSR APIs are ready). Declared in
    // dependency order — the fitter holds raw pointers to the three above it, so it is destroyed first.
    std::unique_ptr<rc::MaskIngestor> mask_ingestor_;   // masks reading (owns the parsed MasksPacket)
    std::unique_ptr<rc::BottleSceneGraph> scene_graph_;  // DSR node/RT I/O (table lookup, birth, write-back)
    std::unique_ptr<rc::BottleEvaluator>  evaluator_;    // Webots-GT / sweep / eval CSV (no-op unless flagged)
    std::unique_ptr<rc::BottleFitter>     fitter_;       // active-inference fit core (owns the instance map)
    std::unique_ptr<rc::BottleLidarIngestor> lidar_ingestor_;  // lidar3D media plane → room-frame sweep (feeds fitter)

    // Multi-instance birth/associate/death (shared rc::InstanceTracker) — the only instance-lifecycle
    // path. Associates masks to instances (gated 1-to-1), spawns new bottles from unexplained masks,
    // retires unsupported ones.
    rc::InstanceTracker tracker_;
    void run_instance_tracker();   // called each cycle from compute()
    // Collapse two instances fitted to the SAME physical bottle (circle footprints overlap beyond
    // Tracker.MergeOverlap), keeping the more-observed one. Runs before associate/birth each cycle.
    void merge_overlapping_instances();
    // Retire an instance whose fit has explained no data (belief energy == 0) for cfg_.diverged_retire_frames
    // consecutive frames — a diverged/degenerate model (radius/centre runaway) that must not keep writing a
    // garbage node. Same deletion sequence as a tracker DEATH. No-op when the bound is 0.
    void retire_diverged_instances();

    int place_settle_ = 0;   // cycles waited after a start-placement move, before fitting (gate-lock guard)

signals:
    void presenceReady();
    void presenceLost();
};

#endif // SPECIFICWORKER_H
