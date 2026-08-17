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
 * Reads YOLO masks (the "masks" DSR node written by the retina), selects the
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
#include "../../common/exclusion/exclusion.h"   // rc::exclusion::Claim (SHARED)
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <genericworker.h>
#include <fps/fps.h>
#include <Eigen/Dense>

#include <dsr/api/dsr_rt_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/api/dsr_inner_gaussian_api.h>   // Part B: chain covariance propagation

#include "../../common/agent_presence_coordinator/agent_presence_coordinator.h"
#include "../../common/concept_presence/concept_presence.h"   // rc::presence::ConceptProtocol (SHARED)
#include "bottle_instance.h"    // rc::BottleInstance
#include "epistemic_planner.h"  // rc::EpistemicPlanner (hidden-face next-best-view)
#include "bottle_config.h"       // rc::BottleConfig + rc::load_bottle_config
#include "bottle_evaluator.h"   // rc::BottleEvaluator (validation harness)
#include "../../common/mask_ingestor/mask_ingestor.h"  // rc::MaskIngestor (masks reading)
#include "bottle_scene_graph.h" // rc::BottleSceneGraph (DSR node/RT I/O)
#include "bottle_fitter.h"
#include "bottle_existence.h"
#include "../../common/phantom_log/phantom_log.h"   // rc::history::PhantomLog (shadow-mode birth/death record)      // rc::BottleFitter (active-inference fit core)
#include "../../common/phantom_log/observer_pose.h"   // rc::history::note_observer (SHARED)
#include "../../common/agent_exit/terminal_exit.h"   // rc::agent::terminal_exit (SHARED)
#include "../../common/lidar_ingestor/concept_lidar_ingestor.h"  // rc::ConceptLidarIngestor (lidar3D media plane → room-frame sweep)
#include "../../common/instance_tracker/instance_tracker.h"   // rc::InstanceTracker (birth/associate/death)
#include "../../common/dashboard/belief_inspector.h"
#include "../../common/dashboard/belief_strip.h"   // rc::BeliefInspector (full-belief bottom panel)
#include "../../common/dashboard/evidence_monitor.h"   // rc::EvidenceMonitor (counter strip)
#include "../../common/dashboard/custom_widget.h"      // Custom_widget (dockable dashboard host)
#include "../../common/dashboard/timeseries_plot.h"    // rc::TimeSeriesPlot

// ─── (rc::BottleInstance / rc::BottleConfig moved to bottle_instance.h / bottle_config.h) ─


// ─── SpecificWorker ──────────────────────────────────────────────────────────

class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:

    // Other concepts' standing claims on room space (SHARED, common/exclusion). Refreshed ONCE
    // per compute() cycle — one graph walk feeding both the birth filter and the existence
    // occupancy discount, so the two can never disagree about who is where.
    std::vector<rc::exclusion::Claim> foreign_claims_;
    SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check);
    ~SpecificWorker();

public slots:
    void initialize();
    void compute();
    // SHADOW-MODE birth/death recorder (CONCEPT_AGENT_LIFECYCLE.md §4.2). Records ONLY — it can never
    // alter a birth or a removal. Attribution fields captured at death separate a genuine classifier
    // phantom from one of our own removal defects.
    void log_phantom_event(std::string_view event, std::uint64_t id, std::string_view name,
                           float x, float y, const rc::BottleInstance* inst, std::string_view note);

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
    // Last mask frame_id that CONTRIBUTED birth evidence. Agents feed the tracker every compute cycle on
    // purpose (a candidate with no matching detection expires), but persisting a candidate and accruing
    // evidence into it are different things — conflating them made birth_frames count COMPUTE CYCLES. See
    // common/instance_tracker/birth_evidence.h rule 1.
    long  last_birth_mask_frame_ = -1;

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
    void on_optional_peer_lost(const std::string& name, std::uint32_t id);
    void on_optional_peer_ready(const std::string& name, std::uint32_t id);
    // Delete every "bottle*" cylinder node this agent owns (startup sweep + teardown).
    void remove_owned_bottle_nodes();

    // ── Primary-input (masks) stream gate — mirrors table_concept / room_concept's LiDAR gate ──
    // Admission probe (Waiting→Operating gate): the `masks` node is present and advertising a frame id.
    bool masks_stream_ready(std::string *detail = nullptr) const;
    // Operating stall predicate: no NEW masks frame for cfg_.masks_stall_timeout_ms, with a cold-start grace
    // measured from presence_protocol_.operating_since_ms() before the first frame arrives.
    // false when the gate is disabled.
    bool masks_stream_stalled(std::int64_t *age_ms_out = nullptr) const;
    // Admission predicate: the producer is CURRENTLY publishing fresh frames (a frame within the timeout
    // window). Distinct from masks_stream_ready() (node-exists, which persists after the producer dies) —
    // admitting on node-exists causes an instant re-stall flap. Requires refresh() to be pumped while Waiting.
    bool masks_stream_live() const;

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
    std::atomic<bool> shutting_down_{false};
    // The presence protocol AND the gate state it owns (operating_since_ms / stall_reported /
    // degraded_from_input / first_operating_done) — SHARED, common/concept_presence. The transitions set
    // those, so they belong with the transitions; masks_stream_stalled() reads the baseline back.
    rc::presence::ConceptProtocol presence_protocol_;
    AgentPresenceCoordinator presence_coordinator_;

    // Primary-input stream-gate bookkeeping (mirrors table_concept / room_concept). All main-thread (FSM hooks).

    rc::BottleConfig                                 cfg_;
    rc::EpistemicPlanner                             epistemic_planner_;   // hidden-face next-best-view

    // Live belief dashboard — its OWN top-level window (extracted from the DSR graph dock so it shows
    // independently of Agent.graph; mirrors room_concept/kinova_controller). Geometry persisted via QSettings.
    Custom_widget*      custom_widget_ = nullptr;
    rc::TimeSeriesPlot* ts_plot_          = nullptr;   // FE (+ baseline)
    rc::TimeSeriesPlot* ts_surprise_plot_ = nullptr;   // FE surprise (attention signal), own panel/scale
    rc::TimeSeriesPlot* ts_cov_plot_      = nullptr;   // belief uncertainty U(Σ)
    rc::TimeSeriesPlot* ts_res_plot_      = nullptr;   // residual point count
    // Bottom panel (replaces the old Σ[cx,cy] time-series): the WHOLE belief — every state DOF with its
    // posterior σ and Σ as a correlation heatmap. The bottle publishes no σ*, so the inspector drops the
    // σ*/adequacy columns rather than show invented targets; a cylinder has no discrete modes either.
    rc::BeliefInspector* belief_inspector_ = nullptr;
    // Compact belief strip: ONE ROW PER INSTANCE, each row a 60 s trace. Separate small top-level window —
    // the standing display, with the big dashboard as the drill-down. Mirrors chair/table/cabinet/door.
    QWidget*         strip_window_ = nullptr;
    rc::BeliefStrip* belief_strip_ = nullptr;
    void refresh_belief_strip();
    void restore_strip_geometry();
    void save_strip_geometry() const;
    void refresh_belief_inspector();
    // Section 1: the evidence-pipeline counter strip (same struct + widget as every other concept agent).
    QWidget*             dashboard_window_ = nullptr;   // combined window: counters over plots + inspector
    rc::EvidenceMonitor* evidence_monitor_ = nullptr;
    rc::EvidenceGlobals  ev_g_{};                       // per-cycle fields reset at the head of compute()
    void refresh_evidence_monitor();                    // throttled push of BOTH dashboard sections
    std::chrono::steady_clock::time_point last_monitor_tp_{};   // ~5 Hz dashboard tick
    std::chrono::steady_clock::time_point last_compute_tp_{};   // compute-rate estimate for the strip
    FPSCounter                            fps_counter_;         // overall compute()-cycle rate (std::cout heartbeat)
    void restore_dashboard_geometry();
    void save_dashboard_geometry() const;

    std::ofstream epistemic_csv_;   // optional per-cycle epistemic/affordance log (Epistemic.CsvPath)

    std::unique_ptr<DSR::RT_API> rt_api_;
    std::unique_ptr<DSR::InnerEigenAPI> inner_eigen_;
    std::unique_ptr<DSR::InnerGaussianAPI> gaussian_api_;   // Part B: source→target chain covariance
    uint64_t                     room_node_id_ = 0;
    // Room delimiting polygon, from room_concept's `delimiting_polygon_x/y`. Used ONLY for the NBV's
    // reachability test: a bottle's far-side viewpoint is the one that lands outside the room, and without
    // this rc::nbv::is_reachable imposes no constraint (it refuses to guess on an empty polygon).
    std::vector<Eigen::Vector2f> room_polygon_;
    void refresh_room_polygon();

    // Collaborators (constructed in initialize(), after G + the DSR APIs are ready). Declared in
    // dependency order — the fitter holds raw pointers to the three above it, so it is destroyed first.
    std::unique_ptr<rc::MaskIngestor> mask_ingestor_;   // masks reading (owns the parsed MasksPacket)
    std::unique_ptr<rc::BottleSceneGraph> scene_graph_;  // DSR node/RT I/O (table lookup, birth, write-back)
    std::unique_ptr<rc::BottleEvaluator>  evaluator_;    // Webots-GT / sweep / eval CSV (no-op unless flagged)
    std::unique_ptr<rc::BottleFitter>     fitter_;       // active-inference fit core (owns the instance map)
    rc::history::PhantomLog                             phantom_log_;   // shadow-mode birth/death record
    std::unique_ptr<rc::ConceptLidarIngestor> lidar_ingestor_;  // lidar3D media plane → room-frame sweep (feeds fitter)
    std::unique_ptr<rc::BottleExistence>  existence_;     // existence log-odds + removal (the shared rc::exist policy)

    // Multi-instance birth/associate/death (shared rc::InstanceTracker) — the only instance-lifecycle
    // path. Associates masks to instances (gated 1-to-1), spawns new bottles from unexplained masks,
    // retires unsupported ones.
    rc::InstanceTracker tracker_;
    void run_instance_tracker();   // called each cycle from compute()
    // Collapse two instances fitted to the SAME physical bottle (circle footprints overlap beyond
    // Tracker.MergeOverlap), keeping the more-observed one. Runs before associate/birth each cycle.
    void retire_instance(std::uint64_t id);   // shared teardown: affordance + fitter forget + graph delete
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
