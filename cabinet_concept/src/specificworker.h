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
 * specificworker.h — cabinet_concept agent (orchestration + Qt/DSR glue).
 *
 * Per compute() cycle: ingest ZED YOLO "cabinet" masks → InstanceTracker (birth / associate / merge) →
 * process_cabinet_node (one AI2 CabinetBelief update per assigned slice) → publish the fit back to DSR (RT edge +
 * dims + mesh + covariance) and emit epistemic proposals when a cabinet stays under-observed. Ricoh-360
 * detections are bearing-only PERIPHERAL ATTENTION (they never birth or fit). Also owns the two standalone
 * top-level windows (belief dashboard + evidence monitor). The fit core is rc::CabinetFitter, perception
 * rc::MaskIngestor, DSR I/O rc::CabinetSceneGraph. See CABINET.md for the belief/fit core.
 */

#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#include <atomic>
#include "../../common/exclusion/exclusion.h"   // rc::exclusion::Claim (SHARED)
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <fstream>

#include <genericworker.h>
#include <fps/fps.h>
#include <Eigen/Dense>
#include <unordered_set>

#include "cabinet_config.h"      // rc::CabinetConfig + load_cabinet_config
#include "cabinet_instance.h"    // rc::CabinetInstance
#include "../../common/mask_ingestor/mask_ingestor.h"     // rc::MaskIngestor (perception)
#include "../../common/lidar_ingestor/concept_lidar_ingestor.h"                          // rc::ConceptLidarIngestor (YOLO-independent LiDAR)
#include "../../common/instance_tracker/instance_tracker.h"   // rc::InstanceTracker (birth/associate/death)
#include "cabinet_scene_graph.h" // rc::CabinetSceneGraph (DSR node/RT I/O)
#include "cabinet_fitter.h"
#include "../../common/phantom_log/phantom_log.h"   // rc::history::PhantomLog (shadow-mode birth/death record)      // rc::CabinetFitter (active-inference core)
#include "../../common/phantom_log/observer_pose.h"   // rc::history::note_observer (SHARED)
#include "../../common/agent_exit/terminal_exit.h"   // rc::agent::terminal_exit (SHARED)
#include "cabinet_kitchen.h"     // rc::KitchenManager (Stage 2 kitchen-of-runs model)
#include "cabinet_existence.h"   // rc::CabinetExistence (evidence-based removal)
#include "../../common/birth_surprise/birth_surprise_probe.h"   // rc::BirthSurpriseProbe (SHARED, read-only: residual grid → birth surprise)
#include "epistemic_planner.h"
#include "../../common/object_affordance/object_affordance.h"
#include "cabinet_model.h"
#include "../../common/dashboard/belief_inspector.h"
#include "../../common/dashboard/belief_strip.h"
#include "../../common/dashboard/custom_widget.h"
#include "../../common/dashboard/evidence_monitor.h"
#include "../../common/dashboard/timeseries_plot.h"
#include "../../common/agent_presence_coordinator/agent_presence_coordinator.h"
#include "../../common/concept_presence/concept_presence.h"   // rc::presence::ConceptProtocol (SHARED)
#include "../../common/epistemic_step/epistemic_step.h"   // rc::epistemic::step (SHARED)

// ─── SpecificWorker ──────────────────────────────────────────────────────────────────────────────

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
    bool is_shutting_down() const noexcept { return shutting_down_.load(); }

public slots:
    void initialize();
    void compute();
    // SHADOW-MODE birth/death recorder (CONCEPT_AGENT_LIFECYCLE.md §4.2). Records ONLY — it can never
    // alter a birth or a removal. The attribution fields it captures at death (p_detect, in-FoV, central)
    // are what separate a genuine classifier phantom from one of our own removal defects.
    void log_phantom_event(std::string_view event, std::uint64_t id, std::string_view name,
                           float x, float y, const rc::CabinetInstance* inst, std::string_view note);

    void emergency();
    void restore();
    int  startup_check();

    void modify_node_slot(std::uint64_t id, const std::string& type);
    // Replaces the update_node_attr_signal slot: the controller-owned protocol flags are POLLED once per
    // cycle instead of pushed per graph attribute write. See the connect block in initialize().
    void poll_affordance_protocol();
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

    // The fit (observe/infer/belief/support-bank) lives in rc::CabinetFitter; perception in
    // rc::MaskIngestor; DSR I/O in rc::CabinetSceneGraph. The worker keeps orchestration + the
    // post-fit epistemic/affordance/Qt-diagnostics steps.
    using CabinetObservation = rc::CabinetFitter::CabinetObservation;

    // ── Orchestration + post-fit steps ────────────────────────────────────────
    void load_config(const ConfigLoader& cfg);
    void process_cabinet_node(const DSR::Node& node);
    void publish_cabinet_cycle(rc::CabinetInstance& inst,
                             const DSR::Node& node,
                             const CabinetObservation& observation,
                             float free_energy);
    bool assess_cabinet_state(rc::CabinetInstance& inst, uint64_t node_id, float free_energy);
    void publish_cabinet_diagnostics(const rc::CabinetInstance& inst,
                                   const CabinetObservation& observation,
                                   float free_energy);
    void publish_cabinet_intentions(rc::CabinetInstance& inst,
                                  uint64_t node_id,
                                  const CabinetObservation& observation,
                                  float free_energy);
    void step_convergence(rc::CabinetInstance& inst, DSR::Node& node, float free_energy);
    void step_epistemic(rc::CabinetInstance& inst, DSR::Node& node);
    void trigger_graph_layout_twopi();   // injected into CabinetSceneGraph as the relayout callback

    // Multi-instance birth/associate/merge (shared rc::InstanceTracker; the only instance-lifecycle path).
    // Associates "cabinet" masks to instances, spawns a cabinet from an unexplained mask, merges overlaps.
    rc::InstanceTracker tracker_;
    void run_instance_tracker();   // called every cycle from compute()
    void shadow_route_kitchen_cells();   // kitchen-model Stage 0 SHADOW: route masks → (wall,tier) cells, log only
    // Stage 2: the kitchen-of-runs model. Cells own the WallRunBeliefs; route masks → per-cell fit → existence →
    // publish derived room-frame boxes. Replaces run_instance_tracker + process_cabinet_node when cfg_.kitchen_model.
    void run_kitchen_model();
    // Read the level-2 END PRIOR off each run's group_member edge and hand it to that run's belief,
    // projected from room-frame targets onto the run's own chart. Cleared first, so a frame that
    // goes quiet stops steering rather than leaving its last word in place.
    void apply_arrangement_end_priors();
    void publish_kitchen_boxes();        // reconcile active cells ↔ DSR cabinet_* box nodes (create/update/delete)
    // Publish a run's room-frame pose covariance on its room→run RT edge. The kitchen path used to write
    // pose with NO covariance (the classic path always wrote one), so every kitchen run advertised itself as
    // equally certain — which silently defeats any consumer that weights by the producer's own Σ.
    void write_kitchen_rt_covariance(std::uint64_t room_id, std::uint64_t node_id,
                                     const rc::KitchenBox& box, std::uint64_t stamp_ms);
    // mask_lidar_dr: median (mask-point range − nearest LiDAR range) in the same direction. Positive
    // ⇒ the masks de-project BEYOND the surface the LiDAR sees, i.e. points land in mid-air.
    void log_kitchen_cells(std::size_t sweep_n, float mask_lidar_dr = 0.0f, int mask_lidar_n = 0);   // per-cycle cell CSV (cfg_.kitchen_cells_csv_path; empty = off)
    void update_kitchen_ego_motion();    // transform-chain camera speed (producer-independent), aligned with chair
    std::vector<rc::SceneObjectBox> read_scene_objects() const;   // OTHER agents' furniture boxes (room frame) to not penetrate
    rc::KitchenManager                        kitchen_mgr_;
    std::unordered_map<std::string, std::uint64_t> kitchen_nodes_;   // cell signature → DSR node id
    // Last published covariance trace per cell — the RT-cov write self-gates on it, so a settled run does
    // not rewrite its edge every cycle (same discipline as CabinetSceneGraph::write_rt_covariance).
    std::unordered_map<std::string, float>         kitchen_cov_trace_;
    // Ego-motion state for the stillness/VOR gate (chair-aligned): camera pose deltas → linear/angular speed.
    Eigen::Vector3f  prev_cam_pos_{0.0f, 0.0f, 0.0f};
    Eigen::Vector3f  prev_cam_fwd_{0.0f, 1.0f, 0.0f};
    std::chrono::steady_clock::time_point prev_cam_tp_{};
    bool             have_prev_cam_ = false;
    float            ego_lin_mps_   = 0.0f;   // camera linear speed (m/s)
    float            ego_ang_radps_ = 0.0f;   // camera angular speed (rad/s), from the forward-axis rotation
    // Read the room's delimiting polygon + interior centroid and push them to the fitter (wall-flush
    // factor + the C2v yaw canonicalize reference). Cheap; called each cycle once the room is known.
    void refresh_room_geometry();
    void retire_instance(std::uint64_t id);   // shared teardown: affordance + fitter forget + graph delete
    // Physical-exclusion invariant: two cabinets cannot share space. Collapse any pair of instances whose
    // oriented footprints overlap beyond Tracker.MergeOverlap, keeping the more-observed one.
    void merge_overlapping_instances();
    // Residual-driven birth: cluster the pooled model-unexplained points; a coherent, separated, elongated
    // arm no believed run covers matures over residual_birth_frames cycles into its own axis-aligned
    // "cabinet_N", pre-seeded from the arm. Called each cycle after the fits (residuals are current then).
    void birth_from_residual();
    // A cabinet RUN cannot be L-shaped: if one YOLO-sem 'cabinet' mask wraps a corner it is TWO runs. Split
    // each such mask into its two perpendicular arms IN PLACE (mask packet), so the tracker sees two clean
    // single-arm detections and the ordinary single-run machinery births/fits each. Called after refresh(),
    // before the tracker. See cabinet_lshape_split.h.
    void split_lshaped_cabinet_masks();

    // ── Presence protocol ────────────────────────────────────────────────────
    void waiting_enter();
    void waiting_loop();
    void operating_enter();
    void operating_loop();
    void degraded_enter();
    void degraded_loop();
    void cleanup_owned_nodes();
    void remove_stale_affordance_nodes();   // sweep affordances parented to a cabinet (start + exit)
    void remove_owned_cabinet_nodes();        // startup stale-sweep of "cabinet*" nodes (mirrors bottle)
    void request_shutdown();
    // Crash-free exit (request_shutdown + DDS reset + _Exit), bypassing the Ice/static teardown abort.
    void terminal_shutdown();
    // Grace before a required-peer loss is treated as terminal (debounce transient presence flaps).
    void on_optional_peer_lost(const std::string &name, std::uint32_t id);
    void on_optional_peer_ready(const std::string &name, std::uint32_t id);

    // ── Primary-input (masks) stream gate — mirrors room_concept's LiDAR gate (see CONCEPT_AGENT_RECIPE.md) ──
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

    // ── Members ──────────────────────────────────────────────────────────────
    bool startup_check_flag = false;
    bool owned_nodes_cleaned_ = false;
    std::atomic<bool> shutting_down_{false};
    // The presence protocol AND the gate state it owns (operating_since_ms / stall_reported /
    // degraded_from_input / first_operating_done) — SHARED, common/concept_presence. The transitions set
    // those, so they belong with the transitions; masks_stream_stalled() reads the baseline back.
    rc::presence::ConceptProtocol presence_protocol_;
    AgentPresenceCoordinator presence_coordinator_;

    // Primary-input stream-gate bookkeeping (mirrors room_concept). All main-thread (FSM hooks).

    rc::CabinetConfig                                         cfg_;
    rc::EpistemicPlanner                                    epistemic_planner_;
    std::unique_ptr<rc::CabinetFitter>                        fitter_;    // active-inference fit core (owns instances)
    rc::history::PhantomLog                             phantom_log_;   // shadow-mode birth/death record
    rc::KitchenRouting                                        kitchen_routing_;   // Stage 0 shadow cell table
    std::unique_ptr<rc::CabinetExistence>                    existence_; // evidence-based removal (existence log-odds)

    // Live belief dashboard + evidence monitor, MERGED into one top-level window (evidence monitor on top,
    // belief plots below, in a vertical splitter — mirrors table_concept). Extracted from the DSR graph dock so
    // it shows independently of Agent.graph. Geometry persisted via QSettings on dashboard_window_.
    QWidget*             dashboard_window_ = nullptr;   // combined window owning the splitter
    Custom_widget*       custom_widget_ = nullptr;
    rc::TimeSeriesPlot*  ts_plot_       = nullptr;   // FE (+ baseline)
    rc::TimeSeriesPlot*  ts_surprise_plot_ = nullptr;   // FE surprise (attention signal), own panel/scale
    std::unordered_set<std::string> ts_known_cabinets_;   // node_names with live timeseries series (for pruning)
    void prune_dead_series();   // drop timeseries series for cabinets removed from the graph (periodic, in compute)

    // ── Ricoh 360 as PERIPHERAL ATTENTION (bearing-only) ──────────────────────────────────────────────────
    // A ricoh detection has a reliable DIRECTION but a biased centroid/extent, so it never births/fits (that
    // caused duplicates + drift). Instead an UNASSIGNED ricoh bearing (no known cabinet lies along it) becomes an
    // attention target: "seek a ZED view in this direction to birth/confirm the cabinet" (peripheral→saccade→fovea).
    struct RicohBearingTarget { float bearing_rad = 0.0f; float range_m = 0.0f; float confidence = 0.0f;
                                Eigen::Vector2f xy = Eigen::Vector2f::Zero(); };
    std::vector<RicohBearingTarget> ricoh_attention_targets_;   // unassigned ricoh bearings this cycle
    void process_ricoh_bearings();   // associate ricoh detections to cabinets BY DIRECTION; collect the unassigned
    rc::TimeSeriesPlot*  ts_cov_plot_   = nullptr;   // belief uncertainty U(Σ) = Σ pos+size posterior std (m)
    rc::TimeSeriesPlot*  ts_res_plot_   = nullptr;   // residual point count
    // Bottom panel (replaces the old σ_w/σ_h time-series): the WHOLE belief — every state DOF with its
    // posterior σ, the consumer's demand σ* and the remaining adequacy gap, Σ as a correlation heatmap,
    // and the discrete-tier posterior. Serves BOTH cabinet models (7-DOF box / 5-DOF wall run).
    rc::BeliefInspector* belief_inspector_ = nullptr;
    // ── Compact belief strip: ONE ROW PER INSTANCE, and the row is a 60 s time series ──────────────
    // A separate small top-level window, deliberately not another panel inside the big dashboard: this is
    // the display you keep in a corner while the dashboard stays closed until something looks wrong.
    // Mirrors table_concept / door_concept. See common/dashboard/belief_strip.h.
    QWidget*         strip_window_ = nullptr;
    rc::BeliefStrip* belief_strip_ = nullptr;
    void refresh_belief_strip();
    void restore_strip_geometry();
    void save_strip_geometry() const;
    void refresh_belief_inspector();
    void build_dashboard();          // create the dashboard + evidence-monitor windows (called from initialize)
    void restore_dashboard_geometry();
    void save_dashboard_geometry() const;

    // Live "evidence consuming" monitor — its OWN top-level window (per-instance snapshot + global counters).
    rc::EvidenceMonitor* evidence_monitor_ = nullptr;
    rc::EvidenceGlobals  ev_g_{};                       // pipeline counters (per-cycle fields reset in compute)
    void refresh_evidence_monitor();                    // throttled build+push of the snapshot (main thread)
    bool read_residual_field();                         // snapshot residual_concept's `grid` node → residual_field_
    void log_birth_surprise();                          // EXPERIMENTAL read-only probe (cfg_.birth_surprise_probe)
    rc::GridField        residual_field_;               // this cycle's residual (surprise) field; read at compute head
    std::ofstream        birth_surprise_csv_;           // etc/birth_surprise.csv (opened lazily when the probe is on)
    std::ofstream        birth_fusion_csv_;             // etc/birth_fusion.csv (detection-conditioned residual mass)
    int                  birth_surprise_log_ctr_ = 0;   // console-throttle counter
    long                 birth_surprise_cycle_ = 0;     // probe cycle index (advances only when the grid was read)
    std::vector<Eigen::Vector2f> last_cabinet_dets_xy_;   // this cycle's ZED "cabinet" detection centroids (room frame)
    std::ofstream        kitchen_cells_csv_;            // per-cycle kitchen-cell log (cfg_.kitchen_cells_csv_path)
    long                 kitchen_cells_cycle_ = 0;      // kitchen CSV cycle index

    // Residual-birth debounce: a candidate arm must recur near the same place for residual_birth_frames
    // cycles before it births (rejects a transient residual flicker). Holds the current candidate.
    bool            residual_cand_active_ = false;
    Eigen::Vector2f residual_cand_xy_     = Eigen::Vector2f::Zero();
    rc::RunSeed     residual_cand_seed_{};
    int             residual_cand_hits_   = 0;
    std::chrono::steady_clock::time_point last_monitor_tp_{};   // ~5 Hz throttle
    std::chrono::steady_clock::time_point last_compute_tp_{};   // compute-rate EMA
    FPSCounter                            fps_counter_;         // overall compute()-cycle rate (std::cout heartbeat)

    std::unique_ptr<DSR::RT_API>                        rt_api_;
    std::unique_ptr<DSR::InnerEigenAPI>                 inner_eigen_;      // for room↔body↔zed extrinsic (silhouette)
    std::unique_ptr<DSR::InnerGaussianAPI>              gaussian_api_;     // Part B: chain covariance propagation
    std::unique_ptr<rc::MaskIngestor>                   mask_ingestor_;    // perception (masks-only)
    std::unique_ptr<rc::ConceptLidarIngestor>             lidar_ingestor_;   // YOLO-independent LiDAR range channel
    std::unique_ptr<rc::CabinetSceneGraph>                scene_graph_;      // DSR node/RT I/O
    uint64_t                                            room_node_id_ = 0;

signals:
    void presenceReady();
    void presenceLost();
};

#endif // SPECIFICWORKER_H
