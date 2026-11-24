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
 * specificworker.h — countable_concept agent (orchestration + Qt/DSR glue).
 *
 * Per compute() cycle: ingest ZED YOLO "bed" masks → InstanceTracker (birth / associate / merge) →
 * process_countable_node (one AI2 CountableBelief update per assigned slice) → publish the fit back to DSR (RT edge +
 * dims + mesh + covariance) and emit epistemic proposals when a bed stays under-observed. Ricoh-360
 * detections are bearing-only PERIPHERAL ATTENTION (they never birth or fit). Also owns the two standalone
 * top-level windows (belief dashboard + evidence monitor). The fit core is rc::CountableFitter, perception
 * rc::MaskIngestor, DSR I/O rc::CountableSceneGraph. See COUNTABLE.md for the belief/fit core.
 */

#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#include <atomic>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <fstream>

#include <genericworker.h>
#include <fps/fps.h>              // shared FPSCounter — logs "Period/Fps/cpu/mem" on std::cout each cycle
#include <Eigen/Dense>
#include <unordered_set>

#include "countable_config.h"      // rc::CountableConfig + load_countable_config
#include "countable_instance.h"    // rc::CountableInstance
#include "../../common/mask_ingestor/mask_ingestor.h"     // rc::MaskIngestor (perception)
#include "../../common/lidar_ingestor/concept_lidar_ingestor.h"                          // rc::ConceptLidarIngestor (YOLO-independent LiDAR)
#include "../../common/rgb_ingestor/rgb_ingestor.h"                 // rc::RgbIngestor  — the SHARED ZED RGB consumer
#include "../../common/rgb_ingestor/depth_ingestor.h"               // rc::DepthIngestor — its metric sibling
#include "../../common/instance_tracker/instance_tracker.h"   // rc::InstanceTracker (birth/associate/death)
#include "../../common/birth_fragment/birth_fragment.h"       // rc::BirthFragment — the probation burst
#include "countable_scene_graph.h" // rc::CountableSceneGraph (DSR node/RT I/O)
#include "countable_fitter.h"
#include "../../common/phantom_log/phantom_log.h"   // rc::history::PhantomLog (shadow-mode birth/death record)      // rc::CountableFitter (active-inference core)
#include "../../common/phantom_log/observer_pose.h"   // rc::history::note_observer (SHARED)
#include "../../common/agent_exit/terminal_exit.h"   // rc::agent::terminal_exit (SHARED)
#include "countable_existence.h"
#include "../../common/exclusion/exclusion.h"   // rc::exclusion::Claim (SHARED)   // rc::CountableExistence (evidence-based removal)
#include "../../common/birth_surprise/birth_surprise_probe.h"   // rc::BirthSurpriseProbe (SHARED, read-only: residual grid → birth surprise)
#include "epistemic_planner.h"
#include "../../common/nbv/graph_obstacles.h"   // rc::nbv::collect_graph_obstacles — DSR-side viewpoint obstacles
#include "../../common/object_affordance/object_affordance.h"
#include "countable_model.h"
#include "../../common/dashboard/belief_inspector.h"
#include "../../common/dashboard/belief_strip.h"
#include "../../common/dashboard/custom_widget.h"
#include "../../common/dashboard/evidence_monitor.h"
#include "../../common/dashboard/timeseries_plot.h"
#include "../../common/agent_presence_coordinator/agent_presence_coordinator.h"
#include "../../common/concept_presence/concept_presence.h"   // rc::presence::ConceptProtocol (SHARED)
#include "../../common/epistemic_step/epistemic_step.h"   // rc::epistemic::step (SHARED)
#include "../../common/graph_layout/graph_layout.h"   // rc::gui::trigger_layout_twopi (SHARED)
#include "../../common/peripheral_channel/peripheral_channel.h"   // rc::peripheral:: (SHARED)
#include "tick_timing.h"
#include "rt_memo.h"
#include "../../common/diag_log/rotating_csv.h"
#include "../../common/birth_surprise/birth_surprise_log.h"   // rc::BirthSurpriseLog (SHARED)

// ─── SpecificWorker ──────────────────────────────────────────────────────────────────────────────

class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:
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
                           float x, float y, const rc::CountableInstance* inst, std::string_view note);

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
    // The fit (observe/infer/belief/support-bank) lives in rc::CountableFitter; perception in
    // rc::MaskIngestor; DSR I/O in rc::CountableSceneGraph. The worker keeps orchestration + the
    // post-fit epistemic/affordance/Qt-diagnostics steps.
    using CountableObservation = rc::CountableFitter::CountableObservation;

    // ── Orchestration + post-fit steps ────────────────────────────────────────
    void load_config(const ConfigLoader& cfg);
    // Push the room geometry (wall polygon + interior centroid) into the fitter each cycle for the wall-flush
    // factor (a bed's back face rests against a room wall). Cheap read of the room node's delimiting polygon.
    void refresh_room_geometry();
    uint64_t    outline_room_ = 0;      // last (room, outline size) logged — log on change
    std::size_t outline_n_    = 0;
    bool        births_held_  = false;  // WAIT episode (no outline) — logged on change
    void process_countable_node(const DSR::Node& node);
    void publish_countable_cycle(rc::CountableInstance& inst,
                             const DSR::Node& node,
                             const CountableObservation& observation,
                             float free_energy);
    bool assess_countable_state(rc::CountableInstance& inst, uint64_t node_id, float free_energy);
    void publish_countable_diagnostics(const rc::CountableInstance& inst,
                                   const CountableObservation& observation,
                                   float free_energy);
    void publish_countable_intentions(rc::CountableInstance& inst,
                                  uint64_t node_id,
                                  const CountableObservation& observation,
                                  float free_energy);
    void step_convergence(rc::CountableInstance& inst, DSR::Node& node, float free_energy);
    void step_epistemic(rc::CountableInstance& inst, DSR::Node& node);
    void trigger_graph_layout_twopi();   // injected into CountableSceneGraph as the relayout callback

    // Multi-instance birth/associate/merge (shared rc::InstanceTracker; the only instance-lifecycle path).
    // Associates "bed" masks to instances, spawns a bed from an unexplained mask, merges overlaps.
    rc::InstanceTracker tracker_;
    // Probation bursts of the tracker's pending births, keyed by candidate id. A candidate matures over
    // Tracker.BirthFrames observations; this keeps those observations so the instance can be seeded from
    // measured geometry — and refused if the union does not read as a bed — instead of being created from
    // one frame's centroid and retracted later. See common/birth_fragment/birth_fragment.h.
    rc::BirthFragment birth_frag_;
    // One mask slice's room-frame support points (bounds-checked). Shared by the per-frame candidate score
    // and the burst banking so both see exactly the same cloud.
    static std::vector<Eigen::Vector3f> slice_cloud(const rc::MaskIngestor::MasksPacket& pkt, int slice_index);
    // SELF-BODY birth term (common/self_body/self_body.h): the fraction ∈[0,1] of a ZED slice's support points that lie
    // OUTSIDE the robot's own body. Points are carried to the robot frame by the static body←zed mount (no
    // localisation in it). 1 when it cannot be evaluated (asserts nothing it has not measured).
    float slice_world_fraction(const rc::MaskIngestor::MasksPacket& pkt, int slice_index);
    // Other objects in the graph as robot-inflated footprints, so the NBV never proposes standing on furniture.
    // Viewpoint obstacles come from rc::nbv::collect_graph_obstacles (common/nbv/graph_obstacles.h) — the
    // shared DSR adapter. The local copy this agent carried read the deprecated int obj_width/obj_depth that
    // nothing writes, so it silently returned an empty list on every call; see that header for both defects.
    rc::nbv::Sensor zed_sensor_model() const;   // real FoVs + mount height for the NBV
    void run_instance_tracker();   // called every cycle from compute()
    void retire_instance(std::uint64_t id);   // shared teardown: affordance + fitter forget + graph delete
    // Physical-exclusion invariant: two beds cannot share space. Collapse any pair of instances whose
    // oriented footprints overlap beyond Tracker.MergeOverlap, keeping the more-observed one.
    void merge_overlapping_instances();
    void apply_class_factor();        // the class factor of existence: soft singleton + "is this really a <class>?" retire

    // ── Presence protocol ────────────────────────────────────────────────────
    void waiting_enter();
    void waiting_loop();
    void operating_enter();
    void operating_loop();
    void degraded_enter();
    void degraded_loop();
    void cleanup_owned_nodes();
    // True once the config gave this process a class AND an agent name. Before that (startup-check, or any
    // path reaching cleanup before load_config) every owned-node sweep and geometry save must be a no-op:
    // an empty name prefix matches every node in the shared graph.
    bool owned_identity_known() const { return not cfg_.concept_class.empty() and not cfg_.agent_name.empty(); }
    // Log prefix: the [Agent] name, or the binary's name while the config is not loaded yet.
    std::string log_name() const { return cfg_.agent_name.empty() ? std::string("countable_concept") : cfg_.agent_name; }
    void remove_stale_affordance_nodes();   // sweep affordances parented to a bed (start + exit)
    void remove_owned_countable_nodes();        // startup stale-sweep of "bed*" nodes (mirrors bottle)
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
    bool warned_recall_area_ = false;   // log "no room polygon" for the recall area once
    std::atomic<bool> shutting_down_{false};
    // The presence protocol AND the gate state it owns (operating_since_ms / stall_reported /
    // degraded_from_input / first_operating_done) — SHARED, common/concept_presence. The transitions set
    // those, so they belong with the transitions; masks_stream_stalled() reads the baseline back.
    rc::presence::ConceptProtocol presence_protocol_;
    AgentPresenceCoordinator presence_coordinator_;

    // Primary-input stream-gate bookkeeping (mirrors room_concept). All main-thread (FSM hooks).

    rc::CountableConfig                                         cfg_;
    rc::EpistemicPlanner                                    epistemic_planner_;
    std::unique_ptr<rc::CountableFitter>                        fitter_;    // active-inference fit core (owns instances)
    rc::history::PhantomLog                             phantom_log_;   // shadow-mode birth/death record
    std::unique_ptr<rc::CountableExistence>                    existence_; // evidence-based removal (existence log-odds)

    // Live belief dashboard — its OWN top-level window (extracted from the DSR graph dock so it shows
    // independently of Agent.graph; mirrors room_concept/kinova_controller). Geometry persisted via QSettings.
    QWidget*             dashboard_window_ = nullptr;   // single top-level container: evidence monitor + belief plots
    Custom_widget*       custom_widget_ = nullptr;
    rc::TimeSeriesPlot*  ts_plot_       = nullptr;   // FE (+ baseline)
    rc::TimeSeriesPlot*  ts_surprise_plot_ = nullptr;   // FE surprise (attention signal), own panel/scale
    std::unordered_set<std::string> ts_known_beds_;   // node_names with live timeseries series (for pruning)
    void prune_dead_series();   // drop timeseries series for beds removed from the graph (periodic, in compute)

    // ── Ricoh 360 as PERIPHERAL ATTENTION (bearing-only) ──────────────────────────────────────────────────
    // A ricoh detection has a reliable DIRECTION but a biased centroid/extent, so it never births/fits (that
    // caused duplicates + drift). Instead an UNASSIGNED ricoh bearing (no known bed lies along it) becomes an
    // attention target: "seek a ZED view in this direction to birth/confirm the bed" (peripheral→saccade→fovea).
    // Unassigned ricoh bearings this cycle — the "go and check" candidates. ★The per-agent
    // RicohBearingTarget that used to be declared here was a byte-identical copy of
    // rc::peripheral::AttentionTarget in all four agents, and NOTHING reads its fields: only
    // .size() is consumed, for the evidence counter. Kept (it is the hook the proto-object
    // go-and-check work will consume) but on the shared type, so there is one definition.
    std::vector<rc::peripheral::AttentionTarget> ricoh_attention_targets_;
    void process_ricoh_bearings();   // associate ricoh detections to beds BY DIRECTION; collect the unassigned
    void log_detect_probe();         // rc::probe row per live instance: viewpoint + framing + detector outcome
    std::ofstream detect_probe_csv_;
    rc::RtMemo rt_memo_;                 // per-tick transform memo (active only inside compute())
    rc::TickTiming timing_;              // per-stage timing of the current compute() tick
    std::ofstream  timing_csv_;          // timing_log.csv (one row per tick)
    bool           timing_csv_failed_ = false;
    void write_timing_row();
    rc::TimeSeriesPlot*  ts_cov_plot_   = nullptr;   // belief uncertainty U(Σ) = Σ pos+size posterior std (m)
    rc::TimeSeriesPlot*  ts_res_plot_   = nullptr;   // residual point count
    // Bottom panel (replaces the old σ_w/σ_h time-series): the WHOLE belief — every state DOF with its
    // posterior σ, the consumer's demand σ* and the remaining adequacy gap, Σ as a correlation heatmap,
    // and the door-facing mode posterior. Fed by refresh_belief_inspector() on the evidence-monitor tick.
    rc::BeliefInspector* belief_inspector_ = nullptr;
    void refresh_belief_inspector();
    void build_dashboard();          // create the dashboard + evidence-monitor windows (called from initialize)
    void restore_dashboard_geometry();
    void save_dashboard_geometry() const;

    // ── Compact belief strip — its OWN SMALL top-level window ─────────────────────────────────────────
    // One row per instance, and the row is a 60 s time series of the certainty channel + p(existence).
    // This is the window meant to stay open; the big dashboard is the drill-down its "details ▸" opens.
    QWidget*         strip_window_ = nullptr;
    rc::BeliefStrip* belief_strip_ = nullptr;
    void refresh_belief_strip();
    void restore_strip_geometry();
    void save_strip_geometry() const;


    // Live "evidence consuming" monitor — its OWN top-level window (per-instance snapshot + global counters).
    rc::EvidenceMonitor* evidence_monitor_ = nullptr;
    rc::EvidenceGlobals  ev_g_{};                       // pipeline counters (per-cycle fields reset in compute)
    void refresh_evidence_monitor();                    // throttled build+push of the snapshot (main thread)
    void read_occupied_cells();                         // G6: residual `grid` latched cells → occupied_cells_ (stamp-keyed)
    bool read_residual_field();                         // snapshot residual_concept's `grid` node → residual_field_
    void log_birth_surprise();                          // EXPERIMENTAL read-only probe (cfg_.birth_surprise_probe)
    rc::nbv::OccupiedCells occupied_cells_;             // G6: for the standpoint check
    std::uint64_t        occupied_stamp_ = 0;
    rc::GridField        residual_field_;               // this cycle's residual (surprise) field; read at compute head
    // The birth-surprise / birth-fusion probe: two CSVs and two counters, bundled so a new agent cannot
    // adopt the writer and forget a counter. SHARED — common/birth_surprise/birth_surprise_log.h.
    rc::BirthSurpriseLog birth_surprise_log_;
    std::vector<Eigen::Vector2f> last_countable_dets_xy_;   // this cycle's ZED "bed" detection centroids (room frame)
    // Highest mask frame_id already COUNTED toward birth maturation. The tracker is fed every cycle (so pending
    // candidates never expire on a stale cycle), but a candidate may only ACCRUE on a genuinely new observation
    // — otherwise BirthFrames counts compute cycles, not evidence, and one mask frame re-presented at 10 Hz
    // matures a phantom in well under a second. See [[bed-birth-tightening]].
    long last_tracker_mask_frame_ = -1;
    std::chrono::steady_clock::time_point last_monitor_tp_{};   // ~5 Hz throttle
    std::chrono::steady_clock::time_point last_compute_tp_{};   // compute-rate EMA
    FPSCounter fps_counter_;   // per-cycle "[countable_concept Compute] Period/Fps/cpu%/mem" heartbeat (std::cout)

    std::unique_ptr<DSR::RT_API>                        rt_api_;
    std::unique_ptr<DSR::InnerEigenAPI>                 inner_eigen_;      // for room↔body↔zed extrinsic (silhouette)
    std::unique_ptr<DSR::InnerGaussianAPI>              gaussian_api_;     // Part B: chain covariance propagation
    std::unique_ptr<rc::MaskIngestor>                   mask_ingestor_;    // perception (masks-only)
    std::unique_ptr<rc::ConceptLidarIngestor>             lidar_ingestor_;   // YOLO-independent LiDAR range channel

    // Other concepts' standing claims on room space (SHARED, common/exclusion). Refreshed ONCE per compute()
    // cycle — one graph walk feeding both the birth filter and the existence occupancy discount, so the two
    // can never disagree about who is where.
    std::vector<rc::exclusion::Claim> foreign_claims_;
    // ★MIGRATED to the shared ingestor. This agent carried a private copy that was byte-identical to
    // common/rgb_ingestor once the object noun was substituted — which is how a media-plane fix lands in
    // two places out of three. The shared header had already named this agent as one to point here.
    std::unique_ptr<rc::RgbIngestor>                          rgb_ingestor_;     // ZED RGB media plane
    std::unique_ptr<rc::DepthIngestor>                        depth_ingestor_;   // ZED depth (CV_32F m, NaN = no return)
    // One flag for both planes: either consumer is reason enough to bring the camera up. Its ADDRESS is
    // handed to the ingestors, so it must outlive them — a member, never a temporary.
    bool                                                      camera_planes_wanted_ = false;
    std::unique_ptr<rc::CountableSceneGraph>                scene_graph_;      // DSR node/RT I/O
    uint64_t                                            room_node_id_ = 0;

signals:
    void presenceReady();
    void presenceLost();
};

#endif // SPECIFICWORKER_H
