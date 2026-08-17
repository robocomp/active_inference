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
 * door_concept — Active Inference agent for door instance detection and maintenance.
 *
 * Owns the generative model (7-param state + compound SDF) for every door
 * node in the DSR graph.  Runs a free-energy minimisation loop, maintains a
 * binned historical sample queue, and emits epistemic action proposals to
 * mission-controller when door surfaces remain under-observed.
 *
 * See ../CONCEPT_AGENT_RECIPE.md for the full design specification.
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

#include <genericworker.h>
#include <fps/fps.h>
#include <Eigen/Dense>
#include <unordered_set>

#include "door_config.h"      // rc::DoorConfig + load_door_config
#include "door_instance.h"    // rc::DoorInstance
#include "../../common/mask_ingestor/mask_ingestor.h"     // rc::MaskIngestor (perception)
#include "door_scene_graph.h" // rc::DoorSceneGraph (DSR node/RT I/O)
#include "door_fitter.h"
#include "door_bearing_range.h"   // rc::door::ResidualField / range_along_bearing (bearing → range cascade)
#include "../../common/phantom_log/phantom_log.h"   // rc::history::PhantomLog (shadow-mode birth/death record)      // rc::DoorFitter (active-inference core)
#include "../../common/phantom_log/observer_pose.h"   // rc::history::note_observer (SHARED)
#include "../../common/agent_exit/terminal_exit.h"   // rc::agent::terminal_exit (SHARED)
#include "epistemic_planner.h"
#include "../../common/object_affordance/object_affordance.h"
#include "door_model.h"
#include "../../common/dashboard/belief_inspector.h"
#include "../../common/dashboard/belief_strip.h"
#include "../../common/dashboard/evidence_monitor.h"
#include "../../common/dashboard/custom_widget.h"
#include "../../common/dashboard/timeseries_plot.h"
#include "../../common/agent_presence_coordinator/agent_presence_coordinator.h"
#include "../../common/instance_tracker/instance_tracker.h"   // rc::InstanceTracker (birth/associate/death)

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
    bool is_shutting_down() const noexcept { return shutting_down_.load(); }

public slots:
    void initialize();
    void compute();
    // SHADOW-MODE birth/death recorder (CONCEPT_AGENT_LIFECYCLE.md §4.2). Records ONLY — it can never
    // alter a birth or a removal. Attribution fields captured at death separate a genuine classifier
    // phantom from one of our own removal defects.
    void log_phantom_event(std::string_view event, std::uint64_t id, std::string_view name,
                           float x, float y, const rc::DoorInstance* inst, std::string_view note);

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
    // The fit (observe/infer/belief/support-bank) lives in rc::DoorFitter; perception in
    // rc::MaskIngestor; DSR I/O in rc::DoorSceneGraph. The worker keeps orchestration + the
    // post-fit epistemic/affordance/Qt-diagnostics steps.
    using DoorObservation = rc::DoorFitter::DoorObservation;

    // ── Orchestration + post-fit steps ────────────────────────────────────────
    void load_config(const ConfigLoader& cfg);
    void process_door_node(const DSR::Node& node);
    void run_instance_tracker();          // data-driven birth/associate/death (the only instance-lifecycle path)
    void retire_instance(std::uint64_t id);   // shared teardown: affordance + fitter forget + graph delete
    void merge_overlapping_instances();   // collapse two instances on the same door (seat-footprint overlap)
    // NBV decision monitor → etc/door_nbv_log.csv (where the door is vs where we told the robot to stand).
    void log_nbv_decision(const rc::DoorInstance& inst, const rc::nbv::Plan& plan,
                          const rc::EpistemicProposal& prop);
    int  nbv_obstacle_count_ = 0;   // obstacles fed to the last plan (0 ⇒ the walls never reached it)
    void update_existence_beliefs();      // continuous existence log-odds → evidence-based removal (no age immunity)
    // ── Identity re-acquisition ───────────────────────────────────────────────
    // A door that is removed leaves a GHOST: its name and the belief it had converged to. A later detection
    // landing within Existence.ReacquireRadiusM of a ghost is the SAME physical door coming back, so it takes
    // that name and resumes that geometry instead of being born as door_N+1 with template priors. Existence
    // itself restarts from the birth prior — the shape is remembered, the confidence is re-earned.
    struct DoorGhost
    {
        std::string     name;
        Eigen::Vector2f xy = Eigen::Vector2f::Zero();
        rc::DoorBelief  belief;
        int             lived_cycles = 0;
    };
    void remember_ghost(const rc::DoorInstance& inst);                 // called just before a node is deleted
    const DoorGhost* match_ghost(const Eigen::Vector2f& xy) const;     // nearest ghost within the radius, else null
    void forget_ghost(const Eigen::Vector2f& xy);                      // consume the ghost(s) at a re-acquired place
    std::vector<std::string> reserved_names() const;                   // names a ghost OR a stored identity still holds
    std::vector<DoorGhost> ghosts_;

    // ── Identity ACROSS RUNS (etc/door_identities.csv) ────────────────────────────────────────────
    // A ghost lives in RAM and every owned door node is deleted on shutdown, so at the next launch the
    // numbering restarted from birth ORDER — i.e. from whichever door the robot happened to see first.
    // Live 2026-08-09: door_1/door_2 traded physical doors across a restart, which is exactly what
    // "the affordances are switched" looks like from outside. A door is a hole in a wall and outlives
    // the process, so its name must too: this table maps NAME ↔ APERTURE PLACE and is written to disk.
    // Identity only — no belief. A door coming back re-earns its geometry; it does not re-earn its name.
    struct DoorIdentity
    {
        std::string     name;
        Eigen::Vector2f xy = Eigen::Vector2f::Zero();
    };
    std::vector<DoorIdentity> identities_;
    void load_identities();                                            // once, at startup
    void save_identities() const;                                      // whole table, locale-independent
    void note_identity(const std::string& name, const Eigen::Vector2f& ap);   // upsert by PLACE, then save
    const DoorIdentity* match_identity(const Eigen::Vector2f& xy) const;
    void remember_live_identities();                                   // shutdown: persist the doors still alive
    void refresh_room_geometry();         // load the room delimiting polygon into the fitter (containment pose prior)
    // Residual field (residual_concept's `residual` node): P(occupied ∧ ¬explained) per cell. Read at the
    // birth path only, and used ONLY to give a peripheral bearing a range — see door_bearing_range.h. The
    // same attribute trio table/cabinet/refrigerator already consume, so this is one more reader, not a
    // new dependency. Empty ⇒ the cascade falls through to the nominal range and the hypothesis stays a glance.
    bool read_residual_field();
    rc::door::ResidualField residual_field_;
    void publish_door_cycle(rc::DoorInstance& inst,
                             const DSR::Node& node,
                             const DoorObservation& observation,
                             float free_energy);
    bool assess_door_state(rc::DoorInstance& inst, uint64_t node_id, float free_energy);
    void publish_door_diagnostics(const rc::DoorInstance& inst,
                                   const DoorObservation& observation,
                                   float free_energy);
    void publish_door_intentions(rc::DoorInstance& inst,
                                  uint64_t node_id,
                                  const DoorObservation& observation,
                                  float free_energy);
    void step_convergence(rc::DoorInstance& inst, DSR::Node& node, float free_energy);
    void step_epistemic(rc::DoorInstance& inst, DSR::Node& node);
    void trigger_graph_layout_twopi();   // injected into DoorSceneGraph as the relayout callback

    // ── Presence protocol ────────────────────────────────────────────────────
    void waiting_enter();
    void waiting_loop();
    void operating_enter();
    void operating_loop();
    void degraded_enter();
    void degraded_loop();
    void cleanup_owned_nodes();
    void remove_stale_affordance_nodes();   // sweep affordances parented to a door (start + exit)
    void remove_owned_door_nodes();        // startup stale-sweep of "door*" nodes (mirrors bottle)
    void request_shutdown();
    // Crash-free exit (request_shutdown + DDS reset + _Exit), bypassing the Ice/static teardown abort.
    void terminal_shutdown();
    // Grace before a required-peer loss is treated as terminal (debounce transient presence flaps).
    static constexpr int REQUIRED_LOSS_GRACE_MS = 3000;
    void on_optional_peer_lost(const std::string &name, std::uint32_t id);
    void on_optional_peer_ready(const std::string &name, std::uint32_t id);

    // ── Primary-input (masks) stream gate — mirrors table_concept (see CLAUDE.md primary-input gate) ──
    // Admission probe (Waiting→Operating gate): the `masks` node is present and advertising a frame id.
    bool masks_stream_ready(std::string *detail = nullptr) const;
    // Operating stall predicate: no NEW masks frame for cfg_.masks_stall_timeout_ms, with a cold-start grace
    // measured from operating_since_ms_ before the first frame ever arrives. false when the gate is disabled.
    bool masks_stream_stalled(std::int64_t *age_ms_out = nullptr) const;
    // Admission predicate: the producer is CURRENTLY publishing fresh frames (a frame within the timeout
    // window). Distinct from masks_stream_ready() (node-exists, which persists after the producer dies) —
    // admitting on node-exists causes an instant re-stall flap. Requires refresh() to be pumped while Waiting.
    bool masks_stream_live() const;

    // ── Members ──────────────────────────────────────────────────────────────
    bool startup_check_flag = false;
    bool owned_nodes_cleaned_ = false;
    std::atomic<bool> shutting_down_{false};
    AgentPresenceCoordinator presence_coordinator_;

    // Primary-input stream-gate bookkeeping (mirrors table_concept). All main-thread (FSM hooks).
    std::int64_t operating_since_ms_   = 0;      // wall ms at Operating entry — cold-start stall-grace baseline
    bool         masks_stall_reported_ = false;  // one-shot: emit presenceLost once per stall episode (reset on entry)
    bool         degraded_from_masks_  = false;  // Degraded reason: recoverable mask-stall vs a real peer loss
    // One-shot: the startup stale-affordance sweep. ★NOT unguarded — on a RE-entry to Operating (a
    // transient required-peer flap → Degraded → recover) the affordance nodes in the graph are THIS
    // run's LIVE ones, so sweeping every bounce deletes and re-creates them and they flicker. Five
    // agents had this flag; this one swept unconditionally.
    bool         startup_affordance_sweep_done_ = false;
    std::int64_t last_wait_log_ms_     = 0;      // throttle for the "why still Waiting" line

    rc::DoorConfig                                         cfg_;
    rc::EpistemicPlanner                                    epistemic_planner_;
    std::unique_ptr<rc::DoorFitter>                    fitter_;   // active-inference fit core (owns instances)
    rc::history::PhantomLog                             phantom_log_;   // shadow-mode birth/death record

    // Live belief dashboard — its OWN top-level window (extracted from the DSR graph dock so it shows
    // independently of Agent.graph; mirrors room_concept/kinova_controller). Geometry persisted via QSettings.
    Custom_widget*       custom_widget_ = nullptr;
    rc::TimeSeriesPlot*  ts_plot_       = nullptr;   // FE (+ baseline)
    rc::TimeSeriesPlot*  ts_surprise_plot_ = nullptr;   // FE surprise (attention signal), own panel/scale
    rc::TimeSeriesPlot*  ts_cov_plot_   = nullptr;   // belief uncertainty U(Σ) = Σ pos+size posterior std (m)
    rc::TimeSeriesPlot*  ts_res_plot_   = nullptr;   // residual point count
    // Bottom panel (replaces the old pose-σ time-series): the WHOLE belief — every state DOF with its
    // posterior σ, Σ as a correlation heatmap, and the 4-mode yaw posterior. The door publishes no σ*,
    // so the inspector drops the σ*/adequacy columns rather than show invented targets.
    rc::BeliefInspector* belief_inspector_ = nullptr;
    void refresh_belief_inspector();
    // Section 1: the evidence-pipeline counter strip (same struct + widget as every other concept agent).
    QWidget*             dashboard_window_ = nullptr;   // combined window: counters over plots + inspector
    rc::EvidenceMonitor* evidence_monitor_ = nullptr;
    rc::EvidenceGlobals  ev_g_{};                       // per-cycle fields reset at the head of compute()
    void refresh_evidence_monitor();                    // throttled push of BOTH dashboard sections
    std::chrono::steady_clock::time_point last_monitor_tp_{};   // ~5 Hz dashboard tick
    std::chrono::steady_clock::time_point last_compute_tp_{};   // compute-rate estimate for the strip
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


    std::unique_ptr<DSR::RT_API>                        rt_api_;
    std::unique_ptr<DSR::InnerEigenAPI>                inner_eigen_;     // for room↔body↔zed extrinsic (silhouette)
    std::unique_ptr<DSR::InnerGaussianAPI>            gaussian_api_;    // Part B: chain covariance propagation
    std::unique_ptr<rc::MaskIngestor>                   mask_ingestor_;   // perception (masks-only)
    std::unique_ptr<rc::DoorSceneGraph>               scene_graph_;     // DSR node/RT I/O
    rc::InstanceTracker                                tracker_;         // multi-instance (Tracker.Enabled)
    // Last mask frame_id that CONTRIBUTED birth evidence. Agents feed the tracker every compute cycle on
    // purpose (a candidate with no matching detection expires), but persisting a candidate and accruing
    // evidence into it are different things — conflating them made birth_frames count COMPUTE CYCLES. See
    // common/instance_tracker/birth_evidence.h rule 1.
    long  last_birth_mask_frame_ = -1;
    int   exist_last_mask_frame_ = -1;    // (rc::BearingHypothesisStager removed with bearing-birth — see common/peripheral_channel)
    uint64_t                                            room_node_id_ = 0;
    FPSCounter                                          fps_counter_;     // overall compute()-cycle rate

signals:
    void presenceReady();
    void presenceLost();
};

#endif // SPECIFICWORKER_H
