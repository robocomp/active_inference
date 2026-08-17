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
 * chair_concept — Active Inference agent for chair instance detection and maintenance.
 *
 * Owns the generative model (7-param state + compound SDF) for every chair
 * node in the DSR graph.  Runs a free-energy minimisation loop, maintains a
 * binned historical sample queue, and emits epistemic action proposals to
 * mission-controller when chair surfaces remain under-observed.
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

#include "chair_config.h"      // rc::ChairConfig + load_chair_config
#include "chair_instance.h"    // rc::ChairInstance
#include "../../common/mask_ingestor/mask_ingestor.h"     // rc::MaskIngestor (perception)
#include "chair_scene_graph.h" // rc::ChairSceneGraph (DSR node/RT I/O)
#include "chair_fitter.h"      // rc::ChairFitter (active-inference core)
#include "../../common/phantom_log/phantom_log.h"   // rc::history::PhantomLog (shadow-mode birth/death record)
#include "epistemic_planner.h"
#include "../../common/object_affordance/object_affordance.h"
#include "chair_model.h"
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
    // The fit (observe/infer/belief/support-bank) lives in rc::ChairFitter; perception in
    // rc::MaskIngestor; DSR I/O in rc::ChairSceneGraph. The worker keeps orchestration + the
    // post-fit epistemic/affordance/Qt-diagnostics steps.
    using ChairObservation = rc::ChairFitter::ChairObservation;

    // ── Orchestration + post-fit steps ────────────────────────────────────────
    void load_config(const ConfigLoader& cfg);
    void process_chair_node(const DSR::Node& node);
    // Read the level-2 arrangement prior off the incoming `group_member` edge (ring_metaconcept is
    // the sole writer) and hand it to the belief. Inert when absent / kappa<=0 / stale.
    void refresh_rig_yaw_prior(rc::ChairInstance& inst, const DSR::Node& node);
    void run_instance_tracker();          // data-driven birth/associate/death (the only instance-lifecycle path)
    void merge_overlapping_instances();   // collapse two instances on the same chair (seat-footprint overlap)
    void update_existence_beliefs();      // continuous existence log-odds → evidence-based removal (no age immunity)
    // SHADOW-MODE birth/death recorder (CONCEPT_AGENT_LIFECYCLE.md §4.2). Records ONLY — never feeds back into
    // a belief, birth or removal. chair_concept matters most here: the radiator-reads-as-a-chair failure is a
    // CHAIR phenomenon, so this is the agent whose log actually tests the (place × bearing) clustering claim.
    void log_phantom_event(std::string_view event, std::uint64_t id, std::string_view name,
                           float x, float y, const rc::ChairInstance* inst, std::string_view note);
    void refresh_room_geometry();         // load the room delimiting polygon into the fitter (containment pose prior)
    void publish_chair_cycle(rc::ChairInstance& inst,
                             const DSR::Node& node,
                             const ChairObservation& observation,
                             float free_energy);
    bool assess_chair_state(rc::ChairInstance& inst, uint64_t node_id, float free_energy);
    void publish_chair_diagnostics(const rc::ChairInstance& inst,
                                   const ChairObservation& observation,
                                   float free_energy);
    void publish_chair_intentions(rc::ChairInstance& inst,
                                  uint64_t node_id,
                                  const ChairObservation& observation,
                                  float free_energy);
    void step_convergence(rc::ChairInstance& inst, DSR::Node& node, float free_energy);
    void step_epistemic(rc::ChairInstance& inst, DSR::Node& node);
    void trigger_graph_layout_twopi();   // injected into ChairSceneGraph as the relayout callback

    // ── Presence protocol ────────────────────────────────────────────────────
    void waiting_enter();
    void waiting_loop();
    void operating_enter();
    void operating_loop();
    void degraded_enter();
    void degraded_loop();
    void cleanup_owned_nodes();
    void remove_stale_affordance_nodes();   // sweep affordances parented to a chair (start + exit)
    void remove_owned_chair_nodes();        // startup stale-sweep of "chair*" nodes (mirrors bottle)
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
    std::int64_t last_wait_log_ms_     = 0;      // throttle for the "why still Waiting" line

    rc::ChairConfig                                         cfg_;
    rc::EpistemicPlanner                                    epistemic_planner_;
    std::unique_ptr<rc::ChairFitter>                    fitter_;   // active-inference fit core (owns instances)
    rc::history::PhantomLog                             phantom_log_;   // shadow-mode birth/death record

    // Live belief dashboard — its OWN top-level window (extracted from the DSR graph dock so it shows
    // independently of Agent.graph; mirrors room_concept/kinova_controller). Geometry persisted via QSettings.
    Custom_widget*       custom_widget_ = nullptr;
    rc::TimeSeriesPlot*  ts_plot_       = nullptr;   // FE (+ baseline)
    rc::TimeSeriesPlot*  ts_surprise_plot_ = nullptr;   // FE surprise (attention signal), own panel/scale
    rc::TimeSeriesPlot*  ts_cov_plot_   = nullptr;   // belief uncertainty U(Σ) = Σ pos+size posterior std (m)
    rc::TimeSeriesPlot*  ts_res_plot_   = nullptr;   // residual point count
    // Bottom panel (replaces the old pose-σ time-series): the WHOLE belief — every state DOF with its
    // posterior σ, Σ as a correlation heatmap, and the 4-mode yaw posterior. The chair publishes no σ*,
    // so the inspector drops the σ*/adequacy columns rather than show invented targets.
    rc::BeliefInspector* belief_inspector_ = nullptr;
    // ── Compact belief strip: ONE ROW PER INSTANCE, and the row is a 60 s time series ──────────────
    // A separate small top-level window: the standing display you keep in a corner while the big
    // dashboard stays closed until something looks wrong. Mirrors table/cabinet/door/refrigerator.
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
    void restore_dashboard_geometry();
    void save_dashboard_geometry() const;

    std::unique_ptr<DSR::RT_API>                        rt_api_;
    std::unique_ptr<DSR::InnerEigenAPI>                inner_eigen_;     // for room↔body↔zed extrinsic (silhouette)
    std::unique_ptr<DSR::InnerGaussianAPI>            gaussian_api_;    // Part B: chain covariance propagation
    std::unique_ptr<rc::MaskIngestor>                   mask_ingestor_;   // perception (masks-only)
    std::unique_ptr<rc::ChairSceneGraph>               scene_graph_;     // DSR node/RT I/O
    rc::InstanceTracker                                tracker_;         // multi-instance (Tracker.Enabled)
    float exist_support_scale_   = 0.0f;   // existence belief: online-calibrated E[npts·range²] (0 = seed from cfg)
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
