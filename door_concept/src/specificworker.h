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
#include "../../common/concept_presence/concept_presence.h"   // rc::presence::ConceptProtocol (SHARED)
#include "../../common/epistemic_step/epistemic_step.h"   // rc::epistemic::step (SHARED)
#include "../../common/graph_layout/graph_layout.h"   // rc::gui::trigger_layout_twopi (SHARED)
#include "../../common/instance_tracker/instance_tracker.h"   // rc::InstanceTracker (birth/associate/death)

// ─── SpecificWorker ──────────────────────────────────────────────────────────

#include "door_semantic_field.h"   // rc::SemanticProbField
#include "door_actuator.h"          // rc::DoorActuator (RoboCompDoorControl client)
#include "door_pragmatics.h"        // rc::door::InteractionState + the three contracts
#include "../../common/lidar_ingestor/concept_lidar_ingestor.h"   // rc::ConceptLidarIngestor (SHARED)
#include "door_world_registration.h" // rc::DoorWorldRegistration (room→world from named doors)
#include "../../common/rgb_ingestor/rgb_ingestor.h"
#include "../../common/rgb_ingestor/depth_ingestor.h"   // rc::RgbIngestor (SHARED media-plane RGB)

#include <QLabel>
#include <QComboBox>

#include <QPushButton>

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

    // ── The three PRAGMATIC door affordances: approach / open / cross ─────────────────────────────
    // Siblings of the epistemic `aff_door_N` under the same door node, and a different KIND of offer:
    // that one asks for a LOOK that shrinks Σ, these ask the consumer to DO something to the door. Each
    // is offered only while this agent believes the action is possible — see door_pragmatics.h for the
    // three preconditions and why none of them is a threshold.
    //
    // ★RUN FOR EVERY LIVE INSTANCE, EVERY CYCLE, from compute() and NOT from process_door_node(). That
    // function bails early on a stale or young instance, and a cycle that cannot compute a precondition
    // must SAY so (rc::pragmatic::Offer::known == false) rather than be skipped: a skipped cycle neither
    // refreshes nor withdraws a standing offer, leaving the consumer ranking a frozen price.
    void step_pragmatic_affordances(rc::DoorInstance& inst);
    rc::door::InteractionState compute_interaction_state(const rc::DoorInstance& inst) const;

    // Assert / retract the `transitable` SELF-EDGE on a door's own node from the passability belief.
    // door --[transitable]--> door, present exactly while we believe this robot can get through it.
    // See DoorPragmaticCfg for why the edge is named after the robot fitting rather than the leaf moving.
    void step_transitable_edge(rc::DoorInstance& inst, const rc::door::InteractionState& st);

    // ★THE ONE REQUEST PATH. The strip buttons and the `open` affordance both come through here, so the
    // frame handling, the registration donation and the phi anticipation exist once. Returns false with
    // `why` filled when the request is refused LOCALLY — no provider, or no transform to build a
    // by-place request from. Refusing locally is the safe failure: a request built on a missing
    // transform can only be rejected, or (far worse) match some OTHER door within 400 mm and open it.
    bool request_door_actuation(rc::DoorInstance& inst, bool open, const std::string& provider_id,
                                const std::string& purpose, std::string& why);
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
    // Place the door on the 2-D graph canvas beside its ACTUAL RT parent (a wall_* once resolved, the
    // room before that), and follow the parent when it changes. See the definition for the misleading
    // picture this closes.
    void place_door_on_canvas(const DSR::Node& node, rc::DoorInstance& inst);

    void refresh_room_geometry();         // load the room delimiting polygon into the fitter (containment pose prior)

    // ── WHICH ROOM ARE WE IN? (the `current` edge, owned by ltsm_agent) ───────────────────────────
    // ★THE ROOM MUST BE FOLLOWED, NOT LATCHED. This agent used to take `get_nodes_by_type("room").front()`
    // once and keep it for ever, re-resolving only if that node was DELETED. During a room handover BOTH
    // rooms exist at the same time — that overlap is deliberate, it is the only window in which the seam
    // between the two frames is measurable — so `front()` is arbitrary and the latch never let go. The
    // agent then stayed anchored to the OLD room, which is not a cosmetic error:
    //   · a door detected in the new room is created as a child of the old one and its RT edge is
    //     published in the old room's frame, i.e. wrong by the whole inter-room seam (metres);
    //   · the room-containment prior then loads the OLD polygon, finds every new-room door outside it,
    //     and removes it at OutOfRoomGain per frame — a path that bypasses the sensor channel entirely,
    //     so nothing can rescue it. The visible symptom is doors flickering in and out in the new room,
    //     which looks nothing like a room-tracking bug.
    // (This is the write-once room memo the fleet audit lists as still open in nine agents; this is one.)
    //
    // Returns the room the `current` edge points at, or nullopt when NO such edge exists anywhere.
    // ★nullopt is NOT "no room" and must never be treated as a release — see step_room_following.
    std::optional<std::uint64_t> resolve_current_room() const;

    // Follow `current`, and LET GO of the room we leave. Called once per cycle before anything reads
    // room_node_id_.
    void step_room_following();

    // ★THE LET-GO RULE (agreed with ltsm_agent 2026-09-12; belongs in CONCEPT_AGENT_LIFECYCLE.md beside
    // REMOVE). An agent anchored to a room that is no longer the current one lets that room go: it stops
    // fitting and removes ITS OWN nodes through its own cleanup path. Nobody deletes anybody else's
    // nodes — a stranger deleting an affordance mid-execution is the stranded-Completed-for-ever defect
    // the fleet already paid for once, and only the owner knows the protocol state.
    void release_room(std::uint64_t old_room, std::uint64_t new_room);
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
    void on_optional_peer_lost(const std::string &name, std::uint32_t id);
    void on_optional_peer_ready(const std::string &name, std::uint32_t id);

    // ── Primary-input (masks) stream gate — mirrors table_concept (see CLAUDE.md primary-input gate) ──
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

    // Primary-input stream-gate bookkeeping (mirrors table_concept). All main-thread (FSM hooks).

    rc::DoorConfig                                         cfg_;
    // The low bpearl dome, via the SHARED media-plane ingestor five other concept agents already use.
    // It is the only source in this agent that can see a leaf the segmenter has stopped labelling.
    std::unique_ptr<rc::ConceptLidarIngestor>               lidar_ingestor_;
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

    // rc::probe row per live door: viewpoint + framing + detector outcome. See the definition for why
    // the ROBOT POSE is the point — absence is integrated as if each frame were an independent trial,
    // and a parked robot manufactures ~100 refutations of one look. Measured in retina on 2026-09-08:
    // while the robot is stopped only 5.1% of frames carry a NEW image (99.6% while moving), so the
    // duplicates are not merely correlated, they are the same pixels. Any fit over these rows must
    // de-duplicate by viewpoint or it will repeat the August failure of a confident fit on copies.
    // retina's graded class posterior, refreshed once per cycle and sampled under each door's own
    // projected contour. Invalid (and inert) whenever retina publishes no posterior.
    rc::SemanticProbField semantic_field_;

    // ── Door actuation (RoboCompDoorControl client) ────────────────────────────────────────────────
    // Ask a provider — Webots supervisor, home automation, or a person over TTS — to move a door. The
    // robot never learns from the reply that the door IS open: `Delivered` says the provider acted, and
    // whether the world changed stays a perceptual question. A protocol event is not evidence.
    rc::DoorActuator                    door_actuator_;
    // room→world learned from doors the OPERATOR has identified by name. Place-matching needs the two
    // frames registered and they are not (measured: 6.45 m apart, ~90.3° rotated), so every successful
    // by-name request donates a labelled correspondence and the transform falls out of ordinary use.
    rc::DoorWorldRegistration           door_registration_;
    std::optional<rc::DoorActuator::Pending> door_pending_;
    QPushButton* door_act_open_btn_  = nullptr;
    QPushButton* door_act_close_btn_ = nullptr;
    QLabel*      door_act_state_     = nullptr;
    QComboBox*   door_actuation_pick_ = nullptr;   // which PROVIDER door id to quote; index 0 = by place
    QLabel*      door_phi_label_      = nullptr;   // the target door's LEAF ANGLE, and whether it is fitted
    // The door a button press would act on: nearest believed instance to the robot, named in the UI so
    // a wrong pick is visible BEFORE the request goes out.
    struct ActuationTarget { std::string name; Eigen::Vector2f xy; float yaw; float width_m; float range_m;
                             float phi; bool phi_fitted; };
    [[nodiscard]] std::optional<ActuationTarget> nearest_door_for_actuation() const;
    void refresh_door_actuation_ui();

    // ZED RGB off the media plane, so this agent can check its OWN projected contour against the image.
    // ★It must be computed HERE and not in retina: retina draws a similar rectangle from the DSR node's
    // oriented box with a transform pinned to a different stamp, and a defence measured on a not-quite-
    // right contour is not a defence. The belief predicts the shape, so the belief tests for it.
    std::unique_ptr<rc::RgbIngestor> rgb_ingestor_;
    // The DEPTH plane's metric half of the same channel: "is there a surface at the distance I predict,
    // with space behind its edge?" — the question that separates a door from a photograph of one, which
    // no amount of RGB gradient can answer.
    std::unique_ptr<rc::DepthIngestor> depth_ingestor_;

    void log_detect_probe();
    std::ofstream detect_probe_csv_;
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
