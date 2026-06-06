/*
 *    Copyright (C) 2026 by YOUR NAME HERE
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
	\brief
	@author authorname
*/

#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

// If you want to reduce the period automatically due to lack of use, you must uncomment the following line
//#define HIBERNATION_ENABLED

#include <genericworker.h>

#include "kinematics.h"
#include "efe_gradient.h"
#include "arm_belief_viewer_3d.h"

#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/api/dsr_rt_api.h>

#include <Eigen/Dense>
#include <memory>
#include <optional>
#include <random>
#include <fstream>
#include <cmath>
#include <algorithm>

/**
 * \brief Class SpecificWorker implements the core functionality of the component.
 */
class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:
    /**
     * \brief Constructor for SpecificWorker.
     * \param configLoader Configuration loader for the component.
     * \param tprx Tuple of proxies required for the component.
     * \param startup_check Indicates whether to perform startup checks.
     */
	SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check);

	/**
     * \brief Destructor for SpecificWorker.
     */
	~SpecificWorker();


public slots:

	/**
	 * \brief Initializes the worker one time.
	 */
	void initialize();

	/**
	 * \brief Main compute loop of the worker.
	 */
	void compute();

	/**
	 * \brief Handles the emergency state loop.
	 */
	void emergency();

	/**
	 * \brief Restores the component from an emergency state.
	 */
	void restore();

    /**
     * \brief Performs startup checks for the component.
     * \return An integer representing the result of the checks.
     */
	int startup_check();

	void modify_node_slot(std::uint64_t, const std::string &type){};
	void modify_node_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names){};
	void modify_edge_slot(std::uint64_t from, std::uint64_t to,  const std::string &type){};
	void modify_edge_attrs_slot(std::uint64_t from, std::uint64_t to, const std::string &type, const std::vector<std::string>& att_names){};
	void del_edge_slot(std::uint64_t from, std::uint64_t to, const std::string &edge_tag){};
	void del_node_slot(std::uint64_t from){};     
private:

	/**
     * \brief Flag indicating whether startup checks are enabled.
     */
	bool startup_check_flag;

	std::unique_ptr<Kinematics> kinematics_;
	std::unique_ptr<ArmBeliefViewer3D> arm_belief_viewer_;

	// Active EFE target — arm-base frame, metres. Updated each compute() cycle
	// from the bottle pose in DSR (which is in turn fed from Webots).
	Eigen::Vector3d reach_target_{0.4, 0.0, 0.1};
	bool proxy_unreachable_warned_         = false;
	bool webots_proxy_unreachable_warned_  = false;
	bool joint_dump_pending_               = true;  // dump first received TJoints, once

	// Controller lifecycle: home → wait for Start → run continuous EFE.
	// Un-checking the button drops back to SendingRestPose → Homing → idle.
	enum class Phase { SendingRestPose, Homing, WaitingForStart, ActiveEFE };
	Phase phase_ = Phase::SendingRestPose;
	bool  run_requested_ = false;   // mirrors the viewer button's checked state
	// Retracted "elbow-behind" ready rest pose: arm pulled back off the table,
	// elbow BEHIND the column and UP, FAR from the mast — elbow at world
	// ≈(−0.72,−0.72,1.23) (0.70 m from the column axis), hand pulled back to
	// ≈(−0.46,−0.39,1.11) (not over the table), camera up. Low μ (≈0.04) — this
	// retracted posture is near-singular, but it's only the parking pose; the arm
	// gains conditioning as it extends to grasp. Override via Controller.rest_pose.
	std::array<double, Kinematics::N_ARM_JOINTS> rest_pose_angles_{
		1.189, 0.244, 0.300, -2.101, 0.651, -1.502, 3.141
	};
	int homing_settled_ticks_ = 0;
	int homing_elapsed_ticks_ = 0;                        // cycles spent in the current homing attempt
	static constexpr double HOMING_TOLERANCE_RAD = 0.09;  // ≈ 5.2° — forgiving enough that a joint drooping a hair short under gravity still counts as "home" (it was jamming the round ~3° short)
	static constexpr int    HOMING_SETTLE_TICKS  = 5;     // consecutive cycles within tolerance
	// Safety timeout: if homing cannot converge within this many cycles (e.g. the
	// target pose is unreachable or the arm is jammed against an obstacle), give up
	// instead of slewing forever — send zero velocities and drop to WaitingForStart.
	// A normal home at ~0.8 rad/s clears even a π move in a few seconds; at a 20 ms
	// period 600 cycles ≈ 12 s is a generous ceiling that only fires on a real jam.
	static constexpr int    HOMING_TIMEOUT_TICKS = 600;

	// Bottle-approach behaviour (replaces the previous random-target cycle).
	// Side grasp: standoff next to the bottle, tool +Z pointing into the bottle
	// horizontally, tool +X aligned with the bottle's vertical axis so the
	// Robotiq fingers wrap around the body.
	static constexpr double APPROACH_STANDOFF_M = 0.12;  // standoff along approach axis
	static constexpr double REACH_TOLERANCE_M   = 0.02;  // 2 cm — "arrived"
	bool arrived_logged_ = false;

	// Phase-1 bring-up gate: when set (Controller.approach_only), the grasp FSM
	// drives to the standoff with the gripper open + correctly oriented and then
	// HOLDS there instead of committing to Inserting. Lets us validate the
	// approach + gripper configuration in isolation before enabling the grasp.
	bool approach_only_ = false;
	bool approach_hold_logged_ = false;
	// Controller.solver: false → closed-form DLS, true → proxQP (reproduces DLS now;
	// scaffold for migrating joint-limit/obstacle terms to hard QP constraints).
	bool use_qp_ = false;
	// Controller.qp_redundancy_weight: weight on the QP's μ/elbow linear term. ≤0 ⇒ λ²
	// (projection-equivalent); larger ⇒ genuine NEO soft competition (task slack yields).
	double qp_redundancy_weight_ = 0.0;
	// Controller.force_confidence: if ≥0, PIN confidence_ to this every cycle (overrides
	// learning + decay) for controlled fluidity/skill experiments. <0 ⇒ normal learning.
	double force_confidence_ = -1.0;
	// Controller.blend_radius: look-ahead coarticulation radius [m]. Within this distance of
	// a transit via-point (lift-top, place-hover) the commanded EE velocity is steered toward
	// the NEXT waypoint so the EE rounds the corner instead of stopping. Effective radius =
	// skill_c()·this, so it is 0 for a novice (= baseline exact stops) and rounds the corners
	// as skill rises. 0 = off. Keep ≤ ~0.05 so the lift corner-cut still clears the rise check.
	double blend_radius_ = 0.0;
	// Closest-approach tracker for the blended-via gate: a corner-cut never reaches the via
	// exactly, so a leg with an active blend advances once e_pos passes its minimum (corner
	// rounded) rather than waiting for e_pos < REACH_TOLERANCE. Reset on entry to each leg.
	double blend_min_dist_ = 1e9;
	void   sample_place_spot();   // pick the place spot from latched_grasp_ (called at Closing→Lifting)
	// ── Preference-field mode (option (c) prototype, steps 1–2) ──────────────────────────
	// Controller.use_preference_field: on the transit legs (lift-top, place-hover) replace the
	// hand-coded look-ahead blend with the 2-via Gaussian-mixture field (EFEParams::use_field).
	// The per-via precision expresses stop-vs-pass; field_prec_pass/stop are the two endpoints,
	// interpolated by skill_c() so a novice sees prec_stop (exact stops = parity, step 1) and a
	// skilled agent sees prec_pass (cruise-through = emergent blend, step 2). off ⇒ blend path.
	bool   use_preference_field_ = false;
	double field_prec_pass_ = 1.0;    // Π of a transit via at full skill (low ⇒ pass through)
	double field_prec_stop_ = 30.0;   // Π of a transit via at novice / of the next attractor (high ⇒ stop)
	double field_prec_ref_  = 6.0;    // λ_c = exp(−Π/ref): precision→terminal-speed map
	double field_overlap_   = 0.06;   // [m] RBF width: how near μ_c the next-via pull ramps in
	// ── Tactile re-centering (anti-tip grasp) ────────────────────────────────────────────
	// Controller.tactile_recenter: in Inserting, use the per-finger tip-force asymmetry
	// (ltipforce−rtipforce) to shift the seat target laterally (along live tool +X, the
	// finger-closing axis) toward the harder-pushing finger, centring the bottle between the
	// pads instead of pushing it over; commit to Closing only when seated AND not pushing.
	bool   tactile_recenter_ = false;
	double recenter_gain_    = 0.02;  // [m] max lateral seat shift at full L/R asymmetry
	double recenter_sign_    = 1.0;   // ±1: maps tip-force asymmetry sign to the +X direction (calibrate in sim)
	// Controller.tip_reflex: stop-and-rectify on a tip-BUMPER collision during Inserting — halt
	// forward motion, back off along −approach, and grow a lateral offset TOWARD the contacting
	// finger so the object enters the gap on retry; commit only when seated AND not in contact.
	// The offset accumulates to cancel the misalignment. Needs TGripper.{l,r}tipcontact.
	bool   tip_reflex_        = false;
	double tip_reflex_offset_ = 0.0;  // accumulated lateral correction [m] along tool +X (per attempt)
	// Controller.fixed_pick_xy = "x y" (world m): if set, respawn the bottle at this
	// fixed spot every rep instead of the R2 sweep — for controlled A/B experiments
	// (hold the pick constant, vary one knob).
	bool   fixed_pick_set_ = false;
	Eigen::Vector2d fixed_pick_xy_{0.0, 0.0};
	// Controller.elbow_gain (default 2.0; 0 ⇒ drop the elbow posture term) and
	// Controller.elbow_target_xy = "x y" (world m; empty ⇒ built-in default).
	double elbow_gain_ = 2.0;
	bool   elbow_target_set_ = false;
	Eigen::Vector2d elbow_target_xy_{0.0, 0.0};

	// Tip-trajectory logging (diagnose approach speed/shape). When tip_log_ is
	// set the ActiveEFE block prints one "[tiplog] ..." CSV line per cycle and
	// tracks the previous tip pose to report the measured Cartesian speed.
	bool                           tip_log_ = false;
	long                           tip_log_cycle_ = 0;
	std::optional<Eigen::Vector3d> tip_log_prev_pos_;

	struct SideGraspTarget
	{
		Eigen::Vector3d stand_off_pos;  // approach target (robot frame, m)
		Eigen::Vector3d grasp_pos;      // bottle body-centre: the grasp point, m
		Eigen::Vector3d z_tool_des;     // unit vector, robot frame
		Eigen::Vector3d x_tool_des;     // unit vector, robot frame
	};

	// Fraction of bottle height (measured from the base origin along the
	// bottle's +Z axis) that we aim the EE at. 0.5 = mid-body (the value that
	// grasped+placed reliably; tipping is now handled by the heavy bottle, not
	// by grasping lower, which only hurt reach on this high-mounted arm).
	static constexpr double BOTTLE_GRASP_HEIGHT_FRAC = 0.5;

	// ── Grasp FSM (inner state machine of Phase::ActiveEFE) ─────────────────
	// Hierarchical active inference: each state installs a prior (preferred
	// pose + gripper state) that the continuous EFE controller realises; a
	// transition fires when the lower level reports the prior is satisfied
	// (error inside the deadband tolerance) or an observation (finger force)
	// confirms the outcome. See EFE_CONTROLLER_MATH.md §1, §6.
	//   Tracking      : approach the live standoff, gripper open, bottle tracked.
	//   Inserting     : commit — latch the grasp frame, ease into the bottle body.
	//   Closing       : hold pose, close gripper, watch finger force.
	//   Lifting       : raise +Δz holding the grasped bottle.
	//   PlaceMoving   : carry the bottle to a hover above a random table spot.
	//   PlaceLowering : lower onto the table spot.
	//   PlaceReleasing: open gripper, let the bottle settle.
	//   PlaceRetreating: rise off the placed bottle.
	//   Retracting    : tilt-reflex withdrawal — bottle started to tip during the
	//                   approach/contact, so reopen and back off to let it settle,
	//                   then re-track. A fast protective loop, not a planned step.
	//   (then) return to rest pose via the outer Homing path, and loop.
	enum class GraspPhase { Tracking, Inserting, Closing, Lifting,
	                        PlaceMoving, PlaceLowering, PlaceReleasing, PlaceRetreating,
	                        Retracting };
	GraspPhase grasp_phase_ = GraspPhase::Tracking;
	int        grasp_settle_ticks_ = 0;   // consecutive converged cycles before committing
	int        closing_ticks_      = 0;   // cycles spent closing (miss timeout)
	int        insert_ticks_       = 0;   // watchdog: cycles spent seating/centring in Inserting
	int        lift_ticks_         = 0;   // watchdog: cycles spent trying to lift
	int        place_ticks_        = 0;   // watchdog within a place sub-state
	int        place_settle_ticks_ = 0;   // consecutive settled+upright cycles before releasing
	double     place_bottle_z_prev_ = 1e9;// previous bottle-z in PlaceLowering: detect "rested on table" (Δz≈0)
	// If the lift can't reach its target in this many cycles the arm grabbed
	// something it can't raise (a toppled/jammed bottle) — treat as a grasp miss
	// instead of looping in Lifting forever. ~5 s at a 20 ms period.
	static constexpr int LIFT_TIMEOUT_TICKS = 250;
	bool       returning_for_cycle_ = false;  // auto-restart Tracking after the rest-return
	int        round_cycles_       = 0;       // Controller.round_cycles: stop after N picks (0 = loop forever)
	int        pick_place_cycles_done_ = 0;   // completed picks in the current round

	// ── Skill-learning instrumentation ──────────────────────────────────────
	// First step toward the novice→skilled (closed-loop → model-based open-loop)
	// transition: probe each rep with a small STRUCTURED perturbation of the grasp
	// command (not noise — a low-discrepancy sweep that EXCITES the parameters we
	// want to identify and measures outcome SENSITIVITY), and log one row per rep
	// to a dataset. Offline this gives the grasp-pose bias, the actuation response,
	// and the per-region tolerance budget that licenses going open-loop. All OFF by
	// default (zero perturbation, no file) so behaviour is unchanged unless enabled.
	struct GraspPerturbation
	{
		double dx_perp     = 0.0;   // m,  offset along tool +X (tangent to the bottle)
		double dz_axis     = 0.0;   // m,  offset along the bottle long axis
		double dazi        = 0.0;   // rad, approach-azimuth rotation about the bottle axis
		double speed_scale = 1.0;   // multiplies the approach/insert speed caps
	};
	GraspPerturbation rep_perturb_{};         // active this rep (applied in compute_side_grasp_target)
	bool   probe_enabled_   = false;          // Controller.probe_variations
	double probe_pos_amp_   = 0.015;          // Controller.probe_pos_amp  [m]   half-range of dx/dz
	double probe_azi_amp_   = 0.15;           // Controller.probe_azi_amp  [rad] half-range of dazi
	double probe_speed_amp_ = 0.25;           // Controller.probe_speed_amp      ± fraction of nominal speed
	long   probe_index_     = 0;              // monotonic rep/attempt counter → Halton index
	int    rep_track_ticks_ = 0;              // Tracking cycles this attempt (convergence time)
	int    rep_attempts_    = 0;              // grasp attempts spent on the current rep
	// Watchdog: Tracking is the one grasp phase with no natural timeout, so an arm
	// jammed during the approach (e.g. wedged against the table) loops there forever.
	// If Tracking can't converge within this many cycles, treat it as a jam: teleport
	// the arm back to rest via Webots2Robocomp_setArmJointsInstant and retry/give up.
	// ~18 s at the 20 ms period — generously longer than any healthy approach.
	int    track_stuck_ticks_ = 0;
	static constexpr int TRACK_TIMEOUT_TICKS = 900;
	static constexpr int TRACK_NOPROGRESS_TICKS = 150;  // ~3 s of no approach progress ⇒ unreachable, give up fast
	double track_best_dist_   = 1e9;   // best (min) Tracking distance this attempt — for the no-progress abort
	int    track_noprog_ticks_ = 0;    // cycles since track_best_dist_ last improved
	// Controller.grasp_align_tol_deg: final-orientation tolerance to COMMIT the grasp. Relaxing
	// it lets a yaw-free approach commit instead of oscillating to perfect a few last degrees.
	double grasp_align_tol_rad_ = 0.14;  // ≈8° (was a fixed 5.7°); config overrides in degrees
	// A rep keeps the SAME perturbation across retries, so a hard probe point can
	// miss forever. Cap the retries: after this many misses, record the failure,
	// count the rep as a (failed) cycle and move on — keeps probe rounds bounded
	// and the dataset honest about which perturbations are ungraspable.
	static constexpr int MAX_REP_ATTEMPTS = 3;
	double rep_commit_epos_ = 0.0;            // standoff terminal position error at commit
	double rep_commit_eang_ = 0.0;            // standoff terminal orientation error at commit
	std::ofstream dataset_;                   // per-rep CSV sink (Controller.dataset_path)
	bool   dataset_open_    = false;
	// Joint-level actuation log: commanded vs measured q̇ per cycle, the clean signal
	// for identifying the actuation model (G, τ) — Cartesian v_cmd is only a speed cap
	// and gave a nonsense gain. last_q_dot_cmd_ holds the previous cycle's command, so
	// pairing it with this cycle's measured velocity aligns command→response.
	std::array<double, Kinematics::N_ARM_JOINTS> last_q_dot_cmd_{};
	std::ofstream joint_log_;                 // Controller.joint_log_path
	bool   joint_log_open_  = false;
	// Per-cycle EE-speed log over the WHOLE pick-place (Controller.fluid_log_path),
	// for offline fluidity metrics: SPARC (spectral arc length) + stop-count vs
	// confidence — the "more fluid as skill rises" measurement.
	std::ofstream fluid_log_;
	bool   fluid_log_open_  = false;
	std::optional<Eigen::Vector3d> fluid_prev_pos_;
	long   joint_log_cycle_ = 0;
	void   begin_rep_probe();                                       // new attempt → next perturbation
	void   log_rep_outcome(bool success, double rise, double xy_gap); // one CSV row

	// ── Retreat sub-skill (leaving the placed bottle) — learnable like the grasp ──────
	// A side grasp leaves the open fingers wrapped around the bottle; backing out badly
	// (before the fingers are open, or too fast) drags and topples it. The retreat is a
	// PURE TRANSLATION straight back along the gripper's own forward axis (the live tool
	// +Z from FK, horizontal = perpendicular to the upright bottle), orientation held —
	// any deviation from that axis sweeps a finger across the bottle. It also waits for
	// the gripper to actually OPEN first. The free outcome signal is the bottle's tilt
	// AFTER the retreat, so the per-rep probe can perturb the timing/speed params and a
	// round LEARNS the tip-free retreat. The direction is fixed (the gripper axis), so
	// what is explored is when-to-move (open threshold) and how-fast.
	double retreat_speed_      = 0.06;   // Controller.retreat_speed      gentle back-off speed [m/s] (NOT skill-boosted)
	double gripper_open_conf_  = 0.85;   // Controller.gripper_open_conf  gs.opening ≥ this ⇒ fingers clear → safe to move
	// Controller.release_ticks: cycles the gripper opens-in-place before the retreat latches.
	// The retreat keeps the gripper open and backs off SLOWER than the fingers open, so the
	// two overlap — this only needs to be long enough for the fingers to start clearing the
	// bottle. Lower ⇒ quicker place tail. Was a fixed 25 (~0.5 s); now tunable for the sweep.
	int    release_ticks_      = 8;      // ~0.16 s
	struct RetreatPerturbation { double dspeed = 0.0; double dopen = 0.0; };
	RetreatPerturbation retreat_perturb_{};   // active this rep
	bool   probe_retreat_      = false;  // Controller.probe_retreat    explore retreat params per rep
	double probe_rspeed_amp_   = 0.50;   // Controller.probe_rspeed_amp ± fraction on retreat speed
	double probe_open_amp_     = 0.10;   // Controller.probe_open_amp   ± on the gripper-open threshold
	// Retreat frame latched at PlaceReleasing→PlaceRetreating from the LIVE tool pose,
	// so the move is a fixed pure back-translation along the real gripper forward axis.
	Eigen::Vector3d retreat_target_pos_{0.0, 0.0, 0.0};
	Eigen::Vector3d retreat_z_des_{0.0, 0.0, 1.0};
	Eigen::Vector3d retreat_x_des_{1.0, 0.0, 0.0};
	Eigen::Vector3d place_world_xy_{0.0, 0.0, 0.0};  // last place spot (world), for the retreat dataset
	std::ofstream retreat_log_;          // Controller.retreat_log_path (params → post-retreat tilt)
	bool   retreat_log_open_   = false;
	void   log_retreat_outcome(double tilt_deg, bool tipped);   // one retreat-dataset row
	// Bottle tilt from vertical [rad], from the live world long axis (0 = upright).
	double bottle_tilt_rad() const
	{ return std::acos(std::clamp(std::abs(bottle_axis_world_.normalized().z()), 0.0, 1.0)); }

	// ── Precision re-weighting: the novice→skilled (closed→open-loop) transition ──
	// The grasp target the controller drives toward is the precision-weighted fusion
	// of the live observation (x_obs, precision Π_s = 1/σ_obs²) and the model belief
	// (x_model, precision Π_m). `confidence_` is the normalised model precision
	// Π_m/(Π_m+Π_s) ∈ [0,1]; it rises with each confirmed grasp (learning) and is
	// PERSISTED across rounds. High confidence ⇒ (a) sample the bottle far less often
	// (the EFE observation schedule: a fresh look gains ~no information once the model
	// predicts well) and (b) move faster (trust the model, less corrective caution) —
	// which is precisely the measured "skill". A large prediction error on a sample
	// (surprise) drops confidence and re-engages feedback. All OFF unless enabled.
	bool   precision_reweighting_ = false;   // Controller.precision_reweighting
	double perception_noise_std_  = 0.0;     // Controller.perception_noise_std  σ_obs [m] (sim2real)
	double confidence_            = 0.0;     // Π_m/(Π_m+Π_s), persisted
	double conf_gain_             = 0.15;    // += per confirmed grasp
	double conf_decay_            = 0.5;     // ×= per miss / surprise
	int    skilled_sample_period_ = 12;      // cycles between observations at full confidence
	double speed_conf_gain_       = 0.6;     // approach-speed boost at full confidence
	double surprise_gate_m_       = 0.05;    // obs−belief beyond this [m] ⇒ surprise → re-engage
	// Skill-scheduled FSM geometry/timing. The same confidence scalar that thins the
	// observation stream also reshapes the *action*: as it rises the controller commits
	// sooner (shorter settle dwell) and reaches more directly (the slow standoff→grasp
	// crawl shrinks), turning the novice's staccato reach-settle-creep into one fluid
	// motion. That is where the measured "skill" turns into shorter cycle time — sensing
	// less is necessary but not sufficient; the FSM gates were the real time sink.
	double standoff_collapse_     = 0.0;     // Controller.standoff_collapse: slide the Tracking waypoint this fraction toward the grasp point at full confidence (0 = always full standoff)
	double insert_conf_gain_      = 0.0;     // Controller.insert_conf_gain: insert speed ×(1+this·confidence)
	std::string confidence_path_;            // Controller.confidence_path (persist the learned skill)
	// Per-rep working state for the fusion + metrics.
	SideGraspTarget belief_grasp_{};         // x_model: committed/fused grasp belief this rep
	bool   belief_valid_     = false;
	int    cycles_since_obs_ = 0;
	int    obs_count_rep_    = 0;            // observations taken this rep (the metric)
	double rep_t0_           = 0.0;          // rep start wall-clock [s]
	std::ofstream metrics_;                  // Controller.metrics_path
	bool   metrics_open_     = false;
	void   load_confidence();
	void   save_confidence();
	// Effective skill ∈ [0,1]: the confidence when re-weighting is enabled, else 0
	// (novice) so every schedule below collapses to the original constants when the
	// feature is off — the changes are inert until precision_reweighting is on.
	double skill_c() const { return precision_reweighting_ ? confidence_ : 0.0; }
	// A transport/approach speed cap boosted with skill (trust the model ⇒ move faster).
	double skilled_speed(double base) const { return base * (1.0 + speed_conf_gain_ * skill_c()); }

	// Bottle respawn (needs the bridge setObjectPose). Teleport a fresh, upright
	// bottle to a deterministic pickup spot at the start of each rep, so a round
	// survives a toppled/fallen/placed-away bottle AND each rep picks from a known,
	// controlled location (clean experiment design). A mid-rep topple still becomes
	// a recorded MISS (bounded by the give-up cap) — the next rep re-stands it.
	void   respawn_bottle(double x, double y);                       // upright, on the table
	bool   respawn_each_rep_ = false;            // Controller.respawn_each_rep
	Eigen::Vector2d last_spawn_xy_{0.0, -0.14}; // last commanded spawn (world x,y)
	// Pickup spawn box (world frame), sampled with an R2 low-discrepancy sequence
	// for uniform 2-D coverage. Sized to the COMFORTABLY reachable right half of the
	// table: arm base is at world (-0.623,-0.023), table top z≈0.742, Gen3 reach
	// ≈0.90 m — so the grasp distance here stays ≲0.83 m (the old box pushed bottles
	// out to the 0.90 m limit). Right side = negative world Y.
	// Cover the arm's half of the table within reach (Webots world frame). The table is
	// 1.8 m wide (X) × 1.0 m deep (Y); the arm lives on the negative-Y half, so its half
	// spans the full ±X width and ~0.5 m of −Y depth. This box is generous in X (the
	// reach band trims the unreachable wings) and stops just shy of the −Y table edge so
	// no bottle spawns off the table. R2 sampling + the reach band fill the reachable half.
	static constexpr double SPAWN_X_MIN = -0.55, SPAWN_X_MAX = 0.55;   // ±X — full reachable width
	static constexpr double SPAWN_Y_MIN = -0.48, SPAWN_Y_MAX = -0.02;  // the arm's −Y half, inside the table edge
	// Reach band from the arm base (world XY): reject spawns nearer than MIN (almost on
	// the column) or past MAX (ungraspable far corner). MAX ≈ the comfortable working
	// edge — the bottle, not the closer standoff, is the farthest the EE must reach.
	static constexpr double SPAWN_REACH_MIN_M = 0.45;
	static constexpr double SPAWN_REACH_MAX_M = 0.86;
	// Continuous fall detection: bottle long axis tilted past this from vertical
	// (or dropped below the table) ⇒ it toppled; abort the current grasp phase.
	static constexpr double FALL_TILT_RAD = 0.60;   // ≈34°
	// Grasp frame latched at Tracking→Inserting so gripper/bottle contact can't
	// make the target chase its own disturbance.
	SideGraspTarget latched_grasp_{};
	Eigen::Vector3d lift_target_{};       // latched grasp_pos + LIFT_HEIGHT·ẑ_world
	Eigen::Vector3d place_pos_{};         // random place point on the table (robot frame)
	Eigen::Vector3d place_hover_{};       // place_pos_ + LIFT_HEIGHT·ẑ_world
	// Place orientation: tool +Z re-pointed radially toward the place spot with
	// tool +Y kept up (bottle upright). Yaw is free, so this avoids the arm
	// contorting to hold the original grasp yaw across the table.
	Eigen::Vector3d place_z_des_{};
	Eigen::Vector3d place_x_des_{};
	std::mt19937    rng_{std::random_device{}()};

	// Tilt reflex (protective): watch the live bottle tilt during the approach/
	// contact phases; if it starts to go over, reopen and back off to this
	// retract target, let it settle, then re-track. reflex_count_ caps retries.
	Eigen::Vector3d retract_target_{};
	int             reflex_count_   = 0;
	int             retract_ticks_  = 0;
	double          bottle_z_at_lift_start_ = 0.0;  // bottle height when the lift began (for lift-confirm)
	int             grasp_force_ticks_      = 0;     // consecutive cycles with force over threshold

	static constexpr int    GRASP_SETTLE_TICKS   = 8;     // converged cycles to commit
	static constexpr double GRASP_ALIGN_TOL_RAD  = 0.10;  // ≈5.7° orientation tolerance to commit
	static constexpr double INSERT_VEL_MS        = 0.05;  // gentle approach speed for soft contact
	// Finger force → object MAY be held. Coarse contact gate only: an empty close
	// still registers ~1.3–2.2 N (fingers loading against each other), while a real
	// grasp reaches tens of N, so this is set above the empty-close band. It is NOT
	// trusted on its own — the authoritative check is the post-lift confirmation
	// below (the bottle must actually rise with the gripper).
	static constexpr float  GRASP_FORCE_THRESH   = 3.0f;
	static constexpr int    GRASP_FORCE_HOLD_TICKS = 3;   // force must persist this many cycles (debounce transients)
	// Insert-stop force: end the approach-into-bottle on the LIGHTEST touch so the
	// gripper stops a hair into contact and closes, instead of shoving the bottle
	// over (the side-insert was reaching f≈1.6 N — enough to tip it). Much smaller
	// than GRASP_FORCE_THRESH, which still gates the actual grasp confirmation.
	static constexpr float  INSERT_TOUCH_FORCE   = 0.3f;
	static constexpr int    CLOSING_TIMEOUT_TICKS = 100;  // ~2 s closing w/o force → miss
	static constexpr int    INSERT_TIMEOUT_TICKS  = 150;  // ~3 s seating/centring → miss (tactile re-centre)
	static constexpr double TIP_REFLEX_STEP_M     = 0.005; // lateral shift per tip-contact cycle
	static constexpr double TIP_REFLEX_MAX_M      = 0.05;  // cap on the accumulated reflex correction
	static constexpr double TIP_REFLEX_BACKOFF_M  = 0.02;  // retreat along −approach while a tip is in contact
	static constexpr double LIFT_HEIGHT_M        = 0.12;  // how high to pick the bottle
	// Authoritative grasp confirmation (ground truth, sensor-independent). After the
	// lift completes, the bottle's world pose (live via getObjectPose) must show that
	// it actually came up WITH the gripper, otherwise the close grabbed air. Two
	// conditions, both must hold:
	//   1. the bottle rose at least this fraction of the commanded lift height, and
	//   2. the bottle is still co-located with the tool (it didn't slip out).
	// If either fails the grasp is a MISS: reopen and re-track, don't place an empty
	// gripper and don't count the cycle.
	static constexpr double LIFT_CONFIRM_RISE_M  = 0.06;  // bottle must rise ≥ this (≈½ the lift)
	static constexpr double LIFT_CONFIRM_HOLD_M  = 0.12;  // bottle centre must stay within this of the tool
	// Tilt reflex. DISABLED for now: with the heavy bottle it can't tip, and the
	// reflex was looping retract→re-track whenever it saw an already-leaning
	// bottle (the "back and forth"). Flip to true to re-enable.
	static constexpr bool   TILT_REFLEX_ENABLED = false;
	static constexpr double TILT_REFLEX_RAD   = 0.14;  // ≈8° of bottle tilt from vertical → abort
	static constexpr double RETRACT_DIST_M    = 0.12;  // back off this far along −approach
	static constexpr int    RETRACT_SETTLE_TICKS = 30; // ~0.6 s backed off to let it settle
	static constexpr int    MAX_REFLEXES      = 3;     // give up (→ hold) after this many tips
	// Place targets: a box on the RIGHT side of the table (WORLD frame, m), the
	// dexterous zone of this right arm (left/centre is the left arm's job). The
	// controller runs in world frame, so up = +Z and the table is horizontal; the
	// EE keeps the grasp height so the bottle base re-seats on the table top.
	static constexpr double PLACE_X_MIN = 0.06, PLACE_X_MAX = 0.30;   // forward extent (world +X) — pushed out
	static constexpr double PLACE_Y_MIN = -0.38, PLACE_Y_MAX = -0.05; // right side (world −Y) — widened
	static constexpr double PLACE_MIN_MOVE_M    = 0.20;  // ≥ this far from the pick spot (more distant placements)
	// Reachability filter on the chosen place EE: the box now extends past the arm's
	// reach, so reject samples whose set-down point is farther than this from the arm
	// base. 0.90 m is the observed working edge (the 10-fold round placed reliably out
	// to ~0.90 m). Keeps "more distant" spots that are still grabbable.
	static constexpr double PLACE_REACH_MAX_M   = 0.90;
	static constexpr int    PLACE_TIMEOUT_TICKS = 300;   // ~6 s safety per place move
	static constexpr int    RELEASE_TICKS       = 25;    // ~0.5 s to let the bottle go
	// Safer set-down: open the fingers only once the bottle is both at the table
	// (position converged) AND upright, held for a short dwell — so a fast transport or
	// overshoot can't release it mid-air or while it leans (which would topple it).
	static constexpr int    PLACE_SETTLE_TICKS    = 5;     // settled+upright cycles to confirm before opening
	static constexpr double PLACE_UPRIGHT_TOL_RAD = 0.15;  // ≈8.6° bottle tilt from vertical still "upright"
	static constexpr double PLACE_ON_TABLE_M      = 0.04;  // bottle base within this of the table top ⇒ "resting" → release
	static constexpr double PLACE_ORIENT_GAIN     = 0.4;   // relaxed gripper-orientation pull while placing (vs 1.0 to pick)
	// Leave the placed bottle by backing straight out along the approach axis (−tool +Z,
	// horizontal) instead of lifting up over it: a shorter move that pulls the open
	// fingers clear sideways with no risk of catching the bottle top and tipping it.
	static constexpr double PLACE_RETREAT_DIST_M  = 0.14;  // horizontal back-off off the bottle

	// Gripper command sent every compute() cycle through
	//   kinovaarm_proxy->setGripperPos(gripper_command_).
	// 1.0 = fully open, 0.0 = fully closed (matching the bridge's KinovaArm
	// implementation, which internally inverts to motor position). The
	// upcoming grasp FSM (Approach / Grasp / Lift) just writes to this.
	// Default open so the controller starts and idles in a safe, harmless
	// configuration.
	float gripper_command_ = 1.0f;
	bool  gripper_proxy_warned_ = false;

	// DSR sub-APIs used by the bottle-approach loop.
	std::unique_ptr<DSR::RT_API>         rt_api_;
	std::unique_ptr<DSR::InnerEigenAPI>  inner_eigen_api_;

	// World-frame state. The controller runs in WORLD coordinates: kinematics_'s
	// base_tf_ is set to the arm's world pose (from the Webots P3Bot node) so FK
	// reports world coords, and grasp/place targets are world poses pulled from
	// Webots. arm_base_world_ is T_world_armbase; bottle_pos/axis_world_ are the
	// latest bottle world position and long axis, refreshed each cycle.
	Eigen::Isometry3d arm_base_world_   = Eigen::Isometry3d::Identity();
	Eigen::Vector3d   bottle_pos_world_ {0.0, 0.0, 0.0};
	Eigen::Vector3d   bottle_axis_world_{0.0, 0.0, 1.0};
	double            bottle_radius_m_  = 0.035;  // from DSR bottle width_m (diameter/2); for the approach obstacle
	double            bottle_height_m_  = 0.22;   // from DSR bottle height_m
	// Controller.bottle_obstacle: treat the target bottle as a soft cylinder obstacle for the
	// TOOL during the FIRST (Tracking) phase only, so the gripper rounds to the standoff
	// without clipping/tipping it. Off from Inserting on (the grasp needs to move in).
	bool              bottle_obstacle_        = false;
	double            bottle_obstacle_margin_ = 0.04;  // [m] repel within this of the bottle surface
	Eigen::Isometry3d table_world_      = Eigen::Isometry3d::Identity();  // T_world_table
	double            table_top_z_     = 0.74;      // world z of the table surface (for hand-table soft constraint)
	bool              scene_world_valid_ = false;   // table/bottle world poses populated
	bool              base_tf_set_      = false;
	// Install base_tf_ from the live P3Bot world pose; idempotent, safe to retry.
	void refresh_arm_base_world();

	// Webots scene-object DEF/name strings (find_scene_node resolves either).
	static constexpr const char* WEBOTS_BOTTLE_DEF = "bottle";
	static constexpr const char* WEBOTS_TABLE_DEF  = "table";
	static constexpr const char* WEBOTS_ROBOT_DEF  = "P3Bot";        // body/mount node
	static constexpr const char* WEBOTS_ARM_DEF    = "kinova_arm_r"; // arm base_link (child of P3Bot)

	// Pull robot+table+bottle world poses from Webots and refresh the DSR RT
	// tree: robot→table (table expressed in the arm-base frame, so the viewer
	// and grasp targeting match the real Webots mount height/rotation) and
	// table→bottle. Replaces the stale static robot→table value in kinova2.json.
	void update_bottle_pose_in_dsr();

	// Build a side-grasp target: read bottle pose in the DSR "robot" frame via
	// InnerEigenAPI, derive a horizontal approach axis perpendicular to the
	// bottle's vertical axis, and back the EE off by APPROACH_STANDOFF_M.
	// Returns nullopt if the chain isn't resolvable or the bottle sits on the
	// robot's vertical axis (radial direction undefined).
	std::optional<SideGraspTarget> compute_side_grasp_target();

	// Project the 8 table corners and the bottle origin into the robot frame
	// via InnerEigenAPI, then push them to the viewer for drawing.
	void update_viewer_scene_objects();

signals:
	//void customSignal();
};

#endif
