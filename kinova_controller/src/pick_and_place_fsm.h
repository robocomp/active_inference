/*
 *    Copyright (C) 2026 by pbustos
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 */

#ifndef PICK_AND_PLACE_FSM_H
#define PICK_AND_PLACE_FSM_H

#include "specificworker.h"
#include "kinematics.h"
#include "efe_gradient.h"

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <optional>
#include <random>
#include <fstream>
#include <string>
#include <utility>

class ConfigLoader;

/**
 * \brief Hierarchical pick-and-place state machine for the Kinova arm.
 *
 * Owns the whole manipulation behaviour layer: the EFE/QP controller primitives
 * (build_efe_params, efe_drive), the grasp-target perception (compute_side_grasp_target,
 * the capability/reach map), and the grasp→place→retreat state machine, plus the
 * skill-learning / probe / retreat instrumentation.
 *
 * The first phase, Tracking, is the validated QP approach-to-bottle (full-frame camera-up,
 * deadband hold). For now `step()` runs Tracking and HOLDS there — it does not commit to
 * Inserting — so the current functionality (reach the bottle pose and track it) is preserved
 * while the downstream grasp/place phases stay wired for later.
 *
 * It is a friend of SpecificWorker and reaches the robot-I/O + scene-perception services
 * (kinematics, proxies, live bottle/arm/table world poses, viewer) through a reference `w_`.
 */
class PickandPlaceFSM
{
public:
    PickandPlaceFSM(SpecificWorker& w, const ConfigLoader& cfg);

    // Run the current phase for this control cycle.
    void step(const std::array<double, Kinematics::N_ARM_JOINTS>& q,
              const Eigen::Vector3d& ee_position);

    // Current grasp sub-phase name (host diagnostics, e.g. the side-force probe).
    const char* current_grasp_phase_name() const;

    // Lifecycle hooks called by SpecificWorker's outer state machine.
    void start();                            // run engaged → begin a fresh pick sequence
    void reset();                            // run disengaged → reset the FSM
    SpecificWorker::Phase on_rest_reached(); // homing settled → restart pick or park
    void maybe_compute_reach_map();          // one-shot capability-map precompute

private:
    SpecificWorker& w_;

    // ── Grasp FSM phases ─────────────────────────────────────────────────────
    enum class GraspPhase { Tracking, Inserting, Closing, Leveling, Lifting,
                            PlaceMoving, PlaceLowering, PlaceReleasing, PlaceRetreating,
                            Retracting };
    GraspPhase grasp_phase_ = GraspPhase::Tracking;
    // Controller.stop_after_grasp: halt the FSM once the grasp is confirmed (Closing) and HOLD it —
    // no Lifting/place. The "stop at grasp" experiment mode for testing the perceived-bottle grasp.
    bool stop_after_grasp_ = false;
    // Controller.stop_after_lift: the experiment goal — once the LIFT is confirmed (bottle risen),
    // count the success and home for the next rep, skipping the place/retreat phases entirely.
    bool stop_after_lift_  = false;
    bool grasp_held_       = false;

    // ── Motion SEGMENTS (partition of c) ──────────────────────────────────────
    // Skill is partitioned per motion segment, not one global scalar: each segment
    // learns its OWN precision c_k = Π_m[k]/(Π_m[k]+Π_s) from its OWN outcome, so the
    // safe legs (transport) cruise while the risky legs (set-down) stay careful — the
    // decoupling a single c can't express. Commit events (close/release) stay hard.
    enum Segment { SEG_APPROACH, SEG_INSERT, SEG_LIFT, SEG_PLACE, SEG_RETREAT, N_SEG };
    static Segment seg_of(GraspPhase p);
    Segment cur_seg_ = SEG_APPROACH;   // segment driving skill_c() this cycle

    // ── Grasp-target descriptor ──────────────────────────────────────────────
    struct SideGraspTarget
    {
        Eigen::Vector3d stand_off_pos;
        Eigen::Vector3d grasp_pos;
        Eigen::Vector3d z_tool_des;
        Eigen::Vector3d x_tool_des;
        Eigen::Vector3d up_axis{0, 0, 1};
        bool            top_down = false;
    };
    struct ReachScore { bool feasible; double manip; double col_clear; double table_clear; };

    // ── Generative lift confirm (two-hypothesis sensory model) ────────────────
    // Replaces the discriminative rise/via/weight thresholds with a sequential
    // log Bayes factor: predict each sensory channel under H_grasped vs H_empty,
    // accumulate precision-weighted log-evidence, decide when it crosses ONE bound.
    // Predictions are parameterized by the EE's OWN vertical rise (ee_rise, from FK),
    // not by a fixed distance — so the decision boundary scales with how far we lift.
    // An occluded/absent channel inflates its σ → it abstains (the false-SLIP fix made
    // principled: a frozen perceived-z no longer votes "empty", the grip/pad channels do).
    struct LiftHypothesis
    {
        // Baselines latched at the close→lift transition.
        double ee_z0 = 0, bottle_z0 = 0, aperture0 = 0, pad0 = 0;
        double log_evidence = 0.0;          // + favours grasped, − favours empty
        // Channel precisions (σ): grasped-world is tight, empty-world is loose. Tunable
        // from the logged dataset; c_lift can tighten σ_track for a skilled lift.
        double sigma_track    = 0.012;      // m: bottle_rise about its prediction (held world)
        double sigma_aper_g   = 0.03;       // aperture frozen when the body blocks the jaws
        double sigma_aper_e   = 0.10;       // aperture wanders when there is nothing between them
        double sigma_pad      = 2.0;        // N: min-pad load about its prediction
        double decide_bound   = 3.0;        // log(~20) → 20:1 odds; the ONE decision knob

        enum class Verdict { Undecided, Grasped, Empty };

        void latch(double ee_z, double bottle_z, double aperture, double pad)
        { ee_z0 = ee_z; bottle_z0 = bottle_z; aperture0 = aperture; pad0 = pad; log_evidence = 0.0; }

        // One Gaussian channel's log-likelihood ratio logN(obs|μ_g,σ) − logN(obs|μ_e,σ).
        static double llr(double obs, double mu_g, double mu_e, double sigma)
        {
            const double inv2s2 = 1.0 / (2.0 * sigma * sigma);
            return (std::pow(obs - mu_e, 2) - std::pow(obs - mu_g, 2)) * inv2s2;
        }

        Verdict update(double ee_z, double bottle_z, double aperture, double min_pad,
                       bool bottle_observable)
        {
            const double ee_rise     = ee_z - ee_z0;
            const double bottle_rise = bottle_z - bottle_z0;
            // 1) TRACK: held bottle rises WITH the EE (μ_g = ee_rise); empty bottle stays (μ_e = 0).
            //    Occluded → σ huge → the channel abstains.
            const double s_track = bottle_observable ? sigma_track : 1e3;
            log_evidence += llr(bottle_rise, ee_rise, 0.0, s_track);
            // 2) GRIP-CONSTRAINT: held → jaws blocked at the body, aperture frozen (tight σ);
            //    empty/cam-open → aperture wanders (loose σ). Variance-ratio test on the increment.
            const double daper = aperture - aperture0;
            log_evidence += std::pow(daper, 2)
                          * (1.0 / (2 * sigma_aper_e * sigma_aper_e) - 1.0 / (2 * sigma_aper_g * sigma_aper_g));
            // 3) PAD LOAD: a SLIP detector only — pads are loaded from t=0 (it's the precondition of
            //    lifting), so loaded pads are NOT evidence the bottle rose. One-sided: contribute only
            //    NEGATIVE (toward empty) when the force DROPS below pad0; never positive. Otherwise the
            //    +pad0²/2σ² term confirms instantly at rise=0. (Grip channel is already ≤0 by construction.)
            log_evidence += std::min(0.0, llr(min_pad, pad0, 0.0, sigma_pad));

            if (log_evidence >  decide_bound) return Verdict::Grasped;
            if (log_evidence < -decide_bound) return Verdict::Empty;
            return Verdict::Undecided;
        }
    };
    LiftHypothesis lift_hyp_{};
    // Controller.generative_lift: A/B toggle. true ⇒ the LiftHypothesis confirm above;
    // false ⇒ the legacy rise/via/weight threshold confirm (the validated baseline).
    bool   generative_lift_ = false;
    // Eased-rise scale set by the cam-open active correction (re-clamp + slow the lift so
    // the grip re-seats), reset to 1.0 at each latch. Multiplies the Lifting v_app.
    double lift_vel_scale_  = 1.0;
    // Aperture increase over aperture0 that trips the cam-open correction before the verdict.
    static constexpr double CAM_OPEN_RESIDUAL = 0.06;

    // ── Generative insert (palm-OCCUPANCY model) ──────────────────────────────
    // MEASURED: the palm DistanceSensor is a bimodal/noisy ray — ≈0.15 (max range) when it MISSES the
    // bottle, 0.03-0.09 when it SEES it in the throat. There is NO smooth 1:1 closing trajectory, so a
    // residual/trajectory model produces garbage (the v1 regression: it slowed every insert). The right
    // generative read is OCCUPANCY of the throat: palm low ⇒ bottle in the throat (on-axis); palm at
    // max while close to the grasp point ⇒ throat empty (off-axis). Drives the descent SPEED: fast
    // while the bottle is genuinely far (far from the grasp point AND palm not yet acquired = nothing
    // close to disturb), slow through the whole near zone (gentle seat whether on- or off-axis). The
    // directional reflex stays contact-driven (single ray gives "something/nothing", not the side).
    // The discriminating signal is the SENSORIMOTOR CONTINGENCY: the close-rate = −d(palm)/d(advance)
    // over a window long enough to beat the per-tick noise. On-axis (bottle in the throat ahead) ⇒
    // close_rate ≈ 1 (the palm closes 1:1 with my forward motion); off-axis / not-acquired ⇒ ≈ 0 (I
    // advance but the palm doesn't respond). The derivative cancels the bimodal offset, so it's robust
    // where the absolute reading isn't. Control law: advance FAST while the palm is closing productively
    // (rate high), FIXED-GENTLE otherwise (about to seat, or off-axis — both want a soft touch). Speeds
    // are FIXED (not the skill-ramped insert_vel, which had gone to ~0.095 = not gentle).
    bool   generative_insert_   = false;   // Controller.generative_insert (A/B vs the threshold insert)
    double insert_palm_near_    = 0.08;    // m: palm reading below this ⇒ bottle acquired in the throat
    double insert_near_zone_    = 0.05;    // m: within this EE-distance of the grasp point ⇒ never go fast
    double insert_fast_vel_     = 0.06;    // m/s while the palm is closing productively + bottle far
    double insert_gentle_vel_   = 0.03;    // m/s near the bottle / when the palm is NOT closing (soft touch)
    double insert_rate_gate_    = 0.4;     // close_rate above this ⇒ productive closing (advance fast)
    Eigen::Vector3d insert_win_ee0_ = Eigen::Vector3d::Zero();   // window baseline EE
    double insert_win_palm0_    = 0.0;     // window baseline palm reading
    double insert_close_rate_   = 0.0;     // EMA of −d(palm)/d(advance)
    int    insert_throat_miss_  = 0;       // debounce: cycles at grasp depth, palm not closing (off-axis)
    static constexpr double INSERT_RATE_WINDOW_M = 0.012;   // advance per close-rate estimate (beats palm noise)

    // ── Phase logic ──────────────────────────────────────────────────────────
    void run_tracking(const std::array<double, Kinematics::N_ARM_JOINTS>& q,
                      const Eigen::Vector3d& ee_position);   // = validated QP approach + hold
    void run_grasp_phases(const std::array<double, Kinematics::N_ARM_JOINTS>& q,
                          const Eigen::Vector3d& ee_position); // Inserting..Retreating (dormant)

    // ── Controller primitives ────────────────────────────────────────────────
    EFEParams build_efe_params(const Eigen::Vector3d& z_des,
                               const Eigen::Vector3d& x_des,
                               double v_app) const;
    std::pair<double,double> efe_drive(
        const std::array<double, Kinematics::N_ARM_JOINTS>& q,
        const Eigen::Vector3d& ee_position,
        const Eigen::Vector3d& target,
        const Eigen::Vector3d& z_des,
        const Eigen::Vector3d& x_des,
        double v_app,
        std::optional<Eigen::Vector3d> blend_next = std::nullopt,
        double orient_gain = 1.0);

    // ── Grasp-target perception + capability map ──────────────────────────────
    std::optional<SideGraspTarget> compute_side_grasp_target();
    ReachScore predict_reach(const Eigen::Vector3d& pos,
                             const Eigen::Vector3d& z_des, const Eigen::Vector3d& x_des,
                             const std::array<double, Kinematics::N_ARM_JOINTS>& seed);
    void  compute_reach_map();
    float reach_lookup(double x, double y) const;
    void  sample_place_spot();

    // ── Skill / probe / retreat instrumentation ──────────────────────────────
    void begin_rep_probe();
    void load_confidence();
    void save_confidence();
    void log_rep_outcome(bool success, double rise, double xy_gap);
    void log_retreat_outcome(double tilt_deg, bool tipped);
    void miss_or_give_up(const std::string& reason);
    std::pair<bool,bool>  tip_contacts() const;
    std::pair<float,float> tip_forces() const;
    std::pair<float,float> pad_forces() const;   // (lforce, rforce) — grip strength on the body
    // Raw fingertip force-3d vectors {Fx,Fy,Fz} (N) per pad, in the finger-sensor frame (= tool axes:
    // X closing, Y pad-width, −Z approach). The lateral (X,Y) components drive the shear-based centring.
    std::pair<Eigen::Vector3f, Eigen::Vector3f> pad_force_vecs() const;
    float gripper_force() const;
    float palm_distance() const;   // palm proximity (m) down the throat; 0 if sensor absent
    float gripper_opening() const; // aperture [0,1] (0=closed, 1=open)
    bool  via_reached(double e_pos);
    double bottle_tilt_rad() const;
    // Per-segment skill. c_seg(k) = Π_m[k]/(Π_m[k]+Π_s) (force_confidence pins all).
    // skill_c() is the CURRENT segment's c (cur_seg_ is set at the top of each phase),
    // so every existing knob that reads skill_c()/skilled_speed() is now segment-local.
    // Controller.global_confidence: A/B baseline. true ⇒ all segments share a single Π_m
    // (index 0), so c is one global scalar driven by every outcome (a miss anywhere deflates
    // the whole skill); false ⇒ the per-segment partition. Per-deposit evidence is scaled by
    // 1/N_SEG in global mode so the per-episode accumulation rate matches the partition.
    bool   global_confidence_ = false;
    // ── Ablation toggles (WAF reviewer #3/#4/#7) ──────────────────────────────────────────────
    // anticipation_: gate the lift-aware grasp azimuth sweep AND the approach→insert handoff
    //   (orient/align conf gains). false ⇒ greedy geometric grasp + tight gate = the non-
    //   anticipatory baseline that chronically fails the lift (isolates anticipation's reliability
    //   contribution from precision's speed contribution — reviewer #4).
    // success_rate_baseline_: drive EVERY skill knob from a global Beta(1,1) success-rate estimator
    //   instead of model precision — the apples-to-apples test of "what does precision buy over a
    //   success score?" (#2). succ_count_/attempt_count_ accumulate per attempt within a round.
    bool   anticipation_          = true;
    bool   success_rate_baseline_ = false;
    long   succ_count_ = 0, attempt_count_ = 0;
    double c_seg(Segment s) const
    {
        if (force_confidence_ >= 0.0) return force_confidence_;
        const int k = global_confidence_ ? 0 : static_cast<int>(s);
        return pi_m_[k] / (pi_m_[k] + pi_s_);
    }
    double skill_c() const
    {
        if (success_rate_baseline_)
            return (1.0 + succ_count_) / (2.0 + attempt_count_);   // Beta(1,1) posterior-mean success rate
        return precision_reweighting_ ? c_seg(cur_seg_) : 0.0;
    }
    double skilled_speed(double base) const { return base * (1.0 + speed_conf_gain_ * skill_c()); }
    // Free-space wrist slew rate: skilled → faster reorientation DURING the approach travel, so
    // the gripper arrives already oriented (no asymptotic orientation "crawl" after the EE
    // reaches the standoff). Safe because the reorientation is in free space (no contact) — a
    // "fast where safe" leg. Scales gain_orient (and ω_max, in build_efe_params) for Tracking.
    double skilled_orient(double base) const { return base * (1.0 + orient_conf_gain_ * skill_c()); }
    double c_overall() const;                       // mean c over segments (for logs)
    void   deposit(Segment s, double quality);      // confirmed outcome → Π_m[s] += unit·quality
    void   deflate(Segment s);                      // surprise → Π_m[s] *= conf_decay

    // ── Approach (Tracking) constants + state ─────────────────────────────────
    static constexpr double APPROACH_STANDOFF_M = 0.12;
    static constexpr double REACH_TOLERANCE_M   = 0.02;
    // Fraction up the bottle axis to aim the grasp. 0.6 grasped the NECK of the tapered BeerBottle
    // (body is the lower ~half); lowered to grab the wide body. Config: Controller.grasp_height_frac.
    double bottle_grasp_height_frac_ = 0.35;
    // Insertion depth: tool origin relative to the bottle axis along the approach. The old
    // +15 mm BACK-off (to keep the body out of the void behind the then-back-less gripper) left
    // the bottle at the FINGERTIPS, where closing pinches it into a pivot/tilt. The proto now
    // has a PALM/backstop, so we can seat deep: 0 = tool at the bottle axis; NEGATIVE drives a
    // few mm PAST the axis to wedge the body against the palm/finger-base (the stable seat).
    // Re-testing from 0 now that the steady 20 ms loop no longer lurches the tool during insert.
    static constexpr double GRASP_DEPTH_BACKOFF_M    = 0.0;
    static constexpr double BOTTLE_TOP_GRASP_FRAC    = 0.88;
    bool   approach_hold_logged_ = false;
    bool   approach_respawn_done_ = false;
    // Lift-aware azimuth chosen once per pick episode (the symmetric-cylinder approach direction
    // whose manipulability stays high through BOTH the grasp and the straight-up lift).
    std::optional<Eigen::Vector3d> grasp_azimuth_z_;
    int    azimuth_retries_ = 0;   // real-μ standoff guard re-approaches this episode
    static constexpr double LIFT_MU_MIN        = 0.07;  // commit only above this manipulability
    static constexpr int    MAX_AZIMUTH_RETRIES = 4;
    static constexpr double AZIMUTH_RETRY_STEP  = 0.35; // ~20° azimuth rotation per re-approach
    long   ctrl_cycle_ = 0;
    double grasp_align_tol_rad_ = 0.14;
    // Skilled approach commits at a looser orientation gate (the insert finishes the alignment):
    // effective tol = grasp_align_tol_rad_·(1 + align_tol_conf_gain_·c_approach). 0 ⇒ fixed gate.
    double align_tol_conf_gain_ = 1.0;

    // ── Per-cycle motion monitor (debug elbow-table dives + approach/place oscillation) ──
    // A single throttled [mon] line, emitted from efe_drive (the common chokepoint every
    // moving phase passes through), so EVERY phase — not just Tracking — gets telemetry.
    // Shows the elbow/lowest-link height above the table (elbow-dive watch), the bottle
    // tilt, and oscillation signals (Δe sign-flips, measured EE speed, |q̇|).
    bool   monitor_log_    = true;
    int    monitor_period_ = 40;     // print every N efe_drive calls (100 Hz loop → ~2.5 Hz)
    long   mon_cycle_      = 0;
    double mon_prev_epos_  = 1e9;
    double mon_prev_time_  = 0.0;
    std::optional<Eigen::Vector3d> mon_prev_ee_;
    static const char* phase_name(GraspPhase p);
    void log_monitor(const std::array<double, Kinematics::N_ARM_JOINTS>& q,
                     const Eigen::Vector3d& ee_position, const Eigen::Vector3d& target,
                     double e_pos, double e_ang, const EFEDebug& dbg);

    // ── Controller config ─────────────────────────────────────────────────────
    bool   use_qp_ = false;
    double qp_redundancy_weight_ = 0.0;
    double force_confidence_ = -1.0;
    double blend_radius_ = 0.0;
    double blend_min_dist_ = 1e9;
    bool   use_preference_field_ = false;
    double field_prec_pass_ = 1.0, field_prec_stop_ = 30.0, field_prec_ref_ = 6.0, field_overlap_ = 0.06;
    bool   tactile_recenter_ = false;
    double recenter_gain_ = 0.02, recenter_sign_ = 1.0;
    // Compliant close: during Closing, micro-translate the gripper laterally to NULL the pad-force
    // asymmetry so the bottle centres between the pads instead of edge-gripping (the off-centre/canted
    // grip tips the light free-standing bottle on lift — the per-grasp ~50% slip that retries mask).
    // A force-balance servo coupled to the finger close; gentle (small accumulation rate) so it does
    // not drag the bottle over. Sign maps L−R asymmetry onto tool +X — flip if centring worsens it
    // (cf. recenter_sign history). Opt-in via Controller.compliant_close.
    bool   compliant_close_       = false;
    double compliant_close_gain_  = 0.00002;  // m lateral per (N·tick) of L−R pad asymmetry
    double compliant_close_max_   = 0.03;      // m cap on the accumulated lateral shift
    double compliant_close_speed_ = 0.05;      // m/s EE tracking of the shifting hold target
    double compliant_close_sign_  = 1.0;       // ±1: maps L−R asymmetry onto tool +X
    Eigen::Vector3d compliant_close_offset_{0.0, 0.0, 0.0};   // accumulated lateral shift this close
    static constexpr float COMPLIANT_CONTACT_MIN_N = 3.0f;    // min pad load to trust the asymmetry signal
    // Force-3d SHEAR close (opt-in A/B vs the motor-torque magnitude law above). The fingertip force-3d
    // vectors carry the LATERAL direction the scalar motor-torque difference is blind to. The finger-
    // sensor frame shares the tool axes (col0 = closing / sensor X, col1 = pad-width / sensor Y, col2 ≈
    // −approach). Summed over both pads the closing-axis (X) components of the two opposed pads cancel and
    // leave the residual asymmetry, while the pad-WIDTH (Y) shear ADDS — the off-centre direction along
    // the pad face. Decompose the net pad force onto col0 + col1 and shift the EE to null it, so centring
    // works in BOTH lateral axes, not just along the assumed tool +X. Sign still via compliant_close_sign_.
    bool   compliant_close_shear_      = false;
    double compliant_close_shear_gain_ = 0.004;    // m lateral per N of the PEAK first-contact lateral spike
    // The force-3d nets to ~0 in static grip but SPIKES at first contact (the slider reacts steady load,
    // not impact). Peak-HOLD the net lateral force over the close (world frame) and drive a decisive HELD
    // offset to null it — integrating the instantaneous value washes the transient out (it's ~0 between
    // spikes). Captured WITHOUT the motor-force gate: the spike precedes motor-torque build-up.
    Eigen::Vector3d compliant_close_side_peak_{0.0, 0.0, 0.0};
    // Generative compliant close: a forward model predicts SYMMETRIC pad loading on a centred bottle.
    // The prediction error is the FRACTIONAL asymmetry (lf−rf)/(lf+rf) — ≈±1 at FIRST contact (one pad
    // only), so the lateral correction is strongest EARLY, when forces are low and the light bottle is
    // easiest to recentre (the raw force-difference is weak there). And the commit is SYMMETRY-GATED:
    // only a centred (balanced + loaded) grip advances to Lifting — an off-centre grip is corrected or
    // rejected, never lifted (that was the slip cause). A/B via Controller.generative_close.
    bool   generative_close_   = false;
    double close_asym_gain_    = 0.0008;   // lateral m per (fractional-asym · tick)
    double close_balance_tol_  = 0.12;     // commit only when |lf−rf|/(lf+rf) < this (centred)
    bool   tip_reflex_ = false;
    double tip_reflex_offset_ = 0.0;
    // Deliberate tactile reflex sub-machine inside Inserting: on a fingertip touch, STOP, back off a
    // little (Retreat), step laterally away from the contacting finger (Shift), then re-descend
    // (Descend). reflex_force_thresh = the tip-FORCE level that counts as a touch — far more sensitive
    // than the bridge's binary 0.5 N bool, so a graze is caught before the light bottle tips.
    enum class InsertSub { Descend, Retreat, Shift };
    InsertSub insert_sub_ = InsertSub::Descend;
    Eigen::Vector3d reflex_retreat_target_{0.0, 0.0, 0.0};   // backed-off pose to reach during Retreat
    Eigen::Vector3d reflex_shift_target_{0.0, 0.0, 0.0};     // lateral pose at backed-off depth (Shift)
    int    insert_reflex_count_ = 0;     // reflex maneuvers this insert (capped by MAX_INSERT_REFLEXES)
    double reflex_force_thresh_ = 0.15;  // N — tip force that triggers the reflex (configurable)
    // Lateral seat step per fingertip hit (m). Raised from the old 0.005: a 5 mm nudge per
    // contact barely cleared the off-centre collision before the next graze, so the reflex
    // crept sideways over many hits (and often hit MAX_INSERT_REFLEXES first). A bigger step
    // moves decisively off the contacting finger in one maneuver. Clamped to TIP_REFLEX_MAX_M.
    double tip_reflex_step_ = 0.012;     // Controller.tip_reflex_step
    float  insert_tip_peak_     = 0.0f;  // diag: peak fingertip force seen during the current insert
    Eigen::Vector3d closing_hold_pos_{0.0, 0.0, 0.0};  // EE pose frozen at seat: close HERE, don't ram deeper
    // Persistent tactile CALIBRATION: the lateral reflex offset that finally seated the grasp is
    // accumulated here (world frame) and added to the committed grasp target on every future rep, so
    // the systematic perception bias (model drawn in front of / beside the points) is learned out.
    Eigen::Vector3d tactile_calib_{0.0, 0.0, 0.0};
    bool   bottle_obstacle_ = false;
    double bottle_obstacle_margin_ = 0.04;
    double elbow_gain_ = 2.0;
    double col_radius_ = 0.225;   // body-collision cylinder radius (m). 0.225 → 0.45 m dia for the wide tray (was 0.03 column). Config: Controller.col_radius.
    bool   elbow_target_set_ = false;
    Eigen::Vector2d elbow_target_xy_{0.0, 0.0};
    std::array<double, Kinematics::N_ARM_JOINTS> last_q_dot_cmd_{};

    // ── Predictive selection / reach map ─────────────────────────────────────
    bool   predictive_place_ = false;
    bool   predictive_grasp_ = false;
    bool   force_top_down_   = false;
    bool   precompute_reach_map_ = true;
    bool   reach_map_done_       = false;
    std::string reach_map_path_  = "experiments/reach_map.csv";
    std::vector<float> rm_mu_;
    double rm_x0_ = -0.40, rm_y0_ = -0.90, rm_res_ = 0.05;
    int    rm_nx_ = 0, rm_ny_ = 0;

    // ── Round / cycle bookkeeping ────────────────────────────────────────────
    bool returning_for_cycle_   = false;
    bool retrying_              = false;   // a homing return is a RETRY of the same rep (not a new rep)
    int  round_cycles_          = 0;
    int  pick_place_cycles_done_ = 0;

    // ── Grasp-phase tick counters + latched frame ────────────────────────────
    int grasp_settle_ticks_ = 0, closing_ticks_ = 0, insert_ticks_ = 0, lift_ticks_ = 0, level_ticks_ = 0;
    int place_ticks_ = 0, place_settle_ticks_ = 0;
    double place_bottle_z_prev_ = 1e9;
    int track_stuck_ticks_ = 0, track_noprog_ticks_ = 0, rep_track_ticks_ = 0, rep_attempts_ = 0;
    double track_best_dist_ = 1e9;
    SideGraspTarget latched_grasp_{};
    Eigen::Vector3d lift_target_{}, place_pos_{}, place_hover_{}, place_z_des_{}, place_x_des_{};
    Eigen::Vector3d retract_target_{};
    int    reflex_count_ = 0, retract_ticks_ = 0, grasp_force_ticks_ = 0;
    double bottle_z_at_lift_start_ = 0.0;
    double rep_commit_epos_ = 0.0, rep_commit_eang_ = 0.0;
    // Failure diagnostics (why grasps slip — esp. with a heavier object):
    Eigen::Vector3d bottle_at_grasp_{0,0,0};   // bottle pose when the grasp was committed
    float  close_lf_ = 0.0f, close_rf_ = 0.0f; // L/R pad forces at the close→lift transition
    double lift_wrist_fz0_ = 0.0;              // wrist vertical-wrench baseline captured at lift start
    std::mt19937 rng_{std::random_device{}()};

    // ── Probe / skill structs + state ────────────────────────────────────────
    struct GraspPerturbation { double dx_perp = 0, dz_axis = 0, dazi = 0, speed_scale = 1.0; };
    GraspPerturbation rep_perturb_{};
    bool   probe_enabled_ = false;
    double probe_pos_amp_ = 0.015, probe_azi_amp_ = 0.15, probe_speed_amp_ = 0.25;
    long   probe_index_ = 0;
    std::ofstream dataset_;   bool dataset_open_ = false;

    bool   precision_reweighting_ = false;
    double perception_noise_std_ = 0.0;   // σ_obs (m): injected bottle-pose observation noise (sim2real)
    double surprise_chi_         = 3.0;    // Mahalanobis k: surprise if ‖innov‖ > k·√(P+σ_obs²)
    double conf_gain_ = 0.15, conf_decay_ = 0.5;
    int    skilled_sample_period_ = 12;
    // Synthetic perception-latency harness: a real look-up (segmentation + pose) costs tens of ms;
    // in sim compute_side_grasp_target() is free, so the timing benefit of looking LESS is invisible.
    // When > 0, each actual look-up sleeps this long, genuinely slowing the loop on look-up cycles
    // (the arm holds q̇ for that interval — the real held-velocity exposure). As c_approach rises and
    // look-ups thin out, the effective approach control rate recovers toward 1/T_ctrl. The per-episode
    // effective rate is logged so the decoupling is MEASURED, not assumed. 0 = off (unchanged).
    double      perception_latency_ms_ = 0.0;
    // Legacy synthetic-latency harness. Default OFF: the real perception cost is now the producer's
    // refresh bandwidth (model_generation bumps, w_.bottle_refresh_period_s_), not a CPU sleep — so a
    // graph read is only fused when it carries new info. Set true to restore the old sleep(perception_
    // latency_ms) for reproducible cost sweeps that don't depend on the live perception rate.
    bool        use_synthetic_latency_ = false;
    std::string latency_log_path_;
    std::ofstream latency_log_;   bool latency_log_open_ = false;
    double      approach_t0_ = 0.0;   // wall-clock at the first Tracking cycle of the episode
    double speed_conf_gain_ = 0.6, surprise_gate_m_ = 0.05;
    // Lift is the risky leg: a slow, deliberate, straight-up rise that does NOT cam the grip open
    // or twist the held bottle. Deliberately NOT skill-boosted (skilled_speed) — the lift stays
    // careful regardless of c. Configurable via Controller.lift_speed.
    double lift_speed_ = 0.10;        // m/s vertical rise during Lifting
    // Soft-start ramp length (loop ticks) over which the lift velocity rises 0→full. LONGER = gentler
    // lift-off acceleration ⇒ the bottle is not jerked out of the grip at the instant of departure (the
    // sub-throat slide). Configurable via Controller.lift_soft_ticks. 100 Hz loop ⇒ ticks·10 ms.
    double lift_soft_ticks_ = 60.0;   // was a fixed 25 (≈0.25 s); 60 ≈ 0.6 s gentler ramp
    // DIAGNOSTIC observe mode: during Lifting, keep rising to full height and HOLD there with the gripper
    // clamped — ignore the confirm/slip/timeout aborts (no MISS/reopen/home) — so the physical outcome
    // (does the bottle rise, tilt, slide out?) is directly visible. Controller.lift_observe.
    bool   lift_observe_ = false;
    // Minimum confirmed rise (m): even once the generative TRACK channel has decided "grasped", keep
    // rising until the bottle has visibly come up by this much before declaring the lift done. Makes
    // the vertical motion clearly observable and matches the preference (lift TO a height, not just
    // "it started to move"). Must be < LIFT_HEIGHT_M (the target the arm climbs toward).
    double lift_min_confirm_rise_ = 0.12;   // Controller.lift_min_confirm_rise
    double orient_conf_gain_ = 1.5;   // skilled → faster free-space wrist slew during the approach
    double standoff_collapse_ = 0.0, insert_conf_gain_ = 0.0;
    // ── Tactile-safe insertion (novice slow → skilled fast) ──────────────────────────────
    // Until the insert segment's precision (skill_c) builds from confirmed grasps, creep the
    // insert so a fingertip graze is caught and rectified BEFORE the light bottle is tipped. The
    // insert speed is scaled by insert_novice_frac at c=0, ramping to full at c=1 — the same
    // precision that the perception look-up uses, so "slow & feeling" and "fast & confident" are
    // one state. 1.0 ⇒ old behaviour (no novice slowdown).
    double insert_novice_frac_ = 0.4;
    // ── Palm-proximity grasp gate ────────────────────────────────────────────────────────
    // Commit to Closing only when the palm sensor confirms the bottle is actually in the throat,
    // killing the "close on empty air" failures. Auto-bypassed if the sensor is absent (reads 0 →
    // palm_active_ never set). palm_grasp_dist = the throat depth (m) below which the bottle counts
    // as seated between the jaws.
    bool   use_palm_gate_   = true;
    double palm_grasp_dist_ = 0.08;
    bool   palm_active_     = false;   // set once a nonzero palm reading proves the sensor exists
    // Palm-depth SEAT trigger: stop the insert and close the moment the palm ray first sees the bottle
    // at grasp depth (palm_d < palm_seat_dist), so the bottle is captured BETWEEN the finger pads
    // instead of being rammed all the way to the palm (palm≈0.01) where the pads close on empty air
    // (gap≈0.004, every lift slips). Only used when the palm sensor is present; else fall back to e<tol.
    double palm_seat_dist_  = 0.04;
    std::string confidence_path_;
    // ── Per-segment precision learning from doing ────────────────────────────
    // c_k = Π_m[k]/(Π_m[k]+Π_s) is NOT a scripted ramp: each segment's Π_m ACCUMULATES
    // from ITS confirmed outcomes (Π_m[k] += unit·quality) and DEFLATES on its surprise
    // (×conf_decay). So each leg gets skilled purely by doing and self-calibrates, and
    // the partition decouples speed from reliability — safe legs (transport) cruise while
    // risky legs (set-down) stay careful. Π_s fixed ⇒ c_k saturates ~n/(n+Π_s). Persisted.
    double pi_m_[N_SEG] = {0, 0, 0, 0, 0};   // per-segment accumulated model precision
    double pi_s_ = 2.0;                       // Controller.sensory_precision (fixed)
    // Prior-precision FLOOR: a segment never becomes infinitely unskilled. Without it, repeated
    // insert misses multiply Π_m by conf_decay every time (0.5×) → Π_m→0 → c→0 → the novice-slow
    // insert creeps too slowly to seat → it times out → deflates again: a death-spiral (run 2026-06-21
    // collapsed reps 5-9, c_insert 0.11→3e-5). The floor pins c_seg ≥ c_floor_, keeping the insert at
    // the speed that grasped reliably in the early reps. pi_m_floor_ = pi_s_·c_floor/(1−c_floor).
    double c_floor_     = 0.3;                 // Controller.c_floor
    double pi_m_floor_  = 0.0;                 // derived from pi_s_ and c_floor_ at init
    double evidence_unit_ = 1.0;              // Controller.evidence_unit: Π_m added per clean leg
    SideGraspTarget belief_grasp_{};
    bool   belief_valid_ = false;
    double belief_var_   = 1e9;   // scalar Kalman pose-belief variance P (m²); set on first look
    int    cycles_since_obs_ = 0, obs_count_rep_ = 0;
    double rep_t0_ = 0.0;
    double rep_pick_s_ = 0.0;   // time from rep start to grasp-confirm (the c-scheduled portion)
    std::ofstream metrics_;   bool metrics_open_ = false;

    struct RetreatPerturbation { double dspeed = 0.0; double dopen = 0.0; };
    RetreatPerturbation retreat_perturb_{};
    double retreat_speed_ = 0.06, gripper_open_conf_ = 0.85;
    int    release_ticks_ = 8;
    bool   probe_retreat_ = false;
    double probe_rspeed_amp_ = 0.50, probe_open_amp_ = 0.10;
    Eigen::Vector3d retreat_target_pos_{0,0,0}, retreat_z_des_{0,0,1}, retreat_x_des_{1,0,0};
    Eigen::Vector3d place_world_xy_{0,0,0};
    std::ofstream retreat_log_;   bool retreat_log_open_ = false;

    // ── Experiment overrides ──────────────────────────────────────────────────
    // Controller.learn_pick_place: false ⇒ Tracking holds at the standoff (the validated
    // approach baseline). true ⇒ Tracking commits and runs the full pick→place→retreat
    // loop, so the Lifting confirm fires and precision (c) accumulates from doing.
    bool   learn_pick_place_ = false;
    bool   respawn_each_rep_ = false;
    bool   fixed_pick_set_ = false;   Eigen::Vector2d fixed_pick_xy_{0.1, -0.25};
    // #3 feed-forward perception-bias cancel (world-frame XY, m), added to the perceived bottle in
    // compute_side_grasp_target so the grasp target lands on the REAL bottle. Set = −biasPG (the
    // measured perceived−GT). Controller.perception_bias_ff_xy. Zero = off.
    Eigen::Vector2d perception_bias_ff_{0.0, 0.0};
    bool   fixed_place_set_ = false;  Eigen::Vector2d fixed_place_xy_{0.15, -0.25};

    // ── Grasp / place tuning constants ───────────────────────────────────────
    static constexpr int    GRASP_SETTLE_TICKS   = 8;
    static constexpr float  GRASP_FORCE_THRESH   = 3.0f;
    static constexpr int    GRASP_FORCE_HOLD_TICKS = 3;
    static constexpr float  INSERT_TOUCH_FORCE   = 0.3f;
    static constexpr double INSERT_VEL_MS        = 0.06;
    static constexpr int    CLOSING_TIMEOUT_TICKS = 100;
    // The Robotiq 2F-85 takes ~0.5 s to physically close. A 3-tick (30 ms) force blip during the
    // close MOTION used to fire the Lifting transition while the jaws were still moving — the arm
    // "moved up without closing", knocking the bottle. HOLD the seated pose for at least this many
    // ticks so the jaws fully close+settle BEFORE the grasp is judged or the lift starts.
    static constexpr int    MIN_CLOSE_DWELL_TICKS = 55;   // ~0.55 s at 100 Hz; < CLOSING_TIMEOUT_TICKS
    // After the dwell, require a sustained two-finger load: both pads must read above this (N), so an
    // empty close (jaws meet each other) or a one-sided graze does NOT count as a grasp.
    static constexpr float  GRASP_PAD_MIN_N      = 5.0f;
    // Lift weight-confirmation: a truly held bottle adds ~m·g to the wrist wrench (0.4 kg ≈ 3.9 N).
    // Require at least this vertical-wrench gain over the pre-lift baseline to confirm a real lift.
    static constexpr double LIFT_WEIGHT_MIN_N    = 2.0;
    // The insert creeps the full APPROACH_STANDOFF_M (0.12 m) at INSERT_VEL_MS. At 0.06 m/s that is
    // ~200 ticks (10 ms each); 150 timed out MID-descent (the spurious "reflex stuck" MISS that made
    // it retreat from good positions). Budget the real travel time + margin.
    static constexpr int    INSERT_TIMEOUT_TICKS  = 300;
    static constexpr double TIP_REFLEX_MAX_M      = 0.05;
    static constexpr double TIP_REFLEX_BACKOFF_M  = 0.04;   // MUST exceed REACH_TOLERANCE_M (0.02): a 0.02 backoff
                                                            // target is already within tolerance the instant it is
                                                            // set, so the retreat completes immediately and the arm
                                                            // shows no visible stop-and-back-off on a tip graze.
    static constexpr int    MAX_INSERT_REFLEXES   = 6;   // touch→retreat→shift→retry cycles before MISS
    // A fingertip graze this close to the seated grasp point is the canted-finger graze (the wrist
    // commits ~10° off the vertical floor, so the lower fingertip always touches just before seating).
    // The jaws straddle the bottle here, so CLOSE on it (the gripper self-centres) instead of backing
    // off — backing off never re-seats, times out, and retreats from an otherwise-good grasp.
    static constexpr double GRASP_TOUCH_COMMIT_M  = 0.045;
    // Post-grasp Leveling: lift a small clearance off the table AND rotate the gripper to the
    // (horizontal) grasp frame with orientation ENFORCED, so the ~15° canted commit is undone
    // while the bottle hangs free and low — before the main lift, which then stays upright.
    static constexpr double LEVEL_CLEAR_M        = 0.04;   // clearance lifted during leveling (m)
    static constexpr double LEVEL_TOL_RAD        = 0.10;   // ~5.7°: leveled enough → Lifting
    static constexpr int    LEVEL_TIMEOUT_TICKS  = 80;     // give up leveling, lift anyway
    static constexpr double LIFT_HEIGHT_M        = 0.18;
    static constexpr int    LIFT_TIMEOUT_TICKS   = 250;
    // Soft-start: the √ position-ramp eases IN near the target but saturates v_des instantly when the
    // target is far, so the close→lift target jump steps the commanded velocity from 0 → full = a jerk.
    // The lift-speed ramp length now lives in lift_soft_ticks_ (Controller.lift_soft_ticks, tunable).
    static constexpr double LIFT_CONFIRM_RISE_M  = 0.06;
    static constexpr double LIFT_CONFIRM_HOLD_M  = 0.18;  // relaxed: a held bottle may pivot/tilt as it lifts
    static constexpr int    TRACK_TIMEOUT_TICKS  = 900;
    static constexpr int    TRACK_NOPROGRESS_TICKS = 150;
    static constexpr int    MAX_REP_ATTEMPTS     = 3;
    static constexpr double FALL_TILT_RAD        = 0.60;
    static constexpr int    RETRACT_SETTLE_TICKS = 30;
    static constexpr int    MAX_REFLEXES         = 3;
    static constexpr double PLACE_X_MIN = -0.35, PLACE_X_MAX = 0.05;
    static constexpr double PLACE_Y_MIN = -0.85, PLACE_Y_MAX = 0.45;
    static constexpr double PLACE_MIN_MOVE_M    = 0.20;
    static constexpr double PLACE_REACH_MAX_M   = 0.85;
    static constexpr int    PLACE_TIMEOUT_TICKS = 300;
    static constexpr int    PLACE_SETTLE_TICKS    = 5;
    static constexpr double PLACE_UPRIGHT_TOL_RAD = 0.15;
    static constexpr double PLACE_ON_TABLE_M      = 0.04;
    static constexpr double PLACE_ORIENT_GAIN     = 0.4;
    static constexpr double PLACE_RETREAT_DIST_M  = 0.14;
    // Bottle respawn region (Webots/world frame). MUST be ON the table (Webots x∈[0.1,0.9],
    // y∈[-1.1,0.7]) with edge margins, AND within arm reach. Old paper values (x[-0.40,0.05]) were
    // off this table → the bottle fell off the front edge. Config: Controller.spawn_{x,y}_{min,max}.
    double spawn_x_min_ = 0.25, spawn_x_max_ = 0.55;
    double spawn_y_min_ = -0.65, spawn_y_max_ = -0.15;
    static constexpr double SPAWN_REACH_MIN_M = 0.35;
    static constexpr double SPAWN_REACH_MAX_M = 0.82;

    // Tip-over recovery: if the standing bottle's axis tilts past this, it has fallen — re-stand it
    // at the rep's spawn spot and restart the attempt. (GT mode only; uses the live bottle axis.)
    static constexpr double TIP_OVER_RAD = 0.52;   // ~30°
    static constexpr int    TIP_SETTLE_TICKS = 80; // after any respawn, wait this long before re-checking
                                                   // tilt — setObjectPose is buffered + the bottle must
                                                   // settle + the GT snapshot must update (else infinite loop)
    int             tip_cooldown_ = 0;
    Eigen::Vector2d rep_spawn_xy_{0.1, -0.25};     // where the current rep's bottle was placed (for re-standing)
};

#endif
