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
    float gripper_force() const;
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
    double c_seg(Segment s) const
    {
        if (force_confidence_ >= 0.0) return force_confidence_;
        const int k = global_confidence_ ? 0 : static_cast<int>(s);
        return pi_m_[k] / (pi_m_[k] + pi_s_);
    }
    double skill_c() const { return precision_reweighting_ ? c_seg(cur_seg_) : 0.0; }
    double skilled_speed(double base) const { return base * (1.0 + speed_conf_gain_ * skill_c()); }
    double c_overall() const;                       // mean c over segments (for logs)
    void   deposit(Segment s, double quality);      // confirmed outcome → Π_m[s] += unit·quality
    void   deflate(Segment s);                      // surprise → Π_m[s] *= conf_decay

    // ── Approach (Tracking) constants + state ─────────────────────────────────
    static constexpr double APPROACH_STANDOFF_M = 0.12;
    static constexpr double REACH_TOLERANCE_M   = 0.02;
    // Grasp ABOVE the bottle's centre of mass so gravity pendulums it upright in the gripper
    // instead of letting an off-centre grip tip it (raised 0.5→0.6 of bottle height).
    static constexpr double BOTTLE_GRASP_HEIGHT_FRAC = 0.6;
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
    long   ctrl_cycle_ = 0;
    double grasp_align_tol_rad_ = 0.14;

    // ── Per-cycle motion monitor (debug elbow-table dives + approach/place oscillation) ──
    // A single throttled [mon] line, emitted from efe_drive (the common chokepoint every
    // moving phase passes through), so EVERY phase — not just Tracking — gets telemetry.
    // Shows the elbow/lowest-link height above the table (elbow-dive watch), the bottle
    // tilt, and oscillation signals (Δe sign-flips, measured EE speed, |q̇|).
    bool   monitor_log_    = true;
    int    monitor_period_ = 10;     // print every N efe_drive calls
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
    bool   tip_reflex_ = false;
    double tip_reflex_offset_ = 0.0;
    bool   bottle_obstacle_ = false;
    double bottle_obstacle_margin_ = 0.04;
    double elbow_gain_ = 2.0;
    bool   elbow_target_set_ = false;
    Eigen::Vector2d elbow_target_xy_{0.0, 0.0};
    bool   tip_log_ = false;
    long   tip_log_cycle_ = 0;
    std::optional<Eigen::Vector3d> tip_log_prev_pos_;
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
    std::mt19937 rng_{std::random_device{}()};

    // ── Probe / skill structs + state ────────────────────────────────────────
    struct GraspPerturbation { double dx_perp = 0, dz_axis = 0, dazi = 0, speed_scale = 1.0; };
    GraspPerturbation rep_perturb_{};
    bool   probe_enabled_ = false;
    double probe_pos_amp_ = 0.015, probe_azi_amp_ = 0.15, probe_speed_amp_ = 0.25;
    long   probe_index_ = 0;
    std::ofstream dataset_;   bool dataset_open_ = false;

    bool   precision_reweighting_ = false;
    double perception_noise_std_ = 0.0;
    double conf_gain_ = 0.15, conf_decay_ = 0.5;
    int    skilled_sample_period_ = 12;
    double speed_conf_gain_ = 0.6, surprise_gate_m_ = 0.05;
    double standoff_collapse_ = 0.0, insert_conf_gain_ = 0.0;
    std::string confidence_path_;
    // ── Per-segment precision learning from doing ────────────────────────────
    // c_k = Π_m[k]/(Π_m[k]+Π_s) is NOT a scripted ramp: each segment's Π_m ACCUMULATES
    // from ITS confirmed outcomes (Π_m[k] += unit·quality) and DEFLATES on its surprise
    // (×conf_decay). So each leg gets skilled purely by doing and self-calibrates, and
    // the partition decouples speed from reliability — safe legs (transport) cruise while
    // risky legs (set-down) stay careful. Π_s fixed ⇒ c_k saturates ~n/(n+Π_s). Persisted.
    double pi_m_[N_SEG] = {0, 0, 0, 0, 0};   // per-segment accumulated model precision
    double pi_s_ = 2.0;                       // Controller.sensory_precision (fixed)
    double evidence_unit_ = 1.0;              // Controller.evidence_unit: Π_m added per clean leg
    SideGraspTarget belief_grasp_{};
    bool   belief_valid_ = false;
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
    bool   fixed_place_set_ = false;  Eigen::Vector2d fixed_place_xy_{0.15, -0.25};

    // ── Grasp / place tuning constants ───────────────────────────────────────
    static constexpr int    GRASP_SETTLE_TICKS   = 8;
    static constexpr float  GRASP_FORCE_THRESH   = 3.0f;
    static constexpr int    GRASP_FORCE_HOLD_TICKS = 3;
    static constexpr float  INSERT_TOUCH_FORCE   = 0.3f;
    static constexpr double INSERT_VEL_MS        = 0.05;
    static constexpr int    CLOSING_TIMEOUT_TICKS = 100;
    static constexpr int    INSERT_TIMEOUT_TICKS  = 150;
    static constexpr double TIP_REFLEX_STEP_M     = 0.005;
    static constexpr double TIP_REFLEX_MAX_M      = 0.05;
    static constexpr double TIP_REFLEX_BACKOFF_M  = 0.02;
    // Post-grasp Leveling: lift a small clearance off the table AND rotate the gripper to the
    // (horizontal) grasp frame with orientation ENFORCED, so the ~15° canted commit is undone
    // while the bottle hangs free and low — before the main lift, which then stays upright.
    static constexpr double LEVEL_CLEAR_M        = 0.04;   // clearance lifted during leveling (m)
    static constexpr double LEVEL_TOL_RAD        = 0.10;   // ~5.7°: leveled enough → Lifting
    static constexpr int    LEVEL_TIMEOUT_TICKS  = 80;     // give up leveling, lift anyway
    static constexpr double LIFT_HEIGHT_M        = 0.12;
    static constexpr int    LIFT_TIMEOUT_TICKS   = 250;
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
    static constexpr double SPAWN_X_MIN = -0.40, SPAWN_X_MAX = 0.05;
    static constexpr double SPAWN_Y_MIN = -0.82, SPAWN_Y_MAX = 0.45;
    static constexpr double SPAWN_REACH_MIN_M = 0.35;
    static constexpr double SPAWN_REACH_MAX_M = 0.82;
};

#endif
