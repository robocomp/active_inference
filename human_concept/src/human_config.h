/*
 * human_config.h
 *
 * Plain-data configuration for the human_concept agent + a loader that fills it
 * from a RoboComp ConfigLoader. Mirrors bottle_config.h, but the tunables are the
 * active-inference pose-estimator knobs (the SDF/mask/voxel knobs do not apply —
 * human_concept fits a kinematic model to BODY_18 keypoints, see cpp/core).
 */

#pragma once

#include <string>
#include <vector>

class ConfigLoader;   // RoboComp config façade (defined in genericworker.h)

namespace rc {

struct HumanConfig
{
    std::string priors_path = "etc/object_priors.toml";

    // ── Skeleton input source (decoupled; replay-first, see skeleton_source.h) ──────────────────────
    std::string source_kind = "replay";   // "replay" (CSV) | "live" (future: ZED / media plane)
    std::string replay_path = "";          // CSV of keypoints (frame[,id],54 kp[,18 conf]); empty = none
    bool        replay_loop = true;        // wrap to the first frame at EOF

    // ── AInfLaplacePoseEstimator knobs (forwarded to rc::human::InferenceConfig) ────────────────────
    std::vector<int> anchors = {1, 2, 5, 8, 11};   // neck, shoulders, hips
    float sigma_obs  = 0.06f;
    float sigma_dyn  = 0.25f;
    float sigma_min  = 0.02f;
    float sigma_max  = 0.15f;
    float w_limits   = 5.0f;
    float w_sym      = 1.0f;
    int   gn_steps   = 2;
    float damping    = 1e-3f;

    // Anti arm-cross (sidedness) prior — filters YOLO L/R swaps. Soft; allows folded arms.
    float w_cross          = 3.0f;
    float arm_cross_margin = 0.05f;   // m past the opposite shoulder before it penalises
    // Neutral-pose prior — weak pull of arm DOFs toward rest; pins under-observed DOFs (stops drift).
    float w_neutral        = 1.0f;
    // Innovation gate (rad): reject + hold on a per-frame angle jump bigger than this (glitch filter).
    float max_innovation   = 1.0f;

    // Velocity/accel limits — applied by the OUTPUT CONTROLLER (human_controller.h), NOT inside the
    // fit. The in-fit penalty weights stay 0 (the controller replaced them). omega/alpha = angle DOFs;
    // vlin/alin = lower-body translation. dt = measured inter-fit interval (fallback from Period.Compute).
    float dt         = 0.05f;   // s between fits (fallback; runtime uses the measured interval)
    float w_vel      = 0.0f;    // in-fit velocity penalty (OFF — controller owns this)
    float w_acc      = 0.0f;    // in-fit acceleration penalty (OFF)
    float omega_max  = 3.0f;    // rad/s  (angle DOFs) — natural limb speed
    float alpha_max  = 12.0f;   // rad/s² (angle DOFs) — gentle ease-in/out
    float vlin_max   = 3.0f;    // m/s    (lb_x, lb_z)
    float alin_max   = 30.0f;   // m/s²   (lb_x, lb_z)
    std::string fit_csv_path = "";   // non-empty → per-cycle fit-diagnostics CSV (gate)

    // ── Simple active-inference action policy (the prototype's "action hint") ────────────────────────
    int   death_frames       = 60;     // cycles a person may go unseen before its node is removed (~3 s @20 Hz)
    int   min_valid          = 12;     // fewer valid joints ⇒ raise the look affordance
    float uncertainty_thresh = 0.05f;  // tr(cov) above this ⇒ raise the look affordance

    // ── Epistemic "reduce-occlusion" affordance ─────────────────────────────────────────────────────
    float epistemic_obs_distance    = 1.5f;   // stand-off (m) from the person at the look viewpoint
    float epistemic_view_info       = 50.0f;  // Fisher precision a clearer view is expected to add (ΔH scale)
    int   epistemic_cooldown_cycles = 120;    // post-completion hold (cycles) during which the gain is suppressed
    std::string epistemic_csv_path  = "";     // non-empty → per-cycle epistemic/affordance CSV

    // ── Observation-stillness (affordance contract .still): hold base speed below these for a clean look ─
    float still_vel   = 0.10f;   // m/s
    float still_omega = 0.15f;   // rad/s

    // ── Fisher CSV (per-cycle belief evolution; gated) ──────────────────────────────────────────────
    std::string fisher_csv_path = "";

    // Pose covariance written on the room→person RT edge: a diagonal scaled by tr(cov) (m²). The
    // estimator's Lambda is over joint angles, not the global pose, so this is a coarse proxy until a
    // full free-flyer reformulation folds the pose into the belief (see plan / Pinocchio note).
    float pose_cov_scale = 0.02f;
};

// Fill a HumanConfig from a RoboComp ConfigLoader (all keys + defaults).
HumanConfig load_human_config(const ConfigLoader& cfg);

}  // namespace rc
