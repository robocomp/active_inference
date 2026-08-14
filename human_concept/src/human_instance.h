/*
 * human_instance.h
 *
 * Per-person runtime state shared by the human_concept collaborators (HumanFitter
 * fits it, HumanSceneGraph publishes it). One HumanInstance per "person_N" DSR node.
 * Mirrors bottle_instance.h, but the belief is the kinematic-model Laplace estimator
 * (cpp/core) rather than an SDF fit, so there is no support bank / sample queue.
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>

#include "../../common/object_affordance/object_affordance.h"   // rc::ObjectAffordance (SHARED)
#include "vfe_inference.h"        // rc::human::AInfLaplacePoseEstimator / InferenceResult
#include "../../common/belief_stabilizer/belief_stabilizer.h"   // rc::StabilizerState

namespace rc {

struct HumanInstance
{
    std::uint64_t node_id = 0;
    std::string   node_name;
    int           track_id = 0;     // SkeletonSource body id this node tracks

    // Per-person kinematic model (own bone lengths, calibrated online from observed limb distances —
    // see human::calibrate_lengths). The estimator references THIS model, so the fit uses this person's
    // proportions. Declared BEFORE the estimator so it outlives it; unordered_map keeps its address stable.
    human::HumanKinematicModel model;
    bool calib_init = false;

    // The active-inference belief: one stateful estimator per tracked person (mu carries across frames).
    std::unique_ptr<human::AInfLaplacePoseEstimator> estimator;
    human::InferenceResult last_result;   // most recent infer() output (pose, mu, uncertainty) = TARGET
    bool has_result = false;

    // Output-side motion model (see human_controller.h): velocity/accel-limited command that tracks
    // the estimator's target angles. cmd_kp is the published (smoothed) room-frame pose.
    human::Vec11 theta_cmd = human::Vec11::Zero();
    human::Vec11 theta_vel = human::Vec11::Zero();
    bool cmd_init = false;
    human::KpArray cmd_kp = human::KpArray::Zero();
    bool has_cmd = false;
    float track_err = 0.f;   // mean |target - cmd| over the angle DOFs (tracking lag, for the CSV)

    // Smoothed GLOBAL pose (the per-frame Kabsch R,t to noisy keypoints jitters the facing). The
    // published pose is R_sm · forward(theta_cmd) + t_sm. R EMA'd via quaternion slerp, t via lerp.
    Eigen::Matrix3f R_sm = Eigen::Matrix3f::Identity();
    Eigen::Vector3f t_sm = Eigen::Vector3f::Zero();
    bool pose_init = false;

    // Epistemic "reduce-occlusion" affordance for the controller (next-best-view to see hidden joints).
    ObjectAffordance affordance;
    bool  epistemic_pending  = false;
    int   epistemic_cooldown = 0;
    float last_epistemic_gain = 0.0f;

    int matched_frames   = 0;     // frames with fresh keypoints
    // Wall-clock of the previous fresh fit — the real inter-fit dt for the speed/accel limits (the
    // fit rate is gated by the data stream, not the compute loop). nullopt until the first fit.
    std::optional<std::chrono::steady_clock::time_point> last_fit_time;
    int processed_cycles = 0;     // per-person compute cycles (log throttling)
    int model_generation = 0;
    float prev_free_energy = std::numeric_limits<float>::max();

    // ── Detection aliveness (active-perception feedback for the affordance contract) ────────────────
    int   frames_since_detection   = 100000;   // cycles since the last fresh skeleton (0 = just seen)
    float last_mask_confidence     = 0.0f;     // mean joint confidence of the last fresh skeleton
    bool  detection_alive          = false;
    bool  last_pub_detection_alive = false;
    float last_pub_detection_conf  = -1.0f;

    // Dead-band traces so a settled belief stops rewriting the node / RT edge.
    float last_written_x = std::numeric_limits<float>::max();
    float last_written_y = std::numeric_limits<float>::max();

    // Shared per-DOF belief stabiliser over the 11 angle DOFs (diagnostic / dashboard).
    StabilizerState<11> stab;
    std::array<float, 11> prev_diag_state{};
    bool has_prev_diag = false;

    // RT parent the person hangs from (always the room for a free-standing human).
    std::uint64_t parent_id = 0;
    std::string   parent_name = "room";
};

}  // namespace rc
