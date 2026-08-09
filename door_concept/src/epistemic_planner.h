/*
 * epistemic_planner.h
 *
 * Computes epistemic action proposals for the door-concept agent.
 *
 * For the highest-uncertainty vertical face of the door, computes:
 *   v* = face_centre + outward_normal × d_view  (projected to floor z)
 *   heading = atan2(door_centre.y - v.y,
 *                   door_centre.x - v.x)        (facing the door centre)
 * where d_view is chosen to keep the selected face inside a conservative
 * horizontal camera field-of-view while staying close enough to orbit the door.
 *
 * Reference: ../table_concept/TABLE.md §10.
 */

#pragma once

#include <array>
#include <cmath>

#include "door_belief.h"                             // AI2 belief: Σ + predicted_information for the NBV
#include "../../common/nbv/viewpoint_score.h"        // rc::nbv — the shared DETECTION-WEIGHTED NBV core

// ─── Proposal ────────────────────────────────────────────────────────────────


namespace rc {
struct EpistemicProposal
{
    float epistemic_target_x_m = 0.0f;
    float epistemic_target_y_m = 0.0f;
    float epistemic_target_yaw_rad = 0.0f;
    float epistemic_gain = 0.0f;
    bool  valid = false;

    bool is_finite() const
    {
        return std::isfinite(epistemic_target_x_m) &&
               std::isfinite(epistemic_target_y_m) &&
               std::isfinite(epistemic_target_yaw_rad) &&
               std::isfinite(epistemic_gain);
    }
};

// ─── Planner ─────────────────────────────────────────────────────────────────

class EpistemicPlanner
{
public:
    /** @param d_obs  Comfortable maximum stand-off from the target face (m). */
    explicit EpistemicPlanner(float d_obs = 1.8f);

    /**
     * AI2-native next-best-view: scores each vertical seat-face by the D-optimal expected entropy
     * reduction on the belief's full Σ — gain(i) = ½·ln det(I₈ + Σ·ΔI(i)), ΔI(i) = Σₚ (1/Rᵢ) Jₚ Jₚᵀ
     * (DoorBelief::predicted_information), Rᵢ = σ_base² + (lat_rate·standoffᵢ)². Targets the dominant
     * uncertainty eigen-direction (an unobserved extent / seat depth / yaw).
     */
    /// The returned gain is DETECTION-WEIGHTED: p_detect(face) · ΔH_D-optimal(face). A face the detector
    /// cannot fire from is worth ~0 nats, so it cannot out-bid nav_dist in the controller's EFE and the robot
    /// stops paying travel for a look that could not have produced a mask.
    /// @param obstacles other objects, TRUE extents; a viewpoint inside one is rejected, a partly-occluded
    ///                  one is degraded continuously via visible_frac — never dropped.
    EpistemicProposal compute(const DoorBelief& belief, float lat_rate, float sigma_base,
                              const rc::nbv::Sensor& sensor_in,
                              const std::vector<rc::nbv::Obstacle>& obstacles = {},
    /// @param room_polygon the REACHABLE region. Without it the winner can be the face on the far side of the
    ///                  wall: it is fully visible through the aperture and completely unreachable (live
    ///                  2026-08-07 — +y and -y tied exactly and the target landed at y=7.27 in a 4.63 m room).
                              const std::vector<Eigen::Vector2f>& room_polygon = {},
    /// @param plan_out optional: the full four-face plan behind the decision (per-face stand-off, visibility,
    ///                  reachability, P(detect), expected gain). For monitoring — see etc/door_nbv_log.csv.
                              rc::nbv::Plan* plan_out = nullptr) const;

    // ★The old `MinStandOffM = 2.0` floor is GONE. It was a hand-picked stand-in for the near shoulder of the
    // detector envelope ("YOLO misses doors seen too close") — which is now MODELLED: P(detect) is unimodal in
    // projected fill, so the stand-off is its argmax and the near limit falls out of the same curve the
    // removal channel weights absence by. Config DoorConcept.MinStandOffM is now unread.
    void set_detector_envelope(const rc::detect::DetectorEnvelope& e) { det_env_ = e; }


    // Footprint radius of the robot: the geometric floor under every stand-off. A physical dimension.
    void set_robot_radius(float m) { robot_radius_m_ = m; }

private:
    float d_obs_;
    // The envelope is OWNED (config-driven); the camera GEOMETRY arrives per call, because the zed
    // intrinsics appear only once robot_concept starts publishing frames. See sensor_from_graph().
    rc::detect::DetectorEnvelope det_env_{};
    float robot_radius_m_ = 0.30f;   // Shadow
};

}  // namespace rc
