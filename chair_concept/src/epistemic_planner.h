/*
 * epistemic_planner.h
 *
 * Computes epistemic action proposals for the chair-concept agent.
 *
 * For the highest-uncertainty vertical face of the chair, computes:
 *   v* = face_centre + outward_normal × d_view  (projected to floor z)
 *   heading = atan2(chair_centre.y - v.y,
 *                   chair_centre.x - v.x)        (facing the chair centre)
 * where d_view is chosen to keep the selected face inside a conservative
 * horizontal camera field-of-view while staying close enough to orbit the chair.
 *
 * Reference: ../table_concept/TABLE.md §10.
 */

#pragma once

#include <array>
#include <cmath>

#include "chair_belief.h"                             // AI2 belief: Σ + predicted_information for the NBV
#include "../../common/nbv/viewpoint_score.h"          // rc::nbv — the shared DETECTION-WEIGHTED NBV core

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
     * (ChairBelief::predicted_information), Rᵢ = σ_base² + (lat_rate·standoffᵢ)². Targets the dominant
     * uncertainty eigen-direction (an unobserved extent / seat depth / yaw).
     */
    /// The returned gain is DETECTION-WEIGHTED: p_detect(face) · ΔH_D-optimal(face). A face the detector
    /// cannot fire from is worth ~0 nats, so it cannot out-bid nav_dist in the controller's EFE and the robot
    /// stops paying travel for a look that could not have produced a mask.
    /// @param obstacles other objects, TRUE extents; a viewpoint inside one is rejected, a partly-occluded
    ///                  one is degraded continuously via visible_frac — never dropped.
    EpistemicProposal compute(const ChairBelief& belief, float lat_rate, float sigma_base,
                              const rc::nbv::Sensor& sensor_in,
    /// @param room_polygon the REACHABLE region (room_concept's `delimiting_polygon_x/y`). NOT optional in
    ///                   practice: the raw information term is direction-blind, so a face whose viewpoint lies
    ///                   OUTSIDE the room scores identically to the one inside and can win the tie. Nothing
    ///                   downstream refuses it either — the controller REPAIRS an unroutable standpoint with
    ///                   nearest_reachable, measured FROM the robot, which snaps it to the floor at the object.
    ///                   Left empty, rc::nbv::is_reachable imposes no constraint (it refuses to guess).
                              const std::vector<rc::nbv::Obstacle>& obstacles = {},
                              const std::vector<Eigen::Vector2f>& room_polygon = {}) const;

    // ★The old `MinStandOffM = 1.8` floor is GONE. It was a hand-picked stand-in for the near shoulder of the
    // detector envelope ("YOLO misses a chair that fills the frame from too close") — which is now MODELLED:
    // P(detect) is unimodal in projected fill, so the stand-off is its argmax and the near limit falls out of
    // the same curve the removal channel weights absence by. Config ChairConcept.MinStandOffM is now unread.
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
