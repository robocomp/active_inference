/*
 * epistemic_planner.h  —  AI2-native next-best-view proposal for the cabinet-concept agent.
 *
 * Scores each of the four vertical cabinet faces by the D-optimal expected entropy reduction on the belief's
 * FULL covariance Σ and proposes the robot pose (viewpoint + heading) that observes the winning face. The
 * viewpoint sits at v* = face_centre + outward_normal × standoff (projected to floor z), heading facing the
 * face centre; the standoff keeps the face inside a conservative horizontal FoV while staying close enough
 * to orbit the cabinet. Reference: CABINET.md §10.
 */

#pragma once

#include <array>
#include <cmath>

#include "cabinet_belief.h"                          // AI2 belief: Σ + predicted_information for the NBV
#include "../../common/nbv/viewpoint_score.h"        // rc::nbv — the shared DETECTION-WEIGHTED NBV core

// ─── Proposal ────────────────────────────────────────────────────────────────

namespace rc {

// An epistemic viewpoint proposal. Two representations of the SAME next-best-view:
//   • the OBJECT-RELATIVE constraint (authoritative): the ranked candidate faces + per-face gain, the
//     sensor-model stand-off band, framing fill, and the precision demand Σ*. The controller resolves
//     these into a collision-free reachable pose (it owns global occupancy/reachability).
//   • the absolute room-frame pose (epistemic_target_*): a NON-authoritative HINT = the argmax face's
//     framed viewpoint, kept for the legacy controller path until it consumes the object-relative form.
struct EpistemicProposal
{
    // Absolute-pose HINT (argmax face) — legacy path.
    float epistemic_target_x_m = 0.0f;
    float epistemic_target_y_m = 0.0f;
    float epistemic_target_yaw_rad = 0.0f;
    float epistemic_gain = 0.0f;          // gain of the winning face (the hint pose's gain)
    bool  valid = false;

    // Object-relative viewpoint constraint (authoritative) — see common/affordance_protocol.h.
    std::array<float, 4> face_gains{};    // [+x,-x,+y,-y] adequacy-bounded ΔH (nats), object frame
    float standoff_min_m = 0.0f;          // sensor-model stand-off band (FoV framing sweet spot)
    float standoff_max_m = 0.0f;
    float framing_fill   = 0.0f;          // desired projected fill fraction
    std::array<float, 7> sigma_star{};    // [cx,cy,yaw,L,d,z0,z1] precision demand (m / rad)

    bool is_finite() const
    {
        return std::isfinite(epistemic_target_x_m) &&
               std::isfinite(epistemic_target_y_m) &&
               std::isfinite(epistemic_target_yaw_rad) &&
               std::isfinite(epistemic_gain);
    }
};

// ─── Planner ─────────────────────────────────────────────────────────────────

// Stateless next-best-view scorer over the four vertical cabinet faces.
class EpistemicPlanner
{
public:
    /// @param d_obs  Comfortable maximum stand-off from the target face (m).
    explicit EpistemicPlanner(float d_obs = 1.8f);

    /// AI2-native next-best-view: the robot pose that best resolves the belief's dominant uncertainty.
    ///
    /// Scores each vertical face by the D-optimal expected entropy reduction on the belief's FULL covariance Σ:
    ///   gain(i) = ½·ln det( I₆ + Σ · ΔI(i) ),   ΔI(i) = Σₚ (1/Rᵢ) Jₚ Jₚᵀ  (belief.predicted_information)
    /// with Rᵢ = sigma_base² + (lat_rate·standoffᵢ)² — range-aware so far faces yield less information.
    /// Targets the dominant uncertainty eigen-direction of Σ (an unobserved extent / yaw). A low but
    /// finite gain is NOT withdrawn here; the controller's EFE selection simply won't pick a low-nat target.
    /// The returned per-face gains are DETECTION-WEIGHTED: p_detect(face) · ΔH_D-optimal(face). A face the
    /// detector cannot fire from is worth ~0 nats, so it cannot out-bid nav_dist in the controller's EFE and
    /// the robot stops paying travel for a look that could not have produced a mask.
    /// @param obstacles other objects, TRUE extents; a viewpoint inside one is rejected, a partly-occluded
    ///                  one is degraded continuously via visible_frac — never dropped.
    EpistemicProposal compute(const CabinetBelief& belief, float lat_rate, float sigma_base,
                              const rc::nbv::Sensor& sensor_in,
                              bool verbose = false,
    /// @param room_polygon the REACHABLE region (room_concept's `delimiting_polygon_x/y`). NOT optional in
    ///                   practice: the raw information term is direction-blind, so a face whose viewpoint lies
    ///                   OUTSIDE the room scores identically to the one inside and can win the tie. Nothing
    ///                   downstream refuses it either — the controller REPAIRS an unroutable standpoint with
    ///                   nearest_reachable, measured FROM the robot, which snaps it to the floor at the object.
    ///                   Left empty, rc::nbv::is_reachable imposes no constraint (it refuses to guess).
                              const std::vector<rc::nbv::Obstacle>& obstacles = {},
                              const std::vector<Eigen::Vector2f>& room_polygon = {}) const;

    // The detector's operating envelope. The stand-off is the argmax of THIS, so the viewpoint we ask for is
    // the one where the detector is most likely to fire — the same model the removal channel weights absence by.
    // This REPLACES the old kMinimumStandOffM = 1.15 literal ("YOLO won't fire too close"), which was a
    // hand-picked stand-in for the near shoulder of exactly this curve.
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
