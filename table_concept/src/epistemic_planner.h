/*
 * epistemic_planner.h  —  AI2-native next-best-view proposal for the table-concept agent.
 *
 * Scores each of the four vertical table faces by the D-optimal expected entropy reduction on the belief's
 * FULL covariance Σ and proposes the robot pose (viewpoint + heading) that observes the winning face. The
 * viewpoint sits at v* = face_centre + outward_normal × standoff (projected to floor z), heading facing the
 * face centre; the standoff keeps the face inside a conservative horizontal FoV while staying close enough
 * to orbit the table. Reference: TABLE.md §10.
 */

#pragma once

#include <array>
#include <cmath>
#include <vector>

#include "table_belief.h"                                  // AI2 belief: Σ + predicted_information for the NBV
#include "../../common/nbv/viewpoint_score.h"               // rc::nbv — the shared DETECTION-WEIGHTED NBV core

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
    std::array<float, 4> face_gains{};    // [+x,-x,+y,-y] RAW ΔH (nats) — NOT adequacy-clamped (clamping made all four faces tie; see .cpp), object frame
    float standoff_min_m = 0.0f;          // sensor-model stand-off band (FoV framing sweet spot)
    float standoff_max_m = 0.0f;
    float framing_fill   = 0.0f;          // desired projected fill fraction
    // ── what the NBV actually CHOSE, for the record ──────────────────────────────────────────────────
    // The affordance node holds the FROZEN pose during an executing claim, and stdout is gone the moment the
    // terminal scrolls — so without these, "why is the target so close?" is unanswerable after the fact.
    // nbv_vfov is here on purpose: 0 proves the camera model was still incomplete when this was proposed.
    float chosen_standoff_m = 0.0f;       // face-relative stand-off the plan realised
    float chosen_p_detect   = 0.0f;       // P(detect) at the chosen viewpoint
    float chosen_fill       = 0.0f;       // predicted roi_fill there
    float sensor_vfov_rad   = 0.0f;       // the camera model in force AT PROPOSAL TIME
    std::array<float, 6> sigma_star{};    // [cx,cy,H,w,h,yaw] precision demand (m / rad)

    bool is_finite() const
    {
        return std::isfinite(epistemic_target_x_m) and std::isfinite(epistemic_target_y_m) and std::isfinite(epistemic_target_yaw_rad) and std::isfinite(epistemic_gain);
    }
};

// ─── Planner ─────────────────────────────────────────────────────────────────

// Stateless next-best-view scorer over the four vertical table faces.
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
    // An obstacle the viewpoint must neither stand inside nor look through: an oriented footprint from the
    // DSR graph, in TRUE extents. rc::nbv inflates by the robot radius only for the stand-inside test; the
    // line-of-sight test needs the real ones (see the note on rc::nbv::stands_inside).
    using Obstacle = rc::nbv::Obstacle;

    /// @param obstacles  other objects; a viewpoint inside one is rejected, one that only partly SEES the
    ///                   face is degraded continuously via visible_frac — never dropped.
    ///
    /// The returned per-face gains are DETECTION-WEIGHTED: p_detect(face) · ΔH_D-optimal(face). A face the
    /// detector cannot fire from is worth ~0 nats, so it cannot out-bid nav_dist in the controller's EFE and
    /// the robot stops paying travel for a look that could not have produced a mask.
    EpistemicProposal compute(const TableBelief& belief, float lat_rate, float sigma_base,
                              const rc::nbv::Sensor& sensor_in,
                              const std::vector<Obstacle>& obstacles = {},
                              const std::vector<Eigen::Vector2f>& room_polygon = {}) const;

    // The detector's operating envelope. The stand-off is the argmax of THIS, so the viewpoint we ask for is
    // the one where the detector is most likely to fire — the same model the removal channel weights absence by.
    void set_detector_envelope(const rc::detect::DetectorEnvelope& e) { det_env_ = e; }


    // Footprint radius of the robot: the geometric floor under every stand-off, and the clearance the
    // stand-inside test allows. A physical dimension, not a knob.
    void set_robot_radius(float m) { robot_radius_m_ = m; }

private:
    float d_obs_;
    // The envelope is OWNED (config-driven); the camera GEOMETRY arrives per call, because the zed
    // intrinsics appear only once robot_concept starts publishing frames. See sensor_from_graph().
    rc::detect::DetectorEnvelope det_env_{};
    float robot_radius_m_ = 0.30f;   // Shadow
};

}  // namespace rc
