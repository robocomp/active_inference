/*
 * epistemic_planner.h  —  AI2-native next-best-view proposal for the hood-concept agent.
 *
 * Scores each of the four vertical hood faces by the D-optimal expected entropy reduction on the belief's
 * FULL covariance Σ, weights each by the probability the detector actually FIRES from there, and proposes the
 * robot pose (viewpoint + heading) that observes the winning face. The viewpoint sits at
 * v* = face_centre + outward_normal × standoff (projected to floor z), heading facing the face centre.
 *
 * The stand-off is not a constant and not a horizontal framing formula: it is the argmax of the detector's
 * inverse model over BOTH image axes, found per face by `common/nbv/viewpoint_score.h`. This file owns the
 * BELIEF half (Σ, ΔI, the adequacy gap to Σ*); that module owns the SENSOR half. Reference: HOOD.md §10.
 */

#pragma once

#include <array>
#include <cmath>
#include <vector>

#include "hood_belief.h"            // AI2 belief: Σ + predicted_information for the Σ-based NBV
#include "../../common/detectability/detectability.h"   // rc::detect::DetectorEnvelope
#include "../../common/nbv/viewpoint_score.h"           // rc::nbv — the shared detection-weighted NBV core

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
    std::array<float, 4> face_gains{};    // [+x,-x,+y,-y] EXPECTED ΔH (nats) = P(detect)·raw — NOT adequacy-clamped (clamping made all four faces tie; see .cpp), object frame
    float standoff_min_m = 0.0f;          // sensor-model stand-off band (FoV framing sweet spot), FACE-relative
    float standoff_max_m = 0.0f;
    float framing_fill   = 0.0f;          // projected fill the servo should drive to = argmax of the detector envelope
    std::array<float, 6> sigma_star{};    // [cx,cy,H,w,h,yaw] precision demand (m / rad)

    // Diagnostic mirror of the winning candidate, so the agent can LOG what it proposed instead of only
    // printing it. Not part of the affordance contract; nothing downstream reads these.
    float chosen_standoff_m = 0.0f;   // FACE-relative distance the proposal realises
    float chosen_p_detect   = 0.0f;   // P(detect) there — ~0 means we are proposing a blind spot
    float sensor_vfov_rad   = 0.0f;   // the vertical FoV actually used (0 ⇒ no proposal was made)

    bool is_finite() const
    {
        return std::isfinite(epistemic_target_x_m) and std::isfinite(epistemic_target_y_m) and std::isfinite(epistemic_target_yaw_rad) and std::isfinite(epistemic_gain);
    }
};

// ─── Planner ─────────────────────────────────────────────────────────────────

// Stateless next-best-view scorer over the four vertical hood faces.
class EpistemicPlanner
{
public:
    EpistemicPlanner() = default;

    // An obstacle the viewpoint must not be placed inside / must not look through: an oriented footprint from
    // the DSR graph (any other `object`/`box` node), already inflated by the robot radius by the caller. Same
    // type the shared NBV core takes, so it passes straight through.
    using Obstacle = rc::nbv::Obstacle;

    /// AI2-native next-best-view: the robot pose that best resolves the belief's dominant uncertainty.
    ///
    /// Each vertical face contributes the D-optimal expected entropy reduction on the belief's FULL covariance Σ:
    ///   raw(i) = ½·ln det( I₆ + Σ · ΔI(i) ),   ΔI(i) = Σₚ (1/Rᵢ) Jₚ Jₚᵀ  (belief.predicted_information)
    /// with Rᵢ = sigma_base² + (lat_rate·standoffᵢ)² — range-aware so far faces yield less information.
    ///
    /// That raw number is the information a detection WOULD yield; what selection needs is the information we
    /// EXPECT TO GET, so `rc::nbv` weights it by P(detect | framing) and places each face at its own stand-off.
    /// A low but finite gain is NOT withdrawn here; the controller's EFE selection simply won't pick it.
    ///
    /// @param sensor     the camera's REAL FoVs (h = 2·atan(W/2fx), v = 2·atan(H/2fy)) and mount height above
    ///                   the floor datum. The VERTICAL channel is not optional for a hood: a 1.7 m box
    ///                   overflows the frame vertically at 1.2 m, where it still fills only a quarter of it
    ///                   horizontally, so a horizontal-only stand-off parks the robot where no mask can form.
    ///                   Left default (vfov 0, height 0) the vertical channel is off and that failure returns.
    /// @param obstacles  other objects in the room: standing inside one is rejected outright, looking through
    ///                   one degrades the face's score continuously via visible_frac.
    /// @param room_polygon the REACHABLE region (room_concept's `delimiting_polygon_x/y`). NOT optional in
    ///                   practice for this concept: a hood stands against a wall, so the viewpoints of
    ///                   its back and side faces lie OUTSIDE the room. The raw information term is
    ///                   direction-blind — front and back look identical to it — so nothing else breaks that
    ///                   tie, and `rc::nbv::is_reachable` deliberately imposes no constraint when the polygon
    ///                   is empty (it refuses to guess). Passing nothing therefore lets a through-the-wall pose
    ///                   win outright; the controller then repairs the unroutable goal with nearest_reachable,
    ///                   measured FROM the robot, which snaps it to the floor right at the fridge. That is how
    ///                   "the affordance target is too close to look at" is produced — see the live evidence in
    ///                   etc/archive/ai2_log_2026-08-08_live.csv (target 2.14 m out on the far side of the
    ///                   wall; realised robot range 0.69–0.90 m at predicted fill 1.35–4.0).
    EpistemicProposal compute(const HoodBelief& belief, float lat_rate, float sigma_base,
                              const rc::nbv::Sensor& sensor = {},
                              const std::vector<Obstacle>& obstacles = {},
                              const std::vector<Eigen::Vector2f>& room_polygon = {}) const;

    // The detector's operating envelope. The stand-off is the argmax of THIS, so the viewpoint we ask for is
    // the one where the detector is most likely to fire — the same model the removal channel weights absence by.
    void set_detector_envelope(const rc::detect::DetectorEnvelope& e) { det_env_ = e; }

private:
    rc::detect::DetectorEnvelope det_env_{};
};

}  // namespace rc
