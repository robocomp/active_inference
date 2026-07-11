/*
 * epistemic_planner.h  —  AI2-native next-best-view proposal for the table-concept agent.
 *
 * Scores each of the four vertical table faces by the D-optimal expected entropy reduction on the belief's
 * FULL covariance Σ and proposes the robot pose (viewpoint + heading) that observes the winning face. The
 * viewpoint sits at v* = face_centre + outward_normal × standoff (projected to floor z), heading facing the
 * face centre; the standoff keeps the face inside a conservative horizontal FoV while staying close enough
 * to orbit the table. Reference: TABLE_CONCEPT.md §8.
 */

#pragma once

#include <array>
#include <cmath>

#include "table_belief.h"            // AI2 belief: Σ + predicted_information for the Σ-based NBV

// ─── Proposal ────────────────────────────────────────────────────────────────

namespace rc {

// A single epistemic viewpoint proposal: target robot pose (room frame) + its expected information gain.
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
    EpistemicProposal compute(const TableBelief& belief, float lat_rate, float sigma_base) const;

private:
    float d_obs_;
};

}  // namespace rc
