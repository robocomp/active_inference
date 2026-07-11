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

#include "chair_belief.h"            // AI2 belief: Σ + predicted_information for the Σ-based NBV

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
    EpistemicProposal compute(const ChairBelief& belief, float lat_rate, float sigma_base) const;

    // Minimum stand-off (m) from the target: for a small object like a chair the FoV-fit distance is tiny,
    // so the viewpoint always sits at this floor. Set it further out than the FoV needs — YOLO misses a
    // chair that fills/overflows the frame from too close. Config ChairConcept.MinStandOffM.
    void set_min_standoff(float m) { min_standoff_ = m; }

private:
    float d_obs_;
    float min_standoff_ = 1.8f;   // chair viewing distance floor (m)
};

}  // namespace rc
