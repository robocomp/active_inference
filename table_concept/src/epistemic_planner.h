/*
 * epistemic_planner.h
 *
 * Computes epistemic action proposals for the table-concept agent.
 *
 * For the lowest-coverage vertical face of the table, computes:
 *   v* = face_centre + outward_normal × d_view  (projected to floor z)
 *   heading = atan2(face_centre.y - v.y,
 *                   face_centre.x - v.x)        (facing the face centre)
 * where d_view is chosen to keep the selected face inside a conservative
 * horizontal camera field-of-view while staying close enough to orbit the
 * table instead of drifting far away from it.
 *   ΔH ∝ face_area / σ²                         (practical gain proxy)
 *
 * Reference: TABLE_CONCEPT.md §8.
 */

#pragma once

#include <array>
#include <cmath>

#include "sample_queue_geometry.h"   // common SampleQueue<Model> + table's geometry policy (face_coverage)
#include "table_model.h"
#include "table_belief.h"            // AI2 belief: Σ + predicted_information for the Σ-based NBV

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
    /**
     * @param delta_min      Minimum coverage count per face before declaring full.
     * @param gain_threshold Minimum ΔH to emit a proposal.
    * @param d_obs          Comfortable maximum stand-off from the target face (m).
     */
    explicit EpistemicPlanner(float d_obs = 1.8f);

    /**
     * Score the four vertical faces and return a viewpoint proposal for the most
     * informative one.
     *
     * When use_info_gain is set, each face's gain is the expected entropy reduction
     *   ΔH(f) = Σ_j ½·log(1 + I_pred_j(f) / Y_j)
     * where I_pred_j(f) is the per-DOF Fisher information that observing face f would
     * provide (predicted by evaluating the SDF-likelihood Fisher on synthetic face
     * samples) and Y_j is the current accumulated posterior precision (posterior_info,
     * = TableInstance::fisher_info_raw). ΔH→0 as a face becomes well-observed, so a low best gain
     * means "nothing left to learn" (belief→knowledge governor) — it is NOT withdrawn here; the
     * controller's EFE selection simply won't pick a low-nat target.
     */
    EpistemicProposal compute(const TableModel&  model,
                              const SampleQueue<TableModel>& queue,
                              const std::array<float, 8>& posterior_info = {}) const;

    /**
     * AI2-native next-best-view (no sample queue, no Fisher diagonal). Scores each vertical face by the
     * D-optimal expected entropy reduction on the belief's FULL covariance Σ:
     *   gain(i) = ½·ln det( I₆ + Σ · ΔI(i) ),   ΔI(i) = Σₚ (1/Rᵢ) Jₚ Jₚᵀ  (belief.predicted_information)
     * with Rᵢ = sigma_base² + (lat_rate·standoffᵢ)² — range-aware so far faces yield less information.
     * Targets the dominant uncertainty eigen-direction of Σ (an unobserved extent / yaw). Placement
     * (stand-off + heading) and the persist-low-gain policy are identical to the legacy compute().
     */
    EpistemicProposal compute(const TableBelief& belief, float lat_rate, float sigma_base) const;

private:
    float d_obs_;
};

}  // namespace rc
