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

#include "sample_queue.h"
#include "table_model.h"

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
    explicit EpistemicPlanner(float delta_min      = 20.0f,
                              float gain_threshold  = 0.1f,
                              float d_obs           = 1.8f,
                              bool  use_info_gain   = true,
                              float min_info_gain   = 0.3f);

    /**
     * Score the four vertical faces and return a viewpoint proposal for the most
     * informative one.
     *
     * When use_info_gain is set, each face's gain is the expected entropy reduction
     *   ΔH(f) = Σ_j ½·log(1 + I_pred_j(f) / Y_j)
     * where I_pred_j(f) is the per-DOF Fisher information that observing face f would
     * provide (predicted by evaluating the SDF-likelihood Fisher on synthetic face
     * samples) and Y_j is the current accumulated posterior precision (posterior_info,
     * = TableInstance::fisher_info_raw). ΔH→0 as a face becomes well-observed, so a
     * sub-threshold best gain means "nothing left to learn" (belief→knowledge governor).
     * Otherwise it falls back to the legacy face_area·deficit/σ² coverage proxy.
     *
     * Returns valid==false if the best gain is below the threshold.
     */
    EpistemicProposal compute(const TableModel&  model,
                              const SampleQueue& queue,
                              const std::array<float, 8>& posterior_info = {}) const;

private:
    float delta_min_;
    float gain_threshold_;
    float d_obs_;
    bool  use_info_gain_;
    float min_info_gain_;
};

}  // namespace rc
