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
    explicit EpistemicPlanner(float delta_min     = 20.0f,
                              float gain_threshold = 0.1f,
                              float d_obs          = 1.8f);

    /**
     * Evaluate coverage of the four vertical faces and return a viewpoint
     * proposal targeting the lowest-coverage one.
     *
     * Returns a proposal with valid==false if all faces are sufficiently
     * covered or the computed gain is below gain_threshold.
     */
    EpistemicProposal compute(const TableModel&  model,
                              const SampleQueue& queue) const;

private:
    float delta_min_;
    float gain_threshold_;
    float d_obs_;
};

}  // namespace rc
