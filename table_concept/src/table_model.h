/*
 * table_model.h
 *
 * Generative model for a table instance.
 *
 * State vector θ = [cx, cy, w, h, table_height, leg_length, yaw]  (7 params)
 * Fixed geometry: TOP_THICKNESS = 0.03 m, LEG_RADIUS = 0.025 m
 *
 * Compound SDF = min(SDF_top, min_k SDF_leg_k)
 * Free Energy  = Σᵢ wᵢ · ρ(SDF(θ,yᵢ)) / σ²  +  λ‖θ−θ_prior‖²_Σ
 * Gradient computed with PyTorch autograd and optimised with Adam/SGD.
 */

#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <Eigen/Dense>

#include "../../common/robust_metrics/robust_metrics.h"

// ─── Parameter structs ────────────────────────────────────────────────────────

struct TableState
{
    float cx           = 0.0f;   // Room-frame X of table centre
    float cy           = 0.0f;   // Room-frame Y of table centre
    float w            = 1.0f;   // Width  (table-local X)
    float h            = 0.6f;   // Depth  (table-local Y)
    float table_height = 0.75f;  // Height of table surface from floor
    float leg_length   = 0.72f;  // Length of legs from floor to underside of top
    float yaw          = 0.0f;   // Rotation around Z axis (room frame)

    // Serialise/deserialise as 7-vector
    std::array<float, 7> to_array() const
    {
        return {cx, cy, w, h, table_height, leg_length, yaw};
    }

    static TableState from_array(const std::array<float, 7>& a)
    {
        return {a[0], a[1], a[2], a[3], a[4], a[5], a[6]};
    }
};

struct TableModelParams
{
    // Observation noise (SDF likelihood width)
    float sigma_obs = 0.05f;

    // KL regulariser weights (toward prior)
    float lambda_size  = 0.5f;   // Prior table dimensions
    float lambda_pos   = 0.05f;  // Position stays close to prior
    float lambda_state = 0.02f;  // Size state transition
    float lambda_angle = 0.01f;  // Angle transition

    // Prior geometry (used by KL term)
    float prior_w            = 1.0f;
    float prior_h            = 0.6f;
    float prior_table_height = 0.75f;
    float prior_size_std     = 0.15f;

    // Gradient-descent optimiser
    int   optimization_iters = 10;
    float optimization_lr    = 0.05f;
    float grad_clip          = 2.0f;

    // Optimizer selection for autograd loop: "adam" or "sgd"
    std::string optimizer_type = "adam";
    // Momentum used when optimizer_type == "sgd"
    float sgd_momentum = 0.9f;
    // Robust loss applied to SDF residuals before scaling by 1 / sigma_obs^2
    RobustLossType robust_loss = RobustLossType::Quadratic;
    float robust_loss_scale = 0.10f;
};

// ─── TableModel ──────────────────────────────────────────────────────────────

class TableModel
{
public:
    static constexpr float TOP_THICKNESS = 0.03f;
    static constexpr float LEG_RADIUS    = 0.025f;
    static constexpr float SDF_SMOOTH_K  = 0.02f;
    static constexpr float SDF_INSIDE_SCALE = 0.3f;

    TableModel() = default;
    TableModel(const TableState& prior, const TableModelParams& params);

    // ── SDF ──────────────────────────────────────────────────────────────────

    /** SDF for a single 3-D point (room frame). */
    float sdf_point(const Eigen::Vector3f& p) const;

    /** SDF for a batch of points. Returns one value per point. */
    std::vector<float> compute_sdf(const std::vector<Eigen::Vector3f>& points) const;

    // ── Free Energy ──────────────────────────────────────────────────────────

    /**
    * F(θ) = Σᵢ wᵢ·ρ(SDF(yᵢ))/σ²  +  KL(q‖p)
     *
     * @param points   Room-frame 3-D observations.
     * @param weights  Per-point weights (1.0 = equal); pass empty for uniform.
     */
    float compute_free_energy(const std::vector<Eigen::Vector3f>& points,
                              const std::vector<float>& weights) const;

    // ── Gradient step ────────────────────────────────────────────────────────

    /**
     * Perform `params_.optimization_iters` gradient-descent steps on θ.
     * Returns the free energy after the last step.
     */
    float gradient_step(const std::vector<Eigen::Vector3f>& points,
                        const std::vector<float>& weights);

    // ── State access ─────────────────────────────────────────────────────────

    const TableState& state()  const { return state_; }
    const TableState& prior()  const { return prior_; }
    const TableModelParams& params() const { return params_; }

    void set_state(const TableState& s) { state_ = s; apply_constraints(); }
    void set_prior(const TableState& p) { prior_ = p; }

    /**
     * Clamp leg_length so that leg_length ≤ table_height − TOP_THICKNESS,
     * and enforce positive dimensions.
     */
    void apply_constraints();

    /**
     * Axis-aligned bounding box of the full table (top + legs) in room frame.
     * Returns {min_corner, max_corner}.
     */
    std::pair<Eigen::Vector3f, Eigen::Vector3f> bounding_box() const;

private:
    // Evaluate free energy for an arbitrary state (scalar float path)
    float fe_at(const TableState& s,
                const std::vector<Eigen::Vector3f>& pts,
                const std::vector<float>& weights) const;

    // Prior (size regularisation) energy for a given state
    float prior_energy(const TableState& s) const;

    // SDF evaluated for a given explicit state
    float sdf_point_at(const Eigen::Vector3f& p, const TableState& s) const;

    TableState        state_;
    TableState        prior_;
    TableModelParams  params_;
};
