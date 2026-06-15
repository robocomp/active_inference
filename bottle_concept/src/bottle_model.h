/*
 * bottle_model.h
 *
 * Generative model for a bottle instance — a single vertical cylinder.
 *
 * State vector θ = [cx, cy, cz, radius, height]  (5 params)
 *   No yaw: a cylinder is rotationally symmetric about its vertical axis, so the
 *   azimuth is unobservable and is dropped (one fewer free parameter, no gauge
 *   ambiguity for the optimiser to chase).
 *   (cx, cy, cz) is the cylinder centre in the room frame; the bottle stands on
 *   the table, so cz ≈ table_top + height/2 — the host anchors cz's prior there.
 *
 * Cylinder SDF (axis = +Z): d_radial = ‖(x,y)−(cx,cy)‖ − radius,
 *                           d_vertical = |z−cz| − height/2.
 * Free Energy  = Σᵢ wᵢ · ρ(SDF(θ,yᵢ)) / σ²  +  KL(q‖p)
 * Gradient computed with PyTorch autograd and optimised with Adam/SGD,
 * mirroring TableModel so the perception loop is identical bar the geometry.
 *
 * pose_covariance() returns the 3×3 position covariance from the Laplace
 * (curvature) approximation of F at the optimum: many well-spread observations
 * ⇒ tight covariance; few/no points ⇒ only the prior curvature survives ⇒ the
 * covariance grows. This is the P_bottle the controller consumes off the RT edge
 * to decide how slowly/often to look.
 */

#pragma once

#include <array>
#include <cmath>
#include <functional>
#include <string>
#include <utility>
#include <vector>
#include <Eigen/Dense>

#include "../../common/robust_metrics/robust_metrics.h"

// ─── Parameter structs ────────────────────────────────────────────────────────

struct BottleState
{
    float cx     = 0.0f;    // Room-frame X of cylinder centre
    float cy     = 0.0f;    // Room-frame Y of cylinder centre
    float cz     = 0.85f;   // Room-frame Z of cylinder centre (table_top + height/2)
    float radius = 0.035f;  // Cylinder radius
    float height = 0.20f;   // Cylinder height (full, not half)

    // Serialise/deserialise as 5-vector
    std::array<float, 5> to_array() const
    {
        return {cx, cy, cz, radius, height};
    }

    static BottleState from_array(const std::array<float, 5>& a)
    {
        return {a[0], a[1], a[2], a[3], a[4]};
    }
};

struct BottleModelParams
{
    // Observation noise (SDF likelihood width). Tighter than the table: a bottle
    // is small, so a few-cm SDF error is already a large fraction of its size.
    float sigma_obs = 0.02f;

    // KL regulariser weights (toward prior)
    float lambda_size  = 0.5f;    // Prior bottle dimensions (radius, height)
    float lambda_pos   = 0.05f;   // Centre stays close to prior centre
    float lambda_state = 0.02f;   // Size state transition

    // Prior geometry (used by KL term)
    float prior_radius   = 0.035f;
    float prior_height   = 0.20f;
    float prior_size_std = 0.03f;

    // Gradient-descent optimiser
    int   optimization_iters = 15;
    float optimization_lr    = 0.005f;   // cm-scale object: 10× smaller than the table's
    float grad_clip          = 2.0f;

    // Optimizer selection for autograd loop: "adam" or "sgd"
    std::string optimizer_type = "adam";
    // Momentum used when optimizer_type == "sgd"
    float sgd_momentum = 0.9f;
    // Robust loss applied to SDF residuals before scaling by 1 / sigma_obs^2
    RobustLossType robust_loss = RobustLossType::Quadratic;
    float robust_loss_scale = 0.05f;
};

struct FreeEnergyDecomposition
{
    float likelihood_current    = 0.0f;
    float likelihood_historical  = 0.0f;
    float prior                 = 0.0f;
    float total_fe              = 0.0f;
    float effective_weight_mass  = 0.0f;
    float current_weight_mass   = 0.0f;
    float historical_weight_mass = 0.0f;
};

// ─── BottleModel ───────────────────────────────────────────────────────────────

class BottleModel
{
    public:
        using IterationObserver = std::function<void(int, const BottleState&, const FreeEnergyDecomposition&)>;

        // Minimum physical extents (clamped after every step)
        static constexpr float MIN_RADIUS = 0.005f;
        static constexpr float MIN_HEIGHT = 0.02f;

        BottleModel() = default;
        BottleModel(const BottleState& prior, const BottleModelParams& params);

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

        FreeEnergyDecomposition compute_free_energy_decomposition(
            const std::vector<Eigen::Vector3f>& points,
            const std::vector<float>& weights,
            std::size_t historical_count) const;

        // ── Gradient step ────────────────────────────────────────────────────────

        /**
         * Perform `params_.optimization_iters` gradient-descent steps on θ.
         * Returns the free energy after the last step.
         */
        float gradient_step(const std::vector<Eigen::Vector3f>& points,
                            const std::vector<float>& weights,
                            std::size_t historical_count = 0,
                            const IterationObserver& observer = {});

        // ── Pose uncertainty ───────────────────────────────────────────────────────

        /**
         * 3×3 position covariance (m²) from the Laplace approximation of F at the
         * current state: Σ = H⁻¹, with H the finite-difference Hessian of
         * compute_free_energy w.r.t. (cx, cy, cz). With many well-spread points the
         * radial/cap curvature is high ⇒ tight Σ; with few/no points only the
         * prior-position curvature (2·lambda_pos) survives ⇒ Σ inflates. This is the
         * P_bottle written on the room→bottle RT edge. Singular/ill-conditioned H
         * falls back to a large isotropic covariance (max_var).
         */
        Eigen::Matrix3f pose_covariance(const std::vector<Eigen::Vector3f>& points,
                                        const std::vector<float>& weights,
                                        float fd_step = 0.005f,
                                        float max_var = 1.0f) const;

        // ── State access ─────────────────────────────────────────────────────────

        const BottleState& state()  const { return state_; }
        const BottleState& prior()  const { return prior_; }
        const BottleModelParams& params() const { return params_; }

        void set_state(const BottleState& s) { state_ = s; apply_constraints(); }
        void set_prior(const BottleState& p) { prior_ = p; }

        /** Enforce positive, physically plausible radius and height. */
        void apply_constraints();

        /**
         * Axis-aligned bounding box of the cylinder in room frame.
         * Returns {min_corner, max_corner}.
         */
        std::pair<Eigen::Vector3f, Eigen::Vector3f> bounding_box() const;

    private:
        // Evaluate free energy for an arbitrary state (scalar float path)
        float fe_at(const BottleState& s,
                    const std::vector<Eigen::Vector3f>& pts,
                    const std::vector<float>& weights) const;

        FreeEnergyDecomposition fe_terms_at(const BottleState& s,
                                            const std::vector<Eigen::Vector3f>& pts,
                                            const std::vector<float>& weights,
                                            std::size_t historical_count) const;

        // Prior (size + position) energy for a given state
        float prior_energy(const BottleState& s) const;

        // SDF evaluated for a given explicit state
        float sdf_point_at(const Eigen::Vector3f& p, const BottleState& s) const;

        BottleState        state_;
        BottleState        prior_;
        BottleModelParams  params_;
};
