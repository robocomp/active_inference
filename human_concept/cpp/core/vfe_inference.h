// vfe_inference.h
// Active Inference (Laplace approximation) pose estimator over the 11-DOF human
// kinematic model. Maintains a belief q(theta) ~ N(mu, Sigma) and updates mu by a
// precision-weighted Gauss-Newton / Laplace step per frame.
//
// Ported from vfe_inference.py. Pure Eigen (float32); the Jacobian is computed by
// central finite differences through the FK + Kabsch chain (no autograd / torch).
#pragma once

#include <Eigen/Dense>
#include <array>
#include <optional>
#include <vector>

#include "body18.h"
#include "human_kinematic_model.h"

namespace rc::human
{
struct InferenceConfig
{
    std::vector<int> anchors = {1, 2, 5, 8, 11};

    // Observation noise base (used when no per-joint confidence is provided).
    float sigma_obs = 0.06f;

    // Dynamics prior noise (how fast angles may change frame-to-frame).
    float sigma_dyn = 0.25f;

    // Precision mapping from confidence.
    float sigma_min = 0.02f;
    float sigma_max = 0.15f;

    // Prior weights.
    float w_limits = 5.0f;
    float w_sym    = 1.0f;

    // AInf update settings.
    int   gn_steps = 2;       // Gauss-Newton / Laplace steps per frame
    float damping  = 1e-3f;   // Levenberg damping for stability
};

struct InferenceResult
{
    Vec11 mu;                                 // posterior mean of angles (11)
    Eigen::Matrix<float, 11, 11> Lambda;      // posterior precision approx (11x11)
    KpArray kp_pred_aligned;                  // (18,3) prediction aligned to live
    Eigen::Matrix3f R;
    Eigen::Vector3f t;
    KpArray diff;                             // (18,3) live - pred
    std::array<float, NUM_KP> per_l2;         // per-joint L2 (NaN where invalid)
    float mean_l2 = 0.f;
    float rmse    = 0.f;
    std::vector<int> used_anchors;
    int   valid_count = 0;
    float uncertainty_trace = 0.f;            // trace(cov) = trace(Lambda^-1)
};

class AInfLaplacePoseEstimator
{
public:
    AInfLaplacePoseEstimator(const HumanKinematicModel &model, const InferenceConfig &cfg);

    // live: (18,3) observed keypoints (NaN rows = missing).
    // conf: optional per-joint confidence in [0,100].
    InferenceResult infer(const KpArray &live,
                          const std::optional<std::array<float, NUM_KP>> &conf = std::nullopt);

    const Vec11 &mu() const { return mu_; }
    const Eigen::Matrix<float, 11, 11> &Lambda() const { return Lambda_; }

private:
    struct Aligned
    {
        KpArray pred;
        Eigen::Matrix3f R;
        Eigen::Vector3f t;
    };

    Aligned predict_aligned(const Vec11 &x, const KpArray &live,
                            const std::vector<int> &anchors) const;

    // Weighted, flattened residual over the valid joints (size 3*M).
    Eigen::VectorXf residual(const Vec11 &x, const KpArray &live,
                             const std::vector<int> &anchors,
                             const std::vector<int> &valid_idx,
                             const Eigen::VectorXf &sqrt_pi) const;

    const HumanKinematicModel &model_;
    InferenceConfig cfg_;
    JointLimits lim_;

    Vec11 mu_;
    Eigen::Matrix<float, 11, 11> Lambda_;
};

// Free functions exposed for testing / reuse.
struct KabschResult { Eigen::Matrix3f R; Eigen::Vector3f t; };
// src, dst: (N,3). Rigid alignment (rotation + translation, no scale).
KabschResult kabsch(const Eigen::MatrixXf &src, const Eigen::MatrixXf &dst);

float symmetry_prior(const Vec11 &x);
Vec11 symmetry_prior_grad(const Vec11 &x);
float joint_limits_prior(const Vec11 &x, const JointLimits &lim);
Vec11  joint_limits_prior_grad(const Vec11 &x, const JointLimits &lim);

}  // namespace rc::human
