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
    // Sidedness (anti arm-cross) prior: soft penalty when a wrist/elbow crosses past the OPPOSITE
    // shoulder by more than arm_cross_margin (m). Allows folded arms, rejects YOLO L/R swaps. 0 = off.
    float w_cross         = 0.0f;   // off in pure core (agent sets it); margin allows folding
    float arm_cross_margin = 0.05f;
    // Neutral-pose prior: weak L2 pull of the arm angle DOFs toward rest (arms down), pinning the
    // under-observed null-space so it doesn't drift at constant FE. 0 = off (pure core default).
    float w_neutral       = 0.0f;

    // Kinematic-plausibility limits (soft one-sided hinge penalties beyond the bound, dt-aware;
    // applied to the per-frame change in theta). A weight of 0 disables that term. Angle DOFs use
    // omega/alpha (rad); the two lower-body translation DOFs (lb_x, lb_z) use vlin/alin (m).
    float dt        = 0.1f;    // s between successful fits (fallback; caller passes the measured dt)
    float w_vel     = 0.0f;    // velocity-limit penalty weight (0 = off in pure core; agent sets it)
    float w_acc     = 0.0f;    // acceleration-limit penalty weight (0 = off in pure core)
    float omega_max = 8.0f;    // max angular speed (rad/s) ≈ 460°/s — a fast limb, not normal motion
    float alpha_max = 80.0f;   // max angular acceleration (rad/s²)
    float vlin_max  = 3.0f;    // max lower-body linear speed (m/s)
    float alin_max  = 30.0f;   // max lower-body linear acceleration (m/s²)

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
    // Per-keypoint positional posterior std in mm: sqrt(trace(J·cov·Jᵀ)) with J=∂kp/∂x at mu. The
    // keypoints (e.g. wrists) are NOT state DOFs, so this propagates the angle covariance through the
    // FK Jacobian to give the positional uncertainty of each joint. -1 where degenerate/unobserved.
    std::array<float, NUM_KP> pos_std_milli;
    // Limit diagnostics (for the gated fit CSV): dt used this fit, and how many DOFs the converged
    // estimate left at/over the velocity / acceleration thresholds (0 = no limit firing).
    float dt = 0.f;
    int   vel_clamped = 0;
    int   acc_clamped = 0;
};

class AInfLaplacePoseEstimator
{
public:
    AInfLaplacePoseEstimator(const HumanKinematicModel &model, const InferenceConfig &cfg);

    // live: (18,3) observed keypoints (NaN rows = missing).
    // conf: optional per-joint confidence in [0,100].
    // dt_override: actual seconds since the previous fit (drives the speed/accel limits). <=0 → use
    //   cfg_.dt. The CALLER should pass the MEASURED inter-fit interval — it is NOT the loop period
    //   when fits are gated by the data rate (e.g. 20 Hz loop consuming a 10 Hz skeleton stream).
    InferenceResult infer(const KpArray &live,
                          const std::optional<std::array<float, NUM_KP>> &conf = std::nullopt,
                          float dt_override = -1.f);

    const Vec11 &mu() const { return mu_; }
    const Eigen::Matrix<float, 11, 11> &Lambda() const { return Lambda_; }

    // Forward the model at x and Kabsch-align it to `live` (using the configured torso anchors),
    // giving room-frame keypoints. Used by the output controller to render a SMOOTHED command pose
    // without re-running the fit. Does not touch the belief state.
    KpArray predict_aligned_kp(const Vec11 &x, const KpArray &live) const;

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

    // History for the speed/accel limits: mu_ holds theta_{t-1} between calls; mu_prev2_ holds
    // theta_{t-2}. frames_seen_ gates the limits on so the initial snap-to-pose isn't penalised
    // (velocity needs ≥1 prior fit, acceleration ≥2).
    Vec11 mu_prev2_ = Vec11::Zero();
    int   frames_seen_ = 0;
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
