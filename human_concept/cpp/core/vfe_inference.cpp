// vfe_inference.cpp  — port of vfe_inference.py
#include "vfe_inference.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace rc::human
{
namespace
{
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

bool row_finite(const KpArray &kp, int i)
{
    return std::isfinite(kp(i, 0)) && std::isfinite(kp(i, 1)) && std::isfinite(kp(i, 2));
}

inline float relu(float v) { return v > 0.f ? v : 0.f; }
}  // namespace

// ---------------- Kabsch ----------------
KabschResult kabsch(const Eigen::MatrixXf &src, const Eigen::MatrixXf &dst)
{
    const Eigen::RowVector3f src_mean = src.colwise().mean();
    const Eigen::RowVector3f dst_mean = dst.colwise().mean();

    const Eigen::MatrixXf X = src.rowwise() - src_mean;
    const Eigen::MatrixXf Y = dst.rowwise() - dst_mean;

    const Eigen::Matrix3f H = X.transpose() * Y;

    Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Matrix3f U = svd.matrixU();
    const Eigen::Matrix3f V = svd.matrixV();  // numpy's V = Vh.T

    // Proper rotation (avoid reflection).
    float d = (V * U.transpose()).determinant();
    d = (d > 0.f) ? 1.f : ((d < 0.f) ? -1.f : 1.f);
    Eigen::Matrix3f D = Eigen::Matrix3f::Identity();
    D(2, 2) = d;

    KabschResult out;
    out.R = V * D * U.transpose();
    out.t = dst_mean.transpose() - out.R * src_mean.transpose();
    return out;
}

// ---------------- priors ----------------
// F_sym = (shL_yaw + shR_yaw)^2 + (shL_pitch - shR_pitch)^2 + (shL_roll + shR_roll)^2
//         + (elL - elR)^2
float symmetry_prior(const Vec11 &x)
{
    const float sh = (x(0) + x(3)) * (x(0) + x(3))
                   + (x(1) - x(4)) * (x(1) - x(4))
                   + (x(2) + x(5)) * (x(2) + x(5));
    const float hinge = (x(6) - x(7)) * (x(6) - x(7));
    return sh + hinge;
}

Vec11 symmetry_prior_grad(const Vec11 &x)
{
    Vec11 g = Vec11::Zero();
    const float a = 2.f * (x(0) + x(3));
    const float b = 2.f * (x(1) - x(4));
    const float c = 2.f * (x(2) + x(5));
    const float e = 2.f * (x(6) - x(7));
    g(0) = a;  g(3) = a;
    g(1) = b;  g(4) = -b;
    g(2) = c;  g(5) = c;
    g(6) = e;  g(7) = -e;
    return g;
}

// penalty(v) = relu(vmin - v)^2 + relu(v - vmax)^2  (per component)
float joint_limits_prior(const Vec11 &x, const JointLimits &lim)
{
    auto pen = [](float v, float vmin, float vmax) {
        const float lo = relu(vmin - v);
        const float hi = relu(v - vmax);
        return lo * lo + hi * hi;
    };
    float p = 0.f;
    for (int k = 0; k < 3; ++k)  // sh_L
        p += pen(x(k), lim.sh_min(k), lim.sh_max(k));
    for (int k = 0; k < 3; ++k)  // sh_R
        p += pen(x(3 + k), lim.sh_min(k), lim.sh_max(k));
    p += pen(x(6), lim.el_min, lim.el_max);
    p += pen(x(7), lim.el_min, lim.el_max);
    p += pen(x(8), lim.lb_x_min, lim.lb_x_max);
    p += pen(x(9), lim.lb_z_min, lim.lb_z_max);
    p += pen(x(10), lim.lb_roll_min, lim.lb_roll_max);
    return p;
}

Vec11 joint_limits_prior_grad(const Vec11 &x, const JointLimits &lim)
{
    auto dpen = [](float v, float vmin, float vmax) {
        return -2.f * relu(vmin - v) + 2.f * relu(v - vmax);
    };
    Vec11 g = Vec11::Zero();
    for (int k = 0; k < 3; ++k)
        g(k) = dpen(x(k), lim.sh_min(k), lim.sh_max(k));
    for (int k = 0; k < 3; ++k)
        g(3 + k) = dpen(x(3 + k), lim.sh_min(k), lim.sh_max(k));
    g(6)  = dpen(x(6), lim.el_min, lim.el_max);
    g(7)  = dpen(x(7), lim.el_min, lim.el_max);
    g(8)  = dpen(x(8), lim.lb_x_min, lim.lb_x_max);
    g(9)  = dpen(x(9), lim.lb_z_min, lim.lb_z_max);
    g(10) = dpen(x(10), lim.lb_roll_min, lim.lb_roll_max);
    return g;
}

// Anti arm-cross (sidedness) penalty in the MODEL frame, where +X = right, -X = left. A wrist/elbow
// should not cross PAST the opposite shoulder by more than `margin`: penalise relu(over)^2. Allows
// folded arms (crossing the midline) but rejects gross L/R swaps. Scalar; gradient taken by FD.
float arm_cross_penalty(const KpArray &kp, float margin)
{
    const float rsx = kp(KP::R_SHOULDER, 0);   // right shoulder x (most +X)
    const float lsx = kp(KP::L_SHOULDER, 0);   // left  shoulder x (most -X)
    auto sq = [](float v) { return v > 0.f ? v * v : 0.f; };
    float p = 0.f;
    // Left joints (should stay <= rsx): penalise going past the right shoulder.
    p += sq(kp(KP::L_WRIST, 0) - rsx - margin);
    p += sq(kp(KP::L_ELBOW, 0) - rsx - margin);
    // Right joints (should stay >= lsx): penalise going past the left shoulder.
    p += sq(lsx - kp(KP::R_WRIST, 0) - margin);
    p += sq(lsx - kp(KP::R_ELBOW, 0) - margin);
    return p;
}

// Symmetric soft rate limit: penalty(u) = relu(u-t)^2 + relu(-u-t)^2 with threshold t>=0. Returns
// d penalty / du. Used for the velocity limit (u = Δθ, t = ω_max·dt) and the acceleration limit
// (u = θ_t − 2θ_{t-1} + θ_{t-2}, t = α_max·dt²). Both u are linear in θ_t with ∂u/∂θ_t = 1, so this
// equals the gradient w.r.t. θ_t.
inline float rate_limit_grad1(float u, float t)
{
    return 2.f * relu(u - t) - 2.f * relu(-u - t);
}

// ---------------- estimator ----------------
AInfLaplacePoseEstimator::AInfLaplacePoseEstimator(const HumanKinematicModel &model,
                                                   const InferenceConfig &cfg)
    : model_(model), cfg_(cfg), lim_(default_joint_limits_radians())
{
    mu_     = Vec11::Zero();
    Lambda_ = Eigen::Matrix<float, 11, 11>::Identity();
}

KpArray AInfLaplacePoseEstimator::predict_aligned_kp(const Vec11 &x, const KpArray &live) const
{
    return predict_aligned(x, live, cfg_.anchors).pred;
}

AInfLaplacePoseEstimator::Aligned
AInfLaplacePoseEstimator::predict_aligned(const Vec11 &x, const KpArray &live,
                                          const std::vector<int> &anchors) const
{
    const KpArray kp_pred = model_.forward(x);

    const int A = static_cast<int>(anchors.size());
    Eigen::MatrixXf src(A, 3), dst(A, 3);
    for (int k = 0; k < A; ++k)
    {
        src.row(k) = kp_pred.row(anchors[k]);
        dst.row(k) = live.row(anchors[k]);
    }
    const KabschResult kb = kabsch(src, dst);

    Aligned out;
    out.R = kb.R;
    out.t = kb.t;
    out.pred.resize(NUM_KP, 3);
    for (int i = 0; i < NUM_KP; ++i)
        out.pred.row(i) = (kb.R * kp_pred.row(i).transpose() + kb.t).transpose();
    return out;
}

Eigen::VectorXf
AInfLaplacePoseEstimator::residual(const Vec11 &x, const KpArray &live,
                                   const std::vector<int> &anchors,
                                   const std::vector<int> &valid_idx,
                                   const Eigen::VectorXf &sqrt_pi) const
{
    const Aligned a = predict_aligned(x, live, anchors);
    const int M = static_cast<int>(valid_idx.size());
    Eigen::VectorXf r(3 * M);
    for (int k = 0; k < M; ++k)
    {
        const int i = valid_idx[k];
        const float w = sqrt_pi(k);
        r(3 * k + 0) = w * (live(i, 0) - a.pred(i, 0));
        r(3 * k + 1) = w * (live(i, 1) - a.pred(i, 1));
        r(3 * k + 2) = w * (live(i, 2) - a.pred(i, 2));
    }
    return r;
}

InferenceResult AInfLaplacePoseEstimator::infer(
    const KpArray &live, const std::optional<std::array<float, NUM_KP>> &conf, float dt_override)
{
    const int D = DOF;
    // Real seconds since the previous fit drives the speed/accel thresholds; fall back to cfg_.dt.
    const float dt = (dt_override > 0.f) ? dt_override : cfg_.dt;

    // Valid joints.
    std::vector<int> valid_idx;
    valid_idx.reserve(NUM_KP);
    for (int i = 0; i < NUM_KP; ++i)
        if (row_finite(live, i)) valid_idx.push_back(i);
    const int valid_count = static_cast<int>(valid_idx.size());

    // Choose anchors that are valid; fall back to all-valid; else bail.
    std::vector<int> anchors;
    for (int i : cfg_.anchors)
        if (i >= 0 && i < NUM_KP && row_finite(live, i)) anchors.push_back(i);
    if (static_cast<int>(anchors.size()) < 3)
        anchors = valid_idx;

    auto make_nan_result = [&](const std::vector<int> &used) {
        InferenceResult res;
        res.mu = mu_;
        res.Lambda = Lambda_;
        res.kp_pred_aligned = KpArray::Constant(kNaN);
        res.R = Eigen::Matrix3f::Identity();
        res.t = Eigen::Vector3f::Zero();
        res.diff = KpArray::Constant(kNaN);
        res.per_l2.fill(kNaN);
        res.mean_l2 = kNaN;
        res.rmse = kNaN;
        res.used_anchors = used;
        res.valid_count = valid_count;
        res.uncertainty_trace = std::numeric_limits<float>::infinity();
        res.pos_std_milli.fill(-1.f);
        return res;
    };

    if (static_cast<int>(anchors.size()) < 3)
        return make_nan_result(anchors);

    // Dynamics prior: theta_t ~ N(theta_{t-1}, sigma_dyn^2 I).
    const Vec11 mu_prior = mu_;
    const Eigen::Matrix<float, 11, 11> Lambda_dyn =
        Eigen::Matrix<float, 11, 11>::Identity() * (1.0f / (cfg_.sigma_dyn * cfg_.sigma_dyn));

    // Per-DOF speed/accel thresholds (per frame). Angle DOFs use omega/alpha (rad); the two
    // lower-body translation DOFs (8=lb_x, 9=lb_z) use vlin/alin (m). Gated by frames_seen_ so the
    // initial snap-to-pose isn't limited (velocity needs ≥1 prior fit, acceleration ≥2).
    Vec11 d_vel, d_acc;
    for (int k = 0; k < 11; ++k)
    {
        const bool is_trans = (k == 8 or k == 9);
        d_vel(k) = (is_trans ? cfg_.vlin_max : cfg_.omega_max) * dt;
        d_acc(k) = (is_trans ? cfg_.alin_max : cfg_.alpha_max) * dt * dt;
    }
    const bool apply_vel = cfg_.w_vel > 0.f and dt > 0.f and frames_seen_ >= 1;
    const bool apply_acc = cfg_.w_acc > 0.f and dt > 0.f and frames_seen_ >= 2;

    // Per-valid-joint precision pi (M,) and its sqrt.
    const int M = valid_count;
    Eigen::VectorXf pi(M), sqrt_pi(M);
    if (!conf.has_value())
    {
        const float p = 1.0f / (cfg_.sigma_obs * cfg_.sigma_obs);
        pi.setConstant(p);
    }
    else
    {
        for (int k = 0; k < M; ++k)
        {
            float cv = (*conf)[valid_idx[k]];
            cv = std::clamp(cv, 0.0f, 100.0f) / 100.0f;
            const float sigma = cfg_.sigma_min + (cfg_.sigma_max - cfg_.sigma_min) * (1.0f - cv);
            pi(k) = 1.0f / (sigma * sigma);
        }
    }
    sqrt_pi = pi.cwiseSqrt();
    const float Mf = std::max(1.0f, static_cast<float>(M));

    // Gauss-Newton / Laplace steps.
    const float eps = 1e-3f;  // central finite-difference step (angles ~ O(1))
    for (int step = 0; step < cfg_.gn_steps; ++step)
    {
        const Vec11 x = mu_;  // expansion point for this step

        const Eigen::VectorXf r = residual(x, live, anchors, valid_idx, sqrt_pi);

        // Finite-difference Jacobian J = dr/dmu  (3M x D), central differences.
        Eigen::MatrixXf J(3 * M, D);
        for (int j = 0; j < D; ++j)
        {
            Vec11 xp = x, xm = x;
            xp(j) += eps;
            xm(j) -= eps;
            const Eigen::VectorXf rp = residual(xp, live, anchors, valid_idx, sqrt_pi);
            const Eigen::VectorXf rm = residual(xm, live, anchors, valid_idx, sqrt_pi);
            J.col(j) = (rp - rm) / (2.0f * eps);
        }

        // Gradient of free energy F = F_like + F_dyn + w_sym F_sym + w_lim F_lim.
        const Vec11 dmu = x - mu_prior;
        Vec11 g = (J.transpose() * r) / Mf
                + Lambda_dyn * dmu
                + cfg_.w_sym * symmetry_prior_grad(x)
                + cfg_.w_limits * joint_limits_prior_grad(x, lim_);

        // Speed / acceleration limits (soft hinge, gradient-only like sym/limits). Δθ = x − θ_{t-1};
        // 2nd difference = x − 2θ_{t-1} + θ_{t-2}.
        if (apply_vel)
            for (int k = 0; k < 11; ++k)
                g(k) += cfg_.w_vel * rate_limit_grad1(x(k) - mu_prior(k), d_vel(k));
        if (apply_acc)
            for (int k = 0; k < 11; ++k)
                g(k) += cfg_.w_acc * rate_limit_grad1(x(k) - 2.f * mu_prior(k) + mu_prev2_(k), d_acc(k));

        // Sidedness (anti arm-cross) prior — finite-difference gradient of the scalar penalty on the
        // forward pose. Only when active (a wrist/elbow has crossed past the opposite shoulder), so
        // it's free in the common case. Gradient-only (low weight, rare), like the limits prior.
        if (cfg_.w_cross > 0.f)
        {
            const float pc0 = arm_cross_penalty(model_.forward(x), cfg_.arm_cross_margin);
            if (pc0 > 0.f)
                for (int k = 0; k < 11; ++k)
                {
                    Vec11 xc = x; xc(k) += eps;
                    const float pck = arm_cross_penalty(model_.forward(xc), cfg_.arm_cross_margin);
                    g(k) += cfg_.w_cross * (pck - pc0) / eps;
                }
        }

        // Gauss-Newton Hessian: likelihood + dynamics + damping (sym/limits via grad only).
        Eigen::Matrix<float, 11, 11> H =
            (J.transpose() * J) / Mf
            + Lambda_dyn + cfg_.damping * Eigen::Matrix<float, 11, 11>::Identity();

        // Add the curvature of the active speed/accel hinges (diagonal: ∂²penalty/∂θ² = 2 when active,
        // and ∂u/∂θ = 1 for both Δθ and the 2nd difference). Gradient-only here overshoots once the
        // hinges bite every frame; the matching Hessian keeps the Newton step properly scaled.
        if (apply_vel)
            for (int k = 0; k < 11; ++k)
            {
                const float u = x(k) - mu_prior(k);
                if (std::abs(u) > d_vel(k)) H(k, k) += 2.f * cfg_.w_vel;
            }
        if (apply_acc)
            for (int k = 0; k < 11; ++k)
            {
                const float u = x(k) - 2.f * mu_prior(k) + mu_prev2_(k);
                if (std::abs(u) > d_acc(k)) H(k, k) += 2.f * cfg_.w_acc;
            }

        // Neutral-pose prior: a weak quadratic pull toward rest (0). Applied to the arm angle DOFs
        // (shoulders 0..5, elbows 6,7) AND lb_roll (10, body twist). These are the under-observed /
        // degenerate directions — the likelihood is flat there, so the estimate would otherwise drift
        // along the null space at constant FE (lb_roll is only constrained by the hips, so it swings
        // wildly → unnatural body twist). The pull pins them without biasing well-observed DOFs (their
        // large likelihood curvature dominates this small term). NOT applied to lb_x/lb_z (8,9) — that
        // would drag the person's position. Proper quadratic, so add the Hessian for a well-scaled step.
        if (cfg_.w_neutral > 0.f)
            for (int k = 0; k < 11; ++k)
            {
                if (k == 8 or k == 9) continue;           // skip lower-body translation (position)
                g(k)    += cfg_.w_neutral * 2.f * x(k);   // d/dθ of (θ - 0)^2
                H(k, k) += cfg_.w_neutral * 2.f;
            }

        const Vec11 delta = H.ldlt().solve(g);

        mu_     = x - delta;
        Lambda_ = H;
    }

    // Hard joint-limit projection: the soft limits prior can be overrun by a bad observation (robot
    // motion / YOLO glitch drives a shoulder to -257° etc.), so clamp every DOF to its physical range.
    {
        Vec11 lo, hi;
        lo.segment<3>(0) = lim_.sh_min; lo.segment<3>(3) = lim_.sh_min;
        hi.segment<3>(0) = lim_.sh_max; hi.segment<3>(3) = lim_.sh_max;
        lo(6) = lim_.el_min; lo(7) = lim_.el_min; hi(6) = lim_.el_max; hi(7) = lim_.el_max;
        lo(8) = lim_.lb_x_min; lo(9) = lim_.lb_z_min; lo(10) = lim_.lb_roll_min;
        hi(8) = lim_.lb_x_max; hi(9) = lim_.lb_z_max; hi(10) = lim_.lb_roll_max;
        mu_ = mu_.cwiseMax(lo).cwiseMin(hi);
    }

    // Innovation reject: a per-frame jump larger than max_innovation (rad) on any DOF is a glitch — a
    // static person can't move that fast, and during robot motion a transient bad room transform makes
    // the fit lurch. Hold the previous belief (mu_prior) rather than chase it; outputs below recompute
    // from the held mu_, so the controller target doesn't move and FE reflects the good fit.
    bool rejected = false;
    if (frames_seen_ > 0 and cfg_.max_innovation > 0.f
        and (mu_ - mu_prior).cwiseAbs().maxCoeff() > cfg_.max_innovation)
    {
        mu_ = mu_prior;
        rejected = true;
    }

    // Limit diagnostics for the gated CSV: how many DOFs the converged fit left at/over the bounds.
    int vel_clamped = 0, acc_clamped = 0;
    if (apply_vel)
        for (int k = 0; k < 11; ++k)
            if (std::abs(mu_(k) - mu_prior(k)) > d_vel(k)) ++vel_clamped;
    if (apply_acc)
        for (int k = 0; k < 11; ++k)
            if (std::abs(mu_(k) - 2.f * mu_prior(k) + mu_prev2_(k)) > d_acc(k)) ++acc_clamped;

    // Advance the speed/accel history: this fit's theta_{t-1} becomes next call's theta_{t-2}.
    mu_prev2_ = mu_prior;
    ++frames_seen_;

    // Final forward pass for outputs.
    const Aligned a = predict_aligned(mu_, live, anchors);

    InferenceResult res;
    res.mu = mu_;
    res.Lambda = Lambda_;
    res.kp_pred_aligned = a.pred;
    res.R = a.R;
    res.t = a.t;
    res.diff = KpArray::Constant(kNaN);
    res.per_l2.fill(kNaN);

    float sum_l2 = 0.f, sum_sq = 0.f;
    int n = 0;
    for (int i = 0; i < NUM_KP; ++i)
    {
        if (!row_finite(live, i)) continue;
        const Eigen::Vector3f dvec = live.row(i).transpose() - a.pred.row(i).transpose();
        res.diff.row(i) = dvec.transpose();
        const float l2 = dvec.norm();
        res.per_l2[i] = l2;
        sum_l2 += l2;
        sum_sq += l2 * l2;
        ++n;
    }
    res.mean_l2 = n ? sum_l2 / n : kNaN;
    res.rmse = n ? std::sqrt(sum_sq / n) : kNaN;
    res.used_anchors = anchors;
    res.valid_count = valid_count;

    // Uncertainty proxy: trace of covariance = trace(Lambda^-1).
    const Eigen::Matrix<float, 11, 11> cov = Lambda_.inverse();
    const float tr = cov.trace();
    res.uncertainty_trace = std::isfinite(tr) ? tr : std::numeric_limits<float>::infinity();
    res.dt = dt;
    res.vel_clamped = vel_clamped;
    res.acc_clamped = acc_clamped;
    res.rejected = rejected;

    // Per-keypoint positional posterior std (mm). Propagate cov through the FK Jacobian J=∂kp/∂x at
    // mu (forward differences). trace(J·cov·Jᵀ) is rotation-invariant, so the model-frame Jacobian
    // gives the same magnitude as the Kabsch-aligned frame — no need to apply R.
    res.pos_std_milli.fill(-1.f);
    if (std::isfinite(tr))
    {
        const KpArray kp_base = model_.forward(mu_);
        Eigen::Matrix<float, NUM_KP * 3, 11> J;
        constexpr float eps = 1e-4f;
        for (int d = 0; d < 11; ++d)
        {
            Vec11 xp = mu_;
            xp(d) += eps;
            const KpArray dkp = (model_.forward(xp) - kp_base) / eps;   // (18,3)
            for (int i = 0; i < NUM_KP; ++i)
                J.block<3, 1>(i * 3, d) = dkp.row(i).transpose();
        }
        for (int i = 0; i < NUM_KP; ++i)
        {
            const Eigen::Matrix<float, 3, 11> Ji = J.block<3, 11>(i * 3, 0);
            const float var = (Ji * cov * Ji.transpose()).trace();
            res.pos_std_milli[i] = (std::isfinite(var) and var >= 0.f) ? 1000.0f * std::sqrt(var) : -1.f;
        }
    }

    return res;
}

}  // namespace rc::human
