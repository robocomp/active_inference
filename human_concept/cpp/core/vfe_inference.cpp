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

// ---------------- estimator ----------------
AInfLaplacePoseEstimator::AInfLaplacePoseEstimator(const HumanKinematicModel &model,
                                                   const InferenceConfig &cfg)
    : model_(model), cfg_(cfg), lim_(default_joint_limits_radians())
{
    mu_     = Vec11::Zero();
    Lambda_ = Eigen::Matrix<float, 11, 11>::Identity();
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
    const KpArray &live, const std::optional<std::array<float, NUM_KP>> &conf)
{
    const int D = DOF;

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
        return res;
    };

    if (static_cast<int>(anchors.size()) < 3)
        return make_nan_result(anchors);

    // Dynamics prior: theta_t ~ N(theta_{t-1}, sigma_dyn^2 I).
    const Vec11 mu_prior = mu_;
    const Eigen::Matrix<float, 11, 11> Lambda_dyn =
        Eigen::Matrix<float, 11, 11>::Identity() * (1.0f / (cfg_.sigma_dyn * cfg_.sigma_dyn));

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

        // Gauss-Newton Hessian: likelihood + dynamics + damping (sym/limits via grad only).
        const Eigen::Matrix<float, 11, 11> H_like =
            (J.transpose() * J) / Mf;
        const Eigen::Matrix<float, 11, 11> H =
            H_like + Lambda_dyn + cfg_.damping * Eigen::Matrix<float, 11, 11>::Identity();

        const Vec11 delta = H.ldlt().solve(g);

        mu_     = x - delta;
        Lambda_ = H;
    }

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

    return res;
}

}  // namespace rc::human
