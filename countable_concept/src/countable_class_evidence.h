// countable_class_evidence — is this cloud a <class>, or just some other furniture? Pure math, no Qt, no DSR.
//
// Λ (nats) is the log evidence ratio between "the fitted size belongs to this class" and "it belongs to some other
// furniture". Both are Gaussians on (H, w, h) CONVOLVED with the fit's own uncertainty (Σ_lik), so a poorly-seen shape
// is neither accepted nor rejected: with no information Λ = 0, with wide information Λ → 0. No threshold anywhere; the
// caller adds logit(π_cls) and reads a probability.
// Size order is (H, w, h). The w↔h labelling of a box is arbitrary, so BOTH sides are a 50/50 mixture over it.
#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

namespace rc::countable
{
struct SizeGaussian
{
    Eigen::Vector3f mean = Eigen::Vector3f::Zero();   // (H, w, h)
    Eigen::Matrix3f info = Eigen::Matrix3f::Zero();   // information (Σ⁻¹); zero = no data
};

struct ClassShapePrior
{
    float H_mean = 0, H_std = 1, w_mean = 0, w_std = 1, h_mean = 0, h_std = 1;
    float H_top = 0;            // tallest declared part (bed: headboard 1.25; else H_mean)
    float p_label_true = 0.5f;  // π_cls
};

namespace detail
{
inline double log_sum_exp(double a, double b) { const double m = std::max(a, b); return m + std::log(std::exp(a - m) + std::exp(b - m)); }

// log N(d; 0, S) for a k×k covariance S
inline double log_gauss(const Eigen::VectorXd& d, const Eigen::MatrixXd& S)
{
    const Eigen::LDLT<Eigen::MatrixXd> ldlt(S);
    const Eigen::VectorXd D = ldlt.vectorD();
    double logdet = 0; for (int i = 0; i < D.size(); ++i) logdet += std::log(D[i]);
    return -0.5 * (d.dot(ldlt.solve(d)) + logdet + double(d.size()) * std::log(2.0 * std::numbers::pi));
}

inline double log_phi(double x) { return -0.5 * x * x - 0.5 * std::log(2.0 * std::numbers::pi); }
inline double log_normal(double x, double m, double s) { return log_phi((x - m) / s) - std::log(s); }
}

inline float class_log_evidence(const SizeGaussian& lik, const ClassShapePrior& c, float alt_std)
{
    using namespace detail;
    const Eigen::Matrix3d info = lik.info.cast<double>();
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(0.5 * (info + info.transpose()));
    std::vector<int> idx;                                          // the directions the data inform
    for (int i = 0; i < 3; ++i) if (es.eigenvalues()[i] > 0.0) idx.push_back(i);
    if (idx.empty()) return 0.0f;
    const int k = int(idx.size());
    Eigen::MatrixXd U(3, k); Eigen::MatrixXd Sl = Eigen::MatrixXd::Zero(k, k);
    for (int j = 0; j < k; ++j) { U.col(j) = es.eigenvectors().col(idx[j]); Sl(j, j) = 1.0 / es.eigenvalues()[idx[j]]; }

    const Eigen::Vector3d th = lik.mean.cast<double>();
    const Eigen::Matrix3d Sc = Eigen::Vector3d(double(c.H_std) * c.H_std, double(c.w_std) * c.w_std, double(c.h_std) * c.h_std).asDiagonal();
    const Eigen::Matrix3d Sa = Eigen::Matrix3d::Identity() * (double(alt_std) * alt_std);
    const Eigen::MatrixXd Scu = U.transpose() * Sc * U + Sl, Sau = U.transpose() * Sa * U + Sl;

    // the labelling: (w,h) as declared, or swapped (mean AND std swap)
    auto side = [&](const Eigen::MatrixXd& Su, bool swapped_std, const Eigen::Matrix3d& S3)
    {
        double lp[2];
        for (int s = 0; s < 2; ++s)
        {
            const Eigen::Vector3d mu = s == 0 ? Eigen::Vector3d(c.H_mean, c.w_mean, c.h_mean) : Eigen::Vector3d(c.H_mean, c.h_mean, c.w_mean);
            Eigen::MatrixXd S = Su;
            if (swapped_std and s == 1)
            {
                Eigen::Matrix3d P = S3; std::swap(P(1, 1), P(2, 2));
                S = U.transpose() * P * U + Sl;
            }
            lp[s] = log_gauss(U.transpose() * (th - mu), S);
        }
        return log_sum_exp(lp[0], lp[1]) - std::log(2.0);
    };
    return float(side(Scu, true, Sc) - side(Sau, false, Sa));
}

// Birth: class side = (1−ε)·[Uniform(H_mean, H_top) ⊛ N(0,σ_z²)] + ε·N(H_mean, alt_std²); other = N(H_mean, alt_std²).
// For H_top ≤ H_mean the uniform degenerates to N(H_mean, H_std²+σ_z²).
inline float birth_height_log_evidence(float z_extent, const ClassShapePrior& c, float sigma_z, float eps, float alt_std)
{
    using namespace detail;
    const double x = z_extent, sz = sigma_z;
    const double l_alt = log_normal(x, c.H_mean, alt_std);
    double l_cls;
    if (c.H_top <= c.H_mean) l_cls = log_normal(x, c.H_mean, std::sqrt(double(c.H_std) * c.H_std + sz * sz));
    else
    {
        const double a = c.H_mean, b = c.H_top;
        const double hi = 0.5 * std::erfc(-(x - a) / (sz * std::numbers::sqrt2));     // Φ((x−a)/σ)
        const double lo = 0.5 * std::erfc(-(x - b) / (sz * std::numbers::sqrt2));     // Φ((x−b)/σ)
        l_cls = std::log(std::max(hi - lo, 1e-300) / (b - a));
    }
    const double mix = log_sum_exp(std::log1p(-double(eps)) + l_cls, std::log(double(eps)) + l_alt);
    return float(mix - l_alt);
}

inline float robust_z_extent(const std::vector<Eigen::Vector3f>& pts)      // q0.97 − q0.03 of z
{
    if (pts.size() < 2) return 0.0f;
    std::vector<float> z; z.reserve(pts.size());
    for (const auto& p : pts) z.push_back(p.z());
    std::ranges::sort(z);
    auto q = [&](double t) { return z[size_t(std::lround(t * double(z.size() - 1)))]; };
    return q(0.97) - q(0.03);
}

inline float class_probability(float L) { return 1.0f / (1.0f + std::exp(-L)); }
inline float logit(float p) { return std::log(p / (1.0f - p)); }
}
