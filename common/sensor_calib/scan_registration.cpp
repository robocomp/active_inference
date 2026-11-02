#include "scan_registration.h"

#include <nanoflann.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdint>
#include <numbers>

namespace sc::calib
{
namespace
{
struct Cloud
{
    const std::vector<Eigen::Vector2d>& pts;
    size_t kdtree_get_point_count() const { return pts.size(); }
    double kdtree_get_pt(size_t i, size_t d) const { return pts[i][int(d)]; }
    template <class BBox> bool kdtree_get_bbox(BBox&) const { return false; }
};
using Tree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, Cloud>, Cloud, 2>;

// log of the regularised lower incomplete gamma P(a, x) (series for x < a+1, Lentz continued fraction otherwise).
double log_gamma_p(double a, double x)
{
    if (x <= 0.0) return -std::numeric_limits<double>::infinity();
    if (x < a + 1.0)
    {
        double term = 1.0 / a, sum = term;
        for (int n = 1; n < 500 and term > 1e-17 * sum; ++n) { term *= x / (a + n); sum += term; }
        return -x + a * std::log(x) - std::lgamma(a) + std::log(sum);
    }
    const double tiny = 1e-300;
    double b = x + 1.0 - a, c = 1.0 / tiny, d = 1.0 / b, h = d;
    for (int i = 1; i < 500; ++i)
    {
        const double an = -i * (i - a);
        b += 2.0;
        d = an * d + b; if (std::abs(d) < tiny) d = tiny;
        c = b + an / c; if (std::abs(c) < tiny) c = tiny;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::abs(del - 1.0) < 1e-15) break;
    }
    const double log_q = -x + a * std::log(x) - std::lgamma(a) + std::log(h);
    return std::log1p(-std::min(std::exp(log_q), 1.0 - 1e-300));
}

// P(real structure | lambda) by Bayesian model comparison, prior odds 1:1.
//   H0 (noise only): lambda ~ Gamma(shape 1/rel^2, scale F rel^2)      (mean F, relative sd rel)
//   H1 (real):       lambda = G + S, G as in H0, S ~ Exponential(mean mu)  (a wall end's worth of information)
// f1(lambda) = beta e^{-beta lambda} (1 - beta theta)^{-a} P(a, lambda (1 - beta theta) / theta), beta = 1/mu.
double p_real(double lambda, double F, double rel, double mu)
{
    if (not (F > 0.0)) return lambda > 0.0 ? 1.0 : 0.0;     // nothing is manufactured on this axis
    if (not (lambda > 0.0)) return 0.0;
    const double a = 1.0 / (rel * rel), theta = F * rel * rel, beta = 1.0 / mu;
    if (beta * theta >= 1.0) return 1.0;                    // prior flatter than the noise's own scale: no evidence for H0
    const double lg0 = -std::lgamma(a) - a * std::log(theta) + (a - 1.0) * std::log(lambda) - lambda / theta;
    const double lg1 = std::log(beta) - beta * lambda - a * std::log1p(-beta * theta)
                     + log_gamma_p(a, lambda * (1.0 - beta * theta) / theta);
    return 1.0 / (1.0 + std::exp(std::clamp(lg0 - lg1, -700.0, 700.0)));
}
}  // namespace

ScanRegResult register_scans_2d(const std::vector<Eigen::Vector2f>& ref, const std::vector<Eigen::Vector2f>& cur,
                                const Eigen::Vector3d& init_xyt, const ScanRegConfig& cfg)
{
    ScanRegResult out; out.xyt = init_xyt;
    const int k = std::max(4, cfg.normal_k);
    if (int(ref.size()) < k or cur.size() < 3) return out;

    // double copy: the pose steps converge to ~1e-7 m and a float query would quantise them
    const std::vector<Eigen::Vector2d> refd = [&] { std::vector<Eigen::Vector2d> v; v.reserve(ref.size()); for (const auto& p : ref) v.push_back(p.cast<double>()); return v; }();
    const Cloud cloud{refd};
    Tree tree(2, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    // PCA normal of each reference point: eigenvector of the smallest eigenvalue of its k-neighbourhood covariance.
    std::vector<Eigen::Vector2d> normal(ref.size()), centre(ref.size());   // fitted local line = (centre, normal)
    std::vector<double> var_tilt(ref.size());   // tilt variance of each fitted line
    std::vector<std::uint32_t> nbr(ref.size() * size_t(k));   // the k neighbours behind each normal
    {
        std::vector<std::uint32_t> idx(k); std::vector<double> d2(k);
        for (size_t i = 0; i < ref.size(); ++i)
        {
            tree.knnSearch(refd[i].data(), size_t(k), idx.data(), d2.data());
            std::copy(idx.begin(), idx.end(), nbr.begin() + std::ptrdiff_t(i) * k);
            Eigen::Vector2d m = Eigen::Vector2d::Zero();
            for (int a = 0; a < k; ++a) m += refd[idx[a]];
            m /= k; centre[i] = m;
            Eigen::Matrix2d S = Eigen::Matrix2d::Zero();
            for (int a = 0; a < k; ++a) { const Eigen::Vector2d e = refd[idx[a]] - m; S += e * e.transpose(); }
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(S);
            // Var(tilt) = s_i^2 / Sxx (slope of an OLS line), s_i^2 = local perpendicular noise = lambda0 / (k-2): the noise of a
            // ray scan is heteroscedastic (it shrinks at grazing incidence), so it is read off each fit, not one global sigma
            var_tilt[i] = (es.eigenvalues()(0) / double(k - 2)) / std::max(es.eigenvalues()(1), 1e-12);
            normal[i] = es.eigenvectors().col(0);                       // ascending eigenvalues: col 0 = least spread
        }
    }

    const double inv_w = 1.0 / cfg.outlier_window_m;
    // Deterministic annealing of the inlier sigma: from the half-window down to sigma_m, x0.7 per iteration. A fixed
    // 1 cm Gaussian gives zero weight to every pair of a scan displaced by tens of cm (the basin would be empty);
    // the annealed EM starts wide, the inlier Gaussian tightens as the alignment does. A schedule, not a gate.
    double sigma = std::max(cfg.sigma_m, 0.5 * cfg.outlier_window_m);
    double pi_in = cfg.pi_inlier0;
    Eigen::Vector3d x = init_xyt;
    Eigen::Matrix3d H = Eigen::Matrix3d::Identity();
    std::vector<double> w(cur.size()), r(cur.size()); std::vector<std::uint32_t> nn(cur.size());
    std::vector<Eigen::Vector3d> J(cur.size());
    bool converged = false;

    auto linearise = [&]()   // residuals, EM responsibilities, Jacobians at the current pose x
    {
        const double c = std::cos(x.z()), s = std::sin(x.z());
        double wsum = 0.0;
        for (size_t j = 0; j < cur.size(); ++j)
        {
            const Eigen::Vector2d p = cur[j].cast<double>();
            const Eigen::Vector2d pr(c * p.x() - s * p.y() + x.x(), s * p.x() + c * p.y() + x.y());
            std::uint32_t ni = 0; double nd2 = 0.0;
            tree.knnSearch(pr.data(), 1, &ni, &nd2);
            nn[j] = ni;
            const Eigen::Vector2d& n = normal[ni];
            r[j] = n.dot(pr - centre[ni]);   // distance to the neighbourhood's fitted line (not to the single noisy point q_i)
            J[j] = Eigen::Vector3d(n.x(), n.y(), n.dot(Eigen::Vector2d(-s * p.x() - c * p.y(), c * p.x() - s * p.y())));
            const double a = pi_in * std::exp(-0.5 * r[j] * r[j] / (sigma * sigma)) / (std::sqrt(2.0 * std::numbers::pi) * sigma);
            const double b = (1.0 - pi_in) * inv_w;
            w[j] = (a + b) > 0.0 ? a / (a + b) : 0.0;
            wsum += w[j];
        }
        return wsum / double(cur.size());                               // EM update of the inlier share
    };

    // Gaussian prior on the registration pose, centred on the initial guess: variance (scan extent)^2 per translation,
    // (1 rad)^2 for yaw, i.e. nothing beyond the scene scale. The iteration minimises the MAP cost (data + this prior),
    // so on an axis the scans do not observe (along a corridor) the step is held by the prior instead of sliding on a
    // flat, NN-flip-jittered cost for max_iters. In H units (H = sigma^2 x information) the prior adds sigma^2 P0^-1.
    double extent2 = 1e-12;
    for (const auto& p : cur) extent2 = std::max(extent2, double(p.squaredNorm()));
    const Eigen::Matrix3d P0inv = Eigen::Vector3d(1.0 / extent2, 1.0 / extent2, 1.0).asDiagonal().toDenseMatrix();
    for (int it = 0; it < cfg.max_iters; ++it)
    {
        out.iters = it + 1;
        pi_in = std::clamp(linearise(), 1e-6, 1.0 - 1e-6);              // keeps log-odds finite, not a gate
        H.setZero(); Eigen::Vector3d g = Eigen::Vector3d::Zero();
        for (size_t j = 0; j < cur.size(); ++j) { H += w[j] * J[j] * J[j].transpose(); g += w[j] * r[j] * J[j]; }
        const Eigen::Matrix3d Hp = H + sigma * sigma * P0inv;
        const Eigen::Vector3d dx = -Hp.ldlt().solve(g + sigma * sigma * P0inv * (x - init_xyt));
        x += dx;
        // Converged when the step is small against the posterior's own spread: dx^T (H + sigma^2 P0^-1) dx / sigma^2 is
        // the step in posterior standard deviations (squared), so metres and radians, and observed and unobserved axes,
        // are weighed consistently. 1e-2 = a tenth of a standard deviation (the NN-flip limit cycle sits well below).
        const bool annealed = sigma <= cfg.sigma_m;
        if (annealed and dx.dot(Hp * dx) / (sigma * sigma) < 1e-2) { converged = true; break; }
        sigma = std::max(cfg.sigma_m, 0.7 * sigma);
    }

    // Final statistics at the solution (sigma is at sigma_m once annealed).
    sigma = cfg.sigma_m;
    pi_in = std::clamp(linearise(), 1e-6, 1.0 - 1e-6);
    H.setZero(); double wsum = 0.0, wr2 = 0.0;
    for (size_t j = 0; j < cur.size(); ++j) { H += w[j] * J[j] * J[j].transpose(); wsum += w[j]; wr2 += w[j] * r[j] * r[j]; }
    // The normals are fitted to noisy points, so every J carries a tilt error d(J) = tau (t, t.R'p) with
    // Var(tau) = var_tilt. E[J J^T] = J0 J0^T + Var(tau) u u^T: that last term is information the noise
    // MANUFACTURES (a corridor's walls "constrain" the axis along them). Remove its expectation (errors-in-variables).
    const double sig2 = (wsum > 0.0 ? wr2 / wsum : 0.0) / (1.0 + 1.0 / k);
    Eigen::Matrix3d Hfake = Eigen::Matrix3d::Zero();
    {
        const double c = std::cos(x.z()), s = std::sin(x.z());
        for (size_t j = 0; j < cur.size(); ++j)
        {
            const Eigen::Vector2d p = cur[j].cast<double>(), n = normal[nn[j]], t(-n.y(), n.x());
            const Eigen::Vector2d dp(-s * p.x() - c * p.y(), c * p.x() - s * p.y());
            const Eigen::Vector3d u(t.x(), t.y(), t.dot(dp));
            Hfake += w[j] * var_tilt[nn[j]] * u * u.transpose();
        }
    }
    // Real or manufactured, axis by axis (eigen-axes of H): Bayesian comparison of the measured information lambda
    // against the manufactured-information distribution F +- rel (see p_real). The axis carries P(real) (lambda - F)+.
    // n_eff = independent neighbourhoods (~n/k, each tilt is shared by ~k residuals); rel^2 = 2/n_eff (chi-square).
    // mu = one wall end's information: a wall facing the axis subtends about a quarter of a 360 deg scan, each
    // point contributing ~1 (unit Jacobian along its normal).
    const double rel = std::sqrt(2.0 * k / std::max(wsum, 1.0));
    const double mu = std::max(0.25 * wsum, 1.0);
    Eigen::Matrix3d Hc;
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> e(H);
        Eigen::Vector3d lam = Eigen::Vector3d::Zero();
        for (int m = 0; m < 3; ++m)
        {
            const Eigen::Vector3d v = e.eigenvectors().col(m);
            const double l = e.eigenvalues()[m], F = v.dot(Hfake * v);
            lam[m] = p_real(l, F, rel, mu) * std::max(l - F, 0.0);
        }
        Hc = e.eigenvectors() * lam.asDiagonal() * e.eigenvectors().transpose();
    }
    // Reference-noise correlation. A reference point sits in ~k fitted lines, so it shifts all their residuals
    // together, each by 1/k: M = sum_i a_i a_i^T, a_i = sum_{j: i in N(nn_j)} w_j J_j / k. The measurement then has
    // covariance sigma^2 (H^-1 + H^-1 M H^-1) = sigma^2 H (H+M)^-1 H inverted, i.e. information H (H+M)^-1 H / sigma^2
    // (written this way it stays finite on an axis with H = 0).
    // The same pose prior as the iteration, in the final sigma's H units.
    const Eigen::Matrix3d Pinv = sig2 * P0inv;
    std::vector<Eigen::Vector3d> a(ref.size(), Eigen::Vector3d::Zero());
    for (size_t j = 0; j < cur.size(); ++j)
        for (int m = 0; m < k; ++m) a[nbr[size_t(nn[j]) * k + m]] += w[j] * J[j] / k;
    Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
    for (const auto& ai : a) M += ai * ai.transpose();
    const Eigen::Matrix3d Hhat = Hc * (Hc + M + Pinv).inverse() * Hc + Pinv;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> hes(0.5 * (Hhat + Hhat.transpose()));
    const Eigen::Matrix3d Hi = hes.eigenvectors() * hes.eigenvalues().cwiseInverse().asDiagonal() * hes.eigenvectors().transpose();
    const bool pd = hes.eigenvalues().minCoeff() > 0.0 and Hi.allFinite();
    out.xyt = x; out.inlier_share = pi_in; out.n_used = int(std::llround(wsum));
    out.rms = wsum > 0.0 ? std::sqrt(wr2 / wsum) : 0.0;
    if (wsum > 0.0 and pd) out.cov = sig2 * Hi;
    out.ok = converged and pd;   // converged + finite: consumers must still read cov (an axis may be uninformed)
    return out;
}
}  // namespace sc::calib
