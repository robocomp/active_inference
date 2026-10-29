#include "scan_registration.h"

#include <nanoflann.hpp>

#include <algorithm>
#include <cmath>
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

    for (int it = 0; it < cfg.max_iters; ++it)
    {
        out.iters = it + 1;
        pi_in = std::clamp(linearise(), 1e-6, 1.0 - 1e-6);              // keeps log-odds finite, not a gate
        H.setZero(); Eigen::Vector3d g = Eigen::Vector3d::Zero();
        for (size_t j = 0; j < cur.size(); ++j) { H += w[j] * J[j] * J[j].transpose(); g += w[j] * r[j] * J[j]; }
        const Eigen::Matrix3d Hs = H + 1e-12 * H.trace() * Eigen::Matrix3d::Identity();   // conditioning only
        const Eigen::Vector3d dx = -Hs.ldlt().solve(g);
        x += dx;
        const bool annealed = sigma <= cfg.sigma_m;
        if (annealed and dx.norm() < 1e-4) { converged = true; break; }
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
    // Information after removing the expected manufactured part (projected to PSD: numerical housekeeping). The
    // subtracted part is itself random (relative sd sqrt(2k/n): ~n/k independent tilts each shared by ~k residuals),
    // so each axis' remaining information lambda is shrunk by its reliability lambda^2 / (lambda^2 + s^2),
    // s = rel * (manufactured information on that axis): continuous, -> lambda when it dominates its noise, -> 0 when
    // it is within the noise (the posterior-mean weight of a signal of variance lambda^2 in noise s^2).
    const double rel = std::sqrt(2.0 * k / std::max(wsum, 1.0));
    Eigen::Matrix3d Hc;
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> e(H - Hfake);
        Eigen::Vector3d lam = e.eigenvalues().cwiseMax(0.0);
        for (int m = 0; m < 3; ++m)
        {
            const Eigen::Vector3d v = e.eigenvectors().col(m);
            const double sm = rel * v.dot(Hfake * v);
            lam[m] *= lam[m] * lam[m] / std::max(lam[m] * lam[m] + sm * sm, 1e-300);
        }
        Hc = e.eigenvectors() * lam.asDiagonal() * e.eigenvectors().transpose();
    }
    // Reference-noise correlation. A reference point sits in ~k fitted lines, so it shifts all their residuals
    // together, each by 1/k: M = sum_i a_i a_i^T, a_i = sum_{j: i in N(nn_j)} w_j J_j / k. The measurement then has
    // covariance sigma^2 (H^-1 + H^-1 M H^-1) = sigma^2 H (H+M)^-1 H inverted, i.e. information H (H+M)^-1 H / sigma^2
    // (written this way it stays finite on an axis with H = 0).
    // Gaussian prior on the registration pose: variance (scan extent)^2 per translation, (1 rad)^2 for yaw, i.e. we
    // know nothing beyond the scene scale. In H units (H = sigma^2 x information) the prior adds sigma^2 P0^-1.
    double extent2 = 1e-12;
    for (const auto& p : cur) extent2 = std::max(extent2, double(p.squaredNorm()));
    const Eigen::Matrix3d Pinv = sig2 * Eigen::Vector3d(1.0 / extent2, 1.0 / extent2, 1.0).asDiagonal().toDenseMatrix();
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
