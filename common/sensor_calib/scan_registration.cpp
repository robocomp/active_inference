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
    const int k = std::max(3, cfg.normal_k);
    if (int(ref.size()) < k or cur.size() < 3) return out;

    // double copy: the pose steps converge to ~1e-7 m and a float query would quantise them
    const std::vector<Eigen::Vector2d> refd = [&] { std::vector<Eigen::Vector2d> v; v.reserve(ref.size()); for (const auto& p : ref) v.push_back(p.cast<double>()); return v; }();
    const Cloud cloud{refd};
    Tree tree(2, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    // PCA normal of each reference point: eigenvector of the smallest eigenvalue of its k-neighbourhood covariance.
    std::vector<Eigen::Vector2d> normal(ref.size());
    std::vector<std::uint32_t> nbr(ref.size() * size_t(k));   // the k neighbours behind each normal
    {
        std::vector<std::uint32_t> idx(k); std::vector<double> d2(k);
        for (size_t i = 0; i < ref.size(); ++i)
        {
            tree.knnSearch(refd[i].data(), size_t(k), idx.data(), d2.data());
            std::copy(idx.begin(), idx.end(), nbr.begin() + std::ptrdiff_t(i) * k);
            Eigen::Vector2d m = Eigen::Vector2d::Zero();
            for (int a = 0; a < k; ++a) m += refd[idx[a]];
            m /= k;
            Eigen::Matrix2d S = Eigen::Matrix2d::Zero();
            for (int a = 0; a < k; ++a) { const Eigen::Vector2d e = refd[idx[a]] - m; S += e * e.transpose(); }
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(S);
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
            r[j] = n.dot(pr - refd[ni]);
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
    const Eigen::Matrix3d Hs = H + 1e-12 * H.trace() * Eigen::Matrix3d::Identity();
    Eigen::LDLT<Eigen::Matrix3d> ldlt(Hs);
    const bool pd = ldlt.info() == Eigen::Success and ldlt.isPositive() and ldlt.vectorD().minCoeff() > 0.0;
    out.xyt = x; out.inlier_share = pi_in; out.n_used = int(std::llround(wsum));
    out.rms = wsum > 0.0 ? std::sqrt(wr2 / wsum) : 0.0;
    if (wsum > 0.0 and pd)
    {
        // Covariance = propagation of BOTH scans' range noise through the estimator. Cur point noise is independent
        // per residual: sigma^2 H^-1. A ref point's noise enters every normal that fitted it (k of them, 1/k each) and
        // so is COMMON to ~k cur residuals: its pose-error contribution is H^-1 (sum_i a_i a_i^T) H^-1 with
        // a_i = sum_{j: i in N(nn_j)} w_j r-Jacobian_j / k. (Treating the ref noise as already inside the residual
        // variance under-reports the yaw sd by ~1.6x, measured: ref-only noise gave 0.45x, cur-only 0.85x.)
        // Both scans come from the same sensor, so one sigma: residual var = sigma^2 (1 + 1/k) -> sigma^2.
        std::vector<Eigen::Vector3d> a(ref.size(), Eigen::Vector3d::Zero());
        for (size_t j = 0; j < cur.size(); ++j)
            for (int m = 0; m < k; ++m) a[nbr[size_t(nn[j]) * k + m]] += w[j] * J[j] / k;
        Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
        for (const auto& ai : a) M += ai * ai.transpose();
        const Eigen::Matrix3d Hi = ldlt.solve(Eigen::Matrix3d::Identity());
        const double sig2 = (wr2 / wsum) / (1.0 + 1.0 / k);
        out.cov = sig2 * (Hi + Hi * M * Hi);
    }
    out.ok = converged and pd;
    return out;
}
}  // namespace sc::calib
