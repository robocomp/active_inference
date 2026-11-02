#include "multi_scan_fit.h"
#include <algorithm>
#include <cmath>

namespace rc::rivals
{
namespace
{
float wrap(float a) { return std::remainder(a, 2.f * static_cast<float>(M_PI)); }
}

rc::reloc::Mode multi_scan_fit(const rc::reloc::LikelihoodRaster& K, const std::vector<Eigen::Vector2f>& polygon,
                               const std::vector<std::pair<Eigen::Vector3f, std::vector<Eigen::Vector2f>>>& scans,
                               const Eigen::Vector3f& prior, const Eigen::Matrix3f& prior_cov, const rc::reloc::Params& p)
{
    rc::reloc::Mode out;
    out.pose = prior; out.cov = prior_cov; out.log_lik = std::numeric_limits<float>::quiet_NaN();
    if (polygon.size() < 3 or not K.valid()) return out;
    // Every keyframe's returns in the latest keyframe's frame, each scan on the SAME subset the search polishes on.
    std::vector<Eigen::Vector2f> joint;
    for (const auto& [rel, pts] : scans)
    {
        const float c = std::cos(rel.z()), s = std::sin(rel.z());
        for (const auto& q : rc::reloc::polish_points(pts, p))
            joint.emplace_back(rel.x() + c * q.x() - s * q.y(), rel.y() + s * q.x() + c * q.y());
    }
    if (joint.empty()) return out;
    rc::reloc::Params pp = p;
    pp.prior_modes.clear();   // the prior is fused below, once, as a Gaussian factor
    auto m = rc::reloc::polish(polygon, joint, prior, pp, K.extent());
    if (not m.pose.allFinite() or not m.cov.allFinite()) return out;
    // Information-form fusion of the scans' Laplace Gaussian with the prior (angles wrapped about the prior).
    // EFFECTIVE SAMPLE SIZE. The polish treats every point of every keyframe as independent, so N keyframes of one view
    // (a parked robot) would multiply the information by N. A keyframe is worth 1 / (its soft count of keyframes with the same
    // view): w_i = 1 / sum_j exp(-d_ij' B^-1 d_ij / 2), d_ij the relative pose, B the likelihood's own basin (sigma_sdf in
    // translation, sigma_sdf / median range in rotation). A continuous kernel, not a gate: identical keyframes give
    // N_eff = 1, views a basin apart give N_eff = N. The Laplace information is scaled by N_eff / N (no sigma floor).
    float r_med = 1.f;
    {
        std::vector<float> r;
        for (const auto& q : joint) r.push_back(q.norm());
        std::ranges::nth_element(r, r.begin() + static_cast<std::ptrdiff_t>(r.size() / 2));
        r_med = std::max(r[r.size() / 2], p.sigma_sdf);
    }
    const Eigen::Vector3f binv(1.f / (p.sigma_sdf * p.sigma_sdf), 1.f / (p.sigma_sdf * p.sigma_sdf), (r_med / p.sigma_sdf) * (r_med / p.sigma_sdf));
    double n_eff = 0.0;
    for (std::size_t i = 0; i < scans.size(); ++i)
    {
        double soft = 0.0;
        for (std::size_t j = 0; j < scans.size(); ++j)
        {
            Eigen::Vector3f d = scans[i].first - scans[j].first;
            d.z() = wrap(d.z());
            soft += std::exp(-0.5 * static_cast<double>(d.cwiseAbs2().dot(binv)));
        }
        n_eff += 1.0 / soft;
    }
    const float scale = static_cast<float>(n_eff / static_cast<double>(scans.size()));
    const Eigen::Matrix3f Il = m.cov.inverse() * scale, Ip = prior_cov.inverse();
    const Eigen::Matrix3f P = (Il + Ip).inverse();
    Eigen::Vector3f dl = m.pose - prior; dl.z() = wrap(dl.z());
    Eigen::Vector3f x = prior + P * (Il * dl);
    x.z() = wrap(x.z());
    out.pose = x;
    out.cov = 0.5f * (P + P.transpose());
    out.log_lik = rc::reloc::log_likelihood(polygon, joint, x, pp, K.extent(), &out.median_abs);
    return out;
}
}
