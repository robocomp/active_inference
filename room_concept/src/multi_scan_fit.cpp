#include "multi_scan_fit.h"
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
    const Eigen::Matrix3f Il = m.cov.inverse(), Ip = prior_cov.inverse();
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
