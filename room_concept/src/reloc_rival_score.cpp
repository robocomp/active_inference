#include "reloc_rival_score.h"

namespace rc::rivals
{
float score(const rc::reloc::LikelihoodRaster& raster, const std::vector<Eigen::Vector2f>& polygon,
            const std::vector<Eigen::Vector2f>& pts_robot, const Eigen::Vector3f& pose, const rc::reloc::Params& p)
{
    const auto pts = rc::reloc::polish_points(pts_robot, p);
    const auto bl = rc::reloc::beyond_lines(pts, p);
    return rc::reloc::log_likelihood(polygon, pts, pose, p, raster.extent(), nullptr, &bl);
}

std::vector<float> dll(const rc::reloc::LikelihoodRaster& raster, const std::vector<Eigen::Vector2f>& polygon,
                       const std::vector<Eigen::Vector2f>& pts_robot, const Eigen::Vector3f& incumbent,
                       const std::vector<Eigen::Vector3f>& rivals, const rc::reloc::Params& p)
{
    const auto pts = rc::reloc::polish_points(pts_robot, p);
    const auto bl = rc::reloc::beyond_lines(pts, p);
    const float l0 = rc::reloc::log_likelihood(polygon, pts, incumbent, p, raster.extent(), nullptr, &bl);
    std::vector<float> out;
    out.reserve(rivals.size());
    for (const auto& r : rivals)
        out.push_back(rc::reloc::log_likelihood(polygon, pts, r, p, raster.extent(), nullptr, &bl) - l0);
    return out;
}
}
