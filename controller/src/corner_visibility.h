#pragma once

#include <vector>
#include <cmath>
#include <Eigen/Dense>

namespace rc {
namespace corner_visibility {

inline bool point_in_polygon(const Eigen::Vector2f& pt,
                             const std::vector<Eigen::Vector2f>& poly)
{
    const int n = static_cast<int>(poly.size());
    bool inside = false;
    for (int i = 0, j = n - 1; i < n; j = i++)
    {
        if (((poly[i].y() > pt.y()) != (poly[j].y() > pt.y())) &&
            (pt.x() < (poly[j].x() - poly[i].x()) * (pt.y() - poly[i].y())
                        / (poly[j].y() - poly[i].y()) + poly[i].x()))
            inside = !inside;
    }
    return inside;
}

inline bool segments_cross(const Eigen::Vector2f& p1, const Eigen::Vector2f& p2,
                           const Eigen::Vector2f& p3, const Eigen::Vector2f& p4)
{
    const Eigen::Vector2f d1 = p2 - p1;
    const Eigen::Vector2f d2 = p4 - p3;
    const float denom = d1.x() * d2.y() - d1.y() * d2.x();
    if (std::abs(denom) < 1e-10f) return false;

    const Eigen::Vector2f d3 = p3 - p1;
    const float t = (d3.x() * d2.y() - d3.y() * d2.x()) / denom;
    const float u = (d3.x() * d1.y() - d3.y() * d1.x()) / denom;
    constexpr float eps = 1e-4f;
    return t > eps && t < (1.f - eps) && u > eps && u < (1.f - eps);
}

inline bool is_corner_visible(const Eigen::Vector2f& robot_pos,
                              int corner_idx,
                              const std::vector<Eigen::Vector2f>& polygon,
                              float max_range = 15.f)
{
    const int count = static_cast<int>(polygon.size());
    if (corner_idx < 0 || corner_idx >= count) return false;

    const Eigen::Vector2f& corner = polygon[corner_idx];
    if ((corner - robot_pos).squaredNorm() > max_range * max_range)
        return false;

    const int seg_before = (corner_idx - 1 + count) % count;
    const int seg_after  = corner_idx;

    for (int index = 0; index < count; ++index)
    {
        if (index == seg_before || index == seg_after) continue;
        if (segments_cross(robot_pos, corner, polygon[index], polygon[(index + 1) % count]))
            return false;
    }
    return true;
}

inline std::vector<int> visible_corners(const Eigen::Vector2f& robot_pos,
                                        const std::vector<Eigen::Vector2f>& polygon,
                                        float max_range = 15.f)
{
    std::vector<int> result;
    for (int index = 0; index < static_cast<int>(polygon.size()); ++index)
        if (is_corner_visible(robot_pos, index, polygon, max_range))
            result.push_back(index);
    return result;
}

inline Eigen::Matrix<float, 2, 3> corner_observation_jacobian(
    const Eigen::Vector2f& robot_pos, float robot_theta,
    const Eigen::Vector2f& corner_world)
{
    const float ct = std::cos(robot_theta);
    const float st = std::sin(robot_theta);
    const float dx = corner_world.x() - robot_pos.x();
    const float dy = corner_world.y() - robot_pos.y();
    const float z0 =  ct * dx + st * dy;
    const float z1 = -st * dx + ct * dy;

    Eigen::Matrix<float, 2, 3> jacobian;
    jacobian << -ct, -st,  z1,
                 st, -ct, -z0;
    return jacobian;
}

inline Eigen::Matrix3f corner_fim(const Eigen::Vector2f& robot_pos,
                                  float robot_theta,
                                  const std::vector<int>& vis_indices,
                                  const std::vector<Eigen::Vector2f>& polygon,
                                  float corner_sigma = 0.04f)
{
    Eigen::Matrix3f fim = Eigen::Matrix3f::Zero();
    const float inv_var = 1.f / (corner_sigma * corner_sigma);
    for (int idx : vis_indices)
    {
        const auto jacobian = corner_observation_jacobian(robot_pos, robot_theta, polygon[idx]);
        fim.noalias() += inv_var * (jacobian.transpose() * jacobian);
    }
    return fim;
}

inline float d_optimality_gain(const Eigen::Matrix3f& prior_precision,
                               const Eigen::Matrix3f& corner_fim_mat)
{
    const float ld_prior = std::log(std::max(prior_precision.determinant(), 1e-30f));
    const float ld_post  = std::log(std::max((prior_precision + corner_fim_mat).determinant(), 1e-30f));
    return ld_post - ld_prior;
}

} // namespace corner_visibility
} // namespace rc