/*
 * bottle_cylinder_contour.cpp — see bottle_cylinder_contour.h.
 */

#include "bottle_cylinder_contour.h"

#include <cmath>
#include <numbers>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace rc
{

rc::edges::ContourSet cylinder_contour_set(float cx, float cy, float z0, float z1, float radius,
                                           const Eigen::Vector3d& camera_pos_world,
                                           const rc::edges::PointProjector& project)
{
    rc::edges::ContourSet out;
    if (not project or radius <= 1e-4f or z1 <= z0)
        return out;

    // Rim samples. 24 per circle: the hull of a 24-gon is within r(1-cos 7.5deg) = 0.9% of r of the true
    // ellipse, i.e. sub-pixel for any bottle large enough to be scored at all.
    constexpr int N = 24;
    std::vector<Eigen::Vector3d> pts;
    std::vector<cv::Point2f>     px;
    pts.reserve(2 * N);
    px.reserve(2 * N);
    for (const float z : {z0, z1})
        for (int i = 0; i < N; ++i)
        {
            const double a = 2.0 * std::numbers::pi * i / N;
            const Eigen::Vector3d P(cx + radius * std::cos(a), cy + radius * std::sin(a), z);
            const auto v = project(P);
            if (not v.has_value())
                return out;               // a rim point behind the camera ⇒ nothing measured this frame
            pts.push_back(P);
            px.push_back(v->px);
        }

    // The silhouette = hull of the projected rims, chosen in the image. Indices, so the 3-D vertices can be
    // handed on in hull order.
    std::vector<int> hull;
    cv::convexHull(px, hull, /*clockwise=*/false, /*returnPoints=*/false);
    if (hull.size() < 3)
        return out;
    std::vector<Eigen::Vector3d> loop;
    loop.reserve(hull.size());
    for (const int i : hull)
        loop.push_back(pts[static_cast<std::size_t>(i)]);

    // Slide: horizontal, perpendicular to the viewing ray.
    Eigen::Vector3d ray(cx - camera_pos_world.x(), cy - camera_pos_world.y(), 0.0);
    Eigen::Vector3d u(1.0, 0.0, 0.0);
    if (ray.norm() > 1e-6)
    {
        ray.normalize();
        u = Eigen::Vector3d(-ray.y(), ray.x(), 0.0);
    }
    return rc::edges::project_loop_slid(loop, u, 2.0f * radius, project);
}

}   // namespace rc
