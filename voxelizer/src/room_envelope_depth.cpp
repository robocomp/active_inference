#include "room_envelope_depth.h"

#include <opencv2/imgproc.hpp>
#include <dsr/api/dsr_camera_api.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace rc::depth
{

cv::Mat room_envelope_depth(int width, int height, float fx, float fy,
                            const Mat::RTMat& room_T_zed,
                            std::span<const float> poly_x, std::span<const float> poly_y,
                            float room_height, int decimate)
{
    if (width <= 0 or height <= 0 or fx <= 0.f or fy <= 0.f
        or poly_x.size() < 3 or poly_x.size() != poly_y.size() or room_height <= 0.f)
        return {};

    const int step = std::clamp(decimate, 1, 8);
    const int W = (width  + step - 1) / step;
    const int H = (height + step - 1) / step;
    const float cx = static_cast<float>(width)  * 0.5f;
    const float cy = static_cast<float>(height) * 0.5f;

    const Eigen::Matrix3d R = room_T_zed.linear();
    const Eigen::Vector3d o = room_T_zed.translation();
    const std::size_t nseg = poly_x.size();

    cv::Mat small_(H, W, CV_32FC1, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));

    for (int sy = 0; sy < H; ++sy)
    {
        auto* row = small_.ptr<float>(sy);
        const double v = static_cast<double>(sy * step);
        for (int sx = 0; sx < W; ++sx)
        {
            const double u = static_cast<double>(sx * step);
            // Ray in the ZED frame: x right, y DEPTH, z up. The y-component is 1 by construction, so
            // the ray parameter t is the depth the camera would report — see the header.
            const Eigen::Vector3d d_cam((u - cx) / fx, 1.0, (cy - v) / fy);
            const Eigen::Vector3d D = R * d_cam;

            double best = std::numeric_limits<double>::infinity();

            // Floor (z = 0) and ceiling (z = room_height): one division each.
            if (std::abs(D.z()) > 1e-9)
            {
                const double t_floor = (0.0            - o.z()) / D.z();
                const double t_ceil  = (room_height    - o.z()) / D.z();
                if (t_floor > 1e-3) best = std::min(best, t_floor);
                if (t_ceil  > 1e-3) best = std::min(best, t_ceil);
            }

            // Walls: one vertical plane per polygon edge, bounded by the edge's extent and by
            // z in [0, room_height]. Solving in 2D and checking the height afterwards is cheaper
            // than a full 3D polygon clip and is exact for a vertical extrusion.
            for (std::size_t i = 0; i < nseg; ++i)
            {
                const std::size_t j = (i + 1) % nseg;
                const double ax = poly_x[i], ay = poly_y[i];
                const double ex = poly_x[j] - ax, ey = poly_y[j] - ay;
                // Solve o.xy + t·D.xy = a + s·e  for t and s.
                const double den = D.x() * ey - D.y() * ex;
                if (std::abs(den) < 1e-12)
                    continue;                                   // ray parallel to this wall
                const double rx = ax - o.x(), ry = ay - o.y();
                const double t = (rx * ey - ry * ex) / den;
                if (t <= 1e-3 or t >= best)
                    continue;                                   // behind us, or already beaten
                const double s = (rx * D.y() - ry * D.x()) / den;
                if (s < 0.0 or s > 1.0)
                    continue;                                   // crosses the line, misses the segment
                const double z = o.z() + t * D.z();
                if (z < 0.0 or z > room_height)
                    continue;                                   // passes over or under the wall
                best = t;
            }

            if (std::isfinite(best))
                row[sx] = static_cast<float>(best);
        }
    }

    if (step == 1)
        return small_;
    cv::Mat out;
    // NEAREST on purpose: the envelope has genuine depth DISCONTINUITIES at wall corners and at the
    // floor/wall join, and interpolating across one would invent a surface that is not there — which
    // in a residual image reads as model error exactly where the model is in fact correct.
    cv::resize(small_, out, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
    return out;
}

namespace
{
// Nearest positive intersection of the ray o + t·D with the room envelope. `D` need not be unit —
// t is returned in units of |D|, so pass a UNIT direction when you want a true range.
double envelope_hit(const Eigen::Vector3d& o, const Eigen::Vector3d& D,
                    std::span<const float> poly_x, std::span<const float> poly_y, double room_height)
{
    double best = std::numeric_limits<double>::infinity();
    if (std::abs(D.z()) > 1e-9)
    {
        const double tf = (0.0         - o.z()) / D.z();
        const double tc = (room_height - o.z()) / D.z();
        if (tf > 1e-3) best = std::min(best, tf);
        if (tc > 1e-3) best = std::min(best, tc);
    }
    const std::size_t n = poly_x.size();
    for (std::size_t i = 0; i < n; ++i)
    {
        const std::size_t j = (i + 1) % n;
        const double ax = poly_x[i], ay = poly_y[i];
        const double ex = poly_x[j] - ax, ey = poly_y[j] - ay;
        const double den = D.x() * ey - D.y() * ex;
        if (std::abs(den) < 1e-12) continue;
        const double rx = ax - o.x(), ry = ay - o.y();
        const double t = (rx * ey - ry * ex) / den;
        if (t <= 1e-3 or t >= best) continue;
        const double s = (rx * D.y() - ry * D.x()) / den;
        if (s < 0.0 or s > 1.0) continue;
        const double z = o.z() + t * D.z();
        if (z < 0.0 or z > room_height) continue;
        best = t;
    }
    return best;
}
}   // namespace

cv::Mat room_envelope_range_equirect(DSR::CameraAPI& cam, int width, int height,
                                     const Mat::RTMat& room_T_cam,
                                     std::span<const float> poly_x, std::span<const float> poly_y,
                                     float room_height, int decimate)
{
    if (width <= 0 or height <= 0 or poly_x.size() < 3 or poly_x.size() != poly_y.size()
        or room_height <= 0.f)
        return {};

    const int step = std::clamp(decimate, 1, 8);
    const int W = (width  + step - 1) / step;
    const int H = (height + step - 1) / step;
    const Eigen::Matrix3d R = room_T_cam.linear();
    const Eigen::Vector3d o = room_T_cam.translation();

    cv::Mat small_(H, W, CV_32FC1, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    for (int sy = 0; sy < H; ++sy)
    {
        auto* row = small_.ptr<float>(sy);
        for (int sx = 0; sx < W; ++sx)
        {
            // Unit camera-frame ray, from the graph's own model (equirect / cylindrical / pinhole).
            const Eigen::Vector3d d_cam = cam.ray_from_pixel(static_cast<double>(sx * step),
                                                             static_cast<double>(sy * step));
            const double t = envelope_hit(o, R * d_cam, poly_x, poly_y, room_height);
            if (std::isfinite(t))
                row[sx] = static_cast<float>(t);
        }
    }
    if (step == 1)
        return small_;
    cv::Mat out;
    cv::resize(small_, out, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
    return out;
}

}   // namespace rc::depth
