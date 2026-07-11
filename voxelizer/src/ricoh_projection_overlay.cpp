#include "ricoh_projection_overlay.h"

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_camera_api.h>

#include "../../common/depth_projection/depth_projection.h"   // reproject_cloud (shared, CameraAPI-based)

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <string>

namespace rc
{

namespace
{

// BGR overlay colour per object category (the ricoh popup canvas is BGR). Same palette intent as the
// ZED overlay: tables amber, bottles cyan, chairs violet, obstacles red, generic objects green.
cv::Scalar color_for_category(const std::string& category)
{
    if (category == "model_table" or category == "table")   return {0, 165, 255};    // orange
    if (category == "bottle" or category == "cylinder")      return {220, 220, 0};    // cyan
    if (category == "chair")                                 return {255, 120, 200};  // violet
    if (category == "obstacle")                              return {45, 45, 255};    // red
    if (category == "object")                                return {90, 220, 60};    // green
    return {0, 230, 255};                                                             // yellow fallback
}

// Range → BGR jet ramp (near = red, mid = green, far = blue) for the sparse lidar-depth overlay.
cv::Scalar color_by_range(float range, float rmin = 0.4f, float rmax = 8.0f)
{
    const float t = std::clamp((range - rmin) / (rmax - rmin), 0.f, 1.f);   // 0 near … 1 far
    return { 255.f * t, 255.f * (1.f - std::abs(2.f * t - 1.f)), 255.f * (1.f - t) };   // (B,G,R)
}

}  // namespace

RicohProjectionOverlay::RicohProjectionOverlay(std::shared_ptr<DSRGraph> graph, std::string ricoh_node_name)
    : graph_(std::move(graph)), ricoh_node_name_(std::move(ricoh_node_name))
{
}

RicohProjectionOverlay::~RicohProjectionOverlay() = default;

bool RicohProjectionOverlay::ensure_camera_api()
{
    if (camera_api_)
        return true;
    if (not graph_)
        return false;
    const auto ricoh_node = graph_->get_node(ricoh_node_name_);   // one-time; ricoh is a static node
    if (not ricoh_node.has_value())
        return false;
    camera_api_ = graph_->get_camera_api(ricoh_node.value());   // cam_fov≈2π ⇒ equirectangular model
    return static_cast<bool>(camera_api_);
}

void RicohProjectionOverlay::draw(cv::Mat& pano_bgr, std::span<const GraphObjectBox> boxes,
                                  const Mat::RTMat& room_T_ricoh,
                                  std::span<const float> room_poly_x, std::span<const float> room_poly_y,
                                  float room_height)
{
    if (pano_bgr.empty() or not ensure_camera_api())
        return;
    DSR::CameraAPI& cam = *camera_api_;

    const Mat::RTMat ricoh_T_room = room_T_ricoh.inverse();   // room point → ricoh frame for CameraAPI
    const double half_w = pano_bgr.cols * 0.5;
    const int max_x = pano_bgr.cols - 1, max_y = pano_bgr.rows - 1;

    // Project a room-frame point → panorama pixel via the shared CameraAPI equirectangular model.
    const auto project = [&](const Eigen::Vector3f& p_room) -> cv::Point2d
    {
        const Eigen::Vector2d uv = cam.project(ricoh_T_room * p_room.cast<double>());
        return {uv.x(), uv.y()};
    };

    // Draw one 3D segment as a subdivided, seam-split polyline. Straight 3D lines curve in
    // equirectangular, and a sub-segment crossing the wrap-seam (|Δcol| > W/2) must NOT be joined
    // across the image. cv::line clips to the canvas internally; coords are clamped for good measure.
    const auto to_pt = [&](const cv::Point2d& q)
    {
        return cv::Point(std::clamp(static_cast<int>(std::lround(q.x)), 0, max_x),
                         std::clamp(static_cast<int>(std::lround(q.y)), 0, max_y));
    };
    const auto draw_edge = [&](const Eigen::Vector3f& a, const Eigen::Vector3f& b, const cv::Scalar& color, int subdiv)
    {
        std::optional<cv::Point2d> prev;
        for (int k = 0; k <= subdiv; ++k)
        {
            const float t = static_cast<float>(k) / static_cast<float>(subdiv);
            const cv::Point2d cur = project(a + t * (b - a));
            if (prev.has_value() and std::isfinite(cur.x) and std::isfinite(cur.y)
                and std::abs(cur.x - prev->x) < half_w)   // same side of the seam
                cv::line(pano_bgr, to_pt(*prev), to_pt(cur), color, 2, cv::LINE_AA);
            prev = cur;
        }
    };

    // ── Room floor + ceiling + wall wireframes (light) ──────────────────────────────────────────
    const std::size_t n = std::min(room_poly_x.size(), room_poly_y.size());
    if (n >= 3)
    {
        const cv::Scalar floor_col(255, 180, 90);    // bluish (BGR)
        const cv::Scalar ceil_col(90, 90, 235);      // reddish (BGR)
        const cv::Scalar wall_col(200, 120, 150);    // purple (BGR)
        for (std::size_t i = 0; i < n; ++i)
        {
            const std::size_t j = (i + 1) % n;
            const Eigen::Vector3f fi(room_poly_x[i], room_poly_y[i], 0.f);
            const Eigen::Vector3f fj(room_poly_x[j], room_poly_y[j], 0.f);
            const Eigen::Vector3f ci(room_poly_x[i], room_poly_y[i], room_height);
            const Eigen::Vector3f cj(room_poly_x[j], room_poly_y[j], room_height);
            draw_edge(fi, fj, floor_col, 40);   // floor edge
            draw_edge(ci, cj, ceil_col,  40);   // ceiling edge
            draw_edge(fi, ci, wall_col,  24);   // vertical wall edge at corner i
        }
    }

    // ── Model-instance boxes (12-edge wireframe + name label) ───────────────────────────────────
    static constexpr int box_edges[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0},   // bottom
        {4, 5}, {5, 6}, {6, 7}, {7, 4},   // top
        {0, 4}, {1, 5}, {2, 6}, {3, 7}    // verticals
    };
    for (const auto& box : boxes)
    {
        const float c = std::cos(box.yaw_rad), s = std::sin(box.yaw_rad);
        const float hw = box.half_extents.x(), hd = box.half_extents.y(), hh = box.half_extents.z();
        const std::array<Eigen::Vector3f, 8> lo = {
            Eigen::Vector3f{-hw, -hd, -hh}, Eigen::Vector3f{ hw, -hd, -hh},
            Eigen::Vector3f{ hw,  hd, -hh}, Eigen::Vector3f{-hw,  hd, -hh},
            Eigen::Vector3f{-hw, -hd,  hh}, Eigen::Vector3f{ hw, -hd,  hh},
            Eigen::Vector3f{ hw,  hd,  hh}, Eigen::Vector3f{-hw,  hd,  hh}
        };
        std::array<Eigen::Vector3f, 8> corners;
        for (std::size_t i = 0; i < lo.size(); ++i)
            corners[i] = Eigen::Vector3f(box.center.x() + c * lo[i].x() - s * lo[i].y(),
                                         box.center.y() + s * lo[i].x() + c * lo[i].y(),
                                         box.center.z() + lo[i].z());

        const cv::Scalar col = color_for_category(box.category);
        for (const auto& e : box_edges)
            draw_edge(corners[e[0]], corners[e[1]], col, 12);

        // Name label at the projected top-front corner (corner 4), if it lands inside the image.
        if (not box.node_name.empty())
        {
            const cv::Point2d uv = project(corners[4]);
            if (std::isfinite(uv.x) and std::isfinite(uv.y)
                and uv.x >= 0 and uv.x <= max_x and uv.y >= 0 and uv.y <= max_y)
                cv::putText(pano_bgr, box.node_name,
                            cv::Point(static_cast<int>(std::lround(uv.x)) + 3, static_cast<int>(std::lround(uv.y)) - 3),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv::LINE_AA);
        }
    }
}

void RicohProjectionOverlay::draw_lidar_points(cv::Mat& pano_bgr, std::span<const Eigen::Vector3f> cloud_room,
                                               const Mat::RTMat& room_T_ricoh)
{
    if (pano_bgr.empty() or cloud_room.empty() or not ensure_camera_api())
        return;

    // Reproject the room-frame cloud into the ricoh via the shared helper (camera_T_room = inverse).
    const auto pts = rc::depth::reproject_cloud(cloud_room, *camera_api_, room_T_ricoh.inverse());
    const int max_x = pano_bgr.cols - 1, max_y = pano_bgr.rows - 1;
    for (const auto& p : pts)
    {
        const int u = static_cast<int>(std::lround(p.u));
        const int v = static_cast<int>(std::lround(p.v));
        if (u < 0 or u > max_x or v < 0 or v > max_y)
            continue;
        cv::circle(pano_bgr, {u, v}, 1, color_by_range(p.range), cv::FILLED, cv::LINE_AA);
    }
}

}  // namespace rc
