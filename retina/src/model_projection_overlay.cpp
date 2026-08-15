#include "model_projection_overlay.h"

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_camera_api.h>

#include <opencv2/imgproc.hpp>

#include <array>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

namespace rc
{

namespace
{

// Room→camera basis: the camera-frame images of the room-frame origin and unit axes. A room point p
// maps to camera frame as origin + p.x·axis_x + p.y·axis_y + p.z·axis_z. Built from camera_T_room
// (= room_T_zed.inverse()), so NO graph traversal is needed.
struct RoomToCameraBasis
{
    Mat::Vector3d origin{0.0, 0.0, 0.0};
    Mat::Vector3d axis_x{0.0, 0.0, 0.0};
    Mat::Vector3d axis_y{0.0, 0.0, 0.0};
    Mat::Vector3d axis_z{0.0, 0.0, 0.0};

    Mat::Vector3d to_camera(double x, double y, double z) const
    {
        return origin + x * axis_x + y * axis_y + z * axis_z;
    }
    Mat::Vector3d to_camera(const Eigen::Vector3f& p) const
    {
        return to_camera(static_cast<double>(p.x()), static_cast<double>(p.y()), static_cast<double>(p.z()));
    }
};

// RGB overlay colour per object category (the YoloViewer canvas is RGB / Format_RGB888, NOT BGR —
// cv::Scalar is (R,G,B) here). tables amber, bottles cyan, chairs violet, obstacles red, objects green.
cv::Scalar color_for_category(const std::string& category)
{
    if (category == "model_table" or category == "table")   return {255, 165, 0};    // orange
    if (category == "bottle" or category == "cylinder")      return {0, 220, 220};    // cyan
    if (category == "chair")                                 return {200, 120, 255};  // violet
    if (category == "obstacle")                              return {255, 45, 45};    // red (matches 3D viewer residual_concept obstacles)
    if (category == "object")                                return {60, 220, 90};    // green
    return {255, 230, 0};                                                             // yellow fallback
}

constexpr double kNearY = 1e-4;   // camera depth-axis (Y) near plane

// Clip a camera-frame polygon against the near plane (keep camera-Y >= near) — Sutherland-Hodgman —
// so surfaces partly behind the camera still fill correctly. Returns the clipped ring (may be empty).
std::vector<Mat::Vector3d> clip_polygon_near(const std::vector<Mat::Vector3d>& cam)
{
    std::vector<Mat::Vector3d> out;
    const std::size_t n = cam.size();
    for (std::size_t i = 0; i < n; ++i)
    {
        const Mat::Vector3d& curr = cam[i];
        const Mat::Vector3d& next = cam[(i + 1) % n];
        const bool in_curr = curr.y() >= kNearY;
        const bool in_next = next.y() >= kNearY;
        if (in_curr)
            out.push_back(curr);
        if (in_curr != in_next)
        {
            const double t = (kNearY - curr.y()) / (next.y() - curr.y());
            out.push_back(curr + t * (next - curr));
        }
    }
    return out;
}

// Clip a 2D polygon against one half-plane (Sutherland-Hodgman), double precision.
template <class Inside, class Isect>
std::vector<Eigen::Vector2d> clip_half_plane(const std::vector<Eigen::Vector2d>& in, Inside inside, Isect isect)
{
    std::vector<Eigen::Vector2d> out;
    const std::size_t n = in.size();
    for (std::size_t i = 0; i < n; ++i)
    {
        const Eigen::Vector2d& cur = in[i];
        const Eigen::Vector2d& nxt = in[(i + 1) % n];
        const bool ci = inside(cur), ni = inside(nxt);
        if (ci) out.push_back(cur);
        if (ci != ni) out.push_back(isect(cur, nxt));
    }
    return out;
}

// Clip a 2D polygon to [xmin,xmax]×[ymin,ymax]. CRITICAL before cv::fillPoly: a room-spanning face
// near-plane-clipped in 3D projects to billion-pixel coords, and fillPoly (no internal clipping)
// overflows its scanline fill. Clipping bounds every coordinate.
std::vector<Eigen::Vector2d> clip_poly_to_rect(std::vector<Eigen::Vector2d> poly,
                                               double xmin, double ymin, double xmax, double ymax)
{
    poly = clip_half_plane(poly, [&](const Eigen::Vector2d& p){ return p.x() >= xmin; },
        [&](const Eigen::Vector2d& a, const Eigen::Vector2d& b){ const double t=(xmin-a.x())/(b.x()-a.x()); return Eigen::Vector2d(xmin, a.y()+t*(b.y()-a.y())); });
    if (poly.empty()) return poly;
    poly = clip_half_plane(poly, [&](const Eigen::Vector2d& p){ return p.x() <= xmax; },
        [&](const Eigen::Vector2d& a, const Eigen::Vector2d& b){ const double t=(xmax-a.x())/(b.x()-a.x()); return Eigen::Vector2d(xmax, a.y()+t*(b.y()-a.y())); });
    if (poly.empty()) return poly;
    poly = clip_half_plane(poly, [&](const Eigen::Vector2d& p){ return p.y() >= ymin; },
        [&](const Eigen::Vector2d& a, const Eigen::Vector2d& b){ const double t=(ymin-a.y())/(b.y()-a.y()); return Eigen::Vector2d(a.x()+t*(b.x()-a.x()), ymin); });
    if (poly.empty()) return poly;
    poly = clip_half_plane(poly, [&](const Eigen::Vector2d& p){ return p.y() <= ymax; },
        [&](const Eigen::Vector2d& a, const Eigen::Vector2d& b){ const double t=(ymax-a.y())/(b.y()-a.y()); return Eigen::Vector2d(a.x()+t*(b.x()-a.x()), ymax); });
    return poly;
}

// Project a near-clipped camera-frame ring to image pixels, clipped to the image rectangle so every
// coordinate is in-bounds (fillPoly-safe). nullopt if <3 vertices survive or any projection is bad.
std::optional<std::vector<cv::Point>> project_ring(DSR::CameraAPI& cam_api,
                                                   const std::vector<Mat::Vector3d>& clipped,
                                                   int cols, int rows)
{
    if (clipped.size() < 3)
        return std::nullopt;
    std::vector<Eigen::Vector2d> uv;
    uv.reserve(clipped.size());
    for (const auto& p : clipped)
    {
        const Eigen::Vector2d q = cam_api.project(p);
        if (not std::isfinite(q.x()) or not std::isfinite(q.y()))
            return std::nullopt;
        uv.push_back(q);
    }
    const auto clipped2d = clip_poly_to_rect(std::move(uv), 0.0, 0.0,
                                             static_cast<double>(cols - 1), static_cast<double>(rows - 1));
    if (clipped2d.size() < 3)
        return std::nullopt;
    std::vector<cv::Point> poly;
    poly.reserve(clipped2d.size());
    for (const auto& q : clipped2d)
        poly.emplace_back(static_cast<int>(std::lround(q.x())), static_cast<int>(std::lround(q.y())));
    return poly;
}

// A translucent mesh face queued for a single blended draw pass.
struct MeshFace
{
    std::vector<cv::Point> poly;
    cv::Scalar             fill;
    cv::Scalar             edge;
    std::string            label;   // drawn at the projected centroid (empty = none)
};

}  // namespace

ModelProjectionOverlay::ModelProjectionOverlay(std::shared_ptr<DSRGraph> graph, std::string camera_node_name)
    : graph_(std::move(graph)), camera_node_name_(std::move(camera_node_name))
{
}

ModelProjectionOverlay::~ModelProjectionOverlay() = default;

bool ModelProjectionOverlay::ensure_camera_api()
{
    if (camera_api_)
        return true;
    if (not graph_)
        return false;
    const auto zed_node = graph_->get_node(camera_node_name_);   // one-time; zed is a static node
    if (not zed_node.has_value())
        return false;
    camera_api_ = graph_->get_camera_api(zed_node.value());
    return static_cast<bool>(camera_api_);
}

void ModelProjectionOverlay::draw(cv::Mat& canvas, std::span<const GraphObjectBox> boxes,
                                  const Mat::RTMat& room_T_zed,
                                  std::span<const float> room_poly_x, std::span<const float> room_poly_y,
                                  float room_height)
{
    if (canvas.empty() or not ensure_camera_api())
        return;
    DSR::CameraAPI& cam = *camera_api_;

    // Room→camera basis from the already-computed room←zed transform (NO graph traversal): a room
    // point maps to the camera frame via camera_T_room = room_T_zed.inverse().
    const Mat::RTMat camera_T_room = room_T_zed.inverse();
    RoomToCameraBasis basis;
    basis.origin = camera_T_room.translation();
    basis.axis_x = camera_T_room.linear().col(0);
    basis.axis_y = camera_T_room.linear().col(1);
    basis.axis_z = camera_T_room.linear().col(2);

    // ── 1) Light translucent meshes: room floor + ceiling + one wall per polygon edge ───────────
    std::vector<MeshFace> faces;
    const std::size_t n = std::min(room_poly_x.size(), room_poly_y.size());
    if (n >= 3)
    {
        // Floor (z=0) and ceiling (z=room_height): the whole room polygon as a filled ring.
        const auto add_plane = [&](float z, const cv::Scalar& fill, const cv::Scalar& edge, std::string label)
        {
            std::vector<Mat::Vector3d> cam3;
            cam3.reserve(n);
            for (std::size_t i = 0; i < n; ++i)
                cam3.push_back(basis.to_camera(static_cast<double>(room_poly_x[i]),
                                               static_cast<double>(room_poly_y[i]), static_cast<double>(z)));
            if (auto poly = project_ring(cam, clip_polygon_near(cam3), canvas.cols, canvas.rows); poly.has_value())
                faces.push_back(MeshFace{std::move(*poly), fill, edge, std::move(label)});
        };
        add_plane(0.f,          {80, 150, 255},  {110, 180, 255}, "floor");     // bluish (RGB)
        add_plane(room_height,  {235, 110, 110}, {255, 140, 140}, "ceiling");   // reddish (RGB)

        // Walls: each polygon edge extruded from z=0 to z=room_height (bottom-left, bottom-right,
        // top-right, top-left). Named by index — the physical wall between corner i and i+1.
        for (std::size_t i = 0; i < n; ++i)
        {
            const std::size_t j = (i + 1) % n;
            std::vector<Mat::Vector3d> cam3 = {
                basis.to_camera(room_poly_x[i], room_poly_y[i], 0.0f),
                basis.to_camera(room_poly_x[j], room_poly_y[j], 0.0f),
                basis.to_camera(room_poly_x[j], room_poly_y[j], room_height),
                basis.to_camera(room_poly_x[i], room_poly_y[i], room_height)
            };
            if (auto poly = project_ring(cam, clip_polygon_near(cam3), canvas.cols, canvas.rows); poly.has_value())
                faces.push_back(MeshFace{std::move(*poly), {120, 90, 200}, {150, 120, 230},
                                         "wall_" + std::to_string(i)});   // light purple (RGB)
        }
    }

    if (not faces.empty())
    {
        constexpr double alpha = 0.28;
        cv::Mat overlay = canvas.clone();
        for (const auto& f : faces)
        {
            // Explicit single-contour (Point**/npts) overload. A bare std::vector<cv::Point> is read
            // as InputArrayOfArrays where total()==#points → one contour PER POINT → wild OOB corruption.
            const cv::Point* pts = f.poly.data();
            const int npts = static_cast<int>(f.poly.size());
            cv::fillPoly(overlay, &pts, &npts, 1, f.fill, cv::LINE_AA);
        }
        cv::addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0.0, canvas);

        for (const auto& f : faces)
        {
            const cv::Point* pts = f.poly.data();
            const int npts = static_cast<int>(f.poly.size());
            cv::polylines(canvas, &pts, &npts, 1, /*isClosed=*/true, f.edge, 1, cv::LINE_AA);
            if (not f.label.empty())
            {
                cv::Point centroid(0, 0);
                for (const auto& p : f.poly) centroid += p;
                centroid /= static_cast<int>(f.poly.size());
                cv::putText(canvas, f.label, centroid, cv::FONT_HERSHEY_SIMPLEX, 0.45, f.edge, 1, cv::LINE_AA);
            }
        }
    }

    // ── 2) Oriented model-instance boxes on top (opaque edges) ──────────────────────────────────
    const auto project_clipped_segment = [&](Mat::Vector3d a, Mat::Vector3d b,
                                             cv::Point& out_a, cv::Point& out_b) -> bool
    {
        if (a.y() <= kNearY and b.y() <= kNearY)
            return false;
        if (a.y() <= kNearY)
            a = a + ((kNearY - a.y()) / (b.y() - a.y())) * (b - a);
        else if (b.y() <= kNearY)
            b = b + ((kNearY - b.y()) / (a.y() - b.y())) * (a - b);

        const Eigen::Vector2d uv0 = cam.project(a);
        const Eigen::Vector2d uv1 = cam.project(b);
        if (not std::isfinite(uv0.x()) or not std::isfinite(uv0.y())
            or not std::isfinite(uv1.x()) or not std::isfinite(uv1.y()))
            return false;
        out_a = cv::Point(static_cast<int>(std::lround(uv0.x())), static_cast<int>(std::lround(uv0.y())));
        out_b = cv::Point(static_cast<int>(std::lround(uv1.x())), static_cast<int>(std::lround(uv1.y())));
        return true;
    };

    static constexpr int box_edges[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0},   // bottom
        {4, 5}, {5, 6}, {6, 7}, {7, 4},   // top
        {0, 4}, {1, 5}, {2, 6}, {3, 7}    // verticals
    };

    for (const auto& box : boxes)
    {
        const float c = std::cos(box.yaw_rad), s = std::sin(box.yaw_rad);
        const float hw = box.half_extents.x(), hd = box.half_extents.y(), hh = box.half_extents.z();
        const std::array<Eigen::Vector3f, 8> local = {
            Eigen::Vector3f{-hw, -hd, -hh}, Eigen::Vector3f{ hw, -hd, -hh},
            Eigen::Vector3f{ hw,  hd, -hh}, Eigen::Vector3f{-hw,  hd, -hh},
            Eigen::Vector3f{-hw, -hd,  hh}, Eigen::Vector3f{ hw, -hd,  hh},
            Eigen::Vector3f{ hw,  hd,  hh}, Eigen::Vector3f{-hw,  hd,  hh}
        };

        std::array<Mat::Vector3d, 8> corners_cam;
        for (std::size_t i = 0; i < local.size(); ++i)
        {
            const Eigen::Vector3f p_room(box.center.x() + c * local[i].x() - s * local[i].y(),
                                         box.center.y() + s * local[i].x() + c * local[i].y(),
                                         box.center.z() + local[i].z());
            corners_cam[i] = basis.to_camera(p_room);
        }

        const cv::Scalar col = color_for_category(box.category);
        for (const auto& e : box_edges)
        {
            cv::Point p0, p1;
            if (project_clipped_segment(corners_cam[e[0]], corners_cam[e[1]], p0, p1)
                and cv::clipLine(canvas.size(), p0, p1))
                cv::line(canvas, p0, p1, col, 2, cv::LINE_AA);
        }

        // Name label at the projected top-front corner (corner 4), in front of the camera and in-bounds.
        if (not box.node_name.empty() and corners_cam[4].y() > kNearY)
        {
            const Eigen::Vector2d uv = cam.project(corners_cam[4]);
            if (std::isfinite(uv.x()) and std::isfinite(uv.y())
                and uv.x() >= 0 and uv.x() < canvas.cols and uv.y() >= 0 and uv.y() < canvas.rows)
                cv::putText(canvas, box.node_name,
                            cv::Point(static_cast<int>(std::lround(uv.x())) + 3,
                                      static_cast<int>(std::lround(uv.y())) - 3),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv::LINE_AA);
        }
    }
}

}  // namespace rc
