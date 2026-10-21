/*
 * table_contour.h — the table's SILHOUETTE for the classifier-free contour channel (common/contour_edge).
 *
 * ★THE CONTOUR IS THE TABLETOP SLAB, NOT THE BOUNDING BOX. The depth half of the channel reads a few pixels
 * INSIDE every contour edge expecting the object's own surface at the predicted range, and tests that the
 * world RECEDES just outside it. A table's bounding box is mostly AIR (between and under the legs), so on a
 * box face those inside samples land on the floor behind and the channel refutes every real table — the
 * failure chair_contour.h measured for the chair. The top slab [z_top − t, z_top] is solid and CONVEX, so its
 * silhouette from any viewpoint is the convex hull of its projected corners: every hull edge is an occluding
 * boundary, everything inside is the slab, and the hull vertices ARE projected 3-D corners — no lifting back
 * to 3-D is needed. Legs are left out (a few pixels wide at working range). A round table is the same slab
 * on a 16-gon.
 *
 * ★THE NULL SLIDES PERPENDICULAR TO THE LINE OF SIGHT, along the floor, by the slab's own width in that
 * direction: same range, same floor, same light. Sliding along the view ray would change the range and
 * compare against a different surface — a different question, not a control.
 *
 * Header-only, no graph and no camera — the caller supplies the projector — so a test harness scores the
 * SAME silhouette the agent does.
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/imgproc.hpp>

#include "../../common/contour_edge/contour_edge_project.h"

namespace rc
{

struct TableSlab
{
    double cx = 0.0, cy = 0.0, yaw = 0.0;   // footprint centre + orientation, room frame
    double w = 1.0, d = 0.6;                // full extents along local x / y (round: the two diameters)
    double z_top = 0.75, thickness = 0.03;  // tabletop surface height and slab thickness
    bool   round = false;
};

// The believed tabletop's outline + its slid null. Empty face ⇒ NOT MEASURED (a corner behind the camera, or
// an outline too small to have a boundary) — the caller must HOLD, never read it as a refutation.
[[nodiscard]] inline rc::edges::ContourSet table_contour_set(const TableSlab& t, const Eigen::Vector2d& cam_xy,
                                                             const rc::edges::PointProjector& project)
{
    rc::edges::ContourSet none;
    if (not project)
        return none;
    const double c = std::cos(t.yaw), sn = std::sin(t.yaw), hw = 0.5 * t.w, hd = 0.5 * t.d;
    std::vector<Eigen::Vector2d> rim;                                 // footprint outline, room frame
    if (t.round)
    {
        constexpr int kSides = 16;
        for (int i = 0; i < kSides; ++i)
        {
            const double a = 2.0 * M_PI * i / kSides;
            const double lx = hw * std::cos(a), ly = hd * std::sin(a);
            rim.emplace_back(t.cx + c * lx - sn * ly, t.cy + sn * lx + c * ly);
        }
    }
    else
        for (const auto& [lx, ly] : std::array<std::pair<double, double>, 4>{{{-hw, -hd}, {hw, -hd}, {hw, hd}, {-hw, hd}}})
            rim.emplace_back(t.cx + c * lx - sn * ly, t.cy + sn * lx + c * ly);

    std::vector<Eigen::Vector3d> corners;
    std::vector<cv::Point2f>     px;
    for (const double z : {t.z_top - t.thickness, t.z_top})
        for (const auto& r : rim)
        {
            const Eigen::Vector3d P(r.x(), r.y(), z);
            const auto v = project(P);
            if (not v.has_value() or not (v->depth_m > 0.0f))
                return none;                                          // all-or-nothing: a partial slab is another shape
            corners.push_back(P);
            px.push_back(v->px);
        }
    std::vector<int> hull;
    cv::convexHull(px, hull, false, false);
    const cv::Rect2f bb = cv::boundingRect(px);
    if (hull.size() < 3 or bb.width < 4.0f or bb.height < 4.0f)
        return none;                                                  // a few pixels across: no boundary to measure
    std::vector<Eigen::Vector3d> loop;
    loop.reserve(hull.size());
    for (const int i : hull)
        loop.push_back(corners[static_cast<std::size_t>(i)]);

    Eigen::Vector2d los = Eigen::Vector2d(t.cx, t.cy) - cam_xy;
    if (los.norm() < 1e-3)
        return none;
    los.normalize();
    const Eigen::Vector2d lat(-los.y(), los.x());
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    for (const auto& r : rim)
    {
        lo = std::min(lo, lat.dot(r));
        hi = std::max(hi, lat.dot(r));
    }
    return rc::edges::project_loop_slid(loop, Eigen::Vector3d(lat.x(), lat.y(), 0.0), static_cast<float>(hi - lo), project);
}

// ★THE DEPTH SCORER PREDICTS THE INSIDE SAMPLE AT THE EDGE'S OWN DEPTH, which is exact for a vertical face
// (fridge, door, chair back) and wrong for a tabletop seen from above: across a horizontal surface the depth
// changes fast with image row, so a sample `offset_px` inside the far edge is nearer than the edge by
// (depth gradient × offset). Unmodelled, that reads as "the surface is not where predicted" and the channel
// half-refutes a table that is there (synthetic: depth verdict +0.19 present, −0.73 with a 10 cm pose error).
// This is the prediction's own uncertainty, so it belongs in σ_d: the steepest depth-per-pixel along the
// outline's edges times the scorer's largest inward offset. Combine in quadrature with the position σ.
[[nodiscard]] inline float inward_depth_slack_m(const rc::edges::Contour& face, int offset_px)
{
    float grad = 0.0f;                                               // metres of depth per image pixel
    const std::size_t n = face.px.size();
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j)
        {
            const float dpx = static_cast<float>(cv::norm(face.px[i] - face.px[j]));
            if (dpx >= 1.0f)
                grad = std::max(grad, std::abs(face.depth_m[i] - face.depth_m[j]) / dpx);
        }
    return grad * static_cast<float>(offset_px);
}

}  // namespace rc
