/*
 * chair_contour.h — the chair's SILHOUETTE for the classifier-free contour channel (common/contour_edge).
 *
 * ★A CHAIR IS NOT A BOX, which is why this is not refrigerator_concept's project_box_face. The depth half
 * of the channel reads a few pixels INSIDE every contour edge expecting the object's own surface at the
 * predicted range, and tests that the world RECEDES just outside it. A chair's bounding box is mostly AIR
 * (between the legs, under the seat, beside the backrest), so on a box face those inside samples land on
 * the floor or wall behind and the channel refutes every real chair — measured on the synthetic harness:
 * a present chair seen from the front scored depth verdict −0.34 with a +1.83 m bias on its box face.
 *
 * So the contour is the OUTER SILHOUETTE OF THE SOLID PARTS: the seat slab ∪ the backrest slab, projected.
 * Every edge of a solid's outer silhouette is an occluding boundary (something farther lies beyond it) and
 * everything inside it is that solid's surface, which are exactly the two things the depth scorer assumes.
 * The legs are left out: 7.5 cm posts are a few pixels wide at working range, and a silhouette that wraps
 * them is mostly edges whose inside samples miss.
 *
 * ★THE SILHOUETTE IS COMPUTED, NOT CHOSEN PER VIEW. A first version picked a hand-drawn outline by which
 * side the camera was on (backrest front face + seat top from the front, rear face from behind). It was
 * right from the front and the back and WRONG from the side, where the backrest's front-face edges collapse
 * onto a 7.5 cm sliver and half their inside samples are air (present chair, side view: verdict −0.07 at
 * 1.5 m, −0.73 at 4 m). Each slab is convex, so its silhouette is the convex hull of its 8 projected
 * corners; the union of the two hulls is the solid's silhouette from ANY viewpoint, with no cases.
 *
 * The union is rasterised (robust to the collinear edges two coplanar slabs produce, which break an exact
 * polygon union) and each vertex of its boundary is lifted back to 3-D on the hull edge it lies on —
 * perspective-correctly — so the result is a loop of WORLD points that project_loop_slid can reproject and
 * slide along the floor for the null, exactly as door and fridge do.
 *
 * Header-only, no graph and no camera — the caller supplies the projector — so the test harness scores the
 * SAME silhouette the agent does.
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/imgproc.hpp>

#include "../../common/contour_edge/contour_edge_project.h"

namespace rc
{

struct ChairOutlineDims
{
    float seat_w = 0.60f, seat_d = 0.52f, seat_h = 0.595f, back_h = 0.655f, seat_thickness = 0.075f;
    float floor_z = 0.0f;
};

// The believed chair's silhouette + its slid null. Empty face ⇒ NOT MEASURED (a corner behind the camera,
// or a silhouette too small to have a boundary) — the caller must HOLD, never read it as a refutation.
// Local frame: +y is the seat's front, the backrest stands on the −y edge (matches ChairBelief::sdf_back).
[[nodiscard]] inline rc::edges::ContourSet chair_contour_set(float cx, float cy, float yaw,
                                                             const ChairOutlineDims& d,
                                                             const rc::edges::PointProjector& project)
{
    rc::edges::ContourSet none;
    if (not project)
        return none;
    const double c = std::cos(yaw), sn = std::sin(yaw);
    const double hw = 0.5 * d.seat_w, hd = 0.5 * d.seat_d, t = d.seat_thickness;
    const double zs = d.floor_z + d.seat_h;   // seat top
    const double zb = zs - t;                 // seat underside
    const double zt = zs + d.back_h;          // backrest top
    const double yf = -hd + t;                // backrest FRONT face
    const auto W = [&](double lx, double ly, double lz) -> Eigen::Vector3d
    { return {cx + c * lx - sn * ly, cy + sn * lx + c * ly, lz}; };

    struct Corner { Eigen::Vector3d P; cv::Point2f px; double depth; };
    // The backrest slab is taken down to the seat UNDERSIDE: the solid is identical (that strip is inside
    // the seat anyway) and the two projections then OVERLAP rather than merely touch along the hinge.
    const std::array<std::array<double, 6>, 2> slabs{{
        {-hw, hw, -hd, hd, zb, zs},           // seat
        {-hw, hw, -hd, yf, zb, zt}}};         // backrest
    std::array<std::vector<Corner>, 2> part;
    for (std::size_t k = 0; k < slabs.size(); ++k)
    {
        const auto& b = slabs[k];
        for (int i = 0; i < 8; ++i)
        {
            const Eigen::Vector3d P = W(b[(i & 1) ? 1 : 0], b[(i & 2) ? 3 : 2], b[(i & 4) ? 5 : 4]);
            const auto v = project(P);
            // All-or-nothing, like project_loop: a slab with a corner behind the camera has no meaningful
            // silhouette, and a partial one is a different shape.
            if (not v.has_value() or not (v->depth_m > 0.0f))
                return none;
            part[k].push_back({P, v->px, static_cast<double>(v->depth_m)});
        }
    }

    // Each slab's silhouette = convex hull of its projected corners (as indices, to keep the 3-D).
    std::array<std::vector<int>, 2> hull;
    cv::Rect2f bb;
    bool have_bb = false;
    for (std::size_t k = 0; k < 2; ++k)
    {
        std::vector<cv::Point2f> pts;
        for (const auto& q : part[k]) pts.push_back(q.px);
        cv::convexHull(pts, hull[k], false, false);
        const cv::Rect2f r = cv::boundingRect(pts);
        bb = have_bb ? (bb | r) : r;
        have_bb = true;
    }
    if (bb.width < 4.0f or bb.height < 4.0f)
        return none;                          // a few pixels across: no boundary to measure

    // Rasterise the union in a canvas around the silhouette (it may extend off-image at close range —
    // that is kept, the scorers skip off-image samples themselves). Capped at 2048 px for a chair that
    // nearly touches the lens; the boundary is lifted back onto exact hull edges below, so the cap costs
    // vertex placement only, not geometry.
    constexpr float kPad = 4.0f;
    const float scale = std::min(1.0f, 2048.0f / std::max(bb.width, bb.height));
    const cv::Point2f org(bb.x - kPad / scale, bb.y - kPad / scale);
    cv::Mat mask = cv::Mat::zeros(static_cast<int>(std::ceil(bb.height * scale + 2 * kPad)) + 1,
                                  static_cast<int>(std::ceil(bb.width * scale + 2 * kPad)) + 1, CV_8UC1);
    constexpr int kShift = 4;                 // sub-pixel vertex placement
    for (std::size_t k = 0; k < 2; ++k)
    {
        std::vector<cv::Point> poly;
        for (const int i : hull[k])
        {
            const cv::Point2f q = (part[k][i].px - org) * scale;
            poly.emplace_back(static_cast<int>(std::lround(q.x * (1 << kShift))),
                              static_cast<int>(std::lround(q.y * (1 << kShift))));
        }
        cv::fillConvexPoly(mask, poly, cv::Scalar(255), cv::LINE_8, kShift);
    }
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty())
        return none;
    const auto& outer = *std::ranges::max_element(contours, {}, [](const auto& ct) { return cv::contourArea(ct); });
    std::vector<cv::Point> approx;
    cv::approxPolyDP(outer, approx, 1.0, true);
    if (approx.size() < 3)
        return none;

    // Lift each boundary vertex onto the nearest hull edge, in 3-D. Inverse depth is what is linear in
    // image space, so the 3-D point at image fraction s along an edge is the inverse-depth-weighted blend.
    std::vector<Eigen::Vector3d> loop;
    loop.reserve(approx.size());
    for (const auto& a : approx)
    {
        const cv::Point2f q = cv::Point2f(static_cast<float>(a.x) + 0.5f, static_cast<float>(a.y) + 0.5f) / scale + org;
        double best = std::numeric_limits<double>::infinity();
        Eigen::Vector3d P = Eigen::Vector3d::Zero();
        for (std::size_t k = 0; k < 2; ++k)
            for (std::size_t e = 0; e < hull[k].size(); ++e)
            {
                const Corner& A = part[k][hull[k][e]];
                const Corner& B = part[k][hull[k][(e + 1) % hull[k].size()]];
                const cv::Point2f ab = B.px - A.px;
                const double L2 = static_cast<double>(ab.dot(ab));
                const double s = L2 > 1e-9 ? std::clamp(static_cast<double>((q - A.px).dot(ab)) / L2, 0.0, 1.0) : 0.0;
                const cv::Point2f foot = A.px + static_cast<float>(s) * ab;
                const double dist = cv::norm(q - foot);
                if (dist < best)
                {
                    best = dist;
                    const double wa = (1.0 - s) / A.depth, wb = s / B.depth;
                    P = (wa * A.P + wb * B.P) / (wa + wb);
                }
            }
        if (loop.empty() or (loop.back() - P).norm() > 1e-3)
            loop.push_back(P);
    }
    if (loop.size() >= 2 and (loop.front() - loop.back()).norm() <= 1e-3)
        loop.pop_back();
    if (loop.size() < 3)
        return none;

    // The null: the same silhouette slid along the floor in the chair's own lateral direction by ±1 and
    // ±1.6 seat widths — same range, same floor, same light. In 3-D, not in pixels, so each copy keeps the
    // chair's foreshortening (the door's lesson, common/contour_edge/contour_edge_project.h).
    return rc::edges::project_loop_slid(loop, Eigen::Vector3d(c, sn, 0.0), d.seat_w, project);
}

}  // namespace rc
