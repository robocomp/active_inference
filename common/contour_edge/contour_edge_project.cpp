#include "contour_edge_project.h"

#include <opencv2/imgproc.hpp>

#include <cmath>

namespace rc::edges
{

namespace
{

// Project one loop of world points into a Contour. All-or-nothing: a single corner the caller's
// projector rejects (behind the camera, non-finite) invalidates the contour, because a quad missing a
// corner is not a smaller quad, it is a different shape.
//
// ⚠KEPT even when partly OUT OF FRAME. The scorers skip off-image samples themselves and report how
// many they actually took, so a contour whose lintel is clipped contributes proportionally less with no
// special case — and close range, where the object overflows the frame, is precisely the situation this
// channel exists to survive.
[[nodiscard]] std::optional<Contour> project_loop(const std::vector<Eigen::Vector3d>& pts,
                                                  const PointProjector& project)
{
    Contour c;
    c.px.reserve(pts.size());
    c.depth_m.reserve(pts.size());
    for (const auto& p : pts)
    {
        const auto v = project(p);
        if (not v.has_value())
            return std::nullopt;
        c.px.emplace_back(static_cast<int>(std::lround(v->px.x)), static_cast<int>(std::lround(v->px.y)));
        c.depth_m.push_back(v->depth_m);
    }
    return c;
}

}   // namespace

ContourSet project_quad(const std::array<Eigen::Vector3d, 4>& corners,
                        const Eigen::Vector3d& slide_dir,
                        float slide_span_m,
                        const PointProjector& project)
{
    ContourSet out;
    if (not project or slide_span_m <= 1e-4f)
        return out;
    const double dir_norm = slide_dir.norm();
    if (not std::isfinite(dir_norm) or dir_norm < 1e-6)
        return out;
    const Eigen::Vector3d u = slide_dir / dir_norm;

    const std::vector<Eigen::Vector3d> loop(corners.begin(), corners.end());
    auto face = project_loop(loop, project);
    if (not face.has_value())
        return out;                      // a corner behind the camera ⇒ nothing measured this frame
    out.face = std::move(*face);
    out.area_px = static_cast<float>(std::abs(cv::contourArea(out.face.px)));

    // The null: the same quad, translated in 3-D along its own plane, then reprojected. Translating in
    // 3-D rather than in pixels is the whole point — a displaced copy keeps the object's range and
    // foreshortening, so it is the same shape seen the same way from the same place, which is what makes
    // it a comparison rather than a coincidence.
    for (const float k : kControlOffsets)
    {
        const Eigen::Vector3d shift = u * static_cast<double>(k * slide_span_m);
        std::vector<Eigen::Vector3d> moved;
        moved.reserve(loop.size());
        for (const auto& p : loop)
            moved.push_back(p + shift);
        if (auto c = project_loop(moved, project); c.has_value())
            out.controls.push_back(std::move(*c));
        // A control that falls behind the camera is DROPPED, not counted as a zero: an unmeasurable
        // control says nothing about the object, and averaging a zero into s_control would manufacture
        // support the image never gave.
    }
    return out;
}

ContourSet project_box_face(const BoxFootprint& box,
                            const Eigen::Vector3d& camera_pos_world,
                            const PointProjector& project)
{
    ContourSet out;
    if (box.w <= 1e-4f or box.d <= 1e-4f or box.z_max <= box.z_min or not project)
        return out;

    const double c = std::cos(box.yaw), s = std::sin(box.yaw);
    const double hw = 0.5 * box.w, hd = 0.5 * box.d;
    const auto to_world = [&](double lx, double ly, double lz) -> Eigen::Vector3d
    { return {box.cx + c * lx - s * ly, box.cy + s * lx + c * ly, lz}; };

    // The four vertical side faces: outward LOCAL normal, then the two footprint corners of the bottom
    // edge. Corner order per face makes the projected loop bottom_a → bottom_b → top_b → top_a.
    struct Face { double nx, ny, ax, ay, bx, by, width; };
    const std::array<Face, 4> faces{{
        { 1.0,  0.0,   hw, -hd,   hw,  hd,  box.d },   // +x
        { 0.0,  1.0,   hw,  hd,  -hw,  hd,  box.w },   // +y
        {-1.0,  0.0,  -hw,  hd,  -hw, -hd,  box.d },   // −x
        { 0.0, -1.0,  -hw, -hd,   hw, -hd,  box.w },   // −y
    }};

    // ★THE FACE IS CHOSEN BY PROJECTED AREA, NOT BY WHICH SIDE WE CALL THE FRONT. The biggest visible
    // face is the one carrying the most boundary to measure, and it is a fact about this view rather
    // than a standing assumption about the object's orientation — which is itself a belief that can be
    // wrong, and would then silently pick a face the camera can barely see.
    float best_area = 0.0f;
    for (const auto& f : faces)
    {
        // Camera-facing test: the outward normal must point toward the camera.
        const Eigen::Vector3d n_world(c * f.nx - s * f.ny, s * f.nx + c * f.ny, 0.0);
        const Eigen::Vector3d centre = to_world(0.5 * (f.ax + f.bx), 0.5 * (f.ay + f.by),
                                                0.5 * (box.z_min + box.z_max));
        if (n_world.dot(camera_pos_world - centre) <= 0.0)
            continue;                                    // back-facing: not visible, not evidence
        ++out.n_faces_visible;

        // Slide along the face's own horizontal in-plane axis — for an upright box, the support plane.
        Eigen::Vector3d along = to_world(f.bx, f.by, 0.0) - to_world(f.ax, f.ay, 0.0);
        along.z() = 0.0;
        if (along.norm() < 1e-6)
            continue;

        const std::array<Eigen::Vector3d, 4> quad{
            to_world(f.ax, f.ay, box.z_min), to_world(f.bx, f.by, box.z_min),
            to_world(f.bx, f.by, box.z_max), to_world(f.ax, f.ay, box.z_max)};

        ContourSet cand = project_quad(quad, along, static_cast<float>(f.width), project);
        if (cand.face.valid() and cand.area_px > best_area)
        {
            best_area = cand.area_px;
            const int seen = out.n_faces_visible;
            out = std::move(cand);
            out.n_faces_visible = seen;
            out.outward_normal = Eigen::Vector2f(static_cast<float>(c * f.nx - s * f.ny),
                                                 static_cast<float>(s * f.nx + c * f.ny));
        }
    }
    return out;
}

}   // namespace rc::edges
