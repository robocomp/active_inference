/*
 * bottle_projection.cpp — see bottle_projection.h.
 */

#include "bottle_projection.h"

#include <cmath>

#include "../../common/room_resolve/room_resolve.h"   // rc::room::current_room_frame — never the literal "room"

namespace rc
{

std::optional<Eigen::Matrix4d> room_T_zed_at(DSR::DSRGraph& G, DSR::InnerEigenAPI& inner_eigen,
                                             std::uint64_t stamp_ms)
{
    // Pin the moving room→body hop to the frame's capture stamp; keep the rigid body→zed mount at latest
    // (it carries only a bootstrap stamp — a pinned query would fail). Same split as hood/refrigerator.
    const auto rtb = inner_eigen.get_transformation_matrix(rc::room::current_room_frame(G), "body", stamp_ms, "RT",
                                                           DSR::RT_API::TimeQuery::Interpolated);
    const auto btz = inner_eigen.get_transformation_matrix("body", "zed", 0);
    if (not (rtb.has_value() and btz.has_value()))
        return std::nullopt;
    const auto to_mat4 = [](const Mat::RTMat& T)
    {
        Eigen::Matrix4d m;
        const auto& s = T.matrix();
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) m(i, j) = s(i, j);   // no aligned load
        return m;
    };
    return to_mat4(rtb.value()) * to_mat4(btz.value());
}

rc::edges::PointProjector make_zed_projector(const Eigen::Matrix4d& zed_T_room, const DSR::CameraAPI& cam,
                                             int frame_cols, int frame_rows)
{
    const float W = static_cast<float>(cam.get_width());
    const float H = static_cast<float>(cam.get_height());
    if (W <= 0.f or H <= 0.f)
        return {};
    // The CameraAPI projects into ITS intrinsic frame; rescale to the delivered image if they differ, or the
    // contour lands somewhere plausible but wrong and reads as the object having moved.
    const float sx = (frame_cols > 0) ? static_cast<float>(frame_cols) / W : 1.0f;
    const float sy = (frame_rows > 0) ? static_cast<float>(frame_rows) / H : 1.0f;
    return [zed_T_room, &cam, sx, sy](const Eigen::Vector3d& Pr) -> std::optional<rc::edges::ProjectedVertex>
    {
        const Eigen::Vector4d Pc = zed_T_room * Pr.homogeneous();
        if (Pc.y() <= 0.20) return std::nullopt;                       // behind / at the image plane
        const Eigen::Vector2d uv = cam.project(Eigen::Vector3d(Pc.x(), Pc.y(), Pc.z()));
        if (not std::isfinite(uv.x()) or not std::isfinite(uv.y())) return std::nullopt;
        rc::edges::ProjectedVertex v;
        v.px = cv::Point2f(static_cast<float>(uv.x()) * sx, static_cast<float>(uv.y()) * sy);
        v.depth_m = static_cast<float>(Pc.y());   // ZED depth plane = camera-frame FORWARD coordinate
        return v;
    };
}

}   // namespace rc
