/*
 * residual_zed_boost.cpp  —  see residual_zed_boost.h
 */

#include "residual_zed_boost.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <tuple>

namespace rc
{

namespace
{
// 2D oriented-box footprint SDF (same convention as ResidualBelief/ResidualModel).
float footprint_sdf(const BoostBox& b, float x, float y)
{
    const float c = std::cos(b.yaw), s = std::sin(b.yaw);
    const float dx = x - b.cx, dy = y - b.cy;
    const float lx =  c * dx + s * dy;
    const float ly = -s * dx + c * dy;
    const float qx = std::abs(lx) - 0.5f * b.w, qy = std::abs(ly) - 0.5f * b.d;
    const float outside = std::sqrt(std::max(qx, 0.0f) * std::max(qx, 0.0f) + std::max(qy, 0.0f) * std::max(qy, 0.0f));
    return outside + std::min(std::max(qx, qy), 0.0f);
}
}  // namespace

std::vector<Eigen::Vector3f> backproject_box_region(const DepthImage& depth, const CamIntrinsics& K,
                                                    const Eigen::Matrix4f& room_T_cam, const BoostBox& box,
                                                    const ZedBoostParams& params)
{
    std::vector<Eigen::Vector3f> out;
    if (depth.width <= 0 or depth.height <= 0 or static_cast<int>(depth.depth.size()) < depth.width * depth.height
        or K.fx <= 0.0f or K.fy <= 0.0f)
        return out;

    const Eigen::Matrix4f cam_T_room = room_T_cam.inverse();

    // 1) Project the 8 box corners into the image to get the pixel bounding box. The DSR "zed" frame in this
    //    codebase is x-right, y-DEPTH (optical axis), z-up (matches retina/graph_publisher + bottle_fitter):
    //    u = fx·x/y + cx,  v = cy − fy·z/y,  in front ⇔ y > min_depth.
    const float c = std::cos(box.yaw), s = std::sin(box.yaw), hw = 0.5f * box.w, hd = 0.5f * box.d;
    float umin = 1e9f, umax = -1e9f, vmin = 1e9f, vmax = -1e9f;
    bool any_infront = false;
    for (int i = 0; i < 4; ++i)
    {
        const float sx = (i & 1) ? 1.f : -1.f, sy = (i & 2) ? 1.f : -1.f;
        const float rx = box.cx + (c * sx * hw - s * sy * hd);
        const float ry = box.cy + (s * sx * hw + c * sy * hd);
        for (float rz : {box.z_min, box.z_max})
        {
            const Eigen::Vector4f pc = cam_T_room * Eigen::Vector4f(rx, ry, rz, 1.0f);
            if (pc.y() <= params.min_depth_m) continue;                 // behind / too close (depth is +Y)
            any_infront = true;
            const float u = K.fx * pc.x() / pc.y() + K.cx;
            const float v = K.cy - K.fy * pc.z() / pc.y();
            umin = std::min(umin, u); umax = std::max(umax, u);
            vmin = std::min(vmin, v); vmax = std::max(vmax, v);
        }
    }
    if (not any_infront)
        return out;

    const int u0 = std::max(0, static_cast<int>(std::floor(umin)));
    const int u1 = std::min(depth.width  - 1, static_cast<int>(std::ceil(umax)));
    const int v0 = std::max(0, static_cast<int>(std::floor(vmin)));
    const int v1 = std::min(depth.height - 1, static_cast<int>(std::ceil(vmax)));
    if (u0 > u1 or v0 > v1)
        return out;

    // 2) Backproject the depth in that region; keep points inside the box (+margins) → rejects background.
    const int step = std::max(1, params.subsample);
    for (int v = v0; v <= v1; v += step)
        for (int u = u0; u <= u1; u += step)
        {
            const float dz = depth.depth[static_cast<std::size_t>(v) * depth.width + u];
            if (not std::isfinite(dz) or dz < params.min_depth_m or dz > params.max_depth_m)
                continue;
            // y-depth zed frame: x=(u−cx)·d/fx (right), y=d (depth), z=(cy−v)·d/fy (up).
            const Eigen::Vector4f pc((static_cast<float>(u) - K.cx) / K.fx * dz, dz,
                                     (K.cy - static_cast<float>(v)) / K.fy * dz, 1.0f);
            const Eigen::Vector4f pr = room_T_cam * pc;
            if (pr.z() < box.z_min - params.z_margin_m or pr.z() > box.z_max + params.z_margin_m)
                continue;
            if (footprint_sdf(box, pr.x(), pr.y()) > params.xy_margin_m)
                continue;
            out.emplace_back(pr.x(), pr.y(), pr.z());
            if (static_cast<int>(out.size()) >= params.max_points)
                return out;
        }
    return out;
}

std::vector<Eigen::Vector3f> voxel_downsample(const std::vector<Eigen::Vector3f>& pts, float leaf_m, int max_out)
{
    std::vector<Eigen::Vector3f> out;
    if (pts.empty() or leaf_m <= 0.0f)
        return out;
    // Accumulate each voxel's centroid. std::map keeps a deterministic (sorted) cell order — no RNG.
    std::map<std::tuple<int, int, int>, std::pair<Eigen::Vector3f, int>> cells;
    const float inv = 1.0f / leaf_m;
    for (const auto& p : pts)
    {
        const auto key = std::make_tuple(static_cast<int>(std::floor(p.x() * inv)),
                                         static_cast<int>(std::floor(p.y() * inv)),
                                         static_cast<int>(std::floor(p.z() * inv)));
        auto& acc = cells[key];
        acc.first += p; ++acc.second;
    }
    out.reserve(cells.size());
    for (const auto& [key, acc] : cells)
    {
        out.emplace_back(acc.first / static_cast<float>(acc.second));
        if (max_out > 0 and static_cast<int>(out.size()) >= max_out)
            break;
    }
    return out;
}

std::vector<Eigen::Vector3f> backproject_fov(const DepthImage& depth, const CamIntrinsics& K,
                                             const Eigen::Matrix4f& room_T_cam, const ZedBoostParams& params)
{
    std::vector<Eigen::Vector3f> raw;
    if (depth.width <= 0 or depth.height <= 0 or static_cast<int>(depth.depth.size()) < depth.width * depth.height
        or K.fx <= 0.0f or K.fy <= 0.0f)
        return raw;
    // GUARD: a bad room←cam transform (RT extrapolation on a stale/bad depth stamp) scatters points to ~1e31.
    // Never let garbage reach clustering — reject a non-finite transform up front, and each point that lands
    // outside a sane room envelope below.
    if (not room_T_cam.allFinite())
        return raw;

    const int step = std::max(1, params.fov_subsample);
    raw.reserve(static_cast<std::size_t>(depth.width / step) * (depth.height / step));
    for (int v = 0; v < depth.height; v += step)
        for (int u = 0; u < depth.width; u += step)
        {
            const float dz = depth.depth[static_cast<std::size_t>(v) * depth.width + u];
            if (not std::isfinite(dz) or dz < params.min_depth_m or dz > params.max_depth_m)
                continue;
            // y-depth zed frame: x=(u−cx)·d/fx (right), y=d (depth), z=(cy−v)·d/fy (up).
            const Eigen::Vector4f pc((static_cast<float>(u) - K.cx) / K.fx * dz, dz,
                                     (K.cy - static_cast<float>(v)) / K.fy * dz, 1.0f);
            const Eigen::Vector4f pr = room_T_cam * pc;
            if (not pr.allFinite() or std::abs(pr.x()) > 50.0f or std::abs(pr.y()) > 50.0f
                or pr.z() < -2.0f or pr.z() > 10.0f)                 // outside a sane room envelope ⇒ garbage
                continue;
            raw.emplace_back(pr.x(), pr.y(), pr.z());
        }
    return voxel_downsample(raw, params.voxel_m, params.fov_max_points);
}

// ─── Self-test: a synthetic depth image of a box; backprojection must recover points ON the box ──────────
bool zed_boost_self_test()
{
    bool ok = true;
    auto check = [&](bool c, const char* m) { if (!c) { ok = false; std::printf("  FAIL: %s\n", m); } };

    // Camera at room origin looking +x (room). This codebase's zed frame is x-right, y-DEPTH, z-up, so a
    // camera facing +x has: y_cam(depth)→+x room, z_cam(up)→+z room, x_cam(right)→-y room. Build room_T_cam:
    Eigen::Matrix4f room_T_cam = Eigen::Matrix4f::Identity();
    room_T_cam.col(0).head<3>() = Eigen::Vector3f(0, -1, 0);   // x_cam (right)  → -y room
    room_T_cam.col(1).head<3>() = Eigen::Vector3f(1, 0, 0);    // y_cam (depth)  → +x room
    room_T_cam.col(2).head<3>() = Eigen::Vector3f(0, 0, 1);    // z_cam (up)     → +z room
    room_T_cam.col(3).head<3>() = Eigen::Vector3f(0, 0, 1.0f); // camera 1 m up

    const CamIntrinsics K{500.f, 500.f, 320.f, 240.f};
    const int W = 640, H = 480;

    // A box 2 m in front (room x=2), 0.6 wide, 0.4 deep, z 0.4..1.0. Render its FRONT face depth into the
    // image (the face at x = 2 - 0.2 = 1.8 → camera z = 1.8 - ... actually camera at x=0, so depth = 1.8).
    const BoostBox box{2.0f, 0.0f, 0.0f, 0.6f, 0.4f, 0.4f, 1.0f};
    const float face_x = box.cx - 0.5f * box.d;   // 1.8 (nearest face, room x)

    DepthImage img; img.width = W; img.height = H; img.depth.assign(static_cast<std::size_t>(W) * H, 0.0f);
    // Fill the image: for each pixel, if the ray hits the box front face within its yz extent, depth=face.
    const Eigen::Matrix4f cam_T_room = room_T_cam.inverse();
    // Background behind the box (room x=4) everywhere else, to test rejection.
    for (int v = 0; v < H; ++v)
        for (int u = 0; u < W; ++u)
        {
            // ray dir in room for this pixel (y-depth zed frame: x-right, y-forward, z-up)
            const Eigen::Vector3f d_cam((u - K.cx) / K.fx, 1.0f, (K.cy - v) / K.fy);
            const Eigen::Vector3f d_room = room_T_cam.topLeftCorner<3,3>() * d_cam;
            const Eigen::Vector3f o_room = room_T_cam.col(3).head<3>();
            // hit the box front plane x=face_x: o.x + t*d.x = face_x
            float best = 4.0f;   // background depth (camera-frame y) if no box hit
            if (std::abs(d_room.x()) > 1e-6f)
            {
                const float t = (face_x - o_room.x()) / d_room.x();
                if (t > 0)
                {
                    const Eigen::Vector3f p = o_room + t * d_room;
                    if (std::abs(p.y() - box.cy) <= 0.5f * box.w and p.z() >= box.z_min and p.z() <= box.z_max)
                    {
                        // camera-frame depth = (cam_T_room * p).y
                        const Eigen::Vector4f pc = cam_T_room * Eigen::Vector4f(p.x(), p.y(), p.z(), 1.0f);
                        best = pc.y();
                    }
                }
            }
            img.depth[static_cast<std::size_t>(v) * W + u] = best;
        }

    ZedBoostParams P;
    const auto pts = backproject_box_region(img, K, room_T_cam, box, P);
    std::printf("  backprojected %zu points\n", pts.size());
    check(pts.size() > 50, "should recover a dense set of box-face points");

    // All returned points must lie on the box front face (x≈1.8) and within the box yz extent.
    int on_face = 0, in_extent = 0;
    for (const auto& p : pts)
    {
        if (std::abs(p.x() - face_x) < 0.05f) ++on_face;
        if (std::abs(p.y() - box.cy) <= 0.5f * box.w + P.xy_margin_m and
            p.z() >= box.z_min - P.z_margin_m and p.z() <= box.z_max + P.z_margin_m) ++in_extent;
    }
    std::printf("  on-face=%d in-extent=%d (of %zu)\n", on_face, in_extent, pts.size());
    check(on_face == static_cast<int>(pts.size()), "all points should be on the box face (background rejected)");
    check(in_extent == static_cast<int>(pts.size()), "all points should be within the box extent");

    // ── whole-FoV pass: NOT box-gated → returns the box face AND the background wall, voxel-downsampled ──
    ZedBoostParams F; F.voxel_m = 0.03f; F.fov_subsample = 2; F.fov_max_points = 100000;
    const auto scene = backproject_fov(img, K, room_T_cam, F);
    std::printf("  fov scene points (downsampled) = %zu\n", scene.size());
    check(scene.size() > 100, "FoV pass should return a dense scene cloud");
    int on_face_fov = 0, on_bg = 0;
    for (const auto& p : scene)
    {
        if (std::abs(p.x() - face_x)  < 0.05f) ++on_face_fov;   // box front face at room x≈1.8
        if (std::abs(p.x() - 4.0f)    < 0.05f) ++on_bg;         // background wall at room x=4
    }
    std::printf("  fov: on-face=%d on-background=%d\n", on_face_fov, on_bg);
    check(on_face_fov > 0, "FoV pass must include the box face");
    check(on_bg > 0,       "FoV pass must include the background (explainers remove it, not the backproject)");

    // Voxel-downsample must not return two points in the same leaf cell.
    const float leaf = 0.03f;
    bool dup = false;
    for (std::size_t i = 1; i < scene.size() and not dup; ++i)
    {
        const auto& a = scene[i - 1]; const auto& b = scene[i];
        dup = (std::floor(a.x() / leaf) == std::floor(b.x() / leaf) and
               std::floor(a.y() / leaf) == std::floor(b.y() / leaf) and
               std::floor(a.z() / leaf) == std::floor(b.z() / leaf));
    }
    check(not dup, "voxel_downsample must emit one representative per cell");

    std::printf("zed_boost_self_test %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace rc
