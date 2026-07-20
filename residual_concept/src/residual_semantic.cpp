/*
 * residual_semantic.cpp  —  see residual_semantic.h
 */

#include "residual_semantic.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace rc
{

namespace
{
// Is `label` one of the walkable floor/ground classes? (small set → linear scan is fine and branch-predictable).
bool is_floor_class(std::uint8_t label, const std::vector<std::uint8_t>& ids)
{
    return std::find(ids.begin(), ids.end(), label) != ids.end();
}
}  // namespace

std::vector<float> semantic_obstacle_weights(const std::vector<Eigen::Vector3f>& points_room,
                                             const Eigen::Matrix4f& cam_T_room, const CamIntrinsics& K,
                                             int img_w, int img_h,
                                             const SemanticMap& sem, const Eigen::Vector3f& origin_room,
                                             float age_s, const SemanticFloorParams& p)
{
    std::vector<float> w(points_room.size(), 1.0f);        // default: full weight (a no-op)
    if (not p.enabled or not sem.valid() or img_w <= 0 or img_h <= 0
        or K.fx <= 0.0f or K.fy <= 0.0f or not cam_T_room.allFinite())
        return w;

    // FRESHNESS-as-precision: an older label map is less trustworthy (the scene may have shifted in the image as
    // the robot moved since the segmentation ran). Halve the suppression authority every fresh_half_life_s. This
    // is a continuous decay, not a staleness cutoff. (Ego-motion trust is applied separately, upstream, to the
    // whole ZED hit weight in the grid — so it is NOT re-applied here, to avoid double-counting motion.)
    const float freshness = p.fresh_half_life_s > 0.0f
                          ? std::exp2(-std::max(0.0f, age_s) / p.fresh_half_life_s) : 1.0f;
    const float suppress = std::clamp(p.floor_suppress, 0.0f, 1.0f) * freshness;
    if (suppress <= 0.0f) return w;

    // The label map may be at a different resolution than the depth image the intrinsics belong to — scale the
    // reprojected pixel into the label grid (they are the same ZED frame, so this is normally 1:1).
    const float sx = static_cast<float>(sem.width)  / static_cast<float>(img_w);
    const float sy = static_cast<float>(sem.height) / static_cast<float>(img_h);
    const float inv_hs2 = p.height_scale_m > 0.0f ? 1.0f / (p.height_scale_m * p.height_scale_m) : 0.0f;

    for (std::size_t i = 0; i < points_room.size(); ++i)
    {
        const Eigen::Vector3f& pr = points_room[i];
        // room → camera (y-depth zed frame: x-right, y=depth along optical axis, z-up).
        const Eigen::Vector4f pc = cam_T_room * Eigen::Vector4f(pr.x(), pr.y(), pr.z(), 1.0f);
        const float d = pc.y();                            // camera-frame depth
        if (d <= 1e-3f) continue;                          // behind / at the camera → no label, keep full weight
        // pinhole (y-depth): u = fx·x/d + cx,  v = cy − fy·z/d
        const int u = static_cast<int>(std::lround((K.fx * pc.x() / d + K.cx) * sx));
        const int v = static_cast<int>(std::lround((K.cy - K.fy * pc.z() / d) * sy));
        if (u < 0 or u >= sem.width or v < 0 or v >= sem.height) continue;   // out of frame → keep full weight
        const std::uint8_t label = sem.labels[static_cast<std::size_t>(v) * sem.width + u];
        if (not is_floor_class(label, p.floor_class_ids)) continue;          // obstacle class → keep full weight

        // Floor-class label. HEIGHT GATE: its authority decays with the point's height above the floor band, so a
        // clearly-elevated return (a real obstacle the net mislabelled) is never suppressed. gate=1 at the band → 0 high up.
        const float range    = std::hypot(pr.x() - origin_room.x(), pr.y() - origin_room.y());
        const float band_top = p.floor_z0 + p.floor_slope * range;
        const float h_above  = std::max(0.0f, pr.z() - band_top);
        const float gate     = inv_hs2 > 0.0f ? std::exp(-h_above * h_above * inv_hs2) : (h_above <= 0.0f ? 1.0f : 0.0f);
        w[i] = 1.0f - suppress * gate;                     // ∈ (1−suppress, 1]; multiplied into the grid hit weight
    }
    return w;
}

// ─── Self-test: the height gate + freshness + class selectivity + safety property ───────────────────────────
bool semantic_self_test()
{
    bool ok = true;
    auto check = [&](bool c, const char* m) { if (not c) { ok = false; std::printf("  FAIL: %s\n", m); } };

    // Camera at room origin looking +x (zed frame: x_cam→-y room, y_cam(depth)→+x room, z_cam(up)→+z room).
    Eigen::Matrix4f room_T_cam = Eigen::Matrix4f::Identity();
    room_T_cam.col(0).head<3>() = Eigen::Vector3f(0, -1, 0);
    room_T_cam.col(1).head<3>() = Eigen::Vector3f(1,  0, 0);
    room_T_cam.col(2).head<3>() = Eigen::Vector3f(0,  0, 1);
    room_T_cam.col(3).head<3>() = Eigen::Vector3f(0,  0, 1.0f);          // 1 m up
    const Eigen::Matrix4f cam_T_room = room_T_cam.inverse();
    const CamIntrinsics K{500.f, 500.f, 320.f, 240.f};
    const int W = 640, H = 480;

    // A label map that is ALL floor (class 3) → the strongest possible suppression everywhere it applies.
    SemanticMap sem; sem.width = W; sem.height = H; sem.stamp_ms = 0; sem.frame_id = 1;
    sem.labels.assign(static_cast<std::size_t>(W) * H, 3);

    SemanticFloorParams p;               // enabled=false by default
    // Two points 3 m ahead (room x=3, within the camera's vertical FoV given fy=500,H=480): one right at floor
    // level (z≈0) and one 1 m up (a chest-high obstacle). Nearer floor points fall below the image and are (safely)
    // left at full weight — exactly the geometry a 1 m-high camera looking horizontally sees.
    const Eigen::Vector3f p_floor(3.0f, 0.0f, 0.02f), p_high(3.0f, 0.0f, 1.0f);
    const Eigen::Vector3f origin = room_T_cam.col(3).head<3>();
    std::vector<Eigen::Vector3f> pts{p_floor, p_high};

    // (1) Disabled ⇒ no-op (all weights 1).
    { auto w = semantic_obstacle_weights(pts, cam_T_room, K, W, H, sem, origin, 0.0f, p);
      check(w.size() == 2 and w[0] == 1.0f and w[1] == 1.0f, "disabled must be a no-op"); }

    p.enabled = true; p.floor_suppress = 0.6f; p.fresh_half_life_s = 0.5f; p.height_scale_m = 0.15f;
    // (2) Enabled, fresh map: the FLOOR-level floor-labelled point is suppressed toward 1−0.6=0.4 …
    {
        auto w = semantic_obstacle_weights(pts, cam_T_room, K, W, H, sem, origin, 0.0f, p);
        check(w[0] < 0.5f, "near-floor floor-label must be down-weighted");
        // (3) …but the ELEVATED point (z=1 m) keeps ~full weight — the safety height gate (never suppress an obstacle).
        check(w[1] > 0.98f, "elevated point must keep full weight (height gate = safety)");
    }
    // (4) FRESHNESS: an old map suppresses LESS than a fresh one (authority decays with age).
    {
        auto w_fresh = semantic_obstacle_weights(pts, cam_T_room, K, W, H, sem, origin, 0.0f, p);
        auto w_stale = semantic_obstacle_weights(pts, cam_T_room, K, W, H, sem, origin, 1.0f, p);
        check(w_stale[0] > w_fresh[0], "an older label map must suppress less (freshness-as-precision)");
    }
    // (5) CLASS SELECTIVITY: a non-floor class (16=table) is never suppressed, even at floor level.
    {
        SemanticMap tbl = sem; tbl.labels.assign(static_cast<std::size_t>(W) * H, 16);
        auto w = semantic_obstacle_weights(pts, cam_T_room, K, W, H, tbl, origin, 0.0f, p);
        check(w[0] == 1.0f and w[1] == 1.0f, "non-floor class must never be suppressed");
    }
    // (6) SAFETY: the suppression can never drive a weight to 0 (down-weight, never discard).
    {
        SemanticFloorParams p2 = p; p2.floor_suppress = 1.0f;
        auto w = semantic_obstacle_weights({p_floor}, cam_T_room, K, W, H, sem, origin, 0.0f, p2);
        check(w[0] >= 0.0f and w[0] <= 1.0f, "weight stays in [0,1]");
    }

    std::printf("[residual_semantic] self_test %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace rc
