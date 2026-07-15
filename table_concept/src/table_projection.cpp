/*
 * table_projection.cpp — camera-projection unit for table_concept (extracted from TableFitter).
 *
 * Implements the camera→room extrinsic (room_T_zed, room→body pinned to a capture stamp), the model-box →
 * normalised in-image ROI used by the controller's lock-on, and the PIXEL-LEVEL silhouette existence evidence
 * (project the tabletop top face + leg axes, then over the predicted-visible pixels vote occupancy / absence /
 * occlusion against the current YOLO foreground). Owns the ZED CameraAPI (lazily bound to the "zed" node) and
 * uses the shared graph, InnerEigenAPI, and MaskIngestor packet source.
 */

#include "table_projection.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace rc {

// ─── Camera extrinsic (room_T_zed) ──────────────────────────────────────────────────────────────

// room_T_zed (camera→room): room→body pinned to pose_ts_ms (Nearest), the rigid body→zed mount at latest.
std::optional<Eigen::Matrix4d> TableProjection::room_T_zed_matrix(std::uint64_t pose_ts_ms) const
{
    if (not inner_eigen_)
        return std::nullopt;
    // Pin the moving room→body hop to the frame's capture stamp (Nearest); keep the rigid body→zed mount
    // at latest (it carries only a bootstrap stamp — a pinned query would fail). ts=0 → current pose.
    const auto rtb = inner_eigen_->get_transformation_matrix("room", "body", pose_ts_ms);
    const auto btz = inner_eigen_->get_transformation_matrix("body", "zed", 0);
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

// ─── Projected ROI ──────────────────────────────────────────────────────────────────────────────

// Project the model box through the camera extrinsic → normalised in-image ROI (centre offset + fill).
//
// Stored on the instance for the controller's centring/dwell lock-on. Degenerate projections (robot too
// close / a corner grazing the image plane → the bbox explodes) are marked invalid and clamped so consumers
// and logs never see garbage.
void TableProjection::compute_projected_roi(TableInstance& inst)
{
    inst.roi_valid = false;
    if (not inner_eigen_)
        return;
    if (not camera_api_)
    {
        const auto zed = G_->get_node("zed");
        if (not zed.has_value()) return;
        camera_api_ = G_->get_camera_api(zed.value());
        if (not camera_api_) return;
    }

    const auto Mopt = room_T_zed_matrix();   // room_T_zed (camera→room)
    if (not Mopt.has_value())
        return;
    const Eigen::Matrix4d zed_T_room = Mopt.value().inverse();   // room point → camera frame

    const float fx = camera_api_->get_focal_x();
    const float fy = camera_api_->get_focal_y();
    const float W  = static_cast<float>(camera_api_->get_width());
    const float H  = static_cast<float>(camera_api_->get_height());
    if (fx <= 0.f || fy <= 0.f || W <= 0.f || H <= 0.f)
        return;
    const float cx_px = W * 0.5f, cy_px = H * 0.5f;

    // Project the 8 box corners (top slab + footprint at floor) into the image. Camera convention
    // matches the producer: X=right, Y=forward(depth), Z=up ⇒ col=cx+X/Y·fx, row=cy−Z/Y·fy.
    const auto& s = inst.model.state();
    const float c = std::cos(s.yaw), sn = std::sin(s.yaw);
    const float hw = s.w * 0.5f, hh = s.h * 0.5f;
    float min_col = 1e9f, min_row = 1e9f, max_col = -1e9f, max_row = -1e9f;
    int in_front = 0;
    for (const int ix : {-1, 1})
        for (const int iy : {-1, 1})
            for (const float z : {0.0f, s.table_height})
            {
                const float lx = static_cast<float>(ix) * hw, ly = static_cast<float>(iy) * hh;
                const Eigen::Vector4d Pr(s.cx + c * lx - sn * ly, s.cy + sn * lx + c * ly, z, 1.0);
                const Eigen::Vector4d Pc = zed_T_room * Pr;
                const double X = Pc.x(), Y = Pc.y(), Z = Pc.z();
                if (Y <= 0.20) continue;   // skip corners at/near the image plane: X/Y explodes there
                ++in_front;
                const float col = cx_px + static_cast<float>(X / Y) * fx;
                const float row = cy_px - static_cast<float>(Z / Y) * fy;
                min_col = std::min(min_col, col); max_col = std::max(max_col, col);
                min_row = std::min(min_row, row); max_row = std::max(max_row, row);
            }

    if (in_front < 4)   // need most of the box in front of the camera to trust the ROI
        return;

    const float roi_cx = 0.5f * (min_col + max_col);
    const float roi_cy = 0.5f * (min_row + max_row);
    const float off_x = (roi_cx - cx_px) / (0.5f * W);   // [-1,1], 0 = centred
    const float off_y = (roi_cy - cy_px) / (0.5f * H);
    const float fill  = std::max((max_col - min_col) / W, (max_row - min_row) / H);
    // Reject degenerate projections (robot too close / a corner grazing the image plane → the
    // bbox explodes to absurd offsets). Beyond a sane bound the ROI is unusable for centring:
    // mark invalid (the controller then keeps sweeping / treats framing as unknown) and clamp the
    // stored values so consumers/logs never see garbage.
    const bool sane = std::isfinite(off_x) && std::isfinite(off_y) && std::isfinite(fill)
                      && std::abs(off_x) < 3.0f && std::abs(off_y) < 3.0f && fill < 4.0f;
    inst.roi_offset_x = std::clamp(off_x, -3.0f, 3.0f);
    inst.roi_offset_y = std::clamp(off_y, -3.0f, 3.0f);
    inst.roi_fill     = std::clamp(fill, 0.0f, 4.0f);
    inst.roi_valid    = sane;
}

// ─── Silhouette existence (pixel-level) ─────────────────────────────────────────────────────────

// PIXEL-LEVEL silhouette existence evidence (occupancy / absence / occlusion) — see the header / SilhouetteExistence.
//
// Projects the tabletop top face + the 4 leg axes onto the image and, over the predicted-visible pixels,
// splits into: lit by a "table" mask (occupancy), lit by a non-table mask (occlusion → HOLD), or lit by
// nothing (absence — the KEY "gone" signal that fires even when NO YOLO mask does this frame).
SilhouetteExistence TableProjection::compute_silhouette_existence(const TableInstance& inst)
{
    SilhouetteExistence out;
    if (not inner_eigen_)
        return out;
    if (not camera_api_)
    {
        const auto zed = G_->get_node("zed");
        if (not zed.has_value()) return out;
        camera_api_ = G_->get_camera_api(zed.value());
        if (not camera_api_) return out;
    }
    const auto Mopt = room_T_zed_matrix();   // room→zed pinned to current pose
    if (not Mopt.has_value())
        return out;
    const Eigen::Matrix4d zed_T_room = Mopt.value().inverse();

    const float fx = camera_api_->get_focal_x(), fy = camera_api_->get_focal_y();
    const float W  = static_cast<float>(camera_api_->get_width());
    const float Himg = static_cast<float>(camera_api_->get_height());
    if (fx <= 0.f or fy <= 0.f or W <= 0.f or Himg <= 0.f)   // fx/fy: sanity only; projection uses CameraAPI
        return out;

    // Hashed pixel-cell coverage of the current YOLO foreground, split table (occupancy) vs other (occluder).
    // A CELL-px cell absorbs mask-boundary jitter and makes membership O(1). Key packs the two cell indices.
    const auto& pkt = mask_ingestor_->packet();
    if (not pkt.valid or pkt.mask_pixels.empty())
        return out;
    constexpr float CELL = 6.0f;
    const auto key = [&](float col, float row) -> std::int64_t
    {
        const std::int64_t cx = static_cast<std::int64_t>(std::floor(col / CELL));
        const std::int64_t cy = static_cast<std::int64_t>(std::floor(row / CELL));
        return (cx << 32) ^ (cy & 0xffffffffLL);
    };
    std::unordered_set<std::int64_t> table_cells, occluder_cells;
    for (const auto& sl : pkt.slices)
    {
        const std::size_t b = std::min(sl.pixel_begin, pkt.mask_pixels.size());
        const std::size_t e = std::min(sl.pixel_end,   pkt.mask_pixels.size());
        auto& dst = (sl.label == "table") ? table_cells : occluder_cells;
        for (std::size_t i = b; i < e; ++i)
            dst.insert(key(pkt.mask_pixels[i].x(), pkt.mask_pixels[i].y()));
    }

    const auto& s = inst.ai2_belief.state();
    const float c = std::cos(s.yaw), sn = std::sin(s.yaw), hw = 0.5f * s.w, hd = 0.5f * s.h;

    // Classify ONE room-frame silhouette sample: project it, then vote occupancy / absence / occlusion. A
    // sample lit by a "table" mask is occupancy; lit by nothing is ABSENCE (the "gone" signal that fires even
    // with NO YOLO mask); lit by a non-table mask is OCCLUDED (a nearer object hides the point) ⇒ HOLD.
    double range_sum = 0.0;
    const auto classify = [&](float lx, float ly, float lz)
    {
        ++out.n_total;                                                 // one silhouette sample of the WHOLE object
        const Eigen::Vector4d Pr(s.cx + c * lx - sn * ly, s.cy + sn * lx + c * ly, lz, 1.0);
        const Eigen::Vector4d Pc = zed_T_room * Pr;
        const double X = Pc.x(), Y = Pc.y(), Z = Pc.z();
        if (Y <= 0.20) return;                                         // behind / at the image plane (near clip)
        // Real camera FRUSTUM: project through the DSR CameraAPI (true intrinsics/principal point, and the
        // node's projection model — pinhole for zed, equirectangular for a 360 sensor). A sample is "expected
        // visible" only if it lands inside the actual image, so out-of-FoV samples are NOT counted detectable.
        const Eigen::Vector2d uv = camera_api_->project(Eigen::Vector3d(Pc.x(), Pc.y(), Pc.z()));
        const float col = static_cast<float>(uv.x()), row = static_cast<float>(uv.y());
        if (col < 0.f or col >= W or row < 0.f or row >= Himg) return; // out of frustum ⇒ not detectable
        const std::int64_t k = key(col, row);
        if (occluder_cells.contains(k) and not table_cells.contains(k))
        { ++out.n_occluded; return; }                                 // nearer object hides it ⇒ HOLD
        ++out.n_detectable;
        if (col > 0.25f * W and col < 0.75f * W and row > 0.25f * Himg and row < 0.75f * Himg)
            ++out.n_central;                                          // central image region ⇒ the robot is looking AT it
        range_sum += std::sqrt(X * X + Y * Y + Z * Z);                 // camera→sample distance (absence conf ∝ 1/range)
        if (table_cells.contains(k)) out.e_occ  += 1.0f;              // still there
        else                         out.e_free += 1.0f;              // predicted-but-absent
    };

    // (a) tabletop TOP face (z = H): regular grid over the footprint.
    constexpr int NX = 24, NY = 24;
    for (int ix = 0; ix < NX; ++ix)
        for (int iy = 0; iy < NY; ++iy)
            classify((-1.0f + 2.0f * (ix + 0.5f) / NX) * hw, (-1.0f + 2.0f * (iy + 0.5f) / NY) * hd, s.H);

    // (b) the 4 LEGS (vertical axes at the inset corners, floor→join): valuable extra evidence, especially from
    // edge-on/low views where the top face projects to a thin sliver — the legs project BELOW the tabletop and
    // extend the detectable footprint (occupancy when masked, absence when the volume is empty).
    const float leg_top = std::max(0.0f, s.H - rc::TableModel::TOP_THICKNESS);
    const float inset    = rc::TableModel::LEG_RADIUS;
    constexpr int NZ = 16;
    for (const float sx : {-1.0f, 1.0f})
        for (const float sy : {-1.0f, 1.0f})
        {
            const float lx = sx * std::max(0.0f, hw - inset), ly = sy * std::max(0.0f, hd - inset);
            for (int iz = 0; iz < NZ; ++iz)
                classify(lx, ly, leg_top * (iz + 0.5f) / NZ);
        }
    if (out.n_detectable > 0)
        out.mean_range_m = static_cast<float>(range_sum / out.n_detectable);
    return out;
}

}  // namespace rc
