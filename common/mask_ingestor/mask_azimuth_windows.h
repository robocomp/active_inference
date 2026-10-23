#pragma once
// Which azimuth ranges the 360 detector actually ran on a frame ("masks_ricoh" / mask_azimuth_windows).
//
// The retina runs YOLO on the panorama as K vertical strips in rotation, so on any one frame most bearings were
// NOT looked at. That is "no evidence", not "no door" -- a consumer that reads it as absence biases its
// detectability low. The producer publishes the looked-at intervals as a flat [from,to] x K list of
// CAMERA-frame azimuths (rad, CCW, each end wrapped to (-pi,pi]; an interval may straddle the +-pi seam).
// A full-panorama frame is sent as >= 3 adjacent windows: a single 2*pi window would be read as width 0.
// ABSENT or EMPTY = UNKNOWN (a producer predating the attribute): consumers must treat every bearing as looked at.
//
// STDLIB + Eigen only, so the unit test builds without DSR.

#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <vector>

namespace rc::masks
{

// Express camera-frame windows in the target (room) frame. The two ends of each window are rotated EXACTLY as
// MaskIngestor rotates a slice's azimuth (unit ray through R_tgt_src, then atan2), so a window and the bearing it
// is meant to contain can never disagree. A trailing odd float is dropped. Order (from, to) is preserved:
// a proper rotation keeps the CCW sense of an interval narrower than pi.
[[nodiscard]] inline std::vector<Eigen::Vector2f> rotate_azimuth_windows(const std::vector<float>& flat,
                                                                         const Eigen::Matrix3d& R_tgt_src,
                                                                         bool needs_transform)
{
    const auto rot = [&](float az_cam) -> float
    {
        if (not needs_transform)
            return az_cam;
        const Eigen::Vector3d ray = R_tgt_src * Eigen::Vector3d(std::cos(az_cam), std::sin(az_cam), 0.0);
        return static_cast<float>(std::atan2(ray.y(), ray.x()));
    };
    std::vector<Eigen::Vector2f> out;
    out.reserve(flat.size() / 2);
    for (std::size_t i = 0; i + 1 < flat.size(); i += 2)
        out.emplace_back(rot(flat[i]), rot(flat[i + 1]));
    return out;
}

} // namespace rc::masks
