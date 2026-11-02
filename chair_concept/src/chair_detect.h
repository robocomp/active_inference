// chair_detect.h -- the chair's P(detect), as a pure function of what the projection measured.
// Header-only, framework-free: the unit test calls it without Qt / DSR.
//
// ★2026-10-31 (perception cards, Task 9): this REPLACES ChairFitter::zed_detectability's private centre ramp x range
// ramp (Existence.ZedEdgeOffset / ZedRangeFull / ZedRangeRef). The shared countable formula now applies:
//     P(detect) = envelope(roi_fill) * in_fov * central * wall_line_of_sight
// - envelope: rc::card::term_envelope (single-fill overload, card axes = 1), ChairModel.DetectMinFill/MaxFill/Soft.
//   It carries BOTH shoulders, so range falls out of the fill (a far chair is few pixels) and a chair that overflows
//   the frame is a blind spot, not absence. The old range ramp is therefore not needed.
// - in_fov: fraction of the projected box inside the image (continuous), central: 1 - normalised ROI offset.
// - wall line of sight: 0 when a room wall crosses camera -> chair (the chair is in another room / round a corner),
//   else 1. That is a PROBABILITY of seeing, not a gate: no image can show through a wall. Same as table.
#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <vector>

#include "../../common/occlusion/occlusion.h"
#include "../../common/perception_card/terms.h"

namespace rc
{
struct ChairDetectInputs
{
    bool  roi_valid = false;
    float roi_fill = 0.0f;                 // max(dcol/W, drow/H), the single-fill axis
    float in_fov = 1.0f;                   // fraction of the projected box inside the image, [0,1]
    float roi_offset_x = 0.0f, roi_offset_y = 0.0f;   // normalised, 0 = centred, 1 = image edge
    Eigen::Vector2f cam_xy = Eigen::Vector2f::Zero(); // camera in the room frame
    Eigen::Vector2f chair_xy = Eigen::Vector2f::Zero();
};

inline float chair_p_detect(const ChairDetectInputs& in, const rc::detect::DetectorEnvelope& env,
                            const std::vector<Eigen::Vector2f>& room_polygon)
{
    if (not in.roi_valid)
        return 0.0f;
    if (rc::occlusion::walls_block(in.cam_xy, in.chair_xy, room_polygon, rc::occlusion::kOwnWallSkipM))
        return 0.0f;
    const float central = std::clamp(1.0f - std::hypot(in.roi_offset_x, in.roi_offset_y), 0.0f, 1.0f);
    return rc::card::term_envelope(in.roi_fill, 1.0f, env)
         * rc::card::term_in_fov(in.in_fov) * rc::card::term_central(central);
}
}  // namespace rc
