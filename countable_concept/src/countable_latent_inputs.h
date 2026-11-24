/*
 * countable_latent_inputs.h — the persistent path's per-frame latent inputs (spec 2026-11-15 lidar-measures §7c.2;
 * shared-engine G1, Task 5 ruling). Pure: no DSR, no Qt; the fitter calls it, latent_inputs_test drives it.
 *
 *   sig2_cam = the MEASURED camera detector's boundary error squared on the position and size DOFs (cx, cy, H, w, h) —
 *              kZedYoloBoundaryStdM below (plan rev 2 F1.6) — and the configured AI2CommonModeYawStd² on yaw (no measured
 *              row: a boundary grown or shrunk uniformly carries no rotation, so the measurement says nothing about yaw);
 *   sig2_L   = the constant metric common-mode stds squared (AI2CommonMode{Pos,Size,Yaw}Std), per DOF: the LiDAR is not the
 *              camera's detector, and nothing measured its shared error yet.
 *   Neither has the range growth or the obliquity yaw term, nor the pose chain: the chain's share is the latent e, whose
 *   marginal Σ_pose stage_persistent stages separately — adding it here as well would count it twice.
 *   fresh             = the ego-motion variances (MotionCm* · motion_magnitude · periphery), re-drawn every frame.
 *
 * The frame's view (range, obliquity, chain variance) is passed in on purpose: it is what the fitter holds when it stages
 * the latents, and latent_inputs_test checks that none of it leaks into the marginals.
 */
#pragma once

#include <array>

namespace rc {

struct LatentInputs
{
    std::array<float, 6> sig2_cam{}, sig2_L{}, fresh{};
};

struct CommonModeStds { float pos = 0.0f, size = 0.0f, yaw = 0.0f; };   // AI2CommonMode{Pos,Size,Yaw}Std

// The ZED YOLO-seg detector's boundary error, in metres at the object, constant in range (plan rev 2 F1.6; the T5 ruling that
// the camera latent is metric). Source: docs/superpowers/results/2026-11-24-step0.md, Step 4, the refrigerator row (line 130:
// 66 frames, sigma_px_boundary 4.51 px — "the cleanest box ... if one value is taken as the sensor's boundary error, it should be
// this one", line 138). The same statistic in metres: the std over those 66 frames of the per-frame mean signed boundary offset
// × the frame's range / f (f = 448.1 px), re-run with tools/mask_error/measure_mask_error.py --classes refrigerator --out on
// objects_step0/zed: 0.0289 m (mean −0.0031 m; mean |offset| 0.0131 m, the step-0 range-bin table's fridge rows). An upper bound:
// the residual layer mismatch of the fridge box is inside it. A shared offset of every boundary pixel moves the face it lies on,
// so it is the shared error of the centre (one face seen) and of an extent (its two faces), hence on cx, cy, H, w, h.
inline constexpr float kZedYoloBoundaryStdM = 0.0289f;

// What the fitter knows about the frame when it stages the persistent latents.
struct LatentView
{
    float range_m = 0.0f;          // camera → mask depth
    float obliquity_cos = 1.0f;    // |cos| of the camera → top ray's incidence
    float chain_var_xx = 0.0f;     // the pose chain's position variance at the object (carried by e, not here)
    float chain_var_yy = 0.0f;
    float mot_pos_var = 0.0f, mot_size_var = 0.0f, mot_yaw_var = 0.0f;   // ego-motion (fresh) variances
};

inline LatentInputs persistent_latent_inputs(const CommonModeStds& cm, const LatentView& v)
{
    const float p2 = cm.pos * cm.pos, s2 = cm.size * cm.size, y2 = cm.yaw * cm.yaw;
    const float b2 = kZedYoloBoundaryStdM * kZedYoloBoundaryStdM;
    LatentInputs li;
    li.sig2_cam = {b2, b2, b2, b2, b2, y2};
    li.sig2_L   = {p2, p2, s2, s2, s2, y2};
    li.fresh    = {v.mot_pos_var, v.mot_pos_var, v.mot_size_var, v.mot_size_var, v.mot_size_var, v.mot_yaw_var};
    return li;
}

}  // namespace rc
