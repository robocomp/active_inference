/*
 * countable_latent_inputs.h — the persistent path's per-frame latent inputs (spec 2026-11-15 lidar-measures §7c.2;
 * shared-engine G1, Task 5 ruling). Pure: no DSR, no Qt; the fitter calls it, latent_inputs_test drives it.
 *
 *   sig2_cam = sig2_L = the constant metric common-mode stds squared (AI2CommonMode{Pos,Size,Yaw}Std), per DOF
 *                       (cx, cy, H, w, h, yaw). NOT the range growth, NOT the obliquity yaw term (step 0: the mask boundary
 *                       error is shape-dominated, neither pixel- nor metre-constant, so no measured sensor row replaces
 *                       them), and NOT the pose chain: the chain's share is the latent e, whose marginal Σ_pose
 *                       stage_persistent stages separately — adding it here as well would count it twice.
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
    LatentInputs li;
    li.sig2_cam = {p2, p2, s2, s2, s2, y2};
    li.sig2_L   = li.sig2_cam;
    li.fresh    = {v.mot_pos_var, v.mot_pos_var, v.mot_size_var, v.mot_size_var, v.mot_size_var, v.mot_yaw_var};
    return li;
}

}  // namespace rc
