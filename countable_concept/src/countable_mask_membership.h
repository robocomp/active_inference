/*
 * countable_mask_membership.h — how surely a LiDAR return belongs to the camera's mask, m = pi_mask · Phi(s/sigma_px)
 * (spec 2026-11-15 lidar-measures G2). s is the signed pixel distance to the mask boundary (positive inside); sigma_px
 * is the projection uncertainty of the return in pixels, which GROWS as the range shrinks (a continuous covariance,
 * not a gate). OpenCV is confined to build(); the object is plain vectors afterwards.
 */
#pragma once

#include <cmath>
#include <functional>
#include <optional>
#include <span>
#include <vector>

#include <Eigen/Dense>

#include "countable_depth_order.h"

namespace rc {

class MaskMembership
{
public:
    // pixels in the DELIVERED image (col, row).
    static MaskMembership build(std::span<const Eigen::Vector2f> pixels);

    float stride_px() const { return q_; }   // median nearest distinct column step within a row; 1 if dense/empty
    bool  empty() const { return empty_; }
    // Positive inside; a mask pixel reads +q/2, a pixel just outside −q/2; bilinear. Outside the raster:
    // s(nearest border cell) − distance to it. An empty mask reads −infinity-like (very negative) distance 0 ⇒ -1e9.
    float signed_distance_px(float col, float row) const;

private:
    bool  empty_ = true;
    float q_ = 1.f;
    float c0_ = 0.f, r0_ = 0.f;    // pixel coordinates of raster cell (1,1) (the slice minimum)
    int   w_ = 0, h_ = 0;          // raster size including the one-cell border
    std::vector<float> s_;         // signed distance in px at every raster cell, row-major
};

inline float projection_sigma_px(float f_px, float z_m, float sigma_range_m, float sigma_chain_m,
                                 float mot_pos_var_m2, float q_px)
{
    const float k = f_px / z_m;
    return std::sqrt(k * k * (sigma_range_m * sigma_range_m + sigma_chain_m * sigma_chain_m + mot_pos_var_m2)
                     + q_px * q_px);
}

inline float membership_weight(float s_px, float sigma_px)
{
    return 0.5f * std::erfc(-s_px / (sigma_px * 1.41421356237f));
}

// (col, row, depth) in the delivered image, or nullopt (behind / outside the image / unavailable).
using Projector = std::function<std::optional<Eigen::Vector3f>(const Eigen::Vector3f&)>;

// nullopt from the projector ⇒ 0; else pi_mask · membership_weight(s, projection_sigma_px(...)). Captures mm by value.
// pi_mask is a detector confidence in [0,1]; it is NOT clamped here — callers pass a valid confidence.
std::function<float(const Eigen::Vector3f&)> make_membership(Projector proj, const MaskMembership& mm, float f_px,
                                                             float sigma_range_m, float sigma_chain_m,
                                                             float mot_pos_var_m2, float pi_mask);

// The components behind make_membership's value, for the label-error measurement (spec 7c.8 a0). Same inputs, same
// arithmetic; q equals make_membership's result. in_image == false ⇒ the projector returned nothing (s, sigma, q = 0).
struct MembershipComponents
{
    bool  in_image = false;
    float s_px = 0.f, sigma_px = 0.f, pi_mask = 0.f, q = 0.f;
};
std::function<MembershipComponents(const Eigen::Vector3f&)> make_membership_components(
    Projector proj, const MaskMembership& mm, float f_px, float sigma_range_m, float sigma_chain_m,
    float mot_pos_var_m2, float pi_mask);

// The persistent label of a LiDAR return (spec 7c.8 a0): q = pi_mask * Phi(s/sigma_px) is today's membership, and
// m = eps_px + (1 - 2 eps_px) q keeps it inside [eps_px, 1 - eps_px] so that neither a certain member nor a certain
// non-member exists (the camera's mask edge is wrong by a measured pixel-error rate eps_px). nullopt = no camera evidence
// (the projector returned nothing: behind / outside the image).
struct PersistentLabel
{
    double m = 0.0;
    float  q = 0.f;
};
// The persistent label with the depth ordering: the camera's label q applies to the return only as far as the return is NOT
// behind the object along the camera ray, q_eff = q·(1 − P(behind)), m = ε + (1 − 2ε)·q_eff. An empty p_behind is the plain
// make_persistent_label.
std::function<std::optional<PersistentLabel>(const Eigen::Vector3f&)> make_persistent_label(
    Projector proj, const MaskMembership& mm, float f_px, float sigma_range_m, float sigma_chain_m,
    float mot_pos_var_m2, float pi_mask, double eps_px,
    std::function<double(const Eigen::Vector3f&)> p_behind = {});

}  // namespace rc
