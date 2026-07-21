/*
 *  ray_anisotropic_cov.h — single-view measurement covariance for a partial-view object centroid.
 *
 *  A one-shot mask centroid is a BIASED estimate of an object's true centre: you see the near face +
 *  top, so the centroid sits toward the camera ALONG THE VIEWING RAY by ~half the unseen depth. The
 *  bias is unknowable from one view — so we don't fight it, we DECLARE THE RAY DIRECTION UNTRUSTWORTHY:
 *
 *      R = σ_perp²·(I − d dᵀ)  +  σ_along²·(d dᵀ)          σ_along ≫ σ_perp
 *
 *  d = horizontal unit viewing ray (camera→centroid). ⊥-ray is tight (silhouette pins lateral position);
 *  along-ray is loose (near-face/depth ambiguity). Because the bias lies along d, it falls in the
 *  near-null-space of R⁻¹ — a precision-weighted fuser (information filter) uses ONLY the unbiased
 *  ⊥-ray component, and views from diverse azimuths triangulate the true centre. In the σ_along→∞ limit
 *  this degenerates to bearing-only triangulation. See table_concept/TABLE_TRIANGULATION.md.
 *
 *  Header-only, Eigen only. Build d and R in the SAME frame the consumer applies the information in
 *  (for room_concept's object-anchor factor that is the ROBOT/body frame — the residual lives there).
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <Eigen/Dense>

namespace rc::object_anchor
{
    struct RayCovParams
    {
        float sigma_perp     = 0.04f;   // lateral (⊥-ray) noise floor (m)
        float c_bias         = 0.7f;    // fraction of the along-ray box half-shadow → σ_along (near-face bias)
        float depth_alpha    = 0.01f;   // ZED depth noise: σ_depth = alpha·range² (m); folds into σ_along
        float obliquity_floor= 0.15f;   // clamp for the grazing-view 1/obliquity inflation of σ_along
        float sigma_along_min= 0.06f;   // floor so a head-on view still keeps a little depth slack (m)
    };

    /// Anisotropic 2×2 measurement covariance R for a single-view centroid, expressed in the frame of
    /// `d`/`cam`/`centroid`.  `half_ext`=(a,b) and `box_yaw` describe the object box IN THAT SAME FRAME
    /// (so the along-ray box shadow L_ray is exact); pass a conservative (a,b) with box_yaw=0 if the
    /// orientation in this frame is unknown — L_ray then over-estimates slightly (safe: weaker view).
    /// `range` = camera→object depth (m); `obliquity_cos` = |cos(incidence)| of the ray vs the tabletop
    /// normal (1 = top-down, →0 grazing).
    inline Eigen::Matrix2f ray_anisotropic_cov(const Eigen::Vector2f& centroid_xy,
                                               const Eigen::Vector2f& cam_xy,
                                               const Eigen::Vector2f& half_ext,
                                               float box_yaw, float range, float obliquity_cos,
                                               const RayCovParams& p = {})
    {
        Eigen::Vector2f d = centroid_xy - cam_xy;
        const float dn = d.norm();
        const Eigen::Matrix2f I = Eigen::Matrix2f::Identity();
        if (dn < 1e-6f)                                    // degenerate: fall back to isotropic floor
            return p.sigma_perp * p.sigma_perp * I;
        d /= dn;

        // Box half-shadow on the ray: L_ray = a|cos φ| + b|sin φ|,  φ = ∠(d, box local x-axis).
        const float c = std::cos(box_yaw), s = std::sin(box_yaw);
        const float dx =  c * d.x() + s * d.y();           // d in box-local frame
        const float dy = -s * d.x() + c * d.y();
        const float L_ray = std::abs(half_ext.x()) * std::abs(dx)
                          + std::abs(half_ext.y()) * std::abs(dy);

        // σ_along: unseen near-face depth (∝ L_ray) + sensor depth noise, inflated for grazing views.
        const float obl = std::max(p.obliquity_floor, obliquity_cos);
        const float sig_along = std::max(p.sigma_along_min,
                                         (p.c_bias * L_ray + p.depth_alpha * range * range) / obl);
        const float sig_perp  = p.sigma_perp;

        const Eigen::Matrix2f ddT = d * d.transpose();
        return sig_perp * sig_perp * (I - ddT) + sig_along * sig_along * ddT;   // PSD, elongated along d
    }
} // namespace rc::object_anchor
