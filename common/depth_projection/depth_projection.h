#pragma once

/*
 * depth_projection.h — reproject a point cloud into ANY DSR camera (via the shared CameraAPI) and
 * score the depth information a mask receives from it. Header-only.
 *
 * Motivation: the ricoh 360 masks used to be published bearing-only (no depth). With a lidar cloud
 * reprojected into the panorama (CameraAPI equirectangular model), those masks can carry depth — and a
 * per-mask RANGE VARIANCE that the consumer ADDS to its R, in the SAME currency as the interaction-
 * matrix motion variance (common/motion_corruption): R = R_base + motion_var + depth_var. So it
 * composes, and degrades gracefully with no extra policy — sparse/scattered lidar ⇒ large range_var ⇒
 * the range term vanishes and the mask falls back to bearing/silhouette. NOT a hard has-depth switch.
 *
 * Mirrors motion_corruption's contract: this owns the (hits → range, variance) math, stateless/pure;
 * the CONSUMER adds range_var into its innovation covariance. The camera model (pinhole / equirect)
 * lives entirely in CameraAPI, so this works for the ricoh AND the zed with the same code. The caller
 * supplies camera_T_source and associates points to masks.
 */

#include <dsr/api/dsr_camera_api.h>
#include <dsr/api/dsr_eigen_defs.h>   // Mat::RTMat

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

namespace rc::depth
{

// One cloud point reprojected into the camera image.
struct ProjectedPoint
{
    float           u = 0.f, v = 0.f;                    // image pixel
    float           range = 0.f;                         // ‖p_cam‖ (spherical depth; valid for pinhole too)
    Eigen::Vector3f xyz_cam = Eigen::Vector3f::Zero();   // point in the camera frame
};

// Reproject `cloud_source` (SOURCE frame) into `cam`. `camera_T_source` maps a source point into the
// camera frame. Keeps points in front of the camera (range ≥ min_range) that project to finite pixels.
// No z-buffer — the caller associates points to mask regions and scores them there.
inline std::vector<ProjectedPoint> reproject_cloud(std::span<const Eigen::Vector3f> cloud_source,
                                                   const DSR::CameraAPI& cam,
                                                   const Mat::RTMat& camera_T_source,
                                                   float min_range = 0.05f)
{
    std::vector<ProjectedPoint> out;
    out.reserve(cloud_source.size());
    const Eigen::Matrix3d R = camera_T_source.linear();
    const Eigen::Vector3d t = camera_T_source.translation();
    for (const auto& ps : cloud_source)
    {
        const Eigen::Vector3d pc = R * ps.cast<double>() + t;
        const double r = pc.norm();
        if (r < static_cast<double>(min_range))
            continue;
        const Eigen::Vector2d uv = cam.project(pc);
        if (not std::isfinite(uv.x()) or not std::isfinite(uv.y()))
            continue;
        out.push_back({static_cast<float>(uv.x()), static_cast<float>(uv.y()),
                       static_cast<float>(r), pc.cast<float>()});
    }
    return out;
}

// Depth estimate + a RANGE VARIANCE for a mask, from the reprojected points inside it (the caller
// filters by the mask's bbox/polygon). range_var (m²) is the term the consumer ADDS to R along the
// mask's bearing ray — same "variance to add to R" contract as motion_corruption's var_m. Small when
// many range-consistent hits; large (→ bearing-only) when sparse or scattered.
struct MaskDepth
{
    bool            has_depth = false;
    int             n_hits    = 0;
    float           range     = 0.f;                       // robust central range (median of hits, m)
    float           spread    = 0.f;                       // MAD of ranges (m) — depth scatter within the mask
    Eigen::Vector3f xyz_cam   = Eigen::Vector3f::Zero();   // representative 3D point (camera frame)
    float           range_var = 0.f;                       // σ_range² (m²) to ADD to the consumer's R
    float           quality   = 0.f;                       // convenience ∈(0,1] = floor_var / range_var
};

struct MaskDepthParams
{
    float floor_sigma_m  = 0.03f;   // irreducible per-hit range noise (m) → floor variance, even for a perfect mask
    float coverage_n0    = 5.0f;    // hit-count scale for the coverage penalty (few hits ⇒ inflate range_var)
    float sparse_sigma_m = 2.0f;    // range σ a single-hit mask is penalised toward (≈ bearing-only)
};

// Score a mask given ONLY the reprojected points inside it. Variance model (terms in m²):
//   range_var = floor²  +  (1.4826·MAD)² / n_hits  +  sparse²·exp(-n_hits/n0)
//               └ sensor    └ scatter, averaged      └ low-coverage penalty (→ 0 as hits grow)
inline MaskDepth score_mask_depth(std::span<const ProjectedPoint> hits, const MaskDepthParams& p = {})
{
    MaskDepth md;
    md.n_hits = static_cast<int>(hits.size());
    if (hits.empty())
        return md;   // has_depth=false, quality=0 → graceful bearing-only downstream

    // Robust central range (median) + spread (median absolute deviation), immune to the mask spanning
    // a foreground/background depth step (a few outlier ranges won't drag the estimate).
    std::vector<float> ranges;
    ranges.reserve(hits.size());
    for (const auto& h : hits) ranges.push_back(h.range);
    std::nth_element(ranges.begin(), ranges.begin() + ranges.size() / 2, ranges.end());
    const float med = ranges[ranges.size() / 2];
    std::vector<float> dev;
    dev.reserve(hits.size());
    for (float r : ranges) dev.push_back(std::abs(r - med));
    std::nth_element(dev.begin(), dev.begin() + dev.size() / 2, dev.end());
    const float mad = dev[dev.size() / 2];

    // Representative 3D point = the hit closest to the median range (avoids picking an outlier).
    const ProjectedPoint* rep = &hits.front();
    float best = std::abs(hits.front().range - med);
    for (const auto& h : hits)
    {
        const float d = std::abs(h.range - med);
        if (d < best) { best = d; rep = &h; }
    }

    const float n         = static_cast<float>(hits.size());
    const float floor_var = p.floor_sigma_m * p.floor_sigma_m;
    const float sigma     = 1.4826f * mad;                                        // robust σ from MAD
    const float scatter   = (sigma * sigma) / n;                                  // estimate shrinks with more hits
    const float sparse    = p.sparse_sigma_m * p.sparse_sigma_m
                            * std::exp(-n / std::max(1e-3f, p.coverage_n0));       // low-coverage penalty

    md.has_depth = true;
    md.range     = med;
    md.spread    = mad;
    md.xyz_cam   = rep->xyz_cam;
    md.range_var = floor_var + scatter + sparse;
    md.quality   = floor_var / md.range_var;   // 1 ⇒ trust depth fully; →0 ⇒ effectively bearing-only
    return md;
}

}  // namespace rc::depth
