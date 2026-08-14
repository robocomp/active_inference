/*
 * lidar_select.h — which LiDAR returns belong to THIS instance, and what one look was worth. SHARED.
 *
 * Extracted 2026-08-14 from four copies (cabinet · hood · refrigerator · table) measured at 85.6% pairwise
 * identical lines with the object's name normalised away — the highest family in the fleet after voxel_bank
 * was taken. refrigerator and table were BYTE-IDENTICAL: the normalised diff between them is empty.
 *
 * ★THIS FILE HAS ALREADY PRODUCED TWO LIVE DEFECTS, both in the part that is NOT shared:
 *
 *   1. THE BAND. hood inherited refrigerator's floor-referenced `[−margin, BirthHeightM + margin]`. A hood
 *      hangs, so that band was [−0.10, 0.85] m against a body at [1.79, 2.29] — DISJOINT. 109 returns per
 *      cycle were selected off the floor and the worktop at a mean 1.33 m from the model, every real hood
 *      return was excluded before the factor saw one, and `coverage` still reported 1.0 because 109 rays is
 *      more than LidarCoverageN0 = 60. The fit was silently mask-only.
 *   2. THE COVERAGE WEIGHT reads the RAY COUNT, and a count says nothing about whether those rays could
 *      have resolved the object. Same file, still open — the LiDAR has no counterpart of the camera's
 *      p_detect. Noted here because this is where it would go.
 *
 * So the seam is drawn where the damage was: the agent answers WHERE ITS BODY IS (a radius and a z-band)
 * and WHAT ITS SURFACE IS (an SDF for the residual probe). Everything after those two — the selection loop,
 * the ray-count and angular-coverage down-weights, the "near?" raw count, the z-calibration probe — is
 * identical across the fleet and lives here.
 *
 * Pure: Eigen + the standard library + rc::ai::LidarRays. No DSR, no config type, no instance type.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

#include <Eigen/Dense>

#include "../ai_belief/lidar_ray_factor.h"   // rc::ai::LidarRays

namespace rc::lidar_select
{

// WHERE THE BODY IS — the agent's answer, and the only part of the selection that is per-object.
// ★The z-band must be the BODY's, never the floor's. See the header note: a floor-referenced band on a
// hanging object is disjoint from it, and nothing downstream can tell.
struct Box
{
    float radius_xy = 1.0f;   // rotation-agnostic footprint radius + margin
    float z_lo = 0.0f;        // band low  (already includes the agent's margin)
    float z_hi = 0.0f;        // band high
};

// The two continuous informativeness down-weights on the staged precision. Identical in all four agents.
struct Weights
{
    float coverage_n0        = 60.0f;   // ray count that constitutes a resolving look
    float coverage_ang_power = 1.0f;    // exponent on the angular-spread weight
    float robust_c_m         = 0.10f;   // Huber scale handed to the ray factor
};

// What the probe measured, for the agent's own diagnostics/CSV.
struct Diag
{
    int   rays = 0, raw = 0;
    float cov_ang = 1.0f;
    float resid_m = -1.0f;              // mean |SDF| of the selected returns — the fit's own error
    float meanz = -1.0f, topz = -1.0f, floorz = -1.0f;   // z-calibration probe
};

// Stage the returns inside `box` (centred on `centroid_xy`) into `out`, and apply the two down-weights to
// `precision_base`. Returns the staged ray count.
//
// The angular weight is 1 − R where R is the resultant length of the returns' bearings: rays all arriving
// from one direction see one face and constrain far less than the same number spread around the object.
inline int select_into(const std::vector<Eigen::Vector3f>& sweep, const Eigen::Vector3f& origin,
                       const Eigen::Vector2f& centroid_xy, const Box& box,
                       float precision_base, const Weights& w,
                       rc::ai::LidarRays& out, float* cov_ang = nullptr)
{
    const float r2 = box.radius_xy * box.radius_xy;
    out.endpoints.clear();
    out.endpoints.reserve(256);
    double bear_c = 0.0, bear_s = 0.0;
    for (const auto& p : sweep)
    {
        const float dx = p.x() - centroid_xy.x(), dy = p.y() - centroid_xy.y();
        if (dx * dx + dy * dy > r2 or p.z() < box.z_lo or p.z() > box.z_hi) continue;
        out.endpoints.push_back(p);
        const float phi = std::atan2(dy, dx);
        bear_c += std::cos(phi); bear_s += std::sin(phi);
    }
    const int n = static_cast<int>(out.endpoints.size());
    float ca = 1.0f;
    if (n > 0)
    {
        const float coverage = std::min(1.0f, static_cast<float>(n) / std::max(1.0f, w.coverage_n0));
        const float R        = static_cast<float>(std::hypot(bear_c, bear_s) / n);
        ca = std::pow(std::max(0.0f, 1.0f - R), w.coverage_ang_power);
        out.origin     = origin;
        out.precision  = precision_base * coverage * ca;
        out.robust_c_m = w.robust_c_m;
    }
    if (cov_ang) *cov_ang = ca;
    return n;
}

// A generous "is anything near this object at all?" count at 1.5x the box — a diagnostic, deliberately
// wider than the selection so an empty selection can be told apart from an empty neighbourhood.
inline int raw_count(const std::vector<Eigen::Vector3f>& sweep, const Eigen::Vector2f& c, const Box& box)
{
    const float r2 = (1.5f * box.radius_xy) * (1.5f * box.radius_xy);
    int raw = 0;
    for (const auto& p : sweep)
    {
        const float dx = p.x() - c.x(), dy = p.y() - c.y();
        if (dx * dx + dy * dy <= r2 and p.z() >= box.z_lo and p.z() <= box.z_hi) ++raw;
    }
    return raw;
}

// The fit-quality residual + the z-calibration probe, over the returns already staged in `rays`.
//
// ★resid_m IS THE INSTRUMENT THAT NAMED THE BAND DEFECT. It read 1.33 m on hood while the selection was
// picking up the floor, and 0.07 m once the band was corrected. `sdf` is the agent's own surface — a
// compound SDF for a box with legs, a plain box for a run — which is why it is a callback rather than
// something this file could compute.
//
// The z probe reports the mean of ALL selected returns and of the highest / lowest 20%. A persistent gap
// between topz and the fitted top is a lidar3D→room z-offset, not a fit error — the two are indistinguishable
// from the residual alone, which is why both are reported.
inline void probe(const rc::ai::LidarRays& rays,
                  const std::function<float(const Eigen::Vector3f&)>& sdf, Diag& d)
{
    const std::size_t n = rays.endpoints.size();
    if (n == 0) return;
    double resid_sum = 0.0, zsum = 0.0;
    std::vector<float> zs;
    zs.reserve(n);
    for (const auto& p : rays.endpoints)
    {
        resid_sum += std::abs(sdf(p));
        zs.push_back(p.z());
        zsum += p.z();
    }
    d.resid_m = static_cast<float>(resid_sum / static_cast<double>(n));
    std::sort(zs.begin(), zs.end());
    const std::size_t k = std::max<std::size_t>(1, zs.size() / 5);
    double topsum = 0.0; for (std::size_t i = zs.size() - k; i < zs.size(); ++i) topsum += zs[i];
    double botsum = 0.0; for (std::size_t i = 0; i < k; ++i) botsum += zs[i];
    d.meanz  = static_cast<float>(zsum   / static_cast<double>(n));
    d.topz   = static_cast<float>(topsum / static_cast<double>(k));
    d.floorz = static_cast<float>(botsum / static_cast<double>(k));
}

}  // namespace rc::lidar_select
