// door_leaf_birth.h — a leaf seen side-on points to an opening in its FRAME wall (one-model, spec §3.6). Header-only, DSR-free.
//
// A mask about door-high and a few centimetres deep is a leaf seen edge-on far more often than a door frame. Its plan view is a
// segment (end_a, end_b). For each end taken as the HINGE the frame wall is the wall that end touches under the wall-band prior; the
// aperture runs along that wall from the hinge, on either side, as wide as the leaf is long (under the typical-size prior); the
// leaf's angle to the wall is phi and the side of the wall it stands on is the swing. Each (end, side) is one hypothesis with its
// log weight (band + size + base rate); the caller adds the LiDAR opening (door_opening_edges.h opening_llr) that scores it.
//
// ★LEAF EDGE-ON VS APERTURE (Task 9 ruling 5): the mask's own plan extent decides which of the two a mask is, without a cutoff.
// A shut door's face runs ALONG its wall: the aperture mode predicts the segment's direction at angle 0 to the wall with the
// direction error of a segment of that length, sigma_theta = atan(sqrt(2) * sigma_end / length), so its angular likelihood (peak
// normalised to 1, as every term of place_prior.h) is exp(-theta^2 / 2 sigma_theta^2). An open leaf is anything that is not shut:
// its angular likelihood is the complement, 1 - exp(-theta^2 / 2 sigma_theta^2). The two sum to one at every angle: a mask flush
// with the wall belongs to the aperture and a perpendicular one to the leaf, and in between they share it continuously.
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <span>
#include <vector>

#include <Eigen/Dense>

#include "core/place_prior.h"   // SizePrior, wall_groups, wall_base_rate (WallRates)
#include "door_geometry.h"
#include "door_room_band.h"

namespace rc::door
{

struct LeafApertureHyp
{
    Aperture aperture;
    LeafState leaf;
    int wall = -1;          // the frame wall (wall_groups id)
    double log_w = 0;       // band (hinge end) + size (leaf length) + base rate (the wall's first-door rate); the caller adds opening_llr
};

/// The plan-view segment of a mask: the extremes of its points along the principal axis, its door height, and the spread across.
struct MaskSegment
{
    Eigen::Vector2f a = Eigen::Vector2f::Zero(), b = Eigen::Vector2f::Zero();
    float height = 0.f;     // z extent of the points
    float length = 0.f;     // |b - a|
    float across_sd = 0.f;  // sd of the points across the principal axis (the mask's depth)
    bool  ok = false;
};

[[nodiscard]] inline MaskSegment plan_segment(std::span<const Eigen::Vector3f> pts)
{
    MaskSegment s;
    if (pts.size() < 2) return s;
    Eigen::Vector2d mean = Eigen::Vector2d::Zero();
    float zlo = std::numeric_limits<float>::max(), zhi = std::numeric_limits<float>::lowest();
    for (const auto& p : pts) { mean += p.head<2>().cast<double>(); zlo = std::min(zlo, p.z()); zhi = std::max(zhi, p.z()); }
    mean /= static_cast<double>(pts.size());
    Eigen::Matrix2d C = Eigen::Matrix2d::Zero();
    for (const auto& p : pts) { const Eigen::Vector2d d = p.head<2>().cast<double>() - mean; C += d * d.transpose(); }
    C /= static_cast<double>(pts.size());
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(C);
    const Eigen::Vector2d e = es.eigenvectors().col(1), nrm(-e.y(), e.x());   // the larger eigenvalue's axis
    double lo = std::numeric_limits<double>::max(), hi = std::numeric_limits<double>::lowest();
    for (const auto& p : pts) { const double t = e.dot(p.head<2>().cast<double>() - mean); lo = std::min(lo, t); hi = std::max(hi, t); }
    s.a = (mean + lo * e).cast<float>(); s.b = (mean + hi * e).cast<float>();
    s.height = zhi - zlo; s.length = static_cast<float>(hi - lo);
    s.across_sd = static_cast<float>(std::sqrt(std::max(0.0, nrm.dot(C * nrm))));
    s.ok = s.length > 0.f;
    return s;
}

/// The direction error (rad) of a segment of `length` whose two ends are each known to sigma_end: atan(sqrt(2) sigma_end / length).
[[nodiscard]] inline float segment_angle_sd(float length, float sigma_end)
{
    return std::atan2(std::numbers::sqrt2_v<float> * std::max(sigma_end, 1e-3f), std::max(length, 1e-3f));   // 1e-3: a numerical floor on both
}

/// The aperture mode's angular likelihood of a mask at angle theta (rad, 0..pi/2) to its wall: 1 flush, -> 0 perpendicular.
[[nodiscard]] inline double aperture_angle_likelihood(float theta, float sigma_theta)
{
    const double u = double(theta) / double(std::max(sigma_theta, 1e-3f));
    return std::exp(-0.5 * u * u);
}

namespace detail
{
/// The polygon edge nearest to q and the distance to it; -1 for fewer than 3 vertices.
[[nodiscard]] inline int nearest_edge(std::span<const Eigen::Vector2f> poly, const Eigen::Vector2f& q, float* dist = nullptr)
{
    const std::size_t n = poly.size();
    if (n < 3) return -1;
    int best = -1; float bd = std::numeric_limits<float>::max();
    for (std::size_t i = 0; i < n; ++i)
    {
        const Eigen::Vector2f a = poly[i], ab = poly[(i + 1) % n] - a;
        const float l2 = ab.squaredNorm(), t = l2 > 1e-12f ? std::clamp((q - a).dot(ab) / l2, 0.f, 1.f) : 0.f;
        if (const float d = (q - (a + t * ab)).norm(); d < bd) { bd = d; best = static_cast<int>(i); }
    }
    if (dist) *dist = bd;
    return best;
}
}   // namespace detail

/// From a side-on leaf mask (its two vertical edges in the room frame and its height): for each end as the hinge, the frame wall is
/// the wall (wall_groups) that end touches under the band prior; the aperture runs along that wall from the hinge on either side,
/// width = the leaf length under the typical-size prior, phi = angle(leaf, wall), swing = the side of the wall the leaf is on.
/// log_w = band + size + base rate (+ opening_llr, added by the caller). `rates` / `r_others`: the per-wall rates and the other doors'
/// existence probabilities on that wall (wall_base_rate.h); none = the seeds (a first door on its wall, 0.4).
/// ★The leaf's measured length is fused with the typical width as two Gaussians (sd of the length = sqrt(2) x the room band's sigma: both ends
/// are known to the room frame's own precision), and the size term is the typical prior's predictive at the measured length.
/// ★The mask's height is a LOWER BOUND on the door's (spec §3.4: a raw mask's height is never scored against the prior), so it only
/// lifts the aperture's height when it exceeds the typical one.
/// An empty polygon (< 3 vertices: no room, no wall) gives no hypothesis.
[[nodiscard]] inline std::vector<LeafApertureHyp> apertures_from_leaf(const Eigen::Vector2f& end_a, const Eigen::Vector2f& end_b,
    float height_m, std::span<const Eigen::Vector2f> polygon, std::span<const int> wall_groups,
    const core::SizePrior& typical, const RoomBand& band, const core::WallRates* rates = nullptr,
    std::span<const float> r_others = {})
{
    std::vector<LeafApertureHyp> out;
    const std::size_t n = polygon.size();
    const Eigen::Vector2f leaf = end_b - end_a;
    const float L = leaf.norm();
    if (n < 3 or not (L > 1e-4f) or wall_groups.size() < n) return out;
    const float s_len = std::numbers::sqrt2_v<float> * std::max(band.sigma_m, 1e-3f);
    const float v_m = s_len * s_len, v_0 = typical.w_sd * typical.w_sd;
    const float w_fused = (L / v_m + typical.w_m / v_0) / (1.f / v_m + 1.f / v_0);
    const float size = -0.5f * (L - typical.w_m) * (L - typical.w_m) / (v_0 + v_m);   // normalised to 0 at the typical width (place_prior.h)
    const float base = rates != nullptr ? core::wall_base_rate(*rates, r_others) : std::log(0.4f / 0.6f);
    for (int hinge_end = 0; hinge_end < 2; ++hinge_end)
    {
        const Eigen::Vector2f hinge = hinge_end == 0 ? end_a : end_b, free_end = hinge_end == 0 ? end_b : end_a;
        const int e = detail::nearest_edge(polygon, hinge);
        if (e < 0) continue;
        const Eigen::Vector2f p0 = polygon[static_cast<std::size_t>(e)], p1 = polygon[(static_cast<std::size_t>(e) + 1) % n];
        const Eigen::Vector2f u = (p1 - p0).normalized();
        const Eigen::Vector2f nrm(-u.y(), u.x());
        const Eigen::Vector2f foot = p0 + u * (hinge - p0).dot(u);        // the hinge's foot on the wall line: the aperture's origin
        const float band_lp = band_log_prior(signed_boundary_distance(std::vector<Eigen::Vector2f>(polygon.begin(), polygon.end()), hinge), band);
        const Eigen::Vector2f d = (free_end - hinge) / L;
        for (const float side : {+1.f, -1.f})
        {
            LeafApertureHyp h;
            Aperture& ap = h.aperture;
            ap.wall_O = foot; ap.wall_u = u; ap.w = w_fused;
            ap.s = side > 0.f ? 0.f : -w_fused;   // the aperture spans [0, w] (hinge at its near jamb) or [-w, 0] (hinge at the far jamb)
            ap.h = std::max(typical.h_m, height_m); ap.floor_z = 0.f; ap.thickness = band.wall_thickness_m;
            h.leaf.hinge = side > 0.f ? HingeSide::Near : HingeSide::Far;
            const float cos_phi = std::clamp(d.dot(side * u), -1.f, 1.f);
            h.leaf.phi = std::acos(cos_phi);
            h.leaf.swing = d.dot(nrm) >= 0.f ? +1.f : -1.f;
            h.wall = wall_groups[static_cast<std::size_t>(e)];
            h.log_w = double(band_lp) + double(size) + double(base);
            out.push_back(h);
        }
    }
    return out;
}

}   // namespace rc::door
