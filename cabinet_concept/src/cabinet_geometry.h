/*
 * cabinet_geometry.h — pure footprint-geometry + uncertainty helpers for cabinet_concept (header-only).
 *
 * Free functions in rc::geom shared by the worker's dashboard, convergence, and instance-merge paths (split
 * across specificworker*.cpp translation units): a scalar belief-uncertainty readout, and oriented-rectangle
 * footprint overlap (corners → Sutherland–Hodgman clip → overlap-area ratio) used to detect two instances
 * fitted to the same physical cabinet. No state, no DSR — pure math on the belief covariance + the fitted state.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include <Eigen/Dense>

#include "../../common/footprint/footprint.h"   // rc::geom:: Footprint / overlap_ratio (SHARED)

#include "cabinet_instance.h"   // rc::CabinetInstance (belief covariance)
#include "cabinet_model.h"      // rc::CabinetState

namespace rc::geom {

// Scalar model-uncertainty readout for model_uncertainty_att / the dashboard: the sum of the belief's per-DOF
// posterior stds over position (cx,cy) + size (w,h), in metres, from the AI2 covariance Σ over
// [cx,cy,H,w,h,yaw]. Shrinks as the robot gathers viewpoints. 0 before the belief is seeded.
inline float belief_uncertainty(const rc::CabinetInstance& inst)
{
    if (not inst.ai2_initialized)
        return 0.0f;
    const auto& S = inst.ai2_belief.covariance();
    const auto sd = [&](int i) { return std::sqrt(std::max(0.0f, S(i, i))); };
    return sd(0) + sd(1) + sd(3) + sd(4);
}

// Two cabinets cannot share physical space. The footprint is the oriented rectangle (cx,cy,w,h,yaw) in the room
// plane; these helpers compute the overlap area between two footprints so the merge operator can collapse
// duplicate instances. Corners are returned CCW (local order (-,-),(+,-),(+,+),(-,+)).
// ─── Footprint geometry: the plane maths is SHARED (common/footprint) ────────────────────────────
//
// corners / poly_area / clip_poly / overlap_ratio were FIVE BYTE-IDENTICAL copies across the fleet — 100.0%
// pairwise, character for character. What stays here is the only per-object part: a RUN names its two
// footprint extents (L, d) where a box names them (w, h).
inline rc::geom::Footprint footprint_of(const rc::CabinetState& s)
{ return { s.cx, s.cy, s.L, s.d, s.yaw }; }

inline bool point_in_footprint(const rc::CabinetState& s, const Eigen::Vector2f& p, float margin = 0.0f)
{
    const float c = std::cos(s.yaw), sn = std::sin(s.yaw);
    const Eigen::Vector2f d(p.x() - s.cx, p.y() - s.cy);
    const float along = d.x() * c + d.y() * sn;      // along the run axis
    const float lat   = -d.x() * sn + d.y() * c;     // across it
    return std::abs(along) <= 0.5f * s.L + margin and std::abs(lat) <= 0.5f * s.d + margin;
}

// Overlap area as a fraction of the SMALLER footprint (1.0 = one cabinet fully inside the other).
inline float footprint_overlap_ratio(const rc::CabinetState& a, const rc::CabinetState& b)
{ return rc::geom::overlap_ratio(footprint_of(a), footprint_of(b)); }

// ─── Run-specific operators ──────────────────────────────────────────────────────────────────────
//
// A cabinet run is LONG, so every centroid-based metric inherited from table/chair breaks on it: the
// centre of a 4 m run can be metres away from any single mask fragment of that same run. The two
// operators below replace the point-centroid assumptions with segment ones.

// ── (1) ASSOCIATION: the along-axis position covariance of a run ─────────────────────────────────
//
// A mask fragment of a run can appear ANYWHERE along it, so the fragment centroid's along-axis
// coordinate carries essentially no information about WHICH run it belongs to — only its lateral
// offset does. The honest generative statement is that the fragment centroid is ~uniform along the
// run's length, and a uniform distribution over an interval of length L has variance L²/12.
//
// So instead of hacking a segment distance into the shared tracker, inflate the track's position
// covariance along the run axis by L²/12 and let the existing Mahalanobis gate (S = P + R²I) do
// exactly the right thing: permissive along the run, unchanged across it. This is the same
// anisotropic-covariance move used for the object anchor, and it keeps common/instance_tracker
// untouched for table/chair/bottle.
inline Eigen::Matrix2f run_position_cov(const rc::CabinetState& s, const Eigen::Matrix2f& base)
{
    const float c = std::cos(s.yaw), sn = std::sin(s.yaw);
    const Eigen::Vector2f axis(c, sn);
    const float var_along = std::max(0.0f, s.L * s.L / 12.0f);   // variance of U(−L/2, +L/2)
    return base + var_along * (axis * axis.transpose());
}

// ── (2) MERGE: collinear-run fusion ──────────────────────────────────────────────────────────────
//
// Oriented-rectangle overlap CANNOT merge two co-linear, non-overlapping fragments of one run — and
// that is the dominant duplicate mode here (the robot sees the left half, then the right half, and
// births two instances that never touch). Without this operator the agent would spawn cabinet_1..N
// along one counter, i.e. exactly the per-module individuation the run model exists to avoid.
//
// Two runs are the same run when all three hold, each tested against the pair's JOINT uncertainty
// rather than a fixed tolerance:
//   (a) their axes are parallel,
//   (b) their axis LINES coincide (equal lateral offset),
//   (c) their along-axis intervals overlap OR abut.
struct RunMerge
{
    bool  merge = false;
    float cx = 0.0f, cy = 0.0f, L = 0.0f, yaw = 0.0f;   // the fused run (UNION interval)
    float d_yaw = 0.0f, d_lat = 0.0f, gap = 0.0f;       // diagnostics (rad, m, m)
};

// sigma_yaw / sigma_lat / sigma_L are the pair's JOINT stds (√(σa²+σb²) + a sensor floor), supplied by
// the caller from the two beliefs' Σ. n_sigma is the χ-style acceptance width.
inline RunMerge collinear_merge(const rc::CabinetState& a, const rc::CabinetState& b,
                                float sigma_yaw, float sigma_lat, float sigma_L, float n_sigma)
{
    RunMerge r;
    const auto wrap_pi = [](float x)
    {
        while (x >  static_cast<float>(M_PI)) x -= 2.0f * static_cast<float>(M_PI);
        while (x <= -static_cast<float>(M_PI)) x += 2.0f * static_cast<float>(M_PI);
        return x;
    };

    // (a) Parallel axes. Compare MODULO π: the run's axis is a line, so a fragment whose 180° C2v fold
    // resolved the other way is still the same physical run (canonicalize normally aligns them, but a
    // fragment that has not yet seen the room interior may not be folded yet).
    float dyaw = wrap_pi(a.yaw - b.yaw);
    if (dyaw >  0.5f * static_cast<float>(M_PI)) dyaw -= static_cast<float>(M_PI);
    if (dyaw < -0.5f * static_cast<float>(M_PI)) dyaw += static_cast<float>(M_PI);
    r.d_yaw = dyaw;
    if (std::abs(dyaw) > n_sigma * std::max(1e-3f, sigma_yaw))
        return r;

    // Shared axis = the mean direction (they are parallel to within the test above; dyaw = a.yaw − b.yaw
    // modulo π, so a.yaw − dyaw/2 is the bisector).
    const float yaw_mean = wrap_pi(a.yaw - 0.5f * dyaw);
    const Eigen::Vector2f u(std::cos(yaw_mean), std::sin(yaw_mean));
    const Eigen::Vector2f n(-u.y(), u.x());
    const Eigen::Vector2f ca(a.cx, a.cy), cb(b.cx, b.cy);

    // (b) Same axis line: equal lateral (across-run) offset. Allow the carcass depth difference too —
    // a base run and the wall units above it are NOT the same run, but they differ in z, which the
    // caller separates; here two runs at different lateral offsets are different runs (e.g. the two
    // sides of a galley kitchen).
    r.d_lat = (cb - ca).dot(n);
    if (std::abs(r.d_lat) > n_sigma * std::max(1e-3f, sigma_lat))
        return r;

    // (c) Along-axis intervals overlap or abut. Project both runs onto the shared axis through a's centre.
    const float sa = 0.0f, sb = (cb - ca).dot(u);
    const float a0 = sa - 0.5f * a.L, a1 = sa + 0.5f * a.L;
    const float b0 = sb - 0.5f * b.L, b1 = sb + 0.5f * b.L;
    r.gap = std::max(0.0f, std::max(b0 - a1, a0 - b1));   // 0 when they overlap
    if (r.gap > n_sigma * std::max(1e-3f, sigma_L))
        return r;

    // Fuse: the UNION interval. Keeping only one fragment would throw away the extent the other
    // proved — and since the extent channel is grow-only/censored, that evidence would have to be
    // re-observed from scratch.
    const float lo = std::min(a0, b0), hi = std::max(a1, b1);
    const Eigen::Vector2f centre = ca + u * (0.5f * (lo + hi));
    r.merge = true;
    r.cx = centre.x(); r.cy = centre.y();
    r.L = hi - lo;
    r.yaw = yaw_mean;
    return r;
}

}  // namespace rc::geom
