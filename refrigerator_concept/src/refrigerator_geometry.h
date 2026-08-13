/*
 * refrigerator_geometry.h — pure footprint-geometry + uncertainty helpers for refrigerator_concept (header-only).
 *
 * Free functions in rc::geom shared by the worker's dashboard, convergence, and instance-merge paths (split
 * across specificworker*.cpp translation units): a scalar belief-uncertainty readout, and oriented-rectangle
 * footprint overlap (corners → Sutherland–Hodgman clip → overlap-area ratio) used to detect two instances
 * fitted to the same physical refrigerator. No state, no DSR — pure math on the belief covariance + the fitted state.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include <Eigen/Dense>

#include "../../common/footprint/footprint.h"   // rc::geom:: Footprint / overlap_ratio (SHARED)

#include "refrigerator_instance.h"   // rc::RefrigeratorInstance (belief covariance)
#include "refrigerator_model.h"      // rc::RefrigeratorState

namespace rc::geom {

// Scalar model-uncertainty readout for model_uncertainty_att / the dashboard: the sum of the belief's per-DOF
// posterior stds over position (cx,cy) + size (w,h), in metres, from the AI2 covariance Σ over
// [cx,cy,H,w,h,yaw]. Shrinks as the robot gathers viewpoints. 0 before the belief is seeded.
inline float belief_uncertainty(const rc::RefrigeratorInstance& inst)
{
    if (not inst.ai2_initialized)
        return 0.0f;
    const auto& S = inst.ai2_belief.covariance();
    const auto sd = [&](int i) { return std::sqrt(std::max(0.0f, S(i, i))); };
    return sd(0) + sd(1) + sd(3) + sd(4);
}

// ─── Footprint geometry: the plane maths is SHARED (common/footprint) ────────────────────────────
//
// corners / poly_area / clip_poly / footprint_overlap_ratio were FIVE BYTE-IDENTICAL copies across the
// fleet — 100.0% pairwise, character for character. They answer "are these two instances the same physical
// object?" for merge_overlapping_instances, a lifecycle decision with exactly the profile decide_removal had
// before it drifted three ways. What stays here is the only per-object part: how this object's state names
// its two footprint extents.
inline rc::geom::Footprint footprint_of(const rc::RefrigeratorState& s)
{ return { s.cx, s.cy, s.w, s.h, s.yaw }; }

inline float footprint_overlap_ratio(const rc::RefrigeratorState& a, const rc::RefrigeratorState& b)
{ return rc::geom::overlap_ratio(footprint_of(a), footprint_of(b)); }

}  // namespace rc::geom
