/*
 * door_model.cpp
 *
 * Compound SDF (seat slab + backrest + 4 square legs) for a door instance, scalar (Eigen)
 * implementation. The recursive belief update lives in door_belief.* (AI2); this file is only the
 * geometry/state container used for the mask candidate/residual split and the mesh.
 */

#include "door_model.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace rc {

// ─── Helpers ─────────────────────────────────────────────────────────────────

namespace
{

/** Exact box SDF from per-axis face distances (|local| − half_extent). */
float box_sdf(float dx, float dy, float dz)
{
    const float ox = std::max(dx, 0.0f), oy = std::max(dy, 0.0f), oz = std::max(dz, 0.0f);
    const float outside = std::sqrt(ox * ox + oy * oy + oz * oz);
    const float inside  = std::min(std::max(dx, std::max(dy, dz)), 0.0f);
    return outside + inside;
}

} // anonymous namespace

// ─── DoorModel ──────────────────────────────────────────────────────────────

DoorModel::DoorModel(const DoorState& prior, const DoorModelParams& params)
    : state_(prior), prior_(prior), params_(params)
{
    apply_constraints();
}

// ─── SDF ─────────────────────────────────────────────────────────────────────

float DoorModel::sdf_point_at(const Eigen::Vector3f& p, const DoorState& s) const
{
    const float cos_t = std::cos(-s.yaw);
    const float sin_t = std::sin(-s.yaw);
    const float px = p.x() - s.cx;
    const float py = p.y() - s.cy;
    const float lx = px * cos_t - py * sin_t;   // along the wall
    const float ly = px * sin_t + py * cos_t;   // across the wall
    const float lz = p.z() - s.cz;              // from the floor base

    const float half_w = s.w * 0.5f;
    const float half_t = s.thickness * 0.5f;
    const float half_h = s.h * 0.5f;

    // Single thin panel: centred horizontally at (cx,cy), rising from the floor base cz to cz+h.
    return box_sdf(std::abs(lx) - half_w, std::abs(ly) - half_t, std::abs(lz - half_h) - half_h);
}

float DoorModel::sdf_point(const Eigen::Vector3f& p) const
{
    return sdf_point_at(p, state_);
}

// ─── Constraints ─────────────────────────────────────────────────────────────

void DoorModel::apply_constraints()
{
    // The belief (door_belief) owns the strong w/h priors; here just enforce physical positivity so a
    // stray write can't invert a panel dimension. thickness/cz/pose are as written by the belief.
    state_.w = std::max(0.05f, state_.w);
    state_.h = std::max(0.05f, state_.h);
    state_.thickness = std::max(0.005f, state_.thickness);
}

}  // namespace rc
