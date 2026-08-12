/*
 * hood_model.cpp  —  compound SDF (box top + 4 corner cylinders) for a hood instance, scalar (Eigen).
 *
 * The recursive belief update lives in hood_belief.* (AI2); this file is only the geometry/state container
 * used for the mask candidate/residual split and the mesh. Height-based attribution splits top vs legs.
 */

#include "hood_model.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace rc {

// ─── Helpers ─────────────────────────────────────────────────────────────────

namespace
{

/** Standard box SDF with smooth interior (log-sum-exp inside). */
float box_sdf(float dx, float dy, float dz)
{
    // Outside component: Euclidean distance from the surface
    float ox = std::max(dx, 0.0f);
    float oy = std::max(dy, 0.0f);
    float oz = std::max(dz, 0.0f);
    float outside = std::sqrt(ox*ox + oy*oy + oz*oz);

    // Inside component: smooth min via log-sum-exp
    float inside = 0.0f;
    if (dx < 0.0f and dy < 0.0f and dz < 0.0f)
    {
        constexpr float k = HoodModel::SDF_SMOOTH_K;
        constexpr float s = HoodModel::SDF_INSIDE_SCALE;
        // logsumexp(-[dx,dy,dz]/k)
        float a = dx / k, b = dy / k, c = dz / k;         // all negative → -a,-b,-c positive
        float mx = std::max({-a, -b, -c});
        float lse = mx + std::log(std::exp(-a - mx) + std::exp(-b - mx) + std::exp(-c - mx));
        inside = s * (-k * lse);                            // negative (inside box)
    }

    return outside + inside;
}

// (The finite-cylinder SDF that lived here was table_concept's LEG primitive. A hood is one solid
//  floor-anchored cuboid with no legs — sdf_point_at() is a single box_sdf — so it was dead from the day this
//  agent was cloned. Removed 2026-08-03; the compiler had been flagging it as unused ever since.)

} // anonymous namespace

// ─── HoodModel ──────────────────────────────────────────────────────────────

HoodModel::HoodModel(const HoodState& prior, const HoodModelParams& params)
    : state_(prior), prior_(prior), params_(params)
{
    apply_constraints();
}

float HoodModel::top_split_z(const HoodState& s) const
{
    // A point is a top-slab observation if it sits no more than (slab thickness + one obs-sigma)
    // below the body's TOP — z1(), never a height measured from the floor.
    return s.z1() - (TOP_THICKNESS + params_.sigma_obs);
}

// ─── SDF ─────────────────────────────────────────────────────────────────────

float HoodModel::sdf_point_at(const Eigen::Vector3f& p, const HoodState& s) const
{
    const float cx = s.cx, cy = s.cy;
    const float w = s.w, h = s.h;
    const float yaw = s.yaw;

    // Transform to hood-local frame
    const float cos_t = std::cos(-yaw);
    const float sin_t = std::sin(-yaw);
    const float px = p.x() - cx;
    const float py = p.y() - cy;
    const float local_x = px * cos_t - py * sin_t;
    const float local_y = px * sin_t + py * cos_t;
    const float local_z = p.z();

    // ── SOLID HANGING BOX ────────────────────────────────────────────────────
    // One cuboid spanning z ∈ [z0(), z1()], centred at zc() with half-extent extent/2. No top slab, no legs.
    //
    // ★THIS IS THE SDF THAT DECIDES WHICH MASK POINTS ARE THE HOOD (HoodFitter::observe splits on
    // |sdf| < SdfThresholdForStorage). It spanned z ∈ [0, hood_height] — the floor to the hood's top — so
    // every wall, backsplash and worktop point in the 2 m column BENEATH the hood that lay within the
    // threshold of the phantom box's side faces was admitted as hood surface, and the belief was then fit to
    // them. That is the reported "hood bites the wall" and the yaw sitting 9° off it: the footprint and
    // centre were being pulled by the wall the box was standing against. The hood's real underside, at
    // z0(), sat ~0.5 m INSIDE that phantom box, so its own points were classified as residual and thrown
    // away. Both halves of the split were wrong, in opposite directions.
    const float half_w  = w * 0.5f;
    const float half_h  = h * 0.5f;
    const float half_dz = s.half_extent();

    const float dx = std::abs(local_x) - half_w;
    const float dy = std::abs(local_y) - half_h;
    const float dz = std::abs(local_z - s.zc()) - half_dz;

    return box_sdf(dx, dy, dz);
}

float HoodModel::sdf_point(const Eigen::Vector3f& p) const
{
    return sdf_point_at(p, state_);
}

void HoodModel::apply_constraints()
{
    state_.w      = std::max(state_.w, 0.1f);
    state_.h      = std::max(state_.h, 0.1f);
    state_.extent = std::max(state_.extent, 0.05f);
    // The body hangs, so the only physical bound on its placement is that its UNDERSIDE stays above the
    // floor — not that its top does. z_top ≥ extent is exactly that statement, and it is the one constraint
    // the floor-anchored form could not express.
    state_.z_top  = std::max(state_.z_top, state_.extent);
}

}  // namespace rc
