/*
 * countable_model.cpp  —  compound SDF (box top + 4 corner cylinders) for a bed instance, scalar (Eigen).
 *
 * The recursive belief update lives in countable_belief.* (AI2); this file is only the geometry/state container
 * used for the mask candidate/residual split and the mesh. Height-based attribution splits top vs legs.
 */

#include "countable_model.h"
#include "countable_vertical_span.h"

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
        constexpr float k = CountableModel::SDF_SMOOTH_K;
        constexpr float s = CountableModel::SDF_INSIDE_SCALE;
        // logsumexp(-[dx,dy,dz]/k)
        float a = dx / k, b = dy / k, c = dz / k;         // all negative → -a,-b,-c positive
        float mx = std::max({-a, -b, -c});
        float lse = mx + std::log(std::exp(-a - mx) + std::exp(-b - mx) + std::exp(-c - mx));
        inside = s * (-k * lse);                            // negative (inside box)
    }

    return outside + inside;
}

// (The finite-cylinder SDF that lived here was table_concept's LEG primitive. A bed is one solid
//  floor-anchored cuboid with no legs — sdf_point_at() is a single box_sdf — so it was dead from the day this
//  agent was cloned. Removed 2026-08-03; the compiler had been flagging it as unused ever since.)

} // anonymous namespace

// ─── CountableModel ──────────────────────────────────────────────────────────────

CountableModel::CountableModel(const CountableState& prior, const CountableModelParams& params)
    : state_(prior), prior_(prior), params_(params)
{
    apply_constraints();
}

float CountableModel::top_split_z(const CountableState& s) const
{
    // A point is a top-slab observation if it sits no more than (slab thickness + one obs-sigma)
    // below the belief surface height.
    return s.countable_height - (TOP_THICKNESS + params_.sigma_obs);
}

// ─── SDF ─────────────────────────────────────────────────────────────────────

float CountableModel::sdf_point_at(const Eigen::Vector3f& p, const CountableState& s) const
{
    const float cx = s.cx, cy = s.cy;
    const float w = s.w, h = s.h;
    const float countable_height = s.countable_height;
    const float leg_length   = s.leg_length;
    const float yaw          = s.yaw;

    // Transform to bed-local frame
    const float cos_t = std::cos(-yaw);
    const float sin_t = std::sin(-yaw);
    const float px = p.x() - cx;
    const float py = p.y() - cy;
    const float local_x = px * cos_t - py * sin_t;
    const float local_y = px * sin_t + py * cos_t;
    const float local_z = p.z();
    (void) leg_length;   // no legs on a bed (single solid box); field kept inert for compatibility

    // ── SOLID BOX ────────────────────────────────────────────────────────────
    // One cuboid spanning z∈[0, countable_height] (floor-anchored: centre at half-height, half extents (w/2, h/2,
    // countable_height/2)), or z∈[H − e, H] when the class hangs (VerticalSpan). No top slab, no legs.
    const float half_w = w * 0.5f;
    const float half_h = h * 0.5f;
    const rc::countable::VerticalSpan span{params_.vertical_extent_m};

    const float dx = std::abs(local_x) - half_w;
    const float dy = std::abs(local_y) - half_h;
    const float dz = std::abs(local_z - span.zc(countable_height)) - span.half(countable_height);

    const float d_box = box_sdf(dx, dy, dz);
    if (params_.headboard_h_m <= 0.0f) return d_box;
    // Headboard panel on the back face (+Y local), full width, floor to headboard_h_m (see CountableBeliefParams).
    const float t   = std::min(params_.headboard_t_m, h);
    const float yc  = half_h - 0.5f * t;
    const float hz  = 0.5f * params_.headboard_h_m;
    const float d_hb = box_sdf(dx, std::abs(local_y - yc) - 0.5f * t, std::abs(local_z - hz) - hz);
    return std::min(d_box, d_hb);
}

float CountableModel::sdf_point(const Eigen::Vector3f& p) const
{
    return sdf_point_at(p, state_);
}

void CountableModel::apply_constraints()
{
    state_.w            = std::max(state_.w, 0.1f);
    state_.h            = std::max(state_.h, 0.1f);
    state_.countable_height = std::max(state_.countable_height, 0.05f + TOP_THICKNESS);
    const float max_leg = state_.countable_height - TOP_THICKNESS;
    state_.leg_length   = std::clamp(state_.leg_length, 0.05f, max_leg);
    // leg_inset is FROZEN (not estimated): legs sit at the OUTER edge of the top, their outer rim
    // flush with the bed edge (centre at half_w − LEG_RADIUS).
    state_.leg_inset    = LEG_RADIUS;
}

}  // namespace rc
