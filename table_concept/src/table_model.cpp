/*
 * table_model.cpp
 *
 * Compound SDF (box top + 4 corner cylinders) for a table instance, scalar (Eigen) implementation.
 * The recursive belief update lives in table_belief.* (AI2); this file is only the geometry/state
 * container used for the mask candidate/residual split and the mesh.
 */

#include "table_model.h"

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
        constexpr float k = TableModel::SDF_SMOOTH_K;
        constexpr float s = TableModel::SDF_INSIDE_SCALE;
        // logsumexp(-[dx,dy,dz]/k)
        float a = dx / k, b = dy / k, c = dz / k;         // all negative → -a,-b,-c positive
        float mx = std::max({-a, -b, -c});
        float lse = mx + std::log(std::exp(-a - mx) + std::exp(-b - mx) + std::exp(-c - mx));
        inside = s * (-k * lse);                            // negative (inside box)
    }

    return outside + inside;
}

/** Finite cylinder SDF (radius r, half-height hh, centred at origin). */
float cylinder_sdf(float dx_local, float dy_local, float dz_local, float r, float hh)
{
    float d_radial   = std::sqrt(dx_local*dx_local + dy_local*dy_local) - r;
    float d_vertical = std::abs(dz_local) - hh;

    float or_ = std::max(d_radial, 0.0f);
    float ov  = std::max(d_vertical, 0.0f);
    float outside = std::sqrt(or_*or_ + ov*ov);
    float inside  = std::min(std::max(d_radial, d_vertical), 0.0f);
    return outside + inside;
}

} // anonymous namespace

// ─── TableModel ──────────────────────────────────────────────────────────────

TableModel::TableModel(const TableState& prior, const TableModelParams& params)
    : state_(prior), prior_(prior), params_(params)
{
    apply_constraints();
}

float TableModel::top_split_z(const TableState& s) const
{
    // A point is a top-slab observation if it sits no more than (slab thickness + one obs-sigma)
    // below the belief surface height.
    return s.table_height - (TOP_THICKNESS + params_.sigma_obs);
}

// ─── SDF ─────────────────────────────────────────────────────────────────────

float TableModel::sdf_point_at(const Eigen::Vector3f& p, const TableState& s) const
{
    const float cx = s.cx, cy = s.cy;
    const float w = s.w, h = s.h;
    const float table_height = s.table_height;
    const float leg_length   = s.leg_length;
    const float yaw          = s.yaw;

    // Transform to table-local frame
    const float cos_t = std::cos(-yaw);
    const float sin_t = std::sin(-yaw);
    const float px = p.x() - cx;
    const float py = p.y() - cy;
    const float local_x = px * cos_t - py * sin_t;
    const float local_y = px * sin_t + py * cos_t;
    const float local_z = p.z();

    // ── TABLE TOP ────────────────────────────────────────────────────────────
    const float half_w = w * 0.5f;
    const float half_h = h * 0.5f;
    const float half_t = TOP_THICKNESS * 0.5f;
    const float top_cz = table_height - half_t;

    const float dx_top = std::abs(local_x) - half_w;
    const float dy_top = std::abs(local_y) - half_h;
    const float dz_top = std::abs(local_z - top_cz) - half_t;

    const float sdf_top = box_sdf(dx_top, dy_top, dz_top);

    // ── LEGS (4 cylinders at corners) ────────────────────────────────────────
    const float leg_inset    = s.leg_inset;
    const float leg_center_z = leg_length * 0.5f;
    const float leg_half_h   = leg_length * 0.5f;

    const float leg_offsets[4][2] = {
        { half_w - leg_inset,  half_h - leg_inset},
        {-half_w + leg_inset,  half_h - leg_inset},
        {-half_w + leg_inset, -half_h + leg_inset},
        { half_w - leg_inset, -half_h + leg_inset},
    };

    float sdf_leg_min = std::numeric_limits<float>::max();
    for (const auto& off : leg_offsets)
    {
        const float dlx = local_x - off[0];
        const float dly = local_y - off[1];
        const float dlz = local_z - leg_center_z;
        sdf_leg_min = std::min(sdf_leg_min, cylinder_sdf(dlx, dly, dlz, LEG_RADIUS, leg_half_h));
    }

    // Height-based attribution: above the split ⇒ top slab, below ⇒ nearest leg.
    return (local_z >= top_split_z(s)) ? sdf_top : sdf_leg_min;
}

float TableModel::sdf_point(const Eigen::Vector3f& p) const
{
    return sdf_point_at(p, state_);
}

void TableModel::apply_constraints()
{
    state_.w            = std::max(state_.w, 0.1f);
    state_.h            = std::max(state_.h, 0.1f);
    state_.table_height = std::max(state_.table_height, 0.05f + TOP_THICKNESS);
    const float max_leg = state_.table_height - TOP_THICKNESS;
    state_.leg_length   = std::clamp(state_.leg_length, 0.05f, max_leg);
    // leg_inset is FROZEN (not estimated): legs sit at the OUTER edge of the top, their outer rim
    // flush with the table edge (centre at half_w − LEG_RADIUS).
    state_.leg_inset    = LEG_RADIUS;
}

}  // namespace rc
