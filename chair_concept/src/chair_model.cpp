/*
 * chair_model.cpp
 *
 * Compound SDF (seat slab + backrest + 4 square legs) for a chair instance, scalar (Eigen)
 * implementation. The recursive belief update lives in chair_belief.* (AI2); this file is only the
 * geometry/state container used for the mask candidate/residual split and the mesh.
 */

#include "chair_model.h"

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
        constexpr float k = ChairModel::SDF_SMOOTH_K;
        constexpr float s = ChairModel::SDF_INSIDE_SCALE;
        // logsumexp(-[dx,dy,dz]/k)
        float a = dx / k, b = dy / k, c = dz / k;         // all negative → -a,-b,-c positive
        float mx = std::max({-a, -b, -c});
        float lse = mx + std::log(std::exp(-a - mx) + std::exp(-b - mx) + std::exp(-c - mx));
        inside = s * (-k * lse);                            // negative (inside box)
    }

    return outside + inside;
}

} // anonymous namespace

// ─── ChairModel ──────────────────────────────────────────────────────────────

ChairModel::ChairModel(const ChairState& prior, const ChairModelParams& params)
    : state_(prior), prior_(prior), params_(params)
{
    apply_constraints();
}

// ─── SDF ─────────────────────────────────────────────────────────────────────

float ChairModel::sdf_point_at(const Eigen::Vector3f& p, const ChairState& s) const
{
    const float cos_t = std::cos(-s.yaw);
    const float sin_t = std::sin(-s.yaw);
    const float px = p.x() - s.cx;
    const float py = p.y() - s.cy;
    const float lx = px * cos_t - py * sin_t;
    const float ly = px * sin_t + py * cos_t;
    const float lz = p.z() - s.cz;

    const float half_w = s.seat_w * 0.5f;
    const float half_d = s.seat_d * 0.5f;
    const float half_t = SEAT_THICKNESS * 0.5f;

    const float seat_cz = s.seat_h - half_t;
    const float sdf_seat = box_sdf(std::abs(lx) - half_w, std::abs(ly) - half_d, std::abs(lz - seat_cz) - half_t);

    const float back_cy = -half_d + half_t;
    const float back_cz =  s.seat_h + s.back_h * 0.5f;
    const float sdf_back = box_sdf(std::abs(lx) - half_w, std::abs(ly - back_cy) - half_t, std::abs(lz - back_cz) - s.back_h * 0.5f);

    const float leg_top    = s.seat_h - SEAT_THICKNESS;
    const float leg_cz     = leg_top * 0.5f;
    const float leg_half_z = leg_top * 0.5f;
    const float ox = half_w - LEG_HALF;
    const float oy = half_d - LEG_HALF;
    const float leg_off[4][2] = {{ ox, oy}, {-ox, oy}, {-ox, -oy}, { ox, -oy}};
    float sdf_legs = std::numeric_limits<float>::max();
    for (const auto& o : leg_off)
        sdf_legs = std::min(sdf_legs, box_sdf(std::abs(lx - o[0]) - LEG_HALF, std::abs(ly - o[1]) - LEG_HALF, std::abs(lz - leg_cz) - leg_half_z));

    return std::min(sdf_seat, std::min(sdf_back, sdf_legs));
}

float ChairModel::sdf_point(const Eigen::Vector3f& p) const
{
    return sdf_point_at(p, state_);
}

// ─── Constraints ─────────────────────────────────────────────────────────────

void ChairModel::apply_constraints()
{
    // Physical bounds for a wooden chair (Webots WoodenChair ≈ 0.45×0.45, seat ~0.46 m, total ~0.92).
    // A hard ceiling stops the flat seat slab from ballooning to swallow contaminated mask points.
    state_.seat_w = std::clamp(state_.seat_w, 0.30f, 0.65f);
    state_.seat_d = std::clamp(state_.seat_d, 0.30f, 0.65f);
    state_.seat_h = std::clamp(state_.seat_h, 0.38f, 0.55f);
    state_.back_h = std::clamp(state_.back_h, 0.30f, 0.65f);
}

}  // namespace rc
