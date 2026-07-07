/*
 * residual_model.cpp  —  see residual_model.h
 */

#include "residual_model.h"

#include <algorithm>
#include <cmath>

namespace rc
{

void ResidualModel::apply_constraints()
{
    state_.w = std::clamp(state_.w, params_.min_size, params_.max_size);
    state_.d = std::clamp(state_.d, params_.min_size, params_.max_size);
    if (state_.height < 0.02f) state_.height = 0.02f;
}

float ResidualModel::footprint_sdf(const Eigen::Vector2f& p) const
{
    const float c = std::cos(state_.yaw), s = std::sin(state_.yaw);
    const float dx = p.x() - state_.cx, dy = p.y() - state_.cy;
    const float lx =  c * dx + s * dy;   // R(−yaw)
    const float ly = -s * dx + c * dy;
    const float qx = std::abs(lx) - 0.5f * state_.w;
    const float qy = std::abs(ly) - 0.5f * state_.d;
    const float outside = std::sqrt(std::max(qx, 0.0f) * std::max(qx, 0.0f) +
                                    std::max(qy, 0.0f) * std::max(qy, 0.0f));
    return outside + std::min(std::max(qx, qy), 0.0f);
}

std::vector<Eigen::Vector2f> ResidualModel::box_polygon() const
{
    const float c = std::cos(state_.yaw), s = std::sin(state_.yaw);
    const float hw = 0.5f * state_.w, hd = 0.5f * state_.d;
    const Eigen::Vector2f ctr(state_.cx, state_.cy);
    const auto corner = [&](float sx, float sy) {
        const float lx = sx * hw, ly = sy * hd;
        return ctr + Eigen::Vector2f(c * lx - s * ly, s * lx + c * ly);
    };
    return {corner(+1, +1), corner(-1, +1), corner(-1, -1), corner(+1, -1)};   // CCW
}

}  // namespace rc
