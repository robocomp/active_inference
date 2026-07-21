/*
 * cabinet_model.cpp — geometry holder + oriented-box SDF for one cabinet RUN (see cabinet_model.h).
 */

#include "cabinet_model.h"

#include <algorithm>

namespace rc {

namespace { constexpr float kMinExtent = 0.05f; }

CabinetModel::CabinetModel(const CabinetState& prior, const CabinetModelParams& params)
    : state_(prior), prior_(prior), params_(params)
{
    apply_constraints();
}

void CabinetModel::apply_constraints()
{
    state_.L = std::max(kMinExtent, state_.L);
    state_.d = std::max(kMinExtent, state_.d);
    state_.z0 = std::max(0.0f, state_.z0);
    state_.z1 = std::max(state_.z0 + kMinExtent, state_.z1);
}

// Exact oriented-box SDF. Local frame: x along the run's long axis, y along its front normal, z up.
// One primitive — a cabinet run is a solid carcass, so there is no top/leg attribution to make.
float CabinetModel::sdf_point_at(const Eigen::Vector3f& p, const CabinetState& s) const
{
    const float c = std::cos(s.yaw), sn = std::sin(s.yaw);
    const float dx = p.x() - s.cx, dy = p.y() - s.cy;
    const Eigen::Vector3f q(dx * c + dy * sn,          // along the axis
                            -dx * sn + dy * c,         // along the front normal
                            p.z() - s.zc());
    const Eigen::Vector3f half(0.5f * std::max(kMinExtent, s.L),
                               0.5f * std::max(kMinExtent, s.d),
                               0.5f * std::max(kMinExtent, s.height()));
    const Eigen::Vector3f e = q.cwiseAbs() - half;
    return e.cwiseMax(0.0f).norm() + std::min(e.maxCoeff(), 0.0f);
}

float CabinetModel::sdf_point(const Eigen::Vector3f& p) const { return sdf_point_at(p, state_); }

}  // namespace rc
