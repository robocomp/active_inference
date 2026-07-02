/*
 * bottle_model.cpp
 *
 * Single vertical-cylinder SDF + state carrier for a bottle instance. The fit itself is the AI2 belief
 * (bottle_belief.h); this file supplies the cylinder SDF (used by observe() to split candidate/residual
 * mask points) and holds the posterior state + the mask silhouette rays the belief consumes.
 */

#include "bottle_model.h"
#include <algorithm>
#include <cmath>

namespace rc {

// ─── Helpers ─────────────────────────────────────────────────────────────────

namespace
{

    /** Finite cylinder SDF (radius r, half-height hh), point already centred. */
    float cylinder_sdf(float dx, float dy, float dz, float r, float hh)
    {
        const float d_radial   = std::sqrt(dx*dx + dy*dy) - r;
        const float d_vertical = std::abs(dz) - hh;

        const float or_ = std::max(d_radial, 0.0f);
        const float ov  = std::max(d_vertical, 0.0f);
        const float outside = std::sqrt(or_*or_ + ov*ov);
        const float inside  = std::min(std::max(d_radial, d_vertical), 0.0f);
        return outside + inside;
    }

    /**
     * Robust IRLS weight ω(r) = ρ'(r)/(2r), normalised so the quadratic loss gives ω≡1. This is the
     * per-residual multiplier turning a squared error into the M-estimator's effective precision;
     * used to down-weight outliers in the Fisher information (an outlier should contribute little
     * curvature, exactly as it contributes little gradient). Mirrors TableModel's robust_irls_weight.
     */
    // radius / half-height carried as tensors so the optimiser can size the bottle.
} // anonymous namespace

// ─── BottleModel ───────────────────────────────────────────────────────────────

BottleModel::BottleModel(const BottleState& prior, const BottleModelParams& params)
    : state_(prior), prior_(prior), params_(params)
{
    apply_constraints();
}

// ─── SDF ─────────────────────────────────────────────────────────────────────

float BottleModel::sdf_point_at(const Eigen::Vector3f& p, const BottleState& s) const
{
    const float dx = p.x() - s.cx;
    const float dy = p.y() - s.cy;
    const float dz = p.z() - s.cz;
    return cylinder_sdf(dx, dy, dz, s.radius, s.height * 0.5f);
}

float BottleModel::sdf_point(const Eigen::Vector3f& p) const
{
    return sdf_point_at(p, state_);
}

std::vector<float> BottleModel::compute_sdf(const std::vector<Eigen::Vector3f>& points) const
{
    std::vector<float> out;
    out.reserve(points.size());
    for (const auto& p : points)
        out.push_back(sdf_point(p));
    return out;
}

// ─── Free Energy ─────────────────────────────────────────────────────────────

// ─── Gradient step ───────────────────────────────────────────────────────────

// ─── Pose covariance (Laplace approximation) ───────────────────────────────────

// ─── Observation Fisher information (per-DOF) ──────────────────────────────────

// ─── Constraints ─────────────────────────────────────────────────────────────

void BottleModel::apply_constraints()
{
    state_.radius = std::max(state_.radius, MIN_RADIUS);
    state_.height = std::max(state_.height, MIN_HEIGHT);
}

// ─── Bounding box ────────────────────────────────────────────────────────────

std::pair<Eigen::Vector3f, Eigen::Vector3f> BottleModel::bounding_box() const
{
    const float r  = state_.radius;
    const float hh = state_.height * 0.5f;
    return {
        Eigen::Vector3f{state_.cx - r, state_.cy - r, state_.cz - hh},
        Eigen::Vector3f{state_.cx + r, state_.cy + r, state_.cz + hh}
    };
}

}  // namespace rc
