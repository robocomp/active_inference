/*
 * residual_model.h  —  geometry state carrier for one residual obstacle (room frame)
 *
 * The residual concept has NO shape prior, so unlike BottleModel (a cylinder SDF) this is a thin holder for
 * the fitted FOOTPRINT (cx,cy,yaw,w,d — the ResidualBelief posterior, copied back here by the fitter) plus
 * the anchored vertical band (cz,height, from the cluster z-range) and the Layer-B convex-hull footprint.
 * It carries the 2D oriented-box footprint SDF used for the merge/overlap and explained-interior (dissolve)
 * tests. No inference lives here (that is ResidualBelief); set_state() copies the posterior in + clamps.
 */

#pragma once

#include <array>
#include <vector>
#include <Eigen/Dense>

namespace rc
{

struct ResidualState
{
    float cx = 0.0f, cy = 0.0f, cz = 0.40f;   // room-frame centre; cz anchored to the cluster z-mid
    float yaw = 0.0f;                          // footprint orientation
    float w = 0.40f, d = 0.40f;                // footprint extents (belief posterior)
    float height = 0.80f;                      // anchored from the cluster z-range

    std::array<float, 7> to_array() const { return {cx, cy, cz, yaw, w, d, height}; }
};

struct ResidualModelParams
{
    float min_size = 0.05f;
    float max_size = 4.0f;
};

class ResidualModel
{
public:
    ResidualModel() = default;
    explicit ResidualModel(const ResidualModelParams& p) : params_(p) {}

    const ResidualState&      state()  const { return state_; }
    const ResidualModelParams& params() const { return params_; }
    void set_params(const ResidualModelParams& p) { params_ = p; }
    // Copy the fitted posterior in (belief footprint + anchored cz/height) and clamp the extents.
    void set_state(const ResidualState& s) { state_ = s; apply_constraints(); }

    const std::vector<Eigen::Vector2f>& hull() const { return hull_; }
    void set_hull(std::vector<Eigen::Vector2f> h) { hull_ = std::move(h); }

    // 2D oriented-box footprint SDF (matches ResidualBelief's box SDF). Negative inside. Used for
    // merge/overlap and the explained-interior test.
    float footprint_sdf(const Eigen::Vector2f& p) const;
    // The oriented-rectangle footprint polygon (4 CCW corners) — the safe-superset footprint the controller
    // consumes via width_m/depth_m/yaw; a conservative bound of the (tighter) convex hull.
    std::vector<Eigen::Vector2f> box_polygon() const;

private:
    void apply_constraints();

    ResidualState                state_;
    std::vector<Eigen::Vector2f> hull_;
    ResidualModelParams          params_;
};

}  // namespace rc
