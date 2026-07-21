/*
 * cabinet_model.h  —  geometry / state container for a cabinet instance.
 *
 * The recursive belief update lives in cabinet_belief.* (the AI2 full-covariance filter, CABINET.md);
 * this class is only the state holder + the compound SDF (box top + 4 corner legs = min(SDF_top, min_k
 * SDF_leg_k)), used to split a mask's support points into on-surface (candidate) vs off-surface (residual)
 * sets in CabinetFitter::observe and to render the mesh in CabinetSceneGraph. State θ = [cx, cy, w, h,
 * cabinet_height, leg_length, yaw, leg_inset]; fixed geometry TOP_THICKNESS = 0.03 m, LEG_RADIUS = 0.025 m.
 */

#pragma once

#include <array>
#include <cmath>
#include <Eigen/Dense>

namespace rc {

// Accepted run pose + dimensions (room frame): the geometry the model renders, splits mask points
// against, and publishes. Mirrors CabinetBeliefState 1:1 — unlike the table (whose 8-DOF geometry
// state deliberately differed from its belief state), a run's geometry IS its belief state, so
// keeping a second parameterisation would only invite the two to drift apart.
struct CabinetState
{
    float cx  = 0.0f;    // room-frame X of the run centre
    float cy  = 0.0f;    // room-frame Y of the run centre
    float yaw = 0.0f;    // direction of the run's LONG axis (room frame)
    float L   = 1.0f;    // length along that axis
    float d   = 0.60f;   // carcass depth, front face → back face
    float z0  = 0.0f;    // bottom height above the floor
    float z1  = 0.90f;   // top height above the floor

    float height() const { return std::max(0.0f, z1 - z0); }
    float zc()     const { return 0.5f * (z0 + z1); }

    std::array<float, 7> to_array() const { return {cx, cy, yaw, L, d, z0, z1}; }
    static CabinetState from_array(const std::array<float, 7>& a)
    { return {a[0], a[1], a[2], a[3], a[4], a[5], a[6]}; }
};

// Fixed model parameters (not fitted).
struct CabinetModelParams
{
    float sigma_obs = 0.05f;   // surface band used when splitting mask points into candidate/residual
};

// ─── CabinetModel ──────────────────────────────────────────────────────────────

// State holder + SDF for one cabinet RUN. A run is a single solid box — no legs, no top/leg
// attribution band (contrast the table's slab + 4 corner cylinders), so the compound SDF collapses
// to one oriented-box distance.
class CabinetModel
{
public:
    using State = CabinetState;   // for any shared Model-templated helper

    CabinetModel() = default;
    CabinetModel(const CabinetState& prior, const CabinetModelParams& params);

    /** Oriented-box SDF for a single 3-D point (room frame). 0 on the surface; >0 outside; <0 inside. */
    float sdf_point(const Eigen::Vector3f& p) const;

    // ── State access ─────────────────────────────────────────────────────────
    const CabinetState& state()  const { return state_; }
    const CabinetState& prior()  const { return prior_; }
    const CabinetModelParams& params() const { return params_; }

    void set_state(const CabinetState& s) { state_ = s; apply_constraints(); }
    void set_prior(const CabinetState& p) { prior_ = p; }

    /** Keep extents positive and the vertical band correctly ordered. */
    void apply_constraints();

private:
    float sdf_point_at(const Eigen::Vector3f& p, const CabinetState& s) const;

    CabinetState        state_;
    CabinetState        prior_;
    CabinetModelParams  params_;
};

}  // namespace rc
