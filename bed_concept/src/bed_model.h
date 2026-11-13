/*
 * bed_model.h  —  geometry / state container for a bed instance.
 *
 * The recursive belief update lives in bed_belief.* (the AI2 full-covariance filter, BED.md);
 * this class is only the state holder + the compound SDF (box top + 4 corner legs = min(SDF_top, min_k
 * SDF_leg_k)), used to split a mask's support points into on-surface (candidate) vs off-surface (residual)
 * sets in BedFitter::observe and to render the mesh in BedSceneGraph. State θ = [cx, cy, w, h,
 * bed_height, leg_length, yaw, leg_inset]; fixed geometry TOP_THICKNESS = 0.03 m, LEG_RADIUS = 0.025 m.
 */

#pragma once

#include <array>
#include <cmath>
#include <Eigen/Dense>

namespace rc {

// Accepted bed pose + dimensions (room frame); the 8-DOF geometry the model renders and splits against.
struct BedState
{
    float cx           = 0.0f;   // Room-frame X of bed centre
    float cy           = 0.0f;   // Room-frame Y of bed centre
    float w            = 0.60f;  // Width  (bed-local X)  — standard bed footprint ≈ 0.60
    float h            = 0.60f;  // Depth  (bed-local Y)  — standard bed footprint ≈ 0.60
    float bed_height = 1.70f;  // Height of bed from floor (varies a lot; broad prior)
    float leg_length   = 0.72f;  // Length of legs from floor to underside of top
    float yaw          = 0.0f;   // Rotation around Z axis (room frame)
    float leg_inset    = 0.025f; // FROZEN at LEG_RADIUS: legs at outer edge (rim flush w/ bed edge)

    // Serialise/deserialise as 8-vector
    std::array<float, 8> to_array() const
    {
        return {cx, cy, w, h, bed_height, leg_length, yaw, leg_inset};
    }

    static BedState from_array(const std::array<float, 8>& a)
    {
        return {a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]};
    }
};

// Fixed model parameters (not fitted): the top/leg attribution band.
struct BedModelParams
{
    // Top/leg split band: a point no more than (TOP_THICKNESS + sigma_obs) below the top face is
    // attributed to the top slab (rather than a leg) by the compound SDF.
    float sigma_obs = 0.05f;
    // Headboard panel (same template as BedBeliefParams): its points are ON the bed's surface, not residual.
    float headboard_h_m = -1.0f;   // <=0 = no headboard
    float headboard_t_m = 0.10f;
};

// ─── BedModel ──────────────────────────────────────────────────────────────

// State holder + compound SDF (box top + 4 corner-leg cylinders) for one bed instance.
class BedModel
{
public:
    using State = BedState;   // for any shared Model-templated helper

    static constexpr float TOP_THICKNESS = 0.03f;
    static constexpr float LEG_RADIUS    = 0.025f;
    static constexpr float SDF_SMOOTH_K  = 0.02f;
    static constexpr float SDF_INSIDE_SCALE = 0.3f;

    BedModel() = default;
    BedModel(const BedState& prior, const BedModelParams& params);

    /** Compound SDF for a single 3-D point (room frame). 0 on the surface; >0 outside; <0 inside. */
    float sdf_point(const Eigen::Vector3f& p) const;

    // ── State access ─────────────────────────────────────────────────────────
    const BedState& state()  const { return state_; }
    const BedState& prior()  const { return prior_; }
    const BedModelParams& params() const { return params_; }

    void set_state(const BedState& s) { state_ = s; apply_constraints(); }
    void set_prior(const BedState& p) { prior_ = p; }

    /** Clamp leg_length ≤ bed_height − TOP_THICKNESS, positive dims, leg_inset = LEG_RADIUS. */
    void apply_constraints();

private:
    // SDF evaluated for a given explicit state.
    float sdf_point_at(const Eigen::Vector3f& p, const BedState& s) const;
    // Height at which the top/leg attribution splits: points above ⇒ top slab, below ⇒ legs.
    float top_split_z(const BedState& s) const;

    BedState        state_;
    BedState        prior_;
    BedModelParams  params_;
};

}  // namespace rc
