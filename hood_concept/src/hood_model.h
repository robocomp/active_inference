/*
 * hood_model.h  —  geometry / state container for a hood instance.
 *
 * The recursive belief update lives in hood_belief.* (the AI2 full-covariance filter, HOOD.md);
 * this class is only the state holder + the compound SDF (box top + 4 corner legs = min(SDF_top, min_k
 * SDF_leg_k)), used to split a mask's support points into on-surface (candidate) vs off-surface (residual)
 * sets in HoodFitter::observe and to render the mesh in HoodSceneGraph. State θ = [cx, cy, w, h,
 * hood_height, leg_length, yaw, leg_inset]; fixed geometry TOP_THICKNESS = 0.03 m, LEG_RADIUS = 0.025 m.
 */

#pragma once

#include <array>
#include <cmath>
#include <Eigen/Dense>

namespace rc {

// Accepted hood pose + dimensions (room frame); the 8-DOF geometry the model renders and splits against.
struct HoodState
{
    float cx           = 0.0f;   // Room-frame X of hood centre
    float cy           = 0.0f;   // Room-frame Y of hood centre
    float w            = 0.60f;  // Width  (hood-local X)  — standard fridge footprint ≈ 0.60
    float h            = 0.60f;  // Depth  (hood-local Y)  — standard fridge footprint ≈ 0.60
    // ═══════════════════════════════════════════════════════════════════════════════════════════════
    // ★★UNFINISHED: A HOOD DOES NOT STAND ON THE FLOOR. This model was cloned from refrigerator_concept,
    // whose entire vertical parameterisation assumes a FLOOR-ANCHORED box — the solid spans z ∈ [0, H],
    // so one number (H) fixes both the extent and the placement, and `leg_length` below is a leftover of
    // that lineage. A range hood HANGS: its underside sits ~1.55 m up and its top ~2.05 m, so it needs TWO
    // vertical parameters (z0, z1) or an anchor-plus-extent pair, and z0 is a real DOF the belief must
    // estimate rather than a constant it inherits from the floor.
    //
    // Everything downstream reads this: the carve box (rc::exist::carve_box takes z_min/z_max), the
    // projected silhouette, the NBV Target's z0/z1, and the RT pose's z. Leaving H as "height from floor"
    // makes all four quietly describe a fridge-shaped object standing under the hob.
    //
    // NOT patched here on purpose: it is a BELIEF-STATE change (the DOF vector, its Σ, the dof_spec, the
    // CSV header and the fitter's Jacobians all move together), which is exactly the "real work" the
    // scaffold is meant to leave visible instead of hiding under a rename. See CONCEPT_AGENT_RECIPE.md
    // §"Step 6 — the object-specific work", which lists this as hood's step 6.
    // ═══════════════════════════════════════════════════════════════════════════════════════════════
    float hood_height = 1.70f;  // Height of hood from floor (varies a lot; broad prior)
    float leg_length   = 0.72f;  // Length of legs from floor to underside of top
    float yaw          = 0.0f;   // Rotation around Z axis (room frame)
    float leg_inset    = 0.025f; // FROZEN at LEG_RADIUS: legs at outer edge (rim flush w/ hood edge)

    // Serialise/deserialise as 8-vector
    std::array<float, 8> to_array() const
    {
        return {cx, cy, w, h, hood_height, leg_length, yaw, leg_inset};
    }

    static HoodState from_array(const std::array<float, 8>& a)
    {
        return {a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]};
    }
};

// Fixed model parameters (not fitted): the top/leg attribution band.
struct HoodModelParams
{
    // Top/leg split band: a point no more than (TOP_THICKNESS + sigma_obs) below the top face is
    // attributed to the top slab (rather than a leg) by the compound SDF.
    float sigma_obs = 0.05f;
};

// ─── HoodModel ──────────────────────────────────────────────────────────────

// State holder + compound SDF (box top + 4 corner-leg cylinders) for one hood instance.
class HoodModel
{
public:
    using State = HoodState;   // for any shared Model-templated helper

    static constexpr float TOP_THICKNESS = 0.03f;
    static constexpr float LEG_RADIUS    = 0.025f;
    static constexpr float SDF_SMOOTH_K  = 0.02f;
    static constexpr float SDF_INSIDE_SCALE = 0.3f;

    HoodModel() = default;
    HoodModel(const HoodState& prior, const HoodModelParams& params);

    /** Compound SDF for a single 3-D point (room frame). 0 on the surface; >0 outside; <0 inside. */
    float sdf_point(const Eigen::Vector3f& p) const;

    // ── State access ─────────────────────────────────────────────────────────
    const HoodState& state()  const { return state_; }
    const HoodState& prior()  const { return prior_; }
    const HoodModelParams& params() const { return params_; }

    void set_state(const HoodState& s) { state_ = s; apply_constraints(); }
    void set_prior(const HoodState& p) { prior_ = p; }

    /** Clamp leg_length ≤ hood_height − TOP_THICKNESS, positive dims, leg_inset = LEG_RADIUS. */
    void apply_constraints();

private:
    // SDF evaluated for a given explicit state.
    float sdf_point_at(const Eigen::Vector3f& p, const HoodState& s) const;
    // Height at which the top/leg attribution splits: points above ⇒ top slab, below ⇒ legs.
    float top_split_z(const HoodState& s) const;

    HoodState        state_;
    HoodState        prior_;
    HoodModelParams  params_;
};

}  // namespace rc
