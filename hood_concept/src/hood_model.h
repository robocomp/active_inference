/*
 * hood_model.h  —  geometry / state container for a hood instance.
 *
 * The recursive belief update lives in hood_belief.* (the AI2 full-covariance filter, HOOD.md);
 * this class is only the state holder + the compound SDF (box top + 4 corner legs = min(SDF_top, min_k
 * SDF_leg_k)), used to split a mask's support points into on-surface (candidate) vs off-surface (residual)
 * sets in HoodFitter::observe and to render the mesh in HoodSceneGraph. State θ = [cx, cy, w, h,
 * z_top, extent, yaw]; the body spans z ∈ [z_top − extent, z_top] — see HoodState.
 */

#pragma once

#include <array>
#include <cmath>
#include <Eigen/Dense>

namespace rc {

// Accepted hood pose + dimensions (room frame); the geometry the model renders and splits against.
struct HoodState
{
    float cx           = 0.0f;   // Room-frame X of hood centre
    float cy           = 0.0f;   // Room-frame Y of hood centre
    float w            = 0.90f;  // Width  (hood-local X) — along the wall
    float h            = 0.50f;  // Depth  (hood-local Y) — out from the wall
    // ═══════════════════════════════════════════════════════════════════════════════════════════════
    // ★THE BODY HANGS, AND ITS SPAN IS STATED HERE ONCE — z ∈ [z0(), z1()].
    //
    // This replaces a single `hood_height` meaning "height from the floor", inherited from
    // refrigerator_concept where a fridge stands on the floor so ONE number fixes both the extent and the
    // placement. That conflation is not a naming problem: it is read as a span by every consumer, so a
    // hanging object silently became a fridge-shaped box standing under the hob. It survived five separate
    // repairs because each site re-derived the span from `hood_height` in its own arithmetic — the SDF that
    // splits mask points, the projected silhouette, the ROI, the voxel-ownership band, the front-face warp,
    // the LiDAR selection and carve, and the RT pose. Eight sites, one fact.
    //
    // So the fact lives HERE and nothing may restate it. Ask for z0()/z1()/zc(); never rebuild them from a
    // height and a floor. `leg_length`/`leg_inset` are gone with the same lineage (a hood has no legs; the
    // SDF already `(void)`-cast them).
    //
    // ★extent IS A PARAMETER, NOT A DOF — the belief estimates the placement, not the size. See
    // common/concept_manifest/hood.concept.toml, which argues it: a hood's underside is a crisp edge against
    // the hob gap, while its top merges into the wall or leaves the frame at any usable stand-off, and
    // estimating a vertical extent the data cannot resolve is how the size-oscillation bugs in this lineage
    // started. It is carried in the state (not read from config at each site) so that this struct alone
    // answers "where is the body", which is the whole point.
    // ═══════════════════════════════════════════════════════════════════════════════════════════════
    float z_top  = 2.05f;   // TOP of the body above the floor (m) — the estimated vertical placement
    float extent = 0.50f;   // vertical extent of the body (m) — a parameter, set from cfg.vertical_extent_m
    float yaw    = 0.0f;    // Rotation around Z axis (room frame)

    float z0() const { return z_top - extent; }          // underside
    float z1() const { return z_top; }                   // top
    float zc() const { return z_top - 0.5f * extent; }   // vertical centre
    float half_extent() const { return 0.5f * extent; }

    // Serialise/deserialise as 7-vector
    std::array<float, 7> to_array() const
    {
        return {cx, cy, w, h, z_top, extent, yaw};
    }

    static HoodState from_array(const std::array<float, 7>& a)
    {
        return {a[0], a[1], a[2], a[3], a[4], a[5], a[6]};
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

    /** Positive footprint, positive extent, and a body that cannot sink through the floor. */
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
