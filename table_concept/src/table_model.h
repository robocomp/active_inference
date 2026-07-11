/*
 * table_model.h  —  geometry / state container for a table instance.
 *
 * The recursive belief update lives in table_belief.* (the AI2 full-covariance filter, TABLE_FIT_AI2.md);
 * this class is only the state holder + the compound SDF (box top + 4 corner legs = min(SDF_top, min_k
 * SDF_leg_k)), used to split a mask's support points into on-surface (candidate) vs off-surface (residual)
 * sets in TableFitter::observe and to render the mesh in TableSceneGraph. State θ = [cx, cy, w, h,
 * table_height, leg_length, yaw, leg_inset]; fixed geometry TOP_THICKNESS = 0.03 m, LEG_RADIUS = 0.025 m.
 */

#pragma once

#include <array>
#include <cmath>
#include <Eigen/Dense>

namespace rc {

// Accepted table pose + dimensions (room frame); the 8-DOF geometry the model renders and splits against.
struct TableState
{
    float cx           = 0.0f;   // Room-frame X of table centre
    float cy           = 0.0f;   // Room-frame Y of table centre
    float w            = 1.0f;   // Width  (table-local X)
    float h            = 0.6f;   // Depth  (table-local Y)
    float table_height = 0.75f;  // Height of table surface from floor
    float leg_length   = 0.72f;  // Length of legs from floor to underside of top
    float yaw          = 0.0f;   // Rotation around Z axis (room frame)
    float leg_inset    = 0.025f; // FROZEN at LEG_RADIUS: legs at outer edge (rim flush w/ table edge)

    // Serialise/deserialise as 8-vector
    std::array<float, 8> to_array() const
    {
        return {cx, cy, w, h, table_height, leg_length, yaw, leg_inset};
    }

    static TableState from_array(const std::array<float, 8>& a)
    {
        return {a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]};
    }
};

// Fixed model parameters (not fitted): the top/leg attribution band.
struct TableModelParams
{
    // Top/leg split band: a point no more than (TOP_THICKNESS + sigma_obs) below the top face is
    // attributed to the top slab (rather than a leg) by the compound SDF.
    float sigma_obs = 0.05f;
};

// ─── TableModel ──────────────────────────────────────────────────────────────

// State holder + compound SDF (box top + 4 corner-leg cylinders) for one table instance.
class TableModel
{
public:
    using State = TableState;   // for any shared Model-templated helper

    static constexpr float TOP_THICKNESS = 0.03f;
    static constexpr float LEG_RADIUS    = 0.025f;
    static constexpr float SDF_SMOOTH_K  = 0.02f;
    static constexpr float SDF_INSIDE_SCALE = 0.3f;

    TableModel() = default;
    TableModel(const TableState& prior, const TableModelParams& params);

    /** Compound SDF for a single 3-D point (room frame). 0 on the surface; >0 outside; <0 inside. */
    float sdf_point(const Eigen::Vector3f& p) const;

    // ── State access ─────────────────────────────────────────────────────────
    const TableState& state()  const { return state_; }
    const TableState& prior()  const { return prior_; }
    const TableModelParams& params() const { return params_; }

    void set_state(const TableState& s) { state_ = s; apply_constraints(); }
    void set_prior(const TableState& p) { prior_ = p; }

    /** Clamp leg_length ≤ table_height − TOP_THICKNESS, positive dims, leg_inset = LEG_RADIUS. */
    void apply_constraints();

private:
    // SDF evaluated for a given explicit state.
    float sdf_point_at(const Eigen::Vector3f& p, const TableState& s) const;
    // Height at which the top/leg attribution splits: points above ⇒ top slab, below ⇒ legs.
    float top_split_z(const TableState& s) const;

    TableState        state_;
    TableState        prior_;
    TableModelParams  params_;
};

}  // namespace rc
