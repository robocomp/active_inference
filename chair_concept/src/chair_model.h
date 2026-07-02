/*
 * chair_model.h
 *
 * Geometry / state container for a chair instance.
 *
 * The recursive belief update lives in chair_belief.* (the AI2 full-covariance filter). This class is
 * now only the state holder + the compound SDF (seat slab + backrest + 4 legs), used to split a mask's
 * support points into on-surface (candidate) vs off-surface (residual) sets in ChairFitter::observe and
 * to render the mesh in ChairSceneGraph.
 *
 * State vector θ = [cx, cy, cz, yaw, seat_w, seat_d, seat_h, back_h]  (8 params)
 * Fixed geometry: SEAT_THICKNESS = 0.04 m, LEG_HALF = 0.02 m (square leg half-side)
 * Compound SDF = min(SDF_seat, SDF_back, min_k SDF_leg_k)
 */

#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>
#include <Eigen/Dense>

namespace rc {

struct ChairState
{
    float cx     = 0.0f;   // room-frame X of chair centre
    float cy     = 0.0f;   // room-frame Y of chair centre
    float cz     = 0.0f;   // floor height (ANCHORED — set by support step, not freely fit)
    float yaw    = 0.0f;   // heading about Z (room frame); +Y local = forward
    float seat_w = 0.45f;  // seat width  (local X)
    float seat_d = 0.45f;  // seat depth  (local Y)
    float seat_h = 0.45f;  // seat-TOP height above floor
    float back_h = 0.45f;  // backrest height above the seat top
    std::array<float, 8> to_array() const { return {cx, cy, cz, yaw, seat_w, seat_d, seat_h, back_h}; }
    static ChairState from_array(const std::array<float, 8>& a)
    { return {a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]}; }
};

struct ChairModelParams
{
    // Observation noise band used by the compound-SDF candidate/residual split.
    float sigma_obs = 0.05f;
};

// ─── ChairModel ──────────────────────────────────────────────────────────────

class ChairModel
{
public:
    using State = ChairState;   // for any shared Model-templated helper

    static constexpr float SEAT_THICKNESS = 0.04f;
    static constexpr float LEG_HALF       = 0.02f;   // square leg half-side
    static constexpr float SDF_SMOOTH_K  = 0.02f;
    static constexpr float SDF_INSIDE_SCALE = 0.3f;

    ChairModel() = default;
    ChairModel(const ChairState& prior, const ChairModelParams& params);

    /** Compound SDF for a single 3-D point (room frame). 0 on the surface; >0 outside; <0 inside. */
    float sdf_point(const Eigen::Vector3f& p) const;

    // ── State access ─────────────────────────────────────────────────────────
    const ChairState& state()  const { return state_; }
    const ChairState& prior()  const { return prior_; }
    const ChairModelParams& params() const { return params_; }

    void set_state(const ChairState& s) { state_ = s; apply_constraints(); }
    void set_prior(const ChairState& p) { prior_ = p; }

    /** Enforce wooden-chair physical bounds on the size DOFs; leave cx,cy,cz,yaw free. */
    void apply_constraints();

private:
    // SDF evaluated for a given explicit state.
    float sdf_point_at(const Eigen::Vector3f& p, const ChairState& s) const;

    ChairState        state_;
    ChairState        prior_;
    ChairModelParams  params_;
};

}  // namespace rc
