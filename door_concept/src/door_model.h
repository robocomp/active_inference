/*
 * door_model.h
 *
 * Geometry / state container for a door instance.
 *
 * The recursive belief update lives in door_belief.* (the AI2 full-covariance filter over the wall-frame
 * θ=[s,w,h]). This class is now only a ROOM-FRAME state holder + the single thin-box panel SDF, used to
 * split a mask's support points into on-surface (candidate) vs off-surface (residual) sets in
 * DoorFitter::observe and to render the mesh in DoorSceneGraph. The fitter writes the belief's room-frame
 * read-back (centre_xy/yaw/width/height/thickness) into this DoorState each cycle.
 *
 * State (room frame): centre (cx,cy), floor base cz (panel spans [cz, cz+h]), heading yaw (= wall
 * tangent), width w (local x, along the wall), height h (vertical), thickness (local y, across the wall).
 * Single box SDF centred at (cx,cy,cz+h/2) with half-extents (w/2, thickness/2, h/2).
 */

#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>
#include <Eigen/Dense>

namespace rc {

struct DoorState
{
    float cx        = 0.0f;   // room-frame X of the panel centre
    float cy        = 0.0f;   // room-frame Y of the panel centre
    float cz        = 0.0f;   // floor base height (panel spans [cz, cz+h]); pinned, not fit
    float yaw       = 0.0f;   // heading about Z (room frame) = wall tangent; local +X runs along the wall
    float w         = 0.70f;  // panel width  (local X, along the wall)
    float h         = 2.00f;  // panel height (vertical)
    float thickness = 0.05f;  // panel thickness (local Y, across the wall — fixed)
    std::array<float, 7> to_array() const { return {cx, cy, cz, yaw, w, h, thickness}; }
    static DoorState from_array(const std::array<float, 7>& a)
    { return {a[0], a[1], a[2], a[3], a[4], a[5], a[6]}; }
};

struct DoorModelParams
{
    // Observation noise band used by the single-box-SDF candidate/residual split.
    float sigma_obs = 0.05f;
};

// ─── DoorModel ──────────────────────────────────────────────────────────────

class DoorModel
{
public:
    using State = DoorState;   // for any shared Model-templated helper

    DoorModel() = default;
    DoorModel(const DoorState& prior, const DoorModelParams& params);

    /** Single thin-box panel SDF for a 3-D point (room frame). 0 on the surface; >0 outside; <0 inside. */
    float sdf_point(const Eigen::Vector3f& p) const;

    // ── State access ─────────────────────────────────────────────────────────
    const DoorState& state()  const { return state_; }
    const DoorState& prior()  const { return prior_; }
    const DoorModelParams& params() const { return params_; }

    void set_state(const DoorState& s) { state_ = s; apply_constraints(); }
    void set_prior(const DoorState& p) { prior_ = p; }

    /** Enforce physical positivity on w,h; leave cx,cy,cz,yaw,thickness as written by the belief. */
    void apply_constraints();

private:
    // SDF evaluated for a given explicit state.
    float sdf_point_at(const Eigen::Vector3f& p, const DoorState& s) const;

    DoorState        state_;
    DoorState        prior_;
    DoorModelParams  params_;
};

}  // namespace rc
