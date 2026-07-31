/*
 * door_model.h
 *
 * Geometry / state container for a door instance.
 *
 * The recursive belief update lives in door_belief.* (the AI2 full-covariance filter over the wall-frame
 * θ=[s,w,h]), and ALL panel geometry lives in door_geometry.h (rc::door — the single source of truth for
 * the aperture/leaf split). This class is only the ROOM-FRAME state holder that DoorFitter::refresh_geometry
 * writes once per cycle, and which the scene-graph publish, the display mesh and the tracker read back.
 *
 * State (room frame): LEAF centre (cx,cy), floor base cz (panel spans [cz, cz+h]), leaf heading yaw, width
 * w (hinge → free edge), height h, thickness; plus the leaf's opening angle phi and the APERTURE pose
 * (ap_cx, ap_cy, ap_yaw). At phi = 0 the leaf is flush in the aperture and the two poses coincide exactly.
 */

#pragma once

#include <cmath>
#include <Eigen/Dense>

#include "door_geometry.h"      // rc::door:: — the single source of truth for panel geometry

namespace rc {

// Room-frame read-back of the fitted door, written once per cycle by DoorFitter::refresh_geometry.
//
// It carries BOTH halves of the aperture/leaf split (see door_geometry.h), because different consumers
// legitimately need different ones:
//   · cx, cy, yaw  → the LEAF, wherever it currently is. Used by the display mesh, the projected ROI, the
//                    tracker's association centre and the voxel-ownership gate — all of which want the
//                    thing the sensor actually sees.
//   · ap_*         → the APERTURE, the static hole in the wall. Used by the DSR RT edge, resolve_wall,
//                    the merge footprint, ghost identity and the room-containment prior — all of which
//                    must NOT be dragged by a swinging leaf.
// With phi pinned at 0 (M0) the two coincide exactly, so every consumer is unchanged.
struct DoorState
{
    float cx        = 0.0f;   // room-frame X of the LEAF centre
    float cy        = 0.0f;   // room-frame Y of the LEAF centre
    float cz        = 0.0f;   // floor base height (panel spans [cz, cz+h]); pinned, not fit
    float yaw       = 0.0f;   // heading about Z of the LEAF (at phi=0 this is the wall tangent)
    float w         = 0.70f;  // panel width  (local X, hinge → free edge)
    float h         = 2.00f;  // panel height (vertical)
    float thickness = 0.05f;  // panel thickness (local Y, across the panel face — fixed)
    // ── Aperture + articulation (M0) ──
    float phi       = 0.0f;   // leaf opening angle (rad); 0 = flush in the aperture. Pinned in M0.
    float ap_cx     = 0.0f;   // room-frame X of the APERTURE centre (rigid in the wall)
    float ap_cy     = 0.0f;   // room-frame Y of the APERTURE centre
    float ap_yaw    = 0.0f;   // room-frame yaw of the APERTURE = the wall tangent
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

    // NOTE: this class no longer owns an SDF. There used to be a second, independent panel SDF here
    // (sdf_point/sdf_point_at) alongside DoorBelief::sdf_panel, and the two could — and did — disagree
    // about where the panel is. Both now route through rc::door::leaf_sdf (door_geometry.h). The one
    // caller, DoorFitter::observe's candidate/residual split, uses the instance's cached leaf_pose.

    // ── State access ─────────────────────────────────────────────────────────
    const DoorState& state()  const { return state_; }
    const DoorState& prior()  const { return prior_; }
    const DoorModelParams& params() const { return params_; }

    void set_state(const DoorState& s) { state_ = s; apply_constraints(); }
    void set_prior(const DoorState& p) { prior_ = p; }

    /** Enforce physical positivity on w,h; leave cx,cy,cz,yaw,thickness as written by the belief. */
    void apply_constraints();

private:
    DoorState        state_;
    DoorState        prior_;
    DoorModelParams  params_;
};

}  // namespace rc
