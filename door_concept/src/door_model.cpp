/*
 * door_model.cpp
 *
 * Room-frame state container for a door instance. The recursive belief update lives in door_belief.*
 * (AI2), and ALL panel geometry — SDF, silhouette samples, ROI corners, mesh vertices, footprints —
 * lives in door_geometry.h (rc::door). This file is now only the state holder plus its positivity
 * constraints; it deliberately owns no SDF of its own (see the note in door_model.h).
 */

#include "door_model.h"

#include <algorithm>

namespace rc {

// ─── DoorModel ──────────────────────────────────────────────────────────────

DoorModel::DoorModel(const DoorState& prior, const DoorModelParams& params)
    : state_(prior), prior_(prior), params_(params)
{
    apply_constraints();
}

// ─── Constraints ─────────────────────────────────────────────────────────────

void DoorModel::apply_constraints()
{
    // The belief (door_belief) owns the strong w/h priors; here just enforce physical positivity so a
    // stray write can't invert a panel dimension. thickness/cz/pose are as written by the belief.
    state_.w = std::max(0.05f, state_.w);
    state_.h = std::max(0.05f, state_.h);
    state_.thickness = std::max(0.005f, state_.thickness);
}

}  // namespace rc
