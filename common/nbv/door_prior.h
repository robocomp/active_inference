#pragma once
// common/nbv/door_prior.h — the door a camera look is planned for, when no particular door is known yet.
//
// One door geometry for every agent that asks "could the detector have found a door HERE?": room_concept's camera
// coverage of a room's boundary (room_concept/src/camera_coverage.h) and — follow-up, NOT migrated yet — door_concept's
// look at a remembered door with no fitted pose (door_concept/src/door_look.h Params prior_w_m/prior_h_m/thickness_m,
// the values copied here).
// ⚠ door_concept's fit prior (door_config.h Door.PriorWidthM/PriorHeightM = 1.00 x 2.10 m) is a DIFFERENT number for a
//   different question (the belief's prior mean on a detected leaf); reconciling the two is part of that follow-up.
namespace rc::nbv
{
struct DoorPrior
{
    float w_m = 0.90f;   ///< leaf width: the window a detection needs along the wall
    float h_m = 2.00f;   ///< leaf height (floor to head)
    float t_m = 0.05f;   ///< leaf/wall thickness: occluder depth only (occlusion by a plane barely depends on it)
};
}   // namespace rc::nbv
