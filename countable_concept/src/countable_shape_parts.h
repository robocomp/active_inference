// countable_shape_parts.h — the carve parts of a countable object laid out on its CURRENT fitted box.
// Box frame: x across width w, +y toward the BACK face (countable_belief.cpp), z from the floor.
// The geometry lives in common/ai_belief/parts_geometry.h (K2a: one table gives the camera SDF, the beam parts and this carve);
// this wrapper keeps the manifest-struct signature the existence channel and the carve replay tool call.
#pragma once

#include <optional>
#include <span>
#include <vector>
#include "existence_belief/existence_belief.h"
#include "../../common/concept_manifest/shape_parts.h"
#include "../../common/ai_belief/parts_geometry.h"

namespace rc::countable {

// Empty manifest list ⇒ one whole-box part. A back panel is appended as {0, h/2 − t/2, w, t, 0, panel.height_m}.
// z0 = the body's underside (K1, VerticalSpan): 0 for a floor-anchored class (bit-identical: 0 + f·(H − 0) = f·H), H − e for a
// hanging one (the hood). The whole box spans [z0, H] and the parts' z fractions are scaled over that band, never over the
// floor → top column (a hood's column would claim the worktop and the hob under it). The back panel's height is absolute
// (from the floor), as the manifest declares it.
inline std::vector<rc::exist::BoxPart> carve_parts_for(float w, float h, float H,
                                                       std::span<const rc::manifest::ShapePart> parts,
                                                       std::optional<rc::manifest::BackPanel> panel,
                                                       float z0 = 0.0f)
{
    rc::ai::parts::Table T;
    T.parts = parts;
    if (panel) { T.panel_h_m = panel->height_m; T.panel_t_m = panel->thickness_m; }
    return rc::ai::parts::carve_parts(w, h, H, T, z0);
}

}  // namespace rc::countable
