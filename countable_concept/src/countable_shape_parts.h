// countable_shape_parts.h — the carve parts of a countable object laid out on its CURRENT fitted box.
// Box frame: x across width w, +y toward the BACK face (countable_belief.cpp), z from the floor.
#pragma once

#include <algorithm>
#include <optional>
#include <span>
#include <vector>
#include "existence_belief/existence_belief.h"
#include "../../common/concept_manifest/shape_parts.h"

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
    std::vector<rc::exist::BoxPart> out;
    const float dz = H - z0;
    if (parts.empty()) out.push_back({0.0f, 0.0f, w, h, z0, H});
    for (const auto& p : parts)
        out.push_back({p.x_f * w, p.y_f * h, p.w_f * w, p.h_f * h, z0 + p.z0_f * dz, z0 + p.z1_f * dz});
    if (panel)
    {
        const float t = std::min(panel->thickness_m, h);
        out.push_back({0.0f, h * 0.5f - t * 0.5f, w, t, 0.0f, panel->height_m});
    }
    return out;
}

}  // namespace rc::countable
