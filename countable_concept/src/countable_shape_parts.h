// countable_shape_parts.h — the carve parts of a countable object laid out on its CURRENT fitted box.
// Box frame: x across width w, +y toward the BACK face (countable_belief.cpp), z from the floor.
#pragma once

#include <algorithm>
#include <optional>
#include <span>
#include <vector>
#include "existence_belief/existence_belief.h"
#include "shape_parts.h"

namespace rc::countable {

// Empty manifest list ⇒ one whole-box part. A back panel is appended as {0, h/2 − t/2, w, t, 0, panel.height_m}.
inline std::vector<rc::exist::BoxPart> carve_parts_for(float w, float h, float H,
                                                       std::span<const rc::manifest::ShapePart> parts,
                                                       std::optional<rc::manifest::BackPanel> panel)
{
    std::vector<rc::exist::BoxPart> out;
    if (parts.empty()) out.push_back({0.0f, 0.0f, w, h, 0.0f, H});
    for (const auto& p : parts)
        out.push_back({p.x_f * w, p.y_f * h, p.w_f * w, p.h_f * h, p.z0_f * H, p.z1_f * H});
    if (panel)
    {
        const float t = std::min(panel->thickness_m, h);
        out.push_back({0.0f, h * 0.5f - t * 0.5f, w, t, 0.0f, panel->height_m});
    }
    return out;
}

}  // namespace rc::countable
