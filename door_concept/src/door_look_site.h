// door_look_site.h — WHICH of the door's viewpoints to offer for the epistemic look, given the standpoints the consumer has
// handed back (D4). Header-only, Eigen only: tests/test_door_look_site.cpp.
//
// ★THE LOOP THIS ENDS. The epistemic look (aff_door_<id>, the object-relative viewpoint) is chosen by the NBV planner from the
// door's belief, walls and graph furniture. The occupancy the CONTROLLER plans on (a bed, a person) is not in that set, so the
// look could be offered at a place the body cannot stand — gold13: (-0.07, 2.82), inside Bed(1) — the controller found no route
// and spun. The approach already had the cure (door_approach_pose.h: a hand-back is decaying evidence that its site is blocked);
// the look had none. Same model, same memory type (ApproachMemory, in the door's frame), same τ.
// ★THE MODEL. Each face's expected gain G_i (detection-weighted, existence included) is multiplied by P(site i standable) =
// 1 − P(refused | site i): continuous in the site and in the age of the refusal, no gate. The face with the best weighted value
// is offered, at that weighted value. A look whose every usable face is a remembered hand-back (value exactly 0 — a re-offer at
// the very site that was handed back, which the planner reproduces deterministically) is UNPLACEABLE: it is withdrawn, and
// comes back by itself as the refusals decay (the weight rises continuously from 0).
#pragma once

#include <array>
#include <cmath>

#include <Eigen/Dense>

namespace rc::door
{
struct LookSite
{
    int   face = -1;            // the face to offer; -1 when none is usable at all
    float value = 0.0f;         // its gain × P(standable)
    float p_refused = 0.0f;     // P(refused | that site)
    bool  unplaceable = false;  // usable faces exist, but every one is a remembered hand-back
};

// total[i]: face i's expected gain (meaningful only if usable[i]); pos[i]: its viewpoint; p_refused(q): the memory.
template <class Refused>
[[nodiscard]] inline LookSite choose_look_site(const std::array<float, 4>& total, const std::array<bool, 4>& usable,
                                               const std::array<Eigen::Vector2f, 4>& pos, Refused&& p_refused)
{
    LookSite out;
    bool any_free_value = false;   // some usable face is worth something with nothing remembered
    float best = -1.0f;
    for (int i = 0; i < 4; ++i)
    {
        if (not usable[i]) continue;
        const float pr = std::clamp(static_cast<float>(p_refused(pos[i])), 0.0f, 1.0f);
        const float v = total[i] * (1.0f - pr);
        any_free_value = any_free_value or total[i] > 0.0f;
        if (v > best) { best = v; out.face = i; out.value = v; out.p_refused = pr; }
    }
    out.unplaceable = any_free_value and not (out.value > 0.0f);
    return out;
}
}  // namespace rc::door
