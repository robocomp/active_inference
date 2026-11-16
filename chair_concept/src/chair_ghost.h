// chair_ghost.h — the stand-in chair at a remembered chair's old spot (changed-object spec): what the camera WOULD
// see there if the chair still stood on it. ghost_p_vis projects it to weigh "nothing there" by P(detect).
#pragma once

#include "chair_model.h"

namespace rc
{
/// A prior-sized chair at (x, y), yaw 0, in the room frame.
[[nodiscard]] inline ChairModel ghost_model(float x, float y, const ChairModelParams& params)
{
    ChairState st;
    st.seed_size(params.chair);
    st.cx = x;
    st.cy = y;
    st.yaw = 0.0f;
    // Built FROM the prior: a default ChairModel has all-zero bounds and clamps every dimension to 0 (live run
    // moved_chair_2 — a zero-size ghost projected to fill 0, P(detect) 0, so the old spot never gathered evidence).
    return ChairModel(st, params);
}
}   // namespace rc
