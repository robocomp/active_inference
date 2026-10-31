// chair_mesh.h -- the chair's display mesh (seat slab + backrest + 4 square legs), header-only so a unit test can
// build it without Qt / DSR. Every fixed dimension (seat thickness, leg half-side) comes from the ChairPrior.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "chair_model.h"

namespace rc
{
// Flat triangle list (room frame): seat slab + backrest + 4 legs.
[[nodiscard]] inline std::vector<float> make_chair_mesh(const ChairState& s, const ChairPrior& prior)
{
    // Flat triangle list (room frame): seat slab + backrest + 4 square legs = 6 boxes × 12 tri.
    std::vector<float> verts;
    verts.reserve(6 * 108);

    const float cy = std::cos(s.yaw);
    const float sy = std::sin(s.yaw);

    auto push_box = [&](float bx, float by, float bz, float hw, float hd, float hh)
    {
        auto push = [&](float lx, float ly, float lz)
        {
            verts.push_back(bx + cy * lx - sy * ly);
            verts.push_back(by + sy * lx + cy * ly);
            verts.push_back(bz + lz);
        };
        push(-hw,-hd,-hh); push( hw,-hd,-hh); push( hw, hd,-hh);  // bottom
        push(-hw,-hd,-hh); push( hw, hd,-hh); push(-hw, hd,-hh);
        push(-hw,-hd, hh); push( hw, hd, hh); push( hw,-hd, hh);  // top
        push(-hw,-hd, hh); push(-hw, hd, hh); push( hw, hd, hh);
        push(-hw,-hd,-hh); push( hw,-hd,-hh); push( hw,-hd, hh);  // front -y
        push(-hw,-hd,-hh); push( hw,-hd, hh); push(-hw,-hd, hh);
        push( hw, hd,-hh); push(-hw, hd,-hh); push(-hw, hd, hh);  // back  +y
        push( hw, hd,-hh); push(-hw, hd, hh); push( hw, hd, hh);
        push(-hw,-hd,-hh); push(-hw,-hd, hh); push(-hw, hd, hh);  // left  -x
        push(-hw,-hd,-hh); push(-hw, hd, hh); push(-hw, hd,-hh);
        push( hw,-hd,-hh); push( hw, hd,-hh); push( hw, hd, hh);  // right +x
        push( hw,-hd,-hh); push( hw, hd, hh); push( hw,-hd, hh);
    };

    const float hw = s.seat_w * 0.5f;
    const float hd = s.seat_d * 0.5f;
    const float ht = prior.seat_thick * 0.5f;

    // Seat slab — its TOP face sits at cz + seat_h, so its centre is half a thickness below.
    push_box(s.cx, s.cy, s.cz + s.seat_h - ht, hw, hd, ht);

    // Backrest — a thin slab along the -y edge (local), rising back_h above the seat top.
    const float back_cy = -hd + ht;               // local y of backrest centre
    const float back_cz = s.cz + s.seat_h + s.back_h * 0.5f;
    {
        const float brx = s.cx + cy * 0.0f - sy * back_cy;
        const float bry = s.cy + sy * 0.0f + cy * back_cy;
        push_box(brx, bry, back_cz, hw, ht, s.back_h * 0.5f);
    }

    // 4 legs — square cross-section (2×leg_half), at the seat corners, floor → seat underside.
    const float lr  = prior.leg_half;   // one manifest value, shared with the SDF, the belief and the card
    const float leg_top = s.seat_h - prior.seat_thick;   // underside of the seat slab
    const float lhz = std::max(0.01f, leg_top) * 0.5f;
    for (int ix : {-1, 1})
        for (int iy : {-1, 1})
        {
            const float lx = ix * (hw - lr);
            const float ly = iy * (hd - lr);
            const float rx = s.cx + cy * lx - sy * ly;
            const float ry = s.cy + sy * lx + cy * ly;
            push_box(rx, ry, s.cz + lhz, lr, lr, lhz);
        }

    return verts;
}
}  // namespace rc
