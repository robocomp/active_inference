/*
 * ★ TEST REFERENCE ONLY (K3, 2026-11-26). This was common/nbv/standpoint_free.h, Task 8's object-side MIRROR of the controller's
 * feasibility test. No agent includes it any more: the countables' p_stand now calls the SHARED standpoint oracle
 * (common/standpoint/standpoint_oracle.h), which IS the controller's rasterisation and fit. It is kept, unchanged below this note,
 * as the reference p_stand_parity_test.cpp measures the switch against.
 *
 * common/nbv/standpoint_free.h — can the robot's BODY stand at an offered standpoint? (G6)  p_stand ∈ {0, 1}.
 *
 * WHY. The controller hands back an epistemic standpoint when "a wall, object or unbelieved residual cell lies under the
 * body in the turn it ends with" (controller_session.cpp, decide_standpoint; Step 0 census: 44 of 55 refusals). The object
 * agent used to offer such a standpoint at full value. The goal here is that the object and the controller AGREE on where the
 * robot can stand, so this reproduces the controller's own test on the same published data and is a deterministic check:
 *
 *     the footprint, at each heading it will occupy, covers no cell that is off the room, inside an obstacle polygon, or
 *     a HARD latched cell of residual's (one the published belief field does not cover).
 *
 * It mirrors GridPlanner: the body is RobotFootprint (placed by its own cell_offsets_yaw, the quarter turn applied once there),
 * the heading is bucketed to kHeadings = 8 as GridPlanner::heading_bucket does, and a pose that ends in a turn in place
 * (every non-Reach contract: Servo's facing turn, the lock-on search, Orient) is judged over ALL 8 headings, like can_turn_here.
 *
 * WHAT IT IS NOT. (1) No probability: residual's grid_occupancy_prob is P(occupied ∧ ¬explained) with a Beta-mean floor per
 * cell, so a product over it is 0.33 at best and ~1e-27 over unexplored ground — that was the rejected first draft. The
 * controller's own verdict is hard/not-hard, so this is too. (2) No safety margin: the controller adds
 * footprint_safety_margin_m, which is another agent's config and is not readable here, so the object side uses margin 0 and is
 * therefore SLIGHTLY MORE PERMISSIVE than the controller (exposing the margin is an owner decision). (3) Latched cells are
 * split exactly as the controller splits them (OccupiedCells): only a cell the published field does not cover is hard. (4) Cells are
 * rasterised on residual's lattice, the controller's on its own. (5) The controller also accepts a door-passage polygon as room
 * (GridPlanner::inside_room_or_passage); passages come from its active crossing state, which the object does not hold, so a
 * standpoint inside a passage but outside the room polygon is refused here.
 *
 * Pure Eigen/std + robot_footprint (ai_common::geometry). Pinned by tests/standpoint_free_test.cpp.
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>

#include "../../nbv/viewpoint_score.h"                  // Obstacle, stands_inside, detail::point_in_polygon
#include "../../robot_footprint/robot_footprint.h"     // rc::RobotFootprint
#include "../residual_risk.h"                         // rc::residual_risk — the controller's own cell_risk_nats / risks_from_field (pure std+Eigen)

namespace rc::nbv
{

inline constexpr int kStandHeadings = 8;   // GridPlanner::kHeadings

// The body's covered-cell offsets at each of the 8 headings, for one cell size.
struct StandBody
{
    float cell_m = 0.0f;
    std::array<std::vector<Eigen::Vector2i>, kStandHeadings> offsets;

    static StandBody of(const RobotFootprint& body, float cell_m)
    {
        StandBody b;
        if (not (cell_m > 0.0f)) return b;
        b.cell_m = cell_m;
        RobotFootprint bare = body;
        bare.set_safety_margin(0.0f);   // see the header: the controller's margin is not ours to read
        for (int h = 0; h < kStandHeadings; ++h)
            b.offsets[static_cast<std::size_t>(h)] =
                bare.cell_offsets_yaw(cell_m, 2.0f * static_cast<float>(M_PI) * static_cast<float>(h) / kStandHeadings);
        return b;
    }
    bool valid() const { return cell_m > 0.0f; }
};

// residual's latched occupied cells (grid_occupied_cells centres) on their own lattice — ONLY the HARD ones.
// The controller prices a latched cell the published field covers (residual_risk.h cell_risk_nats: a toll, not a wall) and hands a
// standpoint back only for a cell the field does NOT cover (outside it, or no usable moments) or whose risk is not finite. The same
// partition is made here by calling the controller's own risks_from_field, so a soft cell is simply not stored. A collapsed cell
// (P <= 0: explained by a modelled object, or speckle) has risk 0 there and is soft here too.
struct OccupiedCells
{
    float cell_m = 0.0f;
    Eigen::Vector2f origin{0.0f, 0.0f};          // a cell centre: the lattice is origin + k·cell_m
    std::unordered_set<std::int64_t> keys;

    static std::int64_t key(std::int64_t i, std::int64_t j) { return (i << 32) ^ (j & 0xffffffffLL); }
    int index(float v, float o) const { return static_cast<int>(std::lround((v - o) / cell_m)); }

    // `xyz` = the attribute's flat [x,y,z] triples; `field` = residual's belief field (an invalid one makes every cell hard, as in the
    // controller). The lattice origin is the first latched cell, hard or not.
    static OccupiedCells from_xyz(std::span<const float> xyz, float cell_m, const residual_risk::BeliefField& field)
    {
        OccupiedCells c;
        if (not (cell_m > 0.0f)) return c;
        c.cell_m = cell_m;
        if (xyz.size() >= 3) c.origin = {xyz[0], xyz[1]};
        std::vector<Eigen::Vector2f> centres;
        centres.reserve(xyz.size() / 3);
        for (std::size_t i = 0; i + 2 < xyz.size(); i += 3) centres.emplace_back(xyz[i], xyz[i + 1]);
        const auto risk = residual_risk::risks_from_field(centres, field);
        for (std::size_t k = 0; k < centres.size(); ++k)
            if (not std::isfinite(risk[k]))
                c.keys.insert(key(c.index(centres[k].x(), c.origin.x()), c.index(centres[k].y(), c.origin.y())));
        return c;
    }
    bool has(int i, int j) const { return keys.contains(key(i, j)); }
};

// Everything the check reads. Empty room / obstacles / cells each mean "no constraint of that kind" (the planner's convention).
struct StandWorld
{
    const StandBody*                 body = nullptr;
    const OccupiedCells*             occupied = nullptr;
    std::span<const Eigen::Vector2f> room;
    std::span<const Obstacle>        obstacles;
    bool valid() const { return body != nullptr and body->valid(); }
};

namespace detail
{
inline int stand_heading(float yaw)
{
    const float two_pi = 2.0f * static_cast<float>(M_PI);
    float t = std::fmod(yaw, two_pi);
    if (t < 0.0f) t += two_pi;
    return static_cast<int>(std::lround(t / two_pi * kStandHeadings)) % kStandHeadings;
}
}   // namespace detail

// GridPlanner::pose_free (one heading) / can_turn_here (all headings): no covered cell is off the room, inside an obstacle, or latched.
inline bool pose_free(const StandWorld& w, const Eigen::Vector2f& pos, int heading)
{
    const auto& offs = w.body->offsets[static_cast<std::size_t>(heading)];
    const float c = w.body->cell_m;
    const Eigen::Vector2f o = w.occupied != nullptr ? w.occupied->origin : Eigen::Vector2f(0.0f, 0.0f);
    const int ix = static_cast<int>(std::lround((pos.x() - o.x()) / c)), iy = static_cast<int>(std::lround((pos.y() - o.y()) / c));
    for (const auto& d : offs)
    {
        const int i = ix + d.x(), j = iy + d.y();
        const Eigen::Vector2f centre{o.x() + static_cast<float>(i) * c, o.y() + static_cast<float>(j) * c};
        if (not w.room.empty() and not detail::point_in_polygon(centre, w.room)) return false;
        if (stands_inside(centre, w.obstacles)) return false;
        if (w.occupied != nullptr and w.occupied->has(i, j)) return false;
    }
    return true;
}

// 1 if the body can stand at `pos` facing `yaw` (forward = (cos yaw, sin yaw)) — through a turn in place when `turn` — else 0.
// An unusable world carries no belief and cannot veto: 1 (the controller's "no belief yet: no verdict").
inline double p_stand(const StandWorld& w, const Eigen::Vector2f& pos, float yaw, bool turn)
{
    if (not w.valid()) return 1.0;
    if (not turn) return pose_free(w, pos, detail::stand_heading(yaw)) ? 1.0 : 0.0;
    for (int h = 0; h < kStandHeadings; ++h)
        if (not pose_free(w, pos, h)) return 0.0;
    return 1.0;
}

}   // namespace rc::nbv
