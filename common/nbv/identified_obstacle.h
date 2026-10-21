/*
 * common/nbv/identified_obstacle.h — ONE object/box node as a named viewpoint obstacle: the pure half of
 * graph_obstacles.h. Header-only, Eigen only (no DSR).
 *
 * ★WHY IT IS SPLIT OUT (door_concept DoorCore, 2026-10-21). door_concept's birth evidence is weighted by the
 * other concepts' claims on the room (common/exclusion), and that rule is being moved into a DSR-free core that
 * replays recorded inputs. The core receives each node's attributes and its room-frame pose as plain data and
 * must turn them into the SAME footprint the graph reader builds — so the arithmetic lives here, once, and both
 * collect_graph_obstacles_identified (live, from the graph) and the core (from a recorded snapshot) call it.
 * A second copy of these five lines is exactly the drift graph_obstacles.h's own header warns about.
 */
#pragma once

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>

#include <Eigen/Dense>

#include "viewpoint_score.h"   // rc::nbv::Obstacle

namespace rc::nbv
{

struct IdentifiedObstacle
{
    Obstacle      fp{};
    std::string   name;
    std::uint64_t id = 0;
    float         z0 = 0.0f;   // base, room frame (m)
    float         z1 = 0.0f;   // top; <= z0 ⇒ height unknown ⇒ treat as unbounded
};

/// Does a node with these width_m / depth_m attributes have a footprint at all? (Both present and positive —
/// the TRUE extents; see graph_obstacles.h for the two silent bugs this rule fixed.)
[[nodiscard]] inline bool has_footprint(const std::optional<float>& w, const std::optional<float>& d)
{
    return w.has_value() and d.has_value() and w.value() > 0.0f and d.value() > 0.0f;
}

/// The obstacle of a node of TRUE extents w × d (has_footprint), optional height_m, whose pose in the room frame is
/// `room_T_node` (the ts=0 transform the graph reader asks cortex for).
[[nodiscard]] inline IdentifiedObstacle identified_obstacle(std::string name, std::uint64_t id, float w, float d,
                                                            const std::optional<float>& h,
                                                            const Eigen::Matrix4d& room_T_node)
{
    const auto& M = room_T_node;
    // Vertical band: origin is the BASE by fleet convention, top = base + height_m. A node with no
    // height_m leaves z1 == z0, which every consumer reads as "unknown ⇒ spans all heights".
    const float z0 = static_cast<float>(M(2, 3));
    const float z1 = (h.has_value() and h.value() > 0.0f) ? z0 + h.value() : z0;
    return {{static_cast<float>(M(0, 3)), static_cast<float>(M(1, 3)), w, d,
             std::atan2(static_cast<float>(M(1, 0)), static_cast<float>(M(0, 0)))},
            std::move(name), id, z0, z1};
}

}  // namespace rc::nbv
