#pragma once
/*
 * door_memory_rooms.h — what LTSM's remembered rooms say about how many doors a wall and a room have (one-model, spec §3.4/§6).
 *
 * Learned, not fixed: the per-wall rates (Beta beliefs of core/wall_base_rate.h) and the room's door-count prior
 * (core/room_door_count.h count_prior_from_rooms) are counted over the rooms LTSM remembers, EXCLUDING the room the robot is in
 * (its own doors are the evidence being judged; counting them would make the prior confirm itself).
 *
 * MEMORY SHAPE (verified in ltsm_agent/src/room_eviction.cpp, see the Task 4 report):
 *   room twin  type "room"  name r<k>_<live name>  room_id = k
 *   floor twin type "floor" name r<k>_floor...     parent = room twin
 *   wall twin  type "wall"  name r<k>_<wall>       parent = FLOOR twin (hang_walls_on_floor), RT in the floor frame
 *   door twin  an `object` with object_subtype "door", name r<k>_door_<n>: parent = its LIVE parent's twin — the wall twin when the
 *              live door hung from a wall, the floor/room twin otherwise (then: nearest wall segment in the room frame).
 * The pure part is DSR-free; define DOOR_MEMORY_ROOMS_PURE_ONLY to compile it without the DSR headers.
 */
#include "core/wall_base_rate.h"

#include <cstdint>
#include <span>
#include <vector>

#ifndef DOOR_MEMORY_ROOMS_PURE_ONLY
#include <dsr/api/dsr_api.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <string>
#endif

namespace rc::door::memrooms
{
struct RememberedWall { int doors = 0; };
struct RememberedRoom
{
    std::uint64_t room_id = 0;
    std::vector<RememberedWall> walls;
};

/// Pure: Beta counts. first: a += walls with >=1 door, b += walls with 0; another: over walls with >=1 door,
/// a += walls with >=2, b += walls with exactly 1.
[[nodiscard]] inline core::WallRates rates_from_rooms(std::span<const RememberedRoom> rooms, core::WallRates seed = {})
{
    for (const auto& r : rooms)
        for (const auto& w : r.walls)
        {
            if (w.doors >= 1) seed.first.a += 1.0; else seed.first.b += 1.0;
            if (w.doors >= 2) seed.another.a += 1.0;
            else if (w.doors == 1) seed.another.b += 1.0;
        }
    return seed;
}

/// Doors per room (all its walls), for count_prior_from_rooms (which ignores rooms with none).
[[nodiscard]] inline std::vector<int> door_counts(std::span<const RememberedRoom> rooms)
{
    std::vector<int> out;
    out.reserve(rooms.size());
    for (const auto& r : rooms)
    {
        int n = 0;
        for (const auto& w : r.walls) n += w.doors;
        out.push_back(n);
    }
    return out;
}

#ifndef DOOR_MEMORY_ROOMS_PURE_ONLY
namespace detail
{
struct Seg { std::uint64_t id; Eigen::Vector2f mid; float yaw, half; };

/// name "r<digits>_..." -> its "r<digits>_" prefix; empty for anything else (a live room "room_1" is not a memory twin).
inline std::string memory_prefix(const std::string& name)
{
    if (name.size() < 3 or name[0] != 'r') return {};
    std::size_t i = 1;
    while (i < name.size() and name[i] >= '0' and name[i] <= '9') ++i;
    if (i == 1 or i >= name.size() or name[i] != '_') return {};
    return name.substr(0, i + 1);
}

/// RT translation xy + yaw of the edge parent -> child (the pose in the parent's frame), read off the edge attributes.
inline std::optional<std::pair<Eigen::Vector2f, float>> edge_pose(DSR::DSRGraph& G, std::uint64_t parent, std::uint64_t child)
{
    const auto e = G.get_edge(parent, child, "RT");
    if (not e.has_value()) return std::nullopt;
    const auto t = G.get_attrib_by_name<rt_translation_att>(e.value());
    const auto r = G.get_attrib_by_name<rt_rotation_euler_xyz_att>(e.value());
    if (not (t.has_value() and r.has_value() and t.value().get().size() >= 3 and r.value().get().size() >= 3)) return std::nullopt;
    return std::pair{Eigen::Vector2f(t.value().get()[0], t.value().get()[1]), r.value().get()[2]};
}

inline bool is_door_twin(DSR::DSRGraph& G, const DSR::Node& n, const std::string& prefix)
{
    if (G.get_attrib_by_name<object_subtype_att>(n).value_or(std::string{}) == "door") return true;
    return n.name().starts_with(prefix + "door_");
}
}   // namespace detail

/// Graph reader: LTSM memory room twins (type "room", name prefix r<k>_), their wall twins (type "wall") and door twins,
/// EXCLUDING the memory room of `current_room_idx` (the live room's room_id: the twin carries the same room_id; nullopt = a
/// fresh, unrecognised room, nothing excluded). A door is assigned to the wall twin it hangs under; one parented to the
/// floor/room goes to the nearest wall segment in the memory room frame (floor -> room is the identity RT).
[[nodiscard]] inline std::vector<RememberedRoom> read_remembered_rooms(DSR::DSRGraph& G, std::optional<std::uint64_t> current_room_idx)
{
    std::vector<RememberedRoom> out;
    const auto walls_all = G.get_nodes_by_type("wall");
    const auto objects   = G.get_nodes_by_type("object");
    for (const auto& room : G.get_nodes_by_type("room"))
    {
        const std::string prefix = detail::memory_prefix(room.name());
        const auto idx = G.get_attrib_by_name<room_id_att>(room);
        if (prefix.empty() or not idx.has_value()) continue;
        if (current_room_idx.has_value() and idx.value() == *current_room_idx) continue;

        // this room's wall twins, with their segment in the room frame
        std::vector<detail::Seg> segs;
        std::map<std::uint64_t, std::size_t> seg_of;
        for (const auto& w : walls_all)
        {
            if (not w.name().starts_with(prefix)) continue;
            const auto parent = G.get_attrib_by_name<parent_att>(w);
            if (not parent.has_value()) continue;
            const auto pose = detail::edge_pose(G, parent.value(), w.id());
            if (not pose.has_value()) continue;
            seg_of[w.id()] = segs.size();
            segs.push_back({w.id(), pose->first, pose->second, 0.5f * G.get_attrib_by_name<width_m_att>(w).value_or(1.0f)});
        }
        RememberedRoom rr;
        rr.room_id = idx.value();
        rr.walls.resize(segs.size());
        for (const auto& d : objects)
        {
            if (not d.name().starts_with(prefix) or not detail::is_door_twin(G, d, prefix)) continue;
            const auto parent = G.get_attrib_by_name<parent_att>(d);
            if (not parent.has_value()) continue;
            if (const auto it = seg_of.find(parent.value()); it != seg_of.end())
            {
                ++rr.walls[it->second].doors;
                continue;
            }
            const auto pose = detail::edge_pose(G, parent.value(), d.id());   // in the floor/room frame = the room frame
            if (not pose.has_value() or segs.empty()) continue;
            std::size_t best = 0;
            float best_d2 = std::numeric_limits<float>::max();
            for (std::size_t i = 0; i < segs.size(); ++i)
            {
                const Eigen::Vector2f dir(std::cos(segs[i].yaw), std::sin(segs[i].yaw));
                const float proj = std::clamp((pose->first - segs[i].mid).dot(dir), -segs[i].half, segs[i].half);
                const float d2 = (pose->first - (segs[i].mid + proj * dir)).squaredNorm();
                if (d2 < best_d2) { best_d2 = d2; best = i; }
            }
            ++rr.walls[best].doors;
        }
        out.push_back(std::move(rr));
    }
    return out;
}
#endif   // DOOR_MEMORY_ROOMS_PURE_ONLY
}   // namespace rc::door::memrooms
