/*
 * topology_layout.h — the long-term memory as a topological MAP: rooms placed by their own geometry.
 *
 * Memory rooms hang from root with identity RT edges that carry no metric meaning (room_eviction.h), so the
 * only thing that relates two rooms is a passage's two door faces (seam_through_passage). The layout is a
 * breadth-first walk over the passage graph from the ANCHOR room, whose frame is the map's global frame: every
 * other room is placed in it through the seams of the passages that reach it. A component the anchor cannot
 * reach has no transform to it at all; it is drawn in its own first room's frame, packed beside.
 * Read-only on the graph, Qt-free.
 */
#pragma once

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

#include <Eigen/Dense>

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace ltsm::topo
{

struct Room
{
    std::string name;
    std::vector<Eigen::Vector2d> outline;   ///< world frame of the map
    Eigen::Vector2d centre{0, 0};           ///< area centroid of the outline (world)
    Mat::RTMat world = Mat::RTMat::Identity();   ///< T(map <- room)
    int component = 0;
};

struct Door
{
    std::string name;
    std::size_t room = 0;                   ///< index into Map::rooms
    Eigen::Vector2d a{0, 0}, b{0, 0};       ///< the aperture's two ends (world), width_m along the door's x
    Eigen::Vector2d centre{0, 0};
};

struct Passage
{
    std::string name;
    std::optional<std::size_t> room_a, room_b;   ///< indices into Map::rooms
    std::optional<std::size_t> door_a, door_b;   ///< indices into Map::doors
    double alpha = 1, beta = 1;             ///< Beta over p(passable without intervention)
    int crossings = 0;
    bool seam_ok = false;                   ///< false ⇒ the two faces do not compose (drawn dashed)
    bool loop = false;                      ///< joins two rooms already placed through other passages
    double closure_m = 0;                   ///< loop only: distance between the two sides' door centres
    [[nodiscard]] double p_pass() const { return alpha / (alpha + beta); }
};

struct Map
{
    std::vector<Room> rooms;
    std::vector<Door> doors;
    std::vector<Passage> passages;
    std::optional<std::size_t> anchor;      ///< the room whose frame IS the map frame
    [[nodiscard]] std::optional<std::size_t> room_index(const std::string &name) const;
};

/// Build the map from the memory graph, in the frame of the room named `anchor` -- a LIVE name such as "room_1",
/// matched against memory's r<idx>_<name> (or the exact memory name). No such room ⇒ the lowest room_id anchors.
/// `live_outlines` (memory room name → outline in that room's frame) replaces memory's snapshot: memory holds the
/// polygon a room had when it was seeded or evicted, the live room node holds the one it has NOW.
/// Call on the main thread (fresh inner_eigen instances, ts == 0).
Map build(DSR::DSRGraph &mem, const std::string &anchor = "room_1",
          const std::map<std::string, std::vector<Eigen::Vector2d>> &live_outlines = {});

}   // namespace ltsm::topo
