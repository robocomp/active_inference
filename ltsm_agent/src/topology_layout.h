/*
 * topology_layout.h — the long-term memory as a topological MAP: rooms placed by their own geometry.
 *
 * Memory rooms hang from root with identity RT edges that carry no metric meaning (room_eviction.h), so the
 * only thing that relates two rooms is a passage's two door faces (seam_through_passage). The layout is a
 * breadth-first walk over the passage graph from the ANCHOR room, whose frame is the map's global frame: every
 * other room is placed in it through the seams of the passages that reach it. A component the anchor cannot
 * reach has no transform to it at all; it is drawn in its own first room's frame, packed beside.
 *
 * With a LayoutCache the walk is only the INITIAL GUESS: the poses drawn are the MAP estimate of layout_optimizer.h
 * (every seam with its covariance + the architectural priors). That solve runs only when the memory's STRUCTURE
 * changes — the set of rooms and of passages whose seam resolves — never on the display's refresh clock; between
 * changes the cached poses are reused. Read-only on the graph, Qt-free.
 */
#pragma once

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

#include "layout_optimizer.h"

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
    Mat::RTMat world_raw = Mat::RTMat::Identity();   ///< the spanning-tree placement (what `world` was before the MAP)
    std::vector<Eigen::Vector2d> outline_raw;        ///< the outline placed by world_raw
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();   ///< Laplace covariance of (x, y, yaw); zero = the gauge or no solve
    double p_manhattan = -1;                         ///< P(square to the building); −1 = not assessed
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
    double seam_d2 = -1;                    ///< Mahalanobis² of the seam at the MAP poses; −1 = no solve
    double p_inlier = -1;                   ///< P(the seam is right | MAP); −1 = no solve
    int stated_faces = 0;                   ///< faces whose RT edge states a covariance (0..2); the rest are defaulted
    [[nodiscard]] double p_pass() const { return alpha / (alpha + beta); }
};

/// What the last MAP solve found, kept across rebuilds. build() re-solves only when `key` no longer matches the graph.
struct LayoutCache
{
    std::string key;                                  ///< the structure it was solved for: rooms | resolvable passages
    std::map<std::string, layout::Vec3> pose;         ///< room → T(map <- room), before component packing; also the
                                                      ///< initial guess of the next solve (seeded from disk at start-up)
    std::map<std::string, layout::Mat3> cov;
    std::map<std::string, double> p_manhattan;
    std::map<std::string, double> seam_d2;            ///< passage → Mahalanobis² at the optimum
    std::map<std::string, double> seam_p_inlier;      ///< passage → P(seam right | optimum)
    std::map<std::string, int> stated_faces;          ///< passage → faces with a stated covariance
    std::optional<double> building_axis;              ///< ψ of the anchor's component (rad, mod π/2)
    layout::Terms before, after;                      ///< energy per family, summed over components
    int iterations = 0, rooms = 0, passages = 0;
    int starts = 0, restarted = 0;                    ///< initial guesses tried; components won by a start other than the given poses
    bool converged = true;
    double ms = 0;
    int solves = 0;
    bool fresh = false;   ///< build() just solved: the owner logs and persists it, then clears this
};

struct Map
{
    std::vector<Room> rooms;
    std::vector<Door> doors;
    std::vector<Passage> passages;
    std::optional<std::size_t> anchor;      ///< the room whose frame IS the map frame
    bool optimised = false;                 ///< `world` is the MAP estimate (else the spanning tree)
    [[nodiscard]] std::optional<std::size_t> room_index(const std::string &name) const;
};

/// Build the map from the memory graph, in the frame of the room named `anchor` -- a LIVE name such as "room_1",
/// matched against memory's r<idx>_<name> (or the exact memory name). No such room ⇒ the lowest room_id anchors.
/// `live_outlines` (memory room name → outline in that room's frame) replaces memory's snapshot: memory holds the
/// polygon a room had when it was seeded or evicted, the live room node holds the one it has NOW.
/// `cache` non-null ⇒ the MAP layout (solved when the structure changed, reused otherwise); null ⇒ the spanning tree.
/// Call on the main thread (fresh inner_eigen instances, ts == 0).
Map build(DSR::DSRGraph &mem, const std::string &anchor = "room_1",
          const std::map<std::string, std::vector<Eigen::Vector2d>> &live_outlines = {},
          LayoutCache *cache = nullptr, const layout::Model &model = {});

}   // namespace ltsm::topo
