/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp — see room_scene_graph.h.
 */
#pragma once

// ── THE LAYOUT EXPLORER, LIVE — a PORT of the offline one, not a re-design ───────────────────────────────
//
// While the room's layout is still being learnt (start-up proto-room, Estimate mode) the robot is driven to
// the place a scan would teach the layout the most, per metre of travel. That objective was built and
// validated OFFLINE in tools/wall_slam_selftest.cpp (run_explore: the feasibility grid + A*, the expected
// information gain `eig_of`, the selection over the best 16 candidates, the termination ledger
// `collect_entropy_targets`). This file is that code moved verbatim into a pure, live-callable unit, so the
// agent runs what the bench validated (memory: bench-must-run-the-agents-code). Do NOT re-tune it here;
// tools/layout_explorer_selftest.cpp checks the port picks the SAME target as a frozen copy of the offline
// code on the same state.
//
// WHAT CHANGES LIVE, and only this:
//   · FEASIBILITY. Offline, the grid of standpoints is the TRUE room (the simulator's job). Live there is no
//     truth: the grid is built from the published start-up proto polygon — confirmed free space, already
//     eroded by the body (free_space_polygon.h) — so a target can never be outside what the controller is
//     handed (controller_session.cpp marks such targets OutsideRoom). `clearance` is then measured from that
//     ring, not from a wall; RoomSceneGraph passes the body's circumscribed radius so any yaw fits.
//   · THE START CELL. Offline the robot only ever stands on its own A* paths, so its cell is always free.
//     Live it can stand anywhere the controller put it; if its cell is not a feasible standpoint the walk
//     starts from the nearest one that is (otherwise the explorer would return nothing for ever).
//   · ONE FRAME. Offline, candidates live in the truth frame and are mapped into the map frame (`to_map`).
//     Live everything is already in the estimator's internal frame.
// Nothing else: same 36-beam expected-gain model through the occupancy grid and the estimated polygon, same
// wall dwell term ½·ln(det(Λ+J)/det Λ), same corner σ-ratio term, same gain/(1+d) ranking, same top-16
// reachability fallback, same "finished only with a closed polygon and nothing left to learn".

#include <Eigen/Dense>

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "wall_map.h"

namespace rc::layout
{

struct ExplorerParams
{
    float cell       = 0.30f;   ///< m — standpoint grid (offline Explorer::cell)
    float clearance  = 0.35f;   ///< m — from the feasibility polygon's boundary (offline: from the true walls)
    int   beams      = 36;      ///< expected-gain ray fan (offline NB)
    bool  occl_pred  = true;    ///< beams stop at matter the grid holds (offline WS_NO_OCCL_PRED unset)
    bool  dwell      = true;    ///< wall information term (offline WS_NO_DWELL unset)
    int   keep_best  = 16;      ///< candidates tried for reachability, best first
};

/// Standpoints a robot may occupy: a regular grid over a polygon, `clearance` inside it. Offline `Explorer`.
class FeasibleGrid
{
public:
    FeasibleGrid(const std::vector<Eigen::Vector2f>& poly, float cell, float clearance);
    [[nodiscard]] Eigen::Vector2f at(int i, int j) const { return {x0 + (i + 0.5f) * cell, y0 + (j + 0.5f) * cell}; }
    [[nodiscard]] bool is_free(int i, int j) const
    { return i >= 0 and i < nx and j >= 0 and j < ny and free_[static_cast<size_t>(j * nx + i)] != 0; }
    [[nodiscard]] std::pair<int, int> cell_of(const Eigen::Vector2f& p) const
    { return {static_cast<int>((p.x() - x0) / cell), static_cast<int>((p.y() - y0) / cell)}; }
    /// 8-connected A* without corner cutting, cell centres from `from`'s cell to `to`'s. Empty = no route.
    [[nodiscard]] std::vector<Eigen::Vector2f> astar(const Eigen::Vector2f& from, const Eigen::Vector2f& to) const;
    /// The free standpoint nearest `p` (grid BFS), or `p` itself when no cell is free.
    [[nodiscard]] Eigen::Vector2f nearest_free(const Eigen::Vector2f& p) const;
    [[nodiscard]] int free_count() const;
    /// Mark cells for which `blocked(centre)` holds as not free (live: furniture the polygon's outline cannot show).
    template <class F> void block_if(F&& blocked)
    {
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                if (auto& f = free_[static_cast<size_t>(j * nx + i)]; f != 0 and blocked(at(i, j))) f = 0;
    }

    float cell = 0.30f, clearance = 0.35f;
    float x0 = 0.f, y0 = 0.f;
    int nx = 0, ny = 0;
private:
    std::vector<char> free_;
};

/// Binary entropy of a log-odds, in nats (offline entropy_nats).
float entropy_nats(float lodds);

/// The expected information gain, in nats, of one scan taken at `v` (map frame), predicted through the
/// estimated polygon `poly` and the map's own occupancy grid. Offline `eig_of`. 0 when the polygon is not
/// closed or the grid is not ready.
float expected_information_gain(const wallmap::WallMap& map, const wallmap::Polygon& poly,
                                const Eigen::Vector2f& v, const ExplorerParams& p);

struct Unknown { Eigen::Vector2f p; float w; };
/// Everything the map is still uncertain about, each in its own nats (offline collect_entropy_targets).
/// Used for the termination ledger only, exactly as offline (the EIG objective does not read it).
std::vector<Unknown> collect_entropy_targets(const wallmap::WallMap& map);

struct Plan
{
    bool ok = false;                       ///< a reachable target was chosen
    bool finished = false;                 ///< closed polygon AND nothing left to learn
    Eigen::Vector2f target{0.f, 0.f};      ///< map frame
    float score = 0.f;                     ///< nats per metre (gain / (1 + distance)), the ranking key
    float gain  = 0.f;                     ///< nats (score * (1 + distance))
    std::vector<Eigen::Vector2f> path;     ///< A* cell centres, robot's cell excluded (offline `path`)
    int candidates = 0;                    ///< feasible standpoints evaluated
    int with_gain  = 0;                    ///< of which gain > 0
    int unknowns   = 0;                    ///< termination ledger size
    Eigen::Vector2f start{0.f, 0.f};       ///< where the walk started (robot, or nearest feasible cell)
    std::string why;                       ///< one word on every path: ok | finished | no_feasible | no_gain | unreachable
};

/// LIVE-ONLY hook on the candidate loop: a standpoint the caller has just been REFUSED at (OutsideRoom,
/// Infeasible, Unreachable, an expired lease) must not be offered again straight away, or the pair loops
/// offer → refuse → offer (the bench has no consumer that can refuse). Empty = every candidate admissible,
/// i.e. exactly the bench.
using Admissible = std::function<bool(const Eigen::Vector2f&)>;

/// One replan (offline: the body of `if (--replan_in <= 0 or path.empty())` in run_explore, EIG branch).
/// `poly` is the estimated polygon the gain is predicted through (live: build_polygon_with the window's
/// information; offline: build_polygon()). Everything in the map frame.
Plan plan(const wallmap::WallMap& map, const wallmap::Polygon& poly, const FeasibleGrid& grid,
          const Eigen::Vector2f& robot, const ExplorerParams& p, const Admissible& admissible = {});

}   // namespace rc::layout
