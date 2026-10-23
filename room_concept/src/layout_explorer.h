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
#include <optional>
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
    /// Free cells 4-connected to `from`'s cell (0 when that cell is not free) — the diagnostic for "unreachable".
    [[nodiscard]] int component_size(const Eigen::Vector2f& from) const;
    /// Mark cells for which `occupied_by_robot(centre)` holds as free: the robot is standing there, so the floor under
    /// its body is feasible whatever the polygon clearance says (live: a cramped start dropped the robot's own cell).
    /// Returns how many cells `occupied_by_robot` covers (freed or already free): the robot's own disc.
    template <class F> int free_if(F&& occupied_by_robot)
    {
        int n = 0;
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                if (occupied_by_robot(at(i, j))) { ++n; free_[static_cast<size_t>(j * nx + i)] = 1; }
        return n;
    }
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
    int deferred_with_gain = 0;            ///< standpoints with gain held back only by a recent refusal (plan_coverage)
    int evaluated = 0;                     ///< plan_coverage: standpoints scored at all, deferred ones included
    float best_gain = 0.f;                 ///< plan_coverage: the most ANY of them would reveal (room_explored.h)
    float price = 0.f;                     ///< plan_coverage: its min_gain, the price of a trip (0 = plan(): none)
    Eigen::Vector2f start{0.f, 0.f};       ///< where the walk started (robot, or nearest feasible cell)
    int reach_cells = -1;                  ///< free cells connected to `start` (filled on "unreachable" only)
    int free_cells  = -1;                  ///< free cells in the whole grid (idem)
    std::optional<Eigen::Vector2f> best_unreached;   ///< on "unreachable": the best candidate by gain per straight metre
    float best_unreached_gain = 0.f;       ///< the gain of `best_unreached`, in the planner's own unit (set with it)
    float nats_per_gain = 1.f;             ///< conversion of `gain` to nats: 1 for plan() (already nats); plan_coverage counts
                                           ///< m2 of unknown floor, its caller sets coverage_nats_per_m2(cell)
    std::string why;                       ///< one word on every path: ok | finished | no_feasible | no_gain | unreachable
};

/// Nats of expected information per m2 of still-unknown floor: an unknown occupancy cell is a binary variable at
/// p = 1/2, i.e. ln 2 nats, and a cell covers cell^2 m2. The currency the controller compares across affordances.
inline float coverage_nats_per_m2(float cell) { return 0.69314718f / (cell * cell); }

/// BOXED IN BY THE GRID, NOT BY THE WORLD: before the first visit "unreachable" says nothing about the room, so the
/// best candidate goes to the controller (its body-aware planner judges reachability and reports Unreachable through
/// the existing protocol). The offer carries the candidate's OWN gain: zeroing it made the controller's filter drop
/// the only offer. Returns true when the plan was promoted to a handoff.
bool boxed_in_handoff(Plan& p, int visits);

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

/// COVERAGE PLANNER (2026-10-10, used while plain SLAM owns the pose): gain = what a standpoint's view would REVEAL
/// of the map (m² of still-unknown cells, `gain`), cost = the A* ROUTE length; score = gain / (1 + route m). The old
/// planner scored what a view teaches the WALL MAP — flat and small once the layout came from the SLAM map, so the
/// distance term won and it offered spots next to the robot. `finished` = no admissible standpoint reveals at least
/// `min_gain` (the robot's own footprint area: a view that cannot reveal even that is not worth the trip).
/// `deferred` (optional) marks standpoints held back only for NOW (a recent refusal by the controller). Their gain still
/// counts as gain left: when only deferred standpoints would reveal anything, the plan is "deferred", never "finished".
using GainFn = std::function<float(const Eigen::Vector2f&)>;
Plan plan_coverage(const FeasibleGrid& grid, const Eigen::Vector2f& robot, const ExplorerParams& p,
                   const Admissible& admissible, const GainFn& gain, float min_gain, const Admissible& deferred = {});

}   // namespace rc::layout
