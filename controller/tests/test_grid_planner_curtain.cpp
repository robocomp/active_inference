// try96 leg 3 (2026-10-19): a thin, weakly-evidenced residual "curtain" sealed the robot off from a door approach,
// and the reachability fallback never fired because its record was keyed on a world hash that changes every cycle.
// This pins both fixes: residual belief as a COST (residual_risk.h + GridPlanner soft layer) and a failure memory
// that survives hash churn (route_failure_memory.h). Also runs GridPlanner::self_test() so the planner's own
// invariants are re-checked against the soft layer.
// Build+run (from controller/; pure Eigen + std):
//   g++ -std=c++23 -O2 -Wall -Wextra -I/usr/include/eigen3 -Isrc tests/test_grid_planner_curtain.cpp
//       src/grid_planner.cpp ../common/robot_footprint/robot_footprint.cpp ../common/robot_footprint/mesh_hull.cpp
//       -o /tmp/tgpc && /tmp/tgpc   (one line)

#include "grid_planner.h"
#include "residual_risk.h"
#include "route_failure_memory.h"

#include <cmath>
#include <cstdio>
#include <set>
#include <vector>

namespace
{
int failures = 0;
void check(bool ok, const char *what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (not ok) ++failures;
}

const std::vector<Eigen::Vector2f> kRoom{{0.f, 0.f}, {6.f, 0.f}, {6.f, 6.f}, {0.f, 6.f}};
constexpr float kRes = 0.05f;   // residual's cell

// A curtain two residual cells thick across the whole room at x ≈ 3, optionally leaving a gap near the top wall.
// Centres on a lattice offset like residual's real one (its grid starts at the room bbox, not at 0).
std::vector<Eigen::Vector2f> curtain(bool with_gap)
{
    std::vector<Eigen::Vector2f> c;
    for (float y = 0.0215f; y < 6.f; y += kRes)
    {
        if (with_gap and y > 4.6f and y < 5.6f) continue;   // a 1.0 m opening the body fits through
        c.emplace_back(2.9785f, y);
        c.emplace_back(3.0285f, y);
    }
    return c;
}

// Beta moments for residual's construction: Jeffreys (½,½) + κ_obs of evidence at occupancy p.
std::pair<float, float> moments(float p, float kobs)
{
    const double a = 0.5 + kobs * p, b = 0.5 + kobs * (1.0 - p), k = a + b;
    return {static_cast<float>(a / k), static_cast<float>(a * b / (k * k * (k + 1.0)))};
}
std::vector<float> uniform_risk(std::size_t n, float p, float kobs)
{
    const auto [P, V] = moments(p, kobs);
    return std::vector<float>(n, rc::residual_risk::cell_risk_nats(P, V));
}

rc::GridPlanner planner_with(const std::vector<Eigen::Vector2f> &cells, std::vector<float> risk)
{
    rc::GridPlanner p;
    p.params.cell_size_m = 0.10f;
    p.params.safety_margin_m = 0.05f;
    p.set_world(kRoom, {}, rc::OccupiedCells{.centres = cells, .cell_size_m = kRes, .risk_nats = std::move(risk)});
    return p;
}
}  // namespace

int main()
{
    std::printf("GridPlanner::self_test (the planner's own invariants)\n");
    check(rc::GridPlanner::self_test(), "GridPlanner::self_test passes with the soft layer compiled in");

    // ── 1. THE RISK OF ONE CELL ──────────────────────────────────────────────────────────────────────
    std::printf("\n1. cell risk from residual's Beta\n");
    {
        using rc::residual_risk::cell_risk_nats;
        const auto [Pu, Vu] = moments(0.5f, 0.f);                    // unobserved: Jeffreys
        const float unknown = cell_risk_nats(Pu, Vu);
        const auto [Pw, Vw] = moments(0.6f, 1.5f);                   // a one/two-hit latch: the curtain
        const float weak = cell_risk_nats(Pw, Vw);
        const auto [Pm, Vm] = moments(0.9f, 10.f);
        const float mid = cell_risk_nats(Pm, Vm);
        const auto [Pc, Vc] = moments(0.993f, 40.f);                 // at residual's concentration cap
        const float confirmed = cell_risk_nats(Pc, Vc);
        std::printf("  unknown %.3f  weak %.3f  mid %.3f  confirmed %.3f nats\n", unknown, weak, mid, confirmed);
        check(std::abs(unknown - std::log(2.f)) < 1e-3f, "Jeffreys (P=.5, Var=.125) costs exactly ln 2: Pr(theta<1/2) = 1/2");
        check(weak > 0.f and weak < 2.f, "a one/two-hit curtain cell is a toll of about a nat");
        check(unknown < weak and weak < mid and mid < confirmed, "risk grows monotonically with evidence");
        check(confirmed > 20.f, "a cell at residual's cap costs > 20 nats (Pr free ~ 2^-40)");
        check(cell_risk_nats(0.f, 0.f) == 0.f, "a cell residual collapsed (explained) carries no risk");
        check(std::isinf(cell_risk_nats(std::nanf(""), 0.1f)), "no usable belief -> HARD");
        rc::residual_risk::BeliefField none;
        const std::vector<Eigen::Vector2f> one{{1.f, 1.f}};
        check(std::isinf(rc::residual_risk::risks_from_field(one, none)[0]), "no field at all -> HARD (the old planner)");
    }

    // ── 2. THE CURTAIN ───────────────────────────────────────────────────────────────────────────────
    std::printf("\n2. a curtain between start and goal\n");
    const Eigen::Vector2f start{1.f, 2.f}, goal{5.f, 2.f};
    const auto wall = curtain(/*with_gap=*/false);
    {
        auto empty = planner_with({}, {});
        const auto free_route = empty.plan(start, goal);
        check(free_route.has_value(), "empty room: routes");
        const float free_cost = empty.last_route_cost_m();

        // (a) the try96 failure, reproduced: the latched cells as walls (no belief offered) disconnect the room.
        auto hard = planner_with(wall, {});
        check(not hard.plan(start, goal).has_value(), "(a) curtain as HARD cells: no route (try96's 'no route')");
        std::printf("      %s\n", hard.last_failure().c_str());

        // (b) the same curtain, priced by a WEAK belief: a route exists and costs more than open floor.
        auto weak = planner_with(wall, uniform_risk(wall.size(), 0.6f, 1.5f));
        const auto weak_route = weak.plan(start, goal);
        std::printf("      weak: %s, cost %.2f m vs %.2f m open, worst pose %.1f nats; soft in %d, hard in %d\n",
                    weak_route ? "routed" : weak.last_failure().c_str(), weak.last_route_cost_m(), free_cost,
                    weak.last_route_risk_nats(), weak.residual_soft_in(), weak.residual_hard_in());
        check(weak_route.has_value(), "(b) WEAK curtain: a route exists — belief raises the cost, never disconnects");
        check(weak.last_route_cost_m() > free_cost + 0.5f, "(b) ...and it costs more than the open room");
        check(weak.last_route_risk_nats() > 0.f, "(b) ...and the planner reports the risk it paid");
        check(weak.world_hash() == empty.world_hash(), "(b) soft cells do not change the map's HARD identity");

        // (c) a confirmed curtain WITH a 1 m gap: the route takes the gap, never the curtain.
        const auto gapped = curtain(/*with_gap=*/true);
        auto confirmed_gap = planner_with(gapped, uniform_risk(gapped.size(), 0.993f, 40.f));
        const auto around = confirmed_gap.plan(start, goal);
        bool via_gap = false;
        if (around)
            for (std::size_t i = 0; i + 1 < around->size(); ++i)
            {
                const auto &a = (*around)[i], &b = (*around)[i + 1];
                if ((a.x() - 3.f) * (b.x() - 3.f) <= 0.f and std::abs(b.x() - a.x()) > 1e-6f)
                {
                    const float y = a.y() + (3.f - a.x()) * (b.y() - a.y()) / (b.x() - a.x());
                    via_gap = via_gap or (y > 4.6f and y < 5.6f);
                }
            }
        check(around.has_value() and via_gap, "(c) CONFIRMED curtain with a gap: the route detours through the gap");
        check(around.has_value() and confirmed_gap.last_route_risk_nats() < 1.f,
              "(c) ...touching no confirmed cell on the way");

        // (d) the same weak curtain with the same gap: the cost is a toll, so a detour that long is NOT worth it.
        auto weak_gap = planner_with(gapped, uniform_risk(gapped.size(), 0.6f, 1.5f));
        const auto weak_through = weak_gap.plan(start, goal);
        std::printf("      weak+gap: cost %.2f m, worst pose %.1f nats; confirmed+gap: cost %.2f m\n",
                    weak_gap.last_route_cost_m(), weak_gap.last_route_risk_nats(), confirmed_gap.last_route_cost_m());
        check(weak_through.has_value(), "(d) weak curtain with a far gap: routes");

        // (d2) ★THE SEMANTICS TO KNOW: a CONFIRMED belief the LiDAR does not see right now (occluded, out of range)
        //      is a toll so steep it is taken only when NOTHING else connects — it is not a literal wall. What
        //      stops the body at it is the live LiDAR (harden_corroborated next cycle, and the body guard).
        auto confirmed_unseen = planner_with(wall, uniform_risk(wall.size(), 0.993f, 40.f));
        const auto through = confirmed_unseen.plan(start, goal);
        std::printf("      confirmed, unseen, no gap: %s, cost %.0f m (weak: %.0f m)\n",
                    through ? "routed" : confirmed_unseen.last_failure().c_str(),
                    confirmed_unseen.last_route_cost_m(), weak.last_route_cost_m());
        check(through.has_value() and confirmed_unseen.last_route_cost_m() > 20.f * weak.last_route_cost_m(),
              "(d2) confirmed-but-unseen: routable only at > 20x the weak curtain's cost");

        // (e) HIGH CONFIDENCE that the LiDAR SEES now: the cells go back to HARD -> no route, and the
        //     reachability fallback gives a pose on the robot's side that DOES route.
        std::vector<float> seen = uniform_risk(wall.size(), 0.993f, 40.f);
        std::vector<float> xs, ys, zs;
        for (const auto &c : wall) { xs.push_back(c.x() + 0.01f); ys.push_back(c.y() - 0.01f); zs.push_back(0.5f); }
        const int hardened = rc::residual_risk::harden_corroborated(wall, kRes, seen, xs, ys, zs, 0.05f, 1.45f);
        check(hardened == static_cast<int>(wall.size()), "(e) every curtain cell with a live return in the body band is HARDENED");
        auto confirmed = planner_with(wall, seen);
        check(not confirmed.plan(start, goal).has_value(), "(e) a curtain the LiDAR sees: NO route (stays a wall)");
        const auto reach = confirmed.nearest_reachable(start, goal);
        std::printf("      nearest_reachable: %s\n",
                    reach ? std::format("({:.2f},{:.2f})", reach->x(), reach->y()).c_str() : "none");
        check(reach.has_value() and reach->x() < 3.f, "(e) nearest_reachable is on the robot's side");
        check(reach.has_value() and confirmed.plan(start, *reach).has_value(), "(e) ...and it ROUTES");
        check(reach.has_value() and (*reach - goal).norm() < (start - goal).norm(), "(e) ...and is closer to the goal");

        // (f) returns OUTSIDE the body band (floor, overhead) do not harden anything.
        std::vector<float> high = uniform_risk(wall.size(), 0.6f, 1.5f);
        std::vector<float> zhigh(zs.size(), 1.9f);
        check(rc::residual_risk::harden_corroborated(wall, kRes, high, xs, ys, zhigh, 0.05f, 1.45f) == 0,
              "(f) a return above the body does not make a residual cell a wall");
        // ...nor does one in a NEIGHBOURING lattice square.
        std::vector<float> shifted_x;
        for (const auto x : xs) shifted_x.push_back(x + 0.10f);   // two residual cells east of the curtain
        check(rc::residual_risk::harden_corroborated(wall, kRes, high, shifted_x, ys, zs, 0.05f, 1.45f) == 0,
              "(f) a return two cells away does not harden the curtain");
    }

    // ── 3. THE FALLBACK FIRES WHILE THE WORLD HASH CHURNS ───────────────────────────────────────────
    // The controller's loop, reduced to its routing decisions: plan(); on failure note it; next cycle, if
    // routing to this target already failed from here, drive to nearest_reachable instead. Every cycle the
    // world gets one fresh HARD residual cell somewhere irrelevant, so world_hash() changes every cycle — the
    // churn that kept the old (goal, robot, MAP)-keyed record from ever matching.
    std::printf("\n3. fallback under world-hash churn\n");
    {
        const std::string name = "aff_door_1_approach";
        rc::RouteFailureMemory memory;
        std::optional<Eigen::Vector2f> fix;
        std::set<std::size_t> hashes;
        std::size_t prev_hash = 0;
        bool hash_changed_every_cycle = true, old_key_ever_matched = false;
        struct OldRecord { Eigen::Vector2f goal, robot; std::size_t map; };
        std::optional<OldRecord> old_record;
        Eigen::Vector2f robot = start;
        int first_fix_cycle = -1, routed_cycles = 0, held_cycles = 0;
        constexpr int kCycles = 30;
        for (int cycle = 0; cycle < kCycles; ++cycle)
        {
            auto cells = wall;
            std::vector<float> risk(cells.size(), rc::residual_risk::kHard);   // a wall the LiDAR sees
            cells.emplace_back(0.5f + 0.05f * static_cast<float>(cycle), 5.7f);  // churn: one new hard cell
            risk.push_back(rc::residual_risk::kHard);
            auto p = planner_with(cells, risk);
            if (cycle > 0 and p.world_hash() == prev_hash) hash_changed_every_cycle = false;
            prev_hash = p.world_hash();
            hashes.insert(p.world_hash());

            // The old predicate, for the record.
            if (old_record and (old_record->goal - goal).norm() < 0.05f and (old_record->robot - robot).norm() < 0.05f
                and old_record->map == p.world_hash())
                old_key_ever_matched = true;

            const bool failed_here = memory.failed_here(name, goal, robot, fix.has_value());
            if (not failed_here) fix.reset();
            else if (not fix.has_value())
            {
                fix = p.nearest_reachable(robot, goal);
                if (fix and first_fix_cycle < 0) first_fix_cycle = cycle;
            }
            const Eigen::Vector2f target = fix.value_or(goal);
            const auto route = p.plan(robot, target);
            if (route.has_value())
            {
                ++routed_cycles;
                // Drive 10 cm along the route: the robot MOVES, which must not void the engaged fallback.
                const Eigen::Vector2f next = route->size() > 1 ? (*route)[1] : target;
                const Eigen::Vector2f d = next - robot;
                if (d.norm() > 1e-6f) robot += d.normalized() * std::min(0.10f, d.norm());
            }
            else
            {
                ++held_cycles;
                memory.note_failure(name, goal, robot);
                old_record = OldRecord{goal, robot, p.world_hash()};
                fix.reset();
            }
        }
        std::printf("  %zu distinct hashes in %d cycles; fallback engaged at cycle %d; routed %d, held %d; robot ended at (%.2f,%.2f)\n",
                    hashes.size(), kCycles, first_fix_cycle, routed_cycles, held_cycles, robot.x(), robot.y());
        check(hash_changed_every_cycle, "the world hash changed on EVERY cycle (the churn is real)");
        check(not old_key_ever_matched, "the OLD (goal, robot, map) key never matched — the try96 dead fallback");
        check(first_fix_cycle == 1, "the NEW memory engages nearest_reachable on the very next cycle");
        check(held_cycles == 1, "the robot held for exactly one cycle, not for the 5 s liveness");
        check(routed_cycles == kCycles - 1, "every later cycle routed (to the reachable pose)");
        check(fix.has_value() and (robot - start).norm() > 0.5f,
              "the robot DROVE toward the fallback and the fallback stayed engaged while it moved");
    }

    // ── 4. THE MEMORY'S KEY ─────────────────────────────────────────────────────────────────────────
    std::printf("\n4. route failure memory key\n");
    {
        rc::RouteFailureMemory m;
        m.note_failure("aff_a", {1.f, 1.f}, {0.f, 0.f});
        check(m.failed_here("aff_a", {1.02f, 1.f}, {0.01f, 0.f}, false), "same target, robot still: failed here");
        check(not m.failed_here("aff_b", {1.f, 1.f}, {0.f, 0.f}, false), "another node: not this failure");
        check(not m.failed_here("aff_a", {1.5f, 1.f}, {0.f, 0.f}, false),
              "same node, another published pose (room's one afford_room node): not this failure");
        check(not m.failed_here("aff_a", {1.f, 1.f}, {0.5f, 0.f}, false), "robot moved and no fallback held: ask again");
        check(m.failed_here("aff_a", {1.f, 1.f}, {0.5f, 0.f}, true), "robot moved WHILE driving the fallback: still held");
        m.clear();
        check(not m.failed_here("aff_a", {1.f, 1.f}, {0.f, 0.f}, true), "cleared (approach end): asks the planner again");
    }

    // ── 5. THE LOCAL POLICY'S DENSITY (Task 8) ────────────────────────────────────────────────────────────────
    std::printf("\n5. body risk for the policy's cost-to-go\n");
    {
        auto weak = planner_with(wall, uniform_risk(wall.size(), 0.6f, 1.5f));
        const Eigen::Vector2f on = wall[wall.size() / 2];
        const float cell = weak.risk_at(on), body = weak.body_risk_at(on);
        std::printf("      on the curtain: one cell %.2f nats, under the body (cheapest heading) %.2f nats\n", cell, body);
        check(body > cell and cell > 0.f, "(5) the body covers more than one soft cell: its risk exceeds one cell's");
        check(weak.body_risk_at(start) == 0.f, "(5) open floor: no risk under the body");
        auto empty = planner_with({}, {});
        check(empty.body_risk_at(on) == 0.f, "(5) no soft layer: 0");
        check((weak.grid_origin() - Eigen::Vector2f(weak.grid_origin())).norm() == 0.f and weak.cell_size() > 0.f,
              "(5) grid geometry exposed");
    }

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILED", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
