// Task 3f (tour3A leg 1): the doorway's cost-to-go past a shallow newborn room, `through` as a posterior, and a bounded clearing.
// Build+run (from controller/; one line):
//   g++ -std=c++23 -O2 -Wall -Wextra -I/usr/include/eigen3 tests/test_field_extent.cpp src/local_policy/cost_to_go.cpp
//     -o /tmp/tfe && /tmp/tfe
//
// The fixture is tour3A's field dump (controller/policy_field_dump_1792613629022.txt), straightened: room_2 just born as a
// 2.1-m-deep rectangle, the planner raster ending 0.54 m past its far edge, the robot 0.12 m past the door plane, and the doorway
// goal line 2.84 m past the door — beyond the raster.
#include "../src/local_policy/cost_to_go.h"
#include "../src/local_policy/cross_through.h"
#include "../src/local_policy/field_extent.h"
#include "../src/local_policy/policy_tick_gate.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

using namespace rc::policy;
using V = Eigen::Vector2f;
static int fails = 0;
#define CHECK(name, cond) do { const bool ok_ = (cond); std::printf("%s %s\n", ok_ ? "PASS" : "FAIL", name); if (not ok_) ++fails; } while (0)

namespace
{
// the newborn room's outline (room frame) and the planner's raster around it (its outside-room margin is occupied)
constexpr float kRx0 = -1.0f, kRx1 = 3.5f, kRy0 = 0.06f, kRy1 = 2.17f;
const PlannerExtent kExt{{-1.212f, -0.414f}, {3.708f, 2.706f}};
constexpr float kPlannerCell = 0.10f, kRins = 0.20f, kFieldCell = 0.05f, kCirc = 0.3183f;
// the planner's EDT: distance to the nearest occupied cell — everything outside the outline is occupied (0)
float planner_distance(const V& p)
{
    if (p.x() < kRx0 or p.x() > kRx1 or p.y() < kRy0 or p.y() > kRy1) return 0.f;
    return std::min({p.x() - kRx0, kRx1 - p.x(), p.y() - kRy0, kRy1 - p.y()});
}
// the door is at (-0.2, 0.06), normal +y; the measured opening's corridor: 1 m back into the room left, one cell past the line
constexpr float kDoorY = 0.06f, kPast = 2.84f;
bool corridor(const V& p) { return p.x() >= -0.65f and p.x() <= 0.25f and p.y() >= kDoorY - 1.0f and p.y() <= kDoorY + kPast + 0.05f; }
std::vector<V> goal_line()
{
    std::vector<V> g;
    for (float x = -0.6f; x <= 0.2001f; x += 0.08f) g.emplace_back(x, kDoorY + kPast);
    return g;
}
bool free_of(float d) { return d - 0.5f * kPlannerCell > kRins; }   // goal_adapter.h field_free_from_distance
}  // namespace

int main()
{
    const auto goals = goal_line();
    const V robot{-0.22f, 0.18f};   // the centre 0.12 m past the plane: switched, the body not yet wholly through
    const float unseen = unseen_prior_density(0.2446f, kPlannerCell, 1.f);   // Shadow's hull area

    // ── (b) BEFORE: the field over the planner's extent only (controller_policy_drive.cpp until Task 3f) ⇒ inf everywhere
    {
        const V o = kExt.lo;
        const V span = kExt.hi - kExt.lo;
        const CostToGo::Grid g{o, kFieldCell, int(std::ceil(span.x() / kFieldCell)) + 1, int(std::ceil(span.y() / kFieldCell)) + 1};
        auto free = [](const V& p) { return free_of(planner_distance(p)) or corridor(p); };
        CostToGo c;
        c.build(g, free, [](const V&) { return 0.f; }, goals);
        CHECK("before_3f_goal_off_the_planner_grid_field_inf_at_robot", c.seeded_goal_cells() == 0 and std::isinf(c.at(robot)));
    }
    // ── (b) AFTER: the field's grid covers extent ∪ goal points ∪ robot; off-extent cells are unseen (priced, not walls)
    CostToGo c;
    {
        const CostToGo::Grid g = field_grid(kExt, goals, robot, kCirc + kFieldCell, kFieldCell);
        auto dist = [](const V& p) { return extended_distance(kExt, p, planner_distance); };
        auto free = [&](const V& p) { return free_of(dist(p)) or corridor(p); };
        auto extra = [&](const V& p) { return kExt.contains(p) ? 0.f : unseen; };
        c.build(g, free, extra, goals);
        const float f = c.at(robot);
        CHECK("goal_points_on_the_field_grid", c.seeded_goal_cells() > 0);
        CHECK("field_finite_at_robot_past_the_door", std::isfinite(f));
        // the route length is the straight way down the corridor; the price adds the unseen strip's toll, finite
        const float straight = kDoorY + kPast - robot.y();
        CHECK("route_metres_close_to_straight", std::abs(c.at_length(robot) - straight) < 0.15f);
        std::printf("   field %.2f m (route %.2f m, unseen density %.2f m/m)\n", f, c.at_length(robot), unseen);
        CHECK("price_includes_the_unseen_toll", f > c.at_length(robot) + 0.5f * unseen * (kDoorY + kPast - kExt.hi.y()));
        // a cell just past the raster's edge, outside the corridor, is next to the raster's occupied margin: not free
        CHECK("unseen_cell_beside_the_margin_not_free", not free(V(2.0f, kExt.hi.y() + 0.10f)));
        // ... one well away from the raster is free (unseen space is not a wall) and priced at the prior
        CHECK("unseen_cell_far_from_the_margin_free_and_priced", free(V(2.0f, kExt.hi.y() + 0.40f)) and extra(V(2.0f, kExt.hi.y() + 0.40f)) == unseen);
        CHECK("unseen_prior_is_ln2_per_planner_cell", std::abs(unseen - 0.2446f / 0.01f * std::log(2.f)) < 1e-3f);
    }

    // ── (a) THROUGH AS A POSTERIOR: the robot just past the door, room_concept has switched it into room_2. Walk it down the
    //    field (5 cm steps toward the cheapest neighbour) until P(through | s, switch) > 1/2 — and that comes well before the
    //    old 2.0-m line, and inside the newborn outline.
    {
        const float h = 0.2840f;   // Shadow's |x|max: the body's farthest point from its centre (a stated bound on its rear)
        const double sigma_s = 0.05, sigma_r = 0.06;
        V p = robot;
        int steps = 0;
        double pt = p_through_given_switch(p.y() - kDoorY, sigma_s, h, sigma_r);
        while (pt <= 0.5 and steps < 200)
        {
            V best = p;
            float fb = c.at(p);
            for (int k = 0; k < 16; ++k)
            {
                const float a = k * 2.f * float(M_PI) / 16.f;
                const V q = p + 0.05f * V(std::cos(a), std::sin(a));
                if (const float fq = c.at(q); fq < fb) { fb = fq; best = q; }
            }
            if (best == p) break;
            p = best;
            ++steps;
            pt = p_through_given_switch(p.y() - kDoorY, sigma_s, h, sigma_r);
        }
        std::printf("   through at s %.2f m after %d steps (P %.3f)\n", p.y() - kDoorY, steps, pt);
        CHECK("through_reached_on_the_finite_field", pt > 0.5);
        CHECK("through_well_before_the_2m_standoff", p.y() - kDoorY < 1.0f and p.y() < kRy1);
    }
    {
        // the posterior's shape: monotone in s; a switch on a vague pose says less; the switch alone does not make a body
        // whose centre is on the plane wholly past it; deep past the plane it is ~1
        const double h = 0.284;
        CHECK("p_monotone_in_s", p_through_given_switch(0.2, 0.05, h, 0.05) < p_through_given_switch(0.3, 0.05, h, 0.05)
                                     and p_through_given_switch(0.3, 0.05, h, 0.05) < p_through_given_switch(0.4, 0.05, h, 0.05));
        CHECK("p_centre_on_plane_not_through", p_through_given_switch(0.0, 0.05, h, 0.05) < 0.01);
        CHECK("p_deep_past_through", p_through_given_switch(1.0, 0.05, h, 0.05) > 0.999);
        // the pose says s ~ 0.2 ± 0.3 (vague): a precise switch (σ_r small) cuts away s < 0 and RAISES P; a vague one barely
        const double p_none = 0.5 * std::erfc(-(0.2 - h) / (0.3 * std::sqrt(2.0)));
        const double p_sharp = p_through_given_switch(0.2, 0.3, h, 0.01), p_vague = p_through_given_switch(0.2, 0.3, h, 3.0);
        std::printf("   vague pose s 0.2 +- 0.3: no switch %.3f, vague switch %.3f, sharp switch %.3f\n", p_none, p_vague, p_sharp);
        CHECK("switch_is_evidence_weighted_by_its_sigma", p_sharp > p_vague and p_vague >= p_none - 1e-9 and p_vague - p_none < 0.05);
        CHECK("p_nonfinite_s_judges_nothing", p_through_given_switch(std::numeric_limits<double>::quiet_NaN(), 0.05, h, 0.05) == 0.0);
    }

    // ── (c) A CLEARING THE POLICY CANNOT SERVE IS GIVEN UP — bounded, never for ever
    {
        ClearingNoRoute clk;
        UnreachLatch latch;
        std::uint64_t t = 1000, gave_up_at = 0;
        for (int i = 0; i < 2000 and gave_up_at == 0; ++i, t += 50)
            if (clk.on_tick(t, NoRouteObs::Infinite, false, false, latch)) gave_up_at = t;
        CHECK("infinite_field_gives_clearing_up", gave_up_at != 0 and gave_up_at - 1000 <= kNoRouteReportMs + 100);
        ClearingNoRoute c2;
        t = 1000;
        bool any = false;
        for (int i = 0; i < 2000; ++i, t += 50)   // Blind / frame-wait ticks evaluate no field: no evidence, never given up
            any = c2.on_tick(t, i == 0 ? NoRouteObs::Infinite : NoRouteObs::NotEvaluated, false, false, latch) or any;
        CHECK("unevaluated_ticks_never_give_up", not any);
        ClearingNoRoute c3;
        t = 1000;
        any = false;
        for (int i = 0; i < 2000; ++i, t += 50)   // a finite field now and then: the clock restarts
            any = c3.on_tick(t, i % 50 == 49 ? NoRouteObs::Finite : NoRouteObs::Infinite, false, false, latch) or any;
        CHECK("finite_field_resets_the_clock", not any);
    }

    std::printf("%s (%d failures)\n", fails == 0 ? "ALL PASS" : "FAILURES", fails);
    return fails == 0 ? 0 : 1;
}
