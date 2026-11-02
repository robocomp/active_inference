// chained_action.h — each step ends where the next one wants to begin (spec 2026-11-02-chained-actions-design.md).
// Build + run:
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 test_chained_action.cpp -o /tmp/test_chained_action && /tmp/test_chained_action
#include "../src/chained_action.h"

#include <cmath>
#include <numbers>
#include <print>

namespace
{
int failures = 0, checks = 0;
void check(bool ok, int line, std::string_view what)
{
    ++checks;
    if (not ok) { ++failures; std::println("FAIL line {}: {}", line, what); }
}
#define CHECK(x) check((x), __LINE__, #x)
namespace ch = rc::chain;
constexpr float kDeg = 180.f / std::numbers::pi_v<float>;
}   // namespace

int main()
{
    // The door: aperture centre (2, 0) in a wall along x; the room being LEFT is y > 0 (n = +y); the robot travels toward -y.
    ch::EndRegion r;
    r.centre = {2.f, 0.f};
    r.n = {0.f, 1.f};
    r.s_min = 0.6f;
    r.s_max = 1.3f;
    const float travel = -std::numbers::pi_v<float> / 2.f;

    {   // 1. the heading is the route's first segment to the next goal, to within a few degrees
        const ch::NextGoal g{"door_3", {-1.5f, -2.5f}, 0.f};
        const auto c = ch::choose_end(r, g);
        CHECK(c.has_value());
        const Eigen::Vector2f d = (g.p - c->p).normalized();
        CHECK(std::abs(ch::wrap_pi(c->yaw - std::atan2(d.y(), d.x()))) * kDeg < 3.f);
        CHECK(c->s >= r.s_min - 1e-4f and c->s <= r.s_max + 1e-4f);
        CHECK(c->p.x() == 2.f);   // on the axis
        // the optimum: no other standoff in the region is cheaper (brute force, same cost)
        float best = 1e9f;
        for (float s = r.s_min; s <= r.s_max + 1e-4f; s += 0.01f)
        {
            const Eigen::Vector2f p = r.centre - s * r.n, dd = g.p - p;
            const float yaw = std::atan2(dd.y(), dd.x());
            best = std::min(best, dd.norm() / r.v_max + std::abs(ch::wrap_pi(yaw - travel)) / r.w_max);
        }
        CHECK(c->cost_s <= best + 0.02f);
    }
    {   // 2. a goal STRAIGHT AHEAD on the axis: no turn at all
        const auto c = ch::choose_end(r, {"d", {2.f, -4.f}, 0.f});
        CHECK(c.has_value() and c->turn_deg < 1.f);
    }
    {   // 3. a goal far to the side and BEHIND the wall line: the further the end, the smaller the turn -> the choice is interior
        //    or at an end of the region, but never outside it
        const auto c = ch::choose_end(r, {"d", {-3.f, -0.5f}, 0.f});
        CHECK(c.has_value() and c->s >= r.s_min - 1e-4f and c->s <= r.s_max + 1e-4f);
    }
    {   // 4. the planner's first segment wins over the straight line (a wall between): the heading follows the ROUTE
        const ch::NextGoal g{"d", {-1.5f, -2.5f}, 0.f};
        const ch::RouteFn detour = [&](const Eigen::Vector2f& p) -> std::optional<ch::RouteFrom>
        { return ch::RouteFrom{(g.p - p).norm() + 1.f, Eigen::Vector2f(0.f, -1.f)}; };   // first leg straight down the axis
        const auto c = ch::choose_end(r, g, detour);
        CHECK(c.has_value() and std::abs(ch::wrap_pi(c->yaw - travel)) * kDeg < 1.f and c->turn_deg < 1.f);
    }
    {   // 5. the leaf's swept extent is a CONTINUOUS term: P=0 -> body only, P=1 -> the whole leaf, unknown -> no freedom
        CHECK(std::abs(ch::swept_standoff(0.4f, 0.f, 0.9f, 1.3f) - 0.4f) < 1e-5f);
        CHECK(std::abs(ch::swept_standoff(0.4f, 0.5f, 0.9f, 1.3f) - 0.85f) < 1e-5f);
        CHECK(std::abs(ch::swept_standoff(0.4f, 1.f, 0.9f, 1.3f) - 1.3f) < 1e-5f);   // capped by the published end
        CHECK(ch::swept_standoff(0.4f, std::nullopt, 0.9f, 1.3f) == 1.3f);
        ch::EndRegion un = r;
        un.s_min = ch::swept_standoff(0.4f, std::nullopt, 0.9f, un.s_max);
        const auto c = ch::choose_end(un, {"d", {-1.5f, -2.5f}, 0.f});
        CHECK(c.has_value() and std::abs(c->s - 1.3f) < 1e-4f and not c->moved);   // today's end
    }
    {   // 6. the look-ahead: `cross R2; close R1; open R3` — the next GOAL after the cross skips the close
        using rc::mission::Step;
        std::vector<Step> steps{Step{.id = "1", .op = "cross"}, Step{.id = "2", .op = "close"}, Step{.id = "3", .op = "open"}};
        CHECK(ch::next_step(steps, "1") != nullptr and ch::next_step(steps, "1")->id == "3");
        CHECK(ch::following_step(steps, "1") != nullptr and ch::following_step(steps, "1")->op == "close");
        CHECK(ch::next_step(steps, "2") != nullptr and ch::next_step(steps, "2")->id == "3");   // from the close itself
        CHECK(ch::next_step(steps, "3") == nullptr);                  // the last step: unknown -> old behaviour
        CHECK(ch::next_step(steps, "nope") == nullptr);
        std::vector<Step> nested{Step{.id = "1", .op = "seq", .steps = {Step{.id = "1.1", .op = "cross"}, Step{.id = "1.2", .op = "open"}}}};
        CHECK(ch::next_step(nested, "1.1") != nullptr and ch::next_step(nested, "1.1")->id == "1.2");
    }
    std::println("test_chained_action: {} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
