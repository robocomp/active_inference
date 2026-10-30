// door_route.h — the door route's waypoints from door_concept's published targets (controller_door_passage.h ROUTE MODE).
//   1. a free axis: [P_pre] for a shut door; [P_pre, P_ap, P_post] in that order, on the axis, on the right sides, for an open one
//   2. a BED in front of the door (live 2026-10-24): P_pre stays ON THE AXIS and SHORTENS to the largest free standoff
//   3. nothing free on the axis down to the body's reach ⇒ no plan (why says so) — never a sideways pose
//   4. the leaf standing on the published cross target ⇒ P_post slides OUTWARD along the axis
//   5. the robot already at the vantage (after an open-wait) ⇒ P_pre is dropped, the route starts at the aperture
// Build + run:
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 test_door_route.cpp -o /tmp/test_door_route && /tmp/test_door_route
#include "../src/door_route.h"

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
namespace dr = rc::door_route;
// The door: aperture centre (2, 0) in a wall along x; the current room is y > 0; door_concept's approach vantage 1.6 m out,
// facing the door (yaw -pi/2); its cross target 0.8 m past (y = -0.8), travel heading -pi/2.
const Eigen::Vector2f kCentre(2.f, 0.f);
const dr::Target kAppr{{2.f, 1.6f}, -std::numbers::pi_v<float> / 2.f};
const dr::Target kCross{{2.f, -0.8f}, -std::numbers::pi_v<float> / 2.f};
bool on_axis(const Eigen::Vector2f& p) { return std::abs(p.x() - 2.f) < 1e-4f; }
}   // namespace

int main()
{
    dr::Params prm;
    prm.body_reach_m = 0.33f; prm.step_m = 0.05f; prm.post_slide_m = 0.8f;
    const auto free_all = [](const Eigen::Vector2f&, float) { return true; };
    const Eigen::Vector2f robot_far(4.f, 3.f);

    std::println("1. a free axis");
    {
        const auto shut = dr::plan(kAppr, std::nullopt, kCentre, robot_far, false, prm, free_all);
        CHECK(shut.why.empty() and shut.waypoints.size() == 1 and (shut.waypoints[0] - kAppr.p).norm() < 1e-4f);
        CHECK(not shut.through and not shut.pre_moved);
        const auto open = dr::plan(kAppr, kCross, kCentre, robot_far, true, prm, free_all);
        std::println("    open: {} waypoints, pre ({:.2f},{:.2f}) s={:.2f}, ap ({:.2f},{:.2f}), post ({:.2f},{:.2f}) s={:.2f}",
                     open.waypoints.size(), open.pre.x(), open.pre.y(), open.s_pre, open.ap.x(), open.ap.y(), open.post.x(),
                     open.post.y(), open.s_post);
        CHECK(open.why.empty() and open.waypoints.size() == 3);
        CHECK((open.waypoints[0] - kAppr.p).norm() < 1e-4f);              // P_pre first, in the current room (y > 0)
        CHECK((open.waypoints[1] - kCentre).norm() < 1e-4f);              // then the aperture
        CHECK((open.waypoints[2] - kCross.p).norm() < 1e-4f);             // then just past the door (y < 0)
        CHECK(open.waypoints[0].y() > 0.f and open.waypoints[2].y() < 0.f);
        for (const auto& w : open.waypoints) CHECK(on_axis(w));
        CHECK(std::abs(open.s_pre - 1.6f) < 1e-4f and std::abs(open.s_post - 0.8f) < 1e-4f);
        CHECK(std::abs(open.post_yaw - kCross.yaw) < 1e-4f);            // travel heading through the door
    }

    std::println("2. a bed foot in front of the door: everything farther than 0.9 m out is occupied");
    {
        const auto bed = [](const Eigen::Vector2f& p, float) { return not (p.y() > 0.9f and p.y() < 3.f and std::abs(p.x() - 2.f) < 1.f); };
        const auto pl = dr::plan(kAppr, std::nullopt, kCentre, robot_far, false, prm, bed);
        std::println("    pre ({:.2f},{:.2f}) s={:.2f} moved {}", pl.pre.x(), pl.pre.y(), pl.s_pre, pl.pre_moved);
        CHECK(pl.why.empty() and pl.pre_moved);
        CHECK(on_axis(pl.pre));                                           // never sideways
        CHECK(pl.s_pre <= 0.9f + 1e-4f and pl.s_pre > 0.85f - 1e-4f);    // the LARGEST free standoff (the grid's step)
        CHECK(pl.s_pre >= prm.body_reach_m);
    }

    std::println("3. the whole axis blocked down to the body's reach");
    {
        const auto wall = [](const Eigen::Vector2f& p, float) { return not (p.y() > 0.f and p.y() < 3.f); };
        const auto pl = dr::plan(kAppr, std::nullopt, kCentre, robot_far, false, prm, wall);
        std::println("    why: {}", pl.why);
        CHECK(not pl.why.empty() and pl.waypoints.empty());
    }

    std::println("4. the open leaf stands on the cross target: P_post slides outward along the axis");
    {
        const auto leaf = [](const Eigen::Vector2f& p, float) { return not (p.y() < -0.5f and p.y() > -1.05f); };
        const auto pl = dr::plan(kAppr, kCross, kCentre, robot_far, true, prm, leaf);
        std::println("    post ({:.2f},{:.2f}) s={:.2f} moved {}", pl.post.x(), pl.post.y(), pl.s_post, pl.post_moved);
        CHECK(pl.why.empty() and pl.post_moved and on_axis(pl.post));
        CHECK(pl.s_post >= 1.05f - 1e-4f and pl.s_post <= 1.1f + 1e-4f);
        const auto all_leaf = [](const Eigen::Vector2f& p, float) { return not (p.y() < -0.5f); };
        CHECK(not dr::plan(kAppr, kCross, kCentre, robot_far, true, prm, all_leaf).why.empty());
    }

    std::println("5. P_pre is never farther from the hole than the robot (no corridor gate)");
    {
        // at the vantage (an open-wait there): P_pre is where it stands, the route goes on through
        const auto at = dr::plan(kAppr, kCross, kCentre, Eigen::Vector2f(2.03f, 1.58f), true, prm, free_all);
        CHECK(at.why.empty() and at.waypoints.size() == 3);
        CHECK(at.s_pre <= (Eigen::Vector2f(2.03f, 1.58f) - kCentre).norm() + 1e-4f);
        // recog_proto12 leg 2: on the axis 1.1 m out, facing the door — the published 1.6 m vantage is BEHIND it; now P_pre
        // is at its own distance (no way back), whatever its lateral offset
        const Eigen::Vector2f in_front(2.f, 1.1f);
        const auto pl = dr::plan(kAppr, kCross, kCentre, in_front, true, prm, free_all);
        std::println("    robot 1.1 m out: pre s={:.2f}", pl.s_pre);
        CHECK(pl.why.empty() and std::abs(pl.s_pre - 1.1f) < 1e-4f and on_axis(pl.pre));
        // farther than the vantage: the published vantage
        const auto far = dr::plan(kAppr, kCross, kCentre, Eigen::Vector2f(2.5f, 2.4f), true, prm, free_all);
        CHECK(far.waypoints.size() == 3 and std::abs(far.s_pre - 1.6f) < 1e-4f);
        // beside the doorway (recog_try16): the axis comes BEFORE the jambs, at the robot's distance from the hole
        const Eigen::Vector2f beside(3.f, 0.2f);
        const auto bs = dr::plan(kAppr, kCross, kCentre, beside, true, prm, free_all);
        std::println("    beside the door: pre ({:.2f},{:.2f}) s={:.2f}", bs.pre.x(), bs.pre.y(), bs.s_pre);
        CHECK(bs.waypoints.size() == 3 and on_axis(bs.pre) and std::abs(bs.s_pre - (beside - kCentre).norm()) < 1e-4f);
        // already in the doorway (nearer than the body's reach): P_pre is where the robot stands when the grid admits that
        // pose — never behind it; where it does not, it slides OUT along the axis to the first free pose, at most the reach
        const auto close = dr::plan(kAppr, kCross, kCentre, Eigen::Vector2f(2.f, 0.15f), true, prm, free_all);
        CHECK(close.why.empty() and std::abs(close.s_pre - 0.15f) < 1e-4f);
        const auto jambs = [](const Eigen::Vector2f& p, float) { return std::abs(p.y()) > 0.3f; };   // the footprint clears them 0.3 m out
        const auto cj = dr::plan(kAppr, kCross, kCentre, Eigen::Vector2f(2.f, 0.15f), true, prm, jambs);
        CHECK(cj.why.empty() and cj.s_pre > 0.3f and cj.s_pre <= prm.body_reach_m + 1e-4f);
        // recog_proto13 leg 2: IN the doorway (0.21 m), no footprint free between it and the reach: no vantage, the route
        // starts at the aperture — while a robot FARTHER than the published vantage with nothing free still fails (case 3)
        const auto doorway = [](const Eigen::Vector2f& p, float) { return std::abs(p.y()) > 0.4f; };
        const auto dw = dr::plan(kAppr, kCross, kCentre, Eigen::Vector2f(2.f, 0.21f), true, prm, doorway);
        CHECK(dw.why.empty() and dw.pre_omitted and dw.waypoints.size() == 2);
        CHECK(dw.waypoints.size() == 2 and (dw.waypoints[0] - kCentre).norm() < 1e-4f);
        const auto dws = dr::plan(kAppr, std::nullopt, kCentre, Eigen::Vector2f(2.f, 0.21f), false, prm, doorway);
        // recog_proto14 leg 2: an APPROACH leg from the doorway — already past every vantage: no waypoints, and no failure
        CHECK(dws.why.empty() and dws.pre_omitted and dws.waypoints.empty());
    }

    std::println("7. the PERCEIVED hole (interaction anchor) is the axis; the room-frame door 0.45 m off is ignored");
    {
        // the room-frame door says x = 2.45; the anchor's opening says x = 2.0, normal +y into this room
        const dr::Target appr_off{{2.45f, 1.6f}, -std::numbers::pi_v<float> / 2.f};
        const dr::Target cross_off{{2.45f, -0.8f}, -std::numbers::pi_v<float> / 2.f};
        const dr::Axis seen{kCentre, Eigen::Vector2f(0.f, 1.f)};
        const auto pl = dr::plan(appr_off, cross_off, Eigen::Vector2f(2.45f, 0.f), Eigen::Vector2f(2.f, 1.1f), true, prm, free_all, seen);
        std::println("    pre ({:.2f},{:.2f}) ap ({:.2f},{:.2f}) post ({:.2f},{:.2f})", pl.pre.x(), pl.pre.y(), pl.ap.x(), pl.ap.y(),
                     pl.post.x(), pl.post.y());
        CHECK(pl.why.empty() and pl.perceived and pl.waypoints.size() == 3);
        for (const auto& w : pl.waypoints) CHECK(on_axis(w));
        CHECK((pl.ap - kCentre).norm() < 1e-4f and std::abs(pl.s_pre - 1.1f) < 1e-4f and std::abs(pl.s_post - 0.8f) < 1e-4f);
        const auto room = dr::plan(appr_off, cross_off, Eigen::Vector2f(2.45f, 0.f), Eigen::Vector2f(2.f, 1.1f), true, prm, free_all);
        CHECK(not room.perceived and std::abs(room.ap.x() - 2.45f) < 1e-4f);   // without the anchor: the room-frame door
    }

    std::println("6. a lateral door-node offset is projected onto the published axis");
    {
        const auto pl = dr::plan(kAppr, kCross, Eigen::Vector2f(2.3f, 0.05f), robot_far, true, prm, free_all);
        CHECK(on_axis(pl.ap) and std::abs(pl.ap.y() - 0.05f) < 1e-4f);
    }
    std::println("test_door_route: {} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
