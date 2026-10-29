// Task 12 (2026-10-29): a room keeps proposing epistemic poses until its BOUNDARY has been put before the CAMERAS,
// not only before the LiDAR (src/camera_coverage.h). Live recog_proto2 leg 4: room_3's outline was LiDAR-complete,
// the explorer finished, and DOOR_2 — on a wall no camera had faced — was never detected.
//
//   make -C build camera_coverage_test && ../bin/camera_coverage_test
//
//   A. ownership (rc::camcov::explorer_owns_room, the predicate RoomSceneGraph::layout_explorer_owns_room calls): a
//      promoted fresh room with camera coverage open is the explorer's; once finished it hands to the pose planner;
//      the old rules (proto survey, contested, partly recalled) are unchanged.
//   B. the explorer's own plan on a synthetic 6 x 4 m room whose outline is fully known (LiDAR-mapped) but whose EAST
//      wall no camera has faced: rc::layout::plan_coverage with the camera gain and price proposes a standpoint, the
//      look faces east, the plan is NOT finished and P(explored) stays < 1; after looking where it says, it finishes
//      (P = 1) with the east wall seen.
//   C. an unreachable best standpoint does not end the exploration: the plan re-plans to another one.
#include "camera_coverage.h"
#include "layout_explorer.h"
#include "room_explored.h"

#include <clocale>
#include <cmath>
#include <numbers>
#include <print>
#include <vector>

namespace
{
int failures = 0, checks = 0;
void check(bool ok, int line, std::string_view what)
{
    ++checks;
    if (not ok) { ++failures; std::println("FAIL line {}: {}", line, what); }
}
#define CHECK(x) check((x), __LINE__, #x)
constexpr float kPi = std::numbers::pi_v<float>;
using Poly = std::vector<Eigen::Vector2f>;

rc::nbv::Sensor zed()
{
    rc::nbv::Sensor s;   // a ZED-like pinhole (the agent reads the real one from the graph's intrinsics)
    s.hfov_rad = 110.f * kPi / 180.f;
    s.vfov_rad = 70.f * kPi / 180.f;
    s.height_m = 0.945f;   // ROBOT_GEOMETRY.md: zed mount
    return s;
}

void ownership()
{
    using rc::camcov::explorer_owns_room;
    using rc::camcov::OwnerInputs;
    // a crossing-born room: survey while proto …
    CHECK(explorer_owns_room({.proto = true, .promoted = false}));
    // … promoted with the boundary not yet camera-covered: still the explorer's
    CHECK(explorer_owns_room({.proto = true, .promoted = true, .camera_open = true}));
    // … camera coverage finished, nothing contested: the pose planner's (as before Task 12)
    CHECK(not explorer_owns_room({.proto = true, .promoted = true, .camera_open = false}));
    // contested cells still keep it, whatever the cameras say
    CHECK(explorer_owns_room({.proto = true, .promoted = true, .contested = true, .camera_open = false}));
    // a partly remembered room: until finished (unchanged)
    CHECK(explorer_owns_room({.promoted = true, .recalled_partial = true, .layout_finished = false}));
    CHECK(not explorer_owns_room({.promoted = true, .recalled_partial = true, .layout_finished = true, .camera_open = true}));
}

// One full turn of camera frames at `p` except the headings within `skip_half` of `skip_dir`.
void look_around(rc::camcov::Coverage& cov, const Eigen::Vector2f& p, const rc::nbv::Sensor& s, float skip_dir, float skip_half)
{
    for (int k = 0; k < 36; ++k)
    {
        const float yaw = -kPi + 2.f * kPi * static_cast<float>(k) / 36.f;
        if (std::abs(rc::camcov::wrap(yaw - skip_dir)) < skip_half) continue;
        cov.observe(p, yaw, s);
    }
}

float wall_seen(const rc::camcov::Coverage& cov, float x_eq)   // mean `seen` of the windows on the wall x = x_eq
{
    float a = 0.f, n = 0.f;
    for (const auto& c : cov.cells()) if (std::abs(c.c.x() - x_eq) < 1e-3f) { a += c.seen; n += 1.f; }
    return n > 0.f ? a / n : -1.f;
}

rc::layout::Plan plan(const rc::camcov::Coverage& cov, const Poly& room, const Eigen::Vector2f& robot,
                      const rc::nbv::Sensor& s, const rc::layout::Admissible& standable)
{
    rc::layout::ExplorerParams ep;
    ep.clearance = 0.625f;   // the agent's StartupProto.ExplorerClearance
    rc::layout::FeasibleGrid grid(room, ep.cell, 0.4f);
    return rc::layout::plan_coverage(grid, robot, ep, standable,
                                     [&](const Eigen::Vector2f& v) { return cov.best_look(v, s).gain; }, cov.price());
}

void synthetic_room()
{
    const Poly room{{0.f, 0.f}, {6.f, 0.f}, {6.f, 4.f}, {0.f, 4.f}};   // CCW, the published outline
    const auto s = zed();
    rc::camcov::Coverage cov;
    cov.set_outline(room);
    CHECK(cov.ready());
    CHECK(cov.cells().size() >= 20);   // 2 x (7 + 4) door-width windows
    // The robot entered at the west and toured the west half, its cameras never turned east (a LiDAR-complete map).
    for (const Eigen::Vector2f p : {Eigen::Vector2f(1.5f, 1.2f), Eigen::Vector2f(1.5f, 2.8f), Eigen::Vector2f(2.5f, 2.0f)})
        look_around(cov, p, s, 0.f, 0.5f * kPi + 0.5f * s.hfov_rad);
    std::println("after the west tour: west wall seen {:.2f}, east wall seen {:.2f}, unseen {:.2f} m",
                 wall_seen(cov, 0.f), wall_seen(cov, 6.f), cov.unseen_length());
    CHECK(wall_seen(cov, 0.f) > 0.f);
    CHECK(wall_seen(cov, 6.f) == 0.f);

    const Eigen::Vector2f robot(1.5f, 2.f);
    const auto any = [](const Eigen::Vector2f&) { return true; };
    auto P = plan(cov, room, robot, s, any);
    std::println("plan: {} target=({:.2f},{:.2f}) gain={:.2f} m price={:.2f} best_gain={:.2f} evaluated={}",
                 P.why, P.target.x(), P.target.y(), P.gain, P.price, P.best_gain, P.evaluated);
    CHECK(P.ok);
    CHECK(not P.finished);
    const auto look = cov.best_look(P.target, s);
    std::println("look from the target: yaw {:.0f} deg, gain {:.2f} m", look.yaw * 180.f / kPi, look.gain);
    CHECK(std::cos(look.yaw) > std::cos(0.5f * s.hfov_rad));   // faces the unseen EAST wall
    // P(explored) — the explorer's own price against what is left — stays below 1 (and ≤ ½: not finished)
    rc::explored::Tracker tr;
    const auto p_open = tr.plan("room_3", rc::explored::Phase::Cameras, P.evaluated, P.best_gain, P.price, false);
    CHECK(p_open.has_value() and *p_open < 1.f and *p_open <= 0.5f);

    // C. the best standpoint is unreachable (the controller refused it): the plan re-plans to another standpoint
    const Eigen::Vector2f refused = P.target;
    auto P2 = plan(cov, room, robot, s, [&](const Eigen::Vector2f& v) { return (v - refused).norm() > 0.75f; });
    std::println("with ({:.2f},{:.2f}) refused: {} target=({:.2f},{:.2f}) gain={:.2f}", refused.x(), refused.y(), P2.why,
                 P2.target.x(), P2.target.y(), P2.gain);
    CHECK(P2.ok and not P2.finished and (P2.target - refused).norm() > 0.75f);

    // Go where the explorer says and look where it says, until it is finished.
    Eigen::Vector2f at = robot;
    int trips = 0;
    for (; trips < 12 and P.ok; ++trips)
    {
        at = P.target;
        const auto lk = cov.best_look(at, s);
        for (int k = -2; k <= 2; ++k) cov.observe(at, lk.yaw + 0.1f * static_cast<float>(k), s);   // dwell, a small sway
        P = plan(cov, room, at, s, any);
    }
    std::println("after {} trips: {} — east wall seen {:.2f}, unseen {:.2f} m", trips, P.why, wall_seen(cov, 6.f), cov.unseen_length());
    CHECK(P.finished);
    CHECK(trips >= 1 and trips < 12);
    CHECK(wall_seen(cov, 6.f) > 0.f);
    const auto p_done = tr.plan("room_3", rc::explored::Phase::Cameras, P.evaluated, P.best_gain, P.price, true);
    CHECK(p_done.has_value() and *p_done == 1.f);

    // the history survives a re-estimated outline (a wall moved by 3 cm): the camera's views are not thrown away
    const float before = wall_seen(cov, 6.f);
    cov.set_outline({{0.f, 0.f}, {6.03f, 0.f}, {6.03f, 4.f}, {0.f, 4.f}});
    float after = 0.f, n = 0.f;
    for (const auto& c : cov.cells()) if (std::abs(c.c.x() - 6.03f) < 1e-3f) { after += c.seen; n += 1.f; }
    CHECK(n > 0.f and std::abs(after / n - before) < 1e-6f);
}
}   // namespace

int main()
{
    std::setlocale(LC_ALL, "");
    ownership();
    synthetic_room();
    std::println("{} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
