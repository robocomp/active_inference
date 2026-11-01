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
//   C. review fixes: price = one door at the ATTAINABLE P (a 1.8 m unseen stub is worth a trip); arrival heading gain
//      (a wrong-way arrival is credited by what the camera saw, and the plan still asks); unreachable targets are
//      excluded and re-planned (never "unreachable"); a too-narrow room finishes with residual > price (the warning);
//      Phase::Cameras remembers the boundary as known (a re-entry resumes the camera question, not a LiDAR survey).
#include "camera_coverage.h"
#include "layout_explorer.h"
#include "room_explored.h"

#include <clocale>
#include <cmath>
#include <numbers>
#include <algorithm>
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
    // a re-entry whose camera phase was unfinished: the camera memory reopens it (not a LiDAR rest re-survey)
    CHECK(explorer_owns_room({.proto = true, .promoted = true, .recalled_partial = false, .camera_open = true}));
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

float heading(const Eigen::Vector2f& from, const Eigen::Vector2f& to) { return std::atan2(to.y() - from.y(), to.x() - from.x()); }

// The agent's camera-phase plan (RoomSceneGraph::plan_camera_cover): gain at the ARRIVAL heading, the reachable planner,
// the price = one door at the attainable P. `raw` = plain plan_coverage (no unreachable exclusion) for comparison.
rc::layout::Plan plan(const rc::camcov::Coverage& cov, const rc::layout::FeasibleGrid& grid, const Eigen::Vector2f& robot,
                      const rc::nbv::Sensor& s, const rc::layout::Admissible& standable, bool raw = false,
                      float price = -1.f)
{
    rc::layout::ExplorerParams ep;
    ep.clearance = 0.625f;   // the agent's StartupProto.ExplorerClearance
    const rc::layout::GainFn gain = [&](const Eigen::Vector2f& v)
    { return (v - robot).squaredNorm() > 1e-9f ? cov.look_gain(v, heading(robot, v), s) : 0.f; };
    const float pr = price > 0.f ? price : cov.price(s);
    return raw ? rc::layout::plan_coverage(grid, robot, ep, standable, gain, pr)
               : rc::layout::plan_coverage_reachable(grid, robot, ep, standable, gain, pr);
}
rc::layout::FeasibleGrid grid_of(const Poly& room) { return rc::layout::FeasibleGrid(room, rc::layout::ExplorerParams{}.cell, 0.4f); }
const auto any = [](const Eigen::Vector2f&) { return true; };

// user 2026-11-01: the camera phase sent the robot into the L's short leg and left it FACING THE END WALL — its arrival
// heading. A look is a choice of heading: the published yaw is the one that shows the detector the most unseen wall.
void best_look_case()
{
    const Poly room{{0.f, 0.f}, {6.f, 0.f}, {6.f, 4.f}, {0.f, 4.f}};
    const auto s = zed();
    rc::camcov::Coverage cov;
    cov.set_outline(room);
    const Eigen::Vector2f from(3.f, 2.f), v(5.6f, 2.f);   // driven east, it would arrive 0.4 m from the east wall, facing it
    const float arrival = cov.look_gain(v, heading(from, v), s);
    const auto [g, yaw] = cov.best_look(v, s);
    float sampled = 0.f;
    for (int k = 0; k < 72; ++k) sampled = std::max(sampled, cov.look_gain(v, -kPi + kPi * float(k) / 36.f, s));
    std::println("best look at (5.6,2.0): {:.2f} m at {:.0f} deg; arrival (east) {:.2f} m; 5-deg sampled max {:.2f} m", g,
                 yaw * 180.f / kPi, arrival, sampled);
    CHECK(g > arrival);                      // facing the near wall is a poor look
    CHECK(g >= 0.9f * sampled);              // and the scan finds (nearly) the best heading
    CHECK(std::abs(cov.look_gain(v, yaw, s) - g) < 1e-5f);   // the yaw returned is the one scored
}

void synthetic_room()
{
    const Poly room{{0.f, 0.f}, {6.f, 0.f}, {6.f, 4.f}, {0.f, 4.f}};   // CCW, the published outline
    const auto s = zed();
    rc::camcov::Coverage cov;
    cov.set_outline(room);
    CHECK(cov.ready());
    CHECK(cov.cells().size() >= 20);   // 2 x (7 + 4) door-width windows
    const float pm = cov.attainable_p(s);
    std::println("attainable P(detect) for a door window: {:.3f} ⇒ price {:.3f} m", pm, cov.price(s));
    CHECK(pm > 0.f and pm <= 1.f);
    CHECK(std::abs(cov.price(s) - cov.params().w_m * pm) < 1e-6f);
    // The robot entered at the west and toured the west half, its cameras never turned east (a LiDAR-complete map).
    for (const Eigen::Vector2f p : {Eigen::Vector2f(1.5f, 1.2f), Eigen::Vector2f(1.5f, 2.8f), Eigen::Vector2f(2.5f, 2.0f)})
        look_around(cov, p, s, 0.f, 0.5f * kPi + 0.5f * s.hfov_rad);
    std::println("after the west tour: west wall seen {:.2f}, east wall seen {:.2f}, residual {:.2f} m",
                 wall_seen(cov, 0.f), wall_seen(cov, 6.f), cov.residual_length(s));
    CHECK(wall_seen(cov, 0.f) > 0.f);
    CHECK(wall_seen(cov, 6.f) == 0.f);

    const auto grid = grid_of(room);
    const Eigen::Vector2f robot(1.5f, 2.f);
    auto P = plan(cov, grid, robot, s, any);
    std::println("plan: {} target=({:.2f},{:.2f}) gain={:.2f} m price={:.2f} best_gain={:.2f} evaluated={}",
                 P.why, P.target.x(), P.target.y(), P.gain, P.price, P.best_gain, P.evaluated);
    CHECK(P.ok);
    CHECK(not P.finished);
    CHECK(std::cos(heading(robot, P.target)) > std::cos(0.5f * s.hfov_rad));   // arriving, it faces the unseen EAST wall
    // P(explored) — the explorer's own price against what is left — stays below 1 (and ≤ ½: not finished)
    rc::explored::Tracker tr;
    const auto p_open = tr.plan("room_3", rc::explored::Phase::Cameras, P.evaluated, P.best_gain, P.price, false);
    CHECK(p_open.has_value() and *p_open < 1.f and *p_open <= 0.5f);
    // the camera phase runs after a COMPLETE LiDAR survey: memory holds the boundary as known (a re-entry resumes the
    // camera question from the agent's camera memory, it does not re-survey with the LiDAR)
    CHECK(tr.memory().recall("room_3") == std::optional<float>(1.f));

    // WRONG-WAY ARRIVAL (review 4): the robot gets to the target facing back the way it came. The visit is credited by
    // what the camera actually saw: the east wall stays unseen and the next plan still asks for it — not finished.
    {
        rc::camcov::Coverage c2 = cov;
        const float back = heading(robot, P.target) + kPi;
        for (int k = -2; k <= 2; ++k) c2.observe(P.target, back + 0.1f * static_cast<float>(k), s);
        const auto Pw = plan(c2, grid, P.target, s, any);
        std::println("wrong-way arrival: east wall seen {:.2f}; next plan {} gain={:.2f}", wall_seen(c2, 6.f), Pw.why, Pw.gain);
        CHECK(wall_seen(c2, 6.f) == 0.f);
        CHECK(Pw.ok and not Pw.finished);
    }

    // UNREACHABLE (review SPEC): a barrier cuts the east half off. Plain plan_coverage stops at "unreachable"; the camera
    // phase's planner excludes those targets and re-plans — it never reports "unreachable", and finishes only when no
    // REACHABLE look is worth its price.
    {
        auto cut = grid_of(room);
        cut.block_if([](const Eigen::Vector2f& c) { return c.x() > 3.0f and c.x() < 3.7f; });
        const auto Praw = plan(cov, cut, robot, s, any, true);
        const auto Pr = plan(cov, cut, robot, s, any);
        std::println("barrier: raw {} | reachable {} target=({:.2f},{:.2f}) gain={:.2f}", Praw.why, Pr.why, Pr.target.x(),
                     Pr.target.y(), Pr.gain);
        CHECK(Pr.why != "unreachable");
        CHECK(Pr.finished or (Pr.ok and Pr.target.x() < 3.0f));
        // every standpoint worth the trip is beyond the barrier (only the east side is standable)
        const rc::layout::Admissible east = [](const Eigen::Vector2f& v) { return v.x() > 3.7f; };
        const auto Eraw = plan(cov, cut, robot, s, east, true);
        const auto Er = plan(cov, cut, robot, s, east);
        std::println("barrier, east-only standpoints: raw {} | reachable {}", Eraw.why, Er.why);
        CHECK(Eraw.why == "unreachable");
        CHECK(Er.why != "unreachable" and Er.finished);   // nothing REACHABLE is worth its price: that, only, finishes
    }

    // Go where the explorer says. The controller here IGNORES the yaw: the robot arrives facing its travel direction —
    // the heading the gain was scored at — so every trip delivers what it promised and the phase ends.
    Eigen::Vector2f at = robot;
    int trips = 0;
    for (; trips < 12 and P.ok; ++trips)
    {
        const float arrive = heading(at, P.target);
        at = P.target;
        for (int k = -2; k <= 2; ++k) cov.observe(at, arrive + 0.1f * static_cast<float>(k), s);   // dwell, a small sway
        P = plan(cov, grid, at, s, any);
    }
    std::println("after {} trips: {} — east wall seen {:.2f} (attainable {:.2f}), residual {:.2f} m", trips, P.why,
                 wall_seen(cov, 6.f), pm, cov.residual_length(s));
    CHECK(P.finished);
    CHECK(trips >= 1 and trips < 12);
    CHECK(wall_seen(cov, 6.f) > 0.5f * pm);
    const auto p_done = tr.plan("room_3", rc::explored::Phase::Cameras, P.evaluated, P.best_gain, P.price, true);
    CHECK(p_done.has_value() and *p_done == 1.f);

    // the history survives a re-estimated outline (a wall moved by 3 cm): the camera's views are not thrown away
    const float before = wall_seen(cov, 6.f);
    cov.set_outline({{0.f, 0.f}, {6.03f, 0.f}, {6.03f, 4.f}, {0.f, 4.f}});
    float after = 0.f, n = 0.f;
    for (const auto& c : cov.cells()) if (std::abs(c.c.x() - 6.03f) < 1e-3f) { after += c.seen; n += 1.f; }
    CHECK(n > 0.f and std::abs(after / n - before) < 1e-6f);
}

// Review 1: a single 1.8 m stretch of wall never shown to the camera (two door windows — a jamb return, a stub: where
// doors sit) is worth a trip. RED with the old price (a full door width at P = 1).
void unseen_stub()
{
    const Poly room{{0.f, 0.f}, {6.f, 0.f}, {6.f, 3.6f}, {0.f, 3.6f}};
    const auto s = zed();
    rc::camcov::Coverage cov;
    cov.set_outline(room);
    std::vector<std::size_t> stub;
    float stub_len = 0.f;
    for (std::size_t i = 0; i < cov.cells().size(); ++i)
        if (std::abs(cov.cells()[i].c.x() - 6.f) < 1e-3f and cov.cells()[i].c.y() < 1.8f) { stub.push_back(i); stub_len += cov.cells()[i].len; }
    CHECK(stub.size() == 2 and std::abs(stub_len - 1.8f) < 1e-4f);
    // everything else seen from everywhere, never a frame that would show the stub at all
    for (float x = 0.7f; x < 5.5f; x += 0.4f)
        for (float y = 0.7f; y < 3.0f; y += 0.4f)
            for (int k = 0; k < 36; ++k)
            {
                const float yaw = -kPi + 2.f * kPi * static_cast<float>(k) / 36.f;
                const Eigen::Vector2f p(x, y);
                if (std::ranges::all_of(stub, [&](std::size_t i) { return cov.p_detect(i, p, yaw, s) == 0.f; }))
                    cov.observe(p, yaw, s);
            }
    float stub_seen = 0.f;
    for (const auto i : stub) stub_seen = std::max(stub_seen, cov.cells()[i].seen);
    const auto grid = grid_of(room);
    const Eigen::Vector2f robot(1.5f, 1.8f);
    const auto P = plan(cov, grid, robot, s, any);
    const auto Pold = plan(cov, grid, robot, s, any, false, cov.params().w_m);   // the pre-review price
    std::println("stub: seen {:.2f}; plan {} target=({:.2f},{:.2f}) gain={:.2f} price={:.2f} | old price {:.2f}: {}",
                 stub_seen, P.why, P.target.x(), P.target.y(), P.gain, P.price, cov.params().w_m, Pold.why);
    CHECK(stub_seen == 0.f);
    CHECK(P.ok and not P.finished);
    float from_stub = 0.f;   // the trip is FOR the stub
    for (const auto i : stub) from_stub += cov.cells()[i].len * cov.p_detect(i, P.target, heading(robot, P.target), s);
    CHECK(from_stub > 0.f);
}

// Review 2: a room narrower than the envelope's band finishes at once with wall never shown — the condition the agent
// warns about (latch_camera_phase: no camera-phase visit and residual > price).
void narrow_room()
{
    const Poly room{{0.f, 0.f}, {2.0f, 0.f}, {2.0f, 2.0f}, {0.f, 2.0f}};
    const auto s = zed();
    rc::camcov::Coverage cov;
    cov.set_outline(room);
    const Eigen::Vector2f robot(1.f, 1.f);
    for (int k = 0; k < 36; ++k) cov.observe(robot, -kPi + 2.f * kPi * static_cast<float>(k) / 36.f, s);
    const auto P = plan(cov, grid_of(room), robot, s, any);
    std::println("narrow 2x2 m: plan {} residual {:.2f} m > price {:.2f}", P.why, cov.residual_length(s), cov.price(s));
    CHECK(P.finished);
    CHECK(cov.residual_length(s) > cov.price(s));
}
}   // namespace

int main()
{
    std::setlocale(LC_ALL, "");
    ownership();
    best_look_case();
    synthetic_room();
    unseen_stub();
    narrow_room();
    std::println("{} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
