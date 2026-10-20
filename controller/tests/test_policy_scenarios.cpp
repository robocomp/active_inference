// Task 6 (local-motion policy): the 2-D closed loop and the catalogued wedge scenarios.
//
// Each tick (20 Hz):
//   · SENSORS (review H1): Shadow's two LiDARs at their real mounts (helios y -0.155, bpearl +0.140, robot frame), 1440
//     beams each, sigma_r 0.02, the mast occluding the body's own beams; each sweep is re-expressed in the room frame on
//     the pose BELIEVED AT CAPTURE and kept in a buffer of the last kBufferCycles = 2 cycles (the controller's room-frame
//     buffer of ~2 merged sweeps): a pose jump or drift makes the older sweep disagree with the newer, as it does live.
//   · + unseen-contact hypotheses (aggregate = false) -> Navigator::step() (navigator.h: decide() + the monitor + traps +
//     Unreachable — the same code the controller calls) -> the differential base.
//   · PLANT MISMATCH (review H2; sim2d::Base): latency 60 ms +- 20 ms jitter per command, deadband = the model's x U[0.8, 1.2]
//     per run, slip 5 % (v and w each x (1 + N(0, 0.05)) per tick). The policy keeps its nominal model.
//   · Contact = the TRUE bare hull (RobotFootprint::contains_yaw, independent of the policy's geometry) against the world's
//     surfaces every 5 mm, motion integrated in <= 5 mm sub-steps; a blocked sub-step is a contact and the body stays put.
// The cost-to-go fields are built by the Navigator from the MAP's free space (cells whose clearance to the obstacles the map
// knows is at least the body's inscribed radius — necessary for the body to be there at all).
//
// Oracles: contacts == 0 (ALL motion, release motion counted separately as well); reached unless stated; no window longer
// than 20 s without the (trapped) field decreasing; TRAJECTORY (review H3): after a trap is laid the TRUE robot centre,
// once outside the trap's disc (taken to the true frame at the moment it is laid), never enters it again before the run
// ends.
//
// Build+run (from controller/; one line):
//   g++ -std=c++23 -O2 -Wall -Wextra -pthread -I/usr/include/eigen3 tests/test_policy_scenarios.cpp
//     src/local_policy/local_policy.cpp src/local_policy/contact.cpp src/local_policy/cost_to_go.cpp src/local_policy/navigator.cpp
//     ../common/robot_footprint/robot_footprint.cpp ../common/robot_footprint/mesh_hull.cpp -o /tmp/tps && /tmp/tps
//   Env: ONLY=<scenario> runs one scenario; TRACE=<scenario>:<seed> prints every tick of that run; SEEDS=<n> (default 20).
//   DIAGNOSTIC switches (NOT the brief's configuration): SIGMA_R=<m> LiDAR range sigma (default 0.02); NO_UNSEEN=1 drops the
//     UnseenContact hypotheses from the returns (ON by default, ruling D3).
#include "sim2d/sim2d.h"
#include "../src/local_policy/local_policy.h"
#include "../src/local_policy/navigator.h"
#include "../src/local_policy/unseen_contact.h"

#include <charconv>
#include <chrono>
#include <deque>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <string>
#include <vector>

using namespace rc::policy;
using sim2d::V2;
using sim2d::World;
using sim2d::Box;
constexpr float kPi = sim2d::kPi, kDeg = kPi / 180.f;

static int fails = 0;
#define CHECK(name, cond) do { const bool ok_ = (cond); std::printf("%s %s\n", ok_ ? "PASS" : "FAIL", name); if (not ok_) ++fails; } while (0)

// ---------------------------------------------------------------- base and body
static BaseLimits limits()
{
    // ★ Task-2/4 test values (T7 measures the deadband later): 0.5 m/s, 1.5 rad/s, 1 m/s^2, 2 rad/s^2.
    BaseLimits b; b.v_max = 0.5f; b.w_max = 1.5f; b.a_lin = 1.f; b.d_lin = 1.f; b.a_rot = 2.f; b.d_rot = 2.f;
    b.v_min_eff = 0.03f; b.w_min_eff = 0.08f; return b;
}
static const rc::RobotFootprint& body() { static const rc::RobotFootprint f = rc::RobotFootprint::shadow(); return f; }
constexpr int kBufferCycles = 2;
static const sim2d::Mount kHelios{V2(0.f, -0.155f)}, kBpearl{V2(0.f, 0.140f)};
constexpr float kSlipSigma = 0.05f, kLatencyJitterS = 0.02f, kDeadbandSpread = 0.2f;

// ---------------------------------------------------------------- scenario plumbing
struct Bounds { float x0, y0, x1, y1; };
struct Jump { float t; V2 de; float deyaw; };
struct Options
{
    Bounds grid{-3.f, -3.f, 3.f, 3.f};
    bool ou = false;
    std::vector<Jump> jumps;
    std::string trace_name;     // scenario name, for TRACE matching
    // ★ ON by default (ruling D3: the free prediction follows the base's ramp, prior 0.01).
    bool unseen = std::getenv("NO_UNSEEN") == nullptr;
};

struct ScenarioResult
{
    bool reached = false, unreachable = false, stuck = false, trajectory_ok = true;
    int contacts = 0, release_contacts = 0;
    float max_no_progress_s = 0.f, path_m = 0.f;
    int releases = 0, traps = 0;
    float time_s = 0.f, geodesic_m = 0.f, min_clear = 1e9f;
    float goal_truth_clear = 1e9f;   // pose_jump: the world clearance of the TRUE place the believed goal maps to
    Pose2 end;                       // TRUE end pose
    double build_ms = 0.0; int builds = 0;
    std::vector<float> step_ms;      // wall time of every Navigator::step (decide + traps)
    std::size_t returns_max = 0;     // largest return set handed to the policy
    std::string why;
};

static std::vector<V2> goal_points(const Goal& g)
{
    if (g.kind != Goal::Kind::Doorway) return {g.p};
    // ★ "distance to being through": a line across the gap, 0.6 m past its centre along the normal, over the lateral
    //   span where the body's centre can be (half-width minus the inscribed radius; the centre point if none).
    const V2 n = g.normal.normalized(), lat(-n.y(), n.x());
    const float half = std::max(0.f, 0.5f * g.width - body().inscribed_radius());
    std::vector<V2> pts;
    for (float t = -half; t <= half + 1e-4f; t += 0.05f) pts.push_back(g.p + 0.6f * n + t * lat);
    if (pts.empty()) pts.push_back(g.p + 0.6f * n);
    return pts;
}

// The map's free cells (clearance to MAP-known obstacles >= the inscribed radius), precomputed once per run.
struct MapFree
{
    CostToGo::Grid grid;
    std::vector<char> free_;
    MapFree(const World& w, const Bounds& b)
    {
        grid = {V2(b.x0, b.y0), 0.05f, int(std::lround((b.x1 - b.x0) / 0.05f)) + 1, int(std::lround((b.y1 - b.y0) / 0.05f)) + 1};
        free_.assign(size_t(grid.w * grid.h), 0);
        const float r_ins = body().inscribed_radius();
        for (int j = 0; j < grid.h; ++j)
            for (int i = 0; i < grid.w; ++i) free_[size_t(j * grid.w + i)] = w.clearance(grid.origin + V2(i * grid.cell, j * grid.cell), true) >= r_ins;
    }
    bool operator()(const V2& c) const
    {
        const int i = int(std::lround((c.x() - grid.origin.x()) / grid.cell)), j = int(std::lround((c.y() - grid.origin.y()) / grid.cell));
        return i >= 0 and j >= 0 and i < grid.w and j < grid.h and free_[size_t(j * grid.w + i)];
    }
};

// LiDAR range sigma: 0.02 m (the brief); SIGMA_R=<m> overrides it for diagnosis only (std::from_chars: locale-proof).
static float sigma_r()
{
    static const float s = [] {
        float v = 0.02f;
        if (const char* e = std::getenv("SIGMA_R")) { const auto r = std::from_chars(e, e + std::strlen(e), v); if (r.ec != std::errc{}) v = 0.02f; }
        return v; }();
    return s;
}

static bool trace_on(const std::string& name, unsigned seed)
{
    const char* t = std::getenv("TRACE");
    if (t == nullptr) return false;
    return std::string(t) == name + ":" + std::to_string(seed);
}

// ---------------------------------------------------------------- the closed loop
static ScenarioResult run_scenario(const World& world, Pose2 start, Goal goal, float time_limit_s, unsigned seed, const Options& opt)
{
    ScenarioResult res;
    std::mt19937 rng(seed * 7919u + 13u);
    const bool tr = trace_on(opt.trace_name, seed);
    const auto& B = body();
    sim2d::ContactTester contact(world, B);
    sim2d::Base base; base.lim = limits(); base.latency_s = 0.06f;
    base.rng = &rng; base.jitter_s = kLatencyJitterS; base.slip_sigma = kSlipSigma;
    base.deadband_scale = std::uniform_real_distribution<float>(1.f - kDeadbandSpread, 1.f + kDeadbandSpread)(rng);
    std::normal_distribution<float> slip(0.f, kSlipSigma);
    sim2d::PoseError err; err.on = opt.ou; err.pivot = start.p;
    UnseenContact unseen;
    const MapFree mapfree(world, opt.grid);
    Navigator nav;
    nav.set_goal(goal, mapfree.grid, [&mapfree](const V2& c) { return mapfree(c); }, [](const V2&) { return 0.f; }, goal_points(goal),
                 B.circumscribed_radius());
    res.build_ms += nav.last_build_ms(); ++res.builds;
    PolicyParams pp;
    Pose2 truth = start;
    res.geodesic_m = nav.field().at(start.p);
    if (contact.touches(truth)) { res.why = "start pose overlaps the world"; res.contacts = 1; res.end = truth; return res; }
    if (not std::isfinite(res.geodesic_m)) { res.unreachable = true; res.why = "goal not reachable on the map"; res.end = truth; return res; }

    constexpr float kTick = 0.05f, kSub = 0.005f;
    float t = 0.f, best_f = res.geodesic_m, last_progress_t = 0.f;
    bool in_release_episode = false;
    size_t next_jump = 0;
    struct TrueTrap { V2 c; float r; bool left = false; };
    std::vector<TrueTrap> true_traps;
    std::deque<std::vector<Return>> buffer;      // last kBufferCycles sweeps (both sensors), room frame at capture

    for (; t < time_limit_s; t += kTick)
    {
        err.step(kTick, rng);
        while (next_jump < opt.jumps.size() and opt.jumps[next_jump].t <= t)
        {
            err.jump(truth.p, opt.jumps[next_jump].de, opt.jumps[next_jump].deyaw); ++next_jump;
            // the place the believed goal now maps to in the world (the scenario must ask for a place the body can be)
            for (const auto& gp : nav.goal_points()) res.goal_truth_clear = std::min(res.goal_truth_clear, world.clearance(err.to_truth(gp), false));
        }
        const Pose2 bel = err.believed(truth);

        // ---- both LiDARs from the TRUE pose, re-expressed in the room frame on the pose believed at capture, buffered
        std::vector<Return> sweep;
        const V2 bf(std::cos(bel.yaw), std::sin(bel.yaw)), br(std::sin(bel.yaw), -std::cos(bel.yaw));
        for (const auto& m : {kHelios, kBpearl})
            for (const V2& q : sim2d::scan_mount(world, truth, m, rng, sigma_r()))
                sweep.push_back({bel.p + q.x() * br + q.y() * bf, sigma_r(), 1.f});
        buffer.push_back(std::move(sweep));
        while (int(buffer.size()) > kBufferCycles) buffer.pop_front();
        std::vector<Return> ret;
        for (const auto& sw : buffer) ret.insert(ret.end(), sw.begin(), sw.end());
        if (opt.unseen) for (const auto& h : unseen.hypotheses()) ret.push_back(h);

        PolicyInput in;
        in.pose = bel; in.v_now = base.v; in.w_now = base.w;
        in.sigma_rel_xy = 0.01f; in.sigma_rel_yaw = 0.01f;                                       // odometry-scale RELATIVE error
        if (opt.ou) { in.pose_drift_sigma_m = err.sigma_xy; in.pose_drift_time_s = err.tau; }   // the OU's own process (monitor)
        in.returns_age_s = float(kBufferCycles - 1) * kTick;                                     // the older sweep's age
        in.returns = ret; in.body = &B; in.base = base.lim; in.latency_s = base.latency_s; in.now_s = t;
        const auto t_step = std::chrono::steady_clock::now();
        const Navigator::Step st = nav.step(in, pp);
        res.step_ms.push_back(float(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_step).count()));
        res.returns_max = std::max(res.returns_max, ret.size());
        const Decision& d = st.d;
        const float fnow = nav.field().at(bel.p);
        const float clear = contact.clearance(truth);
        res.min_clear = std::min(res.min_clear, clear);
        if (tr)
            std::printf("    t=%6.2f true(%+.3f,%+.3f,%+6.1f) bel(%+.3f,%+.3f) %-7s v=%+.2f w=%+.2f el=%d P=%.3f lpc=%.2f prog=%+.3f clear=%+.3f f=%.3f | %s\n",
                        t, truth.p.x(), truth.p.y(), truth.yaw / kDeg, bel.p.x(), bel.p.y(), mode_name(d.mode), d.v, d.w, d.eligible,
                        1.f - std::exp(d.log_p_safe), d.log_p_cont, d.progress_m, clear, fnow, d.why.c_str());

        if (d.mode == Decision::Mode::Arrived) { res.reached = true; res.why = "arrived"; break; }
        if (d.mode == Decision::Mode::Stuck or d.mode == Decision::Mode::Blind) { res.stuck = true; res.why = d.why; break; }
        if (d.mode == Decision::Mode::Release and not in_release_episode) { in_release_episode = true; ++res.releases; }
        if (tr and st.trap_suppressed_by_goal) std::printf("    no trap: its disc would contain the goal\n");
        if (st.trap)
        {
            ++res.traps; res.build_ms += nav.last_build_ms(); ++res.builds;
            true_traps.push_back({err.to_truth(st.trap->c), st.trap->r});
            best_f = nav.field().at(bel.p);      // the trapped field changed: progress is measured on the new one (clock kept)
            if (tr) std::printf("    TRAP at (%+.3f,%+.3f) dir (%+.2f,%+.2f) cost %.2f m; field now %.3f\n", st.trap->c.x(), st.trap->c.y(),
                                st.trap->dir.x(), st.trap->dir.y(), st.trap->cost, best_f);
        }
        if (st.unreachable) { res.unreachable = true; res.why = "no plan without the trap"; break; }

        // ---- the base: command latency (+ jitter), ramps, deadband (x scale), slip; contact every <= 5 mm
        base.send(t, d.v, d.w);
        const Pose2 before = truth;
        const float f_before_tick = nav.field().at(bel.p);
        const float v_meas = base.v, w_meas = base.w;     // odometry at the start of the tick
        const float sv = 1.f + slip(rng), sw = 1.f + slip(rng);
        bool touched = false;
        for (float s = 0.f; s < kTick - 1e-6f; s += kSub)
        {
            base.step_velocity(t + s, kSub);
            const Pose2 nxt = sim2d::integrate(truth, base.v * sv, base.w * sw, kSub);
            // the world pushes back: the body stays put and the base stalls (its wheels stop)
            if (contact.touches(nxt)) { touched = true; base.v = base.w = 0.f; continue; }
            truth = nxt;
        }
        if (touched) { ++res.contacts; if (d.mode == Decision::Mode::Release) ++res.release_contacts; }
        // odometry: the TRUE displacement along the heading at the start of the tick, and the turn
        const V2 fwd(std::cos(before.yaw), std::sin(before.yaw));
        const float ds = (truth.p - before.p).dot(fwd), dyaw = sim2d::wrap(truth.yaw - before.yaw);
        // ★ D3: the command IN EFFECT at the wheels and the measured velocity at the tick's start
        unseen.update(base.cmd_v, base.cmd_w, v_meas, w_meas, ds, dyaw, kTick, bel, base.lim, B, 0.005f);
        res.path_m += (truth.p - before.p).norm();

        // ★ TRAJECTORY ORACLE (review H3): once the true centre has left a trap's disc it must never re-enter it
        for (auto& tt : true_traps)
        {
            const float dist = (truth.p - tt.c).norm();
            if (dist > tt.r) tt.left = true;
            else if (tt.left) res.trajectory_ok = false;
        }

        const float fn = nav.field().at(err.believed(truth).p);
        if (fn < best_f - 1e-3f) { best_f = fn; last_progress_t = t + kTick; }
        if (d.mode == Decision::Mode::Drive and fn < f_before_tick) in_release_episode = false;   // Drive with field decrease
        res.max_no_progress_s = std::max(res.max_no_progress_s, t + kTick - last_progress_t);
    }
    res.time_s = t;
    res.end = truth;
    if (res.why.empty()) res.why = "time limit";
    return res;
}

// ---------------------------------------------------------------- worlds
// DOOR_1 (door frame: x along the wall, the opening's centre x = 0, room_1 at y < 0, room_2's wall face y = 0), copied from
// test_jamb_cross.cpp scene(). leaf: 0 none, 1 open 90 deg into room_1 (the copied scene), 2 open 90 deg into room_2.
// casing_inner: the casings' inner faces at +-casing_inner (0.45 = the 0.90 m clear gap).
static World door_world(int leaf, float casing_inner = 0.45f)
{
    World w;
    w.boxes = {
        {-3.06f, -0.11f, -0.53f, 0.f}, {0.53f, -0.11f, 3.06f, 0.f},
        {-1.21f, -0.24f, -0.501f, -0.13f}, {0.499f, -0.24f, 0.67f, -0.13f},
        {-0.5f, -0.20f, -casing_inner, 0.f}, {casing_inner, -0.20f, 0.5f, 0.f},
        {-1.22f, -1.47f, -1.11f, -0.24f}, {-1.67f, -1.58f, -1.11f, -1.47f},
        {-1.67f, -2.31f, -1.56f, -1.47f}, {-6.f, -2.42f, -1.15f, -2.31f},
        {-3.17f, 0.f, -3.06f, 4.11f}, {3.06f, 0.f, 3.17f, 4.11f}, {-3.2f, 4.11f, 3.2f, 4.22f},
        {-6.f, -7.f, 6.f, -6.9f}};
    if (leaf == 1) w.boxes.push_back({-0.525f, -1.15f, -0.475f, -0.20f});
    if (leaf == 2) w.boxes.push_back({-0.525f, 0.f, -0.475f, 0.95f});
    for (int i = 0; i < 10; ++i)
    {
        const float y1 = -0.25f - 4.21f * float(i) / 10.f, y0 = -0.25f - 4.21f * float(i + 1) / 10.f;
        const float x = 0.574f - 0.1f * (-0.25f - 0.5f * (y0 + y1)) / 4.21f;
        w.boxes.push_back({x, y0, x + 0.11f, y1});
    }
    return w;
}
static const Bounds kDoorGrid{-3.2f, -4.6f, 3.2f, 4.2f};
static Goal doorway(int dir)    // dir +1: into room_2, -1: back into room_1
{ Goal g{.kind = Goal::Kind::Doorway, .p = V2(0.f, -0.1f), .yaw = {}}; g.normal = V2(0.f, float(dir)); g.width = 0.90f; return g; }

// room_1's corridor west of the x ~ 0 partition (room_1 frame): partition face at x = -0.06, west wall face at x = -1.62
// (1.6 m wide... 1.56 between faces, as measured), closed at y = 0.1 and y = 4.9.
static World corridor_world()
{
    World w;
    w.boxes = {{-0.06f, 0.f, 0.05f, 5.f}, {-1.73f, 0.f, -1.62f, 5.f}, {-1.73f, 0.f, 0.05f, 0.1f}, {-1.73f, 4.9f, 0.05f, 5.f}};
    return w;
}
static const Bounds kCorridorGrid{-2.0f, -0.3f, 0.4f, 5.3f};

static World dead_end_world()
{
    World w;
    w.boxes = {
        {-4.1f, -3.1f, 4.1f, -3.f}, {-4.1f, 5.f, 4.1f, 5.1f}, {-4.1f, -3.f, -4.f, 5.f}, {4.f, -3.f, 4.1f, 5.f},   // outer
        {-4.f, 0.f, -0.4f, 2.f}, {0.4f, 0.f, 2.5f, 2.f}, {3.5f, 0.f, 4.f, 2.f},                                   // the block
        {-0.4f, 1.3f, 0.4f, 1.4f, false}};   // ★ STALE: the wall closing the 0.8 m passage is not in the map
    return w;
}
static const Bounds kDeadEndGrid{-4.2f, -3.2f, 4.2f, 5.2f};

static World open_world()
{
    World w;
    w.boxes = {{-3.1f, -3.1f, 3.1f, -3.f}, {-3.1f, 3.f, 3.1f, 3.1f}, {-3.1f, -3.f, -3.f, 3.f}, {3.f, -3.f, 3.1f, 3.f}};
    return w;
}
static const Bounds kOpenGrid{-3.2f, -3.2f, 3.2f, 3.2f};

// ---------------------------------------------------------------- reporting
struct Summary { std::vector<float> step_ms; std::size_t returns_max = 0; float goal_truth_clear = 1e9f; double build_ms = 0.0; int builds = 0; int n = 0, reached = 0, unreachable = 0, stuck = 0, contacts = 0, rel_contacts = 0, max_releases = 0, total_releases = 0;
                 bool traj_ok = true; int total_traps = 0; float max_np = 0.f, max_path_ratio = 0.f, min_clear = 1e9f; std::vector<float> tgoal; };

static Summary run_seeds(const std::string& name, const std::function<ScenarioResult(unsigned)>& fn)
{
    int nseeds = 20;
    if (const char* s = std::getenv("SEEDS")) nseeds = std::max(1, std::atoi(s));
    std::vector<std::future<ScenarioResult>> fut;
    for (int s = 0; s < nseeds; ++s) fut.push_back(std::async(std::launch::async, fn, unsigned(s)));
    Summary S;
    std::printf("== %s\n   seed reached unreach stuck contacts(rel) releases  t_goal  path  geo  max_np  min_clear  traj  why\n", name.c_str());
    for (int s = 0; s < nseeds; ++s)
    {
        const auto r = fut[s].get();
        ++S.n; S.reached += r.reached; S.unreachable += r.unreachable; S.stuck += r.stuck; S.contacts += r.contacts;
        S.rel_contacts += r.release_contacts; S.max_releases = std::max(S.max_releases, r.releases); S.total_releases += r.releases;
        S.traj_ok = S.traj_ok and r.trajectory_ok; S.total_traps += r.traps; S.max_np = std::max(S.max_np, r.max_no_progress_s);
        S.min_clear = std::min(S.min_clear, r.min_clear);
        if (r.geodesic_m > 0.f) S.max_path_ratio = std::max(S.max_path_ratio, r.path_m / r.geodesic_m);
        if (r.reached) S.tgoal.push_back(r.time_s);
        S.step_ms.insert(S.step_ms.end(), r.step_ms.begin(), r.step_ms.end()); S.returns_max = std::max(S.returns_max, r.returns_max);
        S.build_ms += r.build_ms; S.builds += r.builds; S.goal_truth_clear = std::min(S.goal_truth_clear, r.goal_truth_clear);
        std::printf("   %4d %7d %7d %5d %5d(%d) %8d %7.2f %5.2f %4.2f %6.1f %+9.3f %6d  %s\n", s, int(r.reached), int(r.unreachable), int(r.stuck),
                    r.contacts, r.release_contacts, r.releases, r.time_s, r.path_m, r.geodesic_m, r.max_no_progress_s, r.min_clear,
                    int(r.trajectory_ok), r.why.c_str());
    }
    std::ranges::sort(S.tgoal);
    const float med = S.tgoal.empty() ? NAN : S.tgoal[S.tgoal.size() / 2], mx = S.tgoal.empty() ? NAN : S.tgoal.back();
    std::printf("   -> reached %d/%d, unreachable %d, stuck %d, contacts %d (release %d), releases total %d max %d, time-to-goal median %.2f s max %.2f s, "
                "max no-progress %.1f s, min true clearance %+.3f m, max path/geodesic %.2f\n",
                S.reached, S.n, S.unreachable, S.stuck, S.contacts, S.rel_contacts, S.total_releases, S.max_releases, med, mx, S.max_np,
                S.min_clear, S.max_path_ratio);
    std::printf("   field builds: %d, mean %.1f ms each (grid as stated per world)\n", S.builds, S.builds ? S.build_ms / S.builds : 0.0);
    std::ranges::sort(S.step_ms);
    if (not S.step_ms.empty())
        std::printf("   decision time (Navigator::step, all seeds/ticks): median %.2f ms, p90 %.2f ms, max %.2f ms; returns up to %zu\n",
                    S.step_ms[S.step_ms.size() / 2], S.step_ms[S.step_ms.size() * 9 / 10], S.step_ms.back(), S.returns_max);
    return S;
}

static bool want(const char* name)
{
    const char* o = std::getenv("ONLY");
    return o == nullptr or std::strcmp(o, name) == 0;
}

static void standard_oracles(const std::string& n, const Summary& S)
{
    CHECK((n + ": contacts == 0").c_str(), S.contacts == 0);
    CHECK((n + ": reached on every seed").c_str(), S.reached == S.n);
    CHECK((n + ": no window > 20 s without field decrease").c_str(), S.max_np <= 20.f);
    CHECK((n + ": after a trap the true robot never re-enters it (trajectory)").c_str(), S.traj_ok);
}

int main()
{
    const auto& B = body();
    std::printf("body: shadow() |x| 0.284 |y| 0.230, inscribed %.3f circumscribed %.3f\n", B.inscribed_radius(), B.circumscribed_radius());

    // ---- tour3_1_leg3 (ruling: by GEOMETRY): partition face x = -0.06, flank parallel, bare hull 0.025 m clear (centre
    //      0.284 + 0.025 = 0.309 m from the face), heading +y so the RIGHT flank faces the partition; goal 0.7 m away at
    //      bearing +150 deg (behind-left, away from the wall).
    if (want("tour3_1_leg3"))
    {
        const World w = corridor_world();
        const Pose2 s{V2(-0.06f - 0.284f - 0.025f, 2.22f), 90.f * kDeg};
        const float bear = s.yaw + 150.f * kDeg;
        const Goal g{.kind = Goal::Kind::Place, .p = s.p + 0.7f * V2(std::cos(bear), std::sin(bear)), .yaw = {}, .tol_xy = 0.10f};
        sim2d::ContactTester ct(w, B);
        std::printf("tour3_1_leg3 start clearance %.4f m, goal (%.3f, %.3f)\n", ct.clearance(s), g.p.x(), g.p.y());
        Options o{.grid = kCorridorGrid, .jumps = {}, .trace_name = "tour3_1_leg3"};
        const auto S = run_seeds("tour3_1_leg3", [&](unsigned seed) { return run_scenario(w, s, g, 30.f, seed, o); });
        standard_oracles("tour3_1_leg3", S);
    }
    // ---- try101_corridor: the 1 m stub before DOOR_1 (leaf open 90 deg on the near side), start 0.37 m from the east wall
    if (want("try101_corridor"))
    {
        const World w = door_world(1);
        const float y = -1.2f, east = 0.574f - 0.1f * (-0.25f - y) / 4.21f;
        const Pose2 s{V2(east - 0.37f, y), 90.f * kDeg};
        Options o{.grid = kDoorGrid, .jumps = {}, .trace_name = "try101_corridor"};
        const auto S = run_seeds("try101_corridor", [&](unsigned seed) { return run_scenario(w, s, doorway(+1), 60.f, seed, o); });
        standard_oracles("try101_corridor", S);
    }
    // ---- try105_jamb: 0.37 m before the plane (room_2's wall face y = 0), 0.14 m east of centre, 15 deg east of the normal
    if (want("try105_jamb"))
    {
        const World w = door_world(1);
        const Pose2 s{V2(0.14f, -0.37f), 75.f * kDeg};
        sim2d::ContactTester ct(w, B);
        std::printf("try105_jamb start clearance %.4f m\n", ct.clearance(s));
        Options o{.grid = kDoorGrid, .jumps = {}, .trace_name = "try105_jamb"};
        const auto S = run_seeds("try105_jamb", [&](unsigned seed) { return run_scenario(w, s, doorway(+1), 60.f, seed, o); });
        standard_oracles("try105_jamb", S);
    }
    // ---- jamb_1p6cm_start (ruling): the case the OLD controller stalled on (task-1-report): (-0.15, -0.43), heading 104 deg,
    //      the wheel 1.6 cm from the west casing; goal through the door.
    if (want("jamb_1p6cm_start"))
    {
        const World w = door_world(1);
        // ★ at (-0.15, -0.43) this hull is 0.025 m from the leaf's jamb end, not 1.6 cm (task-1-report measured 1.6 cm on the
        //   pre-wheel body); the ruling asks for the 1.6 cm case, so x is moved west until the bare hull is 0.016 m clear.
        sim2d::ContactTester ct(w, B);
        float x = -0.15f;
        while (ct.clearance(Pose2{V2(x, -0.43f), 104.f * kDeg}) > 0.016f) x -= 0.0002f;
        const Pose2 s{V2(x, -0.43f), 104.f * kDeg};
        std::printf("jamb_1p6cm_start x = %.4f\n", x);
        std::printf("jamb_1p6cm_start start clearance %.4f m\n", ct.clearance(s));
        Options o{.grid = kDoorGrid, .jumps = {}, .trace_name = "jamb_1p6cm_start"};
        const auto S = run_seeds("jamb_1p6cm_start", [&](unsigned seed) { return run_scenario(w, s, doorway(+1), 60.f, seed, o); });
        standard_oracles("jamb_1p6cm_start", S);
    }
    // ---- door_crossing_both_ways: leaf open 90 deg into room_1 (near side going out, far side coming back), then into room_2
    if (want("door_crossing_both_ways"))
        for (int leaf : {1, 2})
        {
            const World w = door_world(leaf);
            const std::string n = std::string("door_crossing_both_ways leaf ") + (leaf == 1 ? "room_1" : "room_2");
            Options o{.grid = kDoorGrid, .jumps = {}, .trace_name = "door_crossing_both_ways"};
            const auto S = run_seeds(n, [&](unsigned seed)
            {
                const auto a = run_scenario(w, Pose2{V2(0.f, -1.0f), 90.f * kDeg}, doorway(+1), 60.f, seed, o);
                if (not a.reached) return a;
                auto b = run_scenario(w, a.end, doorway(-1), 60.f, seed + 1000u, o);
                b.contacts += a.contacts; b.release_contacts += a.release_contacts; b.releases += a.releases; b.time_s += a.time_s;
                b.path_m += a.path_m; b.geodesic_m += a.geodesic_m; b.min_clear = std::min(a.min_clear, b.min_clear);
                b.max_no_progress_s = std::max(a.max_no_progress_s, b.max_no_progress_s); b.trajectory_ok = a.trajectory_ok and b.trajectory_ok; b.traps += a.traps; b.build_ms += a.build_ms; b.builds += a.builds;
                return b;
            });
            standard_oracles(n, S);
        }
    // ---- dead_end_release: the map says the 0.8 m passage leads to the goal; a wall it does not know closes it
    if (want("dead_end_release"))
    {
        const World w = dead_end_world();
        const Goal g{.kind = Goal::Kind::Place, .p = V2(0.f, 3.5f), .yaw = {}};
        Options o{.grid = kDeadEndGrid, .jumps = {}, .trace_name = "dead_end_release"};
        const auto S = run_seeds("dead_end_release", [&](unsigned seed) { return run_scenario(w, Pose2{V2(0.f, -1.5f), 90.f * kDeg}, g, 120.f, seed, o); });
        standard_oracles("dead_end_release", S);
        CHECK("dead_end_release: it released at least once on every seed", S.total_releases >= S.n);
    }
    // ---- narrow_doorway_unreachable: 0.55 m clear gap (the body is 0.568 m wide) -> unreachable, no contact, <= 2 releases
    if (want("narrow_doorway_unreachable"))
    {
        const World w = door_world(1, 0.275f);
        Options o{.grid = kDoorGrid, .jumps = {}, .trace_name = "narrow_doorway_unreachable"};
        Goal g = doorway(+1); g.width = 0.55f;
        const auto S = run_seeds("narrow_doorway_unreachable", [&](unsigned seed) { return run_scenario(w, Pose2{V2(0.f, -1.0f), 90.f * kDeg}, g, 60.f, seed, o); });
        CHECK("narrow_doorway_unreachable: contacts == 0", S.contacts == 0);
        CHECK("narrow_doorway_unreachable: unreachable on every seed", S.unreachable == S.n);
        CHECK("narrow_doorway_unreachable: <= 2 releases", S.max_releases <= 2);
    }
    // ---- goal_behind_open_space: path <= 1.5 x geodesic
    if (want("goal_behind_open_space"))
    {
        const World w = open_world();
        const Goal g{.kind = Goal::Kind::Place, .p = V2(-2.f, 0.5f), .yaw = {}};
        Options o{.grid = kOpenGrid, .jumps = {}, .trace_name = "goal_behind_open_space"};
        const auto S = run_seeds("goal_behind_open_space", [&](unsigned seed) { return run_scenario(w, Pose2{V2(0.f, 0.f), 0.f}, g, 30.f, seed, o); });
        standard_oracles("goal_behind_open_space", S);
        CHECK("goal_behind_open_space: path <= 1.5 x geodesic", S.max_path_ratio <= 1.5f);
        // the same with the measured OU pose-error process (0.5 m / 13 deg, tau 20 s): contact is judged on the true pose
        Options ou = o; ou.ou = true; ou.trace_name = "goal_behind_open_space_ou";
        const auto S2 = run_seeds("goal_behind_open_space + OU pose error", [&](unsigned seed) { return run_scenario(w, Pose2{V2(0.f, 0.f), 0.f}, g, 30.f, seed, ou); });
        CHECK("goal_behind_open_space OU: contacts == 0", S2.contacts == 0);
        CHECK("goal_behind_open_space OU: reached on every seed", S2.reached == S2.n);
    }
    // ---- pose_jump: 0.3 m / 10 deg jump of the believed pose mid-corridor -> contacts 0, reached
    if (want("pose_jump"))
    {
        const World w = corridor_world();
        const Goal g{.kind = Goal::Kind::Place, .p = V2(-0.84f, 4.2f), .yaw = {}};
        // ★ +0.3 m east and +10 deg (CCW). The scenario asks the policy to stay out of contact through the jump, not to
        //   reach a goal inside a wall: at the jump the harness records the world clearance of the TRUE place the believed
        //   goal now maps to, and that place must be one the body can occupy (clearance >= the inscribed radius).
        Options o{.grid = kCorridorGrid, .jumps = {{2.5f, V2(0.3f, 0.f), 10.f * kDeg}}, .trace_name = "pose_jump"};
        const auto S = run_seeds("pose_jump", [&](unsigned seed) { return run_scenario(w, Pose2{V2(-0.84f, 0.8f), 90.f * kDeg}, g, 40.f, seed, o); });
        standard_oracles("pose_jump", S);
        std::printf("   the believed goal maps, after the jump, to a true place %.3f m from the nearest wall\n", S.goal_truth_clear);
        CHECK("pose_jump: the jumped goal is a place the body can be (true clearance >= inscribed radius)", S.goal_truth_clear >= body().inscribed_radius());
    }
    std::printf("%s (%d failing)\n", fails == 0 ? "ALL PASS" : "FAILURES", fails);
    return fails == 0 ? 0 : 1;
}
