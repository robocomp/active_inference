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
//   Env: POLICY_CAMERA=0 turns the Task-R camera model OFF (default ON, as in the controller); POLICY_NOFACING=1 drops the doorway facing term;
//        ONLY=<scenario> runs one scenario; TRACE=<scenario>:<seed> prints every tick of that run; SEEDS=<n> (default 20).
//   DIAGNOSTIC switches (NOT the brief's configuration): SIGMA_R=<m> LiDAR range sigma (default 0.02); NO_UNSEEN=1 drops the
//     UnseenContact hypotheses from the returns (ON by default, ruling D3); NO_JAMBS=1 drops F4's measured jambs from a
//     Doorway's field; NO_LEAF=1 leaves leaf_known_hinge_side's smear unexplained (A/B).
#include "sim2d/sim2d.h"
#include "../src/local_policy/local_policy.h"
#include "../src/local_policy/navigator.h"
#include "../src/local_policy/unseen_contact.h"
#include "../src/local_policy/doorway_corridor.h"
#include "../src/local_policy/door_leaf.h"
#include "../src/grid_planner.h"
#include "../src/route_spline.h"
#include "../src/route_optimizer.h"
#include "../src/trajectory_controller.h"
#include "../src/chain_monitor/chain_monitor.h"

#include <charconv>
#include <chrono>
#include <deque>
#include <fstream>
#include <string_view>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <mutex>
#include <string>
#include <vector>

using namespace rc::policy;
using sim2d::V2;
using sim2d::World;
using sim2d::Box;
constexpr float kPi = sim2d::kPi, kDeg = kPi / 180.f;

static int fails = 0;
// ★ Plant / tracking sweeps (Fable review 2026-11-10): PLANT_LAT (s), PLANT_TAU (s), PLANT_GAIN, SIGMA_TRACK (m). Unset = old values.
static float env_or(const char* k, float d) { float v = d; if (const char* e = std::getenv(k)) { const auto r = std::from_chars(e, e + std::strlen(e), v); if (r.ec != std::errc{}) v = d; } return v; }
static void plant_from_env(sim2d::Base& b) { b.latency_s = env_or("PLANT_LAT", b.latency_s); b.tau_s = env_or("PLANT_TAU", 0.f); b.gain = env_or("PLANT_GAIN", 1.f); }
// ★ PROFILE of the hybrid tick's parts (edge suite: whole-tick spikes of 120-280 ms); printed at the end of main
struct Prof { const char* name; double sum = 0; double max = 0; long n = 0; std::vector<float> all; };
static Prof g_prof[] = {{"start_episode (corridor + field)"}, {"replan_live (marks + planner + spline)"}, {"passability (connectivity)"},
                        {"Navigator::step"}, {"TrajectoryController::compute"}, {"band (spline.deform)"}, {"marks seen-free"}, {"window test"}, {"  gp.set_world"}, {"  gp.plan (A*)"}, {"  spline.build (+optimizer)"}};
static std::mutex g_prof_mx;   // run_seeds runs seeds on std::async threads: the profile is shared (unlocked, it corrupted the heap)
struct ProfScope { int k; std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    ~ProfScope() { const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                   const std::lock_guard lk(g_prof_mx);
                   auto& P = g_prof[k]; P.sum += ms; P.max = std::max(P.max, ms); ++P.n; P.all.push_back(float(ms)); } };
static bool camera_model_on() { const char* e = std::getenv("POLICY_CAMERA"); return e == nullptr or std::string(e) != "0"; }
static long g_moves = 0, g_rev = 0, g_rel = 0, g_rk1 = 0, g_rk2 = 0, g_rk0 = 0, g_rtight = 0, g_rclear = 0;   // Task R: Drive/Release decisions and how many reversed
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
    // F2 scenario (c): the base's velocity (and the command in force) at t = 0 — a run that starts MOVING
    float v0 = 0.f, w0 = 0.f;
    // F1 scenario (a): the live base's measured wheel deceleration (Policy.WheelDecelMps2 0.5) for plant AND policy; 0 = limits()
    float d_lin = 0.f;
    // F4 DIAGNOSTIC (NOT the brief's configuration): NO_JAMBS=1 drops the measured jambs from a Doorway's field (A/B)
    bool no_jambs = std::getenv("NO_JAMBS") != nullptr;
    // Task 3b: residual's cells as the controller's planner holds them (centres, nats; +inf = HARD: a live return, occupied),
    // and the door's leaf with the hypothesis-dependent excess of the soft ones — a KNOWN obstacle in the field's density, as
    // the controller's extra_fn adds it (door_leaf.h door_leaf_density); empty = none
    std::vector<V2> res_cells = {};
    std::vector<float> res_risk = {};
    DoorLeafEvidence leaf = {};
    // APARTMENT STRESS (2026-11-10): the goal's position re-measured every tick with this isotropic sigma (m) — a cross on a
    // PREDICTED anchor (clean38: the doorway target moved 0.2-0.3 m per decision); the field is rebuilt every tick as the
    // controller rebuilds it on every new anchor. 0 = a still goal.
    float goal_jitter_m = 0.f;
    // LiDAR stalls: [t0, t1) windows with no new sweep (the buffer keeps the last ones, their age grows)
    std::vector<std::pair<float, float>> lidar_dropout = {};
    // ★ MOVING OBSTACLES (edge suite 2026-11-11; hybrid regime only): a box of half extents `half` following `path` (time,
    //   centre) on its OWN clock, linear between points; it is not in the map. A person does not walk into a robot: a step
    //   that would bring it within 0.10 m of the robot's true hull is not taken (its clock waits).
    struct Mover { V2 half; std::vector<std::pair<float, V2>> path;
                   // a LiDAR-flickering obstacle (a person's legs): a fraction gap_frac of the time the sensor sees through it, in
                   // gaps of mean length gap_s (a two-state Markov chain; it still collides while seen through)
                   float gap_frac = 0.f, gap_s = 0.3f; };
    std::vector<Mover> movers = {};
    // THE SPLINE CHAIN (2026-11-10): REGIME=chain drives every scenario with planner -> optimised spline -> band -> PLAIN
    // tracker (run_chain) instead of the look-ahead policy; same world, plant, sensors and oracles
    bool chain = std::getenv("REGIME") != nullptr and (std::string(std::getenv("REGIME")) == "chain" or std::string(std::getenv("REGIME")) == "hybrid");
    // THE HAND-OFF (controller-two-regimes §3.2): the chain drives; a stretch it cannot drive goes to the look-ahead policy as one
    // episode toward a local subgoal on the route
    bool hybrid = std::getenv("REGIME") != nullptr and std::string(std::getenv("REGIME")) == "hybrid";
    // the specialist's field over the chain's route corridor (default) or over its own map (CORRIDOR=0, for comparison)
    // the ORACLE's yardstick on the TRUE world (things the map lacks included): for a scenario whose point is that the map is
    // wrong (a shut door), progress measured on the map's field reads every correct detour as retreat
    bool truth_yardstick = false;
    bool corridor_field = std::getenv("CORRIDOR") == nullptr or std::string(std::getenv("CORRIDOR")) != "0";
    // ★ Fable review 2026-11-10 §3: the two 8 s clocks replaced by model quantities (MONITOR=0 = the old clocks, for A/B):
    //   chain: predicted vs realised arc length, judged against the projection's resolution; episode: passability of the
    //   stretch from the contact model; the policy's hand-back re-plans instead of ending the run.
    bool monitors = std::getenv("MONITOR") == nullptr or std::string(std::getenv("MONITOR")) != "0";
    bool live_corridor = std::getenv("LIVE_CORR") != nullptr and std::string(std::getenv("LIVE_CORR")) == "1";   // OFF: narrowing discs to live clearance re-created the clean38 disconnection (apt_person_in_door seed 2: 50 s of Release circles)
    bool clear_marks = std::getenv("CLEAR_MARKS") == nullptr or std::string(std::getenv("CLEAR_MARKS")) != "0";
    bool plan_first = std::getenv("PLAN_FIRST") == nullptr or std::string(std::getenv("PLAN_FIRST")) != "0";
    bool wait_blocked = std::getenv("WAIT_BLOCKED") == nullptr or std::string(std::getenv("WAIT_BLOCKED")) != "0";
    bool pass_ends_episode = std::getenv("PASS_ENDS") == nullptr or std::string(std::getenv("PASS_ENDS")) != "0";
    bool fail_marks = std::getenv("FAIL_MARKS") == nullptr or std::string(std::getenv("FAIL_MARKS")) != "0";
    // MARK BELIEF (2026-11-11): a mark clears when P(the obstacle left) >= 1 - risk_eps, not on one see-through sweep
    bool mark_belief = std::getenv("MARK_BELIEF") == nullptr or std::string(std::getenv("MARK_BELIEF")) != "0";
    // REPLAN_ON_CLEAR=0: a cleared mark re-plans only while WAITING (live chain9: 33 of 47 re-plans were clears near the route)
    bool replan_on_clear = std::getenv("REPLAN_ON_CLEAR") == nullptr or std::string(std::getenv("REPLAN_ON_CLEAR")) != "0";
    bool anchor_marks = std::getenv("ANCHOR_MARKS") == nullptr or std::string(std::getenv("ANCHOR_MARKS")) != "0";
    // the localiser's REPORTED position sigma; < 0 = the mock stand-in (the true misplacement). A parked live robot's
    // covariance collapses (memory: pose-covariance-collapses-when-parked) — loc_sigma_reported = 0 reproduces that.
    float loc_sigma_reported = -1.f;
    float map_sigma = env_or("MAP_SIGMA", 0.02f);   // ★FLAGGED: the map's own position sigma (live: room_concept's published layout sigma, ~2 cm floor)
    // the OU pose error's stationary sigmas (< 0 = sim2d's defaults, 0.5 m / 13 deg: the catalogue's stress level)
    float ou_sigma_xy = -1.f, ou_sigma_yaw = -1.f;
};

struct ScenarioResult
{
    bool reached = false, unreachable = false, stuck = false, trajectory_ok = true;
    int contacts = 0, release_contacts = 0;
    float max_no_progress_s = 0.f, path_m = 0.f;
    int releases = 0, traps = 0, failed_predictions = 0;
    float time_s = 0.f, geodesic_m = 0.f, min_clear = 1e9f;
    float goal_truth_clear = 1e9f;   // pose_jump: the world clearance of the TRUE place the believed goal maps to
    Pose2 end;                       // TRUE end pose
    double build_ms = 0.0; int builds = 0;
    std::vector<float> step_ms;      // wall time of every Navigator::step (decide + traps)
    std::size_t returns_max = 0;     // largest return set handed to the policy
    std::string why;
    // F2: the first decision, every Release(stop), and the first decision made after a stop (v = w = 0 at rest)
    std::string first_mode; float first_v = 0.f, first_w = 0.f;
    bool stuck_moving = false;
    // F1: a Stuck is a HOLD (the controller holds and asks again); the hand back is the reachability belief's (Step::hand_back)
    int stuck_ticks = 0; bool hand_back = false; float hand_back_L = 0.f, hand_back_bar = 0.f;
    float first_no_escape_t = -1.f;     // the first decision that found no escape (Stuck / no-gain release / stop)
    int backups = 0;                    // F1 fix round 1 (I3): decisions (replanned) that back up (v < 0)
    int stops = 0; std::string after_stop_mode; float after_stop_v = 0.f, after_stop_w = 0.f;
    // APARTMENT STRESS: reverse travel where the TRUE bare-hull clearance is >= 0.5 m (open space: should be ~0), and the
    // largest RETREAT — how far the true pose's distance-to-go (on the still goal's first field) rose above its best
    float open_reverse_m = 0.f, max_retreat_m = 0.f;
    // THE CHAIN: the true centre's distance from the route it follows, while the chain drives (for sigma_track); HYBRID: how
    // many times the chain handed a stretch to the policy, and why
    double xtrack_sum2 = 0.0; long xtrack_n = 0;
    int handoffs = 0, handoff_window = 0, handoff_stall = 0, replans = 0, waits = 0, wait_exits = 0;
};

// ★ Task 8 ruling I3 — ONE definition of "through": the PROTOCOL's (the controller's rc::cross_progress_complete: s >= the
//   standoff along the gap normal). decide() never reports Arrived for a Doorway; the driver checks this predicate every
//   tick, and the field's goal line lies 2 tol_xy BEYOND it (real slack, as the controller's goal_adapter places it).
// 0.35 m = where the harness's crossings completed before I3 (field within tol_xy 0.25 of a goal line 0.6 m past the gap), so
// the reviewed time-to-goal baselines measure the same completion point.
constexpr float kThroughS = 0.35f;
// APARTMENT (2026-11-10): THROUGH THE OPENING, not merely past its line — a half-plane let a detour count (DOOR_0's far side is
// also reachable around room_1's partition, so a SHUT door read "reached"). The lateral bound is the opening's half width: a body
// that went through the door had its centre within it.
static bool doorway_through(const Goal& g, const V2& p)
{
    const V2 n = g.normal.normalized(), lat(-n.y(), n.x());
    const bool past = (p - g.p).dot(n) >= kThroughS;
    return past and (g.width <= 0.f or std::abs((p - g.p).dot(lat)) <= 0.5f * g.width);   // a body through a door has its centre within it
}
static std::vector<V2> goal_points(const Goal& g)
{
    if (g.kind != Goal::Kind::Doorway) return {g.p};
    // "distance to being through": a line across the gap, kThroughS + 2 tol_xy past its centre along the normal, over the
    // lateral span where the body's centre can be (half-width minus the inscribed radius; the centre point if none).
    const V2 n = g.normal.normalized(), lat(-n.y(), n.x());
    const float half = std::max(0.f, 0.5f * g.width - body().inscribed_radius());
    const float past = kThroughS + 2.f * g.tol_xy;
    std::vector<V2> pts;
    for (float t = -half; t <= half + 1e-4f; t += 0.05f) pts.push_back(g.p + past * n + t * lat);
    if (pts.empty()) pts.push_back(g.p + past * n);
    return pts;
}

// The map's free cells (clearance to MAP-known obstacles >= the inscribed radius), precomputed once per run.
struct MapFree
{
    CostToGo::Grid grid;
    std::vector<char> free_;
    MapFree(const World& w, const Bounds& b, std::span<const V2> hard_cells = {}, float cell = 0.05f, bool map_only = true)
    {
        grid = {V2(b.x0, b.y0), 0.05f, int(std::lround((b.x1 - b.x0) / 0.05f)) + 1, int(std::lround((b.y1 - b.y0) / 0.05f)) + 1};
        free_.assign(size_t(grid.w * grid.h), 0);
        const float r_ins = body().inscribed_radius();
        for (int j = 0; j < grid.h; ++j)
            for (int i = 0; i < grid.w; ++i)
            {
                const V2 q = grid.origin + V2(i * grid.cell, j * grid.cell);
                bool f = w.clearance(q, map_only) >= r_ins;
                // Task 3b: residual's HARD cells, as the planner's distance field holds them (field_free_from_distance's rule)
                for (const auto& c : hard_cells) f = f and (c - q).norm() - 0.5f * cell > r_ins;
                free_[size_t(j * grid.w + i)] = f;
            }
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

// ---------------------------------------------------------------- THE SPLINE CHAIN (2026-11-10)
// The controller's chain for a single target, as the session drives it with Policy.Enabled = false
// (controller_session.cpp smooth_plan / step_route_band; values from etc/config.toml [Planner] and [Controller]):
//   GridPlanner (cell 0.06, clearance 1.5 / 0.9) on the MAP -> RouteSpline with make_route_optimizer_config's settings ->
//   every tick TrajectoryController::compute(pose, cloud) (PLAIN; its live ESDF from this tick's LiDAR) -> the BAND on the
//   window ahead against that ESDF (step_route_band's lead/window/proximal anchors) -> update_path_geometry.
// MOCK SIMPLIFICATIONS, stated: a confirmed blockage (path_blocked) re-plans from the believed pose with the blocked disc as
// an obstacle (the session's local repair is not reproduced); the room model injected into the ESDF is the map's walls and
// boxes (the session's room polygon + furniture); a jittering goal does not re-plan (see the loop). Frames: the controller's robot
// frame is +x right, +y forward, theta = yaw - pi/2 (ControllerRobotPose::as_transform); its rot is CLOCKWISE (plain_tracker:
// out.rot = -omega), so the base's CCW rate is -rot.
// A GOAL WITH A SIGMA IS A REGION (controller-two-regimes §3.1): the route ends at the reachable place that maximises the
// Gaussian terminal likelihood N(x; goal, sigma) — with no travel term in the terminal cost that is the reachable free place
// NEAREST the target, for any sigma (no standoff constant). A target the body can stand on is unchanged. Reachable = finite
// geodesic from `from` on the map. Live: a remembered chair's centre (policy_field_dump_1794309704136: goal_point free 0).
static Goal region_goal(const World& w, const Bounds& grid, const V2& from, Goal g)
{
    if (g.kind != Goal::Kind::Place) return g;
    const MapFree mf(w, grid);
    if (mf(g.p)) return g;
    CostToGo f;
    const std::vector<V2> src{from};
    f.build(mf.grid, [&mf](const V2& c) { return mf(c); }, [](const V2&) { return 0.f; }, src);
    float best = 1e9f; V2 bp = g.p;
    for (int j = 0; j < mf.grid.h; ++j)
        for (int i = 0; i < mf.grid.w; ++i)
        {
            const V2 c = mf.grid.origin + V2(float(i) * mf.grid.cell, float(j) * mf.grid.cell);
            if (not mf(c) or not std::isfinite(f.at(c))) continue;
            if (const float d = (c - g.p).norm(); d < best) { best = d; bp = c; }
        }
    g.p = bp;
    return g;
}

static ScenarioResult run_chain(const World& world, Pose2 start, Goal goal, float time_limit_s, unsigned seed, const Options& opt)
{
    ScenarioResult res;
    std::mt19937 rng(seed * 7919u + 13u);
    const bool tr = trace_on(opt.trace_name, seed);
    const auto& B = body();
    sim2d::ContactTester contact(world, B);
    World wnow = world;                                   // the world as it is NOW (movers placed); the map never sees them
    std::vector<float> mclock(opt.movers.size(), 0.f);
    World wscan = world;                                  // what the LiDAR sees now (a flickering mover may be absent)
    std::vector<char> mgap(opt.movers.size(), 0);
    const auto mover_at = [&](std::size_t k, float c) {
        const auto& P = opt.movers[k].path;
        if (c <= P.front().first) return P.front().second;
        for (std::size_t i = 1; i < P.size(); ++i)
            if (c <= P[i].first) { const float a = (c - P[i - 1].first) / std::max(1e-6f, P[i].first - P[i - 1].first); return V2(P[i - 1].second + a * (P[i].second - P[i - 1].second)); }
        return P.back().second; };
    sim2d::Base base; base.lim = limits(); base.latency_s = 0.06f; plant_from_env(base);
    base.rng = &rng; base.jitter_s = kLatencyJitterS; base.slip_sigma = kSlipSigma;
    base.deadband_scale = std::uniform_real_distribution<float>(1.f - kDeadbandSpread, 1.f + kDeadbandSpread)(rng);
    std::normal_distribution<float> slip(0.f, kSlipSigma);
    sim2d::PoseError err; err.on = opt.ou; err.pivot = start.p;
    if (opt.ou_sigma_xy >= 0.f) err.sigma_xy = opt.ou_sigma_xy;
    if (opt.ou_sigma_yaw >= 0.f) err.sigma_yaw = opt.ou_sigma_yaw;
    // the MEASURING STICK, not the driver: the policy harness's field on the map, for progress, retreat and the geodesic
    const MapFree mapfree(world, opt.grid);
    const MapFree yard(world, opt.grid, {}, 0.05f, not opt.truth_yardstick);
    Navigator stick;
    stick.set_goal(goal, yard.grid, [&yard](const V2& c) { return yard(c); }, [](const V2&) { return 0.f; }, goal_points(goal),
                   B.circumscribed_radius());
    CostToGo ref_field = stick.field();
    res.geodesic_m = ref_field.at(start.p);
    if (contact.touches(start)) { res.why = "start pose overlaps the world"; res.contacts = 1; res.end = start; return res; }
    const bool region_case = opt.hybrid and goal.kind == Goal::Kind::Place;   // a target the body cannot stand on is resolved below
    if (not std::isfinite(res.geodesic_m) and not region_case) { res.unreachable = true; res.why = "goal not reachable on the map"; res.end = start; return res; }

    // ---- the chain, configured as etc/config.toml configures it
    rc::GridPlanner gp;
    gp.params.cell_size_m = 0.06f; gp.params.clearance_weight = 1.5f; gp.params.clearance_pref_m = 0.9f;
    gp.set_footprint(B);
    const Bounds& g = opt.grid;
    const std::vector<V2> roomp = {V2(g.x0, g.y0), V2(g.x1, g.y0), V2(g.x1, g.y1), V2(g.x0, g.y1)};
    std::vector<std::vector<V2>> map_obst;
    for (const auto& s : world.segments)
        if (s.in_map)
        {
            const V2 d = (s.b - s.a); const float L = d.norm(); if (L < 1e-4f) continue;
            // half-width cell * sqrt(2) / 2: the planner marks a cell only if its CENTRE is inside a polygon (grid_planner.cpp,
            // "sub-cell slivers are missed by design"), so a 2 cm wall polygon on a 6 cm grid marked no cells and every interior
            // wall was invisible to the chain — it planned through room_1's north wall (mock bug, 2026-11-10). Live the free space
            // is the room polygon itself. This width always covers a connected chain of cell centres; it moves each face ~4 cm in.
            const V2 n = (0.5f * 1.4143f * gp.params.cell_size_m) * V2(-d.y(), d.x()) / L;
            map_obst.push_back({s.a - n, s.b - n, s.b + n, s.a + n});
        }
    for (const auto& b : world.boxes)
        if (b.in_map) map_obst.push_back({V2(b.x0, b.y0), V2(b.x1, b.y0), V2(b.x1, b.y1), V2(b.x0, b.y1)});
    gp.set_world(roomp, map_obst);
    const auto opt_cfg = [&]() {
        rc::RouteOptimizerConfig o;
        o.enabled = true;
        o.distance = [&gp](const V2& p) { return gp.distance_at(p); };
        o.distance_gradient = [&gp](const V2& p) { return gp.distance_gradient_at(p); };
        o.clearance_barrier = true; o.clear_sigma_m = 0.20f;
        o.d_target = B.circumscribed_radius() + 0.6f;                   // ComfortStandoff 0.6
        o.rho = base.lim.v_max * base.lim.v_max / 1.0f;                // MaxLateralAccel 1.0
        o.sigma_a = 0.30f;
        o.support_radius = [&B](float yaw, const V2& d) { return B.support_radius_yaw(yaw, d); };
        o.clearance_floor = B.inscribed_radius();
        o.iterations = 30; o.safety_bias = 0.75f; o.safety_margin = 0.10f; o.w_jerk = 0.5f;
        return o; };
    rc::RouteSpline spline;
    rc::TrajectoryController tc;
    tc.set_footprint(B);
    tc.set_control_mode(rc::TrajectoryController::ControlMode::PLAIN);
    tc.params.plain_L = 0.50f; tc.params.plain_T_lag = 0.42f; tc.params.plain_g_dc = 1.124f; tc.params.plain_W = 0.40f;
    tc.params.plain_rot_headroom = 0.70f; tc.params.plain_brake_k = 0.25f; tc.params.plain_proj_window = 2.0f;
    tc.params.carrot_rate_limit_factor = 3.0f; tc.params.carrot_lookahead = 2.0f;
    tc.params.blockage_esdf_threshold = 0.20f; tc.params.blockage_min_waypoints = 4; tc.params.blockage_lookahead_m = 1.5f;
    tc.params.blockage_confirm_cycles = 15; tc.params.blockage_cooldown_cycles = 100;
    tc.params.max_adv = base.lim.v_max;      // the mock's base (the policy runs on the same 0.5 m/s); MaxAdvSpeed is 0.6 live
    tc.params.max_rot = 1.0f;                // MaxRotSpeed
    tc.params.d_safe = 0.6f;                 // ComfortStandoff
    // the room MODEL injected into every ESDF build, as the session does (set_room_boundary + the obstacle tracker's
    // set_static_obstacles): the controller drops LiDAR returns within ~0.44 m of its centre as self-returns, and without the
    // model a wall that close is invisible to it. Here the model is the map-known walls and boxes (the doors stand open).
    tc.set_static_obstacles(map_obst);
    const auto target_of = [](const Goal& gl) -> V2 {
        return gl.kind == Goal::Kind::Doorway ? V2(gl.p + (kThroughS + 2.f * gl.tol_xy) * gl.normal.normalized()) : gl.p; };
    float s_prog = 0.f;
    const auto install = [&](const V2& from, const V2& to) -> bool {
        const auto poly = [&] { ProfScope ps_{9}; return gp.plan(from, to); }();
        if (not poly.has_value() or poly->size() < 2) return false;
        const auto ocfg = opt_cfg();
        const bool built = [&] { ProfScope ps_{10}; return spline.build(*poly, 0.05f, [&gp](const V2& p, float th) { return gp.pose_free(p, th); }, 0.40f, &ocfg); }();
        if (not built) return false;
        tc.set_path_presmoothed(spline.samples());
        tc.set_route(&spline, true);
        s_prog = 0.f;
        return true; };
    // THE GOAL REGION, resolved with the CHAIN's own feasibility (the body + margin at some heading, gp.pose_free) and
    // reachability (map geodesic from the start): the nearest such place to the target. A plannable target is unchanged.
    if (opt.hybrid and goal.kind == Goal::Kind::Place and not install(start.p, goal.p))
    {
        CostToGo reach;
        const std::vector<V2> src{start.p};
        reach.build(mapfree.grid, [&mapfree](const V2& c) { return mapfree(c); }, [](const V2&) { return 0.f; }, src);
        std::vector<std::pair<float, V2>> cand;
        for (int j = 0; j < mapfree.grid.h; ++j)
            for (int i = 0; i < mapfree.grid.w; ++i)
            {
                const V2 c = mapfree.grid.origin + V2(float(i) * mapfree.grid.cell, float(j) * mapfree.grid.cell);
                if (std::isfinite(reach.at(c))) cand.push_back({(c - goal.p).norm(), c});
            }
        std::ranges::sort(cand, {}, &std::pair<float, V2>::first);
        for (const auto& [d, c] : cand)
        {
            bool fits = false;
            for (int k = 0; k < 16 and not fits; ++k) fits = gp.pose_free(c, float(k) * kPi / 8.f);
            if (fits) { if (tr) std::printf("    goal region: target (%+.2f,%+.2f) -> (%+.2f,%+.2f), %.2f m away\n", goal.p.x(), goal.p.y(), c.x(), c.y(), d); goal.p = c; break; }
        }
        // the yardstick follows the goal the robot is sent to
        stick.set_goal(goal, yard.grid, [&yard](const V2& c) { return yard(c); }, [](const V2&) { return 0.f; }, goal_points(goal),
                       B.circumscribed_radius());
        ref_field = stick.field();
        res.geodesic_m = ref_field.at(start.p);
    }
    V2 target = target_of(goal);
    if (not install(start.p, target)) { res.unreachable = true; res.why = "chain: no plan (" + gp.last_failure() + ")"; res.end = start; return res; }

    // ---- THE HAND-OFF (hybrid). sigma_track: the chain's lateral tracking error. ★FLAGGED: 0.05 m is what THIS mock measures
    //      (cross-track rms 0.024-0.049 m); the robot measured 0.086 and 0.153 m (tools/tracker_sim.cpp) — live it must be the
    //      measured value. risk_eps: the policy's own chance constraint (PolicyParams), not a new number.
    const float kSigmaTrack = env_or("SIGMA_TRACK", 0.05f), kSigmaTrackYaw = 0.05f;
    const PolicyParams pp;
    const float lp_ok = std::log1p(-pp.risk_eps);
    bool episode = false;                       // the policy is driving
    bool episode_final = false;                 // the episode's goal is the run's own goal: its Arrived ends the run
    float ep_s_end = 0.f;                       // arc length where the episode's stretch ends (its subgoal, or one band window)
    // ★ CHAIN PROGRESS MONITOR: what the tracker's commands predict over its own horizon vs the arc length realised
    rc::chain::ProgressMonitor cmd_hist;   // chain_monitor.h

    float ep_best = INFINITY, ep_t0 = 0.f;      // the episode's own progress clock, on its field AS BUILT at the episode's start
    CostToGo ep_field;                          // (the policy rebuilds its live field on every trap: values jump, not progress)
    Navigator nav;                              // the specialist
    UnseenContact unseen;
    Pose2 odom = start;
    // ★ MARKS LIVE IN THE ODOMETRY FRAME (Fable round 2 #2): odometry never jumps; the localiser does. A mark is stored where
    //   odometry put it and drawn into the room frame through the CURRENT room<-odom relation, so a relocalisation (a jump, a
    //   door-seam correction) carries every mark with the world instead of leaving it as a phantom 0.2-0.3 m off.
    Pose2 cur_bel = start, bel_prev = start, odom_prev = start;
    const auto rot = [](float a, const V2& d) { return V2(std::cos(a) * d.x() - std::sin(a) * d.y(), std::sin(a) * d.x() + std::cos(a) * d.y()); };
    const auto to_odom = [&](const V2& q) { return V2(odom.p + rot(odom.yaw - cur_bel.yaw, q - cur_bel.p)); };
    const auto to_room = [&](const V2& q) { return V2(cur_bel.p + rot(cur_bel.yaw - odom.yaw, q - odom.p)); };
    float stall_s0 = 0.f, stall_t0 = 0.f;       // arc length and time when the progress clock last restarted
    const auto window_poses = [&](float s0, float len) {
        std::vector<Pose2> w;
        for (float s = s0; s <= std::min(spline.length(), s0 + len) + 1e-4f; s += 0.10f) w.push_back(Pose2{spline.position_at(s), spline.heading_at(s)});
        return w; };
    // starts a policy episode toward the first route point past the failing stretch that stays clear for a body length
    // (the stretch is handed over BEFORE the chain drives it); none clear within the window -> the goal itself
    float t_now = 0.f;
    const auto start_episode = [&](const std::vector<Return>& cells, int first_bad, const char* why, const V2& bel_now) {
        ProfScope ps_{0};
        ++res.handoffs;
        Goal sub = goal;   // default: the whole goal (a doorway keeps its own goal and through test)
        episode_final = true;
        ep_s_end = spline.valid() ? std::min(spline.length(), s_prog + 4.0f) : 0.f;
        if (spline.valid() and goal.kind != Goal::Kind::Doorway)
        {
            const float s_bad = s_prog + 0.10f * float(std::max(0, first_bad));
            const float body_len = 2.f * B.circumscribed_radius();
            float clear_run = 0.f;
            for (float s = s_bad; s + 0.10f <= spline.length(); s += 0.10f)
            {
                const Pose2 q{spline.position_at(s), spline.heading_at(s)};
                const std::vector<Pose2> seg{q, Pose2{spline.position_at(s + 0.10f), spline.heading_at(s + 0.10f)}};
                if (log_p_no_contact_cells(B, seg, cells, kSigmaTrack, kSigmaTrackYaw, 0.01f) >= lp_ok) clear_run += 0.10f; else clear_run = 0.f;
                if (clear_run >= body_len) { sub = Goal{.kind = Goal::Kind::Place, .p = q.p, .yaw = {}, .tol_xy = 0.15f}; episode_final = false; ep_s_end = s; break; }
            }
        }
        // THE SPECIALIST MEASURES PROGRESS ALONG THE CHAIN'S ROUTE. Its field's free space is the union of the planner's
        // clearance discs along the route (distance to map + marks at each sample) and a disc at the robot reaching its
        // projection on the route; contact stays the policy's own LiDAR model. Its own free map could cut the robot off the
        // goal (clean38: policy_field_dump_1794309020589, field inf at a robot 0.77 m from any obstacle -> Release -> 8 m
        // reversal); the route corridor is connected by construction wherever the route runs.
        auto corr = std::make_shared<std::vector<char>>(std::size_t(mapfree.grid.w * mapfree.grid.h), 0);
        const auto& G = mapfree.grid;
        const auto mark_disc = [&](const V2& c, float r) {
            const int i0 = int(std::floor((c.x() - r - G.origin.x()) / G.cell)), i1 = int(std::ceil((c.x() + r - G.origin.x()) / G.cell));
            const int j0 = int(std::floor((c.y() - r - G.origin.y()) / G.cell)), j1 = int(std::ceil((c.y() + r - G.origin.y()) / G.cell));
            for (int j = std::max(0, j0); j <= std::min(G.h - 1, j1); ++j)
                for (int i = std::max(0, i0); i <= std::min(G.w - 1, i1); ++i)
                    if ((G.origin + V2(float(i) * G.cell, float(j) * G.cell) - c).norm() <= r) (*corr)[std::size_t(j * G.w + i)] = 1; };
        if (spline.valid())
        {
            // each disc's radius is the clearance the robot can SEE there now: the planner's (map + marks) and the live LiDAR's,
            // whichever is smaller, never below one cell (the chain of discs along the route stays connected: the field is
            // never cut by a return, so noise cannot recreate the clean38 disconnection). An unmapped barrier then narrows
            // the corridor to its gap: approaching the barrier elsewhere no longer reads as progress (edge suite, apt_gap_*).
            const auto live_d = [&cells](const V2& q) { float m = INFINITY; for (const auto& c : cells) m = std::min(m, (c.p - q).norm()); return m; };
            for (float s = 0.f; s <= spline.length(); s += G.cell)
            { const V2 q = spline.position_at(s); mark_disc(q, opt.live_corridor ? std::max(G.cell, std::min(gp.distance_at(q), live_d(q))) : gp.distance_at(q)); }
            const V2 pr = spline.position_at(s_prog);
            mark_disc(bel_now, std::max(gp.distance_at(bel_now), (bel_now - pr).norm() + G.cell));
        }
        else mark_disc(bel_now, gp.distance_at(bel_now));
        const auto free_fn = [corr, G](const V2& c) {
            const int i = int(std::lround((c.x() - G.origin.x()) / G.cell)), j = int(std::lround((c.y() - G.origin.y()) / G.cell));
            return i >= 0 and j >= 0 and i < G.w and j < G.h and (*corr)[std::size_t(j * G.w + i)] != 0; };
        nav = Navigator{};
        if (opt.corridor_field) nav.set_goal(sub, G, free_fn, [](const V2&) { return 0.f; }, goal_points(sub), B.circumscribed_radius());
        else nav.set_goal(sub, mapfree.grid, [&mapfree](const V2& c) { return mapfree(c); }, [](const V2&) { return 0.f; }, goal_points(sub),
                          B.circumscribed_radius());
        episode = true; ep_best = INFINITY; ep_t0 = t_now; ep_field = nav.field(); cmd_hist.clear();
        if (tr) std::printf("    HAND-OFF (%s) -> policy toward (%+.2f,%+.2f)\n", why, sub.p.x(), sub.p.y());
    };
    // THE PLANNER PLANS ON MAP + LIVE OBSTACLES (marking, as a costmap does). On a stall or a confirmed blockage the chain
    // first re-plans with the LiDAR cells the map does not explain (farther than one planner cell from every map obstacle and
    // every earlier mark). A different way round (a shut door, a person in a corridor) keeps the chain driving. A re-plan that
    // did not help — the next stall within a body radius of the last re-plan's — means the stretch is tight but passable:
    // that is the specialist's (owner 2026-11-10). Marks persist (MOCK: no clearing; live they clear when seen free).
    // marks: unmapped obstacles in the planner's world, with the belief that each is still there; failure marks (where the
    // specialist could not get through) lapse when an observed mark near them clears — see chain_monitor.h
    rc::chain::MarkSet ms;
    const auto square = [&](const V2& c) { const float h = 0.5f * 1.4143f * gp.params.cell_size_m; return std::vector<V2>{c + V2(-h, -h), c + V2(h, -h), c + V2(h, h), c + V2(-h, h)}; };
    const auto world_obst = [&] { auto o = map_obst; for (auto& q : ms.polygons(gp.params.cell_size_m)) o.push_back(std::move(q)); return o; };
    V2 last_replan_at(1e9f, 1e9f);
    // A return the MAP (+ earlier marks) does not explain: farther from it than one planner cell plus 3 sigma of the
    // localiser's uncertainty there (★FLAGGED MOCK STAND-IN as in the passability test: the true misplacement; live: the pose
    // covariance). Under a pose error every wall is seen a little off the map; this keeps those returns from becoming marks.
    // ★ "Explained by the map" as a CHANCE statement (Fable round 2 #1): a return at distance d from the map is explained when
    //   d <= cell + k sigma_e, sigma_e^2 = sigma_r^2 + sigma_loc^2 + sigma_map^2 and k the two-sided quantile of risk_eps (the
    //   window's own chance constraint: P(|z| > k) = risk_eps). sigma_map keeps the test from collapsing when a parked robot's
    //   pose covariance does.
    const float k_eps = rc::chain::two_sided_quantile(pp.risk_eps);
    const auto sigma_loc_at = [&](const V2& q) { return opt.loc_sigma_reported >= 0.f ? opt.loc_sigma_reported : (err.believed(Pose2{q, 0.f}).p - q).norm(); };
    const rc::chain::UnmappedTest unmapped_test{&gp, k_eps, sigma_r(), opt.map_sigma};
    const auto unmapped = [&](const V2& q) { return unmapped_test(q, sigma_loc_at(q)); };
    const auto replan_live = [&](const std::vector<Return>& cells, const V2& at) -> bool {
        ProfScope ps_{1};
        if ((at - last_replan_at).norm() <= B.circumscribed_radius()) return false;   // it did not help last time
        last_replan_at = at;
        const float h = 0.5f * 1.4143f * gp.params.cell_size_m;
        std::size_t added = 0;
        for (const auto& c : cells)
            if (unmapped(c.p))
            { if (ms.add(c.p, gp.params.cell_size_m)) ++added; }
        if (added > 0) { auto obst = world_obst(); { ProfScope ps_{8}; gp.set_world(roomp, obst); } }
        const bool ok = install(at, target);
        if (tr) std::printf("    RE-PLAN on map + live (%zu new marks, %zu total): %s\n", added, ms.marks.size(), ok ? "installed" : "no route");
        if (tr) { for (std::size_t k = ms.marks.size() - added; k < ms.marks.size(); ++k) { const V2 c = ms.marks[k].c; std::printf("      mark (%+.2f,%+.2f) planner d %.2f\n", c.x(), c.y(), gp.distance_at(c)); } }
        if (ok) ++res.replans;
        return ok; };

    // ★ BLOCKED BY SOMETHING THE MAP DOES NOT HAVE (owner 2026-11-11: "it should wait"): when no route exists on map + live
    //   marks but one does on the map alone, the way is blocked by an unmapped obstacle (a person in the only doorway, a box
    //   put down). The robot WAITS, stopped and observing, instead of declaring the goal unreachable; marks clear when the
    //   LiDAR sees through them and the chain re-plans. A way the MAP blocks is unreachable as before.
    //   ★FLAGGED: no give-up time here — how long a mission waits is the caller's (mission/task) decision.
    bool waiting = false;
    const auto wait_or_fail = [&](const V2& at, const char* why) -> bool {
        episode = false;
        if (opt.wait_blocked)
        {   // called when the live world (map + marks + failure marks) gives no usable way: if the MAP alone routes, what blocks
            // is not the map -> wait and watch (Fable round 2, loop C: a refused re-plan used to read as "unreachable")
            gp.set_world(roomp, map_obst);
            const bool on_map = gp.plan(at, target).has_value();
            auto obst = world_obst(); { ProfScope ps_{8}; gp.set_world(roomp, obst); }
            if (on_map) { waiting = true; ++res.waits; if (tr) std::printf("    WAIT: blocked by what the map does not have (%zu marks, %zu failure marks), watching\n", ms.marks.size(), ms.fail.size()); return true; }
        }
        res.unreachable = true; res.why = why; return false; };
    // THE SPECIALIST FAILED ON [s0, s1]: re-plan with the live marks; if that changes nothing, the failure itself is the
    // evidence — close the route's centre line over the failed section (failure marks) and re-plan; no way left -> wait (map
    // routes) or unreachable (map does not). true = carry on (new route or waiting), false = the run ends unreachable.
    const auto episode_failed = [&](const std::vector<Return>& cells, const V2& at, float s0, float s1, const char* why) -> bool {
        episode = false; cmd_hist.clear();
        if (replan_live(cells, at)) return true;
        if (spline.valid() and opt.fail_marks)
        {
            const float h = 0.5f * 1.4143f * gp.params.cell_size_m;
            for (float s = std::max(0.f, s0); s <= std::min(spline.length(), s1); s += gp.params.cell_size_m)
            { const V2 c = spline.position_at(s); ms.add_fail(c); }
            auto obst = world_obst(); { ProfScope ps_{8}; gp.set_world(roomp, obst); }
            if (tr) std::printf("    FAILURE MARKS on s %.2f..%.2f (%zu total): re-plan\n", s0, s1, ms.fail.size());
            if (install(at, target)) { ++res.replans; last_replan_at = at; return true; }
        }
        return wait_or_fail(at, why); };
    const auto stretch_blocked_at = [&](float s0, float s1, const std::vector<Return>& pc, const Pose2& bel) -> float {
        ProfScope ps_{2};
        // CONNECTIVITY over (route s x lateral d), chain_monitor.h. ★FLAGGED MOCK STAND-IN: sigma_loc = the true misplacement
        // of a world point (live: the pose covariance).
        const auto r = rc::chain::stretch_blocked_at(B, spline, gp, s0, s1, pc, bel, pp.risk_eps, k_eps, sigma_loc_at, opt.map_sigma);
        if (tr) std::printf("    PASS-EVAL s %.2f..%.2f (%d sections x %d offsets): %s (furthest section reached %d)\n", s0, s1, r.sections, r.offsets, r.blocked_at < 0.f ? "passable" : "BLOCKED", r.reach_k);
        return r.blocked_at; };
    Pose2 truth = start;
    constexpr float kTick = 0.05f, kSub = 0.005f;
    float t = 0.f, best_f = res.geodesic_m, last_progress_t = 0.f, ref_best = res.geodesic_m, last_sweep_t = 0.f;
    // ★ POSE AGE (Fable review §1; live: age >= 0.5 s on 30 % of rows, controller_policy_drive.cpp:446). The localiser's pose
    //   reaches the controller as samples every POSE_PERIOD s, each POSE_LAT s old, held until the next (sample-and-hold, NO
    //   odometry compensation: the worst case). Unset = the current pose every tick (the old mock).
    const float pose_lat = env_or("POSE_LAT", 0.f), pose_period = env_or("POSE_PERIOD", 0.f);
    std::deque<std::pair<float, Pose2>> truth_hist;
    Pose2 held = start; float next_pose_t = 0.f;
    // ODOM_COMP=1: the stale pose is brought to now by the odometry since its stamp (pose = sample (+) (odom_now (-) odom_then))
    const bool odom_comp = std::getenv("ODOM_COMP") != nullptr and std::string(std::getenv("ODOM_COMP")) == "1";
    std::deque<std::pair<float, Pose2>> odom_hist;
    Pose2 held_odom = start;
    float ep_next_pass_t = 0.f;
    size_t next_jump = 0;
    std::deque<std::vector<V2>> buffer;
    for (; t < time_limit_s; t += kTick)
    {
        t_now = t;
        if (not opt.movers.empty())
        {
            wnow = world;
            for (std::size_t k = 0; k < opt.movers.size(); ++k)
            {
                const V2 h = opt.movers[k].half, nxt = mover_at(k, mclock[k] + kTick);
                World one; one.boxes.push_back({nxt.x() - h.x(), nxt.y() - h.y(), nxt.x() + h.x(), nxt.y() + h.y(), false});
                float dmin = 1e9f;
                for (const V2& q : one.surface_points(0.02f)) dmin = std::min(dmin, rc::policy::hull_distance(B, truth, q));
                if (dmin >= 0.10f) mclock[k] += kTick;   // a person yields: it does not step into the robot
                const V2 c = mover_at(k, mclock[k]);
                wnow.boxes.push_back({c.x() - h.x(), c.y() - h.y(), c.x() + h.x(), c.y() + h.y(), false});
            }
            contact = sim2d::ContactTester(wnow, B);
            // what the LiDAR sees: a mover in a see-through gap is not in it (it still collides)
            wscan = world;
            for (std::size_t k = 0; k < opt.movers.size(); ++k)
            {
                const auto& M = opt.movers[k];
                if (M.gap_frac > 0.f)
                {
                    const float u = std::uniform_real_distribution<float>(0.f, 1.f)(rng);
                    if (mgap[k]) { if (u < kTick / M.gap_s) mgap[k] = false; }
                    else if (u < kTick / M.gap_s * M.gap_frac / (1.f - M.gap_frac)) mgap[k] = true;
                }
                if (not mgap[k]) wscan.boxes.push_back(wnow.boxes[world.boxes.size() + k]);
            }
        }
        err.step(kTick, rng);
        while (next_jump < opt.jumps.size() and opt.jumps[next_jump].t <= t)
        { err.jump(truth.p, opt.jumps[next_jump].de, opt.jumps[next_jump].deyaw); ++next_jump; }
        truth_hist.emplace_back(t, truth); odom_hist.emplace_back(t, odom);
        while (truth_hist.size() > 2 and truth_hist[1].first <= t - pose_lat - 1.f) { truth_hist.pop_front(); odom_hist.pop_front(); }
        if (pose_lat <= 0.f and pose_period <= 0.f) held = truth;
        else if (t + 1e-6f >= next_pose_t)
        {
            held = truth_hist.front().second; held_odom = odom_hist.front().second;
            for (std::size_t k = 0; k < truth_hist.size(); ++k)
                if (truth_hist[k].first <= t - pose_lat + 1e-6f) { held = truth_hist[k].second; held_odom = odom_hist[k].second; }
            next_pose_t = t + std::max(pose_period, kTick);
        }
        const Pose2 bel = [&] {
            Pose2 b = err.believed(held);
            if (not odom_comp) return b;
            const float c = std::cos(-held_odom.yaw), sn = std::sin(-held_odom.yaw);
            const V2 d = odom.p - held_odom.p, dl(c * d.x() - sn * d.y(), sn * d.x() + c * d.y());
            const float cb = std::cos(b.yaw), sb = std::sin(b.yaw);
            b.p += V2(cb * dl.x() - sb * dl.y(), sb * dl.x() + cb * dl.y());
            b.yaw = sim2d::wrap(b.yaw + sim2d::wrap(odom.yaw - held_odom.yaw));
            return b; }();
        if (goal.kind == Goal::Kind::Doorway and doorway_through(goal, bel.p)) { res.reached = true; res.why = "through"; break; }
        // ★ MARKS FOLLOW RELOCALISATIONS, NOT ODOMETRY DRIFT. Marks live in the room frame. Each tick the room<-odom relation
        //   moves by a little odometry noise (slip) or, when the localiser corrects, by a step. A mark carried through the
        //   change (room_prev -> odom -> room_now) moves by `disp`; odometry's own noise model allows sigma = slip x (distance
        //   driven + angle turned x lever arm). A move beyond k_eps sigma is a relocalisation: every mark is carried with it.
        //   (First version anchored marks in odometry outright: 5 % slip dragged them 3-5 cm per few seconds off their
        //   obstacles — seen through, re-marked, re-planned, 20 s lost.)
        if (opt.anchor_marks and not ms.empty())
        {
            const auto transport = [&](const V2& q) { const V2 qo = odom_prev.p + rot(odom_prev.yaw - bel_prev.yaw, q - bel_prev.p); return V2(bel.p + rot(bel.yaw - odom.yaw, qo - odom.p)); };
            const float dp = (odom.p - odom_prev.p).norm(), dth = std::abs(sim2d::wrap(odom.yaw - odom_prev.yaw));
            const float worst = ms.carry_if_relocalised(transport, [&](const V2& c) { return kSlipSigma * (dp + dth * (c - bel.p).norm()) + 1e-4f; },
                                                        k_eps, gp.params.cell_size_m);
            if (worst >= 0.f)
            {
                auto obst = world_obst(); { ProfScope ps_{8}; gp.set_world(roomp, obst); }
                if (tr) std::printf("    MARKS carried with a relocalisation (%.2f m)\n", worst);
            }
        }
        bel_prev = bel; odom_prev = odom; cur_bel = bel;
        // A re-measured goal (opt.goal_jitter_m) does NOT re-plan here: the design (controller-two-regimes §3.1) moves only the
        // route's far end, and re-installing every tick resets the tracker every tick (measured in the mock: arc-length progress
        // pinned at 0, the robot pivoting for 60 s), which no live session does. The doorway oracle judges against the true goal.
        // ---- both LiDARs from the TRUE pose, expressed in the room frame on the believed pose, buffered (as run_scenario)
        const bool dropout = std::ranges::any_of(opt.lidar_dropout, [&](const auto& w) { return t >= w.first and t < w.second; });
        std::vector<V2> sweep;
        const V2 bf(std::cos(bel.yaw), std::sin(bel.yaw)), br(std::sin(bel.yaw), -std::cos(bel.yaw));
        for (const auto& m : {kHelios, kBpearl})
            for (const V2& q : sim2d::scan_mount(wscan, truth, m, rng, sigma_r()))
                sweep.push_back(bel.p + q.x() * br + q.y() * bf);
        if (not dropout) { buffer.push_back(std::move(sweep)); last_sweep_t = t; }
        while (int(buffer.size()) > kBufferCycles) buffer.pop_front();
        // ★ LIVE MARKS CLEAR WHEN SEEN FREE (edge suite: a person in a doorway was marked and the robot detoured 6 m after it
        //   had left). A mark is removed when the current sweep sees THROUGH it: a return on its bearing beyond it. A mark the
        //   sensor cannot see (occluded, behind the robot's mast, out of range) holds. Once marks clear, the chain re-plans.
        if (not ms.marks.empty() and not dropout and opt.clear_marks)
        {
            ProfScope ps_{6};
            std::vector<std::pair<float, float>> br;   // (bearing, range) of the latest sweep
            for (const V2& q : buffer.back()) { const V2 r = q - bel.p; br.push_back({std::atan2(r.y(), r.x()), r.norm()}); }
            const auto look = ms.observe(bel.p, br, gp.params.cell_size_m, B.circumscribed_radius(), pp.risk_eps,
                                         2.f * B.circumscribed_radius(), opt.mark_belief);
            if (tr and look.lapsed > 0) std::printf("    FAILURE MARKS lapsed: %zu (the obstacle they were judged against is gone)\n", look.lapsed);
            if (look.cleared > 0)
            {
                auto obst = world_obst(); { ProfScope ps_{8}; gp.set_world(roomp, obst); }
                last_replan_at = V2(1e9f, 1e9f);
                if (tr) std::printf("    MARKS seen free: %zu cleared, %zu left\n", look.cleared, ms.marks.size());
                // ★ re-plan only for a SHORTER route (rc::chain::shorter_route) unless waiting — the live rule, same code
                const bool worth = waiting or [&] {
                    if (not opt.replan_on_clear) return false;
                    const auto cand = gp.plan(bel.p, target);
                    return cand.has_value() and rc::chain::shorter_route(*cand, spline.length() - s_prog, gp.params.cell_size_m); }();
                if (not episode and worth and install(bel.p, target)) { cmd_hist.clear(); stall_s0 = 0.f; stall_t0 = t; if (waiting) { ++res.wait_exits; if (tr) std::printf("    WAIT over: the way is clear\n"); } waiting = false; }
            }
        }
        // ---- the controller's frames: theta = yaw - pi/2, robot frame +x right / +y forward
        Eigen::Affine2f pose = Eigen::Affine2f::Identity();
        pose.translation() = bel.p;
        pose.linear() = Eigen::Rotation2Df(bel.yaw - 0.5f * kPi).toRotationMatrix();
        const Eigen::Matrix2f R = pose.linear(), Rt = R.transpose();
        std::vector<Eigen::Vector3f> cloud;
        for (const auto& sw : buffer)
            for (const V2& q : sw) { const V2 r = Rt * (q - bel.p); cloud.emplace_back(r.x(), r.y(), 0.5f); }
        const auto t_step = std::chrono::steady_clock::now();
        if (opt.hybrid and not waiting)
        {
            std::vector<Return> ret;
            for (const auto& sw : buffer) for (const V2& q : sw) ret.push_back({q, sigma_r(), 1.f});
            const auto cells = denoise_returns(ret);
            if (not episode and spline.valid())
            {
                s_prog = spline.project(bel.p, s_prog, 3.0f);
                // (a) the window ahead, judged on LIVE LiDAR surfaces only (never the injected model: it moves with the pose error)
                // the chain's NEXT motion starts with whatever turn in place aligns the body with the route (PLAIN pivots at a
                // cusp or a large heading error): that sweep is part of the window, at the robot's own position
                ProfScope ps_win_{7};
                // (the window, its pivot onto the route and the per-segment chance constraint: chain_monitor.h window_test)
                const auto win = rc::chain::window_test(B, spline, s_prog, bel, cells, kSigmaTrack, kSigmaTrackYaw, pp.risk_eps);
                const std::vector<Pose2>& W = win.poses;
                int first_bad = win.first_bad; const float lp_w = win.lp_worst;
                if (tr and t < 0.3f) std::printf("    window: %zu poses (pivot %+.0f deg first), worst lp %.4f, bel yaw %+.0f route heading %+.0f\n", W.size(),
                                                 sim2d::wrap(spline.heading_at(s_prog) - bel.yaw) / kDeg, lp_w, bel.yaw / kDeg, spline.heading_at(s_prog) / kDeg);
                // HAND OVER WHEN THE STRETCH IS REACHED, not when it is seen: the chain drives up to it. "Reached" = within the
                // tracker's own preview (v T_lag + W/2, the band's lead) plus the body's circumscribed radius; a pivot is now.
                const float s_bad = rc::chain::window_s_bad(win, s_prog);
                const float reach_m = std::abs(base.v) * tc.params.plain_T_lag + 0.5f * tc.params.plain_W + B.circumscribed_radius();
                if (first_bad >= 0 and s_bad - s_prog > reach_m) first_bad = -1;   // seen, not yet reached: the chain keeps driving
                if (first_bad >= 0)
                {
                    if (tr)
                    {
                        float dmin = 1e9f;
                        for (const auto& c : cells) dmin = std::min(dmin, hull_distance(B, W[std::size_t(first_bad)], c.p));
                        std::printf("    window worst lp %.4f (ok >= %.4f), %zu poses; first bad #%d at (%+.2f,%+.2f,%+.0f deg), hull distance to the nearest cell %.3f m\n",
                                    lp_w, lp_ok, W.size(), first_bad, W[std::size_t(first_bad)].p.x(), W[std::size_t(first_bad)].p.y(), W[std::size_t(first_bad)].yaw / kDeg, dmin);
                    }
                    // PLAN FIRST when the stretch ahead is IMPASSABLE (no connected way through it on the live LiDAR): the planner
                    // re-plans around it at once (map + live marks); only a tight-but-passable stretch goes to the policy
                    // (edge suite: unmapped clutter cost 55-75 s through hand-off -> passability -> re-plan)
                    bool replanned = false;
                    // ...or when the failing stretch is failed by something the MAP does not have (an unmapped panel in open
                    // floor is a planning problem: the policy's field does not know it either and pushes into it — measured
                    // apt_slalom seed 2: 45 s). A mapped-but-tight stretch (a doorway's jambs) stays the policy's.
                    bool unmapped_cause = false;
                    if (opt.plan_first)
                    {
                        const Pose2& wb = W[std::size_t(first_bad)];
                        for (const auto& c : cells)
                            // the returns that fail the window: within 3 sigma_track (+ one cell) of the body's hull at the failing pose
                            if (hull_distance(B, wb, c.p) <= 3.f * kSigmaTrack + gp.params.cell_size_m and unmapped(c.p)) { unmapped_cause = true; break; }
                    }
                    if (opt.plan_first and (unmapped_cause or stretch_blocked_at(s_prog, std::min(spline.length(), s_prog + 4.0f), cells, bel) >= 0.f))
                    {
                        if (tr) std::printf("    PLAN FIRST: the window ahead is %s\n", unmapped_cause ? "failed by an unmapped obstacle" : "impassable");
                        replanned = replan_live(cells, bel.p);
                        if (replanned) { cmd_hist.clear(); stall_s0 = 0.f; stall_t0 = t; }
                    }
                    if (not replanned) { ++res.handoff_window; start_episode(cells, first_bad, "window", bel.p); }
                }
                // (b) the chain's own progress prediction failed: no arc-length progress over one band window's travel time
                else if (opt.monitors)
                {
                    // predicted: the arc length the commands sent one tracker lag ago should have produced over the horizon H
                    // (the tracker's own preview time: T_lag + (W/2)/v_max); realised: the projection's advance over H.
                    // Fails when the prediction exceeds the projection's resolution and the realisation does not:
                    //   res = sqrt(sigma_pose^2 + h^2/12 + (v * pose_age)^2)   (h = the projection's sample spacing)
                    const float age = pose_lat + 0.5f * pose_period;
                    rc::chain::ProgressMonitor::In mi;
                    mi.t = t; mi.s_prog = s_prog; mi.dt = kTick; mi.T_lag = tc.params.plain_T_lag; mi.W = tc.params.plain_W; mi.v_max = base.lim.v_max;
                    mi.v = base.v; mi.w = base.w; mi.age = age; mi.sigma_pose = err.on ? 0.f : 0.01f;   // the OU error is common-mode on a route in the same frame
                    mi.circumscribed_r = B.circumscribed_radius();
                    mi.goal_applies = goal.kind != Goal::Kind::Doorway; mi.goal_dist = (bel.p - goal.p).norm(); mi.goal_tol = goal.tol_xy;
                    const auto mv = cmd_hist.judge(mi);
                    const float pred = mv.pred, resn = mv.resn, H = mv.H, s_then = s_prog - mv.realised;
                    if (mv.fail)
                    {
                        if (tr) std::printf("    CHAIN MONITOR: predicted %.3f m, realised %.3f m, resolution %.3f m over %.2f s\n", pred, s_prog - s_then, resn, H);
                        cmd_hist.clear();
                        ++res.handoff_stall; start_episode(cells, 0, "stall", bel.p);
                    }
                }
                else if (s_prog > stall_s0 + 0.10f) { stall_s0 = s_prog; stall_t0 = t; }
                else if (t - stall_t0 > 4.0f / base.lim.v_max)
                {
                    stall_s0 = 0.f; stall_t0 = t;
                    ++res.handoff_stall; start_episode(cells, 0, "stall", bel.p);
                }
            }
            if (episode)
            {
                std::vector<Return> pin = ret;
                for (const auto& h : unseen.hypotheses(bel, odom)) pin.push_back(h);
                PolicyInput in;
                in.pose = bel; in.v_now = base.v; in.w_now = base.w; in.sigma_rel_xy = 0.01f; in.sigma_rel_yaw = 0.01f;
                in.camera_half_fov_rad = 0.96f; in.affordance_facing = true;
                in.returns_age_s = float(kBufferCycles - 1) * kTick + (t - last_sweep_t);
                in.returns = pin; in.body = &B; in.base = base.lim; in.latency_s = base.latency_s; in.now_s = t;
                const Navigator::Step st = [&] { ProfScope ps_{3}; return nav.step(in, pp); }();
                // THE SPECIALIST CANNOT PASS EITHER: no progress on its own field over one band window's travel time (the
                // chain's stall clock) -> the world differs from the map here (a shut door, a person): the planner re-plans
                // on map + live and the chain drives the new way. Marking only now: with a pose error every wall is seen a
                // little off the map, and marking at the first stall drew ghost walls across the doors (measured: pose_jump
                // 3/6, 5.9 m retreats). No new way -> the goal is unreachable from here.
                if (opt.monitors and spline.valid() and t + 1e-6f >= ep_next_pass_t)
                {
                    // PASSABILITY of the stretch from the contact model (Fable review §3): a cross-section of the route corridor
                    // is passable when SOME lateral offset (within the planner's clearance there) and heading gives
                    // P(no contact) >= 1 - risk_eps for the body moving one step along the route. Optimistic by design (offsets
                    // chosen per section: the policy may weave), so "impassable" means no way through the section at all.
                    // Evaluated twice a second (the LiDAR's own refresh of what it sees ahead is not faster than that).
                    ep_next_pass_t = t + 0.5f;
                    s_prog = spline.project(bel.p, s_prog, 3.0f);   // where the policy has brought the robot along the route
                    const auto pc = denoise_returns(ret);
                    const float blocked_at = stretch_blocked_at(s_prog, ep_s_end, pc, bel);
                    if (blocked_at >= 0.f)
                    {
                        if (tr) std::printf("    PASSABILITY: no way through the section at s %.2f (robot at %.2f): re-plan\n", blocked_at, s_prog);
                        episode = false; cmd_hist.clear();
                        if (not episode_failed(pc, bel.p, blocked_at - 0.10f, blocked_at + 0.10f, "hybrid: the stretch is impassable and no other way")) break;
                        continue;
                    }
                }
                if (not opt.monitors)
                {
                    const float fe = ep_field.at(bel.p);
                    if (tr and std::fmod(t, 1.0f) < kTick) std::printf("    EPISODE clock: f %.3f best %.3f since %.2f s\n", fe, ep_best, t - ep_t0);
                    if (fe < ep_best - 0.10f) { ep_best = fe; ep_t0 = t; }
                    else if (t - ep_t0 > 4.0f / base.lim.v_max)
                    {
                        episode = false; stall_s0 = 0.f; stall_t0 = t;
                        if (not episode_failed(denoise_returns(ret), bel.p, s_prog, s_prog + 2.f * B.circumscribed_radius(), "hybrid: the specialist could not pass and no other way")) break;
                        continue;
                    }
                }
                const Decision& d = st.d;
                if (d.mode == Decision::Mode::Arrived and episode_final) { res.reached = true; res.why = "arrived (specialist)"; break; }
                // an intermediate episode is OVER when the robot has got PAST its stretch along the route (projection beyond the
                // subgoal's arc length) — the subgoal point itself may not hold the body (edge suite apt_slalom seed 0: a subgoal
                // 0.15 m from a panel, 85 s of circling 0.2 m from it)
                bool past = false;
                if (not episode_final and spline.valid() and opt.pass_ends_episode)
                {
                    s_prog = spline.project(bel.p, s_prog, 3.0f);
                    if (s_prog >= ep_s_end)
                    {
                        // ...and the chain can take over HERE: the body turning onto the route and the route ahead within the
                        // tracker's reach pass the chain's own window test (the hand-back must not drop the chain beside a
                        // panel: measured 1 mm of clearance and a contact without this)
                        const float h_route = spline.heading_at(s_prog), dh = sim2d::wrap(h_route - bel.yaw);
                        std::vector<Pose2> Wb;
                        const int n = std::max(1, int(std::ceil(std::abs(dh) / 0.10f)));
                        for (int k = 0; k <= n; ++k) Wb.push_back(Pose2{bel.p, bel.yaw + dh * float(k) / float(n)});
                        const float reach_m = std::abs(base.v) * tc.params.plain_T_lag + 0.5f * tc.params.plain_W + B.circumscribed_radius();
                        for (const auto& q : window_poses(s_prog, reach_m + 2.f * B.circumscribed_radius())) Wb.push_back(q);
                        const auto cw = denoise_returns(ret);
                        past = true;
                        for (std::size_t i = 0; i + 1 < Wb.size() and past; ++i)
                        { const std::vector<Pose2> seg{Wb[i], Wb[i + 1]}; if (log_p_no_contact_cells(B, seg, cw, kSigmaTrack, kSigmaTrackYaw, 0.01f) < lp_ok) past = false; }
                        // past the stretch but the chain cannot take over here: the EPISODE EXTENDS — the policy keeps the wheel
                        // toward the next route point past here that stays clear for a body length (as start_episode picks it)
                        if (not past)
                        {
                            --res.handoffs;   // one continued episode, not a new hand-off
                            start_episode(cw, 0, "extend", bel.p);
                            if (tr) std::printf("    EPISODE EXTENDED past s %.2f\n", s_prog);
                            continue;
                        }
                    }
                }
                if (d.mode == Decision::Mode::Arrived or past)
                {   // the stretch is behind: the chain re-plans from here
                    episode = false;
                    if (not install(bel.p, target)) { res.unreachable = true; res.why = "hybrid: no re-plan after the episode"; break; }
                    stall_s0 = 0.f; stall_t0 = t; cmd_hist.clear();
                }
                else if ((st.unreachable or st.hand_back) and opt.monitors)
                {   // the policy cannot reach its stretch: that is evidence for the planner, not the end of the goal (Fable §1.4)
                    if (tr) std::printf("    POLICY HAND-BACK (%s): re-plan on map + live\n", st.unreachable ? "unreachable" : "hand_back");
                    episode = false; cmd_hist.clear();
                    s_prog = spline.valid() ? spline.project(bel.p, s_prog, 3.0f) : 0.f;
                    if (not episode_failed(denoise_returns(ret), bel.p, s_prog, s_prog + 2.f * B.circumscribed_radius(), "hybrid: the policy handed back and no other way")) break;
                    continue;
                }
                else if (st.unreachable or st.hand_back) { res.unreachable = true; res.why = "hybrid: the policy found the stretch unreachable"; break; }
                const float v_cmd = (d.mode == Decision::Mode::Drive or d.mode == Decision::Mode::Release) ? d.v : 0.f;
                const float w_cmd = (d.mode == Decision::Mode::Drive or d.mode == Decision::Mode::Release) ? d.w : 0.f;
                if (d.mode == Decision::Mode::Release and d.v < 0.f) ++res.backups;
                if (tr) std::printf("    t=%6.2f POLICY %s v=%+.2f w=%+.2f bel(%+.2f,%+.2f,%+.0f) true(%+.2f,%+.2f) L %.2f/%.2f | %s\n", t, mode_name(d.mode), d.v, d.w, bel.p.x(), bel.p.y(), bel.yaw / kDeg, truth.p.x(), truth.p.y(), st.unreach_L, st.unreach_bar, d.why.c_str());
                // the base, as below (with the odometry the policy's unseen-contact model needs)
                base.send(t, v_cmd, w_cmd);
                const Pose2 before = truth, odom_before = odom;
                const float v_meas = base.v, w_meas = base.w;
                const float sv = 1.f + slip(rng), sw2 = 1.f + slip(rng);
                bool touched = false;
                std::vector<UnseenContact::CmdSeg> in_force;
                for (float s = 0.f; s < kTick - 1e-6f; s += kSub)
                {
                    base.step_velocity(t + s, kSub);
                    if (not in_force.empty() and in_force.back().v == base.cmd_v and in_force.back().w == base.cmd_w) in_force.back().dt += kSub;
                    else in_force.push_back({kSub, base.cmd_v, base.cmd_w});
                    const Pose2 nxt = sim2d::integrate(truth, base.v * sv, base.w * sw2, kSub);
                    if (contact.touches(nxt)) { touched = true; base.v = base.w = 0.f; continue; }
                    odom = sim2d::integrate(odom, base.v, base.w, kSub);
                    if (base.v * sv < 0.f and contact.clearance(truth) >= 0.5f) res.open_reverse_m += (nxt.p - truth.p).norm();
                    truth = nxt;
                }
                if (touched) ++res.contacts;
                const V2 fwd(std::cos(odom_before.yaw), std::sin(odom_before.yaw));
                unseen.update(std::span<const UnseenContact::CmdSeg>(in_force), v_meas, w_meas, (odom.p - odom_before.p).dot(fwd),
                              sim2d::wrap(odom.yaw - odom_before.yaw), odom, base.lim, B, 0.005f);
                res.path_m += (truth.p - before.p).norm();
                res.min_clear = std::min(res.min_clear, contact.clearance(truth));
                if (const float fr = ref_field.at(truth.p); std::isfinite(fr)) { ref_best = std::min(ref_best, fr); res.max_retreat_m = std::max(res.max_retreat_m, fr - ref_best); }
                const float fn = ref_field.at(err.believed(truth).p);
                if (fn < best_f - 1e-3f) { best_f = fn; last_progress_t = t + kTick; }
                res.max_no_progress_s = std::max(res.max_no_progress_s, t + kTick - last_progress_t);
                continue;   // the chain does not run while the policy drives
            }
        }
        const auto out = [&] { ProfScope ps_{4}; return tc.compute(pose, cloud); }();
        // ---- the BAND (step_route_band): the window ahead, deformed against the ESDF compute() just built
        if (spline.valid())
        {
            s_prog = spline.project(bel.p, s_prog, 3.0f);
            const auto& P = spline.control_points();
            const std::size_t M = P.size();
            const float len = std::max(0.01f, spline.length()), frac = std::clamp(s_prog / len, 0.f, 1.f);
            const float h = std::max(0.05f, spline.control_spacing());
            const auto i_robot = static_cast<std::size_t>(frac * static_cast<float>(M));
            const float preview_m = std::abs(base.v) * tc.params.plain_T_lag + 0.5f * tc.params.plain_W;
            const auto lead = static_cast<std::size_t>(std::floor(preview_m / h)) + 1;    // BandProximalSigmaM > 0: the preview lead
            const auto span = static_cast<std::size_t>(std::ceil(std::max(h, 4.0f) / h)); // BandWindow 4.0
            const std::size_t fb = std::min(i_robot + lead, M), fa = std::min(fb + span, M);
            if (fa > fb + 1)
            {
                rc::RouteOptimizerConfig cfg = opt_cfg();
                cfg.distance = [&tc, Rt, p0 = bel.p](const V2& p) { const V2 r = Rt * (p - p0); return tc.clearance_at(r.x(), r.y()); };
                cfg.distance_gradient = [&tc, R, Rt, p0 = bel.p](const V2& p) { const V2 r = Rt * (p - p0); return V2(R * tc.clearance_gradient_at(r.x(), r.y())); };
                cfg.freeze_before = fb; cfg.freeze_after = fa;
                cfg.iterations = 4; cfg.w_clear = 0.5f; cfg.verbose = false;   // BandIterations 4, BandClearanceWeight 0.5
                for (std::size_t i = std::max<std::size_t>(1, fb); i + 1 < M and i < fa; ++i)
                { cfg.anchors.push_back((P[i - 1] + 4.f * P[i] + P[i + 1]) / 6.f); cfg.anchor_s.push_back(static_cast<float>(i) * h); }
                cfg.sigma_a = 0.03f;                                             // BandProximalSigmaM
                const auto rep = [&] { ProfScope ps_{5}; return spline.deform(cfg); }();
                if (rep.ran and not rep.rejected) tc.update_path_geometry(spline.samples());
            }
        }
        res.step_ms.push_back(float(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_step).count()));
        res.returns_max = std::max(res.returns_max, cloud.size());
        if (spline.valid())   // sigma_track: the TRUE centre's distance from the curve (the pose error is a separate term)
        {
            const float sx = spline.project(truth.p, s_prog, 3.0f);
            const float e = (truth.p - spline.position_at(sx)).norm();
            res.xtrack_sum2 += double(e) * e; ++res.xtrack_n;
        }
        const float clear = contact.clearance(truth);
        res.min_clear = std::min(res.min_clear, clear);
        if (tr)
            std::printf("    t=%6.2f true(%+.3f,%+.3f,%+6.1f) adv=%+.2f rot=%+.2f blocked=%d ahead=%d reached=%d dgoal=%.2f clear=%+.3f s=%.2f/%.2f\n",
                        t, truth.p.x(), truth.p.y(), truth.yaw / kDeg, out.adv, out.rot, int(out.path_blocked), int(out.blockage_detected_ahead),
                        int(out.goal_reached), out.dist_to_goal, clear, s_prog, spline.length());
        if (out.goal_reached and goal.kind != Goal::Kind::Doorway)
        {
            // judged at the GOAL's own tolerance (Fable round 2: a 0.5 m floor here inflated every reach rate)
            if ((bel.p - goal.p).norm() <= goal.tol_xy) { res.reached = true; res.why = "arrived"; break; }
            if (opt.hybrid and not episode and not waiting)
            {   // the tracker stopped short of the tolerance: the specialist finishes the approach (the whole goal)
                std::vector<Return> ret;
                for (const auto& sw : buffer) for (const V2& q : sw) ret.push_back({q, sigma_r(), 1.f});
                const auto cells = denoise_returns(ret);
                const bool keep = spline.valid();
                if (keep) s_prog = spline.length();   // the stretch left is the goal itself
                ++res.handoff_stall; start_episode(cells, 0, "finish", bel.p);
                episode_final = true;
                continue;
            }
            if (not opt.hybrid) { res.stuck = true; res.why = "chain: the tracker reported arrival away from the goal"; break; }
        }
        // HYBRID: the chain's own confirmed blockage IS "the chain cannot drive this stretch" — the third hand-off trigger; the
        // specialist judges the stretch on LiDAR. (An invented permanent obstacle walled off a real 0.9 m door seen 5 cm off
        // under pose drift and sent the robot on a 25 m detour.)
        if (opt.hybrid and out.path_blocked and not episode and not waiting)
        {
            std::vector<Return> ret;
            for (const auto& sw : buffer) for (const V2& q : sw) ret.push_back({q, sigma_r(), 1.f});
            const auto cells = denoise_returns(ret);
            bool replanned = false;
            if (opt.plan_first and spline.valid() and stretch_blocked_at(s_prog, std::min(spline.length(), s_prog + 4.0f), cells, bel) >= 0.f)
            {
                if (tr) std::printf("    PLAN FIRST: blocked and impassable\n");
                replanned = replan_live(cells, bel.p);
                if (replanned) { cmd_hist.clear(); stall_s0 = 0.f; stall_t0 = t; }
            }
            if (not replanned) { ++res.handoff_stall; start_episode(cells, 0, "blocked", bel.p); }
        }
        else if (out.path_blocked)   // MOCK (chain only): re-plan around the blocked disc (the session's local repair is not reproduced)
        {
            ++res.traps;
            if (tr) std::printf("    PATH BLOCKED at (%+.2f,%+.2f) r %.2f -> re-plan around it\n", out.blockage_center_room.x(), out.blockage_center_room.y(), out.blockage_radius);
            std::vector<V2> disc;
            for (int k = 0; k < 12; ++k)
            {
                const float a = 2.f * kPi * float(k) / 12.f;
                disc.push_back(out.blockage_center_room + std::max(0.1f, out.blockage_radius) * V2(std::cos(a), std::sin(a)));
            }
            map_obst.push_back(disc);
            gp.set_world(roomp, map_obst);
            if (not install(bel.p, target)) { res.unreachable = true; res.why = "chain: blocked and no re-plan (" + gp.last_failure() + ")"; break; }
        }
        // ---- the base (identical to run_scenario): latency, ramps, deadband, slip; contact every <= 5 mm
        base.send(t, waiting ? 0.f : out.adv, waiting ? 0.f : -out.rot);
        if (opt.hybrid)
        {
            // the command's component ALONG the route (a turn-around arc commands speed but predicts no arc length)
            const float along = spline.valid() ? out.adv * std::cos(sim2d::wrap(spline.heading_at(s_prog) - bel.yaw)) : 0.f;
            cmd_hist.record(t, along, s_prog, out.rot);
        }
        const Pose2 before = truth;
        const float sv = 1.f + slip(rng), sw = 1.f + slip(rng);
        bool touched = false;
        for (float s = 0.f; s < kTick - 1e-6f; s += kSub)
        {
            base.step_velocity(t + s, kSub);
            const Pose2 nxt = sim2d::integrate(truth, base.v * sv, base.w * sw, kSub);
            if (contact.touches(nxt)) { touched = true; base.v = base.w = 0.f; continue; }
            if (base.v * sv < 0.f and contact.clearance(truth) >= 0.5f) res.open_reverse_m += (nxt.p - truth.p).norm();
            if (opt.hybrid) odom = sim2d::integrate(odom, base.v, base.w, kSub);
            truth = nxt;
        }
        if (touched) ++res.contacts;
        res.path_m += (truth.p - before.p).norm();
        if (const float fr = ref_field.at(truth.p); std::isfinite(fr)) { ref_best = std::min(ref_best, fr); res.max_retreat_m = std::max(res.max_retreat_m, fr - ref_best); }
        const float fn = ref_field.at(err.believed(truth).p);
        if (fn < best_f - 1e-3f) { best_f = fn; last_progress_t = t + kTick; }
        res.max_no_progress_s = std::max(res.max_no_progress_s, t + kTick - last_progress_t);
    }
    res.time_s = t;
    res.end = truth;
    if (res.why.empty()) res.why = "time limit";
    return res;
}

static ScenarioResult run_scenario(const World& world, Pose2 start, Goal goal, float time_limit_s, unsigned seed, const Options& opt)
{
    if (opt.chain) return run_chain(world, start, goal, time_limit_s, seed, opt);
    ScenarioResult res;
    std::mt19937 rng(seed * 7919u + 13u);
    const bool tr = trace_on(opt.trace_name, seed);
    const auto& B = body();
    sim2d::ContactTester contact(world, B);
    sim2d::Base base; base.lim = limits(); base.latency_s = 0.06f; plant_from_env(base);
    if (opt.d_lin > 0.f) base.lim.d_lin = opt.d_lin;
    base.rng = &rng; base.jitter_s = kLatencyJitterS; base.slip_sigma = kSlipSigma;
    base.deadband_scale = std::uniform_real_distribution<float>(1.f - kDeadbandSpread, 1.f + kDeadbandSpread)(rng);
    base.v = base.cmd_v = opt.v0; base.w = base.cmd_w = opt.w0; base.rv = base.v / base.gain; base.rw = base.w / base.gain;
    std::normal_distribution<float> slip(0.f, kSlipSigma);
    sim2d::PoseError err; err.on = opt.ou; err.pivot = start.p;
    if (opt.ou_sigma_xy >= 0.f) err.sigma_xy = opt.ou_sigma_xy;
    if (opt.ou_sigma_yaw >= 0.f) err.sigma_yaw = opt.ou_sigma_yaw;
    UnseenContact unseen;
    std::vector<V2> hard_cells;
    for (std::size_t i = 0; i < opt.res_cells.size() and i < opt.res_risk.size(); ++i)
        if (not std::isfinite(opt.res_risk[i])) hard_cells.push_back(opt.res_cells[i]);
    const MapFree mapfree(world, opt.grid, hard_cells);
    Navigator nav;
    // ★ F4 — as the controller does: a Doorway's measured jambs enter the field's per-metre density over its corridor
    //   (residual_risk_weight 1, the config default). The gap's sigmas as the [gap] detector states them for a LiDAR range
    //   sigma sigma_r (doorway_gap.h doorway_gap_meas_cov: each axis sigma_r / sqrt 2), the contact model's shared offset
    //   from the relative sigma the policy is given below (0.01) and PolicyInput's body-geometry sigma.
    std::optional<DoorwayRisk> jambs;   // the goal's opening is the anchor; the harness has no separate gap measurement
    if (goal.kind == Goal::Kind::Doorway and not opt.no_jambs)
    {
        rc::RobotFootprint bare = B; bare.set_safety_margin(0.f);
        const float sg = sigma_r() * 0.70710678f;
        jambs = DoorwayRisk{};
        jambs->anchor = doorway_jambs(RoomGap{goal.p, goal.normal.normalized(), goal.width}, sg, sg,
                              std::hypot(0.01f, PolicyInput{}.body_geometry_sigma_m), sigma_r(), bare, 1.0f,
                              kThroughS + 2.f * goal.tol_xy + 0.05f);
    }
    const auto install_goal = [&](const Goal& gl) {
    nav.set_goal(gl, mapfree.grid, [&mapfree](const V2& c) { return mapfree(c); },
                 [jambs, &opt, r_ins = B.inscribed_radius()](const V2& c)
                 {
                     // residual's soft cells under the inscribed disc (the planner's price), then the door's factor
                     float soft = 0.f;
                     for (std::size_t i = 0; i < opt.res_cells.size() and i < opt.res_risk.size(); ++i)
                         if (std::isfinite(opt.res_risk[i]) and (opt.res_cells[i] - c).norm() - 0.025f <= r_ins) soft += opt.res_risk[i];
                     return (jambs ? doorway_risk_density(*jambs, c) : 0.f) + soft + door_leaf_density(opt.leaf, c);
                 }, goal_points(gl),
                 B.circumscribed_radius());
    };
    install_goal(goal);
    // ★ Live fix 2 ruling 2 — as the controller does: a Doorway's measured opening (at least the body's width) vetoes traps
    if (goal.kind == Goal::Kind::Doorway)
    {
        const RoomGap gap{goal.p, goal.normal.normalized(), goal.width};
        rc::RobotFootprint bare = B; bare.set_safety_margin(0.f);
        const float body_w = bare.support_radius(0.f, V2(1.f, 0.f)) + bare.support_radius(0.f, V2(-1.f, 0.f));
        const float r_ins = bare.inscribed_radius(), ahead = kThroughS + 2.f * goal.tol_xy + 0.05f;
        nav.set_trap_veto([gap, r_ins, body_w, ahead](const V2& c, float r) { return disc_meets_doorway_corridor(gap, c, r, 1.0f, ahead, r_ins, body_w); });
    }
    res.build_ms += nav.last_build_ms(); ++res.builds;
    PolicyParams pp;
    Pose2 truth = start;
    // ★ ODOMETRY (review fix round 2): the wheels' velocities integrated (the encoders), i.e. the motion WITHOUT the
    //   plant's slip — it drifts from the truth by the slip and never jumps. UnseenContact lives in this frame.
    Pose2 odom = start;
    res.geodesic_m = nav.field().at(start.p);
    CostToGo ref_field = nav.field();                // the still goal's field: the retreat oracle's yardstick
    // TODAY'S CONTROLLER with a target it cannot stand on (a remembered chair's centre): the policy's field is +inf
    // everywhere, and it DRIVES anyway (live: Release, ~2 m back, no_route). The yardstick is the region goal's field, so
    // the oracle judges it against the nearest reachable place; the driver keeps the raw goal (no region in today's code).
    if (opt.truth_yardstick)
    {
        const MapFree yard(world, opt.grid, {}, 0.05f, false);
        Navigator stick;
        stick.set_goal(goal, yard.grid, [&yard](const V2& c) { return yard(c); }, [](const V2&) { return 0.f; }, goal_points(goal),
                       B.circumscribed_radius());
        ref_field = stick.field();
        res.geodesic_m = ref_field.at(start.p);
    }
    bool region_stick = false;
    V2 region_p = goal.p;
    if (not std::isfinite(res.geodesic_m) and goal.kind == Goal::Kind::Place)
    {
        const Goal rg = region_goal(world, opt.grid, start.p, goal);
        Navigator stick;
        stick.set_goal(rg, mapfree.grid, [&mapfree](const V2& c) { return mapfree(c); }, [](const V2&) { return 0.f; }, goal_points(rg),
                       B.circumscribed_radius());
        ref_field = stick.field();
        res.geodesic_m = ref_field.at(start.p);
        region_stick = std::isfinite(res.geodesic_m);
        region_p = rg.p;
        if (tr) std::printf("    raw goal unreachable on the map; region goal (%+.2f,%+.2f), geodesic %.2f m\n", rg.p.x(), rg.p.y(), res.geodesic_m);
    }
    float ref_best = ref_field.at(start.p);
    float last_sweep_t = 0.f;
    std::normal_distribution<float> jit(0.f, 1.f);
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
        // the protocol's through (I3), judged on the believed pose like the controller's measured gap
        if (goal.kind == Goal::Kind::Doorway and doorway_through(goal, bel.p)) { res.reached = true; res.why = "through"; break; }
        if (region_stick and (bel.p - region_p).norm() <= goal.tol_xy) { res.reached = true; res.why = "at the region goal"; break; }

        if (opt.goal_jitter_m > 0.f)   // a PREDICTED anchor: the goal re-measured, the field rebuilt (as the controller does)
        {
            Goal gj = goal; gj.p += opt.goal_jitter_m * V2(jit(rng), jit(rng));
            install_goal(gj); res.build_ms += nav.last_build_ms(); ++res.builds;
        }
        const bool dropout = std::ranges::any_of(opt.lidar_dropout, [&](const auto& w) { return t >= w.first and t < w.second; });
        // ---- both LiDARs from the TRUE pose, re-expressed in the room frame on the pose believed at capture, buffered
        std::vector<Return> sweep;
        const V2 bf(std::cos(bel.yaw), std::sin(bel.yaw)), br(std::sin(bel.yaw), -std::cos(bel.yaw));
        for (const auto& m : {kHelios, kBpearl})
            for (const V2& q : sim2d::scan_mount(world, truth, m, rng, sigma_r()))
                sweep.push_back({bel.p + q.x() * br + q.y() * bf, sigma_r(), 1.f});
        if (not dropout) { buffer.push_back(std::move(sweep)); last_sweep_t = t; }
        while (int(buffer.size()) > kBufferCycles) buffer.pop_front();
        std::vector<Return> ret;
        for (const auto& sw : buffer) ret.insert(ret.end(), sw.begin(), sw.end());
        if (opt.unseen) for (const auto& h : unseen.hypotheses(bel, odom)) ret.push_back(h);   // odometry -> room frame

        PolicyInput in;
        in.pose = bel; in.v_now = base.v; in.w_now = base.w;
        in.sigma_rel_xy = 0.01f; in.sigma_rel_yaw = 0.01f;                                       // odometry-scale RELATIVE error
        if (std::getenv("POLICY_CAMERA") == nullptr or std::string(std::getenv("POLICY_CAMERA")) != "0") { in.camera_half_fov_rad = 0.96f; in.affordance_facing = not std::getenv("POLICY_NOFACING"); }   // Task R: ZED2i 110 deg HFOV
        if (opt.ou) { in.pose_drift_sigma_m = err.sigma_xy; in.pose_drift_time_s = err.tau; }   // the OU's own process (monitor)
        in.returns_age_s = float(kBufferCycles - 1) * kTick + (t - last_sweep_t);               // the older sweep's age (grows in a stall)
        in.returns = ret; in.body = &B; in.base = base.lim; in.latency_s = base.latency_s; in.now_s = t;
        const auto t_step = std::chrono::steady_clock::now();
        const Navigator::Step st = nav.step(in, pp);
        res.step_ms.push_back(float(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_step).count()));
        res.returns_max = std::max(res.returns_max, ret.size());
        const Decision& d = st.d;
        if (res.first_mode.empty()) { res.first_mode = mode_name(d.mode); res.first_v = d.v; res.first_w = d.w; }
        const bool is_stop = d.mode == Decision::Mode::Release and d.v == 0.f and d.w == 0.f;
        if (is_stop and st.replanned) ++res.stops;
        else if (res.stops > 0 and res.after_stop_mode.empty() and st.replanned) { res.after_stop_mode = mode_name(d.mode); res.after_stop_v = d.v; res.after_stop_w = d.w; }
        const float fnow = nav.field().at(bel.p);
        const float clear = contact.clearance(truth);
        res.min_clear = std::min(res.min_clear, clear);
        if (tr)
            std::printf("    t=%6.2f true(%+.3f,%+.3f,%+6.1f) bel(%+.3f,%+.3f) %-7s v=%+.2f w=%+.2f el=%d P=%.3f lpc=%.2f prog=%+.3f clear=%+.3f f=%.3f | %s\n",
                        t, truth.p.x(), truth.p.y(), truth.yaw / kDeg, bel.p.x(), bel.p.y(), mode_name(d.mode), d.v, d.w, d.eligible,
                        1.f - std::exp(d.log_p_safe), d.log_p_cont, d.progress_m, clear, fnow, d.why.c_str());

        if (d.mode == Decision::Mode::Arrived) { res.reached = true; res.why = "arrived"; break; }
        // Blind: the controller HOLDS and decides again on the next sweep (its stream watchdog); a stall that never ends is the time limit
        if (d.mode == Decision::Mode::Blind and opt.lidar_dropout.empty()) { res.stuck = true; res.why = d.why; break; }
        if (d.mode == Decision::Mode::Stuck)
        {
            // ★ F1: a Stuck is one decision, not a verdict — the base holds (d = 0, 0 is sent below) and the next tick decides
            //   again; the PLACE-level verdict is the reachability belief's hand back (below)
            res.stuck = true; ++res.stuck_ticks; res.why = d.why;
            // F2: a Stuck while MOVING (the policy's own deadband test on the odometry it was given) — not even the stop releasable
            res.stuck_moving = res.stuck_moving or std::abs(in.v_now) >= in.base.v_min_eff or std::abs(in.w_now) >= in.base.w_min_eff;
        }
        const bool no_escape = st.replanned and (d.mode == Decision::Mode::Stuck
                               or (d.mode == Decision::Mode::Release and not (d.log_p_cont > d.log_p_cont_now)));
        if (no_escape and res.first_no_escape_t < 0.f) res.first_no_escape_t = t;
        if (st.replanned and (d.mode == Decision::Mode::Drive or d.mode == Decision::Mode::Release)) {
            ++g_moves;
            if (d.v < 0.f)
            {
                ++g_rev;
                // Task R reverse breakdown. Policy view: Release = unlock/unstall; Drive kind 1 = no seen-progress plan exists (dead
                // end / stall / tight corridor), kind 2 = an alternative existed but the reversal outscored it; kind 0 = other.
                // Independent view (true bare-hull clearance at the decision): tight = an obstacle within 0.25 m (~ inscribed radius).
                // P4 buckets: release/unlock | dead end / stall (no non-reversing plan descends) | tight (one did, but an obstacle is
                // within 0.25 m ~ the inscribed radius) | OPEN (one did and nothing is that close: should be ~0)
                if (d.mode == Decision::Mode::Release) ++g_rel;
                else if (not d.forward_alt) ++g_rk1;
                else if (clear < 0.25f) ++g_rk2;
                else ++g_rk0;
                if (clear < 0.25f) ++g_rtight; else ++g_rclear;
            }
        }   // Task R reverse share
        if (st.replanned and d.v < 0.f and (d.mode == Decision::Mode::Drive or d.mode == Decision::Mode::Release)) ++res.backups;
        if (tr) std::printf("    belief L %+.2f bar %.2f judged %+d armed %d early %d%s\n", st.unreach_L, st.unreach_bar, d.judged, int(d.armed), int(st.completed_early), st.hand_back ? "  HAND BACK" : "");
        if (d.mode == Decision::Mode::Release and not in_release_episode) { in_release_episode = true; ++res.releases; }
        if (tr and st.replanned) std::printf("    (replanned; remaining-plan P was %.3f)\n", st.plan_remaining_p);
        if (tr and st.trap_suppressed_by_goal) std::printf("    no trap: its disc would contain the goal\n");
        if (st.trap)
        {
            ++res.traps; res.build_ms += nav.last_build_ms(); ++res.builds;
            true_traps.push_back({err.to_truth(st.trap->c), st.trap->r});
            if (not opt.truth_yardstick) best_f = nav.field().at(bel.p);      // the trapped field changed: progress is measured on the new one (clock kept)
            if (tr) std::printf("    TRAP (blocked disc) at (%+.3f,%+.3f) dir (%+.2f,%+.2f); field now %.3f\n", st.trap->c.x(), st.trap->c.y(),
                                st.trap->dir.x(), st.trap->dir.y(), best_f);
        }
        if (st.unreachable) { res.unreachable = true; res.why = "goal not reachable on the trapped field"; break; }
        if (st.hand_back and std::getenv("NO_HANDBACK") == nullptr)   // NO_HANDBACK=1: DIAGNOSTIC — is the goal passable at all?
        {
            res.unreachable = true; res.hand_back = true; res.hand_back_L = st.unreach_L; res.hand_back_bar = st.unreach_bar;
            char buf[160]; std::snprintf(buf, sizeof buf, "reachability belief: L %.2f > bar %.2f (%s)", st.unreach_L, st.unreach_bar, d.why.c_str());
            res.why = buf; break;
        }

        // ---- the base: command latency (+ jitter), ramps, deadband (x scale), slip; contact every <= 5 mm
        base.send(t, d.v, d.w);
        const Pose2 before = truth, odom_before = odom;
        const float v_meas = base.v, w_meas = base.w;     // odometry at the start of the tick
        const float sv = 1.f + slip(rng), sw = 1.f + slip(rng);
        bool touched = false;
        std::vector<UnseenContact::CmdSeg> in_force;     // the commands the wheels actually followed this tick
        for (float s = 0.f; s < kTick - 1e-6f; s += kSub)
        {
            base.step_velocity(t + s, kSub);
            if (not in_force.empty() and in_force.back().v == base.cmd_v and in_force.back().w == base.cmd_w) in_force.back().dt += kSub;
            else in_force.push_back({kSub, base.cmd_v, base.cmd_w});
            const Pose2 nxt = sim2d::integrate(truth, base.v * sv, base.w * sw, kSub);
            // the world pushes back: the body stays put and the base stalls (its wheels stop)
            if (contact.touches(nxt)) { touched = true; base.v = base.w = 0.f; continue; }
            odom = sim2d::integrate(odom, base.v, base.w, kSub);
            if (base.v * sv < 0.f and contact.clearance(truth) >= 0.5f) res.open_reverse_m += (nxt.p - truth.p).norm();
            truth = nxt;
        }
        if (touched) { ++res.contacts; if (d.mode == Decision::Mode::Release) ++res.release_contacts; }
        // odometry: the encoders' displacement along the heading at the start of the tick, and the turn
        const V2 fwd(std::cos(odom_before.yaw), std::sin(odom_before.yaw));
        const float ds = (odom.p - odom_before.p).dot(fwd), dyaw = sim2d::wrap(odom.yaw - odom_before.yaw);
        // ★ D3: the command IN EFFECT at the wheels and the measured velocity at the tick's start
        // ★ review fix round 3: the commands IN FORCE per sub-interval (the latency queue), not the latest one
        unseen.update(std::span<const UnseenContact::CmdSeg>(in_force), v_meas, w_meas, ds, dyaw, odom, base.lim, B, 0.005f);
        res.path_m += (truth.p - before.p).norm();
        if (const float fr = ref_field.at(truth.p); std::isfinite(fr)) { ref_best = std::min(ref_best, fr); res.max_retreat_m = std::max(res.max_retreat_m, fr - ref_best); }

        // ★ TRAJECTORY ORACLE (review H3): once the true centre has left a trap's disc it must never re-enter it
        for (auto& tt : true_traps)
        {
            const float dist = (truth.p - tt.c).norm();
            if (dist > tt.r) tt.left = true;
            else if (tt.left) res.trajectory_ok = false;
        }

        const float fn = opt.truth_yardstick ? ref_field.at(truth.p) : nav.field().at(err.believed(truth).p);
        if (fn < best_f - 1e-3f) { best_f = fn; last_progress_t = t + kTick; }
        if (d.mode != Decision::Mode::Release) in_release_episode = false;   // round 4: an EPISODE = consecutive releases
        res.max_no_progress_s = std::max(res.max_no_progress_s, t + kTick - last_progress_t);
    }
    res.time_s = t;
    res.end = truth;
    res.failed_predictions = nav.failed_predictions();
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
    if (leaf == 1 or leaf == 3) w.boxes.push_back({-0.525f, -1.15f, -0.475f, -0.20f, leaf == 1});   // 3: explained (Task 3b)
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
struct Summary { int max_failed = 0; std::vector<float> step_ms; std::size_t returns_max = 0; float goal_truth_clear = 1e9f; double build_ms = 0.0; int builds = 0; int n = 0, reached = 0, unreachable = 0, stuck = 0, contacts = 0, rel_contacts = 0, max_releases = 0, total_releases = 0;
                 int hand_backs = 0, stuck_ticks = 0, min_backups = 1 << 30; std::vector<float> tverdict_after_no_escape;
                 bool traj_ok = true; int total_traps = 0, total_stops = 0, stuck_moving = 0; std::vector<float> tunreach, tstuck; float max_np = 0.f, max_path_ratio = 0.f, min_clear = 1e9f; std::vector<float> tgoal;
                 float max_open_reverse = 0.f, max_retreat = 0.f; double xtrack_sum2 = 0.0; long xtrack_n = 0; int handoffs = 0, handoff_window = 0, handoff_stall = 0, replans = 0, waits = 0, wait_exits = 0, max_waits = 0; };

static Summary run_seeds(const std::string& name, const std::function<ScenarioResult(unsigned)>& fn)
{
    int nseeds = 20;
    if (const char* s = std::getenv("SEEDS")) nseeds = std::max(1, std::atoi(s));
    std::vector<std::future<ScenarioResult>> fut;
    int first = 0;   // FIRST_SEED=<k> (diagnostic): run seeds k .. k+n-1
    if (const char* f = std::getenv("FIRST_SEED")) first = std::max(0, std::atoi(f));
    for (int s = 0; s < nseeds; ++s) fut.push_back(std::async(std::launch::async, fn, unsigned(first + s)));
    Summary S;
    std::printf("== %s\n   seed reached unreach stuck contacts(rel) releases  t_goal  path  geo  max_np  min_clear  traj  why\n", name.c_str());
    for (int s = 0; s < nseeds; ++s)
    {
        const auto r = fut[s].get();
        ++S.n; S.reached += r.reached; S.unreachable += r.unreachable; S.stuck += r.stuck; S.contacts += r.contacts;
        S.rel_contacts += r.release_contacts; S.max_releases = std::max(S.max_releases, r.releases); S.total_releases += r.releases;
        S.traj_ok = S.traj_ok and r.trajectory_ok; S.total_traps += r.traps; S.total_stops += r.stops; S.max_np = std::max(S.max_np, r.max_no_progress_s);
        S.min_clear = std::min(S.min_clear, r.min_clear);
        if (r.geodesic_m > 0.f) S.max_path_ratio = std::max(S.max_path_ratio, r.path_m / r.geodesic_m);
        if (r.reached) S.tgoal.push_back(r.time_s);
        if (r.unreachable) S.tunreach.push_back(r.time_s);
        if (r.stuck) S.tstuck.push_back(r.time_s);
        S.stuck_moving += r.stuck_moving;
        S.hand_backs += r.hand_back; S.stuck_ticks += r.stuck_ticks; S.min_backups = std::min(S.min_backups, r.backups);
        if (r.unreachable and r.first_no_escape_t >= 0.f) S.tverdict_after_no_escape.push_back(r.time_s - r.first_no_escape_t);
        S.max_failed = std::max(S.max_failed, r.failed_predictions);
        S.step_ms.insert(S.step_ms.end(), r.step_ms.begin(), r.step_ms.end()); S.returns_max = std::max(S.returns_max, r.returns_max);
        S.build_ms += r.build_ms; S.builds += r.builds; S.goal_truth_clear = std::min(S.goal_truth_clear, r.goal_truth_clear);
        S.max_open_reverse = std::max(S.max_open_reverse, r.open_reverse_m); S.max_retreat = std::max(S.max_retreat, r.max_retreat_m);
        S.xtrack_sum2 += r.xtrack_sum2; S.xtrack_n += r.xtrack_n; S.handoffs += r.handoffs; S.handoff_window += r.handoff_window; S.handoff_stall += r.handoff_stall; S.replans += r.replans;
        S.waits += r.waits; S.wait_exits += r.wait_exits; S.max_waits = std::max(S.max_waits, r.waits);
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
    std::printf("   open-space reverse travel max %.2f m per run; largest retreat from the goal %.2f m\n", S.max_open_reverse, S.max_retreat);
    if (S.xtrack_n > 0)
        std::printf("   chain: cross-track rms %.3f m over %ld ticks; hand-offs to the policy %d (window %d, stall %d); re-plans on map + live %d\n",
                    std::sqrt(S.xtrack_sum2 / double(S.xtrack_n)), S.xtrack_n, S.handoffs, S.handoff_window, S.handoff_stall, S.replans);
    std::printf("   waits %d (max %d per run), wait exits %d\n", S.waits, S.max_waits, S.wait_exits);
    std::printf("   Release(stop) decisions (F2): %d over all seeds; Stuck ticks held (F1) %d; belief hand backs %d\n", S.total_stops,
                S.stuck_ticks, S.hand_backs);
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

// ★ REVIEW FIX ROUND 4 — PERFORMANCE ORACLE: each scenario's median time-to-goal <= 1.15 x its BASELINE median.
// ★ RE-BASELINED (Task 8 Live fix 2 ruling, 2026-10-20): the old baselines (review round 2) were measured on a base with
//   INDEPENDENT v / w ramps, which does not exist; the plant is now wheel-coupled (sim2d::Base, rc::policy::rollout). The
//   baselines are the 20-seed medians of the first wheel-coupled run (commit 1e4df350), in s:
//     tour3_1_leg3 2.75 (was 2.55) · try101_corridor 4.05 (3.45) · try105_jamb 2.85 (2.50) · jamb_1p6cm_start 5.55 (4.95) ·
//     door_crossing_both_ways 4.60 / 4.60 (5.10 / 5.05; the controller's stated 5.75 s cap kept) · dead_end_release 29.00
//     (27.85) · goal_behind_open_space 4.05 (4.30) · its OU variant 4.40 (4.65) · pose_jump 6.70 (7.00).
static void performance_oracle(const std::string& n, const Summary& S, float base_median, float cap = 1e9f)
{
    const float limit = std::min(1.15f * base_median, cap);
    const float med = S.tgoal.empty() ? 1e9f : S.tgoal[S.tgoal.size() / 2];
    std::printf("   time-to-goal median %.2f s, limit %.2f s (1.15 x %.2f)\n", med, limit, base_median);
    CHECK((n + ": median time-to-goal <= 1.15 x the wheel-coupled baseline").c_str(), med <= limit);
}

static void standard_oracles(const std::string& n, const Summary& S)
{
    CHECK((n + ": contacts == 0").c_str(), S.contacts == 0);
    CHECK((n + ": reached on every seed").c_str(), S.reached == S.n);
    CHECK((n + ": no window > 20 s without field decrease").c_str(), S.max_np <= 20.f);
    CHECK((n + ": after a trap the true robot never re-enters it (trajectory)").c_str(), S.traj_ok);
}


// ---------------------------------------------------------------- the apartment (threerooms-piso.wbt)
// sim2d/apartment_threerooms.txt, exported by sim2d/export_apartment.py from the Webots world (walls and the door frames at the
// LiDAR band, 0.5 m): 'seg x0 y0 x1 y1' and 'door name cx cy nx ny tx ty width'. Parsed with std::from_chars (locale-proof).
struct AptDoor { std::string name; V2 c, n, tng; float width; };
struct Apartment { World w; std::vector<AptDoor> doors; };
static const Apartment& apartment()
{
    static const Apartment A = [] {
        Apartment a;
        const char* path = std::getenv("APARTMENT");
        std::ifstream f(path ? path : "tests/sim2d/apartment_threerooms.txt");
        std::string line;
        const auto num = [](std::string_view s, float& v) { return std::from_chars(s.data(), s.data() + s.size(), v).ec == std::errc{}; };
        while (std::getline(f, line))
        {
            std::vector<std::string_view> tok;
            for (std::size_t i = 0; i < line.size();)
            {
                while (i < line.size() and line[i] == ' ') ++i;
                const std::size_t b = i;
                while (i < line.size() and line[i] != ' ') ++i;
                if (i > b) tok.emplace_back(line.data() + b, i - b);
            }
            if (tok.size() == 5 and tok[0] == "seg")
            {
                float v[4]; bool ok = true;
                for (int k = 0; k < 4; ++k) ok = ok and num(tok[std::size_t(k) + 1], v[k]);
                if (ok) a.w.segments.push_back({V2(v[0], v[1]), V2(v[2], v[3])});
            }
            else if (tok.size() == 9 and tok[0] == "door")
            {
                float v[7]; bool ok = true;
                for (int k = 0; k < 7; ++k) ok = ok and num(tok[std::size_t(k) + 2], v[k]);
                if (ok) a.doors.push_back({std::string(tok[1]), V2(v[0], v[1]), V2(v[2], v[3]).normalized(), V2(v[4], v[5]).normalized(), v[6]});
            }
        }
        return a; }();
    return A;
}
static const AptDoor& apt_door(const std::string& n)
{
    for (const auto& d : apartment().doors) if (d.name == n) return d;
    std::printf("apartment: no door %s\n", n.c_str()); std::exit(2);
}
// a leaf 0.9 m long, hinged at the jamb on the +tangent (hinge > 0) or -tangent side, opened 90 deg toward `side` (+-1 along n)
static World with_leaf(World w, const AptDoor& d, int hinge, int side, bool in_map)
{
    const V2 h = d.c + float(hinge) * 0.5f * d.width * d.tng, tip = h + float(side) * 0.9f * d.n;
    const V2 lo = h.cwiseMin(tip) - V2(0.02f, 0.02f), hi = h.cwiseMax(tip) + V2(0.02f, 0.02f);
    w.boxes.push_back({lo.x(), lo.y(), hi.x(), hi.y(), in_map});
    return w;
}
static constexpr Bounds kAptGrid{-8.9f, -0.6f, 2.2f, 14.0f};
struct AptRoom { const char* name; Bounds r; };
static const AptRoom kAptRooms[] = {{"room_1", {-8.0f, 0.6f, -0.6f, 6.6f}}, {"room_2", {-1.3f, 7.7f, 1.4f, 13.2f}}, {"room_3", {-8.0f, 9.8f, -2.1f, 13.2f}}};
// a pose in room r whose TRUE clearance is >= clear (any yaw), drawn with the seed's own generator
static Pose2 apt_sample(const World& w, const Bounds& r, float clear, std::mt19937& g)
{
    std::uniform_real_distribution<float> ux(r.x0, r.x1), uy(r.y0, r.y1), uyaw(-kPi, kPi);
    for (int k = 0; k < 10000; ++k)
    {
        const V2 p(ux(g), uy(g));
        if (w.clearance(p, false) >= clear) return Pose2{p, uyaw(g)};
    }
    return Pose2{V2(0.5f * (r.x0 + r.x1), 0.5f * (r.y0 + r.y1)), 0.f};
}
// ★ APARTMENT ORACLES: the standard ones, plus open-space reversing and retreat. A long traverse in open space never needs either:
//   0.5 m of reverse travel and a 1.0 m retreat are generous acceptance bounds for a test, not model parameters.
static void apartment_oracles(const std::string& n, const Summary& S, bool reach = true)
{
    CHECK((n + ": contacts == 0").c_str(), S.contacts == 0);
    if (reach) CHECK((n + ": reached on every seed").c_str(), S.reached == S.n);
    CHECK((n + ": no window > 20 s without field decrease").c_str(), S.max_np <= 20.f);
    CHECK((n + ": open-space reverse travel <= 0.5 m per run").c_str(), S.max_open_reverse <= 0.5f);
    CHECK((n + ": never retreats > 1.0 m from its best distance to the goal").c_str(), S.max_retreat <= 1.0f);
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
        // ★ Task R (R1): the baseline encoded the OPENING back-up arc (goal 0.7 m behind-left, wall on the right: 2.75 s). The rule "back up only
        //   to unlock / unstall / in a tight manoeuvre" turns left away from the wall and drives forward: measured median 5.30 s (with the reverse dispreference).
        performance_oracle("tour3_1_leg3", S, camera_model_on() ? 5.30f : 2.75f);
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
        performance_oracle("try101_corridor", S, 4.05f);
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
        performance_oracle("try105_jamb", S, 2.85f);
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
        performance_oracle("jamb_1p6cm_start", S, 5.55f);
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
            // ★ Task R (R1): the return leg's goal is BEHIND the robot once it has crossed. The old 4.60 s (cap 5.75 s) encoded backing
            //   through the door - the live observation this task answers ("you just crossed backwards the door"). Now it turns round
            //   and crosses facing the opening: measured medians 6.65 s (leaf into room_1) / 6.50 s (room_2), same limit rule (1.15x).
            performance_oracle(n, S, camera_model_on() ? (leaf == 1 ? 6.65f : 6.50f) : 4.60f, camera_model_on() ? 1e9f : 5.75f);
        }
    // ---- leaf_known_hinge_side (Task 3b, rtD1c leg 5 + fix round 1): the leaf open 90 deg into room_1 and a CHAIR inside its
    //      swept region (2 sigma_phi east of the free edge, against the leaf's tip) are physical but not in the map; residual
    //      holds them as the controller's planner would: the leaf as it hangs and the chair HARD (live returns: occupied,
    //      never explained away — I-1), the smear the leaf left while it swung SOFT (4 nats a cell). door_concept's leaf (two
    //      hypotheses: the west hinge P 0.9, the east one 0.1; sigma 2 cm, sigma_phi 3 deg) re-prices the soft belief. The
    //      robot starts on the hinge side, 0.65 m west of the leaf, facing away from the door: the straight line to the door
    //      runs through the leaf. Diagnostics: NO_LEAF=1 (the smear unexplained: residual alone), SX/SY/SYAW (start).
    //      ⚑ A SAFETY scenario, not a discriminating one: with the live returns kept hard the leaf blocks as it did in the map,
    //      and NO_LEAF=1 also reaches 20/20 (measured, fix round 1). It checks that the re-pricing frees nothing occupied (the
    //      chair stays non-free, 0 contacts) and costs no crossing. The discriminating cases are test_door_leaf.cpp's (the
    //      coupling: 41 vs 2.3 nats on a seen leaf; the unseen leaf; the smear's price at the pocket 13.8 -> 8.1).
    if (want("leaf_known_hinge_side"))
    {
        // DOOR_1 with room_1's west side opened up to x = -1.80 (the copied scene's hinge-side pocket is 0.585 m for a 0.568 m
        // body: no room to turn, a test of being boxed, not of the leaf)
        World w = door_world(3);
        std::erase_if(w.boxes, [](const sim2d::Box& b) { return b.x1 <= -1.10f and b.y1 > -2.32f and b.y0 < -0.23f and b.x0 > -2.f; });
        w.boxes.push_back({-1.91f, -2.42f, -1.80f, -0.11f});
        w.boxes.push_back({-1.91f, -0.24f, -0.50f, -0.11f});
        constexpr float kSphi = 3.f * kDeg;
        auto hyp = [&](float hinge_x, float p)
        {
            LeafHypothesis h;
            h.hinge = V2(hinge_x, -0.2f); h.dir = V2(0.f, -1.f); h.length = 0.95f; h.p = p;
            h.pivot = V2(0.f, -0.1f); h.P_xy = Eigen::Matrix2f::Identity() * (0.02f * 0.02f);
            h.var_psi = 0.01f * 0.01f; h.var_phi = kSphi * kSphi;
            return h;
        };
        // the chair: 0.10 x 0.10 m, centred 2 sigma_phi east of the MAP free edge (where only the leaf's swing could be)
        const V2 chair(-0.5f + 0.95f * std::sin(2.f * kSphi), -0.2f - 0.95f * std::cos(2.f * kSphi));
        w.boxes.push_back({chair.x() - 0.05f, chair.y() - 0.05f, chair.x() + 0.05f, chair.y() + 0.05f, false});
        Options o{.grid = kDoorGrid, .jumps = {}, .trace_name = "leaf_known_hinge_side"};
        auto add = [&](V2 c, float r)
        {
            c = V2(std::round(c.x() / 0.05f) * 0.05f, std::round(c.y() / 0.05f) * 0.05f);
            for (const auto& e : o.res_cells) if ((e - c).squaredNorm() < 1e-8f) return;
            o.res_cells.push_back(c); o.res_risk.push_back(r);
        };
        const float inf = std::numeric_limits<float>::infinity();
        for (int i = 0; i <= 19; ++i) add(V2(-0.5f, -0.2f - 0.05f * float(i)), inf);                        // the leaf now: hard
        for (float x = chair.x() - 0.05f; x <= chair.x() + 0.05f + 1e-4f; x += 0.05f)                         // the chair: hard
            for (float y = chair.y() - 0.05f; y <= chair.y() + 0.05f + 1e-4f; y += 0.05f) add(V2(x, y), inf);
        for (const float a : {-2.f, -1.f, 1.f, 2.f})                                                           // the smear: soft
            for (int i = 0; i <= 19; ++i)
            {
                const float r = 0.05f * float(i);
                add(V2(-0.5f + r * std::sin(a * kSphi), -0.2f - r * std::cos(a * kSphi)), 4.f);
            }
        DoorLeaf L; L.hyps = {hyp(-0.5f, 0.9f), hyp(0.5f, 0.1f)};
        if (std::getenv("NO_LEAF") == nullptr) o.leaf = explain_residual_by_leaf(L, o.res_cells, 0.05f, o.res_risk, B.inscribed_radius());
        int hard = 0; for (const float r : o.res_risk) hard += not std::isfinite(r);
        std::printf("leaf_known_hinge_side: %zu residual cells (%d hard, kept hard: %d in the leaf's swept region); %zu carry the door's excess\n",
                    o.res_cells.size(), hard, o.leaf.hard_in_swept, o.leaf.cells.size());
        {
            std::vector<V2> hc; for (std::size_t i = 0; i < o.res_cells.size(); ++i) if (not std::isfinite(o.res_risk[i])) hc.push_back(o.res_cells[i]);
            const MapFree mf(w, kDoorGrid, hc);
            CHECK("leaf_known_hinge_side: the field stays non-free at the chair (a hard cell in the swept region)", not mf(chair));
        }
        auto envf = [](const char* k, float d) { float v = d; if (const char* e = std::getenv(k)) std::from_chars(e, e + std::strlen(e), v); return v; };
        const Pose2 s{V2(envf("SX", -1.15f), envf("SY", -0.80f)), envf("SYAW", -90.f) * kDeg};
        sim2d::ContactTester ct(w, B);
        std::printf("leaf_known_hinge_side start clearance %.4f m\n", ct.clearance(s));
        const auto S = run_seeds("leaf_known_hinge_side", [&](unsigned seed) { return run_scenario(w, s, doorway(+1), 60.f, seed, o); });
        standard_oracles("leaf_known_hinge_side", S);
        std::printf("   belief hand backs %d (a passable door with a known leaf must not lean Unreachable)\n", S.hand_backs);
        CHECK("leaf_known_hinge_side: 0 belief hand backs", S.hand_backs == 0);
    }
    // ---- dead_end_release: the map says the 0.8 m passage leads to the goal; a wall it does not know closes it
    if (want("dead_end_release"))
    {
        const World w = dead_end_world();
        const Goal g{.kind = Goal::Kind::Place, .p = V2(0.f, 3.5f), .yaw = {}};
        Options o{.grid = kDeadEndGrid, .jumps = {}, .trace_name = "dead_end_release"};
        const auto S = run_seeds("dead_end_release", [&](unsigned seed) { return run_scenario(w, Pose2{V2(0.f, -1.5f), 90.f * kDeg}, g, 120.f, seed, o); });
        standard_oracles("dead_end_release", S);
        performance_oracle("dead_end_release", S, 29.00f);
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
        // ★ round 5 (RF4-c): FAILED predictions before Unreachable (successful releases are legitimate repositioning)
        std::printf("   failed predictions before Unreachable: max %d\n", S.max_failed);
        CHECK("narrow_doorway_unreachable: <= 2 failed predictions", S.max_failed <= 2);
    }
    // ---- goal_behind_open_space: path <= 1.5 x geodesic
    if (want("goal_behind_open_space"))
    {
        const World w = open_world();
        const Goal g{.kind = Goal::Kind::Place, .p = V2(-2.f, 0.5f), .yaw = {}};
        Options o{.grid = kOpenGrid, .jumps = {}, .trace_name = "goal_behind_open_space"};
        const auto S = run_seeds("goal_behind_open_space", [&](unsigned seed) { return run_scenario(w, Pose2{V2(0.f, 0.f), 0.f}, g, 30.f, seed, o); });
        standard_oracles("goal_behind_open_space", S);
        // ★ Task R (R1): the baseline encoded REVERSING: the goal 2 m behind was reached by backing up straight (4.05 s). The user's rule - only
        //   back up to unlock / unstall / in a tight manoeuvre - makes the robot turn (~pi at w_max 1.5 = 2.1 s) and drive forward
        //   (~2.2 m at 0.5 m/s with the ramps): measured 5.60 s with the camera model on. All safety/arrival checks are kept.
        performance_oracle("goal_behind_open_space", S, camera_model_on() ? 5.60f : 4.05f);
        CHECK("goal_behind_open_space: path <= 1.5 x geodesic", S.max_path_ratio <= 1.5f);
        // the same with the measured OU pose-error process (0.5 m / 13 deg, tau 20 s): contact is judged on the true pose
        Options ou = o; ou.ou = true; ou.trace_name = "goal_behind_open_space_ou";
        const auto S2 = run_seeds("goal_behind_open_space + OU pose error", [&](unsigned seed) { return run_scenario(w, Pose2{V2(0.f, 0.f), 0.f}, g, 30.f, seed, ou); });
        CHECK("goal_behind_open_space OU: contacts == 0", S2.contacts == 0);
        CHECK("goal_behind_open_space OU: reached on every seed", S2.reached == S2.n);
        performance_oracle("goal_behind_open_space OU", S2, camera_model_on() ? 6.35f : 4.40f);   // Task R (R1): as above, measured 6.35 s with the pose-error process
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
        performance_oracle("pose_jump", S, 6.70f);
        std::printf("   the believed goal maps, after the jump, to a true place %.3f m from the nearest wall\n", S.goal_truth_clear);
        CHECK("pose_jump: the jumped goal is a place the body can be (true clearance >= inscribed radius)", S.goal_truth_clear >= body().inscribed_radius());
    }
    // ---- stop_is_the_only_release (F2, 2026-10-21 doorway-release spec, scenario (c)): the robot CREEPS forward into a
    //      pocket — a wall across its nose, a wall across its tail, the sides open. Every reverse backs further than the gap
    //      behind; every in-place turn is rolled out from the MOVING state with its wheels coupled, so it arcs forward and swings
    //      a front corner into the nose wall; only the brake-to-rest sweep is releasable. Before F2 (stop not a candidate) the
    //      first decision is Stuck on every seed. With F2: Release(stop), then — at rest — an ordinary decision that moves.
    //      The band is narrow because a decision from rest evaluates the same post-coast motions, only without the coupled arc.
    //   (c1) TRAP: nose 6 cm past the rest pose, tail 5 cm behind, walls 0.70 m long (v0 0.10 m/s). No 90 deg turn fits between
    //        the walls, so every run must end in a verdict about the PLACE: the trapped field's Unreachable (15/20) or a Stuck
    //        FROM REST (5/20 — the stop phase now ends at rest, navigator.cpp, and from some rest poses no turn is releasable).
    //        Never a Stuck while moving. Pre-F1 (fix round 1): 15/20 the trapped field's Unreachable, 5/20 a terminal Stuck from
    //        rest, verdict max 3.90 s. ★ F1: every seed ends Unreachable (the from-rest Stuck is a decision folded into the belief,
    //        never a terminal verdict). MODEL-STATED TIME (fix round 1): after the stop phase (~0.15 s to rest) the pocket's
    //        decisions from rest find no eligible option that shortens T (full counts, +log(n_eff + 1) ~ 4 each at exposure
    //        ~0.75-1) against bar log(T / dt_obs) with dt_obs the release holds (~1 s): 2 of them when the field is untouched; when
    //        a failed release lays a trap the belief restarts on the new route and needs 2 more. Between two of them the robot
    //        holds a release (horizon_s = 1 s) and brakes (<= 0.15 s): <= 0.2 + 2 x 1.15 = 2.5 s. Measured: max 2.40 s.
    //   (c2) REACH: nose 8 cm past the rest pose, tail 16 cm behind, walls 1.20 m long, v0 0.20 m/s; goal 1.5 m to the left.
    //        Pre-F2 code: Stuck 20/20 on the first tick. With F2: Release(stop) 20/20, then from rest the in-place turn clears the
    //        walls (the 0.318 m corner circle fits) and the robot drives out — reached 20/20, time-to-goal median 5.05 s, max
    //        5.75 s (fix round 1); oracle: median <= 1.15 x 5.05 s.
    if (want("stop_is_the_only_release"))
    {
        constexpr float kTrapVerdictMaxS = 2.5f;      // (c1)'s model-stated bound on the time to Unreachable (F1, above)
        struct Pocket { const char* name; float v0, gap_nose, gap_tail, half; };
        auto run_pocket = [&](const Pocket& pk, int* first_stop, int* moves_after) {
            const float coast = pk.v0 * 0.06f + pk.v0 * pk.v0 / (2.f * limits().d_lin), nose = 0.23f + coast + pk.gap_nose, tail = -0.23f - pk.gap_tail;
            World w = open_world();
            w.boxes.push_back({-pk.half, nose, pk.half, nose + 0.1f});
            w.boxes.push_back({-pk.half, tail - 0.1f, pk.half, tail});
            const Goal g{.kind = Goal::Kind::Place, .p = V2(-1.5f, 0.f), .yaw = {}};
            Options o{.grid = kOpenGrid, .jumps = {}, .trace_name = pk.name, .v0 = pk.v0};
            std::mutex m;
            *first_stop = 0; *moves_after = 0;
            const auto S = run_seeds(pk.name, [&](unsigned seed) {
                auto r = run_scenario(w, Pose2{V2(0.f, 0.f), 90.f * kDeg}, g, 30.f, seed, o);
                const std::lock_guard lk(m);
                *first_stop += r.first_mode == "Release" and r.first_v == 0.f and r.first_w == 0.f;
                *moves_after += (r.after_stop_mode == "Release" or r.after_stop_mode == "Drive") and not (r.after_stop_v == 0.f and r.after_stop_w == 0.f);
                return r; });
            std::printf("   first decision Release(stop) on %d/%d seeds; the first decision from rest moved on %d/%d\n", *first_stop, S.n, *moves_after, S.n);
            return S;
        };
        for (const Pocket& pk : {Pocket{"stop_is_the_only_release trap", 0.10f, 0.06f, 0.05f, 0.35f},
                                 Pocket{"stop_is_the_only_release reach", 0.20f, 0.08f, 0.16f, 0.60f}})
        {
            int first_stop = 0, moves_after = 0;
            const auto S = run_pocket(pk, &first_stop, &moves_after);
            const std::string n = pk.name;
            CHECK((n + ": contacts == 0").c_str(), S.contacts == 0);
            CHECK((n + ": the first decision is Release(stop) on every seed (not Stuck)").c_str(), first_stop == S.n);
            CHECK((n + ": never a Stuck while MOVING (F2: the stop is releasable)").c_str(), S.stuck_moving == 0);
            if (n.ends_with("trap"))
            {
                // ★ F1: the from-rest Stuck is folded into the reachability belief — every seed ends Unreachable (the trapped
                //   field's, or the belief's hand back), never a terminal Stuck; the bound on the time is TRAP_VERDICT below
                auto t = S.tunreach; std::ranges::sort(t);
                const float med = t.empty() ? NAN : t[t.size() / 2], mx = t.empty() ? 1e9f : t.back();
                std::printf("   verdicts: Unreachable %d (belief hand backs %d); time to the verdict median %.2f s, max %.2f s (bound %.2f s)\n",
                            S.unreachable, S.hand_backs, med, mx, kTrapVerdictMaxS);
                CHECK((n + ": every seed ends Unreachable (F1: a Stuck from rest is folded, not a verdict)").c_str(), S.unreachable == S.n);
                CHECK((n + ": the verdict comes within the bound").c_str(), mx <= kTrapVerdictMaxS);
            }
            else
            {
                CHECK((n + ": from rest the next decision moves (no second stop, no Stuck)").c_str(), moves_after == S.n);
                CHECK((n + ": never a Stuck").c_str(), S.stuck == 0);
                CHECK((n + ": reached on every seed").c_str(), S.reached == S.n);
                performance_oracle(n, S, 5.05f);
            }
        }
    }
    // ---- off_centre_fast_doorway (F1, doorway-release spec scenario (a)): rtD1b's geometry — 0.10 m east of the gap centre,
    //      yawed 20 deg toward the east casing, 0.45 m before room_2's wall face, entering at 0.35 m/s on the live base's measured
    //      wheel deceleration (Policy.WheelDecelMps2 0.5: ~0.13 m of coast). Some decisions find even the stop unreleasable
    //      (Stuck while moving): the coast explains them, the base holds and decides again. Pre-F1 (one Stuck = a hand back) this
    //      handed the crossing back on the 7 of 20 seeds that hit one. Oracle: crosses on every seed, never handed back, no contact.
    if (want("off_centre_fast_doorway"))
    {
        const World w = door_world(1);
        Options o{.grid = kDoorGrid, .jumps = {}, .trace_name = "off_centre_fast_doorway", .v0 = 0.35f, .d_lin = 0.5f};
        const auto S = run_seeds("off_centre_fast_doorway", [&](unsigned seed) { return run_scenario(w, Pose2{V2(0.10f, -0.45f), 70.f * kDeg}, doorway(+1), 60.f, seed, o); });
        std::printf("   Stuck ticks held %d (Stuck while moving on %d seeds), belief hand backs %d\n", S.stuck_ticks, S.stuck_moving, S.hand_backs);
        CHECK("off_centre_fast_doorway: contacts == 0", S.contacts == 0);
        CHECK("off_centre_fast_doorway: never handed back (no Unreachable)", S.unreachable == 0);
        CHECK("off_centre_fast_doorway: crosses on every seed", S.reached == S.n);
        CHECK("off_centre_fast_doorway: the scene exercises the hold (a Stuck while moving on some seed)", S.stuck_moving > 0);
    }
    // ---- blocked_doorway (F1, scenario (b)): DOOR_1 with a box the MAP does not know 0.30 m past room_2's wall face, across the
    //      whole opening. The field still leads through the gap and the measured opening vetoes every trap in its corridor
    //      (Live fix 2 ruling 2), so the route-level Unreachable can never come: pre-F1 the robot jockeys at the blocked face and
    //      dithers stop / turn until the time limit on every seed (no Stuck, no hand back). The belief must hand it back.
    //      MODEL-STATED (fix round 1, I5) — in DECISIONS: at the face the prior is logit(0.01) = -4.6 (no route doubt) and the bar
    //      log(T / dt_obs) ~ log(1.6 s / 1 s) = 0.5, so the face is handed back after ~5.1 nats: two FULL counts (n_eff ~ 70,
    //      +4.3 each) at one cell — two decisions there whose eligible options, over two steps, shorten T by nothing. Claims are
    //      honest here (the policy sees the box: its plans predict stopping short of it and deliver exactly that), so they carry
    //      ~nothing. The SECONDS are how long the policy takes to park where no safe option shortens T, which the model does not
    //      predict: measured (fix round 1, 20 seeds, 240 s limit) 19/20 handed back, median 76.2 s, max 194.3 s; within 60 s: 9/20. Fix round 2 (base-rate prior, no T = inf evidence, no bar before a measured interval): 19/20, median 50.5 s, max 193.4 s. F1 fix round 3 (f438dc31): 20/20 within 240 s, median 34.4 s, max 61.3 s.
    //      Ruled acceptable (fix round 1): guard >= 19/20 within 240 s, 0 contacts, never reached (no false progress).
    if (want("blocked_doorway"))
    {
        World w = door_world(1);
        w.boxes.push_back({-0.6f, 0.30f, 0.6f, 0.40f, false});
        Options o{.grid = kDoorGrid, .jumps = {}, .trace_name = "blocked_doorway"};
        const auto S = run_seeds("blocked_doorway", [&](unsigned seed) { return run_scenario(w, Pose2{V2(0.f, -1.0f), 90.f * kDeg}, doorway(+1), 240.f, seed, o); });
        auto t = S.tverdict_after_no_escape; std::ranges::sort(t);
        auto tu = S.tunreach; std::ranges::sort(tu);
        std::printf("   hand backs %d/%d; time to the verdict median %.2f s max %.2f s; after the first no-escape decision median %.2f s max %.2f s\n",
                    S.hand_backs, S.n, tu.empty() ? NAN : tu[tu.size() / 2], tu.empty() ? NAN : tu.back(), t.empty() ? NAN : t[t.size() / 2], t.empty() ? NAN : t.back());
        CHECK("blocked_doorway: contacts == 0", S.contacts == 0);
        CHECK("blocked_doorway: every hand back is the belief's (traps are vetoed here)", S.hand_backs == S.unreachable);
        // ★ RULING (coordinator, fix round 1): a slow verdict on a truly blocked doorway is ACCEPTED — a false hand back of a passable
        //   goal costs a tour far more (doors are actuator-opened; a door that will not open is reported as door_wont_open; mission
        //   leg timeouts bound the rest). Guard on the measured distribution: >= 19/20 Unreachable within 240 s. The way to tighten
        //   it is the PARKED model term "no net progress over repeated visits to a place" (a limit cycle neither a claim nor a count
        //   can see).
        CHECK("blocked_doorway: >= 19/20 handed back within 240 s (measured distribution guard)", S.unreachable >= 19 * S.n / 20);
        CHECK("blocked_doorway: never reached (the box is real)", S.reached == 0);
    }
    // ---- heading_and_final_turn (F1 fix round 1, C1): a HEADING goal (lock-on / orient: turn 90 deg in place) and a PLACE goal
    //      whose last act is a turn (drive 1 m, then face +120 deg). Their remaining route is ~0 while the turn is owed: with the
    //      goal valued by its route alone (round 0) the bar was <= 0 and the first turn decision handed the goal back. The goal's
    //      value is its remaining TIME (route / v_max + |dyaw| / w_max). Oracle: both complete on every seed, never handed back.
    if (want("heading_and_final_turn"))
    {
        const World w = open_world();
        Options o{.grid = kOpenGrid, .jumps = {}, .trace_name = "heading_and_final_turn"};
        const Goal gh{.kind = Goal::Kind::Heading, .p = V2(0.f, 0.f), .yaw = 90.f * kDeg};
        const auto S1 = run_seeds("heading goal (turn 90 deg in place)", [&](unsigned seed) { return run_scenario(w, Pose2{V2(0.f, 0.f), 0.f}, gh, 20.f, seed, o); });
        CHECK("heading goal: contacts == 0", S1.contacts == 0);
        CHECK("heading goal: completes on every seed, never handed back", S1.reached == S1.n and S1.unreachable == 0);
        const Goal gp{.kind = Goal::Kind::Place, .p = V2(1.f, 0.f), .yaw = 120.f * kDeg, .tol_xy = 0.10f};
        const auto S2 = run_seeds("place goal with a final turn (1 m, then face 120 deg)", [&](unsigned seed) { return run_scenario(w, Pose2{V2(0.f, 0.f), 0.f}, gp, 20.f, seed, o); });
        CHECK("place + final turn: contacts == 0", S2.contacts == 0);
        CHECK("place + final turn: completes on every seed, never handed back", S2.reached == S2.n and S2.unreachable == 0);
    }
    // ---- realign_doorway (F1 fix round 1, I3): DOOR_1 with a 0.80 m clear gap (casings at +-0.40), the robot 0.25 m east of the
    //      gap centre, 0.75 m before room_2's wall face, yawed 70 deg off the normal toward the east wall: it must back up and
    //      realign (>= 4 back-up decisions on every seed) before the gap admits it. A realignment is not a failed escape: its
    //      decisions are graded by how much shortening they promise (reachability_belief.h), not counted as "no escape".
    //      Oracle: crosses on every seed, never handed back, no contact, and the scene does exercise >= 2 back-ups per seed.
    if (want("realign_doorway"))
    {
        const World w = door_world(1, 0.40f);
        Goal g = doorway(+1); g.width = 0.80f;
        Options o{.grid = kDoorGrid, .jumps = {}, .trace_name = "realign_doorway"};
        const auto S = run_seeds("realign_doorway", [&](unsigned seed) { return run_scenario(w, Pose2{V2(0.25f, -0.75f), 20.f * kDeg}, g, 60.f, seed, o); });
        std::printf("   back-up decisions: at least %d on every seed\n", S.min_backups);
        CHECK("realign_doorway: contacts == 0", S.contacts == 0);
        CHECK("realign_doorway: crosses on every seed, never handed back", S.reached == S.n and S.unreachable == 0);
        CHECK("realign_doorway: the scene needs >= 2 back-ups on every seed", S.min_backups >= 2);
    }

    // ════════════════════════════════════ APARTMENT STRESS (2026-11-10, threerooms-piso.wbt geometry) ════════════════════════════
    // The owner: "Driving a robot using the 2 lidars 3D is a long time ago solved problem" — every situation the apartment is known
    // to produce, against the same closed loop. A doorway goal's normal points the way the robot crosses.
    {
        const auto& A = apartment();
        if (A.w.segments.empty()) { std::printf("apartment: no geometry loaded (run from controller/, or APARTMENT=<file>)\n"); ++fails; }
        const auto dgoal = [](const AptDoor& d, int sgn) { return Goal{.kind = Goal::Kind::Doorway, .p = d.c, .normal = float(sgn) * d.n, .width = d.width}; };
        Options base_o{.grid = kAptGrid, .jumps = {}, .trace_name = ""};
        // (1) clean38: room_1, the robot 6 m south of DOOR_0 facing AWAY from it (DOOR_0's n points INTO room_1, so north is -n); the cross goal on a PREDICTED anchor (jitter 0.15 m)
        for (const float jit : {0.f, 0.15f})
        {
            const std::string n = jit > 0.f ? "apt_far_doorway_jitter" : "apt_far_doorway";
            if (not want(n.c_str())) continue;
            Options o = base_o; o.goal_jitter_m = jit; o.trace_name = n;
            const auto& d = apt_door("DOOR_0");
            const auto S = run_seeds(n, [&](unsigned seed) { return run_scenario(A.w, Pose2{V2(-2.5f, 1.2f), -90.f * kDeg}, dgoal(d, -1), 90.f, seed, o); });
            apartment_oracles(n, S);
        }
        // (2) every door, both ways, from 2.5 m before it, a random heading; the anchor jitters 0.10 m
        for (const char* dn : {"DOOR_0", "DOOR_1", "DOOR_2"})
            for (const int sgn : {+1, -1})
            {
                const std::string n = std::string("apt_cross_") + dn + (sgn > 0 ? "_pos" : "_neg");
                if (not want(n.c_str())) continue;
                Options o = base_o; o.goal_jitter_m = 0.10f; o.trace_name = n;
                const auto& d = apt_door(dn);
                const auto S = run_seeds(n, [&](unsigned seed) {
                    std::mt19937 g(seed * 31u + 7u);
                    const float yaw = std::uniform_real_distribution<float>(-kPi, kPi)(g);
                    // 2.5 m before the door, except DOOR_0 from the north: that pocket is only ~2 m deep (room_3's wall at y ~9.3),
                    // so 2.5 m started the robot in ANOTHER room (measured: a 6 m detour read as a crossing failure)
                    const float back = (std::string(dn) == "DOOR_0" and sgn > 0) ? 1.4f : 2.5f;
                    return run_scenario(A.w, Pose2{d.c - float(sgn) * back * d.n, yaw}, dgoal(d, sgn), 60.f, seed, o); });
                apartment_oracles(n, S);
            }
        // (3) random place goals within each room, random start and heading (true clearance >= 0.6 m at both ends)
        for (const auto& R : kAptRooms)
        {
            const std::string n = std::string("apt_place_") + R.name;
            if (not want(n.c_str())) continue;
            Options o = base_o; o.trace_name = n;
            const auto S = run_seeds(n, [&](unsigned seed) {
                std::mt19937 g(seed * 101u + 3u);
                const Pose2 s = apt_sample(A.w, R.r, 0.6f, g), e = apt_sample(A.w, R.r, 0.6f, g);
                return run_scenario(A.w, s, Goal{.kind = Goal::Kind::Place, .p = e.p, .yaw = {}, .tol_xy = 0.15f}, 60.f, seed, o); });
            apartment_oracles(n, S);
        }
        // (4) the goal 3 m straight BEHIND the robot, in open space, in each room
        for (const auto& R : kAptRooms)
        {
            const std::string n = std::string("apt_goal_behind_") + R.name;
            if (not want(n.c_str())) continue;
            Options o = base_o; o.trace_name = n;
            const auto S = run_seeds(n, [&](unsigned seed) {
                std::mt19937 g(seed * 17u + 5u);
                for (int k = 0; k < 1000; ++k)
                {
                    const Pose2 s = apt_sample(A.w, R.r, 0.8f, g);
                    const V2 e = s.p - 3.f * V2(std::cos(s.yaw), std::sin(s.yaw));
                    // the goal in the SAME room, the straight line to it clear (seed 0 of room_2 put it beyond a wall: geodesic inf)
                    const bool in_room = e.x() >= R.r.x0 and e.x() <= R.r.x1 and e.y() >= R.r.y0 and e.y() <= R.r.y1;
                    bool line_clear = in_room;
                    for (int q = 0; q <= 6 and line_clear; ++q) line_clear = A.w.clearance(s.p + (float(q) / 6.f) * (e - s.p), false) >= 0.6f;
                    if (line_clear)
                        return run_scenario(A.w, s, Goal{.kind = Goal::Kind::Place, .p = e, .yaw = {}, .tol_xy = 0.15f}, 40.f, seed, o);
                }
                return ScenarioResult{}; });
            apartment_oracles(n, S);
        }
        // (5) room to room: room_1 -> room_2 -> room_3 centres as PLACE goals through the open doors (the long traverse)
        if (want("apt_traverse_r1_r3"))
        {
            Options o = base_o; o.trace_name = "apt_traverse_r1_r3";
            const auto S = run_seeds("apt_traverse_r1_r3", [&](unsigned seed) {
                std::mt19937 g(seed * 13u + 1u);
                const Pose2 s = apt_sample(A.w, kAptRooms[0].r, 0.6f, g);
                return run_scenario(A.w, s, Goal{.kind = Goal::Kind::Place, .p = V2(-5.0f, 11.5f), .yaw = {}, .tol_xy = 0.20f}, 120.f, seed, o); });
            apartment_oracles("apt_traverse_r1_r3", S);
        }
        // (6) start 8 cm from a wall, FACING it; the goal 4 m behind (room_1's south wall)
        if (want("apt_facing_wall"))
        {
            Options o = base_o; o.trace_name = "apt_facing_wall";
            const float y = 0.08f + 0.230f + 0.12f;   // the south wall's inner face is about y 0.12 near x -6.5 (checked below)
            const Pose2 s{V2(-6.5f, y), -90.f * kDeg};
            std::printf("apt_facing_wall start clearance %.3f m\n", sim2d::ContactTester(A.w, body()).clearance(s));
            const auto S = run_seeds("apt_facing_wall", [&](unsigned seed) { return run_scenario(A.w, s, Goal{.kind = Goal::Kind::Place, .p = V2(-6.5f, 4.5f), .yaw = {}, .tol_xy = 0.15f}, 40.f, seed, o); });
            apartment_oracles("apt_facing_wall", S);
        }
        // (7) an UNMAPPED chair (0.5 x 0.5 m) on the straight line of a room_1 traverse
        // LIVE FAILURE 1 (language-agent chat_e2e2, policy_field_dump_1794309704136): "go to" a remembered chair — the target
        // is the chair's own centre, which no body can stand on. Today: field +inf everywhere, Release backs ~2 m, no_route.
        // The design (§3.1): the route ends at the reachable place nearest the target.
        if (want("apt_goal_in_object"))
        {
            World w = A.w; w.boxes.push_back({-4.25f, 2.75f, -3.75f, 3.25f, true});   // a MAPPED chair, 0.5 m square
            Options o = base_o; o.trace_name = "apt_goal_in_object";
            const auto S = run_seeds("apt_goal_in_object", [&](unsigned seed) { return run_scenario(w, Pose2{V2(-5.6f, 3.9f), -30.f * kDeg}, Goal{.kind = Goal::Kind::Place, .p = V2(-4.0f, 3.0f), .yaw = {}, .tol_xy = 0.15f}, 40.f, seed, o); });
            apartment_oracles("apt_goal_in_object", S);
        }
        // LIVE FAILURE 2 (clean38 class, policy_field_dump_1794309020589): the policy's OWN free map cuts the robot's region off
        // the doorway (live: a pinch 0.21 m from the map's obstacles under its inscribed-radius test, the robot 0.77 m clear,
        // field +inf at the robot -> Release -> 8 m reversal in open space). Here: a ring of residual HARD cells 1 m before
        // DOOR_0, wall to wall, that the world does not have; the planner does not hold it (as in the dump: planner distance
        // 0.21 m at the cut).
        if (want("apt_phantom_strip"))
        {
            const auto& d = apt_door("DOOR_0");
            Options o = base_o; o.trace_name = "apt_phantom_strip";
            for (float a = 0.f; a <= kPi + 1e-3f; a += 0.04f)
            {
                const V2 c = d.c + 1.0f * (std::cos(a) * d.tng + std::sin(a) * d.n);
                o.res_cells.push_back(c); o.res_risk.push_back(std::numeric_limits<float>::infinity());
            }
            const auto S = run_seeds("apt_phantom_strip", [&](unsigned seed) { return run_scenario(A.w, Pose2{V2(-3.5f, 4.0f), 90.f * kDeg}, dgoal(d, -1), 60.f, seed, o); });
            apartment_oracles("apt_phantom_strip", S);
        }
        // ==== EDGE SUITE (2026-11-11): situations the controller had not met. Hybrid regime (movers exist only there). ====
        const auto must_end_cleanly = [](const std::string& n, const Summary& S) {
            CHECK((n + ": contacts == 0").c_str(), S.contacts == 0);
            CHECK((n + ": every seed ends (reached, or declared unreachable), never the time limit").c_str(), S.reached + S.unreachable == S.n);
            CHECK((n + ": no window > 20 s without field decrease (no thrash)").c_str(), S.max_np <= 20.f);
            CHECK((n + ": open-space reverse travel <= 0.5 m per run").c_str(), S.max_open_reverse <= 0.5f); };
        const Pose2 r1_w{V2(-6.5f, 3.0f), 0.f};
        // (E1) a person crossing the room_1 traverse at 1 m/s, reaching the route as the robot does
        if (want("apt_person_crossing"))
        {
            Options o = base_o; o.trace_name = "apt_person_crossing";
            o.movers = {{V2(0.2f, 0.2f), {{0.f, V2(-4.0f, 0.9f)}, {2.5f, V2(-4.0f, 0.9f)}, {7.1f, V2(-4.0f, 5.5f)}}}};
            const auto S = run_seeds("apt_person_crossing", [&](unsigned seed) { return run_scenario(A.w, r1_w, Goal{.kind = Goal::Kind::Place, .p = V2(-1.5f, 3.0f), .yaw = {}, .tol_xy = 0.15f}, 45.f, seed, o); });
            apartment_oracles("apt_person_crossing", S);
        }
        // (E2) a person walking head-on along the route at 0.8 m/s; it stops in front of the robot (never into it) and waits
        if (want("apt_person_headon"))
        {
            Options o = base_o; o.trace_name = "apt_person_headon";
            o.movers = {{V2(0.2f, 0.2f), {{0.f, V2(-1.6f, 3.0f)}, {6.f, V2(-6.4f, 3.0f)}, {7.f, V2(-7.4f, 3.0f)}}}};
            const auto S = run_seeds("apt_person_headon", [&](unsigned seed) { return run_scenario(A.w, r1_w, Goal{.kind = Goal::Kind::Place, .p = V2(-1.5f, 3.0f), .yaw = {}, .tol_xy = 0.15f}, 60.f, seed, o); });
            apartment_oracles("apt_person_headon", S);
        }
        // (E3) a person standing IN DOOR_0 for 8 s, then leaving through it and stepping aside
        if (want("apt_person_in_door"))
        {
            const auto& d = apt_door("DOOR_0");
            Options o = base_o; o.trace_name = "apt_person_in_door";
            o.movers = {{V2(0.2f, 0.2f), {{0.f, d.c}, {8.f, d.c}, {9.5f, V2(d.c - 1.2f * d.n)}, {11.f, V2(d.c - 1.2f * d.n + 1.0f * d.tng)}}}};
            const auto S = run_seeds("apt_person_in_door", [&](unsigned seed) { return run_scenario(A.w, Pose2{V2(-3.5f, 4.8f), 90.f * kDeg}, dgoal(d, -1), 60.f, seed, o); });
            // the robot WAITS for the person (owner 2026-11-11), or takes another way the planner finds: no-progress and
            // retreat oracles do not apply; a bump, a failure to arrive, or reversing in open space do
            CHECK("apt_person_in_door: contacts == 0", S.contacts == 0);
            CHECK("apt_person_in_door: reached on every seed", S.reached == S.n);
            CHECK("apt_person_in_door: open-space reverse travel <= 0.5 m per run", S.max_open_reverse <= 0.5f);
        }
        // (E15) THE WAIT/RE-MARK CYCLE (2026-11-11): a person standing IN DOOR_0 for 20 s whose legs the LiDAR sees through
        //       30 % of the time in ~0.3 s gaps, then leaving. One see-through sweep must not send the robot off (stop-go):
        //       one wait, one exit, no bump, arrival.
        if (want("apt_person_flicker"))
        {
            const auto& d = apt_door("DOOR_0");
            Options o = base_o; o.trace_name = "apt_person_flicker";
            o.movers = {{V2(0.2f, 0.2f), {{0.f, d.c}, {20.f, d.c}, {21.5f, V2(d.c - 1.2f * d.n)}, {23.f, V2(d.c - 1.2f * d.n + 1.0f * d.tng)}}, 0.3f, 0.3f}};
            const auto S = run_seeds("apt_person_flicker", [&](unsigned seed) { return run_scenario(A.w, Pose2{V2(-3.5f, 4.8f), 90.f * kDeg}, dgoal(d, -1), 60.f, seed, o); });
            CHECK("apt_person_flicker: contacts == 0", S.contacts == 0);
            CHECK("apt_person_flicker: reached on every seed", S.reached == S.n);
            CHECK("apt_person_flicker: no stop-go (at most one wait per run)", S.max_waits <= 1);
        }
        // (E4/E5) an UNMAPPED barrier across room_1 at x -5 with one gap: 0.70 m (the body is 0.568 m wide: 6.6 cm a side) and
        //         0.50 m (the body cannot pass; room_1's east part has no other way in from the west: unreachable)
        for (const float gap : {0.70f, 0.50f})
        {
            const std::string n = gap > 0.6f ? "apt_gap_tight" : "apt_gap_closed";
            if (not want(n.c_str())) continue;
            World w = A.w;
            w.boxes.push_back({-5.15f, -0.5f, -4.85f, 3.0f - 0.5f * gap, false});
            w.boxes.push_back({-5.15f, 3.0f + 0.5f * gap, -4.85f, 7.0f, false});
            Options o = base_o; o.trace_name = n; o.truth_yardstick = true;
            // 120 s: the honest way round is the ring through rooms 2 and 3 (~23 m: a 0.65 m squeeze between the barrier's end
            // and the partition at x -4.2 closes the short one); 60 s cut seeds off mid-detour
            const auto S = run_seeds(n, [&](unsigned seed) { return run_scenario(w, r1_w, Goal{.kind = Goal::Kind::Place, .p = V2(-3.0f, 3.0f), .yaw = {}, .tol_xy = 0.15f}, 120.f, seed, o); });
            if (gap > 0.6f) apartment_oracles(n, S); else must_end_cleanly(n, S);
        }
        // (E6) an UNMAPPED slalom: three panels force an S through room_1
        if (want("apt_slalom"))
        {
            World w = A.w;
            w.boxes.push_back({-6.2f, 1.9f, -5.8f, 4.1f, false});
            w.boxes.push_back({-4.9f, 2.2f, -4.5f, 6.9f, false});
            w.boxes.push_back({-3.6f, -0.2f, -3.2f, 3.8f, false});
            Options o = base_o; o.trace_name = "apt_slalom"; o.truth_yardstick = true;
            const auto S = run_seeds("apt_slalom", [&](unsigned seed) { return run_scenario(w, Pose2{V2(-7.3f, 3.0f), 0.f}, Goal{.kind = Goal::Kind::Place, .p = V2(-1.8f, 3.0f), .yaw = {}, .tol_xy = 0.15f}, 90.f, seed, o); });
            apartment_oracles("apt_slalom", S);
        }
        // (E7) MAP/WORLD MISMATCH in open floor (room_1 west): a mapped 1 x 1 m table is really 0.5 m south of where the map
        //      has it; the route the planner lays past the map's table runs into the real one, with 1.5 m of free floor south
        //      (first version sat beside a stub wall: the real table left a 0.65 m channel and the 21 m detour was honest)
        if (want("apt_furniture_shifted"))
        {
            World w = A.w;
            sim2d::Box ghost{-6.9f, 3.6f, -5.9f, 4.6f, true}; ghost.in_world = false;
            w.boxes.push_back(ghost);
            w.boxes.push_back({-6.9f, 3.1f, -5.9f, 4.1f, false});
            Options o = base_o; o.trace_name = "apt_furniture_shifted"; o.truth_yardstick = true;
            const auto S = run_seeds("apt_furniture_shifted", [&](unsigned seed) { return run_scenario(w, Pose2{V2(-7.6f, 3.0f), 0.f}, Goal{.kind = Goal::Kind::Place, .p = V2(-5.2f, 3.0f), .yaw = {}, .tol_xy = 0.15f}, 60.f, seed, o); });
            apartment_oracles("apt_furniture_shifted", S);
        }
        // (E8) a goal 0.22 m from room_1's south wall: the body cannot be centred there (goal region)
        if (want("apt_goal_near_wall"))
        {
            Options o = base_o; o.trace_name = "apt_goal_near_wall";
            const auto S = run_seeds("apt_goal_near_wall", [&](unsigned seed) { return run_scenario(A.w, Pose2{V2(-6.0f, 3.0f), -90.f * kDeg}, Goal{.kind = Goal::Kind::Place, .p = V2(-5.0f, 0.30f), .yaw = {}, .tol_xy = 0.15f}, 40.f, seed, o); });
            apartment_oracles("apt_goal_near_wall", S);
        }
        // (E9) a goal 0.35 m straight BEHIND the robot
        if (want("apt_goal_behind_close"))
        {
            Options o = base_o; o.trace_name = "apt_goal_behind_close";
            const auto S = run_seeds("apt_goal_behind_close", [&](unsigned seed) { return run_scenario(A.w, Pose2{V2(-5.0f, 3.0f), 0.f}, Goal{.kind = Goal::Kind::Place, .p = V2(-5.35f, 3.0f), .yaw = {}, .tol_xy = 0.10f}, 30.f, seed, o); });
            apartment_oracles("apt_goal_behind_close", S);
        }
        // (E10) a 2 s LiDAR stall and (E11) a 0.13 m / 4 deg relocalisation jump, both WHILE crossing DOOR_0
        if (want("apt_door_dropout"))
        {
            Options o = base_o; o.lidar_dropout = {{2.5f, 4.5f}}; o.trace_name = "apt_door_dropout";
            const auto S = run_seeds("apt_door_dropout", [&](unsigned seed) { return run_scenario(A.w, Pose2{V2(-3.5f, 4.8f), 90.f * kDeg}, dgoal(apt_door("DOOR_0"), -1), 60.f, seed, o); });
            apartment_oracles("apt_door_dropout", S);
        }
        if (want("apt_door_jump"))
        {
            Options o = base_o; o.jumps = {{3.0f, V2(0.12f, 0.05f), 4.f * kDeg}}; o.trace_name = "apt_door_jump";
            const auto S = run_seeds("apt_door_jump", [&](unsigned seed) { return run_scenario(A.w, Pose2{V2(-3.5f, 4.8f), 90.f * kDeg}, dgoal(apt_door("DOOR_0"), -1), 60.f, seed, o); });
            apartment_oracles("apt_door_jump", S);
        }
        // (E13) A PARKED-COVARIANCE LIE (Fable round 2 #1): the pose is 5 cm / 1.5 deg off the whole run while the localiser
        //       reports sigma 0 (a live covariance collapses when parked). Every wall is seen a few cm off the map; none may
        //       become a mark, a wait or an "unreachable". The room_1 -> room_3 traverse, through two doors.
        if (want("apt_parked_offset"))
        {
            Options o = base_o; o.trace_name = "apt_parked_offset"; o.loc_sigma_reported = 0.f;
            o.jumps = {{0.f, V2(0.04f, -0.03f), 1.5f * kDeg}};
            const auto S = run_seeds("apt_parked_offset", [&](unsigned seed) { return run_scenario(A.w, Pose2{V2(-6.0f, 3.0f), 45.f * kDeg}, Goal{.kind = Goal::Kind::Place, .p = V2(-5.0f, 11.5f), .yaw = {}, .tol_xy = 0.20f}, 120.f, seed, o); });
            apartment_oracles("apt_parked_offset", S);
        }
        // (E14) MARKS UNDER A RELOCALISATION (Fable round 2 #2): the slalom (unmapped panels -> marks), then at 15 s the pose
        //       jumps 0.2 m / 4 deg. Marks anchored in the room frame would sit 0.2 m off their panels as phantoms.
        if (want("apt_marks_jump"))
        {
            World w = A.w;
            w.boxes.push_back({-6.2f, 1.9f, -5.8f, 4.1f, false});
            w.boxes.push_back({-4.9f, 2.2f, -4.5f, 6.9f, false});
            w.boxes.push_back({-3.6f, -0.2f, -3.2f, 3.8f, false});
            Options o = base_o; o.trace_name = "apt_marks_jump"; o.truth_yardstick = true;
            o.jumps = {{15.f, V2(0.15f, -0.13f), 4.f * kDeg}};
            const auto S = run_seeds("apt_marks_jump", [&](unsigned seed) { return run_scenario(w, Pose2{V2(-7.3f, 3.0f), 0.f}, Goal{.kind = Goal::Kind::Place, .p = V2(-1.8f, 3.0f), .yaw = {}, .tol_xy = 0.15f}, 120.f, seed, o); });
            apartment_oracles("apt_marks_jump", S);
        }
        // (E12) the room_1 -> room_3 traverse with DOOR_1 blocked (unmapped): the way round is through room_2 and DOOR_2
        if (want("apt_traverse_door1_blocked"))
        {
            const auto& d = apt_door("DOOR_1");
            World w = A.w; const V2 a = d.c - 0.5f * d.width * d.tng, b = d.c + 0.5f * d.width * d.tng;
            w.boxes.push_back({std::min(a.x(), b.x()) - 0.02f, std::min(a.y(), b.y()) - 0.02f, std::max(a.x(), b.x()) + 0.02f, std::max(a.y(), b.y()) + 0.02f, false});
            Options o = base_o; o.trace_name = "apt_traverse_door1_blocked"; o.truth_yardstick = true;
            const auto S = run_seeds("apt_traverse_door1_blocked", [&](unsigned seed) { return run_scenario(w, Pose2{V2(-6.0f, 3.0f), 45.f * kDeg}, Goal{.kind = Goal::Kind::Place, .p = V2(-5.0f, 11.5f), .yaw = {}, .tol_xy = 0.20f}, 150.f, seed, o); });
            must_end_cleanly("apt_traverse_door1_blocked", S);
        }
        if (want("apt_unmapped_chair"))
        {
            World w = A.w; w.boxes.push_back({-4.25f, 2.75f, -3.75f, 3.25f, false});
            Options o = base_o; o.trace_name = "apt_unmapped_chair";
            const auto S = run_seeds("apt_unmapped_chair", [&](unsigned seed) { return run_scenario(w, Pose2{V2(-6.5f, 3.0f), 0.f}, Goal{.kind = Goal::Kind::Place, .p = V2(-1.5f, 3.0f), .yaw = {}, .tol_xy = 0.15f}, 60.f, seed, o); });
            apartment_oracles("apt_unmapped_chair", S);
        }
        // (8) DOOR_2's leaf open 90 deg toward the approaching robot (room_3 side, x < -1.7), crossing into room_2
        if (want("apt_leaf_toward"))
        {
            const auto& d = apt_door("DOOR_2");
            const World w = with_leaf(A.w, d, +1, +1, true);
            Options o = base_o; o.goal_jitter_m = 0.10f; o.trace_name = "apt_leaf_toward";
            const auto S = run_seeds("apt_leaf_toward", [&](unsigned seed) { return run_scenario(w, Pose2{d.c + 2.5f * d.n + V2(0.f, -0.6f), 180.f * kDeg}, dgoal(d, -1), 60.f, seed, o); });
            apartment_oracles("apt_leaf_toward", S);
        }
        // (9) a 1.5 s LiDAR stall in the middle of the far-doorway traverse
        if (want("apt_lidar_stall"))
        {
            Options o = base_o; o.lidar_dropout = {{4.f, 5.5f}}; o.trace_name = "apt_lidar_stall";
            const auto S = run_seeds("apt_lidar_stall", [&](unsigned seed) { return run_scenario(A.w, Pose2{V2(-2.5f, 1.2f), 90.f * kDeg}, dgoal(apt_door("DOOR_0"), -1), 90.f, seed, o); });
            apartment_oracles("apt_lidar_stall", S);
        }
        // (10) a relocalisation jump (0.3 m, 8 deg) and (11) OU pose drift, on the room_1 -> room_3 traverse
        if (want("apt_pose_jump"))
        {
            Options o = base_o; o.jumps = {{6.f, V2(0.3f, -0.1f), 8.f * kDeg}}; o.trace_name = "apt_pose_jump";
            const auto S = run_seeds("apt_pose_jump", [&](unsigned seed) { return run_scenario(A.w, Pose2{V2(-6.0f, 3.0f), 45.f * kDeg}, Goal{.kind = Goal::Kind::Place, .p = V2(-5.0f, 11.5f), .yaw = {}, .tol_xy = 0.20f}, 120.f, seed, o); });
            apartment_oracles("apt_pose_jump", S);
        }
        if (want("apt_ou_drift"))
        {
            // the localiser's REALISTIC drift for this robot (centimetres; 0.05 m / 2 deg, tau 20 s) — the catalogue's 0.5 m / 13 deg
            // puts every 0.9 m doorway of the believed map beside the real one, which no map-based driver can thread
            Options o = base_o; o.ou = true; o.ou_sigma_xy = 0.05f; o.ou_sigma_yaw = 2.f * kDeg; o.trace_name = "apt_ou_drift";
            const auto S = run_seeds("apt_ou_drift", [&](unsigned seed) { return run_scenario(A.w, Pose2{V2(-6.0f, 3.0f), 45.f * kDeg}, Goal{.kind = Goal::Kind::Place, .p = V2(-5.0f, 11.5f), .yaw = {}, .tol_xy = 0.20f}, 120.f, seed, o); });
            apartment_oracles("apt_ou_drift", S);
        }
        // (12) a SHUT door: DOOR_0's leaf across the opening — unreachable; it must say so, never wander or bump
        if (want("apt_shut_door"))
        {
            const auto& d = apt_door("DOOR_0");
            World w = A.w; const V2 a = d.c - 0.5f * d.width * d.tng, b = d.c + 0.5f * d.width * d.tng;
            w.boxes.push_back({std::min(a.x(), b.x()) - 0.02f, std::min(a.y(), b.y()) - 0.02f, std::max(a.x(), b.x()) + 0.02f, std::max(a.y(), b.y()) + 0.02f, false});
            Options o = base_o; o.trace_name = "apt_shut_door"; o.truth_yardstick = true;   // progress on the world with the door SHUT
            const auto S = run_seeds("apt_shut_door", [&](unsigned seed) { return run_scenario(w, Pose2{V2(-3.5f, 4.0f), 90.f * kDeg}, dgoal(d, -1), 120.f, seed, o); });
            // The far side of DOOR_0 is ALSO reachable around it (the pocket opens east into room_2's L): a detour is an honest
            // outcome, and so is "unreachable". What must not happen is a bump or a thrash.
            CHECK("apt_shut_door: contacts == 0", S.contacts == 0);
            CHECK("apt_shut_door: every seed ends (reached by a detour, or declared unreachable), never the time limit", S.reached + S.unreachable == S.n);
            CHECK("apt_shut_door: no window > 20 s without field decrease (no thrash)", S.max_np <= 20.f);
            CHECK("apt_shut_door: open-space reverse travel <= 0.5 m per run", S.max_open_reverse <= 0.5f);
        }
    }
    std::printf("REVERSE BREAKDOWN (of %ld reversing decisions): release/unlock %ld | dead end/stall (no forward progress) %ld | OPEN (forward alternative, no obstacle within 0.25 m; should be ~0) %ld | tight (forward alternative, obstacle within 0.25 m) %ld ;  true clearance < 0.25 m: %ld, >= 0.25 m: %ld\n", g_rev, g_rel, g_rk1, g_rk0, g_rk2, g_rtight, g_rclear);
    std::printf("REVERSE SHARE: %ld of %ld Drive/Release decisions (%.1f%%) camera=%s\n", g_rev, g_moves, g_moves ? 100.0 * double(g_rev) / double(g_moves) : 0.0, (std::getenv("POLICY_CAMERA") == nullptr or std::string(std::getenv("POLICY_CAMERA")) != "0") ? "on" : "off");
    for (auto& P : g_prof)
        if (P.n > 0)
        {
            std::ranges::sort(P.all);
            std::printf("PROFILE %-40s calls %7ld  mean %7.2f ms  p99 %7.2f ms  max %8.2f ms\n", P.name, P.n, P.sum / double(P.n), P.all[std::size_t(0.99 * double(P.all.size() - 1))], P.max);
        }
    std::printf("%s (%d failing)\n", fails == 0 ? "ALL PASS" : "FAILURES", fails);
    return fails == 0 ? 0 : 1;
}
