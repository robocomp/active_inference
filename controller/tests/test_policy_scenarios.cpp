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

#include <charconv>
#include <chrono>
#include <deque>
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
};

// ★ Task 8 ruling I3 — ONE definition of "through": the PROTOCOL's (the controller's rc::cross_progress_complete: s >= the
//   standoff along the gap normal). decide() never reports Arrived for a Doorway; the driver checks this predicate every
//   tick, and the field's goal line lies 2 tol_xy BEYOND it (real slack, as the controller's goal_adapter places it).
// 0.35 m = where the harness's crossings completed before I3 (field within tol_xy 0.25 of a goal line 0.6 m past the gap), so
// the reviewed time-to-goal baselines measure the same completion point.
constexpr float kThroughS = 0.35f;
static bool doorway_through(const Goal& g, const V2& p) { return (p - g.p).dot(g.normal.normalized()) >= kThroughS; }
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
    MapFree(const World& w, const Bounds& b, std::span<const V2> hard_cells = {}, float cell = 0.05f)
    {
        grid = {V2(b.x0, b.y0), 0.05f, int(std::lround((b.x1 - b.x0) / 0.05f)) + 1, int(std::lround((b.y1 - b.y0) / 0.05f)) + 1};
        free_.assign(size_t(grid.w * grid.h), 0);
        const float r_ins = body().inscribed_radius();
        for (int j = 0; j < grid.h; ++j)
            for (int i = 0; i < grid.w; ++i)
            {
                const V2 q = grid.origin + V2(i * grid.cell, j * grid.cell);
                bool f = w.clearance(q, true) >= r_ins;
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
static ScenarioResult run_scenario(const World& world, Pose2 start, Goal goal, float time_limit_s, unsigned seed, const Options& opt)
{
    ScenarioResult res;
    std::mt19937 rng(seed * 7919u + 13u);
    const bool tr = trace_on(opt.trace_name, seed);
    const auto& B = body();
    sim2d::ContactTester contact(world, B);
    sim2d::Base base; base.lim = limits(); base.latency_s = 0.06f;
    if (opt.d_lin > 0.f) base.lim.d_lin = opt.d_lin;
    base.rng = &rng; base.jitter_s = kLatencyJitterS; base.slip_sigma = kSlipSigma;
    base.deadband_scale = std::uniform_real_distribution<float>(1.f - kDeadbandSpread, 1.f + kDeadbandSpread)(rng);
    base.v = base.cmd_v = opt.v0; base.w = base.cmd_w = opt.w0;
    std::normal_distribution<float> slip(0.f, kSlipSigma);
    sim2d::PoseError err; err.on = opt.ou; err.pivot = start.p;
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
    nav.set_goal(goal, mapfree.grid, [&mapfree](const V2& c) { return mapfree(c); },
                 [jambs, &opt, r_ins = B.inscribed_radius()](const V2& c)
                 {
                     // residual's soft cells under the inscribed disc (the planner's price), then the door's factor
                     float soft = 0.f;
                     for (std::size_t i = 0; i < opt.res_cells.size() and i < opt.res_risk.size(); ++i)
                         if (std::isfinite(opt.res_risk[i]) and (opt.res_cells[i] - c).norm() - 0.025f <= r_ins) soft += opt.res_risk[i];
                     return (jambs ? doorway_risk_density(*jambs, c) : 0.f) + soft + door_leaf_density(opt.leaf, c);
                 }, goal_points(goal),
                 B.circumscribed_radius());
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
        if (opt.unseen) for (const auto& h : unseen.hypotheses(bel, odom)) ret.push_back(h);   // odometry -> room frame

        PolicyInput in;
        in.pose = bel; in.v_now = base.v; in.w_now = base.w;
        in.sigma_rel_xy = 0.01f; in.sigma_rel_yaw = 0.01f;                                       // odometry-scale RELATIVE error
        if (std::getenv("POLICY_CAMERA") == nullptr or std::string(std::getenv("POLICY_CAMERA")) != "0") { in.camera_half_fov_rad = 0.96f; in.affordance_facing = not std::getenv("POLICY_NOFACING"); }   // Task R: ZED2i 110 deg HFOV
        if (opt.ou) { in.pose_drift_sigma_m = err.sigma_xy; in.pose_drift_time_s = err.tau; }   // the OU's own process (monitor)
        in.returns_age_s = float(kBufferCycles - 1) * kTick;                                     // the older sweep's age
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
        if (d.mode == Decision::Mode::Blind) { res.stuck = true; res.why = d.why; break; }
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
                if (d.mode == Decision::Mode::Release) ++g_rel;
                else if (d.reversal == 1) ++g_rk1;
                else if (d.reversal == 2) ++g_rk2;
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
            best_f = nav.field().at(bel.p);      // the trapped field changed: progress is measured on the new one (clock kept)
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

        // ★ TRAJECTORY ORACLE (review H3): once the true centre has left a trap's disc it must never re-enter it
        for (auto& tt : true_traps)
        {
            const float dist = (truth.p - tt.c).norm();
            if (dist > tt.r) tt.left = true;
            else if (tt.left) res.trajectory_ok = false;
        }

        const float fn = nav.field().at(err.believed(truth).p);
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
                 bool traj_ok = true; int total_traps = 0, total_stops = 0, stuck_moving = 0; std::vector<float> tunreach, tstuck; float max_np = 0.f, max_path_ratio = 0.f, min_clear = 1e9f; std::vector<float> tgoal; };

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
        performance_oracle("tour3_1_leg3", S, 2.75f);
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
            performance_oracle(n, S, leaf == 1 ? 4.60f : 4.60f, 5.75f);
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
    std::printf("REVERSE BREAKDOWN (of %ld reversing decisions): release/unlock %ld | drive, no seen-progress plan (dead end/stall) %ld | drive, outscored an alternative (tight) %ld | OTHER %ld ;  true clearance < 0.25 m: %ld, >= 0.25 m: %ld\n", g_rev, g_rel, g_rk1, g_rk2, g_rk0, g_rtight, g_rclear);
    std::printf("REVERSE SHARE: %ld of %ld Drive/Release decisions (%.1f%%) camera=%s\n", g_rev, g_moves, g_moves ? 100.0 * double(g_rev) / double(g_moves) : 0.0, (std::getenv("POLICY_CAMERA") == nullptr or std::string(std::getenv("POLICY_CAMERA")) != "0") ? "on" : "off");
    std::printf("%s (%d failing)\n", fails == 0 ? "ALL PASS" : "FAILURES", fails);
    return fails == 0 ? 0 : 1;
}
