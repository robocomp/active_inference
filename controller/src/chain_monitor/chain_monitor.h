#pragma once
// THE CHAIN'S MODEL-BASED MONITORS (controller-two-regimes; mock: tests/test_policy_scenarios.cpp run_chain, 2026-11-11).
// ONE implementation for the 2-D mock and the live controller_session, so the bench runs the agent's own code.
//
//   window_test         P(no contact) of the body along the route ahead (+ the turn in place onto it), on LIVE returns,
//                       with the chain's lateral tracking error sigma_track; fails where one segment breaks the chance
//                       constraint P(contact) <= risk_eps.
//   stretch_blocked_at  connectivity of the stretch over (route s x lateral d): impassable -> the planner re-plans first.
//   UnmappedTest        a return the map does not explain, as a chance statement (cell + k_eps sigma_e).
//   MarkSet             controller-local marks (unmapped obstacles in the planner's world) with a BELIEF that the obstacle
//                       is still there, failure marks that lapse, and the carry on relocalisation steps.
//   ProgressMonitor     predicted vs realised arc length over the tracker's own horizon (replaces the stall timers).
#include <deque>
#include <functional>
#include <span>
#include <vector>
#include <Eigen/Dense>
#include "../route_spline.h"
#include "../grid_planner.h"
#include "../local_policy/policy_types.h"
#include "../../../common/robot_footprint/robot_footprint.h"

namespace rc::chain
{
using V2 = Eigen::Vector2f;
using rc::policy::Pose2;
using rc::policy::Return;

// k such that P(|z| > k) = risk_eps for a standard normal z (the chance constraint's own two-sided quantile)
float two_sided_quantile(float risk_eps);

// "Explained by the map" as a chance statement (Fable round 2 #1): a return at distance d from the planner's obstacles is
// unmapped when d > cell + k_eps sqrt(sigma_r^2 + sigma_loc^2 + sigma_map^2). sigma_map keeps the test from collapsing
// when a parked robot's pose covariance does. ★FLAGGED: sigma_map (live: room_concept's published layout sigma).
struct UnmappedTest
{
    const GridPlanner* gp = nullptr;
    float k_eps = 2.576f, sigma_r = 0.02f, sigma_map = 0.02f;
    bool operator()(const V2& q, float sigma_loc) const;
};

struct WindowResult
{
    std::vector<Pose2> poses;   // the pivot onto the route at the robot, then the route every 0.10 m
    std::size_t n_pivot = 0;    // poses of the pivot (the route part starts after them)
    int first_bad = -1;         // first failing segment (index into poses), -1 = the window is clear
    float lp_worst = 0.f;       // worst segment log P(no contact)
};
// window over [s_prog, s_prog + window_m]; each segment judged in its own frame with (sigma_track, sigma_track_yaw)
WindowResult window_test(const rc::RobotFootprint& body, const RouteSpline& spline, float s_prog, const Pose2& bel,
                         std::span<const Return> cells, float sigma_track, float sigma_track_yaw, float risk_eps,
                         float window_m = 4.0f);
// arc length of the window's first failing pose (0 if clear)
float window_s_bad(const WindowResult& w, float s_prog);

// CONNECTIVITY over (route position s, lateral offset d) on [s0, s1]: -1 = passable, else the arc length where the furthest
// reachable section ends. sigma_loc_at: the localiser's uncertainty at a point (live: pose covariance).
struct PassResult { float blocked_at = -1.f; int sections = 0, offsets = 0, reach_k = -1; };
PassResult stretch_blocked_at(const rc::RobotFootprint& body, const RouteSpline& spline, const GridPlanner& gp, float s0,
                              float s1, std::span<const Return> cells, const Pose2& bel, float risk_eps, float k_eps,
                              const std::function<float(const V2&)>& sigma_loc_at, float sigma_map);

// ★ A CLEARED MARK RE-PLANS ONLY FOR A SHORTER ROUTE (owner 2026-11-11). Removing an obstacle never makes the current route
//   unsafe; what a re-plan can buy is length, and every re-plan disturbs tracking (live chain9: 33 of 47 re-plans were clears,
//   tracking rms 0.037 -> 0.075 m). The candidate is the planner's polyline (never shorter than its smoothed spline), the
//   current route's remaining arc length is measured on its spline: the comparison is biased toward KEEPING the route, and
//   a gain must exceed the planner's own resolution (one cell) to be a gain at all.
float polyline_length(std::span<const V2> poly);
bool shorter_route(std::span<const V2> candidate, float remaining_m, float cell);

// ---------------------------------------------------------------------------------------------------------------- marks
// ★ MARK BELIEF (2026-11-11, the wait/re-mark cycle): whether an unmapped obstacle is still there is a belief, not a flag one
//   see-through sweep flips. Per mark, about the sweeps since its last hit:
//     PRESENT: a sweep sees through the obstacle with an UNKNOWN chance theta ~ Beta(1,1), learned from this mark's own
//              looks (a box: never; a person's legs: often);
//     LEFT:    it left right after the last hit; every sweep since is through except a false return (p_fa = risk_eps).
//   P(left) after a through-run m: odds (1 - p_fa)^m / P_present(run m); clears at P(left) >= 1 - risk_eps (the chance
//   constraint, no new level). ★FLAGGED: equal prior odds on "it left at the last hit".
//   (Rejected: a Markov hit<->through chain with the gap length learned from the current gap — "present but see-through for
//    ever" stays likely as 1/m and a once-seen mark never clears; mock person_in_door stuck 2/6.)
//   A sweep that does not look at a mark's cell (too close, occluded, no beam) is no evidence: the belief holds.
// FAILURE MARKS: where the specialist could not get through a stretch the planner still routes through; evidence about
//   this robot's ability, not cleared by sight. They LAPSE when an observed mark within lapse_r of them clears.
class MarkSet
{
public:
    struct Mark { V2 c = V2::Zero(); int hits = 1, throughs = 0, run = 0; };   // creation = a hit
    std::vector<Mark> marks;
    std::vector<V2> fail;

    // A sighting INSIDE an existing mark's square (the polygon it already gives the planner) is already THAT mark — not a new
    // obstacle (observe() scores the sweep's hit on it; counting it here too would double the evidence): live chain1 (2026-11-11) re-added every unmapped cell on every plan-first verdict, 975 marks
    // in 4 min, and the belief's cost grew with them. Returns true when a new mark was made.
    bool add(const V2& c, float cell);
    void add_fail(const V2& c) { fail.push_back(c); }
    bool empty() const { return marks.empty() and fail.empty(); }
    // the planner's polygons: one square per mark then per failure mark (half side = cell * sqrt(2) / 2)
    std::vector<std::vector<V2>> polygons(float cell) const;

    // one sweep, as (bearing, range) from the sensor at `origin` (room frame)
    struct Look { std::size_t cleared = 0, lapsed = 0; std::vector<V2> cleared_at; };
    Look observe(const V2& origin, std::span<const std::pair<float, float>> bearing_range, float cell, float near_r,
                 float risk_eps, float lapse_r, bool belief = true);
    // ★ MARKS FOLLOW RELOCALISATIONS, NOT ODOMETRY DRIFT: `transport` carries a room point through the room<-odom change of
    //   this cycle; odometry's own noise allows sigma_slip(c) there. A move beyond k_eps sigma (and half a cell) is a
    //   relocalisation: every mark is carried. Returns the largest displacement when carried, else a negative value.
    float carry_if_relocalised(const std::function<V2(const V2&)>& transport, const std::function<float(const V2&)>& sigma_slip,
                               float k_eps, float cell);
};

// ------------------------------------------------------------------------------------------------------- progress
// ★ CHAIN PROGRESS MONITOR: the arc length the tracker's commands (sent one lag ago) predict over its own horizon
//   H = T_lag + (W/2)/v_max against the projection's realised advance; fails when the prediction exceeds the projection's
//   resolution sqrt(sigma_pose^2 + h^2/12 + (v age)^2) and the realisation does not, or when the tracker predicts NOTHING
//   while the goal is not reached. A projection pinned at the route start is censored.
class ProgressMonitor
{
public:
    struct Rec { float t, adv, s, rot; };
    void record(float t, float adv, float s, float rot);
    void clear() { hist_.clear(); }
    struct In
    {
        float t = 0.f, s_prog = 0.f, dt = 0.05f;             // now, projection, command period
        float T_lag = 0.f, W = 0.f, v_max = 1.f;             // the tracker's lag, preview window, top speed
        float v = 0.f, w = 0.f, age = 0.f, sigma_pose = 0.f; // measured speeds, pose age, pose sigma along the route
        float circumscribed_r = 0.3f;
        bool goal_applies = false;                           // a place goal (a doorway is judged by its own through test)
        float goal_dist = 0.f, goal_tol = 0.f;               // believed distance to the goal, its tolerance
    };
    struct Verdict { bool fail = false, idle = false, censored = false, have = false; float pred = 0.f, realised = 0.f, resn = 0.f, H = 0.f; };
    Verdict judge(const In& in) const;
private:
    std::deque<Rec> hist_;
};
}  // namespace rc::chain
