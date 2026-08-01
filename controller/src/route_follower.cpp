/*
 * route_follower.cpp — see route_follower.h
 */

#include "route_follower.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace rc
{

bool RouteFollower::build(const Eigen::Vector2f &start,
                          const std::vector<Eigen::Vector2f> &waypoints,
                          int laps,
                          const PlanFn &plan,
                          const FreeFn &is_free,
                          float spacing,
                          float smoothing_m)
{
    spline_ = RouteSpline{};
    wp_s_.clear();
    poly_.clear();
    wp_pos_.clear();
    wp_per_lap_ = 0;
    progress_ = 0.f;
    if (waypoints.size() < 2 or not plan) return false;

    laps = std::max(1, laps);
    laps_ = laps;
    wp_per_lap_ = 0;          // set from what lap 1 actually records, not from what was asked for
    spacing_ = std::max(0.02f, spacing);
    smoothing_ = smoothing_m;

    // 1. One polyline for the WHOLE RUN. Each hop is planned, never interpolated: a straight line
    //    between two waypoints is not known to be drivable, and a route containing a segment the planner
    //    never approved is exactly the lossy shortcut the grid planner exists to remove.
    poly_.push_back(start);
    Eigen::Vector2f from = start;
    int unreachable = 0;
    for (int lap = 0; lap < laps; ++lap)
        for (const auto &wp : waypoints)
        {
            const auto hop = plan(from, wp);
            if (not hop.has_value() or hop->size() < 2)
            {
                // UNREACHABLE, not merely blocked. The caller has already repaired this waypoint to the
                // nearest footprint-FEASIBLE pose — but feasible is not the same as reachable, and
                // nearest_free has no connectivity requirement, so it can land the waypoint in an enclosed
                // pocket (between an obstacle and a wall) that the planner can never enter. Observed live:
                // waypoint 14 repaired 0.70 m into such a pocket, A* exhausted 67k states over 16k free
                // cells, and ONE waypoint killed the whole 30-waypoint tour.
                //
                // Dropping it keeps the tour drivable, which is better than refusing to move at all — the
                // same policy the caller already applies to a boxed-in waypoint. It CHANGES THE ROUTE, so
                // it is said out loud rather than swallowed.
                ++unreachable;
                std::printf("[route] waypoint (%.2f,%.2f) is UNREACHABLE from (%.2f,%.2f) — skipping it. "
                            "The driven route no longer matches the recorded one.\n",
                            wp.x(), wp.y(), from.x(), from.y());
                std::fflush(stdout);
                continue;                       // `from` stays put: the next hop starts where we still are
            }
            for (std::size_t i = 1; i < hop->size(); ++i) poly_.push_back((*hop)[i]);
            wp_pos_.push_back(wp);
            if (lap == 0) ++wp_per_lap_;      // the reachable count, which is what lap_at must divide by
            from = wp;
        }

    if (wp_per_lap_ < 2 or wp_pos_.size() < 2)
    {
        std::printf("[route] build FAILED: only %zu of %d waypoints are reachable — that is not a route.\n",
                    wp_pos_.size(), wp_per_lap_ * laps);
        std::fflush(stdout);
        return false;
    }

    // 2-3. One curve through all of it, then every waypoint's arc length. Both live in
    //      fit_from_polyline so a repair reproduces them by the same code.
    if (not fit_from_polyline(is_free)) return false;

    std::printf("[route] built: %d lap(s) x %d reachable waypoints (%d UNREACHABLE, skipped) -> %.2f m, "
                "%zu samples at %.0f cm, %d feasibility corrections\n",
                laps, wp_per_lap_, unreachable, spline_.length(), spline_.samples().size(),
                spline_.spacing() * 100, spline_.corrections());
    std::fflush(stdout);
    return true;
}

bool RouteFollower::fit_from_polyline(const FreeFn &is_free, std::size_t freeze_before)
{
    if (poly_.size() < 2) return false;

    // The optimiser's LIKELIHOOD term is the route's own authored waypoints, so they are filled in here
    // rather than asked of the caller — nothing outside this class knows what the route was authored from.
    RouteOptimizerConfig opt = opt_;
    opt.anchors = wp_pos_;
    opt.anchor_s = wp_s_;   // bind by arc length, not by nearest control point (see RouteOptimizerConfig)
    opt.freeze_before = freeze_before;
    const RouteOptimizerConfig *popt = (opt.enabled and opt.distance) ? &opt : nullptr;

    if (not spline_.build(poly_, spacing_, is_free, smoothing_, popt)) return false;

    // Arc length of every waypoint. Projected in ORDER with a forward-only hint so a self-crossing route
    // cannot bind a waypoint to the wrong passage — this tour crosses itself, so a global nearest-point
    // search would do exactly that.
    wp_s_.clear();
    wp_s_.reserve(wp_pos_.size());
    float s_hint = 0.f;
    for (const auto &w : wp_pos_)
    {
        s_hint = spline_.project(w, s_hint, 8.0f);
        wp_s_.push_back(s_hint);
    }
    return true;
}

RouteFollower::RepairResult RouteFollower::repair(const Eigen::Vector2f &robot_pos,
                                                  float back_m,
                                                  float ahead_m,
                                                  const PlanFn &plan,
                                                  const FreeFn &is_free)
{
    if (not spline_.valid() or poly_.size() < 2 or not plan) return RepairResult::Failed;

    const float L = spline_.length();
    const float s0 = std::clamp(progress_ - std::max(0.f, back_m), 0.f, L);
    const float s1 = std::clamp(progress_ + std::max(0.f, ahead_m), 0.f, L);
    // Too near the end for a detour to rejoin anything. Not a failure of planning — there is simply no
    // route left to re-author, so say no rather than splice a degenerate window.
    if (s1 - s0 < 0.5f) return RepairResult::NotNeeded;

    // ── IS THE ROUTE ACTUALLY BLOCKED? ────────────────────────────────────────────────────────────
    // Ask the question the caller could not: walk the curve across the window and test it with the SAME
    // footprint predicate the planner and the feasibility pass use. If every pose still fits, the thing
    // that fired the reflex is not on our path — it is beside it, behind it, or already gone — and the
    // route needs no repair. This is the difference between "an obstacle appeared" and "I cannot get
    // through", and conflating them is what produced a detour storm in places that were plainly drivable.
    if (is_free)
    {
        bool blocked = false;
        for (float s = s0; s <= s1 and not blocked; s += spline_.spacing())
            if (not is_free(spline_.position_at(s), spline_.heading_at(s)))
                blocked = true;
        if (not blocked) return RepairResult::NotNeeded;
    }

    // Map the arc-length window onto the POLYLINE, which is what gets spliced. Do it by ARC LENGTH, never
    // by nearest-vertex search: this route crosses itself (the tour has two passes 0.6 m apart), so a
    // spatial search for "the vertex nearest the window start" can bind to the branch driven ten metres
    // ago and splice a detour into the wrong pass. Walking cumulative length is monotone in s by
    // construction, so that failure is not merely unlikely — it is unrepresentable.
    // The polyline and the curve differ in length (smoothing cuts corners), so s is rescaled by their
    // ratio rather than compared raw.
    std::vector<float> cum(poly_.size(), 0.f);
    for (std::size_t i = 1; i < poly_.size(); ++i)
        cum[i] = cum[i - 1] + (poly_[i] - poly_[i - 1]).norm();
    if (cum.back() < 1e-3f) return RepairResult::Failed;
    const float scale = cum.back() / std::max(L, 1e-6f);
    // The bracket must CONTAIN the window, so the two ends round outwards: i0 is the last vertex at or
    // before s0 and i1 the first at or after s1. Rounding both the same way (the obvious lower_bound for
    // each) walks the whole window forward by up to one vertex, which would start the detour AHEAD of the
    // robot — exactly what back_m exists to prevent.
    const auto index_before = [&cum, scale](float s)
    {
        const float target = std::clamp(s * scale, 0.f, cum.back());
        const auto it = std::upper_bound(cum.begin(), cum.end(), target);
        return it == cum.begin() ? std::size_t{0} : static_cast<std::size_t>(it - cum.begin() - 1);
    };
    const auto index_after = [&cum, scale](float s)
    {
        const float target = std::clamp(s * scale, 0.f, cum.back());
        const auto it = std::lower_bound(cum.begin(), cum.end(), target);
        return it == cum.end() ? cum.size() - 1 : static_cast<std::size_t>(it - cum.begin());
    };
    const std::size_t i0 = index_before(s0);
    const std::size_t i1 = index_after(s1);
    if (i1 <= i0 or i1 >= poly_.size()) return RepairResult::Failed;

    // ── PLAN THROUGH THE AUTHORED WAYPOINTS INSIDE THE WINDOW ─────────────────────────────────────
    // A single hop from i0 to i1 asks A* for the SHORTEST path across the window, which cuts whatever
    // excursion the tour was authored to make through it. Measured: three repairs took the route from
    // 35.64 m to 30.99 m, and again 35.96 -> 31.66 m on the next run — about 13% of the tour deleted,
    // one window at a time. A detour is LONGER than what it replaces; anything shorter is a shortcut,
    // and a shortcut silently rewrites the stimulus the run is supposed to measure.
    // So the window is re-planned as a CHAIN through the waypoints it contains. A repair can then route
    // around a blocker but can never straighten the tour.
    std::vector<Eigen::Vector2f> via;
    for (std::size_t j = 0; j < wp_s_.size() and j < wp_pos_.size(); ++j)
        if (wp_s_[j] > s0 + 0.10f and wp_s_[j] < s1 - 0.10f) via.push_back(wp_pos_[j]);

    std::vector<Eigen::Vector2f> hop{poly_[i0]};
    Eigen::Vector2f from = poly_[i0];
    int dropped = 0;
    for (const auto &v : via)
    {
        const auto seg = plan(from, v);
        // A waypoint we cannot reach IS the thing we are routing around — drop that one and carry on,
        // rather than failing the whole repair. Dropping it is a real change to the route, so it is
        // counted and reported rather than swallowed.
        if (not seg.has_value() or seg->size() < 2) { ++dropped; continue; }
        for (std::size_t k = 1; k < seg->size(); ++k) hop.push_back((*seg)[k]);
        from = v;
    }
    const auto tail = plan(from, poly_[i1]);
    if (not tail.has_value() or tail->size() < 2) return RepairResult::Failed;
    for (std::size_t k = 1; k < tail->size(); ++k) hop.push_back((*tail)[k]);
    if (hop.size() < 2) return RepairResult::Failed;

    std::vector<Eigen::Vector2f> spliced;
    spliced.reserve(poly_.size() + hop.size());
    spliced.insert(spliced.end(), poly_.begin(), poly_.begin() + static_cast<std::ptrdiff_t>(i0) + 1);
    for (std::size_t i = 1; i < hop.size(); ++i) spliced.push_back(hop[i]);   // hop[0] == poly_[i0]
    spliced.insert(spliced.end(), poly_.begin() + static_cast<std::ptrdiff_t>(i1) + 1, poly_.end());

    auto previous = poly_;
    poly_ = std::move(spliced);
    // FREEZE WHAT THE ROBOT IS ALREADY DRIVING. The refit re-optimises the control polygon, and moving the
    // stretch under the robot would put a step in the cross-track error at the worst possible moment. The
    // window start s0 is behind the robot by construction, so everything before it is safe to pin. Control
    // points sit every `ctrl_step` along the POLYLINE, hence the same rescaling used for the bracket.
    const float ctrl_step = std::max(spacing_, smoothing_);
    const std::size_t freeze = static_cast<std::size_t>(std::max(0.f, s0 * scale / ctrl_step));
    if (not fit_from_polyline(is_free, freeze))
    {
        poly_ = std::move(previous);        // a failed refit must not leave a half-repaired route behind
        fit_from_polyline(is_free);
        return RepairResult::Failed;
    }

    // Re-anchor progress on the NEW curve. The arc-length scale changed under us, so the monotonicity
    // invariant of advance() does not carry across a repair — this is the one place progress_ may move
    // backwards, and it does so by construction rather than by a projection accident. Search forward from
    // the window start, which is behind the robot, with a window wide enough to contain the detour.
    const float hint = std::max(0.f, s0 - 0.5f);
    progress_ = spline_.project(robot_pos, hint, std::max(4.f, back_m + ahead_m));

    std::printf("[route] repaired %.2f m window at s=%.2f through %zu authored waypoint(s)"
                "%s -> new length %.2f m, %zu samples, %d feasibility corrections\n",
                s1 - s0, progress_, via.size() - static_cast<std::size_t>(dropped),
                dropped > 0 ? " (some unreachable and DROPPED)" : "",
                spline_.length(), spline_.samples().size(), spline_.corrections());
    std::fflush(stdout);
    return RepairResult::Repaired;
}

float RouteFollower::advance(const Eigen::Vector2f &robot_pos)
{
    if (not spline_.valid()) return progress_;
    // Forward-only, and never past the end.
    progress_ = std::clamp(spline_.project(robot_pos, progress_), progress_, spline_.length());
    return progress_;
}

int RouteFollower::lap_at(float s) const
{
    if (wp_s_.empty() or wp_per_lap_ <= 0) return 1;
    // The lap containing s = how many whole laps of waypoints are already behind it.
    int passed = 0;
    for (const float ws : wp_s_)
        if (s >= ws) ++passed;
    return std::min(laps_, 1 + passed / wp_per_lap_);
}

int RouteFollower::laps_completed_at(float s) const
{
    if (wp_s_.empty() or wp_per_lap_ <= 0) return 0;
    int passed = 0;
    for (const float ws : wp_s_)
        if (s >= ws) ++passed;
    return std::min(laps_, passed / wp_per_lap_);   // NO +1: a lap counts once its last waypoint is behind us
}

bool RouteFollower::self_test()
{
    bool ok = true;
    auto check = [&](bool c, const char *m) { if (not c) { ok = false; std::printf("  FAIL: %s\n", m); } };

    // A trivial "planner" that returns the straight segment — enough to exercise the concatenation,
    // the arc-length bookkeeping and the lap mapping without a grid.
    const PlanFn straight = [](const Eigen::Vector2f &a, const Eigen::Vector2f &b)
        -> std::optional<std::vector<Eigen::Vector2f>> { return std::vector<Eigen::Vector2f>{a, b}; };

    // A 2 m square, driven twice.
    const std::vector<Eigen::Vector2f> square{{2.f, 0.f}, {2.f, 2.f}, {0.f, 2.f}, {0.f, 0.f}};
    {
        RouteFollower r;
        check(r.build({0.f, 0.f}, square, 2, straight, {}), "a square route must build");
        check(r.waypoint_arclengths().size() == 8, "2 laps x 4 waypoints = 8 waypoint arc lengths");
        check(r.laps_completed_at(0.f) == 0, "nothing driven yet is zero laps completed");
        check(r.laps_completed_at(r.length()) == 2, "driving the whole route completes every lap");
        // The real path: DRIVE the route (advance is forward-only within a window, so it cannot be
        // teleported to the end), stopping just short exactly as finished() permits. The lap count must
        // still be complete — a strict arc-length test loses the final lap here, because the last
        // waypoint's arc length IS the route length.
        const float stop_at = r.length() - 0.5f * RouteFollower::finish_tol_m;
        for (float s = 0.f; s <= stop_at; s += 0.5f) r.advance(r.spline().position_at(s));
        r.advance(r.spline().position_at(stop_at));
        std::printf("  finish tolerance: stopped at s=%.2f of %.2f m -> %d laps done\n",
                    r.progress(), r.length(), r.laps_done());
        check(r.laps_done() == 2, "a run that ends within the finish tolerance has completed every lap");
        check(r.waypoints_per_lap() == 4, "waypoints per lap must be remembered");
        // 8 m per lap, 2 laps; smoothing cuts the corners so it is a little shorter.
        std::printf("  square x2: length %.2f m (raw 16.00), samples %zu\n", r.length(), r.path().size());
        check(r.length() > 14.f and r.length() < 16.5f, "length must be about two laps of the square");

        // Arc lengths must be strictly increasing — a waypoint bound to the wrong passage on lap 2
        // would show up as a step backwards here, and would silently mis-attribute every metric.
        const auto &ws = r.waypoint_arclengths();
        for (std::size_t i = 1; i < ws.size(); ++i)
            check(ws[i] > ws[i - 1], "waypoint arc lengths must increase monotonically across laps");

        // Lap mapping.
        check(r.lap_at(0.1f) == 1, "the start is lap 1");
        check(r.lap_at(r.length() - 0.1f) == 2, "the end is on lap 2");
        std::printf("  mapping: s=0.1 -> lap %d | s=%.1f -> lap %d\n",
                    r.lap_at(0.1f), r.length() - 0.1f, r.lap_at(r.length() - 0.1f));
    }

    // PROGRESS IS MONOTONE. Feeding poses that step backwards must not move progress backwards: on a
    // route that revisits the same place each lap, a backward jump would re-run a lap of metrics.
    {
        RouteFollower r;
        r.build({0.f, 0.f}, square, 2, straight, {});
        const float a = r.advance({2.f, 1.0f});
        const float b = r.advance({0.5f, 0.f});      // back near the start of lap 1
        check(b >= a, "progress must never run backwards, even if the pose does");
        std::printf("  monotonicity: %.2f m then a backward pose -> %.2f m\n", a, b);
    }

    // LOCAL REPAIR. A blocker appears mid-route; the detour must splice in, keep the waypoint bookkeeping
    // consistent, and leave progress pointing at the robot — not at the old arc length, which the splice
    // has just invalidated.
    {
        RouteFollower r;
        check(r.build({0.f, 0.f}, square, 1, straight, {}), "a square route must build for the repair test");
        const float len_before = r.length();
        const std::size_t wps_before = r.waypoint_arclengths().size();

        r.advance({2.f, 0.8f});                      // partway up the first side
        const float s_before = r.progress();
        check(s_before > 1.f, "the robot must have made progress before the repair");

        // A "planner" that bypasses any hop via a point 1.5 m PERPENDICULAR to it. The offset must be
        // perpendicular, not along a fixed axis: a fixed axis can land the bypass point back on the hop's
        // own endpoint (it did, on the leg running along −x), producing a degenerate "detour" that
        // silently tests nothing.
        Eigen::Vector2f bypass_point{0.f, 0.f};
        const PlanFn detour = [&bypass_point](const Eigen::Vector2f &a, const Eigen::Vector2f &b)
            -> std::optional<std::vector<Eigen::Vector2f>>
        {
            const Eigen::Vector2f d = (b - a).normalized();
            const Eigen::Vector2f n{-d.y(), d.x()};
            bypass_point = 0.5f * (a + b) + 1.5f * n;
            return std::vector<Eigen::Vector2f>{a, bypass_point, b};
        };

        check(r.repair({2.f, 0.8f}, 0.5f, 1.5f, detour, {}) == RepairResult::Repaired,
              "a repair with a reachable detour must succeed");
        check(r.valid(), "the route must still be valid after a repair");
        // Assert the detour is actually IN the curve rather than asserting it got longer — whether a
        // bypass lengthens or shortens a route depends on the planner's geometry, so a length inequality
        // would be testing the stub, not the splice.
        float to_bypass = std::numeric_limits<float>::max();
        for (const auto &p : r.path()) to_bypass = std::min(to_bypass, (p - bypass_point).norm());
        check(to_bypass < 0.40f, "the repaired curve must actually pass through the planned detour");
        check(std::abs(r.length() - len_before) > 1e-3f, "a repair must change the route's length");
        check(r.waypoint_arclengths().size() == wps_before,
              "a repair must not add or drop waypoints — only re-derive their arc lengths");
        check(r.waypoints_per_lap() == 4, "waypoints per lap must survive a repair");
        // Progress must name the robot's own place on the NEW curve. Testing this against a fixed
        // distance would be meaningless — after a detour the robot is legitimately off the route, so any
        // loose bound passes for the wrong reason. The real invariant is that the re-anchored projection
        // is the CLOSEST point on the repaired curve, so compare it against a brute-force scan.
        const Eigen::Vector2f robot{2.f, 0.8f};
        const float err = (r.spline().position_at(r.progress()) - robot).norm();
        float best = std::numeric_limits<float>::max();
        for (const auto &p : r.path()) best = std::min(best, (p - robot).norm());
        check(err < best + 0.05f, "re-anchored progress must be the closest point on the repaired curve");
        std::printf("  repair: length %.2f -> %.2f m, progress %.2f -> %.2f m, "
                    "re-anchor %.3f m (global closest %.3f m)\n",
                    len_before, r.length(), s_before, r.progress(), err, best);

        // A repair with no reachable detour must leave the route untouched and usable.
        const PlanFn none = [](const Eigen::Vector2f &, const Eigen::Vector2f &)
            -> std::optional<std::vector<Eigen::Vector2f>> { return std::nullopt; };
        const float len_kept = r.length();
        check(r.repair({2.f, 0.8f}, 0.5f, 1.5f, none, {}) == RepairResult::Failed,
              "an unreachable detour must report failure");
        check(r.valid() and std::abs(r.length() - len_kept) < 1e-3f,
              "a failed repair must leave the route exactly as it was");
    }

    // A REPAIR MUST NOT EAT THE ROUTE. The window is re-planned through the waypoints it contains, so a
    // shortest-path planner cannot straighten the tour. Measured before this: three repairs took a 35.6 m
    // route to 31.0 m, one window at a time, silently changing the stimulus the run was measuring.
    {
        RouteFollower r;
        // A big square: waypoints far enough apart that a 3 m window contains an interior one.
        const std::vector<Eigen::Vector2f> big{{4.f, 0.f}, {4.f, 4.f}, {0.f, 4.f}, {0.f, 0.f}};
        // is_free says the route IS blocked, so the repair actually runs; the planner returns straight
        // segments, which is the WORST case for shortcutting — it will cut any corner it is allowed to.
        const FreeFn blocked = [](const Eigen::Vector2f &, float) { return false; };
        check(r.build({0.f, 0.f}, big, 1, straight, {}), "the big square must build");
        const float len_before = r.length();
        r.advance({4.f, 1.5f});
        const auto res = r.repair({4.f, 1.5f}, 1.0f, 3.0f, straight, blocked);
        check(res == RepairResult::Repaired, "a blocked route must actually be repaired");
        // The corner at (4,4) lies inside the window. If it were bypassed the route would collapse
        // diagonally and lose metres; planning through it keeps the length.
        std::printf("  no-shortcut: %.2f m -> %.2f m across a repair spanning a corner\n",
                    len_before, r.length());
        check(r.length() > len_before - 0.60f,
              "a repair must not shorten the route — that is a shortcut, not a detour");
    }

    // A REFLEX THAT FIRED ON NOTHING must not re-author the route. This is the common case in the field:
    // a blockage is seen beside the path, or has already gone by the time the repair runs.
    {
        RouteFollower r;
        const FreeFn all_free = [](const Eigen::Vector2f &, float) { return true; };
        check(r.build({0.f, 0.f}, square, 1, straight, all_free), "a square route must build");
        r.advance({2.f, 0.8f});
        const float len_before = r.length();
        const float s_before = r.progress();
        // A "planner" that would produce a wild detour if it were ever consulted — it must NOT be.
        bool planner_called = false;
        const PlanFn spy = [&planner_called](const Eigen::Vector2f &a, const Eigen::Vector2f &b)
            -> std::optional<std::vector<Eigen::Vector2f>>
        { planner_called = true; return std::vector<Eigen::Vector2f>{a, {9.f, 9.f}, b}; };
        check(r.repair({2.f, 0.8f}, 0.5f, 1.5f, spy, all_free) == RepairResult::NotNeeded,
              "a still-feasible route must report NotNeeded");
        check(not planner_called, "NotNeeded must not even consult the planner");
        check(std::abs(r.length() - len_before) < 1e-4f and std::abs(r.progress() - s_before) < 1e-4f,
              "NotNeeded must leave the route and the progress untouched");
        std::printf("  not-needed: feasible route left alone (%.2f m, s=%.2f), planner not consulted\n",
                    r.length(), r.progress());
    }

    // AN UNREACHABLE WAYPOINT MUST NOT KILL THE WHOLE TOUR. `nearest_free` guarantees a waypoint is
    // FEASIBLE, never that it is REACHABLE — it has no connectivity requirement, so it can repair one into
    // an enclosed pocket. Observed live: one such waypoint made a 30-waypoint tour refuse to build at all,
    // repeatedly. Skipping it keeps the route connected (the next hop starts from where we still are) and
    // drivable; it changes the route, which is why it is counted and announced.
    {
        RouteFollower r;
        const PlanFn broken = [](const Eigen::Vector2f &a, const Eigen::Vector2f &b)
            -> std::optional<std::vector<Eigen::Vector2f>>
        { if (b.y() > 1.5f) return std::nullopt; return std::vector<Eigen::Vector2f>{a, b}; };
        check(r.build({0.f, 0.f}, square, 1, broken, {}),
              "two unreachable waypoints out of four must still yield a drivable route");
        check(r.valid(), "the surviving route must be valid");
        check(r.waypoint_arclengths().size() == 2,
              "only the REACHABLE waypoints may be recorded, or the arc lengths describe a route we skipped");
        std::printf("  unreachable: 2 of 4 waypoints skipped -> %.2f m route still built\n", r.length());
    }

    // ...but a route that has lost almost everything is not a route.
    {
        RouteFollower r;
        const PlanFn nearly_all_broken = [](const Eigen::Vector2f &a, const Eigen::Vector2f &b)
            -> std::optional<std::vector<Eigen::Vector2f>>
        { if (not (b.x() > 1.5f and b.y() < 0.5f)) return std::nullopt; return std::vector<Eigen::Vector2f>{a, b}; };
        check(not r.build({0.f, 0.f}, square, 1, nearly_all_broken, {}),
              "fewer than two reachable waypoints must fail the build");
        check(not r.valid(), "a failed build must leave nothing valid behind");
    }

    std::printf("RouteFollower::self_test %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace rc
