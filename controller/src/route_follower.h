/*
 * route_follower.h — drives a mission as ONE continuous route in arc-length coordinates.
 *
 * Owns the whole-route curve and the single scalar that says where the robot is on it. It replaces the
 * waypoint/target/arrival loop entirely: there is no "current waypoint" to reach, no arrival radius, and
 * no per-waypoint replan. See route_spline.h for why that scheme could not be repaired.
 *
 * THERE ARE NO LEGS. The recorded waypoints define the route and nothing else; the trajectory is
 * characterised by continuous quantities (cross-track error, lateral acceleration, speed variation)
 * measured at every instant, not by aggregating waypoint-to-waypoint segments.
 *
 * LAPS ARE CONCATENATED, not re-issued. Handing the follower a fresh path each lap would reset the MPPI
 * warm start at every lap boundary — the same defect as the per-waypoint replan, thirty times rarer but
 * the same kind. One path for the whole run means the state is reset exactly once.
 *
 * Pure Eigen/STL — no Qt, no DSR → unit-testable in isolation (self_test()).
 */

#pragma once

#include <cstdint>
#include <limits>
#include <functional>
#include <optional>
#include <vector>

#include <Eigen/Dense>

#include "route_spline.h"

namespace rc
{

class RouteFollower
{
public:
    // Plans every waypoint-to-waypoint hop with `plan` (the grid planner, so each hop is
    // footprint-feasible), repeats the route `laps` times, and fits one curve through the lot.
    // `plan(from, to)` returns nullopt if that hop is unreachable, which aborts the build — a route with
    // a hole in it is not a route.
    using PlanFn = std::function<std::optional<std::vector<Eigen::Vector2f>>(const Eigen::Vector2f &,
                                                                            const Eigen::Vector2f &)>;
    using FreeFn = std::function<bool(const Eigen::Vector2f &, float)>;

    bool build(const Eigen::Vector2f &start,
               const std::vector<Eigen::Vector2f> &waypoints,
               int laps,
               const PlanFn &plan,
               const FreeFn &is_free,
               float spacing = 0.05f,
               float smoothing_m = 0.40f);

    // LOCAL REPAIR — re-plan a WINDOW of the route around a newly-discovered blocker.
    //
    // A route is built once and driven in arc length, so the old "reset current_plan_ and let the next
    // cycle replan" recovery cannot work: there is no per-target replan left to trigger. Rebuilding the
    // whole route instead is not the answer either — it re-plans every hop and, worse, resets `progress_`,
    // so laps and finished() would believe the robot was back at the start. What actually changed is
    // LOCAL: one blocker, on one stretch of curve. So re-solve locally and splice.
    //
    // The window is [progress − back_m, progress + ahead_m]: `back_m` because the robot has just reversed
    // out and the detour must start somewhere it can reach, `ahead_m` because the detour has to clear the
    // blocker and rejoin the route. Both are STATED ALLOWANCES (how much route may be re-authored), in the
    // same sense as RouteSpline's smoothing scale — not safety numbers.
    //
    // `plan` must already see the new obstacle (the caller refreshes the planner grid every cycle).
    // Returns false if the window is degenerate or no detour exists; the caller then holds, and may retry
    // once the obstacle ages out. Waypoint arc lengths are re-derived and progress is re-anchored, so laps
    // and finished() stay meaningful across a repair.
    // NotNeeded is not a failure — it is the common case and the reason this is an enum. A recovery
    // reflex fires on "something happened near the robot", which is NOT the same question as "is my
    // route still drivable". Re-authoring a route that is still footprint-feasible is how a transient
    // blockage turns into a permanent detour, and it was observed doing exactly that (13 repairs in 23 s,
    // in places the robot had driven cleanly many times).
    enum class RepairResult { NotNeeded, Repaired, Failed };
    RepairResult repair(const Eigen::Vector2f &robot_pos,
                        float back_m,
                        float ahead_m,
                        const PlanFn &plan,
                        const FreeFn &is_free);

    // Hand the follower a distance field and weights, and every fit — build AND repair — is variationally
    // optimised (see route_optimizer.h). Anchors are filled in from the route's own waypoints, so the
    // caller supplies only the field and the weights. Leave unset and the route is fitted as before.
    void set_optimizer(const RouteOptimizerConfig &cfg) { opt_ = cfg; }

    bool valid() const { return spline_.valid(); }
    const std::vector<Eigen::Vector2f> &path() const { return spline_.samples(); }
    // The PLANNED POLYLINE the curve is fitted to — the concatenated A* hops, before any smoothing.
    // Exposed so the two geometries can be written side by side and compared directly.
    const std::vector<Eigen::Vector2f> &polyline() const { return poly_; }
    const RouteSpline &spline() const { return spline_; }
    float length() const { return spline_.length(); }

    // Advance progress to the robot's projection on the curve. Monotone: progress never runs backwards,
    // so a route that crosses itself cannot teleport the robot forwards or backwards along it.
    float advance(const Eigen::Vector2f &robot_pos);
    float progress() const { return progress_; }
    float remaining() const { return std::max(0.f, spline_.length() - progress_); }
    // A mission is finished when the ROBOT is at the end of the route — not when a projection is.
    // `progress_` is a forward-only nearest-point search, so it creeps toward the end whether or not the
    // robot is actually following the curve. Measured live: cross-track 1.10 m rms / 4.03 m max, progress
    // 35.42 of 35.62 m, and the run reported "completed" with the robot standing at waypoint 6.
    // Both conditions are now required, and the second one cannot be satisfied by a bookkeeping error.
    // How much arc length may remain and still count as driven. Shared by finished() and laps_done() —
    // they disagreed by exactly this amount, which is how a fully driven 3-lap run reported 2 of 3.
    static constexpr float finish_tol_m = 0.20f;

    bool  finished(const Eigen::Vector2f &robot_pos, float tolerance_m = finish_tol_m) const
    {
        if (remaining() > tolerance_m) return false;
        if (not spline_.valid()) return true;
        return (robot_pos - spline_.position_at(spline_.length())).norm() <= end_reach_m;
    }
    // How close to the route's END POINT the robot must physically be for the route to count as driven.
    // Generous on purpose: this is a sanity check against a projection running away, not an arrival
    // tolerance — the follower has no arrival radius by design.
    static constexpr float end_reach_m = 1.0f;

    // Which lap the given arc length falls in (1-based). There is no leg_at(): legs were the old
    // waypoint segmentation, and once the route is one curve, cutting it at the recorded waypoints
    // measures an artefact of how it USED to be driven rather than anything about the trajectory.
    int lap_at(float s) const;
    int lap() const { return lap_at(progress_); }
    // Laps FINISHED behind `s`. Distinct from lap_at, which is the lap you are IN — and the distinction
    // is not cosmetic: lap_at is clamped to laps_, so deriving "completed" as lap_at-1 can never reach
    // laps_ and a finished single-lap run reported 0 of 1 forever.
    int laps_completed_at(float s) const;
    int laps_done() const
    {
        // The route ENDS at its last waypoint, so that waypoint's arc length IS the route length — and
        // finished() lets the run end with up to finish_tol_m still nominally remaining. The last
        // waypoint is therefore never "passed" by a strict s >= ws test, and the final lap was lost.
        // If the route is driven, every lap in it is driven; there is nothing to infer.
        if (remaining() <= finish_tol_m) return laps_;
        return laps_completed_at(progress_);
    }
    // Arc length at which each waypoint sits, in build order (laps concatenated).
    const std::vector<float> &waypoint_arclengths() const { return wp_s_; }
    int waypoints_per_lap() const { return wp_per_lap_; }
    int corrections() const { return spline_.corrections(); }

    static bool self_test();

private:
    // Fit the curve to poly_ and re-derive every waypoint's arc length from wp_pos_. Shared by build()
    // and repair() so a repaired route is bookkept exactly like a freshly built one — the arc lengths
    // cannot drift apart from the geometry, because there is only one place that computes them.
    // `freeze_before` pins that many leading CONTROL POINTS — used on a repair so the stretch the robot
    // is already driving cannot move under it. 0 on a fresh build, where nothing is being driven yet.
    bool fit_from_polyline(const FreeFn &is_free, std::size_t freeze_before = 0,
                           std::size_t freeze_after = std::numeric_limits<std::size_t>::max());
    // Arc length of every waypoint ALONG THE POLYLINE — the metric the optimiser binds anchors in, and
    // the only one that is defined before the curve exists. Distinct from wp_s_, which is arc length
    // along the fitted CURVE and exists for progress and lap bookkeeping. Conflating the two deleted 15 m
    // of a 3-lap tour; see fit_from_polyline.
    std::vector<float> anchor_polyline_arclengths() const;

    RouteSpline spline_;
    // The planned polyline the curve is fitted to. Kept (build() used to discard it) because a local
    // repair needs something feasible to splice INTO: the curve's samples are a smoothed, resampled
    // product, not a sequence of planner-approved hops.
    std::vector<Eigen::Vector2f> poly_;
    std::vector<Eigen::Vector2f> wp_pos_;   // waypoint positions, laps concatenated — re-projected after a repair
    std::vector<float> wp_s_;      // arc length of every waypoint, laps concatenated
    RouteOptimizerConfig opt_;     // disabled until set_optimizer supplies a distance field
    float spacing_ = 0.05f;        // fit parameters, remembered so a repair refits identically
    float smoothing_ = 0.40f;
    // Waypoints RECORDED per lap — the reachable ones, not the mission's count. lap_at divides by this,
    // and the number of laps is stored explicitly rather than inferred as wp_s_.size()/wp_per_lap_: that
    // inference silently returned 0 the moment one waypoint was skipped (29/30), which zeroed the lap
    // counter and left laps_remaining pinned at its initial value forever.
    int   wp_per_lap_ = 0;
    int   laps_ = 1;
    float progress_ = 0.f;
};

}  // namespace rc
