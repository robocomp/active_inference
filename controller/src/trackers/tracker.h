#ifndef RC_TRACKER_H
#define RC_TRACKER_H

// A TRACKER turns (path, pose) into (adv, side, rot). Nothing else.
//
// Trackers used to live inside TrajectoryController's 3100-line body, which made it impossible to see
// what each one actually depended on — and that mattered, because the difference between the modes is
// almost entirely WHICH INPUTS THEY READ, not which arithmetic they do. Splitting them into files makes
// that visible; splitting the WORLD they read into two halves below makes it enforceable.
//
// ★THE TWO HALVES ARE THE POINT.
//   PathWorld  — the route, and the robot's own body. Pure geometry, no live measurement.
//   FieldWorld — the live ESDF built from lidar each cycle. Obstacle measurement.
// A tracker asks for the half it needs. PlainTracker takes PathWorld and therefore CANNOT query an
// obstacle even by accident: the premise "the path is self-adjusting and safe" stops being a comment
// in a header and becomes something the compiler checks. PD and MPPI take both, because they do their
// own avoidance.
//
// ★PlainTracker reads NO obstacle data of any kind — not a handle, not a scalar, nothing. It is given a
// path and a pose and must follow the path. Safety belongs to the planner's footprint predicate, to the
// band, and to blockage->replan, all of which run outside this file. A tracker that also watches
// obstacles becomes a second controller, and this codebase has now grown one twice (the clearance bound,
// then a "last-resort halt" that ended up firing on 13.8% of cycles with a rotate-away). Do not add a
// third.

#include <Eigen/Dense>

#include "../controller_types.h"

namespace rc
{
class RouteSpline;

// ── The geometry half ────────────────────────────────────────────────────────────────────────────
class PathWorld
{
public:
    virtual ~PathWorld() = default;
    // The route being followed, or nullptr when the mission is not running a continuous route.
    virtual const RouteSpline* route_spline() const = 0;
    // The body's reach toward `dir`, for a body at `heading`. Geometry of the robot, not of the world.
    virtual float body_extent(const Eigen::Vector2f& dir, float heading) const = 0;
    virtual float body_extent_max() const = 0;
};

// ── The live-measurement half ────────────────────────────────────────────────────────────────────
class FieldWorld
{
public:
    virtual ~FieldWorld() = default;
    // Signed distance to the nearest obstacle at a ROBOT-FRAME point, and its gradient.
    virtual float esdf_at(float rx, float ry) const = 0;
    virtual Eigen::Vector2f esdf_gradient_at(float rx, float ry) const = 0;
    // Body reach toward whatever is nearest at that point — the footprint support function along the
    // ESDF's negative gradient, so `esdf < extent` is an exact collision test, not a disc approximation.
    virtual float body_extent_toward_obstacle(float rx, float ry, float theta) const = 0;
    virtual float body_extent_here() const = 0;
};

// ── Per-cycle inputs, identical for every tracker ────────────────────────────────────────────────
struct TrackerInput
{
    Eigen::Affine2f robot_pose = Eigen::Affine2f::Identity();
    // Steering target in the ROBOT frame, already clipped to what the body can reach. Unused by
    // PlainTracker, which steers at the route itself rather than at a point on it.
    Eigen::Vector2f carrot_robot = Eigen::Vector2f::Zero();
    Eigen::Vector2f goal_robot   = Eigen::Vector2f::Zero();
    // The robot's projection onto the followed path, in ROOM coordinates, as compute_carrot last
    // computed it. PD's Stanley term needs it; it is an OUTPUT of the shared carrot machinery, not
    // something a tracker should recompute for itself.
    Eigen::Vector2f path_proj_room = Eigen::Vector2f::Zero();
    bool path_proj_valid = false;
};

class Tracker
{
public:
    virtual ~Tracker() = default;
    virtual const char* name() const = 0;
    // Called when the path changes or the robot stops, so a tracker can drop stale state.
    virtual void reset() {}
    virtual ControlOutput& compute(ControlOutput& out,
                                   const TrackerInput& in,
                                   const TrackerParams& p) = 0;
};

}   // namespace rc

#endif
