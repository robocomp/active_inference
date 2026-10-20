// doorway_corridor.h — the measured doorway as the local policy's field and trap logic see it (Task 8). Pure (Eigen + std):
// the controller (goal_adapter.h) and the Task-6 harness (tests/test_policy_scenarios.cpp) share it.
#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace rc::policy
{
// The measured doorway gap in the ROOM frame. RelPose is the opening in the robot frame (x right, y forward): x.head<2>
// the centre, x.z() the wall's yaw psi; its normal (the direction of travel through it) is (-sin psi, cos psi) there
// (controller_anchor_servo.h anchor_cross_geometry).
struct RoomGap { Eigen::Vector2f centre = Eigen::Vector2f::Zero(), normal{0.f, 1.f}; float width = 0.f; };
// ★ THE DOORWAY'S CORRIDOR (where the cost-to-go may run through the opening). The planner's map is the CURRENT room plus
//   the door passage's corridor, which ends 2.0 m past the aperture (controller_door_passage.h: the cross standpoint 1.2 m
//   + 0.8 m) — the far room is not in it, so a goal line past the standoff would be unreachable on it. The opening AS
//   MEASURED supplies the rest: a strip along the normal from back_m before the gap centre to ahead_m past it, over the
//   lateral span the body's centre can occupy (|lateral| <= half the width - the inscribed radius; empty when the
//   opening is narrower than the body, so a too-narrow door stays unreachable). Contact with the jambs, the leaf and the
//   far room is the LiDAR's to judge (contact model), not this strip's.
inline bool in_doorway_corridor(const RoomGap& g, const Eigen::Vector2f& p, float back_m, float ahead_m, float inscribed_radius_m)
{
    const float half = 0.5f * g.width - inscribed_radius_m;
    if (not (half >= 0.f)) return false;
    const Eigen::Vector2f d = p - g.centre;
    const float s = d.dot(g.normal), lat = std::abs(d.dot(Eigen::Vector2f(-g.normal.y(), g.normal.x())));
    return s >= -back_m and s <= ahead_m and lat <= half;
}

// ★ Live fix 2 ruling 2: does a trap disc (centre c, radius r) MEET the measured doorway's corridor (the strip
//   in_doorway_corridor tests)? Only an opening at least the body's full width counts as evidence that the body fits;
//   a narrower one vetoes nothing.
inline bool disc_meets_doorway_corridor(const RoomGap& g, const Eigen::Vector2f& c, float r, float back_m, float ahead_m,
                                        float inscribed_radius_m, float body_width_m)
{
    if (not (g.width >= body_width_m)) return false;
    const float half = std::max(0.f, 0.5f * g.width - inscribed_radius_m);
    const Eigen::Vector2f d = c - g.centre, lat_dir(-g.normal.y(), g.normal.x());
    const float s = d.dot(g.normal), lat = d.dot(lat_dir);
    const float ds = s - std::clamp(s, -back_m, ahead_m), dl = lat - std::clamp(lat, -half, half);
    return std::hypot(ds, dl) <= r;
}

}  // namespace rc::policy
