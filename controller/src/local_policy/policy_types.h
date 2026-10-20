// policy_types.h — plain value types of the look-ahead local motion policy (room frame, Eigen only, no Qt/DSR).
#pragma once
#include <Eigen/Dense>

namespace rc::policy
{
// ★ yaw is the FORWARD heading (forward = (cos yaw, sin yaw)), the convention of RobotFootprint::*_yaw. The
// footprint's own forward axis is +y; the quarter turn lives in the footprint class and in contact.cpp, once.
struct Pose2 { Eigen::Vector2f p = Eigen::Vector2f::Zero(); float yaw = 0.f; };

// One LiDAR/map return in the room frame. sigma = range sigma (m); p_exist = probability it is a real obstacle.
// sxx/sxy/syy: scatter (m^2) of the hits a surface cell was merged from, about p — zero for a raw return; it is the
// cell's extent along the surface (contact.h: a cell is swept as that segment, and the scatter projected on the
// clearance direction widens its sigma — a jamb tip seen end-on is not pinned down by hits spread along it).
struct Return { Eigen::Vector2f p = Eigen::Vector2f::Zero(); float sigma = 0.02f; float p_exist = 1.f;
                float sxx = 0.f, sxy = 0.f, syy = 0.f; };

// Speed/acceleration envelope of the base. v_min_eff / w_min_eff are its deadband: the smallest speeds it
// actually executes (filled in from measurements in a later task).
struct BaseLimits { float v_max = 0.f, w_max = 0.f, a_lin = 0.f, d_lin = 0.f, a_rot = 0.f, d_rot = 0.f,
                    v_min_eff = 0.f, w_min_eff = 0.f; };
}  // namespace rc::policy
