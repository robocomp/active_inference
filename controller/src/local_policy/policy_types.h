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
// aggregate: true for LiDAR/map returns (merged into bearing surface cells, contact.h); false for a measurement of a
// DIFFERENT source — an unseen-contact hypothesis (unseen_contact.h) — which is passed to the contact model as its own
// obstacle and never grouped with LiDAR hits (round-3 ruling R1: a sigma-0.005 hypothesis captured sigma-0.02 casing
// hits into one certain cell at the hypothesis's place, and the robot drove into the casing).
// seg (review fix round 2): a SURFACE SAMPLE's half-extent vector — the obstacle is the segment p +- seg (zero for a
// point). Set by denoise_returns() only.
struct Return { Eigen::Vector2f p = Eigen::Vector2f::Zero(); float sigma = 0.02f; float p_exist = 1.f;
                float sxx = 0.f, sxy = 0.f, syy = 0.f; bool aggregate = true; Eigen::Vector2f seg = Eigen::Vector2f::Zero(); };

// Speed/acceleration envelope of the base. v_min_eff / w_min_eff are its deadband: the smallest speeds it
// actually executes (filled in from measurements in a later task).
// track_m (Live fix 2 ruling 1): the axle length L of the differential base — the rollout ramps the two WHEEL speeds
// (v -+ w L/2) at a_lin / d_lin each. 0 = the L the other limits imply, 2 a_lin / a_rot (so a_rot stays what a pure
// rotation ramps at).
struct BaseLimits { float v_max = 0.f, w_max = 0.f, a_lin = 0.f, d_lin = 0.f, a_rot = 0.f, d_rot = 0.f,
                    v_min_eff = 0.f, w_min_eff = 0.f, track_m = 0.f; };
}  // namespace rc::policy
