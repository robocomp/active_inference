// contact.h — rollout of a (v, w) command and the swept-body probability of touching nothing.
#pragma once
#include <span>
#include <vector>
#include "policy_types.h"
#include "../../../common/robot_footprint/robot_footprint.h"

namespace rc::policy
{
// Hold (v, w) for hold_s, then brake both axes independently to rest at d_lin / d_rot. Poses every dt (start pose
// first, final rest pose last). Differential drive, exact arc integration (straight-line limit as |w| -> 0).
std::vector<Pose2> rollout(const Pose2& start, float v, float w, float hold_s, const BaseLimits& b, float dt);

// Signed distance (m) from a room-frame point to the body at `pose`, margin included: negative inside the
// margin-grown polygon. Same quarter-turn convention as RobotFootprint::contains_yaw (tested against it).
float signed_distance(const rc::RobotFootprint& body, const Pose2& pose, const Eigen::Vector2f& q);

// Log-probability that the swept body touches none of the returns. Per return j: sd_j = min over the sweep of the
// signed distance; sigma_j^2 = sigma_r^2 + sigma_xy^2 + (rho_j*sigma_yaw)^2, rho_j = distance of the return from
// the pose centre; P(contact_j) = p_exist * Phi(-sd_j/sigma_j); result = sum_j log(1 - P(contact_j)).
// A return already inside the body at the START pose counts only if the sweep makes it deeper.
float log_p_no_contact(const rc::RobotFootprint& body, std::span<const Pose2> sweep, std::span<const Return> r,
                       float sigma_pose_xy, float sigma_pose_yaw);
}  // namespace rc::policy
