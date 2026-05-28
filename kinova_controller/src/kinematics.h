#pragma once

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <array>
#include <memory>
#include <string>
#include <vector>

/**
 * Rigid-body kinematics wrapper around Pinocchio for the Kinova Gen3
 * + Robotiq 2F-85 URDF used by the kinova_controller agent.
 *
 * Joint convention.  The URDF has 7 arm joints joint_1..joint_7, of which
 * three are revolute (with limits) and four are continuous (unbounded).
 * Pinocchio represents continuous joints with two configuration entries
 * (cos, sin), so model.nq != model.nv:
 *   nq = 3*1 + 4*2 = 11
 *   nv = 7
 * The KinovaArm proxy gives us 7 joint angles; this class converts
 * between the 7-vector of arm angles and the 11-vector Pinocchio config.
 */
class Kinematics
{
public:
    static constexpr int N_ARM_JOINTS = 7;

    /** Load the URDF; throws std::runtime_error on failure. */
    explicit Kinematics(const std::string& urdf_path,
                        const std::string& ee_frame_name = "tool_frame");

    /** Configuration- and DOF-vector sizes (Pinocchio's view). */
    int nq() const { return model_.nq; }
    int nv() const { return model_.nv; }

    /** End-effector position in world frame at the given arm angles (7-vector). */
    Eigen::Vector3d forward_kinematics(const std::array<double, N_ARM_JOINTS>& angles);

    /** End-effector position at the model's neutral configuration. */
    Eigen::Vector3d forward_kinematics_neutral();

    /** Spatial Jacobian (6 x nv) at the given arm angles, expressed in the
     *  world frame, for the end-effector frame.  The first three rows are
     *  the linear part (used for position control), the last three rows
     *  are the angular part. */
    Eigen::Matrix<double, 6, Eigen::Dynamic> jacobian(
        const std::array<double, N_ARM_JOINTS>& angles);

    /** Linear part (3 x 7) of the tool-frame Jacobian, restricted to the
     *  seven arm joints. Maps arm joint velocities → tool linear velocity
     *  in world frame. The gripper and other non-arm DOFs are dropped. */
    Eigen::Matrix<double, 3, N_ARM_JOINTS> arm_jacobian_linear(
        const std::array<double, N_ARM_JOINTS>& angles);

    /** Full 6×7 arm Jacobian: rows 0-2 = linear, rows 3-5 = angular.
     *  Both expressed in the world frame (LOCAL_WORLD_ALIGNED convention).
     *  Used by efe_gradient_step when an orientation constraint is active. */
    Eigen::Matrix<double, 6, N_ARM_JOINTS> arm_jacobian_full(
        const std::array<double, N_ARM_JOINTS>& angles);

    /** Position + rotation of tool_frame in the world frame at the given
     *  arm angles. `rotation` columns are the tool's local x/y/z axes
     *  expressed in world coordinates — col(2) is the approach direction. */
    struct ToolPose
    {
        Eigen::Vector3d position;
        Eigen::Matrix3d rotation;
    };
    ToolPose tool_pose(const std::array<double, N_ARM_JOINTS>& angles);

    struct MeshLinkPose
    {
        std::string mesh_filename;
        Eigen::Isometry3d pose;
    };

    /** Mesh poses in world coordinates for visual arm links. */
    std::vector<MeshLinkPose> arm_mesh_link_poses(
        const std::array<double, N_ARM_JOINTS>& angles);

    /** World-space points for a lightweight arm skeleton visualization.
     *  Order: world origin, joint_1..joint_7 origins, tool_frame origin. */
    std::vector<Eigen::Vector3d> arm_skeleton_points(
        const std::array<double, N_ARM_JOINTS>& angles);

    /** idx_v of each arm joint inside the full nv-vector (Pinocchio ordering). */
    std::array<int, N_ARM_JOINTS> arm_joint_idx_v() const;

    /** (lower, upper) position limits for each arm joint, in radians. Continuous
     *  joints report (-inf, +inf). Used by the gradient module for limit
     *  avoidance on the three revolute joints (joint_2, joint_4, joint_6). */
    std::array<std::pair<double, double>, N_ARM_JOINTS> arm_joint_position_limits() const;

    /** Per-joint velocity limit (rad/s) declared in the URDF, for the 7 arm joints. */
    std::array<double, N_ARM_JOINTS> arm_joint_velocity_limits() const;

    /** Dump model summary (nq, nv, joint names + their idx_q/nv) to std::cout. */
    void print_info() const;

private:
    /** Map a 7-vector of arm joint angles into the model's nq-vector. */
    Eigen::VectorXd angles_to_q(const std::array<double, N_ARM_JOINTS>& angles) const;

    pinocchio::Model model_;
    // Held by pointer so it is constructed exactly once, after `model_` has
    // been populated by buildModel. A direct-member data_ would be
    // default-constructed against the empty model_ at initialiser-list time
    // and then reassigned, which leaves Pinocchio's internal check hash in
    // an inconsistent state (model.check(data) asserts otherwise).
    std::unique_ptr<pinocchio::Data> data_;
    std::string ee_frame_name_;
    pinocchio::FrameIndex ee_frame_id_ = 0;

    // Map of arm joint name → joint index in the model, in the order
    // joint_1..joint_7. Initialised once in the constructor.
    std::array<pinocchio::JointIndex, N_ARM_JOINTS> arm_joint_ids_{};
};
