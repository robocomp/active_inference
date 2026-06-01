#include "kinematics.h"

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/rnea.hpp>

#include <limits>
#include <print>
#include <stdexcept>


Kinematics::Kinematics(const std::string& urdf_path, const std::string& ee_frame_name)
    : ee_frame_name_(ee_frame_name)
{
    pinocchio::urdf::buildModel(urdf_path, model_);
    data_ = std::make_unique<pinocchio::Data>(model_);

    if (not model_.existFrame(ee_frame_name_))
        throw std::runtime_error("Kinematics: end-effector frame '" + ee_frame_name_ + "' not in URDF");
    ee_frame_id_ = model_.getFrameId(ee_frame_name_);

    for (int i = 0; i < N_ARM_JOINTS; ++i)
    {
        const std::string name = "joint_" + std::to_string(i + 1);
        if (not model_.existJointName(name))
            throw std::runtime_error("Kinematics: arm joint '" + name + "' not in URDF");
        arm_joint_ids_[i] = model_.getJointId(name);
    }

    // The controller and the KinovaArm proxy both work in the arm-base frame
    // (URDF base_link: +X forward, +Y left, +Z up, z=0 at the table the arm
    // stands on). The Webots Robot-node world rotation never enters the control
    // loop — getCenterOfTool(base) is base-relative and targets are sampled in
    // this same frame — and the KinovaGen3 proto frame is generated from this
    // very URDF. So base_tf_ is identity; FK already matches Webots in the
    // arm-base frame.
    base_tf_ = Eigen::Isometry3d::Identity();
}

Eigen::VectorXd Kinematics::angles_to_q(const std::array<double, N_ARM_JOINTS>& angles) const
{
    Eigen::VectorXd q = pinocchio::neutral(model_);
    for (int i = 0; i < N_ARM_JOINTS; ++i)
    {
        const auto& joint = model_.joints[arm_joint_ids_[i]];
        const int idx_q = joint.idx_q();
        if (joint.nq() == 1)
            q[idx_q] = angles[i];
        else if (joint.nq() == 2)  // continuous joint: (cos, sin)
        {
            q[idx_q]     = std::cos(angles[i]);
            q[idx_q + 1] = std::sin(angles[i]);
        }
    }
    return q;
}

Eigen::Vector3d Kinematics::forward_kinematics(const std::array<double, N_ARM_JOINTS>& angles)
{
    const Eigen::VectorXd q = angles_to_q(angles);
    pinocchio::framesForwardKinematics(model_, *data_, q);
    return base_tf_ * data_->oMf[ee_frame_id_].translation();
}

Eigen::Vector3d Kinematics::forward_kinematics_neutral()
{
    const Eigen::VectorXd q = pinocchio::neutral(model_);
    pinocchio::framesForwardKinematics(model_, *data_, q);
    return base_tf_ * data_->oMf[ee_frame_id_].translation();
}

Eigen::Matrix<double, 6, Eigen::Dynamic>
Kinematics::jacobian(const std::array<double, N_ARM_JOINTS>& angles)
{
    const Eigen::VectorXd q = angles_to_q(angles);
    Eigen::Matrix<double, 6, Eigen::Dynamic> J(6, model_.nv);
    J.setZero();
    pinocchio::computeFrameJacobian(model_, *data_, q, ee_frame_id_,
                                     pinocchio::LOCAL_WORLD_ALIGNED, J);
    // Rotate the world-aligned Jacobian into the base-mount frame so velocities
    // are expressed consistently with the (rotated) FK poses.
    const Eigen::Matrix3d R = base_tf_.linear();
    J.topRows(3)    = R * J.topRows(3);
    J.bottomRows(3) = R * J.bottomRows(3);
    return J;
}

Eigen::Matrix<double, 3, Kinematics::N_ARM_JOINTS>
Kinematics::arm_jacobian_linear(const std::array<double, N_ARM_JOINTS>& angles)
{
    const auto J_full = jacobian(angles);
    const auto idx_v  = arm_joint_idx_v();
    Eigen::Matrix<double, 3, N_ARM_JOINTS> J_arm;
    for (int i = 0; i < N_ARM_JOINTS; ++i)
        J_arm.col(i) = J_full.block<3, 1>(0, idx_v[i]);
    return J_arm;
}

Eigen::Matrix<double, 6, Kinematics::N_ARM_JOINTS>
Kinematics::arm_jacobian_full(const std::array<double, N_ARM_JOINTS>& angles)
{
    const auto J_full = jacobian(angles);
    const auto idx_v  = arm_joint_idx_v();
    Eigen::Matrix<double, 6, N_ARM_JOINTS> J_arm;
    for (int i = 0; i < N_ARM_JOINTS; ++i)
        J_arm.col(i) = J_full.col(idx_v[i]);
    return J_arm;
}

Kinematics::ToolPose Kinematics::tool_pose(const std::array<double, N_ARM_JOINTS>& angles)
{
    const Eigen::VectorXd q = angles_to_q(angles);
    pinocchio::framesForwardKinematics(model_, *data_, q);
    const auto& T = data_->oMf[ee_frame_id_];
    return ToolPose{base_tf_ * T.translation(), base_tf_.linear() * T.rotation()};
}

std::vector<Kinematics::MeshLinkPose>
Kinematics::arm_mesh_link_poses(const std::array<double, N_ARM_JOINTS>& angles)
{
    const Eigen::VectorXd q = angles_to_q(angles);
    pinocchio::forwardKinematics(model_, *data_, q);

    std::vector<MeshLinkPose> poses;
    poses.reserve(8);

    auto to_iso = [this](const pinocchio::SE3& T)
    {
        return base_tf_ * Eigen::Isometry3d(T.toHomogeneousMatrix());
    };

    poses.push_back({"base_link.STL", base_tf_});
    poses.push_back(MeshLinkPose{"shoulder_link.STL", to_iso(data_->oMi[arm_joint_ids_[0]])});
    poses.push_back(MeshLinkPose{"half_arm_1_link.STL", to_iso(data_->oMi[arm_joint_ids_[1]])});
    poses.push_back(MeshLinkPose{"half_arm_2_link.STL", to_iso(data_->oMi[arm_joint_ids_[2]])});
    poses.push_back(MeshLinkPose{"forearm_link.STL", to_iso(data_->oMi[arm_joint_ids_[3]])});
    poses.push_back(MeshLinkPose{"spherical_wrist_1_link.STL", to_iso(data_->oMi[arm_joint_ids_[4]])});
    poses.push_back(MeshLinkPose{"spherical_wrist_2_link.STL", to_iso(data_->oMi[arm_joint_ids_[5]])});
    poses.push_back(MeshLinkPose{"bracelet_with_vision_link.STL", to_iso(data_->oMi[arm_joint_ids_[6]])});

    return poses;
}

std::vector<Eigen::Vector3d>
Kinematics::arm_skeleton_points(const std::array<double, N_ARM_JOINTS>& angles)
{
    const Eigen::VectorXd q = angles_to_q(angles);
    pinocchio::forwardKinematics(model_, *data_, q);
    pinocchio::updateFramePlacements(model_, *data_);

    std::vector<Eigen::Vector3d> points;
    points.reserve(N_ARM_JOINTS + 2);

    points.emplace_back(base_tf_ * Eigen::Vector3d::Zero());
    for (int i = 0; i < N_ARM_JOINTS; ++i)
        points.emplace_back(base_tf_ * data_->oMi[arm_joint_ids_[i]].translation());
    points.emplace_back(base_tf_ * data_->oMf[ee_frame_id_].translation());

    return points;
}

std::array<double, Kinematics::N_ARM_JOINTS>
Kinematics::arm_gravity_torque(const std::array<double, N_ARM_JOINTS>& angles)
{
    const Eigen::VectorXd q = angles_to_q(angles);
    pinocchio::computeGeneralizedGravity(model_, *data_, q);   // fills data_->g (nv)
    const auto idx_v = arm_joint_idx_v();
    std::array<double, N_ARM_JOINTS> g{};
    for (int i = 0; i < N_ARM_JOINTS; ++i)
        g[i] = data_->g(idx_v[i]);
    return g;
}

Eigen::Matrix<double, 6, 1>
Kinematics::estimate_tool_wrench(const std::array<double, N_ARM_JOINTS>& angles,
                                const std::array<double, N_ARM_JOINTS>& joint_torque,
                                double damping)
{
    // Capture J BEFORE arm_gravity_torque (both write data_; J is a local copy).
    const auto J = arm_jacobian_full(angles);                  // 6×7, LOCAL_WORLD_ALIGNED
    const auto g = arm_gravity_torque(angles);
    Eigen::Matrix<double, N_ARM_JOINTS, 1> tau_ext;
    for (int i = 0; i < N_ARM_JOINTS; ++i)
        tau_ext(i) = g[i] - joint_torque[i];                   // Jᵀ w_ext = g − τ
    const Eigen::Matrix<double, 6, 6> A =
        J * J.transpose() + damping * damping * Eigen::Matrix<double, 6, 6>::Identity();
    return A.ldlt().solve(J * tau_ext);
}

std::array<int, Kinematics::N_ARM_JOINTS> Kinematics::arm_joint_idx_v() const
{
    std::array<int, N_ARM_JOINTS> idx_v{};
    for (int i = 0; i < N_ARM_JOINTS; ++i)
        idx_v[i] = model_.joints[arm_joint_ids_[i]].idx_v();
    return idx_v;
}

std::array<std::pair<double, double>, Kinematics::N_ARM_JOINTS>
Kinematics::arm_joint_position_limits() const
{
    std::array<std::pair<double, double>, N_ARM_JOINTS> lims{};
    for (int i = 0; i < N_ARM_JOINTS; ++i)
    {
        const auto& j = model_.joints[arm_joint_ids_[i]];
        const int idx_q = j.idx_q();
        // Continuous joints (nq=2) carry no meaningful (lower, upper) — Pinocchio
        // stores ±1 there. Report ±inf so the limit-avoidance term is a no-op.
        if (j.nq() == 2)
            lims[i] = {-std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::infinity()};
        else
            lims[i] = {model_.lowerPositionLimit[idx_q],
                       model_.upperPositionLimit[idx_q]};
    }
    return lims;
}

std::array<double, Kinematics::N_ARM_JOINTS> Kinematics::arm_joint_velocity_limits() const
{
    std::array<double, N_ARM_JOINTS> vlim{};
    for (int i = 0; i < N_ARM_JOINTS; ++i)
        vlim[i] = model_.velocityLimit[model_.joints[arm_joint_ids_[i]].idx_v()];
    return vlim;
}

void Kinematics::print_info() const
{
    std::print("[Kinematics] URDF loaded.  nq={}, nv={}, n_joints={} (incl. universe)\n",
               model_.nq, model_.nv, model_.njoints);
    std::print("[Kinematics] EE frame '{}' id={}\n", ee_frame_name_, ee_frame_id_);
    std::print("[Kinematics] Arm joints (joint_1..joint_7):\n");
    for (int i = 0; i < N_ARM_JOINTS; ++i)
    {
        const auto& j = model_.joints[arm_joint_ids_[i]];
        std::print("  joint_{}: id={}, idx_q={}, nq={}, nv={}\n",
                   i + 1, arm_joint_ids_[i], j.idx_q(), j.nq(), j.nv());
    }
}
