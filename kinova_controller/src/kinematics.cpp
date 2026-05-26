#include "kinematics.h"

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>

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
    return data_->oMf[ee_frame_id_].translation();
}

Eigen::Vector3d Kinematics::forward_kinematics_neutral()
{
    const Eigen::VectorXd q = pinocchio::neutral(model_);
    pinocchio::framesForwardKinematics(model_, *data_, q);
    return data_->oMf[ee_frame_id_].translation();
}

Eigen::Matrix<double, 6, Eigen::Dynamic>
Kinematics::jacobian(const std::array<double, N_ARM_JOINTS>& angles)
{
    const Eigen::VectorXd q = angles_to_q(angles);
    Eigen::Matrix<double, 6, Eigen::Dynamic> J(6, model_.nv);
    J.setZero();
    pinocchio::computeFrameJacobian(model_, *data_, q, ee_frame_id_,
                                     pinocchio::LOCAL_WORLD_ALIGNED, J);
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
    return ToolPose{T.translation(), T.rotation()};
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
