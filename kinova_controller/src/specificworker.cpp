/*
 *    Copyright (C) 2026 by YOUR NAME HERE
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    RoboComp is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "specificworker.h"

#include <cmath>
#include <print>
#include <stdexcept>

namespace
{
    // Hardcoded for the first sanity test.  Move to etc/config once we have
    // confirmed Pinocchio loads the URDF correctly.
    constexpr auto URDF_PATH =
        "/home/pbustos/robocomp/components/active_inference/kinova_controller/gen3_robotiq_2f_85-mod.urdf";

    /// Shortest angular distance |a − b| modulo 2π, in [0, π].
    /// Needed because continuous joints accumulate revolutions across runs
    /// (the Webots encoder can report +8.6 rad even though physically the
    /// joint is at +8.6 − 2π = +2.32 rad).
    inline double angular_distance(double a, double b)
    {
        return std::abs(std::atan2(std::sin(a - b), std::cos(a - b)));
    }
}

SpecificWorker::SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check) : GenericWorker(configLoader, tprx)
{
	this->startup_check_flag = startup_check;
	if(this->startup_check_flag)
	{
		this->startup_check();
	}
	else
	{
		#ifdef HIBERNATION_ENABLED
			hibernationChecker.start(500);
		#endif

		statemachine.setChildMode(QState::ExclusiveStates);
		statemachine.start();

		auto error = statemachine.errorString();
		if (error.length() > 0){
			qWarning() << error;
			throw error;
		}
	}
}

SpecificWorker::~SpecificWorker()
{
	std::cout << "Destroying SpecificWorker" << std::endl;

	// Safety: send zero velocities so the arm halts when the agent exits.
	// Without this the bridge holds the last commanded q̇ and the arm keeps
	// drifting freely in Webots after Ctrl+C. Wrapped in try/catch because
	// the proxy may already be unreachable on shutdown (bridge closed first,
	// network gone, etc.) — in that case there is nothing we can do.
	try
	{
		RoboCompKinovaArm::TJointSpeeds stop;
		stop.jointSpeeds.assign(Kinematics::N_ARM_JOINTS, 0.0f);
		kinovaarm_proxy->moveJointsWithSpeed(stop);
		std::cout << "[shutdown] Sent zero velocities to arm." << std::endl;
	}
	catch (const Ice::Exception& e)
	{
		std::cerr << "[shutdown] Could not send stop command: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::cerr << "[shutdown] Unknown error sending stop command." << std::endl;
	}
}


void SpecificWorker::initialize()
{
    std::cout << "initialize worker" << std::endl;
	GenericWorker::initialize();

	//Subscription to DSR graph update signals. 
	// If multiple graphs exist, it is necessary to specify the graph name 
	// using 'Graphs.at("name")' to connect its signals to the Worker's slots.
	//connect(Graphs.at("").get(), &DSR::DSRGraph::update_node_signal, this, &SpecificWorker::modify_node_slot);
	//connect(Graphs.at("").get(), &DSR::DSRGraph::update_edge_signal, this, &SpecificWorker::modify_edge_slot);
	//connect(Graphs.at("").get(), &DSR::DSRGraph::update_node_attr_signal, this, &SpecificWorker::modify_node_attrs_slot);
	//connect(Graphs.at("").get(), &DSR::DSRGraph::update_edge_attr_signal, this, &SpecificWorker::modify_edge_attrs_slot);
	//connect(Graphs.at("").get(), &DSR::DSRGraph::del_edge_signal, this, &SpecificWorker::del_edge_slot);
	//connect(Graphs.at("").get(), &DSR::DSRGraph::del_node_signal, this, &SpecificWorker::del_node_slot);

	/***
	Custom Widget
	In addition to the predefined viewers, Graph Viewer allows you to add various widgets designed by the developer.
	The add_custom_widget_to_dock method is used. This widget can be defined like any other Qt widget,
	either with a QtDesigner or directly from scratch in a class of its own.
	The add_custom_widget_to_dock method receives a name for the widget and a reference to the class instance.
	***/
	//If you have more than one graph, you need to connect to the specific graph with the name
	//graph_viewers.at("")->add_custom_widget_to_dock("CustomWidget", &custom_widget);

    //initializeCODE
    /////////GET PARAMS, OPEND DEVICES....////////
    //int period = configLoader.get<int>("Period.Compute") //NOTE: If you want get period of compute use getPeriod("compute")
    //std::string device = configLoader.get<std::string>("Device.name")

    // Sanity-load the URDF via Pinocchio. Prints model summary + tool_frame
    // position at the neutral configuration. No control loop yet.
    try
    {
        kinematics_ = std::make_unique<Kinematics>(URDF_PATH);
        kinematics_->print_info();
        const auto ee_neutral = kinematics_->forward_kinematics_neutral();
        std::print("[Kinematics] tool_frame at neutral config (world frame): "
                   "[{:+.4f}, {:+.4f}, {:+.4f}] m\n",
                   ee_neutral.x(), ee_neutral.y(), ee_neutral.z());
    }
    catch (const std::exception& e)
    {
        std::print(stderr, "[Kinematics] FATAL: {}\n", e.what());
        throw;
    }
}


void SpecificWorker::compute()
{
    // EFE-reach test loop.
    //   1. read current arm joint angles via the KinovaArm proxy
    //   2. compute q̇ = −α J_linᵀ (f(q) − x*) via efe_gradient_step
    //   3. send q̇ back as TJointSpeeds
    //   4. log end-effector error
    // Wrapped in try/catch so the agent survives if WebotsBridge isn't up
    // yet — we just suppress the spam after the first warning.
    if (not kinematics_)
        return;

    try
    {
        const auto js = kinovaarm_proxy->getJointsState();
        if (static_cast<int>(js.joints.size()) < Kinematics::N_ARM_JOINTS)
        {
            std::print("[compute] proxy returned only {} joints; expected ≥ {}\n",
                       js.joints.size(), Kinematics::N_ARM_JOINTS);
            return;
        }

        std::array<double, Kinematics::N_ARM_JOINTS> q{};
        for (int i = 0; i < Kinematics::N_ARM_JOINTS; ++i)
            q[i] = js.joints[i].angle;

        if (joint_dump_pending_)
        {
            std::print("[joint-dump] received {} joints from KinovaArm proxy\n", js.joints.size());
            for (size_t i = 0; i < js.joints.size(); ++i)
                std::print("  [{}] id={} angle={:+.4f} rad  velocity={:+.4f}  torque={:+.3f}\n",
                           i, js.joints[i].id, js.joints[i].angle,
                           js.joints[i].velocity, js.joints[i].torque);
            const auto ee_at_received = kinematics_->forward_kinematics(q);
            std::print("[joint-dump] FK at received angles: tool_frame=[{:+.4f}, {:+.4f}, {:+.4f}] m\n",
                       ee_at_received.x(), ee_at_received.y(), ee_at_received.z());
            std::print("[joint-dump] Cross-check this position against Webots' viewer.\n"
                       "             If they disagree we have a base-frame or sign-convention mismatch.\n");
            joint_dump_pending_ = false;
        }

        // ── Lifecycle: home to rest pose, then run EFE ──────────────────────
        if (phase_ == Phase::SendingRestPose)
        {
            RoboCompKinovaArm::TJointAngles target;
            target.jointAngles.assign(rest_pose_angles_.begin(), rest_pose_angles_.end());
            kinovaarm_proxy->moveJointsWithAngle(target);
            std::print("[homing] Sent rest pose [{:+.3f} {:+.3f} {:+.3f} {:+.3f} {:+.3f} {:+.3f} {:+.3f}] rad\n",
                       rest_pose_angles_[0], rest_pose_angles_[1], rest_pose_angles_[2],
                       rest_pose_angles_[3], rest_pose_angles_[4], rest_pose_angles_[5],
                       rest_pose_angles_[6]);
            phase_ = Phase::Homing;
            proxy_unreachable_warned_ = false;
            return;
        }

        if (phase_ == Phase::Homing)
        {
            double max_err = 0.0;
            for (int i = 0; i < Kinematics::N_ARM_JOINTS; ++i)
                max_err = std::max(max_err, angular_distance(q[i], rest_pose_angles_[i]));
            if (max_err < HOMING_TOLERANCE_RAD)
                ++homing_settled_ticks_;
            else
                homing_settled_ticks_ = 0;

            std::print("[homing] max joint err = {:.4f} rad  ({}/{} settled)\n",
                       max_err, homing_settled_ticks_, HOMING_SETTLE_TICKS);

            if (homing_settled_ticks_ >= HOMING_SETTLE_TICKS)
            {
                std::print("[homing] Rest pose reached — switching to EFE control.\n");
                phase_ = Phase::ActiveEFE;
            }
            proxy_unreachable_warned_ = false;
            return;   // do NOT run EFE while homing
        }

        // ── EFE-reach control (Phase::ActiveEFE) ────────────────────────────
        const auto q_dot = efe_gradient_step(*kinematics_, q, reach_target_);

        RoboCompKinovaArm::TJointSpeeds cmd;
        cmd.jointSpeeds.assign(q_dot.begin(), q_dot.end());
        kinovaarm_proxy->moveJointsWithSpeed(cmd);

        const auto pose = kinematics_->tool_pose(q);
        const Eigen::Vector3d x_ee = pose.position;
        const Eigen::Vector3d z_tool = pose.rotation.col(2);
        const double err_pos = (x_ee - reach_target_).norm();
        // Angle (deg) between tool z-axis and (0,0,−1).
        const double cos_align = std::clamp(z_tool.z() * -1.0, -1.0, 1.0);
        const double err_orient_deg = std::acos(cos_align) * 180.0 / M_PI;
        std::print("[reach] x_ee=[{:+.3f} {:+.3f} {:+.3f}]  z_tool=[{:+.2f} {:+.2f} {:+.2f}]  "
                   "err_pos={:.4f} m  err_orient={:.2f}°  |q̇|={:.3f}\n",
                   x_ee.x(), x_ee.y(), x_ee.z(),
                   z_tool.x(), z_tool.y(), z_tool.z(),
                   err_pos, err_orient_deg,
                   Eigen::Map<const Eigen::Matrix<double, Kinematics::N_ARM_JOINTS, 1>>(q_dot.data()).norm());
        proxy_unreachable_warned_ = false;
    }
    catch (const Ice::Exception& e)
    {
        if (not proxy_unreachable_warned_)
        {
            std::print(stderr, "[compute] KinovaArm proxy unreachable: {}\n", e.what());
            std::print(stderr, "[compute] (suppressing further proxy errors until reconnect)\n");
            proxy_unreachable_warned_ = true;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////

void SpecificWorker::emergency()
{
    std::cout << "Emergency worker" << std::endl;
    //emergencyCODE
    //
    //if (SUCCESSFUL) //The componet is safe for continue
    //  emmit goToRestore()
}


//Execute one when exiting to emergencyState
void SpecificWorker::restore()
{
    std::cout << "Restore worker" << std::endl;
    //restoreCODE
    //Restore emergency component

}


int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, QCoreApplication::instance(), SLOT(quit()));
	return 0;
}



/**************************************/
// From the RoboCompKinovaArm you can call this methods:
// RoboCompKinovaArm::bool this->kinovaarm_proxy->closeGripper()
// RoboCompKinovaArm::TPose this->kinovaarm_proxy->getCenterOfTool(ArmJoints referencedTo)
// RoboCompKinovaArm::TGripper this->kinovaarm_proxy->getGripperState()
// RoboCompKinovaArm::TJoints this->kinovaarm_proxy->getJointsState()
// RoboCompKinovaArm::TToolInfo this->kinovaarm_proxy->getToolInfo()
// RoboCompKinovaArm::void this->kinovaarm_proxy->moveJointsWithAngle(TJointAngles angles)
// RoboCompKinovaArm::void this->kinovaarm_proxy->moveJointsWithSpeed(TJointSpeeds speeds)
// RoboCompKinovaArm::void this->kinovaarm_proxy->openGripper()
// RoboCompKinovaArm::void this->kinovaarm_proxy->setCenterOfTool(TPose pose, ArmJoints referencedTo)
// RoboCompKinovaArm::bool this->kinovaarm_proxy->setGripperPos(float pos)

/**************************************/
// From the RoboCompKinovaArm you can use this types:
// RoboCompKinovaArm::TPose
// RoboCompKinovaArm::TAxis
// RoboCompKinovaArm::TToolInfo
// RoboCompKinovaArm::TGripper
// RoboCompKinovaArm::TJoint
// RoboCompKinovaArm::TJoints
// RoboCompKinovaArm::TJointSpeeds
// RoboCompKinovaArm::TJointAngles

