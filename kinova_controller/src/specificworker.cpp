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
#include <random>
#include <sstream>
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
    arm_belief_viewer_ = std::make_unique<ArmBeliefViewer3D>();
    arm_belief_viewer_->set_mesh_root("/home/pbustos/robocomp/components/webots-kinova/protos/kinova_arm_meshes");
    if (graph_viewers.contains(""))
        graph_viewers.at("")->add_custom_widget_to_dock("beliefs_3d", arm_belief_viewer_.get());

    // Mirror the viewer's checkable Start button: checked → arm EFE engages,
    // unchecked → arm returns to rest and parks.
    connect(arm_belief_viewer_.get(), &ArmBeliefViewer3D::run_state_changed,
            this, [this](bool running) {
                run_requested_ = running;
                std::print("[ui] Run requested = {}\n", running);
            });

    // Optional config: auto-arm the run (as if the Start button were checked)
    // and stream tip-trajectory CSV. Both default off so normal runs are
    // unchanged; flip them in etc/config for headless diagnosis.
    try { run_requested_ = configLoader.get<bool>("Controller.auto_start"); } catch (...) {}
    try { tip_log_        = configLoader.get<bool>("Controller.tip_log");   } catch (...) {}
    if (run_requested_) std::print("[ui] auto_start: run requested from config\n");

    // Rest pose, tunable without recompiling: Controller.rest_pose = "j1 .. j7"
    // (rad). Lets us iterate the "ready over the table, camera-up" posture by
    // editing config.toml and relaunching. Falls back to the hard-coded default.
    try
    {
        std::istringstream iss(configLoader.get<std::string>("Controller.rest_pose"));
        std::array<double, Kinematics::N_ARM_JOINTS> v{};
        int n = 0; double x;
        while (n < Kinematics::N_ARM_JOINTS and (iss >> x)) v[n++] = x;
        if (n == Kinematics::N_ARM_JOINTS)
        {
            rest_pose_angles_ = v;
            std::print("[config] rest_pose = [{:+.3f} {:+.3f} {:+.3f} {:+.3f} {:+.3f} {:+.3f} {:+.3f}]\n",
                       v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
        }
    } catch (...) {}

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

        // Metrics for the chosen rest pose, to tune the posture numerically
        // alongside the Webots view: EE position, upper-arm tilt (0=parallel to
        // table), tool-+Y up-component (camera-up ≈ +1), approach z, and μ.
        {
            const auto p  = kinematics_->tool_pose(rest_pose_angles_);
            const auto sk = kinematics_->arm_skeleton_points(rest_pose_angles_);
            const auto J  = kinematics_->arm_jacobian_full(rest_pose_angles_);
            const double mu = std::sqrt(std::max(0.0, (J * J.transpose()).determinant()));
            std::print("[rest-pose] EE=({:+.3f},{:+.3f},{:+.3f}) upperarm_dz={:+.3f} "
                       "toolY_up={:+.2f} approach_z={:+.2f} mu={:.4f}\n",
                       p.position.x(), p.position.y(), p.position.z(),
                       sk[4].z() - sk[2].z(), p.rotation(2, 1), p.rotation(2, 2), mu);
        }

    }
    catch (const std::exception& e)
    {
        std::print(stderr, "[Kinematics] FATAL: {}\n", e.what());
        throw;
    }

    // DSR sub-APIs for the bottle-approach loop.
    if (G)
    {
        rt_api_          = G->get_rt_api();
        inner_eigen_api_ = G->get_inner_eigen_api();
    }
    else
    {
        std::print(stderr, "[init] WARNING: DSR graph G is null — bottle-approach disabled.\n");
    }
}


void SpecificWorker::update_bottle_pose_in_dsr()
{
    if (not G or not rt_api_) return;

    RoboCompWebots2Robocomp::ObjectPose bottle_w_pose;
    RoboCompWebots2Robocomp::ObjectPose table_w_pose;
    try
    {
        bottle_w_pose = webots2robocomp_proxy->getObjectPose(WEBOTS_BOTTLE_DEF);
        table_w_pose  = webots2robocomp_proxy->getObjectPose(WEBOTS_TABLE_DEF);
        webots_proxy_unreachable_warned_ = false;
    }
    catch (const Ice::Exception& e)
    {
        if (not webots_proxy_unreachable_warned_)
        {
            std::print(stderr, "[bottle] Webots2Robocomp proxy unreachable: {}\n", e.what());
            webots_proxy_unreachable_warned_ = true;
        }
        return;
    }

    // Webots returns mm; DSR stores metres. Both axes are Z-up.
    const Eigen::Vector3d bottle_w(bottle_w_pose.position.x / 1000.0,
                                   bottle_w_pose.position.y / 1000.0,
                                   bottle_w_pose.position.z / 1000.0);
    const Eigen::Vector3d table_w (table_w_pose.position.x  / 1000.0,
                                   table_w_pose.position.y  / 1000.0,
                                   table_w_pose.position.z  / 1000.0);

    const Eigen::Quaterniond q_table_w(table_w_pose.orientation.w,
                                       table_w_pose.orientation.x,
                                       table_w_pose.orientation.y,
                                       table_w_pose.orientation.z);
    const Eigen::Quaterniond q_bottle_w(bottle_w_pose.orientation.w,
                                        bottle_w_pose.orientation.x,
                                        bottle_w_pose.orientation.y,
                                        bottle_w_pose.orientation.z);
    const Eigen::Quaterniond q_table_inv = q_table_w.conjugate();
    const Eigen::Vector3d    bottle_in_table = q_table_inv * (bottle_w - table_w);
    const Eigen::Vector3d    euler_bt =
        (q_table_inv * q_bottle_w).toRotationMatrix().eulerAngles(0, 1, 2);

    auto table_node = G->get_node("table");
    auto bottle_node = G->get_node("bottle");
    if (not table_node.has_value() or not bottle_node.has_value()) return;

    rt_api_->insert_or_assign_edge_RT(
        table_node.value(), bottle_node.value().id(),
        std::vector<float>{static_cast<float>(bottle_in_table.x()),
                           static_cast<float>(bottle_in_table.y()),
                           static_cast<float>(bottle_in_table.z())},
        std::vector<float>{static_cast<float>(euler_bt.x()),
                           static_cast<float>(euler_bt.y()),
                           static_cast<float>(euler_bt.z())});
}


std::optional<SpecificWorker::SideGraspTarget>
SpecificWorker::compute_side_grasp_target()
{
    if (not inner_eigen_api_) return std::nullopt;

    // Full 4×4 robot ← bottle transform; the 3×3 rotation block is what we
    // need to read the bottle's local axes in robot coordinates.
    auto rt_opt = inner_eigen_api_->get_transformation_matrix("robot", "bottle");
    if (not rt_opt.has_value()) return std::nullopt;
    const auto& T_RB = rt_opt.value();   // Eigen::Isometry3d
    const Eigen::Vector3d bottle_pos_R = T_RB.translation();
    const Eigen::Vector3d z_bot_R      = T_RB.linear().col(2).normalized();

    // Radial approach direction in the horizontal plane (robot frame).
    // Reject the radial onto the plane ⟂ z_bot_R so the gripper stays
    // perpendicular to the bottle's long axis even when the bottle is tilted.
    const Eigen::Vector3d radial_xy(bottle_pos_R.x(), bottle_pos_R.y(), 0.0);
    if (radial_xy.norm() < 1e-4) return std::nullopt;  // bottle on robot's z-axis: ambiguous

    Eigen::Vector3d z_tool_des = radial_xy.normalized();
    z_tool_des -= (z_tool_des.dot(z_bot_R)) * z_bot_R;  // project ⟂ bottle z
    if (z_tool_des.norm() < 1e-4) return std::nullopt;
    z_tool_des.normalize();

    // Aim at the bottle's body, not its base. The WaterBottle PROTO origin is
    // bottom-centre, so without this offset the EE drives the wrist down to
    // table level and the arm folds onto the desk.
    double bottle_height_m = 0.22;
    if (auto bottle_node = G->get_node("bottle"); bottle_node.has_value())
        if (auto h = G->get_attrib_by_name<height_m_att>(bottle_node.value()); h.has_value())
            bottle_height_m = h.value();
    const Eigen::Vector3d body_centre =
        bottle_pos_R + z_bot_R * (bottle_height_m * BOTTLE_GRASP_HEIGHT_FRAC);

    // Full grasp frame at convergence:
    //   tool +Z = z_tool_des            approach axis, horizontal into bottle
    //   tool +X = z_bot_R × z_tool_des  horizontal tangent — fingers close
    //                                   horizontally around the bottle body
    //   tool +Y = Z × X = z_bot_R       up (wrist camera on the upper side)
    // x_tool_des is the *correct* (horizontal-finger) target; it only became
    // a "disaster" under the old two-cross-term controller, which flailed on
    // the 90° roll. The geodesic orientation term (gain_secondary>0) now
    // reaches it as one coordinated motion. See efe_gradient.cpp §3(b).
    SideGraspTarget out;
    out.z_tool_des    = z_tool_des;
    out.x_tool_des    = z_bot_R.cross(z_tool_des).normalized();
    out.grasp_pos     = body_centre;                              // the grasp point
    out.stand_off_pos = body_centre - z_tool_des * APPROACH_STANDOFF_M;
    return out;
}



void SpecificWorker::update_viewer_scene_objects()
{
    if (not arm_belief_viewer_ or not G or not inner_eigen_api_) return;

    // Table size lives on the table node (width_m, depth_m, height_m).
    auto table_node = G->get_node("table");
    if (not table_node.has_value()) return;
    const auto w_opt = G->get_attrib_by_name<width_m_att>(table_node.value());
    const auto d_opt = G->get_attrib_by_name<depth_m_att>(table_node.value());
    const auto h_opt = G->get_attrib_by_name<height_m_att>(table_node.value());
    if (not w_opt.has_value() or not d_opt.has_value() or not h_opt.has_value()) return;
    const double hw = w_opt.value() * 0.5;
    const double hd = d_opt.value() * 0.5;
    const double h  = h_opt.value();

    // Box corners in the table frame. The table origin is bottom-center
    // (matches Webots/PROTO convention and the DSR robot→table = [0,0,-0.7]
    // translation, which places the table FLOOR 0.7 m below the arm base so
    // the table TOP at +h coincides with the arm base at robot z = 0).
    // Order: bottom CCW (0..3), then top CCW (4..7).
    const std::array<Eigen::Vector3d, 8> corners_in_table{{
        {-hw, -hd, 0}, {+hw, -hd, 0}, {+hw, +hd, 0}, {-hw, +hd, 0},
        {-hw, -hd, h}, {+hw, -hd, h}, {+hw, +hd, h}, {-hw, +hd, h},
    }};

    std::vector<Eigen::Vector3d> corners_in_robot;
    corners_in_robot.reserve(8);
    for (const auto& c : corners_in_table)
    {
        auto p = inner_eigen_api_->transform("robot", c, "table");
        if (not p.has_value()) return;
        corners_in_robot.push_back(p.value());
    }

    // Bottle pose in robot frame: origin (translation) + long axis (Z column).
    auto T_RB_opt = inner_eigen_api_->get_transformation_matrix("robot", "bottle");
    if (not T_RB_opt.has_value()) return;
    const Eigen::Vector3d bottle_origin_R = T_RB_opt.value().translation();
    const Eigen::Vector3d bottle_axis_R   = T_RB_opt.value().linear().col(2);

    // Bottle node carries width_m (diameter) and height_m. Defaults if absent
    // so the viewer still draws something the first time the node is created.
    auto bottle_node = G->get_node("bottle");
    double diameter = 0.07, height = 0.22;
    if (bottle_node.has_value())
    {
        if (auto v = G->get_attrib_by_name<width_m_att>(bottle_node.value());  v.has_value()) diameter = v.value();
        if (auto v = G->get_attrib_by_name<height_m_att>(bottle_node.value()); v.has_value()) height   = v.value();
    }

    arm_belief_viewer_->update_scene_objects(corners_in_robot, bottle_origin_R,
                                             bottle_axis_R, diameter * 0.5, height);
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
        std::array<double, Kinematics::N_ARM_JOINTS> tau{};
        for (int i = 0; i < Kinematics::N_ARM_JOINTS; ++i)
        {
            q[i]   = js.joints[i].angle;
            tau[i] = js.joints[i].torque;     // motor torque feedback from the bridge
        }

        // Wrist 6-axis wrench estimate from joint torques (Webots has no native
        // F/T node): w = (J Jᵀ + λ²I)⁻¹ J (g(q) − τ), in the base-aligned frame
        // at the tool point. Logged for now (not yet consumed by the FSM); a
        // no-contact reading is the bias to tare against. See EFE_CONTROLLER_MATH.md.
        if (tip_log_)
        {
            const Eigen::Matrix<double, 6, 1> w = kinematics_->estimate_tool_wrench(q, tau);
            std::print("[wrench] F=({:.2f},{:.2f},{:.2f}) N  T=({:.3f},{:.3f},{:.3f}) N·m\n",
                       w(0), w(1), w(2), w(3), w(4), w(5));
        }

        // Drive the gripper to whatever the current FSM sub-state wants.
        // setGripperPos takes [0, 1] where 1 = fully open in the bridge.
        // Wrapped in its own try so a gripper-side fault doesn't abort the
        // joint-velocity loop above.
        try
        {
            kinovaarm_proxy->setGripperPos(gripper_command_);
            gripper_proxy_warned_ = false;
        }
        catch (const Ice::Exception& e)
        {
            if (not gripper_proxy_warned_)
            {
                std::print(stderr, "[gripper] setGripperPos failed: {}\n", e.what());
                gripper_proxy_warned_ = true;
            }
        }

        const auto mesh_link_poses = kinematics_->arm_mesh_link_poses(q);
        std::vector<ArmBeliefViewer3D::LinkPose> viewer_link_poses;
        viewer_link_poses.reserve(mesh_link_poses.size());
        for (const auto& lp : mesh_link_poses)
            viewer_link_poses.push_back({lp.mesh_filename, lp.pose});
        const auto ee_position = kinematics_->forward_kinematics(q);
        if (arm_belief_viewer_)
            arm_belief_viewer_->update_beliefs(q, viewer_link_poses, reach_target_, ee_position);

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
            // Command continuous joints to the equivalent of the rest angle
            // NEAREST the current encoder value, so the motor doesn't unwind
            // accumulated revolutions (j5 can read +26 rad ≈ 4 turns). Bounded
            // joints (finite limits) are commanded as-is.
            constexpr double TWO_PI = 6.283185307179586;
            const auto lims = kinematics_->arm_joint_position_limits();
            RoboCompKinovaArm::TJointAngles target;
            target.jointAngles.resize(Kinematics::N_ARM_JOINTS);
            for (int i = 0; i < Kinematics::N_ARM_JOINTS; ++i)
            {
                double t = rest_pose_angles_[i];
                if (not std::isfinite(lims[i].first) or not std::isfinite(lims[i].second))
                    t += std::round((q[i] - t) / TWO_PI) * TWO_PI;   // nearest equivalent
                target.jointAngles[i] = static_cast<float>(t);
            }
            kinovaarm_proxy->moveJointsWithAngle(target);
            std::print("[homing] Sent rest pose [{:+.3f} {:+.3f} {:+.3f} {:+.3f} {:+.3f} {:+.3f} {:+.3f}] rad\n",
                       rest_pose_angles_[0], rest_pose_angles_[1], rest_pose_angles_[2],
                       rest_pose_angles_[3], rest_pose_angles_[4], rest_pose_angles_[5],
                       rest_pose_angles_[6]);
            // Reset the settle counter so a re-entry from ActiveEFE (button
            // unchecked) re-runs the full convergence check from scratch.
            homing_settled_ticks_ = 0;
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
                if (returning_for_cycle_ and run_requested_)
                {
                    // Autonomous loop: rest reached after a place — pick again.
                    returning_for_cycle_ = false;
                    grasp_phase_         = GraspPhase::Tracking;
                    grasp_settle_ticks_  = 0;
                    gripper_command_     = 1.0f;
                    phase_               = Phase::ActiveEFE;
                    std::print("[cycle] rest reached → starting next pick-and-place\n");
                }
                else
                {
                    returning_for_cycle_ = false;
                    std::print("[homing] Rest pose reached — waiting for Start button.\n");
                    phase_ = Phase::WaitingForStart;
                }
            }
            proxy_unreachable_warned_ = false;
            return;   // do NOT run EFE while homing
        }

        // While idle or waiting for Start: keep the DSR bottle pose fresh and
        // draw the scene, so the user can see where the bottle is before
        // committing to the approach.
        update_bottle_pose_in_dsr();
        update_viewer_scene_objects();

        if (phase_ == Phase::WaitingForStart)
        {
            // Hold the arm still until the user toggles Start.
            RoboCompKinovaArm::TJointSpeeds stop;
            stop.jointSpeeds.assign(Kinematics::N_ARM_JOINTS, 0.0f);
            kinovaarm_proxy->moveJointsWithSpeed(stop);
            if (run_requested_)
            {
                std::print("[start] Entering EFE bottle-approach.\n");
                phase_              = Phase::ActiveEFE;
                grasp_phase_        = GraspPhase::Tracking;  // fresh grasp sequence
                grasp_settle_ticks_ = 0;
                gripper_command_    = 1.0f;                  // open
            }
            proxy_unreachable_warned_ = false;
            return;
        }

        // If the user toggled the button off mid-approach, abandon EFE and
        // command joint-space return to the rest pose. The existing
        // SendingRestPose → Homing → WaitingForStart loop handles the rest.
        if (not run_requested_)
        {
            std::print("[stop] Run unchecked — returning to rest pose.\n");
            arrived_logged_      = false;
            grasp_phase_         = GraspPhase::Tracking;  // reset FSM
            grasp_settle_ticks_  = 0;
            returning_for_cycle_ = false;
            gripper_command_     = 1.0f;                  // release / open
            phase_               = Phase::SendingRestPose;
            return;
        }

        // ── Phase::ActiveEFE: grasp FSM ─────────────────────────────────────
        // Hierarchical active inference: each GraspPhase installs a preferred
        // pose + gripper state that the EFE controller realises; transitions
        // fire when the lower level converges (error inside the deadband) or a
        // finger-force observation confirms the grasp. See specificworker.h.

        // Build the shared EFE parameters; per-state code varies only the
        // target, the desired tool axes, and the cruise speed.
        const auto make_params = [](const Eigen::Vector3d& z_des,
                                    const Eigen::Vector3d& x_des,
                                    double v_app)
        {
            EFEParams p;
            p.desired_approach  = z_des;
            p.desired_secondary = x_des;
            p.gain_secondary    = 1.0;
            p.C_pos             = Eigen::Vector3d::Ones();  // straight-line flow
            p.dls_lambda        = 0.05;
            p.v_approach        = v_app;
            p.a_approach        = 0.60;
            p.omega_max         = 2.0;
            // Manipulability null-space ascent: keep the arm in the dexterous
            // region (with the new rest pose) through the whole pick-and-place,
            // so it doesn't drift into the near-singular contortions that tip
            // the bottle. Spends only the redundant DOF — can't disturb the pose.
            p.gain_mu           = 0.3;
            return p;
        };

        // Run one coordinated EFE step toward (target, z_des, x_des) at v_app,
        // send it, and return {position error, orientation geodesic angle}.
        const auto drive = [&](const Eigen::Vector3d& target,
                               const Eigen::Vector3d& z_des,
                               const Eigen::Vector3d& x_des,
                               double v_app) -> std::pair<double, double>
        {
            reach_target_ = target;
            const EFEParams params = make_params(z_des, x_des, v_app);
            const auto q_dot = efe_gradient_step(*kinematics_, q, reach_target_, params);
            RoboCompKinovaArm::TJointSpeeds cmd;
            cmd.jointSpeeds.assign(q_dot.begin(), q_dot.end());
            kinovaarm_proxy->moveJointsWithSpeed(cmd);

            const double e_pos = (ee_position - target).norm();

            // Orientation residual: geodesic angle between the current tool
            // frame and R_des = [x⟂, z×x, z], matching efe_gradient's metric.
            double e_ang = 3.1416;
            const Eigen::Vector3d zc = z_des.normalized();
            Eigen::Vector3d xc = x_des - x_des.dot(zc) * zc;
            if (xc.norm() > 1e-6)
            {
                xc.normalize();
                Eigen::Matrix3d R_des;
                R_des.col(0) = xc;
                R_des.col(1) = zc.cross(xc);
                R_des.col(2) = zc;
                const auto tool = kinematics_->tool_pose(q);
                const Eigen::AngleAxisd er(R_des * tool.rotation.transpose());
                e_ang = std::abs(er.angle());
            }

            if (tip_log_)
            {
                const double dt   = std::max(1, getPeriod("Compute")) / 1000.0;
                const double vcmd = std::min(params.v_approach,
                    std::sqrt(2.0 * params.a_approach * std::max(0.0, e_pos - params.arrive_deadband)));
                const double vmeas = tip_log_prev_pos_.has_value()
                    ? (ee_position - tip_log_prev_pos_.value()).norm() / dt : 0.0;
                std::print("[tiplog] {},{:.3f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n",
                           tip_log_cycle_, tip_log_cycle_ * dt,
                           ee_position.x(), ee_position.y(), ee_position.z(),
                           target.x(), target.y(), target.z(), e_pos, vcmd, vmeas);
                tip_log_prev_pos_ = ee_position;
                ++tip_log_cycle_;
            }
            return {e_pos, e_ang};
        };

        // Largest finger/tip force from the gripper; 0 if the proxy hiccups.
        const auto gripper_force = [&]() -> float
        {
            try
            {
                const auto gs = kinovaarm_proxy->getGripperState();
                return std::max({gs.lforce, gs.rforce, gs.ltipforce, gs.rtipforce});
            }
            catch (const Ice::Exception&) { return 0.0f; }
        };

        switch (grasp_phase_)
        {
            case GraspPhase::Tracking:
            {
                // Approach the LIVE standoff, gripper open, bottle pose tracked.
                auto grasp_opt = compute_side_grasp_target();
                if (not grasp_opt.has_value())
                {
                    RoboCompKinovaArm::TJointSpeeds stop;
                    stop.jointSpeeds.assign(Kinematics::N_ARM_JOINTS, 0.0f);
                    kinovaarm_proxy->moveJointsWithSpeed(stop);
                    return;
                }
                const auto& g = grasp_opt.value();
                gripper_command_ = 1.0f;  // open
                const auto [e_pos, e_ang] =
                    drive(g.stand_off_pos, g.z_tool_des, g.x_tool_des, 0.35);

                if (e_pos < REACH_TOLERANCE_M and e_ang < GRASP_ALIGN_TOL_RAD)
                {
                    if (++grasp_settle_ticks_ >= GRASP_SETTLE_TICKS)
                    {
                        // Commit: latch the grasp frame so contact can't make
                        // the target chase its own disturbance.
                        latched_grasp_ = g;
                        lift_target_   = g.grasp_pos + Eigen::Vector3d(0.0, 0.0, LIFT_HEIGHT_M);
                        grasp_phase_   = GraspPhase::Inserting;
                        grasp_settle_ticks_ = 0;
                        std::print("[grasp] standoff settled (e={:.3f} m, {:.1f}°) → Inserting\n",
                                   e_pos, e_ang * 57.29578);
                    }
                }
                else grasp_settle_ticks_ = 0;
                break;
            }
            case GraspPhase::Inserting:
            {
                // Ease into the bottle body along the latched approach axis.
                const auto& g = latched_grasp_;
                gripper_command_ = 1.0f;  // stay open
                const auto [e_pos, e_ang] =
                    drive(g.grasp_pos, g.z_tool_des, g.x_tool_des, INSERT_VEL_MS);
                (void) e_ang;
                const float f = gripper_force();
                if (e_pos < REACH_TOLERANCE_M or f > GRASP_FORCE_THRESH)
                {
                    grasp_phase_   = GraspPhase::Closing;
                    closing_ticks_ = 0;
                    std::print("[grasp] at grasp point (e={:.3f} m, f={:.2f}) → Closing\n", e_pos, f);
                }
                break;
            }
            case GraspPhase::Closing:
            {
                // Hold the grasp pose (deadband keeps it still) and close.
                const auto& g = latched_grasp_;
                gripper_command_ = 0.0f;  // close
                drive(g.grasp_pos, g.z_tool_des, g.x_tool_des, INSERT_VEL_MS);
                const float f = gripper_force();
                if (tip_log_ and closing_ticks_ % 10 == 0)
                    std::print("[grasp] closing… f={:.3f} (thresh {:.2f})\n", f, GRASP_FORCE_THRESH);
                if (f > GRASP_FORCE_THRESH)
                {
                    grasp_phase_ = GraspPhase::Lifting;
                    std::print("[grasp] grasped (f={:.2f}) → Lifting\n", f);
                }
                else if (++closing_ticks_ > CLOSING_TIMEOUT_TICKS)
                {
                    std::print("[grasp] MISS — no contact in {} cycles → reopen, back to Tracking\n",
                               CLOSING_TIMEOUT_TICKS);
                    gripper_command_    = 1.0f;  // reopen
                    grasp_phase_        = GraspPhase::Tracking;
                    grasp_settle_ticks_ = 0;
                }
                break;
            }
            case GraspPhase::Lifting:
            {
                const auto& g = latched_grasp_;
                gripper_command_ = 0.0f;  // keep holding the bottle
                const auto [e_pos, e_ang] =
                    drive(lift_target_, g.z_tool_des, g.x_tool_des, 0.20);
                (void) e_ang;
                if (e_pos < REACH_TOLERANCE_M)
                {
                    // Pick a random reachable table spot ≥ PLACE_MIN_MOVE_M from
                    // the pick point; z = grasp height so the base lands on the
                    // table. Orientation stays latched (pure translation keeps
                    // the bottle upright).
                    std::uniform_real_distribution<double> ux(PLACE_X_MIN, PLACE_X_MAX);
                    std::uniform_real_distribution<double> uy(PLACE_Y_MIN, PLACE_Y_MAX);
                    Eigen::Vector3d p;
                    for (int k = 0; k < 10; ++k)
                    {
                        p = Eigen::Vector3d(ux(rng_), uy(rng_), g.grasp_pos.z());
                        if ((p.head<2>() - g.grasp_pos.head<2>()).norm() > PLACE_MIN_MOVE_M) break;
                    }
                    place_pos_   = p;
                    place_hover_ = p + Eigen::Vector3d(0.0, 0.0, LIFT_HEIGHT_M);
                    // Re-point the approach radially toward the place spot, +Y up
                    // (bottle upright). Yaw-only change from the grasp frame, so
                    // the arm reaches the spot comfortably instead of contorting.
                    place_z_des_ = Eigen::Vector3d(place_pos_.x(), place_pos_.y(), 0.0).normalized();
                    place_x_des_ = Eigen::Vector3d(0.0, 0.0, 1.0).cross(place_z_des_).normalized();
                    grasp_phase_ = GraspPhase::PlaceMoving;
                    place_ticks_ = 0;
                    std::print("[place] lifted → carry to ({:.3f}, {:.3f}) → PlaceMoving\n",
                               place_pos_.x(), place_pos_.y());
                }
                break;
            }
            case GraspPhase::PlaceMoving:
            {
                gripper_command_ = 0.0f;  // hold the bottle
                const auto [e_pos, e_ang] =
                    drive(place_hover_, place_z_des_, place_x_des_, 0.20);
                (void) e_ang;
                if (e_pos < REACH_TOLERANCE_M or ++place_ticks_ > PLACE_TIMEOUT_TICKS)
                {
                    grasp_phase_ = GraspPhase::PlaceLowering;
                    place_ticks_ = 0;
                    std::print("[place] above spot → PlaceLowering\n");
                }
                break;
            }
            case GraspPhase::PlaceLowering:
            {
                gripper_command_ = 0.0f;  // still holding
                const auto [e_pos, e_ang] =
                    drive(place_pos_, place_z_des_, place_x_des_, 0.08);  // gentle set-down
                (void) e_ang;
                if (e_pos < REACH_TOLERANCE_M or ++place_ticks_ > PLACE_TIMEOUT_TICKS)
                {
                    grasp_phase_ = GraspPhase::PlaceReleasing;
                    place_ticks_ = 0;
                    std::print("[place] on table → PlaceReleasing (open)\n");
                }
                break;
            }
            case GraspPhase::PlaceReleasing:
            {
                gripper_command_ = 1.0f;  // open — let the bottle go
                drive(place_pos_, place_z_des_, place_x_des_, 0.05);  // hold still while opening
                if (++place_ticks_ > RELEASE_TICKS)
                {
                    grasp_phase_ = GraspPhase::PlaceRetreating;
                    place_ticks_ = 0;
                    std::print("[place] released → PlaceRetreating\n");
                }
                break;
            }
            case GraspPhase::PlaceRetreating:
            {
                gripper_command_ = 1.0f;  // stay open
                const auto [e_pos, e_ang] =
                    drive(place_hover_, place_z_des_, place_x_des_, 0.20);
                (void) e_ang;
                if (e_pos < REACH_TOLERANCE_M or ++place_ticks_ > PLACE_TIMEOUT_TICKS)
                {
                    // Return to rest via the outer Homing path, then auto-restart
                    // the whole pick-and-place (handled in the Homing branch).
                    RoboCompKinovaArm::TJointAngles target;
                    target.jointAngles.assign(rest_pose_angles_.begin(), rest_pose_angles_.end());
                    kinovaarm_proxy->moveJointsWithAngle(target);
                    returning_for_cycle_  = true;
                    homing_settled_ticks_ = 0;
                    grasp_phase_          = GraspPhase::Tracking;  // for the next cycle
                    place_ticks_          = 0;
                    phase_                = Phase::Homing;
                    std::print("[place] retreated → returning to rest, then new cycle\n");
                    return;
                }
                break;
            }
        }
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

