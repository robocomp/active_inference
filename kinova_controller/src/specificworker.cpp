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
#include <chrono>
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

    /// Minimum distance between two 3D segments [p1,p2] and [q1,q2].
    /// Used to measure arm-link clearance to the (vertical) column segment.
    double segment_segment_distance(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
                                    const Eigen::Vector3d& q1, const Eigen::Vector3d& q2)
    {
        const Eigen::Vector3d d1 = p2 - p1, d2 = q2 - q1, r = p1 - q1;
        const double a = d1.dot(d1), e = d2.dot(d2), f = d2.dot(r);
        double s, t;
        const double c = d1.dot(r);
        const double b = d1.dot(d2);
        const double den = a * e - b * b;
        s = (den > 1e-12) ? std::clamp((b * f - c * e) / den, 0.0, 1.0) : 0.0;
        t = (e > 1e-12) ? (b * s + f) / e : 0.0;
        if (t < 0.0) { t = 0.0; s = std::clamp(-c / a, 0.0, 1.0); }
        else if (t > 1.0) { t = 1.0; s = std::clamp((b - c) / a, 0.0, 1.0); }
        return ((p1 + s * d1) - (q1 + t * d2)).norm();
    }

    /// Shortest angular distance |a − b| modulo 2π, in [0, π].
    /// Needed because continuous joints accumulate revolutions across runs
    /// (the Webots encoder can report +8.6 rad even though physically the
    /// joint is at +8.6 − 2π = +2.32 rad).
    inline double angular_distance(double a, double b)
    {
        return std::abs(std::atan2(std::sin(a - b), std::cos(a - b)));
    }

    /// Halton low-discrepancy sequence value in [0,1) for the given index/base.
    /// Used to spread the per-rep grasp perturbations EVENLY over the probe
    /// envelope (deterministic, reproducible coverage — better identifiability
    /// than i.i.d. random, which clumps and leaves gaps over a short round).
    inline double halton(long index, int base)
    {
        double f = 1.0, r = 0.0;
        long i = index + 1;                 // skip 0 (which is always 0)
        while (i > 0) { f /= base; r += f * (i % base); i /= base; }
        return r;
    }

    /// Monotonic wall-clock seconds, for per-rep cycle-time metrics.
    inline double now_seconds()
    {
        using namespace std::chrono;
        return duration<double>(steady_clock::now().time_since_epoch()).count();
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
    try { approach_only_  = configLoader.get<bool>("Controller.approach_only"); } catch (...) {}
    try { round_cycles_   = configLoader.get<int>("Controller.round_cycles");   } catch (...) {}
    if (round_cycles_ > 0) std::print("[ui] round_cycles: stop after {} pick-and-place cycles\n", round_cycles_);
    if (run_requested_) std::print("[ui] auto_start: run requested from config\n");
    if (approach_only_) std::print("[ui] approach_only: FSM will hold at the standoff\n");

    // Skill-learning probe + dataset (all optional; default OFF = unchanged behaviour).
    try { probe_enabled_   = configLoader.get<bool>  ("Controller.probe_variations"); } catch (...) {}
    try { probe_pos_amp_   = configLoader.get<double>("Controller.probe_pos_amp");     } catch (...) {}
    try { probe_azi_amp_   = configLoader.get<double>("Controller.probe_azi_amp");     } catch (...) {}
    try { probe_speed_amp_ = configLoader.get<double>("Controller.probe_speed_amp");   } catch (...) {}
    try { respawn_each_rep_ = configLoader.get<bool>("Controller.respawn_each_rep");    } catch (...) {}
    if (respawn_each_rep_)
        std::print("[spawn] per-rep bottle respawn ON (deterministic pickup, survives falls)\n");
    if (probe_enabled_)
        std::print("[probe] structured grasp perturbations ON  (pos ±{:.3f} m, azi ±{:.3f} rad, speed ±{:.0f}%)\n",
                   probe_pos_amp_, probe_azi_amp_, probe_speed_amp_ * 100.0);
    try
    {
        const auto path = configLoader.get<std::string>("Controller.dataset_path");
        if (not path.empty())
        {
            dataset_.open(path, std::ios::out | std::ios::app);
            if (dataset_.is_open())
            {
                dataset_open_ = true;
                // Header only if the file is new/empty.
                if (dataset_.tellp() == std::streampos(0))
                    dataset_ << "probe_idx,rep,success,dx_perp,dz_axis,dazi,speed_scale,"
                                "gx,gy,gz,bx,by,bz,axz,commit_epos,commit_eang,track_ticks,"
                                "bottle_rise,xy_gap\n";
                std::print("[probe] per-rep dataset → {}\n", path);
            }
            else std::print("[probe] WARN could not open dataset {}\n", path);
        }
    } catch (...) {}
    try
    {
        const auto jpath = configLoader.get<std::string>("Controller.joint_log_path");
        if (not jpath.empty())
        {
            joint_log_.open(jpath, std::ios::out | std::ios::trunc);
            if (joint_log_.is_open())
            {
                joint_log_open_ = true;
                joint_log_ << "probe_idx,cycle";
                for (int i = 0; i < Kinematics::N_ARM_JOINTS; ++i) joint_log_ << ",qd_cmd" << i;
                for (int i = 0; i < Kinematics::N_ARM_JOINTS; ++i) joint_log_ << ",qd_meas" << i;
                joint_log_ << '\n';
                std::print("[probe] joint-q̇ actuation log → {}\n", jpath);
            }
        }
    } catch (...) {}

    // Precision re-weighting (novice→skilled) + sim2real perception noise + metrics.
    try { precision_reweighting_ = configLoader.get<bool>  ("Controller.precision_reweighting"); } catch (...) {}
    try { perception_noise_std_  = configLoader.get<double>("Controller.perception_noise_std");   } catch (...) {}
    try { conf_gain_             = configLoader.get<double>("Controller.conf_gain");               } catch (...) {}
    try { skilled_sample_period_ = configLoader.get<int>   ("Controller.skilled_sample_period");  } catch (...) {}
    try { speed_conf_gain_       = configLoader.get<double>("Controller.speed_conf_gain");         } catch (...) {}
    try { confidence_path_       = configLoader.get<std::string>("Controller.confidence_path");    } catch (...) {}
    load_confidence();
    if (precision_reweighting_)
        std::print("[skill] precision re-weighting ON (confidence={:.2f}, σ_obs={:.3f} m, "
                   "skilled sample/{} cy, speed +{:.0f}%)\n",
                   confidence_, perception_noise_std_, skilled_sample_period_, speed_conf_gain_*100.0);
    try
    {
        const auto mpath = configLoader.get<std::string>("Controller.metrics_path");
        if (not mpath.empty())
        {
            metrics_.open(mpath, std::ios::out | std::ios::trunc);
            if (metrics_.is_open())
            {
                metrics_open_ = true;
                metrics_ << "rep,success,confidence,cycle_time_s,observations\n";
                std::print("[skill] per-rep metrics → {}\n", mpath);
            }
        }
    } catch (...) {}

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
        // Put FK into world frame using the live P3Bot pose before any metric.
        refresh_arm_base_world();
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


void SpecificWorker::refresh_arm_base_world()
{
    // Install kinematics_'s base_tf_ = T_world_armbase so FK reports WORLD coords.
    // CRITICAL: the URDF base_link is NOT the P3Bot node — the KinovaGen3 (arm
    // base_link) is a child of P3Bot with its OWN mount offset+rotation (a 90°
    // rotation about Y in arm_table.wbt). So compose both:
    //     base_tf_ = T_world_P3Bot · T_P3Bot_kinova
    // Missing the second factor rotates the whole model ~90°, which is why the
    // viewer arm looked nothing like Webots and FK was meters off.
    if (not kinematics_) return;
    auto to_iso = [](const RoboCompWebots2Robocomp::ObjectPose& p)
    {
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.linear() = Eigen::Quaterniond(p.orientation.w, p.orientation.x,
                                        p.orientation.y, p.orientation.z)
                         .normalized().toRotationMatrix();
        T.translation() = Eigen::Vector3d(p.position.x / 1000.0,
                                          p.position.y / 1000.0,
                                          p.position.z / 1000.0);
        return T;
    };
    // Fixed arm→body mount (KinovaGen3 child pose inside the P3Bot Robot node in
    // arm_table.wbt): a 7 cm drop and a 90° rotation about Y. The arm is bolted
    // to the body so this is constant; only the body (P3Bot) pose is read live.
    // (The bridge's getObjectPose can't return child-node poses — find_scene_node
    // scans only top-level world children — so we can't query kinova_arm_r.)
    static const Eigen::Isometry3d T_p3bot_arm = []
    {
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.linear()      = Eigen::AngleAxisd(1.5711172845720163,
                                            Eigen::Vector3d(0.0, 1.0, 0.0)).toRotationMatrix();
        T.translation() = Eigen::Vector3d(-2.0036834417203053e-05,
                                          -7.905216286097083e-05,
                                          -0.07119293708923391);
        return T;
    }();
    try
    {
        // P3Bot is top-level → its field pose is world.
        const Eigen::Isometry3d T_world_p3bot = to_iso(webots2robocomp_proxy->getObjectPose(WEBOTS_ROBOT_DEF));
        arm_base_world_ = T_world_p3bot * T_p3bot_arm;
        kinematics_->set_base_transform(arm_base_world_);
        if (not base_tf_set_)
        {
            const Eigen::Vector3d t = arm_base_world_.translation();
            const Eigen::Vector3d e = arm_base_world_.linear().eulerAngles(0, 1, 2);
            std::print("[base] arm base_link world pose installed: t=({:.3f},{:.3f},{:.3f}) "
                       "rpy=({:+.2f},{:+.2f},{:+.2f}) — FK now in world frame\n",
                       t.x(), t.y(), t.z(), e.x(), e.y(), e.z());
            std::print("[base-json] root→robot RT for kinova2.json: "
                       "rt_translation=[{:.6f}, {:.6f}, {:.6f}]  rt_rotation_euler_xyz=[{:.6f}, {:.6f}, {:.6f}]\n",
                       t.x(), t.y(), t.z(), e.x(), e.y(), e.z());
            base_tf_set_ = true;
        }
    }
    catch (const Ice::Exception&) { /* Webots not ready yet; retry next call */ }
}

void SpecificWorker::update_bottle_pose_in_dsr()
{
    if (not G or not rt_api_) return;

    RoboCompWebots2Robocomp::ObjectPose robot_w_pose;
    RoboCompWebots2Robocomp::ObjectPose bottle_w_pose;
    RoboCompWebots2Robocomp::ObjectPose table_w_pose;
    try
    {
        robot_w_pose  = webots2robocomp_proxy->getObjectPose(WEBOTS_ROBOT_DEF);
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
    const Eigen::Vector3d robot_w (robot_w_pose.position.x  / 1000.0,
                                   robot_w_pose.position.y  / 1000.0,
                                   robot_w_pose.position.z  / 1000.0);
    const Eigen::Vector3d bottle_w(bottle_w_pose.position.x / 1000.0,
                                   bottle_w_pose.position.y / 1000.0,
                                   bottle_w_pose.position.z / 1000.0);
    const Eigen::Vector3d table_w (table_w_pose.position.x  / 1000.0,
                                   table_w_pose.position.y  / 1000.0,
                                   table_w_pose.position.z  / 1000.0);

    const Eigen::Quaterniond q_robot_w(robot_w_pose.orientation.w,
                                       robot_w_pose.orientation.x,
                                       robot_w_pose.orientation.y,
                                       robot_w_pose.orientation.z);
    const Eigen::Quaterniond q_table_w(table_w_pose.orientation.w,
                                       table_w_pose.orientation.x,
                                       table_w_pose.orientation.y,
                                       table_w_pose.orientation.z);
    const Eigen::Quaterniond q_bottle_w(bottle_w_pose.orientation.w,
                                        bottle_w_pose.orientation.x,
                                        bottle_w_pose.orientation.y,
                                        bottle_w_pose.orientation.z);

    // Cache world poses for grasp targeting + the viewer (both run in world
    // frame). Bottle long axis = bottle local +Z in world.
    // Optional perception noise (sim2real): corrupt the OBSERVED bottle position
    // with Gaussian σ so the sensory channel has finite precision Π_s = 1/σ². The
    // precision-reweighting fusion then has a real noisy x_obs to down-weight; with
    // σ=0 (ideal sim) the transition is driven by the EFE sampling schedule alone.
    bottle_pos_world_  = bottle_w;
    if (precision_reweighting_ and perception_noise_std_ > 0.0)
    {
        std::normal_distribution<double> n(0.0, perception_noise_std_);
        bottle_pos_world_.x() += n(rng_);
        bottle_pos_world_.y() += n(rng_);
    }
    bottle_axis_world_ = (q_bottle_w * Eigen::Vector3d::UnitZ()).normalized();
    table_world_.linear()      = q_table_w.normalized().toRotationMatrix();
    table_world_.translation() = table_w;
    scene_world_valid_ = true;

    auto robot_node  = G->get_node("robot");
    auto table_node  = G->get_node("table");
    auto bottle_node = G->get_node("bottle");
    if (not robot_node.has_value() or not table_node.has_value() or not bottle_node.has_value())
        return;

    // robot→table: express the table in the arm-base frame, so the kinova2.json
    // static value (stale p3bot geometry) is replaced by the live Webots mount.
    // The P3Bot Robot node is the arm-base mount (KinovaGen3 is its child at
    // identity), and the KinovaGen3 proto frame is the URDF base_link the
    // controller uses, so its local frame *is* our arm-base frame.
    const Eigen::Quaterniond q_robot_inv  = q_robot_w.conjugate();
    const Eigen::Vector3d    table_in_robot = q_robot_inv * (table_w - robot_w);
    const Eigen::Vector3d    euler_rt =
        (q_robot_inv * q_table_w).toRotationMatrix().eulerAngles(0, 1, 2);
    // One-shot startup diagnostic: the live Webots mount geometry. Mismatches
    // here (e.g. a stale arm height) are exactly what silently breaks the viewer
    // and grasp targeting, so surface it once.
    static bool geom_logged = false;
    if (not geom_logged)
    {
        std::print("[geom] robot_w=({:.3f},{:.3f},{:.3f}) table_w=({:.3f},{:.3f},{:.3f}) "
                   "-> robot→table=({:+.3f},{:+.3f},{:+.3f}) euler=({:+.2f},{:+.2f},{:+.2f})\n",
                   robot_w.x(), robot_w.y(), robot_w.z(), table_w.x(), table_w.y(), table_w.z(),
                   table_in_robot.x(), table_in_robot.y(), table_in_robot.z(),
                   euler_rt.x(), euler_rt.y(), euler_rt.z());
        geom_logged = true;
    }
    rt_api_->insert_or_assign_edge_RT(
        robot_node.value(), table_node.value().id(),
        std::vector<float>{static_cast<float>(table_in_robot.x()),
                           static_cast<float>(table_in_robot.y()),
                           static_cast<float>(table_in_robot.z())},
        std::vector<float>{static_cast<float>(euler_rt.x()),
                           static_cast<float>(euler_rt.y()),
                           static_cast<float>(euler_rt.z())});

    // table→bottle: bottle expressed in the table frame.
    const Eigen::Quaterniond q_table_inv = q_table_w.conjugate();
    const Eigen::Vector3d    bottle_in_table = q_table_inv * (bottle_w - table_w);
    const Eigen::Vector3d    euler_bt =
        (q_table_inv * q_bottle_w).toRotationMatrix().eulerAngles(0, 1, 2);
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
    // Everything here is WORLD frame (FK reports world coords). The bottle world
    // pose is cached each cycle by update_bottle_pose_in_dsr(); the arm base is
    // at arm_base_world_.translation().
    if (not base_tf_set_) return std::nullopt;            // FK not yet world-calibrated
    const Eigen::Vector3d bottle_pos = bottle_pos_world_;
    const Eigen::Vector3d z_bot      = bottle_axis_world_.normalized();  // ≈ +Z (upright)
    const Eigen::Vector3d base_pos   = arm_base_world_.translation();

    // Aim at the bottle's body, not its base. The WaterBottle PROTO origin is
    // bottom-centre, so offset up the bottle axis by a fraction of its height.
    double bottle_height_m = 0.22;
    if (auto bottle_node = G->get_node("bottle"); bottle_node.has_value())
        if (auto h = G->get_attrib_by_name<height_m_att>(bottle_node.value()); h.has_value())
            bottle_height_m = h.value();
    const Eigen::Vector3d body_centre =
        bottle_pos + z_bot * (bottle_height_m * BOTTLE_GRASP_HEIGHT_FRAC);

    // Approach azimuth is FREE for a vertical cylinder. Instead of base→bottle
    // (which drags the arm across the front of the column), approach PERPENDICULAR
    // to the column→bottle line, from the robot's RIGHT — same grasp, but the
    // forearm stays off the mast. Of the two perpendicular sides, pick the one
    // whose standoff sits farther to the robot's right (smaller world Y).
    const Eigen::Vector2d col_xy(-0.62477, -0.056064);   // real SolidPipe column
    Eigen::Vector3d u = bottle_pos - Eigen::Vector3d(col_xy.x(), col_xy.y(), bottle_pos.z());
    u -= u.dot(z_bot) * z_bot;                            // column→bottle, ⟂ bottle axis
    if (u.norm() < 1e-4) return std::nullopt;
    u.normalize();
    const Eigen::Vector3d perp = z_bot.cross(u).normalized();   // side direction (⟂ both)
    const auto standoff_y = [&](const Eigen::Vector3d& zt)
    { return (body_centre - zt * APPROACH_STANDOFF_M).y(); };
    Eigen::Vector3d z_tool_des = (standoff_y(perp) < standoff_y(-perp)) ? perp : -perp;

    // Full grasp frame at convergence:
    //   tool +Z = z_tool_des          approach axis, horizontal into bottle
    //   tool +X = z_bot × z_tool_des  horizontal tangent — fingers close around body
    //   tool +Y = Z × X = z_bot       up (wrist camera on the upper side)
    // Per-rep structured perturbation (skill-learning probe; zero when disabled):
    // rotate the approach azimuth about the bottle axis, and offset the grasp point
    // tangentially and along the axis. This perturbs the COMMAND relative to the
    // PERCEIVED bottle, so the per-rep dataset relates command offset → outcome.
    if (std::abs(rep_perturb_.dazi) > 1e-9)
        z_tool_des = (Eigen::AngleAxisd(rep_perturb_.dazi, z_bot) * z_tool_des).normalized();
    const Eigen::Vector3d x_tool_des = z_bot.cross(z_tool_des).normalized();
    const Eigen::Vector3d grasp_pt =
        body_centre + x_tool_des * rep_perturb_.dx_perp + z_bot * rep_perturb_.dz_axis;

    SideGraspTarget out;
    out.z_tool_des    = z_tool_des;
    out.x_tool_des    = x_tool_des;
    out.grasp_pos     = grasp_pt;                                 // the grasp point (world, perturbed)
    out.stand_off_pos = grasp_pt - z_tool_des * APPROACH_STANDOFF_M;

    // One-shot reachability check: distances from the arm base (world). The Gen3
    // reaches ~0.90 m to the tool; beyond that the approach stalls at extension.
    static bool reach_logged = false;
    if (not reach_logged)
    {
        std::print("[reach] bottle={:.3f} m  grasp(body)={:.3f} m  standoff={:.3f} m  (Gen3 reach ≈0.90 m)\n",
                   (bottle_pos - base_pos).norm(), (out.grasp_pos - base_pos).norm(),
                   (out.stand_off_pos - base_pos).norm());
        reach_logged = true;
    }
    return out;
}

void SpecificWorker::load_confidence()
{
    if (confidence_path_.empty()) return;
    std::ifstream f(confidence_path_);
    double c;
    if (f >> c) confidence_ = std::clamp(c, 0.0, 1.0);
}

void SpecificWorker::save_confidence()
{
    if (confidence_path_.empty()) return;
    std::ofstream f(confidence_path_, std::ios::out | std::ios::trunc);
    if (f) f << confidence_ << '\n';
}

void SpecificWorker::begin_rep_probe()
{
    // Called once at the start of each pick attempt. Advances the low-discrepancy
    // sequence and sets this rep's perturbation. With probing off the perturbation
    // is identically zero, so the grasp is exactly the nominal one.
    rep_track_ticks_ = 0;
    rep_attempts_    = 0;
    // Reset the per-rep fusion belief + metrics. cycles_since_obs_ large ⇒ the first
    // Tracking cycle takes an observation to seed the model belief.
    belief_valid_     = false;
    cycles_since_obs_ = 1 << 20;
    obs_count_rep_    = 0;
    rep_t0_           = now_seconds();

    // Deterministic pickup location for this rep (Halton sweep over the right-side
    // spawn box), teleported in via the bridge — guarantees a fresh upright bottle
    // at a known spot regardless of where the last rep left/dropped it.
    if (respawn_each_rep_)
    {
        // R2 (Roberts) low-discrepancy sequence — uniform 2-D coverage without the
        // axis-correlation Halton suffers at low indices with large bases (which made
        // earlier spawns march in a diagonal line). a1,a2 = 1/φ₂, 1/φ₂² (plastic).
        constexpr double a1 = 0.7548776662466927, a2 = 0.5698402909980532;
        const double u = std::fmod(0.5 + a1 * static_cast<double>(probe_index_), 1.0);
        const double v = std::fmod(0.5 + a2 * static_cast<double>(probe_index_), 1.0);
        const double sx = SPAWN_X_MIN + u * (SPAWN_X_MAX - SPAWN_X_MIN);
        const double sy = SPAWN_Y_MIN + v * (SPAWN_Y_MAX - SPAWN_Y_MIN);
        respawn_bottle(sx, sy);
    }

    if (probe_enabled_)
    {
        // Halton over 4 coprime bases → even coverage of the 4-D probe envelope,
        // each dim mapped from [0,1) to its signed half-range.
        rep_perturb_.dx_perp     = (halton(probe_index_, 2) - 0.5) * 2.0 * probe_pos_amp_;
        rep_perturb_.dz_axis     = (halton(probe_index_, 3) - 0.5) * 2.0 * probe_pos_amp_;
        rep_perturb_.dazi        = (halton(probe_index_, 5) - 0.5) * 2.0 * probe_azi_amp_;
        rep_perturb_.speed_scale = 1.0 + (halton(probe_index_, 7) - 0.5) * 2.0 * probe_speed_amp_;
        std::print("[probe] rep {} perturb: dx_perp={:+.3f} dz_axis={:+.3f} dazi={:+.3f} speed×{:.2f}\n",
                   probe_index_, rep_perturb_.dx_perp, rep_perturb_.dz_axis,
                   rep_perturb_.dazi, rep_perturb_.speed_scale);
    }
    else rep_perturb_ = GraspPerturbation{};

    ++probe_index_;
}

void SpecificWorker::respawn_bottle(double x, double y)
{
    if (not scene_world_valid_) return;          // need the table top z first
    RoboCompWebots2Robocomp::ObjectPose pose{};
    pose.position.x = static_cast<float>(x * 1000.0);
    pose.position.y = static_cast<float>(y * 1000.0);
    pose.position.z = static_cast<float>((table_top_z_ + 0.002) * 1000.0);  // 2 mm above → settles
    pose.orientation.w = 1.0f;                   // identity = bottle authored upright
    pose.orientation.x = pose.orientation.y = pose.orientation.z = 0.0f;
    try { webots2robocomp_proxy->setObjectPose(WEBOTS_BOTTLE_DEF, pose); }
    catch (const Ice::Exception& e)
    {
        std::print(stderr, "[spawn] setObjectPose failed: {}\n", e.what());
        return;
    }
    last_spawn_xy_ = Eigen::Vector2d(x, y);
    std::print("[spawn] bottle → ({:.3f},{:.3f},{:.3f}) upright\n", x, y, table_top_z_ + 0.002);
}

void SpecificWorker::log_rep_outcome(bool success, double rise, double xy_gap)
{
    if (not dataset_open_) return;
    const auto& g = latched_grasp_;
    dataset_ << (probe_index_ - 1) << ',' << pick_place_cycles_done_ << ','
             << (success ? 1 : 0) << ','
             << rep_perturb_.dx_perp << ',' << rep_perturb_.dz_axis << ','
             << rep_perturb_.dazi    << ',' << rep_perturb_.speed_scale << ','
             << g.grasp_pos.x() << ',' << g.grasp_pos.y() << ',' << g.grasp_pos.z() << ','
             << bottle_pos_world_.x() << ',' << bottle_pos_world_.y() << ',' << bottle_pos_world_.z() << ','
             << bottle_axis_world_.z() << ','
             << rep_commit_epos_ << ',' << rep_commit_eang_ << ',' << rep_track_ticks_ << ','
             << rise << ',' << xy_gap << '\n';
    dataset_.flush();
}

void SpecificWorker::update_viewer_scene_objects()
{
    // The viewer renders the arm in WORLD coords (FK via base_tf_), so the table
    // and bottle must be in WORLD too — otherwise they float apart. Use the live
    // Webots world poses (table_world_, bottle_*_world_) cached by
    // update_bottle_pose_in_dsr(), NOT the DSR robot-frame transforms.
    if (not arm_belief_viewer_ or not G or not scene_world_valid_) return;

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

    // Box corners in the table-local frame (origin bottom-centre, Webots PROTO
    // convention), transformed to WORLD by the live table pose.
    // Order: bottom CCW (0..3), then top CCW (4..7).
    const std::array<Eigen::Vector3d, 8> corners_in_table{{
        {-hw, -hd, 0}, {+hw, -hd, 0}, {+hw, +hd, 0}, {-hw, +hd, 0},
        {-hw, -hd, h}, {+hw, -hd, h}, {+hw, +hd, h}, {-hw, +hd, h},
    }};
    table_top_z_ = (table_world_ * Eigen::Vector3d(0.0, 0.0, h)).z();   // world z of the table surface

    std::vector<Eigen::Vector3d> corners_world;
    corners_world.reserve(8);
    for (const auto& c : corners_in_table)
        corners_world.push_back(table_world_ * c);

    // Bottle node carries width_m (diameter) and height_m. Defaults if absent.
    auto bottle_node = G->get_node("bottle");
    double diameter = 0.07, height = 0.22;
    if (bottle_node.has_value())
    {
        if (auto v = G->get_attrib_by_name<width_m_att>(bottle_node.value());  v.has_value()) diameter = v.value();
        if (auto v = G->get_attrib_by_name<height_m_att>(bottle_node.value()); v.has_value()) height   = v.value();
    }

    arm_belief_viewer_->update_scene_objects(corners_world, bottle_pos_world_,
                                             bottle_axis_world_, diameter * 0.5, height);

    // Support column (the Webots mast): a vertical cylinder from the floor (z=0)
    // up to the shoulder (arm base), at the arm-base xy. Matches the SolidPipe in
    // arm_table.wbt (radius 0.05) — the obstacle the arm must reach around.
    const Eigen::Vector3d shoulder = arm_base_world_.translation();
    const Eigen::Vector3d col_base(shoulder.x(), shoulder.y(), 0.0);
    arm_belief_viewer_->set_column(col_base, shoulder, 0.05);
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
        std::array<double, Kinematics::N_ARM_JOINTS> qd_meas{};
        for (int i = 0; i < Kinematics::N_ARM_JOINTS; ++i)
        {
            q[i]       = js.joints[i].angle;
            tau[i]     = js.joints[i].torque;     // motor torque feedback from the bridge
            qd_meas[i] = js.joints[i].velocity;   // measured joint velocity (rad/s)
        }

        // Actuation log: pair the PREVIOUS cycle's commanded q̇ (last_q_dot_cmd_) with
        // this cycle's measured q̇ — the response to it. Only while actively driving
        // in velocity mode (ActiveEFE), so position-mode homing doesn't pollute the fit.
        if (joint_log_open_ and phase_ == Phase::ActiveEFE)
        {
            joint_log_ << (probe_index_ - 1) << ',' << joint_log_cycle_++;
            for (int i = 0; i < Kinematics::N_ARM_JOINTS; ++i) joint_log_ << ',' << last_q_dot_cmd_[i];
            for (int i = 0; i < Kinematics::N_ARM_JOINTS; ++i) joint_log_ << ',' << qd_meas[i];
            joint_log_ << '\n';
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
            homing_elapsed_ticks_ = 0;
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

            // Safety: abort a homing that cannot converge instead of slewing forever.
            // Leaving the arm straining toward an unreachable/jammed pose is exactly
            // what makes it thrash in "large displacements"; halt and wait instead.
            if (homing_settled_ticks_ < HOMING_SETTLE_TICKS
                and ++homing_elapsed_ticks_ > HOMING_TIMEOUT_TICKS)
            {
                RoboCompKinovaArm::TJointSpeeds stop;
                stop.jointSpeeds.assign(Kinematics::N_ARM_JOINTS, 0.0f);
                try { kinovaarm_proxy->moveJointsWithSpeed(stop); }
                catch (const Ice::Exception&) {}
                std::print("[homing] TIMEOUT after {} cycles (max err {:.4f} rad) — "
                           "pose unreachable/jammed; halting, waiting for Start.\n",
                           HOMING_TIMEOUT_TICKS, max_err);
                run_requested_       = false;   // abort any autonomous round
                returning_for_cycle_ = false;
                phase_               = Phase::WaitingForStart;
                return;
            }

            if (homing_settled_ticks_ >= HOMING_SETTLE_TICKS)
            {
                if (returning_for_cycle_ and run_requested_
                    and (round_cycles_ <= 0 or pick_place_cycles_done_ < round_cycles_))
                {
                    // Autonomous loop: rest reached after a place — pick again.
                    returning_for_cycle_ = false;
                    grasp_phase_         = GraspPhase::Tracking;
                    grasp_settle_ticks_  = 0;
                    gripper_command_     = 1.0f;
                    phase_               = Phase::ActiveEFE;
                    begin_rep_probe();   // next rep's structured perturbation
                    std::print("[cycle] rest reached → starting pick-and-place {}\n",
                               pick_place_cycles_done_ + 1);
                }
                else if (returning_for_cycle_ and round_cycles_ > 0
                         and pick_place_cycles_done_ >= round_cycles_)
                {
                    // Round complete — park at rest, stop the loop.
                    returning_for_cycle_ = false;
                    run_requested_       = false;
                    std::print("[round] {}/{} pick-and-place cycles complete — round done, parked at rest.\n",
                               pick_place_cycles_done_, round_cycles_);
                    phase_ = Phase::WaitingForStart;
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
        if (not base_tf_set_) refresh_arm_base_world();   // retry until installed
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
                phase_                   = Phase::ActiveEFE;
                grasp_phase_             = GraspPhase::Tracking;  // fresh grasp sequence
                grasp_settle_ticks_      = 0;
                gripper_command_         = 1.0f;                  // open
                pick_place_cycles_done_  = 0;                     // fresh round
                begin_rep_probe();                               // first rep's perturbation
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
        const auto make_params = [&](const Eigen::Vector3d& z_des,
                                     const Eigen::Vector3d& x_des,
                                     double v_app)
        {
            EFEParams p;
            p.desired_approach  = z_des;
            p.desired_secondary = x_des;
            // Yaw-free grasp: pin tool +Y to the bottle axis (fingers stay ⟂ the
            // bottle, a valid cylinder grasp) and leave the YAW (approach azimuth)
            // FREE — the arm spends that DOF to stay clear of the column. tool +Y
            // in the grasp frame is z_des × x_des (= the bottle's up axis).
            p.align_tool_y      = true;
            p.desired_tool_y    = z_des.cross(x_des).normalized();
            p.gain_secondary    = 1.0;   // ignored while align_tool_y is set
            p.C_pos             = Eigen::Vector3d::Ones();  // straight-line flow
            p.dls_lambda        = 0.05;
            p.v_approach        = v_app;
            p.a_approach        = 0.60;
            p.omega_max         = 2.0;
            // Null-space elbow placement: pull the elbow into the BACK-RIGHT zone
            // (behind the column and to the robot's right, away from the mast)
            // using the redundant DOF — doesn't disturb the hand. Target = column
            // xy pushed −0.25 m in X (back) and −0.30 m in Y (right).
            p.gain_mu           = 0.0;
            p.gain_elbow        = 2.0;
            p.elbow_target      = Eigen::Vector3d(-0.62477 - 0.25, -0.056064 - 0.30, 0.0);
            // WHOLE-ARM column repulsion: push every movable joint (j3..j7) off the
            // REAL column (SolidPipe in arm_table.wbt) — axis, radius, z-extent —
            // not just the elbow. This catches the wrist/forearm, which was the
            // part actually grazing the mast.
            p.gain_mast         = 3.0;
            p.col_xy            = Eigen::Vector2d(-0.62477, -0.056064);  // SolidPipe axis
            p.col_radius        = 0.05;
            p.col_z_lo          = -0.10;          // SolidPipe: center z=0.6, height 1.4
            p.col_z_hi          =  1.30;
            p.col_margin        = 0.10;           // repel within 10 cm of the column surface
            // Hand-table repulsion: keep the tool from diving into the table.
            p.gain_table        = 2.0;
            p.table_z           = table_top_z_;
            p.table_safe        = 0.06;           // start pushing up within 6 cm of the surface
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
            last_q_dot_cmd_ = q_dot;   // for the actuation log (paired next cycle)

            const double e_pos = (ee_position - target).norm();

            // Orientation residual matching the controller's metric. With the
            // yaw-free (pin tool +Y) grasp mode, "aligned" = fingers ⟂ the bottle,
            // i.e. the angle between tool +Y and the bottle axis — yaw is ignored,
            // so the grasp can commit at any azimuth.
            double e_ang = 3.1416;
            if (params.align_tool_y)
            {
                const auto tool = kinematics_->tool_pose(q);
                e_ang = std::acos(std::clamp(
                    tool.rotation.col(1).dot(params.desired_tool_y.normalized()), -1.0, 1.0));
            }
            else
            {
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
            }

            if (tip_log_)
            {
                const double dt   = std::max(1, getPeriod("Compute")) / 1000.0;
                const double vcmd = std::min(params.v_approach,
                    std::sqrt(2.0 * params.a_approach * std::max(0.0, e_pos - params.arrive_deadband)));
                const double vmeas = tip_log_prev_pos_.has_value()
                    ? (ee_position - tip_log_prev_pos_.value()).norm() / dt : 0.0;
                const Eigen::Vector3d elb = kinematics_->elbow_position(q);
                // Whole-arm clearance to the ACTUAL column (SolidPipe): min over
                // all arm links of segment-segment distance to the column axis,
                // minus the column radius. Negative ⇒ penetrating.
                const auto sk = kinematics_->arm_skeleton_points(q);
                const Eigen::Vector3d col_lo(-0.62477, -0.056064, -0.10);
                const Eigen::Vector3d col_hi(-0.62477, -0.056064,  1.30);
                double col_min = 1e9; int col_link = -1;
                for (int k = 2; k + 1 < (int)sk.size(); ++k)   // skip base→j1→j2 (the mount, always at the column)
                {
                    const double dseg = segment_segment_distance(sk[k], sk[k+1], col_lo, col_hi) - 0.05;
                    if (dseg < col_min) { col_min = dseg; col_link = k; }
                }
                std::print("[tiplog] {},{:.3f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},eang={:.1f},elbow=({:+.3f},{:+.3f},{:+.3f}),colClr={:+.3f},colLink={},probe={}\n",
                           tip_log_cycle_, tip_log_cycle_ * dt,
                           ee_position.x(), ee_position.y(), ee_position.z(),
                           target.x(), target.y(), target.z(), e_pos, vcmd, vmeas,
                           e_ang * 57.29578, elb.x(), elb.y(), elb.z(), col_min, col_link,
                           probe_index_ - 1);
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

        // A grasp miss either retries the SAME rep (perturbation unchanged) or, once
        // the per-rep attempt cap is hit, gives up: count it as a failed cycle and
        // return to rest so the round advances (otherwise a hard probe point loops
        // forever). Returns true if it gave up (caller should break).
        const auto miss_or_give_up = [&](const char* why) -> bool
        {
            gripper_command_    = 1.0f;            // reopen
            grasp_settle_ticks_ = 0;
            if (++rep_attempts_ < MAX_REP_ATTEMPTS)
            {
                // A miss often means the bottle was toppled/nudged; re-stand a fresh
                // upright one at this rep's spot so the retry faces a clean target
                // instead of chasing a fallen bottle.
                if (respawn_each_rep_) respawn_bottle(last_spawn_xy_.x(), last_spawn_xy_.y());
                grasp_phase_ = GraspPhase::Tracking;   // retry same rep
                return false;
            }
            std::print("[probe] rep {} GIVE UP after {} attempts ({}) → failed cycle, re-home\n",
                       probe_index_ - 1, rep_attempts_, why);
            // Learning: a failed rep lowers confidence (Π_m), re-engaging feedback.
            if (precision_reweighting_)
            {
                confidence_ = std::max(0.0, confidence_ * conf_decay_);
                save_confidence();
            }
            if (metrics_open_)
                metrics_ << (probe_index_ - 1) << ",0," << confidence_ << ','
                         << (now_seconds() - rep_t0_) << ',' << obs_count_rep_ << '\n', metrics_.flush();
            RoboCompKinovaArm::TJointAngles rest;
            rest.jointAngles.assign(rest_pose_angles_.begin(), rest_pose_angles_.end());
            try { kinovaarm_proxy->moveJointsWithAngle(rest); } catch (const Ice::Exception&) {}
            returning_for_cycle_  = true;          // next rep starts after homing
            homing_settled_ticks_ = 0;
            homing_elapsed_ticks_ = 0;
            grasp_phase_          = GraspPhase::Tracking;
            place_ticks_          = 0;
            ++pick_place_cycles_done_;             // count the failed rep toward the round
            phase_                = Phase::Homing;
            return true;
        };

        // ── Tilt reflex ────────────────────────────────────────────────────
        // The bottle pose is refreshed live every cycle (update_bottle_pose_in_dsr).
        // During the approach/contact phases, if the bottle's axis tips past
        // TILT_REFLEX_RAD from vertical, abort immediately: reopen and back off
        // along −approach so we don't knock it over, let it settle, then re-track.
        if (TILT_REFLEX_ENABLED and base_tf_set_ and (grasp_phase_ == GraspPhase::Tracking
                          or  grasp_phase_ == GraspPhase::Inserting
                          or  grasp_phase_ == GraspPhase::Closing))
        {
            const double tilt = std::acos(std::clamp(std::abs(bottle_axis_world_.z()), 0.0, 1.0));
            if (tilt > TILT_REFLEX_RAD)
            {
                gripper_command_ = 1.0f;                       // release
                Eigen::Vector3d away = ee_position - bottle_pos_world_;
                away.z() = 0.0;
                away = (away.norm() > 1e-6) ? away.normalized() : Eigen::Vector3d(1, 0, 0);
                retract_target_ = ee_position + away * RETRACT_DIST_M + Eigen::Vector3d(0, 0, 0.04);
                retract_ticks_  = 0;
                ++reflex_count_;
                grasp_phase_    = GraspPhase::Retracting;
                std::print("[reflex] bottle tilting {:.0f}° → reopen + retract (#{}/{})\n",
                           tilt * 57.29578, reflex_count_, MAX_REFLEXES);
            }
        }

        // Continuous fall-abort: every cycle in the pre-lift manipulation phases,
        // check the live bottle. If it toppled (axis tilted past FALL_TILT_RAD) or
        // dropped off the table, abort this attempt now instead of grasping at a
        // fallen bottle — miss_or_give_up re-stands a fresh one and retries/gives up.
        bool aborted_for_fall = false;
        if (respawn_each_rep_ and scene_world_valid_
            and (grasp_phase_ == GraspPhase::Tracking
                 or grasp_phase_ == GraspPhase::Inserting
                 or grasp_phase_ == GraspPhase::Closing))
        {
            const double btilt = std::acos(std::clamp(std::abs(bottle_axis_world_.z()), 0.0, 1.0));
            if (btilt > FALL_TILT_RAD or bottle_pos_world_.z() < table_top_z_ - 0.08)
            {
                std::print("[fall] bottle down (tilt {:.0f}°, z={:.3f}) → abort phase\n",
                           btilt * 57.29578, bottle_pos_world_.z());
                log_rep_outcome(false, 0.0, 0.0);
                miss_or_give_up("bottle fell");
                aborted_for_fall = true;
            }
        }

        if (not aborted_for_fall)
        switch (grasp_phase_)
        {
            case GraspPhase::Retracting:
            {
                // Protective withdrawal after a tilt: stay open, back off to the
                // retract point, hold to let the bottle settle, then re-track —
                // or give up (return to rest) after too many tips.
                gripper_command_ = 1.0f;
                Eigen::Vector3d zdes = bottle_pos_world_ - retract_target_;
                zdes.z() = 0.0;
                zdes = (zdes.norm() > 1e-6) ? zdes.normalized() : Eigen::Vector3d(1, 0, 0);
                const Eigen::Vector3d xdes = Eigen::Vector3d(0, 0, 1).cross(zdes).normalized();
                const auto [e_pos, e_ang] = drive(retract_target_, zdes, xdes, 0.30);
                (void) e_ang;
                if (e_pos < REACH_TOLERANCE_M or ++retract_ticks_ > RETRACT_SETTLE_TICKS)
                {
                    if (reflex_count_ >= MAX_REFLEXES)
                    {
                        std::print("[reflex] {} tips — giving up; returning to rest.\n", reflex_count_);
                        reflex_count_  = 0;
                        run_requested_ = false;               // abort → outer loop homes to rest
                    }
                    else
                    {
                        std::print("[reflex] settled → re-tracking\n");
                        grasp_phase_ = GraspPhase::Tracking;
                    }
                }
                break;
            }
            case GraspPhase::Tracking:
            {
                gripper_command_ = 1.0f;  // open
                ++rep_track_ticks_;       // convergence-time sample for the dataset
                const auto stop_if_no_target = [&](auto& opt) -> bool
                {
                    if (opt.has_value()) return false;
                    RoboCompKinovaArm::TJointSpeeds stop;
                    stop.jointSpeeds.assign(Kinematics::N_ARM_JOINTS, 0.0f);
                    kinovaarm_proxy->moveJointsWithSpeed(stop);
                    return true;
                };

                SideGraspTarget g;
                double vscale = rep_perturb_.speed_scale;
                if (precision_reweighting_)
                {
                    // ── Precision-weighted target: fuse observation x_obs and model
                    // belief x_model, and sample x_obs on the EFE schedule (sample
                    // period grows with confidence — a fresh look gains ~no info once
                    // the model predicts well). The fusion gain on the observation is
                    // w = Π_s/(Π_m+Π_s) = 1−confidence_ (floored so vision is never
                    // fully ignored). Novice: period=1, w=1 → closed-loop. Skilled:
                    // sample rarely, w small → act on the model = open-loop.
                    const int period = 1 + static_cast<int>(
                        std::lround(confidence_ * (skilled_sample_period_ - 1)));
                    if (not belief_valid_ or ++cycles_since_obs_ >= period)
                    {
                        auto obs_opt = compute_side_grasp_target();   // x_obs (possibly noisy)
                        if (stop_if_no_target(obs_opt)) return;
                        const auto& obs = obs_opt.value();
                        cycles_since_obs_ = 0;
                        ++obs_count_rep_;
                        if (not belief_valid_) { belief_grasp_ = obs; belief_valid_ = true; }
                        else
                        {
                            double w = std::max(0.05, 1.0 - confidence_);
                            if ((obs.grasp_pos - belief_grasp_.grasp_pos).norm() > surprise_gate_m_)
                            {   // surprise → re-engage feedback, drop confidence
                                w = 1.0;
                                confidence_ = std::max(0.0, confidence_ * conf_decay_);
                            }
                            belief_grasp_.grasp_pos     += w * (obs.grasp_pos     - belief_grasp_.grasp_pos);
                            belief_grasp_.stand_off_pos += w * (obs.stand_off_pos - belief_grasp_.stand_off_pos);
                            belief_grasp_.z_tool_des = (belief_grasp_.z_tool_des + w*(obs.z_tool_des - belief_grasp_.z_tool_des)).normalized();
                            belief_grasp_.x_tool_des = (belief_grasp_.x_tool_des + w*(obs.x_tool_des - belief_grasp_.x_tool_des)).normalized();
                        }
                    }
                    g = belief_grasp_;                                 // act on the belief
                    vscale *= (1.0 + speed_conf_gain_ * confidence_);  // skilled ⇒ faster
                }
                else
                {
                    auto grasp_opt = compute_side_grasp_target();
                    if (stop_if_no_target(grasp_opt)) return;
                    g = grasp_opt.value();
                }
                const auto [e_pos, e_ang] =
                    drive(g.stand_off_pos, g.z_tool_des, g.x_tool_des, 0.35 * vscale);

                if (e_pos < REACH_TOLERANCE_M and e_ang < GRASP_ALIGN_TOL_RAD)
                {
                    // Phase-1 bring-up: converged at the standoff with the gripper
                    // open and correctly oriented — hold here, don't grasp.
                    if (approach_only_)
                    {
                        if (not approach_hold_logged_)
                        {
                            std::print("[approach] standoff reached & aligned "
                                       "(e={:.3f} m, {:.1f}°), gripper open — HOLDING.\n",
                                       e_pos, e_ang * 57.29578);
                            approach_hold_logged_ = true;
                        }
                        break;
                    }
                    if (++grasp_settle_ticks_ >= GRASP_SETTLE_TICKS)
                    {
                        // Commit: latch the grasp frame so contact can't make
                        // the target chase its own disturbance.
                        latched_grasp_ = g;
                        // Lift along the true up (bottle long axis = table normal),
                        // not arm-base +Z — the mount is tilted ~30°, so a +Z lift
                        // would drag the bottle sideways.
                        const Eigen::Vector3d up = g.z_tool_des.cross(g.x_tool_des).normalized();
                        lift_target_   = g.grasp_pos + up * LIFT_HEIGHT_M;
                        rep_commit_epos_ = e_pos;   // terminal standoff errors → dataset
                        rep_commit_eang_ = e_ang;
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
                    drive(g.grasp_pos, g.z_tool_des, g.x_tool_des, INSERT_VEL_MS * rep_perturb_.speed_scale);
                (void) e_ang;
                const float f = gripper_force();
                if (e_pos < REACH_TOLERANCE_M or f > GRASP_FORCE_THRESH)
                {
                    grasp_phase_       = GraspPhase::Closing;
                    closing_ticks_     = 0;
                    grasp_force_ticks_ = 0;
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
                // Debounce: the force must persist, not just spike for one cycle, before
                // we even tentatively call it a grasp. The real confirmation is the
                // post-lift bottle-rise check below.
                grasp_force_ticks_ = (f > GRASP_FORCE_THRESH) ? grasp_force_ticks_ + 1 : 0;
                if (grasp_force_ticks_ >= GRASP_FORCE_HOLD_TICKS)
                {
                    grasp_phase_           = GraspPhase::Lifting;
                    reflex_count_          = 0;   // clean grasp — clear the tilt-reflex tally
                    grasp_force_ticks_     = 0;
                    lift_ticks_            = 0;   // start the lift watchdog
                    bottle_z_at_lift_start_ = bottle_pos_world_.z();  // baseline for lift-confirm
                    std::print("[grasp] contact (f={:.2f}, held {} cy) → Lifting (will confirm by bottle rise)\n",
                               f, GRASP_FORCE_HOLD_TICKS);
                }
                else if (++closing_ticks_ > CLOSING_TIMEOUT_TICKS)
                {
                    std::print("[grasp] MISS — no contact in {} cycles → reopen\n",
                               CLOSING_TIMEOUT_TICKS);
                    log_rep_outcome(false, 0.0, 0.0);   // never reached contact
                    miss_or_give_up("no contact");
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
                    // ── Authoritative grasp confirmation ──────────────────────────
                    // The gripper is now LIFT_HEIGHT_M above the grasp point. If the
                    // bottle is truly held it rose with us; if we closed on air it
                    // stayed on the table (rise ≈ 0), and a bottle that slipped out
                    // mid-lift falls (rise < 0). Force alone lied (an empty close reads
                    // a few N), so decide on the bottle's own vertical motion — ground
                    // truth from getObjectPose, refreshed live every cycle. Rise is
                    // origin-offset-free (only the Δz of the bottle base matters), so
                    // it's robust regardless of where the bottle frame sits.
                    const double rise = bottle_pos_world_.z() - bottle_z_at_lift_start_;
                    // Horizontal co-location also guards against having scooped the
                    // bottle sideways without holding it (lift is straight up, so a
                    // held bottle stays under the tool; LIFT_CONFIRM_HOLD_M is a slack
                    // ceiling, the rise above is the decisive test).
                    const double xy_gap =
                        (bottle_pos_world_.head<2>() - ee_position.head<2>()).norm();
                    if (rise < LIFT_CONFIRM_RISE_M or xy_gap > LIFT_CONFIRM_HOLD_M)
                    {
                        std::print("[grasp] MISS — bottle not held (rose {:.3f} m, need {:.2f}; "
                                   "xy gap {:.3f} m, max {:.2f}) → reopen\n",
                                   rise, LIFT_CONFIRM_RISE_M, xy_gap, LIFT_CONFIRM_HOLD_M);
                        log_rep_outcome(false, rise, xy_gap);
                        miss_or_give_up("bottle not held");
                        break;                        // do NOT place an empty gripper
                    }
                    log_rep_outcome(true, rise, xy_gap);
                    // Learning: a confirmed grasp raises confidence (Π_m), so the next
                    // reps sample vision less and move faster — the skill consolidating.
                    if (precision_reweighting_)
                    {
                        confidence_ = std::min(1.0, confidence_ + conf_gain_);
                        save_confidence();
                    }
                    if (metrics_open_)
                        metrics_ << (probe_index_ - 1) << ",1," << confidence_ << ','
                                 << (now_seconds() - rep_t0_) << ',' << obs_count_rep_ << '\n', metrics_.flush();
                    std::print("[grasp] CONFIRMED held (rose {:.3f} m, gap {:.3f} m, conf {:.2f}, "
                               "{} obs, {:.1f}s) → place\n",
                               rise, xy_gap, confidence_, obs_count_rep_, now_seconds() - rep_t0_);
                    lift_ticks_ = 0;
                    // Pick a place spot on the RIGHT side of the table (world box),
                    // ≥ PLACE_MIN_MOVE_M from the pick. We run in world frame, so
                    // up = +Z and the table is horizontal: keep the EE at the grasp
                    // height (grasp_pos.z) so the bottle base sets back on the table.
                    const Eigen::Vector3d up(0.0, 0.0, 1.0);
                    std::uniform_real_distribution<double> ux(PLACE_X_MIN, PLACE_X_MAX);
                    std::uniform_real_distribution<double> uy(PLACE_Y_MIN, PLACE_Y_MAX);
                    // Sample the right-side box for a spot that is (a) ≥ PLACE_MIN_MOVE_M
                    // from the pick and (b) within reach of the arm (the box now extends
                    // past the reach edge — see PLACE_REACH_MAX_M). Among the valid draws
                    // keep the FARTHEST-from-base one, so placements bias toward the distant
                    // edge of the reachable workspace as requested.
                    const Eigen::Vector3d base_xyz = arm_base_world_.translation();
                    Eigen::Vector3d p = g.grasp_pos;
                    double best_reach = -1.0;
                    bool   have = false;
                    for (int k = 0; k < 40; ++k)
                    {
                        Eigen::Vector3d c(ux(rng_), uy(rng_), g.grasp_pos.z());
                        const double far_from_pick = (c.head<2>() - g.grasp_pos.head<2>()).norm();
                        const double reach = (c - base_xyz).norm();
                        if (far_from_pick < PLACE_MIN_MOVE_M or reach > PLACE_REACH_MAX_M) continue;
                        if (reach > best_reach) { best_reach = reach; p = c; have = true; }
                    }
                    if (not have)   // degenerate box/pick combo — fall back to nearest reachable
                        for (int k = 0; k < 40; ++k)
                        {
                            Eigen::Vector3d c(ux(rng_), uy(rng_), g.grasp_pos.z());
                            if ((c - base_xyz).norm() <= PLACE_REACH_MAX_M) { p = c; break; }
                        }
                    place_pos_   = p;
                    place_hover_ = p + up * LIFT_HEIGHT_M;
                    // Side approach re-pointed toward the place spot, bottle upright
                    // (tool +Z horizontal toward the spot, tool +Y = up).
                    Eigen::Vector3d radial = place_pos_ - arm_base_world_.translation();
                    radial.z() = 0.0;
                    place_z_des_ = (radial.norm() > 1e-6) ? radial.normalized() : g.z_tool_des;
                    place_x_des_ = up.cross(place_z_des_).normalized();
                    grasp_phase_ = GraspPhase::PlaceMoving;
                    place_ticks_ = 0;
                    std::print("[place] lifted → carry to right-side spot ({:.3f},{:.3f},{:.3f}) → PlaceMoving\n",
                               place_pos_.x(), place_pos_.y(), place_pos_.z());
                }
                else if (++lift_ticks_ > LIFT_TIMEOUT_TICKS)
                {
                    // Couldn't reach the lift target — grabbed a toppled/jammed bottle
                    // or wedged. Don't loop forever; record a miss and retry/give up.
                    const double rise = bottle_pos_world_.z() - bottle_z_at_lift_start_;
                    std::print("[grasp] MISS — lift stalled ({} cy, rose {:.3f} m) → reopen\n",
                               LIFT_TIMEOUT_TICKS, rise);
                    log_rep_outcome(false, rise, 0.0);
                    miss_or_give_up("lift stalled");
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
                    homing_elapsed_ticks_ = 0;
                    grasp_phase_          = GraspPhase::Tracking;  // for the next cycle
                    place_ticks_          = 0;
                    phase_                = Phase::Homing;
                    ++pick_place_cycles_done_;
                    std::print("[cycle] {}/{} pick-and-place complete → returning to rest\n",
                               pick_place_cycles_done_, round_cycles_ > 0 ? round_cycles_ : pick_place_cycles_done_);
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

