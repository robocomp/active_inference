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
#include "pick_and_place_fsm.h"

#include <chrono>
#include <cmath>
#include <print>
#include <sstream>
#include <stdexcept>

namespace
{
    // Hardcoded for the first sanity test.  Move to etc/config once we have
    // confirmed Pinocchio loads the URDF correctly.
    constexpr auto URDF_PATH =
        "/home/pbustos/robocomp/components/active_inference/common/kinematics/gen3_robotiq_2f_85-mod.urdf";

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

	// On exit, PARK the arm at the rest pose: send a POSITION command the bridge holds and servos to,
	// so the arm travels home and stays there even after this agent is gone — rather than being left
	// over the table mid-grasp. nearest_equiv_target wraps the continuous joints to the nearest
	// revolution of the measured config so it doesn't unwind a full turn. A bare zero-velocity only
	// halts in place, so it's kept solely as a fallback if the position command can't be delivered.
	// Wrapped in try/catch because the proxy may already be unreachable on shutdown — nothing we can do.
	try
	{
		kinovaarm_proxy->moveJointsWithAngle(nearest_equiv_target(cur_q_, rest_pose_angles_));
		std::cout << "[shutdown] Commanded arm to rest pose." << std::endl;
	}
	catch (const Ice::Exception& e)
	{
		std::cerr << "[shutdown] Could not command rest pose: " << e.what() << std::endl;
		// Fallback: at least halt drift in place with zero velocity.
		try
		{
			RoboCompKinovaArm::TJointSpeeds stop;
			stop.jointSpeeds.assign(Kinematics::N_ARM_JOINTS, 0.0f);
			kinovaarm_proxy->moveJointsWithSpeed(stop);
			std::cout << "[shutdown] Fallback: sent zero velocities." << std::endl;
		}
		catch (...) {}
	}
	catch (...)
	{
		std::cerr << "[shutdown] Unknown error commanding rest pose." << std::endl;
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

    // Auto-arm the run as if the Start button were checked (default off).
    try { run_requested_ = configLoader.get<bool>("Controller.auto_start"); } catch (...) {}
    if (run_requested_) std::print("[ui] auto_start: run requested from config\n");
    try { exit_on_homing_timeout_ = configLoader.get<bool>("Controller.exit_on_homing_timeout"); } catch (...) {}
    try { bottle_from_graph_ = configLoader.get<bool>("Controller.bottle_from_graph"); } catch (...) {}
    if (bottle_from_graph_) std::print("[scene] bottle pose READ from DSR graph (table->bottle edge), not Webots\n");
    try { bottle_monitor_ = configLoader.get<bool>("Controller.bottle_monitor"); } catch (...) {}
    try { publish_scene_to_graph_ = configLoader.get<bool>("Controller.publish_scene_to_graph"); } catch (...) {}
    if (publish_scene_to_graph_) std::print("[scene] publishing robot->table->bottle to DSR graph (Webots ground truth)\n");
    try { joint_buffer_enabled_ = configLoader.get<bool>("Controller.publish_joint_buffer"); } catch (...) {}
    if (joint_buffer_enabled_) std::print("[joint-buffer] publishing (stamp_ms,q) ring on kinova_arm_r for self_calibration\n");

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

    // The manipulation behaviour layer: EFE/QP controller + grasp-target perception +
    // pick-and-place state machine. Loads its own Controller.* behaviour config.
    fsm_ = std::make_unique<PickandPlaceFSM>(*this, configLoader);

    // Loop-rate guard: the control law assumes a ~100 Hz cadence (the velocity commands are held by
    // the bridge until the next cycle). Make sure the compute period actually targets 100 Hz — if the
    // config left it slower, tighten it to LOOP_TARGET_PERIOD_MS and say so.
    const int cfg_period = getPeriod("Compute");
    if (cfg_period > LOOP_TARGET_PERIOD_MS)
    {
        std::print("[looprate] Period.Compute={} ms is slower than the {} ms (100 Hz) target — setting it to {} ms\n",
                   cfg_period, LOOP_TARGET_PERIOD_MS, LOOP_TARGET_PERIOD_MS);
        setPeriod("Compute", LOOP_TARGET_PERIOD_MS);
    }
    std::print("[looprate] control loop target = {} Hz (Period.Compute = {} ms)\n",
               1000 / std::max(1, getPeriod("Compute")), getPeriod("Compute"));
}

// Publish the rolling (stamp_ms, q) ring onto the kinova_arm_r node so the
// self_calibration agent can nearest-match q to an image stamp_ms. Flattened to
// float vectors (the voxelizer-proven runtime-attr path); stamps are float OFFSETS
// from a uint64 base so ms precision survives float32. The graph write is throttled
// to ~20 Hz to bound the update_node signal cascade on the arm node's subscribers
// (escape hatch if it still churns: a dedicated (stamp,q) DDS stream — see memory).
void SpecificWorker::publish_joint_buffer(std::uint64_t stamp_ms,
                                          const std::array<double, Kinematics::N_ARM_JOINTS>& q)
{
    std::array<float, Kinematics::N_ARM_JOINTS> qf{};
    for (int i = 0; i < Kinematics::N_ARM_JOINTS; ++i)
        qf[i] = static_cast<float>(q[i]);
    joint_ring_.emplace_back(stamp_ms, qf);
    while (joint_ring_.size() > JOINT_BUFFER_N)
        joint_ring_.pop_front();

    if (++joint_buffer_tick_ % JOINT_BUFFER_WRITE_EVERY != 0)
        return;
    if (not G or joint_ring_.empty())
        return;
    auto node = G->get_node(WEBOTS_ARM_DEF);   // "kinova_arm_r"
    if (not node.has_value())
        return;

    const std::uint64_t base_ms = joint_ring_.front().first;
    std::vector<float> stamp_off;
    std::vector<float> qflat;
    stamp_off.reserve(joint_ring_.size());
    qflat.reserve(joint_ring_.size() * Kinematics::N_ARM_JOINTS);
    for (const auto& [s, a] : joint_ring_)
    {
        stamp_off.push_back(static_cast<float>(s - base_ms));   // ms offset, exact in float (< 2^24)
        for (float v : a)
            qflat.push_back(v);
    }
    G->add_or_modify_attrib_local<joint_buffer_base_ms_att>     (node.value(), base_ms);
    G->add_or_modify_attrib_local<joint_buffer_stamp_off_ms_att>(node.value(), stamp_off);
    G->add_or_modify_attrib_local<joint_buffer_q_att>          (node.value(), qflat);
    G->add_or_modify_attrib_local<joint_buffer_dof_att>        (node.value(), static_cast<int>(Kinematics::N_ARM_JOINTS));
    G->update_node(node.value());
}

void SpecificWorker::compute()
{
    if (not kinematics_) return;
    try
    {
        // Per-section wall-clock timing to attribute control-loop hitches (the cause of the
        // approach oscillation: a long compute() blocks the RoboComp timer, the bridge holds
        // the last q̇ for that whole interval, the arm overshoots). A [loophitch] line is
        // emitted only when the cycle exceeds the budget, so the slow-cycle culprit is named.
        using clk = std::chrono::steady_clock;
        const auto ms = [](clk::time_point a, clk::time_point b)
        { return std::chrono::duration<double, std::milli>(b - a).count(); };
        const auto t0 = clk::now();

        // Loop-rate guard: measure the ACTUAL call-to-call period and EMA-smooth it. Warn (throttled)
        // if the loop sustains below ~80% of target — a sag the per-cycle [loophitch] check misses.
        if (have_last_compute_t0_)
        {
            const double dt_ms = ms(last_compute_t0_, t0);
            loop_period_ema_ms_ = loop_period_ema_ms_ > 0.0 ? 0.97 * loop_period_ema_ms_ + 0.03 * dt_ms
                                                            : dt_ms;
            const double target = static_cast<double>(getPeriod("Compute"));
            if (++loop_rate_log_cycle_ % 200 == 0 and loop_period_ema_ms_ > 1.25 * target)
                std::print("[looprate] loop running at {:.0f} Hz (mean period {:.1f} ms vs {:.0f} ms target) "
                           "— compute() or system load is stretching the cycle\n",
                           1000.0 / loop_period_ema_ms_, loop_period_ema_ms_, target);
        }
        last_compute_t0_      = t0;
        have_last_compute_t0_ = true;

        // The one and only call to the KinovaArm proxy in the main loop.  All data
        const auto js = kinovaarm_proxy->getJointsState();
        if (static_cast<int>(js.joints.size()) < Kinematics::N_ARM_JOINTS)
        {
            std::print("[compute] proxy returned only {} joints; expected ≥ {}\n",
                       js.joints.size(), Kinematics::N_ARM_JOINTS);
            return;
        }
        const auto t_read = clk::now();

        std::array<double, Kinematics::N_ARM_JOINTS> q{}, tau{};
        for (int i = 0; i < Kinematics::N_ARM_JOINTS; ++i)
        {
            q[i]   = js.joints[i].angle;
            tau[i] = js.joints[i].torque;
        }
        cur_q_ = q;

        // Publish the rolling (stamp_ms, q) ring for the self_calibration agent.
        // js.timestamp is epoch ms (same clock as the media stamp_ms — confirmed by
        // [ts-probe]); the ring keeps filling at 100 Hz, the graph write is throttled.
        if (joint_buffer_enabled_)
            publish_joint_buffer(static_cast<std::uint64_t>(js.timestamp), q);

        send_gripper_command();
        const auto t_grip = clk::now();

        const auto ee_position = kinematics_->forward_kinematics(q);
        // Time-series sensor plot: fed EVERY cycle (100 Hz) so a brief contact transient is captured,
        // not aliased. Cheap (ring-buffer push; the plot self-throttles its repaint to ~33 Hz).
        feed_force_plot(q, tau);
        // Heavy 3D belief view (per-link FK for the meshes) throttled to ~20 Hz — keeps it off the
        // 100 Hz loop (the original per-cycle update was a loop-hitch suspect).
        if (arm_belief_viewer_ and (++viewer_tick_ % 5 == 0))
            update_viewer(q, tau, ee_position);
        const auto t_view = clk::now();
        handle_joint_dump(q, ee_position, js);

        // Outer lifecycle: home → wait for Start → run the manipulation FSM.
        if (phase_ == Phase::SendingRestPose) { run_sending_rest_pose(q); return; }
        if (phase_ == Phase::Homing)          { run_homing(q); return; }

        update_scene();
        const auto t_scene = clk::now();

        if (phase_ == Phase::WaitingForStart) { run_waiting_for_start(); return; }
        if (not run_requested_)               { abort_to_rest(); return; }

        if (monitor_bottle_and_recover()) return;   // GT bottle tipped/rolled → aborted + re-stood

        fsm_->step(q, ee_position);   // ActiveEFE: pick-and-place behaviour (currently: approach + hold)
        const auto t_fsm = clk::now();

        if (const double tot = ms(t0, t_fsm); tot > 40.0)
            std::print("[loophitch] cycle={:.1f}ms  read={:.1f}  grip={:.1f}  viewer={:.1f}  "
                       "scene={:.1f}  fsm={:.1f}\n",
                       tot, ms(t0, t_read), ms(t_read, t_grip), ms(t_grip, t_view),
                       ms(t_view, t_scene), ms(t_scene, t_fsm));
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

RoboCompKinovaArm::TJointAngles SpecificWorker::nearest_equiv_target(
    const std::array<double, Kinematics::N_ARM_JOINTS>& q,
    const std::array<double, Kinematics::N_ARM_JOINTS>& desired) const
{
    constexpr double TWO_PI = 6.283185307179586;
    const auto lims = kinematics_->arm_joint_position_limits();
    RoboCompKinovaArm::TJointAngles target;
    target.jointAngles.resize(Kinematics::N_ARM_JOINTS);
    for (int i = 0; i < Kinematics::N_ARM_JOINTS; ++i)
    {
        double t = desired[i];
        if (not std::isfinite(lims[i].first) or not std::isfinite(lims[i].second))
            t += std::round((q[i] - t) / TWO_PI) * TWO_PI;
        target.jointAngles[i] = static_cast<float>(t);
    }
    return target;
}

void SpecificWorker::teleport_to_rest()
{
    RoboCompKinovaArm::TJointAngles rest;
    rest.jointAngles.assign(rest_pose_angles_.begin(), rest_pose_angles_.end());
    try { webots2robocomp_proxy->setArmJointsInstant(rest); }
    catch (const Ice::Exception& e)
    { std::print(stderr, "[recovery] setArmJointsInstant failed: {}\n", e.what()); }
    std::print("[recovery] arm teleported to rest pose (jam recovery)\n");
}

void SpecificWorker::send_gripper_command()
{
    // EXPERIMENT (loop-hitch isolation): skip the setGripperPos round-trip when the command
    // is unchanged — during approach/transit the gripper is static, so this is a pure
    // redundant Ice call every cycle.
    static float last_sent = 2.0f;   // out of [0,1] range ⇒ first call always sends
    if (gripper_command_ == last_sent) return;
    try
    {
        kinovaarm_proxy->setGripperPos(gripper_command_);
        last_sent = gripper_command_;
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
}

void SpecificWorker::update_viewer(
    const std::array<double, Kinematics::N_ARM_JOINTS>& q,
    const std::array<double, Kinematics::N_ARM_JOINTS>& tau,
    const Eigen::Vector3d& ee_position)
{
    const auto mesh_link_poses = kinematics_->arm_mesh_link_poses(q);
    std::vector<ArmBeliefViewer3D::LinkPose> viewer_link_poses;
    viewer_link_poses.reserve(mesh_link_poses.size());
    for (const auto& lp : mesh_link_poses)
        viewer_link_poses.push_back({lp.mesh_filename, lp.pose});
    (void) tau;   // wrist-force feed moved to feed_force_plot (per-cycle); 3D view is throttled
    if (arm_belief_viewer_)
        arm_belief_viewer_->update_beliefs(q, viewer_link_poses, reach_target_, ee_position);
}

void SpecificWorker::feed_force_plot(
    const std::array<double, Kinematics::N_ARM_JOINTS>& q,
    const std::array<double, Kinematics::N_ARM_JOINTS>& tau)
{
    if (not arm_belief_viewer_) return;
    // wristFz channel: external wrench estimate from the gravity-compensated torque residual.
    // Pinocchio RNEA + Jacobian are microseconds — negligible vs the gripper Ice read below.
    const auto g_tau = kinematics_->arm_gravity_torque(q);
    Eigen::Matrix<double, Kinematics::N_ARM_JOINTS, 1> tau_ext;
    for (int i = 0; i < Kinematics::N_ARM_JOINTS; ++i) tau_ext[i] = tau[i] - g_tau[i];
    const auto J6 = kinematics_->arm_jacobian_full(q);
    const Eigen::Matrix<double, 6, 1> wrench =
        (J6 * J6.transpose() + 1e-6 * Eigen::Matrix<double, 6, 6>::Identity())
            .ldlt().solve(J6 * tau_ext);
    const float wrist_force = static_cast<float>(wrench.head<3>().norm());
    // Signed vertical component (base +Z up): a held bottle adds ~m·g, shifting this. The FSM reads
    // it to weight-confirm the lift independently of the simulator ground-truth rise.
    wrist_fz_ = static_cast<float>(wrench[2]);
    try
    {   // one localhost gripper read per cycle (100 Hz) so the tip-force impact spike is captured
        const auto gs = kinovaarm_proxy->getGripperState();
        const float ltip = std::sqrt(gs.lfx*gs.lfx + gs.lfy*gs.lfy + gs.lfz*gs.lfz);   // |force-3d| (was gs.ltipforce)
        const float rtip = std::sqrt(gs.rfx*gs.rfx + gs.rfy*gs.rfy + gs.rfz*gs.rfz);
        // Side (lateral) force = the pad-WIDTH shear, sensor-frame Fy — the off-centre signal the
        // |force-3d| magnitude hides and the compliant-close centring acts on. Signed (direction matters).
        arm_belief_viewer_->update_forces(gs.lforce, gs.rforce, ltip, rtip,
                                          gs.lfy, gs.rfy, wrist_force,
                                          gs.ltipcontact, gs.rtipcontact, gs.distance);
        // Side-force probe: the force-3d nets to ~0 in static grip but SPIKES on first contact. Catch
        // that spike WHEREVER it occurs (which grasp phase, what magnitude/sign) — the throttled Closing
        // [cshear] log aliases it. Triggered on the spike (not the rate) so it doesn't flood.
        static int side_probe_n = 0;
        const float sidemag = std::max({std::abs(gs.lfx), std::abs(gs.lfy), std::abs(gs.rfx), std::abs(gs.rfy)});
        if (sidemag > 0.003f and (side_probe_n++ % 4 == 0) and fsm_)
            std::print("[side-probe] phase={} |{:.4f}| Lxy=({:+.4f},{:+.4f}) Rxy=({:+.4f},{:+.4f}) |tip|=({:.4f},{:.4f})\n",
                       fsm_->current_grasp_phase_name(), sidemag, gs.lfx, gs.lfy, gs.rfx, gs.rfy, ltip, rtip);
    }
    catch (const Ice::Exception&) {}
}

void SpecificWorker::handle_joint_dump(
    const std::array<double, Kinematics::N_ARM_JOINTS>& q,
    const Eigen::Vector3d& ee_position,
    const RoboCompKinovaArm::TJoints& js)
{
    if (not joint_dump_pending_) return;
    std::print("[joint-dump] received {} joints from KinovaArm proxy\n", js.joints.size());
    for (size_t i = 0; i < js.joints.size(); ++i)
        std::print("  [{}] id={} angle={:+.4f} rad  velocity={:+.4f}  torque={:+.3f}\n",
                   i, js.joints[i].id, js.joints[i].angle,
                   js.joints[i].velocity, js.joints[i].torque);
    std::print("[joint-dump] FK at received angles: tool_frame=[{:+.4f}, {:+.4f}, {:+.4f}] m\n",
               ee_position.x(), ee_position.y(), ee_position.z());
    std::print("[joint-dump] Cross-check this position against Webots' viewer.\n"
               "             If they disagree we have a base-frame or sign-convention mismatch.\n");
    joint_dump_pending_ = false;
}

void SpecificWorker::run_sending_rest_pose(
    const std::array<double, Kinematics::N_ARM_JOINTS>& q)
{
    kinovaarm_proxy->moveJointsWithAngle(nearest_equiv_target(q, rest_pose_angles_));
    homing_settled_ticks_ = 0;
    homing_elapsed_ticks_ = 0;
    phase_ = Phase::Homing;
    proxy_unreachable_warned_ = false;
}

void SpecificWorker::run_homing(
    const std::array<double, Kinematics::N_ARM_JOINTS>& q)
{
    kinovaarm_proxy->moveJointsWithAngle(nearest_equiv_target(q, rest_pose_angles_));

    double max_err = 0.0;
    for (int i = 0; i < Kinematics::N_ARM_JOINTS; ++i)
        max_err = std::max(max_err, angular_distance(q[i], rest_pose_angles_[i]));
    if (max_err < HOMING_TOLERANCE_RAD)
        ++homing_settled_ticks_;
    else
        homing_settled_ticks_ = 0;

    if (homing_settled_ticks_ < HOMING_SETTLE_TICKS
        and ++homing_elapsed_ticks_ > HOMING_TIMEOUT_TICKS)
    {
        RoboCompKinovaArm::TJointSpeeds stop;
        stop.jointSpeeds.assign(Kinematics::N_ARM_JOINTS, 0.0f);
        try { kinovaarm_proxy->moveJointsWithSpeed(stop); } catch (const Ice::Exception&) {}
        run_requested_ = false;
        if (exit_on_homing_timeout_)
        {
            std::print(stderr, "[warn] homing TIMEOUT after {} cycles (max err {:.4f} rad) — jammed; "
                       "exit_on_homing_timeout ⇒ quitting (fail-fast for unattended runs).\n",
                       HOMING_TIMEOUT_TICKS, max_err);
            QTimer::singleShot(200, QCoreApplication::instance(), SLOT(quit()));
            return;
        }
        std::print(stderr, "[warn] homing TIMEOUT after {} cycles (max err {:.4f} rad) — "
                   "pose unreachable/jammed; halting, waiting for Start.\n",
                   HOMING_TIMEOUT_TICKS, max_err);
        phase_         = Phase::WaitingForStart;
        return;
    }

    // Settled: let the FSM decide whether to restart a pick cycle or park at rest.
    if (homing_settled_ticks_ >= HOMING_SETTLE_TICKS)
        phase_ = fsm_->on_rest_reached();
    proxy_unreachable_warned_ = false;
}

void SpecificWorker::update_scene()
{
    if (not base_tf_set_) refresh_arm_base_world();
    update_bottle_pose_in_dsr();
    // update_viewer_scene_objects();   // OpenGL viewer disabled (loop-hitch diagnosis)
    fsm_->maybe_compute_reach_map();
}

void SpecificWorker::run_waiting_for_start()
{
    RoboCompKinovaArm::TJointSpeeds stop;
    stop.jointSpeeds.assign(Kinematics::N_ARM_JOINTS, 0.0f);
    kinovaarm_proxy->moveJointsWithSpeed(stop);
    if (run_requested_)
    {
        std::print("[start] Entering EFE bottle-approach.\n");
        phase_ = Phase::ActiveEFE;
        fsm_->start();
    }
    proxy_unreachable_warned_ = false;
}

void SpecificWorker::abort_to_rest()
{
    std::print("[stop] Run unchecked — returning to rest pose.\n");
    fsm_->reset();
    phase_ = Phase::SendingRestPose;
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
    // Fixed offset from the queried Webots robot node to the arm base_link = the KinovaGen3
    // child pose inside that Robot node. In shadow_arm.wbt the bridge is attached to the arm's
    // OWN top-level Robot (DEF KINOVA), and its KinovaGen3 child is IDENTITY — so the queried
    // world pose already IS the arm base and this stays IDENTITY. Do NOT fold the Shadow-body→
    // arm mount (T_mount, Rz(+90°)+t=(0.272385,0.014412,0.520)) in here: that transform lives
    // in shadow_arm.json (body→kinova RT edge) for the arm-FRAME servo refactor, and applying
    // it here too would double-count (the controller queries KINOVA, not the Shadow body).
    static const Eigen::Isometry3d T_p3bot_arm = Eigen::Isometry3d::Identity();
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

    bottle_obs_fresh_ = false;   // set true below only if the producer bumped model_generation this cycle

    // robot + table are STATIC scene geometry: each getObjectPose is a slow,
    // step-synchronized Webots SUPERVISOR round-trip (the dominant loop-hitch source), so
    // query them ONCE and cache. Only the bottle moves ⇒ the only per-cycle supervisor call.
    static bool static_scene_cached = false;
    static RoboCompWebots2Robocomp::ObjectPose robot_w_pose;
    static RoboCompWebots2Robocomp::ObjectPose table_w_pose;
    RoboCompWebots2Robocomp::ObjectPose bottle_w_pose;
    try
    {
        if (not static_scene_cached)
        {
            robot_w_pose = webots2robocomp_proxy->getObjectPose(WEBOTS_ROBOT_DEF);
            // The table is queried from Webots ground truth ONLY when we publish the scene to
            // the graph. In bottle_from_graph (Option-A) mode the perception stack owns the
            // table node, so its pose is read from the graph below — and there is no DEF
            // "table" in shadow_arm.wbt, so this call would throw.
            if (not bottle_from_graph_)
                table_w_pose = webots2robocomp_proxy->getObjectPose(WEBOTS_TABLE_DEF);
            static_scene_cached = true;
        }
        if (not bottle_from_graph_ or publish_scene_to_graph_)   // need the Webots bottle as control source and/or to publish
            bottle_w_pose = webots2robocomp_proxy->getObjectPose(WEBOTS_BOTTLE_DEF);
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

    // Cache world poses for grasp targeting + the viewer (both run in world frame).
    // Bottle long axis = bottle local +Z in world.
    if (bottle_from_graph_)
    {
        // Option-A: the perception stack owns the scene. Read the table in the arm-base frame
        // from the graph (inner_eigen walks root→…→table and root→…→kinova_arm_r) and lift it to
        // the controller's world frame via arm_base_world_ — node 280 (kinova_arm_r) is the URDF
        // base_link, so arm_base_world_ is exactly its arm-base→world map. Then read the bottle in
        // the table frame and lift via the fresh table_world_, so the FSM still operates in world
        // frame. A missing edge = no detection this cycle: keep the previous belief (a dropout the
        // look-up belief predicts through).
        if (base_tf_set_ and inner_eigen_api_)
        {
            if (auto tt = inner_eigen_api_->transform_axis("kinova_arm_r", "table"); tt.has_value())
            {
                const auto& v = tt.value();   // [x,y,z, rx,ry,rz]: metres + xyz-euler in the arm-base frame
                const Eigen::Matrix3d R_at =
                    (Eigen::AngleAxisd(v[3], Eigen::Vector3d::UnitX())
                   * Eigen::AngleAxisd(v[4], Eigen::Vector3d::UnitY())
                   * Eigen::AngleAxisd(v[5], Eigen::Vector3d::UnitZ())).toRotationMatrix();
                table_world_.linear()      = arm_base_world_.linear() * R_at;
                table_world_.translation() = arm_base_world_ * Eigen::Vector3d(v[0], v[1], v[2]);
            }
            // bottle_concept publishes the bottle as a `cylinder` node named bottle_* (parented to
            // room), so resolve it by TYPE, not a literal name; express it in the table frame and
            // lift via table_world_.
            std::string bottle_name;
            if (G)
                if (const auto cyls = G->get_nodes_by_type("cylinder"); not cyls.empty())
                {
                    const auto& bnode = cyls.front();
                    bottle_name = bnode.name();
                    // model_generation is the perception clock: it advances only when bottle_concept
                    // re-fits the bottle, so an unchanged value means this read is the SAME info as
                    // last cycle (the loop is necessarily open-loop until the next bump). Measure the
                    // wall-time between bumps = the real τ_perc the lag-safety contract needs.
                    if (const auto gen = G->get_attrib_by_name<model_generation_att>(bnode); gen.has_value())
                    {
                        const int g_now = gen.value();
                        if (g_now != bottle_model_generation_)
                        {
                            const auto t_now = std::chrono::steady_clock::now();
                            if (bottle_gen_time_valid_)
                            {
                                const double dt = std::chrono::duration<double>(t_now - bottle_last_gen_time_).count();
                                bottle_refresh_period_s_ = bottle_refresh_period_s_ > 0.0
                                    ? 0.8 * bottle_refresh_period_s_ + 0.2 * dt   // EWMA, smooths jitter
                                    : dt;
                            }
                            bottle_last_gen_time_    = t_now;
                            bottle_gen_time_valid_   = true;
                            bottle_model_generation_ = g_now;
                            bottle_obs_fresh_        = true;
                        }
                    }
                }
            if (not bottle_name.empty())
                if (auto bt = inner_eigen_api_->transform_axis("table", bottle_name); bt.has_value())
                {
                    const auto& v = bt.value();   // [x,y,z, rx,ry,rz]: metres + xyz-euler in the table frame
                    const Eigen::Vector3d p_table(v[0], v[1], v[2]);
                    const Eigen::Matrix3d R_tb =
                        (Eigen::AngleAxisd(v[3], Eigen::Vector3d::UnitX())
                       * Eigen::AngleAxisd(v[4], Eigen::Vector3d::UnitY())
                       * Eigen::AngleAxisd(v[5], Eigen::Vector3d::UnitZ())).toRotationMatrix();
                    bottle_pos_world_  = table_world_ * p_table;
                    bottle_axis_world_ = (table_world_.linear() * R_tb * Eigen::Vector3d::UnitZ()).normalized();
                    bottle_valid_ = true;
                }
        }
    }
    else
    {
        table_world_.linear()      = q_table_w.normalized().toRotationMatrix();
        table_world_.translation() = table_w;
        bottle_pos_world_  = bottle_w;
        bottle_axis_world_ = (q_bottle_w * Eigen::Vector3d::UnitZ()).normalized();
        bottle_valid_ = true;
    }
    scene_world_valid_ = bottle_valid_;   // table + at least one bottle pose seen

    // DSR scene publishing disabled (loop-hitch diagnosis): this controller does NOT read the
    // graph for control — its DSR signal slots are stubs — so the per-cycle
    // insert_or_assign_edge_RT writes below only publish over DDS for other agents/viewer.
    // DDS publishing is a controller-side per-cycle stall source; the pose caching above is
    // what the FSM actually consumes. Re-enable if a sibling agent needs the live edges.
    (void) robot_w; (void) q_robot_w;
    if (not publish_scene_to_graph_) return;   // opt-in: write the scene to the graph (test / siblings)

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

bool SpecificWorker::monitor_bottle_and_recover()
{
    if (not bottle_monitor_ or not scene_world_valid_) return false;
    if (monitor_cooldown_ > 0) { --monitor_cooldown_; return false; }
    // Throttle the GT read off the per-cycle control path: an Ice round-trip every cycle compounds
    // the bridge's loop-jitter stall (the held-velocity arm then lurches and mis-grasps). Every ~12
    // cycles is still well within the time a tip/roll takes to develop.
    if (++monitor_tick_ % 12 != 0) return false;

    RoboCompWebots2Robocomp::ObjectPose b;
    try { b = webots2robocomp_proxy->getObjectPose(WEBOTS_BOTTLE_DEF); }
    catch (const Ice::Exception&) { return false; }   // bridge hiccup: skip this cycle

    const double z = b.position.z / 1000.0;
    // Publish the GT bottle position for the FSM lift-confirm (tracks the bottle while it is held).
    gt_bottle_pos_world_ = Eigen::Vector3d(b.position.x / 1000.0, b.position.y / 1000.0, z);
    gt_bottle_valid_     = true;
    const Eigen::Quaterniond q(b.orientation.w, b.orientation.x, b.orientation.y, b.orientation.z);
    const double tilt = std::acos(std::clamp(std::abs((q.normalized() * Eigen::Vector3d::UnitZ()).z()), 0.0, 1.0));

    // Standing on the table (upright, z≈table): remember the spot to re-stand at.
    if (tilt < BOTTLE_TIP_RAD and std::abs(z - table_top_z_) < 0.05)
    {
        last_upright_bottle_xy_ = Eigen::Vector2d(b.position.x / 1000.0, b.position.y / 1000.0);
        return false;
    }
    // Lifted/held (z well above the table, still upright): a legitimate carry — don't abort.
    if (z > table_top_z_ + 0.05 and tilt < BOTTLE_TIP_RAD)
        return false;

    // Otherwise it has TIPPED (tilt large) or ROLLED off (z below the table). Abort + re-stand.
    std::print("[monitor] GT bottle FELL (tilt={:.0f}° z={:.3f}) — abort + re-stand at ({:.3f},{:.3f})\n",
               tilt * 57.29578, z, last_upright_bottle_xy_.x(), last_upright_bottle_xy_.y());
    gripper_command_ = 1.0f;                                  // open the gripper
    respawn_bottle(last_upright_bottle_xy_.x(), last_upright_bottle_xy_.y());
    if (fsm_) fsm_->reset();                                  // abort the attempt → back to Tracking
    monitor_cooldown_ = 80;                                   // let the re-stand settle + GT update
    return true;
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
    double diameter = 0.056, height = 0.2;   // arm_base.wbt: radius 0.028, height 0.2
    if (bottle_node.has_value())
    {
        if (auto v = G->get_attrib_by_name<width_m_att>(bottle_node.value());  v.has_value()) diameter = v.value();
        if (auto v = G->get_attrib_by_name<height_m_att>(bottle_node.value()); v.has_value()) height   = v.value();
    }
    bottle_radius_m_ = diameter * 0.5;   // cache for the approach-phase bottle obstacle
    bottle_height_m_ = height;

    arm_belief_viewer_->update_scene_objects(corners_world, bottle_pos_world_,
                                             bottle_axis_world_, diameter * 0.5, height);

    // Support column (the Webots mast): a vertical cylinder from the floor (z=0)
    // up to the shoulder (arm base), at the arm-base xy. Matches the SolidPipe in
    // arm_table.wbt (radius 0.05) — the obstacle the arm must reach around.
    const Eigen::Vector3d shoulder = arm_base_world_.translation();
    const Eigen::Vector3d col_base(shoulder.x(), shoulder.y(), 0.0);
    arm_belief_viewer_->set_column(col_base, shoulder, 0.05);
}

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
