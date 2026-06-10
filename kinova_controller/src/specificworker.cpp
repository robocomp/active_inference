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
    try { simple_track_   = configLoader.get<bool>("Controller.simple_track");  } catch (...) {}
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
    // Pragmatic-solve backend: "dls" (closed-form, default) or "qp" (proxQP). With no
    // inequality constraints the QP reproduces DLS — the foothold for migrating
    // joint-limit / obstacle terms to hard constraints. See EFE_CONTROLLER_MATH.md §4.
    try { use_qp_ = (configLoader.get<std::string>("Controller.solver") == "qp"); } catch (...) {}
    try { qp_redundancy_weight_ = configLoader.get<double>("Controller.qp_redundancy_weight"); } catch (...) {}
    try { force_confidence_ = configLoader.get<double>("Controller.force_confidence"); } catch (...) {}
    try { blend_radius_ = configLoader.get<double>("Controller.blend_radius"); } catch (...) {}
    try { use_preference_field_ = configLoader.get<bool>  ("Controller.use_preference_field"); } catch (...) {}
    try { field_prec_pass_      = configLoader.get<double>("Controller.field_prec_pass"); } catch (...) {}
    try { field_prec_stop_      = configLoader.get<double>("Controller.field_prec_stop"); } catch (...) {}
    try { field_prec_ref_       = configLoader.get<double>("Controller.field_prec_ref");  } catch (...) {}
    try { field_overlap_        = configLoader.get<double>("Controller.field_overlap");   } catch (...) {}
    if (use_preference_field_)
        std::print("[field] preference-field mode ON (prec pass={:.1f} stop={:.1f} ref={:.1f} overlap={:.3f})\n",
                   field_prec_pass_, field_prec_stop_, field_prec_ref_, field_overlap_);
    try { tactile_recenter_ = configLoader.get<bool>  ("Controller.tactile_recenter"); } catch (...) {}
    try { recenter_gain_    = configLoader.get<double>("Controller.recenter_gain");    } catch (...) {}
    try { recenter_sign_    = configLoader.get<double>("Controller.recenter_sign");    } catch (...) {}
    if (tactile_recenter_)
        std::print("[tactile] anti-tip re-centering ON (gain={:.3f} m, sign={:+.0f})\n",
                   recenter_gain_, recenter_sign_);
    try { tip_reflex_             = configLoader.get<bool>  ("Controller.tip_reflex");              } catch (...) {}
    if (tip_reflex_)
        std::print("[reflex] tip-bumper stop-and-rectify ON (step={:.3f} max={:.3f} backoff={:.3f} sign={:+.0f})\n",
                   TIP_REFLEX_STEP_M, TIP_REFLEX_MAX_M, TIP_REFLEX_BACKOFF_M, recenter_sign_);
    try { bottle_obstacle_        = configLoader.get<bool>  ("Controller.bottle_obstacle");        } catch (...) {}
    try { bottle_obstacle_margin_ = configLoader.get<double>("Controller.bottle_obstacle_margin"); } catch (...) {}
    if (bottle_obstacle_)
        std::print("[obstacle] bottle-as-obstacle ON in approach phase (margin={:.3f} m)\n",
                   bottle_obstacle_margin_);
    if (force_confidence_ >= 0.0)
        std::print("[experiment] confidence PINNED at {:.2f} (overrides learning/decay)\n", force_confidence_);
    try {
        std::istringstream fp(configLoader.get<std::string>("Controller.fixed_pick_xy"));
        double fx, fy;
        if (fp >> fx >> fy) { fixed_pick_xy_ = {fx, fy}; fixed_pick_set_ = true;
            std::print("[experiment] fixed pick spot = ({:.3f}, {:.3f}) world\n", fx, fy); }
    } catch (...) {}
    try {
        std::istringstream fp(configLoader.get<std::string>("Controller.fixed_place_xy"));
        double fx, fy;
        if (fp >> fx >> fy) { fixed_place_xy_ = {fx, fy}; fixed_place_set_ = true;
            std::print("[experiment] fixed place spot = ({:.3f}, {:.3f}) world\n", fx, fy); }
    } catch (...) {}
    try { elbow_gain_ = configLoader.get<double>("Controller.elbow_gain"); } catch (...) {}
    try {
        std::istringstream et(configLoader.get<std::string>("Controller.elbow_target_xy"));
        double ex, ey;
        if (et >> ex >> ey) { elbow_target_xy_ = {ex, ey}; elbow_target_set_ = true;
            std::print("[experiment] elbow_target = ({:.3f}, {:.3f}) world\n", ex, ey); }
    } catch (...) {}
    std::print("[solver] pragmatic resolved-rate backend = {}{}\n",
               use_qp_ ? "QP (proxQP)" : "DLS (closed form)",
               (use_qp_ and qp_redundancy_weight_ > 0.0)
                   ? std::format("  redundancy_weight={:.4f} (genuine NEO)", qp_redundancy_weight_)
                   : std::string{"  redundancy=projection-equivalent"});
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
    try
    {
        const auto fpath = configLoader.get<std::string>("Controller.fluid_log_path");
        if (not fpath.empty())
        {
            fluid_log_.open(fpath, std::ios::out | std::ios::trunc);
            if (fluid_log_.is_open())
            {
                fluid_log_open_ = true;
                fluid_log_ << "rep,confidence,phase,ee_speed\n";   // one row per ActiveEFE cycle
                std::print("[fluid] per-cycle EE-speed log → {}\n", fpath);
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
    try { standoff_collapse_     = configLoader.get<double>("Controller.standoff_collapse");        } catch (...) {}
    try { insert_conf_gain_      = configLoader.get<double>("Controller.insert_conf_gain");         } catch (...) {}
    standoff_collapse_ = std::clamp(standoff_collapse_, 0.0, 1.0);
    // Retreat sub-skill params + exploration.
    try { retreat_speed_         = configLoader.get<double>("Controller.retreat_speed");            } catch (...) {}
    try { gripper_open_conf_     = configLoader.get<double>("Controller.gripper_open_conf");        } catch (...) {}
    try { release_ticks_         = configLoader.get<int>   ("Controller.release_ticks");            } catch (...) {}
    try { grasp_align_tol_rad_   = configLoader.get<double>("Controller.grasp_align_tol_deg") * M_PI / 180.0; } catch (...) {}
    try { predictive_place_      = configLoader.get<bool>  ("Controller.predictive_place");          } catch (...) {}
    if (predictive_place_) std::print("[predict] predictive place-spot selection ON (μ + mast clearance)\n");
    try { predictive_grasp_      = configLoader.get<bool>  ("Controller.predictive_grasp");          } catch (...) {}
    if (predictive_grasp_) std::print("[predict] predictive grasp selection ON (side azimuths + top-down fallback)\n");
    try { precompute_reach_map_  = configLoader.get<bool>       ("Controller.precompute_reach_map"); } catch (...) {}
    try { reach_map_path_        = configLoader.get<std::string>("Controller.reach_map_path");        } catch (...) {}
    try { probe_retreat_         = configLoader.get<bool>  ("Controller.probe_retreat");            } catch (...) {}
    try { probe_rspeed_amp_      = configLoader.get<double>("Controller.probe_rspeed_amp");         } catch (...) {}
    try { probe_open_amp_        = configLoader.get<double>("Controller.probe_open_amp");           } catch (...) {}
    try
    {
        const auto rpath = configLoader.get<std::string>("Controller.retreat_log_path");
        if (not rpath.empty())
        {
            retreat_log_.open(rpath, std::ios::out | std::ios::trunc);
            if ((retreat_log_open_ = retreat_log_.is_open()))
            {
                retreat_log_ << "probe_idx,retreat_speed,open_thresh,place_x,place_y,post_tilt_deg,tipped\n";
                std::print("[retreat] outcome dataset → {}\n", rpath);
            }
        }
    } catch (...) {}
    load_confidence();
    if (precision_reweighting_)
        std::print("[skill] precision re-weighting ON (confidence={:.2f}, σ_obs={:.3f} m, "
                   "skilled sample/{} cy, speed +{:.0f}%, standoff collapse {:.0f}%, insert +{:.0f}%)\n",
                   confidence_, perception_noise_std_, skilled_sample_period_, speed_conf_gain_*100.0,
                   standoff_collapse_*100.0, insert_conf_gain_*100.0);
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


void SpecificWorker::compute()
{
    if (not kinematics_) return;
    try
    {
        // The one and only call to the KinovaArm proxy in the main loop.  All data
        const auto js = kinovaarm_proxy->getJointsState();
        if (static_cast<int>(js.joints.size()) < Kinematics::N_ARM_JOINTS)
        {
            std::print("[compute] proxy returned only {} joints; expected ≥ {}\n",
                       js.joints.size(), Kinematics::N_ARM_JOINTS);
            return;
        }

        // 
        std::array<double, Kinematics::N_ARM_JOINTS> q{}, tau{}, qd_meas{};
        for (int i = 0; i < Kinematics::N_ARM_JOINTS; ++i)
        {
            q[i]       = js.joints[i].angle;
            tau[i]     = js.joints[i].torque;
            qd_meas[i] = js.joints[i].velocity;
        }
        cur_q_ = q;

        // Log the commanded and measured joint velocities for this cycle, for offline analysis of the actuation patterns across the probe envelope.  Only when logging is enabled and we're in the active phase (not rest-pose or waiting-for-start).
        log_joint_actuation(qd_meas);

        //
        if (tip_log_)
        {
            const Eigen::Matrix<double, 6, 1> w = kinematics_->estimate_tool_wrench(q, tau);
            std::print("[wrench] F=({:.2f},{:.2f},{:.2f}) N  T=({:.3f},{:.3f},{:.3f}) N·m\n",
                       w(0), w(1), w(2), w(3), w(4), w(5));
        }

        send_gripper_command();

        const auto ee_position = kinematics_->forward_kinematics(q);
        update_viewer(q, tau, ee_position);

        if (force_confidence_ >= 0.0) confidence_ = force_confidence_;

        log_fluidity(ee_position);
        handle_joint_dump(q, ee_position, js);

        if (phase_ == Phase::SendingRestPose) { run_sending_rest_pose(q); return; }
        if (phase_ == Phase::Homing)          { run_homing(q); return; }

        update_scene();

        if (phase_ == Phase::WaitingForStart) { run_waiting_for_start(); return; }
        if (not run_requested_)               { abort_to_rest(); return; }

        if (simple_track_)
            run_simple_track(q, ee_position);
        else if (approach_only_)
            run_approach_only(q, ee_position, qd_meas);
        else
            run_grasp_fsm(q, ee_position);

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
// compute() helpers extracted from the monolithic body
////////////////////////////////////////////////////////////////////////////////////////////////

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

void SpecificWorker::log_joint_actuation(
    const std::array<double, Kinematics::N_ARM_JOINTS>& qd_meas)
{
    if (not joint_log_open_ or phase_ != Phase::ActiveEFE) return;
    joint_log_ << (probe_index_ - 1) << ',' << joint_log_cycle_++;
    for (int i = 0; i < Kinematics::N_ARM_JOINTS; ++i) joint_log_ << ',' << last_q_dot_cmd_[i];
    for (int i = 0; i < Kinematics::N_ARM_JOINTS; ++i) joint_log_ << ',' << qd_meas[i];
    joint_log_ << '\n';
}

void SpecificWorker::send_gripper_command()
{
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
    if (arm_belief_viewer_)
    {
        arm_belief_viewer_->update_beliefs(q, viewer_link_poses, reach_target_, ee_position);
        const auto g_tau = kinematics_->arm_gravity_torque(q);
        Eigen::Matrix<double, Kinematics::N_ARM_JOINTS, 1> tau_ext;
        for (int i = 0; i < Kinematics::N_ARM_JOINTS; ++i) tau_ext[i] = tau[i] - g_tau[i];
        const auto J6 = kinematics_->arm_jacobian_full(q);
        const Eigen::Matrix<double, 6, 1> wrench =
            (J6 * J6.transpose() + 1e-6 * Eigen::Matrix<double, 6, 6>::Identity())
                .ldlt().solve(J6 * tau_ext);
        const float wrist_force = static_cast<float>(wrench.head<3>().norm());
        try
        {
            const auto gs = kinovaarm_proxy->getGripperState();
            arm_belief_viewer_->update_forces(gs.lforce, gs.rforce, gs.ltipforce, gs.rtipforce,
                                              wrist_force, gs.ltipcontact, gs.rtipcontact);
        }
        catch (const Ice::Exception&) {}
    }
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

void SpecificWorker::log_fluidity(const Eigen::Vector3d& ee_position)
{
    if (not fluid_log_open_ or phase_ != Phase::ActiveEFE) return;
    const double dt = std::max(1, getPeriod("Compute")) / 1000.0;
    const double sp = fluid_prev_pos_.has_value()
        ? (ee_position - fluid_prev_pos_.value()).norm() / dt : 0.0;
    fluid_log_ << (probe_index_ - 1) << ',' << confidence_ << ','
               << static_cast<int>(grasp_phase_) << ',' << sp << '\n';
    fluid_prev_pos_ = ee_position;
}

void SpecificWorker::run_sending_rest_pose(
    const std::array<double, Kinematics::N_ARM_JOINTS>& q)
{
    kinovaarm_proxy->moveJointsWithAngle(nearest_equiv_target(q, rest_pose_angles_));
    std::print("[homing] Sent rest pose [{:+.3f} {:+.3f} {:+.3f} {:+.3f} {:+.3f} {:+.3f} {:+.3f}] rad\n",
               rest_pose_angles_[0], rest_pose_angles_[1], rest_pose_angles_[2],
               rest_pose_angles_[3], rest_pose_angles_[4], rest_pose_angles_[5],
               rest_pose_angles_[6]);
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

    std::print("[homing] max joint err = {:.4f} rad  ({}/{} settled)\n",
               max_err, homing_settled_ticks_, HOMING_SETTLE_TICKS);

    if (homing_settled_ticks_ < HOMING_SETTLE_TICKS
        and ++homing_elapsed_ticks_ > HOMING_TIMEOUT_TICKS)
    {
        RoboCompKinovaArm::TJointSpeeds stop;
        stop.jointSpeeds.assign(Kinematics::N_ARM_JOINTS, 0.0f);
        try { kinovaarm_proxy->moveJointsWithSpeed(stop); } catch (const Ice::Exception&) {}
        std::print("[homing] TIMEOUT after {} cycles (max err {:.4f} rad) — "
                   "pose unreachable/jammed; halting, waiting for Start.\n",
                   HOMING_TIMEOUT_TICKS, max_err);
        run_requested_       = false;
        returning_for_cycle_ = false;
        phase_               = Phase::WaitingForStart;
        return;
    }

    if (homing_settled_ticks_ >= HOMING_SETTLE_TICKS)
    {
        if (returning_for_cycle_ and run_requested_
            and (round_cycles_ <= 0 or pick_place_cycles_done_ < round_cycles_))
        {
            returning_for_cycle_ = false;
            grasp_phase_         = GraspPhase::Tracking;
            grasp_settle_ticks_  = 0;
            gripper_command_     = 1.0f;
            phase_               = Phase::ActiveEFE;
            begin_rep_probe();
            std::print("[cycle] rest reached → starting pick-and-place {}\n",
                       pick_place_cycles_done_ + 1);
        }
        else if (returning_for_cycle_ and round_cycles_ > 0
                 and pick_place_cycles_done_ >= round_cycles_)
        {
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
}

void SpecificWorker::update_scene()
{
    if (not base_tf_set_) refresh_arm_base_world();
    update_bottle_pose_in_dsr();
    update_viewer_scene_objects();
    if (precompute_reach_map_ and not reach_map_done_ and base_tf_set_)
    {
        compute_reach_map();
        reach_map_done_ = true;
    }
}

void SpecificWorker::run_waiting_for_start()
{
    RoboCompKinovaArm::TJointSpeeds stop;
    stop.jointSpeeds.assign(Kinematics::N_ARM_JOINTS, 0.0f);
    kinovaarm_proxy->moveJointsWithSpeed(stop);
    if (run_requested_)
    {
        std::print("[start] Entering EFE bottle-approach.\n");
        phase_                   = Phase::ActiveEFE;
        grasp_phase_             = GraspPhase::Tracking;
        grasp_settle_ticks_      = 0;
        gripper_command_         = 1.0f;
        pick_place_cycles_done_  = 0;
        begin_rep_probe();
    }
    proxy_unreachable_warned_ = false;
}

void SpecificWorker::abort_to_rest()
{
    std::print("[stop] Run unchecked — returning to rest pose.\n");
    arrived_logged_      = false;
    grasp_phase_         = GraspPhase::Tracking;
    grasp_settle_ticks_  = 0;
    returning_for_cycle_ = false;
    gripper_command_     = 1.0f;
    phase_               = Phase::SendingRestPose;
}

EFEParams SpecificWorker::build_efe_params(
    const Eigen::Vector3d& z_des,
    const Eigen::Vector3d& x_des,
    double v_app) const
{
    EFEParams p;
    p.desired_approach  = z_des;
    p.desired_secondary = x_des;
    // Pin the full grasp frame (align_tool_y=false). The old yaw-free mode left the
    // wrist yaw to the null-space, allowing jaws to settle ~90° off while the metric
    // (which only checked tool+Y) still read "aligned". Full-frame pinning drives
    // tool+X → x_des so the fingers actually straddle the bottle body.
    p.align_tool_y      = false;
    p.desired_tool_y    = z_des.cross(x_des).normalized();
    // Full-frame orientation: pin tool+Z to the bottle AND tool+Y up (= z_des×x_des =
    // bottle vertical axis), so the WRIST CAMERA sits on the upper side. Roll-free mode
    // (gain_secondary=0) left the camera wherever the null-space put it (toolY_up≈0.47).
    // Full-frame used to oscillate on this mount, but the QP rotation slack
    // (orient_slack) now lets the solver target the whole frame and leave any unreachable
    // residual as cheap slack instead of contorting — so the camera holds up without
    // the old instability.
    p.gain_secondary    = 1.0;
    p.C_pos             = Eigen::Vector3d::Ones();
    p.dls_lambda        = 0.05;
    p.use_qp            = use_qp_;   // honour Controller.solver ("qp" → NEO QP); the QP
                                     // enforces column/table/bottle as HARD velocity dampers
    p.obs_damper_xi     = 0.5;
    // Rotation slack (QP path): weight the orientation task-slack well below the
    // position slack so the solver targets the tool rotation but does NOT contort the
    // arm into a near-singular pose to perfect it. Without this the EE touched the
    // standoff then drifted ~13 cm back out while μ collapsed toward 0. Position stays
    // tightly enforced; the few-degree approach-axis error is left as cheap slack.
    // Low slack-weight on orientation = the rotation error is CHEAP to leave, so the
    // QP gives position the DOF it needs and converges cleanly (e≈9 mm, 5–10°). Raising
    // this enforces orientation harder, which re-creates the position-vs-orientation
    // fight and stalls the arm ~10 cm short, 25° off — 0.05 is the validated optimum.
    p.orient_slack      = 0.05;
    p.redundancy_weight = qp_redundancy_weight_;
    p.v_approach        = v_app;
    p.a_approach        = 0.60;
    p.omega_max         = 2.0;
    // Hold tolerance width. The √-deceleration profile has infinite slope at the origin,
    // so a tight deadband leaves the arm hunting (the EE chased ±2-4 cm, grazing/tipping
    // the bottle). In approach_only validation we want the arm to settle DEAD-STILL at the
    // standoff without locking: widen the preference tolerance so the controller commands
    // zero linear/angular twist once inside ~2 cm / ~9° of the goal, then coasts to rest.
    // It is NOT a freeze — if the bottle moves beyond the tolerance the error re-engages
    // the EFE pull. The grasp FSM keeps the tight defaults (precise insertion needs them).
    if (approach_only_)
    {
        p.arrive_deadband = 0.02;   // m;   hold (zero linear flow) within 2 cm of the standoff
        p.orient_deadband = 0.16;   // rad; hold (zero angular flow) within ~9° — stops wrist hunting
    }
    p.gain_mu           = 0.0;
    p.gain_elbow        = elbow_gain_;
    p.elbow_target      = elbow_target_set_
        ? Eigen::Vector3d(elbow_target_xy_.x(), elbow_target_xy_.y(), 0.0)
        : Eigen::Vector3d(-0.625, -1.5, 0.0);
    p.gain_mast         = 3.0;
    p.col_xy            = Eigen::Vector2d(-0.56477, -0.056064);
    p.col_radius        = 0.03;
    p.col_z_lo          = -0.10;
    p.col_z_hi          =  1.30;
    p.col_margin        = 0.06;
    p.gain_table        = 2.0;
    p.table_z           = table_top_z_;
    p.table_safe        = 0.06;
    if (use_qp_ and bottle_obstacle_ and grasp_phase_ == GraspPhase::Tracking)
    {
        p.gain_bottle   = 1.0;
        p.bottle_xy     = bottle_pos_world_.head<2>();
        p.bottle_radius = bottle_radius_m_;
        p.bottle_z_lo   = bottle_pos_world_.z();
        p.bottle_z_hi   = bottle_pos_world_.z() + bottle_height_m_;
        p.bottle_margin = bottle_obstacle_margin_;
    }
    return p;
}

std::pair<double,double> SpecificWorker::efe_drive(
    const std::array<double, Kinematics::N_ARM_JOINTS>& q,
    const Eigen::Vector3d& ee_position,
    const Eigen::Vector3d& target,
    const Eigen::Vector3d& z_des,
    const Eigen::Vector3d& x_des,
    double v_app,
    std::optional<Eigen::Vector3d> blend_next,
    double orient_gain)
{
    reach_target_ = target;
    EFEParams params = build_efe_params(z_des, x_des, v_app);
    params.gain_orient  = orient_gain;
    params.blend_next   = blend_next;
    params.blend_radius = blend_next.has_value() ? skill_c() * blend_radius_ : 0.0;
    if (use_preference_field_ and blend_next.has_value())
    {
        params.use_field    = true;
        params.prec_current = field_prec_stop_
                            + skill_c() * (field_prec_pass_ - field_prec_stop_);
        params.prec_next    = field_prec_stop_;
        params.prec_ref     = field_prec_ref_;
        params.field_overlap = field_overlap_;
    }
    auto q_dot = efe_gradient_step(*kinematics_, q, reach_target_, params);
    // Safety clamp: bridge artifact configurations (joints outside URDF limits) can
    // produce huge null-space velocities despite the internal clip in efe_gradient_step.
    {
        double scale = 1.0;
        for (const auto& v : q_dot)
            if (std::abs(v) > params.max_joint_vel)
                scale = std::min(scale, params.max_joint_vel / std::abs(v));
        if (scale < 1.0)
        {
            for (auto& v : q_dot) v *= scale;
            std::print("[ctrl] velocity overflow clipped (scale={:.4f})\n", scale);
        }
    }
    RoboCompKinovaArm::TJointSpeeds cmd;
    cmd.jointSpeeds.assign(q_dot.begin(), q_dot.end());
    kinovaarm_proxy->moveJointsWithSpeed(cmd);
    last_q_dot_cmd_ = q_dot;

    const double e_pos = (ee_position - target).norm();

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
        const auto tool = kinematics_->tool_pose(q);
        if (params.gain_secondary <= 0.0)
        {
            e_ang = std::acos(std::clamp(tool.rotation.col(2).dot(zc), -1.0, 1.0));
        }
        else
        {
            Eigen::Vector3d xc = x_des - x_des.dot(zc) * zc;
            if (xc.norm() > 1e-6)
            {
                xc.normalize();
                Eigen::Matrix3d R_des;
                R_des.col(0) = xc;
                R_des.col(1) = zc.cross(xc);
                R_des.col(2) = zc;
                const Eigen::AngleAxisd er(R_des * tool.rotation.transpose());
                e_ang = std::abs(er.angle());
            }
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
        const auto sk = kinematics_->arm_skeleton_points(q);
        const Eigen::Vector3d col_lo(-0.56477, -0.056064, -0.10);
        const Eigen::Vector3d col_hi(-0.56477, -0.056064,  1.30);
        double col_min = 1e9; int col_link = -1;
        for (int k = 2; k + 1 < (int)sk.size(); ++k)
        {
            const double dseg = segment_segment_distance(sk[k], sk[k+1], col_lo, col_hi) - 0.05;
            if (dseg < col_min) { col_min = dseg; col_link = k; }
        }
        std::print("[tiplog] {},{:.3f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},"
                   "eang={:.1f},elbow=({:+.3f},{:+.3f},{:+.3f}),colClr={:+.3f},colLink={},probe={}\n",
                   tip_log_cycle_, tip_log_cycle_ * dt,
                   ee_position.x(), ee_position.y(), ee_position.z(),
                   target.x(), target.y(), target.z(), e_pos, vcmd, vmeas,
                   e_ang * 57.29578, elb.x(), elb.y(), elb.z(), col_min, col_link,
                   probe_index_ - 1);
        tip_log_prev_pos_ = ee_position;
        ++tip_log_cycle_;
    }
    return {e_pos, e_ang};
}

////////////////////////////////////////////////////////////////////////////////////////////////
// ── Simple track mode ──────────────────────────────────────────────────────────
// Bypasses all FSM logic. Reads the bottle pose from Webots, computes a standoff
// target with grasping-ready orientation (tool +Z into bottle, tool +X vertical),
// and runs EFE control continuously. The bottle can be moved in Webots and the
// arm will track it.

void SpecificWorker::run_simple_track(
    const std::array<double, Kinematics::N_ARM_JOINTS>& q,
    const Eigen::Vector3d& ee_position)
{
    gripper_command_ = 1.0f;   // keep gripper open

    // Position-only control: drive the EE to a point near the bottle body,
    // with zero orientation pull — the arm only has to point toward the bottle.
    if (not scene_world_valid_ or not base_tf_set_)
    {
        RoboCompKinovaArm::TJointSpeeds stop;
        stop.jointSpeeds.assign(Kinematics::N_ARM_JOINTS, 0.0f);
        kinovaarm_proxy->moveJointsWithSpeed(stop);
        return;
    }

    // Target: a point APPROACH_STANDOFF_M away from the bottle body centre
    // along the radial direction from arm base to bottle (so the EE sits
    // near the bottle, ready for a future grasp).
    const Eigen::Vector3d bottle_pos = bottle_pos_world_;
    const Eigen::Vector3d base_pos   = arm_base_world_.translation();
    Eigen::Vector3d radial = bottle_pos - base_pos;
    radial.z() = 0.0;
    if (radial.norm() < 1e-4) return;
    radial.normalize();
    // Bottle body centre at its world position z plus a small offset for mid-body.
    const Eigen::Vector3d body_centre(bottle_pos.x(), bottle_pos.y(),
                                      bottle_pos.z() + 0.05);  // 5 cm above bottle base
    const Eigen::Vector3d target = body_centre - radial * APPROACH_STANDOFF_M;

    // Drive with orient_gain=0: no orientation constraint, position only.
    const double v_app = 0.25;
    const auto [e_pos, e_ang] = efe_drive(q, ee_position, target,
                                           Eigen::Vector3d::UnitZ(),   // z_des unused with zero orient gain
                                           Eigen::Vector3d::UnitX(),   // x_des unused
                                           v_app, std::nullopt, 0.0);  // orient_gain = 0
    (void) e_ang;

    if (ctrl_cycle_ % 25 == 0 or e_pos < REACH_TOLERANCE_M)
    {
        std::print("[simple] cy={}  bottle=({:.3f},{:.3f},{:.3f})  "
                   "target=({:.3f},{:.3f},{:.3f})  "
                   "ee=({:.3f},{:.3f},{:.3f})  "
                   "e_pos={:.4f}m{}\n",
                   ctrl_cycle_,
                   bottle_pos.x(), bottle_pos.y(), bottle_pos.z(),
                   target.x(), target.y(), target.z(),
                   ee_position.x(), ee_position.y(), ee_position.z(),
                   e_pos,
                   e_pos < REACH_TOLERANCE_M ? "  *** AT TARGET ***" : "");
    }
    ++ctrl_cycle_;
}

////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////


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
    // Fixed arm→body mount = the KinovaGen3 child pose inside the P3Bot Robot node in the
    // world file. In the current arm_base.wbt the KinovaGen3 child is IDENTITY (no
    // translation/rotation), so the arm base coincides with the P3Bot Robot frame — and the
    // Robot is yaw-only about Z, giving a VERTICAL shoulder. (The earlier value baked a 90°
    // rotation about Y + 7 cm drop from the old side-mounted .wbt; that tipped the shoulder
    // horizontal and no joint values could fix it.) Keep this in sync with the .wbt child pose.
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
    if (not scene_world_valid_) return std::nullopt;      // bottle pose not yet received
    const Eigen::Vector3d bottle_pos = bottle_pos_world_;
    const Eigen::Vector3d z_bot      = bottle_axis_world_.normalized();  // ≈ +Z (upright)
    const Eigen::Vector3d base_pos   = arm_base_world_.translation();

    // Aim at the bottle's body, not its base. The WaterBottle PROTO origin is
    // bottom-centre, so offset up the bottle axis by a fraction of its height.
    double bottle_height_m = 0.2;   // arm_base.wbt: height 0.2
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
    const Eigen::Vector2d col_xy(-0.56477, -0.056064);   // real SolidPipe column
    Eigen::Vector3d u = bottle_pos - Eigen::Vector3d(col_xy.x(), col_xy.y(), bottle_pos.z());
    u -= u.dot(z_bot) * z_bot;                            // column→bottle, ⟂ bottle axis
    if (u.norm() < 1e-4) return std::nullopt;
    u.normalize();
    const Eigen::Vector3d perp = z_bot.cross(u).normalized();   // side direction (⟂ both)
    const auto standoff_y = [&](const Eigen::Vector3d& zt)
    { return (body_centre - zt * APPROACH_STANDOFF_M).y(); };
    Eigen::Vector3d z_tool_des = (standoff_y(perp) < standoff_y(-perp)) ? perp : -perp;
    Eigen::Vector3d grasp_centre = body_centre;
    bool top_down = false;

    if (force_top_down_)
    {
        // REACTIVE top-down: the side approach already FAILED to converge in execution (real
        // feedback, not a static prediction), so grasp the cap from straight above — the
        // corner-near-column case where no side approach can reach without hitting something.
        grasp_centre = bottle_pos + z_bot * (bottle_height_m * BOTTLE_TOP_GRASP_FRAC);
        z_tool_des   = -z_bot;   // tool +Z points straight down
        top_down     = true;
    }
    else if (predictive_grasp_)
    {
        // PROACTIVE map gate: predict, BEFORE the first attempt, whether the side approach to
        // this bottle cell is usable. The capability map (multi-seed IK + column/table clearance,
        // recomputed per scene) returns μ at the cell, or ≤0 if unreachable/blocked. If the side
        // is predicted unusable, go TOP-DOWN now instead of failing first. The reactive branch
        // above remains the safety net for prediction error.
        const float mu = reach_lookup(body_centre.x(), body_centre.y());
        if (mu <= 0.0f)
        {
            grasp_centre = bottle_pos + z_bot * (bottle_height_m * BOTTLE_TOP_GRASP_FRAC);
            z_tool_des   = -z_bot;
            top_down     = true;
            std::print("[grasp] map predicts side unusable at bottle cell (μ={:.3f}) → TOP-DOWN before attempt\n", mu);
        }
    }

    // Full grasp frame at convergence:
    //   tool +Z = z_tool_des          approach axis, horizontal into bottle
    //   tool +X = z_bot × z_tool_des  horizontal tangent — fingers close around body
    //   tool +Y = Z × X = z_bot       up (wrist camera on the upper side)
    // Per-rep structured perturbation (skill-learning probe; zero when disabled):
    // rotate the approach azimuth about the bottle axis, and offset the grasp point
    // tangentially and along the axis. This perturbs the COMMAND relative to the
    // PERCEIVED bottle, so the per-rep dataset relates command offset → outcome.
    if (std::abs(rep_perturb_.dazi) > 1e-9 and not top_down)
        z_tool_des = (Eigen::AngleAxisd(rep_perturb_.dazi, z_bot) * z_tool_des).normalized();
    // Finger-closing axis. For a side grasp this is z_bot × z_tool (horizontal tangent); for a
    // TOP-DOWN grasp z_tool = ±z_bot so that cross is ~0 — the fingers close in the horizontal
    // plane, so fall back to `perp`.
    Eigen::Vector3d x_tool_des = z_bot.cross(z_tool_des);
    x_tool_des = (x_tool_des.norm() > 1e-3) ? x_tool_des.normalized() : perp;
    const Eigen::Vector3d grasp_pt =
        grasp_centre + x_tool_des * rep_perturb_.dx_perp + z_bot * rep_perturb_.dz_axis;

    SideGraspTarget out;
    out.z_tool_des    = z_tool_des;
    out.x_tool_des    = x_tool_des;
    out.up_axis       = z_bot;       // lift/place along the bottle long axis (true up), not tool+Y
    out.top_down      = top_down;
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

SpecificWorker::ReachScore
SpecificWorker::predict_reach(const Eigen::Vector3d& pos,
                             const Eigen::Vector3d& z_des, const Eigen::Vector3d& x_des,
                             const std::array<double, Kinematics::N_ARM_JOINTS>& seed)
{
    // Desired tool frame R_des = [x⟂, z×x, z] (z = approach).
    const Eigen::Vector3d zc = z_des.normalized();
    Eigen::Vector3d xc = x_des - x_des.dot(zc) * zc;
    xc = (xc.norm() > 1e-6) ? xc.normalized() : zc.unitOrthogonal();
    Eigen::Matrix3d R_des; R_des.col(0) = xc; R_des.col(1) = zc.cross(xc); R_des.col(2) = zc;

    // Damped-least-squares IK from the given seed — predicts the config to reach `pos`.
    std::array<double, Kinematics::N_ARM_JOINTS> q = seed;
    const auto lims = kinematics_->arm_joint_position_limits();
    double pe = 1e9, oe = 1e9;
    for (int it = 0; it < 40; ++it)
    {
        const auto tp = kinematics_->tool_pose(q);
        const Eigen::Vector3d ep = pos - tp.position;
        const Eigen::AngleAxisd aa(R_des * tp.rotation.transpose());
        const Eigen::Vector3d eo = aa.angle() * aa.axis();
        pe = ep.norm(); oe = eo.norm();
        if (pe < 0.005 and oe < 0.05) break;
        Eigen::Matrix<double, 6, 1> e; e << ep, eo;
        const Eigen::Matrix<double, 6, Kinematics::N_ARM_JOINTS> J = kinematics_->arm_jacobian_full(q);
        const Eigen::Matrix<double, 6, 6> A =
            J * J.transpose() + 0.01 * Eigen::Matrix<double, 6, 6>::Identity();
        Eigen::Matrix<double, Kinematics::N_ARM_JOINTS, 1> dq = J.transpose() * A.ldlt().solve(e);
        const double n = dq.norm();
        if (n > 0.4) dq *= 0.4 / n;                       // step clamp for stability
        for (int j = 0; j < Kinematics::N_ARM_JOINTS; ++j)
        {
            q[j] += dq[j];
            if (lims[j].first < lims[j].second)            // skip the continuous (limit-less) joints
                q[j] = std::clamp(q[j], lims[j].first, lims[j].second);
        }
    }
    // Manipulability + clearances at the predicted config.
    const Eigen::Matrix<double, 6, Kinematics::N_ARM_JOINTS> Jf = kinematics_->arm_jacobian_full(q);
    const double manip = std::sqrt(std::max(0.0, (Jf * Jf.transpose()).determinant()));
    double col = 1e9, tab = 1e9;
    for (int j = 2; j <= 6; ++j)
    {
        const Eigen::Vector3d pj = kinematics_->joint_position(q, j);
        col = std::min(col, std::hypot(pj.x() + 0.56477, pj.y() + 0.056064) - 0.05);  // to mast surface
        tab = std::min(tab, pj.z() - table_top_z_);                                   // above table
    }
    // Feasible = POSITION reached with the approach roughly attainable. The orientation
    // tolerance is loose because the grasp is YAW-FREE (only tool+Y is pinned to the bottle
    // axis); demanding the full frame falsely rejected reachable poses.
    return { (pe < 0.02 and oe < 0.5), manip, col, tab };
}

float SpecificWorker::reach_lookup(double x, double y) const
{
    if (rm_mu_.empty()) return 1.0f;                 // no map computed → don't filter
    const int ix = static_cast<int>(std::lround((x - rm_x0_) / rm_res_));
    const int iy = static_cast<int>(std::lround((y - rm_y0_) / rm_res_));
    if (ix < 0 or ix >= rm_nx_ or iy < 0 or iy >= rm_ny_) return -1.0f;   // off-grid = unusable
    return rm_mu_[static_cast<size_t>(ix) * rm_ny_ + iy];                 // μ, or -1 if unreachable/blocked
}

void SpecificWorker::compute_reach_map()
{
    const double t0 = now_seconds();
    std::ofstream f(reach_map_path_, std::ios::out | std::ios::trunc);
    if (not f.is_open()) { std::print("[reachmap] cannot open {}\n", reach_map_path_); return; }
    f << "x,y,reachable,manip,col_clear\n";
    const Eigen::Vector3d base = arm_base_world_.translation();
    const Eigen::Vector3d up(0.0, 0.0, 1.0);
    const double gz = table_top_z_ + 0.10;     // grasp height (bottle body centre over the table)
    // Multi-seed for reliability (single-seed IK falls into local minima): rest pose, the live
    // config, and two spread perturbations. A cell is reachable if ANY seed converges.
    std::vector<std::array<double, Kinematics::N_ARM_JOINTS>> seeds = {rest_pose_angles_, cur_q_};
    auto sp = rest_pose_angles_, sm = rest_pose_angles_;
    for (int j = 0; j < Kinematics::N_ARM_JOINTS; ++j) { sp[j] += 0.5; sm[j] -= 0.5; }
    seeds.push_back(sp); seeds.push_back(sm);

    rm_x0_ = -0.40; rm_y0_ = -0.90; rm_res_ = 0.05;
    rm_nx_ = int(std::lround((0.40 - rm_x0_) / rm_res_)) + 1;     // 17 cols (X)
    rm_ny_ = int(std::lround((0.90 - rm_y0_) / rm_res_)) + 1;     // 37 rows (Y)
    rm_mu_.assign(static_cast<size_t>(rm_nx_) * rm_ny_, -1.0f);
    int ncells = 0, nreach = 0;
    for (int ix = 0; ix < rm_nx_; ++ix)
        for (int iy = 0; iy < rm_ny_; ++iy)
        {
            const double x = rm_x0_ + ix * rm_res_, y = rm_y0_ + iy * rm_res_;
            const Eigen::Vector3d c(x, y, gz);
            // A cell is graspable if SOME side approach works (the real grasp picks an azimuth),
            // so try a few: radial from the base, and ⟂ to the column→cell line (both signs) —
            // matching compute_side_grasp_target. Single-orientation maps false-flag cells.
            Eigen::Vector3d rad = c - base; rad.z() = 0.0;
            rad = (rad.norm() > 1e-6) ? rad.normalized() : Eigen::Vector3d(1, 0, 0);
            Eigen::Vector3d u(x + 0.56477, y + 0.056064, 0.0);          // column→cell, horizontal
            u = (u.norm() > 1e-6) ? u.normalized() : rad;
            const Eigen::Vector3d perp = up.cross(u).normalized();
            const std::array<Eigen::Vector3d, 3> zts{rad, perp, -perp};
            bool reach = false; double best_mu = 0.0, best_col = 0.0;
            for (const auto& zt : zts)
            {
                const Eigen::Vector3d xt = up.cross(zt).normalized();
                for (const auto& seed : seeds)
                {
                    const ReachScore s = predict_reach(c, zt, xt, seed);
                    // Usable = IK converged AND clears the column by a real MARGIN (not by a hair)
                    // AND is well-CONDITIONED (μ-floor). Bare col_clear>0 + reach was too optimistic:
                    // it blessed left-near-column cells that the controller, fighting the column
                    // damper, struggles to execute. Margin + μ-floor make the map match reality.
                    const bool ok = s.feasible and s.col_clear > 0.05 and s.table_clear > -0.02
                                    and s.manip > 0.025;
                    if (ok and s.manip > best_mu) { reach = true; best_mu = s.manip; best_col = s.col_clear; }
                }
            }
            rm_mu_[static_cast<size_t>(ix) * rm_ny_ + iy] = reach ? static_cast<float>(best_mu) : -1.0f;
            f << x << ',' << y << ',' << (reach ? 1 : 0) << ',' << best_mu << ',' << best_col << '\n';
            ++ncells; nreach += reach ? 1 : 0;
        }
    f.close();
    const double dt = now_seconds() - t0;
    std::print("[reachmap] {} cells, {} reachable ({:.0f}%), {} seeds/cell → {} in {:.3f} s\n",
               ncells, nreach, 100.0 * nreach / std::max(1, ncells), seeds.size(), reach_map_path_, dt);
}

void SpecificWorker::sample_place_spot()
{
    // Sampled at Closing→Lifting (was: at lift-confirm) so place_hover_ is known DURING the
    // lift — the look-ahead blend needs the next waypoint while still rising. Picks a spot on
    // the RIGHT-side table box (world frame, up=+Z, table horizontal): (a) ≥ PLACE_MIN_MOVE_M
    // from the pick and (b) within reach (≤ PLACE_REACH_MAX_M), at the grasp height so the
    // bottle base sets back on the table. R2 (Roberts) low-discrepancy, first valid draw, the
    // index offset off the spawn stream so pick and place spots don't correlate.
    const auto& g = latched_grasp_;
    const Eigen::Vector3d up(0.0, 0.0, 1.0);
    constexpr double a1 = 0.7548776662466927, a2 = 0.5698402909980532;
    const Eigen::Vector3d base_xyz = arm_base_world_.translation();
    Eigen::Vector3d p = g.grasp_pos;
    bool have_valid = false;     // geometric fallback = the first reachable draw (always a valid spot)
    double best = -1e18;
    // Controlled-experiment override: pin the place spot so the whole pick→place is
    // repeatable. Skips both the R2 sweep and the predictive scorer.
    for (int k = 0; fixed_place_set_ ? false : k < 24; ++k)
    {
        const double idx = static_cast<double>(probe_index_ + 104729 + k * 11939);
        const double uu = std::fmod(0.5 + a1 * idx, 1.0);
        const double vv = std::fmod(0.5 + a2 * idx, 1.0);
        const Eigen::Vector3d c(PLACE_X_MIN + uu * (PLACE_X_MAX - PLACE_X_MIN),
                                PLACE_Y_MIN + vv * (PLACE_Y_MAX - PLACE_Y_MIN),
                                g.grasp_pos.z());
        const double far_from_pick = (c.head<2>() - g.grasp_pos.head<2>()).norm();
        const double reach = (c - base_xyz).norm();
        if (far_from_pick < PLACE_MIN_MOVE_M or reach > PLACE_REACH_MAX_M) continue;
        if (not predictive_place_) { p = c; break; }   // legacy: first valid draw
        if (not have_valid) { p = c; have_valid = true; }   // fall back to first reachable if none score
        // Capability-map pre-filter (O(1) lookup, reliable): the precomputed map gives μ at this
        // cell, or -1 if unreachable / column-blocked. Skip unusable spots; among usable ones,
        // pick the best-conditioned (highest μ). Replaces the flaky single-seed online IK.
        const float mu = reach_lookup(c.x(), c.y());
        if (mu <= 0.0f) continue;
        if (mu > best) { best = mu; p = c; }
    }
    if (fixed_place_set_)
    {
        p = Eigen::Vector3d(fixed_place_xy_.x(), fixed_place_xy_.y(), g.grasp_pos.z());
        std::print("[place] fixed spot ({:.3f},{:.3f})\n", p.x(), p.y());
    }
    else if (predictive_place_)
        std::print("[place] predictive spot ({:.3f},{:.3f}) score {:.4f}\n", p.x(), p.y(), best);
    place_pos_      = p;
    place_world_xy_ = p;                       // for the retreat dataset
    place_hover_    = p + up * LIFT_HEIGHT_M;
    Eigen::Vector3d radial = place_pos_ - arm_base_world_.translation();
    radial.z() = 0.0;
    if (g.top_down)
    {
        // Bottle held from above (gripper pointing down): set it down by lowering straight
        // down, gripper stays vertical, fingers close in the horizontal plane.
        place_z_des_ = -up;                                              // tool +Z = straight down
        place_x_des_ = (radial.norm() > 1e-6) ? radial.normalized() : g.x_tool_des;
    }
    else
    {
        // Side hold: tool +Z horizontal toward the spot, tool +Y = up (bottle upright).
        place_z_des_ = (radial.norm() > 1e-6) ? radial.normalized() : g.z_tool_des;
        place_x_des_ = up.cross(place_z_des_).normalized();
    }
}

void SpecificWorker::begin_rep_probe()
{
    // Called once at the start of each pick attempt. Advances the low-discrepancy
    // sequence and sets this rep's perturbation. With probing off the perturbation
    // is identically zero, so the grasp is exactly the nominal one.
    rep_track_ticks_ = 0;
    rep_attempts_    = 0;
    track_stuck_ticks_ = 0;
    track_noprog_ticks_ = 0; track_best_dist_ = 1e9;   // fresh no-progress abort window
    force_top_down_ = false;   // each new bottle is tried SIDE-first; top-down is the reactive fallback
    fluid_prev_pos_.reset();   // don't carry EE speed across the homing gap between reps
    // Reset the per-rep fusion belief + metrics. cycles_since_obs_ large ⇒ the first
    // Tracking cycle takes an observation to seed the model belief.
    belief_valid_     = false;
    cycles_since_obs_ = 1 << 20;
    obs_count_rep_    = 0;
    rep_t0_           = now_seconds();

    // Deterministic pickup location for this rep (Halton sweep over the right-side
    // spawn box), teleported in via the bridge — guarantees a fresh upright bottle
    // at a known spot regardless of where the last rep left/dropped it.
    if (respawn_each_rep_ and fixed_pick_set_)
    {
        // Controlled-experiment override: same pick spot every rep, so an A/B over one
        // knob (e.g. qp_redundancy_weight) isn't confounded by pick-location variance.
        respawn_bottle(fixed_pick_xy_.x(), fixed_pick_xy_.y());
    }
    else if (respawn_each_rep_)
    {
        // R2 (Roberts) low-discrepancy sequence — uniform 2-D coverage without the
        // axis-correlation Halton suffers at low indices with large bases (which made
        // earlier spawns march in a diagonal line). a1,a2 = 1/φ₂, 1/φ₂² (plastic).
        constexpr double a1 = 0.7548776662466927, a2 = 0.5698402909980532;
        // Sample the (far-reaching) box but reject points outside the reach band, so the
        // far half-table gets used without ever spawning an ungraspable bottle. On a
        // reject, skip far ahead on the R2 sequence (coprime stride) so kept points stay
        // low-discrepancy instead of clustering at the first valid neighbour.
        const Eigen::Vector3d base_xyz = arm_base_world_.translation();
        double sx = 0.0, sy = SPAWN_Y_MAX;
        for (int k = 0; k < 24; ++k)
        {
            const double idx = static_cast<double>(probe_index_ + k * 11939);
            const double u = std::fmod(0.5 + a1 * idx, 1.0);
            const double v = std::fmod(0.5 + a2 * idx, 1.0);
            sx = SPAWN_X_MIN + u * (SPAWN_X_MAX - SPAWN_X_MIN);
            sy = SPAWN_Y_MIN + v * (SPAWN_Y_MAX - SPAWN_Y_MIN);
            const double reach = std::hypot(sx - base_xyz.x(), sy - base_xyz.y());
            if (reach >= SPAWN_REACH_MIN_M and reach <= SPAWN_REACH_MAX_M) break;
        }
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

    if (probe_retreat_)
    {
        // Explore the retreat's timing/speed (direction is fixed = the gripper axis).
        // Halton bases 11/13, disjoint from the grasp probe's bases, so the two probes
        // stay decorrelated within a rep.
        retreat_perturb_.dspeed = (halton(probe_index_, 11) - 0.5) * 2.0 * probe_rspeed_amp_;
        retreat_perturb_.dopen  = (halton(probe_index_, 13) - 0.5) * 2.0 * probe_open_amp_;
        std::print("[probe] rep {} retreat: speed×{:.2f} open_thresh{:+.2f}\n",
                   probe_index_, 1.0 + retreat_perturb_.dspeed, retreat_perturb_.dopen);
    }
    else retreat_perturb_ = RetreatPerturbation{};

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

void SpecificWorker::log_retreat_outcome(double tilt_deg, bool tipped)
{
    if (not retreat_log_open_) return;
    retreat_log_ << (probe_index_ - 1) << ','
                 << (retreat_speed_ * (1.0 + retreat_perturb_.dspeed)) << ','
                 << (gripper_open_conf_ + retreat_perturb_.dopen) << ','
                 << place_world_xy_.x() << ',' << place_world_xy_.y() << ','
                 << tilt_deg << ',' << (tipped ? 1 : 0) << '\n';
    retreat_log_.flush();
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

