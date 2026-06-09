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
        cur_q_ = q;   // latest config — IK seed for the predictive target scorer

        // Map a desired joint-target onto the nearest equivalent angle for the four
        // CONTINUOUS joints (the infinite-limit revolutes), so a position command
        // never asks the motor to unwind — or wind up — accumulated revolutions: the
        // commanded target stays within ±π of the current encoder value (j5 has been
        // seen at +26 rad ≈ 4 turns). Bounded joints are passed through untouched.
        // Routing every moveJointsWithAngle below through this keeps the continuous
        // joints from accumulating turns during normal operation, without putting
        // hard stops in the model.
        auto nearest_equiv_target = [&](const std::array<double, Kinematics::N_ARM_JOINTS> &desired)
        {
            constexpr double TWO_PI = 6.283185307179586;
            const auto lims = kinematics_->arm_joint_position_limits();
            RoboCompKinovaArm::TJointAngles target;
            target.jointAngles.resize(Kinematics::N_ARM_JOINTS);
            for (int i = 0; i < Kinematics::N_ARM_JOINTS; ++i)
            {
                double t = desired[i];
                if (not std::isfinite(lims[i].first) or not std::isfinite(lims[i].second))
                    t += std::round((q[i] - t) / TWO_PI) * TWO_PI;   // nearest equivalent
                target.jointAngles[i] = static_cast<float>(t);
            }
            return target;
        };

        // Recovery primitive: snap the arm straight back to the rest pose through the
        // supervisor (no dynamics, no swept collision), used when a jam is detected so
        // a single stuck grasp can't hang a long unattended round. Unlike a driven
        // re-home, this works even from a pose wedged against the table.
        auto teleport_to_rest = [&]()
        {
            RoboCompKinovaArm::TJointAngles rest;
            rest.jointAngles.assign(rest_pose_angles_.begin(), rest_pose_angles_.end());
            try { webots2robocomp_proxy->setArmJointsInstant(rest); }
            catch (const Ice::Exception &e)
            { std::print(stderr, "[recovery] setArmJointsInstant failed: {}\n", e.what()); }
            std::print("[recovery] arm teleported to rest pose (jam recovery)\n");
        };

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
        {
            arm_belief_viewer_->update_beliefs(q, viewer_link_poses, reach_target_, ee_position);
            // Estimated EXTERNAL wrist wrench from the measured joint torques, gravity-
            // compensated: τ_ext = τ − τ_gravity(q); w = (J6 J6ᵀ)⁻¹ J6 τ_ext. Its force
            // magnitude is the load the environment puts on the hand — a held bottle shows a
            // steady ~weight, an empty/dropped hand ~0. Plotted as wristFz; usable as a
            // "bottle is being held" signal complementing the post-lift rise confirm.
            const auto g_tau = kinematics_->arm_gravity_torque(q);
            Eigen::Matrix<double, Kinematics::N_ARM_JOINTS, 1> tau_ext;
            for (int i = 0; i < Kinematics::N_ARM_JOINTS; ++i) tau_ext[i] = tau[i] - g_tau[i];
            const auto J6 = kinematics_->arm_jacobian_full(q);
            const Eigen::Matrix<double, 6, 1> wrench =
                (J6 * J6.transpose() + 1e-6 * Eigen::Matrix<double, 6, 6>::Identity())
                    .ldlt().solve(J6 * tau_ext);
            const float wrist_force = static_cast<float>(wrench.head<3>().norm());
            // Stream the gripper force channels + wrist wrench into the viewer's time-series.
            try
            {
                const auto gs = kinovaarm_proxy->getGripperState();
                arm_belief_viewer_->update_forces(gs.lforce, gs.rforce, gs.ltipforce, gs.rtipforce,
                                                  wrist_force, gs.ltipcontact, gs.rtipcontact);
            }
            catch (const Ice::Exception&) { /* proxy hiccup — skip this sample */ }
        }

        if (force_confidence_ >= 0.0) confidence_ = force_confidence_;  // controlled experiments: pin

        // Fluidity instrumentation: one row per ActiveEFE cycle (rep, confidence, FSM
        // sub-phase, EE speed). Offline → SPARC + stop-count over the whole pick-place,
        // measured against confidence. fluid_prev_pos_ is reset at each rep start.
        if (fluid_log_open_ and phase_ == Phase::ActiveEFE)
        {
            const double dt = std::max(1, getPeriod("Compute")) / 1000.0;
            const double sp = fluid_prev_pos_.has_value()
                ? (ee_position - fluid_prev_pos_.value()).norm() / dt : 0.0;
            fluid_log_ << (probe_index_ - 1) << ',' << confidence_ << ','
                       << static_cast<int>(grasp_phase_) << ',' << sp << '\n';
            fluid_prev_pos_ = ee_position;
        }

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
            // NEAREST the current encoder value (see nearest_equiv_target above),
            // so the motor doesn't unwind accumulated revolutions.
            kinovaarm_proxy->moveJointsWithAngle(nearest_equiv_target(rest_pose_angles_));
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
            // Re-assert the rest-pose target every cycle. Sending it once is not enough:
            // if the arm drifts (e.g. it sagged under gravity while no controller was
            // commanding it, between rounds/restarts), a one-shot position command does
            // not pull it back and the error climbs. Continuously commanding the target
            // keeps driving the arm home and holds it there. Continuous joints are mapped
            // to the nearest equivalent of the current encoder value so they don't unwind.
            kinovaarm_proxy->moveJointsWithAngle(nearest_equiv_target(rest_pose_angles_));

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

        // Precompute the capability/reachability map once, after the base + table are known
        // (it bakes in THIS static scene: arm pose, column, table). Recompute per mission when
        // the environment changes. Saved to reach_map_path_; timing reported.
        if (precompute_reach_map_ and not reach_map_done_ and base_tf_set_)
        {
            compute_reach_map();
            reach_map_done_ = true;
        }

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
            // FULL-FRAME grasp orientation. The azimuth is ALREADY chosen in
            // compute_side_grasp_target (it picks the approach side to keep the forearm off
            // the mast), so x_des/z_des fully determine the grasp frame — we must PIN it.
            // The old yaw-free mode (align_tool_y) pinned only tool +Y to the bottle axis and
            // left the wrist yaw to the redundancy; that let the finger-closing axis settle
            // ~90° off (broadside to the bottle) while the metric — which only checked tool +Y
            // — still read "aligned". Pinning the full frame drives tool +X → x_des (jaws ⟂ the
            // approach) so the fingers actually straddle the body. Column clearance comes from
            // the null-space elbow term, not from floating the gripper yaw.
            p.align_tool_y      = false;
            p.desired_tool_y    = z_des.cross(x_des).normalized();
            // Approach-axis only: pin tool+Z into the bottle; leave wrist roll free.
            // Full-frame (gain_secondary=1) can't converge on the new vertical mount —
            // the arm reaches the standoff ~25° off in roll and oscillates. For a
            // cylindrical bottle any wrist roll still straddles the body; roll is handled
            // by the null-space elbow term. Restore to 1.0 if per-phase roll pinning is needed.
            p.gain_secondary    = 0.0;
            p.C_pos             = Eigen::Vector3d::Ones();  // straight-line flow
            p.dls_lambda        = 0.05;
            p.use_qp            = false;   // force DLS: QP skips the column/table repulsion terms entirely
            p.obs_damper_xi     = 0.5;       // QP path: max approach speed (m/s) at the obstacle band edge
            p.redundancy_weight = qp_redundancy_weight_;  // QP: μ/elbow linear-term weight (≤0 ⇒ λ²)
            p.v_approach        = v_app;
            p.a_approach        = 0.60;
            p.omega_max         = 2.0;
            // Null-space elbow placement: pull the elbow into the BACK-RIGHT zone
            // (behind the column and to the robot's right, away from the mast)
            // using the redundant DOF — doesn't disturb the hand. Target = column
            // xy pushed −0.25 m in X (back) and −0.30 m in Y (right).
            p.gain_mu           = 0.0;
            p.gain_elbow        = elbow_gain_;   // Controller.elbow_gain (0 ⇒ no elbow term)
            // Posture prior target. Controllable via Controller.elbow_target_xy for tuning.
            // Default = far −Y (the natural away-from-mast direction: the elbow hangs to
            // the robot's right at world y≈−0.6, same x as the mast). A far, saturating
            // target makes the redundant DOF push the elbow as far right as the null space
            // allows ⇒ MAX mast clearance. The old back-left target (−0.875,−0.356) was
            // only 0.34 m from the mast and actively dragged the elbow inward (0.47 m);
            // this raises it to ~0.57 m at the same pick with grasps intact (controlled
            // sweep, see EFE_CONTROLLER_MATH.md / experiments).
            p.elbow_target      = elbow_target_set_
                ? Eigen::Vector3d(elbow_target_xy_.x(), elbow_target_xy_.y(), 0.0)
                : Eigen::Vector3d(-0.625, -1.5, 0.0);
            // WHOLE-ARM column repulsion: push every movable joint (j3..j7) off the
            // REAL column (SolidPipe in arm_table.wbt) — axis, radius, z-extent —
            // not just the elbow. This catches the wrist/forearm, which was the
            // part actually grazing the mast.
            p.gain_mast         = 3.0;
            p.col_xy            = Eigen::Vector2d(-0.56477, -0.056064);  // SolidPipe axis
            p.col_radius        = 0.03;           // SolidPipe r=0.02 + 1 cm margin
            p.col_z_lo          = -0.10;          // SolidPipe: center z=0.6, height 1.4
            p.col_z_hi          =  1.30;
            p.col_margin        = 0.06;           // new vertical mount: arm clears column at ~8.5 cm; 10 cm was always on (wrist repulsion → orientation non-convergence)
            // Hand-table repulsion: keep the tool from diving into the table.
            p.gain_table        = 2.0;
            p.table_z           = table_top_z_;
            p.table_safe        = 0.06;           // start pushing up within 6 cm of the surface
            // Bottle-as-obstacle: ONLY in the first (Tracking) approach phase — the tool goes
            // to the standoff (outside the bottle), so the damper just stops the gripper
            // clipping/tipping it; OFF from Inserting on (the grasp must move into the bottle).
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
        };

        // Run one coordinated EFE step toward (target, z_des, x_des) at v_app,
        // send it, and return {position error, orientation geodesic angle}.
        const auto drive = [&](const Eigen::Vector3d& target,
                               const Eigen::Vector3d& z_des,
                               const Eigen::Vector3d& x_des,
                               double v_app,
                               std::optional<Eigen::Vector3d> blend_next = std::nullopt,
                               double orient_gain = 1.0) -> std::pair<double, double>
        {
            reach_target_ = target;
            EFEParams params = make_params(z_des, x_des, v_app);
            params.gain_orient = orient_gain;   // <1 relaxes the orientation pull (place needs less precision than pick)
            // Look-ahead coarticulation at a transit via-point: steer toward `blend_next`
            // within a skill-gated radius (inert for a novice ⇒ exact terminal stop).
            params.blend_next   = blend_next;
            params.blend_radius = blend_next.has_value() ? skill_c() * blend_radius_ : 0.0;
            // Option (c) prototype: the SAME transit via, as a 2-via precision field instead
            // of the hand-coded blend. The current via's precision interpolates stop→pass with
            // skill (novice ⇒ prec_stop ⇒ exact stop = parity; skilled ⇒ prec_pass ⇒ cruise-
            // through = emergent blend). Mutually exclusive with the blend branch (use_field
            // is checked first in efe_gradient_step).
            if (use_preference_field_ and blend_next.has_value())
            {
                params.use_field    = true;
                params.prec_current = field_prec_stop_
                                    + skill_c() * (field_prec_pass_ - field_prec_stop_);
                params.prec_next    = field_prec_stop_;
                params.prec_ref     = field_prec_ref_;
                params.field_overlap = field_overlap_;
            }
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
                const auto tool = kinematics_->tool_pose(q);
                if (params.gain_secondary <= 0.0)
                {
                    // Approach-axis only: angle between tool+Z and z_des.
                    // Matches the controller's single-axis mode so the commit
                    // gate measures what the controller actually minimises.
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
                // Whole-arm clearance to the ACTUAL column (SolidPipe): min over
                // all arm links of segment-segment distance to the column axis,
                // minus the column radius. Negative ⇒ penetrating.
                const auto sk = kinematics_->arm_skeleton_points(q);
                const Eigen::Vector3d col_lo(-0.56477, -0.056064, -0.10);
                const Eigen::Vector3d col_hi(-0.56477, -0.056064,  1.30);
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

        // ── APPROACH-ONLY VALIDATION: flat EFE step + dense instrument log ────────────
        gripper_command_ = 1.0f;  // open throughout

        auto g_opt = compute_side_grasp_target();
        if (not g_opt.has_value())
        {
            RoboCompKinovaArm::TJointSpeeds stop;
            stop.jointSpeeds.assign(Kinematics::N_ARM_JOINTS, 0.0f);
            kinovaarm_proxy->moveJointsWithSpeed(stop);
            if (ctrl_cycle_ % 25 == 0)
                std::print("[ctrl] cy={} NO TARGET — "
                           "base_tf={} scene={} "
                           "bottle=({:.3f},{:.3f},{:.3f})\n",
                           ctrl_cycle_, base_tf_set_, scene_world_valid_,
                           bottle_pos_world_.x(), bottle_pos_world_.y(), bottle_pos_world_.z());
            ++ctrl_cycle_;
            return;
        }
        const auto& g = g_opt.value();

        // EFE step (DLS resolved-rate with elbow+mast+table constraints).
        const auto [e_pos, e_ang] = drive(g.stand_off_pos, g.z_tool_des, g.x_tool_des, 0.25);

        // ── Dense per-cycle instrument ────────────────────────────────────────────────
        {
            const auto J6   = kinematics_->arm_jacobian_full(q);
            const double mu = std::sqrt(std::max(0.0, (J6 * J6.transpose()).determinant()));
            const auto elb  = kinematics_->elbow_position(q);
            // Whole-arm column clearance (min over links 2..N-1)
            const auto sk   = kinematics_->arm_skeleton_points(q);
            const Eigen::Vector3d col_lo(-0.56477, -0.056064, -0.10);
            const Eigen::Vector3d col_hi(-0.56477, -0.056064,  1.30);
            double col_min = 1e9;
            for (int k = 2; k + 1 < (int)sk.size(); ++k)
                col_min = std::min(col_min,
                    segment_segment_distance(sk[k], sk[k+1], col_lo, col_hi) - 0.05);

            const bool arrived = e_pos < REACH_TOLERANCE_M and e_ang < grasp_align_tol_rad_;
            std::print("[ctrl] cy={:4d}  "
                       "bot({:.3f},{:.3f},{:.3f})  "
                       "tgt({:.3f},{:.3f},{:.3f})  "
                       "ee({:.3f},{:.3f},{:.3f})  "
                       "e={:.4f}m {:.1f}°  "
                       "mu={:.4f}  col={:.3f}m{}\n",
                       ctrl_cycle_,
                       bottle_pos_world_.x(), bottle_pos_world_.y(), bottle_pos_world_.z(),
                       g.stand_off_pos.x(),   g.stand_off_pos.y(),   g.stand_off_pos.z(),
                       ee_position.x(),       ee_position.y(),       ee_position.z(),
                       e_pos, e_ang * 57.29578, mu, col_min,
                       arrived ? "  *** AT STANDOFF ***" : "");
            // Full joint data every 10 cycles or when converged
            if (ctrl_cycle_ % 10 == 0 or arrived)
            {
                std::print("[ctrl]  q  =[{:+.3f},{:+.3f},{:+.3f},{:+.3f},{:+.3f},{:+.3f},{:+.3f}]\n",
                           q[0],q[1],q[2],q[3],q[4],q[5],q[6]);
                std::print("[ctrl]  cmd=[{:+.3f},{:+.3f},{:+.3f},{:+.3f},{:+.3f},{:+.3f},{:+.3f}]\n",
                           last_q_dot_cmd_[0],last_q_dot_cmd_[1],last_q_dot_cmd_[2],
                           last_q_dot_cmd_[3],last_q_dot_cmd_[4],last_q_dot_cmd_[5],
                           last_q_dot_cmd_[6]);
                std::print("[ctrl]  meas=[{:+.3f},{:+.3f},{:+.3f},{:+.3f},{:+.3f},{:+.3f},{:+.3f}]\n",
                           qd_meas[0],qd_meas[1],qd_meas[2],
                           qd_meas[3],qd_meas[4],qd_meas[5],qd_meas[6]);
                std::print("[ctrl]  elb=({:.3f},{:.3f},{:.3f})  "
                           "z_tool_des=({:.3f},{:.3f},{:.3f})\n",
                           elb.x(), elb.y(), elb.z(),
                           g.z_tool_des.x(), g.z_tool_des.y(), g.z_tool_des.z());
            }
            ++ctrl_cycle_;
        }
#if 0  // entire FSM disabled for approach-only validation
        switch (grasp_phase_)
        {
// ── Retracting ──
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
                // Jam watchdog: Tracking is the only grasp phase with no natural
                // timeout, so an arm wedged against the table during the approach would
                // loop here forever (this is what hung the unattended round). On stall,
                // teleport-recover to rest and route into the normal miss handling.
                if (++track_stuck_ticks_ > TRACK_TIMEOUT_TICKS)
                {
                    std::print("[recovery] Tracking stalled {} cy — possible jam; "
                               "teleport to rest, then retry/give up\n", track_stuck_ticks_);
                    teleport_to_rest();
                    track_stuck_ticks_ = 0;
                    miss_or_give_up("track stalled — possible jam");
                    break;
                }
                // Once approach_only_ hold is established, freeze on the first standoff —
                // don't re-track the bottle if it falls/rotates, which would produce an
                // impossible standoff and drag the arm off its hold position.
                if (approach_only_ and approach_hold_logged_)
                {
                    track_stuck_ticks_ = 0;   // prevent jam-watchdog from firing while holding
                    const auto [e_pos, e_ang] =
                        drive(held_track_target_, held_z_des_, held_x_des_, 0.35);
                    if (rep_track_ticks_ % 50 == 1)
                        std::print("[track] FROZEN HOLD ticks={} ee=({:.3f},{:.3f},{:.3f}) "
                                   "e_pos={:.3f}m e_ang={:.1f}°\n",
                                   rep_track_ticks_,
                                   ee_position.x(), ee_position.y(), ee_position.z(),
                                   e_pos, e_ang * 57.29578);
                    break;
                }

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
                // Phase-collapse: slide the Tracking waypoint from the full standoff
                // toward the grasp point as skill rises, so the slow Inserting crawl that
                // follows shrinks (skilled = one continuous reach; novice = reach to a
                // safe standoff, then creep in). The EFE deadband still decelerates into
                // the now-closer waypoint, and Inserting's force/contact gate is unchanged,
                // so a collapsed approach still seats softly.
                const double c = skill_c();
                const Eigen::Vector3d track_target =
                    g.grasp_pos + (g.stand_off_pos - g.grasp_pos) * (1.0 - standoff_collapse_ * c);
                // No coarticulation here: standoff→grasp is collinear (not a corner) and
                // ends in a precision contact needing alignment — its smoothness is the
                // standoff_collapse merge's job, not the look-ahead blend's.
                const auto [e_pos, e_ang] =
                    drive(track_target, g.z_tool_des, g.x_tool_des, 0.35 * vscale);

                // Tracking diagnostic: bottle world pos, standoff target, EE position, errors.
                if (rep_track_ticks_ % 50 == 1)
                {
                    const Eigen::Vector3d& bp = bottle_pos_world_;
                    std::print("[track] ticks={} bottle=({:.3f},{:.3f},{:.3f}) "
                               "standoff=({:.3f},{:.3f},{:.3f}) ee=({:.3f},{:.3f},{:.3f}) "
                               "e_pos={:.3f}m e_ang={:.1f}°\n",
                               rep_track_ticks_,
                               bp.x(), bp.y(), bp.z(),
                               track_target.x(), track_target.y(), track_target.z(),
                               ee_position.x(), ee_position.y(), ee_position.z(),
                               e_pos, e_ang * 57.29578);
                }

                // Fast non-convergence abort: if the EE stops making progress toward the target
                // while still far from it, the pose is effectively unreachable (singularity /
                // obstacle-damper conflict / yaw-free redundancy with no smooth solution). Give
                // up NOW with a clear message instead of thrashing in front of the bottle until
                // the 18 s jam watchdog × 3 attempts (≈ the 47 s "bubbling" reported).
                if (e_pos < track_best_dist_ - 0.004)
                {
                    track_best_dist_ = e_pos;
                    track_noprog_ticks_ = 0;
                }
                else if (e_pos > 3.0 * REACH_TOLERANCE_M and ++track_noprog_ticks_ > TRACK_NOPROGRESS_TICKS)
                {
                    teleport_to_rest();
                    track_stuck_ticks_ = 0; track_noprog_ticks_ = 0; track_best_dist_ = 1e9;
                    if (predictive_grasp_ and not force_top_down_)
                    {
                        // Reactive fallback: the SIDE approach couldn't converge (real feedback).
                        // Don't just give up — retry this rep grasping from ABOVE.
                        force_top_down_ = true;
                        std::print("[grasp] side approach not converging (best {:.3f} m) → RETRY TOP-DOWN\n",
                                   track_best_dist_);
                        miss_or_give_up("side not converging → top-down");
                    }
                    else
                    {
                        std::print("[recovery] Tracking NOT converging (best {:.3f} m, e_ang {:.1f}°) — "
                                   "pose unreachable/over-constrained → give up\n",
                                   track_best_dist_, e_ang * 57.29578);
                        miss_or_give_up("not converging");
                    }
                    break;
                }

                if (e_pos < REACH_TOLERANCE_M and e_ang < grasp_align_tol_rad_)
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
                            held_track_target_ = track_target;
                            held_z_des_ = g.z_tool_des;
                            held_x_des_ = g.x_tool_des;
                        }
                        break;
                    }
                    // Confidence-scheduled dwell: a novice settles for GRASP_SETTLE_TICKS
                    // cycles to re-verify before committing; a skilled agent trusts the
                    // model and commits almost at once (≥1 cycle). This removes the dead
                    // time spent decelerated-to-near-zero at the standoff.
                    const long settle_need = std::max(1L,
                        std::lround(GRASP_SETTLE_TICKS * (1.0 - c)));
                    if (++grasp_settle_ticks_ >= settle_need)
                    {
                        // Commit: latch the grasp frame so contact can't make
                        // the target chase its own disturbance.
                        latched_grasp_ = g;
                        // Lift along the bottle long axis = true up (cached in the grasp). For a
                        // side grasp this equals tool+Y as before; for a TOP-DOWN grasp tool+Y is
                        // horizontal, so using the stored axis lifts the bottle straight up either way.
                        const Eigen::Vector3d up = g.up_axis.normalized();
                        lift_target_   = g.grasp_pos + up * LIFT_HEIGHT_M;
                        rep_commit_epos_ = e_pos;   // terminal standoff errors → dataset
                        rep_commit_eang_ = e_ang;
                        grasp_phase_   = GraspPhase::Inserting;
                        grasp_settle_ticks_ = 0;
                        insert_ticks_       = 0;
                        tip_reflex_offset_  = 0.0;   // fresh stop-and-rectify correction per attempt
                        std::print("[grasp] standoff settled (e={:.3f} m, {:.1f}°) → Inserting\n",
                                   e_pos, e_ang * 57.29578);
                    }
                }
                else grasp_settle_ticks_ = 0;
                break;
            }
#if 0  // approach-only validation: grasp/place FSM disabled
            case GraspPhase::Inserting:
            {
                // Ease into the bottle body along the latched approach axis.
                const auto& g = latched_grasp_;
                gripper_command_ = 1.0f;  // stay open
                // Skilled inserts faster: trust the model, so the gentle creep into the
                // bottle no longer dominates the cycle (deadband + INSERT_TOUCH_FORCE
                // still arrest it at contact).
                const double insert_vel = INSERT_VEL_MS * rep_perturb_.speed_scale
                                          * (1.0 + insert_conf_gain_ * skill_c());
                const double e_grasp = (ee_position - g.grasp_pos).norm();

                if (tip_reflex_)
                {
                    // ── Stop-and-rectify reflex (tip bumpers) ───────────────────────────────
                    // A distal-tip bumper firing = a finger hit the object on a misaligned
                    // approach (it won't slide into the gripper). HALT forward motion, back off
                    // along −approach, and grow a lateral offset TOWARD the contacting finger so
                    // the object can enter the gap on the retry. The offset accumulates to cancel
                    // the misalignment; commit to Closing only when seated AND no tip in contact.
                    const auto [lc, rc] = tip_contacts();
                    const Eigen::Vector3d lat      = kinematics_->tool_pose(q).rotation.col(0);
                    const Eigen::Vector3d approach  = g.z_tool_des.normalized();
                    const Eigen::Vector3d off       = recenter_sign_ * tip_reflex_offset_ * lat;
                    if (lc or rc)
                    {
                        // Shift toward the side that hit (one finger ⇒ signed; both ⇒ pure back-off).
                        const double dir = (rc and not lc) ? +1.0 : (lc and not rc) ? -1.0 : 0.0;
                        tip_reflex_offset_ = std::clamp(tip_reflex_offset_ + dir * TIP_REFLEX_STEP_M,
                                                        -TIP_REFLEX_MAX_M, TIP_REFLEX_MAX_M);
                        const Eigen::Vector3d target = g.grasp_pos + off - approach * TIP_REFLEX_BACKOFF_M;
                        drive(target, g.z_tool_des, g.x_tool_des, insert_vel);
                        if (tip_log_ and insert_ticks_ % 5 == 0)
                            std::print("[reflex] tip L{} R{} → back off + shift {:+.3f} m\n",
                                       int(lc), int(rc), tip_reflex_offset_);
                    }
                    else
                    {
                        const Eigen::Vector3d target = g.grasp_pos + off;
                        drive(target, g.z_tool_des, g.x_tool_des, insert_vel);
                        if ((ee_position - target).norm() < REACH_TOLERANCE_M)
                        {
                            grasp_phase_       = GraspPhase::Closing;
                            closing_ticks_     = 0;
                            grasp_force_ticks_ = 0;
                            insert_ticks_      = 0;
                            std::print("[grasp] seated (reflex offset {:+.3f} m) → Closing\n", tip_reflex_offset_);
                        }
                    }
                    if (grasp_phase_ == GraspPhase::Inserting and ++insert_ticks_ > INSERT_TIMEOUT_TICKS)
                    {
                        std::print("[grasp] MISS — reflex could not seat in {} cy (offset {:+.3f}) → reopen\n",
                                   INSERT_TIMEOUT_TICKS, tip_reflex_offset_);
                        log_rep_outcome(false, 0.0, 0.0);
                        miss_or_give_up("reflex stuck");
                    }
                    break;
                }

                if (tactile_recenter_)
                {
                    // ── Tactile re-centering (anti-tip) ─────────────────────────────────────
                    // Fingertips straddle the bottle along tool +X (the finger-closing axis,
                    // x_tool_des = z_bot×z_tool_des). If the gripper is off-centre, ONE finger
                    // contacts and PUSHES the bottle sideways → it tips, and the old gate
                    // (f>GRASP_FORCE_THRESH) would commit to Closing on the now-displaced
                    // bottle. Instead: read the L/R tip asymmetry and shift the insert target
                    // laterally toward the harder-pushing finger to centre the bottle between
                    // the pads; slow the approach while in contact; and commit to Closing only
                    // when SEATED and NOT pushing (symmetric/clear). Re-centre uses the LIVE
                    // tool X (yaw is free in the grasp, so the commanded x_tool_des may differ).
                    const auto [fl, fr] = tip_forces();
                    const float contact = fl + fr;
                    const float rel_asym = (contact > 1e-3f) ? (fl - fr) / contact : 0.0f;  // ∈[-1,1]
                    const Eigen::Vector3d lat = kinematics_->tool_pose(q).rotation.col(0);
                    const Eigen::Vector3d target =
                        g.grasp_pos + recenter_sign_ * recenter_gain_ * rel_asym * lat;
                    // Creep while a finger is loaded so centring leads seating (don't ram).
                    const double v_app = (contact > INSERT_TOUCH_FORCE) ? insert_vel * 0.3 : insert_vel;
                    const auto [e_tgt, e_ang] = drive(target, g.z_tool_des, g.x_tool_des, v_app);
                    (void) e_ang; (void) e_tgt;
                    if (tip_log_ and insert_ticks_ % 10 == 0)
                        std::print("[grasp] insert e={:.3f} tipL={:.2f} tipR={:.2f} asym={:+.2f}\n",
                                   e_grasp, fl, fr, rel_asym);
                    // Seated AND not actively pushing the bottle → safe to close.
                    if (e_grasp < REACH_TOLERANCE_M and contact <= INSERT_TOUCH_FORCE)
                    {
                        grasp_phase_       = GraspPhase::Closing;
                        closing_ticks_     = 0;
                        grasp_force_ticks_ = 0;
                        insert_ticks_      = 0;
                        std::print("[grasp] seated & centred (e={:.3f} m, tipL={:.2f} R={:.2f}) → Closing\n",
                                   e_grasp, fl, fr);
                    }
                    else if (++insert_ticks_ > INSERT_TIMEOUT_TICKS)
                    {
                        std::print("[grasp] MISS — could not seat/centre in {} cy (e={:.3f}, asym={:+.2f}) → reopen\n",
                                   INSERT_TIMEOUT_TICKS, e_grasp, rel_asym);
                        log_rep_outcome(false, 0.0, 0.0);
                        miss_or_give_up("insert stuck");
                    }
                    break;
                }

                const auto [e_pos, e_ang] =
                    drive(g.grasp_pos, g.z_tool_des, g.x_tool_des, insert_vel);
                (void) e_ang;
                const float f = gripper_force();
                // Commit to Closing when SEATED, or on contact force BUT ONLY near the grasp —
                // a proximity guard. On the vertical mount the gripper picks up a spurious force
                // ~9 cm short (brushing table/bottle on the angled descent); without this guard it
                // committed there and closed on air. Force only counts within 2·REACH of the grasp.
                if (e_pos < REACH_TOLERANCE_M or (f > GRASP_FORCE_THRESH and e_pos < 2.0 * REACH_TOLERANCE_M))
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
                    sample_place_spot();          // choose place_hover_ NOW so Lifting can blend toward it
                    blend_min_dist_        = 1e9; // fresh closest-approach tracker for the Lift corner
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
                    drive(lift_target_, g.z_tool_des, g.x_tool_des, skilled_speed(0.20),
                          place_hover_);   // look-ahead blend: round the up→lateral corner into PlaceMoving
                (void) e_ang;
                if (via_reached(e_pos))
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
                    // Elbow clearance to the mast at the grasp — the redundancy term's job.
                    // Lets the projector-equivalent vs genuine-NEO A/B show the elbow effect.
                    {
                        const Eigen::Vector3d elb = kinematics_->elbow_position(q);
                        const double mast_clr = std::hypot(elb.x() + 0.56477, elb.y() + 0.056064) - 0.05;
                        std::print("[elbow] clearance to mast = {:.3f} m  pos=({:.3f},{:.3f},{:.3f})\n",
                                   mast_clr, elb.x(), elb.y(), elb.z());
                    }
                    lift_ticks_ = 0;
                    // place_pos_/place_hover_ were already sampled at Closing→Lifting so the
                    // lift could blend toward the hover; just hand over to PlaceMoving here.
                    grasp_phase_ = GraspPhase::PlaceMoving;
                    place_ticks_ = 0;
                    blend_min_dist_ = 1e9;   // fresh closest-approach tracker for the PlaceMoving corner
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
                    drive(place_hover_, place_z_des_, place_x_des_, skilled_speed(0.20),
                          place_pos_, PLACE_ORIENT_GAIN);   // blend into PlaceLowering, relaxed orientation
                (void) e_ang;
                if (via_reached(e_pos) or ++place_ticks_ > PLACE_TIMEOUT_TICKS)
                {
                    grasp_phase_ = GraspPhase::PlaceLowering;
                    place_ticks_ = 0;
                    place_settle_ticks_ = 0;
                    place_bottle_z_prev_ = 1e9;   // fresh "rested on table" tracker
                    std::print("[place] above spot → PlaceLowering\n");
                }
                break;
            }
            case GraspPhase::PlaceLowering:
            {
                gripper_command_ = 0.0f;  // still holding
                // Brisk set-down, skill-scheduled (skilled places faster). RELAXED gripper
                // orientation: near the column the arm can't hit the exact pose, so a strong
                // orientation pull just makes it oscillate; placing needs far less orientation
                // precision than picking (the bottle only has to end vertical on the table).
                const auto [e_pos, e_ang] =
                    drive(place_pos_, place_z_des_, place_x_des_, skilled_speed(0.18),
                          std::nullopt, PLACE_ORIENT_GAIN);
                (void) e_ang;
                const double tilt = std::acos(std::clamp(std::abs(bottle_axis_world_.normalized().z()), 0.0, 1.0));
                // RELEASE when the BOTTLE itself is reasonably placed — its base resting on the
                // table (ground truth: within PLACE_ON_TABLE_M of the table top) + stopped
                // descending + upright — REGARDLESS of whether the EE reached place_pos with the
                // commanded orientation. This stops the "dozens of seconds oscillating to perfect
                // the gripper pose" near the column once the bottle is already down and vertical.
                // e_pos<REACH stays as a fast path when the pose IS cleanly reachable.
                const double dz = std::abs(bottle_pos_world_.z() - place_bottle_z_prev_);
                place_bottle_z_prev_ = bottle_pos_world_.z();
                const bool upright    = tilt < PLACE_UPRIGHT_TOL_RAD;
                const bool bottle_down = (bottle_pos_world_.z() - table_top_z_) < PLACE_ON_TABLE_M and dz < 0.0008;
                const bool ok = (bottle_down or e_pos < REACH_TOLERANCE_M) and upright;
                place_settle_ticks_ = ok ? place_settle_ticks_ + 1 : 0;
                if (place_settle_ticks_ >= PLACE_SETTLE_TICKS or ++place_ticks_ > PLACE_TIMEOUT_TICKS)
                {
                    grasp_phase_ = GraspPhase::PlaceReleasing;
                    place_ticks_ = 0;
                    place_settle_ticks_ = 0;
                    std::print("[place] set down (e={:.3f} m, tilt {:.1f}°) → PlaceReleasing (open)\n",
                               e_pos, tilt * 57.29578);
                }
                break;
            }
            case GraspPhase::PlaceReleasing:
            {
                gripper_command_ = 1.0f;  // open — let the bottle go
                drive(place_pos_, place_z_des_, place_x_des_, 0.05);  // hold still while opening
                // Give the fingers a short FIXED moment to open, then retreat. The bridge
                // does not report a usable gripper aperture (getGripperState().opening
                // reads 0), so the previous opening-threshold confirm silently fell through
                // to the 6 s PLACE_TIMEOUT every cycle — that was the long dead-stand after
                // each place. The probe showed the gripper-axis retreat is tip-free even
                // from a partly-open gripper, so RELEASE_TICKS (~0.5 s) is plenty.
                if (++place_ticks_ >= release_ticks_)
                {
                    // Latch the retreat as a pure back-translation along the gripper's OWN
                    // forward axis (live tool +Z), orientation held — perpendicular to the
                    // upright bottle and precisely on the gripper axis, so no part of the
                    // move sweeps a finger sideways across the bottle.
                    const auto tp = kinematics_->tool_pose(q);
                    retreat_z_des_      = tp.rotation.col(2).normalized();  // gripper forward
                    retreat_x_des_      = tp.rotation.col(0).normalized();
                    retreat_target_pos_ = tp.position - retreat_z_des_ * PLACE_RETREAT_DIST_M;
                    grasp_phase_ = GraspPhase::PlaceRetreating;
                    place_ticks_ = 0;
                    place_settle_ticks_ = 0;
                    std::print("[place] released ({} cy) → PlaceRetreating along gripper axis\n", release_ticks_);
                }
                break;
            }
            case GraspPhase::PlaceRetreating:
            {
                gripper_command_ = 1.0f;  // stay open
                // Pure translation straight back along the latched gripper forward axis,
                // orientation held — gentle (NOT skill-boosted: speed here only risks
                // catching the bottle). Explored speed comes from the retreat probe.
                const double rspeed = retreat_speed_ * (1.0 + retreat_perturb_.dspeed);
                const auto [e_pos, e_ang] =
                    drive(retreat_target_pos_, retreat_z_des_, retreat_x_des_, rspeed);
                (void) e_ang;
                if (e_pos < REACH_TOLERANCE_M or ++place_ticks_ > PLACE_TIMEOUT_TICKS)
                {
                    // Score this retreat by the bottle's tilt now that the fingers are
                    // clear — the free outcome signal the retreat probe learns from.
                    const double tilt_deg = bottle_tilt_rad() * 57.29578;
                    const bool   tipped   = tilt_deg > (FALL_TILT_RAD * 57.29578);
                    log_retreat_outcome(tilt_deg, tipped);
                    std::print("[retreat] done — bottle tilt {:.1f}°{}\n",
                               tilt_deg, tipped ? "  TIPPED" : " (upright)");
                    // Return to rest via the outer Homing path, then auto-restart
                    // the whole pick-and-place (handled in the Homing branch).
                    kinovaarm_proxy->moveJointsWithAngle(nearest_equiv_target(rest_pose_angles_));
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
#endif  // inner: Inserting..PlaceRetreating
        }  // end switch
#endif  // outer: entire FSM disabled for approach-only validation
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

