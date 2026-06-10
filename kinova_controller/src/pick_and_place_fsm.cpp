#include "pick_and_place_fsm.h"

#include <cmath>
#include <chrono>
#include <print>
#include <format>
#include <sstream>
#include <algorithm>

namespace
{
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

    inline double halton(long index, int base)
    {
        double f = 1.0, r = 0.0;
        long i = index + 1;
        while (i > 0) { f /= base; r += f * (i % base); i /= base; }
        return r;
    }

    inline double now_seconds()
    {
        using namespace std::chrono;
        return duration<double>(steady_clock::now().time_since_epoch()).count();
    }
}

// ── Construction: load the manipulation/behaviour config ─────────────────────
PickandPlaceFSM::PickandPlaceFSM(SpecificWorker& w, const ConfigLoader& cfg) : w_(w)
{
    try { tip_log_         = cfg.get<bool>("Controller.tip_log");   } catch (...) {}
    try { round_cycles_    = cfg.get<int>("Controller.round_cycles"); } catch (...) {}
    if (round_cycles_ > 0) std::print("[ui] round_cycles: stop after {} pick-and-place cycles\n", round_cycles_);

    try { probe_enabled_   = cfg.get<bool>  ("Controller.probe_variations"); } catch (...) {}
    try { probe_pos_amp_   = cfg.get<double>("Controller.probe_pos_amp");     } catch (...) {}
    try { probe_azi_amp_   = cfg.get<double>("Controller.probe_azi_amp");     } catch (...) {}
    try { probe_speed_amp_ = cfg.get<double>("Controller.probe_speed_amp");   } catch (...) {}
    try { respawn_each_rep_ = cfg.get<bool>("Controller.respawn_each_rep");    } catch (...) {}
    try { use_qp_ = (cfg.get<std::string>("Controller.solver") == "qp"); } catch (...) {}
    try { qp_redundancy_weight_ = cfg.get<double>("Controller.qp_redundancy_weight"); } catch (...) {}
    try { force_confidence_ = cfg.get<double>("Controller.force_confidence"); } catch (...) {}
    try { blend_radius_ = cfg.get<double>("Controller.blend_radius"); } catch (...) {}
    try { use_preference_field_ = cfg.get<bool>  ("Controller.use_preference_field"); } catch (...) {}
    try { field_prec_pass_      = cfg.get<double>("Controller.field_prec_pass"); } catch (...) {}
    try { field_prec_stop_      = cfg.get<double>("Controller.field_prec_stop"); } catch (...) {}
    try { field_prec_ref_       = cfg.get<double>("Controller.field_prec_ref");  } catch (...) {}
    try { field_overlap_        = cfg.get<double>("Controller.field_overlap");   } catch (...) {}
    if (use_preference_field_)
        std::print("[field] preference-field mode ON (prec pass={:.1f} stop={:.1f} ref={:.1f} overlap={:.3f})\n",
                   field_prec_pass_, field_prec_stop_, field_prec_ref_, field_overlap_);
    try { tactile_recenter_ = cfg.get<bool>  ("Controller.tactile_recenter"); } catch (...) {}
    try { recenter_gain_    = cfg.get<double>("Controller.recenter_gain");    } catch (...) {}
    try { recenter_sign_    = cfg.get<double>("Controller.recenter_sign");    } catch (...) {}
    try { tip_reflex_             = cfg.get<bool>  ("Controller.tip_reflex");              } catch (...) {}
    try { bottle_obstacle_        = cfg.get<bool>  ("Controller.bottle_obstacle");        } catch (...) {}
    try { bottle_obstacle_margin_ = cfg.get<double>("Controller.bottle_obstacle_margin"); } catch (...) {}
    if (force_confidence_ >= 0.0)
        std::print("[experiment] confidence PINNED at {:.2f} (overrides learning/decay)\n", force_confidence_);
    try {
        std::istringstream fp(cfg.get<std::string>("Controller.fixed_pick_xy"));
        double fx, fy;
        if (fp >> fx >> fy) { fixed_pick_xy_ = {fx, fy}; fixed_pick_set_ = true;
            std::print("[experiment] fixed pick spot = ({:.3f}, {:.3f}) world\n", fx, fy); }
    } catch (...) {}
    try {
        std::istringstream fp(cfg.get<std::string>("Controller.fixed_place_xy"));
        double fx, fy;
        if (fp >> fx >> fy) { fixed_place_xy_ = {fx, fy}; fixed_place_set_ = true;
            std::print("[experiment] fixed place spot = ({:.3f}, {:.3f}) world\n", fx, fy); }
    } catch (...) {}
    try { elbow_gain_ = cfg.get<double>("Controller.elbow_gain"); } catch (...) {}
    try {
        std::istringstream et(cfg.get<std::string>("Controller.elbow_target_xy"));
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
        std::print("[spawn] per-rep bottle respawn ON\n");
    if (probe_enabled_)
        std::print("[probe] structured grasp perturbations ON  (pos ±{:.3f} m, azi ±{:.3f} rad, speed ±{:.0f}%)\n",
                   probe_pos_amp_, probe_azi_amp_, probe_speed_amp_ * 100.0);
    try {
        const auto path = cfg.get<std::string>("Controller.dataset_path");
        if (not path.empty()) {
            dataset_.open(path, std::ios::out | std::ios::app);
            if (dataset_.is_open()) {
                dataset_open_ = true;
                if (dataset_.tellp() == std::streampos(0))
                    dataset_ << "probe_idx,rep,success,dx_perp,dz_axis,dazi,speed_scale,"
                                "gx,gy,gz,bx,by,bz,axz,commit_epos,commit_eang,track_ticks,"
                                "bottle_rise,xy_gap\n";
                std::print("[probe] per-rep dataset → {}\n", path);
            }
        }
    } catch (...) {}

    try { learn_pick_place_      = cfg.get<bool>  ("Controller.learn_pick_place");        } catch (...) {}
    try { precision_reweighting_ = cfg.get<bool>  ("Controller.precision_reweighting"); } catch (...) {}
    try { perception_noise_std_  = cfg.get<double>("Controller.perception_noise_std");   } catch (...) {}
    try { conf_gain_             = cfg.get<double>("Controller.conf_gain");               } catch (...) {}
    try { conf_decay_            = cfg.get<double>("Controller.conf_decay");              } catch (...) {}
    try { pi_s_                  = cfg.get<double>("Controller.sensory_precision");       } catch (...) {}
    try { evidence_unit_         = cfg.get<double>("Controller.evidence_unit");           } catch (...) {}
    try { skilled_sample_period_ = cfg.get<int>   ("Controller.skilled_sample_period");  } catch (...) {}
    try { speed_conf_gain_       = cfg.get<double>("Controller.speed_conf_gain");         } catch (...) {}
    try { confidence_path_       = cfg.get<std::string>("Controller.confidence_path");    } catch (...) {}
    try { standoff_collapse_     = cfg.get<double>("Controller.standoff_collapse");        } catch (...) {}
    try { insert_conf_gain_      = cfg.get<double>("Controller.insert_conf_gain");         } catch (...) {}
    standoff_collapse_ = std::clamp(standoff_collapse_, 0.0, 1.0);
    try { retreat_speed_         = cfg.get<double>("Controller.retreat_speed");            } catch (...) {}
    try { gripper_open_conf_     = cfg.get<double>("Controller.gripper_open_conf");        } catch (...) {}
    try { release_ticks_         = cfg.get<int>   ("Controller.release_ticks");            } catch (...) {}
    try { grasp_align_tol_rad_   = cfg.get<double>("Controller.grasp_align_tol_deg") * M_PI / 180.0; } catch (...) {}
    try { predictive_place_      = cfg.get<bool>  ("Controller.predictive_place");          } catch (...) {}
    if (predictive_place_) std::print("[predict] predictive place-spot selection ON\n");
    try { predictive_grasp_      = cfg.get<bool>  ("Controller.predictive_grasp");          } catch (...) {}
    if (predictive_grasp_) std::print("[predict] predictive grasp selection ON\n");
    try { precompute_reach_map_  = cfg.get<bool>       ("Controller.precompute_reach_map"); } catch (...) {}
    try { reach_map_path_        = cfg.get<std::string>("Controller.reach_map_path");        } catch (...) {}
    try { probe_retreat_         = cfg.get<bool>  ("Controller.probe_retreat");            } catch (...) {}
    try { probe_rspeed_amp_      = cfg.get<double>("Controller.probe_rspeed_amp");         } catch (...) {}
    try { probe_open_amp_        = cfg.get<double>("Controller.probe_open_amp");           } catch (...) {}
    try {
        const auto rpath = cfg.get<std::string>("Controller.retreat_log_path");
        if (not rpath.empty()) {
            retreat_log_.open(rpath, std::ios::out | std::ios::trunc);
            if ((retreat_log_open_ = retreat_log_.is_open())) {
                retreat_log_ << "probe_idx,retreat_speed,open_thresh,place_x,place_y,post_tilt_deg,tipped\n";
                std::print("[retreat] outcome dataset → {}\n", rpath);
            }
        }
    } catch (...) {}
    load_confidence();
    try {
        const auto mpath = cfg.get<std::string>("Controller.metrics_path");
        if (not mpath.empty()) {
            metrics_.open(mpath, std::ios::out | std::ios::trunc);
            if (metrics_.is_open()) {
                metrics_open_ = true;
                metrics_ << "rep,success,confidence,cycle_time_s,observations\n";
                std::print("[skill] per-rep metrics → {}\n", mpath);
            }
        }
    } catch (...) {}
}

// ── Lifecycle hooks ──────────────────────────────────────────────────────────
void PickandPlaceFSM::step(const std::array<double, Kinematics::N_ARM_JOINTS>& q,
                           const Eigen::Vector3d& ee_position)
{
    if (force_confidence_ >= 0.0) confidence_ = force_confidence_;
    if (grasp_phase_ == GraspPhase::Tracking) run_tracking(q, ee_position);
    else                                      run_grasp_phases(q, ee_position);
}

void PickandPlaceFSM::start()
{
    grasp_phase_            = GraspPhase::Tracking;
    grasp_settle_ticks_     = 0;
    approach_hold_logged_   = false;
    w_.gripper_command_     = 1.0f;
    pick_place_cycles_done_ = 0;
    begin_rep_probe();
}

void PickandPlaceFSM::reset()
{
    grasp_phase_          = GraspPhase::Tracking;
    grasp_settle_ticks_   = 0;
    returning_for_cycle_  = false;
    approach_hold_logged_ = false;
    w_.gripper_command_   = 1.0f;
}

SpecificWorker::Phase PickandPlaceFSM::on_rest_reached()
{
    if (returning_for_cycle_ and w_.run_requested_
        and (round_cycles_ <= 0 or pick_place_cycles_done_ < round_cycles_))
    {
        returning_for_cycle_ = false;
        grasp_phase_         = GraspPhase::Tracking;
        grasp_settle_ticks_  = 0;
        w_.gripper_command_  = 1.0f;
        begin_rep_probe();
        std::print("[cycle] rest reached → starting pick-and-place {}\n", pick_place_cycles_done_ + 1);
        return SpecificWorker::Phase::ActiveEFE;
    }
    if (returning_for_cycle_ and round_cycles_ > 0 and pick_place_cycles_done_ >= round_cycles_)
    {
        returning_for_cycle_ = false;
        w_.run_requested_    = false;
        std::print("[round] {}/{} pick-and-place cycles complete — round done, parked at rest.\n",
                   pick_place_cycles_done_, round_cycles_);
        return SpecificWorker::Phase::WaitingForStart;
    }
    returning_for_cycle_ = false;
    std::print("[homing] Rest pose reached — waiting for Start button.\n");
    return SpecificWorker::Phase::WaitingForStart;
}

void PickandPlaceFSM::maybe_compute_reach_map()
{
    if (precompute_reach_map_ and not reach_map_done_ and w_.base_tf_set_)
    {
        compute_reach_map();
        reach_map_done_ = true;
    }
}

// ── Phase 1: validated QP approach-to-bottle (camera-up, deadband hold) ───────
void PickandPlaceFSM::run_tracking(const std::array<double, Kinematics::N_ARM_JOINTS>& q,
                                   const Eigen::Vector3d& ee_position)
{
    w_.gripper_command_ = 1.0f;

    // Baseline-hold only: place the bottle at the fixed-pick spot once. In learn mode the
    // per-rep respawn is done by begin_rep_probe (fresh bottle at the pick spot each rep).
    if (not learn_pick_place_ and fixed_pick_set_ and not approach_respawn_done_ and w_.scene_world_valid_)
    {
        w_.respawn_bottle(fixed_pick_xy_.x(), fixed_pick_xy_.y());
        approach_respawn_done_ = true;
        return;
    }

    auto g_opt = compute_side_grasp_target();
    if (not g_opt.has_value())
    {
        RoboCompKinovaArm::TJointSpeeds stop;
        stop.jointSpeeds.assign(Kinematics::N_ARM_JOINTS, 0.0f);
        w_.kinovaarm_proxy->moveJointsWithSpeed(stop);
        if (ctrl_cycle_ % 25 == 0)
            std::print("[ctrl] cy={} NO TARGET — base_tf={} scene={} bottle=({:.3f},{:.3f},{:.3f})\n",
                       ctrl_cycle_, w_.base_tf_set_, w_.scene_world_valid_,
                       w_.bottle_pos_world_.x(), w_.bottle_pos_world_.y(), w_.bottle_pos_world_.z());
        ++ctrl_cycle_;
        return;
    }
    const auto& g = g_opt.value();

    // ── Learn mode: commit the grasp; skill c reshapes the approach (geometry + dwell) ──
    if (learn_pick_place_)
    {
        ++rep_track_ticks_;
        const double c = skill_c();

        // Jam / no-progress watchdogs: a wedged approach can't loop forever.
        if (++track_stuck_ticks_ > TRACK_TIMEOUT_TICKS)
        {
            w_.teleport_to_rest();
            track_stuck_ticks_ = 0;
            miss_or_give_up("track stalled — possible jam");
            return;
        }
        // standoff_collapse: skilled reaches more directly (slide the waypoint toward the grasp).
        const Eigen::Vector3d track_target =
            g.grasp_pos + (g.stand_off_pos - g.grasp_pos) * (1.0 - standoff_collapse_ * c);
        const auto [ep, ea] = efe_drive(q, ee_position, track_target,
                                        g.z_tool_des, g.x_tool_des, skilled_speed(0.25));
        if (ep < track_best_dist_ - 0.004) { track_best_dist_ = ep; track_noprog_ticks_ = 0; }
        else if (ep > 3.0 * REACH_TOLERANCE_M and ++track_noprog_ticks_ > TRACK_NOPROGRESS_TICKS)
        {
            w_.teleport_to_rest();
            track_stuck_ticks_ = 0; track_noprog_ticks_ = 0; track_best_dist_ = 1e9;
            miss_or_give_up("not converging");
            return;
        }
        if (ep < REACH_TOLERANCE_M and ea < grasp_align_tol_rad_)
        {
            // Skilled commits sooner (shorter settle dwell); novice re-verifies longer.
            const long settle_need = std::max(1L, std::lround(GRASP_SETTLE_TICKS * (1.0 - c)));
            if (++grasp_settle_ticks_ >= settle_need)
            {
                latched_grasp_   = g;
                lift_target_     = g.grasp_pos + g.up_axis.normalized() * LIFT_HEIGHT_M;
                rep_commit_epos_ = ep;
                rep_commit_eang_ = ea;
                grasp_phase_     = GraspPhase::Inserting;
                grasp_settle_ticks_ = 0; insert_ticks_ = 0; tip_reflex_offset_ = 0.0;
                std::print("[grasp] standoff settled (e={:.3f} m, {:.1f}°, c={:.2f}) → Inserting\n",
                           ep, ea * 57.29578, c);
            }
        }
        else grasp_settle_ticks_ = 0;
        ++ctrl_cycle_;
        return;
    }

    // ── Baseline: validated QP approach-to-bottle, then hold (no commit) ──
    const auto [e_pos, e_ang] = efe_drive(q, ee_position,
                                          g.stand_off_pos, g.z_tool_des, g.x_tool_des, 0.25);

    if (e_pos < REACH_TOLERANCE_M and not approach_hold_logged_)
    {
        approach_hold_logged_ = true;
        std::print("[ctrl] cy={:4d} *** AT STANDOFF *** e={:.4f}m {:.1f}° — settling to still hold\n",
                   ctrl_cycle_, e_pos, e_ang * 57.29578);
    }

    {
        const auto J6   = w_.kinematics_->arm_jacobian_full(q);
        const double mu = std::sqrt(std::max(0.0, (J6 * J6.transpose()).determinant()));
        const auto sk   = w_.kinematics_->arm_skeleton_points(q);
        const Eigen::Vector3d col_lo(-0.56477, -0.056064, -0.10);
        const Eigen::Vector3d col_hi(-0.56477, -0.056064,  1.30);
        double col_min = 1e9;
        for (int k = 2; k + 1 < (int)sk.size(); ++k)
            col_min = std::min(col_min, segment_segment_distance(sk[k], sk[k+1], col_lo, col_hi) - 0.05);
        const double cam_up = w_.kinematics_->tool_pose(q).rotation.col(1).z();

        const bool converged = e_pos < REACH_TOLERANCE_M and e_ang < grasp_align_tol_rad_;
        std::print("[ctrl] cy={:4d}  bot({:.3f},{:.3f},{:.3f})  tgt({:.3f},{:.3f},{:.3f})  "
                   "ee({:.3f},{:.3f},{:.3f})  e={:.4f}m {:.1f}°  mu={:.4f}  col={:.3f}m  camUp={:+.2f}{}\n",
                   ctrl_cycle_,
                   w_.bottle_pos_world_.x(), w_.bottle_pos_world_.y(), w_.bottle_pos_world_.z(),
                   g.stand_off_pos.x(), g.stand_off_pos.y(), g.stand_off_pos.z(),
                   ee_position.x(), ee_position.y(), ee_position.z(),
                   e_pos, e_ang * 57.29578, mu, col_min, cam_up,
                   converged ? "  *** CONVERGED ***" : "");
        ++ctrl_cycle_;
    }
}

double PickandPlaceFSM::bottle_tilt_rad() const
{ return std::acos(std::clamp(std::abs(w_.bottle_axis_world_.normalized().z()), 0.0, 1.0)); }

// ── EFE/QP controller primitives ─────────────────────────────────────────────
EFEParams PickandPlaceFSM::build_efe_params(const Eigen::Vector3d& z_des,
                                            const Eigen::Vector3d& x_des,
                                            double v_app) const
{
    EFEParams p;
    p.desired_approach  = z_des;
    p.desired_secondary = x_des;
    p.align_tool_y      = false;
    p.desired_tool_y    = z_des.cross(x_des).normalized();
    // Full-frame orientation: pin tool+Z to the bottle AND tool+Y up (camera on the
    // upper side). The QP rotation slack lets the solver leave any unreachable residual
    // as cheap slack instead of contorting into a singularity.
    p.gain_secondary    = 1.0;
    p.C_pos             = Eigen::Vector3d::Ones();
    p.dls_lambda        = 0.05;
    p.use_qp            = use_qp_;
    p.obs_damper_xi     = 0.5;
    // Rotation slack (QP): orientation task-slack weighted well below position so the
    // solver targets the rotation but does not sacrifice position to perfect it.
    p.orient_slack      = 0.05;
    p.redundancy_weight = qp_redundancy_weight_;
    p.v_approach        = v_app;
    p.a_approach        = 0.60;
    p.omega_max         = 2.0;
    // Hold tolerance width: in Tracking the arm settles dead-still at the standoff within
    // ~2 cm / ~9° instead of hunting. The grasp/place phases keep the TIGHT EFEParams
    // defaults (0.005 / 0.03) — insertion and set-down need to reach their target precisely.
    if (grasp_phase_ == GraspPhase::Tracking)
    {
        p.arrive_deadband = 0.02;
        p.orient_deadband = 0.16;
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
    p.table_z           = w_.table_top_z_;
    p.table_safe        = 0.06;
    // Bottle-as-obstacle ONLY while withdrawing from the JUST-PLACED bottle
    // (PlaceRetreating): the gripper has released and is backing off to rest, so the
    // bottle standing on the table must not be knocked over. bottle_pos_world_ is the
    // live pose, which at this point is the placed bottle. It is OFF during the pick
    // approach (Tracking) — there the gripper must reach the standoff and grasp — and
    // off during the grasp/lift/place legs that move toward the bottle.
    if (use_qp_ and bottle_obstacle_ and grasp_phase_ == GraspPhase::PlaceRetreating)
    {
        p.gain_bottle   = 1.0;
        p.bottle_xy     = w_.bottle_pos_world_.head<2>();
        p.bottle_radius = w_.bottle_radius_m_;
        p.bottle_z_lo   = w_.bottle_pos_world_.z();
        p.bottle_z_hi   = w_.bottle_pos_world_.z() + w_.bottle_height_m_;
        p.bottle_margin = bottle_obstacle_margin_;
    }
    return p;
}

std::pair<double,double> PickandPlaceFSM::efe_drive(
    const std::array<double, Kinematics::N_ARM_JOINTS>& q,
    const Eigen::Vector3d& ee_position,
    const Eigen::Vector3d& target,
    const Eigen::Vector3d& z_des,
    const Eigen::Vector3d& x_des,
    double v_app,
    std::optional<Eigen::Vector3d> blend_next,
    double orient_gain)
{
    w_.reach_target_ = target;
    EFEParams params = build_efe_params(z_des, x_des, v_app);
    params.gain_orient  = orient_gain;
    params.blend_next   = blend_next;
    params.blend_radius = blend_next.has_value() ? skill_c() * blend_radius_ : 0.0;
    if (use_preference_field_ and blend_next.has_value())
    {
        params.use_field    = true;
        params.prec_current = field_prec_stop_ + skill_c() * (field_prec_pass_ - field_prec_stop_);
        params.prec_next    = field_prec_stop_;
        params.prec_ref     = field_prec_ref_;
        params.field_overlap = field_overlap_;
    }
    auto q_dot = efe_gradient_step(*w_.kinematics_, q, w_.reach_target_, params);
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
    w_.kinovaarm_proxy->moveJointsWithSpeed(cmd);
    last_q_dot_cmd_ = q_dot;

    const double e_pos = (ee_position - target).norm();

    double e_ang = 3.1416;
    if (params.align_tool_y)
    {
        const auto tool = w_.kinematics_->tool_pose(q);
        e_ang = std::acos(std::clamp(
            tool.rotation.col(1).dot(params.desired_tool_y.normalized()), -1.0, 1.0));
    }
    else
    {
        const Eigen::Vector3d zc = z_des.normalized();
        const auto tool = w_.kinematics_->tool_pose(q);
        if (params.gain_secondary <= 0.0)
            e_ang = std::acos(std::clamp(tool.rotation.col(2).dot(zc), -1.0, 1.0));
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
        const double dt   = std::max(1, w_.getPeriod("Compute")) / 1000.0;
        const double vcmd = std::min(params.v_approach,
            std::sqrt(2.0 * params.a_approach * std::max(0.0, e_pos - params.arrive_deadband)));
        const double vmeas = tip_log_prev_pos_.has_value()
            ? (ee_position - tip_log_prev_pos_.value()).norm() / dt : 0.0;
        const Eigen::Vector3d elb = w_.kinematics_->elbow_position(q);
        const auto sk = w_.kinematics_->arm_skeleton_points(q);
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

// ── Grasp-target perception ──────────────────────────────────────────────────
std::optional<PickandPlaceFSM::SideGraspTarget> PickandPlaceFSM::compute_side_grasp_target()
{
    if (not w_.base_tf_set_) return std::nullopt;
    if (not w_.scene_world_valid_) return std::nullopt;
    const Eigen::Vector3d bottle_pos = w_.bottle_pos_world_;
    const Eigen::Vector3d z_bot      = w_.bottle_axis_world_.normalized();
    const Eigen::Vector3d base_pos   = w_.arm_base_world_.translation();

    double bottle_height_m = 0.2;
    if (auto bottle_node = w_.G->get_node("bottle"); bottle_node.has_value())
        if (auto h = w_.G->get_attrib_by_name<height_m_att>(bottle_node.value()); h.has_value())
            bottle_height_m = h.value();
    const Eigen::Vector3d body_centre =
        bottle_pos + z_bot * (bottle_height_m * BOTTLE_GRASP_HEIGHT_FRAC);

    const Eigen::Vector2d col_xy(-0.56477, -0.056064);
    Eigen::Vector3d u = bottle_pos - Eigen::Vector3d(col_xy.x(), col_xy.y(), bottle_pos.z());
    u -= u.dot(z_bot) * z_bot;
    if (u.norm() < 1e-4) return std::nullopt;
    u.normalize();
    const Eigen::Vector3d perp = z_bot.cross(u).normalized();
    const auto standoff_y = [&](const Eigen::Vector3d& zt)
    { return (body_centre - zt * APPROACH_STANDOFF_M).y(); };
    Eigen::Vector3d z_tool_des = (standoff_y(perp) < standoff_y(-perp)) ? perp : -perp;
    Eigen::Vector3d grasp_centre = body_centre;
    bool top_down = false;

    if (force_top_down_)
    {
        grasp_centre = bottle_pos + z_bot * (bottle_height_m * BOTTLE_TOP_GRASP_FRAC);
        z_tool_des   = -z_bot;
        top_down     = true;
    }
    else if (predictive_grasp_)
    {
        const float mu = reach_lookup(body_centre.x(), body_centre.y());
        if (mu <= 0.0f)
        {
            grasp_centre = bottle_pos + z_bot * (bottle_height_m * BOTTLE_TOP_GRASP_FRAC);
            z_tool_des   = -z_bot;
            top_down     = true;
            std::print("[grasp] map predicts side unusable at bottle cell (μ={:.3f}) → TOP-DOWN before attempt\n", mu);
        }
    }

    if (std::abs(rep_perturb_.dazi) > 1e-9 and not top_down)
        z_tool_des = (Eigen::AngleAxisd(rep_perturb_.dazi, z_bot) * z_tool_des).normalized();
    Eigen::Vector3d x_tool_des = z_bot.cross(z_tool_des);
    x_tool_des = (x_tool_des.norm() > 1e-3) ? x_tool_des.normalized() : perp;
    const Eigen::Vector3d grasp_pt =
        grasp_centre + x_tool_des * rep_perturb_.dx_perp + z_bot * rep_perturb_.dz_axis;

    SideGraspTarget out;
    out.z_tool_des    = z_tool_des;
    out.x_tool_des    = x_tool_des;
    out.up_axis       = z_bot;
    out.top_down      = top_down;
    out.grasp_pos     = grasp_pt;
    out.stand_off_pos = grasp_pt - z_tool_des * APPROACH_STANDOFF_M;

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

PickandPlaceFSM::ReachScore
PickandPlaceFSM::predict_reach(const Eigen::Vector3d& pos,
                               const Eigen::Vector3d& z_des, const Eigen::Vector3d& x_des,
                               const std::array<double, Kinematics::N_ARM_JOINTS>& seed)
{
    const Eigen::Vector3d zc = z_des.normalized();
    Eigen::Vector3d xc = x_des - x_des.dot(zc) * zc;
    xc = (xc.norm() > 1e-6) ? xc.normalized() : zc.unitOrthogonal();
    Eigen::Matrix3d R_des; R_des.col(0) = xc; R_des.col(1) = zc.cross(xc); R_des.col(2) = zc;

    std::array<double, Kinematics::N_ARM_JOINTS> q = seed;
    const auto lims = w_.kinematics_->arm_joint_position_limits();
    double pe = 1e9, oe = 1e9;
    for (int it = 0; it < 40; ++it)
    {
        const auto tp = w_.kinematics_->tool_pose(q);
        const Eigen::Vector3d ep = pos - tp.position;
        const Eigen::AngleAxisd aa(R_des * tp.rotation.transpose());
        const Eigen::Vector3d eo = aa.angle() * aa.axis();
        pe = ep.norm(); oe = eo.norm();
        if (pe < 0.005 and oe < 0.05) break;
        Eigen::Matrix<double, 6, 1> e; e << ep, eo;
        const Eigen::Matrix<double, 6, Kinematics::N_ARM_JOINTS> J = w_.kinematics_->arm_jacobian_full(q);
        const Eigen::Matrix<double, 6, 6> A =
            J * J.transpose() + 0.01 * Eigen::Matrix<double, 6, 6>::Identity();
        Eigen::Matrix<double, Kinematics::N_ARM_JOINTS, 1> dq = J.transpose() * A.ldlt().solve(e);
        const double n = dq.norm();
        if (n > 0.4) dq *= 0.4 / n;
        for (int j = 0; j < Kinematics::N_ARM_JOINTS; ++j)
        {
            q[j] += dq[j];
            if (lims[j].first < lims[j].second)
                q[j] = std::clamp(q[j], lims[j].first, lims[j].second);
        }
    }
    const Eigen::Matrix<double, 6, Kinematics::N_ARM_JOINTS> Jf = w_.kinematics_->arm_jacobian_full(q);
    const double manip = std::sqrt(std::max(0.0, (Jf * Jf.transpose()).determinant()));
    double col = 1e9, tab = 1e9;
    for (int j = 2; j <= 6; ++j)
    {
        const Eigen::Vector3d pj = w_.kinematics_->joint_position(q, j);
        col = std::min(col, std::hypot(pj.x() + 0.56477, pj.y() + 0.056064) - 0.05);
        tab = std::min(tab, pj.z() - w_.table_top_z_);
    }
    return { (pe < 0.02 and oe < 0.5), manip, col, tab };
}

float PickandPlaceFSM::reach_lookup(double x, double y) const
{
    if (rm_mu_.empty()) return 1.0f;
    const int ix = static_cast<int>(std::lround((x - rm_x0_) / rm_res_));
    const int iy = static_cast<int>(std::lround((y - rm_y0_) / rm_res_));
    if (ix < 0 or ix >= rm_nx_ or iy < 0 or iy >= rm_ny_) return -1.0f;
    return rm_mu_[static_cast<size_t>(ix) * rm_ny_ + iy];
}

void PickandPlaceFSM::compute_reach_map()
{
    const double t0 = now_seconds();
    std::ofstream f(reach_map_path_, std::ios::out | std::ios::trunc);
    if (not f.is_open()) { std::print("[reachmap] cannot open {}\n", reach_map_path_); return; }
    f << "x,y,reachable,manip,col_clear\n";
    const Eigen::Vector3d base = w_.arm_base_world_.translation();
    const Eigen::Vector3d up(0.0, 0.0, 1.0);
    const double gz = w_.table_top_z_ + 0.10;
    std::vector<std::array<double, Kinematics::N_ARM_JOINTS>> seeds = {w_.rest_pose_angles_, w_.cur_q_};
    auto sp = w_.rest_pose_angles_, sm = w_.rest_pose_angles_;
    for (int j = 0; j < Kinematics::N_ARM_JOINTS; ++j) { sp[j] += 0.5; sm[j] -= 0.5; }
    seeds.push_back(sp); seeds.push_back(sm);

    rm_x0_ = -0.40; rm_y0_ = -0.90; rm_res_ = 0.05;
    rm_nx_ = int(std::lround((0.40 - rm_x0_) / rm_res_)) + 1;
    rm_ny_ = int(std::lround((0.90 - rm_y0_) / rm_res_)) + 1;
    rm_mu_.assign(static_cast<size_t>(rm_nx_) * rm_ny_, -1.0f);
    int ncells = 0, nreach = 0;
    for (int ix = 0; ix < rm_nx_; ++ix)
        for (int iy = 0; iy < rm_ny_; ++iy)
        {
            const double x = rm_x0_ + ix * rm_res_, y = rm_y0_ + iy * rm_res_;
            const Eigen::Vector3d c(x, y, gz);
            Eigen::Vector3d rad = c - base; rad.z() = 0.0;
            rad = (rad.norm() > 1e-6) ? rad.normalized() : Eigen::Vector3d(1, 0, 0);
            Eigen::Vector3d uu(x + 0.56477, y + 0.056064, 0.0);
            uu = (uu.norm() > 1e-6) ? uu.normalized() : rad;
            const Eigen::Vector3d perp = up.cross(uu).normalized();
            const std::array<Eigen::Vector3d, 3> zts{rad, perp, -perp};
            bool reach = false; double best_mu = 0.0, best_col = 0.0;
            for (const auto& zt : zts)
            {
                const Eigen::Vector3d xt = up.cross(zt).normalized();
                for (const auto& seed : seeds)
                {
                    const ReachScore s = predict_reach(c, zt, xt, seed);
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

void PickandPlaceFSM::load_confidence()
{
    // Persist the accumulated MODEL precision Π_m (the learned quantity); c is derived.
    if (not confidence_path_.empty())
    {
        std::ifstream f(confidence_path_);
        double v;
        if (f >> v and v >= 0.0) pi_m_ = v;
    }
    confidence_ = pi_m_ / (pi_m_ + pi_s_);
}

void PickandPlaceFSM::save_confidence()
{
    if (confidence_path_.empty()) return;
    std::ofstream f(confidence_path_, std::ios::out | std::ios::trunc);
    if (f) f << pi_m_ << '\n';
}

// The natural fine-tuning: model precision accumulates from confirmed outcomes, and
// confidence c = Π_m/(Π_m+Π_s) is its normalised form. No reward, no schedule — c rises
// because the act of doing keeps confirming the model, and deflates on surprise.
void PickandPlaceFSM::update_confidence_from_outcome(bool success, double pe_norm, double rise_frac)
{
    if (not precision_reweighting_) return;
    if (success)
    {
        // Evidence quality ∈ (0,1]: 1 when the outcome matched the model's prediction
        // (clean standoff convergence + full lift), → 0 as it barely confirmed.
        const double quality = std::exp(-0.5 * pe_norm * pe_norm) * std::clamp(rise_frac, 0.0, 1.0);
        pi_m_ += evidence_unit_ * quality;
    }
    else
        pi_m_ *= conf_decay_;   // surprise → lose accumulated precision, re-engage feedback
    confidence_ = pi_m_ / (pi_m_ + pi_s_);
    save_confidence();
    std::print("[skill] outcome {} → Π_m={:.3f}  c={:.3f}\n",
               success ? "CONFIRMED" : "SURPRISE", pi_m_, confidence_);
}

void PickandPlaceFSM::log_rep_outcome(bool success, double rise, double xy_gap)
{
    if (not dataset_open_) return;
    const auto& g = latched_grasp_;
    dataset_ << (probe_index_ - 1) << ',' << pick_place_cycles_done_ << ','
             << (success ? 1 : 0) << ','
             << rep_perturb_.dx_perp << ',' << rep_perturb_.dz_axis << ','
             << rep_perturb_.dazi    << ',' << rep_perturb_.speed_scale << ','
             << g.grasp_pos.x() << ',' << g.grasp_pos.y() << ',' << g.grasp_pos.z() << ','
             << w_.bottle_pos_world_.x() << ',' << w_.bottle_pos_world_.y() << ',' << w_.bottle_pos_world_.z() << ','
             << w_.bottle_axis_world_.z() << ','
             << rep_commit_epos_ << ',' << rep_commit_eang_ << ',' << rep_track_ticks_ << ','
             << rise << ',' << xy_gap << '\n';
    dataset_.flush();
}

void PickandPlaceFSM::log_retreat_outcome(double tilt_deg, bool tipped)
{
    if (not retreat_log_open_) return;
    retreat_log_ << (probe_index_ - 1) << ','
                 << (retreat_speed_ * (1.0 + retreat_perturb_.dspeed)) << ','
                 << (gripper_open_conf_ + retreat_perturb_.dopen) << ','
                 << place_world_xy_.x() << ',' << place_world_xy_.y() << ','
                 << tilt_deg << ',' << (tipped ? 1 : 0) << '\n';
    retreat_log_.flush();
}

void PickandPlaceFSM::sample_place_spot()
{
    const auto& g = latched_grasp_;
    const Eigen::Vector3d up(0.0, 0.0, 1.0);
    constexpr double a1 = 0.7548776662466927, a2 = 0.5698402909980532;
    const Eigen::Vector3d base_xyz = w_.arm_base_world_.translation();
    Eigen::Vector3d p = g.grasp_pos;
    bool have_valid = false;
    double best = -1e18;
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
        if (not predictive_place_) { p = c; break; }
        if (not have_valid) { p = c; have_valid = true; }
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
    place_world_xy_ = p;
    place_hover_    = p + up * LIFT_HEIGHT_M;
    Eigen::Vector3d radial = place_pos_ - w_.arm_base_world_.translation();
    radial.z() = 0.0;
    if (g.top_down)
    {
        place_z_des_ = -up;
        place_x_des_ = (radial.norm() > 1e-6) ? radial.normalized() : g.x_tool_des;
    }
    else
    {
        place_z_des_ = (radial.norm() > 1e-6) ? radial.normalized() : g.z_tool_des;
        place_x_des_ = up.cross(place_z_des_).normalized();
    }
}

void PickandPlaceFSM::begin_rep_probe()
{
    rep_track_ticks_ = 0;
    rep_attempts_    = 0;
    track_stuck_ticks_ = 0;
    track_noprog_ticks_ = 0; track_best_dist_ = 1e9;
    force_top_down_ = false;
    belief_valid_     = false;
    cycles_since_obs_ = 1 << 20;
    obs_count_rep_    = 0;
    rep_t0_           = now_seconds();

    if (respawn_each_rep_ and fixed_pick_set_)
        w_.respawn_bottle(fixed_pick_xy_.x(), fixed_pick_xy_.y());
    else if (respawn_each_rep_)
    {
        constexpr double a1 = 0.7548776662466927, a2 = 0.5698402909980532;
        const Eigen::Vector3d base_xyz = w_.arm_base_world_.translation();
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
        w_.respawn_bottle(sx, sy);
    }

    if (probe_enabled_)
    {
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
        retreat_perturb_.dspeed = (halton(probe_index_, 11) - 0.5) * 2.0 * probe_rspeed_amp_;
        retreat_perturb_.dopen  = (halton(probe_index_, 13) - 0.5) * 2.0 * probe_open_amp_;
        std::print("[probe] rep {} retreat: speed×{:.2f} open_thresh{:+.2f}\n",
                   probe_index_, 1.0 + retreat_perturb_.dspeed, retreat_perturb_.dopen);
    }
    else retreat_perturb_ = RetreatPerturbation{};

    ++probe_index_;
}

// ── Grasp-FSM helpers ────────────────────────────────────────────────────────
void PickandPlaceFSM::miss_or_give_up(const std::string& reason)
{
    std::print("[grasp] MISS — {}; reopening gripper, returning to rest\n", reason);
    update_confidence_from_outcome(false, 0, 0);   // every miss is a surprise → deflate Π_m once
    w_.gripper_command_ = 1.0f;
    RoboCompKinovaArm::TJointAngles rest;
    rest.jointAngles.assign(w_.rest_pose_angles_.begin(), w_.rest_pose_angles_.end());
    try { w_.kinovaarm_proxy->moveJointsWithAngle(rest); }
    catch (const Ice::Exception& e)
    { std::print(stderr, "[grasp] moveJointsWithAngle failed: {}\n", e.what()); }

    ++rep_attempts_;
    if (rep_attempts_ >= MAX_REP_ATTEMPTS)
    {
        log_rep_outcome(false, 0.0, 0.0);
        if (metrics_open_)
            metrics_ << (probe_index_ - 1) << ",0," << confidence_ << ','
                     << (now_seconds() - rep_t0_) << ',' << obs_count_rep_ << '\n', metrics_.flush();
        w_.run_requested_ = false;
        std::print("[probe] rep {} failed after {} attempts — moving on\n", probe_index_ - 1, rep_attempts_);
    }
    else
    {
        grasp_phase_           = GraspPhase::Tracking;
        grasp_settle_ticks_    = 0;
        insert_ticks_          = 0;
        closing_ticks_         = 0;
        lift_ticks_            = 0;
        place_ticks_           = 0;
        track_stuck_ticks_     = 0;
        track_noprog_ticks_    = 0;
        track_best_dist_       = 1e9;
        retract_ticks_         = 0;
        w_.phase_              = SpecificWorker::Phase::Homing;
        returning_for_cycle_   = true;
        w_.homing_settled_ticks_ = 0;
        w_.homing_elapsed_ticks_ = 0;
        std::print("[grasp] attempt {}/{} — returning to rest to retry\n", rep_attempts_, MAX_REP_ATTEMPTS);
    }
}

std::pair<bool,bool> PickandPlaceFSM::tip_contacts() const
{
    try { const auto gs = w_.kinovaarm_proxy->getGripperState(); return {gs.ltipcontact, gs.rtipcontact}; }
    catch (const Ice::Exception&) { return {false, false}; }
}

std::pair<float,float> PickandPlaceFSM::tip_forces() const
{
    try { const auto gs = w_.kinovaarm_proxy->getGripperState(); return {gs.ltipforce, gs.rtipforce}; }
    catch (const Ice::Exception&) { return {0.0f, 0.0f}; }
}

float PickandPlaceFSM::gripper_force() const
{
    try { const auto gs = w_.kinovaarm_proxy->getGripperState(); return gs.lforce + gs.rforce; }
    catch (const Ice::Exception&) { return 0.0f; }
}

bool PickandPlaceFSM::via_reached(double e_pos)
{
    if (e_pos < blend_min_dist_) { blend_min_dist_ = e_pos; return false; }
    return e_pos < REACH_TOLERANCE_M * 2.0;
}

// ── Dormant grasp/place phases (Inserting → PlaceRetreating, Retracting) ──────
void PickandPlaceFSM::run_grasp_phases(const std::array<double, Kinematics::N_ARM_JOINTS>& q,
                                       const Eigen::Vector3d& ee_position)
{
    switch (grasp_phase_)
    {
    case GraspPhase::Tracking: break;   // handled by run_tracking()

    case GraspPhase::Retracting:
    {
        w_.gripper_command_ = 1.0f;
        Eigen::Vector3d zdes = w_.bottle_pos_world_ - retract_target_;
        zdes.z() = 0.0;
        zdes = (zdes.norm() > 1e-6) ? zdes.normalized() : Eigen::Vector3d(1, 0, 0);
        const Eigen::Vector3d xdes = Eigen::Vector3d(0, 0, 1).cross(zdes).normalized();
        const auto [e_pos, e_ang] = efe_drive(q, ee_position, retract_target_, zdes, xdes, 0.30);
        (void) e_ang;
        if (e_pos < REACH_TOLERANCE_M or ++retract_ticks_ > RETRACT_SETTLE_TICKS)
        {
            if (reflex_count_ >= MAX_REFLEXES)
            {
                std::print("[reflex] {} tips — giving up; returning to rest.\n", reflex_count_);
                reflex_count_       = 0;
                w_.run_requested_   = false;
            }
            else { std::print("[reflex] settled → re-tracking\n"); grasp_phase_ = GraspPhase::Tracking; }
        }
        break;
    }

    case GraspPhase::Inserting:
    {
        const auto& lg = latched_grasp_;
        w_.gripper_command_ = 1.0f;
        const double insert_vel = INSERT_VEL_MS * rep_perturb_.speed_scale
                                  * (1.0 + insert_conf_gain_ * skill_c());
        const double e_grasp = (ee_position - lg.grasp_pos).norm();

        if (tip_reflex_)
        {
            const auto [lc, rc] = tip_contacts();
            const Eigen::Vector3d lat      = w_.kinematics_->tool_pose(q).rotation.col(0);
            const Eigen::Vector3d approach  = lg.z_tool_des.normalized();
            const Eigen::Vector3d off       = recenter_sign_ * tip_reflex_offset_ * lat;
            if (lc or rc)
            {
                const double dir = (rc and not lc) ? +1.0 : (lc and not rc) ? -1.0 : 0.0;
                tip_reflex_offset_ = std::clamp(tip_reflex_offset_ + dir * TIP_REFLEX_STEP_M,
                                                -TIP_REFLEX_MAX_M, TIP_REFLEX_MAX_M);
                const Eigen::Vector3d target = lg.grasp_pos + off - approach * TIP_REFLEX_BACKOFF_M;
                efe_drive(q, ee_position, target, lg.z_tool_des, lg.x_tool_des, insert_vel);
            }
            else
            {
                const Eigen::Vector3d target = lg.grasp_pos + off;
                efe_drive(q, ee_position, target, lg.z_tool_des, lg.x_tool_des, insert_vel);
                if ((ee_position - target).norm() < REACH_TOLERANCE_M)
                {
                    grasp_phase_ = GraspPhase::Closing; closing_ticks_ = 0;
                    grasp_force_ticks_ = 0; insert_ticks_ = 0;
                    std::print("[grasp] seated (reflex offset {:+.3f} m) → Closing\n", tip_reflex_offset_);
                }
            }
            if (grasp_phase_ == GraspPhase::Inserting and ++insert_ticks_ > INSERT_TIMEOUT_TICKS)
            { log_rep_outcome(false, 0.0, 0.0); miss_or_give_up("reflex stuck"); }
            break;
        }

        if (tactile_recenter_)
        {
            const auto [fl, fr] = tip_forces();
            const float contact = fl + fr;
            const float rel_asym = (contact > 1e-3f) ? (fl - fr) / contact : 0.0f;
            const Eigen::Vector3d lat = w_.kinematics_->tool_pose(q).rotation.col(0);
            const Eigen::Vector3d target = lg.grasp_pos + recenter_sign_ * recenter_gain_ * rel_asym * lat;
            const double v_app = (contact > INSERT_TOUCH_FORCE) ? insert_vel * 0.3 : insert_vel;
            const auto [e_tgt, e_ang] = efe_drive(q, ee_position, target, lg.z_tool_des, lg.x_tool_des, v_app);
            (void) e_ang; (void) e_tgt;
            if (e_grasp < REACH_TOLERANCE_M and contact <= INSERT_TOUCH_FORCE)
            {
                grasp_phase_ = GraspPhase::Closing; closing_ticks_ = 0;
                grasp_force_ticks_ = 0; insert_ticks_ = 0;
                std::print("[grasp] seated & centred (e={:.3f} m) → Closing\n", e_grasp);
            }
            else if (++insert_ticks_ > INSERT_TIMEOUT_TICKS)
            { log_rep_outcome(false, 0.0, 0.0); miss_or_give_up("insert stuck"); }
            break;
        }

        const auto [e_pos, e_ang] = efe_drive(q, ee_position, lg.grasp_pos, lg.z_tool_des, lg.x_tool_des, insert_vel);
        (void) e_ang;
        const float f = gripper_force();
        if (e_pos < REACH_TOLERANCE_M or (f > GRASP_FORCE_THRESH and e_pos < 2.0 * REACH_TOLERANCE_M))
        {
            grasp_phase_ = GraspPhase::Closing; closing_ticks_ = 0; grasp_force_ticks_ = 0;
            std::print("[grasp] at grasp point (e={:.3f} m, f={:.2f}) → Closing\n", e_pos, f);
        }
        break;
    }

    case GraspPhase::Closing:
    {
        const auto& lg = latched_grasp_;
        w_.gripper_command_ = 0.0f;
        efe_drive(q, ee_position, lg.grasp_pos, lg.z_tool_des, lg.x_tool_des, INSERT_VEL_MS);
        const float f = gripper_force();
        grasp_force_ticks_ = (f > GRASP_FORCE_THRESH) ? grasp_force_ticks_ + 1 : 0;
        if (grasp_force_ticks_ >= GRASP_FORCE_HOLD_TICKS)
        {
            grasp_phase_ = GraspPhase::Lifting; reflex_count_ = 0; grasp_force_ticks_ = 0; lift_ticks_ = 0;
            bottle_z_at_lift_start_ = w_.bottle_pos_world_.z();
            sample_place_spot();
            blend_min_dist_ = 1e9;
            std::print("[grasp] contact (f={:.2f}, held {} cy) → Lifting\n", f, GRASP_FORCE_HOLD_TICKS);
        }
        else if (++closing_ticks_ > CLOSING_TIMEOUT_TICKS)
        { log_rep_outcome(false, 0.0, 0.0); miss_or_give_up("no contact"); }
        break;
    }

    case GraspPhase::Lifting:
    {
        const auto& lg = latched_grasp_;
        w_.gripper_command_ = 0.0f;
        const auto [e_pos, e_ang] = efe_drive(q, ee_position, lift_target_,
                                              lg.z_tool_des, lg.x_tool_des, skilled_speed(0.20), place_hover_);
        (void) e_ang;
        const double rise   = w_.bottle_pos_world_.z() - bottle_z_at_lift_start_;
        const double xy_gap = (w_.bottle_pos_world_.head<2>() - ee_position.head<2>()).norm();
        // Confirm on the ACTUAL outcome — the bottle is genuinely up and still co-located —
        // the moment it's true, instead of first waiting for the EE to reach the full lift
        // target (an over-strict kinematic proxy: a slow lift tripped the timeout even
        // though the bottle was firmly held). The arm still aims for the full LIFT_HEIGHT,
        // we just don't BLOCK convergence on reaching it.
        if (rise >= LIFT_CONFIRM_RISE_M and xy_gap <= LIFT_CONFIRM_HOLD_M)
        {
            log_rep_outcome(true, rise, xy_gap);
            // Natural fine-tuning: a confirmed grasp is evidence the model predicted the
            // outcome. Deposit precision weighted by how cleanly it matched (terminal
            // standoff error vs tolerance, lift fraction) — NOT a fixed increment.
            update_confidence_from_outcome(true, rep_commit_epos_ / REACH_TOLERANCE_M, rise / LIFT_HEIGHT_M);
            if (metrics_open_)
                metrics_ << (probe_index_ - 1) << ",1," << confidence_ << ','
                         << (now_seconds() - rep_t0_) << ',' << obs_count_rep_ << '\n', metrics_.flush();
            std::print("[grasp] CONFIRMED held (rose {:.3f} m, conf {:.2f}) → place\n", rise, confidence_);
            lift_ticks_ = 0; grasp_phase_ = GraspPhase::PlaceMoving; place_ticks_ = 0; blend_min_dist_ = 1e9;
            std::print("[place] lifted → carry to spot ({:.3f},{:.3f},{:.3f}) → PlaceMoving\n",
                       place_pos_.x(), place_pos_.y(), place_pos_.z());
        }
        // EE reached the lift target but the bottle did NOT come up → closed on air.
        else if (via_reached(e_pos))
        {
            std::print("[grasp] MISS — bottle not held (rose {:.3f} m, gap {:.3f} m) → reopen\n", rise, xy_gap);
            log_rep_outcome(false, rise, xy_gap);
            miss_or_give_up("bottle not held");
        }
        else if (++lift_ticks_ > LIFT_TIMEOUT_TICKS)
        {
            log_rep_outcome(false, rise, 0.0);
            miss_or_give_up("lift stalled");
        }
        break;
    }

    case GraspPhase::PlaceMoving:
    {
        w_.gripper_command_ = 0.0f;
        const auto [e_pos, e_ang] = efe_drive(q, ee_position, place_hover_, place_z_des_, place_x_des_,
                                              skilled_speed(0.20), place_pos_, PLACE_ORIENT_GAIN);
        (void) e_ang;
        if (via_reached(e_pos) or ++place_ticks_ > PLACE_TIMEOUT_TICKS)
        {
            grasp_phase_ = GraspPhase::PlaceLowering; place_ticks_ = 0;
            place_settle_ticks_ = 0; place_bottle_z_prev_ = 1e9;
            std::print("[place] above spot → PlaceLowering\n");
        }
        break;
    }

    case GraspPhase::PlaceLowering:
    {
        w_.gripper_command_ = 0.0f;
        const auto [e_pos, e_ang] = efe_drive(q, ee_position, place_pos_, place_z_des_, place_x_des_,
                                              skilled_speed(0.18), std::nullopt, PLACE_ORIENT_GAIN);
        (void) e_ang;
        const double tilt = std::acos(std::clamp(std::abs(w_.bottle_axis_world_.normalized().z()), 0.0, 1.0));
        const double dz = std::abs(w_.bottle_pos_world_.z() - place_bottle_z_prev_);
        place_bottle_z_prev_ = w_.bottle_pos_world_.z();
        const bool upright    = tilt < PLACE_UPRIGHT_TOL_RAD;
        const bool bottle_down = (w_.bottle_pos_world_.z() - w_.table_top_z_) < PLACE_ON_TABLE_M and dz < 0.0008;
        const bool ok = (bottle_down or e_pos < REACH_TOLERANCE_M) and upright;
        place_settle_ticks_ = ok ? place_settle_ticks_ + 1 : 0;
        if (place_settle_ticks_ >= PLACE_SETTLE_TICKS or ++place_ticks_ > PLACE_TIMEOUT_TICKS)
        {
            grasp_phase_ = GraspPhase::PlaceReleasing; place_ticks_ = 0; place_settle_ticks_ = 0;
            std::print("[place] set down (e={:.3f} m, tilt {:.1f}°) → PlaceReleasing\n", e_pos, tilt * 57.29578);
        }
        break;
    }

    case GraspPhase::PlaceReleasing:
    {
        w_.gripper_command_ = 1.0f;
        efe_drive(q, ee_position, place_pos_, place_z_des_, place_x_des_, 0.05);
        if (++place_ticks_ >= release_ticks_)
        {
            const auto tp = w_.kinematics_->tool_pose(q);
            retreat_z_des_      = tp.rotation.col(2).normalized();
            retreat_x_des_      = tp.rotation.col(0).normalized();
            retreat_target_pos_ = tp.position - retreat_z_des_ * PLACE_RETREAT_DIST_M;
            grasp_phase_ = GraspPhase::PlaceRetreating; place_ticks_ = 0; place_settle_ticks_ = 0;
            std::print("[place] released ({} cy) → PlaceRetreating along gripper axis\n", release_ticks_);
        }
        break;
    }

    case GraspPhase::PlaceRetreating:
    {
        w_.gripper_command_ = 1.0f;
        const double rspeed = retreat_speed_ * (1.0 + retreat_perturb_.dspeed);
        const auto [e_pos, e_ang] = efe_drive(q, ee_position, retreat_target_pos_,
                                              retreat_z_des_, retreat_x_des_, rspeed);
        (void) e_ang;
        if (e_pos < REACH_TOLERANCE_M or ++place_ticks_ > PLACE_TIMEOUT_TICKS)
        {
            const double tilt_deg = bottle_tilt_rad() * 57.29578;
            const bool   tipped   = tilt_deg > (FALL_TILT_RAD * 57.29578);
            log_retreat_outcome(tilt_deg, tipped);
            std::print("[retreat] done — bottle tilt {:.1f}°{}\n", tilt_deg, tipped ? "  TIPPED" : " (upright)");
            w_.kinovaarm_proxy->moveJointsWithAngle(w_.nearest_equiv_target(q, w_.rest_pose_angles_));
            returning_for_cycle_   = true;
            w_.homing_settled_ticks_ = 0;
            w_.homing_elapsed_ticks_ = 0;
            grasp_phase_           = GraspPhase::Tracking;
            place_ticks_           = 0;
            w_.phase_              = SpecificWorker::Phase::Homing;
            ++pick_place_cycles_done_;
            std::print("[cycle] {}/{} pick-and-place complete → returning to rest\n",
                       pick_place_cycles_done_, round_cycles_ > 0 ? round_cycles_ : pick_place_cycles_done_);
            return;
        }
        break;
    }
    } // end switch
}
