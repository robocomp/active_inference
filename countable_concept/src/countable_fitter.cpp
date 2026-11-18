/*
 * countable_fitter.cpp — the active-inference fit core for countable_concept (see countable_fitter.h).
 *
 * Implements the per-"countable_*" instance lifecycle and the AI2 full-covariance belief update: ensure_instance
 * (birth-seed + RT/prior warm-start + NaN sanitize), observe_slice (mask-cloud → candidate/residual SDF split
 * + per-slice R inputs), and run_inference (lazy footprint-moment birth, support-bank ingest, one CountableBelief
 * update with range/motion covariance, the step-bound divergence net, FE-surprise attention, orientation-mode
 * resolution, and write-back into the legacy CountableState). Collaborates with MaskIngestor, CountableSceneGraph,
 * CountableProjection, CountableLidarRangeChannel, and the header-only support bank; SpecificWorker owns orchestration.
 */

#include "countable_fitter.h"

#include "../../common/exclusion/exclusion.h"   // rc::exclusion — the SHARED no-two-objects rule

#include "../../common/diag_log/rotating_csv.h"   // keep the previous run instead of wiping it
#include "countable_support_bank.h"
#include "../../common/object_anchor/object_anchor_contract.h"
#include "../../common/object_anchor/ray_anisotropic_cov.h"
#include "../../common/media_transport/media_transport.h"   // rc::media::sensor_model_from_graph, max_ring_gap_rad
#include "countable_mask_membership.h"   // rc::MaskMembership, make_membership (lidar-measures G2)
#include "countable_viewpoint.h"         // ρ from odometry (lidar-measures §7c.2)
#include "../../common/place_memory/place_map.h"   // rc::place::inv_se2 (the robot in the room from T(robot←room))

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <locale>
#include <print>
#include <random>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include "../../common/room_resolve/room_resolve.h"   // rc::room::current_room_frame — the room is room_1/room_2/…, never the literal "room"

namespace rc {

// ─── Construction & chain covariance ──────────────────────────────────────────────────────────────

CountableFitter::CountableFitter(std::shared_ptr<DSR::DSRGraph> graph,
                         DSR::InnerEigenAPI* inner_eigen,
                         CountableConfig& cfg,
                         MaskIngestor* mask_ingestor,
                         CountableSceneGraph* scene_graph)
    : G_(graph), inner_eigen_(inner_eigen), cfg_(cfg),
      mask_ingestor_(mask_ingestor), scene_graph_(scene_graph),
      projection_(std::make_unique<CountableProjection>(graph, inner_eigen, mask_ingestor, cfg.concept_class))
{
    CountableBeliefState::use_quotient = cfg.quotient_chart;   // C2v symmetry-quotient optimisation chart (global mode)
    projection_->set_central_region_frac(cfg.central_region_frac);
    projection_->set_front_params(cfg.front_min_face_area_px, cfg.front_min_confidence);
}

// ─── Ego-motion "be-still-to-update" (ported 1:1 from chair_concept) ─────────────────────────────────

// Robot/camera speed in the room frame from the transform chain (producer-independent). Uses the SAME room_T_zed
// extrinsic the fit uses (CountableProjection), pinned to the CURRENT pose (ts=0). Call once per compute cycle.
void CountableFitter::update_ego_motion()
{
    const auto M = projection_->room_T_zed_matrix(0);   // camera→room (ts=0 = current robot pose)
    const auto now = std::chrono::steady_clock::now();
    if (cfg_.lidar_measures_on_gated) read_odometry_and_pose(M);   // lidar-measures §7c.2: odometry + pose for ρ
    if (not M.has_value())
    {
        have_prev_cam_ = false;   // pose chain unavailable → can't judge motion; reset the baseline
        return;
    }
    const Eigen::Vector3f pos(static_cast<float>(M->coeff(0, 3)),
                              static_cast<float>(M->coeff(1, 3)),
                              static_cast<float>(M->coeff(2, 3)));
    const Eigen::Vector3f fwd(static_cast<float>(M->coeff(0, 1)),   // zed +y is the depth/forward axis
                              static_cast<float>(M->coeff(1, 1)),
                              static_cast<float>(M->coeff(2, 1)));
    if (have_prev_cam_)
    {
        const float dt = std::max(1e-3f, std::chrono::duration<float>(now - prev_cam_tp_).count());
        ego_lin_mps_ = (pos - prev_cam_pos_).norm() / dt;
        const float fa = prev_cam_fwd_.norm(), fb = fwd.norm();
        const float cang = (fa > 1e-6f and fb > 1e-6f)
            ? std::clamp(prev_cam_fwd_.dot(fwd) / (fa * fb), -1.0f, 1.0f) : 1.0f;
        ego_ang_radps_ = std::acos(cang) / dt;
    }
    prev_cam_pos_ = pos; prev_cam_fwd_ = fwd; prev_cam_tp_ = now; have_prev_cam_ = true;
}

// ─── Camera attributes, LiDAR measures (Model.LidarMeasuresOnGated; spec 2026-11-15 lidar-measures §7c) ────────────

// The ZED mount on the body frame (ROBOT_GEOMETRY.md, from robot_concept/shadow.json; body x-right, y-forward, z-up):
// zed T[0, −0.075, +0.945]. Used ONLY to recover the body centre from room_T_zed (the chain's reference pose); the Helios
// position is the LiDAR channel's staged sweep origin, measured through the RT chain, not a mount constant.
namespace {
constexpr float kZedMountY      = -0.075f;
Eigen::Vector2f rotate_yaw(float yaw, const Eigen::Vector2f& v)
{
    const float c = std::cos(yaw), s = std::sin(yaw);
    return {c * v.x() - s * v.y(), s * v.x() + c * v.y()};
}
}  // namespace

// Once per cycle (main/compute thread): the robot's odometry from the robot node's TYPED speed attributes, the current
// robot pose (room frame) from the same room_T_zed the fit uses, and the Helios ring gap from its media descriptor.
void CountableFitter::read_odometry_and_pose(const std::optional<Eigen::Matrix4d>& M)
{
    have_pose_now_ = false;
    odo_ts_ = 0;
    if (not G_) return;
    if (M.has_value())
    {
        const Eigen::Vector3f zed(static_cast<float>(M->coeff(0, 3)), static_cast<float>(M->coeff(1, 3)),
                                  static_cast<float>(M->coeff(2, 3)));
        // zed +y is the forward axis and the zed mount has no rotation (ROBOT_GEOMETRY.md), so it is the robot's +y.
        const float fx = static_cast<float>(M->coeff(0, 1)), fy = static_cast<float>(M->coeff(1, 1));
        robot_yaw_now_ = std::atan2(-fx, fy);   // R(yaw)·(0, 1) = (−sin yaw, cos yaw) = forward
        zed_xy_now_    = zed.head<2>();
        robot_xy_now_  = zed_xy_now_ - rotate_yaw(robot_yaw_now_, {0.0f, kZedMountY});
        have_pose_now_ = true;
    }
    // The Helios position: this cycle's staged sweep origin (the worker stages the sweep before update_ego_motion); on a
    // cycle without a fresh sweep the last staged origin is kept.
    if (const auto o = lidar_channel_.helios_origin(); o.has_value()) { helios_now_ = *o; have_helios_ = true; }
    // The robot node (Shadow, type "robot"): found once by type, then read by id (no per-cycle copy of every robot node).
    std::optional<DSR::Node> robot;
    if (robot_node_id_ != 0) robot = G_->get_node(robot_node_id_);
    if (not robot.has_value())
    {
        const auto robots = G_->get_nodes_by_type("robot");
        if (not robots.empty()) { robot = robots.front(); robot_node_id_ = robots.front().id(); }
    }
    if (robot.has_value())
    {
        // The shared reader (one place for the advance/side axis convention): linear = [side (x), advance (y), 0].
        if (const auto tw = rc::rt::robot_body_twist(*G_, robot.value()); tw.has_value() and tw->stamp_ms != 0)
        { odo_adv_ = tw->linear.y(); odo_side_ = tw->linear.x(); odo_omega_ = tw->angular.z(); odo_ts_ = tw->stamp_ms; }
        // The twist is per SIMULATION second on a simulator: feed the wall→sim clock (identity on real hardware).
        if (const auto st = rc::rt::robot_twist_sim_stamps(*G_, robot.value()); st.has_value())
            sim_clock_.observe(st->first, st->second);
    }
    // The Helios beam pattern (spec §7c.2: g = max_ring_gap_rad of its SensorModel). It can be advertised late, so it is
    // looked up every cycle until found; absent ⇒ ρ_L = 1 (logged per instance in stage_persistent).
    if (helios_ring_gap_rad_ <= 0.0f)
        if (const auto sm = rc::media::sensor_model_from_graph(*G_, "helios"); sm.has_value())
            if (const auto g = rc::media::max_ring_gap_rad(*sm); g.has_value())
            {
                helios_ring_gap_rad_ = *g;
                std::print("[lidar-measures] Helios SensorModel: max ring gap {:.4f} rad\n", *g);
            }
}

// Integrate the odometry since this instance's last sample into the two viewpoint paths and the chain's odometry
// displacement. dt is the PRODUCER-clock interval between the two speed stamps (rc::rt::SimClockRate::interval_s, as
// door_concept's anchor does): the twist is per simulation second on a simulator, and a paused simulator integrates
// nothing. A stale or absent speed gives no step (fails closed, ρ → 1).
void CountableFitter::integrate_odometry(CountableInstance& inst)
{
    if (not have_pose_now_) return;   // no pose: nothing is consumed; the next interval covers this one
    // Three outcomes for the interval since this instance's last odometry sample:
    //  - integrable (stamp advanced, producer interval < 1 s): the step goes into both paths and the chain's odometry;
    //  - unknown (no speed/stamp, first sample, interval ≥ 1 s, or a stamp frozen for ≥ 1 s of producer time): no step
    //    (Δ = 0, ρ_cam/ρ_L fail closed), and the chain absorbs the stretch's pose increment — it is not a localiser
    //    correction (final review I1: fail closed, like the paths);
    //  - pending (stamp not advanced yet, < 1 s old): nothing is consumed, so the next sample's interval covers it.
    // "1 s ⇒ unknown" is the producer's convention (robot_concept specificworker.cpp ~2288:
    // `period_ms = … own_ms - prev < 1000 ? … : 0`, "0 (unknown) on the first or after a gap"), not a new constant.
    std::optional<Eigen::Vector3f> step3;   // (dx, dy, dpsi) room frame
    bool pending = false;
    if (odo_ts_ != 0 and inst.last_odo_ts != 0)
    {
        if (odo_ts_ > inst.last_odo_ts)
        {
            // dt is PRODUCER-clock time (rc::rt::SimClockRate, as door_concept's anchor): the twist is per simulation
            // second on a simulator, and a paused simulator integrates nothing. Identity on real hardware.
            const double dt = sim_clock_.interval_s(static_cast<std::int64_t>(inst.last_odo_ts),
                                                    static_cast<std::int64_t>(odo_ts_));
            if (dt < 1.0)
                step3 = rc::odometry_step_room_se2(odo_adv_, odo_side_, odo_omega_, robot_yaw_now_, static_cast<float>(dt));
        }
        else
        {
            const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::system_clock::now().time_since_epoch()).count();
            pending = sim_clock_.interval_s(static_cast<std::int64_t>(odo_ts_), now_ms) < 1.0;
        }
    }
    if (pending) return;
    // e's displacement (spec 7c.7 b): the odometry step; over a stretch the odometry cannot integrate, the pose increment
    // (the robot did move; the localiser's share of it is what ρ_chain already reads as a correction).
    if (step3.has_value())         inst.chain_dr += step3->head<2>();
    else if (inst.chain.have)      inst.chain_dr += robot_xy_now_ - inst.chain.prev.head<2>();
    if (step3.has_value())
    {
        const Eigen::Vector2f step = step3->head<2>();
        const auto& s = inst.ai2_belief.state();
        const Eigen::Vector2f centre(s.cx, s.cy);
        rc::accumulate_path(inst.path_cam, zed_xy_now_, step, centre);
        if (have_helios_) rc::accumulate_path(inst.path_L, helios_now_.head<2>(), step, centre);
    }
    rc::chain_observe(inst.chain, Eigen::Vector3f(robot_xy_now_.x(), robot_xy_now_.y(), robot_yaw_now_), step3);
    if (odo_ts_ > inst.last_odo_ts) inst.last_odo_ts = odo_ts_;
}

// Stage the persistent-path inputs on a frame: ρ of each latent from the paths accumulated since its channel's last update
// (§7c.2), the latents' marginals and the fresh ego-motion variances. Pure function of the instance + this cycle's pose.
void CountableFitter::stage_persistent(CountableInstance& inst, CountableFrame& frame, const LatentInputs& li) const
{
    const auto& s = inst.ai2_belief.state();
    const Eigen::Vector2f c(s.cx, s.cy);
    // Camera: ρ_cam = exp(−|Δφ|/atan(w⊥/2z) − |Δz|/z), w⊥ = the box's extent across the line of sight.
    float rho_cam = 1.0f;
    if (have_pose_now_)
    {
        const Eigen::Vector2f r = zed_xy_now_ - c;
        const float z = r.norm();
        if (z > 0.0f)
        {
            const Eigen::Vector2f t(-r.y() / z, r.x() / z);
            const Eigen::Vector2f ux(std::cos(s.yaw), std::sin(s.yaw)), uy(-std::sin(s.yaw), std::cos(s.yaw));
            const float w_perp = s.w * std::abs(ux.dot(t)) + s.h * std::abs(uy.dot(t));
            if (w_perp > 0.0f) rho_cam = rc::rho_camera(inst.path_cam, z, w_perp);   // degenerate geometry ⇒ parked
        }
    }
    // LiDAR: lateral DOFs and yaw by the ring gap; H by the ring ε that grazes the top moving over it with range.
    float rho_lat = 1.0f, rho_H = 1.0f;
    if (helios_ring_gap_rad_ > 0.0f and have_helios_)
    {
        const float zl = (helios_now_.head<2>() - c).norm();
        rho_lat = rc::rho_lidar_lateral(inst.path_L, helios_ring_gap_rad_);
        if (zl > 0.0f)
            rho_H = rc::rho_lidar_height(inst.path_L, zl, std::atan2(s.H - helios_now_.z(), zl), helios_ring_gap_rad_);
    }
    else if (helios_ring_gap_rad_ <= 0.0f and not inst.dbg_rhoL_absent_logged)
    {
        std::print("[{}] [lidar-measures] no Helios SensorModel on the graph ⇒ ρ_L = 1\n", inst.node_name);
        inst.dbg_rhoL_absent_logged = true;
    }
    // e, the robot's pose error (spec 7c.7 b): Σ_pose and the robot's room position at the mask's capture stamp, from the
    // robot→room edge; ρ_chain = exp(−½cᵀΣ_pose⁻¹c), c = the localiser's SE(2) correction (pose increment ⊖ odometry
    // increment, ψ included) since e's last update — a relocalisation jump gives ρ ≈ 0 by the formula itself. No pose
    // covariance on the edge ⇒ Σ_pose = 0 (e a zero-variance latent), logged once per instance.
    const PoseAtStamp ps = pose_at_stamp(inst.last_mask_timestamp_ms);
    if (not ps.ok and not inst.dbg_pose_cov_absent_logged)
    {
        std::print("[{}] [lidar-measures] no robot pose covariance on the robot→room edge at the capture stamp ⇒ Σ_pose = 0 "
                   "(e is a zero-variance latent)\n", inst.node_name);
        inst.dbg_pose_cov_absent_logged = true;
    }
    const Eigen::Vector3f cc = inst.chain.have ? rc::chain_correction(inst.chain) : Eigen::Vector3f::Zero();
    const float rho_ch = static_cast<float>(rc::rho_chain(cc.cast<double>(), ps.sigma));
    inst.dbg_chain_c  = cc;
    inst.dbg_spose_sd = ps.sigma.diagonal().cwiseMax(0.0).cwiseSqrt();
    frame.rho_cam.fill(rho_cam);
    frame.rho_L   = {rho_lat, rho_lat, rho_H, rho_lat, rho_lat, rho_lat};
    frame.rho_chain  = rho_ch;
    frame.sigma_pose = ps.sigma;
    frame.chain_dr   = inst.chain_dr;
    frame.robot_xy   = ps.ok ? ps.robot_xy : robot_xy_now_;
    frame.sig2_cam   = li.sig2_cam;
    frame.sig2_L     = li.sig2_L;
    frame.fresh_cam  = li.fresh;
    frame.fresh_L    = li.fresh;
}

// After an ACCEPTED persistent update: the measuring channels' paths restart, and the chain's reference pose moves here.
void CountableFitter::consume_paths(CountableInstance& inst, const CountableFrame& frame, bool camera, bool lidar)
{
    // The log records the ρ THIS accepted update used, per channel that measured (neutral 1 / 0 otherwise, see run_inference).
    if (camera)
    {
        inst.dbg_rho_cam  = frame.rho_cam[0];
        inst.dbg_dphi_cam = inst.path_cam.dphi_rad; inst.dbg_dz_cam = inst.path_cam.dz_m;
        inst.path_cam = {};
    }
    if (lidar)
    {
        inst.dbg_rho_L = frame.rho_L[0];
        inst.path_L    = {};
    }
    if (camera or lidar)
    {
        inst.dbg_rho_chain = frame.rho_chain;
        rc::chain_rebase(inst.chain);
        inst.chain_dr.setZero();                   // e's lever and transport have consumed this displacement
    }
}

// The robot's pose in the room at a capture stamp (spec 7c.7 b, R4-C3), from the robot→room RT edge's history ring: the slot
// whose stamp is nearest ts (an untimestamped edge has one block), T(robot←room) at that slot inverted for the robot's room
// position, and that slot's rt_covariance block carried into the room frame (pose_cov_in_room, double). The covariance is
// one 36-block per slot, or a single block that belongs to the newest slot (room_concept proposal_adoption convention).
CountableFitter::PoseAtStamp CountableFitter::pose_at_stamp(std::uint64_t ts_ms) const
{
    PoseAtStamp out;
    if (not G_ or robot_node_id_ == 0) return out;
    const auto room = G_->get_node(rc::room::current_room_frame(*G_));
    if (not room.has_value()) return out;
    const auto e = G_->get_edge(robot_node_id_, room->id(), "RT");
    if (not e.has_value()) return out;
    const auto tr  = G_->get_attrib_by_name<rt_translation_att>(e.value());
    const auto rot = G_->get_attrib_by_name<rt_rotation_euler_xyz_att>(e.value());
    const auto cv  = G_->get_attrib_by_name<rt_covariance_att>(e.value());
    if (not tr.has_value() or not rot.has_value() or not cv.has_value()) return out;
    const std::vector<float>& t = tr->get();
    const std::vector<float>& r = rot->get();
    const std::vector<float>& c = cv->get();
    if (t.empty() or t.size() % 3 != 0 or r.size() != t.size() or c.empty() or c.size() % 36 != 0) return out;
    const std::size_t nb = t.size() / 3;
    std::size_t idx = nb - 1, newest = nb - 1;
    if (const auto st = G_->get_attrib_by_name<rt_timestamps_att>(e.value()); st.has_value() and st->get().size() == nb)
    {
        const std::vector<std::uint64_t>& stamps = st->get();
        bool any = false;
        std::uint64_t best_d = 0, best_t = 0;
        for (std::size_t b = 0; b < nb; ++b)
        {
            if (stamps[b] == 0) continue;                                  // a slot never written carries no pose
            const std::uint64_t d = stamps[b] > ts_ms ? stamps[b] - ts_ms : ts_ms - stamps[b];
            if (not any or d < best_d) { best_d = d; idx = b; }
            if (not any or stamps[b] >= best_t) { best_t = stamps[b]; newest = b; }
            any = true;
        }
        if (not any) return out;
    }
    std::span<const float> block;
    if (c.size() == 36 * nb)                   block = std::span<const float>(c).subspan(36 * idx, 36);
    else if (c.size() == 36 and idx == newest) block = std::span<const float>(c).first(36);
    else return out;                                                         // no covariance block for that slot
    const Eigen::Vector3f edge_pose(t[3 * idx], t[3 * idx + 1], r[3 * idx + 2]);    // T(robot←room): (x, y, ψ)
    const Eigen::Vector3f room_robot = rc::place::inv_se2(edge_pose);              // the robot in the room
    out.sigma    = rc::pose_cov_in_room(block, edge_pose);
    out.robot_xy = room_robot.head<2>();
    out.ok       = true;
    return out;
}

// The camera's membership m(p) for this cycle's assigned ZED slice (spec G2, §7b): the slice's pixels (delivered image),
// the return projected with the extrinsic at the MASK stamp (zed_T_room), σ_px from the projection's own uncertainty.
// Binds the CameraAPI explicitly (CountableProjection::ensure_camera) — project_to_zed / focal_px never bind. Any input
// missing ⇒ an empty function (no membership: predict on a gated cycle, unweighted rays on an admissible one).
std::function<float(const Eigen::Vector3f&)> CountableFitter::build_membership(const CountableInstance& inst,
                                                                                int slice_index,
                                                                                const Eigen::Matrix4f& zed_T_room,
                                                                                bool have_cam, float mot_pos_var)
{
    if (rgb_frame_.empty() and not rgb_absent_logged_)
    {
        std::print("[lidar-measures] no RGB frame staged ⇒ no delivered image size ⇒ no membership: gated cycles "
                   "predict, admissible ones use unweighted rays\n");
        rgb_absent_logged_ = true;
    }
    if (slice_index < 0 or not have_cam or not mask_ingestor_ or rgb_frame_.empty())
        return {};
    const auto& pkt = mask_ingestor_->packet();
    if (not pkt.valid or slice_index >= static_cast<int>(pkt.slices.size()))
        return {};
    const auto& sl = pkt.slices[slice_index];
    const std::size_t b = std::min(sl.pixel_begin, pkt.mask_pixels.size());
    const std::size_t e = std::min(sl.pixel_end,   pkt.mask_pixels.size());
    if (e <= b or not projection_->ensure_camera())
        return {};
    const int cols = rgb_frame_.cols, rows = rgb_frame_.rows;   // the DELIVERED image size (mask pixels live there)
    const float f = projection_->focal_px(cols, rows);
    if (not (f > 0.0f))
        return {};
    const auto mm = rc::MaskMembership::build(std::span<const Eigen::Vector2f>(pkt.mask_pixels.data() + b, e - b));
    if (mm.empty())
        return {};
    const CountableProjection* proj = projection_.get();
    rc::Projector project = [proj, zed_T_room, cols, rows](const Eigen::Vector3f& p)
    { return proj->project_to_zed(zed_T_room, p, cols, rows); };
    // The robot pose error between the two legs of the membership: a Helios return goes into the room at the SWEEP's
    // capture stamp (lidar_plane_reader.cpp:181, room<-helios, Interpolated) and back into the ZED image with zed_T_room at
    // the MASK's capture stamp (countable_projection.cpp:136, room<-body, Interpolated) — the same robot→room ring, two
    // different stamps, so the error does not cancel exactly (ruling T13-6). Flag on: the in-image-relevant std of the
    // robot's pose error at the mask stamp, √(½(Σ_pose,xx + Σ_pose,yy)) (spec 7c.7 b); flag off: the chain cov at the
    // belief centre, as before (compute_chain_cov).
    float sigma_chain = std::sqrt(std::max(0.0f, 0.5f * (inst.chain_cov_xx + inst.chain_cov_yy)));
    if (cfg_.lidar_measures_on_gated)
    {
        const Eigen::Matrix3d Sp = pose_at_stamp(inst.last_mask_timestamp_ms).sigma;
        sigma_chain = static_cast<float>(std::sqrt(std::max(0.0, 0.5 * (Sp(0, 0) + Sp(1, 1)))));
    }
    return rc::make_membership(std::move(project), mm, f, cfg_.existence_sensor_sigma_m, sigma_chain, mot_pos_var,
                               sl.confidence);
}

// Robust combined ego-motion magnitude (m/s): the per-mask corruption speed OR'd with the robot's own measured
// speed (linear + a lever-arm conversion of angular), so it works whether or not the producer populated motion_dotd.
float CountableFitter::motion_magnitude(const CountableInstance& inst) const
{
    return std::max(std::abs(inst.last_motion_dotd),
                    ego_lin_mps_ + cfg_.ai2_ang_lever_m * ego_ang_radps_);
}

// Off-axis penalty ∈ [0,1]: 0 on the optical axis (a centred mask has no peripheral smear/distortion), → 1 at
// centroid radius ai2_periph_ref. This is the "well-centred masks stay trustworthy" lever, expressed continuously.
float CountableFitter::periphery_penalty(const CountableInstance& inst) const
{
    const float ref = std::max(1e-6f, cfg_.ai2_periph_ref);
    const float r = std::max(0.0f, inst.last_centroid_radius) / ref;
    return std::clamp(r * r, 0.0f, 1.0f);
}

// "Be-still-to-update": true ⇒ this frame may only CONFIRM (predict-only), never move/reshape the geometry mean.
// Robot linear/angular speed above the still-level, OR the mask's own ego-motion corruption (motion_dotd) above
// its still-level. EXCEPTION: a well-centred mask (near the principal point) is trustworthy even while moving.
// Is THIS raw mask good enough to move geometry? Byte-for-byte the admissibility the UPDATE path applies to an
// existing instance (the trunc gate + confirm_only), but evaluated on a slice that has no instance yet — every
// input is already on the slice (motion_dotd, centroid_radius, trunc_frac) plus this cycle's ego speeds.
//
// ★A frame that may not MOVE an existing belief must not CREATE one. Birth used to admit frames the fit itself
// would refuse — a smeared mask taken while the robot drives past, or one clipped at the image border — which is
// how furniture appeared in passing. One rule, defined once, applied to both ends of the lifecycle.
bool CountableFitter::frame_admissible(const rc::MaskIngestor::MaskSlice& sl) const
{
    if (sl.trunc_frac > cfg_.ai2_trunc_gate_frac)
        return false;                                   // truncated ⇒ extent is a lower bound ⇒ never births
    if (not cfg_.ai2_motion_confirm_only)
        return true;
    const bool moving = ego_lin_mps_   > cfg_.ai2_still_lin_mps
                     or ego_ang_radps_ > cfg_.ai2_still_ang_radps
                     or std::abs(sl.motion_dotd) > cfg_.ai2_still_dotd;
    if (not moving)
        return true;                                    // robot still ⇒ admissible
    // Moving: only a well-CENTRED mask is trusted (the robot is looking AT it), exactly as in confirm_only.
    return cfg_.ai2_moving_update_center_radius >= 0.0f
       and sl.centroid_radius <= cfg_.ai2_moving_update_center_radius;
}

bool CountableFitter::confirm_only(const CountableInstance& inst) const
{
    if (not cfg_.ai2_motion_confirm_only)
        return false;
    const bool moving = ego_lin_mps_   > cfg_.ai2_still_lin_mps
                     or ego_ang_radps_ > cfg_.ai2_still_ang_radps
                     or std::abs(inst.last_motion_dotd) > cfg_.ai2_still_dotd;
    if (not moving)
        return false;
    if (cfg_.ai2_moving_update_center_radius >= 0.0f
        and inst.last_centroid_radius <= cfg_.ai2_moving_update_center_radius)
        return false;   // well-centred mask → trust it even while moving
    return true;        // moving AND the mask is off-centre → confirmation only
}

// ─── Appearance FRONT (door) resolver ───────────────────────────────────────────────────────────────

// Project the fitted box into the newest ZED RGB, score door-ness per visible face, and fold the winning cue
// into the belief's discrete door-mode resolver. No-op unless the feature is on and a FRESH RGB frame is staged;
// the per-instance capture-stamp gate stops a stalled/duplicated frame from re-integrating the same appearance.
void CountableFitter::resolve_front_from_rgb(CountableInstance& inst, float mode_evidence_weight)
{
    inst.last_front_conf = 0.0f;
    if (not cfg_.front_detect_enabled or not rgb_fresh_ or rgb_frame_.empty())
        return;
    // Freshness-as-precision: skip a frame we already folded in (same capture stamp). stamp==0 (unstamped
    // producer) falls back to the per-cycle rgb_fresh_ flag, which the worker only sets on a genuinely new frame.
    if (rgb_stamp_ms_ != 0 and rgb_stamp_ms_ == inst.last_front_stamp_ms)
        return;
    inst.last_front_stamp_ms = rgb_stamp_ms_;

    // Project the FRESHLY-UPDATED belief box (not the not-yet-written-back model state) into the RGB.
    const auto& bs = inst.ai2_belief.state();
    CountableState box = inst.model.state();
    box.cx = bs.cx; box.cy = bs.cy; box.countable_height = bs.H; box.w = bs.w; box.h = bs.h; box.yaw = bs.yaw;
    const auto cue = projection_->detect_front(box, rgb_frame_, rgb_stamp_ms_);
    if (not cue.has_value())
        return;
    // Where this look came FROM, for the accumulator's novelty budget (see FrontCue::view_bearing_rad).
    // ★PINNED TO THE CAPTURE STAMP, not ts=0. Asking for the LATEST camera pose files a look under wherever
    // the camera is NOW, so while the robot moves the novelty bookkeeping mis-registers itself: it can credit
    // an exhausted viewpoint as fresh and charge a genuinely new one as spent. detect_front already pins its
    // own projection to this same stamp (countable_projection.cpp), so using ts=0 here also meant the cue
    // and its provenance disagreed about which camera pose produced them.
    if (const auto M = projection_->room_T_zed_matrix(rgb_stamp_ms_); M.has_value())
    {
        const float camx = static_cast<float>((*M)(0, 3)), camy = static_cast<float>((*M)(1, 3));
        const_cast<FrontCue&>(*cue).view_bearing_rad = std::atan2(camy - bs.cy, camx - bs.cx);
    }
    inst.last_front_conf    = cue->confidence;
    inst.last_front_bearing = cue->bearing_rad;
    // evidence_weight = ego-motion reliability (a moving frame's smeared door face barely votes on the discrete
    // orientation); the belief additionally weights by the cue confidence. Adopt-on-argmax is inside the belief.
    const bool flipped = inst.ai2_belief.resolve_front(*cue, mode_evidence_weight);
    if (cfg_.front_log and (flipped or should_log(inst)))
        std::print("[{}] FRONT cue bearing={:.3f} conf={:.2f} w={:.2f} → {} | door_mode={} p={:.2f} σyaw={:.2f}deg\n",
                   inst.node_name, cue->bearing_rad, cue->confidence, mode_evidence_weight,
                   flipped ? "ADOPTED door flip" : "held", inst.ai2_belief.front_mode(),
                   inst.ai2_belief.front_confidence(),
                   std::sqrt(inst.ai2_belief.yaw_marginal_var()) * 57.29578f);
}

// Enable Part-B chain-covariance propagation from source_frame (no-op unless a gaussian API + frame are given).
void CountableFitter::set_chain_cov_source(DSR::InnerGaussianAPI* gaussian, std::string source_frame)
{
    gaussian_          = gaussian;
    chain_src_frame_   = std::move(source_frame);
    chain_cov_enabled_ = (gaussian_ != nullptr) and not chain_src_frame_.empty();
}

void CountableFitter::set_object_observation(bool enabled, std::string robot_frame)
{
    obs_robot_frame_ = std::move(robot_frame);
    obs_enabled_     = enabled and (inner_eigen_ != nullptr) and not obs_robot_frame_.empty();
}

// Build the object-anchor observation z_o for room_concept's landmark factor.
//
// z_o MUST be independent of the robot pose the localizer is estimating, or the factor just
// re-anchors to the last pose (the residual is ~0 at the current estimate). So we take the bed's
// RAW camera-frame mask centroid (this frame's ZED measurement, no localization in it) and carry it
// to the robot base by the STATIC body←zed extrinsic (ts=0, a fixed calibrated mount). Position-only:
// a single view's yaw is biased (the grazing/obliquity problem), so we publish [x,y] and let the
// consumer treat it as a 2-DOF landmark. Gated OFF by default.
void CountableFitter::compute_object_observation(CountableInstance& inst)
{
    inst.obs_robot_valid = false;
    if (not obs_enabled_ or not inner_eigen_ or not mask_ingestor_)
        return;
    const auto& packet = mask_ingestor_->packet();
    if (packet.support_points_cam.empty())
        return;

    // This frame's ZED slice assigned to this bed (the pinhole camera cloud; skip ricoh depth_var>0).
    const MaskIngestor::MaskSlice* zed_slice = nullptr;
    for (const int idx : inst.assigned_mask_idxs)
    {
        if (idx < 0 or idx >= static_cast<int>(packet.slices.size())) continue;
        const auto& sl = packet.slices[idx];
        if (sl.depth_var > 0.0f) continue;                      // ricoh / lidar-depth mask, not the ZED frame
        if (sl.support_end > sl.support_begin) { zed_slice = &sl; break; }
    }
    if (zed_slice == nullptr)
        return;

    // Centroid of the RAW camera-frame support points — the pose-INDEPENDENT sensor measurement.
    const std::size_t b = std::min<std::size_t>(zed_slice->support_begin, packet.support_points_cam.size());
    const std::size_t e = std::min<std::size_t>(zed_slice->support_end,   packet.support_points_cam.size());
    if (e <= b)
        return;
    Eigen::Vector3d c_cam = Eigen::Vector3d::Zero();
    for (std::size_t k = b; k < e; ++k)
        c_cam += packet.support_points_cam[k].cast<double>();
    c_cam /= static_cast<double>(e - b);

    // STATIC camera→base extrinsic (ts=0, rigid mount) — carries NO localization pose ⇒ z_o ⟂ robot pose.
    const auto m = inner_eigen_->get_transformation_matrix(obs_robot_frame_, obs_cam_frame_, 0);
    if (not m.has_value())
        return;
    const Mat::Vector3d p = m.value() * c_cam;
    inst.obs_robot       = {static_cast<float>(p.x()), static_cast<float>(p.y()), 0.0f};  // position-only

    // Anisotropic measurement covariance R_o (body frame): loose along the viewing ray (near-face/partial-
    // view depth bias), tight ⊥. Ray origin = camera optical centre in body = the extrinsic translation.
    // box_yaw in body = belief yaw (room) + yaw(body←room). See COUNTABLE_TRIANGULATION.md.
    {
        const Mat::Vector3d cam_o = m.value() * Mat::Vector3d(0, 0, 0);
        const Eigen::Vector2f cam_xy{static_cast<float>(cam_o.x()), static_cast<float>(cam_o.y())};
        const Eigen::Vector2f cen_xy{inst.obs_robot.x(), inst.obs_robot.y()};
        const auto bs = inst.ai2_belief.state();
        float box_yaw_body = bs.yaw;
        if (const auto br = inner_eigen_->get_transformation_matrix(obs_robot_frame_, rc::room::current_room_frame(*G_), 0); br.has_value())
            box_yaw_body = bs.yaw + static_cast<float>(std::atan2(br.value().linear()(1, 0),
                                                                  br.value().linear()(0, 0)));
        inst.obs_robot_cov = rc::object_anchor::ray_anisotropic_cov(
            cen_xy, cam_xy, {0.5f * bs.w, 0.5f * bs.h}, box_yaw_body,
            std::max(0.1f, inst.last_range), std::clamp(inst.dbg_obliquity_cos, 0.f, 1.f),
            rc::object_anchor::RayCovParams{});
    }
    inst.obs_robot_valid = true;
}

// Compute the localization/chain covariance term (J·Σ_chain·Jᵀ) at the bed centre; store it on the instance.
//
// Transform the centre to the measurement frame and back to room with ZERO input cov, so InnerGaussianAPI
// returns exactly the chain contribution (Σ_chain from each RT edge's rt_covariance), pinned to the mask
// capture stamp. The bed is fit in room but its position stays conditional on the robot pose
// (camera→robot→room), so this per-frame SHARED localization error feeds the belief common-mode.
void CountableFitter::compute_chain_cov(CountableInstance& inst)
{
    inst.chain_cov_xx = 0.0f;
    inst.chain_cov_yy = 0.0f;
    if (not chain_cov_enabled_ or not gaussian_ or not inner_eigen_)
        return;
    // Localization/chain term J·Σ_chain·Jᵀ at the bed centre: transform it to the measurement frame,
    // then back to room with ZERO input cov — InnerGaussianAPI returns exactly the chain contribution
    // (Σ_chain from each RT edge's rt_covariance), pinned to the mask capture stamp. The bed is fit in
    // room but its position is still conditional on the robot pose (camera→robot→room), so this applies.
    const auto& s = inst.model.state();
    const Mat::Vector3d centre(s.cx, s.cy, 0.0);   // bed node origin = base on the floor (z=0)
    const auto c_src = inner_eigen_->transform(chain_src_frame_, centre, rc::room::current_room_frame(*G_), inst.last_mask_timestamp_ms);
    if (not c_src.has_value())
        return;
    DSR::GaussianPoint3D gp;
    gp.mean = c_src.value();
    gp.covariance = DSR::Cov3d::Zero();
    const auto g = gaussian_->transform_point(rc::room::current_room_frame(*G_), gp, chain_src_frame_, inst.last_mask_timestamp_ms);
    if (not g.has_value())
        return;
    inst.chain_cov_xx = static_cast<float>(g->covariance(0, 0));
    inst.chain_cov_yy = static_cast<float>(g->covariance(1, 1));
}

// ─── Instance lifecycle ───────────────────────────────────────────────────────────────────────────

// Create the instance for a "countable_*" node if absent; return true only the first time it is created.
//
// Warm-starts from the node's size attribs and the room→bed RT edge, then OVERRIDES with the tracker
// birth seed when present (the RT edge written at birth is not reliably queryable this same cycle — it reads
// 0,0 — which would freeze the model at the origin and cause endless re-births). Sanitizes any non-finite
// field before it can poison the SDF and lock the optimizer.
bool CountableFitter::ensure_instance(const DSR::Node& node, std::uint64_t room_id)
{
    room_node_id_ = room_id;
    if (instances_.count(node.id()))
        return false;

    CountableState init_state;
    init_state.cx  = 0.0f;
    init_state.cy  = 0.0f;
    init_state.yaw = 0.0f;

    if (auto v = G_->get_attrib_by_name<width_m_att> (node); v.has_value()) init_state.w            = v.value();
    if (auto v = G_->get_attrib_by_name<depth_m_att> (node); v.has_value()) init_state.h            = v.value();
    if (auto v = G_->get_attrib_by_name<height_m_att>(node); v.has_value()) init_state.countable_height = v.value();

    // Read RT pose from room→bed edge
    if (room_node_id_ != 0)
    {
        if (const auto edge = G_->get_edge(room_node_id_, node.id(), "RT"); edge.has_value())
        {
            if (const auto tr = G_->get_attrib_by_name<rt_translation_att>(edge.value()); tr.has_value())
            {
                const auto& tvec = tr.value().get();
                if (tvec.size() >= 2) { init_state.cx = tvec[0]; init_state.cy = tvec[1]; }
            }
            if (const auto rot = G_->get_attrib_by_name<rt_rotation_euler_xyz_att>(edge.value()); rot.has_value())
            {
                const auto& rvec = rot.value().get();
                if (rvec.size() >= 3) init_state.yaw = rvec[2];
            }
        }
    }

    // Tracker birth seed: authoritative for a freshly-born instance. The room→bed RT edge written at
    // birth is not reliably queryable this same cycle (it reads as 0,0), and the warm-start would then
    // freeze the model at the origin forever → the tracker never associates and re-births endlessly.
    // Only the XY is seeded here. All cold-start GEOMETRY (centre, height, footprint, yaw) is owned by
    // run_inference's lazy snap, which overwrites it from the observed cloud on the first frame with points —
    // anything set here beyond xy would simply be discarded one frame later.
    if (auto it = birth_seeds_.find(node.id()); it != birth_seeds_.end())
    {
        init_state.cx = it->second.x();
        init_state.cy = it->second.y();
        std::print("[{}] birth-seed applied → cx={:.2f} cy={:.2f}\n", node.name(), init_state.cx, init_state.cy);
        birth_seeds_.erase(it);
    }
    else
        std::print("[{}] NO birth-seed (id={}) → init cx={:.2f} cy={:.2f}\n",
                   node.name(), node.id(), init_state.cx, init_state.cy);

    init_state.leg_length = std::max(0.05f, init_state.countable_height - CountableModel::TOP_THICKNESS);

    // Sanitize: a NaN/Inf from a corrupted RT edge would poison the SDF and lock the
    // optimizer; replace any non-finite field with a safe default before it reaches the model.
    {
        const auto fix = [&](float& v, float fallback, const char* name)
        {
            if (!std::isfinite(v))
            {
                std::print("{}: WARNING non-finite {} for '{}' → reset to {:.3f}\n", cfg_.agent_name, name, node.name(), fallback);
                v = fallback;
            }
        };
        fix(init_state.cx, 0.0f, "cx");
        fix(init_state.cy, 0.0f, "cy");
        fix(init_state.yaw, 0.0f, "yaw");
        fix(init_state.w, 0.60f, "w");                          // standard bed footprint ≈ 0.60×0.60
        fix(init_state.h, 0.60f, "h");
        fix(init_state.countable_height, 1.70f, "countable_height");
        fix(init_state.leg_inset, CountableModel::LEG_RADIUS, "leg_inset");  // FROZEN at outer edge
        init_state.leg_length = std::max(0.05f, init_state.countable_height - CountableModel::TOP_THICKNESS);
    }

    CountableInstance inst;
    inst.node_id   = node.id();
    inst.node_name = node.name();

    inst.model = CountableModel(init_state, make_model_params());
    inst.affordance.init(G_, node.id(), node.name(), cfg_.concept_class);
    inst.subtype = cfg_.concept_class;
    inst.class_belief = rc::exist::ExistenceBelief(rc::countable::logit(cfg_.class_p_label_true), cfg_.existence_logodds_max);   // L_cls starts at logit(π_cls)

    instances_.emplace(node.id(), std::move(inst));
    std::print("{}: created instance for node '{}' id={}\n", cfg_.agent_name, node.name(), node.id());
    return true;
}

// ─── Observation ──────────────────────────────────────────────────────────────────────────────────

// Build an observation from ONE assigned mask slice: latch that slice's R inputs, then SDF-split its cloud.
//
// Classify-don't-destroy split into candidate (near the current surface) vs residual (off-surface) points.
// The caller (process_countable_node) invokes this once per assigned slice and runs a belief update for each —
// sequential fusion that keeps every sensor's R and common-mode separate. RICOH slices (depth_var>0) are
// BEARING-ONLY and return empty (never fitted); they drive only the attention path.
CountableFitter::CountableObservation CountableFitter::observe_slice(CountableInstance& inst, int slice_index)
{
    CountableObservation observation;
    const auto& masks_packet = mask_ingestor_->packet();
    if (not masks_packet.valid or slice_index < 0
        or slice_index >= static_cast<int>(masks_packet.slices.size()))
        return observation;

    const auto& slice = masks_packet.slices[slice_index];
    // RICOH (depth_var>0) is BEARING-ONLY now: a peripheral 360 detection has a reliable DIRECTION but a biased
    // centroid/extent (partial oblique view), so it must NEVER touch pose/extent/birth — it drove duplicate
    // births + drift/inflation. It is handled entirely by the attention path (process_ricoh_bearings): an
    // unassigned ricoh bearing raises attention so the robot seeks a good ZED view. Return empty here (never
    // fitted). Defensive — run_instance_tracker also skips ricoh from assignment, so this normally isn't reached.
    if (slice.depth_var > 0.0f)
        return observation;   // has_fresh_data=false ⇒ no fit
    observation.slice_index = slice_index;   // its pixels attribute the LiDAR returns (lidar-measures G2)
    // A slice assigned to this bed ⇒ detection is alive; latch its per-slice R inputs.
    inst.frames_since_detection = 0;
    inst.last_mask_confidence   = slice.confidence;
    inst.dbg_pkt_frame_id       = masks_packet.frame_id;       // provenance of THIS fit (instrumentation)
    inst.dbg_pkt_ts_ms          = masks_packet.timestamp_ms;
    inst.last_mask_timestamp_ms = masks_packet.timestamp_ms;   // chain-cov pinning (Part B)
    // Ego-motion capture-corruption + depth uncertainty for THIS mask (AI2 obs precision / bias gate; 0 if the
    // producer predates the feature, or for a dense zed slice). See MASK_MOTION_CORRUPTION.md / depth_projection.
    inst.last_motion_var      = slice.motion_var;
    inst.last_motion_dotd     = slice.motion_dotd;
    inst.last_trunc_frac      = slice.trunc_frac;
    inst.last_centroid_radius = slice.centroid_radius;
    inst.last_range           = slice.range;
    inst.last_depth_var       = slice.depth_var;   // mask depth uncertainty → R (ricoh lidar-depth masks)

    const std::size_t begin = std::min(slice.support_begin, masks_packet.support_points.size());
    const std::size_t end   = std::min(slice.support_end,   masks_packet.support_points.size());

    observation.candidate_pts.reserve(end > begin ? end - begin : 0);
    observation.residual_pts.reserve(end > begin ? end - begin : 0);
    for (std::size_t i = begin; i < end; ++i)
    {
        const auto& p = masks_packet.support_points[i];
        // ★A POINT ANOTHER OBJECT ALREADY EXPLAINS IS NOT EVIDENCE FOR THIS ONE (SHARED, common/exclusion).
        // Occam, stated locally: growing this object's extent to cover it buys no likelihood and costs
        // complexity, so the fit stops there of its own accord — no clamp, no arbitration, and an ABUTTING
        // neighbour pays nothing because it does not overlap. The point's z is part of the question: a claim
        // is a VOLUME, so a hood over a worktop shares its footprint without sharing its space.
        // Measured 2026-08-17: bottle_2 walked its radius to 4.51 m, converging on a flat disc at worktop
        // height — the worktop's own returns, admitted as evidence of bottle.
        if (foreign_claims_ and rc::exclusion::explained_by_other(p.x(), p.y(), p.z(), *foreign_claims_))
        {
            // NOT gated on a verbose flag: this is the one line that says whether the rule is running at
            // all, which is the first thing anyone asks. Throttled rather than silenced.
            if ((n_explained_away_++ % 200) == 0)
                std::print("[{}] [exclusion] point dropped — already explained by another object "
                           "({} so far this run)\n", inst.node_name, n_explained_away_);
            continue;
        }
        const float sdf = inst.model.sdf_point(p);
        if (std::abs(sdf) < cfg_.sdf_threshold_for_storage)
            observation.candidate_pts.push_back(p);
        else
            observation.residual_pts.push_back(p);
    }

    const float total = static_cast<float>(observation.candidate_pts.size() + observation.residual_pts.size());
    observation.has_fresh_data   = total > 0.0f;

    if (should_log(inst))
        std::print("[{}] masks={} slice={} label='{}' conf={:.2f} dvar={:.4f} support={} cand={} resid={} centroid=({:.2f},{:.2f},{:.2f})\n",
                   inst.node_name, masks_packet.frame_id, slice_index, slice.label, slice.confidence, slice.depth_var,
                   end - begin, observation.candidate_pts.size(), observation.residual_pts.size(),
                   slice.centroid.x(), slice.centroid.y(), slice.centroid.z());
    return observation;
}

// Fallback when the tracker assigned NO mask slice this cycle: always a stale observation (age the belief).
CountableFitter::CountableObservation CountableFitter::observe(CountableInstance& inst, const DSR::Node& node)
{
    // No mask slice was assigned this cycle. There is no node-attrib sensing path (nothing writes
    // candidate/residual points onto the node), so this is always a stale observation: has_fresh_data=false
    // ⇒ run_inference ages the belief (predict-only) instead of fitting.
    (void)inst; (void)node;
    return CountableObservation{};
}

// True on the config-driven log period (one in countable_log_period_frames cycles).
bool CountableFitter::should_log(const CountableInstance& inst) const
{
    const int period = std::max(1, cfg_.countable_log_period_frames);
    return (inst.processed_cycles % period) == 0;
}

// ─── Room walls (wall-flush factor; simplified from cabinet_concept — no run/seg/collinear-merge) ─────

// Nearest room-polygon edge to a room-frame point q, as a WallRef {foot on the edge, UNIT normal pointing
// INTO the room, wall sigma}. Returns ok=false when no room polygon is known ⇒ the wall factor is inert
// (the bed is treated as free-standing). The inward normal is the one that points toward room_interior_.
// The room's principal axis: the length-weighted circular mean of the edge directions on the 4-fold circle
// (4θ), so walls at 0°, 90°, 180°, 270° all vote for the same axis. nullopt without a polygon.
std::optional<float> CountableFitter::room_axis() const
{
    const std::size_t n = room_polygon_.size();
    if (not have_room_geometry_ or n < 3) return std::nullopt;
    double sx = 0.0, sy = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        const Eigen::Vector2f e = room_polygon_[(i + 1) % n] - room_polygon_[i];
        const double len = e.norm(), th = std::atan2(e.y(), e.x());
        sx += len * std::cos(4.0 * th); sy += len * std::sin(4.0 * th);
    }
    if (sx * sx + sy * sy < 1e-12) return std::nullopt;
    return static_cast<float>(0.25 * std::atan2(sy, sx));
}

// BIRTH ON THE PRIORS. A wall-standing object's first view is its FRONT; the principal-axis yaw of that view is
// noise on a near-square footprint and the depth is unseen. So the birth takes what the priors already know:
//  · the wall: the one nearest the cloud (the class prior says it stands against one — WallFlushPrior);
//  · yaw: back face onto that wall, front into the room (local +Y = back = −n);
//  · width/depth: the cloud's spread ALONG and ACROSS that wall (p5–p95), depth from the prior unless the
//    cloud really spans it (the same band the depth-observability prior uses);
//  · centre: the cloud's middle along the wall, half a depth in from it.
// Not a gate on the fit: the belief starts here and the data then moves it (an object that is genuinely away
// from the wall loses the flush weight as its pose tightens — see flush_weight).
void CountableFitter::birth_on_priors(CountableBeliefState& s0, const std::vector<Eigen::Vector3f>& pts, float z_lo, float z_hi,
                                const std::string& name) const
{
    std::vector<Eigen::Vector2f> xy;
    for (const auto& p : pts) if (p.z() >= z_lo and p.z() <= z_hi) xy.emplace_back(p.x(), p.y());
    if (xy.size() < 10) return;
    Eigen::Vector2f c = Eigen::Vector2f::Zero();
    for (const auto& q : xy) c += q;
    c /= static_cast<float>(xy.size());
    const WallRef w = nearest_wall(c);
    if (not w.ok) return;
    const Eigen::Vector2f n = w.n, t(-n.y(), n.x());          // into the room, and along the wall
    std::vector<float> a, b;
    for (const auto& q : xy) { a.push_back((q - w.p).dot(t)); b.push_back((q - w.p).dot(n)); }
    const auto pct = [](std::vector<float> v, float f)
    { const std::size_t k = static_cast<std::size_t>(f * (v.size() - 1)); std::nth_element(v.begin(), v.begin() + k, v.end()); return v[k]; };
    const float a0 = pct(a, 0.05f), a1 = pct(a, 0.95f), b0 = pct(b, 0.05f), b1 = pct(b, 0.95f);
    const float prior_d = cfg_.ai2_prior_depth_m > 0.0f ? cfg_.ai2_prior_depth_m : cfg_.ai2_prior_footprint_m;
    const float across  = b1 - b0;
    s0.w   = std::max(0.10f, a1 - a0);
    s0.h   = across > cfg_.ai2_depth_obs_band_m ? std::max(across, b1) : prior_d;   // a seen side reaches the wall
    s0.yaw = std::atan2(n.x(), -n.y());                       // R(yaw)·ŷ = −n: the back faces the wall
    const Eigen::Vector2f ctr = w.p + t * (0.5f * (a0 + a1)) + n * (0.5f * s0.h);
    std::print("[{}] birth on priors: wall n=({:.2f},{:.2f}) at {:.2f} m from the cloud → yaw={:.1f}° "
               "w={:.2f} h={:.2f} centre ({:.2f},{:.2f}) [cloud centre ({:.2f},{:.2f})]\n",
               name, n.x(), n.y(), 0.5f * (b0 + b1), s0.yaw * 57.29578f, s0.w, s0.h, ctr.x(), ctr.y(), c.x(), c.y());
    s0.cx = ctr.x(); s0.cy = ctr.y();
}

WallRef CountableFitter::nearest_wall(const Eigen::Vector2f& q) const
{
    WallRef w;
    const std::size_t n = room_polygon_.size();
    if (not have_room_geometry_ or n < 2) return w;   // no model ⇒ inert ⇒ free-standing
    float best = std::numeric_limits<float>::max();
    std::size_t best_i = n;
    Eigen::Vector2f best_foot = Eigen::Vector2f::Zero();
    for (std::size_t i = 0; i < n; ++i)
    {
        const Eigen::Vector2f& a = room_polygon_[i];
        const Eigen::Vector2f  ab = room_polygon_[(i + 1) % n] - a;
        const float len2 = ab.squaredNorm();
        if (len2 < 1e-8f) continue;
        const float t = std::clamp((q - a).dot(ab) / len2, 0.0f, 1.0f);
        const Eigen::Vector2f foot = a + t * ab;
        const float dist2 = (q - foot).squaredNorm();
        if (dist2 < best) { best = dist2; best_i = i; best_foot = foot; }
    }
    if (best_i >= n) return w;
    const Eigen::Vector2f& a = room_polygon_[best_i];
    const Eigen::Vector2f  ab = room_polygon_[(best_i + 1) % n] - a;
    Eigen::Vector2f nrm(-ab.y(), ab.x());
    nrm.normalize();
    if (nrm.dot(room_interior_ - best_foot) < 0.0f) nrm = -nrm;   // point INTO the room
    w.ok = true; w.p = best_foot; w.n = nrm; w.sigma_m = 0.02f;
    w.length_m = ab.norm();   // the segment's length ⇒ how well its DIRECTION is known (see WallRef)
    return w;
}

// Continuous interior support at q (see the header). Even-odd containment test against the room polygon, then a
// smooth falloff over one bed half-depth measured from the boundary — so a bed legitimately pushed against
// a wall keeps most of its support, while a detection at or past the boundary has none. No hard cutoff.
float CountableFitter::interior_support(const Eigen::Vector2f& q) const
{
    const std::size_t n = room_polygon_.size();
    if (not have_room_geometry_ or n < 3) return 1.0f;   // no room model ⇒ the room asserts nothing
    bool inside = false;
    for (std::size_t i = 0, j = n - 1; i < n; j = i++)
    {
        const Eigen::Vector2f& a = room_polygon_[i];
        const Eigen::Vector2f& b = room_polygon_[j];
        if ((a.y() > q.y()) != (b.y() > q.y()) and
            q.x() < (b.x() - a.x()) * (q.y() - a.y()) / (b.y() - a.y() + 1e-12f) + a.x())
            inside = not inside;
    }
    if (not inside) return 0.0f;                          // outside the layout: the room explains it, not a bed
    const auto w = nearest_wall(q);
    if (not w.ok) return 1.0f;
    const float d = (q - w.p).norm();                     // distance in from the boundary
    // Reuse the flush factor's own scale: within AI2WallReachM of the boundary you are effectively AT the wall
    // (that is precisely what that constant means), so no new knob. A real bed's mask centroid sits well
    // beyond it — a front-only view is ~a depth in — and keeps ~full support.
    const float reach = std::max(1e-3f, cfg_.ai2_wall_reach_m);
    return 1.0f - std::exp(-(d * d) / (2.0f * reach * reach));
}

// ─── Inference ────────────────────────────────────────────────────────────────────────────────────

// One recursive full-covariance belief update (or age-only step) for this instance; returns the free energy.
//
// Lazy first-frame init (snap centre/height to the cloud, footprint-moment birth of w/h/yaw), support-bank
// ingest, then the range/motion covariance and the CountableBelief update guarded by a step-bound divergence net,
// FE-surprise attention baseline, and orientation-mode resolution. On a stale frame it ages the belief
// (Σ grows on the agent's clock) instead of freezing. Result is written back into the legacy CountableState.
float CountableFitter::run_inference(CountableInstance& inst, const CountableObservation& observation)
{
    const int npts = static_cast<int>(observation.candidate_pts.size() + observation.residual_pts.size());

    // Lazy init: warm-start the belief from the model state, but on the FIRST frame snap the centre/height
    // to the observed cloud — a box far from the points would see them all as clutter (zero gradient) and
    // never converge (the AI2 analogue of the legacy cold-start snap).
    if (not inst.ai2_initialized)
    {
        const auto& m = inst.model.state();
        CountableBeliefState s0{m.cx, m.cy, m.countable_height, m.w, m.h, m.yaw};

        // The cold-start cloud is THIS FRAME's mask only.
        //
        // ★Do not union the candidate's probation burst in here. It was tried (common/birth_fragment) and
        // measurably made birth WORSE: unioning ~8 frames of room-frame points spreads them by the robot-pose
        // error accumulated over the window, and the footprint moment reads that spread as extent. Live A/B on
        // the same bed — single frame born at w=0.669 h=0.600, already the converged values; burst union
        // born at w=0.915 (+37%) h=0.494, then ~255 cycles crawling back to w=0.673 h=0.626. The give-away is
        // that ext_MAJOR inflated too: a genuine second face lifts only ext_minor, blur stretches both. This is
        // Khronos's spatio-temporal local consistency (arXiv:2402.13817 eq. 8) failing over a birth window —
        // and at BirthFrames=8 / ~10 Hz the robot cannot reach a second face anyway, so the union can only ever
        // be the same face smeared. See [[bed-birth-burst-smear]].
        if (npts > 0)
        {
            Eigen::Vector3f sum = Eigen::Vector3f::Zero();
            std::vector<float> zs; zs.reserve(npts);
            const auto scan = [&](const std::vector<Eigen::Vector3f>& v)
            { for (const auto& p : v) { sum += p; zs.push_back(p.z()); } };
            scan(observation.candidate_pts); scan(observation.residual_pts);
            s0.cx = sum.x() / npts; s0.cy = sum.y() / npts;
            const std::size_t k = static_cast<std::size_t>(0.95f * (zs.size() - 1));
            std::nth_element(zs.begin(), zs.begin() + k, zs.end());
            s0.H = zs[k];   // observed top face

            // Birth seed (w, yaw) from the footprint second-moment of the BODY cloud: start the box near the
            // mask's real width and orientation so the per-point mixture associates immediately.
            //
            // ★The band is the whole box, NOT a top slab. The table-concept original sampled
            // [H − TOP_THICKNESS − 3σ, H + 3σ] — for a table that IS the tabletop (the full footprint), but for a
            // solid bed it is a ~9 cm sliver at 1.9 m seen grazing, so its minor extent collapsed and line
            // `h = max(0.10, ext_minor)` birthed EVERY bed as a 10 cm-deep slab (live: every birth in
            // etc/ai2_log.csv starts h=0.10). See [[bed-table-geometry-churn]].
            //
            // DEPTH (h) is not seeded from a single view at all: one camera pose sees ONE vertical face, whose
            // footprint projects to a line, so ext_minor measures mask noise rather than the bed's depth. Depth
            // is UNOBSERVED until the cloud spans it — exactly the condition the fit's depth-observability prior
            // already tests (ai2_depth_obs_band_m). Below that span h stays at the footprint prior mean and the
            // AI2DepthUnobsPrecision term holds it there; above it the view genuinely resolves depth, so use it.
            // Skip on a TRIMMED mask (extent is only a lower bound → would birth too small); wait for a full mask.
            if (cfg_.footprint_moment_precision > 0.0f and inst.last_trunc_frac <= 1e-3f)
            {
                std::vector<Eigen::Vector3f> pts;
                pts.reserve(static_cast<std::size_t>(npts));
                pts.insert(pts.end(), observation.candidate_pts.begin(), observation.candidate_pts.end());
                pts.insert(pts.end(), observation.residual_pts.begin(), observation.residual_pts.end());
                // Body band: the full observed height minus the floor-contact and above-top over-segmentation
                // tails, which are cloud ARTEFACTS (not bed surface) and would bias the footprint outward.
                const float z_hi = s0.H + 3.0f * cfg_.ai2_sigma_base_m;
                const float z_lo = 3.0f * cfg_.ai2_sigma_base_m;
                if (const auto mom = CountableBelief::footprint_moment(pts, z_lo, z_hi); mom.ok)
                {
                    s0.cx = mom.cx; s0.cy = mom.cy;
                    s0.w = std::max(0.10f, mom.ext_major);          // major axis = the seen face's width (reliable)
                    s0.h = (mom.ext_minor > cfg_.ai2_depth_obs_band_m) ? mom.ext_minor
                                                                       : (cfg_.ai2_prior_depth_m > 0.0f
                                                                              ? cfg_.ai2_prior_depth_m
                                                                              : cfg_.ai2_prior_footprint_m);
                    // Expected: "prior". From ONE camera pose the footprint is a line, so ext_minor is mask
                    // noise (live: ~0.03 against a 0.10 band) and depth correctly defers to the prior. A birth
                    // reporting DEPTH RESOLVED from a single frame means the cloud is not one face — check
                    // ext_major against the bed's real width before believing it.
                    std::print("[{}] cold-start footprint: {} pts → ext_major={:.3f} ext_minor={:.3f} "
                               "(band={:.3f}) → h={:.3f} [{}]\n",
                               inst.node_name, npts, mom.ext_major, mom.ext_minor,
                               cfg_.ai2_depth_obs_band_m, s0.h,
                               mom.ext_minor > cfg_.ai2_depth_obs_band_m ? "DEPTH RESOLVED" : "prior");
                    // Only commit the orientation when the footprint is clearly anisotropic. A near-square
                    // footprint has an ill-defined principal axis (phi swings ±90° on noise), so birthing yaw
                    // from a single frame would seed a random ±90°; leave yaw at its RT/default seed and let
                    // resolve_orientation settle it once evidence accrues.
                    const float sum = mom.ext_major + mom.ext_minor;
                    if (sum > 1e-4f and (mom.ext_major - mom.ext_minor) / sum > 0.10f)
                        s0.yaw = mom.phi;
                    // ★BORN ON THE FURNITURE PRIORS (owner 2026-11-14): room-axis yaw, back on the nearest wall.
                    if (have_room_geometry_) birth_on_priors(s0, pts, z_lo, z_hi, inst.node_name);
                }
            }
        }
        CountableBeliefParams p;
        p.sigma_base_m    = cfg_.ai2_sigma_base_m;
        p.clutter_frac    = cfg_.ai2_clutter_frac;
        p.clutter_scale_m = cfg_.ai2_clutter_scale_m;
        p.prior_size_std  = cfg_.ai2_prior_size_std;
        p.prior_footprint_m   = cfg_.ai2_prior_footprint_m;
        p.prior_footprint_std = cfg_.ai2_prior_footprint_std;
        p.prior_depth_m       = cfg_.ai2_prior_depth_m;
        p.wall_flush_prior    = cfg_.ai2_wall_flush_prior;
        p.room_axis_std       = cfg_.ai2_room_axis_std_deg * 0.017453293f;
        p.headboard_h_m       = cfg_.ai2_headboard_h_m;
        p.headboard_t_m       = cfg_.ai2_headboard_t_m;
        p.prior_height_m      = cfg_.ai2_prior_height_m;
        p.prior_height_std    = cfg_.ai2_prior_height_std;
        p.depth_unobs_precision = cfg_.ai2_depth_unobs_precision;
        p.depth_obs_band_m      = cfg_.ai2_depth_obs_band_m;
        p.top_no_float_precision = cfg_.ai2_top_no_float_precision;
        p.top_no_float_margin_m  = cfg_.ai2_top_no_float_margin_m;
        p.wall_precision          = cfg_.ai2_wall_precision;
        p.wall_parallel_precision = cfg_.ai2_wall_parallel_precision;
        p.wall_reach_m            = cfg_.ai2_wall_reach_m;
        p.wall_no_cross_precision = cfg_.ai2_wall_no_cross_precision;
        p.wall_no_cross_margin_m  = cfg_.ai2_wall_no_cross_margin_m;
        // The wall may only explain away mask points to the extent the room itself is trusted: with no room
        // polygon pushed this session there is nothing to explain away with, so the component is OFF and the
        // mixture is box+clutter exactly as before. (When room_concept publishes a map-trust scalar, multiply
        // here — a poorly-localized room must return the bed hypothesis its freedom, not silently veto it.)
        p.wall_explain_frac    = have_room_geometry_ ? cfg_.ai2_wall_explain_frac : 0.0f;
        p.wall_explain_sigma_m = cfg_.ai2_wall_explain_sigma_m;
        p.door_clearance_gain  = cfg_.ai2_door_clearance_gain;
        p.volatility_infer     = cfg_.ai2_volatility_infer;
        p.volatility_lr        = cfg_.ai2_volatility_lr;
        p.volatility_sigma     = cfg_.ai2_volatility_sigma;
        p.pixel_sigma_over_f     = cfg_.pixel_sigma_over_f;
        p.depth_sigma0_m         = cfg_.depth_sigma0_m;
        p.depth_sigma_range_coef = cfg_.depth_sigma_range_coef;
        p.model_sigma_m          = cfg_.model_sigma_m;
        p.footprint_residual     = cfg_.footprint_residual;   // a1′+a2′: 2-D footprint residual + shared depth-affine
        p.depth_bias_std         = cfg_.depth_bias_std;
        p.depth_scale_std        = cfg_.depth_scale_std;
        p.process_std_m   = cfg_.ai2_process_std_m;
        p.process_std_yaw = cfg_.ai2_process_std_yaw;
        p.common_mode_pos_std  = cfg_.ai2_common_mode_pos_std;
        p.common_mode_size_std = cfg_.ai2_common_mode_size_std;
        p.common_mode_yaw_std  = cfg_.ai2_common_mode_yaw_std;
        p.gn_iters        = cfg_.ai2_gn_iters;
        p.coverage_precision  = cfg_.coverage_precision;
        p.coverage_robust_c_m = cfg_.coverage_robust_c_m;
        p.free_space_precision = cfg_.free_space_precision;
        p.footprint_moment_precision = cfg_.footprint_moment_precision;
        p.footprint_moment_completeness_gain = cfg_.footprint_moment_completeness_gain;
        p.footprint_moment_min_completeness  = cfg_.footprint_moment_min_completeness;
        p.top_thickness   = CountableModel::TOP_THICKNESS;
        p.leg_radius      = CountableModel::LEG_RADIUS;
        // "Is this really a bed?" plausibility + short-height prior parameters (mis-detection filter).
        p.plaus_aspect_scale      = cfg_.plaus_aspect_scale;
        p.plaus_size_scale        = cfg_.plaus_size_scale;
        p.plaus_height_min        = cfg_.plaus_height_min;
        p.plaus_height_soft       = cfg_.plaus_height_soft;
        p.plaus_fe_ref            = cfg_.plaus_fe_ref;
        p.plaus_fe_scale          = cfg_.plaus_fe_scale;
        p.plaus_height_prior_gain = cfg_.countable_filter_enabled ? cfg_.plaus_height_prior_gain : 0.0f;
        p.persistent = cfg_.lidar_measures_on_gated;   // persistent shared errors + class prior once (spec §7c.1–7c.2)
        inst.ai2_belief = CountableBelief(s0, p);
        inst.ai2_initialized = true;
        std::print("[{}] cold-start state: cx={:.2f} cy={:.2f} H={:.2f} w={:.2f} h={:.2f} yaw={:.2f}\n",
                   inst.node_name, s0.cx, s0.cy, s0.H, s0.w, s0.h, s0.yaw);
    }

    if (observation.has_fresh_data)
    {
        rc::support_bank::ingest(inst, observation.candidate_pts, observation.residual_pts, cfg_);
        // ★NOT for a viewer — nothing ever read the published attribute. The bank is how this agent
        // gathers 3D points from its own masks over many frames so evaluate_shape() can DISCRIMINATE
        // SHAPES (round vs square) from an accumulated cloud rather than one frame's partial view.
        inst.last_residual_pts = observation.residual_pts;   // model-unexplained points for the viewer layer
    }

    // lidar-measures (spec 2026-11-15 §7c.2): the odometry since this instance's last call goes into the viewpoint paths on
    // EVERY call, stale ones included (aged and predict-only cycles add up). Flag off: nothing read, nothing integrated.
    inst.dbg_lidar_only = 0;
    inst.dbg_lidar_msum = 0.0f;
    inst.dbg_rho_cam = 1.0f; inst.dbg_rho_L = 1.0f; inst.dbg_rho_chain = 1.0f;   // neutral: no update ⇒ no ρ applied
    inst.dbg_dphi_cam = 0.0f; inst.dbg_dz_cam = 0.0f;
    inst.dbg_ray_n = 0;
    inst.dbg_itr_cam = 0.0f; inst.dbg_itr_L = 0.0f;
    if (cfg_.lidar_measures_on_gated) integrate_odometry(inst);

    const auto now = std::chrono::steady_clock::now();

    // Freeze-vs-age on stale: no fresh mask this cycle. Historically the belief just froze (Σ held —
    // information-filter axiom), so a dead mask/ZED feed read downstream as a confident-but-stale bed.
    // With AI2AgeNominalDtS>0 we instead AGE it: run the predict-only step on the AGENT's clock so Σ grows
    // by Q·(dt/dt_nominal) for as long as the stream is silent, and the propagated RT covariance reflects
    // how old the evidence is. No gate, no emergency flag — the generative-model form of a stale sensor.
    if (not observation.has_fresh_data)
    {
        if (cfg_.ai2_age_nominal_dt_s > 0.0f and inst.last_belief_touch.time_since_epoch().count() != 0)
        {
            const float dt = std::chrono::duration<float>(now - inst.last_belief_touch).count();
            inst.ai2_belief.inflate_for_age(dt, cfg_.ai2_age_nominal_dt_s);
        }
        inst.last_belief_touch = now;
        inst.dbg_gate_fresh = false;   // no mask reached the fit ⇒ the gate verdict below is STALE this cycle
        projection_->compute_projected_roi(inst);
        return inst.dbg_energy;   // HOLD the last free energy — no new mask ≠ FE 0 (the fit is unchanged)
    }
    // Fresh path: update()/predict() below carry their own one-step Q, so just reset the age clock here.
    inst.last_belief_touch = now;

    // Static range weighting (motion-free). Even at zero camera motion, deprojection noise grows with
    // distance AND a far mask subtends a tiny angle, so pose — orientation most of all — becomes
    // unobservable: a 7 m view should confirm existence but never rotate a converged bed. The
    // motion×distance term is already in last_motion_var (the retina interaction matrix carries 1/Z),
    // but that vanishes when still; this is the missing static part. Pure continuous covariance growth (no
    // gate): the per-frame information CAP (common-mode) rises with range, so the frame's yaw gain against a
    // converged prior shrinks smoothly toward zero. mask_range = mean camera→mask depth Z, from the producer.
    const float range         = std::max(0.0f, inst.last_range);
    const float lat_std       = cfg_.ai2_range_noise_lat_per_m * range;   // m   (lateral deprojection)
    const float yaw_std       = cfg_.ai2_range_noise_yaw_per_m * range;   // rad (orientation lever-arm ∝ 1/range)
    const float size_std      = cfg_.ai2_range_noise_size_per_m * range;  // m   (extent unresolved at range)
    const float range_lat_var = lat_std * lat_std;
    const float range_yaw_var = yaw_std * yaw_std;
    const float range_size_var = size_std * size_std;

    const float R = cfg_.ai2_sigma_base_m * cfg_.ai2_sigma_base_m
                  + std::max(0.0f, inst.last_motion_var) + std::max(0.0f, inst.last_depth_var) + range_lat_var;

    // ── Ego-motion → COMMON-MODE: the "be still to UPDATE, else CONFIRM" fixation term (like the VOR / an
    // animal taking in detail only during fixations). A moving frame's mask is smeared/displaced by ego-motion
    // by ≈ effective-lag · motion-speed — a per-mask SHARED error that per-point R averages away (N points
    // collapse it). Route it into the per-frame COMMON-MODE instead: the engine's Woodbury marginalisation
    // then caps the frame's authority to move the GEOMETRY MEAN (size/pose/yaw), so geometric updates
    // concentrate at stillness while a moving frame contributes CONFIRMATION only (existence/association don't
    // read this). motion_dotd = Z·‖ṡ‖ (m/s) from the retina. CONTINUOUS, no gate: at dotd→0 the term
    // vanishes (a still frame updates fully); the gains are ~effective-lag (s), 0 disables a channel.
    // Use the ROBUST combined motion signal (motion_magnitude = max(|motion_dotd|, ego_lin + lever·ego_ang)),
    // scaled by the off-axis periphery penalty — the SAME signal the discrete confirm_only gate reads, so the
    // continuous common-mode and the discrete gate share one motion measure (matches chair_concept). A still OR
    // well-centred mask ⇒ ~0 common-mode (full authority); moving AND peripheral ⇒ large common-mode (confirm only).
    const float mot_mag     = motion_magnitude(inst);
    const float mot_periph  = periphery_penalty(inst);
    const float mot_pos_var = std::pow(cfg_.motion_cm_pos_gain  * mot_mag, 2.0f) * mot_periph;   // m²  (cx,cy)
    const float mot_size_var= std::pow(cfg_.motion_cm_size_gain * mot_mag, 2.0f) * mot_periph;   // m²  (w,h,H)
    const float mot_yaw_var = std::pow(cfg_.motion_cm_yaw_gain  * mot_mag, 2.0f) * mot_periph;   // rad² (yaw)

    // Gate → CONFIRMATION-ONLY (predict only; age Σ, hold the geometry mean; association ran upstream). Two causes:
    //  (1) TRUNCATION: a mask clipped by the image border has a chopped silhouette that biases the fit.
    //  (2) "BE-STILL-TO-UPDATE" (confirm_only): the robot is MOVING and the mask is off-centre — a moving frame's
    //      mask is a shared smear whose centroid is unreliable, so it may only CONFIRM the bed, not move/reshape
    //      it (a well-centred mask is trusted even while moving — the exception). The continuous common-mode above
    //      is the graceful backstop; this discrete gate concentrates geometry updates at stillness (matches chair).
    const bool gated = inst.last_trunc_frac > cfg_.ai2_trunc_gate_frac or confirm_only(inst);

    // Pose-chain covariance at the bed centre (cx,cy) — the per-frame SHARED localization error. Fed
    // into the belief as part of the common-mode so the frame's information saturates (calibrated σ).
    // Computed once here (before the update) and reused for the published RT cov below.
    compute_chain_cov(inst);

    // ── Rogue-mask yaw instrumentation (NO effect on the fit) ──────────────────────────────────────
    // Snapshot yaw at cycle entry and compute OBLIQUITY — the grazing-view covariate the truncation gate does
    // not measure. A near-horizontal camera→bedtop ray foreshortens the top face, so its 2D footprint (hence
    // yaw/extent) is biased even with a full in-frame mask. |cos(incidence)| = |ray·ẑ|/|ray|: 1 top-down, →0
    // grazing. Completeness + the per-channel yaw split are filled after the update. See the [yaw-jump] log.
    inst.dbg_yaw_pre = inst.ai2_belief.state().yaw;
    inst.dbg_dyaw_points = 0.0f; inst.dbg_dyaw_moment = 0.0f; inst.dbg_dyaw_flip = 0.0f; inst.dbg_completeness = 1.0f;
    Eigen::Vector3f cam_origin_room = Eigen::Vector3f::Zero();   // for the anisotropic per-point R (Stage 1)
    Eigen::Matrix4f zed_T_room = Eigen::Matrix4f::Identity();    // room→zed: transform points to the CAMERA frame
    bool have_cam = false;
    if (const auto rTz = projection_->room_T_zed_matrix(inst.last_mask_timestamp_ms); rTz)
    {
        const Eigen::Vector3d cam = rTz->block<3, 1>(0, 3);   // camera origin in the room frame
        cam_origin_room = cam.cast<float>(); have_cam = true;
        zed_T_room = rTz->inverse().cast<float>();            // for the camera-frame azimuth (tilt-state identifiability)
        const auto& s0b = inst.ai2_belief.state();
        const Eigen::Vector3d ray(s0b.cx - cam.x(), s0b.cy - cam.y(), s0b.H - cam.z());   // camera→bedtop centre
        const double rn = ray.norm();
        inst.dbg_obliquity_cos = (rn > 1e-6) ? static_cast<float>(std::abs(ray.z()) / rn) : 1.0f;
    }

    float energy = inst.dbg_energy;   // default = HOLD last FE (a gated / rejected cycle took no measurement)

    // ── Camera attributes, LiDAR measures (Model.LidarMeasuresOnGated; spec 2026-11-15 §7c). Flag off: membership stays
    //    empty, latent_inputs / stage_persistent are never called, and both branches below run exactly as before. ──
    // The camera's membership m(p) of THIS slice (extrinsic at the mask stamp); empty ⇒ unavailable.
    std::function<float(const Eigen::Vector3f&)> membership;
    // Only with a staged sweep: the membership is consumed by the ray selection alone (final review M4; result-preserving).
    if (cfg_.lidar_measures_on_gated and lidar_channel_.has_sweep())
        membership = build_membership(inst, observation.slice_index, zed_T_room, have_cam, mot_pos_var);
    // The latents' marginals (§7c.2): b_cam = the config floors + range lateral/size + the obliquity yaw cap (today's camera
    // fill WITHOUT chain and ego-motion); b_L = the config floors. e's marginal is Σ_pose, staged by stage_persistent (§7c.7).
    // The ego-motion terms are FRESH every frame (ρ = 0), marginalised inside each channel.
    const auto latent_inputs = [&](float obliquity_yaw_var)
    {
        const float p2 = cfg_.ai2_common_mode_pos_std  * cfg_.ai2_common_mode_pos_std;
        const float s2 = cfg_.ai2_common_mode_size_std * cfg_.ai2_common_mode_size_std;
        const float y2 = cfg_.ai2_common_mode_yaw_std  * cfg_.ai2_common_mode_yaw_std;
        LatentInputs li;
        li.sig2_cam = {p2 + range_lat_var, p2 + range_lat_var, s2 + range_size_var, s2 + range_size_var,
                       s2 + range_size_var, y2 + range_yaw_var + obliquity_yaw_var};
        li.sig2_L   = {p2, p2, s2, s2, s2, y2};
        li.fresh    = {mot_pos_var, mot_pos_var, mot_size_var, mot_size_var, mot_size_var, mot_yaw_var};
        return li;
    };
    const auto rays_usable = [](const CountableFrame& f)
    {
        const auto usable = [](const rc::ai::LidarRays& r) { return not r.endpoints.empty() and r.precision > 0.0f; };
        return usable(f.lidar) or std::ranges::any_of(f.lidar_extra, usable);
    };
    const auto ray_count = [](const CountableFrame& f)
    {
        int n = static_cast<int>(f.lidar.endpoints.size());
        for (const auto& r : f.lidar_extra) n += static_cast<int>(r.endpoints.size());
        return n;
    };
    // After an ACCEPTED flag-on update: the measuring channels' paths restart, and each channel's information is logged.
    const auto record_channels = [&](const CountableFrame& f, bool camera, bool lidar)
    {
        consume_paths(inst, f, camera, lidar);
        const auto itr = [](const Eigen::Matrix<float, 6, 6>& Id, const std::array<float, 6>& fresh)
        {
            Eigen::Matrix<float, 6, 6> Sf = Eigen::Matrix<float, 6, 6>::Zero();
            for (int i = 0; i < 6; ++i) Sf(i, i) = std::max(0.0f, fresh[i]);
            return static_cast<float>(rc::ai::persistent::marginalise_fresh<6>(Id, Eigen::Matrix<float, 6, 1>::Zero(), Sf)
                                          .I.trace());
        };
        inst.dbg_itr_cam = camera ? itr(inst.ai2_belief.dbg_Id_cam(), f.fresh_cam) : 0.0f;
        inst.dbg_itr_L   = lidar  ? itr(inst.ai2_belief.dbg_Id_L(),   f.fresh_L)   : 0.0f;
    };

    // WALL-FLUSH factor: give the interior (picks the BACK face) to the belief and stage the nearest room
    // wall to the bed's BACK-face centre. Inert (ok=false) unless a room polygon was pushed this session.
    const auto stage_room = [&](CountableFrame& frame)
    {
        if (have_room_geometry_)
        {
            inst.ai2_belief.set_room_interior(room_interior_);
            const auto& bs_now = inst.ai2_belief.state();
            frame.wall = nearest_wall(inst.ai2_belief.back_centre(bs_now));
            if (const auto ax = room_axis(); ax) { frame.room_axis_ok = true; frame.room_axis = *ax; }
        }
    };

    // Divergence safety net (mirrors bottle): snapshot state+Σ, run the update, and if the centre teleports
    // beyond a physical bound in one frame (corrupted mask cloud / one-sided LiDAR runaway → the cx=−200m
    // event) REJECT it — restore the snapshot, widen Σ via a predict so the next good frame re-associates,
    // and accrue frames_diverged. A non-finite state is treated the same. 0 disables. NOT a magic gate: a
    // static bed cannot physically move max_step_m in one frame, so such a step is definitionally spurious.
    // Shared by the admissible (camera + rays) and the LiDAR-only update. Returns true ⇔ the update was accepted.
    const auto update_guarded = [&](const CountableFrame& frame) -> bool
    {
        const CountableBelief pre_belief = inst.ai2_belief;   // value copy (state + Σ + prior + flip_evidence)
        const auto&       ps         = pre_belief.state();
        // T6c-1: on the persistent path an instance's FIRST update starts from the birth seed, not a posterior — its move is
        // the class prior correcting the seed, so there is nothing for the step bound to protect (the finite checks stay).
        const bool first_update = pre_belief.params().persistent and not pre_belief.first_update_done();
        energy = inst.ai2_belief.update(frame);
        const auto& ns = inst.ai2_belief.state();
        // Full-state jump: centre (cx,cy,H) AND EXTENT (w,h). A static bed can't grow/shrink its extent by
        // max_step_m in one frame any more than it can teleport — a close-range OVER-SEGMENTED mask (floor/wall
        // points) + the grow-only coverage term used to inflate w→5 m in a couple of frames, unguarded because
        // the old step omitted w,h. Now that blow-up is a definitionally-spurious frame → rejected.
        const float step = std::sqrt((ns.cx - ps.cx) * (ns.cx - ps.cx) + (ns.cy - ps.cy) * (ns.cy - ps.cy)
                                   + (ns.H  - ps.H)  * (ns.H  - ps.H)
                                   + (ns.w  - ps.w)  * (ns.w  - ps.w) + (ns.h - ps.h) * (ns.h - ps.h));
        const bool bad = not (std::isfinite(ns.cx) and std::isfinite(ns.cy) and std::isfinite(ns.H)
                              and std::isfinite(ns.w) and std::isfinite(ns.h) and std::isfinite(ns.yaw));
        if (cfg_.max_step_m > 0.0f and (bad or (not first_update and step > cfg_.max_step_m)))
        {
            std::print("[{}] AI2 step-bound REJECT: state moved {:.2f}m (>{:.2f}){} — outlier frame dropped (w={:.2f} h={:.2f})\n",
                       inst.node_name, step, cfg_.max_step_m, bad ? " [non-finite]" : "", ns.w, ns.h);
            inst.ai2_belief = pre_belief;     // reject the corrupted update (restore state + Σ)
            inst.ai2_belief.predict();        // widen Σ so the next good frame re-associates
            energy = inst.dbg_energy;         // HOLD last FE — a rejected outlier frame took no valid measurement
            ++inst.frames_diverged;
            return false;
        }
        inst.frames_diverged = 0;
        return true;
    };

    if (gated)
    {
        // G1 (lidar-measures): a gated camera cycle with an assigned mask still lets the LiDAR measure. The frame carries the
        // mask-weighted rays only — no camera points, z_top_obs = −1 — plus the room context the one-sided constraints read.
        // The select box is anchored on the slice's mask cloud (candidate ∪ residual), which is NOT fed to the engine (§7b).
        // No membership (flag off, no slice/camera/delivered size) or no usable rays ⇒ exactly today's predict.
        bool lidar_updated = false;
        if (cfg_.lidar_measures_on_gated and membership)
        {
            CountableFrame frame;
            frame.lidar_only = true;
            std::vector<Eigen::Vector3f> anchor;
            anchor.reserve(static_cast<std::size_t>(npts));
            anchor.insert(anchor.end(), observation.candidate_pts.begin(), observation.candidate_pts.end());
            anchor.insert(anchor.end(), observation.residual_pts.begin(), observation.residual_pts.end());
            // §7b: the LiDAR-only frame's common mode = pose chain + ego-motion (no camera deprojection terms). Read only
            // by the per-frame Woodbury path; the persistent path takes the latents' inputs staged below.
            frame.chain_cov_xx   = inst.chain_cov_xx + mot_pos_var;
            frame.chain_cov_yy   = inst.chain_cov_yy + mot_pos_var;
            frame.chain_cov_yaw  = mot_yaw_var;
            frame.chain_cov_size = mot_size_var;
            stage_room(frame);
            stage_persistent(inst, frame, latent_inputs(0.0f));
            lidar_channel_.feed(inst, frame, anchor, membership);
            if (rays_usable(frame))
            {
                lidar_updated = true;
                inst.dbg_lidar_only = 1;
                inst.dbg_ray_n = ray_count(frame);
                if (update_guarded(frame)) record_channels(frame, false, true);
                // Energy, FE baseline, surprise, resolve_orientation and resolve_front are camera quantities: untouched.
                energy = inst.dbg_energy;
            }
        }
        if (not lidar_updated)
            inst.ai2_belief.predict();   // Σ inflates, mean unchanged
    }
    else
    {
        CountableFrame frame;
        frame.points.reserve(static_cast<std::size_t>(npts));
        frame.points.insert(frame.points.end(), observation.candidate_pts.begin(), observation.candidate_pts.end());
        frame.points.insert(frame.points.end(), observation.residual_pts.begin(), observation.residual_pts.end());
        frame.R.assign(frame.points.size(), R);
        // Robust observed cloud TOP (z-p97): the top-no-float factor caps the box top here so a floor-anchored H
        // cannot ratchet ABOVE the mask. p97 (not max) ignores a handful of stray high points.
        if (not frame.points.empty())
        {
            std::vector<float> zt; zt.reserve(frame.points.size());
            // ★Only points the BOX explains better than the headboard: the p97 of a cloud that includes a 1.25 m
            // headboard IS the headboard, and the firm top anchor would pin the mattress height there.
            const auto& sb = inst.ai2_belief.state();
            for (const auto& p : frame.points)
                if (inst.ai2_belief.sdf_box(p, sb) <= inst.ai2_belief.sdf_headboard(p, sb)) zt.push_back(p.z());
            if (zt.empty()) for (const auto& p : frame.points) zt.push_back(p.z());
            const std::size_t kt = static_cast<std::size_t>(0.97f * (zt.size() - 1));
            std::nth_element(zt.begin(), zt.begin() + kt, zt.end());
            frame.z_top_obs = zt[kt];
            // Down-weight points ABOVE the robust top: they are almost surely mask OVER-SEGMENTATION (wall / cabinet
            // above the bed) and would otherwise ratchet the floor-anchored box top up to claim them (covering a
            // point is "free"). Grow their measurement variance with height above the top (R += (σ_per_m·above)²) so
            // they fade from the fit — a covariance keyed on the covariate, not a hard z-clip. The two-sided
            // top-anchor then pins H to the real top unopposed.
            if (cfg_.ai2_top_overseg_sigma_per_m > 0.0f)
                for (std::size_t i = 0; i < frame.points.size(); ++i)
                {
                    const float above = frame.points[i].z() - frame.z_top_obs;
                    // a headboard point is not over-segmentation: the headboard explains it, keep its precision
                    if (above > 0.0f and inst.ai2_belief.sdf_box(frame.points[i], sb)
                                         <= inst.ai2_belief.sdf_headboard(frame.points[i], sb))
                    {
                        const float s = cfg_.ai2_top_overseg_sigma_per_m * above;
                        frame.R[i] += s * s;
                    }
                }
        }
        // Ray geometry (needed by the footprint residual + the anisotropic R later).
        if (cfg_.footprint_residual and have_cam)
        { frame.cam_origin = cam_origin_room; frame.has_rays = true; }
        // Camera-frame azimuth per point (about the optical axis; ZED frame x-right, y-depth) for the N=7 depth-
        // tilt STATE — computed from the TRUE extrinsic, NOT the belief's drifting bed-centre, so a persistent
        // tilt is coherent across viewpoints (the identifiability the tilt estimate needs). See PRECISION_AS_INFO.
        if (cfg_.footprint_residual and have_cam)
        {
            frame.point_azim.reserve(frame.points.size());
            for (const auto& p : frame.points)
            {
                const Eigen::Vector4f pc = zed_T_room * Eigen::Vector4f(p.x(), p.y(), p.z(), 1.0f);
                frame.point_azim.push_back(std::atan2(pc.x(), pc.y()));   // signed azimuth about the optical axis
            }
        }
        frame.chain_cov_xx  = inst.chain_cov_xx + range_lat_var + mot_pos_var;   // pose-chain + range + EGO-MOTION
        frame.chain_cov_yy  = inst.chain_cov_yy + range_lat_var + mot_pos_var;
        // Obliquity yaw cap: at an edge-on (grazing) view the bedtop cloud is ~1-D along the near edge, so yaw
        // is barely observable and the per-point GN snaps between the box's symmetric orientations (r_π / w↔h —
        // the CSV flips). Grow the SHARED yaw variance as the view grazes (|cos(incidence)|→0 ⇒ 1/cos→∞), so a
        // grazing frame confirms the bed but cannot rotate it — the same continuous-covariance form as the range
        // term, keyed on view angle instead of distance. Validated live 2026-07-11 → always on.
        //
        // THIS IS THE LIVE YAW CAP on the shipped (FootprintResidual == false) branch below — a tuned coefficient,
        // deliberately kept. Its intended derived replacement, CountableBelief::tilt_yaw_common_mode() (a2′), is WIP and
        // NOT wired here: the fitter never calls it, and its self_test drift check is disabled. So on the
        // FootprintResidual == true branch there is currently NO obliquity/tilt yaw cap wired in (chain_cov_yaw
        // carries only the white range term) — that branch is experimental. Do not delete kObliquityYawGain until
        // tilt_yaw_common_mode is called here and validated. See countable_belief.cpp::tilt_yaw_common_mode.
        constexpr float kObliquityYawGain = 0.05f;   // ~30° σ_yaw at cos=0.09 → per-point GN holds yaw at grazing
        const float oblq_cos          = std::clamp(inst.dbg_obliquity_cos, 0.05f, 1.0f);
        const float obliquity_yaw_std = kObliquityYawGain * (1.0f / oblq_cos - 1.0f);   // 0 at top-down
        // FootprintResidual on ⇒ the depth tilt is an ESTIMATED N=7 state (no per-frame yaw cap — that ratcheted
        // and needed a fragile sweet-spot); chain_cov_yaw carries only the WHITE range term. Else the legacy tuned
        // obliquity+range form. (The tilt STATE replaces both the cap and the obliquity/range yaw gains.)
        frame.chain_cov_yaw = mot_yaw_var + (cfg_.footprint_residual
            ? range_yaw_var
            : range_yaw_var + obliquity_yaw_std * obliquity_yaw_std);   // range + grazing-view cap + EGO-MOTION
        frame.chain_cov_size = range_size_var + mot_size_var;      // range freezes afar + EGO-MOTION freezes while moving
        // Footprint-moment SHARED per-frame variance: ego-motion mask corruption + a GENTLE range term (a global
        // footprint fit is far more range-robust than a single boundary point, so a coefficient << the per-point
        // AI2RangeNoiseSizePerM). This makes the moment ACCUMULATE across frames and back off when the robot is
        // moving, instead of snapping the belief to each frame's noisy/corrupted footprint (the live wander).
        // The ego-motion term uses motion_dotd (camera→mask relative motion): a "going-away/rotation" frame yields
        // a degraded/split mask, and this inflates the moment variance so it CANNOT reshape the established fit
        // (the observed reshape/rotate came in exactly on those frames). motion_var (interaction-matrix) is often
        // under-reported for split masks, so motion_dotd is the load-bearing signal here.
        const float moment_range_std  = cfg_.footprint_moment_range_per_m * range;
        const float moment_motion_std = cfg_.footprint_moment_motion_gain * std::abs(inst.last_motion_dotd);
        // Obliquity → moment variance: a grazing/foreshortened view biases the 2D inertia tensor (the SAME cause
        // that spoils the per-point yaw, but on the moment channel — where the CSV rogue rotations actually came
        // in). Mirror the per-point obliquity cap onto moment_extra_var so an edge-on frame confirms the bed but
        // cannot rotate/reshape it via the moment. Reuses oblq_cos computed above for chain_cov_yaw. 0 = OFF.
        const float moment_oblq_std   = cfg_.obliquity_moment_gain * (1.0f / oblq_cos - 1.0f);
        frame.moment_extra_var = std::max(0.0f, inst.last_motion_var)
                               + moment_range_std * moment_range_std
                               + moment_motion_std * moment_motion_std
                               + moment_oblq_std * moment_oblq_std;
        // YOLO-independent LiDAR range channel: stage returns landing on the legs/rim. No-op if precision==0
        // or no fresh sweep. The shared factor (accumulate_lidar_rays<6> in CountableBelief::accumulate_extra)
        // sphere-traces this belief's own SDF, so the same call the bottle uses drops in unchanged.
        // Flag on: the persistent latents' inputs (ρ from the odometry paths, marginals, fresh ego-motion) and the camera's
        // membership on the rays (empty ⇒ today's unweighted selection, bit for bit).
        if (cfg_.lidar_measures_on_gated)
            stage_persistent(inst, frame, latent_inputs(cfg_.footprint_residual
                                                         ? 0.0f : obliquity_yaw_std * obliquity_yaw_std));
        lidar_channel_.feed(inst, frame, frame.points, membership);
        inst.dbg_ray_n = ray_count(frame);

        // WALL-FLUSH factor (see stage_room above).
        stage_room(frame);

        // Stage 1 (PRECISION_AS_INFORMATION.md): replace the scalar per-point R with the anisotropic deprojection
        // noise projected on the SDF normal. Ego-motion variance is preserved as an isotropic floor (Stage 2 will
        // subsume it into the nuisance Jacobian). No-op unless the flag is on and we know the camera origin — then
        // a grazing view's yaw-carrying points get huge R → the per-point GN cannot rotate a converged bed,
        // WITHOUT the obliquity/range yaw gains above (which this is designed to make redundant).
        // (the footprint residual runs inside ai2_belief.update via accumulate_footprint; its tilt→yaw common-mode
        //  was folded into frame.chain_cov_yaw above so the engine Woodbury caps the total yaw information)

        // Divergence safety net: see update_guarded above (snapshot, update, step-bound reject).
        if (update_guarded(frame))
        {
            if (cfg_.lidar_measures_on_gated) record_channels(frame, not frame.points.empty(), rays_usable(frame));
            // Log the TRUE free energy (mean −log mixture likelihood, clutter included) at the converged state —
            // the engine's return zeroed misfit via clutter (read ≈0 even on a bad fit). This is a meaningful F.
            energy = inst.ai2_belief.mean_energy(frame.points, inst.ai2_belief.state(), R);
            // FE SURPRISE (attention trigger): baseline tracks DOWN fast (consolidate a better fit) / UP slow (a
            // sustained rise = the bed moved shows as surprise before the baseline accepts it); surprise = the
            // smoothed positive gap FE−baseline. Updated only on a genuine measurement (this accepted branch).
            if (inst.fe_baseline < 0.0f)
                inst.fe_baseline = energy;
            else
            {
                const float a = (energy < inst.fe_baseline) ? cfg_.fe_baseline_adapt_down
                                                            : cfg_.fe_baseline_adapt_up;
                inst.fe_baseline += a * (energy - inst.fe_baseline);
            }
            const float gap = std::max(0.0f, energy - inst.fe_baseline);
            inst.fe_surprise += cfg_.fe_surprise_smooth * (gap - inst.fe_surprise);

            // "IS THIS REALLY A COUNTABLE?" plausibility is now judged EVERY cycle from the CURRENT fitted shape in
            // the worker's singleton/existence step (specificworker_lifecycle.cpp), NOT only here on an accepted
            // mask-update. Reason: a mis-detection that has diverged to a cabinet shape (w≫h, short) then coasts
            // out-of-FoV would otherwise FREEZE its birth-time positive evidence and stay immortal. Judging the
            // shape per-cycle lets it keep losing plausibility so the singleton decays it. (countable_plausibility is
            // shape-only — aspect·size·height, FE unused — so it needs no fresh mask.)
        }
        // Near-square yaw disambiguation: sequential Bayesian comparison of the two orientation modes
        // (current vs the w↔h swap ≡ 90° rotation). Owns the GENUINE mode flip so the per-frame MAP no
        // longer SNAPS 90° on extent noise; the reported yaw uncertainty (yaw_marginal_var) stays honest
        // until an orbit resolves it. See COUNTABLE.md.
        // Ego-motion reliability: a moving frame's mask must barely vote on the discrete w↔h mode (split/degraded
        // masks during motion were flipping it). Continuous down-weight, not a gate. Static (dotd≈0) → weight 1.
        const float mref = std::max(1e-3f, cfg_.orientation_motion_ref);
        const float dotd = std::abs(inst.last_motion_dotd);
        const float mode_evidence_weight = 1.0f / (1.0f + (dotd / mref) * (dotd / mref));
        // The quotient chart makes the w↔h mode a continuous, single-valued coordinate — the discrete flip
        // accumulator is unrepresentable and unnecessary (skipped). Legacy path keeps resolve_orientation.
        if (not rc::CountableBeliefState::use_quotient
            and inst.ai2_belief.resolve_orientation(frame.points, R, mode_evidence_weight) and should_log(inst))
            std::print("[{}] orientation mode FLIP (w↔h) | flip_ev={:.3f} p_alt={:.2f}\n",
                       inst.node_name, inst.ai2_belief.flip_evidence(), inst.ai2_belief.mode_posterior());

        // ── Rogue-mask yaw attribution (instrumentation) ──────────────────────────────────────────
        // Split this cycle's yaw move across the three channels (per-point GN / footprint-moment / 90° flip)
        // and, on a jump beyond kYawJumpLogRad (LOG-ONLY diagnostic trigger — NOT a control gate), dump the
        // covariate vector so we can see WHICH frame geometry drove it: obliquity + completeness (the gate's
        // blind spots) alongside anisotropy / motion / trunc / range. This is how we CATCH the rogue frames.
        {
            const auto  wrap    = [](float a) { return std::remainder(a, 2.0f * static_cast<float>(M_PI)); };
            const auto& st      = inst.ai2_belief.state();
            const float yaw_now = st.yaw;
            const float y_pts   = inst.ai2_belief.dbg_yaw_after_points();
            const float y_mom   = inst.ai2_belief.dbg_yaw_after_moment();
            inst.dbg_dyaw_points = wrap(y_pts - inst.dbg_yaw_pre);
            inst.dbg_dyaw_moment = wrap(y_mom - y_pts);
            inst.dbg_dyaw_flip   = wrap(yaw_now - y_mom);
            const float d_total  = wrap(yaw_now - inst.dbg_yaw_pre);
            const float em = inst.ai2_belief.dbg_moment_ext_major(), en = inst.ai2_belief.dbg_moment_ext_minor();
            inst.dbg_completeness = (st.w * st.h > 1e-4f and em > 0.0f) ? (em * en) / (st.w * st.h) : 1.0f;
            constexpr float kYawJumpLogRad = 0.087f;   // ≈5°: diagnostic PRINT trigger only (no fit effect)
            constexpr float kRad2Deg       = 57.29578f;
            if (std::abs(d_total) > kYawJumpLogRad)
                std::print("[yaw-jump] {} dpsi={:.1f}deg (pts={:.1f} mom={:.1f} flip={:.1f}) | oblq_cos={:.2f} compl={:.2f} "
                           "aniso={:.2f} r_yaw={:.3f} | trunc={:.2f} mvar={:.4f} dotd={:.2f} range={:.2f} npts={}\n",
                           inst.node_name, d_total * kRad2Deg, inst.dbg_dyaw_points * kRad2Deg,
                           inst.dbg_dyaw_moment * kRad2Deg, inst.dbg_dyaw_flip * kRad2Deg,
                           inst.dbg_obliquity_cos, inst.dbg_completeness,
                           inst.ai2_belief.dbg_moment_aniso(), inst.ai2_belief.dbg_moment_r_yaw(),
                           inst.last_trunc_frac, inst.last_motion_var, inst.last_motion_dotd, inst.last_range, npts);
        }

        // Appearance FRONT (door) resolver: project the freshly-updated box into the live ZED RGB and fold the
        // door-ness cue into the belief's discrete door-mode accumulator. Runs on an ACCEPTED update only (the
        // belief yaw/w/h it may adopt is propagated by the model write-back below). Reuses the ego-motion weight.
        resolve_front_from_rgb(inst, mode_evidence_weight);
    }

    // Write the belief back into the legacy CountableState so all downstream publish/viewer/RT code is
    // unchanged. Legs are derived: leg_length = H − TOP_THICKNESS, inset frozen at the outer edge.
    const auto& bs = inst.ai2_belief.state();
    CountableState ms = inst.model.state();
    ms.cx = bs.cx; ms.cy = bs.cy; ms.countable_height = bs.H; ms.w = bs.w; ms.h = bs.h; ms.yaw = bs.yaw;
    ms.leg_length = std::max(0.05f, bs.H - CountableModel::TOP_THICKNESS);
    ms.leg_inset  = CountableModel::LEG_RADIUS;
    inst.model.set_state(ms);

    ++inst.matched_frames;
    inst.detection_alive = inst.frames_since_detection < cfg_.detection_alive_max_frames;

    projection_->compute_projected_roi(inst);
    // (chain cov already computed above, before the belief update)

    // Object-anchor observation z_o: settled fit expressed in the localizer's base frame (gated OFF).
    compute_object_observation(inst);

    if (should_log(inst))
        std::print("[{}] AI2 npts={} R={:.4f} dotd={:.2f} trunc={:.2f}{} mom={} | FE={:.2f} base={:.2f} surprise={:.2f} | cx={:.3f} cy={:.3f} H={:.3f} w={:.3f} h={:.3f} ψ={:.3f} | σ(w,h,H)mm=({:.0f},{:.0f},{:.0f}) | lidar {}/{} bp{} resid={:.3f}m topz={:.3f}(H={:.3f}) floorz={:.3f} covA={:.2f} div={}\n",
                   inst.node_name, npts, R, inst.last_motion_dotd, inst.last_trunc_frac,
                   std::string(gated ? " GATED" : "") + (inst.dbg_lidar_only ? " LIDAR-ONLY" : ""),
                   inst.ai2_belief.last_moment_pts(),
                   energy, inst.fe_baseline, inst.fe_surprise,
                   bs.cx, bs.cy, bs.H, bs.w, bs.h, bs.yaw,
                   1000.f * std::sqrt(std::max(0.f, inst.ai2_belief.covariance()(3, 3))),
                   1000.f * std::sqrt(std::max(0.f, inst.ai2_belief.covariance()(4, 4))),
                   1000.f * std::sqrt(std::max(0.f, inst.ai2_belief.covariance()(2, 2))),
                   inst.dbg_lidar_rays, inst.dbg_lidar_raw, inst.dbg_lidar_bpearl_rays, inst.dbg_lidar_resid_m,
                   inst.dbg_lidar_topz_m, bs.H, inst.dbg_lidar_floorz_m, inst.dbg_lidar_cov_ang, inst.frames_diverged);

    // EvidenceMonitor snapshot: persist this frame's fit evidence (else it dies with the local observation).
    inst.dbg_cand_pts  = static_cast<int>(observation.candidate_pts.size());
    inst.dbg_resid_pts = static_cast<int>(observation.residual_pts.size());
    inst.dbg_energy    = energy;
    inst.dbg_R         = R;
    inst.dbg_gated     = gated;
    inst.dbg_gate_fresh = true;   // computed from THIS frame's mask — see countable_instance.h

    log_ai2_csv(inst, npts, R, gated, energy);

    // Periodic shape model-selection (round vs square) on the accumulated cloud → inst.subtype.
    evaluate_shape(inst);
    return energy;
}

// STRIPPED for countable_concept: a bed has a single fixed CLASS (a solid box) — there is no
// round/square shape hypothesis to select. The subtype is the constant class discriminator "bed"
// (set once at birth + republished by the scene-graph). Kept as a no-op so the call site / header are
// unchanged. (table_concept's round-vs-square model-evidence test lived here; not applicable to one box.)
void CountableFitter::evaluate_shape(CountableInstance& inst)
{
    inst.subtype = cfg_.concept_class;
}

// Append one AI2 belief row (state + Σ-diag std + mask R/bias/trunc + mode evidence + LiDAR diag) to the CSV.
void CountableFitter::log_ai2_csv(const CountableInstance& inst, int npts, float R, bool gated, float energy)
{
    if (cfg_.ai2_csv_path.empty())
        return;
    if (not ai2_csv_.is_open())
    {
        rc::diag::open_rotating(ai2_csv_, cfg_.ai2_csv_path);
        ai2_csv_.imbue(std::locale::classic());   // ★Qt imbues the global locale, which inserts THOUSANDS SEPARATORS
                                            // into integers (pkt_ts 1785763853131 -> "1,785,763,853,131"), splitting
                                            // one CSV field into five and making the whole log unparseable by column.
                                            // Pin "C" so the log is machine-readable regardless of the UI locale.
        if (not ai2_csv_.is_open()) { cfg_.ai2_csv_path.clear(); return; }
        ai2_csv_ << "cycle,node,pkt_fid,pkt_ts,npts,gated,energy,fe_baseline,fe_surprise,R,motion_var,depth_var,motion_dotd,trunc_frac,range,"
                 << "cx,cy,H,w,h,yaw,std_cx,std_cy,std_H,std_w,std_h,std_yaw,"
                 // door-mode resolver state. These REPLACE flip_ev/p_alt, which were flip_evidence() /
                 // mode_posterior() — hardcoded `return 0.0f` stubs left from table_concept's w↔h mode machinery
                 // that this asymmetric-box model does not use. They logged 0 for every row of every run, so the
                 // ONE thing that can resolve yaw on a square footprint was completely uninstrumented.
                 << "std_yaw_within,front_mode,front_conf,cue_conf,cue_bearing,wall_resp,"
                 << "lidar_rays,lidar_raw,lidar_bpearl,lidar_resid_m,lidar_meanz,lidar_topz,lidar_floorz,lidar_cov_ang,"
                 << "dyaw_points,dyaw_moment,dyaw_flip,obliquity_cos,completeness,moment_aniso,moment_r_yaw,"
                 << "mom_major,mom_minor,mom_phi,mom_pts,"   // RAW footprint statistic (basin diagnosis)   // rogue-mask diag
                 << "ex_L,ex_p,ex_locc,ex_lfree,ex_lfree_eff,ex_ln,ex_socc,ex_sfree,ex_sfree_eff,ex_sndet,ex_streak,"
                 // Classifier-free contour channel (common/contour_edge). ★A COUNT OF 0 MEANS NOT
                 // MEASURED, which is a different state from a verdict of 0, and only these columns can
                 // tell them apart: ct_en==0 = no RGB/no surviving control, ct_dn==0 = no depth read.
                 // ct_bias is SIGNED (observed − predicted depth, m) — a low verdict with a large bias
                 // is a FIT error, not an absence, and that distinction is the whole reason it is logged.
                 // ORDER MATTERS: these sit here because the row below emits them here.
                 << "ct_exc,ct_en,ct_v,ct_bias,ct_dn,ct_dL,"
                 << "ex_pdetect,ex_central,ex_verify,ex_wantsverify,"
                 << "omega_w,omega_yaw,tau_w,tau_yaw,"   // HISTORY: inferred log-volatility + retention (frames)   // existence-removal + verification-gate diag
                 // DETECTOR-ENVELOPE CALIBRATION. min_fill/max_fill/soft are an admitted PRIOR, never measured,
                 // and every stand-off in the fleet now inherits them. These are the columns the offline fit
                 // needs: (roi_fill, det_alive) is the (framing, hit) pair, ex_p above weights the trial so a
                 // miss caused by the object not being there is not charged to the envelope, and the two axes
                 // let the fit control for grazing views — see roi_fill_h/roi_fill_v in countable_instance.h
                 // for why the max alone biases the result.
                 << "roi_fill,roi_fill_h,roi_fill_v,roi_valid,det_alive,frames_since_det,"
                 // What the NBV actually proposed (lags the state columns by one cycle; see the instance).
                 // nbv_vfov=0 ⇒ the sensor model was incomplete and NO proposal was made that cycle.
                 << "nbv_standoff,nbv_tx,nbv_ty,nbv_pdetect,nbv_vfov,nbv_gain_raw,nbv_gain_pub,ricoh_det,ricoh_attn,p_occ,p_cls,"
                 // the likelihood-only (H, w, h) size summary the class factor scores (final review m3): its mean, its
                 // per-variable std 1/sqrt(info diag) (inf when the data inform that variable not at all), and Λ (nats,
                 // the last class_log_evidence; it may lag this row by one cycle)
                 << "lik_H,lik_w,lik_h,lik_sd_H,lik_sd_w,lik_sd_h,lam,"
                 // Camera attributes, LiDAR measures (Model.LidarMeasuresOnGated; spec 2026-11-15 §7c.2 "Log per cycle").
                 // lidar_only = this cycle's update was a LiDAR-only frame; lidar_msum = Σ membership of the staged rays
                 // (0 = no membership); rho_* = the lag-1 correlation THIS cycle's accepted update used, per channel that measured (cx
                 // component; 1 when none was applied: no update, a rejected one, or the channel did not measure); dphi_cam/dz_cam = the camera path
                 // that ρ_cam consumed; b* / sb* = latent means / stds (cam and LiDAR on H); ray_n = rays in
                 // the updating frame; Itr_* = trace of each channel's information after its fresh marginalisation.
                 // §7c.7 (e = the robot's pose error): e_* / se_* = its mean / √diag Σ_e, lever_* = ℓ, spose_* = √diag Σ_pose
                 // at the capture stamp (also reported while parked), c_* = the localiser correction ρ_chain used.
                 // All zero / ρ = 1 with the flag off.
                 << "lidar_only,lidar_msum,rho_cam,rho_L,rho_chain,dphi_cam,dz_cam,bcam_H,bL_H,"
                 << "sbcam_H,sbL_H,ray_n,Itr_cam,Itr_L,"
                 << "e_x,e_y,e_psi,se_x,se_y,se_psi,lever_x,lever_y,spose_x,spose_y,spose_psi,c_x,c_y,c_psi\n";
    }
    const auto& s = inst.ai2_belief.state();
    const auto& S = inst.ai2_belief.covariance();
    const auto sd = [&](int i) { return std::sqrt(std::max(0.0f, S(i, i))); };   // posterior std (m / rad)
    // std_yaw is now the MARGINAL (mode-entropy-inflated) yaw std; std_yaw_within is the within-mode Σ(5,5).
    ai2_csv_ << inst.processed_cycles << ',' << inst.node_name << ','
             << inst.dbg_pkt_frame_id << ',' << inst.dbg_pkt_ts_ms << ','
             << npts << ',' << (gated ? 1 : 0) << ','
             << energy << ',' << inst.fe_baseline << ',' << inst.fe_surprise << ','
             << R << ',' << inst.last_motion_var << ',' << inst.last_depth_var << ',' << inst.last_motion_dotd << ','
             << inst.last_trunc_frac << ',' << inst.last_range << ','
             << s.cx << ',' << s.cy << ',' << s.H << ',' << s.w << ',' << s.h << ',' << s.yaw << ','
             << sd(0) << ',' << sd(1) << ',' << sd(2) << ',' << sd(3) << ',' << sd(4) << ','
             << std::sqrt(std::max(0.0f, inst.ai2_belief.yaw_marginal_var())) << ','
             << sd(5) << ','
             // front_mode/front_conf = the belief's adopted door quarter-turn + its posterior peakedness;
             // cue_conf/cue_bearing = what detect_front actually returned THIS frame (0 ⇒ no cue was emitted:
             // no RGB frame, <2 faces qualified, or the margin fell under FrontMinConfidence).
             << inst.ai2_belief.front_mode() << ',' << inst.ai2_belief.front_confidence() << ','
             << inst.last_front_conf << ',' << inst.last_front_bearing << ','
             << inst.ai2_belief.last_wall_resp() << ','   // mean per-point WALL responsibility (→1 ⇒ it IS a wall)
             << inst.dbg_lidar_rays << ',' << inst.dbg_lidar_raw << ',' << inst.dbg_lidar_bpearl_rays << ',' << inst.dbg_lidar_resid_m << ','
             << inst.dbg_lidar_meanz_m << ',' << inst.dbg_lidar_topz_m << ',' << inst.dbg_lidar_floorz_m << ','
             << inst.dbg_lidar_cov_ang << ','
             << inst.dbg_dyaw_points << ',' << inst.dbg_dyaw_moment << ',' << inst.dbg_dyaw_flip << ','
             << inst.dbg_obliquity_cos << ',' << inst.dbg_completeness << ','
             << inst.ai2_belief.dbg_moment_aniso() << ',' << inst.ai2_belief.dbg_moment_r_yaw() << ','   // rogue-mask diag
             << inst.ai2_belief.dbg_moment_ext_major() << ',' << inst.ai2_belief.dbg_moment_ext_minor() << ','
             << inst.ai2_belief.dbg_moment_phi() << ',' << inst.ai2_belief.dbg_moment_pts() << ','
             << inst.existence.logodds() << ',' << inst.p_exists() << ','   // p_exists = the PRODUCT; p_occ, p_cls at the end
             << inst.dbg_ex_lidar_occ << ',' << inst.dbg_ex_lidar_free << ',' << inst.dbg_ex_lidar_free_eff << ',' << inst.dbg_ex_lidar_n << ','
             << inst.dbg_ex_sil_occ << ',' << inst.dbg_ex_sil_free << ',' << inst.dbg_ex_sil_free_eff << ',' << inst.dbg_ex_sil_ndet << ','
             << inst.existence_debounce.streak << ','
             << inst.dbg_ex_edge_excess << ',' << inst.dbg_ex_edge_n << ','
             << inst.dbg_ex_depth_verdict << ',' << inst.dbg_ex_depth_bias_m << ','
             << inst.dbg_ex_depth_n << ',' << inst.dbg_ex_contour_dL << ','
             << inst.dbg_ex_pdetect << ',' << inst.dbg_ex_central << ','
             << inst.verify_surprise << ',' << (inst.wants_verification ? 1 : 0) << ','
             << inst.ai2_belief.log_volatility()(3) << ',' << inst.ai2_belief.log_volatility()(5) << ','
             << inst.ai2_belief.retention_frames()(3) << ',' << inst.ai2_belief.retention_frames()(5) << ','   // existence-removal + verification-gate diag
             << inst.roi_fill << ',' << inst.roi_fill_h << ',' << inst.roi_fill_v << ','
             << (inst.roi_valid ? 1 : 0) << ',' << (inst.detection_alive ? 1 : 0) << ','
             << inst.frames_since_detection << ','
             << inst.dbg_nbv_standoff << ',' << inst.dbg_nbv_target_x << ',' << inst.dbg_nbv_target_y << ','
             << inst.dbg_nbv_pdetect << ',' << inst.dbg_nbv_vfov << ','
             << inst.dbg_nbv_gain_raw << ',' << inst.dbg_nbv_gain_pub
             // Peripheral (ricoh-360) attention: unassigned 360 detections of this label this
             // cycle. It was computed and shown nowhere — the whole peripheral channel was
             // invisible in every agent's log, which is why 'is it producing anything?' could
             // not be answered offline for cabinet/hood, the two it exists for.
             << ',' << ricoh_dets_ << ',' << ricoh_attention_
             << ',' << inst.existence.p_exists() << ',' << inst.class_belief.p_exists();
    {
        const auto& lik = inst.ai2_belief.size_likelihood();
        const auto lsd = [&](int i)
        { return lik.info(i, i) > 0.0f ? 1.0f / std::sqrt(lik.info(i, i)) : std::numeric_limits<float>::infinity(); };
        ai2_csv_ << ',' << lik.mean(0) << ',' << lik.mean(1) << ',' << lik.mean(2)
                 << ',' << lsd(0) << ',' << lsd(1) << ',' << lsd(2) << ',' << inst.class_lambda;
    }
    {
        using B = CountableBelief;
        const auto& b  = inst.ai2_belief.latent_mean();
        const auto  sb = inst.ai2_belief.latent_std();
        ai2_csv_ << ',' << inst.dbg_lidar_only << ',' << inst.dbg_lidar_msum
                 << ',' << inst.dbg_rho_cam << ',' << inst.dbg_rho_L << ',' << inst.dbg_rho_chain
                 << ',' << inst.dbg_dphi_cam << ',' << inst.dbg_dz_cam
                 << ',' << b(B::kCam + 2) << ',' << b(B::kL + 2)
                 << ',' << sb(B::kCam + 2) << ',' << sb(B::kL + 2)
                 << ',' << inst.dbg_ray_n << ',' << inst.dbg_itr_cam << ',' << inst.dbg_itr_L
                 << ',' << b(B::kChain) << ',' << b(B::kChain + 1) << ',' << b(B::kChain + 2)
                 << ',' << sb(B::kChain) << ',' << sb(B::kChain + 1) << ',' << sb(B::kChain + 2)
                 << ',' << inst.ai2_belief.chain_lever().l.x() << ',' << inst.ai2_belief.chain_lever().l.y()
                 << ',' << inst.dbg_spose_sd(0) << ',' << inst.dbg_spose_sd(1) << ',' << inst.dbg_spose_sd(2)
                 << ',' << inst.dbg_chain_c(0) << ',' << inst.dbg_chain_c(1) << ',' << inst.dbg_chain_c(2) << '\n';
    }
    ai2_csv_.flush();
}

// ─── Factory helpers ──────────────────────────────────────────────────────────────────────────────

// Build the CountableModel params from config (the SDF split band for candidate/residual classification).
CountableModelParams CountableFitter::make_model_params() const
{
    CountableModelParams p;
    p.headboard_h_m = cfg_.ai2_headboard_h_m;
    p.headboard_t_m = cfg_.ai2_headboard_t_m;
    p.sigma_obs = cfg_.sigma_obs;   // top/leg SDF split band for the candidate/residual classification
    return p;
}

// ─── One-shot fitter self-check: motion_magnitude robustness + the "be-still-to-update" confirm_only gate ─────
// Pure: builds a fitter with a LOCAL default config and a null graph (update_ego_motion is not called, so no DSR is
// needed) and drives ego speeds directly. Mirrors chair_concept's ported policy 1:1.
bool CountableFitter::self_test()
{
    bool all = true;
    CountableConfig cfg;   // defaults: AI2MotionConfirmOnly=true, still-levels, ang-lever=2.0, center-radius=0.35
    CountableFitter f(nullptr, nullptr, cfg, nullptr, nullptr);

    // (1) motion_magnitude returns the ROBOT-TWIST term when the producer never populated motion_dotd (=0), and
    //     the |motion_dotd| term when THAT dominates. Producer-independence is the whole point of the max().
    CountableInstance inst;
    inst.last_motion_dotd = 0.0f;                                  // producer gave nothing
    f.ego_lin_mps_ = 0.30f; f.ego_ang_radps_ = 0.0f;
    const float mm_lin = f.motion_magnitude(inst);                 // = ego_lin
    f.ego_lin_mps_ = 0.0f;  f.ego_ang_radps_ = 0.10f;
    const float mm_ang = f.motion_magnitude(inst);                 // = ang_lever · ego_ang
    f.ego_lin_mps_ = 0.0f;  f.ego_ang_radps_ = 0.0f; inst.last_motion_dotd = 0.50f;
    const float mm_dotd = f.motion_magnitude(inst);                // = |motion_dotd|
    const bool ok_mm = std::abs(mm_lin  - 0.30f) < 1e-4f
                   and std::abs(mm_ang  - cfg.ai2_ang_lever_m * 0.10f) < 1e-4f
                   and std::abs(mm_dotd - 0.50f) < 1e-4f;
    std::print("CountableFitter::self_test [motion_magnitude]   {}  lin={:.3f} ang={:.3f} dotd={:.3f}\n",
               ok_mm ? "PASS" : "FAIL", mm_lin, mm_ang, mm_dotd);
    all = all and ok_mm;

    // (2) confirm_only ("only update if the robot is still"): MOVING+off-centre ⇒ true; STILL ⇒ false;
    //     MOVING+well-centred ⇒ false (the trusted-centre exception).
    CountableInstance ci; ci.last_motion_dotd = 0.0f;
    f.ego_lin_mps_ = 0.30f; f.ego_ang_radps_ = 0.0f; ci.last_centroid_radius = 0.60f;   // moving & off-centre
    const bool co_move_off = f.confirm_only(ci);
    f.ego_lin_mps_ = 0.0f;  f.ego_ang_radps_ = 0.0f; ci.last_motion_dotd = 0.0f;         // still
    const bool co_still = f.confirm_only(ci);
    f.ego_lin_mps_ = 0.30f; ci.last_centroid_radius = 0.20f;                              // moving & well-centred
    const bool co_move_centre = f.confirm_only(ci);
    const bool ok_co = co_move_off and not co_still and not co_move_centre;
    std::print("CountableFitter::self_test [confirm_only]       {}  move_off={} still={} move_centre={}\n",
               ok_co ? "PASS" : "FAIL", co_move_off, co_still, co_move_centre);
    all = all and ok_co;

    // (3) The gate's EFFECT: confirm_only ⇒ predict-only ⇒ the geometry MEAN is HELD, whereas a still frame's
    //     update MOVES it toward the cloud. This is exactly run_inference's branch: if (gate) predict(); else update().
    CountableBeliefParams P;
    const CountableBeliefState s0{0.0f, 0.0f, 1.70f, 0.60f, 0.60f, 0.0f};
    std::mt19937 rng(777); std::normal_distribution<float> nz(0.0f, 0.01f);
    CountableFrame fr;                                          // a box cloud offset from the seed by (0.15,-0.10)
    for (int i = 0; i < 1200; ++i)
    {
        const int fc = i & 3; const float u = (i % 60) / 60.0f - 0.5f, z = (i % 100) / 100.0f * 1.7f;
        float lx, ly;
        if      (fc == 0) { lx =  0.30f; ly = u * 0.60f; }
        else if (fc == 1) { lx = -0.30f; ly = u * 0.60f; }
        else if (fc == 2) { ly =  0.30f; lx = u * 0.60f; }
        else              { ly = -0.30f; lx = u * 0.60f; }
        fr.points.push_back({0.15f + lx + nz(rng), -0.10f + ly + nz(rng), z + nz(rng)});
    }
    fr.R.assign(fr.points.size(), P.sigma_base_m * P.sigma_base_m);
    CountableBelief bpred(s0, P);
    bpred.predict();                                              // confirm-only branch → mean unchanged
    const auto mp = bpred.state();
    const float d_pred = std::hypot(mp.cx - s0.cx, mp.cy - s0.cy);
    CountableBelief bupd(s0, P);
    for (int it = 0; it < 12; ++it) bupd.update(fr);              // still branch → mean tracks the cloud
    const auto mu = bupd.state();
    const float d_upd = std::hypot(mu.cx - s0.cx, mu.cy - s0.cy);
    const bool ok_branch = d_pred < 1e-4f and d_upd > 0.05f;
    std::print("CountableFitter::self_test [confirm-only hold]  {}  |dPredict|={:.4f} |dUpdate|={:.4f}\n",
               ok_branch ? "PASS" : "FAIL", d_pred, d_upd);
    all = all and ok_branch;

    // (4) Depth observability, the assumption the cold-start snap rests on: a footprint moment from ONE
    //     viewpoint CANNOT resolve depth. One camera pose sees a single vertical face, whose footprint
    //     projects to a line, so ext_minor measures mask noise (~0.03) and stays under the band — which is why
    //     run_inference does h = (ext_minor > depth_obs_band) ? ext_minor : prior and lands on the prior.
    //     The union of two perpendicular faces is what it would take to clear the band. This is also why the
    //     probation burst is NOT unioned in: it never contains a second face at BirthFrames=8, so anything
    //     that clears the band there is pose-error blur, not depth (see the ★ note in run_inference).
    {
        std::mt19937 rng(4242);
        std::uniform_real_distribution<float> U01(0.0f, 1.0f);
        std::normal_distribution<float> jitter(0.0f, 0.01f);            // mask/de-projection noise
        const float H_gt = 1.85f, W_gt = 0.70f, D_gt = 0.60f, yaw_gt = 0.40f;
        const float cg = std::cos(yaw_gt), sg = std::sin(yaw_gt);
        const auto to_room = [&](float lx, float ly, float z) -> Eigen::Vector3f
        { return {0.20f + cg * lx - sg * ly, -0.30f + sg * lx + cg * ly, z}; };
        // The +y face: a line in footprint, spanning the WIDTH only.
        std::vector<Eigen::Vector3f> face_y, face_x;
        for (int i = 0; i < 600; ++i)
            face_y.push_back(to_room((U01(rng) - 0.5f) * W_gt + jitter(rng),
                                     0.5f * D_gt + jitter(rng), U01(rng) * H_gt));
        // The +x face: perpendicular, spanning the DEPTH — only a second viewpoint ever sees this.
        for (int i = 0; i < 600; ++i)
            face_x.push_back(to_room(0.5f * W_gt + jitter(rng),
                                     (U01(rng) - 0.5f) * D_gt + jitter(rng), U01(rng) * H_gt));

        const float z_lo = 0.03f, z_hi = H_gt + 0.03f;
        const auto m_single = CountableBelief::footprint_moment(face_y, z_lo, z_hi);
        std::vector<Eigen::Vector3f> both = face_y;
        both.insert(both.end(), face_x.begin(), face_x.end());
        const auto m_union = CountableBelief::footprint_moment(both, z_lo, z_hi);

        const float band = cfg.ai2_depth_obs_band_m;
        const bool ok_depth = m_single.ok and m_union.ok
                          and m_single.ext_minor < band       // one face   → depth UNRESOLVED, h stays at prior
                          and m_union.ext_minor  > band       // two faces  → only then is depth observable
                          and std::abs(m_union.ext_major - m_single.ext_major) < 0.30f;  // width must AGREE:
                          // a true second face lifts only the minor axis. Blur stretches BOTH — that is how
                          // the live burst gave itself away (ext_major 0.669 -> 0.915 while claiming depth).
        std::print("CountableFitter::self_test [depth observ.]     {}  ext_minor: single={:.3f} union={:.3f}"
                   "  (band={:.3f})  ext_major: single={:.3f} union={:.3f}\n",
                   ok_depth ? "PASS" : "FAIL", m_single.ext_minor, m_union.ext_minor, band,
                   m_single.ext_major, m_union.ext_major);
        all = all and ok_depth;
    }

    return all;
}

}  // namespace rc
