/*
 * specificworker_anchor.cpp — door_concept OWNS the interaction anchor during a door passage.
 *
 * While the controller executes this door's `approach` or `cross`, a shallow copy of the door (`door_N@robot`)
 * hangs from the robot node (common/interaction_anchor). Its pose is the OPENING in the robot frame, initialised
 * ONCE from the room frames (robot←room ∘ room←opening), then predicted by the body twist and corrected by the
 * bpearl rays and the door mask, both in the robot frame (door_anchor_update.h). The room frame is not read again
 * until the consumer accepts a NEW epoch of the same affordance — so a relocalisation, a proto-room promotion or a
 * room-frame door fit 0.45 m off (try63 leg 2) cannot move the target under the crossing.
 *
 * ─── THE TRIGGER STATE MACHINE (one anchor per passage) ──────────────────────────────────────────────────────
 * The authoritative mark is the consumer-owned EDGE controller --[executing]--> affordance, read through
 * PragmaticAffordance::claimed()/claim() — which is AffordanceManager::read_executing, with the legacy
 * `active and epistemic_pending` fallback for a consumer that writes no edge. This agent NEVER writes that edge.
 *   CREATE   no anchor and some door's `cross` (preferred) or `approach` is claimed ⇒ from_room + Anchor::create.
 *   LINK     the affordance currently claimed (cross over approach) names the anchor (interaction_node_id). On the
 *            approach → cross handover the SAME anchor is re-linked (link_affordance drops the approach's link):
 *            same opening, same robot-frame belief, nothing re-initialised. Re-asserted EVERY cycle: a consumer's
 *            whole-node update of the affordance can erase an attribute it never read.
 *   RE-INIT  from_room again ONLY when (i) the claimed affordance is the one already linked and its executing epoch
 *            changed (the consumer accepted a new proposal), or (ii) the first claim after a room-switch RELINK (a
 *            new door instance: the old belief is not carried into a new passage). The axes keep their side.
 *   PARK     approach ended (any outcome), `open` claimed, or the door's instance is gone (merged): the anchor is
 *            kept, predicted and corrected — the passage may go on (open, then cross).
 *   ROOM     release_room(): a switch while the anchored door's CROSS is claimed IS the cross's terminal transition
 *            ⇒ REMOVE (the room switch normally lands before door_concept completes the cross). A switch during
 *            approach/open RELINKS the anchor to the door re-identified in the new room, and marks it for re-init.
 *   REMOVE   (a) the cross claim of the anchored door ENDED while its instance lives (completed, aborted,
 *                released): the passage is over; or the room switched during the cross (above);
 *            (b) nothing of this door's approach/open/cross is claimed AND THE SAME CONSUMER (the agent node of the
 *                claim we served, ExecutingClaim::agent_node_id) holds an `executing` edge to some other affordance:
 *                it has turned to something else (close, afford_room, another door). Other consumers' edges are
 *                not this passage's business. A legacy consumer (no edge, consumer unknown) never triggers it;
 *            (c) the agent exits (cleanup_owned_nodes), or the anchor node vanished under us.
 *   A parked anchor whose consumer holds NO executing edge is kept (a gap between two passage steps looks exactly
 *   like that); it is removed by (b) as soon as that consumer claims anything, and by (c).
 *   Room-less cycles (room_node_id_ == 0, mid-switch) still run the step: prediction and the robot-frame LiDAR need
 *   no room — only a re-init does, and it waits for one.
 *
 * Every graph access here is on the main thread (compute()), ts==0 on the main-thread-owned inner_eigen_.
 */

#include "specificworker.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <limits>
#include <locale>
#include <print>

#include "door_anchor_update.h"
#include "../../common/graph_provenance/creation_stamp.h"   // rc::provenance::now_ms — the clock read_affordance_anchor checks
#include "../../common/room_resolve/room_resolve.h"
#include "../../common/rt_twist/rt_twist.h"

namespace
{
// ★ σ_model (fix round 1, controller ruling): the room-frame-vs-sensor DISCREPANCY the anchor exists to absorb —
//   the error a confident room-frame chain can still carry and that neither the door belief nor the localisation
//   covariance can see (try63 leg 2: the door belief's σ_s was 0.04 m with the door 0.45 m off):
//   lateral 0.3 m  — a door fit can sit a jamb's width off along its wall;
//   depth   0.2 m  — the room polygon line vs the wall's mid-plane, plus the wall's thickness;
//   yaw     0.1 rad — a wall direction from a few metres of polygon edge.
// ADDED to the door belief's variance and to the projected localisation covariance (rc::door::anchor_prior_cov):
// independent sources add — no max, no floor.
const Eigen::Vector3f kSigmaModel(0.30f, 0.20f, 0.10f);
// Ray preselection (door_anchor_update.h asks the caller for the doorway NEIGHBOURHOOD): a beam whose segment
// passes within this distance of the predicted opening centre. COST, not information — rays farther away cannot
// cross the opening or its jambs, and add ~linear time to the free-energy evaluation. Not a decision level on
// the pose (no ray near the opening is ever dropped).
constexpr float kRayReachM = 2.0f;
// The wall band in z (robot frame: the base is at floor level, ROBOT_GEOMETRY.md): above the floor, below the
// lintel. Defines WHICH surfaces the model describes (walls, jambs, the leaf), as select_doorway_rays does;
// floor returns of the downward bpearl dome are not wall-plane measurements.
constexpr float kFloorBandM = 0.05f;
// ⚑ FLAGGED (Task 7, fix 3) — a PUBLICATION level, not a model term: the RT edge Shadow→anchor is rewritten only on
// cycles whose update consumed evidence with P(door in view) ≥ this. Between observations the anchor is predicted
// internally and not published, so the controller's kAnchorStaleMs rule retires it from driving (the room route
// takes over) instead of servoing on a dead-reckoned target nobody is looking at.
constexpr float kPublishViewLevel = 0.5f;
// ⚑ FLAGGED (Task 7, fix 1) — the zed's depth range for the camera's visibility term: the camera node carries the
// field of view (cam_rgb_width / cam_rgb_focalx, or cam_fov) but no range attribute.
constexpr float kZedMaxRangeM = 8.0f;

// Bring a robot-frame point captured `age_s` ago into the robot frame of NOW: the robot moved by the body twist
// over that interval (constant twist), so the point moves by its inverse — the same algebra as the anchor's own
// prediction (relative_pose.h move_mean). z is untouched.
Eigen::Vector3f to_now(const Eigen::Vector3f& p, const rc::rt::Twist& tw, float age_s)
{
    const Eigen::Vector3f q = rc::interaction::move_mean(Eigen::Vector3f(p.x(), p.y(), 0.f), tw.linear.x() * age_s,
                                                         tw.linear.y() * age_s, tw.angular.z() * age_s);
    return {q.x(), q.y(), p.z()};
}
}  // namespace

std::string SpecificWorker::robot_node_name() const
{
    if (not G) return {};
    const auto robots = G->get_nodes_by_type("robot");   // by TYPE, as rc::interaction / rc::rt do
    return robots.empty() ? std::string{} : robots.front().name();
}

// The OPENING's pose relative to the robot, from the room frames: robot←room ∘ room←opening. The opening frame:
// origin at the aperture centre on the wall line, x along the wall, y the wall normal pointing AWAY from the robot
// (the robot on −y, door_anchor_update.h). With `keep_side_of`, the axes are chosen to stay on the side the
// existing anchor already has (a re-init must not flip yaw by π under the consumer).
std::optional<rc::interaction::RelPose>
SpecificWorker::anchor_from_room(const rc::DoorInstance& inst,
                                 const std::optional<rc::interaction::RelPose>& keep_side_of, float* axis_sign_out) const
{
    if (not G or not inner_eigen_) return std::nullopt;
    const std::string room = rc::room::current_room_frame(*G);
    const std::string robot = robot_node_name();
    if (room.empty() or robot.empty()) return std::nullopt;
    const auto T_room_robot = inner_eigen_->get_transformation_matrix(room, robot, 0);
    if (not T_room_robot.has_value()) return std::nullopt;

    const auto& ap = inst.aperture;
    const Eigen::Vector2f c = ap.centre_xy();
    Eigen::Vector2f ux = ap.wall_u.normalized(), uy = ap.across_u().normalized();
    const Eigen::Vector2f p_robot = T_room_robot->translation().head<2>().cast<float>();
    float sign = 1.f;
    if (uy.dot(p_robot - c) > 0.0f) { ux = -ux; uy = -uy; sign = -1.f; }   // rotate by π: the robot on −y, still right-handed

    auto make = [&](const Eigen::Vector2f& ax, const Eigen::Vector2f& ay)
    {
        Eigen::Affine3d T = Eigen::Affine3d::Identity();
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        R.block<2, 1>(0, 0) = ax.cast<double>();
        R.block<2, 1>(0, 1) = ay.cast<double>();
        T.linear() = R;
        T.translation() = Eigen::Vector3d(c.x(), c.y(), 0.0);
        return T;
    };

    auto r = rc::interaction::from_room(T_room_robot.value(), make(ux, uy), Eigen::Matrix3f::Identity());
    if (keep_side_of.has_value()
        and std::abs(rc::interaction::wrap_pi(r.x.z() - keep_side_of->x.z())) > float(M_PI_2))
    {
        r = rc::interaction::from_room(T_room_robot.value(), make(-ux, -uy), Eigen::Matrix3f::Identity());
        sign = -sign;
    }
    if (axis_sign_out != nullptr) *axis_sign_out = sign;

    // ── P0 = door belief + localisation + σ_model, all in the ANCHOR's axes, then into RelPose coordinates ──────
    // (1) The door belief: variance of the opening's CENTRE along the wall, centre = O + (s + w/2)·u ⇒ J = [1, ½, 0]
    //     on [s, w, h]. The belief says nothing across the wall or about yaw (the wall is the room's). No belief ⇒ 0.
    Eigen::Matrix3f P_belief = Eigen::Matrix3f::Zero();
    if (inst.ai2_initialized)
    {
        const auto& S = inst.ai2_belief.covariance();
        if (const float v = S(0, 0) + 0.25f * S(1, 1) + S(0, 1); std::isfinite(v) and v > 0.0f)
            P_belief(0, 0) = v;
    }
    // (2) The localisation covariance room_concept writes on the robot↔room RT edge (pose covariance, PARENT frame),
    //     projected through the composition onto the anchor. This fleet anchors it robot → room; room → robot is
    //     handled too. No direct edge or no covariance on it ⇒ this term is 0 (stated, not invented).
    Eigen::Matrix3f P_robot = Eigen::Matrix3f::Zero();
    if (rt_api_)
        if (const auto room_n = G->get_node(room), robot_n = G->get_node(robot); room_n.has_value() and robot_n.has_value())
        {
            auto se2 = [](const Eigen::Matrix<double, 6, 6>& C6)
            {
                constexpr int k[3] = {0, 1, 5};
                Eigen::Matrix3f C = Eigen::Matrix3f::Zero();
                for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) C(i, j) = float(C6(k[i], k[j]));
                return C;
            };
            if (const auto e = G->get_edge(robot_n->id(), room_n->id(), "RT"); e.has_value())
            {
                if (const auto C6 = rt_api_->get_edge_RT_covariance(e.value()); C6.has_value() and C6->allFinite())
                {
                    const Eigen::Vector2f room_in_robot = T_room_robot->inverse().translation().head<2>().cast<float>();
                    P_robot = rc::door::robot_pose_cov_on_anchor_from_room_in_robot(se2(C6.value()), room_in_robot, r.x);
                }
            }
            else if (const auto e2 = G->get_edge(room_n->id(), robot_n->id(), "RT"); e2.has_value())
                if (const auto C6 = rt_api_->get_edge_RT_covariance(e2.value()); C6.has_value() and C6->allFinite())
                {
                    const auto& L = T_room_robot->linear();
                    P_robot = rc::door::robot_pose_cov_on_anchor_from_robot_in_room(
                        se2(C6.value()), float(std::atan2(L(1, 0), L(0, 0))), r.x);
                }
        }
    const Eigen::Matrix3f P_anchor = rc::door::anchor_prior_cov(P_belief, P_robot, kSigmaModel);
    r.P = rc::door::anchor_axes_to_robot(P_anchor, r.x.z());
    std::print("door_concept: [anchor] P0 for {} (anchor axes, σ): belief {:.3f} m · localisation ({:.3f}, {:.3f}) m "
               "{:.3f} rad · model ({:.2f}, {:.2f}) m {:.2f} rad ⇒ lateral {:.3f} m\n", inst.node_name,
               std::sqrt(P_belief(0, 0)), std::sqrt(std::max(0.f, P_robot(0, 0))), std::sqrt(std::max(0.f, P_robot(1, 1))),
               std::sqrt(std::max(0.f, P_robot(2, 2))), kSigmaModel.x(), kSigmaModel.y(), kSigmaModel.z(),
               std::sqrt(P_anchor(0, 0)));
    r.stamp_ms = static_cast<std::int64_t>(rc::provenance::now_ms());
    return r;
}

bool SpecificWorker::create_passage_anchor(rc::DoorInstance& inst, const rc::pragmatic::PragmaticAffordance& aff)
{
    float axis_sign = 1.f;
    const auto init = anchor_from_room(inst, std::nullopt, &axis_sign);
    if (not init.has_value())
        return false;   // no room / robot transform this cycle: try again next cycle
    // ★Fix round 1: the anchor is BORN STALE. Anchor::create writes the RT edge at the pose's stamp, and the room-frame
    //  initialisation is not an observation: stamped `now` it was driven on and drawn for up to kAnchorStaleMs with no
    //  sensor evidence. Its edge is stamped one tick past the staleness horizon instead, so the node always has a pose
    //  but the controller refuses it until the first update with P(door in view) ≥ kPublishViewLevel rewrites it. The
    //  internal belief keeps stamp `now` (the prediction's clock).
    rc::interaction::RelPose born = init.value();
    born.stamp_ms = init->stamp_ms - rc::interaction::kAnchorStaleMs - 1;
    auto a = rc::interaction::Anchor::create(*G, inst.node_id, aff.node_id(), agent_id, born);
    if (not a.has_value())
    {
        std::print("door_concept: [anchor] could NOT hang {} from the robot for {} (create failed)\n",
                   inst.node_name, aff.kind());
        return false;
    }
    PassageAnchor pa;
    pa.anchor       = std::move(a);
    pa.door_id      = inst.node_id;
    pa.door_name    = inst.node_name;
    pa.linked_aff   = aff.node_id();
    pa.linked_epoch = aff.claim().has_value() ? aff.claim()->epoch : -1;
    pa.consumer_id  = aff.claim().has_value() ? aff.claim()->agent_node_id : 0;
    pa.pose         = init.value();
    pa.axis_sign    = axis_sign;
    pa.width        = inst.aperture.w;
    pa.thickness    = inst.aperture.thickness;
    pa.height       = inst.aperture.h;
    pa.approach_was_claimed = inst.aff_approach.claimed();
    pa.cross_was_claimed    = inst.aff_cross.claimed();
    passage_anchor_ = std::move(pa);
    const auto& p = passage_anchor_->pose;
    std::print("door_concept: [anchor] {}@robot HUNG from the robot for {} (epoch {}): opening at ({:.2f},{:.2f}) "
               "yaw {:+.2f} in the robot frame, σ_lat {:.2f} m\n", inst.node_name, aff.kind(),
               passage_anchor_->linked_epoch, p.x.x(), p.x.y(), p.x.z(),
               std::sqrt(std::max(0.0f, p.P(0, 0) + p.P(1, 1))));

    if (not anchor_log_.is_open())
    {
        anchor_log_.open("etc/anchor_log.csv", std::ios::out | std::ios::trunc);
        anchor_log_.imbue(std::locale::classic());   // a comma decimal separator can never reach the file
        if (anchor_log_.is_open())
            anchor_log_ << "t_ms,x,y,yaw,sx,sy,syaw,n_through,n_blocked,n_short,p_open,camera_nis,p_view\n";
    }
    return true;
}

void SpecificWorker::remove_passage_anchor(std::string_view why)
{
    if (not passage_anchor_.has_value())
        return;
    if (passage_anchor_->anchor.has_value() and G)
        passage_anchor_->anchor->remove(*G);   // the affordance's link first, then the node (its RT edge goes with it)
    std::print("door_concept: [anchor] {}@robot REMOVED — {}\n", passage_anchor_->door_name, why);
    passage_anchor_.reset();
}

// The passage door re-identified (room switch: the entry mirror / recreated door is the crossed one). The anchor is
// NOT touched otherwise — it hangs from the robot, so no room frame reaches it.
void SpecificWorker::relink_passage_anchor(std::uint64_t new_door_id, std::string_view why)
{
    if (not passage_anchor_.has_value() or not passage_anchor_->anchor.has_value() or not G)
        return;
    auto& pa = *passage_anchor_;
    if (new_door_id == 0 or new_door_id == pa.door_id)
        return;
    if (not pa.anchor->relink(*G, new_door_id))
        std::print("door_concept: [anchor] relink of {}@robot to {} FAILED ({}) — keeping the old link\n",
                   pa.door_name, new_door_id, why);
    else
        std::print("door_concept: [anchor] {}@robot now names door {} ({})\n", pa.door_name, new_door_id, why);
    pa.door_id = new_door_id;
    // A different door instance: its affordances are new nodes. The old door's claims are not this one's history,
    // and the affordance linked so far went with the room left behind.
    pa.approach_was_claimed = pa.cross_was_claimed = false;
    pa.linked_aff   = 0;
    pa.linked_epoch = -1;
    pa.reinit_on_next_claim = true;   // a later claim on the new door is a new passage: re-init, never inherit
}

// ── STEP 0 (Task 7): are the anchor's measurements in the anchor's frame? One-shot, per anchor ────────────────────
// The anchor is the opening in the robot frame (Shadow: x right, y forward). The robot-frame bpearl sweep is the
// room-frame sweep's SAME capture restaged (LidarPlaneReader::restage), so room←robot at the capture stamp applied to
// it must reproduce the room-frame sweep point for point; and room←robot NOW differs from it by exactly the motion
// since the capture — the epoch gap the update closes with to_now. Returns false when this cycle cannot answer
// (no paired room sweep, no room, a transform missing): it is retried on the next capture.
bool SpecificWorker::check_anchor_ray_frames(const std::string& robot_name, float age_s, const rc::rt::Twist& tw)
{
    if (not lidar_ingestor_ or not inner_eigen_ or robot_name.empty()) return false;
    if (not lidar_ingestor_->bpearl_fresh() or not lidar_ingestor_->bpearl_robot_fresh()) return false;   // same pump = same capture
    const std::string room = rc::room::current_room_frame(*G);
    if (room.empty()) return false;
    const auto& pr = lidar_ingestor_->sweep_bpearl_robot();
    const auto& pw = lidar_ingestor_->sweep_bpearl_room();
    const auto ts = static_cast<std::uint64_t>(lidar_ingestor_->bpearl_robot_stamp_ms());
    const auto T_cap = inner_eigen_->get_transformation_matrix(room, robot_name, ts, "RT", DSR::RT_API::TimeQuery::Interpolated);
    const auto T_now = inner_eigen_->get_transformation_matrix(room, robot_name, 0);
    if (not T_cap.has_value() or not T_now.has_value() or pr.empty() or pr.size() != pw.size())
    {
        if (not pr.empty() and pr.size() != pw.size())
            std::print("door_concept: [anchor-frames] rays: robot sweep {} pts vs room sweep {} pts — not the same capture, retrying\n",
                       pr.size(), pw.size());
        return false;
    }
    const Eigen::Matrix4d Mc = T_cap->matrix(), Mn = T_now->matrix();
    std::vector<double> e_cap, e_now;
    e_cap.reserve(pr.size()); e_now.reserve(pr.size());
    for (std::size_t i = 0; i < pr.size(); ++i)
    {
        const Eigen::Vector4d q = Mc * Eigen::Vector4d(pr[i].x(), pr[i].y(), pr[i].z(), 1.0);
        e_cap.push_back((q.head<3>() - pw[i].cast<double>()).norm());
        const Eigen::Vector3f p_now = to_now(pr[i], tw, age_s);
        const Eigen::Vector4d qn = Mn * Eigen::Vector4d(p_now.x(), p_now.y(), p_now.z(), 1.0);
        e_now.push_back((qn.head<2>() - pw[i].head<2>().cast<double>()).norm());
    }
    auto stats = [](std::vector<double> v) { std::ranges::sort(v); return std::pair{v[v.size() / 2], v.back()}; };
    const auto [med_c, max_c] = stats(e_cap);
    const auto [med_n, max_n] = stats(e_now);
    const double yaw_gap = std::atan2(Mn(1, 0), Mn(0, 0)) - std::atan2(Mc(1, 0), Mc(0, 0));
    std::print("door_concept: [anchor-frames] rays ({} pts, room {}): room←robot@capture ∘ robot-frame sweep vs the room-frame "
               "sweep: median {:.4f} m, max {:.4f} m. Capture age {:.0f} ms; room←robot now vs @capture: yaw {:+.4f} rad, "
               "xy {:.3f} m (twist predicts yaw {:+.4f} rad). Brought to now by the twist, then room←robot now: median "
               "{:.4f} m, max {:.4f} m (2-D)\n",
               pr.size(), room, med_c, max_c, age_s * 1e3f, std::remainder(yaw_gap, 2.0 * M_PI),
               (Mn.block<2, 1>(0, 3) - Mc.block<2, 1>(0, 3)).norm(), tw.angular.z() * age_s, med_n, max_n);
    return true;
}

// The mask: support_points_cam (zed frame: x right, y DEPTH, z up — ROBOT_GEOMETRY.md) through robot←zed and then
// room←robot at the mask's capture stamp must reproduce MasksPacket::support_points (the ingestor's room frame).
bool SpecificWorker::check_anchor_mask_frames(const std::string& robot_name, const Eigen::Matrix4d& robot_T_zed)
{
    if (not mask_ingestor_ or not inner_eigen_ or robot_name.empty()) return false;
    const auto& pkt = mask_ingestor_->packet();
    if (not pkt.valid or pkt.support_points_cam.empty() or pkt.support_points_cam.size() != pkt.support_points.size())
        return false;
    const std::string room = rc::room::current_room_frame(*G);
    if (room.empty()) return false;
    const auto T_cap = pkt.timestamp_ms > 0
        ? inner_eigen_->get_transformation_matrix(room, robot_name, pkt.timestamp_ms, "RT", DSR::RT_API::TimeQuery::Interpolated)
        : inner_eigen_->get_transformation_matrix(room, robot_name, 0);
    if (not T_cap.has_value()) return false;
    const Eigen::Matrix4d M = T_cap->matrix() * robot_T_zed;
    std::vector<double> e;
    e.reserve(pkt.support_points.size());
    for (std::size_t i = 0; i < pkt.support_points.size(); ++i)
    {
        const auto& q = pkt.support_points_cam[i];
        const Eigen::Vector4d r = M * Eigen::Vector4d(q.x(), q.y(), q.z(), 1.0);
        e.push_back((r.head<3>() - pkt.support_points[i].cast<double>()).norm());
    }
    std::ranges::sort(e);
    std::print("door_concept: [anchor-frames] mask ({} pts, room {}, capture stamp {}): room←robot@capture ∘ robot←zed ∘ "
               "support_points_cam vs support_points: median {:.4f} m, max {:.4f} m\n",
               e.size(), room, pkt.timestamp_ms, e[e.size() / 2], e.back());
    return true;
}

void SpecificWorker::step_passage_anchor()
{
    if (not G or not fitter_ or not cfg_.pragmatic.enabled)
        return;
    auto& instances = fitter_->instances();

    // ── CREATE ───────────────────────────────────────────────────────────────────────────────────────────────
    if (not passage_anchor_.has_value())
    {
        for (auto& [id, inst] : instances)
        {
            const rc::pragmatic::PragmaticAffordance* aff = inst.aff_cross.claimed()    ? &inst.aff_cross
                                                          : inst.aff_approach.claimed() ? &inst.aff_approach
                                                                                        : nullptr;
            if (aff != nullptr and aff->node_id() != 0)
            {
                create_passage_anchor(inst, *aff);
                break;   // one passage at a time
            }
        }
        if (not passage_anchor_.has_value())
            return;
    }
    auto& pa = *passage_anchor_;
    rc::DoorInstance* inst = nullptr;
    if (auto it = instances.find(pa.door_id); it != instances.end())
        inst = &it->second;

    const bool app = inst != nullptr and inst->aff_approach.claimed();
    const bool crs = inst != nullptr and inst->aff_cross.claimed();
    const bool opn = inst != nullptr and inst->aff_open.claimed();

    // ── REMOVE (a): the crossing's own claim ended ───────────────────────────────────────────────────────────
    if (inst != nullptr and pa.cross_was_claimed and not crs)
    {
        remove_passage_anchor(std::format("the cross claim ended ({})",
                                          rc::pragmatic::PragmaticAffordance::state_name(inst->aff_cross.state())));
        return;
    }
    // ── REMOVE (b): nothing of this passage is claimed and the consumer is executing something else ──────────
    if (not (app or crs or opn))
    {
        const std::uint64_t mine[3] = {inst ? inst->aff_approach.node_id() : 0, inst ? inst->aff_open.node_id() : 0,
                                       inst ? inst->aff_cross.node_id() : 0};
        for (const auto& e : G->get_edges_by_type("executing"))
            if (pa.consumer_id != 0 and e.from() == pa.consumer_id
                and e.to() != mine[0] and e.to() != mine[1] and e.to() != mine[2])
            {
                const auto n = G->get_node(e.to());
                remove_passage_anchor(std::format("the consumer now executes '{}'", n.has_value() ? n->name() : "?"));
                return;
            }
    }
    pa.approach_was_claimed = app;
    pa.cross_was_claimed    = crs;

    // ── LINK + RE-INIT ───────────────────────────────────────────────────────────────────────────────────────
    if (inst != nullptr)
    {
        pa.width     = inst->aperture.w;
        pa.thickness = inst->aperture.thickness;
        pa.height    = inst->aperture.h;
        if (const rc::pragmatic::PragmaticAffordance* aff = crs ? &inst->aff_cross : app ? &inst->aff_approach : nullptr;
            aff != nullptr and aff->node_id() != 0)
        {
            const int epoch = aff->claim().has_value() ? aff->claim()->epoch : -1;
            if (aff->claim().has_value() and aff->claim()->agent_node_id != 0)
                pa.consumer_id = aff->claim()->agent_node_id;
            if (pa.reinit_on_next_claim)
            {
                // The first claim on the door re-identified after a room switch: a new passage's start.
                if (const auto r = anchor_from_room(*inst, pa.pose, &pa.axis_sign); r.has_value())
                {
                    std::print("door_concept: [anchor] {}@robot RE-INITIALISED from the room: first {} claim after "
                               "the room-switch relink\n", pa.door_name, aff->kind());
                    pa.pose = r.value();
                    pa.reinit_on_next_claim = false;
                }
            }
            else if (aff->node_id() == pa.linked_aff)
            {
                if (epoch >= 0 and pa.linked_epoch >= 0 and epoch != pa.linked_epoch)
                    if (const auto r = anchor_from_room(*inst, pa.pose, &pa.axis_sign); r.has_value())
                    {
                        std::print("door_concept: [anchor] {}@robot RE-INITIALISED from the room: {} epoch {} → {}\n",
                                   pa.door_name, aff->kind(), pa.linked_epoch, epoch);
                        pa.pose = r.value();
                    }
            }
            else
                std::print("door_concept: [anchor] {}@robot now serves {} (epoch {}) — same anchor, belief kept\n",
                           pa.door_name, aff->kind(), epoch);
            pa.linked_aff   = aff->node_id();
            pa.linked_epoch = epoch;
        }
    }
    if (pa.linked_aff != 0)
        pa.anchor->link_affordance(*G, pa.linked_aff);   // re-asserted every cycle (ruling 2); no write when present

    // ── PREDICT: the body twist moves the opening in the robot frame ─────────────────────────────────────────
    const auto now = static_cast<std::int64_t>(rc::provenance::now_ms());
    const std::string robot_name = robot_node_name();
    const auto robot = robot_name.empty() ? std::nullopt : G->get_node(robot_name);
    std::optional<rc::rt::Twist> twist;
    if (robot.has_value())
        twist = rc::rt::robot_body_twist(*G, robot.value());
    if (twist.has_value() and pa.pose.stamp_ms > 0 and now > pa.pose.stamp_ms)
    {
        Eigen::Vector3f var = Eigen::Vector3f::Zero();   // [var_adv, var_side, var_rot]; none stated ⇒ none added
        if (const auto v = G->get_attrib_by_name<robot_current_speed_variance_att>(robot.value());
            v.has_value() and v->get().size() >= 3)
            var = {v->get()[0], v->get()[1], v->get()[2]};
        const float dt = float(now - pa.pose.stamp_ms) * 1e-3f;
        pa.pose = rc::interaction::predict(pa.pose, twist.value(), var, dt);
    }
    // ★ ONE CLOCK: the anchor is stamped with system-clock epoch ms (provenance::now_ms), the clock
    //   read_affordance_anchor's staleness check uses. predict() copies the TWIST's stamp; overwrite it.
    pa.pose.stamp_ms = now;

    // ── THIS DOOR'S WALL RUN in anchor coordinates (Task 7, fix 2) ───────────────────────────────────────────
    // The run is [0, wall_len] from the belief's wall origin along wall_u (door_fitter's associate_wall); the anchor's
    // origin is the opening's centre and its x-axis is ±wall_u (axis_sign). Unknown run (no wall associated, len 0)
    // ⇒ ⚑ the FLAGGED default: the aperture ± one width each side. Kept from the last cycle while the instance is gone.
    if (inst != nullptr)
    {
        // The anchor's x-axis vs THIS instance's wall_u, re-derived every cycle (fix round 1): a relink to a mirrored
        // door instance (the entry mirror in the next room) can carry the opposite wall_u, and a sign latched at
        // creation would then flip [u_min, u_max]. Only the SIGN is read from the room frame — the anchor's pose is
        // not. No room transform this cycle ⇒ the last sign stands.
        {
            const Eigen::Vector2f wu = (inst->ai2_initialized ? inst->ai2_belief.params().wall_u : inst->aperture.wall_u).normalized();
            if (const std::string room = rc::room::current_room_frame(*G); not room.empty() and not robot_name.empty() and inner_eigen_)
                if (const auto T = inner_eigen_->get_transformation_matrix(room, robot_name, 0); T.has_value())
                {
                    const Eigen::Vector2d ax(std::cos(pa.pose.x.z()), std::sin(pa.pose.x.z()));
                    const Eigen::Vector2f ax_room = (T->linear().topLeftCorner<2, 2>() * ax).cast<float>();
                    if (const float d = ax_room.dot(wu); d != 0.f) pa.axis_sign = d > 0.f ? 1.f : -1.f;
                }
        }
        const float w = inst->aperture.w;
        float lo = -1.5f * w, hi = 1.5f * w;
        if (inst->ai2_initialized)
            if (const auto& bp = inst->ai2_belief.params(); bp.wall_len > 0.f)
            {
                const float sc = (inst->aperture.centre_xy() - bp.wall_O).dot(bp.wall_u.normalized());
                lo = -sc;
                hi = bp.wall_len - sc;
            }
        if (pa.axis_sign < 0.f) { const float t = lo; lo = -hi; hi = -t; }
        pa.wall_u_min = lo;
        pa.wall_u_max = hi;
    }
    else if (pa.wall_u_max <= pa.wall_u_min)
    {
        pa.wall_u_min = -1.5f * pa.width;
        pa.wall_u_max = 1.5f * pa.width;
    }

    // ── UPDATE: bpearl rays + the door mask, in the ROBOT frame OF NOW, each capture used once ──────────────
    // ★ STEP 0 (Task 7) — ONE EPOCH. The anchor is predicted to `now`; a sweep is restaged at its CAPTURE stamp and a
    //   mask at ITS capture stamp, i.e. in the robot frame of THEN. Both are brought forward to now with the same body
    //   twist the anchor is predicted by (to_now). A robot turning at 0.5 rad/s with a capture 100 ms old saw the
    //   wall 0.05 rad away from where the anchor (rightly) expects it.
    std::vector<Eigen::Vector3f> rays, mask_pts;
    Eigen::Vector3f lidar_origin = Eigen::Vector3f::Zero();
    const Eigen::Vector2f c = pa.pose.x.head<2>();
    if (lidar_ingestor_ and lidar_ingestor_->bpearl_robot_stamp_ms() > pa.last_lidar_stamp)
    {
        pa.last_lidar_stamp = lidar_ingestor_->bpearl_robot_stamp_ms();
        const float age_s = std::max<std::int64_t>(0, now - pa.last_lidar_stamp) * 1e-3f;
        const rc::rt::Twist tw = twist.value_or(rc::rt::Twist{});
        if (not pa.frames_checked_rays)
            pa.frames_checked_rays = check_anchor_ray_frames(robot_name, age_s, tw);
        lidar_origin = to_now(lidar_ingestor_->origin_bpearl_robot(), tw, age_s);
        // The wall band, plus the floor returns that went THROUGH the predicted opening (rc::door::keep_floor_return:
        // through the opening's span, inside the height band, landed beyond the far face by more than the prior's
        // depth σ — so a mis-predicted shut door cannot read open), within kRayReachM of the opening.
        std::vector<Eigen::Vector3f> sweep_now;
        sweep_now.reserve(lidar_ingestor_->sweep_bpearl_robot().size());
        for (const auto& p_then : lidar_ingestor_->sweep_bpearl_robot()) sweep_now.push_back(to_now(p_then, tw, age_s));
        rays = rc::door::select_anchor_rays(pa.pose, pa.width, pa.thickness, kFloorBandM, pa.height, lidar_origin,
                                            sweep_now, kRayReachM);
    }
    // The zed: its pose in the robot frame (static mount) and its frustum, for the mask AND the camera visibility.
    rc::door::CameraFrustum cam;
    std::optional<Eigen::Matrix4d> robot_T_zed;
    if (not robot_name.empty() and inner_eigen_)
        if (const auto T = inner_eigen_->get_transformation_matrix(robot_name, "zed", 0); T.has_value())   // static: ts==0, main-thread instance
        {
            robot_T_zed = T->matrix();
            if (const auto zed = G->get_node("zed"); zed.has_value())
            {
                float half_fov = 0.f;
                const auto wpx = G->get_attrib_by_name<cam_rgb_width_att>(zed.value());
                const auto fx  = G->get_attrib_by_name<cam_rgb_focalx_att>(zed.value());
                if (wpx.has_value() and fx.has_value() and fx.value() > 0)
                    half_fov = std::atan(0.5f * float(wpx.value()) / float(fx.value()));
                else if (const auto fov = G->get_attrib_by_name<cam_fov_att>(zed.value()); fov.has_value())
                    half_fov = 0.5f * fov.value();
                if (half_fov > 0.f)
                {
                    cam.known     = true;
                    cam.origin    = robot_T_zed->block<2, 1>(0, 3).cast<float>();
                    cam.forward   = robot_T_zed->block<2, 1>(0, 1).cast<float>().normalized();   // zed y = DEPTH
                    cam.half_fov  = half_fov;
                    cam.max_range = kZedMaxRangeM;
                }
            }
        }
    if (mask_ingestor_ and robot_T_zed.has_value())
        if (const auto& pkt = mask_ingestor_->packet(); pkt.valid and pkt.frame_id != pa.last_mask_frame)
        {
            pa.last_mask_frame = pkt.frame_id;
            const float age_s = pkt.timestamp_ms > 0 ? std::max<std::int64_t>(0, now - std::int64_t(pkt.timestamp_ms)) * 1e-3f : 0.f;
            const rc::rt::Twist tw = twist.value_or(rc::rt::Twist{});
            if (not pa.frames_checked_mask)
                pa.frames_checked_mask = check_anchor_mask_frames(robot_name, *robot_T_zed);
            const Eigen::Matrix4d& M = *robot_T_zed;
            // The door slice nearest the PREDICTED opening (robot frame); a misplaced one loses its pull
            // through the Student-t tail inside update_door_anchor, not through a gate here.
            float best = std::numeric_limits<float>::infinity();
            for (const auto& sl : pkt.slices)
            {
                if (sl.label != "door" or not sl.may_fit_geometry() or sl.support_end <= sl.support_begin
                    or sl.support_end > pkt.support_points_cam.size())
                    continue;
                std::vector<Eigen::Vector3f> pts;
                pts.reserve(sl.support_end - sl.support_begin);
                Eigen::Vector2f mean = Eigen::Vector2f::Zero();
                for (std::size_t i = sl.support_begin; i < sl.support_end; ++i)
                {
                    const auto& q = pkt.support_points_cam[i];
                    const Eigen::Vector4d r = M * Eigen::Vector4d(q.x(), q.y(), q.z(), 1.0);
                    pts.push_back(to_now(Eigen::Vector3f(float(r.x()), float(r.y()), float(r.z())), tw, age_s));
                    mean += pts.back().head<2>();
                }
                mean /= float(pts.size());
                if (const float dist = (mean - c).norm(); dist < best) { best = dist; mask_pts = std::move(pts); }
            }
        }
    rc::door::AnchorEvidence ev;
    const bool measured = not rays.empty() or mask_pts.size() >= 2;
    if (measured)
    {
        // The door's own P(open) (its leaf-angle posterior): what the visibility check predicts in the opening — a gap
        // or a leaf. Unknown this cycle ⇒ the last known value; never known ⇒ 0.5.
        if (inst != nullptr and inst->interaction.p_open_known)
            pa.p_open_belief = std::clamp(inst->interaction.p_open, 0.f, 1.f);
        auto [post, e] = rc::door::update_door_anchor(pa.pose, pa.width, pa.thickness, pa.wall_u_min, pa.wall_u_max,
                                                      lidar_origin, rays, mask_pts, cam, pa.p_open_belief);
        post.stamp_ms = now;
        pa.pose = post;
        ev = e;
    }

    // ── WRITE (Task 7, fix 3): only on a cycle whose update consumed evidence of a door IN VIEW ─────────────────
    // Between observations the belief is predicted (above) but not published; the edge then ages past
    // kAnchorStaleMs and the controller drops back to the room route. The anchor node itself stays.
    const bool publish = measured and ev.p_view >= kPublishViewLevel;
    if (measured and publish != pa.publishing)   // a transition is judged on a MEASURED cycle only
    {
        pa.publishing = publish;
        std::print("door_concept: [anchor] {}@robot {} publishing — P(door in view) {:.2f} (p_lidar {:.2f}, p_cam {:.2f})\n",
                   pa.door_name, publish ? "RESUMED" : "PAUSED", ev.p_view, ev.p_lidar, ev.p_cam);
    }
    const bool alive = publish ? (pa.anchor->write(*G, pa.pose) or G->get_node(pa.anchor->id()).has_value())
                               : G->get_node(pa.anchor->id()).has_value();
    if (not alive)
    {
        std::print("door_concept: [anchor] {}@robot vanished from the graph — forgetting it\n", pa.door_name);
        passage_anchor_.reset();   // nothing left to remove; a fresh claim re-creates it
        return;
    }

    // ── LOG: one row per cycle while the anchor exists. -1 in the evidence columns = nothing measured this cycle
    //    (a sentinel OUTSIDE every column's range: counts ≥ 0, p_open ∈ (0,1), camera_nis ≥ 0, p_view ∈ [0,1]).
    if (anchor_log_.is_open())
    {
        const auto& P = pa.pose.P;
        anchor_log_ << now << ',' << pa.pose.x.x() << ',' << pa.pose.x.y() << ',' << pa.pose.x.z() << ','
                    << std::sqrt(std::max(0.0f, P(0, 0))) << ',' << std::sqrt(std::max(0.0f, P(1, 1))) << ','
                    << std::sqrt(std::max(0.0f, P(2, 2))) << ','
                    << (measured ? ev.n_through : -1) << ',' << (measured ? ev.n_blocked : -1) << ','
                    << (measured ? ev.n_short : -1) << ',' << (measured ? ev.p_open : -1.0f) << ','
                    << (measured ? ev.camera_nis : -1.0f) << ',' << (measured ? ev.p_view : -1.0f) << '\n';
        anchor_log_.flush();
    }
}
