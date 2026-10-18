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
                                 const std::optional<rc::interaction::RelPose>& keep_side_of) const
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
    if (uy.dot(p_robot - c) > 0.0f) { ux = -ux; uy = -uy; }   // rotate by π: the robot on −y, still right-handed

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
        r = rc::interaction::from_room(T_room_robot.value(), make(-ux, -uy), Eigen::Matrix3f::Identity());

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
    const auto init = anchor_from_room(inst, std::nullopt);
    if (not init.has_value())
        return false;   // no room / robot transform this cycle: try again next cycle
    auto a = rc::interaction::Anchor::create(*G, inst.node_id, aff.node_id(), agent_id, init.value());
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
            anchor_log_ << "t_ms,x,y,yaw,sx,sy,syaw,n_through,n_blocked,n_short,p_open,camera_nis\n";
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
                if (const auto r = anchor_from_room(*inst, pa.pose); r.has_value())
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
                    if (const auto r = anchor_from_room(*inst, pa.pose); r.has_value())
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
    if (robot.has_value() and pa.pose.stamp_ms > 0 and now > pa.pose.stamp_ms)
        if (const auto tw = rc::rt::robot_body_twist(*G, robot.value()); tw.has_value())
        {
            Eigen::Vector3f var = Eigen::Vector3f::Zero();   // [var_adv, var_side, var_rot]; none stated ⇒ none added
            if (const auto v = G->get_attrib_by_name<robot_current_speed_variance_att>(robot.value());
                v.has_value() and v->get().size() >= 3)
                var = {v->get()[0], v->get()[1], v->get()[2]};
            const float dt = float(now - pa.pose.stamp_ms) * 1e-3f;
            pa.pose = rc::interaction::predict(pa.pose, tw.value(), var, dt);
        }
    // ★ ONE CLOCK: the anchor is stamped with system-clock epoch ms (provenance::now_ms), the clock
    //   read_affordance_anchor's staleness check uses. predict() copies the TWIST's stamp; overwrite it.
    pa.pose.stamp_ms = now;

    // ── UPDATE: bpearl rays + the door mask, in the ROBOT frame, each capture used once ──────────────────────
    std::vector<Eigen::Vector3f> rays, mask_pts;
    Eigen::Vector3f lidar_origin = Eigen::Vector3f::Zero();
    const Eigen::Vector2f c = pa.pose.x.head<2>();
    if (lidar_ingestor_ and lidar_ingestor_->bpearl_robot_stamp_ms() > pa.last_lidar_stamp)
    {
        pa.last_lidar_stamp = lidar_ingestor_->bpearl_robot_stamp_ms();
        lidar_origin = lidar_ingestor_->origin_bpearl_robot();
        const Eigen::Vector2f o = lidar_origin.head<2>();
        for (const auto& p : lidar_ingestor_->sweep_bpearl_robot())
        {
            if (p.z() < kFloorBandM or p.z() > pa.height) continue;   // the wall band
            const Eigen::Vector2f d = p.head<2>() - o;
            const float len2 = d.squaredNorm();
            if (len2 < 1e-6f) continue;
            const float t = std::clamp((c - o).dot(d) / len2, 0.0f, 1.0f);
            if ((o + t * d - c).norm() <= kRayReachM) rays.push_back(p);
        }
    }
    if (mask_ingestor_)
        if (const auto& pkt = mask_ingestor_->packet(); pkt.valid and pkt.frame_id != pa.last_mask_frame)
        {
            pa.last_mask_frame = pkt.frame_id;
            // The STATIC robot←zed mount: ts==0 on the main-thread-owned instance (CLAUDE.md ownership rule).
            if (const auto T = (robot_name.empty() or not inner_eigen_) ? std::nullopt
                                                  : inner_eigen_->get_transformation_matrix(robot_name, "zed", 0);
                T.has_value())
            {
                const Eigen::Matrix4d M = T->matrix();
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
                        pts.emplace_back(float(r.x()), float(r.y()), float(r.z()));
                        mean += pts.back().head<2>();
                    }
                    mean /= float(pts.size());
                    if (const float dist = (mean - c).norm(); dist < best) { best = dist; mask_pts = std::move(pts); }
                }
            }
        }
    rc::door::AnchorEvidence ev;
    const bool measured = not rays.empty() or mask_pts.size() >= 2;
    if (measured)
    {
        auto [post, e] = rc::door::update_door_anchor(pa.pose, pa.width, pa.thickness, lidar_origin, rays, mask_pts);
        post.stamp_ms = now;
        pa.pose = post;
        ev = e;
    }

    // ── WRITE ────────────────────────────────────────────────────────────────────────────────────────────────
    if (not pa.anchor->write(*G, pa.pose))
        if (not G->get_node(pa.anchor->id()).has_value())
        {
            std::print("door_concept: [anchor] {}@robot vanished from the graph — forgetting it\n", pa.door_name);
            passage_anchor_.reset();   // nothing left to remove; a fresh claim re-creates it
            return;
        }

    // ── LOG: one row per cycle while the anchor exists. -1 in the evidence columns = nothing measured this cycle
    //    (a sentinel OUTSIDE every column's range: counts ≥ 0, p_open ∈ (0,1), camera_nis ≥ 0).
    if (anchor_log_.is_open())
    {
        const auto& P = pa.pose.P;
        anchor_log_ << now << ',' << pa.pose.x.x() << ',' << pa.pose.x.y() << ',' << pa.pose.x.z() << ','
                    << std::sqrt(std::max(0.0f, P(0, 0))) << ',' << std::sqrt(std::max(0.0f, P(1, 1))) << ','
                    << std::sqrt(std::max(0.0f, P(2, 2))) << ','
                    << (measured ? ev.n_through : -1) << ',' << (measured ? ev.n_blocked : -1) << ','
                    << (measured ? ev.n_short : -1) << ',' << (measured ? ev.p_open : -1.0f) << ','
                    << (measured ? ev.camera_nis : -1.0f) << '\n';
        anchor_log_.flush();
    }
}
