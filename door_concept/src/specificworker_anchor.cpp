/*
 * specificworker_anchor.cpp — door_concept OWNS the interaction anchor during a door passage.
 *
 * While the controller executes this door's `cross` (Task 19: offered only on a door believed OPEN), a shallow copy of the door (`door_N@robot`)
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
 *   SIDE     Task 21: every from_room (create, re-init, retry) puts the robot on the anchor's NEAR side (s < 0, +y through
 *            the opening): the side the cross leaves, else where the robot is — never the side an older belief had.
 *   CREATE   no anchor and some door's `cross` is claimed ⇒ from_room + Anchor::create (same P0: belief + localisation +
 *            σ_model). Task 19 (user decision): NOT on `approach` — approach and open run anchor-less on the room-frame
 *            standpoint; the anchor is born on an open door (rc::door::anchor_creation_claim).
 *   LINK     the claimed cross names the anchor (interaction_node_id). Re-asserted EVERY cycle: a consumer's
 *            whole-node update of the affordance can erase an attribute it never read. (No approach → cross relink.)
 *   RE-INIT  from_room again ONLY for a NEW passage: (i) the claimed affordance is the one already linked and its
 *            executing epoch changed (the consumer accepted a new proposal) — but NOT on a RETRY (Task 9: after a claim of
 *            this door's cross ENDED without the robot crossing, the next claim is the same passage tried
 *            again, and the corrected robot-frame belief is the one thing the retry must keep); or (ii) the first claim
 *            after a room-switch RELINK (a new door instance: the old belief is not carried into a new passage). The
 *            axes keep their side. The door's GEOMETRY (width, thickness, height, its wall run) is read from the room
 *            frame at the same two moments and nowhere else (Task 9: try66's belief jumped 0.15 m wider and 0.35 m along
 *            its wall as the leaf swung open, and the anchor followed it every cycle).
 *   PARK     the CROSS ended without succeeding (Unreachable, Infeasible,
 *            Timeout, aborted, released, re-offered — Task 9), or the door's instance is gone (merged): the anchor is
 *            kept, predicted and corrected — the passage may go on (open, then cross, or the cross again).
 *   ROOM     release_room(): a switch while the anchored door's CROSS is claimed KEEPS the anchor (Task 12 Fix E: crossing_through)
 *            and RELINKS it (the robot still crosses on it until through). Task 21: that passage is then OVER — the anchor is
 *            removed — as soon as this door is claimed again (any step: a new passage gets a FRESH anchor) or the predicted
 *            window is spent (it is no longer driven on), or by REMOVE (b). No anchor outlives its passage. A switch during
 *            approach/open of a PARKED cross retry RELINKS the anchor to the door re-identified in the new room, and marks it for re-init.
 *   REMOVE   (a) the cross claim of the anchored door ended SATISFIED while its instance lives: the robot is through,
 *                the passage's crossing is over; or the room switched during the cross (above);
 *            (b) nothing of this door's approach/open/cross is claimed AND THE SAME CONSUMER (the agent node of the
 *                claim we served, ExecutingClaim::agent_node_id) holds an `executing` edge to some other affordance:
 *                it has turned to something else (another object, or this door's `close`: the passage is ending).
 *                Other consumers' edges are not this passage's business. A legacy consumer (no edge, consumer
 *                unknown) never triggers it;
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
#include "door_anchor_open.h"   // Task 11: the anchor's beams as φ evidence; the wait pose
#include "door_phi_posterior.h"   // rc::door::phi_grid — the door's own φ grid
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
                                 float* axis_sign_out) const
{
    if (not G or not inner_eigen_) return std::nullopt;
    const std::string room = rc::room::current_room_frame(*G);
    const std::string robot = robot_node_name();
    if (room.empty() or robot.empty()) return std::nullopt;
    const auto T_room_robot = inner_eigen_->get_transformation_matrix(room, robot, 0);
    if (not T_room_robot.has_value()) return std::nullopt;

    const auto& ap = inst.aperture;
    const Eigen::Vector2f c = ap.centre_xy();
    const Eigen::Vector2f p_robot = T_room_robot->translation().head<2>().cast<float>();
    // ★ Task 21: the robot's NEAR side, always — the side the cross leaves (latched), else where the robot is. Never the side
    //   an older belief had (try89 leg 3: a re-init kept leg 2's side ⇒ s +1.78, the normal flipped). rc::door::anchor_near_side.
    const auto axes = rc::door::anchor_opening_axes(
        ap.wall_u, ap.across_u(), rc::door::anchor_near_side(inst.cross_side_latched, ap.across_u().normalized().dot(p_robot - c)));
    const Eigen::Vector2f ux = axes.ux, uy = axes.uy;
    float sign = axes.sign;

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
    const auto init = anchor_from_room(inst, &axis_sign);
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
    pa.init_pose    = init.value();   // Task 11 fix 4
    pa.axis_sign    = axis_sign;
    anchor_geometry_from_door(inst, pa);   // Task 9: the geometry is read with the pose, once (width_m set below)
    pa.cross_was_claimed = inst.aff_cross.claimed();
    pa.last_seen_ms = init->stamp_ms;   // never seen yet: unseen since creation (unseen_time)
    passage_anchor_ = std::move(pa);
    passage_anchor_->anchor->set_width(*G, passage_anchor_->clear_width);
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

// The door's geometry as the anchor's model needs it — width, thickness, height and its wall RUN in anchor coordinates —
// read from the door instance (its room-frame belief). Task 9: called when the pose is (re)initialised from the room and
// at no other time, for the same reason the pose is: try66 leg 1's belief jumped from w 1.06 to 1.22 and 0.35 m along
// its wall when the leaf swung open, which moved the run's end INSIDE the opening's half-width, and the anchor (reading
// it every cycle) lost its wall there. The one exception is an UNKNOWN run (no wall associated yet ⇒ ⚑ the flagged
// default, the aperture ± one width each side): the first known run is taken once — that is first information about the
// wall, not a refit of it.
void SpecificWorker::anchor_geometry_from_door(const rc::DoorInstance& inst, PassageAnchor& pa) const
{
    pa.width     = inst.aperture.w;   // the WALL GAP: the anchor update's thin-wall opening
    pa.clear_width = rc::door::anchor_clear_width(pa.width, cfg_.frame_casing_m);   // the passage: leaf + servo (fix round 1)
    pa.thickness = inst.aperture.thickness;
    pa.height    = inst.aperture.h;
    float lo = -1.5f * pa.width, hi = 1.5f * pa.width;
    pa.run_known = false;
    if (inst.ai2_initialized)
        if (const auto& bp = inst.ai2_belief.params(); bp.wall_len > 0.f)
        {
            const float sc = (inst.aperture.centre_xy() - bp.wall_O).dot(bp.wall_u.normalized());
            lo = -sc;
            hi = bp.wall_len - sc;
            pa.run_known = true;
        }
    if (pa.axis_sign < 0.f) { const float t = lo; lo = -hi; hi = -t; }
    pa.wall_u_min = lo;
    pa.wall_u_max = hi;
}

// A RETRY's claim (fix round 1, controller ruling): neither kept blindly (a diverged belief would never be re-initialised)
// nor replaced (try66: the room init 0.15 m / 18° off, its normal flipped). The belief is BROADENED by σ_model — the same
// room-vs-sensor discrepancy term P0 carries (kSigmaModel), added in the anchor's axes — and fused with the room
// initialisation (its P0, which already holds σ_model) as one Gaussian product, the yaw innovation wrapped. A good belief
// moves by at most a fraction of the room's error and its beams pull it back within frames; a diverged one is pulled
// toward the room and, broadened, can be re-acquired by its beams.
void SpecificWorker::retry_broad_prior(const rc::DoorInstance& inst, PassageAnchor& pa, std::string_view kind)
{
    float near_sign = pa.axis_sign;
    const auto r = anchor_from_room(inst, &near_sign);   // Task 21: the robot's near side, not the belief's
    if (not r.has_value())
    {
        std::print("door_concept: [anchor] {}@robot re-claimed for {} — a retry; no room transform: belief kept\n", pa.door_name, kind);
        return;
    }
    // Task 21: a belief on the OTHER side (its axes π from the near-side frame) is not the same quantity — fusing them would
    // average two yaws π apart. It is replaced by the near-side initialisation (it cannot happen within one passage: the
    // anchor is born near-side and the latched side holds for the attempt).
    if (std::abs(rc::interaction::wrap_pi(r->x.z() - pa.pose.x.z())) > float(M_PI_2))
    {
        std::print("door_concept: [anchor] {}@robot re-claimed for {} — the belief was on the far side: RE-INITIALISED on the "
                   "robot's near side\n", pa.door_name, kind);
        pa.pose = r.value();
        pa.init_pose = r.value();
        pa.axis_sign = near_sign;
        pa.bias = {};
        pa.view_lost_llr = 0.0;
        refresh_anchor_geometry(inst, pa);
        pa.retry = false;
        return;
    }
    // ★ Fix round 1 (I-2): weighted by P(the robot is OUT of the doorway band). Inside the band the room initialisation is the
    //   worst view of the opening there is, and a belief broadened there re-acquires an ALIAS (try70 leg 5: fused at the mouth,
    //   two frames later the update sat 74–86° off with the leaf playing the wall). w = 1 − P(in band) scales both the
    //   broadening and the room's information: K = w·P_b (w·P_b + R)⁻¹, continuous, 0 inside the band (the belief stands).
    // ★ Regression fix (try73 leg 1, attempt 3): the band is the doorway where the ROOM's view is worst — so the room is refused
    //   only when the robot is in the band by BOTH estimates (independent: P_in = P_in(belief)·P_in(room)). try73's belief
    //   put the robot at s −0.07 with σ 0.02 m while it truly stood 1.87 m out; the room said so (1.63 m away) and the
    //   belief-only band refused it (weight 0.00), leaving the anchor in its depth alias for the whole attempt.
    const float w_out = rc::door::retry_weight_out_of_band(pa.pose, r.value(), cfg_.pragmatic.cross_clearance_m);
    // ★ Task 17 (try78 leg 2): …and the belief is the SUSPECT when it and the room disagree far beyond their covariance
    //   (rc::door::belief_lost): with P(lost) the room's initialisation is the answer, a continuous mixture (retry_reinit).
    Eigen::Vector3f innov = r->x - pa.pose.x;
    innov.z() = rc::interaction::wrap_pi(innov.z());
    const auto lost = rc::door::belief_lost(pa.pose, r.value(), pa.clear_width, pa.view_lost_llr);
    const auto fused = rc::door::retry_reinit(pa.pose, r.value(),
                                              rc::door::anchor_axes_to_robot(Eigen::Matrix3f(kSigmaModel.cwiseProduct(kSigmaModel).asDiagonal()),
                                                                             pa.pose.x.z()),
                                              w_out, lost.p_lost);
    Eigen::Vector3f dx = fused.x - pa.pose.x;
    dx.z() = rc::interaction::wrap_pi(dx.z());
    pa.pose.x = fused.x;
    pa.pose.P = fused.P;
    pa.view_lost_llr = 0.0;   // Task 17 add-on: a new belief, a new visibility record
    pa.bias = {};   // Task 10: the fused belief carries no bias bookkeeping — it restarts from this P
    // Fix round 2: one fusion per retry claim (a later claim_rose in the same passage is not pulled again unless another
    // claim fails), and nothing seen before it may be published as "predicted" after it.
    pa.retry = false;
    pa.seen_once = false;
    pa.init_pose = r.value();   // Task 11 fix 4: nothing seen since this retry ⇒ the room's initialisation is the wait pose
    pa.init_pose.stamp_ms = pa.pose.stamp_ms;
    pa.mode = rc::door::AnchorPublish::None;
    std::print("door_concept: [anchor] {}@robot re-claimed for {} — a retry: belief broadened by σ_model and fused with the room "
               "init (room − belief {:.2f} m / {:+.1f}°, weight P(out of the doorway band) {:.2f}, Mahalanobis² {:.1f}, view log-odds {:+.1f} ⇒ P(belief lost) {:.2f}): "
               "moved {:.3f} m / {:+.1f}°, σ_lat {:.3f} m\n",
               pa.door_name, kind, innov.head<2>().norm(), innov.z() * 57.2958f, w_out, lost.d2, lost.view_llr, lost.p_lost, dx.head<2>().norm(), dx.z() * 57.2958f,
               std::sqrt(std::max(0.f, pa.pose.P(0, 0) + pa.pose.P(1, 1))));
}

// The geometry read with the pose, and the anchor's published width_m set to the CLEAR width (fix round 1): the controller's
// servo sizes the jambs and P(body fits) from it, and DOOR_1's wall gap (1.06) put each jamb ~8 cm outside the real frame.
void SpecificWorker::refresh_anchor_geometry(const rc::DoorInstance& inst, PassageAnchor& pa)
{
    anchor_geometry_from_door(inst, pa);
    if (pa.anchor.has_value() and G) pa.anchor->set_width(*G, pa.clear_width);
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
    pa.cross_was_claimed = false;
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
    // The twist's clock is observed FIRST, before any create / re-init (fix round 2): every interval below is mapped by the
    // current samples (rc::rt::SimClockRate::interval_s).
    if (const std::string rn = robot_node_name(); not rn.empty())
        if (const auto rnode = G->get_node(rn); rnode.has_value())
            if (const auto st = rc::rt::robot_twist_sim_stamps(*G, rnode.value()); st.has_value())
                sim_clock_.observe(st->first, st->second);

    // ── CREATE ───────────────────────────────────────────────────────────────────────────────────────────────
    if (not passage_anchor_.has_value())
    {
        for (auto& [id, inst] : instances)
        {
            // ★ Task 19 (user decision): the anchor is created on the CROSS claim only — offered only while the door is
            //   believed passable, i.e. OPEN. Approach and open run without one (the controller uses the room-frame
            //   standpoint, as before the anchor); most anchor failures built up while it was updated against a shut door
            //   and a swinging leaf. rc::door::anchor_creation_claim.
            const rc::pragmatic::PragmaticAffordance* aff =
                rc::door::anchor_creation_claim(inst.aff_cross.claimed(), inst.aff_approach.claimed()) ? &inst.aff_cross : nullptr;
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
    // Fix E: past a room switch during the cross the old door's cross is gone with its room, but the robot is still
    // crossing on this anchor (the controller servos it until `through`): for PUBLICATION the cross counts as claimed.
    const bool crs_pub = crs or pa.crossing_through;
    const bool opn = inst != nullptr and inst->aff_open.claimed();

    // ── REMOVE (a): the crossing's own claim ended SATISFIED — else PARK it for the retry (Task 9) ────────────────
    // ★try66 leg 2: every Unreachable of the cross removed the anchor ("the cross claim ended (completed)"), the re-claim
    //  hung a new one born stale from the room — 0.15 m and 18° from the corrected belief, and, with the robot already
    //  past the wall plane, with its normal flipped (the room init puts the robot on −y): the controller was told to turn
    //  155° in the doorway. A cross that ended without the robot crossing is the same passage, to be tried again.
    if (inst != nullptr and pa.cross_was_claimed and not crs)   // Task 19: the cross is the only claim an anchor serves
    {
        const auto& ended = inst->aff_cross;
        const auto outcome = ended.outcome();
        const std::string why = std::format("the {} claim ended ({}{}{})", ended.kind(),
                                            rc::pragmatic::PragmaticAffordance::state_name(ended.state()),
                                            outcome == rc::affordance::Outcome::None ? "" : ", ",
                                            rc::affordance::to_string(outcome));
        if (outcome == rc::affordance::Outcome::Satisfied)
        {
            remove_passage_anchor(why + ": the robot is through");
            return;
        }
        if (outcome != rc::affordance::Outcome::Satisfied)
        {
            pa.retry = true;
            std::print("door_concept: [anchor] {}@robot PARKED — {}: kept, predicted and corrected for this door's next "
                       "cross claim (a retry: the belief is not re-initialised)\n", pa.door_name, why);
        }
    }
    // ── REMOVE (c'): Task 21 — after the room switch the passage is over once this door is asked anything again ─────────
    //    (rc::door::passage_over_after_through; the predicted-window half is tested where the window is charged, below).
    if (inst != nullptr
        and rc::door::passage_over_after_through(pa.crossing_through,
                                                  app or crs or opn or inst->aff_close.claimed(), false))
    {
        remove_passage_anchor("the passage is over (through the room switch): this door is claimed again — a new passage gets a "
                              "fresh anchor");
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
    const bool claim_rose = crs and not pa.cross_was_claimed;   // Task 19: no approach→cross relink — the cross alone
    pa.cross_was_claimed    = crs;

    // ── LINK + RE-INIT ───────────────────────────────────────────────────────────────────────────────────────
    if (inst != nullptr)
    {
        if (not pa.run_known)   // the first KNOWN wall run is taken once (anchor_geometry_from_door)
            if (inst->ai2_initialized and inst->ai2_belief.params().wall_len > 0.f)
            {
                const float w = pa.width, cw = pa.clear_width, t = pa.thickness, h = pa.height;
                anchor_geometry_from_door(*inst, pa);
                pa.width = w; pa.clear_width = cw; pa.thickness = t; pa.height = h;   // the run only: the rest stays the initialisation's
            }
        if (const rc::pragmatic::PragmaticAffordance* aff = crs ? &inst->aff_cross : nullptr;
            aff != nullptr and aff->node_id() != 0)
        {
            const int epoch = aff->claim().has_value() ? aff->claim()->epoch : -1;
            if (aff->claim().has_value() and aff->claim()->agent_node_id != 0)
                pa.consumer_id = aff->claim()->agent_node_id;
            if (pa.reinit_on_next_claim)
            {
                // The first claim on the door re-identified after a room switch: a new passage's start.
                if (const auto r = anchor_from_room(*inst, &pa.axis_sign); r.has_value())
                {
                    std::print("door_concept: [anchor] {}@robot RE-INITIALISED from the room: first {} claim after "
                               "the room-switch relink\n", pa.door_name, aff->kind());
                    pa.pose = r.value();
                    pa.init_pose = r.value();   // Task 11 fix 4
                    pa.bias = {};
                    pa.view_lost_llr = 0.0;   // Task 17 add-on: a re-initialised belief, a new visibility record
                    refresh_anchor_geometry(*inst, pa);
                    pa.reinit_on_next_claim = false;
                    pa.retry = false;
                }
            }
            else if (aff->node_id() == pa.linked_aff)
            {
                if (epoch >= 0 and pa.linked_epoch >= 0 and epoch != pa.linked_epoch)
                {
                    if (pa.retry)
                        retry_broad_prior(*inst, pa, aff->kind());
                    else if (const auto r = anchor_from_room(*inst, &pa.axis_sign); r.has_value())
                    {
                        std::print("door_concept: [anchor] {}@robot RE-INITIALISED from the room: {} epoch {} → {}\n",
                                   pa.door_name, aff->kind(), pa.linked_epoch, epoch);
                        pa.pose = r.value();
                        pa.init_pose = r.value();   // Task 11 fix 4
                        pa.bias = {};
                        pa.view_lost_llr = 0.0;   // Task 17 add-on: a re-initialised belief, a new visibility record
                        refresh_anchor_geometry(*inst, pa);
                    }
                }
                else if (pa.retry and claim_rose)
                    retry_broad_prior(*inst, pa, aff->kind());
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
    // ★ Task 9 (try66 leg 1): the twist is per SIMULATION second on a simulator, so every interval it is integrated over —
    //   the prediction's dt and the captures' ages below — is producer SIM time (rc::rt::SimClockRate; the wall clock on
    //   real hardware). Integrated over the wall clock, the anchor's prediction ran 6 % long in translation and 14 % in
    //   rotation in a sim at ~0.94× real time: a 290° spin left it 15° / 1.1 m off with σ 0.3° / 7 mm, and its own
    //   overconfident prior then rejected the beams that would have corrected it — 0 of 332 leg-1 frames published.
    //   Fix round 1/2: the interval is the producer's SIM time between the pose's stamp and now, both mapped by the CURRENT
    //   consecutive twist samples (rc::rt::SimClockRate::interval_s) — a paused simulator integrates nothing, a restart
    //   cannot freeze it, and an anchor created before the first sample is predicted like any other.
    auto sim_age_s = [&](std::int64_t wall_then) { return float(sim_clock_.interval_s(wall_then, now)); };
    if (twist.has_value() and pa.pose.stamp_ms > 0 and now > pa.pose.stamp_ms)
    {
        Eigen::Vector3f var = Eigen::Vector3f::Zero();   // [var_adv, var_side, var_rot]; none stated ⇒ none added
        if (const auto v = G->get_attrib_by_name<robot_current_speed_variance_att>(robot.value());
            v.has_value() and v->get().size() >= 3)
            var = {v->get()[0], v->get()[1], v->get()[2]};
        const float dt = sim_age_s(pa.pose.stamp_ms);
        const Eigen::Vector3f x_before = pa.pose.x;
        pa.pose = rc::interaction::predict(pa.pose, twist.value(), var, dt);
        // Task 10: the persistent bias moves with the same step (its misfit part decorrelating as the view changes).
        rc::door::transport_bias(pa.bias, x_before, twist->linear.x() * dt, twist->linear.y() * dt, twist->angular.z() * dt, pa.width);
        // The last SEEN posterior is carried by the same odometry (fix round 1: published while predicted, kAnchorPredictMs).
        // Task 11 fix 4: the not-seen wait pose's carry, exclusive (rc::door::carry_wait_poses).
        rc::door::carry_wait_poses(pa.seen_once, pa.seen_pose, pa.init_pose, twist.value(), var, dt);
        // The held mask frame (fix 1) moves with the ROBOT (odometry), not with the anchor's posterior.
        rc::door::carry_held_mask(pa.cam_hold, twist->linear.x() * dt, twist->linear.y() * dt, twist->angular.z() * dt);
    }
    // ★ ONE CLOCK: the anchor is stamped with system-clock epoch ms (provenance::now_ms), the clock
    //   read_affordance_anchor's staleness check uses. predict() copies the TWIST's stamp; overwrite it.
    pa.pose.stamp_ms = now;

    // ── THIS DOOR'S WALL RUN in anchor coordinates (Task 7, fix 2) ───────────────────────────────────────────
    // The run is [0, wall_len] from the belief's wall origin along wall_u (door_fitter's associate_wall); the anchor's
    // origin is the opening's centre and its x-axis is ±wall_u (axis_sign). Unknown run (no wall associated, len 0)
    // ⇒ ⚑ the FLAGGED default: the aperture ± one width each side. Task 9: the run is read with the pose
    // (anchor_geometry_from_door), in ANCHOR coordinates, so it stays the same physical wall whatever instance the anchor
    // is linked to later; only the axis sign is re-derived here, for the leaf belief below.
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
    }
    if (pa.wall_u_max <= pa.wall_u_min)
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
        const float age_s = sim_age_s(pa.last_lidar_stamp);
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
    bool new_mask_frame = false;          // Task 8 fix 1: a NEW mask frame was consumed this cycle (door slice or not)
    std::int64_t mask_stamp = now;        // ...captured at
    if (mask_ingestor_ and robot_T_zed.has_value())
        if (const auto& pkt = mask_ingestor_->packet(); pkt.valid and pkt.frame_id != pa.last_mask_frame)
        {
            pa.last_mask_frame = pkt.frame_id;
            new_mask_frame = true;
            if (pkt.timestamp_ms > 0) mask_stamp = std::int64_t(pkt.timestamp_ms);
            const float age_s = pkt.timestamp_ms > 0 ? sim_age_s(std::int64_t(pkt.timestamp_ms)) : 0.f;
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
    const rc::interaction::RelPose prior_pose = pa.pose;   // for the frame record (fix 3)
    const rc::door::AnchorBias prior_bias = pa.bias;       // ...and the bias factor handed with it (record v4)
    rc::door::LeafPrior leaf;                              // the door's leaf belief (below), also recorded
    rc::door::AnchorStaging staging;                       // the update's two budgeted ray sets, for the frame record
    if (measured)
    {
        // The door's own P(open) (its leaf-angle posterior): what the visibility check predicts in the opening — a gap
        // or a leaf. Unknown this cycle ⇒ the last known value; never known ⇒ 0.5.
        // Right after an `open` command the leaf-angle posterior is withheld on purpose (phi_from_command: our own
        // request is not EVIDENCE that the door opened). For the visibility check — what the opening is EXPECTED to
        // look like — the commanded state is exactly the right prediction: π = passability under N(phi_est, σ_φ).
        const auto& pg = cfg_.pragmatic;
        // (Task 11 fix round 1: inst->interaction.p_open already carries this anchor's beams, through the φ posterior.)
        if (inst != nullptr and inst->interaction.p_open_known)
            pa.p_open_belief = std::clamp(inst->interaction.p_open, 0.f, 1.f);
        else if (inst != nullptr and inst->phi_from_command and pg.phi_sigma_rad > 0.f)
        {
            std::vector<std::pair<float, float>> curve;
            for (int i = -8; i <= 8; ++i)
            {
                const float z = 0.25f * float(i);
                curve.emplace_back(std::clamp(inst->phi_est + z * pg.phi_sigma_rad, 0.f, std::clamp(cfg_.phi_max_rad, 0.f, float(M_PI))),
                                   std::exp(-0.5f * z * z));
            }
            if (const auto p = rc::door::passable_prob(curve, inst->aperture.w, pg.robot_passage_width_m + pg.passage_margin_m);
                p.has_value())
                pa.p_open_belief = std::clamp(*p, 0.f, 1.f);
        }
        // The swung leaf, from the door's own belief (hinge, swing side, φ), in ANCHOR coordinates: the hinge jamb is
        // the near edge (−w/2 along wall_u) or the far one, and the anchor's axes are ±(wall_u, across_u) (axis_sign).
        // Fix round 4: a BELIEF, marginalised per frame by update_door_anchor — hinge × swing × φ over the door's whole
        // range, plus a "not as believed" branch (LeafPrior::p_not_as_believed). door_concept carries NO posterior over
        // hinge / swing (they are Openable config pins or a side inherited from the other room; the M2 hypotheses are
        // not built), so p_hinge = p_swing = 0.5: uniform, the data decide. Without an AI2 belief φ is unknown too.
        // (Task 11: the hinge now HAS a posterior — DoorInstance::hinge_llr, and inst->leaf.hinge is its MAP jamb. It is
        //  deliberately not passed as p_hinge yet: the anchor update is Task 10's, and its uniform hinge already lets the
        //  beams decide per frame. σ(hinge_llr) is the value to pass when that is revisited.)
        if (inst != nullptr)
        {
            const float near = inst->leaf.hinge == rc::door::HingeSide::Near ? 1.f : -1.f;
            const float phi_max = std::clamp(cfg_.phi_max_rad, 0.f, float(M_PI));   // the door model's hinge travel
            leaf.known     = true;
            leaf.hinge_u   = pa.axis_sign * (near > 0.f ? -0.5f : 0.5f) * pa.clear_width;   // fix round 1: a casing's inner face
            leaf.len       = pa.clear_width;
            leaf.dir_u     = pa.axis_sign * near;
            leaf.swing_y   = pa.axis_sign * inst->leaf.swing;
            leaf.phi_max   = phi_max;
            leaf.phi_known = inst->ai2_initialized;
            leaf.phi       = std::clamp(inst->phi_est, 0.f, phi_max);
            leaf.phi_sigma = std::max(0.f, pg.phi_sigma_rad);
            leaf.p_hinge   = 0.5f;
            leaf.p_swing   = 0.5f;
        }
        // Floor returns (below the wall band) are told apart by z: in the visibility they PASS, never end (Task 8).
        auto [post, e] = rc::door::update_door_anchor(pa.pose, pa.width, pa.thickness, pa.wall_u_min, pa.wall_u_max,
                                                      lidar_origin, rays, mask_pts, cam, pa.p_open_belief, leaf, kFloorBandM,
                                                      &staging, &pa.bias);
        post.stamp_ms = now;
        pa.pose = post;
        ev = e;
    }
    // ── Task 8 fix 1 (fix round 1): the last mask frame is HELD within the controller's staleness window ───────────
    //    A new frame replaces it (no door slice ⇒ no points: the camera looked and saw none). Its camera term is
    //    recomputed at the CURRENT posterior every measured cycle, so the published p_view stays the published pose's.
    if (new_mask_frame) rc::door::hold_mask_frame(pa.cam_hold, mask_pts, mask_stamp);
    if (measured)
    {
        ev.p_cam  = rc::door::held_camera_view(pa.cam_hold, now, rc::interaction::kAnchorStaleMs, cam, pa.width, pa.pose);
        ev.p_view = rc::door::combine_view(ev.p_lidar, ev.p_cam);
        // Task 17 add-on: a measured frame is a frame the belief predicted the doorway in — its p_view is evidence about it.
        if (pa.view_llr_stamp_ms > 0)
            pa.view_lost_llr += rc::door::view_lost_llr_step(ev.p_view, 1e-3 * double(now - pa.view_llr_stamp_ms),
                                                             1e-3 * double(rc::interaction::kAnchorPredictMs));
        // ★ Task 11 fix 2 (fix round 1): the anchor's own beams as evidence about the LEAF ANGLE, at the POSTERIOR it is
        //   about to publish, tempered by that posterior's p_view — scored under the swung leaf over door_concept's own φ
        //   grid, for the believed jamb and the other (rc::door::anchor_phi_loglik), and handed to the door's estimate_phi,
        //   which ADDS it to its φ likelihood once. NOT a shut-vs-not ratio fused into P(passable): an AJAR leaf passes
        //   that test (review: 30° read P 0.989) and fails this one, because passable_logit asks the clear span.
        // ★ Fix round 1 (I-3): only a frame whose pose is being PUBLISHED as seen (p_view ≥ kPublishViewLevel, the same level)
        //   is evidence about the leaf. An ALIASED pose (try70 leg 5: 60–86° off, p_view 0.33–0.48) scores the beams against a
        //   wrong opening, and the hinge and swing log-odds accumulate with no forgetting — one such stretch could flip and
        //   latch them. The publication test is the model's own judgement of whether this frame sees THIS doorway.
        if (not rays.empty() and inst != nullptr and inst->ai2_initialized and leaf.known
            and rc::door::anchor_frame_is_leaf_evidence(ev.p_view, kPublishViewLevel))
        {
            const auto g = rc::door::phi_grid(cfg_.phi_max_rad, cfg_.phi_step_rad);
            auto ac = rc::door::anchor_phi_loglik(pa.pose, pa.width, pa.thickness, leaf, lidar_origin, rays, ev.p_view,
                                                  g.lo, g.hi, g.n);
            inst->anchor_phi_ll       = std::move(ac.ll_this);
            inst->anchor_phi_ll_other = std::move(ac.ll_other);
            inst->anchor_phi_hinge    = inst->leaf.hinge;   // the jamb `leaf` was built from
            inst->anchor_swing_llr    = float(ac.swing_llr); // Task 12 fix D: this frame's swing evidence…
            inst->anchor_phi_swing    = inst->leaf.swing;    // …relative to the swing `leaf` was built from
        }
        if (not pa.az_logged and ev.az_step > 0.f)
        {
            pa.az_logged = true;
            std::print("door_concept: [anchor] {}@robot: the bpearl's measured azimuth step is {:.3f} deg ({} rays this frame)\n",
                       pa.door_name, ev.az_step * 180.f / float(M_PI), rays.size());
        }
    }
    pa.view_llr_stamp_ms = now;   // every cycle: unmeasured time passes and carries no evidence (Task 17 add-on)
    // ── Task 8 fix 3: the frame record (etc/anchor_frames.bin) — every measured cycle, inputs and outputs ──────────
    if (measured)
    {
        rc::door::rec::Frame fr;
        fr.t_ms = now;
        fr.prior = prior_pose;
        fr.bias  = prior_bias;
        fr.width = pa.width; fr.thickness = pa.thickness; fr.wall_u_min = pa.wall_u_min; fr.wall_u_max = pa.wall_u_max;
        fr.floor_below = kFloorBandM;
        fr.lidar_origin = lidar_origin;
        fr.staging = std::move(staging);
        fr.mask = mask_pts;
        fr.cam = cam;
        fr.leaf = leaf;
        fr.pi_open = pa.p_open_belief;
        fr.post = pa.pose;
        fr.p_lidar = ev.p_lidar; fr.p_cam = ev.p_cam; fr.p_view = ev.p_view;
        fr.lidar_open = ev.lidar_open; fr.lidar_door = ev.lidar_door; fr.lidar_wall = ev.lidar_wall;
        const bool was_capped = anchor_frames_.capped();
        anchor_frames_.append(fr);
        if (anchor_frames_.capped() and not was_capped)
            std::print("door_concept: [anchor] etc/anchor_frames.bin reached its {} MB cap — frame recording stopped for this run\n",
                       rc::door::rec::kMaxBytes / (1024 * 1024));
    }

    // ── WRITE (Task 7, fix 3): only on a cycle whose update consumed evidence of a door IN VIEW ─────────────────
    // Between observations the belief is predicted (above) but not published; the edge then ages past
    // kAnchorStaleMs and the controller drops back to the room route. The anchor node itself stays.
    const bool publish = measured and ev.p_view >= kPublishViewLevel;
    if (measured and publish != pa.publishing)   // a transition is judged on a MEASURED cycle only
    {
        pa.publishing = publish;
        std::print("door_concept: [anchor] {}@robot {} publishing — P(door in view) {:.2f} (p_lidar {:.2f} = opening {:.2f} · "
                   "P(door) {:.2f} · walls {:.2f}; p_cam {:.2f})\n",
                   pa.door_name, publish ? "RESUMED" : "PAUSED", ev.p_view, ev.p_lidar, ev.lidar_open, ev.lidar_door,
                   ev.lidar_wall, ev.p_cam);
    }
    // ── Fix round 1 (user-approved): the softer rule near the door — while the CROSS is claimed and the doorway has been
    //    seen in this passage, the last seen posterior, carried by odometry with its covariance grown by the motion, is
    //    published for up to kAnchorPredictMs after the last sighting (rc::door::anchor_publish_decision). The anchor
    //    node's unseen_time says how old that sighting is (0 = seen): the controller tells seen from predicted by it.
    //    An unmeasured cycle changes nothing about what was last seen: a Seen mode is not rewritten (as before), a
    //    Predicted one goes on until its window ends.
    // ★ Task 12 fix B: the predicted window bounds the CHARGED unseen time — wall time weighted by P(out of the doorway band)
    //   at the predicted pose (rc::door::anchor_doorway_band / anchor_unseen_charge): inside the band "not seen" is what
    //   the geometry predicts, and the carried pose's covariance grows with the motion, not with the wait.
    if (publish) { pa.seen_once = true; pa.last_seen_ms = now; pa.seen_pose = pa.pose; pa.unseen_charge_ms = 0; }
    else if (pa.charge_stamp_ms > 0)
    {
        pa.unseen_charge_ms = rc::door::anchor_unseen_charge(pa.unseen_charge_ms, now - pa.charge_stamp_ms,
                                                             rc::door::anchor_doorway_band(pa.seen_pose, cfg_.pragmatic.cross_clearance_m));
        // Fix round 1 (I-1): an unseen second re-tests nothing, inside the band too — the carried pose's covariance grows
        // toward the anchor's MEASURED under-coverage (rc::interaction::grow_unseen), so the servo slows on an old sighting.
        if (pa.seen_once)
            rc::interaction::grow_unseen(pa.seen_pose, 1e-3f * float(now - pa.charge_stamp_ms),
                                         1e-3f * float(rc::interaction::kAnchorPredictMs));
    }
    pa.charge_stamp_ms = now;
    const auto mode = (measured or pa.mode == rc::door::AnchorPublish::Predicted)
        ? rc::door::anchor_publish_decision(publish, crs_pub, pa.seen_once, pa.unseen_charge_ms, rc::interaction::kAnchorPredictMs)
        : pa.mode;
    if (mode != pa.mode)
    {
        if (mode == rc::door::AnchorPublish::Predicted)
            std::print("door_concept: [anchor] {}@robot PREDICTED — not seen now, the cross is claimed: publishing the last seen "
                       "pose carried by odometry (for ≤ {} ms of unseen time OUTSIDE the doorway band; P(in band) now {:.2f})\n",
                       pa.door_name, rc::interaction::kAnchorPredictMs,
                       rc::door::anchor_doorway_band(pa.seen_pose, cfg_.pragmatic.cross_clearance_m));
        else if (pa.mode == rc::door::AnchorPublish::Predicted)
            std::print("door_concept: [anchor] {}@robot predicted publication ended — {}\n", pa.door_name,
                       mode == rc::door::AnchorPublish::Seen ? "seen again" : "unseen beyond kAnchorPredictMs, or the cross ended");
        pa.mode = mode;
    }
    // Task 21: after the room switch, once the predicted window is spent the anchor is no longer driven on — the passage is
    // over (the same kAnchorPredictMs bound that ended its publication; rc::door::passage_over_after_through).
    if (rc::door::passage_over_after_through(pa.crossing_through, false,
                                             pa.seen_once and pa.unseen_charge_ms > rc::interaction::kAnchorPredictMs))
    {
        remove_passage_anchor("the passage is over (through the room switch): the predicted window is spent");
        return;
    }
    bool wrote = false;
    // Final review I-2: every block is stamped monotonically; "drive on it" / "only look at it" is the node's drivable mark
    // (Anchor::set_drivable — written AFTER a drivable block, BEFORE a wait pose; interaction_anchor.h says why).
    if (measured and mode == rc::door::AnchorPublish::Seen)
        wrote = pa.anchor->write(*G, pa.pose) and pa.anchor->set_unseen_ms(*G, 0) and pa.anchor->set_drivable(*G, true);
    else if (mode == rc::door::AnchorPublish::Predicted)
    {
        rc::interaction::RelPose out = pa.seen_pose;
        out.stamp_ms = now;   // fresh by the controller's staleness rule; unseen_time says it is a prediction
        wrote = pa.anchor->write(*G, out) and pa.anchor->set_unseen_ms(*G, int(now - pa.last_seen_ms))
            and pa.anchor->set_drivable(*G, true);
    }
    else if (mode == rc::door::AnchorPublish::None and pa.last_seen_ms < now - rc::interaction::kAnchorStaleMs)
    {
        // Fix round 2 (try68): NOT seen, nothing to drive on — but the belief, predicted by odometry, still says where to
        // LOOK. Written NOT DRIVABLE (final review I-2: the drivable mark cleared FIRST, then the pose stamped `now` — an
        // older "born stale" stamp was re-sorted or DROPPED by the RT history ring, so the last predicted block stayed
        // fresh for ~kAnchorStaleMs): the controller never drives on it, and its wait turns toward the opening where it
        // is NOW, not where it was last published (try68: a pose
        // published once, 22 s and a full turn earlier — the controller turned toward it, faced a wall, and waited 40 s).
        // ★ Task 11 fix 4: the last SEEN pose carried by odometry, else the room initialisation carried the same way —
        //   NEVER pa.pose, the posterior p_view just refused to publish (try66 L2: it follows a wedge's odometry 81–89° off,
        //   and the wait-turn swung hard at the jamb toward it). rc::door::wait_pose.
        rc::interaction::RelPose out = rc::door::wait_pose(pa.seen_once, pa.seen_pose, pa.init_pose);
        out.stamp_ms = now;
        wrote = pa.anchor->set_drivable(*G, false) and pa.anchor->write(*G, out)
            and pa.anchor->set_unseen_ms(*G, int(std::min<std::int64_t>(now - pa.last_seen_ms, std::numeric_limits<int>::max())));
    }
    // ── Task 12 fix A: the SWUNG LEAF on the anchor, in anchor coordinates (interaction_anchor/anchor_leaf.h) — the door's
    //    own belief (MAP hinge, swing, φ and its posterior σ), as the anchor update's LeafPrior builds it, so the controller's
    //    guards know the obstacle the LiDAR self-filter hides and aim at the clear gap beside it. Not known (no leaf belief,
    //    φ not measured) ⇒ the attributes are REMOVED, never a sentinel.
    if (wrote)
    {
        // Fix round 1 (I-4): every hinge × swing hypothesis with its probability (σ of the door's hinge / swing log-odds),
        // so the guards weigh the minority leaf too and the cross line moves only as far as the MAP is believed.
        std::vector<rc::interaction::AnchorLeaf> al;
        if (inst != nullptr and inst->ai2_initialized and inst->interaction.phi_known)
        {
            const float near = inst->leaf.hinge == rc::door::HingeSide::Near ? 1.f : -1.f;
            const float phi_max = std::clamp(cfg_.phi_max_rad, 0.f, float(M_PI));
            const float sy = pa.axis_sign * inst->leaf.swing;
            al = rc::interaction::anchor_leaf_hypotheses(pa.axis_sign * (near > 0.f ? -0.5f : 0.5f) * pa.clear_width,
                                                         sy * 0.5f * pa.thickness, pa.axis_sign * near, sy, pa.clear_width,
                                                         std::clamp(inst->phi_est, 0.f, phi_max), inst->phi_sigma,
                                                         inst->hinge_llr, inst->swing_llr);
        }
        if (not al.empty() != pa.leaf_published)
        {
            pa.leaf_published = not al.empty();
            if (not al.empty())
                std::print("door_concept: [anchor] {}@robot carries its leaf: MAP hinge ({:+.2f},{:+.2f}) free edge ({:+.2f},{:+.2f}) "
                           "[anchor frame], phi {:.0f} deg σ {:.1f} deg, P(MAP) {:.2f} (hinge log-odds {:+.1f}, swing {:+.1f})\n",
                           pa.door_name, al[0].hinge.x(), al[0].hinge.y(), al[0].tip.x(), al[0].tip.y(), al[0].phi * 57.2958f,
                           al[0].sigma_phi * 57.2958f, al[0].p, inst->hinge_llr, inst->swing_llr);
            else
                std::print("door_concept: [anchor] {}@robot leaf withdrawn — no leaf angle measured\n", pa.door_name);
        }
        pa.anchor->set_leaves(*G, al);
    }
    const bool alive = wrote or G->get_node(pa.anchor->id()).has_value();
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
