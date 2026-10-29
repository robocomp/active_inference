/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp — see room_scene_graph.h.
 */
// ADOPTING OR REJECTING A RECOGNITION PROPOSAL (room-transition protocol, spec C2.7;
// docs/superpowers/specs/2026-10-29-room-transition-protocol-design.md, case 2 step 7).
// ltsm's recogniser, convinced that the provisional room P is a remembered room K, posts ONE proposal under P and
// changes nothing else live (proposal_adoption.h has the node's contract). This file is room_concept's side:
//   1. A proposal this agent cannot judge is answered "rejected" at once: P is not this agent's provisional room any
//      more (RF1), K's name is held by another node so the rename could not happen (RF3), or the proposal is malformed.
//   2. Otherwise the SAME fit the late adoption used decides (RoomConcept::seed_reloc_prior + late_recall_verdict):
//      K's outline becomes the GIVEN layout — placed in the internal frame through the proposal's T(P←K), so the
//      internal frame and the pose do not jump — and the robot-in-K modes the relocaliser's prior. While it runs NOTHING
//      is written to the graph: P keeps its name, polygon, walls and frame (the robot's RT is published in P's frame as
//      before) and P's survey is suspended.
//   3. Rejected or withdrawn: RoomConcept restores the survey from its snapshot (end_hypothesis_fit) and "rejected" is
//      written (nothing on a vanished proposal). While the fit runs, the PUBLISHED pose is the survey's (plain SLAM's
//      shadow, PosePublisher), so nothing a consumer sees moves before the verdict. P vanished / the robot crossed out:
//      the fit is abandoned (the survey restored, then the crossing / re-birth proceeds as it would have without it).
//   4. Accepted: P becomes K IN PLACE — same node id renamed to K's live name (cortex rename, 43bfe16), room_id and
//      outline K's, P's own RT children re-expressed by T(K←P), the published frame becomes K's, proto edge removed,
//      the walls built/synced on K's outline — and "accepted" is written. T(K←P) is the FIT's (robot_K ∘ robot_P⁻¹), not
//      ltsm's prior, and it is written on the proposal NODE in the same update_node as "accepted" (Task 20: one fact,
//      every reader); no fitted pose ⇒ rejected, never the prior. A refused rename is a rejection.
// The proposal node is ltsm's: read, `state` written, never deleted. (This replaced the late recall's INSERT of a new
// room node; ltsm no longer posts late recalls, 4d9f7ae4.)
#include "room_scene_graph.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include <QDebug>

namespace rc
{

double RoomSceneGraph::recognition_log_odds(std::uint64_t room_id) const
{
    // ltsm's recogniser posts one `recognition` metaconcept: recog_room_ids (the candidates) and recog_posterior (P over them,
    // then P(new) LAST; ltsm_agent/src/room_recognizer.cpp). Its log-odds of K against "a new room" is the prior of the
    // adoption test (RoomConcept::seed_reloc_prior). Absent node / attribute / entry: no recognition evidence, 0.
    const auto n = G_->get_node(std::string("recognition"));
    if (not n.has_value() or n->type() != "metaconcept"
        or G_->get_attrib_by_name<object_subtype_att>(n.value()).value_or(std::string{}) != "recognition")
        return 0.0;
    const auto ids  = G_->get_attrib_by_name<recog_room_ids_att>(n.value());
    const auto post = G_->get_attrib_by_name<recog_posterior_att>(n.value());
    if (not ids.has_value() or not post.has_value()) return 0.0;
    const auto& iv = ids->get(); const auto& pv = post->get();
    if (pv.size() != iv.size() + 1) return 0.0;
    const auto it = std::ranges::find(iv, room_id);
    if (it == iv.end()) return 0.0;
    // Numerical guard: a posterior that underflowed to 0 is floored at the smallest normal float, not taken as -inf.
    const auto ln = [](float p) { return std::log(std::max(static_cast<double>(p), static_cast<double>(std::numeric_limits<float>::min()))); };
    const float pk = pv[static_cast<std::size_t>(it - iv.begin())], pn = pv.back();
    if (not std::isfinite(pk) or not std::isfinite(pn)) return 0.0;
    return ln(pk) - ln(pn);
}

std::uint64_t RoomSceneGraph::provisional_room() const
{
    // This agent's room, still surveyed (not promoted), and not the legacy crossing proto. The graph half of
    // "provisional" (the `proto` self-edge) is checked by rc::proposal::admit.
    return (room_node_created_ and dsr_room_id_ != 0 and startup_proto_ and not startup_promoted_ and proto_room_id_ == 0)
           ? dsr_room_id_ : 0;
}

bool RoomSceneGraph::begin_proposal_fit(const rc::proposal::Proposal& p)
{
    Eigen::Affine2f T_pub_int;
    { std::scoped_lock lk(gauge_mutex_); T_pub_int = T_pub_int_; }
    // K's frame in the internal one: internal ← P(published) ← K. The internal frame stays where it is; only the layout
    // the localiser is scored against changes, and RoomConcept snapshots the survey so a rejection costs nothing.
    const Eigen::Affine2f T_int_K = T_pub_int.inverse() * p.T_P_K;
    std::vector<Eigen::Vector2f> K_int;
    K_int.reserve(p.outline_K.size());
    for (const auto& v : p.outline_K) K_int.push_back(T_int_K * v);
    const auto modes = rc::proposal::modes_in(p.pose_prior, T_int_K);
    const double prior = recognition_log_odds(p.room_index);
    if (not room_concept_->begin_hypothesis_fit(std::move(K_int), modes, prior)) return false;
    prop_T_int_K_ = T_int_K;
    prop_judged_ = p;
    const auto pn = G_->get_node(p.parent);
    qWarning().noquote() << QString("[room][proposal] judging %1: %2 is remembered %3 — K's outline (%4 vertices) is the given "
                                    "layout, %5 pose mode(s) the relocaliser's prior, test starts at log-odds %6; the survey "
                                    "pose is published and nothing written until the verdict")
                                .arg(p.node_id).arg(QString::fromStdString(pn.has_value() ? pn->name() : std::string("?")))
                                .arg(QString::fromStdString(p.live_name)).arg(p.outline_K.size()).arg(modes.size())
                                .arg(prior, 0, 'f', 2);
    return true;
}

void RoomSceneGraph::end_proposal_fit(const char* why)
{
    // Back to P's survey exactly as it was (RoomConcept::end_hypothesis_fit restores layout, emission and belief; the
    // pose is plain SLAM's shadow, which never followed K). Nothing of P was written while the fit ran.
    room_concept_->end_hypothesis_fit();
    qInfo().noquote() << "[room][proposal] fit ended (" << why << "): P's survey resumes";
    prop_judged_.reset();
}

void RoomSceneGraph::abandon_proposal_fit(const char* why)
{
    if (not prop_judged_.has_value()) return;
    room_concept_->end_hypothesis_fit();   // the survey as it was: what a crossing / a re-birth then starts from
    qWarning().noquote() << QString("[room][proposal] %1 abandoned: %2").arg(QString::fromStdString(prop_judged_->live_name)).arg(why);
    prop_judged_.reset();
}

bool RoomSceneGraph::accept_proposal(const rc::RoomConcept::UpdateResult& res)
{
    auto p = *prop_judged_;
    const Eigen::Affine2f T_K_int = prop_T_int_K_.inverse();
    Eigen::Affine2f T_old;
    { std::scoped_lock lk(gauge_mutex_); T_old = T_pub_int_; }
    // THE ACCEPTED TRANSFORM (Task 20). ltsm's T(P←K) is only the prior the fit started from — recog_proto6: a half turn
    // off, which the fit corrected (3.12 m / 179°). The accepted fact is the fit's: the robot's pose in P as published this
    // frame (`res` carries the survey's pose, rc::publish::substitute_survey) and its fitted pose in K (the localiser's own
    // result, tracked against K) are one physical pose ⇒ T(P←K) = robot_P ∘ robot_K⁻¹. It re-expresses this agent's own
    // edges and is written on the proposal before "accepted", so every reader of the verdict reads this one fact.
    const Eigen::Affine2f T_P_K_prior = p.T_P_K;
    // Same internal frame for both poses: the survey (plain SLAM's shadow) is seeded from and compared against the
    // localiser's window pose directly (record_slam_shadow), and the only internal-frame move (reanchor_map_frame) is
    // suppressed while a start-up proto owns the published frame (freeze_internal_frame_), which P is.
    const auto fit = room_concept_->get_last_result();
    if (not fit.has_value() or not fit->ok)
    {
        // No fitted pose ⇒ no accepted transform: accepting would publish ltsm's prior as the fact. Reject instead.
        rc::proposal::answer(*G_, p.node_id, rc::proposal::kRejected);
        end_proposal_fit("no fitted pose at acceptance");
        qWarning().noquote() << QString("[room][proposal] %1 accepted by the fit but no fitted pose is available to state the "
                                        "accepted T(P<-K): REJECTED").arg(QString::fromStdString(p.live_name));
        return false;
    }
    p.T_P_K = rc::proposal::accepted_T_P_K(T_old * res.robot_pose, T_K_int * fit->robot_pose);
    // The fit's pose covariance, internal frame → K's (rotation only; θ unchanged): what T(P←K) inherits.
    Eigen::Matrix3f J = Eigen::Matrix3f::Identity();
    J.topLeftCorner<2, 2>() = T_K_int.linear();
    const Eigen::Matrix3f robot_K_cov = J * fit->covariance * J.transpose();
    {
        const Eigen::Affine2f d = T_P_K_prior.inverse() * p.T_P_K;   // the fit's correction of the prior, in K
        qWarning().noquote() << QString("[room][proposal] accepted T(P<-K) = (%1, %2, %3 deg); ltsm's prior (%4, %5, %6 deg); "
                                        "correction %7 m / %8 deg")
                                    .arg(p.T_P_K.translation().x(), 0, 'f', 2).arg(p.T_P_K.translation().y(), 0, 'f', 2)
                                    .arg(rc::proposal::angle_of(p.T_P_K) * 180.f / M_PI, 0, 'f', 1)
                                    .arg(T_P_K_prior.translation().x(), 0, 'f', 2).arg(T_P_K_prior.translation().y(), 0, 'f', 2)
                                    .arg(rc::proposal::angle_of(T_P_K_prior) * 180.f / M_PI, 0, 'f', 1)
                                    .arg(d.translation().norm(), 0, 'f', 2).arg(rc::proposal::angle_of(d) * 180.f / M_PI, 0, 'f', 1);
    }
    { std::scoped_lock lk(gauge_mutex_); T_pub_int_ = T_K_int; }
    const auto a = rc::proposal::adopt_in_place(*G_, p, dsr_robot_id_);
    if (not a.ok)
    {
        { std::scoped_lock lk(gauge_mutex_); T_pub_int_ = T_old; }
        rc::proposal::answer(*G_, p.node_id, rc::proposal::kRejected);
        end_proposal_fit("adoption refused");
        qWarning().noquote() << QString("[room][proposal] %1 accepted by the fit but NOT adopted (%2): rejected")
                                    .arg(QString::fromStdString(p.live_name)).arg(QString::fromStdString(a.why));
        return false;
    }
    room_concept_->keep_hypothesis_fit();   // K stays the given room
    note_room_name_issued(p.room_index);
    // The published-frame state of P's survey does not survive the frame change.
    reset_free_space(T_K_int);
    std::string reason;
    promote_startup_proto(reason, true);       // K's walls on K's outline (published frame now K's), planner re-seeded
    sync_walls_to_polygon(p.outline_K);        // rule 1: walls that already exist follow K's outline in place
    recalled_visit_ = true;                    // the remembered outline is not re-refined on this visit
    Recall r;
    r.name = p.live_name; r.room_index = p.room_index; r.poly_pub = p.outline_K;
    note_recall_memory(r);                     // P(explored) from what memory restores (room_explored.h)
    startup_event_ = "recalled";
    if (not rc::proposal::answer_accepted(*G_, p, p.T_P_K, robot_K_cov))   // T + covariance + "accepted": one update_node
        qWarning().noquote() << "[room][proposal] the accepted verdict could not be written (the proposal or its edge is gone)";
    prop_judged_.reset();
    qWarning().noquote() << QString("[room][proposal] ACCEPTED: room %1 is %2, adopted in place (same id; %3 own RT edge(s) "
                                    "re-expressed by T(K<-P); promotion: %4)")
                                .arg(dsr_room_id_).arg(QString::fromStdString(p.live_name)).arg(a.reexpressed)
                                .arg(QString::fromStdString(reason));
    return true;
}

bool RoomSceneGraph::step_proposal(const rc::RoomConcept::UpdateResult& res)
{
    if (G_ == nullptr) return false;
    if (prop_judged_.has_value())
    {
        // The fit ends by MODEL EVENTS, never by waiting: the proposal or P is gone (ltsm withdrew it, or re-proposed
        // under a new id), or this agent's room is no longer P (a crossing).
        const auto& p = *prop_judged_;
        const auto pn = G_->get_node(p.node_id);
        const bool still_pending = pn.has_value()
            and G_->get_attrib_by_name<state_att>(pn.value()).value_or(std::string(rc::proposal::kPending)) == rc::proposal::kPending;
        if (not G_->get_node(p.parent).has_value() or dsr_room_id_ != p.parent)
        {
            abandon_proposal_fit("P is gone or no longer this agent's room");
            return false;
        }
        if (not still_pending)
        {
            qWarning().noquote() << QString("[room][proposal] %1 withdrawn before the verdict: K's layout released, "
                                            "P's survey restored").arg(QString::fromStdString(p.live_name));
            end_proposal_fit("withdrawn");
            return false;
        }
        switch (room_concept_->late_recall_verdict())
        {
        case rc::RoomConcept::LateVerdict::Rejected:
            rc::proposal::answer(*G_, p.node_id, rc::proposal::kRejected);
            qWarning().noquote() << QString("[room][proposal] %1 REJECTED by the adoption fit: P stays a new room")
                                        .arg(QString::fromStdString(p.live_name));
            end_proposal_fit("rejected");
            return false;
        case rc::RoomConcept::LateVerdict::Accepted:
            return accept_proposal(res);
        default:
            return false;   // still judging
        }
    }
    const std::int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    if (now - prop_last_poll_ms_ < 1000) return false;
    prop_last_poll_ms_ = now;
    const auto standing = rc::proposal::pending(*G_);
    if (standing.empty()) return false;
    const auto mine = provisional_room();
    for (const auto& p : standing)
    {
        // Every room is this agent's (room_concept is the only room producer), so every pending proposal is answered.
        const auto adm = rc::proposal::admit(*G_, p, mine);
        if (adm == rc::proposal::Admission::Judge and prop_judged_.has_value()) continue;   // one at a time
        if (adm == rc::proposal::Admission::Judge and begin_proposal_fit(p)) continue;
        // PlainSlam off: the survey has no pose independent of K, so the fit could not run without moving what is
        // published (and could not be undone) — not judgeable here. NOTE: "rejected" here means "CANNOT JUDGE", not
        // "judged not K"; ltsm treats both alike (keeps P as a new room, rebases its evidence).
        const char* why = adm == rc::proposal::Admission::Judge ? "no independent survey pose to publish during the fit (plain SLAM not driving)"
                                                                : rc::proposal::to_string(adm);
        rc::proposal::answer(*G_, p.node_id, rc::proposal::kRejected);
        if (adm == rc::proposal::Admission::Judge)
            qCritical().noquote() << QString("[room][proposal] %1 (id %2) CANNOT BE JUDGED — answered rejected: %3. "
                                             "Recognition cannot adopt any room in this configuration (RoomShape.PlainSlam)")
                                         .arg(QString::fromStdString(p.live_name)).arg(p.node_id).arg(why);
        else
            qWarning().noquote() << QString("[room][proposal] %1 (id %2) rejected without a fit: %3")
                                        .arg(QString::fromStdString(p.live_name)).arg(p.node_id).arg(why);
    }
    return false;
}

}   // namespace rc
