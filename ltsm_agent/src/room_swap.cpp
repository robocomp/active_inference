#include "room_swap.h"

#include "room_forget.h"
#include "room_recall.h"      // RoomRecall::live_name_of
#include "seam_fusion.h"
#include "../../common/room_resolve/room_resolve.h"   // rc::room::is_proto, current_edge_room, robot_rt_room

#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/core/types/type_checking/dsr_attr_name.h>

#include <QDebug>
#include <QString>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <format>
#include <limits>
#include <unordered_set>

namespace ltsm
{
namespace
{
constexpr const char *kStaged = "staged";

Eigen::Vector3d se2_of(const Mat::RTMat &T) { return {T.translation().x(), T.translation().y(), std::atan2(T.linear()(1, 0), T.linear()(0, 0))}; }
ltsm::seam::Measured meas(const Mat::RTMat &T, const Eigen::Matrix3d &C) { return {se2_of(T), C}; }

/// Covariance of T(K <- previous room) = T(K <- robot) . T(robot <- previous room) when only the robot's pose in K is uncertain:
/// the YAW LEVER ARM (robot yaw sigma x distance to the door) comes through the SE(2) composition Jacobian.
Eigen::Matrix3d seam_cov_through_robot(const Mat::RTMat &K_from_robot, const Eigen::Matrix3d &C_robot, const Mat::RTMat &K_from_prev)
{
    const Mat::RTMat robot_from_prev(K_from_robot.inverse() * K_from_prev);
    return ltsm::seam::compose_cov(meas(K_from_robot, C_robot), meas(robot_from_prev, Eigen::Matrix3d::Zero())).cov;
}

/// T(K <- robot) from the robot's pose in K, as the recogniser's joint posterior holds it (SE(2): x, y, yaw).
Mat::RTMat se2_to_rtmat(double x, double y, double th)
{
    Mat::RTMat T = Mat::RTMat::Identity();
    T.linear() = Eigen::AngleAxisd(th, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    T.translation() = Eigen::Vector3d(x, y, 0.0);
    return T;
}

/// recall_pose_prior: 10 floats per component [w,x,y,th,Sxx,Sxy,Sxth,Syy,Syth,Sthth].
std::vector<float> pack_pose_prior(const rc::recog::PoseMix &m)
{
    std::vector<float> out;
    for (const auto &g : m)
        for (const float f : {g.w, g.mu.x(), g.mu.y(), g.mu.z(), g.S(0, 0), g.S(0, 1), g.S(0, 2), g.S(1, 1), g.S(1, 2), g.S(2, 2)})
            out.push_back(f);
    return out;
}

std::string subtype_of(DSR::DSRGraph &g, const DSR::Node &n)
{
    return g.get_attrib_by_name<object_subtype_att>(n).value_or(std::string{});
}

/// Put a staged twin's subtype back as it was: an empty one means the attribute did not exist, so it is removed.
void restore_subtype(DSR::DSRGraph &g, DSR::Node t, const std::string &prior)
{
    if (prior.empty()) g.remove_attrib_local<object_subtype_att>(t);
    else g.add_or_modify_attrib_local<object_subtype_att>(t, std::string(prior));
    g.update_node(t);
}
}   // namespace

RoomSwap::RoomSwap(std::shared_ptr<DSR::DSRGraph> live, std::shared_ptr<DSR::DSRGraph> mem, RoomEviction &evictor)
    : live_(live), mem_(mem), ev_(evictor), proposal_(live, mem) {}

std::optional<DSR::Node> RoomSwap::robot_node() const
{
    const auto robots = live_->get_nodes_by_type("robot");
    return robots.empty() ? std::nullopt : std::optional<DSR::Node>(robots.front());
}

int RoomSwap::drop_subtree(const std::string &memory_room)
{
    // the walk (RT + parent attribute, passages with a face in it, deepest first) is plan_forget's
    auto plan = plan_forget(*mem_, memory_room, true);
    if (not plan.has_value()) return 0;
    // A passage with a face OUTSIDE the twin (the previous room's) survives with its crossing history: only the twin's own face
    // goes (the node deletion drops the match edge). Passages wholly inside the twin go with it.
    std::unordered_set<std::uint64_t> inside(plan->subtree.begin(), plan->subtree.end());
    inside.insert(plan->room_id);
    std::erase_if(plan->passages, [&](std::uint64_t pid) {
        const auto p = mem_->get_node(pid);
        return p.has_value() and std::ranges::any_of(DSR::DSRGraph::get_node_edges_by_type(p.value(), "match"),
                                                      [&](const auto &e) { return not inside.contains(e.to()); }); });
    return forget(*mem_, plan.value());
}

RoomSwap::Event RoomSwap::step()
{
    if (state_ == State::Adopted) { retry_carries(); return Event::None; }
    if (state_ != State::Staged) return Event::None;
    // room_concept's verdict first: an accepted proposal stands even if the robot moves on in the same cycle (it then leaves
    // the ADOPTED room, which the eviction path handles).
    switch (proposal_.verdict())
    {
        case RoomProposal::Verdict::Accepted: commit(); return Event::Accepted;
        case RoomProposal::Verdict::Rejected: abandon("REJECTED by room_concept"); return Event::Rejected;
        case RoomProposal::Verdict::Gone:     abandon("the proposal or the provisional room left the live graph"); return Event::Withdrawn;
        case RoomProposal::Verdict::Pending:  break;
    }
    // Review Focus 4: the robot left P before a verdict. room_concept's fact (the robot's live RT room), not ltsm's `current`;
    // unknown (none or several live edges, a hand-over) decides nothing, and neither does the room the robot was ALREADY in at
    // swap time (a proposal made before room_concept's switch into P must not withdraw itself): only a change AWAY from it,
    // to a room that is not P.
    if (const auto r = rc::room::robot_rt_room(*live_); r.has_value() and r.value() != prov_id_ and r != rt_at_swap_)
    {
        abandon("the robot left the provisional room before a verdict");
        return Event::Withdrawn;
    }
    return Event::None;
}

bool RoomSwap::swap(std::uint64_t known_room_id, const rc::recog::PoseMix &pose_in_k, std::uint64_t provisional_live_id, bool withdraw_crossing)
{
    if (state_ == State::Staged) return known_id_ == known_room_id;      // already proposed for this room: idempotent
    if (state_ == State::Adopted) return false;                          // the robot is still in an adopted room

    // the recognised room K: an archived room that is not a staged draft
    std::optional<DSR::Node> K;
    for (const auto &r : mem_->get_nodes_by_type("room"))
        if (mem_->get_attrib_by_name<room_id_att>(r).value_or(0) == known_room_id and not is_staged_room(*mem_, r)) { K = r; break; }
    if (not K.has_value() or pose_in_k.empty()) return false;
    const auto robot = robot_node();
    if (not robot.has_value()) return false;

    // the provisional room P: the one the recogniser names, else a proto hung from the robot
    std::optional<DSR::Node> P;
    if (provisional_live_id != 0) P = live_->get_node(provisional_live_id);
    else
        for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(robot.value(), "RT"))
            if (const auto n = live_->get_node(e.to()); n.has_value() and n->type() == "room" and rc::room::is_proto(*live_, n->id()))
            { P = n; break; }
    if (not P.has_value() or P->type() != "room") return false;

    // CASE B iff memory already holds the provisional room's twin
    std::optional<DSR::Node> twin;
    if (const auto rid = live_->get_attrib_by_name<room_id_att>(P.value()); rid.has_value())
        if (const auto t = mem_->get_node(std::format("r{}_{}", rid.value(), P->name())); t.has_value() and not is_staged_room(*mem_, t.value()))
            twin = t;

    // the robot in K (the strongest component of the mixture)
    const auto best = std::ranges::max_element(pose_in_k, {}, &rc::recog::PoseGauss::w);
    const Mat::RTMat K_from_robot = se2_to_rtmat(best->mu.x(), best->mu.y(), best->mu.z());
    const Mat::RTMat robot_from_K = K_from_robot.inverse();
    auto ie = live_->get_inner_eigen_api();

    // T(P <- K), the proposal's RT edge: P's frame <- the robot <- K (the recognised pose)
    const auto P_from_robot = ie->get_transformation_matrix(P->name(), robot->name());
    if (not P_from_robot.has_value()) return false;
    const Mat::RTMat P_from_K(P_from_robot.value() * robot_from_K);

    // the crossed door of the room being left while it is live (the door of L nearest P's entry mirror): the carry-over seam
    std::uint64_t crossed = 0;
    if (not twin.has_value())
        if (const auto cur = rc::room::current_edge_room(*live_); cur.has_value() and cur.value() != P->id())
            if (const auto L = live_->get_node(cur.value()); L.has_value())
            {
                std::optional<DSR::Node> M;
                for (const auto &d : live_->get_nodes_by_type("object"))
                    if (is_door(*live_, d) and live_->get_attrib_by_name<parent_att>(d).value_or(0) == P->id()) { M = d; break; }
                const auto TM = M.has_value() ? ie->get_transformation_matrix(L->name(), M->name()) : std::nullopt;
                double best_d = std::numeric_limits<double>::infinity();
                for (const auto &d : live_->get_nodes_by_type("object"))
                {
                    // a door OF L at any depth: door_concept hangs a door from its WALL; only the entry mirror hangs from the room
                    // itself (a parent==L test kept only that mirror, i.e. the door the robot came IN by -- recog wiring test 10-24)
                    if (not is_door(*live_, d) or not is_descendant(*live_, d.id(), L->name())) continue;
                    const auto Td = ie->get_transformation_matrix(L->name(), d.name());
                    const auto ref = TM.has_value() ? TM : ie->get_transformation_matrix(L->name(), robot->name());
                    if (not Td.has_value() or not ref.has_value()) continue;
                    if (const double dist = (Td->translation() - ref->translation()).head<2>().norm(); dist < best_d)
                    { best_d = dist; crossed = d.id(); }
                }
            }

    if (not twin.has_value() and crossed == 0)
        qInfo() << "[swap] no crossed door of the room being left survives: no carry-over seam from it";

    // case B: T(K <- twin) now, while the provisional room is still live and the robot chain resolves
    std::optional<Mat::RTMat> twin_from_K;
    if (twin.has_value())
    {
        const auto robot_from_P = ie->get_transformation_matrix(robot->name(), P->name());
        if (not robot_from_P.has_value()) return false;
        twin_from_K = Mat::RTMat(K_from_robot * robot_from_P.value());
    }

    // the seam T(K <- previous room), while the chains resolve, one per crossed door (see room_swap.h CARRY-OVER).
    std::vector<Carry> carries;
    const Eigen::Matrix3d S_robot = best->S.cast<double>();
    if (not twin.has_value() and crossed != 0)
    {
        const auto d = live_->get_node(crossed);
        const auto Lid = rc::room::current_edge_room(*live_);
        const auto Ln = Lid.has_value() ? live_->get_node(Lid.value()) : std::nullopt;
        if (d.has_value() and Ln.has_value())
        {
            std::optional<Mat::RTMat> K_from_L;
            // the MOUTH room_concept published on the provisional room (frame = the room being left): T(P <- L) is the inverse of
            // it, exact, no door node. Else the plain robot chain.
            const auto mt = live_->get_attrib_by_name<rt_translation_att>(P.value());
            const auto mr = live_->get_attrib_by_name<rt_rotation_euler_xyz_att>(P.value());
            const auto mf = live_->get_attrib_by_name<interaction_node_id_att>(P.value());
            const auto mc = live_->get_attrib_by_name<rt_covariance_att>(P.value());      // a mouth without its covariance cannot be weighed: not used
            Eigen::Matrix3d C_chain = Eigen::Matrix3d::Zero();
            Eigen::Matrix3d side = Eigen::Matrix3d::Zero();     // covariance of the robot-side half T(robot <- L) (the mouth's)
            bool mouth = false;
            if (mt.has_value() and mr.has_value() and mf.has_value() and mc.has_value() and mf.value() == Ln->id()
                and mt->get().size() >= 2 and mr->get().size() >= 3 and mc->get().size() == 36)
                if (const auto robot_from_P = ie->get_transformation_matrix(robot->name(), P->name()); robot_from_P.has_value())
                {
                    const Mat::RTMat L_from_P = se2_to_rtmat(mt->get()[0], mt->get()[1], mr->get()[2]);
                    K_from_L = Mat::RTMat(K_from_robot * robot_from_P.value() * L_from_P.inverse());
                    static constexpr int slot[3] = {0, 1, 5};      // [x y z rx ry rz], row-major
                    Eigen::Matrix3d Cm;
                    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) Cm(i, j) = mc->get()[slot[i] * 6 + slot[j]];
                    // K<-robot (robot cov) . robot<-P (exact) . inverse(mouth) (mouth cov): both enter through their Jacobians
                    const auto kp = ltsm::seam::compose_cov(meas(K_from_robot, S_robot), meas(robot_from_P.value(), Eigen::Matrix3d::Zero()));
                    C_chain = ltsm::seam::compose_cov(kp, ltsm::seam::inverse_cov(meas(L_from_P, Cm))).cov;
                    side = ltsm::seam::compose_cov(meas(robot_from_P.value(), Eigen::Matrix3d::Zero()),
                                                   ltsm::seam::inverse_cov(meas(L_from_P, Cm))).cov;
                    mouth = true;
                }
            if (not K_from_L.has_value())
                if (const auto robot_from_L = ie->get_transformation_matrix(robot->name(), Ln->name()); robot_from_L.has_value())
                    K_from_L = Mat::RTMat(K_from_robot * robot_from_L.value());
            if (K_from_L.has_value())
            {
                const std::string pre = ev_.memory_prefix(Ln.value());
                if (not mouth) C_chain = seam_cov_through_robot(K_from_robot, S_robot, K_from_L.value());
                carries.push_back({pre + d->name(), pre + Ln->name(), d->name(), K_from_L.value(), C_chain, Ln->name()});
                carries.back().side_cov = side;
            }
        }
    }
    if (twin.has_value() and twin_from_K.has_value())
        for (const auto &pn : mem_->get_nodes_by_type("metaconcept"))
        {
            if (subtype_of(*mem_, pn) != "passage") continue;
            std::optional<DSR::Node> mine, other;
            for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(pn, "match"))
                if (const auto f = mem_->get_node(e.to()); f.has_value())
                    (is_descendant(*mem_, f->id(), twin->name()) ? mine : other) = f;
            if (not mine.has_value() or not other.has_value()) continue;
            std::optional<DSR::Node> room;                       // the previous room: the face's room ancestor
            for (auto n = other; n.has_value() and not room.has_value();)
            {
                if (n->type() == "room") room = n;
                else if (const auto par = mem_->get_attrib_by_name<parent_att>(n.value()); par.has_value()) n = mem_->get_node(par.value());
                else break;
            }
            if (not room.has_value()) continue;
            const auto T = ltsm::seam_through_passage(*mem_, pn.name(), twin->name(), room->name());       // T(twin <- previous room)
            if (not T.has_value()) continue;
            const std::string rp = "r" + std::to_string(mem_->get_attrib_by_name<room_id_att>(room.value()).value_or(0)) + "_";
            carries.push_back({other->name(), room->name(), other->name().starts_with(rp) ? other->name().substr(rp.size()) : other->name(),
                               Mat::RTMat(twin_from_K.value() * T.value()),
                               seam_cov_through_robot(K_from_robot, S_robot, Mat::RTMat(twin_from_K.value() * T.value()))});
        }

    if (withdraw_crossing) carries.clear();        // no crossing happened: nothing to carry into K (K IS the previous room)
    // Every seam is T(K <- robot) . T(robot <- prev): keep the robot-side half, so the ACCEPTED verdict can rebuild the seam
    // from room_concept's T(P <- K) and the FIT's covariance (commit) instead of this prior mode's.
    for (auto &k : carries) k.robot_from_prev = Mat::RTMat(robot_from_K * k.known_from_prev);
    // ── execute: stage, propose, hold. NOTHING is written to memory here (C2.9: the carry-over is the accepted verdict's) ──
    const std::string known_mem = K->name();
    std::string prior_subtype;
    if (twin.has_value())
    {
        prior_subtype = subtype_of(*mem_, twin.value());
        auto t = twin.value();
        mem_->add_or_modify_attrib_local<object_subtype_att>(t, std::string(kStaged));
        if (not mem_->update_node(t)) return false;
    }
    if (proposal_.post(P->id(), known_mem, P_from_K, pack_pose_prior(pose_in_k)) == 0)
    {
        if (twin.has_value())
            if (const auto t = mem_->get_node(twin->id()); t.has_value()) restore_subtype(*mem_, t.value(), prior_subtype);
        return false;
    }
    ev_.hold(known_mem);

    carry_ = std::move(carries);
    for (const auto &k : carry_)
        carry_over(k.prev_face, known_room_id, true);   // the MAP face association, logged for the tour driver; nothing written
    state_ = State::Staged;
    K_from_robot_ = K_from_robot;
    known_mem_ = known_mem;
    known_id_ = known_room_id;
    prov_id_ = P->id();
    prov_name_ = P->name();
    prov_rid_ = live_->get_attrib_by_name<room_id_att>(P.value());
    rt_at_swap_ = rc::room::robot_rt_room(*live_);
    promoted_case_ = twin.has_value();
    withdraw_ = withdraw_crossing;
    prov_twin_ = twin.has_value() ? twin->name() : std::string{};
    prov_twin_subtype_ = prior_subtype;
    qInfo().noquote() << QString::fromStdString(std::format("[swap] PROPOSED: {} is {} (case {}); memory staged, nothing written until room_concept's verdict",
            P->name(), known_mem, promoted_case_ ? "B, twin " + prov_twin_ + " staged" : std::string("A")));
    return true;
}

void RoomSwap::clear()
{
    state_ = State::Entering;
    carry_.clear();
    known_mem_.clear(); prov_twin_.clear(); prov_twin_subtype_.clear(); prov_name_.clear();
    known_id_ = 0; prov_id_ = 0; prov_rid_.reset(); rt_at_swap_.reset(); promoted_case_ = false; withdraw_ = false;
}

bool RoomSwap::names_known(const std::string &room) const
{
    return room.empty() or room == known_mem_ or room == RoomRecall::live_name_of(known_mem_) or room == prov_name_;
}

void RoomSwap::abandon(std::string_view why)
{
    proposal_.withdraw();
    if (promoted_case_)
        if (const auto t = mem_->get_node(prov_twin_); t.has_value()) restore_subtype(*mem_, t.value(), prov_twin_subtype_);
    ev_.release(known_mem_);
    qInfo().noquote() << QString::fromStdString(std::format("[swap] WITHDRAWN: {} is not {} ({}); memory unstaged, {} stays a new room",
            prov_name_, known_mem_, why, prov_name_));
    clear();
}

void RoomSwap::retry_carries()
{
    std::erase_if(carry_, [&](const Carry &k) {
        if (not mem_->get_node(k.prev_face).has_value()) return false;     // its room is not archived yet: nothing to write
        const bool ok = carry_over(k.prev_face, known_id_);
        if (ok)
            qInfo().noquote() << QString::fromStdString(std::format("[swap] deferred carry-over of {} into {} written (the previous room is archived now)", k.prev_face, known_mem_));
        return ok; });
}

void RoomSwap::drop_withdrawn_passages(const std::string &twin)
{
    // Every passage that joins only K and P's twin was written by the eviction this crossing caused: a passage to a room that
    // does not exist. It goes, with K's face of it. A passage holding a face of any third room is a real hole memory already
    // knew (a known door the robot did not actually pass): it stays, and P's face goes with the twin.
    for (const auto &pn : mem_->get_nodes_by_type("metaconcept"))
    {
        if (subtype_of(*mem_, pn) != "passage") continue;
        bool in_twin = false, only_K_and_twin = true;
        std::vector<std::uint64_t> K_faces;
        for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(pn, "match"))
        {
            if (is_descendant(*mem_, e.to(), twin)) in_twin = true;
            else if (is_descendant(*mem_, e.to(), known_mem_)) K_faces.push_back(e.to());
            else only_K_and_twin = false;
        }
        if (not in_twin or not only_K_and_twin) continue;
        for (const auto f : K_faces) mem_->delete_node(f);
        qInfo().noquote() << QString::fromStdString(std::format("[swap] withdrawn crossing: passage {} and {} face(s) of {} removed from memory",
                                                                pn.name(), K_faces.size(), known_mem_));
        mem_->delete_node(pn.id());
    }
}

void RoomSwap::commit()
{
    // THE ACCEPTED TRANSFORM (Task 20). Every seam in carry_ was computed at swap time through ltsm's PRIOR (the recogniser's
    // best pose mode, its covariance S_robot). room_concept's fit may have corrected it (recog_proto6: a half turn) and states
    // the accepted T(P <- K) and the FIT's robot-in-K covariance on the proposal NODE, in the same update as "accepted"
    // (atomic; the edge is a different DDS topic and is NOT read). Each seam is REBUILT from them:
    //   T(K <- robot)_acc = C . T(K <- robot)_prior,  C = T_acc(P <- K)^-1 . T_prior(P <- K)
    //   seam = T(K <- robot)_acc . T(robot <- prev),  cov = compose(fit cov, robot-side cov)   (the prior mode's S is gone)
    // Absent ⇒ the verdict is malformed: NO seam is written (a wrong one would be permanent), nothing guessed.
    if (const auto acc = proposal_.accepted(); acc.has_value())
    {
        const Mat::RTMat C(acc->P_from_K.inverse() * proposal_.prior());
        const Mat::RTMat K_from_robot(C * K_from_robot_);
        for (auto &k : carry_)
        {
            k.known_from_prev = Mat::RTMat(K_from_robot * k.robot_from_prev);
            k.cov = ltsm::seam::compose_cov(meas(K_from_robot, acc->robot_K_cov), meas(k.robot_from_prev, k.side_cov)).cov;
        }
        const Eigen::Vector3d c = se2_of(C), a = se2_of(acc->P_from_K), pr = se2_of(proposal_.prior());
        qInfo().noquote() << QString::fromStdString(std::format(
            "[swap] accepted T(P<-K) = ({:.2f}, {:.2f}, {:.1f} deg), ltsm's prior ({:.2f}, {:.2f}, {:.1f} deg): {} seam(s) rebuilt "
            "(correction {:.2f}, {:.2f}, {:.1f} deg) with the fit's covariance", a.x(), a.y(), a.z() * 180.0 / M_PI, pr.x(), pr.y(),
            pr.z() * 180.0 / M_PI, carry_.size(), c.x(), c.y(), c.z() * 180.0 / M_PI));
    }
    else
    {
        qWarning().noquote() << QString::fromStdString(std::format(
            "[swap] MALFORMED accepted verdict: the proposal node carries no accepted T(P<-K)/covariance; {} carry-over seam(s) "
            "NOT written (never the prior)", carry_.size()));
        carry_.clear();
    }
    std::erase_if(carry_, [&](const Carry &k) { return carry_over(k.prev_face, known_id_); });
    for (const auto &k : carry_)
        qInfo().noquote() << QString::fromStdString(std::format("[swap] carry-over of {} into {} deferred: retried when the robot leaves it", k.prev_face, known_mem_));
    // P's twin goes (after the carry-over: its face is the passage's second end meanwhile). Case B: the staged one; case A: the
    // twin a promotion may have written for P while the verdict was open.
    std::string twin = prov_twin_;
    if (twin.empty() and prov_rid_.has_value())
        if (const auto t = mem_->get_node(std::format("r{}_{}", prov_rid_.value(), prov_name_)); t.has_value() and t->name() != known_mem_)
            twin = t->name();
    if (withdraw_ and not twin.empty()) drop_withdrawn_passages(twin);
    if (not twin.empty()) drop_subtree(twin);
    state_ = State::Adopted;
    if (withdraw_)
        qInfo().noquote() << QString::fromStdString(std::format("[swap] CROSSING WITHDRAWN: the robot never left {}; {} removed", known_mem_, prov_name_));
    qInfo().noquote() << QString::fromStdString(std::format("[swap] ACCEPTED: {} adopted as {}; {}; K held until the robot leaves it",
            prov_name_, known_mem_, twin.empty() ? std::string("no twin to drop") : "twin " + twin + " dropped"));
}

bool RoomSwap::undo()
{
    if (state_ != State::Staged) return false;
    abandon("undo requested by the recogniser");
    return true;
}

void RoomSwap::on_room_left(const std::string &room)
{
    if (state_ == State::Entering or not names_known(room)) return;
    if (state_ == State::Staged) { abandon("the robot left before a verdict"); return; }
    for (const auto &k : std::vector<Carry>(carry_))
        if (not carry_over(k.prev_face, known_id_))
            qWarning().noquote() << QString::fromStdString(std::format("[swap] carry-over of {} into {} failed", k.prev_face, known_mem_));
    ev_.release(known_mem_);
    proposal_.withdraw();
    qInfo().noquote() << QString::fromStdString(std::format("[swap] COMMITTED: left {}; hold released, proposal withdrawn", known_mem_));
    clear();
}

namespace
{
// FLAGGED model constants: the accuracy a door pose stored in memory carries (the same figures eviction uses for a seam
// read back from memory: it was itself a fused estimate).
constexpr double kDoorSigmaXY = 0.1, kDoorSigmaYaw = 0.03;

double wrap_pi(double a) { return std::atan2(std::sin(a), std::cos(a)); }

/// Shoelace area of a room's outline; 0 when none is stored.
double outline_area(DSR::DSRGraph &g, const DSR::Node &room)
{
    const auto px = g.get_attrib_by_name<delimiting_polygon_x_att>(room);
    const auto py = g.get_attrib_by_name<delimiting_polygon_y_att>(room);
    if (not px.has_value() or not py.has_value() or px->get().size() < 3 or px->get().size() != py->get().size()) return 0.0;
    double a = 0.0;
    const auto &x = px->get(), &y = py->get();
    for (std::size_t i = 0; i < x.size(); ++i) a += double(x[i]) * y[(i + 1) % x.size()] - double(x[(i + 1) % x.size()]) * y[i];
    return std::abs(a) * 0.5;
}
}   // namespace

bool RoomSwap::carry_over(const std::string &prev_room_face, std::uint64_t known_room_id, bool log_only)
{
    std::optional<DSR::Node> K;
    for (const auto &r : mem_->get_nodes_by_type("room"))
        if (mem_->get_attrib_by_name<room_id_att>(r).value_or(0) == known_room_id and not is_staged_room(*mem_, r)) { K = r; break; }
    const auto prev = mem_->get_node(prev_room_face);
    // FRESH -> FRESH crossing: the room being left (L) is archived only when the robot LEAVES the recognised room, so at swap time
    // its face is not in memory yet -- it is a LIVE node. The swap-time log line (log_only) reads it from the live graph; the
    // write (the release) always finds it in memory, after L's eviction.
    const auto live_it = std::ranges::find(carry_, prev_room_face, &Carry::prev_face);
    const bool live_only = not prev.has_value() and log_only and live_it != carry_.end() and not live_it->live_room.empty();
    if (not K.has_value() or (not prev.has_value() and not live_only))
    {
        if (log_only) qWarning().noquote() << QString::fromStdString(std::format("[swap] crossed door {}: no association (K {}, face in memory {}, live face {})",
                prev_room_face, K.has_value() ? "found" : "missing", prev.has_value() ? "yes" : "no", live_only ? "yes" : "no"));
        return false;
    }

    // the passage this face belongs to (case B: the one the twin's face hung from; case A: written by the eviction)
    std::optional<DSR::Node> pass;
    if (prev.has_value())
        for (const auto &pn : mem_->get_nodes_by_type("metaconcept"))
            if (subtype_of(*mem_, pn) == "passage")
                for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(pn, "match"))
                    if (e.to() == prev->id()) pass = pn;
    const auto face_of_k = [&](const DSR::Node &p) -> std::optional<DSR::Node> {
        for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(p, "match"))
            if (const auto f = mem_->get_node(e.to()); f.has_value() and is_descendant(*mem_, f->id(), K->name())) return f;
        return std::nullopt;
    };
    if (pass.has_value() and face_of_k(pass.value()).has_value())                     // already carried: idempotent
    {
        if (log_only)
            qInfo().noquote() << QString::fromStdString(std::format("[swap] crossed door {} <-> K face {}", prev_room_face, face_of_k(pass.value())->name()));
        return true;
    }

    const auto it = std::ranges::find(carry_, prev_room_face, &Carry::prev_face);
    if (it == carry_.end()) return false;                                          // no seam for it: nothing is invented
    // A passage is one hole between TWO rooms. This one already joins the previous room to a room that is not K, so K cannot be
    // written onto it (cmp_br2: a third face made the layout mark the seam WRONG). The face of the provisional room's own twin does
    // not count (case B: that face IS what K's replaces at the verdict). Memory's two faces stand and the conflict is left to the
    // map: nothing to write, so nothing to retry.
    const std::string own_twin = prov_rid_.has_value() ? std::format("r{}_{}", prov_rid_.value(), prov_name_) : std::string{};
    const auto room_of_face = [&](std::uint64_t id) -> std::optional<DSR::Node> {
        auto cur = mem_->get_node(id);
        for (int depth = 0; cur.has_value() and depth < 16; ++depth)
        {
            if (cur->type() == "room") return cur;
            const auto par = mem_->get_attrib_by_name<parent_att>(cur.value());
            if (not par.has_value() or par.value() == cur->id()) break;
            cur = mem_->get_node(par.value());
        }
        return std::nullopt; };
    if (pass.has_value() and std::ranges::any_of(DSR::DSRGraph::get_node_edges_by_type(pass.value(), "match"), [&](const auto &e) {
            const auto r = room_of_face(e.to());
            return e.to() != prev->id() and r.has_value() and r->name() != it->prev_room and r->name() != K->name()
                   and r->name() != own_twin and not is_staged_room(*mem_, r.value()); }))
    {
        qWarning().noquote() << QString::fromStdString(std::format("[swap] crossed door {}: {} already joins it to another room; {} is NOT written onto that passage",
                                                                   prev_room_face, pass->name(), K->name()));
        return true;
    }
    auto ie = mem_->get_inner_eigen_api();
    const auto F = live_only ? live_->get_inner_eigen_api()->get_transformation_matrix(it->live_room, it->base)
                             : ie->get_transformation_matrix(it->prev_room, prev_room_face);     // the previous room's face, in its own frame
    if (not F.has_value()) return false;
    const Eigen::Vector3d t = se2_of(Mat::RTMat(it->known_from_prev * F.value()));    // the aperture in K's frame, through the chain
    // the aperture's covariance: the seam's (robot lever arm, mouth) carried to the face, plus the error of the previous room's
    // stored face (the same FLAGGED prior a stored door carries)
    const Eigen::Matrix3d Cd = Eigen::Vector3d(kDoorSigmaXY * kDoorSigmaXY, kDoorSigmaXY * kDoorSigmaXY, kDoorSigmaYaw * kDoorSigmaYaw).asDiagonal();
    const Eigen::Matrix3d C = ltsm::seam::compose_cov(meas(it->known_from_prev, it->cov), meas(F.value(), Eigen::Matrix3d::Zero())).cov + Cd;
    const Eigen::Matrix3d S = C + Cd;
    const auto Sldlt = S.ldlt();
    const double log_norm = -1.5 * std::log(2.0 * M_PI) - 0.5 * Sldlt.vectorD().array().log().sum();

    // ASSOCIATION over {each unowned K door} U {none -> a new face}. FLAGGED: a MAP decision with a symmetric 0-1 loss.
    // Doors are pi-symmetric, so each is scored in both leaf orientations and the better one kept.
    std::unordered_set<std::uint64_t> owned;
    for (const auto &pn : mem_->get_nodes_by_type("metaconcept"))
        if (subtype_of(*mem_, pn) == "passage")
            for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(pn, "match")) owned.insert(e.to());
    struct Cand { DSR::Node door; Eigen::Vector3d pose; double dth, ll; };
    std::vector<Cand> cands;
    for (const auto &d : mem_->get_nodes_by_type("object"))
    {
        if (not is_door(*mem_, d) or owned.contains(d.id()) or not is_descendant(*mem_, d.id(), K->name())) continue;
        const auto there = ie->get_transformation_matrix(K->name(), d.name());
        if (not there.has_value()) continue;
        const Eigen::Vector3d D = se2_of(there.value());
        Cand best_c{d, D, 0.0, -std::numeric_limits<double>::infinity()};
        for (const double turn : {0.0, M_PI})
        {
            Eigen::Vector3d r(D.x() - t.x(), D.y() - t.y(), wrap_pi(D.z() + turn - t.z()));
            const double ll = log_norm - 0.5 * r.dot(Sldlt.solve(r));
            if (ll > best_c.ll) best_c = {d, D, turn, ll};
        }
        cands.push_back(best_c);
    }
    const double area = outline_area(*mem_, K.value());
    const double log_none = std::log(0.5) - std::log((area > 0.0 ? area : 1.0) * 2.0 * M_PI);
    std::optional<Cand> adopt;
    double best_ll = log_none;
    for (const auto &cd : cands)
        if (const double ll = std::log(0.5 / double(cands.size())) + cd.ll; ll > best_ll) { best_ll = ll; adopt = cd; }

    if (log_only)       // swap time: the MAP association, logged for the tour driver; nothing is written
    {
        qInfo().noquote() << QString::fromStdString(std::format("[swap] crossed door {} <-> K face {}", prev_room_face,
                                                                adopt.has_value() ? adopt->door.name() : std::string("new")));
        return true;
    }
    std::optional<std::uint64_t> fid;
    if (adopt.has_value())
    {
        // PRECISION FUSION of the door's stored pose with the chain pose, on the orientation branch consistent with the
        // previous face; written back through the door's own parent edge.
        const Eigen::Vector3d Db(adopt->pose.x(), adopt->pose.y(), t.z() + wrap_pi(adopt->pose.z() + adopt->dth - t.z()));
        // information-form fusion as a Kalman gain, no explicit inverse: x = t + C (C + Cd)^-1 (D - t)
        Eigen::Vector3d fused = t + C * Sldlt.solve(Eigen::Vector3d(Db - t));
        fused.z() = wrap_pi(fused.z());
        const Mat::RTMat fusedK = [&] { Mat::RTMat T = Mat::RTMat::Identity(); T.linear() = Eigen::AngleAxisd(fused.z(), Eigen::Vector3d::UnitZ()).toRotationMatrix(); T.translation() = Eigen::Vector3d(fused.x(), fused.y(), 0.0); return T; }();
        const auto par = mem_->get_attrib_by_name<parent_att>(adopt->door);
        const auto pn = par.has_value() ? mem_->get_node(par.value()) : std::nullopt;
        const auto K_from_par = pn.has_value() ? ie->get_transformation_matrix(K->name(), pn->name()) : std::nullopt;
        if (not K_from_par.has_value() or not pn.has_value())
        {
            qWarning().noquote() << QString::fromStdString(std::format("[swap] carry-over: adopted door {} has no resolvable parent in {}: NOT fused, nothing written", adopt->door.name(), K->name()));
            return false;
        }
        mem_->insert_or_assign_edge(DSR::Edge::create<RT_edge_type>(pn->id(), adopt->door.id(), RoomEviction::rt_attributes(Mat::RTMat(K_from_par->inverse() * fusedK))));
        fid = adopt->door.id();
    }
    else
    {
        const std::string name0 = "r" + std::to_string(known_room_id) + "_" + it->base;
        std::string name = name0;
        while (mem_->get_node(name).has_value()) name += "_x";
        DSR::Node f;
        f.type("object"); f.name(name);
        mem_->add_or_modify_attrib_local<object_subtype_att>(f, std::string("door"));
        mem_->add_or_modify_attrib_local<width_m_att>(f, mem_->get_attrib_by_name<width_m_att>(prev.value()).value_or(1.f));
        mem_->add_or_modify_attrib_local<parent_att>(f, K->id());
        mem_->add_or_modify_attrib_local<level_att>(f, mem_->get_node_level(K.value()).value_or(0) + 1);
        fid = mem_->insert_node(f);
        if (not fid.has_value()) return false;
        Mat::RTMat T = Mat::RTMat::Identity();
        T.linear() = Eigen::AngleAxisd(t.z(), Eigen::Vector3d::UnitZ()).toRotationMatrix();
        T.translation() = Eigen::Vector3d(t.x(), t.y(), 0.0);
        mem_->insert_or_assign_edge(DSR::Edge::create<RT_edge_type>(K->id(), fid.value(), RoomEviction::rt_attributes(T)));
    }
    if (not pass.has_value())
    {
        const auto id = ev_.mint_passage();
        if (id == 0) return false;
        mem_->insert_or_assign_edge(DSR::Edge::create<match_edge_type>(id, prev->id()));
        pass = mem_->get_node(id);
    }
    mem_->insert_or_assign_edge(DSR::Edge::create<match_edge_type>(pass->id(), fid.value()));
    qInfo().noquote() << QString::fromStdString(std::format("[swap] carry-over: {} ({}) is a passage of {} and the previous room ({})",
            mem_->get_node(fid.value()).value().name(), adopt.has_value() ? "adopted, fused" : "new face", K->name(), pass->name()));
    return true;
}

void RoomSwap::on_evict(const Candidate &cand)
{
    leaving_ = false;
    if (state_ == State::Entering or cand.discard) return;
    const auto old = live_->get_node(cand.old_room_id);
    const bool leaving_P = cand.old_room_id == prov_id_
                           or (old.has_value() and ev_.memory_prefix(old.value()) + old->name() == known_mem_);   // adopted: renamed to K
    if (not leaving_P) return;
    if (state_ == State::Staged)
    {   // Review Focus 4: no verdict yet. Unstage FIRST so this eviction writes P's twin as a NEW room.
        abandon("the robot left the provisional room before a verdict");
        return;
    }
    ev_.release(known_mem_);          // the robot is leaving the adopted K: the eviction must be able to write it
    leaving_ = true;
}

void RoomSwap::on_evicted(const Candidate &cand, bool ok)
{
    if (cand.discard) return;
    if (ok and leaving_) on_room_left(known_mem_);       // the eviction stood: the adopted room's window closes
    else if (ok and state_ == State::Adopted) retry_carries();   // e.g. the previous room archived after the verdict: its face now exists
    else if (not ok and leaving_ and state_ == State::Adopted)
        ev_.hold(known_mem_);                            // it failed: the robot is still in K
    leaving_ = false;
}

int RoomSwap::remove_owned()
{
    const int n = proposal_.id() != 0 and live_->get_node(proposal_.id()).has_value() ? 1 : 0;
    proposal_.withdraw();
    return n;
}

int RoomSwap::sweep_stale()
{
    int n = 0;
    for (const auto &r : mem_->get_nodes_by_type("room"))
        if (is_staged_room(*mem_, r)) n += drop_subtree(r.name());
    // late recalls are no longer posted (the proposal replaced them); an older build's leftovers still go
    for (const auto &node : live_->get_nodes_by_type("metaconcept"))
        if (node.name().starts_with("recall_") and live_->get_attrib_by_name<recall_late_att>(node).value_or(false))
            if (live_->delete_node(node.id())) ++n;
    n += proposal_.sweep_stale();
    if (state_ != State::Entering) ev_.release(known_mem_);
    leaving_ = false;
    clear();
    return n;
}
}   // namespace ltsm
