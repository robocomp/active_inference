#include "room_swap.h"

#include "room_forget.h"
#include "seam_fusion.h"
#include "../../common/room_resolve/room_resolve.h"   // rc::room::is_proto, current_edge_room

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

RoomSwap::RoomSwap(std::shared_ptr<DSR::DSRGraph> live, std::shared_ptr<DSR::DSRGraph> mem, RoomEviction &evictor, RoomRecall &recall)
    : live_(std::move(live)), mem_(std::move(mem)), ev_(evictor), recall_(recall) {}

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

void RoomSwap::retire_all_late()
{
    for (const auto &m : late_) recall_.remove_recall_of(m);
    late_.clear();
}

void RoomSwap::step()
{
    for (auto it = late_.begin(); it != late_.end();)
    {
        // K's own late recall is the channel room_concept reports a failed late adoption on (valid = false), written AFTER it
        // re-births the room from it: it stays until the undo window closes. Only the twin's recall is retired here.
        if (state_ == State::Staged and *it == known_mem_) { ++it; continue; }
        const auto rec = live_->get_node("recall_" + RoomRecall::live_name_of(*it));
        if (not rec.has_value()) { it = late_.erase(it); continue; }
        const auto room = live_->get_node(RoomRecall::live_name_of(*it));
        if (room.has_value() and room->type() == "room" and live_->get_attrib_by_name<valid_att>(rec.value()).value_or(true))
        {
            recall_.remove_recall_of(*it);       // the room is live again: the recall did its job
            it = late_.erase(it);
        }
        else ++it;
    }
}

bool RoomSwap::swap(std::uint64_t known_room_id, const rc::recog::PoseMix &pose_in_k, std::uint64_t provisional_live_id)
{
    if (state_ == State::Staged) return known_id_ == known_room_id;      // already staged for this room: idempotent
    retire_all_late();                                                   // a stale recall (the twin's, after an undo) must not outlive this

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

    // the anchor: the crossed door of the room being left while it is live (the door of L nearest P's entry mirror), else the robot
    std::uint64_t anchor = robot->id();
    Mat::RTMat anchor_from_K = robot_from_K;
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
                    if (not is_door(*live_, d) or live_->get_attrib_by_name<parent_att>(d).value_or(0) != L->id()) continue;
                    const auto Td = ie->get_transformation_matrix(L->name(), d.name());
                    const auto ref = TM.has_value() ? TM : ie->get_transformation_matrix(L->name(), robot->name());
                    if (not Td.has_value() or not ref.has_value()) continue;
                    if (const double dist = (Td->translation() - ref->translation()).head<2>().norm(); dist < best_d)
                    {
                        const auto Tdr = ie->get_transformation_matrix(d.name(), robot->name());
                        if (not Tdr.has_value()) continue;
                        best_d = dist; anchor = d.id(); anchor_from_K = Mat::RTMat(Tdr.value() * robot_from_K);
                    }
                }
            }

    if (not twin.has_value() and anchor == robot->id())
        qInfo() << "[swap] no crossed door of the room being left survives: the late recall hangs under the robot";

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
    if (not twin.has_value() and anchor != robot->id())
    {
        const auto d = live_->get_node(anchor);
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
                    mouth = true;
                }
            if (not K_from_L.has_value())
                if (const auto robot_from_L = ie->get_transformation_matrix(robot->name(), Ln->name()); robot_from_L.has_value())
                    K_from_L = Mat::RTMat(K_from_robot * robot_from_L.value());
            if (K_from_L.has_value())
            {
                const std::string pre = ev_.memory_prefix(Ln.value());
                if (not mouth) C_chain = seam_cov_through_robot(K_from_robot, S_robot, K_from_L.value());
                carries.push_back({pre + d->name(), pre + Ln->name(), d->name(), K_from_L.value(), C_chain});
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

    // ── execute: stage, recall, discard, hold ──────────────────────────────────────────────────────────────────
    const std::string known_mem = K->name();
    std::string prior_subtype;
    if (twin.has_value())
    {
        prior_subtype = subtype_of(*mem_, twin.value());
        auto t = twin.value();
        mem_->add_or_modify_attrib_local<object_subtype_att>(t, std::string(kStaged));
        if (not mem_->update_node(t)) return false;
    }
    const auto unstage = [&] {
        if (not twin.has_value()) return;
        if (const auto t = mem_->get_node(twin->id()); t.has_value()) restore_subtype(*mem_, t.value(), prior_subtype);
    };
    if (recall_.post_late(anchor, known_mem, anchor_from_K, pack_pose_prior(pose_in_k)) == 0)
    {
        unstage();
        return false;
    }
    late_.insert(known_mem);
    Candidate c;
    c.discard = true;
    c.new_room_id = P->id();
    c.new_room_name = P->name();
    if (not ev_.evict(c).ok)
    {
        recall_.remove_recall_of(known_mem);
        late_.erase(known_mem);
        unstage();
        return false;
    }
    ev_.hold(known_mem);

    carry_ = std::move(carries);
    state_ = State::Staged;
    known_mem_ = known_mem;
    known_id_ = known_room_id;
    promoted_case_ = twin.has_value();
    prov_twin_ = twin.has_value() ? twin->name() : std::string{};
    prov_twin_subtype_ = prior_subtype;
    twin_from_known_ = twin_from_K;
    robot_from_known_ = robot_from_K;
    qInfo().noquote() << QString::fromStdString(std::format("[swap] STAGED: {} is {} (case {}); late recall under {}",
            P->name(), known_mem, promoted_case_ ? "B, twin " + prov_twin_ + " staged" : std::string("A"), anchor == robot->id() ? robot->name() : "the crossed door"));
    return true;
}

bool RoomSwap::undo()
{
    if (state_ != State::Staged) return false;
    recall_.remove_recall_of(known_mem_);
    late_.erase(known_mem_);
    if (promoted_case_)
    {
        if (const auto t = mem_->get_node(prov_twin_); t.has_value()) restore_subtype(*mem_, t.value(), prov_twin_subtype_);
        // the provisional room comes back through the same mechanism: a recall of its twin, posted under the robot.
        // T(robot <- twin) = T(robot <- K) T(K <- twin): the robot's pose in K as room_concept now holds it when K is live
        // again, else the pose handed over at the swap.
        const auto robot = robot_node();
        if (robot.has_value() and twin_from_known_.has_value())
        {
            std::optional<Mat::RTMat> robot_from_K = robot_from_known_;
            if (const auto live_k = live_->get_node(RoomRecall::live_name_of(known_mem_)); live_k.has_value())
                if (const auto T = live_->get_inner_eigen_api()->get_transformation_matrix(robot->name(), live_k->name()); T.has_value())
                    robot_from_K = T;
            if (robot_from_K.has_value())
                if (recall_.post_late(robot->id(), prov_twin_, Mat::RTMat(robot_from_K.value() * twin_from_known_.value()), {}) != 0)
                    late_.insert(prov_twin_);
        }
    }
    ev_.release(known_mem_);
    carry_.clear();
    qInfo().noquote() << QString::fromStdString(std::format("[swap] UNDONE: {} is not {}", prov_twin_.empty() ? "the provisional room" : prov_twin_, known_mem_));
    state_ = State::Entering;
    known_mem_.clear(); prov_twin_.clear(); prov_twin_subtype_.clear();
    known_id_ = 0; promoted_case_ = false;
    twin_from_known_.reset(); robot_from_known_.reset();
    return true;
}

void RoomSwap::on_room_left(const std::string &room)
{
    if (state_ != State::Staged) return;
    if (not room.empty() and room != known_mem_ and room != RoomRecall::live_name_of(known_mem_)) return;
    retire_all_late();
    for (const auto &k : std::vector<Carry>(carry_))              // before the twin goes: its face is the passage's second end meanwhile
        if (not carry_over(k.prev_face, known_id_))
            qWarning().noquote() << QString::fromStdString(std::format("[swap] carry-over of {} into {} failed", k.prev_face, known_mem_));
    carry_.clear();
    if (promoted_case_) drop_subtree(prov_twin_);                 // the undo window is closed: the draft goes
    ev_.release(known_mem_);
    qInfo().noquote() << QString::fromStdString(std::format("[swap] COMMITTED: left {}; {}", known_mem_,
            promoted_case_ ? "staged twin " + prov_twin_ + " deleted" : std::string("nothing staged in memory")));
    state_ = State::Entering;
    known_mem_.clear(); prov_twin_.clear(); prov_twin_subtype_.clear();
    known_id_ = 0; promoted_case_ = false;
    twin_from_known_.reset(); robot_from_known_.reset();
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

bool RoomSwap::carry_over(const std::string &prev_room_face, std::uint64_t known_room_id)
{
    std::optional<DSR::Node> K;
    for (const auto &r : mem_->get_nodes_by_type("room"))
        if (mem_->get_attrib_by_name<room_id_att>(r).value_or(0) == known_room_id and not is_staged_room(*mem_, r)) { K = r; break; }
    const auto prev = mem_->get_node(prev_room_face);
    if (not K.has_value() or not prev.has_value()) return false;

    // the passage this face belongs to (case B: the one the twin's face hung from; case A: written by the eviction)
    std::optional<DSR::Node> pass;
    for (const auto &pn : mem_->get_nodes_by_type("metaconcept"))
        if (subtype_of(*mem_, pn) == "passage")
            for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(pn, "match"))
                if (e.to() == prev->id()) pass = pn;
    const auto face_of_k = [&](const DSR::Node &p) -> std::optional<DSR::Node> {
        for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(p, "match"))
            if (const auto f = mem_->get_node(e.to()); f.has_value() and is_descendant(*mem_, f->id(), K->name())) return f;
        return std::nullopt;
    };
    if (pass.has_value() and face_of_k(pass.value()).has_value()) return true;       // already carried: idempotent

    const auto it = std::ranges::find(carry_, prev_room_face, &Carry::prev_face);
    if (it == carry_.end()) return false;                                          // no seam for it: nothing is invented
    auto ie = mem_->get_inner_eigen_api();
    const auto F = ie->get_transformation_matrix(it->prev_room, prev_room_face);     // the previous room's face, in its own frame
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
    if (state_ != State::Staged or cand.discard) return;
    if (const auto old = live_->get_node(cand.old_room_id); old.has_value())
        if (ev_.memory_prefix(old.value()) + old->name() == known_mem_)
        {
            ev_.release(known_mem_);          // the robot is leaving K: the eviction must be able to write it. NOTHING is deleted yet.
            leaving_ = true;
        }
}

void RoomSwap::on_evicted(const Candidate &cand, bool ok)
{
    if (cand.discard) return;
    if (ok)
    {
        if (leaving_) on_room_left(known_mem_);          // the eviction stood: the undo window closes, the staged twin goes
        else if (state_ == State::Entering) retire_all_late();
    }
    else if (leaving_ and state_ == State::Staged)
        ev_.hold(known_mem_);                            // it failed: the robot is still in K's undo window
    leaving_ = false;
}

int RoomSwap::sweep_stale()
{
    int n = 0;
    for (const auto &r : mem_->get_nodes_by_type("room"))
        if (is_staged_room(*mem_, r)) n += drop_subtree(r.name());
    for (const auto &node : live_->get_nodes_by_type("metaconcept"))
        if (node.name().starts_with("recall_") and live_->get_attrib_by_name<recall_late_att>(node).value_or(false))
            if (live_->delete_node(node.id())) ++n;
    if (state_ == State::Staged) ev_.release(known_mem_);
    late_.clear();
    leaving_ = false;
    state_ = State::Entering;
    known_mem_.clear(); prov_twin_.clear(); prov_twin_subtype_.clear();
    known_id_ = 0; promoted_case_ = false;
    twin_from_known_.reset(); robot_from_known_.reset();
    carry_.clear();
    return n;
}
}   // namespace ltsm
