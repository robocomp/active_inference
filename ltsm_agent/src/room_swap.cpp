#include "room_swap.h"

#include "room_forget.h"
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

    // the crossed door(s) in K's frame, while the chains resolve: robot-in-K (the relocalisation) through the live door (case A),
    // or K <- twin through the twin's face of each passage (case B). The previous room's face is the other end of the passage.
    std::vector<Carry> carries;
    if (not twin.has_value() and anchor != robot->id())
    {
        if (const auto d = live_->get_node(anchor); d.has_value())
            if (const auto L = rc::room::current_edge_room(*live_); L.has_value())
                if (const auto Ln = live_->get_node(L.value()); Ln.has_value())
                    carries.push_back({ev_.memory_prefix(Ln.value()) + d->name(), Mat::RTMat(anchor_from_K.inverse())});
    }
    if (twin.has_value() and twin_from_K.has_value())
    {
        auto mie = mem_->get_inner_eigen_api();
        for (const auto &pn : mem_->get_nodes_by_type("metaconcept"))
        {
            if (subtype_of(*mem_, pn) != "passage") continue;
            std::optional<DSR::Node> mine, other;
            for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(pn, "match"))
                if (const auto f = mem_->get_node(e.to()); f.has_value())
                    (is_descendant(*mem_, f->id(), twin->name()) ? mine : other) = f;
            if (not mine.has_value() or not other.has_value()) continue;
            if (const auto T = mie->get_transformation_matrix(twin->name(), mine->name()); T.has_value())
                carries.push_back({other->name(), Mat::RTMat(twin_from_K.value() * T.value())});
        }
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
    if (it == carry_.end()) return false;                                          // no pose for it: nothing is invented
    const Mat::RTMat T = it->in_known;
    const double half_w = 0.5 * mem_->get_attrib_by_name<width_m_att>(prev.value()).value_or(1.f);

    // K already has a door at the aperture that no passage owns (the live K was evicted into it with the re-born door): adopt it.
    std::optional<std::uint64_t> fid;
    auto ie = mem_->get_inner_eigen_api();
    std::unordered_set<std::uint64_t> owned;
    for (const auto &pn : mem_->get_nodes_by_type("metaconcept"))
        if (subtype_of(*mem_, pn) == "passage")
            for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(pn, "match")) owned.insert(e.to());
    double best = half_w;
    for (const auto &d : mem_->get_nodes_by_type("object"))
    {
        if (not is_door(*mem_, d) or owned.contains(d.id()) or not is_descendant(*mem_, d.id(), K->name())) continue;
        if (const auto there = ie->get_transformation_matrix(K->name(), d.name()); there.has_value())
            if (const double dist = (there->translation() - T.translation()).head<2>().norm(); dist < best) { best = dist; fid = d.id(); }
    }
    if (not fid.has_value())
    {
        std::string base = prev_room_face;
        if (base.size() > 1 and base[0] == 'r')
            if (const auto us = base.find('_'); us != std::string::npos and us > 1 and std::ranges::all_of(base.substr(1, us - 1), [](char ch) { return ch >= '0' and ch <= '9'; }))
                base = base.substr(us + 1);
        const std::string prefix = "r" + std::to_string(known_room_id) + "_";
        std::string name = prefix + base;
        while (mem_->get_node(name).has_value()) name += "_x";
        DSR::Node f;
        f.type("object"); f.name(name);
        mem_->add_or_modify_attrib_local<object_subtype_att>(f, std::string("door"));
        mem_->add_or_modify_attrib_local<width_m_att>(f, mem_->get_attrib_by_name<width_m_att>(prev.value()).value_or(1.f));
        mem_->add_or_modify_attrib_local<parent_att>(f, K->id());
        mem_->add_or_modify_attrib_local<level_att>(f, mem_->get_node_level(K.value()).value_or(0) + 1);
        fid = mem_->insert_node(f);
        if (not fid.has_value()) return false;
        mem_->insert_or_assign_edge(DSR::Edge::create<RT_edge_type>(K->id(), fid.value(), RoomEviction::rt_attributes(T)));
    }
    if (not pass.has_value())
    {
        std::uint64_t n = 0;
        for (const auto &pn : mem_->get_nodes_by_type("metaconcept"))
            if (pn.name().starts_with("passage_"))
            {
                std::uint64_t v = 0;
                const auto s = pn.name().substr(8);
                if (std::from_chars(s.data(), s.data() + s.size(), v).ec == std::errc{}) n = std::max(n, v + 1);
            }
        DSR::Node dw;
        dw.type("metaconcept"); dw.name(std::format("passage_{}", n));
        mem_->add_or_modify_attrib_local<object_subtype_att>(dw, std::string("passage"));
        mem_->add_or_modify_attrib_local<level_att>(dw, 0);
        const auto id = mem_->insert_node(dw);
        if (not id.has_value()) return false;
        mem_->insert_or_assign_edge(DSR::Edge::create<match_edge_type>(id.value(), prev->id()));
        pass = mem_->get_node(id.value());
    }
    mem_->insert_or_assign_edge(DSR::Edge::create<match_edge_type>(pass->id(), fid.value()));
    qInfo().noquote() << QString::fromStdString(std::format("[swap] carry-over: {} is a passage of {} and the previous room ({})",
            mem_->get_node(fid.value()).value().name(), known_mem_.empty() ? K->name() : known_mem_, pass->name()));
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
