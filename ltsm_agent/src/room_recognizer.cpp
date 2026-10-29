#include "room_recognizer.h"

#include "seam_fusion.h"
#include "../../common/room_resolve/room_resolve.h"   // rc::room::is_proto, current_edge_room

#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/api/dsr_rt_api.h>

#include <QDebug>
#include <QString>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <format>
#include <fstream>
#include <locale>
#include <limits>
#include <ranges>
#include <unordered_set>
#include <utility>

namespace ltsm
{
namespace
{
using seam::Mat3;
using seam::Measured;
using seam::Vec3;

constexpr const char *kRecognition = "recognition";
constexpr const char *kEvidence    = "place_evidence";

// Door-pose covariance prior per FACE: the value room_chain.cpp and seam_evict_test use (3 cm, 0.01 rad). Memory stores no
// covariance today, and a mirror whose edge states none reads as zeros; the prior is what a face that says nothing gets.
Mat3 face_prior()
{
    Mat3 C = Mat3::Zero();
    C.diagonal() << 0.03 * 0.03, 0.03 * 0.03, 0.01 * 0.01;
    return C;
}

Vec3 se2_of(const Mat::RTMat &T)
{
    return {T.translation().x(), T.translation().y(), std::atan2(T.rotation()(1, 0), T.rotation()(0, 0))};
}

/// The (x, y, yaw) block of an RT edge's 6x6 pose covariance; nullopt when the edge states none (all zeros reads as unstated).
std::optional<Mat3> cov3_of(DSR::DSRGraph &g, const DSR::Edge &e)
{
    const auto C6 = g.get_rt_api()->get_edge_RT_covariance(e);
    if (not C6.has_value()) return std::nullopt;
    static constexpr int slot[3] = {0, 1, 5};
    Mat3 S;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) S(i, j) = (*C6)(slot[i], slot[j]);
    if (not (S.cwiseAbs().maxCoeff() > 0.0)) return std::nullopt;
    return S;
}

std::string subtype_of(DSR::DSRGraph &g, const DSR::Node &n)
{
    return g.get_attrib_by_name<object_subtype_att>(n).value_or(std::string{});
}

bool is_staged(DSR::DSRGraph &g, const DSR::Node &room) { return subtype_of(g, room) == "staged"; }

std::optional<DSR::Node> owning_room(DSR::DSRGraph &g, const DSR::Node &n)
{
    auto cur = std::optional<DSR::Node>(n);
    for (int depth = 0; cur.has_value() and depth < 32; ++depth)
    {
        if (cur->type() == "room") return cur;
        const auto p = g.get_attrib_by_name<parent_att>(cur.value());
        if (not p.has_value() or p.value() == cur->id()) return std::nullopt;
        cur = g.get_node(p.value());
    }
    return std::nullopt;
}

std::optional<Eigen::Vector2d> centroid_of(DSR::DSRGraph &g, const DSR::Node &room)
{
    const auto px = g.get_attrib_by_name<delimiting_polygon_x_att>(room);
    const auto py = g.get_attrib_by_name<delimiting_polygon_y_att>(room);
    if (not px.has_value() or not py.has_value() or px->get().empty() or px->get().size() != py->get().size()) return std::nullopt;
    Eigen::Vector2d c = Eigen::Vector2d::Zero();
    for (const auto &[x, y] : std::views::zip(px->get(), py->get())) c += Eigen::Vector2d(x, y);
    return c / static_cast<double>(px->get().size());
}

/// T(room <- node) with its covariance, composed from the LIVE RT edges along `node`'s parent chain up to `room_id`
/// (a door hangs from a wall, the wall from the floor or the room). Each edge's covariance is in its parent's frame;
/// compose_cov rotates it into the room's. An edge that states none contributes none, except the node's OWN edge, which
/// gets the face prior (a door whose pose says nothing about its precision is still not exact). nullopt when the chain
/// does not reach `room_id` through stated RT edges.
std::optional<Measured> pose_in_room(DSR::DSRGraph &g, std::uint64_t room_id, const DSR::Node &node)
{
    auto rt = g.get_rt_api();
    Measured acc{Vec3::Zero(), Mat3::Zero()};
    DSR::Node cur = node;
    for (int depth = 0; depth < 32; ++depth)
    {
        if (cur.id() == room_id) return acc;
        const auto p = g.get_attrib_by_name<parent_att>(cur);
        if (not p.has_value() or p.value() == cur.id()) return std::nullopt;
        const auto e = g.get_edge(p.value(), cur.id(), "RT");
        if (not e.has_value()) return std::nullopt;
        const auto T = rt->get_edge_RT_as_rtmat(e.value());
        if (not T.has_value()) return std::nullopt;
        const Mat3 C = cov3_of(g, e.value()).value_or(depth == 0 ? face_prior() : Mat3::Zero());
        acc = seam::compose_cov(Measured{se2_of(T.value()), C}, acc);      // T(parent <- node) = T(parent <- cur) . T(cur <- node)
        const auto pn = g.get_node(p.value());
        if (not pn.has_value()) return std::nullopt;
        cur = pn.value();
    }
    return std::nullopt;
}

/// A valid covariance for the posterior: finite, symmetric, strictly positive definite (it has no PSD guard of its own).
std::optional<Eigen::Matrix3f> as_covariance(const Eigen::Matrix3d &S)
{
    if (not S.allFinite()) return std::nullopt;
    const Eigen::Matrix3d Ss = 0.5 * (S + S.transpose());
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(Ss);
    if (es.info() != Eigen::Success or not (es.eigenvalues().minCoeff() > 0.0)) return std::nullopt;
    return Ss.cast<float>();
}

/// The passage nodes of `g` (metaconcept, subtype "passage") with the set of rooms their faces belong to.
struct PassageRooms { std::string name; std::vector<std::string> rooms; std::vector<std::uint64_t> faces; };

std::vector<PassageRooms> passages_of(DSR::DSRGraph &g)
{
    std::vector<PassageRooms> out;
    for (const auto &n : g.get_nodes_by_type("metaconcept"))
    {
        if (subtype_of(g, n) != "passage") continue;
        PassageRooms p{.name = n.name(), .rooms = {}, .faces = {}};
        for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(n, "match"))
        {
            p.faces.push_back(e.to());
            if (const auto f = g.get_node(e.to()); f.has_value())
                if (const auto r = owning_room(g, f.value()); r.has_value()) p.rooms.push_back(r->name());
        }
        out.push_back(std::move(p));
    }
    return out;
}
}   // namespace

//////////////////////////////////////////////////////////////////////////////////////////////////
RoomRecognizer::RoomRecognizer(std::shared_ptr<DSR::DSRGraph> live, std::shared_ptr<DSR::DSRGraph> mem, const RoomEviction &evictor)
    : live_(std::move(live)), mem_(std::move(mem)), ev_(evictor) {}

int RoomRecognizer::sweep_stale()
{
    int n = 0;
    for (const auto &node : live_->get_nodes_by_type("metaconcept"))
        if (const auto st = subtype_of(*live_, node); st == kRecognition or st == kEvidence)
            if (live_->delete_node(node.id())) ++n;
    node_id_ = 0;
    published_.clear();
    published_ids_.clear();
    return n;
}

int RoomRecognizer::remove_owned()
{
    int n = 0;
    for (const auto &node : live_->get_nodes_by_type("metaconcept"))
        if (subtype_of(*live_, node) == kRecognition and live_->delete_node(node.id())) ++n;
    node_id_ = 0;
    published_.clear();
    published_ids_.clear();
    return n;
}

void RoomRecognizer::end_episode()
{
    if (swap_ack_ and prov_id_ != 0) released_ = true;     // an executed swap's window closes with its episode
    if (node_id_ != 0)
    {
        if (live_->get_node(node_id_).has_value()) live_->delete_node(node_id_);
        node_id_ = 0;
    }
    if (prov_id_ != 0)
        qInfo().noquote() << QString::fromStdString(std::format("[recog] episode for provisional room {} ended", prov_id_));
    prov_id_ = 0;
    ids_.clear();
    commit_ = {rc::recog::Commit::State::Entering, -1};
    base_.clear();
    metric_filt_.clear();
    metric_filt_ms_ = 0;
    L_id_ = 0;
    ep_L_mem_.clear();
    start_.clear();
    path_.clear();
    fresh_.reset();
    promoted_ = false;
    swap_ack_ = false;
    rebase_next_ = false;
    post_entry_.clear();
    published_.clear();
    published_ids_.clear();
}

void RoomRecognizer::track_current()
{
    const auto cur = rc::room::current_edge_room(*live_);
    std::string mem_name;
    if (cur.has_value())
        if (const auto room = live_->get_node(cur.value()); room.has_value())
            if (const auto m = ev_.memory_prefix(room.value()) + room->name(); mem_->get_node(m).has_value()) mem_name = m;
    if (mem_name == cur_mem_) return;

    // (a running episode is judged by step(): it spans the promotion)
    prev_mem_.clear();
    prev_passage_.clear();
    if (not cur_mem_.empty() and not mem_name.empty())
        for (const auto &p : passages_of(*mem_))
            if (std::ranges::contains(p.rooms, cur_mem_) and std::ranges::contains(p.rooms, mem_name))
            {
                prev_mem_ = cur_mem_;
                prev_passage_ = p.name;
                break;
            }
    cur_mem_ = mem_name;
}

void RoomRecognizer::publish(const std::vector<float> &post)
{
    if (node_id_ == 0 or not live_->get_node(node_id_).has_value())
    {
        node_id_ = 0;
        if (const auto n = live_->get_node(kRecognition); n.has_value()) node_id_ = n->id();
    }
    if (node_id_ != 0 and ids_ == published_ids_ and post == published_) return;      // identical: no rewrite
    std::optional<DSR::Node> node = node_id_ != 0 ? live_->get_node(node_id_) : std::nullopt;
    const bool fresh = not node.has_value();
    if (fresh)
    {
        DSR::Node n;
        n.type("metaconcept");
        n.name(kRecognition);
        node = n;
        live_->add_or_modify_attrib_local<object_subtype_att>(node.value(), std::string(kRecognition));
        live_->add_or_modify_attrib_local<level_att>(node.value(), 0);
        live_->add_or_modify_attrib_local<pos_x_att>(node.value(), 0.f);
        live_->add_or_modify_attrib_local<pos_y_att>(node.value(), 120.f);
    }
    live_->add_or_modify_attrib_local<recog_room_ids_att>(node.value(), std::vector<std::uint64_t>(ids_));
    live_->add_or_modify_attrib_local<recog_posterior_att>(node.value(), std::vector<float>(post));
    if (fresh)
    {
        const auto id = live_->insert_node(node.value());      // LVALUE (the rvalue overload is not instantiated in libdsr)
        if (not id.has_value()) return;
        node_id_ = id.value();
        qInfo().noquote() << QString::fromStdString(std::format("[recog] recognition node posted: {} candidate room(s)", ids_.size()));
    }
    else
        live_->update_node(node.value());
    published_ = post;
    published_ids_ = ids_;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
RoomRecognizer::Step RoomRecognizer::abort_episode(Step out)
{
    // An episode that ends while a commit stands, other than by the robot leaving the room it was committed in (the
    // eviction fence), has lost the ground the commit stood on: ask for the undo.
    if (prov_id_ != 0 and commit_.state == rc::recog::Commit::State::Committed and commit_.room >= 0 and commit_.room < int(ids_.size()))
    {
        out.act = Step::Act::Undo;
        out.room_index = commit_.room;
        out.room_id = ids_[commit_.room];
        if (commit_.room < int(post_entry_.size())) out.pose_in_k = post_entry_[commit_.room].pose_in_k;
        qInfo().noquote() << QString::fromStdString(std::format("[recog] UNDO requested: room_id {} (the episode ended under a commit)", out.room_id));
    }
    end_episode();
    return out;
}

RoomRecognizer::Step RoomRecognizer::ineligible(Step out, const std::string &room, std::string_view reason)
{
    const std::string v = std::format("{}: {}", room, reason);
    if (v != verdict_)
    {
        verdict_ = v;
        qInfo().noquote() << QString::fromStdString(std::format("[recog] {} not eligible: {}", room, reason));
    }
    return abort_episode(out);
}

void RoomRecognizer::on_swap_executed(std::uint64_t room_id)
{
    if (prov_id_ == 0 or commit_.state != rc::recog::Commit::State::Committed or commit_.room < 0 or commit_.room >= int(ids_.size())
        or ids_[commit_.room] != room_id)
        return;
    swap_ack_ = true;
    qInfo().noquote() << QString::fromStdString(std::format("[recog] swap to room_id {} executed: episode held until the robot leaves it", room_id));
}

void RoomRecognizer::on_swap_failed()
{
    if (swap_ack_ or commit_.state != rc::recog::Commit::State::Committed) return;
    commit_ = {rc::recog::Commit::State::Entering, -1};
    qInfo() << "[recog] swap failed: back to entering";
}

void RoomRecognizer::on_undo_executed() { swap_ack_ = false; }

void RoomRecognizer::on_proposal_rejected()
{
    if (prov_id_ == 0 or not swap_ack_) return;
    const auto rid = commit_.room >= 0 and commit_.room < int(ids_.size()) ? ids_[commit_.room] : 0;
    commit_ = {rc::recog::Commit::State::Entering, -1};
    swap_ack_ = false;
    rebase_next_ = true;
    qInfo().noquote() << QString::fromStdString(std::format("[recog] proposal for room_id {} rejected: back to entering, evidence rebased at the rejection", rid));
}

RoomRecognizer::Step RoomRecognizer::held_step(Step out)
{
    // The swap stands: its proposal waits for room_concept's verdict (RoomSwap reads it; a rejection comes back through
    // on_proposal_rejected), or it was accepted. Nothing here is judged. Only the robot LEAVING the committed room ends the
    // episode, and with no Undo.
    const auto cur = rc::room::current_edge_room(*live_);
    if (not cur.has_value()) return out;
    const auto room = live_->get_node(cur.value());
    const auto rid = room.has_value() ? live_->get_attrib_by_name<room_id_att>(room.value()) : std::nullopt;
    const bool at_known = commit_.room >= 0 and commit_.room < int(ids_.size()) and rid.has_value() and rid.value() == ids_[commit_.room];
    if (cur.value() == L_id_ or cur.value() == prov_id_ or at_known) return out;
    end_episode();
    return out;
}

std::optional<std::pair<DSR::Node, DSR::Edge>> RoomRecognizer::provisional_room(const DSR::Node &robot, std::uint64_t cur,
                                                                                std::optional<std::uint64_t> except_room_id,
                                                                                std::uint64_t except_node) const
{
    // Hung from the robot by RT, not the current room, and a proto -- or (the hand-over case promotion_evidence also takes)
    // live while the robot->current edge is retired.
    bool cur_retired = false;
    if (const auto ce = live_->get_edge(robot.id(), cur, "RT"); ce.has_value())
        cur_retired = not live_->get_attrib_by_name<valid_att>(ce.value()).value_or(true);
    for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(robot, "RT"))
    {
        const auto room = live_->get_node(e.to());
        if (not room.has_value() or room->type() != "room" or room->id() == cur or room->id() == except_node) continue;
        const bool entered = cur_retired and live_->get_attrib_by_name<valid_att>(e).value_or(true);
        if (not rc::room::is_proto(*live_, room->id()) and not entered) continue;
        if (except_room_id.has_value() and live_->get_attrib_by_name<room_id_att>(room.value()) == except_room_id) continue;
        return std::pair{room.value(), e};
    }
    return std::nullopt;
}

RoomRecognizer::Step RoomRecognizer::step()
{
    Step s = step_impl();
    s.released = std::exchange(released_, false);
    return s;
}

RoomRecognizer::Step RoomRecognizer::step_impl()
{
    Step out;
    track_current();
    const auto cur = rc::room::current_edge_room(*live_);
    std::optional<DSR::Node> robot;
    if (cur.has_value())
        for (const auto &e : live_->get_edges_by_type("current"))
            if (e.to() == cur.value()) { robot = live_->get_node(e.from()); break; }

    // ── 0. A NEWER PROVISIONAL ROOM ends the running episode. Once the robot has moved on from L (into P, promoted, or
    //    into the committed room K), a proto hung from the robot that is not this episode's P is a NEW crossing, out of
    //    the room the episode was about: its question replaces this one. A standing commit is handled as any episode end:
    //    requested-but-unexecuted => Undo (abort_episode); EXECUTED (held) => the robot is leaving K through another door,
    //    which is the fence (no Undo; Step::released tells the caller to close RoomSwap's window). K's own late re-birth is
    //    a proto too (room_concept adopt_late_recall), so a room carrying the committed room id is never "newer". When the
    //    episode simply ends, eligibility runs for the newer room IN THIS CYCLE; when ending it requests an Undo, that
    //    request is this cycle's and eligibility runs on the next. Structural, no timer. ──────────────────────────────────
    if (prov_id_ != 0 and cur.has_value() and robot.has_value() and cur.value() != L_id_)
    {
        const bool committed = commit_.state == rc::recog::Commit::State::Committed and commit_.room >= 0 and commit_.room < int(ids_.size());
        if (const auto Pn = provisional_room(robot.value(), cur.value(), committed ? std::optional(ids_[commit_.room]) : std::nullopt, prov_id_);
            Pn.has_value())
        {
            qInfo().noquote() << QString::fromStdString(std::format("[recog] newer provisional room {} behind the current room: the episode for {} ends",
                                                                    Pn->first.name(), prov_id_));
            if (swap_ack_) end_episode();
            else if (out = abort_episode(out); out.act != Step::Act::None) return out;     // one request per cycle
        }
    }

    if (swap_ack_ and prov_id_ != 0) return held_step(out);
    if (not cur.has_value()) return abort_episode(out);
    if (not robot.has_value()) return abort_episode(out);

    auto ie = live_->get_inner_eigen_api();
    auto rt = live_->get_rt_api();

    // the robot in a room, with its covariance (promotion_evidence's reading: no covariance, no pose)
    const auto read_robot_in = [&](const DSR::Edge &e) -> std::optional<Measured> {
        if (not live_->get_attrib_by_name<valid_att>(e).value_or(true)) return std::nullopt;
        const auto E = rt->get_edge_RT_as_rtmat(e);
        const auto C = cov3_of(*live_, e);
        if (not E.has_value() or not C.has_value()) return std::nullopt;
        return seam::inverse_cov(Measured{se2_of(E.value()), C.value()}); };

    std::string L_mem;
    std::optional<Measured> robot_in_P;
    std::optional<ChainPose> fresh;
    std::optional<std::uint64_t> P_rid;

    // ── 1. WHICH PHASE. The episode spans the promotion: it lives while the current room is L (before), the provisional
    //    room (its promoted twin) or the committed room; it ends at the eviction fence (the robot leaves it). ────────────
    bool phase_b = false;
    if (prov_id_ != 0)
    {
        const auto cur_rid = live_->get_node(cur.value()).has_value()
                                 ? live_->get_attrib_by_name<room_id_att>(live_->get_node(cur.value()).value()) : std::nullopt;
        if (cur.value() == prov_id_) { phase_b = true; promoted_ = true; }
        else if (commit_.state == rc::recog::Commit::State::Committed and commit_.room >= 0 and commit_.room < int(ids_.size())
                 and cur_rid.has_value() and cur_rid.value() == ids_[commit_.room])
        {   // the recognised room is current: the evidence frame (the provisional room) is gone, the commit stands
            promoted_ = true;
            return out;
        }
        else if (cur.value() != L_id_)
        {
            if (promoted_) { end_episode(); return out; }       // the fence: the robot left the room, the commit stays
            return abort_episode(out);
        }
    }

    if (phase_b)
    {
        // After promotion the old room may be gone: the metric uses the seam cached while it was measurable, and the robot
        // edge to the (now current) twin.
        L_mem = ep_L_mem_;
        const auto Pn = live_->get_node(prov_id_);
        if (not Pn.has_value()) return abort_episode(out);
        P_rid = live_->get_attrib_by_name<room_id_att>(Pn.value());
        if (const auto pe = live_->get_edge(robot->id(), prov_id_, "RT"); pe.has_value()) robot_in_P = read_robot_in(pe.value());
        fresh = fresh_;
    }
    else
    {
    // ── 2. ELIGIBILITY: current room L, provisional room P with an entry mirror, a crossed door nobody has used ──────
    // L's MEMORY IDENTITY (cur_mem_): the twin memory already holds of the live room. A room that became current by a
    // promotion has one from that moment (eviction step 3 copies the entered room's subtree, the stub its later eviction
    // fills); the startup room has the seed's. The twin is a COPY of L's frame, so the live L frame and the twin's are one
    // frame: that identity is what joins L's LIVE geometry to the memory chains. Without a twin nothing joins them.
    const auto L = live_->get_node(cur.value());
    const auto Pp = provisional_room(robot.value(), cur.value(), std::nullopt);
    if (not Pp.has_value()) { verdict_.clear(); return abort_episode(out); }
    if (not L.has_value() or cur_mem_.empty())
        return ineligible(out, Pp->first.name(), "the current room has no memory twin (cur_mem_ unset)");
    const std::optional<DSR::Node> P = Pp->first;
    robot_in_P = read_robot_in(Pp->second);

    // the entry mirror: a door whose parent is P. The nearest to the robot (an argmin, nothing is excluded by distance).
    std::optional<DSR::Node> M;
    double best_m = std::numeric_limits<double>::infinity();
    for (const auto &d : live_->get_nodes_by_type("object"))
    {
        if (not is_door(*live_, d)) continue;
        const auto par = live_->get_attrib_by_name<parent_att>(d);
        if (not par.has_value() or par.value() != P->id()) continue;
        double dist = 0.0;
        if (robot_in_P.has_value())
            if (const auto T = ie->get_transformation_matrix(P->name(), d.name()); T.has_value())
                dist = (T->translation().head<2>() - robot_in_P->pose.head<2>()).norm();
        if (dist < best_m) { best_m = dist; M = d; }
    }
    if (not M.has_value()) return ineligible(out, P->name(), "no entry-mirror door hangs from it");

    // its counterpart D in L: the door the live passage matched to it, else the nearest door of L (both in L's frame).
    std::optional<DSR::Node> D;
    for (const auto &n : live_->get_nodes_by_type("metaconcept"))
    {
        if (subtype_of(*live_, n) != "passage") continue;
        const auto m = DSR::DSRGraph::get_node_edges_by_type(n, "match");
        if (std::ranges::none_of(m, [&](const auto &e) { return e.to() == M->id(); })) continue;
        for (const auto &e : m)
            if (e.to() != M->id())
                if (const auto d = live_->get_node(e.to()); d.has_value())
                    if (const auto r = owning_room(*live_, d.value()); r.has_value() and r->id() == L->id()) D = d;
    }
    if (not D.has_value())
    {
        const auto TM = ie->get_transformation_matrix(L->name(), M->name());
        double best = std::numeric_limits<double>::infinity();
        if (TM.has_value())
            for (const auto &d : live_->get_nodes_by_type("object"))
            {
                if (not is_door(*live_, d)) continue;
                const auto r = owning_room(*live_, d);
                if (not r.has_value() or r->id() != L->id()) continue;
                if (const auto T = ie->get_transformation_matrix(L->name(), d.name()); T.has_value())
                    if (const double dist = (T->translation() - TM->translation()).head<2>().norm(); dist < best) { best = dist; D = d; }
            }
    }
    if (not D.has_value()) return ineligible(out, P->name(), "no crossed door found in the current room");

    L_mem = cur_mem_;
    // The crossed door must have NO passage in memory. THE PLACE DECIDES, not the name: door_concept re-makes door nodes
    // under new names (recog_try12: a new door_7 matched the entry door's face r3_door_7 by name => known => room_4 never
    // examined). RoomEviction::remembered_hole is the one resolver every consumer uses: D's centre inside a remembered
    // face's aperture in L's twin (half the door width: a point inside a segment, a membership test, not a tuning
    // constant); the name is the fallback only when D's pose in L is unresolvable.
    const bool known_door = ev_.remembered_hole(D.value()).has_value();
    if (known_door) return ineligible(out, P->name(), std::format("the crossed door {} is already a passage in memory", D->name()));
    if (prov_id_ != P->id())
    {
        end_episode();
        prov_id_ = P->id();
        L_id_ = L->id();
        ep_L_mem_ = L_mem;
        promoted_ = false;
        // the path taken: the passage that brought the robot into L (from its previous room), else L itself is the reference
        path_.clear();
        start_ = L_mem;
        if (not prev_passage_.empty() and not prev_mem_.empty()) { path_ = {prev_passage_}; start_ = prev_mem_; }
        qInfo().noquote() << QString::fromStdString(std::format("[recog] episode begins: provisional {} behind {}", P->name(), L->name()));
    }
    P_rid = live_->get_attrib_by_name<room_id_att>(P.value());

    // The fresh seam T(L <- P) from the crossed door's two faces, BOTH LIVE: D in L (door_concept's pose of the door, its RT
    // chain and covariance) and its entry mirror M in P. L is still live here (this block runs only while L is current), so
    // nothing has to be archived first -- a first-ever door has no memory face, and needs none. Frames:
    //     fL     = T(L <- D)       live, composed up D's parent chain (pose_in_room)
    //     fP     = T(P <- M)       live
    //     seam   = T(L <- P) = fL . [turn by pi] . fP^-1
    // L's live frame IS its memory twin's frame (above), so the seam continues the memory path chain that ends at L_mem.
    // nullopt (=> metric unavailable) when either face or the robot pose in P is unresolvable.
    if (robot_in_P.has_value())
    {
        const auto fL_live = pose_in_room(*live_, L->id(), D.value());
        auto fP_live = pose_in_room(*live_, P->id(), M.value());
        if (not fP_live.has_value())
            if (const auto TP = ie->get_transformation_matrix(P->name(), M->name()); TP.has_value())
                fP_live = Measured{se2_of(TP.value()), face_prior()};
        if (fL_live.has_value() and fP_live.has_value())
        {
            const Measured fL = fL_live.value();
            const Measured inv_fP = seam::inverse_cov(fP_live.value());
            Measured seam_LP = seam::compose_cov(fL, inv_fP);

            // The two faces are ONE hole, possibly TURNED by pi (door_concept orients a door's frame from the room that fits
            // it, so the mirror seen from P is the same doorway rotated half a turn). The provisional room lies on the
            // far side of the hole from L, and the robot is in it by definition: if the seam lands the robot on L's side
            // of the face, the turned face is the one. A side test, not a tolerance; the same rule seam_through_passage
            // applies with the two outlines' centroids. L's side is read from its LIVE outline (the twin's may be the
            // promotion-time stub), the twin's when the live room publishes none.
            auto c_L = centroid_of(*live_, L.value());
            if (not c_L.has_value())
                if (const auto mem_L = mem_->get_node(L_mem); mem_L.has_value()) c_L = centroid_of(*mem_, mem_L.value());
            if (c_L.has_value())
            {
                const Eigen::Vector2d r_in_L = seam_LP.pose.head<2>() + seam::rot(seam_LP.pose.z()) * robot_in_P->pose.head<2>();
                const Eigen::Vector2d o = fL.pose.head<2>();
                const Eigen::Vector2d n(-std::sin(fL.pose.z()), std::cos(fL.pose.z()));
                if (n.dot(c_L.value() - o) * n.dot(r_in_L - o) > 0.0)
                    seam_LP = seam::compose_cov(seam::compose_cov(fL, Measured{Vec3(0.0, 0.0, M_PI), Mat3::Zero()}), inv_fP);
            }
            fresh = ChainPose{seam_LP.pose, seam_LP.cov};
        }
    }
    fresh_ = fresh;

    }

    ChainPose robot_prov;
    if (robot_in_P.has_value()) robot_prov = {robot_in_P->pose, robot_in_P->cov};
    auto q = build_metric_query(*mem_, path_, start_, robot_in_P.has_value() ? fresh : std::nullopt, robot_prov);

    // Candidates: archived rooms that are not staged (build_metric_query already drops them), not L (the crossing rules it
    // out), and not P's own twin if memory holds one. Sorted by room_id: the order retina's evidence is keyed to.
    std::erase_if(q.rooms, [&](const KnownRoomMetric &k) {
        return k.mem_room == L_mem or (P_rid.has_value() and k.room_id == P_rid.value()); });
    std::ranges::sort(q.rooms, {}, &KnownRoomMetric::room_id);
    if (q.rooms.empty()) return abort_episode(out);

    bool skip_decision = false;
    // the candidate list changed under a running episode: carry the commit by id, drop it if its room left
    {
        std::vector<std::uint64_t> ids;
        for (const auto &k : q.rooms) ids.push_back(k.room_id);
        if (ids != ids_)
        {
            if (commit_.state == rc::recog::Commit::State::Committed)
            {
                const auto old = (commit_.room >= 0 and commit_.room < int(ids_.size())) ? std::optional(ids_[commit_.room]) : std::nullopt;
                const auto it = old.has_value() ? std::ranges::find(ids, old.value()) : ids.end();
                if (it == ids.end())
                {   // the committed room is no longer a candidate: the commit lost its ground
                    out.act = Step::Act::Undo;
                    out.room_index = commit_.room;
                    out.room_id = old.value_or(0);
                    if (commit_.room < int(post_entry_.size())) out.pose_in_k = post_entry_[commit_.room].pose_in_k;
                    commit_ = {rc::recog::Commit::State::Entering, -1};
                    skip_decision = true;
                    qInfo().noquote() << QString::fromStdString(std::format("[recog] UNDO requested: room_id {} left the candidates", out.room_id));
                }
                else
                    commit_ = {rc::recog::Commit::State::Committed, int(it - ids.begin())};
            }
            ids_ = std::move(ids);
        }
    }
    const std::size_t n = ids_.size();

    // ── 3. EVIDENCE ──────────────────────────────────────────────────────────────────────────────────────────────────
    float log_new = 0.f;
    auto metric = metric_evidence(q, log_new);
    // ── 3a. The metric evidence is a FIT OF THE ROOM ESTIMATE AS IT STANDS — one frame of a correlated sequence, not
    //        independent news. Believed frame by frame, a single transient fit of a half-built layout flipped the posterior
    //        from 1e-6 to 0.84 after eleven frames that all said "new" (recog_try30: room_3 swapped for room_2 while the
    //        robot never left room_3). Filter it: each hypothesis's log-evidence is the exponentially weighted mean of its
    //        frames, the weight of the past decaying with the correlation time of the map evidence. FLAGGED model constant:
    //        kMetricTauS, the time over which the room estimate's fit decorrelates (the layout refits at ~1 Hz and settles in
    //        seconds). Not a gate: every frame counts, none decides alone.
    {
        constexpr double kMetricTauS = 5.0;
        const std::int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::steady_clock::now().time_since_epoch()).count();
        const double keep = metric_filt_ms_ == 0 ? 0.0 : std::exp(-double(now_ms - metric_filt_ms_) / 1000.0 / kMetricTauS);
        metric_filt_ms_ = now_ms;
        const auto filt = [&](std::int64_t key, float x) -> float {
            if (not std::isfinite(x)) return x;
            const auto it = metric_filt_.find(key);
            const float y = it == metric_filt_.end() ? x : float(keep * it->second + (1.0 - keep) * x);
            metric_filt_[key] = y;
            return y;
        };
        for (std::size_t k = 0; k < metric.size() and k < ids_.size(); ++k)
            if (metric[k].available) metric[k].log_inside = filt(std::int64_t(ids_[k]), metric[k].log_inside);
        log_new = filt(-1, log_new);
    }
    std::vector<float> area(n, 0.f);
    for (std::size_t k = 0; k < n; ++k)
    {
        area[k] = float(q.rooms[k].area);
        // The posterior has no PSD guard: a pose density that is not a covariance (or whose room has no area to normalise
        // the pose prior) is dropped, leaving the room's inside-mass; that is validity, not a model threshold.
        auto &pm = metric[k].pose_in_k;
        for (auto &g : pm)
        {
            const auto S = as_covariance(g.S.cast<double>());
            if (S.has_value() and g.mu.allFinite()) g.S = S.value(); else g.w = -1.f;
        }
        std::erase_if(pm, [](const rc::recog::PoseGauss &g) { return not (g.w >= 0.f); });
        if (not (area[k] > 0.f)) pm.clear();
    }

    // appearance: retina's place_evidence, matched BY room id (it may carry an older candidate list)
    std::vector<rc::recog::AppearanceEvidence> app(n);
    for (const auto &pe : live_->get_nodes_by_type("metaconcept"))
    {
        if (subtype_of(*live_, pe) != kEvidence) continue;
        const auto ids = live_->get_attrib_by_name<recog_room_ids_att>(pe);
        const auto llr = live_->get_attrib_by_name<recog_loglr_att>(pe);
        const auto mix = live_->get_attrib_by_name<recog_mix_att>(pe);
        const auto mixn = live_->get_attrib_by_name<recog_mix_n_att>(pe);
        if (not ids.has_value() or not llr.has_value() or ids->get().size() != llr->get().size()) break;
        std::size_t off = 0;
        for (std::size_t j = 0; j < ids->get().size(); ++j)
        {
            const std::size_t cnt = (mixn.has_value() and j < mixn->get().size()) ? std::size_t(mixn->get()[j]) : 0;
            const auto it = std::ranges::find(ids_, ids->get()[j]);
            if (it != ids_.end() and std::isfinite(llr->get()[j]))
            {
                auto &a = app[it - ids_.begin()];
                a.available = true;
                a.loglr = llr->get()[j];
                if (mix.has_value())
                    for (std::size_t c = 0; c < cnt and (off + c + 1) * 10 <= mix->get().size(); ++c)
                    {
                        const float *f = mix->get().data() + (off + c) * 10;
                        Eigen::Matrix3d S;
                        S << f[4], f[5], f[6],
                             f[5], f[7], f[8],
                             f[6], f[8], f[9];
                        const auto cov = as_covariance(S);
                        if (cov.has_value() and std::isfinite(f[0]) and f[0] >= 0.f and std::isfinite(f[1]) and std::isfinite(f[2]) and std::isfinite(f[3]))
                            a.pose_in_k.push_back({Eigen::Vector3f(f[1], f[2], rc::place::wrap_angle(f[3])), cov.value(), f[0]});
                    }
            }
            off += cnt;
        }
        break;
    }

    // ── 4. POSTERIOR and the staged commit ────────────────────────────────────────────────────────────────────────
    // Every decision uses the evidence gathered SINCE THE LAST STATE CHANGE (episode entry, commit, undo): per hypothesis the
    // log-evidence now minus its value at that change (the metric as a delta too, appearance rebased the same way), under the
    // uniform prior. With nothing new the delta is zero: the posterior is the prior, so no undo and no re-commit.
    post_entry_ = rc::recog::posterior(metric, app, area, log_new);        // all evidence now (published, and the pose density)
    const auto key = [&](const rc::recog::Hypothesis &h) -> std::int64_t { return h.room < 0 ? -1 : std::int64_t(ids_[h.room]); };
    // A REJECTED proposal (on_proposal_rejected): the evidence is rebased HERE, on the first frame after the rejection, so the
    // delta below is zero and only what arrives from now on can commit again.
    if (std::exchange(rebase_next_, false))
    {
        base_.clear();
        for (const auto &h : post_entry_) base_[key(h)] = h.log_post;
    }
    std::vector<rc::recog::Hypothesis> since = post_entry_;
    {
        std::vector<double> d(since.size());
        for (std::size_t i = 0; i < since.size(); ++i)
        {
            const double now = since[i].log_post;
            if (base_.empty()) { d[i] = now; continue; }                      // episode entry: all evidence is new
            const auto b = base_.find(key(since[i]));
            d[i] = b == base_.end() ? 0.0                                      // new to the list: no history
                                     : ((std::isfinite(now) or std::isfinite(b->second)) ? now - double(b->second) : 0.0);
        }
        const double z = rc::recog::detail::logsumexp(d);
        for (std::size_t i = 0; i < since.size(); ++i) since[i].log_post = std::isfinite(z) ? float(d[i] - z) : float(-std::log(double(d.size())));
    }
    const auto rebase = [&] {
        base_.clear();
        for (const auto &h : post_entry_) base_[key(h)] = h.log_post;
    };

    if (not skip_decision)
    {
        const auto next = rc::recog::commit_step(commit_, since, since);
        if (commit_.state == rc::recog::Commit::State::Entering and next.state == rc::recog::Commit::State::Committed)
        {
            out.act = Step::Act::Swap;
            out.room_index = next.room;
            out.room_id = ids_[next.room];
            out.pose_in_k = post_entry_[next.room].pose_in_k;
            rebase();
            qInfo().noquote() << QString::fromStdString(std::format("[recog] SWAP requested: room_id {} (posterior {:.3f}; metric log_inside {:.2f}, appearance loglr {:.2f}{}, new: metric log_new {:.2f})",
                                                                    out.room_id, std::exp(post_entry_[next.room].log_post),
                                                                    metric[next.room].available ? metric[next.room].log_inside : 0.f,
                                                                    app[next.room].available ? app[next.room].loglr : 0.f,
                                                                    app[next.room].available ? "" : " (unavailable)", log_new));
        }
        else if (commit_.state == rc::recog::Commit::State::Committed and next.state == rc::recog::Commit::State::Entering)
        {
            out.act = Step::Act::Undo;
            out.room_index = commit_.room;
            out.room_id = ids_[commit_.room];
            out.pose_in_k = post_entry_[commit_.room].pose_in_k;
            rebase();
            qInfo().noquote() << QString::fromStdString(std::format("[recog] UNDO requested: room_id {} contradicted by evidence since the commit", out.room_id));
        }
        commit_ = next;
    }
    else
        rebase();

    std::vector<float> post;
    for (const auto &h : post_entry_) post.push_back(std::exp(h.log_post));
    // per-step trace for the gold-tour report (etc/recognition_trace.csv, truncated per run, classic locale)
    if (not trace_.is_open() and not trace_tried_)
    {
        trace_tried_ = true;
        trace_.imbue(std::locale::classic());
        trace_.open("etc/recognition_trace.csv", std::ios::out | std::ios::trunc);
        if (trace_.is_open()) trace_ << "stamp_ms,provisional_id,act,room_id,candidates(id:posterior:metric_log_inside:appearance_loglr;new:posterior::log_new)\n";
    }
    if (trace_.is_open())
    {
        trace_ << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()
               << ',' << prov_id_ << ',' << (out.act == Step::Act::Swap ? "Swap" : out.act == Step::Act::Undo ? "Undo" : "None")
               << ',' << out.room_id << ',';
        for (std::size_t k = 0; k < n; ++k)
            trace_ << ids_[k] << ':' << std::exp(post_entry_[k].log_post) << ':' << (metric[k].available ? metric[k].log_inside : 0.f)
                   << ':' << (app[k].available ? app[k].loglr : 0.f) << ';';
        trace_ << "new:" << std::exp(post_entry_.back().log_post) << "::" << log_new << '\n';
        trace_.flush();
    }
    publish(post);
    return out;
}
}   // namespace ltsm
