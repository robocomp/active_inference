/*
 * room_eviction.cpp — see room_eviction.h for the shape of the transaction and ../EVICTION.md for
 * why it is shaped that way.
 */
#include "room_eviction.h"
#include "seam_fusion.h"

#include "../../common/room_resolve/room_resolve.h"   // rc::room::is_proto
#include "../../common/interaction_anchor/interaction_anchor.h"   // rc::interaction::is_anchor

#include <QDebug>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <deque>
#include <cmath>
#include <Eigen/Dense>
#include <utility>
#include <format>

namespace ltsm
{

// A door in this fleet is an `object` node carrying object_subtype == "door" — there IS a `door`
// node type registered in cortex and NOBODY uses it, so get_nodes_by_type("door") returns nothing.
// (door_scene_graph.cpp:128-137.) The name check is the belt to that braces.
bool is_door(DSR::DSRGraph &g, const DSR::Node &n)
{
    // A door's interaction anchor (hung from the robot while it is being crossed) copies the subtype and
    // the name; it is a pose handle, not a door, and must never be archived as one.
    if (rc::interaction::is_anchor(g, n)) return false;
    if (const auto st = g.get_attrib_by_name<object_subtype_att>(n); st.has_value() and st.value() == "door")
        return true;
    return n.name().starts_with("door_");
}

namespace
{
std::string trailing_index(const std::string &name)
{
    auto it = name.end();
    while (it != name.begin() and std::isdigit(static_cast<unsigned char>(*(it - 1)))) --it;
    return std::string(it, name.end());
}
}   // namespace

RoomEviction::RoomEviction(std::shared_ptr<DSR::DSRGraph> live, std::shared_ptr<DSR::DSRGraph> memory)
    : live_(std::move(live)), mem_(std::move(memory)) {}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Detection
//////////////////////////////////////////////////////////////////////////////////////////////////
std::optional<DSR::Node> RoomEviction::robot_node() const
{
    // The localisation frame is the `robot`-typed node (ROBOT_GEOMETRY.md: "Shadow"), which is what
    // room_concept hangs the room from. Prefer the one that actually has rooms under it, so a
    // second robot-typed node in the graph cannot steer this.
    std::optional<DSR::Node> fallback;
    for (const auto &n : live_->get_nodes_by_type("robot"))
    {
        if (not fallback.has_value()) fallback = n;
        if (not rooms_under_robot(n).empty()) return n;
    }
    return fallback;
}

std::vector<DSR::Node> RoomEviction::rooms_under_robot(const DSR::Node &robot) const
{
    // ★ PROTO-ROOMS ARE NOT ROOMS HERE. room_concept births a proto-room (a `room` with a `proto`
    // self-edge) on a door crossing and hangs it from the robot exactly like a real one; its identity
    // is unresolved and `current` stays on the room it came from. Counting it would make detect() see
    // two rooms and EVICT the apartment on the next cycle, and would stop ensure_current_edge() from
    // placing `current` on the one real room. It becomes a room when promotion removes the edge.
    std::vector<DSR::Node> rooms;
    for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(robot, "RT"))
        if (const auto child = live_->get_node(e.to());
            child.has_value() and child->type() == "room" and not rc::room::is_proto(*live_, child->id()))
            rooms.push_back(child.value());
    return rooms;
}

std::optional<Candidate> RoomEviction::detect() const
{
    const auto robot = robot_node();
    if (not robot.has_value()) return {};

    auto rooms = rooms_under_robot(robot.value());
    if (rooms.size() < 2) return {};

    // WHICH ONE IS NEW: the later birth stamp. An ORDERING, not a threshold — the stamp is written
    // by rc::provenance at creation, so the second room is strictly later by construction. A room
    // with no stamp cannot be ordered, and guessing would be worse than waiting.
    const auto birth = [this](const DSR::Node &n)
    { return live_->get_attrib_by_name<timestamp_creation_att>(n).value_or(0ULL); };

    if (std::ranges::any_of(rooms, [&](const auto &r) { return birth(r) == 0; }))
    {
        if (not std::exchange(warned_missing_stamp_, true))
            qWarning() << "[eviction] two rooms but at least one has no timestamp_creation -- refusing "
                          "to guess which is the new one. Rooms:" << static_cast<int>(rooms.size());
        return {};
    }
    warned_missing_stamp_ = false;

    std::ranges::sort(rooms, [&](const auto &a, const auto &b) { return birth(a) < birth(b); });

    // ONCE PER ROOM. The old room stays hung from the robot until its owners let it go, which is
    // several cycles after the flip -- re-running the transaction in between would rewrite memory
    // for nothing.
    if (archived_.contains(rooms.front().id()))
        return {};

    Candidate c;
    c.robot_id     = robot->id();
    c.robot_name   = robot->name();
    c.old_room_id  = rooms.front().id();          // oldest first: one eviction per cycle
    c.old_room_name= rooms.front().name();
    c.old_birth_ms = birth(rooms.front());
    c.new_room_id  = rooms.back().id();
    c.new_room_name= rooms.back().name();
    c.new_birth_ms = birth(rooms.back());
    return c;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Helpers
//////////////////////////////////////////////////////////////////////////////////////////////////
std::uint64_t RoomEviction::room_index(const DSR::Node &room) const
{
    // MEMOISED. room_concept now names its first room `room_1` and writes room_id = 1, so the live fleet
    // takes the attribute branch below and the index is stable by construction. The counter stays as the
    // fallback for a room carrying neither (a graph from before the rename, whose room is plain "room"
    // with no digits; an offline fixture) -- and an unmemoised counter hands the SAME room a different index
    // every time it is asked (seed → r0_, its eviction → r1_), which duplicates it in memory.
    if (const auto it = index_of_.find(room.id()); it != index_of_.end())
        return it->second;

    const auto compute = [&]() -> std::uint64_t
    {
        if (const auto id = live_->get_attrib_by_name<room_id_att>(room); id.has_value())
            return id.value();
        if (const auto digits = trailing_index(room.name()); not digits.empty())
        {
            std::uint64_t v = 0;
            // from_chars, never stoull/atoi: these machines run es_ES and CLAUDE.md is explicit.
            if (std::from_chars(digits.data(), digits.data() + digits.size(), v).ec == std::errc{})
                return v;
        }
        // The counter must not reuse an index memory already holds (a persisted generation, or a
        // room indexed earlier in this run), or two different rooms would share a prefix.
        for (const auto &r : mem_->get_nodes_by_type("room"))
            if (const auto i = mem_->get_attrib_by_name<room_id_att>(r); i.has_value())
                next_index_ = std::max(next_index_, i.value() + 1);
        for (const auto &[_, i] : index_of_)
            next_index_ = std::max(next_index_, i + 1);
        return next_index_++;
    };
    const auto idx = compute();
    index_of_.emplace(room.id(), idx);
    return idx;
}

std::optional<DSR::Node> RoomEviction::remembered(const DSR::Node &live_room) const
{
    for (const auto &r : mem_->get_nodes_by_type("room"))
        if (const auto i = mem_->get_attrib_by_name<room_id_att>(r);
            i.has_value() and r.name() == std::format("r{}_{}", i.value(), live_room.name()))
        {
            index_of_.try_emplace(live_room.id(), i.value());   // its eviction must reuse this prefix
            return r;
        }
    return {};
}

std::map<std::string, DSR::Attribute> RoomEviction::rt_attrs(const Mat::RTMat &t)
{
    const Mat::Vector3d tr  = t.translation();
    // ZYX-convention Euler angles, which is what rt_rotation_euler_xyz holds and what RT_API packs.
    const Mat::Vector3d rpy = t.rotation().eulerAngles(2, 1, 0).reverse();
    std::map<std::string, DSR::Attribute> a;
    a.emplace(rt_translation_str.data(),
              DSR::Attribute{std::vector<float>{static_cast<float>(tr.x()), static_cast<float>(tr.y()),
                                                static_cast<float>(tr.z())}, 0, 0});
    a.emplace(rt_rotation_euler_xyz_str.data(),
              DSR::Attribute{std::vector<float>{static_cast<float>(rpy.x()), static_cast<float>(rpy.y()),
                                                static_cast<float>(rpy.z())}, 0, 0});
    return a;
}

std::optional<std::uint64_t> RoomEviction::twin(const DSR::Node &src, std::uint64_t mem_parent_id,
                                                const std::string &prefix, std::uint64_t room_idx)
{
    const std::string mem_name = prefix + src.name();

    // Attributes come over wholesale — including timestamp_creation, so memory keeps the node's
    // REAL birth rather than the instant it was archived.
    auto attrs = src.attrs();
    attrs.erase(parent_str.data());
    attrs.erase(level_str.data());
    // `collapsed` is a VIEWER default (room_concept seeds it on the floor to fold the wall fan in the
    // busy live view). Copied into memory it folds r<k>_floor's walls away, and the doors that hang from
    // them are drawn detached from the room -- memory's view exists to show exactly that structure.
    attrs.erase(collapsed_str.data());

    const int parent_level = [&]
    {
        if (const auto p = mem_->get_node(mem_parent_id); p.has_value())
            return mem_->get_node_level(p.value()).value_or(0);
        return 0;
    }();

    // ALREADY THERE ⇒ FILL, don't duplicate. This is what makes a second eviction land on the stub
    // the first one left behind, and what makes re-running an eviction harmless.
    if (auto existing = mem_->get_node(mem_name); existing.has_value())
    {
        auto &e = existing.value();
        // RE-PARENT, never a second RT parent. The seed hangs a room's nodes where the live graph
        // had them at startup; if the room is restructured before its eviction (a door moved to
        // another wall) the caller's new edge would otherwise sit BESIDE the old one, and the RT tree
        // stops being a tree. Delete first: update_node below rewrites `parent` anyway.
        if (const auto old_parent = mem_->get_attrib_by_name<parent_att>(e);
            old_parent.has_value() and old_parent.value() != mem_parent_id)
            mem_->delete_edge(old_parent.value(), e.id(), "RT");
        for (auto &[k, v] : attrs) e.attrs()[k] = v;
        mem_->add_or_modify_attrib_local<parent_att>(e, mem_parent_id);
        mem_->add_or_modify_attrib_local<level_att>(e, parent_level + 1);
        mem_->add_or_modify_attrib_local<room_id_att>(e, room_idx);
        if (not mem_->update_node(e)) return {};
        return e.id();
    }

    DSR::Node n;
    n.type(src.type());
    n.name(mem_name);
    n.attrs(attrs);
    mem_->add_or_modify_attrib_local<parent_att>(n, mem_parent_id);
    mem_->add_or_modify_attrib_local<level_att>(n, parent_level + 1);
    mem_->add_or_modify_attrib_local<room_id_att>(n, room_idx);
    // LVALUE, not std::move: the forwarding-reference overload deduces No = DSR::Node from an
    // rvalue, and libdsr only instantiates insert_node<Node&> and <Node&&> -- so std::move here
    // links against nothing at all (undefined reference at link time, not a compile error).
    return mem_->insert_node(n);
}

bool RoomEviction::copy_subtree(std::uint64_t live_id, std::uint64_t mem_parent_id,
                                const std::string &prefix, std::uint64_t room_idx,
                                const std::map<std::string, DSR::Attribute> &root_edge_attrs,
                                IdMap &map, Outcome &out, Scope scope)
{
    struct Item { std::uint64_t live_id, mem_parent; std::map<std::string, DSR::Attribute> edge_attrs; };
    std::deque<Item> q{{live_id, mem_parent_id, root_edge_attrs}};

    while (not q.empty())
    {
        const auto item = q.front(); q.pop_front();

        const auto src = live_->get_node(item.live_id);
        if (not src.has_value()) continue;               // vanished under us: skip, do not abort

        const auto mem_id = twin(src.value(), item.mem_parent, prefix, room_idx);
        if (not mem_id.has_value())
        {
            out.reason = "could not write " + prefix + src->name() + " into memory";
            return false;
        }
        map[item.live_id] = mem_id.value();
        ++out.nodes_copied;

        if (auto parent = mem_->get_node(item.mem_parent); parent.has_value())
        {
            auto edge = DSR::Edge::create<RT_edge_type>(item.mem_parent, mem_id.value(), item.edge_attrs);
            if (mem_->insert_or_assign_edge(std::move(edge))) ++out.edges_copied;
        }

        // STRUCTURE stops at a door: what hangs below one (its affordances) is not the room's shape.
        if (scope == Scope::Structure and is_door(*live_, src.value())) continue;

        for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(src.value(), "RT"))
        {
            const auto child = live_->get_node(e.to());
            if (not child.has_value()) continue;
            if (child->type() == "robot" or child->type() == "root") continue;   // never climb out
            // ltsm's own recall hypotheses (room_recall.cpp) hang under a live door; they are working-memory pointers
            // INTO long-term memory, not part of the room — archiving one wrote `r2_recall_room_1` (live try22).
            if (live_->get_attrib_by_name<object_subtype_att>(child.value()).value_or(std::string{}) == "room_recall")
                continue;
            if (scope == Scope::Structure and child->type() != "floor" and child->type() != "wall"
                and not is_door(*live_, child.value()))
                continue;
            q.push_back({e.to(), mem_id.value(), e.attrs()});
        }
    }
    return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<DSR::Node> RoomEviction::passages() const
{
    // ★ THE SUBTYPE FILTER IS LOAD-BEARING. `metaconcept` is a shared type -- ring_metaconcept's
    // dining_set is one -- and eviction copies whatever hangs under a room, so memory can hold
    // metaconcepts that are not passages. Type alone would sweep them in.
    std::vector<DSR::Node> out;
    for (const auto &n : mem_->get_nodes_by_type("metaconcept"))
        if (const auto st = mem_->get_attrib_by_name<object_subtype_att>(n);
            st.has_value() and st.value() == "passage")
            out.push_back(n);
    return out;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// The seam through the crossed aperture (proto-rooms)
//////////////////////////////////////////////////////////////////////////////////////////////////
namespace
{
ltsm::seam::Vec3 se2_of(const Mat::RTMat &T)
{
    return {T.translation().x(), T.translation().y(), std::atan2(T.rotation()(1, 0), T.rotation()(0, 0))};
}

Mat::RTMat pose_to_rtmat(const ltsm::seam::Vec3 &p)
{
    Mat::RTMat T = Mat::RTMat::Identity();
    T.translate(Mat::Vector3d(p.x(), p.y(), 0.0));
    T.rotate(Eigen::AngleAxisd(p.z(), Eigen::Vector3d::UnitZ()));
    return T;
}

// The 6x6 RT covariance is over [x y z rx ry rz] (cortex convention, edge-parent frame); SE(2) = slots {0, 1, 5}.
std::optional<ltsm::seam::Mat3> se2_cov_of(const std::optional<Eigen::Matrix<double, 6, 6>> &C6)
{
    if (not C6.has_value()) return std::nullopt;
    static constexpr int slot[3] = {0, 1, 5};
    ltsm::seam::Mat3 S;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            S(i, j) = (*C6)(slot[i], slot[j]);
    return S;
}

/// One RT edge as a pose with covariance; `has_cov` says whether the covariance was stated (zero otherwise).
std::optional<ltsm::seam::Measured> edge_measured(DSR::DSRGraph &G, std::uint64_t from, std::uint64_t to, bool &has_cov)
{
    const auto e = G.get_edge(from, to, "RT");
    if (not e.has_value()) return std::nullopt;
    auto rt = G.get_rt_api();
    const auto T = rt->get_edge_RT_as_rtmat(e.value());
    if (not T.has_value()) return std::nullopt;
    const auto C = se2_cov_of(rt->get_edge_RT_covariance(e.value()));
    has_cov = C.has_value();
    return ltsm::seam::Measured{se2_of(T.value()), C.value_or(ltsm::seam::Mat3::Zero().eval())};
}

/// `node`'s pose in `room`, composed down the parent chain room -> floor -> wall -> node. Ancestors without a stated
/// covariance contribute none; `leaf_cov` is whether the NODE's own edge stated one (the gate for trusting the result).
std::optional<ltsm::seam::Measured> pose_in_room(DSR::DSRGraph &G, std::uint64_t room_id, const DSR::Node &node, bool &leaf_cov)
{
    ltsm::seam::Measured acc{ltsm::seam::Vec3::Zero(), ltsm::seam::Mat3::Zero()};
    std::uint64_t n = node.id();
    leaf_cov = false;
    for (int depth = 0; n != room_id and depth < 16; ++depth)
    {
        const auto nn = G.get_node(n);
        if (not nn.has_value()) return std::nullopt;
        const auto p = G.get_attrib_by_name<parent_att>(nn.value());
        if (not p.has_value()) return std::nullopt;
        bool hc = false;
        const auto m = edge_measured(G, p.value(), n, hc);
        if (not m.has_value()) return std::nullopt;
        if (depth == 0) leaf_cov = hc;
        acc = ltsm::seam::compose_cov(m.value(), acc);
        n = p.value();
    }
    return n == room_id ? std::optional{acc} : std::nullopt;
}
}   // namespace

// The door's recess behind the room's mouth (see aperture_seam). A model constant of the building's geometry, FLAGGED:
// not stored on the graph. Observed 0.49-0.50 m twice (gold2); uniform on [0, 0.5] would be mean 0.25, sigma 0.14 -- sigma
// is taken wider (0.25) so a recess the two observations did not span is still an inlier. room_concept writing the mouth
// pose would retire this.
constexpr double kRecessMean = 0.25, kRecessSigma = 0.25;

std::optional<Mat::RTMat> RoomEviction::fuse_seam_through_aperture(const Candidate &cand, const DSR::Node &door,
                                                                   const Mat::RTMat &door_in_old,
                                                                   const Mat::RTMat &chain_seam_rt,
                                                                   DSR::InnerEigenAPI &ie, bool new_room_was_remembered) const
{
    using namespace ltsm::seam;
    const auto robot_in_old = ie.get_transformation_matrix(cand.old_room_name, cand.robot_name);
    if (not robot_in_old.has_value()) return std::nullopt;
    const Eigen::Vector2d robot_xy(robot_in_old->translation().x(), robot_in_old->translation().y());
    const Vec3 door_pose = se2_of(door_in_old);

    // ── THE APERTURE'S SEAM ─────────────────────────────────────────────────────────────────────────────────────
    bool leaf = false;
    const auto door_m = pose_in_room(*live_, cand.old_room_id, door, leaf);
    if (not door_m.has_value() or not leaf)
    {
        // No stated pose covariance for the door: nothing can weigh the aperture, and a seam nothing can audit is not
        // improved by averaging it with one. Keep the robot chain (the pre-Task-S behaviour).
        qWarning().noquote() << QString::fromStdString(std::format(
            "[eviction][seam] door {} has no pose covariance in {} -- aperture seam unavailable, robot chain kept", door.name(),
            cand.old_room_name));
        return std::nullopt;
    }
    Measured ap;
    const char *how = "proto frame";
    std::optional<Measured> face_new;
    if (cand.new_door_id != 0)
        if (const auto nd = live_->get_node(cand.new_door_id); nd.has_value())
        {
            bool nleaf = false;
            if (auto m = pose_in_room(*live_, cand.new_room_id, nd.value(), nleaf); m.has_value() and nleaf)
                face_new = m;
        }
    if (face_new.has_value())
    {   // the entry mirror is the same hole at the same pose: T(new<-old) = T(new<-face_new) . inverse(T(old<-door))
        ap = compose_cov(face_new.value(), inverse_cov(door_m.value()));
        how = "door faces";
    }
    else if (
             new_room_was_remembered or not (cand.crossing_born or rc::room::is_proto(*live_, cand.new_room_id)))
    {
        // ★ THE APERTURE-AS-FRAME SEAM IS A HYPOTHESIS ABOUT THE NEW ROOM (Task S, round 2): it holds only for a room
        //   born through this crossing. An established room has an arbitrary frame; there the aperture seam is wrong
        //   and the mixture would reject a CORRECT chain. No provenance => the aperture carries no information about
        //   T(new<-old) (infinite frame uncertainty): keep the chain.
        //   ★ A RE-ENTERED (recalled) ROOM IS NEVER BORN BY THIS CROSSING (gold2, 2026-10-23): promotion_candidate flags
        //   every promotion, including the return into room_1, which memory already holds with its own frame. The old
        //   seam stored for that return was (+0.04,-0.01,+3.13) instead of ~ inverse(first) = (-0.64,+4.61,-0.014).
        qInfo().noquote() << QString::fromStdString(std::format(
            "[eviction][seam] {} is not known to be crossing-born and has no entry face of {} -- robot chain kept",
            cand.new_room_name, door.name()));
        return std::nullopt;
    }
    else
        // A proto's frame IS the aperture, so T(new<-old) is the inverse of the aperture pose in the old room. Its
        // pose comes from the door node (x = wall tangent, door_apertures.cpp:137); the sign of y is the side the
        // robot is on.
        ap = aperture_seam(se2_of(door_in_old), door_m->cov, robot_xy, kRecessMean, kRecessSigma);
    (void)door_pose;

    // ── THE ROBOT CHAIN, WITH ITS OWN COVARIANCE ───────────────────────────────────────────────────────────────
    bool hn = false, ho = false;
    const auto e_new = edge_measured(*live_, cand.robot_id, cand.new_room_id, hn);
    const auto e_old = edge_measured(*live_, cand.robot_id, cand.old_room_id, ho);
    if (not e_new.has_value() or not e_old.has_value() or not hn or not ho)
    {
        qWarning().noquote() << QString::fromStdString(std::format(
            "[eviction][seam] robot RT edges carry no covariance -- seam taken from the aperture alone ({} <- {}, {})",
            cand.new_room_name, cand.old_room_name, how));
        return pose_to_rtmat(ap.pose);
    }
    Measured chain = chain_seam(e_new.value(), e_old.value());
    chain.pose = se2_of(chain_seam_rt);   // the graph's own composition is the mean; the covariance is from the edges

    // Old room's extent from its origin (the outlier component's scale); the door's distance when no outline is stored.
    double radius = door_pose.head<2>().norm();
    if (const auto orn = live_->get_node(cand.old_room_id); orn.has_value())
    {
        const auto px = live_->get_attrib_by_name<delimiting_polygon_x_att>(orn.value());
        const auto py = live_->get_attrib_by_name<delimiting_polygon_y_att>(orn.value());
        if (px.has_value() and py.has_value())
            for (std::size_t i = 0; i < std::min(px->get().size(), py->get().size()); ++i)
                radius = std::max(radius, std::hypot(double(px->get()[i]), double(py->get()[i])));
    }
    radius = std::max(radius, 0.5);   // a door's own width: an outline-less room must not make U infinite

    const Fused f = fuse(chain, ap, radius);
    qInfo().noquote() << QString::fromStdString(std::format(
        "[eviction][seam] {} <- {} ({}): chain ({:+.3f},{:+.3f},{:+.3f}) aperture ({:+.3f},{:+.3f},{:+.3f}) d2 {:.2f} "
        "P(chain right) {:.4g} log P(chain wrong) {:.2f} -> ({:+.3f},{:+.3f},{:+.3f})",
        cand.new_room_name, cand.old_room_name, how, chain.pose.x(), chain.pose.y(), chain.pose.z(), ap.pose.x(), ap.pose.y(),
        ap.pose.z(), f.d2, f.p_inlier, f.log_p_wrong, f.pose.x(), f.pose.y(), f.pose.z()));
    return pose_to_rtmat(f.pose);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// The transaction
//////////////////////////////////////////////////////////////////////////////////////////////////
Outcome RoomEviction::evict(const Candidate &cand)
{
    Outcome out;

    // ── 1. THE SEAM, FIRST ──────────────────────────────────────────────────────────────────────
    // T(room_new <- room_old), composed through the robot. Computable ONLY while both RT edges
    // exist, which is why nothing is written before this line.
    auto ie = live_->get_inner_eigen_api();
    const auto seam = ie->get_transformation_matrix(cand.new_room_name, cand.old_room_name);
    if (not seam.has_value())
    {
        out.reason = std::format("no RT chain {} <- {}", cand.new_room_name, cand.old_room_name);
        return out;
    }
    out.seam = seam.value();
    out.seam_ok = true;
    // ★ Asked BEFORE step 3 stubs the new room into memory, or every room would look remembered (a re-entry has a twin
    //   already; a room born by this crossing has none).
    const bool new_room_was_remembered = remembered_room(live_->get_node(cand.new_room_id));

    const std::uint64_t old_idx = room_index(live_->get_node(cand.old_room_id).value());
    const std::uint64_t new_idx = room_index(live_->get_node(cand.new_room_id).value());
    const std::string old_prefix = std::format("r{}_", old_idx);
    const std::string new_prefix = std::format("r{}_", new_idx);

    // ── 2. COPY THE OLD ROOM'S SUBTREE ─────────────────────────────────────────────────────────
    // It hangs off whatever memory already holds: the stub a previous eviction left for this room
    // (keeping that edge, which carries the seam measured back then), or root for the first one.
    const auto mem_root = mem_->get_node("root");
    if (not mem_root.has_value()) { out.reason = "memory graph has no root"; return out; }

    // ★ EVERY ROOM HANGS FROM ROOT (2026-10-12, user decision). Memory used to chain the rooms (root → r1_room_1 →
    //   r2_room_2, the seam on the RT edge between them), which made the first room visited the parent of every other
    //   one — an accident of the tour, not a fact about the building. Each room is now its own frame under root; the
    //   identity on the root edge carries NO metric meaning (there is no global frame), and the relation between two
    //   rooms is the passage: its two door faces, one per room (seam_through_passage()).
    //   ⚠ inner_eigen across two rooms now composes through root's identities and returns a WRONG transform rather
    //   than nullopt. Cross-room geometry must go through seam_through_passage().
    const std::uint64_t mem_parent = mem_root->id();
    const auto root_edge = rt_attrs(Mat::RTMat::Identity());

    IdMap map;
    if (not copy_subtree(cand.old_room_id, mem_parent, old_prefix, old_idx, root_edge, map, out))
        return out;
    out.mem_old_room_id = map.at(cand.old_room_id);

    // ── 3. THE NEW ROOM, ALSO UNDER ROOT ───────────────────────────────────────────────────────
    // Born with the new room's geometry as it stands now and FILLED, not duplicated, when that room is itself
    // evicted.
    const auto new_room_live = live_->get_node(cand.new_room_id);
    if (not new_room_live.has_value()) { out.reason = "the new room vanished mid-eviction"; return out; }

    // ★ WHOLE SUBTREE, not only a stub: after a PROMOTION the old room is deleted from the live graph
    // and door_concept retires the entry mirror with its source, so this is the only moment the new
    // room's side of the crossed door can be remembered. For a room with nothing under it yet (the
    // two-room stage) the subtree IS the stub.
    IdMap new_map;
    if (not copy_subtree(cand.new_room_id, mem_parent, new_prefix, new_idx, root_edge, new_map, out))
        return out;
    out.mem_new_room_id = new_map.at(cand.new_room_id);
    hang_walls_on_floor(map);
    hang_walls_on_floor(new_map);

    // ── 4. THE DOOR, ONCE PER SIDE ─────────────────────────────────────────────────────────────
    // The crossed door is the one nearest the robot in the room it is leaving: an argmin over the
    // doors that exist, not a distance test — no door is ever excluded by it.
    std::vector<DSR::Node> doors;
    for (const auto &[live_id, _] : map)
        if (const auto n = live_->get_node(live_id); n.has_value() and is_door(*live_, n.value()))
            doors.push_back(n.value());

    // THE DOOR PAIR, WHEN KNOWN, IS NOT RE-GUESSED: a promotion hands over the pair the live passage
    // matched while the room was proto (entry mirror ↔ the current room's door). Otherwise, the door
    // nearest the robot in the room being left.
    if (cand.old_door_id != 0 and map.contains(cand.old_door_id))
        std::erase_if(doors, [&](const DSR::Node &d) { return d.id() != cand.old_door_id; });

    if (not doors.empty())
    {
        const auto robot_in_old = ie->get_transformation_matrix(cand.old_room_name, cand.robot_name);
        const Mat::Vector3d ref = robot_in_old.has_value() ? robot_in_old->translation()
                                                           : Mat::Vector3d{0, 0, 0};
        const DSR::Node *chosen = nullptr;
        Mat::RTMat chosen_in_old = Mat::RTMat::Identity(), chosen_in_new = Mat::RTMat::Identity();
        double best = std::numeric_limits<double>::max();
        for (const auto &d : doors)
        {
            const auto in_old = ie->get_transformation_matrix(cand.old_room_name, d.name());
            const auto in_new = ie->get_transformation_matrix(cand.new_room_name, d.name());
            if (not in_old.has_value() or not in_new.has_value()) continue;
            if (const double dist = (in_old->translation() - ref).norm(); dist < best)
            {
                best = dist; chosen = &d; chosen_in_old = in_old.value(); chosen_in_new = in_new.value();
            }
        }

        if (chosen != nullptr)
        {
            // ★ A PROTO'S FRAME IS THE APERTURE (2026-10-22, Task S): when the new room is a proto-room the seam is also
            //   measured THROUGH THE CROSSED DOOR -- T(new<-old) = inverse of the aperture pose in the old room -- and
            //   fused with the robot chain by covariance (seam_fusion.h). gold1 stored a chain seam of (-14262, +1668)
            //   between rooms 4 m apart; nothing contradicted it because the chain was the only measurement.
            if (const auto fused = fuse_seam_through_aperture(cand, *chosen, chosen_in_old, out.seam, *ie, new_room_was_remembered); fused.has_value())
            {
                out.seam = fused.value();
                chosen_in_new = out.seam * chosen_in_old;
            }

            out.door_name   = chosen->name();
            out.door_in_old = chosen_in_old.translation();
            out.door_in_new = chosen_in_new.translation();

            // ★ ONE DOORWAY NODE, FRAMELESS. The room-side FACE (`r0_door_1`, already in memory with
            // the subtree) is room 0's own measurement of the hole, in room 0's frame. The doorway
            // node is the physical hole itself: no RT parent and no pose, only identity and, later,
            // the passage statistic. Three things follow, and they are the reason for the shape:
            //
            //  - The far side's pose is NOT stored. It is composable -- memory's RT tree is
            //    r0_room --RT[seam]--> r1_room, so inner_eigen answers "where is this door in room
            //    1?" from the same numbers. Storing it too would be a second pose for one physical
            //    hole, i.e. a thing that can disagree with itself.
            //  - The far FACE, when door_concept has one (the entry mirror), comes over with the new
            //    room's subtree and takes the second `match` edge: an INDEPENDENT measurement of the
            //    same hole, whose disagreement with the first is a measurement of the seam.
            //    When it has none, a face is DERIVED from the seam measured here (see below) -- the
            //    only way memory can relate the two rooms -- and replaced by the real one later.
            //  - `other_side_door_name` is gone. Two `match` edges into one node say it
            //    structurally, and a string naming a peer node is exactly what goes stale when
            //    door_concept recycles a freed `door_N`.
            const auto mem_face = map.find(chosen->id());
            if (mem_face != map.end())
            {
                out.mem_door_old = old_prefix + chosen->name();

                // Reuse the passage this face already belongs to; only mint one the first time.
                std::uint64_t doorway_id = 0;
                for (const auto &dw : passages())
                    for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(dw, "match"))
                        if (e.to() == mem_face->second) doorway_id = dw.id();

                // ★ THE SAME HOLE UNDER A NEW NAME (2026-10-12). A re-entered room's crossed door is RECREATED by
                //   door_concept with a fresh name, so the face just written (r1_door_39) is new while memory already
                //   holds that hole as r1_door_19 in passage_0 — and a second passage was minted for it (live try29).
                //   Same memory room, same place (the room is archived in its own frame both times) ⇒ same hole: reuse
                //   that passage and let the new face replace the stale one.
                if (doorway_id == 0)
                {
                    auto mie = mem_->get_inner_eigen_api();
                    const std::string mroom = old_prefix + cand.old_room_name;
                    const auto here = mie->get_transformation_matrix(mroom, old_prefix + chosen->name());
                    const double half_w = 0.5 * live_->get_attrib_by_name<width_m_att>(*chosen).value_or(1.f);
                    double best_d = half_w;
                    std::uint64_t stale_face = 0;
                    if (here.has_value())
                        for (const auto &dw : passages())
                            for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(dw, "match"))
                            {
                                const auto f = mem_->get_node(e.to());
                                if (not f.has_value() or f->id() == mem_face->second or not is_descendant(f->id(), mroom)) continue;
                                const auto there = mie->get_transformation_matrix(mroom, f->name());
                                if (not there.has_value()) continue;
                                if (const double d = (there->translation() - here->translation()).head<2>().norm(); d < best_d)
                                { best_d = d; doorway_id = dw.id(); stale_face = f->id(); }
                            }
                    if (stale_face != 0)
                    {
                        const auto stale = mem_->get_node(stale_face);
                        qInfo().noquote() << QString::fromStdString(std::format(
                            "[eviction] {} is the hole memory already knows as {} ({:.2f} m apart) — same passage, face replaced",
                            old_prefix + chosen->name(), stale.has_value() ? stale->name() : std::string("?"), best_d));
                        mem_->delete_node(stale_face);
                    }
                }

                if (doorway_id == 0)
                {
                    std::uint64_t n = 0;
                    for (const auto &dw : passages())
                        if (const auto digits = trailing_index(dw.name()); not digits.empty())
                        {
                            std::uint64_t v = 0;
                            if (std::from_chars(digits.data(), digits.data() + digits.size(), v).ec == std::errc{})
                                n = std::max(n, v + 1);
                        }
                    // TYPE `metaconcept`, SUBTYPE "passage". Its registered comment in cortex is the
                    // spec: "a concept OVER concepts (level-2): an arrangement/grouping that BELIEVES
                    // in a relation among other nodes rather than in a solid body of its own" — an
                    // abstract passage is exactly that, a relation between two rooms with no body.
                    // ★ NOT the `door` node type, which only LOOKS vacant: door_concept's owned-node
                    // sweep is {types object,door} x {name starts with "door"}
                    // (common/owned_nodes/owned_nodes.h:58), so a door-typed node named doorway_*
                    // would be deleted by name at door_concept's next startup if one ever existed in
                    // the LIVE graph — silently, with nobody logging why. Immaterial on domain 2,
                    // free to avoid.
                    DSR::Node dw;
                    dw.type("metaconcept");
                    dw.name(std::format("passage_{}", n));
                    mem_->add_or_modify_attrib_local<object_subtype_att>(dw, std::string("passage"));
                    mem_->add_or_modify_attrib_local<level_att>(dw, 0);
                    if (const auto id = mem_->insert_node(dw); id.has_value())
                        doorway_id = id.value();
                }

                if (doorway_id != 0)
                {
                    if (const auto dwn = mem_->get_node(doorway_id); dwn.has_value())
                        out.mem_doorway = dwn->name();
                    // match → each face of the hole, one per room. No `exit` edge (removed 2026-10-12): the room
                    // a face belongs to is its RT ancestor, so the two faces already name both rooms, and "exit"
                    // recorded only the direction of the FIRST crossing.
                    auto m = DSR::Edge::create<match_edge_type>(doorway_id, mem_face->second);
                    mem_->insert_or_assign_edge(std::move(m));

                    // THE SECOND FACE: the new room's own door node for the same hole (the entry mirror,
                    // copied with the new room's subtree above). Two `match` edges into one passage.
                    if (const auto nf = new_map.find(cand.new_door_id); cand.new_door_id != 0 and nf != new_map.end())
                    {
                        auto m2 = DSR::Edge::create<match_edge_type>(doorway_id, nf->second);
                        mem_->insert_or_assign_edge(std::move(m2));
                        if (const auto nn = mem_->get_node(nf->second); nn.has_value())
                            out.mem_door_new = nn->name();
                    }

                    // ★ ONE FACE PER ROOM PER PASSAGE (2026-10-18, live try53). A RE-ENTERED room already has its face of
                    //   this hole in memory (r1_door_1, written when it was evicted through it), so a real face just
                    //   matched retires any derived stand-in for that room, and with no real face the remembered one is
                    //   kept rather than deriving a second. try53 derived r1_door_1_entry beside r1_door_1: passage_0 then
                    //   had two room_1 faces, recall took room_1 as the room behind room_1's own door (no recall_room_2),
                    //   and the seam composed the turned face without its π (room_2 drawn over room_1).
                    std::vector<DSR::Node> new_room_faces;
                    if (const auto nroom = mem_->get_node(out.mem_new_room_id); nroom.has_value())
                        if (const auto dwn = mem_->get_node(doorway_id); dwn.has_value())
                            for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(dwn.value(), "match"))
                                if (const auto f = mem_->get_node(e.to()); f.has_value() and is_descendant(f->id(), nroom->name()))
                                    new_room_faces.push_back(f.value());
                    if (not out.mem_door_new.empty())
                    {
                        for (const auto &f : new_room_faces)
                            if (f.name() != out.mem_door_new and f.name().ends_with(kDerivedFaceSuffix))
                            {
                                qInfo().noquote() << QString::fromStdString(std::format(
                                    "[eviction] {} replaces the derived face {}", out.mem_door_new, f.name()));
                                mem_->delete_node(f.id());
                            }
                    }
                    else if (const auto real = std::ranges::find_if(new_room_faces, [](const auto &f)
                                 { return not f.name().ends_with(kDerivedFaceSuffix); });
                             real != new_room_faces.end())
                    {
                        out.mem_door_new = real->name();
                        qInfo().noquote() << QString::fromStdString(std::format(
                            "[eviction] {} already holds its face of {} as {} — kept, nothing derived",
                            cand.new_room_name, chosen->name(), real->name()));
                    }

                    // ★ NO ENTRY MIRROR ⇒ A DERIVED FACE (2026-10-18, live 12:22: room_2 born through door_5 had none,
                    //   so passage_0 kept one face and memory could not relate room_2 to room_1 at all). The pose is
                    //   T(new <- door), read through the ROBOT while both its RT edges exist -- the same measurement
                    //   the seam is, not a convention -- so seam_through_passage() composes back to exactly that seam.
                    //   Named <door>_entry: door_concept only mints door_N, so it can never collide with a real face,
                    //   and when the new room is itself evicted through this hole, the "same hole under a new name"
                    //   rule above deletes it in favour of door_concept's own face.
                    if (out.mem_door_new.empty())
                        if (const auto nroom = mem_->get_node(out.mem_new_room_id); nroom.has_value())
                        {
                            const std::string fname = new_prefix + chosen->name() + std::string(kDerivedFaceSuffix);
                            std::optional<std::uint64_t> fid;
                            if (const auto existing = mem_->get_node(fname); existing.has_value())
                                fid = existing->id();
                            else
                            {
                                DSR::Node f;
                                f.type("object");
                                f.name(fname);
                                mem_->add_or_modify_attrib_local<object_subtype_att>(f, std::string("door"));
                                mem_->add_or_modify_attrib_local<width_m_att>(
                                    f, live_->get_attrib_by_name<width_m_att>(*chosen).value_or(1.f));
                                mem_->add_or_modify_attrib_local<parent_att>(f, nroom->id());
                                mem_->add_or_modify_attrib_local<level_att>(f, mem_->get_node_level(nroom.value()).value_or(0) + 1);
                                fid = mem_->insert_node(f);   // LVALUE (see twin())
                            }
                            if (fid.has_value())
                            {
                                auto rt = DSR::Edge::create<RT_edge_type>(nroom->id(), fid.value(), rt_attrs(chosen_in_new));
                                mem_->insert_or_assign_edge(std::move(rt));
                                auto m3 = DSR::Edge::create<match_edge_type>(doorway_id, fid.value());
                                mem_->insert_or_assign_edge(std::move(m3));
                                out.mem_door_new = fname;
                                out.mem_door_new_derived = true;
                                qInfo().noquote() << QString::fromStdString(std::format(
                                    "[eviction] {} has no face of {}: DERIVED {} from the seam measured through the robot",
                                    cand.new_room_name, chosen->name(), fname));
                            }
                        }
                }
            }
        }
    }

    if (out.mem_door_new.empty())
        qWarning().noquote() << QString::fromStdString(std::format(
            "[eviction] ⚠ {} has no face of the crossed door, so memory does not relate it to {} (no passage pair)",
            cand.new_room_name, cand.old_room_name));

    // ── 5. VERIFY, THEN FLIP `current` ─────────────────────────────────────────────────────────
    // The flip is the fence: it is what tells every other agent that the room it was working in is
    // no longer current, so each can let ITS OWN nodes go. Nothing is deleted here.
    for (const auto &[live_id, mem_id] : map)
        if (not mem_->get_node(mem_id).has_value())
        {
            out.reason = std::format("memory lost node {} right after writing it", mem_id);
            return out;
        }

    for (const auto &[live_id, mem_id] : new_map)
        if (not mem_->get_node(mem_id).has_value())
        {
            out.reason = std::format("memory lost node {} right after writing it", mem_id);
            return out;
        }

    archived_.insert(cand.old_room_id);

    // PROMOTE IN PLACE: same node id, the `proto` self-edge removed (room_concept's contract). Before the
    // flip, so no reader ever sees `current` pointing at a room still marked proto.
    if (cand.promote)
        out.promoted = live_->delete_edge(cand.new_room_id, cand.new_room_id, "proto")
                       or not live_->get_edge(cand.new_room_id, cand.new_room_id, "proto").has_value();

    live_->delete_edge(cand.robot_id, cand.old_room_id, "current");
    auto cur = DSR::Edge::create<current_edge_type>(cand.robot_id, cand.new_room_id);
    out.current_flipped = live_->insert_or_assign_edge(std::move(cur));
    out.ok = out.current_flipped and (not cand.promote or out.promoted);
    if (not out.ok)
    {
        out.reason = "the copy landed but the `current` edge / proto promotion could not be written";
        return out;
    }

    // THE OLD ROOM LEAVES THE WORKING MEMORY. Decided 2026-09-14 (user): ltsm deletes it, after the copy
    // is verified and the fence has moved. Owners that follow `current` (door_concept's release_room) will
    // already be letting their own nodes go; this removes whatever is left, room_concept's room/floor/walls
    // included.
    if (cand.remove_old)
        out.live_removed = remove_live_room(cand.old_room_id);
    return out;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
bool is_descendant(DSR::DSRGraph &mem, std::uint64_t node_id, const std::string &room)
{
    const auto r = mem.get_node(room);
    if (not r.has_value()) return false;
    for (auto n = mem.get_node(node_id); n.has_value();)
    {
        if (n->id() == r->id()) return true;
        const auto p = mem.get_attrib_by_name<parent_att>(n.value());
        if (not p.has_value() or p.value() == n->id()) return false;
        n = mem.get_node(p.value());
    }
    return false;
}

std::optional<Mat::RTMat> seam_through_passage(DSR::DSRGraph &mem, const std::string &passage, const std::string &room_to,
                                               const std::string &room_from)
{
    // The two faces are ONE physical hole (door_concept births the entry mirror at the exact pose of its source), so
    // T(face_to <- face_from) = I and   T(to <- from) = T(to <- face_to) · T(face_from <- from).
    const auto dw = mem.get_node(passage);
    if (not dw.has_value()) return std::nullopt;
    auto ie = mem.get_inner_eigen_api();
    std::optional<Mat::RTMat> to_face, face_from;
    bool to_derived = false, from_derived = false;
    for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(dw.value(), "match"))
    {
        const auto f = mem.get_node(e.to());
        if (not f.has_value()) continue;
        // Which room a face belongs to: the one that composes to it (its RT ancestor). Siblings under root also
        // "compose" through root's identities, so ask for the room the face is a DESCENDANT of.
        // A room holding both a real and a derived face of one hole (memory written before 2026-10-18) uses the real one.
        const bool d = f->name().ends_with(kDerivedFaceSuffix);
        if (is_descendant(mem, f->id(), room_to) and (not to_face.has_value() or (to_derived and not d)))
            if (auto t = ie->get_transformation_matrix(room_to, f->name()); t.has_value())
            { to_face = t; to_derived = d; }
        if (is_descendant(mem, f->id(), room_from) and (not face_from.has_value() or (from_derived and not d)))
            if (auto t = ie->get_transformation_matrix(room_from, f->name()); t.has_value())
            { face_from = t->inverse(); from_derived = d; }
    }
    if (not to_face.has_value() or not face_from.has_value()) return std::nullopt;
    const bool derived = to_derived or from_derived;
    Mat::RTMat seam(to_face.value() * face_from.value());
    // ★ THE TWO FACES MAY BE THE SAME HOLE TURNED BY π (2026-10-12). door_concept orients a door's frame from the room
    //   that fits it (+y into that room), so the face seen from the other room is the same doorway rotated half a turn.
    //   Rooms do not overlap: if room_from's outline lands on room_to's side of the face, the turned face is the one.
    const auto centroid = [&](const std::string &room) -> std::optional<Eigen::Vector2d>
    {
        const auto r = mem.get_node(room);
        if (not r.has_value()) return std::nullopt;
        const auto px = mem.get_attrib_by_name<delimiting_polygon_x_att>(r.value());
        const auto py = mem.get_attrib_by_name<delimiting_polygon_y_att>(r.value());
        if (not px.has_value() or not py.has_value() or px->get().empty() or px->get().size() != py->get().size())
            return std::nullopt;
        Eigen::Vector2d c = Eigen::Vector2d::Zero();
        for (std::size_t i = 0; i < px->get().size(); ++i) c += Eigen::Vector2d(px->get()[i], py->get()[i]);
        return c / static_cast<double>(px->get().size());
    };
    //   NOT for a DERIVED face: it was built from the other face through the measured seam, so the two are exactly
    //   consistent by construction and there is no independent orientation to correct.
    if (derived) return seam;
    if (const auto c_to = centroid(room_to), c_from = centroid(room_from); c_to.has_value() and c_from.has_value())
    {
        const Eigen::Vector2d o = to_face->translation().head<2>();
        const Eigen::Vector2d n = to_face->linear().col(1).head<2>();
        const Eigen::Vector2d from_in_to = (seam * Eigen::Vector3d(c_from->x(), c_from->y(), 0.0)).head<2>();
        if (n.dot(c_to.value() - o) * n.dot(from_in_to - o) > 0.0)
        {
            Mat::RTMat flip = Mat::RTMat::Identity();
            flip.linear() = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()).toRotationMatrix();
            seam = Mat::RTMat(to_face.value() * flip * face_from.value());
        }
    }
    return seam;
}

std::optional<Mat::RTMat> RoomEviction::seam_through_passage(const std::string &passage, const std::string &room_to,
                                                             const std::string &room_from) const
{
    return ltsm::seam_through_passage(*mem_, passage, room_to, room_from);
}

bool RoomEviction::is_descendant(std::uint64_t node_id, const std::string &room) const
{
    return ltsm::is_descendant(*mem_, node_id, room);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
void RoomEviction::hang_walls_on_floor(const IdMap &map)
{
    auto ie = live_->get_inner_eigen_api();
    for (const auto &[live_id, mem_id] : map)
    {
        const auto wall = live_->get_node(live_id);
        if (not wall.has_value() or wall->type() != "wall") continue;
        const auto live_parent = live_->get_attrib_by_name<parent_att>(wall.value());
        if (not live_parent.has_value()) continue;
        const auto room = live_->get_node(live_parent.value());
        if (not room.has_value() or room->type() != "room") continue;       // already under a floor

        // The floor of THAT room, among the nodes copied with it.
        std::optional<DSR::Node> floor;
        for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(room.value(), "RT"))
            if (const auto c = live_->get_node(e.to()); c.has_value() and c->type() == "floor" and map.contains(c->id()))
                floor = c;
        if (not floor.has_value()) continue;                                  // no floor: the room is the only parent there is

        const auto t = ie->get_transformation_matrix(floor->name(), wall->name());
        if (not t.has_value()) continue;
        const auto mem_floor = map.at(floor->id());
        const auto mem_room  = map.at(room->id());

        mem_->delete_edge(mem_room, mem_id, "RT");
        auto e = DSR::Edge::create<RT_edge_type>(mem_floor, mem_id, rt_attrs(t.value()));
        mem_->insert_or_assign_edge(std::move(e));
        if (auto w = mem_->get_node(mem_id); w.has_value())
        {
            mem_->add_or_modify_attrib_local<parent_att>(w.value(), mem_floor);
            if (const auto f = mem_->get_node(mem_floor); f.has_value())
                mem_->add_or_modify_attrib_local<level_att>(w.value(), mem_->get_node_level(f.value()).value_or(0) + 1);
            mem_->update_node(w.value());
        }
    }
}

int RoomEviction::remove_live_room(std::uint64_t room_id)
{
    const auto room = live_->get_node(room_id);
    const std::string name = room.has_value() ? room->name() : std::string("<gone>");
    int removed = sweep_orphans(room_id, "the room it hung from was promoted away");
    if (live_->delete_node(room_id)) ++removed;
    qInfo() << "[eviction] removed room" << QString::fromStdString(name) << "from the working memory ("
            << removed << "node(s))";
    return removed;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Promotion evidence
//////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<PromotionReading> RoomEviction::promotion_evidence(double body_radius) const
{
    std::vector<PromotionReading> out;
    const auto robot = robot_node();
    if (not robot.has_value()) return out;
    auto rt = live_->get_rt_api();

    // ★ NOT ONLY WHILE IT IS A PROTO (2026-10-11). room_concept removes `proto` when the new room's LAYOUT is promoted
    //   — its own meaning of the edge — and that can happen before this agent has decided the identity (live: 0.1 s
    //   after birth, so `current` never moved and nothing would ever move it). The evidence is unchanged by that: the
    //   room keeps the crossed aperture as its frame (room_concept promotes in place). So a room also qualifies when
    //   the robot's RT to it is LIVE while its RT to the current room is RETIRED (valid=false) — the crossing hand-over.
    const auto cur = rc::room::current_edge_room(*live_);
    bool cur_retired = false;
    if (cur.has_value())
        if (const auto ce = live_->get_edge(robot->id(), cur.value(), "RT"); ce.has_value())
            cur_retired = not live_->get_attrib_by_name<valid_att>(ce.value()).value_or(true);
    for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(robot.value(), "RT"))
    {
        const auto proto = live_->get_node(e.to());
        if (not proto.has_value() or proto->type() != "room") continue;
        const bool entered = cur_retired and proto->id() != cur.value()
                             and live_->get_attrib_by_name<valid_att>(e).value_or(true);
        if (not rc::room::is_proto(*live_, proto->id()) and not entered)
            continue;

        PromotionReading r;
        r.proto_id = proto->id();
        r.proto_name = proto->name();
        r.body_radius = body_radius;

        // E = T(robot <- proto), the edge as written; its covariance is over (x, y, yaw) of E, in the
        // PARENT (robot) frame -- cortex's convention, and what room_concept propagates into.
        const auto E = rt->get_edge_RT_as_rtmat(e);
        const auto C = rt->get_edge_RT_covariance(e);
        if (not E.has_value()) { r.why_not = "no pose on the robot->proto edge"; out.push_back(r); continue; }
        // ★ NO COVARIANCE ⇒ NO DECISION. Φ of a zero width is a step function, i.e. a hidden threshold on
        // the mean; a pose that does not say how well it is known cannot say the robot is surely inside.
        if (not C.has_value()) { r.why_not = "no pose covariance on the robot->proto edge"; out.push_back(r); continue; }

        const Eigen::Vector2d te(E->translation().x(), E->translation().y());
        const double th = std::atan2(E->rotation()(1, 0), E->rotation()(0, 0));
        // Robot in the proto frame: t = -R(th)^T te. Jacobian w.r.t. (te, th):
        //   ∂t/∂te = -R(-th),   ∂t/∂th = R'(-th)·te,  R'(a) = [[-sin a, -cos a], [cos a, -sin a]].
        const double c = std::cos(-th), s = std::sin(-th);
        Eigen::Matrix2d Rm;  Rm  << c, -s, s, c;
        Eigen::Matrix2d dR;  dR  << -s, -c, c, -s;
        const Eigen::Vector2d t = -Rm * te;
        Eigen::Matrix<double, 2, 3> J;
        J.leftCols<2>() = -Rm;
        J.col(2) = dR * te;
        static constexpr int slot[3] = {0, 1, 5};   // SE(2) inside the 6x6 SE(3) block [x y z rx ry rz]
        Eigen::Matrix3d S;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                S(i, j) = (*C)(slot[i], slot[j]);
        const Eigen::Matrix2d St = J * S * J.transpose();

        r.s = t.y();                                   // +y is OUTWARD from the room left behind
        r.sigma_s = std::sqrt(std::max(St(1, 1), 0.));
        if (not (r.sigma_s > 0.)) { r.why_not = "zero pose variance across the aperture"; out.push_back(r); continue; }
        r.p_inside = 0.5 * std::erfc(-((r.s - body_radius) / r.sigma_s) / std::sqrt(2.));
        r.valid = true;
        out.push_back(r);
    }
    return out;
}

std::optional<Candidate> RoomEviction::promotion_candidate(const PromotionReading &reading) const
{
    const auto robot = robot_node();
    const auto cur = rc::room::current_edge_room(*live_);
    if (not robot.has_value() or not cur.has_value()) return {};     // no fence to move: nothing to promote from
    const auto old_room = live_->get_node(cur.value());
    const auto new_room = live_->get_node(reading.proto_id);
    if (not old_room.has_value() or not new_room.has_value()) return {};
    if (archived_.contains(old_room->id())) return {};

    Candidate c;
    c.robot_id      = robot->id();
    c.robot_name    = robot->name();
    c.old_room_id   = old_room->id();
    c.old_room_name = old_room->name();
    c.new_room_id   = new_room->id();
    c.new_room_name = new_room->name();
    c.old_birth_ms  = live_->get_attrib_by_name<timestamp_creation_att>(old_room.value()).value_or(0ULL);
    c.new_birth_ms  = live_->get_attrib_by_name<timestamp_creation_att>(new_room.value()).value_or(0ULL);
    // ★ `proto` IS ROOM_CONCEPT'S (2026-10-12, user decision): "layout still provisional", removed by room_concept when
    //   the layout is promoted. This agent's identity decision is the `current` move alone.
    c.promote    = false;
    c.remove_old = true;
    c.p_inside   = static_cast<float>(reading.p_inside);
    c.crossing_born = true;   // it was read through the crossing evidence: its frame is the crossed aperture
    return c;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Startup seed
//////////////////////////////////////////////////////////////////////////////////////////////////
SeedOutcome RoomEviction::seed_from_working_memory()
{
    SeedOutcome so;
    const auto mem_root = mem_->get_node("root");
    if (not mem_root.has_value()) return so;

    // Oldest first, so when the working memory holds more than one room (a hand-over in progress)
    // the later ones hang off the earlier one exactly as an eviction would place them. A room with
    // no birth stamp sorts first; the order only decides who is the anchor, never what is copied.
    auto rooms = live_->get_nodes_by_type("room");
    const auto birth = [this](const DSR::Node &n)
    { return live_->get_attrib_by_name<timestamp_creation_att>(n).value_or(0ULL); };
    std::ranges::stable_sort(rooms, [&](const auto &a, const auto &b) { return birth(a) < birth(b); });

    auto ie = live_->get_inner_eigen_api();
    std::optional<std::pair<std::string, std::uint64_t>> anchor;   // (live name, memory id)

    for (const auto &room : rooms)
    {
        // A proto-room's identity is still undecided (that decision is this agent's, later); writing it
        // into long-term memory now would remember a room that may turn out to be the one we are in.
        if (rc::room::is_proto(*live_, room.id()))
        {
            ++so.rooms_proto;
            continue;
        }
        ++so.rooms_seen;

        // ★ ALREADY REMEMBERED ⇒ LEAVE IT. Memory's copy may carry what perception cannot rebuild
        // (the passage history, a seam measured at a crossing); a startup snapshot must not overwrite
        // it. Identity is the name until place recognition exists -- with persistence off (the
        // default) memory starts empty and this never fires; with it on, a DIFFERENT room that
        // room_concept happens to name the same would be taken for the remembered one.
        if (const auto known = remembered(room); known.has_value())
        {
            ++so.rooms_known;
            so.rooms.emplace_back(room.name(), known->name());
            if (not anchor.has_value()) anchor.emplace(room.name(), known->id());
            continue;
        }

        const std::uint64_t idx = room_index(room);
        const std::string prefix = std::format("r{}_", idx);

        // WHERE IT HANGS. The first room sits on memory's root. Any further room hangs off the first,
        // its pose in that room's frame read through the live RT tree -- the same measurement the
        // eviction's seam is. No chain between them (a room not linked to the robot yet) ⇒ root, and
        // the eviction re-parents it when it measures the seam.
        std::uint64_t parent = mem_root->id();
        auto edge = rt_attrs(Mat::RTMat::Identity());
        if (anchor.has_value())
        {
            if (const auto t = ie->get_transformation_matrix(anchor->first, room.name()); t.has_value())
            {
                parent = anchor->second;
                edge = rt_attrs(t.value());
            }
            else
                qWarning() << "[seed] no RT chain" << QString::fromStdString(anchor->first) << "<-"
                           << QString::fromStdString(room.name())
                           << "-- hanging it on memory's root until an eviction measures the seam";
        }

        IdMap map;
        Outcome out;
        if (not copy_subtree(room.id(), parent, prefix, idx, edge, map, out, Scope::Structure))
        {
            qWarning() << "[seed] could not write" << QString::fromStdString(room.name())
                       << "into memory:" << QString::fromStdString(out.reason);
            continue;
        }

        hang_walls_on_floor(map);
        ++so.rooms_created;
        so.nodes += out.nodes_copied;
        for (const auto &[live_id, _] : map)
            if (const auto n = live_->get_node(live_id); n.has_value() and is_door(*live_, n.value()))
                ++so.doors;
        const std::string mem_name = prefix + room.name();
        so.rooms.emplace_back(room.name(), mem_name);
        if (not anchor.has_value()) anchor.emplace(room.name(), map.at(room.id()));
    }
    return so;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
bool RoomEviction::ensure_current_edge()
{
    const auto robot = robot_node();
    if (not robot.has_value()) return false;

    // BOOTSTRAP ONLY: once ANY `current` edge exists, moving it is a decision (promotion / eviction), never this.
    // It used to check only "is THIS room already current": after a crossing the robot has an RT edge to both
    // rooms, and live (2026-10-11, eviction OFF) this re-added `current` -> room_1 beside the promoted room_2 — two
    // `current` edges, door_concept stopped following and the controller flipped back to room_1.
    // Only a `current` edge to a LIVE room counts: one left pointing at a room deleted by a previous run (the robot
    // node outlives the rooms) is removed here, or it would block the bootstrap for ever (live 2026-10-11).
    bool has_live_current = false;
    for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(robot.value(), "current"))
    {
        if (const auto dst = live_->get_node(e.to()); dst.has_value() and dst->type() == "room") has_live_current = true;
        else if (live_->delete_edge(robot->id(), e.to(), "current"))
            qInfo() << "[eviction] removed a stale `current` edge to node" << e.to() << "(no such room)";
    }
    if (has_live_current) return false;
    const auto rooms = rooms_under_robot(robot.value());
    if (rooms.size() != 1) return false;                        // 0: nothing to point at. 2+: eviction's job.

    for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(robot.value(), "current"))
        if (e.to() == rooms.front().id()) return false;         // already there

    auto cur = DSR::Edge::create<current_edge_type>(robot->id(), rooms.front().id());
    const bool ok = live_->insert_or_assign_edge(std::move(cur));
    if (ok)
        qInfo() << "[eviction] bootstrap: placed the first `current` edge on"
                << QString::fromStdString(rooms.front().name());
    return ok;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
int RoomEviction::sweep_orphans(std::uint64_t room_id, std::string_view why)
{
    // BACKSTOP ONLY. Everything here should have been removed by its OWNER when the room stopped
    // being current; what is left belongs to an agent that is not running. Deepest first, so no
    // node is orphaned mid-sweep.
    std::vector<std::pair<std::uint64_t, int>> victims;   // id, level
    std::deque<std::uint64_t> q{room_id};
    while (not q.empty())
    {
        const auto id = q.front(); q.pop_front();
        const auto n = live_->get_node(id);
        if (not n.has_value()) continue;
        if (id != room_id)
            victims.emplace_back(id, live_->get_node_level(n.value()).value_or(0));
        for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(n.value(), "RT"))
            if (const auto c = live_->get_node(e.to()); c.has_value() and c->type() != "robot" and c->type() != "root")
                q.push_back(e.to());
    }
    std::ranges::sort(victims, [](const auto &a, const auto &b) { return a.second > b.second; });

    int removed = 0;
    for (const auto &[id, level] : victims)
    {
        const auto n = live_->get_node(id);
        const std::string name = n.has_value() ? n->name() : std::string("<gone>");
        if (live_->delete_node(id))
        {
            qInfo().noquote() << QString::fromStdString(std::format(
                "[eviction] removed {} id {} level {} -- {}", name, id, level, why));
            ++removed;
        }
    }
    return removed;
}

}   // namespace ltsm
