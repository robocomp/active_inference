/*
 * room_eviction.cpp — see room_eviction.h for the shape of the transaction and ../EVICTION.md for
 * why it is shaped that way.
 */
#include "room_eviction.h"

#include "../../common/room_resolve/room_resolve.h"   // rc::room::is_proto

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

    const std::uint64_t old_idx = room_index(live_->get_node(cand.old_room_id).value());
    const std::uint64_t new_idx = room_index(live_->get_node(cand.new_room_id).value());
    const std::string old_prefix = std::format("r{}_", old_idx);
    const std::string new_prefix = std::format("r{}_", new_idx);

    // ── 2. COPY THE OLD ROOM'S SUBTREE ─────────────────────────────────────────────────────────
    // It hangs off whatever memory already holds: the stub a previous eviction left for this room
    // (keeping that edge, which carries the seam measured back then), or root for the first one.
    const auto mem_root = mem_->get_node("root");
    if (not mem_root.has_value()) { out.reason = "memory graph has no root"; return out; }

    std::uint64_t mem_parent = mem_root->id();
    std::map<std::string, DSR::Attribute> root_edge = rt_attrs(Mat::RTMat::Identity());
    if (const auto stub = mem_->get_node(old_prefix + cand.old_room_name); stub.has_value())
    {
        const auto p = mem_->get_attrib_by_name<parent_att>(stub.value());
        if (p.has_value() and mem_->get_node(p.value()).has_value())
        {
            mem_parent = p.value();
            if (const auto e = mem_->get_edge(mem_parent, stub->id(), "RT"); e.has_value())
                root_edge = e->attrs();          // keep the seam this room was archived with
        }
    }

    IdMap map;
    if (not copy_subtree(cand.old_room_id, mem_parent, old_prefix, old_idx, root_edge, map, out))
        return out;
    out.mem_old_room_id = map.at(cand.old_room_id);

    // ── 3. THE NEW ROOM AS A STUB, THE SEAM ON ITS RT EDGE ─────────────────────────────────────
    // Memory's RT tree IS the topological map: root → r0_room → r1_room → … so inner_eigen
    // composes across rooms with no special case. The stub is born with the new room's geometry as
    // it stands now and is FILLED, not duplicated, when that room is itself evicted.
    const auto new_room_live = live_->get_node(cand.new_room_id);
    if (not new_room_live.has_value()) { out.reason = "the new room vanished mid-eviction"; return out; }

    // ★ WHOLE SUBTREE, not only a stub: after a PROMOTION the old room is deleted from the live graph
    // and door_concept retires the entry mirror with its source, so this is the only moment the new
    // room's side of the crossed door can be remembered. For a room with nothing under it yet (the
    // two-room stage) the subtree IS the stub.
    // ★ INVERSE on the root edge. An RT edge parent→child carries the CHILD's pose in the PARENT's
    // frame, i.e. T(room_old <- room_new) = seam⁻¹. Writing the seam itself put memory's r1_room 6.23 m
    // off on the stage fixture; only a seam that is a 180° rotation would have hidden it.
    // The `exit` edge is NOT on this RT edge: it hangs off the passage node below, because a room pair
    // can have more than one door and only the passage says which hole was crossed.
    IdMap new_map;
    if (not copy_subtree(cand.new_room_id, out.mem_old_room_id, new_prefix, new_idx,
                         rt_attrs(out.seam.inverse()), new_map, out))
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
            //  - The far FACE is not fabricated either. When room 1 is evicted, door_concept's own
            //    door node in room 1 comes over with that room's subtree and takes the second
            //    `match` edge: an INDEPENDENT measurement of the same hole, whose disagreement with
            //    the first is a measurement of the seam. A face I derived here could never be that.
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
                    // match → the face we have actually seen; exit → the room reached through THIS
                    // hole. `exit` sits here rather than between the two rooms because a room pair
                    // can have more than one doorway, and only this edge says which one was used.
                    auto m = DSR::Edge::create<match_edge_type>(doorway_id, mem_face->second);
                    mem_->insert_or_assign_edge(std::move(m));
                    auto x = DSR::Edge::create<exit_edge_type>(doorway_id, out.mem_new_room_id);
                    mem_->insert_or_assign_edge(std::move(x));

                    // THE SECOND FACE: the new room's own door node for the same hole (the entry mirror,
                    // copied with the new room's subtree above). Two `match` edges into one passage.
                    if (const auto nf = new_map.find(cand.new_door_id); cand.new_door_id != 0 and nf != new_map.end())
                    {
                        auto m2 = DSR::Edge::create<match_edge_type>(doorway_id, nf->second);
                        mem_->insert_or_assign_edge(std::move(m2));
                        if (const auto nn = mem_->get_node(nf->second); nn.has_value())
                            out.mem_door_new = nn->name();
                    }
                }
            }
        }
    }

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

    for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(robot.value(), "RT"))
    {
        const auto proto = live_->get_node(e.to());
        if (not proto.has_value() or proto->type() != "room" or not rc::room::is_proto(*live_, proto->id()))
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
    c.promote    = true;
    c.remove_old = true;
    c.p_inside   = static_cast<float>(reading.p_inside);
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
