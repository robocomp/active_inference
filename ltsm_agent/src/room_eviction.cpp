/*
 * room_eviction.cpp — see room_eviction.h for the shape of the transaction and ../EVICTION.md for
 * why it is shaped that way.
 */
#include "room_eviction.h"

#include <QDebug>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <deque>
#include <utility>
#include <format>

namespace ltsm
{

namespace
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
    std::vector<DSR::Node> rooms;
    for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(robot, "RT"))
        if (const auto child = live_->get_node(e.to()); child.has_value() and child->type() == "room")
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
    if (const auto id = live_->get_attrib_by_name<room_id_att>(room); id.has_value())
        return id.value();
    if (const auto digits = trailing_index(room.name()); not digits.empty())
    {
        std::uint64_t v = 0;
        // from_chars, never stoull/atoi: these machines run es_ES and CLAUDE.md is explicit.
        if (std::from_chars(digits.data(), digits.data() + digits.size(), v).ec == std::errc{})
            return v;
    }
    return next_index_++;
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
                                IdMap &map, Outcome &out)
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

        for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(src.value(), "RT"))
        {
            const auto child = live_->get_node(e.to());
            if (not child.has_value()) continue;
            if (child->type() == "robot" or child->type() == "root") continue;   // never climb out
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

    const auto mem_new_room = twin(new_room_live.value(), out.mem_old_room_id, new_prefix, new_idx);
    if (not mem_new_room.has_value()) { out.reason = "could not write the new room's stub"; return out; }
    out.mem_new_room_id = mem_new_room.value();
    {
        auto e = DSR::Edge::create<RT_edge_type>(out.mem_old_room_id, out.mem_new_room_id, rt_attrs(out.seam));
        mem_->insert_or_assign_edge(std::move(e));
        // The `exit` edge is NOT here: it hangs off the doorway node below, because a room pair can
        // have more than one door and only the doorway says which hole was crossed. This RT edge
        // carries the seam and nothing else.
    }

    // ── 4. THE DOOR, ONCE PER SIDE ─────────────────────────────────────────────────────────────
    // The crossed door is the one nearest the robot in the room it is leaving: an argmin over the
    // doors that exist, not a distance test — no door is ever excluded by it.
    std::vector<DSR::Node> doors;
    for (const auto &[live_id, _] : map)
        if (const auto n = live_->get_node(live_id); n.has_value() and is_door(*live_, n.value()))
            doors.push_back(n.value());

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

    archived_.insert(cand.old_room_id);
    live_->delete_edge(cand.robot_id, cand.old_room_id, "current");
    auto cur = DSR::Edge::create<current_edge_type>(cand.robot_id, cand.new_room_id);
    out.current_flipped = live_->insert_or_assign_edge(std::move(cur));
    out.ok = out.current_flipped;
    if (not out.ok) out.reason = "the copy landed but the `current` edge could not be written";
    return out;
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
int RoomEviction::sweep_orphans(std::uint64_t room_id)
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
            qWarning() << "[eviction backstop] removed orphan" << QString::fromStdString(name)
                       << "id" << static_cast<qulonglong>(id) << "level" << level
                       << "— its owner was not running to let it go";
            ++removed;
        }
    }
    return removed;
}

}   // namespace ltsm
