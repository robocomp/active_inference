// PASSAGE IDENTITY (door-identity design, part 2): the passage, not the door name, is the identity of a physical hole, and
// it is ONE identity live and in memory.
//   run from the component root:  ./bin/passage_identity_test
// Scenarios, each on fresh in-process graphs (etc/stage_two_rooms.json live, ltsm_root.json memory):
//   numbering  live and memory passages share ONE numbering; the eviction archives the hole under the LIVE name
//   recross    a hole archived by an eviction, re-crossed with the door re-made as door_17: the live passage takes memory's name
//   renamed    memory knows the hole as r0_door_3 (door_3), live door is door_17 at the same place: same passage, by place
//   two_holes  memory holds a passage for a DIFFERENT hole of the same room: the live crossing gets a fresh number
#include "../src/room_eviction.h"
#include "../src/passage_live.h"
#include "../src/topology_publish.h"
#include "../../common/ltsm_topology/topology_decode.h"
#include <QCoreApplication>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

static int failures = 0;
static void check(bool ok, const std::string &what) { if (not ok) ++failures; std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str()); }

namespace
{
std::vector<std::shared_ptr<DSR::DSRGraph>> keep;   // graphs stay alive until _Exit (DDS teardown is not under test)
int next_domain = 61;

std::shared_ptr<DSR::DSRGraph> graph(const std::string &name, const std::string &file)
{
    const int d = next_domain++;
    auto g = std::make_shared<DSR::DSRGraph>(name, 200 + d, file, true, d);
    keep.push_back(g);
    return g;
}

void put_rt(DSR::DSRGraph &G, std::uint64_t from, std::uint64_t to, double x, double y, double yaw)
{
    auto n = G.get_node(from).value();
    G.get_rt_api()->insert_or_assign_edge_RT(n, to, std::vector<float>{(float)x, (float)y, 0.f}, std::vector<float>{0.f, 0.f, (float)yaw});
}

double yaw_of(const Mat::RTMat &T) { return std::atan2(T.rotation()(1, 0), T.rotation()(0, 0)); }

std::uint64_t add_door(DSR::DSRGraph &G, const std::string &name, std::uint64_t parent, const Mat::RTMat &T_parent_door)
{
    DSR::Node d; d.type("object"); d.name(name);
    G.add_or_modify_attrib_local<object_subtype_att>(d, std::string("door"));
    G.add_or_modify_attrib_local<width_m_att>(d, 0.9f);
    G.add_or_modify_attrib_local<parent_att>(d, parent);
    G.add_or_modify_attrib_local<level_att>(d, G.get_node_level(G.get_node(parent).value()).value_or(0) + 1);
    const auto id = G.insert_node(d).value();
    put_rt(G, parent, id, T_parent_door.translation().x(), T_parent_door.translation().y(), yaw_of(T_parent_door));
    return id;
}

/// The live crossing: room_0 current (door `door_name` on wall_1 at door_1's pose), proto room_1 with an entry mirror
/// door_5 at the same physical place. Returns the hole's pose in room_0 and in room_1.
std::pair<Mat::RTMat, Mat::RTMat> stage_crossing(DSR::DSRGraph &live, const std::string &door_name)
{
    auto ie = live.get_inner_eigen_api();
    const auto T_wall_door = ie->get_transformation_matrix("wall_1", "door_1").value();
    const auto T_room_door = ie->get_transformation_matrix("room_0", "door_1").value();
    const auto T_proto_door = ie->get_transformation_matrix("room_1", "door_1").value();
    if (door_name != "door_1")
    {   // door_concept re-made the door under another name: the old node goes, a new one stands at the same place
        live.delete_node(live.get_node("door_1")->id());
        add_door(live, door_name, 311, T_wall_door);
    }
    add_door(live, "door_5", 400, T_proto_door);
    live.insert_or_assign_edge(DSR::Edge::create<proto_edge_type>(400, 400));
    live.insert_or_assign_edge(DSR::Edge::create<current_edge_type>(200, 300));
    return {T_room_door, T_proto_door};
}

/// Memory: twins r0_room_0 / r1_room_1 under root and a passage `pname` matching one face in each, the room_0 face at
/// `T0` (in r0_room_0), the room_1 face at `T1` (in r1_room_1).
void remember_hole(DSR::DSRGraph &mem, const std::string &pname, const std::string &face0, const Mat::RTMat &T0,
                   const std::string &face1, const Mat::RTMat &T1)
{
    const auto root = mem.get_node("root")->id();
    auto room = [&](const std::string &name, std::uint64_t rid) {
        if (const auto r = mem.get_node(name); r.has_value()) return r->id();
        DSR::Node t; t.type("room"); t.name(name);
        mem.add_or_modify_attrib_local<room_id_att>(t, rid);
        mem.add_or_modify_attrib_local<parent_att>(t, root);
        const auto id = mem.insert_node(t).value();
        put_rt(mem, root, id, 0, 0, 0);
        return id; };
    const auto r0 = room("r0_room_0", 0), r1 = room("r1_room_1", 1);
    const auto f0 = add_door(mem, face0, r0, T0);
    const auto f1 = add_door(mem, face1, r1, T1);
    DSR::Node p; p.type("metaconcept"); p.name(pname);
    mem.add_or_modify_attrib_local<object_subtype_att>(p, std::string("passage"));
    const auto pid = mem.insert_node(p).value();
    mem.insert_or_assign_edge(DSR::Edge::create<match_edge_type>(pid, f0));
    mem.insert_or_assign_edge(DSR::Edge::create<match_edge_type>(pid, f1));
}

std::vector<std::string> memory_passages(DSR::DSRGraph &mem)
{
    std::vector<std::string> out;
    for (const auto &n : mem.get_nodes_by_type("metaconcept"))
        if (mem.get_attrib_by_name<object_subtype_att>(n).value_or(std::string{}) == "passage") out.push_back(n.name());
    std::ranges::sort(out);
    return out;
}

/// Run the live passage step and the promotion eviction it feeds, exactly as SpecificWorker does.
struct Crossing { std::string live_passage; ltsm::Outcome out; };
Crossing cross(const std::shared_ptr<DSR::DSRGraph> &live, ltsm::RoomEviction &ev)
{
    ltsm::PassageLive pl(live);
    pl.set_memory(&ev);
    pl.step();
    Crossing c;
    const auto owned = pl.owned();
    check(owned.size() == 1, "exactly one live passage after the step (" + std::to_string(owned.size()) + ")");
    if (owned.empty()) return c;
    c.live_passage = owned.front().name();
    ltsm::PromotionReading r; r.proto_id = 400; r.proto_name = "room_1"; r.valid = true; r.p_inside = 1.0;
    auto cand = ev.promotion_candidate(r);
    check(cand.has_value(), "a promotion candidate room_0 -> room_1");
    if (not cand.has_value()) return c;
    if (const auto pair = pl.pair_for(400); pair.has_value()) std::tie(cand->old_door_id, cand->new_door_id) = pair.value();
    c.out = ev.evict(cand.value());
    check(c.out.ok, "evict ok (" + c.out.reason + ")");
    return c;
}
}   // namespace

int main(int argc, char **argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    QCoreApplication app(argc, argv);

    std::printf("[numbering] live and memory passages share ONE numbering\n");
    std::string archived_name;
    std::shared_ptr<DSR::DSRGraph> mem_a;
    {
        auto live = graph("pid_live_a", "etc/stage_two_rooms.json");
        mem_a = graph("pid_mem_a", "ltsm_root.json");
        stage_crossing(*live, "door_1");
        {   // an unrelated hole memory already holds, numbered 3
            DSR::Node p; p.type("metaconcept"); p.name("passage_3");
            mem_a->add_or_modify_attrib_local<object_subtype_att>(p, std::string("passage"));
            mem_a->insert_node(p);
        }
        ltsm::RoomEviction ev(live, mem_a);
        check(ltsm::next_passage_index(*live, mem_a.get()) == 4, "next_passage_index counts memory's passage_3 => 4");
        const auto c = cross(live, ev);
        check(c.live_passage == "passage_4", "the live passage is numbered after memory's: " + c.live_passage);
        check(c.out.mem_doorway == c.live_passage, "the eviction archives the hole under the LIVE name: " + c.out.mem_doorway);
        check(memory_passages(*mem_a) == std::vector<std::string>{"passage_3", "passage_4"}, "memory: passage_3, passage_4 and nothing else");
        check(ev.next_passage_index() == 5, "after the archive the next number is 5");
        const auto fresh = ev.mint_passage();
        check(fresh != 0 and mem_a->get_node(fresh)->name() == "passage_5", "mint_passage() takes the shared numbering: passage_5");
        mem_a->delete_node(fresh);
        archived_name = c.out.mem_doorway;
    }

    std::printf("[recross] the archived hole re-crossed with its door re-made as door_17 keeps its passage name\n");
    {
        auto live = graph("pid_live_b", "etc/stage_two_rooms.json");   // room_0 re-entered, published in its remembered frame
        stage_crossing(*live, "door_17");
        ltsm::RoomEviction ev(live, mem_a);
        const auto before = memory_passages(*mem_a);
        const auto c = cross(live, ev);
        check(c.live_passage == archived_name, "the live passage takes memory's name " + archived_name + ": " + c.live_passage);
        check(c.out.mem_doorway == archived_name, "the eviction re-uses it: " + c.out.mem_doorway);
        check(memory_passages(*mem_a) == before, "no passage minted for the same hole");
        // Names are opaque (02bd9d4f): which label the face keeps is not the contract. The INVARIANT is: the hole has exactly
        // one face in room_0, and it is in the same passage.
        int faces_in_r0 = 0;
        if (const auto pw = mem_a->get_node(archived_name); pw.has_value())
            for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(pw.value(), "match"))
                if (ltsm::is_descendant(*mem_a, e.to(), "r0_room_0")) ++faces_in_r0;
        int r0_doors = 0;
        for (const auto &n : mem_a->get_nodes_by_type("object"))
            if (ltsm::is_door(*mem_a, n) and ltsm::is_descendant(*mem_a, n.id(), "r0_room_0")) ++r0_doors;
        check(faces_in_r0 == 1 and r0_doors == 1,
              "room_0 holds exactly one face of the hole, in the same passage " + archived_name + " (faces in passage "
              + std::to_string(faces_in_r0) + ", doors in r0_room_0 " + std::to_string(r0_doors) + ")");
    }

    std::printf("[renamed] memory knows the hole as door_3, the live door is door_17 at the same place\n");
    {
        auto live = graph("pid_live_c", "etc/stage_two_rooms.json");
        auto mem = graph("pid_mem_c", "ltsm_root.json");
        const auto [T, T1] = stage_crossing(*live, "door_17");
        remember_hole(*mem, "passage_7", "r0_door_3", T, "r1_door_9", T1);
        ltsm::RoomEviction ev(live, mem);
        const auto door = live->get_node("door_17").value();
        const auto hole = ev.remembered_hole(door);
        check(hole.has_value() and hole->passage.name() == "passage_7" and hole->face.name() == "r0_door_3",
              "remembered_hole(door_17) = passage_7 by its face r0_door_3 (by place)");
        const auto c = cross(live, ev);
        check(c.live_passage == "passage_7", "the live passage is passage_7: " + c.live_passage);
        check(c.out.mem_doorway == "passage_7", "the eviction re-uses passage_7: " + c.out.mem_doorway);
        check(memory_passages(*mem) == std::vector<std::string>{"passage_7"}, "memory holds ONE passage");
    }

    std::printf("[two_holes] a passage of ANOTHER hole of room_0 is never handed to this one\n");
    {
        auto live = graph("pid_live_d", "etc/stage_two_rooms.json");
        auto mem = graph("pid_mem_d", "ltsm_root.json");
        const auto [T, T1] = stage_crossing(*live, "door_1");
        Mat::RTMat other = T, other1 = T1;
        other.translation().y() -= 2.0;                     // 2 m along the wall: a different hole, same room
        other1.translation().x() -= 2.0;                    // and its room_1 face is not at the entry mirror either
        remember_hole(*mem, "passage_2", "r0_door_1", other, "r1_door_9", other1);   // same NAME as the live door, other place
        ltsm::RoomEviction ev(live, mem);
        check(not ev.remembered_hole(live->get_node("door_1").value()).has_value(),
              "remembered_hole(door_1) = none: the name r0_door_1 matches, the place does not");
        const auto c = cross(live, ev);
        check(c.live_passage == "passage_3", "the live passage is fresh: " + c.live_passage);
        check(c.out.mem_doorway == "passage_3", "memory archives it as passage_3: " + c.out.mem_doorway);
        check(memory_passages(*mem) == std::vector<std::string>{"passage_2", "passage_3"}, "two holes, two passages");
        bool disjoint = true;
        const auto p2 = mem->get_node("passage_2"), p3 = mem->get_node("passage_3");
        if (p2 and p3)
            for (const auto &a : DSR::DSRGraph::get_node_edges_by_type(p2.value(), "match"))
                for (const auto &b : DSR::DSRGraph::get_node_edges_by_type(p3.value(), "match"))
                    disjoint = disjoint and a.to() != b.to();
        check(p2 and p3 and disjoint, "the two passages share no face");
        const auto kept = mem->get_inner_eigen_api()->get_transformation_matrix("r0_room_0", "r0_door_1");
        check(kept.has_value() and (kept->translation() - other.translation()).head<2>().norm() < 1e-3,
              "the other hole's face r0_door_1 is NOT overwritten by the namesake live door (still at its own place)");
        check(c.out.mem_door_old == "r0_door_1_h1", "this hole's face is archived under its own name: " + c.out.mem_door_old);
    }

    std::printf("[same_room] a second door of the provisional room, once it is current, is never paired with its own room's doors\n");
    {   // recog_proto8: room_3 provisional AND current; DOOR_2 (door_50, 3 m from the entry) got 'passage_1 match door_49 ↔ door_50'
        auto live = graph("pid_live_e", "etc/stage_two_rooms.json");
        const auto [T, T1] = stage_crossing(*live, "door_1");
        ltsm::PassageLive pl(live);
        pl.step();
        check(pl.owned().size() == 1, "the crossing makes one passage");
        live->delete_edge(200, 300, "current");
        live->insert_or_assign_edge(DSR::Edge::create<current_edge_type>(200, 400));   // room_1 (still proto) is current
        Mat::RTMat far = T1;
        far.translation().x() += 3.0;
        const auto d6 = add_door(*live, "door_6", 400, far);                           // another hole of room_1
        pl.step();
        check(pl.owned().size() == 1, "still one passage (" + std::to_string(pl.owned().size()) + ")");
        check(not pl.passage_of(d6).has_value(), "door_6 has no passage: it stays a NEW door for `cross new`");
    }

    std::printf("[live_door] the summary's live door of a passage whose face was re-made under another name\n");
    {   // explore_tour13: memory knew the hole as r1_door_2, door_concept re-made it as door_4 ('passage_0 match door_3 <->
        // door_4') — the summary looked for a live 'door_2', found none, and never named door_4: the controller could not
        // resolve the entry face, crossed the same doorway again, and its final Return asked for a name that was gone.
        auto live = graph("pid_live_f", "etc/stage_two_rooms.json");
        auto mem = graph("pid_mem_f", "ltsm_root.json");
        const auto [T, T1] = stage_crossing(*live, "door_17");   // room_0 current; its door is door_17
        remember_hole(*mem, "passage_7", "r0_door_3", T, "r1_door_9", T1);
        if (auto p = mem->get_node("passage_7"); p.has_value())
        { mem->add_or_modify_attrib_local<passage_crossings_att>(p.value(), 1); mem->update_node(p.value()); }
        const auto before = ltsm::build_topology_json(*mem, *live);
        check(before.find("\"live_door\"") == std::string::npos, "no live passage yet: no live door (never a guess)");
        DSR::Node lp; lp.type("metaconcept"); lp.name("passage_7");
        live->add_or_modify_attrib_local<object_subtype_att>(lp, std::string("passage"));
        const auto lpid = live->insert_node(lp).value();
        live->insert_or_assign_edge(DSR::Edge::create<match_edge_type>(lpid, live->get_node("door_17")->id()));
        live->insert_or_assign_edge(DSR::Edge::create<match_edge_type>(lpid, live->get_node("door_5")->id()));
        const auto after = ltsm::build_topology_json(*mem, *live);
        check(after.find("\"live_door\":\"door_17\"") != std::string::npos,
              "the live passage's door in the CURRENT room names it: door_17 (not the proto's door_5): " + after);
        check(rc::topo::live_door_for(after, "door_9") == std::optional<std::string>("door_17"),
              "live_door_for(the far face's name) = door_17 — the entry face after a return");
        check(rc::topo::crossed_doors(after).contains("door_17"), "door_17 counts as crossed (no re-crossing)");
    }

    std::printf("%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    std::fflush(stdout);
    std::_Exit(failures ? 1 : 0);
}
