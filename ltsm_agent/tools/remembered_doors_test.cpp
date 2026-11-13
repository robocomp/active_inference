// remembered_doors: LTSM posts the memory faces of the robot's CURRENT room as remembered_door notes.   run from the component root:   ./bin/remembered_doors_test
#include "../src/remembered_doors.h"
#include <QCoreApplication>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

static int failures = 0;
static void check(bool ok, const std::string &what) { if (!ok) ++failures; std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str()); }

static std::shared_ptr<DSR::DSRGraph> live, mem;
static std::uint64_t robot_id = 0, live_room = 0, live_room2 = 0, mem_twin = 0;

static void put_edge(DSR::DSRGraph &G, std::uint64_t from, std::uint64_t to, double x, double y, double th)
{
    auto n = G.get_node(from).value();
    G.get_rt_api()->insert_or_assign_edge_RT(n, to, std::vector<float>{(float)x, (float)y, 0.f}, std::vector<float>{0.f, 0.f, (float)th});
}

static void reset()
{
    for (auto *g : {live.get(), mem.get()})
        for (const char *t : {"room", "floor", "wall", "object", "metaconcept", "robot"})
            for (const auto &n : g->get_nodes_by_type(t)) g->delete_node(n.id());
}

static std::uint64_t mem_node(const char *type, const std::string &name, std::uint64_t parent, double x, double y, double th,
                              std::optional<std::uint64_t> rid = std::nullopt, const char *subtype = nullptr, float evidence = -1.f)
{
    DSR::Node n; n.type(type); n.name(name);
    if (rid) mem->add_or_modify_attrib_local<room_id_att>(n, *rid);
    if (subtype) mem->add_or_modify_attrib_local<object_subtype_att>(n, std::string(subtype));
    if (evidence >= 0.f) mem->add_or_modify_attrib_local<door_exist_evidence_att>(n, evidence);
    if (parent != 0) mem->add_or_modify_attrib_local<parent_att>(n, parent);
    const auto id = mem->insert_node(n).value();
    put_edge(*mem, parent != 0 ? parent : mem->get_node("root").value().id(), id, x, y, th);
    return id;
}

static std::uint64_t live_node(const char *type, const std::string &name, std::uint64_t parent, std::optional<std::uint64_t> rid)
{
    DSR::Node n; n.type(type); n.name(name);
    if (rid) live->add_or_modify_attrib_local<room_id_att>(n, *rid);
    live->add_or_modify_attrib_local<parent_att>(n, parent);
    const auto id = live->insert_node(n).value();
    put_edge(*live, parent, id, 0, 0, 0);
    return id;
}

static void set_current(std::uint64_t room)
{
    for (const auto &e : live->get_edges_by_type("current")) live->delete_edge(e.from(), e.to(), "current");
    live->insert_or_assign_edge(DSR::Edge::create<current_edge_type>(robot_id, room));
}

static void build(bool staged = false)
{
    DSR::Node r; r.type("robot"); r.name("Shadow");
    robot_id = live->insert_node(r).value();
    put_edge(*live, live->get_node("root").value().id(), robot_id, 0, 0, 0);
    live_room = live_node("room", "room_3", robot_id, 3);
    live_room2 = live_node("room", "room_5", robot_id, 5);
    set_current(live_room);
    mem_twin = mem_node("room", "r3_room_3", 0, 0, 0, 0, 3, staged ? "staged" : nullptr);
    mem_node("object", "r3_door_5", mem_twin, 2, 0, 0.5, std::nullopt, "door", 0.9f);
    mem_node("object", "r3_door_6", mem_twin, 4, 1, 1.0, std::nullopt, "door", 0.8f);
    mem_node("object", "r3_door_7", mem_twin, 0, 3, 1.5, std::nullopt, "door", 0.7f);
}

static int notes()
{
    int n = 0;
    for (const auto &x : live->get_nodes_by_type("metaconcept"))
        if (live->get_attrib_by_name<object_subtype_att>(x).value_or("") == "remembered_door") ++n;
    return n;
}

int main(int argc, char **argv)
{
    std::setlocale(LC_ALL, "");
    setvbuf(stdout, nullptr, _IONBF, 0);
    QCoreApplication app(argc, argv);
    live = std::make_shared<DSR::DSRGraph>("rd_live", 198, "ltsm_root.json", true, 48);
    mem  = std::make_shared<DSR::DSRGraph>("rd_mem", 199, "ltsm_root.json", true, 49);

    std::printf("posts_one_note_per_face_of_current_room\n");
    {
        reset(); build();
        ltsm::RememberedDoors rd(live, mem);
        check(rd.step() == 3 and notes() == 3, "3 faces -> 3 notes");
        const auto n = live->get_node("remembered_door_5");
        check(n.has_value() and n->type() == "metaconcept", "remembered_door_5 is a metaconcept");
        if (n.has_value())
        {
            check(std::fabs(live->get_attrib_by_name<door_exist_evidence_att>(n.value()).value_or(-1.f) - 0.9f) < 1e-6f, "door_exist_evidence copied");
            check(live->get_attrib_by_name<parent_att>(n.value()).value_or(0) == live_room, "parent = live room");
            check(live->get_attrib_by_name<room_id_att>(n.value()).value_or(0) == 3, "room_id = 3");
            const auto e = live->get_edge(live_room, n->id(), "RT");
            const auto m = mem->get_edge(mem_twin, mem->get_node("r3_door_5").value().id(), "RT");
            check(e.has_value() and m.has_value() and e->attrs() == m->attrs(), "RT == the face's RT");
        }
        check(rd.step() == 3, "idempotent");
        check(rd.remove_all() == 3 and notes() == 0, "remove_all sweeps them");
    }
    std::printf("staged_twin_posts_nothing\n");
    {
        reset(); build(true);
        ltsm::RememberedDoors rd(live, mem);
        check(rd.step() == 0 and notes() == 0, "staged twin: no notes");
    }
    std::printf("adopted_note_deleted_not_reposted\n");
    {
        reset(); build();
        ltsm::RememberedDoors rd(live, mem);
        rd.step();
        auto n = live->get_node("remembered_door_5").value();
        live->add_or_modify_attrib_local<door_adopted_att>(n, true);
        live->update_node(n);
        check(rd.step() == 2, "adopted note deleted");
        check(rd.step() == 2 and not live->get_node("remembered_door_5").has_value(), "and not reposted");
    }
    std::printf("room_change_retires_notes\n");
    {
        reset(); build();
        ltsm::RememberedDoors rd(live, mem);
        rd.step();
        check(notes() == 3, "3 notes in room_3");
        set_current(live_room2);   // room_5 has no memory twin
        check(rd.step() == 0 and notes() == 0, "robot elsewhere: notes retired");
        set_current(live_room);
        check(rd.step() == 3, "back in room_3: reposted");
    }
    std::printf("notes_are_metaconcept_never_object\n");
    {
        reset(); build();
        ltsm::RememberedDoors rd(live, mem);
        rd.step();
        int objs = 0;
        for (const auto &o : live->get_nodes_by_type("object")) if (o.name().starts_with("remembered_")) ++objs;
        check(objs == 0 and notes() == 3, "no remembered_ object nodes");
    }
    std::printf("\n%s (%d failure(s))\n", failures == 0 ? "ALL OK" : "FAILED", failures);
    std::fflush(stdout);
    std::_Exit(failures == 0 ? 0 : 1);   // like the other graph tests: skip the DDS teardown
}
