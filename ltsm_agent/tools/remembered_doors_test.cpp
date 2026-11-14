// remembered_doors: LTSM posts the memory faces of the robot's CURRENT room as remembered_door notes.   run from the component root:   ./bin/remembered_doors_test
#include "../src/remembered_doors.h"
#include "../src/room_eviction.h"
#include <QCoreApplication>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <string>

static int failures = 0;
static void check(bool ok, const std::string &what) { if (not ok) ++failures; std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str()); }

static std::shared_ptr<DSR::DSRGraph> live, mem;
static bool with_geometry = true;   // a door face carries its own width/height/depth
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
    if (subtype and std::string(subtype) == "door" and with_geometry)
    {
        mem->add_or_modify_attrib_local<width_m_att>(n, 0.9f);
        mem->add_or_modify_attrib_local<height_m_att>(n, 2.0f);
        mem->add_or_modify_attrib_local<depth_m_att>(n, 0.1f);
    }
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
    // the REAL shape: the door hangs from its wall (wall-local pose), the wall from the room; door_7 via a floor as well
    const auto wall = mem_node("wall", "r3_wall_0", mem_twin, 3, 2, 0.6);
    mem_node("object", "r3_door_6", wall, 1.5, 0.2, 0.3, std::nullopt, "door", 0.8f);
    const auto fl = mem_node("floor", "r3_floor", mem_twin, 0, 0, 0);
    const auto wall2 = mem_node("wall", "r3_wall_1", fl, -1, 4, -0.4);
    mem_node("object", "r3_door_7", wall2, 0.5, 0.1, 1.2, std::nullopt, "door", 0.7f);
}

// I5 fixture: the robot is IN room_3 alone (its memory twin r3_room_3 exists); memory holds r3_door_1 with evidence 1.0 and the
// live room holds door_1 at the same place with evidence 5.0, seen at 1234 ms.
static std::uint64_t live_door1 = 0, mem_door1 = 0;
static void build_refresh(double live_x = -2.0)
{
    reset(); build();
    live->delete_node(live_room2);
    auto r = live->get_node(robot_id).value();
    live->add_or_modify_attrib_local<parent_att>(r, live->get_node("root").value().id());
    live->update_node(r);
    mem_door1 = mem_node("object", "r3_door_1", mem_twin, -2, -3, 0.2, std::nullopt, "door", 1.0f);
    DSR::Node d; d.type("object"); d.name("door_1");
    live->add_or_modify_attrib_local<parent_att>(d, live_room);
    live->add_or_modify_attrib_local<object_subtype_att>(d, std::string("door"));
    live->add_or_modify_attrib_local<width_m_att>(d, 0.9f);
    live->add_or_modify_attrib_local<height_m_att>(d, 2.0f);
    live->add_or_modify_attrib_local<depth_m_att>(d, 0.1f);
    live->add_or_modify_attrib_local<door_exist_evidence_att>(d, 5.0f);
    live->add_or_modify_attrib_local<door_seen_ms_att>(d, std::uint64_t{1234});
    live_door1 = live->insert_node(d).value();
    put_edge(*live, live_room, live_door1, live_x, -3, 0.2);
    // sync reads the robot's parent: hang it under the room (the real fleet's robot_node/parent_att)
    auto rb = live->get_node(robot_id).value();
    live->add_or_modify_attrib_local<parent_att>(rb, live_room);
    live->update_node(rb);
}
static float mem_evidence() { return mem->get_attrib_by_name<door_exist_evidence_att>(mem->get_node("r3_door_1").value()).value_or(-1.f); }

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
            check(e.has_value(), "RT edge room -> note");
        }
        // composed pose T(room <- face), by hand: yaw about z, x_room = R(yaw) p + t
        const auto pose_of = [&](const char *name, double tx, double ty, double th, double fx, double fy, double fth)
        {
            const double x = tx + std::cos(th) * fx - std::sin(th) * fy, y = ty + std::sin(th) * fx + std::cos(th) * fy;
            const auto note = live->get_node(name);
            const auto T = live->get_inner_eigen_api()->get_transformation_matrix("room_3", name);
            check(note.has_value() and T.has_value(), std::string(name) + ": pose resolvable");
            if (not T.has_value()) return;
            const double yaw = std::atan2(T->rotation()(1, 0), T->rotation()(0, 0));
            check(std::fabs(T->translation().x() - x) < 1e-5 and std::fabs(T->translation().y() - y) < 1e-5
                  and std::fabs(std::remainder(yaw - (th + fth), 2 * M_PI)) < 1e-5,
                  std::string(name) + ": note pose in the room == composed wall*face pose");
        };
        pose_of("remembered_door_5", 0, 0, 0, 2, 0, 0.5);
        pose_of("remembered_door_6", 3, 2, 0.6, 1.5, 0.2, 0.3);
        pose_of("remembered_door_7", -1, 4, -0.4, 0.5, 0.1, 1.2);

        const int u = rd.updates();
        check(rd.step() == 3 and rd.updates() == u, "idempotent: a second step rewrites no note (no update_node)");
        check(rd.remove_all() == 3 and notes() == 0, "remove_all sweeps them");
    }
    std::printf("negative_yaw_survives_the_hand_off\n");
    {
        // the writer must put a planar (0,0,yaw) on the note edge, and the reader (yaw = rz) must get the composed yaw back
        for (const double wall_yaw : {-1.0, -2.8})
        {
            reset(); build();
            const auto wall = mem_node("wall", "r3_wall_9", mem_twin, 0, 0, wall_yaw);
            mem_node("object", "r3_door_9", wall, 1.0, 0.0, 0.0, std::nullopt, "door", 0.9f);
            ltsm::RememberedDoors rd(live, mem);
            rd.step();
            const auto n = live->get_node("remembered_door_9");
            const auto e = n.has_value() ? live->get_edge(live_room, n->id(), "RT") : std::nullopt;
            check(e.has_value(), std::format("yaw {}: RT edge present", wall_yaw));
            if (not e.has_value()) continue;
            const auto r = live->get_attrib_by_name<rt_rotation_euler_xyz_att>(e.value());
            check(r.has_value() and r.value().get().size() >= 3, "euler attribute present");
            if (not r.has_value()) continue;
            const auto &v = r.value().get();
            check(std::fabs(v[0]) < 1e-6f and std::fabs(v[1]) < 1e-6f, std::format("yaw {}: roll/pitch are 0", wall_yaw));
            check(std::fabs(std::remainder(v[2] - wall_yaw, 2 * M_PI)) < 1e-5, std::format("yaw {}: decoded z = {:.6f}", wall_yaw, v[2]));
        }
    }
    std::printf("face_without_geometry_is_not_posted\n");
    {
        reset(); build();
        with_geometry = false;
        mem_node("object", "r3_door_5_entry", mem_twin, 1, 1, 0.0, std::nullopt, "door", 0.5f);
        with_geometry = true;
        ltsm::RememberedDoors rd(live, mem);
        check(rd.step() == 3 and not live->get_node("remembered_door_5_entry").has_value(), "geometry-less face: no note; the 3 real faces still posted");
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
    std::printf("sync_refreshes_known_door_attrs\n");
    {
        build_refresh();
        ltsm::RoomEviction ev(live, mem);
        const std::size_t doors_before = mem->get_nodes_by_type("object").size();
        check(ev.sync_found_doors() > 0, "sync returns > 0");
        const auto m = mem->get_node("r3_door_1");
        check(m.has_value() and m->id() == mem_door1, "memory node is still r3_door_1");
        check(m.has_value() and std::fabs(mem_evidence() - 5.0f) < 1e-6f, "evidence 1.0 -> 5.0");
        check(m.has_value() and mem->get_attrib_by_name<door_seen_ms_att>(m.value()).value_or(0) == 1234, "door_seen_ms 1234 copied");
        check(mem->get_nodes_by_type("object").size() == doors_before, "no second door created");
        std::printf("sync_refresh_is_write_on_change\n");
        check(ev.sync_found_doors() == 0, "second sync with nothing changed returns 0");
    }
    std::printf("sync_refreshes_pose\n");
    {
        build_refresh();
        ltsm::RoomEviction ev(live, mem);
        ev.sync_found_doors();
        put_edge(*live, live_room, live_door1, -2.05, -3, 0.2);   // 0.05 m along the x axis
        check(ev.sync_found_doors() > 0, "moved live door: sync returns > 0");
        const auto e = mem->get_edge(mem_twin, mem_door1, "RT");
        const auto t = e.has_value() ? mem->get_attrib_by_name<rt_translation_att>(e.value()) : std::nullopt;
        check(t.has_value() and std::fabs(t.value().get()[0] - (-2.05f)) < 1e-5f, "memory edge x follows (-2.05)");
        check(ev.sync_found_doors() == 0, "and is stable afterwards");
    }
    std::printf("sync_no_refresh_when_held\n");
    {
        build_refresh();
        ltsm::RoomEviction ev(live, mem);
        ev.hold("r3_room_3");
        ev.sync_found_doors();
        check(std::fabs(mem_evidence() - 1.0f) < 1e-6f, "held room: memory evidence unchanged");
    }
    std::printf("\n%s (%d failure(s))\n", failures == 0 ? "ALL OK" : "FAILED", failures);
    std::fflush(stdout);
    std::_Exit(failures == 0 ? 0 : 1);   // like the other graph tests: skip the DDS teardown
}
