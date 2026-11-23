// test_door_memory_rooms.cpp — Task 4 (one-model): per-wall rates and the room door-count prior learned from LTSM's remembered rooms.
//  - the pure part: rates_from_rooms / door_counts on hand-made rooms;
//  - the graph part: read_remembered_rooms on a standalone DSRGraph (a fake memory: room twin, floor twin, 4 wall twins, 2 door twins
//    under one wall, a door twin parented to the floor (geometric assignment), and the current room's twin, which must be excluded).
// Build/run: see tests/CMakeLists.txt (target test_door_memory_rooms). The pure part alone:
//   g++ -std=c++23 -O1 -DDOOR_MEMORY_ROOMS_PURE_ONLY -I/usr/include/eigen3 tests/test_door_memory_rooms.cpp -o /tmp/tdmr && /tmp/tdmr
#include "door_memory_rooms.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace
{
int failures = 0;
void check(bool ok, int line, const char* what) { if (not ok) { ++failures; std::printf("FAIL line %d: %s\n", line, what); } }
#define CHECK(x) check((x), __LINE__, #x)
bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }

using namespace rc::door;
using namespace rc::door::memrooms;

RememberedRoom room_of(std::uint64_t id, std::initializer_list<int> doors)
{
    RememberedRoom r;
    r.room_id = id;
    for (const int d : doors) r.walls.push_back({d});
    return r;
}
}

int main()
{
    {   // rates_from_empty_memory
        const core::WallRates seed;
        const auto r = rates_from_rooms({}, seed);
        CHECK(r.first.a == seed.first.a and r.first.b == seed.first.b);
        CHECK(r.another.a == seed.another.a and r.another.b == seed.another.b);
    }
    const std::vector<RememberedRoom> rooms = {room_of(1, {1, 0, 0, 1}), room_of(2, {2, 0, 0, 0})};
    {   // rates_from_rooms_counts
        const auto r = rates_from_rooms(rooms);
        CHECK(near(r.first.a, 2 + 3) and near(r.first.b, 3 + 5));
        CHECK(near(r.another.a, 0.05 + 1) and near(r.another.b, 4.95 + 2));
    }
    {   // room_without_walls_ignored
        std::vector<RememberedRoom> v = rooms;
        v.push_back(room_of(3, {}));
        const auto a = rates_from_rooms(rooms), b = rates_from_rooms(v);
        CHECK(a.first.a == b.first.a and a.first.b == b.first.b and a.another.a == b.another.a and a.another.b == b.another.b);
        CHECK(std::isfinite(b.first.mean()) and std::isfinite(b.another.mean()));
    }
    {   // door_counts_per_room
        const auto c = door_counts(rooms);
        CHECK(c.size() == 2 and c[0] == 2 and c[1] == 2);
    }
#ifndef DOOR_MEMORY_ROOMS_PURE_ONLY
    {   // the graph reader on a fake memory
        DSR::DSRGraph G("door_memory_rooms_test", 993, std::string(SHADOW_JSON), true, static_cast<int8_t>(87), DSR::Queue);
        const auto add = [&](DSR::Node n, std::uint64_t parent, std::uint64_t idx, float x, float y, float yaw, float width) -> std::uint64_t
        {
            G.add_or_modify_attrib_local<room_id_att>(n, idx);
            if (width > 0.f) G.add_or_modify_attrib_local<width_m_att>(n, width);
            if (parent != 0) G.add_or_modify_attrib_local<parent_att>(n, parent);
            const auto id = G.insert_node(n).value();
            if (parent != 0)
            {
                auto e = DSR::Edge::create<RT_edge_type>(parent, id);
                G.add_or_modify_attrib_local<rt_translation_att>(e, std::vector<float>{x, y, 0.f});
                G.add_or_modify_attrib_local<rt_rotation_euler_xyz_att>(e, std::vector<float>{0.f, 0.f, yaw});
                G.insert_or_assign_edge(e);
            }
            return id;
        };
        const auto make_room = [&](const std::string& p, std::uint64_t k) -> std::pair<std::uint64_t, std::vector<std::uint64_t>>
        {
            const auto room  = add(DSR::Node::create<room_node_type>(p + "room_x"), 0, k, 0, 0, 0, 0);
            const auto floor = add(DSR::Node::create<floor_node_type>(p + "floor"), room, k, 0, 0, 0, 0);
            const float pi2 = 1.5707963f;
            std::vector<std::uint64_t> w;
            w.push_back(add(DSR::Node::create<wall_node_type>(p + "wall_0"), floor, k, 0.f, 2.f, 0.f, 4.f));
            w.push_back(add(DSR::Node::create<wall_node_type>(p + "wall_1"), floor, k, 2.f, 0.f, pi2, 4.f));
            w.push_back(add(DSR::Node::create<wall_node_type>(p + "wall_2"), floor, k, 0.f, -2.f, 0.f, 4.f));
            w.push_back(add(DSR::Node::create<wall_node_type>(p + "wall_3"), floor, k, -2.f, 0.f, pi2, 4.f));
            w.push_back(floor);
            return {room, w};
        };
        const auto [room3, w3] = make_room("r3_", 3);
        add(DSR::Node::create<object_node_type>("r3_door_1"), w3[0], 3, -1.f, 0.f, 0.f, 0.9f);
        add(DSR::Node::create<object_node_type>("r3_door_2"), w3[0], 3, 1.f, 0.f, 0.f, 0.9f);
        const auto [room4, w4] = make_room("r4_", 4);   // the current room's twin: three doors, must not count
        add(DSR::Node::create<object_node_type>("r4_door_1"), w4[1], 4, 0.f, 0.f, 0.f, 0.9f);
        add(DSR::Node::create<object_node_type>("r4_door_2"), w4[2], 4, 0.f, 0.f, 0.f, 0.9f);
        add(DSR::Node::create<object_node_type>("r4_door_3"), w4[3], 4, 0.f, 0.f, 0.f, 0.9f);
        add(DSR::Node::create<room_node_type>("room_live"), 0, 4, 0, 0, 0, 0);   // a LIVE room: no r<k>_ prefix, never read

        const auto rr = read_remembered_rooms(G, 4);
        CHECK(rr.size() == 1 and rr[0].room_id == 3 and rr[0].walls.size() == 4);
        const auto r = rates_from_rooms(rr);
        CHECK(near(r.first.a, 2 + 1) and near(r.first.b, 3 + 3));
        CHECK(near(r.another.a, 0.05 + 1) and near(r.another.b, 4.95 + 0));
        CHECK(door_counts(rr).size() == 1 and door_counts(rr)[0] == 2);
        CHECK(read_remembered_rooms(G, std::nullopt).size() == 2);   // a fresh room excludes nothing

        // a door twin parented to the FLOOR goes to the nearest wall segment (wall_1 at x = 2)
        auto d = DSR::Node::create<object_node_type>("r3_door_3");
        G.add_or_modify_attrib_local<object_subtype_att>(d, std::string("door"));
        add(d, w3[4], 3, 2.02f, 0.3f, 0.f, 0.9f);
        const auto r2 = rates_from_rooms(read_remembered_rooms(G, 4));
        CHECK(near(r2.first.a, 2 + 2) and near(r2.first.b, 3 + 2));
        CHECK(near(r2.another.a, 0.05 + 1) and near(r2.another.b, 4.95 + 1));
    }
#endif
    std::printf(failures == 0 ? "test_door_memory_rooms: ALL PASSED\n" : "test_door_memory_rooms: %d FAILED\n", failures);
    return failures;
}
