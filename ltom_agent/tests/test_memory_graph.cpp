// Offline test of src/memory_graph.h: the repertory projected onto a standalone DSR graph (private DDS domain 63). No fleet, no viewer.
#include "../src/memory_graph.h"
#include "../../language_agent/src/ltom_query_dsr.h"   // the reader on the other side of this graph

#include <cmath>
#include <memory>
#include <print>

static int failures = 0;
#define CHECK(c) do { if (not (c)) { std::println(stderr, "FAIL {}:{}: {}", __FILE__, __LINE__, #c); ++failures; } } while (0)

namespace
{
using Entries = std::map<std::uint64_t, ltom::Entry>;
constexpr std::uint64_t uid_of(int n) { return (std::uint64_t{20} << 40) | static_cast<std::uint64_t>(n); }

ltom::Entry chair(int n, std::int64_t room, double x, double y)
{
    ltom::Entry e;
    e.uid = uid_of(n); e.cls = "chair"; e.name = "chair_" + std::to_string(n);
    e.w = 0.6f; e.d = 0.5f; e.h = 1.2f; e.n_sightings = 128;
    e.lives.push_back({.born_ms = 1000, .first_seen_ms = 1100, .last_seen_ms = 5000, .end_ms = 5200, .end = ltom::End::death});
    e.lives.push_back({.born_ms = 9000, .first_seen_ms = 9100, .last_seen_ms = 9900, .end_ms = -1, .end = ltom::End::alive});
    e.keyframes.push_back({.t_ms = 1100, .room_key = room, .pose = {x, y, 0.5, 0.09, 0.08, 3.4}});
    return e;
}
void add(Entries& es, ltom::Entry e) { es.emplace(e.uid, std::move(e)); }

struct Fixture
{
    std::shared_ptr<DSR::DSRGraph> G;
    ltom::MemoryGraph mem;
    Fixture() : G(std::make_shared<DSR::DSRGraph>("ltom_mem_test", 63, std::string(LTOM_ROOT_JSON), true, static_cast<int8_t>(63), DSR::Queue)), mem(G) {}
    std::uint64_t id(const std::string& n) { const auto x = G->get_node(n); return x ? x->id() : 0; }
    std::optional<std::vector<float>> translation(const std::string& room, const std::string& obj)
    {
        const auto e = G->get_edge(id(room), id(obj), "RT");
        if (not e) return std::nullopt;
        return G->get_rt_api()->get_translation(*G->get_node(room), id(obj)).transform([](const auto& v) { return std::vector<float>{static_cast<float>(v.x()), static_cast<float>(v.y())}; });
    }
    int count_objects() { return static_cast<int>(G->get_nodes_by_type("object").size()); }
};

void test_objects_hang_under_their_remembered_room()
{
    Fixture f;
    Entries es;
    add(es, chair(1, 1, -1.5, -2.5));
    add(es, chair(2, 1, 2.0, 0.5));
    add(es, chair(3, 2, 0.25, 1.0));
    f.mem.sync(es, {});
    CHECK(f.id("room_1") != 0 and f.id("room_2") != 0);
    CHECK(f.G->get_node("room_1")->type() == "room");
    CHECK(f.G->get_edge(f.id("root"), f.id("room_1"), "RT").has_value());
    CHECK(f.G->get_node("chair_1")->type() == "object");
    CHECK(f.G->get_attrib_by_name<parent_att>(*f.G->get_node("chair_1")) == f.id("room_1"));
    CHECK(f.G->get_attrib_by_name<parent_att>(*f.G->get_node("chair_3")) == f.id("room_2"));
    CHECK(f.G->get_edge(f.id("room_2"), f.id("chair_3"), "RT").has_value());
    const auto t = f.translation("room_1", "chair_1");
    CHECK(t.has_value() and std::abs((*t)[0] + 1.5f) < 1e-3f and std::abs((*t)[1] + 2.5f) < 1e-3f);
    const auto e = f.G->get_edge(f.id("room_1"), f.id("chair_1"), "RT");
    CHECK(e.has_value() and f.G->get_attrib_by_name<rt_covariance_att>(*e).has_value());
}

void test_a_click_on_an_object_shows_what_is_retained()
{
    Fixture f;
    Entries es;
    add(es, chair(1, 1, 1.0, 1.0));
    f.mem.sync(es, {});
    const auto n = *f.G->get_node("chair_1");
    CHECK(f.G->get_attrib_by_name<object_uid_att>(n) == uid_of(1));
    CHECK(f.G->get_attrib_by_name<object_subtype_att>(n) == std::string("chair"));
    CHECK(f.G->get_attrib_by_name<width_m_att>(n) == 0.6f);
    CHECK(f.G->get_attrib_by_name<depth_m_att>(n) == 0.5f);
    CHECK(f.G->get_attrib_by_name<height_m_att>(n) == 1.2f);
    CHECK(f.G->get_attrib_by_name<level_att>(n) == 2);
    CHECK(f.G->get_attrib_by_name<pos_x_att>(n).has_value() and f.G->get_attrib_by_name<pos_y_att>(n).has_value());
    const auto history = f.G->get_attrib_by_name<em_record_att>(n);
    CHECK(history.has_value());
    if (history)
    {
        CHECK(history->contains("sightings: 128"));
        CHECK(history->contains("life 1"));
        CHECK(history->contains("death"));
        CHECK(history->contains("life 2"));
        CHECK(history->contains("alive"));
        CHECK(history->contains("keyframes: 1"));
    }
}

void test_live_or_only_remembered_follows_the_live_graph()
{
    Fixture f;
    Entries es;
    add(es, chair(1, 1, 1.0, 1.0));
    add(es, chair(2, 1, 2.0, 2.0));
    f.mem.sync(es, {uid_of(1)});
    CHECK(f.G->get_attrib_by_name<state_att>(*f.G->get_node("chair_1")) == std::string("live"));
    CHECK(f.G->get_attrib_by_name<state_att>(*f.G->get_node("chair_2")) == std::string("remembered"));
    f.mem.sync(es, {uid_of(2)});
    CHECK(f.G->get_attrib_by_name<state_att>(*f.G->get_node("chair_1")) == std::string("remembered"));
    CHECK(f.G->get_attrib_by_name<state_att>(*f.G->get_node("chair_2")) == std::string("live"));
}

// The contract with language_agent: what this graph holds is what its reader gets back.
void test_language_agent_reads_only_what_is_absent_with_room_and_time()
{
    Fixture f;
    Entries es;
    add(es, chair(1, 1, 1.0, 1.0));
    add(es, chair(2, 2, 2.0, 2.0));
    auto no_keyframe = chair(3, 1, 0.0, 0.0);
    no_keyframe.keyframes.clear();
    add(es, std::move(no_keyframe));
    f.mem.sync(es, {uid_of(1)});

    const auto mem = rc::chat::load_remembered(*f.G, 20000);
    CHECK(mem.now_ms == 20000);
    CHECK(mem.objects.size() == 2);
    const auto find = [&](const char* name) { return std::ranges::find(mem.objects, name, &rc::chat::RememberedObject::name); };
    CHECK(find("chair_1") == mem.objects.end());
    const auto two = find("chair_2");
    CHECK(two != mem.objects.end() and two->subtype == "chair" and two->room == "room_2" and two->last_seen_ms == 9900);
    CHECK(two != mem.objects.end() and std::abs(two->x - 2.0) < 1e-3 and std::abs(two->y - 2.0) < 1e-3);
    const auto three = find("chair_3");
    CHECK(three != mem.objects.end() and three->room.empty());
}

void test_sync_is_repeatable_and_follows_the_newest_keyframe()
{
    Fixture f;
    Entries es;
    add(es, chair(1, 1, 1.0, 1.0));
    f.mem.sync(es, {});
    const auto first = f.id("chair_1");
    f.mem.sync(es, {});
    CHECK(f.count_objects() == 1 and f.id("chair_1") == first);
    es.at(uid_of(1)).keyframes.push_back({.t_ms = 2000, .room_key = 1, .pose = {3.0, -1.0, 0.0, 0.05, 0.05, 1.0}});
    f.mem.sync(es, {});
    const auto t = f.translation("room_1", "chair_1");
    CHECK(f.id("chair_1") == first);
    CHECK(t.has_value() and std::abs((*t)[0] - 3.0f) < 1e-3f and std::abs((*t)[1] + 1.0f) < 1e-3f);
}

void test_an_object_last_seen_in_another_room_moves_there()
{
    Fixture f;
    Entries es;
    add(es, chair(1, 1, 1.0, 1.0));
    f.mem.sync(es, {});
    es.at(uid_of(1)).keyframes.push_back({.t_ms = 2000, .room_key = 2, .pose = {0.5, 0.5, 0.0, 0.05, 0.05, 1.0}});
    f.mem.sync(es, {});
    CHECK(f.count_objects() == 1);
    CHECK(f.G->get_attrib_by_name<parent_att>(*f.G->get_node("chair_1")) == f.id("room_2"));
    CHECK(f.G->get_edge(f.id("room_2"), f.id("chair_1"), "RT").has_value());
    CHECK(not f.G->get_edge(f.id("room_1"), f.id("chair_1"), "RT").has_value());
}

void test_every_retained_object_is_shown_even_with_no_known_room()
{
    Fixture f;
    Entries es;
    add(es, chair(1, -1, 1.0, 1.0));
    auto no_pose = chair(2, 1, 0.0, 0.0);
    no_pose.keyframes.clear();
    add(es, std::move(no_pose));
    f.mem.sync(es, {});
    CHECK(f.id("room_unknown") != 0);
    CHECK(f.G->get_attrib_by_name<parent_att>(*f.G->get_node("chair_1")) == f.id("room_unknown"));
    CHECK(f.G->get_attrib_by_name<parent_att>(*f.G->get_node("chair_2")) == f.id("room_unknown"));
}
}   // namespace

int main()
{
    test_objects_hang_under_their_remembered_room();
    test_a_click_on_an_object_shows_what_is_retained();
    test_live_or_only_remembered_follows_the_live_graph();
    test_language_agent_reads_only_what_is_absent_with_room_and_time();
    test_sync_is_repeatable_and_follows_the_newest_keyframe();
    test_an_object_last_seen_in_another_room_moves_there();
    test_every_retained_object_is_shown_even_with_no_known_room();
    std::println("{}", failures == 0 ? "ALL PASS" : "FAILED");
    return failures == 0 ? 0 : 1;
}
