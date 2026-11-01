// Offline test of src/recall_publisher.h: a standalone DSR graph on the private DDS domain 63, seeded from
// ltom_root.json. No fleet, no viewer.
#include "../src/recall_publisher.h"
#include "../../common/owned_nodes/owned_nodes.h"

#include <cmath>
#include <memory>
#include <print>

static int failures = 0;
#define CHECK(c) do { if (not (c)) { std::println(stderr, "FAIL {}:{}: {}", __FILE__, __LINE__, #c); ++failures; } } while (0)

namespace
{
using Id = std::uint64_t;
constexpr std::uint64_t kUid3 = (std::uint64_t{20} << 40) | 3;

Id add_node(DSR::DSRGraph& G, DSR::Node n, Id parent)
{
    G.add_or_modify_attrib_local<parent_att>(n, parent);
    G.add_or_modify_attrib_local<level_att>(n, 1);
    const auto id = G.insert_node(n);
    if (not id) throw std::runtime_error("insert_node failed: " + n.name());
    return *id;
}
void add_rt(DSR::DSRGraph& G, Id from, Id to)
{
    auto e = DSR::Edge::create<RT_edge_type>(from, to);
    G.add_or_modify_attrib_local<rt_translation_att>(e, std::vector<float>{0.f, 0.f, 0.f});
    G.add_or_modify_attrib_local<rt_rotation_euler_xyz_att>(e, std::vector<float>{0.f, 0.f, 0.f});
    G.insert_or_assign_edge(e);
}
ltom::Recall recall(std::uint64_t uid, double x, double y, double yaw, double v = 0.02)
{
    ltom::Recall r; r.uid = uid; r.cls = "chair";
    r.kf.pose = {x, y, yaw, v, v * 2, v * 3};
    return r;
}

struct Fixture
{
    std::shared_ptr<DSR::DSRGraph> G;
    Id root = 0, room1 = 0, room2 = 0;
    std::unique_ptr<ltom::RecallPublisher> pub;
    Fixture()
    {
        G = std::make_shared<DSR::DSRGraph>("ltom_test", 63, std::string(LTOM_ROOT_JSON), true, static_cast<int8_t>(63), DSR::Queue);
        root = G->get_node("root")->id();
        room1 = add_node(*G, DSR::Node::create<room_node_type>("room_1"), root);
        add_rt(*G, root, room1);
        room2 = add_node(*G, DSR::Node::create<room_node_type>("room_2"), root);
        add_rt(*G, root, room2);
        pub = std::make_unique<ltom::RecallPublisher>(G);
    }
    std::optional<DSR::Node> node(const std::string& n) { return G->get_node(n); }
    int recall_count()
    {
        int c = 0;
        for (const auto& n : G->get_nodes_by_type("metaconcept")) c += n.name().starts_with("ltom_recall_");
        return c;
    }
};
std::string rname(std::uint64_t uid) { return "ltom_recall_" + std::to_string(uid); }
}   // namespace

static void test_inserts_under_room(Fixture& f)
{
    const std::vector<ltom::Recall> want{recall(kUid3, 1.5, -2.0, 0.5)};
    f.pub->sync(f.room1, want);
    const auto n = f.node(rname(kUid3));
    CHECK(n.has_value());
    if (not n) return;
    CHECK(n->type() == "metaconcept");
    CHECK(f.G->get_attrib_by_name<object_uid_att>(*n).value_or(0) == kUid3);
    CHECK(f.G->get_attrib_by_name<object_subtype_att>(*n).value_or("") == "chair");
    CHECK(not f.G->get_attrib_by_name<width_m_att>(*n).has_value());
    CHECK(f.G->get_attrib_by_name<parent_att>(*n).value_or(0) == f.room1);
    const auto e = f.G->get_edge(f.room1, n->id(), "RT");
    CHECK(e.has_value());
    if (not e) return;
    const auto T = f.G->get_rt_api()->get_edge_RT_as_rtmat(*e);
    const auto cov = f.G->get_attrib_by_name<rt_covariance_att>(*e);
    CHECK(T and cov);
    if (not (T and cov)) return;
    CHECK(std::abs(T->translation().x() - 1.5) < 1e-5 and std::abs(T->translation().y() + 2.0) < 1e-5);
    CHECK(std::abs(std::atan2(T->linear()(1, 0), T->linear()(0, 0)) - 0.5) < 1e-5);
    const auto& c = cov.value().get();
    CHECK(c.size() == 36 and std::abs(c[0] - 0.02f) < 1e-6f and std::abs(c[7] - 0.04f) < 1e-6f and std::abs(c[35] - 0.06f) < 1e-6f);
}

static void test_moves_when_keyframe_changes(Fixture& f)
{
    f.pub->sync(f.room1, std::vector{recall(kUid3, 1.0, 1.0, 0.0)});
    const auto id1 = f.node(rname(kUid3))->id();
    f.pub->sync(f.room1, std::vector{recall(kUid3, 3.0, 4.0, 1.0, 0.05)});
    const auto n = f.node(rname(kUid3));
    CHECK(n.has_value() and n->id() == id1);                       // moved, not re-inserted
    CHECK(f.recall_count() == 1);
    const auto e = f.G->get_edge(f.room1, id1, "RT");
    CHECK(e.has_value());
    if (not e) return;
    const auto T = f.G->get_rt_api()->get_edge_RT_as_rtmat(*e);   // the newest ring block
    CHECK(T.has_value());
    if (T)
    {
        CHECK(std::abs(T->translation().x() - 3.0) < 1e-5 and std::abs(T->translation().y() - 4.0) < 1e-5);
        CHECK(std::abs(std::atan2(T->linear()(1, 0), T->linear()(0, 0)) - 1.0) < 1e-5);
    }
    CHECK(std::abs(f.G->get_attrib_by_name<rt_covariance_att>(*e).value().get()[0] - 0.05f) < 1e-6f);
    // A different room: deleted and re-created under room_2.
    f.pub->sync(f.room2, std::vector{recall(kUid3, 3.0, 4.0, 1.0, 0.05)});
    CHECK(f.recall_count() == 1);
    const auto m = f.node(rname(kUid3));
    CHECK(m.has_value() and f.G->get_edge(f.room2, m->id(), "RT").has_value());
    CHECK(m.has_value() and not f.G->get_edge(f.room1, m->id(), "RT").has_value());
}

static void test_removed_when_alive(Fixture& f)
{
    f.pub->sync(f.room1, std::vector{recall(kUid3, 1.0, 1.0, 0.0)});
    CHECK(f.recall_count() == 1);
    f.pub->sync(f.room1, std::vector<ltom::Recall>{});
    CHECK(f.recall_count() == 0);
}

static void test_no_room_no_recall(Fixture& f)
{
    f.pub->sync(f.room1, std::vector{recall(kUid3, 1.0, 1.0, 0.0)});
    CHECK(f.recall_count() == 1);
    f.pub->sync(std::nullopt, std::vector{recall(kUid3, 1.0, 1.0, 0.0)});   // even a non-empty want is ignored
    CHECK(f.recall_count() == 0);
}

static void test_stale_sweep(Fixture& f)
{
    add_node(*f.G, DSR::Node::create<metaconcept_node_type>("ltom_recall_9"), f.room1);
    add_node(*f.G, DSR::Node::create<metaconcept_node_type>("recall_room_1"), f.room1);
    CHECK(f.recall_count() == 1);
    const rc::owned::Spec spec{.agent = "ltom_agent", .name_prefix = "ltom_recall_", .node_types = {"metaconcept"}};
    CHECK(rc::owned::remove_instance_nodes(*f.G, spec) == 1);
    CHECK(f.recall_count() == 0);
    CHECK(f.node("recall_room_1").has_value());
}

static void test_never_touches_other_nodes(Fixture& f)
{
    add_node(*f.G, DSR::Node::create<metaconcept_node_type>("recall_room_1"), f.room1);
    add_node(*f.G, DSR::Node::create<object_node_type>("chair_3"), f.room1);
    add_node(*f.G, DSR::Node::create<object_node_type>("ltom_recall_77"), f.room1);   // right name, wrong type
    f.pub->sync(f.room1, std::vector{recall(kUid3, 1.0, 1.0, 0.0)});
    f.pub->sync(f.room1, std::vector<ltom::Recall>{});
    f.pub->sync(std::nullopt, {});
    CHECK(f.node("recall_room_1").has_value());
    CHECK(f.node("chair_3").has_value());
    CHECK(f.node("ltom_recall_77").has_value());
}

int main()
{
    const auto run = [](const char* name, void (*fn)(Fixture&))
    {
        const int before = failures;
        Fixture f;
        fn(f);
        std::println("{} {}", failures == before ? "ok  " : "FAIL", name);
    };
    run("test_inserts_under_room", test_inserts_under_room);
    run("test_moves_when_keyframe_changes", test_moves_when_keyframe_changes);
    run("test_removed_when_alive", test_removed_when_alive);
    run("test_no_room_no_recall", test_no_room_no_recall);
    run("test_stale_sweep", test_stale_sweep);
    run("test_never_touches_other_nodes", test_never_touches_other_nodes);
    return failures ? 1 : 0;
}
