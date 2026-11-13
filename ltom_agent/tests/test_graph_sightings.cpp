// Offline test of src/graph_sightings.h: a standalone DSR graph on the private DDS domain 63, seeded from
// ltom_root.json. No fleet, no viewer.
#include "../src/graph_sightings.h"

#include <dsr/api/dsr_api.h>

#include <clocale>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <limits>
#include <memory>
#include <print>

static int failures = 0;
#define CHECK(c) do { if (not (c)) { std::println(stderr, "FAIL {}:{}: {}", __FILE__, __LINE__, #c); ++failures; } } while (0)

namespace
{
using Id = std::uint64_t;
constexpr std::uint64_t kUid3 = (std::uint64_t{20} << 40) | 3;

std::uint64_t add_node(DSR::DSRGraph& G, DSR::Node n, Id parent)
{
    G.add_or_modify_attrib_local<parent_att>(n, parent);
    G.add_or_modify_attrib_local<level_att>(n, 1);
    const auto id = G.insert_node(n);
    if (not id) throw std::runtime_error("insert_node failed: " + n.name());
    return *id;
}
void add_rt(DSR::DSRGraph& G, Id from, Id to, std::vector<float> t, float yaw, float var)
{
    auto e = DSR::Edge::create<RT_edge_type>(from, to);
    G.add_or_modify_attrib_local<rt_translation_att>(e, std::move(t));
    G.add_or_modify_attrib_local<rt_rotation_euler_xyz_att>(e, std::vector<float>{0.f, 0.f, yaw});
    std::vector<float> cov(36, 0.f);
    cov[0] = cov[7] = cov[35] = var;
    G.add_or_modify_attrib_local<rt_covariance_att>(e, std::move(cov));
    G.insert_or_assign_edge(e);
}
Id add_object(DSR::DSRGraph& G, const std::string& name, Id parent, std::optional<std::uint64_t> uid,
              std::vector<float> t, float yaw, float var, float w = 0.5f, float d = 0.4f, float h = 1.f)
{
    auto n = DSR::Node::create<object_node_type>(name);
    if (uid) G.add_or_modify_attrib_local<object_uid_att>(n, *uid);
    G.add_or_modify_attrib_local<width_m_att>(n, w);
    G.add_or_modify_attrib_local<depth_m_att>(n, d);
    G.add_or_modify_attrib_local<height_m_att>(n, h);
    G.add_or_modify_attrib_local<timestamp_creation_att>(n, std::uint64_t{1700000000123});
    const auto id = add_node(G, n, parent);
    add_rt(G, parent, id, std::move(t), yaw, var);
    return id;
}
const ltom::Sighting* find(const ltom::GraphView& v, const std::string& name)
{
    for (const auto& s : v.alive) if (s.name == name) return &s;
    return nullptr;
}
const std::vector<std::string> kClasses{"chair", "table", "bottle", "cabinet", "refrigerator"};

struct Fixture
{
    std::shared_ptr<DSR::DSRGraph> G;
    std::unique_ptr<DSR::InnerEigenAPI> ie;
    Id root = 0, room = 0, table = 0;
    Fixture()
    {
        G = std::make_shared<DSR::DSRGraph>("ltom_test", 63, std::string(LTOM_ROOT_JSON), true, static_cast<int8_t>(63), DSR::Queue);
        ie = G->get_inner_eigen_api();
        root = G->get_node("root")->id();
        auto r = DSR::Node::create<room_node_type>("room_1");
        G->add_or_modify_attrib_local<room_id_att>(r, std::uint64_t{1});
        room = add_node(*G, r, root);
        add_rt(*G, root, room, {0.f, 0.f, 0.f}, 0.f, 0.01f);
        const auto c3 = add_object(*G, "chair_3", room, kUid3, {1.f, 2.f, 0.f}, 0.5f, 0.01f);
        { auto n = *G->get_node(c3); G->add_or_modify_attrib_local<obj_exist_prob_att>(n, 0.8f); G->update_node(n); }
        table = add_object(*G, "table_1", room, (std::uint64_t{21} << 40) | 1, {3.f, 1.f, 0.f}, 0.5f, 0.02f);
        add_object(*G, "bottle_2", table, (std::uint64_t{22} << 40) | 2, {0.2f, 0.f, 0.7f}, 0.f, 0.03f);
        add_object(*G, "door_1", room, (std::uint64_t{23} << 40) | 1, {0.f, 0.f, 0.f}, 0.f, 0.01f);
        auto p = DSR::Node::create<room_node_type>("room_9");
        const auto proto = add_node(*G, p, root);
        add_rt(*G, root, proto, {9.f, 0.f, 0.f}, 0.f, 0.01f);
        G->insert_or_assign_edge(DSR::Edge::create<proto_edge_type>(proto, proto));
        add_object(*G, "chair_9", proto, (std::uint64_t{20} << 40) | 9, {1.f, 1.f, 0.f}, 0.f, 0.01f);
        add_node(*G, DSR::Node::create<agent_node_type>("chair_concept 20"), root);
    }
    ltom::GraphView read() { return ltom::read_graph(*G, *ie, kClasses); }
};
}   // namespace

static void test_reads_chair_pose_and_sizes(Fixture& f)
{
    const auto v = f.read();
    const auto* s = find(v, "chair_3");
    CHECK(s != nullptr);
    if (not s) return;
    CHECK(s->uid == kUid3);
    CHECK(s->cls == "chair");
    CHECK(s->room_key == 1);
    CHECK(std::abs(s->pose.x - 1.0) < 1e-5 and std::abs(s->pose.y - 2.0) < 1e-5 and std::abs(s->pose.yaw - 0.5) < 1e-5);
    CHECK(std::abs(s->pose.vx - 0.01) < 1e-6 and std::abs(s->pose.vy - 0.01) < 1e-6 and std::abs(s->pose.vyaw - 0.01) < 1e-6);
    CHECK(s->w == 0.5f and s->d == 0.4f and s->h == 1.f);
    CHECK(s->born_ms == 1700000000123);
    CHECK(s->exist_prob == 0.8f);
}
static void test_bottle_on_table_resolves_room(Fixture& f)
{
    const auto v = f.read();
    const auto* s = find(v, "bottle_2");
    CHECK(s != nullptr);
    if (not s) return;
    CHECK(s->room_key == 1);
    CHECK(std::abs(s->pose.x - (3.0 + 0.2 * std::cos(0.5))) < 1e-4);
    CHECK(std::abs(s->pose.y - (1.0 + 0.2 * std::sin(0.5))) < 1e-4);
    CHECK(std::abs(s->pose.yaw - 0.5) < 1e-4);
    CHECK(std::abs(s->pose.vx - 0.03) < 1e-6);   // the bottle's own edge, not the table's
}
static void test_door_and_proto_skipped(Fixture& f)
{
    // stderr captured: an object skipped because its room is provisional must SAY so (2026-11-11: a chair in a
    // proto-room during a from-scratch tour looked like "LTOM does not see it", with nothing in the terminal).
    const std::string path = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") + "/ltom_held_stderr.txt";
    std::fflush(stderr);
    const int saved = dup(2);
    std::FILE* cap = std::fopen(path.c_str(), "w");
    dup2(fileno(cap), 2);
    const auto v = f.read();
    std::fflush(stderr);
    dup2(saved, 2); close(saved); std::fclose(cap);
    std::stringstream log; log << std::ifstream(path).rdbuf();
    CHECK(find(v, "door_1") == nullptr);
    CHECK(find(v, "chair_9") == nullptr);
    CHECK(v.alive.size() == 3);
    CHECK(log.str().find("chair_9") != std::string::npos and log.str().find("provisional") != std::string::npos);
}
static void test_missing_uid_skipped(Fixture& f)
{
    add_object(*f.G, "chair_5", f.room, std::nullopt, {0.f, 0.f, 0.f}, 0.f, 0.01f);
    CHECK(find(f.read(), "chair_5") == nullptr);
}
static void test_owners_present_parsed(Fixture& f)
{
    const auto v = f.read();
    CHECK(v.owners_present == std::set<std::uint32_t>({20}));
}
static void test_nonfinite_pose_skipped(Fixture& f)
{
    add_object(*f.G, "chair_6", f.room, (std::uint64_t{20} << 40) | 6, {0.f, 0.f, 0.f}, 0.f,
               std::numeric_limits<float>::quiet_NaN());
    add_object(*f.G, "chair_7", f.room, (std::uint64_t{20} << 40) | 7, {std::numeric_limits<float>::infinity(), 0.f, 0.f}, 0.f, 0.01f);
    const auto v = f.read();
    CHECK(find(v, "chair_6") == nullptr);
    CHECK(find(v, "chair_7") == nullptr);
    CHECK(find(v, "chair_3") != nullptr);
}
static void test_duplicate_uid_emitted_once(Fixture& f)
{
    add_object(*f.G, "chair_8", f.room, kUid3, {5.f, 5.f, 0.f}, 0.f, 0.01f);   // same uid as chair_3
    const auto v = f.read();
    int n = 0;
    for (const auto& s : v.alive) n += (s.uid == kUid3);
    CHECK(n == 1);
    CHECK(find(v, "chair_3") != nullptr);   // first by node id wins
    CHECK(find(v, "chair_8") == nullptr);
}

static void test_unresolvable_instance_is_present(Fixture& f)
{
    // no covariance on its parent edge
    auto n = DSR::Node::create<object_node_type>("chair_10");
    f.G->add_or_modify_attrib_local<object_uid_att>(n, (std::uint64_t{20} << 40) | 10);
    const auto id = add_node(*f.G, n, f.room);
    auto e = DSR::Edge::create<RT_edge_type>(f.room, id);
    f.G->add_or_modify_attrib_local<rt_translation_att>(e, std::vector<float>{0.f, 0.f, 0.f});
    f.G->add_or_modify_attrib_local<rt_rotation_euler_xyz_att>(e, std::vector<float>{0.f, 0.f, 0.f});
    f.G->insert_or_assign_edge(e);
    // room ancestor deleted (room_concept's clean stop): the chair node stays
    const Id orphan = add_object(*f.G, "chair_11", f.room, (std::uint64_t{20} << 40) | 11, {0.f, 0.f, 0.f}, 0.f, 0.01f);
    CHECK(f.G->delete_node(f.room));
    const auto v = f.read();
    const std::uint64_t u10 = (std::uint64_t{20} << 40) | 10, u11 = (std::uint64_t{20} << 40) | 11;
    (void)orphan;
    CHECK(find(v, "chair_10") == nullptr and find(v, "chair_11") == nullptr);
    CHECK(v.present.contains(u10) and v.present.contains(u11));
}
static void test_degraded_owner_not_present(Fixture& f)
{
    auto a = f.G->get_node("chair_concept 20");
    CHECK(a.has_value());
    if (not a) return;
    f.G->add_or_modify_attrib_local<agent_presence_state_att>(*a, std::string("Degraded"));
    f.G->update_node(*a);
    CHECK(f.read().owners_present.empty());
    f.G->add_or_modify_attrib_local<agent_presence_state_att>(*a, std::string("Operating"));
    f.G->update_node(*a);
    CHECK(f.read().owners_present == std::set<std::uint32_t>({20}));
}

int main()
{
    std::setlocale(LC_ALL, "");
    auto run = [](const char* name, void (*fn)(Fixture&))
    {
        const int before = failures;
        Fixture f;
        fn(f);
        std::println("{} {}", failures == before ? "ok  " : "FAIL", name);
    };
    run("test_door_and_proto_skipped", test_door_and_proto_skipped);   // FIRST: the held line is logged once per object
    run("test_reads_chair_pose_and_sizes", test_reads_chair_pose_and_sizes);
    run("test_bottle_on_table_resolves_room", test_bottle_on_table_resolves_room);
    run("test_missing_uid_skipped", test_missing_uid_skipped);
    run("test_owners_present_parsed", test_owners_present_parsed);
    run("test_nonfinite_pose_skipped", test_nonfinite_pose_skipped);
    run("test_duplicate_uid_emitted_once", test_duplicate_uid_emitted_once);
    run("test_unresolvable_instance_is_present", test_unresolvable_instance_is_present);
    run("test_degraded_owner_not_present", test_degraded_owner_not_present);
    return failures ? 1 : 0;
}
