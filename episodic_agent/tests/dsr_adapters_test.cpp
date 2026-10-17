// Offline checks for episodic_agent's DSR adapters (DsrFactSource, LiveDiff, EmGraph) and em_query_dsr.h
// load_store. Builds standalone DSR graphs on private DDS domains (60/61) seeded from em_root.json — no fleet,
// no viewer. Plain CHECK macro, non-zero exit on failure, like common/episodic/tests.
#include "../src/dsr_fact_source.h"
#include "../src/em_graph.h"
#include "../../common/episodic/em_query_dsr.h"

#include <dsr/api/dsr_api.h>

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

static int failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (not(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } \
    } while (0)

namespace
{
std::shared_ptr<DSR::DSRGraph> make_graph(const std::string& name, int id, int domain)
{
    return std::make_shared<DSR::DSRGraph>(name, id, std::string(EM_ROOT_JSON), true, static_cast<int8_t>(domain),
                                           DSR::Queue);
}

std::uint64_t add_node(DSR::DSRGraph& G, DSR::Node n, std::uint64_t parent)
{
    G.add_or_modify_attrib_local<parent_att>(n, parent);
    G.add_or_modify_attrib_local<level_att>(n, 1);
    const auto id = G.insert_node(n);
    if (not id) throw std::runtime_error("insert_node failed: " + n.name());
    return *id;
}

void add_rt(DSR::DSRGraph& G, std::uint64_t from, std::uint64_t to, std::vector<float> t, std::vector<float> r,
            std::optional<std::vector<float>> cov)
{
    auto e = DSR::Edge::create<RT_edge_type>(from, to);
    G.add_or_modify_attrib_local<rt_translation_att>(e, std::move(t));
    G.add_or_modify_attrib_local<rt_rotation_euler_xyz_att>(e, std::move(r));
    if (cov) G.add_or_modify_attrib_local<rt_covariance_att>(e, std::move(*cov));
    G.insert_or_assign_edge(e);
}

bool has_kind(const std::vector<rc::em::Event>& ev, const std::string& kind, const std::string& node)
{
    return std::ranges::any_of(ev, [&](const rc::em::Event& e) { return e.kind == kind and e.node == node; });
}
}   // namespace

static void fact_source(DSR::DSRGraph& live, DSR::DSRGraph& mem, std::shared_ptr<DSR::DSRGraph> live_p,
                        std::shared_ptr<DSR::DSRGraph> mem_p)
{
    const auto root = live.get_node("root")->id();
    auto room = DSR::Node::create<room_node_type>("room_1");
    live.add_or_modify_attrib_local<room_id_att>(room, std::uint64_t{1});
    const auto room_id = add_node(live, room, root);

    auto door = DSR::Node::create<object_node_type>("door_3");
    live.add_or_modify_attrib_local<object_subtype_att>(door, std::string("door"));
    live.add_or_modify_attrib_local<door_open_prob_att>(door, 0.75f);
    live.add_or_modify_attrib_local<passage_passable_att>(door, true);
    const auto door_id = add_node(live, door, room_id);
    std::vector<float> cov(36, 0.f);
    cov[0] = 0.25f;
    cov[7] = 0.5f;
    cov[35] = 0.1f;
    add_rt(live, room_id, door_id, {1.f, 2.f, 0.f}, {0.f, 0.f, 0.5f}, cov);

    auto bare = DSR::Node::create<object_node_type>("door_9");     // no RT edge, no attrs, no memory face
    live.add_or_modify_attrib_local<object_subtype_att>(bare, std::string("door"));
    add_node(live, bare, room_id);

    // Memory: face r1_door_3 under its room, passage_0 matching it.
    const auto mroot = mem.get_node("root")->id();
    const auto mroom = add_node(mem, DSR::Node::create<room_node_type>("r1_room_1"), mroot);
    const auto face = add_node(mem, DSR::Node::create<object_node_type>("r1_door_3"), mroom);
    auto passage = DSR::Node::create<metaconcept_node_type>("passage_0");
    mem.add_or_modify_attrib_local<object_subtype_att>(passage, std::string("passage"));
    const auto pid = add_node(mem, passage, mroot);
    mem.insert_or_assign_edge(DSR::Edge::create<match_edge_type>(pid, face));

    rc::em::DsrFactSource fs(live_p, mem_p);
    const auto f = fs.fact("door_3");
    CHECK(f.has_value());
    if (f)
    {
        CHECK(f->node == "door_3");
        CHECK(f->type == "object");
        CHECK(f->ltsm_id == "passage_0");
        const auto& v = f->values;
        CHECK(v.size() == 7);   // door_open_prob passage_passable x y yaw cov_xx cov_yy
        CHECK(v.contains("door_open_prob") and std::abs(v.at("door_open_prob") - 0.75) < 1e-6);
        CHECK(v.contains("passage_passable") and v.at("passage_passable") == 1.0);
        CHECK(not v.contains("door_reach_prob"));   // absent is not zero
        CHECK(v.contains("x") and std::abs(v.at("x") - 1.0) < 1e-6);
        CHECK(v.contains("y") and std::abs(v.at("y") - 2.0) < 1e-6);
        CHECK(v.contains("yaw") and std::abs(v.at("yaw") - 0.5) < 1e-6);
        CHECK(v.contains("cov_xx") and std::abs(v.at("cov_xx") - 0.25) < 1e-6);
        CHECK(v.contains("cov_yy") and std::abs(v.at("cov_yy") - 0.5) < 1e-6);
    }
    const auto b = fs.fact("door_9");
    CHECK(b.has_value() and b->values.empty() and b->ltsm_id.empty());
    CHECK(not fs.fact("no_such_node").has_value());

    rc::em::DsrFactSource no_mem(live_p, nullptr);
    CHECK(no_mem.fact("door_3")->ltsm_id.empty());

    CHECK(fs.current_room().empty());               // no current edge yet
    const auto robot = add_node(live, DSR::Node::create<robot_node_type>("Shadow"), root);
    live.insert_or_assign_edge(DSR::Edge::create<current_edge_type>(robot, room_id));
    CHECK(fs.current_room() == "room_1");
}

static void live_diff(DSR::DSRGraph& live)
{
    rc::em::LiveDiff diff;
    CHECK(diff.poll(live).empty());                 // primes: the existing rooms/doors are not "born"
    CHECK(diff.poll(live).empty());

    const auto root = live.get_node("root")->id();
    const auto room_id = live.get_node("room_1")->id();
    const auto table = add_node(live, DSR::Node::create<table_node_type>("table_1"), room_id);
    const auto aff = add_node(live, DSR::Node::create<affordance_node_type>("aff_table_1"), table);
    const auto room2 = add_node(live, DSR::Node::create<room_node_type>("room_2"), root);
    auto ev = diff.poll(live);
    CHECK(has_kind(ev, "born", "table_1"));
    CHECK(has_kind(ev, "offered", "aff_table_1"));
    CHECK(has_kind(ev, "born", "room_2"));
    CHECK(ev.size() == 3);

    const auto robot = live.get_node("Shadow")->id();
    live.delete_edge(robot, room_id, "current");
    CHECK(diff.poll(live).empty());                 // no current edge: ambiguous, nothing reported
    live.insert_or_assign_edge(DSR::Edge::create<current_edge_type>(robot, room2));
    ev = diff.poll(live);
    CHECK(ev.size() == 1 and has_kind(ev, "current_moved", "room_2"));

    live.delete_node(aff);
    live.delete_node(table);
    ev = diff.poll(live);
    CHECK(has_kind(ev, "withdrawn", "aff_table_1"));
    CHECK(has_kind(ev, "died", "table_1"));
    CHECK(ev.size() == 2);
}

static void em_graph(std::shared_ptr<DSR::DSRGraph> em_p)
{
    auto& em = *em_p;
    rc::em::EmGraph g(em_p);
    rc::em::Record m{.id = "20261017-1", .parent = "", .level = rc::em::Level::Mission, .author = rc::em::Author::Human,
                     .kind = "mission", .target = "", .target_ltsm = "", .action_json = R"({"title":"go"})",
                     .source = "chat", .t0 = 1000, .t1 = {}, .room = "room_1", .pre = {}, .post = {}, .outcome = "",
                     .attempt = 0, .iteration = 0, .promised = {}, .delivered = {}, .alternatives_json = "", .events = {}};
    rc::em::Record s = m;
    s.id = "20261017-2";
    s.parent = m.id;
    s.level = rc::em::Level::Step;
    s.kind = "cross";
    s.target = "door_3";
    s.t0 = 1100;
    s.pre = {rc::em::Fact{.node = "door_3", .type = "object", .ltsm_id = "passage_0", .values = {{"x", 1.0}}}};
    CHECK(g.upsert(m));
    CHECK(g.upsert(s));
    const auto mn = em.get_node(m.id), sn = em.get_node(s.id);
    CHECK(mn.has_value() and mn->type() == "metaconcept");
    CHECK(sn.has_value());
    if (mn and sn)
    {
        CHECK(em.get_edge(em.get_node("root")->id(), mn->id(), "has").has_value());
        CHECK(em.get_edge(mn->id(), sn->id(), "has").has_value());
    }

    rc::em::Record orphan = s;                      // parent not in the graph (e.g. already folded) => under root
    orphan.id = "20261017-3";
    orphan.parent = "20261016-9";
    orphan.t0 = 1200;
    CHECK(g.upsert(orphan));
    CHECK(em.get_edge(em.get_node("root")->id(), em.get_node(orphan.id)->id(), "has").has_value());

    const auto before = em.get_nodes_by_type("metaconcept").size();
    s.t1 = 1500;
    s.outcome = "succeeded";
    CHECK(g.upsert(s));                             // update in place, no second node
    CHECK(em.get_nodes_by_type("metaconcept").size() == before);

    rc::em::Stats st{.key_kind = "cross", .key_identity = "passage_0", .outcomes = {{"succeeded", 2}}, .n = 2,
                     .dur_mean_s = 3.0, .dur_m2 = 2.0, .n_gain = 0, .promised_sum = 0, .delivered_sum = 0,
                     .first_t0 = 10, .last_t0 = 20};
    rc::em::Stats any = st;
    any.key_kind = "mission";
    any.key_identity = "";
    CHECK(rc::em::EmGraph::stats_name(st) == "stats_cross_passage_0");
    CHECK(rc::em::EmGraph::stats_name(any) == "stats_mission_any");
    CHECK(g.upsert(st));
    CHECK(g.upsert(any));
    st.n = 3;
    CHECK(g.upsert(st));

    const rc::em::Store store = rc::em::load_store(em);
    CHECK(store.today.size() == 3);
    CHECK(store.stats.size() == 2);
    if (store.today.size() == 3)
    {
        CHECK(store.today[0] == m);
        CHECK(store.today[1] == s);                 // the closed version
        CHECK(store.today[2] == orphan);
    }
    CHECK(std::ranges::count(store.stats, st) == 1);
    CHECK(std::ranges::count(store.stats, any) == 1);

    CHECK(g.remove(orphan.id));
    CHECK(not em.get_node(orphan.id).has_value());
    CHECK(not g.remove(orphan.id));
    CHECK(rc::em::load_store(em).today.size() == 2);
}

int main()
{
    if (not std::setlocale(LC_ALL, "es_ES.UTF-8")) std::setlocale(LC_ALL, "");
    try
    {
        auto live = make_graph("em_test_live", 961, 60);
        auto mem = make_graph("em_test_mem", 962, 61);
        auto em = make_graph("em_test_em", 963, 62);
        fact_source(*live, *mem, live, mem);
        live_diff(*live);
        em_graph(em);
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "FAIL exception: %s\n", e.what());
        return 1;
    }
    if (failures) { std::fprintf(stderr, "dsr_adapters_test: %d failure(s)\n", failures); return 1; }
    std::puts("dsr_adapters_test: all passed");
    return 0;
}
