// Offline checks for episodic_agent's DSR adapters (DsrFactSource, LiveDiff, EmGraph) and em_query_dsr.h
// load_store. Builds standalone DSR graphs on private DDS domains (60/61) seeded from em_root.json — no fleet,
// no viewer. Plain CHECK macro, non-zero exit on failure, like common/episodic/tests.
#include "../src/dsr_fact_source.h"
#include "../src/em_graph.h"
#include "../src/episode_text.h"
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

    // The scene: names per kEventNodeTypes, affordances apart, sorted; the robot is not a thing in it.
    add_node(live, DSR::Node::create<affordance_node_type>("aff_door_3_open"), door_id);
    const auto sc = fs.scene();
    CHECK(sc.has_value());
    CHECK(sc and sc->things == std::vector<std::string>({"door_3", "door_9", "room_1"}));
    CHECK(sc and sc->offered == std::vector<std::string>({"aff_door_3_open"}));
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

    rc::em::Stats st{.key_level = "claim", .key_kind = "cross", .key_identity = "passage_0", .outcomes = {{"succeeded", 2}},
                     .n = 2, .n_dur = 2, .dur_mean_s = 3.0, .dur_m2 = 2.0, .n_gain = 0, .promised_sum = 0, .delivered_sum = 0,
                     .first_t0 = 10, .last_t0 = 20};
    rc::em::Stats any = st;
    any.key_level = "mission";
    any.key_kind = "mission";
    any.key_author = "human";
    any.key_identity = "";
    CHECK(rc::em::EmGraph::stats_name(st) == "stats_claim_cross_passage_0");
    CHECK(rc::em::EmGraph::stats_name(any) == "stats_mission_mission_human_any");
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

    // "View data" text, keyed by node name (episode_text.h): a record → its mission's story (episode_story.h); a stats
    // node → its line; root → the day's missions and the statistics.
    const auto has = [](const std::string& text, std::string_view part) { return text.find(part) != std::string::npos; };
    const std::string mission_text = rc::em::node_text(store, m.id);
    CHECK(has(mission_text, "MISSION \"go\"   [20261017-1]"));
    CHECK(has(mission_text, "Asked by a person through the chat."));
    CHECK(has(mission_text, "END     not yet"));       // m is still open
    CHECK(has(mission_text, "STEP cross door_3  — succeeded"));
    CHECK(has(mission_text, "CONSEQUENCES OF ITS ACTIONS"));
    CHECK(has(mission_text, "cross <door>"));          // the abstract form: instance → type
    CHECK(not has(mission_text, "»"));                  // nothing marked when the mission itself is clicked
    const std::string step_text = rc::em::node_text(store, s.id);
    CHECK(has(step_text, "MISSION \"go\""));           // a step tells its whole mission…
    CHECK(has(step_text, "» "));                        // …with its own lines marked
    CHECK(has(step_text, "belong to 20261017-2"));
    CHECK(has(rc::em::node_text(store, "stats_claim_cross_passage_0"), "claim cross (all time): 3 times"));
    const std::string root_text = rc::em::node_text(store, "root");
    CHECK(has(root_text, "\"go\", asked by a person — still running   [20261017-1]"));
    CHECK(has(root_text, "all-time statistics:"));
    CHECK(has(rc::em::node_text(store, "nonesuch"), "no episode record"));
    CHECK(rc::em::node_text(rc::em::Store{}, "root") == "episodic memory is empty");
    {   // the story keeps what concerns the mission: a flickering door is one line, a passing chair goes to MEANWHILE
        rc::em::Record fm = m, fs = s;
        fm.t1 = 9000;
        fm.outcome = "completed";
        fs.t1 = 9000;
        fs.outcome = "completed";                   // same end as the mission: told once, by the mission's line
        for (int i = 0; i < 6; ++i) fm.events.push_back({.dt_ms = 100 + 100 * i, .kind = i % 2 ? "died" : "born", .node = "door_3"});
        fm.events.push_back({.dt_ms = 50, .kind = "born", .node = "chair_1"});
        fs.events.push_back({.dt_ms = 10, .kind = "offered", .node = "aff_door_3_open"});
        const std::string t = *rc::em::story::tell(rc::em::Store{.today = {fm, fs}, .stats = {}}, fm.id);
        const auto count = [&](std::string_view part)
        {
            std::size_t n = 0;
            for (auto p = t.find(part); p != std::string::npos; p = t.find(part, p + 1)) ++n;
            return n;
        };
        CHECK(count("door_3 kept appearing and dropping out") == 1);
        CHECK(not has(t, "appeared in its world model: door_3"));
        CHECK(has(t, "it could now: open door_3"));
        CHECK(has(t, "MEANWHILE, NOT RELATED TO THE MISSION\n  1 other thing came and went in the world model: chair_1"));
        // what applying it changed: the door's belief, and the room that appeared (net, within the step's window)
        rc::em::Record fc = fs;                     // an applied affordance under the step
        fc.id = "20261017-4";
        fc.parent = fs.id;
        fc.level = rc::em::Level::Claim;
        fc.kind = "open";
        fc.t0 = 2000;
        fc.t1 = 3000;
        fc.outcome = "satisfied";
        fc.pre = {rc::em::Fact{.node = "door_3", .type = "object", .ltsm_id = "", .values = {{"door_open_prob", 0.0}}}};
        fc.post = {rc::em::Fact{.node = "door_3", .type = "object", .ltsm_id = "", .values = {{"door_open_prob", 0.9}}}};
        fc.events = {{.dt_ms = 900, .kind = "offered", .node = "aff_door_3_cross"},
                     {.dt_ms = 950, .kind = "offered", .node = "aff_door_3_open"}};     // its own affordance: not an effect
        rc::em::Record fs2 = fs;                    // the step settled (em_builder.h SETTLING) in the next room
        fs2.room_after = "room_2";
        fs2.settled = true;
        fs2.t_post = 10500;
        const std::string c = *rc::em::story::tell(rc::em::Store{.today = {fm, fs2, fc}, .stats = {}}, fc.id);
        CHECK(has(c, "robot moved: room_1 → room_2"));
        CHECK(has(c, "⇒ the robot is in another room"));
        CHECK(has(c, "APPLIED open door_3  — succeeded"));
        CHECK(has(c, "belief: door_3 open 0% → 90%"));
        CHECK(has(c, "became possible: cross door_3"));
        CHECK(not has(c, "became possible: open door_3"));
        CHECK(has(c, "open <door>          succeeded        ⇒ <door> open 0% → 90%; then possible: cross <door>"));
        CHECK(has(t, "the mission ended — done, the order was carried out"));
        CHECK(has(t, ", 8 s later — done"));
    }

    // Folding a parent while its child is still open (a mission closed before midnight, its claim not): deleting the
    // parent must re-hang the child under root — cortex delete_node does not cascade or re-parent.
    CHECK(g.remove(m.id));
    CHECK(not em.get_node(m.id).has_value());
    {
        const auto rid = em.get_node("root")->id();
        const auto child = em.get_node(s.id);
        CHECK(child.has_value());
        if (child)
        {
            CHECK(em.get_edge(rid, child->id(), "has").has_value());
            CHECK(em.get_attrib_by_name<parent_att>(*child).value_or(0) == rid);
        }
        s.post = s.pre;                             // a later upsert keeps it hung
        CHECK(g.upsert(s));
        CHECK(em.get_edge(rid, em.get_node(s.id)->id(), "has").has_value());
    }
    CHECK(rc::em::load_store(em).today.size() == 2);
    CHECK(g.upsert(m));                             // restore the mission for the checks below

    CHECK(g.remove(orphan.id));
    CHECK(not em.get_node(orphan.id).has_value());
    CHECK(not g.remove(orphan.id));
    CHECK(rc::em::load_store(em).today.size() == 2);   // m (re-upserted) + s
}

// Finding 4: the journal lives on its own intention node "em_journal", not on the controller's agent node.
static void journal_node(DSR::DSRGraph& live)
{
    CHECK(not rc::em::read_journal(live).has_value());
    const auto root = live.get_node("root")->id();
    auto agent = DSR::Node::create<agent_node_type>("controller 4");   // the old place: no longer read
    live.add_or_modify_attrib_local<em_journal_att>(agent, std::string(R"({"epoch":1,"entries":[]})"));
    add_node(live, agent, root);
    CHECK(not rc::em::read_journal(live).has_value());
    auto n = DSR::Node::create<intention_node_type>(rc::em::kJournalNodeName);
    const auto id = add_node(live, n, root);
    CHECK(not rc::em::read_journal(live).has_value());             // no journal yet
    auto got = live.get_node(id);
    live.add_or_modify_attrib_local<em_journal_att>(*got, std::string(R"({"epoch":2,"entries":[]})"));
    live.update_node(*got);
    CHECK(rc::em::read_journal(live) == std::string(R"({"epoch":2,"entries":[]})"));
    live.delete_node(id);
    CHECK(not rc::em::read_journal(live).has_value());
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
        journal_node(*live);
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
