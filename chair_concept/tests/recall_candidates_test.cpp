// Offline test of src/recall_candidates.h on a standalone DSR graph (private DDS domain 65). No fleet, no viewer.
#include "../src/recall_candidates.h"
#include "../../common/rt_covariance/rt_covariance.h"

#include <cmath>
#include <memory>
#include <print>

static int failures = 0;
#define CHECK(c) do { if (not (c)) { std::println(stderr, "FAIL {}:{}: {}", __FILE__, __LINE__, #c); ++failures; } } while (0)

namespace
{
using Id = std::uint64_t;
constexpr std::uint32_t kMe = 20;
constexpr std::uint64_t uid_of(std::uint32_t agent, int n) { return (std::uint64_t{agent} << 40) | static_cast<std::uint64_t>(n); }

Id add_node(DSR::DSRGraph& G, DSR::Node n, Id parent)
{
    G.add_or_modify_attrib_local<parent_att>(n, parent);
    G.add_or_modify_attrib_local<level_att>(n, 1);
    const auto id = G.insert_node(n);
    if (not id) throw std::runtime_error("insert_node failed: " + n.name());
    return *id;
}

struct Fixture
{
    std::shared_ptr<DSR::DSRGraph> G;
    Id root = 0, room = 0, other_room = 0;
    Fixture()
    {
        G = std::make_shared<DSR::DSRGraph>("chair_recall_test", 65, std::string(CHAIR_TEST_ROOT_JSON), true, static_cast<int8_t>(65), DSR::Queue);
        root = G->get_node("root")->id();
        room = add_node(*G, DSR::Node::create<room_node_type>("room_1"), root);
        auto rn = *G->get_node("root");
        G->get_rt_api()->insert_or_assign_edge_RT(rn, room, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f});
        other_room = add_node(*G, DSR::Node::create<room_node_type>("room_2"), root);
        G->get_rt_api()->insert_or_assign_edge_RT(rn, other_room, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f});
    }
    // A recall node as ltom_agent writes it.
    void recall(Id parent, std::uint64_t uid, float x, float y, float var, const std::string& cls = "chair", bool with_cov = true)
    {
        auto n = DSR::Node::create<metaconcept_node_type>("ltom_recall_" + std::to_string(uid));
        G->add_or_modify_attrib_local<object_uid_att>(n, uid);
        G->add_or_modify_attrib_local<object_subtype_att>(n, cls);
        G->add_or_modify_attrib_local<parent_att>(n, parent);
        G->add_or_modify_attrib_local<level_att>(n, 2);
        const auto id = G->insert_node(n);
        auto pn = *G->get_node(parent);
        G->get_rt_api()->insert_or_assign_edge_RT(pn, *id, {x, y, 0.f}, {0.f, 0.f, 0.f});
        if (with_cov)
        {
            float last = 0.f;
            rc::rtcov::publish(*G, parent, *id, {var, var * 2, 1e-4f, 5e-4f, 5e-4f, 0.1f}, last, true, "t");
        }
    }
};
}   // namespace

static void test_reads_own_chair_recalls(Fixture& f)
{
    f.recall(f.room, uid_of(kMe, 3), 1.5f, -2.0f, 0.02f);
    const auto c = rc::recall_candidates(*f.G, f.room, kMe, 100, {});
    CHECK(c.size() == 1);
    if (c.size() != 1) return;
    CHECK(c[0].cand.uid == uid_of(kMe, 3) and c[0].n == 3);
    CHECK(std::abs(c[0].cand.mu.x() - 1.5f) < 1e-5f and std::abs(c[0].cand.mu.y() + 2.0f) < 1e-5f);
    CHECK(std::abs(c[0].cand.cov(0, 0) - 0.02f) < 1e-6f and std::abs(c[0].cand.cov(1, 1) - 0.04f) < 1e-6f);
    CHECK(c[0].cand.cov(0, 1) == 0.f);
}

static void test_live_name_excluded(Fixture& f)
{
    f.recall(f.room, uid_of(kMe, 3), 1.f, 1.f, 0.02f);
    f.recall(f.room, uid_of(kMe, 4), 2.f, 2.f, 0.02f);
    const auto c = rc::recall_candidates(*f.G, f.room, kMe, 100, {"chair_3"});
    CHECK(c.size() == 1 and c[0].n == 4);
}

static void test_other_agent_uid_excluded(Fixture& f)
{
    f.recall(f.room, uid_of(kMe + 1, 3), 1.f, 1.f, 0.02f);         // another agent's uid
    f.recall(f.room, uid_of(kMe, 5), 1.f, 1.f, 0.02f, "table");     // not a chair
    f.recall(f.other_room, uid_of(kMe, 6), 1.f, 1.f, 0.02f);        // another room
    f.recall(f.room, uid_of(kMe, 7), 1.f, 1.f, 0.02f, "chair", false);   // no rt_covariance: skipped
    CHECK(rc::recall_candidates(*f.G, f.room, kMe, 100, {}).empty());
}

static void test_candidate_above_counter_excluded(Fixture& f)
{
    f.recall(f.room, uid_of(kMe, 9), 1.f, 1.f, 0.02f);
    f.recall(f.room, uid_of(kMe, 4), 2.f, 2.f, 0.02f);
    const auto c = rc::recall_candidates(*f.G, f.room, kMe, 5, {});
    CHECK(c.size() == 1 and c[0].n == 4);
}

static void test_candidate_taken_once()
{
    std::vector<rc::RecallCand> v(3);
    v[0].n = 1; v[1].n = 2; v[2].n = 3;
    const auto taken = rc::take_candidate(v, 1);
    CHECK(taken.n == 2 and v.size() == 2 and v[0].n == 1 and v[1].n == 3);
    // Two births at one spot in one cycle: the first resurrects, the second finds nothing left to take.
    std::vector<rc::RecallCand> one(1);
    one[0].n = 9; one[0].cand.uid = 99; one[0].cand.mu = {0.f, 0.f}; one[0].cand.cov = Eigen::Matrix2f::Identity() * 0.01f;
    const Eigen::Matrix2f R = Eigen::Matrix2f::Identity() * 0.04f;
    rc::recall::Params p; p.pi_move = 0.5f;
    int resurrected = 0;
    for (int birth = 0; birth < 2; ++birth)
    {
        const auto d = rc::recall::decide({0.f, 0.f}, R, rc::as_candidates(one), p);
        if (d.index) { rc::take_candidate(one, *d.index); ++resurrected; }
    }
    CHECK(resurrected == 1 and one.empty());
}

static void test_polygon_area()
{
    const std::vector<Eigen::Vector2f> sq{{0.f, 0.f}, {4.f, 0.f}, {4.f, 3.f}, {0.f, 3.f}};
    CHECK(std::abs(rc::polygon_area(sq) - 12.f) < 1e-5f);
    std::vector<Eigen::Vector2f> rev(sq.rbegin(), sq.rend());
    CHECK(std::abs(rc::polygon_area(rev) - 12.f) < 1e-5f);
    CHECK(rc::polygon_area({}) == 0.f);
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
    run("test_reads_own_chair_recalls", test_reads_own_chair_recalls);
    run("test_live_name_excluded", test_live_name_excluded);
    run("test_other_agent_uid_excluded", test_other_agent_uid_excluded);
    run("test_candidate_above_counter_excluded", test_candidate_above_counter_excluded);
    const auto plain = [](const char* name, void (*fn)())
    {
        const int before = failures;
        fn();
        std::println("{} {}", failures == before ? "ok  " : "FAIL", name);
    };
    plain("test_candidate_taken_once", test_candidate_taken_once);
    plain("test_polygon_area", test_polygon_area);
    return failures ? 1 : 0;
}
