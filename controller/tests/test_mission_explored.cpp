// `explore room_X until explored(room_X)>=b` — the controller's explored() predicate (src/mission_explored.h) on a
// REAL DSR graph, driven by the REAL interpreter (common/mission), Task 5 2026-10-21. room_concept writes
// `room_explored_p` on the room node (room_concept/src/room_explored.h); the predicate reads it, type-attributed.
//   · no attribute (never assessed) / no such room ⇒ unknown ⇒ never satisfies: the explore runs its budget out;
//   · attribute present ⇒ the mission's boundary decides, every tick (the belief rising mid-step ends the step);
//   · a recalled room published explored on entry (complete memory, p = 1) ⇒ nothing is started at all.
// Build+run (a real DSR graph on a private DDS domain, like test_explore_room_scope):
//   g++ -std=c++23 -O0 -g -fPIC -Wno-deprecated-literal-operator -I/usr/include/eigen3 -I../../common \
//       -DSHADOW_JSON="\"$PWD/../../robot_concept/shadow.json\"" $(pkg-config --cflags Qt6Core) -o /tmp/tme \
//       test_mission_explored.cpp -ldsr_api -ldsr_core -lfastcdr -lfastdds $(pkg-config --libs Qt6Core) && /tmp/tme
#include "../src/mission_explored.h"

#include "../../common/mission/mission_interpreter.h"
#include "../../common/mission/mission_text.h"

#include <dsr/api/dsr_api.h>

#include <memory>
#include <print>

namespace
{
int failures = 0, checks = 0;
void check(bool ok, int line, std::string_view what)
{
    ++checks;
    if (not ok) { ++failures; std::println("FAIL line {}: {}", line, what); }
}
#define CHECK(x) check((x), __LINE__, #x)

using namespace rc::mission;
using S = Interpreter::State;

/// The controller's MissionHost reduced to what an explore step touches: start/cancel are logged, eval is the REAL
/// predicate on the graph.
struct GraphHost final : Host
{
    explicit GraphHost(DSR::DSRGraph &g) : G(g) {}
    DSR::DSRGraph &G;
    int started = 0, cancelled = 0;
    std::optional<std::string> start(const Step &) override { ++started; return std::nullopt; }
    LeafStatus poll(const Step &) override { return {}; }
    void cancel(const Step &) override { ++cancelled; }
    void say(const std::string &) override {}
    void halt() override {}
    std::optional<Pose> pose() override { return std::nullopt; }
    std::optional<double> eval(const Pred &p) override
    {
        if (p.name == "explored") return rc::mission::explored_belief(G, p.refs.at("room").name);
        return std::nullopt;
    }
};

Mission parse(std::string_view text)
{
    auto m = parse_text(text);
    if (not m.has_value()) std::println("  parse: {}", m.error().message());
    return m.value_or(Mission{});
}

std::shared_ptr<DSR::DSRGraph> make_graph(int k)
{
    auto G = std::make_shared<DSR::DSRGraph>("mission_explored_test_" + std::to_string(k), 960 + k,
                                             std::string(SHADOW_JSON), true, static_cast<int8_t>(60 + k), DSR::Queue);
    DSR::Node room = DSR::Node::create<room_node_type>("room_2");
    G->add_or_modify_attrib_local<level_att>(room, 1);
    CHECK(G->insert_node(room).has_value());
    return G;
}

void set_explored(DSR::DSRGraph &G, float p)
{
    auto n = G.get_node("room_2");
    CHECK(n.has_value());
    if (not n.has_value()) return;
    G.add_or_modify_attrib_local<room_explored_p_att>(n.value(), p);
    CHECK(G.update_node(n.value()));
}

constexpr std::string_view kExplore = "mission\n  explore room_2 until=explored(room_2)>=0.9 budget_s=30s\n";

void unknown_never_satisfies()
{
    auto G = make_graph(0);
    CHECK(not rc::mission::explored_belief(*G, "room_2").has_value());    // never assessed
    CHECK(not rc::mission::explored_belief(*G, "room_9").has_value());    // no such room
    GraphHost h(*G);
    Interpreter in(h);
    in.start(parse(kExplore), 0);
    in.tick(0);
    CHECK(h.started == 1);
    in.tick(29'000);
    CHECK(in.state() == S::Running);
    in.tick(30'000);
    CHECK(in.status() == "failed|1|budget_exhausted");                    // the budget, not the belief, ended it
}

void attribute_decides()
{
    auto G = make_graph(1);
    set_explored(*G, 0.4f);                                               // the survey has much left
    CHECK(rc::mission::explored_belief(*G, "room_2").has_value());
    GraphHost h(*G);
    Interpreter in(h);
    in.start(parse(kExplore), 0);
    in.tick(0);
    CHECK(h.started == 1);
    set_explored(*G, 0.85f);                                              // below the person's boundary: keep going
    in.tick(5'000);
    CHECK(in.state() == S::Running and h.cancelled == 0);
    set_explored(*G, 0.93f);                                              // the best look left is worth < 1/9 trip
    in.tick(6'000);
    CHECK(in.state() == S::Completed and h.cancelled == 1);
    const auto v = rc::mission::explored_belief(*G, "room_2");
    CHECK(v.has_value() and std::abs(*v - 0.93) < 1e-6);
}

void recalled_room_starts_explored()
{
    auto G = make_graph(2);
    set_explored(*G, 1.f);                                                // room_concept: complete memory on re-entry
    GraphHost h(*G);
    Interpreter in(h);
    in.start(parse(kExplore), 0);
    in.tick(0);
    CHECK(in.state() == S::Completed and h.started == 0);                 // already true: nothing is started
}
}   // namespace

int main()
{
    unknown_never_satisfies();
    attribute_decides();
    recalled_room_starts_explored();
    std::println("test_mission_explored: {}/{} checks passed", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
