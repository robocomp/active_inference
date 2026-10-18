// DoorPassage (controller_door_passage.h): the close-behind step after a crossing, and the hand-back to affordances.
// Build+run (a real DSR graph on a private DDS domain, like common/affordance_manager/tests):
//   g++ -std=c++23 -O0 -g -fPIC -I/usr/include/eigen3 $(pkg-config --cflags Qt6Core) -o /tmp/tdp test_door_passage.cpp \
//       -ldsr_api -ldsr_core -lfastcdr -lfastdds $(pkg-config --libs Qt6Core) && /tmp/tdp
#include "../src/controller_door_passage.h"

#include <print>

namespace
{
int failures = 0, checks = 0;
void check(bool ok, int line, std::string_view what)
{
    ++checks;
    if (not ok)
    {
        ++failures;
        std::println("FAIL line {}: {}", line, what);
    }
}
#define CHECK(x) check((x), __LINE__, #x)

using rc::DoorPassage;
using Step = DoorPassage::Step;

std::shared_ptr<DSR::DSRGraph> make_graph(int k)
{
    return std::make_shared<DSR::DSRGraph>("door_passage_test_" + std::to_string(k), 970 + k,
                                           std::string(SHADOW_JSON), true, static_cast<int8_t>(60 + k), DSR::Queue);
}

/// An affordance node in a protocol state: offered (pending, not active), executing (both), completed (neither).
void put(DSR::DSRGraph& G, const std::string& name, bool pending, bool active,
         std::optional<rc::affordance::Outcome> outcome = std::nullopt)
{
    auto n = G.get_node(name);
    if (not n.has_value())
    {
        DSR::Node m = DSR::Node::create<affordance_node_type>(name);
        G.add_or_modify_attrib_local<epistemic_pending_att>(m, pending);
        G.add_or_modify_attrib_local<active_att>(m, active);
        G.insert_node(m);
        n = G.get_node(name);
    }
    if (outcome.has_value()) rc::affordance::write_outcome(G, *n, *outcome);
    G.add_or_modify_attrib_local<epistemic_pending_att>(*n, pending);
    G.add_or_modify_attrib_local<active_att>(*n, active);
    G.update_node(*n);
}
void offered(DSR::DSRGraph& G, const std::string& n) { put(G, n, true, false); }
void executing(DSR::DSRGraph& G, const std::string& n) { put(G, n, true, true); }
void satisfied(DSR::DSRGraph& G, const std::string& n) { put(G, n, false, false, rc::affordance::Outcome::Satisfied); }

/// approach → open → cross, then the room changes: where every case below starts from.
void cross_into_room_2(DSR::DSRGraph& G, DoorPassage& p, const std::string& door)
{
    const std::string a = "aff_" + door + "_";
    offered(G, a + "approach");
    p.update(G, nullptr, "room_1");
    executing(G, a + "approach");
    p.update(G, nullptr, "room_1");
    satisfied(G, a + "approach");
    p.update(G, nullptr, "room_1");          // approach SATISFIED -> open
    offered(G, a + "cross");                 // the door is open: cross on offer, open is not
    p.update(G, nullptr, "room_1");          // skip to cross
    executing(G, a + "cross");
    p.update(G, nullptr, "room_1");          // cross_room_ = room_1
    p.update(G, nullptr, "room_2");          // room_concept moved the robot: the crossing is done
}
}   // namespace

int main()
{
    std::uint64_t now = 1'000'000;
    const auto clock = [&now]() { return now; };

    // ── a single command's passage: cross, then CLOSE the door behind, then hand the base back to affordances ─────
    {
        auto G = make_graph(0);
        DoorPassage p;
        p.set_clock(clock);
        p.start("door_9", Step::Close, /*then_affordances=*/true);
        cross_into_room_2(*G, p, "door_9");
        CHECK(p.view().step == Step::Close);
        CHECK(not p.take_finished());
        offered(*G, "aff_door_9_close");     // door_concept, in the new room: the door is open and the robot clear of it
        auto out = p.update(*G, nullptr, "room_2");
        CHECK(out.required == "aff_door_9_close");   // the selector may take only `close` now
        CHECK(p.active());
        executing(*G, "aff_door_9_close");
        p.update(*G, nullptr, "room_2");
        satisfied(*G, "aff_door_9_close");   // perceived shut: door_concept completes it
        out = p.update(*G, nullptr, "room_2");
        CHECK(p.view().step == Step::Done);
        CHECK(out.required.empty());
        CHECK(out.event.has_value() and out.event->find("close") != std::string::npos);
        CHECK(p.take_finished());            // once: the caller switches to AffordancesOnly
        CHECK(not p.take_finished());
        CHECK(p.view().then_affordances);
    }

    // ── no `close` ever offered (no door actuator in this world): the step ends at its deadline, door left open ────
    {
        auto G = make_graph(1);
        DoorPassage p;
        p.set_clock(clock);
        p.set_close_wait_ms(5000);
        p.start("door_7", Step::Close, true);
        cross_into_room_2(*G, p, "door_7");
        CHECK(p.view().step == Step::Close);
        now += 4000;
        p.update(*G, nullptr, "room_2");
        CHECK(p.view().step == Step::Close);   // still waiting for door_concept to offer it
        now += 2000;
        const auto out = p.update(*G, nullptr, "room_2");
        CHECK(p.view().step == Step::Done);
        CHECK(out.event.has_value() and out.event->find("left open") != std::string::npos);
        CHECK(p.take_finished());
    }

    // ── a mission's `cross` (stop after Cross): the room change ends it, as before — no close, no hand-back ────────
    {
        auto G = make_graph(2);
        DoorPassage p;
        p.set_clock(clock);
        p.start("door_5", Step::Cross);
        cross_into_room_2(*G, p, "door_5");
        CHECK(p.view().step == Step::Done);
        CHECK(p.take_finished());
        CHECK(not p.view().then_affordances);
    }

    // ── a FAILED close does not hold the robot: the door is left open and the passage ends ─────────────────────────
    {
        auto G = make_graph(3);
        DoorPassage p;
        p.set_clock(clock);
        p.start("door_4", Step::Close, true);
        cross_into_room_2(*G, p, "door_4");
        offered(*G, "aff_door_4_close");
        p.update(*G, nullptr, "room_2");
        executing(*G, "aff_door_4_close");
        p.update(*G, nullptr, "room_2");
        put(*G, "aff_door_4_close", false, false, rc::affordance::Outcome::Timeout);
        const auto out = p.update(*G, nullptr, "room_2");
        CHECK(p.view().step == Step::Done);
        CHECK(out.event.has_value() and out.event->find("left open") != std::string::npos);
    }

    std::println("test_door_passage: {} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
