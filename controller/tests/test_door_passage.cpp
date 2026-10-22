// DoorPassage (controller_door_passage.h): the close-behind step after a crossing, and the hand-back to affordances.
// Build+run (a real DSR graph on a private DDS domain, like common/affordance_manager/tests):
//   g++ -std=c++23 -O0 -g -fPIC -I/usr/include/eigen3 -I../../common -DSHADOW_JSON="\"$PWD/../../robot_concept/shadow.json\"" \
//       $(pkg-config --cflags Qt6Core) -o /tmp/tdp test_door_passage.cpp \
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
/// The standpoint an affordance names (epistemic_target_x/y_m), as door_concept writes it.
void place(DSR::DSRGraph& G, const std::string& name, float x, float y)
{
    auto n = G.get_node(name);
    if (not n.has_value()) return;
    G.add_or_modify_attrib_local<epistemic_target_x_m_att>(*n, x);
    G.add_or_modify_attrib_local<epistemic_target_y_m_att>(*n, y);
    G.update_node(*n);
}
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

    // ── final review I-1: an ANCHORED cross clears past the room switch; its anchor is then lost with no gap to carry it.
    //    The session drops `engaged` (rc::anchor_gone_action == EndClearing); the passage must END (room-switch rule), not
    //    stay `clearing` for ever. ──────────────────────────────────────────────────────────────────────────────────────
    {
        auto G = make_graph(4);
        DoorPassage p;
        p.set_clock(clock);
        p.start("door_3", Step::Cross);
        const std::string a = "aff_door_3_";
        offered(*G, a + "approach");
        p.update(*G, nullptr, "room_1");
        executing(*G, a + "approach");
        p.update(*G, nullptr, "room_1");
        satisfied(*G, a + "approach");
        p.update(*G, nullptr, "room_1");
        offered(*G, a + "cross");
        p.update(*G, nullptr, "room_1");
        executing(*G, a + "cross");
        p.note_anchor(/*engaged=*/true, /*through=*/false);
        p.update(*G, nullptr, "room_1");
        p.update(*G, nullptr, "room_2");            // room switch on the anchor: keep going until through
        CHECK(p.clearing());
        CHECK(p.view().step == Step::Cross);
        p.update(*G, nullptr, "room_2");
        CHECK(p.clearing());                        // still engaged: still clearing
        CHECK(not p.take_finished());
        p.note_anchor(/*engaged=*/false, /*through=*/false);   // the anchor is gone, no gap: the session gives it up
        const auto out = p.update(*G, nullptr, "room_2");
        CHECK(not p.clearing());
        CHECK(p.view().step == Step::Done);
        CHECK(out.event.has_value() and out.event->find("room changed") != std::string::npos);
        CHECK(p.take_finished());
    }

    // ── Task 3f r1 (M-1): a clearing GIVEN UP while the session's posterior says the body may straddle the doorway is NOT
    //    through: the passage must not go on to close the door on the robot. With P(through) > 1/2 it closes as before. ──
    for (const double p_thr : {0.20, 0.90})
    {
        auto G = make_graph(p_thr < 0.5 ? 6 : 7);
        DoorPassage p;
        p.set_clock(clock);
        p.start("door_6", Step::Close);
        const std::string a = "aff_door_6_";
        offered(*G, a + "approach");
        p.update(*G, nullptr, "room_1");
        executing(*G, a + "approach");
        p.update(*G, nullptr, "room_1");
        satisfied(*G, a + "approach");
        p.update(*G, nullptr, "room_1");
        offered(*G, a + "cross");
        p.update(*G, nullptr, "room_1");
        executing(*G, a + "cross");
        p.note_anchor(/*engaged=*/true, /*through=*/false);
        p.update(*G, nullptr, "room_1");
        p.update(*G, nullptr, "room_2");
        CHECK(p.clearing());
        p.note_through_p(p_thr);
        p.note_anchor(/*engaged=*/false, /*through=*/false);   // cross_hand_back while clearing: given up
        const auto out = p.update(*G, nullptr, "room_2");
        CHECK(not p.clearing());
        if (p_thr < 0.5)
        {
            CHECK(p.view().step == Step::Done);
            CHECK(out.event.has_value() and out.event->find("NOT through") != std::string::npos);
        }
        else
            CHECK(p.view().step == Step::Close);
    }

    // ── Task 3f r1 (M-8): the switch counts only INTO a room this door leads to. ltsm's passage node matches door_8 (in room_8,
    //    the far face) and door_7 (in room_7, where the cross begins): a relocaliser jump to room_9 is not a switch. ────────
    {
        auto G = make_graph(8);
        auto add = [&](const std::string &name, auto type_tag) {
            DSR::Node n = DSR::Node::create<decltype(type_tag)>(name);
            G->insert_node(n);
            return G->get_node(name)->id();
        };
        const auto r7 = add("room_7", room_node_type{}), r8 = add("room_8", room_node_type{}), r9 = add("room_9", room_node_type{});
        (void)r9;
        const auto d7 = add("door_7", object_node_type{}), d8 = add("door_8", object_node_type{});
        const auto ps = add("passage_7", metaconcept_node_type{});
        auto edge = [&](std::uint64_t from, std::uint64_t to, auto type_tag) {
            auto e = DSR::Edge::create<decltype(type_tag)>(from, to);
            G->insert_or_assign_edge(e);
        };
        edge(r7, d7, RT_edge_type{});
        edge(r8, d8, RT_edge_type{});
        edge(ps, d7, match_edge_type{});
        edge(ps, d8, match_edge_type{});
        DoorPassage p;
        p.set_clock(clock);
        p.start("door_7", Step::Cross);
        const std::string a = "aff_door_7_";
        offered(*G, a + "approach");
        p.update(*G, nullptr, "room_7");
        executing(*G, a + "approach");
        p.update(*G, nullptr, "room_7");
        satisfied(*G, a + "approach");
        p.update(*G, nullptr, "room_7");
        offered(*G, a + "cross");
        p.update(*G, nullptr, "room_7");
        executing(*G, a + "cross");
        p.note_anchor(/*engaged=*/true, /*through=*/false);
        p.update(*G, nullptr, "room_7");
        p.update(*G, nullptr, "room_9");   // a jump into a room door_7 does not lead to
        CHECK(not p.clearing());
        CHECK(p.view().step == Step::Cross);
        p.update(*G, nullptr, "room_8");   // the far face's room: the switch through this door
        CHECK(p.clearing());
    }


    // ── Live fix 1 (+ review): the fallback to `approach` after reaching the door needs EVIDENCE the robot is out of
    //    position — its measured distance from door_concept's approach standpoint beyond the arrival tolerance — and a last
    //    outcome unset or Satisfied. Helper: approach → open satisfied, robot at the approach standpoint (0, 0), tol 0.25.
    auto to_cross = [&](DSR::DSRGraph& G, DoorPassage& p) {
        p.set_clock(clock);
        p.start("door_2", Step::Cross);
        p.note_robot(Eigen::Vector2f(0.f, 0.f), 0.25f);
        offered(G, "aff_door_2_approach"); place(G, "aff_door_2_approach", 0.f, 0.f);
        p.update(G, nullptr, "room_1");
        executing(G, "aff_door_2_approach"); p.update(G, nullptr, "room_1");
        satisfied(G, "aff_door_2_approach"); p.update(G, nullptr, "room_1");   // -> open
        offered(G, "aff_door_2_open"); place(G, "aff_door_2_open", 0.1f, 0.1f);
        p.update(G, nullptr, "room_1");
        executing(G, "aff_door_2_open"); p.update(G, nullptr, "room_1");
        satisfied(G, "aff_door_2_open"); p.update(G, nullptr, "room_1");       // -> cross
    };
    {   // (c) the live case: the robot drifted 0.26 m BEYOND the arrival tolerance — falls back, then proceeds to cross
        auto G = make_graph(5);
        DoorPassage p;
        to_cross(*G, p);
        CHECK(p.view().step == Step::Cross);
        p.update(*G, nullptr, "room_1");
        CHECK(p.view().step == Step::Cross);                  // nothing offered: it waits
        p.note_robot(Eigen::Vector2f(0.f, -0.51f), 0.25f);    // released 0.51 m from the approach standpoint
        offered(*G, "aff_door_2_approach");
        auto out = p.update(*G, nullptr, "room_1");
        CHECK(p.view().step == Step::Approach);               // back to approach — not a wait for ever
        CHECK(out.event.has_value() and out.event->find("back to") != std::string::npos);
        CHECK(out.required == "aff_door_2_approach");
        executing(*G, "aff_door_2_approach");
        p.update(*G, nullptr, "room_1");
        p.note_robot(Eigen::Vector2f(0.f, 0.f), 0.25f);
        satisfied(*G, "aff_door_2_approach");                 // back inside the band
        offered(*G, "aff_door_2_cross");
        p.update(*G, nullptr, "room_1");
        p.update(*G, nullptr, "room_1");
        CHECK(p.view().step == Step::Cross);
        out = p.update(*G, nullptr, "room_1");
        CHECK(out.required == "aff_door_2_cross");
        executing(*G, "aff_door_2_cross");                    // existing protocol: an executing cross is not pulled back
        p.update(*G, nullptr, "room_1");
        p.note_robot(Eigen::Vector2f(0.f, -0.8f), 0.25f);
        offered(*G, "aff_door_2_approach");
        p.update(*G, nullptr, "room_1");
        CHECK(p.view().step == Step::Cross);
    }
    {   // (a) cross flickers absent for one cycle while the robot is INSIDE the band and approach is offered: NO fallback
        auto G = make_graph(6);
        DoorPassage p;
        to_cross(*G, p);
        offered(*G, "aff_door_2_cross"); p.update(*G, nullptr, "room_1");
        CHECK(p.view().step == Step::Cross);
        put(*G, "aff_door_2_cross", false, false);            // withdrawn for a cycle (a P(passable) flicker)
        offered(*G, "aff_door_2_approach");
        p.note_robot(Eigen::Vector2f(0.05f, 0.1f), 0.25f);    // at the door, inside the tolerance
        p.update(*G, nullptr, "room_1");
        CHECK(p.view().step == Step::Cross);                  // waits for door_concept as before
        offered(*G, "aff_door_2_cross");
        const auto out = p.update(*G, nullptr, "room_1");
        CHECK(out.required == "aff_door_2_cross");
    }
    {   // (b) open absent AND cross absent, robot AT the approach standpoint: no Approach<->Open cycling, no repeated events
        auto G = make_graph(7);
        DoorPassage p;
        p.set_clock(clock);
        p.start("door_2", Step::Cross);
        p.note_robot(Eigen::Vector2f(0.f, 0.f), 0.25f);
        offered(*G, "aff_door_2_approach"); place(*G, "aff_door_2_approach", 0.f, 0.f);
        p.update(*G, nullptr, "room_1");
        executing(*G, "aff_door_2_approach"); p.update(*G, nullptr, "room_1");
        satisfied(*G, "aff_door_2_approach"); p.update(*G, nullptr, "room_1");   // -> open
        CHECK(p.view().step == Step::Open);
        offered(*G, "aff_door_2_approach");                    // door_concept keeps offering approach; open and cross absent
        int events = 0;
        for (int i = 0; i < 10; ++i) { const auto o = p.update(*G, nullptr, "room_1"); if (o.event.has_value()) ++events; }
        CHECK(p.view().step == Step::Open);
        CHECK(events == 0);
    }
    {   // (2) after a FAILED cross the existing retry path is unchanged: no fallback, "waiting for it to be offered again"
        auto G = make_graph(8);
        DoorPassage p;
        to_cross(*G, p);
        offered(*G, "aff_door_2_cross"); p.update(*G, nullptr, "room_1");
        executing(*G, "aff_door_2_cross"); p.update(*G, nullptr, "room_1");
        put(*G, "aff_door_2_cross", false, false, rc::affordance::Outcome::Unreachable);
        const auto failed = p.update(*G, nullptr, "room_1");
        CHECK(failed.event.has_value() and failed.event->find("waiting for it to be offered again") != std::string::npos);
        p.note_robot(Eigen::Vector2f(0.f, -0.8f), 0.25f);    // even out of position
        offered(*G, "aff_door_2_approach");
        p.update(*G, nullptr, "room_1");
        CHECK(p.view().step == Step::Cross);                  // the retry path waits for cross; no fallback
        CHECK(p.view().last_outcome != "satisfied" and not p.view().last_outcome.empty());
    }


    // ── Live fix 3 (tourP7 leg 1): `cross door_2` while door_2 is NOT in the graph but door_concept offers the epistemic look
    //    `aff_door_look_door_2` (a remembered door): the passage's first step is that LOOK; when the door is re-born it goes
    //    on to approach (and the rest as today). The look withdrawn without the door (RULED OUT) fails the passage.
    {
        auto G = make_graph(9);
        DoorPassage p;
        p.set_clock(clock);
        offered(*G, "aff_door_look_door_2");
        CHECK(DoorPassage::look_available(*G, "door_2"));
        p.start("door_2", Step::Cross, false, /*look_first=*/true);
        auto out = p.update(*G, nullptr, "room_1");
        CHECK(p.view().step == Step::Look);
        CHECK(out.required == "aff_door_look_door_2");          // the selector may take only the look
        executing(*G, "aff_door_look_door_2");
        out = p.update(*G, nullptr, "room_1");
        CHECK(p.view().step == Step::Look and out.required == "aff_door_look_door_2");
        // the door is re-born: its node appears (and door_concept withdraws the look)
        DSR::Node door = DSR::Node::create<door_node_type>("door_2");
        G->insert_node(door);
        offered(*G, "aff_door_2_approach");
        out = p.update(*G, nullptr, "room_1");
        CHECK(p.view().step == Step::Approach);
        CHECK(out.event.has_value() and out.event->find("re-born") != std::string::npos);
        out = p.update(*G, nullptr, "room_1");
        CHECK(out.required == "aff_door_2_approach");
        CHECK(not p.view().failed);
    }
    {   // the look RULED OUT: withdrawn (node gone) and the door never appears — the passage fails, the mission sees it
        auto G = make_graph(10);
        DoorPassage p;
        p.set_clock(clock);
        offered(*G, "aff_door_look_door_3");
        p.start("door_3", Step::Cross, false, /*look_first=*/true);
        p.update(*G, nullptr, "room_1");
        CHECK(p.view().step == Step::Look);
        G->delete_node("aff_door_look_door_3");                 // door_concept: RULED OUT this visit
        const auto out = p.update(*G, nullptr, "room_1");
        CHECK(p.view().failed);
        CHECK(p.view().last_outcome == "look_ruled_out");
        CHECK(out.event.has_value() and out.event->find("not in the graph") != std::string::npos);
        CHECK(not p.active());
    }
    {   // never remembered: no look offered and no door — nothing to start from
        auto G = make_graph(11);
        CHECK(not DoorPassage::look_available(*G, "door_4"));
    }


    // ── Live fix 4 ruling 3: an `open` PRE-EMPTED by `cross` whose door is MEASURED open (door_open_prob known and > 0.5)
    //    counts as opened — no cross -> open -> cross flip when the released open is re-offered (tourP8). Unmeasured
    //    (door_open_prob -1): the existing "the door is shut: back to open" rule stands (control).
    for (const float p_open : {0.92f, -1.0f})
    {
        auto G = make_graph(p_open > 0.f ? 12 : 13);
        DoorPassage p;
        p.set_clock(clock);
        DSR::Node door = DSR::Node::create<door_node_type>("door_5");
        G->add_or_modify_attrib_local<door_open_prob_att>(door, p_open);
        G->insert_node(door);
        p.start("door_5", Step::Cross);
        offered(*G, "aff_door_5_open");                      // skip approach: open on offer
        p.update(*G, nullptr, "room_1");
        executing(*G, "aff_door_5_open");
        p.update(*G, nullptr, "room_1");
        CHECK(p.view().step == Step::Open);
        offered(*G, "aff_door_5_cross");                     // cross pre-empts the executing open
        p.update(*G, nullptr, "room_1");
        CHECK(p.view().step == Step::Cross);
        offered(*G, "aff_door_5_open");                      // the released open is Offered again for a cycle
        p.update(*G, nullptr, "room_1");
        if (p_open > 0.f) CHECK(p.view().step == Step::Cross and p.view().opened);
        else CHECK(p.view().step == Step::Open);
    }
    // ── Live fix 4 ruling 4: reposition() — the session hands a stale-anchor cross back and the passage returns to approach
    {
        auto G = make_graph(14);
        DoorPassage p;
        p.set_clock(clock);
        DSR::Node door = DSR::Node::create<door_node_type>("door_6");
        G->insert_node(door);
        p.start("door_6", Step::Cross);
        offered(*G, "aff_door_6_cross");
        p.update(*G, nullptr, "room_1");
        executing(*G, "aff_door_6_cross");
        p.update(*G, nullptr, "room_1");
        CHECK(p.view().step == Step::Cross);
        p.reposition("the anchor stayed unpublished from here");
        CHECK(p.view().step == Step::Approach and not p.view().reached_door);
        put(*G, "aff_door_6_cross", false, false, rc::affordance::Outcome::Unreachable);   // the hand-back lands
        offered(*G, "aff_door_6_approach");
        const auto out = p.update(*G, nullptr, "room_1");
        CHECK(out.required == "aff_door_6_approach" and p.view().attempts == 0 and not p.view().failed);
    }

    // ── Task 3e (5), rtD1h leg 5: `open` WITHDRAWN BECAUSE ITS GOAL HOLDS. After the approach was satisfied the passage
    //    moved to `open`; door_concept had withdrawn aff_door_1_open (P(in band)·P(shut) = 0: the door was open) and the
    //    passage waited for it 7 min. An `open` step not on offer while the door is MEASURED open (door_open_prob known and
    //    > 1/2 — the evidence ruling 3 uses) is Satisfied: opened, and on to `cross`. Unmeasured (-1) or measured shut: the
    //    wait stands (control).
    for (const float p_open : {0.97f, -1.0f, 0.2f})
    {
        auto G = make_graph(p_open > 0.9f ? 15 : p_open < 0.f ? 16 : 17);
        DoorPassage p;
        p.set_clock(clock);
        DSR::Node door = DSR::Node::create<door_node_type>("door_8");
        G->add_or_modify_attrib_local<door_open_prob_att>(door, p_open);
        G->insert_node(door);
        p.start("door_8", Step::Cross);
        offered(*G, "aff_door_8_approach");
        p.update(*G, nullptr, "room_1");
        executing(*G, "aff_door_8_approach");
        p.update(*G, nullptr, "room_1");
        satisfied(*G, "aff_door_8_approach");
        p.update(*G, nullptr, "room_1");                      // approach SATISFIED -> open (not on offer: never published)
        offered(*G, "aff_door_8_approach");                   // door_concept re-offers approach; `cross` not (yet) on offer
        const auto out = p.update(*G, nullptr, "room_1");
        if (p_open > 0.5f)
        {
            CHECK(p.view().step == Step::Cross and p.view().reached_door);
            CHECK(out.required == "aff_door_8_cross");
            offered(*G, "aff_door_8_cross");                  // and it is taken as soon as it is offered
            CHECK(p.update(*G, nullptr, "room_1").required == "aff_door_8_cross");
        }
        else
            CHECK(p.view().step == Step::Open);
    }
    // ── Task 3e fix round 1 (I-3): the measured-open skip is NOT `open` succeeding — it must not latch opened_. A later re-offer
    //    of `open` is door_concept's own MAP that the door is shut (it may have closed): the passage goes back to `open`, it does
    //    not wait at the door for a `cross` that cannot come.
    {
        auto G = make_graph(19);
        DoorPassage p;
        p.set_clock(clock);
        DSR::Node door = DSR::Node::create<door_node_type>("door_2");
        G->add_or_modify_attrib_local<door_open_prob_att>(door, 0.95f);
        G->insert_node(door);
        p.start("door_2", Step::Cross);
        offered(*G, "aff_door_2_approach");
        p.update(*G, nullptr, "room_1");
        executing(*G, "aff_door_2_approach");
        p.update(*G, nullptr, "room_1");
        satisfied(*G, "aff_door_2_approach");
        p.update(*G, nullptr, "room_1");                      // -> open
        offered(*G, "aff_door_2_approach");
        p.update(*G, nullptr, "room_1");                      // open not offered, door measured open -> cross
        CHECK(p.view().step == Step::Cross and not p.view().opened);
        p.update(*G, nullptr, "room_1");                      // a cycle on cross (its room latched)
        if (auto dn = G->get_node("door_2"); dn.has_value())  // the door is pushed shut: door_concept measures it and offers `open`
        { G->add_or_modify_attrib_local<door_open_prob_att>(*dn, 0.05f); G->update_node(*dn); }
        offered(*G, "aff_door_2_open");
        const auto out = p.update(*G, nullptr, "room_1");
        CHECK(p.view().step == Step::Open and out.required == "aff_door_2_open");
    }
    // ── M-4: the measured-open evidence must come from a LIVE owner. The door node carries no stamp, so liveness is read off the
    //    owner's standing offers for this door (approach / cross offered or executing): none ⇒ door_open_prob may be a dead
    //    agent's last word, and the step waits as before.
    {
        auto G = make_graph(20);
        DoorPassage p;
        p.set_clock(clock);
        DSR::Node door = DSR::Node::create<door_node_type>("door_4");
        G->add_or_modify_attrib_local<door_open_prob_att>(door, 0.95f);
        G->insert_node(door);
        p.start("door_4", Step::Cross);
        offered(*G, "aff_door_4_approach");
        p.update(*G, nullptr, "room_1");
        executing(*G, "aff_door_4_approach");
        p.update(*G, nullptr, "room_1");
        satisfied(*G, "aff_door_4_approach");                 // completed, and nothing re-offered: no standing offer
        p.update(*G, nullptr, "room_1");
        p.update(*G, nullptr, "room_1");
        CHECK(p.view().step == Step::Open);
    }
    // …and an `open` that IS on offer is a shut-door belief: it is taken, never skipped by the measured-open rule
    {
        auto G = make_graph(18);
        DoorPassage p;
        p.set_clock(clock);
        DSR::Node door = DSR::Node::create<door_node_type>("door_3");
        G->add_or_modify_attrib_local<door_open_prob_att>(door, 0.9f);
        G->insert_node(door);
        p.start("door_3", Step::Cross);
        offered(*G, "aff_door_3_open");
        const auto out = p.update(*G, nullptr, "room_1");
        CHECK(p.view().step == Step::Open and out.required == "aff_door_3_open");
    }

    // ── Task E: a poor door estimate makes the EPISTEMIC look win over the goal-directed step — on one scale (nats) ───────
    {
        const auto gain = [&](DSR::DSRGraph& G, const std::string& n, float g, float x, float y)
        {
            auto nd = G.get_node(n);
            G.add_or_modify_attrib_local<epistemic_gain_att>(*nd, g);
            G.add_or_modify_attrib_local<epistemic_target_x_m_att>(*nd, x);
            G.add_or_modify_attrib_local<epistemic_target_y_m_att>(*nd, y);
            G.update_node(*nd);
        };
        // the pure comparison
        using C = DoorPassage;
        CHECK(C::choose_look_or_step(0.40f, 3.f, 0.15f, 3.f, 0.2f, 0.5f, false).look == false);   // good belief: the step
        CHECK(C::choose_look_or_step(0.20f, 3.f, 1.80f, 3.f, 0.2f, 0.5f, false).look == true);    // degraded: the look
        CHECK(C::choose_look_or_step(std::nullopt, 0.f, 1.00f, 3.f, 0.2f, 0.5f, false).look == true);   // step withdrawn: the look
        CHECK(C::choose_look_or_step(std::nullopt, 0.f, 0.10f, 3.f, 0.2f, 0.5f, false).look == false);   // look worth less than the trip: stay
        CHECK(C::choose_look_or_step(std::nullopt, 0.f, std::nullopt, 0.f, 0.2f, 0.5f, false).look == false);
        CHECK(C::choose_look_or_step(0.45f, 3.f, 0.50f, 3.f, 0.2f, 0.5f, true).look == true);     // incumbent look keeps the margin
        CHECK(C::choose_look_or_step(0.45f, 3.f, 0.50f, 3.f, 0.2f, 0.5f, false).look == false);    // the step is the incumbent
        // through the graph
        auto G = make_graph(30);
        DoorPassage p;
        p.set_clock(clock);
        p.set_efe(0.2f, 0.5f); p.set_look_enabled(true);
        p.note_robot(Eigen::Vector2f(0.f, 0.f), 0.25f);
        p.start("door_5", Step::Cross);
        offered(*G, "aff_door_5_approach"); place(*G, "aff_door_5_approach", 3.f, 0.f);
        offered(*G, "aff_door_5");          place(*G, "aff_door_5", 2.f, 0.f);
        gain(*G, "aff_door_5_approach", 0.40f, 3.f, 0.f);
        gain(*G, "aff_door_5", 0.12f, 2.f, 0.f);                       // good estimate: little to learn
        auto out = p.update(*G, nullptr, "room_1");
        CHECK(out.required == "aff_door_5_approach");
        gain(*G, "aff_door_5_approach", 0.17f, 3.f, 0.f);             // the estimate degrades: P(complete) falls...
        gain(*G, "aff_door_5", 1.80f, 2.f, 0.f);                      // ...and the information to be had rises
        out = p.update(*G, nullptr, "room_1");
        CHECK(out.required == "aff_door_5");                          // the repositioning came out of the comparison
        CHECK(out.event.has_value() and out.event->find("epistemic look") != std::string::npos);
        CHECK(p.view().step == Step::Approach);                      // the passage is still at approach
        executing(*G, "aff_door_5");
        out = p.update(*G, nullptr, "room_1");
        CHECK(out.required == "aff_door_5");
        satisfied(*G, "aff_door_5");                                  // the look is done: its gain falls with the uncertainty
        gain(*G, "aff_door_5", 0.f, 2.f, 0.f);
        out = p.update(*G, nullptr, "room_1");
        CHECK(out.required == "aff_door_5_approach");                 // back to the goal-directed step
        // the approach is WITHDRAWN (the ab7 dead end) and the look is on offer: no waiting
        auto G2 = make_graph(31);
        DoorPassage q;
        q.set_clock(clock);
        q.set_efe(0.2f, 0.5f); q.set_look_enabled(true);
        q.note_robot(Eigen::Vector2f(0.f, 0.f), 0.25f);
        q.start("door_6", Step::Cross);
        offered(*G2, "aff_door_6"); place(*G2, "aff_door_6", 2.f, 0.f); gain(*G2, "aff_door_6", 1.5f, 2.f, 0.f);
        out = q.update(*G2, nullptr, "room_1");
        CHECK(out.required == "aff_door_6");
        // GATE: the old chain ([Policy] off) never takes the look, whatever the prices
        {
            auto Gg = make_graph(33);
            DoorPassage g;
            g.set_clock(clock); g.set_efe(0.2f, 0.5f); g.note_robot(Eigen::Vector2f(0.f, 0.f), 0.25f);
            g.start("door_4", Step::Cross);
            offered(*Gg, "aff_door_4_approach"); place(*Gg, "aff_door_4_approach", 3.f, 0.f); gain(*Gg, "aff_door_4_approach", 0.1f, 3.f, 0.f);
            offered(*Gg, "aff_door_4");          place(*Gg, "aff_door_4", 2.f, 0.f);          gain(*Gg, "aff_door_4", 5.0f, 2.f, 0.f);
            CHECK(g.update(*Gg, nullptr, "room_1").required == "aff_door_4_approach");
        }
        // THE LOOP (persistent off-template width): the producer's look value is the information a view can still RESOLVE, so it
        // is spent once the look has been done; the passage must reach Cross in a bounded number of cycles, taking the look once.
        {
            auto GL = make_graph(34);
            DoorPassage l;
            l.set_clock(clock); l.set_efe(0.2f, 0.5f); l.set_look_enabled(true);
            l.start("door_3", Step::Cross);
            offered(*GL, "aff_door_3_approach"); offered(*GL, "aff_door_3");
            float look_gain = 1.8f;
            Eigen::Vector2f robot(0.f, 0.f);
            int looks_taken = 0, cycles = 0;
            bool reached_cross = false;
            for (; cycles < 60 and not reached_cross; ++cycles)
            {
                l.note_robot(robot, 0.25f);
                gain(*GL, "aff_door_3_approach", 0.4f, 3.f, 0.f);
                gain(*GL, "aff_door_3", look_gain, 6.f, 0.f);          // the look's viewpoint 6 m out
                const auto o = l.update(*GL, nullptr, "room_1");
                if (o.required == "aff_door_3")
                {
                    executing(*GL, "aff_door_3");
                    robot = Eigen::Vector2f(6.f, 0.f);                 // drives out to the look viewpoint
                    l.update(*GL, nullptr, "room_1");
                    satisfied(*GL, "aff_door_3");                      // looked: done
                    ++looks_taken;
                    look_gain = 0.f;                                   // ...and what it could resolve is resolved
                    l.update(*GL, nullptr, "room_1");
                    offered(*GL, "aff_door_3_approach");
                }
                else if (o.required == "aff_door_3_approach")
                {
                    executing(*GL, "aff_door_3_approach");
                    robot = Eigen::Vector2f(3.f, 0.f);
                    l.update(*GL, nullptr, "room_1");
                    satisfied(*GL, "aff_door_3_approach");
                    l.update(*GL, nullptr, "room_1");
                    offered(*GL, "aff_door_3_cross");
                    reached_cross = l.update(*GL, nullptr, "room_1").required == "aff_door_3_cross";
                }
            }
            std::println("    loop test: {} look(s), Cross reached after {} cycles", looks_taken, cycles);
            CHECK(reached_cross and cycles <= 6 and looks_taken <= 1);
        }
        // nothing worth looking at, approach withdrawn: the old wait
        auto G3 = make_graph(32);
        DoorPassage r;
        r.set_clock(clock);
        r.set_efe(0.2f, 0.5f); r.set_look_enabled(true); r.note_robot(Eigen::Vector2f(0.f, 0.f), 0.25f);
        r.start("door_8", Step::Cross);
        out = r.update(*G3, nullptr, "room_1");
        CHECK(out.required == "aff_door_8_approach");
    }

    std::println("test_door_passage: {} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
