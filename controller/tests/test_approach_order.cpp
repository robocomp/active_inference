// ApproachOrder (controller_approach_order.h): `approach <object> [from <side>]` — require the object's ONE approach
// affordance (and the chosen option), end when its producer completes it, and say how the base is left.
// Build+run (a real DSR graph on a private DDS domain, like test_door_passage.cpp):
//   g++ -std=c++23 -O0 -g -fPIC -I/usr/include/eigen3 $(pkg-config --cflags Qt6Core) \
//       -DSHADOW_JSON='"'$PWD/../../robot_concept/shadow.json'"' -o /tmp/tao test_approach_order.cpp ../src/chat_inbox.cpp \
//       -ldsr_api -ldsr_core -lfastcdr -lfastdds $(pkg-config --libs Qt6Core) && /tmp/tao
#include "../src/controller_approach_order.h"
#include "../src/chat_inbox.h"

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

using rc::ApproachOrder;

std::shared_ptr<DSR::DSRGraph> make_graph(int k)
{
    return std::make_shared<DSR::DSRGraph>("approach_order_test_" + std::to_string(k), 990 + k,
                                           std::string(SHADOW_JSON), true, static_cast<int8_t>(80 + k), DSR::Queue);
}

/// The object's approach node in a protocol state, listing `options`.
void put(DSR::DSRGraph& G, const std::string& name, bool pending, bool active, const std::string& options = "xn,yp",
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
    G.add_or_modify_attrib_local<aff_option_names_att>(*n, options);
    G.add_or_modify_attrib_local<epistemic_pending_att>(*n, pending);
    G.add_or_modify_attrib_local<active_att>(*n, active);
    G.update_node(*n);
}
void offered(DSR::DSRGraph& G, const std::string& n, const std::string& o = "xn,yp") { put(G, n, true, false, o); }
void executing(DSR::DSRGraph& G, const std::string& n) { put(G, n, true, true); }
void ended(DSR::DSRGraph& G, const std::string& n, rc::affordance::Outcome o) { put(G, n, false, false, "xn,yp", o); }
}   // namespace

int main()
{
    // ── what may be asked: the object must offer approach, and the side must be one it lists ────────────────────
    {
        auto G = make_graph(0);
        CHECK(ApproachOrder::check(*G, "table_1", "").has_value());                       // no approach node
        CHECK(ApproachOrder::check(*G, "table_1", "")->find("table_1") != std::string::npos);
        offered(*G, "aff_table_1_approach");
        CHECK(not ApproachOrder::check(*G, "table_1", "").has_value());
        CHECK(not ApproachOrder::check(*G, "table_1", "yp").has_value());
        const auto r = ApproachOrder::check(*G, "table_1", "xp");                        // against the wall: not listed
        CHECK(r.has_value() and r->find("xn") != std::string::npos and r->find("yp") != std::string::npos);
    }

    // ── a single command: require the node + option, done on SATISFIED, and the base is held ─────────────────────
    {
        auto G = make_graph(1);
        offered(*G, "aff_table_1_approach");
        ApproachOrder o;
        o.start("table_1", "yp", ApproachOrder::After::Hold);
        CHECK(o.active());
        CHECK(o.take_just_started());
        CHECK(not o.take_just_started());
        auto out = o.update(*G);
        CHECK(out.required == "aff_table_1_approach" and out.option == "yp");
        executing(*G, "aff_table_1_approach");
        out = o.update(*G);
        CHECK(out.required == "aff_table_1_approach");
        CHECK(not o.take_finished());
        ended(*G, "aff_table_1_approach", rc::affordance::Outcome::Satisfied);
        out = o.update(*G);
        CHECK(out.required.empty());
        CHECK(not o.active());
        CHECK(o.take_finished());
        CHECK(not o.take_finished());
        CHECK(o.view().after == ApproachOrder::After::Hold);
        CHECK(o.view().outcome == "satisfied");
        CHECK(out.event.has_value() and out.event->find("table_1") != std::string::npos);
    }

    // ── a completed node seen BEFORE our claim is someone else's completion, not ours ────────────────────────────
    {
        auto G = make_graph(2);
        ended(*G, "aff_chair_1_approach", rc::affordance::Outcome::Satisfied);   // left over from an earlier approach
        ApproachOrder o;
        o.start("chair_1", "", ApproachOrder::After::Return);
        o.update(*G);
        CHECK(o.active());
        CHECK(not o.take_finished());
        offered(*G, "aff_chair_1_approach");
        CHECK(o.update(*G).option.empty());   // no side: the node's default
    }

    // ── a failed approach ends the order too (the robot is held, not sent to try for ever) ───────────────────────
    {
        auto G = make_graph(3);
        offered(*G, "aff_table_2_approach");
        ApproachOrder o;
        o.start("table_2", "", ApproachOrder::After::Hold);
        o.update(*G);
        executing(*G, "aff_table_2_approach");
        o.update(*G);
        ended(*G, "aff_table_2_approach", rc::affordance::Outcome::Refused);
        const auto out = o.update(*G);
        CHECK(not o.active() and o.take_finished());
        CHECK(o.view().outcome == "refused");
        CHECK(out.event.has_value() and out.event->find("refused") != std::string::npos);
    }

    // ── the object's approach disappears under the claim (the object was removed): the order ends ────────────────
    {
        auto G = make_graph(4);
        offered(*G, "aff_table_3_approach");
        ApproachOrder o;
        o.start("table_3", "", ApproachOrder::After::Hold);
        o.update(*G);
        executing(*G, "aff_table_3_approach");
        o.update(*G);
        G->delete_node(G->get_node("aff_table_3_approach")->id());
        const auto out = o.update(*G);
        CHECK(not o.active() and o.take_finished());
        CHECK(out.event.has_value() and out.event->find("gone") != std::string::npos);
        CHECK(o.view().outcome == "gone");
    }

    // ── stop(): no longer active, and not a finish (nothing to hand back) ────────────────────────────────────────
    {
        auto G = make_graph(5);
        offered(*G, "aff_table_4_approach");
        ApproachOrder o;
        o.start("table_4", "", ApproachOrder::After::Hold);
        o.update(*G);
        o.stop();
        CHECK(not o.active());
        CHECK(not o.take_finished());
        CHECK(o.update(*G).required.empty());
    }

    // ── every start is a new order (its id), so a status report can tell "superseded" from "done" ───────────────
    {
        ApproachOrder o;
        o.start("table_1", "", ApproachOrder::After::Hold);
        const auto a = o.view().id;
        o.start("table_1", "", ApproachOrder::After::Hold);
        CHECK(o.view().id != a and a != 0);
    }

    // ── what chat is told (ChatInbox::status_of_approach) ────────────────────────────────────────────────────────
    {
        using T = rc::ChatInbox::Tracked;
        const T t{.kind = T::Kind::Approach, .approach_id = 7};
        ApproachOrder::View v{.object = "table_1", .id = 7, .active = true, .state = "executing"};
        auto [st, bt] = rc::ChatInbox::status_of_approach(t, v, true);
        CHECK(st == "in_progress" and bt.find("executing") != std::string::npos and bt.ends_with("driving=on"));
        v.active = false;
        v.outcome = "satisfied";
        CHECK(rc::ChatInbox::status_of_approach(t, v, true).first == "completed");
        v.outcome = "unreachable";
        CHECK(rc::ChatInbox::status_of_approach(t, v, true).first == "failed|unreachable");
        v.outcome.clear();   // stopped
        CHECK(rc::ChatInbox::status_of_approach(t, v, false).first == "aborted");
        v.id = 8;            // another approach was started
        CHECK(rc::ChatInbox::status_of_approach(t, v, true).first == "aborted");
    }

    std::println("test_approach_order: {} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
