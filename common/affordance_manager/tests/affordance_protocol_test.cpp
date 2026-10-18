#include "../affordance_manager.h"

#include <dsr/api/dsr_api.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace
{
struct TestFailure
{
    bool failed = false;

    void expect(bool condition, const std::string &message)
    {
        if (condition)
            return;

        failed = true;
        std::cerr << "[FAIL] " << message << '\n';
    }
};

struct GraphFixture
{
    std::shared_ptr<DSR::DSRGraph> graph;
    std::uint64_t room_id = 0;
};

GraphFixture create_graph_fixture()
{
    GraphFixture fixture;
    static int fixture_index = 0;
    const auto graph_name = "affordance_protocol_test_" + std::to_string(fixture_index);
    const auto agent_id = 919 + fixture_index;
    const auto domain_id = static_cast<int8_t>(40 + fixture_index);
    ++fixture_index;

    fixture.graph = std::make_shared<DSR::DSRGraph>(graph_name,
                                                    agent_id,
                                                    std::string(AFFORDANCE_TEST_SHADOW_JSON),
                                                    true,
                                                    domain_id,
                                                    DSR::Queue);

    DSR::Node room = DSR::Node::create<room_node_type>(graph_name + "_room");
    fixture.graph->add_or_modify_attrib_local<level_att>(room, 1);
    const auto room_id = fixture.graph->insert_node(room);
    if (!room_id.has_value())
        throw std::runtime_error("could not insert test room node");
    fixture.room_id = room_id.value();
    return fixture;
}

std::optional<DSR::Node> get_required_node(const std::shared_ptr<DSR::DSRGraph> &graph, const std::string &name, TestFailure &failure)
{
    const auto node_opt = graph->get_node(name);
    failure.expect(node_opt.has_value(), "missing node '" + name + "'");
    return node_opt;
}

void expect_protocol_state(const std::shared_ptr<DSR::DSRGraph> &graph,
                           const std::string &name,
                           bool expected_active,
                           bool expected_pending,
                           TestFailure &failure)
{
    const auto node_opt = get_required_node(graph, name, failure);
    if (!node_opt.has_value())
        return;

    const bool active = graph->get_attrib_by_name<active_att>(node_opt.value()).value_or(false);
    const bool pending = graph->get_attrib_by_name<epistemic_pending_att>(node_opt.value()).value_or(false);
    failure.expect(active == expected_active,
                   "unexpected active flag for '" + name + "': expected=" + std::to_string(expected_active) + " actual=" + std::to_string(active));
    failure.expect(pending == expected_pending,
                   "unexpected pending flag for '" + name + "': expected=" + std::to_string(expected_pending) + " actual=" + std::to_string(pending));
}

void test_multi_producer_claim_order(TestFailure &failure)
{
    auto [graph, room_id] = create_graph_fixture();

    rc::AffordanceManager room_producer{"room_concept_affordance"};
    rc::AffordanceManager table_producer{"table_concept_affordance"};
    rc::AffordanceManager controller;

    failure.expect(room_producer.publish_target(graph, room_id, 1.f, 1.f, 0.1f, 0.4f), "room producer failed to publish target");
    failure.expect(table_producer.publish_target(graph, room_id, 2.f, 2.f, 0.2f, 0.9f), "table producer failed to publish target");

    expect_protocol_state(graph, "room_concept_affordance", false, true, failure);
    expect_protocol_state(graph, "table_concept_affordance", false, true, failure);

    const auto first = controller.select_target(graph);
    failure.expect(first.has_value(), "controller did not select any affordance");
    if (first.has_value())
        failure.expect(first->node_name == "table_concept_affordance", "controller did not pick highest-gain affordance first");

    expect_protocol_state(graph, "table_concept_affordance", true, true, failure);
    expect_protocol_state(graph, "room_concept_affordance", false, true, failure);
    failure.expect(table_producer.is_executing(graph), "table affordance should be executing after claim");
    failure.expect(!room_producer.is_executing(graph), "room affordance should remain offered while table is executing");

    controller.mark_reached(graph, rc::affordance::Outcome::Satisfied);

    expect_protocol_state(graph, "table_concept_affordance", false, false, failure);
    const auto second = controller.select_target(graph);
    failure.expect(second.has_value(), "controller did not continue with remaining offered affordance");
    if (second.has_value())
        failure.expect(second->node_name == "room_concept_affordance", "controller did not pick remaining room affordance second");

    expect_protocol_state(graph, "room_concept_affordance", true, true, failure);
    controller.mark_reached(graph, rc::affordance::Outcome::Satisfied);
    expect_protocol_state(graph, "room_concept_affordance", false, false, failure);
}

void test_republish_same_node_after_completion(TestFailure &failure)
{
    auto [graph, room_id] = create_graph_fixture();

    rc::AffordanceManager room_producer{"room_concept_affordance"};
    rc::AffordanceManager controller;

    failure.expect(room_producer.publish_target(graph, room_id, 1.f, 1.f, 0.0f, 0.3f), "initial affordance publish failed");
    const auto first = controller.select_target(graph);
    failure.expect(first.has_value(), "controller failed to claim first affordance");
    controller.mark_reached(graph, rc::affordance::Outcome::Satisfied);
    expect_protocol_state(graph, "room_concept_affordance", false, false, failure);

    failure.expect(room_producer.publish_target(graph, room_id, 4.f, 5.f, 0.7f, 0.8f), "republish on same affordance node failed");
    expect_protocol_state(graph, "room_concept_affordance", false, true, failure);

    const auto second = controller.select_target(graph);
    failure.expect(second.has_value(), "controller failed to reclaim republished affordance");
    if (second.has_value())
    {
        failure.expect(second->node_name == "room_concept_affordance", "controller selected wrong affordance on republish");
        failure.expect(std::abs(second->room_pos.x() - 4.f) < 1e-4f && std::abs(second->room_pos.y() - 5.f) < 1e-4f,
                       "controller did not observe updated target coordinates after republish");
        failure.expect(std::abs(second->yaw_rad - 0.7f) < 1e-4f, "controller did not observe updated yaw after republish");
        failure.expect(std::abs(second->epistemic_gain - 0.8f) < 1e-4f, "controller did not observe updated gain after republish");
    }
}

void test_invalid_state_is_not_claimed(TestFailure &failure)
{
    auto [graph, room_id] = create_graph_fixture();

    rc::AffordanceManager room_producer{"room_concept_affordance"};
    rc::AffordanceManager controller;

    failure.expect(room_producer.publish_target(graph, room_id, 1.f, 2.f, 0.3f, 0.5f), "producer failed to publish affordance for invalid-state test");

    const auto node_opt = get_required_node(graph, "room_concept_affordance", failure);
    if (!node_opt.has_value())
        return;

    auto node = node_opt.value();
    graph->add_or_modify_attrib_local<active_att>(node, true);
    graph->add_or_modify_attrib_local<epistemic_pending_att>(node, false);
    graph->update_node(node);

    const auto selection = controller.select_target(graph);
    failure.expect(!selection.has_value(), "controller should ignore invalid affordance state active=true,pending=false");
    failure.expect(!room_producer.is_executing(graph), "producer should not report invalid state as executing");

    failure.expect(room_producer.publish_target(graph, room_id, 6.f, 7.f, 0.9f, 0.6f), "producer failed to recover from invalid state by republishing");
    const auto recovered = controller.select_target(graph);
    failure.expect(recovered.has_value(), "controller failed to claim affordance after valid republish");
}

void test_completion_event_is_consumed_once(TestFailure &failure)
{
    auto [graph, room_id] = create_graph_fixture();

    rc::AffordanceManager room_producer{"room_concept_affordance"};
    rc::AffordanceManager controller;

    failure.expect(room_producer.publish_target(graph, room_id, 1.f, 1.f, 0.1f, 0.4f), "producer failed to publish affordance for completion-event test");
    const auto first = controller.select_target(graph);
    failure.expect(first.has_value(), "controller failed to claim affordance for completion-event test");

    room_producer.monitor_execution(graph);
    failure.expect(!room_producer.consume_completion_event(), "completion event should not be raised while affordance is still executing");

    controller.mark_reached(graph, rc::affordance::Outcome::Satisfied);
    room_producer.monitor_execution(graph);
    failure.expect(room_producer.consume_completion_event(), "producer did not observe completion event after controller finished affordance");
    failure.expect(!room_producer.consume_completion_event(), "completion event should be consumable only once");
}

// The claim observer (the controller's episodic journal): Start once per claimed (affordance, epoch), End
// "preempted" when the claim moves to another affordance or is released, End <outcome> on mark_reached.
void claim_observer_start_preempt_outcome(TestFailure &failure)
{
    auto [graph, room_id] = create_graph_fixture();
    // publish_executing writes from OUR agent node; a bare test graph may not have one yet.
    if (not rc::AffordanceManager::own_agent_node_id(graph).has_value())
    {
        DSR::Node agent = DSR::Node::create<agent_node_type>("claim_observer_test_agent");
        graph->add_or_modify_attrib_local<agent_id_att>(agent, static_cast<std::uint32_t>(graph->get_agent_id()));
        graph->add_or_modify_attrib_local<level_att>(agent, 0);
        failure.expect(graph->insert_node(agent).has_value(), "could not insert the test agent node");
    }
    rc::AffordanceManager a_producer{"aff_table_epistemic"};
    rc::AffordanceManager b_producer{"aff_door_cross"};
    rc::AffordanceManager controller;
    failure.expect(a_producer.publish_target(graph, room_id, 1.f, 1.f, 0.f, 0.4f), "producer A failed to publish");
    failure.expect(b_producer.publish_target(graph, room_id, 2.f, 2.f, 0.f, 0.9f), "producer B failed to publish");
    const auto a = graph->get_node("aff_table_epistemic");
    const auto b = graph->get_node("aff_door_cross");
    failure.expect(a.has_value() and b.has_value(), "affordance nodes missing");
    if (not a.has_value() or not b.has_value()) return;
    {   // B carries a pragmatic kind; A has none (the epistemic look-at-it offer)
        auto n = b.value();
        graph->add_or_modify_attrib_local<aff_kind_att>(n, std::string("cross"));
        graph->update_node(n);
    }

    std::vector<rc::AffordanceManager::ClaimEvent> events;
    controller.set_claim_observer([&events](const rc::AffordanceManager::ClaimEvent &e) { events.push_back(e); });
    using K = rc::AffordanceManager::ClaimEvent::Kind;

    failure.expect(controller.publish_executing(graph, a->id(), 3, 1.f, 1.f, 0.f), "claim A failed");
    failure.expect(controller.publish_executing(graph, a->id(), 3, 1.1f, 1.f, 0.f), "re-claim A (moved pose) failed");
    failure.expect(events.size() == 1, "same (affordance, epoch) must fire one Start, got " + std::to_string(events.size()));
    if (events.size() >= 1)
    {
        const auto &e = events[0];
        failure.expect(e.kind == K::Start and e.affordance == "aff_table_epistemic" and e.epoch == 3,
                       "first event must be Start A epoch 3");
        failure.expect(std::abs(e.gain - 0.4f) < 1e-5f, "Start A must quote the advertised gain");
        failure.expect(e.kind_label == "epistemic", "a node with no aff_kind is the epistemic offer, got '" + e.kind_label + "'");
        failure.expect(e.target == graph->get_node(room_id)->name() and e.target_type == "room",
                       "target must be the affordance's parent, got '" + e.target + "'/'" + e.target_type + "'");
        failure.expect(not e.producer.empty(), "producer must name the agent that created the node");
        failure.expect(e.contract.find("\"policy\":\"reach\"") != std::string::npos, "contract JSON must carry the policy: " + e.contract);
    }

    failure.expect(controller.publish_executing(graph, b->id(), 7, 2.f, 2.f, 0.f), "claim B failed");
    failure.expect(events.size() == 3, "switching claims must fire End A + Start B, got " + std::to_string(events.size()));
    if (events.size() == 3)
    {
        failure.expect(events[1].kind == K::End and events[1].affordance == "aff_table_epistemic"
                       and events[1].outcome == "preempted", "second event must be End A preempted");
        failure.expect(events[2].kind == K::Start and events[2].affordance == "aff_door_cross" and events[2].epoch == 7
                       and events[2].kind_label == "cross", "third event must be Start B (kind cross)");
    }
    failure.expect(controller.publish_executing(graph, b->id(), 7, 2.f, 2.f, 0.f), "republish B failed");
    failure.expect(events.size() == 3, "republishing the same claim must fire nothing");

    // mark_reached stamps its outcome only on the claim it completes: the selector takes A (B is pragmatic, never
    // taken by default), so claiming it makes A the current affordance.
    const auto sel = controller.select_target(graph);
    failure.expect(sel.has_value() and sel->node_name == "aff_table_epistemic", "the selector should hold A");
    failure.expect(controller.publish_executing(graph, a->id(), 3, 1.f, 1.f, 0.f), "claim the selected A failed");
    failure.expect(events.size() == 5 and events[3].outcome == "preempted" and events[4].kind == K::Start,
                   "claiming A must end B preempted and start A");
    controller.mark_reached(graph, rc::affordance::Outcome::Satisfied);
    failure.expect(events.size() == 6 and events.back().kind == K::End and events.back().affordance == "aff_table_epistemic"
                   and events.back().outcome == "satisfied", "mark_reached must fire End A satisfied");
    controller.mark_reached(graph, rc::affordance::Outcome::Satisfied);
    failure.expect(events.size() == 6, "a second mark_reached with no open claim must fire nothing");

    // An ABANDONED claim (clear_current: a click target or a mission took the base) is preempted, never stamped
    // with the outcome of whatever mark_reached completes next.
    failure.expect(controller.publish_executing(graph, a->id(), 5, 1.f, 1.f, 0.f), "claim A for the abandon case failed");
    controller.clear_current();
    controller.mark_reached(graph, rc::affordance::Outcome::Unreachable);
    failure.expect(events.size() == 8 and events.back().kind == K::End and events.back().affordance == "aff_table_epistemic"
                   and events.back().outcome == "preempted",
                   "mark_reached after clear_current must end the abandoned claim 'preempted', got '"
                   + (events.empty() ? std::string() : events.back().outcome) + "'");

    failure.expect(controller.publish_executing(graph, a->id(), 4, 1.f, 1.f, 0.f), "claim A again failed");
    controller.release_execution_claim(graph);
    failure.expect(events.size() == 10 and events.back().kind == K::End and events.back().outcome == "preempted",
                   "releasing a claim must fire End preempted");

    // An executing edge persisted from a previous run (same affordance, epoch and pose): the new run still reports it.
    failure.expect(controller.publish_executing(graph, a->id(), 6, 1.f, 1.f, 0.f), "claim A epoch 6 failed");
    rc::AffordanceManager next_run;
    std::vector<rc::AffordanceManager::ClaimEvent> next_events;
    next_run.set_claim_observer([&next_events](const rc::AffordanceManager::ClaimEvent &e) { next_events.push_back(e); });
    failure.expect(next_run.publish_executing(graph, a->id(), 6, 1.f, 1.f, 0.f), "next run's re-claim failed");
    failure.expect(next_events.size() == 1 and next_events[0].kind == K::Start and next_events[0].epoch == 6,
                   "an unchanged persisted edge must still fire Start once in a new run");

    // A PRE-ROLLOUT producer (no epoch, e.g. afford_room): the controller drives it without an `executing` edge, and
    // note_claim still reports it — live 2026-10-18, a whole day of afford_room drives was missing from the journal.
    rc::AffordanceManager consumer;
    std::vector<rc::AffordanceManager::ClaimEvent> ev0;
    consumer.set_claim_observer([&ev0](const rc::AffordanceManager::ClaimEvent &e) { ev0.push_back(e); });
    consumer.note_claim(graph, b->id(), 0);
    consumer.note_claim(graph, b->id(), 0);
    failure.expect(ev0.size() == 1 and ev0[0].kind == K::Start and ev0[0].epoch == 0
                   and ev0[0].affordance == "aff_door_cross", "an epoch-0 claim must fire one Start, epoch 0");
    consumer.note_claim(graph, a->id(), 0);
    failure.expect(ev0.size() == 3 and ev0[1].kind == K::End and ev0[1].outcome == "preempted" and ev0[2].kind == K::Start,
                   "moving an epoch-0 claim must end the old one preempted and start the new one");
    consumer.release_execution_claim(graph);
    failure.expect(ev0.size() == 4 and ev0.back().kind == K::End and ev0.back().outcome == "preempted",
                   "releasing an epoch-0 claim must end it");
    consumer.note_claim(graph, 0, 0);
    failure.expect(ev0.size() == 4, "no affordance (id 0) is no claim");
}
// A CONSUMER's release must free the claim it holds (live 2026-10-18: `stop` during a door crossing left
// aff_door_1_cross Executing with nobody driving it; the next `cross door_1` found nothing eligible and stalled).
// release_execution_claim used to look only at the PRODUCER's managed node, which a consumer (the controller,
// constructed with no managed name) does not have — so it released nothing, ever.
void consumer_release_frees_its_claim(TestFailure &failure)
{
    auto [graph, room_id] = create_graph_fixture();
    rc::AffordanceManager producer{"door_cross_affordance"};
    rc::AffordanceManager controller;

    failure.expect(producer.publish_target(graph, room_id, 1.f, 1.f, 0.f, 0.5f), "producer failed to publish");
    failure.expect(controller.select_target(graph).has_value(), "controller did not claim the offered affordance");
    expect_protocol_state(graph, "door_cross_affordance", true, true, failure);   // Executing
    failure.expect(controller.release_execution_claim(graph), "release must report that it freed the claim");
    expect_protocol_state(graph, "door_cross_affordance", false, true, failure);  // Offered again

    // ABANDONED first (clear_current: a click target, a mission or a stop took the base), released after: the
    // claim is still the consumer's to free, not left Executing for the producer to wait out.
    failure.expect(controller.select_target(graph).has_value(), "controller did not re-claim");
    expect_protocol_state(graph, "door_cross_affordance", true, true, failure);
    controller.clear_current();
    failure.expect(controller.release_execution_claim(graph), "release after clear_current must free the abandoned claim");
    expect_protocol_state(graph, "door_cross_affordance", false, true, failure);
    failure.expect(not controller.release_execution_claim(graph), "a second release has nothing left to free");
}
} // namespace

int main()
{
    TestFailure failure;

    try
    {
        test_multi_producer_claim_order(failure);
        test_republish_same_node_after_completion(failure);
        test_invalid_state_is_not_claimed(failure);
        test_completion_event_is_consumed_once(failure);
        claim_observer_start_preempt_outcome(failure);
        consumer_release_frees_its_claim(failure);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[FAIL] exception: " << e.what() << '\n';
        return EXIT_FAILURE;
    }

    if (failure.failed)
        return EXIT_FAILURE;

    std::cout << "affordance protocol tests passed" << std::endl;
    return EXIT_SUCCESS;
}