#include "../affordance_manager.h"

#include <dsr/api/dsr_api.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string>

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

    controller.mark_reached(graph);

    expect_protocol_state(graph, "table_concept_affordance", false, false, failure);
    const auto second = controller.select_target(graph);
    failure.expect(second.has_value(), "controller did not continue with remaining offered affordance");
    if (second.has_value())
        failure.expect(second->node_name == "room_concept_affordance", "controller did not pick remaining room affordance second");

    expect_protocol_state(graph, "room_concept_affordance", true, true, failure);
    controller.mark_reached(graph);
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
    controller.mark_reached(graph);
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

    controller.mark_reached(graph);
    room_producer.monitor_execution(graph);
    failure.expect(room_producer.consume_completion_event(), "producer did not observe completion event after controller finished affordance");
    failure.expect(!room_producer.consume_completion_event(), "completion event should be consumable only once");
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