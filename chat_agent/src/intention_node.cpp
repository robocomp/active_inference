#include "intention_node.h"

#include <print>

#include "../../common/graph_provenance/creation_stamp.h"

namespace rc::chat
{

int IntentionNode::sweep_stale()
{
    int n = 0;
    for (const auto& node : G_->get_nodes_by_type("intention"))
        if (node.name() == kName and node.id() != id_ and G_->delete_node(node.id()))
        {
            ++n;
            std::println("[chat_agent] removed stale '{}' node id={} (a previous run's)", kName, node.id());
        }
    return n;
}

bool IntentionNode::ensure()
{
    if (id_ != 0 and not G_->get_node(id_).has_value())
    {
        std::println("[chat_agent] our '{}' node {} disappeared — recreating it", kName, id_);
        id_ = 0;
        edge_done_ = false;
    }
    if (id_ == 0)
    {
        if (const auto other = G_->get_node(kName); other.has_value())
        {
            G_->delete_node(other->id());
            std::println("[chat_agent] removed stale '{}' node id={} before creating ours", kName, other->id());
        }
        DSR::Node node = DSR::Node::create<intention_node_type>(kName);
        const auto robot = G_->get_node(kRobotNode);
        G_->add_or_modify_attrib_local<level_att>(node, robot.has_value() ? G_->get_node_level(robot.value()).value_or(0) + 1 : 1);
        if (robot.has_value()) G_->add_or_modify_attrib_local<parent_att>(node, robot->id());
        G_->add_or_modify_attrib_local<pos_x_att>(node, 420.f);
        G_->add_or_modify_attrib_local<pos_y_att>(node, -260.f);
        G_->add_or_modify_attrib_local<current_intention_att>(node, std::string{});
        G_->add_or_modify_attrib_local<timestamp_agent_att>(node, std::uint64_t{0});
        rc::provenance::stamp_creation(*G_, node);
        const auto ins = G_->insert_node(node);
        if (not ins.has_value())
        {
            std::println("[chat_agent] could not insert '{}'", kName);
            return false;
        }
        id_ = ins.value();
        std::println("[chat_agent] created '{}' id={}", kName, id_);
    }
    if (not edge_done_)
        if (const auto robot = G_->get_node(kRobotNode); robot.has_value())
        {
            auto edge = DSR::Edge::create<has_intention_edge_type>(robot->id(), id_);
            if (G_->insert_or_assign_edge(edge))
            {
                edge_done_ = true;
                std::println("[chat_agent] has_intention edge {} -> {} in place", kRobotNode, kName);
                if (auto n = G_->get_node(id_); n.has_value())
                {
                    G_->add_or_modify_attrib_local<parent_att>(n.value(), robot->id());
                    G_->update_node(n.value());
                }
            }
        }
    return true;
}

std::optional<std::uint64_t> IntentionNode::send(const std::string& command)
{
    if (not ensure()) return std::nullopt;
    auto node = G_->get_node(id_);
    if (not node.has_value()) return std::nullopt;
    const std::uint64_t stamp = rc::provenance::now_ms();
    G_->add_or_modify_attrib_local<current_intention_att>(node.value(), command);
    G_->add_or_modify_attrib_local<timestamp_agent_att>(node.value(), stamp);
    if (not G_->update_node(node.value())) return std::nullopt;
    return stamp;
}

IntentionStatus IntentionNode::read() const
{
    IntentionStatus s;
    if (id_ == 0) return s;
    const auto node = G_->get_node(id_);
    if (not node.has_value()) return s;
    if (const auto c = G_->get_attrib_by_name<current_intention_att>(node.value()); c.has_value()) s.command = c->get();
    s.stamp = G_->get_attrib_by_name<timestamp_agent_att>(node.value()).value_or(0);
    for (const auto& e : G_->get_edges_to_id(id_))
        if (e.type() == "has_intention")
        {
            s.ack_stamp = G_->get_attrib_by_name<timestamp_agent_att>(e).value_or(0);
            if (s.ack_stamp == s.stamp and s.stamp != 0)
            {
                s.state = G_->get_attrib_by_name<state_att>(e).value_or(std::string{});
                s.bt_state = G_->get_attrib_by_name<bt_state_att>(e).value_or(std::string{});
            }
            break;
        }
    return s;
}

}   // namespace rc::chat
