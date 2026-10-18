#include "journal_node.h"

#include <print>

#include "../../common/episodic/em_journal.h"
#include "../../common/graph_provenance/creation_stamp.h"

namespace rc
{

bool JournalNode::ensure(DSR::DSRGraph& G)
{
    const char* name = rc::em::kJournalNodeName;
    if (id_ != 0 and not G.get_node(id_).has_value())
    {
        std::println("[journal] our '{}' node {} disappeared — recreating it", name, id_);
        id_ = 0;
        edge_done_ = false;
        last_.clear();
    }
    if (id_ == 0)
    {
        if (const auto other = G.get_node(name); other.has_value())
        {
            G.delete_node(other->id());
            std::println("[journal] removed stale '{}' node id={} (a previous run's)", name, other->id());
        }
        DSR::Node node = DSR::Node::create<intention_node_type>(name);
        const auto robot = G.get_node(kRobotNode);
        G.add_or_modify_attrib_local<level_att>(node, robot.has_value() ? G.get_node_level(robot.value()).value_or(0) + 1 : 1);
        if (robot.has_value()) G.add_or_modify_attrib_local<parent_att>(node, robot->id());
        G.add_or_modify_attrib_local<pos_x_att>(node, 420.f);
        G.add_or_modify_attrib_local<pos_y_att>(node, -60.f);
        rc::provenance::stamp_creation(G, node);
        const auto ins = G.insert_node(node);
        if (not ins.has_value())
        {
            std::println("[journal] could not insert '{}'", name);
            return false;
        }
        id_ = ins.value();
    }
    if (not edge_done_)
        if (const auto robot = G.get_node(kRobotNode); robot.has_value())
            edge_done_ = G.insert_or_assign_edge(DSR::Edge::create<has_edge_type>(robot->id(), id_));
    return true;
}

bool JournalNode::publish(DSR::DSRGraph& G, const std::string& wire)
{
    if (wire.empty()) return true;
    if (id_ != 0 and wire == last_ and G.get_node(id_).has_value()) return true;   // unchanged and still there
    if (not ensure(G)) return false;
    auto node = G.get_node(id_);
    if (not node.has_value()) return false;
    G.add_or_modify_attrib_local<em_journal_att>(node.value(), wire);
    if (not G.update_node(node.value())) return false;
    last_ = wire;
    return true;
}

void JournalNode::remove(DSR::DSRGraph& G)
{
    if (id_ != 0 and G.delete_node(id_)) std::println("[journal] node '{}' removed", rc::em::kJournalNodeName);
    id_ = 0;
}

}   // namespace rc
