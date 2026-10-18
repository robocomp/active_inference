#include "mission_node.h"

#include <format>
#include <print>

#include "../../common/graph_provenance/creation_stamp.h"
#include "../../common/mission/mission_progress.h"

namespace rc
{

bool MissionNode::ensure(DSR::DSRGraph& G)
{
    if (id_ != 0 and not G.get_node(id_).has_value())
    {
        std::println("[mission] our '{}' node {} disappeared — recreating it", kName, id_);
        id_ = 0;
        edge_done_ = false;
        last_.clear();
    }
    if (id_ == 0)
    {
        if (const auto other = G.get_node(kName); other.has_value())
        {
            G.delete_node(other->id());
            std::println("[mission] removed stale '{}' node id={} (a previous run's)", kName, other->id());
        }
        DSR::Node node = DSR::Node::create<intention_node_type>(kName);
        const auto robot = G.get_node(kRobotNode);
        G.add_or_modify_attrib_local<level_att>(node, robot.has_value() ? G.get_node_level(robot.value()).value_or(0) + 1 : 1);
        if (robot.has_value()) G.add_or_modify_attrib_local<parent_att>(node, robot->id());
        G.add_or_modify_attrib_local<pos_x_att>(node, 420.f);
        G.add_or_modify_attrib_local<pos_y_att>(node, -160.f);
        rc::provenance::stamp_creation(G, node);
        const auto ins = G.insert_node(node);
        if (not ins.has_value())
        {
            std::println("[mission] could not insert '{}'", kName);
            return false;
        }
        id_ = ins.value();
    }
    if (not edge_done_)
        if (const auto robot = G.get_node(kRobotNode); robot.has_value())
            edge_done_ = G.insert_or_assign_edge(DSR::Edge::create<has_edge_type>(robot->id(), id_));
    return true;
}

void MissionNode::publish(DSR::DSRGraph& G, const rc::mission::Interpreter& in, const std::string& source,
                          const std::string& ir_json)
{
    if (in.state() == rc::mission::Interpreter::State::Idle) return;   // nothing to say until a mission has started
    const std::string progress = rc::mission::encode_progress(in, source);
    const std::string state = in.status();
    const std::string line = in.progress();
    std::string key = std::format("{}\x1f{}\x1f{}\x1f{}\x1f{}", in.serial(), ir_json, progress, state, line);
    if (key == last_ or not ensure(G)) return;
    auto node = G.get_node(id_);
    if (not node.has_value()) return;
    G.add_or_modify_attrib_local<current_intention_att>(node.value(), ir_json);
    G.add_or_modify_attrib_local<mission_serial_att>(node.value(), int(in.serial()));
    G.add_or_modify_attrib_local<mission_source_att>(node.value(), source);
    G.add_or_modify_attrib_local<state_att>(node.value(), state);
    G.add_or_modify_attrib_local<bt_state_att>(node.value(), line);
    G.add_or_modify_attrib_local<timestamp_agent_att>(node.value(), in.span().first);
    G.add_or_modify_attrib_local<mission_progress_att>(node.value(), progress);
    if (G.update_node(node.value())) last_ = std::move(key);
}

void MissionNode::remove(DSR::DSRGraph& G)
{
    if (id_ != 0 and G.delete_node(id_)) std::println("[mission] node '{}' removed", kName);
    id_ = 0;
}

}   // namespace rc
