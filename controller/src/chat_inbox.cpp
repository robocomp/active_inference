#include "chat_inbox.h"

#include <format>
#include <print>

namespace rc
{

std::optional<ChatInbox::Command> ChatInbox::poll(DSR::DSRGraph& G, std::uint64_t now_ms)
{
    const auto node = G.get_node(kNodeName);
    if (not node.has_value() or node->type() != "intention") return std::nullopt;
    const std::uint64_t stamp = G.get_attrib_by_name<timestamp_agent_att>(node.value()).value_or(0);
    const auto text_ref = G.get_attrib_by_name<current_intention_att>(node.value());
    std::string text = text_ref.has_value() ? std::string(text_ref->get()) : std::string{};

    if (node->id() != node_id_)
    {
        // First sight of this channel: adopt its stamp; execute only a command that is still fresh.
        node_id_ = node->id();
        last_stamp_ = stamp;
        const bool fresh = stamp != 0 and not text.empty() and now_ms >= stamp and now_ms - stamp <= kFirstSightMaxAgeMs;
        std::println("[chat] inbox: following node '{}' id={}{}", kNodeName, node_id_,
                     fresh ? "" : (text.empty() ? "" : std::format(" (stale command '{}' ignored)", text)));
        if (not fresh) return std::nullopt;
        return Command{.text = std::move(text), .stamp_ms = stamp};
    }
    if (stamp == 0 or stamp == last_stamp_) return std::nullopt;
    last_stamp_ = stamp;
    if (text.empty()) return std::nullopt;
    return Command{.text = std::move(text), .stamp_ms = stamp};
}

void ChatInbox::publish(DSR::DSRGraph& G, std::uint64_t stamp, const std::string& state, const std::string& bt_state)
{
    if (node_id_ == 0) return;
    std::optional<DSR::Edge> edge;
    for (const auto& e : G.get_edges_to_id(node_id_))
        if (e.type() == "has_intention") { edge = e; break; }
    if (not edge.has_value()) return;
    const bool same = G.get_attrib_by_name<timestamp_agent_att>(edge.value()).value_or(0) == stamp
                      and G.get_attrib_by_name<state_att>(edge.value()).value_or(std::string{}) == state
                      and G.get_attrib_by_name<bt_state_att>(edge.value()).value_or(std::string{}) == bt_state;
    if (same) return;
    G.add_or_modify_attrib_local<timestamp_agent_att>(edge.value(), stamp);
    G.add_or_modify_attrib_local<state_att>(edge.value(), state);
    G.add_or_modify_attrib_local<bt_state_att>(edge.value(), bt_state);
    G.insert_or_assign_edge(edge.value());
}

std::pair<std::string, std::string> ChatInbox::status_of(const Tracked& t, const DoorPassage::View& v, bool driving)
{
    using K = Tracked::Kind;
    const std::string drv = driving ? "driving=on" : "driving=off";
    switch (t.kind)
    {
        case K::None: return {"", ""};
        case K::Rejected: return {"failed|" + t.reason, std::format("-|{}|{}", t.reason, drv)};
        case K::Stop: return {"aborted", std::format("stopped|robot stopped by chat|{}", drv)};
        case K::Uncross: return {"aborted", std::format("idle|{}|{}", v.last_event, drv)};
        case K::Motion: return {"", ""};   // status_of_motion
        case K::Mission: return {"", ""};  // the interpreter reports its own (service_mission)
        case K::Cross:
        {
            const std::string bt = std::format("{}|{}|{}", DoorPassage::step_name(v.step), v.last_event, drv);
            if (v.door != t.door) return {"aborted", std::format("superseded|another passage ({}) was started|{}",
                                                                 v.door.empty() ? std::string("none") : v.door, drv)};
            if (v.step == DoorPassage::Step::Done) return {"completed", bt};
            if (v.step == DoorPassage::Step::Idle) return {"aborted", bt};
            return {"in_progress", bt};
        }
    }
    return {"", ""};
}

std::pair<std::string, std::string> ChatInbox::status_of_motion(const Tracked& t, const RelativeMotion::View& v,
                                                               bool driving)
{
    const std::string drv = driving ? "driving=on" : "driving=off";
    // While running, the event is the STEP (it changes only between steps), not the per-cycle progress: every
    // change of bt_state is a DDS write here and a status line in the chat window.
    const bool running = v.phase == RelativeMotion::Phase::Turning or v.phase == RelativeMotion::Phase::Moving;
    const std::string bt = std::format("{}|{}|{}", RelativeMotion::phase_name(v.phase), running ? v.current : v.last_event, drv);
    if (v.id != t.motion_id) return {"aborted", std::format("superseded|another move or turn was started|{}", drv)};
    using P = RelativeMotion::Phase;
    switch (v.phase)
    {
        case P::Turning:
        case P::Moving: return {"in_progress", bt};
        case P::Done: return {"completed", bt};
        case P::Failed: return {"failed|" + v.last_event, bt};
        case P::Cancelled: return {"aborted", bt};
        case P::Idle: break;
    }
    return {"", ""};
}

}   // namespace rc
