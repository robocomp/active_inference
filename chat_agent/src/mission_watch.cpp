#include "mission_watch.h"

#include <print>

#include "../../common/mission/mission_json.h"

namespace rc::chat
{

MissionWatch::Update MissionWatch::poll()
{
    Update u;
    const auto node = G_->get_node(kName);
    if (not node.has_value()) return u;

    // A new IR first, so the caller builds the rows before it applies the snapshot that names them.
    // The same script run twice repeats the IR text: the serial is what says it is another mission.
    const int serial = G_->get_attrib_by_name<mission_serial_att>(node.value()).value_or(0);
    if (const auto ir = G_->get_attrib_by_name<current_intention_att>(node.value());
        ir.has_value() and (ir->get() != last_ir_ or serial != last_serial_))
    {
        last_ir_ = ir->get();
        last_serial_ = serial;
        last_progress_.clear();   // the next snapshot belongs to the new rows: apply it even if its text repeats
        if (auto m = rc::mission::decode_json(last_ir_); m.has_value())
            u.mission = std::make_pair(std::move(*m), last_ir_);
        else
            std::println("[chat_agent] controller_mission: undecodable IR ({})", m.error().message());
    }
    if (const auto p = G_->get_attrib_by_name<mission_progress_att>(node.value()); p.has_value() and p->get() != last_progress_)
    {
        last_progress_ = p->get();
        if (auto d = rc::mission::decode_progress(last_progress_); d.has_value())
            u.progress = std::move(*d);
        else
            std::println("[chat_agent] controller_mission: undecodable progress ({})", d.error());
    }
    return u;
}

}   // namespace rc::chat
