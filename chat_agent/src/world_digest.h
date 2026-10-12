#pragma once
/*
 * world_digest.h — the robot's situation as a compact text the language model may answer from (and nothing else).
 *
 * Rebuilt every ~500 ms on the MAIN thread from the graph (polled; no DSR signal is connected — CLAUDE.md):
 *   · the current room (rc::room::current_room — the fleet's one rule) and the other rooms in the graph
 *   · the robot pose in the current room's frame (`body`, the de-facto base frame — ROBOT_GEOMETRY.md)
 *   · the doors of the current room (rc::world, the same predicate the controller crosses with): distance and
 *     bearing from `body`, door_open_prob, and the state of door_concept's three affordances
 *   · remembered rooms behind doors (ltsm_agent's recall_<room> metaconcepts, parent = the door)
 *   · the active chat command and how the controller says it is going (state / bt_state on its has_intention edge)
 *   · whether the controller is present
 *
 * The same pass fills GateFacts, so CommandGate validates against exactly what the model was shown.
 * Every get_transformation_matrix optional is checked; a door whose pose does not resolve is listed without one.
 */

#include <cstdint>
#include <memory>
#include <string>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

#include "command_gate.h"

namespace rc::chat
{

struct Digest
{
    std::string text;    ///< the STATE block for the system prompt
    GateFacts facts;     ///< what CommandGate validates against
};

struct IntentionStatus
{
    std::string command;    ///< current_intention
    std::uint64_t stamp = 0;       ///< timestamp_agent of that command (node)
    std::uint64_t ack_stamp = 0;   ///< the command the controller's report below is about (edge timestamp_agent)
    std::string state;      ///< in_progress / completed / aborted / failed|reason — "" until acknowledged
    std::string bt_state;   ///< "<step>|<last_event>|driving=on|off"
};

class WorldDigest
{
public:
    WorldDigest(std::shared_ptr<DSR::DSRGraph> G, std::string robot_frame = "body");
    /// Build the digest now. `inner` must be the main thread's instance (CLAUDE.md: ts==0 cache ownership).
    Digest build(DSR::InnerEigenAPI& inner, bool controller_present, const IntentionStatus& intention) const;

private:
    std::shared_ptr<DSR::DSRGraph> G_;
    std::string robot_frame_;
};

}   // namespace rc::chat
