#pragma once
/*
 * intention_node.h — chat_agent's ONE owned node: `chat_intention`, the command channel to the controller.
 *
 *   type `intention`, has_intention edge  Shadow → chat_intention  (the robot node; added once it exists)
 *   node  current_intention  the command, in the controller's grammar (cross door_19 | stop | uncross) [chat_agent]
 *   node  timestamp_agent    epoch ms of the command; a new stamp IS a new command                     [chat_agent]
 *   edge  state, bt_state    how it is going: in_progress → completed|aborted|failed|<reason>          [controller]
 *   edge  timestamp_agent    which command that report is about                                        [controller]
 *
 * ★ ONE WRITER PER OBJECT (controller/src/chat_inbox.h has the why): cortex's LWW update_node re-versions every
 *   attribute of the writer's local copy, so two agents writing different attributes of ONE node can silently drop
 *   each other's writes. chat_agent writes the node only; the controller writes the has_intention edge only.
 *
 * OWNERSHIP (CONCEPT_AGENT_LIFECYCLE.md §5): listed in [Owns] (exact name), so the presence coordinator deletes it
 * on a graceful exit. A node with this name that THIS process did not create is a leftover of a crashed run and is
 * deleted before ours is born (sweep_stale in initialize() and again on first Operating; ensure() checks too).
 * Stamped with rc::provenance::stamp_creation before insert. Main thread only.
 */

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <dsr/api/dsr_api.h>

#include "world_digest.h"   // IntentionStatus

namespace rc::chat
{

class IntentionNode
{
public:
    static constexpr const char* kName = "chat_intention";
    static constexpr const char* kRobotNode = "Shadow";

    explicit IntentionNode(std::shared_ptr<DSR::DSRGraph> G) : G_(std::move(G)) {}

    /// Delete any chat_intention we did not create (a crashed run's). Returns how many were deleted.
    int sweep_stale();
    /// Create the node if missing, and the has_intention edge once the robot node exists. Cheap when nothing to do.
    bool ensure();
    /// Write a new command (+ stamp). Returns the stamp, or nullopt if the node is unavailable.
    std::optional<std::uint64_t> send(const std::string& command);
    /// The node's current command and the controller's report on it (from the edge).
    [[nodiscard]] IntentionStatus read() const;
    [[nodiscard]] std::uint64_t id() const { return id_; }

private:
    std::shared_ptr<DSR::DSRGraph> G_;
    std::uint64_t id_ = 0;      ///< the node this process created
    bool edge_done_ = false;
};

}   // namespace rc::chat
