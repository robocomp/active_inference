#pragma once
/*
 * journal_node.h — where the controller publishes its episodic journal (final review 2026-10-17, finding 4).
 *
 * One node, `em_journal` (type intention, Shadow --has--> it), owned and written ONLY by the controller, carrying the
 * `em_journal` attribute (common/episodic/em_journal.h wire). episodic_agent reads it by name.
 *
 * Why not the controller's agent node (where it lived before): cortex's 1 Hz AgentInfo heartbeat read-modify-writes
 * that node, so a stale heartbeat copy reverted the journal, and the re-assert that fixed it re-broadcast the whole
 * wire after every heartbeat. A node nobody else writes needs neither: the wire is written only when it changed.
 *
 * Lifecycle mirrors MissionNode (mission_node.h): created on the first publish, a previous run's replaced, recreated
 * if it disappears, deleted on a graceful shutdown (cleanup_owned_nodes).
 */

#include <cstdint>
#include <string>

#include <dsr/api/dsr_api.h>

namespace rc
{

class JournalNode
{
public:
    static constexpr const char* kRobotNode = "Shadow";

    /// Write `wire` if it differs from what was last written. Creates the node (and its edge) the first time.
    /// False when it could not be written (the caller keeps it dirty and retries next pass).
    bool publish(DSR::DSRGraph& G, const std::string& wire);
    /// Graceful shutdown: delete our node.
    void remove(DSR::DSRGraph& G);

private:
    bool ensure(DSR::DSRGraph& G);

    std::uint64_t id_ = 0;
    bool edge_done_ = false;
    std::string last_;   ///< the wire last written
};

}   // namespace rc
