// The ltsm_topology summary node (see topology_digest.h): read the memory graph, write ONE metaconcept node on the
// LIVE graph carrying the JSON in ltsm_topology_att. Owned by ltsm_agent; type `metaconcept` (never `door`, whose
// sweep would reap it), no RT edge, hung under root.
#pragma once

#include <dsr/api/dsr_api.h>

#include "../../common/ltsm_topology/door_memory.h"

#include <memory>
#include <string>

namespace ltsm
{
/// Memory rooms and passages as the JSON of encode_topology(). Room names are LIVE names (memory prefix r<idx>_
/// stripped); a room is `live` when the live graph holds a room node of that name; a passage face carries a
/// `live_door` when its room is live and the live graph holds the door of that name.
std::string build_topology_json(DSR::DSRGraph& memory, DSR::DSRGraph& live);

/// DOOR MEMORY (door_memory.h, 2026-11-15): the door names' counter and the change hazard. What memory holds is the record on
/// its node `ltsm_door_memory` (door_next_index_att / door_change_hazard_att), with the counter raised past every door face memory keeps, so a name
/// memory holds is never handed out again even when the record itself was lost (a SIGKILL before the next save).
rc::doormem::DoorMemory remembered_door_memory(DSR::DSRGraph& memory);
/// door_concept's live record (on its own agent node, agent_name "door_concept"); nullopt while it has written none.
std::optional<rc::doormem::DoorMemory> agent_door_memory(DSR::DSRGraph& live);
/// Merge door_concept's live record into memory's root. true when memory's record changed (the caller marks the store dirty).
bool keep_door_memory(DSR::DSRGraph& memory, DSR::DSRGraph& live);

class TopologyPublisher
{
public:
    static constexpr const char* kNodeName = "ltsm_topology";
    TopologyPublisher(std::shared_ptr<DSR::DSRGraph> live, std::shared_ptr<DSR::DSRGraph> memory)
        : live_(std::move(live)), mem_(std::move(memory)) {}

    /// Rebuild the summary and write it when it differs from what is already published. true when written.
    bool refresh();
    /// Delete the node (graceful exit / startup sweep of a crashed run's).
    void remove();

private:
    std::shared_ptr<DSR::DSRGraph> live_, mem_;
    std::string last_;
};
}   // namespace ltsm
