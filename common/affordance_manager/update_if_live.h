// update_if_live.h — a write on an affordance node that may be deleted under us (2026-11-15).
//
// An affordance belongs to its PRODUCER (door_concept, room_concept, …); a CONSUMER (the controller) writes its claim flags on
// it. The producer may delete the node at any moment — withdrawing an offer is its right — and the delete arrives on the DDS
// thread. Every consumer write here is `get_node` → modify → `update_node`; when the delete lands between the two, cortex's
// update_node THROWS ("Cannot update node in G, <id> is deleted", dsr_api.cpp) instead of returning false, and the uncaught
// exception aborted the controller (names_ltsm6: release_execution_claim on aff_door_1_open, 0.3 s after door_concept
// deleted it). A deleted node is the end of the claim, not an error: nothing is left to write. Any OTHER throw is rethrown.
#pragma once

#include <dsr/api/dsr_api.h>
#include <stdexcept>

namespace rc::affordance
{
/// update_node, but a node the owner deleted meanwhile is reported as `false` (nothing written) instead of an exception.
template <class N>
[[nodiscard]] bool update_if_live(DSR::DSRGraph& g, N&& node)
{
    const auto id = node.id();
    try
    {
        return g.update_node(std::forward<N>(node));
    }
    catch (const std::runtime_error&)
    {
        if (g.get_node(id).has_value()) throw;   // the node is still there: a real failure, not the race
        return false;
    }
}
}   // namespace rc::affordance
