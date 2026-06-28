/*
 * bottle_affordance.h
 *
 * Manages the lifecycle of an "affordance" DSR node that hangs from a bottle (cylinder) node via a
 * "has_intention" edge. The node advertises the robot pose that maximises epistemic value for this
 * bottle — the HIDDEN-FACE viewpoint computed by EpistemicPlanner — and tracks the mission
 * controller's progress through a simple state machine:
 *
 *   idle ──update()──► pending/offered ──controller sets active=true──► executing
 *                                                                             │
 *                              controller sets active=false,pending=false ───► satisfied
 *                                                                             │
 *                         ◄──── update() republishes offered state or remove()/on_node_deleted()
 *
 * DSR node type  : affordance
 * DSR edge type  : has_intention   (bottle_node → affordance_node)
 *
 * Mirrors table_concept/table_affordance.h (object-agnostic mechanics; only the parent node differs).
 */

#pragma once

#include <string>
#include <memory>
#include <cstdint>
#include <print>

#include <dsr/api/dsr_api.h>

#include "epistemic_planner.h"

namespace rc {
class BottleAffordance
{
public:
    enum class State { idle, pending, executing, satisfied, aborted };

    BottleAffordance() = default;

    /// Late initialisation — call once after the bottle DSR node is known.
    void init(std::shared_ptr<DSR::DSRGraph> G,
              uint64_t    bottle_node_id,
              std::string bottle_node_name);

    /// Create (first call) or refresh (subsequent calls) the affordance node. No-op if not initialised.
    void update(const EpistemicProposal& prop);

    /// Delete the affordance node (instance reset / teardown).
    void remove();

    /// Called from modify_node_attrs_slot — tracks controller-owned protocol transitions.
    void on_node_modified(uint64_t id);
    /// Called from del_node_slot — resets to idle if the affordance node was deleted externally.
    void on_node_deleted(uint64_t id);

    State    state()     const { return state_; }
    bool     is_active() const { return state_ != State::idle; }
    uint64_t node_id()   const { return affordance_node_id_; }

    static std::string_view state_name(State s);

private:
    std::shared_ptr<DSR::DSRGraph> G_;
    uint64_t    bottle_node_id_   = 0;
    std::string bottle_node_name_;

    uint64_t affordance_node_id_ = 0;
    bool     node_created_       = false;
    State    state_              = State::idle;

    void create_node(const EpistemicProposal& prop);
    void update_node(const EpistemicProposal& prop);
    void refresh_edge();
    void reset();
};

}  // namespace rc
