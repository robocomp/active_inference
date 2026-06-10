/*
 * table_affordance.h
 *
 * Manages the lifecycle of an "affordance" DSR node that hangs from a table
 * node via a "has_intention" edge.  The node advertises the robot pose that
 * maximises epistemic value for this table, and tracks the mission-controller's
 * progress through a simple state machine:
 *
 *   idle ──update()──► pending/offered ──controller sets active=true──► executing
 *                                                                             │
 *                              controller sets active=false,pending=false ───► satisfied
 *                                                                             │
 *                         ◄──── update() republishes offered state or remove()/on_node_deleted()
 *
 * DSR node type  : affordance
 * DSR edge type  : has_intention   (table_node → affordance_node)
 *
 * Attributes written to the affordance node:
 *   epistemic_target_x_m_att    float  target robot X in room frame (m)
 *   epistemic_target_y_m_att    float  target robot Y in room frame (m)
 *   epistemic_target_yaw_rad_att float  robot heading at target (rad)
 *   epistemic_gain_att          float  expected information gain
 *   epistemic_pending_att       bool   true=offered/executing, false=completed
 *   active_att                  bool   false=offered/completed, true=controller executing
 */

#pragma once

#include <string>
#include <memory>
#include <cstdint>
#include <print>

#include <dsr/api/dsr_api.h>

#include "epistemic_planner.h"

// ─────────────────────────────────────────────────────────────────────────────
class TableAffordance
{
public:
    enum class State { idle, pending, executing, satisfied, aborted };

    TableAffordance() = default;

    /// Late initialisation — call once after the table DSR node is known.
    void init(std::shared_ptr<DSR::DSRGraph> G,
              uint64_t    table_node_id,
              std::string table_node_name);

    // ── Compute-cycle interface ───────────────────────────────────────────────

    /// Create (first call) or refresh (subsequent calls) the affordance node.
    /// No-op if not initialised.
    void update(const EpistemicProposal& prop);

    /// Delete the affordance node (model became stable or instance reset).
    void remove();

    // ── DSR slot callbacks ───────────────────────────────────────────────────

    /// Called from modify_node_attrs_slot. Tracks controller-owned protocol
    /// state transitions on the affordance node.
    void on_node_modified(uint64_t id);

    /// Called from del_node_slot.  If the affordance node was deleted externally
    /// (controller satisfied / aborted) the state machine resets to idle.
    void on_node_deleted(uint64_t id);

    // ── Accessors ────────────────────────────────────────────────────────────

    State    state()     const { return state_; }
    bool     is_active() const { return state_ != State::idle; }
    uint64_t node_id()   const { return affordance_node_id_; }

    static std::string_view state_name(State s);

private:
    std::shared_ptr<DSR::DSRGraph> G_;
    uint64_t    table_node_id_   = 0;
    std::string table_node_name_;

    uint64_t affordance_node_id_ = 0;
    bool     node_created_       = false;
    State    state_              = State::idle;

    void create_node(const EpistemicProposal& prop);
    void update_node(const EpistemicProposal& prop);
    void refresh_edge();
    void reset();
};
