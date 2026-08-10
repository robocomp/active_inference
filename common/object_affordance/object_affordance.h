/*
 * object_affordance.h  —  lifecycle of the "affordance" DSR node for one object. SHARED.
 *
 * Manages an affordance node that hangs from an object node via a "has_intention" edge. The node advertises
 * the robot pose that maximises epistemic value for this object and tracks the mission-controller's progress
 * through a simple state machine:
 *
 *   idle ──update()──► pending/offered ──controller sets active=true──► executing
 *                                                                             │
 *                              controller sets active=false,pending=false ───► satisfied
 *                                                                             │
 *                         ◄──── update() republishes offered state or remove()/on_node_deleted()
 *
 * DSR node type  : affordance
 * DSR edge type  : has_intention   (object_node → affordance_node)
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

#include <vector>
#include <cmath>

namespace rc {


// ─── The narrow planner→affordance interface ──────────────────────────────────────────────────────
//
// ★MEASURED, not guessed: across the seven agents the affordance producer reads exactly ELEVEN fields off
// their EpistemicProposal — gain, target x/y/yaw, valid, is_finite, face_gains, sigma_star, standoff_min/max,
// framing_fill. Everything else in those structs (80+ distinct field names across the fleet) is planner
// INTERNALS that never crossed this boundary. So the shared producer takes this, and each agent keeps its own
// EpistemicProposal untouched and fills a target at the call site. That is what makes the extraction cheap:
// the interface was always narrow, it was just never written down.
//
// face_gains / sigma_star are vectors rather than std::array because the agents disagree on size (4 faces vs
// none; 5, 6 and 7 DOF) — the one real dimension of variation, and the reason a single fixed struct never
// emerged by itself.
struct AffordanceTarget
{
    float x_m = 0.0f, y_m = 0.0f, yaw_rad = 0.0f;
    float gain = 0.0f;                  // expected information gain (nats) — what the controller ranks on
    bool  valid = false;

    std::vector<float> face_gains;      // per-face expected gain, if the planner enumerates faces
    std::vector<float> sigma_star;      // per-DOF precision demand; empty = none published
    float standoff_min_m = 0.0f;        // sensor-model stand-off band
    float standoff_max_m = 0.0f;
    float framing_fill   = 0.0f;        // desired projected fill after arrival

    bool is_finite() const
    {
        return std::isfinite(x_m) and std::isfinite(y_m) and std::isfinite(yaw_rad) and std::isfinite(gain);
    }
};

// Owns the affordance node's create/refresh/remove + the controller-protocol state machine for one object.
class ObjectAffordance
{
public:
    enum class State { idle, pending, executing, satisfied, aborted };

    ObjectAffordance() = default;

    /// Late initialisation — call once after the object's DSR node is known.
    // object_type selects the execution CONTRACT (rc::affordance::default_contract_for) — "table",
    // "bottle", "door", … ★It must have a real case in affordance_protocol.h: the fallback returns a
    // valid-LOOKING Contract::reach(), so a missing one is invisible and the robot silently arrives facing
    // the wrong way. That bit door and cabinet once already (invariant 11).
    void init(std::shared_ptr<DSR::DSRGraph> G,
              uint64_t    parent_node_id,
              std::string parent_node_name,
              std::string object_type);

    // ── Compute-cycle interface ───────────────────────────────────────────────

    /// Create (first call) or refresh (subsequent calls) the affordance node.
    /// No-op if not initialised.
    void update(const AffordanceTarget& prop);

    // ── NO PROPOSAL THIS CYCLE IS NOT A WITHDRAWAL ────────────────────────────────────────────────
    // Call on any cycle where a proposal could NOT be computed — the belief has not started, the NBV
    // planner refused (incomplete camera model, degenerate extent, no reachable face). update() is
    // skipped on those paths, and the RE-OFFER lives inside update(), so an affordance the controller
    // had just completed stayed Completed for ever: skipped by every selection branch, its gain frozen
    // at whatever it last published. Measured live 2026-08-07 — aff_table_1 sat at gain=5 (the
    // verify-me maximum) and Completed, while afford_room's gain ticked on beside it.
    // The two things are DIFFERENT statements and must not share a code path: the protocol state says
    // whether the producer still WANTS this look; the target and gain say where and how much. Failing
    // to compute the second must never silently answer the first.
    // Leaves an EXECUTING claim alone — that one belongs to the controller.
    void hold_offered();

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
    uint64_t    parent_node_id_   = 0;
    std::string parent_node_name_;
    std::string object_type_;

    uint64_t affordance_node_id_ = 0;
    bool     node_created_       = false;
    State    state_              = State::idle;

    void create_node(const AffordanceTarget& prop);
    void update_node(const AffordanceTarget& prop);
    void refresh_edge();
    void reset();
};

}  // namespace rc
