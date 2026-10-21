// input_latch.h — the DSR-free half of door_concept's input snapshot (Task 3 of the DoorCore record/replay plan,
// docs/superpowers/specs/2026-10-21-door-core-replay-design.md §6.4): the BETWEEN-CYCLE event latch, the actuator
// transcript, and the own-node filter. input_snapshot.{h,cpp} is the DSR side that fills it from the graph; this
// header holds the rules, so tests/test_input_recorder.cpp can check them without a graph.
//
// ★WHAT IS LATCHED. Anything that reaches SpecificWorker between two compute() calls — a node-deleted / door-node-
// upserted slot, a strip-window button press, the startup identity-table + world-registration load, a startup sweep,
// and any actuator call made from a Qt callback — is stored here with its own steady stamp and handed to the NEXT
// cycle's DoorCycleInput, in arrival order. Actuator calls made INSIDE a cycle go straight into that cycle's
// transcript (the port is synchronous: ruling "actuator is an INJECTED synchronous port").
//
// ★OWN NODES ARE NOT RECORDED (ruling, Task 2 review I-3): node events and object rows are kept only for nodes NOT
// created by this agent (core::created_by — the creator's agent id rides in the low 12 bits of every DSR id). The core
// mirrors what it wrote; replaying our own deletes back into it would double-apply them.
#pragma once
#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>
#include "core/door_cycle_io.h"

namespace rc::door
{

/// True when a node id must be recorded: it was created by SOMEBODY ELSE.
[[nodiscard]] inline bool is_foreign(std::uint64_t node_id, std::uint32_t agent_id)
{ return not core::created_by(node_id, agent_id); }

/// Drop our own rows from a snapshot list (objects[] / walls[] — anything with an `id`).
template <class Row>
void keep_foreign(std::vector<Row>& rows, std::uint32_t agent_id)
{
    std::erase_if(rows, [agent_id](const Row& r) { return not is_foreign(r.id, agent_id); });
}

class InputLatch
{
public:
    explicit InputLatch(std::uint32_t agent_id = 0) : agent_id_(agent_id) {}
    void set_agent_id(std::uint32_t id) { agent_id_ = id; }
    [[nodiscard]] std::uint32_t agent_id() const { return agent_id_; }

    // ── events ────────────────────────────────────────────────────────────────────────────────────────────────
    /// A node deleted. Own ids are dropped (returns false).
    bool node_deleted(std::uint64_t id, std::int64_t t_ns)
    {
        if (not is_foreign(id, agent_id_)) return false;
        pending_.push_back({t_ns, core::NodeDeleted{id}});
        return true;
    }
    /// A foreign door node appeared/changed. Only the LAST upsert per id per cycle is kept, at the position of its
    /// latest arrival (an earlier upsert of the same id is superseded, not reordered around).
    bool door_upserted(core::DoorNodeUpserted u, std::int64_t t_ns)
    {
        if (not is_foreign(u.node.id, agent_id_)) return false;
        const std::uint64_t id = u.node.id;
        std::erase_if(pending_, [id](const core::DoorEvent& e)
        {
            const auto* p = std::get_if<core::DoorNodeUpserted>(&e.payload);
            return p != nullptr and p->node.id == id;
        });
        pending_.push_back({t_ns, std::move(u)});
        return true;
    }
    void ui_request(core::UiActuationRequest r, std::int64_t t_ns) { pending_.push_back({t_ns, std::move(r)}); }
    void owned_swept(core::OwnedNodesSwept s, std::int64_t t_ns) { pending_.push_back({t_ns, std::move(s)}); }
    void node_id_assigned(core::NodeIdAssigned a, std::int64_t t_ns) { pending_.push_back({t_ns, std::move(a)}); }
    /// The startup load (spec §6.6): delivered once, on the first recorded cycle.
    void identity_load(core::IdentityLoad l) { identity_load_ = std::move(l); }

    // ── actuator transcript ───────────────────────────────────────────────────────────────────────────────────
    /// A call through the port. Inside a cycle it belongs to that cycle; between cycles to the next one.
    void actuator_call(core::ActuatorCall c)
    {
        if (cycle_ != nullptr) cycle_->actuator_calls.push_back(std::move(c));
        else pending_calls_.push_back(std::move(c));
    }

    // ── cycle bracket ─────────────────────────────────────────────────────────────────────────────────────────
    /// Start a cycle: everything latched since the previous one moves into `in` (events, the between-cycle actuator
    /// calls, the identity load once). Calls made until end_cycle() are appended to `in` directly.
    void begin_cycle(core::DoorCycleInput& in)
    {
        in.events.insert(in.events.end(), std::make_move_iterator(pending_.begin()),
                         std::make_move_iterator(pending_.end()));
        pending_.clear();
        in.actuator_calls.insert(in.actuator_calls.end(), std::make_move_iterator(pending_calls_.begin()),
                                 std::make_move_iterator(pending_calls_.end()));
        pending_calls_.clear();
        if (identity_load_.has_value())
        {
            in.identity_load = std::move(identity_load_);
            identity_load_.reset();
        }
        cycle_ = &in;
    }
    void end_cycle() { cycle_ = nullptr; }
    [[nodiscard]] bool in_cycle() const { return cycle_ != nullptr; }

    [[nodiscard]] std::size_t pending_events() const { return pending_.size(); }
    [[nodiscard]] std::size_t pending_calls() const { return pending_calls_.size(); }
    [[nodiscard]] bool identity_load_pending() const { return identity_load_.has_value(); }

private:
    std::uint32_t agent_id_ = 0;
    std::vector<core::DoorEvent>      pending_;
    std::vector<core::ActuatorCall>   pending_calls_;
    std::optional<core::IdentityLoad> identity_load_;
    core::DoorCycleInput*             cycle_ = nullptr;
};

}  // namespace rc::door
