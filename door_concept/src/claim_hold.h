#pragma once
// claim_hold.h — a door is not deleted while a consumer's claim executes on one of its affordances (trace finding S1;
// model: docs/superpowers/specs/tla/RoomTransitionExit.tla, REAP_HOLD / Claimed(d) / Reapable(x)).
//
// The consumer (the controller) holds an `executing` edge to aff_<door>_cross from the moment it claims the cross until the
// crossing is over — which is AFTER the room switch. door_concept lets go of the room left at the switch, and used to tear
// the crossed door down right there (its affordances first, then the node): the claim's target vanished under the claim.
// Now such a door is HELD: it stops being a door of this agent's world (no fit, no births, no offers, no adoption — the
// caller keeps it out exactly as it keeps a door of a room left behind), its cross affordance node is left in the graph
// so the claim keeps its target, and both are deleted on the first cycle the claim is gone. Event-driven: the only
// condition is the graph's own `executing` edges; there is no timer and no threshold.
//
// Pure std (no DSR, no Qt): the graph half — collecting the names the `executing` edges target, deleting the nodes — is
// the caller's. tests/test_claim_hold.cpp.
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rc::door
{
/// Does one of `executing_targets` (the names of the nodes the graph's `executing` edges point at) hang from `door_name`?
/// A door's affordances are named aff_<door> (the look) and aff_<door>_<verb> (approach / open / cross / close).
/// "aff_door_2_" does not match aff_door_21_cross: the separator is part of the prefix.
[[nodiscard]] inline bool claim_executes_on(std::string_view door_name, std::span<const std::string> executing_targets)
{
    if (door_name.empty()) return false;
    const std::string look = "aff_" + std::string(door_name);
    const std::string prefix = look + "_";
    for (const auto& t : executing_targets)
        if (t == look or t.starts_with(prefix)) return true;
    return false;
}

/// The doors held for a claim. `delete_door`: the door node itself is to be deleted when the claim ends (its room is
/// gone, or the door was being removed for another reason); false = only its kept cross affordance is (a door of a room
/// left behind that is still live stays HELD there and goes with that room, as before).
class ClaimHolds
{
public:
    struct Hold { std::string door_name; std::uint64_t cross_aff_id = 0; bool delete_door = false; };
    struct Freed { std::uint64_t door_id = 0; Hold hold; };

    /// Hold `door_id`. Returns true the first time (the caller logs once). A later call may only ADD the deletion.
    bool hold(std::uint64_t door_id, std::string door_name, std::uint64_t cross_aff_id, bool delete_door)
    {
        if (const auto it = holds_.find(door_id); it != holds_.end())
        {
            it->second.delete_door = it->second.delete_door or delete_door;
            if (it->second.cross_aff_id == 0) it->second.cross_aff_id = cross_aff_id;
            return false;
        }
        holds_.emplace(door_id, Hold{std::move(door_name), cross_aff_id, delete_door});
        return true;
    }
    [[nodiscard]] bool holds(std::uint64_t door_id) const { return holds_.contains(door_id); }
    /// Is `aff_id` a cross affordance kept for a claim? (The consumer still executing on it is the SAME crossing going on,
    /// not "the consumer executes something else".)
    [[nodiscard]] bool holds_affordance(std::uint64_t aff_id) const
    {
        if (aff_id == 0) return false;
        for (const auto& [id, h] : holds_) if (h.cross_aff_id == aff_id) return true;
        return false;
    }
    [[nodiscard]] bool empty() const { return holds_.empty(); }
    [[nodiscard]] std::size_t size() const { return holds_.size(); }
    /// The door node was deleted by somebody (another agent, or this one at exit): nothing is left to hold.
    void forget(std::uint64_t door_id) { holds_.erase(door_id); }

    /// The holds whose claim is gone, removed from the set: the caller deletes what each one says, now.
    [[nodiscard]] std::vector<Freed> take_free(std::span<const std::string> executing_targets)
    {
        std::vector<Freed> out;
        for (auto it = holds_.begin(); it != holds_.end();)
            if (not claim_executes_on(it->second.door_name, executing_targets))
            {
                out.push_back({it->first, std::move(it->second)});
                it = holds_.erase(it);
            }
            else ++it;
        return out;
    }

private:
    std::unordered_map<std::uint64_t, Hold> holds_;
};
}  // namespace rc::door
