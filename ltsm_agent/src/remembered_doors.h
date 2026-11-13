/*
 * remembered_doors.h — the doors long-term memory holds for the robot's CURRENT room, posted into the live graph as
 * `remembered_door` notes (door memory through LTSM, 2026-11-13).
 *
 * door_concept publishes its existence memory on its door nodes (door_exist_evidence, door_seen_ms, ...); when a room is
 * evicted those ride along on the memory room's door faces (RoomEviction::twin copies every attribute). On returning to a room
 * memory knows, each of its faces becomes one `metaconcept` note named `remembered_<face live name>`
 * (object_subtype "remembered_door"), RT child of the live room, carrying the face's attributes and, on the RT edge, the
 * face's pose in memory. door_concept restores a door from it and sets `door_adopted = true`; the note is then deleted here
 * and not reposted while that room stays current.
 *
 * Owned: notes are retired when the room is no longer current / not in memory / staged, and on a graceful exit; stale notes
 * from a crashed run are swept at start (remove_all).
 */
#pragma once

#include <dsr/api/dsr_api.h>

#include <cstdint>
#include <memory>
#include <unordered_set>

namespace ltsm
{
class RememberedDoors
{
public:
    RememberedDoors(std::shared_ptr<DSR::DSRGraph> live, std::shared_ptr<DSR::DSRGraph> memory);

    /// One cycle: post / refresh / retire. Returns how many notes are live after it.
    int step();
    /// Delete every remembered_door note in the live graph (startup sweep and graceful exit).
    int remove_all();

private:
    int retire_where_parent_is_not(std::uint64_t room_id);   ///< deletes notes not hanging from room_id; room_id 0 = all
    std::shared_ptr<DSR::DSRGraph> live_, mem_;
    std::unordered_set<std::uint64_t> adopted_;   ///< memory face ids door_concept has restored this stay
    std::uint64_t current_ = 0;                   ///< the live room the notes belong to
};
}   // namespace ltsm
