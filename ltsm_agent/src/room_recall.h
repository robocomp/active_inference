/*
 * room_recall.h — what lies behind a door, posted BEFORE the robot crosses it (2026-10-12, return to room 1).
 *
 * For every live door whose face long-term memory holds in a passage, the room on the passage's OTHER face is
 * posted into the live graph as a hypothesis: a `metaconcept` named `recall_<room>` (object_subtype "room_recall"),
 * RT child of that door, carrying the remembered room's outline (delimiting_polygon_x/y, its own frame), its
 * room_id, and on the RT edge the remembered room's frame in the door's frame (memory: T(face ← room)).
 *
 * room_concept reads it at the moment it creates the room behind a crossed door (it cannot wait: a node's name is
 * fixed at birth) and re-creates THAT room — same name, same frame, remembered outline as the given layout.
 *
 * The two faces of a passage are one physical hole (door_concept births the entry mirror at the exact pose of its
 * source), so T(live door ← remembered room) = T(remembered face ← remembered room).
 *
 * Owned: removed when its door goes, when the remembered room is live again, and on a graceful exit; stale
 * recall_* nodes from a crashed run are swept at start.
 */
#pragma once

#include <dsr/api/dsr_api.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

namespace ltsm
{
class RoomEviction;

class RoomRecall
{
public:
    RoomRecall(std::shared_ptr<DSR::DSRGraph> live, std::shared_ptr<DSR::DSRGraph> memory, const RoomEviction &evictor);

    /// One cycle: post / refresh / retire. Returns how many recall nodes are live after it.
    int step();
    /// Delete every recall_* node in the live graph (startup sweep and graceful exit).
    int remove_all();

private:
    std::shared_ptr<DSR::DSRGraph> live_, mem_;
    const RoomEviction &ev_;
    std::unordered_map<std::uint64_t, std::uint64_t> owned_;   ///< live door id -> recall node id
};
}   // namespace ltsm
