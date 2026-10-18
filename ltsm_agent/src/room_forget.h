/*
 * room_forget.h — remove one room from long-term memory, leaving the rest coherent.
 *
 * What goes, and why each part must:
 *   - the room and EVERYTHING below it: the RT subtree (floor, walls, doors, furniture copies) and any node whose
 *     `parent` names a node in it, so nothing is left hanging from a deleted parent;
 *   - every PASSAGE with a face (match edge) in it. A passage is the only thing relating two rooms
 *     (seam_through_passage); with one face gone it relates nothing, and RoomRecall would keep resolving it
 *     to a room that no longer exists. Its Betas describe crossings of a doorway memory no longer holds.
 * Deleted passages first, then the subtree deepest first, the room last: no node is ever an orphan mid-way.
 *
 * Planning is separate from doing so a UI can say what a deletion will take before it happens. Memory graph only;
 * whether the robot is in the room is the caller's question (it needs the live graph).
 */
#pragma once

#include <dsr/api/dsr_api.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ltsm
{

struct ForgetPlan
{
    std::string room;                         ///< memory name, e.g. r2_room_2
    std::vector<std::uint64_t> passages;      ///< passages with a face in the room
    std::vector<std::uint64_t> subtree;       ///< below the room, deepest first (the room itself excluded)
    std::uint64_t room_id = 0;
};

/// nullopt when `room` is not a room node of `mem`.
std::optional<ForgetPlan> plan_forget(DSR::DSRGraph &mem, const std::string &room);

/// Execute a plan. Returns the number of nodes deleted (passages + subtree + room).
int forget(DSR::DSRGraph &mem, const ForgetPlan &plan);

}   // namespace ltsm
