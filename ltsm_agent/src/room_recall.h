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
#include <dsr/api/dsr_inner_eigen_api.h>

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
    /// LATE recall (RoomSwap): the robot is already inside `memory_room` (a known room recognised AFTER the crossing), so the
    /// recall hangs under `anchor_id` (the crossed door, or the robot when no door survives) with `recall_late = true` and,
    /// when given, `recall_pose_prior` (10 floats per component: w x y th Sxx Sxy Sxth Syy Syth Sthth, the robot in that
    /// room). `anchor_from_room` = T(anchor <- room). One node per room name: a second call refreshes it in place.
    /// Not owned by step(): step() neither refreshes nor retires it. Returns the node id, 0 on failure.
    std::uint64_t post_late(std::uint64_t anchor_id, const std::string &memory_room, const Mat::RTMat &anchor_from_room,
                            const std::vector<float> &pose_prior);
    /// Delete the recall node named recall_<live name of memory_room>. Returns whether one went.
    bool remove_recall_of(const std::string &memory_room);
    /// How many late recalls are live.
    [[nodiscard]] int late_count() const;
    /// The live name of a memory room (its r<idx>_ prefix removed).
    [[nodiscard]] static std::string live_name_of(const std::string &memory_room);

    /// Delete every recall_* node in the live graph (startup sweep and graceful exit).
    int remove_all();

    /// WARM START (docs/superpowers/specs/2026-11-01-ltsm-warm-start-design.md, [WarmStart] Room): the START-UP recall of the
    /// remembered room `live_room` (memory `r<k>_<live_room>`), the robot's starting room — `recall_<live_room>`, object_subtype
    /// "room_recall", room_id k, the remembered outline, hung from the ROBOT node, no RT edge, no `recall_late`
    /// (rc::room_transition::is_startup_recall). Idempotent. Returns its id, 0 when the memory room, its outline or the robot
    /// node is missing (logged once per reason).
    std::uint64_t post_startup(const std::string &live_room);
    /// Retires the start-up recall once `current` points at another room: the robot has left it, every agent has latched it.
    void step_startup();

private:
    std::shared_ptr<DSR::DSRGraph> live_, mem_;
    const RoomEviction &ev_;
    std::unordered_map<std::uint64_t, std::uint64_t> owned_;   ///< live door id -> recall node id
    std::uint64_t startup_id_ = 0;        ///< the start-up recall (post_startup), 0 = none
    std::string   startup_room_;          ///< its room's live name
    std::string   startup_missing_;       ///< the last "why not posted" logged (once per reason)
};
}   // namespace ltsm
