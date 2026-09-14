/*
 * room_eviction.h — move a departed room out of the live graph and into long-term spatial memory.
 *
 * The design, the contract room_concept must satisfy, and WHY the fleet lets go of its own nodes
 * instead of this agent deleting them, are in ../EVICTION.md. Read that first; this header only
 * says what the code does.
 *
 * One transaction, in an order that is the whole point:
 *
 *   1. MEASURE THE SEAM.  T(room_new <- room_old), read through the robot while BOTH RT edges
 *      still exist. This window is the only time it is computable, so nothing else happens first.
 *   2. COPY the old room's subtree into the memory graph (nodes, attributes, RT edges).
 *   3. PLACE the new room in memory as a stub hanging off the old one, the seam ON that RT edge,
 *      so memory's RT tree IS the topological map and inner_eigen composes across rooms.
 *   4. WRITE THE DOOR TWICE, once per side, match-linked, with the vocabulary cortex reserved for
 *      it (other_side_door_name / connected_room_name).
 *   5. VERIFY, then move the `current` edge to the new room. That flip is the fence that tells
 *      every other agent to let its own nodes go; this agent deletes nothing it does not own.
 */
#pragma once

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ltsm
{

/// The two rooms the robot is holding at the instant of the hand-over.
struct Candidate
{
    std::uint64_t robot_id     = 0;
    std::string   robot_name;
    std::uint64_t old_room_id  = 0;      ///< the one being left  (earlier timestamp_creation)
    std::uint64_t new_room_id  = 0;      ///< the one just entered (later  timestamp_creation)
    std::string   old_room_name, new_room_name;
    std::uint64_t old_birth_ms = 0, new_birth_ms = 0;
};

/// What one eviction did, in enough detail for the self-test to assert on it.
struct Outcome
{
    bool        ok = false;
    std::string reason;                  ///< why it stopped, when not ok

    int nodes_copied = 0;                ///< nodes written into memory (the old room's subtree)
    int edges_copied = 0;                ///< RT edges written into memory

    Mat::RTMat  seam = Mat::RTMat::Identity();   ///< T(room_new <- room_old)
    bool        seam_ok = false;

    std::string     door_name;           ///< live name of the crossed door, "" if the room had none
    Mat::Vector3d   door_in_old{0, 0, 0};
    Mat::Vector3d   door_in_new{0, 0, 0};
    std::string     mem_door_old;        ///< the room-side FACE in memory (this room's measurement)
    std::string     mem_doorway;         ///< the frameless `passage` node both faces will match into

    std::uint64_t mem_old_room_id = 0;   ///< the evicted room's twin in memory
    std::uint64_t mem_new_room_id = 0;   ///< the stub for the room just entered
    bool          current_flipped = false;
};

class RoomEviction
{
public:
    RoomEviction(std::shared_ptr<DSR::DSRGraph> live, std::shared_ptr<DSR::DSRGraph> memory);

    /// Two or more rooms hung from the robot by RT ⇒ the oldest one is a candidate. Nothing else
    /// arms this: no timer, no distance, no sigma (CLAUDE.md: no thresholds).
    [[nodiscard]] std::optional<Candidate> detect() const;

    /// The transaction above. Idempotent in the sense that a room already present in memory is
    /// filled in rather than duplicated.
    Outcome evict(const Candidate &cand);

    /// BOOTSTRAP. With a single room and no `current` edge anywhere, place one. The let-go rule
    /// reads "a current edge exists AND points elsewhere", so a graph with no edge at all releases
    /// nothing — which is the fleet today, and is why somebody has to write the first one.
    bool ensure_current_edge();

    /// BACKSTOP, not policy. Nodes still hanging under a non-current room whose owner is not
    /// running (crashed, SIGKILLed) cannot let go by themselves. Off unless explicitly armed.
    int sweep_orphans(std::uint64_t room_id);

private:
    std::shared_ptr<DSR::DSRGraph> live_, mem_;

    /// live id → memory id, for the subtree being copied.
    using IdMap = std::unordered_map<std::uint64_t, std::uint64_t>;

    [[nodiscard]] std::optional<DSR::Node> robot_node() const;
    [[nodiscard]] std::vector<DSR::Node>   rooms_under_robot(const DSR::Node &robot) const;
    /// Memory's abstract passages: type `metaconcept` AND object_subtype "passage" -- the type is
    /// shared with dining-set style groupings, which eviction can copy in from a room.
    [[nodiscard]] std::vector<DSR::Node>   passages() const;

    /// "room_3" → 3; falls back to the room_id attribute, then to a running counter, so a room
    /// whose name carries no index still gets a unique prefix in memory.
    [[nodiscard]] std::uint64_t room_index(const DSR::Node &room) const;

    /// Copy `live_id`'s subtree under `mem_parent_id`, prefixing every name with `prefix`.
    /// `root_edge_attrs` is what goes on the RT edge into the subtree root (identity for the
    /// first room in memory, the seam for every one after it).
    bool copy_subtree(std::uint64_t live_id, std::uint64_t mem_parent_id,
                      const std::string &prefix, std::uint64_t room_idx,
                      const std::map<std::string, DSR::Attribute> &root_edge_attrs,
                      IdMap &map, Outcome &out);

    /// One node's twin in memory: same type, same attributes, name prefixed, parent/level remapped.
    [[nodiscard]] std::optional<std::uint64_t> twin(const DSR::Node &src, std::uint64_t mem_parent_id,
                                                    const std::string &prefix, std::uint64_t room_idx);

    static std::map<std::string, DSR::Attribute> rt_attrs(const Mat::RTMat &t);

    mutable std::uint64_t next_index_ = 0;   ///< fallback room numbering

    /// Rooms already archived, so the transaction runs ONCE. Between the `current` flip and the
    /// moment each owner lets its nodes go, the old room is still hung from the robot and would
    /// otherwise be re-evicted every cycle.
    std::unordered_set<std::uint64_t> archived_;
    mutable bool warned_missing_stamp_ = false;
};

}   // namespace ltsm
