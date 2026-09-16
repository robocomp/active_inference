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
#include <dsr/api/dsr_rt_api.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ltsm
{

/// A door in this fleet: an `object` with object_subtype "door" (or, belt and braces, named door_*).
bool is_door(DSR::DSRGraph &g, const DSR::Node &n);

/// The two rooms the robot is holding at the instant of the hand-over.
struct Candidate
{
    std::uint64_t robot_id     = 0;
    std::string   robot_name;
    std::uint64_t old_room_id  = 0;      ///< the one being left  (earlier timestamp_creation)
    std::uint64_t new_room_id  = 0;      ///< the one just entered (later  timestamp_creation)
    std::string   old_room_name, new_room_name;
    std::uint64_t old_birth_ms = 0, new_birth_ms = 0;

    // ── PROMOTION (a proto-room the robot has walked into) ──────────────────────────────────────
    bool promote    = false;             ///< new room is a proto-room: remove its `proto` self-edge
    bool remove_old = false;             ///< delete the old room's subtree from the LIVE graph after the flip
    std::uint64_t old_door_id = 0;       ///< the crossed door on the old side (0 ⇒ nearest to the robot)
    std::uint64_t new_door_id = 0;       ///< its face on the new side (the entry mirror), 0 ⇒ none
    float p_inside = 0.f;                ///< the evidence the promotion was taken on (log only)
};

/// What the promotion evidence reads for one proto-room, every cycle (log + decision).
struct PromotionReading
{
    std::uint64_t proto_id = 0;
    std::string   proto_name;
    double s = 0., sigma_s = 0., body_radius = 0., p_inside = 0.;
    bool   valid = false;
    std::string why_not;                 ///< when !valid
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
    std::uint64_t mem_new_room_id = 0;   ///< the new room's twin in memory (with its subtree)
    std::string   mem_door_new;          ///< the new room's FACE of the crossed door in memory, "" if none
    bool          current_flipped = false;
    bool          promoted = false;      ///< the proto self-edge was removed
    int           live_removed = 0;      ///< live nodes deleted with the old room (remove_old)
};

/// What the startup seed did: which rooms of the working memory were already remembered and which
/// were written into memory for the first time.
struct SeedOutcome
{
    int rooms_seen    = 0;               ///< non-proto room nodes in the live graph
    int rooms_proto   = 0;               ///< proto-rooms skipped (identity not decided yet)
    int rooms_known   = 0;               ///< already in memory (a previous generation, or an earlier seed)
    int rooms_created = 0;               ///< written into memory by this call
    int doors         = 0;               ///< door nodes written with them
    int nodes         = 0;               ///< every node written (rooms + floor + walls + doors)
    /// live room name → memory room name, for every room seen (known or created)
    std::vector<std::pair<std::string, std::string>> rooms;
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

    /// STARTUP. Every room in the working memory that long-term memory does not already hold gets a
    /// room node in memory, with its STRUCTURE: the floor, the walls and every door that exists now,
    /// under the same RT chain as in the live graph so each door keeps its pose in the room's frame.
    /// Furniture and affordances are NOT copied here -- they are short-lived beliefs, and the
    /// eviction copies whatever is really there at the fence. Reads the live graph, never writes it.
    ///
    /// Uses the same `r<idx>_` naming as evict(), memoised per live room, so a later eviction of the
    /// same room FILLS these nodes instead of minting a second copy.
    SeedOutcome seed_from_working_memory();

    /// PROMOTION EVIDENCE for every proto-room hung from the robot: P(the whole footprint is past the
    /// crossed aperture) = Φ((s − r_body)/σ_s), s = the robot's +y in the proto-room frame (origin = the
    /// aperture centre, +y outward, room_concept's construction), σ_s from the robot→proto RT edge's
    /// pose covariance propagated through the inversion. No decision here -- see detect_promotion().
    [[nodiscard]] std::vector<PromotionReading> promotion_evidence(double body_radius) const;

    /// The promotion candidate for `reading`: old = the room `current` points at, new = the proto-room.
    [[nodiscard]] std::optional<Candidate> promotion_candidate(const PromotionReading &reading) const;

    /// BOOTSTRAP. With a single room and no `current` edge anywhere, place one. The let-go rule
    /// reads "a current edge exists AND points elsewhere", so a graph with no edge at all releases
    /// nothing — which is the fleet today, and is why somebody has to write the first one.
    bool ensure_current_edge();

    /// BACKSTOP, not policy. Nodes still hanging under a non-current room whose owner is not
    /// running (crashed, SIGKILLed) cannot let go by themselves. Off unless explicitly armed.
    int sweep_orphans(std::uint64_t room_id,
                      std::string_view why = "its owner was not running to let it go");

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
    /// MEMOISED per live room id: the startup seed and every later eviction of the same room must
    /// agree on the prefix, or the eviction writes a second copy beside the seeded one.
    [[nodiscard]] std::uint64_t room_index(const DSR::Node &room) const;

    /// The memory twin of a live room, if memory already holds one: a `room` node named exactly
    /// `r<its room_id>_<live name>`. ★ IDENTITY BY NAME -- the only key available until place
    /// recognition exists; see seed_from_working_memory().
    [[nodiscard]] std::optional<DSR::Node> remembered(const DSR::Node &live_room) const;

    /// Whole = everything under the room (eviction). Structure = floor, walls and doors only, and
    /// nothing below a door (the startup seed).
    enum class Scope { Whole, Structure };

    /// Copy `live_id`'s subtree under `mem_parent_id`, prefixing every name with `prefix`.
    /// `root_edge_attrs` is what goes on the RT edge into the subtree root (identity for the
    /// first room in memory, the seam for every one after it).
    bool copy_subtree(std::uint64_t live_id, std::uint64_t mem_parent_id,
                      const std::string &prefix, std::uint64_t room_idx,
                      const std::map<std::string, DSR::Attribute> &root_edge_attrs,
                      IdMap &map, Outcome &out, Scope scope = Scope::Whole);

    /// One node's twin in memory: same type, same attributes, name prefixed, parent/level remapped.
    /// An existing twin under a DIFFERENT parent has its old RT edge removed, so the caller's new
    /// edge re-parents it instead of giving it two RT parents.
    [[nodiscard]] std::optional<std::uint64_t> twin(const DSR::Node &src, std::uint64_t mem_parent_id,
                                                    const std::string &prefix, std::uint64_t room_idx);

    static std::map<std::string, DSR::Attribute> rt_attrs(const Mat::RTMat &t);

    /// WALLS HANG FROM THE FLOOR in memory, whatever the live graph had: a wall still RT-parented to its
    /// room (room_concept's legacy walls, authored before the floor existed) is re-hung under that room's
    /// floor with its pose recomposed through the live RT tree, so the viewer can fold the whole wall
    /// fan behind the floor's badge.
    void hang_walls_on_floor(const IdMap &map);

    /// Delete a room and everything RT-below it from the LIVE graph, deepest first; never climbs into a
    /// robot or root node. Returns the number of nodes deleted.
    int remove_live_room(std::uint64_t room_id);

    mutable std::uint64_t next_index_ = 0;   ///< fallback room numbering
    mutable std::unordered_map<std::uint64_t, std::uint64_t> index_of_;   ///< live room id → memory index

    /// Rooms already archived, so the transaction runs ONCE. Between the `current` flip and the
    /// moment each owner lets its nodes go, the old room is still hung from the robot and would
    /// otherwise be re-evicted every cycle.
    std::unordered_set<std::uint64_t> archived_;
    mutable bool warned_missing_stamp_ = false;
};

}   // namespace ltsm
