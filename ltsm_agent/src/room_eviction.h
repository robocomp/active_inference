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
 *   3. PLACE the new room in memory, under root like every room (2026-10-12): rooms are siblings, each its own
 *      frame; what relates two rooms is the passage's two door faces (seam_through_passage).
 *   4. WRITE THE DOOR TWICE, once per side, match-linked, with the vocabulary cortex reserved for
 *      it (other_side_door_name / connected_room_name).
 *   5. VERIFY, then move the `current` edge to the new room. That flip is the fence that tells
 *      every other agent to let its own nodes go; this agent deletes nothing it does not own.
 */
#pragma once

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/api/dsr_rt_api.h>

#include "seam_fusion.h"

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

/// Suffix of a door face DERIVED from the seam measured through the robot (no entry mirror existed at the crossing).
inline constexpr std::string_view kDerivedFaceSuffix = "_entry";

/// True when memory node `node_id` is `room` or hangs (through parent_att) below it.
bool is_descendant(DSR::DSRGraph &mem, std::uint64_t node_id, const std::string &room);

/// T(room_to <- room_from) composed through `passage`'s two door faces in `mem` (one face per room); nullopt unless
/// the passage has a face in each room. Free so a read-only consumer (the topology view) needs no RoomEviction.
std::optional<Mat::RTMat> seam_through_passage(DSR::DSRGraph &mem, const std::string &passage,
                                               const std::string &room_to, const std::string &room_from);

/// A memory room flagged `object_subtype "staged"`: a draft held by RoomSwap (the undo snapshot of a provisional room's
/// twin), never a candidate, never recalled, never part of a passage consumers read.
bool is_staged_room(DSR::DSRGraph &g, const DSR::Node &room);

// ── PASSAGE IDENTITY (2026-10-29, door-identity design part 2) ─────────────────────────────────────────────────────
// A door's NAME is door_concept's uniqueness slot among LIVE nodes, re-made on every crossing; it is a label, never the
// identity of the hole. The identity of a physical doorway is its `passage_N`, the same N live and in memory: ONE
// numbering (next_passage_index) for both graphs, and a live passage of a hole memory already knows takes memory's name.

/// passage_<N> → N; nullopt for any other name.
std::optional<std::uint64_t> passage_index(std::string_view name);

/// THE ONE passage numbering: 1 + the highest N over every `passage_N` metaconcept in `live` AND in `mem` (0 when there
/// is none). Every passage minted anywhere — PassageLive, RoomEviction::mint_passage, RoomSwap's carry-over — takes its
/// number here, so two holes can never be handed the same name by two counters. `mem` may be null (live only).
std::uint64_t next_passage_index(DSR::DSRGraph &live, DSR::DSRGraph *mem);

/// The LIVE passage (metaconcept, subtype "passage") holding a `match` edge to this live door, if any.
std::optional<DSR::Node> live_passage_of(DSR::DSRGraph &live, std::uint64_t door_id);

/// A hole memory already knows: the passage and the remembered face of it.
struct RememberedHole
{
    DSR::Node passage;
    DSR::Node face;
    double    distance = 0.;   ///< metres between the live door and that face, in the room's frame (0 when matched by name)
};
/// The same seam as an SE(2) pose with its covariance, propagated from the two faces' RT covariances (in room_to's
/// frame). `stated_faces` (0..2) counts the faces whose own edge states one; the others enter at the remembered-seam
/// sigmas. For the memory layout's MAP estimate (layout_optimizer.h).
std::optional<ltsm::seam::Measured> seam_measured_through_passage(DSR::DSRGraph &mem, const std::string &passage,
                                                                  const std::string &room_to, const std::string &room_from,
                                                                  int &stated_faces);

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
    /// The new room was born THROUGH a crossing, so its frame is the crossed aperture (promotion_candidate sets it: the
    /// room passed the crossing evidence). A room that is still a `proto` at eviction time counts too. Without either,
    /// nothing says the new room's frame is the aperture and the aperture-as-frame seam is not used.
    bool crossing_born = false;

    // ── DISCARD (RoomSwap: the provisional room turned out to be a known one) ───────────────────
    /// Delete the room `new_room_id` and its live subtree and write NOTHING to memory (no seam, no copy, no `current` flip).
    /// Only new_room_id / new_room_name are read.
    bool discard = false;
};

/// What the promotion evidence reads for one proto-room, every cycle (log + decision).
/// Promotion trigger (room-transition protocol C1.5/C2.5): room_concept's robot->room RT edge names a room that is not
/// the one `current` points at. No level: the edge is the fact.
[[nodiscard]] inline bool promotion_due(std::optional<std::uint64_t> robot_room, std::optional<std::uint64_t> current_edge_room)
{
    return robot_room.has_value() and robot_room != current_edge_room;
}

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
    bool          mem_door_new_derived = false;   ///< that face was DERIVED from the measured seam (no entry mirror)
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

    /// READ-ONLY ARCHIVE (RoomSwap, while a swap is Staged). evict() writes nothing into a held memory room's twin:
    /// as the room being LEFT it is refused (reason "deferred", nothing written; the caller releases the hold when the
    /// robot leaves, then evicts), as the room being ENTERED its twin is not copied into nor given a derived door face
    /// (the current flip and everything on the old side proceed; the carry-over writes the new side at release).
    void hold(const std::string &memory_room) { held_.insert(memory_room); }
    void release(const std::string &memory_room) { held_.erase(memory_room); }
    [[nodiscard]] bool held(const std::string &memory_room) const { return held_.contains(memory_room); }

    /// Mint an empty abstract passage node in memory (shared by eviction and RoomSwap's carry-over). 0 on failure.
    /// `name` = the identity the hole already carries live (its live passage's name); empty, or already taken in memory,
    /// ⇒ passage_<next_passage_index()>.
    std::uint64_t mint_passage(std::string_view name = {});

    /// next_passage_index over this agent's live graph and memory.
    [[nodiscard]] std::uint64_t next_passage_index() const { return ltsm::next_passage_index(*live_, mem_.get()); }

    /// THE HOLE MEMORY ALREADY KNOWS behind a LIVE door, resolved BY PLACE: the door's room has a memory twin (same
    /// prefix eviction writes it under) and the room is published in its remembered frame, so the live door and that
    /// twin's passage faces are directly comparable — the face whose aperture contains the door's centre (argmin, within
    /// half the door's width: a point-in-segment membership test, not a tuning constant). The door's NAME is consulted
    /// only when its pose in its room cannot be resolved. nullopt = memory holds no passage for this hole.
    /// `staged_twin_ok`: also resolve through a twin RoomSwap has STAGED (a recognition proposal for the door's room is
    /// pending). Only RoomRecall asks for it (F3, TLA ping-pong): the staged twin's face still names the hole; every other
    /// caller treats a staged twin as absent, as before.
    [[nodiscard]] std::optional<RememberedHole> remembered_hole(const DSR::Node &live_door, bool staged_twin_ok = false) const;

    /// BOOTSTRAP. With a single room and no `current` edge anywhere, place one. The let-go rule
    /// reads "a current edge exists AND points elsewhere", so a graph with no edge at all releases
    /// nothing — which is the fleet today, and is why somebody has to write the first one.
    bool ensure_current_edge();

    /// T(room_to <- room_from) composed through `passage`'s two door faces in MEMORY (one face per room). nullopt
    /// unless the passage has a face in each room. This is the only valid cross-room transform in memory: rooms
    /// hang from root with identities that carry no metric meaning.
    [[nodiscard]] std::optional<Mat::RTMat> seam_through_passage(const std::string &passage, const std::string &room_to,
                                                                 const std::string &room_from) const;

    /// The `r<idx>_` prefix this live room's nodes carry in memory (memoised, the same one evict() uses).
    [[nodiscard]] std::string memory_prefix(const DSR::Node &live_room) const
    { return "r" + std::to_string(room_index(live_room)) + "_"; }

    /// Planar-safe packing of a transform as RT edge attributes (rt_translation / rt_rotation_euler_xyz).
    /// T(new<-old) for a PROTO new room, fused from the robot chain and the crossed aperture (seam_fusion.h).
    /// nullopt when the robot is not resolvable in the old room.
    [[nodiscard]] std::optional<Mat::RTMat> fuse_seam_through_aperture(const Candidate &cand, const DSR::Node &door,
                                                                       const Mat::RTMat &door_in_old, const Mat::RTMat &chain_seam,
                                                                       DSR::InnerEigenAPI &ie, bool new_room_was_remembered,
                                                                       const std::optional<Mat::RTMat> &remembered_seam) const;
    [[nodiscard]] static std::map<std::string, DSR::Attribute> rt_attributes(const Mat::RTMat &t) { return rt_attrs(t); }

    /// BACKSTOP, not policy. Nodes still hanging under a non-current room whose owner is not
    /// running (crashed, SIGKILLed) cannot let go by themselves. Off unless explicitly armed.
    int sweep_orphans(std::uint64_t room_id,
                      std::string_view why = "its owner was not running to let it go");

    /// A DOOR FOUND IS REMEMBERED WHEN IT IS FOUND (user 2026-11-01: "we don't have the second bathroom door in ltsm; if we find
    /// it now it should be added"). While the robot is in a REMEMBERED room, its live doors are mirrored into that room's
    /// memory: a live door the memory lacks is added (r<k>_<name>, hung from the remembered room node at its room-frame pose —
    /// the room is live in its stored frame); a door THIS sync added and door_concept has since deleted (existence below its
    /// removal bound) is removed again. Never overwrites: a memory node of that name at ANOTHER place is left alone (eviction's
    /// renaming handles a recycled name). Returns the number of memory changes (> 0 ⇒ the caller marks the store dirty).
    int sync_found_doors();

private:
    struct SyncedDoor { std::string mem_name; std::uint64_t live_room = 0; };
    std::unordered_map<std::uint64_t, SyncedDoor> synced_doors_;   ///< live door id -> what sync_found_doors added
    std::shared_ptr<DSR::DSRGraph> live_, mem_;

    /// live id → memory id, for the subtree being copied.
    using IdMap = std::unordered_map<std::uint64_t, std::uint64_t>;

    [[nodiscard]] std::optional<DSR::Node> robot_node() const;
    /// True when memory node `node_id` is `room` or hangs (through parent_att) below it.
    [[nodiscard]] bool is_descendant(std::uint64_t node_id, const std::string &room) const;
    [[nodiscard]] std::vector<DSR::Node>   rooms_under_robot(const DSR::Node &robot) const;
    /// Memory's abstract passages: type `metaconcept` AND object_subtype "passage" -- the type is
    /// shared with dining-set style groupings, which eviction can copy in from a room.
    [[nodiscard]] std::vector<DSR::Node>   passages(bool staged_ok = false) const;   // staged_ok: also passages into a STAGED twin (remembered_hole)

    /// "room_3" → 3; falls back to the room_id attribute, then to a running counter, so a room
    /// whose name carries no index still gets a unique prefix in memory.
    /// MEMOISED per live room id: the startup seed and every later eviction of the same room must
    /// agree on the prefix, or the eviction writes a second copy beside the seeded one.
    [[nodiscard]] std::uint64_t room_index(const DSR::Node &room) const;

    /// The memory twin of a live room, if memory already holds one: a `room` node named exactly
    /// `r<its room_id>_<live name>`. ★ IDENTITY BY NAME -- the only key available until place
    /// recognition exists; see seed_from_working_memory().
    [[nodiscard]] std::optional<DSR::Node> remembered(const DSR::Node &live_room) const;
    [[nodiscard]] bool remembered_room(const std::optional<DSR::Node> &live_room) const
    { return live_room.has_value() and remembered(live_room.value()).has_value(); }

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

    /// live door id → its memory name when the plain `r<idx>_<door>` already names a DIFFERENT hole's passage face
    /// (decided by place, guard_recycled_door_names); twin() writes the door there instead of over that face.
    std::unordered_map<std::uint64_t, std::string> mem_name_of_;
    void guard_recycled_door_names(std::uint64_t live_room_id, const std::string &prefix);
    /// ONE MEMORY WALL PER WALL (2026-11-01, pingpong40: r1_wall_0..92 for a 31-wall room). room_concept never reuses a wall
    /// name (wall_<k> is an identity per process), so a RECALLED room comes back as wall_31.., wall_62.. and a by-name archive
    /// added a fresh set on every eviction. Each live wall of the room's floor is paired with the remembered wall of the same
    /// memory floor at the same place (mutual nearest segment: no distance gate) and archived under that name (mem_name_of_).
    void match_remembered_walls(std::uint64_t live_room_id, const std::string &prefix);
    /// After the copy: the remembered walls of this room's floor that no live wall was archived onto are the layout's old
    /// edges — removed, unless something still hangs from them (a door face). Returns how many were removed.
    int drop_stale_walls(std::uint64_t live_room_id, const std::string &prefix, const IdMap &map);

    mutable std::uint64_t next_index_ = 0;   ///< fallback room numbering
    mutable std::unordered_map<std::uint64_t, std::uint64_t> index_of_;   ///< live room id → memory index

    /// Rooms already archived, so the transaction runs ONCE. Between the `current` flip and the
    /// moment each owner lets its nodes go, the old room is still hung from the robot and would
    /// otherwise be re-evicted every cycle.
    std::unordered_set<std::uint64_t> archived_;
    mutable bool warned_missing_stamp_ = false;
    std::unordered_set<std::string> held_;   ///< memory rooms whose twin is read-only (see hold())
};

}   // namespace ltsm
