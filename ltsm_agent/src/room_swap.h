#pragma once
/*
 * room_swap.h -- EXECUTE the recogniser's requests: the room behind a fresh door is a KNOWN one (Swap), or that was wrong
 * (Undo). Staged and reversible; it runs through the recall/promotion machinery, not a new node lifecycle.
 *
 * TWO CASES (room_recognizer.h decides WHEN; this decides WHAT happens in the graph):
 *   A. the provisional room P is still a PROTO (no memory twin): swap posts a LATE recall of the known room K under the
 *      crossed door (room_recall.h: recall_late, recall_pose_prior) and discards P from the live graph (Candidate::discard:
 *      nothing is written to memory). room_concept (its Task 9) re-births K from the recall. undo removes that recall and
 *      room_concept re-runs the normal proto birth at the crossing. Memory is untouched in BOTH directions.
 *   B. P was already PROMOTED (its memory twin exists, the previous room is archived with a passage to it): swap flags the
 *      twin `object_subtype "staged"` (kept until the robot leaves: the undo snapshot), posts the late recall for K and
 *      discards the live P. The previous room's passage face into the twin is NOT rewritten. undo re-posts a recall for
 *      the staged twin (the same mechanism, P as the "remembered room"), clears `staged`, and removes K's late recall.
 *
 * READ-ONLY ARCHIVE WHILE STAGED. K's memory twin is held in the RoomEviction (hold()): an evict() of K as the room being
 * LEFT is refused, as the room being ENTERED its twin is not copied into. The robot leaving K ends the undo window:
 * on_room_left() deletes the staged twin and the late recall and releases the hold; on_evict() is that call made from the
 * eviction path BEFORE evict() (the hold would otherwise refuse the very eviction that moves `current` on).
 * CARRY-OVER. At the release (on_room_left) the crossed door becomes a passage of BOTH rooms: carry_over() puts K's face of it at
 * T(K <- previous room) . (the previous room's own face pose) and match-links it into the passage that holds the previous
 * room's face (minted by RoomEviction::mint_passage when there is none). T(K <- previous room) is read at swap time, while the
 * chains resolve: robot-in-K (the recogniser's relocalisation) through the live previous room (case A) -- or, when the
 * provisional room carries the MOUTH POSE room_concept publishes (Task M, the same attribute fuse_seam_through_aperture
 * reads), through the provisional frame and the inverse of that mouth (the exact aperture, no door node) -- or, in case B,
 * through T(K <- twin) and the passage into the twin. fuse_seam_through_aperture itself is NOT reused: it fits a PROTO frame,
 * and K's frame is already metric.
 * When K already holds an unmatched door at the aperture (its own face of the same hole), it is ASSOCIATED, not duplicated:
 * MAP over {each unowned K door} U {none -> new face}, door likelihood Gaussian in the chain pose (both leaf orientations,
 * doors are pi-symmetric), 'none' uniform over K's area and yaw. An adopted door's pose is precision-fused with the chain
 * pose (orientation on the branch consistent with the previous face) and written back. The twin's face in a case-B passage
 * goes with the twin; the passage and its crossing history stay and now point at K.
 * KNOWN GAP: carry-over runs ONLY at on_room_left. If the robot never leaves K, or the agent dies first, the crossed door
 * keeps one face (carry_ is in memory, not persisted) until the next crossing writes it; the recogniser then reads it known.
 *
 * LATE-ADOPTION CONTRACT (for room_concept, Task 9). The late recall node `recall_<room>` (type metaconcept, subtype
 * "room_recall", recall_late = true) is owned by ltsm. room_concept re-births the room from it and fits it with the evidence it
 * has. If that late-adoption fit REJECTS the recalled room, room_concept writes the existing registered bool attribute
 * `valid = false` on that recall node (and must not delete it). RoomRecognizer polls it every cycle while a swap stands and
 * requests the Undo; RoomSwap::undo() removes the recall and the recognizer is acknowledged. Absent or true = no objection.
 * Every late recall RoomSwap posts (K's, and the staged twin's after an undo) is tracked: step() retires it once its room is
 * live again (unless marked invalid), on_room_left()/on_evicted() when the robot leaves, sweep_stale() at start.
 *
 * ANCHOR. The late recall hangs under the crossed door D of the room being left while it is still live (case A); when no
 * door survives (case B: the previous room is archived and P's mirror dies with P) it hangs under the robot, where
 * T(robot <- K) is the inverse of the pose the recogniser handed over (the robot in K).
 */
#include "room_eviction.h"
#include "room_recall.h"
#include "../../common/room_recognition/room_posterior.h"

#include <dsr/api/dsr_api.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace ltsm
{
class RoomSwap
{
public:
    RoomSwap(std::shared_ptr<DSR::DSRGraph> live, std::shared_ptr<DSR::DSRGraph> mem, RoomEviction &evictor, RoomRecall &recall);

    /// Execute a Swap request. `provisional_live_id` = the provisional room (RoomRecognizer::provisional_id()); 0 = look for
    /// a proto / just-entered room hung from the robot. True when the request was accepted and executed.
    bool swap(std::uint64_t known_room_id, const rc::recog::PoseMix &pose_in_k, std::uint64_t provisional_live_id = 0);
    /// Back to "just entering". True when a swap stood and is undone.
    bool undo();
    /// The undo window closes (the robot left the recognised room): drop the staged twin and the late recall, release the
    /// archive. `room` = the recognised room's live or memory name; empty = whichever is staged. Ignored when it names another.
    void on_room_left(const std::string &room = {});
    /// The eviction path, BEFORE evict(): the room being left is the staged one => its hold is released (the very eviction that
    /// moves `current` on would otherwise be refused). Nothing is deleted yet.
    void on_evict(const Candidate &cand);
    /// The eviction path, AFTER evict(): on success the undo window closes (on_room_left); on failure the hold is restored.
    void on_evicted(const Candidate &cand, bool ok);
    /// CARRY-OVER of the crossed door into K (see above): `prev_room_face` = the memory name of the previous room's face of the
    /// crossed door, `known_room_id` = K. Writes K's face and match-links both into one passage. Idempotent; true when the
    /// passage holds a face in each room afterwards. Called by on_room_left for every door staged at swap time.
    bool carry_over(const std::string &prev_room_face, std::uint64_t known_room_id, bool log_only = false);
    /// One cycle: retire the late recalls whose room is live again.
    void step();
    /// Startup: leftover staged twins (memory) and late recalls (live) from a crashed run. Returns how many nodes went.
    int sweep_stale();

    enum class State { Entering, Staged };
    [[nodiscard]] State state() const { return state_; }

private:
    std::shared_ptr<DSR::DSRGraph> live_, mem_;
    RoomEviction &ev_;
    RoomRecall &recall_;

    State state_ = State::Entering;
    std::set<std::string> late_;                  ///< memory rooms whose late recall this class posted (and so retires)
    bool leaving_ = false;                        ///< on_evict released the hold; on_evicted decides
    void retire_all_late();
    std::string known_mem_;                       ///< memory name of the recognised room K
    std::uint64_t known_id_ = 0;
    bool promoted_case_ = false;                  ///< case B (the provisional room had a memory twin)
    std::string prov_twin_;                       ///< case B: the staged twin's memory name
    std::string prov_twin_subtype_;               ///< its object_subtype before staging (restored by undo)
    std::optional<Mat::RTMat> twin_from_known_;   ///< case B: T(K <- provisional twin), at swap time
    std::optional<Mat::RTMat> robot_from_known_;  ///< T(robot <- K) at swap time

    /// A crossed door, taken at swap time: the previous room's memory name and face, the seam T(K <- previous room) and the
    /// covariance of the robot's pose in K that it inherits (x, y, yaw).
    struct Carry { std::string prev_face, prev_room, base; Mat::RTMat known_from_prev; Eigen::Matrix3d cov; std::string live_room; /* L's LIVE name while it is not archived yet; empty = memory only */ };
    std::vector<Carry> carry_;
    int drop_subtree(const std::string &memory_room);
    std::optional<DSR::Node> robot_node() const;
};
}   // namespace ltsm
