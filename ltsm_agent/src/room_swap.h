#pragma once
/*
 * room_swap.h -- EXECUTE the recogniser's requests: the provisional room behind a fresh door is a KNOWN one (Swap), or that
 * was wrong (Undo). Since the room-transition protocol (spec case 2, C2.6/C2.9) a swap PROPOSES; it changes nothing live but
 * its own proposal node (rule 1: P belongs to room_concept, which adopts it in place or rejects the proposal).
 *
 * SWAP (C2.6). Writes `proposal_<K>` under the provisional room P (room_proposal.h: T(P <- K), K's room_id and outline, the
 * robot-in-K modes, state "pending") and STAGES memory, reversibly:
 *   - K's archive is held read-only (RoomEviction::hold): no eviction writes into it while the verdict is open;
 *   - case B (P already promoted: memory holds its twin): the twin is flagged `object_subtype "staged"` (the undo snapshot);
 *   - the seam T(K <- previous room) of each crossed door is read NOW, while the chains resolve (CARRY-OVER below), and the
 *     MAP association with K's own faces is logged ("[swap] crossed door X <-> K face Y", read by the tour driver).
 *   NOTHING is written to memory at swap time.
 *
 * VERDICT (step(), polled each cycle; no DSR signal):
 *   accepted  => COMMIT (C2.9): the carry-over is written (the crossed door becomes a passage of BOTH rooms), the staged twin
 *                is dropped (or, case A, the twin a later promotion wrote for P). The proposal stays -- door_concept reads it
 *                (Task 8) -- and K stays held until the robot LEAVES it (on_room_left / on_evict+on_evicted), which releases the
 *                hold, retries any carry-over that could not be written yet, and withdraws the proposal.
 *   rejected  => UNDO: the twin's subtype restored, K released, the proposal deleted. P stays a new room; the recogniser rebases
 *                its evidence (Event::Rejected; recognition_cycle.h) and may propose again.
 *   gone      => as rejected (P or the proposal left the live graph).
 *   the robot leaves P before a verdict (Review Focus 4: rc::room::robot_rt_room names another room; the recogniser's episode
 *   is released; or the eviction leaving P) => WITHDRAW: the same undo, so P is archived as NEW.
 *
 * CARRY-OVER. The crossed door becomes a passage of BOTH rooms: carry_over() puts K's face of it at
 * T(K <- previous room) . (the previous room's own face pose) and match-links it into the passage that holds the previous
 * room's face (minted by RoomEviction::mint_passage when there is none). T(K <- previous room) is read at swap time, while the
 * chains resolve: robot-in-K (the recogniser's relocalisation) through the live previous room (case A) -- or, when the
 * provisional room carries the MOUTH POSE room_concept publishes (Task M, the same attribute fuse_seam_through_aperture
 * reads), through the provisional frame and the inverse of that mouth (the exact aperture, no door node) -- or, in case B,
 * through T(K <- twin) and the passage into the twin. fuse_seam_through_aperture itself is NOT reused: it fits a PROTO frame,
 * and K's frame is already metric.
 * ACCEPTED TRANSFORM (Task 20): every swap-time seam goes through ltsm's PRIOR T(P <- K). On "accepted" (commit) each one is
 * rebuilt from the T(P <- K) and the fit's robot-in-K covariance room_concept states on the proposal NODE in the same update
 * as "accepted" (its fit's, which may differ from the prior by a half turn: recog_proto6), so the prior never reaches memory.
 * A verdict without them is malformed: no seam is written.
 * When K already holds an unmatched door at the aperture (its own face of the same hole), it is ASSOCIATED, not duplicated:
 * MAP over {each unowned K door} U {none -> new face}, door likelihood Gaussian in the chain pose (both leaf orientations,
 * doors are pi-symmetric), 'none' uniform over K's area and yaw. An adopted door's pose is precision-fused with the chain
 * pose (orientation on the branch consistent with the previous face) and written back. The twin's face in a case-B passage
 * goes with the twin; the passage and its crossing history stay and now point at K.
 * A previous room still LIVE at the verdict (fresh -> fresh, or a verdict landing before room_concept's switch) has no face in
 * memory yet: its carry-over is retried on the FACT that writes it -- the eviction that archives that room (on_evicted) -- and
 * on every step() while Adopted (idempotent), so the way back is known while the robot is still in K (try28). KNOWN GAP:
 * carry_ is in memory, not persisted.
 *
 * Late recalls (recall_late) are no longer posted by the swap; sweep_stale still removes any a crashed older build left.
 */
#include "room_eviction.h"
#include "room_proposal.h"
#include "../../common/room_recognition/room_posterior.h"

#include <dsr/api/dsr_api.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ltsm
{
class RoomSwap
{
public:
    RoomSwap(std::shared_ptr<DSR::DSRGraph> live, std::shared_ptr<DSR::DSRGraph> mem, RoomEviction &evictor);

    /// Execute a Swap request: post the proposal and stage memory. `provisional_live_id` = the provisional room
    /// (RoomRecognizer::provisional_id()); 0 = look for a proto hung from the robot. True when the proposal stands.
    bool swap(std::uint64_t known_room_id, const rc::recog::PoseMix &pose_in_k, std::uint64_t provisional_live_id = 0);
    /// Back to "just entering" before a verdict: withdraw the proposal, unstage. True when a pending swap stood.
    bool undo();
    /// The robot left the recognised room (`room` = its live or memory name; empty = whichever is staged; ignored when it
    /// names another). Pending => withdrawn (P archived as new). Adopted => the hold is released, leftover carry-over written,
    /// the proposal withdrawn.
    void on_room_left(const std::string &room = {});
    /// The eviction path, BEFORE evict(): leaving P before a verdict withdraws (so P's twin is written as a NEW room); leaving
    /// the adopted room releases its hold (the very eviction that moves `current` on would otherwise be refused).
    void on_evict(const Candidate &cand);
    /// The eviction path, AFTER evict(): on success the adopted room's window closes (on_room_left); on failure the hold is restored.
    void on_evicted(const Candidate &cand, bool ok);
    /// CARRY-OVER of the crossed door into K (see above): `prev_room_face` = the memory name of the previous room's face of the
    /// crossed door, `known_room_id` = K. Writes K's face and match-links both into one passage. Idempotent; true when the
    /// passage holds a face in each room afterwards. `log_only` = the swap-time association line, nothing written.
    bool carry_over(const std::string &prev_room_face, std::uint64_t known_room_id, bool log_only = false);

    /// What one cycle observed of room_concept's verdict (or the robot's movement) on the standing proposal.
    enum class Event { None, Accepted, Rejected, Withdrawn };
    /// One cycle: read the verdict and act on it (commit / undo / withdraw). Rejected and Withdrawn mean the swap no longer
    /// stands: the caller tells the recogniser (RoomRecognizer::on_proposal_rejected).
    Event step();
    /// Startup: leftover staged twins (memory), late recalls and proposals (live) from a crashed run. Returns how many nodes went.
    int sweep_stale();
    /// Graceful exit: delete the proposal ltsm owns. Returns how many went.
    int remove_owned();

    /// Entering: no swap. Staged: proposal pending. Adopted: accepted, the robot is still in the adopted room.
    enum class State { Entering, Staged, Adopted };
    [[nodiscard]] State state() const { return state_; }

private:
    std::shared_ptr<DSR::DSRGraph> live_, mem_;
    RoomEviction &ev_;
    RoomProposal proposal_;

    State state_ = State::Entering;
    bool leaving_ = false;                        ///< on_evict released the hold; on_evicted decides
    std::string known_mem_;                       ///< memory name of the recognised room K
    std::uint64_t known_id_ = 0;
    std::uint64_t prov_id_ = 0;                   ///< the provisional room P (its id survives adoption: renamed in place)
    std::string prov_name_;                       ///< P's name at swap time
    std::optional<std::uint64_t> prov_rid_;       ///< P's room_id at swap time
    std::optional<std::uint64_t> rt_at_swap_;     ///< the robot's live RT room at swap time (RF4: only a change AWAY from it counts)
    bool promoted_case_ = false;                  ///< case B (the provisional room had a memory twin)
    std::string prov_twin_;                       ///< case B: the staged twin's memory name
    std::string prov_twin_subtype_;               ///< its object_subtype before staging (restored by undo)

    void commit();                                ///< the accepted verdict: carry-over written, twin dropped
    void retry_carries();                         ///< write the carry-overs still deferred (idempotent; a no-op when none)
    void abandon(std::string_view why);           ///< no verdict will be acted on: withdraw, unstage, release
    void clear();
    [[nodiscard]] bool names_known(const std::string &room) const;

    /// A crossed door, taken at swap time: the previous room's memory name and face, the seam T(K <- previous room) and the
    /// covariance of the robot's pose in K that it inherits (x, y, yaw).
    /// robot_from_prev / side_cov: the seam's robot-side half T(robot <- prev) and its own covariance (the mouth's; else 0),
    /// from which the accepted verdict rebuilds the seam (commit).
    struct Carry { std::string prev_face, prev_room, base; Mat::RTMat known_from_prev; Eigen::Matrix3d cov; std::string live_room; /* L's LIVE name while it is not archived yet; empty = memory only */
                   Mat::RTMat robot_from_prev = Mat::RTMat::Identity(); Eigen::Matrix3d side_cov = Eigen::Matrix3d::Zero(); };
    Mat::RTMat K_from_robot_ = Mat::RTMat::Identity();   ///< the prior robot-in-K pose (best mode) every swap-time seam used
    std::vector<Carry> carry_;
    int drop_subtree(const std::string &memory_room);
    std::optional<DSR::Node> robot_node() const;
};
}   // namespace ltsm
