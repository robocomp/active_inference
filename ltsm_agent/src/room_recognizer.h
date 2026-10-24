#pragma once
/*
 * room_recognizer.h -- "which known room have I just walked into?", asked while the room behind a FRESH door is still
 * provisional. Evidence in, posterior out, a staged commit state; it only REQUESTS a swap or an undo (RoomSwap executes).
 *
 * WHEN IT RUNS (structural, nothing thresholded). Exactly while all of these hold:
 *   - the live graph has a CURRENT room L and a PROVISIONAL room P hung from the robot (a proto-room, or the hand-over
 *     case promotion_evidence also accepts: the robot->L edge retired while robot->P is live);
 *   - P carries an entry mirror of a door of L (door_concept: a door node whose parent is P);
 *   - that crossed door has NO passage in memory (a door crossed before is a known passage; recall is trusted there,
 *     user decision 2026-10-23: no audit of known passages).
 * It stops, and its `recognition` node is removed, the moment any of these stops holding (P promoted, deleted, ...).
 *
 * EVIDENCE. Two experts over the SAME pose in room k (rc::recog::posterior):
 *   metric       the robot pose through the seam chain vs room k's own chain (room_chain.h). The robot->P edge gives the
 *                robot in P with its covariance; the FRESH seam T(L <- P) is built HERE from the crossed door's two faces:
 *                its face in L (memory, the node the eviction/seed wrote for it) and its entry mirror in P (live).
 *                A state, read again each cycle; the decision uses its change since the last state change.
 *   appearance   retina's `place_evidence` node (recog_loglr / recog_mix / recog_mix_n, cumulative and already temporally
 *                discounted), absent today (metric-first) => AppearanceEvidence::available = false.
 * Candidates = the memory rooms that are not staged and are not L (the crossing rules L out structurally); they are sorted
 * by room_id so the order retina sees in `recog_room_ids` is stable.
 *
 * STAGED COMMIT (rc::recog::commit_step). Every decision uses the evidence gathered SINCE THE LAST STATE CHANGE (episode
 * entry, commit, undo): per hypothesis the log-evidence now minus its value at that change (the metric as a DELTA too,
 * appearance rebased the same way), under the uniform prior. No new data => posterior = prior => no undo, no re-commit.
 * Swap/Undo are emitted on the TRANSITION only. A missing or unresolvable expert is omitted from the product, never zeroed.
 *
 * THE EPISODE SPANS THE PROMOTION: it lives while the current room is L, the provisional room (its promoted twin; the fresh
 * seam cached while L was still measurable) or the committed room (held, no new evidence: the evidence frame is gone), and
 * ends at the eviction fence (the robot leaves it). Ending any other way under a commit, or losing the committed room from
 * the candidates, emits an Undo request.
 *
 * OWNERSHIP. ltsm owns the live `recognition` metaconcept (recog_room_ids, recog_posterior; the last posterior entry is
 * "new"). It is removed on exit and swept at start (a SIGKILLed run leaves it; so can retina leave a `place_evidence`,
 * which retina rewrites when it runs again). No DSR signal is connected: ltsm polls from compute().
 */
#include "room_chain.h"
#include "room_eviction.h"
#include "../../common/room_recognition/room_posterior.h"

#include <dsr/api/dsr_api.h>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace ltsm
{
class RoomRecognizer
{
public:
    RoomRecognizer(std::shared_ptr<DSR::DSRGraph> live, std::shared_ptr<DSR::DSRGraph> mem, const RoomEviction &evictor);

    struct Step
    {
        enum class Act { None, Swap, Undo } act = Act::None;
        int           room_index = -1;      ///< index into the candidates (recog_room_ids order)
        std::uint64_t room_id = 0;
        rc::recog::PoseMix pose_in_k;       ///< the robot in that room's frame, as the joint posterior holds it
    };

    /// One cycle. Swap/Undo is a REQUEST, executed by RoomSwap.
    Step step();

    /// ACKNOWLEDGEMENTS from RoomSwap (a request is not an execution).
    /// A requested Swap was executed: from here the graph changes the swap itself causes (the provisional room discarded
    /// before `current` moves, the crossed door gaining a passage) are NOT episode loss -- the episode is HELD (no new
    /// evidence: its frame is gone) until the robot leaves the committed room (the eviction fence) or an Undo is executed.
    void on_swap_executed(std::uint64_t room_id);
    /// A requested Swap failed: the commit reverts to Entering (no Undo is requested: nothing was done).
    void on_swap_failed();
    /// A requested Undo was executed: the swap no longer stands.
    void on_undo_executed();
    /// The provisional room of the running episode (0 = none), and whether an executed swap is being held.
    [[nodiscard]] std::uint64_t provisional_id() const { return prov_id_; }
    [[nodiscard]] bool swapped() const { return swap_ack_; }

    /// Startup / graceful exit: remove the `recognition` node and any stale `place_evidence`. Returns how many went.
    int sweep_stale();

    /// Graceful exit: remove only what ltsm owns (the `recognition` node); retina's `place_evidence` is not ours to delete.
    int remove_owned();

    /// The latest since-entry posterior (known rooms in candidate order, `new` last); empty when no episode is running.
    [[nodiscard]] const std::vector<rc::recog::Hypothesis> &posterior() const { return post_entry_; }
    [[nodiscard]] const std::vector<std::uint64_t> &candidates() const { return ids_; }

private:
    std::shared_ptr<DSR::DSRGraph> live_, mem_;
    const RoomEviction &ev_;

    // ── the crossing history: the memory passage the robot used to enter the current room ──
    std::string cur_mem_;                       ///< memory name of the current room, "" when unknown
    std::string prev_mem_, prev_passage_;       ///< the room before it and the passage between them ("" when none)

    // ── the episode ──
    std::uint64_t prov_id_ = 0;                 ///< the provisional room this episode is about (0 = none)
    std::vector<std::uint64_t> ids_;            ///< candidate room ids, sorted
    rc::recog::Commit commit_{rc::recog::Commit::State::Entering, -1};
    std::map<std::int64_t, float> base_;               ///< log posterior of each hypothesis (-1 = new) at the last state change
    std::uint64_t L_id_ = 0;                    ///< the live room the robot came from
    std::string ep_L_mem_;                      ///< its memory name
    std::string start_;                         ///< path start room (memory) and the passages crossed to L
    std::vector<std::string> path_;
    std::optional<ChainPose> fresh_;            ///< the fresh seam T(L <- P), cached while measurable (L is gone after promotion)
    bool swap_ack_ = false;                     ///< RoomSwap executed the Swap: the episode is held, not judged
    bool promoted_ = false;                     ///< the provisional room became current (or the committed room did)
    std::vector<rc::recog::Hypothesis> post_entry_;
    std::vector<float> published_;              ///< the last recog_posterior written (skip identical rewrites)
    std::vector<std::uint64_t> published_ids_;
    std::uint64_t node_id_ = 0;

    void end_episode();
    Step held_step(Step out);                   ///< an executed swap: hold until the fence
    Step abort_episode(Step out);               ///< end an episode; Undo is requested when a commit stood
    void track_current();
    void publish(const std::vector<float> &post);
};
}   // namespace ltsm
