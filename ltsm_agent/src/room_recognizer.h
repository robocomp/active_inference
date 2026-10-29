#pragma once
/*
 * room_recognizer.h -- "which known room have I just walked into?", asked while the room behind a FRESH door is still
 * provisional. Evidence in, posterior out, a staged commit state; it only REQUESTS a swap or an undo (RoomSwap executes).
 *
 * WHEN IT RUNS (structural, nothing thresholded). Exactly while all of these hold:
 *   - the live graph has a CURRENT room L that memory holds a twin of, and a PROVISIONAL room P hung from the robot (a
 *     proto-room, or the hand-over case promotion_evidence also accepts: the robot->L edge retired while robot->P is
 *     live). L may itself be the promoted room of the previous episode (fresh -> fresh crossings);
 *   - P carries an entry mirror of a door of L (door_concept: a door node whose parent is P);
 *   - that crossed door has NO passage in memory (a door crossed before is a known passage; recall is trusted there,
 *     user decision 2026-10-23: no audit of known passages).
 * It is BEGUN only while these hold (it then spans P's promotion, see below); it stops, and its `recognition` node is
 * removed, when P is lost before promotion, at the fence, or when a newer provisional room appears.
 *
 * EVIDENCE. Two experts over the SAME pose in room k (rc::recog::posterior):
 *   metric       the robot pose through the seam chain vs room k's own chain (room_chain.h). The robot->P edge gives the
 *                robot in P with its covariance; the FRESH seam T(L <- P) is built HERE from the crossed door's two LIVE
 *                faces: D in L (its RT chain + covariance, door_concept's) and its entry mirror M in P. Nothing about D
 *                has to be archived (a first-ever door has no memory face). FRAMES: the live L frame IS its memory twin's
 *                (the twin is a copy of L, written at L's promotion or by the seed), so the memory path chain
 *                start -> ... -> L_mem continues with T(L <- P) and then robot-in-P; every other room k is placed by its
 *                own memory chain from the same start. The fresh seam is cached while L is live (it is evicted when P is
 *                promoted). A state, read again each cycle; the decision uses its change since the last state change.
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
 * ends at the eviction fence (the robot leaves it) or when a NEWER provisional room hangs from the robot (a new crossing
 * out of the room the episode was about: that crossing's episode replaces it -- in the same cycle, or in the next when
 * ending the old one requests an Undo). Ending any other way
 * under an unexecuted commit, or losing the committed room from the candidates, emits an Undo request; an executed
 * (held) swap ends with no Undo, as at the fence.
 *
 * VERDICT. An executed Swap posts a PROPOSAL (room_proposal.h) and the episode is HELD until room_concept answers. Accepted:
 * held until the robot leaves the adopted room. Rejected (or withdrawn): on_proposal_rejected() -- judging resumes with the
 * evidence rebased at the rejection, so only evidence gathered after it can propose again.
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
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
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
        /// An EXECUTED (held) swap's episode ended this cycle (the fence, or a newer provisional room): the undo window is
        /// closed. The caller closes RoomSwap's window from this (on_room_left), BEFORE executing `act`, which may already
        /// belong to the next episode. One-shot: reported on exactly one Step.
        bool released = false;
    };

    /// One cycle. Swap/Undo is a REQUEST, executed by RoomSwap.
    Step step();

    /// ACKNOWLEDGEMENTS from RoomSwap (a request is not an execution).
    /// A requested Swap was executed (its proposal posted): from here the graph changes that follow it (room_concept adopting
    /// P in place, the crossed door gaining a passage) are NOT episode loss -- the episode is HELD (not judged) until the
    /// verdict comes back rejected (on_proposal_rejected), the robot leaves the committed room (the eviction fence), or an
    /// Undo is executed.
    void on_swap_executed(std::uint64_t room_id);
    /// A requested Swap failed: the commit reverts to Entering (no Undo is requested: nothing was done).
    void on_swap_failed();
    /// A requested Undo was executed: the swap no longer stands.
    void on_undo_executed();
    /// room_concept REJECTED the proposal (or it was withdrawn: P gone, the robot left P before a verdict). The swap no longer
    /// stands: back to Entering, unheld, and the evidence is REBASED at the rejection -- the next posterior counts only what
    /// arrives after it (spec: rejection is per proposal; ltsm may propose again with NEW evidence).
    void on_proposal_rejected();
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
    std::map<std::int64_t, float> metric_filt_;        ///< the metric log-evidence per hypothesis (-1 = new), filtered over the episode
    std::int64_t metric_filt_ms_ = 0;                  ///< wall time of the last filter update (0 = none this episode)
    std::uint64_t L_id_ = 0;                    ///< the live room the robot came from
    std::string ep_L_mem_;                      ///< its memory name
    std::string start_;                         ///< path start room (memory) and the passages crossed to L
    std::vector<std::string> path_;
    std::optional<ChainPose> fresh_;            ///< the fresh seam T(L <- P), cached while measurable (L is gone after promotion)
    bool swap_ack_ = false;                     ///< RoomSwap executed the Swap: the episode is held, not judged
    bool rebase_next_ = false;                  ///< a rejection: the next evidence frame becomes the base (delta 0)
    bool promoted_ = false;                     ///< the provisional room became current (or the committed room did)
    std::vector<rc::recog::Hypothesis> post_entry_;
    bool trace_tried_ = false;                  ///< the trace file is opened once per process, never retried
    std::ofstream trace_;                       ///< etc/recognition_trace.csv, opened on the first step
    std::vector<float> published_;              ///< the last recog_posterior written (skip identical rewrites)
    std::vector<std::uint64_t> published_ids_;
    std::uint64_t node_id_ = 0;

    bool released_ = false;                     ///< a held episode ended since the last Step was returned
    Step step_impl();
    void end_episode();
    Step held_step(Step out);                   ///< an executed swap: hold until the fence
    Step abort_episode(Step out);               ///< end an episode; Undo is requested when a commit stood
    /// abort_episode + ONE log line when the reason a provisional room is not eligible CHANGES (not every cycle).
    Step ineligible(Step out, const std::string &room, std::string_view reason);
    std::string verdict_;                       ///< the last eligibility verdict logged ("" = none)
    void track_current();
    void publish(const std::vector<float> &post);
    /// The provisional room hung from `robot` (a proto, or the hand-over case: live while robot->current is retired), other
    /// than the current room `cur`, the live node `except_node` (0 = none) and a room carrying `except_room_id`; with the
    /// robot->P edge. The FIRST such room in edge order (excluding by node, not by order, is what makes a lookup for "another
    /// provisional room" independent of that order).
    [[nodiscard]] std::optional<std::pair<DSR::Node, DSR::Edge>> provisional_room(const DSR::Node &robot, std::uint64_t cur,
                                                                                  std::optional<std::uint64_t> except_room_id,
                                                                                  std::uint64_t except_node = 0) const;
};
}   // namespace ltsm
