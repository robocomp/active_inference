#pragma once
/*
 * room_proposal.h -- recognition PROPOSES (room-transition protocol, spec case 2 step 6; plan Task 5).
 *
 * When the recogniser is convinced that the provisional room P is a remembered room K, ltsm does NOT touch P (rule 1: P is
 * room_concept's). It writes ONE node, owned by ltsm, that room_concept reads and answers:
 *
 *   proposal_<K live name>   type metaconcept, object_subtype "room_proposal"
 *     parent              = P's id, level = P's + 1, RT edge P -> proposal holding T(P <- K)
 *     room_id             = K's room_id
 *     delimiting_polygon_x/y = K's remembered outline (K's own frame)
 *     recall_pose_prior   = the robot in K, the recogniser's modes (10 floats each: w x y th Sxx Sxy Sxth Syy Syth Sthth)
 *     state               = "pending" (ltsm) -> "accepted" | "rejected" (room_concept)
 *
 * No new attribute: all of these exist (spec "Settled after the builder check": No new attributes).
 *
 * ONE PROPOSAL AT A TIME (Review Focus 2). post() deletes the previous proposal it made, and any other room_proposal hanging
 * from the same P, before it writes the new one: a superseded proposal is never left for room_concept to answer. A fresh node
 * (fresh id) every post, so a verdict written on an older one can never be read as the answer to the newer.
 *
 * VERDICT. Read from the graph each cycle (poll; no DSR signal connected): the `state` attribute, or Gone when the proposal
 * or P has left the live graph (someone deleted it, or room_concept dropped P). ltsm deletes its proposal itself (withdraw)
 * when the verdict has been consumed, when the robot leaves P before a verdict (Review Focus 4), on a graceful exit; and at
 * start any leftover from a crashed run (sweep_stale).
 */
#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ltsm
{
class RoomProposal
{
public:
    RoomProposal(std::shared_ptr<DSR::DSRGraph> live, std::shared_ptr<DSR::DSRGraph> mem);

    /// Propose "P is the memory room K_mem": `P_from_K` = T(P <- K), `prior` = the packed robot-in-K modes. Replaces the
    /// previous proposal (this one's, and any other room_proposal under P). Returns the new node id, 0 on failure.
    std::uint64_t post(std::uint64_t P, const std::string &K_mem, const Mat::RTMat &P_from_K, const std::vector<float> &prior);

    enum class Verdict { Pending, Accepted, Rejected, Gone };
    /// room_concept's answer to the standing proposal; Gone when none stands, or it or P has left the live graph.
    [[nodiscard]] Verdict verdict() const;
    /// Delete the standing proposal (ltsm owns it). Idempotent.
    void withdraw();

    [[nodiscard]] std::uint64_t id() const { return id_; }
    [[nodiscard]] std::uint64_t provisional() const { return P_; }

    /// Startup sweep and graceful exit: every room_proposal node in the live graph (ltsm owns them all). Returns how many went.
    int sweep_stale();

    static constexpr const char *kSubtype = "room_proposal";
    static constexpr const char *kPending = "pending";
    static constexpr const char *kAccepted = "accepted";
    static constexpr const char *kRejected = "rejected";

private:
    std::shared_ptr<DSR::DSRGraph> live_, mem_;
    std::uint64_t id_ = 0;      ///< the standing proposal (0 = none)
    std::uint64_t P_ = 0;       ///< the provisional room it hangs from
};
}   // namespace ltsm
