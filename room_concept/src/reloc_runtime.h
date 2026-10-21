#pragma once
// The relocaliser's Task R1/R1b runtime state, one member of RoomConcept (rt_) — kept here so the agent's header does
// not grow with it. Localiser thread only, except `pending_passage` (RoomConcept::passage_mutex_).
#include <Eigen/Dense>
#include <cstdint>
#include <optional>
#include <vector>
#include "passage_prior.h"

namespace rc::reloc
{
    struct Runtime
    {
        /// The last search HELD the pose (msprt_decide → Hold): nothing decisive, the held pose alive; rivals carry it.
        bool last_search_held = false;
        /// A held episode is open: its rivals accumulate what the search could not decide; re-searching would re-seed
        /// them and discard that. Ends when they resolve, are rejected (lower SPRT bound) or empty, or P(lost) < ½.
        bool hold_open = false;
        std::uint64_t searches = 0;                              ///< searches fired by the lost belief (diagnostic)
        std::optional<rc::passage::Passage> pending_passage;     ///< set by note_passage_handover, consumed with the reset
        std::optional<rc::passage::Passage> passage;             ///< the passage of the last hand-over
        std::vector<rc::passage::Anchor>    door_anchor;         ///< forward / reversed components, carried by odometry
        Eigen::Matrix3f held_spread = Eigen::Matrix3f::Zero();   ///< while a held episode is open: Σ w_k d_k d_kᵀ
        int   door_anchor_frames = 0;                            ///< frames since: the kidnap hazard's exposure
        Eigen::Vector3f anchor_prev_est = Eigen::Vector3f::Zero();   ///< the pose the next prediction is built on
        bool  anchor_prev_valid = false;
        float passage_term = 0.f;                                ///< this frame's doorway term (0 = none)
    };

    /// A lost search's posterior over its modes and the held pose (RoomConcept::mode_posterior).
    struct ModePosterior { std::vector<double> log_post; int top_idx = 0, keep_idx = -1; bool keep_travelled = false; };
}   // namespace rc::reloc
