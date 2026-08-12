/*
 * ring_hypothesis.h  —  one spatial cluster's arrangement hypothesis
 *
 * Everything in here used to be a SINGLE member of SpecificWorker (`ring_belief_`, `ring_seeded_`,
 * `cavity_`), which is precisely what made "one rig, globally" structural rather than accidental
 * (SCHEMA_GENERALITY_TODO.md §2.1). One of these exists per cluster the partition proposes.
 *
 * `key` is a process-local counter and is NEVER reused. It is not the DSR node id: a hypothesis
 * exists before its node is born (while the evidence is still below zero) and survives a few cycles
 * after its members vanish, so the two lifetimes are genuinely different and conflating them would
 * reintroduce the churn the association step exists to prevent.
 */

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "ring_belief.h"

namespace rc
{

// The message sent DOWN to member i, computed from an arrangement refitted WITHOUT i. Moved out of
// SpecificWorker unchanged — see SpecificWorker::compute_cavity_priors for why the leave-one-out is
// not optional.
struct CavityPrior
{
    float yaw     = 0.0f;
    float std_dev = 0.0f;
    float shift   = 0.0f;   // how far the centre moved when i was removed — what i's own pose was worth
};

struct RingHypothesis
{
    std::uint64_t key = 0;              // stable, monotonic, never reused; printed in both CSVs
    RingBelief    belief;
    bool          seeded = false;

    std::vector<std::uint64_t> member_ids;      // ★SORTED — the association key and the re-seed trigger
    std::uint64_t              anchor_id     = 0;
    float                      anchor_weight = 0.0f;

    std::unordered_map<std::uint64_t, CavityPrior> cavity;

    // Consecutive cycles with no matching cluster. A hypothesis is HELD, not deleted, while this is
    // below the hold period — a member that flickers for one cycle must not kill the rig node.
    int miss = 0;
};

}  // namespace rc
