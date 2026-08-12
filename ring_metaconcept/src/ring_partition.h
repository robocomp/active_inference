/*
 * ring_partition.h  —  WHICH members form a rig, and WHICH anchor is its centre
 *
 * The level-2 belief (ring_belief.h) answers "given these members, what arrangement explains them?".
 * It cannot answer "which members?" — and until now nobody did: the partition was silently fixed to
 * "every member of the ring class in the room", and the anchor was `anchors.front()`, i.e. whatever
 * order get_nodes_by_type() happened to return. With one table in the graph that is right by luck.
 * With two it is a coin flip, and losing it welds the ring centre onto the wrong table forever
 * (measured live 2026-08-11: centre pinned 5 m away, radius saturated at max_radius_m, log_odds
 * −3.2, so the rig node was never born). SCHEMA_GENERALITY_TODO.md §2.1 calls this a bug, not a gap.
 *
 * So the partition is part of the latent, and it is inferred here — by the SAME model comparison
 * the agent already uses one level up. Two objects are linked when
 *
 *      log p(separation | they belong to one rig)  −  log p(separation | independent objects)  >  0
 *
 * which is the "0 nats = the two hypotheses are equally likely" boundary, not a tuned cutoff. There
 * is deliberately NO linking distance in RingConfig: a metres-valued knob here would be exactly the
 * magic threshold CLAUDE.md forbids. Every quantity below is mirrored from RingBeliefParams or the
 * room polygon, so the partition and the arrangement cannot disagree about what a ring is.
 *
 * Two kernels, and the asymmetry between them is load-bearing:
 *   · member ↔ member — both occupy DISTINCT slots, so their separation is a CHORD of the seat ring;
 *   · anchor ↔ member — the anchor is AT the centre, so its separation is the RADIUS.
 * A single symmetric distance kernel gets one of the two wrong: on the live scene the chairs are
 * 0.905 m apart (a chord) while each sits 0.53 m from its table (a radius).
 *
 * The anchor is then attached by that likelihood — and, crucially, the prior's STRENGTH is the
 * attachment's own posterior probability (`Cluster::anchor_weight`). There is no accept/reject: a
 * table the chairs did not select is attached with weight ≈ 0 and is therefore inert, continuously.
 * That is what makes a mis-attachment structurally unable to weld the fit ever again.
 *
 * Pure Eigen — no DSR, no Qt — so it is unit-testable in isolation via partition_self_test().
 */

#pragma once

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include <Eigen/Dense>

namespace rc
{

// A candidate for the partition: identity, room-frame position, and the SCALAR position variance its
// own agent published (trace/2 of the 2×2 Σ — the same summary build_ring_frame already feeds the
// engine). Carrying the peer's own Σ here is what makes a poorly-localised member link weakly by
// construction, with no confidence gate: a wide Σ widens the kernel and flattens the log-Bayes factor.
struct PartitionMember
{
    std::uint64_t   id  = 0;
    Eigen::Vector2f xy  = Eigen::Vector2f::Zero();
    float           var = 1.0f;      // m², position variance published by the member's own agent
};

// Every value here is MIRRORED from RingBeliefParams / the room polygon. There is no partition-only
// tunable by design — see the header comment.
struct PartitionParams
{
    float prior_radius_m   = 0.80f;   // RingBeliefParams::prior_radius_m
    float prior_radius_std = 0.40f;   // RingBeliefParams::prior_radius_std
    float sigma_slot_m     = 0.20f;   // RingBeliefParams::sigma_slot_m
    float room_area_m2     = 50.0f;   // the null's support — from the room polygon, not a constant
};

// One spatial group of RING members, plus the anchor (if any) the model believes is its centre.
struct Cluster
{
    std::vector<std::uint64_t> member_ids;      // ★SORTED ascending — this is the association key
    std::vector<std::size_t>   member_index;    // parallel indices into the input `ring` vector

    std::uint64_t anchor_id      = 0;           // 0 ⇒ no anchor attached (a legal, fittable state)
    std::size_t   anchor_index   = 0;           // index into the input `anchors` vector
    float         anchor_logodds = -std::numeric_limits<float>::infinity();
    float         anchor_weight  = 0.0f;        // sigmoid(anchor_logodds) — SCALES the centre prior
};

// Result of matching this cycle's clusters onto the hypotheses that already exist.
struct Association
{
    std::vector<std::int64_t>  cluster_to_rig;   // per cluster: rig key, or -1 ⇒ unmatched (birth)
    std::vector<std::uint64_t> unmatched_rigs;   // rig keys with no cluster this cycle (hold, then retire)
};

// ── The three kernels ────────────────────────────────────────────────────────
// All are 1-D log densities in the separation `d` (m), so they are directly comparable. `var_pair`
// is the SUM of the two objects' published position variances.

// Two members on one ring occupy two distinct slots ⇒ d is a chord. Marginalised over the slot-count
// hypotheses and over which pair of slots, because at link time neither is known yet.
float log_pair_ring(float d, float var_pair, const PartitionParams& p);

// The anchor sits AT the centre ⇒ d is the ring radius.
float log_radius_ring(float d, float var_pair, const PartitionParams& p);

// Independent objects: two uniform draws over the room. The density of their separation is 2πd/A —
// the SAME uniform-over-the-room null RingBelief::null_nll uses one level up.
float log_pair_null(float d, const PartitionParams& p);

// Single-link over RING members using (log_pair_ring − log_pair_null > 0), then greedy 1-to-1 anchor
// attachment by the summed (log_radius_ring − log_pair_null) over each cluster's members. Anchors
// never link to each other and can never create a cluster: a table with no chairs around it belongs
// to nothing, which is what makes "there have to be chairs around" structural rather than a check.
std::vector<Cluster> partition_members(const std::vector<PartitionMember>& ring,
                                       const std::vector<PartitionMember>& anchors,
                                       const PartitionParams& p);

// Cluster ↔ existing-rig association by member-set overlap. `rigs` is (rig key, SORTED member ids).
// Max-overlap first with Jaccard as tiebreak: a rig that loses one of four chairs keeps its identity
// against the survivor cluster, and a cluster that splits hands its identity to the LARGER half.
// Without this the rig node births and dies every time a member flickers (the ★hazard in
// SCHEMA_GENERALITY_TODO.md §4 Step 1).
Association associate_clusters(const std::vector<Cluster>& clusters,
                               const std::vector<std::pair<std::uint64_t, std::vector<std::uint64_t>>>& rigs);

bool partition_self_test();

}  // namespace rc
