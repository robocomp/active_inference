/*
 * ring_scene_graph.h — ALL DSR writes for the ring_metaconcept agent (level-2).
 *
 * Mirrors the level-1 `<obj>_scene_graph` role, with the data direction inverted: the level-1 units
 * publish what a sensor saw, this one publishes a LATENT fitted from peers plus the top-down message
 * that latent sends back down to them.
 *
 * Two things are written, and the split matters:
 *   · the `dining_set_*` NODE — generic type `object`, class in `object_subtype`, holding the
 *     arrangement latent (centre via its room→rig RT edge, radius, slot count, log-odds);
 *   · a non-RT `group_member` EDGE rig→member per constituent, carrying that member's empirical
 *     prior (rig_yaw_prior / rig_yaw_kappa / …).
 *
 * ★The message rides the EDGE, never the member's node. CRDTSyncEngine::update_node_raw resets any
 * attribute present in a node's local registry but absent from the submitted copy, so a member agent
 * doing get_node → modify → update_node with a slightly stale copy would silently DELETE a prior
 * written onto its node. This agent is the sole writer of the edge, so on the edge it is safe.
 * (Single-writer discipline, same as the RT chain covariance.)
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>
#include <dsr/api/dsr_api.h>

#include "ring_belief.h"
#include "ring_config.h"

namespace rc
{

// The per-member top-down message. Assembled by the worker from the CAVITY fit (the arrangement
// refitted WITHOUT this member) — never from the full fit, or the rig feeds a member its own output.
struct MemberPrior
{
    std::uint64_t   member_id  = 0;
    int             slot_index = -1;
    float           yaw_prior  = 0.0f;   // rad, in the MEMBER's own yaw convention
    float           yaw_kappa  = 0.0f;   // rad⁻²; 0 ⇒ inert, the consumer ignores it
    Eigen::Vector2f slot_xy    = Eigen::Vector2f::Zero();   // Phase 2 radial prior
    float           info_xx    = 0.0f;
    float           info_yy    = 0.0f;
};

class RingSceneGraph
{
public:
    RingSceneGraph(std::shared_ptr<DSR::DSRGraph> G, DSR::RT_API* rt_api, const RingConfig& cfg);

    std::uint64_t rig_node_id() const { return rig_node_id_; }

    // Create the rig node if absent, else refresh its latent. Returns its id (0 on failure).
    std::uint64_t ensure_rig_node(const RingBelief& belief, std::uint64_t room_id);
    // Delete the rig node (its outgoing group_member edges go with it) — the arrangement stopped
    // explaining the scene, or we are shutting down.
    void          remove_rig_node();

    // Write / refresh one member's prior on the rig→member group_member edge. Self-gates: an
    // unchanged message is not rewritten, so a settled arrangement does not churn the graph.
    void          publish_member_prior(const MemberPrior& prior);
    // A member that is no longer part of the rig: zero its precision so a stale prior cannot linger
    // in a consumer that reads before noticing the edge is gone, THEN drop the edge.
    void          drop_member(std::uint64_t member_id);
    // Drop every member not in `keep` (departed chairs).
    void          retain_members(const std::unordered_set<std::uint64_t>& keep);

private:
    std::shared_ptr<DSR::DSRGraph> G_;
    DSR::RT_API*                   rt_api_ = nullptr;
    RingConfig                     cfg_;

    std::uint64_t                          rig_node_id_ = 0;
    std::unordered_set<std::uint64_t>      members_;        // members we currently hold an edge to
    // Last published message per member, for the self-gate.
    std::unordered_map<std::uint64_t, MemberPrior>    last_published_;
    // Wall-clock ms of the last actual write per member — drives the liveness heartbeat that keeps
    // the payload self-gate from starving the consumer's staleness check.
    std::unordered_map<std::uint64_t, std::uint64_t>  last_publish_ms_;

    float last_pub_radius_ = -1.0f;
    int   last_pub_slots_  = -1;
};

}  // namespace rc
