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

// Identifies WHICH rig a call is about: the owning RingHypothesis::key. Deliberately not the DSR node
// id — a hypothesis exists before its node is born and can outlive it by a few cycles, so keying on
// the node id would lose the very state the association step exists to preserve.
using RigKey = std::uint64_t;

class RingSceneGraph
{
public:
    RingSceneGraph(std::shared_ptr<DSR::DSRGraph> G, DSR::RT_API* rt_api, const RingConfig& cfg);

    // 0 when this rig has no node (never born, or retired).
    std::uint64_t rig_node_id(RigKey key) const;
    // The DSR node name, or "-" when unborn. For the diagnostics CSV.
    std::string   rig_node_name(RigKey key) const;

    // Create this rig's node if absent, else refresh its latent. Returns its id (0 on failure).
    std::uint64_t ensure_rig_node(RigKey key, const RingBelief& belief, std::uint64_t room_id);
    // Delete this rig's node (its outgoing group_member edges go with it) — the arrangement stopped
    // explaining the scene, or we are shutting down.
    void          remove_rig_node(RigKey key);
    // Every rig at once. Shutdown path: each removal is logged individually, and the prefix sweep in
    // the presence file then catches anything this map never knew about.
    void          remove_all_rig_nodes();

    // Write / refresh one member's prior on the rig→member group_member edge. Self-gates: an
    // unchanged message is not rewritten, so a settled arrangement does not churn the graph.
    void          publish_member_prior(RigKey key, const MemberPrior& prior);
    // A member that is no longer part of the rig: zero its precision so a stale prior cannot linger
    // in a consumer that reads before noticing the edge is gone, THEN drop the edge.
    void          drop_member(RigKey key, std::uint64_t member_id);
    // Drop every member of THIS rig not in `keep` (departed chairs).
    void          retain_members(RigKey key, const std::unordered_set<std::uint64_t>& keep);

private:
    // ★Every piece of per-rig state lives in here, not just the node id. Leaving any of the self-gate
    // maps global would make two rigs gate each other's writes: rig B's identical yaw would suppress
    // rig A's genuine update, and the heartbeat that proves liveness to the consumer would be shared
    // between nodes that must prove it separately.
    struct RigEntry
    {
        std::uint64_t node_id = 0;
        std::string   name;                                    // allocated once, at birth
        std::unordered_set<std::uint64_t>                members;
        std::unordered_map<std::uint64_t, MemberPrior>   last_published;
        std::unordered_map<std::uint64_t, std::uint64_t> last_publish_ms;
        float last_pub_radius = -1.0f;
        int   last_pub_slots  = -1;
    };

    std::shared_ptr<DSR::DSRGraph> G_;
    DSR::RT_API*                   rt_api_ = nullptr;
    RingConfig                     cfg_;

    std::unordered_map<RigKey, RigEntry> rigs_;
    // ★Monotonic name high-water mark. Scanning the graph alone is not enough: two rigs born in the
    // SAME cycle would both see the same max and collide, and a retired number handed to a different
    // rig silently transplants one arrangement's history onto another (the defect chair_concept fixed
    // in 9aef57a). A name, once used, is never reused for the lifetime of the process.
    int next_name_index_ = 0;
};

}  // namespace rc
