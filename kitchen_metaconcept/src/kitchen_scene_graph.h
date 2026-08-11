/*
 * kitchen_scene_graph.h — ALL DSR writes for kitchen_metaconcept.
 *
 * Mirrors the level-1 `<obj>_scene_graph` role with the data direction inverted: the level-1 units
 * publish what a sensor saw, this publishes a latent fitted from peers plus the message that latent
 * sends back down to them.
 *
 * Two things are written, and the split matters:
 *   · the `kitchen_*` NODE — DSR type `metaconcept` (NOT `object`, or every consumer's
 *     get_nodes_by_type("object") sweep would draw it as a solid, carve it out of the residual grid
 *     and try to associate against it), holding the frame latent;
 *   · a non-RT `group_member` EDGE kitchen→member per constituent, carrying that member's prior.
 *
 * ★The message rides the EDGE, never the member's node. CRDTSyncEngine::update_node_raw resets any
 * attribute present in a node's local registry but absent from the submitted copy, so a member agent
 * doing get_node → modify → update_node with a slightly stale copy would silently DELETE a prior
 * written onto its node. This agent is the sole writer of the edge, so on the edge it is safe.
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

#include "kitchen_belief.h"
#include "kitchen_config.h"

namespace rc {

// The per-member top-down message. Assembled from the CAVITY fit and, for the ends, from the
// continuous-outline geometry. Every field is optional: an `info` of 0 means "nothing to say about
// this DOF", and the consumer must treat it as inert rather than as a zero-valued prior.
struct KitchenMemberPrior
{
    std::uint64_t member_id = 0;

    float yaw   = 0.0f, yaw_kappa = 0.0f;      // grid axis resolved to THIS member's own arm (rad, rad⁻²)
    // Where each end should sit for the shape to be continuous, room frame, with its precision (m⁻²).
    // "lo" is the end at −u from the centre, "hi" the end at +u; the member projects each onto its
    // own chart, so nothing here assumes how the member parameterises itself.
    Eigen::Vector2f end_lo{0, 0};  float end_lo_info = 0.0f;
    Eigen::Vector2f end_hi{0, 0};  float end_hi_info = 0.0f;
};

class KitchenSceneGraph
{
public:
    KitchenSceneGraph(std::shared_ptr<DSR::DSRGraph> G, DSR::RT_API* rt_api, const KitchenConfig& cfg);

    std::uint64_t node_id() const { return node_id_; }

    // Create the kitchen node if absent, else refresh its latent. Returns its id (0 on failure).
    std::uint64_t ensure_node(const KitchenBelief& belief, std::uint64_t room_id);
    // Remove it — the arrangement stopped explaining the scene, or we are shutting down.
    void          remove_node();

    // Write / refresh one member's prior on the kitchen→member group_member edge. Self-gates: an
    // unchanged message is not rewritten, so a settled kitchen does not churn the graph. A liveness
    // heartbeat still refreshes the stamp, or the consumer's staleness check would starve.
    void publish_member_prior(const KitchenMemberPrior& p);
    // A member that has left: zero its precisions FIRST so a stale prior cannot linger in a consumer
    // that reads before noticing the edge is gone, THEN drop the edge.
    void drop_member(std::uint64_t member_id);
    void retain_members(const std::unordered_set<std::uint64_t>& keep);

private:
    std::shared_ptr<DSR::DSRGraph> G_;
    DSR::RT_API*                   rt_api_ = nullptr;
    KitchenConfig                  cfg_;

    std::uint64_t                                        node_id_ = 0;
    std::unordered_set<std::uint64_t>                    members_;
    std::unordered_map<std::uint64_t, KitchenMemberPrior> last_published_;
    std::unordered_map<std::uint64_t, std::uint64_t>      last_publish_ms_;
};

}  // namespace rc
