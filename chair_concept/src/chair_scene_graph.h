/*
 * chair_scene_graph.h
 *
 * DSR node/RT I/O layer for chair_concept (mirrors bottle_concept/bottle_scene_graph.h):
 * scaffolds missing "chair_N" nodes from priors matched to masks, writes the fitted model
 * back (geometry attrs + mesh + RFE/voxel-bank export + room→chair RT edge), reads the
 * robot localisation covariance, and writes the epistemic action proposal.
 *
 * Plain class (no Q_OBJECT) constructed by SpecificWorker once G + the DSR APIs are ready.
 * The graph relayout is injected as a callback to stay decoupled from the GUI (graph_viewers).
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_rt_api.h>

#include "chair_config.h"       // rc::ChairConfig
#include "chair_instance.h"     // rc::ChairInstance, ChairState
#include "../../common/mask_ingestor/mask_ingestor.h"      // MaskIngestor::MasksPacket
#include "epistemic_planner.h"  // EpistemicProposal

namespace rc {

class ChairSceneGraph
{
public:
    ChairSceneGraph(std::shared_ptr<DSR::DSRGraph> graph,
                    DSR::RT_API* rt_api,
                    const ChairConfig& cfg,
                    std::function<void()> relayout);

    // Data-driven birth: create a fresh "chair_N" node at the detection centroid (room frame), seeded
    // with the Tracker.Birth* default geometry. Returns the new node id (0 on failure). Used by the
    // worker's instance tracker in place of the prior-scaffold when Tracker.Enabled.
    std::uint64_t create_instance_from_detection(const Eigen::Vector3f& centroid_room,
                                                 std::uint64_t room_node_id);

    // Publish the instance's fitted model to its DSR node (geometry + FE + mesh + RFE/voxel-bank) and
    // the room→chair RT edge. persist_* resolves the node by id first; both no-op if the node is gone.
    bool persist_chair_belief(ChairInstance& inst, std::uint64_t node_id, std::uint64_t room_id, float free_energy);
    void step_write_model(ChairInstance& inst, DSR::Node& node, std::uint64_t room_id, float free_energy);
    void write_rt_pose(std::uint64_t room_id, ChairInstance& inst);

    // Attach the chair pose covariance (rt_covariance_att, 6×6 SE3) on the room→chair RT edge, mapped
    // from the belief's full Σ. Writes when `force` (a geometry republish)
    // OR the covariance trace changed meaningfully since last write — so a stationary-but-tightening
    // chair stays current without per-cycle edge churn. No-op if disabled / the edge is absent.
    void write_rt_covariance(std::uint64_t room_id, ChairInstance& inst, bool force);

    // Write the epistemic next-best-view proposal onto a chair node.
    void write_epistemic_proposal(DSR::Node& node, const EpistemicProposal& prop);

    // Flat triangle list (room frame): top slab + 4 legs.
    static std::vector<float> make_chair_mesh(const ChairState& s);

private:
    void write_chair_mesh(ChairInstance& inst, DSR::Node& node);

    std::shared_ptr<DSR::DSRGraph> G_;
    DSR::RT_API*          rt_api_ = nullptr;
    const ChairConfig&    cfg_;
    std::function<void()> relayout_;
};

}  // namespace rc
