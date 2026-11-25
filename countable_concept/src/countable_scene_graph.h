/*
 * countable_scene_graph.h — DSR node/RT I/O layer for countable_concept (mirrors bottle_concept/bottle_scene_graph.h).
 *
 * Births "countable_N" nodes from tracker detections and writes the fitted model back: geometry attrs + mesh +
 * residual/support-bank export + free energy + the room→bed RT pose, plus the 6×6 pose covariance (mapped from
 * the belief Σ, with a flat roll/pitch prior) and the epistemic action proposal. Plain class (no Q_OBJECT),
 * constructed by SpecificWorker once G + the DSR APIs are ready; the graph relayout is injected as a callback
 * to stay decoupled from the GUI (graph_viewers).
 */

#pragma once
#include "countable_vertical_span.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_rt_api.h>

#include "countable_config.h"       // rc::CountableConfig
#include "countable_instance.h"     // rc::CountableInstance, CountableState
#include "../../common/mask_ingestor/mask_ingestor.h"      // MaskIngestor::MasksPacket
#include "epistemic_planner.h"  // EpistemicProposal
#include <optional>
#include "../../common/recall_match/recall_match.h"   // rc::Resurrect
#include "../../common/instance_identity/instance_identity.h"   // rc::identity::InstanceCounter

namespace rc {

class CountableSceneGraph
{
public:
    CountableSceneGraph(std::shared_ptr<DSR::DSRGraph> graph,
                    DSR::RT_API* rt_api,
                    const CountableConfig& cfg,
                    std::function<void()> relayout);

    // Birth a brand-new "countable_N" node from an unexplained mask detection (tracker path). Auto-names one
    // past the highest existing countable_N, seeds default geometry from the Tracker.Birth* config, anchors
    // it to the room at the detected room-frame centroid. Returns the new node id (0 on failure).
    // A resurrection (ltom recall): the newborn takes the remembered bed's number and uid instead of issuing new ones.
    // Last number the persisted counter issued (opens the counter on first use).
    int identity_last();
    using Resurrect = rc::Resurrect;
    std::uint64_t create_instance_from_detection(const Eigen::Vector3f& centroid_room,
                                                 std::uint64_t room_node_id,
                                                 std::optional<Resurrect> r = std::nullopt);

    // Publish the instance's fitted model to its DSR node (geometry + FE + mesh + RFE/support-bank) and
    // the room→bed RT edge. persist_* resolves the node by id first; both no-op if the node is gone.
    bool persist_countable_belief(CountableInstance& inst, std::uint64_t node_id, std::uint64_t room_id, float free_energy);
    void step_write_model(CountableInstance& inst, DSR::Node& node, std::uint64_t room_id, float free_energy);
    void write_rt_pose(std::uint64_t room_id, CountableInstance& inst);

    // Attach the bed pose covariance (rt_covariance_att, 6×6 SE3) on the room→bed RT edge, mapped
    // from the belief's full Σ over [cx,cy,H,w,h,yaw]. Writes when `force` (a geometry republish) OR the
    // covariance trace changed meaningfully since last write — so a stationary-but-tightening bed stays
    // current without per-cycle edge churn. No-op until the belief is seeded / if the edge is absent.
    void write_rt_covariance(std::uint64_t room_id, CountableInstance& inst, bool force);

    // Write the epistemic next-best-view proposal onto a bed node.
    void write_epistemic_proposal(DSR::Node& node, const EpistemicProposal& prop);

    // Flat triangle list (room frame): one solid box, floor-anchored or hanging (span).
    static std::vector<float> make_countable_mesh(const CountableState& s, rc::countable::VerticalSpan span = {});

private:
    void write_countable_mesh(CountableInstance& inst, DSR::Node& node);

    std::shared_ptr<DSR::DSRGraph> G_;
    DSR::RT_API*          rt_api_ = nullptr;
    const CountableConfig&    cfg_;
    std::function<void()> relayout_;

    std::optional<rc::identity::InstanceCounter> identity_;   // persisted countable_<N> numbers (common/instance_identity), opened at the first birth
};

}  // namespace rc
