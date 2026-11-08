/*
 * table_scene_graph.h — DSR node/RT I/O layer for table_concept (mirrors bottle_concept/bottle_scene_graph.h).
 *
 * Births "table_N" nodes from tracker detections and writes the fitted model back: geometry attrs + mesh +
 * residual/support-bank export + free energy + the room→table RT pose, plus the 6×6 pose covariance (mapped from
 * the belief Σ, with a flat roll/pitch prior) and the epistemic action proposal. Plain class (no Q_OBJECT),
 * constructed by SpecificWorker once G + the DSR APIs are ready; the graph relayout is injected as a callback
 * to stay decoupled from the GUI (graph_viewers).
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_rt_api.h>

#include "table_config.h"       // rc::TableConfig
#include "table_instance.h"     // rc::TableInstance, TableState
#include "../../common/mask_ingestor/mask_ingestor.h"      // MaskIngestor::MasksPacket
#include "epistemic_planner.h"  // EpistemicProposal
#include <optional>
#include "../../common/recall_match/recall_match.h"   // rc::Resurrect
#include "../../common/instance_identity/instance_identity.h"   // rc::identity::InstanceCounter

namespace rc {

class TableSceneGraph
{
public:
    TableSceneGraph(std::shared_ptr<DSR::DSRGraph> graph,
                    DSR::RT_API* rt_api,
                    const TableConfig& cfg,
                    std::function<void()> relayout);

    // Birth a brand-new "table_N" node from an unexplained mask detection (tracker path). Auto-names one
    // past the highest existing table_N, seeds default geometry from the Tracker.Birth* config, anchors
    // it to the room at the detected room-frame centroid. Returns the new node id (0 on failure).
    // A resurrection (ltom recall): the newborn takes the remembered table's number and uid instead of issuing new ones.
    // Last number the persisted counter issued (opens the counter on first use).
    int identity_last();
    using Resurrect = rc::Resurrect;
    std::uint64_t create_instance_from_detection(const Eigen::Vector3f& centroid_room,
                                                 std::uint64_t room_node_id,
                                                 std::optional<Resurrect> r = std::nullopt);

    // Publish the instance's fitted model to its DSR node (geometry + FE + mesh + RFE/support-bank) and
    // the room→table RT edge. persist_* resolves the node by id first; both no-op if the node is gone.
    bool persist_table_belief(TableInstance& inst, std::uint64_t node_id, std::uint64_t room_id, float free_energy);
    void step_write_model(TableInstance& inst, DSR::Node& node, std::uint64_t room_id, float free_energy);
    void write_rt_pose(std::uint64_t room_id, TableInstance& inst);

    // Attach the table pose covariance (rt_covariance_att, 6×6 SE3) on the room→table RT edge, mapped
    // from the belief's full Σ over [cx,cy,H,w,h,yaw]. Writes when `force` (a geometry republish) OR the
    // covariance trace changed meaningfully since last write — so a stationary-but-tightening table stays
    // current without per-cycle edge churn. No-op until the belief is seeded / if the edge is absent.
    void write_rt_covariance(std::uint64_t room_id, TableInstance& inst, bool force);

    // Write the epistemic next-best-view proposal onto a table node.
    void write_epistemic_proposal(DSR::Node& node, const EpistemicProposal& prop);

    // Flat triangle list (room frame): top slab + 4 legs.
    static std::vector<float> make_table_mesh(const TableState& s);

private:
    void write_table_mesh(TableInstance& inst, DSR::Node& node);

    std::shared_ptr<DSR::DSRGraph> G_;
    DSR::RT_API*          rt_api_ = nullptr;
    const TableConfig&    cfg_;
    std::function<void()> relayout_;

    std::optional<rc::identity::InstanceCounter> identity_;   // persisted table_<N> numbers (common/instance_identity), opened at the first birth
};

}  // namespace rc
