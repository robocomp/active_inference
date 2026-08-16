/*
 * table_fitter.h — the active-inference fit core of table_concept (mirrors bottle_concept/bottle_fitter.h).
 *
 * Owns the per-table instance map and runs the AI2 full-covariance belief update for each "table_*" node:
 * instance lifecycle (ensure_instance + the TableModel factory), observation (split the selected mask's
 * support points into on-surface vs off-surface sets), inference (support-bank ingest + one recursive TableBelief
 * update with the mask-motion channel as the observation precision R / bias gate, written back into inst.model),
 * and the table-owned support-point memory (ownership gate + FNV cell keys). Collaborates with MaskIngestor (masks),
 * TableSceneGraph (robot covariance), TableProjection (camera projection), and TableLidarRangeChannel;
 * SpecificWorker keeps the orchestration (process_table_node), the DSR write-back, and the post-fit
 * epistemic / affordance / Qt-diagnostics steps. Plain class (no Q_OBJECT).
 */

#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/api/dsr_inner_gaussian_api.h>   // Part B: chain covariance propagation
#include <dsr/api/dsr_camera_api.h>

#include "table_config.h"        // rc::TableConfig
#include "table_instance.h"      // rc::TableInstance, TableState
#include "table_model.h"         // TableModel / TableModelParams
#include "table_projection.h"    // rc::TableProjection, rc::SilhouetteExistence
#include "table_lidar_range_channel.h"   // rc::TableLidarRangeChannel
#include "../../common/mask_ingestor/mask_ingestor.h"
#include "table_scene_graph.h"

namespace rc {

class TableFitter
{
public:
    struct TableObservation
    {
        bool has_fresh_data = false;
        std::vector<Eigen::Vector3f> candidate_pts;
        std::vector<Eigen::Vector3f> residual_pts;
        // Ricoh slices are bearing-only and never fitted (see observe_slice / process_ricoh_bearings): a
        // peripheral 360 detection has a reliable DIRECTION but a biased centroid/extent, so it only drives
        // the attention path and never contributes candidate/residual points to a belief update.
    };

    TableFitter(std::shared_ptr<DSR::DSRGraph> graph,
                DSR::InnerEigenAPI* inner_eigen,
                TableConfig& cfg,
                MaskIngestor* mask_ingestor,
                TableSceneGraph* scene_graph);

    // Create the instance for a "table_*" node if absent (from prior/RT). Returns true the
    // first time it is created (so the worker can register Qt series / canvas pos). Latches room_id.
    bool ensure_instance(const DSR::Node& node, std::uint64_t room_id);

    // Build an observation from ONE assigned mask slice (index into the current packet): candidate/residual
    // split against the current SDF + this slice's R inputs (motion_var/depth_var/range) latched on the
    // instance. Called once per slice in the same cycle for multi-sensor fusion (ZED + ricoh). Sets
    // frames_since_detection=0 (a detection is alive). Does NOT gate on frame_id — the caller owns freshness.
    TableObservation observe_slice(TableInstance& inst, int slice_index);
    // Fallback path when the tracker assigned NO mask slice this cycle: candidate/residual point attributes
    // written directly on the node (legacy), else a stale observation (has_fresh_data=false → age the belief).
    TableObservation observe(TableInstance& inst, const DSR::Node& node);
    // One recursive full-covariance belief update (TableBelief) on this frame's mask points, with the
    // mask-motion channel as the observation precision R / bias gate. Writes the result into inst.model
    // so all downstream publish/viewer code is unchanged. Returns the update free energy.
    float run_inference(TableInstance& inst, const TableObservation& observation);

    std::unordered_map<std::uint64_t, TableInstance>& instances() { return instances_; }
    void forget_node(std::uint64_t id) { instances_.erase(id); }
    // Part B (chain covariance): enable adding the localization/chain term J·Σ_chain·Jᵀ (measurement
    // frame → room, capture-stamp pinned) to each instance, read by the scene-graph's RT-cov write.
    void set_chain_cov_source(DSR::InnerGaussianAPI* gaussian, std::string source_frame);
    // Object-anchor observation: publish this frame's ROBOT-frame fit (z_o) so the room localizer can
    // use the table as an SE(2) pose landmark. robot_frame = the localizer's base node ("body"). OFF unless enabled.
    void set_object_observation(bool enabled, std::string robot_frame);
    // Room-frame XY a NEWLY born instance's model should start at (from the tracker's detection). The
    // room→table RT edge written at birth is NOT reliably queryable in the same cycle, so ensure_instance
    // would read the 0,0 default and the model would freeze there. Consumed once by ensure_instance.
    void note_birth(std::uint64_t id, const Eigen::Vector2f& xy) { birth_seeds_[id] = xy; }
    bool should_log(const TableInstance& inst) const;

    // May THIS mask frame move geometry? — the agent's UPDATE-admissibility predicate, evaluated on a raw slice
    // that has no instance yet. rc::birth (common/instance_tracker/birth_evidence.h) rule 2: a frame that may not
    // MOVE an existing table's geometry must not CREATE one, so birth is admitted by the same predicate the fit
    // uses — the FIXATION gate — rather than a second, weaker birth-only notion of "good enough". Mirrors
    // RefrigeratorFitter::frame_admissible; the conditions here are table_fitter.cpp's fixation block.
    bool frame_admissible(const MaskIngestor::MaskSlice& sl) const;

    // Emit a CSV row for an OUT-OF-VIEW instance whose EXISTENCE evidence integrated this cycle (LiDAR/silhouette
    // carve) but which the fitter did NOT fit (no fresh mask) — so the existence log-odds trajectory that drives a
    // removal is captured even across the silent out-of-FoV stretch where no fit row is written. Fit fields are the
    // last converged values (held); the existence columns are FRESH (integrated just before this call). Called from
    // TableExistence::update_and_remove for non-observed instances only, so it never double-logs a fit row.
    void log_existence_cycle(const TableInstance& inst)
    { log_ai2_csv(inst, 0, inst.dbg_R, /*gated=*/true, inst.dbg_energy); }

    // YOLO-independent LiDAR range channel: stage this cycle's sweep (room frame) + sensor origin for the
    // range factor. clear_lidar_sweep() each cycle first so a stale sweep never leaks into a frame with no
    // fresh LiDAR. Set/cleared from the compute() main thread by SpecificWorker (fed by ConceptLidarIngestor).
    void set_lidar_sweep(const std::vector<Eigen::Vector3f>& sweep_room, const Eigen::Vector3f& origin_room)
    {
        lidar_channel_.set_sweep(sweep_room, origin_room);
        // Same sweep also arms the projection's line-of-sight oracle, so the silhouette existence channel can
        // tell "I looked and it is gone" from "a WALL is in the way" — YOLO masks alone cannot (see
        // TableProjection::set_lidar_los). Fed on the compute() main thread, cleared with the sweep.
        projection_->set_lidar_los(sweep_room, origin_room, cfg_.existence_los_margin_m, cfg_.existence_los_azim_bins);
    }
    void set_lidar_sweep_bpearl(const std::vector<Eigen::Vector3f>& sweep_room, const Eigen::Vector3f& origin_room)
    { lidar_channel_.set_sweep_bpearl(sweep_room, origin_room); }
    // Room walls → the projection's line-of-sight test, so a silhouette sample behind a wall is NOT counted
    // "predicted visible". Mirrors RefrigeratorFitter::set_room_geometry. Empty ⇒ test inactive.
    void set_room_polygon(std::vector<Eigen::Vector2f> poly)
    { if (projection_) projection_->set_room_polygon(std::move(poly)); }
    void clear_lidar_sweep()
    {
        lidar_channel_.clear();
        projection_->set_lidar_los({}, Eigen::Vector3f::Zero(), cfg_.existence_los_margin_m, cfg_.existence_los_azim_bins);
    }

    // PIXEL-LEVEL silhouette existence evidence (EXISTENCE_BELIEF_PLAN.md, mask channel). Delegates to
    // TableProjection; see rc::SilhouetteExistence in table_projection.h. Called from update_existence.
    SilhouetteExistence compute_silhouette_existence(const TableInstance& inst)
    { return projection_->compute_silhouette_existence(inst); }

    // Unassigned peripheral detections this cycle, for the ai2 log. Set by the worker after
    // process_ricoh_bearings; the fitter owns the CSV, so the count has to reach it.
    void set_ricoh_attention(int n) { ricoh_attention_ = n; }

private:
    int ricoh_attention_ = 0;
    // Compute the localization/chain covariance term (J·Σ_chain·Jᵀ) at the table centre by transforming
    // it from the measurement frame back to room with ZERO input cov; stored on the instance for the
    // RT-cov write. No-op unless set_chain_cov_source enabled it.
    void compute_chain_cov(TableInstance& inst);
    // Periodic round-vs-square shape model-selection on the accumulated support bank → inst.subtype
    // (bounded log-Bayes-factor, no threshold). Runs every cfg_.shape_eval_period cycles.
    void evaluate_shape(TableInstance& inst);

    TableModelParams  make_model_params() const;

    // Append one AI2 belief row (state + Σ diag std + mask R/bias/trunc + gate flag) to cfg_.ai2_csv_path.
    void log_ai2_csv(const TableInstance& inst, int point_count, float R, bool gated, float energy);

    std::shared_ptr<DSR::DSRGraph> G_;
    DSR::InnerEigenAPI*            inner_eigen_ = nullptr;
    DSR::InnerGaussianAPI*         gaussian_    = nullptr;   // Part B: chain covariance (set_chain_cov_source)
    std::string                    chain_src_frame_;          // measurement frame the chain cov is computed from
    bool                           chain_cov_enabled_ = false;
    bool                           obs_enabled_ = false;       // publish z_o (object-anchor observation)?
    std::string                    obs_robot_frame_ = "body";  // localizer base frame z_o is expressed in
    std::string                    obs_cam_frame_ = "zed";     // camera frame the raw mask cloud lives in
    void compute_object_observation(TableInstance& inst);      // fill inst.obs_robot (gated)
    TableConfig&                   cfg_;
    MaskIngestor*                  mask_ingestor_ = nullptr;
    TableSceneGraph*               scene_graph_   = nullptr;
    std::unique_ptr<TableProjection> projection_;   // camera-projection unit (owns the ZED CameraAPI)

    std::unordered_map<std::uint64_t, TableInstance> instances_;
    std::unordered_map<std::uint64_t, Eigen::Vector2f> birth_seeds_;   // tracker-provided birth XY (see note_birth)
    std::uint64_t                  room_node_id_ = 0;   // latched per ensure_instance call
    std::ofstream                  ai2_csv_;            // per-cycle AI2 belief log (optional)

    // YOLO-independent LiDAR range channel: owns the staged per-cycle sweep + return selection (feed).
    TableLidarRangeChannel         lidar_channel_{cfg_};
};

}  // namespace rc
