/*
 * table_fitter.h
 *
 * The active-inference core of table_concept (mirrors bottle_concept/bottle_fitter.h). Owns the
 * per-table instance map and runs the AI2 full-covariance belief update for each "table_*" node:
 *   - instance lifecycle (ensure_instance + the TableModel factory),
 *   - observation: split the selected mask's support points into on-surface vs off-surface sets,
 *   - inference: voxel-bank ingest + one recursive belief update (TableBelief) with the mask-motion
 *     channel as the observation precision R / bias gate, written back into inst.model,
 *   - the table-owned voxel memory (ownership gate + FNV voxel keys).
 *
 * Collaborates with MaskIngestor (masks) and TableSceneGraph (robot covariance). SpecificWorker
 * keeps the orchestration (process_table_node), the DSR write-back call, and the post-fit epistemic
 * / affordance / Qt-diagnostics steps. Plain class (no Q_OBJECT).
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
#include "../../common/mask_ingestor/mask_ingestor.h"
#include "table_scene_graph.h"

namespace rc {

class TableFitter
{
public:
    struct TableObservation
    {
        bool has_fresh_data = false;
        float explanation_ratio = 1.0f;
        std::vector<Eigen::Vector3f> candidate_pts;
        std::vector<Eigen::Vector3f> residual_pts;
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
    void set_chain_cov_source(DSR::InnerGaussianAPI* gaussian, std::string source_frame, bool enabled);
    // Room-frame XY a NEWLY born instance's model should start at (from the tracker's detection). The
    // room→table RT edge written at birth is NOT reliably queryable in the same cycle, so ensure_instance
    // would read the 0,0 default and the model would freeze there. Consumed once by ensure_instance.
    void note_birth(std::uint64_t id, const Eigen::Vector2f& xy) { birth_seeds_[id] = xy; }
    bool should_log(const TableInstance& inst) const;

    // YOLO-independent LiDAR range channel: stage this cycle's sweep (room frame) + sensor origin for the
    // range factor. clear_lidar_sweep() each cycle first so a stale sweep never leaks into a frame with no
    // fresh LiDAR. Set/cleared from the compute() main thread by SpecificWorker (fed by TableLidarIngestor).
    void set_lidar_sweep(const std::vector<Eigen::Vector3f>& sweep_room, const Eigen::Vector3f& origin_room)
    { lidar_sweep_room_ = sweep_room; lidar_origin_room_ = origin_room; lidar_have_sweep_ = true; }
    void clear_lidar_sweep() { lidar_have_sweep_ = false; }

    // PIXEL-LEVEL silhouette existence evidence (EXISTENCE_BELIEF_PLAN.md, mask channel). Projects the tabletop
    // TOP face + the 4 LEG axes onto the image and, over the predicted-VISIBLE pixels, counts how many are lit by a "table" YOLO
    // mask (e_occ ⇒ still there) vs by nothing at all (e_free ⇒ predicted-visible-but-ABSENT ⇒ evidence it is
    // gone, EVEN WITH NO YOLO MASK this frame). Pixels covered by a NON-table mask are OCCLUDED (a nearer object
    // hides the tabletop) and excluded from n_detectable → HOLD, never false absence. n_detectable==0 (out of
    // FoV / fully occluded) ⇒ the caller HOLDs. Feeds rc::exist::mask_evidence. Called from update_existence.
    struct SilhouetteExistence { float e_occ = 0.0f, e_free = 0.0f; int n_detectable = 0; };
    SilhouetteExistence compute_silhouette_existence(const TableInstance& inst);

private:
    // Compute the localization/chain covariance term (J·Σ_chain·Jᵀ) at the table centre by transforming
    // it from the measurement frame back to room with ZERO input cov; stored on the instance for the
    // RT-cov write. No-op unless set_chain_cov_source enabled it.
    void compute_chain_cov(TableInstance& inst);
    // room_T_zed (camera→room). pose_ts_ms pins the room→body hop to the mask's capture time (Nearest RT
    // query); the rigid body→zed mount is always queried latest. 0 → current pose.
    std::optional<Eigen::Matrix4d> room_T_zed_matrix(std::uint64_t pose_ts_ms = 0) const;
    // Project the current model through the camera extrinsic → normalised in-image ROI (centre
    // offset + fill), stored on the instance for the controller's centring/dwell lock-on search.
    void compute_projected_roi(TableInstance& inst);

    void ingest_observation_voxels(TableInstance& inst, const TableObservation& observation);
    bool is_voxel_owned_by_table(const TableInstance& inst, const Eigen::Vector3f& point) const;
    static std::uint64_t voxel_key(const Eigen::Vector3f& point, float quantization_m);

    // Select this cycle's LiDAR returns that land on THIS table and stage them on the frame's range channel.
    // Box is ANCHORED on the fresh mask-cloud centroid (never the fitted state — a diverging fit would drag
    // the box into empty space and starve LiDAR exactly when it is most needed) and SIZED from the BIRTH
    // footprint (not the fitted w/h, so a blown-up extent can't explode the region), deliberately spanning
    // legs + rim. Final on/off membership is the factor's own geometric sphere-trace test, not this box.
    void feed_lidar(TableInstance& inst, TableFrame& frame) const;

    TableModelParams  make_model_params() const;

    // Append one AI2 belief row (state + Σ diag std + mask R/bias/trunc + gate flag) to cfg_.ai2_csv_path.
    void log_ai2_csv(const TableInstance& inst, int point_count, float R, bool gated, float energy);

    std::shared_ptr<DSR::DSRGraph> G_;
    DSR::InnerEigenAPI*            inner_eigen_ = nullptr;
    DSR::InnerGaussianAPI*         gaussian_    = nullptr;   // Part B: chain covariance (set_chain_cov_source)
    std::string                    chain_src_frame_;          // measurement frame the chain cov is computed from
    bool                           chain_cov_enabled_ = false;
    TableConfig&                   cfg_;
    MaskIngestor*                  mask_ingestor_ = nullptr;
    TableSceneGraph*               scene_graph_   = nullptr;
    std::unique_ptr<DSR::CameraAPI> camera_api_;   // ZED intrinsics, lazily bound to the "zed" node

    std::unordered_map<std::uint64_t, TableInstance> instances_;
    std::unordered_map<std::uint64_t, Eigen::Vector2f> birth_seeds_;   // tracker-provided birth XY (see note_birth)
    std::uint64_t                  room_node_id_ = 0;   // latched per ensure_instance call
    std::ofstream                  ai2_csv_;            // per-cycle AI2 belief log (optional)

    // Staged LiDAR sweep for the range factor (room frame). Refreshed each compute() cycle; have flag false
    // ⇒ no fresh sweep this cycle ⇒ feed_lidar is a no-op (never reuses a stale sweep).
    std::vector<Eigen::Vector3f>   lidar_sweep_room_;
    Eigen::Vector3f                lidar_origin_room_ = Eigen::Vector3f::Zero();
    bool                           lidar_have_sweep_  = false;
};

}  // namespace rc
