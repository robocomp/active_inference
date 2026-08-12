/*
 * cabinet_fitter.h — the active-inference fit core of cabinet_concept (mirrors bottle_concept/bottle_fitter.h).
 *
 * Owns the per-cabinet instance map and runs the AI2 full-covariance belief update for each "cabinet_*" node:
 * instance lifecycle (ensure_instance + the CabinetModel factory), observation (split the selected mask's
 * support points into on-surface vs off-surface sets), inference (voxel-bank ingest + one recursive CabinetBelief
 * update with the mask-motion channel as the observation precision R / bias gate, written back into inst.model),
 * and the cabinet-owned voxel memory (ownership gate + FNV voxel keys). Collaborates with MaskIngestor (masks),
 * CabinetSceneGraph (robot covariance), CabinetProjection (camera projection), and CabinetLidarRangeChannel;
 * SpecificWorker keeps the orchestration (process_cabinet_node), the DSR write-back, and the post-fit
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

#include "cabinet_config.h"        // rc::CabinetConfig
#include "cabinet_instance.h"      // rc::CabinetInstance, CabinetState
#include "cabinet_kitchen_cells.h" // rc::KitchenWall (kitchen-model Stage 0 cell table)
#include "cabinet_model.h"         // CabinetModel / CabinetModelParams
#include "cabinet_projection.h"    // rc::CabinetProjection, rc::SilhouetteExistence
#include "cabinet_lidar_range_channel.h"   // rc::CabinetLidarRangeChannel
#include "../../common/mask_ingestor/mask_ingestor.h"
#include "cabinet_scene_graph.h"

namespace rc {

class CabinetFitter
{
public:
    struct CabinetObservation
    {
        bool has_fresh_data = false;
        std::vector<Eigen::Vector3f> candidate_pts;
        std::vector<Eigen::Vector3f> residual_pts;
        // Points of THIS mask slice that are flush to a DIFFERENT room wall than the one this run is built
        // against — i.e. the perpendicular arm of an L/U-corner mask. Excluded from the fit (never enter
        // frame.points, so the box cannot grow/tilt into them) and handed to residual-birth to spawn the
        // neighbouring run. Empty for a single-wall run or an island. See observe_slice's wall-split.
        std::vector<Eigen::Vector3f> foreign_pts;
        // Ricoh slices are bearing-only and never fitted (see observe_slice / process_ricoh_bearings): a
        // peripheral 360 detection has a reliable DIRECTION but a biased centroid/extent, so it only drives
        // the attention path and never contributes candidate/residual points to a belief update.
    };

    CabinetFitter(std::shared_ptr<DSR::DSRGraph> graph,
                DSR::InnerEigenAPI* inner_eigen,
                CabinetConfig& cfg,
                MaskIngestor* mask_ingestor,
                CabinetSceneGraph* scene_graph);

    // Create the instance for a "cabinet_*" node if absent (from prior/RT). Returns true the
    // first time it is created (so the worker can register Qt series / canvas pos). Latches room_id.
    bool ensure_instance(const DSR::Node& node, std::uint64_t room_id);

    // Build an observation from ONE assigned mask slice (index into the current packet): candidate/residual
    // split against the current SDF + this slice's R inputs (motion_var/depth_var/range) latched on the
    // instance. Called once per slice in the same cycle for multi-sensor fusion (ZED + ricoh). Sets
    // frames_since_detection=0 (a detection is alive). Does NOT gate on frame_id — the caller owns freshness.
    CabinetObservation observe_slice(CabinetInstance& inst, int slice_index);
    // Fallback path when the tracker assigned NO mask slice this cycle: candidate/residual point attributes
    // written directly on the node (legacy), else a stale observation (has_fresh_data=false → age the belief).
    CabinetObservation observe(CabinetInstance& inst, const DSR::Node& node);
    // One recursive full-covariance belief update (CabinetBelief) on this frame's mask points, with the
    // mask-motion channel as the observation precision R / bias gate. Writes the result into inst.model
    // so all downstream publish/viewer code is unchanged. Returns the update free energy.
    float run_inference(CabinetInstance& inst, const CabinetObservation& observation);

    // The SAME admissibility, on a RAW mask slice that has no instance yet: may this frame move geometry?
    // Birth is gated on it — a frame the fit would REFUSE can never create an object (birth_evidence.h rule 2).
    // Mirrors this agent's own gate, which is TRUNCATION only: cabinet's fitter carries no ego-motion terms,
    // so no motion condition is invented here (a birth-only condition the fit does not apply is exactly the
    // "second, weaker set of conditions" the shared policy forbids).
    bool frame_admissible(const rc::MaskIngestor::MaskSlice& sl) const
    { return sl.trunc_frac <= cfg_.ai2_trunc_gate_frac; }

    std::unordered_map<std::uint64_t, CabinetInstance>& instances() { return instances_; }
    void forget_node(std::uint64_t id) { instances_.erase(id); }
    // Part B (chain covariance): enable adding the localization/chain term J·Σ_chain·Jᵀ (measurement
    // frame → room, capture-stamp pinned) to each instance, read by the scene-graph's RT-cov write.
    void set_chain_cov_source(DSR::InnerGaussianAPI* gaussian, std::string source_frame);
    // Object-anchor observation: publish this frame's ROBOT-frame fit (z_o) so the room localizer can
    // use the cabinet as an SE(2) pose landmark. robot_frame = the localizer's base node ("body"). OFF unless enabled.
    void set_object_observation(bool enabled, std::string robot_frame);
    // Room-frame XY a NEWLY born instance's model should start at (from the tracker's detection). The
    // room→cabinet RT edge written at birth is NOT reliably queryable in the same cycle, so ensure_instance
    // would read the 0,0 default and the model would freeze there. Consumed once by ensure_instance.
    void note_birth(std::uint64_t id, const Eigen::Vector3f& c) { birth_seeds_[id] = c; }

    // As note_birth, but for a RESIDUAL-born run: carries a full room-axis seed (cx,cy,yaw,L) of the arm
    // so ensure_instance/lazy-init commit the box to THIS arm instead of re-deriving it from the shared
    // mask slice (whose dominant arm belongs to the parent). Marks the instance residual_born.
    void note_birth_seed(std::uint64_t id, const RunSeed& seed) { birth_full_seeds_[id] = seed; }

    // Room interior reference point (polygon centroid, room frame) + the room's wall polygon. The
    // first resolves the box's 180° C2v yaw ambiguity (the front normal must face the room); the
    // second feeds the per-frame wall-flush factor. Both are set by the worker once the room is known.
    void set_room_geometry(const Eigen::Vector2f& interior, std::vector<Eigen::Vector2f> polygon)
    { room_interior_ = interior; room_polygon_ = std::move(polygon); rebuild_wall_ids();
      // The projection unit needs the same walls: a silhouette sample behind one is NOT "predicted visible".
      // Without it a cabinet in the next room votes its own removal — see CabinetProjection::set_room_polygon.
      if (projection_) projection_->set_room_polygon(room_polygon_); }
    // The REACHABLE region, for the NBV. A cabinet is wall-anchored, so two of its four faces have their
    // viewpoints OUTSIDE the room; without this rc::nbv::is_reachable imposes no constraint (it refuses to
    // guess) and the direction-blind information term cannot break the tie, so a through-the-wall face can
    // win outright. The controller then REPAIRS the unroutable goal onto the cabinet. Same defect the
    // refrigerator had — see nbv-room-polygon-missing-wall-face.
    bool has_room_polygon() const { return room_polygon_.size() >= 3; }
    const std::vector<Eigen::Vector2f>& room_polygon() const { return room_polygon_; }
    bool should_log(const CabinetInstance& inst) const;
    // Kitchen-model Stage 0: the collinear-merged room walls as KitchenWall (corners + inward normal), one
    // per canonical wall id — the geometry the (wall_id, tier) cell table is built on. Empty if no polygon.
    std::vector<rc::KitchenWall> kitchen_walls() const;

    // Localization/chain covariance J·Σ_chain·Jᵀ at an arbitrary ROOM-frame point, pinned to `ts_ms`.
    // Identical construction to the private compute_chain_cov (room → measurement frame → back with ZERO
    // input cov, so InnerGaussianAPI returns exactly the chain contribution) but WITHOUT a CabinetInstance:
    // the kitchen runs are owned by KitchenManager and have no instance, yet their room-frame pose is just
    // as conditional on the robot pose. Publishing their Σ without this term would advertise a run as more
    // certain than the localization allows. Returns false (outputs untouched) when the chain source is
    // disabled or the transform does not resolve.
    bool chain_cov_at(const Eigen::Vector2f& xy_room, std::uint64_t ts_ms, float& vxx, float& vyy) const;

    // Emit a CSV row for an OUT-OF-VIEW instance whose EXISTENCE evidence integrated this cycle (LiDAR/silhouette
    // carve) but which the fitter did NOT fit (no fresh mask) — so the existence log-odds trajectory that drives a
    // removal is captured even across the silent out-of-FoV stretch where no fit row is written. Fit fields are the
    // last converged values (held); the existence columns are FRESH (integrated just before this call). Called from
    // CabinetExistence::update_and_remove for non-observed instances only, so it never double-logs a fit row.
    void log_existence_cycle(const CabinetInstance& inst)
    { log_ai2_csv(inst, 0, inst.dbg_R, /*gated=*/true, inst.dbg_energy); }

    // YOLO-independent LiDAR range channel: stage this cycle's sweep (room frame) + sensor origin for the
    // range factor. clear_lidar_sweep() each cycle first so a stale sweep never leaks into a frame with no
    // fresh LiDAR. Set/cleared from the compute() main thread by SpecificWorker (fed by CabinetLidarIngestor).
    void set_lidar_sweep(const std::vector<Eigen::Vector3f>& sweep_room, const Eigen::Vector3f& origin_room)
    { lidar_channel_.set_sweep(sweep_room, origin_room); }
    void set_lidar_sweep_bpearl(const std::vector<Eigen::Vector3f>& sweep_room, const Eigen::Vector3f& origin_room)
    { lidar_channel_.set_sweep_bpearl(sweep_room, origin_room); }
    void clear_lidar_sweep() { lidar_channel_.clear(); }

    // PIXEL-LEVEL silhouette existence evidence (EXISTENCE_BELIEF_PLAN.md, mask channel). Delegates to
    // CabinetProjection; see rc::SilhouetteExistence in cabinet_projection.h. Called from update_existence.
    SilhouetteExistence compute_silhouette_existence(const CabinetInstance& inst)
    { return projection_->compute_silhouette_existence(inst); }

private:
    // Nearest room-polygon wall segment to a room-frame point, as a WallRef (point on the line +
    // INWARD unit normal). ok=false when no polygon is available, which makes the wall-flush factor
    // inert rather than guessing — an unknown room is exactly the free-standing case.
    WallRef nearest_wall(const Eigen::Vector2f& p) const;
    // Build a WallRef for a SPECIFIC committed wall (canonical seg_id), projecting q onto that wall's
    // nearest edge. Returns ok=false if the id no longer exists (polygon changed) → caller re-commits.
    // Lets an instance reuse its persistent wall instead of re-choosing the nearest one every frame.
    WallRef wall_ref_by_seg_id(int seg_id, const Eigen::Vector2f& q) const;
    // Nearest wall to a point: canonical id, distance, and the wall's unit DIRECTION. The lean per-point
    // version for the wall-split in observe_slice. Uses wall_seg_id_ precomputed by rebuild_wall_ids.
    struct PointWall { int id = -1; float dist = 1e9f; Eigen::Vector2f dir = Eigen::Vector2f::UnitX(); };
    PointWall point_wall(const Eigen::Vector2f& p) const;
    // Group the room polygon's edges into WALLS (collinear-merged) and cache each edge's canonical wall id
    // so a point can be attributed to a wall in O(edges). Rebuilt whenever the polygon is (re)set.
    void rebuild_wall_ids();
    // Shared WallRef builder for a given polygon edge (foot/normal/segment/seg_id), q projected onto it.
    WallRef build_wall_ref(std::size_t edge_i, const Eigen::Vector2f& q) const;

    // Compute the localization/chain covariance term (J·Σ_chain·Jᵀ) at the cabinet centre by transforming
    // it from the measurement frame back to room with ZERO input cov; stored on the instance for the
    // RT-cov write. No-op unless set_chain_cov_source enabled it.
    void compute_chain_cov(CabinetInstance& inst);

    CabinetModelParams  make_model_params() const;

    // Append one AI2 belief row (state + Σ diag std + mask R/bias/trunc + gate flag) to cfg_.ai2_csv_path.
    void log_ai2_csv(const CabinetInstance& inst, int point_count, float R, bool gated, float energy);

    std::shared_ptr<DSR::DSRGraph> G_;
    DSR::InnerEigenAPI*            inner_eigen_ = nullptr;
    DSR::InnerGaussianAPI*         gaussian_    = nullptr;   // Part B: chain covariance (set_chain_cov_source)
    std::string                    chain_src_frame_;          // measurement frame the chain cov is computed from
    bool                           chain_cov_enabled_ = false;
    bool                           obs_enabled_ = false;       // publish z_o (object-anchor observation)?
    std::string                    obs_robot_frame_ = "body";  // localizer base frame z_o is expressed in
    std::string                    obs_cam_frame_ = "zed";     // camera frame the raw mask cloud lives in
    void compute_object_observation(CabinetInstance& inst);      // fill inst.obs_robot (gated)
    CabinetConfig&                   cfg_;
    MaskIngestor*                  mask_ingestor_ = nullptr;
    CabinetSceneGraph*               scene_graph_   = nullptr;
    std::unique_ptr<CabinetProjection> projection_;   // camera-projection unit (owns the ZED CameraAPI)

    std::unordered_map<std::uint64_t, CabinetInstance> instances_;
    std::unordered_map<std::uint64_t, Eigen::Vector3f> birth_seeds_;   // tracker-provided birth centroid XYZ (see note_birth)
    std::unordered_map<std::uint64_t, RunSeed>         birth_full_seeds_;   // residual-born full seed (see note_birth_seed)
    Eigen::Vector2f                room_interior_ = Eigen::Vector2f::Zero();   // polygon centroid: C2v yaw fold
    std::vector<Eigen::Vector2f>   room_polygon_;                              // room walls: wall-flush factor
    std::vector<int>               wall_seg_id_;                               // per-edge canonical wall id (merged)
    std::uint64_t                  room_node_id_ = 0;   // latched per ensure_instance call
    std::ofstream                  ai2_csv_;            // per-cycle AI2 belief log (optional)

    // YOLO-independent LiDAR range channel: owns the staged per-cycle sweep + return selection (feed).
    CabinetLidarRangeChannel         lidar_channel_{cfg_};
};

}  // namespace rc
