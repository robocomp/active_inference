/*
 * bottle_fitter.h
 *
 * The active-inference core of bottle_concept. Owns the per-bottle instance map and
 * runs the free-energy fit for each "bottle_*" cylinder node every cycle:
 *   - instance lifecycle (ensure_instance + the BottleModel/SampleQueue factories),
 *   - observation: split the selected mask's support points into queue anchors vs
 *     residuals against the current SDF (observe),
 *   - inference: voxel-bank ingest + cold-start seed (with camera-ward de-projection)
 *     + queue update + RGB silhouette likelihood + the gradient/free-energy step,
 *   - the bottle-owned voxel memory (ownership gate + FNV voxel keys).
 *
 * Pure belief engine (mirrors TableFitter): exposes ensure_instance → observe → run_inference and
 * does NOT write DSR. READS via MaskIngestor (masks) and BottleSceneGraph (table lookups / support
 * surface / robot covariance). The worker (SpecificWorker::process_bottle_node) owns the write-back
 * (scene_graph_->step_write_model) and the eval log. Plain class (no Q_OBJECT).
 */

#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/api/dsr_camera_api.h>

#include "bottle_config.h"
#include "bottle_instance.h"
#include "bottle_model.h"        // BottleModel / BottleState / BottleModelParams
#include "prior_store.h"         // BottlePrior
#include "../../common/mask_ingestor/mask_ingestor.h"
#include "bottle_scene_graph.h"

namespace rc {

class BottleFitter
{
public:
    BottleFitter(std::shared_ptr<DSR::DSRGraph> graph,
                 DSR::InnerEigenAPI* inner_eigen,
                 BottleConfig& cfg,
                 const std::vector<BottlePrior>& priors,
                 MaskIngestor* perception,
                 BottleSceneGraph* scene_graph);

    // Per-cycle fresh observation: the selected mask's support split into queue anchors vs residuals.
    struct BottleObservation
    {
        bool has_fresh_data = false;
        float explanation_ratio = 1.0f;
        std::vector<Eigen::Vector3f> candidate_pts;
        std::vector<Eigen::Vector3f> residual_pts;
    };

    // Pure belief API (no DSR write-back: the worker orchestrates persist + eval). Mirrors TableFitter:
    //   ensure_instance → observe → run_inference; run_inference also does the support-surface decision
    //   and the table-top z anchor (object-specific belief steps), reading — never writing — via the
    //   scene_graph. Returns the free energy. The worker then calls scene_graph_->step_write_model and
    //   the evaluator.
    // Create the instance for a "bottle_*" node if absent. Returns true the first time it is created
    // (so the worker can do one-time setup); latches room_node_id every call.
    bool ensure_instance(const DSR::Node& node, std::uint64_t room_node_id);
    BottleObservation observe(BottleInstance& inst, const DSR::Node& node);
    // The fit: one recursive full-covariance AI2 belief update (bottle_belief.h) on this frame's mask
    // points + silhouette. Writes the result into inst.model so downstream publish/RT code is unchanged.
    float run_inference(BottleInstance& inst, const BottleObservation& observation);
    bool should_log(const BottleInstance& inst) const;

    // The live instance map (the validation sweep mutates it; del_node prunes it).
    std::unordered_map<std::uint64_t, BottleInstance>& instances() { return instances_; }
    void forget_node(std::uint64_t id) { instances_.erase(id); }
    // Room-frame XY a NEWLY born instance's model should start at (from the tracker's detection). The
    // room→bottle RT written at birth is not reliably composable by inner_eigen in the same cycle, so
    // the model could cold-start at 0,0; the fit normally drags it to the points, but seeding it removes
    // that dependency (and the matching tracker re-birth risk). Consumed once by ensure_instance.
    void note_birth(std::uint64_t id, const Eigen::Vector2f& xy) { birth_seeds_[id] = xy; }

private:
    // Object-specific belief pre-step: decide the resting surface (room vs a table) + set the table-top
    // z anchor (hysteretic re-parent). Reads via scene_graph_; called at the head of run_inference.
    void update_support_surface(BottleInstance& inst);
    // YOLO detection score → per-observation reliability weight w∈[0,1] (monotone map). Scales the
    // silhouette precision so a weak mask can't over-tighten radius. 1.0 when cfg_.mask_conf_weight is off.
    float mask_confidence_weight(float confidence) const;

    // Voxel bank (bottle-owned historical memory).
    void ingest_observation_voxels(BottleInstance& inst, const BottleObservation& observation);
    bool is_voxel_owned_by_bottle(const BottleInstance& inst, const Eigen::Vector3f& point) const;
    static std::uint64_t voxel_key(const Eigen::Vector3f& point, float quantization_m);

    // Append one AI2 belief row (state + Σ diag std) to cfg_.ai2_csv_path. No-op if the path is empty.
    void log_ai2_csv(const BottleInstance& inst, int point_count, float R, float energy);

    // Feed the fitted model the RGB-mask edge rays as a silhouette likelihood.
    void feed_silhouette(BottleInstance& inst);
    // Set inst.expected_visible: true iff the bottle centre projects inside the camera frustum now. Drives
    // the tracker's negative-information DEATH gate (persist out-of-FoV; retire only if in-view & absent).
    void update_expected_visible(BottleInstance& inst);
    // room_T_zed (camera→room) as a plain 4×4, composed room→body→zed at ts=0 (alignment-safe).
    std::optional<Eigen::Matrix4d> room_T_zed_matrix(std::uint64_t timestamp_ms = 0) const;

    // Factory helper (config → model geometry/prior params; the model carries the SDF + state).
    BottleModelParams make_model_params() const;

    std::shared_ptr<DSR::DSRGraph>  G_;
    DSR::InnerEigenAPI*             inner_eigen_ = nullptr;
    BottleConfig&                    cfg_;
    const std::vector<BottlePrior>& priors_;
    MaskIngestor*               mask_ingestor_  = nullptr;
    BottleSceneGraph*               scene_graph_ = nullptr;   // READS only (support surface, table-top, robot cov)

    std::unique_ptr<DSR::CameraAPI> camera_api_;   // ZED intrinsics, lazily bound to the "zed" node
    std::unordered_map<std::uint64_t, BottleInstance> instances_;
    std::unordered_map<std::uint64_t, Eigen::Vector2f> birth_seeds_;   // tracker-provided birth XY (see note_birth)
    std::uint64_t                   room_node_id_ = 0;   // refreshed each process_bottle_node call
    std::ofstream                   ai2_csv_;            // per-cycle AI2 belief log (optional)
};

}  // namespace rc
