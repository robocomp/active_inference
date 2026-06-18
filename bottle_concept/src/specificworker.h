/*
 *    Copyright (C) 2026 by RoboComp CORTEX Team
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    RoboComp is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * bottle_concept — Active Inference agent for bottle instance detection.
 *
 * Reads YOLO masks (the "masks" DSR node written by the voxelizer), selects the
 * slices labelled "bottle", and fits a vertical-cylinder generative model
 * (5-param state + cylinder SDF) by free-energy minimisation. The fitted pose
 * AND its Laplace-curvature covariance (P_bottle) are written on the room→bottle
 * RT edge, so the kinova_controller can read the bottle pose with an explicit
 * uncertainty that drives its closed→open-loop look-up rate.
 *
 * Bottle nodes use the DSR `cylinder` node type (geometrically exact: radius +
 * height) named "bottle_N". This is a focused port of table_concept: the mask
 * reading, RT machinery and FE loop are kept; tracks, the epistemic planner, the
 * table affordance, the warm-start policy, the Qt widgets and the presence
 * protocol are dropped for the MVP.
 */

#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#include <atomic>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <genericworker.h>
#include <Eigen/Dense>

#include <dsr/api/dsr_rt_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

#include "../../common/agent_presence_coordinator/agent_presence_coordinator.h"
#include "../../common/robust_metrics/robust_metrics.h"
#include "bottle_model.h"
#include "prior_store.h"
#include "sample_queue.h"

// ─── Per-bottle instance state ────────────────────────────────────────────────

struct BottleInstance
{
    uint64_t    node_id = 0;
    std::string node_name;

    BottleModel model;
    SampleQueue queue;

    int  matched_frames        = 0;     // frames with fresh sensing data
    int  frames_converged      = 0;     // consecutive frames with |ΔFE| < fe_eps
    int  last_masks_frame_seen = -1;    // last masks packet frame consumed
    int  processed_cycles      = 0;     // per-bottle compute cycles for log throttling
    int  model_generation      = 0;
    float prev_free_energy     = std::numeric_limits<float>::max();
    // Dead-band tracking for write_rt_pose — suppress tiny oscillations
    float last_written_cx = std::numeric_limits<float>::max();
    float last_written_cy = std::numeric_limits<float>::max();
    // Last model PUBLISHED to the graph — gate node/edge writes to meaningful changes so a
    // stable fit stops rewriting the node (big mesh + model_generation) and the RT edge.
    float last_pub_radius = std::numeric_limits<float>::max();
    float last_pub_height = std::numeric_limits<float>::max();
    float last_pub_cx     = std::numeric_limits<float>::max();
    float last_pub_cy     = std::numeric_limits<float>::max();
    float last_pub_cz     = std::numeric_limits<float>::max();
    SampleQueueMetrics      last_queue_metrics;
    FreeEnergyDecomposition last_fe_terms;
    // Bottle-owned voxel memory bank (room frame), independent of per-frame uploads.
    std::vector<Eigen::Vector3f>      voxel_bank_pts;
    std::unordered_set<std::uint64_t> voxel_bank_keys;
};

// ─── Agent configuration ─────────────────────────────────────────────────────

struct AgentConfig
{
    float fe_eps           = 1e-3f;
    int   K_stable         = 30;
    float write_threshold  = 1e-3f;
    int   log_period_frames = 30;
    int   voxel_bank_max_points     = 4000;
    float voxel_bank_quantization_m = 0.01f;
    float voxel_select_radius_margin_m = 0.10f;
    float voxel_select_height_margin_m = 0.10f;

    // BottleModel parameters (forwarded to BottleModelParams)
    float sigma_obs         = 0.02f;
    float lambda_size       = 0.5f;
    float lambda_pos        = 0.05f;
    float lambda_state      = 0.02f;
    float prior_radius      = 0.035f;
    float prior_height      = 0.20f;
    float prior_size_std    = 0.03f;
    int   optimization_iters = 15;
    float optimization_lr   = 0.005f;   // cm-scale object: 10× smaller than the table's
    float grad_clip         = 2.0f;
    std::string optimizer_type = "adam";
    float sgd_momentum      = 0.9f;
    RobustLossType robust_loss = RobustLossType::Quadratic;
    float robust_loss_scale = 0.05f;

    // SampleQueue parameters (forwarded to SampleQueueParams)
    int   num_angle_bins               = 16;
    int   num_z_bins                   = 6;
    int   max_per_bin                  = 2;
    float sdf_threshold_for_storage    = 0.03f;
    int   min_frames_before_historical = 10;
    int   historical_warmup_frames     = 5;
    int   max_new_points_per_frame     = 20;
    float rfe_alpha                    = 0.98f;
    float rfe_max_threshold            = 2.0f;
    float rfe_weight_gain              = 0.25f;
    float min_anchor_weight            = 0.12f;
    float edge_bonus_weight            = 0.3f;
    float edge_proximity_threshold     = 0.01f;
    float z_bin_size                   = 0.04f;

    // Covariance write
    float yaw_variance = 9.87f;   // ≈π² — yaw is unobservable for a symmetric cylinder

    // ── Static ground-truth evaluation (Webots) ────────────────────────────────
    // The bottle is stationary during perception, so its Webots pose is a constant
    // expressed in the room frame (DEF bottle → Shadow→room). When enabled, the
    // tracker logs per-cycle position/size error and NEES (covariance calibration).
    bool        eval_enabled = false;
    std::string eval_log_path = "etc/bottle_eval.csv";
    std::string eval_gt_source = "webots";   // "webots" (live getObjectPose) | "config"
    std::string eval_bottle_def = "bottle";  // Webots DEF of the bottle
    std::string eval_robot_def  = "shadow";  // Webots DEF of the Shadow robot (== DSR body frame)
    float gt_cx = 0.0f, gt_cy = 0.0f, gt_cz = 0.0f;   // cylinder CENTRE, room frame
    float gt_radius = 0.0f, gt_height = 0.0f;
};

// ─── SpecificWorker ──────────────────────────────────────────────────────────

class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:
    SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check);
    ~SpecificWorker();

public slots:
    void initialize();
    void compute();
    void emergency();
    void restore();
    int  startup_check();

    void modify_node_slot(std::uint64_t, const std::string& type){};
    void modify_node_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names){};
    void modify_edge_slot(std::uint64_t from, std::uint64_t to, const std::string& type){};
    void modify_edge_attrs_slot(std::uint64_t from, std::uint64_t to,
                                const std::string& type, const std::vector<std::string>& att_names){};
    void del_edge_slot(std::uint64_t from, std::uint64_t to, const std::string& edge_tag){};
    void del_node_slot(std::uint64_t from);

private:
    struct BottleObservation
    {
        bool has_fresh_data = false;
        float explanation_ratio = 1.0f;
        std::vector<Eigen::Vector3f> candidate_pts;
        std::vector<Eigen::Vector3f> residual_pts;
    };

    struct MaskSlice
    {
        std::string label;
        float class_id = -1.0f;
        float confidence = 0.0f;
        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        Eigen::Vector3f bbox_min = Eigen::Vector3f::Zero();
        Eigen::Vector3f bbox_max = Eigen::Vector3f::Zero();
        std::size_t support_begin = 0;
        std::size_t support_end = 0;
    };

    struct MasksPacket
    {
        bool valid = false;
        int frame_id = -1;
        std::vector<MaskSlice> slices;
        std::vector<Eigen::Vector3f> support_points;
    };

    // ── Presence protocol ──────────────────────────────────────────────────────
    void waiting_enter();
    void waiting_loop();
    void operating_enter();
    void operating_loop();
    void degraded_enter();
    void degraded_loop();
    void cleanup_owned_nodes();
    void request_shutdown();
    void on_optional_peer_lost(const std::string& name, std::uint32_t id);
    void on_optional_peer_ready(const std::string& name, std::uint32_t id);
    // Delete every "bottle*" cylinder node this agent owns (startup sweep + teardown).
    void remove_owned_bottle_nodes();

    // ── Initialisation helpers ────────────────────────────────────────────────
    void load_config(const ConfigLoader& cfg);
    void scaffold_missing_bottle_nodes();
    void ensure_instance(const DSR::Node& node);
    bool should_log(const BottleInstance& inst) const;
    bool refresh_masks_packet();
    std::optional<MaskSlice> select_mask_for_bottle(const BottleInstance& inst) const;

    // ── Per-bottle pipeline ────────────────────────────────────────────────────
    void process_bottle_node(const DSR::Node& node);
    BottleObservation observe_bottle_node(BottleInstance& inst, const DSR::Node& node);
    void ingest_observation_voxels(BottleInstance& inst, const BottleObservation& observation);
    bool is_voxel_owned_by_bottle(const BottleInstance& inst, const Eigen::Vector3f& point) const;
    float run_bottle_inference(BottleInstance& inst, const BottleObservation& observation);
    void step_queue_update(BottleInstance& inst,
                           const std::vector<Eigen::Vector3f>& candidate_pts,
                           float observation_precision);
    float step_model_update(BottleInstance& inst,
                            const std::vector<Eigen::Vector3f>& residual_pts,
                            float residual_precision);
    void step_write_model(BottleInstance& inst, DSR::Node& node, float free_energy);
    // Append one fit-vs-ground-truth row (position/size error + NEES) to the eval CSV.
    void log_eval(const BottleInstance& inst, float free_energy);
    // One-shot: query the bottle's static Webots pose and express its CENTRE in the room
    // frame, storing it in cfg_.gt_*. Returns false if the bridge/transforms aren't ready.
    bool acquire_webots_gt();

    // ── DSR helpers ──────────────────────────────────────────────────────────
    std::vector<Eigen::Vector3f> read_pts_attrib(const DSR::Node& node,
                                                 const std::string& att_name) const;
    static std::uint64_t voxel_key(const Eigen::Vector3f& point, float quantization_m);
    Eigen::Matrix2f read_robot_covariance() const;
    void write_rt_pose(uint64_t room_id, BottleInstance& inst);
    static std::vector<float> make_cylinder_mesh(const BottleState& s, int segments = 16);

    // ── Factory helpers ────────────────────────────────────────────────────────
    BottleModelParams make_model_params() const;
    SampleQueueParams make_queue_params() const;

    // ── Members ──────────────────────────────────────────────────────────────
    bool startup_check_flag = false;
    bool owned_nodes_cleaned_ = false;
    std::atomic<bool> shutting_down_{false};
    AgentPresenceCoordinator presence_coordinator_;

    AgentConfig                                 cfg_;
    std::unique_ptr<PriorStore>                 prior_store_;
    std::vector<BottlePrior>                    priors_cache_;
    std::unordered_map<uint64_t, BottleInstance> instances_;

    std::unique_ptr<DSR::RT_API> rt_api_;
    std::unique_ptr<DSR::InnerEigenAPI> inner_eigen_;
    bool                         gt_acquired_ = false;
    uint64_t                     room_node_id_ = 0;
    int                          last_masks_frame_seen_ = -1;
    MasksPacket                  masks_packet_;

    std::string priors_path_;
    std::string checkpoint_path_;

    std::ofstream eval_log_;   // open when cfg_.eval_enabled; header written on first row

signals:
    void presenceReady();
    void presenceLost();
};

#endif // SPECIFICWORKER_H
