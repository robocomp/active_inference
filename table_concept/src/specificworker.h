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
 * table_concept — Active Inference agent for table instance detection and maintenance.
 *
 * Owns the generative model (7-param state + compound SDF) for every table
 * node in the DSR graph.  Runs a free-energy minimisation loop, maintains a
 * binned historical sample queue, and emits epistemic action proposals to
 * mission-controller when table surfaces remain under-observed.
 *
 * See TABLE_CONCEPT.md for the full design specification.
 */

#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <genericworker.h>
#include <Eigen/Dense>

#include "epistemic_planner.h"
#include "prior_store.h"
#include "sample_queue.h"
#include "table_affordance.h"
#include "table_model.h"
#include "custom_widget.h"
#include "timeseries_plot.h"

// ─── Per-table instance state ────────────────────────────────────────────────

struct TableInstance
{
    uint64_t    node_id;
    std::string node_name;

    TableModel  model;
    SampleQueue queue;

    int  last_frame_seen    = -1;     // last_sensing_frame_att value read
    int  matched_frames     = 0;      // frames with fresh sensing data
    int  frames_converged    = 0;      // consecutive frames with |ΔFE| < fe_eps
    int  frames_rising      = 0;      // consecutive frames with F increasing
    bool model_stable       = false;
    int  model_generation   = 0;
    bool epistemic_pending  = false;
    float prev_free_energy  = std::numeric_limits<float>::max();
    // Dead-band tracking for write_rt_pose — suppress tiny oscillations
    float last_written_cx   = std::numeric_limits<float>::max();
    float last_written_cy   = std::numeric_limits<float>::max();
    // Last coverage deficit (written by step_convergence, read by plot)
    float last_coverage_deficit = 0.f;
    // Higher-level confidence state driving warm-start precision from above
    float warm_confidence = 0.0f;
    // Epistemic action request published to DSR
    TableAffordance affordance;
};

// ─── Agent configuration ─────────────────────────────────────────────────────

struct AgentConfig
{
    // Tunable via etc/config.toml
    float fe_eps            = 1e-3f;   // |ΔFE| threshold for convergence
    int   K_stable          = 30;
    int   M_diverge         = 20;
    float staleness_frames  = 90.0f;
    float explanation_ratio_thresh = 0.3f;
    float write_threshold   = 1e-3f;   // min ‖Δθ‖ before writing RT to DSR
    float obs_distance      = 1.8f;    // d_obs for epistemic planner
    float delta_min         = 20.0f;   // min face coverage count
    float gain_threshold    = 0.1f;    // min ΔH for epistemic proposal

    // TableModel parameters (forwarded to TableModelParams)
    float sigma_obs         = 0.05f;
    float lambda_size       = 0.5f;
    float lambda_pos        = 0.05f;
    float lambda_state      = 0.02f;
    float lambda_angle      = 0.01f;
    int   optimization_iters = 10;
    float optimization_lr   = 0.05f;
    float grad_clip         = 2.0f;
    std::string optimizer_type = "adam";
    float sgd_momentum     = 0.9f;
    RobustLossType robust_loss = RobustLossType::Quadratic;
    float robust_loss_scale = 0.10f;

    // SampleQueue parameters (forwarded to SampleQueueParams)
    int   num_angle_bins               = 24;
    int   num_z_bins                   = 10;
    int   max_per_bin                  = 2;
    float sdf_threshold_for_storage    = 0.08f;
    int   min_frames_before_historical = 10;
    int   historical_warmup_frames     = 50;
    int   max_new_points_per_frame     = 5;
    float rfe_alpha                    = 0.98f;
    float rfe_max_threshold            = 2.0f;
    float edge_bonus_weight            = 0.3f;
    float edge_proximity_threshold     = 0.05f;

    // Observability-aware warm-start acceptance
    float warm_pts_min                 = 12.0f;
    float warm_pts_max                 = 30.0f;
    float warm_coverage_min_side       = 2.0f;
    float warm_rho_freeze              = 0.25f;
    float warm_lambda_pos_base         = 0.15f;
    float warm_lambda_pos_gain         = 0.45f;
    float warm_lambda_size_base        = 0.02f;
    float warm_lambda_size_gain        = 0.18f;
    float warm_lambda_yaw_base         = 0.01f;
    float warm_lambda_yaw_gain         = 0.12f;
    float warm_confidence_decay         = 0.70f;
    float warm_confidence_coverage_gain = 0.35f;
    float warm_confidence_residual_gain = 0.65f;
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

    void modify_node_slot(std::uint64_t id, const std::string& type);
    void modify_node_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names);
    void modify_edge_slot(std::uint64_t from, std::uint64_t to, const std::string& type){};
    void modify_edge_attrs_slot(std::uint64_t from, std::uint64_t to,
                                const std::string& type, const std::vector<std::string>& att_names){};
    void del_edge_slot(std::uint64_t from, std::uint64_t to, const std::string& edge_tag){};
    void del_node_slot(std::uint64_t from);

private:
    struct TableObservation
    {
        bool has_fresh_data = false;
        float explanation_ratio = 1.0f;
        std::vector<Eigen::Vector3f> candidate_pts;
        std::vector<Eigen::Vector3f> residual_pts;
    };

    struct TableBeliefEvidence
    {
        std::vector<Eigen::Vector3f> fit_pts;
        std::vector<float> fit_weights;
        std::vector<Eigen::Vector3f> eval_pts;
        std::vector<float> eval_weights;
        int residual_count = 0;
        int trusted_point_count = 0;
        float residual_precision = 0.0f;

        [[nodiscard]] bool has_evaluation() const { return not eval_pts.empty(); }
        [[nodiscard]] bool can_optimize() const { return not fit_pts.empty(); }
    };

    struct TableBeliefPolicy
    {
        static float clamp01(float value);
        static float lerp(float start, float end, float gain);
        static float wrap_angle(float angle);
        static float angle_lerp(float start, float end, float gain);
        static TableState apply_observability_warm_start(const TableState& previous,
                                                         const TableState& raw,
                                                         const TableModelParams& params,
                                                         const AgentConfig& cfg,
                                                         float confidence,
                                                         const std::array<float, 6>& coverage,
                                                         int point_count);
        static float update_warm_confidence(float previous_confidence,
                                            const AgentConfig& cfg,
                                            const std::array<float, 6>& coverage,
                                            int point_count,
                                            int residual_count,
                                            float residual_precision);
    };

    // ── Initialisation helpers ────────────────────────────────────────────────
    void load_config(const ConfigLoader& cfg);
    void scaffold_missing_table_nodes();
    void ensure_instance(const DSR::Node& node);
    void process_table_node(const DSR::Node& node);
    TableObservation observe_table_node(TableInstance& inst, const DSR::Node& node);
    float run_table_inference(TableInstance& inst, const TableObservation& observation);
    void publish_table_cycle(TableInstance& inst,
                             const DSR::Node& node,
                             const TableObservation& observation,
                             float free_energy);
    bool persist_table_belief(TableInstance& inst, uint64_t node_id, float free_energy);
    bool assess_table_state(TableInstance& inst, uint64_t node_id, float free_energy);
    void publish_table_diagnostics(const TableInstance& inst,
                                   const TableObservation& observation,
                                   float free_energy);
    void publish_table_intentions(TableInstance& inst,
                                  uint64_t node_id,
                                  const TableObservation& observation,
                                  float free_energy);
    TableBeliefEvidence compose_belief_evidence(const TableInstance& inst,
                                                const std::vector<Eigen::Vector3f>& residual_pts,
                                                float residual_precision) const;
    void evolve_table_belief(TableInstance& inst, const TableBeliefEvidence& evidence);
    float accept_table_belief(TableInstance& inst,
                              const TableState& previous_state,
                              const TableBeliefEvidence& evidence);
    void refresh_table_memory(TableInstance& inst);

    // ── Per-table per-cycle steps (§11.2) ────────────────────────────────────
    void step_queue_update(TableInstance& inst,
                           const std::vector<Eigen::Vector3f>& candidate_pts,
                           float observation_precision);
    float step_model_update(TableInstance& inst,
                            const std::vector<Eigen::Vector3f>& residual_pts,
                            float residual_precision);
    void step_write_model(TableInstance& inst, DSR::Node& node, float free_energy);
    void step_convergence(TableInstance& inst, DSR::Node& node, float free_energy);
    void step_epistemic(TableInstance& inst, DSR::Node& node);
    void step_refresh_check(TableInstance& inst, DSR::Node& node,
                            float free_energy, float explanation_ratio);

    // ── DSR helpers ──────────────────────────────────────────────────────────
    std::vector<Eigen::Vector3f> read_pts_attrib(const DSR::Node& node,
                                                  const std::string& att_name) const;
    Eigen::Matrix2f read_robot_covariance() const;
    void write_rt_pose(uint64_t room_id, TableInstance& inst);
    void write_bool_att(DSR::Node& node, const std::string& key, bool val);
    void write_float_att(DSR::Node& node, const std::string& key, float val);
    void write_int_att(DSR::Node& node, const std::string& key, int val);
    void write_epistemic_proposal(DSR::Node& node, const EpistemicProposal& prop);
    void write_table_mesh(TableInstance& inst, DSR::Node& node);
    void trigger_graph_layout_twopi();
    static std::vector<float> make_table_mesh(const TableState& s);

    // ── Instance management ──────────────────────────────────────────────────
    TableModelParams  make_model_params() const;
    SampleQueueParams make_queue_params() const;
    TableState        prior_to_state(const TablePrior& p) const;

    // ── Members ──────────────────────────────────────────────────────────────
    bool startup_check_flag = false;

    AgentConfig                                         cfg_;
    std::unique_ptr<PriorStore>                         prior_store_;
    EpistemicPlanner                                    epistemic_planner_;
    std::unordered_map<uint64_t, TableInstance>         instances_;

    Custom_widget*       custom_widget_ = nullptr;
    rc::TimeSeriesPlot*  ts_plot_       = nullptr;   // FE
    rc::TimeSeriesPlot*  ts_cov_plot_   = nullptr;   // coverage deficit
    rc::TimeSeriesPlot*  ts_res_plot_   = nullptr;   // residual point count

    std::unique_ptr<DSR::RT_API>                        rt_api_;
    uint64_t                                            room_node_id_ = 0;

    // Paths resolved from ConfigLoader
    std::string priors_path_;
    std::string checkpoint_path_;

signals:
    // void customSignal();
};

#endif // SPECIFICWORKER_H
