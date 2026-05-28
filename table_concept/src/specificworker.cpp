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
 * SpecificWorker — table_concept agent
 *
 * Implements the Active Inference loop described in TABLE_CONCEPT.md §11.2:
 *
 *  ① Read sensing attributes from DSR table nodes
 *  ② Update the historical sample queue with fresh near-surface candidates
 *  ③ Run gradient-descent steps on the 7-param generative model (SDF + FE)
 *  ④ Write updated model parameters back to DSR (RT edge + geometry attrs)
 *  ⑤ Check convergence and set model_stable_att
 *  ⑥ Compute epistemic action proposals (viewpoint → mission-controller)
 *  ⑦ Detect divergence and set request_full_sample_att
 */

#include "specificworker.h"

#include <filesystem>
#include <print>

#include <algorithm>

// DSR attribute name tags — generated from dsr_attr_name.h
#include <dsr/api/dsr_api.h>

float SpecificWorker::TableBeliefPolicy::clamp01(float value)
{
    return std::clamp(value, 0.0f, 1.0f);
}

float SpecificWorker::TableBeliefPolicy::lerp(float start, float end, float gain)
{
    return start + gain * (end - start);
}

float SpecificWorker::TableBeliefPolicy::wrap_angle(float angle)
{
    while (angle > M_PIf) angle -= 2.0f * M_PIf;
    while (angle < -M_PIf) angle += 2.0f * M_PIf;
    return angle;
}

float SpecificWorker::TableBeliefPolicy::angle_lerp(float start, float end, float gain)
{
    return wrap_angle(start + gain * wrap_angle(end - start));
}

TableState SpecificWorker::TableBeliefPolicy::apply_observability_warm_start(
    const TableState& previous,
    const TableState& raw,
    const TableModelParams& params,
    const AgentConfig& cfg,
    float confidence,
    const std::array<float, 6>& coverage,
    int point_count)
{
    constexpr float kCoverageEps = 1e-3f;

    const float cov_px = coverage[0];
    const float cov_nx = coverage[1];
    const float cov_py = coverage[2];
    const float cov_ny = coverage[3];

    const float rho_x = std::min(cov_px, cov_nx) / (std::max(cov_px, cov_nx) + kCoverageEps);
    const float rho_y = std::min(cov_py, cov_ny) / (std::max(cov_py, cov_ny) + kCoverageEps);
    const float pts_span = std::max(1e-3f, cfg.warm_pts_max - cfg.warm_pts_min);
    const float rho_pts = clamp01((static_cast<float>(point_count) - cfg.warm_pts_min) / pts_span);

    const float rho_pos = rho_pts * std::max(rho_x, rho_y);
    const float rho_size_x = rho_pts * rho_x;
    const float rho_size_y = rho_pts * rho_y;
    const float rho_vertical = rho_pts;
    const float yaw_support = clamp01(0.25f * std::max(rho_x, rho_y) + 0.75f * std::sqrt(rho_x * rho_y));

    const float lambda_pos = lerp(cfg.warm_lambda_pos_base + cfg.warm_lambda_pos_gain * rho_pos, 0.95f, confidence);
    const float lambda_size_x = lerp(cfg.warm_lambda_size_base + cfg.warm_lambda_size_gain * rho_size_x, 0.95f, confidence);
    const float lambda_size_y = lerp(cfg.warm_lambda_size_base + cfg.warm_lambda_size_gain * rho_size_y, 0.95f, confidence);
    const float lambda_vertical = lerp(cfg.warm_lambda_size_base + cfg.warm_lambda_size_gain * rho_vertical, 0.90f, confidence);
    const float lambda_yaw = lerp(cfg.warm_lambda_yaw_base + cfg.warm_lambda_yaw_gain * (rho_pts * yaw_support), 0.70f, confidence);

    const float effective_side_min = cfg.warm_coverage_min_side * (1.0f - 0.6f * confidence);
    const float effective_rho_freeze = cfg.warm_rho_freeze * (1.0f - 0.7f * confidence);

    const bool freeze_x = std::min(cov_px, cov_nx) < effective_side_min || rho_x < effective_rho_freeze;
    const bool freeze_y = std::min(cov_py, cov_ny) < effective_side_min || rho_y < effective_rho_freeze;

    TableState accepted = raw;
    accepted.cx = lerp(previous.cx, raw.cx, lambda_pos);
    accepted.cy = lerp(previous.cy, raw.cy, lambda_pos);
    accepted.w = freeze_x ? previous.w : lerp(previous.w, raw.w, lambda_size_x);
    accepted.h = freeze_y ? previous.h : lerp(previous.h, raw.h, lambda_size_y);
    accepted.table_height = lerp(previous.table_height, raw.table_height, lambda_vertical);
    accepted.leg_length = lerp(previous.leg_length, raw.leg_length, lambda_vertical);
    accepted.yaw = angle_lerp(previous.yaw, raw.yaw, lambda_yaw);

    return accepted;
}

float SpecificWorker::TableBeliefPolicy::update_warm_confidence(
    float previous_confidence,
    const AgentConfig& cfg,
    const std::array<float, 6>& coverage,
    int point_count,
    int residual_count,
    float residual_precision)
{
    constexpr float kCoverageEps = 1e-3f;

    const float cov_px = coverage[0];
    const float cov_nx = coverage[1];
    const float cov_py = coverage[2];
    const float cov_ny = coverage[3];

    const float rho_x = std::min(cov_px, cov_nx) / (std::max(cov_px, cov_nx) + kCoverageEps);
    const float rho_y = std::min(cov_py, cov_ny) / (std::max(cov_py, cov_ny) + kCoverageEps);
    const float pts_span = std::max(1e-3f, cfg.warm_pts_max - cfg.warm_pts_min);
    const float rho_pts = clamp01((static_cast<float>(point_count) - cfg.warm_pts_min) / pts_span);
    const float residual_ratio = clamp01(residual_precision * static_cast<float>(residual_count) /
                                         static_cast<float>(std::max(1, point_count + residual_count)));

    // Dense observations can raise confidence, but only balanced bilateral
    // coverage should drive it close to one.
    const float bilateral_support = 0.5f * (rho_x + rho_y);
    const float coverage_evidence = rho_pts * lerp(0.15f, 1.0f, bilateral_support);
    const float evidence = cfg.warm_confidence_coverage_gain * coverage_evidence +
                           cfg.warm_confidence_residual_gain * residual_ratio;

    const float updated = cfg.warm_confidence_decay * previous_confidence +
                          (1.0f - cfg.warm_confidence_decay) * evidence;
    return clamp01(updated);
}

// ─── Constructor / Destructor ─────────────────────────────────────────────────

SpecificWorker::SpecificWorker(const ConfigLoader& configLoader,
                               TuplePrx tprx,
                               bool startup_check)
    : GenericWorker(configLoader, tprx),
      startup_check_flag(startup_check)
{
    if (startup_check_flag)
    {
        this->startup_check();
        return;
    }

    load_config(configLoader);

    statemachine.setChildMode(QState::ExclusiveStates);
    statemachine.start();

    const auto err = statemachine.errorString();
    if (err.length() > 0)
    {
        qWarning() << err;
        throw err;
    }
}

SpecificWorker::~SpecificWorker()
{
    // Save convergence checkpoints for all instances
    if (prior_store_)
    {
        for (const auto& [id, inst] : instances_)
        {
            prior_store_->save_checkpoint(TableCheckpoint{
                inst.node_name,
                inst.model.state().w,
                inst.model.state().h,
                inst.model.state().table_height,
                inst.model.state().cx,
                inst.model.state().cy,
                inst.model.state().yaw,
                inst.prev_free_energy,
                inst.model_stable
            });
        }
    }
    // Remove affordance nodes first (children of table nodes)
    if (G)
    {
        for (auto& [id, inst] : instances_)
        {
            if (inst.affordance.is_active())
            {
                G->delete_node(inst.affordance.node_id());
                std::print("table_concept: removed affordance node for '{}'\n", inst.node_name);
            }
        }
        // Remove table nodes themselves
        for (const auto& [id, inst] : instances_)
        {
            G->delete_node(id);
            std::print("table_concept: removed table node '{}'\n", inst.node_name);
        }
    }
    std::print("table_concept: SpecificWorker destroyed, checkpoints saved.\n");
}

// ─── Initialisation ──────────────────────────────────────────────────────────

void SpecificWorker::initialize()
{
    std::print("table_concept: initialize()\n");
    GenericWorker::initialize();

    if (not G)
    {
        qWarning() << "table_concept: DSR graph not available in initialize()";
        return;
    }

    rt_api_ = G->get_rt_api();

    // Subscribe to graph signals
    connect(G.get(), &DSR::DSRGraph::update_node_signal,
            this, &SpecificWorker::modify_node_slot);
    connect(G.get(), &DSR::DSRGraph::update_node_attr_signal,
            this, &SpecificWorker::modify_node_attrs_slot);
    connect(G.get(), &DSR::DSRGraph::del_node_signal,
            this, &SpecificWorker::del_node_slot);

    // Resolve room node
    const auto rooms = G->get_nodes_by_type("room");
    if (not rooms.empty())
        room_node_id_ = rooms.front().id();
    else
        qWarning() << "table_concept: no room node found at startup";

    // Setup prior store
    prior_store_ = std::make_unique<PriorStore>(priors_path_, checkpoint_path_);

    // Create any table nodes listed in priors that are absent from DSR
    scaffold_missing_table_nodes();

    // Build EpistemicPlanner with configured parameters
    epistemic_planner_ = EpistemicPlanner(cfg_.delta_min, cfg_.gain_threshold, cfg_.obs_distance);

    // Remove any stale affordance nodes left by a previous run
    for (const auto& aff_node : G->get_nodes_by_type("affordance"))
    {
        std::print("table_concept: removing stale affordance node '{}' id={}\n",
                   aff_node.name(), aff_node.id());
        G->delete_node(aff_node.id());
    }

    // ── Time-series widget ──────────────────────────────────────────────────
    if (not graph_viewers.empty())
    {
        custom_widget_ = new Custom_widget();
        graph_viewers.at("")->add_custom_widget_to_dock("Table Inference", custom_widget_);

        // Create plot inside frame_series
        auto* series_layout = new QVBoxLayout(custom_widget_->frame_series);
        series_layout->setContentsMargins(0, 0, 0, 0);
        custom_widget_->frame_series->setLayout(series_layout);

        ts_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
        ts_plot_->set_visible_window(60.f);
        series_layout->addWidget(ts_plot_);

        ts_cov_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
        ts_cov_plot_->set_visible_window(60.f);
        series_layout->addWidget(ts_cov_plot_);

        ts_res_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
        ts_res_plot_->set_visible_window(60.f);
        series_layout->addWidget(ts_res_plot_);

        // GenericWorker::initialize() may have already started compute(), so
        // some instances can exist before the plots are constructed.
        for (const auto& [_, inst] : instances_)
        {
            ts_plot_->add_series(inst.node_name + "_fe", QColor(255, 170, 0), 1.1f);
            ts_cov_plot_->add_series(inst.node_name + "_cov", QColor(0, 190, 255), 1.1f);
            ts_res_plot_->add_series(inst.node_name + "_res", QColor(170, 80, 255), 1.1f);
        }
    }
}

// ─── Main compute loop ───────────────────────────────────────────────────────

void SpecificWorker::compute()
{
    if (not G or not rt_api_)
        return;

    // Refresh room node id if not yet found
    if (room_node_id_ == 0)
    {
        const auto rooms = G->get_nodes_by_type("room");
        if (rooms.empty()) return;
        room_node_id_ = rooms.front().id();
    }

    const auto table_nodes = G->get_nodes_by_type("table");
    for (const auto& node : table_nodes)
        process_table_node(node);
}

///////////////////////////////////////////////////////////////
void SpecificWorker::process_table_node(const DSR::Node& node)
{
    ensure_instance(node);

    auto& inst = instances_.at(node.id());
    const auto observation = observe_table_node(inst, node);

    // Stale check: skip heavy update if data hasn't moved for too long
    if (not observation.has_fresh_data and inst.matched_frames < 5)
        return;

    const float free_energy = run_table_inference(inst, observation);
    publish_table_cycle(inst, node, observation, free_energy);
    inst.prev_free_energy = free_energy;
}

SpecificWorker::TableObservation SpecificWorker::observe_table_node(TableInstance& inst,
                                                                    const DSR::Node& node)
{
    TableObservation observation;

    int last_frame = -1;
    if (const auto v = G->get_attrib_by_name<last_sensing_frame_att>(node); v.has_value())
        last_frame = v.value();

    observation.has_fresh_data = (last_frame > inst.last_frame_seen);
    if (not observation.has_fresh_data)
        return observation;

    inst.last_frame_seen = last_frame;
    observation.candidate_pts = read_pts_attrib(node, "candidate_pts_att");
    observation.residual_pts  = read_pts_attrib(node, "residual_pts_att");

    if (const auto v = G->get_attrib_by_name<explanation_ratio_att>(node); v.has_value())
        observation.explanation_ratio = v.value();

    // ↓ Bottom-up: new sensory evidence arriving from robot_concept
    std::print("[{}] ↓ frame={} cands={} resid={} expl={:.2f}\n",
               inst.node_name, last_frame,
               observation.candidate_pts.size(), observation.residual_pts.size(),
               observation.explanation_ratio);
    return observation;
}

float SpecificWorker::run_table_inference(TableInstance& inst,
                                          const TableObservation& observation)
{
    if (observation.has_fresh_data and not observation.candidate_pts.empty())
    {
        // Cold-start: on first observation snap model & prior to voxel centroid
        // so gradient descent begins at the right place rather than the prior.
        if (inst.matched_frames == 0)
        {
            Eigen::Vector3f sum = Eigen::Vector3f::Zero();
            for (const auto& p : observation.candidate_pts)
                sum += p;
            const Eigen::Vector3f cen = sum / static_cast<float>(observation.candidate_pts.size());
            auto s  = inst.model.state();
            s.cx    = cen.x();
            s.cy    = cen.y();
            inst.model.set_state(s);
            inst.model.set_prior(s);   // zero KL so data term dominates from the start
            std::print("[{}] cold-start snap → cx={:.2f} cy={:.2f} ({} pts)\n",
                       inst.node_name, s.cx, s.cy, observation.candidate_pts.size());
            // Pose is already correct: bypass warmup gate AND progress ramp
            // so the queue admits up to max_new_points_per_frame immediately.
            inst.matched_frames = cfg_.min_frames_before_historical
                                + cfg_.historical_warmup_frames + 1;
        }
        else
            ++inst.matched_frames;

        step_queue_update(inst, observation.candidate_pts,
                          TableBeliefPolicy::clamp01(observation.explanation_ratio));
    }

    const float residual_precision = TableBeliefPolicy::clamp01(
        inst.warm_confidence * TableBeliefPolicy::clamp01(observation.explanation_ratio));
    const float free_energy = step_model_update(inst, observation.residual_pts, residual_precision);

    // ↑ Top-down: model state after gradient descent
    const auto& s = inst.model.state();
    std::print("[{}] FE={:.4f}  cx={:.3f} cy={:.3f}  w={:.3f} h={:.3f} H={:.3f} L={:.3f} ψ={:.3f}  pts={}\n",
               inst.node_name, free_energy,
               s.cx, s.cy, s.w, s.h, s.table_height, s.leg_length, s.yaw,
               inst.queue.size() + static_cast<int>(observation.residual_pts.size()));

    return free_energy;
}

void SpecificWorker::publish_table_cycle(TableInstance& inst,
                                         const DSR::Node& node,
                                         const TableObservation& observation,
                                         float free_energy)
{
    const auto node_id = node.id();
    if (not persist_table_belief(inst, node_id, free_energy))
        return;
    if (not assess_table_state(inst, node_id, free_energy))
        return;
    publish_table_diagnostics(inst, observation, free_energy);
    publish_table_intentions(inst, node_id, observation, free_energy);
}

bool SpecificWorker::persist_table_belief(TableInstance& inst, uint64_t node_id, float free_energy)
{
    auto node_opt = G->get_node(node_id);
    if (not node_opt.has_value())
        return false;

    step_write_model(inst, node_opt.value(), free_energy);
    return true;
}

bool SpecificWorker::assess_table_state(TableInstance& inst, uint64_t node_id, float free_energy)
{
    auto node_opt = G->get_node(node_id);
    if (not node_opt.has_value())
        return false;

    step_convergence(inst, node_opt.value(), free_energy);
    return true;
}

void SpecificWorker::publish_table_diagnostics(const TableInstance& inst,
                                               const TableObservation& observation,
                                               float free_energy)
{

    if (ts_plot_)
    {
        ts_plot_->add_point(inst.node_name + "_fe",  free_energy);
        if (ts_cov_plot_)
            ts_cov_plot_->add_point(inst.node_name + "_cov", inst.last_coverage_deficit);
        if (ts_res_plot_)
            ts_res_plot_->add_point(inst.node_name + "_res", static_cast<float>(observation.residual_pts.size()));
    }

    std::print("[{}] series: FE={:.4f} cov={:.1f} res={}\n",
               inst.node_name,
               free_energy,
               inst.last_coverage_deficit,
               observation.residual_pts.size());
}

void SpecificWorker::publish_table_intentions(TableInstance& inst,
                                              uint64_t node_id,
                                              const TableObservation& observation,
                                              float free_energy)
{
    if (inst.last_coverage_deficit > 0.f)
    {
        auto node_opt = G->get_node(node_id);
        if (node_opt.has_value())
            step_epistemic(inst, node_opt.value());
    }
    else if (inst.affordance.is_active())
    {
        // All faces covered — remove the epistemic action request
        inst.affordance.remove();
    }

    if (observation.has_fresh_data)
    {
        auto node_opt = G->get_node(node_id);
        if (node_opt.has_value())
            step_refresh_check(inst, node_opt.value(), free_energy, observation.explanation_ratio);
    }
}

// ─── Initialisation helpers ──────────────────────────────────────────────────

void SpecificWorker::load_config(const ConfigLoader& cfg)
{
    // Helper lambdas to read with fallback (ConfigLoader::get has no default overload;
    // TOML numeric floats are stored as double, so cast explicitly).
    auto getf = [&](const std::string& k, float def) -> float {
        return cfg.exists(k) ? static_cast<float>(cfg.get<double>(k)) : def;
    };
    auto geti = [&](const std::string& k, int def) -> int {
        return cfg.exists(k) ? cfg.get<int>(k) : def;
    };
    auto gets = [&](const std::string& k, std::string def) -> std::string {
        return cfg.exists(k) ? cfg.get<std::string>(k) : def;
    };

    // Paths
    priors_path_     = gets("TableConcept.PriorsPath",     "etc/object_priors.toml");
    checkpoint_path_ = gets("TableConcept.CheckpointPath", "etc/checkpoint.toml");

    // Agent convergence
    cfg_.fe_eps                   = getf("TableConcept.FEps",                   1e-3f);
    cfg_.K_stable                 = geti("TableConcept.KStable",                30);
    cfg_.M_diverge                = geti("TableConcept.MDiverge",               20);
    cfg_.staleness_frames         = getf("TableConcept.StalenessFrames",        90.0f);
    cfg_.explanation_ratio_thresh = getf("TableConcept.ExplanationRatioThresh", 0.3f);
    cfg_.write_threshold          = getf("TableConcept.WriteThreshold",         1e-3f);
    cfg_.obs_distance             = getf("TableConcept.ObsDistance",            1.8f);
    cfg_.delta_min                = getf("TableConcept.DeltaMin",               20.0f);
    cfg_.gain_threshold           = getf("TableConcept.GainThreshold",          0.1f);

    // TableModel
    cfg_.sigma_obs          = getf("TableModel.SigmaObs",          0.05f);
    cfg_.lambda_size        = getf("TableModel.LambdaSize",        0.5f);
    cfg_.lambda_pos         = getf("TableModel.LambdaPos",         0.05f);
    cfg_.lambda_state       = getf("TableModel.LambdaState",       0.02f);
    cfg_.lambda_angle       = getf("TableModel.LambdaAngle",       0.01f);
    cfg_.optimization_iters = geti("TableModel.OptimizationIters", 10);
    cfg_.optimization_lr    = getf("TableModel.OptimizationLr",    0.05f);
    cfg_.grad_clip          = getf("TableModel.GradClip",          2.0f);
    cfg_.optimizer_type     = gets("TableModel.OptimizerType",     "adam");
    cfg_.sgd_momentum       = getf("TableModel.SgdMomentum",       0.9f);
    {
        const auto loss_name = gets("TableModel.RobustLoss", "quadratic");
        const auto loss_type = robust_loss_type_from_string(loss_name);
        if (loss_type.has_value())
            cfg_.robust_loss = loss_type.value();
        else
        {
            qWarning() << "table_concept: unknown robust loss" << loss_name.c_str() << "- using quadratic";
            cfg_.robust_loss = RobustLossType::Quadratic;
        }
    }
    cfg_.robust_loss_scale  = getf("TableModel.RobustLossScale",  0.10f);

    // SampleQueue
    cfg_.num_angle_bins               = geti("SampleQueue.NumAngleBins",              24);
    cfg_.num_z_bins                   = geti("SampleQueue.NumZBins",                  10);
    cfg_.max_per_bin                  = geti("SampleQueue.MaxPerBin",                 2);
    cfg_.sdf_threshold_for_storage    = getf("SampleQueue.SdfThresholdForStorage",    0.30f);  // voxels are volumetric; admit pts within 30 cm of any surface
    cfg_.min_frames_before_historical = geti("SampleQueue.MinFramesBeforeHistorical", 10);
    cfg_.historical_warmup_frames     = geti("SampleQueue.HistoricalWarmupFrames",    5);   // reach full capacity quickly
    cfg_.max_new_points_per_frame     = geti("SampleQueue.MaxNewPointsPerFrame",      30);  // enough pts to constrain gradient
    cfg_.rfe_alpha                    = getf("SampleQueue.RfeAlpha",                  0.98f);
    cfg_.rfe_max_threshold            = getf("SampleQueue.RfeMaxThreshold",           2.0f);
    cfg_.edge_bonus_weight            = getf("SampleQueue.EdgeBonusWeight",           0.3f);
    cfg_.edge_proximity_threshold     = getf("SampleQueue.EdgeProximityThreshold",    0.05f);

    // WarmStart
    cfg_.warm_pts_min                  = getf("WarmStart.PtsMin",                  12.0f);
    cfg_.warm_pts_max                  = getf("WarmStart.PtsMax",                  30.0f);
    cfg_.warm_coverage_min_side        = getf("WarmStart.CoverageMinSide",         2.0f);
    cfg_.warm_rho_freeze               = getf("WarmStart.RhoFreeze",               0.25f);
    cfg_.warm_lambda_pos_base          = getf("WarmStart.LambdaPosBase",           0.15f);
    cfg_.warm_lambda_pos_gain          = getf("WarmStart.LambdaPosGain",           0.45f);
    cfg_.warm_lambda_size_base         = getf("WarmStart.LambdaSizeBase",          0.02f);
    cfg_.warm_lambda_size_gain         = getf("WarmStart.LambdaSizeGain",          0.18f);
    cfg_.warm_lambda_yaw_base          = getf("WarmStart.LambdaYawBase",           0.01f);
    cfg_.warm_lambda_yaw_gain          = getf("WarmStart.LambdaYawGain",           0.12f);
    cfg_.warm_confidence_decay         = getf("WarmStart.ConfidenceDecay",         0.70f);
    cfg_.warm_confidence_coverage_gain = getf("WarmStart.ConfidenceCoverageGain",  0.35f);
    cfg_.warm_confidence_residual_gain = getf("WarmStart.ConfidenceResidualGain",  0.65f);

    std::print("table_concept: configuration loaded.\n");
}

void SpecificWorker::scaffold_missing_table_nodes()
{
    if (not prior_store_) return;
    const auto priors = prior_store_->load_priors();

    for (const auto& p : priors)
    {
        if (G->get_node(p.node_name).has_value())
        {
            std::print("table_concept: node '{}' already in DSR\n", p.node_name);
            continue;
        }

        // Node does not exist — create it from the prior
        auto room_opt = G->get_node(room_node_id_);
        if (not room_opt.has_value())
        {
            qWarning() << "table_concept: room node missing, cannot scaffold" << p.node_name.c_str();
            continue;
        }

        DSR::Node table_node = DSR::Node::create<table_node_type>(p.node_name);
        G->add_or_modify_attrib_local<width_m_att> (table_node, p.width_m);
        G->add_or_modify_attrib_local<depth_m_att> (table_node, p.depth_m);
        G->add_or_modify_attrib_local<height_m_att>(table_node, p.height_m);
        G->add_or_modify_attrib_local<level_att>   (table_node, 3);
        G->add_or_modify_attrib_local<parent_att>  (table_node, room_node_id_);
        // Canvas position: derive from room node + fixed offset so the viewer
        // doesn't randomize pos_x/pos_y on every render tick.
        {
            const float rpx = G->get_attrib_by_name<pos_x_att>(room_opt.value()).value_or(200.f);
            const float rpy = G->get_attrib_by_name<pos_y_att>(room_opt.value()).value_or(200.f);
            G->add_or_modify_attrib_local<pos_x_att>(table_node, rpx + 150.f);
            G->add_or_modify_attrib_local<pos_y_att>(table_node, rpy +  50.f);
        }

        const auto id_opt = G->insert_node(table_node);
        if (not id_opt.has_value())
        {
            qWarning() << "table_concept: failed to insert node" << p.node_name.c_str();
            continue;
        }

        const float z = p.height_m * 0.5f;
        rt_api_->insert_or_assign_edge_RT(room_opt.value(), id_opt.value(),
                                          {p.room_x_m, p.room_y_m, z},
                                          {0.0f, 0.0f, p.yaw_rad});

        trigger_graph_layout_twopi();

        std::print("table_concept: created node '{}' id={} at ({}, {})\n",
                   p.node_name, id_opt.value(), p.room_x_m, p.room_y_m);
    }
}

void SpecificWorker::ensure_instance(const DSR::Node& node)
{
    if (instances_.count(node.id()))
        return;

    // Build initial state from prior (or from checkpoint if available)
    TableState init_state;
    init_state.cx  = 0.0f;
    init_state.cy  = 0.0f;
    init_state.yaw = 0.0f;

    // Read geometry attributes that may already be in the node
    if (auto v = G->get_attrib_by_name<width_m_att> (node); v.has_value()) init_state.w            = v.value();
    if (auto v = G->get_attrib_by_name<depth_m_att> (node); v.has_value()) init_state.h            = v.value();
    if (auto v = G->get_attrib_by_name<height_m_att>(node); v.has_value()) init_state.table_height = v.value();

    // Read RT pose from room→table edge
    if (room_node_id_ != 0)
    {
        if (const auto edge = G->get_edge(room_node_id_, node.id(), "RT"); edge.has_value())
        {
            if (const auto tr = G->get_attrib_by_name<rt_translation_att>(edge.value()); tr.has_value())
            {
                const auto& tvec = tr.value().get();
                if (tvec.size() >= 2)
                {
                    init_state.cx = tvec[0];
                    init_state.cy = tvec[1];
                }
            }
            if (const auto rot = G->get_attrib_by_name<rt_rotation_euler_xyz_att>(edge.value()); rot.has_value())
            {
                const auto& rvec = rot.value().get();
                if (rvec.size() >= 3)
                    init_state.yaw = rvec[2];
            }
        }
    }

    init_state.leg_length = std::max(0.05f, init_state.table_height - TableModel::TOP_THICKNESS);

    // Check for a convergence checkpoint
    if (prior_store_)
    {
        const auto ckpt = prior_store_->load_checkpoint(node.name());
        if (ckpt.has_value())
        {
            init_state.w            = ckpt->width_m;
            init_state.h            = ckpt->depth_m;
            init_state.table_height = ckpt->height_m;
            init_state.cx           = ckpt->room_x_m;
            init_state.cy           = ckpt->room_y_m;
            init_state.yaw          = ckpt->yaw_rad;
            init_state.leg_length   = std::max(0.05f, ckpt->height_m - TableModel::TOP_THICKNESS);
            std::print("table_concept: restored checkpoint for '{}'\n", node.name());
        }
    }

    TableInstance inst;
    inst.node_id   = node.id();
    inst.node_name = node.name();
    inst.model     = TableModel(init_state, make_model_params());
    inst.queue     = SampleQueue(make_queue_params());
    inst.affordance.init(G, node.id(), node.name());

    instances_.emplace(node.id(), std::move(inst));
    std::print("table_concept: created instance for node '{}' id={}\n", node.name(), node.id());

    // Register per-instance time-series (one FE + one coverage series per table)
    if (ts_plot_)
    {
        ts_plot_->add_series(node.name() + "_fe",  QColor(255, 170,   0), 1.1f);
        if (ts_cov_plot_)
            ts_cov_plot_->add_series(node.name() + "_cov", QColor(  0, 190, 255), 1.1f);
        if (ts_res_plot_)
            ts_res_plot_->add_series(node.name() + "_res", QColor(170,  80, 255), 1.1f);
    }

    // Ensure canvas position is set — viewer randomizes pos_x/pos_y if absent.
    if (not G->get_attrib_by_name<pos_x_att>(node).has_value())
    {
        auto n_mut = node;
        float rpx = 200.f, rpy = 200.f;
        if (room_node_id_ != 0)
            if (const auto rn = G->get_node(room_node_id_); rn.has_value())
            {
                rpx = G->get_attrib_by_name<pos_x_att>(rn.value()).value_or(200.f);
                rpy = G->get_attrib_by_name<pos_y_att>(rn.value()).value_or(200.f);
            }
        G->add_or_modify_attrib_local<pos_x_att>(n_mut, rpx + 150.f);
        G->add_or_modify_attrib_local<pos_y_att>(n_mut, rpy +  50.f);
        G->update_node(n_mut);
    }
}

// ─── Per-cycle steps ─────────────────────────────────────────────────────────

void SpecificWorker::step_queue_update(TableInstance& inst,
                                       const std::vector<Eigen::Vector3f>& candidate_pts,
                                       float observation_precision)
{
    // Compute SDF for candidates under the current model
    const auto sdf_vals = inst.model.compute_sdf(candidate_pts);
    const float precision = std::max(0.05f, observation_precision);
    Eigen::Matrix2f robot_cov = read_robot_covariance();
    // In AIF terms, low explanatory adequacy means low sensory precision.
    // Represent that by inflating the capture covariance of new evidence.
    robot_cov /= precision;
    const int q_before = inst.queue.size();
    inst.queue.insert(candidate_pts, sdf_vals, robot_cov, inst.model, inst.matched_frames);
    const int admitted = inst.queue.size() - q_before;
    // New points from a fresh view → unlock the optimizer so it can re-converge.
    if (admitted > 0 && inst.frames_converged >= cfg_.K_stable)
        inst.frames_converged = cfg_.K_stable / 2;
    std::print("[{}] queue: admitted={} size={} obs_precision={:.2f}\n",
               inst.node_name,
               admitted,
               inst.queue.size(),
               observation_precision);
}

float SpecificWorker::step_model_update(TableInstance& inst,
                                         const std::vector<Eigen::Vector3f>& residual_pts,
                                         float residual_precision)
{
    const TableState previous_state = inst.model.state();

    const auto evidence = compose_belief_evidence(inst, residual_pts, residual_precision);
    if (not evidence.has_evaluation())
    {
        refresh_table_memory(inst);
        return inst.model.compute_free_energy({}, {});
    }

    evolve_table_belief(inst, evidence);
    const float free_energy = accept_table_belief(inst, previous_state, evidence);
    refresh_table_memory(inst);
    return free_energy;
}

SpecificWorker::TableBeliefEvidence SpecificWorker::compose_belief_evidence(
    const TableInstance& inst,
    const std::vector<Eigen::Vector3f>& residual_pts,
    float residual_precision) const
{
    TableBeliefEvidence evidence;
    evidence.fit_pts = inst.queue.points();
    evidence.fit_weights = inst.queue.weights();
    evidence.eval_pts = evidence.fit_pts;
    evidence.eval_weights = evidence.fit_weights;
    evidence.residual_count = static_cast<int>(residual_pts.size());
    evidence.residual_precision = residual_precision;

    for (const auto& residual_pt : residual_pts)
    {
        evidence.eval_pts.push_back(residual_pt);
        evidence.eval_weights.push_back(1.0f);
        if (residual_precision > 1e-3f)
        {
            evidence.fit_pts.push_back(residual_pt);
            evidence.fit_weights.push_back(residual_precision);
        }
    }

    evidence.trusted_point_count = static_cast<int>(evidence.fit_pts.size());
    return evidence;
}

void SpecificWorker::evolve_table_belief(TableInstance& inst, const TableBeliefEvidence& evidence)
{
    // Freeze gradient descent once converged to prevent oscillation.
    // Unlocked automatically by step_queue_update when new points arrive.
    if (inst.frames_converged < cfg_.K_stable && evidence.can_optimize())
        inst.model.gradient_step(evidence.fit_pts, evidence.fit_weights);
    else
        inst.model.compute_free_energy(evidence.fit_pts, evidence.fit_weights);
}

float SpecificWorker::accept_table_belief(TableInstance& inst,
                                          const TableState& previous_state,
                                          const TableBeliefEvidence& evidence)
{
    const TableState raw_state = inst.model.state();
    const auto coverage = inst.queue.face_coverage(inst.model);

    inst.warm_confidence = TableBeliefPolicy::update_warm_confidence(
        inst.warm_confidence, cfg_, coverage,
        evidence.trusted_point_count,
        evidence.residual_count,
        evidence.residual_precision);

    const TableState accepted_state = TableBeliefPolicy::apply_observability_warm_start(
        previous_state, raw_state, inst.model.params(), cfg_, inst.warm_confidence,
        coverage, evidence.trusted_point_count);
    inst.model.set_state(accepted_state);
    inst.model.set_prior(accepted_state);

    const float free_energy = inst.model.compute_free_energy(evidence.eval_pts, evidence.eval_weights);

    std::print("[{}] warm-start: conf={:.2f} rho_x={:.2f} rho_y={:.2f} residual={} trusted_pts={} residual_precision={:.2f} raw(w={:.3f},h={:.3f},psi={:.3f}) accepted(w={:.3f},h={:.3f},psi={:.3f})\n",
               inst.node_name,
               inst.warm_confidence,
               std::min(coverage[0], coverage[1]) / (std::max(coverage[0], coverage[1]) + 1e-3f),
               std::min(coverage[2], coverage[3]) / (std::max(coverage[2], coverage[3]) + 1e-3f),
               evidence.residual_count,
               evidence.trusted_point_count,
               evidence.residual_precision,
               raw_state.w, raw_state.h, raw_state.yaw,
               accepted_state.w, accepted_state.h, accepted_state.yaw);

    return free_energy;
}

void SpecificWorker::refresh_table_memory(TableInstance& inst)
{
    const Eigen::Matrix2f robot_cov = read_robot_covariance();
    inst.queue.update_rfe(inst.model, robot_cov);
}

void SpecificWorker::step_write_model(TableInstance& inst,
                                       DSR::Node& node,
                                       float free_energy)
{
    const auto& s = inst.model.state();

    // Geometry attributes
    G->add_or_modify_attrib_local<width_m_att> (node, s.w);
    G->add_or_modify_attrib_local<depth_m_att> (node, s.h);
    G->add_or_modify_attrib_local<height_m_att>(node, s.table_height);
    G->add_or_modify_attrib_local<free_energy_att>(node, free_energy);
    G->add_or_modify_attrib_local<model_generation_att>(node, ++inst.model_generation);

    // Mesh for voxelizer 3D viewer
    write_table_mesh(inst, node);

    // Export the current historical RFE queue as XYZ triples in table-node
    // attributes dedicated to remembered evidence.
    {
        const auto qpts = inst.queue.points();
        std::vector<float> qflat;
        qflat.reserve(qpts.size() * 3);
        for (const auto& p : qpts)
        {
            qflat.push_back(p.x());
            qflat.push_back(p.y());
            qflat.push_back(p.z());
        }
        G->runtime_checked_add_or_modify_attrib_local(node, "rfe_pts", qflat);
    }

    G->update_node(node);

    // RT edge (pose)
    write_rt_pose(room_node_id_, inst);
}

// ─── Table mesh generator ────────────────────────────────────────────────────
//
// Returns a flat triangle list in room frame:
//   [x0,y0,z0, x1,y1,z1, x2,y2,z2, ...]  — every 9 floats = 1 triangle.
//
// Geometry: 1 top slab + 4 square legs = 5 boxes × 12 triangles = 540 floats.

std::vector<float> SpecificWorker::make_table_mesh(const TableState& s)
{
    std::vector<float> verts;
    verts.reserve(5 * 108);   // 5 boxes × 12 tri × 3 vtx × 3 floats

    const float cy = std::cos(s.yaw);
    const float sy = std::sin(s.yaw);

    // Emit one box: centroid in room frame (bx,by,bz), local half-extents (hw,hd,hh).
    auto push_box = [&](float bx, float by, float bz,
                        float hw, float hd, float hh)
    {
        // Transform a local-frame corner to room frame and push xyz.
        auto push = [&](float lx, float ly, float lz)
        {
            verts.push_back(bx + cy * lx - sy * ly);
            verts.push_back(by + sy * lx + cy * ly);
            verts.push_back(bz + lz);
        };
        // 6 faces × 2 triangles (winding consistent but not critical for wire/fill)
        push(-hw,-hd,-hh); push( hw,-hd,-hh); push( hw, hd,-hh);  // bottom
        push(-hw,-hd,-hh); push( hw, hd,-hh); push(-hw, hd,-hh);
        push(-hw,-hd, hh); push( hw, hd, hh); push( hw,-hd, hh);  // top
        push(-hw,-hd, hh); push(-hw, hd, hh); push( hw, hd, hh);
        push(-hw,-hd,-hh); push( hw,-hd,-hh); push( hw,-hd, hh);  // front -y
        push(-hw,-hd,-hh); push( hw,-hd, hh); push(-hw,-hd, hh);
        push( hw, hd,-hh); push(-hw, hd,-hh); push(-hw, hd, hh);  // back  +y
        push( hw, hd,-hh); push(-hw, hd, hh); push( hw, hd, hh);
        push(-hw,-hd,-hh); push(-hw,-hd, hh); push(-hw, hd, hh);  // left  -x
        push(-hw,-hd,-hh); push(-hw, hd, hh); push(-hw, hd,-hh);
        push( hw,-hd,-hh); push( hw, hd,-hh); push( hw, hd, hh);  // right +x
        push( hw,-hd,-hh); push( hw, hd, hh); push( hw,-hd, hh);
    };

    // Top slab — centred at floor + leg_length + half slab thickness
    const float ht  = TableModel::TOP_THICKNESS * 0.5f;
    push_box(s.cx, s.cy, s.leg_length + ht,
             s.w * 0.5f, s.h * 0.5f, ht);

    // 4 legs — square cross-section (2×LEG_RADIUS), inset from table corners
    const float lr  = TableModel::LEG_RADIUS;
    const float lhz = s.leg_length * 0.5f;
    for (int ix : {-1, 1})
        for (int iy : {-1, 1})
        {
            const float lx = ix * (s.w * 0.5f - lr);
            const float ly = iy * (s.h * 0.5f - lr);
            // Rotate leg offset to room frame
            const float rx = s.cx + cy * lx - sy * ly;
            const float ry = s.cy + sy * lx + cy * ly;
            push_box(rx, ry, lhz,  lr, lr, lhz);
        }

    return verts;
}

void SpecificWorker::write_table_mesh(TableInstance& inst, DSR::Node& node)
{
    // Throttle: only update when the model generation changes (already guaranteed
    // by the caller), but skip if the mesh would be identical to save DSR bandwidth.
    const std::vector<float> verts = make_table_mesh(inst.model.state());
    G->add_or_modify_attrib_local<mesh_vertices_att>(node, verts);
}

void SpecificWorker::step_convergence(TableInstance& inst,
                                       DSR::Node& node,
                                       float free_energy)
{
    const float fe_delta = std::abs(free_energy - inst.prev_free_energy);
    if (fe_delta < cfg_.fe_eps)
    {
        inst.frames_converged = std::min(inst.frames_converged + 1, cfg_.K_stable);
    }
    else
    {
        inst.frames_converged = 0;
        inst.model_stable     = false;
    }

    // Check all four vertical faces are covered (for affordance/uncertainty reporting only)
    const auto coverage = inst.queue.face_coverage(inst.model);

    // Compute per-face coverage deficit for model_uncertainty_att
    float total_deficit = 0.0f;
    for (int i = 0; i < 4; ++i)
        total_deficit += std::max(0.0f, cfg_.delta_min - coverage[i]);
    inst.last_coverage_deficit = total_deficit;   // expose to plot
    G->add_or_modify_attrib_local<model_uncertainty_att>(node, total_deficit);

    // ↑ Top-down: generative model prediction vs. coverage evidence
    std::print("[{}] coverage: +x={:.1f} -x={:.1f} +y={:.1f} -y={:.1f}  "
               "stable={}/{} U={:.1f}\n",
               inst.node_name,
               coverage[0], coverage[1], coverage[2], coverage[3],
               inst.frames_converged, cfg_.K_stable,
               total_deficit);

    if (inst.frames_converged >= cfg_.K_stable)
    {
        if (not inst.model_stable)
        {
            inst.model_stable = true;
            G->add_or_modify_attrib_local<model_stable_att>(node, true);
            G->update_node(node);
            std::print("table_concept: node '{}' STABLE (F={:.4f})\n",
                       inst.node_name, free_energy);
        }
    }
    else
    {
        if (inst.model_stable)
        {
            G->add_or_modify_attrib_local<model_stable_att>(node, false);
            G->update_node(node);
        }
        inst.model_stable = false;
    }
}

void SpecificWorker::step_epistemic(TableInstance& inst, DSR::Node& node)
{
    const auto prop = epistemic_planner_.compute(inst.model, inst.queue);
    if (not prop.valid)
        return;

    // Write attributes to the table node (read by legacy consumers)
    write_epistemic_proposal(node, prop);
    // Publish / refresh dedicated affordance node
    const auto affordance_node_before = inst.affordance.node_id();
    inst.affordance.update(prop);
    if (affordance_node_before == 0 && inst.affordance.node_id() != 0)
        trigger_graph_layout_twopi();
    inst.epistemic_pending = true;
}

void SpecificWorker::step_refresh_check(TableInstance& inst,
                                         DSR::Node& node,
                                         float free_energy,
                                         float explanation_ratio)
{
    if (free_energy > inst.prev_free_energy)
        ++inst.frames_rising;
    else
        inst.frames_rising = 0;

    if (inst.frames_rising >= cfg_.M_diverge and explanation_ratio < cfg_.explanation_ratio_thresh)
    {
        G->add_or_modify_attrib_local<request_full_sample_att>(node, true);
        G->update_node(node);
        inst.frames_rising   = 0;
        inst.queue.clear();
        inst.matched_frames  = 0;
        std::print("table_concept: divergence detected for '{}' — requesting full resample\n",
                   inst.node_name);
    }
}

// ─── DSR helpers ─────────────────────────────────────────────────────────────

std::vector<Eigen::Vector3f> SpecificWorker::read_pts_attrib(
    const DSR::Node& node, const std::string& att_name) const
{
    std::vector<Eigen::Vector3f> pts;

    // Retrieve vector<float> attribute by name; interleaved XYZ
    std::optional<std::reference_wrapper<const std::vector<float>>> opt;
    if (att_name == "candidate_pts_att")
        opt = G->get_attrib_by_name<candidate_pts_att>(node);
    else if (att_name == "residual_pts_att")
        opt = G->get_attrib_by_name<residual_pts_att>(node);
    else
        return pts;

    if (not opt.has_value())
        return pts;

    const auto& data = opt.value().get();
    const std::size_t n = data.size() / 3;
    pts.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        pts.emplace_back(data[i*3], data[i*3+1], data[i*3+2]);

    return pts;
}

Eigen::Matrix2f SpecificWorker::read_robot_covariance() const
{
    // Try to read SE2 covariance from the room→robot RT edge
    const auto robots = G->get_nodes_by_type("robot");
    if (not robots.empty() and room_node_id_ != 0)
    {
        const auto edge = G->get_edge(room_node_id_, robots.front().id(), "RT");
        if (edge.has_value())
        {
            const auto cov_opt = G->get_attrib_by_name<rt_se2_covariance_att>(edge.value());
            if (cov_opt.has_value())
            {
                const auto& c = cov_opt.value().get();
                // rt_se2_covariance is a 9-vector (3×3 row-major for [x,y,θ])
                if (c.size() >= 4)
                {
                    Eigen::Matrix2f m;
                    m << c[0], c[1], c[3], c[4];
                    return m;
                }
            }
        }
    }
    // Fallback: small identity (high confidence)
    return Eigen::Matrix2f::Identity() * 0.01f;
}

void SpecificWorker::write_rt_pose(uint64_t room_id, TableInstance& inst)
{
    if (room_id == 0 or not rt_api_)
        return;

    const auto& s = inst.model.state();

    // Dead-band: suppress RT edge updates when position hasn't moved by more
    // than 5 cm — prevents pos_x/pos_y churn from small gradient oscillations.
    constexpr float kMinWriteDistSq = 0.05f * 0.05f;
    const float dx = s.cx - inst.last_written_cx;
    const float dy = s.cy - inst.last_written_cy;
    if (dx*dx + dy*dy < kMinWriteDistSq)
        return;

    auto room_opt = G->get_node(room_id);
    if (not room_opt.has_value())
        return;

    const float z = s.table_height * 0.5f;
    rt_api_->insert_or_assign_edge_RT(room_opt.value(), inst.node_id,
                                      {s.cx, s.cy, z},
                                      {0.0f, 0.0f, s.yaw});
    inst.last_written_cx = s.cx;
    inst.last_written_cy = s.cy;
}

void SpecificWorker::write_epistemic_proposal(DSR::Node& node,
                                               const EpistemicProposal& prop)
{
    G->add_or_modify_attrib_local<epistemic_target_x_m_att>  (node, prop.target_x_m);
    G->add_or_modify_attrib_local<epistemic_target_y_m_att>  (node, prop.target_y_m);
    G->add_or_modify_attrib_local<epistemic_target_yaw_rad_att>(node, prop.target_yaw_rad);
    G->add_or_modify_attrib_local<epistemic_gain_att>        (node, prop.gain);
    G->add_or_modify_attrib_local<epistemic_pending_att>     (node, true);
    G->update_node(node);
}

// ─── Factory helpers ─────────────────────────────────────────────────────────

TableModelParams SpecificWorker::make_model_params() const
{
    TableModelParams p;
    p.sigma_obs          = cfg_.sigma_obs;
    p.lambda_size        = cfg_.lambda_size;
    p.lambda_pos         = cfg_.lambda_pos;
    p.lambda_state       = cfg_.lambda_state;
    p.lambda_angle       = cfg_.lambda_angle;
    p.optimization_iters = cfg_.optimization_iters;
    p.optimization_lr    = cfg_.optimization_lr;
    p.grad_clip          = cfg_.grad_clip;
    p.optimizer_type     = cfg_.optimizer_type;
    p.sgd_momentum       = cfg_.sgd_momentum;
    p.robust_loss        = cfg_.robust_loss;
    p.robust_loss_scale  = cfg_.robust_loss_scale;
    return p;
}

SampleQueueParams SpecificWorker::make_queue_params() const
{
    SampleQueueParams p;
    p.num_angle_bins               = cfg_.num_angle_bins;
    p.num_z_bins                   = cfg_.num_z_bins;
    p.max_per_bin                  = cfg_.max_per_bin;
    p.sdf_threshold_for_storage    = cfg_.sdf_threshold_for_storage;
    p.min_frames_before_historical = cfg_.min_frames_before_historical;
    p.historical_warmup_frames     = cfg_.historical_warmup_frames;
    p.max_new_points_per_frame     = cfg_.max_new_points_per_frame;
    p.rfe_alpha                    = cfg_.rfe_alpha;
    p.rfe_max_threshold            = cfg_.rfe_max_threshold;
    p.edge_bonus_weight            = cfg_.edge_bonus_weight;
    p.edge_proximity_threshold     = cfg_.edge_proximity_threshold;
    return p;
}

void SpecificWorker::trigger_graph_layout_twopi()
{
    const auto it = graph_viewers.find("");
    if (it == graph_viewers.end() || !it->second)
        return;

    QWidget* graph_widget = it->second->get_widget(DSR::DSRViewer::view::graph);
    auto* graph_viewer = qobject_cast<DSR::GraphViewer*>(graph_widget);
    if (!graph_viewer)
        return;

    // Run now and once queued, so layout also happens after pending node/edge
    // update signals are processed by the viewer.
    graph_viewer->compute_layout("twopi");
    QMetaObject::invokeMethod(graph_viewer,
                              [graph_viewer]() { graph_viewer->compute_layout("twopi"); },
                              Qt::QueuedConnection);
}
// ─── DSR signal slots ────────────────────────────────────────────────────────

void SpecificWorker::modify_node_slot(std::uint64_t id, const std::string& type)
{
    if (type != "table")
        return;

    const auto node_opt = G->get_node(id);
    if (not node_opt.has_value())
        return;

    ensure_instance(node_opt.value());
}

void SpecificWorker::modify_node_attrs_slot(std::uint64_t id,
                                             const std::vector<std::string>& att_names)
{
    // Delegate to the affordance state machine for any instance whose affordance
    // node was modified (controller setting epistemic_pending=false)
    for (auto& [table_id, inst] : instances_)
        if (inst.affordance.node_id() == id)
            inst.affordance.on_node_modified(id);

    // React to mission-controller clearing epistemic_pending on the table node itself
    if (instances_.count(id))
    {
        const bool pending_cleared = std::any_of(att_names.begin(), att_names.end(),
            [](const std::string& s) { return s == "epistemic_pending"; });

        if (pending_cleared)
        {
            auto node_opt = G->get_node(id);
            if (node_opt.has_value())
            {
                const auto v = G->get_attrib_by_name<epistemic_pending_att>(node_opt.value());
                if (v.has_value() and not v.value())
                    instances_.at(id).epistemic_pending = false;
            }
        }
    }
}

void SpecificWorker::del_node_slot(std::uint64_t id)
{
    // Notify affordance in case its own DSR node was deleted externally
    for (auto& [table_id, inst] : instances_)
        if (inst.affordance.node_id() == id)
            inst.affordance.on_node_deleted(id);

    if (instances_.count(id))
    {
        std::print("table_concept: node {} removed from DSR, destroying instance\n", id);
        instances_.erase(id);
    }
}

// ─── Lifecycle stubs ─────────────────────────────────────────────────────────

void SpecificWorker::emergency()
{
    std::print("table_concept: emergency()\n");
}

void SpecificWorker::restore()
{
    std::print("table_concept: restore()\n");
}

int SpecificWorker::startup_check()
{
    std::print("table_concept: startup_check()\n");
    return 0;
}




