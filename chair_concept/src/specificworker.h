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
 * chair_concept — Active Inference agent for chair instance detection and maintenance.
 *
 * Owns the generative model (7-param state + compound SDF) for every chair
 * node in the DSR graph.  Runs a free-energy minimisation loop, maintains a
 * binned historical sample queue, and emits epistemic action proposals to
 * mission-controller when chair surfaces remain under-observed.
 *
 * See TABLE_CONCEPT.md for the full design specification.
 */

#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#include <atomic>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <genericworker.h>
#include <fps/fps.h>
#include <Eigen/Dense>
#include <unordered_set>

#include "../../common/robust_metrics/robust_metrics.h"
#include "chair_config.h"      // rc::ChairConfig + load_chair_config
#include "chair_instance.h"    // rc::ChairInstance
#include "../../common/mask_ingestor/mask_ingestor.h"     // rc::MaskIngestor (perception)
#include "chair_scene_graph.h" // rc::ChairSceneGraph (DSR node/RT I/O)
#include "chair_fitter.h"      // rc::ChairFitter (active-inference core)
#include "epistemic_planner.h"
#include "prior_store.h"
#include "../../common/sample_queue/sample_queue.h"
#include "chair_affordance.h"
#include "chair_model.h"
#include "../../common/dashboard/custom_widget.h"
#include "../../common/dashboard/timeseries_plot.h"
#include "../../common/agent_presence_coordinator/agent_presence_coordinator.h"
#include "../../common/instance_tracker/instance_tracker.h"   // rc::InstanceTracker (birth/associate/death)

// ─── SpecificWorker ──────────────────────────────────────────────────────────

class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:
    SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check);
    ~SpecificWorker();
    bool is_shutting_down() const noexcept { return shutting_down_.load(); }

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
    // The fit (observe/infer/belief/voxel-bank) lives in rc::ChairFitter; perception in
    // rc::MaskIngestor; DSR I/O in rc::ChairSceneGraph. The worker keeps orchestration + the
    // post-fit epistemic/affordance/Qt-diagnostics steps.
    using ChairObservation = rc::ChairFitter::ChairObservation;

    // ── Orchestration + post-fit steps ────────────────────────────────────────
    void load_config(const ConfigLoader& cfg);
    void process_chair_node(const DSR::Node& node);
    void run_instance_tracker();          // data-driven birth/associate/death when cfg_.tracker_enabled
    void merge_overlapping_instances();   // collapse two instances on the same chair (seat-footprint overlap)
    void publish_chair_cycle(rc::ChairInstance& inst,
                             const DSR::Node& node,
                             const ChairObservation& observation,
                             float free_energy);
    bool assess_chair_state(rc::ChairInstance& inst, uint64_t node_id, float free_energy);
    void publish_chair_diagnostics(const rc::ChairInstance& inst,
                                   const ChairObservation& observation,
                                   float free_energy);
    void publish_chair_intentions(rc::ChairInstance& inst,
                                  uint64_t node_id,
                                  const ChairObservation& observation,
                                  float free_energy);
    void step_convergence(rc::ChairInstance& inst, DSR::Node& node, float free_energy);
    void step_epistemic(rc::ChairInstance& inst, DSR::Node& node);
    void step_refresh_check(rc::ChairInstance& inst, DSR::Node& node,
                            float free_energy, float explanation_ratio);
    void trigger_graph_layout_twopi();   // injected into ChairSceneGraph as the relayout callback

    // ── Presence protocol ────────────────────────────────────────────────────
    void waiting_enter();
    void waiting_loop();
    void operating_enter();
    void operating_loop();
    void degraded_enter();
    void degraded_loop();
    void cleanup_owned_nodes();
    void remove_stale_affordance_nodes();   // sweep affordances parented to a chair (start + exit)
    void remove_owned_chair_nodes();        // startup stale-sweep of "chair*" nodes (mirrors bottle)
    void request_shutdown();
    // Crash-free exit (request_shutdown + DDS reset + _Exit), bypassing the Ice/static teardown abort.
    void terminal_shutdown();
    // Grace before a required-peer loss is treated as terminal (debounce transient presence flaps).
    static constexpr int REQUIRED_LOSS_GRACE_MS = 3000;
    void on_optional_peer_lost(const std::string &name, std::uint32_t id);
    void on_optional_peer_ready(const std::string &name, std::uint32_t id);

    // ── Members ──────────────────────────────────────────────────────────────
    bool startup_check_flag = false;
    bool owned_nodes_cleaned_ = false;
    std::atomic<bool> shutting_down_{false};
    AgentPresenceCoordinator presence_coordinator_;

    rc::ChairConfig                                         cfg_;
    std::unique_ptr<rc::PriorStore>                         prior_store_;
    std::vector<rc::ChairPrior>                             priors_cache_;
    rc::EpistemicPlanner                                    epistemic_planner_;
    std::unique_ptr<rc::ChairFitter>                    fitter_;   // active-inference fit core (owns instances)

    Custom_widget*       custom_widget_ = nullptr;
    rc::TimeSeriesPlot*  ts_plot_       = nullptr;   // FE
    rc::TimeSeriesPlot*  ts_cov_plot_   = nullptr;   // coverage deficit
    rc::TimeSeriesPlot*  ts_res_plot_   = nullptr;   // residual point count
    rc::TimeSeriesPlot*  ts_state_plot_ = nullptr;   // inferred dimensions w/h (stability check)
    rc::TimeSeriesPlot*  ts_ce_plot_    = nullptr;   // (D) counter-evidence accumulator S_w/S_h (only when the gate is on)

    std::unique_ptr<DSR::RT_API>                        rt_api_;
    std::unique_ptr<DSR::InnerEigenAPI>                inner_eigen_;     // for room↔body↔zed extrinsic (silhouette)
    std::unique_ptr<rc::MaskIngestor>                   mask_ingestor_;   // perception (masks-only)
    std::unique_ptr<rc::ChairSceneGraph>               scene_graph_;     // DSR node/RT I/O
    rc::InstanceTracker                                tracker_;         // multi-instance (Tracker.Enabled)
    uint64_t                                            room_node_id_ = 0;
    FPSCounter                                          fps_counter_;     // overall compute()-cycle rate

    // Paths resolved from ConfigLoader
    std::string priors_path_;

signals:
    void presenceReady();
    void presenceLost();
};

#endif // SPECIFICWORKER_H
