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

#include <atomic>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <genericworker.h>
#include <Eigen/Dense>
#include <unordered_set>

#include "../../common/robust_metrics/robust_metrics.h"
#include "table_config.h"      // rc::TableConfig + load_table_config
#include "table_instance.h"    // rc::TableInstance
#include "../../common/mask_ingestor/mask_ingestor.h"     // rc::MaskIngestor (perception)
#include "../../common/instance_tracker/instance_tracker.h"   // rc::InstanceTracker (birth/associate/death)
#include "table_scene_graph.h" // rc::TableSceneGraph (DSR node/RT I/O)
#include "table_fitter.h"      // rc::TableFitter (active-inference core)
#include "epistemic_planner.h"
#include "prior_store.h"
#include "../../common/sample_queue/sample_queue.h"
#include "table_affordance.h"
#include "table_model.h"
#include "../../common/dashboard/custom_widget.h"
#include "../../common/dashboard/timeseries_plot.h"
#include "../../common/agent_presence_coordinator/agent_presence_coordinator.h"

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
    // The fit (observe/infer/belief/voxel-bank) lives in rc::TableFitter; perception in
    // rc::MaskIngestor; DSR I/O in rc::TableSceneGraph. The worker keeps orchestration + the
    // post-fit epistemic/affordance/Qt-diagnostics steps.
    using TableObservation = rc::TableFitter::TableObservation;

    // ── Orchestration + post-fit steps ────────────────────────────────────────
    void load_config(const ConfigLoader& cfg);
    void process_table_node(const DSR::Node& node);
    void publish_table_cycle(rc::TableInstance& inst,
                             const DSR::Node& node,
                             const TableObservation& observation,
                             float free_energy);
    bool assess_table_state(rc::TableInstance& inst, uint64_t node_id, float free_energy);
    void publish_table_diagnostics(const rc::TableInstance& inst,
                                   const TableObservation& observation,
                                   float free_energy);
    void publish_table_intentions(rc::TableInstance& inst,
                                  uint64_t node_id,
                                  const TableObservation& observation,
                                  float free_energy);
    void step_convergence(rc::TableInstance& inst, DSR::Node& node, float free_energy);
    void step_epistemic(rc::TableInstance& inst, DSR::Node& node);
    void step_refresh_check(rc::TableInstance& inst, DSR::Node& node,
                            float free_energy, float explanation_ratio);
    void trigger_graph_layout_twopi();   // injected into TableSceneGraph as the relayout callback

    // Multi-instance birth/associate/death (shared, when cfg_.tracker_enabled). Associates "table" masks
    // to instances, spawns a table from an unexplained mask, retires a long-unobserved one.
    rc::InstanceTracker tracker_;
    void run_instance_tracker();   // called from compute() in place of scaffold when tracker_enabled
    // Physical-exclusion invariant: two tables cannot share space. Collapse any pair of instances whose
    // oriented footprints overlap beyond Tracker.MergeOverlap, keeping the more-observed one.
    void merge_overlapping_instances();

    // ── Presence protocol ────────────────────────────────────────────────────
    void waiting_enter();
    void waiting_loop();
    void operating_enter();
    void operating_loop();
    void degraded_enter();
    void degraded_loop();
    void cleanup_owned_nodes();
    void remove_stale_affordance_nodes();   // sweep affordances parented to a table (start + exit)
    void remove_owned_table_nodes();        // startup stale-sweep of "table*" nodes (mirrors bottle)
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
    bool startup_affordance_sweep_done_ = false;   // one-time startup stale-affordance sweep; re-running
                                                   // on every Operating re-entry would wipe live affordances.
    std::atomic<bool> shutting_down_{false};
    AgentPresenceCoordinator presence_coordinator_;

    rc::TableConfig                                         cfg_;
    std::unique_ptr<rc::PriorStore>                         prior_store_;
    std::vector<rc::TablePrior>                             priors_cache_;
    rc::EpistemicPlanner                                    epistemic_planner_;
    std::unique_ptr<rc::TableFitter>                    fitter_;   // active-inference fit core (owns instances)

    Custom_widget*       custom_widget_ = nullptr;
    rc::TimeSeriesPlot*  ts_plot_       = nullptr;   // FE
    rc::TimeSeriesPlot*  ts_cov_plot_   = nullptr;   // coverage deficit
    rc::TimeSeriesPlot*  ts_res_plot_   = nullptr;   // residual point count
    rc::TimeSeriesPlot*  ts_state_plot_ = nullptr;   // inferred dimensions w/h (stability check)
    rc::TimeSeriesPlot*  ts_ce_plot_    = nullptr;   // (D) counter-evidence accumulator S_w/S_h (only when the gate is on)

    std::unique_ptr<DSR::RT_API>                        rt_api_;
    std::unique_ptr<DSR::InnerEigenAPI>                inner_eigen_;     // for room↔body↔zed extrinsic (silhouette)
    std::unique_ptr<rc::MaskIngestor>                   mask_ingestor_;   // perception (masks-only)
    std::unique_ptr<rc::TableSceneGraph>               scene_graph_;     // DSR node/RT I/O
    uint64_t                                            room_node_id_ = 0;

    // Paths resolved from ConfigLoader
    std::string priors_path_;

signals:
    void presenceReady();
    void presenceLost();
};

#endif // SPECIFICWORKER_H
