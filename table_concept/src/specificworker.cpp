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
#include <cstdlib>   // std::_Exit — crash-free terminal shutdown
#include <thread>    // brief DDS flush before _Exit
#include <chrono>
#include <iostream>  // std::cout/cerr flush

#include <algorithm>
#include <cmath>
#include <sstream>

// DSR attribute name tags — generated from dsr_attr_name.h
#include <dsr/api/dsr_api.h>


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

    const int period = configLoader.get<int>("Period.Compute");

    states["Waiting"] = std::make_unique<GRAFCETStep>("Waiting", period,
        std::bind(&SpecificWorker::waiting_loop, this),
        std::bind(&SpecificWorker::waiting_enter, this));
    states["Operating"] = std::make_unique<GRAFCETStep>("Operating", period,
        std::bind(&SpecificWorker::operating_loop, this),
        std::bind(&SpecificWorker::operating_enter, this));
    states["Degraded"] = std::make_unique<GRAFCETStep>("Degraded", period,
        std::bind(&SpecificWorker::degraded_loop, this),
        std::bind(&SpecificWorker::degraded_enter, this));

    // Compute → Waiting on start
    states["Compute"]->addTransition(states["Compute"].get(), SIGNAL(entered()), states["Waiting"].get());
    // Waiting → Operating when all required peers are ready
    states["Waiting"]->addTransition(this, SIGNAL(presenceReady()), states["Operating"].get());
    // Operating → Degraded when a required peer is lost
    states["Operating"]->addTransition(this, SIGNAL(presenceLost()), states["Degraded"].get());
    // Degraded → Waiting immediately (self-kill scheduled inside degraded_enter)
    states["Degraded"]->addTransition(states["Degraded"].get(), SIGNAL(entered()), states["Waiting"].get());

    statemachine.addState(states["Waiting"].get());
    statemachine.addState(states["Operating"].get());
    statemachine.addState(states["Degraded"].get());

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
    request_shutdown();
    std::print("table_concept: SpecificWorker destroyed.\n");
}

void SpecificWorker::request_shutdown()
{
    if (shutting_down_.exchange(true))
        return;

    save_window_settings();

    cleanup_owned_nodes();

    // Drop the InnerEigenAPI now (the fitter only holds a raw pointer and is null-guarded): letting it
    // destruct later with the rest of the object can fault inside DSR. Mirrors bottle_concept.
    inner_eigen_.reset();
}

void SpecificWorker::terminal_shutdown()
{
    static std::atomic<bool> terminating{false};
    if (terminating.exchange(true))
        return;   // _Exit is coming; never run this twice

    // Crash-free terminal exit (matches bottle_concept). After our cleanup (owned
    // nodes deleted, peers notified) hard-exit instead of returning into the Ice communicator teardown +
    // C++ static destruction, which run with undefined cross-TU order and abort (a global/DDS holder
    // copies a graph Node after the node-type registry static is gone → Node::type() throws).
    request_shutdown();
    if (G)
    {
        try { G->reset(); }   // clean DDS participant/entity removal without touching the Ice communicator
        catch (...) { /* best-effort: we are exiting regardless */ }
    }
    std::cout.flush();
    std::cerr.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));   // let the del-deltas reach peers
    std::_Exit(EXIT_SUCCESS);
}

// ─── Initialisation ──────────────────────────────────────────────────────────

void SpecificWorker::initialize()
{
    std::print("table_concept: initialize()\n");
    GenericWorker::initialize();
 
    // Ignore payload attributes in local graph updates to avoid unnecessary copying and processing of potentially large data
    G->set_ignored_attributes<cam_rgb_att, cam_depth_att, laser_X_att, laser_Y_att, laser_Z_att>();
    qInfo() << "Ignoring DSR RGBD payload attributes cam_rgb/cam_depth in local graph updates";


    if (not G)
    {
        qWarning() << "table_concept: DSR graph not available in initialize()";
        return;
    }

    presence_coordinator_.configure(configLoader, G, static_cast<std::uint32_t>(agent_id));
    presence_coordinator_.set_transition_hooks({
        .request_presence_ready = [this]() { emit presenceReady(); },
        .request_presence_lost  = [this]() { emit presenceLost(); },
    });
    presence_coordinator_.set_peer_hooks({
        .on_peer_restarted = [](std::uint32_t id)
        {
            qInfo() << "[Presence] peer" << id << "restarted";
        },
        .on_optional_peer_lost = [this](const std::string &name, std::uint32_t id)
        {
            on_optional_peer_lost(name, id);
        },
        .on_optional_peer_ready = [this](const std::string &name, std::uint32_t id)
        {
            on_optional_peer_ready(name, id);
        },
    });
    presence_coordinator_.set_lifecycle_hooks({
        .on_waiting_enter = [this]()
        {
            const auto missing = presence_coordinator_.missing_required_names();
            if (missing.empty())
                qInfo("[SM] -> Waiting");
            else
            {
                QString m;
                for (const auto &label : missing)
                    m += " " + QString::fromStdString(label);
                qInfo() << "[SM] -> Waiting (missing:" << m.trimmed() << ")";
            }
        },
        .on_operating_enter = []()
        {
            qInfo("[SM] -> Operating: all required peers present");
        },
        .on_operating_loop = [this]()
        {
            compute();
        },
        .on_degraded_enter = [this]()
        {
            if (shutting_down_)
                return;
            // Debounce: a transient required-peer flap (startup handshake, brief node churn) fires
            // presenceLost momentarily and then recovers; tearing down here would kill the agent on a
            // blip. Wait a grace period and only shut down if a required peer is STILL missing.
            qInfo("[SM] -> Degraded: required peer lost — %d ms grace before shutdown", REQUIRED_LOSS_GRACE_MS);
            QTimer::singleShot(REQUIRED_LOSS_GRACE_MS, this, [this]()
            {
                if (shutting_down_)
                    return;
                if (presence_coordinator_.all_required_ready())
                {
                    qInfo("[SM] required peers recovered during grace — staying alive");
                    return;
                }
                qWarning("[SM] required peer still missing after grace — shutting down cleanly");
                terminal_shutdown();
            });
        },
    });
    presence_coordinator_.start();

    QObject::connect(QCoreApplication::instance(), &QCoreApplication::aboutToQuit,
                     this, &SpecificWorker::terminal_shutdown, Qt::UniqueConnection);

    rt_api_ = G->get_rt_api();
    inner_eigen_ = G->get_inner_eigen_api();
    mask_ingestor_ = std::make_unique<rc::MaskIngestor>(G);
    scene_graph_ = std::make_unique<rc::TableSceneGraph>(
        G, rt_api_.get(), [this] { trigger_graph_layout_twopi(); });

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
    prior_store_ = std::make_unique<rc::PriorStore>(priors_path_);
    priors_cache_ = prior_store_->load_priors();

    // Active-inference fit core. Owns the instance map; collaborates with the ingestor + scene graph.
    fitter_ = std::make_unique<rc::TableFitter>(
        G, inner_eigen_.get(), cfg_, priors_cache_, mask_ingestor_.get(), scene_graph_.get());

    // Missing table nodes are scaffolded lazily from priors only after masks
    // provide some table evidence in the current scene.

    // Build rc::EpistemicPlanner with configured parameters
    epistemic_planner_ = rc::EpistemicPlanner(cfg_.delta_min, cfg_.gain_threshold, cfg_.obs_distance);

    // Remove stale affordance nodes created by this agent only.
    // Keep foreign affordance nodes untouched.
    constexpr std::string_view own_affordance_prefix = "table_afford";
    for (const auto& aff_node : G->get_nodes_by_type("affordance"))
    {
        if (not aff_node.name().starts_with(own_affordance_prefix))
            continue;

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
        for (const auto& [_, inst] : fitter_->instances())
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

    mask_ingestor_->refresh();
    scene_graph_->scaffold_missing_table_nodes(priors_cache_, mask_ingestor_->packet(), room_node_id_);

    const auto table_nodes = G->get_nodes_by_type("table");
    for (const auto& node : table_nodes)
        process_table_node(node);
}

///////////////////////////////////////////////////////////////
void SpecificWorker::process_table_node(const DSR::Node& node)
{
    const bool created = fitter_->ensure_instance(node, room_node_id_);
    auto& inst = fitter_->instances().at(node.id());

    if (created)
    {
        // Register per-instance time-series (Qt dashboards stay in the worker).
        if (ts_plot_)
        {
            ts_plot_->add_series(inst.node_name + "_fe",  QColor(255, 170,   0), 1.1f);
            if (ts_cov_plot_) ts_cov_plot_->add_series(inst.node_name + "_cov", QColor(  0, 190, 255), 1.1f);
            if (ts_res_plot_) ts_res_plot_->add_series(inst.node_name + "_res", QColor(170,  80, 255), 1.1f);
        }
        // Canvas position — viewer randomizes pos_x/pos_y if absent.
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

    ++inst.processed_cycles;
    const auto observation = fitter_->observe(inst, node);

    // Stale check: skip heavy update if data hasn't moved for too long
    if (not observation.has_fresh_data and inst.matched_frames < 5)
        return;

    const float free_energy = fitter_->run_inference(inst, observation);
    publish_table_cycle(inst, node, observation, free_energy);
    inst.prev_free_energy = free_energy;
}


void SpecificWorker::publish_table_cycle(rc::TableInstance& inst,
                                         const DSR::Node& node,
                                         const TableObservation& observation,
                                         float free_energy)
{
    const auto node_id = node.id();
    if (not scene_graph_->persist_table_belief(inst, node_id, room_node_id_, free_energy))
        return;
    if (not assess_table_state(inst, node_id, free_energy))
        return;
    publish_table_diagnostics(inst, observation, free_energy);
    publish_table_intentions(inst, node_id, observation, free_energy);
}

bool SpecificWorker::assess_table_state(rc::TableInstance& inst, uint64_t node_id, float free_energy)
{
    auto node_opt = G->get_node(node_id);
    if (not node_opt.has_value())
        return false;

    step_convergence(inst, node_opt.value(), free_energy);
    return true;
}

void SpecificWorker::publish_table_diagnostics(const rc::TableInstance& inst,
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

    if (fitter_->should_log_table(inst))
        std::print("[{}] series: FE={:.4f} cov={:.1f} res={}\n",
                   inst.node_name,
                   free_energy,
                   inst.last_coverage_deficit,
                   observation.residual_pts.size());
}

void SpecificWorker::publish_table_intentions(rc::TableInstance& inst,
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
    cfg_ = rc::load_table_config(cfg);
    priors_path_ = cfg_.priors_path;           // mirrored into the existing members for now
}


// ─── Per-cycle steps ─────────────────────────────────────────────────────────


void SpecificWorker::step_convergence(rc::TableInstance& inst,
                                       DSR::Node& node,
                                       float free_energy)
{
    // Convergence on STATE stability, not |ΔFE|: the free energy keeps jittering with queue
    // churn / point-count even when the fitted geometry is settled, so it never latched. Track
    // how much the accepted state moved between cycles instead.
    const auto& s = inst.model.state();
    const auto& p = inst.prev_conv_state;
    const float state_delta = inst.has_prev_conv_state
        ? (std::abs(s.cx - p.cx) + std::abs(s.cy - p.cy) + std::abs(s.w - p.w) + std::abs(s.h - p.h) +
           std::abs(s.table_height - p.table_height) + std::abs(s.yaw - p.yaw) + std::abs(s.leg_inset - p.leg_inset))
        : std::numeric_limits<float>::max();
    inst.prev_conv_state = s;
    inst.has_prev_conv_state = true;
    if (state_delta < cfg_.state_eps)
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
    if (fitter_->should_log_table(inst))
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

void SpecificWorker::step_epistemic(rc::TableInstance& inst, DSR::Node& node)
{
    const auto prop = epistemic_planner_.compute(inst.model, inst.queue);
    if (not prop.valid)
        return;

    if (not prop.is_finite())
    {
        qWarning() << "table_concept: rejecting non-finite epistemic proposal for"
                   << inst.node_name.c_str();
        return;
    }

    // Write attributes to the table node (read by legacy consumers)
    scene_graph_->write_epistemic_proposal(node, prop);
    // Publish / refresh dedicated affordance node
    const auto affordance_node_before = inst.affordance.node_id();
    inst.affordance.update(prop);
    if (affordance_node_before == 0 && inst.affordance.node_id() != 0)
        trigger_graph_layout_twopi();
    inst.epistemic_pending = true;
}

void SpecificWorker::step_refresh_check(rc::TableInstance& inst,
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

    fitter_->ensure_instance(node_opt.value(), room_node_id_);
}

void SpecificWorker::modify_node_attrs_slot(std::uint64_t id,
                                             const std::vector<std::string>& att_names)
{
    // Delegate to the affordance state machine for any instance whose affordance
    // node was modified (controller claim/completion updates active/pending)
    for (auto& [table_id, inst] : fitter_->instances())
        if (inst.affordance.node_id() == id)
            inst.affordance.on_node_modified(id);

    // React to mission-controller clearing epistemic_pending on the table node itself
    if (fitter_->instances().count(id))
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
                    fitter_->instances().at(id).epistemic_pending = false;
            }
        }
    }
}

void SpecificWorker::del_node_slot(std::uint64_t id)
{
    // Notify affordance in case its own DSR node was deleted externally
    for (auto& [table_id, inst] : fitter_->instances())
        if (inst.affordance.node_id() == id)
            inst.affordance.on_node_deleted(id);

    if (fitter_->instances().count(id))
    {
        std::print("table_concept: node {} removed from DSR, destroying instance\n", id);
        fitter_->instances().erase(id);
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




