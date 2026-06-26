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
 * SpecificWorker — bottle_concept agent: RoboComp lifecycle + presence protocol +
 * orchestration only. compute() wires the collaborators into the per-cycle pipeline:
 *   mask_ingestor_ (read masks) → scene_graph_ (scaffold bottle nodes) →
 *   fitter_ (per-bottle free-energy fit + write-back) → evaluator_ (validation drivers).
 */

#include "specificworker.h"
#include <cstdlib>   // std::_Exit for the crash-free terminal shutdown
#include <thread>    // std::this_thread::sleep_for — let DDS flush before _Exit
#include <chrono>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <print>
#include <sstream>

#include <QCoreApplication>
#include <QTimer>

#include <dsr/api/dsr_api.h>

SpecificWorker::SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check)
    : GenericWorker(configLoader, tprx)
{
    this->startup_check_flag = startup_check;
    if (this->startup_check_flag)
    {
        this->startup_check();
        return;
    }

    cfg_ = rc::load_bottle_config(configLoader);

#ifdef HIBERNATION_ENABLED
    hibernationChecker.start(500);
#endif

    // Agent-presence state machine: Waiting → Operating → Degraded.
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

    auto error = statemachine.errorString();
    if (error.length() > 0)
    {
        qWarning() << error;
        throw error;
    }
}

SpecificWorker::~SpecificWorker()
{
    request_shutdown();
    std::print("bottle_concept: SpecificWorker destroyed.\n");
}

void SpecificWorker::request_shutdown()
{
    if (shutting_down_.exchange(true))
        return;

    // Sever graph callbacks BEFORE any teardown. On Ctrl+C a del_node delta can be
    // delivered from a DSR/DDS internal thread and invoke del_node_slot (instances_.erase)
    // on this already-destructing object — the exit segfault. Dropping inner_eigen_ here
    // (while G is still fully alive) also unsubscribes its internal graph signals cleanly.
    if (G)
        disconnect(G.get(), nullptr, this, nullptr);
    inner_eigen_.reset();

    cleanup_owned_nodes();
}

void SpecificWorker::terminal_shutdown()
{
    static std::atomic<bool> terminating{false};
    if (terminating.exchange(true))
        return;   // _Exit is coming; never run this twice

    // 1) Sever our graph callbacks, delete our owned DSR nodes (publishes del-deltas) and notify
    //    peers. Idempotent (shutting_down_ guard).
    request_shutdown();

    // 2) Cleanly remove THIS agent's DDS participant and entities from the shared graph. This is the
    //    crucial step that a bare _Exit skips: without it, peers (voxelizer) keep seeing our
    //    half-deleted 'bottle_*' node (node present, room->bottle RT edge gone) and SEGV walking the
    //    RT tree, and a fast restart hits "agent id 10 already connected". DSRGraph::reset() runs
    //    remove_participant_and_entities() (the clean "Publisher unmatched" path) WITHOUT touching
    //    the Ice communicator, so it does not trip the IceUtil::Mutex teardown abort.
    if (G)
    {
        try { G->reset(); }
        catch (...) { /* best-effort: we are exiting regardless */ }
    }

    // 3) Give the DDS writers a brief window to actually transmit the entity removals + participant
    //    departure to peers before the process vanishes, so no peer is left reading a stale node.
    std::cout.flush();
    std::cerr.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    // 4) Hard-exit, skipping C++ static destruction and the Ice::Application communicator teardown
    //    that races an Ice worker thread against IceUtil::Mutex destruction (ThreadSyscallException
    //    EINVAL) — bottle is the only agent with an active OUTGOING Ice client proxy, so the only one
    //    that hits it. State is persisted, graph presence cleanly removed; the OS reclaims the rest.
    std::_Exit(EXIT_SUCCESS);
}

void SpecificWorker::initialize()
{
    std::print("bottle_concept: initialize()\n");
    GenericWorker::initialize();

    if (not G)
    {
        qWarning() << "bottle_concept: DSR graph not available in initialize()";
        return;
    }

    // Agent-presence protocol wiring.
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
        .on_optional_peer_lost = [this](const std::string& name, std::uint32_t id)
        {
            on_optional_peer_lost(name, id);
        },
        .on_optional_peer_ready = [this](const std::string& name, std::uint32_t id)
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
                for (const auto& label : missing)
                    m += " " + QString::fromStdString(label);
                qInfo() << "[SM] -> Waiting (missing:" << m.trimmed() << ")";
            }
        },
        .on_operating_enter = [this]()
        {
            qInfo("[SM] -> Operating: all required peers present");
            // Stale-node sweep on (re)entering Operating: remove leftover affordance nodes from a
            // previous run so a fresh create doesn't collide and get a DSR-generated name. Keyed on
            // the parent object type, not the node name.
            remove_stale_affordance_nodes();
        },
        .on_operating_loop = [this]()
        {
            compute();
        },
        .on_degraded_enter = [this]()
        {
            if (shutting_down_)
                return;
            // DEBOUNCE — do NOT cleanup/exit on entry. A transient required-peer flap (startup
            // handshake, brief DSR node churn, a peer restarting) fires presenceLost momentarily and
            // then recovers; tearing down here deleted our own node and disconnected the graph, then
            // the agent "recovered" into a broken half-shutdown state and later aborted. Instead wait
            // a grace period and only shut down if a required peer is STILL genuinely missing.
            qInfo("[SM] -> Degraded: required peer lost — %d ms grace before shutdown",
                  REQUIRED_LOSS_GRACE_MS);
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

    // Route SIGTERM/Ctrl+C (sigwatch -> a.quit() -> aboutToQuit) through the crash-free terminal
    // shutdown so the intentional exit never hits the Ice teardown abort either.
    QObject::connect(QCoreApplication::instance(), &QCoreApplication::aboutToQuit,
                     this, &SpecificWorker::terminal_shutdown, Qt::UniqueConnection);

    rt_api_ = G->get_rt_api();
    inner_eigen_ = G->get_inner_eigen_api();
    mask_ingestor_ = std::make_unique<rc::MaskIngestor>(G);
    scene_graph_ = std::make_unique<rc::BottleSceneGraph>(G, rt_api_.get(), inner_eigen_.get(), cfg_,
                                                          [this] { trigger_graph_layout_twopi(); });

    connect(G.get(), &DSR::DSRGraph::del_node_signal, this, &SpecificWorker::del_node_slot);

    // Remove any "bottle*" cylinder nodes left behind by a previous (crashed) run
    // so this agent always starts from a clean slate and never double-scaffolds.
    remove_owned_bottle_nodes();

    const auto rooms = G->get_nodes_by_type("room");
    if (not rooms.empty())
        room_node_id_ = rooms.front().id();
    else
        qWarning() << "bottle_concept: no room node found at startup";

    prior_store_  = std::make_unique<rc::PriorStore>(cfg_.priors_path);
    priors_cache_ = prior_store_->load_priors();

    // Validation harness (no-op unless an Eval.*/Scene.* flag is set). Table-top lookup injected as a
    // callback so the evaluator stays decoupled from the scene-graph layer.
    evaluator_ = std::make_unique<rc::BottleEvaluator>(
        cfg_, webots2robocomp_proxy, inner_eigen_.get(),
        [this](float bx, float by) { return scene_graph_->find_table_top(bx, by); });

    // Hidden-face next-best-view planner (epistemic affordance).
    epistemic_planner_ = rc::EpistemicPlanner(cfg_.epistemic_obs_distance, cfg_.epistemic_view_info);

    // Active-inference fit core (pure belief). Owns the instance map; READS via scene_graph_ but the
    // worker (process_bottle_node) owns the write-back + eval — see the canonical concept-agent loop.
    fitter_ = std::make_unique<rc::BottleFitter>(
        G, inner_eigen_.get(), cfg_, priors_cache_,
        mask_ingestor_.get(), scene_graph_.get());

    // ── Live "Bottle Inference" dashboard (docked in the DSR graph window) ─────────────────────────
    // TimeSeriesPlot is a plain QWidget (no QOpenGL backing store), so docking it is safe. Mirrors
    // table_concept's dashboard; fed each cycle in publish_bottle_diagnostics.
    if (not graph_viewers.empty())
    {
        custom_widget_ = new Custom_widget("Bottle Model — Free Energy, Dimensions (r,h), Posterior σ & Epistemic ΔH");
        graph_viewers.at("")->add_custom_widget_to_dock("Bottle Inference", custom_widget_);

        auto* series_layout = new QVBoxLayout(custom_widget_->frame_series);
        series_layout->setContentsMargins(0, 0, 0, 0);
        custom_widget_->frame_series->setLayout(series_layout);

        const auto add_plot = [&](rc::TimeSeriesPlot*& plot)
        {
            plot = new rc::TimeSeriesPlot(custom_widget_->frame_series);
            plot->set_visible_window(60.f);
            series_layout->addWidget(plot);
        };
        add_plot(ts_fe_plot_);
        add_plot(ts_dim_plot_);
        add_plot(ts_sigma_plot_);
        add_plot(ts_ce_plot_);

        // GenericWorker::initialize() may have started compute() already, so some instances can exist.
        for (auto& [_, inst] : fitter_->instances())
            publish_bottle_diagnostics(inst, inst.prev_free_energy);
    }
}

namespace { constexpr int PLACE_SETTLE_CYCLES = 30; }   // ~settle time after a start-placement move

void SpecificWorker::compute()
{
    if (not G or not rt_api_)
        return;

    if (room_node_id_ == 0)
    {
        const auto rooms = G->get_nodes_by_type("room");
        if (rooms.empty()) return;
        room_node_id_ = rooms.front().id();
    }

    // One-shot: place the bottle on its arm-side spot BEFORE any fit, then let the scene settle so
    // the voxelizer captures it there before scaffold_missing_bottle_nodes() creates the node — a
    // node created from a pre-move camera frame would lock the XY ownership gate at the old pose.
    if (cfg_.place_on_start and not evaluator_->place_done())
    {
        evaluator_->place_bottle_on_start();
        return;
    }
    if (cfg_.place_on_start and place_settle_ < PLACE_SETTLE_CYCLES)
    {
        ++place_settle_;
        return;
    }

    mask_ingestor_->refresh();
    scene_graph_->scaffold_missing_bottle_nodes(priors_cache_, mask_ingestor_->packet(), room_node_id_);

    // Bottle instances are DSR `cylinder` nodes named "bottle_*".
    for (const auto& node : G->get_nodes_by_type("cylinder"))
        if (node.name().starts_with("bottle"))
            process_bottle_node(node);

    // Validation drivers (Webots) also teleport the bottle, so they are mutually exclusive with the
    // arm-side start placement — skip them when place_on_start owns the bottle pose.
    if (not cfg_.place_on_start)
    {
        // Static-restart takes precedence over the continuous sweep.
        if (cfg_.static_pose_test)
            evaluator_->place_static_test_pose();
        else
            evaluator_->step_move_experiment(fitter_->instances());
    }
}

// Canonical per-node orchestration (mirrors table_concept::process_table_node): the fitter runs the
// pure belief (ensure_instance → observe → run_inference, no DSR writes); the worker owns the DSR
// write-back (scene_graph_->step_write_model) and the eval log.
void SpecificWorker::process_bottle_node(const DSR::Node& node)
{
    fitter_->ensure_instance(node, room_node_id_);
    auto& inst = fitter_->instances().at(node.id());
    ++inst.processed_cycles;

    const auto observation = fitter_->observe(inst, node);
    if (not observation.has_fresh_data and inst.matched_frames < 5)
        return;

    const float free_energy = fitter_->run_inference(inst, observation);

    if (auto node_opt = G->get_node(node.id()); node_opt.has_value())
        scene_graph_->step_write_model(inst, node_opt.value(), free_energy);

    // Epistemic capability: publish/refresh the hidden-face affordance for the controller.
    step_epistemic(inst);

    // Live dashboard (after step_epistemic so last_epistemic_gain is current).
    publish_bottle_diagnostics(inst, free_energy);

    // Eval logs every compute cycle, independent of the graph-write change-gate inside step_write_model.
    evaluator_->log_eval(inst, free_energy);

    inst.prev_free_energy = free_energy;
}

// Feed the live dashboard. Series are registered lazily & idempotently here (instances can be created
// via the graph-signal path before the plots exist). Sampled every cycle so a flat trace = a settled
// belief between fresh masks.
void SpecificWorker::publish_bottle_diagnostics(rc::BottleInstance& inst, float free_energy)
{
    if (not ts_fe_plot_)
        return;   // no graph viewer / dashboard this run

    const auto& s = inst.model.state();
    // Posterior std (mm) from the Fisher precision; -1 (drawn as a floor) until a DOF is first observed.
    const auto sigma_mm = [&](int j) -> float {
        return rc::BeliefStabilizer<5>::posterior_std_milli(inst.stab, j);
    };

    ts_fe_plot_->add_series(inst.node_name + "_fe", QColor(255, 170, 0), 1.1f);
    ts_fe_plot_->add_point (inst.node_name + "_fe", free_energy);

    ts_dim_plot_->add_series(inst.node_name + "_r", QColor(255, 90, 90), 1.1f);
    ts_dim_plot_->add_series(inst.node_name + "_h", QColor(90, 200, 90), 1.1f);
    ts_dim_plot_->add_point (inst.node_name + "_r", s.radius);
    ts_dim_plot_->add_point (inst.node_name + "_h", s.height);

    ts_sigma_plot_->add_series(inst.node_name + "_sr", QColor(255, 90, 90), 1.1f);
    ts_sigma_plot_->add_series(inst.node_name + "_sh", QColor(90, 200, 90), 1.1f);
    ts_sigma_plot_->add_point (inst.node_name + "_sr", sigma_mm(3));   // radius (depth-degenerate)
    ts_sigma_plot_->add_point (inst.node_name + "_sh", sigma_mm(4));   // height

    // CUSUM/SPRT counter-evidence Sⱼ for the size DOFs (radius idx 3, height idx 4): signed run of
    // surprise vs the committed belief. ≈0 = coherent/locked; spike-then-decay = rejected glitch;
    // sustained ramp then reset = a real change re-opened the fit. (Diagnostic — see update_fisher_filter.)
    ts_ce_plot_->add_series(inst.node_name + "_ceR", QColor(255, 90, 90), 1.1f);
    ts_ce_plot_->add_series(inst.node_name + "_ceH", QColor(90, 200, 90), 1.1f);
    ts_ce_plot_->add_point (inst.node_name + "_ceR", inst.stab.counter_evidence[3]);
    ts_ce_plot_->add_point (inst.node_name + "_ceH", inst.stab.counter_evidence[4]);
}

// Publish/refresh the "go see the hidden face" affordance (mirrors table_concept::step_epistemic). The
// node persists and re-offers every cycle; a low ΔH is published as-is so the controller's EFE
// selection won't pick a well-seen bottle (belief→knowledge governor without deleting the node).
void SpecificWorker::step_epistemic(rc::BottleInstance& inst)
{
    if (inst.epistemic_cooldown > 0)
        --inst.epistemic_cooldown;

    // Controller-completion hold: when the controller completes (active=false, pending=false), keep the
    // node but suppress its gain for a cooldown so it isn't immediately re-claimed before the belief settles.
    if (const auto aid = inst.affordance.node_id(); aid != 0)
        if (auto an = G->get_node(aid); an.has_value())
        {
            const bool a = G->get_attrib_by_name<active_att>(an.value()).value_or(false);
            const bool p = G->get_attrib_by_name<epistemic_pending_att>(an.value()).value_or(true);
            if (not a and not p and inst.epistemic_cooldown == 0)
            {
                inst.epistemic_cooldown = cfg_.epistemic_cooldown_cycles;
                std::print("[{}] controller completed affordance → hold {} cycles (node kept, gain suppressed)\n",
                           inst.node_name, cfg_.epistemic_cooldown_cycles);
            }
        }

    // ZED origin in the room frame — defines which arc of the bottle is hidden from the camera.
    Eigen::Vector2f camera_xy(std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::quiet_NaN());
    if (inner_eigen_)
        if (const auto c = inner_eigen_->transform("room", Mat::Vector3d(0.0, 0.0, 0.0), "zed", 0);
            c.has_value())
            camera_xy = Eigen::Vector2f(static_cast<float>(c->x()), static_cast<float>(c->y()));

    auto prop = epistemic_planner_.compute(inst.model, camera_xy, inst.stab.fisher_info_raw);
    if (not prop.valid or not prop.is_finite())
        return;   // no camera pose / degenerate ray this cycle → leave the existing affordance untouched

    if (inst.epistemic_cooldown > 0)
        prop.epistemic_gain = 0.0f;

    inst.last_epistemic_gain = prop.epistemic_gain;   // expose to the dashboard

    const auto affordance_node_before = inst.affordance.node_id();
    inst.affordance.update(prop);
    if (affordance_node_before == 0 and inst.affordance.node_id() != 0)
        trigger_graph_layout_twopi();
    inst.epistemic_pending = true;

    log_epistemic_csv(inst, prop, camera_xy);   // gated CSV: ΔH + viewpoint + affordance state (fresh here)
}

// Optional gated CSV of the epistemic/affordance evolution (no-op unless Epistemic.CsvPath is set).
// One row per cycle with a valid proposal: the published ΔH (post-cooldown suppression), the far-side
// target, the camera + bottle positions that define the hidden side, and the affordance protocol state.
void SpecificWorker::log_epistemic_csv(const rc::BottleInstance& inst,
                                       const rc::EpistemicProposal& prop,
                                       const Eigen::Vector2f& camera_xy)
{
    if (cfg_.epistemic_csv_path.empty())
        return;

    if (not epistemic_csv_.is_open())
    {
        epistemic_csv_.open(cfg_.epistemic_csv_path, std::ios::out | std::ios::trunc);
        if (not epistemic_csv_.is_open())
        {
            std::print("bottle_concept: [epistemic] cannot open CSV '{}'\n", cfg_.epistemic_csv_path);
            cfg_.epistemic_csv_path.clear();   // disable further attempts
            return;
        }
        epistemic_csv_ << "cycle,node,gain,pending,cooldown,aff_state,aff_node,"
                          "target_x,target_y,target_yaw,cam_x,cam_y,bottle_cx,bottle_cy,radius,sigma_r_mm\n";
    }

    const auto& s = inst.model.state();
    const float sigma_r = rc::BeliefStabilizer<5>::posterior_std_milli(inst.stab, 3);   // radius (mm)
    epistemic_csv_ << inst.processed_cycles << ',' << inst.node_name << ','
                   << prop.epistemic_gain << ',' << (inst.epistemic_pending ? 1 : 0) << ','
                   << inst.epistemic_cooldown << ','
                   << rc::BottleAffordance::state_name(inst.affordance.state()) << ','
                   << inst.affordance.node_id() << ','
                   << prop.epistemic_target_x_m << ',' << prop.epistemic_target_y_m << ','
                   << prop.epistemic_target_yaw_rad << ','
                   << camera_xy.x() << ',' << camera_xy.y() << ','
                   << s.cx << ',' << s.cy << ',' << s.radius << ',' << sigma_r << '\n';
    epistemic_csv_.flush();   // flush each row so a plot can tail the file during a live run
}

void SpecificWorker::del_node_slot(std::uint64_t id)
{
    if (fitter_)
    {
        // If the deleted node was an instance's affordance node (controller satisfied / external delete),
        // reset its state machine so it re-creates on the next epistemic cycle.
        for (auto& [_, inst] : fitter_->instances())
            inst.affordance.on_node_deleted(id);
        fitter_->forget_node(id);
    }
    // A node left the graph — re-run the twopi layout so the view stays coherent.
    trigger_graph_layout_twopi();
}

void SpecificWorker::modify_node_attrs_slot(std::uint64_t id, const std::vector<std::string>&)
{
    // Track controller-owned protocol transitions on each instance's affordance node (active/pending).
    if (fitter_)
        for (auto& [_, inst] : fitter_->instances())
            inst.affordance.on_node_modified(id);
}

void SpecificWorker::emergency()
{
    std::print("bottle_concept: emergency()\n");
}

void SpecificWorker::restore()
{
    std::print("bottle_concept: restore()\n");
}

int SpecificWorker::startup_check()
{
    std::print("bottle_concept: startup_check()\n");
    QTimer::singleShot(200, QCoreApplication::instance(), SLOT(quit()));
    return 0;
}
