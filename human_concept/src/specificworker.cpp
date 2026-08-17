/*
 *    Copyright (C) 2026 by RoboComp CORTEX Team — GNU GPL v3.
 */

/**
 * SpecificWorker — human_concept agent: RoboComp lifecycle + presence protocol + orchestration only.
 * compute() polls the SkeletonSource, scaffolds person nodes for new tracks, and runs the canonical
 * per-person pipeline: fitter_ (observe → Laplace fit) → scene_graph_ (persist pose + uncertainty) →
 * step_epistemic (reduce-occlusion affordance) → dashboard.
 */

#include "specificworker.h"

#include "../../common/diag_log/rotating_csv.h"   // keep the previous run instead of wiping it
#include "../../common/nbv/graph_obstacles.h"   // rc::nbv::sensor_from_graph / collect_graph_obstacles

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <locale>
#include <print>
#include <thread>

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

    cfg_ = rc::load_human_config(configLoader);

#ifdef HIBERNATION_ENABLED
    hibernationChecker.start(500);
#endif

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

    states["Compute"]->addTransition(states["Compute"].get(), SIGNAL(entered()), states["Waiting"].get());
    states["Waiting"]->addTransition(this, SIGNAL(presenceReady()), states["Operating"].get());
    states["Operating"]->addTransition(this, SIGNAL(presenceLost()), states["Degraded"].get());
    states["Degraded"]->addTransition(states["Degraded"].get(), SIGNAL(entered()), states["Waiting"].get());

    statemachine.addState(states["Waiting"].get());
    statemachine.addState(states["Operating"].get());
    statemachine.addState(states["Degraded"].get());

    statemachine.setChildMode(QState::ExclusiveStates);
    statemachine.start();

    if (auto error = statemachine.errorString(); error.length() > 0)
    {
        qWarning() << error;
        throw error;
    }
}

SpecificWorker::~SpecificWorker()
{
    request_shutdown();
    std::print("human_concept: SpecificWorker destroyed.\n");
}

void SpecificWorker::request_shutdown()
{
    if (shutting_down_.exchange(true))
        return;
    save_window_settings();
    if (G)
        disconnect(G.get(), nullptr, this, nullptr);
    inner_eigen_.reset();
    cleanup_owned_nodes();
}

void SpecificWorker::terminal_shutdown()
{
    static std::atomic<bool> terminating{false};
    if (terminating.exchange(true))
        return;
    request_shutdown();
    if (G)
    {
        try { G->reset(); }
        catch (...) { /* best-effort: exiting regardless */ }
    }
    std::cout.flush();
    std::cerr.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    std::_Exit(EXIT_SUCCESS);
}

void SpecificWorker::initialize()
{
    std::print("human_concept: initialize()\n");
    GenericWorker::initialize();

    if (not G)
    {
        qWarning() << "human_concept: DSR graph not available in initialize()";
        return;
    }

    presence_coordinator_.configure(configLoader, G, static_cast<std::uint32_t>(agent_id));
    // Colour this agent's node in the graph view by its live health: the coordinator already
    // publishes the presence lifecycle; this adds the external FSM axis (Initialize/Compute/
    // Emergency/Restore). Generic discovery via objectName(), so genericworker regeneration
    // cannot break it.
    presence_coordinator_.attach_state_machine(&statemachine);
    presence_coordinator_.set_transition_hooks({
        .request_presence_ready = [this]() { emit presenceReady(); },
        .request_presence_lost  = [this]() { emit presenceLost(); },
    });
    presence_coordinator_.set_peer_hooks({
        .on_peer_restarted = [](std::uint32_t id) { qInfo() << "[Presence] peer" << id << "restarted"; },
        .on_optional_peer_lost  = [this](const std::string& n, std::uint32_t id) { on_optional_peer_lost(n, id); },
        .on_optional_peer_ready = [this](const std::string& n, std::uint32_t id) { on_optional_peer_ready(n, id); },
    });
    presence_coordinator_.set_lifecycle_hooks({
        .on_waiting_enter = [this]()
        {
            const auto missing = presence_coordinator_.missing_required_names();
            if (missing.empty()) qInfo("[SM] -> Waiting");
            else
            {
                QString m;
                for (const auto& label : missing) m += " " + QString::fromStdString(label);
                qInfo() << "[SM] -> Waiting (missing:" << m.trimmed() << ")";
            }
        },
        .on_operating_enter = [this]()
        {
            qInfo("[SM] -> Operating: all required peers present");
            remove_stale_affordance_nodes();
        },
        .on_operating_loop = [this]()
        {
            compute();
            // Feed the compute-loop FPS to the DSR main-window status bar (lower bar), like the other agents.
            if (auto it = graph_viewers.find(""); it != graph_viewers.end() and it->second)
                it->second->set_external_fps(states.at("Operating")->getActualFps());
        },
        .on_degraded_enter = [this]()
        {
            if (shutting_down_) return;
            qInfo("[SM] -> Degraded: required peer lost — %d ms grace before shutdown", REQUIRED_LOSS_GRACE_MS);
            QTimer::singleShot(REQUIRED_LOSS_GRACE_MS, this, [this]()
            {
                if (shutting_down_) return;
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

    rt_api_      = G->get_rt_api();
    inner_eigen_ = G->get_inner_eigen_api();
    scene_graph_ = std::make_unique<rc::HumanSceneGraph>(G, rt_api_.get(), inner_eigen_.get(), cfg_,
                                                         [this] { trigger_graph_layout_twopi(); });

    connect(G.get(), &DSR::DSRGraph::del_node_signal, this, &SpecificWorker::del_node_slot);

    remove_owned_person_nodes();

    if (const auto rooms = G->get_nodes_by_type("room"); not rooms.empty())
        room_node_id_ = rooms.front().id();
    else
        qWarning() << "human_concept: no room node found at startup";

    prior_store_  = std::make_unique<rc::PriorStore>(cfg_.priors_path);
    priors_cache_ = prior_store_->load_priors();

    skeleton_source_ = rc::make_skeleton_source(cfg_.source_kind, cfg_.replay_path, cfg_.replay_loop,
                                                G, inner_eigen_.get());
    epistemic_planner_ = rc::EpistemicPlanner(cfg_.epistemic_obs_distance, cfg_.epistemic_view_info);
    // ONE detector envelope: the far-side viewpoint is the argmax of the same model absence is weighted by,
    // and the published gain is multiplied by P(detect) there — so an orbit the detector could not fire from
    // stops bidding for the walk AROUND a person.
    epistemic_planner_.set_detector_envelope(rc::detect::DetectorEnvelope{});
    epistemic_planner_.set_robot_radius(0.30f);   // Shadow's footprint radius
    // ★The camera model is read PER CYCLE at the compute site (rc::nbv::sensor_from_graph),
    // NOT once here: the zed intrinsics are published by robot_concept when frames start
    // arriving, so reading them in initialize() races the producer. Losing that race leaves
    // vfov = 0, which silently collapses the fill model to horizontal-only — the exact bug
    // rc::nbv exists to fix, and it drives the robot nose-to-nose with tall objects.

    // ── Live "Human Inference" dashboard ─────────────────────────────────────────
    if (not graph_viewers.empty())
    {
        custom_widget_ = new Custom_widget("Human Model — Free Energy, tr(cov) & Posterior σ");
        graph_viewers.at("")->add_custom_widget_to_dock("Human Inference", custom_widget_);

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
        add_plot(ts_unc_plot_);
        add_plot(ts_sigma_plot_);
    }
}

void SpecificWorker::compute()
{
    if (not G or not rt_api_ or not skeleton_source_ or not fitter_)
        return;

    if (room_node_id_ == 0)
    {
        const auto rooms = G->get_nodes_by_type("room");
        if (rooms.empty()) return;
        room_node_id_ = rooms.front().id();
    }

    // Poll this cycle's bodies once; hand them to the fitter and scaffold any new tracks.
    auto bodies = skeleton_source_->poll();
    scene_graph_->scaffold_missing_person_nodes(bodies, room_node_id_);
    fitter_->set_frame(std::move(bodies));

    for (const auto& node : G->get_nodes_by_type("person"))
        if (node.name().starts_with("person"))
            process_person_node(node);

    prune_absent_persons();

    fps_counter_.print("[Compute]", 3000);
}

void SpecificWorker::prune_absent_persons()
{
    // Persistence/death: a short occlusion keeps the (frozen) model alive — the controller holds the
    // last pose. Only once a person has gone unseen for DeathFrames cycles (enough evidence of absence)
    // do we remove the node + its affordance and forget the instance.
    std::vector<std::uint64_t> dead;
    for (auto& [id, inst] : fitter_->instances())
        if (inst.frames_since_detection > cfg_.death_frames)
            dead.push_back(id);
    for (const auto id : dead)
    {
        fitter_->instances().at(id).affordance.remove();
        G->delete_node(id);
        fitter_->forget_node(id);
        std::print("human_concept: person id={} removed (unseen > {} cycles)\n", id, cfg_.death_frames);
    }
}

void SpecificWorker::process_person_node(const DSR::Node& node)
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

    step_epistemic(inst);
    publish_human_diagnostics(inst, free_energy);
    log_fit_csv(inst, free_energy);

    inst.prev_free_energy = free_energy;
}

void SpecificWorker::publish_human_diagnostics(rc::HumanInstance& inst, float free_energy)
{
    if (not ts_fe_plot_ or not inst.has_result)
        return;

    ts_fe_plot_->add_series(inst.node_name + "_fe", QColor(255, 170, 0), 1.1f);
    ts_fe_plot_->add_point (inst.node_name + "_fe", free_energy);

    ts_unc_plot_->add_series(inst.node_name + "_tr", QColor(90, 160, 255), 1.1f);
    ts_unc_plot_->add_series(inst.node_name + "_nv", QColor(160, 160, 160), 0.8f);
    ts_unc_plot_->add_point (inst.node_name + "_tr", inst.last_result.uncertainty_trace);
    ts_unc_plot_->add_point (inst.node_name + "_nv", static_cast<float>(inst.last_result.valid_count));

    // Plot 3 — per-joint positional posterior σ (mm): the angle covariance propagated through the FK
    // Jacobian to each BODY_18 keypoint (pos_std_milli). Uniform units + naming across joints so they
    // are directly comparable. (Elbows AND wrists are keypoints, not DOFs — this replaces the old
    // elbow-flex DOF σ, which was in mrad and read -1 until the stabiliser matured.) -1 until observed.
    namespace KP = rc::human::KP;
    const auto kp_sigma = [&](int kp) -> float { return inst.last_result.pos_std_milli[kp]; };
    const struct { const char* name; int kp; QColor color; } joints[] = {
        {"_l_elbow", KP::L_ELBOW, QColor(255, 90, 90)},
        {"_r_elbow", KP::R_ELBOW, QColor(90, 200, 90)},
        {"_l_wrist", KP::L_WRIST, QColor(210, 90, 220)},
        {"_r_wrist", KP::R_WRIST, QColor(90, 200, 220)},
    };
    for (const auto& j : joints)
    {
        ts_sigma_plot_->add_series(inst.node_name + j.name, j.color, 1.1f);
        ts_sigma_plot_->add_point (inst.node_name + j.name, kp_sigma(j.kp));
    }
}

void SpecificWorker::step_epistemic(rc::HumanInstance& inst)
{
    if (not inst.has_result)
        return;

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
                inst.epistemic_cooldown = cfg_.epistemic_cooldown_cycles;
        }

    // Person root (pelvis) in the room frame.
    const Eigen::Vector3f pelvis = rc::HumanSceneGraph::pelvis_of(inst.last_result.kp_pred_aligned);
    if (not pelvis.allFinite())
        return;
    const Eigen::Vector2f person_xy(pelvis.x(), pelvis.y());

    // ZED origin in the room frame.
    Eigen::Vector2f camera_xy(std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::quiet_NaN());
    if (inner_eigen_)
        if (const auto c = inner_eigen_->transform("room", Mat::Vector3d(0.0, 0.0, 0.0), "zed", 0);
            c.has_value())
            camera_xy = Eigen::Vector2f(static_cast<float>(c->x()), static_cast<float>(c->y()));

    // Worst-constrained DOF posterior precision drives ΔH.
    float worst_info = std::numeric_limits<float>::max();
    for (int j = 0; j < 11; ++j)
        worst_info = std::min(worst_info, inst.stab.fisher_info_raw[j]);
    if (not std::isfinite(worst_info))
        worst_info = 1e-3f;

    auto prop = epistemic_planner_.compute(person_xy, camera_xy, worst_info,
                                           rc::nbv::sensor_from_graph(*G, inner_eigen_.get()));
    if (not prop.valid or not prop.is_finite())
        return;

    if (inst.epistemic_cooldown > 0)
        prop.epistemic_gain = 0.0f;

    inst.last_epistemic_gain = prop.epistemic_gain;

    const auto affordance_node_before = inst.affordance.node_id();
    // Planner internals stay in EpistemicProposal; the producer takes the shared eleven-field view.
    rc::AffordanceTarget tgt;
    tgt.x_m     = prop.epistemic_target_x_m;
    tgt.y_m     = prop.epistemic_target_y_m;
    tgt.yaw_rad = prop.epistemic_target_yaw_rad;
    tgt.gain    = prop.epistemic_gain;
    tgt.valid   = prop.valid;
    inst.affordance.update(tgt);
    if (affordance_node_before == 0 and inst.affordance.node_id() != 0)
        trigger_graph_layout_twopi();
    inst.epistemic_pending = true;

    log_epistemic_csv(inst, prop, camera_xy);
}

void SpecificWorker::log_epistemic_csv(const rc::HumanInstance& inst, const rc::EpistemicProposal& prop,
                                       const Eigen::Vector2f& camera_xy)
{
    if (cfg_.epistemic_csv_path.empty())
        return;
    if (not epistemic_csv_.is_open())
    {
        rc::diag::open_rotating(epistemic_csv_, cfg_.epistemic_csv_path);
        if (not epistemic_csv_.is_open())
        {
            std::print("human_concept: [epistemic] cannot open CSV '{}'\n", cfg_.epistemic_csv_path);
            cfg_.epistemic_csv_path.clear();
            return;
        }
        // Decimal POINT always, whatever locale the process booted into — the readers (and any
        // future re-parse of our own log) assume it. Cheap insurance; see cpp/core/csv_parse.h.
        epistemic_csv_.imbue(std::locale::classic());
        epistemic_csv_ << "cycle,node,gain,pending,cooldown,aff_state,aff_node,"
                          "target_x,target_y,target_yaw,cam_x,cam_y,valid,tr_cov\n";
    }
    epistemic_csv_ << inst.processed_cycles << ',' << inst.node_name << ','
                   << prop.epistemic_gain << ',' << (inst.epistemic_pending ? 1 : 0) << ','
                   << inst.epistemic_cooldown << ','
                   << rc::ObjectAffordance::state_name(inst.affordance.state()) << ','
                   << inst.affordance.node_id() << ','
                   << prop.epistemic_target_x_m << ',' << prop.epistemic_target_y_m << ','
                   << prop.epistemic_target_yaw_rad << ','
                   << camera_xy.x() << ',' << camera_xy.y() << ','
                   << inst.last_result.valid_count << ',' << inst.last_result.uncertainty_trace << '\n';
    epistemic_csv_.flush();
}

void SpecificWorker::log_fit_csv(const rc::HumanInstance& inst, float free_energy)
{
    if (cfg_.fit_csv_path.empty() or not inst.has_result)
        return;
    const auto& r = inst.last_result;
    namespace KP = rc::human::KP;
    if (not fit_csv_.is_open())
    {
        rc::diag::open_rotating(fit_csv_, cfg_.fit_csv_path);
        if (not fit_csv_.is_open())
        {
            std::print("human_concept: [fit] cannot open CSV '{}'\n", cfg_.fit_csv_path);
            cfg_.fit_csv_path.clear();
            return;
        }
        fit_csv_.imbue(std::locale::classic());   // decimal POINT always (see above)
        // mu0..mu10 = the 11 joint-angle DOFs (the belief TARGET θ*); sig_* = positional posterior σ
        // (mm); track_err = mean |θ* − θ_cmd| (controller lag); vel_sat/acc_sat = #DOFs the controller
        // held at its speed/accel limit this step.
        fit_csv_ << "cycle,node,dt,fe,valid,tr_cov,"
                    "mu0,mu1,mu2,mu3,mu4,mu5,mu6,mu7,mu8,mu9,mu10,"
                    "sig_l_elbow,sig_r_elbow,sig_l_wrist,sig_r_wrist,track_err,vel_sat,acc_sat,rej\n";
    }
    fit_csv_ << inst.processed_cycles << ',' << inst.node_name << ','
             << r.dt << ',' << free_energy << ',' << r.valid_count << ',' << r.uncertainty_trace;
    for (int k = 0; k < 11; ++k)
        fit_csv_ << ',' << r.mu(k);
    fit_csv_ << ',' << r.pos_std_milli[KP::L_ELBOW] << ',' << r.pos_std_milli[KP::R_ELBOW]
             << ',' << r.pos_std_milli[KP::L_WRIST] << ',' << r.pos_std_milli[KP::R_WRIST]
             << ',' << inst.track_err << ',' << r.vel_clamped << ',' << r.acc_clamped
             << ',' << (r.rejected ? 1 : 0) << '\n';
    fit_csv_.flush();
}

void SpecificWorker::del_node_slot(std::uint64_t id)
{
    if (fitter_)
    {
        for (auto& [_, inst] : fitter_->instances())
            inst.affordance.on_node_deleted(id);
        fitter_->forget_node(id);
    }
    trigger_graph_layout_twopi();
}

void SpecificWorker::modify_node_attrs_slot(std::uint64_t id, const std::vector<std::string>&)
{
    if (fitter_)
        for (auto& [_, inst] : fitter_->instances())
            inst.affordance.on_node_modified(id);
}

void SpecificWorker::emergency() { std::print("human_concept: emergency()\n"); }
void SpecificWorker::restore()   { std::print("human_concept: restore()\n"); }

int SpecificWorker::startup_check()
{
    std::print("human_concept: startup_check()\n");
    QTimer::singleShot(200, QCoreApplication::instance(), SLOT(quit()));
    return 0;
}
