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
 * SpecificWorker — bottle_concept agent.
 *
 *  ① Read the YOLO masks node from DSR ("masks", written by the voxelizer)
 *  ② Select the slice labelled "bottle", split its support points into
 *     near-surface candidates (queue anchors) and residuals (drive expansion)
 *  ③ Update the historical sample queue + run gradient descent on the cylinder
 *     generative model (SDF + FE)
 *  ④ Write the fitted pose AND its Laplace covariance (P_bottle) on the
 *     room→bottle RT edge, plus geometry attrs + a cylinder mesh
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

// ─── Constructor / Destructor ─────────────────────────────────────────────────

SpecificWorker::SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check)
    : GenericWorker(configLoader, tprx)
{
    this->startup_check_flag = startup_check;
    if (this->startup_check_flag)
    {
        this->startup_check();
        return;
    }

    load_config(configLoader);

#ifdef HIBERNATION_ENABLED
    hibernationChecker.start(500);
#endif

    // ── Agent-presence state machine (Waiting → Operating → Degraded) ──────────
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
    std::print("bottle_concept: SpecificWorker destroyed, checkpoints saved.\n");
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

    if (prior_store_)
    {
        for (const auto& [id, inst] : instances_)
        {
            const auto& s = inst.model.state();
            prior_store_->save_checkpoint(BottleCheckpoint{
                inst.node_name, s.radius, s.height, s.cx, s.cy, s.cz,
                inst.prev_free_energy, inst.frames_converged >= cfg_.K_stable});
        }
    }

    cleanup_owned_nodes();
}

void SpecificWorker::terminal_shutdown()
{
    static std::atomic<bool> terminating{false};
    if (terminating.exchange(true))
        return;   // _Exit is coming; never run this twice

    // 1) Save checkpoints, sever our graph callbacks, delete our owned DSR nodes (publishes
    //    del-deltas) and notify peers. Idempotent (shutting_down_ guard).
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

// ─── Initialisation ──────────────────────────────────────────────────────────

void SpecificWorker::initialize()
{
    std::print("bottle_concept: initialize()\n");
    GenericWorker::initialize();

    if (not G)
    {
        qWarning() << "bottle_concept: DSR graph not available in initialize()";
        return;
    }

    // ── Agent-presence protocol wiring ─────────────────────────────────────────
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

    connect(G.get(), &DSR::DSRGraph::del_node_signal, this, &SpecificWorker::del_node_slot);

    // Remove any "bottle*" cylinder nodes left behind by a previous (crashed) run
    // so this agent always starts from a clean slate and never double-scaffolds.
    remove_owned_bottle_nodes();

    const auto rooms = G->get_nodes_by_type("room");
    if (not rooms.empty())
        room_node_id_ = rooms.front().id();
    else
        qWarning() << "bottle_concept: no room node found at startup";

    prior_store_  = std::make_unique<PriorStore>(priors_path_, checkpoint_path_);
    priors_cache_ = prior_store_->load_priors();
}

// ─── Main compute loop ───────────────────────────────────────────────────────

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

    refresh_masks_packet();
    scaffold_missing_bottle_nodes();

    // Bottle instances are DSR `cylinder` nodes named "bottle_*".
    for (const auto& node : G->get_nodes_by_type("cylinder"))
        if (node.name().starts_with("bottle"))
            process_bottle_node(node);

    // Validation drivers (Webots). Static-restart takes precedence over the continuous sweep.
    if (cfg_.static_pose_test)
        place_static_test_pose();
    else
        step_move_experiment();
}

// ─── Per-bottle pipeline ───────────────────────────────────────────────────────

void SpecificWorker::process_bottle_node(const DSR::Node& node)
{
    ensure_instance(node);

    auto& inst = instances_.at(node.id());
    ++inst.processed_cycles;
    const auto observation = observe_bottle_node(inst, node);

    if (not observation.has_fresh_data and inst.matched_frames < 5)
        return;

    const float free_energy = run_bottle_inference(inst, observation);

    if (auto node_opt = G->get_node(node.id()); node_opt.has_value())
        step_write_model(inst, node_opt.value(), free_energy);

    // Eval logs every compute cycle, independent of the graph-write change-gate above.
    log_eval(inst, free_energy);

    inst.prev_free_energy = free_energy;
}

SpecificWorker::BottleObservation SpecificWorker::observe_bottle_node(BottleInstance& inst,
                                                                      const DSR::Node& node)
{
    BottleObservation observation;

    // Primary path: YOLO masks (room frame). Classify-don't-destroy SDF split —
    // inliers become queue anchors, the rest drive model expansion.
    if (masks_packet_.valid and masks_packet_.frame_id > inst.last_masks_frame_seen)
    {
        const auto selected_mask = select_mask_for_bottle(inst);
        if (selected_mask.has_value())
        {
            const auto& slice = selected_mask.value();
            const std::size_t begin = std::min(slice.support_begin, masks_packet_.support_points.size());
            const std::size_t end   = std::min(slice.support_end,   masks_packet_.support_points.size());

            std::vector<Eigen::Vector3f> candidate_pts;
            std::vector<Eigen::Vector3f> residual_pts;
            candidate_pts.reserve(end > begin ? end - begin : 0);
            residual_pts.reserve(end > begin ? end - begin : 0);

            for (std::size_t i = begin; i < end; ++i)
            {
                const auto& p = masks_packet_.support_points[i];
                const float sdf = inst.model.sdf_point(p);
                if (std::abs(sdf) < cfg_.sdf_threshold_for_storage)
                    candidate_pts.push_back(p);
                else
                    residual_pts.push_back(p);
            }

            observation.has_fresh_data = true;
            observation.candidate_pts = std::move(candidate_pts);
            observation.residual_pts  = std::move(residual_pts);

            if (not observation.candidate_pts.empty() or not observation.residual_pts.empty())
            {
                const float total = static_cast<float>(observation.candidate_pts.size() + observation.residual_pts.size());
                observation.explanation_ratio = total > 0.0f
                    ? static_cast<float>(observation.candidate_pts.size()) / total
                    : 0.0f;

                inst.last_masks_frame_seen = masks_packet_.frame_id;

                if (should_log(inst))
                {
                    // DIAG: per-axis extent of the received cloud — tells us whether the 6× radius
                    // blob is a depth slab (one axis) or a too-wide mask (lateral/both axes).
                    Eigen::Vector3f lo = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
                    Eigen::Vector3f hi = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());
                    for (std::size_t i = begin; i < end; ++i)
                    {
                        lo = lo.cwiseMin(masks_packet_.support_points[i]);
                        hi = hi.cwiseMax(masks_packet_.support_points[i]);
                    }
                    const Eigen::Vector3f ext = hi - lo;
                    std::print("[{}] masks={} label='{}' conf={:.2f} support={} cand={} resid={} centroid=({:.2f},{:.2f},{:.2f}) ext=({:.3f},{:.3f},{:.3f})\n",
                               inst.node_name, masks_packet_.frame_id, slice.label, slice.confidence,
                               end - begin, observation.candidate_pts.size(), observation.residual_pts.size(),
                               slice.centroid.x(), slice.centroid.y(), slice.centroid.z(),
                               ext.x(), ext.y(), ext.z());
                }
                return observation;
            }
        }
    }

    // Fallback: candidate/residual point attributes written directly on the node.
    observation.candidate_pts = read_pts_attrib(node, "candidate_pts_att");
    observation.residual_pts  = read_pts_attrib(node, "residual_pts_att");
    observation.has_fresh_data = not observation.candidate_pts.empty() or not observation.residual_pts.empty();
    return observation;
}

float SpecificWorker::run_bottle_inference(BottleInstance& inst, const BottleObservation& observation)
{
    inst.queue.begin_cycle();

    if (observation.has_fresh_data)
        ingest_observation_voxels(inst, observation);

    if (observation.has_fresh_data and not inst.voxel_bank_pts.empty())
    {
        // Cold-start: snap model & prior to the first observation centroid so
        // gradient descent begins at the right place rather than the prior.
        if (inst.matched_frames == 0)
        {
            Eigen::Vector3f sum = Eigen::Vector3f::Zero();
            const auto& src = observation.candidate_pts.empty() ? inst.voxel_bank_pts
                                                                : observation.candidate_pts;
            for (const auto& p : src) sum += p;
            const Eigen::Vector3f cen = sum / static_cast<float>(src.size());
            auto s = inst.model.state();
            s.cx = cen.x(); s.cy = cen.y(); s.cz = cen.z();
            inst.model.set_state(s);
            inst.model.set_prior(s);   // zero KL so the data term dominates from the start
            std::print("[{}] cold-start snap → ({:.2f},{:.2f},{:.2f}) ({} pts)\n",
                       inst.node_name, s.cx, s.cy, s.cz, src.size());
            inst.matched_frames = cfg_.min_frames_before_historical + cfg_.historical_warmup_frames + 1;
        }
        else
            ++inst.matched_frames;

        step_queue_update(inst, inst.voxel_bank_pts, std::clamp(observation.explanation_ratio, 0.0f, 1.0f));
    }

    const float explanation_confidence = std::clamp(observation.explanation_ratio, 0.0f, 1.0f);
    float residual_precision = 0.0f;
    if (observation.has_fresh_data and observation.candidate_pts.empty() and not observation.residual_pts.empty())
        residual_precision = 0.10f;   // only residuals: keep a floor so the model can expand
    else
        residual_precision = std::clamp(1.0f - explanation_confidence, 0.0f, 1.0f);

    const float free_energy = step_model_update(inst, observation.residual_pts, residual_precision);

    const auto& s = inst.model.state();
    if (should_log(inst))
        std::print("[{}] FE={:.4f}  c=({:.3f},{:.3f},{:.3f}) r={:.3f} h={:.3f}  pts={}\n",
                   inst.node_name, free_energy, s.cx, s.cy, s.cz, s.radius, s.height,
                   inst.queue.size() + static_cast<int>(observation.residual_pts.size()));

    return free_energy;
}

void SpecificWorker::step_queue_update(BottleInstance& inst,
                                       const std::vector<Eigen::Vector3f>& candidate_pts,
                                       float observation_precision)
{
    const auto sdf_vals = inst.model.compute_sdf(candidate_pts);
    const float precision = std::max(0.05f, observation_precision);
    Eigen::Matrix2f robot_cov = read_robot_covariance();
    // Low explanatory adequacy ⇒ low sensory precision ⇒ inflate capture covariance.
    robot_cov /= precision;
    const int q_before = inst.queue.size();
    inst.queue.insert(candidate_pts, sdf_vals, robot_cov, inst.model, inst.matched_frames);
    const int admitted = inst.queue.size() - q_before;
    // New points from a fresh view → unlock the optimizer so it can re-converge.
    if (admitted > 0 and inst.frames_converged >= cfg_.K_stable)
        inst.frames_converged = cfg_.K_stable / 2;
}

float SpecificWorker::step_model_update(BottleInstance& inst,
                                        const std::vector<Eigen::Vector3f>& residual_pts,
                                        float residual_precision)
{
    const BottleState previous_state = inst.model.state();

    // Compose evidence: historical queue anchors + this frame's residuals.
    std::vector<Eigen::Vector3f> fit_pts = inst.queue.points();
    std::vector<float>           fit_weights = inst.queue.weights();
    const std::size_t historical_anchor_count = fit_pts.size();

    for (const auto& r : residual_pts)
        if (residual_precision > 1e-3f)
        {
            fit_pts.push_back(r);
            fit_weights.push_back(residual_precision);
        }

    if (fit_pts.empty())
    {
        inst.last_queue_metrics = inst.queue.metrics();
        return inst.model.compute_free_energy({}, {});
    }

    float free_energy = previous_state.cx;  // placeholder, overwritten below
    if (inst.frames_converged < cfg_.K_stable)
    {
        // Refresh queue scores against the moving model during descent.
        auto observer = [&](int, const BottleState& state, const FreeEnergyDecomposition&)
        {
            BottleModel shadow = inst.model;
            shadow.set_state(state);
            inst.queue.refresh_scores(shadow);
        };
        free_energy = inst.model.gradient_step(fit_pts, fit_weights, historical_anchor_count, observer);
    }
    else
    {
        free_energy = inst.model.compute_free_energy(fit_pts, fit_weights);
        inst.queue.refresh_scores(inst.model);
    }

    // Reject a non-finite optimum (NaN SDF gradient) — revert to the last good state.
    const auto& s = inst.model.state();
    const bool finite = std::isfinite(s.cx) and std::isfinite(s.cy) and std::isfinite(s.cz) and
                        std::isfinite(s.radius) and std::isfinite(s.height) and std::isfinite(free_energy);
    if (not finite)
    {
        inst.model.set_state(previous_state);
        inst.model.set_prior(previous_state);
        free_energy = inst.model.compute_free_energy(fit_pts, fit_weights);
    }

    // Convergence bookkeeping.
    if (std::abs(free_energy - inst.prev_free_energy) < cfg_.fe_eps)
        inst.frames_converged = std::min(inst.frames_converged + 1, cfg_.K_stable);
    else
        inst.frames_converged = 0;

    inst.last_fe_terms = inst.model.compute_free_energy_decomposition(fit_pts, fit_weights, historical_anchor_count);
    inst.last_queue_metrics = inst.queue.metrics();
    return free_energy;
}

// ─── Voxel bank (bottle-owned historical memory) ───────────────────────────────

void SpecificWorker::ingest_observation_voxels(BottleInstance& inst, const BottleObservation& observation)
{
    std::size_t inserted = 0;
    std::size_t rejected_foreign = 0;
    const auto max_points = static_cast<std::size_t>(std::max(1, cfg_.voxel_bank_max_points));

    auto ingest = [&](const std::vector<Eigen::Vector3f>& src)
    {
        for (const auto& p : src)
        {
            if (inst.voxel_bank_pts.size() >= max_points)
                break;
            if (not is_voxel_owned_by_bottle(inst, p))
            {
                ++rejected_foreign;
                continue;
            }
            const auto key = voxel_key(p, cfg_.voxel_bank_quantization_m);
            if (inst.voxel_bank_keys.insert(key).second)
            {
                inst.voxel_bank_pts.push_back(p);
                ++inserted;
            }
        }
    };

    ingest(observation.candidate_pts);
    ingest(observation.residual_pts);

    if (inserted > 0 and should_log(inst))
        std::print("[{}] voxel-bank: +{} total={} (cap={}) reject_foreign={}\n",
                   inst.node_name, inserted, inst.voxel_bank_pts.size(), max_points, rejected_foreign);
}

bool SpecificWorker::is_voxel_owned_by_bottle(const BottleInstance& inst, const Eigen::Vector3f& point) const
{
    const auto& s = inst.model.state();

    // XY ownership gate: bottle-centred radius + margin.
    const float gate_radius = s.radius + cfg_.voxel_select_radius_margin_m;
    const float dx = point.x() - s.cx;
    const float dy = point.y() - s.cy;
    if (std::hypot(dx, dy) > gate_radius)
        return false;

    // Height gate around the cylinder span.
    const float z_min = s.cz - s.height * 0.5f - cfg_.voxel_select_height_margin_m;
    const float z_max = s.cz + s.height * 0.5f + cfg_.voxel_select_height_margin_m;
    return point.z() >= z_min and point.z() <= z_max;
}

// ─── Masks reading (verbatim from table_concept) ───────────────────────────────

bool SpecificWorker::refresh_masks_packet()
{
    const auto masks_node_opt = G->get_node("masks");
    if (not masks_node_opt.has_value())
    {
        masks_packet_ = {};
        return false;
    }

    const auto& masks_node = masks_node_opt.value();
    const auto& attrs = masks_node.attrs();

    const auto find_attr = [&](const std::string& key) -> const DSR::Attribute*
    {
        const auto it = attrs.find(key);
        return (it != attrs.end()) ? &it->second : nullptr;
    };

    const DSR::Attribute* frame_attr     = find_attr("mask_frame_id");
    const DSR::Attribute* count_attr     = find_attr("mask_count");
    const DSR::Attribute* labels_attr    = find_attr("mask_labels");
    const DSR::Attribute* label_ids_attr = find_attr("mask_label_ids");
    const DSR::Attribute* confs_attr     = find_attr("mask_confidences");
    const DSR::Attribute* offsets_attr   = find_attr("mask_support_offsets");
    const DSR::Attribute* points_attr    = find_attr("mask_support_points");
    const DSR::Attribute* centroids_attr = find_attr("mask_centroids_xyz");
    const DSR::Attribute* bbox_min_attr  = find_attr("mask_bbox_min_xyz");
    const DSR::Attribute* bbox_max_attr  = find_attr("mask_bbox_max_xyz");

    if (frame_attr == nullptr or count_attr == nullptr or labels_attr == nullptr or
        label_ids_attr == nullptr or confs_attr == nullptr or offsets_attr == nullptr or
        points_attr == nullptr or centroids_attr == nullptr or bbox_min_attr == nullptr or
        bbox_max_attr == nullptr)
    {
        masks_packet_ = {};
        return false;
    }

    const int frame_id = frame_attr->dec();
    if (frame_id <= last_masks_frame_seen_)
        return false;

    const int mask_count = std::max(0, count_attr->dec());
    const auto& labels = labels_attr->str();
    const auto& label_ids = label_ids_attr->float_vec();
    const auto& confidences = confs_attr->float_vec();
    const auto& offsets = offsets_attr->float_vec();
    const auto& support_flat = points_attr->float_vec();
    const auto& centroids_flat = centroids_attr->float_vec();
    const auto& bbox_min_flat = bbox_min_attr->float_vec();
    const auto& bbox_max_flat = bbox_max_attr->float_vec();

    const std::size_t support_count = support_flat.size() / 3;
    const std::size_t centroid_count = centroids_flat.size() / 3;
    const std::size_t bbox_min_count = bbox_min_flat.size() / 3;
    const std::size_t bbox_max_count = bbox_max_flat.size() / 3;

    std::vector<std::string> label_tokens;
    {
        std::stringstream ss(labels);
        std::string token;
        while (std::getline(ss, token, '|'))
            label_tokens.push_back(token);
    }

    MasksPacket packet;
    packet.valid = true;
    packet.frame_id = frame_id;
    packet.support_points.reserve(support_count);
    for (std::size_t i = 0; i < support_count; ++i)
        packet.support_points.emplace_back(support_flat[i*3], support_flat[i*3+1], support_flat[i*3+2]);

    packet.slices.reserve(static_cast<std::size_t>(mask_count));
    for (int i = 0; i < mask_count; ++i)
    {
        const std::size_t begin = (i < static_cast<int>(offsets.size())) ? static_cast<std::size_t>(std::max(0.0f, offsets[static_cast<std::size_t>(i)])) : 0;
        const std::size_t end = (i + 1 < static_cast<int>(offsets.size())) ? static_cast<std::size_t>(std::max(0.0f, offsets[static_cast<std::size_t>(i + 1)])) : begin;
        const std::size_t clamped_begin = std::min(begin, packet.support_points.size());
        const std::size_t clamped_end = std::min(end, packet.support_points.size());

        const auto fetch_vec3 = [&](const std::vector<float>& flat, std::size_t count) -> Eigen::Vector3f
        {
            if (static_cast<std::size_t>(i) >= count)
                return Eigen::Vector3f::Zero();
            return {flat[i*3], flat[i*3+1], flat[i*3+2]};
        };

        MaskSlice slice;
        if (static_cast<std::size_t>(i) < label_tokens.size())
            slice.label = label_tokens[static_cast<std::size_t>(i)];
        if (static_cast<std::size_t>(i) < label_ids.size())
            slice.class_id = label_ids[static_cast<std::size_t>(i)];
        if (static_cast<std::size_t>(i) < confidences.size())
            slice.confidence = confidences[static_cast<std::size_t>(i)];
        slice.support_begin = clamped_begin;
        slice.support_end = clamped_end;
        slice.centroid = fetch_vec3(centroids_flat, centroid_count);
        slice.bbox_min = fetch_vec3(bbox_min_flat, bbox_min_count);
        slice.bbox_max = fetch_vec3(bbox_max_flat, bbox_max_count);
        packet.slices.push_back(slice);
    }

    masks_packet_ = std::move(packet);
    last_masks_frame_seen_ = frame_id;
    return true;
}

std::optional<SpecificWorker::MaskSlice> SpecificWorker::select_mask_for_bottle(const BottleInstance& inst) const
{
    if (not masks_packet_.valid or masks_packet_.slices.empty())
        return std::nullopt;

    const auto& state = inst.model.state();
    const Eigen::Vector3f bottle_centroid(state.cx, state.cy, state.cz);

    float best_cost = std::numeric_limits<float>::max();
    std::optional<MaskSlice> best;
    for (const auto& slice : masks_packet_.slices)
    {
        if (slice.label != "bottle")
            continue;

        const float dx = slice.centroid.x() - bottle_centroid.x();
        const float dy = slice.centroid.y() - bottle_centroid.y();
        const float dz = slice.centroid.z() - bottle_centroid.z();
        const float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (dist < best_cost)
        {
            best_cost = dist;
            best = slice;
        }
    }

    return best;
}

// ─── Scaffold / instance management ────────────────────────────────────────────

void SpecificWorker::scaffold_missing_bottle_nodes()
{
    if (not prior_store_ or not masks_packet_.valid or priors_cache_.empty())
        return;

    std::vector<bool> mask_used(masks_packet_.slices.size(), false);

    for (const auto& p : priors_cache_)
    {
        if (G->get_node(p.node_name).has_value())
            continue;

        int best_mask_idx = -1;
        int fallback_mask_idx = -1;
        float best_dist_xy = std::numeric_limits<float>::max();
        const float max_match_dist_xy = std::max(0.5f, 3.0f * p.sigma_pose + p.radius_m);
        for (std::size_t i = 0; i < masks_packet_.slices.size(); ++i)
        {
            if (mask_used[i])
                continue;

            const auto& slice = masks_packet_.slices[i];
            if (slice.label != "bottle")
                continue;
            if (slice.support_end <= slice.support_begin)
                continue;

            if (fallback_mask_idx < 0)
                fallback_mask_idx = static_cast<int>(i);

            const float dx = slice.centroid.x() - p.room_x_m;
            const float dy = slice.centroid.y() - p.room_y_m;
            const float dist_xy = std::hypot(dx, dy);
            if (dist_xy <= max_match_dist_xy and dist_xy < best_dist_xy)
            {
                best_dist_xy = dist_xy;
                best_mask_idx = static_cast<int>(i);
            }
        }

        if (best_mask_idx < 0)
            best_mask_idx = fallback_mask_idx;
        if (best_mask_idx < 0)
            continue;

        const auto& matched_slice = masks_packet_.slices[static_cast<std::size_t>(best_mask_idx)];
        mask_used[static_cast<std::size_t>(best_mask_idx)] = true;

        auto room_opt = G->get_node(room_node_id_);
        if (not room_opt.has_value())
        {
            qWarning() << "bottle_concept: room node missing, cannot scaffold" << p.node_name.c_str();
            continue;
        }

        DSR::Node bottle_node = DSR::Node::create<cylinder_node_type>(p.node_name);
        G->add_or_modify_attrib_local<width_m_att> (bottle_node, 2.0f * p.radius_m);
        G->add_or_modify_attrib_local<depth_m_att> (bottle_node, 2.0f * p.radius_m);
        G->add_or_modify_attrib_local<height_m_att>(bottle_node, p.height_m);
        G->add_or_modify_attrib_local<level_att>   (bottle_node, 3);
        G->add_or_modify_attrib_local<parent_att>  (bottle_node, room_node_id_);
        {
            const float rpx = G->get_attrib_by_name<pos_x_att>(room_opt.value()).value_or(200.f);
            const float rpy = G->get_attrib_by_name<pos_y_att>(room_opt.value()).value_or(200.f);
            G->add_or_modify_attrib_local<pos_x_att>(bottle_node, rpx + 180.f);
            G->add_or_modify_attrib_local<pos_y_att>(bottle_node, rpy +  80.f);
        }

        const auto id_opt = G->insert_node(bottle_node);
        if (not id_opt.has_value())
        {
            qWarning() << "bottle_concept: failed to insert node" << p.node_name.c_str();
            continue;
        }

        const float cz = matched_slice.centroid.z() > 1e-3f ? matched_slice.centroid.z() : p.room_z_m;
        rt_api_->insert_or_assign_edge_RT(room_opt.value(), id_opt.value(),
                                          {matched_slice.centroid.x(), matched_slice.centroid.y(), cz},
                                          {0.0f, 0.0f, 0.0f});

        std::print("bottle_concept: created node '{}' id={} from masks frame={} at ({:.2f}, {:.2f}, {:.2f})\n",
                   p.node_name, id_opt.value(), masks_packet_.frame_id,
                   matched_slice.centroid.x(), matched_slice.centroid.y(), cz);
    }
}

void SpecificWorker::ensure_instance(const DSR::Node& node)
{
    if (instances_.count(node.id()))
        return;

    BottleState init_state;
    init_state.radius = cfg_.prior_radius;
    init_state.height = cfg_.prior_height;

    if (auto v = G->get_attrib_by_name<width_m_att> (node); v.has_value()) init_state.radius = 0.5f * v.value();
    if (auto v = G->get_attrib_by_name<height_m_att>(node); v.has_value()) init_state.height = v.value();

    // Read RT pose from the room→bottle edge.
    if (room_node_id_ != 0)
    {
        if (const auto edge = G->get_edge(room_node_id_, node.id(), "RT"); edge.has_value())
        {
            if (const auto tr = G->get_attrib_by_name<rt_translation_att>(edge.value()); tr.has_value())
            {
                const auto& tvec = tr.value().get();
                if (tvec.size() >= 3)
                {
                    init_state.cx = tvec[0];
                    init_state.cy = tvec[1];
                    init_state.cz = tvec[2];
                }
            }
        }
    }

    // Checkpoint overrides RT/attrs if present — but only when its size is physically
    // plausible. A checkpoint saved from a diverged fit (e.g. radius blown up by sensor
    // outliers) would otherwise poison BOTH the initial state AND the size prior below
    // (mparams.prior_radius = init_state.radius), anchoring the KL term at the bad value.
    if (prior_store_)
    {
        if (const auto ckpt = prior_store_->load_checkpoint(node.name()); ckpt.has_value())
        {
            constexpr float kMaxRadius = 0.20f;   // generous "is this a bottle?" bound (m)
            constexpr float kMaxHeight = 0.60f;
            const bool size_ok = std::isfinite(ckpt->radius_m) and std::isfinite(ckpt->height_m)
                and ckpt->radius_m > 0.0f and ckpt->radius_m <= kMaxRadius
                and ckpt->height_m > 0.0f and ckpt->height_m <= kMaxHeight;
            if (size_ok)
            {
                init_state.radius = ckpt->radius_m;
                init_state.height = ckpt->height_m;
            }
            else
                std::print("bottle_concept: REJECTED out-of-range checkpoint size for '{}' "
                           "(r={:.3f} h={:.3f}); falling back to prior (r={:.3f} h={:.3f})\n",
                           node.name(), ckpt->radius_m, ckpt->height_m, cfg_.prior_radius, cfg_.prior_height);
            init_state.cx = ckpt->room_x_m;
            init_state.cy = ckpt->room_y_m;
            init_state.cz = ckpt->room_z_m;
            std::print("bottle_concept: restored checkpoint for '{}'\n", node.name());
        }
    }

    // Sanitize non-finite fields (a corrupted checkpoint would poison the SDF).
    const auto fix = [&](float& v, float fallback)
    {
        if (not std::isfinite(v)) v = fallback;
    };
    fix(init_state.cx, 0.0f);
    fix(init_state.cy, 0.0f);
    fix(init_state.cz, cfg_.prior_radius > 0.f ? 0.85f : 0.85f);
    fix(init_state.radius, cfg_.prior_radius);
    fix(init_state.height, cfg_.prior_height);

    BottleModelParams mparams = make_model_params();
    // The size PRIOR is a fixed generative-model belief ("a bottle is ~2 cm"), NOT the restored
    // estimate. Anchoring it to init_state.radius let a diverged/checkpointed radius become its own
    // prior (size_energy≈0 there) → locked forever. A single depth view cannot observe radius (only
    // the front arc), so the prior must govern that direction; keep it pinned to config. The warm-
    // started init_state still seeds the optimizer, but is now regularized back toward the prior.
    mparams.prior_radius = cfg_.prior_radius;
    mparams.prior_height = cfg_.prior_height;
    if (prior_store_)
    {
        const auto it = std::find_if(priors_cache_.begin(), priors_cache_.end(),
                                     [&](const BottlePrior& pr){ return pr.node_name == node.name(); });
        if (it != priors_cache_.end())
            mparams.prior_size_std = std::max(mparams.prior_size_std, it->sigma_size);
    }

    BottleInstance inst;
    inst.node_id   = node.id();
    inst.node_name = node.name();
    inst.model     = BottleModel(init_state, mparams);
    inst.queue     = SampleQueue(make_queue_params());

    instances_.emplace(node.id(), std::move(inst));
    std::print("bottle_concept: created instance for node '{}' id={}\n", node.name(), node.id());
}

bool SpecificWorker::should_log(const BottleInstance& inst) const
{
    const int period = std::max(1, cfg_.log_period_frames);
    return (inst.processed_cycles % period) == 0;
}

// ─── Write model back to DSR ───────────────────────────────────────────────────

void SpecificWorker::step_write_model(BottleInstance& inst, DSR::Node& node, float free_energy)
{
    const auto& s = inst.model.state();

    // Publish to the graph only when the model meaningfully changed. Once the fit is stable this
    // stops the per-cycle rewrite of the node (incl. the large mesh + the ever-incrementing
    // model_generation) and the RT edge — sparing the DSR network and every agent's graph viewer.
    // FE jitter is deliberately NOT a trigger (it never fully settles), only pose/size do.
    constexpr float kPosEps  = 0.005f;   // 5 mm
    constexpr float kSizeEps = 0.002f;   // 2 mm
    const bool changed =
        std::abs(s.radius - inst.last_pub_radius) > kSizeEps or
        std::abs(s.height - inst.last_pub_height) > kSizeEps or
        std::abs(s.cx     - inst.last_pub_cx)     > kPosEps  or
        std::abs(s.cy     - inst.last_pub_cy)     > kPosEps  or
        std::abs(s.cz     - inst.last_pub_cz)     > kPosEps;
    if (not changed)
        return;

    inst.last_pub_radius = s.radius;
    inst.last_pub_height = s.height;
    inst.last_pub_cx     = s.cx;
    inst.last_pub_cy     = s.cy;
    inst.last_pub_cz     = s.cz;

    G->add_or_modify_attrib_local<width_m_att> (node, 2.0f * s.radius);
    G->add_or_modify_attrib_local<depth_m_att> (node, 2.0f * s.radius);
    G->add_or_modify_attrib_local<height_m_att>(node, s.height);
    G->add_or_modify_attrib_local<free_energy_att>(node, free_energy);
    G->add_or_modify_attrib_local<model_generation_att>(node, ++inst.model_generation);
    G->add_or_modify_attrib_local<mesh_vertices_att>(node, make_cylinder_mesh(s));

    // Export the historical RFE queue (remembered evidence) as XYZ triples.
    {
        const auto qpts = inst.queue.points();
        std::vector<float> qflat;
        qflat.reserve(qpts.size() * 3);
        for (const auto& p : qpts) { qflat.push_back(p.x()); qflat.push_back(p.y()); qflat.push_back(p.z()); }
        G->runtime_checked_add_or_modify_attrib_local(node, "rfe_pts", qflat);
    }

    G->update_node(node);

    write_rt_pose(room_node_id_, inst);
}

bool SpecificWorker::acquire_webots_gt()
{
    if (not inner_eigen_ or not webots2robocomp_proxy)
        return false;

    RoboCompWebots2Robocomp::ObjectPose bottle_w, robot_w;
    try
    {
        bottle_w = webots2robocomp_proxy->getObjectPose(cfg_.eval_bottle_def);
        robot_w  = webots2robocomp_proxy->getObjectPose(cfg_.eval_robot_def);
    }
    catch (const std::exception& e)
    {
        std::print("bottle_concept: [eval] getObjectPose failed ({}); will retry\n", e.what());
        return false;
    }

    auto to_iso = [](const RoboCompWebots2Robocomp::ObjectPose& p)
    {
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.linear() = Eigen::Quaterniond(p.orientation.w, p.orientation.x,
                                        p.orientation.y, p.orientation.z).normalized().toRotationMatrix();
        T.translation() = Eigen::Vector3d(p.position.x / 1000.0,   // Webots reports mm
                                          p.position.y / 1000.0,
                                          p.position.z / 1000.0);
        return T;
    };

    // The WaterBottle proto origin is the bottle BASE (geometry shifted +height/2), so the
    // queried pose is the base. Express the base in the DSR `body` frame (== Webots Shadow
    // frame), then walk the graph room←body to land in the room frame the tracker fits in.
    const Eigen::Isometry3d T_world_bottle = to_iso(bottle_w);
    const Eigen::Isometry3d T_world_body   = to_iso(robot_w);
    const Eigen::Vector3d base_in_body = (T_world_body.inverse() * T_world_bottle).translation();

    const auto room_T_body = inner_eigen_->get_transformation_matrix("room", "body", 0);
    if (not room_T_body.has_value())
        return false;

    const Eigen::Vector3d base_in_room =
        room_T_body.value().linear() * base_in_body + room_T_body.value().translation();

    cfg_.gt_cx = static_cast<float>(base_in_room.x());
    cfg_.gt_cy = static_cast<float>(base_in_room.y());
    cfg_.gt_cz = static_cast<float>(base_in_room.z()) + 0.5f * cfg_.gt_height;   // base → centre
    std::print("bottle_concept: [eval] Webots GT (room frame, centre) = ({:.3f}, {:.3f}, {:.3f})\n",
               cfg_.gt_cx, cfg_.gt_cy, cfg_.gt_cz);
    return true;
}

void SpecificWorker::step_move_experiment()
{
    if (not cfg_.move_experiment or not webots2robocomp_proxy)
        return;

    // Place the bottle at home + (dx,dy) in the WORLD frame (Webots), keeping z/orientation.
    const auto place = [this](float dx_mm, float dy_mm)
    {
        RoboCompWebots2Robocomp::ObjectPose p = move_home_;
        p.position.x = move_home_.position.x + dx_mm;
        p.position.y = move_home_.position.y + dy_mm;
        try { webots2robocomp_proxy->setObjectPose(cfg_.eval_bottle_def, p); }
        catch (const std::exception& e) { std::print("bottle_concept: [move-exp] setObjectPose failed ({})\n", e.what()); }
    };

    // Lazy start: capture the home pose and build a centred N×N grid over the table.
    if (not move_started_)
    {
        try { move_home_ = webots2robocomp_proxy->getObjectPose(cfg_.eval_bottle_def); }
        catch (const std::exception& e)
        {
            std::print("bottle_concept: [move-exp] getObjectPose(home) failed ({}); retrying\n", e.what());
            return;
        }
        const int n = std::max(1, cfg_.move_grid_n);
        const float step_mm = cfg_.move_step_m * 1000.0f;
        const float c = (n - 1) / 2.0f;     // centre the grid on home
        move_offsets_.clear();
        for (int iy = 0; iy < n; ++iy)
            for (int ix = 0; ix < n; ++ix)
                move_offsets_.emplace_back((ix - c) * step_mm, (iy - c) * step_mm);
        move_idx_ = 0;
        move_settle_ = 0;
        move_started_ = true;
        place(move_offsets_[0].first, move_offsets_[0].second);
        std::print("bottle_concept: [move-exp] started — {} poses, step {:.3f} m, settle {} cycles, log -> {}\n",
                   move_offsets_.size(), cfg_.move_step_m, cfg_.move_settle_cycles, cfg_.eval_log_path);
        return;
    }

    if (move_idx_ >= move_offsets_.size())
        return;   // finished

    // Hold each pose for move_settle_cycles so the fit converges (logged every cycle with the
    // 'settled' flag set once the hold completes), then step to the next grid pose.
    if (++move_settle_ < cfg_.move_settle_cycles)
        return;

    ++move_idx_;
    move_settle_ = 0;
    if (move_idx_ >= move_offsets_.size())
    {
        std::print("bottle_concept: [move-exp] DONE ({} poses) — returning bottle home\n", move_offsets_.size());
        try { webots2robocomp_proxy->setObjectPose(cfg_.eval_bottle_def, move_home_); } catch (...) {}
        cfg_.move_experiment = false;   // stop stepping; static eval (if enabled) continues
        return;
    }
    place(move_offsets_[move_idx_].first, move_offsets_[move_idx_].second);
}

bool SpecificWorker::place_static_test_pose()
{
    if (not cfg_.static_pose_test or not webots2robocomp_proxy)
        return false;
    if (move_started_)
        return true;   // already placed for this run

    // Home = persisted grid origin. Captured the first time the home file is absent, then reused
    // across restarts so the grid targets stay consistent even though prior runs left the bottle
    // elsewhere. (Before starting the sweep, delete the file with the bottle at its true home.)
    const std::string home_path = "etc/bottle_home_pose.txt";
    bool have_home = false;
    {
        std::ifstream hf(home_path);
        if (hf and (hf >> move_home_.position.x >> move_home_.position.y >> move_home_.position.z
                       >> move_home_.orientation.x >> move_home_.orientation.y
                       >> move_home_.orientation.z >> move_home_.orientation.w))
            have_home = true;
    }
    if (not have_home)
    {
        try { move_home_ = webots2robocomp_proxy->getObjectPose(cfg_.eval_bottle_def); }
        catch (const std::exception& e)
        {
            std::print("bottle_concept: [static-test] getObjectPose(home) failed ({}); retrying\n", e.what());
            return false;
        }
        std::ofstream hf(home_path);
        hf << move_home_.position.x << ' ' << move_home_.position.y << ' ' << move_home_.position.z << ' '
           << move_home_.orientation.x << ' ' << move_home_.orientation.y << ' '
           << move_home_.orientation.z << ' ' << move_home_.orientation.w << '\n';
        std::print("bottle_concept: [static-test] captured home ({:.0f},{:.0f},{:.0f}) mm -> {}\n",
                   move_home_.position.x, move_home_.position.y, move_home_.position.z, home_path);
    }

    // This run's grid pose (same centred N×N grid as the moving experiment).
    const int n = std::max(1, cfg_.move_grid_n);
    const int idx = std::clamp(cfg_.static_pose_index, 0, n * n - 1);
    const float step_mm = cfg_.move_step_m * 1000.0f;
    const float c = (n - 1) / 2.0f;
    const int ix = idx % n, iy = idx / n;
    RoboCompWebots2Robocomp::ObjectPose target = move_home_;
    target.position.x = move_home_.position.x + (ix - c) * step_mm;
    target.position.y = move_home_.position.y + (iy - c) * step_mm;
    try { webots2robocomp_proxy->setObjectPose(cfg_.eval_bottle_def, target); }
    catch (const std::exception& e)
    {
        std::print("bottle_concept: [static-test] setObjectPose failed ({})\n", e.what());
        return false;
    }
    move_idx_ = idx;
    move_started_ = true;
    std::print("bottle_concept: [static-test] pose {}/{} -> bottle at ({:.0f},{:.0f},{:.0f}) mm\n",
               idx, n * n, target.position.x, target.position.y, target.position.z);
    return true;
}

void SpecificWorker::log_eval(const BottleInstance& inst, float free_energy)
{
    if (not cfg_.eval_enabled)
        return;

    // Live GT from Webots. In the moving-bottle experiment the bottle changes pose, so refresh
    // GT EVERY cycle; otherwise acquire the static pose once (lazy — the bridge/transforms may not
    // be ready at startup). Until GT is available, skip logging.
    if (cfg_.eval_gt_source == "webots")
    {
        if (cfg_.move_experiment or cfg_.static_pose_test)
        {
            // Bottle was just moved (this run / this step): read its actual pose every cycle.
            if (not acquire_webots_gt())
                return;
        }
        else if (not gt_acquired_)
        {
            if (acquire_webots_gt())
                gt_acquired_ = true;
            else
                return;
        }
    }

    if (not eval_log_.is_open())
    {
        // Static-restart test APPENDS (each restart adds rows to the same CSV); otherwise truncate.
        const bool append = cfg_.static_pose_test;
        eval_log_.open(cfg_.eval_log_path, std::ios::out | (append ? std::ios::app : std::ios::trunc));
        if (not eval_log_.is_open())
        {
            std::print("bottle_concept: [eval] cannot open log '{}'\n", cfg_.eval_log_path);
            cfg_.eval_enabled = false;   // don't retry every cycle
            return;
        }
        if (not append or eval_log_.tellp() == std::streampos(0))
            eval_log_ << "frame,node,move_idx,settled,fe,cx,cy,cz,radius,height,"
                         "gt_cx,gt_cy,gt_cz,gt_radius,gt_height,"
                         "ex,ey,ez,pos_err,radius_err,height_err,nees,trace_P\n";
    }

    const auto& s = inst.model.state();

    // 3×3 Laplace position covariance (same P_bottle written on the RT edge).
    const auto fit_pts = inst.queue.points();
    if (fit_pts.empty())
        return;
    const Eigen::Matrix3f cov = inst.model.pose_covariance(fit_pts, inst.queue.weights());

    const Eigen::Vector3f e(s.cx - cfg_.gt_cx, s.cy - cfg_.gt_cy, s.cz - cfg_.gt_cz);

    // NEES = eᵀ P⁻¹ e (3-DOF position; χ² expectation ≈ 3 when P is calibrated).
    // Guard a singular/ill-conditioned P with a tiny isotropic regulariser.
    const Eigen::Matrix3f cov_reg = cov + Eigen::Matrix3f::Identity() * 1e-9f;
    const float nees = e.dot(cov_reg.ldlt().solve(e));

    const int settled = (cfg_.move_experiment and move_settle_ >= cfg_.move_settle_cycles) ? 1 : 0;
    eval_log_ << inst.processed_cycles << ',' << inst.node_name << ','
              << move_idx_ << ',' << settled << ',' << free_energy << ','
              << s.cx << ',' << s.cy << ',' << s.cz << ',' << s.radius << ',' << s.height << ','
              << cfg_.gt_cx << ',' << cfg_.gt_cy << ',' << cfg_.gt_cz << ','
              << cfg_.gt_radius << ',' << cfg_.gt_height << ','
              << e.x() << ',' << e.y() << ',' << e.z() << ',' << e.norm() << ','
              << (s.radius - cfg_.gt_radius) << ',' << (s.height - cfg_.gt_height) << ','
              << nees << ',' << cov.trace() << '\n';
    eval_log_.flush();
}

// ─── DSR helpers ───────────────────────────────────────────────────────────────

std::vector<Eigen::Vector3f> SpecificWorker::read_pts_attrib(const DSR::Node& node,
                                                             const std::string& att_name) const
{
    std::vector<Eigen::Vector3f> pts;

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

std::uint64_t SpecificWorker::voxel_key(const Eigen::Vector3f& point, float quantization_m)
{
    const float q = std::max(1e-4f, quantization_m);
    const int ix = static_cast<int>(std::floor(point.x() / q));
    const int iy = static_cast<int>(std::floor(point.y() / q));
    const int iz = static_cast<int>(std::floor(point.z() / q));

    std::uint64_t h = 1469598103934665603ULL;  // FNV-1a offset basis
    auto mix = [&](std::uint64_t v) { h ^= v; h *= 1099511628211ULL; };
    mix(static_cast<std::uint64_t>(ix));
    mix(static_cast<std::uint64_t>(iy));
    mix(static_cast<std::uint64_t>(iz));
    return h;
}

Eigen::Matrix2f SpecificWorker::read_robot_covariance() const
{
    const auto robots = G->get_nodes_by_type("robot");
    if (not robots.empty() and room_node_id_ != 0)
    {
        const auto edge = G->get_edge(room_node_id_, robots.front().id(), "RT");
        if (edge.has_value())
        {
            const auto cov_opt = G->get_attrib_by_name<rt_covariance_att>(edge.value());
            if (cov_opt.has_value())
            {
                const auto& c = cov_opt.value().get();
                // rt_covariance is the 6×6 SE3 covariance (row-major); the XY
                // block sits at indices (0,0),(0,1),(1,0),(1,1) with stride 6.
                if (c.size() >= 8)
                {
                    Eigen::Matrix2f m;
                    m << c[0], c[1], c[6], c[7];
                    return m;
                }
            }
        }
    }
    return Eigen::Matrix2f::Identity() * 0.01f;
}

void SpecificWorker::write_rt_pose(uint64_t room_id, BottleInstance& inst)
{
    if (room_id == 0 or not rt_api_)
        return;

    const auto& s = inst.model.state();

    // Dead-band: suppress RT updates below ~1 cm to avoid pos churn from jitter.
    constexpr float kMinWriteDistSq = 0.01f * 0.01f;
    const float dx = s.cx - inst.last_written_cx;
    const float dy = s.cy - inst.last_written_cy;
    if (dx*dx + dy*dy < kMinWriteDistSq)
        return;

    auto room_opt = G->get_node(room_id);
    if (not room_opt.has_value())
        return;

    rt_api_->insert_or_assign_edge_RT(room_opt.value(), inst.node_id,
                                      {s.cx, s.cy, s.cz},
                                      {0.0f, 0.0f, 0.0f});

    // Attach P_bottle (Laplace covariance) on the RT edge as the 6×6 SE3
    // covariance (rt_covariance_att, row-major [x,y,z,rx,ry,rz]). The 3×3
    // translation block comes from the model; orientation is unobservable for a
    // symmetric upright cylinder, so its diagonal is large. The controller reads
    // this to set its closed→open-loop look-up rate.
    if (auto edge = G->get_edge(room_id, inst.node_id, "RT"); edge.has_value())
    {
        const auto fit_pts = inst.queue.points();
        const Eigen::Matrix3f cov = inst.model.pose_covariance(fit_pts, inst.queue.weights());
        std::vector<float> cov_flat(36, 0.0f);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                cov_flat[r * 6 + c] = cov(r, c);
        cov_flat[3 * 6 + 3] = cfg_.yaw_variance;   // rx
        cov_flat[4 * 6 + 4] = cfg_.yaw_variance;   // ry
        cov_flat[5 * 6 + 5] = cfg_.yaw_variance;   // rz
        G->add_or_modify_attrib_local<rt_covariance_att>(edge.value(), cov_flat);
        G->insert_or_assign_edge(edge.value());
    }

    inst.last_written_cx = s.cx;
    inst.last_written_cy = s.cy;
}

std::vector<float> SpecificWorker::make_cylinder_mesh(const BottleState& s, int segments)
{
    // Flat triangle list (room frame): side wall + top/bottom caps.
    std::vector<float> verts;
    const int n = std::max(6, segments);
    verts.reserve(static_cast<std::size_t>(n) * 4 * 9);

    const float r  = s.radius;
    const float zt = s.cz + s.height * 0.5f;
    const float zb = s.cz - s.height * 0.5f;
    const float two_pi = 2.0f * std::numbers::pi_v<float>;

    auto push = [&](float x, float y, float z) { verts.push_back(x); verts.push_back(y); verts.push_back(z); };

    for (int i = 0; i < n; ++i)
    {
        const float a0 = two_pi * static_cast<float>(i)     / static_cast<float>(n);
        const float a1 = two_pi * static_cast<float>(i + 1) / static_cast<float>(n);
        const float x0 = s.cx + r * std::cos(a0), y0 = s.cy + r * std::sin(a0);
        const float x1 = s.cx + r * std::cos(a1), y1 = s.cy + r * std::sin(a1);

        // Side wall (two triangles).
        push(x0, y0, zb); push(x1, y1, zb); push(x1, y1, zt);
        push(x0, y0, zb); push(x1, y1, zt); push(x0, y0, zt);
        // Top cap.
        push(s.cx, s.cy, zt); push(x0, y0, zt); push(x1, y1, zt);
        // Bottom cap.
        push(s.cx, s.cy, zb); push(x1, y1, zb); push(x0, y0, zb);
    }

    return verts;
}

// ─── Factory helpers ─────────────────────────────────────────────────────────

BottleModelParams SpecificWorker::make_model_params() const
{
    BottleModelParams p;
    p.sigma_obs          = cfg_.sigma_obs;
    p.lambda_size        = cfg_.lambda_size;
    p.lambda_pos         = cfg_.lambda_pos;
    p.lambda_state       = cfg_.lambda_state;
    p.prior_radius       = cfg_.prior_radius;
    p.prior_height       = cfg_.prior_height;
    p.prior_size_std     = cfg_.prior_size_std;
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
    p.rfe_weight_gain              = cfg_.rfe_weight_gain;
    p.min_anchor_weight            = cfg_.min_anchor_weight;
    p.edge_bonus_weight            = cfg_.edge_bonus_weight;
    p.edge_proximity_threshold     = cfg_.edge_proximity_threshold;
    p.z_bin_size                   = cfg_.z_bin_size;
    return p;
}

// ─── Config ────────────────────────────────────────────────────────────────────

void SpecificWorker::load_config(const ConfigLoader& cfg)
{
    auto getf = [&](const std::string& k, float def) -> float {
        return cfg.exists(k) ? static_cast<float>(cfg.get<double>(k)) : def;
    };
    auto geti = [&](const std::string& k, int def) -> int {
        return cfg.exists(k) ? cfg.get<int>(k) : def;
    };
    auto gets = [&](const std::string& k, std::string def) -> std::string {
        return cfg.exists(k) ? cfg.get<std::string>(k) : def;
    };

    priors_path_     = gets("BottleConcept.PriorsPath",     "etc/object_priors.toml");
    checkpoint_path_ = gets("BottleConcept.CheckpointPath", "etc/checkpoint.toml");

    cfg_.fe_eps            = getf("BottleConcept.FEps",            1e-3f);
    cfg_.K_stable          = geti("BottleConcept.KStable",         30);
    cfg_.write_threshold   = getf("BottleConcept.WriteThreshold",  1e-3f);
    cfg_.log_period_frames = geti("BottleConcept.LogPeriodFrames", 30);
    cfg_.voxel_bank_max_points        = geti("BottleConcept.VoxelBankMaxPoints",        4000);
    cfg_.voxel_bank_quantization_m    = getf("BottleConcept.VoxelBankQuantizationM",    0.01f);
    cfg_.voxel_select_radius_margin_m = getf("BottleConcept.VoxelSelectRadiusMarginM",  0.10f);
    cfg_.voxel_select_height_margin_m = getf("BottleConcept.VoxelSelectHeightMarginM",  0.10f);

    cfg_.sigma_obs          = getf("BottleModel.SigmaObs",          0.02f);
    cfg_.lambda_size        = getf("BottleModel.LambdaSize",        0.5f);
    cfg_.lambda_pos         = getf("BottleModel.LambdaPos",         0.05f);
    cfg_.lambda_state       = getf("BottleModel.LambdaState",       0.02f);
    cfg_.prior_radius       = getf("BottleModel.PriorRadius",       0.035f);
    cfg_.prior_height       = getf("BottleModel.PriorHeight",       0.20f);
    cfg_.prior_size_std     = getf("BottleModel.PriorSizeStd",      0.03f);
    cfg_.optimization_iters = geti("BottleModel.OptimizationIters", 15);
    cfg_.optimization_lr    = getf("BottleModel.OptimizationLr",    0.005f);
    cfg_.grad_clip          = getf("BottleModel.GradClip",          2.0f);
    cfg_.optimizer_type     = gets("BottleModel.OptimizerType",     "adam");
    cfg_.sgd_momentum       = getf("BottleModel.SgdMomentum",       0.9f);
    {
        const auto loss_name = gets("BottleModel.RobustLoss", "quadratic");
        const auto loss_type = robust_loss_type_from_string(loss_name);
        cfg_.robust_loss = loss_type.value_or(RobustLossType::Quadratic);
    }
    cfg_.robust_loss_scale  = getf("BottleModel.RobustLossScale",  0.05f);

    cfg_.num_angle_bins               = geti("SampleQueue.NumAngleBins",              16);
    cfg_.num_z_bins                   = geti("SampleQueue.NumZBins",                  6);
    cfg_.max_per_bin                  = geti("SampleQueue.MaxPerBin",                 2);
    cfg_.sdf_threshold_for_storage    = getf("SampleQueue.SdfThresholdForStorage",    0.03f);
    cfg_.min_frames_before_historical = geti("SampleQueue.MinFramesBeforeHistorical", 10);
    cfg_.historical_warmup_frames     = geti("SampleQueue.HistoricalWarmupFrames",    5);
    cfg_.max_new_points_per_frame     = geti("SampleQueue.MaxNewPointsPerFrame",      20);
    cfg_.rfe_alpha                    = getf("SampleQueue.RfeAlpha",                  0.98f);
    cfg_.rfe_max_threshold            = getf("SampleQueue.RfeMaxThreshold",           2.0f);
    cfg_.rfe_weight_gain              = getf("SampleQueue.RfeWeightGain",             0.25f);
    cfg_.min_anchor_weight            = getf("SampleQueue.MinAnchorWeight",           0.12f);
    cfg_.edge_bonus_weight            = getf("SampleQueue.EdgeBonusWeight",           0.3f);
    cfg_.edge_proximity_threshold     = getf("SampleQueue.EdgeProximityThreshold",    0.01f);
    cfg_.z_bin_size                   = getf("SampleQueue.ZBinSize",                  0.04f);

    cfg_.yaw_variance = getf("BottleConcept.YawVariance", 9.87f);

    // Static Webots ground-truth evaluation (room frame; cylinder centre).
    auto getb = [&](const std::string& k, bool def) -> bool {
        return cfg.exists(k) ? cfg.get<bool>(k) : def;
    };
    cfg_.eval_enabled    = getb("Eval.Enabled",  false);
    cfg_.eval_log_path   = gets("Eval.LogPath",  "etc/bottle_eval.csv");
    cfg_.eval_gt_source  = gets("Eval.GtSource", "webots");
    cfg_.eval_bottle_def = gets("Eval.BottleDef", "bottle");
    cfg_.eval_robot_def  = gets("Eval.RobotDef",  "shadow");
    cfg_.gt_cx     = getf("Eval.GtCx",     0.0f);
    cfg_.gt_cy     = getf("Eval.GtCy",     0.0f);
    cfg_.gt_cz     = getf("Eval.GtCz",     0.0f);
    cfg_.gt_radius = getf("Eval.GtRadius", 0.0f);
    cfg_.gt_height = getf("Eval.GtHeight", 0.0f);

    // Moving-bottle validation experiment.
    cfg_.move_experiment    = getb("Eval.MoveExperiment", false);
    cfg_.move_settle_cycles = geti("Eval.MoveSettleCycles", 25);
    cfg_.move_step_m        = getf("Eval.MoveStep", 0.06f);
    cfg_.move_grid_n        = geti("Eval.MoveGridN", 5);
    // Static-restart validation (one grid pose per run; index from env BOTTLE_TEST_POSE).
    cfg_.static_pose_test   = getb("Eval.StaticPoseTest", false);
    if (const char* p = std::getenv("BOTTLE_TEST_POSE"))
        cfg_.static_pose_index = std::atoi(p);
    if (cfg_.static_pose_test)
        cfg_.move_experiment = false;          // static-restart takes precedence
    if (cfg_.move_experiment or cfg_.static_pose_test)
        cfg_.eval_enabled = true;              // the experiment is pointless without logging

    std::print("bottle_concept: configuration loaded.\n");
}

// ─── Graph slots / lifecycle ───────────────────────────────────────────────────

void SpecificWorker::del_node_slot(std::uint64_t id)
{
    instances_.erase(id);
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
