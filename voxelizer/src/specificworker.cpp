/*
 *    Copyright (C) 2026 by YOUR NAME HERE
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
#include "specificworker.h"
#include <fstream>   // [perf-probe] CSV timing logs (remove with the probes)
#include "scene_processor.h"
#include "../../common/media_transport/rt_extrapolate.h"
#include "yolo_processor.h"
#include "yolo_human.h"
#include "yolo_semantic.h"
#include "graph_publisher.h"
#include <dsr/api/dsr_camera_api.h>   // ricoh CameraAPI for the main-thread lidar depth-fill (reproject_cloud)
#include "../../common/depth_projection/depth_projection.h"   // lidar→panorama reprojection + mask depth scoring
#ifdef emit
#undef emit
#endif
#include "yolo_viewer.h"
#include "image_popup_viewer.h"
#include "perception_worker.h"
#include "seg_stage.h"
#include "pose_stage.h"
#include "semantic_stage.h"
#include "semantic_mask_stage.h"
#include "sam2_stage.h"
#include "bearing_stage.h"
#include "depth_stage.h"
#include "room_envelope_depth.h"
#include "ricoh_source.h"
#include "zed_source.h"
#include <dsr/gui/viewers/graph_viewer/graph_viewer.h>
#include <QHBoxLayout>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <cmath>
#include <limits>
#include <filesystem>
#include <map>
#include <print>
#include <sstream>
#include <thread>
#include <unordered_map>

#include <QCoreApplication>
#include <QEventLoop>

SpecificWorker::SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check)
    : GenericWorker(configLoader, tprx)
{
    this->startup_check_flag = startup_check;
    if (this->startup_check_flag)
    {
        this->startup_check();
    }
    else
    {
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

        auto error = statemachine.errorString();
        if (error.length() > 0)
        {
            qWarning() << error;
            throw error;
        }
    }
}

SpecificWorker::~SpecificWorker()
{
    request_shutdown();
    qInfo() << "Destroying SpecificWorker";
}

void SpecificWorker::request_shutdown()
{
    if (shutting_down_.exchange(true))
        return;

    save_window_settings();
    save_external_window_geometry();
    ricoh_worker_.reset();        // stop+join BEFORE scene_processor is destroyed (RicohSource holds a raw ptr to it)
    zed_worker_.reset();          // stop+join the ZED perception worker (self-contained; order not critical)
    scene_processor.reset();
    cleanup_owned_nodes();
}

void SpecificWorker::initialize()
{
    qInfo() << "initialize voxelizer worker";
    GenericWorker::initialize();

    // --- Configuration ---
    params = load_voxelizer_params(configLoader);
    verbose_debug_ = params.VERBOSE_DEBUG;

    // Cap OpenCV's implicit thread pool. The per-cycle preprocessing (resize/split/convertTo of the
    // YOLO frames) otherwise fans out across ALL cores in short bursts, inflating the process's
    // core-count for negligible latency gain on a 640² tensor. 2 threads is plenty here.
    cv::setNumThreads(2);

    // --- ZED perception worker: a set of Stages (seg, pose, …) on ONE thread, off the main compute/render
    // tick. Main hands it a deep-copied frame each cycle and publishes the self-contained bundle it hands
    // back (see compute()). Each Stage owns its own ONNX session. ---
    YoloProcessor::Config yolo_config;
    yolo_config.model_path          = params.YOLO_MODEL_PATH;
    yolo_config.conf_thresh         = params.YOLO_CONF_THRESH;
    yolo_config.iou_thresh          = params.YOLO_IOU_THRESH;
    yolo_config.input_size          = params.YOLO_INPUT_SIZE;
    yolo_config.use_gpu             = params.YOLO_USE_GPU;
    yolo_config.use_trt             = params.YOLO_USE_TRT;
    yolo_config.mask_erode_kernel   = params.YOLO_MASK_ERODE_KERNEL;
    yolo_config.mask_tray           = params.YOLO_MASK_TRAY;
    yolo_config.tray_mask_ref_width = params.YOLO_TRAY_MASK_REF_WIDTH;
    yolo_config.tray_mask_ref_height= params.YOLO_TRAY_MASK_REF_HEIGHT;
    yolo_config.tray_mask_polygon_px= params.YOLO_TRAY_MASK_POLYGON_PX;
    yolo_config.tray_drop_fraction  = params.YOLO_TRAY_DROP_FRACTION;
    yolo_config.accepted_labels     = params.YOLO_ACCEPTED_LABELS;
    yolo_config.second_best_margin  = params.YOLO_SECOND_BEST_MARGIN;
    yolo_config.verbose_debug       = verbose_debug_;
    // Build the ZED worker's stage list (seg + optional pose/semantic; each loads its ONNX model here).
    // The worker itself is STARTED later, after the media plane is up — its ZedSource pulls from it.
    std::vector<std::unique_ptr<rc::Stage>> zed_stages;
    zed_stages.push_back(std::make_unique<rc::SegStage>(yolo_config));

    // --- Human pose (optional second model → BODY_18 skeletons on the 'skeleton' node) ---
    if (params.HUMAN_POSE_ENABLED)
    {
        rc::human_pose::YoloHumanProcessor::Config pose_config;
        pose_config.model_path    = params.HUMAN_POSE_MODEL_PATH;
        pose_config.conf_thresh   = params.HUMAN_POSE_CONF_THRESH;
        pose_config.iou_thresh    = params.HUMAN_POSE_IOU_THRESH;
        pose_config.input_size    = params.HUMAN_POSE_INPUT_SIZE;
        pose_config.use_gpu       = params.HUMAN_POSE_USE_GPU;
        pose_config.use_trt       = params.HUMAN_POSE_USE_TRT;
        pose_config.hold_ms       = params.HUMAN_POSE_HOLD_MS;
        pose_config.verbose_debug = verbose_debug_;
        zed_stages.push_back(std::make_unique<rc::PoseStage>(pose_config, params.HUMAN_POSE_DECIMATION));
    }

    // --- Semantic segmentation (optional dense ADE20K-150 class map → viewer overlay). Starts DISABLED
    // (gated by the ZED-window "Semantic" toggle) so the heavy model runs only when shown. ---
    if (params.SEMANTIC_SEG_ENABLED)
    {
        rc::semantic::YoloSemanticProcessor::Config sem_config;
        sem_config.model_path    = params.SEMANTIC_SEG_MODEL_PATH;
        sem_config.conf_thresh   = params.SEMANTIC_SEG_CONF_THRESH;
        sem_config.input_size    = params.SEMANTIC_SEG_INPUT_SIZE;
        sem_config.use_gpu       = params.SEMANTIC_SEG_USE_GPU;
        sem_config.use_trt       = params.SEMANTIC_SEG_USE_TRT;
        sem_config.verbose_debug = verbose_debug_;
        // want_scores ON when we publish semantic masks (they need the per-pixel confidence to be scored).
        auto sem_stage = std::make_unique<rc::SemanticStage>(sem_config, params.SEMANTIC_SEG_DECIMATION,
                                                             /*want_scores=*/params.SEMANTIC_PUBLISH_MASKS);
        // Run when the overlay is on OR when producing semantic masks (publish_masks) — so the model runs
        // even without the viewer toggle (mirrors the SAM2 publish_refined pattern below).
        rc::SemanticStage* sem_stage_ptr = sem_stage.get();
        sem_stage->set_enabled(semantic_overlay_enabled_ or params.SEMANTIC_PUBLISH_MASKS or params.SEMANTIC_PUBLISH_NODE);
        zed_stages.push_back(std::move(sem_stage));

        // Semantic-derived instance masks (cabinet/hood/shelf/door → 'masks' node, SAM2-refined). Resolve the
        // accepted label strings to ADE20K class ids via the model's own class table, once, here.
        if (params.SEMANTIC_PUBLISH_MASKS and sem_stage_ptr->processor())
        {
            const auto& names = sem_stage_ptr->processor()->class_names();
            const auto iequals = [](const std::string& a, const std::string& b)
            {
                return a.size() == b.size() and std::equal(a.begin(), a.end(), b.begin(),
                    [](char x, char y) { return std::tolower((unsigned char)x) == std::tolower((unsigned char)y); });
            };
            // Some ADE20K classes are emitted UNDER a different downstream label: e.g. YOLO-seg (COCO) has no
            // "radiator", and radiators are otherwise mistaken for chairs, so the ADE20K classes that a radiator
            // segments into ("railing"/"bannister" — the grille/fins) are relabelled to "radiator".
            // Key = ADE20K canonical name (lowercase); value = output label the mask carries.
            static const std::vector<std::pair<std::string, std::string>> label_alias = {
                {"railing", "radiator"}, {"bannister", "radiator"}
            };
            std::vector<std::pair<int, std::string>> accepted_classes;
            for (const auto& want : params.SEMANTIC_ACCEPTED_LABELS)
            {
                const auto it = std::find_if(names.begin(), names.end(),
                                             [&](const std::string& n) { return iequals(n, want); });
                if (it != names.end())
                {
                    std::string out_label = *it;   // default: emit under the ADE20K name itself
                    for (const auto& [ade, alias] : label_alias)
                        if (iequals(*it, ade)) { out_label = alias; break; }
                    accepted_classes.emplace_back(static_cast<int>(std::distance(names.begin(), it)), out_label);
                }
                else
                    std::println("[SemanticMasks] class '{}' not in the model's ADE20K table — skipped", want);
            }
            if (not accepted_classes.empty())
                zed_stages.push_back(std::make_unique<rc::SemanticMaskStage>(
                    std::move(accepted_classes), params.SEMANTIC_MASK_MIN_AREA_FRAC,
                    params.SEMANTIC_MASK_OVERLAP_DROP_FRAC, params.SEMANTIC_MASK_MORPH_KERNEL,
                    params.SEMANTIC_MASK_SCORE_DEFAULT));
        }
        else if (params.SEMANTIC_PUBLISH_MASKS)
            std::println("[SemanticMasks] semantic model not ready — semantic masks disabled");
    }

    // --- SAM2 mask refinement (optional). Runs AFTER SegStage so it can read its detections and sharpen
    // the target-class masks. Starts DISABLED (gated by the ZED-window "SAM2" toggle) so the heavy 1024²
    // encoder runs only when shown. Its refined masks land in PerceptionResult::refined_masks. ---
    if (params.SAM2_ENABLED)
    {
        rc::sam2::Config sam2_config;
        sam2_config.encoder_path = params.SAM2_ENCODER_PATH;
        sam2_config.decoder_path = params.SAM2_DECODER_PATH;
        sam2_config.use_gpu          = params.SAM2_USE_GPU;
        sam2_config.encoder_use_trt  = params.SAM2_ENCODER_USE_TRT;
        sam2_config.decoder_use_trt  = params.SAM2_DECODER_USE_TRT;
        sam2_config.mask_prior       = params.SAM2_MASK_PRIOR;
        sam2_config.mask_prior_logit = params.SAM2_MASK_PRIOR_LOGIT;
        sam2_config.verbose          = verbose_debug_;
        // Refine the semantic furniture masks too. If SAM2_REFINE_LABELS is empty it already means "all",
        // so only union when it is an explicit (non-empty) allow-list — otherwise we'd accidentally restrict it.
        std::vector<std::string> refine_labels = params.SAM2_REFINE_LABELS;
        if (params.SEMANTIC_PUBLISH_MASKS and not refine_labels.empty())
            for (const auto& l : params.SEMANTIC_ACCEPTED_LABELS)
                if (std::find(refine_labels.begin(), refine_labels.end(), l) == refine_labels.end())
                    refine_labels.push_back(l);
        auto sam2_stage = std::make_unique<rc::Sam2Stage>(sam2_config, refine_labels,
                                                          params.SAM2_DECIMATION, params.SAM2_METRICS_LOG);
        // Run when the overlay is on OR when routing refined masks to the fitters (publish_refined).
        sam2_stage->set_enabled(sam2_overlay_enabled_ or params.SAM2_PUBLISH_REFINED);
        zed_stages.push_back(std::move(sam2_stage));
    }

    // Depth model on the ZED frame — for VALIDATION only (the camera measures depth; the model does
    // not improve on that). Starts DISABLED and is armed by the ZED window's ZDepth button, so the
    // extra inference is only paid while the comparison is being looked at.
    if (params.ZED_DEPTH_ENABLED)
    {
        rc::depth::DepthProcessor::Config zdcfg;
        zdcfg.model_path    = params.RICOH_DEPTH_MODEL_PATH;
        zdcfg.input_size    = params.RICOH_DEPTH_INPUT_SIZE;
        zdcfg.use_gpu       = params.RICOH_DEPTH_USE_GPU;
        zdcfg.use_trt       = params.RICOH_DEPTH_USE_TRT;
        zdcfg.verbose_debug = verbose_debug_;
        auto zdepth = std::make_unique<rc::DepthStage>(zdcfg, rc::depth::Depth360Config{}, 1);
        zdepth->set_enabled(false);
        zed_stages.push_back(std::move(zdepth));
    }

    // Custom drawing windows (Voxel3D GL + YOLO raster), each in its own top-level window — see
    // specificworker_viewers.cpp. Both config-gated (Voxel.show_voxel_viewer / show_yolo_viewer).
    setup_custom_viewers();

    // --- Scene processor (DSR-only: no proxies) ---
    inner_eigen_api = G->get_inner_eigen_api();
    scene_processor = std::make_unique<SceneProcessor>(G);
    scene_processor->configure(inner_eigen_api.get(),
                               params.TRANSFORMS_INTERPOLATE_RT, verbose_debug_,
                               params.MASK_POSE_EXTRAPOLATE, params.MASK_POSE_EXTRAP_MAX_DT_S);
    scene_processor->init_media_plane(static_cast<std::uint32_t>(params.MEDIA_DOMAIN_ID),
                                      params.MEDIA_RGB_TOPIC, params.MEDIA_DEPTH_TOPIC);
    scene_processor->init_lidar_media_plane(static_cast<std::uint32_t>(params.MEDIA_DOMAIN_ID),
                                            params.MEDIA_LIDAR_TOPIC, params.LIDAR_USE_MEDIA);

    // Start the ZED worker in PULL mode now that the media plane is up: its ZedSource drains the aligned
    // RGBD + resolves room<-zed on the WORKER thread, so the whole RGBD path is off the main compute tick.
    {
        auto zed_src = std::make_shared<rc::ZedSource>(scene_processor.get(), G);
        zed_worker_ = std::make_unique<rc::PerceptionWorker>();
        rc::PerceptionWorker::Config wcfg;
        wcfg.name             = "zed";
        wcfg.perf_log         = params.PERF_LOG;
        // Poll granularity for the pull loop, NOT a rate limit — a cycle that finds a frame runs at its
        // own cost and never sleeps. At 15 ms this WAS the limiter once SAM2 came off: 25.3 ms of work
        // against a 25 ms camera still gave a 49 ms loop, because a near-miss cost a full 15 ms sleep.
        wcfg.target_period_ms = params.ZED_THREAD_PERIOD_MS;
        if (!zed_worker_->start(std::move(zed_stages), wcfg, [zed_src]() { return (*zed_src)(); }))
        {
            std::println("[Zed] perception worker failed to start (model load?) — disabling");
            zed_worker_.reset();
        }
        // Feed the semantic class-name table for the ZED-window hover readout, now that the stage exists.
        if (yolo_viewer_)
            if (auto* s = dynamic_cast<rc::SemanticStage*>(zed_worker_ ? zed_worker_->stage("semantic") : nullptr);
                s and s->processor())
                yolo_viewer_->set_class_names(s->processor()->class_names());
    }
    // ZED-image model-instance projection overlay (gated by the "Models" toggle in the ZED popup).
    // Does NO live graph traversal at draw time (caches the zed CameraAPI once); all geometry is fed
    // from the frame's already-gathered room←zed transform + room polygon.
    model_overlay_ = std::make_unique<rc::ModelProjectionOverlay>(G);
    // Ricoh-360 counterpart (equirectangular wireframe projection via the shared CameraAPI; caches the
    // ricoh CameraAPI once, then pure math per frame).
    ricoh_model_overlay_ = std::make_unique<rc::RicohProjectionOverlay>(G);
    // Media subscriber is needed for EITHER the popup or peripheral detection — independent of whether
    // the popup window is ever shown, so 360-YOLO can run without anyone watching.
    if (params.SHOW_RICOH_VIEWER or params.RICOH_YOLO_ENABLED)
        scene_processor->init_ricoh_media_plane(static_cast<std::uint32_t>(params.MEDIA_DOMAIN_ID),
                                                params.MEDIA_RICOH_TOPIC);

    // Ricoh 360 peripheral YOLO: own thread, own model/session — decoupled from compute()'s budget
    // (see ricoh_yolo_worker.h). Started here (after the media plane is up); stopped explicitly in
    // request_shutdown() before scene_processor is destroyed.
    if (params.RICOH_YOLO_ENABLED)
    {
        // Ricoh 360 = a PULL PerceptionWorker: its RicohSource polls the panorama plane + resolves
        // room<-ricoh at the stamp; SegStage(is_360) runs the 3-strip model; BearingStage turns each mask
        // into a room-frame bearing (all on the worker thread). LiDAR depth-fill stays main-side (compute()).
        YoloProcessor::Config ricoh_yolo_cfg;
        ricoh_yolo_cfg.model_path        = params.YOLO_MODEL_PATH;
        ricoh_yolo_cfg.conf_thresh       = params.YOLO_CONF_THRESH;
        ricoh_yolo_cfg.iou_thresh        = params.YOLO_IOU_THRESH;
        ricoh_yolo_cfg.input_size        = params.YOLO_INPUT_SIZE;
        ricoh_yolo_cfg.use_gpu           = params.YOLO_USE_GPU;
        ricoh_yolo_cfg.use_trt           = params.YOLO_USE_TRT;
        ricoh_yolo_cfg.mask_erode_kernel = params.YOLO_MASK_ERODE_KERNEL;
        ricoh_yolo_cfg.accepted_labels   = params.YOLO_ACCEPTED_LABELS;
        ricoh_yolo_cfg.second_best_margin= params.YOLO_SECOND_BEST_MARGIN;
        ricoh_yolo_cfg.verbose_debug     = verbose_debug_;

        Detection360Config cfg360;
        cfg360.n_strips   = params.RICOH_YOLO_N_STRIPS;
        cfg360.overlap_px = params.RICOH_YOLO_STRIP_OVERLAP_PX;
        cfg360.merge_iou  = params.RICOH_YOLO_MERGE_IOU;

        std::vector<std::unique_ptr<rc::Stage>> ricoh_stages;
        ricoh_stages.push_back(std::make_unique<rc::SegStage>(ricoh_yolo_cfg, cfg360));
        if (params.RICOH_PUBLISH_MASKS)
            ricoh_stages.push_back(std::make_unique<rc::BearingStage>(G));   // runs after seg (reads masks)
        // Monocular depth on the same panorama, own strip geometry ([RicohDepth]). Independent of seg —
        // it reads only in.rgbd.bgr — so stage order does not matter; last keeps the cheap stages first.
        if (params.RICOH_DEPTH_ENABLED)
        {
            rc::depth::DepthProcessor::Config dcfg;
            dcfg.model_path    = params.RICOH_DEPTH_MODEL_PATH;
            dcfg.input_size    = params.RICOH_DEPTH_INPUT_SIZE;
            dcfg.use_gpu       = params.RICOH_DEPTH_USE_GPU;
            dcfg.use_trt       = params.RICOH_DEPTH_USE_TRT;
            dcfg.verbose_debug = verbose_debug_;

            rc::depth::Depth360Config dcfg360;
            dcfg360.n_strips           = params.RICOH_DEPTH_N_STRIPS;
            dcfg360.overlap_px         = params.RICOH_DEPTH_OVERLAP_PX;
            dcfg360.band_half_elev_deg = params.RICOH_DEPTH_BAND_HALF_ELEV_DEG;
            dcfg360.gnomonic           = params.RICOH_DEPTH_GNOMONIC;
            dcfg360.gnomonic_fov_deg   = params.RICOH_DEPTH_GNOMONIC_FOV_DEG;
            dcfg360.zdepth_to_range    = params.RICOH_DEPTH_ZDEPTH_TO_RANGE;

            // A map fitted on a previous drive applies from the first frame — the dataset accumulates
            // across runs, so the correction should too.
            if (depth_fit_map_.load("etc/ricoh_depth_map.csv"))
                std::println("[depth-map] loaded etc/ricoh_depth_map.csv — a={:.3f} r={:+.3f} "
                             "resid ±{:.0f}% over {:.2f}..{:.2f} m ({} frames)",
                             depth_fit_map_.a, depth_fit_map_.r,
                             100.0 * (std::exp(depth_fit_map_.resid_rms) - 1.0),
                             depth_fit_map_.range_lo, depth_fit_map_.range_hi, depth_fit_map_.n_frames);

            auto depth_stage = std::make_unique<rc::DepthStage>(dcfg, dcfg360, params.RICOH_DEPTH_DECIMATION);
            // Gated by the popup's "Depth" toggle: a disabled stage is skipped entirely, so the model
            // does NO work while nobody is looking (same contract as the ZED semantic stage).
            depth_stage->set_enabled(ricoh_depth_overlay_enabled_);
            ricoh_stages.push_back(std::move(depth_stage));
        }

        auto ricoh_src = std::make_shared<rc::RicohSource>(scene_processor.get(), G, params.RICOH_AZIMUTH_TUNE_DEG);

        ricoh_worker_ = std::make_unique<rc::PerceptionWorker>();
        rc::PerceptionWorker::Config rcfg;
        rcfg.name             = "ricoh";
        rcfg.perf_log         = params.PERF_LOG;
        rcfg.target_period_ms = params.RICOH_YOLO_THREAD_PERIOD_MS;
        if (!ricoh_worker_->start(std::move(ricoh_stages), rcfg, [ricoh_src]() { return (*ricoh_src)(); }))
        {
            std::println("[Ricoh] perception worker failed to start (media plane not ready?) — disabling");
            ricoh_worker_.reset();
        }
    }

    // Perception-rate regulator: floor = the user-configured pose decimation, ceiling =

    // All DSR semantic_grid exports (masks). Relayout is injected so the publisher stays
    // decoupled from the GUI (graph_viewers).
    graph_publisher_ = std::make_unique<GraphPublisher>(
        G, params, [this]() { trigger_graph_layout_twopi(); });

    graph_publisher_->cleanup_semantic_grid_nodes();

    // Decoupled render timer: only worth running when the 3D viewer exists. 50 ms (20 Hz) matches the
    // viewer's repaint throttle, giving a stable 20 Hz refresh without re-running perception faster.
    // Created stopped; started in Operating (so graph reads happen only after the join completes).
    // Was gated on the 3-D GL widget existing; that widget now lives in the `viewer3d` agent, but this
    // timer also drives the ricoh popup, the ZED window and the publish-hold watchdog — so it is
    // unconditional. Gating it on a widget that moved out would have silently stopped all three.
    {
        render_timer_ = std::make_unique<QTimer>(this);
        render_timer_->setInterval(50);
        QObject::connect(render_timer_.get(), &QTimer::timeout, this, &SpecificWorker::on_render_tick);
    }

    presence_coordinator_.configure(configLoader, G, static_cast<std::uint32_t>(agent_id));
    // Colour this agent's node in the graph view by its live health: the coordinator already
    // publishes the presence lifecycle; this adds the external FSM axis (Initialize/Compute/
    // Emergency/Restore). Generic discovery via objectName(), so genericworker regeneration
    // cannot break it.
    presence_coordinator_.attach_state_machine(&statemachine);
    presence_coordinator_.set_transition_hooks({
        // Required peers are ready — but only advance to Operating once room_concept has also published
        // the 'room' node (its stability signal). If the room isn't up yet, stay in Waiting; on_waiting_loop
        // re-checks each tick and emits presenceReady the moment the room node appears.
        .request_presence_ready = [this]()
        {
            if (room_node_present())
                Q_EMIT presenceReady();
            else
                qInfo() << "[SM] required peers ready, but no stable 'room' node yet — holding in Waiting";
        },
        .request_presence_lost  = [this]()
        {
            if (current_sm_state_ == "Operating")   // tag WHY we're leaving Operating; the next Waiting event reports it
            {
                std::string peers;
                for (const auto &n : presence_coordinator_.missing_required_names())
                    peers += (peers.empty() ? "" : ",") + n;
                pending_exit_reason_ = "required_peer_lost";
                pending_exit_detail_ = "missing:" + peers;
            }
            Q_EMIT presenceLost();
        },
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
            qInfo() << "[SM] -> Waiting: holding until all required peers are present AND the room is stable"
                       " (no compute/graph access)";
            const auto missing = presence_coordinator_.missing_required_names();
            if (!missing.empty())
            {
                QString m;
                for (const auto &label : missing)
                    m += " " + QString::fromStdString(label);
                qInfo() << "[SM]   waiting for peer(s):" << m << "— e.g. launch room_concept";
            }
            else if (!room_node_present())
                qInfo() << "[SM]   peers present, waiting for a stable 'room' node from room_concept";

            // Structured event. If we got here by leaving Operating, report that reason (pending_exit_*);
            // otherwise this is a startup / dependency wait.
            if (!pending_exit_reason_.empty())
                log_sm_event(current_sm_state_, "Waiting", pending_exit_reason_, pending_exit_detail_);
            else
            {
                std::string detail;
                for (const auto &label : missing)
                    detail += (detail.empty() ? "" : ",") + label;
                log_sm_event(current_sm_state_, "Waiting", "awaiting_dependencies",
                             missing.empty() ? "room node not yet stable" : "missing:" + detail);
            }
            pending_exit_reason_.clear();
            pending_exit_detail_.clear();
            current_sm_state_ = "Waiting";
            last_waiting_log_ = std::chrono::steady_clock::now();   // reset the throttle so the loop re-logs after the interval
        },
        // Ongoing feedback + the room-stability gate re-check. The coordinator emits presence-ready only on
        // peer-state changes, so if the room node appears AFTER peers are already ready nothing else would
        // advance us — this loop closes that gap and re-emits. Logging is throttled to ~3 s (the loop runs
        // at the ~40 Hz Compute period) so it doesn't spam.
        .on_waiting_loop = [this]()
        {
            const bool peers_ready = presence_coordinator_.all_required_ready();
            const bool room_ready  = room_node_present();
            if (peers_ready and room_ready)
            {
                Q_EMIT presenceReady();   // peers were ready and the stable room node has now appeared → advance
                return;
            }

            const auto now = std::chrono::steady_clock::now();
            if (now - last_waiting_log_ < std::chrono::seconds(3))
                return;
            last_waiting_log_ = now;
            if (not peers_ready)
            {
                QString m;
                for (const auto &label : presence_coordinator_.missing_required_names())
                    m += " " + QString::fromStdString(label);
                qInfo() << "[SM] Waiting — required peer(s) not running:" << m;
            }
            else   // peers ready, room node not yet published
                qInfo() << "[SM] Waiting — required peers present, waiting for a stable 'room' node from room_concept";
        },
        .on_operating_enter = [this]()
        {
            log_sm_event(current_sm_state_, "Operating", "constraints_satisfied",
                         "required peers present and room stable");
            current_sm_state_ = "Operating";
            qInfo() << "[SM] -> Operating: all required constraints satisfied";
            room_absent_since_.reset();   // fresh session — clear any pending reverse-gate debounce
            if (render_timer_)
                render_timer_->start();   // start fluid viewer refresh only once the graph is joined
        },
        .on_operating_loop = [this]()
        {
            // Reverse room-stability gate: room_concept removes the 'room' node when localization goes
            // unstable, so a missing room node means we must stop operating and fall back to Waiting.
            // Debounced (~1.5 s) so a transient CRDT re-import gap during peer join/restart doesn't drop us.
            if (not room_node_present())
            {
                const auto now = std::chrono::steady_clock::now();
                if (not room_absent_since_)
                    room_absent_since_ = now;
                if (now - *room_absent_since_ >= std::chrono::milliseconds(1500))
                {
                    qInfo() << "[SM] 'room' node gone for >1.5s (room unstable) — leaving Operating";
                    pending_exit_reason_ = "room_unstable";       // reported by the next Waiting-enter event
                    pending_exit_detail_ = "room node absent >1.5s";
                    room_absent_since_.reset();
                    Q_EMIT presenceLost();   // → Degraded → Waiting; on_waiting_loop re-admits when the room returns
                    return;                  // skip compute this tick (its room reads would be empty anyway)
                }
            }
            else
                room_absent_since_.reset();   // room back within the grace window → cancel the pending drop

            compute();
            if (auto it = graph_viewers.find(""); it != graph_viewers.end() && it->second)
                it->second->set_external_fps(states.at("Operating")->getActualFps());
        },
        .on_degraded_enter = [this]()
        {
            // Transient pass-through (auto-transitions to Waiting). Deliberately does NOT touch
            // current_sm_state_ so the Waiting-enter event still reports from="Operating" with the exit reason.
            qInfo() << "[SM] -> Degraded: a required peer or node is no longer available";
            if (render_timer_)
                render_timer_->stop();   // no graph access outside Operating
        },
    });
    presence_coordinator_.start();

    QObject::connect(QCoreApplication::instance(), &QCoreApplication::aboutToQuit,
                     this, &SpecificWorker::request_shutdown, Qt::UniqueConnection);

    restore_window_settings();
}

// Decoupled viewer refresh (GUI thread, Operating only). Reads the LATEST robot pose from the graph
// and pushes it to the 3D viewer, independent of the perception/camera pipeline, so robot motion is
// fluid. set_robot_pose() triggers the viewer's own throttled repaint — no explicit update() here.
void SpecificWorker::on_render_tick()
{
    if (shutting_down_ || !scene_processor)
        return;

    // [perf] this timer shares the Qt main thread with compute() (both are plain QTimers on the
    // same event loop) — a slow render tick delays the next compute() tick just as much as a slow
    // compute() delays the next render tick. Logged separately from viewer_perf.csv (different timer).
    const auto perf_t0 = std::chrono::steady_clock::now();
    auto perf_ms = [](auto a, auto b){ return std::chrono::duration<double, std::milli>(b - a).count(); };


    // Ricoh 360 popup: refresh the window when visible. When the 360-YOLO worker is running it OWNS
    // polling (its own thread, see ricoh_yolo_worker.h) — just read its thread-safe snapshot here, no
    // decode on this thread. Otherwise (Ricoh.yolo_enabled=false) poll directly, same as always.
    double ricoh_ms = 0.0;
    {
        const auto perf_ricoh0 = std::chrono::steady_clock::now();
        const bool popup_visible = ricoh_viewer_ and ricoh_window_ and ricoh_window_->isVisible();

        if (ricoh_worker_)
        {
            if (popup_visible)
            {
                auto rres = ricoh_worker_->latest_result();
                if (rres and not rres->frame.rgbd.bgr.empty())
                {
                    cv::Mat pano = rres->frame.rgbd.bgr;   // BGR panorama the worker processed
                    // Project the DSR scene (model boxes + room floor/ceiling/walls) onto the panorama
                    // when the Ricoh "Models" toggle is on. Draw on a clone (BGR) so we never mutate
                    // the worker's frame; the popup viewer converts BGR→RGB on display.
                    if ((ricoh_model_overlay_enabled_ or ricoh_lidar_overlay_enabled_)
                        and ricoh_model_overlay_ and ricoh_scene_.valid)
                    {
                        pano = pano.clone();
                        if (ricoh_lidar_overlay_enabled_)
                            ricoh_model_overlay_->draw_lidar_points(pano, ricoh_scene_.lidar_room, ricoh_scene_.room_T_ricoh);
                        if (ricoh_model_overlay_enabled_)
                            ricoh_model_overlay_->draw(pano, ricoh_scene_.boxes, ricoh_scene_.room_T_ricoh,
                                                       ricoh_scene_.poly_x, ricoh_scene_.poly_y, ricoh_scene_.room_height);
                    }
                    // Monocular-depth ramp UNDER the seg silhouettes: it is a dense full-band layer, so
                    // drawing it last would bury the detections. Per-strip normalised with the seam
                    // lines drawn — see depth_processor.h for why it must not be read across a seam.
                    // ★The RoomΔ overlay below is NOT part of this block. The room envelope is a purely
                    // geometric prediction — it needs the polygon and the ricoh pose, and nothing from
                    // the monocular model — so nesting it under the Depth toggle (as it was) made the
                    // button silently inert whenever Depth was OFF or the LiDAR anchor was missing,
                    // while the ZED window's identical button kept working. Only mode 2 (DIFF) has a
                    // real dependency on the model, and it states it for itself.
                    auto* dp = dynamic_cast<rc::DepthStage*>(ricoh_worker_->stage("depth"));
                    rc::depth::DepthMap shown;
                    bool have_model = false, corrected = false;
                    if (ricoh_depth_overlay_enabled_ and dp and rres->depth and not rres->depth->empty())
                    {
                        // With a fitted map + this frame's LiDAR anchor, rewrite the raw per-view
                        // log-depth into METRIC log-range and draw it on one fixed scale. Without
                        // them, fall through to the per-strip relative ramp — the only honest way
                        // to render six mutually inconsistent scales.
                        have_model = true;
                        corrected = depth_fit_map_.valid and depth_anchor_.any();
                        shown = *rres->depth;
                        if (corrected)
                        {
                            shown.log_depth = rres->depth->log_depth.clone();   // never touch the worker's buffer
                            const int sw = (shown.n_strips > 0) ? shown.log_depth.cols / shown.n_strips
                                                                : shown.log_depth.cols;
                            // ★Offset INTERPOLATED across azimuth, not applied per-view as a step.
                            // b is fitted per view but the ROOM is continuous, so a piecewise
                            // constant correction puts a discontinuity at every seam whose size is
                            // exactly |b_i − b_{i+1}| — measured ~0.10 log ≈ a 10% depth step, which
                            // is what survives after the big per-view jumps are removed. Treating
                            // each b as SAMPLED AT ITS VIEW CENTRE and lerping circularly between
                            // neighbours makes the field continuous by construction while still
                            // reproducing the fitted value at each centre. This is display
                            // smoothing over a quantity that is genuinely per-image, so it belongs
                            // here at compose time and NOT in anything that feeds a belief.
                            const int nv = std::max(1, shown.n_strips);
                            for (int y = 0; y < shown.log_depth.rows; ++y)
                            {
                                auto* row = shown.log_depth.ptr<float>(y);
                                const auto* sid = shown.strip_id.ptr<unsigned char>(y);
                                const float t = std::clamp((y - 0.5f * shown.log_depth.rows)
                                                           / (0.5f * shown.log_depth.rows), -1.f, 1.f);
                                for (int x = 0; x < shown.log_depth.cols; ++x)
                                {
                                    if (not std::isfinite(row[x]) or sid[x] >= rc::depth::kMaxViews)
                                        continue;
                                    const float s = sw > 0 ? std::clamp((x - (sid[x] + 0.5f) * sw)
                                                                        / (0.5f * sw), -1.f, 1.f) : 0.f;
                                    // Position in "view units": f = 0 at view 0's centre, 1 at view 1's…
                                    const float f  = sw > 0 ? (static_cast<float>(x) / sw - 0.5f) : 0.f;
                                    const int   v0 = static_cast<int>(std::floor(f));
                                    const float w  = f - static_cast<float>(v0);
                                    const int   i0 = ((v0 % nv) + nv) % nv;          // wraps the seam
                                    const int   i1 = (i0 + 1) % nv;
                                    const float bb = (1.f - w) * depth_anchor_.b[i0] + w * depth_anchor_.b[i1];
                                    row[x] = depth_fit_map_.apply_with(row[x], bb, s, t);
                                }
                            }
                        }
                    }

                    // ── The ROOM BELIEF's own prediction, in the panorama ────────────────────────
                    // Mode 1 (PREDICTED) is pure geometry: cast the polygon+floor+ceiling envelope
                    // through every pixel. Mode 2 (DIFF) subtracts the monocular model from it, so
                    // that one — and only that one — needs Depth ON and a metric (anchored) model.
                    // ★In mode 2 NEITHER side is a measurement: it is one prediction against another.
                    // Furniture is bright by construction (the model sees a table, the envelope has no
                    // furniture), so the reading that matters is disagreement on the SHELL — a wall or
                    // the floor lighting up means the two models genuinely differ about the room.
                    bool drew_room = false;
                    cv::Mat room_log_range;          // metric ln(range); feeds the hover readout
                    if (ricoh_room_mode_ != 0 and dp and ricoh_scene_.valid)
                    {
                        // CameraAPI carries the node's projection model AND its azimuth
                        // sign/offset, so ray_from_pixel() is correct for whichever model the
                        // ricoh actually declares — cylindrical here, not equirectangular.
                        if (not ricoh_camera_api_)
                            if (const auto rn = G->get_node("ricoh"); rn.has_value())
                                ricoh_camera_api_ = G->get_camera_api(rn.value());
                        if (ricoh_camera_api_ and not ricoh_scene_.poly_x.empty())
                        {
                            // Sized off the PANORAMA, not off the model's map: with Depth OFF there is
                            // no map to size against, and the overlay has to land on the frame anyway.
                            const cv::Mat env = rc::depth::room_envelope_range_equirect(
                                *ricoh_camera_api_, pano.cols, pano.rows,
                                ricoh_scene_.room_T_ricoh, ricoh_scene_.poly_x,
                                ricoh_scene_.poly_y, ricoh_scene_.room_height,
                                params.ZED_ROOM_DEPTH_DECIMATE);
                            if (not env.empty() and env.size() == pano.size())
                            {
                                // ln(range) of the envelope, built ONCE for BOTH modes: mode 1 draws
                                // it, mode 2 subtracts the model from it, and either way it is what
                                // the hover reports as "room". Metric by construction — it is a
                                // ray-cast through a believed geometry, not a scaled network output.
                                room_log_range = cv::Mat(env.size(), CV_32FC1,
                                    cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
                                for (int y = 0; y < env.rows; ++y)
                                {
                                    const float* e = env.ptr<float>(y);
                                    auto* d = room_log_range.ptr<float>(y);
                                    for (int x = 0; x < env.cols; ++x)
                                        if (std::isfinite(e[x]) and e[x] > 0.f) d[x] = std::log(e[x]);
                                }
                                if (ricoh_room_mode_ == 2)
                                {
                                    if (have_model and corrected and shown.log_depth.size() == env.size())
                                    {
                                        cv::Mat model_m(shown.log_depth.size(), CV_32FC1,
                                                        cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
                                        for (int y = 0; y < model_m.rows; ++y)
                                        {
                                            const float* l = shown.log_depth.ptr<float>(y);
                                            auto* d = model_m.ptr<float>(y);
                                            for (int x = 0; x < model_m.cols; ++x)
                                                if (std::isfinite(l[x])) d[x] = std::exp(l[x]);
                                        }
                                        const cv::Mat diff = rc::depth::compose_difference(
                                            model_m, env, params.ZED_DEPTH_DIFF_SPAN_M);
                                        if (not diff.empty())
                                        {
                                            // ★addWeighted writes IN PLACE, and `pano` is still a shallow
                                            // handle on the WORKER's frame unless an overlay above cloned
                                            // it — blending into it would scribble on another thread's
                                            // buffer (CLAUDE.md: deep-copy every cv::Mat at a boundary).
                                            if (pano.data == rres->frame.rgbd.bgr.data)
                                                pano = pano.clone();
                                            cv::addWeighted(pano, 1.0 - params.RICOH_DEPTH_OVERLAY_ALPHA,
                                                            diff, params.RICOH_DEPTH_OVERLAY_ALPHA, 0.0, pano);
                                            drew_room = true;
                                        }
                                    }
                                }
                                else
                                {
                                    rc::depth::DepthMap envmap;
                                    envmap.log_depth = room_log_range;
                                    envmap.strip_id = cv::Mat(env.size(), CV_8UC1, cv::Scalar(0));
                                    envmap.n_strips = 1; envmap.band_y0 = 0; envmap.band_y1 = env.rows;
                                    pano = dp->compose(pano, envmap, params.RICOH_DEPTH_OVERLAY_ALPHA,
                                                       true, params.RICOH_DEPTH_METRIC_LO_M,
                                                       params.RICOH_DEPTH_METRIC_HI_M);
                                    drew_room = true;
                                }
                            }
                        }
                    }
                    if (not drew_room and have_model)
                        pano = dp->compose(pano, shown, params.RICOH_DEPTH_OVERLAY_ALPHA, corrected,
                                           params.RICOH_DEPTH_METRIC_LO_M, params.RICOH_DEPTH_METRIC_HI_M);

                    // Hover readout = WHATEVER IS ON SCREEN, in metres, or nothing. The room envelope
                    // is metric by construction; the monocular field is metric only once anchored
                    // (uncorrected it is a per-view relative scale, and printing "3.4 m" off it would
                    // be inventing a measurement). ★Must be reached on EVERY path, including the ones
                    // that draw no depth at all — it is what CLEARS the readout. Left unreached (as it
                    // was, being nested under the Depth toggle) the viewer kept its last cloned field
                    // and went on reporting frozen metres with Depth OFF.
                    // ★The MODEL side is offered only when ANCHORED. Handing over an uncorrected
                    // field would print a per-view relative number next to a genuine metre reading
                    // and invite subtracting one from the other — the Δ line would then be pure
                    // fiction. Unanchored, the hover shows the room alone and says nothing about
                    // the model, which is exactly what is known.
                    const cv::Mat model_readout = (have_model and corrected) ? shown.log_depth : cv::Mat();
                    ricoh_viewer_->set_depth_readout(room_log_range, model_readout,
                                                     not room_log_range.empty() or not model_readout.empty());

                    if (rres->masks and not rres->masks->empty())
                    {
                        // pano is BGR; SegStage::compose is base-order-preserving and the popup viewer
                        // converts BGR→RGB on display, so keep everything BGR. (The old BGR2RGB pre-swap
                        // here double-converted once the producer began tagging its true RGB order.)
                        auto* seg = dynamic_cast<rc::SegStage*>(ricoh_worker_->stage("seg"));
                        ricoh_viewer_->update_image(seg ? seg->compose(pano, *rres->masks) : pano);
                    }
                    else
                        ricoh_viewer_->update_image(pano);
                }
            }
        }
        else
        {
            // Always drain the ricoh plane (cheap; poll_ricoh decodes only when the popup is visible)
            // so its stream rate is known for the HUD even with the popup closed.
            scene_processor->poll_ricoh();
            if (popup_visible)
            {
                const cv::Mat pano = scene_processor->ricoh_bgr_copy();
                if (not pano.empty())
                    ricoh_viewer_->update_image(pano);
            }
        }

        if (const std::uint64_t rs = scene_processor->ricoh_last_stamp_ms(); rs > 0)
            stream_mon_.tick("rgb360", rs);
        ricoh_ms = perf_ms(perf_ricoh0, std::chrono::steady_clock::now());
    }


    // Input-stream health: report rates every 5 s and flag stalls. Runs here (GUI tick,
    // independent of the perception frame path) so a FULLY stalled RGB stream — which
    // makes compute() early-return before it could log — is still detected.
    if (auto rep = stream_mon_.maybe_report(5.0, 1.5); rep.has_value())
    {
        if (rep->any_stall)
            qWarning().noquote() << "[streams]" << QString::fromStdString(rep->summary)
                                 << "— producer stall? (see presence/emergency)";
        else
            qInfo().noquote() << "[streams]" << QString::fromStdString(rep->summary);
    }

    // Publish-hold watchdog (debounced, hysteresis). Enter hold when RGB has been stale for
    // HOLD_ENTER_S; resume only after RGB stays fresh for HOLD_RECOVER_S — so the first frame
    // after a stall doesn't immediately un-hold (anti-flap). Reacts to producer-side stalls that
    // presence can't see (the peer is alive, only its data-plane stopped).
    if (const double rgb_idle = stream_mon_.idle_s("rgb"); rgb_idle >= 0.0)
    {
        using namespace std::chrono;
        const bool stale = rgb_idle > params.HOLD_ENTER_S;
        const auto now = steady_clock::now();
        if (not perception_hold_)
        {
            if (stale)
            {
                perception_hold_ = true;
                rgb_fresh_since_ = {};
                qWarning().noquote() << QString::asprintf(
                    "[hold] RGB stale %.1fs — perception publish HELD (not emitting stale masks/skeletons)", rgb_idle);
            }
        }
        else if (not stale)   // candidate recovery: require sustained freshness
        {
            if (rgb_fresh_since_.time_since_epoch().count() == 0) rgb_fresh_since_ = now;
            else if (duration<double>(now - rgb_fresh_since_).count() >= params.HOLD_RECOVER_S)
            {
                perception_hold_ = false;
                qInfo().noquote() << "[hold] RGB recovered — resuming perception publish";
            }
        }
        else                  // relapsed while recovering → reset the recovery timer
            rgb_fresh_since_ = {};
    }

    // [perf] one row per render tick (~20 Hz nominal) → etc/viewer_perf_render.csv. Lets a spike in
    // viewer_perf.csv's compute_ms be cross-checked against a concurrent render-tick stall (same
    // Qt thread) — e.g. Ricoh decode/repaint — instead of only blaming scene/yolo/pose.
    if (params.PERF_LOG)
    {
        const double render_tick_ms = perf_ms(perf_t0, std::chrono::steady_clock::now());
        static const auto perf_log_start = std::chrono::steady_clock::now();
        static std::ofstream perf_csv("etc/viewer_perf_render.csv", std::ios::trunc);
        static const bool perf_hdr = [](std::ofstream& f)
            { f << "t_ms,render_tick_ms,ricoh_ms\n"; return true; }(perf_csv);
        (void)perf_hdr;
        const long long t_ms = static_cast<long long>(perf_ms(perf_log_start, std::chrono::steady_clock::now()));
        perf_csv << t_ms << ',' << render_tick_ms << ',' << ricoh_ms << '\n';
        perf_csv.flush();
    }
}

void SpecificWorker::compute()
{
    if (!scene_processor)
        return;

    // [perf] per-frame timing → etc/viewer_perf.csv when Voxel.perf_log=true.
    const auto perf_t0 = std::chrono::steady_clock::now();
    auto perf_ms = [](auto a, auto b){ return std::chrono::duration<double, std::milli>(b - a).count(); };

    const auto perf_scene0 = std::chrono::steady_clock::now();
    const auto frame = process_scene_frame(fps_counter_);
    const double scene_ms = perf_ms(perf_scene0, std::chrono::steady_clock::now());

    // No RGBD frame this cycle (sensor not ready / gated transforms). Everything below dereferences
    // `frame`, so we MUST stop here — a fall-through hit frame->rgbd on an empty optional → SEGV.
    if (!frame.has_value())
        return;

    // Cache the scene the Ricoh-360 overlays need (they render in on_render_tick, a different timer).
    // Only when a toggle is on, to avoid per-frame copies otherwise. Same thread → no lock.
    // ★RoomΔ is a THIRD consumer of this cache, not just Models/Lidar: it ray-casts poly_x/poly_y/
    // room_height through room_T_ricoh. Leaving it out of this condition left `valid` false whenever
    // Models and Lidar were both off, so RoomΔ silently fell through to the plain depth ramp — the
    // same class of bug as nesting it under the Depth toggle. Every field it reads is gathered here.
    if (ricoh_model_overlay_enabled_ or ricoh_lidar_overlay_enabled_ or ricoh_room_mode_ != 0)
    {
        ricoh_scene_.boxes        = frame->graph_object_boxes;
        ricoh_scene_.room_T_ricoh = frame->room_T_ricoh;
        ricoh_scene_.poly_x       = frame->room_poly_x;
        ricoh_scene_.poly_y       = frame->room_poly_y;
        ricoh_scene_.lidar_room   = ricoh_lidar_overlay_enabled_ ? frame->lidar_points_room : std::vector<Eigen::Vector3f>{};
        ricoh_scene_.room_height  = frame->room_height;
        ricoh_scene_.valid        = frame->ricoh_valid;   // only draw once the ricoh pose is known
    }

    // ZED perception runs on its own PULL worker: ZedSource drains the aligned RGBD + resolves room<-zed
    // ON THAT THREAD, so the whole RGBD path is off the main tick. Consume the newest completed bundle;
    // nullopt = no new ZED frame this cycle — the scene/viewer/lidar already updated above, so return.
    std::optional<rc::PerceptionResult> zed_res;
    if (zed_worker_)
        zed_res = zed_worker_->take_result();
    if (!zed_res)
    {
        fps_counter_.print("[Compute]", 3000);
        return;
    }
    stream_mon_.tick("rgb", zed_res->frame.stamp);   // input-rate telemetry / stall detection

    static const std::vector<SegDetection> kNoSegDetections;
    static const std::vector<rc::human_pose::PoseDetection> kNoPoses;
    const std::vector<SegDetection>& detections = zed_res->masks ? *zed_res->masks : kNoSegDetections;
    const std::vector<rc::human_pose::PoseDetection>& poses = zed_res->poses ? *zed_res->poses : kNoPoses;
    const bool   poses_fresh = zed_res->poses_fresh;   // pose model actually ran → gate skeleton publish
    const double yolo_ms = 0.0;   // inference is off-thread now (per-stage timing in etc/viewer_perf_zed_worker.csv)
    const double pose_ms = 0.0;

    // Semantic segmentation now runs as a SemanticStage in the ZED worker (decimated, gated by the
    // "Semantic" toggle → the stage's enabled flag). The dense class map arrives in the bundle; reach the
    // stage only for the viewer compose passthrough.
    auto* sem_stage = dynamic_cast<rc::SemanticStage*>(zed_worker_ ? zed_worker_->stage("semantic") : nullptr);
    const rc::semantic::SemanticMap* sem_map = (zed_res and zed_res->semantic) ? &*zed_res->semantic : nullptr;

    if (yolo_viewer_ and yolo_window_ and yolo_window_->isVisible())
    {
        // rgbd.bgr is the order seg/sam2 consume (the producer tags its true RGB order and
        // MediaPlaneSource converts RGB→BGR). The ZED popup overlays + YoloViewer work in RGB, so convert
        // once here — this also detaches viewer_rgb from the shared worker frame.
        cv::Mat viewer_rgb;
        cv::cvtColor(zed_res->frame.rgbd.bgr, viewer_rgb, cv::COLOR_BGR2RGB);
        // Dense semantic class-map underlay (blended) first, then skeletons + seg masks on top.
        if (sem_stage and sem_stage->processor() and semantic_overlay_enabled_
            and sem_map and not sem_map->labels.empty())
            viewer_rgb = sem_stage->processor()->compose_semantic_canvas(viewer_rgb, *sem_map);
        // Draw the detected skeletons (green bones, red joints, orange bbox) under the seg overlay.
        // The pose model lives in the ZED worker's PoseStage now; reach it for the compose passthrough.
        if (auto* ps = dynamic_cast<rc::PoseStage*>(zed_worker_ ? zed_worker_->stage("pose") : nullptr);
            ps and ps->processor() and not poses.empty())
            viewer_rgb = ps->processor()->compose_pose_canvas(viewer_rgb, poses);
        // Project every graph model instance (table/bottle/chair/obstacle BBs) onto the image when the
        // ZED-window "Models" toggle is on. Independent, self-contained overlay (model_projection_overlay).
        if (model_overlay_enabled_ and model_overlay_)
        {
            if (viewer_rgb.data == zed_res->frame.rgbd.bgr.data)
                viewer_rgb = viewer_rgb.clone();   // don't scribble on the shared source frame
            // boxes/polygon from the main-thread scene gather; room<-zed from the worker's own frame.
            model_overlay_->draw(viewer_rgb, frame->graph_object_boxes, zed_res->frame.room_T_sensor,
                                 frame->room_poly_x, frame->room_poly_y, frame->room_height);
        }
        // SAM2-refined masks (magenta) when the ZED-window "SAM2" toggle is on — for eyeballing SAM2
        // vs the raw YOLO masks. compose_canvas returns a fresh Mat, so the shared frame is untouched.
        if (sam2_overlay_enabled_ and zed_res->refined_masks and not zed_res->refined_masks->empty())
            if (auto* s2 = dynamic_cast<rc::Sam2Stage*>(zed_worker_ ? zed_worker_->stage("sam2") : nullptr);
                s2 and s2->refiner())
                viewer_rgb = s2->refiner()->compose_canvas(viewer_rgb, *zed_res->refined_masks, /*is_bgr=*/true);
        // ── What the ROOM BELIEF predicts, vs what the ZED MEASURES ──────────────────────────────
        // Ray-cast the room envelope room_concept believes in (floor z=0, ceiling z=room_height, one
        // vertical wall per polygon edge) through every pixel, and difference it against the camera's
        // own depth. This is the belief's per-pixel residual — the quantity the free energy is built
        // from — rendered as an image. No neural model is involved and none is needed: the ZED already
        // measures depth, and the room belief already predicts it.
        //
        // ★A well-fitted ROOM goes black; furniture and people stay bright, because the envelope
        // deliberately excludes them. Bright blobs = things the room does not explain (correct and
        // expected). Bright STRUCTURE — a whole wall tilting, the floor shading off with range — is
        // the interesting case: that is room-model error, not clutter.
        if (zed_depth_mode_ != 0 and zed_res->frame.rgbd.width > 0
            and not zed_res->frame.rgbd.depth.empty() and not frame->room_poly_x.empty())
        {
            const cv::Mat measured(zed_res->frame.rgbd.height, zed_res->frame.rgbd.width, CV_32FC1,
                                   const_cast<float*>(zed_res->frame.rgbd.depth.data()));
            const cv::Mat predicted = rc::depth::room_envelope_depth(
                zed_res->frame.rgbd.width, zed_res->frame.rgbd.height,
                zed_res->frame.rgbd.focal_x, zed_res->frame.rgbd.focal_y,
                zed_res->frame.room_T_sensor, frame->room_poly_x, frame->room_poly_y,
                frame->room_height, params.ZED_ROOM_DEPTH_DECIMATE);

            if (not predicted.empty() and predicted.size() == measured.size())
            {
                if (zed_depth_mode_ == 2)
                {
                    const cv::Mat diff = rc::depth::compose_difference(predicted, measured,
                                                                       params.ZED_DEPTH_DIFF_SPAN_M);
                    if (not diff.empty())
                        cv::addWeighted(viewer_rgb, 1.0 - params.RICOH_DEPTH_OVERLAY_ALPHA,
                                        diff, params.RICOH_DEPTH_OVERLAY_ALPHA, 0.0, viewer_rgb);
                }
                else
                {
                    // Mode 1: the prediction itself, on the same fixed metric ramp as everything else.
                    rc::depth::DepthMap shown;
                    shown.log_depth = cv::Mat(predicted.size(), CV_32FC1,
                                              cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
                    for (int y = 0; y < predicted.rows; ++y)
                    {
                        const float* p = predicted.ptr<float>(y);
                        auto* d = shown.log_depth.ptr<float>(y);
                        for (int x = 0; x < predicted.cols; ++x)
                            if (std::isfinite(p[x]) and p[x] > 0.f) d[x] = std::log(p[x]);
                    }
                    shown.strip_id = cv::Mat(predicted.size(), CV_8UC1, cv::Scalar(0));
                    shown.n_strips = 1; shown.band_y0 = 0; shown.band_y1 = predicted.rows;
                    if (ricoh_worker_)
                        if (auto* dp = dynamic_cast<rc::DepthStage*>(ricoh_worker_->stage("depth")))
                            viewer_rgb = dp->compose(viewer_rgb, shown, params.RICOH_DEPTH_OVERLAY_ALPHA,
                                                     /*metric=*/true, params.RICOH_DEPTH_METRIC_LO_M,
                                                     params.RICOH_DEPTH_METRIC_HI_M);
                }
                if (yolo_viewer_)
                    yolo_viewer_->update_depth(measured, predicted, true);

                // Score the belief where the envelope is the ONLY thing that should be visible: report
                // the median |residual| over pixels where the two agree to within a metre, which
                // excludes furniture without needing to segment it.
                static int zc = 0;
                if ((zc++ % 40) == 0)
                {
                    std::vector<float> e;
                    e.reserve(4096);
                    for (int y = 0; y < predicted.rows; y += 4)
                        for (int x = 0; x < predicted.cols; x += 4)
                        {
                            const float p = predicted.at<float>(y, x), z = measured.at<float>(y, x);
                            if (std::isfinite(p) and std::isfinite(z) and z > 0.f and std::abs(p - z) < 1.0f)
                                e.push_back(std::abs(p - z));
                        }
                    if (e.size() > 64)
                    {
                        std::nth_element(e.begin(), e.begin() + e.size() / 2, e.end());
                        std::println("[room-depth] envelope vs ZED: median |Δ| = {:.3f} m over {} "
                                     "surface px (pixels beyond 1 m are furniture, not counted)",
                                     e[e.size() / 2], e.size());
                    }
                }
            }
        }
        else if (yolo_viewer_ and zed_depth_mode_ == 0)
            yolo_viewer_->update_depth({}, {}, false);

        // The "YOLO" toggle in the ZED window gates only the seg-detection overlay (masks/bboxes);
        // the semantic underlay, skeletons and model projections above are independent.
        static const std::vector<SegDetection> kNoDetections;
        yolo_viewer_->update_frame(viewer_rgb, yolo_overlay_enabled_ ? detections : kNoDetections);
        // Feed the dense label map for the hover readout (cleared internally when not active).
        if (sem_map)
            yolo_viewer_->update_semantic(sem_map->labels, semantic_overlay_enabled_);
        // Size the RGB window to the image once (only when no saved geometry was restored).
        if (yolo_window_needs_image_size_ and yolo_window_ != nullptr and not viewer_rgb.empty())
        {
            yolo_window_->resize(viewer_rgb.cols, viewer_rgb.rows);
            yolo_window_needs_image_size_ = false;
        }
    }

    // Publish only when the input is fresh. On a producer stall the watchdog holds publishing so we
    // never push stale-but-confident masks/skeletons that downstream would fuse as current. (On a
    // full stall compute() already early-returns on the dedup above; the gate also covers the
    // recovery window, where a first fresh frame arrives before freshness is confirmed sustained.)
    double publish_ms = 0.0;
    double skel_ms = 0.0;
    if (not perception_hold_)
    {
        const auto perf_pub0 = std::chrono::steady_clock::now();

        // RGB-360 (ricoh) peripheral detections → shared "masks" node (Part B, RICOH_360_PERIPHERAL_DETECTION.md).
        // SPLIT across threads: the RicohYoloWorker resolves each detection's room-frame BEARING on its own
        // thread (own InnerEigenAPI, room<-ricoh at the panorama stamp — ricoh_yolo_worker.cpp::compute_bearings);
        // the LiDAR DEPTH-FILL runs HERE on the main thread, where the lidar cloud lives, augmenting those
        // bearings in place (has_depth=1 + support points + mask_depth_var). d.mask + bearing are a 1:1 snapshot.
        // Monocular depth vs LiDAR, per gnomonic view. Independent of publish_masks — this is the
        // measurement that says whether the depth model is seeing the room, and it must be available
        // whether or not the 360 masks are being published.
        if (params.RICOH_DEPTH_ENABLED and params.RICOH_DEPTH_LIDAR_DIAG and ricoh_worker_
            and frame->ricoh_valid and not frame->lidar_points_room.empty())
            if (auto rres = ricoh_worker_->latest_result(); rres and rres->depth and not rres->depth->empty())
            {
                // The room belief AS IT WAS WHEN THIS PANORAMA WAS TAKEN. Recorded per admitted frame
                // in the sidecar, because the dataset's (rx,ry,rtheta) cannot express a polygon, a
                // ceiling height, or the camera's own z — and the offline enrichment pass needs all
                // three to ray-cast the envelope. Without it, a rebuild months later has no choice but
                // to assume the room never changed.
                rc::depth::RoomGeometry rg;
                rg.room       = frame->room_name;
                rg.poly_x     = frame->room_poly_x;
                rg.poly_y     = frame->room_poly_y;
                rg.height     = frame->room_height;
                rg.room_T_cam = frame->room_T_ricoh;
                rg.cam_z      = static_cast<float>(frame->room_T_ricoh.translation().z());
                log_ricoh_depth_lidar_correlation(*rres->depth, frame->lidar_points_room,
                                                  frame->lidar_plane_id, frame->room_T_ricoh,
                                                  rres->frame.rgbd.bgr, rres->frame.stamp, rg);
            }

        std::vector<BearingDetection> bearing_dets;
        if (params.RICOH_PUBLISH_MASKS and ricoh_worker_)
        {
            std::vector<SegDetection> ricoh_dets;
            if (auto rres = ricoh_worker_->latest_result(); rres and rres->masks and rres->bearings)
            {
                ricoh_dets   = std::move(*rres->masks);       // 1:1 with bearings (BearingStage order)
                bearing_dets = std::move(*rres->bearings);
            }

            if (params.RICOH_MASK_DEPTH and frame->ricoh_valid
                and not bearing_dets.empty() and bearing_dets.size() == ricoh_dets.size()
                and not frame->lidar_points_room.empty())
            {
                if (not ricoh_camera_api_)
                    if (const auto rn = G->get_node("ricoh"); rn.has_value())
                        ricoh_camera_api_ = G->get_camera_api(rn.value());
                if (ricoh_camera_api_)
                {
                    // Reproject the lidar into the panorama ONCE (helios-only by default: co-located with the
                    // ricoh ⇒ least occlusion parallax).
                    std::vector<Eigen::Vector3f> cloud;
                    cloud.reserve(frame->lidar_points_room.size());
                    for (std::size_t i = 0; i < frame->lidar_points_room.size(); ++i)
                    {
                        if (params.RICOH_MASK_DEPTH_HELIOS_ONLY
                            and i < frame->lidar_plane_id.size() and frame->lidar_plane_id[i] != 0)
                            continue;   // skip bpearl (plane 1)
                        cloud.push_back(frame->lidar_points_room[i]);
                    }
                    const auto lidar_pano =
                        rc::depth::reproject_cloud(cloud, *ricoh_camera_api_, frame->room_T_ricoh.inverse());

                    for (std::size_t di = 0; di < ricoh_dets.size() and not lidar_pano.empty(); ++di)
                    {
                        const auto& d = ricoh_dets[di];
                        if (d.mask.empty()) continue;
                        // TWO gates keep FLOOR out of the object's support points:
                        //   (1) SILHOUETTE: select by the segmentation MASK (tight on the object), not the bbox.
                        //   (2) FOREGROUND: anchor on the NEAREST silhouette surface (low range percentile) and
                        //       keep only returns within RICOH_MASK_FG_BAND_M — occluded floor-behind is farther
                        //       along the ray → dropped. Physical object-depth prior (see [[no-threshold-patches]]).
                        std::vector<rc::depth::ProjectedPoint> hits;
                        for (const auto& p : lidar_pano)
                        {
                            const int ui = static_cast<int>(std::lround(p.u));
                            const int vi = static_cast<int>(std::lround(p.v));
                            if (ui < 0 or ui >= d.mask.cols or vi < 0 or vi >= d.mask.rows)
                                continue;
                            if (d.mask.at<std::uint8_t>(vi, ui) < 127)   // outside the object silhouette → skip
                                continue;
                            hits.push_back(p);
                        }
                        if (hits.empty()) continue;
                        std::vector<float> rr;
                        rr.reserve(hits.size());
                        for (const auto& h : hits) rr.push_back(h.range);
                        const std::size_t k = rr.size() / 10;                // ~10th-percentile = near surface
                        std::nth_element(rr.begin(), rr.begin() + k, rr.end());
                        const float r_near = rr[k];
                        std::vector<rc::depth::ProjectedPoint> fg;
                        fg.reserve(hits.size());
                        for (const auto& h : hits)
                            if (h.range - r_near <= params.RICOH_MASK_FG_BAND_M)   // drop the far (floor) tail
                                fg.push_back(h);
                        if (const auto md = rc::depth::score_mask_depth(fg); md.has_depth)
                        {
                            BearingDetection& bd = bearing_dets[di];   // augment the worker-computed bearing
                            bd.has_lidar_depth = true;
                            bd.range_var = md.range_var;
                            bd.support_room.reserve(fg.size());
                            Eigen::Vector3f sum = Eigen::Vector3f::Zero();
                            for (const auto& h : fg)   // ricoh-frame hit → room frame
                            {
                                const Eigen::Vector3f pr = (frame->room_T_ricoh * h.xyz_cam.cast<double>()).cast<float>();
                                bd.support_room.push_back(pr);
                                sum += pr;
                            }
                            bd.centroid_room = sum / static_cast<float>(fg.size());
                        }
                    }
                }
            }
        }

        // Masks: publish the WORKER's bundle (its own frame's rgbd/transform/stamp) so the seg masks are
        // deprojected against their own depth. Ricoh bearings ride along in the same node update.
        // When Sam2.publish_refined is on, swap each detection's YOLO mask for its SAM2-refined counterpart
        // (matched by label+bbox) so the concept fitters get the tighter support-point clouds. Only on
        // FRESH SAM2 cycles — a held-last refined mask against the current frame would be mildly stale.
        const std::vector<SegDetection>* masks_src = zed_res->masks ? &*zed_res->masks : &kNoSegDetections;
        std::vector<SegDetection> masks_refined;
        if (params.SAM2_PUBLISH_REFINED and zed_res->masks and zed_res->refined_fresh
            and zed_res->refined_masks and not zed_res->refined_masks->empty())
        {
            masks_refined = *zed_res->masks;
            for (auto& d : masks_refined)
                for (const auto& r : *zed_res->refined_masks)
                    if (not r.mask.empty() and r.label == d.label and r.bbox == d.bbox)
                    { d.mask = r.mask; break; }
            masks_src = &masks_refined;
        }
        graph_publisher_->publish(zed_res->frame.rgbd, zed_res->frame.room_T_sensor,
                                  *masks_src, zed_res->frame.stamp, bearing_dets);
        publish_ms = perf_ms(perf_pub0, std::chrono::steady_clock::now());

        // Human-pose branch: BODY_18 skeletons (camera frame) on the 'skeleton' node for human_concept.
        // Only on cycles the worker's PoseStage actually ran the model (fresh), and published against the
        // bundle's own frame so the skeleton depth matches.
        if (poses_fresh and zed_res)
        {
            const auto perf_skel0 = std::chrono::steady_clock::now();
            graph_publisher_->publish_skeletons(zed_res->frame.rgbd, poses, zed_res->frame.stamp);
            skel_ms = perf_ms(perf_skel0, std::chrono::steady_clock::now());
        }

        // Dense semantic label map → 'semantic' node under 'zed'. LOW-FREQUENCY (large blob): rate-capped,
        // and only on cycles the model actually ran (semantic_fresh). Gated by Semantic.publish_node.
        if (params.SEMANTIC_PUBLISH_NODE and zed_res->semantic_fresh
            and zed_res->semantic and not zed_res->semantic->labels.empty())
        {
            using namespace std::chrono;
            const auto now = steady_clock::now();
            if (duration<double>(now - last_semantic_pub_).count() >= params.SEMANTIC_PUBLISH_MIN_INTERVAL_S)
            {
                graph_publisher_->publish_semantic(zed_res->semantic->labels, zed_res->frame.stamp);
                last_semantic_pub_ = now;
            }
        }
    }

    const double compute_ms = perf_ms(perf_t0, std::chrono::steady_clock::now());

    // (The homeostatic rate regulator lived here and was REMOVED — see specificworker.h for why.
    //  Its one sound output, the source cadence, is now StreamRateMonitor::feed_hz(name).)

    // [perf] one row per real (fresh-frame) cycle: total + the model-inference breakdown.
    if (params.PERF_LOG)
    {
        static const auto perf_log_start = std::chrono::steady_clock::now();
        static std::ofstream perf_csv("etc/viewer_perf.csv", std::ios::trunc);
        static const bool perf_hdr = [](std::ofstream& f)
            { f << "t_ms,compute_ms,scene_ms,yolo_ms,pose_ms,publish_ms,skel_ms,n_det,rgbd_ts_ms\n"; return true; }(perf_csv);
        (void)perf_hdr;
        const long long t_ms = static_cast<long long>(perf_ms(perf_log_start, std::chrono::steady_clock::now()));
        perf_csv << t_ms << ',' << compute_ms << ',' << scene_ms << ',' << yolo_ms << ',' << pose_ms
                 << ',' << publish_ms << ',' << skel_ms
                 << ',' << detections.size() << ',' << zed_res->frame.stamp << '\n';
        perf_csv.flush();
    }

    fps_counter_.print("[Compute]", 3000);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
std::optional<SpecificWorker::SceneFrame> SpecificWorker::process_scene_frame(FPSCounter& compute_fps)
{
    scene_processor->check_input_stream_startup_status();
    const auto [room_name, robot_name] = scene_processor->get_room_robot_names_for_compute();

    // Diagnostic: throttled (every ~2s) report of which gate drops the frame. Verbose-only.
    static auto last_gate_report = std::chrono::steady_clock::now();
    const auto gate_log = [&](const char* gate)
    {
        if (not verbose_debug_)
            return;
        const auto now = std::chrono::steady_clock::now();
        if (now - last_gate_report >= std::chrono::seconds(2))
        {
            std::println("[SceneGate] dropped at '{}' (room='{}' robot='{}')", gate, room_name, robot_name);
            last_gate_report = now;
        }
    };

    // RGBD is drained on the ZED worker thread now (ZedSource). This gather provides SCENE CONTEXT only
    // (lidar + transforms + viewer + boxes + ricoh pose), at the LATEST pose — no RGBD gate/stamp, and it
    // runs every cycle (decoupled from RGBD arrival).
    const std::uint64_t frame_ts_ms = 0;   // latest (viewer/boxes don't need frame-precision here)

    if (!scene_processor->ensure_room_and_robot_ready(compute_fps, room_name, robot_name))
        { gate_log("ensure_room_and_robot_ready"); return std::nullopt; }

    const auto room_T_robot = scene_processor->get_room_robot_transform(compute_fps, room_name, robot_name, frame_ts_ms);
    if (!room_T_robot.has_value())
        { gate_log("get_room_robot_transform"); return std::nullopt; }

    scene_processor->mark_room_rt_ready();
    const auto graph_object_boxes = scene_processor->get_graph_object_boxes(room_name, frame_ts_ms);

    std::vector<Eigen::Vector3f> lidar_points_room;
    std::vector<std::uint8_t>    lidar_plane_id;   // per-point source plane (helios=0, bpearl=1) for colouring
    if (auto lidar_data = scene_processor->get_lidar3D(); lidar_data.has_value())
    {
        if (lidar_data->timestamp_ms > 0)
            stream_mon_.tick("lidar", lidar_data->timestamp_ms);   // input-rate telemetry / stall detection

        // [RT-clamp telemetry] Is the scan stamp AHEAD of the newest room<-robot RT block? Then
        // InterpolatedRT clamps at the leading edge (DSR does not extrapolate velocity) and the cloud
        // lags/steps until the next RT arrives — the shimmer source when room's RT rate < LiDAR rate.
        // Logs how often the scan outruns the RT and by how much, every 5 s. Verbose-only — the
        // per-cycle graph reads (get_node ×2 + get_edge_RT + attrib) are skipped entirely otherwise.
        if (verbose_debug_ && G && G->get_rt_api() && lidar_data->timestamp_ms > 0)
        {
            static std::uint64_t n_total = 0, n_clamp = 0, lag_sum = 0, lag_max = 0;
            static auto last_rep = std::chrono::steady_clock::now();
            const auto robot_n = G->get_node(robot_name);
            const auto room_n  = G->get_node(room_name);
            if (robot_n.has_value() && room_n.has_value())
                if (auto e = G->get_rt_api()->get_edge_RT(robot_n.value(), room_n->id()); e.has_value())
                    if (auto ts = G->get_attrib_by_name<rt_timestamps_att>(e.value()); ts.has_value())
                    {
                        std::uint64_t newest = 0;
                        for (const auto t : ts->get()) newest = std::max<std::uint64_t>(newest, static_cast<std::uint64_t>(t));
                        ++n_total;
                        if (newest > 0 && lidar_data->timestamp_ms > newest)
                        {
                            const std::uint64_t lag = lidar_data->timestamp_ms - newest;
                            ++n_clamp; lag_sum += lag; lag_max = std::max(lag_max, lag);
                        }
                    }
            const auto now = std::chrono::steady_clock::now();
            if (now - last_rep >= std::chrono::seconds(5))
            {
                if (n_total > 0)
                    std::println("[RT-clamp] {}/{} scans AHEAD of newest RT ({:.0f}%) — lag mean={}ms max={}ms "
                                 "(RT lagging LiDAR ⇒ interpolation clamps ⇒ cloud shimmer)",
                                 n_clamp, n_total, 100.0 * static_cast<double>(n_clamp) / static_cast<double>(n_total),
                                 n_clamp ? lag_sum / n_clamp : 0, lag_max);
                n_total = n_clamp = lag_sum = lag_max = 0;
                last_rep = now;
            }
        }

        Mat::RTMat room_T_robot_lidar = room_T_robot.value();
        if (inner_eigen_api != nullptr && lidar_data->timestamp_ms > 0)
        {
            const auto time_query = params.TRANSFORMS_INTERPOLATE_RT
                ? DSR::RT_API::TimeQuery::Interpolated
                : DSR::RT_API::TimeQuery::Nearest;
            if (auto interpolated = inner_eigen_api->get_transformation_matrix(
                    room_name,
                    robot_name,
                    lidar_data->timestamp_ms,
                    "RT",
                    time_query); interpolated.has_value())
            {
                room_T_robot_lidar = interpolated.value();
            }
        }

        // Efference-copy: extrapolate room<-robot FORWARD to the scan stamp over the RT-clamp gap using the
        // RT-edge velocities. NO-OP when the robot is STATIC (velocities≈0 ⇒ zero correction) — the ~90 ms
        // clamp only shifts the cloud when MOVING (displacement = velocity·lag, worst under rotation).
        rc::media::extrapolate_room_T_robot(G, room_name, robot_name, lidar_data->timestamp_ms,
                                            0.25f, room_T_robot_lidar, room_T_robot_lidar);

        // Ceiling crop: drop LiDAR returns at/above the room ceiling. Room-frame z is height above
        // the floor, so the ceiling is z == room_height (a float attribute the room agent writes on
        // the "room" node). Fall back to +inf (no crop) when the node/attr isn't present yet, so a
        // late-arriving room never silently blanks the cloud.
        float ceiling_z = std::numeric_limits<float>::infinity();
        if (const auto room_n = G->get_node(room_name); room_n.has_value())
            if (const auto h = G->get_attrib_by_name<room_height_att>(room_n.value()); h.has_value())
                ceiling_z = h.value();

        const std::size_t count = std::min({lidar_data->xs.size(), lidar_data->ys.size(), lidar_data->zs.size()});
        const std::size_t plane_count = std::min(count, lidar_data->plane_id.size());
        lidar_points_room.reserve(count);
        lidar_plane_id.reserve(plane_count);
        const Eigen::Matrix3f room_rotation = room_T_robot_lidar.linear().cast<float>();
        const Eigen::Vector3f room_translation = room_T_robot_lidar.translation().cast<float>();
        for (std::size_t i = 0; i < count; ++i)
        {
            const Eigen::Vector3f point_robot(lidar_data->xs[i], lidar_data->ys[i], lidar_data->zs[i]);
            const Eigen::Vector3f point_room = room_rotation * point_robot + room_translation;
            if (point_room.z() >= ceiling_z) continue;   // above the ceiling → drop
            lidar_points_room.emplace_back(point_room);
            if (i < plane_count) lidar_plane_id.push_back(lidar_data->plane_id[i]);
        }
    }


    // Gather the room floor polygon + ceiling height HERE (main thread) so the projection overlay
    // never traverses the graph itself — its per-frame reads raced DDS-thread residual inserts.
    std::vector<float> room_poly_x, room_poly_y;
    float room_height = 0.f;
    scene_processor->get_room_layout(room_poly_x, room_poly_y, room_height);

    // Ricoh pose in the room frame — single source of truth is the graph (body→ricoh RT + the ricoh
    // node's equirect intrinsics, applied by CameraAPI). Resolve the static robot→ricoh mount ONCE,
    // then compose room_T_ricoh = room_T_robot·robot_T_ricoh (pure math, no per-frame tree walk).
    // Left invalid (overlay no-ops) if the ricoh node isn't in the graph (e.g. a shadow model without it).
    if (not robot_T_ricoh_ and inner_eigen_api != nullptr)
        robot_T_ricoh_ = inner_eigen_api->get_transformation_matrix(robot_name, "ricoh", 0);
    Mat::RTMat room_T_ricoh = Mat::RTMat::Identity();
    const bool ricoh_valid = robot_T_ricoh_.has_value();
    if (ricoh_valid)
    {
        room_T_ricoh = room_T_robot.value() * robot_T_ricoh_.value();
        // Live azimuth fine-tune (DEGREES): extra yaw about the ricoh's up axis, on top of the graph
        // intrinsics. Post-multiply ⇒ rotates in the ricoh frame ⇒ shifts azimuth for BOTH overlay + bearing.
        if (params.RICOH_AZIMUTH_TUNE_DEG != 0.0f)
            room_T_ricoh.rotate(Eigen::AngleAxisd(params.RICOH_AZIMUTH_TUNE_DEG * M_PI / 180.0,
                                                  Eigen::Vector3d::UnitZ()));
    }

    return SceneFrame{room_T_robot.value(),
                      std::move(lidar_points_room),
                      std::move(lidar_plane_id),
                      graph_object_boxes,
                      room_name,
                      std::move(room_poly_x),
                      std::move(room_poly_y),
                      room_height,
                      room_T_ricoh,
                      ricoh_valid,
                      frame_ts_ms};
}

void SpecificWorker::log_ricoh_depth_lidar_correlation(const rc::depth::DepthMap& map,
                                                       const std::vector<Eigen::Vector3f>& lidar_room,
                                                       const std::vector<std::uint8_t>& plane_id,
                                                       const Mat::RTMat& room_T_ricoh,
                                                       const cv::Mat& panorama_bgr,
                                                       std::uint64_t frame_stamp_ms,
                                                       const rc::depth::RoomGeometry& room_geom)
{
    if (map.empty() or lidar_room.empty())
        return;
    if (not ricoh_camera_api_)
        if (const auto rn = G->get_node("ricoh"); rn.has_value())
            ricoh_camera_api_ = G->get_camera_api(rn.value());
    if (not ricoh_camera_api_)
        return;

    // HELIOS ONLY, for the same reason the mask depth-fill uses it: helios is co-located with the
    // ricoh, so its returns share the optical centre and carry almost no occlusion parallax. bpearl
    // sits low and looks down — its points would land in the panorama at pixels the ricoh sees from a
    // materially different viewpoint, which is measurement error dressed up as model error.
    std::vector<Eigen::Vector3f> cloud;
    cloud.reserve(lidar_room.size());
    for (std::size_t i = 0; i < lidar_room.size(); ++i)
        if (i >= plane_id.size() or plane_id[i] == 0)
            cloud.push_back(lidar_room[i]);
    if (cloud.empty())
        return;

    const auto proj = rc::depth::reproject_cloud(cloud, *ricoh_camera_api_, room_T_ricoh.inverse());
    if (proj.empty())
        return;

    // Per-view accumulators for Pearson r and the least-squares fit of log_range ≈ a·log_model + b.
    // BOTH sides are ray range from the optical centre — ProjectedPoint::range is ‖p_cam‖ and
    // zdepth_to_range makes the model emit range too — so they are directly comparable with no
    // further geometry. With that flag OFF the model is still z-depth and `a` would absorb part of
    // the mismatch, which is exactly why it defaults on.
    // sbase = Σ base(log_model, s) — the map's full nonlinear response minus the offset. The anchor
    // solves b = mean(log_range) − mean(base); with the piecewise-linear response `a*mean(lm)` is NO
    // LONGER the mean of the base, so it must be accumulated sample by sample.
    struct Acc { long n = 0; double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, sbase = 0; };
    std::vector<Acc> acc(static_cast<std::size_t>(std::max(1, map.n_strips)));

    // Dataset capture rides on the SAME pass — every pair the correlation consumes is exactly a
    // training sample, so collecting costs one push_back and cannot drift out of sync with what the
    // diagnostic reports.
    rc::depth::DepthFrame cap;
    // Collect at most ONCE per distinct panorama — see depth_collect_last_stamp_.
    const bool collecting = ricoh_depth_collect_enabled_
                        and (frame_stamp_ms == 0 or frame_stamp_ms != depth_collect_last_stamp_);
    const int  strip_w    = (map.n_strips > 0) ? map.log_depth.cols / map.n_strips : map.log_depth.cols;
    long       kept_stride_ = 0;

    for (const auto& p : proj)
    {
        const int u = static_cast<int>(std::lround(p.u));
        const int v = static_cast<int>(std::lround(p.v));
        if (u < 0 or u >= map.log_depth.cols or v < 0 or v >= map.log_depth.rows)
            continue;
        const float m = map.log_depth.at<float>(v, u);
        if (not std::isfinite(m) or p.range <= 1e-3f)
            continue;                                    // outside any view's frustum, or a bad return
        const unsigned char sid = map.strip_id.at<unsigned char>(v, u);
        if (sid >= acc.size())
            continue;
        const double x = m, y = std::log(static_cast<double>(p.range));
        const float s_in_view = strip_w > 0
            ? std::clamp((u - (sid + 0.5f) * strip_w) / (0.5f * strip_w), -1.f, 1.f) : 0.f;
        // ★t must come along now. It used to be irrelevant (base() ignored it), but once enrichment
        // gives ct*t^2 support the anchor's b = mean(log_range) − mean(base) is WRONG unless base()
        // sees the same t the map was fitted with. Same expression as the recorded sample below.
        const float t_in_pano = std::clamp((v - 0.5f * map.log_depth.rows) / (0.5f * map.log_depth.rows),
                                           -1.f, 1.f);
        Acc& a = acc[sid];
        ++a.n; a.sx += x; a.sy += y; a.sxx += x * x; a.syy += y * y; a.sxy += x * y;
        if (depth_fit_map_.valid)
            a.sbase += depth_fit_map_.base(m, s_in_view, t_in_pano);

        // STRIDE the recorded samples. A frame yields ~20k LiDAR hits and they are heavily redundant
        // (neighbouring returns on one surface say the same thing), so keeping all of them cost 635 MB
        // for 406 frames without adding information the fit can use. The correlation above still uses
        // every hit — only what is WRITTEN is thinned.
        if (collecting and (kept_stride_++ % std::max(1, params.RICOH_DEPTH_SAMPLE_STRIDE)) == 0)
        {
            rc::depth::DepthSample smp;
            smp.log_model = m;
            smp.log_range = static_cast<float>(y);
            smp.view      = sid;
            // Position WITHIN the view (s) and within the panorama height (t), both in [-1,1] — the
            // coordinates the map's quadratic terms are expressed in, stored now so the offline fit
            // never has to reconstruct the slicing geometry from a pixel index.
            smp.s = strip_w > 0 ? std::clamp((u - (sid + 0.5f) * strip_w) / (0.5f * strip_w), -1.f, 1.f) : 0.f;
            smp.t = std::clamp((v - 0.5f * map.log_depth.rows) / (0.5f * map.log_depth.rows), -1.f, 1.f);
            cap.samples.push_back(smp);
        }
    }

    if (collecting and not cap.samples.empty())
    {
        // ★The PANORAMA'S OWN capture stamp, not wall-clock-now. It is the join key between the CSV
        // rows and the saved image, so it must identify the frame the samples came from — a collection
        // timestamp would drift from it by however long the worker held the bundle.
        cap.stamp_ms = frame_stamp_ms != 0
            ? frame_stamp_ms
            : static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch()).count());
        const Eigen::Vector3d t = room_T_ricoh.translation();
        const Eigen::Matrix3d R = room_T_ricoh.linear();
        cap.rx = static_cast<float>(t.x());
        cap.ry = static_cast<float>(t.y());
        cap.rtheta = static_cast<float>(std::atan2(R(1, 0), R(0, 0)));
        const std::uint64_t stamp = cap.stamp_ms;

        // ── EPISTEMIC ADMISSION ──────────────────────────────────────────────────────────────────
        // Score the frame by the information it adds about the map parameters, not by how far the
        // robot moved. Two guards, in this order:
        //   1. CONSISTENCY first. D-optimality rates a corrupted frame (bad pose, bad registration,
        //      someone walking through) as maximally informative, because leverage and corruption
        //      look identical to it. So a frame must behave like the model expects BEFORE its
        //      novelty is believed. Skipped when no map exists yet — nothing to be consistent with.
        //   2. INFORMATION. Admit while the frame adds at least `min_gain_nats`. As Λ grows this
        //      naturally saturates: the 300th view of the same wall scores ~0 and is dropped, while a
        //      long sightline still scores high because the spline's ends are barely constrained.
        bool admit = true;
        double gain_nats = 0.0;
        if (params.RICOH_DEPTH_INFO_SELECT and depth_selector_.ready())
        {
            const double med = depth_selector_.residual(cap.samples, depth_fit_map_,
                                                        params.RICOH_DEPTH_N_STRIPS);
            if (med >= 0.0 and depth_fit_map_.valid
                and med > params.RICOH_DEPTH_SUSPECT_RESID_MULT * depth_fit_map_.resid_anchored)
            {
                admit = false;
                ++depth_collect_suspect_;
            }
            else
            {
                gain_nats = depth_selector_.gain(cap.samples, depth_fit_map_, params.RICOH_DEPTH_N_STRIPS);
                if (gain_nats < params.RICOH_DEPTH_MIN_GAIN_NATS)
                {
                    admit = false;
                    ++depth_collect_rejected_;
                }
            }
        }

        // Pose novelty stays as a cheap PRE-filter only when the information rule is off; with it on,
        // information decides and the distance test would only veto informative views from a spot the
        // robot happens to have visited.
        const float move_gate = params.RICOH_DEPTH_INFO_SELECT ? 0.0f : 0.10f;
        const float turn_gate = params.RICOH_DEPTH_INFO_SELECT ? 0.0f : 0.087f;
        if (admit and depth_dataset_.add_frame(std::move(cap), move_gate, turn_gate))
        {
            depth_collect_last_stamp_ = stamp;
            ++depth_collect_session_;
            if (params.RICOH_DEPTH_INFO_SELECT)
                depth_selector_.accumulate(depth_dataset_.last_frame_samples(), depth_fit_map_,
                                           params.RICOH_DEPTH_N_STRIPS);
            depth_dataset_.append_csv("etc/ricoh_depth_dataset.csv");   // survives a crash mid-drive

            // ★Record the ROOM BELIEF this frame was captured under, in its own sidecar. The dataset
            // row cannot hold it: rows are per-SAMPLE and a polygon is per-FRAME, and repeating it on
            // every one of ~1200 rows per frame would bloat a 5 MB file for nothing. Written for the
            // same admitted frames as the .jpg, keyed on the same stamp, so the three join. Whether
            // the offline pass has to guess the geometry is decided HERE, at collection time.
            if (not rc::depth::append_room_geometry(kDepthRoomsCsv, stamp, room_geom)
                and room_geom.valid())
                std::println(stderr, "[depth-collect] could not write {}", kDepthRoomsCsv);

            // Save the panorama for the OFFLINE enrichment pass (YOLO-sem → ceiling/floor/wall
            // segmentation → synthetic samples from the room envelope). Only for ADMITTED frames, so
            // the image count matches the CSV's frame count exactly. imwrite reads the Mat and never
            // writes it, so sharing the worker's buffer here is safe; the refcount keeps it alive.
            if (params.RICOH_DEPTH_SAVE_FRAMES and not panorama_bgr.empty())
            {
                std::error_code ec;
                std::filesystem::create_directories(params.RICOH_DEPTH_FRAMES_DIR, ec);
                const auto path = std::format("{}/{}.jpg", params.RICOH_DEPTH_FRAMES_DIR, stamp);
                const std::vector<int> jpeg{cv::IMWRITE_JPEG_QUALITY,
                                            std::clamp(params.RICOH_DEPTH_FRAME_QUALITY, 50, 100)};
                if (not cv::imwrite(path, panorama_bgr, jpeg))
                    std::println(stderr, "[depth-collect] could not write {}", path);
            }

            if (depth_collect_session_ % 10 == 0)
                std::println("[depth-collect] {} kept ({:.2f} nats last) | {} uninformative, {} suspect"
                             " | {} samples total{}",
                             depth_collect_session_, gain_nats, depth_collect_rejected_,
                             depth_collect_suspect_, depth_dataset_.sample_count(),
                             params.RICOH_DEPTH_SAVE_FRAMES
                                 ? std::format(" (+ {}/)", params.RICOH_DEPTH_FRAMES_DIR) : "");
        }
    }

    // ── Per-frame scale anchor ───────────────────────────────────────────────────────────────────
    // With `a` fixed at its DATASET value, the offset that best explains this frame's LiDAR is just
    // b = mean(log_range) - a*mean(log_model) over that view's hits — one division, no solve. This is
    // the quantity that must be re-estimated every frame (the model re-draws its scale per image);
    // the dataset can never supply it. Views without enough hits keep the map's fallback b, so a view
    // the LiDAR cannot reach degrades to the pooled estimate instead of to nonsense.
    if (depth_fit_map_.valid)
    {
        for (std::size_t s = 0; s < acc.size() and s < rc::depth::kMaxViews; ++s)
        {
            const Acc& A = acc[s];
            depth_anchor_.n[s]  = A.n;
            depth_anchor_.ok[s] = (A.n >= 64);
            depth_anchor_.b[s]  = depth_anchor_.ok[s]
                ? static_cast<float>((A.sy - A.sbase) / A.n)     // mean(log_range) − mean(base)
                : depth_fit_map_.b[s];
        }
    }

    if (not ricoh_depth_csv_open_attempted_)
    {
        ricoh_depth_csv_open_attempted_ = true;
        ricoh_depth_csv_.open("etc/ricoh_depth_lidar.csv", std::ios::trunc);
        if (ricoh_depth_csv_.is_open())
            ricoh_depth_csv_ << "cycle,view,n,pearson_r,a,b,resid_rms_log,median_model_m,median_lidar_m\n";
    }

    static long cycle = 0;
    ++cycle;
    double r_sum = 0.0;
    long   r_cnt = 0, n_tot = 0;
    std::string line;
    for (std::size_t s = 0; s < acc.size(); ++s)
    {
        const Acc& a = acc[s];
        n_tot += a.n;
        // 24 is not a threshold on the physics — it is where a correlation coefficient stops carrying
        // information. Views below it are simply not summarised; `n` is in the CSV either way.
        if (a.n < 24)
            continue;
        const double n   = static_cast<double>(a.n);
        const double cov = a.sxy / n - (a.sx / n) * (a.sy / n);
        const double vx  = a.sxx / n - (a.sx / n) * (a.sx / n);
        const double vy  = a.syy / n - (a.sy / n) * (a.sy / n);
        if (vx <= 1e-12 or vy <= 1e-12)
            continue;
        const double r     = cov / std::sqrt(vx * vy);
        const double A     = cov / vx;                                  // 1.0 ⇒ the model's scale is already right
        const double B     = a.sy / n - A * (a.sx / n);                 // exp(B) is the metric scale factor
        const double resid = std::sqrt(std::max(0.0, vy - A * cov));    // RMS of log_range about the fit
        r_sum += r; ++r_cnt;
        if (ricoh_depth_csv_.is_open())
            ricoh_depth_csv_ << cycle << ',' << s << ',' << a.n << ',' << r << ',' << A << ',' << B
                             << ',' << resid << ',' << std::exp(a.sx / n) << ',' << std::exp(a.sy / n) << '\n';
        line += std::format(" v{}:r={:+.2f}(n={},a={:.2f})", s, r, a.n, A);
    }
    if (ricoh_depth_csv_.is_open())
        ricoh_depth_csv_.flush();

    // ── How good is it, in units a person can act on? ────────────────────────────────────────────
    // `a` and `r` describe the FIT; they say nothing about how wrong a metre reading is. These are the
    // standard monocular-depth measures, computed on the corrected field against the LiDAR:
    //   AbsRel = median |d − d*| / d*      "typical error, as a fraction of true range"
    //   med_err = median |d − d*|          the same thing in metres
    //   δ1     = fraction within 1.25x     "how often is it in the right ballpark"
    // ★IN-SAMPLE: b was solved from these very points, so this is the optimistic bound — it measures
    // the model + map, NOT generalisation to a pose the anchor did not see. Reported as such.
    if (depth_fit_map_.valid and depth_anchor_.any())
    {
        std::vector<float> rel, abs_m;
        rel.reserve(proj.size());
        abs_m.reserve(proj.size());
        int within125 = 0, within156 = 0;
        for (const auto& p : proj)
        {
            const int u = static_cast<int>(std::lround(p.u));
            const int v = static_cast<int>(std::lround(p.v));
            if (u < 0 or u >= map.log_depth.cols or v < 0 or v >= map.log_depth.rows) continue;
            const float m = map.log_depth.at<float>(v, u);
            if (not std::isfinite(m) or p.range <= 1e-3f) continue;
            const unsigned char sid = map.strip_id.at<unsigned char>(v, u);
            if (sid >= rc::depth::kMaxViews or not depth_anchor_.ok[sid]) continue;
            const float s = strip_w > 0 ? std::clamp((u - (sid + 0.5f) * strip_w) / (0.5f * strip_w), -1.f, 1.f) : 0.f;
            const float t = std::clamp((v - 0.5f * map.log_depth.rows) / (0.5f * map.log_depth.rows), -1.f, 1.f);
            const float d = std::exp(depth_fit_map_.apply_with(m, depth_anchor_.b[sid], s, t));
            const float e = std::abs(d - p.range);
            rel.push_back(e / p.range);
            abs_m.push_back(e);
            const float ratio = std::max(d / p.range, p.range / d);
            if (ratio < 1.25f) ++within125;
            if (ratio < 1.5625f) ++within156;
        }
        if (rel.size() > 32)
        {
            const auto med = [](std::vector<float>& v)
            { std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end()); return v[v.size() / 2]; };
            const double n = static_cast<double>(rel.size());
            static int qc = 0;
            if ((qc++ % 40) == 0)
                std::println("[depth-quality] typical error {:.0f}% ({:.2f} m) | within 1.25x: {:.0f}% | "
                             "within 1.56x: {:.0f}% | n={} (IN-SAMPLE — the anchor was fitted on these points)",
                             100.0 * med(rel), med(abs_m), 100.0 * within125 / n, 100.0 * within156 / n,
                             rel.size());
        }
    }

    static int lc = 0;
    if ((lc++ % 40) == 0)
    {
        if (r_cnt > 0)
            std::println("[depth-lidar] mean r={:+.3f} over {} views ({} lidar hits in-frustum){}",
                         r_sum / static_cast<double>(r_cnt), r_cnt, n_tot, line);
        else
            std::println("[depth-lidar] no view had enough in-frustum lidar hits ({} total) — the "
                         "gnomonic band is only ±37°, so a helios sweep at robot height barely enters it",
                         n_tot);
    }
}

void SpecificWorker::sync_ricoh_depth_stage()
{
    // The stage runs if EITHER the overlay wants to draw it OR collection needs its output. See the
    // declaration for why tying it to the overlay toggle alone was a bug.
    if (not ricoh_worker_)
        return;
    if (auto* s = ricoh_worker_->stage("depth"))
        s->set_enabled(ricoh_depth_overlay_enabled_ or ricoh_depth_collect_enabled_);
}

void SpecificWorker::arm_depth_collection(bool on)
{
    ricoh_depth_collect_enabled_ = on;
    sync_ricoh_depth_stage();          // pressing Collect turns the model on by itself

    if (not on)
    {
        std::println("[depth-collect] stopped — {} frames added, {} rejected as uninformative, "
                     "{} rejected as inconsistent", depth_collect_session_,
                     depth_collect_rejected_, depth_collect_suspect_);
        return;
    }

    depth_collect_session_  = 0;
    depth_collect_rejected_ = 0;
    depth_collect_suspect_  = 0;
    if (not params.RICOH_DEPTH_INFO_SELECT)
        return;

    // ★SEED THE SELECTOR FROM WHAT IS ALREADY ON DISK. Without this the first frames of every session
    // look informative merely because the selector has forgotten the existing dataset, and the set
    // would fill with duplicates of what it already contains.
    rc::depth::DepthDataset prior;
    depth_selector_.reset(rc::depth::map_n_params(depth_fit_map_, params.RICOH_DEPTH_N_STRIPS));
    if (prior.load_csv("etc/ricoh_depth_dataset.csv") and prior.frame_count() > 0)
    {
        for (std::size_t i = 0; i < prior.frame_count(); ++i)
            depth_selector_.accumulate(prior.frame_samples(i), depth_fit_map_,
                                       params.RICOH_DEPTH_N_STRIPS);
        std::println("[depth-collect] armed — selector seeded from {} existing frames "
                     "({} samples); log det Λ = {:.1f}",
                     prior.frame_count(), prior.sample_count(), depth_selector_.logdet());
    }
    else
        std::println("[depth-collect] armed — no existing dataset, starting from the prior");
    std::println("[depth-collect] admitting frames with gain ≥ {:.2f} nats "
                 "(consistency guard: median resid ≤ {:.1f}× the map's {:.3f})",
                 params.RICOH_DEPTH_MIN_GAIN_NATS, params.RICOH_DEPTH_SUSPECT_RESID_MULT,
                 depth_fit_map_.valid ? depth_fit_map_.resid_anchored : 0.f);
}

std::string SpecificWorker::rebuild_depth_fit_map()
{
    // Re-read the WHOLE accumulated set, not just this session — the point of appending across runs
    // is that every drive improves the same map.
    const std::string path = "etc/ricoh_depth_dataset.csv";
    rc::depth::DepthDataset all;
    if (not all.load_csv(path))
    {
        // ALWAYS name the resolved path: every data path in this component is relative to the CWD, so
        // an instance launched from the wrong directory looks for etc/etc/... and finds nothing. That
        // is exactly how two instances produced mutually contradictory maps.
        const auto abs = std::filesystem::absolute(path).string();
        std::println("[depth-map] cannot open {} — check the working directory", abs);
        return std::format("cannot open {}", abs);
    }
    if (all.frame_count() == 0)
    {
        std::println("[depth-map] {} has no frames — collect some first", path);
        return "dataset empty";
    }
    const std::size_t before = all.frame_count();
    // ★Do NOT dedup a set built by the information selector. Pose novelty was the ADMISSION rule back
    // when frames were taken on distance; with info_select on, every stored frame already earned its
    // place by sharpening the parameters, and a second pass keyed on distance actively undoes that —
    // measured 2026-08-06: it discarded 7 of 11 information-selected frames, 64% of the set, because
    // the robot had not moved 10 cm between two genuinely informative views. Two admission criteria
    // that disagree is one too many; the information one is strictly better informed.
    const std::size_t removed = params.RICOH_DEPTH_INFO_SELECT ? 0 : all.dedup();
    const int n_views = std::max(1, params.RICOH_DEPTH_N_STRIPS);
    std::println("[depth-map] loaded {} frames / {} samples from {}; {} duplicate poses dropped",
                 before, all.sample_count(), path, removed);
    const auto m = all.fit(n_views);
    if (not m.valid)
    {
        std::println("[depth-map] fit FAILED — {} frames / {} samples after dedup is not enough for "
                     "{} parameters", all.frame_count(), all.sample_count(), 1 + n_views + 2);
        return std::format("fit failed ({} frames)", all.frame_count());
    }

    depth_fit_map_ = m;
    m.save("etc/ricoh_depth_map.csv");
    // ★Report BOTH residuals. `resid_rms` is the pooled fit (b constant per view) and is the
    // PESSIMISTIC number — it charges the map for per-frame scale drift that the runtime's LiDAR
    // anchor cancels. `resid_anchored` is what the system actually delivers. Showing only the first
    // made the map look like it had not improved when the fix had already landed.
    std::println("[depth-map] {} frames ({} duplicate poses dropped), {} samples\n"
                 "            a={:.3f}  cs={:+.3f}  ct={:+.3f}  r={:+.3f}\n"
                 "            resid pooled  {:.3f} (±{:.0f}%)   <- b fixed per view; not how it runs\n"
                 "            resid ANCHORED {:.3f} (±{:.0f}%)  <- b re-solved per frame; THIS is live\n"
                 "            supported range {:.2f}..{:.2f} m — OUTSIDE THAT THE MAP EXTRAPOLATES",
                 before - removed, removed, m.n_samples, m.a, m.cs, m.ct, m.r,
                 m.resid_rms,      100.0 * (std::exp(m.resid_rms) - 1.0),
                 m.resid_anchored, 100.0 * (std::exp(m.resid_anchored) - 1.0),
                 m.range_lo, m.range_hi);
    std::string bs;
    for (int v = 0; v < m.n_views; ++v)
        bs += std::format(" b{}={:+.3f}", v, m.b[static_cast<std::size_t>(v)]);
    std::println("[depth-map]           {}", bs);
    return {};
}

std::string SpecificWorker::run_depth_enrichment(const std::function<void(const std::string&)>& status)
{
    // ── Everything that needs the graph happens HERE, on the main thread ─────────────────────────
    if (G == nullptr)
        return "no graph";
    const auto ricoh_node = G->get_node("ricoh");
    if (not ricoh_node.has_value())
        return "no 'ricoh' node in the graph — the envelope ray-cast needs its projection model";
    auto cam = G->get_camera_api(ricoh_node.value());   // OUR OWN instance, never ricoh_camera_api_
    if (not cam)
        return "could not build a CameraAPI for the ricoh node";

    rc::depth::EnrichConfig cfg;
    cfg.dataset_csv  = "etc/ricoh_depth_dataset.csv";
    cfg.frames_dir   = params.RICOH_DEPTH_FRAMES_DIR;
    cfg.geometry_csv = kDepthRoomsCsv;
    cfg.out_dataset  = "etc/ricoh_depth_dataset_enriched.csv";
    cfg.n_views      = std::max(1, params.RICOH_DEPTH_N_STRIPS);
    // ★The model config MUST be the one the dataset was collected with. It is not assumed to be —
    // DatasetEnricher measures it (check_model_parity) and refuses to mix two different fields.
    cfg.depth_cfg.model_path = params.RICOH_DEPTH_MODEL_PATH;
    cfg.depth_cfg.input_size = params.RICOH_DEPTH_INPUT_SIZE;
    cfg.depth_cfg.use_gpu    = params.RICOH_DEPTH_USE_GPU;
    cfg.depth_cfg.use_trt    = params.RICOH_DEPTH_USE_TRT;
    cfg.depth360.n_strips           = params.RICOH_DEPTH_N_STRIPS;
    cfg.depth360.overlap_px         = params.RICOH_DEPTH_OVERLAP_PX;
    cfg.depth360.band_half_elev_deg = params.RICOH_DEPTH_BAND_HALF_ELEV_DEG;
    cfg.depth360.gnomonic           = params.RICOH_DEPTH_GNOMONIC;
    cfg.depth360.gnomonic_fov_deg   = params.RICOH_DEPTH_GNOMONIC_FOV_DEG;
    cfg.depth360.zdepth_to_range    = params.RICOH_DEPTH_ZDEPTH_TO_RANGE;
    cfg.sem_cfg.model_path  = params.SEMANTIC_SEG_MODEL_PATH;
    cfg.sem_cfg.conf_thresh = params.SEMANTIC_SEG_CONF_THRESH;
    cfg.sem_cfg.input_size  = params.SEMANTIC_SEG_INPUT_SIZE;
    cfg.sem_cfg.use_gpu     = params.SEMANTIC_SEG_USE_GPU;
    cfg.sem_cfg.use_trt     = params.SEMANTIC_SEG_USE_TRT;

    rc::depth::DatasetEnricher enricher(std::move(cfg));
    enricher.bind_camera(std::move(cam));

    // Per-frame geometry recorded at collection time (schema (a)) …
    std::map<std::uint64_t, rc::depth::RoomGeometry> per_frame;
    rc::depth::load_room_geometry(kDepthRoomsCsv, per_frame);
    // … and the fallback for frames older than the sidecar (schema (b)): the room AS THE GRAPH HOLDS
    // IT NOW. The enricher logs, by name, how many frames this assumption covers.
    rc::depth::RoomGeometry fb;
    if (scene_processor)
        scene_processor->get_room_layout(fb.poly_x, fb.poly_y, fb.height);
    if (const auto rn = G->get_nodes_by_type("room"); not rn.empty())
        fb.room = rn.front().name();
    if (not robot_T_ricoh_ and inner_eigen_api != nullptr and scene_processor)
    {
        // ts==0 ⇒ the InnerEigenAPI cache, which is safe only single-threaded per instance. This is
        // the MAIN thread and the same instance every other ts==0 call in this component uses, which
        // is exactly the condition CLAUDE.md requires. Never resolved from the enricher's thread.
        const auto [_room, robot] = scene_processor->get_room_robot_names_for_compute();
        robot_T_ricoh_ = inner_eigen_api->get_transformation_matrix(robot, "ricoh", 0);
    }
    fb.cam_z = ricoh_scene_.valid
        ? static_cast<float>(ricoh_scene_.room_T_ricoh.translation().z())
        : (robot_T_ricoh_ ? static_cast<float>(robot_T_ricoh_->translation().z()) : 0.f);
    enricher.set_geometry(std::move(per_frame), fb);

    // ── Run it OFF the GUI thread, and keep the GUI alive while it does ──────────────────────────
    if (not enricher.start())
        return "enrichment already running";
    while (enricher.running())
    {
        const auto p = enricher.progress();
        if (status)
            status(p.total > 0
                       ? std::format("{} {}/{}", rc::depth::DatasetEnricher::phase_name(p.phase),
                                     p.done, p.total)
                       : std::string(rc::depth::DatasetEnricher::phase_name(p.phase)));
        QCoreApplication::processEvents(QEventLoop::AllEvents, 50);
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
    }
    const auto rep = enricher.join();
    if (not rep.ok)
        return rep.error.empty() ? std::string("enrichment failed") : rep.error;

    // ★Hot-apply and persist ONLY the enriched map, but the LiDAR-only refit travelled back with it so
    // the A/B is in the log and nobody has to trust that "it got better".
    depth_fit_map_ = rep.map_enriched;
    rep.map_enriched.save("etc/ricoh_depth_map.csv");
    std::println("[depth-enrich] map applied: ct {} (dBIC {:+.1f}), supported range {:.2f}..{:.2f} m, "
                 "band t {:+.3f}..{:+.3f} — was {:.2f}..{:.2f} m / {:+.3f}..{:+.3f} on LiDAR alone",
                 rep.map_enriched.ct_active ? "ACTIVE" : "still off", rep.map_enriched.ct_delta_bic,
                 rep.map_enriched.range_lo, rep.map_enriched.range_hi,
                 rep.map_enriched.t_lo, rep.map_enriched.t_hi,
                 rep.map_measured.range_lo, rep.map_measured.range_hi,
                 rep.map_measured.t_lo, rep.map_measured.t_hi);
    return {};
}

void SpecificWorker::emergency()
{
    qInfo() << "Emergency worker";
}

void SpecificWorker::restore()
{
    qInfo() << "Restore worker";
}

int SpecificWorker::startup_check()
{
    qInfo() << "Startup check";
    QTimer::singleShot(200, QCoreApplication::instance(), SLOT(quit()));
    return 0;
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


