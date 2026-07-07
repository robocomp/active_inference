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
#ifdef emit
#undef emit
#endif
#include "voxel_opengl_viewer.h"
#include "yolo_viewer.h"
#include "image_popup_viewer.h"
#include "ricoh_yolo_worker.h"
#include <dsr/gui/viewers/graph_viewer/graph_viewer.h>
#include <QHBoxLayout>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <cmath>
#include <limits>
#include <print>
#include <sstream>
#include <unordered_map>

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
    ricoh_yolo_worker_.reset();   // stop+join BEFORE scene_processor is destroyed (holds a raw ptr to it)
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

    // --- YOLO ---
    yolo_processor = std::make_unique<YoloProcessor>();
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
    yolo_config.verbose_debug       = verbose_debug_;
    yolo_processor->configure(yolo_config);

    // --- Human pose (optional second model → BODY_18 skeletons on the 'skeleton' node) ---
    if (params.HUMAN_POSE_ENABLED)
    {
        try
        {
            yolo_human_processor = std::make_unique<rc::human_pose::YoloHumanProcessor>();
            rc::human_pose::YoloHumanProcessor::Config pose_config;
            pose_config.model_path    = params.HUMAN_POSE_MODEL_PATH;
            pose_config.conf_thresh   = params.HUMAN_POSE_CONF_THRESH;
            pose_config.iou_thresh    = params.HUMAN_POSE_IOU_THRESH;
            pose_config.input_size    = params.HUMAN_POSE_INPUT_SIZE;
            pose_config.use_gpu       = params.HUMAN_POSE_USE_GPU;
            pose_config.use_trt       = params.HUMAN_POSE_USE_TRT;
            pose_config.hold_ms       = params.HUMAN_POSE_HOLD_MS;
            pose_config.verbose_debug = verbose_debug_;
            yolo_human_processor->configure(pose_config);
        }
        catch (const std::exception& e)
        {
            qWarning() << "[HumanPose] disabled — failed to load model" << params.HUMAN_POSE_MODEL_PATH.c_str()
                       << ":" << e.what();
            yolo_human_processor.reset();
        }
    }

    // --- Semantic segmentation (optional dense ADE20K-150 per-pixel class map → viewer overlay) ---
    if (params.SEMANTIC_SEG_ENABLED)
    {
        try
        {
            yolo_semantic_processor = std::make_unique<rc::semantic::YoloSemanticProcessor>();
            rc::semantic::YoloSemanticProcessor::Config sem_config;
            sem_config.model_path    = params.SEMANTIC_SEG_MODEL_PATH;
            sem_config.conf_thresh   = params.SEMANTIC_SEG_CONF_THRESH;
            sem_config.input_size    = params.SEMANTIC_SEG_INPUT_SIZE;
            sem_config.use_gpu       = params.SEMANTIC_SEG_USE_GPU;
            sem_config.use_trt       = params.SEMANTIC_SEG_USE_TRT;
            sem_config.verbose_debug = verbose_debug_;
            yolo_semantic_processor->configure(sem_config);
        }
        catch (const std::exception& e)
        {
            qWarning() << "[Semantic] disabled — failed to load model" << params.SEMANTIC_SEG_MODEL_PATH.c_str()
                       << ":" << e.what();
            yolo_semantic_processor.reset();
        }
    }

    // Custom drawing windows (Voxel3D GL + YOLO raster), each in its own top-level window — see
    // specificworker_viewers.cpp. Both config-gated (Voxel.show_voxel_viewer / show_yolo_viewer).
    setup_custom_viewers();

    // --- Scene processor (DSR-only: no proxies) ---
    inner_eigen_api = G->get_inner_eigen_api();
    scene_processor = std::make_unique<SceneProcessor>(G);
    scene_processor->configure(inner_eigen_api.get(), voxel_viewer_gl.get(),
                               params.TRANSFORMS_INTERPOLATE_RT, verbose_debug_,
                               params.MASK_POSE_EXTRAPOLATE, params.MASK_POSE_EXTRAP_MAX_DT_S);
    scene_processor->init_media_plane(static_cast<std::uint32_t>(params.MEDIA_DOMAIN_ID),
                                      params.MEDIA_RGB_TOPIC, params.MEDIA_DEPTH_TOPIC);
    scene_processor->init_lidar_media_plane(static_cast<std::uint32_t>(params.MEDIA_DOMAIN_ID),
                                            params.MEDIA_LIDAR_TOPIC, params.LIDAR_USE_MEDIA);
    // ZED-image model-instance projection overlay (gated by the "Models" toggle in the ZED popup).
    // Does NO live graph traversal at draw time (caches the zed CameraAPI once); all geometry is fed
    // from the frame's already-gathered room←zed transform + room polygon.
    model_overlay_ = std::make_unique<rc::ModelProjectionOverlay>(G);
    // Ricoh-360 counterpart (equirectangular wireframe projection; pure math, no graph access).
    ricoh_model_overlay_ = std::make_unique<rc::RicohProjectionOverlay>();
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
        ricoh_yolo_worker_ = std::make_unique<rc::RicohYoloWorker>();
        rc::RicohYoloWorker::Config ricoh_worker_cfg;
        ricoh_worker_cfg.yolo_config.model_path        = params.YOLO_MODEL_PATH;
        ricoh_worker_cfg.yolo_config.conf_thresh       = params.YOLO_CONF_THRESH;
        ricoh_worker_cfg.yolo_config.iou_thresh        = params.YOLO_IOU_THRESH;
        ricoh_worker_cfg.yolo_config.input_size        = params.YOLO_INPUT_SIZE;
        ricoh_worker_cfg.yolo_config.use_gpu           = params.YOLO_USE_GPU;
        ricoh_worker_cfg.yolo_config.use_trt           = params.YOLO_USE_TRT;
        ricoh_worker_cfg.yolo_config.mask_erode_kernel = params.YOLO_MASK_ERODE_KERNEL;
        ricoh_worker_cfg.yolo_config.accepted_labels   = params.YOLO_ACCEPTED_LABELS;
        ricoh_worker_cfg.yolo_config.verbose_debug     = verbose_debug_;
        ricoh_worker_cfg.detect_config.n_strips        = params.RICOH_YOLO_N_STRIPS;
        ricoh_worker_cfg.detect_config.overlap_px      = params.RICOH_YOLO_STRIP_OVERLAP_PX;
        ricoh_worker_cfg.detect_config.merge_iou       = params.RICOH_YOLO_MERGE_IOU;
        ricoh_worker_cfg.target_period_ms              = params.RICOH_YOLO_THREAD_PERIOD_MS;
        ricoh_worker_cfg.perf_log                      = params.PERF_LOG;
        if (!ricoh_yolo_worker_->start(scene_processor.get(), ricoh_worker_cfg))
        {
            std::println("[Ricoh] 360-YOLO worker failed to start (media plane not ready?) — disabling");
            ricoh_yolo_worker_.reset();
        }
    }

    // Perception-rate regulator: floor = the user-configured pose decimation, ceiling =
    // RateRegulator.pose_decim_max; target = TARGET_HZ. Holds compute() near target by
    // raising pose decimation only when compute-bound AND the input feed can supply it.
    {
        rc::RateRegulatorConfig rc_cfg;
        rc_cfg.target_hz      = params.TARGET_HZ;
        rc_cfg.pose_decim_min = std::max(1, params.HUMAN_POSE_DECIMATION);
        rc_cfg.pose_decim_max = std::max(rc_cfg.pose_decim_min, params.POSE_DECIM_MAX);
        rate_reg_.configure(rc_cfg);
    }

    // All DSR semantic_grid exports (masks). Relayout is injected so the publisher stays
    // decoupled from the GUI (graph_viewers).
    graph_publisher_ = std::make_unique<GraphPublisher>(
        G, params, [this]() { trigger_graph_layout_twopi(); });

    graph_publisher_->cleanup_semantic_grid_nodes();

    // Decoupled render timer: only worth running when the 3D viewer exists. 50 ms (20 Hz) matches the
    // viewer's repaint throttle, giving a stable 20 Hz refresh without re-running perception faster.
    // Created stopped; started in Operating (so graph reads happen only after the join completes).
    if (voxel_viewer_gl)
    {
        render_timer_ = std::make_unique<QTimer>(this);
        render_timer_->setInterval(50);
        QObject::connect(render_timer_.get(), &QTimer::timeout, this, &SpecificWorker::on_render_tick);
    }

    presence_coordinator_.configure(configLoader, G, static_cast<std::uint32_t>(agent_id));
    presence_coordinator_.set_transition_hooks({
        .request_presence_ready = [this]() { Q_EMIT presenceReady(); },
        .request_presence_lost  = [this]() { Q_EMIT presenceLost(); },
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
            qInfo() << "[SM] -> Waiting";
            const auto missing = presence_coordinator_.missing_required_names();
            if (!missing.empty())
            {
                QString m;
                for (const auto &label : missing)
                    m += " " + QString::fromStdString(label);
                qInfo() << "  missing:" << m;
            }
        },
        .on_operating_enter = [this]()
        {
            qInfo() << "[SM] -> Operating: all required constraints satisfied";
            if (render_timer_)
                render_timer_->start();   // start fluid viewer refresh only once the graph is joined
        },
        .on_operating_loop = [this]()
        {
            compute();
            if (auto it = graph_viewers.find(""); it != graph_viewers.end() && it->second)
                it->second->set_external_fps(states.at("Operating")->getActualFps());
        },
        .on_degraded_enter = [this]()
        {
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

    scene_processor->refresh_viewer_robot_pose_latest();

    // Ricoh 360 popup: refresh the window when visible. When the 360-YOLO worker is running it OWNS
    // polling (its own thread, see ricoh_yolo_worker.h) — just read its thread-safe snapshot here, no
    // decode on this thread. Otherwise (Ricoh.yolo_enabled=false) poll directly, same as always.
    double ricoh_ms = 0.0;
    {
        const auto perf_ricoh0 = std::chrono::steady_clock::now();
        const bool popup_visible = ricoh_viewer_ and ricoh_window_ and ricoh_window_->isVisible();

        if (ricoh_yolo_worker_)
        {
            if (popup_visible)
            {
                cv::Mat pano = ricoh_yolo_worker_->latest_bgr();
                if (not pano.empty())
                {
                    // Project the DSR scene (model boxes + room floor/ceiling/walls) onto the panorama
                    // when the Ricoh "Models" toggle is on. Draw on a clone (BGR) so we never mutate
                    // the worker's cached frame; the popup viewer converts BGR→RGB on display.
                    if (ricoh_model_overlay_enabled_ and ricoh_model_overlay_ and ricoh_scene_.valid)
                    {
                        pano = pano.clone();
                        ricoh_model_overlay_->draw(pano, ricoh_scene_.boxes,
                                                   ricoh_scene_.ricoh_optical_center, ricoh_scene_.room_T_zed,
                                                   ricoh_scene_.poly_x, ricoh_scene_.poly_y, ricoh_scene_.room_height,
                                                   params.RICOH_AZIMUTH_SIGN, params.RICOH_AZIMUTH_OFFSET_RAD);
                    }
                    const auto dets = ricoh_yolo_worker_->latest_detections();
                    if (!dets.empty() and yolo_processor)
                    {
                        cv::Mat pano_rgb;
                        cv::cvtColor(pano, pano_rgb, cv::COLOR_BGR2RGB);
                        ricoh_viewer_->update_image(yolo_processor->compose_detection_canvas(pano_rgb, dets));
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

    // Feed the input-stream rates to the 3D viewer HUD (Render / RGB / RGB360).
    if (voxel_viewer_gl)
        voxel_viewer_gl->set_stream_fps(static_cast<float>(stream_mon_.rate_hz("rgb")),
                                        static_cast<float>(stream_mon_.rate_hz("rgb360")));

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

    // Cache the scene the Ricoh-360 overlay needs (it renders in on_render_tick, a different timer).
    // Only when its toggle is on, to avoid per-frame copies otherwise. Same thread → no lock.
    if (ricoh_model_overlay_enabled_)
    {
        ricoh_scene_.boxes                = frame->graph_object_boxes;
        ricoh_scene_.ricoh_optical_center = frame->ricoh_optical_center;
        ricoh_scene_.room_T_zed           = frame->room_T_zed;
        ricoh_scene_.poly_x               = frame->room_poly_x;
        ricoh_scene_.poly_y               = frame->room_poly_y;
        ricoh_scene_.room_height          = frame->room_height;
        ricoh_scene_.valid                = true;
    }

    // Follow the RGB stream's REAL rate: the media cache repeats the last frame when nothing new
    // arrived this cycle. Skip the RGB-derived work (YOLO, pose, viewer, mask publish) on stale
    // repeats so the displayed FPS and published masks track the camera's actual delivery rate. The
    // lidar/robot-pose viewer updates already ran in process_scene_frame and keep the compute cadence.
    if (frame->frame_ts_ms == last_rgb_ts_)
        return;
    last_rgb_ts_ = frame->frame_ts_ms;
    stream_mon_.tick("rgb", frame->frame_ts_ms);   // input-rate telemetry / stall detection

    // YOLO runs on the CLEAN frame (no tray black-out). The tray would be segmented as a phantom
    // object, but rather than corrupting the image we let it be detected and drop any detection that
    // overlaps the tray region in postprocess (see YoloProcessor::postprocess_yolo_detections).
    const auto perf_yolo0 = std::chrono::steady_clock::now();
    const auto detections = yolo_processor
        ? yolo_processor->detect_segmentation(frame->rgbd.rgb)
        : std::vector<SegDetection>{};
    const double yolo_ms = perf_ms(perf_yolo0, std::chrono::steady_clock::now());

    // Human-pose: the MODEL runs every HumanPose.decimation-th cycle. Default 1 (every frame) keeps the
    // skeleton glued to the moving person; on skipped cycles (decimation>1) the viewer redraws the LAST
    // cached detection, and detect_poses holds it across brief misses so the overlay never flickers.
    const int pose_decim = std::max(1, rate_reg_.pose_decimation());   // regulator-adapted (≥ HumanPose.decimation)
    const bool run_pose = yolo_human_processor and yolo_human_processor->ready()
                          and (pose_frame_counter_++ % pose_decim == 0);
    const auto perf_pose0 = std::chrono::steady_clock::now();
    if (run_pose)
        yolo_human_processor->detect_poses(frame->rgbd.rgb, frame->frame_ts_ms);   // refresh the cache (~6-7 Hz)
    const double pose_ms = run_pose ? perf_ms(perf_pose0, std::chrono::steady_clock::now()) : 0.0;
    static const std::vector<rc::human_pose::PoseDetection> kNoPoses;
    const auto& poses = (yolo_human_processor and yolo_human_processor->ready())
        ? yolo_human_processor->last_poses()
        : kNoPoses;

    // Semantic segmentation: dense ADE20K-150 per-pixel class map. Heavy, so run decimated (the last
    // map is reused for the overlay on skipped cycles). Feeds the viewer overlay only, and the whole
    // pass is gated by the YOLO-window "Semantic" toggle so deactivating it also stops the model work.
    if (yolo_semantic_processor and yolo_semantic_processor->ready() and semantic_overlay_enabled_)
    {
        const int decim = std::max(1, params.SEMANTIC_SEG_DECIMATION);
        if (semantic_frame_counter_++ % decim == 0)
            yolo_semantic_processor->segment(frame->rgbd.rgb);   // refresh the cached label map
    }

    // Ricoh 360 peripheral detection now runs on its own thread (rc::RicohYoloWorker, started in
    // initialize()) — nothing to do here. It's paced to its own target period, fully decoupled from
    // this cycle's budget; see etc/viewer_perf_ricoh_yolo.csv for its own timing.

    if (yolo_viewer_ and yolo_window_ and yolo_window_->isVisible())
    {
        cv::Mat viewer_rgb = frame->rgbd.rgb;   // clean frame (no tray black-out); update_frame clones
        // Dense semantic class-map underlay (blended) first, then skeletons + seg masks on top.
        if (yolo_semantic_processor and semantic_overlay_enabled_
            and not yolo_semantic_processor->last_map().labels.empty())
            viewer_rgb = yolo_semantic_processor->compose_semantic_canvas(viewer_rgb, yolo_semantic_processor->last_map());
        // Draw the detected skeletons (green bones, red joints, orange bbox) under the seg overlay.
        if (yolo_human_processor and not poses.empty())
            viewer_rgb = yolo_human_processor->compose_pose_canvas(viewer_rgb, poses);
        // Project every graph model instance (table/bottle/chair/obstacle BBs) onto the image when the
        // ZED-window "Models" toggle is on. Independent, self-contained overlay (model_projection_overlay).
        if (model_overlay_enabled_ and model_overlay_)
        {
            if (viewer_rgb.data == frame->rgbd.rgb.data)
                viewer_rgb = viewer_rgb.clone();   // don't scribble on the shared source frame
            model_overlay_->draw(viewer_rgb, frame->graph_object_boxes, frame->room_T_zed,
                                 frame->room_poly_x, frame->room_poly_y, frame->room_height);
        }
        // The "YOLO" toggle in the ZED window gates only the seg-detection overlay (masks/bboxes);
        // the semantic underlay, skeletons and model projections above are independent.
        static const std::vector<SegDetection> kNoDetections;
        yolo_viewer_->update_frame(viewer_rgb, yolo_overlay_enabled_ ? detections : kNoDetections);
        // Feed the dense label map for the hover readout (cleared internally when not active).
        if (yolo_semantic_processor)
            yolo_viewer_->update_semantic(yolo_semantic_processor->last_map().labels, semantic_overlay_enabled_);
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

        // RGB-360 (ricoh) peripheral detections → shared "masks" node as NO-DEPTH bearing slices (Part B,
        // RICOH_360_PERIPHERAL_DETECTION.md). The RicohYoloWorker (own thread) already produced panorama-
        // pixel SegDetections; convert each to a room-frame bearing here, where we hold the robot pose.
        std::vector<BearingDetection> bearing_dets;
        if (params.RICOH_PUBLISH_MASKS and ricoh_yolo_worker_)
        {
            const int pano_w = ricoh_yolo_worker_->latest_bgr().cols;
            if (pano_w > 0)
            {
                // Robot heading in the room from the zed pose (forward axis = +y in the deproject convention).
                const Eigen::Vector3d fwd = frame->room_T_zed.linear() * Eigen::Vector3d(0, 1, 0);
                const float robot_yaw = std::atan2(static_cast<float>(fwd.y()), static_cast<float>(fwd.x()));
                for (const auto& d : ricoh_yolo_worker_->latest_detections())
                {
                    const float col_c = static_cast<float>(d.bbox.x) + 0.5f * static_cast<float>(d.bbox.width);
                    // Panorama column → bearing relative to the panorama CENTRE (centre column ⇒ straight ahead),
                    // then apply the calibrated sign/zero knobs (mirror + mounting offset), then add the base
                    // heading for the absolute room bearing. Wrapped to (-π, π]. Convention calibrated 2026-07-04.
                    const float az_pano = 2.0f * static_cast<float>(M_PI) * (col_c / static_cast<float>(pano_w))
                                          - static_cast<float>(M_PI);
                    const float az_rel = std::atan2(std::sin(params.RICOH_AZIMUTH_OFFSET_RAD + params.RICOH_AZIMUTH_SIGN * az_pano),
                                                    std::cos(params.RICOH_AZIMUTH_OFFSET_RAD + params.RICOH_AZIMUTH_SIGN * az_pano));
                    const float az = std::atan2(std::sin(robot_yaw + az_rel), std::cos(robot_yaw + az_rel));
                    bearing_dets.push_back({d.label, static_cast<float>(d.class_id), d.confidence, az});
                }
            }
        }

        graph_publisher_->publish(frame->rgbd, frame->room_T_zed, detections, frame->frame_ts_ms, bearing_dets);
        publish_ms = perf_ms(perf_pub0, std::chrono::steady_clock::now());

        // Human-pose branch: BODY_18 skeletons (camera frame) on the 'skeleton' node for human_concept.
        // Only on cycles we actually ran the pose model (decimated above).
        if (run_pose)
        {
            const auto perf_skel0 = std::chrono::steady_clock::now();
            graph_publisher_->publish_skeletons(frame->rgbd, poses, frame->frame_ts_ms);
            skel_ms = perf_ms(perf_skel0, std::chrono::steady_clock::now());
        }
    }

    const double compute_ms = perf_ms(perf_t0, std::chrono::steady_clock::now());

    // Homeostatic regulator: feed it this cycle's cost + frame stamp; it adapts pose decimation
    // and exposes processed/feed Hz. Log on decimation changes, and warn (throttled) only when
    // WE are the limiter (compute-bound) — if the feed itself is slow, decimation can't help.
    rate_reg_.update(compute_ms, frame->frame_ts_ms);
    if (rate_reg_.changed())
        qInfo().noquote() << QString::asprintf(
            "[rate] pose decimation → %d | compute %.1f Hz, feed %.1f Hz, cycle %.1fms (yolo %.1f, pose %.1f)",
            rate_reg_.pose_decimation(), rate_reg_.processed_hz(), rate_reg_.feed_hz(),
            rate_reg_.compute_ms(), yolo_ms, pose_ms);
    else if (rate_reg_.below_target())
    {
        using namespace std::chrono;
        static steady_clock::time_point last_warn_tp{};
        const auto now_tp = steady_clock::now();
        if (duration_cast<seconds>(now_tp - last_warn_tp).count() >= 5)
        {
            last_warn_tp = now_tp;
            if (rate_reg_.feed_limited())
                qInfo().noquote() << QString::asprintf(
                    "[rate] %.1f Hz < target %.0f Hz but INPUT feed is only %.1f Hz — feed-limited, not compute; holding",
                    rate_reg_.processed_hz(), static_cast<double>(params.TARGET_HZ), rate_reg_.feed_hz());
            else if (rate_reg_.at_pose_cap())
                qWarning().noquote() << QString::asprintf(
                    "[rate] %.1f Hz < target %.0f Hz, pose decim at cap (%d), cycle %.1fms (yolo %.1f) "
                    "— drop to a lighter seg model / lower input res", rate_reg_.processed_hz(),
                    static_cast<double>(params.TARGET_HZ), rate_reg_.pose_decimation(), rate_reg_.compute_ms(), yolo_ms);
        }
    }

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
                 << ',' << detections.size() << ',' << frame->frame_ts_ms << '\n';
        perf_csv.flush();
    }

    fps_counter_.print("[Compute]", 3000);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
std::optional<SpecificWorker::SceneFrame> SpecificWorker::process_scene_frame(FPSCounter& compute_fps)
{
    scene_processor->check_input_stream_startup_status();
    const auto [room_name, robot_name] = scene_processor->get_room_robot_names_for_compute();

    // Diagnostic: throttled (every ~2s) report of which gate drops the frame.
    static auto last_gate_report = std::chrono::steady_clock::now();
    const auto gate_log = [&](const char* gate)
    {
        const auto now = std::chrono::steady_clock::now();
        if (now - last_gate_report >= std::chrono::seconds(2))
        {
            std::println("[SceneGate] dropped at '{}' (room='{}' robot='{}')", gate, room_name, robot_name);
            last_gate_report = now;
        }
    };

    const auto rgbd_opt = scene_processor->get_rgbd_frame_from_dsr();
    if (!rgbd_opt.has_value())
        { gate_log("get_rgbd_frame"); return std::nullopt; }
    const std::uint64_t frame_ts_ms = scene_processor->get_frame_timestamp_ms();

    if (!scene_processor->ensure_room_and_robot_ready(compute_fps, room_name, robot_name))
        { gate_log("ensure_room_and_robot_ready"); return std::nullopt; }

    const auto room_T_robot = scene_processor->get_room_robot_transform(compute_fps, room_name, robot_name, frame_ts_ms);
    if (!room_T_robot.has_value())
        { gate_log("get_room_robot_transform"); return std::nullopt; }
    const auto room_T_zed = scene_processor->get_room_zed_transform(compute_fps, robot_name, room_T_robot.value());
    if (!room_T_zed.has_value())
        { gate_log("get_room_zed_transform"); return std::nullopt; }

    scene_processor->log_room_robot_pose_periodic(room_T_robot.value());
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
        // Logs how often the scan outruns the RT and by how much, every 5 s.
        if (G && G->get_rt_api() && lidar_data->timestamp_ms > 0)
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

    scene_processor->update_viewer_robot_pose(room_T_robot.value());
    scene_processor->update_viewer_lidar_points(lidar_points_room, lidar_plane_id);   // reuse drained+posed sweep + plane tags
    scene_processor->update_viewer_graph_object_boxes(graph_object_boxes);
    scene_processor->update_viewer_object_meshes();
    scene_processor->update_viewer_person_skeletons();
    scene_processor->update_viewer_table_rfe_points();
    scene_processor->update_viewer_mask_points();
    scene_processor->update_room_polygon_periodic();

    // Gather the room floor polygon + ceiling height HERE (main thread) so the projection overlay
    // never traverses the graph itself — its per-frame reads raced DDS-thread residual inserts.
    std::vector<float> room_poly_x, room_poly_y;
    float room_height = 0.f;
    scene_processor->get_room_layout(room_poly_x, room_poly_y, room_height);

    // Ricoh optical centre in the room frame — single source of truth is the body→ricoh RT edge.
    // Resolve the static robot→ricoh mount ONCE, then compose room_T_ricoh = room_T_robot·robot_T_ricoh
    // (pure math, no per-frame tree walk). Fall back to robot xy + configured height if the ricoh node
    // isn't in the graph (e.g. a config that loads a shadow model without it).
    if (not robot_T_ricoh_ and inner_eigen_api != nullptr)
        robot_T_ricoh_ = inner_eigen_api->get_transformation_matrix(robot_name, "ricoh", 0);
    Eigen::Vector3d ricoh_optical_center =
        robot_T_ricoh_.has_value()
            ? Eigen::Vector3d(room_T_robot.value() * robot_T_ricoh_->translation())
            : Eigen::Vector3d(room_T_robot.value().translation().x(), room_T_robot.value().translation().y(),
                              static_cast<double>(params.RICOH_MOUNT_HEIGHT_M));

    return SceneFrame{rgbd_opt.value(),
                      room_T_robot.value(),
                      room_T_zed.value(),
                      std::move(lidar_points_room),
                      graph_object_boxes,
                      room_name,
                      std::move(room_poly_x),
                      std::move(room_poly_y),
                      room_height,
                      ricoh_optical_center,
                      frame_ts_ms};
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


