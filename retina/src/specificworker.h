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

#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#include <genericworker.h>
#include <fps/fps.h>
#include "../../common/agent_presence_coordinator/agent_presence_coordinator.h"

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

#include <Eigen/Core>

#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "depth_processor.h"   // rc::depth::DepthMap (ricoh 360 monocular depth diagnostic)
#include "depth_dataset.h"     // rc::depth::DepthDataset / DepthFitMap (LiDAR-anchored correction)
#include "depth_enrichment.h"  // rc::depth::RoomGeometry / DatasetEnricher (offline room-belief pass)
#include "rgbd_data.h"
#include "retina_params.h"
#include "stream_rate_monitor.h"
#include "model_projection_overlay.h"   // rc::ModelProjectionOverlay + GraphObjectBox (SceneFrame value member)
#include "ricoh_projection_overlay.h"   // rc::RicohProjectionOverlay (360 panorama counterpart)

#include <chrono>

class YoloProcessor;
class SceneProcessor;
class GraphPublisher;
struct SegDetection;
namespace rc::human_pose { class YoloHumanProcessor; }
namespace rc::semantic { class YoloSemanticProcessor; }

namespace rc { class YoloViewer; }
namespace rc { class ImagePopupViewer; }
namespace rc { class PerceptionWorker; }

class QLabel;   // 1 Hz mask-rate readout on the Voxel3D panel (defined in <QLabel>, only the .cpp needs it)

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

        void modify_node_slot(std::uint64_t, const std::string &type){};
        void modify_node_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names){};
        void modify_edge_slot(std::uint64_t from, std::uint64_t to,  const std::string &type){};
        void modify_edge_attrs_slot(std::uint64_t from, std::uint64_t to, const std::string &type, const std::vector<std::string>& att_names){};
        void del_edge_slot(std::uint64_t from, std::uint64_t to, const std::string &edge_tag){};
        void del_node_slot(std::uint64_t from){};

        void waiting_enter();
        void waiting_loop();
        void operating_enter();
        void operating_loop();
        void degraded_enter();
        void degraded_loop();

        void cleanup_owned_nodes();
        void request_shutdown();
        void on_optional_peer_lost(const std::string &name, std::uint32_t id);
        void on_optional_peer_ready(const std::string &name, std::uint32_t id);

        // Waiting→Operating extra gate: room_concept only publishes the 'room' node once its localization
        // is STABLE, so the node's mere presence IS the stability signal we wait on (on top of peer presence).
        bool room_node_present() const;

        // Machine-ingestible FSM telemetry: emit one structured JSON line per state transition to stdout
        // (parseable by log pipelines / netmon per-process capture) carrying a stable `reason` code so a
        // monitor can tell WHY we left Operating (required_peer_lost vs room_unstable).
        void log_sm_event(const std::string& from, const std::string& to,
                          const std::string& reason, const std::string& detail);

    private:
        RetinaParams params;   // loaded via load_retina_params() in initialize()

        struct SceneFrame   // scene CONTEXT gathered on the main thread (RGBD/seg live in the ZED worker now)
        {
            Mat::RTMat                  room_T_robot;
            std::vector<Eigen::Vector3f> lidar_points_room;
            std::vector<std::uint8_t>    lidar_plane_id;    // per-point source plane (helios=0, bpearl=1) — for the ricoh depth-fill
            std::vector<GraphObjectBox>  graph_object_boxes;   // model instances (room frame) — reused for the ZED-image projection overlay
            std::string                 room_name;         // live room node name (frame the boxes/polygon live in)
            std::vector<float>          room_poly_x;       // room floor polygon (room frame, m) — gathered here so the overlay never re-reads the graph
            std::vector<float>          room_poly_y;
            float                       room_height = 0.f;
            Mat::RTMat                  room_T_ricoh = Mat::RTMat::Identity();   // ricoh pose in room (room_T_robot·robot_T_ricoh)
            bool                        ricoh_valid = false;                     // true when the ricoh node/RT resolved
            std::uint64_t               frame_ts_ms = 0;   // capture stamp of rgbd/depth — published so consumers can pin pose to capture time
        };

        std::optional<SceneFrame> process_scene_frame(FPSCounter& compute_fps);
        void setup_custom_viewers();         // Voxel3D GL + YOLO windows (specificworker_viewers.cpp)
        void trigger_graph_layout_twopi();   // injected into GraphPublisher as the relayout callback

        AgentPresenceCoordinator presence_coordinator_;
        bool owned_nodes_cleaned_ = false;
        FPSCounter fps_counter_;

        bool startup_check_flag  = false;
        bool verbose_debug_      = false;
        std::atomic<bool> shutting_down_{false};
        // Human-pose model decimation counter (period read from params.HUMAN_POSE_DECIMATION; default 1).
        int pose_frame_counter_ = 0;
        // Semantic-seg decimation counter (period read from params.SEMANTIC_SEG_DECIMATION).
        int semantic_frame_counter_ = 0;
        std::chrono::steady_clock::time_point last_semantic_pub_{};   // rate cap for the semantic-node publish
        std::chrono::steady_clock::time_point last_waiting_log_{};    // throttle the "waiting for required peers" log
        // Reverse room-stability gate: debounce room-node disappearance while Operating so a transient
        // graph-resync gap (peer join/CRDT re-import) doesn't drop us; nullopt while the room is present.
        std::optional<std::chrono::steady_clock::time_point> room_absent_since_{};
        std::string current_sm_state_ = "Init";   // tracks the live FSM state; used as `from` in log_sm_event
        // Reason we last left Operating, set at the presenceLost site and consumed by the next Waiting-enter
        // event (Degraded is a transient pass-through). Empty ⇒ Waiting was entered without leaving Operating
        // (startup / awaiting dependencies).
        std::string pending_exit_reason_{};
        std::string pending_exit_detail_{};

        // ★THE HOMEOSTATIC RATE REGULATOR WAS REMOVED (2026-08-15), not disabled. It adapted human-pose
        // decimation to hold compute() near a target rate, and it had been inert since the perception
        // pipeline moved onto a worker thread: its pose_decimation() was read ONLY by its own log lines
        // (nothing applied it to PoseStage, which takes a fixed HUMAN_POSE_DECIMATION at construction),
        // and it was fed compute_ms from the MAIN thread, which no longer contains any inference — the
        // yolo_ms/pose_ms it printed were hardcoded 0.0 a few lines away. So it measured the wrong clock
        // and drove an actuator that was not connected, while printing a healthy-looking number.
        // It could not have helped here anyway: measured per-stage, pose is 4.2 ms of a 24 ms frame, so
        // its only actuator was 4% of the cost. The real limiters were SAM2 (52%) and a 15 ms poll
        // granularity, neither of which it could see. Its one sound output — the SOURCE cadence, derived
        // from frame stamps rather than from the mis-measured cost — now lives in StreamRateMonitor as
        // feed_hz(name), which gives it for every channel instead of just the ZED.

        // Input-stream rate telemetry + stall detection (RGB, lidar). No control — see
        // the presence/emergency handling for the react-to-producer-stall path.
        rc::StreamRateMonitor stream_mon_;

        // Publish-hold watchdog: when the RGB producer stalls, HOLD perception publishing (emit
        // nothing rather than stale masks/skeletons). Debounced (enter after HOLD_ENTER_S stale,
        // resume only after HOLD_RECOVER_S of sustained freshness). Layered beside presence, not
        // through it — presence sees peer death; this sees a live peer whose data went stale.
        bool perception_hold_ = false;
        std::chrono::steady_clock::time_point rgb_fresh_since_{};   // recovery timer (0 = not recovering)

        // Camera stamp of the last RGB frame we actually processed. The media cache repeats the last
        // frame when nothing new arrived, so we dedup on this to make the RGB pipeline (YOLO + viewer +
        // mask publish) follow the camera's REAL delivery rate instead of the fixed compute period.
        std::uint64_t last_rgb_ts_ = 0;

        std::shared_ptr<DSR::InnerEigenAPI> inner_eigen_api;

        std::unique_ptr<rc::PerceptionWorker> zed_worker_;   // ZED inference thread: [seg, pose, semantic] stages
        std::unique_ptr<SceneProcessor>    scene_processor;
        std::unique_ptr<GraphPublisher>    graph_publisher_;   // all DSR semantic_grid exports
        std::unique_ptr<rc::YoloViewer>        yolo_viewer_;
        std::unique_ptr<rc::ImagePopupViewer>  ricoh_viewer_;   // RGBD_360 panorama popup
        // Ricoh 360 peripheral YOLO, on its own thread (own model/session) — see ricoh_yolo_worker.h.
        // Started in initialize() only when params.RICOH_YOLO_ENABLED; stopped explicitly in
        // request_shutdown() BEFORE scene_processor is torn down (the worker holds a raw ptr to it).
        std::unique_ptr<rc::PerceptionWorker>  ricoh_worker_;   // ricoh 360 pull worker: [seg360, bearing] stages

        // Decoupled viewer-refresh timer (GUI thread). Pushes the LATEST robot pose to the 3D viewer
        // at a fluid cadence, independent of the ~7-10 Hz perception/camera pipeline, so robot motion
        // is smooth. Started in Operating, stopped in Degraded (graph access stays within Operating).
        std::unique_ptr<QTimer> render_timer_;
        // 1 Hz mask-throughput readout on the Voxel3D panel. Separate from render_timer_ (50 ms) so the
        // numbers are readable — a rate that repaints 20x/s cannot be read, and these are meant to be READ.
        std::unique_ptr<QTimer> rates_timer_;
        QLabel*                 rates_label_ = nullptr;   // owned by the panel (Qt parent), not by us
        // EXACT feed rates: previous arrival counts + the wall instant they were read, differentiated
        // once per readout tick. Counting beats inferring the cadence from processed-frame stamps.
        std::uint64_t feed_rgb_prev_ = 0, feed_ricoh_prev_ = 0, feed_lidar_prev_ = 0;
        double        feed_rgb_hz_ = 0.0, feed_ricoh_hz_ = 0.0, feed_lidar_hz_ = 0.0;
        std::chrono::steady_clock::time_point feed_last_tp_{};
        void on_render_tick();

        // Standalone top-level viewer windows (NOT hosted by the DSR graph viewer — that node-link
        // widget crashes in paintAndFlush under participant churn, so the graph view is disabled).
        // These are the custom widgets shown directly as their own windows; we only read/persist their
        // geometry. Lifetime: shown for the process lifetime (released at exit).
        QWidget* voxel3d_window_ = nullptr;
        QWidget* yolo_window_ = nullptr;
        QWidget* ricoh_window_ = nullptr;            // Ricoh 360 popup (hidden until the top-bar button toggles it)
        bool yolo_window_needs_image_size_ = false;  // size the RGB window to the image on first frame
        bool semantic_overlay_enabled_ = false;      // YOLO-window toggle: run + draw the semantic overlay (starts OFF)
        bool sam2_overlay_enabled_ = false;          // ZED-window "SAM2" toggle: run + draw SAM2-refined masks (starts OFF)
        bool yolo_overlay_enabled_ = true;           // ZED-window "YOLO" toggle: draw the seg detections (starts ON)
        bool model_overlay_enabled_ = false;         // ZED-window "Models" toggle: project graph model-instance BBs (starts OFF)
        std::unique_ptr<rc::ModelProjectionOverlay> model_overlay_;   // projects DSR model boxes onto the ZED image

        // Ricoh-360 counterpart of the ZED Models overlay (equirectangular projection, wireframe).
        bool ricoh_model_overlay_enabled_ = false;   // Ricoh-window "Models" toggle (starts OFF)
        bool ricoh_lidar_overlay_enabled_ = false;   // Ricoh-window "Lidar" toggle: reprojected sparse depth (starts OFF)
        // Ricoh-window "Depth" toggle: monocular depth ramp (yolo26l-depth, [RicohDepth]). Starts OFF and
        // also GATES THE STAGE — the model does no work while the overlay is off. Relative per strip, not
        // metric: see src/depth_processor.h before reading anything geometric off it.
        bool ricoh_depth_overlay_enabled_ = false;
        // ZED-window depth overlay: 0=off, 1=aligned model depth, 2=difference vs the ZED's own depth.
        // The ZED MEASURES depth, so it is the one place the model can be scored densely over a whole
        // frame instead of on the LiDAR's horizon stripe — mode 2 is that comparison made visible.
        int  zed_depth_mode_ = 0;
        // Ricoh window: 0=off, 1=room-belief predicted range, 2=difference vs the MONOCULAR model.
        // ★Unlike the ZED case, NEITHER side here is a measurement — it is one prediction against
        // another. Bright means they disagree, which furniture guarantees (the model sees a table,
        // the envelope does not). The useful signal is disagreement on the SHELL itself.
        int  ricoh_room_mode_ = 0;
        // Panorama azimuth convention, read once from the ricoh node (see the compose site).
        bool   ricoh_equirect_read_ = false;
        double ricoh_az_sign_   = 1.0;
        double ricoh_az_offset_ = 0.0;
        // ★THE measurement that decides whether yolo26l-depth is seeing this room at all. Per gnomonic
        // view, correlate the model's log-depth against the helios LiDAR range at the SAME panorama
        // pixels (both are ray range from the ricoh optical centre once zdepth_to_range is on, so they
        // are directly comparable) and fit log_range ≈ a·log_model + b. r says whether the structure is
        // real; (a,b) IS the metric anchor that would make the views consistent. Writes
        // etc/ricoh_depth_lidar.csv. Main thread — the lidar cloud lives here.
        // `panorama_bgr` + `frame_stamp_ms` are used ONLY when a dataset frame is admitted: the image
        // is written as <frames_dir>/<frame_stamp_ms>.jpg so the offline enrichment pass can segment it,
        // and the same stamp goes in the CSV row, which is what joins the two without a schema change.
        void log_ricoh_depth_lidar_correlation(const rc::depth::DepthMap& map,
                                               const std::vector<Eigen::Vector3f>& lidar_room,
                                               const std::vector<std::uint8_t>& plane_id,
                                               const Mat::RTMat& room_T_ricoh,
                                               const cv::Mat& panorama_bgr,
                                               std::uint64_t frame_stamp_ms,
                                               const rc::depth::RoomGeometry& room_geom);
        std::ofstream ricoh_depth_csv_;
        bool          ricoh_depth_csv_open_attempted_ = false;

        // ── Depth-correction dataset (Ricoh popup: "Collect" / "Rebuild map") ───────────────────
        // Collection is a DELIBERATE act, not background logging: the fit is only as good as its
        // pose coverage, and a robot left standing would bury the set under copies of one viewpoint.
        // add_frame() rejects a pose too close to the last kept one for the same reason.
        bool                  ricoh_depth_collect_enabled_ = false;
        rc::depth::DepthDataset depth_dataset_;
        rc::depth::DepthFitMap  depth_fit_map_;      // applied when valid; loaded at startup if present
        std::size_t           depth_collect_session_ = 0;   // frames kept since the button went ON
        // ★Last panorama stamp actually admitted. latest_result() re-serves the SAME bundle while the
        // worker is busy, so without this a robot that crosses the pose-novelty distance between two
        // compute() ticks would admit ONE panorama as TWO frames — identical samples attributed to two
        // different poses, and the second .jpg write would clobber the first.
        std::uint64_t         depth_collect_last_stamp_ = 0;
        // ★The DepthStage must run if EITHER the overlay wants to draw it OR collection needs its
        // output — they are different needs and neither owns the model. Tying the stage's enable flag
        // to the overlay toggle alone (to save GPU when nobody is looking) quietly made a DISPLAY
        // control a precondition for DATA COLLECTION: with Depth off, `rres->depth` is empty, so the
        // LiDAR correlation and the dataset capture both silently did nothing, with no error and no
        // log line. Call this from both toggles instead of setting the flag directly.
        void sync_ricoh_depth_stage();
        // Load the set, drop duplicate poses, refit, save both, and hot-apply. Main thread.
        // Returns an EMPTY string on success, else why it failed. ★It must report failure explicitly:
        // on an early return depth_fit_map_ still holds whatever was loaded at startup, so a button
        // that just prints depth_fit_map_ shows the OLD map's numbers and a failed rebuild is
        // indistinguishable from a successful one. That cost real debugging time.
        [[nodiscard]] std::string rebuild_depth_fit_map();
        // ── OFFLINE ENRICHMENT (rc::depth::DatasetEnricher) ─────────────────────────────────────
        // Re-runs yolo26l-depth + yolo26l-sem over every SAVED panorama, ray-casts the room envelope
        // and adds synthetic ceiling/floor/wall rows where the LiDAR can never reach, then refits.
        // MINUTES, not a moment: the pass runs on the enricher's OWN thread with its OWN ONNX
        // sessions, and this method only pumps the Qt event loop while polling it — so the GUI keeps
        // repainting. `status` is called on the GUI thread with a short progress string.
        // Returns an EMPTY string on success, else why it failed (same contract as the rebuild).
        [[nodiscard]] std::string run_depth_enrichment(
            const std::function<void(const std::string&)>& status);
        // Path of the per-frame room-geometry sidecar the collector writes (schema (b) of the
        // enrichment's point 3). Frames collected before it existed fall back to the live graph.
        static constexpr const char* kDepthRoomsCsv = "etc/ricoh_depth_rooms.csv";
        // Per-frame per-view offsets, re-solved by the correlation pass each compute() tick. The Ricoh
        // popup redraws on the RENDER tick, so it consumes the anchor from the previous compute() —
        // one tick stale on a quantity that moves with the scene, which is immaterial and much cheaper
        // than duplicating the LiDAR reprojection on the render path.
        rc::depth::DepthAnchor depth_anchor_;
        // Epistemic frame admission — see rc::depth::InfoSelector. Seeded from whatever is already on
        // disk when Collect is armed, so a session cannot re-collect what the dataset already knows.
        rc::depth::InfoSelector depth_selector_;
        long                    depth_collect_rejected_ = 0;   // scored but not informative enough
        long                    depth_collect_suspect_  = 0;   // rejected as inconsistent (outlier guard)
        void arm_depth_collection(bool on);
        std::unique_ptr<rc::RicohProjectionOverlay> ricoh_model_overlay_;
        // The ricoh popup renders in on_render_tick (not compute), so cache the last scene the overlay
        // needs. Both run on the Qt main thread → no lock. Updated at the end of each compute() frame.
        struct RicohSceneCache
        {
            std::vector<GraphObjectBox> boxes;
            Mat::RTMat                  room_T_ricoh = Mat::RTMat::Identity();
            std::vector<float>          poly_x, poly_y;
            std::vector<Eigen::Vector3f> lidar_room;   // room-frame lidar sweep, for the reprojected-depth overlay
            float                       room_height = 0.f;
            bool                        valid = false;
        };
        RicohSceneCache ricoh_scene_;
        // Static robot→ricoh mount (from the body→ricoh RT edge), resolved once via inner_eigen then
        // reused; room_T_ricoh = room_T_robot · robot_T_ricoh gives the optical centre without a
        // per-frame tree walk. Empty until the ricoh node exists (config with no ricoh → config fallback).
        std::optional<Mat::RTMat> robot_T_ricoh_;
        // Ricoh detection→bearing UNPROJECT now runs on the RicohYoloWorker thread (compute_bearings).
        // This CameraAPI is only for the MAIN-thread LiDAR depth-fill (reproject_cloud); cached once.
        std::unique_ptr<DSR::CameraAPI> ricoh_camera_api_;
        void save_external_window_geometry() const;

    signals:
        void presenceReady();
        void presenceLost();
};

#endif
