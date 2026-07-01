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

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "rgbd_data.h"
#include "voxelizer_params.h"
#include "perception_rate_regulator.h"
#include "stream_rate_monitor.h"

#include <chrono>

class YoloProcessor;
class SceneProcessor;
class GraphPublisher;
struct SegDetection;
namespace rc::human_pose { class YoloHumanProcessor; }
namespace rc::semantic { class YoloSemanticProcessor; }

namespace rc { class VoxelOpenGLViewer; }
namespace rc { class YoloViewer; }
namespace rc { class ImagePopupViewer; }

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

    private:
        VoxelizerParams params;   // loaded via load_voxelizer_params() in initialize()

        struct SceneFrame
        {
            RGBDData                    rgbd;
            Mat::RTMat                  room_T_robot;
            Mat::RTMat                  room_T_zed;
            std::vector<Eigen::Vector3f> lidar_points_room;
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

        // Homeostatic perception-rate regulator: adapts pose decimation to hold compute() near
        // params.TARGET_HZ (voxel-free; fed compute cost + frame stamp). Actuator = pose only.
        rc::PerceptionRateRegulator rate_reg_;

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

        std::unique_ptr<YoloProcessor>     yolo_processor;
        std::unique_ptr<rc::human_pose::YoloHumanProcessor> yolo_human_processor;
        std::unique_ptr<rc::semantic::YoloSemanticProcessor> yolo_semantic_processor;
        std::unique_ptr<SceneProcessor>    scene_processor;
        std::unique_ptr<GraphPublisher>    graph_publisher_;   // all DSR semantic_grid exports
        std::unique_ptr<rc::VoxelOpenGLViewer> voxel_viewer_gl;
        std::unique_ptr<rc::YoloViewer>        yolo_viewer_;
        std::unique_ptr<rc::ImagePopupViewer>  ricoh_viewer_;   // RGBD_360 panorama popup

        // Decoupled viewer-refresh timer (GUI thread). Pushes the LATEST robot pose to the 3D viewer
        // at a fluid cadence, independent of the ~7-10 Hz perception/camera pipeline, so robot motion
        // is smooth. Started in Operating, stopped in Degraded (graph access stays within Operating).
        std::unique_ptr<QTimer> render_timer_;
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
        void save_external_window_geometry() const;

    signals:
        void presenceReady();
        void presenceLost();
};

#endif
