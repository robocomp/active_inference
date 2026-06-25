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

#include "graph_object_box.h"
#include "lidar_track_attributor.h"
#include "rgbd_data.h"
#include "voxel_budget_regulator.h"
#include "voxelizer_params.h"

#include <chrono>

class UnifiedVoxelGrid;
class VoxelProcessor;
class YoloProcessor;
class SceneProcessor;
class GraphPublisher;
struct SegDetection;

namespace rc { class VoxelOpenGLViewer; }
namespace rc { class YoloViewer; }

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
            std::vector<GraphObjectBox> graph_object_boxes;
            std::uint64_t               frame_ts_ms = 0;   // capture stamp of rgbd/depth — published so consumers can pin pose to capture time
        };

        std::optional<SceneFrame> process_scene_frame(FPSCounter& compute_fps);
        void setup_custom_viewers();         // Voxel3D GL + YOLO windows (specificworker_viewers.cpp)
        void trigger_graph_layout_twopi();   // injected into GraphPublisher as the relayout callback

        AgentPresenceCoordinator presence_coordinator_;
        bool owned_nodes_cleaned_ = false;
        FPSCounter fps_counter_;

        // Homeostatic voxel-budget control: adapt max_voxels to hold target FPS.
        VoxelBudgetRegulator voxel_budget_;
        std::chrono::steady_clock::time_point last_budget_tick_{};
        void regulate_voxel_budget(float fps);   // call once per compute(); self-throttled

        bool startup_check_flag  = false;
        bool verbose_debug_      = false;
        std::atomic<bool> shutting_down_{false};
        bool include_lidar3d_in_voxels_ = true;

        std::shared_ptr<DSR::InnerEigenAPI> inner_eigen_api;

        std::unique_ptr<YoloProcessor>     yolo_processor;
        std::unique_ptr<LidarTrackAttributor> lidar_track_attributor;
        std::unique_ptr<UnifiedVoxelGrid>  voxel_grid;
        std::unique_ptr<VoxelProcessor>    voxel_processor;
        std::unique_ptr<SceneProcessor>    scene_processor;
        std::unique_ptr<GraphPublisher>    graph_publisher_;   // all DSR semantic_grid exports
        std::unique_ptr<rc::VoxelOpenGLViewer> voxel_viewer_gl;
        std::unique_ptr<rc::YoloViewer>        yolo_viewer_;

        // Own-window holders returned by add_custom_widget_in_own_window. Borrowed (parented to the
        // DSR main window); we only read/persist their geometry, we do not own them.
        QMainWindow* voxel3d_window_ = nullptr;
        QMainWindow* yolo_window_ = nullptr;
        bool yolo_window_needs_image_size_ = false;  // size the RGB window to the image on first frame
        void save_external_window_geometry() const;

    signals:
        void presenceReady();
        void presenceLost();
};

#endif
