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

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "graph_object_box.h"
#include "rgbd_data.h"

class UnifiedVoxelGrid;
class VoxelProcessor;
class YoloProcessor;
class SceneProcessor;
struct SegDetection;

namespace rc { class VoxelOpenGLViewer; }

class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:
    SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check);
    ~SpecificWorker();

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

private:
    struct Params
    {
        std::string YOLO_MODEL_PATH         = "yolo26l-seg.onnx";
        std::vector<std::string> YOLO_ACCEPTED_LABELS;
        float       YOLO_CONF_THRESH        = 0.25f;
        float       YOLO_IOU_THRESH         = 0.45f;
        int         YOLO_INPUT_SIZE         = 640;
        bool        YOLO_USE_GPU            = true;
        bool        YOLO_USE_TRT            = true;
        int         YOLO_MASK_ERODE_KERNEL  = 0;
        bool        YOLO_MASK_TRAY          = false;
        int         YOLO_TRAY_MASK_REF_WIDTH  = 0;
        int         YOLO_TRAY_MASK_REF_HEIGHT = 0;
        std::vector<cv::Point> YOLO_TRAY_MASK_POLYGON_PX;
        float       TRACK_ASSOCIATION_MAX_DISTANCE_M = 0.7f;
        int         TRACK_MAX_MISSED_FRAMES          = 10;
        std::size_t VOXEL_VIEWER_MAX_RENDERED_VOXELS = 30'000;
        int         VOXEL_VIEWER_FPS                 = 10;
        std::size_t VOXEL_DECIMATION_FACTOR          = 2;
        bool        TRANSFORMS_INTERPOLATE_RT        = true;
    } params;

    struct SceneFrame
    {
        RGBDData                    rgbd;
        Mat::RTMat                  room_T_robot;
        Mat::RTMat                  room_T_zed;
        std::vector<GraphObjectBox> graph_object_boxes;
    };

    std::optional<SceneFrame> process_scene_frame(FPSCounter& compute_fps);
    void update_table_nodes_from_tracks(const std::vector<GraphObjectBox>& graph_object_boxes);

    bool startup_check_flag = false;
    bool verbose_debug_     = false;

    std::shared_ptr<DSR::InnerEigenAPI> inner_eigen_api;

    std::unique_ptr<YoloProcessor>     yolo_processor;
    std::unique_ptr<UnifiedVoxelGrid>  voxel_grid;
    std::unique_ptr<VoxelProcessor>    voxel_processor;
    std::unique_ptr<SceneProcessor>    scene_processor;
    std::unique_ptr<rc::VoxelOpenGLViewer> voxel_viewer_gl;

};

#endif
