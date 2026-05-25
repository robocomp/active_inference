#include "scene_processor.h"

#include "rgbd_data.h"
#include "voxel_opengl_viewer.h"

#include <dsr/api/dsr_camera_api.h>

#include <Eigen/Geometry>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <print>
#include <variant>

SceneProcessor::SceneProcessor(const std::shared_ptr<DSR::DSRGraph>& graph)
    : graph_(graph)
{
}

void SceneProcessor::configure(DSR::InnerEigenAPI* inner_eigen_api,
                               rc::VoxelOpenGLViewer* voxel_viewer,
                               bool transforms_interpolate_rt,
                               bool verbose_debug)
{
    inner_eigen_api_ = inner_eigen_api;
    voxel_viewer_ = voxel_viewer;
    transforms_interpolate_rt_ = transforms_interpolate_rt;
    verbose_debug_ = verbose_debug;
}

std::pair<std::string, std::string> SceneProcessor::get_room_robot_names_for_compute()
{
    std::string room_name_snapshot;
    std::string robot_name_snapshot;
    {
        std::scoped_lock lk(node_names_mutex_);
        room_name_snapshot = room_node_name_;
        robot_name_snapshot = robot_node_name_;
    }

    if (room_name_snapshot.empty())
    {
        if (const auto room_nodes = graph_->get_nodes_by_type("room"); !room_nodes.empty())
            room_name_snapshot = room_nodes.front().name();
    }

    if (robot_name_snapshot.empty())
    {
        if (const auto robot_nodes = graph_->get_nodes_by_type("robot"); !robot_nodes.empty())
            robot_name_snapshot = robot_nodes.front().name();
    }

    {
        std::scoped_lock lk(node_names_mutex_);
        if (room_node_name_.empty() && !room_name_snapshot.empty())
            room_node_name_ = room_name_snapshot;
        if (robot_node_name_.empty() && !robot_name_snapshot.empty())
            robot_node_name_ = robot_name_snapshot;
    }

    return {room_name_snapshot, robot_name_snapshot};
}

bool SceneProcessor::ensure_room_and_robot_ready(FPSCounter& compute_fps,
                                                 const std::string& room_name,
                                                 const std::string& robot_name)
{
    if (room_name.empty())
    {
        if (!room_wait_logged_)
        {
            qWarning() << "Room node not found in DSR graph. Voxelization paused until a room exists.";
            room_wait_logged_ = true;
            room_ready_logged_ = false;
        }
        if (verbose_debug_)
            compute_fps.print("[Compute]", 2000);
        return false;
    }

    if (robot_name.empty())
    {
        if (!room_wait_logged_)
        {
            qWarning() << "Robot node not found in DSR graph. Voxelization paused until a robot exists.";
            room_wait_logged_ = true;
            room_ready_logged_ = false;
        }
        if (verbose_debug_)
            compute_fps.print("[Compute]", 2000);
        return false;
    }

    if (!room_ready_logged_)
    {
        qInfo() << "Room node found in DSR graph. Voxelization enabled.";
        room_ready_logged_ = true;
    }
    return true;
}

std::optional<Mat::RTMat> SceneProcessor::get_room_robot_transform(FPSCounter& compute_fps,
                                                                   const std::string& room_name,
                                                                   const std::string& robot_name,
                                                                   std::uint64_t timestamp_ms)
{
    if (inner_eigen_api_ == nullptr)
    {
        if (!room_rt_wait_logged_)
        {
            qWarning() << "InnerEigen API is not available. Voxelization paused.";
            room_rt_wait_logged_ = true;
            room_rt_ready_logged_ = false;
        }
        if (verbose_debug_)
            compute_fps.print("[Compute]", 2000);
        return std::nullopt;
    }

    auto room_T_robot = inner_eigen_api_->get_transformation_matrix(room_name, robot_name, timestamp_ms);
    if (!room_T_robot.has_value())
    {
        if (!room_rt_wait_logged_)
        {
            qWarning() << "robot->room RTMat not available in InnerEigen API. Voxelization paused until transform is available."
                       << "room=" << QString::fromStdString(room_name)
                       << "robot=" << QString::fromStdString(robot_name)
                       << "ts_ms=" << timestamp_ms;
            room_rt_wait_logged_ = true;
            room_rt_ready_logged_ = false;
        }
        if (verbose_debug_)
            compute_fps.print("[Compute]", 2000);
        return std::nullopt;
    }

    return room_T_robot;
}

std::optional<Mat::RTMat> SceneProcessor::get_room_zed_transform(FPSCounter& compute_fps,
                                                                 const std::string& room_name,
                                                                 std::uint64_t timestamp_ms)
{
    if (inner_eigen_api_ == nullptr)
    {
        if (!room_rt_wait_logged_)
        {
            qWarning() << "InnerEigen API is not available. Voxelization paused.";
            room_rt_wait_logged_ = true;
            room_rt_ready_logged_ = false;
        }
        if (verbose_debug_)
            compute_fps.print("[Compute]", 2000);
        return std::nullopt;
    }

    auto room_T_zed = inner_eigen_api_->get_transformation_matrix(room_name, "zed", timestamp_ms);
    if (!room_T_zed.has_value())
    {
        if (!room_rt_wait_logged_)
        {
            qWarning() << "zed->room RTMat not available in InnerEigen API. Voxelization paused until transform is available."
                       << "room=" << QString::fromStdString(room_name)
                       << "zed=zed"
                       << "ts_ms=" << timestamp_ms;
            room_rt_wait_logged_ = true;
            room_rt_ready_logged_ = false;
        }
        if (verbose_debug_)
            compute_fps.print("[Compute]", 2000);
        return std::nullopt;
    }

    return room_T_zed;
}

std::uint64_t SceneProcessor::get_frame_timestamp_ms() const
{
    if (graph_)
    {
        if (auto zed_node = graph_->get_node("zed"); zed_node.has_value())
        {
            if (auto ts = graph_->get_attrib_by_name<cam_rgb_alivetime_att>(zed_node.value()); ts.has_value())
                return ts.value();
        }
    }
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

std::optional<cv::Mat> SceneProcessor::get_rgb_from_dsr() const
{
    if (!graph_)
        return std::nullopt;
    auto zed_node = graph_->get_node("zed");
    if (!zed_node.has_value())
        return std::nullopt;

    auto data_opt    = graph_->get_attrib_by_name<cam_rgb_att>(zed_node.value());
    auto width_opt   = graph_->get_attrib_by_name<cam_rgb_width_att>(zed_node.value());
    auto height_opt  = graph_->get_attrib_by_name<cam_rgb_height_att>(zed_node.value());
    auto depth_opt   = graph_->get_attrib_by_name<cam_rgb_depth_att>(zed_node.value());
    if (!data_opt.has_value() || !width_opt.has_value() || !height_opt.has_value())
        return std::nullopt;

    const int w = width_opt.value();
    const int h = height_opt.value();
    const int ch = depth_opt.has_value() ? depth_opt.value() : 3;
    const auto& bytes = data_opt.value().get();
    if (w <= 0 || h <= 0 || static_cast<int>(bytes.size()) < w * h * ch)
        return std::nullopt;

    // Wrap in cv::Mat without copying, then clone so the node can be released.
    return cv::Mat(h, w, ch == 4 ? CV_8UC4 : CV_8UC3,
                   const_cast<void*>(static_cast<const void*>(bytes.data()))).clone();
}

std::optional<SceneProcessor::LidarData> SceneProcessor::get_lidar_from_dsr() const
{
    if (!graph_)
        return std::nullopt;
    auto lidar_node = graph_->get_node("lidar3D");
    if (!lidar_node.has_value())
        return std::nullopt;

    auto xs_opt = graph_->get_attrib_by_name<laser_X_att>(lidar_node.value());
    auto ys_opt = graph_->get_attrib_by_name<laser_Y_att>(lidar_node.value());
    auto zs_opt = graph_->get_attrib_by_name<laser_Z_att>(lidar_node.value());
    auto ts_opt = graph_->get_attrib_by_name<laser_timestamp_att>(lidar_node.value());
    if (!xs_opt.has_value() || !ys_opt.has_value() || !zs_opt.has_value())
        return std::nullopt;

    LidarData ld;
    ld.xs = std::vector<float>(xs_opt.value().get().begin(), xs_opt.value().get().end());
    ld.ys = std::vector<float>(ys_opt.value().get().begin(), ys_opt.value().get().end());
    ld.zs = std::vector<float>(zs_opt.value().get().begin(), zs_opt.value().get().end());
    ld.timestamp_ms = ts_opt.has_value() ? ts_opt.value() : 0;
    if (ld.xs.empty())
        return std::nullopt;
    return ld;
}

std::optional<RGBDData> SceneProcessor::get_rgbd_frame_from_dsr() const
{
    if (!graph_)
        return std::nullopt;
    auto zed_node = graph_->get_node("zed");
    if (!zed_node.has_value())
        return std::nullopt;

    auto camera_api = graph_->get_camera_api(zed_node.value());
    if (!camera_api)
        return std::nullopt;

    // --- RGB ---
    auto rgb_opt = get_rgb_from_dsr();
    if (!rgb_opt.has_value() || rgb_opt->empty())
        return std::nullopt;

    // --- Dense 3D point cloud from depth stored in DSR (camera frame) ---
    auto pts_opt = camera_api->get_pointcloud("", 1);
    if (!pts_opt.has_value() || pts_opt->empty())
        return std::nullopt;

    const int w = rgb_opt->cols;
    const int h = rgb_opt->rows;
    const auto& raw = pts_opt.value();

    RGBDData data;
    data.rgb    = std::move(*rgb_opt);
    data.width  = w;
    data.height = h;
    data.points.resize(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i)
    {
        data.points[i].x = std::get<0>(raw[i]);
        data.points[i].y = std::get<1>(raw[i]);
        data.points[i].z = std::get<2>(raw[i]);
    }
    return data;
}

void SceneProcessor::check_input_stream_startup_status()
{
    constexpr auto startup_grace = std::chrono::seconds(3);
    const auto now = std::chrono::steady_clock::now();
    if (now - input_stream_watchdog_start_ < startup_grace)
        return;

    if (graph_)
    {
        if (auto zed = graph_->get_node("zed"); zed.has_value())
        {
            const bool has_rgb = graph_->get_attrib_by_name<cam_rgb_att>(zed.value()).has_value();
            if (!has_rgb)
                std::print(stderr, "[voxelizer] DSR 'zed' node exists but cam_rgb_att is empty. Waiting for robot_concept to populate it...\n");
        }
        else
        {
            std::print(stderr, "[voxelizer] DSR 'zed' node not found. Waiting...\n");
        }

        if (auto lidar = graph_->get_node("lidar3D"); lidar.has_value())
        {
            const bool has_pts = graph_->get_attrib_by_name<laser_X_att>(lidar.value()).has_value();
            if (!has_pts)
                std::print(stderr, "[voxelizer] DSR 'lidar3D' node exists but laser_X_att is empty. Waiting...\n");
        }
        else
        {
            std::print(stderr, "[voxelizer] DSR 'lidar3D' node not found. Waiting...\n");
        }
    }
}

void SceneProcessor::log_room_robot_pose_periodic(const Mat::RTMat& room_T_robot) const
{
    (void)room_T_robot;
}

void SceneProcessor::mark_room_rt_ready()
{
    if (!room_rt_ready_logged_)
    {
        room_rt_ready_logged_ = true;
        room_rt_wait_logged_ = false;
    }
}

void SceneProcessor::update_room_polygon_periodic()
{
    ++polygon_check_count_;
    if (polygon_check_count_ % 50 == 0)
        update_room_polygon_in_viewers();
}

bool SceneProcessor::compute_room_to_camera_basis(const std::string& camera_node_name,
                                                  const std::string& room_frame_name,
                                                  std::uint64_t rt_timestamp,
                                                  RoomToCameraBasis& basis) const
{
    if (inner_eigen_api_ == nullptr)
        return false;

    const auto time_query = transforms_interpolate_rt_
        ? DSR::RT_API::TimeQuery::Interpolated
        : DSR::RT_API::TimeQuery::Nearest;

    const auto origin_opt = inner_eigen_api_->transform(camera_node_name, Mat::Vector3d(0.0, 0.0, 0.0), room_frame_name, rt_timestamp, "RT", time_query);
    const auto x_opt = inner_eigen_api_->transform(camera_node_name, Mat::Vector3d(1.0, 0.0, 0.0), room_frame_name, rt_timestamp, "RT", time_query);
    const auto y_opt = inner_eigen_api_->transform(camera_node_name, Mat::Vector3d(0.0, 1.0, 0.0), room_frame_name, rt_timestamp, "RT", time_query);
    const auto z_opt = inner_eigen_api_->transform(camera_node_name, Mat::Vector3d(0.0, 0.0, 1.0), room_frame_name, rt_timestamp, "RT", time_query);
    if (!origin_opt.has_value() || !x_opt.has_value() || !y_opt.has_value() || !z_opt.has_value())
        return false;

    basis.origin = origin_opt.value();
    basis.axis_x = x_opt.value() - basis.origin;
    basis.axis_y = y_opt.value() - basis.origin;
    basis.axis_z = z_opt.value() - basis.origin;
    return true;
}

std::optional<SceneProcessor::RoomPolygonData> SceneProcessor::get_room_polygon_from_graph() const
{
    if (!graph_)
        return std::nullopt;

    std::string room_name_snapshot;
    {
        std::scoped_lock lk(node_names_mutex_);
        room_name_snapshot = room_node_name_;
    }

    if (room_name_snapshot.empty())
    {
        if (const auto room_nodes = graph_->get_nodes_by_type("room"); !room_nodes.empty())
            room_name_snapshot = room_nodes.front().name();
    }

    if (room_name_snapshot.empty())
        return std::nullopt;

    auto room_node = graph_->get_node(room_name_snapshot);
    if (!room_node.has_value())
        return std::nullopt;

    auto polygon_x_opt = graph_->get_attrib_by_name<delimiting_polygon_x_att>(room_node.value());
    auto polygon_y_opt = graph_->get_attrib_by_name<delimiting_polygon_y_att>(room_node.value());
    if (!polygon_x_opt.has_value() || !polygon_y_opt.has_value())
        return std::nullopt;

    const auto& polygon_x_src = polygon_x_opt.value().get();
    const auto& polygon_y_src = polygon_y_opt.value().get();
    if (polygon_x_src.empty() || polygon_y_src.empty())
        return std::nullopt;

    RoomPolygonData data;
    data.room_name = std::move(room_name_snapshot);
    data.polygon_x.assign(polygon_x_src.begin(), polygon_x_src.end());
    data.polygon_y.assign(polygon_y_src.begin(), polygon_y_src.end());
    if (auto height_opt = graph_->get_attrib_by_name<room_height_att>(room_node.value()); height_opt.has_value())
        data.room_height = height_opt.value();

    return data;
}

std::optional<GraphObjectBox> SceneProcessor::build_graph_object_box(const DSR::Node& node,
                                                                     const std::string& room_name,
                                                                     std::uint64_t timestamp_ms) const
{
    if (!graph_ || inner_eigen_api_ == nullptr || room_name.empty())
        return std::nullopt;

    const auto width_opt = graph_->get_attrib_by_name<width_m_att>(node);
    const auto depth_opt = graph_->get_attrib_by_name<depth_m_att>(node);
    const auto height_opt = graph_->get_attrib_by_name<height_m_att>(node);
    if (!width_opt.has_value() || !depth_opt.has_value() || !height_opt.has_value())
    {
        std::println("[build_graph_object_box] node='{}' type='{}': missing dimensions (w={} d={} h={})",
                     node.name(), node.type(),
                     width_opt.has_value(), depth_opt.has_value(), height_opt.has_value());
        return std::nullopt;
    }

    const float width = width_opt.value();
    const float depth = depth_opt.value();
    const float height = height_opt.value();
    if (width <= 0.f || depth <= 0.f || height <= 0.f)
    {
        std::println("[build_graph_object_box] node='{}': non-positive dims w={} d={} h={}",
                     node.name(), width, depth, height);
        return std::nullopt;
    }

    const auto time_query = transforms_interpolate_rt_
        ? DSR::RT_API::TimeQuery::Interpolated
        : DSR::RT_API::TimeQuery::Nearest;
    const auto room_T_object = inner_eigen_api_->get_transformation_matrix(room_name,
                                                                           node.name(),
                                                                           timestamp_ms,
                                                                           "RT",
                                                                           time_query);
    if (!room_T_object.has_value())
    {
        std::println("[build_graph_object_box] node='{}': no RT transform from room '{}'",
                     node.name(), room_name);
        return std::nullopt;
    }

    const float half_width = width * 0.5f;
    const float half_depth = depth * 0.5f;
    const float half_height = height * 0.5f;

    const std::array<Eigen::Vector3d, 8> local_corners = {
        Eigen::Vector3d{-half_width, -half_depth, -half_height},
        Eigen::Vector3d{ half_width, -half_depth, -half_height},
        Eigen::Vector3d{ half_width,  half_depth, -half_height},
        Eigen::Vector3d{-half_width,  half_depth, -half_height},
        Eigen::Vector3d{-half_width, -half_depth,  half_height},
        Eigen::Vector3d{ half_width, -half_depth,  half_height},
        Eigen::Vector3d{ half_width,  half_depth,  half_height},
        Eigen::Vector3d{-half_width,  half_depth,  half_height}
    };

    Eigen::Vector3f min_corner = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
    Eigen::Vector3f max_corner = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());

    for (const auto& local_corner : local_corners)
    {
        const Eigen::Vector3d room_corner = room_T_object->linear() * local_corner + room_T_object->translation();
        min_corner.x() = std::min(min_corner.x(), static_cast<float>(room_corner.x()));
        min_corner.y() = std::min(min_corner.y(), static_cast<float>(room_corner.y()));
        min_corner.z() = std::min(min_corner.z(), static_cast<float>(room_corner.z()));
        max_corner.x() = std::max(max_corner.x(), static_cast<float>(room_corner.x()));
        max_corner.y() = std::max(max_corner.y(), static_cast<float>(room_corner.y()));
        max_corner.z() = std::max(max_corner.z(), static_cast<float>(room_corner.z()));
    }

    std::string category = node.name();
    if (const auto it = node.attrs().find("semantic_class"); it != node.attrs().end() && it->second.selected() == 0)
        category = it->second.str();

    // Keep graph/model tables visually distinct from YOLO-derived table tracks.
    // node.type() == "table" covers all table_1, table_2, … nodes.
    if (node.type() == "table")
        category = "model_table";

    return GraphObjectBox{min_corner, max_corner, std::move(category)};
}

std::vector<GraphObjectBox> SceneProcessor::get_graph_object_boxes(const std::string& room_name,
                                                                   std::uint64_t timestamp_ms) const
{
    std::vector<GraphObjectBox> graph_boxes;
    if (!graph_ || room_name.empty())
        return graph_boxes;

    const auto object_nodes = graph_->get_nodes_by_type("object");
    const auto table_nodes  = graph_->get_nodes_by_type("table");
    graph_boxes.reserve(object_nodes.size() + table_nodes.size());
    auto add_boxes = [&](const auto& nodes)
    {
        for (const auto& node : nodes)
        {
            const auto box = build_graph_object_box(node, room_name, timestamp_ms);
            if (box.has_value())
                graph_boxes.push_back(box.value());
        }
    };
    add_boxes(object_nodes);
    add_boxes(table_nodes);
    return graph_boxes;
}

void SceneProcessor::overlay_room_polygon_on_canvas(cv::Mat& canvas, std::uint64_t frame_ts_ms) const
{
    if (canvas.empty() || !graph_ || inner_eigen_api_ == nullptr)
        return;

    auto room_data = get_room_polygon_from_graph();
    if (!room_data.has_value())
        return;

    auto zed_node = graph_->get_node("zed");
    if (!zed_node.has_value())
        return;

    auto camera_api = graph_->get_camera_api(zed_node.value());
    if (!camera_api)
        return;

    const std::size_t n = std::min(room_data->polygon_x.size(), room_data->polygon_y.size());
    if (n < 2)
        return;

    RoomToCameraBasis basis;
    if (!compute_room_to_camera_basis("zed", room_data->room_name, frame_ts_ms, basis))
        return;

    Eigen::Matrix<double, 3, Eigen::Dynamic> room_points(3, static_cast<Eigen::Index>(n));
    Eigen::Matrix<double, 3, Eigen::Dynamic> room_points_top(3, static_cast<Eigen::Index>(n));
    for (std::size_t i = 0; i < n; ++i)
    {
        room_points(0, static_cast<Eigen::Index>(i)) = static_cast<double>(room_data->polygon_x[i]);
        room_points(1, static_cast<Eigen::Index>(i)) = static_cast<double>(room_data->polygon_y[i]);
        room_points(2, static_cast<Eigen::Index>(i)) = 0.0;
        room_points_top(0, static_cast<Eigen::Index>(i)) = static_cast<double>(room_data->polygon_x[i]);
        room_points_top(1, static_cast<Eigen::Index>(i)) = static_cast<double>(room_data->polygon_y[i]);
        room_points_top(2, static_cast<Eigen::Index>(i)) = static_cast<double>(room_data->room_height);
    }

    Eigen::Matrix3d basis_matrix;
    basis_matrix.col(0) = basis.axis_x;
    basis_matrix.col(1) = basis.axis_y;
    basis_matrix.col(2) = basis.axis_z;
    const Eigen::Vector3d basis_origin = basis.origin;

    Eigen::Matrix<double, 3, Eigen::Dynamic> zed_points =
        (basis_matrix * room_points).colwise() + basis_origin;
    Eigen::Matrix<double, 3, Eigen::Dynamic> zed_points_top =
        (basis_matrix * room_points_top).colwise() + basis_origin;

    auto project_clipped_segment = [&](Eigen::Vector3d a, Eigen::Vector3d b,
                                       cv::Point& out_a, cv::Point& out_b) -> bool
    {
        constexpr double near_y = 1e-4;

        if (a.y() <= near_y && b.y() <= near_y)
            return false;

        if (a.y() <= near_y)
        {
            const double t = (near_y - a.y()) / (b.y() - a.y());
            a = a + t * (b - a);
        }
        else if (b.y() <= near_y)
        {
            const double t = (near_y - b.y()) / (a.y() - b.y());
            b = b + t * (a - b);
        }

        const Eigen::Vector2d uv0 = camera_api->project(a);
        const Eigen::Vector2d uv1 = camera_api->project(b);
        if (!std::isfinite(uv0.x()) || !std::isfinite(uv0.y()) || !std::isfinite(uv1.x()) || !std::isfinite(uv1.y()))
            return false;

        out_a = cv::Point(static_cast<int>(std::lround(uv0.x())), static_cast<int>(std::lround(uv0.y())));
        out_b = cv::Point(static_cast<int>(std::lround(uv1.x())), static_cast<int>(std::lround(uv1.y())));
        return true;
    };

    std::vector<std::optional<cv::Point>> projected_floor(n);
    std::vector<std::optional<cv::Point>> projected_top(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        const Eigen::Vector3d floor_point_zed = zed_points.col(static_cast<Eigen::Index>(i));
        if (floor_point_zed.y() > 1e-6)
        {
            const Eigen::Vector2d uv = camera_api->project(floor_point_zed);
            if (std::isfinite(uv.x()) && std::isfinite(uv.y()))
                projected_floor[i] = cv::Point(static_cast<int>(std::lround(uv.x())), static_cast<int>(std::lround(uv.y())));
        }

        const Eigen::Vector3d top_point_zed = zed_points_top.col(static_cast<Eigen::Index>(i));
        if (top_point_zed.y() > 1e-6)
        {
            const Eigen::Vector2d uv = camera_api->project(top_point_zed);
            if (std::isfinite(uv.x()) && std::isfinite(uv.y()))
                projected_top[i] = cv::Point(static_cast<int>(std::lround(uv.x())), static_cast<int>(std::lround(uv.y())));
        }
    }

    const cv::Scalar floor_colour(255, 0, 255);
    const cv::Scalar top_colour(0, 0, 255);
    const cv::Scalar vertical_colour(0, 200, 255);
    for (std::size_t i = 0; i < n; ++i)
    {
        const std::size_t j = (i + 1) % n;

        cv::Point p0;
        cv::Point p1;
        if (project_clipped_segment(zed_points.col(static_cast<Eigen::Index>(i)),
                                    zed_points.col(static_cast<Eigen::Index>(j)),
                                    p0, p1))
        {
            if (cv::clipLine(canvas.size(), p0, p1))
                cv::line(canvas, p0, p1, floor_colour, 2, cv::LINE_AA);
        }

        if (project_clipped_segment(zed_points_top.col(static_cast<Eigen::Index>(i)),
                                    zed_points_top.col(static_cast<Eigen::Index>(j)),
                                    p0, p1))
        {
            if (cv::clipLine(canvas.size(), p0, p1))
                cv::line(canvas, p0, p1, top_colour, 2, cv::LINE_AA);
        }

        if (project_clipped_segment(zed_points.col(static_cast<Eigen::Index>(i)),
                                    zed_points_top.col(static_cast<Eigen::Index>(i)),
                                    p0, p1))
        {
            if (cv::clipLine(canvas.size(), p0, p1))
                cv::line(canvas, p0, p1, vertical_colour, 2, cv::LINE_AA);
        }
    }

    for (const auto& p : projected_floor)
    {
        if (!p.has_value())
            continue;
        if (p->x >= 0 && p->x < canvas.cols && p->y >= 0 && p->y < canvas.rows)
            cv::circle(canvas, p.value(), 4, floor_colour, cv::FILLED, cv::LINE_AA);
    }

    for (const auto& p : projected_top)
    {
        if (!p.has_value())
            continue;
        if (p->x >= 0 && p->x < canvas.cols && p->y >= 0 && p->y < canvas.rows)
            cv::circle(canvas, p.value(), 4, top_colour, cv::FILLED, cv::LINE_AA);
    }
}

void SceneProcessor::update_room_polygon_in_viewers()
{
    auto room_data = get_room_polygon_from_graph();
    if (!room_data.has_value())
        return;

    try
    {
        if (!room_data->polygon_x.empty() && !room_data->polygon_y.empty() && voxel_viewer_ != nullptr)
            voxel_viewer_->update_room_polygon_dual(room_data->polygon_x, room_data->polygon_y, room_data->room_height);
    }
    catch (const std::exception& e)
    {
        qWarning() << "update_room_polygon_in_viewers failed:" << e.what();
    }
}

void SceneProcessor::update_viewer_graph_object_boxes(std::span<const GraphObjectBox> graph_boxes)
{
    if (voxel_viewer_ == nullptr)
        return;

    if (graph_boxes.empty())
    {
        voxel_viewer_->update_graph_boxes({}, {}, {});
        return;
    }

    std::vector<QVector3D> mins;
    std::vector<QVector3D> maxs;
    std::vector<std::string> categories;
    mins.reserve(graph_boxes.size());
    maxs.reserve(graph_boxes.size());
    categories.reserve(graph_boxes.size());

    for (const auto& box : graph_boxes)
    {
        mins.emplace_back(box.min.x(), box.min.y(), box.min.z());
        maxs.emplace_back(box.max.x(), box.max.y(), box.max.z());
        categories.push_back(box.category);
    }

    voxel_viewer_->update_graph_boxes(mins, maxs, categories);
}

void SceneProcessor::update_viewer_robot_pose(const Mat::RTMat& room_T_robot)
{
    if (voxel_viewer_ == nullptr)
        return;
    const auto& t = room_T_robot.translation();
    const Eigen::Matrix3d R = room_T_robot.rotation();
    const float theta = static_cast<float>(std::atan2(R(1, 0), R(0, 0)));
    voxel_viewer_->set_robot_pose(static_cast<float>(t.x()), static_cast<float>(t.y()), theta);
}

void SceneProcessor::update_viewer_lidar_points(const std::string& room_name,
                                                const std::string& robot_name,
                                                const Mat::RTMat& room_T_robot_fallback)
{
    if (voxel_viewer_ == nullptr)
        return;

    const auto lidar_data = get_lidar_from_dsr();
    if (!lidar_data.has_value() || lidar_data->xs.empty())
    {
        voxel_viewer_->update_lidar_points({});
        return;
    }

    const auto& xs = lidar_data->xs;
    const auto& ys = lidar_data->ys;
    const auto& zs = lidar_data->zs;
    const std::uint64_t lidar_timestamp_ms = lidar_data->timestamp_ms;

    Mat::RTMat room_T_robot = room_T_robot_fallback;
    if (inner_eigen_api_ != nullptr && lidar_timestamp_ms > 0)
    {
        const auto time_query = transforms_interpolate_rt_
            ? DSR::RT_API::TimeQuery::Interpolated
            : DSR::RT_API::TimeQuery::Nearest;
        if (auto interpolated = inner_eigen_api_->get_transformation_matrix(
                room_name,
                robot_name,
                lidar_timestamp_ms,
                "RT",
                time_query); interpolated.has_value())
        {
            room_T_robot = interpolated.value();
        }
    }

    std::vector<QVector3D> lidar_points_room;
    lidar_points_room.reserve(xs.size());
    for (std::size_t i = 0; i < xs.size(); ++i)
    {
        const Eigen::Vector3d point_robot(static_cast<double>(xs[i]),
                                          static_cast<double>(ys[i]),
                                          static_cast<double>(zs[i]));
        const Eigen::Vector3d point_room = room_T_robot.linear() * point_robot + room_T_robot.translation();
        lidar_points_room.emplace_back(static_cast<float>(point_room.x()),
                                       static_cast<float>(point_room.y()),
                                       static_cast<float>(point_room.z()));
    }

    voxel_viewer_->update_lidar_points(lidar_points_room);
}