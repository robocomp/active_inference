#include "scene_processor.h"

#include "voxel_opengl_viewer.h"

#include <dsr/api/dsr_camera_api.h>

#include <Eigen/Geometry>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <print>

SceneProcessor::SceneProcessor(const std::shared_ptr<DSR::DSRGraph>& graph,
                               std::mutex& lidar_points_mutex,
                               std::vector<float>& latest_lidar_xs,
                               std::vector<float>& latest_lidar_ys,
                               std::vector<float>& latest_lidar_zs,
                               std::uint64_t& latest_lidar_timestamp_ms,
                               std::atomic<bool>& lidar_stream_seen,
                               std::atomic<bool>& rgbd_stream_seen,
                               std::atomic<bool>& lidar_stream_wait_logged,
                               std::atomic<bool>& rgbd_stream_wait_logged)
    : graph_(graph)
    , lidar_points_mutex_(lidar_points_mutex)
    , latest_lidar_xs_(latest_lidar_xs)
    , latest_lidar_ys_(latest_lidar_ys)
    , latest_lidar_zs_(latest_lidar_zs)
    , latest_lidar_timestamp_ms_(latest_lidar_timestamp_ms)
    , lidar_stream_seen_(lidar_stream_seen)
    , rgbd_stream_seen_(rgbd_stream_seen)
    , lidar_stream_wait_logged_(lidar_stream_wait_logged)
    , rgbd_stream_wait_logged_(rgbd_stream_wait_logged)
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

std::uint64_t SceneProcessor::get_rgbd_frame_timestamp_ms(const RoboCompCameraRGBDSimple::TRGBD& rgbd) const
{
    if (rgbd.image.alivetime > 0)
        return static_cast<std::uint64_t>(rgbd.image.alivetime);
    if (rgbd.depth.alivetime > 0)
        return static_cast<std::uint64_t>(rgbd.depth.alivetime);
    if (rgbd.points.alivetime > 0)
        return static_cast<std::uint64_t>(rgbd.points.alivetime);
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

void SceneProcessor::check_input_stream_startup_status()
{
    constexpr auto startup_grace = std::chrono::seconds(3);
    const auto now = std::chrono::steady_clock::now();
    if (now - input_stream_watchdog_start_ < startup_grace)
        return;

    if (!rgbd_stream_seen_.load(std::memory_order_relaxed)
        && !rgbd_stream_wait_logged_.exchange(true, std::memory_order_relaxed))
    {
        std::print(stderr, "[read_rgbd] No RGBD frames received since startup. Waiting for input stream...\n");
    }

    if (!lidar_stream_seen_.load(std::memory_order_relaxed)
        && !lidar_stream_wait_logged_.exchange(true, std::memory_order_relaxed))
    {
        std::print(stderr, "[read_lidar] No LiDAR frames received since startup. Waiting for input stream...\n");
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

void SceneProcessor::overlay_room_polygon_on_canvas(cv::Mat& canvas,
                                                    const RoboCompCameraRGBDSimple::TRGBD& rgbd) const
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

    const std::uint64_t frame_ts_ms = get_rgbd_frame_timestamp_ms(rgbd);
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

    std::vector<float> xs;
    std::vector<float> ys;
    std::vector<float> zs;
    std::uint64_t lidar_timestamp_ms = 0;
    {
        std::scoped_lock lk(lidar_points_mutex_);
        xs = latest_lidar_xs_;
        ys = latest_lidar_ys_;
        zs = latest_lidar_zs_;
        lidar_timestamp_ms = latest_lidar_timestamp_ms_;
    }

    if (xs.empty() || ys.size() != xs.size() || zs.size() != xs.size())
    {
        voxel_viewer_->update_lidar_points({});
        return;
    }

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