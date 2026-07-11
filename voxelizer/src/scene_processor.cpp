#include "scene_processor.h"

#include "rgbd_data.h"
#include "voxel_opengl_viewer.h"

#include "../../common/media_transport/media_transport.h"
#include "../../common/media_transport/lidar_plane_reader.h"

#include <dsr/api/dsr_camera_api.h>

#include <Eigen/Geometry>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <print>
#include <sstream>
#include <string_view>
#include <variant>

SceneProcessor::SceneProcessor(const std::shared_ptr<DSR::DSRGraph>& graph)
    : graph_(graph)
    , media_(std::make_unique<MediaPlaneSource>(graph))
{
}

SceneProcessor::~SceneProcessor() = default;

bool SceneProcessor::init_media_plane(std::uint32_t domain_id, const std::string& rgb_topic, const std::string& depth_topic)
{
    return media_->init_media_plane(domain_id, rgb_topic, depth_topic);
}

bool SceneProcessor::init_ricoh_media_plane(std::uint32_t domain_id, const std::string& topic)
{
    return media_->init_ricoh_media_plane(domain_id, topic);
}

void SceneProcessor::poll_ricoh(bool force)
{
    media_->poll_ricoh(force);
}

cv::Mat SceneProcessor::ricoh_bgr_copy() const
{
    return media_->ricoh_bgr_copy();
}

bool SceneProcessor::init_lidar_media_plane(std::uint32_t domain_id, const std::string& topic, bool use_media)
{
    // inner_eigen_api_ is set in configure() (called before this) and backs the device->robot RT.
    return media_->init_lidar_media_plane(inner_eigen_api_, domain_id, topic, use_media);
}

std::optional<LidarData> SceneProcessor::get_lidar3D()
{
    std::string robot_name;
    { std::scoped_lock lk(node_names_mutex_); robot_name = robot_node_name_; }
    return media_->get_lidar3D(robot_name);
}

void SceneProcessor::configure(DSR::InnerEigenAPI* inner_eigen_api,
                               rc::VoxelOpenGLViewer* voxel_viewer,
                               bool transforms_interpolate_rt,
                               bool verbose_debug,
                               bool mask_pose_extrapolate,
                               float mask_pose_extrap_max_dt_s)
{
    inner_eigen_api_ = inner_eigen_api;
    voxel_viewer_ = voxel_viewer;
    transforms_interpolate_rt_ = transforms_interpolate_rt;
    verbose_debug_ = verbose_debug;
    mask_pose_extrapolate_ = mask_pose_extrapolate;
    mask_pose_extrap_max_dt_s_ = mask_pose_extrap_max_dt_s;
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
        static auto last = std::chrono::steady_clock::time_point{};
        if (const auto now = std::chrono::steady_clock::now(); now - last >= std::chrono::seconds(2))
        {
            last = now;
            std::println("[RT] get_transformation_matrix('{}'<-'{}', ts={}) FAILED (room->robot)",
                         room_name, robot_name, timestamp_ms);
        }
        room_rt_ready_logged_ = false;
        if (verbose_debug_)
            compute_fps.print("[Compute]", 2000);
        return std::nullopt;
    }

    // DSR's InterpolatedRT clamps the pose to the newest RT block, which lags the camera capture stamp
    // by ~100 ms — so masks would deproject against a stale robot pose. Extrapolate the pose FORWARD to
    // the capture stamp using the body-frame velocity room_concept writes on the robot→room RT edge
    // (rt_translation_velocity=[adv,side,0], rt_rotation_euler_xyz_velocity=[0,0,rot]). Consumer-side,
    // no producer rate change — the same efference-copy trick the controller uses for its overlay.
    if (mask_pose_extrapolate_ && graph_ && graph_->get_rt_api())
    {
        const auto robot_node = graph_->get_node(robot_name);
        const auto room_node  = graph_->get_node(room_name);
        if (robot_node.has_value() && room_node.has_value())
        {
            if (auto edge = graph_->get_rt_api()->get_edge_RT(robot_node.value(), room_node.value().id());
                edge.has_value())
            {
                const auto ts = graph_->get_attrib_by_name<rt_timestamps_att>(edge.value());
                const auto tv = graph_->get_attrib_by_name<rt_translation_velocity_att>(edge.value());
                const auto rv = graph_->get_attrib_by_name<rt_rotation_euler_xyz_velocity_att>(edge.value());
                if (ts.has_value() && tv.has_value() && tv->get().size() >= 2
                    && rv.has_value() && rv->get().size() >= 3)
                {
                    std::uint64_t newest = 0;
                    for (const auto t : ts->get())
                        newest = std::max(newest, t);
                    if (newest > 0 && timestamp_ms > newest)
                    {
                        float dt = static_cast<float>(timestamp_ms - newest) * 1e-3f;
                        dt = std::min(dt, mask_pose_extrap_max_dt_s_);
                        const double adv = tv->get()[0], side = tv->get()[1], rot = rv->get()[2];
                        auto& T = room_T_robot.value();
                        const Eigen::Matrix3d R = T.linear();
                        const double th  = std::atan2(R(1, 0), R(0, 0));
                        const double raw_x = T.translation().x(), raw_y = T.translation().y();
                        const double dth = rot * dt;
                        const double thm = th + 0.5 * dth;   // midpoint integration
                        const double dx = (adv * std::cos(thm) - side * std::sin(thm)) * dt;
                        const double dy = (adv * std::sin(thm) + side * std::cos(thm)) * dt;
                        T.translation().x() += dx;
                        T.translation().y() += dy;
                        T.linear() = (Eigen::AngleAxisd(dth, Eigen::Vector3d::UnitZ()).toRotationMatrix() * R);

                        // Log raw vs extrapolated pose + dt/velocity/displacement for analysis.
                        if (!pose_extrap_csv_open_attempted_)
                        {
                            pose_extrap_csv_open_attempted_ = true;
                            pose_extrap_csv_.open("etc/pose_extrap_log.csv", std::ios::out | std::ios::trunc);
                            if (pose_extrap_csv_.is_open())
                                pose_extrap_csv_ << "frame_ts_ms,newest_block_ms,dt_s,adv,side,rot,"
                                                    "raw_x,raw_y,raw_th,ext_x,ext_y,ext_th,disp_m,dtheta_rad\n";
                        }
                        if (pose_extrap_csv_.is_open())
                        {
                            pose_extrap_csv_ << timestamp_ms << ',' << newest << ',' << dt << ','
                                             << adv << ',' << side << ',' << rot << ','
                                             << raw_x << ',' << raw_y << ',' << th << ','
                                             << (raw_x + dx) << ',' << (raw_y + dy) << ',' << (th + dth) << ','
                                             << std::hypot(dx, dy) << ',' << dth << '\n';
                            pose_extrap_csv_.flush();
                        }
                    }
                }
            }
        }
    }

    return room_T_robot;
}

std::optional<Mat::RTMat> SceneProcessor::get_room_zed_transform(FPSCounter& compute_fps,
                                                                 const std::string& robot_name,
                                                                 const Mat::RTMat& room_T_robot)
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

    // robot→zed is the rigid camera mount. It carries only its bootstrap timestamp, so a
    // Nearest query pinned to a per-frame timestamp fails on it — query "latest" (0). The
    // robot pose (room→robot) is already resolved at frame time by the caller, so this stays
    // correct when the room/robot become dynamic without any per-frame timestamp here.
    auto robot_T_zed = inner_eigen_api_->get_transformation_matrix(robot_name, "zed", 0);
    if (!robot_T_zed.has_value())
    {
        static auto last = std::chrono::steady_clock::time_point{};
        if (const auto now = std::chrono::steady_clock::now(); now - last >= std::chrono::seconds(2))
        {
            last = now;
            // Re-probe the chain each tick: localize the broken node/hop in robot→body→zed.
            const bool has_body = graph_ and graph_->get_node("body").has_value();
            const bool has_zed  = graph_ and graph_->get_node("zed").has_value();
            const bool sh_body  = inner_eigen_api_->get_transformation_matrix(robot_name, "body", 0).has_value();
            const bool body_zed = inner_eigen_api_->get_transformation_matrix("body", "zed", 0).has_value();
            std::println("[RT] get_transformation_matrix('{}'<-'zed', ts=0) FAILED (robot->zed) | "
                         "node(body)={} node(zed)={} {}->body={} body->zed={}",
                         robot_name, has_body, has_zed, robot_name, sh_body, body_zed);
        }
        room_rt_ready_logged_ = false;
        if (verbose_debug_)
            compute_fps.print("[Compute]", 2000);
        return std::nullopt;
    }

    return room_T_robot * robot_T_zed.value();
}

std::uint64_t SceneProcessor::get_frame_timestamp_ms() const
{
    return media_->get_frame_timestamp_ms();
}

std::optional<RGBDData> SceneProcessor::get_rgbd_frame_from_dsr() const
{
    return media_->get_rgbd_frame_from_dsr();
}

void SceneProcessor::check_input_stream_startup_status()
{
    constexpr auto startup_grace = std::chrono::seconds(3);
    const auto now = std::chrono::steady_clock::now();
    if (now - input_stream_watchdog_start_ < startup_grace)
        return;

    if (graph_)
    {
        if (!media_->rgb_valid())
            std::print(stderr, "[voxelizer] No RGB frame on the media plane yet. Waiting for robot_concept producer...\n");
        if (!media_->depth_valid())
            std::print(stderr, "[voxelizer] No depth frame on the media plane yet. Waiting for robot_concept producer...\n");

        if (auto zed = graph_->get_node("zed"); !zed.has_value())
            std::print(stderr, "[voxelizer] DSR 'zed' node not found. Waiting...\n");
    }
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

    // Furniture like the table stands ON the floor: its node origin is the base, so the box
    // must extend upward (z in [origin, origin+height]). Free objects (e.g. a fitted bottle
    // cylinder) are center-anchored, so they keep z in [origin-h/2, origin+h/2].
    const bool stands_on_floor = (node.type() == "table") or (node.name().rfind("table", 0) == 0);
    const float z_lo = stands_on_floor ? 0.f : -half_height;
    const float z_hi = stands_on_floor ? height : half_height;

    const std::array<Eigen::Vector3d, 8> local_corners = {
        Eigen::Vector3d{-half_width, -half_depth, z_lo},
        Eigen::Vector3d{ half_width, -half_depth, z_lo},
        Eigen::Vector3d{ half_width,  half_depth, z_lo},
        Eigen::Vector3d{-half_width,  half_depth, z_lo},
        Eigen::Vector3d{-half_width, -half_depth, z_hi},
        Eigen::Vector3d{ half_width, -half_depth, z_hi},
        Eigen::Vector3d{ half_width,  half_depth, z_hi},
        Eigen::Vector3d{-half_width,  half_depth, z_hi}
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
    // residual_concept obstacles (type "obstacle", named residual_*) → their own category/colour.
    if (node.type() == "obstacle")
        category = "obstacle";

    // bottle_concept publishes bottles as `cylinder` nodes (bottle_1, …); tag them so
    // the viewer paints their bounding box in a colour distinct from the table.
    if (node.type() == "cylinder")
        category = "bottle";

    const Eigen::Matrix3d& R = room_T_object->linear();
    const float yaw = static_cast<float>(std::atan2(R(1, 0), R(0, 0)));

    // Room-frame box center. The vertical center is half_height above the node
    // origin for floor-standing furniture, and at the origin for center-anchored
    // objects. A yaw-only rotation leaves z unchanged, so the center's z is simply
    // the translation z plus that local offset.
    const Eigen::Vector3d& t = room_T_object->translation();
    const float z_center_local = stands_on_floor ? half_height : 0.f;
    const Eigen::Vector3f center(static_cast<float>(t.x()),
                                 static_cast<float>(t.y()),
                                 static_cast<float>(t.z()) + z_center_local);

    return GraphObjectBox{min_corner, max_corner, center,
                          Eigen::Vector3f(half_width, half_depth, half_height),
                          yaw, node.name(), std::move(category)};
}

std::vector<GraphObjectBox> SceneProcessor::get_graph_object_boxes(const std::string& room_name,
                                                                   std::uint64_t timestamp_ms) const
{
    std::vector<GraphObjectBox> graph_boxes;
    if (!graph_ || room_name.empty())
        return graph_boxes;

    const auto object_nodes   = graph_->get_nodes_by_type("object");
    const auto table_nodes    = graph_->get_nodes_by_type("table");
    const auto cylinder_nodes = graph_->get_nodes_by_type("cylinder");   // bottle_concept bottles
    // residual_concept `obstacle` nodes are NO LONGER drawn as red boxes — the occupancy grid (amber cells,
    // the `grid` node) is now the residual display. The obstacle nodes still exist for the controller.
    graph_boxes.reserve(object_nodes.size() + table_nodes.size() + cylinder_nodes.size());
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
    add_boxes(cylinder_nodes);
    return graph_boxes;
}

bool SceneProcessor::get_room_layout(std::vector<float>& polygon_x, std::vector<float>& polygon_y,
                                     float& room_height) const
{
    auto room_data = get_room_polygon_from_graph();
    if (!room_data.has_value())
        return false;
    polygon_x = std::move(room_data->polygon_x);
    polygon_y = std::move(room_data->polygon_y);
    room_height = room_data->room_height;
    return true;
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
        voxel_viewer_->update_graph_boxes({}, {}, {}, {});
        return;
    }

    std::vector<QVector3D> centers;
    std::vector<QVector3D> half_extents;
    std::vector<float> yaws;
    std::vector<std::string> categories;
    centers.reserve(graph_boxes.size());
    half_extents.reserve(graph_boxes.size());
    yaws.reserve(graph_boxes.size());
    categories.reserve(graph_boxes.size());

    for (const auto& box : graph_boxes)
    {
        centers.emplace_back(box.center.x(), box.center.y(), box.center.z());
        half_extents.emplace_back(box.half_extents.x(), box.half_extents.y(), box.half_extents.z());
        yaws.push_back(box.yaw_rad);
        categories.push_back(box.category);
    }

    voxel_viewer_->update_graph_boxes(centers, half_extents, yaws, categories);
}

void SceneProcessor::update_viewer_object_meshes()
{
    if (voxel_viewer_ == nullptr || graph_ == nullptr)
        return;

    // Tables and chairs are drawn as the solid mesh (mesh_vertices_att) published by
    // table_concept / chair_concept. The mesh is gated at the source (published only when the fit
    // moves past a few mm/mrad), so it is stable. (The graph BOX is intentionally NOT drawn for
    // these types — see the box pass — so the mesh is their only visual.)
    std::vector<std::vector<float>> meshes;
    std::vector<std::string>        categories;   // parallel to meshes → per-class mesh colour in the viewer
    for (const char* type : {"table", "chair"})
        for (const auto& node : graph_->get_nodes_by_type(type))
            if (const auto opt = graph_->get_attrib_by_name<mesh_vertices_att>(node); opt.has_value())
            {
                meshes.emplace_back(opt.value().get());
                categories.emplace_back(type);
            }
    voxel_viewer_->update_object_meshes(meshes, categories);
}

void SceneProcessor::update_viewer_person_skeletons()
{
    if (voxel_viewer_ == nullptr || graph_ == nullptr)
        return;

    // human_concept writes the fitted BODY_18 skeleton (room frame, 18×3 flat) into mesh_vertices_att
    // of each 'person' node. Draw it directly — same room frame as the voxels/object meshes.
    std::vector<std::vector<float>> skeletons;
    for (const auto& node : graph_->get_nodes_by_type("person"))
        if (const auto opt = graph_->get_attrib_by_name<mesh_vertices_att>(node); opt.has_value())
            skeletons.emplace_back(opt.value().get());
    voxel_viewer_->update_skeletons(skeletons);
}

void SceneProcessor::update_viewer_table_rfe_points()
{
    if (voxel_viewer_ == nullptr || graph_ == nullptr)
        return;

    std::vector<QVector3D> residual_points;
    for (const auto& node : graph_->get_nodes_by_type("table"))
    {
        if (auto residual_opt = graph_->get_attrib_by_name<residual_pts_att>(node); residual_opt.has_value())
        {
            const auto& flat = residual_opt.value().get();
            const std::size_t n = flat.size() / 3;
            residual_points.reserve(residual_points.size() + n);
            for (std::size_t i = 0; i < n; ++i)
            {
                const std::size_t idx = i * 3;
                residual_points.emplace_back(flat[idx], flat[idx + 1], flat[idx + 2]);
            }
        }
    }

    voxel_viewer_->update_residual_points(residual_points);
}

void SceneProcessor::update_viewer_grid()
{
    if (voxel_viewer_ == nullptr || graph_ == nullptr)
        return;
    // residual_concept publishes a `grid` node under room: grid_cells_xy = flat [x0,y0,x1,y1,…] residual cell
    // centres (room frame), grid_cell_size = the square edge. Read them and hand cell centres to the viewer.
    // residual_concept's `grid` node carries two layers of flat x,y,z triples: residual_pts = OCCUPIED cells
    // (colour A), candidate_pts = the INFLATED half-robot-width clearance BORDER (colour B).
    const auto read_pts = [](const auto& flat) {
        std::vector<QVector3D> v; const std::size_t n = flat.size() / 3; v.reserve(n);
        for (std::size_t i = 0; i < n; ++i) v.emplace_back(flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]);
        return v;
    };
    std::vector<QVector3D> cells, border;
    std::vector<QVector3D> field_centres;
    std::vector<float> field_prob, field_var;
    if (const auto g = graph_->get_node("grid"); g.has_value())
    {
        if (const auto o = graph_->get_attrib_by_name<grid_occupied_cells_att>(g.value()); o.has_value()) cells  = read_pts(o.value().get());
        if (const auto o = graph_->get_attrib_by_name<grid_border_cells_att>  (g.value()); o.has_value()) border = read_pts(o.value().get());
        // Beta BELIEF FIELD: dense row-major P and Var + meta=[xmin,ymin,cell,w,h]. Reconstruct each cell's room
        // centre from the meta and index — only cells with meaningful occupancy are drawn (filtered in the viewer).
        const auto pa = graph_->get_attrib_by_name<grid_occupancy_prob_att>(g.value());
        const auto va = graph_->get_attrib_by_name<grid_occupancy_var_att> (g.value());
        const auto ma = graph_->get_attrib_by_name<grid_field_meta_att>     (g.value());
        if (pa.has_value() and va.has_value() and ma.has_value())
        {
            const auto& P = pa.value().get(); const auto& V = va.value().get(); const auto& M = ma.value().get();
            if (M.size() >= 5)
            {
                const float xmin = M[0], ymin = M[1], cell = M[2];
                const int w = static_cast<int>(M[3]), h = static_cast<int>(M[4]);
                if (static_cast<int>(P.size()) >= w * h and static_cast<int>(V.size()) >= w * h)
                {
                    field_centres.reserve(P.size()); field_prob.reserve(P.size()); field_var.reserve(P.size());
                    for (int y = 0; y < h; ++y)
                        for (int x = 0; x < w; ++x)
                        {
                            const int i = y * w + x;
                            if (P[i] <= 0.5f) continue;                 // collapsed/free → skip (keeps payload small)
                            field_centres.emplace_back(xmin + (x + 0.5f) * cell, ymin + (y + 0.5f) * cell, 0.03f);
                            field_prob.push_back(P[i]); field_var.push_back(V[i]);
                        }
                }
            }
        }
    }
    voxel_viewer_->update_grid_cells(cells, 0.05f);
    voxel_viewer_->update_grid_border(border);
    voxel_viewer_->update_grid_field(field_centres, field_prob, field_var);
}

void SceneProcessor::update_viewer_mask_points()
{
    if (voxel_viewer_ == nullptr || graph_ == nullptr)
        return;

    // The masks node carries the YOLO support points (flat xyz, room frame) as a runtime
    // attribute. Draw the object slices — "bottle" (bottle_concept), "table" (table_concept) and
    // "chair" — so the Masks toggle shows them; other detections are clutter here. Each is coloured
    // by color_for_category (table green, bottle red, chair cyan). Per-mask point ranges come from
    // mask_support_offsets; the i-th label in mask_labels ('|'-joined) owns range [offsets[i], offsets[i+1]).
    static const std::array<std::string_view, 3> kDrawnMaskLabels{"bottle", "table", "chair"};
    std::vector<QVector3D>   mask_points;
    std::vector<std::string> mask_categories;   // parallel to mask_points → per-class colour in the viewer
    std::vector<float>       mask_sources;       // parallel to mask_points → sensor source (0=zed, 1=ricoh) → brightness
    if (const auto masks_node = graph_->get_node("masks"); masks_node.has_value())
    {
        const auto& attrs = masks_node->attrs();
        const auto pts_it     = attrs.find("mask_support_points");
        const auto off_it     = attrs.find("mask_support_offsets");
        const auto labels_it  = attrs.find("mask_labels");
        const auto src_it     = attrs.find("mask_source");   // optional (older producers omit it → treated as zed/bright)
        if (pts_it != attrs.end() and off_it != attrs.end() and labels_it != attrs.end())
        {
            const auto& flat    = pts_it->second.float_vec();
            const auto& offsets = off_it->second.float_vec();
            const std::vector<float> empty_src;
            const auto& sources = (src_it != attrs.end()) ? src_it->second.float_vec() : empty_src;

            std::vector<std::string> labels;
            std::stringstream ls(labels_it->second.str());
            for (std::string lbl; std::getline(ls, lbl, '|'); )
                labels.push_back(lbl);

            const std::size_t n_masks = labels.size();
            for (std::size_t m = 0; m < n_masks and m + 1 < offsets.size(); ++m)
            {
                if (std::find(kDrawnMaskLabels.begin(), kDrawnMaskLabels.end(), labels[m]) == kDrawnMaskLabels.end())
                    continue;
                const std::size_t begin = static_cast<std::size_t>(offsets[m]);
                const std::size_t end   = static_cast<std::size_t>(offsets[m + 1]);
                const float src = (m < sources.size()) ? sources[m] : 0.0f;   // default zed (bright)
                for (std::size_t i = begin; i < end and (i * 3 + 2) < flat.size(); ++i)
                {
                    mask_points.emplace_back(flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]);
                    mask_categories.push_back(labels[m]);
                    mask_sources.push_back(src);
                }
            }
        }
    }
    voxel_viewer_->update_mask_points(mask_points, mask_categories, mask_sources);
}

void SceneProcessor::update_viewer_robot_pose(const Mat::RTMat& room_T_robot)
{
    if (voxel_viewer_ == nullptr)
        return;
    const auto& t = room_T_robot.translation();
    const Eigen::Matrix3d R = room_T_robot.rotation();
    const float tx = static_cast<float>(t.x());
    const float ty = static_cast<float>(t.y());
    const float ttheta = static_cast<float>(std::atan2(R(1, 0), R(0, 0)));

    // Time-constant EMA (rate-independent: works whether fed at the 60 Hz render tick or 16 Hz frame
    // path). tau ≈ correction interval so prediction noise / correction snaps are averaged out while
    // real motion still tracks with only ~tau of display lag. tau <= 0 would disable smoothing.
    constexpr float kTauS = 0.12f;
    if (not robot_disp_sm_init_)
    {
        robot_disp_sm_x_ = tx;
        robot_disp_sm_y_ = ty;
        robot_disp_sm_theta_ = ttheta;
        robot_disp_sm_last_ = std::chrono::steady_clock::now();
        robot_disp_sm_init_ = true;
    }
    else
    {
        const auto now = std::chrono::steady_clock::now();
        const float dt = std::chrono::duration<float>(now - robot_disp_sm_last_).count();
        robot_disp_sm_last_ = now;
        const float alpha = (kTauS > 0.f and dt > 0.f) ? (1.f - std::exp(-dt / kTauS)) : 1.f;
        robot_disp_sm_x_ += alpha * (tx - robot_disp_sm_x_);
        robot_disp_sm_y_ += alpha * (ty - robot_disp_sm_y_);
        // Wrap the angle delta into (-π, π] before blending so it smooths across the ±π seam.
        float dth = ttheta - robot_disp_sm_theta_;
        while (dth >  static_cast<float>(M_PI)) dth -= 2.f * static_cast<float>(M_PI);
        while (dth < -static_cast<float>(M_PI)) dth += 2.f * static_cast<float>(M_PI);
        robot_disp_sm_theta_ += alpha * dth;
    }
    voxel_viewer_->set_robot_pose(robot_disp_sm_x_, robot_disp_sm_y_, robot_disp_sm_theta_);
}

void SceneProcessor::refresh_viewer_robot_pose_latest()
{
    if (inner_eigen_api_ == nullptr || voxel_viewer_ == nullptr)
        return;
    const auto [room_name, robot_name] = get_room_robot_names_for_compute();
    if (room_name.empty() || robot_name.empty())
        return;
    // ts=0 → latest RT (not pinned to the camera frame stamp), so the robot tracks live odometry
    // at the render-timer rate rather than stepping at the ~7-10 Hz perception cadence.
    if (auto room_T_robot = inner_eigen_api_->get_transformation_matrix(room_name, robot_name, 0);
        room_T_robot.has_value())
        update_viewer_robot_pose(room_T_robot.value());
}

void SceneProcessor::update_viewer_lidar_points(std::span<const Eigen::Vector3f> lidar_points_room,
                                                std::span<const std::uint8_t> plane_id)
{
    if (voxel_viewer_ == nullptr)
        return;
    // Empty ⇒ no fresh sweep drained this cycle: keep the last cloud (never blank it → no flicker). The
    // caller (compute) already drained get_lidar3D() ONCE and posed the cloud into the room frame at the
    // scan stamp; re-draining/re-interpolating here desynced the viewer and shimmered the cloud.
    if (lidar_points_room.empty())
        return;

    std::vector<QVector3D> pts;
    pts.reserve(lidar_points_room.size());
    for (const auto& p : lidar_points_room)
        pts.emplace_back(p.x(), p.y(), p.z());

    voxel_viewer_->update_lidar_points(pts, plane_id);
}