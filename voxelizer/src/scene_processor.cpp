#include "scene_processor.h"

#include "rgbd_data.h"
#include "voxel_opengl_viewer.h"

#include "../../common/media_transport/media_transport.h"

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
{
}

SceneProcessor::~SceneProcessor() = default;

bool SceneProcessor::init_media_plane(std::uint32_t domain_id,
                                      const std::string& rgb_topic,
                                      const std::string& depth_topic)
{
    media_rgb_sub_   = std::make_unique<rc::media::MediaSubscriber>();
    media_depth_sub_ = std::make_unique<rc::media::MediaSubscriber>();

    rc::media::SubscriberConfig rgb_cfg;
    rgb_cfg.domain_id     = domain_id;
    rgb_cfg.topic_name    = rgb_topic;
    rgb_cfg.history_depth = 8;
    rc::media::SubscriberConfig depth_cfg = rgb_cfg;
    depth_cfg.topic_name = depth_topic;

    // Prefer the producer's media descriptor on the "zed" node (its authored domain/topic, on the
    // dedicated media domain isolated from Agent.domain) — the same discovery the other agents use;
    // fall back to the configured domain/topic if the descriptor isn't advertised yet.
    if (graph_)
        if (auto desc = rc::media::descriptor_from_graph(*graph_, "zed"); desc.has_value())
        {
            if (auto c = desc->subscriber_config("rgb");   c.has_value()) { rgb_cfg.domain_id = c->domain_id;   rgb_cfg.topic_name = c->topic_name; }
            if (auto c = desc->subscriber_config("depth"); c.has_value()) { depth_cfg.domain_id = c->domain_id; depth_cfg.topic_name = c->topic_name; }
        }

    const bool rgb_ok   = media_rgb_sub_->init(rgb_cfg);
    const bool depth_ok = media_depth_sub_->init(depth_cfg);
    if (!rgb_ok || !depth_ok)
    {
        std::print(stderr, "[voxelizer] media plane subscriber init FAILED (rgb={}, depth={})\n", rgb_ok, depth_ok);
        return false;
    }
    std::print("[voxelizer] media plane ready rgb domain={} '{}' | depth domain={} '{}' data_sharing={}\n",
               rgb_cfg.domain_id, rgb_cfg.topic_name, depth_cfg.domain_id, depth_cfg.topic_name,
               media_rgb_sub_->data_sharing_active() && media_depth_sub_->data_sharing_active());
    return true;
}

void SceneProcessor::drain_media_plane() const
{
    // Diagnostic: positively confirm media-plane reception (mirrors robot_concept's
    // producer-side "[Media] 5s stats"). Accumulate poll() delivery counts and print
    // every 5 s on stdout, alongside [Tracks].
    static std::uint64_t rx_rgb = 0, rx_depth = 0;
    static auto last_rx_report = std::chrono::steady_clock::now();

    if (media_rgb_sub_)
    {
        rx_rgb += media_rgb_sub_->poll([this](const rc::media::ImageFrame& f, std::int64_t)
        {
            const int w = static_cast<int>(f.width());
            const int h = static_cast<int>(f.height());
            if (w <= 0 || h <= 0)
                return;
            const std::size_t npix = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
            if (f.format() == rc::media::FORMAT_BGR8 || f.format() == rc::media::FORMAT_RGB8)
            {
                if (f.size() < npix * 3)
                    return;
                cv::Mat view(h, w, CV_8UC3, const_cast<std::uint8_t*>(f.data().data()));
                if (f.format() == rc::media::FORMAT_RGB8)
                    cv::cvtColor(view, media_rgb_.bgr, cv::COLOR_RGB2BGR);
                else
                    media_rgb_.bgr = view.clone();
                media_rgb_.width    = w;
                media_rgb_.height   = h;
                media_rgb_.stamp    = f.stamp_ms();
                media_rgb_.frame_id = f.frame_id();
                media_rgb_.valid    = true;
            }
        });
    }

    if (media_depth_sub_)
    {
        rx_depth += media_depth_sub_->poll([this](const rc::media::ImageFrame& f, std::int64_t)
        {
            const int w = static_cast<int>(f.width());
            const int h = static_cast<int>(f.height());
            if (w <= 0 || h <= 0)
                return;
            const std::size_t npix = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
            if (f.format() == rc::media::FORMAT_DEPTH_F32)
            {
                if (f.size() < npix * sizeof(float))
                    return;
                const float* p = reinterpret_cast<const float*>(f.data().data());
                media_depth_.depth.assign(p, p + npix);
            }
            else if (f.format() == rc::media::FORMAT_Z16)
            {
                if (f.size() < npix * sizeof(std::uint16_t))
                    return;
                const std::uint16_t* p = reinterpret_cast<const std::uint16_t*>(f.data().data());
                media_depth_.depth.resize(npix);
                for (std::size_t i = 0; i < npix; ++i)
                    media_depth_.depth[i] = static_cast<float>(p[i]) * 0.001f;  // mm -> m fallback
            }
            else
                return;
            media_depth_.width    = w;
            media_depth_.height   = h;
            media_depth_.stamp    = f.stamp_ms();
            media_depth_.frame_id = f.frame_id();
            media_depth_.valid    = true;
        });
    }

    const auto now_rx = std::chrono::steady_clock::now();
    if (now_rx - last_rx_report >= std::chrono::seconds(5))
    {
        std::println("[MediaRx] 5s stats rgb={} depth={} rgb_valid={} depth_valid={} ({}x{} / {}x{})",
                     rx_rgb, rx_depth, media_rgb_.valid, media_depth_.valid,
                     media_rgb_.width, media_rgb_.height, media_depth_.width, media_depth_.height);
        rx_rgb = rx_depth = 0;
        last_rx_report = now_rx;
    }
}

bool SceneProcessor::init_lidar_media_plane(std::uint32_t domain_id, const std::string& topic, bool use_media)
{
    lidar_use_media_ = use_media;
    if (!use_media)
    {
        std::print("[voxelizer] lidar media plane DISABLED (Voxel.lidar_use_media=false) — no LiDAR source\n");
        return false;
    }

    rc::media::SubscriberConfig cfg;
    cfg.domain_id     = domain_id;
    cfg.topic_name    = topic;
    cfg.history_depth = 4;
    // Prefer the producer's media descriptor on the "lidar3D" node (its authored domain/topic) — the
    // same discovery the other agents use; fall back to the configured domain/topic if not advertised.
    if (graph_)
        if (auto desc = rc::media::descriptor_from_graph(*graph_, "lidar3D"); desc.has_value())
            if (auto c = desc->subscriber_config("lidar"); c.has_value())
            {
                cfg.domain_id  = c->domain_id;
                cfg.topic_name = c->topic_name;
            }

    lidar_sub_ = std::make_unique<rc::media::LidarSubscriber>();
    if (!lidar_sub_->init(cfg))
    {
        std::print(stderr, "[voxelizer] lidar media subscriber init FAILED — no LiDAR source\n");
        lidar_sub_.reset();
        lidar_use_media_ = false;
        return false;
    }
    std::print("[voxelizer] lidar media plane ready domain={} topic='{}' data_sharing={}\n",
               cfg.domain_id, cfg.topic_name, lidar_sub_->data_sharing_active());
    return true;
}

std::optional<SceneProcessor::LidarData> SceneProcessor::get_lidar3D()
{
    if (!lidar_use_media_ || !lidar_sub_)
        return std::nullopt;

    // Diagnostic (every 5 s): fresh = LidarFrames drained this window; served = cycles that returned a
    // scan. fresh==0 while served>0 ⇒ serving a STALE scan (producer stopped publishing).
    static std::uint64_t fresh = 0, served = 0;
    static auto last_report = std::chrono::steady_clock::now();

    // Drain the media plane (non-blocking) and keep the latest scan.
    const int got = lidar_sub_->poll([this](const rc::media::LidarFrame& f, std::int64_t)
    {
        const std::uint32_t stride = f.stride() ? f.stride() : 3u;
        const auto& pts = f.points();
        const std::uint32_t count = f.count();
        LidarData ld;
        ld.xs.reserve(count);
        ld.ys.reserve(count);
        ld.zs.reserve(count);
        for (std::uint32_t i = 0; i < count; ++i)
        {
            const std::size_t base = static_cast<std::size_t>(i) * stride;
            if (base + 2 >= pts.size())
                break;
            ld.xs.push_back(pts[base]);
            ld.ys.push_back(pts[base + 1]);
            ld.zs.push_back(pts[base + 2]);
        }
        ld.timestamp_ms = f.stamp_ms();
        media_lidar_ = std::move(ld);
        media_lidar_valid_ = true;
    });
    fresh += static_cast<std::uint64_t>(std::max(0, got));

    std::optional<LidarData> out;
    if (media_lidar_valid_ && !media_lidar_.xs.empty())
    {
        out = media_lidar_;
        ++served;
    }

    const auto now = std::chrono::steady_clock::now();
    if (now - last_report >= std::chrono::seconds(5))
    {
        std::println("[LidarSrc] 5s media fresh={} served={} ({} pts)",
                     fresh, served, out ? out->xs.size() : 0u);
        fresh = served = 0;
        last_report = now;
    }
    return out;
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
    if (media_rgb_.valid && media_rgb_.stamp != 0)
        return media_rgb_.stamp;
    else if (media_depth_.valid && media_depth_.stamp != 0)
        return media_depth_.stamp;
    else
    {
        return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    }
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

std::optional<RGBDData> SceneProcessor::get_rgbd_frame_from_dsr() const
{
    // Pixels now arrive over the zero-copy media plane (not the DSR graph). Pull
    // the latest RGB and depth frames; camera intrinsics (focal) remain in the
    // static 'zed' node descriptor.
    drain_media_plane();

    if (!media_rgb_.valid || media_rgb_.bgr.empty())
        return std::nullopt;
    if (!media_depth_.valid || media_depth_.depth.empty())
        return std::nullopt;

    const int rgb_w = media_rgb_.width;
    const int rgb_h = media_rgb_.height;
    const int depth_w = media_depth_.width;
    const int depth_h = media_depth_.height;
    if (rgb_w <= 0 || rgb_h <= 0 || depth_w <= 0 || depth_h <= 0)
        return std::nullopt;

    // Voxel pipeline assumes one xyz point per RGB pixel.
    if (depth_w != rgb_w || depth_h != rgb_h)
    {
        qWarning() << "RGB/depth resolution mismatch. RGB=" << rgb_w << "x" << rgb_h
                   << " depth=" << depth_w << "x" << depth_h;
        return std::nullopt;
    }

    const std::size_t depth_size = static_cast<std::size_t>(depth_w) * static_cast<std::size_t>(depth_h);
    if (media_depth_.depth.size() < depth_size)
        return std::nullopt;

    // Camera intrinsics from the static 'zed' node (not per-frame).
    float focal_x = 0.f;
    float focal_y = 0.f;
    if (graph_)
    {
        if (auto zed_node = graph_->get_node("zed"); zed_node.has_value())
        {
            if (auto fx = graph_->get_attrib_by_name<cam_depth_focalx_att>(zed_node.value()); fx.has_value())
                focal_x = static_cast<float>(fx.value());
            if (auto fy = graph_->get_attrib_by_name<cam_depth_focaly_att>(zed_node.value()); fy.has_value())
                focal_y = static_cast<float>(fy.value());
        }
    }
    if (focal_x <= 0.f || focal_y <= 0.f)
        return std::nullopt;

    RGBDData data;
    data.rgb     = media_rgb_.bgr.clone();
    data.width   = rgb_w;
    data.height  = rgb_h;
    data.focal_x = focal_x;
    data.focal_y = focal_y;
    data.depth.assign(media_depth_.depth.begin(),
                      media_depth_.depth.begin() + static_cast<std::ptrdiff_t>(depth_size));
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
        if (!media_rgb_.valid)
            std::print(stderr, "[voxelizer] No RGB frame on the media plane yet. Waiting for robot_concept producer...\n");
        if (!media_depth_.valid)
            std::print(stderr, "[voxelizer] No depth frame on the media plane yet. Waiting for robot_concept producer...\n");

        if (auto zed = graph_->get_node("zed"); !zed.has_value())
            std::print(stderr, "[voxelizer] DSR 'zed' node not found. Waiting...\n");
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

    std::vector<std::vector<float>> meshes;
    for (const auto& node : graph_->get_nodes_by_type("table"))
    {
        auto opt = graph_->get_attrib_by_name<mesh_vertices_att>(node);
        if (opt.has_value())
            meshes.push_back(std::vector<float>(opt.value().get()));
    }
    voxel_viewer_->update_object_meshes(meshes);
}

void SceneProcessor::update_viewer_table_rfe_points()
{
    if (voxel_viewer_ == nullptr || graph_ == nullptr)
        return;

    static int debug_frame_counter = 0;
    std::vector<QVector3D> residual_points;
    std::vector<QVector3D> rfe_points;
    std::vector<QVector3D> candidate_points;
    std::size_t tables_seen = 0;
    std::size_t tables_with_residual = 0;
    std::size_t tables_with_rfe = 0;
    std::size_t tables_with_candidates = 0;
    std::size_t residual_point_count = 0;
    std::size_t rfe_point_count = 0;
    std::size_t candidate_point_count = 0;

    for (const auto& node : graph_->get_nodes_by_type("table"))
    {
        ++tables_seen;

        if (auto residual_opt = graph_->get_attrib_by_name<residual_pts_att>(node); residual_opt.has_value())
        {
            ++tables_with_residual;
            const auto& flat = residual_opt.value().get();
            const std::size_t n = flat.size() / 3;
            residual_point_count += n;
            residual_points.reserve(residual_points.size() + n);
            for (std::size_t i = 0; i < n; ++i)
            {
                const std::size_t idx = i * 3;
                residual_points.emplace_back(flat[idx], flat[idx + 1], flat[idx + 2]);
            }
        }

        if (auto rfe_opt = graph_->get_attrib_by_name<rfe_pts_att>(node); rfe_opt.has_value())
        {
            ++tables_with_rfe;
            const auto& flat = rfe_opt.value().get();
            const std::size_t n = flat.size() / 3;
            rfe_point_count += n;
            rfe_points.reserve(rfe_points.size() + n);
            for (std::size_t i = 0; i < n; ++i)
            {
                const std::size_t idx = i * 3;
                rfe_points.emplace_back(flat[idx], flat[idx + 1], flat[idx + 2]);
            }
        }

        if (auto candidate_opt = graph_->get_attrib_by_name<candidate_pts_att>(node); candidate_opt.has_value())
        {
            ++tables_with_candidates;
            const auto& flat = candidate_opt.value().get();
            const std::size_t n = flat.size() / 3;
            candidate_point_count += n;
            candidate_points.reserve(candidate_points.size() + n);
            for (std::size_t i = 0; i < n; ++i)
            {
                const std::size_t idx = i * 3;
                candidate_points.emplace_back(flat[idx], flat[idx + 1], flat[idx + 2]);
            }
        }
    }

    voxel_viewer_->update_rfe_points(residual_points, rfe_points, candidate_points);
}

void SceneProcessor::update_viewer_mask_points()
{
    if (voxel_viewer_ == nullptr || graph_ == nullptr)
        return;

    // The masks node carries the YOLO support points (flat xyz, room frame) as a runtime
    // attribute. Draw the slices that downstream fitters actually consume — "bottle"
    // (bottle_concept) and "table" (table_concept) — so the Masks toggle shows exactly the
    // support points reaching those agents; other detections are clutter here. Per-mask
    // point ranges come from mask_support_offsets; the i-th label in mask_labels
    // ('|'-joined) owns range [offsets[i], offsets[i+1]).
    static const std::array<std::string_view, 2> kDrawnMaskLabels{"bottle", "table"};
    std::vector<QVector3D>   mask_points;
    std::vector<std::string> mask_categories;   // parallel to mask_points → per-class colour in the viewer
    if (const auto masks_node = graph_->get_node("masks"); masks_node.has_value())
    {
        const auto& attrs = masks_node->attrs();
        const auto pts_it     = attrs.find("mask_support_points");
        const auto off_it     = attrs.find("mask_support_offsets");
        const auto labels_it  = attrs.find("mask_labels");
        if (pts_it != attrs.end() and off_it != attrs.end() and labels_it != attrs.end())
        {
            const auto& flat    = pts_it->second.float_vec();
            const auto& offsets = off_it->second.float_vec();

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
                for (std::size_t i = begin; i < end and (i * 3 + 2) < flat.size(); ++i)
                {
                    mask_points.emplace_back(flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]);
                    mask_categories.push_back(labels[m]);
                }
            }
        }
    }
    voxel_viewer_->update_mask_points(mask_points, mask_categories);
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

    const auto lidar_data = get_lidar3D();
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