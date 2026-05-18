#include "camera_visualizer.h"

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_camera_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/core/types/type_checking/dsr_attr_name.h>

#include <QPainter>
#include <QVBoxLayout>

#include <cmath>
#include <limits>

namespace rc {

static std::vector<Eigen::Vector2f> read_room_polygon_from_dsr(const std::shared_ptr<DSRGraph>& graph, const std::string& room_frame_name)
{
    if (!graph)
        return {};

    const auto room_node_opt = graph->get_node(room_frame_name);
    if (!room_node_opt.has_value())
        return {};

    const auto polygon_x_opt = graph->get_attrib_by_name<delimiting_polygon_x_att>(room_node_opt.value());
    const auto polygon_y_opt = graph->get_attrib_by_name<delimiting_polygon_y_att>(room_node_opt.value());
    if (!polygon_x_opt.has_value() || !polygon_y_opt.has_value())
        return {};

    const auto& polygon_x = polygon_x_opt.value();
    const auto& polygon_y = polygon_y_opt.value();
    if (polygon_x.get().size() < 3 || polygon_x.get().size() != polygon_y.get().size())
        return {};

    std::vector<Eigen::Vector2f> polygon;
    polygon.reserve(polygon_x.get().size());
    for (std::size_t i = 0; i < polygon_x.get().size(); ++i)
        polygon.emplace_back(polygon_x.get()[i], polygon_y.get()[i]);

    return polygon;
}

CameraVisualizer::CameraVisualizer(std::shared_ptr<DSRGraph> graph, const std::vector<Eigen::Vector2f>& room_polygon, QWidget* parent)
    : QDialog(parent), graph_(graph), room_polygon_(room_polygon)
{
    setWindowTitle("Camera Visualization - Room Layout Projection");
    setGeometry(100, 100, 800, 600);

    auto* layout = new QVBoxLayout(this);
    image_label_ = new QLabel();
    image_label_->setMinimumSize(640, 480);
    image_label_->setScaledContents(false);
    image_label_->setAlignment(Qt::AlignCenter);
    layout->addWidget(image_label_);
    setLayout(layout);

    if (graph_)
    {
        if (auto zed_node = graph_->get_node(camera_node_name_); zed_node.has_value())
        {
            camera_api_ = graph_->get_camera_api(zed_node.value());
            inner_eigen_api_ = graph_->get_inner_eigen_api();
        }
        else
        {
            image_label_->setText("No 'zed' node found in DSR");
        }
    }

    fetch_camera_intrinsics();

    // Refresh periodically to show the live camera stream and overlay.
    refresh_timer_ = new QTimer(this);
    refresh_timer_->setTimerType(Qt::PreciseTimer);
    connect(refresh_timer_, &QTimer::timeout, this, &CameraVisualizer::update_frame);
    refresh_timer_->start(50);  // target 20 Hz to guarantee >=10 Hz in practice
    fps_timer_.start();
    update_frame();  // paint first frame immediately
}

bool CameraVisualizer::fetch_rgb_from_dsr(QImage& rgb_image, std::uint64_t& frame_timestamp)
{
    frame_timestamp = 0;

    if (!graph_)
    {
        image_label_->setText("No DSR graph available");
        return false;
    }

    if (!camera_api_)
    {
        if (auto zed_node = graph_->get_node(camera_node_name_); zed_node.has_value())
        {
            camera_api_ = graph_->get_camera_api(zed_node.value());
            inner_eigen_api_ = graph_->get_inner_eigen_api();
            fetch_camera_intrinsics();
        }
        else
        {
            image_label_->setText("No 'zed' node found in DSR");
            return false;
        }
    }

    if (!camera_data_.valid)
    {
        fetch_camera_intrinsics();
    }

    if (auto zed_node = graph_->get_node(camera_node_name_); zed_node.has_value())
    {
        if (auto ts_opt = graph_->get_attrib_timestamp_by_name(zed_node.value(), "cam_rgb"); ts_opt.has_value())
            frame_timestamp = ts_opt.value();
        else if (auto alive_opt = graph_->get_attrib_by_name<cam_rgb_alivetime_att>(zed_node.value()); alive_opt.has_value())
            frame_timestamp = static_cast<std::uint64_t>(alive_opt.value());
    }

    const auto rgb_opt = camera_api_->get_rgb_image();
    if (!rgb_opt.has_value())
    {
        image_label_->setText("No cam_rgb data in 'zed' node");
        return false;
    }

    const auto& rgb = rgb_opt.value();
    const int width = camera_data_.width;
    const int height = camera_data_.height;
    if (width <= 0 || height <= 0)
    {
        image_label_->setText("Invalid camera dimensions");
        return false;
    }

    const std::size_t px_count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (rgb.size() == px_count * 4)
    {
        QImage img(reinterpret_cast<const uchar*>(rgb.data()), width, height, width * 4, QImage::Format_RGBA8888);
        rgb_image = img.copy();
        return true;
    }

    if (rgb.size() == px_count * 3)
    {
        // cam_rgb is expected as interleaved RGB888.
        QImage img(reinterpret_cast<const uchar*>(rgb.data()), width, height, width * 3, QImage::Format_RGB888);
        rgb_image = img.copy();
        return true;
    }

    if (rgb.size() == px_count)
    {
        QImage img(reinterpret_cast<const uchar*>(rgb.data()), width, height, width, QImage::Format_Grayscale8);
        rgb_image = img.copy();
        return true;
    }

    if (rgb.size() > px_count * 3)
    {
        QImage img(reinterpret_cast<const uchar*>(rgb.data()), width, height, width * 3, QImage::Format_RGB888);
        rgb_image = img.copy();
        return true;
    }

    image_label_->setText(QString("cam_rgb payload too small: %1 bytes").arg(rgb.size()));
    return false;
}

bool CameraVisualizer::fetch_camera_intrinsics()
{
    if (!camera_api_)
        return false;

    const float fx = camera_api_->get_focal_x();
    const float fy = camera_api_->get_focal_y();
    const int width = static_cast<int>(camera_api_->get_width());
    const int height = static_cast<int>(camera_api_->get_height());
    const float cx = static_cast<float>(width) * 0.5f;
    const float cy = static_cast<float>(height) * 0.5f;

    if (fx <= 0.f || fy <= 0.f || width <= 0 || height <= 0)
    {
        camera_data_.valid = false;
        return false;
    }

    camera_data_.K << fx, 0, cx,
                      0, fy, cy,
                      0, 0, 1;
    camera_data_.width = width;
    camera_data_.height = height;
    camera_data_.valid = true;

    return true;
}

std::vector<Eigen::Vector3f> CameraVisualizer::get_room_corners_3d() const
{
    std::vector<Eigen::Vector3f> corners_3d;
    float room_height = 2.4f;
    auto room_polygon = room_polygon_;

    if (graph_)
    {
        if (auto room_node = graph_->get_node(room_frame_name_); room_node.has_value())
        {
            if (const auto h = graph_->get_attrib_by_name<room_height_att>(room_node.value()); h.has_value())
                room_height = h.value();
        }

        if (auto polygon_from_dsr = read_room_polygon_from_dsr(graph_, room_frame_name_);
            polygon_from_dsr.size() >= 3)
        {
            room_polygon = std::move(polygon_from_dsr);
        }
    }

    // All coords in metres (polygon X/Y from DSR are in metres, room_height in metres).
    for (const auto& corner_2d : room_polygon)
    {
        corners_3d.emplace_back(corner_2d.x(), corner_2d.y(), 0.0f);
        corners_3d.emplace_back(corner_2d.x(), corner_2d.y(), room_height);
    }

    return corners_3d;
}

std::vector<Eigen::Vector2f> CameraVisualizer::project_points_to_image(
    const std::vector<Eigen::Vector3f>& world_points, std::uint64_t rt_timestamp) const
{
    std::vector<Eigen::Vector2f> image_points;
    image_points.reserve(world_points.size());

    if (!inner_eigen_api_ || !camera_api_)
    {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        for (std::size_t i = 0; i < world_points.size(); ++i)
            image_points.emplace_back(nan, nan);
        return image_points;
    }

    if (!graph_ || !graph_->get_node(room_frame_name_).has_value() || !graph_->get_node(camera_node_name_).has_value())
    {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        for (std::size_t i = 0; i < world_points.size(); ++i)
            image_points.emplace_back(nan, nan);
        return image_points;
    }

    for (std::size_t i = 0; i < world_points.size(); ++i)
    {
        const auto& p = world_points[i];
        const auto p_in_cam_opt = inner_eigen_api_->transform(
            camera_node_name_,
            Mat::Vector3d(p.x(), p.y(), p.z()),
            room_frame_name_,
            rt_timestamp);

        if (!p_in_cam_opt.has_value())
        {
            const float nan = std::numeric_limits<float>::quiet_NaN();
            image_points.emplace_back(nan, nan);
            continue;
        }

        const auto p_cam = p_in_cam_opt.value();

        // CameraAPI::project uses Y as depth axis in camera frame.
        if (p_cam.y() <= 1e-6)
        {
            const float nan = std::numeric_limits<float>::quiet_NaN();
            image_points.emplace_back(nan, nan);
            continue;
        }

        const Eigen::Vector2d uv = camera_api_->project(p_cam);
        image_points.emplace_back(static_cast<float>(uv.x()), static_cast<float>(uv.y()));
    }

    return image_points;
}

void CameraVisualizer::draw_projections(QImage& image, std::uint64_t rt_timestamp)
{
    if (!camera_data_.valid)
        return;

    if (!graph_ || !graph_->get_node(room_frame_name_).has_value() || !graph_->get_node(camera_node_name_).has_value())
        return;

    auto corners_3d = get_room_corners_3d();
    if (corners_3d.size() < 6)
        return;

    auto image_points = project_points_to_image(corners_3d, rt_timestamp);

    QPainter painter(&image);
    painter.setRenderHint(QPainter::Antialiasing, true);

    const int num_corners = static_cast<int>(corners_3d.size() / 2);
    if (!inner_eigen_api_ || !camera_api_ || num_corners < 2)
        return;

    // 1) Transform all room floor/top corners from room frame -> camera frame.
    std::vector<Mat::Vector3d> floor_in_cam;
    std::vector<Mat::Vector3d> top_in_cam;
    floor_in_cam.reserve(num_corners);
    top_in_cam.reserve(num_corners);
    for (int i = 0; i < num_corners; ++i)
    {
        const auto& p_floor_room = corners_3d[i * 2];      // floor corner z=0
        const auto& p_top_room = corners_3d[i * 2 + 1];    // top corner z=room_height

        const auto p_floor_cam_opt = inner_eigen_api_->transform(
            camera_node_name_, Mat::Vector3d(p_floor_room.x(), p_floor_room.y(), p_floor_room.z()), room_frame_name_, rt_timestamp);
        const auto p_top_cam_opt = inner_eigen_api_->transform(
            camera_node_name_, Mat::Vector3d(p_top_room.x(), p_top_room.y(), p_top_room.z()), room_frame_name_, rt_timestamp);
        if (!p_floor_cam_opt.has_value() || !p_top_cam_opt.has_value())
            return;
        floor_in_cam.push_back(p_floor_cam_opt.value());
        top_in_cam.push_back(p_top_cam_opt.value());
    }

    auto finite = [](const Eigen::Vector2f& p)
    {
        return std::isfinite(p.x()) && std::isfinite(p.y());
    };

    auto draw_segment = [&](const Eigen::Vector2f& a, const Eigen::Vector2f& b, const QPen& pen)
    {
        if (!finite(a) || !finite(b))
            return;

        QPointF p0(a.x(), a.y());
        QPointF p1(b.x(), b.y());

        // Draw the projected edge segment itself. Extending the line to the full
        // image can create visual crossings near corner views.
        painter.setPen(pen);
        painter.drawLine(p0, p1);
    };

    auto project_clipped_segment = [&](Mat::Vector3d a, Mat::Vector3d b, Eigen::Vector2f& out_a, Eigen::Vector2f& out_b)
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

        const Eigen::Vector2d uv0 = camera_api_->project(a);
        const Eigen::Vector2d uv1 = camera_api_->project(b);
        out_a = Eigen::Vector2f(static_cast<float>(uv0.x()), static_cast<float>(uv0.y()));
        out_b = Eigen::Vector2f(static_cast<float>(uv1.x()), static_cast<float>(uv1.y()));
        return true;
    };

    // 2) Draw a very light grid on floor and ceiling (different colors).
    float min_x = std::numeric_limits<float>::infinity();
    float max_x = -std::numeric_limits<float>::infinity();
    float min_y = std::numeric_limits<float>::infinity();
    float max_y = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < num_corners; ++i)
    {
        const auto& p = corners_3d[i * 2];
        min_x = std::min(min_x, p.x());
        max_x = std::max(max_x, p.x());
        min_y = std::min(min_y, p.y());
        max_y = std::max(max_y, p.y());
    }

    const float floor_z = 0.f;
    const float ceil_z = corners_3d[1].z();
    const float grid_step = 1.0f;  // metres

    auto draw_grid_plane = [&](float z, const QPen& pen)
    {
        auto project_room_segment = [&](const Mat::Vector3d& p0_room, const Mat::Vector3d& p1_room,
                                        Eigen::Vector2f& out0, Eigen::Vector2f& out1)
        {
            const auto p0_cam_opt = inner_eigen_api_->transform(camera_node_name_, p0_room, room_frame_name_, rt_timestamp);
            const auto p1_cam_opt = inner_eigen_api_->transform(camera_node_name_, p1_room, room_frame_name_, rt_timestamp);
            if (!p0_cam_opt.has_value() || !p1_cam_opt.has_value())
                return false;
            return project_clipped_segment(p0_cam_opt.value(), p1_cam_opt.value(), out0, out1);
        };

        const float x0 = std::floor(min_x / grid_step) * grid_step;
        const float x1 = std::ceil(max_x / grid_step) * grid_step;
        const float y0 = std::floor(min_y / grid_step) * grid_step;
        const float y1 = std::ceil(max_y / grid_step) * grid_step;

        // Lines parallel to room Y axis.
        for (float x = x0; x <= x1 + 1e-4f; x += grid_step)
        {
            Eigen::Vector2f a;
            Eigen::Vector2f b;
            if (project_room_segment(Mat::Vector3d(x, y0, z), Mat::Vector3d(x, y1, z), a, b))
                draw_segment(a, b, pen);
        }

        // Lines parallel to room X axis.
        for (float y = y0; y <= y1 + 1e-4f; y += grid_step)
        {
            Eigen::Vector2f a;
            Eigen::Vector2f b;
            if (project_room_segment(Mat::Vector3d(x0, y, z), Mat::Vector3d(x1, y, z), a, b))
                draw_segment(a, b, pen);
        }
    };

    draw_grid_plane(floor_z, QPen(QColor(80, 140, 255, 65), 1.4));   // slightly stronger blue floor grid
    draw_grid_plane(ceil_z, QPen(QColor(255, 110, 110, 65), 1.4));   // slightly stronger red ceiling grid

    // 3) Draw all visible floor/ceiling edge segments.
    for (int i = 0; i < num_corners; ++i)
    {
        const int j = (i + 1) % num_corners;
        Eigen::Vector2f floor_a;
        Eigen::Vector2f floor_b;
        Eigen::Vector2f top_a;
        Eigen::Vector2f top_b;

        const bool floor_ok = project_clipped_segment(floor_in_cam[i], floor_in_cam[j], floor_a, floor_b);
        const bool top_ok = project_clipped_segment(top_in_cam[i], top_in_cam[j], top_a, top_b);

        if (floor_ok)
            draw_segment(floor_a, floor_b, QPen(Qt::blue, 3));
        if (top_ok)
            draw_segment(top_a, top_b, QPen(Qt::red, 3));

        // Optional endpoints for visual debugging.
        painter.setPen(QPen(Qt::blue, 2));
        painter.setBrush(QColor(0, 100, 255, 100));
        if (floor_ok && finite(floor_a))
            painter.drawEllipse(QPointF(floor_a.x(), floor_a.y()), 4, 4);
        if (floor_ok && finite(floor_b))
            painter.drawEllipse(QPointF(floor_b.x(), floor_b.y()), 4, 4);

        painter.setPen(QPen(Qt::red, 2));
        painter.setBrush(QColor(255, 100, 0, 100));
        if (top_ok && finite(top_a))
            painter.drawEllipse(QPointF(top_a.x(), top_a.y()), 4, 4);
        if (top_ok && finite(top_b))
            painter.drawEllipse(QPointF(top_b.x(), top_b.y()), 4, 4);
    }

    painter.end();
}

void CameraVisualizer::update_frame()
{
    QElapsedTimer frame_timer;
    frame_timer.start();

    QImage rgb_image;
    std::uint64_t frame_timestamp = 0;
    if (fetch_rgb_from_dsr(rgb_image, frame_timestamp))
    {
        draw_projections(rgb_image, frame_timestamp);
        image_label_->setPixmap(QPixmap::fromImage(rgb_image));

        ++frames_since_fps_log_;
        const qint64 elapsed_ms = fps_timer_.elapsed();
        if (elapsed_ms >= 2000)
        {
            fps_timer_.restart();
            frames_since_fps_log_ = 0;
        }
    }
}

}  // namespace rc
