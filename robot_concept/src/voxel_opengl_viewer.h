#pragma once

#include <QColor>
#include <QMatrix4x4>
#include <QOpenGLBuffer>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWidget>
#include <QPoint>
#include <QVector3D>

#include <array>
#include <chrono>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace rc
{
class VoxelOpenGLViewer final : public QOpenGLWidget, protected QOpenGLFunctions
{
public:
    explicit VoxelOpenGLViewer(QWidget* parent = nullptr);
    ~VoxelOpenGLViewer() override;

    void update_voxels(std::span<const QVector3D> positions,
                       std::span<const std::string> categories = {},
                       std::span<const float> confidences = {});

    void update_lidar_points(std::span<const QVector3D> positions);
    void set_show_lidar(bool show);

    void update_room_polygon(std::span<const float> polygon_x,
                             std::span<const float> polygon_y);

    // New: update both floor and ceiling outlines
    void update_room_polygon_dual(std::span<const float> polygon_x,
                                  std::span<const float> polygon_y,
                                  float height);

    // Tracked object boxes in room frame (min/max corners per track).
    void update_track_boxes(std::span<const QVector3D> mins,
                            std::span<const QVector3D> maxs,
                            std::span<const std::string> categories = {});

    // Robot pose in room frame (x, y in meters; theta in radians).
    void set_robot_pose(float x, float y, float theta);

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;

private:
    struct Vertex
    {
        float px, py, pz;
        float r, g, b;
    };

    static QColor color_for_category(const std::string& category);
    void request_update_throttled();
    QOpenGLShaderProgram program_;
    QOpenGLVertexArrayObject vao_;
    QOpenGLBuffer vbo_{QOpenGLBuffer::VertexBuffer};
    QOpenGLVertexArrayObject room_vao_;
    QOpenGLBuffer room_vbo_{QOpenGLBuffer::VertexBuffer};

    std::vector<Vertex> cpu_vertices_;
    std::vector<Vertex> lidar_vertices_;
    std::mutex data_mutex_;

    // Store both floor and ceiling polygons
    std::vector<QVector3D> room_polygon_floor_;
    std::vector<QVector3D> room_polygon_ceiling_;
    std::vector<QVector3D> track_box_mins_;
    std::vector<QVector3D> track_box_maxs_;
    std::vector<std::string> track_box_categories_;
    // Raw polygon coordinates (room frame) plus current debug rotation in 90deg steps.
    std::vector<float> raw_polygon_x_;
    std::vector<float> raw_polygon_y_;
    float raw_polygon_height_ = 0.f;
    int polygon_rotation_quadrants_ = 0; // 0..3, each = +90deg around Y
    bool polygon_flip_x_ = true;
    bool polygon_flip_y_ = false;
    bool voxel_flip_x_ = true;
    bool voxel_flip_y_ = false;
    bool show_voxels_ = true;
    bool show_lidar_ = true;

    // Robot pose (room frame).
    bool have_robot_pose_ = false;
    float robot_x_ = 0.f;
    float robot_y_ = 0.f;
    float robot_theta_ = 0.f;
    std::mutex robot_pose_mutex_;
    std::mutex room_polygon_mutex_;
    std::mutex track_boxes_mutex_;
    void rebuild_polygon_locked_();

    bool gl_ready_ = false;
    bool upload_pending_ = false;
    bool repaint_scheduled_ = false;
    std::chrono::steady_clock::time_point last_update_request_{};
    std::chrono::steady_clock::time_point last_voxel_update_time_{};
    static constexpr std::chrono::milliseconds kMinUpdateIntervalMs{33};
    float voxel_input_fps_ = 0.0f;

    float yaw_ = 0.0f;
    float pitch_ = +0.52f;
    float distance_ = 8.5f;
    QVector3D target_{0.f, 0.f, 0.f};
    bool first_cloud_received_ = false;
    bool room_target_initialized_ = false;
    bool camera_user_moved_ = false;

    QPoint last_mouse_pos_;
};

} // namespace rc
