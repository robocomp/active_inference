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
    void update_rfe_points(std::span<const QVector3D> residual_positions,
                           std::span<const QVector3D> fallback_positions = {});
    // YOLO mask support points (room frame), drawn as a distinct point cloud. Optional per-point
    // categories colour them by class (color_for_category, matching the voxel/box palette); empty
    // → fall back to off-white.
    void update_mask_points(std::span<const QVector3D> positions,
                            std::span<const std::string> categories = {});
    void set_show_lidar(bool show);
    void set_show_masks(bool show);
    void set_show_voxels(bool show);
    void set_show_residual(bool show);
    void set_show_rfe(bool show);
    // Fitted-model layer: the table mesh, the solid bottle cylinder and the graph object boxes.
    void set_show_models(bool show);

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

    // DSR graph object boxes in room frame. Each box is oriented: it is drawn as
    // a wireframe rotated about Z by its yaw (room-frame orientation from the RT
    // edge), rather than an axis-aligned AABB.
    void update_graph_boxes(std::span<const QVector3D> centers,
                            std::span<const QVector3D> half_extents,
                            std::span<const float> yaws,
                            std::span<const std::string> categories = {});

    // Flat triangle-list meshes in room frame [x0,y0,z0, x1,y1,z1, ...]. Optional per-mesh categories
    // colour them by class (table = amber model colour, others via color_for_category).
    void update_object_meshes(std::span<const std::vector<float>> meshes,
                              std::span<const std::string> categories = {});

    // Human skeletons: one flat BODY_18 array per person [x0,y0,z0, ... x17,y17,z17] (54 floats),
    // room frame. NaN joints are skipped. Drawn as bones (BODY18 edges) + joint points.
    void update_skeletons(std::span<const std::vector<float>> skeletons);
    void set_show_skeletons(bool show);

    // Robot pose in room frame (x, y in meters; theta in radians).
    void set_robot_pose(float x, float y, float theta);
    bool load_robot_mesh(const std::string& path);

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
    void load_view_state();
    void save_view_state() const;
    QOpenGLShaderProgram program_;
    QOpenGLVertexArrayObject vao_;
    QOpenGLBuffer vbo_{QOpenGLBuffer::VertexBuffer};
    QOpenGLVertexArrayObject room_vao_;
    QOpenGLBuffer room_vbo_{QOpenGLBuffer::VertexBuffer};

    std::vector<Vertex> cpu_vertices_;
    std::vector<Vertex> lidar_vertices_;
    std::vector<Vertex> residual_vertices_;
    std::vector<Vertex> rfe_vertices_;
    std::vector<Vertex> mask_vertices_;
    std::mutex data_mutex_;

    // Store both floor and ceiling polygons
    std::vector<QVector3D> room_polygon_floor_;
    std::vector<QVector3D> room_polygon_ceiling_;
    std::vector<QVector3D> track_box_mins_;
    std::vector<QVector3D> track_box_maxs_;
    std::vector<std::string> track_box_categories_;
    std::vector<QVector3D> graph_box_centers_;
    std::vector<QVector3D> graph_box_half_extents_;
    std::vector<float> graph_box_yaws_;
    std::vector<std::string> graph_box_categories_;
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
    bool show_lidar_ = false;
    bool show_masks_ = false;
    bool show_residual_ = true;
    bool show_rfe_ = true;
    bool show_models_ = true;
    bool show_skeletons_ = true;
    std::vector<QVector3D> robot_mesh_local_;
    std::mutex robot_mesh_mutex_;

    // Robot pose (room frame).
    bool have_robot_pose_ = false;
    float robot_x_ = 0.f;
    float robot_y_ = 0.f;
    float robot_theta_ = 0.f;
    std::mutex robot_pose_mutex_;
    std::mutex room_polygon_mutex_;
    std::mutex track_boxes_mutex_;
    std::mutex graph_boxes_mutex_;
    std::vector<std::vector<float>> object_meshes_;
    std::vector<std::string>        object_mesh_categories_;   // parallel to object_meshes_
    std::mutex object_meshes_mutex_;
    std::vector<std::vector<float>> skeletons_;                // BODY_18 flat (54 floats) per person, room frame
    std::vector<QVector3D> skeleton_facing_;                   // EMA-smoothed facing (room frame), parallel to skeletons_
    struct FacingTrack { QVector3D chest, facing; };
    std::vector<FacingTrack> facing_tracks_;                   // persistent across updates for the facing EMA
    std::mutex skeletons_mutex_;
    void rebuild_polygon_locked_();

    bool gl_ready_ = false;
    bool upload_pending_ = false;
    bool repaint_scheduled_ = false;
    std::chrono::steady_clock::time_point last_update_request_{};
    std::chrono::steady_clock::time_point last_voxel_update_time_{};
    static constexpr std::chrono::milliseconds kMinUpdateIntervalMs{50};  // cap repaints at 20 Hz
    float voxel_input_fps_ = 0.0f;
    // Real render rate: paints counted over a ~1 s window (the actual repaint cadence of the 3D view).
    // A windowed count, NOT an instantaneous inter-call EMA — paints are event-coalesced by Qt, so
    // per-frame intervals are jittery and would swing the number; counting over a window is stable.
    // Shown in the HUD instead of the voxel-update FPS (which is ~dead while voxels are off).
    std::chrono::steady_clock::time_point fps_window_start_{};
    int   render_frame_count_ = 0;
    float render_fps_ = 0.0f;

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
