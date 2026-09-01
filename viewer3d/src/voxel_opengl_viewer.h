#pragma once

#include <QColor>
#include <QImage>
#include <QMatrix4x4>
#include <QOpenGLBuffer>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWidget>
#include <QPoint>
#include <QString>
#include <QVector2D>
#include <QVector3D>

#include <memory>

#include <array>
#include <chrono>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace rc
{
class VoxelOpenGLViewer final : public QOpenGLWidget, protected QOpenGLFunctions
{
public:
    explicit VoxelOpenGLViewer(QWidget* parent = nullptr);
    ~VoxelOpenGLViewer() override;

    void update_lidar_points(std::span<const QVector3D> positions,
                             std::span<const std::uint8_t> plane_id = {});
    void update_residual_points(std::span<const QVector3D> residual_positions);
    // YOLO mask support points (room frame), drawn as a distinct point cloud. Optional per-point
    // categories colour them by class (color_for_category, matching the voxel/box palette); empty
    // → fall back to off-white. Optional per-point sources (0=zed, 1=ricoh) keep the category HUE but
    // dim + desaturate ricoh (360) points so the front RGB-D and peripheral evidence are told apart.
    void update_mask_points(std::span<const QVector3D> positions,
                            std::span<const std::string> categories = {},
                            std::span<const float> sources = {});
    void set_show_lidar(bool show);
    void set_show_masks(bool show);
    void set_show_residual(bool show);
    // residual_concept occupancy-grid display: the residual cell centres (room frame) + cell edge size.
    void update_grid_cells(std::span<const QVector3D> cell_centres, float cell_size);
    // The inflated half-robot-width clearance border, drawn in a second colour.
    void update_grid_border(std::span<const QVector3D> border_centres);
    void set_show_grid(bool show);
    // Beta-posterior BELIEF FIELD heatmap: per-cell centres + mean occupancy P (RISK, hue) + Var[P] (EPISTEMIC,
    // brightness). Cells the source collapsed to P=0 (a modelled object owns them) are skipped.
    // Residual field, drawn as ONE COLUMN PER CELL on the cell's own footprint (blue→orange→red by
    // height). The grid extent [xmin,ymin] + cell·(w,h) is used only to find neighbours for face culling.
    // `occupied` = cell centres with z = the cell's top height; `base_z` = the matching band BOTTOMS
    // (same order, one per cell) — pass an empty span when the producer publishes no bottom and every
    // column will stand on the floor.
    void update_grid_field(std::span<const QVector3D> occupied, std::span<const float> base_z,
                           float xmin, float ymin, float cell, int w, int h);
    void set_show_field(bool show);
    // Fitted-model layer: the table mesh, the solid bottle cylinder and the graph object boxes.
    void set_show_models(bool show);

    void update_room_polygon(std::span<const float> polygon_x,
                             std::span<const float> polygon_y);

    // New: update both floor and ceiling outlines
    void update_room_polygon_dual(std::span<const float> polygon_x,
                                  std::span<const float> polygon_y,
                                  float height);

    // DSR graph object boxes in room frame. Each box is oriented: it is drawn as
    // a wireframe rotated about Z by its yaw (room-frame orientation from the RT
    // edge), rather than an axis-aligned AABB.
    void update_graph_boxes(std::span<const QVector3D> centers,
                            std::span<const QVector3D> half_extents,
                            std::span<const float> yaws,
                            std::span<const std::string> categories = {},
                            std::span<const std::string> names = {},
                            std::span<const std::string> subtypes = {},
                            std::span<const std::string> rig_schemas = {},
                            std::span<const std::string> mesh_paths = {},
                            std::span<const std::string> mesh_texture_paths = {},
                            // Per-box inferred albedo chromaticity (common/appearance_belief). A zero vector
                            // means the agent published none, and the asset renders with its authored colours.
                            std::span<const QVector3D> mesh_colors = {});

    // Flat triangle-list meshes in room frame [x0,y0,z0, x1,y1,z1, ...]. Optional per-mesh categories
    // colour them by class (table = amber model colour, others via color_for_category). Optional per-mesh
    // node names drive the per-instance shade + the 3D text label.
    void update_object_meshes(std::span<const std::vector<float>> meshes,
                              std::span<const std::string> categories = {},
                              std::span<const std::string> names = {});
    // Draw each drawn node's DSR name as a text label at its position (debug reference). ON by default.
    void set_show_labels(bool show);

    // Human skeletons: one flat BODY_18 array per person [x0,y0,z0, ... x17,y17,z17] (54 floats),
    // room frame. NaN joints are skipped. Drawn as bones (BODY18 edges) + joint points.
    void update_skeletons(std::span<const std::vector<float>> skeletons);
    void set_show_skeletons(bool show);

    // Input-stream delivery rates (Hz) shown in the HUD; <0 ⇒ "--". Pushed from the render tick.
    void set_stream_fps(float rgb_hz, float rgb360_hz) { rgb_fps_ = rgb_hz; rgb360_fps_ = rgb360_hz; }

    // A HELD state, said ON THE CANVAS. The agent goes on rendering while it holds — that is the whole
    // point of a viewer — so a frozen scene looks exactly like a live one unless the reason is drawn
    // over it. Empty ⇒ no banner. Set from the GRAFCET transitions, never per tick.
    void set_status_banner(const QString& text);
    void set_perf_log(bool on) { perf_log_ = on; }   // gate the per-paint CSV probe (off in production)

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
    // Textured furniture vertex: position + a baked lighting multiplier + UV (sampled base-colour × light).
    struct TexVertex
    {
        float px, py, pz;
        // Baked light, per CHANNEL rather than a single scalar, so a per-instance albedo tint can ride along
        // in the existing vertex attribute. Keeping it here means the textured pass still batches purely by
        // texture pointer — two instances of the same asset with different inferred colours share one draw.
        float lr, lg, lb;
        float u, v;
    };
    // One material group of a display template: geometry + its material (flat Kd and/or a base-colour texture).
    struct SubMeshGL
    {
        std::vector<QVector3D> tris;                   // positions (room-frame unit)
        std::vector<QVector2D> uv;                     // per-vertex UV (parallel to tris)
        QColor diffuse;                                // Kd flat colour (used when no texture)
        bool has_diffuse = false;                      // true iff the .mtl gave a Kd
        QImage tex_image;                              // CPU base-colour image (null ⇒ untextured group)
        std::unique_ptr<QOpenGLTexture> tex;           // GPU texture, uploaded lazily in paintGL
    };
    // A furniture display template: the material groups of one OBJ (+ its .mtl). Self-describes appearance.
    struct FurnitureTemplate
    {
        std::vector<SubMeshGL> subs;
        // The asset's OWN mean chromaticity (from the .mtl Kd values and any base-colour textures), computed
        // once at load. An inferred instance colour is applied RELATIVE to this — tint = observed/authored,
        // renormalised to unit mean — so the result matches the observed hue while preserving whatever
        // inter-material contrast the artist authored. A straight replacement would flatten a wood-top /
        // metal-leg asset to a single colour; multiplying raw would drive brown x brown toward black.
        QVector3D mean_chroma{1.f / 3.f, 1.f / 3.f, 1.f / 3.f};
    };

    // tint for a template given an inferred chromaticity (zero ⇒ none): observed/authored, renormalised to
    // unit mean so it shifts hue without changing overall brightness. Returns (1,1,1) when there is nothing
    // to apply, which is the identity everywhere it is used.
    static QVector3D mesh_tint(const FurnitureTemplate& tpl, const QVector3D& inferred_chroma);

    static QColor color_for_category(const std::string& category);
    // Trailing integer of a DSR node name ("cabinet_2" → 2, "table" → 0) → per-instance id.
    static int instance_index_from_name(const std::string& name);
    // Same hue as `base`, but a distinct brightness per instance id so several nodes of one type read apart.
    static QColor shade_for_instance(const QColor& base, int instance_id);
    void request_update_throttled();
    // Load (once, cached by relative asset path) a concept-published display mesh + optional base-colour
    // texture. Called from paintGL (GL context current for the lazy texture upload). Returns nullptr if the
    // path is empty or the OBJ failed to load (a failed load is cached to avoid retrying every frame).
    FurnitureTemplate* get_or_load_template(const std::string& mesh_path, const std::string& texture_path);
    void load_view_state();
    void save_view_state() const;
    QOpenGLShaderProgram program_;
    QOpenGLVertexArrayObject room_vao_;
    QOpenGLBuffer room_vbo_{QOpenGLBuffer::VertexBuffer};
    QOpenGLShaderProgram tex_program_;                 // textured-furniture pass (sampler2D × baked light)
    QOpenGLVertexArrayObject tex_vao_;
    QOpenGLBuffer tex_vbo_{QOpenGLBuffer::VertexBuffer};

    std::vector<Vertex> lidar_vertices_;
    std::vector<Vertex> residual_vertices_;
    std::vector<Vertex> grid_vertices_;        // residual_concept occupancy-grid cells (amber)
    std::vector<Vertex> grid_border_vertices_; // inflated clearance border (cyan)
    std::vector<Vertex> grid_field_vertices_;      // residual COLUMNS: box triangles, one box per occupied cell
    std::vector<Vertex> grid_field_cap_vertices_;  // unused by the column build (its caps are triangles)
    std::vector<Vertex> mask_vertices_;
    std::mutex data_mutex_;

    // Store both floor and ceiling polygons
    std::vector<QVector3D> room_polygon_floor_;
    std::vector<QVector3D> room_polygon_ceiling_;
    std::vector<QVector3D> graph_box_centers_;
    std::vector<QVector3D> graph_box_half_extents_;
    std::vector<float> graph_box_yaws_;
    std::vector<std::string> graph_box_categories_;
    std::vector<std::string> graph_box_names_;   // parallel to graph_box_centers_ → per-instance shade + label
    std::vector<std::string> graph_box_subtypes_;// parallel to graph_box_centers_ → shape subtype ("round"/"square")
    // parallel → level-2 SHAPE declaration ("ring"/"rectilinear"); "" for non-metaconcepts
    std::vector<std::string> graph_box_schemas_;
    std::vector<std::string> graph_box_mesh_paths_;    // parallel → concept-published display OBJ (relative)
    std::vector<std::string> graph_box_mesh_tex_;      // parallel → optional base-colour texture (relative)
    std::vector<QVector3D>   graph_box_mesh_color_;    // parallel → inferred albedo chromaticity ((0,0,0) ⇒ none)
    // Raw polygon coordinates (room frame) plus current debug rotation in 90deg steps.
    std::vector<float> raw_polygon_x_;
    std::vector<float> raw_polygon_y_;
    float raw_polygon_height_ = 0.f;
    int polygon_rotation_quadrants_ = 0; // 0..3, each = +90deg around Y
    bool polygon_flip_x_ = true;
    bool polygon_flip_y_ = false;
    bool voxel_flip_x_ = true;
    bool voxel_flip_y_ = false;
    bool show_lidar_ = false;
    bool show_masks_ = false;
    bool show_residual_ = false;   // table_concept residual debug cloud — OFF by default
    bool show_grid_ = true;        // residual_concept occupancy grid — ON by default (the new safety layer)
    bool show_field_ = true;       // residual_concept Beta belief-field heatmap — ON by default
    bool show_models_ = true;
    bool show_skeletons_ = true;
    bool show_labels_ = true;      // draw DSR node names as 3D text labels (debug reference)
    std::vector<QVector3D> robot_mesh_local_;
    std::mutex robot_mesh_mutex_;
    std::unordered_map<std::string, FurnitureTemplate> mesh_cache_;   // display templates keyed by relative asset path

    // Robot pose (room frame).
    bool have_robot_pose_ = false;
    float robot_x_ = 0.f;
    float robot_y_ = 0.f;
    float robot_theta_ = 0.f;
    std::mutex robot_pose_mutex_;
    std::mutex room_polygon_mutex_;
    std::mutex graph_boxes_mutex_;
    std::vector<std::vector<float>> object_meshes_;
    std::vector<std::string>        object_mesh_categories_;   // parallel to object_meshes_
    std::vector<std::string>        object_mesh_names_;        // parallel to object_meshes_ → shade + label
    std::mutex object_meshes_mutex_;
    std::vector<std::vector<float>> skeletons_;                // BODY_18 flat (54 floats) per person, room frame
    std::vector<QVector3D> skeleton_facing_;                   // EMA-smoothed facing (room frame), parallel to skeletons_
    struct FacingTrack { QVector3D chest, facing; };
    std::vector<FacingTrack> facing_tracks_;                   // persistent across updates for the facing EMA
    std::mutex skeletons_mutex_;
    void rebuild_polygon_locked_();

    bool gl_ready_ = false;
    bool repaint_scheduled_ = false;
    std::chrono::steady_clock::time_point last_update_request_{};
    static constexpr std::chrono::milliseconds kMinUpdateIntervalMs{50};  // cap repaints at 20 Hz
    // Real render rate: paints counted over a ~1 s window (the actual repaint cadence of the 3D view).
    // A windowed count, NOT an instantaneous inter-call EMA — paints are event-coalesced by Qt, so
    // per-frame intervals are jittery and would swing the number; counting over a window is stable.
    // Shown in the HUD instead of the voxel-update FPS (which is ~dead while voxels are off).
    std::chrono::steady_clock::time_point fps_window_start_{};
    int   render_frame_count_ = 0;
    float render_fps_ = 0.0f;
    bool perf_log_ = false;   // when false, skip the per-paint CSV write+flush (see paintGL probe)
    float rgb_fps_    = -1.0f;   // input-stream rates for the HUD (<0 ⇒ unknown/"--")
    float rgb360_fps_ = -1.0f;
    QString status_banner_;      // non-empty ⇒ the agent is HOLDING; drawn over the scene (see set_status_banner)

    float yaw_ = 0.0f;
    float pitch_ = +0.52f;
    float distance_ = 8.5f;
    QVector3D target_{0.f, 0.f, 0.f};
    bool room_target_initialized_ = false;
    bool camera_user_moved_ = false;

    QPoint last_mouse_pos_;
};

} // namespace rc
