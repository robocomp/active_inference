#include "voxel_opengl_viewer.h"

#include <QCoreApplication>
#include <QHash>
#include <QDebug>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QSettings>
#include <QSurfaceFormat>
#include <QTimer>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>

namespace rc
{
namespace
{
constexpr auto kViewerSettingsGroup = "VoxelOpenGLViewer";

struct ObjMeshData
{
    std::vector<QVector3D> triangles;
    QVector3D bb_min;
    QVector3D bb_max;
};

std::optional<std::filesystem::path> resolve_robot_mesh_path(const std::string& mesh_path)
{
    if (mesh_path.empty())
        return std::nullopt;

    const std::filesystem::path input(mesh_path);
    std::vector<std::filesystem::path> candidates;
    if (input.is_absolute())
        candidates.push_back(input);
    else
    {
        candidates.push_back(std::filesystem::current_path() / input);
        const auto app_dir = std::filesystem::path(QCoreApplication::applicationDirPath().toStdString());
        candidates.push_back(app_dir.parent_path() / input);
        candidates.push_back(input);
    }

    for (const auto& candidate : candidates)
        if (!candidate.empty() && std::filesystem::exists(candidate))
            return candidate;

    return std::nullopt;
}

int parse_obj_index(const std::string& token, int vertex_count)
{
    const auto slash = token.find('/');
    const std::string index_str = token.substr(0, slash);
    if (index_str.empty())
        return -1;

    const int raw_index = std::stoi(index_str);
    if (raw_index > 0)
        return raw_index - 1;
    if (raw_index < 0)
        return vertex_count + raw_index;
    return -1;
}

std::optional<ObjMeshData> load_obj_mesh_data(const std::filesystem::path& mesh_path)
{
    std::ifstream input(mesh_path);
    if (!input.is_open())
        return std::nullopt;

    std::vector<QVector3D> vertices;
    std::vector<QVector3D> triangles;
    QVector3D bb_min;
    QVector3D bb_max;
    bool have_bounds = false;
    std::string line;

    while (std::getline(input, line))
    {
        if (line.size() < 2)
            continue;

        std::istringstream stream(line);
        std::string tag;
        stream >> tag;
        if (tag == "v")
        {
            float x = 0.f, y = 0.f, z = 0.f;
            if (!(stream >> x >> y >> z))
                continue;

            const QVector3D vertex(x, y, z);
            vertices.push_back(vertex);
            if (!have_bounds)
            {
                bb_min = vertex;
                bb_max = vertex;
                have_bounds = true;
            }
            else
            {
                bb_min.setX(std::min(bb_min.x(), vertex.x()));
                bb_min.setY(std::min(bb_min.y(), vertex.y()));
                bb_min.setZ(std::min(bb_min.z(), vertex.z()));
                bb_max.setX(std::max(bb_max.x(), vertex.x()));
                bb_max.setY(std::max(bb_max.y(), vertex.y()));
                bb_max.setZ(std::max(bb_max.z(), vertex.z()));
            }
        }
        else if (tag == "f")
        {
            std::vector<int> face_indices;
            std::string token;
            while (stream >> token)
            {
                const int index = parse_obj_index(token, static_cast<int>(vertices.size()));
                if (index >= 0 && index < static_cast<int>(vertices.size()))
                    face_indices.push_back(index);
            }

            if (face_indices.size() < 3)
                continue;

            for (std::size_t i = 1; i + 1 < face_indices.size(); ++i)
            {
                triangles.push_back(vertices[static_cast<std::size_t>(face_indices[0])]);
                triangles.push_back(vertices[static_cast<std::size_t>(face_indices[i])]);
                triangles.push_back(vertices[static_cast<std::size_t>(face_indices[i + 1])]);
            }
        }
    }

    if (triangles.empty() || !have_bounds)
        return std::nullopt;

    return ObjMeshData{std::move(triangles), bb_min, bb_max};
}
}

VoxelOpenGLViewer::VoxelOpenGLViewer(QWidget* parent)
    : QOpenGLWidget(parent)
{
    QSurfaceFormat fmt;
    fmt.setRenderableType(QSurfaceFormat::OpenGL);
    fmt.setVersion(3, 3);
    fmt.setProfile(QSurfaceFormat::CoreProfile);
    fmt.setDepthBufferSize(24);
    fmt.setSwapBehavior(QSurfaceFormat::DoubleBuffer);
    setFormat(fmt);

    setMinimumSize(420, 300);
    setFocusPolicy(Qt::StrongFocus);

    last_update_request_ = std::chrono::steady_clock::now() - kMinUpdateIntervalMs;
    load_view_state();
}

void VoxelOpenGLViewer::load_view_state()
{
    QSettings settings("RoboComp", "robot_concept");
    settings.beginGroup(kViewerSettingsGroup);

    if (!(settings.contains("yaw") && settings.contains("pitch")
          && settings.contains("distance") && settings.contains("target_x")
          && settings.contains("target_y") && settings.contains("target_z")))
    {
        settings.endGroup();
        return;
    }

    yaw_ = settings.value("yaw", yaw_).toFloat();
    pitch_ = std::clamp(settings.value("pitch", pitch_).toFloat(), -1.45f, 1.45f);
    distance_ = std::clamp(settings.value("distance", distance_).toFloat(), 0.2f, 250.0f);
    target_.setX(settings.value("target_x", target_.x()).toFloat());
    target_.setY(settings.value("target_y", target_.y()).toFloat());
    target_.setZ(settings.value("target_z", target_.z()).toFloat());
    room_target_initialized_ = true;
    camera_user_moved_ = true;
    settings.endGroup();
}

void VoxelOpenGLViewer::save_view_state() const
{
    QSettings settings("RoboComp", "robot_concept");
    settings.beginGroup(kViewerSettingsGroup);
    settings.setValue("yaw", yaw_);
    settings.setValue("pitch", pitch_);
    settings.setValue("distance", distance_);
    settings.setValue("target_x", target_.x());
    settings.setValue("target_y", target_.y());
    settings.setValue("target_z", target_.z());
    settings.endGroup();
    settings.sync();
}

void VoxelOpenGLViewer::request_update_throttled()
{
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update_request_);
    if (elapsed >= kMinUpdateIntervalMs)
    {
        last_update_request_ = now;
        update();
        return;
    }

    if (repaint_scheduled_)
        return;

    repaint_scheduled_ = true;
    const auto wait_ms = std::max<std::chrono::milliseconds>(std::chrono::milliseconds{1}, kMinUpdateIntervalMs - elapsed);
    QTimer::singleShot(static_cast<int>(wait_ms.count()), this, [this]
    {
        repaint_scheduled_ = false;
        last_update_request_ = std::chrono::steady_clock::now();
        update();
    });
}

VoxelOpenGLViewer::~VoxelOpenGLViewer()
{
    save_view_state();
    makeCurrent();
    if (room_vbo_.isCreated())
        room_vbo_.destroy();
    doneCurrent();
}

void VoxelOpenGLViewer::update_object_meshes(std::span<const std::vector<float>> meshes,
                                             std::span<const std::string> categories)
{
    std::scoped_lock lk(object_meshes_mutex_);
    object_meshes_.assign(meshes.begin(), meshes.end());
    object_mesh_categories_.assign(categories.begin(), categories.end());
    request_update_throttled();
}

void VoxelOpenGLViewer::update_skeletons(std::span<const std::vector<float>> skeletons)
{
    std::scoped_lock lk(skeletons_mutex_);
    skeletons_.assign(skeletons.begin(), skeletons.end());

    // Facing is under-constrained (per-frame Kabsch to noisy keypoints), so the raw torso normal
    // jitters → the arrow oscillates. EMA-smooth it per person (matched by chest proximity across
    // updates), aligning each new sample to the tracked direction first so transient 180° flips are
    // rejected (a real turn rotates gradually and still tracks). Computed at the DATA rate here.
    constexpr float ALPHA = 0.12f;     // EMA weight on the new sample (lower = smoother)
    constexpr float MATCH2 = 0.4f * 0.4f;
    skeleton_facing_.assign(skeletons_.size(), QVector3D(0, 0, 0));
    std::vector<FacingTrack> next(skeletons_.size());
    for (std::size_t i = 0; i < skeletons_.size(); ++i)
    {
        const auto& s = skeletons_[i];
        if (s.size() < 18 * 3) continue;
        const auto fin = [&](int j){ return std::isfinite(s[j*3]) and std::isfinite(s[j*3+1]) and std::isfinite(s[j*3+2]); };
        if (not (fin(1) and fin(2) and fin(5) and fin(8) and fin(11))) continue;
        const auto P = [&](int j){ return QVector3D(s[j*3], s[j*3+1], s[j*3+2]); };
        const QVector3D neck = P(1), pelvis = (P(8) + P(11)) * 0.5f, chest = (neck + pelvis) * 0.5f;
        QVector3D fwd = QVector3D::crossProduct(neck - pelvis, P(5) - P(2));
        if (fwd.lengthSquared() < 1e-10f) continue;
        fwd.normalize();
        // Front/back sign: the EYES sit forward of the EARS (the nose offset has NO depth in the
        // template, so a nose-vs-neck test is ≈0 → its sign is noise → the arrow flips). eyes 14/15,
        // ears 16/17. Fall back to the (weak) nose test only if the face keypoints are missing.
        if (fin(14) and fin(15) and fin(16) and fin(17))
        {
            const QVector3D eye_mid = (P(14) + P(15)) * 0.5f;
            const QVector3D ear_mid = (P(16) + P(17)) * 0.5f;
            if (QVector3D::dotProduct(fwd, eye_mid - ear_mid) < 0.f) fwd = -fwd;
        }
        else if (fin(0) and QVector3D::dotProduct(fwd, P(0) - neck) < 0.f) fwd = -fwd;

        int best = -1; float bestd = MATCH2;
        for (std::size_t t = 0; t < facing_tracks_.size(); ++t)
        {
            const float d = (facing_tracks_[t].chest - chest).lengthSquared();
            if (facing_tracks_[t].facing.lengthSquared() > 1e-6f and d < bestd) { bestd = d; best = static_cast<int>(t); }
        }
        QVector3D sm = fwd;
        if (best >= 0)
        {
            QVector3D prev = facing_tracks_[best].facing;
            if (QVector3D::dotProduct(prev, fwd) < 0.f) fwd = -fwd;                 // reject transient flip
            sm = (prev * (1.f - ALPHA) + fwd * ALPHA).normalized();
        }
        skeleton_facing_[i] = sm;
        next[i] = {chest, sm};
    }
    facing_tracks_.swap(next);

    request_update_throttled();
}

void VoxelOpenGLViewer::set_show_skeletons(bool show)
{
    show_skeletons_ = show;
    update();
}

void VoxelOpenGLViewer::update_graph_boxes(std::span<const QVector3D> centers,
                                           std::span<const QVector3D> half_extents,
                                           std::span<const float> yaws,
                                           std::span<const std::string> categories)
{
    {
        std::scoped_lock lk(graph_boxes_mutex_);
        graph_box_centers_.assign(centers.begin(), centers.end());
        graph_box_half_extents_.assign(half_extents.begin(), half_extents.end());
        graph_box_yaws_.assign(yaws.begin(), yaws.end());
        graph_box_categories_.assign(categories.begin(), categories.end());
    }
}

void VoxelOpenGLViewer::set_show_lidar(bool show)
{
    show_lidar_ = show;
    request_update_throttled();
}

void VoxelOpenGLViewer::set_show_residual(bool show)
{
    show_residual_ = show;
    request_update_throttled();
}


void VoxelOpenGLViewer::set_show_masks(bool show)
{
    show_masks_ = show;
    request_update_throttled();
}

void VoxelOpenGLViewer::set_show_models(bool show)
{
    show_models_ = show;
    request_update_throttled();
}

void VoxelOpenGLViewer::update_lidar_points(std::span<const QVector3D> positions,
                                            std::span<const std::uint8_t> plane_id)
{
    std::vector<Vertex> new_vertices;
    new_vertices.reserve(positions.size());

    for (std::size_t i = 0; i < positions.size(); ++i)
    {
        const QVector3D& p = positions[i];
        const float fx = voxel_flip_x_ ? -1.f : 1.f;
        const float fy = voxel_flip_y_ ? -1.f : 1.f;
        const QVector3D mapped{fx * p.x(), p.z(), fy * p.y()};
        // Colour by source plane so the floor rings' origin is visible at a glance:
        //   helios (0) = slate blue-gray,  bpearl (1) = orange.
        const bool bpearl = (i < plane_id.size()) and (plane_id[i] == 1);
        const float r = bpearl ? 0.95f : 0.55f;
        const float g = bpearl ? 0.55f : 0.62f;
        const float b = bpearl ? 0.15f : 0.78f;
        new_vertices.push_back(Vertex{mapped.x(), mapped.y(), mapped.z(), r, g, b});
    }

    {
        std::scoped_lock lk(data_mutex_);
        lidar_vertices_ = std::move(new_vertices);
    }
    request_update_throttled();
}

void VoxelOpenGLViewer::update_residual_points(std::span<const QVector3D> residual_positions)
{
    std::vector<Vertex> residual_vertices;
    residual_vertices.reserve(residual_positions.size());
    for (const QVector3D& p : residual_positions)
    {
        const float fx = voxel_flip_x_ ? -1.f : 1.f;
        const float fy = voxel_flip_y_ ? -1.f : 1.f;
        const QVector3D mapped{fx * p.x(), p.z(), fy * p.y()};
        residual_vertices.push_back(Vertex{mapped.x(), mapped.y(), mapped.z(), 0.15f, 0.20f, 0.80f});  // residual: dark blue
    }
    {
        std::scoped_lock lk(data_mutex_);
        residual_vertices_ = std::move(residual_vertices);
    }
    request_update_throttled();
}

void VoxelOpenGLViewer::update_grid_cells(std::span<const QVector3D> cell_centres, float /*cell_size*/)
{
    std::vector<Vertex> grid_vertices;
    grid_vertices.reserve(cell_centres.size());
    for (const QVector3D& p : cell_centres)
    {
        const float fx = voxel_flip_x_ ? -1.f : 1.f;
        const float fy = voxel_flip_y_ ? -1.f : 1.f;
        const QVector3D mapped{fx * p.x(), p.z(), fy * p.y()};
        grid_vertices.push_back(Vertex{mapped.x(), mapped.y(), mapped.z(), 1.00f, 0.55f, 0.05f});  // occupancy grid: amber
    }
    {
        std::scoped_lock lk(data_mutex_);
        grid_vertices_ = std::move(grid_vertices);
    }
    request_update_throttled();
}

void VoxelOpenGLViewer::update_grid_border(std::span<const QVector3D> border_centres)
{
    std::vector<Vertex> v;
    v.reserve(border_centres.size());
    for (const QVector3D& p : border_centres)
    {
        const float fx = voxel_flip_x_ ? -1.f : 1.f;
        const float fy = voxel_flip_y_ ? -1.f : 1.f;
        const QVector3D mapped{fx * p.x(), p.z(), fy * p.y()};
        v.push_back(Vertex{mapped.x(), mapped.y(), mapped.z(), 0.10f, 0.75f, 0.85f});  // clearance border: cyan
    }
    {
        std::scoped_lock lk(data_mutex_);
        grid_border_vertices_ = std::move(v);
    }
    request_update_throttled();
}

void VoxelOpenGLViewer::set_show_grid(bool show)
{
    show_grid_ = show;
    request_update_throttled();
}

void VoxelOpenGLViewer::update_mask_points(std::span<const QVector3D> positions,
                                           std::span<const std::string> categories)
{
    std::vector<Vertex> new_vertices;
    new_vertices.reserve(positions.size());
    for (std::size_t i = 0; i < positions.size(); ++i)
    {
        const QVector3D& p = positions[i];
        const float fx = voxel_flip_x_ ? -1.f : 1.f;
        const float fy = voxel_flip_y_ ? -1.f : 1.f;
        const QVector3D mapped{fx * p.x(), p.z(), fy * p.y()};
        // Colour by class so mask points match their voxel/box colour (bottle magenta, table green);
        // fall back to off-white when no category is supplied. Brightened so the points stay legible.
        QColor c(235, 235, 242);
        if (not categories.empty() and i < categories.size())
            c = categories[i] == "bottle" ? QColor(255, 40, 40)   // bottle mask: red (contrasts the magenta cylinder)
                                          : color_for_category(categories[i]).lighter(125);
        new_vertices.push_back(Vertex{mapped.x(), mapped.y(), mapped.z(), c.redF(), c.greenF(), c.blueF()});
    }
    {
        std::scoped_lock lk(data_mutex_);
        mask_vertices_ = std::move(new_vertices);
    }
    request_update_throttled();
}

void VoxelOpenGLViewer::update_room_polygon(std::span<const float> polygon_x,
                                            std::span<const float> polygon_y)
{
    // Backward compatibility: only floor
    update_room_polygon_dual(polygon_x, polygon_y, 0.f);
}

void VoxelOpenGLViewer::update_room_polygon_dual(std::span<const float> polygon_x,
                                                 std::span<const float> polygon_y,
                                                 float height)
{
    {
        std::scoped_lock lk(room_polygon_mutex_);
        raw_polygon_x_.assign(polygon_x.begin(), polygon_x.end());
        raw_polygon_y_.assign(polygon_y.begin(), polygon_y.end());
        raw_polygon_height_ = height;
        rebuild_polygon_locked_();
    }
    request_update_throttled();
}

void VoxelOpenGLViewer::set_robot_pose(float x, float y, float theta)
{
    {
        std::scoped_lock lk(robot_pose_mutex_);
        robot_x_ = x;
        robot_y_ = y;
        robot_theta_ = theta;
        have_robot_pose_ = true;
    }
    request_update_throttled();
}

bool VoxelOpenGLViewer::load_robot_mesh(const std::string& path)
{
    const auto resolved_path = resolve_robot_mesh_path(path);
    if (!resolved_path.has_value())
    {
        qWarning() << "VoxelOpenGLViewer robot mesh not found:" << QString::fromStdString(path);
        return false;
    }

    const auto mesh = load_obj_mesh_data(resolved_path.value());
    if (!mesh.has_value())
    {
        qWarning() << "VoxelOpenGLViewer failed to load robot mesh:" << QString::fromStdString(resolved_path->string());
        return false;
    }

    const QVector3D center_xy(0.5f * (mesh->bb_min.x() + mesh->bb_max.x()),
                              0.5f * (mesh->bb_min.y() + mesh->bb_max.y()),
                              mesh->bb_min.z());

    std::vector<QVector3D> local_vertices;
    local_vertices.reserve(mesh->triangles.size());
    for (const auto& vertex : mesh->triangles)
        local_vertices.emplace_back(vertex.x() - center_xy.x(),
                                    vertex.y() - center_xy.y(),
                                    vertex.z() - center_xy.z());

    {
        std::scoped_lock lk(robot_mesh_mutex_);
        robot_mesh_local_ = std::move(local_vertices);
    }

    qInfo() << "VoxelOpenGLViewer loaded robot mesh" << QString::fromStdString(resolved_path->string())
            << "triangles=" << mesh->triangles.size() / 3;
    request_update_throttled();
    return true;
}

void VoxelOpenGLViewer::rebuild_polygon_locked_()
{
    // Apply polygon_rotation_quadrants_ * 90deg rotation around the room Z axis
    // (which maps to OpenGL Y axis after our x,z,y mapping below).
    const int q = ((polygon_rotation_quadrants_ % 4) + 4) % 4;
    const float sx = polygon_flip_x_ ? -1.f : 1.f;
    const float sy = polygon_flip_y_ ? -1.f : 1.f;
    auto rot = [q, sx, sy](float x, float y) -> std::pair<float,float> {
        x *= sx; y *= sy;
        switch (q) {
            case 0: return {x, y};
            case 1: return {-y, x};
            case 2: return {-x, -y};
            case 3: return {y, -x};
        }
        return {x, y};
    };

    const std::size_t n = std::min(raw_polygon_x_.size(), raw_polygon_y_.size());
    std::vector<QVector3D> floor, ceiling;
    floor.reserve(n);
    ceiling.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        auto [rx, ry] = rot(raw_polygon_x_[i], raw_polygon_y_[i]);
        floor.emplace_back(rx, 0.f, ry);
        ceiling.emplace_back(rx, raw_polygon_height_, ry);
    }

    // Initialize camera target from room centroid only once, then preserve
    // user camera control (pan/orbit/zoom) across periodic room updates.
    if (!room_target_initialized_ && !camera_user_moved_ && (!room_polygon_floor_.empty() || !floor.empty()))
    {
        const auto& poly = floor.empty() ? room_polygon_floor_ : floor;
        QVector3D centroid{0.f, 0.f, 0.f};
        for (const auto& p : poly) centroid += p;
        if (!poly.empty()) centroid /= static_cast<float>(poly.size());
        target_ = centroid;
        room_target_initialized_ = true;
    }

    room_polygon_floor_ = std::move(floor);
    room_polygon_ceiling_ = std::move(ceiling);
}

void VoxelOpenGLViewer::initializeGL()
{
    initializeOpenGLFunctions();
    glClearColor(0.09f, 0.10f, 0.12f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);

    static constexpr const char* vs_330 = R"(
        #version 330 core
        layout(location = 0) in vec3 in_pos;
        layout(location = 1) in vec3 in_col;
        uniform mat4 u_mvp;
        uniform float u_point_size;
        out vec3 v_col;
        void main()
        {
            gl_Position = u_mvp * vec4(in_pos, 1.0);
            gl_PointSize = u_point_size;
            v_col = in_col;
        }
    )";

    static constexpr const char* fs_330 = R"(
        #version 330 core
        in vec3 v_col;
        out vec4 out_col;
        uniform int u_round_points;
        uniform float u_alpha;
        void main()
        {
            if (u_round_points != 0)
            {
                vec2 uv = gl_PointCoord * 2.0 - 1.0;
                if (dot(uv, uv) > 1.0) discard;
            }
            out_col = vec4(v_col, u_alpha);
        }
    )";

    static constexpr const char* vs_120 = R"(
        #version 120
        attribute vec3 in_pos;
        attribute vec3 in_col;
        uniform mat4 u_mvp;
        uniform float u_point_size;
        varying vec3 v_col;
        void main()
        {
            gl_Position = u_mvp * vec4(in_pos, 1.0);
            gl_PointSize = u_point_size;
            v_col = in_col;
        }
    )";

    static constexpr const char* fs_120 = R"(
        #version 120
        varying vec3 v_col;
        uniform int u_round_points;
        uniform float u_alpha;
        void main()
        {
            if (u_round_points != 0)
            {
                vec2 uv = gl_PointCoord * 2.0 - 1.0;
                if (dot(uv, uv) > 1.0) discard;
            }
            gl_FragColor = vec4(v_col, u_alpha);
        }
    )";

    bool shader_ok = false;
    if (program_.addShaderFromSourceCode(QOpenGLShader::Vertex, vs_330)
        && program_.addShaderFromSourceCode(QOpenGLShader::Fragment, fs_330))
    {
        program_.bindAttributeLocation("in_pos", 0);
        program_.bindAttributeLocation("in_col", 1);
        if (program_.link())
        {
            shader_ok = true;
            qInfo() << "VoxelOpenGLViewer using GLSL 330";
        }
    }

    if (!shader_ok)
    {
        qWarning() << "VoxelOpenGLViewer GLSL 330 failed, trying GLSL 120:" << program_.log();
        program_.removeAllShaders();
        if (program_.addShaderFromSourceCode(QOpenGLShader::Vertex, vs_120)
            && program_.addShaderFromSourceCode(QOpenGLShader::Fragment, fs_120))
        {
            program_.bindAttributeLocation("in_pos", 0);
            program_.bindAttributeLocation("in_col", 1);
            if (program_.link())
            {
                shader_ok = true;
                qInfo() << "VoxelOpenGLViewer using GLSL 120 fallback";
            }
            else
            {
                qWarning() << "VoxelOpenGLViewer GLSL 120 link failed:" << program_.log();
            }
        }
        else
        {
            qWarning() << "VoxelOpenGLViewer GLSL 120 failed:" << program_.log();
        }
    }

    if (!shader_ok)
    {
        gl_ready_ = false;
        return;
    }

    room_vao_.create();
    room_vao_.bind();
    room_vbo_.create();
    room_vbo_.bind();
    room_vbo_.setUsagePattern(QOpenGLBuffer::DynamicDraw);
    program_.bind();
    program_.enableAttributeArray(0);
    program_.enableAttributeArray(1);
    program_.setAttributeBuffer(0, GL_FLOAT, offsetof(Vertex, px), 3, sizeof(Vertex));
    program_.setAttributeBuffer(1, GL_FLOAT, offsetof(Vertex, r), 3, sizeof(Vertex));
    program_.release();
    room_vbo_.release();
    room_vao_.release();

    gl_ready_ = true;
}

void VoxelOpenGLViewer::resizeGL(int w, int h)
{
    glViewport(0, 0, w, h);
}

void VoxelOpenGLViewer::paintGL()
{
    // [perf-probe] wall-clock cost of one paintGL, printed avg/max/count every ~3 s. paintGL runs on
    // the GUI thread, the SAME thread as compute(); compare this against [Compute] to see the split.
    const auto probe_t0 = std::chrono::steady_clock::now();

    // Real render FPS: count paints over a ~1 s window (stable; instantaneous intervals are jittery
    // because Qt event-coalesces update() requests from the render timer + data updates + expose events).
    ++render_frame_count_;
    if (fps_window_start_.time_since_epoch().count() == 0)
        fps_window_start_ = probe_t0;
    else
    {
        const double win_ms = std::chrono::duration<double, std::milli>(probe_t0 - fps_window_start_).count();
        if (win_ms >= 1000.0)
        {
            render_fps_ = static_cast<float>(render_frame_count_ * 1000.0 / win_ms);
            render_frame_count_ = 0;
            fps_window_start_ = probe_t0;
        }
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if (!gl_ready_)
        return;

    std::size_t n_lidar_vertices = 0;
    std::size_t n_residual_vertices = 0;
    std::size_t n_mask_vertices = 0;
    std::vector<Vertex> lidar_draw_vertices;
    std::vector<Vertex> residual_draw_vertices;
    std::vector<Vertex> mask_draw_vertices;
    std::vector<Vertex> grid_draw_vertices;
    std::vector<Vertex> grid_border_draw_vertices;
    {
        std::scoped_lock lk(data_mutex_);
        n_lidar_vertices = lidar_vertices_.size();
        n_residual_vertices = residual_vertices_.size();
        n_mask_vertices = mask_vertices_.size();
        lidar_draw_vertices = lidar_vertices_;
        residual_draw_vertices = residual_vertices_;
        mask_draw_vertices = mask_vertices_;
        grid_draw_vertices = grid_vertices_;
        grid_border_draw_vertices = grid_border_vertices_;
    }
    const bool has_lidar = n_lidar_vertices > 0;
    const bool has_residual = n_residual_vertices > 0;
    const bool has_mask = n_mask_vertices > 0;
    const bool has_grid = not grid_draw_vertices.empty();
    const bool has_grid_border = not grid_border_draw_vertices.empty();

    const float cp = std::cos(pitch_);
    const QVector3D eye(
        target_.x() + distance_ * cp * std::sin(yaw_),
        target_.y() + distance_ * std::sin(pitch_),
        target_.z() + distance_ * cp * std::cos(yaw_));

    QMatrix4x4 view;
    view.lookAt(eye, target_, QVector3D(0.f, 1.f, 0.f));

    QMatrix4x4 proj;
    const float aspect = height() > 0 ? static_cast<float>(width()) / static_cast<float>(height()) : 1.0f;
    proj.perspective(55.0f, aspect, 0.01f, 250.0f);

    const QMatrix4x4 mvp = proj * view;

    program_.bind();
    program_.setUniformValue("u_mvp", mvp);
    program_.setUniformValue("u_point_size", 4.5f);
    program_.setUniformValue("u_alpha", 1.0f);   // opaque by default; only the walls lower it
    program_.setUniformValue("u_round_points", 1);

    if (show_lidar_ && has_lidar)
    {
        glDisable(GL_DEPTH_TEST);
        room_vao_.bind();
        room_vbo_.bind();
        room_vbo_.allocate(lidar_draw_vertices.data(), static_cast<int>(lidar_draw_vertices.size() * sizeof(Vertex)));
        program_.setUniformValue("u_round_points", 1);
        program_.setUniformValue("u_point_size", 5.5f);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(lidar_draw_vertices.size()));
        program_.setUniformValue("u_point_size", 4.5f);
        room_vbo_.release();
        room_vao_.release();
        glEnable(GL_DEPTH_TEST);
    }

    if (has_residual && show_residual_)
    {
        glDisable(GL_DEPTH_TEST);
        room_vao_.bind();
        room_vbo_.bind();
        room_vbo_.allocate(residual_draw_vertices.data(), static_cast<int>(residual_draw_vertices.size() * sizeof(Vertex)));
        program_.setUniformValue("u_round_points", 1);
        program_.setUniformValue("u_point_size", 6.0f);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(residual_draw_vertices.size()));
        program_.setUniformValue("u_point_size", 4.5f);
        room_vbo_.release();
        room_vao_.release();
        glEnable(GL_DEPTH_TEST);
    }

    // residual_concept occupancy grid — inflated clearance BORDER first (cyan, under the obstacle), then the
    // OCCUPIED cells (amber) on top, both square cell-like points. Same `Grid` toggle.
    if (has_grid_border && show_grid_)
    {
        glDisable(GL_DEPTH_TEST);
        room_vao_.bind();
        room_vbo_.bind();
        room_vbo_.allocate(grid_border_draw_vertices.data(), static_cast<int>(grid_border_draw_vertices.size() * sizeof(Vertex)));
        program_.setUniformValue("u_round_points", 0);
        program_.setUniformValue("u_point_size", 5.0f);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(grid_border_draw_vertices.size()));
        program_.setUniformValue("u_round_points", 1);
        program_.setUniformValue("u_point_size", 4.5f);
        room_vbo_.release();
        room_vao_.release();
        glEnable(GL_DEPTH_TEST);
    }
    if (has_grid && show_grid_)
    {
        glDisable(GL_DEPTH_TEST);
        room_vao_.bind();
        room_vbo_.bind();
        room_vbo_.allocate(grid_draw_vertices.data(), static_cast<int>(grid_draw_vertices.size() * sizeof(Vertex)));
        program_.setUniformValue("u_round_points", 0);      // square points → cell-like
        program_.setUniformValue("u_point_size", 5.0f);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(grid_draw_vertices.size()));
        program_.setUniformValue("u_round_points", 1);
        program_.setUniformValue("u_point_size", 4.5f);
        room_vbo_.release();
        room_vao_.release();
        glEnable(GL_DEPTH_TEST);
    }

    // Mask points are drawn LATER (after the solid bottle cylinder + object meshes) so the
    // depth-test-off overlay isn't painted over by opaque geometry sitting on top of it — e.g.
    // from an above view the cylinder cap would otherwise hide the bottle's mask support points.

    // Draw room polygon outlines (floor and ceiling)
    std::vector<QVector3D> local_floor, local_ceiling;
    {
        std::scoped_lock lk(room_polygon_mutex_);
        local_floor = room_polygon_floor_;
        local_ceiling = room_polygon_ceiling_;
    }

    // Draw a floor grid on y=0 for orientation.
    {
        float min_x = 0.f, max_x = 0.f, min_z = 0.f, max_z = 0.f;
        bool have_bounds = false;

        if (!local_floor.empty())
        {
            min_x = max_x = local_floor.front().x();
            min_z = max_z = local_floor.front().z();
            for (const auto& p : local_floor)
            {
                min_x = std::min(min_x, p.x());
                max_x = std::max(max_x, p.x());
                min_z = std::min(min_z, p.z());
                max_z = std::max(max_z, p.z());
            }
            have_bounds = true;
        }

        if (have_bounds)
        {
            const float margin = 1.0f;
            min_x -= margin;
            max_x += margin;
            min_z -= margin;
            max_z += margin;

            constexpr float major_step = 1.0f;
            constexpr float minor_step = 0.5f;

            std::vector<Vertex> grid_vertices;
            grid_vertices.reserve(2048);

            auto push_grid = [&](float x0, float y0, float z0, float x1, float y1, float z1, const QColor& c)
            {
                grid_vertices.push_back(Vertex{x0, y0, z0, c.redF(), c.greenF(), c.blueF()});
                grid_vertices.push_back(Vertex{x1, y1, z1, c.redF(), c.greenF(), c.blueF()});
            };

            const float x_minor_start = std::floor(min_x / minor_step) * minor_step;
            const float x_minor_end   = std::ceil(max_x / minor_step) * minor_step;
            const float z_minor_start = std::floor(min_z / minor_step) * minor_step;
            const float z_minor_end   = std::ceil(max_z / minor_step) * minor_step;

            const QColor minor_col(70, 75, 82);
            const QColor major_col(110, 120, 130);
            const QColor axis_x_col(220, 70, 70);
            const QColor axis_z_col(70, 180, 220);

            for (float x = x_minor_start; x <= x_minor_end + 1e-4f; x += minor_step)
            {
                const bool is_major = std::fabs(std::fmod(x, major_step)) < 1e-4f;
                const QColor c = std::fabs(x) < 1e-4f ? axis_z_col : (is_major ? major_col : minor_col);
                push_grid(x, 0.f, z_minor_start, x, 0.f, z_minor_end, c);
            }

            for (float z = z_minor_start; z <= z_minor_end + 1e-4f; z += minor_step)
            {
                const bool is_major = std::fabs(std::fmod(z, major_step)) < 1e-4f;
                const QColor c = std::fabs(z) < 1e-4f ? axis_x_col : (is_major ? major_col : minor_col);
                push_grid(x_minor_start, 0.f, z, x_minor_end, 0.f, z, c);
            }

            glDisable(GL_DEPTH_TEST);
            room_vao_.bind();
            room_vbo_.bind();
            room_vbo_.allocate(grid_vertices.data(), static_cast<int>(grid_vertices.size() * sizeof(Vertex)));
            program_.setUniformValue("u_round_points", 0);
            glLineWidth(1.2f);
            glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(grid_vertices.size()));
            room_vbo_.release();
            room_vao_.release();
            glEnable(GL_DEPTH_TEST);
        }
    }

    auto draw_outline = [&](const std::vector<QVector3D>& poly, const QColor& line_col, const QColor& corner_col)
    {
        if (poly.empty()) return;
        std::vector<Vertex> line_vertices, corner_vertices;
        line_vertices.reserve(poly.size());
        corner_vertices.reserve(poly.size());
        for (const auto& p : poly)
        {
            line_vertices.push_back(Vertex{p.x(), p.y(), p.z(),
                                           line_col.redF(), line_col.greenF(), line_col.blueF()});
            corner_vertices.push_back(Vertex{p.x(), p.y(), p.z(),
                                             corner_col.redF(), corner_col.greenF(), corner_col.blueF()});
        }
        glDisable(GL_DEPTH_TEST);
        room_vao_.bind();
        room_vbo_.bind();
        room_vbo_.allocate(line_vertices.data(), static_cast<int>(line_vertices.size() * sizeof(Vertex)));
        program_.setUniformValue("u_round_points", 0);
        glLineWidth(4.0f);
        glDrawArrays(GL_LINE_LOOP, 0, static_cast<GLsizei>(line_vertices.size()));
        room_vbo_.allocate(corner_vertices.data(), static_cast<int>(corner_vertices.size() * sizeof(Vertex)));
        program_.setUniformValue("u_round_points", 1);
        program_.setUniformValue("u_point_size", 12.0f);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(corner_vertices.size()));
        program_.setUniformValue("u_point_size", 4.5f);
        room_vbo_.release();
        room_vao_.release();
        glEnable(GL_DEPTH_TEST);
    };

    // Floor: white, magenta corners. Ceiling: cyan, blue corners.
    draw_outline(local_floor, QColor(255,255,255), QColor(255,0,255));
    draw_outline(local_ceiling, QColor(0,255,255), QColor(0,128,255));

    // Draw vertical lines connecting floor and ceiling corners
    if (!local_floor.empty() && local_floor.size() == local_ceiling.size())
    {
        std::vector<Vertex> vertical_lines;
        vertical_lines.reserve(local_floor.size() * 2);
        QColor vert_col(255,0,255); // magenta
        for (std::size_t i = 0; i < local_floor.size(); ++i)
        {
            const auto& f = local_floor[i];
            const auto& c = local_ceiling[i];
            vertical_lines.push_back(Vertex{f.x(), f.y(), f.z(), vert_col.redF(), vert_col.greenF(), vert_col.blueF()});
            vertical_lines.push_back(Vertex{c.x(), c.y(), c.z(), vert_col.redF(), vert_col.greenF(), vert_col.blueF()});
        }
        glDisable(GL_DEPTH_TEST);
        room_vao_.bind();
        room_vbo_.bind();
        room_vbo_.allocate(vertical_lines.data(), static_cast<int>(vertical_lines.size() * sizeof(Vertex)));
        program_.setUniformValue("u_round_points", 0);
        glLineWidth(2.5f);
        glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(vertical_lines.size()));
        room_vbo_.release();
        room_vao_.release();
        glEnable(GL_DEPTH_TEST);
    }

    // Semi-transparent wall panels (floor↔ceiling quads) for a more solid, room-like impression.
    // A subtle per-wall directional shade (XZ normal vs a fixed light) keeps adjacent walls distinct.
    if (local_floor.size() == local_ceiling.size() and local_floor.size() >= 2)
    {
        const std::size_t n = local_floor.size();
        std::vector<Vertex> wall_tris;
        wall_tris.reserve(n * 6);
        constexpr float lx = 0.6f, lz = 0.8f;                 // fixed horizontal light dir (XZ)
        const QColor base(202, 204, 212);                     // soft cool gray
        for (std::size_t i = 0; i < n; ++i)
        {
            const auto& f0 = local_floor[i];
            const auto& f1 = local_floor[(i + 1) % n];
            const auto& c0 = local_ceiling[i];
            const auto& c1 = local_ceiling[(i + 1) % n];
            float nx = -(f1.z() - f0.z()), nz = (f1.x() - f0.x());   // XZ normal ⟂ the wall edge
            if (const float l = std::sqrt(nx * nx + nz * nz); l > 1e-6f) { nx /= l; nz /= l; }
            const float g = 0.55f + 0.45f * std::abs(nx * lx + nz * lz);
            const float r = base.redF() * g, gg = base.greenF() * g, b = base.blueF() * g;
            const auto V = [&](const QVector3D& p){ return Vertex{p.x(), p.y(), p.z(), r, gg, b}; };
            wall_tris.push_back(V(f0)); wall_tris.push_back(V(f1)); wall_tris.push_back(V(c1));
            wall_tris.push_back(V(f0)); wall_tris.push_back(V(c1)); wall_tris.push_back(V(c0));
        }
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDepthMask(GL_FALSE);   // transparent: test against depth but don't write it
        room_vao_.bind();
        room_vbo_.bind();
        room_vbo_.allocate(wall_tris.data(), static_cast<int>(wall_tris.size() * sizeof(Vertex)));
        program_.setUniformValue("u_round_points", 0);
        program_.setUniformValue("u_alpha", 0.22f);
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(wall_tris.size()));
        program_.setUniformValue("u_alpha", 1.0f);
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
        room_vbo_.release();
        room_vao_.release();
    }

    // Draw robot pose marker: dot + forward arrow on the floor (y=0).
    bool have_pose = false;
    float rx = 0.f, ry_room = 0.f, rtheta = 0.f;
    std::vector<QVector3D> robot_mesh_local;
    {
        std::scoped_lock lk(robot_pose_mutex_);
        have_pose = have_robot_pose_;
        rx = robot_x_;
        ry_room = robot_y_;
        rtheta = robot_theta_;
    }
    {
        std::scoped_lock lk(robot_mesh_mutex_);
        robot_mesh_local = robot_mesh_local_;
    }
    if (have_pose)
    {
        const float fx_map = voxel_flip_x_ ? -1.f : 1.f;
        const float fy_map = voxel_flip_y_ ? -1.f : 1.f;

        if (!robot_mesh_local.empty())
        {
            const float c = std::cos(rtheta);
            const float s = std::sin(rtheta);
            const QColor mesh_col(180, 190, 205);
            std::vector<Vertex> robot_mesh_vertices;
            robot_mesh_vertices.reserve(robot_mesh_local.size());
            for (const auto& local : robot_mesh_local)
            {
                const float room_x = rx + c * local.x() - s * local.y();
                const float room_y = ry_room + s * local.x() + c * local.y();
                const float room_z = local.z() + 0.01f;
                robot_mesh_vertices.push_back(Vertex{fx_map * room_x,
                                                     room_z,
                                                     fy_map * room_y,
                                                     mesh_col.redF(),
                                                     mesh_col.greenF(),
                                                     mesh_col.blueF()});
            }

            room_vao_.bind();
            room_vbo_.bind();
            room_vbo_.allocate(robot_mesh_vertices.data(), static_cast<int>(robot_mesh_vertices.size() * sizeof(Vertex)));
            program_.setUniformValue("u_round_points", 0);
            glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(robot_mesh_vertices.size()));
            room_vbo_.release();
            room_vao_.release();
        }
        else
        {
            const float ogl_x = fx_map * rx;
            const float ogl_z = fy_map * ry_room;
            const float ogl_y = 0.02f;
            const float arrow_len = 0.6f;
            const float arrow_x = fx_map * std::sin(rtheta) * arrow_len;
            const float arrow_z = fy_map * std::cos(rtheta) * arrow_len;

            std::vector<Vertex> robot_lines;
            const QColor body_col(255, 220, 0);
            const QColor arrow_col(0, 255, 0);
            robot_lines.push_back(Vertex{ogl_x - 0.15f, ogl_y, ogl_z, body_col.redF(), body_col.greenF(), body_col.blueF()});
            robot_lines.push_back(Vertex{ogl_x + 0.15f, ogl_y, ogl_z, body_col.redF(), body_col.greenF(), body_col.blueF()});
            robot_lines.push_back(Vertex{ogl_x, ogl_y, ogl_z - 0.15f, body_col.redF(), body_col.greenF(), body_col.blueF()});
            robot_lines.push_back(Vertex{ogl_x, ogl_y, ogl_z + 0.15f, body_col.redF(), body_col.greenF(), body_col.blueF()});
            robot_lines.push_back(Vertex{ogl_x, ogl_y, ogl_z, arrow_col.redF(), arrow_col.greenF(), arrow_col.blueF()});
            robot_lines.push_back(Vertex{ogl_x + arrow_x, ogl_y, ogl_z + arrow_z, arrow_col.redF(), arrow_col.greenF(), arrow_col.blueF()});

            glDisable(GL_DEPTH_TEST);
            room_vao_.bind();
            room_vbo_.bind();
            room_vbo_.allocate(robot_lines.data(), static_cast<int>(robot_lines.size() * sizeof(Vertex)));
            program_.setUniformValue("u_round_points", 0);
            glLineWidth(4.0f);
            glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(robot_lines.size()));

            Vertex dot{ogl_x, ogl_y, ogl_z, 1.0f, 1.0f, 1.0f};
            room_vbo_.allocate(&dot, sizeof(Vertex));
            program_.setUniformValue("u_round_points", 1);
            program_.setUniformValue("u_point_size", 14.0f);
            glDrawArrays(GL_POINTS, 0, 1);
            program_.setUniformValue("u_point_size", 4.5f);
            room_vbo_.release();
            room_vao_.release();
            glEnable(GL_DEPTH_TEST);
        }
    }

    // Draw DSR graph object bounding boxes (wireframe) + the solid bottle cylinder, oriented by yaw.
    if (show_models_)
    {
        std::vector<QVector3D> local_centers, local_half;
        std::vector<float> local_yaws;
        std::vector<std::string> local_cats;
        {
            std::scoped_lock lk(graph_boxes_mutex_);
            local_centers = graph_box_centers_;
            local_half = graph_box_half_extents_;
            local_yaws = graph_box_yaws_;
            local_cats = graph_box_categories_;
        }

        if (!local_centers.empty() && local_centers.size() == local_half.size())
        {
            std::vector<Vertex> box_lines;
            box_lines.reserve(local_centers.size() * 24);

            const auto map_room_to_ogl = [&](float x, float y, float z) -> QVector3D
            {
                const float fx = voxel_flip_x_ ? -1.f : 1.f;
                const float fy = voxel_flip_y_ ? -1.f : 1.f;
                return {fx * x, z, fy * y};
            };

            // Bottle (cylinder) nodes are drawn SOLID (collected here, drawn after the wireframes)
            // instead of as a box. Tessellated side + caps; a fixed room-frame light gives a little
            // rounded shading so it reads as a cylinder rather than a flat silhouette.
            std::vector<Vertex> bottle_tris;
            const auto append_cylinder = [&](const QVector3D& ctr, float radius, float half_h,
                                             const QColor& col)
            {
                constexpr int   N      = 28;
                constexpr float TWO_PI = 6.28318530718f;
                constexpr float lx = 0.3526f, ly = 0.2519f, lz = 0.9068f;   // normalised light dir
                const float cr = col.redF(), cg = col.greenF(), cb = col.blueF();
                const auto shade = [&](float nx, float ny, float nz)
                {
                    const float d = std::max(0.0f, nx * lx + ny * ly + nz * lz);
                    return 0.45f + 0.55f * d;   // ambient + diffuse
                };
                const auto vtx = [&](float x, float y, float z, float gain)
                {
                    const QVector3D p = map_room_to_ogl(x, y, z);
                    return Vertex{p.x(), p.y(), p.z(), cr * gain, cg * gain, cb * gain};
                };
                const float zb = ctr.z() - half_h, zt = ctr.z() + half_h;
                const float gt = shade(0, 0, 1), gbm = shade(0, 0, -1);
                for (int k = 0; k < N; ++k)
                {
                    const float a0 = TWO_PI * k / N, a1 = TWO_PI * (k + 1) / N;
                    const float c0 = std::cos(a0), s0 = std::sin(a0);
                    const float c1 = std::cos(a1), s1 = std::sin(a1);
                    const float x0 = ctr.x() + radius * c0, y0 = ctr.y() + radius * s0;
                    const float x1 = ctr.x() + radius * c1, y1 = ctr.y() + radius * s1;
                    const float g0 = shade(c0, s0, 0), g1 = shade(c1, s1, 0);
                    // Side wall (two triangles per segment).
                    bottle_tris.push_back(vtx(x0, y0, zb, g0));
                    bottle_tris.push_back(vtx(x1, y1, zb, g1));
                    bottle_tris.push_back(vtx(x1, y1, zt, g1));
                    bottle_tris.push_back(vtx(x0, y0, zb, g0));
                    bottle_tris.push_back(vtx(x1, y1, zt, g1));
                    bottle_tris.push_back(vtx(x0, y0, zt, g0));
                    // Top + bottom caps (triangle fans).
                    bottle_tris.push_back(vtx(ctr.x(), ctr.y(), zt, gt));
                    bottle_tris.push_back(vtx(x0, y0, zt, gt));
                    bottle_tris.push_back(vtx(x1, y1, zt, gt));
                    bottle_tris.push_back(vtx(ctr.x(), ctr.y(), zb, gbm));
                    bottle_tris.push_back(vtx(x1, y1, zb, gbm));
                    bottle_tris.push_back(vtx(x0, y0, zb, gbm));
                }
            };

            constexpr int edges[12][2] = {
                {0,1}, {1,2}, {2,3}, {3,0},
                {4,5}, {5,6}, {6,7}, {7,4},
                {0,4}, {1,5}, {2,6}, {3,7}
            };
            // Local corner sign pattern (matches the AABB ordering above so the
            // edge table stays valid): bottom face then top face.
            constexpr float sgn[8][3] = {
                {-1,-1,-1}, { 1,-1,-1}, { 1, 1,-1}, {-1, 1,-1},
                {-1,-1, 1}, { 1,-1, 1}, { 1, 1, 1}, {-1, 1, 1}
            };

            for (std::size_t i = 0; i < local_centers.size(); ++i)
            {
                const auto& ctr = local_centers[i];
                const auto& he = local_half[i];
                const float yaw = (i < local_yaws.size()) ? local_yaws[i] : 0.f;
                const float cy = std::cos(yaw);
                const float sy = std::sin(yaw);
                std::string cat;
                if (i < local_cats.size()) cat = local_cats[i];
                // Skip the green table BBs (per request) — shown via mesh/voxels/RFE points.
                if (cat == "table" or cat == "model_table")
                    continue;
                // Bottle: draw a SOLID cylinder (radius from the box footprint, full height = 2·hz)
                // instead of the wireframe box. he = (radius, radius, height/2) for a cylinder node.
                if (cat == "bottle")
                {
                    append_cylinder(ctr, 0.5f * (he.x() + he.y()), he.z(), color_for_category(cat));
                    continue;
                }
                const QColor c = color_for_category(cat);
                const float r = c.redF();
                const float g = c.greenF();
                const float b = c.blueF();

                // Rotate each local corner about Z by yaw, translate to the room-frame
                // center, then map to the OpenGL frame.
                QVector3D corners[8];
                for (int k = 0; k < 8; ++k)
                {
                    const float lx = sgn[k][0] * he.x();
                    const float ly = sgn[k][1] * he.y();
                    const float lz = sgn[k][2] * he.z();
                    const float rx = cy * lx - sy * ly;
                    const float ry = sy * lx + cy * ly;
                    corners[k] = map_room_to_ogl(ctr.x() + rx, ctr.y() + ry, ctr.z() + lz);
                }

                for (const auto& e : edges)
                {
                    const auto& a = corners[e[0]];
                    const auto& d = corners[e[1]];
                    box_lines.push_back(Vertex{a.x(), a.y(), a.z(), r, g, b});
                    box_lines.push_back(Vertex{d.x(), d.y(), d.z(), r, g, b});
                }
            }

            if (!box_lines.empty())
            {
                glDisable(GL_DEPTH_TEST);
                room_vao_.bind();
                room_vbo_.bind();
                room_vbo_.allocate(box_lines.data(), static_cast<int>(box_lines.size() * sizeof(Vertex)));
                program_.setUniformValue("u_round_points", 0);
                glLineWidth(3.0f);
                glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(box_lines.size()));
                room_vbo_.release();
                room_vao_.release();
                glEnable(GL_DEPTH_TEST);
            }

            // Solid bottle cylinders — depth-tested so they occlude properly.
            if (!bottle_tris.empty())
            {
                glEnable(GL_DEPTH_TEST);
                room_vao_.bind();
                room_vbo_.bind();
                room_vbo_.allocate(bottle_tris.data(), static_cast<int>(bottle_tris.size() * sizeof(Vertex)));
                program_.setUniformValue("u_round_points", 0);
                glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(bottle_tris.size()));
                room_vbo_.release();
                room_vao_.release();
            }
        }
    }

    // ── Object meshes (flat triangle list, room frame) — the table model ──────
    if (show_models_)
    {
        std::vector<std::vector<float>> local_meshes;
        std::vector<std::string>        local_mesh_cats;
        { std::scoped_lock lk(object_meshes_mutex_); local_meshes = object_meshes_; local_mesh_cats = object_mesh_categories_; }

        const float fx = voxel_flip_x_ ? -1.f : 1.f;
        const float fy = voxel_flip_y_ ? -1.f : 1.f;
        const QColor amber(255, 200, 100);   // established table-model colour (kept)

        for (std::size_t mi = 0; mi < local_meshes.size(); ++mi)
        {
            const auto& mesh = local_meshes[mi];
            if (mesh.size() < 9 || mesh.size() % 9 != 0) continue;

            // Per-class mesh colour: the table keeps its amber model colour; everything else (chair, …)
            // uses its class colour so the chair model matches its mask/voxels (electric blue).
            const std::string cat = mi < local_mesh_cats.size() ? local_mesh_cats[mi] : std::string{};
            const QColor mc = (cat.empty() || cat == "table") ? amber : color_for_category(cat);
            const float mc_r = mc.redF(), mc_g = mc.greenF(), mc_b = mc.blueF();

            std::vector<Vertex> mv;
            mv.reserve(mesh.size() / 3);
            for (std::size_t i = 0; i + 2 < mesh.size(); i += 3)
                mv.push_back(Vertex{fx * mesh[i], mesh[i + 2], fy * mesh[i + 1],
                                    mc_r, mc_g, mc_b});

            room_vao_.bind();
            room_vbo_.bind();
            room_vbo_.allocate(mv.data(), static_cast<int>(mv.size() * sizeof(Vertex)));
            program_.setUniformValue("u_round_points", 0);
            glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(mv.size()));
            room_vbo_.release();
            room_vao_.release();
        }
    }

    // Human skeletons (BODY_18, room frame): cyan bones + red joint points, drawn depth-test-off so
    // they stay visible over the voxels/meshes. Fed from the DSR 'person' nodes' mesh_vertices.
    if (show_skeletons_)
    {
        std::vector<std::vector<float>> local_skels;
        std::vector<QVector3D> local_facing;
        { std::scoped_lock lk(skeletons_mutex_); local_skels = skeletons_; local_facing = skeleton_facing_; }

        if (!local_skels.empty())
        {
            const float fxs = voxel_flip_x_ ? -1.f : 1.f;
            const float fys = voxel_flip_y_ ? -1.f : 1.f;
            const auto to_ogl = [&](const float* p) -> QVector3D { return {fxs * p[0], p[2], fys * p[1]}; };
            // BODY_18 edges (ZED/OpenPose) — indices match human_concept's body18.h.
            static constexpr int EDGES[17][2] = {
                {1,0}, {0,14}, {14,16}, {0,15}, {15,17},
                {1,2}, {2,3}, {3,4}, {1,5}, {5,6}, {6,7},
                {2,8}, {8,9}, {9,10}, {5,11}, {11,12}, {12,13}
            };
            constexpr int K = 18;
            // Per-BODY_18 joint colour by body region, so joint types are distinguishable:
            // head=yellow, neck=white, shoulders=green, elbows=orange, wrists=red, hips=magenta,
            // knees=blue, ankles=purple. (Bones stay cyan — no joint uses cyan.)
            static constexpr float JOINT_RGB[K][3] = {
                {1.00f, 0.85f, 0.10f},  // 0  nose        head
                {1.00f, 1.00f, 1.00f},  // 1  neck        white
                {0.20f, 0.90f, 0.20f},  // 2  r_shoulder  green
                {1.00f, 0.55f, 0.00f},  // 3  r_elbow     orange
                {1.00f, 0.15f, 0.15f},  // 4  r_wrist     red
                {0.20f, 0.90f, 0.20f},  // 5  l_shoulder  green
                {1.00f, 0.55f, 0.00f},  // 6  l_elbow     orange
                {1.00f, 0.15f, 0.15f},  // 7  l_wrist     red
                {0.90f, 0.20f, 0.90f},  // 8  r_hip       magenta
                {0.30f, 0.50f, 1.00f},  // 9  r_knee      blue
                {0.60f, 0.20f, 0.90f},  // 10 r_ankle     purple
                {0.90f, 0.20f, 0.90f},  // 11 l_hip       magenta
                {0.30f, 0.50f, 1.00f},  // 12 l_knee      blue
                {0.60f, 0.20f, 0.90f},  // 13 l_ankle     purple
                {1.00f, 0.85f, 0.10f},  // 14 r_eye       head
                {1.00f, 0.85f, 0.10f},  // 15 l_eye       head
                {1.00f, 0.85f, 0.10f},  // 16 r_ear       head
                {1.00f, 0.85f, 0.10f},  // 17 l_ear       head
            };
            // Bones drawn as 3D tubes (GL_TRIANGLES), NOT GL_LINES — glLineWidth is clamped to 1px on
            // core-profile drivers, so thickness must be real geometry.
            constexpr float BONE_R = 0.020f;   // tube radius (m)
            std::vector<Vertex> bone_tris;
            std::vector<Vertex> joint_pts;
            const auto append_tube = [](std::vector<Vertex>& out, const QVector3D& a, const QVector3D& b,
                                        float radius, float cr, float cg, float cb)
            {
                const QVector3D axis = b - a;
                const float len = axis.length();
                if (len < 1e-5f) return;
                const QVector3D n = axis / len;
                const QVector3D ref = (std::abs(n.y()) < 0.9f) ? QVector3D(0,1,0) : QVector3D(1,0,0);
                const QVector3D u = QVector3D::crossProduct(n, ref).normalized() * radius;
                const QVector3D v = QVector3D::crossProduct(n, u).normalized() * radius;
                constexpr int NS = 6;
                constexpr float TWO_PI = 6.28318530718f;
                const auto vtx = [&](const QVector3D& p){ return Vertex{p.x(), p.y(), p.z(), cr, cg, cb}; };
                for (int k = 0; k < NS; ++k)
                {
                    const float a0 = TWO_PI * k / NS, a1 = TWO_PI * (k + 1) / NS;
                    const QVector3D r0 = std::cos(a0) * u + std::sin(a0) * v;
                    const QVector3D r1 = std::cos(a1) * u + std::sin(a1) * v;
                    out.push_back(vtx(a + r0)); out.push_back(vtx(a + r1)); out.push_back(vtx(b + r1));
                    out.push_back(vtx(a + r0)); out.push_back(vtx(b + r1)); out.push_back(vtx(b + r0));
                }
            };
            // Solid cone (apex + base ring fan), for the facing-arrow head.
            const auto append_cone = [](std::vector<Vertex>& out, const QVector3D& apex, const QVector3D& base,
                                        float radius, float cr, float cg, float cb)
            {
                const QVector3D axis = apex - base;
                const float len = axis.length();
                if (len < 1e-5f) return;
                const QVector3D n = axis / len;
                const QVector3D ref = (std::abs(n.y()) < 0.9f) ? QVector3D(0,1,0) : QVector3D(1,0,0);
                const QVector3D u = QVector3D::crossProduct(n, ref).normalized() * radius;
                const QVector3D v = QVector3D::crossProduct(n, u).normalized() * radius;
                constexpr int NS = 8;
                constexpr float TWO_PI = 6.28318530718f;
                const auto vtx = [&](const QVector3D& p){ return Vertex{p.x(), p.y(), p.z(), cr, cg, cb}; };
                for (int k = 0; k < NS; ++k)
                {
                    const float a0 = TWO_PI * k / NS, a1 = TWO_PI * (k + 1) / NS;
                    const QVector3D r0 = std::cos(a0) * u + std::sin(a0) * v;
                    const QVector3D r1 = std::cos(a1) * u + std::sin(a1) * v;
                    out.push_back(vtx(apex)); out.push_back(vtx(base + r0)); out.push_back(vtx(base + r1));
                    out.push_back(vtx(base)); out.push_back(vtx(base + r1)); out.push_back(vtx(base + r0));
                }
            };
            const auto append_bone = [&](const QVector3D& a, const QVector3D& b)
            { append_tube(bone_tris, a, b, BONE_R, 0.10f, 0.95f, 0.95f); };
            for (std::size_t si = 0; si < local_skels.size(); ++si)
            {
                const auto& s = local_skels[si];
                if (static_cast<int>(s.size()) < K * 3)
                    continue;
                const auto valid = [&](int j) {
                    return std::isfinite(s[j*3]) and std::isfinite(s[j*3+1]) and std::isfinite(s[j*3+2]);
                };
                for (const auto& e : EDGES)
                    if (valid(e[0]) and valid(e[1]))
                        append_bone(to_ogl(&s[e[0]*3]), to_ogl(&s[e[1]*3]));
                for (int j = 0; j < K; ++j)
                    if (valid(j))
                    {
                        const QVector3D p = to_ogl(&s[j*3]);
                        joint_pts.push_back(Vertex{p.x(), p.y(), p.z(),
                                                   JOINT_RGB[j][0], JOINT_RGB[j][1], JOINT_RGB[j][2]});
                    }

                // Facing arrow — front/back cue. Direction is the EMA-smoothed torso normal computed in
                // update_skeletons (raw per-frame facing jitters); drawn out the chest as a white shaft
                // + RED cone at the front.
                if (valid(1) and valid(8) and valid(11)
                    and si < local_facing.size() and local_facing[si].lengthSquared() > 1e-6f)
                {
                    const auto P = [&](int j){ return QVector3D(s[j*3], s[j*3+1], s[j*3+2]); };   // room frame
                    const QVector3D chest = (P(1) + (P(8) + P(11)) * 0.5f) * 0.5f;
                    const QVector3D fwd = local_facing[si];                       // smoothed, room frame
                    constexpr float L = 0.35f, HEAD = 0.12f;
                    const auto toO = [&](const QVector3D& p){ return QVector3D(fxs * p.x(), p.z(), fys * p.y()); };
                    const QVector3D base     = toO(chest);
                    const QVector3D headBase = toO(chest + fwd * (L - HEAD));
                    const QVector3D tip      = toO(chest + fwd * L);
                    append_tube(bone_tris, base, headBase, 0.012f, 1.0f, 1.0f, 1.0f);   // white shaft
                    append_cone(bone_tris, tip, headBase, 0.045f, 1.0f, 0.15f, 0.15f);  // red front tip
                }
            }

            if (not bone_tris.empty())
            {
                glDisable(GL_DEPTH_TEST);
                room_vao_.bind();
                room_vbo_.bind();
                room_vbo_.allocate(bone_tris.data(), static_cast<int>(bone_tris.size() * sizeof(Vertex)));
                program_.setUniformValue("u_round_points", 0);
                glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(bone_tris.size()));
                room_vbo_.release();
                room_vao_.release();
                glEnable(GL_DEPTH_TEST);
            }
            if (not joint_pts.empty())
            {
                glDisable(GL_DEPTH_TEST);
                room_vao_.bind();
                room_vbo_.bind();
                room_vbo_.allocate(joint_pts.data(), static_cast<int>(joint_pts.size() * sizeof(Vertex)));
                program_.setUniformValue("u_round_points", 1);
                program_.setUniformValue("u_point_size", 6.5f);
                glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(joint_pts.size()));
                program_.setUniformValue("u_point_size", 4.5f);
                room_vbo_.release();
                room_vao_.release();
                glEnable(GL_DEPTH_TEST);
            }
        }
    }

    // Mask support points — drawn LAST as a depth-test-off overlay so they stay visible on top of
    // the solid bottle cylinder / object meshes from any view angle (notably from above).
    if (has_mask && show_masks_)
    {
        glDisable(GL_DEPTH_TEST);
        room_vao_.bind();
        room_vbo_.bind();
        room_vbo_.allocate(mask_draw_vertices.data(), static_cast<int>(mask_draw_vertices.size() * sizeof(Vertex)));
        program_.setUniformValue("u_round_points", 1);
        program_.setUniformValue("u_point_size", 6.0f);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(mask_draw_vertices.size()));
        program_.setUniformValue("u_point_size", 4.5f);
        room_vbo_.release();
        room_vao_.release();
        glEnable(GL_DEPTH_TEST);
    }

    program_.release();

    QPainter painter(this);
    painter.setRenderHint(QPainter::TextAntialiasing, true);
    painter.setPen(QColor(255, 255, 255));
    auto hz = [](float v) { return v >= 0.0f ? QString::number(v, 'f', 1) : QStringLiteral("--"); };
    painter.drawText(QRect(10, 10, width() - 20, 24),
                     Qt::AlignLeft | Qt::AlignTop,
                     QString("Render: %1 FPS   RGB: %2 Hz   RGB360: %3 Hz")
                         .arg(render_fps_ > 0.0f ? QString::number(render_fps_, 'f', 1) : QStringLiteral("--"))
                         .arg(hz(rgb_fps_))
                         .arg(hz(rgb360_fps_)));

    // [perf-probe] append paintGL cost per frame to a CSV. t_ms is a shared steady-clock stamp so this
    // aligns with viewer_perf_frames/compute/yolo on one timeline. File truncated once per launch.
    {
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - probe_t0).count();
        static std::ofstream csv = []
        {
            std::ofstream f("etc/viewer_perf_paint.csv", std::ios::trunc);
            f << "t_ms,paint_ms,lidar\n";
            return f;
        }();
        if (csv)
        {
            const long t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  t1.time_since_epoch()).count();
            csv << t_ms << ',' << ms << ','
                << static_cast<qulonglong>(n_lidar_vertices) << '\n';
            csv.flush();
        }
    }
}

void VoxelOpenGLViewer::mousePressEvent(QMouseEvent* event)
{
    last_mouse_pos_ = event->pos();
    QOpenGLWidget::mousePressEvent(event);
}

void VoxelOpenGLViewer::mouseMoveEvent(QMouseEvent* event)
{
    const QPoint d = event->pos() - last_mouse_pos_;
    last_mouse_pos_ = event->pos();

    if (event->buttons() & Qt::LeftButton)
    {
        camera_user_moved_ = true;
        yaw_   += static_cast<float>(d.x()) * 0.01f;
        pitch_ += static_cast<float>(d.y()) * 0.01f;
        pitch_ = std::clamp(pitch_, -1.45f, 1.45f);
        save_view_state();
        update();
    }
    else if (event->buttons() & Qt::RightButton)
    {
        camera_user_moved_ = true;
        const float pan_scale = 0.0025f * distance_;
        const float cp = std::cos(pitch_);
        const QVector3D eye(
            target_.x() + distance_ * cp * std::sin(yaw_),
            target_.y() + distance_ * std::sin(pitch_),
            target_.z() + distance_ * cp * std::cos(yaw_));
        const QVector3D forward = (target_ - eye).normalized();
        const QVector3D world_up(0.f, 1.f, 0.f);
        const QVector3D right = QVector3D::crossProduct(forward, world_up).normalized();
        const QVector3D camera_up = QVector3D::crossProduct(right, forward).normalized();
        target_ -= right * (static_cast<float>(d.x()) * pan_scale);
        target_ += camera_up * (static_cast<float>(d.y()) * pan_scale);
        save_view_state();
        update();
    }

    QOpenGLWidget::mouseMoveEvent(event);
}

void VoxelOpenGLViewer::wheelEvent(QWheelEvent* event)
{
    camera_user_moved_ = true;
    const float num_steps = static_cast<float>(event->angleDelta().y()) / 120.0f;
    const float scale = std::pow(0.87f, num_steps);
    distance_ = std::clamp(distance_ * scale, 0.2f, 250.0f);
    save_view_state();
    update();
    QOpenGLWidget::wheelEvent(event);
}

void VoxelOpenGLViewer::keyPressEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_R)
    {
        const bool reverse = event->modifiers() & Qt::ShiftModifier;
        {
            std::scoped_lock lk(room_polygon_mutex_);
            polygon_rotation_quadrants_ = ((polygon_rotation_quadrants_ + (reverse ? -1 : 1)) % 4 + 4) % 4;
            rebuild_polygon_locked_();
        }
        update();
        event->accept();
        return;
    }
    if (event->key() == Qt::Key_F)
    {
        {
            std::scoped_lock lk(room_polygon_mutex_);
            polygon_flip_x_ = !polygon_flip_x_;
            rebuild_polygon_locked_();
        }
        update();
        event->accept();
        return;
    }
    if (event->key() == Qt::Key_G)
    {
        {
            std::scoped_lock lk(room_polygon_mutex_);
            polygon_flip_y_ = !polygon_flip_y_;
            rebuild_polygon_locked_();
        }
        update();
        event->accept();
        return;
    }
    if (event->key() == Qt::Key_V)
    {
        voxel_flip_x_ = !voxel_flip_x_;
        event->accept();
        return;
    }
    if (event->key() == Qt::Key_B)
    {
        voxel_flip_y_ = !voxel_flip_y_;
        event->accept();
        return;
    }
    QOpenGLWidget::keyPressEvent(event);
}

QColor VoxelOpenGLViewer::color_for_category(const std::string& category)
{
    if (category == "chair") return QColor(30, 90, 255);   // electric blue (distinct from the yellowish table)
    if (category == "table") return QColor(0, 200, 60);    // green
    if (category == "model_table") return QColor(80, 220, 120); // green for graph/model tables
    if (category == "bottle") return QColor(255, 0, 200);  // hot magenta — bottle cylinder boxes
    if (category == "monitor") return QColor(186, 85, 211); // orchid-violet
    if (category == "obstacle") return QColor(255, 45, 45); // red — residual_concept obstacle boxes (obstacle=red)

    static const std::array<QColor, 20> palette = {
        QColor(220, 20, 60), QColor(0, 90, 181), QColor(34, 139, 34), QColor(255, 140, 0),
        QColor(153, 102, 204), QColor(46, 139, 87), QColor(205, 92, 92), QColor(70, 130, 180),
        QColor(255, 215, 0), QColor(199, 21, 133), QColor(95, 158, 160), QColor(176, 196, 222),
        QColor(210, 105, 30), QColor(32, 178, 170), QColor(219, 112, 147), QColor(85, 107, 47),
        QColor(218, 165, 32), QColor(106, 90, 205), QColor(205, 133, 63), QColor(0, 128, 128)
    };

    const auto hash = static_cast<std::size_t>(qHash(QString::fromStdString(category)));
    return palette[hash % palette.size()];
}

} // namespace rc
