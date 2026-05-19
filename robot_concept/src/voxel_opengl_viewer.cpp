#include "voxel_opengl_viewer.h"

#include <QHash>
#include <QDebug>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QSurfaceFormat>
#include <QTimer>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>

namespace rc
{
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
    makeCurrent();
    if (vbo_.isCreated())
        vbo_.destroy();
    if (room_vbo_.isCreated())
        room_vbo_.destroy();
    doneCurrent();
}

void VoxelOpenGLViewer::update_voxels(std::span<const QVector3D> positions,
                                      std::span<const std::string> categories,
                                      std::span<const float> confidences)
{
    const auto now = std::chrono::steady_clock::now();
    if (last_voxel_update_time_ != std::chrono::steady_clock::time_point{})
    {
        const auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_voxel_update_time_).count();
        if (dt_ms > 0)
        {
            const float inst_fps = 1000.0f / static_cast<float>(dt_ms);
            voxel_input_fps_ = (voxel_input_fps_ > 0.0f) ? (0.85f * voxel_input_fps_ + 0.15f * inst_fps) : inst_fps;
        }
    }
    last_voxel_update_time_ = now;

    std::vector<Vertex> new_vertices;
    new_vertices.reserve(positions.size());

    for (std::size_t i = 0; i < positions.size(); ++i)
    {
        const QVector3D p = positions[i];
        // RoboComp room frame -> OpenGL: X=X, Z(height)->Y, Y(depth)->Z.
        const float fx = voxel_flip_x_ ? -1.f : 1.f;
        const float fy = voxel_flip_y_ ? -1.f : 1.f;
        const QVector3D mapped{fx * p.x(), p.z(), fy * p.y()};

        QColor c = QColor(140, 145, 155);
        if (!categories.empty() && i < categories.size())
            c = color_for_category(categories[i]);

        float gain = 1.0f;
        if (!confidences.empty() && i < confidences.size())
            gain = 0.55f + 0.45f * std::clamp(confidences[i], 0.0f, 1.0f);

        new_vertices.push_back(Vertex{
            mapped.x(), mapped.y(), mapped.z(),
            c.redF() * gain, c.greenF() * gain, c.blueF() * gain
        });
    }

    if (!new_vertices.empty())
    {
        QVector3D bb_min(new_vertices.front().px, new_vertices.front().py, new_vertices.front().pz);
        QVector3D bb_max = bb_min;
        for (const auto& v : new_vertices)
        {
            bb_min.setX(std::min(bb_min.x(), v.px));
            bb_min.setY(std::min(bb_min.y(), v.py));
            bb_min.setZ(std::min(bb_min.z(), v.pz));
            bb_max.setX(std::max(bb_max.x(), v.px));
            bb_max.setY(std::max(bb_max.y(), v.py));
            bb_max.setZ(std::max(bb_max.z(), v.pz));
        }
        const float radius = 0.5f * (bb_max - bb_min).length();
        if (!first_cloud_received_)
        {
            // Only set target/distance once from the very first cloud, so the
            // camera stays stable as the robot moves and voxels accumulate.
            // If a room polygon is already loaded its centroid will override
            // this in rebuild_polygon_locked_().
            target_ = 0.5f * (bb_min + bb_max);
            distance_ = std::clamp(3.6f * std::max(0.25f, radius), 3.0f, 80.0f);
            first_cloud_received_ = true;
        }
    }

    {
        std::scoped_lock lk(data_mutex_);
        cpu_vertices_ = std::move(new_vertices);
        upload_pending_ = true;
    }
    request_update_throttled();
}

void VoxelOpenGLViewer::set_show_lidar(bool show)
{
    show_lidar_ = show;
    request_update_throttled();
}

void VoxelOpenGLViewer::update_lidar_points(std::span<const QVector3D> positions)
{
    std::vector<Vertex> new_vertices;
    new_vertices.reserve(positions.size());

    for (const QVector3D& p : positions)
    {
        const float fx = voxel_flip_x_ ? -1.f : 1.f;
        const float fy = voxel_flip_y_ ? -1.f : 1.f;
        const QVector3D mapped{fx * p.x(), p.z(), fy * p.y()};
        new_vertices.push_back(Vertex{mapped.x(), mapped.y(), mapped.z(), 0.35f, 0.95f, 0.25f});
    }

    {
        std::scoped_lock lk(data_mutex_);
        lidar_vertices_ = std::move(new_vertices);
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

void VoxelOpenGLViewer::update_track_boxes(std::span<const QVector3D> mins,
                                           std::span<const QVector3D> maxs,
                                           std::span<const std::string> categories)
{
    {
        std::scoped_lock lk(track_boxes_mutex_);
        track_box_mins_.assign(mins.begin(), mins.end());
        track_box_maxs_.assign(maxs.begin(), maxs.end());
        track_box_categories_.assign(categories.begin(), categories.end());
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
        void main()
        {
            if (u_round_points != 0)
            {
                vec2 uv = gl_PointCoord * 2.0 - 1.0;
                if (dot(uv, uv) > 1.0) discard;
            }
            out_col = vec4(v_col, 1.0);
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
        void main()
        {
            if (u_round_points != 0)
            {
                vec2 uv = gl_PointCoord * 2.0 - 1.0;
                if (dot(uv, uv) > 1.0) discard;
            }
            gl_FragColor = vec4(v_col, 1.0);
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

    vao_.create();
    vao_.bind();

    vbo_.create();
    vbo_.bind();
    vbo_.setUsagePattern(QOpenGLBuffer::DynamicDraw);
    program_.bind();
    program_.enableAttributeArray(0);
    program_.enableAttributeArray(1);
    program_.setAttributeBuffer(0, GL_FLOAT, offsetof(Vertex, px), 3, sizeof(Vertex));
    program_.setAttributeBuffer(1, GL_FLOAT, offsetof(Vertex, r), 3, sizeof(Vertex));
    program_.release();
    vbo_.release();
    vao_.release();

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
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if (!gl_ready_)
        return;

    std::vector<Vertex> local_vertices;
    bool upload_now = false;
    {
        std::scoped_lock lk(data_mutex_);
        if (upload_pending_)
        {
            local_vertices = cpu_vertices_;
            upload_pending_ = false;
            upload_now = true;
        }
    }

    if (upload_now)
    {
        vao_.bind();
        vbo_.bind();
        vbo_.allocate(local_vertices.data(), static_cast<int>(local_vertices.size() * sizeof(Vertex)));
        vbo_.release();
        vao_.release();
    }

    std::size_t n_vertices = 0;
    std::size_t n_lidar_vertices = 0;
    std::vector<Vertex> draw_vertices;
    std::vector<Vertex> lidar_draw_vertices;
    {
        std::scoped_lock lk(data_mutex_);
        n_vertices = cpu_vertices_.size();
        n_lidar_vertices = lidar_vertices_.size();
        draw_vertices = cpu_vertices_;
        lidar_draw_vertices = lidar_vertices_;
    }
    const bool has_voxels = n_vertices > 0;
    const bool has_lidar = n_lidar_vertices > 0;

    const bool draw_voxels = show_voxels_;

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
    program_.setUniformValue("u_round_points", 1);

    vao_.bind();

    if (draw_voxels && has_voxels)
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(n_vertices));

    // Compatibility fallback: if shader path silently fails on some drivers,
    // draw a second pass using fixed-function calls when available.
    if (has_voxels && context() && context()->format().profile() != QSurfaceFormat::CoreProfile)
    {
        glUseProgram(0);
        glMatrixMode(GL_PROJECTION);
        glLoadMatrixf(proj.constData());
        glMatrixMode(GL_MODELVIEW);
        glLoadMatrixf(view.constData());
        glPointSize(4.5f);
        glBegin(GL_POINTS);
        for (const auto& v : draw_vertices)
        {
            glColor3f(v.r, v.g, v.b);
            glVertex3f(v.px, v.py, v.pz);
        }
        glEnd();
    }

    if (show_lidar_ && has_lidar)
    {
        glDisable(GL_DEPTH_TEST);
        room_vao_.bind();
        room_vbo_.bind();
        room_vbo_.allocate(lidar_draw_vertices.data(), static_cast<int>(lidar_draw_vertices.size() * sizeof(Vertex)));
        program_.setUniformValue("u_round_points", 1);
        program_.setUniformValue("u_point_size", 3.0f);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(lidar_draw_vertices.size()));
        program_.setUniformValue("u_point_size", 4.5f);
        room_vbo_.release();
        room_vao_.release();
        glEnable(GL_DEPTH_TEST);
    }

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
        else if (!draw_vertices.empty())
        {
            min_x = max_x = draw_vertices.front().px;
            min_z = max_z = draw_vertices.front().pz;
            for (const auto& v : draw_vertices)
            {
                min_x = std::min(min_x, v.px);
                max_x = std::max(max_x, v.px);
                min_z = std::min(min_z, v.pz);
                max_z = std::max(max_z, v.pz);
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

    // Draw robot pose marker: dot + forward arrow on the floor (y=0).
    bool have_pose = false;
    float rx = 0.f, ry_room = 0.f, rtheta = 0.f;
    {
        std::scoped_lock lk(robot_pose_mutex_);
        have_pose = have_robot_pose_;
        rx = robot_x_;
        ry_room = robot_y_;
        rtheta = robot_theta_;
    }
    if (have_pose)
    {
        // Map room frame (x, y) to OpenGL (-x, 0, y) to match the existing
        // viewer convention used elsewhere in the project.
        const float ogl_x = -rx;
        const float ogl_z = ry_room;
        const float ogl_y = 0.02f; // slightly above the floor so it's visible
        const float arrow_len = 0.6f;
        // RoboComp convention: forward in robot frame is +Y, right is +X.
        // theta is math-CCW rotation; forward_room = R(theta) * (0, +L) = (-sin*L, cos*L).
        // After mirroring room X into OpenGL X, the arrow X delta flips sign too.
        const float fx     =  std::sin(rtheta) * arrow_len;
        const float fy_room =  std::cos(rtheta) * arrow_len;

        std::vector<Vertex> robot_lines;
        const QColor body_col(255, 220, 0);   // yellow body line (cross)
        const QColor arrow_col(0, 255, 0);    // green forward arrow
        // Cross marker at robot position (so the dot is clearly visible regardless of point size).
        robot_lines.push_back(Vertex{ogl_x - 0.15f, ogl_y, ogl_z, body_col.redF(), body_col.greenF(), body_col.blueF()});
        robot_lines.push_back(Vertex{ogl_x + 0.15f, ogl_y, ogl_z, body_col.redF(), body_col.greenF(), body_col.blueF()});
        robot_lines.push_back(Vertex{ogl_x, ogl_y, ogl_z - 0.15f, body_col.redF(), body_col.greenF(), body_col.blueF()});
        robot_lines.push_back(Vertex{ogl_x, ogl_y, ogl_z + 0.15f, body_col.redF(), body_col.greenF(), body_col.blueF()});
        // Forward arrow.
        robot_lines.push_back(Vertex{ogl_x, ogl_y, ogl_z, arrow_col.redF(), arrow_col.greenF(), arrow_col.blueF()});
        robot_lines.push_back(Vertex{ogl_x + fx, ogl_y, ogl_z + fy_room, arrow_col.redF(), arrow_col.greenF(), arrow_col.blueF()});

        glDisable(GL_DEPTH_TEST);
        room_vao_.bind();
        room_vbo_.bind();
        room_vbo_.allocate(robot_lines.data(), static_cast<int>(robot_lines.size() * sizeof(Vertex)));
        program_.setUniformValue("u_round_points", 0);
        glLineWidth(4.0f);
        glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(robot_lines.size()));

        // Center point as a large round dot.
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

    // Draw tracked object bounding boxes (wireframe).
    {
        std::vector<QVector3D> local_mins, local_maxs;
        std::vector<std::string> local_cats;
        {
            std::scoped_lock lk(track_boxes_mutex_);
            local_mins = track_box_mins_;
            local_maxs = track_box_maxs_;
            local_cats = track_box_categories_;
        }

        if (!local_mins.empty() && local_mins.size() == local_maxs.size())
        {
            std::vector<Vertex> box_lines;
            box_lines.reserve(local_mins.size() * 24);

            const auto map_room_to_ogl = [&](float x, float y, float z) -> QVector3D
            {
                const float fx = voxel_flip_x_ ? -1.f : 1.f;
                const float fy = voxel_flip_y_ ? -1.f : 1.f;
                return {fx * x, z, fy * y};
            };

            constexpr int edges[12][2] = {
                {0,1}, {1,2}, {2,3}, {3,0},
                {4,5}, {5,6}, {6,7}, {7,4},
                {0,4}, {1,5}, {2,6}, {3,7}
            };

            for (std::size_t i = 0; i < local_mins.size(); ++i)
            {
                const auto& mn = local_mins[i];
                const auto& mx = local_maxs[i];
                std::string cat;
                if (i < local_cats.size()) cat = local_cats[i];
                const QColor c = color_for_category(cat);
                const float r = c.redF();
                const float g = c.greenF();
                const float b = c.blueF();

                QVector3D corners[8] = {
                    map_room_to_ogl(mn.x(), mn.y(), mn.z()),
                    map_room_to_ogl(mx.x(), mn.y(), mn.z()),
                    map_room_to_ogl(mx.x(), mx.y(), mn.z()),
                    map_room_to_ogl(mn.x(), mx.y(), mn.z()),
                    map_room_to_ogl(mn.x(), mn.y(), mx.z()),
                    map_room_to_ogl(mx.x(), mn.y(), mx.z()),
                    map_room_to_ogl(mx.x(), mx.y(), mx.z()),
                    map_room_to_ogl(mn.x(), mx.y(), mx.z())
                };

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
                glLineWidth(2.0f);
                glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(box_lines.size()));
                room_vbo_.release();
                room_vao_.release();
                glEnable(GL_DEPTH_TEST);
            }
        }
    }

    vao_.release();
    program_.release();

    QPainter painter(this);
    painter.setRenderHint(QPainter::TextAntialiasing, true);
    painter.setPen(QColor(255, 255, 255));
    const QString fps_text = (voxel_input_fps_ > 0.0f)
                                 ? QString::number(voxel_input_fps_, 'f', 1)
                                 : QStringLiteral("--");
    painter.drawText(QRect(10, 10, width() - 20, 24),
                     Qt::AlignLeft | Qt::AlignTop,
                     QString("LiDAR points: %1   Voxels: %2   Voxel update FPS: %3")
                         .arg(static_cast<qulonglong>(n_lidar_vertices))
                         .arg(static_cast<qulonglong>(n_vertices))
                         .arg(fps_text));
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
        yaw_   -= static_cast<float>(d.x()) * 0.01f;
        pitch_ -= static_cast<float>(d.y()) * 0.01f;
        pitch_ = std::clamp(pitch_, -1.45f, 1.45f);
        update();
    }
    else if (event->buttons() & Qt::RightButton)
    {
        camera_user_moved_ = true;
        const float pan_scale = 0.0025f * distance_;
        const QVector3D right(std::cos(yaw_), 0.f, -std::sin(yaw_));
        const QVector3D up(0.f, 1.f, 0.f);
        target_ -= right * (static_cast<float>(d.x()) * pan_scale);
        target_ += up    * (static_cast<float>(d.y()) * pan_scale);
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
    if (event->key() == Qt::Key_H)
    {
        show_voxels_ = !show_voxels_;
        update();
        event->accept();
        return;
    }
    QOpenGLWidget::keyPressEvent(event);
}

QColor VoxelOpenGLViewer::color_for_category(const std::string& category)
{
    if (category == "chair") return QColor(0, 170, 255);   // cyan-blue
    if (category == "table") return QColor(255, 125, 0);   // orange
    if (category == "monitor") return QColor(186, 85, 211); // orchid-violet

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
