#include "voxel_opengl_viewer.h"

#include <QHash>
#include <QDebug>
#include <QMouseEvent>
#include <QSurfaceFormat>
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
    std::vector<Vertex> new_vertices;
    new_vertices.reserve(positions.size());

    for (std::size_t i = 0; i < positions.size(); ++i)
    {
        const QVector3D p = positions[i];
        const QVector3D mapped{p.x(), p.z(), p.y()};

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
        target_ = 0.5f * (bb_min + bb_max);
        const float radius = 0.5f * (bb_max - bb_min).length();
        if (!first_cloud_received_)
        {
            distance_ = std::clamp(2.8f * std::max(0.25f, radius), 1.5f, 80.0f);
            first_cloud_received_ = true;
        }
    }

    {
        std::scoped_lock lk(data_mutex_);
        cpu_vertices_ = std::move(new_vertices);
        upload_pending_ = true;
    }
    update();
}

void VoxelOpenGLViewer::update_room_polygon(std::span<const float> polygon_x,
                                            std::span<const float> polygon_y)
{
    std::vector<QVector3D> polygon;
    const std::size_t n = std::min(polygon_x.size(), polygon_y.size());
    polygon.reserve(n);

    for (std::size_t i = 0; i < n; ++i)
    {
        // Match voxel mapping used in update_voxels: world (x, y, h) -> GL (x, h, y).
        // Room polygon is floor (h=0), with source coordinates (x, y).
        const QVector3D point{polygon_x[i], 0.f, polygon_y[i]};
        polygon.push_back(point);
    }

    {
        std::scoped_lock lk(room_polygon_mutex_);
        room_polygon_ = std::move(polygon);
    }
    update();
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
    std::vector<Vertex> draw_vertices;
    {
        std::scoped_lock lk(data_mutex_);
        n_vertices = cpu_vertices_.size();
        draw_vertices = cpu_vertices_;
    }
    if (n_vertices == 0)
        return;

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

    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(n_vertices));

    // Compatibility fallback: if shader path silently fails on some drivers,
    // draw a second pass using fixed-function calls when available.
    if (context() && context()->format().profile() != QSurfaceFormat::CoreProfile)
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

    // Draw room polygon outline
    std::vector<QVector3D> local_polygon;
    {
        std::scoped_lock lk(room_polygon_mutex_);
        if (!room_polygon_.empty())
            local_polygon = room_polygon_;
    }

    if (!local_polygon.empty())
    {
        std::vector<Vertex> line_vertices;
        std::vector<Vertex> corner_vertices;
        line_vertices.reserve(local_polygon.size());
        corner_vertices.reserve(local_polygon.size());
        for (const auto& p : local_polygon)
        {
            line_vertices.push_back(Vertex{p.x(), p.y(), p.z(), 1.0f, 1.0f, 1.0f});
            corner_vertices.push_back(Vertex{p.x(), p.y(), p.z(), 1.0f, 0.5f, 0.0f});
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
    }

    vao_.release();
    program_.release();
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
        yaw_   -= static_cast<float>(d.x()) * 0.01f;
        pitch_ -= static_cast<float>(d.y()) * 0.01f;
        pitch_ = std::clamp(pitch_, -1.45f, 1.45f);
        update();
    }
    else if (event->buttons() & Qt::RightButton)
    {
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
    const float num_steps = static_cast<float>(event->angleDelta().y()) / 120.0f;
    const float scale = std::pow(0.87f, num_steps);
    distance_ = std::clamp(distance_ * scale, 0.2f, 250.0f);
    update();
    QOpenGLWidget::wheelEvent(event);
}

QColor VoxelOpenGLViewer::color_for_category(const std::string& category)
{
    if (category == "chair") return QColor(0, 170, 255);   // cyan-blue
    if (category == "table") return QColor(255, 125, 0);   // orange

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
