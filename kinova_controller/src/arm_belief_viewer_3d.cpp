#include "arm_belief_viewer_3d.h"

#include <QColor>
#include <QDebug>
#include <QLabel>
#include <QMatrix3x3>
#include <QMatrix4x4>
#include <QOpenGLBuffer>
#include <QOpenGLContext>
#include <QOpenGLFunctions>
#include <QOpenGLShader>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWidget>
#include <QMouseEvent>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <QVector3D>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace
{
    // Initial orbital camera state (pre-interaction). Reverse-engineered from
    // the previous fixed-camera eye=(1.35,-1.10,0.85), target=(0,0,0.22):
    //   distance = ||eye-target|| ≈ 1.85 m
    //   yaw     = atan2(-1.10, 1.35) ≈ -0.683 rad
    //   pitch   = asin(0.63 / 1.85)   ≈ +0.347 rad
    constexpr QVector3D kInitialTarget{0.0f, 0.0f, 0.22f};
    constexpr float     kInitialDistance = 1.85f;
    constexpr float     kInitialYawRad   = -0.683f;
    constexpr float     kInitialPitchRad = +0.347f;

    struct TriangleFileData
    {
        Eigen::Vector3d v0;
        Eigen::Vector3d v1;
        Eigen::Vector3d v2;
        Eigen::Vector3d normal;
    };

    struct GLVertex
    {
        float px, py, pz;
        float nx, ny, nz;
    };

    struct LineVertex   // overlay GL_LINES, world-space, vertex colour.
    {
        float px, py, pz;
        float r, g, b;
    };

    QMatrix4x4 to_qmatrix4x4(const Eigen::Isometry3d& transform)
    {
        QMatrix4x4 matrix;
        const auto eigen_matrix = transform.matrix();
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 4; ++col)
                matrix(row, col) = static_cast<float>(eigen_matrix(row, col));
        return matrix;
    }

    QMatrix3x3 to_qmatrix3x3(const Eigen::Matrix3d& matrix_in)
    {
        QMatrix3x3 matrix;
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col)
                matrix(row, col) = static_cast<float>(matrix_in(row, col));
        return matrix;
    }

    bool load_binary_stl(const std::filesystem::path& path, std::vector<TriangleFileData>& out)
    {
        std::ifstream in(path, std::ios::binary);
        if (not in)
            return false;

        std::error_code ec;
        const auto file_size = std::filesystem::file_size(path, ec);
        if (ec)
            return false;

        char header[80];
        in.read(header, sizeof(header));
        if (not in)
            return false;

        std::uint32_t tri_count = 0;
        in.read(reinterpret_cast<char*>(&tri_count), sizeof(tri_count));
        if (not in)
            return false;

        const std::uint64_t expected_size = 84ULL + 50ULL * static_cast<std::uint64_t>(tri_count);
        if (expected_size != file_size)
            return false;

        out.clear();
        out.reserve(tri_count);

        for (std::uint32_t i = 0; i < tri_count; ++i)
        {
            float n[3], a[3], b[3], c[3];
            std::uint16_t attr = 0;
            in.read(reinterpret_cast<char*>(n), sizeof(n));
            in.read(reinterpret_cast<char*>(a), sizeof(a));
            in.read(reinterpret_cast<char*>(b), sizeof(b));
            in.read(reinterpret_cast<char*>(c), sizeof(c));
            in.read(reinterpret_cast<char*>(&attr), sizeof(attr));
            if (not in)
            {
                out.clear();
                return false;
            }
            (void)attr;

            TriangleFileData t;
            t.v0 = Eigen::Vector3d(a[0], a[1], a[2]);
            t.v1 = Eigen::Vector3d(b[0], b[1], b[2]);
            t.v2 = Eigen::Vector3d(c[0], c[1], c[2]);
            t.normal = Eigen::Vector3d(n[0], n[1], n[2]);
            if (t.normal.norm() < 1e-9)
                t.normal = (t.v1 - t.v0).cross(t.v2 - t.v0).normalized();
            out.push_back(t);
        }

        return not out.empty();
    }

    bool load_ascii_stl(const std::filesystem::path& path, std::vector<TriangleFileData>& out)
    {
        std::ifstream in(path);
        if (not in)
            return false;

        out.clear();
        std::string line;
        std::vector<Eigen::Vector3d> vertices;
        vertices.reserve(3);

        while (std::getline(in, line))
        {
            if (line.find("vertex") == std::string::npos)
                continue;

            std::istringstream iss(line);
            std::string keyword;
            double x = 0.0, y = 0.0, z = 0.0;
            iss >> keyword >> x >> y >> z;
            if (keyword != "vertex")
                continue;

            vertices.emplace_back(x, y, z);
            if (vertices.size() == 3)
            {
                TriangleFileData t;
                t.v0 = vertices[0];
                t.v1 = vertices[1];
                t.v2 = vertices[2];
                t.normal = (t.v1 - t.v0).cross(t.v2 - t.v0).normalized();
                out.push_back(t);
                vertices.clear();
            }
        }

        return not out.empty();
    }

    bool load_stl(const std::filesystem::path& path, std::vector<TriangleFileData>& out)
    {
        if (load_binary_stl(path, out))
            return true;
        return load_ascii_stl(path, out);
    }
}

class ArmBeliefViewer3D::GLPanel : public QOpenGLWidget, protected QOpenGLFunctions
{
public:
    struct MeshGpu
    {
        QOpenGLVertexArrayObject vao;
        QOpenGLBuffer vbo{QOpenGLBuffer::VertexBuffer};
        int vertex_count = 0;
    };

    explicit GLPanel(QWidget* parent = nullptr)
        : QOpenGLWidget(parent)
    {
        setMinimumSize(460, 300);
    }

    ~GLPanel() override
    {
        release_gl_resources();
    }

    void set_mesh_root(const std::string& mesh_root)
    {
        mesh_root_ = mesh_root;
        if (QOpenGLContext::currentContext() == context() and context() != nullptr)
            destroy_cached_meshes();
        mesh_cache_.clear();
        mesh_load_errors_.clear();
        update();
    }

    void set_beliefs(const std::vector<ArmBeliefViewer3D::LinkPose>& link_poses,
                     const Eigen::Vector3d& target,
                     const Eigen::Vector3d& ee_position)
    {
        link_poses_ = link_poses;
        // Cache the per-link conversions here, not inside paintGL — they only
        // change when the agent posts new beliefs. linear() is fed directly as
        // the normal matrix: arm link poses are rigid, so R⁻ᵀ = R and the
        // explicit inverse-transpose was wasted work.
        cached_model_.resize(link_poses_.size());
        cached_normal_.resize(link_poses_.size());
        for (size_t i = 0; i < link_poses_.size(); ++i)
        {
            cached_model_[i]  = to_qmatrix4x4(link_poses_[i].pose);
            cached_normal_[i] = to_qmatrix3x3(link_poses_[i].pose.linear());
        }
        target_ = target;
        ee_position_ = ee_position;
        update();
    }

    void set_scene_objects(const std::vector<Eigen::Vector3d>& table_corners,
                           const Eigen::Vector3d& bottle_origin,
                           const Eigen::Vector3d& bottle_axis,
                           double bottle_radius,
                           double bottle_height)
    {
        table_corners_   = table_corners;
        bottle_origin_   = bottle_origin;
        bottle_axis_     = bottle_axis;
        bottle_radius_   = bottle_radius;
        bottle_height_   = bottle_height;
        update();
    }

protected:
    void initializeGL() override
    {
        initializeOpenGLFunctions();
        glEnable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);
        glClearColor(0.06f, 0.07f, 0.09f, 1.0f);
        init_shader_program();
        init_line_shader();
        init_overlay_buffers();
    }

    void resizeGL(int width, int height) override
    {
        glViewport(0, 0, width, std::max(1, height));
    }

    // ── Webots-style camera controls ───────────────────────────────────
    // Left  drag → orbit (yaw/pitch around target).
    // Right drag → pan   (translate both eye and target in view plane).
    // Wheel      → zoom  (dolly in/out along view direction).
    void mousePressEvent(QMouseEvent* ev) override
    {
        last_mouse_pos_ = ev->pos();
        ev->accept();
    }

    void mouseMoveEvent(QMouseEvent* ev) override
    {
        const QPoint delta = ev->pos() - last_mouse_pos_;
        last_mouse_pos_ = ev->pos();

        if (ev->buttons() & Qt::LeftButton)
        {
            constexpr float kRotateRadPerPixel = 0.008f;
            camera_yaw_   += static_cast<float>(delta.x()) * kRotateRadPerPixel;
            camera_pitch_ -= static_cast<float>(delta.y()) * kRotateRadPerPixel;
            constexpr float kPitchLimit = 1.5533f;  // ~89° — keep "up" well-defined.
            camera_pitch_ = std::clamp(camera_pitch_, -kPitchLimit, +kPitchLimit);
            update();
        }
        else if (ev->buttons() & Qt::RightButton)
        {
            // Translate target+eye together along the view's right/up axes.
            // Pan speed scales with distance so it feels constant at any zoom.
            const QVector3D forward = (camera_target_ - camera_eye()).normalized();
            const QVector3D right   = QVector3D::crossProduct(forward, QVector3D(0, 0, 1)).normalized();
            const QVector3D up      = QVector3D::crossProduct(right, forward).normalized();
            const float pan_scale   = camera_distance_ * 0.0015f;
            camera_target_ += -right * static_cast<float>(delta.x()) * pan_scale
                              + up   * static_cast<float>(delta.y()) * pan_scale;
            update();
        }
        ev->accept();
    }

    void wheelEvent(QWheelEvent* ev) override
    {
        // angleDelta().y() is ±120 per notch on a standard wheel.
        const float notches = static_cast<float>(ev->angleDelta().y()) / 120.0f;
        const float factor  = std::pow(0.9f, notches);   // wheel up → notches>0 → zoom in
        camera_distance_ = std::clamp(camera_distance_ * factor, 0.15f, 10.0f);
        update();
        ev->accept();
    }

    void paintGL() override
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (shader_ == nullptr)
            return;

        QMatrix4x4 projection;
        projection.perspective(45.0f,
                               static_cast<float>(width()) / static_cast<float>(std::max(1, height())),
                               0.01f,
                               6.0f);

        QMatrix4x4 view;
        view.lookAt(camera_eye(), camera_target_, QVector3D(0.0f, 0.0f, 1.0f));
        const QMatrix4x4 view_proj = projection * view;

        shader_->bind();
        shader_->setUniformValue("u_view_proj", view_proj);
        shader_->setUniformValue("u_light_dir", QVector3D(0.35f, -0.45f, 1.0f));

        for (size_t link_idx = 0; link_idx < link_poses_.size(); ++link_idx)
        {
            auto* mesh = load_mesh_if_needed(link_poses_[link_idx].mesh_filename);
            if (mesh == nullptr or mesh->vertex_count == 0)
                continue;

            const QColor base_color = QColor::fromHsv((static_cast<int>(link_idx) * 35) % 360, 110, 230);
            shader_->setUniformValue("u_model", cached_model_[link_idx]);
            shader_->setUniformValue("u_normal_matrix", cached_normal_[link_idx]);
            shader_->setUniformValue("u_color", QVector3D(base_color.redF(), base_color.greenF(), base_color.blueF()));

            mesh->vao.bind();
            glDrawArrays(GL_TRIANGLES, 0, mesh->vertex_count);
            mesh->vao.release();
        }

        shader_->release();
        draw_overlay_lines(view_proj);
    }

private:
    void init_shader_program()
    {
        shader_ = std::make_unique<QOpenGLShaderProgram>();
        static constexpr const char* kVertexShader = R"(
            #version 330 core
            layout(location = 0) in vec3 a_pos;
            layout(location = 1) in vec3 a_normal;

            uniform mat4 u_view_proj;
            uniform mat4 u_model;
            uniform mat3 u_normal_matrix;

            out vec3 v_normal;

            void main()
            {
                gl_Position = u_view_proj * u_model * vec4(a_pos, 1.0);
                v_normal = normalize(u_normal_matrix * a_normal);
            }
        )";

        static constexpr const char* kFragmentShader = R"(
            #version 330 core
            in vec3 v_normal;

            uniform vec3 u_light_dir;
            uniform vec3 u_color;

            out vec4 frag_color;

            void main()
            {
                float diffuse = max(dot(normalize(v_normal), normalize(u_light_dir)), 0.0);
                float ambient = 0.20;
                vec3 lit = (ambient + 0.80 * diffuse) * u_color;
                frag_color = vec4(lit, 1.0);
            }
        )";

        if (not shader_->addShaderFromSourceCode(QOpenGLShader::Vertex, kVertexShader))
        {
            qWarning() << "ArmBeliefViewer3D vertex shader error:" << shader_->log();
            shader_.reset();
            return;
        }
        if (not shader_->addShaderFromSourceCode(QOpenGLShader::Fragment, kFragmentShader))
        {
            qWarning() << "ArmBeliefViewer3D fragment shader error:" << shader_->log();
            shader_.reset();
            return;
        }
        if (not shader_->link())
        {
            qWarning() << "ArmBeliefViewer3D shader link error:" << shader_->log();
            shader_.reset();
        }
    }

    // Build a single GL_LINES batch each frame and submit it with one draw
    // call. Replaces a QPainter overlay, which on QOpenGLWidget pays a heavy
    // per-frame state-flush cost regardless of how few primitives it draws.
    void draw_overlay_lines(const QMatrix4x4& view_proj)
    {
        if (line_shader_ == nullptr)
            return;

        overlay_vertices_.clear();
        const auto add_line = [&](const Eigen::Vector3d& a,
                                  const Eigen::Vector3d& b,
                                  float r, float g, float bl)
        {
            overlay_vertices_.push_back({float(a.x()), float(a.y()), float(a.z()), r, g, bl});
            overlay_vertices_.push_back({float(b.x()), float(b.y()), float(b.z()), r, g, bl});
        };

        // Table wireframe (12 edges).
        if (table_corners_.size() == 8)
        {
            static constexpr int edges[12][2] = {
                {0, 1}, {1, 2}, {2, 3}, {3, 0},
                {4, 5}, {5, 6}, {6, 7}, {7, 4},
                {0, 4}, {1, 5}, {2, 6}, {3, 7},
            };
            for (auto& e : edges)
                add_line(table_corners_[e[0]], table_corners_[e[1]],
                         0.667f, 0.510f, 0.353f);
        }

        // Bottle cylinder (two rims + four verticals).
        if (bottle_radius_ > 0.0 and bottle_height_ > 0.0)
        {
            const Eigen::Vector3d axis_n = bottle_axis_.normalized();
            const Eigen::Vector3d ref    = std::abs(axis_n.z()) < 0.9
                                           ? Eigen::Vector3d::UnitZ()
                                           : Eigen::Vector3d::UnitX();
            const Eigen::Vector3d e1 = axis_n.cross(ref).normalized();
            const Eigen::Vector3d e2 = axis_n.cross(e1).normalized();
            const Eigen::Vector3d top_centre = bottle_origin_ + axis_n * bottle_height_;

            constexpr int N = 20;
            std::array<Eigen::Vector3d, N> bot{}, top{};
            for (int i = 0; i < N; ++i)
            {
                const double a = 2.0 * M_PI * double(i) / N;
                const Eigen::Vector3d radial =
                    bottle_radius_ * (std::cos(a) * e1 + std::sin(a) * e2);
                bot[i] = bottle_origin_ + radial;
                top[i] = top_centre     + radial;
            }
            for (int i = 0; i < N; ++i)
            {
                const int j = (i + 1) % N;
                add_line(bot[i], bot[j], 0.314f, 0.863f, 0.863f);
                add_line(top[i], top[j], 0.314f, 0.863f, 0.863f);
            }
            for (int k = 0; k < 4; ++k)
                add_line(bot[(k * N) / 4], top[(k * N) / 4], 0.314f, 0.863f, 0.863f);
        }

        // 3-axis cross markers in world space. Length scales with camera
        // distance so the marker stays a consistent on-screen size regardless
        // of zoom — the same intent the QPainter version had via its screen-
        // space ±7 px lines.
        const double m_per_px = double(camera_distance_) * 0.0009;
        const double tgt_h    = 7.0 * m_per_px;
        const double ee_h     = 4.0 * m_per_px;

        const auto cross3d = [&](const Eigen::Vector3d& c, double h,
                                 float r, float g, float b)
        {
            add_line(c - Eigen::Vector3d(h, 0, 0), c + Eigen::Vector3d(h, 0, 0), r, g, b);
            add_line(c - Eigen::Vector3d(0, h, 0), c + Eigen::Vector3d(0, h, 0), r, g, b);
            add_line(c - Eigen::Vector3d(0, 0, h), c + Eigen::Vector3d(0, 0, h), r, g, b);
        };
        cross3d(target_,      tgt_h, 1.000f, 0.314f, 0.314f);  // target (red)
        cross3d(ee_position_, ee_h,  1.000f, 0.784f, 0.000f);  // EE    (yellow)

        if (overlay_vertices_.empty())
            return;

        line_shader_->bind();
        line_shader_->setUniformValue("u_view_proj", view_proj);

        overlay_vao_.bind();
        overlay_vbo_.bind();
        // Single dynamic upload; orphans the previous storage so the driver
        // can pipeline frames without forcing a CPU↔GPU sync.
        overlay_vbo_.allocate(overlay_vertices_.data(),
                              int(overlay_vertices_.size() * sizeof(LineVertex)));
        glLineWidth(1.5f);
        glDrawArrays(GL_LINES, 0, int(overlay_vertices_.size()));
        overlay_vbo_.release();
        overlay_vao_.release();
        line_shader_->release();
    }

    void init_line_shader()
    {
        line_shader_ = std::make_unique<QOpenGLShaderProgram>();
        static constexpr const char* kV = R"(
            #version 330 core
            layout(location = 0) in vec3 a_pos;
            layout(location = 1) in vec3 a_color;
            uniform mat4 u_view_proj;
            out vec3 v_color;
            void main() {
                gl_Position = u_view_proj * vec4(a_pos, 1.0);
                v_color = a_color;
            }
        )";
        static constexpr const char* kF = R"(
            #version 330 core
            in vec3 v_color;
            out vec4 frag_color;
            void main() { frag_color = vec4(v_color, 1.0); }
        )";
        if (not line_shader_->addShaderFromSourceCode(QOpenGLShader::Vertex,   kV) or
            not line_shader_->addShaderFromSourceCode(QOpenGLShader::Fragment, kF) or
            not line_shader_->link())
        {
            qWarning() << "ArmBeliefViewer3D line shader error:" << line_shader_->log();
            line_shader_.reset();
        }
    }

    void init_overlay_buffers()
    {
        overlay_vao_.create();
        overlay_vao_.bind();
        overlay_vbo_.create();
        overlay_vbo_.bind();
        overlay_vbo_.setUsagePattern(QOpenGLBuffer::DynamicDraw);
        // Allocate placeholder storage; draw_overlay_lines re-allocates per
        // frame (cheap; the driver reuses orphaned VBOs).
        overlay_vbo_.allocate(nullptr, int(256 * sizeof(LineVertex)));
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(LineVertex),
                              reinterpret_cast<const void*>(offsetof(LineVertex, px)));
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(LineVertex),
                              reinterpret_cast<const void*>(offsetof(LineVertex, r)));
        glEnableVertexAttribArray(1);
        overlay_vbo_.release();
        overlay_vao_.release();
    }

    MeshGpu* load_mesh_if_needed(const std::string& mesh_filename)
    {
        if (auto it = mesh_cache_.find(mesh_filename); it != mesh_cache_.end())
            return it->second.get();
        if (mesh_load_errors_.contains(mesh_filename) or mesh_root_.empty())
            return nullptr;

        std::filesystem::path path = std::filesystem::path(mesh_root_) / mesh_filename;
        std::vector<TriangleFileData> triangles;
        if (not load_stl(path, triangles))
        {
            mesh_load_errors_.insert(mesh_filename);
            return nullptr;
        }

        std::vector<GLVertex> vertices;
        vertices.reserve(triangles.size() * 3);
        for (const auto& tri : triangles)
        {
            vertices.push_back(GLVertex{static_cast<float>(tri.v0.x()), static_cast<float>(tri.v0.y()), static_cast<float>(tri.v0.z()),
                                        static_cast<float>(tri.normal.x()), static_cast<float>(tri.normal.y()), static_cast<float>(tri.normal.z())});
            vertices.push_back(GLVertex{static_cast<float>(tri.v1.x()), static_cast<float>(tri.v1.y()), static_cast<float>(tri.v1.z()),
                                        static_cast<float>(tri.normal.x()), static_cast<float>(tri.normal.y()), static_cast<float>(tri.normal.z())});
            vertices.push_back(GLVertex{static_cast<float>(tri.v2.x()), static_cast<float>(tri.v2.y()), static_cast<float>(tri.v2.z()),
                                        static_cast<float>(tri.normal.x()), static_cast<float>(tri.normal.y()), static_cast<float>(tri.normal.z())});
        }

        auto mesh = std::make_unique<MeshGpu>();
        mesh->vao.create();
        mesh->vao.bind();

        mesh->vbo.create();
        mesh->vbo.bind();
        mesh->vbo.setUsagePattern(QOpenGLBuffer::StaticDraw);
        mesh->vbo.allocate(vertices.data(), static_cast<int>(vertices.size() * sizeof(GLVertex)));

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GLVertex), reinterpret_cast<const void*>(offsetof(GLVertex, px)));
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(GLVertex), reinterpret_cast<const void*>(offsetof(GLVertex, nx)));
        glEnableVertexAttribArray(1);

        mesh->vbo.release();
        mesh->vao.release();
        mesh->vertex_count = static_cast<int>(vertices.size());

        auto* mesh_ptr = mesh.get();
        mesh_cache_.insert_or_assign(mesh_filename, std::move(mesh));
        return mesh_ptr;
    }

    void destroy_cached_meshes()
    {
        for (auto& [name, mesh] : mesh_cache_)
        {
            (void)name;
            if (mesh != nullptr)
            {
                mesh->vao.destroy();
                mesh->vbo.destroy();
            }
        }
    }

    void release_gl_resources()
    {
        if (context() == nullptr)
            return;

        makeCurrent();
        destroy_cached_meshes();
        mesh_cache_.clear();
        shader_.reset();
        line_shader_.reset();
        if (overlay_vao_.isCreated()) overlay_vao_.destroy();
        if (overlay_vbo_.isCreated()) overlay_vbo_.destroy();
        doneCurrent();
    }

    std::vector<ArmBeliefViewer3D::LinkPose> link_poses_;
    // Cached Eigen → Qt conversions, refreshed only when set_beliefs() runs.
    std::vector<QMatrix4x4> cached_model_;
    std::vector<QMatrix3x3> cached_normal_;
    std::unordered_map<std::string, std::unique_ptr<MeshGpu>> mesh_cache_;
    std::unordered_set<std::string> mesh_load_errors_;
    std::string mesh_root_;

    Eigen::Vector3d target_      = Eigen::Vector3d::Zero();
    Eigen::Vector3d ee_position_ = Eigen::Vector3d::Zero();
    // Scene objects in robot frame (metres). Empty until the agent feeds them.
    std::vector<Eigen::Vector3d> table_corners_;
    Eigen::Vector3d              bottle_origin_   = Eigen::Vector3d::Zero();
    Eigen::Vector3d              bottle_axis_     = Eigen::Vector3d::UnitZ();
    double                       bottle_radius_   = 0.0;
    double                       bottle_height_   = 0.0;
    std::unique_ptr<QOpenGLShaderProgram> shader_;
    std::unique_ptr<QOpenGLShaderProgram> line_shader_;
    QOpenGLVertexArrayObject overlay_vao_;
    QOpenGLBuffer            overlay_vbo_{QOpenGLBuffer::VertexBuffer};
    std::vector<LineVertex>  overlay_vertices_;   // refilled each frame, kept to amortise allocations

    // Orbital camera state (Webots convention).
    QVector3D camera_target_   = kInitialTarget;
    float     camera_distance_ = kInitialDistance;
    float     camera_yaw_      = kInitialYawRad;
    float     camera_pitch_    = kInitialPitchRad;
    QPoint    last_mouse_pos_;

    QVector3D camera_eye() const
    {
        const float cp = std::cos(camera_pitch_);
        const float sp = std::sin(camera_pitch_);
        const float cy = std::cos(camera_yaw_);
        const float sy = std::sin(camera_yaw_);
        return camera_target_ + camera_distance_ * QVector3D(cp * cy, cp * sy, sp);
    }
};

ArmBeliefViewer3D::ArmBeliefViewer3D(QWidget* parent)
    : QWidget(parent)
{
    auto* main_layout = new QVBoxLayout(this);
    main_layout->setContentsMargins(6, 6, 6, 6);
    main_layout->setSpacing(4);

    auto* title = new QLabel("Kinova beliefs: 3D arm state", this);
    QFont title_font = title->font();
    title_font.setBold(true);
    title->setFont(title_font);

    start_button_ = new QPushButton("Start bottle approach", this);
    start_button_->setMaximumHeight(35);
    start_button_->setCheckable(true);
    connect(start_button_, &QPushButton::toggled, this, [this](bool checked) {
        start_button_->setText(checked ? "Stop — return to rest"
                                       : "Start bottle approach");
        emit run_state_changed(checked);
    });

    gl_panel_ = new GLPanel(this);
    status_label_ = new QLabel("Waiting for first belief update...", this);
    status_label_->setTextInteractionFlags(Qt::TextSelectableByMouse);

    main_layout->addWidget(title);
    main_layout->addWidget(start_button_);
    main_layout->addWidget(gl_panel_, 1);
    main_layout->addWidget(status_label_);
}

ArmBeliefViewer3D::~ArmBeliefViewer3D() = default;

void ArmBeliefViewer3D::update_scene_objects(const std::vector<Eigen::Vector3d>& table_corners,
                                             const Eigen::Vector3d& bottle_origin,
                                             const Eigen::Vector3d& bottle_axis,
                                             double bottle_radius,
                                             double bottle_height)
{
    if (gl_panel_ != nullptr)
        gl_panel_->set_scene_objects(table_corners, bottle_origin, bottle_axis,
                                     bottle_radius, bottle_height);
}

void ArmBeliefViewer3D::update_beliefs(const std::array<double, 7>& joint_angles,
                                       const std::vector<LinkPose>& link_poses,
                                       const Eigen::Vector3d& target,
                                       const Eigen::Vector3d& ee_position)
{
    if (gl_panel_ != nullptr)
        gl_panel_->set_beliefs(link_poses, target, ee_position);

    if (status_label_ != nullptr)
    {
        const double err = (ee_position - target).norm();
        status_label_->setText(QString::asprintf(
            "q1 %.2f  q2 %.2f  q3 %.2f  q4 %.2f  q5 %.2f  q6 %.2f  q7 %.2f   |err| %.3f m",
            joint_angles[0], joint_angles[1], joint_angles[2], joint_angles[3],
            joint_angles[4], joint_angles[5], joint_angles[6], err));
    }
}

void ArmBeliefViewer3D::set_mesh_root(std::string mesh_root)
{
    mesh_root_ = std::move(mesh_root);
    if (gl_panel_ != nullptr)
        gl_panel_->set_mesh_root(mesh_root_);
}
