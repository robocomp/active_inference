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
#include <QPainter>
#include <QPen>
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

    void set_column(const Eigen::Vector3d& base, const Eigen::Vector3d& top, double radius)
    {
        column_base_   = base;
        column_top_    = top;
        column_radius_ = radius;
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

    // Build two batches each frame and submit them with two draw calls:
    //   (1) translucent triangle fill for the table's six faces, drawn with
    //       alpha blending and depth-writes disabled so the wireframe and
    //       arm meshes stay visible through it;
    //   (2) the existing opaque GL_LINES batch (wireframe edges, bottle,
    //       crosses), drawn with u_alpha = 1.
    // Same shader, same VBO — only the u_alpha uniform, blend state and
    // the (offset, count) of glDrawArrays change between passes.
    void draw_overlay_lines(const QMatrix4x4& view_proj)
    {
        if (line_shader_ == nullptr)
            return;

        overlay_vertices_.clear();
        overlay_triangles_.clear();

        const auto add_line = [&](const Eigen::Vector3d& a,
                                  const Eigen::Vector3d& b,
                                  float r, float g, float bl)
        {
            overlay_vertices_.push_back({float(a.x()), float(a.y()), float(a.z()), r, g, bl});
            overlay_vertices_.push_back({float(b.x()), float(b.y()), float(b.z()), r, g, bl});
        };
        const auto add_tri = [&](const Eigen::Vector3d& a,
                                 const Eigen::Vector3d& b,
                                 const Eigen::Vector3d& c,
                                 float r, float g, float bl)
        {
            overlay_triangles_.push_back({float(a.x()), float(a.y()), float(a.z()), r, g, bl});
            overlay_triangles_.push_back({float(b.x()), float(b.y()), float(b.z()), r, g, bl});
            overlay_triangles_.push_back({float(c.x()), float(c.y()), float(c.z()), r, g, bl});
        };

        // Table: translucent filled faces + brighter wireframe overlay.
        if (table_corners_.size() == 8)
        {
            // 6 faces × 2 triangles each. Vertex ordering doesn't matter
            // because GL_CULL_FACE is off (see initializeGL).
            static constexpr int faces[6][4] = {
                {0, 1, 2, 3}, {4, 5, 6, 7},   // bottom, top
                {0, 1, 5, 4}, {1, 2, 6, 5},   // -y, +x
                {2, 3, 7, 6}, {3, 0, 4, 7},   // +y, -x
            };
            // Warm wood tone, brighter than the wireframe so the fill reads
            // even when blended at α≈0.3.
            constexpr float fr = 0.78f, fg = 0.58f, fb = 0.35f;
            for (auto& f : faces)
            {
                add_tri(table_corners_[f[0]], table_corners_[f[1]], table_corners_[f[2]], fr, fg, fb);
                add_tri(table_corners_[f[0]], table_corners_[f[2]], table_corners_[f[3]], fr, fg, fb);
            }

            static constexpr int edges[12][2] = {
                {0, 1}, {1, 2}, {2, 3}, {3, 0},
                {4, 5}, {5, 6}, {6, 7}, {7, 4},
                {0, 4}, {1, 5}, {2, 6}, {3, 7},
            };
            // Punch up the edge brightness so they stand out against the
            // translucent fill, instead of the previous muted brown.
            for (auto& e : edges)
                add_line(table_corners_[e[0]], table_corners_[e[1]],
                         0.95f, 0.78f, 0.50f);
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

        // Vertical support column (the Webots mast from shoulder to floor):
        // grey cylinder, side-wall fill (so it reads as solid and occludes the
        // table's far edges) + wireframe rings/verticals.
        if (column_radius_ > 0.0)
        {
            const Eigen::Vector3d axis = column_top_ - column_base_;
            const double len = axis.norm();
            if (len > 1e-6)
            {
                const Eigen::Vector3d axis_n = axis / len;
                const Eigen::Vector3d ref = std::abs(axis_n.z()) < 0.9
                                            ? Eigen::Vector3d::UnitZ()
                                            : Eigen::Vector3d::UnitX();
                const Eigen::Vector3d e1 = axis_n.cross(ref).normalized();
                const Eigen::Vector3d e2 = axis_n.cross(e1).normalized();
                constexpr int N = 16;
                std::array<Eigen::Vector3d, N> cbot{}, ctop{};
                for (int i = 0; i < N; ++i)
                {
                    const double a = 2.0 * M_PI * double(i) / N;
                    const Eigen::Vector3d radial =
                        column_radius_ * (std::cos(a) * e1 + std::sin(a) * e2);
                    cbot[i] = column_base_ + radial;
                    ctop[i] = column_top_  + radial;
                }
                for (int i = 0; i < N; ++i)
                {
                    const int j = (i + 1) % N;
                    add_tri(cbot[i], cbot[j], ctop[j], 0.60f, 0.62f, 0.66f);   // side wall
                    add_tri(cbot[i], ctop[j], ctop[i], 0.60f, 0.62f, 0.66f);
                    add_line(cbot[i], cbot[j], 0.80f, 0.82f, 0.86f);           // rings
                    add_line(ctop[i], ctop[j], 0.80f, 0.82f, 0.86f);
                }
                for (int k = 0; k < 4; ++k)
                    add_line(cbot[(k * N) / 4], ctop[(k * N) / 4], 0.80f, 0.82f, 0.86f);
            }
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

        if (overlay_vertices_.empty() and overlay_triangles_.empty())
            return;

        line_shader_->bind();
        line_shader_->setUniformValue("u_view_proj", view_proj);

        // Pack triangles first (offsets [0..n_tri)), then lines (offsets
        // [n_tri..n_tri+n_line)), in a single VBO upload.
        const int n_tri  = int(overlay_triangles_.size());
        const int n_line = int(overlay_vertices_.size());

        overlay_vao_.bind();
        overlay_vbo_.bind();
        overlay_vbo_.allocate(int((n_tri + n_line) * sizeof(LineVertex)));
        if (n_tri > 0)
            overlay_vbo_.write(0,
                               overlay_triangles_.data(),
                               int(n_tri * sizeof(LineVertex)));
        if (n_line > 0)
            overlay_vbo_.write(int(n_tri * sizeof(LineVertex)),
                               overlay_vertices_.data(),
                               int(n_line * sizeof(LineVertex)));

        // Pass 1 — translucent table/column fill, WRITING depth (with a polygon
        // offset that pushes faces slightly back). This is the hidden-line step:
        // the front faces occupy the depth buffer so the box/column far edges
        // drawn next fail the depth test and are hidden, while the near edges —
        // coincident with the offset-back faces — still pass and show on top.
        if (n_tri > 0)
        {
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glEnable(GL_POLYGON_OFFSET_FILL);
            glPolygonOffset(1.0f, 1.0f);
            glDepthMask(GL_TRUE);
            line_shader_->setUniformValue("u_alpha", 0.45f);
            glDrawArrays(GL_TRIANGLES, 0, n_tri);
            glDisable(GL_POLYGON_OFFSET_FILL);
            glDisable(GL_BLEND);
        }

        // Pass 2 — overlay lines (table/column edges, bottle, crosses), depth-
        // tested against the fill above so self-occluded (far) edges are hidden.
        if (n_line > 0)
        {
            line_shader_->setUniformValue("u_alpha", 1.0f);
            glLineWidth(1.5f);
            glDrawArrays(GL_LINES, n_tri, n_line);
        }

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
            uniform float u_alpha;
            out vec4 frag_color;
            void main() { frag_color = vec4(v_color, u_alpha); }
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
    // Vertical support column (Webots mast), world frame. radius<=0 → hidden.
    Eigen::Vector3d              column_base_     = Eigen::Vector3d::Zero();
    Eigen::Vector3d              column_top_      = Eigen::Vector3d::Zero();
    double                       column_radius_   = 0.0;
    std::unique_ptr<QOpenGLShaderProgram> shader_;
    std::unique_ptr<QOpenGLShaderProgram> line_shader_;
    QOpenGLVertexArrayObject overlay_vao_;
    QOpenGLBuffer            overlay_vbo_{QOpenGLBuffer::VertexBuffer};
    std::vector<LineVertex>  overlay_vertices_;   // refilled each frame, kept to amortise allocations
    std::vector<LineVertex>  overlay_triangles_;  // translucent face fill (GL_TRIANGLES), packed in front of overlay_vertices_ in the VBO

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

// Rolling time-series of the gripper sensors under the 3D view: 4 autoscaled force channels
// (motor grip L/R + fingertip tactile L/R) plus 2 BINARY tip-bumper contacts drawn as steps
// in a strip at the top — a step = the distal tip touched the object (misaligned approach).
// Fixed ring buffer + polylines; no QtCharts dependency.
class ArmBeliefViewer3D::ForcePlot : public QWidget
{
public:
    explicit ForcePlot(QWidget* parent = nullptr) : QWidget(parent)
    {
        setMinimumHeight(130);
        setMaximumHeight(170);
    }
    // Channels 0..3 = forces (N), 4..5 = tip contacts (0/1).
    void push(float l, float r, float ltip, float rtip, float wrist, float lcontact, float rcontact)
    {
        const std::array<float, NCH> s{l, r, ltip, rtip, wrist, lcontact, rcontact};
        for (int c = 0; c < NCH; ++c)
            buf_[c][head_] = s[c];
        head_ = (head_ + 1) % CAP;
        if (count_ < CAP) ++count_;
        update();   // schedule repaint
    }

protected:
    void paintEvent(QPaintEvent*) override
    {
        QPainter p(this);
        const int W = width(), H = height();
        p.fillRect(rect(), QColor(20, 20, 24));
        p.setPen(QColor(60, 60, 70));
        p.drawLine(0, H - 1, W, H - 1);                 // force baseline
        p.drawLine(0, STRIP_H, W, STRIP_H);             // strip divider

        const int last = (head_ - 1 + CAP) % CAP;
        const auto xof = [&](int i) { return (count_ > 1) ? float(i) / (count_ - 1) * (W - 1) : 0.0f; };

        // ── Top strip: the two binary tip-bumper contacts, as steps (0 = low, 1 = up) ──
        static const QColor ccol[2] = {QColor(0, 230, 230), QColor(255, 0, 200)};
        static const char*  cnm[2]  = {"Ltip-contact", "Rtip-contact"};
        for (int k = 0; k < 2; ++k)
        {
            const int c = NFORCE + k;
            const float lo = (k == 0) ? STRIP_H - 3.0f : STRIP_H - 3.0f;  // both ride the strip floor
            const float hi = 3.0f;
            p.setPen(QPen(ccol[k], 1.6));
            QPointF prev;
            for (int i = 0; i < count_; ++i)
            {
                const int idx = (head_ - count_ + i + CAP) % CAP;
                const float y = (buf_[c][idx] > 0.5f) ? hi + k * 2 : lo - k * 2;  // tiny offset so both visible
                const QPointF pt(xof(i), y);
                if (i > 0) p.drawLine(prev, pt);
                prev = pt;
            }
            const bool on = count_ and buf_[c][last] > 0.5f;
            p.drawText(6 + k * 100, STRIP_H - 4, QString(cnm[k]) + (on ? " ON" : " -"));
        }

        // ── Force region (below the strip): autoscaled channels (4 gripper + wrist Fz) ──
        float fmax = 1e-3f;
        for (int c = 0; c < NFORCE; ++c)
            for (int i = 0; i < count_; ++i)
                fmax = std::max(fmax, std::abs(buf_[c][i]));
        static const QColor col[NFORCE] = {QColor(80, 160, 255), QColor(255, 120, 120),
                                           QColor(120, 255, 160), QColor(255, 210, 90),
                                           QColor(230, 230, 230)};
        static const char*  nm[NFORCE]  = {"Lforce", "Rforce", "Ltip", "Rtip", "wristFz"};
        for (int c = 0; c < NFORCE; ++c)
        {
            p.setPen(QPen(col[c], 1.5));
            QPointF prev;
            for (int i = 0; i < count_; ++i)
            {
                const int   idx = (head_ - count_ + i + CAP) % CAP;
                const float y   = (H - 3) - std::abs(buf_[c][idx]) / fmax * (H - STRIP_H - 16);
                const QPointF pt(xof(i), y);
                if (i > 0) p.drawLine(prev, pt);
                prev = pt;
            }
            p.drawText(6, STRIP_H + 13 + c * 12,
                       QString("%1 %2").arg(nm[c]).arg(count_ ? buf_[c][last] : 0.0f, 0, 'f', 2));
        }
        p.setPen(QColor(150, 150, 160));
        p.drawText(W - 78, STRIP_H + 13, QString("max %1 N").arg(fmax, 0, 'f', 1));
    }

private:
    static constexpr int CAP    = 300;   // ~6 s at the 50 Hz compute rate
    static constexpr int NFORCE = 5;     // Lforce, Rforce, Ltip, Rtip, wristFz
    static constexpr int NCH    = 7;     // 5 forces + 2 contacts
    static constexpr int STRIP_H = 26;   // px: top strip height for the contact steps
    std::array<std::array<float, CAP>, NCH> buf_{};
    int head_ = 0, count_ = 0;
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
    force_plot_ = new ForcePlot(this);

    main_layout->addWidget(title);
    main_layout->addWidget(start_button_);
    main_layout->addWidget(gl_panel_, 1);
    main_layout->addWidget(status_label_);
    main_layout->addWidget(force_plot_);
}

void ArmBeliefViewer3D::update_forces(float lforce, float rforce,
                                      float ltipforce, float rtipforce, float wrist_force,
                                      bool ltipcontact, bool rtipcontact)
{
    if (force_plot_ != nullptr)
        force_plot_->push(lforce, rforce, ltipforce, rtipforce, wrist_force,
                          ltipcontact ? 1.0f : 0.0f, rtipcontact ? 1.0f : 0.0f);
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

void ArmBeliefViewer3D::set_column(const Eigen::Vector3d& base,
                                   const Eigen::Vector3d& top,
                                   double radius)
{
    if (gl_panel_ != nullptr)
        gl_panel_->set_column(base, top, radius);
}

void ArmBeliefViewer3D::update_beliefs(const std::array<double, 7>& joint_angles,
                                       const std::vector<LinkPose>& link_poses,
                                       const Eigen::Vector3d& target,
                                       const Eigen::Vector3d& ee_position)
{
    if (gl_panel_ != nullptr)
        gl_panel_->set_beliefs(link_poses, target, ee_position);

    (void) joint_angles;   // q-values dropped from the label: their varying widths/signs
                           // jittered the layout each frame. Show only the (fixed-width) error.
    if (status_label_ != nullptr)
    {
        const double err = (ee_position - target).norm();
        status_label_->setText(QString::asprintf("|err| %6.3f m", err));
    }
}

void ArmBeliefViewer3D::set_mesh_root(std::string mesh_root)
{
    mesh_root_ = std::move(mesh_root);
    if (gl_panel_ != nullptr)
        gl_panel_->set_mesh_root(mesh_root_);
}
