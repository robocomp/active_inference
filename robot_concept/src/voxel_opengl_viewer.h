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

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;

private:
    struct Vertex
    {
        float px, py, pz;
        float r, g, b;
    };

    static QColor color_for_category(const std::string& category);
    QOpenGLShaderProgram program_;
    QOpenGLVertexArrayObject vao_;
    QOpenGLBuffer vbo_{QOpenGLBuffer::VertexBuffer};

    std::vector<Vertex> cpu_vertices_;
    std::mutex data_mutex_;

    bool gl_ready_ = false;
    bool upload_pending_ = false;

    float yaw_ = 0.0f;
    float pitch_ = -0.35f;
    float distance_ = 6.0f;
    QVector3D target_{0.f, 0.f, 0.f};
    bool first_cloud_received_ = false;

    QPoint last_mouse_pos_;
};

} // namespace rc
