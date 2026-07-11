/*
 *  Reusable 3D point-cloud viewer (active_inference/common).
 *
 *  The OpenGL rendering (shaders, orbit/pan/zoom camera, auto-framing, XYZ axes) is copied from
 *  cortex dsr_gui's GraphNodeLaserWidget, but the graph coupling is removed: instead of reading
 *  laser_X/Y/Z off a DSR node, points are pushed in through set_points(). That makes it a generic
 *  widget any active_inference agent can drive from whatever source it owns — in particular a
 *  media-plane LiDAR subscriber (robot_concept). Living in common (next to media_transport) avoids
 *  the cortex→active_inference dependency inversion a cortex-resident media viewer would need.
 *
 *  Deliberately NON-Q_OBJECT: it has no signals/slots of its own (the owner drives it via
 *  set_points()), so it needs no MOC pass when #included into an agent source. Two point groups
 *  (e.g. helios + bpearl) are drawn in distinct colours; pass an empty second span for one cloud.
 */
#ifndef RC_COMMON_GL_POINT_CLOUD_VIEWER_H
#define RC_COMMON_GL_POINT_CLOUD_VIEWER_H

#include <QKeyEvent>
#include <QMatrix4x4>
#include <QMouseEvent>
#include <QOpenGLBuffer>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWidget>
#include <QPainter>
#include <QVector2D>
#include <QVector3D>
#include <QWheelEvent>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace rc::viewers
{

class GLPointCloudViewer : public QOpenGLWidget, protected QOpenGLFunctions
{
public:
	explicit GLPointCloudViewer(QWidget *parent = nullptr) : QOpenGLWidget(parent)
	{
		resize(640, 640);
		setWindowTitle("point cloud");
		setFocusPolicy(Qt::StrongFocus);
		axis_vertices_ = makeAxes(1.0f);
	}

	~GLPointCloudViewer() override
	{
		if(context())
		{
			makeCurrent();
			if(points_vbo_.isCreated()) points_vbo_.destroy();
			if(axis_vbo_.isCreated())   axis_vbo_.destroy();
			if(vao_.isCreated())        vao_.destroy();
			doneCurrent();
		}
	}

	// Replace the displayed cloud. group_a and group_b are drawn in distinct colours (e.g. two
	// LiDAR rings); pass an empty group_b for a single cloud. Auto-frames on the first non-empty
	// cloud until the user interacts with the camera.
	void set_points(std::span<const QVector3D> group_a, std::span<const QVector3D> group_b = {})
	{
		point_vertices_.clear();
		point_vertices_.reserve(group_a.size() + group_b.size());

		float minx = std::numeric_limits<float>::max(), miny = minx, minz = minx;
		float maxx = std::numeric_limits<float>::lowest(), maxy = maxx, maxz = maxx;
		const auto add = [&](std::span<const QVector3D> g)
		{
			for(const QVector3D &p : g)
			{
				if(not (std::isfinite(p.x()) and std::isfinite(p.y()) and std::isfinite(p.z())))
					continue;
				point_vertices_.push_back(p);
				minx = std::min(minx, p.x()); maxx = std::max(maxx, p.x());
				miny = std::min(miny, p.y()); maxy = std::max(maxy, p.y());
				minz = std::min(minz, p.z()); maxz = std::max(maxz, p.z());
			}
		};
		add(group_a);
		group_a_count_ = point_vertices_.size();   // split index for two-colour draw
		add(group_b);

		// Rolling display-rate estimate from set_points() call intervals.
		const auto now = std::chrono::steady_clock::now();
		if(have_last_)
		{
			const double dt_ms = std::chrono::duration<double, std::milli>(now - last_).count();
			if(dt_ms > 0.0 and dt_ms < 10000.0)
			{
				const float inst = 1000.0f / static_cast<float>(dt_ms);
				fps_ = (fps_ > 0.0f) ? (0.85f * fps_ + 0.15f * inst) : inst;
			}
		}
		last_ = now; have_last_ = true;

		if(not point_vertices_.empty())
		{
			center_x_ = (minx + maxx) * 0.5f;
			center_y_ = (miny + maxy) * 0.5f;
			center_z_ = (minz + maxz) * 0.5f;
			const float rx = std::max(0.001f, (maxx - minx) * 0.5f);
			const float ry = std::max(0.001f, (maxy - miny) * 0.5f);
			const float rz = std::max(0.001f, (maxz - minz) * 0.5f);
			cloud_radius_ = std::max({rx, ry, rz});
			if(not user_interacted_)
				cam_dist_ = std::max(2.0f, cloud_radius_ * 3.0f);
		}
		axis_vertices_ = makeAxes(point_vertices_.empty() ? 1.0f : cloud_radius_);
		updateGpuBuffers();
		update();
	}

protected:
	void initializeGL() override
	{
		initializeOpenGLFunctions();
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_PROGRAM_POINT_SIZE);
		glClearColor(0.06f, 0.06f, 0.07f, 1.0f);

		program_.addShaderFromSourceCode(QOpenGLShader::Vertex, R"(
			#version 330 core
			layout(location = 0) in vec3 position;
			uniform mat4 u_mvp;
			uniform float u_point_size;
			void main() { gl_Position = u_mvp * vec4(position, 1.0); gl_PointSize = u_point_size; }
		)");
		program_.addShaderFromSourceCode(QOpenGLShader::Fragment, R"(
			#version 330 core
			out vec4 fragColor;
			uniform vec3 u_color;
			void main() { fragColor = vec4(u_color, 1.0); }
		)");
		program_.link();

		u_mvp_loc_ = program_.uniformLocation("u_mvp");
		u_color_loc_ = program_.uniformLocation("u_color");
		u_point_size_loc_ = program_.uniformLocation("u_point_size");

		vao_.create();
		points_vbo_.create();
		axis_vbo_.create();
		updateGpuBuffers();
	}

	void resizeGL(int w, int h) override { glViewport(0, 0, w, h); }

	void paintGL() override
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		if(not program_.isLinked())
			return;

		const float aspect = static_cast<float>(std::max(1, width())) / static_cast<float>(std::max(1, height()));
		QMatrix4x4 proj;
		proj.perspective(45.0f, aspect, 0.01f, std::max(10000.0f, cam_dist_ * 20.0f + cloud_radius_ * 20.0f));
		QMatrix4x4 view;
		view.translate(pan_x_, pan_y_, -cam_dist_);
		view.rotate(pitch_deg_, 1.0f, 0.0f, 0.0f);
		view.rotate(yaw_deg_, 0.0f, 1.0f, 0.0f);
		view.translate(-center_x_, -center_y_, -center_z_);
		const QMatrix4x4 mvp = proj * view;

		program_.bind();
		program_.setUniformValue(u_mvp_loc_, mvp);
		vao_.bind();

		if(not axis_vertices_.empty())
		{
			axis_vbo_.bind();
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
			glEnableVertexAttribArray(0);
			program_.setUniformValue(u_color_loc_, QVector3D(0.35f, 0.45f, 0.90f));
			program_.setUniformValue(u_point_size_loc_, 1.0f);
			glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(axis_vertices_.size()));
			axis_vbo_.release();
		}

		if(not point_vertices_.empty())
		{
			points_vbo_.bind();
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
			glEnableVertexAttribArray(0);
			program_.setUniformValue(u_point_size_loc_, 3.0f);
			// group A (e.g. helios) — amber; group B (e.g. bpearl) — cyan.
			const GLsizei a = static_cast<GLsizei>(group_a_count_);
			const GLsizei total = static_cast<GLsizei>(point_vertices_.size());
			program_.setUniformValue(u_color_loc_, QVector3D(0.95f, 0.74f, 0.18f));
			glDrawArrays(GL_POINTS, 0, a);
			if(total > a)
			{
				program_.setUniformValue(u_color_loc_, QVector3D(0.35f, 0.80f, 0.95f));
				glDrawArrays(GL_POINTS, a, total - a);
			}
			points_vbo_.release();
		}

		glDisableVertexAttribArray(0);
		vao_.release();
		program_.release();

		QPainter painter(this);
		painter.setRenderHint(QPainter::TextAntialiasing, true);
		painter.setPen(QColor(255, 255, 255));
		painter.drawText(QRect(10, 10, width() - 20, 24), Qt::AlignLeft | Qt::AlignTop,
		                 QString("Points: %1    %2 Hz    [drag=rotate  right/mid=pan  wheel=zoom  R=reset]")
		                     .arg(static_cast<qulonglong>(point_vertices_.size()))
		                     .arg(fps_, 0, 'f', 1));
	}

	void mousePressEvent(QMouseEvent *event) override
	{
		last_mouse_pos_ = event->pos();
		if(event->button() == Qt::LeftButton) { rotating_ = true; user_interacted_ = true; }
		else if(event->button() == Qt::RightButton or event->button() == Qt::MiddleButton) { panning_ = true; user_interacted_ = true; }
	}

	void mouseMoveEvent(QMouseEvent *event) override
	{
		const QPoint delta = event->pos() - last_mouse_pos_;
		last_mouse_pos_ = event->pos();
		if(rotating_)
		{
			yaw_deg_ += static_cast<float>(delta.x()) * 0.4f;
			pitch_deg_ = std::clamp(pitch_deg_ + static_cast<float>(delta.y()) * 0.4f, -89.0f, 89.0f);
			update();
		}
		else if(panning_)
		{
			const float scale = std::max(0.0005f, cam_dist_ * 0.0010f);
			pan_x_ += static_cast<float>(delta.x()) * scale;
			pan_y_ -= static_cast<float>(delta.y()) * scale;
			update();
		}
	}

	void mouseReleaseEvent(QMouseEvent *) override { rotating_ = false; panning_ = false; }

	void wheelEvent(QWheelEvent *event) override
	{
		const float steps = static_cast<float>(event->angleDelta().y()) / 120.0f;
		if(std::abs(steps) < 1e-4f)
			return;
		const float old = cam_dist_;
		cam_dist_ = std::clamp(cam_dist_ * std::pow(0.85f, steps), 0.005f, 1000000.0f);
		const QPointF p = event->position();
		const float nx = static_cast<float>((p.x() / std::max(1, width())) - 0.5);
		const float ny = static_cast<float>(0.5 - (p.y() / std::max(1, height())));
		pan_x_ += nx * (old - cam_dist_) * 0.9f;
		pan_y_ += ny * (old - cam_dist_) * 0.9f;
		user_interacted_ = true;
		update();
	}

	void keyPressEvent(QKeyEvent *event) override
	{
		if(event->key() == Qt::Key_R) { resetView(); update(); return; }
		QOpenGLWidget::keyPressEvent(event);
	}

private:
	void updateGpuBuffers()
	{
		if(not context() or not points_vbo_.isCreated() or not axis_vbo_.isCreated())
			return;
		makeCurrent();
		points_vbo_.bind();
		points_vbo_.allocate(point_vertices_.data(), static_cast<int>(point_vertices_.size() * sizeof(QVector3D)));
		points_vbo_.release();
		axis_vbo_.bind();
		axis_vbo_.allocate(axis_vertices_.data(), static_cast<int>(axis_vertices_.size() * sizeof(QVector3D)));
		axis_vbo_.release();
		doneCurrent();
	}

	std::vector<QVector3D> makeAxes(float scale) const
	{
		const float s = std::max(0.5f, scale);
		return {
			{center_x_, center_y_, center_z_}, {center_x_ + s, center_y_, center_z_},
			{center_x_, center_y_, center_z_}, {center_x_, center_y_ + s, center_z_},
			{center_x_, center_y_, center_z_}, {center_x_, center_y_, center_z_ + s},
		};
	}

	void resetView()
	{
		yaw_deg_ = 25.0f; pitch_deg_ = -20.0f; pan_x_ = 0.0f; pan_y_ = 0.0f;
		cam_dist_ = std::max(2.0f, cloud_radius_ * 3.0f);
		user_interacted_ = false;
	}

	std::vector<QVector3D> point_vertices_, axis_vertices_;
	std::size_t group_a_count_ = 0;

	float center_x_ = 0.0f, center_y_ = 0.0f, center_z_ = 0.0f, cloud_radius_ = 1.0f;
	float cam_dist_ = 5.0f, yaw_deg_ = 25.0f, pitch_deg_ = -20.0f, pan_x_ = 0.0f, pan_y_ = 0.0f;
	bool rotating_ = false, panning_ = false, user_interacted_ = false;
	QPoint last_mouse_pos_;

	float fps_ = 0.0f;
	std::chrono::steady_clock::time_point last_{};
	bool have_last_ = false;

	QOpenGLShaderProgram program_;
	QOpenGLVertexArrayObject vao_;
	QOpenGLBuffer points_vbo_{QOpenGLBuffer::VertexBuffer};
	QOpenGLBuffer axis_vbo_{QOpenGLBuffer::VertexBuffer};
	int u_mvp_loc_ = -1, u_color_loc_ = -1, u_point_size_loc_ = -1;
};

}   // namespace rc::viewers

#endif   // RC_COMMON_GL_POINT_CLOUD_VIEWER_H
