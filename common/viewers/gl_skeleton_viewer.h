/*
 *  Reusable 3D BODY_18 skeleton viewer (active_inference/common).
 *
 *  Renders one or more human skeletons (18 keypoints each, flat [x0,y0,z0, x1,y1,z1, …] in metres)
 *  as coloured joint points + cyan bones in an orbit/pan/zoom OpenGL scene. It is the 3D companion
 *  to gl_point_cloud_viewer.h: same camera machinery and XYZ axes, but a per-vertex-colour shader so
 *  joints are tinted by body region and bones drawn distinctly. Skeletons are pushed in via
 *  set_skeletons(); the widget is source-agnostic (robot_concept feeds it from the DSR 'skeleton'
 *  node's attributes). Bone topology + joint palette match the retina's in-process skeleton draw.
 *
 *  Frame: the caller passes keypoints in the ZED camera frame (x-right, y-depth, z-up); this viewer
 *  maps them to OpenGL Y-up as (x, z, y) for display. NaN keypoints (missing depth) are skipped, and
 *  a bone is drawn only when BOTH its endpoints are finite.
 *
 *  Non-Q_OBJECT: no signals/slots of its own (owner drives set_skeletons()), so no MOC pass needed.
 */
#ifndef RC_COMMON_GL_SKELETON_VIEWER_H
#define RC_COMMON_GL_SKELETON_VIEWER_H

#include <QKeyEvent>
#include <QMatrix4x4>
#include <QMouseEvent>
#include <QOpenGLBuffer>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWidget>
#include <QPainter>
#include <QVector3D>
#include <QWheelEvent>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace rc::viewers
{

class GLSkeletonViewer : public QOpenGLWidget, protected QOpenGLFunctions
{
	struct CVertex { float x, y, z, r, g, b; };   // interleaved position + colour

public:
	static constexpr int K = 18;   // BODY_18 keypoints per skeleton

	explicit GLSkeletonViewer(QWidget *parent = nullptr) : QOpenGLWidget(parent)
	{
		resize(640, 640);
		setWindowTitle("skeletons");
		setFocusPolicy(Qt::StrongFocus);
		rebuildAxes(1.0f);
	}

	~GLSkeletonViewer() override
	{
		if(context())
		{
			makeCurrent();
			if(scene_vbo_.isCreated()) scene_vbo_.destroy();
			if(vao_.isCreated())       vao_.destroy();
			doneCurrent();
		}
	}

	// Replace the displayed skeletons. Each span element is a flat 18*3 keypoint buffer (metres,
	// ZED camera frame); shorter buffers are skipped. Auto-frames on the first non-empty set until
	// the user drives the camera.
	void set_skeletons(std::span<const std::vector<float>> skeletons)
	{
		bone_vertices_.clear();
		joint_vertices_.clear();
		int visible = 0;

		float minx = std::numeric_limits<float>::max(), miny = minx, minz = minx;
		float maxx = std::numeric_limits<float>::lowest(), maxy = maxx, maxz = maxx;
		bool any = false;

		const auto finite = [](const float *p)
		{ return std::isfinite(p[0]) and std::isfinite(p[1]) and std::isfinite(p[2]); };
		// ZED (x-right, y-depth, z-up) → OpenGL Y-up.
		const auto to_ogl = [](const float *p) { return QVector3D(p[0], p[2], p[1]); };

		for(const auto &s : skeletons)
		{
			if(static_cast<int>(s.size()) < K * 3)
				continue;
			bool shown = false;
			for(const auto &e : EDGES)
			{
				const float *a = &s[e[0] * 3], *b = &s[e[1] * 3];
				if(not (finite(a) and finite(b)))
					continue;
				const QVector3D pa = to_ogl(a), pb = to_ogl(b);
				bone_vertices_.push_back({pa.x(), pa.y(), pa.z(), BONE_RGB[0], BONE_RGB[1], BONE_RGB[2]});
				bone_vertices_.push_back({pb.x(), pb.y(), pb.z(), BONE_RGB[0], BONE_RGB[1], BONE_RGB[2]});
				shown = true;
			}
			for(int j = 0; j < K; ++j)
			{
				const float *p = &s[j * 3];
				if(not finite(p))
					continue;
				const QVector3D q = to_ogl(p);
				joint_vertices_.push_back({q.x(), q.y(), q.z(), JOINT_RGB[j][0], JOINT_RGB[j][1], JOINT_RGB[j][2]});
				minx = std::min(minx, q.x()); maxx = std::max(maxx, q.x());
				miny = std::min(miny, q.y()); maxy = std::max(maxy, q.y());
				minz = std::min(minz, q.z()); maxz = std::max(maxz, q.z());
				any = true;
				shown = true;
			}
			if(shown) ++visible;
		}
		skeleton_count_ = visible;

		// Rolling display-rate estimate from set_skeletons() call intervals.
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

		if(any)
		{
			center_x_ = (minx + maxx) * 0.5f;
			center_y_ = (miny + maxy) * 0.5f;
			center_z_ = (minz + maxz) * 0.5f;
			const float rx = std::max(0.001f, (maxx - minx) * 0.5f);
			const float ry = std::max(0.001f, (maxy - miny) * 0.5f);
			const float rz = std::max(0.001f, (maxz - minz) * 0.5f);
			scene_radius_ = std::max({rx, ry, rz});
			if(not user_interacted_)
				cam_dist_ = std::max(1.5f, scene_radius_ * 3.0f);
		}
		rebuildAxes(any ? scene_radius_ : 1.0f);
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
			layout(location = 1) in vec3 color;
			uniform mat4 u_mvp;
			uniform float u_point_size;
			out vec3 v_color;
			void main() { gl_Position = u_mvp * vec4(position, 1.0); gl_PointSize = u_point_size; v_color = color; }
		)");
		program_.addShaderFromSourceCode(QOpenGLShader::Fragment, R"(
			#version 330 core
			in vec3 v_color;
			out vec4 fragColor;
			void main() { fragColor = vec4(v_color, 1.0); }
		)");
		program_.link();
		u_mvp_loc_ = program_.uniformLocation("u_mvp");
		u_point_size_loc_ = program_.uniformLocation("u_point_size");

		vao_.create();
		scene_vbo_.create();
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
		proj.perspective(45.0f, aspect, 0.01f, std::max(10000.0f, cam_dist_ * 20.0f + scene_radius_ * 20.0f));
		QMatrix4x4 view;
		view.translate(pan_x_, pan_y_, -cam_dist_);
		view.rotate(pitch_deg_, 1.0f, 0.0f, 0.0f);
		view.rotate(yaw_deg_, 0.0f, 1.0f, 0.0f);
		view.translate(-center_x_, -center_y_, -center_z_);
		const QMatrix4x4 mvp = proj * view;

		program_.bind();
		program_.setUniformValue(u_mvp_loc_, mvp);
		vao_.bind();
		scene_vbo_.bind();

		const auto draw = [&](GLenum mode, std::size_t first, std::size_t count, float point_size)
		{
			if(count == 0) return;
			program_.setUniformValue(u_point_size_loc_, point_size);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(CVertex),
			                      reinterpret_cast<const void *>(0));
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(CVertex),
			                      reinterpret_cast<const void *>(3 * sizeof(float)));
			glEnableVertexAttribArray(0);
			glEnableVertexAttribArray(1);
			glDrawArrays(mode, static_cast<GLint>(first), static_cast<GLsizei>(count));
		};

		// Layout in the VBO: [axes | bones | joints].
		glLineWidth(2.0f);   // best-effort; core-profile drivers clamp to 1px
		draw(GL_LINES, 0, axis_vertices_.size(), 1.0f);
		draw(GL_LINES, axis_vertices_.size(), bone_vertices_.size(), 1.0f);
		draw(GL_POINTS, axis_vertices_.size() + bone_vertices_.size(), joint_vertices_.size(), 9.0f);

		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		scene_vbo_.release();
		vao_.release();
		program_.release();

		QPainter painter(this);
		painter.setRenderHint(QPainter::TextAntialiasing, true);
		painter.setPen(QColor(255, 255, 255));
		painter.drawText(QRect(10, 10, width() - 20, 24), Qt::AlignLeft | Qt::AlignTop,
		                 QString("Skeletons: %1    %2 Hz    [drag=rotate  right/mid=pan  wheel=zoom  R=reset]")
		                     .arg(skeleton_count_)
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
		cam_dist_ = std::clamp(cam_dist_ * std::pow(0.85f, steps), 0.005f, 1000000.0f);
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
		if(not context() or not scene_vbo_.isCreated())
			return;
		makeCurrent();
		std::vector<CVertex> all;
		all.reserve(axis_vertices_.size() + bone_vertices_.size() + joint_vertices_.size());
		all.insert(all.end(), axis_vertices_.begin(), axis_vertices_.end());
		all.insert(all.end(), bone_vertices_.begin(), bone_vertices_.end());
		all.insert(all.end(), joint_vertices_.begin(), joint_vertices_.end());
		scene_vbo_.bind();
		scene_vbo_.allocate(all.data(), static_cast<int>(all.size() * sizeof(CVertex)));
		scene_vbo_.release();
		doneCurrent();
	}

	void rebuildAxes(float scale)
	{
		const float s = std::max(0.3f, scale);
		const float cx = center_x_, cy = center_y_, cz = center_z_;
		axis_vertices_ = {
			{cx, cy, cz, 0.9f, 0.2f, 0.2f}, {cx + s, cy, cz, 0.9f, 0.2f, 0.2f},   // X red
			{cx, cy, cz, 0.2f, 0.9f, 0.2f}, {cx, cy + s, cz, 0.2f, 0.9f, 0.2f},   // Y green
			{cx, cy, cz, 0.3f, 0.5f, 1.0f}, {cx, cy, cz + s, 0.3f, 0.5f, 1.0f},   // Z blue
		};
	}

	void resetView()
	{
		yaw_deg_ = 25.0f; pitch_deg_ = -20.0f; pan_x_ = 0.0f; pan_y_ = 0.0f;
		cam_dist_ = std::max(1.5f, scene_radius_ * 3.0f);
		user_interacted_ = false;
	}

	// BODY_18 bone topology (ZED/OpenPose) — matches the retina's in-process skeleton draw.
	static constexpr int EDGES[17][2] = {
		{1,0}, {0,14}, {14,16}, {0,15}, {15,17},
		{1,2}, {2,3}, {3,4}, {1,5}, {5,6}, {6,7},
		{2,8}, {8,9}, {9,10}, {5,11}, {11,12}, {12,13}
	};
	static constexpr float BONE_RGB[3] = {0.10f, 0.95f, 0.95f};   // cyan
	// Per-joint colour by body region (head=yellow, neck=white, shoulders=green, elbows=orange,
	// wrists=red, hips=magenta, knees=blue, ankles=purple) — matches the retina viewer.
	static constexpr float JOINT_RGB[K][3] = {
		{1.00f, 0.85f, 0.10f}, {1.00f, 1.00f, 1.00f}, {0.20f, 0.90f, 0.20f}, {1.00f, 0.55f, 0.00f},
		{1.00f, 0.15f, 0.15f}, {0.20f, 0.90f, 0.20f}, {1.00f, 0.55f, 0.00f}, {1.00f, 0.15f, 0.15f},
		{0.90f, 0.20f, 0.90f}, {0.30f, 0.50f, 1.00f}, {0.60f, 0.20f, 0.90f}, {0.90f, 0.20f, 0.90f},
		{0.30f, 0.50f, 1.00f}, {0.60f, 0.20f, 0.90f}, {1.00f, 0.85f, 0.10f}, {1.00f, 0.85f, 0.10f},
		{1.00f, 0.85f, 0.10f}, {1.00f, 0.85f, 0.10f},
	};

	std::vector<CVertex> axis_vertices_, bone_vertices_, joint_vertices_;
	int skeleton_count_ = 0;

	float center_x_ = 0.0f, center_y_ = 0.0f, center_z_ = 0.0f, scene_radius_ = 1.0f;
	float cam_dist_ = 3.0f, yaw_deg_ = 25.0f, pitch_deg_ = -20.0f, pan_x_ = 0.0f, pan_y_ = 0.0f;
	bool rotating_ = false, panning_ = false, user_interacted_ = false;
	QPoint last_mouse_pos_;

	float fps_ = 0.0f;
	std::chrono::steady_clock::time_point last_{};
	bool have_last_ = false;

	QOpenGLShaderProgram program_;
	QOpenGLVertexArrayObject vao_;
	QOpenGLBuffer scene_vbo_{QOpenGLBuffer::VertexBuffer};
	int u_mvp_loc_ = -1, u_point_size_loc_ = -1;
};

}   // namespace rc::viewers

#endif   // RC_COMMON_GL_SKELETON_VIEWER_H
