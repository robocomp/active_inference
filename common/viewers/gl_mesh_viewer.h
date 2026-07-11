/*
 *  Reusable 3D triangle-mesh viewer (active_inference/common).
 *
 *  Renders a triangle soup (flat list, 3 vertices per triangle — e.g. from rc::obj::load_obj_mesh_data)
 *  with flat per-face normals + two-sided Lambert shading and an orbit/pan/zoom camera. Pure renderer:
 *  geometry comes in through set_mesh(); it knows nothing about OBJ files, the graph or the media
 *  plane, so any agent can drive it. Non-Q_OBJECT (owner-driven) → no MOC.
 *
 *  Two-sided lighting (abs(N·L)) is used on purpose: OBJ face winding is often inconsistent, so we
 *  don't cull back-faces and never render a face fully black.
 */
#ifndef RC_COMMON_GL_MESH_VIEWER_H
#define RC_COMMON_GL_MESH_VIEWER_H

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
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace rc::viewers
{

class GLMeshViewer : public QOpenGLWidget, protected QOpenGLFunctions
{
public:
	explicit GLMeshViewer(QWidget *parent = nullptr) : QOpenGLWidget(parent)
	{
		resize(720, 720);
		setWindowTitle("mesh");
		setFocusPolicy(Qt::StrongFocus);
	}

	~GLMeshViewer() override
	{
		if(context())
		{
			makeCurrent();
			if(vbo_.isCreated()) vbo_.destroy();
			if(vao_.isCreated()) vao_.destroy();
			doneCurrent();
		}
	}

	// Replace the mesh. `triangles` is a flat soup: triangles[3k..3k+2] are one triangle. Computes a
	// flat face normal per triangle, auto-frames the camera (until the user moves it).
	void set_mesh(std::span<const QVector3D> triangles)
	{
		verts_.clear();
		const std::size_t ntri = triangles.size() / 3;
		verts_.reserve(ntri * 3);

		float minx = std::numeric_limits<float>::max(), miny = minx, minz = minx;
		float maxx = std::numeric_limits<float>::lowest(), maxy = maxx, maxz = maxx;
		for(std::size_t t = 0; t < ntri; ++t)
		{
			const QVector3D &a = triangles[3 * t + 0];
			const QVector3D &b = triangles[3 * t + 1];
			const QVector3D &c = triangles[3 * t + 2];
			QVector3D n = QVector3D::crossProduct(b - a, c - a);
			if(n.lengthSquared() > 0.0f)
				n.normalize();
			for(const QVector3D &v : {a, b, c})
			{
				verts_.push_back({v.x(), v.y(), v.z(), n.x(), n.y(), n.z()});
				minx = std::min(minx, v.x()); maxx = std::max(maxx, v.x());
				miny = std::min(miny, v.y()); maxy = std::max(maxy, v.y());
				minz = std::min(minz, v.z()); maxz = std::max(maxz, v.z());
			}
		}
		tri_count_ = static_cast<int>(ntri);

		if(not verts_.empty())
		{
			center_ = {0.5f * (minx + maxx), 0.5f * (miny + maxy), 0.5f * (minz + maxz)};
			radius_ = std::max({0.001f, 0.5f * (maxx - minx), 0.5f * (maxy - miny), 0.5f * (maxz - minz)});
			if(not user_interacted_)
				cam_dist_ = std::max(0.5f, radius_ * 3.0f);
		}
		uploadBuffers();
		update();
	}

protected:
	void initializeGL() override
	{
		initializeOpenGLFunctions();
		glEnable(GL_DEPTH_TEST);
		glClearColor(0.10f, 0.11f, 0.13f, 1.0f);

		program_.addShaderFromSourceCode(QOpenGLShader::Vertex, R"(
			#version 330 core
			layout(location = 0) in vec3 position;
			layout(location = 1) in vec3 normal;
			uniform mat4 u_mvp;
			out vec3 v_normal;
			void main() { gl_Position = u_mvp * vec4(position, 1.0); v_normal = normal; }
		)");
		program_.addShaderFromSourceCode(QOpenGLShader::Fragment, R"(
			#version 330 core
			in vec3 v_normal;
			out vec4 fragColor;
			uniform vec3 u_light;   // world-space light direction (normalised)
			uniform vec3 u_base;    // base albedo
			void main()
			{
				vec3 N = normalize(v_normal);
				float d = abs(dot(N, u_light));            // two-sided
				float shade = 0.28 + 0.72 * d;             // ambient + diffuse
				fragColor = vec4(u_base * shade, 1.0);
			}
		)");
		program_.link();
		u_mvp_loc_   = program_.uniformLocation("u_mvp");
		u_light_loc_ = program_.uniformLocation("u_light");
		u_base_loc_  = program_.uniformLocation("u_base");

		vao_.create();
		vbo_.create();
		uploadBuffers();
	}

	void resizeGL(int w, int h) override { glViewport(0, 0, w, h); }

	void paintGL() override
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		if(not program_.isLinked())
			return;

		const float aspect = static_cast<float>(std::max(1, width())) / static_cast<float>(std::max(1, height()));
		QMatrix4x4 proj;
		proj.perspective(45.0f, aspect, std::max(0.001f, cam_dist_ * 0.001f), cam_dist_ * 10.0f + radius_ * 10.0f);
		QMatrix4x4 view;
		view.translate(pan_x_, pan_y_, -cam_dist_);
		view.rotate(pitch_deg_, 1.0f, 0.0f, 0.0f);
		view.rotate(yaw_deg_, 0.0f, 1.0f, 0.0f);
		view.translate(-center_);
		const QMatrix4x4 mvp = proj * view;

		program_.bind();
		program_.setUniformValue(u_mvp_loc_, mvp);
		program_.setUniformValue(u_light_loc_, QVector3D(0.40f, 0.75f, 0.53f).normalized());
		program_.setUniformValue(u_base_loc_, QVector3D(0.72f, 0.76f, 0.82f));
		vao_.bind();
		if(tri_count_ > 0)
		{
			vbo_.bind();
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), nullptr);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
			                      reinterpret_cast<const void *>(offsetof(Vertex, nx)));
			glEnableVertexAttribArray(0);
			glEnableVertexAttribArray(1);
			glDrawArrays(GL_TRIANGLES, 0, tri_count_ * 3);
			glDisableVertexAttribArray(0);
			glDisableVertexAttribArray(1);
			vbo_.release();
		}
		vao_.release();
		program_.release();

		QPainter painter(this);
		painter.setRenderHint(QPainter::TextAntialiasing, true);
		painter.setPen(QColor(235, 235, 235));
		painter.drawText(QRect(10, 8, width() - 20, 22), Qt::AlignLeft | Qt::AlignTop,
		                 (tri_count_ > 0)
		                     ? QString("triangles: %1    [drag=rotate  right/mid=pan  wheel=zoom  R=reset]").arg(tri_count_)
		                     : QStringLiteral("no mesh loaded"));
	}

	void mousePressEvent(QMouseEvent *e) override
	{
		last_ = e->pos();
		if(e->button() == Qt::LeftButton) { rotating_ = true; user_interacted_ = true; }
		else if(e->button() == Qt::RightButton or e->button() == Qt::MiddleButton) { panning_ = true; user_interacted_ = true; }
	}
	void mouseMoveEvent(QMouseEvent *e) override
	{
		const QPoint d = e->pos() - last_;
		last_ = e->pos();
		if(rotating_)
		{
			yaw_deg_ += d.x() * 0.4f;
			pitch_deg_ = std::clamp(pitch_deg_ + d.y() * 0.4f, -89.0f, 89.0f);
			update();
		}
		else if(panning_)
		{
			const float s = std::max(0.0005f, cam_dist_ * 0.0015f);
			pan_x_ += d.x() * s; pan_y_ -= d.y() * s;
			update();
		}
	}
	void mouseReleaseEvent(QMouseEvent *) override { rotating_ = false; panning_ = false; }
	void wheelEvent(QWheelEvent *e) override
	{
		const float steps = e->angleDelta().y() / 120.0f;
		if(std::abs(steps) < 1e-4f) return;
		cam_dist_ = std::clamp(cam_dist_ * std::pow(0.85f, steps), 0.01f, 100000.0f);
		user_interacted_ = true;
		update();
	}
	void keyPressEvent(QKeyEvent *e) override
	{
		if(e->key() == Qt::Key_R)
		{
			yaw_deg_ = 25.0f; pitch_deg_ = -20.0f; pan_x_ = 0.0f; pan_y_ = 0.0f;
			cam_dist_ = std::max(0.5f, radius_ * 3.0f); user_interacted_ = false;
			update();
			return;
		}
		QOpenGLWidget::keyPressEvent(e);
	}

private:
	struct Vertex { float x, y, z, nx, ny, nz; };

	void uploadBuffers()
	{
		if(not context() or not vbo_.isCreated())
			return;
		makeCurrent();
		vbo_.bind();
		vbo_.allocate(verts_.data(), static_cast<int>(verts_.size() * sizeof(Vertex)));
		vbo_.release();
		doneCurrent();
	}

	std::vector<Vertex> verts_;
	int tri_count_ = 0;
	QVector3D center_{0, 0, 0};
	float radius_ = 1.0f;

	float cam_dist_ = 3.0f, yaw_deg_ = 25.0f, pitch_deg_ = -20.0f, pan_x_ = 0.0f, pan_y_ = 0.0f;
	bool rotating_ = false, panning_ = false, user_interacted_ = false;
	QPoint last_;

	QOpenGLShaderProgram program_;
	QOpenGLVertexArrayObject vao_;
	QOpenGLBuffer vbo_{QOpenGLBuffer::VertexBuffer};
	int u_mvp_loc_ = -1, u_light_loc_ = -1, u_base_loc_ = -1;
};

}   // namespace rc::viewers

#endif   // RC_COMMON_GL_MESH_VIEWER_H
