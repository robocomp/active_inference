/*
 *  Reusable 3D residual-grid / belief-field viewer (active_inference/common).
 *
 *  Renders residual_concept's `grid` node the same way the voxelizer's in-process viewer does: the
 *  dense Beta occupancy belief field is drawn as ELEVATED COLUMNS — one vertical bar per cell whose
 *  HEIGHT and HUE encode the mean occupancy P (collision risk, green→red as 0.5→1) and whose
 *  BRIGHTNESS encodes confidence 1−Var/Var_prior (the epistemic term: vivid = well-observed, faded =
 *  uncertain). On top of the field it draws the two discrete cell layers: raw OCCUPIED cells (amber)
 *  and the inflated half-robot-width clearance BORDER (steel blue), as floor points.
 *
 *  Data is pushed in via set_data() in the ROOM frame (x, y horizontal, z = a small display height);
 *  the viewer maps room (x, y, z) → OpenGL Y-up as (x, z, y). Self-contained (no OpenCV, no cortex),
 *  same camera/axes machinery as gl_skeleton_viewer.h. Non-Q_OBJECT (owner drives set_data()).
 */
#ifndef RC_COMMON_GL_GRID_FIELD_VIEWER_H
#define RC_COMMON_GL_GRID_FIELD_VIEWER_H

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
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace rc::viewers
{

class GLGridFieldViewer : public QOpenGLWidget, protected QOpenGLFunctions
{
	struct CVertex { float x, y, z, r, g, b; };   // interleaved position + colour

public:
	explicit GLGridFieldViewer(QWidget *parent = nullptr) : QOpenGLWidget(parent)
	{
		resize(720, 640);
		setWindowTitle("residual grid — belief field");
		setFocusPolicy(Qt::StrongFocus);
		rebuildAxes(1.0f);
	}

	~GLGridFieldViewer() override
	{
		if(context())
		{
			makeCurrent();
			if(scene_vbo_.isCreated()) scene_vbo_.destroy();
			if(vao_.isCreated())       vao_.destroy();
			doneCurrent();
		}
	}

	// Replace the displayed grid. All spans are ROOM frame.
	//   field_centres[i] = (x, y, z) cell centre; prob[i] = P in [0,1]; var[i] = Var[P]  (1:1, P>0.5)
	//   occupied / border = flat cell-centre points (x, y, z) drawn on the floor.
	void set_data(std::span<const QVector3D> field_centres, std::span<const float> prob,
	              std::span<const float> var, std::span<const QVector3D> occupied,
	              std::span<const QVector3D> border)
	{
		bar_vertices_.clear();
		cap_vertices_.clear();
		cell_vertices_.clear();
		field_cells_ = 0;

		float minx = std::numeric_limits<float>::max(), miny = minx, minz = minx;
		float maxx = std::numeric_limits<float>::lowest(), maxy = maxx, maxz = maxx;
		bool any = false;
		const auto grow = [&](const QVector3D &q)
		{
			minx = std::min(minx, q.x()); maxx = std::max(maxx, q.x());
			miny = std::min(miny, q.y()); maxy = std::max(maxy, q.y());
			minz = std::min(minz, q.z()); maxz = std::max(maxz, q.z());
			any = true;
		};
		// ROOM (x, y, z_height) → OpenGL Y-up (x, height, y).
		const auto to_ogl = [](float x, float y, float up) { return QVector3D(x, up, y); };

		// ── Beta belief field → elevated columns (voxelizer update_grid_field, verbatim constants) ──
		constexpr float var_prior  = 0.125f;   // Beta(0.5,0.5) prior variance = "unknown"
		constexpr float max_height = 0.60f;    // room metres for a fully-occupied (P=1) cell
		constexpr float base_z     = 0.02f;    // lift the foot just off the floor plane
		const std::size_t nf = std::min({field_centres.size(), prob.size(), var.size()});
		for(std::size_t i = 0; i < nf; ++i)
		{
			const float p = prob[i];
			if(p <= 0.5f) continue;                                   // collapsed / free-leaning → skip
			const float t    = std::clamp((p - 0.5f) * 2.0f, 0.f, 1.f);
			const float conf = std::clamp(1.0f - var[i] / var_prior, 0.f, 1.f);
			const float r = (0.40f + 0.60f * t) * conf;
			const float g = (0.90f - 0.70f * t) * conf;
			const float b = 0.10f * conf;
			const float x = field_centres[i].x(), y = field_centres[i].y();
			const float top = base_z + t * max_height;
			const QVector3D foot = to_ogl(x, y, base_z), tip = to_ogl(x, y, top);
			bar_vertices_.push_back({foot.x(), foot.y(), foot.z(), 0.20f * r, 0.20f * g, 0.20f * b});   // dim foot
			bar_vertices_.push_back({tip.x(),  tip.y(),  tip.z(),  r,         g,         b});           // bright top
			cap_vertices_.push_back({tip.x(),  tip.y(),  tip.z(),  r,         g,         b});
			grow(tip); grow(foot);
			++field_cells_;
		}

		// ── Discrete cell layers → floor points ──
		const auto add_cells = [&](std::span<const QVector3D> pts, float cr, float cg, float cb)
		{
			for(const QVector3D &p : pts)
			{
				const QVector3D q = to_ogl(p.x(), p.y(), p.z());
				cell_vertices_.push_back({q.x(), q.y(), q.z(), cr, cg, cb});
				grow(q);
			}
		};
		occupied_count_ = occupied.size();
		add_cells(occupied, 1.00f, 0.70f, 0.10f);   // colour A — amber
		add_cells(border,   0.30f, 0.60f, 0.95f);   // colour B — steel blue

		// Rolling display-rate estimate from set_data() call intervals.
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
				cam_dist_ = std::max(2.0f, scene_radius_ * 2.5f);
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
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(CVertex), reinterpret_cast<const void *>(0));
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(CVertex), reinterpret_cast<const void *>(3 * sizeof(float)));
			glEnableVertexAttribArray(0);
			glEnableVertexAttribArray(1);
			glDrawArrays(mode, static_cast<GLint>(first), static_cast<GLsizei>(count));
		};

		// Layout in the VBO: [axes | bars | caps | cells].
		std::size_t off = 0;
		glLineWidth(1.0f);
		draw(GL_LINES,  off, axis_vertices_.size(), 1.0f);           off += axis_vertices_.size();
		glLineWidth(2.0f);   // best-effort; core-profile drivers clamp to 1px
		draw(GL_LINES,  off, bar_vertices_.size(),  1.0f);           off += bar_vertices_.size();
		draw(GL_POINTS, off, cap_vertices_.size(),  6.0f);           off += cap_vertices_.size();
		draw(GL_POINTS, off, cell_vertices_.size(), 7.0f);

		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		scene_vbo_.release();
		vao_.release();
		program_.release();

		QPainter painter(this);
		painter.setRenderHint(QPainter::TextAntialiasing, true);
		painter.setPen(QColor(235, 235, 235));
		painter.drawText(QRect(10, 10, width() - 20, 40), Qt::AlignLeft | Qt::AlignTop,
		                 QString("Field cells: %1   occupied: %2   %3 Hz\n"
		                         "hue=risk P (green→red)  brightness=confidence   [drag=rotate  right/mid=pan  wheel=zoom  R=reset]")
		                     .arg(field_cells_)
		                     .arg(static_cast<qulonglong>(occupied_count_))
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
		all.reserve(axis_vertices_.size() + bar_vertices_.size() + cap_vertices_.size() + cell_vertices_.size());
		all.insert(all.end(), axis_vertices_.begin(), axis_vertices_.end());
		all.insert(all.end(), bar_vertices_.begin(),  bar_vertices_.end());
		all.insert(all.end(), cap_vertices_.begin(),  cap_vertices_.end());
		all.insert(all.end(), cell_vertices_.begin(), cell_vertices_.end());
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
			{cx, cy, cz, 0.2f, 0.9f, 0.2f}, {cx, cy + s, cz, 0.2f, 0.9f, 0.2f},   // Y green (up)
			{cx, cy, cz, 0.3f, 0.5f, 1.0f}, {cx, cy, cz + s, 0.3f, 0.5f, 1.0f},   // Z blue
		};
	}

	void resetView()
	{
		yaw_deg_ = 30.0f; pitch_deg_ = -35.0f; pan_x_ = 0.0f; pan_y_ = 0.0f;
		cam_dist_ = std::max(2.0f, scene_radius_ * 2.5f);
		user_interacted_ = false;
	}

	std::vector<CVertex> axis_vertices_, bar_vertices_, cap_vertices_, cell_vertices_;
	int         field_cells_    = 0;
	std::size_t occupied_count_ = 0;

	float center_x_ = 0.0f, center_y_ = 0.0f, center_z_ = 0.0f, scene_radius_ = 1.0f;
	float cam_dist_ = 4.0f, yaw_deg_ = 30.0f, pitch_deg_ = -35.0f, pan_x_ = 0.0f, pan_y_ = 0.0f;
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

#endif   // RC_COMMON_GL_GRID_FIELD_VIEWER_H
