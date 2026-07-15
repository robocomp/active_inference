/*
 *  Reusable 3D residual-grid / belief-field viewer (active_inference/common).
 *
 *  Renders residual_concept's `grid` node as a 3D SURPRISE LANDSCAPE. The dense Beta occupancy belief
 *  field (grid_occupancy_prob + grid_occupancy_var, a regular w×h grid) is drawn as a continuous LIT
 *  HEIGHTFIELD MESH: the elevation of each vertex encodes the occupancy risk / surprise
 *  t = clamp((P−0.5)·2, 0, 1), its HUE encodes the same risk (green→red), and its BRIGHTNESS the
 *  confidence 1−Var/Var_prior (epistemic term). Per-vertex normals + a fixed world light make the
 *  field read as terrain. A key toggle (M) switches to the older ELEVATED-COLUMNS look (one bar per
 *  risky cell). On top, the two discrete cell layers are drawn as floor points: raw OCCUPIED cells
 *  (amber) and the inflated half-robot-width clearance BORDER (steel blue).
 *
 *  Field data is pushed in via set_data() as the DENSE row-major arrays + grid meta (room frame:
 *  x, y horizontal); the viewer maps room (x, y, height) → OpenGL Y-up as (x, height, y). Self-
 *  contained (no OpenCV, no cortex). Non-Q_OBJECT (owner drives set_data()).
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
#include <array>
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
	struct CVertex { float x, y, z, r, g, b, nx, ny, nz; };   // position + colour + normal

	static constexpr float kBaseZ     = 0.02f;    // lift the surface just off the floor
	static constexpr float kColorRef  = 1.20f;    // height (m) mapped to the top (red) of the colour ramp
	static constexpr float kHeightCap = 1.60f;    // clamp displayed obstacle height (guards runaway z-band)
	static constexpr float kFlipX     = -1.0f;    // mirror room X so the scene's left/right match reality

	// Surprise-landscape colour ramp (low→high): dark floor → blue → orange → red, blended in the mesh.
	static std::array<float, 3> height_ramp(float h)
	{
		static constexpr float S[4][4] = {{0.00f, 0.10f, 0.12f, 0.20f},   // floor: dark slate
		                                  {0.18f, 0.15f, 0.40f, 0.95f},   // low:   blue
		                                  {0.55f, 1.00f, 0.60f, 0.10f},   // mid:   orange
		                                  {1.00f, 0.95f, 0.15f, 0.12f}};  // high:  red
		h = std::clamp(h, 0.f, 1.f);
		for(int i = 0; i < 3; ++i)
			if(h <= S[i + 1][0])
			{
				const float u = (h - S[i][0]) / (S[i + 1][0] - S[i][0]);
				return {S[i][1] + u * (S[i + 1][1] - S[i][1]),
				        S[i][2] + u * (S[i + 1][2] - S[i][2]),
				        S[i][3] + u * (S[i + 1][3] - S[i][3])};
			}
		return {S[3][1], S[3][2], S[3][3]};
	}

public:
	explicit GLGridFieldViewer(QWidget *parent = nullptr) : QOpenGLWidget(parent)
	{
		resize(760, 680);
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

	// Replace the displayed grid. `occupied` are the residual obstacle cells as flat points (x, y, z),
	// z = the cell's REAL top height (room frame) — the 3-D surface rises to that height. `border` are
	// the inflated clearance-ring cells (flat display z). [xmin,ymin] + cell·(w,h) is the grid extent.
	// `robot_tris` (optional) is the robot mesh already in the ROOM frame. Empty spans → layer skipped.
	void set_data(std::span<const QVector3D> occupied, std::span<const QVector3D> border,
	              float xmin, float ymin, float cell, int w, int h,
	              std::span<const QVector3D> robot_tris = {})
	{
		mesh_vertices_.clear();
		bar_vertices_.clear();
		cap_vertices_.clear();
		cell_vertices_.clear();
		robot_vertices_.clear();
		floor_vertices_.clear();
		field_cells_ = 0;

		std::vector<float> LH;
		int   LX = 0, LY = 0;

		float minx = std::numeric_limits<float>::max(), miny = minx, minz = minx;
		float maxx = std::numeric_limits<float>::lowest(), maxy = maxx, maxz = maxx;
		bool any = false;
		const auto grow = [&](float x, float y, float z)
		{
			minx = std::min(minx, x); maxx = std::max(maxx, x);
			miny = std::min(miny, y); maxy = std::max(maxy, y);
			minz = std::min(minz, z); maxz = std::max(maxz, z);
			any = true;
		};
		// ROOM (x, y, up) → OpenGL Y-up (kFlipX·x, up, y). kFlipX mirrors X so left/right match reality.
		// M is orthogonal, so the SAME map applies to normals — lighting stays correct after the mirror.
		const auto to_ogl = [](float x, float y, float up) { return QVector3D(kFlipX * x, up, y); };
		const auto height_col = [](float h_m) { return height_ramp(std::clamp(h_m / kColorRef, 0.f, 1.f)); };

		const bool have_grid = (w >= 2 and h >= 2 and cell > 1e-6f);
		if(have_grid)
		{
			const float x0 = xmin, y0 = ymin;
			const float x1 = xmin + w * cell, y1 = ymin + h * cell;

			// ── COARSE surface: a planar lattice over the grid extent, deformed FROM BELOW by a SUM OF
			//    GAUSSIANS — each occupied cell splats a smooth bump whose amplitude IS its real top
			//    height, MAX-blended so a cluster becomes a plateau at obstacle height (not additive
			//    spires). Cheap scatter-splat (±3σ window per cell), NOT neural Gaussian splatting. ──
			const float sigma  = 2.0f * cell;                             // bump half-width (metres)
			const float span   = std::max(x1 - x0, y1 - y0);
			const float lspace = std::max(sigma * 0.6f, span / 160.f);    // lattice spacing (coarse), capped
			LX = std::clamp(static_cast<int>(std::ceil((x1 - x0) / lspace)) + 1, 2, 200);
			LY = std::clamp(static_cast<int>(std::ceil((y1 - y0) / lspace)) + 1, 2, 200);
			const float sx = (x1 - x0) / (LX - 1), sy = (y1 - y0) / (LY - 1);
			LH.assign(static_cast<std::size_t>(LX) * static_cast<std::size_t>(LY), 0.f);
			const float inv2s2 = 1.0f / (2.0f * sigma * sigma);
			const float radius = 3.0f * sigma;
			for(const QVector3D &oc : occupied)
			{
				const float amp = std::min(oc.z(), kHeightCap);          // real obstacle top height (m), capped
				if(amp <= kBaseZ) continue;
				++field_cells_;
				const float cx = oc.x(), cy = oc.y();
				const int li0 = std::max(0,      static_cast<int>(std::floor((cx - radius - x0) / sx)));
				const int li1 = std::min(LX - 1, static_cast<int>(std::ceil ((cx + radius - x0) / sx)));
				const int lj0 = std::max(0,      static_cast<int>(std::floor((cy - radius - y0) / sy)));
				const int lj1 = std::min(LY - 1, static_cast<int>(std::ceil ((cy + radius - y0) / sy)));
				for(int lj = lj0; lj <= lj1; ++lj)
					for(int li = li0; li <= li1; ++li)
					{
						const float vx = x0 + li * sx, vy = y0 + lj * sy;
						const float d2 = (vx - cx) * (vx - cx) + (vy - cy) * (vy - cy);
						if(d2 > radius * radius) continue;
						const std::size_t k = static_cast<std::size_t>(lj) * LX + li;
						LH[k] = std::max(LH[k], amp * std::exp(-d2 * inv2s2));   // MAX-blend → plateau
					}
			}
			// Emit the lattice as two triangles per quad; colour by real height, normal from the height
			// gradient (mirrored in X to match to_ogl) so the surface is lit as terrain.
			const auto lheight = [&](int li, int lj) { return LH[static_cast<std::size_t>(lj) * LX + li]; };
			const auto lvtx = [&](int li, int lj) -> CVertex
			{
				const float ht = lheight(li, lj);
				const auto col = height_col(ht);
				const int il = std::max(0, li - 1), ir = std::min(LX - 1, li + 1);
				const int jl = std::max(0, lj - 1), jr = std::min(LY - 1, lj + 1);
				const float dHdx = (lheight(ir, lj) - lheight(il, lj)) / ((ir - il) * sx);
				const float dHdy = (lheight(li, jr) - lheight(li, jl)) / ((jr - jl) * sy);
				const QVector3D n = QVector3D(kFlipX * (-dHdx), 1.0f, -dHdy).normalized();
				const QVector3D p = to_ogl(x0 + li * sx, y0 + lj * sy, kBaseZ + ht);
				grow(p.x(), p.y(), p.z());
				return CVertex{p.x(), p.y(), p.z(), col[0], col[1], col[2], n.x(), n.y(), n.z()};
			};
			// Skip quads that are essentially floor (all corners below eps) so the mesh is ONLY the
			// obstacle bumps — free space stays open and the meshed floor grid shows through.
			constexpr float kFloorEps = 0.03f;   // metres
			mesh_vertices_.reserve(static_cast<std::size_t>(LX - 1) * (LY - 1) * 6);
			for(int lj = 0; lj + 1 < LY; ++lj)
				for(int li = 0; li + 1 < LX; ++li)
				{
					if(lheight(li, lj) < kFloorEps and lheight(li + 1, lj) < kFloorEps
					   and lheight(li, lj + 1) < kFloorEps and lheight(li + 1, lj + 1) < kFloorEps)
						continue;
					const CVertex v00 = lvtx(li, lj), v10 = lvtx(li + 1, lj), v01 = lvtx(li, lj + 1), v11 = lvtx(li + 1, lj + 1);
					mesh_vertices_.push_back(v00); mesh_vertices_.push_back(v10); mesh_vertices_.push_back(v11);
					mesh_vertices_.push_back(v00); mesh_vertices_.push_back(v11); mesh_vertices_.push_back(v01);
				}

			// Columns (toggle view): one bar per occupied cell rising to its real height.
			for(const QVector3D &oc : occupied)
			{
				const float hz = std::min(oc.z(), kHeightCap);
				if(hz <= kBaseZ) continue;
				const auto col = height_col(hz);
				const QVector3D foot = to_ogl(oc.x(), oc.y(), kBaseZ), tip = to_ogl(oc.x(), oc.y(), hz);
				bar_vertices_.push_back({foot.x(), foot.y(), foot.z(), 0.20f * col[0], 0.20f * col[1], 0.20f * col[2], 0, 0, 0});
				bar_vertices_.push_back({tip.x(),  tip.y(),  tip.z(),  col[0], col[1], col[2], 0, 0, 0});
				cap_vertices_.push_back({tip.x(),  tip.y(),  tip.z(),  col[0], col[1], col[2], 0, 0, 0});
			}

			// Meshed floor: a subtle grid over the grid extent so the ground plane reads in 3-D.
			const float step = std::max(cell * 4.0f, 0.25f);
			const std::array<float, 3> fc{0.38f, 0.42f, 0.48f};
			const auto floor_line = [&](const QVector3D &a, const QVector3D &b)
			{
				floor_vertices_.push_back({a.x(), a.y(), a.z(), fc[0], fc[1], fc[2], 0, 0, 0});
				floor_vertices_.push_back({b.x(), b.y(), b.z(), fc[0], fc[1], fc[2], 0, 0, 0});
			};
			for(float gx = x0; gx <= x1 + 1e-3f; gx += step)
				floor_line(to_ogl(gx, y0, 0.f), to_ogl(gx, y1, 0.f));
			for(float gy = y0; gy <= y1 + 1e-3f; gy += step)
				floor_line(to_ogl(x0, gy, 0.f), to_ogl(x1, gy, 0.f));
		}

		// Discrete cell layers → floor points (unlit). Only shown in COLUMNS mode; in MESH mode the
		// belief is conveyed entirely by the blended surface, so these separate layers are hidden.
		const auto add_cells = [&](std::span<const QVector3D> pts, float cr, float cg, float cb)
		{
			for(const QVector3D &p : pts)
			{
				const QVector3D q = to_ogl(p.x(), p.y(), p.z());
				cell_vertices_.push_back({q.x(), q.y(), q.z(), cr, cg, cb, 0, 0, 0});
				grow(q.x(), q.y(), q.z());
			}
		};
		occupied_count_ = occupied.size();
		add_cells(occupied, 1.00f, 0.70f, 0.10f);   // colour A — amber
		add_cells(border,   0.30f, 0.60f, 0.95f);   // colour B — steel blue

		// Robot mesh (room frame) → lit gray triangles with a flat per-face normal, so the scene reads
		// with the robot in place. Two triangle winding is respected; back faces just get ambient.
		robot_vertices_.reserve((robot_tris.size() / 3) * 3);
		for(std::size_t tr = 0; tr + 2 < robot_tris.size(); tr += 3)
		{
			const QVector3D &Ar = robot_tris[tr], &Br = robot_tris[tr + 1], &Cr = robot_tris[tr + 2];  // room
			// Face normal in ROOM space, then mapped by the same orthogonal to_ogl transform (so the
			// X-mirror doesn't invert the shading).
			QVector3D nr = QVector3D::crossProduct(Br - Ar, Cr - Ar);
			QVector3D n(kFlipX * nr.x(), nr.z(), nr.y());
			if(n.lengthSquared() > 1e-12f) n.normalize();
			const QVector3D a = to_ogl(Ar.x(), Ar.y(), Ar.z());
			const QVector3D b = to_ogl(Br.x(), Br.y(), Br.z());
			const QVector3D c = to_ogl(Cr.x(), Cr.y(), Cr.z());
			for(const QVector3D &v : {a, b, c})
			{
				robot_vertices_.push_back({v.x(), v.y(), v.z(), 0.62f, 0.66f, 0.72f, n.x(), n.y(), n.z()});
				grow(v.x(), v.y(), v.z());
			}
		}

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
			const float ex = std::max(0.001f, (maxx - minx) * 0.5f);
			const float ey = std::max(0.001f, (maxy - miny) * 0.5f);
			const float ez = std::max(0.001f, (maxz - minz) * 0.5f);
			scene_radius_ = std::max({ex, ey, ez});
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
			layout(location = 2) in vec3 normal;
			uniform mat4 u_mvp;
			uniform float u_point_size;
			out vec3 v_color;
			out vec3 v_normal;
			void main() { gl_Position = u_mvp * vec4(position, 1.0); gl_PointSize = u_point_size; v_color = color; v_normal = normal; }
		)");
		program_.addShaderFromSourceCode(QOpenGLShader::Fragment, R"(
			#version 330 core
			in vec3 v_color;
			in vec3 v_normal;
			uniform float u_lit;          // 1 = apply diffuse lighting, 0 = flat colour
			uniform vec3  u_light_dir;    // world-space direction TO the light
			out vec4 fragColor;
			void main() {
				float shade = 1.0;
				if(u_lit > 0.5) {
					vec3 n = normalize(v_normal);
					float d = max(dot(n, normalize(u_light_dir)), 0.0);
					shade = 0.35 + 0.65 * d;   // ambient + diffuse
				}
				fragColor = vec4(v_color * shade, 1.0);
			}
		)");
		program_.link();
		u_mvp_loc_ = program_.uniformLocation("u_mvp");
		u_point_size_loc_ = program_.uniformLocation("u_point_size");
		u_lit_loc_ = program_.uniformLocation("u_lit");
		u_light_dir_loc_ = program_.uniformLocation("u_light_dir");

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
		program_.setUniformValue(u_light_dir_loc_, QVector3D(0.4f, 0.85f, 0.35f));   // fixed world "sun"
		vao_.bind();
		scene_vbo_.bind();

		const auto bind_attrs = [&]()
		{
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(CVertex), reinterpret_cast<const void *>(0));
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(CVertex), reinterpret_cast<const void *>(3 * sizeof(float)));
			glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(CVertex), reinterpret_cast<const void *>(6 * sizeof(float)));
			glEnableVertexAttribArray(0);
			glEnableVertexAttribArray(1);
			glEnableVertexAttribArray(2);
		};
		const auto draw = [&](GLenum mode, std::size_t first, std::size_t count, float point_size, bool lit)
		{
			if(count == 0) return;
			program_.setUniformValue(u_point_size_loc_, point_size);
			program_.setUniformValue(u_lit_loc_, lit ? 1.0f : 0.0f);
			bind_attrs();
			glDrawArrays(mode, static_cast<GLint>(first), static_cast<GLsizei>(count));
		};

		// VBO layout: [axes | floor | mesh | bars | caps | cells | robot].
		std::size_t off = 0;
		glLineWidth(1.0f);
		draw(GL_LINES, off, axis_vertices_.size(),  1.0f, false);   off += axis_vertices_.size();
		draw(GL_LINES, off, floor_vertices_.size(), 1.0f, false);   off += floor_vertices_.size();
		if(mesh_mode_)
			draw(GL_TRIANGLES, off, mesh_vertices_.size(), 1.0f, true);
		off += mesh_vertices_.size();
		if(not mesh_mode_)
		{
			glLineWidth(2.0f);
			draw(GL_LINES,  off, bar_vertices_.size(), 1.0f, false);
			draw(GL_POINTS, off + bar_vertices_.size(), cap_vertices_.size(), 6.0f, false);
		}
		off += bar_vertices_.size() + cap_vertices_.size();
		if(not mesh_mode_)                                             // cells only in columns mode; the mesh
			draw(GL_POINTS, off, cell_vertices_.size(), 7.0f, false);   //   blends the belief itself
		off += cell_vertices_.size();
		draw(GL_TRIANGLES, off, robot_vertices_.size(), 1.0f, true);   // robot mesh, always lit-shaded

		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(2);
		scene_vbo_.release();
		vao_.release();
		program_.release();

		QPainter painter(this);
		painter.setRenderHint(QPainter::TextAntialiasing, true);
		painter.setPen(QColor(235, 235, 235));
		painter.drawText(QRect(10, 10, width() - 20, 40), Qt::AlignLeft | Qt::AlignTop,
		                 QString("%1   cells: %2   occupied: %3   %4 Hz\n"
		                         "height = obstacle height (m), blue→orange→red low→high   [M=mesh/columns  drag=rotate  wheel=zoom  R=reset]")
		                     .arg(mesh_mode_ ? "MESH" : "COLUMNS")
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
		if(event->key() == Qt::Key_M) { mesh_mode_ = not mesh_mode_; update(); return; }
		QOpenGLWidget::keyPressEvent(event);
	}

private:
	void updateGpuBuffers()
	{
		if(not context() or not scene_vbo_.isCreated())
			return;
		makeCurrent();
		std::vector<CVertex> all;
		all.reserve(axis_vertices_.size() + floor_vertices_.size() + mesh_vertices_.size() + bar_vertices_.size()
		            + cap_vertices_.size() + cell_vertices_.size() + robot_vertices_.size());
		all.insert(all.end(), axis_vertices_.begin(),  axis_vertices_.end());
		all.insert(all.end(), floor_vertices_.begin(), floor_vertices_.end());
		all.insert(all.end(), mesh_vertices_.begin(),  mesh_vertices_.end());
		all.insert(all.end(), bar_vertices_.begin(),   bar_vertices_.end());
		all.insert(all.end(), cap_vertices_.begin(),   cap_vertices_.end());
		all.insert(all.end(), cell_vertices_.begin(),  cell_vertices_.end());
		all.insert(all.end(), robot_vertices_.begin(), robot_vertices_.end());
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
			{cx, cy, cz, 0.9f, 0.2f, 0.2f, 0, 0, 0}, {cx + s, cy, cz, 0.9f, 0.2f, 0.2f, 0, 0, 0},   // X red
			{cx, cy, cz, 0.2f, 0.9f, 0.2f, 0, 0, 0}, {cx, cy + s, cz, 0.2f, 0.9f, 0.2f, 0, 0, 0},   // Y green (up)
			{cx, cy, cz, 0.3f, 0.5f, 1.0f, 0, 0, 0}, {cx, cy, cz + s, 0.3f, 0.5f, 1.0f, 0, 0, 0},   // Z blue
		};
	}

	void resetView()
	{
		yaw_deg_ = 30.0f; pitch_deg_ = -35.0f; pan_x_ = 0.0f; pan_y_ = 0.0f;
		cam_dist_ = std::max(2.0f, scene_radius_ * 2.5f);
		user_interacted_ = false;
	}

	std::vector<CVertex> axis_vertices_, mesh_vertices_, bar_vertices_, cap_vertices_, cell_vertices_, robot_vertices_, floor_vertices_;
	int         field_cells_    = 0;
	std::size_t occupied_count_ = 0;
	bool        mesh_mode_      = true;   // start in the continuous-mesh view

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
	int u_mvp_loc_ = -1, u_point_size_loc_ = -1, u_lit_loc_ = -1, u_light_dir_loc_ = -1;
};

}   // namespace rc::viewers

#endif   // RC_COMMON_GL_GRID_FIELD_VIEWER_H
