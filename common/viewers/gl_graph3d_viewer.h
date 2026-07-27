/*
 *  Stratified 3D viewer for the DSR graph (active_inference/common).
 *
 *  Draws a rc::graph3d::Scene: z is the ABSTRACTION STRATUM (world → instances → affordances →
 *  meta-concepts → agents) and x/y is the node's TRUE METRIC POSE, so hierarchy reads vertically
 *  while the floor plan still reads horizontally. Pure renderer: geometry comes in through
 *  set_scene(); it knows nothing about DSR, the graph or the media plane, so any agent can drive it.
 *  Non-Q_OBJECT (owner-driven) → no MOC; the pick hook is a std::function, not a signal.
 *
 *  Two-sided lighting (abs(N·L)) matches the house style of gl_mesh_viewer.h: glyphs are closed
 *  solids, but never rendering a face fully black keeps small marks readable at a distance.
 *
 *  The scene is in ROOM coordinates, metres, Z-UP. GL is Y-UP, so every position goes through
 *  gl() — that mapping lives here and nowhere else.
 */
#ifndef RC_COMMON_GL_GRAPH3D_VIEWER_H
#define RC_COMMON_GL_GRAPH3D_VIEWER_H

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
#include <QVector4D>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <numbers>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../graph3d/graph3d_scene.h"
#include "../obj/obj_loader.h"

namespace rc::viewers
{

class GLGraph3DViewer : public QOpenGLWidget, protected QOpenGLFunctions
{
public:
	explicit GLGraph3DViewer(QWidget *parent = nullptr) : QOpenGLWidget(parent)
	{
		resize(900, 760);
		setWindowTitle("graph 3D");
		setFocusPolicy(Qt::StrongFocus);
		setMouseTracking(true);
	}

	~GLGraph3DViewer() override
	{
		if(context())
		{
			makeCurrent();
			if(solid_vbo_.isCreated()) solid_vbo_.destroy();
			if(line_vbo_.isCreated())  line_vbo_.destroy();
			if(vao_.isCreated())       vao_.destroy();
			doneCurrent();
		}
	}

	// Replace the scene. Rebuilds both buffers and auto-frames the camera until the user moves it.
	void set_scene(const graph3d::Scene &scene)
	{
		scene_ = scene;
		rebuild_solids();
		rebuild_lines();
		frame_camera();
		upload();
		update();
	}

	// Invoked on a click that lands on a node, with the same (id, type) payload dsr_gui's
	// GraphViewer::view_data_signal carries — so an owner can route both to one handler.
	void set_pick_callback(std::function<void(std::uint64_t, const std::string &)> cb)
	{
		pick_cb_ = std::move(cb);
	}

	[[nodiscard]] std::uint64_t selected() const noexcept { return selected_id_; }

protected:
	void initializeGL() override
	{
		initializeOpenGLFunctions();
		glEnable(GL_DEPTH_TEST);
		glClearColor(0.085f, 0.095f, 0.115f, 1.0f);

		solid_.addShaderFromSourceCode(QOpenGLShader::Vertex, R"(
			#version 330 core
			layout(location = 0) in vec3 position;
			layout(location = 1) in vec3 normal;
			layout(location = 2) in vec3 color;
			uniform mat4 u_mvp;
			out vec3 v_normal;
			out vec3 v_color;
			void main() { gl_Position = u_mvp * vec4(position, 1.0); v_normal = normal; v_color = color; }
		)");
		solid_.addShaderFromSourceCode(QOpenGLShader::Fragment, R"(
			#version 330 core
			in vec3 v_normal;
			in vec3 v_color;
			out vec4 fragColor;
			uniform vec3 u_light;
			void main()
			{
				vec3 N = normalize(v_normal);
				float d = abs(dot(N, u_light));            // two-sided
				float shade = 0.34 + 0.66 * d;             // ambient + diffuse
				fragColor = vec4(v_color * shade, 1.0);
			}
		)");
		solid_.link();
		u_solid_mvp_   = solid_.uniformLocation("u_mvp");
		u_solid_light_ = solid_.uniformLocation("u_light");

		flat_.addShaderFromSourceCode(QOpenGLShader::Vertex, R"(
			#version 330 core
			layout(location = 0) in vec3 position;
			layout(location = 1) in vec4 color;
			uniform mat4 u_mvp;
			out vec4 v_color;
			void main() { gl_Position = u_mvp * vec4(position, 1.0); v_color = color; }
		)");
		flat_.addShaderFromSourceCode(QOpenGLShader::Fragment, R"(
			#version 330 core
			in vec4 v_color;
			out vec4 fragColor;
			void main() { fragColor = v_color; }
		)");
		flat_.link();
		u_flat_mvp_ = flat_.uniformLocation("u_mvp");

		vao_.create();
		solid_vbo_.create();
		line_vbo_.create();
		upload();
	}

	void resizeGL(int w, int h) override { glViewport(0, 0, w, h); }

	void paintGL() override
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		const float aspect = static_cast<float>(std::max(1, width())) / static_cast<float>(std::max(1, height()));
		QMatrix4x4 proj;
		proj.perspective(45.0f, aspect, std::max(0.01f, cam_dist_ * 0.002f), cam_dist_ * 12.0f + radius_ * 12.0f);
		QMatrix4x4 view;
		view.translate(pan_x_, pan_y_, -cam_dist_);
		view.rotate(pitch_deg_, 1.0f, 0.0f, 0.0f);
		view.rotate(yaw_deg_, 0.0f, 1.0f, 0.0f);
		view.translate(-center_);
		last_mvp_ = proj * view;

		vao_.bind();

		if(solid_.isLinked() and solid_count_ > 0)
		{
			solid_.bind();
			solid_.setUniformValue(u_solid_mvp_, last_mvp_);
			solid_.setUniformValue(u_solid_light_, QVector3D(0.42f, 0.78f, 0.46f).normalized());
			solid_vbo_.bind();
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(SolidVertex), nullptr);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(SolidVertex),
			                      reinterpret_cast<const void *>(offsetof(SolidVertex, nx)));
			glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(SolidVertex),
			                      reinterpret_cast<const void *>(offsetof(SolidVertex, r)));
			glEnableVertexAttribArray(0);
			glEnableVertexAttribArray(1);
			glEnableVertexAttribArray(2);
			glDrawArrays(GL_TRIANGLES, 0, solid_count_);
			glDisableVertexAttribArray(0);
			glDisableVertexAttribArray(1);
			glDisableVertexAttribArray(2);
			solid_vbo_.release();
			solid_.release();
		}

		if(flat_.isLinked() and line_count_ > 0)
		{
			// Depth-test but do NOT depth-write: the translucent ownership/member bundles must not
			// occlude each other or the labels' anchors, and there is no meaningful sort order for them.
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			glDepthMask(GL_FALSE);
			flat_.bind();
			flat_.setUniformValue(u_flat_mvp_, last_mvp_);
			line_vbo_.bind();
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(LineVertex), nullptr);
			glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(LineVertex),
			                      reinterpret_cast<const void *>(offsetof(LineVertex, r)));
			glEnableVertexAttribArray(0);
			glEnableVertexAttribArray(1);
			glDrawArrays(GL_LINES, 0, line_count_);
			glDisableVertexAttribArray(0);
			glDisableVertexAttribArray(1);
			line_vbo_.release();
			flat_.release();
			glDepthMask(GL_TRUE);
			glDisable(GL_BLEND);
		}

		vao_.release();

		draw_overlay();
	}

	void mousePressEvent(QMouseEvent *e) override
	{
		last_ = e->pos();
		press_ = e->pos();
		if(e->button() == Qt::LeftButton) { rotating_ = true; }
		else if(e->button() == Qt::RightButton or e->button() == Qt::MiddleButton) { panning_ = true; user_interacted_ = true; }
	}

	void mouseMoveEvent(QMouseEvent *e) override
	{
		const QPoint d = e->pos() - last_;
		last_ = e->pos();
		if(rotating_)
		{
			if(d.manhattanLength() > 0) user_interacted_ = true;
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

	void mouseReleaseEvent(QMouseEvent *e) override
	{
		const bool was_click = (e->pos() - press_).manhattanLength() < 4;
		rotating_ = false; panning_ = false;
		if(was_click and e->button() == Qt::LeftButton)
			pick_at(e->pos());
	}

	void wheelEvent(QWheelEvent *e) override
	{
		const float steps = e->angleDelta().y() / 120.0f;
		if(std::abs(steps) < 1e-4f) return;
		cam_dist_ = std::clamp(cam_dist_ * std::pow(0.85f, steps), 0.05f, 100000.0f);
		user_interacted_ = true;
		update();
	}

	void keyPressEvent(QKeyEvent *e) override
	{
		switch(e->key())
		{
			case Qt::Key_R:   reset_camera();                                    break;
			case Qt::Key_O:   show_all_ownership_ = not show_all_ownership_; rebuild_lines(); upload(); break;
			case Qt::Key_L:   show_labels_ = not show_labels_;                   break;
			case Qt::Key_G:   show_strata_ = not show_strata_; rebuild_lines(); upload(); break;
			case Qt::Key_T:   top_down();                                        break;
			case Qt::Key_Escape: selected_id_ = 0; rebuild_lines(); upload();    break;
			default: QOpenGLWidget::keyPressEvent(e);                            return;
		}
		update();
	}

private:
	struct SolidVertex { float x, y, z, nx, ny, nz, r, g, b; };
	struct LineVertex  { float x, y, z, r, g, b, a; };
	struct CachedMesh  { std::vector<QVector3D> tris; QVector3D lo, hi; bool ok = false; };

	// Scene is room-frame Z-UP; GL is Y-UP. This is the ONLY place the convention is bridged.
	[[nodiscard]] static QVector3D gl(const graph3d::Vec3 &p) noexcept { return {p[0], p[2], -p[1]}; }
	[[nodiscard]] static QVector3D gl(float x, float y, float z) noexcept { return {x, z, -y}; }

	// ── solid geometry ────────────────────────────────────────────────────────────────────────
	void add_tri(const QVector3D &a, const QVector3D &b, const QVector3D &c, const graph3d::Rgb &col)
	{
		QVector3D n = QVector3D::crossProduct(b - a, c - a);
		if(n.lengthSquared() > 0.0f) n.normalize();
		for(const QVector3D &v : {a, b, c})
			solid_verts_.push_back({v.x(), v.y(), v.z(), n.x(), n.y(), n.z(), col[0], col[1], col[2]});
	}
	void add_quad(const QVector3D &a, const QVector3D &b, const QVector3D &c, const QVector3D &d,
	              const graph3d::Rgb &col)
	{
		add_tri(a, b, c, col);
		add_tri(a, c, d, col);
	}

	// Axis-aligned-in-local box, yawed about the world z axis so orientation is visible.
	void add_box(const graph3d::Vec3 &c, float hx, float hy, float hz, float yaw, const graph3d::Rgb &col)
	{
		const float cs = std::cos(yaw), sn = std::sin(yaw);
		const auto p = [&](float lx, float ly, float lz)
		{ return gl(c[0] + cs * lx - sn * ly, c[1] + sn * lx + cs * ly, c[2] + lz); };
		const QVector3D v000 = p(-hx, -hy, -hz), v100 = p(hx, -hy, -hz), v110 = p(hx, hy, -hz), v010 = p(-hx, hy, -hz);
		const QVector3D v001 = p(-hx, -hy, hz),  v101 = p(hx, -hy, hz),  v111 = p(hx, hy, hz),  v011 = p(-hx, hy, hz);
		add_quad(v001, v101, v111, v011, col);   // top
		add_quad(v010, v110, v100, v000, col);   // bottom
		add_quad(v000, v100, v101, v001, col);
		add_quad(v110, v010, v011, v111, col);
		add_quad(v100, v110, v111, v101, col);
		add_quad(v010, v000, v001, v011, col);
	}

	void add_prism(const graph3d::Vec3 &c, float r, float hz, int sides, const graph3d::Rgb &col)
	{
		const float step = 2.0f * std::numbers::pi_v<float> / static_cast<float>(sides);
		for(int i = 0; i < sides; ++i)
		{
			const float a0 = step * static_cast<float>(i), a1 = step * static_cast<float>(i + 1);
			const QVector3D b0 = gl(c[0] + r * std::cos(a0), c[1] + r * std::sin(a0), c[2] - hz);
			const QVector3D b1 = gl(c[0] + r * std::cos(a1), c[1] + r * std::sin(a1), c[2] - hz);
			const QVector3D t0 = gl(c[0] + r * std::cos(a0), c[1] + r * std::sin(a0), c[2] + hz);
			const QVector3D t1 = gl(c[0] + r * std::cos(a1), c[1] + r * std::sin(a1), c[2] + hz);
			add_quad(b0, b1, t1, t0, col);
			add_tri(gl(c[0], c[1], c[2] + hz), t0, t1, col);
			add_tri(gl(c[0], c[1], c[2] - hz), b1, b0, col);
		}
	}

	// Apex-down cone: an agent is a PIN, pointing at the footprint it owns.
	void add_cone_down(const graph3d::Vec3 &c, float r, float h, int sides, const graph3d::Rgb &col)
	{
		const QVector3D apex = gl(c[0], c[1], c[2] - h);
		const float step = 2.0f * std::numbers::pi_v<float> / static_cast<float>(sides);
		for(int i = 0; i < sides; ++i)
		{
			const float a0 = step * static_cast<float>(i), a1 = step * static_cast<float>(i + 1);
			const QVector3D r0 = gl(c[0] + r * std::cos(a0), c[1] + r * std::sin(a0), c[2]);
			const QVector3D r1 = gl(c[0] + r * std::cos(a1), c[1] + r * std::sin(a1), c[2]);
			add_tri(apex, r1, r0, col);
			add_tri(gl(c[0], c[1], c[2]), r0, r1, col);
		}
	}

	void add_octahedron(const graph3d::Vec3 &c, float r, const graph3d::Rgb &col)
	{
		const QVector3D px = gl(c[0] + r, c[1], c[2]), nx = gl(c[0] - r, c[1], c[2]);
		const QVector3D py = gl(c[0], c[1] + r, c[2]), ny = gl(c[0], c[1] - r, c[2]);
		const QVector3D pz = gl(c[0], c[1], c[2] + r), nz = gl(c[0], c[1], c[2] - r);
		add_tri(pz, px, py, col); add_tri(pz, py, nx, col); add_tri(pz, nx, ny, col); add_tri(pz, ny, px, col);
		add_tri(nz, py, px, col); add_tri(nz, nx, py, col); add_tri(nz, ny, nx, col); add_tri(nz, px, ny, col);
	}

	// Subdivided octahedron — cheap ball, no pole pinching, and no trig per vertex.
	void add_ball(const graph3d::Vec3 &c, float r, const graph3d::Rgb &col)
	{
		static const std::vector<std::array<QVector3D, 3>> base = []
		{
			const QVector3D px{1, 0, 0}, nx{-1, 0, 0}, py{0, 1, 0}, ny{0, -1, 0}, pz{0, 0, 1}, nz{0, 0, -1};
			return std::vector<std::array<QVector3D, 3>>{
			    {pz, px, py}, {pz, py, nx}, {pz, nx, ny}, {pz, ny, px},
			    {nz, py, px}, {nz, nx, py}, {nz, ny, nx}, {nz, px, ny}};
		}();
		const auto put = [&](QVector3D a, QVector3D b, QVector3D cc)
		{
			const QVector3D o = gl(c);
			add_tri(o + a * r, o + b * r, o + cc * r, col);
		};
		for(const auto &t : base)
		{
			const QVector3D ab = (t[0] + t[1]).normalized();
			const QVector3D bc = (t[1] + t[2]).normalized();
			const QVector3D ca = (t[2] + t[0]).normalized();
			put(t[0], ab, ca); put(ab, t[1], bc); put(ca, bc, t[2]); put(ab, bc, ca);
		}
	}

	// Flat annulus band — a rig IS a ring of slots, so its glyph says so.
	void add_annulus(const graph3d::Vec3 &c, float r_out, float r_in, float hz, const graph3d::Rgb &col)
	{
		constexpr int kSeg = 24;
		const float step = 2.0f * std::numbers::pi_v<float> / static_cast<float>(kSeg);
		for(int i = 0; i < kSeg; ++i)
		{
			const float a0 = step * static_cast<float>(i), a1 = step * static_cast<float>(i + 1);
			const auto ring = [&](float rad, float a, float dz)
			{ return gl(c[0] + rad * std::cos(a), c[1] + rad * std::sin(a), c[2] + dz); };
			add_quad(ring(r_out, a0, hz), ring(r_out, a1, hz), ring(r_in, a1, hz), ring(r_in, a0, hz), col);
			add_quad(ring(r_in, a0, -hz), ring(r_in, a1, -hz), ring(r_out, a1, -hz), ring(r_out, a0, -hz), col);
			add_quad(ring(r_out, a0, -hz), ring(r_out, a1, -hz), ring(r_out, a1, hz), ring(r_out, a0, hz), col);
			add_quad(ring(r_in, a0, hz), ring(r_in, a1, hz), ring(r_in, a1, -hz), ring(r_in, a0, -hz), col);
		}
	}

	// ── agent-published display meshes ────────────────────────────────────────────────────────
	// A component-relative path ("chair_concept/meshes/chair.obj") is hosted with the PRODUCING
	// agent, under the components root = the parent of this agent's run dir. Same order the
	// voxelizer uses, so both viewers find the same file.
	[[nodiscard]] static std::optional<std::filesystem::path> resolve_mesh(const std::string &rel)
	{
		namespace fs = std::filesystem;
		if(rel.empty())
			return std::nullopt;
		for(const fs::path &root : {fs::current_path().parent_path(), fs::current_path()})
			if(fs::path cand = root / rel; fs::exists(cand))
				return cand;
		return rc::obj::resolve_robot_mesh_path(rel);
	}

	// Cached by path — the scene is rebuilt several times a second and OBJ parsing is file I/O.
	// FAILURES ARE CACHED TOO: a missing mesh must not re-stat the filesystem on every frame.
	[[nodiscard]] const CachedMesh *mesh_for(const std::string &path)
	{
		if(const auto it = mesh_cache_.find(path); it != mesh_cache_.end())
			return it->second.ok ? &it->second : nullptr;
		CachedMesh cm;
		if(const auto p = resolve_mesh(path); p.has_value())
			if(const auto d = rc::obj::load_obj_mesh_data(p.value());
			   d.has_value() and not d->triangles.empty())
			{
				cm.tris = d->triangles;
				cm.lo   = d->bb_min;
				cm.hi   = d->bb_max;
				cm.ok   = true;
			}
		const auto [it, _] = mesh_cache_.emplace(path, std::move(cm));
		return it->second.ok ? &it->second : nullptr;
	}

	// Draw the published mesh in place of the glyph, resized PER LOCAL AXIS to n.draw. The scale is
	// deliberately non-uniform: most nodes ask for a schematic cube (a full-size table would punch
	// through the plane above in a view whose planes are ~1.2 m apart), but a wall asks for its true
	// length and a capped height so the walls trace the real footprint. The mesh is authored Z-up
	// and centred in x/y, so it is seated ON its plane and yawed about the vertical — its own axis
	// convention carries through untouched, which is the whole point of preferring it to a
	// hand-built glyph.
	void add_mesh(const graph3d::Node3D &n, const CachedMesh &m)
	{
		// A flat asset (the wall quad has zero y extent) must not blow the scale up to infinity —
		// leave a degenerate axis alone; every vertex on it is at the centre anyway.
		const auto axis = [](const float span, const float want)
		{ return span > 1e-5f ? want / span : 1.0f; };
		const float sx = axis(m.hi.x() - m.lo.x(), n.draw[0]);
		const float sy = axis(m.hi.y() - m.lo.y(), n.draw[1]);
		const float sz = axis(m.hi.z() - m.lo.z(), n.draw[2]);
		const float cx = 0.5f * (m.lo.x() + m.hi.x());
		const float cy = 0.5f * (m.lo.y() + m.hi.y());
		const float z0 = m.lo.z();
		const float cs = std::cos(n.yaw), sn = std::sin(n.yaw);
		const auto  place = [&](const QVector3D &v)
		{
			const float lx = (v.x() - cx) * sx, ly = (v.y() - cy) * sy, lz = (v.z() - z0) * sz;
			return gl(n.pos[0] + cs * lx - sn * ly, n.pos[1] + sn * lx + cs * ly, n.pos[2] + lz);
		};
		for(std::size_t t = 0; t + 2 < m.tris.size(); t += 3)
			add_tri(place(m.tris[t]), place(m.tris[t + 1]), place(m.tris[t + 2]), n.color);
	}

	void rebuild_solids()
	{
		solid_verts_.clear();
		using graph3d::Glyph;
		for(const auto &n : scene_.nodes)
		{
			// Prefer the agent's own display mesh; the glyphs below are the fallback for everything
			// that publishes none (walls, the floor, obstacles, the room, rigs).
			if(not n.mesh_path.empty())
				if(const CachedMesh *m = mesh_for(n.mesh_path); m != nullptr)
				{
					add_mesh(n, *m);
					continue;
				}
			const float r = n.radius;
			switch(n.glyph)
			{
				case Glyph::Box:      add_box(n.pos, r, r * 0.78f, r * 0.55f, n.yaw, n.color); break;
				case Glyph::Cylinder: add_prism(n.pos, r * 0.62f, r * 0.95f, 14, n.color);     break;
				case Glyph::Chair:
				{
					// Seat + backrest: the backrest is what makes the fitted yaw legible at a glance,
					// which is the whole reason the dining-set rig exists.
					//
					// The backrest goes on −LOCAL_Y, which is the chair model's own convention — its
					// outward normal is (sinψ, −cosψ), i.e. −y at yaw 0. Building it on −local_x
					// instead drew every chair a consistent 90° off. Authoring the glyph in the
					// model's axes keeps that correct without a magic rotation offset, so the
					// backrest is thin in y and wide in x.
					add_box(n.pos, r * 0.72f, r * 0.72f, r * 0.16f, n.yaw, n.color);
					const float cs = std::cos(n.yaw), sn = std::sin(n.yaw);
					const graph3d::Vec3 back{n.pos[0] + sn * r * 0.62f, n.pos[1] - cs * r * 0.62f,
					                         n.pos[2] + r * 0.55f};
					add_box(back, r * 0.72f, r * 0.14f, r * 0.55f, n.yaw, n.color);
					break;
				}
				case Glyph::Diamond:  add_octahedron(n.pos, r * 0.85f, n.color);               break;
				case Glyph::Ring:     add_annulus(n.pos, r * 1.5f, r * 0.95f, r * 0.16f, n.color); break;
				case Glyph::Pin:      add_cone_down(n.pos, r, r * 1.7f, 14, n.color);          break;
				case Glyph::Robot:
					add_prism(n.pos, r * 0.75f, r * 0.5f, 12, n.color);
					add_box({n.pos[0], n.pos[1], n.pos[2] + r * 0.95f}, r * 0.5f, r * 0.42f, r * 0.34f,
					        n.yaw, n.color);
					break;
				case Glyph::Sphere:
				default:              add_ball(n.pos, r * 0.85f, n.color);                     break;
			}
		}
		solid_count_ = static_cast<int>(solid_verts_.size());
	}

	// ── line geometry ─────────────────────────────────────────────────────────────────────────
	void push_line(const QVector3D &a, const QVector3D &b, const graph3d::Rgb &c, float alpha)
	{
		line_verts_.push_back({a.x(), a.y(), a.z(), c[0], c[1], c[2], alpha});
		line_verts_.push_back({b.x(), b.y(), b.z(), c[0], c[1], c[2], alpha});
	}

	void push_dashed(const QVector3D &a, const QVector3D &b, const graph3d::Rgb &c, float alpha)
	{
		constexpr int kSeg = 11;
		for(int i = 0; i < kSeg; i += 2)
		{
			const float t0 = static_cast<float>(i) / kSeg, t1 = static_cast<float>(i + 1) / kSeg;
			push_line(a + (b - a) * t0, a + (b - a) * t1, c, alpha);
		}
	}

	// Quadratic Bezier with a lifted control point, drawn as a 3-strand bundle. GL core profile
	// guarantees only 1-pixel lines, so "thick" has to be faked by parallel strands — which also
	// makes a fan of them read as a ribbon rather than as spokes.
	void push_bundle(const QVector3D &a, const QVector3D &b, const graph3d::Rgb &c, float alpha)
	{
		const QVector3D mid = (a + b) * 0.5f;
		const QVector3D d   = b - a;
		const float     len = d.length();
		QVector3D perp = QVector3D::crossProduct(d, QVector3D(0, 1, 0));
		if(perp.lengthSquared() < 1e-6f) perp = QVector3D(1, 0, 0);
		perp.normalize();
		const QVector3D ctrl = mid + perp * (len * 0.16f);

		constexpr int kSeg = 18;
		for(int s = -1; s <= 1; ++s)
		{
			const QVector3D off = perp * (static_cast<float>(s) * len * 0.012f);
			QVector3D prev = a + off;
			for(int i = 1; i <= kSeg; ++i)
			{
				const float t = static_cast<float>(i) / kSeg;
				const float u = 1.0f - t;
				const QVector3D p = a * (u * u) + ctrl * (2.0f * u * t) + b * (t * t) + off;
				push_line(prev, p, c, alpha * (s == 0 ? 1.0f : 0.55f));
				prev = p;
			}
		}
	}

	void rebuild_lines()
	{
		line_verts_.clear();
		using graph3d::EdgeKind;

		// Faint stratum planes: a rectangle plus a coarse grid at each level, so a node's height is
		// readable as "which plane is it on" and not guessed from perspective.
		if(show_strata_ and not scene_.nodes.empty())
		{
			float minx = std::numeric_limits<float>::max(), miny = minx;
			float maxx = std::numeric_limits<float>::lowest(), maxy = maxx;
			for(const auto &n : scene_.nodes)
			{
				minx = std::min(minx, n.pos[0]); maxx = std::max(maxx, n.pos[0]);
				miny = std::min(miny, n.pos[1]); maxy = std::max(maxy, n.pos[1]);
			}
			for(const auto &p : scene_.ground.polygon)
			{
				minx = std::min(minx, p[0]); maxx = std::max(maxx, p[0]);
				miny = std::min(miny, p[1]); maxy = std::max(maxy, p[1]);
			}
			const float pad = 0.4f;
			minx -= pad; miny -= pad; maxx += pad; maxy += pad;
			constexpr graph3d::Rgb kPlane{0.45f, 0.52f, 0.62f};
			for(std::size_t s = 0; s < scene_.level_z.size(); ++s)
			{
				const float z = scene_.level_z[s];
				// The room plane is the frame of reference everything else is read against, so it
				// gets the strong grid; the rest stay faint.
				const float a = (static_cast<int>(s) == graph3d::kLevelRoom) ? 0.22f : 0.10f;
				push_line(gl(minx, miny, z), gl(maxx, miny, z), kPlane, a);
				push_line(gl(maxx, miny, z), gl(maxx, maxy, z), kPlane, a);
				push_line(gl(maxx, maxy, z), gl(minx, maxy, z), kPlane, a);
				push_line(gl(minx, maxy, z), gl(minx, miny, z), kPlane, a);
				constexpr int kDiv = 4;
				for(int i = 1; i < kDiv; ++i)
				{
					const float fx = minx + (maxx - minx) * static_cast<float>(i) / kDiv;
					const float fy = miny + (maxy - miny) * static_cast<float>(i) / kDiv;
					push_line(gl(fx, miny, z), gl(fx, maxy, z), kPlane, a * 0.5f);
					push_line(gl(minx, fy, z), gl(maxx, fy, z), kPlane, a * 0.5f);
				}
			}
		}

		// The real room footprint, drawn ON THE ROOM PLANE — the frame of reference that makes x/y
		// mean something, with the robot one rung below it and the instances above.
		if(scene_.ground.valid and scene_.ground.polygon.size() >= 3)
		{
			constexpr graph3d::Rgb kWall{0.72f, 0.78f, 0.88f};
			const auto &poly = scene_.ground.polygon;
			const float gz   = scene_.ground.z;
			for(std::size_t i = 0; i < poly.size(); ++i)
			{
				const auto &p = poly[i];
				const auto &q = poly[(i + 1) % poly.size()];
				push_line(gl(p[0], p[1], gz), gl(q[0], q[1], gz), kWall, 0.75f);
			}
		}

		for(const auto &e : scene_.edges)
		{
			const auto *a = scene_.find(e.from);
			const auto *b = scene_.find(e.to);
			if(a == nullptr or b == nullptr)
				continue;

			// Ownership is provenance, not topology: showing all of it at once is unreadable, so it
			// stays off until an agent is selected (or 'O' forces the whole picture).
			if(e.kind == EdgeKind::Ownership and not show_all_ownership_
			   and (selected_id_ == 0 or (e.from != selected_id_ and e.to != selected_id_)))
				continue;

			const bool lit = selected_id_ != 0 and (e.from == selected_id_ or e.to == selected_id_);
			const float alpha = std::clamp(e.alpha * (lit ? 1.6f : 1.0f), 0.0f, 1.0f);

			switch(e.kind)
			{
				case EdgeKind::Member:
				case EdgeKind::Ownership: push_bundle(gl(a->pos), gl(b->pos), e.color, alpha); break;
				case EdgeKind::Has:       push_dashed(gl(a->pos), gl(b->pos), e.color, alpha); break;
				default:                  push_line(gl(a->pos), gl(b->pos), e.color, alpha);   break;
			}
		}

		// Drop lines to the room plane: without them a floating glyph's x/y is ambiguous under
		// perspective. Nodes already on (or below) that plane need none.
		constexpr graph3d::Rgb kDrop{0.55f, 0.60f, 0.70f};
		const float            drop_z = scene_.ground.z;
		for(const auto &n : scene_.nodes)
			if(n.placed and n.pos[2] > drop_z + 1e-3f)
				push_line(gl(n.pos), gl(n.pos[0], n.pos[1], drop_z), kDrop,
				          n.id == selected_id_ ? 0.45f : 0.12f);

		if(selected_id_ != 0)
			if(const auto *s = scene_.find(selected_id_); s != nullptr)
			{
				constexpr graph3d::Rgb kHi{1.0f, 0.95f, 0.55f};
				constexpr int kSeg = 28;
				const float rr = s->radius * 2.0f;
				for(int i = 0; i < kSeg; ++i)
				{
					const float a0 = 2.0f * std::numbers::pi_v<float> * static_cast<float>(i) / kSeg;
					const float a1 = 2.0f * std::numbers::pi_v<float> * static_cast<float>(i + 1) / kSeg;
					push_line(gl(s->pos[0] + rr * std::cos(a0), s->pos[1] + rr * std::sin(a0), s->pos[2]),
					          gl(s->pos[0] + rr * std::cos(a1), s->pos[1] + rr * std::sin(a1), s->pos[2]),
					          kHi, 0.9f);
				}
			}

		line_count_ = static_cast<int>(line_verts_.size());
	}

	// ── camera ────────────────────────────────────────────────────────────────────────────────
	void frame_camera()
	{
		if(scene_.nodes.empty())
			return;
		QVector3D lo{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
		QVector3D hi{std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()};
		const auto grow = [&](const QVector3D &v)
		{
			lo = {std::min(lo.x(), v.x()), std::min(lo.y(), v.y()), std::min(lo.z(), v.z())};
			hi = {std::max(hi.x(), v.x()), std::max(hi.y(), v.y()), std::max(hi.z(), v.z())};
		};
		for(const auto &n : scene_.nodes) grow(gl(n.pos));
		for(const auto &p : scene_.ground.polygon) grow(gl(p[0], p[1], scene_.ground.z));

		// radius_ always tracks the content (the projection far plane depends on it), but the pivot is
		// FROZEN once the user takes control: set_scene() runs several times a second, and re-centring
		// on a live bbox would slide the whole view sideways every time the robot moves.
		radius_ = std::max({0.5f, (hi.x() - lo.x()) * 0.5f, (hi.y() - lo.y()) * 0.5f, (hi.z() - lo.z()) * 0.5f});
		if(not user_interacted_)
		{
			center_   = (lo + hi) * 0.5f;
			cam_dist_ = radius_ * 3.2f;
		}
	}

	void reset_camera()
	{
		yaw_deg_ = 32.0f; pitch_deg_ = -22.0f; pan_x_ = 0.0f; pan_y_ = 0.0f;
		user_interacted_ = false;
		frame_camera();
	}

	void top_down()
	{
		yaw_deg_ = 0.0f; pitch_deg_ = -88.0f; pan_x_ = 0.0f; pan_y_ = 0.0f;
		user_interacted_ = true;
		cam_dist_ = radius_ * 3.2f;
	}

	// ── picking + labels ──────────────────────────────────────────────────────────────────────
	// Screen-space nearest hit. Cheaper and more forgiving than a colour-id pass, and small glyphs
	// (an affordance diamond) stay clickable because the tolerance is in pixels, not in metres.
	[[nodiscard]] bool project(const graph3d::Vec3 &p, QPointF &out, float &depth) const
	{
		const QVector4D clip = last_mvp_ * QVector4D(gl(p), 1.0f);
		if(clip.w() <= 1e-5f)
			return false;
		const float ndx = clip.x() / clip.w(), ndy = clip.y() / clip.w();
		out   = {(ndx * 0.5f + 0.5f) * width(), (1.0f - (ndy * 0.5f + 0.5f)) * height()};
		depth = clip.z() / clip.w();
		return true;
	}

	void pick_at(const QPoint &pos)
	{
		constexpr float kTolPx = 24.0f;
		float         best_d2 = kTolPx * kTolPx;
		float         best_depth = std::numeric_limits<float>::max();
		std::uint64_t hit = 0;
		std::string   hit_type;
		for(const auto &n : scene_.nodes)
		{
			QPointF sp; float depth = 0.0f;
			if(not project(n.pos, sp, depth))
				continue;
			const float dx = static_cast<float>(sp.x() - pos.x()), dy = static_cast<float>(sp.y() - pos.y());
			const float d2 = dx * dx + dy * dy;
			if(d2 > kTolPx * kTolPx)
				continue;
			// Nearest in screen space, ties (overlapping glyphs — an affordance sits directly above
			// its object) broken by whichever is closer to the camera.
			const bool tie = std::abs(d2 - best_d2) < 4.0f;
			if(d2 < best_d2 - 4.0f or (tie and depth < best_depth))
			{
				best_d2 = std::min(best_d2, d2); best_depth = depth; hit = n.id; hit_type = n.type;
			}
		}
		selected_id_ = hit;
		rebuild_lines();
		upload();
		update();
		if(hit != 0 and pick_cb_)
			pick_cb_(hit, hit_type);
	}

	// A legend is mandatory, not decorative: the class palette sits in the CVD 6–8 band and three of
	// its slots are just under 3:1 against the surface, both of which are admissible only because
	// identity is also carried in writing. Lists exactly the classes present, so it shrinks to the
	// scene rather than advertising classes that are not there.
	void draw_legend(QPainter &painter)
	{
		if(not show_labels_)
			return;
		std::vector<std::pair<std::string, graph3d::Rgb>> items;
		for(const auto &n : scene_.nodes)
		{
			if(n.kind != graph3d::Kind::Instance or n.cls.empty())
				continue;
			if(std::none_of(items.begin(), items.end(), [&](const auto &i) { return i.first == n.cls; }))
				items.emplace_back(n.cls, n.color);
		}
		if(items.empty())
			return;
		std::sort(items.begin(), items.end(), [](const auto &a, const auto &b) { return a.first < b.first; });

		const int sw = 10, lh = 15, pad = 7;
		int       tw = 0;
		const QFontMetrics fm = painter.fontMetrics();
		for(const auto &[name, c] : items)
			tw = std::max(tw, fm.horizontalAdvance(QString::fromStdString(name)));
		const int w = pad * 2 + sw + 6 + tw;
		const int h = pad * 2 + lh * static_cast<int>(items.size());
		const int x = width() - w - 12, y = 34;

		painter.fillRect(QRect(x, y, w, h), QColor(18, 20, 26, 190));
		painter.setPen(QColor(70, 78, 92));
		painter.drawRect(QRect(x, y, w, h));
		int row = 0;
		for(const auto &[name, c] : items)
		{
			const QColor col(static_cast<int>(std::clamp(c[0], 0.f, 1.f) * 255),
			                 static_cast<int>(std::clamp(c[1], 0.f, 1.f) * 255),
			                 static_cast<int>(std::clamp(c[2], 0.f, 1.f) * 255));
			const int ry = y + pad + row * lh;
			painter.fillRect(QRect(x + pad, ry + 3, sw, sw), col);
			// Text wears text ink, never the series colour — the swatch beside it carries identity.
			painter.setPen(QColor(198, 205, 218));
			painter.drawText(QPoint(x + pad + sw + 6, ry + lh - 4), QString::fromStdString(name));
			++row;
		}
	}

	void draw_overlay()
	{
		QPainter painter(this);
		painter.setRenderHint(QPainter::TextAntialiasing, true);

		if(show_labels_)
		{
			QFont f = painter.font();
			f.setPointSizeF(std::max(7.0, f.pointSizeF() - 1.0));
			painter.setFont(f);
			for(const auto &n : scene_.nodes)
			{
				QPointF sp; float depth = 0.0f;
				if(not project(n.pos, sp, depth) or depth < -1.0f or depth > 1.0f)
					continue;
				if(sp.x() < -60 or sp.y() < -20 or sp.x() > width() + 60 or sp.y() > height() + 20)
					continue;
				QString text = QString::fromStdString(n.name);
				if(not n.sublabel.empty())
					text += "  " + QString::fromStdString(n.sublabel);
				if(not n.placed)
					text += "  ·no-RT";
				const bool hi = (n.id == selected_id_);
				painter.setPen(hi ? QColor(255, 240, 150)
				                  : QColor(static_cast<int>(std::clamp(n.color[0], 0.f, 1.f) * 235),
				                           static_cast<int>(std::clamp(n.color[1], 0.f, 1.f) * 235),
				                           static_cast<int>(std::clamp(n.color[2], 0.f, 1.f) * 235),
				                           n.dimmed ? 130 : 215));
				painter.drawText(QPointF(sp.x() + 8.0, sp.y() - 6.0), text);
			}
		}

		// Level name plates, anchored to each plane so the ladder is named, not inferred.
		painter.setPen(QColor(150, 165, 185, 190));
		float minx = 0.0f, miny = 0.0f;
		bool  any = false;
		for(const auto &n : scene_.nodes)
		{
			minx = any ? std::min(minx, n.pos[0]) : n.pos[0];
			miny = any ? std::min(miny, n.pos[1]) : n.pos[1];
			any  = true;
		}
		for(std::size_t s = 0; any and s < scene_.level_z.size(); ++s)
		{
			QPointF sp; float depth = 0.0f;
			if(project({minx - 0.6f, miny - 0.6f, scene_.level_z[s]}, sp, depth) and depth <= 1.0f)
				painter.drawText(sp, QString::fromStdString(scene_.level_names[s]));
		}

		draw_legend(painter);

		painter.setPen(QColor(225, 230, 240));
		QString hud = QString("nodes %1   edges %2").arg(scene_.nodes.size()).arg(scene_.edges.size());
		if(selected_id_ != 0)
			if(const auto *s = scene_.find(selected_id_); s != nullptr)
				hud += QString("   ·   selected: %1 [%2%3]")
				           .arg(QString::fromStdString(s->name), QString::fromStdString(s->type),
				                s->subtype.empty() ? QString() : "/" + QString::fromStdString(s->subtype));
		painter.drawText(QRect(10, 6, width() - 20, 20), Qt::AlignLeft | Qt::AlignTop, hud);
		painter.setPen(QColor(150, 160, 175));
		painter.drawText(QRect(10, height() - 24, width() - 20, 20), Qt::AlignLeft | Qt::AlignTop,
		                 QStringLiteral("drag=rotate  right/mid=pan  wheel=zoom  click=select  "
		                                "O=all ownership  L=labels  G=strata  T=top  R=reset  Esc=clear"));
	}

	void upload()
	{
		if(not context() or not solid_vbo_.isCreated() or not line_vbo_.isCreated())
			return;
		makeCurrent();
		solid_vbo_.bind();
		solid_vbo_.allocate(solid_verts_.data(), static_cast<int>(solid_verts_.size() * sizeof(SolidVertex)));
		solid_vbo_.release();
		line_vbo_.bind();
		line_vbo_.allocate(line_verts_.data(), static_cast<int>(line_verts_.size() * sizeof(LineVertex)));
		line_vbo_.release();
		doneCurrent();
	}

	graph3d::Scene                                 scene_;
	std::unordered_map<std::string, CachedMesh>    mesh_cache_;
	std::vector<SolidVertex> solid_verts_;
	std::vector<LineVertex>  line_verts_;
	int                      solid_count_ = 0;
	int                      line_count_  = 0;

	std::uint64_t selected_id_        = 0;
	bool          show_all_ownership_ = false;
	bool          show_labels_        = true;
	bool          show_strata_        = true;

	QVector3D center_{0, 0, 0};
	float     radius_ = 3.0f;
	float     cam_dist_ = 10.0f, yaw_deg_ = 32.0f, pitch_deg_ = -22.0f, pan_x_ = 0.0f, pan_y_ = 0.0f;
	bool      rotating_ = false, panning_ = false, user_interacted_ = false;
	QPoint    last_, press_;
	QMatrix4x4 last_mvp_;

	std::function<void(std::uint64_t, const std::string &)> pick_cb_;

	QOpenGLShaderProgram     solid_, flat_;
	QOpenGLVertexArrayObject vao_;
	QOpenGLBuffer            solid_vbo_{QOpenGLBuffer::VertexBuffer};
	QOpenGLBuffer            line_vbo_{QOpenGLBuffer::VertexBuffer};
	int u_solid_mvp_ = -1, u_solid_light_ = -1, u_flat_mvp_ = -1;
};

}   // namespace rc::viewers

#endif   // RC_COMMON_GL_GRAPH3D_VIEWER_H
