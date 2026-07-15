/*
 *  Graph-attribute "View data" viewers for robot_concept.
 *
 *  Companion to media_stream_viewers.h. Those show media-plane streams; these show data that lives
 *  as ATTRIBUTES on the DSR node itself. dsr_gui forwards view_data_signal for any node type without
 *  a built-in widget (e.g. "room"), so robot_concept opens one of these. Data source is the graph
 *  (get_attrib_by_name), and refresh is arrival-driven for free: we connect to the DSR
 *  update_node_(attr_)signal — QUEUED, never Direct (they fire on the FastDDS reader thread) — and
 *  re-read on each update. Reads run on the main thread, where the graph API is safe.
 *
 *  Non-Q_OBJECT: the signal is bound via a lambda with `this` as the context object, so no MOC.
 */
#ifndef ROBOT_CONCEPT_GRAPH_ATTR_VIEWERS_H
#define ROBOT_CONCEPT_GRAPH_ATTR_VIEWERS_H

#include <QString>

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/api/dsr_eigen_defs.h>

#include <QDebug>

#include "../../common/viewers/polygon_viewer.h"
#include "../../common/viewers/gl_mesh_viewer.h"
#include "../../common/viewers/gl_skeleton_viewer.h"
#include "../../common/viewers/gl_grid_field_viewer.h"
#include "../../common/viewers/semantic_map_viewer.h"
#include "../../common/obj/obj_loader.h"

namespace rc::viewers
{

// Draws a room node's delimiting_polygon_x/y as a closed 2D polygon, refreshed whenever the room
// node's attributes change.
class RoomPolygonViewer : public PolygonViewer
{
public:
	RoomPolygonViewer(std::shared_ptr<DSR::DSRGraph> graph, std::uint64_t node_id,
	                  QString title, QWidget *parent = nullptr)
		: PolygonViewer(parent), g_(std::move(graph)), id_(node_id)
	{
		setWindowTitle(std::move(title));
		setAttribute(Qt::WA_DeleteOnClose);
		// Redraw on any change to THIS node. QueuedConnection is mandatory: these signals are emitted
		// from the FastDDS reader threads and the slot must run on the main thread (CLAUDE.md).
		connect(g_.get(), &DSR::DSRGraph::update_node_attr_signal, this,
		        [this](std::uint64_t id, const std::vector<std::string> &) { if(id == id_) refresh(); },
		        Qt::QueuedConnection);
		connect(g_.get(), &DSR::DSRGraph::update_node_signal, this,
		        [this](std::uint64_t id, const std::string &) { if(id == id_) refresh(); },
		        Qt::QueuedConnection);
		refresh();   // draw whatever is already there
	}

private:
	void refresh()   // main thread
	{
		const auto n = g_->get_node(id_);
		if(not n.has_value())
			return;
		const auto px = g_->get_attrib_by_name<delimiting_polygon_x_att>(n.value());
		const auto py = g_->get_attrib_by_name<delimiting_polygon_y_att>(n.value());
		if(px.has_value() and py.has_value())
			set_polygon(px->get(), py->get());
	}

	std::shared_ptr<DSR::DSRGraph> g_;
	std::uint64_t id_;
};

// Draws a 3D OpenGL view of the .obj mesh named in a node's `path` attribute (e.g. the robot node).
// The mesh is static, so it is (re)loaded only when the path attribute changes — node pose updates
// fire the signal often but don't reparse the file.
class RobotMeshViewer : public GLMeshViewer
{
public:
	RobotMeshViewer(std::shared_ptr<DSR::DSRGraph> graph, std::uint64_t node_id,
	                QString title, QWidget *parent = nullptr)
		: GLMeshViewer(parent), g_(std::move(graph)), id_(node_id)
	{
		setWindowTitle(std::move(title));
		setAttribute(Qt::WA_DeleteOnClose);
		connect(g_.get(), &DSR::DSRGraph::update_node_attr_signal, this,
		        [this](std::uint64_t id, const std::vector<std::string> &) { if(id == id_) refresh(); },
		        Qt::QueuedConnection);
		connect(g_.get(), &DSR::DSRGraph::update_node_signal, this,
		        [this](std::uint64_t id, const std::string &) { if(id == id_) refresh(); },
		        Qt::QueuedConnection);
		refresh();
	}

private:
	void refresh()   // main thread
	{
		const auto n = g_->get_node(id_);
		if(not n.has_value())
			return;
		const auto p = g_->get_attrib_by_name<path_att>(n.value());
		if(not p.has_value())
			return;
		const std::string path = p->get();
		if(path.empty() or path == loaded_path_)   // only reload when the path actually changes
			return;

		const auto resolved = rc::obj::resolve_robot_mesh_path(path);
		if(not resolved.has_value())
		{
			qWarning() << "[view-data] robot mesh not found:" << QString::fromStdString(path);
			return;
		}
		const auto mesh = rc::obj::load_obj_mesh_data(resolved.value());
		if(not mesh.has_value())
		{
			qWarning() << "[view-data] failed to load robot mesh:" << QString::fromStdString(resolved->string());
			return;
		}
		loaded_path_ = path;
		qInfo() << "[view-data] robot mesh loaded" << QString::fromStdString(resolved->string())
		        << "triangles" << mesh->triangles.size() / 3;
		set_mesh(mesh->triangles);
	}

	std::shared_ptr<DSR::DSRGraph> g_;
	std::uint64_t id_;
	std::string loaded_path_;
};

// Colourises the voxelizer's 'semantic' (type semantic_grid) node: a dense ADE20K-150 per-pixel
// class-id map stored as node attributes (semantic_labels row-major bytes + semantic_width/height).
// Refreshed whenever the node is (re)published — low frequency, ~2 Hz. Hover shows the class name.
class SemanticGridViewer : public SemanticMapViewer
{
public:
	SemanticGridViewer(std::shared_ptr<DSR::DSRGraph> graph, std::uint64_t node_id,
	                   QString title, QWidget *parent = nullptr)
		: SemanticMapViewer(parent), g_(std::move(graph)), id_(node_id)
	{
		setWindowTitle(std::move(title));
		setAttribute(Qt::WA_DeleteOnClose);
		// Redraw on any change to THIS node. QueuedConnection is mandatory: these signals are emitted
		// from the FastDDS reader threads and the slot must run on the main thread (CLAUDE.md).
		connect(g_.get(), &DSR::DSRGraph::update_node_attr_signal, this,
		        [this](std::uint64_t id, const std::vector<std::string> &) { if(id == id_) refresh(); },
		        Qt::QueuedConnection);
		connect(g_.get(), &DSR::DSRGraph::update_node_signal, this,
		        [this](std::uint64_t id, const std::string &) { if(id == id_) refresh(); },
		        Qt::QueuedConnection);
		refresh();   // draw whatever is already there
	}

private:
	void refresh()   // main thread
	{
		const auto n = g_->get_node(id_);
		if(not n.has_value())
			return;
		const auto labels = g_->get_attrib_by_name<semantic_labels_att>(n.value());
		const auto w = g_->get_attrib_by_name<semantic_width_att>(n.value());
		const auto h = g_->get_attrib_by_name<semantic_height_att>(n.value());
		if(labels.has_value() and w.has_value() and h.has_value())
			set_label_map(labels->get(), w.value(), h.value());
	}

	std::shared_ptr<DSR::DSRGraph> g_;
	std::uint64_t id_;
};

// 3D view of the voxelizer's 'skeleton' node: BODY_18 human poses published low-freq as raw
// attributes (skeleton_count + skeleton_kp_xyz, count*18*3 floats in the ZED camera frame, NaN =
// missing). Read via the low-level attrs() map + Attribute::float_vec()/dec() — the same path the
// human_concept consumer uses (these attributes are runtime-typed, not type-attributed). Refreshed
// whenever the node is (re)published.
class SkeletonNodeViewer : public GLSkeletonViewer
{
public:
	SkeletonNodeViewer(std::shared_ptr<DSR::DSRGraph> graph, std::uint64_t node_id,
	                   QString title, QWidget *parent = nullptr)
		: GLSkeletonViewer(parent), g_(std::move(graph)), id_(node_id)
	{
		setWindowTitle(std::move(title));
		setAttribute(Qt::WA_DeleteOnClose);
		// QueuedConnection is mandatory: these signals fire on the FastDDS reader threads; the slot
		// must run on the main thread (CLAUDE.md).
		connect(g_.get(), &DSR::DSRGraph::update_node_attr_signal, this,
		        [this](std::uint64_t id, const std::vector<std::string> &) { if(id == id_) refresh(); },
		        Qt::QueuedConnection);
		connect(g_.get(), &DSR::DSRGraph::update_node_signal, this,
		        [this](std::uint64_t id, const std::string &) { if(id == id_) refresh(); },
		        Qt::QueuedConnection);
		refresh();
	}

private:
	void refresh()   // main thread
	{
		const auto n = g_->get_node(id_);
		if(not n.has_value())
			return;
		const auto &attrs = n->attrs();
		const auto find = [&](const std::string &key) -> const DSR::Attribute *
		{
			const auto it = attrs.find(key);
			return (it != attrs.end()) ? &it->second : nullptr;
		};
		const DSR::Attribute *count_attr = find("skeleton_count");
		const DSR::Attribute *xyz_attr   = find("skeleton_kp_xyz");
		if(not count_attr or not xyz_attr)
			return;

		const int count = std::max(0, count_attr->dec());
		const auto &xyz  = xyz_attr->float_vec();   // count*18*3 flat, ZED camera frame
		constexpr int K = GLSkeletonViewer::K;      // 18
		if(static_cast<int>(xyz.size()) < count * K * 3)
			return;

		// Slice the flat buffer into one 18*3 vector per body for the viewer.
		std::vector<std::vector<float>> bodies;
		bodies.reserve(static_cast<std::size_t>(count));
		for(int b = 0; b < count; ++b)
		{
			const std::size_t base = static_cast<std::size_t>(b) * K * 3;
			bodies.emplace_back(xyz.begin() + static_cast<std::ptrdiff_t>(base),
			                    xyz.begin() + static_cast<std::ptrdiff_t>(base + K * 3));
		}
		set_skeletons(bodies);
	}

	std::shared_ptr<DSR::DSRGraph> g_;
	std::uint64_t id_;
};

// 3D view of residual_concept's 'grid' node: the Beta occupancy belief field (dense grid_occupancy_prob
// + grid_occupancy_var indexed by grid_field_meta=[xmin,ymin,cell,w,h]) drawn as elevated risk columns,
// plus the discrete grid_occupied_cells / grid_border_cells floor layers. Mirrors the voxelizer's
// in-process residual-field display. Refreshed whenever the node is (re)published (~2 Hz).
class GridNodeViewer : public GLGridFieldViewer
{
public:
	GridNodeViewer(std::shared_ptr<DSR::DSRGraph> graph, std::uint64_t node_id,
	               QString title, QWidget *parent = nullptr)
		: GLGridFieldViewer(parent), g_(std::move(graph)), id_(node_id)
	{
		setWindowTitle(std::move(title));
		setAttribute(Qt::WA_DeleteOnClose);
		// QueuedConnection is mandatory: these signals fire on the FastDDS reader threads; the slot
		// must run on the main thread (CLAUDE.md).
		connect(g_.get(), &DSR::DSRGraph::update_node_attr_signal, this,
		        [this](std::uint64_t id, const std::vector<std::string> &) { if(id == id_) refresh(); },
		        Qt::QueuedConnection);
		connect(g_.get(), &DSR::DSRGraph::update_node_signal, this,
		        [this](std::uint64_t id, const std::string &) { if(id == id_) refresh(); },
		        Qt::QueuedConnection);
		refresh();
	}

private:
	// Flat [x0,y0,z0, x1,y1,z1, …] → QVector3D room-frame points.
	static std::vector<QVector3D> read_pts(const std::vector<float> &flat)
	{
		std::vector<QVector3D> v;
		const std::size_t n = flat.size() / 3;
		v.reserve(n);
		for(std::size_t i = 0; i < n; ++i)
			v.emplace_back(flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]);
		return v;
	}

	void refresh()   // main thread
	{
		const auto n = g_->get_node(id_);
		if(not n.has_value())
			return;

		std::vector<QVector3D> occupied, border;
		if(const auto o = g_->get_attrib_by_name<grid_occupied_cells_att>(n.value()); o.has_value())
			occupied = read_pts(o->get());
		if(const auto o = g_->get_attrib_by_name<grid_border_cells_att>(n.value()); o.has_value())
			border = read_pts(o->get());

		// Robot mesh (shadow.obj) posed into the room frame, so the scene reads with the robot in place.
		const std::vector<QVector3D> robot = robot_room_triangles();

		// The occupied cells now carry the REAL obstacle top height in z (producer publishes it); the
		// viewer raises a surface to that height. grid_field_meta = [xmin, ymin, cell, w, h] only frames
		// the coarse lattice — no dense field needed.
		const auto ma = g_->get_attrib_by_name<grid_field_meta_att>(n.value());
		if(ma.has_value() and ma->get().size() >= 5)
		{
			const auto &M = ma->get();
			set_data(occupied, border, M[0], M[1], M[2],
			         static_cast<int>(M[3]), static_cast<int>(M[4]), robot);
		}
		else
			set_data(occupied, border, 0.f, 0.f, 0.f, 0, 0, robot);   // no extent yet → cells + robot only
	}

	// Load + recentre the robot mesh once (xy-centre, z-min → base on the floor), mirroring the
	// voxelizer viewer. Empty on failure; retried never (the file is static).
	void load_robot_mesh_once()
	{
		if(robot_loaded_)
			return;
		robot_loaded_ = true;
		const auto p = rc::obj::resolve_robot_mesh_path("meshes/shadow.obj");
		if(not p.has_value())
		{
			qWarning() << "[grid-view] robot mesh meshes/shadow.obj not found";
			return;
		}
		const auto mesh = rc::obj::load_obj_mesh_data(p.value());
		if(not mesh.has_value())
			return;
		const QVector3D ctr(0.5f * (mesh->bb_min.x() + mesh->bb_max.x()),
		                    0.5f * (mesh->bb_min.y() + mesh->bb_max.y()),
		                    mesh->bb_min.z());
		robot_local_.reserve(mesh->triangles.size());
		for(const auto &v : mesh->triangles)
			robot_local_.emplace_back(v.x() - ctr.x(), v.y() - ctr.y(), v.z() - ctr.z());
	}

	// Robot mesh expressed in the ROOM frame via the room←robot SE(2) pose (x, y, yaw), same as the
	// voxelizer draw. Empty if the mesh or the transform isn't available.
	std::vector<QVector3D> robot_room_triangles()   // main thread
	{
		load_robot_mesh_once();
		if(robot_local_.empty())
			return {};
		std::string room_name, robot_name;
		if(const auto r = g_->get_nodes_by_type("room");  not r.empty()) room_name  = r.front().name();
		if(const auto r = g_->get_nodes_by_type("robot"); not r.empty()) robot_name = r.front().name();
		if(room_name.empty() or robot_name.empty())
			return {};
		if(not inner_)
			inner_ = g_->get_inner_eigen_api();
		const auto T = inner_->get_transformation_matrix(room_name, robot_name, 0);   // ts==0: main thread only
		if(not T.has_value())
			return {};
		const auto &M = T.value();
		const float x = static_cast<float>(M.translation().x());
		const float y = static_cast<float>(M.translation().y());
		const float yaw = static_cast<float>(std::atan2(M.linear()(1, 0), M.linear()(0, 0)));
		const float c = std::cos(yaw), s = std::sin(yaw);
		std::vector<QVector3D> out;
		out.reserve(robot_local_.size());
		for(const auto &l : robot_local_)
			out.emplace_back(x + c * l.x() - s * l.y(), y + s * l.x() + c * l.y(), l.z() + 0.01f);
		return out;
	}

	std::shared_ptr<DSR::DSRGraph>          g_;
	std::uint64_t                           id_;
	std::unique_ptr<DSR::InnerEigenAPI>     inner_;         // room←robot pose (created lazily, main thread)
	std::vector<QVector3D>                  robot_local_;   // recentred mesh, robot frame
	bool                                    robot_loaded_ = false;
};

}   // namespace rc::viewers

#endif   // ROBOT_CONCEPT_GRAPH_ATTR_VIEWERS_H
