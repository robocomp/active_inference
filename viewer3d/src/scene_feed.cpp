#include "scene_feed.h"

#include "voxel_opengl_viewer.h"

#include "../../common/media_transport/media_transport.h"
#include "../../common/media_transport/lidar_plane_reader.h"

#include <dsr/api/dsr_camera_api.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <print>
#include <sstream>
#include <string_view>

#include <QVector3D>

using rc::SceneFeed;

namespace
{
// Schema migration: every concept publishes its object as a generic type "object" node with the class in
// the object_subtype string attr (table/chair/bottle/cabinet/refrigerator); node NAME prefixes are unchanged
// (table_*, chair_*, bottle_*, cabinet_*, refrigerator_*). refrigerator_concept is the end-to-end example.
// Derive the class from object_subtype when it names a known class, else from the node NAME prefix. Note that
// for a table, object_subtype instead carries the SHAPE (round/square) — not a class — so it is only accepted
// when it names a class and otherwise falls through to the name prefix. Returns "" for a non-furniture object.
std::string object_class_of(DSR::DSRGraph& g, const DSR::Node& node)
{
    static constexpr std::array<std::string_view, 6> kClasses{"table", "chair", "bottle", "cabinet", "refrigerator", "door"};
    if (const auto s = g.get_attrib_by_name<object_subtype_att>(node); s.has_value())
        if (const std::string& sv = s.value();
            std::find(kClasses.begin(), kClasses.end(), sv) != kClasses.end())
            return sv;
    for (const auto cls : kClasses)
        if (node.name().rfind(cls, 0) == 0)   // NAME-prefix fallback (unchanged across the migration)
            return std::string(cls);
    return {};
}
}   // namespace

void SceneFeed::forward_extrapolate_room_T_robot(Mat::RTMat& room_T_robot, const std::string& room_name,
                                                 const std::string& robot_name, std::uint64_t timestamp_ms) const
{
    if (graph_ == nullptr || graph_->get_rt_api() == nullptr)
        return;
    const auto robot_node = graph_->get_node(robot_name);
    const auto room_node  = graph_->get_node(room_name);
    if (!robot_node.has_value() || !room_node.has_value())
        return;
    const auto edge = graph_->get_rt_api()->get_edge_RT(robot_node.value(), room_node.value().id());
    if (!edge.has_value())
        return;
    const auto ts = graph_->get_attrib_by_name<rt_timestamps_att>(edge.value());
    const auto tv = graph_->get_attrib_by_name<rt_translation_velocity_att>(edge.value());
    const auto rv = graph_->get_attrib_by_name<rt_rotation_euler_xyz_velocity_att>(edge.value());
    if (!ts.has_value() || !tv.has_value() || tv->get().size() < 2 || !rv.has_value() || rv->get().size() < 3)
        return;

    std::uint64_t newest = 0;
    for (const auto t : ts->get())
        newest = std::max(newest, t);
    if (newest == 0 || timestamp_ms <= newest)
        return;

    float dt = static_cast<float>(timestamp_ms - newest) * 1e-3f;
    dt = std::min(dt, mask_pose_extrap_max_dt_s_);
    const double adv = tv->get()[0], side = tv->get()[1], rot = rv->get()[2];
    const Eigen::Matrix3d R = room_T_robot.linear();
    const double th    = std::atan2(R(1, 0), R(0, 0));
    const double raw_x = room_T_robot.translation().x(), raw_y = room_T_robot.translation().y();
    const double dth = rot * dt;
    const double thm = th + 0.5 * dth;   // midpoint integration
    const double dx = (adv * std::cos(thm) - side * std::sin(thm)) * dt;
    const double dy = (adv * std::sin(thm) + side * std::cos(thm)) * dt;
    room_T_robot.translation().x() += dx;
    room_T_robot.translation().y() += dy;
    room_T_robot.linear() = (Eigen::AngleAxisd(dth, Eigen::Vector3d::UnitZ()).toRotationMatrix() * R);

}

std::optional<Mat::RTMat> SceneFeed::room_T_zed_extrapolated(DSR::InnerEigenAPI* eigen,
                                                                  const std::string& room_name,
                                                                  const std::string& robot_name,
                                                                  std::uint64_t stamp) const
{
    if (eigen == nullptr || room_name.empty() || robot_name.empty())
        return std::nullopt;

    // room←robot at the capture stamp (ts!=0 → no InnerEigenAPI cache), then forward-extrapolate to beat
    // the RT lag — the same correction the voxel path applies.
    auto room_T_robot = eigen->get_transformation_matrix(room_name, robot_name, stamp);
    if (!room_T_robot.has_value())
        return std::nullopt;
    if (mask_pose_extrapolate_)
    {
        forward_extrapolate_room_T_robot(room_T_robot.value(), room_name, robot_name, stamp);
    }

    // robot→zed is the rigid, static camera extrinsic → query "latest" (ts==0) on the CALLER's own instance,
    // so the ts==0 cache stays per-thread; the extrinsic never changes so a stale cache is harmless.
    const auto robot_T_zed = eigen->get_transformation_matrix(robot_name, "zed", 0);
    if (!robot_T_zed.has_value())
        return std::nullopt;

    return room_T_robot.value() * robot_T_zed.value();
}

std::optional<SceneFeed::RoomPolygonData> SceneFeed::get_room_polygon_from_graph() const
{
    if (!graph_)
        return std::nullopt;

    // No mutex: unlike the retina's SceneProcessor this class is main-thread only (the refresh is
    // a QTimer), so the memoized names need no guard.
    std::string room_name_snapshot = room_node_name_;

    if (room_name_snapshot.empty())
    {
        if (const auto room_nodes = graph_->get_nodes_by_type("room"); !room_nodes.empty())
            room_name_snapshot = room_nodes.front().name();
    }

    if (room_name_snapshot.empty())
        return std::nullopt;

    auto room_node = graph_->get_node(room_name_snapshot);
    if (!room_node.has_value())
        return std::nullopt;

    auto polygon_x_opt = graph_->get_attrib_by_name<delimiting_polygon_x_att>(room_node.value());
    auto polygon_y_opt = graph_->get_attrib_by_name<delimiting_polygon_y_att>(room_node.value());
    if (!polygon_x_opt.has_value() || !polygon_y_opt.has_value())
        return std::nullopt;

    const auto& polygon_x_src = polygon_x_opt.value().get();
    const auto& polygon_y_src = polygon_y_opt.value().get();
    if (polygon_x_src.empty() || polygon_y_src.empty())
        return std::nullopt;

    RoomPolygonData data;
    data.room_name = std::move(room_name_snapshot);
    data.polygon_x.assign(polygon_x_src.begin(), polygon_x_src.end());
    data.polygon_y.assign(polygon_y_src.begin(), polygon_y_src.end());
    if (auto height_opt = graph_->get_attrib_by_name<room_height_att>(room_node.value()); height_opt.has_value())
        data.room_height = height_opt.value();

    return data;
}

std::optional<GraphObjectBox> SceneFeed::build_graph_object_box(const DSR::Node& node,
                                                                     const std::string& room_name,
                                                                     std::uint64_t timestamp_ms) const
{
    if (!graph_ || inner_eigen_api_ == nullptr || room_name.empty())
        return std::nullopt;

    const auto width_opt = graph_->get_attrib_by_name<width_m_att>(node);
    const auto depth_opt = graph_->get_attrib_by_name<depth_m_att>(node);
    const auto height_opt = graph_->get_attrib_by_name<height_m_att>(node);
    if (!width_opt.has_value() || !depth_opt.has_value() || !height_opt.has_value())
    {
        std::println("[build_graph_object_box] node='{}' type='{}': missing dimensions (w={} d={} h={})",
                     node.name(), node.type(),
                     width_opt.has_value(), depth_opt.has_value(), height_opt.has_value());
        return std::nullopt;
    }

    const float width = width_opt.value();
    const float depth = depth_opt.value();
    const float height = height_opt.value();
    if (width <= 0.f || depth <= 0.f || height <= 0.f)
    {
        std::println("[build_graph_object_box] node='{}': non-positive dims w={} d={} h={}",
                     node.name(), width, depth, height);
        return std::nullopt;
    }

    const auto time_query = transforms_interpolate_rt_
        ? DSR::RT_API::TimeQuery::Interpolated
        : DSR::RT_API::TimeQuery::Nearest;
    const auto room_T_object = inner_eigen_api_->get_transformation_matrix(room_name,
                                                                           node.name(),
                                                                           timestamp_ms,
                                                                           "RT",
                                                                           time_query);
    if (!room_T_object.has_value())
    {
        std::println("[build_graph_object_box] node='{}': no RT transform from room '{}'",
                     node.name(), room_name);
        return std::nullopt;
    }

    const float half_width = width * 0.5f;
    const float half_depth = depth * 0.5f;
    const float half_height = height * 0.5f;

    // Furniture like the table stands ON the floor: its node origin is the base, so the box
    // must extend upward (z in [origin, origin+height]). Free objects (e.g. a fitted bottle
    // cylinder) are center-anchored, so they keep z in [origin-h/2, origin+h/2].
    // Base-origin convention (RT origin at the carcass base, box extends [origin, origin+height]):
    // tables stand on the floor; cabinet_concept runs pin their RT origin at z0 (the carcass base — 0 for a
    // base unit, ~1.45 for a wall unit), so they use the same upward-extending convention, NOT centre-anchor.
    // `object`-type furniture (refrigerator_concept and future generic floor objects) also writes its RT
    // origin at the floor base (z=0) — same upward-extending convention as tables/cabinets. Without this the
    // box would centre on the RT origin and sink half its height below the floor.
    // A metaconcept's RT origin is on the floor too (ring_metaconcept publishes z=0 with a 0.02 m
    // nominal height), so it uses the same upward-extending convention — its outline lies ON the
    // floor rather than straddling it.
    const bool stands_on_floor = (node.type() == "table") or (node.name().rfind("table", 0) == 0)
                              or (node.name().rfind("cabinet_", 0) == 0)
                              or (node.type() == "object") or (node.type() == "metaconcept");
    const float z_lo = stands_on_floor ? 0.f : -half_height;
    const float z_hi = stands_on_floor ? height : half_height;

    const std::array<Eigen::Vector3d, 8> local_corners = {
        Eigen::Vector3d{-half_width, -half_depth, z_lo},
        Eigen::Vector3d{ half_width, -half_depth, z_lo},
        Eigen::Vector3d{ half_width,  half_depth, z_lo},
        Eigen::Vector3d{-half_width,  half_depth, z_lo},
        Eigen::Vector3d{-half_width, -half_depth, z_hi},
        Eigen::Vector3d{ half_width, -half_depth, z_hi},
        Eigen::Vector3d{ half_width,  half_depth, z_hi},
        Eigen::Vector3d{-half_width,  half_depth, z_hi}
    };

    Eigen::Vector3f min_corner = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
    Eigen::Vector3f max_corner = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());

    for (const auto& local_corner : local_corners)
    {
        const Eigen::Vector3d room_corner = room_T_object->linear() * local_corner + room_T_object->translation();
        min_corner.x() = std::min(min_corner.x(), static_cast<float>(room_corner.x()));
        min_corner.y() = std::min(min_corner.y(), static_cast<float>(room_corner.y()));
        min_corner.z() = std::min(min_corner.z(), static_cast<float>(room_corner.z()));
        max_corner.x() = std::max(max_corner.x(), static_cast<float>(room_corner.x()));
        max_corner.y() = std::max(max_corner.y(), static_cast<float>(room_corner.y()));
        max_corner.z() = std::max(max_corner.z(), static_cast<float>(room_corner.z()));
    }

    std::string category = node.name();
    if (const auto it = node.attrs().find("semantic_class"); it != node.attrs().end() && it->second.selected() == 0)
        category = it->second.str();

    // Schema migration: every concept publishes its object as a generic type "object" node with the class in
    // the object_subtype attr (table/chair/bottle/cabinet/refrigerator), NAME prefixes unchanged. Derive the
    // display category from that class (object_subtype, else the NAME prefix — see object_class_of). Tables
    // become "model_table" so graph/model tables stay visually distinct from YOLO-derived table tracks / mask
    // points. bottle→hot-magenta box, cabinet→off-white carcass, refrigerator→scaled fridge template, etc.
    if (node.type() == "object")
    {
        if (const std::string cls = object_class_of(*graph_, node); cls == "table")
            category = "model_table";
        else if (not cls.empty())
            category = cls;
    }
    // residual_concept obstacles (type "obstacle", named residual_*) → their own category/colour.
    if (node.type() == "obstacle")
        category = "obstacle";
    // Level-2 arrangements get ONE category regardless of schema (ring/row/grid): the viewer draws
    // them as an abstract outline, so there is nothing per-class to colour. The schema itself is in
    // object_subtype ("dining_set") and the node NAME, which is what the label shows.
    std::string rig_schema;
    if (node.type() == "metaconcept")
    {
        category = "metaconcept";
        // The SHAPE it declares — the viewer draws a ring or a rectangle off THIS, not off the class,
        // so a new level-2 schema renders correctly without the viewer learning its concept name.
        if (const auto sch = graph_->get_attrib_by_name<rig_schema_att>(node); sch.has_value())
            rig_schema = sch.value();
    }

    // Concept-published display mesh + texture (relative asset paths). The viewer loads & renders these,
    // scaled to this box — the agent owns the appearance, the viewer stays type-agnostic.
    std::string mesh_path, mesh_texture_path;
    if (const auto mp = graph_->get_attrib_by_name<mesh_path_att>(node); mp.has_value())
        mesh_path = mp.value();
    if (const auto mt = graph_->get_attrib_by_name<mesh_texture_path_att>(node); mt.has_value())
        mesh_texture_path = mt.value();

    // Inferred albedo chromaticity (common/appearance_belief → the agent's mesh_color_rgb). Absent for any
    // agent that does not run an appearance belief, in which case the asset renders with its authored colours.
    Eigen::Vector3f mesh_color = Eigen::Vector3f::Zero();
    bool            has_mesh_color = false;
    if (const auto mc = graph_->get_attrib_by_name<mesh_color_rgb_att>(node);
        mc.has_value() and mc.value().size() >= 3)
    {
        mesh_color = Eigen::Vector3f(mc.value()[0], mc.value()[1], mc.value()[2]);
        has_mesh_color = mesh_color.allFinite() and mesh_color.sum() > 1e-6f;
    }

    // Shape hint for the round-top table (disc) render. `object_subtype` now carries the CLASS uniformly
    // (schema convention), so the round/square SHAPE is read from the shape-selected mesh_path filename
    // (table_concept publishes round_table.obj vs table.obj) — same "round" value the viewer keyed on before.
    const std::string subtype = (mesh_path.find("round") != std::string::npos) ? "round" : "square";

    const Eigen::Matrix3d& R = room_T_object->linear();
    const float yaw = static_cast<float>(std::atan2(R(1, 0), R(0, 0)));

    // Room-frame box center. The vertical center is half_height above the node
    // origin for floor-standing furniture, and at the origin for center-anchored
    // objects. A yaw-only rotation leaves z unchanged, so the center's z is simply
    // the translation z plus that local offset.
    const Eigen::Vector3d& t = room_T_object->translation();
    const float z_center_local = stands_on_floor ? half_height : 0.f;
    const Eigen::Vector3f center(static_cast<float>(t.x()),
                                 static_cast<float>(t.y()),
                                 static_cast<float>(t.z()) + z_center_local);

    return GraphObjectBox{min_corner, max_corner, center,
                          Eigen::Vector3f(half_width, half_depth, half_height),
                          yaw, node.name(), std::move(category), std::move(subtype),
                          std::move(rig_schema),
                          std::move(mesh_path), std::move(mesh_texture_path),
                          mesh_color, has_mesh_color};
}

std::vector<GraphObjectBox> SceneFeed::get_graph_object_boxes(const std::string& room_name,
                                                                   std::uint64_t timestamp_ms) const
{
    std::vector<GraphObjectBox> graph_boxes;
    if (!graph_ || room_name.empty())
        return graph_boxes;

    // Schema migration: all concept furniture (table/bottle/cabinet/refrigerator/…) now publishes generic
    // type "object" nodes, so a SINGLE query gathers them all (no per-class table/cylinder/box queries).
    // build_graph_object_box skips any node missing box dimensions, so non-furniture "object" nodes fall out.
    // residual_concept `obstacle` nodes are NOT drawn as red boxes — the occupancy grid is the residual display.
    auto object_nodes = graph_->get_nodes_by_type("object");
    // Level-2 arrangements (ring_metaconcept's dining_set, …) carry their own DSR type so that the
    // CONTROLLER's obstacle sweep — which takes {"object","obstacle"} — cannot see them: a rig is a
    // belief about a relation, nothing occupies its footprint, and the robot must be free to drive
    // through it. This viewer is the opposite case: it SHOULD draw them, as the flat ring outline
    // the box branch renders for the "metaconcept" category. Same reason residual_concept's carve
    // and the level-1 association scans, all keyed on "object", still exclude them.
    for (auto& n : graph_->get_nodes_by_type("metaconcept"))
        object_nodes.push_back(std::move(n));
    graph_boxes.reserve(object_nodes.size());
    for (const auto& node : object_nodes)
    {
        // Chairs with NO display mesh stay mesh-only (fitted mesh pass) — skip them here so the pre-migration
        // behaviour is preserved. A chair that DOES publish a mesh_path is boxed like any other object so the
        // box pass renders its scaled display template (the mesh pass then suppresses its fitted carcass).
        if (object_class_of(*graph_, node) == "chair")
            if (const auto mp = graph_->get_attrib_by_name<mesh_path_att>(node);
                not mp.has_value() or mp.value().empty())
                continue;
        if (const auto box = build_graph_object_box(node, room_name, timestamp_ms); box.has_value())
            graph_boxes.push_back(box.value());
    }
    return graph_boxes;
}

void SceneFeed::update_room_polygon_in_viewers()
{
    auto room_data = get_room_polygon_from_graph();
    if (!room_data.has_value())
        return;

    try
    {
        if (!room_data->polygon_x.empty() && !room_data->polygon_y.empty() && voxel_viewer_ != nullptr)
            voxel_viewer_->update_room_polygon_dual(room_data->polygon_x, room_data->polygon_y, room_data->room_height);
    }
    catch (const std::exception& e)
    {
        qWarning() << "update_room_polygon_in_viewers failed:" << e.what();
    }
}

void SceneFeed::update_viewer_graph_object_boxes(std::span<const GraphObjectBox> graph_boxes)
{
    if (voxel_viewer_ == nullptr)
        return;

    if (graph_boxes.empty())
    {
        voxel_viewer_->update_graph_boxes({}, {}, {}, {});
        return;
    }

    std::vector<QVector3D> centers;
    std::vector<QVector3D> half_extents;
    std::vector<float> yaws;
    std::vector<std::string> categories;
    std::vector<std::string> names;
    std::vector<std::string> subtypes;
    std::vector<std::string> rig_schemas;
    std::vector<std::string> mesh_paths;
    std::vector<std::string> mesh_texture_paths;
    // Per-box albedo tint; a zero vector means "no inferred colour" and the viewer leaves the asset alone.
    std::vector<QVector3D>   mesh_colors;
    centers.reserve(graph_boxes.size());
    half_extents.reserve(graph_boxes.size());
    yaws.reserve(graph_boxes.size());
    categories.reserve(graph_boxes.size());
    names.reserve(graph_boxes.size());
    subtypes.reserve(graph_boxes.size());
    rig_schemas.reserve(graph_boxes.size());
    mesh_paths.reserve(graph_boxes.size());
    mesh_texture_paths.reserve(graph_boxes.size());
    mesh_colors.reserve(graph_boxes.size());

    for (const auto& box : graph_boxes)
    {
        centers.emplace_back(box.center.x(), box.center.y(), box.center.z());
        half_extents.emplace_back(box.half_extents.x(), box.half_extents.y(), box.half_extents.z());
        yaws.push_back(box.yaw_rad);
        categories.push_back(box.category);
        names.push_back(box.node_name);
        subtypes.push_back(box.subtype);
        rig_schemas.push_back(box.rig_schema);
        mesh_paths.push_back(box.mesh_path);
        mesh_texture_paths.push_back(box.mesh_texture_path);
        mesh_colors.emplace_back(box.has_mesh_color ? box.mesh_color.x() : 0.0f,
                                 box.has_mesh_color ? box.mesh_color.y() : 0.0f,
                                 box.has_mesh_color ? box.mesh_color.z() : 0.0f);
    }

    voxel_viewer_->update_graph_boxes(centers, half_extents, yaws, categories, names, subtypes, rig_schemas,
                                      mesh_paths, mesh_texture_paths, mesh_colors);
}

void SceneFeed::update_viewer_object_meshes()
{
    if (voxel_viewer_ == nullptr || graph_ == nullptr)
        return;

    // Tables and chairs are drawn as the solid mesh (mesh_vertices_att) published by
    // table_concept / chair_concept. The mesh is gated at the source (published only when the fit
    // moves past a few mm/mrad), so it is stable. (The graph BOX is intentionally NOT drawn for
    // these types — see the box pass — so the mesh is their only visual.)
    std::vector<std::vector<float>> meshes;
    std::vector<std::string>        categories;   // parallel to meshes → per-class mesh colour in the viewer
    std::vector<std::string>        names;        // parallel to meshes → per-instance shade + 3D text label
    // A node that publishes a display mesh_path is rendered from that template (box pass), so its fitted
    // mesh_vertices is suppressed here to avoid a double-draw.
    const auto has_display_mesh = [&](const DSR::Node& n)
    { return graph_->get_attrib_by_name<mesh_path_att>(n).has_value()
             and not graph_->get_attrib_by_name<mesh_path_att>(n).value().empty(); };
    // Schema migration: table/chair/cabinet all publish generic type "object" nodes now, so a SINGLE query
    // gathers them; the per-instance mesh CATEGORY (its colour) comes from the object class (object_subtype,
    // else name prefix — see object_class_of). Only these three classes carry a solid mesh_vertices carcass
    // to draw here (bottles are box-only; the fridge draws from its display mesh_path in the box pass).
    for (const auto& node : graph_->get_nodes_by_type("object"))
    {
        const std::string cls = object_class_of(*graph_, node);
        if (cls != "table" and cls != "chair" and cls != "cabinet")
            continue;
        const auto opt = graph_->get_attrib_by_name<mesh_vertices_att>(node);
        if (not opt.has_value())
            continue;
        if (has_display_mesh(node))
            continue;
        // table_concept's mesh is ALWAYS a box slab; for a round table it misrepresents the shape.
        // Skip it here so the viewer's box pass draws the round table (disc top + pedestal) instead,
        // keyed off the same object_subtype attribute. Square tables keep the solid box mesh.
        if (cls == "table")
            if (const auto s = graph_->get_attrib_by_name<object_subtype_att>(node);
                s.has_value() and s.value() == "round")
                continue;
        meshes.emplace_back(opt.value().get());
        categories.emplace_back(cls);
        names.emplace_back(node.name());
    }
    voxel_viewer_->update_object_meshes(meshes, categories, names);
}

void SceneFeed::update_viewer_person_skeletons()
{
    if (voxel_viewer_ == nullptr || graph_ == nullptr)
        return;

    // human_concept writes the fitted BODY_18 skeleton (room frame, 18×3 flat) into mesh_vertices_att
    // of each 'person' node. Draw it directly — same room frame as the voxels/object meshes.
    std::vector<std::vector<float>> skeletons;
    for (const auto& node : graph_->get_nodes_by_type("person"))
        if (const auto opt = graph_->get_attrib_by_name<mesh_vertices_att>(node); opt.has_value())
            skeletons.emplace_back(opt.value().get());
    voxel_viewer_->update_skeletons(skeletons);
}

void SceneFeed::update_viewer_table_rfe_points()
{
    if (voxel_viewer_ == nullptr || graph_ == nullptr)
        return;

    std::vector<QVector3D> residual_points;
    const auto gather_residuals = [&](const DSR::Node& node)
    {
        if (auto residual_opt = graph_->get_attrib_by_name<residual_pts_att>(node); residual_opt.has_value())
        {
            const auto& flat = residual_opt.value().get();
            const std::size_t n = flat.size() / 3;
            residual_points.reserve(residual_points.size() + n);
            for (std::size_t i = 0; i < n; ++i)
            {
                const std::size_t idx = i * 3;
                residual_points.emplace_back(flat[idx], flat[idx + 1], flat[idx + 2]);
            }
        }
    };
    // Schema migration: table + cabinet both publish generic type "object" nodes now. Only they write their
    // model-unexplained points into residual_pts_att (table fit residual / cabinet SDF off-surface split);
    // gather_residuals no-ops on any object node lacking the attr, so one "object" query covers both classes.
    for (const auto& node : graph_->get_nodes_by_type("object"))
        gather_residuals(node);

    voxel_viewer_->update_residual_points(residual_points);
}

void SceneFeed::update_viewer_grid()
{
    if (voxel_viewer_ == nullptr || graph_ == nullptr)
        return;
    // residual_concept publishes a `grid` node under room: grid_cells_xy = flat [x0,y0,x1,y1,…] residual cell
    // centres (room frame), grid_cell_size = the square edge. Read them and hand cell centres to the viewer.
    // residual_concept's `grid` node carries two layers of flat x,y,z triples: residual_pts = OCCUPIED cells
    // (colour A), candidate_pts = the INFLATED half-robot-width clearance BORDER (colour B).
    const auto read_pts = [](const auto& flat) {
        std::vector<QVector3D> v; const std::size_t n = flat.size() / 3; v.reserve(n);
        for (std::size_t i = 0; i < n; ++i) v.emplace_back(flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]);
        return v;
    };
    std::vector<QVector3D> cells_real, border;                 // occupied cells keep their REAL top height in z
    float gxmin = 0.f, gymin = 0.f, gcell = 0.f; int gw = 0, gh = 0;
    if (const auto g = graph_->get_node("residual"); g.has_value())   // node renamed "grid"→"residual" (type kept)
    {
        if (const auto o = graph_->get_attrib_by_name<grid_occupied_cells_att>(g.value()); o.has_value()) cells_real = read_pts(o.value().get());
        if (const auto o = graph_->get_attrib_by_name<grid_border_cells_att>  (g.value()); o.has_value()) border     = read_pts(o.value().get());
        // grid_field_meta = [xmin, ymin, cell, w, h] frames the surface lattice.
        if (const auto ma = graph_->get_attrib_by_name<grid_field_meta_att>(g.value()); ma.has_value() and ma.value().get().size() >= 5)
        {
            const auto& M = ma.value().get();
            gxmin = M[0]; gymin = M[1]; gcell = M[2]; gw = static_cast<int>(M[3]); gh = static_cast<int>(M[4]);
        }
    }
    // Flat 2-D amber cells want the floor; flatten a copy. The 3-D surface uses the real height.
    std::vector<QVector3D> cells_flat = cells_real;
    for (auto& c : cells_flat) c.setZ(0.02f);
    voxel_viewer_->update_grid_cells(cells_flat, 0.05f);
    voxel_viewer_->update_grid_border(border);
    voxel_viewer_->update_grid_field(cells_real, gxmin, gymin, gcell, gw, gh);
}

void SceneFeed::update_viewer_mask_points()
{
    if (voxel_viewer_ == nullptr || graph_ == nullptr)
        return;

    // The masks node carries the mask support points (flat xyz, room frame) as a runtime attribute.
    // Draw the object slices — the YOLO concept targets "bottle"/"table"/"chair" AND every YOLO-sem
    // furniture mask (class_id >= 1000, produced by SemanticMaskStage: cabinet/hood/shelf/…) — so the
    // Masks toggle shows them; other detections are clutter here. Coloured by color_for_category. Per-mask
    // point ranges come from mask_support_offsets; label i in mask_labels ('|'-joined) owns [offsets[i], offsets[i+1]).
    static const std::array<std::string_view, 5> kDrawnMaskLabels{"bottle", "table", "chair",
                                                                  "refrigerator", "microwave"};
    std::vector<QVector3D>   mask_points;
    std::vector<std::string> mask_categories;   // parallel to mask_points → per-class colour in the viewer
    std::vector<float>       mask_sources;       // parallel to mask_points → sensor source (0=zed, 1=ricoh) → brightness
    if (const auto masks_node = graph_->get_node("masks"); masks_node.has_value())
    {
        // TYPE-ATTRIBUTED reads (CLAUDE.md), compile-checked against dsr_attr_name.h.
        const auto pts_opt    = graph_->get_attrib_by_name<mask_support_points_cam_att>(masks_node.value());
        const auto mts_opt    = graph_->get_attrib_by_name<mask_timestamp_ms_att>(masks_node.value());
        const auto off_opt    = graph_->get_attrib_by_name<mask_support_offsets_att>(masks_node.value());
        const auto labels_opt = graph_->get_attrib_by_name<mask_labels_att>(masks_node.value());
        const auto ids_opt    = graph_->get_attrib_by_name<mask_label_ids_att>(masks_node.value());  // class_id per mask; semantic masks use 1000+ade_id
        const auto src_opt    = graph_->get_attrib_by_name<mask_source_att>(masks_node.value());     // optional (older producers omit it → treated as zed/bright)
        // The producer publishes these in the ZED/CAMERA frame now, so the VIEWER pays for the room
        // transform — pinned to the mask capture stamp, not "latest", or the cloud lags the robot pose
        // and shimmers under motion. No transform ⇒ draw nothing, rather than dump camera-frame points
        // into a room-frame scene (they would appear glued to the origin).
        std::optional<Mat::RTMat> room_T_zed;
        if (inner_eigen_api_ != nullptr and pts_opt and not pts_opt->get().empty())
        {
            std::string room_name, robot_name;
            { room_name = room_node_name_; robot_name = robot_node_name_; }
            const std::uint64_t mts = mts_opt ? mts_opt.value() : 0;
            room_T_zed = room_T_zed_extrapolated(inner_eigen_api_, room_name, robot_name, mts);
        }
        if (pts_opt and off_opt and labels_opt and room_T_zed.has_value())
        {
            const Eigen::Matrix3f Rz = room_T_zed->linear().cast<float>();
            const Eigen::Vector3f tz = room_T_zed->translation().cast<float>();
            const auto& flat    = pts_opt->get();
            const auto& offsets = off_opt->get();
            const std::vector<float> empty_src;
            const auto& sources = src_opt ? src_opt->get() : empty_src;
            const auto& label_ids = ids_opt ? ids_opt->get() : empty_src;

            std::vector<std::string> labels;
            std::stringstream ls(labels_opt->get());
            for (std::string lbl; std::getline(ls, lbl, '|'); )
                labels.push_back(lbl);

            const std::size_t n_masks = labels.size();
            for (std::size_t m = 0; m < n_masks and m + 1 < offsets.size(); ++m)
            {
                const bool is_semantic = (m < label_ids.size()) and (label_ids[m] >= 1000.0f);
                if (not is_semantic
                    and std::find(kDrawnMaskLabels.begin(), kDrawnMaskLabels.end(), labels[m]) == kDrawnMaskLabels.end())
                    continue;
                const std::size_t begin = static_cast<std::size_t>(offsets[m]);
                const std::size_t end   = static_cast<std::size_t>(offsets[m + 1]);
                const float src = (m < sources.size()) ? sources[m] : 0.0f;   // default zed (bright)
                for (std::size_t i = begin; i < end and (i * 3 + 2) < flat.size(); ++i)
                {
                    const Eigen::Vector3f pr =
                        Rz * Eigen::Vector3f(flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]) + tz;
                    mask_points.emplace_back(pr.x(), pr.y(), pr.z());
                    mask_categories.push_back(labels[m]);
                    mask_sources.push_back(src);
                }
            }
        }
    }
    voxel_viewer_->update_mask_points(mask_points, mask_categories, mask_sources);
}

void SceneFeed::update_viewer_robot_pose(const Mat::RTMat& room_T_robot)
{
    if (voxel_viewer_ == nullptr)
        return;
    const auto& t = room_T_robot.translation();
    const Eigen::Matrix3d R = room_T_robot.rotation();
    const float tx = static_cast<float>(t.x());
    const float ty = static_cast<float>(t.y());
    const float ttheta = static_cast<float>(std::atan2(R(1, 0), R(0, 0)));

    // Time-constant EMA (rate-independent: works whether fed at the 60 Hz render tick or 16 Hz frame
    // path). tau ≈ correction interval so prediction noise / correction snaps are averaged out while
    // real motion still tracks with only ~tau of display lag. tau <= 0 would disable smoothing.
    constexpr float kTauS = 0.12f;
    if (not robot_disp_sm_init_)
    {
        robot_disp_sm_x_ = tx;
        robot_disp_sm_y_ = ty;
        robot_disp_sm_theta_ = ttheta;
        robot_disp_sm_last_ = std::chrono::steady_clock::now();
        robot_disp_sm_init_ = true;
    }
    else
    {
        const auto now = std::chrono::steady_clock::now();
        const float dt = std::chrono::duration<float>(now - robot_disp_sm_last_).count();
        robot_disp_sm_last_ = now;
        const float alpha = (kTauS > 0.f and dt > 0.f) ? (1.f - std::exp(-dt / kTauS)) : 1.f;
        robot_disp_sm_x_ += alpha * (tx - robot_disp_sm_x_);
        robot_disp_sm_y_ += alpha * (ty - robot_disp_sm_y_);
        // Wrap the angle delta into (-π, π] before blending so it smooths across the ±π seam.
        float dth = ttheta - robot_disp_sm_theta_;
        while (dth >  static_cast<float>(M_PI)) dth -= 2.f * static_cast<float>(M_PI);
        while (dth < -static_cast<float>(M_PI)) dth += 2.f * static_cast<float>(M_PI);
        robot_disp_sm_theta_ += alpha * dth;
    }
    voxel_viewer_->set_robot_pose(robot_disp_sm_x_, robot_disp_sm_y_, robot_disp_sm_theta_);
}

void SceneFeed::refresh_viewer_robot_pose_latest()
{
    if (inner_eigen_api_ == nullptr || voxel_viewer_ == nullptr)
        return;
    const auto [room_name, robot_name] = room_robot_names();
    if (room_name.empty() || robot_name.empty())
        return;
    // ts=0 → latest RT (not pinned to the camera frame stamp), so the robot tracks live odometry
    // at the render-timer rate rather than stepping at the ~7-10 Hz perception cadence.
    if (auto room_T_robot = inner_eigen_api_->get_transformation_matrix(room_name, robot_name, 0);
        room_T_robot.has_value())
        update_viewer_robot_pose(room_T_robot.value());
}

void SceneFeed::update_viewer_lidar_points(std::span<const Eigen::Vector3f> lidar_points_room,
                                                std::span<const std::uint8_t> plane_id)
{
    if (voxel_viewer_ == nullptr)
        return;
    // Empty ⇒ no fresh sweep drained this cycle: keep the last cloud (never blank it → no flicker). The
    // caller (compute) already drained get_lidar3D() ONCE and posed the cloud into the room frame at the
    // scan stamp; re-draining/re-interpolating here desynced the viewer and shimmered the cloud.
    if (lidar_points_room.empty())
        return;

    std::vector<QVector3D> pts;
    pts.reserve(lidar_points_room.size());
    for (const auto& p : lidar_points_room)
        pts.emplace_back(p.x(), p.y(), p.z());

    voxel_viewer_->update_lidar_points(pts, plane_id);
}
// ─── Lifecycle / driver ──────────────────────────────────────────────────────────────────────────

namespace
{
// The LiDAR plane reader, kept file-local so the header stays free of media_transport types.
std::unique_ptr<rc::media::LidarPlaneReader> g_lidar_reader;
}

SceneFeed::SceneFeed(std::shared_ptr<DSR::DSRGraph> graph, rc::VoxelOpenGLViewer* viewer)
    : graph_(std::move(graph)), voxel_viewer_(viewer)
{
    if (graph_)
        inner_eigen_owned_ = graph_->get_inner_eigen_api();
    inner_eigen_api_ = inner_eigen_owned_.get();
}

std::pair<std::string, std::string> SceneFeed::room_robot_names()
{
    if (room_node_name_.empty())
        if (const auto rooms = graph_->get_nodes_by_type("room"); not rooms.empty())
            room_node_name_ = rooms.front().name();
    if (robot_node_name_.empty())
        if (const auto robots = graph_->get_nodes_by_type("robot"); not robots.empty())
            robot_node_name_ = robots.front().name();
    return {room_node_name_, robot_node_name_};
}

bool SceneFeed::init_lidar(const std::string& /*topic*/)
{
    // Topic/domain are NOT ours to choose — they come from the media descriptor robot_concept authors
    // on each sensor node (CLAUDE.md, media plane). We only name the PLANES, in preference order, the
    // same pair the retina and the concept agents read.
    if (g_lidar_reader)
        return g_lidar_reader->any_live();
    if (not graph_ or inner_eigen_api_ == nullptr)
        return false;
    g_lidar_reader = std::make_unique<rc::media::LidarPlaneReader>(
        graph_, inner_eigen_api_, std::vector<std::string>{"helios", "bpearl"}, "lidar");
    return g_lidar_reader->any_live();
}

void SceneFeed::poll_lidar()
{
    // The cloud arrives in the LIDAR's own frame; the reader poses it into the room at the SCAN stamp
    // (interpolated), which is the same thing the retina did. Empty ⇒ keep the last cloud rather
    // than blanking the view for one dropped sweep.
    if (not g_lidar_reader)
        return;
    const auto [room, robot] = room_robot_names();
    if (room.empty())
        return;
    // Interpolated RT at the sweep stamp: the room<-robot leg is dynamic, so a Nearest query would
    // register the cloud against a stale pose and the whole scene shimmers while driving.
    if (const auto sweep = g_lidar_reader->poll(room, /*interpolate=*/true); sweep.has_value())
        update_viewer_lidar_points(sweep->points, sweep->plane_id);
}

void SceneFeed::refresh()
{
    if (graph_ == nullptr or voxel_viewer_ == nullptr)
        return;

    const auto [room_name, robot_name] = room_robot_names();

    // The room polygon barely changes and rebuilding it walks every wall; 1 tick in 50 is what the
    // retina used and it is invisible at 5 Hz.
    if (polygon_tick_++ % 50 == 0)
        update_room_polygon_in_viewers();

    // Everything below tolerates a missing room: each no-ops rather than drawing something wrong.
    if (not room_name.empty())
    {
        const auto boxes = get_graph_object_boxes(room_name, 0);
        update_viewer_graph_object_boxes(boxes);
    }
    update_viewer_object_meshes();
    update_viewer_person_skeletons();
    update_viewer_table_rfe_points();
    update_viewer_grid();
    update_viewer_mask_points();
    refresh_viewer_robot_pose_latest();
    poll_lidar();
}
