/*
 * controller_robot_body.cpp — see controller_robot_body.h
 */

#include "controller_robot_body.h"

#include <cmath>
#include <print>

#include <QtGlobal>

namespace rc
{

std::optional<std::filesystem::path> resolve_robot_mesh_path(const std::string &relative)
{
    namespace fs = std::filesystem;
    if (relative.empty()) return std::nullopt;

    const fs::path rel(relative);
    std::error_code ec;
    if (rel.is_absolute())
        return fs::exists(rel, ec) ? std::optional<fs::path>(rel) : std::nullopt;

    const fs::path cwd = fs::current_path(ec);
    // Ordered from "exactly what was written" outward. The third and fourth entries are the ones that
    // actually find it: the controller runs in components/active_inference/controller, and the attribute is
    // written relative to components/active_inference/robot_concept.
    const fs::path candidates[] = {
        cwd / rel,
        cwd.parent_path() / "robot_concept" / rel,
        cwd.parent_path() / rel,
        cwd.parent_path().parent_path() / "robot_concept" / rel,
    };
    for (const auto &c : candidates)
        if (fs::exists(c, ec)) return fs::weakly_canonical(c, ec);
    return std::nullopt;
}

RobotFootprint load_robot_body(DSR::DSRGraph &graph,
                              std::uint64_t robot_id,
                              const std::string &robot_name,
                              float mesh_yaw_deg,
                              RobotBodyReport &report)
{
    report = RobotBodyReport{};
    report.node_id = robot_id;
    report.node_name = robot_name;
    report.yaw_offset_deg = mesh_yaw_deg;
    report.yaw_source = std::abs(mesh_yaw_deg) > 1e-3f ? "config" : "asset-is-robot-frame";

    const auto fall_back = [&report](std::string why)
    {
        report.fell_back = true;
        report.reason = std::move(why);
        report.active = "compiled:shadow";
        return RobotFootprint::shadow();
    };

    // ★SHADOW KEEPS ITS COMPILED HULL. See the header: shadow.obj has no wheels, so deriving from it would
    // shrink the body 29.6 mm per side. This is checked before the file is even opened, so the refusal reads
    // as a decision rather than a parse failure.
    if (robot_name == "Shadow")
        return fall_back("shadow.obj omits the four wheels (|x|max 0.2420 vs the assembly's 0.2464 — "
                         "ROBOT_GEOMETRY.md); the compiled hull is the better measurement for this robot");

    const auto node = graph.get_node(robot_id);
    if (not node.has_value())
        return fall_back("robot node " + std::to_string(robot_id) + " is not in the graph");

    // Type-attributed read (CLAUDE.md). Note `path_att` is a reference_wrapper<const std::string>, unlike the
    // plain std::string `mesh_path_att` the object nodes use — the two mesh contracts differ in type as well
    // as in what their paths are relative to.
    const auto attr = graph.get_attrib_by_name<path_att>(node.value());
    if (not attr.has_value())
        return fall_back("robot node '" + robot_name + "' has no 'path' attribute naming a mesh");
    report.raw_path = attr.value().get();

    const auto resolved = resolve_robot_mesh_path(report.raw_path);
    if (not resolved.has_value())
        return fall_back("cannot find '" + report.raw_path + "' from " +
                         std::filesystem::current_path().string() +
                         " (the robot's path is relative to robot_concept's run dir)");
    report.resolved_path = resolved->string();

    auto body = RobotFootprint::from_obj(report.resolved_path,
                                         mesh_yaw_deg * static_cast<float>(M_PI) / 180.f,
                                         report.mesh);
    if (not body.has_value())
        return fall_back(report.mesh.reason);

    report.active = body->source();
    return *body;
}

void log_robot_body_report(const RobotBodyReport &report, const RobotFootprint &active)
{
    const RobotFootprint reference = RobotFootprint::shadow();

    std::println("[body] robot '{}' (id {}) path='{}'", report.node_name, report.node_id,
                 report.raw_path.empty() ? "<none>" : report.raw_path);
    if (not report.resolved_path.empty())
        std::println("[body]   resolved '{}'", report.resolved_path);

    if (report.fell_back)
    {
        // qCritical, not qInfo: running one robot on another robot's body is the defect this code exists to
        // remove, so it must never be the quiet outcome. It is not a REGRESSION — it is exactly what every
        // build before this one did — but it has to be legible in the log and in the run manifest.
        qCritical("[body] FALLBACK to compiled shadow(): %s", report.reason.c_str());
        std::println("[body]   ACTIVE: compiled:shadow  inscribed {:.4f}  circumscribed {:.4f}",
                     reference.inscribed_radius(), reference.circumscribed_radius());
        return;
    }

    const auto &m = report.mesh;
    std::println("[body]   mesh {} verts ({} rejected)  bbox x[{:.4f},{:.4f}] y[{:.4f},{:.4f}] z[{:.4f},{:.4f}] (MESH frame)",
                 m.vertices, m.vertices_rejected, m.bb_min.x(), m.bb_max.x(),
                 m.bb_min.y(), m.bb_max.y(), m.bb_min.z(), m.bb_max.z());
    std::println("[body]   yaw {:+.1f} deg  source={}  |  hull {} -> {} verts ({:+.2f}% area, outward-only)",
                 report.yaw_offset_deg, report.yaw_source, m.hull_raw, m.hull_simplified,
                 100.f * m.area_growth_frac);
    std::println("[body]   ROBOT frame: inscribed {:.4f}  circumscribed {:.4f}  |x|max {:.4f}  |y|max {:.4f}  area {:.4f}",
                 m.inscribed, m.circumscribed, m.x_max, m.y_max, m.area_m2);
    // The deltas are the point of the whole line. A body that differs from the compiled one by millimetres is
    // a refactor; one that differs by centimetres changes what the robot may drive through, and which
    // direction it moved in is what says whether that is safer or bolder.
    std::println("[body]   vs compiled shadow(): inscribed {:+.4f}  circumscribed {:+.4f}  |x|max {:+.4f}  |y|max {:+.4f}",
                 m.inscribed - reference.inscribed_radius(),
                 m.circumscribed - reference.circumscribed_radius(),
                 m.x_max - 0.2716f, m.y_max - 0.2300f);
    std::println("[body]   hull centroid {:.4f} m from the rotation centre  |  ACTIVE: {}",
                 m.centroid.norm(), report.active);
    (void)active;
}

}   // namespace rc
