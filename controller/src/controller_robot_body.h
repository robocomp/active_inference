/*
 * controller_robot_body.h — build THIS robot's body from the mesh the GRAPH names.
 *
 * ★★★THE DEFECT THIS CLOSES. `GridPlanner` has done exact, per-heading footprint collision since it replaced
 * the visibility graph — obstacles at true extent, no inflation, the body applied at query time. What it was
 * handed, however, was `RobotFootprint::shadow()`: one robot's silhouette, compiled in, given to every robot
 * the controller ever drives. Measured on the robot actually running (P3Bot), in the ROBOT frame:
 *
 *      |x|max (lateral)  compiled 0.2716   mesh 0.2938   -> the planner is 22.2 mm TOO NARROW per side
 *      |y|max (forward)  compiled 0.2300   mesh 0.2266   -> 3.4 mm too long (conservative)
 *      circumscribed     compiled 0.3252   mesh 0.3617   -> 36.5 mm too small
 *
 * Lateral is the direction in which a corridor closes, and every consumer inherits the error: A* collision,
 * the MPPI ESDF body extent, the LiDAR self-filter radius, the route optimiser's acceptance guard, the
 * arrival band, standpoint rejection. ROBOT_GEOMETRY.md already carried this as an OPEN item and already
 * called the compiled transcription "32.5 mm adrift from the mesh it claims to come from".
 *
 * ★THE MESH IS NOT ALWAYS IN THE ROBOT FRAME, AND GETTING THAT WRONG INVERTS THE FIX. P3Bot's mesh is in the
 * proto's NATIVE frame, where forward is +x, while the robot frame here has forward +y — `P3Bot.proto:430`
 * wraps the whole robot in `Pose { rotation 0 0 1 1.5708 }` and says so. Verified independently against the
 * graph's own RT tree: helios proto (-0.0535264, 0.00230873) appears in p3bot.json as (-0.002309, -0.053526),
 * which is exactly Rz(+90); zed (0.088,0) -> (0,0.088); ricoh (-0.05,0) -> (0,-0.05). All exact. Applied with
 * the wrong sign, the same mesh makes the body 45 mm too NARROW instead of 22 mm too wide — the error moves
 * to the dangerous axis. So the angle is explicit, logged, and cross-checked against the mounts; it is never
 * guessed from the mesh's aspect ratio.
 *
 * ★SHADOW IS CROSS-CHECKED AGAINST THE COMPILED HULL (Task 1b, 2026-10-20). The old display mesh `shadow.obj` omitted the two
 * drive wheels (Shadow.proto: r 0.1, h 0.05 on the lateral axle, x = ±0.259): |x|max 0.2420 against the true 0.284, so
 * deriving the body from it would have SHRUNK it ~42 mm per side. The robot node's `path` is now the absolute path of
 * webots-shadow/protos/meshes/shadow_footprint.obj (shadow.stl ∪ the two wheel cylinders, written by
 * controller/tools/shadow_footprint.py), so ONE mesh is the source of the outline for every agent. The mesh body is used only
 * if it agrees with the compiled `shadow()` (|x|max 1 mm, |y|max 2 mm, area 1 %); an older mesh without wheels is refused
 * and the compiled hull stays, with the numbers in the log. The startup report diffs them either way.
 *
 * ★WHAT HAPPENS WHEN IT FAILS. It falls back to `RobotFootprint::shadow()` — today's behaviour, so a failure
 * is never a regression — but it says so at qCritical with a reason, and the source string travels with the
 * body into the mission manifest. A P3Bot silently planning as a Shadow is exactly the bug being fixed; it
 * must not be the quiet outcome of a missing file.
 */

#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

#include "dsr/api/dsr_api.h"
#include "../../common/robot_footprint/robot_footprint.h"

namespace rc
{

struct RobotBodyReport
{
    std::string node_name;
    std::uint64_t node_id = 0;
    std::string raw_path;         // the `path` attribute, verbatim
    std::string resolved_path;    // ...and where it was actually found
    float yaw_offset_deg = 0.f;
    std::string yaw_source;       // "config" | "asset-is-robot-frame"
    RobotFootprint::MeshReport mesh;
    bool fell_back = false;
    std::string reason;           // why, when fell_back
    std::string active;           // "mesh:<path>" | "compiled:shadow"
};

// ★THE PATH IS RELATIVE TO ROBOT_CONCEPT'S RUN DIR, NOT OURS. `p3bot.json` writes "meshes/p3bot.obj", while
// object nodes' `mesh_path` is relative to the components root — two conventions in one graph, which is why
// both 3-D viewers carry a RobotMeshPath config override rather than trusting the attribute. Resolving here
// rather than adding a third convention: absolute, then cwd, then the sibling robot_concept dir, then the
// components root. Returns nullopt if none of them exists.
std::optional<std::filesystem::path> resolve_robot_mesh_path(const std::string &relative);

// The one entry point. Never throws; always returns a usable body. `report` carries everything worth
// printing, including the compiled-hull deltas the caller logs.
RobotFootprint load_robot_body(DSR::DSRGraph &graph,
                              std::uint64_t robot_id,
                              const std::string &robot_name,
                              float mesh_yaw_deg,
                              RobotBodyReport &report);

// Task 14 fix round 5: the robot mesh's XY outline (robot frame), read from the robot node's `path` WHATEVER body the
// planner keeps — for Shadow the planner keeps the compiled hull (the mesh omits the wheels), but the mesh is still the
// best statement of the front cover the LiDAR sees. nullopt (with `why`) when there is no path / file / usable hull.
std::optional<RobotFootprint> load_robot_mesh_outline(DSR::DSRGraph &graph, std::uint64_t robot_id, float mesh_yaw_deg,
                                                      std::string &why, std::string &path_out);

// One block, unconditional, at qInfo — or qCritical when it fell back. The numbers beside the numbers they
// replace, because "a body loaded" is not the question; "which body, and how does it differ from the one the
// robot was driving" is.
void log_robot_body_report(const RobotBodyReport &report, const RobotFootprint &active);

}   // namespace rc
