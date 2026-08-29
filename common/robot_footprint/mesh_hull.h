/*
 * mesh_hull.h — the robot's ground-plane silhouette, read from the mesh the GRAPH names.
 *
 * WHY THIS EXISTS. `RobotFootprint::shadow()` is a vertex list compiled into the binary: one robot's body,
 * hand-transcribed, and handed to every robot the controller ever drives. On P3Bot that is not a style
 * complaint, it is a wrong body — measured against `robot_concept/meshes/p3bot.obj` in the ROBOT frame, the
 * compiled hull is 22.2 mm too NARROW per side (0.2716 against 0.2938) and 34.5 mm short in circumscribed
 * radius. Lateral is the direction in which a corridor closes. ROBOT_GEOMETRY.md already carried the task
 * ("derive shadow() from the assembly", still OPEN) and already called the transcription "32.5 mm adrift from
 * the mesh it claims to come from".
 *
 * WHAT IT DOES. Reads the `v` lines of an OBJ, projects them to the ground plane, takes the convex hull, and
 * reduces the vertex count OUTWARD ONLY. Nothing else in an OBJ matters to a footprint — no faces, no
 * materials, no normals, no UVs — so this reads none of them, and is therefore ~30 lines of parsing rather
 * than a mesh library.
 *
 * ★★★IT PARSES WITH std::from_chars, AND THAT IS NOT A STYLE CHOICE. These machines run LANG=es_ES.UTF-8,
 * where the decimal separator is a COMMA, and every Qt program here calls setlocale(LC_ALL, "") at startup.
 * strtof/atof/strtod/istringstream>> all read through LC_NUMERIC, so on a file written with decimal POINTS
 * they stop at the '.' and return the integer part — silently, with no error flag. "0.2938" becomes 0, and a
 * robot whose every vertex is 0 hulls to nothing. from_chars is locale-independent BY CONSTRUCTION and
 * reports failure instead of guessing. (The two existing OBJ readers in this repo — viewer3d/src/obj_loader
 * and common/obj/obj_loader — use `istringstream >>` and happen to work only because nothing has called
 * std::locale::global yet. Do not copy them. See CLAUDE.md.)
 *
 * ★THE HULL IS TAKEN OVER EVERY VERTEX AT EVERY HEIGHT, deliberately. The 2-D projection of a rigid mesh is
 * invariant to translation in z, so hulling everything makes this function independent of where the mesh puts
 * its floor — and the two meshes disagree about that (shadow.obj sits at z_min -0.050, p3bot.obj at +0.041).
 * A z filter would make that datum load-bearing for a number that does not depend on it. The z_lo/z_hi
 * parameters exist for the height-banded (2.5-D) footprint that comes later, where the datum IS load-bearing
 * and has to be supplied by the caller; pass the whole range for a flat footprint.
 *
 * ★SIMPLIFICATION IS OUTWARD ONLY. A vertex is removed by extending its two neighbouring edge LINES to their
 * intersection, so the result strictly CONTAINS the input and can never shrink clearance. That is the same
 * discipline ROBOT_GEOMETRY.md records for the proto-derived hull ("36-vertex raw hull simplified to 12 at
 * +0.18% area, with outward-only simplification so it is a strict superset"). A simplifier that could cut a
 * corner would quietly hand the planner a body smaller than the robot.
 *
 * Pure Eigen/STL — no Qt, no DSR, no image or mesh library. It belongs to ai_common::geometry precisely so
 * controller/tools/{route_bench,tracker_sim,mppi_bench} can build the SAME body the component runs without
 * dragging in Qt.
 */

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace rc::mesh
{

// What an OBJ's `v` lines said, and nothing else.
struct Vertices
{
    std::vector<Eigen::Vector3f> v;
    Eigen::Vector3f bb_min = Eigen::Vector3f::Zero();
    Eigen::Vector3f bb_max = Eigen::Vector3f::Zero();
    // ★A REJECTED LINE IS NOT A WARNING, IT IS A DIFFERENT FILE. If a `v` line fails to parse, either the file
    // is not what we think it is or the parse is broken — and the second case is the locale bug this module
    // exists to be immune to. Counted so a caller can refuse rather than hull whatever survived.
    long v_lines = 0;
    long v_rejected = 0;
};

// Reads ONLY `v` lines. `err` (optional) receives the reason on failure. Never throws.
std::optional<Vertices> read_obj_vertices(const std::string &path, std::string *err = nullptr);

// Andrew's monotone chain. CCW, colinear points dropped, degenerate input yields fewer than 3 points.
std::vector<Eigen::Vector2f> convex_hull_2d(std::vector<Eigen::Vector2f> pts);

struct Simplified
{
    std::vector<Eigen::Vector2f> poly;
    float area_growth_frac = 0.f;   // (simplified - raw) / raw; >= 0 by construction
};

// Outward-only vertex reduction — see the header note. Stops when `max_verts` is reached or when the next
// removal would cost more than `max_area_growth_frac`, whichever binds first. Never returns fewer than 3.
Simplified simplify_hull_outward(const std::vector<Eigen::Vector2f> &hull,
                                 std::size_t max_verts,
                                 float max_area_growth_frac);

// Signed area (positive for CCW). Exposed because both the simplifier and the caller's report want it.
float polygon_area(const std::vector<Eigen::Vector2f> &poly);

// The whole pipeline: read → project the slab [z_lo, z_hi] → hull → simplify. Pass a slab covering the mesh
// for a flat footprint (see the header on why z must not matter there).
struct HullResult
{
    std::vector<Eigen::Vector2f> hull;        // simplified, CCW, in the MESH frame
    std::vector<Eigen::Vector2f> hull_raw;    // before simplification — the superset check needs it
    Eigen::Vector3f bb_min = Eigen::Vector3f::Zero();
    Eigen::Vector3f bb_max = Eigen::Vector3f::Zero();
    long  vertices = 0;
    long  vertices_rejected = 0;
    float area_raw = 0.f;
    float area = 0.f;
    float area_growth_frac = 0.f;
};

std::optional<HullResult> hull_from_obj(const std::string &path,
                                        float z_lo, float z_hi,
                                        std::size_t max_verts = 24,
                                        float max_area_growth_frac = 0.01f,
                                        std::string *err = nullptr);

// Exercises the locale trap on a literal buffer, the hull's rotation equivariance, and the superset property
// of the simplifier. ★A HARNESS THAT DOES NOT setlocale(LC_ALL, "") IS ANSWERING A DIFFERENT QUESTION than
// the agent asks: with no Qt in the process the C locale stays "C" and the truncation bug simply does not
// occur. The caller must set the locale before running this; it says so if it looks like nobody did.
bool self_test();

}   // namespace rc::mesh
