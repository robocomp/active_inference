#pragma once

// Wavefront OBJ loading for the 3D viewer's robot mesh. Pure file I/O + triangulation, no GL or Qt
// widget state — split out of voxel_opengl_viewer.cpp.

#include <QVector2D>
#include <QVector3D>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace rc::obj
{

// One material group of a mesh (all the faces that share a `usemtl`). Carries its own geometry + material,
// so a single OBJ can mix e.g. a textured body and a flat-coloured handle.
struct ObjSubmesh
{
    std::vector<QVector3D> triangles;   // positions (fan-triangulated)
    std::vector<QVector2D> uvs;         // per-vertex texture coords (parallel to `triangles`; (0,0) if none)
    QVector3D diffuse{0.8f, 0.8f, 0.8f};// Kd from the .mtl
    bool has_diffuse = false;           // true iff the .mtl specified Kd for this material
    std::string texture_path;           // ABSOLUTE path to the .mtl's map_Kd (empty ⇒ untextured); resolved by the loader
};

struct ObjMeshData
{
    std::vector<ObjSubmesh> submeshes;  // ≥1 group (a bare OBJ with no materials yields one default group)
    QVector3D bb_min;
    QVector3D bb_max;
};

// Resolve a (possibly relative) robot mesh path against cwd, the app dir's parent, and the raw path.
// Returns the first existing candidate, or nullopt if none exist.
std::optional<std::filesystem::path> resolve_robot_mesh_path(const std::string& mesh_path);

// Parse a Wavefront OBJ into a triangle soup (fan-triangulated faces) plus a bounding box.
// Returns nullopt on open failure or an empty/boundless mesh.
std::optional<ObjMeshData> load_obj_mesh_data(const std::filesystem::path& mesh_path);

}  // namespace rc::obj
