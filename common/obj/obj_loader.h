#pragma once

// Wavefront OBJ loading for 3D viewers (active_inference/common). Pure file I/O + triangulation,
// no GL or Qt widget state. Shared so any agent's viewer can load a robot/object mesh the same way.

#include <QVector3D>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace rc::obj
{

struct ObjMeshData
{
    std::vector<QVector3D> triangles;   // flat triangle soup: 3 vertices per triangle
    QVector3D bb_min;
    QVector3D bb_max;
};

// Resolve a (possibly relative) mesh path against cwd, the app dir's parent, and the raw path.
// Returns the first existing candidate, or nullopt if none exist.
std::optional<std::filesystem::path> resolve_robot_mesh_path(const std::string& mesh_path);

// Parse a Wavefront OBJ into a triangle soup (fan-triangulated faces) plus a bounding box.
// Returns nullopt on open failure or an empty/boundless mesh.
std::optional<ObjMeshData> load_obj_mesh_data(const std::filesystem::path& mesh_path);

}  // namespace rc::obj
