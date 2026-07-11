#pragma once

// Wavefront OBJ loading for the 3D viewer's robot mesh. Pure file I/O + triangulation, no GL or Qt
// widget state — split out of voxel_opengl_viewer.cpp.

#include <QVector3D>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace rc::obj
{

struct ObjMeshData
{
    std::vector<QVector3D> triangles;
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
