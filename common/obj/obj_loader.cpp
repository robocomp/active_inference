#include "obj_loader.h"

#include <QCoreApplication>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

namespace rc::obj
{

namespace
{
int parse_obj_index(const std::string& token, int vertex_count)
{
    const auto slash = token.find('/');
    const std::string index_str = token.substr(0, slash);
    if (index_str.empty())
        return -1;

    const int raw_index = std::stoi(index_str);
    if (raw_index > 0)
        return raw_index - 1;
    if (raw_index < 0)
        return vertex_count + raw_index;
    return -1;
}
}  // namespace

std::optional<std::filesystem::path> resolve_robot_mesh_path(const std::string& mesh_path)
{
    if (mesh_path.empty())
        return std::nullopt;

    const std::filesystem::path input(mesh_path);
    std::vector<std::filesystem::path> candidates;
    if (input.is_absolute())
        candidates.push_back(input);
    else
    {
        candidates.push_back(std::filesystem::current_path() / input);
        const auto app_dir = std::filesystem::path(QCoreApplication::applicationDirPath().toStdString());
        candidates.push_back(app_dir.parent_path() / input);
        candidates.push_back(input);
    }

    for (const auto& candidate : candidates)
        if (!candidate.empty() && std::filesystem::exists(candidate))
            return candidate;

    return std::nullopt;
}

std::optional<ObjMeshData> load_obj_mesh_data(const std::filesystem::path& mesh_path)
{
    std::ifstream input(mesh_path);
    if (!input.is_open())
        return std::nullopt;

    std::vector<QVector3D> vertices;
    std::vector<QVector3D> triangles;
    QVector3D bb_min;
    QVector3D bb_max;
    bool have_bounds = false;
    std::string line;

    while (std::getline(input, line))
    {
        if (line.size() < 2)
            continue;

        std::istringstream stream(line);
        std::string tag;
        stream >> tag;
        if (tag == "v")
        {
            float x = 0.f, y = 0.f, z = 0.f;
            if (!(stream >> x >> y >> z))
                continue;

            const QVector3D vertex(x, y, z);
            vertices.push_back(vertex);
            if (!have_bounds)
            {
                bb_min = vertex;
                bb_max = vertex;
                have_bounds = true;
            }
            else
            {
                bb_min.setX(std::min(bb_min.x(), vertex.x()));
                bb_min.setY(std::min(bb_min.y(), vertex.y()));
                bb_min.setZ(std::min(bb_min.z(), vertex.z()));
                bb_max.setX(std::max(bb_max.x(), vertex.x()));
                bb_max.setY(std::max(bb_max.y(), vertex.y()));
                bb_max.setZ(std::max(bb_max.z(), vertex.z()));
            }
        }
        else if (tag == "f")
        {
            std::vector<int> face_indices;
            std::string token;
            while (stream >> token)
            {
                const int index = parse_obj_index(token, static_cast<int>(vertices.size()));
                if (index >= 0 && index < static_cast<int>(vertices.size()))
                    face_indices.push_back(index);
            }

            if (face_indices.size() < 3)
                continue;

            for (std::size_t i = 1; i + 1 < face_indices.size(); ++i)
            {
                triangles.push_back(vertices[static_cast<std::size_t>(face_indices[0])]);
                triangles.push_back(vertices[static_cast<std::size_t>(face_indices[i])]);
                triangles.push_back(vertices[static_cast<std::size_t>(face_indices[i + 1])]);
            }
        }
    }

    if (triangles.empty() || !have_bounds)
        return std::nullopt;

    return ObjMeshData{std::move(triangles), bb_min, bb_max};
}

}  // namespace rc::obj
