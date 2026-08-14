#include "obj_loader.h"

#include <QCoreApplication>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

namespace rc::obj
{

namespace
{
struct MtlEntry
{
    QVector3D kd{0.8f, 0.8f, 0.8f};
    bool has_kd = false;
    std::string map_kd;   // absolute path (resolved relative to the .mtl dir); empty if none/missing
};

// Parse a .mtl file: newmtl / Kd r g b / map_Kd <file>. map_Kd is resolved relative to the .mtl's own dir
// and kept only if the file exists. Robust to missing files (returns whatever it could read).
std::unordered_map<std::string, MtlEntry> parse_mtl(const std::filesystem::path& mtl_path)
{
    std::unordered_map<std::string, MtlEntry> mats;
    std::ifstream in(mtl_path);
    if (!in.is_open())
        return mats;
    const std::filesystem::path dir = mtl_path.parent_path();
    std::string line, current;
    while (std::getline(in, line))
    {
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "newmtl")
            ss >> current;
        else if (tag == "Kd" && !current.empty())
        {
            float r = 0, g = 0, b = 0;
            if (ss >> r >> g >> b) { mats[current].kd = {r, g, b}; mats[current].has_kd = true; }
        }
        else if (tag == "map_Kd" && !current.empty())
        {
            std::string file;
            ss >> file;   // simple filename (no option flags/spaces in our assets)
            if (!file.empty())
                if (std::filesystem::path cand = dir / file; std::filesystem::exists(cand))
                    mats[current].map_kd = cand.string();
        }
    }
    return mats;
}
}  // namespace

namespace
{
int resolve_index(const std::string& index_str, int count)
{
    if (index_str.empty())
        return -1;
    const int raw_index = std::stoi(index_str);
    if (raw_index > 0)
        return raw_index - 1;
    if (raw_index < 0)
        return count + raw_index;
    return -1;
}

int parse_obj_index(const std::string& token, int vertex_count)
{
    const auto slash = token.find('/');
    return resolve_index(token.substr(0, slash), vertex_count);
}

// Parse the texcoord index from a "v/vt/vn" (or "v/vt") face token; -1 when absent.
int parse_obj_uv_index(const std::string& token, int uv_count)
{
    const auto first = token.find('/');
    if (first == std::string::npos)
        return -1;
    const auto second = token.find('/', first + 1);
    const std::string uv_str = token.substr(first + 1, second == std::string::npos ? std::string::npos : second - first - 1);
    return resolve_index(uv_str, uv_count);
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
    std::vector<QVector2D> texcoords;
    QVector3D bb_min;
    QVector3D bb_max;
    bool have_bounds = false;
    std::string line;

    // Faces are grouped by their active `usemtl` so each material becomes a submesh. Insertion order is
    // preserved (accs) so the caller draws groups deterministically; acc_index maps material name → slot.
    std::unordered_map<std::string, MtlEntry> materials;   // from the OBJ's mtllib
    struct Acc { std::string material; std::vector<QVector3D> tris; std::vector<QVector2D> uvs; };
    std::vector<Acc> accs;
    std::unordered_map<std::string, std::size_t> acc_index;
    std::string current_material;   // "" until the first usemtl
    const auto group = [&](const std::string& m) -> Acc&
    {
        if (const auto it = acc_index.find(m); it != acc_index.end())
            return accs[it->second];
        acc_index.emplace(m, accs.size());
        accs.push_back(Acc{m, {}, {}});
        return accs.back();
    };

    while (std::getline(input, line))
    {
        if (line.size() < 2)
            continue;

        std::istringstream stream(line);
        std::string tag;
        stream >> tag;
        if (tag == "mtllib")
        {
            std::string file;
            stream >> file;
            if (!file.empty())
                materials = parse_mtl(mesh_path.parent_path() / file);
        }
        else if (tag == "usemtl")
        {
            stream >> current_material;
        }
        else if (tag == "v")
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
        else if (tag == "vt")
        {
            float u = 0.f, v = 0.f;
            if (!(stream >> u >> v))
                continue;
            texcoords.emplace_back(u, v);
        }
        else if (tag == "f")
        {
            std::vector<int> face_indices;    // position indices
            std::vector<int> face_uvs;        // parallel texcoord indices (-1 if none)
            std::string token;
            while (stream >> token)
            {
                const int index = parse_obj_index(token, static_cast<int>(vertices.size()));
                if (index >= 0 && index < static_cast<int>(vertices.size()))
                {
                    face_indices.push_back(index);
                    face_uvs.push_back(parse_obj_uv_index(token, static_cast<int>(texcoords.size())));
                }
            }

            if (face_indices.size() < 3)
                continue;

            const auto uv_at = [&](int uv_idx) -> QVector2D
            {
                return (uv_idx >= 0 && uv_idx < static_cast<int>(texcoords.size()))
                    ? texcoords[static_cast<std::size_t>(uv_idx)] : QVector2D(0.f, 0.f);
            };
            Acc& acc = group(current_material);
            for (std::size_t i = 1; i + 1 < face_indices.size(); ++i)
            {
                acc.tris.push_back(vertices[static_cast<std::size_t>(face_indices[0])]);
                acc.tris.push_back(vertices[static_cast<std::size_t>(face_indices[i])]);
                acc.tris.push_back(vertices[static_cast<std::size_t>(face_indices[i + 1])]);
                acc.uvs.push_back(uv_at(face_uvs[0]));
                acc.uvs.push_back(uv_at(face_uvs[i]));
                acc.uvs.push_back(uv_at(face_uvs[i + 1]));
            }
        }
    }

    if (accs.empty() || !have_bounds)
        return std::nullopt;

    ObjMeshData out;
    out.bb_min = bb_min;
    out.bb_max = bb_max;
    out.submeshes.reserve(accs.size());
    for (auto& acc : accs)
    {
        ObjSubmesh sm;
        sm.triangles = std::move(acc.tris);
        sm.uvs = std::move(acc.uvs);
        if (const auto it = materials.find(acc.material); it != materials.end())
        {
            sm.diffuse = it->second.kd;
            sm.has_diffuse = it->second.has_kd;
            sm.texture_path = it->second.map_kd;
        }
        if (!sm.triangles.empty())
            out.submeshes.push_back(std::move(sm));
    }
    if (out.submeshes.empty())
        return std::nullopt;
    return out;
}

}  // namespace rc::obj
