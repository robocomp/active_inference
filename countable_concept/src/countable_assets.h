/*
 * countable_assets.h — the display assets a countable instance node publishes (mesh_path / mesh_texture_path).
 *
 * Both are CONVENTIONS on the class folder, not config keys: the mesh is <cls>_concept/meshes/<cls>.obj, and the
 * texture is <cls>_concept/meshes/<cls>_basecolor.jpg WHEN THAT FILE EXISTS (nightstand has one; bed and toilet are
 * coloured by their .mtl and publish none). The published strings are relative to the retina 3D viewer's meshes root
 * (the repo directory that holds the class folders); the existence check is relative to the agent's CWD, which is its
 * class folder (concept_agents.sh `cd`s there, as etc/instance_identity.csv already assumes).
 *
 * Header-only, no Qt / DSR, so a standalone test can include it.
 */
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <system_error>

namespace rc {

inline std::string countable_mesh_path(const std::string& cls)
{
    return cls + "_concept/meshes/" + cls + ".obj";
}

// The texture to publish, or nullopt when the class folder ships none. `class_dir` is the class folder on disk.
inline std::optional<std::string> countable_texture_path(const std::string& cls,
                                                         const std::filesystem::path& class_dir = ".")
{
    const std::string file = cls + "_basecolor.jpg";
    std::error_code ec;
    if (not std::filesystem::is_regular_file(class_dir / "meshes" / file, ec))
        return std::nullopt;
    return cls + "_concept/meshes/" + file;
}

}  // namespace rc
