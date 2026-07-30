#pragma once

#include <Eigen/Core>

#include <string>

struct GraphObjectBox
{
    Eigen::Vector3f min          = Eigen::Vector3f::Zero();  // room-frame AABB (orientation lost)
    Eigen::Vector3f max          = Eigen::Vector3f::Zero();
    Eigen::Vector3f center       = Eigen::Vector3f::Zero();  // room-frame box center
    Eigen::Vector3f half_extents = Eigen::Vector3f::Zero();  // local half-sizes (w/2, d/2, h/2)
    float           yaw_rad      = 0.0f;                     // Z-axis yaw of the object in room frame
    std::string     node_name;                               // DSR node name (empty for non-DSR boxes)
    std::string     category;
    std::string     subtype;                                 // object_subtype ("round"/"square" for tables); "" if unset
    std::string     mesh_path;                               // display OBJ (relative asset path) published by the concept agent; "" ⇒ box/fitted
    std::string     mesh_texture_path;                       // optional base-colour texture (relative); "" ⇒ flat colour
    // Inferred albedo CHROMATICITY (r,g,b summing to ~1) published by the concept agent as mesh_color_rgb,
    // already faded toward neutral grey by the agent's own confidence. Applied as a NORMALISED
    // MULTIPLICATIVE tint against the asset's mean chromaticity — never a replacement, so a multi-material
    // mesh (wood top + metal legs) keeps its authored contrast instead of flattening to one colour, and a
    // brown asset seen as brown does not multiply toward black. has_mesh_color=false ⇒ render the asset's
    // authored .mtl colours untouched, which is the correct default when the agent knows nothing.
    Eigen::Vector3f mesh_color     = Eigen::Vector3f::Zero();
    bool            has_mesh_color = false;
};