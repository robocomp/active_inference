#pragma once
/*
 * canvas_model.h — "show me the robot", "draw object chair_1": the 3-D model a node carries, for the canvas' 3-D page.
 *
 * Where the mesh comes from (the graph is the only source; nothing is hard-coded per class):
 *   robot     the robot node's `path` attribute (robot_concept: "meshes/shadow.obj"), relative to robot_concept/;
 *             metric, drawn as it is.
 *   objects   the node's `mesh_path` (each concept agent publishes its own, e.g. "table_concept/meshes/table.obj"),
 *             relative to the active_inference root. The cortex mesh_path contract: a UNIT model (footprint
 *             x,y ∈ [-0.5,0.5], z ∈ [0,1], front at the node's yaw zero), so it is scaled by the node's width_m /
 *             depth_m / height_m when they are known — the model then has the size the robot believes.
 * Which node: "robot" (or the robot's name), else an exact node name, else the nearest object of that class in the
 * current room (approach.h pick_object: the same object "approach the chair" means).
 *
 * Paths are resolved against the agent's install (bin/ → chat_agent/ → active_inference/), not the cwd.
 * Main thread (graph reads); the OBJ is loaded by rc::obj::load_obj_mesh_data.
 */

#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <QVector3D>

#include <dsr/api/dsr_api.h>

#include "object_facts.h"

namespace rc::chat
{

struct ModelToShow
{
    std::string node;                       ///< the node it belongs to
    std::filesystem::path file;             ///< the resolved .obj
    std::vector<QVector3D> triangles;       ///< flat soup, already scaled
    std::optional<QVector3D> size;          ///< the scale applied (w, d, h) — nullopt = drawn as stored
    std::string description;                ///< "chair_1 — chair_concept/meshes/chair.obj, 0.45 x 0.50 x 0.90 m"
};

[[nodiscard]] std::expected<ModelToShow, std::string> find_model(DSR::DSRGraph& G, const std::vector<ObjectFact>& objects,
                                                                 const std::string& what);

/// The active_inference root: two levels above the binary's directory (chat_agent/bin → active_inference).
[[nodiscard]] std::filesystem::path assets_root();

}   // namespace rc::chat
