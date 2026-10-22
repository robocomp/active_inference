/*
 * common/mask_ingestor/mask_nodes.h
 *
 * ONE place that answers "which DSR nodes carry YOLO masks, and in which camera frame?".
 *
 * The retina publishes ONE mask node PER CAMERA ("masks_zed" under "zed", "masks_ricoh" under "ricoh"),
 * each holding only that camera's slices in THAT camera's frame. Consumers must NOT hard-code those names:
 * they enumerate the camera nodes (type "rgbd") and take each camera's child `semantic_grid` node that
 * carries `mask_frame_id` (the camera's other semantic_grid children — "semantic", "skeleton" — do not).
 * A third camera therefore appears here with no consumer change.
 *
 * Header-only on purpose: the controller and viewer3d use it without linking ai_common::perception.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <optional>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_camera_api.h>

namespace rc::masks {

struct MaskNodeRef
{
    std::uint64_t id = 0;       // the mask node
    std::string   name;         // the mask node's name (diagnostics only)
    std::string   camera;       // the PARENT camera node's name == the frame its slices are expressed in
    std::uint64_t camera_id = 0;
};

// Every mask node currently in the graph, one per camera that has one. Order follows get_nodes_by_type("rgbd").
[[nodiscard]] inline std::vector<MaskNodeRef> find_mask_nodes(DSR::DSRGraph& G)
{
    std::vector<MaskNodeRef> out;
    for (const auto& cam : G.get_nodes_by_type("rgbd"))
        for (const auto& [key, edge] : cam.fano())
        {
            if (edge.type() != "RT")
                continue;
            const auto child = G.get_node(key.first);
            if (not child.has_value() or child->type() != "semantic_grid")
                continue;
            if (not G.get_attrib_by_name<mask_frame_id_att>(child.value()).has_value())
                continue;   // a semantic_grid that is not a mask node ("semantic", "skeleton", …)
            out.push_back({child->id(), child->name(), cam.name(), cam.id()});
        }
    return out;
}

// Is this camera the FRONT (pinhole, depth-carrying) one? Decided from the camera's own projection model
// in the graph, never from its name: a panoramic (equirectangular / cylindrical) camera is the PERIPHERAL one.
[[nodiscard]] inline bool camera_is_front(DSR::DSRGraph& G, std::uint64_t camera_id)
{
    const auto cam = G.get_node(camera_id);
    if (not cam.has_value())
        return true;
    try
    {
        const auto api = G.get_camera_api(cam.value());
        return api and api->get_projection_model() == DSR::CameraAPI::ProjectionModel::Pinhole;
    }
    catch (...) { return true; }
}

// The mask node of the first FRONT camera (the one whose frame counter paces "one look per camera frame").
[[nodiscard]] inline std::optional<MaskNodeRef> find_front_mask_node(DSR::DSRGraph& G)
{
    for (const auto& ref : find_mask_nodes(G))
        if (camera_is_front(G, ref.camera_id))
            return ref;
    return std::nullopt;
}

}   // namespace rc::masks
