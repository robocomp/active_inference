/*
 * door_fitter.h
 *
 * The DSR-bound side of door_concept's fit (mirrors bottle_concept/bottle_fitter.h): the per-door instance map
 * (rc::DoorInstance), ensure_instance's read of the node, and the camera pose the remembered-door look reads. (The
 * existence probes — silhouette, aperture frame, obliquity — moved into the DoorCore: src/core/door_existence.h.)
 *
 * ★The FIT itself — observation (the assigned mask slice → candidate / residual points), inference (support-bank
 * ingest + one recursive belief update), the leaf angle, the pre-belief geometry and read-back — moved into the
 * DSR-free DoorCore (src/core/door_fit.h, Task 4b), which runs it inside DoorCore::step(). SpecificWorker keeps the
 * DSR write-back (CoreWorld::after_fit → publish_door_cycle) and the post-fit epistemic / affordance / Qt steps.
 * Plain class (no Q_OBJECT).
 */

#pragma once

#include <algorithm>

#include <cmath>

#include "../../common/exclusion/exclusion.h"   // rc::exclusion::Claim (SHARED)
#include "../../common/contour_edge/contour_edge_check.h"  // rc::edges::PreparedFrame / contour_edge_support

#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/api/dsr_inner_gaussian_api.h>   // Part B: chain covariance propagation
#include <dsr/api/dsr_camera_api.h>

#include "../../common/contour_edge/contour_edge_project.h"   // rc::edges::ContourSet — the shared silhouette + its null

#include "core/door_fit.h"    // rc::door::core::DoorFit — the fit, in the DSR-free DoorCore
#include "door_config.h"        // rc::DoorConfig
#include "door_instance.h"      // rc::DoorInstance, DoorState
#include "door_model.h"         // DoorModel / DoorModelParams
#include "../../common/mask_ingestor/mask_ingestor.h"
#include "../../common/occlusion/occlusion.h"   // rc::occlusion::{cone_blocks, walls_block} — shared LoS occlusion
#include "door_scene_graph.h"
#include "door_semantic_field.h"   // rc::SemanticProbField (retina's graded posterior)
#include "door_frame_evidence.h"   // rc::door::FrameProbe — the aperture-frame existence probe
#include "door_room_band.h"        // rc::door::RoomBand — where a door sits relative to the room polygon
#include <opencv2/core.hpp>

namespace rc {

// (DoorSilhouette and the existence probes — compute_silhouette_existence, compute_frame_probe, door_view_obliquity,
//  aperture_band_offset — moved into the DSR-free DoorCore: src/core/door_existence.h, Task 5.)

class DoorFitter
{
public:
    // ★THE FIT MOVED INTO THE DoorCore (Task 4b; src/core/door_fit.h). What stays here is what needs the graph: the
    // instance map (rc::DoorInstance = the core's rc::DoorFitState + the DSR-bound affordance handles), the node read
    // of ensure_instance, and the existence probes that still query cortex for the camera (Task 5 moves those).
    using DoorObservation = rc::door::core::DoorObservation;

    DoorFitter(std::shared_ptr<DSR::DSRGraph> graph,
                DSR::InnerEigenAPI* inner_eigen,
                DoorConfig& cfg,
                MaskIngestor* mask_ingestor,
                DoorSceneGraph* scene_graph,
                rc::door::core::DoorFit& fit);

    // Create the instance for a "door_*" node if absent (from prior/RT). Returns true the
    // first time it is created (so the worker can register Qt series / canvas pos). Latches room_id.
    // The node's attributes and its room→door RT are read here; the rest is DoorFit::init_instance.
    bool ensure_instance(const DSR::Node& node, std::uint64_t room_id);

    std::unordered_map<std::uint64_t, DoorInstance>& instances() { return instances_; }
    void forget_node(std::uint64_t id) { instances_.erase(id); }
    // ── Room-containment pose prior ───────────────────────────────────────────────────────────────
    // The room polygon (room frame) is a trusted NOMINAL model (authored from the SVG layout, never fitted),
    // so it is a legitimate strong prior: P(a door's centre outside the walls) ≈ 0. Loaded from the room
    // node's delimiting_polygon_{x,y}. Used to suppress out-of-room births AND remove an instance that a
    // localization glitch placed outside (it can't be reached by the sensor to vacate it — behind a wall).
    // ★ONE copy: the DoorCore's fit holds it (the interior point was never read).
    void set_room_geometry(const Eigen::Vector2f& /*interior*/, std::vector<Eigen::Vector2f> polygon)
    { fit_.set_room_polygon(std::move(polygon)); }
    bool has_room_polygon() const { return fit_.has_room_polygon(); }
    // The walls, for anything that needs them as GEOMETRY rather than as a containment prior — notably the
    // NBV sight test, which must know the wall a door is set into is opaque. See rc::nbv::wall_obstacles.
    const std::vector<Eigen::Vector2f>& room_polygon() const { return fit_.room_polygon(); }
    // camera→room at the current pose (ts=0, main thread); nullopt while the chain does not resolve. Public read of
    // the transform the silhouette uses, for the remembered-door look's absence channel.
    std::optional<Eigen::Matrix4d> camera_in_room() const { return room_T_zed_matrix(); }

    // (the existence probes — silhouette, aperture frame, obliquity, band offset — and the central-image box moved
    //  into the DoorCore: src/core/door_existence.h. They read the cycle's snapshot camera, not cortex.)
    [[nodiscard]] rc::door::RoomBand room_band() const { return fit_.room_band(); }
    // true if q is inside the polygon, or outside by no more than margin_m (tolerance for a wall-hugging door
    // whose centroid noise pokes through the wall). No polygon loaded ⇒ always true (unknown room → no prior).
    bool point_in_room(const Eigen::Vector2f& q, float margin_m = 0.0f) const;
    bool should_log(const DoorInstance& inst) const { return fit_.should_log(inst); }

private:
    // room_T_zed (camera→room). pose_ts_ms pins the room→body hop to the mask's capture time (Nearest RT
    // query); the rigid body→zed mount is always queried latest. 0 → current pose.
    std::optional<Eigen::Matrix4d> room_T_zed_matrix(std::uint64_t pose_ts_ms = 0) const;

    std::shared_ptr<DSR::DSRGraph> G_;
    DSR::InnerEigenAPI*            inner_eigen_ = nullptr;
    DoorConfig&                   cfg_;
    MaskIngestor*                  mask_ingestor_ = nullptr;
    DoorSceneGraph*               scene_graph_   = nullptr;
    rc::door::core::DoorFit&      fit_;            // the DoorCore's fit (init_instance, the room polygon)

    std::unordered_map<std::uint64_t, DoorInstance> instances_;
    std::uint64_t                  room_node_id_ = 0;   // latched per ensure_instance call
};

}  // namespace rc
