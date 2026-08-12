/*
 * residual_scene_graph.h
 *
 * DSR node/RT I/O layer for residual_concept. Owns the distributed-graph reads/writes for the residual
 * obstacles:
 *   - birthing "residual_N" nodes of DSR type "obstacle" from tracker detections
 *     (create_instance_from_detection), parented to the room (a residual rests on the floor plane),
 *   - writing the fitted footprint back (width_m/depth_m/height_m attrs + the room→node RT edge carrying
 *     yaw and the Laplace covariance) so the controller's planner avoids it via read_obstacle_polygons,
 *   - publishing the convex-hull footprint (Layer B) as an extra attr for a future planner upgrade.
 *
 * Node type is "obstacle" (what the controller already consumes); the agent owns only the "residual_*"
 * named ones. Much leaner than BottleSceneGraph — no table/support-surface lookups, no mesh, no camera.
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_rt_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/api/dsr_inner_gaussian_api.h>   // chain covariance propagation

#include "residual_config.h"
#include "residual_instance.h"

namespace rc
{

class ResidualSceneGraph
{
public:
    ResidualSceneGraph(std::shared_ptr<DSR::DSRGraph> graph,
                       DSR::RT_API* rt_api,
                       DSR::InnerEigenAPI* inner_eigen,
                       ResidualConfig& cfg,
                       std::function<void()> relayout = {});

    // BIRTH (InstanceTracker): create a fresh auto-named "residual_<N>" node of type "obstacle" at a
    // detection centroid with the cluster's seed footprint, parented to the room. Returns the new id (0 fail).
    std::uint64_t create_instance_from_detection(const Eigen::Vector3f& centroid_room,
                                                 float yaw, float w, float d, float height,
                                                 std::uint64_t room_node_id, std::uint64_t timestamp_ms);

    // Publish the instance's fitted footprint to its node (width_m/depth_m/height_m + FE + hull) and RT edge
    // (pose + yaw + Laplace covariance) — only past the geometry dead-band.
    void step_write_model(ResidualInstance& inst, DSR::Node& node, float free_energy,
                          std::uint64_t room_node_id, std::uint64_t timestamp_ms);

    // Chain/localization cov J·Σ_chain·Jᵀ (source frame → room, capture-stamp pinned) added to the
    // published RT-edge cov, via InnerGaussianAPI. Off until set.
    void set_chain_cov_source(DSR::InnerGaussianAPI* gaussian, std::string source_frame, bool enabled);

    // Sensor origin (room xy) this cycle → the conservative inflation grows AWAY from it (occluded side),
    // pinning the near face to the observed surface. Set by the worker each cycle before publishing.
    void set_sensor_origin(const Eigen::Vector3f& origin_room) { sensor_xy_ = origin_room.head<2>(); }

private:
    void write_rt_pose(ResidualInstance& inst, std::uint64_t room_node_id, std::uint64_t timestamp_ms);

    std::shared_ptr<DSR::DSRGraph> G_;
    DSR::RT_API*        rt_api_      = nullptr;
    DSR::InnerEigenAPI* inner_eigen_ = nullptr;
    DSR::InnerGaussianAPI* gaussian_ = nullptr;
    std::string         chain_src_frame_;
    bool                chain_cov_enabled_ = false;
    ResidualConfig&     cfg_;
    Eigen::Vector2f     sensor_xy_ = Eigen::Vector2f::Zero();   // sensor origin (room xy), for directional inflation
    std::function<void()> relayout_;

    int name_high_water_ = 0;   // highest residual_<N> ever issued; never decreases — see create_instance_from_detection
};

}  // namespace rc
