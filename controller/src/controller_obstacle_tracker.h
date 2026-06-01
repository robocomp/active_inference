#pragma once

#include <genericworker.h>

#include <functional>
#include <optional>
#include <vector>

#include "controller_obstacle_model.h"
#include "controller_runtime_types.h"
#include "trajectory_controller.h"

class ControllerObstacleTracker
{
public:
    void set_params(const ControllerParams *params);
    void set_dependencies(std::shared_ptr<DSR::DSRGraph> graph,
                          DSR::InnerEigenAPI *inner_eigen_api,
                          const ControllerGraphState *graph_state);
    void set_graph_layout_callback(std::function<void()> callback);

    rc::LidarPointBuffer *lidar_buffer() { return &lidar_room_buffer_; }
    const ControllerPolygons &obstacle_polygons() const { return obstacle_polygons_; }
    ControllerPolygons temporary_obstacle_rfe_points() const;
    const std::optional<std::uint64_t> &last_lidar_timestamp_ms() const { return last_lidar_timestamp_ms_; }
    void clear_published_obstacles();

    bool handle_lidar_node(DSR::Node node_copy);
    void update_active_obstacle_polygons(std::uint64_t timestamp_ms, rc::TrajectoryController &path_controller);
    void refresh_temporary_lidar_obstacle(std::uint64_t timestamp_ms,
                                          const ControllerRobotPose &robot_pose,
                                          rc::TrajectoryController &path_controller);
    bool create_temporary_lidar_obstacle(std::uint64_t timestamp_ms,
                                         const ControllerRobotPose &robot_pose,
                                         const Eigen::Vector2f &blockage_center_room,
                                         float blockage_radius_m,
                                         rc::TrajectoryController &path_controller);

private:
    struct RememberedPoint
    {
        Eigen::Vector2f local_point = Eigen::Vector2f::Zero();
        float rfe = 0.f;
    };

    struct TemporaryObstacleInstance
    {
        std::uint64_t id = 0;
        std::uint64_t published_node_id = 0;
        ControllerObstacleModel model;
        std::vector<RememberedPoint> remembered_points;
        float existence_log_odds = 0.f;
        std::uint64_t expires_at_ms = 0;
        std::uint64_t last_seen_ms = 0;
        int missed_updates = 0;
        float free_energy = 0.f;
    };

    std::vector<Eigen::Vector2f> read_temporary_obstacle_points(std::uint64_t timestamp_ms,
                                                                const ControllerRobotPose &robot_pose,
                                                                const Eigen::Vector2f &region_center_room,
                                                                float region_radius_m,
                                                                bool forward_only);
    std::optional<ControllerObstacleObservation> build_temporary_obstacle_observation(std::uint64_t timestamp_ms,
                                                                                      const ControllerRobotPose &robot_pose,
                                                                                      const Eigen::Vector2f &region_center_room,
                                                                                      float region_radius_m,
                                                                                      bool forward_only);
    ControllerObstacleModelParams make_model_params() const;
    ControllerObstacleObservation augment_with_remembered_points(const ControllerObstacleObservation &observation,
                                                                 const TemporaryObstacleInstance &instance) const;
    void update_remembered_points(TemporaryObstacleInstance &instance,
                                  const ControllerObstacleObservation &observation);
    bool has_compelling_absence_evidence(std::uint64_t timestamp_ms,
                                         const ControllerRobotPose &robot_pose,
                                         const TemporaryObstacleInstance &instance);
    static float obstacle_sdf(const ControllerObstacleState &state, const Eigen::Vector2f &point);
    static Eigen::Vector2f to_local_point(const ControllerObstacleState &state, const Eigen::Vector2f &point);
    static Eigen::Vector2f to_room_point(const ControllerObstacleState &state, const Eigen::Vector2f &point);
    void sync_temporary_obstacles_to_dsr(std::uint64_t timestamp_ms);
    void delete_published_obstacle_node(const TemporaryObstacleInstance &instance);
    void prune_expired_temporary_obstacles(std::uint64_t timestamp_ms);
    std::optional<std::size_t> match_temporary_obstacle(const ControllerObstacleObservation &observation) const;

    ControllerPolygons read_obstacle_polygons(std::uint64_t timestamp_ms) const;
    std::vector<Eigen::Vector3f> read_recent_lidar_points_in_room(std::uint64_t timestamp_ms, int max_scans);
    ControllerPolygon make_obstacle_polygon(const Eigen::Vector2f &center,
                                            float yaw,
                                            float width_m,
                                            float depth_m) const;

    const ControllerParams *params_ = nullptr;
    std::shared_ptr<DSR::DSRGraph> graph_;
    DSR::InnerEigenAPI *inner_eigen_api_ = nullptr;
    std::unique_ptr<DSR::RT_API> rt_api_;
    const ControllerGraphState *graph_state_ = nullptr;
    std::function<void()> graph_layout_callback_;
    rc::LidarPointBuffer lidar_room_buffer_{5};
    ControllerPolygons obstacle_polygons_;
    std::vector<TemporaryObstacleInstance> temporary_obstacles_;
    std::uint64_t next_temporary_obstacle_id_ = 1;
    mutable std::optional<std::uint64_t> last_lidar_timestamp_ms_;
    mutable std::uint64_t lidar_period_ms_ = 100;
    mutable std::string obstacle_debug_report_;
};