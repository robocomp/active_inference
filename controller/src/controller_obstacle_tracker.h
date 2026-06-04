#pragma once

#include <genericworker.h>

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
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
        const ControllerObstacleVisuals &display_obstacle_polygons() const { return display_obstacle_polygons_; }
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
        static constexpr int kRememberedEdgeCount = 4;
        static constexpr int kRememberedSlotsPerEdge = 8;
        static constexpr int kRememberedPointsPerSlot = 1;
        static constexpr float kRememberedEdgeDistanceThreshold = 0.12f;
        static constexpr float kRememberedRfeAlpha = 0.97f;
        static constexpr float kRememberedRfeMax = 0.40f;
        static constexpr int kRemovalEvidenceMinMissedUpdates = 4;
        static constexpr int kRemovalEvidenceMinPenetratingPoints = 6;
        static constexpr float kRemovalSupportDistanceThreshold = 0.14f;
        static constexpr float kRemovalPenetrationMargin = 0.05f;
        static constexpr float kRemovalCrossAxisMargin = 0.08f;
        static constexpr float kPublishedObstacleHeightM = 0.8f;
        static constexpr float kTemporaryObstacleMinHeightAboveFloorM = 0.20f;
        static constexpr float kTemporaryObstacleMaxHeightAboveFloorM = 1.8f;

        struct RememberedPoint
        {
            Eigen::Vector2f local_point = Eigen::Vector2f::Zero();
            float rfe = 0.f;
        };

        struct RememberedEdgeSlot
        {
            int edge_index = 0;
            int slot_index = 0;
            float distance_to_edge = 0.f;
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

        struct GraphObstacleRecord
        {
            std::uint64_t node_id = 0;
            std::string node_name;
            ControllerObstacleState state;
            ControllerObstacleKind kind = ControllerObstacleKind::Obstacle;
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
        static std::array<Eigen::Vector2f, kRememberedEdgeCount> make_local_obstacle_corners(const ControllerObstacleState &state);
        static std::optional<RememberedEdgeSlot> classify_remembered_edge_slot(const ControllerObstacleState &state,
                                                                            const Eigen::Vector2f &local_point);
        static int remembered_slot_index(const RememberedEdgeSlot &slot);
        static Eigen::Vector2f estimate_obstacle_center(const ControllerObstacleObservation &observation,
                                                        float width_m,
                                                        float depth_m);
        static Eigen::Vector2f estimate_initial_obstacle_center(const ControllerObstacleObservation &observation);
        static std::string published_obstacle_name(std::uint64_t obstacle_id);
        static float distance_visibility_scale(const ControllerParams *params,
                                            const ControllerRobotPose &robot_pose,
                                            const ControllerObstacleState &state);
        static void recompute_observation_summary(ControllerObstacleObservation &observation,
                                                float padding_m,
                                                float occlusion_depth_m);
        void update_remembered_points(TemporaryObstacleInstance &instance,
                                    const ControllerObstacleObservation &observation);
        float size_hardening_evidence(const TemporaryObstacleInstance &instance) const;
        bool has_compelling_absence_evidence(std::uint64_t timestamp_ms,
                                            const ControllerRobotPose &robot_pose,
                                            const TemporaryObstacleInstance &instance);
        static float obstacle_sdf(const ControllerObstacleState &state, const Eigen::Vector2f &point);
        static bool point_explained_by_obstacle(const ControllerObstacleState &state,
                            const Eigen::Vector2f &point,
                            float margin_m);
        static bool obstacles_overlap(const ControllerObstacleState &lhs,
                          const ControllerObstacleState &rhs,
                          float margin_m);
        static float obstacle_distance_shape_metric(const ControllerObstacleState &reference,
                                const ControllerObstacleState &candidate,
                                bool candidate_has_shape);
        static Eigen::Vector2f to_local_point(const ControllerObstacleState &state, const Eigen::Vector2f &point);
        static Eigen::Vector2f to_room_point(const ControllerObstacleState &state, const Eigen::Vector2f &point);
        void sync_temporary_obstacles_to_dsr(std::uint64_t timestamp_ms);
        void delete_published_obstacle_node(const TemporaryObstacleInstance &instance);
        void prune_expired_temporary_obstacles(std::uint64_t timestamp_ms);
        void retire_temporary_obstacles_explained_by_graph();
        std::optional<std::size_t> match_temporary_obstacle(const ControllerObstacleObservation &observation) const;

        ControllerPolygons read_obstacle_polygons(std::uint64_t timestamp_ms);
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
        std::vector<GraphObstacleRecord> known_graph_obstacles_;
        ControllerPolygons obstacle_polygons_;
        ControllerObstacleVisuals display_obstacle_polygons_;
        std::vector<TemporaryObstacleInstance> temporary_obstacles_;
        std::uint64_t next_temporary_obstacle_id_ = 1;
        mutable std::optional<std::uint64_t> last_lidar_timestamp_ms_;
        mutable std::uint64_t lidar_period_ms_ = 100;
        mutable std::string obstacle_debug_report_;
        mutable std::string graph_object_debug_report_;
        mutable std::string current_obstacles_debug_report_;
};