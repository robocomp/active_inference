#pragma once

#include <genericworker.h>

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
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

        // Diagnostic breakdown of what the planner sees near a query point, split by SOURCE. Lets the
        // proximity CSV tell a self-stuck (robot hugging its OWN generated geometry — a temp-LiDAR
        // phantom or a virtual escape disc — with no real LiDAR support) apart from a genuine obstacle.
        struct ObstacleProximityDiag
        {
            int   n_temp = 0;                    // live temp-LiDAR obstacles
            int   n_virtual = 0;                 // live virtual escape discs
            float near_temp_m = -1.f;            // nearest temp-LiDAR obstacle edge → query (-1 = none)
            float near_virtual_m = -1.f;         // nearest virtual disc edge → query (-1 = none)
            float near_temp_log_odds = 0.f;      // existence log-odds of that nearest temp obstacle (low ⇒ phantom)
            int   near_temp_missed = 0;          // consecutive missed updates of that nearest temp obstacle
            std::uint64_t near_temp_age_ms = 0;  // now − last_seen_ms of that nearest temp obstacle
        };
        ObstacleProximityDiag obstacle_proximity_diag(const Eigen::Vector2f &query_room,
                                                      std::uint64_t now_ms) const;

        // ATTRIBUTION of the nearest planner obstacle. `nearest_obst_m` alone is a min over FOUR
        // independent sources (concept-object boxes, graph obstacle nodes, residual_concept's
        // occupancy hulls, our own temp/virtual geometry) that each inflate differently and cannot
        // retire one another — so a bare distance can't say WHICH layer is squeezing the robot, and
        // therefore can't say which agent to fix. Walks display_obstacle_polygons_ (kind + label are
        // attached there) rather than obstacle_polygons_, so attribution never depends on the two
        // lists staying index-parallel.
        struct NearestObstacleInfo
        {
            float distance_m = -1.f;                                     // robot centre → nearest polygon edge
            float bearing_deg = 0.f;                                     // robot frame, 0 = ahead, + = right
            ControllerObstacleKind kind = ControllerObstacleKind::Obstacle;
            std::string label;                                           // t_1/c_1/o_2/… ({} for grid hulls)
            bool inside = false;                                         // robot centre INSIDE that polygon
        };
        NearestObstacleInfo nearest_obstacle_info(const Eigen::Vector2f &query_room, float robot_theta) const;

        // RAW-cloud proximity, measured BEFORE the [min_h, max_h] z-band cut that handle_lidar_points
        // applies on the way into lidar_room_buffer_. Everything else on the controller side — the
        // ESDF, nearest_lidar_m, the near-shell columns — reads the ALREADY-FILTERED buffer, so
        // "no LiDAR support" has really meant "no support in the 0.20–1.8 m band" all along. Returns
        // below the band still feed residual_concept's grid, so a low obstacle it legitimately sees is
        // structurally invisible to every controller diagnostic. These fields close that blind spot.
        struct RawCloudProximity
        {
            float distance_m = -1.f;      // nearest return of the UNFILTERED sweep (no z-band, no self-cut)
            float z_m = 0.f;              // its height in the robot frame
            float bearing_deg = 0.f;      // robot frame, 0 = ahead, + = right
            int   below_band_within_1m = 0;   // returns under min_h and within 1 m — what the band THREW AWAY
        };
        const RawCloudProximity &raw_cloud_proximity() const { return raw_cloud_proximity_; }
        const std::optional<std::uint64_t> &last_lidar_timestamp_ms() const { return last_lidar_timestamp_ms_; }
        void clear_published_obstacles();

        // Ingest a raw LiDAR point cloud (lidar frame, from the zero-copy media plane)
        // into the room buffer via the RT tree. lidar_node_name is the graph node the RT
        // transforms are queried against (e.g. "lidar3D"). Dedups by timestamp.
        bool handle_lidar_points(const std::string &lidar_node_name,
                                 std::vector<float> xs,
                                 std::vector<float> ys,
                                 std::vector<float> zs,
                                 std::uint64_t timestamp_ms);
        void update_active_obstacle_polygons(std::uint64_t timestamp_ms, rc::TrajectoryController &path_controller);
        void refresh_temporary_lidar_obstacle(std::uint64_t timestamp_ms,
                                            const ControllerRobotPose &robot_pose,
                                            rc::TrajectoryController &path_controller);
        bool create_temporary_lidar_obstacle(std::uint64_t timestamp_ms,
                                            const ControllerRobotPose &robot_pose,
                                            const Eigen::Vector2f &blockage_center_room,
                                            float blockage_radius_m,
                                            rc::TrajectoryController &path_controller);
        // Inject a purely GEOMETRIC, LOCAL-ONLY obstacle at `center` (room frame). Unlike the
        // temporary LiDAR obstacles above, this needs no LiDAR support (it's built from geometry,
        // not observed points) and is NEVER uploaded to DSR — it is appended to the planner/MPPI
        // obstacle set each cycle and pruned on its TTL. Used by stuck recovery to force a detour
        // around something residual_concept never modelled. Refreshes an existing disc that already
        // covers the same spot instead of stacking duplicates.
        void add_virtual_obstacle(std::uint64_t now_ms, const Eigen::Vector2f &center, float radius_m);
        // Proactive "everything unmodelled is an obstacle" sweep (throttled internally): cluster the
        // recent LiDAR returns that no concept-agent object explains and aren't the floor/walls/robot,
        // and create/refresh one temporary obstacle per cluster. The existing existence-filter / prune
        // / retire-when-modelled machinery then manages their lifecycle (multi-instance).
        void scan_for_unmodelled_obstacles(std::uint64_t timestamp_ms,
                                           const ControllerRobotPose &robot_pose,
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
        // LiDAR height band moved to ControllerParams (temporary_obstacle_min/max_height_m) so the
        // floor-reject cutoff can be tuned from config without a rebuild.

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

        // Raw scan held back one frame for the "draw one frame old" overlay path.
        struct PendingLidarScan
        {
            std::vector<float> xs, ys, zs;
            std::uint64_t ts = 0;
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
        // Single-letter viewer tag prefix per modelled object type: table→t, chair→c, cylinder/bottle→b.
        static std::string object_label_prefix(const std::string &object_type);
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
        // Match-or-create a temporary obstacle from an observation + bump its existence filter. Shared
        // by the reactive blockage path and the proactive scan.
        void ingest_obstacle_observation(const ControllerObstacleObservation &observation, std::uint64_t timestamp_ms);
        // Room delimiting polygon in room frame (for the proactive scan's wall/outside filtering).
        std::vector<Eigen::Vector2f> read_room_polygon_room_frame() const;

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
        RawCloudProximity raw_cloud_proximity_;   // measured on the raw sweep, pre z-band (see the struct)
        std::vector<GraphObstacleRecord> known_graph_obstacles_;
        ControllerPolygons obstacle_polygons_;
        ControllerObstacleVisuals display_obstacle_polygons_;
        std::unordered_map<std::string, int> object_label_counts_;   // per-type object tag counter (t/c/b)
        int obstacle_label_count_ = 0;                               // shared o_N counter (graph + temp)
        std::vector<TemporaryObstacleInstance> temporary_obstacles_;
        std::uint64_t next_temporary_obstacle_id_ = 1;

        // Local-only geometric obstacles (stuck recovery). Planner/MPPI-visible, never DSR-synced.
        struct VirtualObstacle
        {
            Eigen::Vector2f center = Eigen::Vector2f::Zero();
            float radius_m = 0.30f;
            std::uint64_t expires_at_ms = 0;
        };
        std::vector<VirtualObstacle> virtual_obstacles_;
        std::uint64_t last_scan_ms_ = 0;   // throttle for scan_for_unmodelled_obstacles
        mutable std::optional<std::uint64_t> last_lidar_timestamp_ms_;   // stamp of the scan actually buffered
        std::optional<std::uint64_t> newest_raw_lidar_ts_;               // newest RAW stamp (dedup + period)
        std::optional<PendingLidarScan> pending_lidar_scan_;             // held-back frame (draw-one-frame-old)
        mutable std::uint64_t lidar_period_ms_ = 100;
        mutable std::string obstacle_debug_report_;
        mutable std::string graph_object_debug_report_;
        mutable std::string current_obstacles_debug_report_;
};