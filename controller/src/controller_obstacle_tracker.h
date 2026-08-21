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
        // residual_concept's occupancy, as CELLS — what the planner marks directly (see the decode
        // site for why the polygon round trip was removed).
        const std::vector<Eigen::Vector2f> &residual_cells() const { return residual_cells_; }
        float residual_cell_size_m() const { return residual_cell_size_m_; }
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
        // How far the NEWEST room←robot RT block leads the scan we just registered, in ms
        // (positive ⇒ a pose newer than the scan exists, so the query could bracket; negative ⇒ the
        // pose feed is BEHIND the scan and every query clamps to a stale block). Empty until a scan
        // has been processed or when the ring carries no timestamps. See update_rt_block_lead().
        const std::optional<std::int64_t> &rt_block_lead_ms() const { return rt_block_lead_ms_; }
        // Signed ms the twist correction walked the pose for the last registered scan (0 = the RT
        // query was inside the ring, so nothing needed repairing). Pairs with rt_block_lead_ms().
        std::int64_t rt_twist_fix_dt_ms() const { return rt_twist_fix_dt_ms_; }
        // Twist-accuracy probe over one lidar period: the horizon used (0 = probe did not run this
        // scan) and the position / heading residual against the RT tree. This is what says whether
        // the one-frame hold in handle_lidar_points can eventually be dropped (it cannot yet — see
        // the note there; the hold also smooths, which this does not measure).
        std::int64_t twist_pred_dt_ms() const { return twist_pred_dt_ms_; }
        const std::optional<float> &twist_pred_err_m() const { return twist_pred_err_m_; }
        const std::optional<float> &twist_pred_err_deg() const { return twist_pred_err_deg_; }
        void clear_published_obstacles();

        // Ingest a raw LiDAR point cloud (lidar frame, from the zero-copy media plane)
        // into the room buffer via the RT tree. lidar_node_name is the graph node the RT
        // transforms are queried against (e.g. "helios"). Dedups by timestamp.
        // plane_ids/plane_stamps come from rc::media::LidarSweep: the cloud is a MERGE of several
        // LiDARs captured at different instants, so each plane is registered with the room←robot
        // pose at ITS OWN stamp instead of one pose at the merged maximum. Both may be empty, which
        // restores the single-pose behaviour (a one-plane sweep, or a caller that has no per-plane
        // information).
        bool handle_lidar_points(const std::string &lidar_node_name,
                                 std::vector<float> xs,
                                 std::vector<float> ys,
                                 std::vector<float> zs,
                                 std::uint64_t timestamp_ms,
                                 std::vector<std::uint8_t> plane_ids = {},
                                 std::vector<std::int64_t> plane_stamps = {});
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
            // Carried with the held-back scan: the per-plane registration below must use the stamps
            // of the frame it actually processes, not the one that arrived meanwhile.
            std::vector<std::uint8_t> plane_ids;
            std::vector<std::int64_t> plane_stamps;
        };

        // Read the room←robot RT ring (newest/oldest block stamps) and the body twist published on
        // that same edge, and record how far the newest block leads `scan_ts`. The lead itself is a
        // diagnostic; the ring bounds + twist are what twist_corrected() below acts on.
        void update_rt_block_lead(std::uint64_t scan_ts);
        // Walk a CLAMPED room←robot pose along the published twist onto `target_ts`. Returns the
        // pose unchanged when the RT query was inside the ring (already interpolated) or when the
        // twist is unavailable. Writes the applied Δt (ms, signed; 0 = untouched) if asked.
        Eigen::Matrix4d twist_corrected(const Eigen::Matrix4d &room_T_robot,
                                        std::uint64_t target_ts,
                                        std::int64_t *applied_dt_ms = nullptr) const;
        // Exp(ξ·Δt) for the cached body twist, Δt in ms and signed.
        Eigen::Matrix4d twist_delta(std::int64_t dt_ms) const;
        // Measure how well the twist predicts ONE LIDAR PERIOD of motion, against the RT tree itself.
        // Health monitor for the twist the correction relies on — see the definition.
        void update_twist_prediction_error(std::uint64_t scan_ts);

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
        std::vector<Eigen::Vector2f> residual_cells_;
        float residual_cell_size_m_ = 0.f;
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
        std::optional<std::int64_t> rt_block_lead_ms_;                   // newest RT block − buffered scan stamp
        std::optional<std::uint64_t> rt_block_newest_ts_, rt_block_oldest_ts_;   // RT ring bounds
        float rt_twist_adv_ = 0.f, rt_twist_side_ = 0.f, rt_twist_rot_ = 0.f;    // body twist, robot frame
        bool rt_twist_valid_ = false;
        std::int64_t rt_twist_fix_dt_ms_ = 0;                            // Δt the last correction walked
        std::int64_t twist_pred_dt_ms_ = 0;                              // horizon the probe used (0 = not run)
        std::optional<float> twist_pred_err_m_, twist_pred_err_deg_;     // twist-vs-RT residual over it
        std::optional<PendingLidarScan> pending_lidar_scan_;             // held-back frame (draw-one-frame-old)
        mutable std::uint64_t lidar_period_ms_ = 100;
        mutable std::string obstacle_debug_report_;
        mutable std::string graph_object_debug_report_;
        mutable std::string current_obstacles_debug_report_;
};