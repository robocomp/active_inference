/*
 * sensor_inputs — the INPUT LAYER of self_calibration.
 *
 * Everything the calibration agent reads, as small plain-data structs, each with its SOURCE timestamp, so the
 * later stages (trajectory harvester, floor / wall / camera-vs-LiDAR factors) consume it without touching DDS
 * or the graph:
 *
 *   media plane (subscribers ONLY through the shared factories in common/media_transport):
 *     helios, bpearl  -> LidarIn    points in the SENSOR (device) frame + the CURRENT graph mount body<-sensor
 *     zed rgb, depth  -> ImageIn    raw bytes, DECIMATED to camera_rate_hz
 *     ricoh panorama  -> ImageIn    raw bytes, DECIMATED to camera_rate_hz
 *     imu             -> ImuIn      full rate
 *   graph (polled, NO DSR signals):
 *     robot node      -> OdometryIn robot_current_{advance,side,angular}_speed (+ variance, stamps)
 *     robot<->room RT -> PoseIn     pose + 6x6 rt_covariance (as stored)
 *     room node       -> PolygonIn  delimiting_polygon_x/y
 *
 * THREADING: single-threaded. poll() runs on the main thread from the Operating compute tick; the DDS readers
 * are drained non-blockingly there. No worker thread, so nothing crosses a thread boundary and there is no
 * shared cv::Mat: images are copied into an owned std::vector<uint8_t> (never a cv::Mat handle).
 *
 * WHAT IS (AND IS NOT) A GATE. camera_rate_hz is a COMPUTE BUDGET: frames beyond it are drained without being
 * copied, because in this repo the cost of a camera stream is the conversion, not the drain. It says nothing
 * about the data. The stall timeout per stream is the one unavoidable time constant (a stream that has stopped
 * has to be declared stopped somewhere): it only marks the stream stalled and logs; it never edits a value.
 */
#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace DSR { class DSRGraph; class InnerEigenAPI; class RT_API; }
namespace rc::media { class LidarSubscriber; class MediaSubscriber; class Image360Subscriber; class ImuSubscriber; }

namespace sc
{

// ───────────────────────────── plain-data outputs ─────────────────────────────

struct LidarIn
{
    std::string                  sensor;            // "helios" | "bpearl"  (also the DSR frame the points live in)
    std::uint64_t                frame_id = 0;
    std::int64_t                 stamp_ms = 0;      // producer CAPTURE stamp
    std::vector<Eigen::Vector3f> points_sensor;     // device frame, metres
    // body<-sensor from the graph AT HAND-OVER (real stamp => never the InnerEigenAPI ts==0 cache). Later
    // factors estimate a CORRECTION to this mount, so it travels with the sweep. nullopt = chain unresolved.
    std::optional<Eigen::Affine3f> body_from_sensor;

    // The same mount into the ROBOT node's frame (Shadow: origin on the floor, z up), which the floor and
    // vertical mount factors need (common/sensor_calib/mount_factors.h). nullopt until the robot node is known.
    std::optional<Eigen::Affine3f> robot_from_sensor;
    // The robot pose in the room AT THIS SWEEP'S STAMP (TimeQuery::Extrapolated on the robot<->room edge ring:
    // interpolated inside it, walked along the published twist past its ends). nullopt when there is no room
    // or the ring answered Stale/Clamped, i.e. no pose evidence for this instant.
    std::optional<Eigen::Affine3d> room_T_robot;
    double pose_walk_ms = 0.0;                      // how far the extrapolation walked (0 inside the ring)

    // points_sensor mapped through body_from_sensor / robot_from_sensor; empty when that mount is unresolved.
    [[nodiscard]] std::vector<Eigen::Vector3f> points_body() const;
    [[nodiscard]] std::vector<Eigen::Vector3f> points_robot() const;
};

// Pinhole intrinsics of the camera's RGB MODEL (CameraAPI: centre = size/2, no separate cx/cy). A depth image of a
// different size is served by scaling these uniformly (zed_floor_stage depth_to_points). 0 = unknown.
struct PinholeIntrinsics { double fx = 0, fy = 0, width = 0, height = 0; };

struct ImageIn
{
    std::string                sensor;              // "zed" | "ricoh"
    std::string                stream;              // "rgb" | "depth" | "rgb360"
    std::uint64_t              frame_id = 0;
    std::int64_t               stamp_ms = 0;
    std::uint32_t              width = 0, height = 0, step = 0, format = 0;   // format: producer's FORMAT_* code
    std::vector<std::uint8_t>  data;                // OWNED deep copy of the valid bytes
    std::optional<Eigen::Affine3f> body_from_sensor;
    std::optional<Eigen::Affine3f> robot_from_sensor;   // as LidarIn: the mount into the ROBOT (Shadow) frame; nullopt = unresolved
    PinholeIntrinsics              intrinsics;           // pinhole cameras (zed) only, read once per sensor from the graph
    std::optional<Eigen::Affine3d> room_T_robot;         // robot pose in the room at this image's stamp (as LidarIn); nullopt = none
};

struct ImuIn
{
    std::uint64_t frame_id = 0;
    std::int64_t  stamp_ms = 0, sim_stamp_ms = 0;
    Eigen::Vector3f acc  = Eigen::Vector3f::Zero();    // m/s^2, sensor frame
    Eigen::Vector3f gyro = Eigen::Vector3f::Zero();    // rad/s, sensor frame
    float roll = 0.f, pitch = 0.f, yaw = 0.f;
    float gyro_var = 0.f, acc_var = 0.f;
};

struct OdometryIn
{
    float adv = 0.f, side = 0.f, rot = 0.f;            // robot_current_*_speed (body +Y forward)
    std::int64_t source_ts_ms = 0;                     // robot_current_speed_timestamp
    std::int64_t sim_ts_ms = 0;                        // 0 when absent
    bool         simulated = false;
    // Producer's own velocity variance {adv, side, rot}; -1 = absent or "unknown" (as room_concept reads it).
    float var_adv = -1.f, var_side = -1.f, var_rot = -1.f;
};

// The localiser's estimate, named by FRAME. room_concept stores the RT edge robot -> room (the room is a child of
// the robot), i.e. the ROOM'S pose in the ROBOT frame, and writes its (x, y, theta) covariance through the Jacobian
// of that inversion (room_scene_graph.cpp dsr_update_pose, the `J` block, ~l.618-628). What every consumer here wants
// is the other quantity, the robot in the room, so BOTH the transform and the covariance are handed over already
// converted: cov_room_robot = J' * Sigma_stored * J'^T with J' the Jacobian of the (self-inverse) map
// (t, th) -> (-R(th)^T t, -th) evaluated at the STORED pose. x/y mix with the rotation through the lever arm; the
// theta variance is invariant. Other slots (z, rx, ry) are copied unchanged (the writer fills only slots 0, 1, 5).
struct PoseIn
{
    std::uint64_t robot_id = 0, room_id = 0;
    Eigen::Affine3d room_T_robot = Eigen::Affine3d::Identity();   // robot pose in the room: maps robot-frame points to the room frame
    // 6x6 ROW-MAJOR over [x y z rx ry rz], covariance of room_T_robot's (x, y, yaw) in the ROOM frame; SE(2) slots
    // are 0 (x), 1 (y), 5 (yaw): heading variance at [35].
    std::array<float, 36> cov_room_robot{};
    bool          has_cov = false;
    std::int64_t  read_stamp_ms = 0;                   // wall clock of the read (the edge carries no capture stamp)
};

// SE(2) covariance change of frame for the robot<->room inversion (see PoseIn). `stored_pose_*` is the pose the
// covariance is attached to; the result is the covariance of the inverse pose. Pure, pinned by the selftest.
[[nodiscard]] Eigen::Matrix3d invert_pose_covariance_se2(const Eigen::Matrix3d& cov_xyt, double x, double y, double theta);

struct PolygonIn
{
    std::uint64_t room_id = 0;
    std::vector<Eigen::Vector2f> vertices;             // room frame
    float ceiling_m = 0.f;                             // room node room_height (m above the floor); 0 = absent
    float ceiling_sigma_m = 0.f;                       // room_height_sigma; 0 = not measured (a configured height)
};

// ───────────────────────────── pure, testable parts ─────────────────────────────

// Odometry from the already-read attribute values. nullopt unless adv, side, rot and a positive timestamp are all
// present (the same preconditions as room_concept's slot). `variance` is the 3-vector attribute if present.
[[nodiscard]] std::optional<OdometryIn> make_odometry(std::optional<float> adv, std::optional<float> side,
                                                      std::optional<float> rot, std::optional<std::uint64_t> ts,
                                                      std::optional<std::uint64_t> sim_ts, std::optional<bool> simulated,
                                                      const std::optional<std::vector<float>>& variance);

// Rate decimator on SOURCE stamps. A fixed schedule (next_due advances by one period per accepted frame), so the
// long-run rate is exactly rate_hz regardless of how the input rate aliases against the period. Resyncs to the
// stream after a gap longer than one period. Compute budget, not a data gate (see file header).
class RateDecimator
{
public:
    explicit RateDecimator(double rate_hz) : period_ms_(rate_hz > 0.0 ? 1000.0 / rate_hz : 0.0) {}
    [[nodiscard]] bool accept(std::int64_t stamp_ms);
private:
    double period_ms_;
    double next_due_ms_ = 0.0;
    bool   started_ = false;
};

// Arrival bookkeeping for one stream: rate over the last window, and stalled / recovered transitions.
class StreamMonitor
{
public:
    using Clock = std::chrono::steady_clock;
    StreamMonitor(std::string name, std::chrono::milliseconds stall_after) : name_(std::move(name)), stall_after_(stall_after) {}

    void note_arrival(Clock::time_point now)
    {
        if (not window_started_) { window_started_ = true; window_start_ = now; window_count_ = 0; }   // first frame opens the window
        ++window_count_; ++total_; last_arrival_ = now; ever_ = true;
    }

    // Re-evaluate at `now`. Returns a message to LOG exactly when the stalled state changes (else empty).
    [[nodiscard]] std::string update(Clock::time_point now);
    // Rate over the window ending at `now`, then restart the window.
    [[nodiscard]] double take_rate_hz(Clock::time_point now);

    [[nodiscard]] bool stalled() const { return stalled_; }
    [[nodiscard]] bool ever_seen() const { return ever_; }
    [[nodiscard]] std::uint64_t total() const { return total_; }
    [[nodiscard]] const std::string& name() const { return name_; }
private:
    std::string name_;
    std::chrono::milliseconds stall_after_;
    Clock::time_point last_arrival_{}, window_start_{};
    bool window_started_ = false, ever_ = false, stalled_ = false;
    std::uint64_t window_count_ = 0, total_ = 0;
};

// ───────────────────────────── the input layer ─────────────────────────────

struct InputsConfig
{
    double camera_rate_hz = 1.5;                       // compute budget for zed rgb/depth and ricoh; <=0 disables cameras
    int    stall_lidar_ms = 2000, stall_camera_ms = 5000, stall_imu_ms = 1000;
    int    log_period_ms  = 10000;                     // stream rates are logged at most this often
    int    discovery_period_ms = 5000;                 // retry period for subscribers that could not be created
    std::size_t max_lidar_queue = 8, max_image_queue = 4, max_imu_queue = 4096, max_odom_queue = 4096;   // memory bounds
};

class SensorInputs
{
public:
    // `graph` must outlive this object (destroy it BEFORE the graph, from the main thread).
    SensorInputs(std::shared_ptr<DSR::DSRGraph> graph, InputsConfig cfg);
    ~SensorInputs();
    SensorInputs(const SensorInputs&) = delete;
    SensorInputs& operator=(const SensorInputs&) = delete;

    // Main thread, Operating tick only. Creates missing subscribers (throttled), drains every stream
    // non-blockingly, polls the graph reads, updates the stall state and logs rates.
    void poll();

    // Hand-over queues (bounded, drop-oldest). The consumer takes and owns what it takes. Nothing here withholds data
    // of a stalled stream: the consumer must check stalled(key)/any_stalled() before using what it takes.
    [[nodiscard]] std::vector<LidarIn>    take_lidar();
    [[nodiscard]] std::vector<ImageIn>    take_images();
    [[nodiscard]] std::vector<ImuIn>      take_imu();
    // Odometry is POLLED, so a sample can be missed when a tick runs late. Never count samples: integrate the
    // velocity piecewise-constant over successive source_ts_ms deltas.
    [[nodiscard]] std::vector<OdometryIn> take_odometry();

    // Latest graph reads (nullopt until first resolvable).
    [[nodiscard]] const std::optional<PoseIn>&    pose() const { return pose_; }
    [[nodiscard]] const PolygonIn&                polygon() const { return polygon_; }

    // body<-sensor from the graph now, at a real stamp (ts!=0 bypasses the ts==0 cache). nullopt if unresolved.
    [[nodiscard]] std::optional<Eigen::Affine3f> body_from_sensor(const std::string& sensor, std::int64_t stamp_ms) const;

    // True when SOME stream that has delivered before is currently stalled.
    [[nodiscard]] bool any_stalled() const;
    // Per-stream rate over the window since the last call (restarts the window), plus graph-read state.
    [[nodiscard]] std::string status_line();
    // True when this stream ("helios","zed/rgb","imu",...) has delivered before and has now stopped.
    // Read-only view for the dashboard: cumulative arrivals and the stalled flag per stream. Unlike status_line() it
    // restarts no rate window, so a display can derive its own rate from the totals without disturbing the logger.
    struct StreamState { std::string key; std::uint64_t total = 0; bool stalled = false, ever_seen = false; };
    [[nodiscard]] std::vector<StreamState> stream_states() const
    {
        std::vector<StreamState> v;
        for (const auto& [k, s] : streams_)
            v.push_back({k, s.mon.total(), s.mon.stalled(), s.mon.ever_seen()});
        return v;
    }
    // Read-only graph-fed inputs for the same display. odom: dedup'd producer samples (stalled when none for
    // stall_imu_ms); pose: distinct robot-in-room poses read; polygon: vertex count (0 = none yet).
    struct GraphState { std::uint64_t odom_total = 0, pose_total = 0; bool odom_seen = false, odom_stalled = false, pose_seen = false; std::size_t polygon_vertices = 0; };
    [[nodiscard]] GraphState graph_state() const
    {
        GraphState g;
        g.odom_total = odom_total_; g.pose_total = pose_total_;
        g.odom_seen = odom_total_ > 0;
        g.odom_stalled = g.odom_seen and std::chrono::steady_clock::now() - odom_last_ > std::chrono::milliseconds(cfg_.stall_imu_ms);
        g.pose_seen = pose_.has_value();
        g.polygon_vertices = polygon_.vertices.size();
        return g;
    }
    [[nodiscard]] bool stalled(const std::string& key) const { const auto it = streams_.find(key); return it != streams_.end() and it->second.mon.stalled(); }

private:
    void ensure_subscribers();
    void drain_lidar(const std::string& sensor, rc::media::LidarSubscriber& sub, StreamMonitor& mon);
    void drain_image(const std::string& sensor, const std::string& stream, rc::media::MediaSubscriber& sub,
                     StreamMonitor& mon, RateDecimator& dec);
    void drain_image360(rc::media::Image360Subscriber& sub, StreamMonitor& mon, RateDecimator& dec);
    void drain_imu();
    void poll_graph();
    void monitor_and_log();
    template <class T> void push(std::deque<T>& q, T&& v, std::size_t cap);

    std::shared_ptr<DSR::DSRGraph>        G_;
    InputsConfig                          cfg_;
    std::unique_ptr<DSR::InnerEigenAPI>   inner_;     // owned by the main thread (CLAUDE.md ownership rule)
    std::unique_ptr<DSR::RT_API>          rt_;

    std::unique_ptr<rc::media::LidarSubscriber>   helios_, bpearl_;
    std::unique_ptr<rc::media::MediaSubscriber>   zed_rgb_, zed_depth_;
    std::unique_ptr<rc::media::Image360Subscriber> ricoh_;
    std::unique_ptr<rc::media::ImuSubscriber>     imu_sub_;

    struct Stream { StreamMonitor mon; RateDecimator dec; };
    std::map<std::string, Stream>         streams_;   // keyed "helios","bpearl","zed/rgb","zed/depth","ricoh","imu"

    std::deque<LidarIn>    lidar_q_;
    std::deque<ImageIn>    image_q_;
    std::deque<ImuIn>      imu_q_;
    std::deque<OdometryIn> odom_q_;
    std::optional<PoseIn>  pose_;
    // Robot node + room cached by poll_graph() for the per-sweep reads in drain_lidar() (one tick behind).
    std::string            robot_name_;
    std::uint64_t          robot_id_ = 0;
    std::optional<std::uint64_t> room_id_;
    [[nodiscard]] std::optional<Eigen::Affine3f> robot_from_sensor(const std::string& sensor, std::int64_t stamp_ms) const;
    // Pinhole intrinsics of a camera node, read through CameraAPI on first use and cached (main thread only).
    [[nodiscard]] PinholeIntrinsics intrinsics_of(const std::string& sensor);
    std::map<std::string, PinholeIntrinsics> intrinsics_;
    [[nodiscard]] std::optional<Eigen::Affine3d> room_T_robot_at(std::int64_t stamp_ms, double* walk_ms) const;
    PolygonIn              polygon_;
    std::int64_t           last_odom_ts_ = 0;
    std::uint64_t          odom_total_ = 0, pose_total_ = 0;   // display counters (read-only accessor graph_state())
    std::chrono::steady_clock::time_point odom_last_{};
    std::chrono::steady_clock::time_point last_poly_{};
    std::chrono::steady_clock::time_point last_discovery_{}, last_log_{};
    std::map<std::string, bool> announced_absent_;
};

}  // namespace sc
