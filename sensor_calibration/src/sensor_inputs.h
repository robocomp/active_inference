/*
 * sensor_inputs — the INPUT LAYER of sensor_calibration.
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

    // points_sensor mapped through body_from_sensor; empty when the mount is unresolved.
    [[nodiscard]] std::vector<Eigen::Vector3f> points_body() const;
};

struct ImageIn
{
    std::string                sensor;              // "zed" | "ricoh"
    std::string                stream;              // "rgb" | "depth" | "rgb360"
    std::uint64_t              frame_id = 0;
    std::int64_t               stamp_ms = 0;
    std::uint32_t              width = 0, height = 0, step = 0, format = 0;   // format: producer's FORMAT_* code
    std::vector<std::uint8_t>  data;                // OWNED deep copy of the valid bytes
    std::optional<Eigen::Affine3f> body_from_sensor;
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

struct PoseIn
{
    std::uint64_t from_id = 0, to_id = 0;              // the RT edge as stored: robot -> room
    Eigen::Affine3d from_T_to = Eigen::Affine3d::Identity();   // rt_translation + Rx*Ry*Rz(rt_rotation_euler_xyz)
    std::array<float, 36> cov{};                       // rt_covariance, 6x6 row-major [x y z rx ry rz]; yaw = [35]
    bool          has_cov = false;
    std::int64_t  read_stamp_ms = 0;                   // wall clock of the read (the edge carries no capture stamp)

    [[nodiscard]] Eigen::Affine3d to_T_from() const { return from_T_to.inverse(); }   // room<-robot pose
};

struct PolygonIn
{
    std::uint64_t room_id = 0;
    std::vector<Eigen::Vector2f> vertices;             // room frame
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

    // Hand-over queues (bounded, drop-oldest). The consumer takes and owns what it takes.
    [[nodiscard]] std::vector<LidarIn>    take_lidar();
    [[nodiscard]] std::vector<ImageIn>    take_images();
    [[nodiscard]] std::vector<ImuIn>      take_imu();
    [[nodiscard]] std::vector<OdometryIn> take_odometry();

    // Latest graph reads (nullopt until first resolvable).
    [[nodiscard]] const std::optional<PoseIn>&    pose() const { return pose_; }
    [[nodiscard]] const PolygonIn&                polygon() const { return polygon_; }

    // body<-sensor from the graph now, at a real stamp (ts!=0 bypasses the ts==0 cache). nullopt if unresolved.
    [[nodiscard]] std::optional<Eigen::Affine3f> body_from_sensor(const std::string& sensor, std::int64_t stamp_ms) const;

    // True while every stream that has ever delivered is currently delivering (no stalled stream).
    [[nodiscard]] bool any_stalled() const;
    // Per-stream rate over the window since the last call (restarts the window), plus graph-read state.
    [[nodiscard]] std::string status_line();
    // True when this stream ("helios","zed/rgb","imu",...) has delivered before and has now stopped.
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
    PolygonIn              polygon_;
    std::int64_t           last_odom_ts_ = 0;
    std::chrono::steady_clock::time_point last_poly_{};
    std::chrono::steady_clock::time_point last_discovery_{}, last_log_{};
    std::map<std::string, bool> announced_absent_;
};

}  // namespace sc
