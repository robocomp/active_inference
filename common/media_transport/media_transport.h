// RoboComp active_inference - media transport plane (zero-copy DDS)
//
// MediaPublisher / MediaSubscriber wrap a dedicated FastDDS topic carrying raw
// image frames OUT of the DSR/CRDT graph. QoS is BEST_EFFORT + KEEP_LAST +
// VOLATILE + shared-memory data-sharing, and the bounded plain ImageFrame type
// (is_plain()==true) allows true zero-copy via writer loans and reader loans.
//
// On a single board the only per-frame cost is one memcpy into a preallocated
// SHM slot; readers map the same segment and read in place. This keeps 30 fps
// RGBD off the graph entirely, so liveness/heartbeat can never be starved.

#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <print>
#include <string>
#include <vector>

#include "generated/idl/image_framePubSubTypes.hpp"
#include "generated/idl/image360_framePubSubTypes.hpp"
#include "generated/idl/lidar_framePubSubTypes.hpp"
#include "generated/idl/imu_framePubSubTypes.hpp"

namespace eprosima::fastdds::dds {
class DomainParticipant;
class Publisher;
class Subscriber;
class Topic;
class DataWriter;
class DataReader;
}  // namespace eprosima::fastdds::dds

namespace rc::media
{

struct PublisherConfig
{
    std::uint32_t domain_id = 0;
    std::string   topic_name;           // e.g. "rc/zed/rgb"
    int           history_depth = 4;    // KEEP_LAST depth
    int           max_instances = 4;    // distinct stream_id keys
    bool          shared_memory_only = true;  // SHM transport for same-board
    // Zero-copy SHM data-sharing. Default OFF: data-sharing maps a fixed SHM segment shared with
    // peers, and FastDDS reconfigures it on participant discovery. In CORTEX, concept agents join/
    // leave constantly, so a media CONSUMER holding a mapped frame across a discovery event reads a
    // remapped/freed segment -> heap corruption (random SEGVs in cv::Mat/onnx/paint). The SHM
    // TRANSPORT (shared_memory_only) still gives same-board speed; only the zero-copy loan layer is
    // disabled, costing one ~2.7MB memcpy/frame. Opt in only for a static, non-churning topology.
    bool          data_sharing = false;
};

struct SubscriberConfig
{
    std::uint32_t domain_id = 0;
    std::string   topic_name;
    int           history_depth = 4;
    int           max_instances = 4;
    bool          shared_memory_only = true;
    bool          data_sharing = false;  // see PublisherConfig::data_sharing — OFF = churn-safe
};

// ---------------------------------------------------------------------------
// Self-describing media plane (graph-based discovery)
// ---------------------------------------------------------------------------
// A publisher advertises HOW to attach to its media plane by writing a JSON
// MediaDescriptor into a DSR node attribute (the descriptor lives in the graph;
// the heavy frames stay out-of-band on the zero-copy DDS plane). Any agent reads
// that string and builds a matching Subscriber/Publisher config with no hardcoded
// topic/domain. Only the *configurable* QoS is carried — reliability/durability/
// history-kind are fixed in this lib, so both ends agree by construction.

// Default DSR attribute name that carries the descriptor JSON string.
inline constexpr const char* MEDIA_DESCRIPTOR_ATTR = "media_descriptor";

// Type-compatibility tags for the zero-copy bounded-plain frame types. Zero-copy
// maps the SHM segment byte-for-byte, so a subscriber MUST refuse a stream whose
// tag differs. BUMP THE RELEVANT TAG whenever the matching .idl changes.
inline constexpr const char* IMAGE_FRAME_TYPE_TAG    = "ImageFrame.v2";
inline constexpr const char* IMAGE360_FRAME_TYPE_TAG = "Image360Frame.v1";
inline constexpr const char* LIDAR_FRAME_TYPE_TAG    = "LidarFrame.v1";
inline constexpr const char* IMU_FRAME_TYPE_TAG      = "ImuFrame.v1";

// ---------------------------------------------------------------------------
// Sensor physics, carried alongside the transport descriptor
// ---------------------------------------------------------------------------
// WHY HERE. A driver is the only party that knows its own geometry and noise;
// every consumer today re-asserts them as its own config constants (room_concept
// has PreintOdomSigma*, residual_concept infers beam geometry). Two sources of
// truth for one physical fact, and nothing makes them disagree loudly. This
// rides the descriptor because that channel already exists end to end --
// MediaPlaneDDS::getMediaDescriptor() returns a string, robot_concept relays it
// VERBATIM onto the sensor node, and it re-relays only when the string changes.
// So no .idsl edit, no Ice stub regeneration, no attribute churn.
//
// ★ ABSENT MEANS UNKNOWN, AND THAT IS A REAL ANSWER. Every field is optional. A
// producer that has not measured its noise must leave the field out rather than
// advertise a plausible default: a nominal constant published as fact is exactly
// what left this channel inert from 2026-08-25 to 08-26. A consumer that finds a
// field absent keeps its own constant.
//
// ★ QUADRATURE, NEVER REPLACEMENT. What a consumer already holds is lumped MODEL
// error (slip, timing, unmodelled dynamics); this is SENSOR noise. They ADD.
// Replacing was tried 2026-08-23 and made the heading prior 135x too confident.
//
// ★ EXTRINSICS ARE NOT HERE ON PURPOSE. Where a sensor sits is the RT tree, which
// is authoritative, timestamped and interpolatable. `frame` only NAMES the frame
// these intrinsics are expressed in; it never restates the pose.
//
// ★ ONE NOISE IDIOM FLEET-WIDE: sigma(x) = hypot(floor, k * x). It is what
// p3bot-bridge already publishes for the wheels, and it degrades correctly -- at
// x = 0 a channel reports its floor, not zero, and not a flat density that would
// inject motion into a parked robot.
struct SensorModel
{
    // ── Provenance. Without it a number is an assertion, not a measurement. ──
    int          version  = 0;      // 0 = no model advertised
    std::string  source;            // "measured"|"calibrated"|"datasheet"|"nominal"|"simulated"
    std::string  ref;               // config file, calibration id, datasheet part
    std::int64_t stamp_ms = 0;      // when this model was produced/calibrated
    std::string  frame;             // DSR frame the intrinsics are expressed in

    // ── Intrinsic geometry (LiDAR). Extrinsics live in the RT tree. ──
    std::optional<int>   rings;
    std::vector<float>   ring_elev_deg;      // per-ring elevation; ';'-separated on the wire
    std::optional<float> azimuth_step_deg;
    std::optional<float> range_min_m, range_max_m;
    std::optional<float> fov_start_deg, fov_end_deg;
    std::optional<float> rate_hz;

    // ── Noise: range channel. sigma_r(r) = hypot(floor, k_rel*r), grown by incidence. ──
    std::optional<float> range_sigma_floor_m;
    std::optional<float> range_k_rel;        // per metre of range
    std::optional<float> range_k_incidence;  // per unit tan(incidence), optional

    // ── Noise: IMU. ──
    std::optional<float> gyro_sigma;         // rad/sqrt(s)
    std::optional<float> gyro_bias;          // rad/s
    std::optional<float> gyro_bias_rw;       // rad/s/sqrt(s), bias random walk
    std::optional<float> acc_sigma;          // m/s^2/sqrt(Hz)

    // ── Noise: wheels. Present for completeness; on P3Bot the wheel model is
    // already recoverable from the per-sample velCov that arrives at 50 Hz
    // (regressing var on speed^2 recovers k_slip to within 4-10%), so a wheel
    // producer need not fill these in. ──
    std::optional<float> v_floor, k_slip_v;  // m/sqrt(s), per (m/s)
    std::optional<float> w_floor, k_slip_w;  // rad/sqrt(s), per (rad/s)

    [[nodiscard]] bool empty() const noexcept { return version == 0; }
};

struct MediaDescriptor
{
    int           version = 1;
    std::uint32_t domain_id = 0;
    std::string   type_name = "ImageFrame";
    std::string   type_tag  = IMAGE_FRAME_TYPE_TAG;  // schema guard for zero-copy
    int           history_depth = 8;
    bool          shared_memory_only = true;
    bool          data_sharing = false;
    bool          ready = false;                      // lifecycle: false until the stream is live
    std::map<std::string, std::string> streams;       // stream key ("rgb","depth","lidar","imu") -> topic name
    // Per-stream IDL type name ("ImageFrame","LidarFrame","ImuFrame"). Lets a
    // consumer pick the right reader when a plane carries mixed payloads. Absent
    // entries default to `type_name` (back-compat: an all-image plane needs none).
    std::map<std::string, std::string> stream_types;

    // Sensor physics for the node this descriptor is advertised on. Default-empty
    // (version 0) emits NOTHING, so an existing producer's descriptor string stays
    // byte-identical -- which matters, because robot_concept re-relays only when
    // that string changes.
    SensorModel model;

    // Flat JSON ({"key": value}, streams emitted as "<key>_topic"). No external deps.
    [[nodiscard]] std::string to_json() const;
    [[nodiscard]] static std::optional<MediaDescriptor> from_json(const std::string& s);

    // Build a config for one advertised stream key. nullopt if the key is absent.
    [[nodiscard]] std::optional<SubscriberConfig> subscriber_config(const std::string& stream_key) const;
    [[nodiscard]] std::optional<PublisherConfig>  publisher_config(const std::string& stream_key) const;
};

// Split a ';'-separated list of floats, locale-independently (std::from_chars).
// Exposed because a PRODUCER has to build ring_elev_deg from its own config, and the
// obvious std::stof/strtof there would silently truncate at the '.' under es_ES.
[[nodiscard]] std::vector<float> parse_float_list(const std::string& s);

// Discovery convenience: read the descriptor JSON from a DSR node attribute and
// parse it. Templated on the graph type so this header stays DSR-free (the
// standalone media_bench never instantiates it, so it links no DSR). Graph must
// expose get_node(name)->optional<Node> and Node::attrs(); the attribute value is
// read as a string via Attribute::str().
template <class Graph>
[[nodiscard]] std::optional<MediaDescriptor>
descriptor_from_graph(Graph& graph, const std::string& node_name,
                      const char* attr_name = MEDIA_DESCRIPTOR_ATTR)
{
    auto node = graph.get_node(node_name);
    if (not node.has_value())
        return std::nullopt;
    const auto& attrs = node->attrs();
    auto it = attrs.find(attr_name);
    if (it == attrs.end())
        return std::nullopt;
    try { return MediaDescriptor::from_json(it->second.str()); }
    catch (...) { return std::nullopt; }   // attribute present but not a string / malformed
}

namespace detail
{
// Type-agnostic writer plumbing shared by every typed publisher below. All the
// participant/QoS/loan logic lives here once; the typed wrappers only add the
// FrameKind selector (which PubSubType to register) and the loan/publish casts.
// Not thread-safe: drive from a single producer thread.
enum class FrameKind { Image, Image360, Lidar, Imu };

class WriterCore
{
public:
    WriterCore() = default;
    ~WriterCore();
    WriterCore(const WriterCore&) = delete;
    WriterCore& operator=(const WriterCore&) = delete;

    bool  init(const PublisherConfig& cfg, FrameKind kind);
    void  close();
    void* loan_raw();
    bool  publish_raw(void* loaned_sample);
    void  discard_raw(void* loaned_sample);
    bool  publish_copy_raw(const void* sample);

    [[nodiscard]] bool data_sharing_active() const { return data_sharing_active_; }

private:
    eprosima::fastdds::dds::DomainParticipant* participant_ = nullptr;
    eprosima::fastdds::dds::Publisher*         publisher_   = nullptr;
    eprosima::fastdds::dds::Topic*             topic_       = nullptr;
    eprosima::fastdds::dds::DataWriter*        writer_      = nullptr;
    std::uint32_t participant_domain_id_ = 0;
    bool participant_shm_only_ = true;
    bool has_participant_ = false;
    bool data_sharing_active_ = false;
};
}  // namespace detail

// One writer per topic, one class per payload type. The fill pattern is the same
// for all: loan() a SHM-resident sample, set its fields + memcpy the payload into
// the inline array, then publish(). loan() returns nullptr if the pool is
// exhausted -> the caller skips the frame. Not thread-safe per instance.
class MediaPublisher
{
public:
    MediaPublisher() = default;
    bool init(const PublisherConfig& cfg) { return core_.init(cfg, detail::FrameKind::Image); }
    void close() { core_.close(); }
    ImageFrame* loan() { return static_cast<ImageFrame*>(core_.loan_raw()); }
    bool        publish(ImageFrame* s) { return core_.publish_raw(s); }
    void        discard(ImageFrame* s) { core_.discard_raw(s); }
    // Fallback copy path (serializes/copies) when the source buffer is owned.
    bool        publish_copy(const ImageFrame& frame) { return core_.publish_copy_raw(&frame); }
    [[nodiscard]] bool data_sharing_active() const { return core_.data_sharing_active(); }
private:
    detail::WriterCore core_;
};

// Publisher for the wide 360 panorama (Image360Frame, ~5.5 MB inline buffer).
// Separate from MediaPublisher so the ZED ImageFrame buffer stays small.
class Image360Publisher
{
public:
    Image360Publisher() = default;
    bool init(const PublisherConfig& cfg) { return core_.init(cfg, detail::FrameKind::Image360); }
    void close() { core_.close(); }
    Image360Frame* loan() { return static_cast<Image360Frame*>(core_.loan_raw()); }
    bool           publish(Image360Frame* s) { return core_.publish_raw(s); }
    void           discard(Image360Frame* s) { core_.discard_raw(s); }
    bool           publish_copy(const Image360Frame& frame) { return core_.publish_copy_raw(&frame); }
    [[nodiscard]] bool data_sharing_active() const { return core_.data_sharing_active(); }
private:
    detail::WriterCore core_;
};

class LidarPublisher
{
public:
    LidarPublisher() = default;
    bool init(const PublisherConfig& cfg) { return core_.init(cfg, detail::FrameKind::Lidar); }
    void close() { core_.close(); }
    LidarFrame* loan() { return static_cast<LidarFrame*>(core_.loan_raw()); }
    bool        publish(LidarFrame* s) { return core_.publish_raw(s); }
    void        discard(LidarFrame* s) { core_.discard_raw(s); }
    [[nodiscard]] bool data_sharing_active() const { return core_.data_sharing_active(); }
private:
    detail::WriterCore core_;
};

class ImuPublisher
{
public:
    ImuPublisher() = default;
    bool init(const PublisherConfig& cfg) { return core_.init(cfg, detail::FrameKind::Imu); }
    void close() { core_.close(); }
    ImuFrame* loan() { return static_cast<ImuFrame*>(core_.loan_raw()); }
    bool      publish(ImuFrame* s) { return core_.publish_raw(s); }
    void      discard(ImuFrame* s) { core_.discard_raw(s); }
    [[nodiscard]] bool data_sharing_active() const { return core_.data_sharing_active(); }
private:
    detail::WriterCore core_;
};

// One reader per topic. The callback receives a const ImageFrame& that is valid
// ONLY for the duration of the call (it is a loaned view into the SHM segment).
class MediaSubscriber
{
public:
    using FrameCallback = std::function<void(const ImageFrame& frame, std::int64_t recv_ns)>;

    MediaSubscriber() = default;
    ~MediaSubscriber();
    MediaSubscriber(const MediaSubscriber&) = delete;
    MediaSubscriber& operator=(const MediaSubscriber&) = delete;

    bool init(const SubscriberConfig& cfg);
    void close();

    // Drain all currently-available samples, invoking cb for each (zero-copy).
    // Returns the number of frames delivered. Non-blocking.
    int poll(const FrameCallback& cb);

    // Block until at least one frame is available or the timeout elapses, then
    // drain. Returns the number of frames delivered.
    int wait_and_poll(const FrameCallback& cb, int timeout_ms);

    [[nodiscard]] bool data_sharing_active() const { return data_sharing_active_; }

private:
    eprosima::fastdds::dds::DomainParticipant* participant_ = nullptr;
    eprosima::fastdds::dds::Subscriber*        subscriber_  = nullptr;
    eprosima::fastdds::dds::Topic*             topic_       = nullptr;
    eprosima::fastdds::dds::DataReader*        reader_      = nullptr;
    std::uint32_t participant_domain_id_ = 0;
    bool participant_shm_only_ = true;
    bool has_participant_ = false;
    bool data_sharing_active_ = false;
};

// One reader per 360 topic. The callback receives a const Image360Frame& valid
// only for the duration of the call (a loaned view into the SHM segment). Mirror
// of MediaSubscriber for the Image360Frame type.
class Image360Subscriber
{
public:
    using FrameCallback = std::function<void(const Image360Frame& frame, std::int64_t recv_ns)>;

    Image360Subscriber() = default;
    ~Image360Subscriber();
    Image360Subscriber(const Image360Subscriber&) = delete;
    Image360Subscriber& operator=(const Image360Subscriber&) = delete;

    bool init(const SubscriberConfig& cfg);
    void close();

    int poll(const FrameCallback& cb);                          // non-blocking drain
    int wait_and_poll(const FrameCallback& cb, int timeout_ms); // block-then-drain

    [[nodiscard]] bool data_sharing_active() const { return data_sharing_active_; }

private:
    eprosima::fastdds::dds::DomainParticipant* participant_ = nullptr;
    eprosima::fastdds::dds::Subscriber*        subscriber_  = nullptr;
    eprosima::fastdds::dds::Topic*             topic_       = nullptr;
    eprosima::fastdds::dds::DataReader*        reader_      = nullptr;
    std::uint32_t participant_domain_id_ = 0;
    bool participant_shm_only_ = true;
    bool has_participant_ = false;
    bool data_sharing_active_ = false;
};

// One reader per LiDAR topic. The callback receives a const LidarFrame& valid only
// for the duration of the call (a loaned view into the SHM segment). Mirror of
// MediaSubscriber for the LidarFrame type — kept a separate concrete class so the
// ImageFrame MediaSubscriber that other agents forward-declare stays unchanged.
class LidarSubscriber
{
public:
    using FrameCallback = std::function<void(const LidarFrame& frame, std::int64_t recv_ns)>;

    LidarSubscriber() = default;
    ~LidarSubscriber();
    LidarSubscriber(const LidarSubscriber&) = delete;
    LidarSubscriber& operator=(const LidarSubscriber&) = delete;

    bool init(const SubscriberConfig& cfg);
    void close();

    int poll(const FrameCallback& cb);                          // non-blocking drain
    int wait_and_poll(const FrameCallback& cb, int timeout_ms); // block-then-drain

    [[nodiscard]] bool data_sharing_active() const { return data_sharing_active_; }

    // How many publishers this reader is currently matched with. A successful init() only means the
    // reader was CREATED — it says nothing about whether the producer was ever discovered, so a
    // wedged SHM discovery port reads exactly like a healthy-but-idle sensor. This is the number
    // that tells the two apart, so a consumer can say "no publisher matched" instead of going quiet.
    [[nodiscard]] int matched_writers() const;

private:
    eprosima::fastdds::dds::DomainParticipant* participant_ = nullptr;
    eprosima::fastdds::dds::Subscriber*        subscriber_  = nullptr;
    eprosima::fastdds::dds::Topic*             topic_       = nullptr;
    eprosima::fastdds::dds::DataReader*        reader_      = nullptr;
    std::uint32_t participant_domain_id_ = 0;
    bool participant_shm_only_ = true;
    bool has_participant_ = false;
    bool data_sharing_active_ = false;
};

// One reader per IMU topic. The callback receives a const ImuFrame& valid only for the
// duration of the call. Mirror of LidarSubscriber for the small ImuFrame type (acc/gyro/
// mag/rpy scalars) so agents can consume the IMU media stream the same uniform way.
class ImuSubscriber
{
public:
    using FrameCallback = std::function<void(const ImuFrame& frame, std::int64_t recv_ns)>;

    ImuSubscriber() = default;
    ~ImuSubscriber();
    ImuSubscriber(const ImuSubscriber&) = delete;
    ImuSubscriber& operator=(const ImuSubscriber&) = delete;

    bool init(const SubscriberConfig& cfg);
    void close();

    int poll(const FrameCallback& cb);                          // non-blocking drain
    int wait_and_poll(const FrameCallback& cb, int timeout_ms); // block-then-drain

    [[nodiscard]] bool data_sharing_active() const { return data_sharing_active_; }

private:
    eprosima::fastdds::dds::DomainParticipant* participant_ = nullptr;
    eprosima::fastdds::dds::Subscriber*        subscriber_  = nullptr;
    eprosima::fastdds::dds::Topic*             topic_       = nullptr;
    eprosima::fastdds::dds::DataReader*        reader_      = nullptr;
    std::uint32_t participant_domain_id_ = 0;
    bool participant_shm_only_ = true;
    bool has_participant_ = false;
    bool data_sharing_active_ = false;
};

// ── Descriptor-driven subscriber factories ───────────────────────────────────
// THE single, shared way every agent brings up a media-plane consumer, so the
// initialization code is identical everywhere (easier maintenance). Each factory:
//   1. reads the producer's descriptor on `node_name` (verifies the sensor node +
//      stream exist — returns nullptr if not, so callers can fall back / retry),
//   2. takes the DDS domain + topic STRAIGHT from that descriptor (never config —
//      always the producer's dedicated media domain),
//   3. creates the typed subscriber and init()s it,
//   4. logs a uniform one-line result.
// Call from the MAIN thread once the graph is loaded (never from a free-running
// worker thread during the DSR join). Returns the ready subscriber, or nullptr.
template <class Graph>
[[nodiscard]] std::unique_ptr<LidarSubscriber>
make_lidar_subscriber_from_graph(Graph& graph, const std::string& node_name,
                                 const std::string& stream_key = "lidar")
{
    // Two DIFFERENT failures, and only one of them is news.
    //   node ABSENT  → this robot does not carry that sensor (p3bot.json has helios but no bpearl,
    //                  shadow.json has both). A permanent, expected state that no amount of waiting
    //                  changes, so return nullptr SILENTLY and let the caller — which knows whether
    //                  it needed that plane — say so once if it cares. Callers retry on a loop, so
    //                  speaking here means speaking for ever: robot_concept's monitor thread paces on
    //                  helios's 100 ms wait_and_poll and so re-asked for bpearl ~10×/s, swamping the
    //                  log on P3Bot.
    //   node PRESENT but no descriptor → the producer has not advertised yet. Transient and worth
    //                  reporting, so this one still speaks.
    if (not graph.get_node(node_name).has_value())
        return nullptr;
    auto desc = descriptor_from_graph(graph, node_name);
    auto cfg  = desc.has_value() ? desc->subscriber_config(stream_key) : std::nullopt;
    if (not cfg.has_value())
    {
        std::print(stderr, "[media] no '{}' stream descriptor on node '{}' — LiDAR subscriber not created\n",
                   stream_key, node_name);
        return nullptr;
    }
    auto sub = std::make_unique<LidarSubscriber>();
    if (not sub->init(*cfg))
    {
        std::print(stderr, "[media] LiDAR subscriber init FAILED (node '{}' topic '{}')\n",
                   node_name, cfg->topic_name);
        return nullptr;
    }
    std::print("[media] LiDAR subscriber up from '{}' descriptor: domain={} topic='{}'\n",
               node_name, cfg->domain_id, cfg->topic_name);
    return sub;
}

template <class Graph>
[[nodiscard]] std::unique_ptr<MediaSubscriber>
make_image_subscriber_from_graph(Graph& graph, const std::string& node_name,
                                 const std::string& stream_key)
{
    auto desc = descriptor_from_graph(graph, node_name);
    auto cfg  = desc.has_value() ? desc->subscriber_config(stream_key) : std::nullopt;
    if (not cfg.has_value())
    {
        std::print(stderr, "[media] no '{}' stream descriptor on node '{}' — image subscriber not created\n",
                   stream_key, node_name);
        return nullptr;
    }
    auto sub = std::make_unique<MediaSubscriber>();
    if (not sub->init(*cfg))
    {
        std::print(stderr, "[media] image subscriber init FAILED (node '{}' topic '{}')\n",
                   node_name, cfg->topic_name);
        return nullptr;
    }
    std::print("[media] image subscriber up from '{}' descriptor ('{}'): domain={} topic='{}'\n",
               node_name, stream_key, cfg->domain_id, cfg->topic_name);
    return sub;
}

template <class Graph>
[[nodiscard]] std::unique_ptr<Image360Subscriber>
make_image360_subscriber_from_graph(Graph& graph, const std::string& node_name,
                                    const std::string& stream_key)
{
    auto desc = descriptor_from_graph(graph, node_name);
    auto cfg  = desc.has_value() ? desc->subscriber_config(stream_key) : std::nullopt;
    if (not cfg.has_value())
    {
        std::print(stderr, "[media] no '{}' stream descriptor on node '{}' — image360 subscriber not created\n",
                   stream_key, node_name);
        return nullptr;
    }
    auto sub = std::make_unique<Image360Subscriber>();
    if (not sub->init(*cfg))
    {
        std::print(stderr, "[media] image360 subscriber init FAILED (node '{}' topic '{}')\n",
                   node_name, cfg->topic_name);
        return nullptr;
    }
    std::print("[media] image360 subscriber up from '{}' descriptor ('{}'): domain={} topic='{}'\n",
               node_name, stream_key, cfg->domain_id, cfg->topic_name);
    return sub;
}

template <class Graph>
[[nodiscard]] std::unique_ptr<ImuSubscriber>
make_imu_subscriber_from_graph(Graph& graph, const std::string& node_name,
                               const std::string& stream_key = "imu")
{
    auto desc = descriptor_from_graph(graph, node_name);
    auto cfg  = desc.has_value() ? desc->subscriber_config(stream_key) : std::nullopt;
    if (not cfg.has_value())
    {
        std::print(stderr, "[media] no '{}' stream descriptor on node '{}' — IMU subscriber not created\n",
                   stream_key, node_name);
        return nullptr;
    }
    auto sub = std::make_unique<ImuSubscriber>();
    if (not sub->init(*cfg))
    {
        std::print(stderr, "[media] IMU subscriber init FAILED (node '{}' topic '{}')\n",
                   node_name, cfg->topic_name);
        return nullptr;
    }
    std::print("[media] IMU subscriber up from '{}' descriptor ('{}'): domain={} topic='{}'\n",
               node_name, stream_key, cfg->domain_id, cfg->topic_name);
    return sub;
}

}  // namespace rc::media
