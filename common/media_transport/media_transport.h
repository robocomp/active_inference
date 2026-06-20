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
#include <string>

#include "generated/idl/image_framePubSubTypes.hpp"

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

// One writer per topic. Not thread-safe: call from a single producer thread.
class MediaPublisher
{
public:
    MediaPublisher() = default;
    ~MediaPublisher();
    MediaPublisher(const MediaPublisher&) = delete;
    MediaPublisher& operator=(const MediaPublisher&) = delete;

    bool init(const PublisherConfig& cfg);
    void close();

    // Zero-copy path: borrow a loaned sample living in the SHM segment, fill it
    // in place (set fields + memcpy pixels into frame->data()), then publish().
    // Returns nullptr if no loan is available (pool exhausted) -> caller may
    // skip the frame or fall back to publish_copy().
    ImageFrame* loan();
    bool        publish(ImageFrame* loaned_sample);
    void        discard(ImageFrame* loaned_sample);

    // Fallback copy path (serializes/copies). Useful when the source buffer is
    // already owned and zero-copy loans are unavailable.
    bool publish_copy(const ImageFrame& frame);

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

}  // namespace rc::media
