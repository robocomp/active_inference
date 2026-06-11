// RoboComp active_inference - media transport plane (zero-copy DDS) impl.

#include "media_transport.h"

#include <chrono>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/domain/qos/DomainParticipantQos.hpp>
#include <fastdds/dds/publisher/DataWriter.hpp>
#include <fastdds/dds/publisher/Publisher.hpp>
#include <fastdds/dds/publisher/qos/DataWriterQos.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/SampleInfo.hpp>
#include <fastdds/dds/subscriber/Subscriber.hpp>
#include <fastdds/dds/subscriber/qos/DataReaderQos.hpp>
#include <fastdds/dds/topic/Topic.hpp>
#include <fastdds/dds/topic/TypeSupport.hpp>
#include <fastdds/dds/core/LoanableSequence.hpp>
#include <fastdds/dds/core/Time_t.hpp>
#include <fastdds/rtps/transport/shared_mem/SharedMemTransportDescriptor.hpp>

namespace efd = eprosima::fastdds::dds;

FASTDDS_SEQUENCE(ImageFrameSeq, rc::media::ImageFrame);

namespace
{
std::int64_t now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

struct ParticipantKey
{
    std::uint32_t domain_id;
    bool shm_only;

    bool operator==(const ParticipantKey& other) const
    {
        return domain_id == other.domain_id && shm_only == other.shm_only;
    }
};

struct ParticipantKeyHash
{
    std::size_t operator()(const ParticipantKey& key) const
    {
        return (static_cast<std::size_t>(key.domain_id) << 1U) ^ static_cast<std::size_t>(key.shm_only);
    }
};

struct ParticipantEntry
{
    efd::DomainParticipant* participant = nullptr;
    std::size_t refs = 0;
};

std::mutex participant_pool_mtx;
std::unordered_map<ParticipantKey, ParticipantEntry, ParticipantKeyHash> participant_pool;

bool effective_shm_only(bool requested)
{
    if (std::getenv("MEDIA_NO_SHM_ONLY"))
        return false;
    return requested;
}

efd::DomainParticipant* make_participant(const ParticipantKey& key)
{
    efd::DomainParticipantQos pqos = efd::PARTICIPANT_QOS_DEFAULT;
    if (key.shm_only)
    {
        // Same-board: drop UDP, keep only shared memory for RTPS control traffic.
        // Payload zero-copy is handled independently by data-sharing.
        pqos.transport().use_builtin_transports = false;
        auto shm = std::make_shared<eprosima::fastdds::rtps::SharedMemTransportDescriptor>();
        pqos.transport().user_transports.push_back(shm);
    }
    return efd::DomainParticipantFactory::get_instance()->create_participant(
        static_cast<int>(key.domain_id), pqos);
}

efd::DomainParticipant* acquire_participant(std::uint32_t domain_id, bool requested_shm_only)
{
    const ParticipantKey key{domain_id, effective_shm_only(requested_shm_only)};
    std::lock_guard<std::mutex> lock(participant_pool_mtx);
    auto it = participant_pool.find(key);
    if (it != participant_pool.end())
    {
        ++it->second.refs;
        return it->second.participant;
    }

    if (efd::DomainParticipant* participant = make_participant(key); participant != nullptr)
    {
        participant_pool.emplace(key, ParticipantEntry{participant, 1});
        return participant;
    }

    return nullptr;
}

void release_participant(std::uint32_t domain_id, bool requested_shm_only, efd::DomainParticipant* participant)
{
    if (participant == nullptr)
        return;

    const ParticipantKey key{domain_id, effective_shm_only(requested_shm_only)};
    std::lock_guard<std::mutex> lock(participant_pool_mtx);
    auto it = participant_pool.find(key);
    if (it == participant_pool.end() || it->second.participant != participant)
        return;

    if (it->second.refs > 1)
    {
        --it->second.refs;
        return;
    }

    efd::DomainParticipantFactory::get_instance()->delete_participant(it->second.participant);
    participant_pool.erase(it);
}
}  // namespace

namespace rc::media
{

// ───────────────────────────── Publisher ─────────────────────────────

bool MediaPublisher::init(const PublisherConfig& cfg)
{
    close();

    participant_ = acquire_participant(cfg.domain_id, cfg.shared_memory_only);
    if (!participant_)
        return false;
    participant_domain_id_ = cfg.domain_id;
    participant_shm_only_ = cfg.shared_memory_only;
    has_participant_ = true;

    efd::TypeSupport type(new ImageFramePubSubType());
    type.register_type(participant_);

    publisher_ = participant_->create_publisher(efd::PUBLISHER_QOS_DEFAULT);
    if (!publisher_)
    {
        close();
        return false;
    }

    topic_ = participant_->create_topic(cfg.topic_name, type.get_type_name(),
                                        efd::TOPIC_QOS_DEFAULT);
    if (!topic_)
    {
        close();
        return false;
    }

    efd::DataWriterQos wqos = efd::DATAWRITER_QOS_DEFAULT;
    // RELIABLE is required for lossless 30fps: with BEST_EFFORT the writer never
    // waits and the reader overwrites ~2/3 of the large (2.76MB) frames before
    // take(). RELIABLE's NACK-repair lets the reader catch every frame.
    wqos.reliability().kind = std::getenv("MEDIA_BEST_EFFORT")
                                  ? efd::BEST_EFFORT_RELIABILITY_QOS
                                  : efd::RELIABLE_RELIABILITY_QOS;
    // Bound the worst-case producer stall: KEEP_LAST means write() only blocks
    // when history is full of unacked samples; cap that wait well under one
    // frame period so a slow/dead consumer can never throttle the producer.
    wqos.reliability().max_blocking_time =
        efd::Duration_t(0, 10 * 1000 * 1000);  // 10 ms
    wqos.durability().kind  = efd::VOLATILE_DURABILITY_QOS;
    wqos.history().kind     = efd::KEEP_LAST_HISTORY_QOS;
    wqos.history().depth    = cfg.history_depth;
    // Bounded, preallocated pool so loan_sample() can hand out SHM slots.
    wqos.resource_limits().max_instances           = cfg.max_instances;
    wqos.resource_limits().max_samples_per_instance = cfg.history_depth;
    wqos.resource_limits().max_samples =
        cfg.max_instances * cfg.history_depth + 2;
    wqos.resource_limits().extra_samples = 2;  // headroom for in-flight loans
    wqos.endpoint().history_memory_policy =
        eprosima::fastdds::rtps::PREALLOCATED_MEMORY_MODE;
    if (std::getenv("MEDIA_NO_DATASHARING"))
        wqos.data_sharing().off();
    else
        wqos.data_sharing().automatic();  // zero-copy on same board for plain types

    writer_ = publisher_->create_datawriter(topic_, wqos);
    if (!writer_)
    {
        close();
        return false;
    }

    data_sharing_active_ =
        writer_->get_qos().data_sharing().kind() != efd::OFF;
    return true;
}

ImageFrame* MediaPublisher::loan()
{
    if (!writer_)
        return nullptr;
    void* sample = nullptr;
    if (writer_->loan_sample(sample) != efd::RETCODE_OK)
        return nullptr;
    return static_cast<ImageFrame*>(sample);
}

bool MediaPublisher::publish(ImageFrame* loaned_sample)
{
    if (!writer_ || !loaned_sample)
        return false;
    if (writer_->write(loaned_sample) == efd::RETCODE_OK)
        return true;

    // Keep the loan pool healthy on write failures.
    void* sample = loaned_sample;
    writer_->discard_loan(sample);
    return false;
}

void MediaPublisher::discard(ImageFrame* loaned_sample)
{
    if (!writer_ || !loaned_sample)
        return;
    void* sample = loaned_sample;
    writer_->discard_loan(sample);
}

bool MediaPublisher::publish_copy(const ImageFrame& frame)
{
    if (!writer_)
        return false;
    return writer_->write(const_cast<ImageFrame*>(&frame)) == efd::RETCODE_OK;
}

void MediaPublisher::close()
{
    if (publisher_ != nullptr && writer_ != nullptr)
        publisher_->delete_datawriter(writer_);
    writer_ = nullptr;

    if (participant_ != nullptr && topic_ != nullptr)
        participant_->delete_topic(topic_);
    topic_ = nullptr;

    if (participant_ != nullptr && publisher_ != nullptr)
        participant_->delete_publisher(publisher_);
    publisher_ = nullptr;

    if (has_participant_)
    {
        release_participant(participant_domain_id_, participant_shm_only_, participant_);
        participant_ = nullptr;
        has_participant_ = false;
    }

    data_sharing_active_ = false;
}

MediaPublisher::~MediaPublisher()
{
    close();
}

// ───────────────────────────── Subscriber ─────────────────────────────

bool MediaSubscriber::init(const SubscriberConfig& cfg)
{
    close();

    participant_ = acquire_participant(cfg.domain_id, cfg.shared_memory_only);
    if (!participant_)
        return false;
    participant_domain_id_ = cfg.domain_id;
    participant_shm_only_ = cfg.shared_memory_only;
    has_participant_ = true;

    efd::TypeSupport type(new ImageFramePubSubType());
    type.register_type(participant_);

    subscriber_ = participant_->create_subscriber(efd::SUBSCRIBER_QOS_DEFAULT);
    if (!subscriber_)
    {
        close();
        return false;
    }

    topic_ = participant_->create_topic(cfg.topic_name, type.get_type_name(),
                                        efd::TOPIC_QOS_DEFAULT);
    if (!topic_)
    {
        close();
        return false;
    }

    efd::DataReaderQos rqos = efd::DATAREADER_QOS_DEFAULT;
    rqos.reliability().kind = std::getenv("MEDIA_BEST_EFFORT")
                                  ? efd::BEST_EFFORT_RELIABILITY_QOS
                                  : efd::RELIABLE_RELIABILITY_QOS;
    rqos.durability().kind  = efd::VOLATILE_DURABILITY_QOS;
    rqos.history().kind     = efd::KEEP_LAST_HISTORY_QOS;
    rqos.history().depth    = cfg.history_depth;
    rqos.resource_limits().max_instances            = cfg.max_instances;
    rqos.resource_limits().max_samples_per_instance = cfg.history_depth;
    rqos.resource_limits().max_samples =
        cfg.max_instances * cfg.history_depth + 2;
    rqos.endpoint().history_memory_policy =
        eprosima::fastdds::rtps::PREALLOCATED_MEMORY_MODE;
    if (std::getenv("MEDIA_NO_DATASHARING"))
        rqos.data_sharing().off();
    else
        rqos.data_sharing().automatic();

    reader_ = subscriber_->create_datareader(topic_, rqos);
    if (!reader_)
    {
        close();
        return false;
    }

    data_sharing_active_ =
        reader_->get_qos().data_sharing().kind() != efd::OFF;
    return true;
}

int MediaSubscriber::poll(const FrameCallback& cb)
{
    if (!reader_)
        return 0;

    int delivered = 0;
    ImageFrameSeq data;
    efd::SampleInfoSeq infos;
    while (reader_->take(data, infos) == efd::RETCODE_OK)
    {
        const std::int64_t recv = now_ns();
        for (efd::LoanableCollection::size_type i = 0; i < infos.length(); ++i)
        {
            if (infos[i].valid_data)
            {
                cb(data[i], recv);
                ++delivered;
            }
        }
        reader_->return_loan(data, infos);
    }
    return delivered;
}

int MediaSubscriber::wait_and_poll(const FrameCallback& cb, int timeout_ms)
{
    if (!reader_)
        return 0;

    const efd::Duration_t timeout(timeout_ms / 1000,
                                  static_cast<std::uint32_t>((timeout_ms % 1000) * 1000000));
    reader_->wait_for_unread_message(timeout);
    return poll(cb);
}

MediaSubscriber::~MediaSubscriber()
{
    close();
}

void MediaSubscriber::close()
{
    if (subscriber_ != nullptr && reader_ != nullptr)
        subscriber_->delete_datareader(reader_);
    reader_ = nullptr;

    if (participant_ != nullptr && topic_ != nullptr)
        participant_->delete_topic(topic_);
    topic_ = nullptr;

    if (participant_ != nullptr && subscriber_ != nullptr)
        participant_->delete_subscriber(subscriber_);
    subscriber_ = nullptr;

    if (has_participant_)
    {
        release_participant(participant_domain_id_, participant_shm_only_, participant_);
        participant_ = nullptr;
        has_participant_ = false;
    }

    data_sharing_active_ = false;
}

}  // namespace rc::media
