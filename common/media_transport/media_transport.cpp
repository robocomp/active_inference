// RoboComp active_inference - media transport plane (zero-copy DDS) impl.

#include "media_transport.h"

#include <cctype>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <string_view>
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
FASTDDS_SEQUENCE(Image360FrameSeq, rc::media::Image360Frame);
FASTDDS_SEQUENCE(LidarFrameSeq, rc::media::LidarFrame);

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

namespace
{
efd::TypeSupport make_type_support(detail::FrameKind kind)
{
    switch (kind)
    {
        case detail::FrameKind::Image360: return efd::TypeSupport(new Image360FramePubSubType());
        case detail::FrameKind::Lidar: return efd::TypeSupport(new LidarFramePubSubType());
        case detail::FrameKind::Imu:   return efd::TypeSupport(new ImuFramePubSubType());
        case detail::FrameKind::Image:
        default:                       return efd::TypeSupport(new ImageFramePubSubType());
    }
}
}  // namespace

bool detail::WriterCore::init(const PublisherConfig& cfg, detail::FrameKind kind)
{
    close();

    participant_ = acquire_participant(cfg.domain_id, cfg.shared_memory_only);
    if (!participant_)
        return false;
    participant_domain_id_ = cfg.domain_id;
    participant_shm_only_ = cfg.shared_memory_only;
    has_participant_ = true;

    efd::TypeSupport type = make_type_support(kind);
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
    // Zero-copy data-sharing only when explicitly opted in AND not force-disabled by env.
    // Default (cfg.data_sharing=false) is OFF — churn-safe; see PublisherConfig::data_sharing.
    if (cfg.data_sharing and not std::getenv("MEDIA_NO_DATASHARING"))
        wqos.data_sharing().automatic();  // zero-copy on same board for plain types
    else
        wqos.data_sharing().off();

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

void* detail::WriterCore::loan_raw()
{
    if (!writer_)
        return nullptr;
    void* sample = nullptr;
    if (writer_->loan_sample(sample) != efd::RETCODE_OK)
        return nullptr;
    return sample;
}

bool detail::WriterCore::publish_raw(void* loaned_sample)
{
    if (!writer_ || !loaned_sample)
        return false;
    if (writer_->write(loaned_sample) == efd::RETCODE_OK)
        return true;

    // Keep the loan pool healthy on write failures.
    writer_->discard_loan(loaned_sample);
    return false;
}

void detail::WriterCore::discard_raw(void* loaned_sample)
{
    if (!writer_ || !loaned_sample)
        return;
    writer_->discard_loan(loaned_sample);
}

bool detail::WriterCore::publish_copy_raw(const void* sample)
{
    if (!writer_ || !sample)
        return false;
    return writer_->write(const_cast<void*>(sample)) == efd::RETCODE_OK;
}

void detail::WriterCore::close()
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

detail::WriterCore::~WriterCore()
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
    // Zero-copy data-sharing only when explicitly opted in AND not force-disabled by env.
    // Default (cfg.data_sharing=false) is OFF — churn-safe; see SubscriberConfig::data_sharing.
    if (cfg.data_sharing and not std::getenv("MEDIA_NO_DATASHARING"))
        rqos.data_sharing().automatic();
    else
        rqos.data_sharing().off();

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

// ──────────────────────────── Image360Subscriber ────────────────────────────
// Mirror of MediaSubscriber for the Image360Frame type (wide panorama). Shares
// the participant pool + QoS conventions; only the registered type differs.

bool Image360Subscriber::init(const SubscriberConfig& cfg)
{
    close();

    participant_ = acquire_participant(cfg.domain_id, cfg.shared_memory_only);
    if (!participant_)
        return false;
    participant_domain_id_ = cfg.domain_id;
    participant_shm_only_ = cfg.shared_memory_only;
    has_participant_ = true;

    efd::TypeSupport type(new Image360FramePubSubType());
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
    if (cfg.data_sharing and not std::getenv("MEDIA_NO_DATASHARING"))
        rqos.data_sharing().automatic();
    else
        rqos.data_sharing().off();

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

int Image360Subscriber::poll(const FrameCallback& cb)
{
    if (!reader_)
        return 0;

    int delivered = 0;
    Image360FrameSeq data;
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

int Image360Subscriber::wait_and_poll(const FrameCallback& cb, int timeout_ms)
{
    if (!reader_)
        return 0;

    const efd::Duration_t timeout(timeout_ms / 1000,
                                  static_cast<std::uint32_t>((timeout_ms % 1000) * 1000000));
    reader_->wait_for_unread_message(timeout);
    return poll(cb);
}

Image360Subscriber::~Image360Subscriber()
{
    close();
}

void Image360Subscriber::close()
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

// ───────────────────────────── LidarSubscriber ─────────────────────────────
// Mirror of MediaSubscriber for the LidarFrame type. Shares the participant pool
// + QoS conventions; only the registered type and the take() sequence differ.

bool LidarSubscriber::init(const SubscriberConfig& cfg)
{
    close();

    participant_ = acquire_participant(cfg.domain_id, cfg.shared_memory_only);
    if (!participant_)
        return false;
    participant_domain_id_ = cfg.domain_id;
    participant_shm_only_ = cfg.shared_memory_only;
    has_participant_ = true;

    efd::TypeSupport type(new LidarFramePubSubType());
    type.register_type(participant_);

    subscriber_ = participant_->create_subscriber(efd::SUBSCRIBER_QOS_DEFAULT);
    if (!subscriber_) { close(); return false; }

    topic_ = participant_->create_topic(cfg.topic_name, type.get_type_name(), efd::TOPIC_QOS_DEFAULT);
    if (!topic_) { close(); return false; }

    efd::DataReaderQos rqos = efd::DATAREADER_QOS_DEFAULT;
    rqos.reliability().kind = std::getenv("MEDIA_BEST_EFFORT")
                                  ? efd::BEST_EFFORT_RELIABILITY_QOS
                                  : efd::RELIABLE_RELIABILITY_QOS;
    rqos.durability().kind  = efd::VOLATILE_DURABILITY_QOS;
    rqos.history().kind     = efd::KEEP_LAST_HISTORY_QOS;
    rqos.history().depth    = cfg.history_depth;
    rqos.resource_limits().max_instances            = cfg.max_instances;
    rqos.resource_limits().max_samples_per_instance = cfg.history_depth;
    rqos.resource_limits().max_samples = cfg.max_instances * cfg.history_depth + 2;
    rqos.endpoint().history_memory_policy = eprosima::fastdds::rtps::PREALLOCATED_MEMORY_MODE;
    if (cfg.data_sharing and not std::getenv("MEDIA_NO_DATASHARING"))
        rqos.data_sharing().automatic();
    else
        rqos.data_sharing().off();

    reader_ = subscriber_->create_datareader(topic_, rqos);
    if (!reader_) { close(); return false; }

    data_sharing_active_ = reader_->get_qos().data_sharing().kind() != efd::OFF;
    return true;
}

int LidarSubscriber::poll(const FrameCallback& cb)
{
    if (!reader_)
        return 0;

    int delivered = 0;
    LidarFrameSeq data;
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

int LidarSubscriber::wait_and_poll(const FrameCallback& cb, int timeout_ms)
{
    if (!reader_)
        return 0;
    const efd::Duration_t timeout(timeout_ms / 1000,
                                  static_cast<std::uint32_t>((timeout_ms % 1000) * 1000000));
    reader_->wait_for_unread_message(timeout);
    return poll(cb);
}

LidarSubscriber::~LidarSubscriber()
{
    close();
}

void LidarSubscriber::close()
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

// ---------------------------------------------------------------------------
// MediaDescriptor: flat-JSON (de)serialization + config builders
// ---------------------------------------------------------------------------
namespace
{
// Minimal scanner for a FLAT JSON object: {"k": "str", "k2": 3, "k3": true}.
// No nesting/arrays (the descriptor schema is deliberately flat), so this stays
// small and robust. Every value is returned as its raw string token; typed
// fields are converted by the caller.
std::map<std::string, std::string> parse_flat_json(const std::string& s)
{
    std::map<std::string, std::string> out;
    const std::size_t n = s.size();
    std::size_t i = 0;
    auto skip_ws = [&] { while (i < n and std::isspace(static_cast<unsigned char>(s[i]))) ++i; };
    auto parse_string = [&](std::string& dst) -> bool
    {
        if (i >= n or s[i] != '"') return false;
        ++i;
        std::string r;
        while (i < n and s[i] != '"')
        {
            if (s[i] == '\\' and i + 1 < n)
            {
                ++i;
                const char c = s[i];
                r += (c == 'n') ? '\n' : (c == 't') ? '\t' : c;   // covers \" \\ too
            }
            else
                r += s[i];
            ++i;
        }
        if (i >= n) return false;
        ++i;                                                       // closing quote
        dst = std::move(r);
        return true;
    };
    skip_ws();
    if (i < n and s[i] == '{') ++i;
    while (true)
    {
        skip_ws();
        if (i >= n or s[i] == '}') break;
        std::string key;
        if (not parse_string(key)) break;
        skip_ws();
        if (i < n and s[i] == ':') ++i; else break;
        skip_ws();
        std::string val;
        if (i < n and s[i] == '"')
        {
            if (not parse_string(val)) break;
        }
        else                                                       // number / bool / null token
        {
            const std::size_t start = i;
            while (i < n and s[i] != ',' and s[i] != '}') ++i;
            val = s.substr(start, i - start);
            while (not val.empty() and std::isspace(static_cast<unsigned char>(val.back()))) val.pop_back();
        }
        out.emplace(std::move(key), std::move(val));
        skip_ws();
        if (i < n and s[i] == ',') { ++i; continue; }
    }
    return out;
}

bool as_bool(const std::string& v) { return v == "true" or v == "1"; }
int  as_int (const std::string& v, int def) { try { return std::stoi(v); } catch (...) { return def; } }
} // namespace

std::string MediaDescriptor::to_json() const
{
    auto esc = [](const std::string& v)
    {
        std::string r;
        for (char c : v) { if (c == '"' or c == '\\') r += '\\'; r += c; }
        return r;
    };
    std::string s = "{";
    s += "\"version\":" + std::to_string(version);
    s += ",\"domain_id\":" + std::to_string(domain_id);
    s += ",\"type_name\":\"" + esc(type_name) + "\"";
    s += ",\"type_tag\":\"" + esc(type_tag) + "\"";
    s += ",\"history_depth\":" + std::to_string(history_depth);
    s += ",\"shared_memory_only\":" + std::string(shared_memory_only ? "true" : "false");
    s += ",\"data_sharing\":" + std::string(data_sharing ? "true" : "false");
    s += ",\"ready\":" + std::string(ready ? "true" : "false");
    for (const auto& [key, topic] : streams)
        s += ",\"" + esc(key) + "_topic\":\"" + esc(topic) + "\"";
    for (const auto& [key, tname] : stream_types)
        s += ",\"" + esc(key) + "_type\":\"" + esc(tname) + "\"";
    s += "}";
    return s;
}

std::optional<MediaDescriptor> MediaDescriptor::from_json(const std::string& s)
{
    const auto kv = parse_flat_json(s);
    if (kv.empty()) return std::nullopt;
    MediaDescriptor d;
    if (auto it = kv.find("version");            it != kv.end()) d.version            = as_int(it->second, d.version);
    if (auto it = kv.find("domain_id");          it != kv.end()) d.domain_id          = static_cast<std::uint32_t>(as_int(it->second, 0));
    if (auto it = kv.find("type_name");          it != kv.end()) d.type_name          = it->second;
    if (auto it = kv.find("type_tag");           it != kv.end()) d.type_tag           = it->second;
    if (auto it = kv.find("history_depth");      it != kv.end()) d.history_depth      = as_int(it->second, d.history_depth);
    if (auto it = kv.find("shared_memory_only"); it != kv.end()) d.shared_memory_only = as_bool(it->second);
    if (auto it = kv.find("data_sharing");       it != kv.end()) d.data_sharing       = as_bool(it->second);
    if (auto it = kv.find("ready");              it != kv.end()) d.ready              = as_bool(it->second);
    // Any "<name>_topic" key is a stream advertisement; "<name>_type" its IDL type.
    auto ends_with = [](const std::string& k, std::string_view sfx)
    { return k.size() > sfx.size() and k.compare(k.size() - sfx.size(), sfx.size(), sfx) == 0; };
    constexpr std::string_view topic_sfx = "_topic";
    constexpr std::string_view type_sfx  = "_type";
    for (const auto& [k, v] : kv)
    {
        if (ends_with(k, topic_sfx))
            d.streams.emplace(k.substr(0, k.size() - topic_sfx.size()), v);
        else if (ends_with(k, type_sfx))
            d.stream_types.emplace(k.substr(0, k.size() - type_sfx.size()), v);
    }
    return d;
}

std::optional<SubscriberConfig> MediaDescriptor::subscriber_config(const std::string& stream_key) const
{
    auto it = streams.find(stream_key);
    if (it == streams.end()) return std::nullopt;
    SubscriberConfig cfg;
    cfg.domain_id          = domain_id;
    cfg.topic_name         = it->second;
    cfg.history_depth      = history_depth;
    cfg.shared_memory_only = shared_memory_only;
    cfg.data_sharing       = data_sharing;
    return cfg;
}

std::optional<PublisherConfig> MediaDescriptor::publisher_config(const std::string& stream_key) const
{
    auto it = streams.find(stream_key);
    if (it == streams.end()) return std::nullopt;
    PublisherConfig cfg;
    cfg.domain_id          = domain_id;
    cfg.topic_name         = it->second;
    cfg.history_depth      = history_depth;
    cfg.shared_memory_only = shared_memory_only;
    cfg.data_sharing       = data_sharing;
    return cfg;
}

}  // namespace rc::media
