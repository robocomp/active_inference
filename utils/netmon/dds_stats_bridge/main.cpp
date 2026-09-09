// dds_stats_bridge: subscribes to Fast DDS's built-in PUBLICATION_THROUGHPUT_TOPIC and
// SUBSCRIPTION_THROUGHPUT_TOPIC statistics topics and dumps {topic_name: bytes_per_second}
// to a JSON file, so a stdlib-only monitor (no DDS bindings) can display real DDS
// publish/receive bandwidth even when the actual data plane uses SharedMemoryOnly transport
// (no bytes on the wire). This is how netmon gets consumer-side (subscriber) throughput
// without any code in the consumer itself -- Fast DDS's own statistics module already
// tracks it per DataReader, same mechanism as the writer side, just a second topic + a
// second discovery callback (on_data_reader_discovery) to resolve reader GUID -> topic
// name. It's a coarser signal than the app-level StreamStats in media_transport.h (bytes/s
// only, no frame_id-gap "drops" or capture-to-receive latency), but "is this consumer
// actually receiving bytes" needs zero changes to the consumer's source.
//
// Requires the monitored process(es) to be started with
//   FASTDDS_STATISTICS=_fastdds_statistics_publication_throughput;_fastdds_statistics_subscription_throughput
// so they actually publish these statistics topics (set once, before their
// DomainParticipant is constructed).
//
// The monitored components here use SharedMemoryOnly transport, which also carries
// discovery (SPDP/EDP) traffic instead of UDP multicast -- this participant must use
// the same SHM-only builtin transport preset or it will never see them.
//
// Samples are drained by polling the reader once per second from the main loop rather
// than via DataReaderListener::on_data_available: in this Fast DDS build the listener
// callback never fires for this reader even though matched samples do land in its
// history (confirmed via reader->get_unread_count()), so polling is the reliable path.

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/domain/DomainParticipantListener.hpp>
#include <fastdds/dds/domain/qos/DomainParticipantQos.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/Subscriber.hpp>
#include <fastdds/dds/subscriber/SampleInfo.hpp>
#include <fastdds/dds/subscriber/qos/DataReaderQos.hpp>
#include <fastdds/dds/subscriber/qos/SubscriberQos.hpp>
#include <fastdds/dds/topic/Topic.hpp>
#include <fastdds/dds/topic/TypeSupport.hpp>
#include <fastdds/rtps/attributes/BuiltinTransports.hpp>
#include <fastdds/rtps/common/Guid.hpp>

#include <fastdds/statistics/topic_names.hpp>

#include "types.hpp"
#include "typesPubSubTypes.hpp"

using namespace eprosima::fastdds::dds;
using eprosima::fastdds::rtps::GUID_t;
using eprosima::fastdds::statistics::EntityData;
using eprosima::fastdds::statistics::EntityDataPubSubType;

namespace {

std::atomic<bool> g_running{true};

void on_signal(int)
{
    g_running = false;
}

std::string guid_to_key(
        const GUID_t& guid)
{
    std::ostringstream os;
    os << guid;
    return os.str();
}

// Shared between the discovery listener (guid -> real topic name) and the drain
// loop (guid -> latest bytes/s sample), and used to render the output JSON.
class SharedState
{
public:
    mutable std::mutex mtx;
    std::map<std::string, std::string> guid_to_topic;
    std::map<std::string, float> guid_to_bps;

    void write_json(
            const std::string& path) const
    {
        std::map<std::string, float> by_topic;
        {
            std::lock_guard<std::mutex> lock(mtx);
            for (const auto& kv : guid_to_bps)
            {
                auto it = guid_to_topic.find(kv.first);
                if (it != guid_to_topic.end())
                {
                    by_topic[it->second] = kv.second;
                }
            }
        }

        std::ostringstream os;
        os << "{";
        bool first = true;
        for (const auto& kv : by_topic)
        {
            if (!first)
            {
                os << ",";
            }
            first = false;
            os << "\"";
            for (char c : kv.first)
            {
                if (c == '"' || c == '\\')
                {
                    os << '\\';
                }
                os << c;
            }
            os << "\":" << kv.second;
        }
        os << "}";

        std::string tmp = path + ".tmp";
        {
            std::ofstream f(tmp, std::ios::trunc);
            f << os.str();
        }
        std::rename(tmp.c_str(), path.c_str());
    }
};

// Tracks real (non-statistics) DataWriters discovered in the domain, so a GUID
// carried inside an EntityData throughput sample can be mapped back to its topic.
class DiscoveryListener : public DomainParticipantListener
{
public:
    explicit DiscoveryListener(
            SharedState& state)
        : state_(state)
    {
    }

    void on_data_writer_discovery(
            DomainParticipant*,
            eprosima::fastdds::rtps::WriterDiscoveryStatus reason,
            const PublicationBuiltinTopicData& info,
            bool& should_be_ignored) override
    {
        should_be_ignored = false;
        std::string key = guid_to_key(info.guid);
        std::lock_guard<std::mutex> lock(state_.mtx);
        if (reason == eprosima::fastdds::rtps::WriterDiscoveryStatus::REMOVED_WRITER)
        {
            state_.guid_to_topic.erase(key);
        }
        else
        {
            state_.guid_to_topic[key] = info.topic_name.to_string();
        }
    }

    // Mirror of on_data_writer_discovery for the reader side, so a GUID carried inside a
    // SUBSCRIPTION_THROUGHPUT_TOPIC sample can be mapped back to its topic the same way.
    // Writer and reader GUIDs never collide (the entity id half differs), so both sides
    // share guid_to_topic safely.
    void on_data_reader_discovery(
            DomainParticipant*,
            eprosima::fastdds::rtps::ReaderDiscoveryStatus reason,
            const SubscriptionBuiltinTopicData& info,
            bool& should_be_ignored) override
    {
        should_be_ignored = false;
        std::string key = guid_to_key(info.guid);
        std::lock_guard<std::mutex> lock(state_.mtx);
        if (reason == eprosima::fastdds::rtps::ReaderDiscoveryStatus::REMOVED_READER)
        {
            state_.guid_to_topic.erase(key);
        }
        else
        {
            state_.guid_to_topic[key] = info.topic_name.to_string();
        }
    }

private:
    SharedState& state_;
};

// Drains whatever throughput samples are sitting in the reader's history.
void drain_throughput_samples(
        DataReader* reader,
        SharedState& state)
{
    EntityData sample;
    SampleInfo info;
    while (RETCODE_OK == reader->take_next_sample(&sample, &info))
    {
        if (!info.valid_data)
        {
            continue;
        }
        GUID_t guid;
        std::memcpy(&guid.guidPrefix, sample.guid().guidPrefix().value().data(),
                sizeof(guid.guidPrefix.value));
        std::memcpy(&guid.entityId, sample.guid().entityId().value().data(),
                sizeof(guid.entityId.value));
        std::lock_guard<std::mutex> lock(state.mtx);
        state.guid_to_bps[guid_to_key(guid)] = sample.data();
    }
}

} // namespace

int main(
        int argc,
        char** argv)
{
    uint32_t domain_id = 7;
    std::string out_path = "/tmp/robocomp_netmon/dds_stats.json";

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--domain" && i + 1 < argc)
        {
            domain_id = static_cast<uint32_t>(std::atoi(argv[++i]));
        }
        else if (arg == "--out" && i + 1 < argc)
        {
            out_path = argv[++i];
        }
        else
        {
            std::cerr << "usage: " << argv[0] << " [--domain N] [--out PATH]\n";
            return 1;
        }
    }

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    SharedState state;
    DiscoveryListener discovery_listener(state);

    DomainParticipantQos pqos = PARTICIPANT_QOS_DEFAULT;
    pqos.name("dds_stats_bridge");
    pqos.wire_protocol().participant_id = 99;   // avoid colliding with sensor drivers' auto-assigned ids
    pqos.setup_transports(eprosima::fastdds::rtps::BuiltinTransports::SHM);

    DomainParticipant* participant = DomainParticipantFactory::get_instance()->create_participant(
            domain_id, pqos, &discovery_listener, StatusMask::all());
    if (nullptr == participant)
    {
        std::cerr << "Failed to create DomainParticipant on domain " << domain_id << "\n";
        return 1;
    }

    TypeSupport type(new EntityDataPubSubType());
    type.register_type(participant);

    Subscriber* subscriber = participant->create_subscriber(SUBSCRIBER_QOS_DEFAULT, nullptr);
    DataReaderQos rqos = DATAREADER_QOS_DEFAULT;

    // Publication and subscription throughput share the exact same EntityData sample shape
    // (Fast DDS handles both through the same code path internally) -- one topic+reader
    // pair each, both draining into the same guid_to_bps/guid_to_topic maps.
    Topic* pub_topic = participant->create_topic(
            eprosima::fastdds::statistics::PUBLICATION_THROUGHPUT_TOPIC, type.get_type_name(), TOPIC_QOS_DEFAULT);
    Topic* sub_topic = participant->create_topic(
            eprosima::fastdds::statistics::SUBSCRIPTION_THROUGHPUT_TOPIC, type.get_type_name(), TOPIC_QOS_DEFAULT);
    DataReader* pub_reader = (nullptr != pub_topic)
            ? subscriber->create_datareader(pub_topic, rqos, nullptr, StatusMask::none()) : nullptr;
    DataReader* sub_reader = (nullptr != sub_topic)
            ? subscriber->create_datareader(sub_topic, rqos, nullptr, StatusMask::none()) : nullptr;
    if (nullptr == pub_reader && nullptr == sub_reader)
    {
        std::cerr << "Failed to create any statistics DataReader\n";
        return 1;
    }

    {
        auto pos = out_path.find_last_of('/');
        if (pos != std::string::npos)
        {
            std::string cmd = "mkdir -p '" + out_path.substr(0, pos) + "'";
            if (system(cmd.c_str()) != 0)
            {
                std::cerr << "warning: could not ensure output dir exists\n";
            }
        }
    }

    std::cerr << "dds_stats_bridge: listening on domain " << domain_id
              << ", writing " << out_path << "\n";

    while (g_running)
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (nullptr != pub_reader)
        {
            drain_throughput_samples(pub_reader, state);
        }
        if (nullptr != sub_reader)
        {
            drain_throughput_samples(sub_reader, state);
        }
        state.write_json(out_path);
    }

    std::cerr << "dds_stats_bridge: shutting down\n";
    if (nullptr != pub_reader)
    {
        subscriber->delete_datareader(pub_reader);
    }
    if (nullptr != sub_reader)
    {
        subscriber->delete_datareader(sub_reader);
    }
    participant->delete_subscriber(subscriber);
    if (nullptr != pub_topic)
    {
        participant->delete_topic(pub_topic);
    }
    if (nullptr != sub_topic)
    {
        participant->delete_topic(sub_topic);
    }
    DomainParticipantFactory::get_instance()->delete_participant(participant);
    return 0;
}
