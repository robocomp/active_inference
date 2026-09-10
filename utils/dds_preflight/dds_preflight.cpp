// dds_preflight — verify that FastDDS SHARED-MEMORY discovery actually works on a domain,
// before a fleet is started on it.
//
// Why this exists (2026-07-21): an unclean shutdown leaves orphaned /dev/shm/fastdds_port*
// segments behind. New participants attach to those dead ring buffers and discovery messages go
// nowhere — SILENTLY, with no error on either side. The whole zero-copy media plane then reads
// 0.0 Hz while everything carried over Ice/TCP keeps working, which looks exactly like a code
// regression and costs hours to diagnose. This turns that silent failure into a loud one.
//
// The check is a self-contained loopback: two SEPARATE participants in this one process (so it
// exercises real inter-participant discovery, not an in-process shortcut), SHM-only transport
// exactly as the media plane configures it, a writer on one and a reader on the other over a
// scratch topic. If they match, SHM discovery is healthy.
//
// Uses the project's smallest generated type (ImuFrame, a handful of scalars) on a scratch topic,
// so the tool stays tiny and needs no type of its own.
//
//   exit 0 = SHM discovery healthy
//   exit 1 = NOT healthy (almost always stale segments: stop everything, rm -f /dev/shm/fastdds_*)
//   exit 2 = could not run the check at all (participant/entity creation failed)
//
// Usage: dds_preflight [domain=7] [timeout_ms=3000]

#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/publisher/DataWriter.hpp>
#include <fastdds/dds/publisher/Publisher.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/SampleInfo.hpp>
#include <fastdds/dds/subscriber/Subscriber.hpp>
#include <fastdds/dds/topic/Topic.hpp>
#include <fastdds/dds/topic/TypeSupport.hpp>
#include <fastdds/rtps/transport/shared_mem/SharedMemTransportDescriptor.hpp>

#include "imu_framePubSubTypes.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>

namespace efd = eprosima::fastdds::dds;
using rc::media::ImuFrame;
using rc::media::ImuFramePubSubType;

namespace
{

efd::DomainParticipant *make_shm_participant(int domain)
{
    // Mirror rc::media::make_participant(): drop the builtin transports and keep ONLY shared
    // memory, so this test fails in exactly the same conditions the media plane fails in.
    efd::DomainParticipantQos pqos = efd::PARTICIPANT_QOS_DEFAULT;
    pqos.transport().use_builtin_transports = false;
    pqos.transport().user_transports.push_back(
        std::make_shared<eprosima::fastdds::rtps::SharedMemTransportDescriptor>());
    return efd::DomainParticipantFactory::get_instance()->create_participant(domain, pqos);
}

}   // namespace

int main(int argc, char **argv)
{
    const int domain     = (argc > 1) ? std::atoi(argv[1]) : 7;
    const int timeout_ms = (argc > 2) ? std::atoi(argv[2]) : 3000;

    // HARD watchdog. A corrupted/orphaned segment does not always fail cleanly: FastDDS can block
    // INSIDE create_participant, i.e. before any timeout of ours is reachable (verified 07-21 by
    // fault injection — the tool hung instead of failing). A preflight that can hang would hang the
    // launcher it gates, which is worse than the problem it detects. So bound the whole run here and
    // _Exit(1) — a hung SHM stack is exactly the "not healthy" verdict we want to report anyway.
    std::thread([timeout_ms]()
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(2 * timeout_ms + 2000));
        std::printf("dds_preflight: FAILED — the SHM stack HUNG (blocked inside FastDDS, most likely a\n");
        std::printf("dds_preflight: corrupt or orphaned segment).\n");
        std::printf("dds_preflight: stop everything on this domain, then: rm -f /dev/shm/fastdds_*\n");
        std::fflush(stdout);
        std::_Exit(1);   // not exit(): static destructors would join the very stack that is stuck
    }).detach();

    auto *factory = efd::DomainParticipantFactory::get_instance();

    // TWO participants: discovery between them is the thing under test.
    auto *p_pub = make_shm_participant(domain);
    auto *p_sub = make_shm_participant(domain);
    if (p_pub == nullptr or p_sub == nullptr)
    {
        std::printf("dds_preflight: FATAL could not create SHM participants on domain %d\n", domain);
        return 2;
    }

    efd::TypeSupport type(new ImuFramePubSubType());
    type.register_type(p_pub);
    type.register_type(p_sub);

    const std::string topic_name = "rc/_preflight/ping";
    auto *t_pub = p_pub->create_topic(topic_name, type.get_type_name(), efd::TOPIC_QOS_DEFAULT);
    auto *t_sub = p_sub->create_topic(topic_name, type.get_type_name(), efd::TOPIC_QOS_DEFAULT);
    if (t_pub == nullptr or t_sub == nullptr)
    {
        std::printf("dds_preflight: FATAL could not create topics on domain %d\n", domain);
        return 2;
    }

    auto *pub = p_pub->create_publisher(efd::PUBLISHER_QOS_DEFAULT);
    auto *sub = p_sub->create_subscriber(efd::SUBSCRIBER_QOS_DEFAULT);
    if (pub == nullptr or sub == nullptr)
    {
        std::printf("dds_preflight: FATAL could not create pub/sub on domain %d\n", domain);
        return 2;
    }

    // RELIABLE/VOLATILE/KEEP_LAST — same shape the media plane uses, so a QoS-level
    // incompatibility would also be caught here rather than in production.
    efd::DataWriterQos wqos = efd::DATAWRITER_QOS_DEFAULT;
    wqos.reliability().kind = efd::RELIABLE_RELIABILITY_QOS;
    wqos.durability().kind  = efd::VOLATILE_DURABILITY_QOS;
    wqos.history().kind     = efd::KEEP_LAST_HISTORY_QOS;
    wqos.history().depth    = 1;

    efd::DataReaderQos rqos = efd::DATAREADER_QOS_DEFAULT;
    rqos.reliability().kind = efd::RELIABLE_RELIABILITY_QOS;
    rqos.durability().kind  = efd::VOLATILE_DURABILITY_QOS;
    rqos.history().kind     = efd::KEEP_LAST_HISTORY_QOS;
    rqos.history().depth    = 1;

    auto *writer = pub->create_datawriter(t_pub, wqos);
    auto *reader = sub->create_datareader(t_sub, rqos);
    if (writer == nullptr or reader == nullptr)
    {
        std::printf("dds_preflight: FATAL could not create writer/reader on domain %d\n", domain);
        return 2;
    }

    // Poll for the match. Discovery is what stale segments break, so matching (not data delivery)
    // is the signal; we still send one sample afterwards to prove the path end to end.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    bool matched = false;
    while (std::chrono::steady_clock::now() < deadline)
    {
        efd::SubscriptionMatchedStatus st{};
        if (reader->get_subscription_matched_status(st) == eprosima::fastdds::dds::RETCODE_OK
            and st.current_count > 0)
        {
            matched = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }

    int rc = 1;
    if (not matched)
    {
        std::printf("dds_preflight: FAILED — no SHM discovery on domain %d within %d ms\n",
                    domain, timeout_ms);
        std::printf("dds_preflight: almost certainly orphaned segments from an unclean shutdown.\n");
        std::printf("dds_preflight: stop everything on this domain, then: rm -f /dev/shm/fastdds_*\n");
    }
    else
    {
        // Prove a sample actually crosses, not just that the endpoints matched.
        ImuFrame ping;
        ping.frame_id(1);
        writer->write(&ping);

        bool got = false;
        const auto data_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (not got and std::chrono::steady_clock::now() < data_deadline)
        {
            ImuFrame        in;
            efd::SampleInfo info;
            if (reader->take_next_sample(&in, &info) == eprosima::fastdds::dds::RETCODE_OK and info.valid_data)
                got = true;
            else
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        if (got)
        {
            std::printf("dds_preflight: OK — SHM discovery and delivery healthy on domain %d\n", domain);
            rc = 0;
        }
        else
        {
            std::printf("dds_preflight: FAILED — endpoints matched on domain %d but no sample "
                        "was delivered over shared memory\n", domain);
            std::printf("dds_preflight: stop everything on this domain, then: rm -f /dev/shm/fastdds_*\n");
        }
    }

    factory->delete_participant(p_pub);
    factory->delete_participant(p_sub);
    return rc;
}
