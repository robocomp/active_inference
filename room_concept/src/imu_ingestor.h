/*
 *  imu_ingestor.h — IMU consumer on the MEDIA PLANE, mirroring LidarIngestor.
 *
 *  WHY THIS EXISTS
 *  ---------------
 *  The IMU used to arrive as DSR node attributes, read in modify_node_attrs_slot. That works, but it
 *  puts a ~125 Hz stream on the shared graph: four attribute writes per sample on one node, CRDT-
 *  replicated to every agent, waking every peer's attribute slot whether it wants the IMU or not. The
 *  media plane exists for exactly this — a dedicated DDS domain, isolated from the DSR domain so that
 *  high-rate churn cannot perturb cortex resync — and it already carried an ImuFrame type, an
 *  ImuPublisher, an ImuSubscriber and a descriptor-driven factory before this class was written.
 *
 *  ★ THE SIM2REAL ARGUMENT IS THE STRONGER ONE. On hardware the IMU component publishes to the DDS
 *  topic directly. If this agent consumes the topic, then the consumer code is IDENTICAL in simulation
 *  and on the robot, and the only difference is which process fills it — invisible from here. Reading
 *  DSR attributes instead would keep a sim-only path alive in the one place a sim/real divergence is
 *  most expensive: the thing being calibrated.
 *
 *  WHAT IT MUST PRESERVE, AND WHY EACH ONE BITES
 *  --------------------------------------------
 *   · integration_ts_ms(), not the wall stamp, as the buffer key. The odometry window bounds that
 *     imu_dtheta() brackets samples against are on the SIM clock when the source is simulated; keying
 *     the IMU on wall time would compare two different clocks and collapse coverage to zero rather
 *     than degrade it.
 *   · sim_clock->observe(). Also fed by the odometry path, so the map still binds without this — but
 *     the IMU is the faster channel and the better estimator of the ratio.
 *   · the dedup. It exists because the graph re-signalled the node for unrelated writes. That reason
 *     disappears on a dedicated topic, but "should no longer happen" is not a measurement, so the
 *     guard stays until one exists.
 *   · gyro_var. Carried so a consumer weights the sample by the noise the producer actually has,
 *     rather than by a nominal constant that may describe a different sensor.
 */
#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

#include "dsr/api/dsr_api.h"

#include "buffer_types.h"
#include "common_types.h"
#include "../../common/media_transport/media_transport.h"

namespace rc
{
class ImuIngestor
{
public:
    ImuIngestor(std::shared_ptr<DSR::DSRGraph> graph, ImuBuffer& buffer, SimClockMap& sim_clock,
                std::string node_name = "imu", std::string stream_key = "imu")
        : G_(std::move(graph)), buffer_(&buffer), sim_clock_(&sim_clock),
          node_name_(std::move(node_name)), stream_key_(std::move(stream_key)) {}
    ~ImuIngestor() { stop(); }

    void start();
    void stop();

    /// Samples accepted since the last stats report; for the health line.
    [[nodiscard]] std::uint64_t served() const { return served_.load(std::memory_order_relaxed); }

private:
    void loop();
    /// One drain. Brings the subscriber up lazily and throttled, exactly as LidarIngestor does: the
    /// factory verifies the node and stream exist and returns nullptr otherwise, so a retry is the
    /// correct response to "the producer has not advertised yet" rather than a startup failure.
    bool pump();

    std::shared_ptr<DSR::DSRGraph>              G_;
    ImuBuffer*                                  buffer_    = nullptr;
    SimClockMap*                                sim_clock_ = nullptr;
    std::string                                 node_name_, stream_key_;
    std::unique_ptr<rc::media::ImuSubscriber>   sub_;

    std::thread             thread_;
    std::atomic<bool>       running_{false};
    std::mutex              wake_mtx_;
    std::condition_variable wake_cv_;
    std::int64_t            last_key_ms_    = 0;
    std::int64_t            last_try_ms_    = 0;
    std::atomic<std::uint64_t> served_{0};
};
} // namespace rc
