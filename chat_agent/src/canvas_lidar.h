#pragma once
/*
 * canvas_lidar.h — the 3-D LiDAR clouds for the canvas' point-cloud page ("show me the lidars").
 *
 * The same picture as robot_concept's "View data" on helios / bpearl (media_stream_viewers.h LidarStreamViewer):
 * both rings, in the ROBOT (`body`) frame, two colours. Unlike that viewer it has no thread of its own — chat_agent
 * is main-thread only — so the canvas timer polls it (non-blocking drain, latest frame only):
 *  - subscribers come from the shared descriptor-driven factory (media_transport.h make_lidar_subscriber_from_graph),
 *    created on the MAIN thread once the sensor node exists, retried at most every retry_period (a robot without a
 *    bpearl is a permanent, quiet state: the factory returns nullptr silently for a missing node);
 *  - the frame handed to poll()'s callback is a loaned SHM view ⇒ its points are copied out before it returns;
 *  - points arrive in the SENSOR frame and are mapped to `body` with the RT chain read on the main thread
 *    (inner.get_transformation_matrix("body", sensor), optional checked; identity if it does not resolve). An
 *    inverted sensor drawn in its own frame shows the room upside down — robot_concept learnt that on 2026-08-29.
 * Subscribers are released when the page closes (release()), so no DDS reader idles while nothing shows them.
 */

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <QVector3D>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

namespace rc::media { class LidarSubscriber; }

namespace rc::chat
{

class LidarGrabber
{
public:
    explicit LidarGrabber(std::shared_ptr<DSR::DSRGraph> G);
    ~LidarGrabber();

    /// Choose the sensors to show ("helios", "bpearl"; empty = both) and bring their subscribers up. Main thread.
    void want(std::vector<std::string> sensors);
    /// Retry the chosen sensors' subscribers that are not up yet; keeps the choice. Self-throttled; main thread.
    void retry();
    /// Drain the newest frame of each sensor. True when any cloud changed. Main thread.
    bool poll(DSR::InnerEigenAPI& inner);
    void release();   ///< drop the subscribers (page closed)

    [[nodiscard]] const std::vector<QVector3D>& cloud(std::size_t i) const { return rings_[i].points; }
    [[nodiscard]] std::uint64_t stamp_ms() const { return stamp_ms_; }
    [[nodiscard]] std::string status() const;   ///< "helios 28734 pts · bpearl 15120 pts" / why none

private:
    struct Ring
    {
        std::string node;
        std::unique_ptr<rc::media::LidarSubscriber> sub;
        std::vector<QVector3D> points;   ///< body frame, last frame received
    };
    std::shared_ptr<DSR::DSRGraph> G_;
    Ring rings_[2];
    std::vector<std::string> wanted_;
    std::uint64_t stamp_ms_ = 0;
    std::chrono::steady_clock::time_point next_try_{};
    static constexpr std::chrono::seconds retry_period{3};
};

}   // namespace rc::chat
