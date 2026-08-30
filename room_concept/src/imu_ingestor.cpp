/*  imu_ingestor.cpp — see imu_ingestor.h for why the IMU moved off the DSR graph. */
#include "imu_ingestor.h"
#include <pthread.h>   // pthread_setname_np: name the worker so a per-thread CPU sample attributes itself

#include <QDateTime>
#include <QDebug>
#include <algorithm>
#include <chrono>

namespace rc
{

void ImuIngestor::start()
{
    if (running_.exchange(true))
        return;
    thread_ = std::thread([this] { pthread_setname_np(pthread_self(), "imu-ingest"); loop(); });
}

void ImuIngestor::stop()
{
    if (not running_.exchange(false))
        return;
    wake_cv_.notify_all();
    if (thread_.joinable())
        thread_.join();
    // Drop the subscriber on THIS thread, while the graph is still alive: it is the thread that
    // created it, and tearing a DDS reader down from another one is the kind of asymmetry that turns
    // a clean shutdown into a crash nobody can reproduce.
    sub_.reset();
}

void ImuIngestor::loop()
{
    while (running_.load(std::memory_order_acquire))
    {
        if (not pump())
        {
            // Nothing available (or no subscriber yet). wait_and_poll already blocked, so this is
            // only the not-yet-connected path; sleep briefly rather than spin.
            std::unique_lock lk(wake_mtx_);
            wake_cv_.wait_for(lk, std::chrono::milliseconds(20),
                              [this] { return not running_.load(std::memory_order_acquire); });
        }
    }
}

bool ImuIngestor::pump()
{
    const auto now_ms = QDateTime::currentMSecsSinceEpoch();
    if (sub_ == nullptr)
    {
        // Throttled retry: the producer advertises its descriptor during ITS initialize(), which may
        // land after ours. A failure here is "not yet", not "never".
        if (now_ms - last_try_ms_ < 1000)
            return false;
        last_try_ms_ = now_ms;
        if (G_ == nullptr)
            return false;
        sub_ = rc::media::make_imu_subscriber_from_graph(*G_, node_name_, stream_key_);
        if (sub_ == nullptr)
            return false;
        qInfo() << "[ImuIngestor] media-plane IMU subscriber up on node" << node_name_.c_str()
                << "stream" << stream_key_.c_str()
                << "| data_sharing=" << (sub_->data_sharing_active() ? "ON (zero-copy)" : "off");
    }

    const int n = sub_->wait_and_poll([this](const rc::media::ImuFrame& f, std::int64_t)
    {
        ImuReading r;
        r.gyro_z       = f.gyro_z();
        r.gyro_var     = f.gyro_var();
        r.acc_x        = f.acc_x();
        r.acc_y        = f.acc_y();
        r.acc_var      = f.acc_var();
        r.source_ts_ms = static_cast<std::int64_t>(f.stamp_ms());
        r.sim_ts_ms    = static_cast<std::int64_t>(f.sim_stamp_ms());
        // 0 IS the "not simulated" signal — the producer sends the sim clock or nothing, so no
        // separate boolean has to be kept consistent with it.
        r.simulated    = (r.sim_ts_ms > 0);
        r.recv_ts_ms   = QDateTime::currentMSecsSinceEpoch();

        // Key on the clock the RATE is measured against. The odometry window bounds that
        // imu_dtheta() brackets these samples against are on the sim clock when the source is
        // simulated; keying on wall time would compare two clocks and take coverage to zero.
        const std::int64_t key = r.integration_ts_ms();
        if (key <= 0 or key == last_key_ms_)
            return;                                   // kept until "no longer needed" is measured
        last_key_ms_ = key;

        if (r.simulated and sim_clock_ != nullptr)
            sim_clock_->observe(r.source_ts_ms, r.sim_ts_ms);

        buffer_->put<0>(std::move(r), static_cast<std::uint64_t>(key));
        served_.fetch_add(1, std::memory_order_relaxed);
    }, /*timeout_ms=*/20);

    return n > 0;
}

} // namespace rc
