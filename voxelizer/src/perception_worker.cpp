#include "perception_worker.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <print>
#include <thread>
#include <utility>

namespace rc
{

PerceptionWorker::~PerceptionWorker()
{
    stop();
}

bool PerceptionWorker::start(std::vector<std::unique_ptr<Stage>> stages, const Config& config, FrameSource source)
{
    stages_ = std::move(stages);
    config_ = config;
    source_ = std::move(source);

    bool any_ready = false;
    for (const auto& s : stages_)
        any_ready = any_ready || (s && s->ready());
    if (!any_ready)
    {
        std::println("[PerceptionWorker:{}] no stage ready — not starting", config_.name);
        return false;
    }

    stop_requested_.store(false, std::memory_order_relaxed);
    thread_ = std::thread(&PerceptionWorker::run, this);
    return true;
}

void PerceptionWorker::stop()
{
    stop_requested_.store(true, std::memory_order_relaxed);
    input_cv_.notify_all();
    if (thread_.joinable())
        thread_.join();
}

void PerceptionWorker::submit(PerceptionFrame&& frame)
{
    {
        std::scoped_lock lk(input_mutex_);
        pending_ = std::move(frame);   // latest wins
    }
    input_cv_.notify_one();
}

std::optional<PerceptionResult> PerceptionWorker::take_result()
{
    std::scoped_lock lk(result_mutex_);
    if (!result_new_)
        return std::nullopt;
    result_new_ = false;
    return std::move(result_);
}

std::optional<PerceptionResult> PerceptionWorker::latest_result() const
{
    std::scoped_lock lk(result_mutex_);
    return result_;   // copy — non-consuming, safe for multiple readers
}

Stage* PerceptionWorker::stage(std::string_view name)
{
    for (const auto& s : stages_)
        if (s && name == s->name())
            return s.get();
    return nullptr;
}

void PerceptionWorker::run()
{
    std::ofstream perf_csv;
    std::chrono::steady_clock::time_point perf_log_start;
    if (config_.perf_log)
    {
        perf_csv.open("etc/viewer_perf_" + config_.name + "_worker.csv", std::ios::trunc);
        perf_csv << "t_ms,stages_ms,n_det,stamp\n";
        perf_log_start = std::chrono::steady_clock::now();
    }

    const bool pull = static_cast<bool>(source_);

    // Runs the stage list on `in`, stores the result bundle, and (optionally) logs timing.
    const auto process = [&](PerceptionFrame&& in)
    {
        const auto t0 = std::chrono::steady_clock::now();
        PerceptionResult res;
        for (const auto& s : stages_)
            if (s && s->enabled() && s->ready())
                s->run(in, res);
        const double stages_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const std::size_t n_det = res.masks ? res.masks->size() : 0;
        const std::uint64_t stamp = in.stamp;
        res.frame = std::move(in);   // stages consumed `in`; hand the frame to the bundle for publish

        {
            std::scoped_lock lk(result_mutex_);
            result_ = std::move(res);
            result_new_ = true;
        }

        if (perf_csv.is_open())
        {
            const long long t_ms = static_cast<long long>(
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - perf_log_start).count());
            perf_csv << t_ms << ',' << stages_ms << ',' << n_det << ',' << stamp << '\n';
            perf_csv.flush();
        }
    };

    while (!stop_requested_.load(std::memory_order_relaxed))
    {
        if (pull)
        {
            // PULL: fetch a frame from the source, process if present, pace to the target period.
            const auto t0 = std::chrono::steady_clock::now();
            if (auto pf = source_(); pf.has_value())
                process(std::move(*pf));
            const auto target = std::chrono::milliseconds(std::max(1, config_.target_period_ms));
            const auto elapsed = std::chrono::steady_clock::now() - t0;
            if (elapsed < target)
                std::this_thread::sleep_for(target - elapsed);
        }
        else
        {
            // PUSH: wait for a submitted frame.
            PerceptionFrame in;
            {
                std::unique_lock lk(input_mutex_);
                input_cv_.wait(lk, [this] { return pending_.has_value() || stop_requested_.load(std::memory_order_relaxed); });
                if (stop_requested_.load(std::memory_order_relaxed))
                    break;
                in = std::move(*pending_);
                pending_.reset();
            }
            process(std::move(in));
        }
    }
}

} // namespace rc
