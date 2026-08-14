#include "perception_worker.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <locale>
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
    // PER-STAGE RADIOGRAPHY. One column per stage, plus `source_ms` for the acquisition half of the loop
    // (drain the media plane + deep-copy the frame + resolve the transform), which is NOT inference and
    // was invisible while only the stage total was logged — it measured 23 ms of an 85 ms budget on the
    // zed worker, i.e. 27% of the frame with no model in it. A total can only ever say "too slow"; this
    // says WHICH.
    //   0 in a stage column = that stage did not run this frame (disabled, not ready, or decimated away).
    //   Deliberately distinguishable from a small non-zero cost — a decimated stage is free on the
    //   frames it skips, and averaging its zeros in would understate what it costs when it DOES run.
    std::vector<double> stage_ms(stages_.size(), 0.0);
    if (config_.perf_log)
    {
        perf_csv.open("etc/viewer_perf_" + config_.name + "_worker.csv", std::ios::trunc);
        // Pin the CLASSIC locale on the way OUT (CLAUDE.md). These machines run LANG=es_ES.UTF-8 and this
        // is a Qt program, so the C library is already on comma decimals; C++ streams format through the
        // C++ global locale, which is "C" only until someone calls std::locale::global. If that ever
        // happens this file starts emitting "62,4" and every reader of it silently truncates to 62.
        perf_csv.imbue(std::locale::classic());
        perf_csv << "t_ms,source_ms,stages_ms";
        for (const auto& s : stages_)
            perf_csv << ',' << (s ? s->name() : "null") << "_ms";
        perf_csv << ",n_det,stamp\n";
        perf_log_start = std::chrono::steady_clock::now();
    }

    const bool pull = static_cast<bool>(source_);

    // Runs the stage list on `in`, stores the result bundle, and (optionally) logs timing.
    // `source_ms` is the cost of ACQUIRING this frame, measured by the caller (0 in push mode, where
    // the frame was handed to us).
    const auto process = [&](PerceptionFrame&& in, double source_ms)
    {
        const auto t0 = std::chrono::steady_clock::now();
        PerceptionResult res;
        std::fill(stage_ms.begin(), stage_ms.end(), 0.0);
        for (std::size_t i = 0; i < stages_.size(); ++i)
        {
            const auto& s = stages_[i];
            if (not s or not s->enabled() or not s->ready())
                continue;
            const auto s0 = std::chrono::steady_clock::now();
            s->run(in, res);
            stage_ms[i] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
        }
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
            perf_csv << t_ms << ',' << source_ms << ',' << stages_ms;
            for (const double ms : stage_ms)
                perf_csv << ',' << ms;
            perf_csv << ',' << n_det << ',' << stamp << '\n';
            perf_csv.flush();
        }
    };

    while (!stop_requested_.load(std::memory_order_relaxed))
    {
        if (pull)
        {
            // PULL: fetch a frame from the source, process if present, pace to the target period.
            // The source call is timed SEPARATELY: on the zed worker it drains the media plane, deep-copies
            // the frame and resolves room<-zed, none of which is inference, and it was hidden inside the
            // loop period until now. A nullopt fetch (no new frame) is not logged — there is no row without
            // a processed frame — so its cost shows up only as loop-period slack, which is correct.
            const auto t0 = std::chrono::steady_clock::now();
            auto pf = source_();
            const double source_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (pf.has_value())
                process(std::move(*pf), source_ms);
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
            process(std::move(in), 0.0);   // PUSH: the frame was handed to us — no acquisition cost here
        }
    }
}

} // namespace rc
