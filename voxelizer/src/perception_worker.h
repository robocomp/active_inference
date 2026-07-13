#pragma once

/*
 * perception_worker.h
 *
 * One camera source's inference thread. Generalises the old ZedYoloWorker: instead of a hardcoded seg
 * model it runs a CONFIGURABLE list of Stages (seg / pose / semantic / future) on each submitted frame,
 * sequentially (the GPU serialises inference anyway). Input is PUSHED — the main thread hands it a
 * DEEP-COPIED PerceptionFrame (latest-wins) — and it hands back a self-contained PerceptionResult bundle
 * {frame + each stage's slot}, so every published map pairs with its own frame's depth/transform/stamp.
 *
 * cv::Mat discipline: the frame handed in is an owned clone; the bundle is moved out on take_result() —
 * single owner at every step, never a shared/drawn-into buffer.
 */

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "perception_stage.h"

namespace rc
{

class PerceptionWorker
{
public:
    // Optional PULL source: called on the worker thread to fetch the next frame (nullopt = nothing new).
    // If set, the worker PULLS (paced by Config.target_period_ms) instead of waiting for submit() (PUSH).
    using FrameSource = std::function<std::optional<PerceptionFrame>()>;

    struct Config
    {
        std::string name = "zed";        // for logs / perf csv (etc/viewer_perf_<name>_worker.csv)
        bool        perf_log = false;
        int         target_period_ms = 0; // pull-mode pacing (ignored in push mode)
    };

    ~PerceptionWorker();

    // Takes ownership of the stages and starts the thread. If `source` is set the worker pulls, else it
    // waits for submit(). Returns false if no stage is ready.
    bool start(std::vector<std::unique_ptr<Stage>> stages, const Config& config, FrameSource source = {});
    void stop();

    // PUSH mode: hand a frame for inference (latest-wins). frame.rgbd.rgb MUST be an owned deep copy.
    void submit(PerceptionFrame&& frame);

    // Consume the newest completed bundle (single reader); nullopt if nothing new since the last call.
    std::optional<PerceptionResult> take_result();
    // Peek the newest completed bundle without consuming (for multiple readers, e.g. popup + publish).
    std::optional<PerceptionResult> latest_result() const;

    // Reach a stage by name (for compose passthroughs / enable toggles). nullptr if absent.
    Stage* stage(std::string_view name);

private:
    void run();

    std::vector<std::unique_ptr<Stage>> stages_;
    Config config_;
    FrameSource source_;

    std::thread thread_;
    std::atomic<bool> stop_requested_{false};

    std::mutex input_mutex_;
    std::condition_variable input_cv_;
    std::optional<PerceptionFrame> pending_;

    mutable std::mutex result_mutex_;
    std::optional<PerceptionResult> result_;
    bool result_new_ = false;
};

} // namespace rc
