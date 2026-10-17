/*
 * frame_history.h — the last few ZED BGR frames by capture stamp, so a mask can be cropped from the frame
 * it was computed on. Fed from the agent's RgbIngestor pump; read on the same (compute) thread. push()
 * deep-copies: the ingestor reuses its buffer, and a shallow cv::Mat copy across that is a heap smash.
 */
#pragma once

#include <cstdint>
#include <deque>
#include <utility>

#include <opencv2/core.hpp>

namespace rc::atlas
{
class FrameHistory
{
public:
    explicit FrameHistory(std::size_t capacity = 8) : cap_(capacity) {}
    void push(std::uint64_t stamp_ms, const cv::Mat& bgr)
    {
        if (bgr.empty()) return;
        q_.emplace_back(stamp_ms, bgr.clone());
        while (q_.size() > cap_) q_.pop_front();
    }
    /// Exact stamp match (the producer stamps masks with the RGB frame's capture stamp); nullptr if gone.
    [[nodiscard]] const cv::Mat* at(std::uint64_t stamp_ms) const
    {
        for (const auto& [s, m] : q_) if (s == stamp_ms) return &m;
        return nullptr;
    }
    [[nodiscard]] std::size_t size() const noexcept { return q_.size(); }
private:
    std::size_t cap_;
    std::deque<std::pair<std::uint64_t, cv::Mat>> q_;
};
} // namespace rc::atlas
