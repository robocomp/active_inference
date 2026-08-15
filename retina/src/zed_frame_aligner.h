#pragma once

// Timestamp-aligns the two ZED media streams. RGB and depth arrive on separate DDS topics but share the
// SAME per-grab acquisition stamp_ms; this buffers both and returns the time-aligned pair, so a mask (RGB
// frame F) is always deprojected against depth frame F — eliminating the "latest of each" RGB/depth skew.
//
// PIMPL on purpose: the underlying BufferSync pulls a threadpool + helper templates that CLASH (ThreadPool
// ctor arity, is_iterable redefinition) with the cortex/DSR headers. Keeping BufferSync in the .cpp lets
// DSR-touching TUs (media_plane_source) use this without ever including doublebuffer_sync.h.

#include <opencv2/core.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

struct ZedRgbFrame   { cv::Mat bgr; int width = 0; int height = 0; };
struct ZedDepthFrame { std::vector<float> depth; int width = 0; int height = 0; };

class ZedFrameAligner
{
public:
    explicit ZedFrameAligner(std::size_t history = 8);
    ~ZedFrameAligner();
    ZedFrameAligner(const ZedFrameAligner&) = delete;
    ZedFrameAligner& operator=(const ZedFrameAligner&) = delete;

    // put_* are const (they mutate the hidden buffer through the PIMPL) so callers on a const path
    // (the media drain) can use them, matching the pre-existing mutable-cache design.
    void put_rgb(ZedRgbFrame&& f, std::uint64_t stamp) const;
    void put_depth(ZedDepthFrame&& f, std::uint64_t stamp) const;

    // Newest RGB stamp actually committed to the buffer (put is async), 0 if none.
    std::uint64_t newest_rgb_stamp() const;

    // The (rgb, depth) pair nearest ref_stamp within max_diff_ms; nullopt if either channel has no match.
    std::optional<std::pair<ZedRgbFrame, ZedDepthFrame>> read_aligned(std::uint64_t ref_stamp,
                                                                      std::uint64_t max_diff_ms) const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
