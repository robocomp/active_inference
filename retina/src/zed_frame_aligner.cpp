#include "zed_frame_aligner.h"

// This TU includes doublebuffer_sync.h and NOTHING from DSR/cortex — that isolation is the whole point
// of the PIMPL (the two clash on ThreadPool / is_iterable). Do not add DSR/genericworker includes here.
#include "doublebuffer_sync/doublebuffer_sync.h"

struct ZedFrameAligner::Impl
{
    using Buffer = BufferSync<InOut<ZedRgbFrame, ZedRgbFrame>, InOut<ZedDepthFrame, ZedDepthFrame>>;
    Buffer buf;   // channel 0 = rgb, channel 1 = depth
    explicit Impl(std::size_t history) : buf(history) {}
};

ZedFrameAligner::ZedFrameAligner(std::size_t history) : p_(std::make_unique<Impl>(history)) {}
ZedFrameAligner::~ZedFrameAligner() = default;

void ZedFrameAligner::put_rgb(ZedRgbFrame&& f, std::uint64_t stamp) const
{
    p_->buf.put<0>(std::move(f), stamp);
}

void ZedFrameAligner::put_depth(ZedDepthFrame&& f, std::uint64_t stamp) const
{
    p_->buf.put<1>(std::move(f), stamp);
}

std::uint64_t ZedFrameAligner::newest_rgb_stamp() const
{
    const auto snap = p_->buf.get_snapshot_with_timestamps<0>();
    return snap.empty() ? 0u : snap.back().second;
}

std::optional<std::pair<ZedRgbFrame, ZedDepthFrame>>
ZedFrameAligner::read_aligned(std::uint64_t ref_stamp, std::uint64_t max_diff_ms) const
{
    auto [rgb, depth] = p_->buf.read(ref_stamp, max_diff_ms);
    if (!rgb.has_value() || !depth.has_value())
        return std::nullopt;
    return std::make_pair(std::move(*rgb), std::move(*depth));
}
