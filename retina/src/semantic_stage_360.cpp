#include "semantic_stage_360.h"

#include <algorithm>
#include <print>

namespace rc
{

SemanticStage360::SemanticStage360(const rc::semantic::YoloSemanticProcessor::Config& cfg,
                                   std::shared_ptr<StripSchedule> sched,
                                   int n_strips, int decimation, bool want_scores)
    : sched_(std::move(sched))
    , n_strips_(std::max(1, n_strips))
    , decimation_(std::max(1, decimation))
    , want_scores_(want_scores)
{
    try
    {
        sem_ = std::make_unique<rc::semantic::YoloSemanticProcessor>();
        sem_->configure(cfg);
    }
    catch (const std::exception& e)
    {
        // A stage that fails to load must DISABLE ITSELF, not take the worker down: PerceptionWorker
        // refuses to start only when NO stage is ready, so the panorama keeps producing YOLO-seg
        // bearings without the semantic ones rather than the whole ricoh path going silent.
        std::println("[SemanticStage360] disabled — failed to load model {}: {}", cfg.model_path, e.what());
        sem_.reset();
    }
}

void SemanticStage360::run(const PerceptionFrame& in, PerceptionResult& out)
{
    if (not ready() or not in.is_360 or in.rgbd.bgr.empty())
        return;
    if (counter_++ % static_cast<std::uint64_t>(decimation_) != 0)
        return;   // skipped frames publish nothing rather than a stale map: see out.semantic below

    const cv::Mat& pano = in.rgbd.bgr;
    const int W = pano.cols, H = pano.rows;
    const int strip_w = W / n_strips_;
    if (strip_w <= 0)
        return;

    // The window seg chose for THIS frame. Empty = it is looking at everything, so we do too.
    std::vector<int> strips = sched_ ? sched_->current() : std::vector<int>{};
    if (strips.empty())
    {
        strips.resize(static_cast<std::size_t>(n_strips_));
        for (int s = 0; s < n_strips_; ++s) strips[static_cast<std::size_t>(s)] = s;
    }

    // Full-panorama canvas, everything IGNORE until a strip fills it. This is the whole trick: the map
    // handed downstream is panorama-sized, so SemanticMaskStage's connected components come out in
    // GLOBAL coordinates and BearingStage converts them without knowing a strip existed.
    // ★NOT carried over between frames. A stale strip would let a component span "what I see now" and
    // "what I saw 100 ms ago" as though they were one observation, and the bearing of that merged blob
    // belongs to neither. Each frame reports only what it actually just looked at — the same contract
    // YOLO-seg follows on this panorama.
    cv::Mat labels(H, W, CV_8UC1, cv::Scalar(rc::semantic::IGNORE_LABEL));
    cv::Mat scores;
    if (want_scores_)
        scores = cv::Mat(H, W, CV_32FC1, cv::Scalar(0.0f));

    // No circular overlap here, unlike YOLO-seg. Overlap exists there so a seam-straddling OBJECT
    // survives as one box; here the pieces are pasted back into the same global canvas at the same
    // pixels, so a component that spans a seam is only split when the two halves are in strips looked
    // at on DIFFERENT frames — and then they are genuinely two observations, not one.
    for (const int s : strips)
    {
        if (s < 0 or s >= n_strips_)
            continue;
        const int x0 = s * strip_w;
        const int w  = (s == n_strips_ - 1) ? (W - x0) : strip_w;   // last strip absorbs the remainder
        if (w <= 0 or x0 + w > W)
            continue;

        const cv::Rect roi(x0, 0, w, H);
        (void) sem_->segment(pano(roi), want_scores_);   // refreshes last_map(), sized to the strip
        const auto& m = sem_->last_map();
        if (m.labels.empty() or m.labels.cols != w or m.labels.rows != H)
            continue;   // model returned something unexpected for this strip — skip it, keep the rest

        m.labels.copyTo(labels(roi));   // copyTo into a ROI is a deep write into OUR buffer
        if (want_scores_ and not m.scores.empty() and m.scores.size() == m.labels.size())
            m.scores.copyTo(scores(roi));
    }

    // `labels`/`scores` are buffers this stage allocated and nothing else references, so handing them
    // over is already the deep copy the worker→main thread boundary requires (CLAUDE.md cv::Mat rule) —
    // unlike the ZED stage, which must clone because it forwards the processor's cached map.
    out.semantic = rc::semantic::SemanticMap{std::move(labels), std::move(scores)};
    out.semantic_fresh = true;
}

}   // namespace rc
