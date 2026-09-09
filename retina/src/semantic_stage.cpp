#include "semantic_stage.h"

#include <algorithm>
#include <print>

namespace rc
{

SemanticStage::SemanticStage(const rc::semantic::YoloSemanticProcessor::Config& cfg, int decimation,
                             bool want_scores)
    : decimation_(std::max(1, decimation))
    , want_scores_(want_scores)
{
    try
    {
        sem_ = std::make_unique<rc::semantic::YoloSemanticProcessor>();
        sem_->configure(cfg);
    }
    catch (const std::exception& e)
    {
        std::println("[SemanticStage] disabled — failed to load model {}: {}", cfg.model_path, e.what());
        sem_.reset();
    }
}

void SemanticStage::run(const PerceptionFrame& in, PerceptionResult& out)
{
    if (!ready() || in.rgbd.bgr.empty())
        return;
    const bool run_now = (counter_++ % static_cast<std::uint64_t>(decimation_) == 0);
    if (run_now)
        (void) sem_->segment(in.rgbd.bgr, want_scores_);   // refreshes last_map_ (reused between runs)
    // DEEP-COPY across the worker→main thread boundary (CLAUDE.md cv::Mat rule): last_map() aliases the
    // processor's cached buffer, so a shallow copy would share it with the drained bundle destroyed on the
    // main thread. Own independent buffers here so no cv::Mat header/refcount is touched from two threads.
    const auto& m = sem_->last_map();
    rc::semantic::SemanticMap copy;
    copy.labels           = m.labels.clone();
    copy.scores           = m.scores.clone();
    copy.inferred_area_px = m.inferred_area_px;
    // ★The graded posteriors must cross this boundary too, or graded() is false downstream no matter
    // which model is loaded — the whole point of the -probs export is that an absence can be weighted
    // by how decisively the classifier lost, and that number lives only in `probs`. Same deep-copy
    // rule as labels/scores: these alias the processor's cached planes.
    copy.prob_class_ids = m.prob_class_ids;
    copy.probs_src_size = m.probs_src_size;
    copy.probs.reserve(m.probs.size());
    for (const auto& plane : m.probs)
        copy.probs.push_back(plane.empty() ? cv::Mat{} : plane.clone());
    out.semantic = std::move(copy);
    out.semantic_fresh = run_now;
}

} // namespace rc
