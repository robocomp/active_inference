#include "semantic_mask_stage.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <print>

namespace rc
{

SemanticMaskStage::SemanticMaskStage(std::vector<std::pair<int, std::string>> accepted_classes,
                                     float min_area_frac, float overlap_drop_frac, int morph_kernel,
                                     float score_default)
    : accepted_(std::move(accepted_classes))
    , min_area_frac_(min_area_frac)
    , overlap_drop_frac_(std::clamp(overlap_drop_frac, 0.0f, 1.0f))
    , morph_kernel_(morph_kernel)
    , score_default_(score_default)
{
    std::string cls;
    for (const auto& [id, label] : accepted_)
        cls += (cls.empty() ? "" : ", ") + label + "(" + std::to_string(id) + ")";
    std::println("[SemanticMasks] enabled for {} classes: {}  min_area_frac={} overlap_drop={} morph={}",
                 accepted_.size(), cls, min_area_frac_, overlap_drop_frac_, morph_kernel_);
}

void SemanticMaskStage::run(const PerceptionFrame& in, PerceptionResult& out)
{
    (void) in;
    // Only extract on a FRESH semantic map (matches the current frame). Held-last maps on decimated cycles
    // would reproject stale pixels against current depth → skew; skip them entirely.
    if (accepted_.empty() || not out.semantic || not out.semantic_fresh)
        return;

    const cv::Mat& labels = out.semantic->labels;   // CV_8UC1, frame-sized, class id per pixel (255 = ignore)
    const cv::Mat& scores = out.semantic->scores;    // CV_32FC1 frame-sized, or empty
    if (labels.empty())
        return;

    const int W = labels.cols, H = labels.rows;
    // ★THE DENOMINATOR IS WHAT WAS LOOKED AT, NOT THE CANVAS. min_area_frac is "how much of an
    // OBSERVATION must a region cover to be worth keeping", and that only equals a fraction of the map
    // when the model ran over the whole map. It does on the ZED; it does NOT on the 360, where
    // SemanticStage360 infers one 640-wide strip and pastes it into a 1920-wide canvas — everything
    // else IGNORE. Dividing by the canvas there demanded 3× the area on the ONE camera whose angular
    // resolution is already the coarser of the two, and the value 0.003 was never the wrong number:
    // it was being applied to an area nobody had looked at.
    // Measured 2026-08-16 on 18 saved panoramas: `hood` regions run 842–6338 px against a canvas-derived
    // floor of 5529 — 3 of 13 survived. Against the strip's own 1843, 8 of 13 do. A cabinet run is
    // 10k–52k px, which is exactly why this stayed invisible: it only ever bit the smallest class.
    // 0 ⇒ the producer looked at everything (ZED, or one predating the field) ⇒ unchanged behaviour.
    const long long inferred = out.semantic->inferred_area_px > 0
        ? out.semantic->inferred_area_px
        : static_cast<long long>(W) * H;
    const int min_area = std::max(1, static_cast<int>(min_area_frac_ * static_cast<double>(inferred)));

    // YOLO-seg priority mask: OR of every existing detection's mask (thresholded @127). A semantic component
    // mostly inside this region is a duplicate of an authoritative YOLO detection → dropped.
    cv::Mat yolo_union;
    if (out.masks and not out.masks->empty())
    {
        yolo_union = cv::Mat::zeros(H, W, CV_8UC1);
        for (const auto& d : *out.masks)
        {
            if (d.mask.empty() || d.mask.size() != yolo_union.size() || d.mask.type() != CV_8UC1)
                continue;
            cv::Mat bin;
            cv::threshold(d.mask, bin, 127, 255, cv::THRESH_BINARY);
            cv::bitwise_or(yolo_union, bin, yolo_union);
        }
    }

    const cv::Mat kernel = morph_kernel_ > 1
        ? cv::getStructuringElement(cv::MORPH_ELLIPSE, {morph_kernel_, morph_kernel_})
        : cv::Mat();

    std::vector<SegDetection> new_dets;
    std::vector<int>          per_class(accepted_.size(), 0);

    for (std::size_t ci = 0; ci < accepted_.size(); ++ci)
    {
        const int         id    = accepted_[ci].first;
        const std::string& label = accepted_[ci].second;

        cv::Mat cls = (labels == id);   // fresh CV_8UC1 0/255
        if (cv::countNonZero(cls) < min_area)
            continue;                    // this class barely present → skip the (cheap) CC pass
        if (not kernel.empty())
        {
            cv::morphologyEx(cls, cls, cv::MORPH_OPEN, kernel);    // drop speckle
            cv::morphologyEx(cls, cls, cv::MORPH_CLOSE, kernel);   // fill pinholes
        }

        cv::Mat lbls, stats, cents;
        const int n = cv::connectedComponentsWithStats(cls, lbls, stats, cents, 8, CV_32S);
        for (int k = 1; k < n; ++k)   // 0 = background
        {
            const int area = stats.at<int>(k, cv::CC_STAT_AREA);
            if (area < min_area)
                continue;
            const cv::Rect bbox(stats.at<int>(k, cv::CC_STAT_LEFT), stats.at<int>(k, cv::CC_STAT_TOP),
                                stats.at<int>(k, cv::CC_STAT_WIDTH), stats.at<int>(k, cv::CC_STAT_HEIGHT));

            cv::Mat comp = (lbls == k);   // fresh full-frame CV_8UC1 0/255 — this instance's silhouette

            // YOLO priority: drop if the component is mostly covered by a YOLO-seg mask.
            if (not yolo_union.empty())
            {
                cv::Mat inter;
                cv::bitwise_and(comp(bbox), yolo_union(bbox), inter);
                if (static_cast<double>(cv::countNonZero(inter)) / area >= overlap_drop_frac_)
                    continue;
            }

            // Score = mean per-pixel semantic confidence over the component (analogue of the YOLO max-softmax
            // class score). Falls back to score_default when the model didn't produce a score map.
            float conf = score_default_;
            if (not scores.empty() and scores.size() == labels.size() and scores.type() == CV_32FC1)
            {
                double sum = 0.0; long cnt = 0;
                for (int y = bbox.y; y < bbox.y + bbox.height; ++y)
                {
                    const std::uint8_t* mrow = comp.ptr<std::uint8_t>(y);
                    const float*        srow = scores.ptr<float>(y);
                    for (int x = bbox.x; x < bbox.x + bbox.width; ++x)
                        if (mrow[x]) { sum += srow[x]; ++cnt; }
                }
                if (cnt > 0)
                    conf = static_cast<float>(sum / cnt);
            }

            // class_id offset by 1000 keeps semantic ids clear of the COCO id range (consumers key on label).
            new_dets.push_back(SegDetection{bbox, 1000 + id, label, conf, comp});
            ++per_class[ci];
        }
    }

    if (new_dets.empty())
        return;
    if (not out.masks)
        out.masks = std::vector<SegDetection>{};
    for (auto& d : new_dets)
        out.masks->push_back(std::move(d));

    // Throttled summary so extraction is visible live without flooding the log.
    if (run_count_++ % 30 == 0)
    {
        std::string per;
        for (std::size_t ci = 0; ci < accepted_.size(); ++ci)
            per += (per.empty() ? "" : " ") + accepted_[ci].second + "=" + std::to_string(per_class[ci]);
        // min_area is printed because it is now DERIVED per frame, not a constant: on the 360 it tracks
        // how many strips were scheduled. A reader comparing two runs needs the floor that was in force.
        std::println("[SemanticMasks] +{} masks ({}) min_area={} of {} px looked at",
                     new_dets.size(), per, min_area, inferred);
    }
}

} // namespace rc
