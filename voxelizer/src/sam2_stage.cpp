#include "sam2_stage.h"

#include <algorithm>
#include <cmath>
#include <print>

namespace rc
{

namespace
{
constexpr float kBleedBand = 0.15f;   // m: an in-mask pixel this far from the mask median depth is "bleed"

struct DepthStat
{
    int    n = 0;         // valid-depth pixel count
    double bleed = 0.0;   // fraction > kBleedBand from the median depth (dominated by object extent for tables)
    double mad = 0.0;     // median-absolute-deviation of in-mask depth (m) — robust, scale-adaptive dispersion
    double far_tail = 0.0;// fraction FARTHER than p75 + 3·IQR-ish → the true background tail behind the object
};

// Robust depth statistics of a mask over its valid-depth pixels (within its bbox). The headline is `mad`:
// background bleed and edge overshoot inflate the in-mask depth dispersion, so a tighter mask lowers MAD —
// with no magic threshold. `far_tail` isolates the one-sided background mode (pixels beyond an upper fence).
DepthStat depth_stat(const cv::Mat& mask, const std::vector<float>& depth, int W, int H, const cv::Rect& bb)
{
    DepthStat s;
    if (mask.empty() || depth.empty() || W <= 0 || H <= 0)
        return s;
    const cv::Rect fr = bb & cv::Rect(0, 0, std::min(W, mask.cols), std::min(H, mask.rows));
    std::vector<float> d;
    d.reserve(static_cast<std::size_t>(fr.width) * fr.height);
    for (int y = fr.y; y < fr.y + fr.height; ++y)
    {
        const std::uint8_t* mrow = mask.ptr<std::uint8_t>(y);
        for (int x = fr.x; x < fr.x + fr.width; ++x)
        {
            if (mrow[x] == 0) continue;
            const float z = depth[static_cast<std::size_t>(y) * W + x];
            if (std::isfinite(z) && z > 0.f) d.push_back(z);
        }
    }
    s.n = static_cast<int>(d.size());
    if (d.size() < 8) return s;                          // too few points to be meaningful

    const auto pct = [&](double q) {
        const std::size_t k = std::clamp<std::size_t>(static_cast<std::size_t>(q * (d.size() - 1)), 0, d.size() - 1);
        std::nth_element(d.begin(), d.begin() + k, d.end());
        return d[k];
    };
    const float med = pct(0.5);
    const float q25 = pct(0.25), q75 = pct(0.75);
    const float fence = q75 + 1.5f * (q75 - q25);        // upper Tukey fence → background sits beyond it

    std::size_t off = 0, far = 0;
    std::vector<float> absdev; absdev.reserve(d.size());
    for (float z : d)
    {
        if (std::fabs(z - med) > kBleedBand) ++off;
        if (z > fence) ++far;
        absdev.push_back(std::fabs(z - med));
    }
    const std::size_t mid = absdev.size() / 2;
    std::nth_element(absdev.begin(), absdev.begin() + mid, absdev.end());
    s.mad      = absdev[mid];
    s.bleed    = static_cast<double>(off) / d.size();
    s.far_tail = static_cast<double>(far) / d.size();
    return s;
}

double mask_iou(const cv::Mat& a, const cv::Mat& b, const cv::Rect& bb)
{
    if (a.empty() || b.empty() || a.size() != b.size()) return 0.0;
    const cv::Rect r = bb & cv::Rect(0, 0, a.cols, a.rows);
    if (r.width <= 0 || r.height <= 0) return 0.0;
    cv::Mat inter, uni;
    cv::bitwise_and(a(r), b(r), inter);
    cv::bitwise_or (a(r), b(r), uni);
    const double u = cv::countNonZero(uni);
    return u > 0.0 ? cv::countNonZero(inter) / u : 0.0;
}
}  // namespace

Sam2Stage::Sam2Stage(const rc::sam2::Config& cfg, std::vector<std::string> refine_labels, int decimation,
                     bool metrics_log)
    : refine_labels_(std::move(refine_labels))
    , decimation_(std::max(1, decimation))
    , metrics_log_(metrics_log)
{
    try
    {
        refiner_ = std::make_unique<rc::sam2::Sam2Refiner>(cfg);
    }
    catch (const std::exception& e)
    {
        std::println("[Sam2Stage] disabled — failed to load SAM2: {}", e.what());
        refiner_.reset();
    }

    if (metrics_log_ && refiner_)
    {
        metrics_csv_.open("etc/sam2_refine_metrics.csv", std::ios::trunc);
        metrics_csv_ << "t_ms,stamp,label,yolo_px,sam2_px,area_ratio,"
                        "yolo_mad_cm,sam2_mad_cm,mad_drop_cm,"
                        "yolo_fartail,sam2_fartail,fartail_drop,"
                        "yolo_bleed,sam2_bleed,bleed_drop,iou_yolo_sam2\n";
        metrics_t0_ = std::chrono::steady_clock::now();
    }
}

void Sam2Stage::run(const PerceptionFrame& in, PerceptionResult& out)
{
    if (!ready() || in.rgbd.rgb.empty() || !out.masks)
        return;

    // Keep only the target classes (empty filter → all). Refining every YOLO box wastes the encoder;
    // the concept fitters only consume table/bottle/chair support points.
    std::vector<SegDetection> targets;
    targets.reserve(out.masks->size());
    for (const auto& d : *out.masks)
        if (refine_labels_.empty() ||
            std::find(refine_labels_.begin(), refine_labels_.end(), d.label) != refine_labels_.end())
            targets.push_back(d);

    const bool run_now = (counter_++ % static_cast<std::uint64_t>(decimation_) == 0);
    if (run_now && !targets.empty())
    {
        last_refined_ = refiner_->refine(in.rgbd.rgb, targets, /*is_bgr=*/true);
        if (metrics_log_ && last_refined_)
            log_metrics(in, targets, *last_refined_);
    }
    else if (run_now)                       // enabled cycle but nothing to refine → clear held-last
        last_refined_ = std::vector<SegDetection>{};

    out.refined_masks = last_refined_;      // may be the held-last set on decimated cycles
    out.refined_fresh = run_now;
}

void Sam2Stage::log_metrics(const PerceptionFrame& in,
                            const std::vector<SegDetection>& yolo,
                            const std::vector<SegDetection>& sam2)
{
    if (!metrics_csv_.is_open())
        return;
    const int W = in.rgbd.width, H = in.rgbd.height;
    const auto& depth = in.rgbd.depth;
    const long long t_ms = static_cast<long long>(
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - metrics_t0_).count());

    const std::size_t n = std::min(yolo.size(), sam2.size());
    for (std::size_t i = 0; i < n; ++i)
    {
        const cv::Rect bb = yolo[i].bbox & cv::Rect(0, 0, W, H);
        if (bb.width < 2 || bb.height < 2) continue;
        const DepthStat y = depth_stat(yolo[i].mask, depth, W, H, bb);
        const DepthStat s = depth_stat(sam2[i].mask, depth, W, H, bb);
        if (y.n < 8 || s.n < 8) continue;
        const double area_ratio = y.n > 0 ? static_cast<double>(s.n) / y.n : 0.0;
        const double iou = mask_iou(yolo[i].mask, sam2[i].mask, bb);
        metrics_csv_ << t_ms << ',' << in.stamp << ',' << yolo[i].label << ','
                     << y.n << ',' << s.n << ',' << area_ratio << ','
                     << 100.0 * y.mad << ',' << 100.0 * s.mad << ',' << 100.0 * (y.mad - s.mad) << ','
                     << y.far_tail << ',' << s.far_tail << ',' << (y.far_tail - s.far_tail) << ','
                     << y.bleed << ',' << s.bleed << ',' << (y.bleed - s.bleed) << ',' << iou << '\n';
        sum_yolo_mad_ += y.mad;   sum_sam2_mad_ += s.mad;
        sum_yolo_far_ += y.far_tail; sum_sam2_far_ += s.far_tail;
        ++n_rows_;
    }
    metrics_csv_.flush();

    // Rolling stdout summary so the improvement is visible without opening the CSV. MAD (robust depth spread)
    // is the headline — scale-honest, unlike the fixed-band bleed%; far_tail is the background-mode share.
    if (n_rows_ > 0 && n_rows_ % 60 == 0)
        std::println("[Sam2Stage] refine over {} objs: depth-MAD YOLO {:.1f}cm → SAM2 {:.1f}cm (−{:.1f}cm) | "
                     "far-tail {:.1f}% → {:.1f}% (−{:.1f}pts)", n_rows_,
                     100.0 * sum_yolo_mad_ / n_rows_, 100.0 * sum_sam2_mad_ / n_rows_,
                     100.0 * (sum_yolo_mad_ - sum_sam2_mad_) / n_rows_,
                     100.0 * sum_yolo_far_ / n_rows_, 100.0 * sum_sam2_far_ / n_rows_,
                     100.0 * (sum_yolo_far_ - sum_sam2_far_) / n_rows_);
}

} // namespace rc
