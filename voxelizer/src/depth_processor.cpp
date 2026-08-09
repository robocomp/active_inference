#include "depth_processor.h"
#include "onnx_providers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <print>
#include <stdexcept>

namespace rc::depth
{

namespace
{
// Circular column padding — the panorama wraps, so the first and last strips get their context from
// the far side of the seam. Same helper as the 360 seg path (yolo_processor.cpp), duplicated rather
// than exported because that one lives in an anonymous namespace.
cv::Mat circular_pad_columns(const cv::Mat& img, int pad)
{
    if (pad <= 0 or img.cols <= 0)
        return img;
    const int p = std::min(pad, img.cols);
    const cv::Mat left  = img(cv::Rect(img.cols - p, 0, p, img.rows));
    const cv::Mat right = img(cv::Rect(0, 0, p, img.rows));
    cv::Mat padded;
    cv::hconcat(std::vector<cv::Mat>{left, img, right}, padded);
    return padded;
}

// Robust range of the finite values in `block`, as [lo, hi] percentiles. Returns false if there is
// nothing finite to scale by (a strip the model failed on, or an all-NaN block).
bool robust_range(const cv::Mat& block, float lo_pct, float hi_pct, float& lo, float& hi)
{
    std::vector<float> v;
    v.reserve(static_cast<std::size_t>(block.total()));
    for (int y = 0; y < block.rows; ++y)
    {
        const float* row = block.ptr<float>(y);
        for (int x = 0; x < block.cols; ++x)
            if (std::isfinite(row[x]))
                v.push_back(row[x]);
    }
    if (v.size() < 16)
        return false;
    const auto pick = [&v](float pct)
    {
        const std::size_t k = std::min(v.size() - 1,
                                       static_cast<std::size_t>(pct * 0.01f * static_cast<float>(v.size() - 1)));
        std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k), v.end());
        return v[k];
    };
    lo = pick(lo_pct);
    hi = pick(hi_pct);
    return hi - lo > 1e-4f;
}
}   // namespace

// ─── DepthEstimator ──────────────────────────────────────────────────────────────────────────────

DepthEstimator::DepthEstimator(const std::string& model_path, int input_size, bool use_gpu, bool use_trt)
    : input_size_(input_size)
{
    try
    {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "DepthEstimator");
        session_opts_.SetIntraOpNumThreads(1);
        session_opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        rc::onnx::append_gpu_providers(session_opts_, use_gpu, use_trt, "[DepthEstimator]");

        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_opts_);

        Ort::AllocatorWithDefaultOptions allocator;
        for (std::size_t i = 0, n = session_->GetInputCount(); i < n; ++i)
        {
            auto name = session_->GetInputNameAllocated(i, allocator);
            input_names_.push_back(strdup(name.get()));
            input_names_cstr_.push_back(input_names_.back());
        }
        for (std::size_t i = 0, n = session_->GetOutputCount(); i < n; ++i)
        {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(strdup(name.get()));
            output_names_cstr_.push_back(output_names_.back());
        }
        std::println("[DepthEstimator] Loaded: {}  inputs={}  outputs={}  imgsz={}",
                     model_path, input_names_.size(), output_names_.size(), input_size_);
    }
    catch (const Ort::Exception& e)
    {
        throw std::runtime_error(std::string("[DepthEstimator] ONNX error: ") + e.what());
    }
}

DepthEstimator::~DepthEstimator()
{
    for (char* n : input_names_)  free(n);
    for (char* n : output_names_) free(n);
}

cv::Mat DepthEstimator::infer_log_depth(const cv::Mat& bgr) const
{
    if (bgr.empty() or not session_)
        return {};

    // Letterbox (grey 114) exactly like the seg/semantic preprocessors, so a strip whose aspect is not
    // 1:1 keeps its geometry instead of being stretched — a stretch would distort the very structure
    // the depth head is reading.
    const int S = input_size_;
    const float scale = std::min(static_cast<float>(S) / static_cast<float>(bgr.cols),
                                 static_cast<float>(S) / static_cast<float>(bgr.rows));
    const int nw = std::max(1, static_cast<int>(std::lround(bgr.cols * scale)));
    const int nh = std::max(1, static_cast<int>(std::lround(bgr.rows * scale)));
    const int pad_l = (S - nw) / 2;
    const int pad_t = (S - nh) / 2;

    cv::Mat resized;
    cv::resize(bgr, resized, {nw, nh}, 0, 0, cv::INTER_LINEAR);
    cv::Mat lb(S, S, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(lb(cv::Rect(pad_l, pad_t, nw, nh)));

    cv::Mat rgb;
    cv::cvtColor(lb, rgb, cv::COLOR_BGR2RGB);
    cv::Mat rgb_f;
    rgb.convertTo(rgb_f, CV_32FC3, 1.0 / 255.0);
    std::vector<cv::Mat> channels(3);
    cv::split(rgb_f, channels);

    const int stride = S * S;
    std::vector<float> tensor(static_cast<std::size_t>(3) * stride);
    for (int c = 0; c < 3; ++c)
        std::memcpy(tensor.data() + static_cast<std::size_t>(c) * stride, channels[c].data,
                    static_cast<std::size_t>(stride) * sizeof(float));

    const std::array<std::int64_t, 4> in_shape{1, 3, S, S};
    const auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(mem_info, tensor.data(), tensor.size(),
                                                       in_shape.data(), in_shape.size());

    std::vector<Ort::Value> outputs;
    try
    {
        outputs = session_->Run(Ort::RunOptions{nullptr},
                                input_names_cstr_.data(), &input, 1,
                                output_names_cstr_.data(), output_names_cstr_.size());
    }
    catch (const Ort::Exception& e)
    {
        std::println(stderr, "[DepthEstimator] inference error: {}", e.what());
        return {};
    }
    if (outputs.empty())
        return {};

    // Expect [1,1,S,S]; accept [1,S,S] / [S,S] too so a re-export with a squeezed head still works.
    const auto info = outputs[0].GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();
    int oh = 0, ow = 0;
    if (shape.size() == 4)      { oh = static_cast<int>(shape[2]); ow = static_cast<int>(shape[3]); }
    else if (shape.size() == 3) { oh = static_cast<int>(shape[1]); ow = static_cast<int>(shape[2]); }
    else if (shape.size() == 2) { oh = static_cast<int>(shape[0]); ow = static_cast<int>(shape[1]); }
    else
    {
        std::println(stderr, "[DepthEstimator] unexpected output rank {}", shape.size());
        return {};
    }
    if (oh <= 0 or ow <= 0)
        return {};

    // Wrap the ORT buffer, then CLONE: the tensor dies with `outputs` at the end of this function, and
    // the result crosses to the caller (and later the main thread) as an owned buffer.
    const cv::Mat out_lb(oh, ow, CV_32FC1, const_cast<float*>(outputs[0].GetTensorData<float>()));

    // Unletterbox in OUTPUT coordinates: the head is a downscaled copy of the S^2 letterbox, so the
    // active (non-padded) region scales by (ow/S, oh/S).
    const float fx = static_cast<float>(ow) / static_cast<float>(S);
    const float fy = static_cast<float>(oh) / static_cast<float>(S);
    cv::Rect active(static_cast<int>(std::lround(pad_l * fx)), static_cast<int>(std::lround(pad_t * fy)),
                    static_cast<int>(std::lround(nw * fx)),    static_cast<int>(std::lround(nh * fy)));
    active &= cv::Rect(0, 0, ow, oh);
    if (active.width <= 0 or active.height <= 0)
        return {};

    cv::Mat full;
    cv::resize(out_lb(active), full, bgr.size(), 0, 0, cv::INTER_LINEAR);
    return full;   // resize wrote into a fresh buffer ⇒ already owned, no aliasing of the ORT tensor
}

// ─── DepthProcessor ──────────────────────────────────────────────────────────────────────────────

void DepthProcessor::configure(const Config& config)
{
    config_ = config;
    estimator_.reset();
    try
    {
        estimator_ = std::make_unique<DepthEstimator>(config_.model_path, config_.input_size,
                                                      config_.use_gpu, config_.use_trt);
    }
    catch (const std::exception& e)
    {
        std::println(stderr, "[DepthProcessor] disabled — {}", e.what());
        estimator_.reset();
    }
}

DepthMap DepthProcessor::estimate_360(const cv::Mat& panorama_bgr, const Depth360Config& cfg)
{
    return cfg.gnomonic ? estimate_gnomonic(panorama_bgr, cfg)
                        : estimate_equirect(panorama_bgr, cfg);
}

// ─── Gnomonic (rectilinear) path ─────────────────────────────────────────────────────────────────

bool DepthProcessor::ensure_maps(int pano_w, int pano_h, const Depth360Config& cfg)
{
    const int   S   = estimator_ ? estimator_->input_size() : 0;
    const int   n   = cfg.n_strips;
    const float fov = (cfg.gnomonic_fov_deg > 0.f)
                    ? std::clamp(cfg.gnomonic_fov_deg, 10.f, 160.f)
                    : std::min(140.f, 360.f / static_cast<float>(std::max(1, n)) * 1.25f);
    if (S <= 0 or n <= 0 or pano_w <= 0 or pano_h <= 0)
        return false;

    if (maps_.size() == static_cast<std::size_t>(n) and maps_w_ == pano_w and maps_h_ == pano_h
        and maps_n_ == n and maps_s_ == S and std::abs(maps_fov_ - fov) < 1e-3f
        and maps_zcorr_ == cfg.zdepth_to_range)
        return true;   // geometry unchanged — the cached tables are still exact

    const int strip_w = pano_w / n;
    if (strip_w <= 0)
        return false;

    constexpr double kPi = 3.14159265358979323846;
    const double f = (S * 0.5) / std::tan(0.5 * fov * kPi / 180.0);

    maps_.assign(static_cast<std::size_t>(n), StripMaps{});
    for (int s = 0; s < n; ++s)
    {
        // The virtual camera looks along the CENTRE of this strip's azimuth span, level with the
        // horizon. Basis: F forward, R right (= increasing azimuth), U up (= world +z).
        const double az_s = ((s + 0.5) * strip_w) / static_cast<double>(pano_w) * 2.0 * kPi;
        const double Fx = std::cos(az_s), Fy = std::sin(az_s);
        const double Rx = -std::sin(az_s), Ry = std::cos(az_s);

        StripMaps m;
        m.fwd_x      = cv::Mat(S, S, CV_32FC1);
        m.fwd_y      = cv::Mat(S, S, CV_32FC1);
        m.range_corr = cv::Mat(S, S, CV_32FC1, cv::Scalar(0.f));

        for (int v = 0; v < S; ++v)
        {
            auto* mx = m.fwd_x.ptr<float>(v);
            auto* my = m.fwd_y.ptr<float>(v);
            auto* rc = m.range_corr.ptr<float>(v);
            const double yn = (v + 0.5 - S * 0.5) / f;
            for (int u = 0; u < S; ++u)
            {
                const double xn = (u + 0.5 - S * 0.5) / f;
                // Ray through this pixel, in world axes: F + xn*R - yn*U (v grows downward).
                const double dx = Fx + xn * Rx;
                const double dy = Fy + xn * Ry;
                const double dz = -yn;
                const double len = std::sqrt(dx * dx + dy * dy + dz * dz);

                double az = std::atan2(dy, dx);
                if (az < 0.0) az += 2.0 * kPi;
                const double el = std::asin(std::clamp(dz / len, -1.0, 1.0));

                mx[u] = static_cast<float>(az / (2.0 * kPi) * pano_w - 0.5);
                my[u] = static_cast<float>((0.5 * kPi - el) / kPi * pano_h - 0.5);
                // Z-depth → range along the ray. In log space this is a plain additive term.
                if (cfg.zdepth_to_range)
                    rc[u] = static_cast<float>(0.5 * std::log(1.0 + xn * xn + yn * yn));
            }
        }

        // Inverse: for every equirect pixel this strip OWNS (its own column span, full height), where
        // does it land in the gnomonic view? Full height because the vertical coverage falls out of the
        // FOV rather than being chosen — the valid mask, not a band constant, defines it.
        m.inv_x     = cv::Mat(pano_h, strip_w, CV_32FC1, cv::Scalar(-1.f));
        m.inv_y     = cv::Mat(pano_h, strip_w, CV_32FC1, cv::Scalar(-1.f));
        m.inv_valid = cv::Mat(pano_h, strip_w, CV_8UC1,  cv::Scalar(0));

        for (int y = 0; y < pano_h; ++y)
        {
            auto* ix = m.inv_x.ptr<float>(y);
            auto* iy = m.inv_y.ptr<float>(y);
            auto* iv = m.inv_valid.ptr<unsigned char>(y);
            const double el = 0.5 * kPi - (y + 0.5) / static_cast<double>(pano_h) * kPi;
            const double cel = std::cos(el), sel = std::sin(el);
            for (int xl = 0; xl < strip_w; ++xl)
            {
                const int gx = s * strip_w + xl;
                const double az = (gx + 0.5) / static_cast<double>(pano_w) * 2.0 * kPi;
                const double dx = cel * std::cos(az), dy = cel * std::sin(az), dz = sel;

                const double zc = dx * Fx + dy * Fy;            // dir . F
                if (zc <= 1e-6)
                    continue;                                    // behind the virtual camera
                const double xc = dx * Rx + dy * Ry;            // dir . R
                const double u = f * (xc / zc) + S * 0.5 - 0.5;
                const double v = f * (-dz / zc) + S * 0.5 - 0.5;
                if (u < 0.0 or v < 0.0 or u > S - 1.0 or v > S - 1.0)
                    continue;                                    // outside this view's frustum
                ix[xl] = static_cast<float>(u);
                iy[xl] = static_cast<float>(v);
                iv[xl] = 255;
            }
        }
        maps_[static_cast<std::size_t>(s)] = std::move(m);
    }

    maps_w_ = pano_w; maps_h_ = pano_h; maps_n_ = n; maps_s_ = S;
    maps_fov_ = fov;  maps_zcorr_ = cfg.zdepth_to_range;
    std::println("[depth360] gnomonic maps built: {} views, fov {:.1f}°, {}x{} each, pano {}x{}"
                 " (z→range {})", n, fov, S, S, pano_w, pano_h, cfg.zdepth_to_range ? "on" : "off");
    return true;
}

DepthMap DepthProcessor::estimate_gnomonic(const cv::Mat& panorama_bgr, const Depth360Config& cfg)
{
    DepthMap map;
    if (not estimator_ or panorama_bgr.empty() or cfg.n_strips <= 0)
        return map;

    const int W = panorama_bgr.cols, H = panorama_bgr.rows;
    if (not ensure_maps(W, H, cfg))
        return map;

    const int strip_w = W / cfg.n_strips;
    map.log_depth = cv::Mat(H, W, CV_32FC1, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    map.strip_id  = cv::Mat(H, W, CV_8UC1,  cv::Scalar(kNoStrip));
    map.n_strips  = cfg.n_strips;

    int y_lo = H, y_hi = 0;
    for (int s = 0; s < cfg.n_strips; ++s)
    {
        const StripMaps& m = maps_[static_cast<std::size_t>(s)];

        // Panorama → virtual pinhole view. BORDER_WRAP closes the azimuth seam for free: a view whose
        // frustum crosses column 0 samples straight through to the far side, no special case.
        cv::Mat persp;
        cv::remap(panorama_bgr, persp, m.fwd_x, m.fwd_y, cv::INTER_LINEAR, cv::BORDER_WRAP);

        // Square and already the model's input size ⇒ infer_log_depth's letterbox is a no-op (scale 1,
        // zero padding). That is the point of sizing the view to input_size.
        cv::Mat ld = estimator_->infer_log_depth(persp);
        if (ld.empty() or ld.size() != persp.size())
            continue;
        if (cfg.zdepth_to_range)
            ld += m.range_corr;   // log_range = log_z + 0.5*log(1 + xn^2 + yn^2)

        // Gnomonic view → this strip's equirect columns. NEAREST would alias the depth field; the
        // maps are smooth so INTER_LINEAR is safe, and BORDER_CONSTANT keeps out-of-frustum samples
        // out rather than smearing the edge inward.
        cv::Mat back;
        cv::remap(ld, back, m.inv_x, m.inv_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                  cv::Scalar(std::numeric_limits<float>::quiet_NaN()));

        back.copyTo(map.log_depth(cv::Rect(s * strip_w, 0, strip_w, H)), m.inv_valid);
        map.strip_id(cv::Rect(s * strip_w, 0, strip_w, H)).setTo(static_cast<unsigned char>(s), m.inv_valid);

        // The covered band is a RESULT of the FOV, not a config constant — report what was actually
        // written so compose() and every reader iterate exactly the rows that exist.
        for (int y = 0; y < H; ++y)
            if (cv::countNonZero(m.inv_valid.row(y)) > 0)
            {
                y_lo = std::min(y_lo, y);
                y_hi = std::max(y_hi, y + 1);
            }
    }
    map.band_y0 = std::min(y_lo, y_hi);
    map.band_y1 = y_hi;

    if (config_.verbose_debug)
    {
        float lo = 0.f, hi = 0.f;
        if (robust_range(map.log_depth, 2.f, 98.f, lo, hi))
            std::println("[depth360] gnomonic {} views — rows [{},{}) — log {:.2f}..{:.2f} "
                         "⇒ {:.2f}..{:.2f} m (PER-VIEW scale)", cfg.n_strips, map.band_y0, map.band_y1,
                         lo, hi, std::exp(lo), std::exp(hi));
    }
    return map;
}

DepthMap DepthProcessor::estimate_perspective(const cv::Mat& bgr) const
{
    DepthMap map;
    if (not estimator_ or bgr.empty())
        return map;
    cv::Mat ld = estimator_->infer_log_depth(bgr);
    if (ld.empty() or ld.size() != bgr.size())
        return map;
    map.log_depth = std::move(ld);
    map.strip_id  = cv::Mat(bgr.rows, bgr.cols, CV_8UC1, cv::Scalar(0));   // one "view"
    map.n_strips  = 1;
    map.band_y0   = 0;
    map.band_y1   = bgr.rows;
    return map;
}

// ─── Raw-equirect path (kept for A/B; see Depth360Config::gnomonic for why it is not the default) ──

DepthMap DepthProcessor::estimate_equirect(const cv::Mat& panorama_bgr, const Depth360Config& cfg) const
{
    DepthMap map;
    if (not estimator_ or panorama_bgr.empty() or cfg.n_strips <= 0)
        return map;

    const int W = panorama_bgr.cols, H = panorama_bgr.rows;
    const int strip_w = W / cfg.n_strips;
    if (strip_w <= 0)
        return map;

    // Elevation band. Equirect rows run +90 deg (row 0) to -90 deg (row H-1), so a band of
    // +/-band_half_elev_deg about the equator is H*(band/180) rows either side of H/2. Clamped to the
    // image, and never smaller than a stride so the model still has something to look at.
    const int half_rows = std::clamp(static_cast<int>(std::lround(H * (cfg.band_half_elev_deg / 180.0f))),
                                     32, H / 2);
    const int y0 = std::max(0, H / 2 - half_rows);
    const int y1 = std::min(H, H / 2 + half_rows);
    const int band_h = y1 - y0;
    if (band_h <= 0)
        return map;

    const int overlap  = std::clamp(cfg.overlap_px, 0, strip_w / 2);
    const int strip_px = strip_w + 2 * overlap;

    map.log_depth = cv::Mat(H, W, CV_32FC1, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    map.strip_id  = cv::Mat(H, W, CV_8UC1,  cv::Scalar(kNoStrip));
    map.n_strips  = cfg.n_strips;
    map.band_y0   = y0;
    map.band_y1   = y1;

    const cv::Mat band   = panorama_bgr(cv::Rect(0, y0, W, band_h));
    const cv::Mat padded = circular_pad_columns(band, overlap);

    for (int s = 0; s < cfg.n_strips; ++s)
    {
        const int local_x0 = s * strip_w;
        if (local_x0 + strip_px > padded.cols)
            continue;   // width not exactly divisible by n_strips — drop the sliver, never slice short

        const cv::Mat strip = padded(cv::Rect(local_x0, 0, strip_px, band_h));
        const cv::Mat ld    = estimator_->infer_log_depth(strip);
        if (ld.empty() or ld.size() != strip.size())
            continue;

        // Keep ONLY the central strip_w columns — the strip's own true region. The overlap existed to
        // give the model context, not to be averaged in: two strips disagree on scale by 2-3x (file
        // header), so blending their overlap would invent a continuity neither of them claims. Each
        // global column is therefore written exactly once, by exactly one strip.
        ld(cv::Rect(overlap, 0, strip_w, band_h))
            .copyTo(map.log_depth(cv::Rect(s * strip_w, y0, strip_w, band_h)));
        map.strip_id(cv::Rect(s * strip_w, y0, strip_w, band_h)).setTo(static_cast<unsigned char>(s));
    }

    if (config_.verbose_debug)
    {
        float lo = 0.f, hi = 0.f;
        if (robust_range(map.log_depth, 2.f, 98.f, lo, hi))
            std::println("[depth360] {} strips {}x{} (overlap {}) band rows [{},{}) — log {:.2f}..{:.2f} "
                         "⇒ {:.2f}..{:.2f} m (PER-STRIP scale, not comparable across seams)",
                         cfg.n_strips, strip_px, band_h, overlap, y0, y1, lo, hi,
                         std::exp(lo), std::exp(hi));
    }
    return map;
}

cv::Mat DepthProcessor::compose_depth_canvas(const cv::Mat& base_bgr, const DepthMap& map, float alpha,
                                             bool metric, float lo_m, float hi_m) const
{
    if (base_bgr.empty())
        return {};
    cv::Mat canvas = base_bgr.clone();
    if (map.empty() or map.log_depth.size() != canvas.size())
        return canvas;

    const float a = std::clamp(alpha, 0.f, 1.f);
    const int strip_w = (map.n_strips > 0) ? canvas.cols / map.n_strips : canvas.cols;
    if (strip_w <= 0)
        return canvas;

    // ★Normalise PER STRIP. The model is scale-invariant per input image, so one global ramp across
    // the whole panorama would render the SAME physical wall in three different colours purely because
    // three different crops guessed three different scales. Per-strip normalisation shows the only
    // thing the model actually asserts: relative depth ordering WITHIN a strip.
    for (int s = 0; s < map.n_strips; ++s)
    {
        const int x0 = s * strip_w;
        const int w  = std::min(strip_w, canvas.cols - x0);
        if (w <= 0)
            break;
        const cv::Rect roi(x0, map.band_y0, w, map.band_y1 - map.band_y0);
        const cv::Mat  ld = map.log_depth(roi);

        float lo = 0.f, hi = 0.f;
        if (metric)
        {
            // Fixed metres → one scale for every view. The whole point of the correction.
            lo = std::log(std::max(1e-3f, lo_m));
            hi = std::log(std::max(lo_m + 1e-3f, hi_m));
        }
        else if (not robust_range(ld, 2.f, 98.f, lo, hi))
            continue;

        // NEAR = warm: TURBO runs blue→red with increasing value, and near should be the hot end, so
        // invert. NaNs (no strip wrote here) map to 0 and are masked out below rather than drawn.
        cv::Mat u8(ld.size(), CV_8UC1, cv::Scalar(0));
        cv::Mat valid(ld.size(), CV_8UC1, cv::Scalar(0));
        for (int y = 0; y < ld.rows; ++y)
        {
            const float* src = ld.ptr<float>(y);
            auto* dst = u8.ptr<unsigned char>(y);
            auto* vld = valid.ptr<unsigned char>(y);
            for (int x = 0; x < ld.cols; ++x)
                if (std::isfinite(src[x]))
                {
                    const float t = std::clamp((src[x] - lo) / (hi - lo), 0.f, 1.f);
                    dst[x] = static_cast<unsigned char>(std::lround((1.0f - t) * 255.0f));
                    vld[x] = 255;
                }
        }

        cv::Mat colored;
        cv::applyColorMap(u8, colored, cv::COLORMAP_TURBO);
        cv::Mat blended;
        cv::addWeighted(canvas(roi), 1.0f - a, colored, a, 0.0, blended);
        blended.copyTo(canvas(roi), valid);
    }

    // Seam markers, RELATIVE MODE ONLY: a thin line at every strip boundary so the eye is never
    // tempted to read a depth gradient across a seam as physical. Once the correction is applied the
    // views share one metric scale and the seams carry no meaning, so drawing them would be the lie.
    if (not metric)
        for (int s = 1; s < map.n_strips; ++s)
        {
            const int x = s * strip_w;
            if (x > 0 and x < canvas.cols)
                cv::line(canvas, {x, map.band_y0}, {x, map.band_y1 - 1}, cv::Scalar(255, 255, 255), 1);
        }
    return canvas;
}

bool align_to_measured(const DepthMap& model, const cv::Mat& measured_m,
                       float min_m, float max_m, DepthAlign& out)
{
    out = DepthAlign{};
    if (model.empty() or measured_m.empty() or model.log_depth.size() != measured_m.size())
        return false;

    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    long   n  = 0;
    for (int y = 0; y < measured_m.rows; ++y)
    {
        const float* mrow = model.log_depth.ptr<float>(y);
        const float* zrow = measured_m.ptr<float>(y);
        for (int x = 0; x < measured_m.cols; ++x)
        {
            const float z = zrow[x];
            // The ZED reports 0 / non-finite where it has no stereo match; those are ABSENT
            // measurements, not zero-range ones, and folding them in would drag the fit to nonsense.
            if (not std::isfinite(mrow[x]) or not std::isfinite(z) or z < min_m or z > max_m)
                continue;
            const double xv = mrow[x], yv = std::log(static_cast<double>(z));
            sx += xv; sy += yv; sxx += xv * xv; sxy += xv * yv;
            ++n;
        }
    }
    if (n < 256)
        return false;
    const double N = static_cast<double>(n);
    const double cov = sxy / N - (sx / N) * (sy / N);
    const double vx  = sxx / N - (sx / N) * (sx / N);
    if (vx <= 1e-12)
        return false;
    out.a = static_cast<float>(cov / vx);
    out.b = static_cast<float>(sy / N - (cov / vx) * (sx / N));
    out.n = n;

    std::vector<float> rel, absm;
    rel.reserve(static_cast<std::size_t>(n));
    absm.reserve(static_cast<std::size_t>(n));
    long within = 0;
    for (int y = 0; y < measured_m.rows; ++y)
    {
        const float* mrow = model.log_depth.ptr<float>(y);
        const float* zrow = measured_m.ptr<float>(y);
        for (int x = 0; x < measured_m.cols; ++x)
        {
            const float z = zrow[x];
            if (not std::isfinite(mrow[x]) or not std::isfinite(z) or z < min_m or z > max_m)
                continue;
            const float d = std::exp(out.a * mrow[x] + out.b);
            rel.push_back(std::abs(d - z) / z);
            absm.push_back(std::abs(d - z));
            if (std::max(d / z, z / d) < 1.25f) ++within;
        }
    }
    const auto med = [](std::vector<float>& v)
    { std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end()); return v[v.size() / 2]; };
    out.med_abs_rel = med(rel);
    out.med_abs_m   = med(absm);
    out.delta125    = static_cast<float>(within) / static_cast<float>(n);
    return true;
}

cv::Mat apply_align(const DepthMap& model, const DepthAlign& al)
{
    if (model.empty())
        return {};
    cv::Mat out(model.log_depth.size(), CV_32FC1,
                cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    for (int y = 0; y < out.rows; ++y)
    {
        const float* src = model.log_depth.ptr<float>(y);
        auto* dst = out.ptr<float>(y);
        for (int x = 0; x < out.cols; ++x)
            if (std::isfinite(src[x]))
                dst[x] = std::exp(al.a * src[x] + al.b);
    }
    return out;
}

cv::Mat compose_difference(const cv::Mat& model_m, const cv::Mat& measured_m, float span_m)
{
    if (model_m.empty() or measured_m.empty() or model_m.size() != measured_m.size())
        return {};
    const float span = std::max(1e-3f, span_m);
    cv::Mat out(model_m.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    for (int y = 0; y < out.rows; ++y)
    {
        const float* mm = model_m.ptr<float>(y);
        const float* zz = measured_m.ptr<float>(y);
        auto* o = out.ptr<cv::Vec3b>(y);
        for (int x = 0; x < out.cols; ++x)
        {
            if (not std::isfinite(mm[x]) or not std::isfinite(zz[x]) or zz[x] <= 0.f)
                continue;                                   // stays black: no comparison possible
            const float d = mm[x] - zz[x];                  // + ⇒ model reads FARTHER than truth
            const int mag = static_cast<int>(std::lround(std::min(1.f, std::abs(d) / span) * 255.f));
            if (d > 0.f) o[x] = cv::Vec3b(0, 0, static_cast<unsigned char>(mag));   // red  = too far
            else         o[x] = cv::Vec3b(static_cast<unsigned char>(mag), 0, 0);   // blue = too near
        }
    }
    return out;
}

}   // namespace rc::depth
