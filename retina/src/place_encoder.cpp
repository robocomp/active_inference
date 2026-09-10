#include "place_encoder.h"

#include "onnx_providers.h"
#include "../../common/place_memory/place_map.h"

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <print>
#include <stdexcept>

namespace rc::place
{
namespace
{
// ImageNet normalisation — identical constants to sam2_processor.cpp, and to the torchvision
// transform in tools/export_dinov2.py. All three must agree or the descriptors are meaningless.
constexpr std::array<float, 3> kMean{0.485f, 0.456f, 0.406f};
constexpr std::array<float, 3> kStd {0.229f, 0.224f, 0.225f};

// The TensorRT engine build is a 30-60 s stall on the worker thread with no output of its own, which
// reads exactly like a hang. Say so, loudly, before it happens.
constexpr const char* kRed   = "\033[1;31m";
constexpr const char* kReset = "\033[0m";

inline void l2_normalise(float* v, int n)
{
    float s = 0.f;
    for (int i = 0; i < n; ++i) s += v[i] * v[i];
    s = std::sqrt(s);
    if (s > 1e-12f) for (int i = 0; i < n; ++i) v[i] /= s;
}
}   // namespace

struct PlaceEncoder::Impl
{
    Ort::SessionOptions opts;
    std::vector<char*>       in_names, out_names;
    std::vector<const char*> in_cstr,  out_cstr;
    ~Impl()
    {
        for (char* n : in_names)  free(n);
        for (char* n : out_names) free(n);
    }
};

PlaceEncoder::PlaceEncoder(const EncoderConfig& cfg) : cfg_(cfg), impl_(std::make_unique<Impl>())
{
    try
    {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "PlaceEncoder");
        impl_->opts.SetIntraOpNumThreads(1);
        impl_->opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (cfg_.use_gpu and cfg_.use_trt)
            std::println("{}[PlaceEncoder] TensorRT enabled — if .trt_cache holds no engine for this "
                         "model yet, the FIRST run builds one and this thread will stall for 30-60 s "
                         "with no further output. That is the engine build, not a hang.{}", kRed, kReset);

        rc::onnx::append_gpu_providers(impl_->opts, cfg_.use_gpu, cfg_.use_trt, "[PlaceEncoder]");
        session_ = std::make_unique<Ort::Session>(*env_, cfg_.model_path.c_str(), impl_->opts);

        Ort::AllocatorWithDefaultOptions alloc;
        for (std::size_t i = 0, n = session_->GetInputCount(); i < n; ++i)
        {
            auto nm = session_->GetInputNameAllocated(i, alloc);
            impl_->in_names.push_back(strdup(nm.get()));
            impl_->in_cstr.push_back(impl_->in_names.back());
        }
        for (std::size_t i = 0, n = session_->GetOutputCount(); i < n; ++i)
        {
            auto nm = session_->GetOutputNameAllocated(i, alloc);
            impl_->out_names.push_back(strdup(nm.get()));
            impl_->out_cstr.push_back(impl_->out_names.back());
        }

        // ★ ASSERT THE INPUT SHAPE, do not merely document it. [RicohDepth].input_size carries a
        // comment saying "must match the exported imgsz" and nothing enforces it; a silent mismatch
        // there is a wrong answer, not an error.
        const auto in_shape = session_->GetInputTypeInfo(0)
                                      .GetTensorTypeAndShapeInfo().GetShape();
        if (in_shape.size() != 4 or in_shape[2] != cfg_.input_h or in_shape[3] != cfg_.input_w)
        {
            why_not_ = std::format("model input is [{}] but config says {}x{} (h x w)",
                                   [&]{ std::string s; for (auto d : in_shape) s += std::to_string(d) + ","; return s; }(),
                                   cfg_.input_h, cfg_.input_w);
            std::println("{}[PlaceEncoder] REFUSING TO RUN: {}{}", kRed, why_not_, kReset);
            return;
        }

        // patch output is [1, gh, gw, dim] — the reshape lives in the ONNX graph on purpose, so a
        // transposed grid is impossible rather than merely unlikely.
        const auto p_shape = session_->GetOutputTypeInfo(1)
                                     .GetTensorTypeAndShapeInfo().GetShape();
        if (p_shape.size() != 4)
        {
            why_not_ = "second output is not a 4-D patch grid [1, gh, gw, dim]";
            std::println("{}[PlaceEncoder] REFUSING TO RUN: {}{}", kRed, why_not_, kReset);
            return;
        }
        gh_ = int(p_shape[1]); gw_ = int(p_shape[2]); dim_ = int(p_shape[3]);

        if (cfg_.n_sectors <= 0 or gw_ % cfg_.n_sectors != 0)
        {
            why_not_ = std::format("n_sectors={} does not divide the {} patch columns",
                                   cfg_.n_sectors, gw_);
            std::println("{}[PlaceEncoder] REFUSING TO RUN: {}{}", kRed, why_not_, kReset);
            return;
        }
        cfg_.band_lo = std::clamp(cfg_.band_lo, 0, gh_);
        cfg_.band_hi = std::clamp(cfg_.band_hi, cfg_.band_lo + 1, gh_);

        ready_ = true;
        std::println("[PlaceEncoder] {}  in {}x{}  grid {}x{}x{}  sectors={} ({:.2f} deg)  "
                     "band=[{},{})  centre={}  pool_p={:.1f}",
                     cfg_.model_path, cfg_.input_w, cfg_.input_h, gh_, gw_, dim_,
                     cfg_.n_sectors, 360.0f / float(cfg_.n_sectors),
                     cfg_.band_lo, cfg_.band_hi, cfg_.center ? "on" : "off", cfg_.pool_p);
    }
    catch (const Ort::Exception& e)
    {
        why_not_ = std::string("ONNX error: ") + e.what();
        std::println("{}[PlaceEncoder] disabled — {}{}", kRed, why_not_, kReset);
    }
}

PlaceEncoder::~PlaceEncoder() = default;

std::vector<float> PlaceEncoder::pool(const std::vector<float>& grid, int gh, int gw, int dim,
                                      const EncoderConfig& cfg)
{
    std::vector<float> out;
    const int lo = std::clamp(cfg.band_lo, 0, gh);
    const int hi = std::clamp(cfg.band_hi, lo + 1, gh);
    if (cfg.n_sectors <= 0 or gw % cfg.n_sectors != 0) return out;
    if (grid.size() != std::size_t(gh) * gw * dim) return out;

    const int rows = hi - lo;
    std::vector<float> band(std::size_t(rows) * gw * dim);
    std::copy_n(grid.begin() + std::size_t(lo) * gw * dim, band.size(), band.begin());

    // per-token L2 first: it is what stops a few high-norm tokens from owning the pooled direction
    for (std::size_t t = 0; t < std::size_t(rows) * gw; ++t)
        l2_normalise(band.data() + t * dim, dim);

    if (cfg.center)
    {
        // ★ per-FRAME mean, computed inside the band. Rolling the panorama permutes columns and
        // leaves this unchanged, so centering is exactly rotation-invariant.
        std::vector<float> mu(std::size_t(dim), 0.f);
        const std::size_t n_tok = std::size_t(rows) * gw;
        for (std::size_t t = 0; t < n_tok; ++t)
            for (int c = 0; c < dim; ++c) mu[std::size_t(c)] += band[t * dim + c];
        for (int c = 0; c < dim; ++c) mu[std::size_t(c)] /= float(n_tok);
        for (std::size_t t = 0; t < n_tok; ++t)
            for (int c = 0; c < dim; ++c) band[t * dim + c] -= mu[std::size_t(c)];
    }

    const int cols_per = gw / cfg.n_sectors;
    out.assign(std::size_t(cfg.n_sectors) * dim, 0.f);
    const bool soft = cfg.sector_soft > 0.f;
    const float halfw = soft ? cfg.sector_soft * float(cols_per) : 0.f;

    std::vector<float> w(std::size_t(gw), 0.f);      // 2-arg form: `w(std::size_t(gw))` is a vexing parse
    for (int s = 0; s < cfg.n_sectors; ++s)
    {
        // Column weights for this sector: a hard bin, or a raised cosine centred on the sector and
        // wrapped around the panorama seam (azimuth is circular — a window that stopped at column 0
        // would make sector 0 and sector S-1 behave unlike every other pair).
        const float centre = (float(s) + 0.5f) * float(cols_per);
        float wsum = 0.f;
        for (int j = 0; j < gw; ++j)
        {
            float d = std::fabs(float(j) + 0.5f - centre);
            d = std::min(d, float(gw) - d);
            w[std::size_t(j)] = soft ? (d < halfw ? 0.5f * (1.f + std::cos(float(M_PI) * d / halfw)) : 0.f)
                                     : ((j >= s * cols_per and j < (s + 1) * cols_per) ? 1.f : 0.f);
            wsum += w[std::size_t(j)];
        }
        if (wsum <= 0.f) continue;

        float* v = out.data() + std::size_t(s) * dim;
        std::vector<float> col(std::size_t(dim), 0.f);
        for (int j = 0; j < gw; ++j)
        {
            const float wj = w[std::size_t(j)] / wsum;
            if (wj <= 0.f) continue;
            // pool this column over the elevation rows first, then blend columns by weight
            std::fill(col.begin(), col.end(), 0.f);
            for (int r = 0; r < rows; ++r)
            {
                const float* t = band.data() + (std::size_t(r) * gw + j) * dim;
                if (std::fabs(cfg.pool_p - 1.0f) < 1e-9f)
                    for (int c = 0; c < dim; ++c) col[std::size_t(c)] += t[c];
                else
                    for (int c = 0; c < dim; ++c)
                        col[std::size_t(c)] += std::copysign(std::pow(std::fabs(t[c]), cfg.pool_p), t[c]);
            }
            for (int c = 0; c < dim; ++c) v[c] += wj * col[std::size_t(c)] / float(rows);
        }
        if (std::fabs(cfg.pool_p - 1.0f) >= 1e-9f)
            for (int c = 0; c < dim; ++c)
                v[c] = std::copysign(std::pow(std::fabs(v[c]), 1.0f / cfg.pool_p), v[c]);
        l2_normalise(v, dim);
    }
    return out;
}

std::vector<float> PlaceEncoder::encode(const cv::Mat& bgr_panorama, std::vector<float>* raw_grid_out)
{
    std::vector<float> out;
    if (not ready_ or bgr_panorama.empty() or bgr_panorama.type() != CV_8UC3) return out;

    // ── preprocessing: must match tools/export_dinov2.py::preprocess step for step ───────────────
    cv::Mat rgb;
    cv::cvtColor(bgr_panorama, rgb, cv::COLOR_BGR2RGB);
    cv::Mat r;
    // ★ INTER_AREA, not INTER_LINEAR: 1920 -> 448 is a 4.3x downscale and only AREA matches
    // torchvision's antialias=True. LINEAR aliases and looks like a bad model.
    cv::resize(rgb, r, {cfg_.input_w, cfg_.input_h}, 0, 0, cv::INTER_AREA);
    cv::Mat rf;
    r.convertTo(rf, CV_32FC3, 1.0 / 255.0);
    std::vector<cv::Mat> ch(3);
    cv::split(rf, ch);
    const int plane = cfg_.input_w * cfg_.input_h;
    std::vector<float> tensor(std::size_t(3) * plane);
    for (int c = 0; c < 3; ++c)
    {
        cv::Mat norm = (ch[std::size_t(c)] - kMean[std::size_t(c)]) / kStd[std::size_t(c)];
        std::memcpy(tensor.data() + std::size_t(c) * plane, norm.data, std::size_t(plane) * sizeof(float));
    }

    try
    {
        const std::array<int64_t, 4> shape{1, 3, cfg_.input_h, cfg_.input_w};
        const auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value in = Ort::Value::CreateTensor<float>(mem, tensor.data(), tensor.size(),
                                                        shape.data(), shape.size());
        auto outs = session_->Run(Ort::RunOptions{nullptr}, impl_->in_cstr.data(), &in, 1,
                                  impl_->out_cstr.data(), impl_->out_cstr.size());
        if (outs.size() < 2) return out;

        const float* cls = outs[0].GetTensorData<float>();
        const float* pg  = outs[1].GetTensorData<float>();
        const std::size_t grid_n = std::size_t(gh_) * gw_ * dim_;

        std::vector<float> grid(pg, pg + grid_n);
        if (raw_grid_out) *raw_grid_out = grid;

        const auto sectors = pool(grid, gh_, gw_, dim_, cfg_);
        if (sectors.empty()) return out;

        out.resize(descriptor_size());
        std::copy_n(cls, dim_, out.begin());
        l2_normalise(out.data(), dim_);                       // CLS: prefilter only
        std::copy(sectors.begin(), sectors.end(), out.begin() + dim_);
    }
    catch (const Ort::Exception& e)
    {
        std::println("{}[PlaceEncoder] inference failed: {}{}", kRed, e.what(), kReset);
        out.clear();
    }
    return out;
}

bool PlaceEncoder::self_test(const cv::Mat& bgr_panorama, std::string* detail)
{
    if (not ready_) { if (detail) *detail = why_not_; return false; }
    const int S = cfg_.n_sectors;
    const auto base = encode(bgr_panorama);
    if (base.empty()) { if (detail) *detail = "encode() returned nothing"; return false; }

    // Roll by exactly one sector's worth of columns. cv::hconcat of the two halves is a cyclic shift
    // toward HIGHER column index, matching np.roll(+px) in the export script.
    const int px = bgr_panorama.cols / S;
    cv::Mat rolled;
    cv::hconcat(bgr_panorama.colRange(bgr_panorama.cols - px, bgr_panorama.cols),
                bgr_panorama.colRange(0, bgr_panorama.cols - px), rolled);
    const auto q = encode(rolled);
    if (q.empty()) { if (detail) *detail = "encode() of the rolled copy returned nothing"; return false; }

    std::vector<float> sim;
    circular_similarity(q.data() + dim_, base.data() + dim_, S, dim_, sim);
    const Peak pk = circular_peak(sim);
    const bool ok = (pk.shift == 1);
    if (detail)
        *detail = std::format("recovered shift {} (expected 1), peak={:.4f}, margin={:+.4f}",
                              pk.shift, pk.sim, pk.margin);
    return ok;
}

}   // namespace rc::place
