#include "sam2_processor.h"
#include "onnx_providers.h"

#include <opencv2/imgproc.hpp>

#include <array>
#include <chrono>
#include <cstring>
#include <print>
#include <stdexcept>

namespace rc::sam2
{

namespace
{
// ImageNet normalisation used by the SAM2 image encoder (matches the samexporter preprocessing).
constexpr std::array<float, 3> kMean{0.485f, 0.456f, 0.406f};
constexpr std::array<float, 3> kStd {0.229f, 0.224f, 0.225f};

// Capture a session's input/output names into strdup'd storage + cstr views (freed in the dtor).
void capture_io(Ort::Session& s, std::vector<char*>& in, std::vector<const char*>& in_cstr,
                std::vector<char*>& out, std::vector<const char*>& out_cstr)
{
    Ort::AllocatorWithDefaultOptions alloc;
    for (std::size_t i = 0; i < s.GetInputCount(); ++i)
    {
        auto n = s.GetInputNameAllocated(i, alloc);
        in.push_back(strdup(n.get()));
        in_cstr.push_back(in.back());
    }
    for (std::size_t i = 0; i < s.GetOutputCount(); ++i)
    {
        auto n = s.GetOutputNameAllocated(i, alloc);
        out.push_back(strdup(n.get()));
        out_cstr.push_back(out.back());
    }
}

// index of the first name containing `needle` (case-sensitive substring), or -1.
int idx_of(const std::vector<const char*>& names, std::string_view needle)
{
    for (int i = 0; i < static_cast<int>(names.size()); ++i)
        if (std::string_view(names[i]).find(needle) != std::string_view::npos)
            return i;
    return -1;
}
}  // namespace

Sam2Refiner::Sam2Refiner(const Config& cfg) : cfg_(cfg)
{
    try
    {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Sam2Refiner");
        for (auto* opts : {&enc_opts_, &dec_opts_})
        {
            opts->SetIntraOpNumThreads(1);
            opts->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        }
        rc::onnx::append_gpu_providers(enc_opts_, cfg_.use_gpu, cfg_.encoder_use_trt, "[Sam2Refiner:enc]");
        rc::onnx::append_gpu_providers(dec_opts_, cfg_.use_gpu, cfg_.decoder_use_trt, "[Sam2Refiner:dec]");

        enc_session_ = std::make_unique<Ort::Session>(*env_, cfg_.encoder_path.c_str(), enc_opts_);
        dec_session_ = std::make_unique<Ort::Session>(*env_, cfg_.decoder_path.c_str(), dec_opts_);

        capture_io(*enc_session_, enc_in_, enc_in_cstr_, enc_out_, enc_out_cstr_);
        capture_io(*dec_session_, dec_in_, dec_in_cstr_, dec_out_, dec_out_cstr_);

        std::println("[Sam2Refiner] loaded encoder='{}' ({} in/{} out)  decoder='{}' ({} in/{} out)",
                     cfg_.encoder_path, enc_in_.size(), enc_out_.size(),
                     cfg_.decoder_path, dec_in_.size(), dec_out_.size());
    }
    catch (const Ort::Exception& e)
    {
        throw std::runtime_error(std::string("[Sam2Refiner] ONNX error: ") + e.what());
    }
}

Sam2Refiner::~Sam2Refiner()
{
    for (auto* v : {&enc_in_, &enc_out_, &dec_in_, &dec_out_})
        for (char* n : *v)
            free(n);
}

Sam2Refiner::Encoded Sam2Refiner::encode(const cv::Mat& rgb) const
{
    Encoded e;
    e.orig_w = rgb.cols;
    e.orig_h = rgb.rows;

    // Plain stretch to 1024² (NOT letterbox — matches the export; point coords map by the same ratios).
    cv::Mat r;
    cv::resize(rgb, r, {ENC, ENC}, 0, 0, cv::INTER_LINEAR);
    cv::Mat rf;
    r.convertTo(rf, CV_32FC3, 1.0 / 255.0);
    std::vector<cv::Mat> ch(3);
    cv::split(rf, ch);
    const int plane = ENC * ENC;
    std::vector<float> tensor(3 * plane);
    for (int c = 0; c < 3; ++c)
    {
        // channels are R,G,B after the BGR2RGB done by the caller
        cv::Mat norm = (ch[c] - kMean[c]) / kStd[c];
        std::memcpy(tensor.data() + c * plane, norm.data, plane * sizeof(float));
    }

    const std::array<int64_t, 4> in_shape{1, 3, ENC, ENC};
    const auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in = Ort::Value::CreateTensor<float>(mem, tensor.data(), tensor.size(),
                                                    in_shape.data(), in_shape.size());
    std::vector<Ort::Value> outs;
    try
    {
        outs = enc_session_->Run(Ort::RunOptions{nullptr}, enc_in_cstr_.data(), &in, 1,
                                 enc_out_cstr_.data(), enc_out_cstr_.size());
    }
    catch (const Ort::Exception& ex)
    {
        std::println("[Sam2Refiner] encoder inference error: {}", ex.what());
        return e;
    }

    // Map the 3 outputs by name (image_embed / high_res_feats_0 / high_res_feats_1).
    const int i_embed = idx_of(enc_out_cstr_, "image_embed");
    const int i_hr0   = idx_of(enc_out_cstr_, "high_res_feats_0");
    const int i_hr1   = idx_of(enc_out_cstr_, "high_res_feats_1");
    if (i_embed < 0 || i_hr0 < 0 || i_hr1 < 0)
    {
        std::println("[Sam2Refiner] encoder outputs not recognised (need image_embed + high_res_feats_0/1)");
        return e;
    }
    const auto grab = [&](int i, std::vector<float>& dst, std::vector<int64_t>& shape)
    {
        const auto info = outs[i].GetTensorTypeAndShapeInfo();
        shape = info.GetShape();
        const std::size_t n = info.GetElementCount();
        const float* p = outs[i].GetTensorData<float>();
        dst.assign(p, p + n);
    };
    grab(i_embed, e.image_embed,      e.embed_shape);
    grab(i_hr0,   e.high_res_feats_0, e.hr0_shape);
    grab(i_hr1,   e.high_res_feats_1, e.hr1_shape);
    e.ok = true;
    return e;
}

cv::Mat Sam2Refiner::decode(const Encoded& enc, const cv::Rect& bbox, const cv::Mat& coarse) const
{
    if (!enc.ok)
        return {};

    const float sx = static_cast<float>(ENC) / static_cast<float>(enc.orig_w);
    const float sy = static_cast<float>(ENC) / static_cast<float>(enc.orig_h);
    const float x0 = bbox.x, y0 = bbox.y, x1 = bbox.x + bbox.width, y1 = bbox.y + bbox.height;

    // Prompt = the box ONLY (top-left label 2, bottom-right label 3), in 1024 encoder space. No centre
    // point: for a table the box centre is the see-through gap between the legs, so a positive point there
    // makes SAM2 segment that hole. The box brackets the whole object; candidate selection (below) does
    // the disambiguation using the YOLO mask.
    std::vector<float> coords{ x0 * sx, y0 * sy,
                               x1 * sx, y1 * sy };
    std::vector<float> labels{ 2.f, 3.f };
    const int64_t n_pts = 2;

    // mask_input prior: write the YOLO silhouette into the decoder's 256² low-res mask slot as logits
    // (+K inside, -K outside), has_mask_input=1. Biases SAM2 toward the located object's shape. Off (zeros,
    // has_mask=0) when disabled or no coarse mask — that path matches the original box-only behaviour.
    std::vector<float> mask_input(256 * 256, 0.f);
    std::vector<float> has_mask{0.f};
    if (cfg_.mask_prior && not coarse.empty() && cv::countNonZero(coarse) > 0)
    {
        cv::Mat m256;
        cv::resize(coarse, m256, {256, 256}, 0, 0, cv::INTER_NEAREST);   // binary → nearest, no grey edges
        const std::uint8_t* mp = m256.data;
        for (int i = 0; i < 256 * 256; ++i)
            mask_input[i] = (mp[i] > 127) ? cfg_.mask_prior_logit : -cfg_.mask_prior_logit;
        has_mask[0] = 1.f;
    }

    // Owned copies of the encoder features so the input tensors stay valid across Run.
    std::vector<float> embed = enc.image_embed, hr0 = enc.high_res_feats_0, hr1 = enc.high_res_feats_1;
    const std::array<int64_t, 3> coords_shape{1, n_pts, 2};
    const std::array<int64_t, 2> labels_shape{1, n_pts};
    const std::array<int64_t, 4> mask_shape{1, 1, 256, 256};
    const std::array<int64_t, 1> hasmask_shape{1};
    const auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Build the feed in the decoder's declared input order, matching each name to its buffer.
    std::vector<Ort::Value> feed;
    feed.reserve(dec_in_cstr_.size());
    for (const char* raw : dec_in_cstr_)
    {
        const std::string_view name(raw);
        auto make = [&](std::vector<float>& buf, const int64_t* shp, std::size_t rank)
        { feed.push_back(Ort::Value::CreateTensor<float>(mem, buf.data(), buf.size(), shp, rank)); };

        if (name.find("image_embed") != std::string_view::npos)
            make(embed, enc.embed_shape.data(), enc.embed_shape.size());
        else if (name.find("high_res_feats_0") != std::string_view::npos)
            make(hr0, enc.hr0_shape.data(), enc.hr0_shape.size());
        else if (name.find("high_res_feats_1") != std::string_view::npos)
            make(hr1, enc.hr1_shape.data(), enc.hr1_shape.size());
        else if (name.find("point_coords") != std::string_view::npos)
            make(coords, coords_shape.data(), coords_shape.size());
        else if (name.find("point_labels") != std::string_view::npos)
            make(labels, labels_shape.data(), labels_shape.size());
        else if (name.find("has_mask") != std::string_view::npos)   // check BEFORE mask_input (substring)
            make(has_mask, hasmask_shape.data(), hasmask_shape.size());
        else if (name.find("mask_input") != std::string_view::npos)
            make(mask_input, mask_shape.data(), mask_shape.size());
        else
        {
            std::println("[Sam2Refiner] unhandled decoder input '{}' — aborting decode", raw);
            return {};
        }
    }

    std::vector<Ort::Value> outs;
    try
    {
        outs = dec_session_->Run(Ort::RunOptions{nullptr}, dec_in_cstr_.data(), feed.data(),
                                 feed.size(), dec_out_cstr_.data(), dec_out_cstr_.size());
    }
    catch (const Ort::Exception& ex)
    {
        std::println("[Sam2Refiner] decoder inference error: {}", ex.what());
        return {};
    }

    const int i_masks = idx_of(dec_out_cstr_, "masks");
    const int i_iou   = idx_of(dec_out_cstr_, "iou");
    if (i_masks < 0)
        return {};

    const auto minfo = outs[i_masks].GetTensorTypeAndShapeInfo();
    const auto mshape = minfo.GetShape();               // [1, M, mh, mw]
    if (mshape.size() != 4)
        return {};
    const int M  = static_cast<int>(mshape[1]);
    const int mh = static_cast<int>(mshape[2]);
    const int mw = static_cast<int>(mshape[3]);
    const float* mdata = outs[i_masks].GetTensorData<float>();

    const float* iou = (i_iou >= 0) ? outs[i_iou].GetTensorData<float>() : nullptr;
    const int    niou = (i_iou >= 0) ? static_cast<int>(outs[i_iou].GetTensorTypeAndShapeInfo().GetElementCount()) : 0;
    const bool   have_coarse = not coarse.empty() && cv::countNonZero(coarse) > 0;

    // Each of the M candidates → a full-frame binary mask; score it. With the YOLO silhouette available we
    // pick the candidate that best OVERLAPS it (IoU of masks) — this rejects the "hole between the legs"
    // candidate that SAM's own predicted-IoU can rank highest. Without a coarse mask, fall back to that.
    cv::Mat best_bin;
    double  best_score = -1.0;
    for (int m = 0; m < M; ++m)
    {
        cv::Mat logits(mh, mw, CV_32FC1,
                       const_cast<float*>(mdata + static_cast<std::size_t>(m) * mh * mw));
        cv::Mat full, bin;
        cv::resize(logits, full, {enc.orig_w, enc.orig_h}, 0, 0, cv::INTER_LINEAR);
        cv::threshold(full, bin, 0.0, 255.0, cv::THRESH_BINARY);
        bin.convertTo(bin, CV_8UC1);

        double score;
        if (have_coarse && bin.size() == coarse.size())
        {
            const double inter = cv::countNonZero(bin & coarse);
            const double uni    = cv::countNonZero(bin | coarse);
            score = (uni > 0.0) ? inter / uni : 0.0;
        }
        else
            score = (iou && m < niou) ? iou[m] : 0.0;

        if (score > best_score)
        {
            best_score = score;
            best_bin = std::move(bin);
        }
    }
    return best_bin;
}

std::vector<SegDetection> Sam2Refiner::refine(const cv::Mat& frame, const std::vector<SegDetection>& dets,
                                              bool is_bgr)
{
    std::vector<SegDetection> out = dets;   // copy; keep original masks as the fallback
    if (!ready() || frame.empty() || dets.empty())
        return out;

    cv::Mat rgb;
    if (is_bgr)
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    else
        rgb = frame;

    const auto t0 = std::chrono::steady_clock::now();
    const Encoded enc = encode(rgb);
    const auto t1 = std::chrono::steady_clock::now();
    if (!enc.ok)
        return out;

    int refined = 0;
    for (auto& d : out)
    {
        const cv::Rect clamped = d.bbox & cv::Rect(0, 0, frame.cols, frame.rows);
        if (clamped.width < 2 || clamped.height < 2)
            continue;
        // Pass the ORIGINAL YOLO mask (d.mask still holds it here) as the candidate-selection reference.
        if (cv::Mat m = decode(enc, clamped, d.mask); not m.empty() && cv::countNonZero(m) > 0)
        {
            d.mask = std::move(m);
            ++refined;
        }
    }
    const auto t2 = std::chrono::steady_clock::now();

    const double enc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double dec_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    // Timing: every run for the first few, then every 60th, so the encoder cost is visible without spam.
    if (cfg_.verbose || frame_count_ < 5 || frame_count_ % 60 == 0)
        std::println("[Sam2Refiner] frame {}: encode {:.1f} ms | decode {} obj {:.1f} ms | refined {}/{}",
                     frame_count_, enc_ms, dets.size(), dec_ms, refined, dets.size());
    ++frame_count_;
    return out;
}

cv::Mat Sam2Refiner::compose_canvas(const cv::Mat& frame, const std::vector<SegDetection>& refined,
                                    bool is_bgr) const
{
    cv::Mat canvas;
    if (is_bgr)
        canvas = frame.clone();
    else
        cv::cvtColor(frame, canvas, cv::COLOR_RGB2BGR);
    if (canvas.empty())
        return canvas;

    const cv::Scalar tint(255, 40, 220);   // magenta so SAM2 masks stand out vs YOLO's overlay colours
    cv::Mat overlay = canvas.clone();
    for (const auto& d : refined)
    {
        if (!d.mask.empty() && d.mask.size() == canvas.size())
            overlay.setTo(tint, d.mask);
        const cv::Rect b = d.bbox & cv::Rect(0, 0, canvas.cols, canvas.rows);
        if (b.width > 0 && b.height > 0)
            cv::rectangle(canvas, b, tint, 1);
    }
    cv::addWeighted(overlay, 0.45, canvas, 0.55, 0.0, canvas);
    return canvas;
}

}  // namespace rc::sam2
