#include "yolo_semantic.h"
#include "onnx_providers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <charconv>
#include <cstring>
#include <string_view>
#include <iostream>
#include <limits>
#include <print>
#include <stdexcept>

namespace rc::semantic
{

YoloSemanticSegmenter::YoloSemanticSegmenter(const std::string& model_path,
                                             const std::vector<std::string>& class_names,
                                             float conf_thresh,
                                             int input_size,
                                             bool use_gpu,
                                             bool use_trt)
    : conf_thresh_(conf_thresh)
    , input_size_(input_size)
{
    class_names_ = class_names.empty() ? default_class_names() : class_names;

    try
    {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "YoloSemanticSegmenter");
        session_opts_.SetIntraOpNumThreads(1);
        session_opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        rc::onnx::append_gpu_providers(session_opts_, use_gpu, use_trt, "[YoloSemanticSegmenter]");

        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_opts_);

        Ort::AllocatorWithDefaultOptions allocator;
        const std::size_t n_in = session_->GetInputCount();
        input_names_.reserve(n_in);
        input_names_cstr_.reserve(n_in);
        for (std::size_t i = 0; i < n_in; ++i)
        {
            auto name = session_->GetInputNameAllocated(i, allocator);
            input_names_.push_back(strdup(name.get()));
            input_names_cstr_.push_back(input_names_.back());
        }
        const std::size_t n_out = session_->GetOutputCount();
        output_names_.reserve(n_out);
        output_names_cstr_.reserve(n_out);
        for (std::size_t i = 0; i < n_out; ++i)
        {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(strdup(name.get()));
            output_names_cstr_.push_back(output_names_.back());
        }
        // ★THE MODEL DECLARES ITS OWN CHANNEL MAP (tools/expose_semantic_logits.py writes it).
        // Read here rather than from config: a re-export with a different --classes order against a
        // stale config key would read P(door) out of the cabinet channel with nothing to catch it.
        // Absent ⇒ an ungraded export ⇒ prob_class_ids_ stays empty ⇒ the old path, unchanged.
        try
        {
            auto meta = session_->GetModelMetadata();
            if (auto v = meta.LookupCustomMetadataMapAllocated("prob_class_ids", allocator); v)
            {
                const std::string csv{v.get()};
                for (std::size_t b = 0, e = 0; b <= csv.size(); b = e + 1)
                {
                    e = csv.find(',', b);
                    if (e == std::string::npos) e = csv.size();
                    int id = -1;
                    const auto* first = csv.data() + b;
                    if (std::from_chars(first, csv.data() + e, id).ec == std::errc{} and id >= 0)
                        prob_class_ids_.push_back(id);
                }
            }
        }
        catch (const Ort::Exception&) { /* no metadata map ⇒ ungraded model, not an error */ }

        std::cout << "[YoloSemanticSegmenter] Loaded: " << model_path
                  << "  inputs=" << n_in << "  outputs=" << n_out
                  << "  classes=" << class_names_.size();
        if (prob_class_ids_.empty())
            std::cout << "  graded=NO (argmax only; P(class) unavailable)\n";
        else
        {
            std::cout << "  graded=YES channels=[";
            for (std::size_t i = 0; i < prob_class_ids_.size(); ++i)
                std::cout << (i ? "," : "") << label_for(prob_class_ids_[i]);
            std::cout << "]\n";
        }
    }
    catch (const Ort::Exception& e)
    {
        throw std::runtime_error(std::string("[YoloSemanticSegmenter] ONNX error: ") + e.what());
    }
}

YoloSemanticSegmenter::~YoloSemanticSegmenter()
{
    for (char* n : input_names_)
        free(n);
    for (char* n : output_names_)
        free(n);
}

const std::string& YoloSemanticSegmenter::label_for(int class_id) const
{
    static const std::string unknown = "unknown";
    if (class_id < 0 || class_id >= static_cast<int>(class_names_.size()))
        return unknown;
    return class_names_[static_cast<std::size_t>(class_id)];
}

YoloSemanticSegmenter::LetterboxResult YoloSemanticSegmenter::preprocess(const cv::Mat& rgb_image) const
{
    const float scale = std::min(static_cast<float>(input_size_) / rgb_image.cols,
                                 static_cast<float>(input_size_) / rgb_image.rows);
    const int new_w = static_cast<int>(std::round(rgb_image.cols * scale));
    const int new_h = static_cast<int>(std::round(rgb_image.rows * scale));
    const int pad_l = (input_size_ - new_w) / 2;
    const int pad_t = (input_size_ - new_h) / 2;

    cv::Mat resized;
    cv::resize(rgb_image, resized, {new_w, new_h}, 0, 0, cv::INTER_LINEAR);
    cv::Mat lb(input_size_, input_size_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(lb(cv::Rect(pad_l, pad_t, new_w, new_h)));

    cv::Mat lb_f;
    lb.convertTo(lb_f, CV_32FC3, 1.0 / 255.0);
    std::vector<cv::Mat> channels(3);
    cv::split(lb_f, channels);

    const int stride = input_size_ * input_size_;
    std::vector<float> tensor(3 * stride);
    for (int c = 0; c < 3; ++c)
        std::memcpy(tensor.data() + c * stride, channels[c].data, stride * sizeof(float));

    return {std::move(tensor), scale, pad_l, pad_t};
}

// Wrap a graded ONNX tensor plane as CV_32FC1. The export is float16 by default (it is 4x smaller
// and the values are probabilities, so 3 decimal digits is ample); float32 is accepted too.
// ★Always materialises a NEW Mat - the ORT buffer dies with `outputs` at the end of segment(),
// so a view would dangle the moment the map crossed to the worker's result.
static cv::Mat graded_plane(const void* base, ONNXTensorElementDataType t, int h, int w, int plane)
{
    cv::Mat out32;
    if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
    {
        const cv::Mat h16(h, w, CV_16FC1,
                          const_cast<void*>(static_cast<const void*>(
                              static_cast<const std::uint16_t*>(base) + std::size_t(plane) * h * w)));
        h16.convertTo(out32, CV_32F);
    }
    else if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    {
        const cv::Mat f32(h, w, CV_32FC1,
                          const_cast<void*>(static_cast<const void*>(
                              static_cast<const float*>(base) + std::size_t(plane) * h * w)));
        out32 = f32.clone();
    }
    return out32;
}

SemanticMap YoloSemanticSegmenter::decode(const void* data,
                                          ONNXTensorElementDataType elem_type,
                                          const std::vector<int64_t>& shape,
                                          const cv::Size& orig_size,
                                          float scale, int pad_left, int pad_top,
                                          bool want_scores,
                                          const GradedOutputs& graded) const
{
    // Accepted output layouts (the *-sem exports differ by dtype):
    //   float  [1,C,h,w] / [C,h,w]  → per-class LOGIT volume; label = argmax over the class axis.
    //   int    [1,h,w] / [1,1,h,w] / [h,w]  → a DENSE class-id map (uint8/int16/int32 per the docs).
    // CRITICAL: a *-sem-ade20k export emits the dense int map, so we must read it with the actual
    // element type — reading it as float (4 B/elem) over-reads a uint8 buffer 4× → SIGSEGV.
    const bool is_float = (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    int channels = 1, oh = 0, ow = 0;
    bool is_logit_volume = false;
    if (shape.size() == 4)
    {
        channels = static_cast<int>(shape[1]);
        oh = static_cast<int>(shape[2]);
        ow = static_cast<int>(shape[3]);
        is_logit_volume = is_float and channels > 1;
    }
    else if (shape.size() == 3)
    {
        // float [C,h,w] (no batch) is a logit volume iff the leading dim == #classes; otherwise it's
        // a [1,h,w] dense map. Integer outputs are always dense maps.
        if (is_float and static_cast<int>(shape[0]) == num_classes() and num_classes() > 1)
        {
            channels = static_cast<int>(shape[0]);
            oh = static_cast<int>(shape[1]);
            ow = static_cast<int>(shape[2]);
            is_logit_volume = true;
        }
        else
        {
            oh = static_cast<int>(shape[1]);
            ow = static_cast<int>(shape[2]);
        }
    }
    else if (shape.size() == 2)
    {
        oh = static_cast<int>(shape[0]);
        ow = static_cast<int>(shape[1]);
    }
    else
        throw std::runtime_error("[YoloSemanticSegmenter] unexpected output rank (want 2, 3 or 4)");

    if (oh <= 0 or ow <= 0)
        throw std::runtime_error("[YoloSemanticSegmenter] empty output spatial dims");

    cv::Mat lb_labels(oh, ow, CV_8UC1);
    cv::Mat lb_scores;
    if (want_scores)
        lb_scores = cv::Mat(oh, ow, CV_32FC1);

    const int plane = oh * ow;

    if (is_logit_volume)
    {
        // Per pixel: a single pass over the C channels gives argmax + max logit + Σexp(logit-max).
        // The winning softmax probability 1/Σexp is a calibrated confidence we threshold on.
        const float* fdata = static_cast<const float*>(data);
        for (int py = 0; py < oh; ++py)
        {
            auto* lrow = lb_labels.ptr<unsigned char>(py);
            auto* srow = want_scores ? lb_scores.ptr<float>(py) : nullptr;
            for (int px = 0; px < ow; ++px)
            {
                const int idx = py * ow + px;
                float max_logit = -std::numeric_limits<float>::infinity();
                int best = 0;
                for (int c = 0; c < channels; ++c)
                {
                    const float v = fdata[c * plane + idx];
                    if (v > max_logit)
                    {
                        max_logit = v;
                        best = c;
                    }
                }
                float sum_exp = 0.0f;
                for (int c = 0; c < channels; ++c)
                    sum_exp += std::exp(fdata[c * plane + idx] - max_logit);
                const float conf = (sum_exp > 0.0f) ? 1.0f / sum_exp : 0.0f;

                lrow[px] = (conf >= conf_thresh_)
                    ? static_cast<unsigned char>(best)
                    : IGNORE_LABEL;
                if (srow)
                    srow[px] = conf;
            }
        }
    }
    else
    {
        // Dense class-id map. Read with the ACTUAL element type so the stride matches the buffer.
        auto id_at = [&](int i) -> int
        {
            switch (elem_type)
            {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:  return static_cast<const std::uint8_t*>(data)[i];
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:   return static_cast<const std::int8_t*>(data)[i];
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return static_cast<const std::uint16_t*>(data)[i];
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:  return static_cast<const std::int16_t*>(data)[i];
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:  return static_cast<const std::int32_t*>(data)[i];
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:  return static_cast<int>(static_cast<const std::int64_t*>(data)[i]);
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:  return static_cast<int>(std::lround(static_cast<const float*>(data)[i]));
                default:
                    throw std::runtime_error("[YoloSemanticSegmenter] unsupported dense-map element type");
            }
        };
        for (int py = 0; py < oh; ++py)
        {
            auto* lrow = lb_labels.ptr<unsigned char>(py);
            auto* srow = want_scores ? lb_scores.ptr<float>(py) : nullptr;
            for (int px = 0; px < ow; ++px)
            {
                const int id = id_at(py * ow + px);
                lrow[px] = (id >= 0 and id < 255) ? static_cast<unsigned char>(id) : IGNORE_LABEL;
                if (srow)
                    srow[px] = 1.0f;
            }
        }
    }

    // Unletterbox: the model output is a downscaled copy of the input_size² letterbox, so the
    // active (non-padded) region scales by (ow/input_size, oh/input_size). Crop it, then resize
    // to the original frame — NEAREST for labels (never interpolate class ids), LINEAR for scores.
    const float fx = static_cast<float>(ow) / static_cast<float>(input_size_);
    const float fy = static_cast<float>(oh) / static_cast<float>(input_size_);
    const int new_w = static_cast<int>(std::round(orig_size.width * scale));
    const int new_h = static_cast<int>(std::round(orig_size.height * scale));

    cv::Rect active(static_cast<int>(std::round(pad_left * fx)),
                    static_cast<int>(std::round(pad_top * fy)),
                    static_cast<int>(std::round(new_w * fx)),
                    static_cast<int>(std::round(new_h * fy)));
    active &= cv::Rect(0, 0, ow, oh);
    if (active.width <= 0 or active.height <= 0)
        active = cv::Rect(0, 0, ow, oh);

    SemanticMap out;
    cv::resize(lb_labels(active), out.labels, orig_size, 0, 0, cv::INTER_NEAREST);

    // ─── GRADED OUTPUTS ───────────────────────────────────────────────────────────────────────────
    // The probability planes come from the classifier head at ITS resolution (80x80 for a 640
    // letterbox), not from the 640x640 label map, so their active region has to be recomputed in
    // their own units — same fractions, different denominator.
    const bool has_graded = graded.valid()
                        and graded.k == static_cast<int>(prob_class_ids_.size());
    cv::Rect g_active;
    if (has_graded)
    {
        const float gfx = static_cast<float>(graded.w) / static_cast<float>(input_size_);
        const float gfy = static_cast<float>(graded.h) / static_cast<float>(input_size_);
        g_active = cv::Rect(static_cast<int>(std::round(pad_left * gfx)),
                            static_cast<int>(std::round(pad_top  * gfy)),
                            static_cast<int>(std::round(new_w    * gfx)),
                            static_cast<int>(std::round(new_h    * gfy)));
        g_active &= cv::Rect(0, 0, graded.w, graded.h);
        if (g_active.width <= 0 or g_active.height <= 0)
            g_active = cv::Rect(0, 0, graded.w, graded.h);

        out.prob_class_ids = prob_class_ids_;
        out.probs_src_size = orig_size;
        out.probs.reserve(prob_class_ids_.size());
        for (int c = 0; c < graded.k; ++c)
        {
            cv::Mat plane = graded_plane(graded.class_probs, graded.elem_type,
                                         graded.h, graded.w, c);
            // clone(): keep only the active window, and own the memory — see graded_plane().
            out.probs.push_back(plane.empty() ? cv::Mat{} : plane(g_active).clone());
        }
    }

    if (want_scores)
    {
        // ★`top_prob` is the max over ALL 150 classes, i.e. the argmax's own confidence, so with a
        // graded model `scores` is finally the real thing for EVERY class — not just the handful in
        // prob_class_ids_. Without it the dense-int path below writes 1.0 everywhere, which is what
        // made mask confidence carry zero information for every semantic-path agent (door, hood,
        // cabinet, shelf) and left [Semantic].conf_thresh an inoperative knob.
        if (has_graded and graded.top_prob)
        {
            cv::Mat top = graded_plane(graded.top_prob, graded.elem_type, graded.h, graded.w, 0);
            if (not top.empty())
                cv::resize(top(g_active), out.scores, orig_size, 0, 0, cv::INTER_LINEAR);
        }
        if (out.scores.empty())
            cv::resize(lb_scores(active), out.scores, orig_size, 0, 0, cv::INTER_LINEAR);
    }
    return out;
}

SemanticMap YoloSemanticSegmenter::segment(const cv::Mat& image, bool is_rgb, bool want_scores) const
{
    if (image.empty())
    {
        std::cerr << "[YoloSemanticSegmenter] segment() called with empty image\n";
        return {};
    }

    cv::Mat rgb;
    if (is_rgb)
        rgb = image;
    else
        cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

    const cv::Size orig_size = rgb.size();
    auto [tensor, scale, pad_l, pad_t] = preprocess(rgb);

    const std::array<int64_t, 4> input_shape{1, 3, input_size_, input_size_};
    const auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info,
                                                              tensor.data(),
                                                              tensor.size(),
                                                              input_shape.data(),
                                                              input_shape.size());

    std::vector<Ort::Value> outputs;
    try
    {
        outputs = session_->Run(Ort::RunOptions{nullptr},
                                input_names_cstr_.data(), &input_tensor, 1,
                                output_names_cstr_.data(), output_names_cstr_.size());
    }
    catch (const Ort::Exception& e)
    {
        std::cerr << "[YoloSemanticSegmenter] Inference error: " << e.what() << '\n';
        return {};
    }

    if (outputs.empty())
    {
        std::cerr << "[YoloSemanticSegmenter] model produced no output tensor\n";
        return {};
    }

    // Locate the graded outputs BY NAME (output order is an exporter detail, never an interface).
    // Absent ⇒ ungraded model ⇒ `graded` stays invalid and decode() takes exactly the old path.
    GradedOutputs graded;
    for (std::size_t i = 0; i < outputs.size() and i < output_names_.size(); ++i)
    {
        if (not outputs[i].IsTensor())
            continue;
        const auto gi = outputs[i].GetTensorTypeAndShapeInfo();
        const auto gs = gi.GetShape();
        if (gs.size() != 4)
            continue;
        const std::string_view nm{output_names_[i]};
        if (nm == "class_probs")
        {
            graded.class_probs = outputs[i].GetTensorData<void>();
            graded.elem_type   = gi.GetElementType();
            graded.k = static_cast<int>(gs[1]);
            graded.h = static_cast<int>(gs[2]);
            graded.w = static_cast<int>(gs[3]);
        }
        else if (nm == "top_prob")
            graded.top_prob = outputs[i].GetTensorData<void>();
    }
    // A channel map that does not match the tensor is a mis-paired export: refuse the posteriors
    // rather than silently attributing one class's probability to another.
    if (graded.class_probs and graded.k != static_cast<int>(prob_class_ids_.size()))
    {
        std::cerr << "[YoloSemanticSegmenter] class_probs has " << graded.k
                  << " channels but metadata declares " << prob_class_ids_.size()
                  << " — ignoring the graded outputs\n";
        graded = {};
    }

    const auto info = outputs[0].GetTensorTypeAndShapeInfo();
    const auto out_shape = info.GetShape();
    const auto elem_type = info.GetElementType();
    const void* out_data = outputs[0].GetTensorData<void>();

    // One-shot: report the actual output layout so the chosen decode path is verifiable.
    if (not logged_layout_)
    {
        logged_layout_ = true;
        std::string dims;
        for (auto d : out_shape)
            dims += (dims.empty() ? "" : "x") + std::to_string(d);
        std::cout << "[YoloSemanticSegmenter] output: shape=[" << dims << "] elem_type="
                  << static_cast<int>(elem_type) << " (1=float; 2=uint8,3=int8,4=uint16,5=int16,6=int32,7=int64)\n";
    }

    try
    {
        return decode(out_data, elem_type, out_shape, orig_size, scale, pad_l, pad_t, want_scores,
                      graded);
    }
    catch (const std::exception& e)
    {
        std::cerr << "[YoloSemanticSegmenter] decode failed: " << e.what() << '\n';
        return {};
    }
}

std::vector<std::string> YoloSemanticSegmenter::default_class_names()
{
    // ADE20K SceneParse150 label set — the 150 classes yolo26*-sem-ade20k is trained on, in
    // model class-id order (0 = wall … 149 = flag). Names are the primary synonym of each ADE20K
    // category. Pixel value 255 is the dataset's "ignore" label (== IGNORE_LABEL here).
    return {
        "wall","building","sky","floor","tree","ceiling","road","bed","windowpane","grass",
        "cabinet","sidewalk","person","earth","door","table","mountain","plant","curtain","chair",
        "car","water","painting","sofa","shelf","house","sea","mirror","rug","field",
        "armchair","seat","fence","desk","rock","wardrobe","lamp","bathtub","railing","cushion",
        "base","box","column","signboard","chest of drawers","counter","sand","sink","skyscraper","fireplace",
        "refrigerator","grandstand","path","stairs","runway","case","pool table","pillow","screen door","stairway",
        "river","bridge","bookcase","blind","coffee table","toilet","flower","book","hill","bench",
        "countertop","stove","palm","kitchen island","computer","swivel chair","boat","bar","arcade machine","hovel",
        "bus","towel","light","truck","tower","chandelier","awning","streetlight","booth","television receiver",
        "airplane","dirt track","apparel","pole","land","bannister","escalator","ottoman","bottle","buffet",
        "poster","stage","van","ship","fountain","conveyer belt","canopy","washer","plaything","swimming pool",
        "stool","barrel","basket","waterfall","tent","bag","minibike","cradle","oven","ball",
        "food","step","tank","trade name","microwave","pot","animal","bicycle","lake","dishwasher",
        "screen","blanket","sculpture","hood","sconce","vase","traffic light","tray","ashcan","fan",
        "pier","crt screen","plate","monitor","bulletin board","shower","radiator","glass","clock","flag"
    };
}

// ─────────────────────────────────────────────────────────────────────────────
//  YoloSemanticProcessor
// ─────────────────────────────────────────────────────────────────────────────

void YoloSemanticProcessor::configure(const Config& config)
{
    config_ = config;
    segmenter_.emplace(config_.model_path,
                       config_.class_names,
                       config_.conf_thresh,
                       config_.input_size,
                       config_.use_gpu,
                       config_.use_trt);
    if (config_.verbose_debug)
        std::println("[YoloSemantic] segmenter ready: {}", config_.model_path);
}

SemanticMap YoloSemanticProcessor::segment(const cv::Mat& rgb_frame, bool want_scores)
{
    if (!segmenter_.has_value() or rgb_frame.empty()
        or rgb_frame.cols <= 0 or rgb_frame.rows <= 0)
        return {};

    last_map_ = segmenter_->segment(rgb_frame, true, want_scores);
    return last_map_;
}

cv::Mat YoloSemanticProcessor::compose_semantic_canvas(const cv::Mat& rgb_frame,
                                                       const SemanticMap& map) const
{
    // Preserve the input colour space (the YOLO viewer feeds — and expects back — RGB). The palette
    // is arbitrary, so the overlay reads the same either way; we just must not swap the frame's R/B.
    if (rgb_frame.empty())
        return {};
    cv::Mat canvas = rgb_frame.clone();
    if (map.labels.empty() or map.labels.size() != canvas.size())
        return canvas;

    // Canonical ADE20K (MIT SceneParse150 / mmsegmentation) colormap: one DISTINCT colour per class
    // id, in [R,G,B] order — applied directly on the RGB viewer frame so it matches published ADE20K
    // visualizations. Indexed by class id (0=wall … 149=flag); out-of-range / ignore → passthrough.
    static const unsigned char ADE20K_PALETTE[150][3] = {
        {120,120,120}, {180,120,120}, {  6,230,230}, { 80, 50, 50}, {  4,200,  3}, {120,120, 80},
        {140,140,140}, {204,  5,255}, {230,230,230}, {  4,250,  7}, {224,  5,255}, {235,255,  7},
        {150,  5, 61}, {120,120, 70}, {  8,255, 51}, {255,  6, 82}, {143,255,140}, {204,255,  4},
        {255, 51,  7}, {204, 70,  3}, {  0,102,200}, { 61,230,250}, {255,  6, 51}, { 11,102,255},
        {255,  7, 71}, {255,  9,224}, {  9,  7,230}, {220,220,220}, {255,  9, 92}, {112,  9,255},
        {  8,255,214}, {  7,255,224}, {255,184,  6}, { 10,255, 71}, {255, 41, 10}, {  7,255,255},
        {224,255,  8}, {102,  8,255}, {255, 61,  6}, {255,194,  7}, {255,122,  8}, {  0,255, 20},
        {255,  8, 41}, {255,  5,153}, {  6, 51,255}, {235, 12,255}, {160,150, 20}, {  0,163,255},
        {140,140,140}, {250, 10, 15}, { 20,255,  0}, { 31,255,  0}, {255, 31,  0}, {255,224,  0},
        {153,255,  0}, {  0,  0,255}, {255, 71,  0}, {  0,235,255}, {  0,173,255}, { 31,  0,255},
        { 11,200,200}, {255, 82,  0}, {  0,255,245}, {  0, 61,255}, {  0,255,112}, {  0,255,133},
        {255,  0,  0}, {255,163,  0}, {255,102,  0}, {194,255,  0}, {  0,143,255}, { 51,255,  0},
        {  0, 82,255}, {  0,255, 41}, {  0,255,173}, { 10,  0,255}, {173,255,  0}, {  0,255,153},
        {255, 92,  0}, {255,  0,255}, {255,  0,245}, {255,  0,102}, {255,173,  0}, {255,  0, 20},
        {255,184,184}, {  0, 31,255}, {  0,255, 61}, {  0, 71,255}, {255,  0,204}, {  0,255,194},
        {  0,255, 82}, {  0, 10,255}, {  0,112,255}, { 51,  0,255}, {  0,194,255}, {  0,122,255},
        {  0,255,163}, {255,153,  0}, {  0,255, 10}, {255,112,  0}, {143,255,  0}, { 82,  0,255},
        {163,255,  0}, {255,235,  0}, {  8,184,170}, {133,  0,255}, {  0,255, 92}, {184,  0,255},
        {255,  0, 31}, {  0,184,255}, {  0,214,255}, {255,  0,112}, { 92,255,  0}, {  0,224,255},
        {112,224,255}, { 70,184,160}, {163,  0,255}, {153,  0,255}, { 71,255,  0}, {255,  0,163},
        {255,204,  0}, {255,  0,143}, {  0,255,235}, {133,255,  0}, {255,  0,235}, {245,  0,255},
        {255,  0,122}, {255,245,  0}, { 10,190,212}, {214,255,  0}, {  0,204,255}, { 20,  0,255},
        {255,255,  0}, {  0,153,255}, {  0, 41,255}, {  0,255,204}, { 41,  0,255}, { 41,255,  0},
        {173,  0,255}, {  0,245,255}, { 71,  0,255}, {122,  0,255}, {  0,255,184}, {  0, 92,255},
        {184,255,  0}, {  0,133,255}, {255,214,  0}, { 25,194,194}, {102,255,  0}, { 92,  0,255},
    };

    // Paint a colour layer from the label map, then blend it over the frame (ignore = passthrough).
    cv::Mat color_layer(canvas.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat valid(canvas.size(), CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < map.labels.rows; ++y)
    {
        const auto* lrow = map.labels.ptr<unsigned char>(y);
        auto* crow = color_layer.ptr<cv::Vec3b>(y);
        auto* vrow = valid.ptr<unsigned char>(y);
        for (int x = 0; x < map.labels.cols; ++x)
        {
            const unsigned char id = lrow[x];
            if (id >= 150)            // IGNORE_LABEL (255) or out-of-range → leave the frame untouched
                continue;
            const auto& col = ADE20K_PALETTE[id];
            crow[x] = cv::Vec3b(col[0], col[1], col[2]);
            vrow[x] = 255;
        }
    }

    cv::Mat blended;
    cv::addWeighted(canvas, 0.55, color_layer, 0.45, 0.0, blended);
    blended.copyTo(canvas, valid);
    return canvas;
}

cv::Mat compose_prob_canvas(const cv::Mat& img, const SemanticMap& map, int class_id, bool img_is_rgb)
{
    if (img.empty() or img.type() != CV_8UC3)
        return img.clone();
    cv::Mat canvas = img.clone();
    const int ch = map.prob_channel(class_id);
    if (ch < 0 or not map.graded() or map.probs[static_cast<std::size_t>(ch)].empty())
        return canvas;

    // Sample the native-resolution plane up to frame size through prob_at's own convention, so what is
    // drawn and what a consumer reads are the same number by construction (the writer and the reader
    // are in different files; this is the only thing that keeps them coherent).
    const cv::Mat& plane = map.probs[static_cast<std::size_t>(ch)];
    cv::Mat p;
    cv::resize(plane, p, canvas.size(), 0, 0, cv::INTER_LINEAR);

    // Colormap over the FULL [0,1] range, fixed — never per-frame normalised. An auto-scaled heat map
    // would make a frame whose best pixel is 0.05 look identical to one whose best is 0.95, which
    // would defeat the entire purpose of showing a posterior.
    cv::Mat p8, heat;
    p.convertTo(p8, CV_8UC1, 255.0, 0.0);
    cv::applyColorMap(p8, heat, cv::COLORMAP_INFERNO);   // emits BGR
    if (img_is_rgb)
        cv::cvtColor(heat, heat, cv::COLOR_BGR2RGB);

    // Per-pixel alpha = P. Done by hand rather than with addWeighted, which takes a scalar weight.
    for (int r = 0; r < canvas.rows; ++r)
    {
        const float*    pr = p.ptr<float>(r);
        const cv::Vec3b* hr = heat.ptr<cv::Vec3b>(r);
        cv::Vec3b*       cr = canvas.ptr<cv::Vec3b>(r);
        for (int c = 0; c < canvas.cols; ++c)
        {
            if (std::isnan(pr[c]))
            {
                // "Not looked at" — a diagonal hatch, so absence of evidence never renders as evidence
                // of absence. Visibly different from any probability, including zero.
                if (((r + c) % 16) < 2)
                    cr[c] = cv::Vec3b(90, 90, 90);
                continue;
            }
            const float a = std::clamp(pr[c], 0.0f, 1.0f);
            for (int k = 0; k < 3; ++k)
                cr[c][k] = cv::saturate_cast<unsigned char>((1.0f - a) * cr[c][k] + a * hr[c][k]);
        }
    }
    return canvas;
}

}  // namespace rc::semantic
