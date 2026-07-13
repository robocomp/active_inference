#pragma once

/*
 * sam2_processor.h
 *
 * SAM2.1 mask-REFINEMENT engine. Unlike the YOLO detectors (which LOCATE objects), SAM2 is used here
 * purely to sharpen the boundary of an ALREADY-LOCATED object: given a YOLO SegDetection, its bbox is
 * turned into a box+centre prompt and SAM2 returns a tight, high-resolution mask for that object.
 *
 * Two-model pipeline (matches the samexporter ONNX export):
 *   encoder.onnx : image (1×3×1024×1024)  → image_embed + 2 high-res feature maps   (run ONCE / frame)
 *   decoder.onnx : those + point prompts  → low-res mask logits (256²) + IoU         (run per object)
 *
 * The encoder is the heavy part (~110 MB, 1024² hiera). encode() runs once per frame; decode() is cheap
 * and runs per detection. The whole thing is gated by a Stage (Sam2Stage) so the encoder only fires when
 * there is at least one target object AND the overlay/consumer is enabled.
 *
 * Self-contained ONNX sessions (mirrors YoloSemanticSegmenter plumbing) via the shared GPU providers.
 */

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "yolo_seg_detector.h"   // SegDetection

namespace rc::sam2
{

struct Config
{
    std::string encoder_path = "sam2.1_hiera_tiny.encoder.onnx";
    std::string decoder_path = "sam2.1_hiera_tiny.decoder.onnx";
    bool use_gpu = true;
    // Separate TRT flags: the encoder is FIXED-shape (1024²) → TRT-friendly, big FP16 win (but a slow
    // first-run engine build, cached after). The decoder is DYNAMIC-shape (num_labels/points) → TRT needs
    // optimization profiles or falls back; it's cheap on CUDA anyway. Default: encoder maybe, decoder no.
    bool encoder_use_trt = false;
    bool decoder_use_trt = false;
    // Feed the YOLO coarse mask as the decoder's mask_input prior (has_mask_input=1) → tighter boundaries
    // and even stronger rejection of the "hole between the legs" candidate. mask_prior_logit is the ±logit
    // magnitude written inside/outside the coarse silhouette (bigger = trust YOLO shape more).
    bool  mask_prior = true;
    float mask_prior_logit = 8.0f;
    bool verbose = false;   // per-frame encode/decode timing to stdout
};

// Owns the encoder+decoder sessions. refine() encodes the frame once, then decodes one box+centre prompt
// per detection, returning copies of the detections with their masks replaced by the SAM2 masks.
class Sam2Refiner
{
public:
    explicit Sam2Refiner(const Config& cfg);   // throws std::runtime_error on load failure
    ~Sam2Refiner();

    Sam2Refiner(const Sam2Refiner&)            = delete;
    Sam2Refiner& operator=(const Sam2Refiner&) = delete;

    [[nodiscard]] bool ready() const noexcept { return enc_session_ && dec_session_; }

    // Refine every detection in `dets`. `frame` is the ZED frame (BGR by default). Each returned
    // SegDetection keeps bbox/class/label/confidence but its `mask` is the full-frame CV_8UC1 (0/255)
    // SAM2 mask. Detections whose SAM2 pass fails keep their original YOLO mask (never worse).
    [[nodiscard]] std::vector<SegDetection> refine(const cv::Mat& frame, const std::vector<SegDetection>& dets,
                                                   bool is_bgr = true);

    // A/B overlay: tint each refined mask + draw its prompt box onto a BGR copy of `frame`.
    [[nodiscard]] cv::Mat compose_canvas(const cv::Mat& frame, const std::vector<SegDetection>& refined,
                                         bool is_bgr = true) const;

private:
    static constexpr int ENC = 1024;   // encoder input side (fixed by the export)

    // One encoded frame: the 3 encoder outputs kept as owned CPU buffers so they can be re-fed to the
    // decoder as input tensors. Names/shapes are captured at load from the actual graph.
    struct Encoded
    {
        std::vector<float>   image_embed, high_res_feats_0, high_res_feats_1;
        std::vector<int64_t> embed_shape, hr0_shape, hr1_shape;
        int orig_w = 0, orig_h = 0;
        bool ok = false;
    };

    [[nodiscard]] Encoded encode(const cv::Mat& rgb) const;
    // Decode a single BOX prompt (full-frame pixel coords) → full-frame CV_8UC1 mask (0/255). SAM2 emits
    // several candidate masks; `coarse` (the YOLO mask for this object, full-frame 0/255) selects the one
    // that best overlaps the located object — robust for concave/see-through shapes where a centre point
    // would land in a hole. Empty `coarse` falls back to SAM's predicted-IoU. Returns empty on failure.
    [[nodiscard]] cv::Mat decode(const Encoded& enc, const cv::Rect& bbox, const cv::Mat& coarse) const;

    std::unique_ptr<Ort::Env>     env_;
    Ort::SessionOptions           enc_opts_, dec_opts_;
    std::unique_ptr<Ort::Session> enc_session_, dec_session_;

    // cached I/O names (owned strdup'd C strings + cstr views), per session
    std::vector<char*>       enc_in_, enc_out_, dec_in_, dec_out_;
    std::vector<const char*> enc_in_cstr_, enc_out_cstr_, dec_in_cstr_, dec_out_cstr_;

    Config cfg_;
    mutable std::uint64_t frame_count_ = 0;   // for periodic timing prints
};

}  // namespace rc::sam2
