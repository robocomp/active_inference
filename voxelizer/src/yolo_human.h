#pragma once

/*
 * yolo_human.h
 *
 * Human-pose branch of the voxelizer perception stack. Runs a YOLO-pose ONNX model
 * (e.g. yolo26m-pose) on the same RGB frame the seg detector consumes, and returns
 * per-person 2D keypoints (COCO-17) + per-joint confidence. The voxelizer deprojects
 * these to CAMERA-frame 3D (graph_publisher) and publishes them on a dedicated
 * 'skeleton' DSR node consumed by human_concept — NOT the 'masks' node (a pose is an
 * ordered keypoint set with per-joint confidence, semantically unlike a silhouette).
 *
 * Self-contained ONNX session (mirrors YoloSegDetector's plumbing). Pose output has no
 * mask protos: a single 'person' class + 17 (x,y,conf) keypoints per detection.
 */

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace rc::human_pose
{

inline constexpr int NUM_COCO_KP = 17;   // ultralytics pose keypoint set
inline constexpr int NUM_BODY18  = 18;   // ZED/OpenPose BODY_18 (= COCO-17 + synthesized neck)

// One COCO-17 keypoint in original-image pixels. conf in [0,1].
struct PoseKp
{
    float x = 0.f;
    float y = 0.f;
    float conf = 0.f;
};

// One detected person: bbox (image px), person confidence, optional track id, COCO-17 keypoints.
struct PoseDetection
{
    cv::Rect bbox;
    float    confidence = 0.f;
    int      track_id = -1;
    std::array<PoseKp, NUM_COCO_KP> kp{};
};

// BODY_18 index -> COCO-17 index. -1 = synthesized (neck = midpoint of shoulders).
// BODY_18 order (rc::human::KP): nose,neck,r_sh,r_el,r_wr,l_sh,l_el,l_wr,r_hip,r_knee,
//                                r_ank,l_hip,l_knee,l_ank,r_eye,l_eye,r_ear,l_ear.
// COCO-17 order: nose,l_eye,r_eye,l_ear,r_ear,l_sh,r_sh,l_el,r_el,l_wr,r_wr,
//                l_hip,r_hip,l_knee,r_knee,l_ank,r_ank.
inline constexpr std::array<int, NUM_BODY18> BODY18_FROM_COCO = {
    0,   // 0  nose       <- nose
    -1,  // 1  neck       <- mid(l_sh=5, r_sh=6)
    6,   // 2  r_shoulder <- r_sh
    8,   // 3  r_elbow    <- r_el
    10,  // 4  r_wrist    <- r_wr
    5,   // 5  l_shoulder <- l_sh
    7,   // 6  l_elbow    <- l_el
    9,   // 7  l_wrist    <- l_wr
    12,  // 8  r_hip      <- r_hip
    14,  // 9  r_knee     <- r_knee
    16,  // 10 r_ankle    <- r_ankle
    11,  // 11 l_hip      <- l_hip
    13,  // 12 l_knee     <- l_knee
    15,  // 13 l_ankle    <- l_ankle
    2,   // 14 r_eye      <- r_eye
    1,   // 15 l_eye      <- l_eye
    4,   // 16 r_ear      <- r_ear
    3,   // 17 l_ear      <- l_ear
};

// COCO indices needed to synthesize the neck.
inline constexpr int COCO_L_SHOULDER = 5;
inline constexpr int COCO_R_SHOULDER = 6;

class YoloPoseDetector
{
public:
    explicit YoloPoseDetector(const std::string& model_path,
                              float conf_thresh = 0.30f,
                              float iou_thresh = 0.45f,
                              int input_size = 640,
                              bool use_gpu = true,
                              bool use_trt = false);
    ~YoloPoseDetector();

    YoloPoseDetector(const YoloPoseDetector&) = delete;
    YoloPoseDetector& operator=(const YoloPoseDetector&) = delete;
    YoloPoseDetector(YoloPoseDetector&&) = default;
    YoloPoseDetector& operator=(YoloPoseDetector&&) = default;

    [[nodiscard]] std::vector<PoseDetection> detect(const cv::Mat& image, bool is_rgb = false) const;

    void set_conf_threshold(float t) noexcept { conf_thresh_ = t; }
    [[nodiscard]] float conf_threshold() const noexcept { return conf_thresh_; }

private:
    std::unique_ptr<Ort::Env>     env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::SessionOptions           session_opts_;
    std::vector<char*>            input_names_;
    std::vector<char*>            output_names_;
    std::vector<const char*>      input_names_cstr_;
    std::vector<const char*>      output_names_cstr_;
    float conf_thresh_;
    float iou_thresh_;
    int   input_size_;

    struct LetterboxResult
    {
        std::vector<float> tensor;
        float scale;
        int   pad_left;
        int   pad_top;
    };

    [[nodiscard]] LetterboxResult preprocess(const cv::Mat& rgb_image) const;
    [[nodiscard]] std::vector<PoseDetection> parse(const float* data,
                                                   const std::vector<int64_t>& shape,
                                                   const cv::Size& orig_size,
                                                   float scale, int pad_left, int pad_top) const;
    [[nodiscard]] std::vector<int> nms(const std::vector<PoseDetection>& dets) const;
    [[nodiscard]] static float iou(const cv::Rect& a, const cv::Rect& b) noexcept;
};

// Thin processor: owns the detector (lazily, like YoloProcessor) + config gating.
class YoloHumanProcessor
{
public:
    struct Config
    {
        std::string model_path = "yolo26m-pose.onnx";
        float conf_thresh = 0.30f;
        float iou_thresh = 0.45f;
        int   input_size = 640;
        bool  use_gpu = true;
        bool  use_trt = false;
        bool  verbose_debug = false;
        // Keep drawing the last good skeleton for up to this long after a detection, so a single
        // decimated frame that momentarily misses the person doesn't blank the overlay (anti-flicker).
        std::uint64_t hold_ms = 500;
    };

    YoloHumanProcessor() = default;

    void configure(const Config& config);
    [[nodiscard]] bool ready() const noexcept { return detector_.has_value(); }

    // Returns detected people with COCO-17 keypoints (image px). Empty if not configured.
    // Also caches the result (see last_poses) so callers running the model on a decimated schedule
    // can redraw the most recent detection every frame without flicker. `now_ms` is the capture stamp
    // of rgb_frame: a fresh non-empty detection refreshes the cache + stamp; an empty result keeps the
    // cached skeleton until `Config::hold_ms` has elapsed since the last good detection, then clears.
    [[nodiscard]] std::vector<PoseDetection> detect_poses(const cv::Mat& rgb_frame, std::uint64_t now_ms);

    // The most recent detect_poses() result. Lets a decimated caller keep drawing the last skeleton
    // on cycles where the model didn't run.
    [[nodiscard]] const std::vector<PoseDetection>& last_poses() const noexcept { return last_poses_; }

    // Overlay skeletons on a copy of rgb_frame (BGR) for the viewer.
    [[nodiscard]] cv::Mat compose_pose_canvas(const cv::Mat& rgb_frame,
                                              const std::vector<PoseDetection>& poses) const;

private:
    Config config_;
    std::optional<YoloPoseDetector> detector_;
    std::vector<PoseDetection> last_poses_;   // cache of the latest non-empty detect_poses() result
    std::uint64_t last_poses_ts_ = 0;         // capture stamp of the cached poses (for the hold window)
};

}  // namespace rc::human_pose
