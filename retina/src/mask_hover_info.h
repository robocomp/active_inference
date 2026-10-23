#pragma once
// Immutable per-frame snapshot of the masks retina published, for the popup hover tooltip.
//
// Built by GraphPublisher at upload time from the very values it writes to masks_zed / masks_ricoh
// (not a graph read), handed to the popups as std::shared_ptr<const MaskHoverSnapshot>. Every cv::Mat
// inside is freshly allocated here (resize into new buffers), so it aliases nothing a worker or the
// publisher keeps drawing into. A value that does not exist for a source is std::nullopt and is
// printed as "—", never as 0.

#include <opencv2/core.hpp>
#include <QString>
#include <QStringList>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct SegDetection;

namespace rc
{

struct MaskHoverInfo
{
    int         index = 0;                 // position inside its DSR node's arrays
    std::string label, raw_label;          // raw_label empty = never normalised
    int         class_id = -1;
    float       confidence = 0.f;
    std::string second_label;              // empty = no runner-up
    float       second_confidence = 0.f;
    bool        accepted = true;
    bool        whitelist_active = false;
    bool        has_depth = false;
    int         pixel_area = 0;
    cv::Rect    bbox;                      // source px
    std::optional<std::array<float, 3>> centroid_cam;   // m
    std::optional<float> range_m;
    std::optional<int>   support_points;
    std::optional<float> depth_sigma_cm;   // sqrt(depth_var); nullopt = dense depth / none
    std::optional<float> azimuth_deg;
    std::optional<std::array<float, 3>> color_rgb;
    std::optional<float> color_neff;
    std::optional<float> motion_dotd, motion_bias, motion_sigma_cm, trunc_frac, centroid_radius;

    // Hit-test silhouette: the bbox crop, downscaled to <=128 px on its long side (CV_8U, 0/255).
    cv::Mat roi_mask;
};

struct MaskHoverSnapshot
{
    std::string   node;                    // "masks_zed" | "masks_ricoh"
    std::string   source;                  // "zed" | "ricoh"
    std::uint64_t frame_id = 0;
    std::uint64_t stamp_ms = 0;            // capture stamp
    std::int64_t  created_steady_ms = 0;   // for "age"
    cv::Size      src_size;                // mask coordinate space
    std::vector<MaskHoverInfo> masks;

    // Fills silhouette, area, bbox, label fields from a detection (empty mask -> hit-test off).
    static MaskHoverInfo from_detection(const SegDetection& d, int index, bool whitelist_active,
                                        bool accepted);
    static std::int64_t steady_now_ms();
    // Tooltip block for the masks under normalised image coords (rx, ry) in [0,1); empty = none.
    QString tooltip_at(double rx, double ry) const;
};

using MaskHoverSnapshotPtr = std::shared_ptr<const MaskHoverSnapshot>;

} // namespace rc
