#pragma once

// Which segmenter produced a ZED mask. Retina-internal provenance: the masks node carries YOLO-seg and
// ADE20K-semantic-derived masks side by side with no per-mask field telling them apart (only the
// 1000+ class_id offset the semantic stage uses, which is a convention, not a contract). This tag is
// set where each mask is assembled (SegStage = yolo by default, SemanticMaskStage = semantic) and read
// ONLY by the diagnostic mask dump (mask_dump_csv.h). It is deliberately NOT published to DSR.

#include <cstdint>
#include <string_view>

namespace rc
{

enum class MaskSegmenter : std::uint8_t
{
    yolo     = 0,   // YOLO-seg instance mask (seg_stage)
    semantic = 1,   // connected component of the ADE20K class field (semantic_mask_stage)
};

constexpr std::string_view to_string(MaskSegmenter s) noexcept
{
    return s == MaskSegmenter::semantic ? "semantic" : "yolo";
}

} // namespace rc
