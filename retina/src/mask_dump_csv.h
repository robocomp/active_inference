/*
 * mask_dump_csv.h — the ZED masks of one dumped frame, as plain CSV, for offline figures.
 *
 * Companion of zed_frame_dump.h ([ZedDump] dump_masks = true). For every frame the ZedDump KEEPS
 * (same keep decision, same capture stamp), retina also writes what it published on `masks_zed` for
 * that stamp, so <dir>/<stamp>.jpg pairs 1:1 with the files below. Diagnostic only: nothing reads
 * these files at run time and nothing here is published to DSR.
 *
 * LAYOUT — three files per frame, all with a header row, comma-separated, decimal POINT always
 * (written through std::format + imbue(std::locale::classic()); read with std::from_chars):
 *
 *   <stamp>.masks.csv       one row per published mask (the index; written LAST, so its presence
 *                           means the other two are complete)
 *       stamp_ms      capture stamp (= mask_timestamp_ms on masks_zed, = the .jpg name)
 *       frame_id      mask_frame_id of that publish
 *       mask          0-based index into the node's arrays (mask_label_ids order)
 *       label         published label ('|' and ',' and newlines replaced by ' ')
 *       label_id      published class id (semantic masks carry 1000 + ADE20K id)
 *       confidence    published confidence
 *       segmenter     yolo | semantic   (retina-internal tag, see mask_segmenter.h; NOT on DSR)
 *       n_pixels      silhouette pixels for this mask in .pix.csv
 *       n_points      support points for this mask in .pts.csv
 *       pixel_stride  decimation stride of the silhouette (VOXEL_DECIMATION_FACTOR): each pixel row
 *                     stands for a stride x stride block of the image
 *       img_w,img_h   ZED image size the pixels are expressed in
 *
 *   <stamp>.masks.pix.csv   long form of mask_pixels_xy / mask_pixel_offsets
 *       mask,col,row        image pixel (decimated foreground, BEFORE the depth gate)
 *
 *   <stamp>.masks.pts.csv   long form of mask_support_points_cam / mask_support_offsets
 *       mask,x,y,z          metres, ZED camera frame (x right, y depth, z up) — before any pose
 *
 * A kept frame whose publish carried no masks still gets a header-only <stamp>.masks.csv, so
 * "no masks" is distinguishable from "not dumped".
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "mask_segmenter.h"

namespace rc::diag
{

struct MaskDumpEntry
{
    std::string        label;
    int                label_id   = -1;
    float              confidence = 0.f;
    rc::MaskSegmenter  segmenter  = rc::MaskSegmenter::yolo;
    std::vector<float> pixels_xy;    // flat (col,row) pairs
    std::vector<float> points_cam;   // flat (x,y,z) triples, ZED camera frame (m)
};

struct MaskDumpRecord
{
    std::uint64_t stamp_ms     = 0;
    int           frame_id     = 0;
    int           img_w        = 0;
    int           img_h        = 0;
    int           pixel_stride = 1;
    std::vector<MaskDumpEntry> masks;
};

// "<dir>/<stamp>.masks<suffix>.csv" — suffix "" (index), ".pix" or ".pts".
std::string mask_dump_path(const std::string& dir, std::uint64_t stamp, const char* suffix = "");

// Writes the three files. Returns false (and logs) if any could not be written.
bool write_mask_dump(const std::string& dir, const MaskDumpRecord& rec);

// Reads them back (used by the unit test; locale-independent). nullopt if the index is missing or
// malformed.
std::optional<MaskDumpRecord> read_mask_dump(const std::string& dir, std::uint64_t stamp);

} // namespace rc::diag
