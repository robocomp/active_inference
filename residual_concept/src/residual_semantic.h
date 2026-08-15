/*
 * residual_semantic.h  —  RGB-semantic floor down-weighting for the residual occupancy grid.
 *
 * A SECOND, uncorrelated cue against the grid's FLOOR PHANTOMS. The geometric floor rejection
 * (nav-band + subtract_infrastructure + bpearl band) fails on noisy ZED stereo returns whose range²
 * depth error lifts a floor point just above the nav band → a phantom obstacle. Appearance fails
 * DIFFERENTLY from geometry, so fusing an RGB semantic label sharpens the floor/obstacle decision.
 *
 * Source = the retina's dense YOLO-sem label map (ADE20K SceneParse150), published on the DSR
 * `semantic` node under `zed`, pixel-aligned at the ZED image resolution. Because depth and the
 * label map are the SAME camera, a room-frame ZED point reprojects EXACTLY back to its own pixel
 * (reproject-through-current-pose recovers the original u,v — no parallax, no feature match), so we
 * sample its class id for free and never store a label in world space.
 *
 * ★SAFETY (non-negotiable): this only ever DOWN-WEIGHTS, never discards, and is HEIGHT-GATED. The one
 * unacceptable failure is a REAL obstacle mislabelled "floor" → suppressed → missed. So a floor label's
 * suppressive authority DECAYS SMOOTHLY with the point's height above the floor band (a clearly-elevated
 * return keeps full weight regardless of label — height is decisive), and a stale label (the map refreshes
 * at ~2 Hz) loses authority with its AGE. No hard thresholds: the weight is a continuous
 * P(obstacle | label, height, freshness) ∈ [0,1] multiplied into the point's grid HIT weight.
 *
 * Pure Eigen/STL, DSR-free → unit-testable in isolation (semantic_self_test()).
 */

#pragma once

#include <cstdint>
#include <vector>
#include <Eigen/Dense>

#include "residual_zed_boost.h"   // rc::CamIntrinsics (y-depth zed pinhole)

namespace rc
{

// A snapshot of the latest ZED semantic label map read from the DSR `semantic` node. `labels` are ADE20K-150
// class ids, row-major at (width×height) = the ZED image resolution. stamp_ms = the map's capture time (used
// for freshness). frame_id lets a consumer skip re-copying the blob when it has not changed.
struct SemanticMap
{
    std::vector<std::uint8_t> labels;
    int           width    = 0;
    int           height   = 0;
    std::uint64_t stamp_ms = 0;
    int           frame_id = -1;
    bool valid() const
    { return width > 0 and height > 0 and static_cast<int>(labels.size()) >= width * height; }
};

struct SemanticFloorParams
{
    bool  enabled          = false;   // master flag (config Semantic.DownweightFloor). false ⇒ weights are all 1.
    float floor_suppress   = 0.60f;   // max hit-weight reduction for a floor-class NEAR-floor point (weight = 1−this)
    float height_scale_m   = 0.15f;   // a floor label's authority decays as exp(−(h/scale)²) over height h above the
                                      //   floor band → a clearly-elevated obstacle is never suppressed (safety gate)
    float fresh_half_life_s = 0.50f;  // suppression strength halves per this much map age (freshness-as-precision).
                                      //   0 ⇒ no age decay. The label map refreshes at ~2 Hz (age up to ~0.5 s).
    float floor_z0    = 0.06f;        // floor-band height at the sensor (mirror the grid's nav band)
    float floor_slope = 0.04f;        // + per metre of horizontal range (grazing)
    // ADE20K-150 class ids treated as WALKABLE floor/ground (down-weight candidates). The ground/soil concept is
    // SPLIT across several ADE20K classes, so all synonyms must be listed or the segmenter's choice among them
    // leaks a floor phantom. Indoor + outdoor walkable surfaces:
    //   3 floor · 6 road · 9 grass · 11 sidewalk · 13 earth/ground · 28 rug · 29 field · 46 sand · 52 path
    //   · 54 runway · 91 dirt track · 94 land/soil/ground
    // Deliberately EXCLUDED (hazards / not floor-flat): 53 stairs, 59 stairway, 121 step, 16 mountain, 68 hill,
    // and all water classes. Everything not in this set (incl. 255 ignore) is an obstacle class → weight 1.
    std::vector<std::uint8_t> floor_class_ids{3, 6, 9, 11, 13, 28, 29, 46, 52, 54, 91, 94};
};

// For each room-frame ZED point, return P(obstacle | label, height, freshness) ∈ [0,1] — a MULTIPLICATIVE
// down-weight for that point's grid HIT (misses/clearing are untouched: clearing free space is always safe).
// Result is parallel to `points_room`; all 1.0 (a no-op) when disabled or the map is invalid. `cam_T_room`
// maps room→camera; K + (img_w,img_h) are the ZED depth image the points came from (scaled to the label map).
// `age_s` = (now − map.stamp_ms) in seconds. `origin_room` = the sensor position (for horizontal range).
std::vector<float> semantic_obstacle_weights(const std::vector<Eigen::Vector3f>& points_room,
                                             const Eigen::Matrix4f& cam_T_room, const CamIntrinsics& K,
                                             int img_w, int img_h,
                                             const SemanticMap& sem, const Eigen::Vector3f& origin_room,
                                             float age_s, const SemanticFloorParams& p);

bool semantic_self_test();

}  // namespace rc
