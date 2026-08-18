/*
 * common/peripheral_channel/peripheral_channel.h
 *
 * THE ONE peripheral (ricoh-360) consumption path, shared by every concept agent.
 *
 * ★WHY THIS EXISTS: three different behaviours had grown on ONE channel.
 *   table / cabinet / refrigerator / hood : a 94-line near-copy of process_ricoh_bearings in each,
 *                                          reading DEPTH-FILLED slices, attention targets only.
 *   bottle                                : the old common/bearing_confirm on BEARING-ONLY slices, confirm only.
 *   chair / door                          : the same, PLUS birthing an instance from an unmatched
 *                                          bearing at a guessed nominal range.
 * Same sensor, same question, three answers — and the divergence was invisible because the four copies
 * carried a comment naming a module (bearing_confirm) that none of them called. Fix one, audit all.
 * ★That module is DELETED as of 2026-08-17 — the readiness sweep found it had no caller left at all,
 *   only three vestigial #includes and comments pointing at it. This file is the whole channel now.
 *
 * ★WHICH SLICE KIND ARRIVES IS A SENSOR ACCIDENT, NOT A POLICY. A ricoh detection carries 3-D points
 * when the LiDAR depth-fill found returns inside its mask, and only an azimuth when it did not. That is
 * a property of where the LiDAR happened to sweep, so an agent must not have DIFFERENT behaviour for the
 * two. Both normalise to one Detection here: a bearing, plus a range when there happens to be one.
 *
 * ★WHAT A PERIPHERAL SLICE MAY DO: confirm that a live instance is still there, or raise an unmatched
 * detection as something to go and look at. It may NOT create or move geometry. Bearing-birth was
 * removed (2026-08-15) rather than spread: it created an object at a GUESSED range from the sensor least
 * able to measure range, in the two agents that happened to have it. Replacing it properly is the
 * proto-object / saccadic-affordance path — raise a candidate, go and look, THEN birth with real range.
 *
 * FRAME: azimuths are ROOM-frame today, matching the producer's mask_azimuth. If that becomes
 * robot-frame (see the retina's Phase-2 notes), only predicted_azimuth() below has to change.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string_view>
#include <vector>

#include <Eigen/Dense>

#include "../mask_ingestor/mask_ingestor.h"

namespace rc::peripheral
{

// One peripheral detection, normalised from either slice kind.
struct Detection
{
    float azimuth_rad = 0.0f;   // robot→object bearing
    float range_m     = 0.0f;   // meaningful only when has_range; a ROUGH, biased indicative value
    bool  has_range   = false;
    float confidence  = 0.0f;
    int   slice_index = -1;
    Eigen::Vector2f xy = Eigen::Vector2f::Zero();   // world point when has_range, else zero
};

// A live instance to associate against.
struct TrackRef
{
    std::uint64_t   id        = 0;
    Eigen::Vector2f xy        = Eigen::Vector2f::Zero();
    float           radius_m  = 0.0f;   // circumscribed footprint radius; sets the ANGULAR tolerance
};

// A detection that lined up with a live instance: peripheral evidence it still exists.
struct Confirm
{
    std::uint64_t track_id       = 0;
    int           slice_index    = -1;
    float         confidence     = 0.0f;
    float         innovation_rad = 0.0f;
};

// A detection that matched nothing — a candidate for the go-and-check path.
struct AttentionTarget
{
    float           azimuth_rad = 0.0f;
    float           range_m     = 0.0f;
    float           confidence  = 0.0f;
    Eigen::Vector2f xy          = Eigen::Vector2f::Zero();
};

struct Result
{
    std::vector<Confirm>         confirms;
    std::vector<AttentionTarget> attention;
};

struct Params
{
    // Extra angular slack ON TOP of the instance's own angular half-size. The tolerance is derived from
    // geometry — atan2(radius, distance) — so a near object is allowed a wide bearing error and a far one
    // a narrow one automatically. This is the only hand-set part.
    float angular_margin_rad = 0.10f;
    // Range slack, used ONLY when the detection carries a range. GENEROUS on purpose: the ricoh range is
    // a rough median over sparse reprojected LiDAR hits, biased and indicative. Its job is not precision,
    // it is to stop a NEW object hiding along the same bearing as a known one from being swallowed by it.
    float range_band_m = 1.0f;
    float min_confidence = 0.0f;
};

inline float wrap_angle(float a) { return std::remainder(a, 2.0f * static_cast<float>(M_PI)); }

// Associate peripheral detections against live instances.
// Matched  → Confirm      (the caller integrates a CONFIRM-ONLY existence delta; see rc::exist)
// Unmatched→ AttentionTarget
//
// A detection must agree in BOTH bearing and (when it has one) range. Bearing alone would collapse a new
// object onto a known one sitting at the same azimuth but a different distance — precisely the case the
// peripheral channel exists to surface.
inline Result associate(const std::vector<TrackRef>&  tracks,
                        const std::vector<Detection>& dets,
                        const Eigen::Vector2f&        robot_xy,
                        const Params&                 p = Params{})
{
    Result out;
    for (const auto& d : dets)
    {
        if (d.confidence < p.min_confidence)
            continue;
        bool matched = false;
        for (const auto& t : tracks)
        {
            const Eigen::Vector2f v = t.xy - robot_xy;
            const float dist = v.norm();
            if (dist < 1e-3f) { matched = true; break; }   // robot on top of it ⇒ bearing undefined
            const float tb  = std::atan2(v.y(), v.x());
            const float tol = std::atan2(std::max(t.radius_m, 0.0f), dist) + p.angular_margin_rad;
            if (std::abs(wrap_angle(d.azimuth_rad - tb)) > tol)
                continue;
            if (d.has_range and std::abs(d.range_m - dist) > t.radius_m + p.range_band_m)
                continue;   // same bearing, different distance ⇒ NOT this one
            out.confirms.push_back({t.id, d.slice_index, d.confidence,
                                    wrap_angle(d.azimuth_rad - tb)});
            matched = true;
            break;
        }
        if (not matched)
            out.attention.push_back({d.azimuth_rad, d.range_m, d.confidence, d.xy});
    }
    return out;
}

// Normalise this frame's ricoh slices of `label` into Detections.
//
// ★ACCEPTS BOTH SLICE KINDS. A depth-filled slice (support points present) yields a bearing AND a rough
// range from the ROBUST MEDIAN of its points — median, not centroid, because a reprojected-LiDAR blob
// has outliers and a mean chases them. A bearing-only slice yields azimuth alone. Which one arrives
// depends on where the LiDAR swept, not on anything the agent decided, so both land here.
//
// ★RICOH ONLY, AND THAT IS THE POINT. is_ricoh() is asked directly rather than inferred from has_depth:
// once the producer began depth-filling ricoh masks it published them with has_depth = 1, so every guard
// written as `if (has_depth)` to mean "from the ZED" silently began accepting 360° detections. Ask the
// source.
inline std::vector<Detection> gather(const MaskIngestor::MasksPacket& pkt,
                                     std::string_view                 label,
                                     const Eigen::Vector2f&           robot_xy,
                                     float                            min_confidence = 0.0f)
{
    std::vector<Detection> out;
    if (not pkt.valid)
        return out;
    for (std::size_t i = 0; i < pkt.slices.size(); ++i)
    {
        const auto& sl = pkt.slices[i];
        if (sl.label != label or not sl.is_ricoh() or sl.confidence < min_confidence)
            continue;

        Detection d;
        d.slice_index = static_cast<int>(i);
        d.confidence  = sl.confidence;

        const std::size_t b = std::min(sl.support_begin, pkt.support_points.size());
        const std::size_t e = std::min(sl.support_end,   pkt.support_points.size());
        if (e > b)
        {
            std::vector<float> xs, ys;
            xs.reserve(e - b); ys.reserve(e - b);
            for (std::size_t k = b; k < e; ++k)
            { xs.push_back(pkt.support_points[k].x()); ys.push_back(pkt.support_points[k].y()); }
            const std::size_t m = xs.size() / 2;
            std::nth_element(xs.begin(), xs.begin() + m, xs.end());
            std::nth_element(ys.begin(), ys.begin() + m, ys.end());
            d.xy        = {xs[m], ys[m]};
            const Eigen::Vector2f v = d.xy - robot_xy;
            d.azimuth_rad = std::atan2(v.y(), v.x());
            d.range_m     = v.norm();
            d.has_range   = true;
        }
        else if (not sl.has_depth)
        {
            d.azimuth_rad = sl.azimuth_room_rad;   // bearing-only slice: no points, no range
            d.has_range   = false;
        }
        else
            continue;   // claims depth but carries no points — nothing usable
        out.push_back(d);
    }
    return out;
}

// ─── the whole per-cycle channel, in one call ─────────────────────────────────────────────────────────────
//
// gather → associate → hand each confirm to the agent → return what is left unassigned. Four agents ran
// exactly this sequence around the two calls above and differed only in how they BUILD the track list (a
// cabinet in kitchen mode associates against wall CELLS, not fitted instances, and its radius comes from
// L/d where the box-shaped agents use w/h) and in where a confirm goes.
//
// ★A CONFIRM IS EVIDENCE, A MISS IS NOT. `on_confirm` is called per match and the agent integrates it
// confirm-only (e_free hard 0), because the 360 detector's p_detect at a given range is UNCHARACTERISED and
// absence weighted by an unknown p_detect is the ratchet that has bitten this fleet before. There is
// deliberately no on_miss hook — the shape of the API is the argument.
//
// ★BOTH COUNTS COME BACK. Zero attention with non-zero dets means the channel is WORKING and everything it
// saw matched — the opposite conclusion from zero of both, and a single number cannot tell them apart.
struct CycleParams
{
    float detect_conf        = 0.0f;   // confidence floor for gather()
    float angular_margin_rad = 0.0f;
    float range_band_m       = 0.0f;
};

struct CycleOut
{
    std::vector<AttentionTarget> attention;    // unassigned bearings — "go and check" candidates
    std::size_t n_dets = 0, n_tracks = 0, n_confirms = 0;
};

template <class OnConfirm>
inline CycleOut run_cycle(const MaskIngestor::MasksPacket& pkt, std::string_view class_name,
                          const Eigen::Vector2f& robot_xy, const std::vector<TrackRef>& tracks,
                          const CycleParams& cp, OnConfirm&& on_confirm)
{
    const auto dets = gather(pkt, class_name, robot_xy, cp.detect_conf);
    Params pp;
    pp.angular_margin_rad = cp.angular_margin_rad;
    pp.range_band_m       = cp.range_band_m;
    const auto res = associate(tracks, dets, robot_xy, pp);
    for (const auto& cf : res.confirms)
        on_confirm(cf);
    return {res.attention, dets.size(), tracks.size(), res.confirms.size()};
}

}   // namespace rc::peripheral
