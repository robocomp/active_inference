/*
 * detect_probe.h — the DETECTOR'S TRUTH TABLE: one row per cycle per live instance recording what the
 * detector was offered and what it did about it. Shared, header-only, one format for every agent.
 *
 * ★WHY IT EXISTS. `common/detectability` models P(detect | geometry) and its own header admits the
 * envelope is "a prior on the detector rather than a fact about it". Absence is charged in proportion to
 * that number, so when it is wrong the removal channel kills objects that are there — the mirror image
 * of a hallucination, and the same defect: a sensor model whose two constants are CONFIGURED rather than
 * MEASURED. p_clutter too low invents objects; p_detect too high deletes them, and both terms sit in the
 * same log-ratio, so one calibration fixes both directions.
 *
 * ★WHAT RUNNING fit_envelope ON THE EXISTING LOGS SHOWED (2026-08-16, and why this file is not just more
 * of the same columns):
 *   · hood       — 9432 usable rows, 6903 of them in ONE fill bin. Hessian not positive-definite,
 *                  max_fill fitted to 9.2e6. Not a format problem: the tour never varied the framing.
 *   · fridge     — 13785 rows across 7 bins, and the empirical rate is NOT unimodal and not even
 *                  monotonic: 0.481, 0.595, 0.787, 0.440, 0.158, **0.813**, 0.029. No function of fill
 *                  alone can produce that. ⇒ FILL IS NOT THE DOMINANT COVARIATE, so calibrating the
 *                  existing three parameters cannot be the cure; the model is missing a variable.
 *   · the label  — the tool's default `det_alive` is a LATCHED liveness flag, 1 in 9432/9432 rows
 *                  (hood's last row: det_alive=1 with frames_since_det=6172). Only `--label fresh`
 *                  (frames_since_det<=0) is a per-cycle outcome. A flag you would never guess was
 *                  silently making a fittable dataset look degenerate.
 *
 * ★WHY A SEPARATE FILE AND NOT MORE ai2_log COLUMNS. cabinet_fitter.cpp carries the scar: its ai2 header
 * had drifted out of step with its body, so "every column past cy parsed under the wrong name". Four
 * more agents appending to four hand-maintained header/body pairs is that bug waiting to happen. This
 * file has ONE header, ONE writer, and no per-agent formatting.
 *
 * ★WHAT EACH NEW COLUMN IS FOR — every one answers a specific question the existing logs cannot:
 *   rx,ry,rtheta,cam_z  VIEWPOINT IDENTITY. Absence is currently integrated as if each frame were an
 *                  independent trial. It is not: a miss from a pose repeats at 20 Hz from the same
 *                  pose, so a stationary robot manufactures ~100 "independent" refutations of one look.
 *                  This is the same error Woodbury common-mode marginalisation already fixes for mask
 *                  POINTS, one channel over. Without the pose you cannot cluster frames into distinct
 *                  viewpoints and cannot even TEST it — which is why this is the first column here.
 *   ocx,ocy,ocz    the object's believed centre. With the pose above, elevation / azimuth / parallax
 *                  are all DERIVED offline rather than logged: fewer columns, and no way for a stored
 *                  angle to disagree with the geometry it came from.
 *   conf           the score of the slice that fired. Detection was binary in the logs, so a mask that
 *                  barely made it read identically to a confident one.
 *   other_label    ★SIBLING CONFUSION. If the detector fired on a CONFUSABLE class at the right place,
 *   other_dist     the object was SEEN and misnamed — the worst possible thing to score as absence,
 *   other_conf     because the detector is actually agreeing with us. Measured this session: the 360
 *                  calls things `microwave` 1.3 m from the hood, and `cabinet`/`hood` are genuinely
 *                  confusable classes for ADE20K.
 *
 * NOT HERE, ON PURPOSE: sub-threshold detections and drop-stage attribution. An agent cannot see either
 * — everything below YOLO's conf floor, and everything dropped by min_area / YOLO-priority / the accept
 * filter, is gone before the masks node is written. Those live in the retina's own probe.
 *
 * Locale: writes through rc::diag::open_rotating, which imbues the classic locale (CLAUDE.md).
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <string_view>

#include <Eigen/Dense>

#include "../diag_log/rotating_csv.h"
#include "../mask_ingestor/mask_ingestor.h"

namespace rc::probe
{

// The nearest ZED slice carrying a DIFFERENT label than the one this agent fits.
//
// ★ZED ONLY, and that is not an optimisation. A ricoh slice's centroid comes from reprojected LiDAR
// inside the mask, which cannot see a high or thin surface at all: measured 2026-08-16, the ricoh's
// `microwave` centroid sat 1.31 m (median, n=296) from the ZED's `hood` centroid for what is very
// likely the same object. A distance computed from that would answer a different question.
struct SiblingHit
{
    std::string label;          // empty ⇒ nothing else was detected this frame
    float       dist_m    = -1.0f;
    float       confidence = 0.0f;
};

inline SiblingHit nearest_other_label(const MaskIngestor::MasksPacket& pkt,
                                      std::string_view                 my_label,
                                      const Eigen::Vector2f&           target_xy)
{
    SiblingHit best;
    if (not pkt.valid)
        return best;
    for (const auto& sl : pkt.slices)
    {
        if (sl.label == my_label or not sl.is_zed() or not sl.has_depth)
            continue;
        const Eigen::Vector2f c{sl.centroid.x(), sl.centroid.y()};
        if (not c.allFinite())
            continue;
        const float d = (c - target_xy).norm();
        if (best.label.empty() or d < best.dist_m)
            best = SiblingHit{sl.label, d, sl.confidence};
    }
    return best;
}

// One cycle, one instance. Every field is what the agent already knows; nothing is recomputed here.
struct Sample
{
    long          cycle    = 0;
    std::string   node;                  // instance node name, so rows can be grouped per object
    std::uint64_t stamp_ms = 0;          // the mask packet's capture stamp (joins to other logs)

    // ── VIEWPOINT (the new axis) ──
    float rx = 0.0f, ry = 0.0f, rtheta = 0.0f;   // robot pose in the room frame
    float cam_z = 0.0f;                          // camera height in the room frame

    // ── the object's believed centre, so angles are derived, not stored ──
    float ocx = 0.0f, ocy = 0.0f, ocz = 0.0f;

    // ── covariates the agents already compute (same values their ai2 rows carry) ──
    float range_m       = 0.0f;
    float roi_fill      = 0.0f;
    float roi_fill_h    = 0.0f;
    float roi_fill_v    = 0.0f;
    bool  roi_valid     = false;
    float obliquity_cos = 0.0f;
    float trunc_frac    = 0.0f;
    float p_detect      = 0.0f;   // what the CURRENT envelope predicted — the thing being tested
    // ★THE PRIOR, NOT THE POSTERIOR — and the column is named for it. log_detect_probe() runs BEFORE
    // this cycle's existence update (deliberately: an instance killed this cycle must still leave the row
    // recording the look that killed it), so this is the belief as it stood BEFORE the outcome in the same
    // row was integrated. Discovered 2026-08-17 by an audit that showed ex_p apparently FALLING on cycles
    // logged as detected, which the ai2 log said never happens — the two columns were one cycle apart.
    // ★It is also the CORRECT weight: weighting a trial by the posterior that this very trial produced is
    // circular. So the offset is right and only the old name (`ex_p`) was wrong.
    float p_exists_prior = 0.0f;

    // ── the outcome ──
    bool  fired = false;          // a slice of THIS agent's label was assigned this cycle
    float conf  = 0.0f;           // that slice's score (0 when nothing fired)
    int   n_my  = 0;              // slices of my label anywhere in the frame
    int   n_all = 0;              // slices of any label
    SiblingHit sibling;
};

inline std::string_view header()
{
    return "cycle,node,stamp_ms,rx,ry,rtheta,cam_z,ocx,ocy,ocz,range,"
           "roi_fill,roi_fill_h,roi_fill_v,roi_valid,obliquity_cos,trunc_frac,p_detect,ex_p_prior,"
           "fired,conf,n_my,n_all,other_label,other_dist,other_conf\n";
}

// Open on first use; a path that cannot be opened disables itself by clearing `path`.
inline bool ensure_open(std::ofstream& out, std::string& path)
{
    if (path.empty())
        return false;
    if (out.is_open())
        return true;
    if (not rc::diag::open_rotating(out, path, header()))
    {
        path.clear();
        return false;
    }
    return true;
}

inline void append(std::ofstream& out, const Sample& s)
{
    if (not out.is_open())
        return;
    out << s.cycle << ',' << s.node << ',' << s.stamp_ms << ','
        << s.rx << ',' << s.ry << ',' << s.rtheta << ',' << s.cam_z << ','
        << s.ocx << ',' << s.ocy << ',' << s.ocz << ',' << s.range_m << ','
        << s.roi_fill << ',' << s.roi_fill_h << ',' << s.roi_fill_v << ','
        << (s.roi_valid ? 1 : 0) << ',' << s.obliquity_cos << ',' << s.trunc_frac << ','
        << s.p_detect << ',' << s.p_exists_prior << ','
        << (s.fired ? 1 : 0) << ',' << s.conf << ',' << s.n_my << ',' << s.n_all << ','
        // an empty label is written as "-" so the column is never zero-width (a bare ",," is the one
        // thing a naive splitter gets wrong, and this file exists to be read by scripts)
        << (s.sibling.label.empty() ? "-" : s.sibling.label) << ','
        << s.sibling.dist_m << ',' << s.sibling.confidence << '\n';
    out.flush();
}

}   // namespace rc::probe
