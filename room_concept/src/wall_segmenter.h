/*
 *  wall_segmenter.h — bottom-up extraction of straight wall segments from one 2-D LiDAR band.
 *
 *  WHY A NEW EXTRACTOR
 *  -------------------
 *  CornerDetector verifies corners the polygon already asserts: it gathers points around each
 *  PREDICTED wall line and splits them by the KNOWN wall directions. With no polygon there is
 *  nothing to predict, so shape estimation needs a detector that finds lines without a model.
 *
 *  ALGORITHM
 *  ---------
 *  Sequential RANSAC, as S-Graphs+ (Bavle et al., arXiv 2212.11770, §IV-A) does per keyframe for
 *  planes: the high band is a projection of ~10 LiDAR rings onto the floor plane, so the point set
 *  is not beam-ordered and clutter (a fridge, upper cabinets, an open door) interleaves with walls in
 *  bearing — which is what breaks an ordered split-and-merge. Two-point hypotheses, inliers by
 *  perpendicular residual against the sensor noise, the best hypothesis refit by PCA on its inliers,
 *  inliers removed, repeat while a supported line remains. Then a collinear merge: two segments fuse
 *  when their (φ, d) difference is not resolvable under their own combined covariance (the corner
 *  detector's merge_chi2 test). A doorway gap does NOT split a wall — walls are infinite lines with
 *  an extent, and a gap is missing data, not a contradiction.
 *
 *  Every segment carries a 2×2 information matrix on (φ, d) (see linefit::info_phi_d) so the
 *  consumer — birth, association, corner intersection — never has to guess how good it is.
 *
 *  THRESHOLDS (flagged, per CLAUDE.md)
 *  -----------------------------------
 *  Segmentation is a discrete decision, so it cannot be a precision. The two constants are χ²
 *  levels: the inlier test is χ²₁ on r²/σ² and the merge test is χ²₂ on the (φ, d) difference —
 *  the same nominal level the corner detector's association gate uses. The RANSAC iteration count
 *  is the standard confidence formula, not a metric bound.
 *
 *  SIGN CONVENTION
 *  ---------------
 *  The normal points INTO the room, i.e. towards the sensor: the origin (the robot) is on the free
 *  side of every wall it sees, so in the robot frame  n·0 − d = −d > 0  ⇔  d < 0. Enforced here,
 *  once; birth inherits it through the pose transform. The four Manhattan classes then encode axis
 *  AND side (a wall to the east and a wall to the west have opposite normals).
 */
#pragma once

#include <Eigen/Dense>
#include <optional>
#include <random>
#include <vector>

#include "line_fit.h"

namespace rc::wallseg
{
    struct Params
    {
        float sensor_sigma      = 0.02f;   // m — LiDAR range noise, perpendicular to the wall (physical)
        // Inlier band. Derived, not chosen: a point is an inlier of a line when the line explains it
        // better than clutter does. Clutter is uniform over the scan's area A (1/A per m²); a wall
        // explains a point as Gaussian across (σ) and uniform along its length L (1/L per m), so
        //     p_wall/p_clutter = A/(L√(2π)σ)·exp(−r²/2σ²) > 1  ⇔  r² < 2σ²·ln(A/(L√(2π)σ)).
        // A comes from the points' bounding box, L is bounded by its diagonal (conservative: a shorter
        // wall would widen the band). ≈3σ for a room; the old 95% χ² band (1.96σ) left the 2–4σ tail
        // of every wall behind, and RANSAC then fitted tilted "walls" through those leftovers.
        // chi2_inlier remains only as the fallback when the area is degenerate (fewer than a few
        // points) and for the corner endpoint test. ⚠ still a discrete decision, now model-derived.
        float chi2_inlier       = 3.841f;  // χ²₁ @95% — fallback band and the endpoint test
        float chi2_merge        = 5.991f;  // χ²₂ @95% — collinear merge on (φ,d) ⚠ threshold
        float ransac_confidence = 0.99f;   // p in k = log(1−p)/log(1−w²)
        int   ransac_max_iters  = 200;     // cap on the adaptive iteration count
        int   min_points        = 3;       // PCA needs ≥3 points for a residual DOF (resid_var)
        int   max_segments      = 16;      // stop extracting after this many (a room has few walls)
        // Fraction of the ORIGINAL point count below which extraction stops: with fewer remaining
        // points than min_points·2 no line can be supported anyway. Not a metric threshold.
        int   min_remaining     = 6;
    };

    struct WallSegment
    {
        Eigen::Vector2f normal;          // unit, points INTO the room (towards the sensor)
        float d = 0.f;                   // n·p = d  (d < 0 in the robot frame by convention)
        float phi = 0.f;                 // atan2(n_y, n_x)
        float resid_var = 0.f;           // per-point perpendicular variance: σ_sensor² + fit scatter (m²)
        int   npts = 0;
        Eigen::Vector2f p0, p1;          // extent endpoints, projected onto the line (p0 at s_min)
        float s_min = 0.f, s_max = 0.f;  // tangent coordinates t·p of the extent
        float gap0 = 0.f, gap1 = 0.f;    // along-wall spacing of the two outermost inliers at each end
        Eigen::Matrix2f info_phi_d;      // Λ on (φ, d), robot frame
        std::vector<int> inliers;        // indices into the input point set
        linefit::Line2D line() const
        {
            linefit::Line2D l; l.normal = normal; l.d = d; l.resid_var = resid_var; l.npts = npts; return l;
        }
        float sigma2() const;            // σ_sensor² + resid_var — the per-point noise this segment implies
    };

    /// Two segments observed MEETING: their intersection lies within each segment's extent up to the
    /// endpoint uncertainty. This is topology observed, not inferred — the pair is adjacent.
    struct ObservedCorner
    {
        int seg_a = -1, seg_b = -1;
        Eigen::Vector2f point;           // robot frame
        Eigen::Matrix2f information;     // Σ over the two walls of n nᵀ/σ² — rank-2 for a real corner
        float chi2_end = 0.f;            // worst endpoint Mahalanobis of the pair (display)
    };

    struct Result
    {
        std::vector<WallSegment>    segments;
        std::vector<ObservedCorner> corners;
        std::vector<int>            unexplained;   // point indices no segment claimed
        int ransac_iters = 0;                      // total hypotheses drawn (cost witness)
        float clutter_area = 0.f;                  // A — bbox area of the scan (m²), the clutter model
        float band = 0.f;                          // inlier band actually used (m)
    };

    /// Per-point log-evidence of "on this line" over "clutter": ln(A/(L√(2π)σ)) − r²/(2σ²). What a
    /// wall candidate earns per point it explains (wall_map birth), with the SAME clutter model as the
    /// inlier band above. L is the line's observed extent, σ² its per-point variance.
    float point_gain_nats(float r, float sigma2, float clutter_area, float extent);

    /// Extract wall segments from robot-frame 2-D points. `rng` is the caller's so the harness can be
    /// deterministic.
    Result segment(const std::vector<Eigen::Vector2f>& pts, const Params& p, std::mt19937& rng);

    /// Enforce the sign convention on a robot-frame line: normal towards the origin, d < 0.
    void orient_towards_origin(Eigen::Vector2f& normal, float& d);

    /// Corner between two segments, if their extents meet within endpoint uncertainty.
    std::optional<ObservedCorner> corner_of(const WallSegment& a, const WallSegment& b,
                                            int ia, int ib, const Params& p);
} // namespace rc::wallseg
