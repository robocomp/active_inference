#pragma once

#include <vector>
#include <optional>
#include <Eigen/Dense>

namespace rc {

/// Detects room corners in lidar scans using model-guided partitioning.
///
/// Pipeline:
///   1. Project model corners (world frame) into robot frame.
///   2. For each predicted corner, gather lidar points within a search radius.
///   3. Partition the neighbourhood into two groups using the known wall
///      directions from the polygon model.
///   4. Fit a line (PCA) to each group and intersect.
///   5. Accept the detection if it passes angle and distance quality gates.
class CornerDetector
{
public:
    // ===== Configuration =====
    struct Params
    {
        float search_radius       = 1.5f;   // meters around predicted corner to gather points
        int   min_points_per_line = 3;       // minimum points per wall group
        float ransac_threshold    = 0.06f;   // inlier band width for optional outlier rejection
        float max_match_distance  = 1.5f;   // max distance between detected and predicted corner (association cap)
        float min_corner_angle    = 25.0f;   // degrees — kept for MODEL-corner selection only (set_model_corners)
        float max_corner_angle    = 155.0f;  // degrees — kept for MODEL-corner selection only (set_model_corners)
        float max_orientation_dev = 20.0f;   // degrees — LEGACY hard gate (unused now; kept for config back-compat)

        // ── Graded-covariance detection (replaces the old hard rej_angle/rej_orient gates) ──
        // A detection is no longer discarded when its geometry is marginal; instead it carries a
        // per-detection 2×2 information matrix Λ_det (robot frame) whose precision shrinks smoothly
        // with wall-fit scatter, orientation deviation, and intersection shallowness. The RFE loss
        // consumes Λ_det anisotropically, so a marginal corner contributes weakly and a clean
        // asymmetric corner (the notch) contributes strongly — no thresholds, covariance → SDF pose.
        float wall_band     = 0.35f;   // meters — perpendicular tolerance gathering wall points around the
                                       // PREDICTED wall line. Must exceed the chronic model misfit (~0.32 m
                                       // here) or the true wall points fall outside the band and NO detection
                                       // forms (the real reason corners almost never fired). Was hardcoded 0.12.
        float base_sigma    = 0.04f;   // meters — corner detection noise floor σ0 (per-wall).
        float orient_tau_deg = 20.0f;  // degrees — smooth orientation-trust scale: ori_scale = exp(−(dev/τ)²).
                                       // dev=τ → 37% weight, 2τ → 2%. Replaces the 20° hard cut.
    };

    // ===== Output types =====

    struct CornerMatch
    {
        int    model_index;         // index into the ORIGINAL polygon vertex list
        Eigen::Vector2f detected;   // detected position (robot frame, meters)
        Eigen::Vector2f predicted;  // predicted position (robot frame, meters)
        Eigen::Vector2f model_world;// model corner world position (for display)
        float  distance;            // ||detected - predicted||
        float  angle_deg;           // angle between the two fitted lines
        Eigen::Matrix2f covariance; // 2×2 detection uncertainty (robot frame) — legacy, display only
        Eigen::Matrix2f information;// 2×2 graded precision Λ_det = Σ_L (ori_scale_L/σ_L²) n_L n_Lᵀ (robot frame).
                                    // Rank-1 when the two walls are near-parallel (shallow corner → the
                                    // bisector direction is left unconstrained). This is what the loss uses.
    };

    struct DetectionResult
    {
        std::vector<CornerMatch> matches;
        int corners_in_fov = 0;
        int corners_detected = 0;
        int corners_accepted = 0;
        // Diagnostic counts — where do in-FOV corners die? With graded covariance most of these are no
        // longer hard rejections but "soft" events kept for observability.
        int rej_occluded = 0;  // ★ model corner NOT visible from the robot (a wall/notch occludes the sight
                               // line) → excluded BEFORE detection: an unreachable corner must not be matched,
                               // must not enter the loss, and must not count toward the early-exit decision.
        int rej_fewpoints = 0; // ★ FORMATION failure: gather grabbed < min_points_per_line on a wall → no
                               // detection formed at all (fires BEFORE the gates). If this dominates, the
                               // wall_band is too tight vs the model misfit — widen it. This is the counter
                               // added to confirm the gather-band hypothesis.
        int rej_dist = 0;      // detected corner > max_match_distance from prediction (association cap, kept)
        int soft_orient = 0;   // orientation trust ori_scale < 0.05 (heavily downweighted, NOT discarded)
        int rej_convex = 0;    // convexity sign mismatch — KEPT as a hard gate (topological disambiguator
                               // for rot180: |dir·model_dir| is 180°-blind, only convexity breaks the tie)
        int rej_unassigned = 0;// survived to candidate but lost the 1-to-1 Hungarian assignment
    };

    // ===== Interface =====

    explicit CornerDetector() = default;
    explicit CornerDetector(const Params& p) : params_(p) {}

    void set_model_corners(const std::vector<Eigen::Vector2f>& polygon_vertices);

    DetectionResult detect(const std::vector<Eigen::Vector3f>& lidar_points,
                           float robot_x, float robot_y, float robot_theta,
                           float max_range = 15.0f) const;

    Params& params() { return params_; }
    const Params& params() const { return params_; }

private:
    Params params_;

    /// Full room polygon (world frame) — retained for the ray-cast occlusion/visibility test so an
    /// occluded corner (behind a wall or the notch step) is excluded before detection.
    std::vector<Eigen::Vector2f> polygon_;

    /// Model corner with its two adjacent wall directions.
    struct ModelCorner
    {
        Eigen::Vector2f position;       // world frame
        Eigen::Vector2f edge_in_dir;    // unit direction of wall arriving at this corner
        Eigen::Vector2f edge_out_dir;   // unit direction of wall leaving this corner
        float convexity_sign;           // sign of edge_in × edge_out (positive = CCW turn)
        float wall_in_length;           // length of the incoming wall (prev→curr)
        float wall_out_length;          // length of the outgoing wall (curr→next)
        int original_index;             // index in original polygon
    };
    std::vector<ModelCorner> model_corners_;

    /// 2D line: normal · p = d   (normal is unit length)
    struct Line2D
    {
        Eigen::Vector2f normal;
        float d;
        float resid_var = 0.f;   // λ_min / N — mean squared perpendicular scatter of the fitted points
                                 // about the line (≈ sensor noise for a clean wall; large for a cluttered
                                 // gather). Inflates σ_L so a poorly-fit wall is trusted less.
        int   npts = 0;          // number of points the line was fit to
        Eigen::Vector2f direction() const { return Eigen::Vector2f(-normal.y(), normal.x()); }
    };

    /// PCA line fit — returns nullopt if fewer than min_points.
    static std::optional<Line2D> fit_line_pca(const std::vector<Eigen::Vector2f>& pts,
                                               int min_points);

    /// Intersect two lines.  Returns nullopt if (nearly) parallel.
    static std::optional<Eigen::Vector2f> intersect(const Line2D& a, const Line2D& b,
                                                     float* angle_deg = nullptr);
};

} // namespace rc
