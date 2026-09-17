/*
 *  reloc_search.h — global relocalisation as a MIXTURE POSTERIOR over SE(2).
 *
 *  WHY THIS REPLACES THE 4-STAGE LATTICE
 *  -------------------------------------
 *  grid_search_initial_pose() scored one pose per torch dispatch (159 µs, 3643 evals = 0.58 s, measured
 *  2026-08-29), on MEDIAN |SDF|. The median is not a sum over points, so it can neither be batched as a
 *  gather-and-sum, nor bounded, nor read as a likelihood — which is why its softmax needed a hand-derived
 *  temperature. It returned ONE pose, and its success paths committed a fixed Identity()*0.1 covariance.
 *
 *  THE GENERATIVE MODEL
 *  --------------------
 *  Each wall-band point, placed in the room by pose x, lands at unsigned distance d_i(x) from the room
 *  polygon. It is either a wall return — |d| half-normal with the SDF observation noise σ — or clutter /
 *  a return through an open door — |d| uniform over the room's extent L. So
 *      ℓ(x) = Σ_i log[ (1−ε)·2·N(d_i(x); 0, σ²) + ε/L ].
 *  A SUM over points: batchable as one integer gather per candidate on a precomputed raster, and exp(ℓ)
 *  IS the likelihood, so the posterior over candidates needs no temperature.
 *  ⚠ ε (clutter fraction) is the one modelling constant; it is physical and measurable on healthy frames.
 *  ⚠ Points are treated as independent. Neighbouring returns are correlated, so ℓ is OVERCONFIDENT in
 *    absolute terms (mode weights too decisive, Laplace Σ too tight). It does not bias WHICH mode wins,
 *    and an exactly symmetric pair still scores exactly equal. The harness measures the NEES.
 *
 *  SEARCH
 *  ------
 *  1. Yaw hypotheses from STRUCTURE: wall segments extracted from the scan without any pose
 *     (wallseg::segment) are paired with every polygon edge; each pair implies θ = ψ_edge − φ_seg. The
 *     implied angles are accumulated as a kernel density on the circle, each kernel as wide as that
 *     segment's own σ_φ, and the density's peaks are the yaws. A budget of peaks, not a threshold. No
 *     segments ⇒ uniform yaw sweep. Seed yaws (current pose, its symmetric images) are always added.
 *  2. For each yaw, ℓ on a translation lattice over the polygon interior, as one gather per point.
 *  3. Local maxima of ℓ seed an IRLS Gauss-Newton polish on the exact polygon distance (no raster),
 *     whose Hessian gives each mode its own Laplace covariance.
 *  4. Mode weight = Laplace evidence  exp(ℓ*)·|Σ|^{1/2}  (the (2π)^{3/2} is common and cancels).
 */
#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <vector>

namespace rc::reloc
{
    struct Params
    {
        float sigma_sdf      = 0.15f;   // m — SDF observation noise (RoomConcept::Params::sigma_sdf)
        float outlier_frac   = 0.30f;   // ε — clutter / through-door fraction ⚠ the modelling constant
        float raster_res     = 0.05f;   // m — likelihood raster cell (≪ σ, so quantisation is invisible)
        float lattice_step   = 0.10f;   // m — translation lattice (< σ, so every basin holds a sample)
        float body_clearance = 0.30f;   // m — robot body radius: a centre closer to a wall is impossible
        int   max_points     = 150;     // subsample of the wall band
        int   max_yaws       = 8;       // budget of structural yaw peaks
        int   fallback_yaws  = 36;      // uniform sweep when the scan yields no wall segments
        int   max_seeds      = 24;      // budget of lattice maxima sent to the polish
        int   max_modes      = 4;       // budget of modes returned (the order of a rectangle's symmetry group)
        int   polish_iters   = 40;
        // m — per-wall COMMON-MODE residual offset (map wall error + clutter bias shared by a wall's points),
        // marginalised out of each mode's covariance. Physical, and the live layout's own corner error is
        // ~2 cm flat. 0 = independent points (measured overconfident: NEES 4–6).
        float wall_offset_sigma = 0.03f;
        // Two polished seeds are the SAME mode when their difference is not resolvable under the pair's
        // combined covariance: χ²₃ at 99%. ⚠ A discrete merge decision, flagged; same form as wallseg's.
        float chi2_merge     = 11.34f;
    };

    /// A rival pose hypothesis the localiser is carrying, and its accumulated log-likelihood ratio against the
    /// committed pose (0 = as well supported; negative = less). Shared with the epistemic planner.
    struct PoseHypothesis
    {
        Eigen::Vector3f pose = Eigen::Vector3f::Zero();
        float llr = 0.f;
    };

    struct Mode
    {
        Eigen::Vector3f pose = Eigen::Vector3f::Zero();       // x, y, θ (θ wrapped to (−π, π])
        Eigen::Matrix3f cov  = Eigen::Matrix3f::Identity();   // Laplace covariance of the polish
        float log_lik    = 0.f;                               // ℓ at the polished pose
        float log_weight = 0.f;                               // ℓ + ½ log|Σ|, normalised over modes
        float weight     = 0.f;                               // exp(log_weight)
        float median_abs = 0.f;                               // median |d| (m) — comparable with the trigger
    };

    struct Result
    {
        std::vector<Mode> modes;       // sorted by weight, descending; empty ⇔ nothing to search
        float entropy = 0.f;           // nats over mode weights (0 = one mode)
        float ess     = 0.f;           // (Σw)²/Σw² over mode weights
        int   n_yaws  = 0;
        int   n_evals = 0;             // lattice candidates scored
        int   n_segments = 0;          // wall segments the yaw hypotheses came from (0 ⇒ fallback sweep)
        float duration_ms = 0.f;
    };

    /// Log-likelihood lookup of the polygon on a grid. Build once per polygon.
    class LikelihoodRaster
    {
    public:
        void build(const std::vector<Eigen::Vector2f>& polygon, const Params& p);
        bool valid() const { return nx_ > 0 and ny_ > 0; }
        /// Log-likelihood of one point at room coordinates (x, y); off-raster ⇒ the pure clutter term.
        float ll_at(float x, float y) const;
        float ll_outside() const { return ll_out_; }
        float res() const { return res_; }
        int nx() const { return nx_; }
        int ny() const { return ny_; }
        float x0() const { return x0_; }
        float y0() const { return y0_; }
        const std::vector<float>& ll() const { return ll_; }
        const std::vector<float>& dist() const { return dist_; }
        float extent() const { return extent_; }
    private:
        std::vector<float> ll_, dist_;
        float x0_ = 0.f, y0_ = 0.f, res_ = 0.05f, ll_out_ = 0.f, extent_ = 1.f;
        int nx_ = 0, ny_ = 0;
    };

    /// Unsigned distance from a room-frame point to the polygon boundary.
    float polygon_distance(const std::vector<Eigen::Vector2f>& polygon, const Eigen::Vector2f& q);
    bool  inside_polygon(const std::vector<Eigen::Vector2f>& polygon, const Eigen::Vector2f& q);

    /// ℓ(pose) evaluated exactly (no raster), plus the median |d| for comparison with the trigger.
    float log_likelihood(const std::vector<Eigen::Vector2f>& polygon,
                         const std::vector<Eigen::Vector2f>& pts_robot,
                         const Eigen::Vector3f& pose, const Params& p, float extent,
                         float* median_abs = nullptr);

    /// Yaw hypotheses from pose-independent wall segments paired with polygon edges.
    std::vector<float> structural_yaws(const std::vector<Eigen::Vector2f>& polygon,
                                       const std::vector<Eigen::Vector2f>& pts_robot,
                                       const Params& p, std::uint32_t seed, int* n_segments = nullptr);

    /// The whole search. `seed_poses` are always evaluated as extra hypotheses (current pose, last good
    /// pose, symmetric images) — they cost nothing and guarantee the search never loses the incumbent.
    Result search(const LikelihoodRaster& raster,
                  const std::vector<Eigen::Vector2f>& polygon,
                  const std::vector<Eigen::Vector2f>& pts_robot,
                  const std::vector<Eigen::Vector3f>& seed_poses,
                  const Params& p, std::uint32_t seed = 12345);

    /// Robust IRLS Gauss-Newton on ℓ from `start`; returns the polished mode (weight unset).
    Mode polish(const std::vector<Eigen::Vector2f>& polygon,
                const std::vector<Eigen::Vector2f>& pts_robot,
                const Eigen::Vector3f& start, const Params& p, float extent);
} // namespace rc::reloc
