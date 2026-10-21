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
 *  With the room's open doorways (Params::apertures, Task R2) a return whose beam goes through a doorway under
 *  pose x is instead explained by the door: shut, or open and the return on a wall of the next room (see Aperture,
 *  BeyondLines). The lattice still scores the room-only raster; the polish and every mode weight use the full ℓ.
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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace rc::reloc
{
    /// ── AN OPEN DOORWAY, AS THE RELOCALISER'S LIKELIHOOD SEES IT (2026-10-21, Task R2) ─────────────────────────
    /// The MOUTH of a doorway (the aperture seated on the room polygon, DoorApertures::seat_in_room), in the ROOM
    /// frame, with `open` = P(a beam that crosses it is not stopped by a surface in it) = 1 − prior_closed.
    /// ★ WHY THE RELOCALISER NEEDS IT. The tracker explains a return through an open door by the per-beam mixture
    ///   "a surface in the aperture, or the next room" (DoorApertures::weight). The relocaliser's ℓ did not: it scored
    ///   every return against the room polygon alone, so a correct pose looking through a door was charged a clutter
    ///   term for each through-door return, while a twin that warped those returns onto other walls was charged
    ///   nothing. Live rtD1i (742.71 s): the robot 0.39 m inside the open door_1 (the aperture ≈104° of the scan),
    ///   the correct pose (1 cm from truth) at median |SDF| 0.134 m on the search's points against a 4.1 m / −160°
    ///   twin at 0.023 m — and the search moved the pose to the twin and kept it to the end of the run.
    ///   The mixture is evaluated PER HYPOTHESIS: each candidate pose puts the sensor somewhere else, so which beams
    ///   cross the mouth — and are explained by "beyond the door" — is a property of that pose, not of the scan.
    struct Aperture
    {
        Eigen::Vector2f a{0.f, 0.f}, b{0.f, 0.f};   ///< the mouth, room frame
        Eigen::Vector2f n_out{0.f, 0.f};            ///< unit normal out of the room; zero when unseated
        float open = 0.f;     ///< per beam, if the door is as believed: P(no leaf in its way) = 1 − prior_closed
        float p_shut = 0.f;   ///< P(the door is in fact shut) — ONE fact for all its beams (common-mode; reloc_search.cpp)
    };

    /// π_b = P(the return of the beam origin→hit came from BEYOND an open doorway), a priori: the union over apertures
    /// of (1 − p_shut)·open × [the beam goes through that mouth]. (ℓ itself treats the door's state as common to all
    /// its beams; this per-beam marginal is for diagnostics and tests.) The crossing is exact geometry, the same two cases as
    /// DoorApertures::weight: a sensor inside the room — the beam crosses the mouth segment before its return; a
    /// sensor standing in the opening (past the mouth line, within its span) — the return ended past the mouth line.
    float exit_prior(const std::vector<Aperture>& apertures, const Eigen::Vector2f& origin, const Eigen::Vector2f& hit);

    /// ── WHAT LIES BEYOND A DOOR IS ANOTHER ROOM'S WALLS (Task R2) ─────────────────────────────────────────────
    /// A return through an open doorway is not "anywhere within 20 m": it lies on a wall of the next room, which this
    /// map does not hold. With "beyond" uniform, a correct pose looking through a door paid ~6 nats per through-door
    /// return against any pose that warped those returns onto a known wall — the reloc_selftest Q world in which the
    /// view through the door plus the room fits the room turned 90°: kept 0/20 (and rtD1i's −160° twin). So the
    /// beyond density is the ROOM TERM ITSELF, evaluated on a wall of UNKNOWN placement: d = the return's distance to a
    /// line segmented from the scan itself (wallseg, robot frame — pose-independent, computed once per point set), with
    /// the room term's own σ and ε (clutter spread over the beyond range). Each such line used as "beyond" pays its
    /// Occam cost, ½·log n per fitted parameter (2): the room's walls are known, the next room's are fitted.
    /// ⚠ The SAME σ as the room term, deliberately: with the line's own 2 cm fit σ against the room's 15 cm, "beyond"
    ///   out-scored "on a known wall" by ~2 nats a return and every pose that put beams through a door gained by it.
    struct BeyondLines
    {
        std::vector<float> r;     ///< per point: perpendicular distance to its own line (m); +∞ when on none
        std::vector<int>   seg;   ///< per point: its line, −1 none
    };

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
        /// Open doorways of the room (Task R2). Empty ⇒ ℓ is the room-only model, exactly as before.
        std::vector<Aperture> apertures;
        /// m — a return from beyond a doorway is spread uniformly over this range past it. The SAME constant as the
        /// tracker's next-room hypothesis (DoorApertures::kBeyondExtentM): one generative model for both.
        float beyond_extent  = 20.f;
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
        float median_abs = 0.f;                               // median |d| (m), door-weighted — comparable with the trigger
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

    /// ℓ(pose) evaluated exactly (no raster), plus the median |d| for comparison with the trigger. With apertures, each
    /// point's term is the per-beam mixture (1−π_b)·room(d) + π_b / beyond_extent under THIS pose, and the median is
    /// weighted by each point's posterior of being a room return — the tracker's door-weighted statistic.
    float log_likelihood(const std::vector<Eigen::Vector2f>& polygon,
                         const std::vector<Eigen::Vector2f>& pts_robot,
                         const Eigen::Vector3f& pose, const Params& p, float extent,
                         float* median_abs = nullptr, const BeyondLines* beyond = nullptr);

    /// The lines of a point set (robot frame) as the beyond-a-door density needs them. Empty when p has no apertures.
    /// Pass it to log_likelihood/polish when scoring many poses on ONE point set; omitted, they compute it themselves.
    BeyondLines beyond_lines(const std::vector<Eigen::Vector2f>& pts_robot, const Params& p);

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
                const Eigen::Vector3f& start, const Params& p, float extent, const BeyondLines* beyond = nullptr);

    /// The point subset search() polishes and scores its modes on. A caller scoring a hypothesis of its own
    /// (the KEEP hypothesis) against the search's modes must use THIS set: evidences over different point
    /// sets are not comparable (09-16 lesson — 150 vs 360 points read as "the polish stops 14–50 nats short").
    std::vector<Eigen::Vector2f> polish_points(const std::vector<Eigen::Vector2f>& pts_all, const Params& p);

    /// Laplace log-evidence ℓ* + ½ log|Σ| of a polished mode — the unnormalised weight search() uses.
    float log_evidence(const Mode& m);

    /// ── WHICH HYPOTHESIS MAY THE POSE BECOME? (multi-hypothesis SPRT, 2026-10-21) ─────────────────────
    /// `log_post` are unnormalised log-posteriors of K mutually exclusive pose hypotheses; `keep` indexes the
    /// one the localiser already holds. The pose moves to `top` (the MAP) only when the posterior SAYS so:
    ///   • `top` is ACCEPTED — its odds against all the others together pass log((1−α)/α), the bound the rival
    ///     SPRT already uses (α = the hazard); or
    ///   • `keep` is REJECTED — its own posterior is ≤ α. Holding a pose the scan has rejected is the one choice
    ///     known to be wrong, so the MAP is committed (with the mixture's spread as its covariance, at the call
    ///     site) even when it is not decisive among the remaining modes.
    /// Otherwise HOLD: no hypothesis is decisive and the held one is still alive, so the decision is deferred to
    /// the evidence of the following scans (the rivals). A 180° twin that explains the scan exactly as well as
    /// the held pose can therefore never take it — which a per-search "move to the top mode" did on a coin flip.
    /// No new constant: α is the hazard. `keep` < 0 means the held pose has no basin of its own (its polish
    /// left it), i.e. it is rejected by the scan itself.
    enum class Decision { Keep, Move, Hold };
    inline Decision msprt_decide(const std::vector<double>& log_post, int top, int keep, double alpha)
    {
        if (top < 0 or top >= static_cast<int>(log_post.size())) return Decision::Keep;
        if (top == keep) return Decision::Keep;
        double hi = -std::numeric_limits<double>::infinity();
        for (const double l : log_post) hi = std::max(hi, l);
        if (not std::isfinite(hi)) return Decision::Keep;
        double z = 0.0;
        for (const double l : log_post) z += std::exp(l - hi);
        const auto post = [&](int k) { return std::exp(log_post[static_cast<std::size_t>(k)] - hi) / z; };
        const double a = std::clamp(alpha, 1e-12, 0.5);
        if (post(top) >= 1.0 - a) return Decision::Move;
        if (keep < 0 or post(keep) <= a) return Decision::Move;
        return Decision::Hold;
    }
} // namespace rc::reloc
