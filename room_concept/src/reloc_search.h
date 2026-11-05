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
 *  ★ Task 7c (2026-11-04): ℓ — every mode weight, polish and rival comparison — is now a BEAM MODEL on the room map
 *  (reloc_search.cpp, BeamTerm): per return along its ray from the sensor, z* = the range to the first mapped wall;
 *  p_short = λe^{−λz} on [0, z*) (unmapped surfaces, Params::unmapped_rate), p_hit = e^{−λz*}·N(z; z*, σ²/cos²),
 *  p_beyond only through an open cell (and only up to the next mapped wall on the ray), p_rand = ε/L. A return SHORT of
 *  a wall is cheap; one LONG of a wall the map says is there is expensive. The SDF band below survives only in the
 *  lattice raster, which proposes seeds.
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
        float sigma_along = 0.f;   ///< m — door_concept's own position σ along the wall (its RT covariance); 0 unknown
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
    /// Each line used as "beyond" pays its FULL marginal (fix round 1, I-2): flat priors over its offset (beyond_extent)
    /// and its angle (π), Laplace over the n returns it explains — log(π·L_b) − log 2π + ½log(n/σ²) + ½log(n·var_t/σ²),
    /// var_t the spread of those returns along it. ≈ 8–13 nats a line for 20–100 returns, not the ½·log n × 2 of BIC.
    struct BeyondLines
    {
        std::vector<float> r;     ///< per point: perpendicular distance to its own line (m); +∞ when on none
        std::vector<float> t;     ///< per point: coordinate along its own line (m)
        std::vector<int>   seg;   ///< per point: its line, −1 none
        std::vector<Eigen::Vector2f> n;   ///< per point: its line's unit normal (robot frame) — the beam's incidence on it
    };

    /// ── A PRIOR ABOUT WHERE THE ROBOT IS IN THIS ROOM, from somewhere other than the scan (Task 9) ─────────────────
    /// One component of a Gaussian mixture over the pose in the ROOM frame: mean (x, y, θ), covariance, weight. The
    /// room recogniser's posterior of "I am in the remembered room K, here" arrives as such a mixture (ltsm's
    /// `recall_pose_prior`); a symmetric room cannot tell its twins apart from the scan, the prior can.
    /// It enters search() as a continuity TERM in each mode's
    /// weight — log Σ_c w_c N(x; μ_c, Σ_c + Σ_mode) with a uniform floor of mass `Params::prior_wrong` — never a gate:
    /// a prior metres off whose covariance admits the truth is overruled by the likelihood, and an empty prior leaves
    /// every number exactly as it was.
    struct PriorMode
    {
        Eigen::Vector3f mu = Eigen::Vector3f::Zero();
        Eigen::Matrix3f S  = Eigen::Matrix3f::Identity();
        float w = 1.f;
    };

    struct Params
    {
        float sigma_sdf      = 0.15f;   // m — SDF observation noise (RoomConcept::Params::sigma_sdf)
        float outlier_frac   = 0.30f;   // ε — clutter / through-door fraction ⚠ the modelling constant
        /// 1/m — λ, the rate of UNMAPPED surfaces along a beam at wall-band height (furniture faces, door leaves, people,
        /// unlisted structure) in the beam model's p_short = λe^{−λz} on [0, z*) (reloc_search.cpp, BeamTerm). A rate of
        /// the WORLD, not a gate: a beam meets one before a wall z* away with probability 1 − e^{−λz*}.
        /// ⚠ FLAGGED world constant. Measured on the recorded truth frames (warm43/44/45, tools/fixtures/reloc_twin_*,
        ///   task-7c report): returns ending > 3σ_r short of z* (σ = 5 cm), per metre of ray travelled (the censored
        ///   exponential's MLE): 0.02–0.16 /m per frame, 0.085 /m pooled over 19 frames. 0.1 /m ≈ one unmapped surface
        ///   per 10 m of ray, i.e. a 4 m beam meets one with probability 0.33.
        float unmapped_rate  = 0.10f;
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
        /// Doorways door_concept lists for the room (Task R2). They set WHERE an opening is likely; they are not the only
        /// place one can be (list_wrong below).
        std::vector<Aperture> apertures;
        /// ── A LISTED DOOR IS AN OPENING OF UNCERTAIN PLACE (fix round 1, I-1) ───────────────────────────────────────
        /// The boundary is cut into cells of about `opening_cell_m` (a doorway's width). Each listed door is shut, or open
        /// in ONE cell: where door_concept lists it (smeared by its own position σ along the wall), or — with probability
        /// `list_wrong` — anywhere on the boundary. One state per door, common to all its beams.
        /// ★ WHY: a door list taken as exact let a MISPLACED door explain a wrong pose's returns almost free while the
        ///   truth's real but unlisted opening charged clutter per return — reloc_selftest O, the list 4.4 m off as it
        ///   is for ~1 s after every recall: the forced search MOVED a correct pose to the twin. Now the truth explains
        ///   its door at the price of log(N_cells/list_wrong) once.
        /// ★ THE OPENINGS ARE A UNION (Task 7b). The boundary is open where ANY door is; a cell's beams are evidence once,
        ///   however many doors could be in it. Marginalising each door on its own (Σ_k log Z_k) charged the cell a wrong
        ///   pose stands next to once PER DOOR (every door reaches every cell through list_wrong): warm43 f358's 178° twin
        ///   out-scored the truth 293 vs 191 nats with the tracker at the truth. Now 84 vs 191 (reloc_selftest X).
        /// ⚠ two modelling constants (priors, not gates), flagged: the cell scale and list_wrong. list_wrong = 0.1 is of
        ///   the order measured live: the crossed door placed 4.3–4.6 m off for ~1 s after each recall (rtD1g2), door_5
        ///   rotated 90° for whole visits (rtD1i).
        float opening_cell_m = 1.0f;    ///< m — a doorway's width
        float list_wrong     = 0.1f;    ///< P(a listed door is not where door_concept lists it)
        /// m — a return from beyond a doorway is spread uniformly over this range past it. The SAME constant as the
        /// tracker's next-room hypothesis (DoorApertures::kBeyondExtentM): one generative model for both.
        float beyond_extent  = 20.f;
        /// A pose prior from outside the scan (Task 9; PriorMode). Empty = none. Its components also seed the polish.
        std::vector<PriorMode> prior_modes;
        /// P(the prior is wrong) — the mass of its uniform floor.
        /// FLAGGED: a modelling constant, the same kind as `list_wrong` (a prior on a hypothesis source, not a gate); 0.1 as
        /// there. No model-level term can replace it: how often the recogniser is wrong is a fact about the recogniser.
        float prior_wrong = 0.1f;
        /// The prior IS the hypothesis under test (a recognition proposal: "K, and the robot here"): only its own modes are
        /// polished — no lattice maxima, no caller seeds — so the search refines the proposal and never jumps to another
        /// basin of K. Whether K at that pose explains the scans is then the verdict's call. Ignored without prior_modes.
        bool prior_only = false;
        /// THE ENTRY AREA AFTER A HAND-OVER (2026-11-01, user: "the relocalization, in these situations, should focus in the
        /// entry area and robot heading, which is much smaller than the whole apartment"). Non-empty ⇒ only these poses and
        /// the caller's seeds (the incumbent) are polished: no lattice maxima anywhere else in the room. The caller fills it
        /// from the door anchor carried since the hand-over, spanning its own heading spread. pingpong39 leg 6: a recall
        /// placed 8° off in room_1's corridor stub, and the room-wide lattice took a 4.35 m / 179° twin that explained the
        /// through-door returns better (the independent-point ℓ outweighs the anchor's few nats), lost for the next leg.
        /// Ignored when prior_only holds (a recognition proposal is its own hypothesis).
        std::vector<Eigen::Vector3f> local_seeds;
    };

    /// log of the prior density at a pose whose own Laplace covariance is `cov`: log[(1−prior_wrong)·Σ_c w_c N(pose;
    /// μ_c, S_c + cov) + prior_wrong·U], U = 1/(area·2π). `area` = the room's area (m²). 0 with no prior (nothing added).
    double log_prior_modes(const Params& p, const Eigen::Vector3f& pose, const Eigen::Matrix3f& cov, double area);
    /// Unsigned polygon area.
    double polygon_area(const std::vector<Eigen::Vector2f>& polygon);

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

    /// ℓ(pose) evaluated exactly (no raster), plus the median |d| for comparison with the trigger. With listed doors,
    /// each door is shut or open in one boundary cell (Params::opening_cell_m), and the beams whose rays reach that cell
    /// under THIS pose are explained by it — the leaf, or a wall of the next room (BeyondLines). The median is weighted by each point's posterior of
    /// being a room return — the tracker's door-weighted statistic.
    float log_likelihood(const std::vector<Eigen::Vector2f>& polygon,
                         const std::vector<Eigen::Vector2f>& pts_robot,
                         const Eigen::Vector3f& pose, const Params& p, float extent,
                         float* median_abs = nullptr, const BeyondLines* beyond = nullptr,
                         const std::vector<Eigen::Vector2f>* origins = nullptr);

    /// The lines of a point set (robot frame) as the beyond-a-door density needs them. Empty without listed doors.
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
                const Eigen::Vector3f& start, const Params& p, float extent, const BeyondLines* beyond = nullptr,
                const std::vector<Eigen::Vector2f>* origins = nullptr);

    /// The point subset search() polishes and scores its modes on. A caller scoring a hypothesis of its own
    /// (the KEEP hypothesis) against the search's modes must use THIS set: evidences over different point
    /// sets are not comparable (09-16 lesson — 150 vs 360 points read as "the polish stops 14–50 nats short").
    std::vector<Eigen::Vector2f> polish_points(const std::vector<Eigen::Vector2f>& pts_all, const Params& p);

    /// Laplace log-evidence ℓ* + ½ log|Σ| of a polished mode — the unnormalised weight search() uses.
    float log_evidence(const Mode& m);
} // namespace rc::reloc
