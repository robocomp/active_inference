/*
 * place_map.h — appearance keyframes keyed to room poses, and the pose likelihood they induce.
 *
 * WHAT THIS IS FOR. The robot has no memory of what places LOOK like. When it gets lost,
 * RoomConcept::grid_search_initial_pose falls back to a blind lattice over the whole room
 * (>=1 m / 90 deg -> ~360 candidates in the 8.5 x 9.3 m apartamento), carrying nothing over from the
 * hundreds of times it has already stood in exactly those places. This holds a selection of
 * panoramic appearance descriptors, each tagged with the room pose it was captured at, and turns a
 * query panorama into a DISTRIBUTION over where the robot might be.
 *
 * ★ THE OUTPUT IS A MIXTURE, NOT AN ANSWER. Appearance cannot always separate two places -- the
 * kitchen end and the dining end of one open apartment may look genuinely similar. The correct
 * response to that is a BIMODAL mixture, not a confident guess, and the geometric SDF refinement
 * downstream is what breaks the tie. So the requirement on this channel is CALIBRATION, not
 * sharpness: a broad honest answer is a success, a narrow wrong one is the only real failure.
 *
 * ★ WHY YAW COMES OUT FOR FREE. Descriptors are stored per azimuth SECTOR of the panorama, in
 * sensor-column order. A yaw change is then exactly a cyclic permutation of the sector array, so a
 * circular cross-correlation both matches the place AND reads off the heading offset. This is the
 * whole reason the input is a 360 panorama and not the forward camera: it removes yaw from the
 * nuisance set by construction instead of forcing a position x yaw map or a search over yaw.
 *
 * ★ WHY EVERY KEYFRAME CARRIES ITS CAPTURE COVARIANCE, and why that removes a threshold. The
 * tempting design gates insertion on "was the localizer confident?". Instead each keyframe stores
 * the pose covariance it was captured with, so a keyframe recorded while the localizer was unsure
 * contributes a BROAD Gaussian to the retrieval likelihood rather than being discarded. The effect
 * is encoded as a covariance that grows with the right covariate, which is what CLAUDE.md asks for,
 * instead of an if.
 *   ⚠ This handles localizer VARIANCE, not localizer BIAS: a keyframe captured while the localizer
 *   was confidently WRONG contributes a narrow Gaussian in the wrong place and no covariance trick
 *   saves you. That is measured, not assumed -- see place_eval --map-consistency.
 *
 * DEPENDENCIES: Eigen + the standard library. NO Qt, NO DSR, NO OpenCV, NO ONNX -- deliberately, so
 * common/run_tests.sh can compile place_map_test.cpp as a single TU (it adds no other sources, which
 * is exactly why view_field.h is header-only too) and so the offline tools build in seconds.
 *
 * Related: common/view_field/view_field.h (the persistence precedent), retina/src/place_stage.cpp
 * (the producer), retina/tools/place_eval.cpp (the measurement).
 */

#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rc::place
{

inline constexpr std::uint32_t kFormatVersion = 1;
inline constexpr std::uint32_t kDtypeF32      = 0;

// ── SE(2) helpers ───────────────────────────────────────────────────────────────────────────────
// A pose is (x, y, theta). "a_T_b" reads "coordinates of b expressed in a", matching FRAMES.md.

inline float wrap_angle(float a) { return std::remainder(a, 2.0f * static_cast<float>(M_PI)); }

inline Eigen::Vector3f inv_se2(const Eigen::Vector3f& p)
{
    const float c = std::cos(p.z()), s = std::sin(p.z());
    return { -(c * p.x() + s * p.y()), -(-s * p.x() + c * p.y()), wrap_angle(-p.z()) };
}

/*
 * The Jacobian of SE(2) inversion, copied deliberately from room_scene_graph.cpp:344-352 so the two
 * agree by construction.
 *
 * ★ WHY THIS EXISTS AT ALL. room_concept publishes the robot pose on an RT edge whose direction
 * FLIPS once the room node is created: from then on the room is a CHILD of the robot, so the edge
 * carries robot->room, and its rt_covariance has already been pushed through this Jacobian. Reading
 * slots {0,1,5} off that edge and treating them as "the robot's uncertainty in the room" is wrong --
 * J mixes yaw variance into position through the lever arm t. Every stored covariance would be
 * silently wrong, and nothing would fail until the calibration test.
 *
 * ★ AND THE INVERSE IS THE SAME EIGHT LINES. SE(2) inversion is an involution, so by the chain rule
 * on f(f(p)) = p we get J(p)^-1 = J(f(p)): to undo the transport, evaluate the SAME Jacobian at the
 * INVERSE pose. No matrix inverse, no second derivation to get wrong. (Unit-tested as a round trip.)
 */
inline Eigen::Matrix3f se2_inverse_jacobian(const Eigen::Vector3f& pose)
{
    const float c = std::cos(pose.z()), s = std::sin(pose.z());
    Eigen::Matrix3f J = Eigen::Matrix3f::Zero();
    J(0, 0) = -c;  J(0, 1) = -s;  J(0, 2) =  s * pose.x() - c * pose.y();
    J(1, 0) =  s;  J(1, 1) = -c;  J(1, 2) =  c * pose.x() + s * pose.y();
    J(2, 2) = -1.f;
    return J;
}

/// Transport an SE(2) covariance through pose inversion. `pose` is the pose being inverted.
inline Eigen::Matrix3f invert_se2_cov(const Eigen::Matrix3f& cov, const Eigen::Vector3f& pose)
{
    const Eigen::Matrix3f J = se2_inverse_jacobian(pose);
    return J * cov * J.transpose();
}


/*
 * The circular cross-correlation, as a free function so the SIGN CONVENTION lives in exactly ONE
 * place. It is used by PlaceMap::match, by the encoder's startup self-test, and by place_eval; three
 * copies of "which way does the shift go" is precisely how a reflection gets shipped.
 *
 *     sim[s] = (1/S) * sum_i <Q[(i+s) mod S], K[i]>
 *
 * ★ ESTABLISHED BY EXPERIMENT (tools/export_dinov2.py roll test, 15/15 shifts): rolling the panorama
 * so content moves toward HIGHER column index by k sectors puts the peak at s = +k.
 *
 * `q_sec` / `k_sec` point at the SECTOR arrays, i.e. past the CLS token. Descriptors are assumed
 * pre-L2-normalised, so a dot product is a cosine.
 */
inline void circular_similarity(const float* q_sec, const float* k_sec, int S, int D,
                                std::vector<float>& sim, std::vector<float>* C_out = nullptr)
{
    std::vector<float> C(std::size_t(S) * std::size_t(S), 0.f);
    for (int a = 0; a < S; ++a)
        for (int b = 0; b < S; ++b)
        {
            float d = 0.f;
            for (int c = 0; c < D; ++c) d += q_sec[a * D + c] * k_sec[b * D + c];
            C[std::size_t(a) * S + b] = d;
        }
    sim.assign(std::size_t(S), 0.f);
    for (int s = 0; s < S; ++s)
    {
        float acc = 0.f;
        for (int i = 0; i < S; ++i) acc += C[std::size_t((i + s) % S) * S + i];
        sim[std::size_t(s)] = acc / float(S);
    }
    if (C_out) *C_out = std::move(C);
}

/// Peak of a circular similarity profile: (shift, sim, margin-over-runner-up).
struct Peak { int shift = 0; float sim = 0.f; float margin = 0.f; };

inline Peak circular_peak(const std::vector<float>& sim)
{
    Peak p;
    if (sim.empty()) return p;
    p.shift = int(std::max_element(sim.begin(), sim.end()) - sim.begin());
    p.sim   = sim[std::size_t(p.shift)];
    float runner = -std::numeric_limits<float>::infinity();
    for (std::size_t s = 0; s < sim.size(); ++s)
        if (int(s) != p.shift) runner = std::max(runner, sim[s]);
    p.margin = std::isfinite(runner) ? p.sim - runner : 0.f;
    return p;
}

// ── records ─────────────────────────────────────────────────────────────────────────────────────

/// Everything a mismatch must be caught on. A map queried with a different model, a different
/// elevation band, a different azimuth calibration or against a re-fitted room is not merely
/// degraded -- it is wrong in ways that look plausible. So all of it is stored and all of it is
/// checked on load.
struct Header
{
    std::uint32_t version   = kFormatVersion;
    std::uint32_t dim       = 384;
    std::uint32_t n_sectors = 16;
    float         pool_p    = 3.0f;
    int           band_lo   = 6;
    int           band_hi   = 12;
    int           center    = 1;
    int           input_w   = 448;
    int           input_h   = 224;
    std::string   model_name;
    std::string   model_sha1;
    /// ★ Room identity. Every stored pose is in the room frame AS ESTIMATED AT CAPTURE. If the room
    /// is re-fitted or its node recreated, every pose shifts by an unknown SE(2) and retrieval
    /// silently proposes poses in an obsolete frame.
    std::string   room_name;
    float         room_w    = 0.f;
    float         room_d    = 0.f;
    /// ★ Baked into room_T_sensor at ricoh_source.cpp:57 -- a map built at one tune and queried at
    /// another is silently rotated.
    float         azimuth_tune_deg = 0.f;
    Eigen::Vector3f robot_T_ricoh { 0.f, 0.f, 0.f };

    [[nodiscard]] std::size_t stride() const { return std::size_t(1 + n_sectors) * dim; }
};

struct Keyframe
{
    std::uint32_t   id       = 0;
    std::uint64_t   stamp_ms = 0;
    /// room<-ricoh: the viewpoint the image was actually taken from, so the correct appearance key.
    Eigen::Vector3f ricoh_pose { 0.f, 0.f, 0.f };
    /// room<-robot: what a consumer of the mixture actually wants back.
    Eigen::Vector3f robot_pose { 0.f, 0.f, 0.f };
    /// SE(2) covariance of robot_pose in the ROOM frame -- already un-inverted from the RT edge.
    Eigen::Matrix3f cov = Eigen::Matrix3f::Identity();
};

struct MatchResult
{
    int   kf          = -1;    ///< index into the keyframe vector
    int   shift       = 0;     ///< best circular shift, in sectors
    float shift_interp = 0.f;  ///< parabolic sub-sector refinement, in sectors (shift + delta)
    float sim         = 0.f;   ///< similarity at the peak
    float margin      = 0.f;   ///< peak minus runner-up: how DECIDED the match was
};

/*
 * Every field here is MEASURED by place_eval, never guessed. A bare constant in this struct with no
 * provenance is the defect the whole evaluation exists to prevent.
 */
struct MixtureParams
{
    /*
     * ★ PRELIMINARY MEASUREMENT, 2026-08-28 — retina/tools/place_eval, 53 panoramas from
     * etc/depth_frames joined to etc/ricoh_depth_dataset.csv poses, 1378 pairs.
     *   fit    s(d) = 0.770 * exp(-d^2 / 2*0.580^2) + 0.190   => r = 0.58 m, d* = 0.24 m
     *   tau    0.0552 by NLL minimisation (the curve has a real minimum, not a flat plateau)
     *
     * ⚠ NOT the P1-P6 verdict, for three reasons that all point the same way: n is small, the samples
     * are not independent (19 distinct stops), and the reference poses are the LOCALISER'S OWN rather
     * than robot_gt_*. Re-measure against ground truth before anything depends on these.
     *
     * ⚠ AND THEY WERE MEASURED WITH HARD SECTOR BINS. sector_soft > 0 changes the similarity scale, so
     * turning it on invalidates all four numbers below — re-run --decay --calibrate if you do.
     */
    float tau          = 0.0552f;  ///< softmax temperature; place_eval --calibrate
    // Similarity decay fit  s(d) = a * exp(-d^2 / 2r^2) + b   from place_eval --decay.
    float decay_a      = 0.770f;
    float decay_b      = 0.190f;
    float decay_r_m    = 0.580f;
    float scene_depth_m = 4.0f;   ///< Rbar, for the parallax-induced yaw term
    float sector_rad   = 2.0f * static_cast<float>(M_PI) / 16.0f;
    float yaw_interp_resid_rad = 0.f;  ///< measured; replaces the quantisation term once known
    /// Whether panorama column index increases with increasing room azimuth is a property of the
    /// ricoh unprojection and azimuth_tune, NOT something to assume: the wrong sign produces a
    /// REFLECTION rather than a rotation -- the exact signature room_concept documented for
    /// robot_gt_angle.
    /// ★ MEASURED, not assumed (2026-08-28, 250 position-matched pairs with a real yaw difference):
    ///     yaw_sign = +1  ->  median |yaw error|  6.53 deg, 99% within half a sector
    ///     yaw_sign = -1  ->  median |yaw error| 85.37 deg,  0% within
    /// Decisive. place_eval --yaw re-scores both conventions and names the winner every run, so this
    /// self-corrects if the ricoh calibration or the panorama convention ever changes.
    float yaw_sign     = 1.0f;
    float max_pos_sigma_m = 6.0f; ///< sanity clamp on the derived spread (room-scale)

    /// Displacement spread implied by an observed similarity, by inverting the decay fit.
    /// High similarity -> tight component, marginal -> broad, CONTINUOUSLY. This is the
    /// covariance-not-a-switch formulation: no gate decides whether a match "counts".
    [[nodiscard]] float rho(float sim) const
    {
        const float frac = (sim - decay_b) / std::max(decay_a, 1e-6f);
        if (not (frac > 1e-4f))              // at or below the floor: no positional information
            return max_pos_sigma_m;
        if (frac >= 1.0f)                    // at or above the peak: as tight as the fit allows
            return decay_r_m * 0.25f;
        return std::min(decay_r_m * std::sqrt(-2.0f * std::log(frac)), max_pos_sigma_m);
    }
};

// ── the mixture ─────────────────────────────────────────────────────────────────────────────────

/// A weighted mixture of Gaussians over SE(2). Speaks the same language as room_concept's
/// moment_match, so it drops into grid_search Stage 1 without translation.
struct PoseMixture
{
    std::vector<float>           w;
    std::vector<Eigen::Vector3f> mu;
    std::vector<Eigen::Matrix3f> sigma;

    [[nodiscard]] std::size_t size() const { return w.size(); }
    [[nodiscard]] bool empty() const { return w.empty(); }

    [[nodiscard]] float logpdf(const Eigen::Vector3f& p) const
    {
        if (w.empty()) return -std::numeric_limits<float>::infinity();
        float best = -std::numeric_limits<float>::infinity();
        std::vector<float> terms(w.size());
        for (std::size_t k = 0; k < w.size(); ++k)
        {
            Eigen::Vector3f d = p - mu[k];
            d.z() = wrap_angle(d.z());                       // ★ angular residual, never linear
            const Eigen::Matrix3f S = sigma[k];
            const Eigen::LLT<Eigen::Matrix3f> llt(S);
            if (llt.info() != Eigen::Success) { terms[k] = -std::numeric_limits<float>::infinity(); continue; }
            const float maha = d.dot(llt.solve(d));
            const float logdet = 2.0f * std::log(llt.matrixL().determinant());
            terms[k] = std::log(std::max(w[k], 1e-30f))
                     - 0.5f * (maha + logdet + 3.0f * std::log(2.0f * float(M_PI)));
            best = std::max(best, terms[k]);
        }
        if (not std::isfinite(best)) return best;
        float acc = 0.f;
        for (float t : terms) if (std::isfinite(t)) acc += std::exp(t - best);
        return best + std::log(acc);
    }

    /// Circular-aware weighted mean.
    [[nodiscard]] Eigen::Vector3f mean() const
    {
        Eigen::Vector3f m = Eigen::Vector3f::Zero();
        float cs = 0.f, sn = 0.f, tw = 0.f;
        for (std::size_t k = 0; k < w.size(); ++k)
        {
            m.head<2>() += w[k] * mu[k].head<2>();
            cs += w[k] * std::cos(mu[k].z());
            sn += w[k] * std::sin(mu[k].z());
            tw += w[k];
        }
        if (tw > 0.f) { m.head<2>() /= tw; m.z() = std::atan2(sn, cs); }
        return m;
    }

    /// Collapse to a single Gaussian: within-component spread plus between-component spread.
    /// The second term is what makes a bimodal answer read as UNCERTAIN rather than as its midpoint.
    [[nodiscard]] Eigen::Matrix3f moment_match() const
    {
        const Eigen::Vector3f m = mean();
        Eigen::Matrix3f S = Eigen::Matrix3f::Zero();
        float tw = 0.f;
        for (std::size_t k = 0; k < w.size(); ++k)
        {
            Eigen::Vector3f d = mu[k] - m;
            d.z() = wrap_angle(d.z());
            S += w[k] * (sigma[k] + d * d.transpose());
            tw += w[k];
        }
        if (tw > 0.f) S /= tw;
        // ★ A Gaussian over an angle is only meaningful while sigma_theta << pi. Cap the yaw
        // variance at that of the UNIFORM distribution on the circle -- the most "no information" a
        // circular distribution can express. This is a bound on representable ignorance, not a
        // threshold on belief.
        constexpr float kUniformCircVar = float(M_PI) * float(M_PI) / 3.0f;
        S(2, 2) = std::min(S(2, 2), kUniformCircVar);
        return S;
    }

    /*
     * ★ HOW grid_search STAGE 1 SHOULD CONSUME THIS: draw its TOP_K candidates from here rather than
     * taking the K highest-weight means. Sampling spends draws in proportion to BOTH weight and
     * spread, so a broad component is explored and a tight one is not over-sampled -- which is the
     * behaviour the blind lattice was approximating uniformly. One-line drop-in for the 360-point
     * lattice, feeding the unchanged Stage 2 -> moment_match contract.
     */
    [[nodiscard]] std::vector<Eigen::Vector3f> sample(std::mt19937& rng, int n) const
    {
        std::vector<Eigen::Vector3f> out;
        if (w.empty() or n <= 0) return out;
        std::discrete_distribution<int> pick(w.begin(), w.end());
        std::normal_distribution<float> gauss(0.f, 1.f);
        out.reserve(std::size_t(n));
        for (int i = 0; i < n; ++i)
        {
            const int k = pick(rng);
            const Eigen::LLT<Eigen::Matrix3f> llt(sigma[k]);
            Eigen::Vector3f z(gauss(rng), gauss(rng), gauss(rng));
            Eigen::Vector3f p = mu[k];
            if (llt.info() == Eigen::Success) p += llt.matrixL() * z;
            p.z() = wrap_angle(p.z());
            out.push_back(p);
        }
        return out;
    }
};

// ── the map ─────────────────────────────────────────────────────────────────────────────────────

class PlaceMap
{
public:
    PlaceMap() = default;
    explicit PlaceMap(const Header& h) : hdr_(h) {}

    [[nodiscard]] const Header& header() const { return hdr_; }
    void set_header(const Header& h) { hdr_ = h; }
    [[nodiscard]] std::size_t size() const { return kfs_.size(); }
    [[nodiscard]] bool empty() const { return kfs_.empty(); }
    [[nodiscard]] const std::vector<Keyframe>& keyframes() const { return kfs_; }
    [[nodiscard]] std::span<const float> descriptor(std::size_t i) const
    { return { desc_.data() + i * hdr_.stride(), hdr_.stride() }; }

    /*
     * ★ MAP DENSITY, NOT A BELIEF GATE. CLAUDE.md forbids thresholds that turn a continuous quantity
     * into a binary claim about what is TRUE. This decides only which samples to STORE -- the same
     * kind of choice as a grid resolution or a lidar decimation. No evidence is discarded: the pose
     * that would have produced a redundant keyframe is already represented by its neighbour plus
     * that neighbour's covariance. And it is falsifiable offline -- place_eval --sweep-dm rebuilds
     * the map at several densities from one logged run and reports the curve.
     *
     * ★ THERE IS DELIBERATELY NO YAW CLAUSE. One keyframe already covers every heading at that
     * position, because a yaw-displaced view is a cyclic permutation of descriptors we already hold.
     * Storing yaw duplicates would bloat the map AND corrupt the mixture, by putting several
     * components at one place and over-weighting it in the softmax.
     */
    [[nodiscard]] bool should_insert(const Eigen::Vector3f& ricoh_pose, float min_dist_m) const
    {
        // O(n) over at most ~1200 keyframes at ~2 Hz is microseconds. A kd-tree here would be
        // complexity with no measurable return; that is a decision, not an oversight.
        for (const auto& k : kfs_)
            if ((k.ricoh_pose.head<2>() - ricoh_pose.head<2>()).norm() < min_dist_m)
                return false;
        return true;
    }

    void add(const Keyframe& kf, std::span<const float> desc)
    {
        if (desc.size() != hdr_.stride()) return;         // caller bug; never silently pad
        kfs_.push_back(kf);
        desc_.insert(desc_.end(), desc.begin(), desc.end());
    }

    /*
     * Two stages, which is what makes 16 or 32 sectors free.
     *
     * A. PREFILTER on the CLS token: one dot product per keyframe, yaw-invariant, microseconds.
     *    ★ CLS is used HERE AND NOWHERE ELSE. It is a bag of scene content that discards the bearing
     *    ARRANGEMENT, and the arrangement is precisely the intra-room position signal -- which is
     *    why a pooled global descriptor varies so smoothly with translation. Letting CLS into the
     *    score would blur exactly what we came for.
     *
     * B. CIRCULAR CROSS-CORRELATION on the shortlist:
     *        sim[s] = (1/S) * sum_i <Q[(i+s) mod S], K[i]>
     *    computed as one S x S cosine matrix (descriptors are pre-L2-normalised) summed along its S
     *    circular diagonals. ~3 MFLOP for 32 candidates at S=16.
     *
     * ★ SIGN, established by experiment and not assumed (tools/export_dinov2.py roll test): rolling
     * the panorama so content moves to HIGHER column index by k sectors puts the peak at s = +k.
     *
     * If `xcorr_out` is non-null it receives the full S x S matrix of the BEST match. It is already
     * computed, and the bend in its diagonal is a bearing-only displacement estimate -- a straight
     * diagonal means pure rotation, a bent one means the viewpoint also translated.
     */
    [[nodiscard]] std::vector<MatchResult> match(std::span<const float> q, int top_n,
                                                 int prefilter_n = 32,
                                                 std::vector<float>* xcorr_out = nullptr) const
    {
        std::vector<MatchResult> out;
        if (kfs_.empty() or q.size() != hdr_.stride() or top_n <= 0) return out;
        const int S = int(hdr_.n_sectors);
        const int D = int(hdr_.dim);

        // Stage A ---------------------------------------------------------------------------------
        std::vector<std::pair<float, int>> pre;
        pre.reserve(kfs_.size());
        for (std::size_t i = 0; i < kfs_.size(); ++i)
        {
            const float* k = desc_.data() + i * hdr_.stride();
            float d = 0.f;
            for (int c = 0; c < D; ++c) d += q[std::size_t(c)] * k[c];
            pre.emplace_back(d, int(i));
        }
        const int keep = std::min<int>(std::max(prefilter_n, top_n), int(pre.size()));
        std::partial_sort(pre.begin(), pre.begin() + keep, pre.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });

        // Stage B ---------------------------------------------------------------------------------
        std::vector<float> C, sim;
        std::vector<float> best_C;
        for (int n = 0; n < keep; ++n)
        {
            const int i = pre[std::size_t(n)].second;
            const float* k = desc_.data() + std::size_t(i) * hdr_.stride() + D;   // skip CLS
            const float* qs = q.data() + D;
            circular_similarity(qs, k, S, D, sim, xcorr_out ? &C : nullptr);
            const Peak pk = circular_peak(sim);

            MatchResult m;
            m.kf = i; m.shift = pk.shift; m.sim = pk.sim; m.margin = pk.margin;
            m.shift_interp = float(pk.shift) + parabolic_delta(sim, pk.shift, S);
            out.push_back(m);
            if (xcorr_out and (out.size() == 1 or m.sim > out.front().sim)) best_C = C;
        }
        std::partial_sort(out.begin(), out.begin() + std::min<int>(top_n, int(out.size())), out.end(),
                          [](const MatchResult& a, const MatchResult& b) { return a.sim > b.sim; });
        out.resize(std::min<std::size_t>(std::size_t(top_n), out.size()));
        if (xcorr_out) *xcorr_out = std::move(best_C);
        return out;
    }

    /// Turn matches into a pose likelihood. See MixtureParams for why every constant is measured.
    [[nodiscard]] PoseMixture to_mixture(const std::vector<MatchResult>& ms,
                                         const MixtureParams& p) const
    {
        PoseMixture mix;
        if (ms.empty()) return mix;
        const float smax = std::max_element(ms.begin(), ms.end(),
            [](auto& a, auto& b) { return a.sim < b.sim; })->sim;
        float wsum = 0.f;
        for (const auto& m : ms)
        {
            if (m.kf < 0 or std::size_t(m.kf) >= kfs_.size()) continue;
            const Keyframe& kf = kfs_[std::size_t(m.kf)];
            const float rho = p.rho(m.sim);

            Eigen::Vector3f mu = kf.robot_pose;
            mu.z() = wrap_angle(kf.robot_pose.z() + p.yaw_sign * m.shift_interp * p.sector_rad);

            // Quantisation (or the measured interpolation residual) plus the PARALLAX term: the best
            // shift for a DISPLACED viewpoint is not the true yaw difference, and that bearing error
            // is ~ rho / scene_depth. At rho = 1 m and Rbar = 4 m it is 14 deg -- not negligible, and
            // it makes yaw uncertainty depend on position uncertainty, which is physically right.
            const float q_quant = p.yaw_interp_resid_rad > 0.f
                                ? p.yaw_interp_resid_rad * p.yaw_interp_resid_rad
                                : p.sector_rad * p.sector_rad / 12.0f;
            const float q_parallax = (rho / std::max(p.scene_depth_m, 1e-3f))
                                   * (rho / std::max(p.scene_depth_m, 1e-3f));

            Eigen::Matrix3f S = kf.cov;
            S(0, 0) += rho * rho;
            S(1, 1) += rho * rho;
            S(2, 2) += q_quant + q_parallax;
            constexpr float kUniformCircVar = float(M_PI) * float(M_PI) / 3.0f;
            S(2, 2) = std::min(S(2, 2), kUniformCircVar);

            const float w = std::exp((m.sim - smax) / std::max(p.tau, 1e-6f));
            mix.w.push_back(w); mix.mu.push_back(mu); mix.sigma.push_back(S);
            wsum += w;
        }
        if (wsum > 0.f) for (auto& w : mix.w) w /= wsum;
        return mix;
    }

    // ── persistence ─────────────────────────────────────────────────────────────────────────────
    // DIAG-ROTATE: exempt -- this is PERSISTED STATE, not a diagnostics log (the door_concept
    // convention). The query LOG rotates; the map does not.
    //
    // Text index + binary blob. Text so the index is greppable and diffable; binary so 384-float
    // descriptors neither bloat nor lose precision through decimal round-tripping. Locale discipline
    // per CLAUDE.md: imbue(classic) to write, from_chars to read -- these machines run es_ES.UTF-8,
    // where a stream would read "0.5" as 0.

    [[nodiscard]] bool save(const std::string& index_path, const std::string& blob_path) const
    {
        std::ofstream b(blob_path, std::ios::binary | std::ios::trunc);
        if (not b) return false;
        // ★ Field by field, NEVER fwrite of a struct: padding makes that unportable and silently wrong.
        const char magic[4] = { 'R', 'C', 'P', 'M' };
        b.write(magic, 4);
        const std::uint32_t fields[5] = { hdr_.version, hdr_.dim, hdr_.n_sectors,
                                          std::uint32_t(kfs_.size()), kDtypeF32 };
        for (std::uint32_t f : fields) b.write(reinterpret_cast<const char*>(&f), sizeof(f));
        b.write(reinterpret_cast<const char*>(desc_.data()),
                std::streamsize(desc_.size() * sizeof(float)));
        if (not b) return false;
        b.close();

        std::ofstream f(index_path, std::ios::trunc);
        if (not f) return false;
        f.imbue(std::locale::classic());          // never emit a comma decimal separator
        // ★ AND enough digits to round-trip a float EXACTLY. The default is 6 SIGNIFICANT digits,
        // which silently truncates any value >= 1: writing 1.041668 back as "1.04167" loses 2e-6 m,
        // and the error GROWS with distance from the room origin -- so a map degrades toward its
        // edges while looking fine near the centre. max_digits10 (9 for float) is the documented
        // round-trip guarantee, and it costs three characters per number.
        f << std::setprecision(std::numeric_limits<float>::max_digits10);
        f << "# place_map v" << hdr_.version << "\n"
          << "# dim=" << hdr_.dim << " n_sectors=" << hdr_.n_sectors << " pool_p=" << hdr_.pool_p
          << " band_lo=" << hdr_.band_lo << " band_hi=" << hdr_.band_hi
          << " center=" << hdr_.center
          << " input_w=" << hdr_.input_w << " input_h=" << hdr_.input_h << "\n"
          << "# model=" << (hdr_.model_name.empty() ? "unknown" : hdr_.model_name)
          << " model_sha1=" << (hdr_.model_sha1.empty() ? "none" : hdr_.model_sha1) << "\n"
          << "# room=" << (hdr_.room_name.empty() ? "room" : hdr_.room_name)
          << " room_w=" << hdr_.room_w << " room_d=" << hdr_.room_d
          << " azimuth_tune_deg=" << hdr_.azimuth_tune_deg << "\n"
          << "# robot_T_ricoh=" << hdr_.robot_T_ricoh.x() << " " << hdr_.robot_T_ricoh.y()
          << " " << hdr_.robot_T_ricoh.z() << "\n"
          << "id,stamp_ms,x,y,theta,rx,ry,rtheta,cxx,cxy,cxt,cyy,cyt,ctt\n";
        for (const auto& k : kfs_)
            f << k.id << ',' << k.stamp_ms << ','
              << k.ricoh_pose.x() << ',' << k.ricoh_pose.y() << ',' << k.ricoh_pose.z() << ','
              << k.robot_pose.x() << ',' << k.robot_pose.y() << ',' << k.robot_pose.z() << ','
              // 6 upper-triangular values, not 9: makes an asymmetric matrix unrepresentable.
              << k.cov(0,0) << ',' << k.cov(0,1) << ',' << k.cov(0,2) << ','
              << k.cov(1,1) << ',' << k.cov(1,2) << ',' << k.cov(2,2) << '\n';
        return bool(f);
    }

    /// Loud, specific refusal on any mismatch: `why_not` names the field that disagreed. A silently
    /// mismatched descriptor file is the classic way this kind of format kills a day.
    [[nodiscard]] bool load(const std::string& index_path, const std::string& blob_path,
                            const Header* expect, std::string* why_not)
    {
        auto fail = [&](std::string m) { if (why_not) *why_not = std::move(m); return false; };
        std::ifstream f(index_path);
        if (not f) return fail("cannot open index " + index_path);

        Header h;
        std::vector<Keyframe> kfs;
        std::string line;
        bool seen_cols = false;
        while (std::getline(f, line))
        {
            if (line.empty()) continue;
            if (line[0] == '#') { parse_meta(line, h); continue; }
            if (not seen_cols and line.rfind("id,", 0) == 0) { seen_cols = true; continue; }
            Keyframe k;
            float c[6] {};
            std::string_view v(line);
            std::uint64_t idv = 0;
            bool ok = next_num(v, idv) and next_num(v, k.stamp_ms)
                  and next_num(v, k.ricoh_pose.x()) and next_num(v, k.ricoh_pose.y())
                  and next_num(v, k.ricoh_pose.z())
                  and next_num(v, k.robot_pose.x()) and next_num(v, k.robot_pose.y())
                  and next_num(v, k.robot_pose.z());
            for (int i = 0; i < 6 and ok; ++i) ok = next_num(v, c[i]);
            if (not ok) continue;                       // malformed row: skip, never fatal
            k.id = std::uint32_t(idv);
            k.cov << c[0], c[1], c[2],  c[1], c[3], c[4],  c[2], c[4], c[5];
            kfs.push_back(k);
        }

        std::ifstream b(blob_path, std::ios::binary);
        if (not b) return fail("cannot open blob " + blob_path);
        char magic[4] {};
        b.read(magic, 4);
        if (std::memcmp(magic, "RCPM", 4) != 0) return fail("blob magic is not RCPM");
        std::uint32_t fld[5] {};
        for (auto& x : fld) b.read(reinterpret_cast<char*>(&x), sizeof(x));
        if (fld[0] != kFormatVersion)
            return fail("blob version " + std::to_string(fld[0]) + " != " + std::to_string(kFormatVersion));
        if (fld[4] != kDtypeF32) return fail("blob dtype " + std::to_string(fld[4]) + " unsupported");
        h.version = fld[0]; h.dim = fld[1]; h.n_sectors = fld[2];
        if (fld[3] != kfs.size())
            return fail("blob has " + std::to_string(fld[3]) + " keyframes, index has "
                        + std::to_string(kfs.size()));
        if (expect)
        {
            if (expect->dim != h.dim)             return fail("dim " + std::to_string(h.dim) + " != expected " + std::to_string(expect->dim));
            if (expect->n_sectors != h.n_sectors) return fail("n_sectors " + std::to_string(h.n_sectors) + " != expected " + std::to_string(expect->n_sectors));
            if (expect->band_lo != h.band_lo or expect->band_hi != h.band_hi)
                return fail("elevation band [" + std::to_string(h.band_lo) + "," + std::to_string(h.band_hi) + ") != expected");
            if (expect->center != h.center)       return fail("centering flag differs");
            if (std::fabs(expect->pool_p - h.pool_p) > 1e-4f) return fail("pool_p differs");
            if (not expect->model_name.empty() and not h.model_name.empty()
                and expect->model_name != h.model_name) return fail("model " + h.model_name + " != expected " + expect->model_name);
            if (not expect->room_name.empty() and not h.room_name.empty()
                and expect->room_name != h.room_name) return fail("room '" + h.room_name + "' != expected '" + expect->room_name + "'");
            if (std::fabs(expect->azimuth_tune_deg - h.azimuth_tune_deg) > 1e-3f)
                return fail("azimuth_tune_deg differs: map built at a different ricoh calibration");
        }
        const std::size_t n = kfs.size() * std::size_t(1 + h.n_sectors) * h.dim;
        std::vector<float> d(n);
        b.read(reinterpret_cast<char*>(d.data()), std::streamsize(n * sizeof(float)));
        if (std::size_t(b.gcount()) != n * sizeof(float))
            return fail("blob short: expected " + std::to_string(n * sizeof(float)) + " bytes of payload");

        hdr_ = h; kfs_ = std::move(kfs); desc_ = std::move(d);
        return true;
    }

private:
    static float parabolic_delta(const std::vector<float>& sim, int bs, int S)
    {
        if (S < 3) return 0.f;
        const float ym = sim[std::size_t((bs - 1 + S) % S)];
        const float y0 = sim[std::size_t(bs)];
        const float yp = sim[std::size_t((bs + 1) % S)];
        const float den = ym - 2.f * y0 + yp;
        if (std::fabs(den) < 1e-9f) return 0.f;
        return std::clamp(0.5f * (ym - yp) / den, -0.5f, 0.5f);
    }

    template <class T>
    static bool to_num(std::string_view s, T& out)
    {
        while (not s.empty() and (s.front() == ' ' or s.front() == '\t')) s.remove_prefix(1);
        while (not s.empty() and (s.back() == ' ' or s.back() == '\t' or s.back() == '\r')) s.remove_suffix(1);
        if (s.empty()) return false;
        // ★ from_chars ONLY. strtof/atof/>> read through LC_NUMERIC; under es_ES they stop at the '.'
        // and return the integer part, silently, with no error flag (CLAUDE.md).
        return std::from_chars(s.data(), s.data() + s.size(), out).ec == std::errc{};
    }

    template <class T>
    static bool next_num(std::string_view& v, T& out)
    {
        const auto p = v.find(',');
        const std::string_view tok = v.substr(0, p);
        v = (p == std::string_view::npos) ? std::string_view{} : v.substr(p + 1);
        return to_num(tok, out);
    }

    static void parse_meta(std::string_view line, Header& h)
    {
        line.remove_prefix(1);                                   // drop '#'
        while (not line.empty())
        {
            while (not line.empty() and line.front() == ' ') line.remove_prefix(1);
            const auto sp = line.find(' ');
            std::string_view tok = line.substr(0, sp);
            line = (sp == std::string_view::npos) ? std::string_view{} : line.substr(sp + 1);
            const auto eq = tok.find('=');
            if (eq == std::string_view::npos) continue;
            const std::string_view key = tok.substr(0, eq), val = tok.substr(eq + 1);
            if      (key == "dim")              to_num(val, h.dim);
            else if (key == "n_sectors")        to_num(val, h.n_sectors);
            else if (key == "pool_p")           to_num(val, h.pool_p);
            else if (key == "band_lo")          to_num(val, h.band_lo);
            else if (key == "band_hi")          to_num(val, h.band_hi);
            else if (key == "center")           to_num(val, h.center);
            else if (key == "input_w")          to_num(val, h.input_w);
            else if (key == "input_h")          to_num(val, h.input_h);
            else if (key == "room_w")           to_num(val, h.room_w);
            else if (key == "room_d")           to_num(val, h.room_d);
            else if (key == "azimuth_tune_deg") to_num(val, h.azimuth_tune_deg);
            else if (key == "model")            h.model_name = std::string(val);
            else if (key == "model_sha1")       h.model_sha1 = std::string(val);
            else if (key == "room")             h.room_name  = std::string(val);
        }
    }

    Header               hdr_;
    std::vector<Keyframe> kfs_;
    std::vector<float>    desc_;      ///< kfs_.size() * (1 + n_sectors) * dim, CLS first
};

}   // namespace rc::place
