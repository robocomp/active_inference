/*
 * chair_belief.h  —  AI2 chair belief (pose-only, on the shared rc::ai engine)
 *
 * A chair is RIGID STANDARD FURNITURE, so the belief estimates only its POSE against a fixed
 * standard-chair TEMPLATE — it does NOT estimate size. This removes the whole degenerate size space
 * (seat_w/seat_d/seat_h/back_h) that the flat-clutter escape valve kept collapsing/inflating one DOF at
 * a time. A global size scale can be added later as one well-conditioned DOF; per-axis size is a further
 * extension. What remains is bottle-level conditioned: position is the point centroid, yaw is fixed by the
 * (asymmetric) backrest.
 *
 * State θ = [cx, cy, yaw]  (3 DOF). cz is PINNED to the floor; all dimensions are the fixed template.
 * Compound model {seat slab + backrest slab + 4 legs + clutter} with soft per-point responsibilities
 * (z-band part attribution), inferred by the shared recursive variational-Laplace filter (3×3 Σ). Pure
 * Eigen, no torch, no DSR — unit-testable in isolation (self_test()).
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <cmath>
#include <vector>
#include <Eigen/Dense>

#include "../../common/ai_belief/recursive_laplace.h"

namespace rc
{

struct ChairBeliefState
{
    float cx = 0.0f, cy = 0.0f, yaw = 0.0f;   // pose only; size is the fixed template, cz pinned to floor

    Eigen::Matrix<float, 3, 1> vec() const
    { return (Eigen::Matrix<float, 3, 1>() << cx, cy, yaw).finished(); }
    static ChairBeliefState from_vec(const Eigen::Matrix<float, 3, 1>& v)
    { return {v(0), v(1), v(2)}; }
};

struct ChairBeliefParams
{
    // ── Fixed standard-chair TEMPLATE (the belief estimates pose only; a global scale can multiply these
    //    later). Legs sit at the seat corners; the backrest on the −y seat edge. ──
    // ★Defaults match the real Webots SimpleChair (2026-07-27). The fitter overrides all of these from
    // config; they matter for anyone constructing a belief directly (e.g. self_test). Keeping them at a
    // fictional 0.45 cube is how the seat/backrest z-band misattribution went unnoticed — see
    // ChairConfig::tracker_birth_seat_* for the full account.
    float tpl_seat_w = 0.60f, tpl_seat_d = 0.52f, tpl_seat_h = 0.595f, tpl_back_h = 0.655f;
    float seat_thickness = 0.075f;
    float leg_half       = 0.0375f;   // square leg half-side
    float floor_z        = 0.0f;    // room-frame floor: cz is PINNED here (not a DOF)
    float floor_std      = 0.03f;   // floor-height uncertainty (m) → published z covariance

    // Observation model
    float sigma_base_m    = 0.03f;
    float clutter_frac    = 0.10f;
    float clutter_scale_m = 0.12f;

    // Priors (init Σ) — pose only.
    float prior_pos_std = 0.30f;
    float prior_yaw_std = 0.60f;

    // Temporal transition (static furniture): small process noise so the pose tracks localisation jitter.
    float process_std_m   = 0.005f;   // cx,cy
    float process_std_yaw = 0.01f;    // yaw
    // Σ CEILING at the prior on predict: Q is authored PER FRAME, so an unobserved chair's pose belief decays
    // as σ ∝ √frames with nothing bounding it — after minutes it is looser than at birth and one rogue mask
    // can reposition it. Cap Σ at Σ₀ (Ornstein-Uhlenbeck decay toward the stationary prior rather than a free
    // random walk): ageing still loosens the belief monotonically, it just saturates at "what I knew before I
    // ever saw this chair". Same term and reasoning as table_concept. See rc::ai::predict.
    bool  clamp_sigma_to_prior = true;

    // Per-frame COMMON-MODE error (shared by all points of a mask → doesn't average out; Woodbury cap in
    // the engine, so N correlated points can't collapse σ). chain_cov_* + range add via the frame.
    float common_mode_pos_std = 0.03f;
    float common_mode_yaw_std = 0.03f;

    // ── Discrete yaw-mode evidence: weight by INFORMATION, not by dwell time ─────────────────────
    // ★2026-07-27. `mixture_nll` is a MEAN per-point NLL, so a 21-point frame votes on the yaw mode
    // exactly as loudly as a 1700-point one, and the only per-frame weight was ego-motion. Measured
    // live: a chair resolved correctly at 3.5 m on ~1700 points was rotated 90° TWICE by 21- and
    // 29-point frames at 6.5 m once the robot drove away, then locked when flip_acc_ saturated.
    //
    // Two coupled fixes, both mirroring discipline the CONTINUOUS channel already has:
    //  · observability weight — the mode differential is carried ENTIRELY by the backrest (the seat
    //    is square and the legs symmetric, so every other term is invariant under a 90° rotation), so
    //    weight each frame by the backrest-attributed point mass it actually observed, saturating:
    //    w_obs = n_back/(n_back + sat). Same shape as the Woodbury cap: past `sat` points more of the
    //    same surface is redundant, because they share the frame's common-mode error.
    // ★A rate-limit alone is NOT enough — measured: bounding how FAST the estimate moves does not bound
    // where it CONVERGES, so 2000 identical biased frames still drag it all the way. Redundancy has to
    // be discounted in the EVIDENCE, not the rate. Each frame's vote is therefore
    //     w = w_egomotion · w_observability · w_novelty
    // and accumulated as a SUM (a real log-odds accumulation). w_novelty = 1/(1+visits to this
    // viewpoint bin) makes re-observing one viewpoint grow evidence only as log(n): a stationary robot
    // converges to roughly one viewpoint's worth of evidence, while orbiting the chair genuinely
    // accumulates. This is the discrete analogue of the continuous channel's common-mode saturation —
    // correlated observations must not multiply into confidence.
    bool  mode_obs_weighting  = true;   // false → the pre-2026-07-27 unweighted running sum (A/B)
    float mode_sat_back_pts   = 60.0f;  // backrest mass at which a frame counts as half-informative
    // Total mode-evidence weight ONE bearing bin may ever contribute, approached asymptotically. This is the
    // "a fixation is worth one good look, not zero and not infinity" constant: a sustained stare converges to
    // this much (decisive — flip_acc_ runs to ±kFlipClamp = 6), staring past it adds ~nothing, and a NEW
    // bearing opens a fresh budget. Replaces the old 1/(1+visits) divisor, under which a viewpoint's total
    // contribution grew only as log(n) and a 3-minute deliberate fixation could not settle a 45° error.
    float view_budget         = 3.0f;
    static constexpr int kViewBins = 24;   // 15° azimuth bins around the chair

    // Optimiser
    int   gn_iters = 4;
    float fd_eps   = 1e-3f;
};

// One fitted frame's evidence (room-frame points + per-point measurement variance R). chain_cov_* carry
// the SHARED pose-chain + static-range variance the fitter computes (added to the common-mode).
struct ChairFrame
{
    std::vector<Eigen::Vector3f> points;
    std::vector<float>           R;
    float chain_cov_xx = 0.0f;
    float chain_cov_yy = 0.0f;
    float chain_cov_yaw = 0.0f;
};

class ChairBelief
{
public:
    static constexpr int N = 3;
    using State = ChairBeliefState;
    using Frame = ChairFrame;

    ChairBelief() = default;
    ChairBelief(const ChairBeliefState& s, const ChairBeliefParams& p) : state_(s), params_(p)
    { Sigma_.setZero(); Sigma_.diagonal() = prior_cov_diag(); }

    const ChairBeliefState&            state()      const { return state_; }
    const Eigen::Matrix<float, 3, 3>&  covariance() const { return Sigma_; }
    const ChairBeliefParams&           params()     const { return params_; }
    void set_state(const ChairBeliefState& s) { state_ = s; }
    void set_params(const ChairBeliefParams& p) { params_ = p; }
    // Seed a BEARING-ONLY hypothesis (Part C-birth, RICOH_360_PERIPHERAL_DETECTION.md): a peripheral 360
    // detection gives a direction but no range, so make Σ huge ALONG the ray (robot→object) and tight
    // ACROSS it (the bearing IS known), with yaw fully uncertain. The caller places the MEAN at a nominal
    // range on the ray; a subsequent depth look collapses the along-ray block (or the instance dies). No
    // range is assumed — the unknown range lives honestly in the covariance.
    void seed_bearing(const Eigen::Vector2f& robot_xy, float azimuth,
                      float along_std, float across_std, float yaw_std)
    {
        (void) robot_xy;   // mean placement is the caller's job; Σ shape only depends on the ray direction
        const Eigen::Vector2f a(std::cos(azimuth), std::sin(azimuth)), c(-a.y(), a.x());
        Sigma_.setZero();
        Sigma_.block<2, 2>(0, 0) = along_std * along_std * (a * a.transpose())
                                 + across_std * across_std * (c * c.transpose());
        Sigma_(2, 2) = yaw_std * yaw_std;
    }
    // Fixed template dims (for the fitter write-back / scene-graph publish) + the pinned floor z.
    float seat_w() const { return params_.tpl_seat_w; }
    float seat_d() const { return params_.tpl_seat_d; }
    float seat_h() const { return params_.tpl_seat_h; }
    float back_h() const { return params_.tpl_back_h; }
    float cz()     const { return params_.floor_z; }

    // ── Inference (delegated to the shared engine) ────────────────────────────
    float update(const ChairFrame& frame) { return ai::update<N>(*this, state_, Sigma_, prior_mean_, frame); }
    void  predict()
    { ai::predict<N>(*this, Sigma_, state_, prior_mean_, 1.0f, params_.clamp_sigma_to_prior); }
    // Age the belief with NO measurement: Σ ← FΣFᵀ + Q·(dt/dt_nominal), mean held, capped at the prior. The
    // fitter calls this when a chair's mask stream is stale/dead so Σ grows on the agent's clock instead of
    // freezing — but saturates at Σ₀ so ageing never erases a converged chair (see rc::ai::predict).
    void  inflate_for_age(float dt_s, float dt_nominal_s)
    { ai::inflate_for_age<N>(*this, Sigma_, state_, prior_mean_, dt_s, dt_nominal_s, params_.clamp_sigma_to_prior); }
    Eigen::Matrix<float, 3, 3> predicted_information(const std::vector<Eigen::Vector3f>& pts, float R) const
    { return ai::predicted_information<N>(*this, state_, pts, R); }

    // ── Generative-model hooks (engine interface + SDF API) ───────────────────
    float sdf_seat(const Eigen::Vector3f& p, const ChairBeliefState& s) const;
    float sdf_back(const Eigen::Vector3f& p, const ChairBeliefState& s) const;
    float sdf_leg (const Eigen::Vector3f& p, const ChairBeliefState& s, int k) const;
    float sdf_compound(const Eigen::Vector3f& p, const ChairBeliefState& s) const;
    float sdf_prim(const Eigen::Vector3f& p, const ChairBeliefState& s, int prim) const;  // 0=seat,1=back,2..5=legs
    Eigen::Matrix<float, 3, 1> sdf_jacobian(const Eigen::Vector3f& p, const ChairBeliefState& s, int prim) const;
    std::array<float, 7> responsibilities(const Eigen::Vector3f& p, const ChairBeliefState& s, float R) const;
    // Mean per-point mixture NEGATIVE LOG-LIKELIHOOD of `pts` (includes clutter) — the multi-instance
    // ASSOCIATION evidence. A far / wrong-instance slice (all points → flat clutter) → tiny likelihood →
    // HIGH nll → not claimed (unlike mean_energy, which scores it ~0).
    float association_nll(const std::vector<Eigen::Vector3f>& pts, float R) const;
    // Mean per-point mixture NLL at an ARBITRARY state — the proper likelihood used to compare yaw modes.
    // Unlike mean_energy it INCLUDES the clutter term, so an orientation that points a primitive at empty
    // space (all those points → clutter) is correctly penalised instead of rewarded (the escape valve that
    // let resolve_orientation flip a chair's backrest 180° onto empty space).
    float mixture_nll(const std::vector<Eigen::Vector3f>& pts, const ChairBeliefState& s, float R) const;
    // Mean per-point data energy (responsibility-weighted SDF), at an ARBITRARY state — for the yaw-mode test.
    float mean_energy(const std::vector<Eigen::Vector3f>& pts, const ChairBeliefState& s, float R) const;
    // Mean clutter responsibility over the frame's points (diagnostic; off-model / contamination fraction).
    float clutter_fraction(const std::vector<Eigen::Vector3f>& pts, float R) const;
    // Yaw disambiguation as sequential Bayesian model comparison over the 4 discrete orientation modes
    // {0,π/2,π,3π/2}: accumulate each rotated mode's mean free energy relative to the current one and adopt
    // the lowest-accumulated at the zero crossing. A backrest-blind frame scores all four ≈equal → no
    // chatter (the seat+legs are symmetric); only real backrest evidence moves it. No threshold. Returns
    // true iff it flipped. (Size is fixed, so a rotation is a pure yaw change — no w↔d swap.)
    // evidence_weight ∈ [0,1] scales this frame's contribution to the mode accumulator — the fitter passes a
    // low value while the robot is moving (a smeared/split mask must not vote on the discrete orientation).
    // `view_azimuth` = room-frame bearing from the chair to the CAMERA (rad). Supplying it enables the
    // novelty discount: repeated frames from one viewpoint stop accumulating evidence. NaN (the
    // default) keeps every frame maximally novel — correct for callers with no viewpoint signal.
    bool  resolve_orientation(const std::vector<Eigen::Vector3f>& pts, float R, float evidence_weight = 1.0f,
                              float view_azimuth = std::numeric_limits<float>::quiet_NaN());

    // ── Discrete orientation-mode uncertainty (report the ambiguity the 3×3 Σ can't hold) ─────────────
    // Σ_ carries only the WITHIN-mode yaw width (~1°). While the 4-way accumulator flip_acc_ is still near
    // tied the orientation is genuinely ambiguous, so the REPORTED yaw variance folds in the discrete-mode
    // entropy: a side-on chair (backrest unseen → all four modes ≈equal) advertises an honest large σ_yaw
    // that a converged one collapses. Consumed by the RT-edge cov upload and the NBV planner. Mirrors
    // TableBelief::{mode_posterior,yaw_marginal_var,covariance_reported} (TABLE.md §4.2), generalised to 4 modes.
    // ── Rig (level-2 arrangement) yaw prior ───────────────────────────────────────────────────────
    // A structural prior published by ring_metaconcept on the group_member edge: "a chair at this
    // spot faces the table". A square seat makes yaw 4-fold ambiguous and the thin backrest often
    // cannot break it — observed live 2026-07-26, a chair converged to the correct heading, snapped
    // 90° at one frame and then held the WRONG mode for 570 cycles, escaping only on a process
    // restart. That is a sticky discrete failure the chair cannot fix from its own mask, which is
    // exactly what an arrangement prior is for.
    //
    // κ = 0 (the default) ⇒ inert; every expression below reduces exactly to the pre-rig behaviour,
    // so an agent running without the rig is unaffected.
    //
    // ★The prior enters at COMPARISON time, never into flip_acc_. flip_acc_ ACCUMULATES across
    // frames; adding a constant prior to it every frame would compound the same evidence into
    // certainty within seconds — the same defect that pinned the rig's own log-odds before it was
    // made a low-pass. Data accumulates, the prior does not.
    // ── The two constants that bound how far a structural prior may go ────────────────────────────
    // kFlipClamp bounds the DATA accumulator (±nats) so a long confident run can still recant a real
    // rotation. kRigKappaMax bounds the PRIOR. The relation:
    //   · flip_acc_[0] ≡ 0 by construction (mode 0 IS the baseline), so a rival mode's data evidence
    //     runs over [−kFlipClamp, +kFlipClamp] and MAXIMALLY CONFIDENT data differs by kFlipClamp;
    //   · a 90° mode error gives the prior a differential of exactly κ (−κcos0 vs −κcos90°).
    // So the rule is: the prior overturns any preference that is NOT saturated, and always loses to
    // one that is. κ sits just below the clamp; the 0.9 margin exists only so the saturated case is
    // decided by the inequality rather than by floating-point equality.
    //
    // ★Both bounds are measured, not guessed. 2026-07-26: two chairs sat one mode wrong with flip_acc_
    // favouring the WRONG mode by ≈4.19 nats (read off std_yaw_rep=0.192 vs std_yaw=0.023) — a cap of
    // 3.96 lost to that by a quarter of a nat and the chairs stayed wrong; a cap of 6.00 tied with
    // saturated evidence and rotated a correctly-seen chair. 5.4 clears both.
    //
    // ⚠The usable window is only (4.19, 6.0) wide, which is narrow. It is narrow because flip_acc_
    // reaches 4.19 nats on a backrest-poor view at all — see [[chair-flip-acc-repetition-defect]]:
    // it SUMS the same static evidence every frame, so confidence grows with dwell time rather than
    // with information. Fix that and this window opens up; until then this cap is load-bearing.
    // ★RE-DERIVED 2026-07-27 (second time — the first derivation was wrong, see below).
    // The BINDING constraint is the 180° case, not the 90° one: rig_mode_cost = −κ·cos(Δ) gives a
    // differential of κ at 90° but 2κ at 180°, so a prior pointing 180° away rotates a chair whose
    // data is saturated unless 2κ < kFlipClamp, i.e. κ < kFlipClamp/2. Take half that again for margin.
    //
    // This was IMPOSSIBLE to satisfy before the mode evidence was information-weighted: flip_acc_ then
    // measured dwell time, so an AMBIGUOUS chair also sat at the ±6 clamp and overturning it needed
    // κ > 4.19 while 180° safety needed κ ≤ 3 — an empty window. With the novelty/observability
    // weighting the two cases separate cleanly (measured): a well-seen chair converges to 6.0, a
    // backrest-blind one to 0.21 even after 2000 frames. The admissible window is now (0.2, 3.0), and
    // κ = 1.5 sits a factor ~7 above the ambiguous case and a factor 2 below the saturated one.
    static constexpr float kFlipClamp   = 6.0f;   // ≈ 400:1 mode odds — decisive but recoverable
    static constexpr float kRigKappaMax = 0.25f * kFlipClamp;   // = 1.5

    void  set_rig_yaw_prior(float yaw_rad, float kappa) { rig_yaw_prior_ = yaw_rad; rig_yaw_kappa_ = std::max(0.0f, kappa); }
    void  clear_rig_yaw_prior() { rig_yaw_kappa_ = 0.0f; }
    float rig_yaw_kappa() const { return rig_yaw_kappa_; }
    // von-Mises log-density (up to a constant) of mode k under the rig prior — LOWER is better, so it
    // adds directly to the flip_acc_ scale.
    float rig_mode_cost(int k) const;

    // Raw per-mode accumulated DATA evidence (relative to the current mode, which is ≡0). Exposed for
    // diagnostics: it is the quantity the rig prior competes against, so when a prior fails to flip a
    // chair this is the number that says whether it was out-voted or never arrived.
    const std::array<float, 4>& flip_acc() const { return flip_acc_; }
    // Mode-evidence budget already spent per bearing bin. Exposed for diagnostics because it is the
    // quantity that explains a COMPLETED visit which resolved nothing: the belief correctly refuses to
    // gain confidence from a viewpoint it has already exhausted, but the epistemic planner does not know
    // that and can keep proposing the same bearing. max_view_spent() is the worst-case exhaustion.
    const std::array<float, ChairBeliefParams::kViewBins>& view_spent() const { return view_spent_; }
    // ── What a LOOK FROM A GIVEN BEARING is actually worth, for the epistemic planner ──────────────
    // The chair's remaining uncertainty is largely DISCRETE — which of four ways it faces — and Sigma_
    // carries only the within-mode width, so a planner scoring candidates on Sigma alone is blind to it.
    // These two make the discrete part plannable.
    //
    // mode_entropy(): Shannon entropy (nats) of the 4-mode posterior. 0 = resolved, ln4 = 1.386 = no idea.
    // It is the information a look that FULLY resolved the mode would deliver.
    float mode_entropy() const
    {
        const auto p = mode_posterior();
        float H = 0.f;
        for (float pk : p) if (pk > 1e-6f) H -= pk * std::log(pk);
        return H;
    }

    // view_novelty(): how much mode evidence a frame from `view_azimuth` can still carry, in [0,1). This
    // is the SAME quantity the accumulator charges itself (novelty = remaining/(remaining+budget)), so the
    // planner and the belief cannot disagree about whether a bearing is exhausted. Without it the planner
    // keeps proposing a bearing whose budget is spent: the robot drives there, frame_w ~ 0, the mode does
    // not move, the gain stays high, and the same viewpoint is proposed again — a completed visit that
    // resolves nothing, forever.
    float view_novelty(float view_azimuth) const
    {
        constexpr int B = ChairBeliefParams::kViewBins;
        const float span = 2.0f * static_cast<float>(M_PI) / static_cast<float>(B);
        const float wrapped = std::atan2(std::sin(view_azimuth), std::cos(view_azimuth));
        int bin = static_cast<int>(std::floor((wrapped + static_cast<float>(M_PI)) / span));
        bin = std::clamp(bin, 0, B - 1);
        const float budget    = std::max(1e-3f, params_.view_budget);
        const float remaining = std::max(0.0f, budget - view_spent_[bin]);
        return remaining / (remaining + budget);
    }

    float max_view_spent() const
    { float m = 0.f; for (float v : view_spent_) m = std::max(m, v); return m; }
    std::array<float, 4> mode_posterior() const;            // p_k over the 4 orientation modes = softmax(−flip_acc_)
    float yaw_marginal_var() const;                         // Σ(2,2) + Var_k[Δ_k] under the mode posterior (rad²)
    Eigen::Matrix<float, 3, 3> covariance_reported() const; // Σ with the yaw marginal folded into (2,2)
    void  apply_constraints(ChairBeliefState& s) const;
    void  canonicalize(ChairBeliefState&) const {}   // chair has a front (backrest) → no symmetry fold
    // ★NO accumulate_extra. The rig prior was briefly wired in as a continuous GN factor on yaw; it was
    // removed 2026-07-27 for two independent reasons. (1) It was INERT: κ≈1.5 against a data information
    // of ~10⁶, and the engine's Woodbury cap saturates Ieff at Σc⁻¹≈1100 regardless — measured effect on
    // σ_yaw was 0.016870→0.016853. (2) It was WRONG: the engine calls the hook on every update(), so the
    // same external prior was re-added every frame — precisely the compounding the discrete side takes
    // care to avoid. The ambiguity this prior exists to resolve is DISCRETE (which of 4 modes), so it acts
    // where it belongs, in the mode comparison. Reinstating a continuous term needs once-per-message
    // application, not a per-frame hook.

    int   gn_iters() const { return params_.gn_iters; }
    int   n_prims()  const { return 6; }             // seat + backrest + 4 legs (clutter is the +1)
    float sigma2()   const { return params_.sigma_base_m * params_.sigma_base_m; }
    Eigen::Matrix<float, 3, 3> transition() const { return Eigen::Matrix<float, 3, 3>::Identity(); }  // static
    Eigen::Matrix<float, 3, 1> process_noise_diag() const;
    Eigen::Matrix<float, 3, 1> prior_cov_diag() const;
    Eigen::Matrix<float, 3, 1> common_mode_inv_diag(const ChairFrame& frame) const;

    static bool self_test();

private:
    // Unnormalized mixture terms [seat, back, leg0..3, clutter] at p (likelihood×prior). Shared by
    // responsibilities() (normalizes) and association_nll() (sums).
    std::array<float, 7> mixture_unnorm(const Eigen::Vector3f& p, const ChairBeliefState& s, float R) const;
    Eigen::Vector2f leg_center_local(int k) const;
    float leg_half_height() const { return 0.5f * std::max(0.0f, params_.tpl_seat_h - params_.seat_thickness); }

    // Accumulated free-energy of each of the 4 orientation modes relative to the current one (flip_acc_[0]≡0).
    std::array<float, 4>       flip_acc_ = {0.f, 0.f, 0.f, 0.f};
    // Visits per viewpoint bin — drives the novelty discount (see mode_obs_weighting).
    std::array<std::uint32_t, ChairBeliefParams::kViewBins> view_visits_{};
    // Mode-evidence weight already SPENT from each bearing bin's budget (see ChairBeliefParams::view_budget).
    // This, not the raw visit count, is what throttles a repeated viewpoint.
    std::array<float, ChairBeliefParams::kViewBins>         view_spent_{};
    // Rig arrangement prior on yaw (κ = 0 ⇒ absent/inert). NOT accumulated — see set_rig_yaw_prior.
    float                      rig_yaw_prior_ = 0.0f;
    float                      rig_yaw_kappa_ = 0.0f;
    ChairBeliefState           state_;
    ChairBeliefParams          params_;
    Eigen::Matrix<float, 3, 3> Sigma_ = Eigen::Matrix<float, 3, 3>::Identity();
    Eigen::Matrix<float, 3, 1> prior_mean_ = Eigen::Matrix<float, 3, 1>::Zero();
};

}  // namespace rc
