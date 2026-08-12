/*
 * ring_belief.h  —  AI2 RING-ARRANGEMENT belief (level 2, on the shared rc::ai engine)
 *
 * The meta-concept analogue of a level-1 object belief (CONCEPT_AGENT_RECIPE.md §6 Delta 2). Same
 * recursive variational-Laplace filter, same hooks — only the MEANING of the terms changes:
 *
 *      level 1                              level 2 (here)
 *      ───────────────────────────────      ─────────────────────────────────────────────
 *      a "point" is a 3D mask point         a "point" is a MEMBER's room-frame position
 *      R is the sensor's per-point noise    R is that member's OWN published rt_covariance
 *      sdf_prim = distance to primitive k   sdf_prim = distance to ring SLOT k
 *      clutter column = off-model points    clutter column = "not a member of this rig"
 *
 * State θ = [cx, cy, yaw, radius] (4 DOF). Slot k sits at c + radius·(cos θk, sin θk) with
 * θk = yaw + 2πk/Nslots, and a chair occupying it FACES INWARD (θk + π) — that inward-facing
 * prediction is the whole point of the agent: it is the empirical prior pushed back down.
 *
 * Nslots is DISCRETE and is chosen by model evidence, not fitted. Two properties make that honest:
 *   · the slot mixture carries a uniform 1/Nslots prior, so a larger ring is automatically diluted
 *     (the Occam factor falls OUT of the generative model — it is not a penalty term someone tuned);
 *   · an OCCUPANCY×VISIBILITY term scores slots that are empty, but only where the robot has
 *     actually looked — an unobserved slot is uninformative, not evidence against the ring.
 *
 * Whether a rig exists at all is a running log-Bayes factor against the independent-objects null
 * (`log_odds()`), clamped so a long confident run can still RECANT when the furniture is rearranged.
 *
 * Pure Eigen — no DSR, no Qt — so it is unit-testable in isolation via self_test().
 */

#pragma once

#include <array>
#include <cstdint>
#include <vector>
#include <Eigen/Dense>

#include "../../common/ai_belief/recursive_laplace.h"

namespace rc
{

// Largest ring we model. Fixes the responsibilities() array size; the ACTIVE slot count is
// `slot_count()` (NOT `slots()` — `slots` is a Qt keyword macro), chosen by evidence among kSlotHypotheses.
inline constexpr int kMaxRingSlots = 8;

struct RingBeliefState
{
    float cx = 0.0f, cy = 0.0f, yaw = 0.0f, radius = 0.8f;

    Eigen::Matrix<float, 4, 1> vec() const
    { return (Eigen::Matrix<float, 4, 1>() << cx, cy, yaw, radius).finished(); }
    static RingBeliefState from_vec(const Eigen::Matrix<float, 4, 1>& v)
    { return {v(0), v(1), v(2), v(3)}; }
};

struct RingBeliefParams
{
    // ── Slot-occupancy observation model ──────────────────────────────────────────────
    // sigma_slot_m is the SLACK between a slot and the chair occupying it — chairs get pushed in
    // and out, so this is deliberately loose (it is not sensor noise; that arrives per-member as R).
    float sigma_slot_m  = 0.20f;
    float clutter_frac  = 0.20f;    // prior mass on "this member belongs to no rig"
    float room_area_m2  = 50.0f;    // support of the null/clutter uniform → its density is 1/area

    // ── Priors (init Σ) ───────────────────────────────────────────────────────────────
    float prior_pos_std    = 1.00f;   // centre is weakly known until the anchor or members speak
    float prior_yaw_std    = 1.81f;   // ~uniform phase over one fundamental domain
    float prior_radius_std = 0.40f;
    float prior_radius_m   = 0.80f;   // nominal seat-ring radius (a dining chair's reach)

    float min_radius_m = 0.25f, max_radius_m = 3.0f;   // physical bounds, not tuning

    // ── Temporal transition (a dining set is static furniture-of-furniture) ───────────
    float process_std_m   = 0.010f;
    float process_std_yaw = 0.020f;
    float process_std_r   = 0.010f;

    // ── Per-frame COMMON-MODE (shared error across all members of one poll) ───────────
    // Every member's room-frame pose is conditional on the SAME robot localisation, so a
    // localisation wobble displaces them together and must not be averaged away by counting
    // members. Woodbury caps the frame's authority at this covariance (see the engine).
    float common_mode_pos_std = 0.05f;
    float common_mode_yaw_std = 0.05f;
    float common_mode_r_std   = 0.05f;

    // ── Member yaw CONVENTION (★verified against live data 2026-07-26) ────────────────
    // The rig predicts a facing DIRECTION (inward, toward the centre). A member's published `yaw`
    // is the rotation of its own local frame, which need not point along that direction — so the
    // down-prior must be converted into the member's convention or it is silently off by a constant.
    //
    // ChairBelief puts the backrest on the −y seat edge, so a chair FACES its local +y:
    //     facing_direction = chair_yaw + π/2   ⇒   chair_yaw = facing_direction − π/2
    // Confirmed on the live dining set: with this offset chair_1 and chair_2 sit 0.55° and 2.65°
    // from perfectly facing the table; without it both look ~90° wrong. Publishing the raw facing
    // direction as a yaw prior would have rotated every chair 90° — the same class of bug as the
    // chair.obj backrest flip, one level up. Re-derive this per member class before reusing.
    float member_yaw_offset = -1.5707963f;   // added to the inward direction → member-frame yaw

    // ── Facing-prior model error ──────────────────────────────────────────────────────
    // ★The rig predicts "a chair faces the table". That is a TENDENCY, not a law: a chair pushed in
    // at an angle still belongs to the set. Propagating only Σ_centre + Σ_member would treat the
    // prediction as exact and hand the consumer a precision it has not earned — measured live
    // 2026-07-26, that gave κ≈87 for a chair sitting a perfectly ordinary 16° askew, easily hard
    // enough to straighten furniture that was never crooked. This is the intrinsic spread of the
    // arrangement itself, added in quadrature, and it is what bounds how hard the rig can ever push.
    // Live estimate from three real chairs: residuals of −7°, +16° ⇒ ~10–15°.
    float facing_model_std = 0.21f;   // rad (~12°)

    // ── Discrete structure ────────────────────────────────────────────────────────────
    float occupancy_q      = 0.70f;   // P(a slot of a real rig is occupied)
    float logodds_clamp    = 8.0f;    // ±nats on the ring-vs-null estimate → it can always recant
    float slot_acc_clamp   = 8.0f;    // ±nats on the slot-count accumulator (same reason)
    // ★Rate at which the ring-vs-null estimate tracks the instantaneous evidence. This is a LOW-PASS,
    // not an accumulator: the furniture is static, so re-observing the same three chairs 480 times is
    // ONE observation, not 480 independent ones. Summing the per-frame log-Bayes factor (≈+12 nats
    // here) saturated the clamp on the very first cycle and pinned p_ring at 0.9997 forever, which
    // made it impossible for the belief to express doubt or to recant when the scene changes.
    float evidence_ema_alpha = 0.05f;

    // ── Optimiser ─────────────────────────────────────────────────────────────────────
    int   gn_iters   = 5;
    float fd_eps     = 1e-3f;
    float jac_clamp  = 10.0f;   // ★clamp the finite-diff slope (recipe §2) — a member sitting exactly
                                //  on a slot centre makes d‖·‖/dθ blow up and spikes the information
};

// One poll of the graph, as evidence. `points` are the RING members' room xy packed as (x, y, 0);
// `R` is each member's own published position variance. The ANCHOR (the table) is NOT a point — it
// enters through accumulate_extra as a precision-weighted prior on the centre, so the slot mixture
// stays purely about ring members and the anchor cannot be mistaken for a chair.
struct RingFrame
{
    std::vector<Eigen::Vector3f> points;
    std::vector<float>           R;

    bool            has_anchor  = false;
    Eigen::Vector2f anchor_xy   = Eigen::Vector2f::Zero();
    Eigen::Matrix2f anchor_info = Eigen::Matrix2f::Zero();   // information (Σ⁻¹) of the anchor position

    // Shared pose-chain uncertainty for this poll (added to the common-mode, not to per-member R).
    float chain_cov_xx = 0.0f;
    float chain_cov_yy = 0.0f;

    // Per-slot visibility ∈ [0,1] — how much the robot has actually been able to SEE slot k this
    // poll. Empty ⇒ treated as fully visible. Gates the occupancy term so a slot behind the robot
    // is not counted as evidence against the ring (the level-2 analogue of the tracker's
    // expected_visible negative-information rule).
    std::vector<float> slot_visibility;
};

class RingBelief
{
public:
    static constexpr int N = 4;
    using State = RingBeliefState;
    using Frame = RingFrame;

    // Slot-count hypotheses compared by evidence. 2 is deliberately absent: two members on opposite
    // sides of a centre are a facing-PAIR schema, not a ring, and admitting it here would let the
    // ring model claim every sofa-and-armchair in the room.
    static constexpr std::array<int, 4> kSlotHypotheses{3, 4, 6, 8};

    RingBelief() { Sigma_.setZero(); Sigma_.diagonal() = prior_cov_diag(); }
    RingBelief(const RingBeliefState& s, const RingBeliefParams& p) : state_(s), params_(p)
    { Sigma_.setZero(); Sigma_.diagonal() = prior_cov_diag(); }

    const RingBeliefState&            state()      const { return state_; }
    const Eigen::Matrix<float, 4, 4>& covariance() const { return Sigma_; }
    const RingBeliefParams&           params()     const { return params_; }
    void set_state (const RingBeliefState& s)  { state_ = s; }
    void set_params(const RingBeliefParams& p) { params_ = p; }

    int  slot_count() const { return slots_; }
    void set_slots(int n);

    // ── Inference (delegated to the shared engine) ────────────────────────────
    float update(const RingFrame& f) { return ai::update<N>(*this, state_, Sigma_, prior_mean_, f); }
    void  predict()                  { ai::predict<N>(*this, Sigma_, state_, prior_mean_); }
    void  inflate_for_age(float dt_s, float dt_nominal_s)
    { ai::inflate_for_age<N>(*this, Sigma_, state_, prior_mean_, dt_s, dt_nominal_s); }

    // ── Geometry / predictions ────────────────────────────────────────────────
    Eigen::Vector2f slot_xy(const RingBeliefState& s, int k) const;
    Eigen::Vector2f slot_xy(int k) const { return slot_xy(state_, k); }
    // Direction from slot k toward the centre (θk + π) — the geometric prediction, convention-free.
    float slot_inward_direction(const RingBeliefState& s, int k) const;
    // The same, in the MEMBER's yaw convention. Used for an EMPTY slot (birth seeding / NBV), where
    // there is no member position to work from.
    float slot_facing_yaw(const RingBeliefState& s, int k) const;
    float slot_facing_yaw(int k) const { return slot_facing_yaw(state_, k); }

    // ── The DOWN-PRIOR, for a member at a known position ──────────────────────
    // For an OCCUPIED slot the prior comes from where the member actually is, not where its slot
    // centre is: ψ = atan2(c − p) + offset. A chair pulled 20 cm out of its slot should be told to
    // face the table from where it stands, not to adopt the slot's heading — and this way the
    // prediction stays exact even while the phase/slot-count are still ambiguous.
    float facing_yaw_for(const Eigen::Vector2f& member_xy) const;
    // Variance of that prediction, propagated through ∂ψ/∂(c,p) from the arrangement's own Σ AND the
    // member's published position covariance. The down-prior precision is p_ring()/this — so a shaky
    // arrangement pushes softly because its prediction is genuinely wide, not via a confidence knob.
    float facing_var_for(const Eigen::Vector2f& member_xy, const Eigen::Matrix2f& member_cov_xy) const;

    // ── Generative-model hooks (the engine's interface) ───────────────────────
    float sdf_prim(const Eigen::Vector3f& p, const RingBeliefState& s, int prim) const;
    Eigen::Matrix<float, 4, 1> sdf_jacobian(const Eigen::Vector3f& p, const RingBeliefState& s, int prim) const;
    std::array<float, kMaxRingSlots + 1> responsibilities(const Eigen::Vector3f& p,
                                                          const RingBeliefState& s, float R) const;
    void accumulate_extra(const RingBeliefState& s, const RingFrame& f,
                          Eigen::Matrix<float, 4, 4>& Id, Eigen::Matrix<float, 4, 1>& bd) const;
    void apply_constraints(RingBeliefState& s) const;
    // A ring has Nslots-fold rotational symmetry: shifting yaw by 2π/N relabels the slots and leaves
    // the arrangement identical. Fold yaw into one fundamental domain so the phase cannot random-walk
    // across equivalent representations (the analogue of the table's w↔h fold).
    void canonicalize(RingBeliefState& s) const;

    int   gn_iters() const { return params_.gn_iters; }
    int   n_prims()  const { return slots_; }
    float sigma2()   const { return params_.sigma_slot_m * params_.sigma_slot_m; }
    Eigen::Matrix<float, 4, 4> transition() const { return Eigen::Matrix<float, 4, 4>::Identity(); }
    Eigen::Matrix<float, 4, 1> process_noise_diag() const;
    Eigen::Matrix<float, 4, 1> prior_cov_diag()     const;
    Eigen::Matrix<float, 4, 1> common_mode_inv_diag(const RingFrame& f) const;

    // ── Evidence: does this arrangement exist, and with how many slots? ───────
    // Mean per-member mixture NEGATIVE LOG-LIKELIHOOD at an arbitrary state, clutter INCLUDED.
    // This — not the engine's surface-only update() return — is the quantity every structural
    // decision reasons about (recipe: "free energy is the true −log marginal likelihood").
    float mixture_nll(const RingFrame& f, const RingBeliefState& s) const;
    // −log density the independent-objects null assigns a member: a uniform over the room.
    float null_nll() const;
    // Soft occupancy of each slot: 1 − Π_i(1 − r_i[k]). Slot k is "filled" to the degree SOME member
    // claims it.
    std::array<float, kMaxRingSlots> slot_occupancy(const RingFrame& f, const RingBeliefState& s) const;
    // log P(observed occupancy | ring of `slots` seats), weighted by per-slot visibility.
    float occupancy_logp(const RingFrame& f, const RingBeliefState& s) const;
    // Total log-evidence of the CURRENT hypothesis on this frame (per-member mixture + occupancy).
    float log_evidence(const RingFrame& f, const RingBeliefState& s) const;

    // Instantaneous ring-vs-null log-Bayes factor for THIS poll (not accumulated).
    float instant_log_odds(const RingFrame& f) const;
    // Track that instantaneous evidence with a low-pass, clamped. `weight` ∈ [0,1] scales this poll's
    // pull (a poll taken while the robot is moving carries smeared member poses and should count for
    // less). ★Deliberately NOT a sum — see RingBeliefParams::evidence_ema_alpha.
    void  update_log_odds(const RingFrame& f, float weight = 1.0f);
    float log_odds() const { return log_odds_; }
    void  set_log_odds(float v) { log_odds_ = v; }
    // P(ring exists) — the model-averaging weight that scales every down-prior precision.
    float p_ring() const;

    // Sequential model comparison over kSlotHypotheses, mirroring ChairBelief::resolve_orientation:
    // refit a COPY per hypothesis, accumulate each one's evidence relative to the current, adopt the
    // best at the zero crossing. Returns true iff the slot count changed.
    bool  resolve_slot_count(const RingFrame& f, float weight = 1.0f);
    std::array<float, 4> slot_count_posterior() const;

    // ★Model-compare the RADIUS against its closed form, for the same reason resolve_slot_count
    // model-compares the slot count: Gauss-Newton optimises the slot-mixture residual, and that
    // residual is NOT the objective the belief reports. As the radius shrinks the slot centres crowd
    // together, a member falls within a slot-σ of several at once, and mixture_unnorm SUMS those
    // components — so crowding manufactures likelihood that no physical seat supplies. occupancy_logp
    // opposes it and log_evidence therefore peaks in the right place, but occupancy never enters the
    // GN step, so the optimiser walks downhill in log_evidence until it hits min_radius_m.
    //
    // Measured live 2026-08-11 on a 2-chair set: GN sat on the 0.25 clamp with member→slot distances
    // of 0.221/0.254, while r = 0.45 gives 0.104/0.148 — every residual more than halved, and the
    // mixture still scored it WORSE (nll 0.972 vs 0.876). log_evidence: −4.335 vs −4.659.
    //
    // Adopts the closed-form radius (with its own closed-form phase) only when it strictly increases
    // log_evidence, so this can never make the fit worse. Returns true iff it was adopted.
    bool  resolve_radius(const RingFrame& f);

    static bool self_test();

private:
    // Unnormalised mixture terms [slot₀…slot_{Nslots−1}, clutter] at p. Shared by responsibilities()
    // (which normalises) and mixture_nll() (which sums — that sum IS the marginal likelihood).
    std::array<float, kMaxRingSlots + 1> mixture_unnorm(const Eigen::Vector3f& p,
                                                        const RingBeliefState& s, float R) const;

    // Closed-form optimal phase for an n-slot ring at state `s`: the circular mean of the members'
    // bearings folded into the fundamental domain [0, 2π/n). Shared by resolve_slot_count (where
    // each slot count needs its own basin) and resolve_radius (where the best phase depends on r).
    float phase_seed_for(const RingFrame& f, const RingBeliefState& s, int n) const;

    int                        slots_ = 4;
    RingBeliefState            state_;
    RingBeliefParams           params_;
    Eigen::Matrix<float, 4, 4> Sigma_      = Eigen::Matrix<float, 4, 4>::Identity();
    Eigen::Matrix<float, 4, 1> prior_mean_ = Eigen::Matrix<float, 4, 1>::Zero();

    float                log_odds_ = 0.0f;                    // ring vs independent-objects null
    std::array<float, 4> slot_acc_ = {0.f, 0.f, 0.f, 0.f};    // per-hypothesis accumulated evidence
};

}  // namespace rc
