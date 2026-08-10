/*
 * kitchen_belief.h — the level-2 RECTILINEAR-FRAME latent (CONCEPT_AGENT_RECIPE.md §6, Delta 2).
 *
 * State θ = [axis, worktop, depth] (3 DOF), shared by every unit of one kitchen:
 *   · axis    — the grid orientation, 90°-PERIODIC (a run along the wall and one across it are the
 *               same grid). Members' yaws are the evidence.
 *   · worktop — the single horizontal work plane every base unit's top lies in.
 *   · depth   — the nominal carcass depth off the wall.
 *
 * WHY THIS IS NOT THE SHARED recursive-Laplace ENGINE (a deliberate, flagged deviation from §6):
 * the engine exists to Laplace-approximate a posterior whose measurement model is nonlinear in the
 * latent — the ring's slot residual is exactly that. Here it is not. `worktop` and `depth` are
 * observed by an IDENTITY map, so their exact posterior is a closed-form precision-weighted fusion
 * that GN would reproduce in one iteration; and `axis` is a wrapped angle whose exact estimator is
 * the circular mean, which GN on a wrapped residual only approximates. The engine also exposes ONE
 * scalar `sigma2()` across all prims, and these three DOFs carry different units (rad, m, m), so it
 * could not express their noises anyway. Running it here would be more machinery for a strictly
 * worse estimate. The §6 PRINCIPLES are kept in full: precision-weighted fusion, a clutter column so
 * a non-member can opt out, evidence against an explicit null, and no thresholds.
 *
 * ★The 90° periodicity is handled by fusing in the 4θ domain (multiply the angle by 4 and the
 * periodicity becomes 2π), then refining with wrapped residuals about that circular mean. Averaging
 * raw yaws would be wrong in the obvious way: a grid at 0.3° and one at 89.6° are the SAME grid, and
 * their arithmetic mean (45°) is 45° away from both.
 *
 * Pure math — no DSR, no Qt. Validated by kitchen_belief_self_test().
 */

#pragma once

#include <cmath>
#include <cstdio>
#include <numbers>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace rc {

// Which KIND of kitchen unit this is. A kitchen has three, and they share different things:
//   Base — floor to worktop (~0.90). Shares the WORKTOP PLANE and the base depth (~0.60).
//   Tall — floor to ~1.9 (refrigerator, larder, oven column). Has NO worktop; shares the tall depth.
//   Wall — mounted at ~1.45 to ~2.1 (upper units, hood). No worktop; shares the wall depth (~0.35).
// ★Modelling only {Base, Wall} is what made the live refrigerator read membership 0.00: floor-standing
// ⇒ classified Base ⇒ its 1.85 m top scored against a 0.90 m worktop plane. A fridge IS a kitchen
// member; it simply has no worktop, exactly as an upper unit has none.
enum class KitchenTier { Base, Tall, Wall };

// One kitchen unit, as the graph-reader measured it. This is the meta-concept's MEASUREMENT: not a
// mask point but a peer concept's published posterior.
struct KitchenMember
{
    std::uint64_t id = 0;
    std::string   name;
    std::string   cls;   // object_subtype (refined by name where a producer under-specifies it)

    // ── CLASS as membership evidence ────────────────────────────────────────────────────────────
    // log[ P(class | this is a kitchen unit) / P(class | it is not) ]. A range hood essentially only
    // occurs in a kitchen; a plain "cabinet" occurs in bathrooms, bedrooms and hallways too and is
    // therefore nearly uninformative on its own. Folded into the mixture's membership weight rather
    // than used as a set test, so an unexpected class is DOWN-weighted, never excluded.
    // ★These are AUTHORED priors (elicited, not measured). That is legitimate for a prior — unlike a
    // threshold — but they should be replaced by counts once a labelled scene corpus exists.
    float class_logodds = 0.0f;

    float yaw = 0.0f;        // room-frame heading of the run (rad)
    float var_yaw = 0.0f;    // its published variance (rad²) — from the peer's own rt_covariance
    bool  pinned = false;    // yaw inherited from the room polygon (⇒ cannot adopt our prior)

    float worktop = 0.0f;    // top of the carcass, room z (m) = RT z + height_m
    float depth   = 0.0f;    // carcass depth off the wall (m) = depth_m
    // ESTIMATION variance of this member's own worktop/depth — how well ITS agent knows them, which
    // is a different quantity from how much real kitchens vary. Not published today (cabinet_concept
    // publishes a POSE covariance only), so the worker fills a length-scaled fallback; see
    // KitchenConfig::worktop_meas_std_m. Without this term the model treats a noisy FIT as if it were
    // a precise measurement and declares a perfectly ordinary kitchen inconsistent.
    float var_worktop = 0.0f;
    float var_depth   = 0.0f;
    // false ⇒ var_worktop/var_depth are the length-scaled stand-in, not the producer's own posterior.
    // Worth surfacing: with the stand-in the frame cannot tell a badly-fitted long run from a good one.
    bool  has_size_cov = false;
    float length  = 0.0f;    // along-run extent (m) — the natural evidence weight
    KitchenTier tier = KitchenTier::Base;
    // Only a BASE unit's top is a work surface, so only a base unit is evidence about the shared
    // worktop plane — and only a base unit may receive a worktop prior.
    bool has_worktop() const { return tier == KitchenTier::Base; }

    float var_x = 0.0f, var_y = 0.0f;   // published position variance (m²) — diagnostics only at M2
    bool  has_cov = false;              // false ⇒ var_* are the wide fallbacks, not published values

    // Fallbacks used when the producer published no Σ. Deliberately WIDE: an uncalibrated member
    // should barely influence the frame, rather than dominate it with a flattering default.
    static constexpr float kNoCovVarYaw = 0.30f;   // (~31°)²
};

struct KitchenBeliefParams
{
    // Intrinsic spread of each shared-DOF hypothesis — how much genuine variation "one kitchen"
    // allows. Added in quadrature to the propagated uncertainty; this is what BOUNDS the down-prior
    // precision, so a member is never told its own well-observed value is wrong by more than the
    // model itself admits. Model spread, NOT a confidence knob.
    float axis_model_std    = 1.5f * std::numbers::pi_v<float> / 180.0f;
    float worktop_model_std = 0.02f;
    float depth_model_std   = 0.03f;

    float clutter_frac       = 0.20f;   // prior mass on "this member belongs to no frame"
    float evidence_ema_alpha = 0.05f;   // low-pass on frame-vs-null (NOT an accumulator)
    float logodds_clamp      = 8.0f;    // ±nats, so the frame can always recant
};

struct KitchenBeliefState
{
    float axis    = 0.0f;    // grid orientation, folded to [0, π/2)
    float worktop = 0.90f;   // shared work-plane height (m) — BASE units only
    // ★Depth is shared PER TIER, not globally. A base carcass is ~0.60 m deep and a wall unit ~0.35
    // (cabinet_config.h: base_depth_m=0.60, wall_depth_m=0.35), so one shared depth scores every
    // upper unit ~8σ off its own nominal and lands it on the clutter column — a legitimate kitchen
    // member reported as not-a-member. Mirrors worktop already being base-only.
    float depth       = 0.60f;   // base tier
    float depth_tall  = 0.65f;   // tall tier (fridge / larder / oven column)
    float depth_upper = 0.35f;   // wall tier
};

// Posterior spread of each DOF (variance). Kept as three scalars rather than a 3×3: the DOFs are
// physically independent here (an axis error does not inform a worktop height), and pretending to a
// full matrix we never populate would be a covariance we have not earned.
struct KitchenBeliefSigma
{
    float axis        = 1.0f;
    float worktop     = 1.0f;
    float depth       = 1.0f;
    float depth_tall  = 1.0f;
    float depth_upper = 1.0f;
};

class KitchenBelief
{
public:
    // Fold an angle into one fundamental domain of the 90°-periodic grid, [0, π/2). Two runs at right
    // angles describe the SAME grid, so this is the analogue of the ring's phase fold and the table's
    // w↔h fold: without it the estimate random-walks across equivalent representations.
    static float fold_axis(float a)
    {
        constexpr float q = 0.5f * std::numbers::pi_v<float>;
        a = std::fmod(a, q);
        return a < 0.0f ? a + q : a;
    }
    // Signed difference between two grid orientations, in (−π/4, π/4]. The largest a genuine
    // misalignment can ever be is 45°, beyond which it is the neighbouring grid axis.
    static float axis_delta(float a, float b)
    {
        constexpr float q = 0.5f * std::numbers::pi_v<float>;
        float d = std::fmod(a - b, q);
        if (d >  0.25f * std::numbers::pi_v<float>) d -= q;
        if (d <= -0.25f * std::numbers::pi_v<float>) d += q;
        return d;
    }

    KitchenBelief() = default;
    explicit KitchenBelief(const KitchenBeliefParams& p) : params_(p) {}

    const KitchenBeliefState& state()  const { return state_; }
    const KitchenBeliefSigma& sigma()  const { return sigma_; }
    const KitchenBeliefParams& params() const { return params_; }
    void set_params(const KitchenBeliefParams& p) { params_ = p; }
    bool seeded() const { return seeded_; }
    int  n_axis_members()    const { return n_axis_; }
    int  n_worktop_members() const { return n_worktop_; }
    int  n_upper_members()   const { return n_upper_; }
    int  n_tall_members()    const { return n_tall_; }
    // The shared depth that applies to a given member — one per tier.
    float depth_for(const KitchenMember& m) const
    { switch (m.tier) { case KitchenTier::Tall: return state_.depth_tall;
                        case KitchenTier::Wall: return state_.depth_upper;
                        default:                return state_.depth; } }
    float depth_var_for(const KitchenMember& m) const
    { switch (m.tier) { case KitchenTier::Tall: return sigma_.depth_tall;
                        case KitchenTier::Wall: return sigma_.depth_upper;
                        default:                return sigma_.depth; } }

    // ── One poll: refit the shared frame from this cycle's members ───────────────────────────────
    // `axis_members` is the subset whose yaw is polygon-pinned when the structural cavity is on (see
    // KitchenConfig::axis_from_pinned_only) — those cannot have adopted a prior from us, so scoring
    // them cannot manufacture confidence. Returns false when there was nothing to fit.
    bool update(const std::vector<KitchenMember>& members, const std::vector<KitchenMember>& axis_members);

    // ── Predictions = the DOWN-PRIORS (what M3 would publish) ───────────────────────────────────
    // For member i, computed from the frame fitted WITHOUT i (leave-one-out). Never send a node the
    // message it sent up — see recipe §6 Delta 3.
    struct MemberPrior
    {
        float yaw   = 0.0f;   // the grid axis resolved to the member's own nearest right angle (rad)
        float kappa = 0.0f;   // its precision (rad⁻²); 0 ⇒ inert, the consumer ignores it
        float worktop = 0.0f, worktop_info = 0.0f;
        float depth   = 0.0f, depth_info   = 0.0f;
        float axis_resid = 0.0f;   // signed misalignment of the member from the frame (rad) — diagnostic
    };
    // The prior this frame would send `m`. `cavity` is the frame refitted without m; pass the full
    // frame only in diagnostics that explicitly want the self-confirming value.
    MemberPrior prior_for(const KitchenMember& m, const KitchenBelief& cavity) const;

    // Resolve the grid to the right angle NEAREST the member's own heading. The frame states an axis
    // mod 90°; a given unit runs along one specific arm of it, and which arm is a fact about that
    // member, not about the kitchen. Snapping to the wrong arm would rotate a cabinet by 90°.
    static float resolve_to_member(float axis, float member_yaw)
    { return member_yaw + axis_delta(axis, member_yaw); }

    // ── Evidence: is there a shared frame at all? ────────────────────────────────────────────────
    // Instantaneous log-Bayes factor of "one rectilinear frame" against the independent-units null,
    // for THIS poll. The null says each member's yaw/worktop/depth is drawn independently from a
    // broad range; the frame says they share three numbers. A genuinely scattered set of furniture
    // makes this negative and no prior is ever pushed.
    float instant_log_odds(const std::vector<KitchenMember>& members) const;
    void  update_log_odds(const std::vector<KitchenMember>& members);

    // Per-member MEMBERSHIP posterior: P(this unit belongs to the frame | its class AND how well it
    // fits). This is the quantity the partition question needs, and it is what decides at M3 whether
    // a member gets a group_member edge at all. Class enters as a prior, geometry as the likelihood —
    // so a hood that fits badly can still be a member, and a perfectly-aligned sideboard need not be.
    float membership_prob(const KitchenMember& m) const;
    // The membership weight used inside the mixture: the clutter prior shifted by the class evidence.
    float member_prior_weight(const KitchenMember& m) const;
    float log_odds() const { return log_odds_; }
    // Carry the existence evidence into a CAVITY copy. Frame existence is a property of the whole
    // arrangement, not of the member being left out, so re-deriving it from N−1 members would make
    // the down-prior weaker for no modelled reason.
    void  set_log_odds(float v) { log_odds_ = v; }
    // P(frame exists) — the model-averaging weight that scales EVERY down-prior precision.
    float p_frame() const { return 1.0f / (1.0f + std::exp(-log_odds_)); }

    // ── JOINT FREE ENERGY: does the coupling actually pay? ──────────────────────────────────────
    // A meta-concept adds a COUPLING term to the fleet's free energy: the elements each minimise
    // their own F_i, and this layer contributes F_couple, whose gradient w.r.t. each element IS the
    // down-prior. By this codebase's own principle (FE = −log marginal likelihood),
    //     F_couple = −[ log P(members | one frame) − log P(members | independent units) ]
    //              = −instant_log_odds
    // so a NEGATIVE F_couple means the coupled model explains the scene better than the factorised
    // one and pushing priors is justified; a POSITIVE one means the coupling would impose structure
    // the data refuses, and the frame must stay quiet.
    float free_energy(const std::vector<KitchenMember>& members) const
    { return -instant_log_odds(members); }

    // F_couple recomputed with every member moved to the value the frame would tell it to adopt (a
    // precision-weighted compromise between its own estimate and the prior — NOT the prior itself,
    // since the element keeps its own likelihood). The DIFFERENCE from free_energy() is the free
    // energy that readjustment would actually buy, which is the testable prediction: if the elements
    // adopt and the residual does not fall, this layer's model is wrong. Cavity-correct only if the
    // caller passes per-member cavity frames; passing `*this` gives the self-confirming upper bound.
    float free_energy_if_adopted(const std::vector<KitchenMember>& members) const;

    // The value member `m` would settle on for one scalar DOF after fusing the frame's prior with its
    // own estimate: the standard precision-weighted compromise, so a well-observed element barely
    // moves and a poorly-observed one moves a lot.
    static float adopted(float own, float own_var, float prior, float prior_info)
    {
        const float own_info = 1.0f / std::max(1e-9f, own_var);
        const float den = own_info + std::max(0.0f, prior_info);
        return den > 0.0f ? (own_info * own + prior_info * prior) / den : own;
    }

    static bool self_test();

private:
    // Weighted circular fusion of the members' yaws in the 4θ domain, then a wrapped-residual
    // refinement. Fills state_.axis / sigma_.axis. Returns the number of contributors.
    int fit_axis(const std::vector<KitchenMember>& members);
    // Plain precision-weighted fusion of a scalar shared DOF. Returns the contributor count.
    // ROBUST precision-weighted fusion of a shared scalar: EM against the same Gaussian-vs-uniform
    // mixture the membership posterior uses. Each member's pull on the shared value is scaled by its
    // own responsibility, recomputed from the current estimate.
    //
    // ★Why this and not a plain weighted mean: a refrigerator stands on the floor but has NO worktop,
    // so its "top" reads ~1.8 m. Live it dragged the shared worktop plane from ~0.90 to 0.968 and the
    // base depth from ~0.60 to 0.677 — and was THEN rejected at membership 0.00. An outlier must not
    // get to move the value it is about to be judged against. EM fixes that for ANY outlier without a
    // tier special-case and without a gate: the responsibility is already the model's own opinion.
    static int fit_scalar(const std::vector<float>& x, const std::vector<float>& w,
                          float prior_weight, float null_range, float& mean, float& var);

    KitchenBeliefState  state_;
    KitchenBeliefSigma  sigma_;
    KitchenBeliefParams params_;
    bool  seeded_    = false;
    int   n_axis_    = 0;
    int   n_worktop_ = 0;
    int   n_upper_   = 0;
    int   n_tall_    = 0;
    float log_odds_  = 0.0f;
};

}  // namespace rc
