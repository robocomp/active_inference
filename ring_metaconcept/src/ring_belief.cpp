/*
 * ring_belief.cpp — the ring-arrangement generative model (level-2 AI2).
 *
 * Only the MODEL hooks live here; the Bayesian machinery is the shared rc::ai engine. See the
 * header for the level-1 ↔ level-2 correspondence and for why the slot mixture's 1/Nslots prior
 * is the Occam factor rather than a tuned penalty.
 */

#include "ring_belief.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <print>

namespace rc
{

namespace
{
constexpr float kTwoPi = 6.28318531f;
constexpr float kPi    = 3.14159265f;

inline float wrap_pi(float a)
{
    while (a >   kPi) a -= kTwoPi;
    while (a <= -kPi) a += kTwoPi;
    return a;
}
// Wrap into [0, period) — the ring's fundamental domain.
inline float wrap_period(float a, float period)
{
    if (period <= 0.0f) return a;
    a = std::fmod(a, period);
    return a < 0.0f ? a + period : a;
}
}   // namespace

void RingBelief::set_slots(int n)
{
    slots_ = std::clamp(n, kSlotHypotheses.front(), kMaxRingSlots);
}

// ─── Geometry ────────────────────────────────────────────────────────────────

Eigen::Vector2f RingBelief::slot_xy(const RingBeliefState& s, int k) const
{
    const float th = s.yaw + kTwoPi * static_cast<float>(k) / static_cast<float>(slots_);
    return {s.cx + s.radius * std::cos(th), s.cy + s.radius * std::sin(th)};
}

float RingBelief::slot_inward_direction(const RingBeliefState& s, int k) const
{
    const float th = s.yaw + kTwoPi * static_cast<float>(k) / static_cast<float>(slots_);
    return wrap_pi(th + kPi);   // from the slot, pointing back at the centre
}

float RingBelief::slot_facing_yaw(const RingBeliefState& s, int k) const
{
    return wrap_pi(slot_inward_direction(s, k) + params_.member_yaw_offset);
}

float RingBelief::facing_yaw_for(const Eigen::Vector2f& member_xy) const
{
    const Eigen::Vector2f d(state_.cx - member_xy.x(), state_.cy - member_xy.y());
    return wrap_pi(std::atan2(d.y(), d.x()) + params_.member_yaw_offset);
}

// ψ = atan2(cy−py, cx−px) + const.  With e = c − p and d² = ‖e‖²:
//     ∂ψ/∂c = (−e_y, e_x)/d²        ∂ψ/∂p = ( e_y, −e_x)/d²   (equal and opposite)
// so the GEOMETRIC part is Jᵀ(Σ_centre + Σ_member)J. Both sources of position uncertainty widen the
// heading prediction, and a member sitting almost ON the centre (d→0) has an undefined bearing — its
// variance blows up, which is the honest answer, so no clamp is needed beyond the 1/d² guard.
//
// ★The geometric part alone is NOT the whole uncertainty. It answers "where is the centre from
// here", but the prior also asserts "a chair points at the centre", and that is a tendency rather
// than a law. facing_model_std is the intrinsic spread of the arrangement, added in quadrature; it
// is what stops the rig straightening a chair that is merely pushed in askew, and it dominates once
// the members are well localised. See RingBeliefParams::facing_model_std.
float RingBelief::facing_var_for(const Eigen::Vector2f& member_xy,
                                 const Eigen::Matrix2f& member_cov_xy) const
{
    const Eigen::Vector2f e(state_.cx - member_xy.x(), state_.cy - member_xy.y());
    const float d2 = std::max(1e-6f, e.squaredNorm());
    const Eigen::Vector2f J(-e.y() / d2, e.x() / d2);

    Eigen::Matrix2f S = Sigma_.block<2, 2>(0, 0) + member_cov_xy;
    const float geometric = static_cast<float>(J.transpose() * S * J);
    const float model     = params_.facing_model_std * params_.facing_model_std;
    return std::max(1e-8f, geometric + model);
}

// ─── Generative-model hooks ──────────────────────────────────────────────────

// The "SDF" of slot k: how far the member is from that seat position. Scalar, so it drops straight
// into the engine's per-point residual slot.
float RingBelief::sdf_prim(const Eigen::Vector3f& p, const RingBeliefState& s, int prim) const
{
    const Eigen::Vector2f slot = slot_xy(s, prim);
    return (p.head<2>() - slot).norm();
}

// Analytic — a finite difference of ‖·‖ is both slower and worse-conditioned near the slot centre.
// d = ‖p − slot‖, û = (p − slot)/d, and ∂slot/∂θ gives the rest by the chain rule. At d ≈ 0 the
// direction is undefined but the residual is already zero, so the correct Jacobian is zero: the
// member contributes no gradient, only the information that it IS there.
Eigen::Matrix<float, 4, 1> RingBelief::sdf_jacobian(const Eigen::Vector3f& p,
                                                    const RingBeliefState& s, int prim) const
{
    Eigen::Matrix<float, 4, 1> J = Eigen::Matrix<float, 4, 1>::Zero();

    const float th = s.yaw + kTwoPi * static_cast<float>(prim) / static_cast<float>(slots_);
    const Eigen::Vector2f slot(s.cx + s.radius * std::cos(th), s.cy + s.radius * std::sin(th));
    const Eigen::Vector2f e = p.head<2>() - slot;
    const float d = e.norm();
    if (d < 1e-5f)
        return J;
    const Eigen::Vector2f u = e / d;

    // ∂slot/∂cx = (1,0)  ∂slot/∂cy = (0,1)
    // ∂slot/∂yaw = radius·(−sin th, cos th)      ∂slot/∂radius = (cos th, sin th)
    const Eigen::Vector2f dslot_dyaw(-s.radius * std::sin(th), s.radius * std::cos(th));
    const Eigen::Vector2f dslot_dr  ( std::cos(th),             std::sin(th));

    J(0) = -u.x();
    J(1) = -u.y();
    J(2) = -u.dot(dslot_dyaw);
    J(3) = -u.dot(dslot_dr);

    // Safety clamp (recipe §2): the yaw/radius entries scale with the radius, so a large ring plus a
    // grazing residual can spike the information. Bound the slope; it never binds in normal geometry.
    const float c = params_.jac_clamp;
    for (int i = 0; i < 4; ++i)
        J(i) = std::clamp(J(i), -c, c);
    return J;
}

// Unnormalised mixture: a 2-D Gaussian per slot carrying the uniform 1/Nslots slot prior — THE Occam
// factor — plus a flat clutter/"not-a-member" component at the room's uniform density.
std::array<float, kMaxRingSlots + 1> RingBelief::mixture_unnorm(const Eigen::Vector3f& p,
                                                                const RingBeliefState& s, float R) const
{
    std::array<float, kMaxRingSlots + 1> t{};
    t.fill(0.0f);

    // The member's OWN published variance plus the slot slack: a chair is not nailed to its seat
    // position, and a poorly-localised chair is not evidence against the ring.
    const float var  = std::max(1e-6f, R + params_.sigma_slot_m * params_.sigma_slot_m);
    const float norm = 1.0f / (kTwoPi * var);
    const float slot_prior = (1.0f - params_.clutter_frac) / static_cast<float>(slots_);

    for (int k = 0; k < slots_; ++k)
    {
        const float d = sdf_prim(p, s, k);
        t[k] = slot_prior * norm * std::exp(-0.5f * d * d / var);
    }
    t[slots_] = params_.clutter_frac / std::max(1e-3f, params_.room_area_m2);
    return t;
}

std::array<float, kMaxRingSlots + 1> RingBelief::responsibilities(const Eigen::Vector3f& p,
                                                                  const RingBeliefState& s, float R) const
{
    auto t = mixture_unnorm(p, s, R);
    float sum = 0.0f;
    for (int k = 0; k <= slots_; ++k) sum += t[k];
    if (sum <= 0.0f) { t.fill(0.0f); t[slots_] = 1.0f; return t; }
    for (int k = 0; k <= slots_; ++k) t[k] /= sum;
    return t;
}

// The ANCHOR (table) as a precision-weighted prior on the arrangement centre. It is deliberately NOT
// a mixture member: a table is not a chair and must never be able to claim a ring slot. As a GN
// factor it also gives the fit a well-conditioned centre from a single observation, which is what
// makes the circle fit converge with only two or three chairs.
void RingBelief::accumulate_extra(const RingBeliefState& s, const RingFrame& f,
                                  Eigen::Matrix<float, 4, 4>& Id, Eigen::Matrix<float, 4, 1>& bd) const
{
    if (not f.has_anchor)
        return;
    const Eigen::Vector2f r(f.anchor_xy.x() - s.cx, f.anchor_xy.y() - s.cy);
    Id.block<2, 2>(0, 0) += f.anchor_info;
    bd.head<2>()         += f.anchor_info * r;
}

void RingBelief::apply_constraints(RingBeliefState& s) const
{
    s.radius = std::clamp(s.radius, params_.min_radius_m, params_.max_radius_m);
    s.yaw    = wrap_pi(s.yaw);
}

// N-fold rotational symmetry: yaw and yaw + 2π/N describe the SAME set of slots (the labels shift by
// one). Fold into a single fundamental domain so the phase cannot drift across equivalent
// representations and so the process noise isn't spent random-walking between them.
void RingBelief::canonicalize(RingBeliefState& s) const
{
    s.yaw = wrap_period(s.yaw, kTwoPi / static_cast<float>(slots_));
}

Eigen::Matrix<float, 4, 1> RingBelief::process_noise_diag() const
{
    const auto& p = params_;
    return (Eigen::Matrix<float, 4, 1>() << p.process_std_m * p.process_std_m,
                                            p.process_std_m * p.process_std_m,
                                            p.process_std_yaw * p.process_std_yaw,
                                            p.process_std_r * p.process_std_r).finished();
}

Eigen::Matrix<float, 4, 1> RingBelief::prior_cov_diag() const
{
    const auto& p = params_;
    return (Eigen::Matrix<float, 4, 1>() << p.prior_pos_std * p.prior_pos_std,
                                            p.prior_pos_std * p.prior_pos_std,
                                            p.prior_yaw_std * p.prior_yaw_std,
                                            p.prior_radius_std * p.prior_radius_std).finished();
}

// Σc⁻¹: every member's room-frame pose is conditional on the SAME robot localisation, so that error
// is shared across the whole poll and must not be averaged away by counting members.
Eigen::Matrix<float, 4, 1> RingBelief::common_mode_inv_diag(const RingFrame& f) const
{
    const auto& p = params_;
    const float vx   = p.common_mode_pos_std * p.common_mode_pos_std + std::max(0.0f, f.chain_cov_xx);
    const float vy   = p.common_mode_pos_std * p.common_mode_pos_std + std::max(0.0f, f.chain_cov_yy);
    const float vyaw = p.common_mode_yaw_std * p.common_mode_yaw_std;
    const float vr   = p.common_mode_r_std   * p.common_mode_r_std;
    return (Eigen::Matrix<float, 4, 1>() << 1.0f / std::max(1e-9f, vx),
                                            1.0f / std::max(1e-9f, vy),
                                            1.0f / std::max(1e-9f, vyaw),
                                            1.0f / std::max(1e-9f, vr)).finished();
}

// ─── Evidence ────────────────────────────────────────────────────────────────

float RingBelief::mixture_nll(const RingFrame& f, const RingBeliefState& s) const
{
    if (f.points.empty())
        return 0.0f;
    const float sig2 = sigma2();
    double acc = 0.0;
    for (std::size_t i = 0; i < f.points.size(); ++i)
    {
        const float R = (i < f.R.size() and f.R[i] > 0.0f) ? f.R[i] : sig2;
        const auto  t = mixture_unnorm(f.points[i], s, R);
        double sum = 0.0;
        for (int k = 0; k <= slots_; ++k) sum += t[k];
        acc += -std::log(std::max(1e-300, sum));
    }
    return static_cast<float>(acc / static_cast<double>(f.points.size()));
}

float RingBelief::null_nll() const
{
    // Independent objects: each member uniform over the room.
    return std::log(std::max(1e-3f, params_.room_area_m2));
}

std::array<float, kMaxRingSlots> RingBelief::slot_occupancy(const RingFrame& f,
                                                            const RingBeliefState& s) const
{
    std::array<float, kMaxRingSlots> occ{};
    occ.fill(0.0f);
    // o_k = 1 − Π_i (1 − r_i[k]): slot k is filled to the degree SOME member claims it.
    std::array<float, kMaxRingSlots> prod{};
    prod.fill(1.0f);

    const float sig2 = sigma2();
    for (std::size_t i = 0; i < f.points.size(); ++i)
    {
        const float R = (i < f.R.size() and f.R[i] > 0.0f) ? f.R[i] : sig2;
        const auto  r = responsibilities(f.points[i], s, R);
        for (int k = 0; k < slots_; ++k)
            prod[k] *= (1.0f - std::clamp(r[k], 0.0f, 1.0f));
    }
    for (int k = 0; k < slots_; ++k)
        occ[k] = 1.0f - prod[k];
    return occ;
}

// log P(occupancy pattern | a ring of `slots_` seats). Two terms:
//
//  (a) EMPTINESS, Bernoulli(q) per slot, weighted by how much of that slot the robot has actually
//      SEEN. An unobserved slot contributes nothing — emptiness is only evidence where we looked.
//      Without this the model confidently collapses onto whatever seats happened to be in view.
//
//  (b) EXCLUSIVITY — ★a seat holds AT MOST ONE chair. The per-member mixture is a plain GMM, so
//      nothing in it stops several members piling onto one fat component: a straight ROW of objects
//      then masquerades as a ring by parking pairs of them on two seats (measured: 4 collinear
//      objects scored Λ=+7.2 against the null, i.e. "definitely a ring"). The generative statement
//      is that the second occupant of a seat CANNOT have been drawn from that seat, so its mass is
//      re-priced at the clutter density it should have come from. Soft count n_k = Σᵢ rᵢ[k] is the
//      mass in slot k; o_k = 1−Πᵢ(1−rᵢ[k]) saturates at one occupant; the excess (n_k − o_k) is
//      exactly the double-booked mass, charged at the (negative) log-ratio clutter/slot.
//      Continuous, no min(), no cutoff — it vanishes identically when each seat holds one member.
//      Note (b) is NOT visibility-weighted: those members were observed, whatever the slot's
//      visibility. Only the emptiness term (a) is.
float RingBelief::occupancy_logp(const RingFrame& f, const RingBeliefState& s) const
{
    const float q  = std::clamp(params_.occupancy_q, 1e-3f, 1.0f - 1e-3f);
    const float lq = std::log(q), l1q = std::log(1.0f - q);
    const float sig2 = sigma2();

    std::array<double, kMaxRingSlots> n{}, prod{}, ratio_w{};
    n.fill(0.0); prod.fill(1.0); ratio_w.fill(0.0);

    for (std::size_t i = 0; i < f.points.size(); ++i)
    {
        const float R = (i < f.R.size() and f.R[i] > 0.0f) ? f.R[i] : sig2;
        const auto  t = mixture_unnorm(f.points[i], s, R);
        float sum = 0.0f;
        for (int k = 0; k <= slots_; ++k) sum += t[k];
        if (sum <= 0.0f) continue;

        for (int k = 0; k < slots_; ++k)
        {
            const double r = std::clamp(t[k] / sum, 0.0f, 1.0f);
            n[k]    += r;
            prod[k] *= (1.0 - r);
            // log(clutter density / this slot's density at this member) — how much worse it is to
            // call this member clutter. Always ≤ 0 for a member the slot explains well.
            // Guard in DOUBLE: a float underflow guard here silently truncates to zero (1e-300f == 0),
            // which is exactly the division-by-zero it was meant to prevent.
            const double clut = std::max(1e-300, static_cast<double>(t[slots_]));
            const double slot = std::max(1e-300, static_cast<double>(t[k]));
            ratio_w[k] += r * std::log(clut / slot);
        }
    }

    double acc = 0.0;
    for (int k = 0; k < slots_; ++k)
    {
        const float vis = (static_cast<std::size_t>(k) < f.slot_visibility.size())
                              ? std::clamp(f.slot_visibility[k], 0.0f, 1.0f)
                              : 1.0f;   // empty vector ⇒ assume fully visible
        const double o = 1.0 - prod[k];
        acc += vis * (o * lq + (1.0 - o) * l1q);                       // (a) emptiness

        const double excess = std::max(0.0, n[k] - o);                 // (b) exclusivity
        if (excess > 0.0 and n[k] > 1e-9)
            acc += excess * (ratio_w[k] / n[k]);                       // mean log-ratio, ≤ 0
    }
    return static_cast<float>(acc);
}

float RingBelief::log_evidence(const RingFrame& f, const RingBeliefState& s) const
{
    if (f.points.empty())
        return 0.0f;
    // Total (not mean) member log-likelihood, so more corroborating members = more evidence, plus
    // the occupancy term which is already a total over slots.
    const float members = -mixture_nll(f, s) * static_cast<float>(f.points.size());
    return members + occupancy_logp(f, s);
}

float RingBelief::instant_log_odds(const RingFrame& f) const
{
    if (f.points.empty())
        return 0.0f;
    const float ring = log_evidence(f, state_);
    const float null = -null_nll() * static_cast<float>(f.points.size());
    return ring - null;
}

// Low-pass, NOT a sum. The furniture is static, so successive polls are re-observations of the SAME
// configuration, not independent evidence about it. Summing per-frame log-Bayes factors let the
// estimate saturate its clamp on the first cycle and pinned p_ring at ~1 forever — confidence
// manufactured from repetition. Tracking the instantaneous evidence instead means the estimate says
// "how ring-like is the scene NOW", so it rises as members corroborate and decays when they scatter.
void RingBelief::update_log_odds(const RingFrame& f, float weight)
{
    if (f.points.empty())
        return;
    const float a = std::clamp(weight, 0.0f, 1.0f) * std::clamp(params_.evidence_ema_alpha, 1e-4f, 1.0f);
    log_odds_ = std::clamp((1.0f - a) * log_odds_ + a * instant_log_odds(f),
                           -params_.logodds_clamp, params_.logodds_clamp);
}

float RingBelief::p_ring() const
{
    return 1.0f / (1.0f + std::exp(-log_odds_));
}

// Sequential model comparison over the slot-count hypotheses, mirroring
// ChairBelief::resolve_orientation: refit a COPY per hypothesis on this frame, accumulate each
// one's evidence relative to the current hypothesis, and adopt the best at the zero crossing. A
// frame that cannot tell the hypotheses apart (two chairs 127° apart) scores them ≈equally and
// therefore does not move the accumulator — no chatter, and no threshold.
bool RingBelief::resolve_slot_count(const RingFrame& f, float weight)
{
    if (f.points.size() < 2)
        return false;

    const float w = std::clamp(weight, 0.0f, 1.0f);

    // ★Each slot count lives in its OWN phase basin, and Gauss-Newton cannot relabel slots to jump
    // between them: refitting an N=3 hypothesis from a state whose phase was converged for N=4 scores
    // N=3 in a bad phase and loses on a technicality, not on evidence. (Measured live 2026-07-26:
    // three chairs around a round table were reported as a 4-seat ring with an empty seat, even
    // though N=3 wins the evidence by ~1.5 nats once its phase is right.) The optimal phase for a
    // given N has a closed form — the circular mean of the members' bearings folded into the
    // fundamental domain [0, 2π/N) — so seed it instead of hoping the optimiser finds it.
    const auto phase_seed = [&](const RingBeliefState& s, int n) -> float
    {
        const float period = kTwoPi / static_cast<float>(n);
        double sx = 0.0, sy = 0.0;
        for (const auto& p : f.points)
        {
            const float b   = std::atan2(p.y() - s.cy, p.x() - s.cx);
            const float phi = wrap_period(b, period) * (kTwoPi / period);   // → full circle
            sx += std::cos(phi);
            sy += std::sin(phi);
        }
        if (sx == 0.0 and sy == 0.0)
            return s.yaw;
        return wrap_period(static_cast<float>(std::atan2(sy, sx)) * (period / kTwoPi), period);
    };

    // Score a hypothesis from BOTH the current state and the seeded phase, keeping whichever
    // explains the frame better. Trying both means a converged-and-good current phase is never
    // thrown away by the seed, and a hypothesis in the wrong basin always gets a fair hearing.
    const auto fit_and_score = [&](int n, RingBeliefState& out_state) -> float
    {
        float best = -std::numeric_limits<float>::infinity();
        for (int variant = 0; variant < 2; ++variant)
        {
            RingBelief trial(*this);
            trial.set_slots(n);
            if (variant == 1)
            {
                RingBeliefState s = trial.state_;
                s.yaw = phase_seed(s, n);
                trial.set_state(s);
            }
            trial.update(f);
            const float e = trial.log_evidence(f, trial.state_);
            if (e > best) { best = e; out_state = trial.state_; }
        }
        return best;
    };

    RingBeliefState cur_state;
    const float e_cur = fit_and_score(slots_, cur_state);

    std::array<RingBeliefState, 4> states;
    int best = 0;
    for (std::size_t j = 0; j < kSlotHypotheses.size(); ++j)
    {
        const float e = fit_and_score(kSlotHypotheses[j], states[j]);
        slot_acc_[j] = std::clamp(slot_acc_[j] + w * (e_cur - e),   // <0 ⇒ hypothesis j is better
                                  -params_.slot_acc_clamp, params_.slot_acc_clamp);
        if (slot_acc_[j] < slot_acc_[best]) best = static_cast<int>(j);
    }

    if (kSlotHypotheses[best] == slots_)
        return false;

    // Adopt, and re-baseline the accumulator around the new hypothesis so the comparison stays
    // relative to whatever is current (same bookkeeping as the chair's 4-mode accumulator).
    const std::array<float, 4> old = slot_acc_;
    for (std::size_t j = 0; j < slot_acc_.size(); ++j)
        slot_acc_[j] = std::clamp(old[j] - old[best], -params_.slot_acc_clamp, params_.slot_acc_clamp);

    set_slots(kSlotHypotheses[best]);
    state_ = states[best];
    return true;
}

std::array<float, 4> RingBelief::slot_count_posterior() const
{
    std::array<float, 4> p{};
    const float mn = *std::min_element(slot_acc_.begin(), slot_acc_.end());
    float sum = 0.0f;
    for (std::size_t j = 0; j < p.size(); ++j) { p[j] = std::exp(-(slot_acc_[j] - mn)); sum += p[j]; }
    for (auto& v : p) v /= std::max(1e-9f, sum);
    return p;
}

// ─── Isolated unit test (pure Eigen; run at startup and standalone) ──────────

bool RingBelief::self_test()
{
    bool ok = true;
    const auto check = [&](bool cond, const char* what)
    {
        if (not cond) { std::print("  [RingBelief::self_test] FAIL: {}\n", what); ok = false; }
    };

    // Ground truth: 4 seats of radius 0.8 around (2, 3), phase 0.3 rad.
    const float gt_cx = 2.0f, gt_cy = 3.0f, gt_r = 0.8f, gt_yaw = 0.3f;
    const int   gt_n  = 4;

    const auto make_frame = [&](int n, float radius, float phase, float jitter)
    {
        RingFrame f;
        for (int k = 0; k < n; ++k)
        {
            const float th = phase + kTwoPi * static_cast<float>(k) / static_cast<float>(n);
            // Deterministic pseudo-jitter so the test is reproducible.
            const float j1 = jitter * std::sin(11.0f * static_cast<float>(k) + 0.7f);
            const float j2 = jitter * std::cos(7.0f  * static_cast<float>(k) + 0.2f);
            f.points.emplace_back(gt_cx + radius * std::cos(th) + j1,
                                  gt_cy + radius * std::sin(th) + j2, 0.0f);
            f.R.push_back(0.01f);
        }
        f.has_anchor  = true;
        f.anchor_xy   = Eigen::Vector2f(gt_cx, gt_cy);
        f.anchor_info = Eigen::Matrix2f::Identity() * 100.0f;   // σ = 10 cm on the table centre
        return f;
    };

    // ── 1. Recover the arrangement from a perturbed start ────────────────────
    {
        const auto f = make_frame(gt_n, gt_r, gt_yaw, 0.02f);
        RingBelief b;
        b.set_slots(gt_n);
        b.set_state({gt_cx + 0.30f, gt_cy - 0.25f, gt_yaw + 0.25f, 0.55f});
        for (int i = 0; i < 25; ++i) b.update(f);

        const auto& s = b.state();
        check(std::abs(s.cx - gt_cx) < 0.06f, "centre x not recovered");
        check(std::abs(s.cy - gt_cy) < 0.06f, "centre y not recovered");
        check(std::abs(s.radius - gt_r) < 0.08f, "radius not recovered");
        // Phase is only defined modulo 2π/N (the ring's symmetry) — compare inside the fold.
        const float period = kTwoPi / static_cast<float>(gt_n);
        const float dphase = std::abs(wrap_pi(wrap_period(s.yaw, period) - wrap_period(gt_yaw, period)));
        check(std::min(dphase, period - dphase) < 0.12f, "phase not recovered (mod 2pi/N)");
    }

    // ── 2. The ring beats the independent-objects null on a real ring ────────
    {
        const auto f = make_frame(gt_n, gt_r, gt_yaw, 0.02f);
        RingBelief b;
        b.set_slots(gt_n);
        b.set_state({gt_cx, gt_cy, gt_yaw, gt_r});
        for (int i = 0; i < 10; ++i) b.update(f);
        for (int i = 0; i < 100; ++i) b.update_log_odds(f);   // EMA needs to converge, not accumulate
        check(b.log_odds() > 1.0f, "a clean ring did not out-evidence the null");
        check(b.p_ring() > 0.7f, "p_ring too low on a clean ring");
    }

    // ── 3. ★A ROW of furniture is NOT a ring — the null must win ─────────────
    // The "don't force structure onto a random scene" guarantee. Four objects in a straight line,
    // evenly spaced: a circle can be fitted through some of them, so this is exactly the case a
    // residual-only score gets wrong.
    {
        RingFrame f;
        for (int k = 0; k < 4; ++k)
        {
            f.points.emplace_back(gt_cx - 0.9f + 0.6f * static_cast<float>(k), gt_cy, 0.0f);
            f.R.push_back(0.01f);
        }
        RingBelief b;
        b.set_slots(4);
        b.set_state({gt_cx, gt_cy, 0.0f, gt_r});
        for (int i = 0; i < 10; ++i) b.update(f);
        for (int i = 0; i < 100; ++i) b.update_log_odds(f);   // EMA needs to converge, not accumulate
        check(b.log_odds() < 0.0f, "a straight ROW was mistaken for a ring");
    }

    // ── 4. Slot count chosen by evidence ─────────────────────────────────────
    {
        const auto f = make_frame(gt_n, gt_r, gt_yaw, 0.02f);
        RingBelief b;
        b.set_slots(8);                                    // start on the WRONG (over-slotted) one
        b.set_state({gt_cx, gt_cy, gt_yaw, gt_r});
        for (int i = 0; i < 20; ++i) { b.update(f); b.resolve_slot_count(f); }
        check(b.slot_count() == gt_n, "slot count not resolved to the true N");
    }

    // ── 4b. ★THE LIVE CASE: three chairs round a table must not read as a 4-seat ring ──
    // Regression for the 2026-07-26 run. Real chairs are not evenly spaced: the measured bearings
    // were 356°/235°/141°, i.e. gaps of 121°/145°/94° around a nominal 120°. On residual alone N=4
    // fits that better, and the selector duly reported a 4-seat ring with a phantom empty seat. N=3
    // wins once the empty-seat penalty and the 1/N dilution are counted — but only if each
    // hypothesis is scored in its OWN phase basin, which is what phase_seed fixes.
    {
        RingFrame f;
        const std::array<float, 3> bearings_deg{356.0f, 235.0f, 141.0f};   // as measured live
        for (const float b : bearings_deg)
        {
            const float th = b / 57.29578f;
            f.points.emplace_back(gt_cx + 0.587f * std::cos(th), gt_cy + 0.587f * std::sin(th), 0.0f);
            f.R.push_back(0.008f);
        }
        f.has_anchor  = true;
        f.anchor_xy   = Eigen::Vector2f(gt_cx, gt_cy);
        f.anchor_info = Eigen::Matrix2f::Identity() * 100.0f;

        RingBelief b;
        b.set_slots(4);                                    // the WRONG answer the live run gave
        b.set_state({gt_cx, gt_cy, 1.117f, 0.587f});       // and the phase it was sitting in
        for (int i = 0; i < 20; ++i) { b.update(f); b.resolve_slot_count(f); }
        check(b.slot_count() == 3, "three unevenly-spaced chairs still read as a 4-seat ring");
    }

    // ── 5. The facing prior is honest, and in the member's convention ────────
    {
        RingBelief b;
        b.set_slots(gt_n);
        b.set_state({gt_cx, gt_cy, gt_yaw, gt_r});

        // A member due EAST of the centre must be told to face WEST — expressed in the member's own
        // yaw convention (chair faces its local +y ⇒ offset −π/2), i.e. π + (−π/2) = π/2.
        const Eigen::Vector2f east(gt_cx + gt_r, gt_cy);
        const float psi = b.facing_yaw_for(east);
        check(std::abs(wrap_pi(psi - (kPi + b.params().member_yaw_offset))) < 1e-3f,
              "facing prior is not in the member's yaw convention");

        // Variance must GROW with the member's own position uncertainty (a badly-localised chair
        // gets a weaker heading prior) and with distance from the centre shrinking.
        const float v_tight = b.facing_var_for(east, Eigen::Matrix2f::Identity() * 1e-4f);
        const float v_loose = b.facing_var_for(east, Eigen::Matrix2f::Identity() * 1e-1f);
        check(v_loose > v_tight, "facing variance does not grow with member uncertainty");
    }

    std::print("[RingBelief::self_test] {}\n", ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace rc
