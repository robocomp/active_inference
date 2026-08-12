/*
 * kitchen_belief.cpp — the rectilinear-frame latent. See kitchen_belief.h for the model and for why
 * this does not run the shared recursive-Laplace engine.
 */

#include "kitchen_belief.h"

#include <algorithm>
#include <numeric>

namespace rc {

namespace {

constexpr float kPi = std::numbers::pi_v<float>;

// Physical SUPPORTS of the independent-units null, not tuning knobs. The null says each member's
// value is drawn from the whole physically plausible range with no shared structure, so its density
// is 1/range. Getting these from physics rather than a constant is what keeps the log-Bayes factor
// from being scaled by an arbitrary number (the same reason ring_metaconcept takes its null support
// from the actual room polygon area).
constexpr float kNullAxisRange    = 0.5f * kPi;   // one full fundamental domain of a 90° grid (exact)
constexpr float kNullWorktopRange = 2.6f;         // floor to ceiling (m)
constexpr float kNullDepthRange   = 1.0f;         // 0.2 .. 1.2 m spans every plausible carcass depth

float gauss_pdf(float x, float mu, float var)
{
    const float v = std::max(1e-12f, var);
    const float d = x - mu;
    return std::exp(-0.5f * d * d / v) / std::sqrt(2.0f * kPi * v);
}

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float logit(float p)   { const float q = std::clamp(p, 1e-6f, 1.0f - 1e-6f); return std::log(q / (1.0f - q)); }

}  // namespace

// The mixture's membership weight for one member: the base prior (1 − clutter_frac) shifted in
// log-odds by the class evidence. A near-diagnostic class (a hood) pushes the weight toward 1; an
// ambiguous one (a plain cabinet) barely moves it; a class typical of OTHER rooms pushes it down.
// Never 0 or 1, so class can only ever re-weight a member — it can never exclude one outright, and a
// surprising class that fits the geometry perfectly still gets in.
float KitchenBelief::member_prior_weight(const KitchenMember& m) const
{
    const float base = 1.0f - std::clamp(params_.clutter_frac, 0.0f, 0.99f);
    return std::clamp(sigmoid(logit(base) + m.class_logodds), 1e-4f, 1.0f - 1e-4f);
}

// P(member belongs to the frame) = the mixture responsibility of the frame component against the
// clutter component, with class as the prior and geometric fit as the likelihood.
float KitchenBelief::membership_prob(const KitchenMember& m) const
{
    if (not seeded_)
        return 0.0f;
    const float w = member_prior_weight(m);
    const float vy = (m.var_yaw > 0.0f ? m.var_yaw : KitchenMember::kNoCovVarYaw)
                     + sigma_.axis + params_.axis_model_std * params_.axis_model_std;
    float lik  = gauss_pdf(axis_delta(m.yaw, state_.axis), 0.0f, vy);
    float null = 1.0f / kNullAxisRange;
    if (m.has_worktop())
    {
        lik  *= gauss_pdf(m.worktop, state_.worktop, sigma_.worktop + m.var_worktop
                          + params_.worktop_model_std * params_.worktop_model_std);
        null *= 1.0f / kNullWorktopRange;
    }
    lik  *= gauss_pdf(m.depth, depth_for(m), depth_var_for(m) + m.var_depth
                      + params_.depth_model_std * params_.depth_model_std);
    null *= 1.0f / kNullDepthRange;

    const float a = w * lik, b = (1.0f - w) * null;
    return (a + b) > 0.0f ? a / (a + b) : 0.0f;
}

// ── Shared-axis fusion in the 4θ domain ──────────────────────────────────────────────────────────
//
// Multiplying the angle by 4 turns the grid's 90° periodicity into a full 2π, so the ordinary
// weighted circular mean applies and members on PERPENDICULAR arms reinforce each other instead of
// cancelling. (Live: runs at 0.28° and 89.01° describe the same grid; averaged raw they would give
// 44.6°, which is 45° wrong.) The resultant direction is then refined with wrapped residuals so the
// answer is a proper precision-weighted mean rather than only a direction.
int KitchenBelief::fit_axis(const std::vector<KitchenMember>& members)
{
    // ★A member the frame does not believe in may not vote on the frame's axis.
    //
    // This fit used to weight purely by the peer's published 1/var_yaw, with no membership term and
    // no outlier term of any kind — unlike fit_scalar directly below, which is robust EM with a
    // median init. So a member the mixture had already expelled to clutter still steered the one DOF
    // this agent exists to estimate, at full strength. Measured 2026-08-11 (kitchen_members.csv,
    // cycle 44): `cabinet_w14_base` sat at membership 2.4e-12 — rejected over a 10 cm worktop
    // disagreement — while `pinned = 1` kept it as one of the four members driving the axis.
    //
    // The weight is now precision × responsibility: model averaging rather than a gate, so a marginal
    // member fades continuously instead of being cut at a cutoff. Same discipline as the rig's p_ring
    // scaling of its down-prior, and the same principle as
    // `hood: an identity prior may not vote on a hypothesis it does not model` (5b283f2).
    //
    // On the FIRST call there is no frame to judge membership against (`seeded_` is false and
    // membership_prob returns 0 by contract), so that pass stays unweighted to bootstrap; from then
    // on the responsibility comes from the previous cycle's converged frame — the same one-step EM
    // lag fit_scalar uses within a single call.
    //
    // No floor is applied to the responsibility on purpose. If EVERY member is rejected the weights
    // scale uniformly, which leaves the weighted mean unchanged but sends sigma_.axis (= 1/den) up —
    // the frame then reports an axis it is honestly unsure of, rather than a confident wrong one.
    const auto weight_of = [this](const KitchenMember& m) -> double
    {
        const float  var  = m.var_yaw > 0.0f ? m.var_yaw : KitchenMember::kNoCovVarYaw;
        const double prec = 1.0 / std::max(1e-9f, var);   // the peer's OWN published precision
        return seeded_ ? prec * static_cast<double>(membership_prob(m)) : prec;
    };

    double cs = 0.0, sn = 0.0, wsum = 0.0;
    int n = 0;
    for (const auto& m : members)
    {
        const double w = weight_of(m);
        cs += w * std::cos(4.0 * static_cast<double>(m.yaw));
        sn += w * std::sin(4.0 * static_cast<double>(m.yaw));
        wsum += w;
        ++n;
    }
    if (n == 0 or wsum <= 0.0 or (cs * cs + sn * sn) < 1e-18)
        return 0;

    const float coarse = static_cast<float>(std::atan2(sn, cs) / 4.0);
    // Refine: with every member now within ±45° of `coarse` the residuals are unimodal, so a plain
    // precision-weighted mean of them is exact and also yields the posterior variance of the mean.
    double num = 0.0, den = 0.0;
    for (const auto& m : members)
    {
        const double w = weight_of(m);
        num += w * static_cast<double>(axis_delta(m.yaw, coarse));
        den += w;
    }
    state_.axis  = fold_axis(coarse + static_cast<float>(num / den));
    sigma_.axis  = static_cast<float>(1.0 / den);   // variance of the MEAN; the model spread is added
    return n;                                        // in prior_for, where a member is PREDICTED
}

// Robust precision-weighted fusion of one shared scalar (see the header for why it is robust).
// `var` is the variance of the fitted MEAN — the model's intrinsic spread is deliberately NOT folded
// in here, because it describes how far an individual member may legitimately sit from the shared
// value, which matters when PREDICTING a member (prior_for), not when estimating the value itself.
int KitchenBelief::fit_scalar(const std::vector<float>& x, const std::vector<float>& w,
                              float prior_weight, float null_range, float& mean, float& var)
{
    int n = 0;
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        if (not (w[i] > 0.0f)) continue;
        num += static_cast<double>(w[i]) * static_cast<double>(x[i]);
        den += static_cast<double>(w[i]);
        ++n;
    }
    if (n == 0 or den <= 0.0)
        return 0;
    var = static_cast<float>(1.0 / den);
    // ★INITIALISE ON THE WEIGHTED MEDIAN, not the weighted mean. EM only refines the basin it is
    // given, and a single gross outlier can drag the mean so far that EVERY member then looks like an
    // outlier — responsibilities all collapse, the loop bails, and the garbage estimate survives.
    // Measured: one refrigerator mis-typed as a base unit pulled the mean initialiser to 1.14 m and
    // the fit stuck there; from the median it converges to 0.903. The median cannot be moved by a
    // minority however extreme, which is exactly the guarantee the initialiser needs.
    {
        std::vector<std::size_t> idx;
        idx.reserve(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) if (w[i] > 0.0f) idx.push_back(i);
        std::ranges::sort(idx, [&](std::size_t a, std::size_t b) { return x[a] < x[b]; });
        double acc = 0.0;
        mean = x[idx.back()];
        for (const auto i : idx)
        { acc += w[i]; if (acc >= 0.5 * den) { mean = x[i]; break; } }
    }

    // EM. Three sweeps is ample: the responsibilities of a clear outlier collapse on the first pass
    // and the estimate is stationary by the third. Deliberately NOT run to convergence — with few
    // members a long run can chase the sample and shrink the frame onto a single unit.
    const float w_prior = std::clamp(prior_weight, 1e-3f, 1.0f - 1e-3f);
    const float null_d  = 1.0f / std::max(1e-6f, null_range);
    for (int it = 0; it < 3; ++it)
    {
        double rnum = 0.0, rden = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            if (not (w[i] > 0.0f)) continue;
            // Responsibility of the SHARED-value component against the uniform null, using this
            // member's own variance (1/w[i] already carries its estimation noise + the model spread).
            const float vi  = 1.0f / w[i];
            const float lik = gauss_pdf(x[i], mean, vi);
            const float a = w_prior * lik, b = (1.0f - w_prior) * null_d;
            const float r = (a + b) > 0.0f ? a / (a + b) : 0.0f;
            rnum += static_cast<double>(r) * w[i] * x[i];
            rden += static_cast<double>(r) * w[i];
        }
        // Every member disowned the value. With the median initialiser this means the members
        // genuinely do not share this quantity, not that the estimate ran away — keep the last good
        // one and let the evidence report the disagreement.
        if (rden <= 1e-9) break;
        mean = static_cast<float>(rnum / rden);
        var  = static_cast<float>(1.0 / rden);
    }
    return n;
}

bool KitchenBelief::update(const std::vector<KitchenMember>& members,
                           const std::vector<KitchenMember>& axis_members)
{
    if (members.empty())
        return false;

    n_axis_ = fit_axis(axis_members.empty() ? members : axis_members);

    // ── Worktop and depth ───────────────────────────────────────────────────────────────────────
    // Each member is weighted by 1/(its own ESTIMATION variance + the model spread) — the honest
    // precision of "this member's value as evidence about the shared one". The estimation term is the
    // part cabinet_concept does not publish (it publishes a POSE covariance only); the worker fills a
    // length-scaled fallback, so a 2.2 m run still outweighs a 0.4 m stub.
    //
    // ★Depth is fitted PER TIER. A base carcass is ~0.60 m deep, a wall unit ~0.35; fusing them gave
    // one number that fits neither, and every upper unit then scored ~8σ off it and was reported as
    // not-a-member. Worktop was already base-only for the same reason — an upper unit has no worktop.
    std::vector<float> wt, wt_w, dpb, dpb_w, dpt, dpt_w, dpu, dpu_w;
    const float wt_model_var = params_.worktop_model_std * params_.worktop_model_std;
    const float dp_model_var = params_.depth_model_std   * params_.depth_model_std;
    for (const auto& m : members)
    {
        if (m.has_worktop())                              // only BASE units share a WORKTOP plane
        { wt.push_back(m.worktop); wt_w.push_back(1.0f / std::max(1e-9f, m.var_worktop + wt_model_var)); }
        const float dw = 1.0f / std::max(1e-9f, m.var_depth + dp_model_var);
        switch (m.tier)                                   // depth is shared WITHIN a tier, not across
        {
            case KitchenTier::Tall: dpt.push_back(m.depth); dpt_w.push_back(dw); break;
            case KitchenTier::Wall: dpu.push_back(m.depth); dpu_w.push_back(dw); break;
            default:                dpb.push_back(m.depth); dpb_w.push_back(dw); break;
        }
    }
    const float w_prior = 1.0f - std::clamp(params_.clutter_frac, 0.0f, 0.99f);
    n_worktop_ = fit_scalar(wt,  wt_w,  w_prior, kNullWorktopRange, state_.worktop, sigma_.worktop);
    fit_scalar(dpb, dpb_w, w_prior, kNullDepthRange, state_.depth, sigma_.depth);
    n_tall_  = fit_scalar(dpt, dpt_w, w_prior, kNullDepthRange, state_.depth_tall,  sigma_.depth_tall);
    n_upper_ = fit_scalar(dpu, dpu_w, w_prior, kNullDepthRange, state_.depth_upper, sigma_.depth_upper);

    seeded_ = n_axis_ > 0;
    return seeded_;
}

// The top-down message for one member, taken from the CAVITY frame (fitted without that member).
KitchenBelief::MemberPrior KitchenBelief::prior_for(const KitchenMember& m, const KitchenBelief& cavity) const
{
    MemberPrior p;
    if (not cavity.seeded_)
        return p;   // kappa/info stay 0 ⇒ inert, and every consumer ignores an inert message

    // Predicting an INDIVIDUAL member from the frame costs the frame's own uncertainty PLUS the
    // model's intrinsic spread: "kitchen units share an axis" is a tendency, not a law, and a unit
    // genuinely set at a slight angle still belongs to the kitchen. That quadrature sum is what
    // bounds how hard this frame can ever push — the same discipline as the ring's facing_model_std,
    // which was added after an unbounded version produced κ≈87 for an ordinary 16° offset.
    const float axis_var = cavity.sigma_.axis + params_.axis_model_std * params_.axis_model_std;
    p.yaw        = resolve_to_member(cavity.state_.axis, m.yaw);
    p.kappa      = cavity.p_frame() / std::max(1e-9f, axis_var);
    p.axis_resid = axis_delta(m.yaw, cavity.state_.axis);

    const float wt_var = cavity.sigma_.worktop + params_.worktop_model_std * params_.worktop_model_std;
    const float dp_var = cavity.depth_var_for(m) + params_.depth_model_std * params_.depth_model_std;
    p.worktop      = cavity.state_.worktop;
    p.worktop_info = m.has_worktop() ? cavity.p_frame() / std::max(1e-9f, wt_var) : 0.0f;
    // The depth prior is the one for THIS member's tier — telling a wall unit to be 0.60 m deep
    // would be worse than saying nothing.
    p.depth        = cavity.depth_for(m);
    p.depth_info   = cavity.p_frame() / std::max(1e-9f, dp_var);
    return p;
}

// ── Evidence: one rectilinear frame, or independent units? ───────────────────────────────────────
float KitchenBelief::instant_log_odds(const std::vector<KitchenMember>& members) const
{
    if (not seeded_ or members.empty())
        return 0.0f;

    const float axis_model_var = params_.axis_model_std * params_.axis_model_std;
    const float wt_model_var   = params_.worktop_model_std * params_.worktop_model_std;
    const float dp_model_var   = params_.depth_model_std   * params_.depth_model_std;

    float total = 0.0f;
    for (const auto& m : members)
    {
        // Under the FRAME: each observable is Gaussian about the shared value, widened by the frame's
        // own posterior spread and the model's intrinsic spread.
        const float vy = (m.var_yaw > 0.0f ? m.var_yaw : KitchenMember::kNoCovVarYaw) + sigma_.axis + axis_model_var;
        float p_frame_dens = gauss_pdf(axis_delta(m.yaw, state_.axis), 0.0f, vy);
        float p_null_dens  = 1.0f / kNullAxisRange;

        if (m.has_worktop())
        {
            p_frame_dens *= gauss_pdf(m.worktop, state_.worktop,
                                      sigma_.worktop + m.var_worktop + wt_model_var);
            p_null_dens  *= 1.0f / kNullWorktopRange;
        }
        p_frame_dens *= gauss_pdf(m.depth, depth_for(m), depth_var_for(m) + m.var_depth + dp_model_var);
        p_null_dens  *= 1.0f / kNullDepthRange;

        // The CLUTTER column is inside the frame hypothesis, not beside it: "there is a kitchen frame
        // AND this particular unit is not part of it" must be expressible, or one bathroom vanity
        // would argue the whole kitchen out of existence.
        //
        // ★The membership weight is per-member and CLASS-DEPENDENT, not the flat clutter constant. It
        // is what stops the evidence being monotone in member count: with a flat weight, ANY extra
        // object that beats the null RAISES the score, so a sideboard standing in the same room made
        // the kitchen look MORE certain (measured: +9.24 nats alone → +25.14 with the sideboard).
        // Weighting by class lets a unit whose class does not belong here sit on the clutter column
        // and contribute ~0 instead of corroborating.
        const float w     = member_prior_weight(m);
        const float mixed = w * p_frame_dens + (1.0f - w) * p_null_dens;
        total += std::log(std::max(1e-30f, mixed)) - std::log(std::max(1e-30f, p_null_dens));
    }
    return total;
}

// F_couple after every element has moved to the compromise the frame proposes. See the header: the
// gap to free_energy() is the free energy readjustment would actually buy, and is the prediction that
// makes this layer falsifiable rather than merely asserted.
float KitchenBelief::free_energy_if_adopted(const std::vector<KitchenMember>& members) const
{
    if (not seeded_)
        return 0.0f;
    std::vector<KitchenMember> moved;
    moved.reserve(members.size());
    for (const auto& m0 : members)
    {
        KitchenMember m = m0;
        const MemberPrior p = prior_for(m0, *this);
        // Yaw: the frame states an axis; the element fuses it with its own published precision. A
        // polygon-pinned run barely moves (its own yaw precision is high and, in cabinet_concept, its
        // yaw is not even a DOF) — which is exactly why the axis coupling has little to buy here.
        const float own_yaw_var = m0.var_yaw > 0.0f ? m0.var_yaw : KitchenMember::kNoCovVarYaw;
        m.yaw = adopted(m0.yaw, own_yaw_var, p.yaw, p.kappa);
        if (m0.has_worktop())
            m.worktop = adopted(m0.worktop, m0.var_worktop + params_.worktop_model_std * params_.worktop_model_std,
                                p.worktop, p.worktop_info);
        m.depth = adopted(m0.depth, m0.var_depth + params_.depth_model_std * params_.depth_model_std,
                          p.depth, p.depth_info);
        moved.push_back(m);
    }
    return -instant_log_odds(moved);
}

void KitchenBelief::update_log_odds(const std::vector<KitchenMember>& members)
{
    // ★A LOW-PASS, not an accumulator. The furniture is static: re-observing the same three runs 500
    // times is ONE observation, not 500 independent ones. Summing pinned ring_metaconcept's p at
    // ~0.9997 on the first cycle and made it unable to express doubt or ever recant.
    const float inst = instant_log_odds(members);
    const float a    = std::clamp(params_.evidence_ema_alpha, 0.0f, 1.0f);
    log_odds_ = std::clamp(log_odds_ + a * (inst - log_odds_), -params_.logodds_clamp, params_.logodds_clamp);
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
bool KitchenBelief::self_test()
{
    bool ok = true;
    const auto check = [&](bool c, const char* what)
    { if (not c) { std::printf("[kitchen_belief::self_test] FAIL: %s\n", what); ok = false; } };
    const auto d2r = [](float d) { return d * kPi / 180.0f; };
    const auto r2d = [](float r) { return r * 180.0f / kPi; };

    // ── (a) the 90° fold and the signed grid difference ──────────────────────────────────────────
    check(std::abs(r2d(fold_axis(d2r(89.6f))) - 89.6f) < 1e-3f, "(a) fold keeps an angle already in [0,90)");
    check(std::abs(r2d(fold_axis(d2r(-0.4f))) - 89.6f) < 1e-3f, "(a) -0.4 deg folds to 89.6 deg (same grid)");
    check(std::abs(r2d(fold_axis(d2r(179.6f))) - 89.6f) < 1e-3f, "(a) 179.6 deg folds to 89.6 deg");
    // The whole point: perpendicular runs are the SAME grid and must differ by ~0.
    check(std::abs(r2d(axis_delta(d2r(0.28f), d2r(89.01f))) - 1.27f) < 0.01f,
          "(a) runs at 0.28 and 89.01 deg differ by 1.27 deg, NOT by 88.7");
    check(std::abs(r2d(axis_delta(d2r(45.0f), d2r(0.0f)))) <= 45.0f + 1e-3f,
          "(a) the grid difference never exceeds 45 deg");

    const auto mk = [&](float yaw_deg, float yaw_std_deg, float worktop, float depth, float len) {
        KitchenMember m;
        m.yaw     = d2r(yaw_deg);
        m.var_yaw = d2r(yaw_std_deg) * d2r(yaw_std_deg);
        m.pinned  = yaw_std_deg <= 3.5f;
        m.worktop = worktop; m.depth = depth; m.length = len; m.tier = KitchenTier::Base;
        return m;
    };

    KitchenBeliefParams par;

    // ── (b) THE LIVE CASE. The three wall runs cabinet_concept actually reports, with the 3° axis σ
    // it now publishes. Their 4θ circular mean is −0.37° ≡ 89.63°; a raw arithmetic mean would give
    // ~59° and be wrong about every one of them.
    {
        const std::vector<KitchenMember> live = {
            mk(0.28f,  3.0f, 0.90f, 0.60f, 2.2f),
            mk(89.01f, 3.0f, 0.758f, 0.523f, 2.25f),
            mk(89.60f, 3.0f, 0.90f, 0.60f, 1.5f),
        };
        KitchenBelief b(par);
        check(b.update(live, live), "(b) the frame fits from three wall runs");
        const float ax = r2d(b.state().axis);
        check(std::abs(axis_delta(b.state().axis, d2r(-0.37f))) < d2r(0.05f),
              "(b) the shared axis fuses to -0.37 deg (== 89.63), the 90-periodic circular mean");
        check(ax >= 0.0f and ax < 90.0f, "(b) the reported axis is folded into [0,90)");
        check(b.n_axis_members() == 3, "(b) all three runs contributed to the axis");
        // Each member must be told to face the arm of the grid IT is on — never rotated 90°.
        for (const auto& m : live)
        {
            const auto p = b.prior_for(m, b);
            check(std::abs(p.yaw - m.yaw) < d2r(2.0f), "(b) the down-prior resolves to the member's OWN arm");
        }
    }

    // ── (c) THE ISLAND. A self-derived, mis-snapped axis at 83.2° with the loose σ cabinet_concept
    // now publishes for it. It must (1) barely move the frame, and (2) be told to rotate back.
    {
        std::vector<KitchenMember> all = {
            mk(0.28f,  3.0f, 0.90f, 0.60f, 2.2f),
            mk(89.01f, 3.0f, 0.758f, 0.523f, 2.25f),
            mk(89.60f, 3.0f, 0.90f, 0.60f, 1.5f),
        };
        const std::vector<KitchenMember> pinned = all;      // the structural cavity: wall runs only
        KitchenMember island = mk(83.21f, 6.4f, 0.896f, 0.405f, 1.9f);
        island.pinned = false;
        all.push_back(island);

        KitchenBelief b(par);
        b.update(all, pinned);
        check(std::abs(axis_delta(b.state().axis, d2r(-0.37f))) < d2r(0.05f),
              "(c) with the axis taken from PINNED members only, the island cannot drag the grid");

        const auto p = b.prior_for(island, b);
        check(std::abs(r2d(p.axis_resid) + 6.42f) < 0.05f,
              "(c) the island's misalignment is reported as ~-6.4 deg");
        check(std::abs(r2d(p.yaw) - 89.63f) < 0.05f,
              "(c) the island is told to sit on the 89.63 deg arm, not left at 83.21");
        check(p.kappa > 0.0f, "(c) the island's yaw prior carries a positive precision");

        // A frame that includes the island in its AXIS evidence is measurably dragged — this is the
        // failure the structural cavity exists to prevent, so assert it is real and not hypothetical.
        KitchenBelief dragged(par);
        dragged.update(all, all);
        check(std::abs(axis_delta(dragged.state().axis, b.state().axis)) > d2r(0.15f),
              "(c) including a self-derived axis in the evidence DOES move the grid (cavity is load-bearing)");
    }

    // ── (d) CAVITY / leave-one-out: the message to a member must not contain that member ──────────
    {
        const std::vector<KitchenMember> two = {
            mk(0.00f, 3.0f, 0.90f, 0.60f, 2.0f),
            mk(4.00f, 3.0f, 0.90f, 0.60f, 2.0f),
        };
        KitchenBelief full(par);  full.update(two, two);
        KitchenBelief cav(par);   cav.update({two[1]}, {two[1]});   // frame WITHOUT member 0
        const auto p_self   = full.prior_for(two[0], full);
        const auto p_cavity = full.prior_for(two[0], cav);
        check(std::abs(p_cavity.axis_resid) > std::abs(p_self.axis_resid) + d2r(0.5f),
              "(d) the cavity prior disagrees with member 0 MORE than the self-confirming one does");
    }

    // ── (g) ★A member the frame has REJECTED may not steer the axis ──────────────────────────────
    // Regression for the 2026-08-11 live case: fit_axis weighted purely by the peer's published
    // 1/var_yaw, so a member expelled to clutter still voted on the one DOF this agent exists to
    // estimate. Here three tidy runs agree on a 90° grid; a fourth publishes a very TIGHT yaw (so the
    // old precision-only weighting would let it dominate) that is 30° off the grid, and a worktop
    // 40 cm out so the mixture rejects it outright.
    {
        std::vector<KitchenMember> good = {
            mk(0.20f,  3.0f, 0.90f, 0.60f, 2.0f),
            mk(89.60f, 3.0f, 0.90f, 0.60f, 2.0f),
            mk(0.10f,  3.0f, 0.89f, 0.61f, 2.0f),
        };
        KitchenBelief b(par);
        b.update(good, good);                       // seed on the clean set
        const float axis_clean = b.state().axis;

        auto intruder = mk(30.0f, 0.3f, 1.30f, 0.60f, 2.0f);   // 100x the precision, 40 cm off
        std::vector<KitchenMember> with = good;
        with.push_back(intruder);
        b.update(with, with);

        check(b.membership_prob(intruder) < 0.01f,
              "(g) the 40 cm-off intruder is rejected by the mixture");
        check(std::abs(axis_delta(b.state().axis, axis_clean)) < d2r(2.0f),
              "(g) a REJECTED member must not drag the axis, however tight its published yaw");
    }

    // ── (e) The NULL is real: scattered furniture must not be declared a kitchen ──────────────────
    {
        const std::vector<KitchenMember> scattered = {
            mk(0.0f,  3.0f, 0.90f, 0.60f, 1.0f),
            mk(37.0f, 3.0f, 0.42f, 0.35f, 1.0f),
            mk(71.0f, 3.0f, 1.35f, 0.95f, 1.0f),
        };
        KitchenBelief b(par);
        b.update(scattered, scattered);
        check(b.instant_log_odds(scattered) < 0.0f,
              "(e) three unrelated objects score BELOW the independent-units null");

        const std::vector<KitchenMember> tidy = {
            mk(0.20f,  3.0f, 0.90f, 0.60f, 2.0f),
            mk(89.60f, 3.0f, 0.90f, 0.60f, 2.0f),
            mk(0.10f,  3.0f, 0.89f, 0.61f, 2.0f),
        };
        KitchenBelief t(par);
        t.update(tidy, tidy);
        check(t.instant_log_odds(tidy) > 0.0f, "(e) a real kitchen scores ABOVE the null");

        // The EMA must not saturate on the first cycle (the bug that pinned the ring at p~1).
        const float after_one = (t.update_log_odds(tidy), t.log_odds());
        check(after_one < par.logodds_clamp * 0.9f, "(e) one poll does not saturate the evidence clamp");
    }

    // ── (f) The shared WORKTOP, weighted by run length ────────────────────────────────────────────
    // The live disagreement this DOF exists to resolve: one run fitted its top at 0.758 m while its
    // neighbours sit at 0.90. A real kitchen has one worktop plane.
    {
        const std::vector<KitchenMember> mixed = {
            mk(0.0f,  3.0f, 0.900f, 0.60f, 2.2f),
            mk(90.0f, 3.0f, 0.758f, 0.523f, 2.25f),
            mk(0.0f,  3.0f, 0.905f, 0.60f, 1.5f),
        };
        KitchenBelief b(par);
        b.update(mixed, mixed);
        // ★The ROBUST fit puts the plane on the MAJORITY (~0.902), not on a compromise dragged toward
        // the outlier. A plain weighted mean landed at ~0.87 — i.e. the one bad run moved the very
        // value it was about to be judged against, which is the defect EM removes.
        check(std::abs(b.state().worktop - 0.902f) < 0.01f,
              "(f) the shared worktop lands on the MAJORITY, not on a compromise with the outlier");
        check(b.n_worktop_members() == 3, "(f) every base unit contributed a worktop sample");
        check(b.membership_prob(mixed[1]) < 0.5f, "(f) the 0.758 outlier is recognised as discordant");
        const auto p = b.prior_for(mixed[1], b);
        check(p.worktop > mixed[1].worktop + 0.10f,
              "(f) ★the low run is told to RAISE its worktop, by the full disagreement");
        check(p.worktop_info > 0.0f, "(f) the worktop prior carries a positive precision");
    }

    // ── (g) An UPPER unit has no worktop to share, and must not poison the plane ──────────────────
    {
        std::vector<KitchenMember> with_upper = {
            mk(0.0f, 3.0f, 0.90f, 0.60f, 2.0f),
            mk(0.0f, 3.0f, 0.90f, 0.60f, 2.0f),
        };
        KitchenMember upper = mk(0.0f, 3.0f, 2.10f, 0.35f, 2.0f);
        upper.tier = KitchenTier::Wall;
        with_upper.push_back(upper);
        KitchenBelief b(par);
        b.update(with_upper, with_upper);
        check(std::abs(b.state().worktop - 0.90f) < 1e-3f, "(g) an upper unit does not drag the worktop plane");
        check(b.n_worktop_members() == 2, "(g) only the base units are worktop evidence");
        check(b.prior_for(upper, b).worktop_info == 0.0f, "(g) an upper unit receives NO worktop prior");
    }

    // ── (i) A TALL unit (refrigerator) is a member WITHOUT a worktop ─────────────────────────────
    // Live defect this fixes: the fridge is floor-standing, so a {Base,Wall} taxonomy called it Base,
    // scored its 1.85 m top against a 0.90 m worktop plane, dragged that plane to 0.968 and the base
    // depth to 0.677, and THEN reported it at membership 0.00 — a real kitchen member evicted, after
    // corrupting the very values it was judged against.
    {
        std::vector<KitchenMember> ms = {
            mk(0.0f,  3.0f, 0.900f, 0.60f, 2.20f),
            mk(90.0f, 3.0f, 0.900f, 0.60f, 2.25f),
            mk(0.0f,  3.0f, 0.910f, 0.61f, 1.50f),
        };
        KitchenMember fridge = mk(90.0f, 3.0f, 1.850f, 0.70f, 0.70f);   // top = 1.85 m, no worktop
        fridge.tier = KitchenTier::Tall;
        ms.push_back(fridge);
        KitchenBelief b(par);
        b.update(ms, ms);
        check(std::abs(b.state().worktop - 0.903f) < 0.02f,
              "(i) a TALL unit does not drag the worktop plane");
        check(b.n_worktop_members() == 3 and b.n_tall_members() == 1,
              "(i) the fridge is counted as tall, not as a worktop sample");
        check(std::abs(b.state().depth - 0.603f) < 0.02f,
              "(i) ...nor the BASE depth (it has its own tier depth)");
        check(b.membership_prob(fridge) > 0.5f,
              "(i) ★the fridge is a MEMBER — it simply has no worktop");
        check(b.prior_for(fridge, b).worktop_info == 0.0f,
              "(i) and it receives no worktop prior");
        // Robustness: even MIS-typed as Base, the EM fit must not let it move the plane much.
        KitchenMember mistyped = fridge; mistyped.tier = KitchenTier::Base;
        std::vector<KitchenMember> ms2(ms.begin(), ms.end() - 1); ms2.push_back(mistyped);
        KitchenBelief b2(par); b2.update(ms2, ms2);
        check(std::abs(b2.state().worktop - 0.903f) < 0.03f,
              "(i) the ROBUST fit contains a mis-typed outlier too (EM, not the tier, does that work)");
    }

    // ── (j) JOINT FREE ENERGY: the coupling must PAY, and readjustment must pay more ──────────────
    // F_couple = −log-odds. Negative ⇒ the coupled model explains the scene better than independent
    // units and pushing priors is justified; positive ⇒ the coupling would impose structure the data
    // refuses. free_energy_if_adopted is the falsifiable part: if elements adopt and F does not fall,
    // this layer's model is wrong.
    {
        const std::vector<KitchenMember> tidy = {
            mk(0.20f,  3.0f, 0.900f, 0.600f, 2.0f),
            mk(89.60f, 3.0f, 0.902f, 0.600f, 2.0f),
            mk(0.10f,  3.0f, 0.898f, 0.601f, 2.0f),
        };
        KitchenBelief t(par); t.update(tidy, tidy);
        check(t.free_energy(tidy) < 0.0f, "(j) a real kitchen has NEGATIVE coupling free energy");
        check(t.free_energy_if_adopted(tidy) <= t.free_energy(tidy) + 1e-3f,
              "(j) readjustment does not INCREASE the coupling free energy");

        // A set that disagrees on the worktop: readjustment should buy a real reduction.
        const std::vector<KitchenMember> spread = {
            mk(0.20f,  3.0f, 0.860f, 0.600f, 2.0f),
            mk(89.60f, 3.0f, 0.940f, 0.600f, 2.0f),
            mk(0.10f,  3.0f, 0.900f, 0.601f, 2.0f),
        };
        KitchenBelief sp(par); sp.update(spread, spread);
        check(sp.free_energy_if_adopted(spread) < sp.free_energy(spread) - 1e-3f,
              "(j) ★where members DISAGREE, readjusting toward the frame lowers the free energy");

        // The null: scattered furniture must have POSITIVE coupling free energy (do not couple).
        const std::vector<KitchenMember> scattered2 = {
            mk(0.0f,  3.0f, 0.90f, 0.60f, 1.0f),
            mk(37.0f, 3.0f, 0.42f, 0.35f, 1.0f),
            mk(71.0f, 3.0f, 1.35f, 0.95f, 1.0f),
        };
        KitchenBelief sc(par); sc.update(scattered2, scattered2);
        check(sc.free_energy(scattered2) > 0.0f,
              "(j) unrelated objects have POSITIVE coupling free energy — the frame must stay quiet");
    }

    // ── (h) An unpublished-Σ member must be nearly inert, not flattering ─────────────────────────
    {
        const std::vector<KitchenMember> good = { mk(0.0f, 3.0f, 0.90f, 0.60f, 2.0f) };
        KitchenMember blind = mk(30.0f, 0.0f, 0.90f, 0.60f, 2.0f);   // var_yaw 0 ⇒ the wide fallback
        blind.var_yaw = 0.0f;
        std::vector<KitchenMember> both = good; both.push_back(blind);
        KitchenBelief b(par);
        b.update(both, both);
        check(std::abs(r2d(axis_delta(b.state().axis, 0.0f))) < 3.0f,
              "(h) a member with no published covariance barely moves the axis");
    }

    // ── (k) THE MESSAGE DOWN-DATE ────────────────────────────────────────────────────────────────
    // Without it, a frame that pushes a value and then reads the result back grows confident on its
    // own echo. The subtraction must recover exactly the member's own opinion.
    {
        // A member whose own estimate is 0.60 ± 0.04, told by us to be 0.90 with the same precision.
        const float own = 0.60f, own_var = 0.04f * 0.04f;
        const SentScalar sent{0.90f, 1.0f / (0.04f * 0.04f)};
        // What it publishes after fusing: the precision-weighted compromise.
        float pub = 0.5f * (own + sent.mean);
        float pub_var = 1.0f / (1.0f / own_var + sent.info);
        check(down_date(pub, pub_var, sent), "(k) the down-date is well posed here");
        check(std::abs(pub - own) < 1e-3f,
              "(k) ★subtracting our own message recovers the member's OWN estimate exactly");
        check(std::abs(std::sqrt(pub_var) - 0.04f) < 1e-3f, "(k) ...and its own precision with it");
    }
    {
        // Nothing sent ⇒ the published value passes through untouched.
        float v = 0.75f, var = 0.01f;
        check(down_date(v, var, SentScalar{}) and std::abs(v - 0.75f) < 1e-6f,
              "(k) with no outstanding message the value is unchanged");
    }
    {
        // ★The PD guard: our message claims MORE information than the member's whole posterior.
        // Skipping is correct; clamping would push a nonsense variance into the fit.
        float v = 0.80f, var = 0.05f * 0.05f;
        check(not down_date(v, var, SentScalar{0.9f, 1.0f / (0.01f * 0.01f)}),
              "(k) ★an over-strong outstanding message is REFUSED, not clamped");
        check(std::abs(v - 0.80f) < 1e-6f, "(k) ...and the value is left untouched when refused");
    }
    {
        // Repeated exchange must not manufacture confidence: down-dating each time holds the
        // member's own precision constant no matter how many rounds are run.
        const float own = 0.60f, own_var = 0.04f * 0.04f;
        float mu = 0.90f, info = 1.0f / (0.05f * 0.05f);
        float last_var = 0.0f;
        for (int round = 0; round < 20; ++round)
        {
            const SentScalar sent{mu, info};
            float pub = (own / own_var + sent.mean * sent.info) / (1.0f / own_var + sent.info);
            float pub_var = 1.0f / (1.0f / own_var + sent.info);
            down_date(pub, pub_var, sent);
            last_var = pub_var;
            mu = pub;                       // the frame re-derives its message from the CAVITY value
        }
        check(std::abs(std::sqrt(last_var) - 0.04f) < 1e-3f,
              "(k) ★★20 rounds of exchange leave the member's own precision unchanged (no echo)");
    }

    if (ok) std::printf("[kitchen_belief::self_test] all checks passed\n");
    return ok;
}

}  // namespace rc
