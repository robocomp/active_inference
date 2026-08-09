/*
 * existence_belief.h  —  shared per-instance EXISTENCE log-odds + occupancy/free-space evidence
 *
 * SHARED, header-only across the concept agents (table/chair/bottle/residual): one existence log-odds per
 * instance, fed by two interchangeable evidence channels (LiDAR carve · mask silhouette) below.
 *
 * See active_inference/EXISTENCE_BELIEF_PLAN.md. Every model instance (table/chair/bottle/residual) carries a
 * scalar existence log-odds L = log P(exists)/P(¬exists), updated each cycle by the log-likelihood-ratio of the
 * frame's evidence under {exists vs not}. Removal is a Bayesian decision on L, NOT a miss-counter.
 *
 * MODALITY 1 — LiDAR occupancy carve (lifted verbatim from residual_concept, which validated it): a beam that
 * RETURNS from inside the object volume ⇒ occupancy evidence; a beam that passes THROUGH to beyond ⇒ free-space
 * (absence) evidence; a beam that stops short (occluded) or misses ⇒ NO evidence (n_reached gate). Physical
 * detection/clutter RATES, not gates. The cycle's beams share a registration error ⇒ the summed ΔL is
 * common-mode SATURATED (tanh) to one confident observation's worth — the same "N correlated points can't
 * collapse σ" discipline as the metric belief's Woodbury cap. Object is a SOLID-box abstraction of a possibly
 * HOLLOW shape (empty under a tabletop); the caller suppresses interior free-evidence while it is observed.
 *
 * MODALITY 2 — mask occupancy (mask_evidence): the SAME log-odds math on projected-silhouette evidence the
 * agent supplies (predicted-detectable pixels that ARE / ARE NOT lit). See mask_evidence() below.
 *
 * Header-only (like lidar_ray_factor.h): pure geometry + Eigen, no DDS/DSR/torch — unit-testable in isolation.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Dense>

namespace rc::exist
{

// ─── Shared primitives: sensor rates · per-cycle evidence · common-mode saturation ───────────────

// Physical sensor rates (interpretable, not gates), shared by the LiDAR and mask channels.
struct SensorModel
{
    float sensor_sigma_m = 0.03f;   // range/localisation noise σ (m) — surface blur for the soft occ/free split
    float detection_prob = 0.85f;   // P(detect stimulus | exists & observable)  — sensor hit rate
    float clutter_prob   = 0.05f;   // P(detect stimulus | ¬exists)               — spurious rate
};
using CarveParams = SensorModel;    // back-compat alias for residual_concept's name

// One modality, one cycle, ALREADY common-mode saturated. n_reached==0 ⇒ the instance was NOT probed this
// cycle (out of range / fully occluded / behind) ⇒ HOLD (no update, no removal).
// ─── THE REMOVAL POLICY (one policy, all agents) ──────────────────────────────────────────────────
//
// Existence is a log-odds ratio. Per probe (a LiDAR beam, a silhouette sample) the ONLY honest update is
//
//     ΔL = p_vis · log[ P(outcome | exists) / P(outcome | ¬exists) ]
//
// where p_vis = P(this probe could have resolved the object | it exists). Everything else follows:
//
//   • p_vis → 0 (out of FoV, occluded, too far)  ⇒ contribution → 0. "HOLD" is not a special case, it is
//     what the ratio DOES when both hypotheses predict the same observation. Every explicit `if (…) return;
//     // HOLD` in this file was a place where the ratio was not being computed and someone clamped instead.
//   • The update is SYMMETRIC by construction. An "occupancy-only" channel is not a likelihood ratio, it is a
//     RATCHET: it can only push L to +L_max, where it sticks forever. That is not conservatism, it is a bug —
//     it made a phantom fridge survive 2885 cycles (~5 min) of confident camera absence with L never once
//     going negative, because an occupancy-only LiDAR channel re-pinned it at the clamp every cycle, after the
//     camera's absence and before the removal test. See [[existence-policy-unification]].
//
// The ONE thing that is genuinely per-agent is the VISIBILITY / SHAPE model that produces p_vis — and that is
// physics, not a tuning knob. `opaque_solid` below is that declaration, and it is what the old
// hollow_guarded_delta(observed=…) hack was standing in for.
//
// THREE outcomes per beam, given "the object exists", not two:
//   surface  — a return AT the object's near face                → consistent   ⇒ evidence FOR
//   interior — a return deep INSIDE the carved volume            → see below
//   through  — the beam crossed the volume and came out the far side ⇒ evidence AGAINST
//
// `interior` is the one that was wrong. For an OPAQUE SOLID (a fridge, a cabinet carcass) a return 1.5 m
// inside the box is IMPOSSIBLE if the object is there — the object would have blocked it. So it is evidence
// AGAINST, not for. Counting it as occupancy is a sign error, and it is exactly what makes a phantom box
// drawn over a wall permanently self-confirming: the wall behind it keeps "proving" it exists.
// For a NON-opaque model (a thin tabletop whose solid band is a lossy abstraction of mostly-empty space) an
// interior return says nothing either way, and neither does a through-beam — so both get p_vis 0. That is the
// hollow guard, derived instead of asserted, and it no longer needs to know whether the object was "observed".
struct Evidence
{
    float log_odds_delta = 0.0f;
    float e_occ = 0.0f, e_free = 0.0f;
    float e_interior = 0.0f;        // returns terminating deep inside the carved volume (see above)
    int   n_reached = 0;
};
using CarveEvidence = Evidence;     // back-compat alias

// Common-mode saturation: cap a cycle's summed raw log-odds to one confident hit's worth (tanh), so N
// registration-correlated observations can't count as N independent ones.
inline float saturate(float raw, float llr_occ)
{
    return (llr_occ > 1e-6f) ? llr_occ * std::tanh(raw / llr_occ) : raw;
}

// ── MODALITY 1: LiDAR oriented-box occupancy carve ─────────────────────────────────────────────────────────
// Lifted verbatim from residual_clusterer::carve_box. Beam RETURNS inside ⇒ e_occ; passes THROUGH ⇒ e_free;
// stops short (occluded) / misses ⇒ ignored. z-band [z_min,z_max] is the solid extent; surface_sigma_m is the
// belief's footprint-position σ (soft surface test). Returns per-cycle evidence with ΔL already saturated.
inline Evidence carve_box(const Eigen::Vector3f& origin, const std::vector<Eigen::Vector3f>& sweep,
                          float cx, float cy, float yaw, float w, float d, float z_min, float z_max,
                          float surface_sigma_m, const SensorModel& p)
{
    Evidence ev;
    const float hw = 0.5f * w, hd = 0.5f * d;
    if (hw <= 0.0f or hd <= 0.0f or sweep.empty()) return ev;

    const float cyaw = std::cos(yaw), syaw = std::sin(yaw);
    const auto  phi  = [](float x) { return 0.5f * std::erfc(-x * 0.70710678f); };   // Φ(x) = ½·erfc(−x/√2)
    const float sigma_surf = std::sqrt(p.sensor_sigma_m * p.sensor_sigma_m + surface_sigma_m * surface_sigma_m);
    const float sigma_z    = std::max(1e-3f, p.sensor_sigma_m);

    const float pd = std::clamp(p.detection_prob, 1e-3f, 1.0f - 1e-3f);
    const float pc = std::clamp(p.clutter_prob,   1e-3f, 1.0f - 1e-3f);
    const float llr_occ  = std::log(pd / pc);                    // >0: returned INSIDE ⇒ still there
    const float llr_free = std::log((1.0f - pd) / (1.0f - pc));  // <0: passed THROUGH ⇒ gone

    const float odx = origin.x() - cx, ody = origin.y() - cy;
    const Eigen::Vector3f O(cyaw * odx + syaw * ody, -syaw * odx + cyaw * ody, origin.z());

    float E_occ = 0.0f, E_free = 0.0f, E_int = 0.0f, reached_mass = 0.0f;
    for (const auto& q : sweep)
    {
        const float pdx = q.x() - cx, pdy = q.y() - cy;
        const Eigen::Vector3f P(cyaw * pdx + syaw * pdy, -syaw * pdx + cyaw * pdy, q.z());
        const Eigen::Vector3f dir = P - O;
        const float len = dir.norm();
        if (len < 1e-4f) continue;

        float t_near = 0.0f, t_far = std::numeric_limits<float>::max();
        bool miss = false;
        for (int a = 0; a < 2 and not miss; ++a)
        {
            const float o = O(a), dd = dir(a), lo = -((a == 0) ? hw : hd), hi = ((a == 0) ? hw : hd);
            if (std::abs(dd) < 1e-6f) { if (o < lo or o > hi) miss = true; }
            else
            {
                float t1 = (lo - o) / dd, t2 = (hi - o) / dd;
                if (t1 > t2) std::swap(t1, t2);
                t_near = std::max(t_near, t1);
                t_far  = std::min(t_far,  t2);
            }
        }
        if (miss or t_near > t_far or t_far < 0.0f) continue;

        const float t_mid = std::clamp(0.5f * (t_near + t_far), 0.0f, 1.0f);
        const float z_mid = O.z() + t_mid * dir.z();
        const float z_weight = std::clamp(phi((z_mid - z_min) / sigma_z) - phi((z_mid - z_max) / sigma_z), 0.0f, 1.0f);
        if (z_weight < 1e-3f) continue;

        const float sigma_t   = sigma_surf / len;
        const float p_reached = phi((1.0f - t_near) / sigma_t);
        const float p_through = phi((1.0f - t_far ) / sigma_t);
        const float reached   = p_reached * z_weight;
        if (reached < 1e-3f) continue;

        // Split "returned inside" into SURFACE vs INTERIOR by how deep past the entry face the return landed.
        // depth = (1 − t_near)·len metres beyond the near face; the return itself is at t = 1 by construction.
        // A real surface hit has depth ≈ 0 (within the same σ that blurs the surface test). A return metres
        // deep is the wall/furniture BEHIND a phantom box — the object, if present, would have occluded it.
        const float inside = (p_reached - p_through) * z_weight;
        const float depth  = std::max(0.0f, (1.0f - t_near) * len);
        const float w_surf = std::exp(-0.5f * (depth * depth) / std::max(1e-6f, sigma_surf * sigma_surf));
        E_occ      += inside * w_surf;              // consistent with the near face  ⇒ FOR
        E_int      += inside * (1.0f - w_surf);     // impossible if the object is a solid ⇒ AGAINST (see below)
        E_free     += p_through * z_weight;
        reached_mass += reached;
    }

    ev.e_occ = E_occ; ev.e_free = E_free; ev.e_interior = E_int;
    ev.n_reached = static_cast<int>(std::lround(reached_mass));
    if (ev.n_reached == 0) return ev;                            // p_vis ≡ 0 for every beam ⇒ nothing to say
    // Default ΔL keeps the historic two-outcome form so existing callers are byte-identical; the unified
    // policy is solid_delta() / hollow_delta() below, which the caller selects by declaring its shape.
    ev.log_odds_delta = saturate((E_occ + E_int) * llr_occ + E_free * llr_free, llr_occ);
    return ev;
}

// ─── The unified per-modality ΔL. Pick ONE by declaring the shape; there is no third option. ──────
//
// OPAQUE SOLID (fridge, cabinet carcass, any faithful filled volume): all three outcomes are informative.
// A surface return is FOR; a through-beam is AGAINST; an interior return is AGAINST too, because a solid
// object would have blocked it. Symmetric ⇒ cannot ratchet ⇒ a phantom over a wall dies instead of being
// confirmed by the wall.
inline float solid_delta(const Evidence& ev, const SensorModel& p)
{
    const float pd = std::clamp(p.detection_prob, 1e-3f, 1.0f - 1e-3f);
    const float pc = std::clamp(p.clutter_prob,   1e-3f, 1.0f - 1e-3f);
    const float llr_occ  = std::log(pd / pc);
    const float llr_free = std::log((1.0f - pd) / (1.0f - pc));
    return saturate(ev.e_occ * llr_occ + (ev.e_free + ev.e_interior) * llr_free, llr_occ);
}

// NON-OPAQUE model (thin tabletop / any band that abstracts mostly-empty space): only the surface return is
// informative. Interior and through beams have p_vis ≈ 0 for the modelled surface — they pass through space
// the model never claimed was filled — so they contribute 0. This is the old hollow guard, but DERIVED from
// the shape declaration rather than asserted per-cycle, and it no longer needs an `observed` flag.
inline float hollow_delta(const Evidence& ev, const SensorModel& p)
{
    const float pd = std::clamp(p.detection_prob, 1e-3f, 1.0f - 1e-3f);
    const float pc = std::clamp(p.clutter_prob,   1e-3f, 1.0f - 1e-3f);
    return saturate(ev.e_occ * std::log(pd / pc), std::log(pd / pc));
}

// HOLLOW-object guard: while the object is actively OBSERVED this cycle, beams passing through its interior
// are NOT absence evidence (a solid box is a lossy abstraction of a hollow shape — e.g. empty under a tabletop)
// → suppress e_free; occupancy always counts. Returns the re-weighted (saturated) ΔL to integrate. Mirrors
// residual_concept's remove_by_occupancy_evidence.
// ★LEGACY, superseded by solid_delta / hollow_delta above. Kept BIT-IDENTICAL for the callers that have not
// been live-revalidated yet (table_concept, cabinet_concept): it must therefore use e_occ + e_interior, because
// before the surface/interior split those were one number. Do not "simplify" it to e_occ — that would silently
// weaken those agents' occupancy evidence. Migrating a caller means deleting the `observed` argument and
// declaring its shape: hollow_delta for a thin-plate abstraction, solid_delta for a faithful filled volume.
inline float hollow_guarded_delta(const Evidence& ev, bool observed, const SensorModel& p)
{
    const float pd = std::clamp(p.detection_prob, 1e-3f, 1.0f - 1e-3f);
    const float pc = std::clamp(p.clutter_prob,   1e-3f, 1.0f - 1e-3f);
    const float llr_occ  = std::log(pd / pc);
    const float llr_free = std::log((1.0f - pd) / (1.0f - pc));
    return saturate((ev.e_occ + ev.e_interior) * llr_occ + (observed ? 0.0f : ev.e_free * llr_free), llr_occ);
}

// ── THE CLUTTER PRIOR, MEASURED INSTEAD OF ASSUMED ─────────────────────────────────────────────────────────
//
// `SensorModel::clutter_prob` is P(a beam returns from this volume | our object is NOT there). Configured as a
// constant (0.05) it silently asserts that THE ALTERNATIVE TO OUR OBJECT IS FREE SPACE. In a furnished room the
// alternative is a wall, a cabinet, a chair — and then P(return | ¬object) ≈ P(return | object), the likelihood
// RATIO is ≈1, and the honest ΔL is ≈0 rather than a full confident hit. Assuming free space is precisely how a
// phantom box drawn over real structure becomes self-confirming: the structure keeps "proving" it exists.
//
// So measure it, per instance per cycle, from the SAME sweep: carve an equal-volume SHELL around the object and
// take its return rate. A √2 dilation in x and y doubles the footprint, so the shell has exactly the object's own
// volume — the two rates are directly comparable samples of the same neighbourhood at the same height, gathered
// by the same beams. No new constant, and it self-calibrates per scene.
//
// ★WHY A CONTRAST AND NOT JUST A BIGGER pc (measured on table_concept, 2026-08-06, ~5400 LiDAR cycles each):
//     real table_1     return rate 1.96%      phantom table_2   return rate 6.56%      model pd asserts 85%
// Both saturated to the SAME +2.83/cycle, so the channel was not merely uninformative — the phantom scored
// 3.3× the real table. Against pd=0.85 every observed rate is tiny, so no value of pc fixes the ORDERING; the
// only statistic that separates them is whether the volume returns MORE THAN ITS SURROUNDINGS. That is what
// contrast_delta computes, and it is 0 by construction when object and neighbourhood are indistinguishable.
struct LocalClutter
{
    float rate      = 0.0f;   // Beta-shrunk P(return | this neighbourhood, our object absent)
    float raw_rate  = 0.0f;   // unshrunk shell rate (diagnostic)
    int   n_reached = 0;      // beams that reached the shell
    bool  ok        = false;  // false ⇒ no shell sample; `rate` fell back to the configured prior
};

// Beta-shrunk return rate. The prior mean is the configured clutter_prob and its strength is 1/pc beams — the
// sample size at which ONE clutter return is expected, i.e. the smallest sample that can say anything about a
// rate that small. Derived from the model's own number, so this introduces nothing to tune. Few beams ⇒ the
// estimate stays at the prior; many beams ⇒ the data dominates. Continuous, no minimum-count gate.
inline float shrunk_rate(float e_occ, int n_reached, float prior_rate)
{
    const float r0 = std::clamp(prior_rate, 1e-3f, 1.0f - 1e-3f);
    const float m0 = 1.0f / r0;                                   // pseudo-beams carrying the prior
    return (e_occ + r0 * m0) / (static_cast<float>(std::max(0, n_reached)) + m0);
}

// The object's neighbourhood, from an equal-volume dilated carve. `outer` must be the SAME z-band with the
// footprint scaled by √2 about the same centre; the shell is outer MINUS inner, so the object's own returns are
// removed and what remains is the background it sits in.
inline LocalClutter measure_clutter(const Evidence& inner, const Evidence& outer, const SensorModel& p)
{
    LocalClutter lc;
    const float e = std::max(0.0f, (outer.e_occ + outer.e_interior) - (inner.e_occ + inner.e_interior));
    const int   n = outer.n_reached - inner.n_reached;
    lc.n_reached = std::max(0, n);
    lc.ok        = lc.n_reached > 0;
    lc.raw_rate  = lc.ok ? e / static_cast<float>(lc.n_reached) : 0.0f;
    lc.rate      = shrunk_rate(e, lc.n_reached, p.clutter_prob);
    return lc;
}

// Occupancy evidence as a CONTRAST against the measured neighbourhood, for a model whose absence half is not
// trustworthy (a thin plate abstracted as a solid band: see hollow_delta). Per returning beam,
//
//     ΔL = log[ P(return | our object is here) / P(return | only what surrounds it is here) ]
//
// with BOTH probabilities ESTIMATED from this cycle's beams rather than one asserted and one configured. The
// numerator is the object volume's own observed rate, not pd: a thin tabletop in a thick band, plus registration
// error, puts the real rate two orders of magnitude below pd, so asserting pd would compare a measurement to a
// fiction. Object and neighbourhood alike ⇒ ratio 1 ⇒ ΔL = 0: the channel falls silent exactly where it cannot
// discriminate, which is the whole of the phantom-on-a-wall case.
//
// ⚠ONE-SIDED BY DECLARATION (floored at 0), and this is the deliberate asymmetry: the same measurement that
// makes the numerator honest — 1.96% observed where the band model predicts 85% — says the band is a poor
// predictor, and evidence from a misspecified model must not be trusted to DELETE a real object. Removal stays
// the camera silhouette's job. Unlike the occupancy-only form this replaces, it is not a ratchet: it can no
// longer pin L at +L_max from structure the object did not cause. Upper clamp = one confident hit's worth,
// the same cap `saturate` already applies.
inline float contrast_delta(const Evidence& ev, const LocalClutter& lc, const SensorModel& p)
{
    if (ev.n_reached <= 0) return 0.0f;                            // p_vis ≡ 0 ⇒ HOLD
    const float pd = std::clamp(p.detection_prob, 1e-3f, 1.0f - 1e-3f);
    const float pc = std::clamp(p.clutter_prob,   1e-3f, 1.0f - 1e-3f);
    const float llr_cap = std::log(pd / pc);                       // one confident hit, the historic ceiling
    const float p_box   = shrunk_rate(ev.e_occ + ev.e_interior, ev.n_reached, pc);
    const float q       = std::max(1e-4f, lc.rate);
    const float llr     = std::clamp(std::log(p_box / q), 0.0f, llr_cap);
    return saturate((ev.e_occ + ev.e_interior) * llr, llr);
}

// ── MODALITY 2: mask silhouette occupancy ──────────────────────────────────────────────────────────────────
// The agent projects its model silhouette into the camera and counts, over the predicted-DETECTABLE footprint
// (in-FoV, in-range, UN-occluded by nearer geometry — the agent decides that with its camera API + occluders):
//   e_occ  = expected silhouette pixels that ARE lit by the observed mask   (present ⇒ still there)
//   e_free = expected silhouette pixels that are NOT lit   (predicted but absent ⇒ gone)
// n_reached = the detectable footprint size (0 ⇒ not observable this cycle ⇒ HOLD). Same detection/clutter LLR
// + tanh saturation as the LiDAR channel — the mask's pixels are registration-correlated too.
inline Evidence mask_evidence(float e_occ, float e_free, int n_detectable, const SensorModel& p)
{
    Evidence ev;
    ev.e_occ = e_occ; ev.e_free = e_free; ev.n_reached = n_detectable;
    if (n_detectable == 0) return ev;                            // not observable ⇒ HOLD
    const float pd = std::clamp(p.detection_prob, 1e-3f, 1.0f - 1e-3f);
    const float pc = std::clamp(p.clutter_prob,   1e-3f, 1.0f - 1e-3f);
    const float llr_occ  = std::log(pd / pc);
    const float llr_free = std::log((1.0f - pd) / (1.0f - pc));
    ev.log_odds_delta = saturate(e_occ * llr_occ + e_free * llr_free, llr_occ);
    return ev;
}

// ── Per-instance existence log-odds ────────────────────────────────────────────────────────────────────────
class ExistenceBelief
{
public:
    explicit ExistenceBelief(float L0 = 0.0f, float L_max = 4.0f) : L_(L0), L_max_(L_max) {}

    // ── Frame-to-frame correlation: N consecutive looks are not N independent observations ──────────────
    //
    // saturate() above already refuses to let N registration-correlated observations WITHIN a cycle count as
    // N independent ones. The identical argument applies ACROSS cycles and was not being made: a detector
    // does not fail by independent coin flip, it fails at a particular framing, so consecutive misses from a
    // barely-changed viewpoint are the SAME observation repeated.
    //
    // MEASURED on table_concept's own log (2026-08-09), restricted to the cycles where absence is actually
    // charged, because over all frames the correlation is dominated by "not in view" and reads far higher:
    //     p_detect>0.3   P(miss)=0.021   P(miss | previous miss)=0.610   rho=0.602
    //     p_detect>0.5   P(miss)=0.010   P(miss | previous miss)=0.333   rho=0.327
    // A miss is rare on those cycles, but once one happens the next is 20-60x more likely than the base rate.
    //
    // N correlated trials carry an EFFECTIVE sample size N_eff(N) = N / (1 + (N-1)·rho), which saturates at
    // 1/rho however long the run gets. So the k-th consecutive observation of the same sign is worth the
    // MARGINAL gain w_k = N_eff(k) − N_eff(k−1): the first counts fully, later ones count for less and less.
    // A run of opposite sign resets the count — that genuinely is new information.
    //
    // WHY THIS MATTERS (the bug it fixes): table_1 was removed and reborn three times. Each death was a run of
    // consecutive no-mask cycles while the robot drove through the detector envelope's shoulder — 7 cycles,
    // ~0.7 s, taking L from 4.00 to −0.36 and onward to the −4 floor. Each individual charge was arithmetically
    // right for ONE independent miss; charging seven of them was not. With the measured rho those 7 misses are
    // worth 1.5–2.4 observations and L lands at +1.9 / +1.4 — absence still charged, object still alive.
    //
    // rho DEFAULTS TO 0, which makes w_k ≡ 1 and reproduces the previous behaviour exactly. An agent opts in by
    // publishing a rho MEASURED from its own log, the same way the detector envelope is fitted rather than
    // picked. Symmetric by construction: confirmations are just as correlated as misses, and damping only one
    // side would be a ratchet.
    void set_frame_correlation(float rho) { rho_ = std::clamp(rho, 0.0f, 0.95f); }
    float frame_correlation() const { return rho_; }
    int   run_length() const { return run_k_; }   // consecutive same-sign observations (diagnostic)

    // Fold one modality's per-cycle evidence. HOLD (no change) when the instance wasn't probed this cycle.
    // `p_detect` = P(the detector could have fired | the object is there) at THIS observation, i.e. how
    // unambiguous the conditions were. Pass it whenever the caller has it; the default of 1 means
    // "conditions were ideal", which is the independent case and reproduces the pre-correlation behaviour.
    void integrate(const Evidence& e, float p_detect = 1.0f)
    {
        if (e.n_reached == 0) return;
        const int sign = (e.log_odds_delta > 0.0f) - (e.log_odds_delta < 0.0f);
        if (sign != 0)
        {
            if (sign == run_sign_) ++run_k_;
            else { run_sign_ = sign; run_k_ = 1; }
        }
        L_ = std::clamp(L_ + e.log_odds_delta * marginal_weight(run_k_, p_detect), -L_max_, L_max_);
    }
    float logodds()  const { return L_; }
    float p_exists() const { return 1.0f / (1.0f + std::exp(-L_)); }
    // Remove when P(occupied) < removal_prob, i.e. L < log(p/(1−p)). The one honest decision boundary.
    bool  should_remove(float removal_prob) const
    {
        const float p = std::clamp(removal_prob, 1e-3f, 0.5f);
        return L_ < std::log(p / (1.0f - p));
    }
    void  set(float L)    { L_ = std::clamp(L, -L_max_, L_max_); }
    void  set_max(float m){ L_max_ = std::max(1e-3f, m); }

private:
    // Marginal worth of the k-th consecutive same-sign observation: N_eff(k) − N_eff(k−1), with
    // N_eff(n) = n / (1 + (n−1)·rho). rho = 0 ⇒ 1 for every k (independent trials, the previous behaviour).
    // ★rho IS NOT A CONSTANT — it falls as the conditions get less ambiguous, and that is what the data
    // says. Measured on table_concept's log, conditioned on the p_detect band the observation sat in:
    //     p_detect>0.3  rho=0.602      p_detect>0.5  rho=0.327      p_detect>0.7  rho=0.198
    // Consecutive misses are correlated when they share a CAUSE — marginal framing, the detector sitting
    // on its envelope shoulder — and that cause weakens as p_detect rises. With the object squarely in
    // view and p_detect near 1 there is no shared excuse left, so a miss is genuine evidence of absence
    // and successive misses count almost independently.
    //
    // rho_eff = rho * (1 - p_detect) has the two limits that matter and no extra parameter:
    //   · p_detect -> 1 (clear conditions): rho_eff -> 0, misses are independent, an absent object is
    //     REMOVED after a few of them. Without this the run saturated at 1/rho observations' worth, which
    //     BOUNDS the total absence evidence — measured at rho=0.327 that is 3.06 observations ~ 4.6 nats
    //     against the 8 nats needed to walk L from the +4 ceiling to the -4 floor, so a phantom the robot
    //     was staring straight at could never be removed at all. That was a real regression, seen live.
    //   · p_detect -> 0 (a look that could never have resolved it): rho_eff -> rho, fully damped — though
    //     there the charge is already ~0, since ΔL is p_detect-weighted upstream.
    float marginal_weight(int k, float p_detect) const
    {
        const float rho = rho_ * std::clamp(1.0f - p_detect, 0.0f, 1.0f);
        if (rho <= 1e-4f or k <= 1) return 1.0f;
        const auto n_eff = [&](int n) { return n <= 0 ? 0.0f
                                                      : static_cast<float>(n) / (1.0f + (n - 1) * rho); };
        return std::max(0.0f, n_eff(k) - n_eff(k - 1));
    }

    float L_, L_max_;
    float rho_      = 0.0f;   // frame-to-frame correlation of same-sign observations; 0 ⇒ independent
    int   run_sign_ = 0;      // +1 confirming run, −1 absence run, 0 = none yet
    int   run_k_    = 0;      // length of the current same-sign run
};

}  // namespace rc::exist
