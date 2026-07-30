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

    // Fold one modality's per-cycle evidence. HOLD (no change) when the instance wasn't probed this cycle.
    void integrate(const Evidence& e)
    {
        if (e.n_reached == 0) return;
        L_ = std::clamp(L_ + e.log_odds_delta, -L_max_, L_max_);
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
    float L_, L_max_;
};

}  // namespace rc::exist
