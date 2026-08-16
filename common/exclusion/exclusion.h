/*
 * common/exclusion/exclusion.h — no two objects occupy the same space. SHARED, header-only.
 *
 * ★WHY THIS EXISTS, measured 2026-08-15. refrigerator_concept created `refrigerator_2` at (-0.51, 4.75)
 * w=0.90 yaw=-0.017, on top of door_concept's `door_3` at (-0.59, 4.61) w=0.91 yaw=-0.010 — centres 16 cm
 * apart, same width to a centimetre, same yaw to 0.4 degrees. Not an overlap: the SAME physical object,
 * claimed by two concepts. It then could not die, because the door's LiDAR returns kept confirming it:
 * occupancy beat free space on 100% of 4122 cycles (mean 316 vs 13), which re-pinned the existence log-odds
 * at its +4 clamp every cycle, so `decide_removal` never found it condemned and reset the debounce streak to
 * zero 4122 times running. An hour of a phantom that no amount of looking could remove.
 *
 * Every agent already refuses to fit two of its OWN instances to one object (merge_overlapping_instances).
 * NONE of them ever asked what a DIFFERENT concept had already claimed — each merge loop runs over
 * `fitter_->instances()` alone. This module is that missing question, asked the same way by everyone.
 *
 * ★IT IS A CONTINUOUS WEIGHT, NOT A GATE, per CLAUDE.md: encode the effect in the generative model and let
 * it fall out of the inference. `p_unclaimed` in [0,1] scales two things — the birth evidence of a CANDIDATE
 * and the occupancy evidence of a JUNIOR instance — so a concept condensing onto a door simply never accrues
 * the evidence to be born, and one that already did stops being confirmed by its neighbour's returns and
 * dies through the ordinary free-space channel. Nothing is deleted by fiat and there is no threshold.
 *
 * ★SENIORITY: THE FIRST CLAIMANT KEEPS THE SPACE. Discounting symmetrically would be the cleaner sentence,
 * but it is wrong here — the kitchen metaconcept deliberately builds a RUN of abutting carcasses whose
 * footprints overlap (0.554 m measured between the fridge and its neighbour), and a symmetric rule would
 * have that run suppress itself. So the weight applies only where a claim was ALREADY STANDING when the
 * newcomer appeared. Two established neighbours never discount each other.
 *
 * The reader is NOT duplicated: rc::nbv::collect_graph_obstacles already walks the graph for exactly these
 * footprints, and had two silent bugs fixed in it (deprecated int attrs; pre-inflated extents). Reuse it.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

#include "../footprint/footprint.h"      // rc::geom::Footprint, overlap_ratio
#include "../nbv/graph_obstacles.h"      // rc::nbv::collect_graph_obstacles — the ONE graph footprint reader

namespace rc::exclusion
{

// One other concept's standing claim on a piece of the room.
//
// ★A CLAIM IS A VOLUME, NOT A FOOTPRINT (2026-08-16). The first version of this module compared footprints
// only, and in a kitchen that is not enough to mean anything: a hood hangs over a worktop, a wall unit over a
// base unit, a bottle stands on a table. Every one of those pairs reads as 100% overlap in plan view while
// the two objects share no space at all. Measured live: hood_1 (z 1.99..2.28) lay entirely inside the
// footprint of cabinet_w13_base (z 0.02..0.76), so the explained-away rule was deleting ~0.97 m of the
// cabinet's own wall run as evidence "already explained" by something 1.2 m above it.
//
// z1 <= z0 means the height was never published ⇒ UNBOUNDED, i.e. exactly the old 2-D behaviour. That is the
// safe direction for an unsized node: it keeps claiming, so the rule can still fail to catch a collision but
// never invents one — the same asymmetry seniority is built on.
struct Claim
{
    std::string        node;   // e.g. "door_3" — what to name in a log, and what identifies it across cycles
    std::uint64_t      id = 0;
    rc::geom::Footprint fp{};
    float              z0 = 0.0f;   // base, room frame (m)
    float              z1 = 0.0f;   // top; <= z0 ⇒ height unknown ⇒ spans every height

    bool has_z() const { return z1 > z0; }

    // Does this claim share any height with the band [q0, q1]? An unknown band on EITHER side means "we
    // cannot separate them vertically", which must answer yes — silence is not evidence of clearance.
    bool overlaps_z(float q0, float q1) const
    {
        if (not has_z() or not (q1 > q0))
            return true;
        return std::max(z0, q0) < std::min(z1, q1);
    }

    // The single-point form: is `pz` inside the claimed band?
    bool contains_z(float pz) const { return not has_z() or (pz >= z0 and pz <= z1); }
};

// Every fitted object in the graph that is NOT one of ours, as a true oriented footprint in the room frame.
// `my_prefix` is this agent's node-name prefix ("refrigerator"), the same string the ownership sweep uses —
// filtering by NAME rather than by id set is what makes this correct for an agent with several instances.
//
// MAIN-THREAD ONLY: collect_graph_obstacles asks for a ts==0 transform, and the InnerEigenAPI ts==0 cache is
// unlocked (CLAUDE.md). Every concept agent calls this from compute().
inline std::vector<Claim> foreign_claims(DSR::DSRGraph& G, DSR::InnerEigenAPI* inner_eigen,
                                         std::string_view my_prefix)
{
    std::vector<Claim> out;
    for (const auto& o : rc::nbv::collect_graph_obstacles_identified(G, inner_eigen, /*self_id=*/0))
    {
        if (std::string_view(o.name).starts_with(my_prefix))
            continue;                        // one of ours — merge_overlapping_instances owns that case
        out.push_back({o.name, o.id, {o.fp.cx, o.fp.cy, o.fp.w, o.fp.h, o.fp.yaw}, o.z0, o.z1});
    }
    return out;
}

// How much of `fp` is already explained by somebody else's object, in [0,1]. The MAXIMUM over claims, not a
// sum: two neighbours each covering half of me is not the same statement as one covering all of me, and
// summing would let a crowd of distant furniture add up to a claim nobody actually makes.
//
// overlap_ratio is a fraction of the SMALLER footprint, so a small object sitting entirely inside a large one
// reads 1.0 — which is the honest answer to "is this volume already accounted for".
//
// `q0..q1` is the querying object's own vertical band. A claim that shares no height with it is not a claim
// on this object at all — a hood at 2.0 m explains nothing about a base unit at 0.5 m, however exactly their
// footprints coincide. Leave the band at its default (q1 <= q0) and the test degrades to the old plan-view
// answer, which is what a caller with no height estimate should get.
inline float claimed_fraction(const rc::geom::Footprint& fp, const std::vector<Claim>& claims,
                              const Claim** who = nullptr, float q0 = 0.0f, float q1 = 0.0f)
{
    float worst = 0.0f;
    const Claim* best = nullptr;
    for (const auto& c : claims)
    {
        if (not c.overlaps_z(q0, q1))
            continue;                        // different storey of the same footprint — not our neighbour
        if (const float r = rc::geom::overlap_ratio(fp, c.fp); r > worst)
        { worst = r; best = &c; }
    }
    if (who) *who = best;
    return worst;
}

// The weight to multiply evidence by: 1 where the space is free, 0 where another object fully explains it.
inline float p_unclaimed(const rc::geom::Footprint& fp, const std::vector<Claim>& claims,
                         const Claim** who = nullptr, float q0 = 0.0f, float q1 = 0.0f)
{
    return 1.0f - claimed_fraction(fp, claims, who, q0, q1);
}

// ─── Explained-away evidence: the object limits its own growth ───────────────────────────────────
//
// ★THE POINT THAT CLOSES THE LOOP. Birth exclusion stops a concept CONDENSING onto another object, and the
// occupancy discount stops a junior being CONFIRMED by its senior's returns. Neither stops a believed object
// from GROWING into a neighbour: cabinet_2 expanded across ~6 m into refrigerator_1 and collapsed its depth
// to 0.12 m, because every point on the fridge's face was, to the cabinet, perfectly good evidence of more
// cabinet. Seniority cannot help — the cabinet was senior, alone and unclaimed when it was born.
//
// The rule is not a limit and not a clamp. It is Occam, stated locally:
//
//     A point another component already explains is not evidence for me. Growing my extent to cover it buys
//     no likelihood and costs complexity, so my own fit stops there.
//
// This is the SAME principle the fleet already applies to birth ("model expansion must lower total free
// energy") — growth is that event made continuous. It needs no arbitration: whichever object has weaker
// support for the contested volume gives it up, because the cost is paid by whoever is wrong.
//
// ★AND IT IS SAFE FOR THE KITCHEN RUN BY GEOMETRY, NOT BY EXEMPTION. Abutting carcasses do not overlap
// (two touching 0.6 m boxes measure an overlap ratio of 0.000), so a run pays nothing. No rig-membership
// whitelist, no "unless it is a kitchen" branch — the special case that would have made this fragile simply
// does not arise.
//
// Point-in-rectangle is geometry, not a tuned cutoff: the point is either inside another object's footprint
// or it is not.
inline bool inside(const rc::geom::Footprint& f, float px, float py)
{
    const float c = std::cos(f.yaw), s = std::sin(f.yaw);
    const float dx = px - f.cx, dy = py - f.cy;
    const float lx =  c * dx + s * dy;      // world → object-local
    const float ly = -s * dx + c * dy;
    return std::abs(lx) <= 0.5f * f.w and std::abs(ly) <= 0.5f * f.h;
}

// Is this room-frame point already explained by somebody else's object? Such points must not be admitted as
// support for THIS object's extent. `self` is excluded by the caller (foreign_claims already drops our own).
//
// ★PASS THE POINT'S z. This is a test on a VOLUME (see Claim): a return off the underside of a hood and a
// return off the worktop below it project to the same (x, y) and are not the same evidence. Dropping the
// second because of the first is how this rule went from protecting a fit to starving one — measured, it was
// discarding ~0.97 m of cabinet_w13_base's wall run under hood_1's footprint. A claim with no published
// height still matches at every z, so nothing that used to be caught stops being caught.
inline bool explained_by_other(float px, float py, float pz, const std::vector<Claim>& claims)
{
    for (const auto& c : claims)
        if (inside(c.fp, px, py) and c.contains_z(pz))
            return true;
    return false;
}

// The heightless form, for a caller that genuinely has no z (a 2-D detection). Identical to the old
// behaviour. Prefer the 3-argument overload wherever the point has a z — which, for every mask/LiDAR point
// in this tree, it does.
inline bool explained_by_other(float px, float py, const std::vector<Claim>& claims)
{
    for (const auto& c : claims)
        if (inside(c.fp, px, py))
            return true;
    return false;
}

// ─── Seniority ───────────────────────────────────────────────────────────────────────────────────
//
// An instance records ONCE, the first time it is evaluated, whether anybody else was already standing where
// it stands. That answer is what makes the rule asymmetric, and it is per-instance state the agent owns.
//
// Why "first evaluation" rather than a node creation timestamp: cortex has no birth stamp yet
// (node_creation_timestamp is blocked on a reinstall), and this needs none — the question is only ever
// "was the other one already there when I appeared", which the first evaluation answers directly.
struct Seniority
{
    bool          resolved = false;   // has the first evaluation happened?
    bool          junior   = false;   // was a foreign claim ALREADY standing here when we appeared?
    std::string   senior_node;        // who it was, for the log
    float         claimed  = 0.0f;    // how much of us they explained at that moment

    // ★RESOLVE AT BIRTH, NOT AT FIRST SIGHT. Call this ONCE, from the agent's create path, with the newborn's
    // footprint. Anything the agent did not create this run — an instance adopted from a graph node that was
    // already there at startup — is left SENIOR, because we have no evidence it came second and the cost of
    // guessing wrong is suppressing a real object.
    //
    // That asymmetry matters on RESTART. Resolving on "the first cycle I evaluate you" would have a real
    // fridge, standing legitimately beside a real cabinet, wake up after a restart, find the cabinet already
    // present, and declare ITSELF the junior — quietly discounting an object that was never in doubt. Birth is
    // the only moment at which "who was here first" is actually observed rather than inferred.
    //
    // `q0..q1` is the newborn's own vertical band; leave it defaulted only if the agent has no height at
    // birth. Getting it right matters most HERE, because seniority is recorded once and never revisited: a
    // fridge born under a hood with no z would be stamped junior for its whole life on a footprint overlap
    // that never existed in three dimensions.
    void resolve_at_birth(const rc::geom::Footprint& fp, const std::vector<Claim>& claims,
                          float q0 = 0.0f, float q1 = 0.0f)
    {
        if (resolved)
            return;
        const Claim* who = nullptr;
        const float frac = claimed_fraction(fp, claims, &who, q0, q1);
        resolved    = true;
        junior      = frac > 0.0f;
        senior_node = who ? who->node : std::string{};
        claimed     = frac;
    }

    // Call every cycle with this instance's CURRENT footprint and the current foreign claims. Returns the
    // weight to apply to this instance's OCCUPANCY evidence: 1.0 for a senior or unclaimed instance, and
    // 1 - overlap for a junior one still sitting inside its senior's footprint (the discount follows the
    // neighbour if it moves, and lifts entirely if the junior ever moves out from under it).
    float occupancy_weight(const rc::geom::Footprint& fp, const std::vector<Claim>& claims,
                           float q0 = 0.0f, float q1 = 0.0f)
    {
        const Claim* who = nullptr;
        const float frac = claimed_fraction(fp, claims, &who, q0, q1);
        if (not resolved)
        {
            // Never observed a birth for this instance (adopted at startup, or an agent that has not wired
            // resolve_at_birth yet) ⇒ SENIOR by default. Safe direction: it can fail to catch a collision,
            // never invent one.
            resolved = true;
            junior   = false;
            return 1.0f;
        }
        // ★A JUNIOR INSTANCE IS NOT DELETED HERE. It stops being CONFIRMED by the returns that belong to its
        // senior; the ordinary free-space channel then removes it if it truly is not there. Deleting it
        // outright would be a verdict where the model already has a way to express a doubt.
        if (not junior)
            return 1.0f;               // we were here first (or alone) — a newcomer never discounts us
        claimed = frac;
        return 1.0f - frac;            // still inside our senior's footprint ⇒ its returns are not our evidence
    }
};

}  // namespace rc::exclusion
