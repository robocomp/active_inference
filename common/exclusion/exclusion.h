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
struct Claim
{
    std::string        node;   // e.g. "door_3" — what to name in a log, and what identifies it across cycles
    std::uint64_t      id = 0;
    rc::geom::Footprint fp{};
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
        out.push_back({o.name, o.id, {o.fp.cx, o.fp.cy, o.fp.w, o.fp.h, o.fp.yaw}});
    }
    return out;
}

// How much of `fp` is already explained by somebody else's object, in [0,1]. The MAXIMUM over claims, not a
// sum: two neighbours each covering half of me is not the same statement as one covering all of me, and
// summing would let a crowd of distant furniture add up to a claim nobody actually makes.
//
// overlap_ratio is a fraction of the SMALLER footprint, so a small object sitting entirely inside a large one
// reads 1.0 — which is the honest answer to "is this volume already accounted for".
inline float claimed_fraction(const rc::geom::Footprint& fp, const std::vector<Claim>& claims,
                              const Claim** who = nullptr)
{
    float worst = 0.0f;
    const Claim* best = nullptr;
    for (const auto& c : claims)
        if (const float r = rc::geom::overlap_ratio(fp, c.fp); r > worst)
        { worst = r; best = &c; }
    if (who) *who = best;
    return worst;
}

// The weight to multiply evidence by: 1 where the space is free, 0 where another object fully explains it.
inline float p_unclaimed(const rc::geom::Footprint& fp, const std::vector<Claim>& claims,
                         const Claim** who = nullptr)
{
    return 1.0f - claimed_fraction(fp, claims, who);
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
    void resolve_at_birth(const rc::geom::Footprint& fp, const std::vector<Claim>& claims)
    {
        if (resolved)
            return;
        const Claim* who = nullptr;
        const float frac = claimed_fraction(fp, claims, &who);
        resolved    = true;
        junior      = frac > 0.0f;
        senior_node = who ? who->node : std::string{};
        claimed     = frac;
    }

    // Call every cycle with this instance's CURRENT footprint and the current foreign claims. Returns the
    // weight to apply to this instance's OCCUPANCY evidence: 1.0 for a senior or unclaimed instance, and
    // 1 - overlap for a junior one still sitting inside its senior's footprint (the discount follows the
    // neighbour if it moves, and lifts entirely if the junior ever moves out from under it).
    float occupancy_weight(const rc::geom::Footprint& fp, const std::vector<Claim>& claims)
    {
        const Claim* who = nullptr;
        const float frac = claimed_fraction(fp, claims, &who);
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
