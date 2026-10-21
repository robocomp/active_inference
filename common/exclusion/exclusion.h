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

#include "claims.h"                  // Claim, claimed_fraction, p_unclaimed, explained_by_other, Seniority (PURE)
#include "../footprint/footprint.h"      // rc::geom::Footprint, overlap_ratio
#include "../nbv/graph_obstacles.h"      // rc::nbv::collect_graph_obstacles — the ONE graph footprint reader


namespace rc::exclusion
{

// Claim, claimed_fraction, p_unclaimed, inside, explained_by_other, Seniority: claims.h (pure, see its header).

// Every fitted object in the graph that is NOT one of ours, as a true oriented footprint in the room frame.
// `my_prefix` is this agent's node-name prefix ("refrigerator"), the same string the ownership sweep uses —
// filtering by NAME rather than by id set is what makes this correct for an agent with several instances.
//
// MAIN-THREAD ONLY: collect_graph_obstacles asks for a ts==0 transform, and the InnerEigenAPI ts==0 cache is
// unlocked (CLAUDE.md). Every concept agent calls this from compute().
inline std::vector<Claim> foreign_claims(DSR::DSRGraph& G, DSR::InnerEigenAPI* inner_eigen,
                                         std::string_view my_prefix)
{
    return foreign_claims_from(rc::nbv::collect_graph_obstacles_identified(G, inner_eigen, /*self_id=*/0), my_prefix);
}

}  // namespace rc::exclusion
