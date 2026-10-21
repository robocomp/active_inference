#pragma once
/*
 * common/track/merge_instances.h — the pairwise merge sweep every concept agent runs, once.
 *
 * WHAT WAS DUPLICATED. All seven agents carried the same 22-line skeleton: snapshot the ids, walk every
 * unordered pair, skip anything already removed, re-`find` both (the map is mutated underneath), keep the
 * more-observed one, retire the other, and — the line everyone got right and nobody could have derived
 * twice by accident — `if (drop == ids[i]) break;`, because once `i` itself is the one retired there is no
 * point comparing it against the rest.
 *
 * WHAT WAS NOT. The DECISION and the CONSEQUENCE, and they differ for real reasons:
 *   · bottle merges on a two-circle lens (a cylinder has no yaw, and its bounding square would overstate
 *     the area by 4/π); chair/door/hood/refrigerator/table on an oriented-rectangle clip; door's footprint
 *     is its APERTURE, not its swung leaf.
 *   · cabinet is the RUN family: it merges COLLINEAR fragments that need not overlap at all, and merging
 *     GRAFTS the fused interval onto the keeper before the other is dropped. A seam that only said
 *     yes/no could not express that, so the decision returns a PAYLOAD and the agent applies it.
 *
 * ★★THE SEAM IS (decide → payload → apply), NOT (overlap ratio → delete). An "overlap_fn + threshold"
 * seam looks tidier and would have locked cabinet out — the family whose merge is the interesting one.
 * The test for what belongs here is the fleet's usual one: does it depend on the belief UNIT? The sweep
 * does not. Everything that does is a callback.
 *
 * ★★★AND IT TAKES THE COUNTER, because three agents proved that an optional one gets forgotten. bottle,
 * chair and door merged instances and never incremented `ev_g_.merges` — bottle even RESET it every cycle —
 * while `evidence_monitor.h` rendered `merges=%d/%ld` on all three dashboards. Three dashboards read
 * "merges=0/0" for as long as they have existed. Counting inside the sweep is the only version of this
 * that cannot drift, so the counter is a parameter and not a caller's afterthought.
 */

#include <cstdint>
#include <optional>
#include <unordered_set>
#include <vector>

#include "../dashboard/evidence_monitor.h"   // rc::EvidenceGlobals — the counter this sweep fills
#include "merge_sweep.h"                  // merge_overlapping_sweep — the loop itself (no Qt)
#include <utility>

namespace rc::track
{

/*
 * Merge overlapping/duplicate instances in `insts` (any map of id → instance): merge_overlapping_sweep
 * (merge_sweep.h — decide / observed / apply are documented there), plus the dashboard counter.
 *
 * Returns the number retired, and adds it to `ev.merges` / `ev.merges_cum`.
 */
template <class Insts, class DecideFn, class ObservedFn, class ApplyFn>
int merge_overlapping(Insts& insts, rc::EvidenceGlobals& ev,
                      DecideFn&& decide, ObservedFn&& observed, ApplyFn&& apply)
{
    const int merged = merge_overlapping_sweep(insts, std::forward<DecideFn>(decide),
                                               std::forward<ObservedFn>(observed), std::forward<ApplyFn>(apply));
    ev.merges      += merged;
    ev.merges_cum  += merged;
    return merged;
}

}  // namespace rc::track
