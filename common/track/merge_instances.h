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

namespace rc::track
{

/*
 * Merge overlapping/duplicate instances in `insts` (any map of id → instance).
 *
 *   decide(a, b)   → std::optional<P>   the whole pairwise question. Empty ⇒ these two are different
 *                                        objects. P is whatever the agent needs downstream (a ratio, a
 *                                        fused geometry) and is handed back untouched.
 *   observed(inst) → comparable          how well seen. The GREATER is kept; ties keep the first id.
 *   apply(keep_id, drop_id, keeper, dropped, payload)
 *                                        do the merge: graft, log, and RETIRE `drop_id`. Ownership of the
 *                                        graph and of the instance map stays with the agent — this sweep
 *                                        never deletes a node, so it needs no DSR and stays testable.
 *
 * ⚠`dropped` is a reference INTO the map and `apply` is expected to erase it; read it before you retire it.
 * The sweep itself never touches the map after `apply` returns — it re-`find`s on the next pair.
 *
 * Returns the number retired, and adds it to `ev.merges` / `ev.merges_cum`.
 */
template <class Insts, class DecideFn, class ObservedFn, class ApplyFn>
int merge_overlapping(Insts& insts, rc::EvidenceGlobals& ev,
                      DecideFn&& decide, ObservedFn&& observed, ApplyFn&& apply)
{
    if (insts.size() < 2)
        return 0;

    // Snapshot the keys: `apply` erases from the map, and iterating it while it is mutated is undefined.
    std::vector<std::uint64_t> ids;
    ids.reserve(insts.size());
    for (auto& [id, _] : insts) ids.push_back(id);

    int merged = 0;
    std::unordered_set<std::uint64_t> removed;
    for (std::size_t i = 0; i < ids.size(); ++i)
    {
        if (removed.count(ids[i])) continue;
        for (std::size_t j = i + 1; j < ids.size(); ++j)
        {
            if (removed.count(ids[j])) continue;
            const auto ia = insts.find(ids[i]), ib = insts.find(ids[j]);
            if (ia == insts.end() or ib == insts.end()) continue;   // retired by an earlier pass

            auto payload = decide(ia->second, ib->second);
            if (not payload.has_value()) continue;

            const bool keep_i = observed(ia->second) >= observed(ib->second);
            const std::uint64_t keep = keep_i ? ids[i] : ids[j];
            const std::uint64_t drop = keep_i ? ids[j] : ids[i];
            apply(keep, drop, keep_i ? ia->second : ib->second,
                              keep_i ? ib->second : ia->second, *payload);
            removed.insert(drop);
            ++merged;

            // `i` is the one that just went; there is nothing left to compare it against.
            if (drop == ids[i]) break;
        }
    }

    ev.merges      += merged;
    ev.merges_cum  += merged;
    return merged;
}

}  // namespace rc::track
