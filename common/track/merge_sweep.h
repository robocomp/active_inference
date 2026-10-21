#pragma once
/*
 * common/track/merge_sweep.h — the pairwise merge SWEEP of common/track/merge_instances.h, without the dashboard
 * counter: header-only, stdlib only (no Qt). Read merge_instances.h's header for what the sweep is and why the
 * seam is (decide → payload → apply).
 *
 * ★WHY THE SPLIT (door_concept DoorCore, 2026-10-21). merge_instances.h takes rc::EvidenceGlobals so the counter
 * cannot be forgotten — and EvidenceGlobals lives in a Qt header. door_concept's merge now runs in a DSR/Qt-free
 * core that replays recorded inputs, so it calls this sweep and hands the count back to the adapter, whose
 * merge_overlapping (below the counter) is unchanged. ONE loop, two entry points; nothing in it changed.
 */
#include <cstdint>
#include <optional>
#include <unordered_set>
#include <vector>

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
 * Returns the number retired. (merge_instances.h's merge_overlapping adds it to the dashboard counters.)
 */
template <class Insts, class DecideFn, class ObservedFn, class ApplyFn>
int merge_overlapping_sweep(Insts& insts, DecideFn&& decide, ObservedFn&& observed, ApplyFn&& apply)
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

    return merged;
}

}  // namespace rc::track
