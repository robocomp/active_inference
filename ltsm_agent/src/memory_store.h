/*
 * memory_store.h — persist the long-term memory graph to disk, and seed it back.
 *
 * WHY IT EXISTS. The memory graph is re-seeded from `ltsm_root.json` every launch, so today a
 * restart wipes everything the robot has ever learnt about places. That is acceptable for a graph
 * that only holds a room's geometry (perception rebuilds it) and NOT acceptable for anything whose
 * value is accumulation — the passage statistic on a `passage` node is the first such thing.
 *
 * WHEN IT SAVES. At the FENCE, not on a change counter: the `current` edge flip is the one moment
 * the memory graph is provably consistent, because it happens only after an eviction's copy has
 * been verified. A counter would fire at arbitrary moments — including halfway through writing a
 * room's subtree — and put a state on disk that never existed. Plus once more on a graceful exit.
 *
 * HOW IT WRITES. Temp file + `rename()`, previous generation kept as `.prev`. A truncated memory
 * file is the failure mode that loses everything the robot knows, and it would surface days later
 * as a parse error; `rename` within one filesystem is atomic, so the file on disk is always a whole
 * generation, old or new.
 *
 * WHAT IT DOES NOT PERSIST. The `current` edge — which does not live in this graph at all, it is
 * written on the LIVE one. Worth stating anyway, because it is the thing that must never come back
 * from a file: where the robot is now is a claim about the present that perception has to
 * re-establish, and by the let-go rule a stale one would make every concept agent anchored
 * elsewhere drop its nodes on the strength of yesterday's file. Persist the map; re-earn the place.
 *
 * ★ THE ZEROED-STAMP CHECK IN `load()` IS AN ALARM, AND IT SHOULD READ ZERO. uint64 attributes on
 * NODES round-trip correctly — `dsr_utils.cpp:105` goes through `toString().toULongLong()`. The
 * attribute switch is written twice, though, and the EDGE copy at `:180` still truncated through a
 * 32-bit `QVariant::toUInt` until door_concept fixed the twin; every uint64 in the registry is a
 * timestamp or an id, so anything carried on an edge was silently zeroed. Hence the count: a
 * non-zero reading here now means something regressed or an edge attribute turned up where one was
 * not expected, which is worth a line in the log either way. `rc::provenance` writes
 * `creation_datetime` (the same instant as a string) beside every stamp, so a birth time would
 * survive even if the number did not — belt and braces, not the load-bearing part.
 */
#pragma once

#include <dsr/api/dsr_api.h>

#include <filesystem>
#include <memory>
#include <string>

namespace ltsm
{

class MemoryStore
{
public:
    MemoryStore(std::shared_ptr<DSR::DSRGraph> memory, std::filesystem::path path, bool enabled);

    [[nodiscard]] bool enabled() const { return enabled_; }
    [[nodiscard]] int  saves()   const { return saves_; }
    [[nodiscard]] bool dirty()   const { return dirty_; }

    /// Something durable changed; the next fence should write.
    void mark_dirty() { dirty_ = true; }

    /// Seed the graph from the persisted file, if enabled and the file is there. Returns the number
    /// of nodes it added (0 = nothing to restore, which is the normal first run).
    int load();

    /// Atomic write of the whole memory graph. No-op when disabled, or when nothing has changed and
    /// `force` is false. `reason` only goes to the log, so a reader can see WHY a generation exists.
    bool save(const std::string &reason, bool force = false);

private:
    std::shared_ptr<DSR::DSRGraph> mem_;
    std::filesystem::path          path_;
    bool enabled_ = false;
    bool dirty_   = false;
    int  saves_   = 0;
};

}   // namespace ltsm
