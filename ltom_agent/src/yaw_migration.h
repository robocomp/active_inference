// yaw_migration.h — the ONE-TIME migration of a class's stored yaws into the countable object frame (shared-engine K2b).
//
// Why: the legacy chair_concept agent published a chair's yaw with its BACKREST on local −y; the chair on the countable binary
// publishes it with the back on +y (the countable convention: R(yaw)·ŷ points from the front into the back). The same physical
// chair therefore reads 180° apart in the two binaries. ltom stores, per entry, keyframe poses (and a Move's from/to keyframes)
// in the producer's frame, and thumbnails keyed on a viewing-angle bin measured against that yaw (rc::thumbs::view_bin). A
// repertory written by the legacy chair must be turned by π (and the bins by kViewBins/2) before the countable chair adds to it.
//
// What is legacy: which binary produced a keyframe is not recorded, so the caller states the time the countable chair took over
// (legacy_until_ms): a keyframe or Move stamped before it is legacy and is turned; one at or after it is already countable and is
// left alone. Thumbnails carry no stamp: an entry whose keyframes are ALL before the cut has legacy thumbnails, turned; an entry
// with a keyframe at or after it may hold views from both binaries, so its thumbnails are DROPPED (a better view replaces a
// missing one, exactly as the loader does for a bin of another kViewBins). An entry with no keyframe keeps nothing to turn but its
// thumbnails, which are treated as legacy.
//
// Idempotent: every visited entry of the class is marked yaw_convention = kYawBackPlusY, and a marked entry is never touched
// again. The marker is persisted by repertory_io (absent in older files = unstated).
// Pure (no Qt, no I/O): the tool ltom_agent/tools/chair_yaw_migrate.cpp does the load and the save.
#pragma once
#include "repertory.h"
#include "../../common/thumbs/thumb_select.h"   // rc::thumbs::kViewBins

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <ranges>
#include <string_view>

namespace ltom
{
struct YawMigrationReport { int migrated = 0, already = 0, other_class = 0, keyframes_turned = 0, moves_turned = 0,
                            thumbs_turned = 0, thumbs_dropped = 0; };

// yaw + π, wrapped to (−π, π]
inline double turned_half(double yaw)
{
    double y = std::remainder(yaw + std::numbers::pi, 2.0 * std::numbers::pi);
    if (y <= -std::numbers::pi) y += 2.0 * std::numbers::pi;
    return y;
}
// the bin seen from the opposite side of the same object
inline int turned_bin(int bin) { return (bin + rc::thumbs::kViewBins / 2) % rc::thumbs::kViewBins; }

inline YawMigrationReport migrate_yaw_convention(Repertory& r, std::string_view cls, std::int64_t legacy_until_ms)
{
    YawMigrationReport rep;
    std::vector<Entry> changed;
    for (const auto& [uid, e0] : r.entries())
    {
        if (e0.cls != cls) { ++rep.other_class; continue; }
        if (e0.yaw_convention == kYawBackPlusY) { ++rep.already; continue; }
        Entry e = e0;
        for (Keyframe& k : e.keyframes)
            if (k.t_ms < legacy_until_ms) { k.pose.yaw = turned_half(k.pose.yaw); ++rep.keyframes_turned; }
        for (Move& m : e.moves)
            if (m.t_ms < legacy_until_ms)
            {   // both ends are keyframes copied at link time, in the frame current then
                m.from.pose.yaw = turned_half(m.from.pose.yaw);
                m.to.pose.yaw = turned_half(m.to.pose.yaw);
                ++rep.moves_turned;
            }
        const bool all_legacy = std::ranges::all_of(e.keyframes, [&](const Keyframe& k) { return k.t_ms < legacy_until_ms; });
        if (all_legacy)
        {
            std::map<int, Thumbnail> t;
            for (auto& [bin, th] : e.thumbs) t[turned_bin(bin)] = std::move(th);
            rep.thumbs_turned += static_cast<int>(t.size());
            e.thumbs = std::move(t);
        }
        else
        {
            rep.thumbs_dropped += static_cast<int>(e.thumbs.size());
            e.thumbs.clear();
        }
        e.yaw_convention = std::string(kYawBackPlusY);
        changed.push_back(std::move(e));
        ++rep.migrated;
    }
    for (Entry& e : changed) r.insert_loaded(std::move(e));
    return rep;
}
}   // namespace ltom
