// phantom_bridge.h — door_concept's side of the private phantom memory (common/phantom_memory): the PlaceTable keys rows by ROOM NAME, the
// file by LTSM room_id (+ the room's own frame). DSR-free; the adapter supplies the room_id <-> name pairs it saw on room entry.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include "core/place_table.h"
#include "../../common/phantom_memory/phantom_memory.h"

namespace rc::door::phantom_bridge
{
/// Inject the room's refuted places as REMOVED rows under `room_name`. The file has no L (a refutation's size/height is what it keeps):
/// the row's evidence is 0 and its measured height is the candidate's height measurement (PlaceTable::remembered_height); its age runs
/// from removed_ms. A removed row already in the table at exactly that place (this run's own, or an earlier injection) is not duplicated.
/// Returns the number of rows added.
inline int inject(core::PlaceTable& table, const rc::phantom::Memory& mem, std::uint64_t room_id, const std::string& room_name, float place_sd)
{
    int added = 0;
    int idx = 0;
    for (const auto& p : mem.of_room(room_id))
    {
        const bool have = std::ranges::any_of(table.rows(), [&](const core::PlaceRow& r)
                                              { return r.room == room_name and r.removed_wall_ms.has_value() and r.xy.x() == p.x and r.xy.y() == p.y; });
        if (have) { ++idx; continue; }
        // The same row a CSV removed row becomes (PlaceTable::row_of), so restore_into ages its L, Sp and ledger through the learned hazard alike.
        IdentityRow i;
        i.room = room_name;
        i.name = "phantom_" + std::to_string(room_id) + "_" + std::to_string(idx++);
        i.xy = {p.x, p.y};
        i.pose = RestorePose{p.yaw, p.width, p.height, 0.f};
        i.exist_logodds = p.exist_logodds; i.exist_physical = p.exist_physical;
        i.seen_wall_ms = p.seen_ms.value_or(p.removed_ms);
        i.removed_ms = p.removed_ms;
        if (not p.ledger.empty()) i.ledger = p.ledger;
        core::PlaceRow r = core::PlaceTable::row_of(i, place_sd, table.geometry_may_refute(), table.lookalike());
        r.height_sd = p.height_sd;
        table.upsert(std::move(r));
        ++added;
    }
    return added;
}

/// The removed places of `room_name`, as file rows of `room_id`.
[[nodiscard]] inline std::vector<rc::phantom::Row> removed_rows(const core::PlaceTable& table, std::uint64_t room_id, const std::string& room_name)
{
    std::vector<rc::phantom::Row> out;
    for (const auto& r : table.rows())
        if (r.room == room_name and r.removed_wall_ms.has_value())
        {
            rc::phantom::Row p{room_id, r.xy.x(), r.xy.y(), r.yaw, r.width, r.height, r.height_sd, *r.removed_wall_ms};
            p.exist_logodds = r.L_evidence; p.exist_physical = r.physical;
            if (r.last_seen_wall_ms > 0) p.seen_ms = r.last_seen_wall_ms;
            if (not r.ledger.views().empty()) p.ledger = r.ledger.serialise();
            out.push_back(std::move(p));
        }
    return out;
}

/// Rows of rooms LTSM no longer remembers go; rooms entered this run are live and stay. An EMPTY list is "LTSM's memory is not loaded yet"
/// (a transient), never a ruling: it forgets nothing. Returns the number dropped.
inline int forget(rc::phantom::Memory& m, std::span<const std::uint64_t> ltsm_ids, const std::map<std::string, std::uint64_t>& entered)
{
    if (ltsm_ids.empty()) return 0;
    std::vector<std::uint64_t> keep(ltsm_ids.begin(), ltsm_ids.end());
    for (const auto& [name, id] : entered) keep.push_back(id);
    return m.forget_except(keep);
}

/// The memory to save: the file's rows of rooms NOT entered this run are kept as they are; every entered room (name -> room_id) is
/// replaced by the table's removed rows of that room. `provisional` rooms (dropped by a recognition swap) contribute nothing.
[[nodiscard]] inline rc::phantom::Memory merged(const core::PlaceTable& table, rc::phantom::Memory file, const std::map<std::string, std::uint64_t>& entered)
{
    for (const auto& [name, id] : entered)
        file.set_room(id, removed_rows(table, id, name));
    return file;
}
}   // namespace rc::door::phantom_bridge
