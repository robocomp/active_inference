/*
 * common/birth_surprise/residual_field_reader.h — read rc::GridField off the DSR `residual` node. SHARED.
 *
 * Separate from birth_surprise_probe.h ON PURPOSE: that header is pure (std + the struct) and stays testable
 * with no graph, no Qt and no DDS. This is the thin adapter that knows where the field is published, so the
 * probe does not acquire a DSR dependency it has no use for.
 */

#pragma once

#include <cstdint>

#include <dsr/api/dsr_api.h>

#include "birth_surprise_probe.h"   // rc::GridField

namespace rc {

// Read the residual field off the `residual` node (type "grid"). Returns false and leaves `out` EMPTY —
// hence invalid, hence inert for every consumer — whenever anything is missing, so a caller never has to
// distinguish "absent" from "stale". Main-thread only (graph read).
//
// ★Was five copies. Four were byte-identical; door's dropped the VARIANCE and the room-node guard, which is
// the usual shape of this kind of drift: the copy that diverges is the one nobody diffed.
inline bool read_residual_field(DSR::DSRGraph& G, std::uint64_t room_node_id, GridField& out)
{
    out = GridField{};
    if (room_node_id == 0) return false;
    const auto gopt = G.get_node("residual");   // node renamed "grid"→"residual" (type stays "grid")
    if (not gopt.has_value()) return false;
    const auto& gnode = gopt.value();
    const auto pa = G.get_attrib_by_name<grid_occupancy_prob_att>(gnode);
    const auto ma = G.get_attrib_by_name<grid_field_meta_att>(gnode);
    if (not (pa.has_value() and ma.has_value())) return false;
    const auto& M = ma.value().get();
    if (M.size() < 5) return false;
    out.prob = pa.value().get();               // snapshot copy (small, ~2 Hz)
    if (const auto va = G.get_attrib_by_name<grid_occupancy_var_att>(gnode); va.has_value())
        out.var = va.value().get();
    out.xmin = M[0]; out.ymin = M[1]; out.cell = M[2];
    out.width = static_cast<int>(M[3]); out.height = static_cast<int>(M[4]);
    return out.valid();
}

}  // namespace rc
