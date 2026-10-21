#pragma once
// rt_ring.h — the newest slot of a cortex RT history ring (rt_timestamps / rt_translation / ...).
// ★ cortex's rt_head_index is NOT an index into rt_timestamps: it is the offset (in packed 3-vectors) of the slot it
//   will write NEXT — write slot k, then head = 3*(k+1) (dsr_rt_api.cpp, insert_or_assign_edge_RT). This mirrors
//   cortex's own decoder (latest_slot_from_head) so the controller reads the stamp of the block it actually gets.
#include <cstddef>

namespace rc::rt
{
    inline constexpr std::size_t kBlockSize = 3;   // cortex RT_BLOCK_SIZE: one translation / euler 3-vector per slot

    [[nodiscard]] constexpr std::size_t latest_slot_from_head(std::size_t head_index, std::size_t history_size)
    {
        const std::size_t next_slot = (head_index / kBlockSize) % history_size;
        return next_slot == 0 ? history_size - 1 : next_slot - 1;
    }
}
