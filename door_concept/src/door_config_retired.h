// door_config_retired.h — config keys the door-existence model (docs/superpowers/plans/2026-11-07-door-existence-model.md)
// retired. They steered the old existence patches (seed, clamp, debounce, re-acquisition registry, streak births). A file that still sets
// one is REJECTED at load rather than ignored: a silently ignored knob is how a retuned config stops meaning anything.
// Pure (no ConfigLoader): the loader passes its own `exists`, the unit test a lambda.
#pragma once
#include <array>
#include <format>
#include <string>
#include <string_view>
#include <vector>

namespace rc
{
inline constexpr std::array<std::string_view, 14> kRetiredDoorKeys{
    "Existence.BirthLogodds", "Existence.MaxLogodds", "Existence.RemovalProb", "Existence.RemoveFrames",
    "Existence.OutOfRoomGain", "Existence.ShortGain", "Existence.MinHeightConf", "Existence.ReacquireRadiusM",
    "Existence.GhostMax", "Existence.CentralRegionFrac",
    "Existence.SensorSigmaM", "Existence.RoomPrior", "Existence.RoomMarginM",   // dead keys: nothing read them
    "Tracker.BirthFrames"};   // R2: the door core births on ONE good hit view (birth_frames = 1 is set in code)

/// The retired keys `exists` reports present, in the table's order.
template <class Exists>
[[nodiscard]] std::vector<std::string> retired_door_keys_present(Exists&& exists)
{
    std::vector<std::string> out;
    for (const auto k : kRetiredDoorKeys)
        if (exists(std::string(k))) out.emplace_back(k);
    return out;
}

/// The message the loader prints before refusing to start. Names the plan so the reader knows where the model went.
[[nodiscard]] inline std::string retired_door_keys_message(const std::vector<std::string>& keys)
{
    std::string list;
    for (const auto& k : keys) list += (list.empty() ? "" : ", ") + k;
    return std::format("door_concept: config REJECTED — retired key(s) present: {}. The old existence patches (seed, clamp, "
                       "debounce, re-acquisition registry, streak births) were replaced by the generative model of "
                       "docs/superpowers/plans/2026-11-07-door-existence-model.md; remove these keys from the config.\n", list);
}
}  // namespace rc
