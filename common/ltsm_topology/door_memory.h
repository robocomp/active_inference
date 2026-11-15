// door_memory.h — the two door facts that outlive a run and are not the belief of any one door (2026-11-15): the next fresh
// door name and the door-change hazard. door_concept writes them on its own agent node (door_next_index_att,
// door_change_hazard_att); ltsm keeps the merge of everything it has seen on its memory graph's root, persists it with the
// memory, and hands it back in ltsm_topology ("door_memory"), which door_concept adopts at start. Before this they lived in
// door_concept's private etc/door_identities.csv. PURE: no graph, no Qt.
#pragma once

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <optional>
#include <string_view>

namespace rc::doormem
{
struct DoorMemory
{
    std::uint64_t next_index = 0;   ///< the next fresh door_<N>: every N below it may already name a door somewhere
    double changes = 0.0;           ///< the hazard's Gamma posterior: door changes found on revisits ...
    double exposure_s = 0.0;        ///< ... over this much summed exposure (s)
    bool operator==(const DoorMemory&) const = default;
};

/// Two records of the same memory: both parts only ever grow, so the merge keeps the larger counter and the hazard with the
/// larger exposure (it has seen everything the other saw, and more). A tie on exposure keeps the larger change count.
[[nodiscard]] inline DoorMemory merged(const DoorMemory& a, const DoorMemory& b)
{
    DoorMemory m;
    m.next_index = std::max(a.next_index, b.next_index);
    const bool a_hazard = a.exposure_s > b.exposure_s or (a.exposure_s == b.exposure_s and a.changes >= b.changes);
    m.changes = a_hazard ? a.changes : b.changes;
    m.exposure_s = a_hazard ? a.exposure_s : b.exposure_s;
    return m;
}

/// The N of a door name: door_<N>, with any suffix (door_5_h1, door_7_entry) — nullopt for any other name. A live name, not a
/// memory one (strip ltsm's r<idx>_ prefix first).
[[nodiscard]] inline std::optional<std::uint64_t> door_number(std::string_view name)
{
    constexpr std::string_view kPrefix = "door_";
    if (not name.starts_with(kPrefix)) return std::nullopt;
    const char* b = name.data() + kPrefix.size();
    const char* e = name.data() + name.size();
    std::uint64_t n = 0;
    const auto [p, ec] = std::from_chars(b, e, n);
    if (ec != std::errc{} or p == b or (p != e and *p != '_')) return std::nullopt;
    return n;
}

/// The counter floor a set of door names imposes: one past the largest N among them (0 when none is a door name).
template <class Names>
[[nodiscard]] std::uint64_t next_index_floor(const Names& names)
{
    std::uint64_t f = 0;
    for (const auto& nm : names)
        if (const auto n = door_number(std::string_view(nm)); n.has_value()) f = std::max(f, *n + 1);
    return f;
}
}   // namespace rc::doormem
