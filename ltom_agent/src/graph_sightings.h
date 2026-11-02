// Live-graph reader: object instances -> Sightings, plus the concept agents currently present.
// Read-only; main thread (it uses the caller's InnerEigenAPI at ts 0, whose cache the main thread owns).
#pragma once
#include "repertory.h"

#include <dsr/api/dsr_api.h>

#include <optional>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace ltom
{
struct GraphView { std::vector<Sighting> alive; std::set<std::uint32_t> owners_present;
                 std::set<std::uint64_t> present; };   // present: every in-scope uid on the graph, resolvable or not

// ltsm_agent's room_index() rule: the room_id attribute, else the trailing digits of the name. nullopt = no key.
std::optional<std::int64_t> room_key_of(DSR::DSRGraph& G, const DSR::Node& room);

GraphView read_graph(DSR::DSRGraph& G, DSR::InnerEigenAPI& ie, std::span<const std::string> classes);
}
