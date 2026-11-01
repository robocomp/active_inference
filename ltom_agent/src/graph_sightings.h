// Live-graph reader: object instances -> Sightings, plus the concept agents currently present.
// Read-only; main thread (it uses the caller's InnerEigenAPI at ts 0, whose cache the main thread owns).
#pragma once
#include "repertory.h"

#include <dsr/api/dsr_api.h>

#include <set>
#include <span>
#include <string>
#include <vector>

namespace ltom
{
struct GraphView { std::vector<Sighting> alive; std::set<std::uint32_t> owners_present; };

GraphView read_graph(DSR::DSRGraph& G, DSR::InnerEigenAPI& ie, std::span<const std::string> classes);
}
