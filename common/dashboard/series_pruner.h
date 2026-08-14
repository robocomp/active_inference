/*
 * common/dashboard/series_pruner.h — drop the timeseries of instances that are gone. SHARED, header-only.
 *
 * When an object is removed from the graph its lines must leave the plots, or a dead instance's traces sit
 * there forever and an object REBORN under the same name inherits them. Four agents wrote the same set-diff
 * over their live instance names against a remembered set, removing the same five suffixed series.
 *
 * The agent supplies the live names and its own remembered set; everything else is identical. The plots are
 * passed as pointers because a headless agent (show_dashboard = false) has none, and the whole thing must
 * no-op rather than branch at every call site — that is how the copies were written and it is correct.
 */

#pragma once

#include <string>
#include <unordered_set>
#include <vector>

namespace rc { class TimeSeriesPlot; }

namespace rc::dashboard
{

// The plots an object's series live on. Any of them may be null (headless, or an agent with fewer panels).
// fe_plot carries BOTH "<name>_fe" and "<name>_base" — one panel, two lines, as all four agents had it.
struct SeriesPlots
{
    rc::TimeSeriesPlot* fe       = nullptr;
    rc::TimeSeriesPlot* surprise = nullptr;
    rc::TimeSeriesPlot* cov      = nullptr;
    rc::TimeSeriesPlot* res      = nullptr;
};

// Remove the series of every remembered name no longer in `live`, then remember the live set. Runs every
// compute cycle; cheap (a set diff over the few live instances). Main-thread only (touches widgets).
//
// Declared in the header and defined in series_pruner.cpp so this header stays free of the TimeSeriesPlot
// definition — the agents that include it from a non-Qt translation unit keep compiling.
void prune_dead_series(const SeriesPlots& plots,
                       const std::vector<std::string>& live,
                       std::unordered_set<std::string>& known);

}  // namespace rc::dashboard
