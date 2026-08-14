/*
 *    common/dashboard/series_pruner.cpp — see series_pruner.h
 */

#include "series_pruner.h"

#include "timeseries_plot.h"

namespace rc::dashboard
{

void prune_dead_series(const SeriesPlots& plots,
                       const std::vector<std::string>& live,
                       std::unordered_set<std::string>& known)
{
    if (not plots.fe)
        return;   // headless: nothing was built

    const std::unordered_set<std::string> live_set(live.begin(), live.end());
    for (auto it = known.begin(); it != known.end();)
    {
        if (live_set.contains(*it)) { ++it; continue; }
        const std::string& n = *it;
        plots.fe->remove_series(n + "_fe");
        plots.fe->remove_series(n + "_base");
        if (plots.surprise) plots.surprise->remove_series(n + "_surprise");
        if (plots.cov)      plots.cov->remove_series(n + "_cov");
        if (plots.res)      plots.res->remove_series(n + "_res");
        it = known.erase(it);
    }
    known.insert(live.begin(), live.end());
}

}  // namespace rc::dashboard
