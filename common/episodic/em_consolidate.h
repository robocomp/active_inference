#pragma once
/*
 * em_consolidate.h — folds CLOSED episodic records into all-time statistics keyed by (kind, target_ltsm)
 * (2026-10-17). Pure, header-only. An open record is never folded; a record's day is day_of(t0).
 */

#include <algorithm>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "em_schema.h"

namespace rc::em
{

/// (kind, target_ltsm); an empty target_ltsm means the record counts per kind only.
inline std::pair<std::string, std::string> stats_key(const Record& r) { return {r.kind, r.target_ltsm}; }

/// Precondition: r.t1 has value. Welford over (t1-t0) in seconds; gain sums only when promised AND delivered exist.
inline void fold(Stats& st, const Record& r)
{
    const auto [kind, identity] = stats_key(r);
    if (st.n == 0)
    {
        st.key_kind = kind;
        st.key_identity = identity;
        st.first_t0 = st.last_t0 = r.t0;
    }
    st.first_t0 = std::min(st.first_t0, r.t0);
    st.last_t0 = std::max(st.last_t0, r.t0);
    ++st.outcomes[r.outcome];
    const double d = (double(*r.t1) - double(r.t0)) / 1000.0;
    ++st.n;
    const double delta = d - st.dur_mean_s;
    st.dur_mean_s += delta / st.n;
    st.dur_m2 += delta * (d - st.dur_mean_s);
    if (r.promised and r.delivered)
    {
        ++st.n_gain;
        st.promised_sum += *r.promised;
        st.delivered_sum += *r.delivered;
    }
}

/// Folds every CLOSED record of days < today ("YYYYMMDD", compared as text) into `stats`; returns the ids folded.
inline std::vector<std::string> rollover(std::map<std::pair<std::string, std::string>, Stats>& stats,
                                         std::span<const Record> records, std::string_view today)
{
    std::vector<std::string> folded;
    for (const Record& r : records)
    {
        if (not r.t1 or std::string_view{day_of(r.t0)} >= today) continue;
        fold(stats[stats_key(r)], r);
        folded.push_back(r.id);
    }
    return folded;
}

}   // namespace rc::em
