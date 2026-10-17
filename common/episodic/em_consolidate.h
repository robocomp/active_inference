#pragma once
/*
 * em_consolidate.h — folds CLOSED episodic records into all-time statistics keyed by (level, kind, author, target_ltsm)
 * (2026-10-17). Pure, header-only. An open record is never folded; a record's day is day_of(t0).
 */

#include <algorithm>
#include <compare>
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

/// level name, kind, author ("human"|"self" for missions, "" otherwise), identity (target_ltsm, "" = per kind only).
struct StatsKey
{
    std::string level, kind, author, identity;
    auto operator<=>(const StatsKey&) const = default;
    bool operator==(const StatsKey&) const = default;
};

inline StatsKey stats_key(const Record& r)
{
    return {detail::name(r.level), r.kind,
            r.level == Level::Mission ? std::string(r.author == Author::Human ? "human" : "self") : std::string{},
            r.target_ltsm};
}

/// The outcomes EM writes itself: their t1 is EM's stop / the owner's disappearance, not the commitment's end.
inline bool em_closed(const Record& r) { return r.outcome == "lost" or r.outcome == "interrupted_by_em_shutdown"; }

/// Precondition: r.t1 has value. Counts every record; Welford over (t1-t0) in seconds only for records not em_closed;
/// gain sums only when promised AND delivered exist.
inline void fold(Stats& st, const Record& r)
{
    if (st.n == 0)
    {
        const StatsKey k = stats_key(r);
        st.key_level = k.level;
        st.key_kind = k.kind;
        st.key_author = k.author;
        st.key_identity = k.identity;
        st.first_t0 = st.last_t0 = r.t0;
    }
    st.first_t0 = std::min(st.first_t0, r.t0);
    st.last_t0 = std::max(st.last_t0, r.t0);
    ++st.outcomes[r.outcome];
    ++st.n;
    if (not em_closed(r))
    {
        const double d = (double(*r.t1) - double(r.t0)) / 1000.0;
        ++st.n_dur;
        const double delta = d - st.dur_mean_s;
        st.dur_mean_s += delta / st.n_dur;
        st.dur_m2 += delta * (d - st.dur_mean_s);
    }
    if (r.promised and r.delivered)
    {
        ++st.n_gain;
        st.promised_sum += *r.promised;
        st.delivered_sum += *r.delivered;
    }
}

/// Folds every CLOSED record of days < today ("YYYYMMDD", compared as text) into `stats`; returns the ids folded.
inline std::vector<std::string> rollover(std::map<StatsKey, Stats>& stats,
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
