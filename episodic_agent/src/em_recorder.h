#pragma once
/*
 * em_recorder.h — episodic_agent's bookkeeping between the Builder and its two stores (2026-10-17): the append-only
 * Archive (durable truth) and a RecordSink (the domain-4 graph, src/em_graph.h). Header-only, Qt Core only, so it is
 * tested without DSR (tests/recorder_test.cpp).
 *
 * FOLD EACH RECORD EXACTLY ONCE (rollover is not idempotent — Task 3 review):
 *   - start(now): first, every record a crashed previous run left OPEN in the archive (any day) is closed `lost` at
 *     `now` and its closing line appended (ruling R21: nobody else will ever close it). Then the all-time stats are
 *     RECOMPUTED FROM SCRATCH by folding every closed record of every past day; today's records are reseeded into the
 *     sink; the stats are upserted. Safe to call twice (the second call finds nothing open and refolds from zero).
 *   - write(): a record whose day is AFTER today is the day change: the closed records of the old day are folded
 *     (rollover), their nodes deleted, today advances. A record of an already-rolled day that CLOSES later (opened
 *     before midnight) is folded at that close and its node deleted. Everything else is upserted.
 * Shutdown-closed records (`interrupted_by_em_shutdown`) go through write() like any other: archive AND graph (R22).
 * Ids: "<day of t0>-<seq>", seq continuing after the archive's largest for that day.
 */

#include "../../common/episodic/em_archive.h"
#include "../../common/episodic/em_consolidate.h"
#include "../../common/episodic/em_schema.h"

#include <algorithm>
#include <cstdint>
#include <format>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace rc::em
{

struct RecordSink                                // the domain-4 graph in the agent; faked in tests
{
    virtual ~RecordSink() = default;
    virtual void upsert(const Record& r) = 0;
    virtual void upsert(const Stats& s) = 0;
    virtual void remove(const std::string& id) = 0;
};

class Recorder
{
public:
    using Key = std::pair<std::string, std::string>;

    Recorder(Archive& archive, RecordSink& sink, std::string today)
        : archive_(archive), sink_(sink), today_(std::move(today)) {}

    /// Returns how many crash-open records it closed `lost`.
    int start(std::uint64_t now_ms)
    {
        int closed = 0;
        for (const std::string& d : archive_.days())
            for (Record r : archive_.load_day(d))
                if (not r.t1)
                {
                    r.t1 = std::max(now_ms, r.t0);
                    r.outcome = "lost";
                    archive_.append(r);
                    ++closed;
                }
        stats_.clear();
        for (const std::string& d : archive_.days())
            if (d < today_) rollover(stats_, archive_.load_day(d), today_);
        for (const Record& r : archive_.load_day(today_)) sink_.upsert(r);
        for (const auto& [k, st] : stats_) sink_.upsert(st);
        return closed;
    }

    std::string next_id(std::uint64_t t0)
    {
        const std::string day = day_of(t0);
        auto it = seq_.find(day);
        if (it == seq_.end()) it = seq_.emplace(day, archive_.next_seq(day)).first;
        return std::format("{}-{}", day, it->second++);
    }

    /// Archive every record and keep the sink and the stats current.
    void write(const std::vector<Record>& records)
    {
        for (const Record& r : records)
        {
            const std::string day = day_of(r.t0);
            if (day > today_) change_day(day);
            archive_.append(r);
            if (day < today_ and r.t1)          // its day already rolled over: fold it now, once
            {
                Stats& st = stats_[stats_key(r)];
                fold(st, r);
                sink_.upsert(st);
                sink_.remove(r.id);
            }
            else sink_.upsert(r);
        }
    }

    const std::map<Key, Stats>& stats() const { return stats_; }
    const std::string& today() const { return today_; }

private:
    Archive& archive_;
    RecordSink& sink_;
    std::string today_;
    std::map<Key, Stats> stats_;
    std::map<std::string, int> seq_;            // day -> next seq

    void change_day(const std::string& day)
    {
        const std::vector<Record> old = archive_.load_day(today_);
        const std::vector<std::string> folded = rollover(stats_, old, day);
        std::set<Key> touched;
        for (const Record& r : old)
            if (std::ranges::find(folded, r.id) != folded.end()) touched.insert(stats_key(r));
        for (const Key& k : touched) sink_.upsert(stats_.at(k));
        for (const std::string& id : folded) sink_.remove(id);
        today_ = day;
    }
};

}   // namespace rc::em
