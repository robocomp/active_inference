#pragma once
/*
 * em_metrics.h — episodic_agent's per-run instrumentation (final review, finding 14): one CSV per run,
 * <archive_dir>/metrics_<YYYYMMDD>_<HHMMSS>.csv, one row every `period_ms` (10 s), so the journal's size / rate and the
 * records' growth are measured from a file, never from pasted console lines. Header-only, std + em_schema only.
 *
 * Columns:
 *   t                           ms since the epoch of the row
 *   journal_bytes               size of the last journal wire read
 *   journal_updates_since_last  journal wires that CHANGED since the previous row
 *   records_open                Builder::open().size() at the row
 *   records_closed_total        closed records written this run
 *   max_record_bytes            largest compact-JSON record among those written since the previous row and those open now
 *   max_events_per_record       most events on one record, same set
 *   poll_ms_last                wall time of the last poll() (ms)
 * Written through imbue(std::locale::classic()) (CLAUDE.md: es_ES would otherwise risk a comma decimal separator).
 */

#include "../../common/episodic/em_schema.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <locale>
#include <span>
#include <string>

#include <QJsonDocument>

namespace rc::em
{

class Metrics
{
public:
    static constexpr const char* kHeader = "t,journal_bytes,journal_updates_since_last,records_open,records_closed_total,"
                                           "max_record_bytes,max_events_per_record,poll_ms_last";

    /// "<dir>/metrics_<YYYYMMDD>_<HHMMSS>.csv" for a run starting at t_ms (local time).
    static std::filesystem::path path_for(const std::filesystem::path& dir, std::uint64_t t_ms)
    {
        using namespace std::chrono;
        const auto lt = zoned_time{current_zone(), sys_time<milliseconds>{milliseconds{std::int64_t(t_ms)}}}.get_local_time();
        const auto day = floor<days>(lt);
        const year_month_day ymd{day};
        const hh_mm_ss hms{floor<seconds>(lt - day)};
        return dir / std::format("metrics_{:04}{:02}{:02}_{:02}{:02}{:02}.csv", int(ymd.year()), unsigned(ymd.month()),
                                 unsigned(ymd.day()), hms.hours().count(), hms.minutes().count(), hms.seconds().count());
    }

    Metrics(const std::filesystem::path& file, std::uint64_t start_ms, std::uint64_t period_ms = 10000)
        : period_(period_ms), next_(start_ms + period_ms)
    {
        std::error_code ec;
        std::filesystem::create_directories(file.parent_path(), ec);
        out_.open(file, std::ios::app);
        out_.imbue(std::locale::classic());
        if (out_) out_ << kHeader << '\n' << std::flush;
    }

    bool ok() const { return bool(out_); }

    void on_wire(std::size_t bytes) { journal_bytes_ = bytes; ++updates_; }
    void on_written(std::span<const Record> rs)
    {
        for (const Record& r : rs)
        {
            look(r);
            if (r.t1) ++closed_total_;
        }
    }
    void on_poll(double ms) { poll_ms_ = ms; }

    /// Appends a row when one is due (`open` = the records still open now). True when it wrote.
    bool maybe_row(std::uint64_t now_ms, std::span<const Record> open)
    {
        if (now_ms < next_) return false;
        for (const Record& r : open) look(r);
        if (out_)
        {
            out_ << now_ms << ',' << journal_bytes_ << ',' << updates_ << ',' << open.size() << ',' << closed_total_ << ','
                 << max_bytes_ << ',' << max_events_ << ',' << poll_ms_ << '\n' << std::flush;
        }
        updates_ = 0;
        max_bytes_ = max_events_ = 0;
        while (next_ <= now_ms) next_ += period_;
        return true;
    }

private:
    std::ofstream out_;
    std::uint64_t period_, next_;
    std::size_t journal_bytes_ = 0, updates_ = 0, closed_total_ = 0, max_bytes_ = 0, max_events_ = 0;
    double poll_ms_ = 0;

    void look(const Record& r)
    {
        max_bytes_ = std::max<std::size_t>(max_bytes_, std::size_t(QJsonDocument(to_json(r)).toJson(QJsonDocument::Compact).size()));
        max_events_ = std::max(max_events_, r.events.size());
    }
};

}   // namespace rc::em
