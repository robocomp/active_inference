// input_recorder.h — writes door_concept's per-cycle DoorCycleInput to a "DCIN" file (core/door_cycle_io.h, v2).
//
// ★WHY. The DoorCore replay (docs/superpowers/specs/2026-10-21-door-core-replay-design.md §3.3) runs the SAME core the
// agent runs over exactly what one live compute() received. This is the tape. DEFAULT OFF (DoorConcept.RecordInputs).
//
// ★ONE FILE PER RUN, the previous runs kept beside it (rc::diag::open_rotating: `<path>.<stamp>`, the 3 newest).
// ★THE SIZE CAP STOPS, IT DOES NOT WRAP (spec §6.5). The start of a run — the identity load, the first room, the
//   restores — is what a regression needs, so at DoorConcept.RecordMaxMB the recorder writes nothing more, closes the
//   file and says so once. A record that would cross the cap is NOT written: the file ends on a record boundary.
// ★A RECORD IS ON DISK WHOLE OR NOT AT ALL. Each record is serialised to memory first, then issued as one write and
//   flushed, so a SIGKILL or a full disk leaves every complete record readable (read_input_file reports Truncated).
// ★ENCODING + WRITING RUN ON A WORKER THREAD (v2, Task 3 live size). Packing a record (~1.5 MB of arrays through zstd)
//   and writing it cost milliseconds; on compute()'s thread that would shift the very timing the replay compares. So
//   record() MOVES the finished input into a queue and returns; the worker serialises, applies the cap and writes, in
//   order. Nothing is shared: the input is moved, never referenced. If the worker falls kQueueMax records behind,
//   record() WAITS (back-pressure) — a dropped record would be a silent hole in the tape, which is worse than a late
//   cycle. A SIGKILL loses at most the queued records; close() (the SIGTERM path) drains the queue first.
// No locale anywhere (binary); DSR-free, so tests/test_input_recorder.cpp builds it with plain g++ (-lzstd).
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <format>
#include <fstream>
#include <mutex>
#include <print>
#include <sstream>
#include <string>
#include <thread>
#include "core/door_cycle_io.h"
#include "../../common/diag_log/rotating_csv.h"

namespace rc::door
{

class DoorInputRecorder
{
public:
    static constexpr std::size_t kQueueMax = 32;

    DoorInputRecorder() = default;
    DoorInputRecorder(const DoorInputRecorder&) = delete;
    DoorInputRecorder& operator=(const DoorInputRecorder&) = delete;
    ~DoorInputRecorder() { close(); }

    /// Open `path` fresh (rotating the previous run's file), write the header, start the writer thread.
    /// `max_bytes` = 0 means no cap.
    [[nodiscard]] bool open(const std::string& path, const core::DoorRecordingHeader& header, std::uint64_t max_bytes)
    {
        close();
        path_ = path;
        max_bytes_ = max_bytes;
        bytes_ = records_ = 0;
        stopped_ = false;
        if (not rc::diag::open_rotating(out_, path))
            return false;
        std::ostringstream hs(std::ios::binary);
        if (not core::write_header(hs, header, core::RecordingKind::Input))
        {
            out_.close();
            return false;
        }
        const std::string h = hs.str();
        out_.write(h.data(), static_cast<std::streamsize>(h.size()));
        out_.flush();
        if (not out_.good())
        {
            out_.close();
            return false;
        }
        bytes_ = h.size();
        open_ = true;
        quit_ = false;
        worker_ = std::thread([this] { run(); });
        return true;
    }

    /// Recording: open and not stopped (at the cap or on a write error).
    [[nodiscard]] bool enabled() const { return open_ and not stopped_; }
    [[nodiscard]] bool stopped() const { return stopped_; }
    [[nodiscard]] std::uint64_t bytes() const { return bytes_; }       // written so far (the worker's count)
    [[nodiscard]] std::uint64_t records() const { return records_; }

    /// Queue one cycle (moved). false when not recording. The cap / write outcome is the worker's: see stopped().
    bool record(core::DoorCycleInput&& in)
    {
        if (not enabled()) return false;
        std::unique_lock lk(m_);
        cv_space_.wait(lk, [this] { return q_.size() < kQueueMax or stopped_; });
        if (stopped_) return false;
        q_.push_back(std::move(in));
        cv_work_.notify_one();
        return true;
    }
    bool record(const core::DoorCycleInput& in) { auto c = in; return record(std::move(c)); }

    /// Wait until every queued record is written (or dropped by a stop).
    void flush()
    {
        std::unique_lock lk(m_);
        cv_space_.wait(lk, [this] { return (q_.empty() and not busy_) or not worker_.joinable(); });
    }

    /// Drain, stop the worker, close the file (the SIGTERM shutdown path calls this).
    void close()
    {
        if (worker_.joinable())
        {
            {
                std::lock_guard lk(m_);
                quit_ = true;
            }
            cv_work_.notify_one();
            worker_.join();
        }
        if (out_.is_open())
        {
            out_.flush();
            out_.close();
        }
        open_ = false;
    }

private:
    void run()
    {
        for (;;)
        {
            core::DoorCycleInput in;
            {
                std::unique_lock lk(m_);
                cv_work_.wait(lk, [this] { return not q_.empty() or quit_; });
                if (q_.empty()) { busy_ = false; cv_space_.notify_all(); return; }   // quit with nothing left
                in = std::move(q_.front());
                q_.pop_front();
                busy_ = true;
            }
            // ★An exception here (bad_alloc while packing, a stream that throws) must stop the RECORDING, not the agent:
            //  escaping a std::thread's function is std::terminate.
            if (not stopped_)
                try { write_one(in); }
                catch (const std::exception& e) { stop(std::string("exception while writing cycle ") + std::to_string(in.cycle) + ": " + e.what()); }
                catch (...) { stop("unknown exception while writing cycle " + std::to_string(in.cycle)); }
            {
                std::lock_guard lk(m_);
                busy_ = false;
                if (stopped_) q_.clear();   // nothing more is written after a stop
            }
            cv_space_.notify_all();
        }
    }

    void write_one(const core::DoorCycleInput& in)
    {
        std::ostringstream rs(std::ios::binary);
        if (not core::write(rs, in))
            return stop("could not serialise cycle " + std::to_string(in.cycle));
        const std::string r = rs.str();
        if (max_bytes_ > 0 and bytes_ + r.size() > max_bytes_)
            return stop(std::format("size cap {:.1f} MB ({} B) reached at cycle {} ({} records, {} bytes) — recording STOPPED, the "
                                    "file ends on the last whole record", double(max_bytes_) / (1 << 20), max_bytes_, in.cycle,
                                    records_.load(), bytes_.load()));
        out_.write(r.data(), static_cast<std::streamsize>(r.size()));
        out_.flush();   // a SIGKILLed run still leaves every complete record on disk
        if (not out_.good())
            return stop("write failed at cycle " + std::to_string(in.cycle) + " (disk full?)");
        bytes_ += r.size();
        ++records_;
    }

    void stop(const std::string& why)
    {
        std::println("door_concept: [record] {}: {}", path_, why);
        stopped_ = true;
        out_.flush();
        out_.close();
    }

    std::ofstream out_;
    std::string   path_;
    std::uint64_t max_bytes_ = 0;
    std::atomic<std::uint64_t> bytes_{0}, records_{0};
    std::atomic<bool> stopped_{false}, open_{false};

    std::thread worker_;
    std::mutex m_;
    std::condition_variable cv_work_, cv_space_;
    std::deque<core::DoorCycleInput> q_;
    bool quit_ = false, busy_ = false;
};

}  // namespace rc::door
