// input_recorder.h — writes door_concept's per-cycle DoorCycleInput to a "DCIN" v1 file (core/door_cycle_io.h).
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
// No locale anywhere (binary); DSR-free, so tests/test_input_recorder.cpp builds it with plain g++.
#pragma once
#include <cstdint>
#include <format>
#include <fstream>
#include <print>
#include <sstream>
#include <string>
#include "core/door_cycle_io.h"
#include "../../common/diag_log/rotating_csv.h"

namespace rc::door
{

class DoorInputRecorder
{
public:
    /// Open `path` fresh (rotating the previous run's file) and write the header. `max_bytes` = 0 means no cap.
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
        bytes_ = h.size();
        if (not out_.good())
        {
            out_.close();
            return false;
        }
        return true;
    }

    [[nodiscard]] bool enabled() const { return out_.is_open(); }
    [[nodiscard]] bool stopped() const { return stopped_; }
    [[nodiscard]] std::uint64_t bytes() const { return bytes_; }
    [[nodiscard]] std::uint64_t records() const { return records_; }

    /// Append one cycle. false when not recording (off, stopped at the cap, or a stream error — which also stops).
    bool record(const core::DoorCycleInput& in)
    {
        if (not enabled()) return false;
        std::ostringstream rs(std::ios::binary);
        if (not core::write(rs, in))
            return stop("could not serialise cycle " + std::to_string(in.cycle));
        const std::string r = rs.str();
        if (max_bytes_ > 0 and bytes_ + r.size() > max_bytes_)
            return stop(std::format("size cap {:.1f} MB ({} B) reached at cycle {} ({} records, {} bytes) — recording STOPPED, the "
                                    "file ends on the last whole record", double(max_bytes_) / (1 << 20), max_bytes_, in.cycle, records_, bytes_));
        out_.write(r.data(), static_cast<std::streamsize>(r.size()));
        out_.flush();   // a SIGKILLed run still leaves every complete record on disk
        if (not out_.good())
            return stop("write failed at cycle " + std::to_string(in.cycle) + " (disk full?)");
        bytes_ += r.size();
        ++records_;
        return true;
    }

    /// Flush and close (the SIGTERM shutdown path calls this).
    void close()
    {
        if (out_.is_open())
        {
            out_.flush();
            out_.close();
        }
    }

private:
    bool stop(const std::string& why)
    {
        std::println("door_concept: [record] {}: {}", path_, why);
        stopped_ = true;
        close();
        return false;
    }

    std::ofstream out_;
    std::string   path_;
    std::uint64_t max_bytes_ = 0, bytes_ = 0, records_ = 0;
    bool          stopped_ = false;
};

}  // namespace rc::door
