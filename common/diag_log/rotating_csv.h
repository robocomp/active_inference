/*
 * rotating_csv.h — open a diagnostics CSV without destroying the previous run's.
 *
 * WHY THIS EXISTS (2026-08-10). Every agent here opens its CSVs with `std::ios::trunc`, so each
 * restart wipes the log. That is fine for a reproducible fault and useless for an intermittent one:
 * we were chasing a cabinet that fitted itself 90 cm deep instead of 60, went to read the run that
 * produced it, and found the file had been truncated from 31 MB to 1.3 MB by a restart minutes
 * earlier. The recorder erases the tape whenever the thing being recorded restarts — and the
 * restart is often provoked BY the fault, so the interesting run is exactly the one that gets lost.
 *
 * Rotating instead of truncating keeps the current filename stable (so every existing plot script
 * and `tail -f` still works) while the previous runs survive beside it as `<name>.<stamp>`.
 *
 * Header-only, no DSR, no Qt. Locale-safe: the stream is imbued with the classic locale so a file
 * written here can never acquire a comma decimal separator (see CLAUDE.md's locale note) — the
 * readers all use std::from_chars, but the writer should not depend on that.
 */

#pragma once

#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <locale>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace rc::diag {

// Keep this many previous runs beside the live file. These logs run to tens of MB per hour, so the
// cap is a disk-space guard, not a policy about how much history is interesting.
inline constexpr int kKeepPreviousRuns = 3;

// Rename `path` (if it exists and is non-empty) to `path.YYYYmmdd_HHMMSS`, delete all but the most
// recent kKeepPreviousRuns of those, then open `path` fresh and write `header` if given.
// Returns true if the stream is open. Every failure is non-fatal: diagnostics must never be able to
// take an agent down, so a rotation that cannot happen just logs and proceeds to open normally.
inline bool open_rotating(std::ofstream& out, const std::string& path, std::string_view header = {})
{
    namespace fs = std::filesystem;
    std::error_code ec;

    if (fs::exists(path, ec) and not ec and fs::file_size(path, ec) > 0 and not ec)
    {
        const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::tm tm{};
        localtime_r(&now, &tm);
        char stamp[32];
        std::strftime(stamp, sizeof stamp, "%Y%m%d_%H%M%S", &tm);
        // ★Uniquify within the second. A one-second stamp is NOT enough: a crash-loop restarts many
        // times per second, every rotation lands on the same name, and each rename silently
        // overwrites the last — so the one situation where the history matters most is exactly the
        // one that loses it. (Measured: six restarts in the same second left one archive.)
        std::string rotated = path + "." + stamp;
        for (int n = 1; fs::exists(rotated, ec) and n < 1000; ++n)
            rotated = path + "." + stamp + "_" + std::to_string(n);
        fs::rename(path, rotated, ec);
        if (ec)
            std::printf("[diag] could not rotate '%s' (%s) — it will be overwritten\n",
                        path.c_str(), ec.message().c_str());
        else
            std::printf("[diag] previous run kept as '%s'\n", rotated.c_str());
    }

    // Prune: collect siblings named "<file>.<something>" and drop the oldest beyond the cap.
    {
        const fs::path p{path};
        const std::string prefix = p.filename().string() + ".";
        const fs::path dir = p.has_parent_path() ? p.parent_path() : fs::path{"."};
        std::vector<fs::path> old;
        for (fs::directory_iterator it{dir, ec}, end; not ec and it != end; it.increment(ec))
            if (it->is_regular_file(ec) and it->path().filename().string().rfind(prefix, 0) == 0)
                old.push_back(it->path());
        std::sort(old.begin(), old.end());                     // the stamp sorts chronologically
        for (std::size_t i = 0; old.size() > kKeepPreviousRuns and i < old.size() - kKeepPreviousRuns; ++i)
            fs::remove(old[i], ec);
    }

    out.open(path, std::ios::out | std::ios::trunc);
    if (not out.is_open())
        return false;
    out.imbue(std::locale::classic());
    if (not header.empty())
        out << header;
    return true;
}

}  // namespace rc::diag
