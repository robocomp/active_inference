// ── WHICH LOG LINES STILL REACH THE TERMINAL — room_concept's one routing table ──────────────────────
// With the status stream ON, EVERY line (qInfo/qWarning/qCritical through the chained Qt handler, and
// every rc::status::print/println/cprintf site) is recorded as a `log` event in
// tmp/agent_events_<start>.jsonl and shown by tools/room_tui.py. This table decides only whether the
// line is ALSO printed on stdout/stderr. Nothing is deleted; it is routed.
//
// With the stream OFF ([Status] Enable = false) this table is not consulted at all: every line prints
// exactly as it did before the stream existed.
//
// The terminal keeps: the config_report banner (printed directly, not routed), start/stop lifecycle
// lines, state-machine transitions, every critical/fatal line, and warnings that need a human now
// (presence lost, a stream stalled, a mount refused, ...). Periodic and diagnostic output leaves it.
//
// TO CHANGE WHAT YOU SEE: edit the three lists below. A tag is the leading "[...]" of a line
// (rc::StatusStream::tag_of), matched exactly; a line with no tag is "".
#pragma once

#include "../../common/status_stream/status_sink.h"

#include <algorithm>
#include <array>
#include <string_view>

namespace rc::room_routing
{
using namespace std::string_view_literals;

/// Tags whose lines reach the terminal at EVERY level (lifecycle and state machine).
inline constexpr std::array kTerminalTags = {
    "SM"sv,           // state-machine transitions and the "why still Waiting" line
    "Presence"sv,     // peers lost/ready/restarted
    "Shutdown"sv,     // stop lifecycle
    "lifecycle"sv,    // start lifecycle (start_status_stream)
    "SAFETY"sv,       // non-finite pose fallback
};

/// Tags whose WARNINGS are still periodic diagnostics and leave the terminal anyway. Their lines
/// remain in the events file and the viewer's Warnings tab.
inline constexpr std::array kQuietWarningTags = {
    "Timing"sv, "Compute"sv, "pumps"sv, "LidarSrc"sv, "band"sv,
    "level2"sv, "exist-kill"sv, "death-exist"sv, "refine"sv,
    // room_boxes_channel's per-frame box-layout traces, written to stderr (hence Warning level)
    // although they are diagnostics, not warnings:
    "cover"sv, "comp"sv, "G"sv, "gauge"sv, "gauge-profile"sv, "cloud"sv, "boxes"sv, "reg"sv, "adopt"sv,
};

/// Warnings carrying one of these words reach the terminal even under a quiet tag.
inline constexpr std::array kUrgentWords = {
    "REFUS"sv, "STALL"sv, "LOST"sv, "lost"sv, "FAIL"sv, "cannot"sv, "could not"sv,
};

inline bool urgent(std::string_view line)
{
    return std::ranges::any_of(kUrgentWords, [&](std::string_view w) { return line.find(w) != std::string_view::npos; });
}

/// The routing decision (thread-safe: a pure function of its arguments). Critical/Fatal never reach
/// here — rc::StatusStream always prints them.
inline bool to_terminal(status::Level level, std::string_view tag, std::string_view line)
{
    if (std::ranges::find(kTerminalTags, tag) != kTerminalTags.end())
        return true;
    if (level >= status::Level::Warning)
        return std::ranges::find(kQuietWarningTags, tag) == kQuietWarningTags.end() or urgent(line);
    return false;   // info/debug: the viewer and the events file only
}

}  // namespace rc::room_routing
