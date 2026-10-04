// ── rc::StatusReporter — room_concept's side of the status stream ───────────────────────────────────
// common/status_stream/ is GENERIC: it carries events and knows nothing about rooms. This class is
// where room_concept decides WHAT to say: the structured events its startup and state machine emit
// (phase, sm, wait, peer, stall, loaded), and the commands a viewer may send.
//
// Every method is a no-op when the stream is off ([Status] Enable = false): the events go through
// rc::status::event(), which does nothing with no sink installed. So a call site never needs a guard.
//
// Viewer: room_concept/tools/room_tui.py (launcher: room_concept/tools/room_run.sh).
#pragma once

#include "../../common/status_stream/status_sink.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace rc
{

class StatusReporter
{
public:
    // ── startup ──
    static void phase(std::string_view name, long long ms, long long cum_ms)
    {
        status::event("phase", status::Obj{}.s("name", name).i("ms", ms).i("cum_ms", cum_ms));
    }
    /// Something read from disk (or deliberately NOT read) that shapes this run.
    static void loaded(std::string_view what, std::string_view path, std::string_view detail = {})
    {
        status::event("loaded", status::Obj{}.s("what", what).s("path", path).s("detail", detail));
    }

    // ── state machine ──
    static void sm(std::string_view state, std::string_view detail = {})
    {
        status::event("sm", status::Obj{}.s("state", state).s("detail", detail));
    }
    /// The "why are we still Waiting" line, as data. Emitted only when it CHANGES, so the replay ring
    /// holds the history of reasons, not one copy per throttle period.
    void waiting(bool peers_ok, const std::vector<std::string>& missing, bool lidar_ok, std::string_view why)
    {
        std::string key = std::string(peers_ok ? "1" : "0") + (lidar_ok ? "1" : "0") + std::string(why);
        for (const auto& m : missing) key += "|" + m;
        if (key == last_wait_key_) return;
        last_wait_key_ = std::move(key);
        status::Arr miss;
        for (const auto& m : missing) miss.s(m);
        status::event("wait", status::Obj{}
                                  .b("peers_ok", peers_ok)
                                  .raw("missing", miss.str())
                                  .b("lidar_ok", lidar_ok)
                                  .s("why", why));
    }
    void reset_waiting() { last_wait_key_.clear(); }

    // ── peers and streams ──
    static void peer(std::string_view event, std::string_view name, std::uint32_t id)
    {
        status::event("peer", status::Obj{}.s("event", event).s("name", name).u("id", id));
    }
    static void stall(std::string_view stream, long long age_ms)
    {
        status::event("stall", status::Obj{}.s("stream", stream).i("age_ms", age_ms));
    }

    /// The fatal stop: recorded and pushed out synchronously. The caller still exits.
    static void fatal(std::string_view msg)
    {
        status::event("fatal", status::Obj{}.s("msg", msg));
        status::flush();
    }

private:
    std::string last_wait_key_;
};

}  // namespace rc
