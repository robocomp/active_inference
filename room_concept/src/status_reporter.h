// ── rc::StatusReporter — room_concept's side of the status stream ───────────────────────────────────
// common/status_stream/ is GENERIC: it carries events and knows nothing about rooms. This class is
// where room_concept decides WHAT to say:
//   · the structured events its startup and state machine emit (phase, sm, wait, peer, stall, loaded);
//   · the 2 Hz `state` snapshot (localisation, room, motion + camera calibration, streams, compute,
//     presence), built on the MAIN thread from scalars compute() already has in hand.
//
// Every method is a no-op when the stream is off ([Status] Enable = false): events go through
// rc::status::event(), which does nothing with no sink installed, and build_state() is never called.
// So a call site never needs a guard.
//
// THREADING: observe(), observe_compute(), note_camcal() and build_state() are MAIN THREAD only (they
// are called from compute(), the camera pumps and a QTimer). The static event helpers are any-thread.
//
// Viewer: room_concept/tools/room_tui.py (launcher: room_concept/tools/room_run.sh).
#pragma once

#include "../../common/status_stream/status_sink.h"

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "room_concept.h"

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
    void sm(std::string_view state, std::string_view detail = {})
    {
        sm_state_ = state;
        status::event("sm", status::Obj{}.s("state", state).s("detail", detail));
    }
    /// The "why are we still Waiting" line, as data. Emitted only when it CHANGES, so the replay ring
    /// holds the history of reasons, not one copy per throttle period.
    void waiting(bool peers_ok, const std::vector<std::string>& missing, bool lidar_ok, std::string_view why);
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

    // ── the 2 Hz snapshot ─────────────────────────────────────────────────────────────────────────────
    /// One stream's monotonic counter and age, sampled by build_state(). count < 0 ⇒ no counter.
    struct StreamSample
    {
        std::string name;
        long long count = -1;
        long long age_ms = -1;   ///< ms since the last frame, -1 = never / unknown
    };
    /// What the snapshot reads that compute() does not hand over. Every getter may be empty or return
    /// nothing useful early in startup; build_state() copes.
    struct Sources
    {
        std::function<void(status::Obj&)>            room;      ///< fills the room-node fields
        std::function<std::vector<StreamSample>()>   streams;
        std::function<std::vector<std::string>()>    missing;   ///< required peers not present
        std::function<bool()>                        overlay_verbose;
    };
    void set_sources(Sources s) { src_ = std::move(s); }

    /// compute(): the result it ALREADY fetched. Only scalars are kept — never the scan.
    void observe(const RoomConcept::UpdateResult& r);
    /// EVERY corrected localiser result (PosePublisher's hand-over, LOCALISER thread): accumulates the
    /// surprise so the snapshot reports a true rate in nats/s, not a 2 Hz sample of a 20 Hz quantity.
    void accumulate_surprise(const RoomConcept::UpdateResult& r);
    /// compute(): this tick's cost.
    void observe_compute(long long us);
    /// The camera mount solve, in physical units, from the same place the Calib window gets it.
    void note_camcal(const std::string& cam, const std::array<float, 4>& value, const std::array<float, 4>& sigma,
                     int informed, float cond, long pairs);
    /// Build and post one `state` event. Returns the JSON (the dump command writes it).
    std::string build_state();
    [[nodiscard]] const std::string& last_state() const { return last_state_; }

private:
    std::string last_wait_key_;
    Sources src_;

    // localisation, from the last observe()
    struct Loc
    {
        bool have = false, ok = false, diverged = false, polished = false;
        float x = 0, y = 0, th = 0, sx = 0, sy = 0, sth = 0, cond = 0;
        int iters = 0;
        float ee = 0, sdf = 0, pred_sdf = 0, innov = 0, kl = 0, mismatch = 0;
        bool scored = false;
        std::uint64_t epoch = 0;
        std::int64_t ts_ms = 0;
        // room-from-result
        int walls = 0, verts = 0;
        bool closed = false, map_ready_view = false;
        // motion calibration
        std::array<float, calib::P_COUNT> cv{}, cs{};
        int informed = 0, applied = 0, episodes = 0, carried = 0, dropped = 0;
        float ccond = 0;
    } loc_;

    // compute timing, since the last snapshot
    long long comp_n_ = 0, comp_sum_us_ = 0, comp_max_us_ = 0;
    long long comp_t0_ms_ = 0;

    struct Cam
    {
        std::array<float, 4> v{}, s{};
        int informed = 0;
        float cond = 0;
        long pairs = 0;
        long long t_ms = 0;
    };
    std::map<std::string, Cam> cams_;

    struct Rate { long long count = -1; long long t_ms = 0; double hz = 0; };
    std::map<std::string, Rate> rates_;

    // surprise accumulated since the last snapshot (written on the localiser thread)
    std::mutex sur_mtx_;
    double kl_acc_ = 0.0, mm_acc_ = 0.0;
    long sur_scored_ = 0, sur_results_ = 0;
    long long sur_t0_ms_ = 0;

    std::string sm_state_ = "?";
    std::string last_state_;
};

}  // namespace rc
