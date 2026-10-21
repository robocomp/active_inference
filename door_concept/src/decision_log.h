// decision_log.h — the binary per-cycle DECISION log of door_concept ("DDEC" v1).
//
// ★WHY IT EXISTS. door_concept's per-cycle logic is being extracted into a DSR-free DoorCore with record/replay
// (docs/superpowers/specs/2026-10-21-door-core-replay-design.md). "The refactor changed nothing" is only checkable
// against what TODAY's code decided, so today's SpecificWorker writes every decision here, at the point it makes it,
// and the replay's output must reproduce this stream bit for bit (spec §6.3). It is a regression instrument, not a
// diagnostic: the CSV logs round numbers through a stream and cannot be diffed exactly, this one stores the doubles
// raw. DEFAULT OFF (DoorConcept.DecisionLog); a file per run, the previous ones kept by rc::diag::open_rotating.
//
// Layout: 8-byte header "DDEC" + u32 version (1), then records, little-endian native (x86; the reader and writer
// are the same build family):
//   u64 cycle | i64 t_ms | u16 kind | u64 door_id | f64 a,b,c,d | u32 len + name | u32 len + note
// A truncated final record (agent killed mid-write) is dropped by read_all, never misread.
//
// Field meaning per kind (a,b,c,d) is documented at DoorDecisionKind. No locale is involved (binary), so the
// es_ES decimal-comma trap of CLAUDE.md cannot apply; the test runs under es_ES anyway.
#pragma once
#include <bit>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <vector>
#include "../../common/diag_log/rotating_csv.h"

static_assert(std::endian::native == std::endian::little, "DDEC v1 stores little-endian native; add byte swaps first");

// ★READING THE STREAMS. They are NOT cycle-aligned: Existence is logged only on cycles that carry a fresh mask frame
// (when the old code decides), Phi on every cycle, the rest when they happen. t_ms is steady_clock milliseconds since its
// epoch (boot), so DoorCycleInput must carry the cycle's steady time for the replay to reproduce it.

namespace rc::door
{

enum class DoorDecisionKind : std::uint16_t
{
    // tracker (run_instance_tracker / merge / release / restore) — a=x, b=y (map/room frame, metres), note = reason
    Birth = 1, Suppress, Merge, Remove, Reacquire, Release, Restore,
    // per door, per fresh-mask cycle: a=L (log-odds), b=p_exists, c=x, d=y
    Existence,
    // per door, per cycle: a=phi_est (rad), b=phi_sigma (rad), c=phi_measured (0/1), d=phi_support
    Phi,
    // identity table: a=x, b=y, c=width_m, d=weight; name = the door name; note = room
    IdentityUpsert, IdentityForget,
    // room switch: id = new room node id, a=old room id, b=1 if the old room still lives; note = room name
    RoomSwitch, CarryPending, CarryInstall,
    // epistemic "look at a remembered door": a=x, b=y, c=EIG (nats); LookDrop note = why
    LookOffer, LookDrop,
    // pragmatic affordances; name = door name, note = kind (approach|open|cross|close),
    // a = epoch, b = P(precondition) or outcome code, c = value
    AffOffered, AffWithdrawn, AffExecuting, AffCompleted,
    // passage anchor: a=x, b=y, c=yaw (robot frame), d=epoch; note = door / reason
    AnchorHung, AnchorRemoved,
    // (appended; values of the kinds above are stable)
    // actuator request, incl. refused locally: a=1 open/0 close, b=1 sent / 0 refused, c=requests so far; note=purpose|why-or-state
    ActuatorRequest,
    // identity table LOAD (spec 6.6): a=rows, b=malformed, c=1 if the file was the old un-keyed format (ignored)
    // and one row per loaded face: name, a=x, b=y, c=1 if restorable (has a pose); note=room
    IdentityLoad,
    // a remembered row lost its restorable data on a FRESH entry: name, note=room
    IdentityStrip,
    // plan_restore skipped a remembered row: name; a = SkipReason (0 NoPose, 1 NameLive, 2 PlaceLive); note=room
    RestoreSkip,
    // a restored door got another name than the row's (DSR names unique): id=node, name=row name, note=assigned name
    RestoreRename,
    // passage-anchor state change other than hang/remove (park, re-init, relink, mode, leaf): a,b,c = pose x,y,yaw, d=epoch; note=what
    AnchorState,
    kCount_   // sentinel: keep LAST; kCount = number of kinds
};
inline constexpr std::uint16_t kCount = static_cast<std::uint16_t>(DoorDecisionKind::kCount_) - 1;

struct DoorDecision
{
    std::uint64_t cycle = 0;
    std::int64_t  t_ms  = 0;
    std::uint16_t kind  = 0;
    std::uint64_t door_id = 0;
    std::string   name;
    double a = 0, b = 0, c = 0, d = 0;
    std::string   note;
    bool operator==(const DoorDecision&) const = default;   // DoorCycleOutput carries these (core/door_cycle_io.h)
};

class DoorDecisionLog
{
public:
    static constexpr std::uint32_t kVersion = 1;

    [[nodiscard]] bool open(const std::string& path)
    {
        close();
        if (not rc::diag::open_rotating(out_, path))
            return false;
        out_.write("DDEC", 4);
        const std::uint32_t v = kVersion;
        out_.write(reinterpret_cast<const char*>(&v), 4);
        out_.flush();
        return out_.good();
    }
    void close() { if (out_.is_open()) out_.close(); }
    [[nodiscard]] bool enabled() const { return out_.is_open(); }

    /// The cycle/time stamped onto every decision emitted through record(); set once per compute().
    void set_clock(std::uint64_t cycle, std::int64_t t_ms) { cycle_ = cycle; t_ms_ = t_ms; }
    std::uint64_t cycle() const { return cycle_; }

    void record(DoorDecisionKind k, std::uint64_t id, std::string name, double a = 0, double b = 0, double c = 0,
              double d = 0, std::string note = {})
    {
        if (not enabled()) return;
        append({cycle_, t_ms_, static_cast<std::uint16_t>(k), id, std::move(name), a, b, c, d, std::move(note)});
    }

    void append(const DoorDecision& x)
    {
        if (not enabled()) return;
        put(x.cycle); put(x.t_ms); put(x.kind); put(x.door_id);
        put(x.a); put(x.b); put(x.c); put(x.d);
        put_str(x.name); put_str(x.note);
        out_.flush();   // a SIGKILLed run still leaves every complete record on disk
    }

    [[nodiscard]] static std::optional<std::vector<DoorDecision>> read_all(const std::string& path)
    {
        std::ifstream f(path, std::ios::binary);
        if (not f.is_open()) return std::nullopt;
        char magic[4]{}; std::uint32_t ver = 0;
        f.read(magic, 4); f.read(reinterpret_cast<char*>(&ver), 4);
        if (not f or std::memcmp(magic, "DDEC", 4) != 0 or ver != kVersion) return std::nullopt;
        std::vector<DoorDecision> v;
        for (;;)
        {
            DoorDecision x;
            if (not (get(f, x.cycle) and get(f, x.t_ms) and get(f, x.kind) and get(f, x.door_id)
                     and get(f, x.a) and get(f, x.b) and get(f, x.c) and get(f, x.d)
                     and get_str(f, x.name) and get_str(f, x.note)))
                break;   // clean EOF or a truncated tail: stop, keep what is whole
            v.push_back(std::move(x));
        }
        return v;
    }

private:
    template <class T> void put(const T& x) { out_.write(reinterpret_cast<const char*>(&x), sizeof x); }
    void put_str(const std::string& s)
    {
        const std::uint32_t n = static_cast<std::uint32_t>(s.size());
        put(n);
        out_.write(s.data(), static_cast<std::streamsize>(n));
    }
    template <class T> static bool get(std::ifstream& f, T& x)
    { return static_cast<bool>(f.read(reinterpret_cast<char*>(&x), sizeof x)); }
    static bool get_str(std::ifstream& f, std::string& s)
    {
        std::uint32_t n = 0;
        if (not get(f, n) or n > (1u << 24)) return false;
        s.resize(n);
        return n == 0 or static_cast<bool>(f.read(s.data(), n));
    }

    std::ofstream out_;
    std::uint64_t cycle_ = 0;
    std::int64_t  t_ms_ = 0;
};

}  // namespace rc::door
