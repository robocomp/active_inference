// Repertory: pure model of the long-term object memory (no DSR, no Qt).
// Entries are keyed by object_uid; each has lives (born..end), pose keyframes and the last sizes seen.
#pragma once
#include <cstdint>
#include <map>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace ltom
{
struct Pose2 { double x = 0, y = 0, yaw = 0; double vx = 0, vy = 0, vyaw = 0; };   // room frame, m/rad; variances

struct Sighting
{
    std::uint64_t uid = 0;
    std::string cls, name;
    std::int64_t room_key = -1;
    Pose2 pose;
    float w = 0, d = 0, h = 0;
    std::int64_t born_ms = 0;
};

enum class End { alive, death, owner_exit, gap };

inline const char* end_name(End e)
{
    switch (e)
    {
        case End::alive: return "alive";
        case End::death: return "death";
        case End::owner_exit: return "owner_exit";
        case End::gap: return "gap";
    }
    return "alive";
}

struct Life { std::int64_t born_ms = 0, first_seen_ms = 0, last_seen_ms = 0, end_ms = -1; End end = End::alive; };
struct Keyframe { std::int64_t t_ms = 0, room_key = -1; Pose2 pose; };

struct Entry
{
    std::uint64_t uid = 0;
    std::string cls, name;
    float w = 0, d = 0, h = 0;
    std::int64_t n_sightings = 0;
    std::vector<Life> lives;
    std::vector<Keyframe> keyframes;
};

struct Changes { std::vector<std::uint64_t> born, reopened, died, owner_exited; };

inline std::uint32_t owner_of(std::uint64_t uid) { return static_cast<std::uint32_t>(uid >> 40); }

// chi2_3(0.99). A record is a binary act, so some decision rule is unavoidable; this one scales with the
// estimate's own uncertainty (sum of both covariances) rather than a metric cutoff. Flagged in the plan.
constexpr double kKeyframeChi2 = 11.34;

class Repertory
{
public:
    explicit Repertory(std::int64_t owner_grace_ms) : grace_ms_(owner_grace_ms) {}

    // `present` = every in-scope uid on the graph, resolvable or not. A uid present but absent from `alive` is HELD
    // (no information is not absence): no update, no end, no pending.
    Changes observe(std::span<const Sighting> alive, const std::set<std::uint32_t>& owners_present,
                    const std::set<std::uint64_t>& present, std::int64_t now_ms);
    void close_open_lives_as_gap();     // at load: every alive life -> End::gap, end_ms = last_seen_ms
    const std::map<std::uint64_t, Entry>& entries() const { return entries_; }
    void insert_loaded(Entry e);        // the loader's door; marks nothing dirty
    bool dirty() const { return dirty_; }
    void clear_dirty() { dirty_ = false; }

private:
    struct Pending { std::int64_t vanish_ms; };
    std::int64_t grace_ms_;
    std::map<std::uint64_t, Entry> entries_;
    std::map<std::uint64_t, Pending> pending_;
    bool dirty_ = false;
};
}
