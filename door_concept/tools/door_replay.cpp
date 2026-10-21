// door_replay — run door_concept's DoorCore over a recorded input tape (DCIN, core/door_cycle_io.h) and diff the decisions
// it makes against the decision log the live run wrote (DDEC, decision_log.h). The seed of Task 8's replay tool.
//
// ★WHAT IT REPLAYS (Task 4). The core holds the TRACKER slice (birth / associate / merge, core/door_core.h). The kinds
// that slice decides are BIRTH, SUPPRESS, MERGE and REACQUIRE; only those are diffed, field for field and bit for bit
// (cycle, t_ms, kind, door id, name, a, b, c, d, note).
// ★WHAT IT CANNOT REPLAY YET, AND WHAT STANDS IN FOR IT. The tracker reads state the later slices own (TrackerWorld in
// door_core.h). Those slices are still in SpecificWorker, so this tool rebuilds their state from the RECORDED decision
// log — the run's own record of what they decided — instead of re-deciding it:
//   · room (Task 6): IdentityLoad.initial_room, then each RoomSwitch row; the containment polygon latched from the
//     recorded rooms[] the way refresh_room_geometry latches it (replaced when ≥ 3 vertices, cleared on a switch);
//   · removals (Task 5): each REMOVE row, applied where update_existence_beliefs ran (after the association), leaving
//     the ghost remember_ghost leaves;
//   · identities (Task 6): the IdentityLoad csv parsed with the agent's parser, then every IdentityUpsert / Forget row
//     re-applied with the agent's own upsert_row / forget rule at the point of the cycle the old code made it;
//   · ★the FIT (observe / run_inference — no slice owns it yet): a door's state after birth is the pre-belief geometry
//     at its birth seed (door_model.h pre_belief_geometry, exactly as ensure_instance builds it), then MOVED to the leaf
//     centre each EXISTENCE row reports (the fitted model.state().cx/cy, logged per fresh-mask cycle), as a box at
//     yaw 0 with the aperture on that centre. No belief: such a door gates association on the metric fallback, not on
//     its Σ, never scores Pass-1 association, and holds no slice. Every mismatch this stand-in can cause is listed by
//     the diff, not hidden.
// Node ids: create_door answers with the id the recorded run got (the BIRTH row of the same cycle at the same
// position), so a reproduced birth diffs equal; an unmatched birth gets a synthetic id and shows up as a mismatch.
//
// Build + run (from door_concept/):
//   g++ -std=c++23 -O2 -I/usr/include/eigen3 tools/door_replay.cpp src/core/door_core.cpp src/core/door_cycle_io.cpp \
//       src/door_belief.cpp -lzstd -o /tmp/door_replay
//   /tmp/door_replay <door_inputs.bin> <door_decisions.bin> [--until CYCLE] [--verbose]
// Exit: 0 = the tracker kinds are equal; 1 = mismatches (listed); 2 = usage / unreadable input.
#include <charconv>
#include <clocale>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <format>
#include <map>
#include <set>
#include <print>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "../src/core/door_core.h"
#include "../src/door_config_fields.h"
#include "../src/door_identity_table.h"

using namespace rc::door::core;
using K = rc::door::DoorDecisionKind;
using rc::door::DoorDecision;

namespace
{
// ─── the recorded effective config → DoorConfig (door_config_fields.h, the agent's own key table) ──────────────────
std::vector<std::string> split_csv(std::string_view line)
{
    std::vector<std::string> out;
    std::string cur;
    bool q = false;
    for (std::size_t i = 0; i < line.size(); ++i)
    {
        const char ch = line[i];
        if (q)
        {
            if (ch == '"' and i + 1 < line.size() and line[i + 1] == '"') { cur += '"'; ++i; }
            else if (ch == '"') q = false;
            else cur += ch;
        }
        else if (ch == '"') q = true;
        else if (ch == ',') { out.push_back(cur); cur.clear(); }
        else cur += ch;
    }
    out.push_back(cur);
    return out;
}

struct EffectiveConfig
{
    std::map<std::string, std::string, std::less<>> file_values;   // key → effective, for origin == file
    int n_file = 0, n_default = 0, n_other = 0;

    explicit EffectiveConfig(const std::string& text)
    {
        std::istringstream in(text);
        for (std::string line; std::getline(in, line);)
        {
            if (line.empty() or line[0] != '"') continue;   // comments, the column header
            const auto c = split_csv(line);
            if (c.size() < 7) continue;
            if (c[2] == "file") { file_values[c[0]] = c[6]; ++n_file; }
            else if (c[2] == "default") ++n_default;
            else ++n_other;
        }
    }
    const std::string* find(std::string_view k) const
    {
        const auto it = file_values.find(k);
        return it == file_values.end() ? nullptr : &it->second;
    }
};

rc::DoorConfig config_from(const EffectiveConfig& e, int& parse_errors)
{
    rc::DoorConfig out;
    // ★Locale-free number parsing (CLAUDE.md): from_chars. ConfigLoader stores TOML floats as double and the reader
    // narrows to float, so a float key is parsed as a double and narrowed the same way.
    auto getf = [&](std::string_view k, float def, std::string_view, auto&&...) -> float
    {
        const auto* v = e.find(k);
        if (v == nullptr) return def;
        double d = 0;
        if (const auto [p, ec] = std::from_chars(v->data(), v->data() + v->size(), d); ec != std::errc{}) { ++parse_errors; return def; }
        return static_cast<float>(d);
    };
    auto geti = [&](std::string_view k, int def, std::string_view, auto&&...) -> int
    {
        const auto* v = e.find(k);
        if (v == nullptr) return def;
        int d = 0;
        if (const auto [p, ec] = std::from_chars(v->data(), v->data() + v->size(), d); ec != std::errc{}) { ++parse_errors; return def; }
        return d;
    };
    auto gets = [&](std::string_view k, std::string def, std::string_view, auto&&...) -> std::string
    {
        const auto* v = e.find(k);
        return v == nullptr ? def : *v;
    };
    auto getb = [&](std::string_view k, bool def, std::string_view, auto&&...) -> bool
    {
        const auto* v = e.find(k);
        if (v == nullptr) return def;
        if (*v == "true") return true;
        if (*v == "false") return false;
        ++parse_errors;
        return def;
    };
    rc::read_door_config_fields(out, getf, geti, gets, getb);
    return out;
}

// ─── the recorded decision log, indexed by cycle ─────────────────────────────────────────────────────────────────
bool is_tracker_kind(std::uint16_t k)
{
    return k == static_cast<std::uint16_t>(K::Birth) or k == static_cast<std::uint16_t>(K::Suppress)
        or k == static_cast<std::uint16_t>(K::Merge) or k == static_cast<std::uint16_t>(K::Reacquire);
}
std::string_view kind_name(std::uint16_t k)
{
    switch (static_cast<K>(k))
    {
        case K::Birth: return "BIRTH"; case K::Suppress: return "SUPPRESS"; case K::Merge: return "MERGE";
        case K::Remove: return "REMOVE"; case K::Reacquire: return "REACQUIRE"; case K::Existence: return "EXISTENCE";
        case K::IdentityUpsert: return "ID_UPSERT"; case K::IdentityForget: return "ID_FORGET";
        case K::RoomSwitch: return "ROOM_SWITCH"; default: return "OTHER";
    }
}
std::string show(const DoorDecision& d)
{
    return std::format("cycle {} t {} {} id {} '{}' a={:.9g} b={:.9g} c={:.9g} d={:.9g} note='{}'", d.cycle, d.t_ms,
                       kind_name(d.kind), d.door_id, d.name, d.a, d.b, d.c, d.d, d.note);
}

// ─── the world of the slices still in SpecificWorker, rebuilt from the recorded decisions (see the header) ──────────
class ReplayWorld final : public TrackerWorld
{
public:
    ReplayWorld(const rc::DoorConfig& cfg, const std::map<std::uint64_t, std::vector<DoorDecision>>& by_cycle)
        : cfg_(cfg), log_(by_cycle) {}

    // ── per cycle, before the core's step: what the old code did before run_instance_tracker ──
    void begin_cycle(const DoorCycleInput& in)
    {
        in_ = &in;
        cycle_ = in.cycle;
        rows_ = nullptr;
        if (const auto it = log_.find(cycle_); it != log_.end()) rows_ = &it->second;
        if (in.identity_load.has_value()) load(in.identity_load.value());
        for_rows([&](const DoorDecision& d)
        {
            if (d.kind == static_cast<std::uint16_t>(K::RoomSwitch) and d.door_id != room_)
            {
                room_ = d.door_id;
                polygon_.clear();   // release_room: fitter_->set_room_geometry({}, {})
            }
        });
        // refresh_room_geometry: the room's delimiting polygon, latched while it has ≥ 3 vertices.
        if (room_ != 0)
            for (const auto& r : in.rooms.rooms)
                if (r.id == room_)
                {
                    room_name_ = r.name;
                    if (r.polygon.size() >= 3) polygon_ = r.polygon;
                }
        // the FIT's stand-in: each door reported by an EXISTENCE row this cycle is where that row says (see header)
        for_rows([&](const DoorDecision& d)
        {
            if (d.kind != static_cast<std::uint16_t>(K::Existence)) return;
            if (auto it = live_.find(d.door_id); it != live_.end())
            {
                place(it->second, static_cast<float>(d.c), static_cast<float>(d.d));
                it->second.ai2_initialized = true;   // fitted (its row exists), but the tool holds no belief
                ++moved_by_existence_;
            }
        });
    }

    // ── after the core's step: what the old code did after run_instance_tracker (remember_door's identity upserts) ──
    void end_cycle()
    {
        for_rows([&](const DoorDecision& d)
        {
            if (d.kind == static_cast<std::uint16_t>(K::IdentityUpsert) and not birth_named(d.name))
                upsert(d);
        });
        in_ = nullptr;
    }

    std::uint64_t room_id() const override { return room_; }
    std::vector<Vec2f> room_polygon() const override { return polygon_; }
    std::vector<TrackedDoor> doors() const override
    {
        std::vector<TrackedDoor> v;
        for (const auto& [id, d] : live_) v.push_back(d);   // ★an unordered_map like fitter_->instances(): same order
        return v;
    }
    std::optional<GhostMatch> match_ghost(const Vec2f& xy) const override
    {
        if (cfg_.exist_reacquire_radius_m <= 0.0f) return std::nullopt;
        const Ghost* best = nullptr;
        float best_d = cfg_.exist_reacquire_radius_m;
        for (const auto& g : ghosts_)
            if (const float d = (g.xy - xy).norm(); d < best_d) { best_d = d; best = &g; }
        if (best == nullptr) return std::nullopt;
        return GhostMatch{best->name, best->xy, best->lived, static_cast<std::size_t>(best - ghosts_.data())};
    }
    std::optional<IdentityMatch> match_identity(const Vec2f& xy) const override
    {
        if (cfg_.exist_reacquire_radius_m <= 0.0f or room_name_.empty()) return std::nullopt;
        const rc::door::IdentityRow* best = nullptr;
        float best_d = cfg_.exist_reacquire_radius_m;
        for (const auto& i : identities_)
            if (i.room != room_name_) continue;
            else if (const float d = (i.xy - xy).norm(); d < best_d) { best_d = d; best = &i; }
        if (best == nullptr) return std::nullopt;
        return IdentityMatch{best->name, best->xy};
    }
    std::vector<std::string> reserved_names() const override
    {
        std::vector<std::string> n;
        for (const auto& g : ghosts_) n.push_back(g.name);
        for (const auto& i : identities_) n.push_back(i.name);
        return n;
    }
    void retire(std::uint64_t id) override { live_.erase(id); }
    void set_assigned_mask(std::uint64_t id, int slice) override
    { if (auto it = live_.find(id); it != live_.end()) it->second.assigned_mask_idx = slice; }

    // update_existence_beliefs' place in the cycle: the recorded REMOVEs (and the identity rows they forget)
    void after_association(const TrackerCycleInfo&) override
    {
        for_rows([&](const DoorDecision& d)
        {
            if (d.kind == static_cast<std::uint16_t>(K::Remove))
            {
                if (auto it = live_.find(d.door_id); it != live_.end())
                {
                    remember_ghost(it->second);
                    live_.erase(it);
                    ++removed_;
                }
            }
            else if (d.kind == static_cast<std::uint16_t>(K::IdentityForget))
                forget(d);
        });
    }

    std::uint64_t create_door(const BirthRequest& r) override
    {
        // The id the recorded run got: its BIRTH row this cycle at this very position (bit-equal floats).
        for_rows([&](const DoorDecision& d)
        {
            if (pending_id_ == 0 and d.kind == static_cast<std::uint16_t>(K::Birth) and not used_births_.contains(&d)
                and d.a == static_cast<double>(r.centroid.x()) and d.b == static_cast<double>(r.centroid.y()))
            { pending_id_ = d.door_id; used_births_.insert(&d); }
        });
        std::uint64_t id = pending_id_;
        pending_id_ = 0;
        if (id == 0) { id = 0xFFF0'0000'0000'0000ull + (++synthetic_); ++unmatched_creates_; }
        // DoorSceneGraph::next_door_name over the names the graph would hold: our live doors + foreign door_* nodes.
        std::vector<std::string> graph_names;
        for (const auto& [lid, d] : live_) graph_names.push_back(d.name);
        if (in_ != nullptr)
            for (const auto& o : in_->objects)
                if (o.name.rfind("door_", 0) == 0) graph_names.push_back(o.name);
        pending_name_ = allocate_door_name(r.preferred_name, graph_names, r.reserved);
        return id;
    }
    void reacquire(std::uint64_t, const GhostMatch&) override {}   // the ghost's BELIEF is not rebuilt (no fit here)
    void forget_ghost(const Vec2f& xy) override
    { std::erase_if(ghosts_, [&](const Ghost& g) { return (g.xy - xy).norm() < cfg_.exist_reacquire_radius_m; }); }
    std::optional<std::string> adopt_birth(std::uint64_t id, int slice, const Vec3f& c) override
    {
        // ensure_instance: the node's attributes as create_instance_from_detection wrote them, the birth seed for the
        // centre, yaw 0 (the RT edge hangs from a wall, or from the room with zero rotation) → refresh_geometry.
        TrackedDoor d;
        d.id = id;
        d.name = pending_name_;
        rc::DoorState s;
        s.w = cfg_.door_prior_w_m; s.thickness = cfg_.door_thickness_m; s.h = cfg_.door_prior_h_m;
        live_[id] = d;
        place(live_[id], c.x(), c.y(), s);
        live_[id].assigned_mask_idx = slice;
        // note_identity(name, birth xy): the IdentityUpsert row the old code wrote right here, if it wrote one
        for_rows([&](const DoorDecision& r)
        {
            if (r.kind == static_cast<std::uint16_t>(K::IdentityUpsert) and r.name == d.name) upsert(r);
        });
        births_named_.push_back(d.name);
        return d.name;
    }

    int moved_by_existence_ = 0, removed_ = 0, unmatched_creates_ = 0;
    [[nodiscard]] std::size_t n_identities() const { return identities_.size(); }

private:
    struct Ghost { std::string name; Vec2f xy; int lived = 0; };

    template <class F> void for_rows(F&& f) { if (rows_ != nullptr) for (const auto& d : *rows_) f(d); }
    bool birth_named(const std::string& name)
    {
        bool b = false;
        for_rows([&](const DoorDecision& d) { b |= d.kind == static_cast<std::uint16_t>(K::Birth) and d.name == name; });
        return b;
    }

    // The stand-in geometry: a box of the door's size at leaf centre (x, y), yaw 0 — pre_belief_geometry's construction.
    void place(TrackedDoor& d, float x, float y, std::optional<rc::DoorState> base = std::nullopt)
    {
        rc::DoorState s = base.value_or(d.state);
        s.cx = x; s.cy = y; s.yaw = 0.0f;
        const auto g = rc::pre_belief_geometry(s);
        rc::read_back_leaf(s, g.leaf_pose, g.leaf);
        d.state = s; d.aperture = g.aperture; d.leaf_pose = g.leaf_pose;
    }

    void load(const IdentityLoad& l)
    {
        if (l.initial_room.has_value()) { room_ = *l.initial_room; room_name_ = l.initial_room_name; }
        identities_.clear();
        if (not l.file_present) return;
        std::istringstream in(l.csv_text);
        auto parsed = rc::door::parse_identity_table(in);
        if (not parsed.unkeyed_format) identities_ = std::move(parsed.rows);
    }
    void upsert(const DoorDecision& d)
    {
        rc::door::IdentityRow row;
        row.name = d.name;
        row.xy = Vec2f(static_cast<float>(d.a), static_cast<float>(d.b));
        const float width_m = static_cast<float>(d.c);
        const float same_hole = cfg_.exist_reacquire_radius_m + std::max(0.0f, std::isfinite(width_m) ? width_m : 0.0f);
        (void) rc::door::upsert_row(identities_, d.note, std::move(row), same_hole, static_cast<float>(d.d));
    }
    void forget(const DoorDecision& d)
    {
        const Vec2f ap(static_cast<float>(d.a), static_cast<float>(d.b));
        const float width_m = static_cast<float>(d.c);
        const float same_hole = cfg_.exist_reacquire_radius_m + std::max(0.0f, std::isfinite(width_m) ? width_m : 0.0f);
        std::erase_if(identities_, [&](const rc::door::IdentityRow& i)
                      { return i.room == d.note and (i.name == d.name or (ap.allFinite() and (i.xy - ap).norm() < same_hole)); });
    }
    void remember_ghost(const TrackedDoor& d)
    {
        // SpecificWorker::remember_ghost — keyed on the aperture (here the stand-in's: the leaf centre, see header)
        if (cfg_.exist_reacquire_radius_m <= 0.0f or not d.ai2_initialized or d.is_bearing_hypothesis) return;
        const Vec2f ap(d.state.ap_cx, d.state.ap_cy);
        std::erase_if(ghosts_, [&](const Ghost& g) { return (g.xy - ap).norm() < cfg_.exist_reacquire_radius_m; });
        ghosts_.push_back({d.name, ap, 0});
        while (static_cast<int>(ghosts_.size()) > std::max(1, cfg_.exist_ghost_max)) ghosts_.erase(ghosts_.begin());
    }

    const rc::DoorConfig& cfg_;
    const std::map<std::uint64_t, std::vector<DoorDecision>>& log_;
    const std::vector<DoorDecision>* rows_ = nullptr;
    const DoorCycleInput* in_ = nullptr;
    std::uint64_t cycle_ = 0, room_ = 0;
    std::string room_name_;
    std::vector<Vec2f> polygon_;
    std::unordered_map<std::uint64_t, TrackedDoor> live_;
    std::vector<Ghost> ghosts_;
    std::vector<rc::door::IdentityRow> identities_;
    std::set<const DoorDecision*> used_births_;
    std::vector<std::string> births_named_;
    std::uint64_t pending_id_ = 0, synthetic_ = 0;
    std::string pending_name_;
};
}  // namespace

int main(int argc, char** argv)
{
    // ★The agents' locale (CLAUDE.md): parse and format exactly as under es_ES, where the live run made these decisions.
    if (std::setlocale(LC_ALL, "es_ES.UTF-8") == nullptr) std::setlocale(LC_ALL, "");
    if (argc < 3)
    {
        std::println("usage: door_replay <door_inputs.bin> <door_decisions.bin> [--until CYCLE] [--verbose]");
        return 2;
    }
    std::uint64_t until = ~0ull;
    bool verbose = false;
    for (int i = 3; i < argc; ++i)
    {
        const std::string_view a = argv[i];
        if (a == "--until" and i + 1 < argc) until = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--verbose") verbose = true;
    }

    // the recorded decisions (the reference) — every kind, indexed by cycle; the tracker kinds kept apart
    const auto ref_all = rc::door::DoorDecisionLog::read_all(argv[2]);
    if (not ref_all.has_value()) { std::println("cannot read the decision log {}", argv[2]); return 2; }
    std::map<std::uint64_t, std::vector<DoorDecision>> by_cycle;
    std::vector<DoorDecision> ref;
    for (const auto& d : *ref_all)
    {
        by_cycle[d.cycle].push_back(d);
        if (is_tracker_kind(d.kind) and d.cycle <= until) ref.push_back(d);
    }

    std::ifstream f(argv[1], std::ios::binary);
    std::uint32_t version = 0;
    const auto hdr = f ? read_header(f, RecordingKind::Input, &version) : std::nullopt;
    if (not hdr.has_value()) { std::println("cannot read the input tape {}", argv[1]); return 2; }
    const EffectiveConfig eff(hdr->config_text);
    int cfg_errors = 0;
    DoorCoreConfig cfg{config_from(eff, cfg_errors)};
    std::println("tape: {} (DCIN v{}, agent {} '{}'); config: {} keys from the file, {} at code default, {} parse errors",
                 argv[1], version, hdr->agent_id, hdr->agent_name, eff.n_file, eff.n_default, cfg_errors);

    ReplayWorld world(cfg.door, by_cycle);
    DoorCore core(cfg);
    core.connect(&world);

    std::vector<DoorDecision> got;
    std::uint64_t n = 0, first = 0, last = 0;
    int stepped = 0, media_only = 0;
    while (true)
    {
        auto in = read_input(f, version);
        if (not in.has_value()) break;
        if (in->cycle > until) break;
        if (n++ == 0) first = in->cycle;
        last = in->cycle;
        world.begin_cycle(*in);
        if (world.room_id() == 0)
        {
            core.observe_media(*in);   // the old compute() returned before the tracker
            ++media_only;
        }
        else
        {
            const auto out = core.step(*in);
            ++stepped;
            for (const auto& d : out.decisions)
                if (is_tracker_kind(d.kind)) got.push_back(d);
        }
        world.end_cycle();
    }
    std::println("replayed {} records (cycles {}..{}): tracker stepped {}, media-only {}", n, first, last, stepped, media_only);
    std::println("stand-ins from the decision log: {} door moves from EXISTENCE rows, {} REMOVEs, {} births without a recorded id",
                 world.moved_by_existence_, world.removed_, world.unmatched_creates_);

    // the diff: the tracker kinds, in order, every field
    std::println("\nreference (live run) tracker decisions: {}   replay: {}", ref.size(), got.size());
    if (verbose)
    {
        for (const auto& d : ref) std::println("  ref    {}", show(d));
        for (const auto& d : got) std::println("  replay {}", show(d));
    }
    int mismatches = 0;
    const std::size_t m = std::max(ref.size(), got.size());
    for (std::size_t i = 0; i < m; ++i)
    {
        const DoorDecision* a = i < ref.size() ? &ref[i] : nullptr;
        const DoorDecision* b = i < got.size() ? &got[i] : nullptr;
        if (a != nullptr and b != nullptr and *a == *b)
        {
            std::println("  EQUAL  {}", show(*a));
            continue;
        }
        ++mismatches;
        std::println("  DIFF #{}:", i);
        std::println("     ref    {}", a ? show(*a) : std::string{"(none)"});
        std::println("     replay {}", b ? show(*b) : std::string{"(none)"});
        if (a and b)
        {
            std::string fields;
            if (a->cycle != b->cycle) fields += " cycle";
            if (a->t_ms != b->t_ms) fields += " t_ms";
            if (a->kind != b->kind) fields += " kind";
            if (a->door_id != b->door_id) fields += " door_id";
            if (a->name != b->name) fields += " name";
            if (a->a != b->a) fields += std::format(" a(Δ={:.3g})", b->a - a->a);
            if (a->b != b->b) fields += std::format(" b(Δ={:.3g})", b->b - a->b);
            if (a->c != b->c) fields += " c";
            if (a->d != b->d) fields += " d";
            if (a->note != b->note) fields += " note";
            std::println("     fields:{}", fields);
        }
    }
    std::println("\n{}: {} tracker decision(s) compared, {} mismatch(es)", mismatches == 0 ? "EQUAL" : "DIFFERENT", m, mismatches);
    return mismatches == 0 ? 0 : 1;
}
