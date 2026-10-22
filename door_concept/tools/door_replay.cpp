// door_replay — run door_concept's DoorCore over a recorded input tape (DCIN, core/door_cycle_io.h) and diff the decisions
// it makes against the decision log the live run wrote (DDEC, decision_log.h). The seed of Task 8's replay tool.
//
// ★WHAT IT REPLAYS (Tasks 4 + 4b + 5). The core holds the TRACKER slice (birth / associate / merge), the FIT (observe /
// run_inference / the leaf angle — core/door_fit.h) and EXISTENCE (silhouette / image geometry / P(detect) / removal —
// core/door_existence.h): every door's belief, its existence and its removal are computed from the tape alone. Diffed:
//   · the tracker kinds BIRTH, SUPPRESS, MERGE, REACQUIRE — field for field and bit for bit (cycle, t_ms, kind, door id,
//     name, a, b, c, d, note);
//   · every PHI row (a = phi_est, b = phi_sigma, c = measured, d = phi_support) — bit for bit, and where a row differs
//     the largest absolute and relative difference per field is reported;
//   · every EXISTENCE row (a = L, b = p_exists, c,d = the fitted centre at the existence update) — bit for bit, with the
//     per-field max difference and the first divergence; and the REMOVE decisions (cycle, door, position, note);
//   · --csv DIR also writes the core's etc/door_existence_log.csv, to diff row by row against the live one.
// ★WHAT IT CANNOT REPLAY YET, AND WHAT STANDS IN FOR IT. The core reads state the later slices own (TrackerWorld in
// door_core.h). Those slices are still in SpecificWorker, so this tool rebuilds their state from the RECORDED decision
// log — the run's own record of what they decided — instead of re-deciding it:
//   · room (Task 6): IdentityLoad.initial_room, then each RoomSwitch row; the containment polygon latched from the
//     recorded rooms[] the way refresh_room_geometry latches it (replaced when ≥ 3 vertices, cleared on a switch);
//   · (removals are the CORE's since Task 5: the world only remembers the ghost and forgets the stored identity the way
//     remember_ghost / forget_identity do — before_remove — and drops the door — remove_door);
//   · identities (Task 6): the IdentityLoad csv parsed with the agent's parser, then every IdentityUpsert / Forget row
//     re-applied with the agent's own upsert_row / forget rule at the point of the cycle the old code made it;
//   · the affordance / anchor / actuation slices (Task 7): NOT rebuilt — a door's hinge_question_live, anchor φ curve
//     and commanded swing stay at rest, which is what they were on a run with no claim, no anchor and no actuator call
//     (the tool counts the recorded AffExecuting / AnchorHung / ActuatorRequest rows and says so).
// ★THE FIT'S CLOCK. The live agent (before Task 4b) read steady_clock mid-cycle for the leaf channel's dt; the core
// reads the cycle's t_steady_ns. A tape recorded by a pre-4b binary therefore cannot reproduce its Phi rows to the bit
// wherever dt enters — the report quantifies that, per field.
// Node ids: create_door answers with the id the recorded run got (the BIRTH row of the same cycle at the same
// position), so a reproduced birth diffs equal; an unmatched birth gets a synthetic id and shows up as a mismatch.
//
// Build + run (from door_concept/):
//   g++ -std=c++23 -O2 -I/usr/include/eigen3 -I/usr/include/opencv4 tools/door_replay.cpp src/core/door_core.cpp \
//       src/core/door_fit.cpp src/core/door_existence.cpp src/core/door_cycle_io.cpp src/door_belief.cpp src/door_model.cpp src/epistemic_planner.cpp \
//       src/door_frame_evidence.cpp ../common/contour_edge/contour_edge_check.cpp \
//       ../common/contour_edge/contour_edge_project.cpp ../common/contour_edge/contour_depth_check.cpp \
//       -lzstd -lopencv_core -lopencv_imgproc -o /tmp/door_replay
//   /tmp/door_replay <door_inputs.bin> <door_decisions.bin> [--until CYCLE] [--verbose] [--csv DIR] [--clock-jitter-ms J]
//       [--look] [--ablate-hull-midpoint]   (Task E: per-cycle look-vs-approach prices on one scale; the Task-W term OFF)
//       [--trace NAME]   (one line per cycle for that door — '*' for every door: its fit, its detection state and every
//                         existence channel)
//       [--recess-prior-std S]   (a sensitivity probe: the leaf-plane depth's prior σ, overriding the recorded config)
// Exit: 0 = the tracker kinds, the Phi rows, the Existence rows and the REMOVEs are equal; 1 = mismatches (listed);
// 2 = usage / unreadable input.
#include <charconv>
#include <chrono>
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
#include "../src/door_dof.h"            // the live consumers' DOF table (dashboard, NBV adequacy)
#include "../src/door_approach_pose.h"  // Task E: the approach offer's price (--look)
#include "../src/epistemic_planner.h"   // the live NBV (rc::EpistemicPlanner) — run here as a smoke on every door

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
                if (mirror_) make_entry_mirror();
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
    }

    // ★--mirror: the live agent, on a room switch, releases the old room's doors and creates an ENTRY MIRROR of the crossed
    // door in the new room (create_entry_mirror: the same door, pose (0, 0.02) yaw π in the new room frame, the source's
    // size and leaf side) whose fit is lazy-initialised from the first mask frame that sees it. The recorded decisions do
    // not carry it, so by default the replay keeps the old instance — which never sees a mask again in the new room (a tape
    // with a collapse AFTER the crossing cannot reproduce it). The crossed door = the live door nearest the robot.
    bool mirror_ = false;
    void make_entry_mirror()
    {
        if (in_ == nullptr or live_.empty() or not in_->robot.room_T_body.has_value()) return;
        const Eigen::Vector2f rb((*in_->robot.room_T_body)(0, 3), (*in_->robot.room_T_body)(1, 3));
        std::uint64_t src = 0; float best = 1e9f;
        for (const auto& [id, d] : live_)
        {
            const auto& m = d.model.state();
            if (const float dd = (Eigen::Vector2f(m.cx, m.cy) - rb).norm(); dd < best) { best = dd; src = id; }
        }
        if (src == 0) return;
        const rc::DoorFitState old = live_.at(src);
        const auto m = old.model.state();
        live_.clear();
        rc::DoorState s;
        s.cx = 0.0f; s.cy = 0.02f; s.yaw = -3.14159265f;
        s.w = m.w; s.thickness = m.thickness; s.h = m.h;
        rc::DoorFitState f;
        const std::uint64_t id = 0xFFE0'0000'0000'0000ull + (++synthetic_);
        fit_->init_instance(f, id, old.node_name, s);
        // install_carried_door: the leaf memory crosses with the door, re-expressed in the new room frame
        // (T_new_old = T_door_new · T_door_old⁻¹, both 2-D rigid maps with the door pose as origin/orientation).
        {
            const auto rot = [](float a) { Eigen::Matrix2f r; r << std::cos(a), -std::sin(a), std::sin(a), std::cos(a); return r; };
            const Eigen::Matrix2f Rn = rot(s.yaw), Ro = rot(m.yaw);
            const Eigen::Matrix2f R = Rn * Ro.transpose();
            const Eigen::Vector2f o = Eigen::Vector2f(s.cx, s.cy) - R * Eigen::Vector2f(m.cx, m.cy);
            if (old.leaf_side_hint.has_value())
                f.leaf_side_hint = std::make_pair(Eigen::Vector2f(R * old.leaf_side_hint->first + o),
                                                  Eigen::Vector2f(R * old.leaf_side_hint->second));
        }
        f.phi_est = old.phi_est; f.phi_sigma = old.phi_sigma; f.hinge_llr = old.hinge_llr; f.swing_llr = old.swing_llr;
        f.phi_curve = old.phi_curve; f.matched_frames = old.matched_frames; f.phi_from_command = false;
        f.phi_last_t = std::chrono::steady_clock::time_point(std::chrono::nanoseconds(in_->t_steady_ns));
        if (const char* e = std::getenv("MIRROR_PHI")) {
            {   // from_chars, not strtof: main() activates the machine's es_ES locale, where strtof("1.553") stops at the '.' and reads 1
                const std::string_view sv = e;
                float v = 0.0f;
                if (std::from_chars(sv.data(), sv.data() + sv.size(), v).ec == std::errc{}) f.phi_est = v;
            }
            f.phi_sigma = 0.035f;
            f.phi_curve.clear();
            for (int i = 0; i <= 24; ++i)   // the crossing's curve: a 25-point grid with the weight on the opened angle
            { const float ph = 0.0f + i * (2.0944f / 24.0f); f.phi_curve.emplace_back(ph, std::exp(-0.5f * std::pow((ph - f.phi_est) / 0.035f, 2.0f))); }
        }

        live_.insert_or_assign(id, std::move(f));
        std::println("MIRROR cycle {} '{}' source w {:.3f} -> mirror at (0,0.02) in the new room; robot {:.2f} {:.2f}", cycle_, old.node_name, m.w, rb.x(), rb.y());
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
    std::vector<std::pair<std::uint64_t, rc::DoorFitState*>> fit_states() const override
    {
        std::vector<std::pair<std::uint64_t, rc::DoorFitState*>> v;
        for (auto& [id, d] : live_) v.emplace_back(id, &d);   // ★an unordered_map like fitter_->instances(): same order
        return v;
    }
    // the fit: every live door, each cycle (live: the graph's door nodes — the same set on a run with no mirror, no
    // anchor and no held door; the ORDER differs, which no fit result depends on: each door's fit reads only its own state)
    std::vector<std::uint64_t> fit_targets() override
    {
        std::vector<std::uint64_t> v;
        for (const auto& [id, d] : live_) v.push_back(id);
        return v;
    }
    rc::DoorFitState* begin_fit(std::uint64_t id) override
    { const auto it = live_.find(id); return it == live_.end() ? nullptr : &it->second; }
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

    void after_association(const TrackerCycleInfo&) override {}   // (live: the pumps; existence is the core's now)

    // ── the existence slice's removal, the world's half (SpecificWorker::CoreWorld::before_remove / remove_door) ──
    void before_remove(std::uint64_t id) override
    {
        if (const auto it = live_.find(id); it != live_.end())
            remember_ghost(it->second);   // ...which also forgets the stored identity (forget_identity)
    }
    void remove_door(std::uint64_t id, float) override
    {
        if (live_.erase(id) > 0) ++removed_;
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
        for (const auto& [lid, d] : live_) graph_names.push_back(d.node_name);
        if (in_ != nullptr)
            for (const auto& o : in_->objects)
                if (o.name.rfind("door_", 0) == 0) graph_names.push_back(o.name);
        pending_name_ = allocate_door_name(r.preferred_name, graph_names, r.reserved);
        // ★THE INSTANCE IS BORN INSIDE THE INSERT, NOT IN adopt_birth. Live, create_instance_from_detection's
        // insert_node emits update_node_signal on the MAIN thread, so modify_node_slot runs synchronously inside it
        // (AutoConnection, same thread ⇒ direct) and ensure_instance reads the node BEFORE its RT edge exists and
        // BEFORE the birth seed is noted: the newborn's pre-belief box sits at (0, 0), yaw 0, the prior size. The
        // replay must build it the same way — the first estimate_phi selects its rays around that box (tourR1:
        // 14686 rays on the birth cycle, against 114 around the seed). See task-4b-report.md (a pre-existing defect).
        rc::DoorState s;
        s.cx = 0.0f; s.cy = 0.0f; s.yaw = 0.0f;
        s.w = cfg_.door_prior_w_m; s.thickness = cfg_.door_thickness_m; s.h = cfg_.door_prior_h_m;
        rc::DoorFitState f;
        fit_->init_instance(f, id, pending_name_, s);
        live_.insert_or_assign(id, std::move(f));
        return id;
    }
    void reacquire(std::uint64_t, const GhostMatch&) override {}   // the ghost's BELIEF is not kept (none on these tapes)
    void forget_ghost(const Vec2f& xy) override
    { std::erase_if(ghosts_, [&](const Ghost& g) { return (g.xy - xy).norm() < cfg_.exist_reacquire_radius_m; }); }
    std::optional<std::string> adopt_birth(std::uint64_t id, int slice, const Vec3f&) override
    {
        // ensure_instance: the instance already exists (create_door above, as the synchronous slot made it live), so it
        // only takes its birth slice.
        const auto it = live_.find(id);
        if (it == live_.end()) return std::nullopt;
        it->second.assigned_mask_idx = slice;
        const std::string name = it->second.node_name;
        // note_identity(name, birth xy): the IdentityUpsert row the old code wrote right here, if it wrote one
        for_rows([&](const DoorDecision& r)
        {
            if (r.kind == static_cast<std::uint16_t>(K::IdentityUpsert) and r.name == name) upsert(r);
        });
        births_named_.push_back(name);
        return name;
    }

    void set_fit(DoorFit* f) { fit_ = f; }
    int removed_ = 0, unmatched_creates_ = 0;
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
    void remember_ghost(const rc::DoorFitState& d)
    {
        // SpecificWorker::remember_ghost — keyed on the fitted aperture
        if (cfg_.exist_reacquire_radius_m <= 0.0f or not d.ai2_initialized or d.is_bearing_hypothesis) return;
        const Vec2f ap(d.model.state().ap_cx, d.model.state().ap_cy);
        std::erase_if(ghosts_, [&](const Ghost& g) { return (g.xy - ap).norm() < cfg_.exist_reacquire_radius_m; });
        ghosts_.push_back({d.node_name, ap, 0});
        // SpecificWorker::forget_identity(name, aperture, width): ABSENCE FORGETS THE NAME ACROSS RUNS — every stored row
        // of this room with this name or within the same hole (the rule the IdentityForget rows record).
        if (not room_name_.empty())
        {
            const float width_m = d.model.state().w;
            const float same_hole = cfg_.exist_reacquire_radius_m + std::max(0.0f, std::isfinite(width_m) ? width_m : 0.0f);
            std::erase_if(identities_, [&](const rc::door::IdentityRow& i)
                          { return i.room == room_name_ and (i.name == d.node_name or (ap.allFinite() and (i.xy - ap).norm() < same_hole)); });
        }
        while (static_cast<int>(ghosts_.size()) > std::max(1, cfg_.exist_ghost_max)) ghosts_.erase(ghosts_.begin());
    }

    const rc::DoorConfig& cfg_;
    const std::map<std::uint64_t, std::vector<DoorDecision>>& log_;
    const std::vector<DoorDecision>* rows_ = nullptr;
    const DoorCycleInput* in_ = nullptr;
    std::uint64_t cycle_ = 0, room_ = 0;
    std::string room_name_;
    std::vector<Vec2f> polygon_;
    mutable std::unordered_map<std::uint64_t, rc::DoorFitState> live_;
    DoorFit* fit_ = nullptr;
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
        std::println("usage: door_replay <door_inputs.bin> <door_decisions.bin> [--until CYCLE] [--verbose] [--csv DIR] [--trace NAME]");
        return 2;
    }
    std::uint64_t until = ~0ull;
    bool verbose = false, mirror = false;
    std::string csv_dir;
    // --clock-jitter-ms J: add a deterministic pseudo-random delay in [0, J) ms to every record's clock (steady and wall).
    // A SENSITIVITY probe, not a replay: it shows how far the Phi rows move when only the time at which the cycle is
    // read moves — which is what separates a pre-4b tape (mid-cycle clocks) from this core (cycle-top clock). The decision
    // stamps move with it, so with J > 0 the Phi comparison ignores t_ms.
    double jitter_ms = 0.0;
    // --trace NAME: one line per stepped cycle for that door — its fit [s, w, h, d], the leaf angle, the detection state and
    // every existence channel the core computed this cycle (Task 5b: the silhouette's occupancy vs absence, P(detect), the
    // image-geometry delta). A diagnostic only: it reads the world's fit states after the step and changes nothing.
    std::string trace_door;
    rc::EpistemicPlanner nbv_smoke(1.8f);   // the live-only consumers' smoke (see the step loop)
    double belief_smoke_sum = 0.0;
    long belief_smoke_runs = 0;
    float recess_prior_std = -2.0f;   // < -1.5: keep the recorded config's value
    bool look_log = false, ablate_hull = false;   // Task E: --look prints the look-vs-approach prices; --ablate-hull-midpoint reproduces the collapse
    for (int i = 3; i < argc; ++i)
    {
        const std::string_view a = argv[i];
        if (a == "--until" and i + 1 < argc) until = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--verbose") verbose = true;
        else if (a == "--mirror") mirror = true;
        else if (a == "--look") look_log = true;
        else if (a == "--ablate-hull-midpoint") ablate_hull = true;
        else if (a == "--csv" and i + 1 < argc) csv_dir = argv[++i];
        else if (a == "--clock-jitter-ms" and i + 1 < argc) jitter_ms = std::strtod(argv[++i], nullptr);
        else if (a == "--trace" and i + 1 < argc) trace_door = argv[++i];
        else if (a == "--recess-prior-std" and i + 1 < argc)
        {   // a SENSITIVITY probe (Task 5b): the leaf-plane depth's prior σ, overriding the recorded config
            const std::string_view v = argv[++i];
            std::from_chars(v.data(), v.data() + v.size(), recess_prior_std);
        }
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
    if (recess_prior_std > -1.5f)
    {
        cfg.door.door_recess_prior_std = recess_prior_std;
        std::println("probe: Door.RecessPriorStd overridden to {}", recess_prior_std);
    }
    if (ablate_hull) { cfg.door.replay_ablate_hull_midpoint = true; std::println("probe: the Task-W hull-mismatch term of the sides midpoint is OFF"); }
    std::println("tape: {} (DCIN v{}, agent {} '{}'); config: {} keys from the file, {} at code default, {} parse errors",
                 argv[1], version, hdr->agent_id, hdr->agent_name, eff.n_file, eff.n_default, cfg_errors);

    nbv_smoke.set_detector_envelope(rc::detect::DetectorEnvelope{cfg.door.detect_min_fill, cfg.door.detect_max_fill, cfg.door.detect_soft});
    ReplayWorld world(cfg.door, by_cycle);
    world.mirror_ = mirror;
    DoorCore core(cfg);
    world.set_fit(&core.fit());
    // A sink that only counts: the fit formats its stdout lines and CSV rows exactly as live (so the timing includes
    // that cost), and the tool writes no file.
    long diag_rows = 0;
    // --csv DIR: the fit's two logs and the existence log, as the live agent writes them (diff them against its own)
    std::ofstream ai2_csv, phi_csv, ex_csv;
    if (not csv_dir.empty())
    {
        ai2_csv.open(csv_dir + "/ai2_log.csv");
        phi_csv.open(csv_dir + "/ai2_log_phi.csv");
        ex_csv.open(csv_dir + "/door_existence_log.csv");
        ai2_csv << kAi2CsvHeader;
        phi_csv << kPhiCsvHeader;
        ex_csv << kExistenceCsvHeader;
    }
    core.connect(&world, {.decision = {}, .diag = [&](const DiagRow& r)
    {
        ++diag_rows;
        if (r.stream == "ai2_log.csv" and ai2_csv.is_open()) ai2_csv << r.line;
        else if (r.stream == "ai2_log_phi.csv" and phi_csv.is_open()) phi_csv << r.line;
        else if (r.stream == "door_existence_log.csv" and ex_csv.is_open()) ex_csv << r.line;
    }});

    // the slices the tool does not rebuild — counted, so a tape that exercised them says so
    int n_exec = 0, n_anchor = 0, n_act = 0;
    for (const auto& d : *ref_all)
    {
        n_exec   += d.kind == static_cast<std::uint16_t>(K::AffExecuting);
        n_anchor += d.kind == static_cast<std::uint16_t>(K::AnchorHung);
        n_act    += d.kind == static_cast<std::uint16_t>(K::ActuatorRequest);
    }

    std::vector<DoorDecision> got, got_phi, got_exist, got_remove;
    std::uint64_t n = 0, first = 0, last = 0;
    int stepped = 0, media_only = 0;
    double step_ms_sum = 0.0, step_ms_max = 0.0, copy_ms_sum = 0.0;
    std::vector<double> step_ms;
    while (true)
    {
        auto in = read_input(f, version);
        if (not in.has_value()) break;
        if (in->cycle > until) break;
        if (n++ == 0) first = in->cycle;
        last = in->cycle;
        if (jitter_ms > 0.0)
        {
            static std::uint64_t lcg = 0x9E3779B97F4A7C15ull;
            lcg = lcg * 6364136223846793005ull + 1442695040888963407ull;
            const double u = static_cast<double>(lcg >> 11) / 9007199254740992.0;   // [0, 1)
            const auto j_ns = static_cast<std::int64_t>(u * jitter_ms * 1e6);
            in->t_steady_ns += j_ns;
            in->t_wall_ms += j_ns / 1'000'000;
        }
        {   // the live snapshot's copy cost, proxied: one deep copy of the cycle's input (the packet, the sweep, the graph rows)
            const auto t0 = std::chrono::steady_clock::now();
            const DoorCycleInput copy = *in;
            copy_ms_sum += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (copy.cycle != in->cycle) return 2;
        }
        world.begin_cycle(*in);
        if (world.room_id() == 0)
        {
            core.observe_media(*in);   // the old compute() returned before the tracker
            ++media_only;
        }
        else
        {
            const auto t0 = std::chrono::steady_clock::now();
            const auto out = core.step(*in);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            step_ms_sum += ms; step_ms_max = std::max(step_ms_max, ms); step_ms.push_back(ms);
            ++stepped;
            for (const auto& d : out.decisions)
            {
                if (is_tracker_kind(d.kind)) got.push_back(d);
                if (d.kind == static_cast<std::uint16_t>(K::Phi)) got_phi.push_back(d);
                if (d.kind == static_cast<std::uint16_t>(K::Existence)) got_exist.push_back(d);
                if (d.kind == static_cast<std::uint16_t>(K::Remove)) got_remove.push_back(d);
            }
            // ★THE LIVE-ONLY BELIEF CONSUMERS, AS A SMOKE (fix for the rtD1g crash). The agent reads every door's belief
            // each cycle in paths the core does not own yet — the epistemic NBV planner, the dashboard's DOF table and
            // certainty, the published covariance — and one of them (EpistemicPlanner::compute, a DOF table sized for a
            // 4-state belief against the 3×3 reported Σ) asserted live where no replay could see it. Run them here on
            // every fitted door, every cycle; built without -DNDEBUG, an index error aborts the replay as it aborted
            // the agent. The NBV's proposal is not compared with anything (it is not on the tape): this is coverage.
            for (const auto& [id, st] : world.fit_states())
            {
                if (st == nullptr or not st->ai2_initialized) continue;
                const auto& b = st->ai2_belief;
                std::vector<Eigen::Vector2f> poly;
                for (const auto& v : world.room_polygon()) poly.emplace_back(v.x(), v.y());
                rc::nbv::Sensor sensor;
                // the Shadow's ZED (2i, 2K): ~110 x 70 deg, optical centre ~1.0 m above the floor in the room frame (shadow.json 0.945 + body)
                sensor.hfov_rad = 1.92f; sensor.vfov_rad = 1.22f; sensor.height_m = 1.0f;
                const auto prop = nbv_smoke.compute(b, cfg.door.ai2_range_noise_lat_per_m, cfg.door.ai2_sigma_base_m, sensor, {}, poly);
                if (look_log and in->robot.room_T_body.has_value())
                {
                    // Task E: the two prices on ONE scale (nats), as SpecificWorker prices them: the look = the NBV gain on the
                    // belief's REPORTED Σ; the approach = ValueApproach x P(complete | the reported width sigma).
                    const auto& pg = cfg.door.pragmatic;
                    const rc::door::Aperture ap = b.aperture();
                    const Eigen::Vector2f robot((*in->robot.room_T_body)(0, 3), (*in->robot.room_T_body)(1, 3));
                    rc::door::ApproachProblem apb;
                    apb.side = rc::door::approach_side(ap, robot, ap.across_u().dot(robot - ap.centre_xy()), 1.7f, poly);
                    apb.d_min = ap.w + 0.5f * pg.robot_passage_width_m + pg.swing_clearance_m;
                    apb.d_max = std::max(pg.actuation_reach_m, apb.d_min + 0.30f);
                    apb.d_min = std::min(apb.d_min, apb.d_max - 0.30f);
                    apb.sigma = pg.pose_sigma_floor_m; apb.sigma_arrival = pg.pose_sigma_floor_m;
                    apb.needed_w = pg.robot_passage_width_m + pg.passage_margin_m;
                    apb.sigma_w = std::sqrt(std::max(0.0f, b.covariance_reported()(1, 1)));
                    apb.footprint_r = 0.5f * pg.robot_passage_width_m + pg.passage_margin_m;
                    const auto ch = rc::door::choose_approach_pose(ap, apb, rc::door::ApproachMemory{}, 0);
                    const float p_reach = rc::door::reach_prob_at(ap, apb, robot);
                    const float approach_value = pg.value_approach * ch.p_complete_live;
                    float standoff = 0.0f, off_deg = 0.0f;
                    if (prop.valid)
                    {
                        const Eigen::Vector2f d = Eigen::Vector2f(prop.epistemic_target_x_m, prop.epistemic_target_y_m) - ap.centre_xy();
                        standoff = d.norm();
                        off_deg = std::acos(std::clamp(std::abs(d.dot(ap.across_u())) / std::max(1e-6f, d.norm()), 0.f, 1.f)) * 57.2958f;
                    }
                    std::println("LOOK {} cycle {} w {:.3f} sd_w_post {:.3f} sd_w_rep {:.3f} | look {:.3f} nats at {:.2f} m {:.1f} deg off the normal (valid {}) | "
                                 "approach {:.3f} (P(complete) {:.3f}, P(reach) {:.2f}) | robot-to-door {:.2f} m | d_look {:.2f} d_appr {:.2f}",
                                 st->node_name, in->cycle, b.state().w, std::sqrt(std::max(0.0f, b.covariance()(1, 1))), apb.sigma_w,
                                 prop.valid ? prop.epistemic_gain : 0.0f, standoff, off_deg, prop.valid ? 1 : 0,
                                 approach_value, ch.p_complete_live, p_reach, (robot - ap.centre_xy()).norm(),
                                 prop.valid ? (robot - Eigen::Vector2f(prop.epistemic_target_x_m, prop.epistemic_target_y_m)).norm() : 0.0f,
                                 (robot - Eigen::Vector2f(ch.pose.x, ch.pose.y)).norm());
                }
                const auto S = b.covariance_reported();
                double acc = rc::any_sigma_star(rc::kDoorDofs) ? rc::adequacy_gap_nats(rc::kDoorDofs, [&](std::size_t j) { return S(j, j); }) : 0.0;
                for (std::size_t j = 0; j < rc::kDoorDofs.size(); ++j) acc += S(static_cast<int>(j), static_cast<int>(j));
                acc += b.covariance()(0, 0) + b.recess_sd();
                belief_smoke_sum += acc;
                ++belief_smoke_runs;
            }
            if (not trace_door.empty())
                for (const auto& [id, st] : world.fit_states())
                    if (st != nullptr and (trace_door == "*" or st->node_name == trace_door))
                    {
                        const auto& b = st->ai2_belief.state();
                        std::size_t mask_px = 0;   // the pixels of the slice the tracker assigned this cycle (0: none)
                        if (in->masks and st->frames_since_detection == 0 and st->assigned_mask_idx >= 0
                            and static_cast<std::size_t>(st->assigned_mask_idx) < in->masks->slices.size())
                        {
                            const auto& sl = in->masks->slices[static_cast<std::size_t>(st->assigned_mask_idx)];
                            mask_px = sl.pixel_end - sl.pixel_begin;
                        }
                        std::println("TRACE {} cycle {} L {:+.2f} | s {:.3f} w {:.3f} h {:.3f} d {:+.3f}±{:.3f} phi {:.3f}±{:.3f} | "
                                     "since_det {} mask {} px {} | sil occ {:.1f} free {:.1f} free_eff {:.2f} p_det {:.3f} | edge dL {:+.2f} "
                                     "frame exc {:+.2f} (true {:.1f} ctrl {:.1f}) | robot {:.3f} {:.3f} | kappa {:.3f}",
                                     st->node_name, in->cycle, st->existence.logodds(), b.s, b.w, b.h, b.d, st->ai2_belief.recess_sd(),
                                     st->phi_est, st->phi_sigma, st->frames_since_detection, st->assigned_mask_idx, mask_px,
                                     st->dbg_sil_occ, st->dbg_sil_free, st->dbg_sil_free_eff, st->dbg_sil_pdetect,
                                     st->dbg_edge_delta, st->dbg_frame_excess, st->dbg_frame_true, st->dbg_frame_ctrl,
                                     in->robot.room_T_body ? (*in->robot.room_T_body)(0, 3) : 0.0, in->robot.room_T_body ? (*in->robot.room_T_body)(1, 3) : 0.0, st->dbg_pose_kappa);
                    }
        }
        world.end_cycle();
    }
    std::println("replayed {} records (cycles {}..{}): stepped {}, media-only {}", n, first, last, stepped, media_only);
    std::println("live-only belief consumers (NBV planner, DOF table, reported Σ) ran {} times, clean (checksum {:.4g})",
                 belief_smoke_runs, belief_smoke_sum);
    std::println("removals decided by the core: {}; births without a recorded id: {}; not rebuilt: {} AffExecuting, "
                 "{} AnchorHung, {} ActuatorRequest rows on this tape", world.removed_, world.unmatched_creates_,
                 n_exec, n_anchor, n_act);
    if (not step_ms.empty())
    {
        std::ranges::sort(step_ms);
        std::println("cost: DoorCore::step {:.3f} ms/step mean, p50 {:.3f}, p99 {:.3f}, max {:.3f} ms ({} fit diag rows formatted); "
                     "input deep copy {:.3f} ms/record mean",
                     step_ms_sum / static_cast<double>(step_ms.size()), step_ms[step_ms.size() / 2],
                     step_ms[std::min(step_ms.size() - 1, step_ms.size() * 99 / 100)], step_ms_max, diag_rows,
                     copy_ms_sum / static_cast<double>(n));
    }
    // ── Phi rows: every one, in order ──
    std::vector<DoorDecision> ref_phi;
    for (const auto& d : *ref_all)
        if (d.kind == static_cast<std::uint16_t>(K::Phi) and d.cycle <= until) ref_phi.push_back(d);
    int phi_mismatch = 0;
    {
        struct FieldDiff { double max_abs = 0, max_rel = 0; long n = 0; std::string first; };
        FieldDiff fa, fb, fc, fd;
        long equal = 0, key_diff = 0;
        const std::size_t m = std::max(ref_phi.size(), got_phi.size());
        const auto note = [](FieldDiff& fd_, double r, double g, const DoorDecision& d, const char* name)
        {
            if (r == g) return;
            ++fd_.n;
            const double ad = std::abs(g - r);
            fd_.max_abs = std::max(fd_.max_abs, ad);
            fd_.max_rel = std::max(fd_.max_rel, ad / std::max(1e-12, std::abs(r)));
            if (fd_.first.empty()) fd_.first = std::format("cycle {} {} {}: ref {:.9g} replay {:.9g}", d.cycle, d.name, name, r, g);
        };
        for (std::size_t i = 0; i < m; ++i)
        {
            const DoorDecision* r = i < ref_phi.size() ? &ref_phi[i] : nullptr;
            const DoorDecision* g = i < got_phi.size() ? &got_phi[i] : nullptr;
            if (r and g and *r == *g) { ++equal; continue; }
            ++phi_mismatch;
            if (r and g and jitter_ms > 0.0 and r->t_ms != g->t_ms)   // the stamp moved with the probe: compare the rest
            {
                DoorDecision gg = *g;
                gg.t_ms = r->t_ms;
                if (*r == gg) { ++equal; --phi_mismatch; continue; }
            }
            if (not (r and g) or r->cycle != g->cycle or r->door_id != g->door_id or r->name != g->name
                or (r->t_ms != g->t_ms and jitter_ms <= 0.0))
            {
                if (++key_diff <= 3)
                    std::println("  PHI row #{} differs in its KEY: ref {} / replay {}", i, r ? show(*r) : std::string{"(none)"},
                                 g ? show(*g) : std::string{"(none)"});
                continue;
            }
            note(fa, r->a, g->a, *r, "phi_est"); note(fb, r->b, g->b, *r, "phi_sigma");
            note(fc, r->c, g->c, *r, "measured"); note(fd, r->d, g->d, *r, "phi_support");
        }
        std::println("\nPHI rows: reference {}, replay {}; {} bit-equal, {} differ ({} in cycle/door/name)",
                     ref_phi.size(), got_phi.size(), equal, phi_mismatch, key_diff);
        const auto line = [](const char* nm, const FieldDiff& x)
        {
            if (x.n == 0) std::println("  {:<11}: all equal", nm);
            else std::println("  {:<11}: {} rows differ, max |diff| {:.3g}, max rel {:.3g}; first {}", nm, x.n, x.max_abs, x.max_rel, x.first);
        };
        line("phi_est", fa); line("phi_sigma", fb); line("measured", fc); line("phi_support", fd);
    }

    // ── Existence rows: every one, in order (a = L, b = p_exists, c,d = the fitted centre) ──
    std::vector<DoorDecision> ref_exist, ref_remove;
    for (const auto& d : *ref_all)
    {
        if (d.cycle > until) continue;
        if (d.kind == static_cast<std::uint16_t>(K::Existence)) ref_exist.push_back(d);
        if (d.kind == static_cast<std::uint16_t>(K::Remove)) ref_remove.push_back(d);
    }
    int exist_mismatch = 0;
    {
        struct FieldDiff { double max_abs = 0; long n = 0; std::string first; };
        FieldDiff fa, fb, fc, fd;
        long equal = 0, key_diff = 0;
        std::string first_diff;
        // per door: the first cycle whose row differs, and how far L drifted at most
        std::map<std::string, std::pair<std::uint64_t, double>> per_door;
        const std::size_t m = std::max(ref_exist.size(), got_exist.size());
        const auto note = [](FieldDiff& x, double r, double g, const DoorDecision& d, const char* nm)
        {
            if (r == g) return;
            ++x.n;
            x.max_abs = std::max(x.max_abs, std::abs(g - r));
            if (x.first.empty()) x.first = std::format("cycle {} {} {}: ref {:.9g} replay {:.9g}", d.cycle, d.name, nm, r, g);
        };
        for (std::size_t i = 0; i < m; ++i)
        {
            const DoorDecision* r = i < ref_exist.size() ? &ref_exist[i] : nullptr;
            const DoorDecision* g = i < got_exist.size() ? &got_exist[i] : nullptr;
            if (r and g and *r == *g) { ++equal; continue; }
            ++exist_mismatch;
            if (first_diff.empty()) first_diff = std::format("ref {} / replay {}", r ? show(*r) : "(none)", g ? show(*g) : "(none)");
            if (not (r and g) or r->cycle != g->cycle or r->door_id != g->door_id or r->name != g->name or r->t_ms != g->t_ms)
            {
                if (++key_diff <= 3)
                    std::println("  EXISTENCE row #{} differs in its KEY: ref {} / replay {}", i, r ? show(*r) : std::string{"(none)"},
                                 g ? show(*g) : std::string{"(none)"});
                continue;
            }
            note(fa, r->a, g->a, *r, "L"); note(fb, r->b, g->b, *r, "p_exists");
            note(fc, r->c, g->c, *r, "x"); note(fd, r->d, g->d, *r, "y");
            auto& pd = per_door[r->name];
            if (pd.first == 0) pd.first = r->cycle;
            pd.second = std::max(pd.second, std::abs(g->a - r->a));
        }
        std::println("\nEXISTENCE rows: reference {}, replay {}; {} bit-equal, {} differ ({} in cycle/door/name/t)",
                     ref_exist.size(), got_exist.size(), equal, exist_mismatch, key_diff);
        const auto line = [](const char* nm, const FieldDiff& x)
        {
            if (x.n == 0) std::println("  {:<9}: all equal", nm);
            else std::println("  {:<9}: {} rows differ, max |diff| {:.3g}; first {}", nm, x.n, x.max_abs, x.first);
        };
        line("L", fa); line("p_exists", fb); line("x", fc); line("y", fd);
        for (const auto& [name, v] : per_door)
            std::println("  {}: first differing row at cycle {}, max |ΔL| {:.3g}", name, v.first, v.second);
        if (not first_diff.empty()) std::println("  first difference: {}", first_diff);
    }
    int remove_mismatch = 0;
    {
        std::println("\nREMOVE decisions: reference {}, replay {}", ref_remove.size(), got_remove.size());
        const std::size_t m = std::max(ref_remove.size(), got_remove.size());
        for (std::size_t i = 0; i < m; ++i)
        {
            const DoorDecision* r = i < ref_remove.size() ? &ref_remove[i] : nullptr;
            const DoorDecision* g = i < got_remove.size() ? &got_remove[i] : nullptr;
            if (r and g and *r == *g) { std::println("  EQUAL  {}", show(*r)); continue; }
            ++remove_mismatch;
            std::println("  DIFF #{}:\n     ref    {}\n     replay {}", i, r ? show(*r) : std::string{"(none)"},
                         g ? show(*g) : std::string{"(none)"});
        }
    }

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
    std::println("\nTRACKER {}: {} decision(s) compared, {} mismatch(es)", mismatches == 0 ? "EQUAL" : "DIFFERENT", m, mismatches);
    std::println("PHI     {}: {} row(s) compared, {} mismatch(es)", phi_mismatch == 0 ? "EQUAL" : "DIFFERENT",
                 std::max(ref_phi.size(), got_phi.size()), phi_mismatch);
    std::println("EXIST   {}: {} row(s) compared, {} mismatch(es)", exist_mismatch == 0 ? "EQUAL" : "DIFFERENT",
                 std::max(ref_exist.size(), got_exist.size()), exist_mismatch);
    std::println("REMOVE  {}: {} decision(s) compared, {} mismatch(es)", remove_mismatch == 0 ? "EQUAL" : "DIFFERENT",
                 std::max(ref_remove.size(), got_remove.size()), remove_mismatch);
    return (mismatches == 0 and phi_mismatch == 0 and exist_mismatch == 0 and remove_mismatch == 0) ? 0 : 1;
}
