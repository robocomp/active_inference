// test_door_core_tracker.cpp — the DoorCore's TRACKER slice on synthetic inputs (Task 4 of the door-core replay plan).
//
//   1. BIRTH: a "door" mask slice seen on successive NEW frames accrues birth evidence until the InstanceTracker
//      promotes it; the core asks the world to create and adopt the door and records a BIRTH decision at the slice
//      centroid, named by the world.
//   2. ASSOCIATION: the next packet at the same place is assigned to the newborn (Pass 2: no belief yet, nearest slice
//      within the fallback gate of its birth spot) and births nothing.
//   3. A REPEATED frame id accrues nothing (rule 1 of birth_evidence.h: an observation, not a cycle).
//   4. SUPPRESSIONS: a detection outside the room polygon is suppressed with the "outside room" decision; a short blob
//      (support top below Existence.MinHeightM, untruncated) with the "top" one; a detection inside a live door's swept
//      arc with that door's id.
//   5. The core is DSR / Qt / Ice / media free (the include scan of tests/test_door_cycle_io.cpp, run over src/core).
//
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 -I/usr/include/opencv4 tests/test_door_core_tracker.cpp src/core/door_core.cpp src/core/door_fit.cpp \
//       src/core/door_existence.cpp src/core/door_cycle_io.cpp src/door_belief.cpp src/door_model.cpp src/door_frame_evidence.cpp \
//       ../common/contour_edge/contour_edge_check.cpp ../common/contour_edge/contour_edge_project.cpp \
//       ../common/contour_edge/contour_depth_check.cpp -lzstd -lopencv_core -lopencv_imgproc -o /tmp/test_door_core_tracker && /tmp/test_door_core_tracker
#include <clocale>
#include <filesystem>
#include <fstream>
#include <map>
#include <print>
#include <regex>
#include <set>
#include <string>
#include <vector>
#include "../src/core/door_core.h"

namespace fs = std::filesystem;
using namespace rc::door::core;
using K = rc::door::DoorDecisionKind;

static int fails = 0;
#define CHECK(c) do { if (not (c)) { ++fails; std::print("FAIL {}:{}: {}\n", __FILE__, __LINE__, #c); } } while (0)

// A world with no DSR: instances in a vector (insertion order = iteration order), names door_1, door_2, …
struct FakeWorld final : TrackerWorld
{
    std::uint64_t room = 7;
    std::vector<Vec2f> polygon{{-5.f, -5.f}, {5.f, -5.f}, {5.f, 5.f}, {-5.f, 5.f}};
    std::vector<TrackedDoor> live;
    std::uint64_t next_id = 1000;
    int created = 0, after_assoc = 0;
    std::vector<std::uint64_t> retired;

    std::uint64_t room_id() const override { return room; }
    std::vector<Vec2f> room_polygon() const override { return polygon; }
    std::vector<TrackedDoor> doors() const override { return live; }
    std::optional<GhostMatch> match_ghost(const Vec2f&) const override { return std::nullopt; }
    std::optional<IdentityMatch> match_identity(const Vec2f&) const override { return std::nullopt; }
    std::vector<std::string> reserved_names() const override { return {}; }
    void retire(std::uint64_t id) override { retired.push_back(id); std::erase_if(live, [&](const auto& d) { return d.id == id; }); }
    void set_assigned_mask(std::uint64_t id, int slice) override
    { for (auto& d : live) if (d.id == id) d.assigned_mask_idx = slice; }
    void after_association(const TrackerCycleInfo&) override { ++after_assoc; }
    void remove_door(std::uint64_t, float) override {}   // (the existence slice: no fit states here, nothing to remove)
    std::uint64_t create_door(const BirthRequest&) override { ++created; return next_id++; }
    void reacquire(std::uint64_t, const GhostMatch&) override {}
    void forget_ghost(const Vec2f&) override {}
    std::optional<std::string> adopt_birth(std::uint64_t id, int slice, const Vec3f& c) override
    {
        // ensure_instance's pre-belief geometry at the birth seed (door_model.h — the agent's own function).
        TrackedDoor d;
        d.id = id;
        d.name = "door_" + std::to_string(live.size() + 1);
        rc::DoorState s;
        s.cx = c.x(); s.cy = c.y(); s.w = 1.0f; s.h = 2.1f; s.thickness = 0.05f;
        const auto g = rc::pre_belief_geometry(s);
        rc::read_back_leaf(s, g.leaf_pose, g.leaf);
        d.state = s; d.aperture = g.aperture; d.leaf_pose = g.leaf_pose;
        d.assigned_mask_idx = slice;
        live.push_back(d);
        return d.name;
    }
};

// One "door" slice at (x, y) with support up to `top` m: zed, depth, confident, at the reference range 2.5 m (full birth weight).
static MaskSlice door_slice(float x, float y, std::uint64_t b, std::uint64_t e, float top = 2.0f)
{
    MaskSlice s;
    s.label = "door"; s.class_id = 1000.f; s.confidence = 1.0f;
    s.centroid = {x, y, 1.0f}; s.bbox_min = {x - 0.4f, y - 0.05f, 0.0f}; s.bbox_max = {x + 0.4f, y + 0.05f, top};
    s.support_begin = b; s.support_end = e;
    s.trunc_frac = 0.0f; s.range = 2.5f; s.source = 0; s.has_depth = true; s.depth_var = 0.0f;
    return s;
}

struct Scene { std::vector<std::pair<Vec2f, float>> doors; };   // (xy, support top)

static DoorCycleInput cycle_input(std::uint64_t cycle, int frame_id, const Scene& sc)
{
    DoorCycleInput in;
    in.cycle = cycle;
    in.t_steady_ns = static_cast<std::int64_t>(cycle) * 100'000'000;   // 10 Hz
    in.robot.present = true;
    in.robot.room_T_body = Mat4d::Identity();                          // the robot stands still ⇒ admissible
    in.robot.body_T_zed = Mat4d::Identity();
    MaskPacket p;
    p.valid = true;
    p.frame_id = frame_id;
    p.timestamp_ms = 1000 + static_cast<std::uint64_t>(frame_id);
    for (const auto& [xy, top] : sc.doors)
    {
        const std::uint64_t b = p.support_points.size();
        for (int i = 0; i < 20; ++i)
            p.support_points.emplace_back(xy.x() - 0.4f + 0.04f * i, xy.y(), 0.1f * (i % 20));
        p.slices.push_back(door_slice(xy.x(), xy.y(), b, p.support_points.size(), top));
    }
    in.masks = std::move(p);
    return in;
}

static std::vector<rc::door::DoorDecision> of_kind(const std::vector<rc::door::DoorDecision>& v, K k)
{
    std::vector<rc::door::DoorDecision> o;
    for (const auto& d : v) if (d.kind == static_cast<std::uint16_t>(k)) o.push_back(d);
    return o;
}

static void collect_includes(const fs::path& file, std::set<fs::path>& seen, std::vector<std::string>& angle)
{
    const auto canon = fs::weakly_canonical(file);
    if (not seen.insert(canon).second or not fs::exists(canon)) return;
    std::ifstream f(canon);
    static const std::regex inc(R"(^\s*#\s*include\s*([<"])([^>"]+)[>"])");
    for (std::string line; std::getline(f, line);)
        if (std::smatch m; std::regex_search(line, m, inc))
        {
            if (m[1] == "<") angle.push_back(m[2].str());
            else             collect_includes(canon.parent_path() / m[2].str(), seen, angle);
        }
}

int main()
{
    // ★The agents' locale (CLAUDE.md): the core's CSV row must not pick up a decimal comma under es_ES.
    if (std::setlocale(LC_ALL, "es_ES.UTF-8") == nullptr) std::setlocale(LC_ALL, "");

    DoorCoreConfig cfg;   // DoorConfig's code defaults: birth_frames 6.6, fallback gate 0.40 m, min height 1.80 m
    const Vec2f door_xy(2.0f, 1.0f);

    // ── 1–3. birth, association, repeated frame ──────────────────────────────────────────────────────────────────
    {
        FakeWorld w;
        DoorCore core(cfg);
        std::vector<std::string> csv_rows;
        core.connect(&w, {.decision = {}, .diag = [&](const DiagRow& r) { if (r.stream == "door_dets_log.csv") csv_rows.push_back(r.line); }});
        const Scene sc{{{door_xy, 2.0f}}};
        std::uint64_t cyc = 0;
        int frame = 0;
        std::vector<rc::door::DoorDecision> births;
        int birth_cycle = -1;
        for (int k = 0; k < 20 and births.empty(); ++k)
        {
            const auto out = core.step(cycle_input(++cyc, ++frame, sc));
            births = of_kind(out.decisions, K::Birth);
            if (not births.empty()) birth_cycle = static_cast<int>(cyc);
            CHECK(core.tracker_info().class_dets == 1);
        }
        std::print("birth after {} new frames\n", birth_cycle);
        CHECK(births.size() == 1);
        CHECK(birth_cycle == 7);                                  // ceil(6.6) observations of weight 1.0
        if (not births.empty())
        {
            CHECK(births[0].door_id == 1000);
            CHECK(births[0].name == "door_1");
            CHECK(births[0].a == static_cast<double>(door_xy.x()) and births[0].b == static_cast<double>(door_xy.y()));
            CHECK(births[0].cycle == static_cast<std::uint64_t>(birth_cycle));
        }
        CHECK(w.created == 1 and w.live.size() == 1);
        CHECK(w.after_assoc == birth_cycle);                      // the old code's existence point ran every cycle
        // the dets CSV row: cx,cy then range 2.5 — written "2.5" under es_ES (classic locale), never "2,5"
        CHECK(not csv_rows.empty() and csv_rows.back().find(",2,1,2.5,") != std::string::npos);

        // the next packet associates to the newborn (Pass 2), and births nothing
        const auto out2 = core.step(cycle_input(++cyc, ++frame, sc));
        CHECK(of_kind(out2.decisions, K::Birth).empty());
        CHECK(w.live.size() == 1 and w.live[0].assigned_mask_idx == 0);
        CHECK(core.tracker_info().assigned == 1);                 // and the InstanceTracker gated it to the track

        // a REPEATED frame id: still associated, still nothing new
        const auto out3 = core.step(cycle_input(++cyc, frame, sc));
        CHECK(of_kind(out3.decisions, K::Birth).empty());
        CHECK(not core.tracker_info().new_frame);
        CHECK(w.created == 1);
    }

    // ── 3b. repeated frames never accrue: 20 cycles of ONE frame id births nothing ───────────────────────────────
    {
        FakeWorld w;
        DoorCore core(cfg);
        core.connect(&w);
        const Scene sc{{{door_xy, 2.0f}}};
        int nb = 0;
        for (std::uint64_t c = 1; c <= 20; ++c)
            nb += static_cast<int>(of_kind(core.step(cycle_input(c, 5, sc)).decisions, K::Birth).size());
        CHECK(nb == 0 and w.created == 0);
    }

    // ── 4. suppressions ──────────────────────────────────────────────────────────────────────────────────────────
    {
        FakeWorld w;
        DoorCore core(cfg);
        core.connect(&w);
        // outside the room (|x| = 8 > 5 + 0.40 margin) and a 1.2 m blob inside it
        const Scene sc{{{Vec2f(8.0f, 0.0f), 2.0f}, {Vec2f(-2.0f, -2.0f), 1.2f}}};
        std::vector<rc::door::DoorDecision> sup;
        for (std::uint64_t c = 1; c <= 8; ++c)
            for (const auto& d : of_kind(core.step(cycle_input(c, static_cast<int>(c), sc)).decisions, K::Suppress))
                sup.push_back(d);
        CHECK(sup.size() >= 2);
        bool outside = false, short_blob = false;
        for (const auto& d : sup)
        {
            outside    |= d.note == "outside room" and d.a == 8.0;
            short_blob |= d.note == "top 1.20m" and d.a == -2.0;
        }
        CHECK(outside and short_blob);
        CHECK(w.created == 0);
    }
    {
        // a live door's swept arc explains a detection 0.9 m from its aperture (r = half_w + w + half_t ≈ 1.5 m) —
        // and the InstanceTracker's own anti-duplicate separation (0.70 m) does not, so this is the arc rule.
        FakeWorld w;
        DoorCore core(cfg);
        core.connect(&w);
        w.adopt_birth(42, -1, Vec3f(0.0f, 0.0f, 1.0f));
        w.live[0].name = "door_9";
        const Scene sc{{{Vec2f(0.9f, 0.0f), 2.0f}}};
        std::vector<rc::door::DoorDecision> sup;
        for (std::uint64_t c = 1; c <= 8; ++c)
            for (const auto& d : of_kind(core.step(cycle_input(c, static_cast<int>(c), sc)).decisions, K::Suppress))
                sup.push_back(d);
        CHECK(not sup.empty());
        if (not sup.empty())
        {
            CHECK(sup[0].door_id == 42 and sup[0].name == "door_9");
            CHECK(sup[0].note.starts_with("leaf swept arc"));
        }
        CHECK(w.created == 0);
    }

    // ── 5. the core is DSR / Qt / Ice / media free ───────────────────────────────────────────────────────────────
    {
        const fs::path core_dir = fs::path(__FILE__).parent_path() / ".." / "src" / "core";
        std::set<fs::path> seen;
        std::vector<std::string> angle;
        int n_files = 0;
        bool has_door_core = false;
        for (const auto& e : fs::recursive_directory_iterator(core_dir))
            if (e.is_regular_file())
            {
                ++n_files;
                has_door_core |= e.path().filename() == "door_core.cpp";
                collect_includes(e.path(), seen, angle);
            }
        static const std::regex forbidden(
            R"((^|/)(dsr/|qt[a-z0-9]*/|ice/|iceutil|media_transport|mask_ingestor|lidar_ingestor|rgb_ingestor|depth_ingestor|genericworker|specificworker|cortex|fastdds|fastrtps))",
            std::regex::icase);
        static const std::regex forbidden_qt(R"((^|/)(Q[A-Z][A-Za-z]*$|Qt[A-Za-z]*/))");
        const auto bad = [](const std::string& x) { return std::regex_search(x, forbidden) or std::regex_search(x, forbidden_qt); };
        int hits = 0;
        for (const auto& a : angle) if (bad(a)) { ++hits; std::print("  forbidden include <{}>\n", a); }
        for (const auto& p : seen)  if (bad(p.string())) { ++hits; std::print("  forbidden local include {}\n", p.string()); }
        std::print("core include scan: {} file(s) under src/core, {} reachable local header(s), {} forbidden\n",
                   n_files, seen.size(), hits);
        CHECK(has_door_core);
        CHECK(hits == 0);
    }

    if (fails == 0) std::print("test_door_core_tracker: ALL PASS\n");
    else            std::print("test_door_core_tracker: {} FAILURE(S)\n", fails);
    return fails == 0 ? 0 : 1;
}
