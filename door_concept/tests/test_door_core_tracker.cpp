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

// (xy, support top); `ricoh` makes every slice a 360-camera one: source 1, LiDAR-depth-filled (depth_var > 0).
struct Scene { std::vector<std::pair<Vec2f, float>> doors; bool ricoh = false; float conf = 1.0f; float depth_var = 0.0f; float range = 2.5f; Vec2f cam = Vec2f::Zero(); };

static DoorCycleInput cycle_input(std::uint64_t cycle, int frame_id, const Scene& sc)
{
    DoorCycleInput in;
    in.cycle = cycle;
    in.t_steady_ns = static_cast<std::int64_t>(cycle) * 100'000'000;   // 10 Hz
    in.robot.present = true;
    in.robot.room_T_body = Mat4d::Identity();
    in.robot.room_T_body->coeffRef(0, 3) = sc.cam.x(); in.robot.room_T_body->coeffRef(1, 3) = sc.cam.y();                          // the robot stands still ⇒ admissible
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
        auto sl = door_slice(xy.x(), xy.y(), b, p.support_points.size(), top);
        sl.confidence = sc.conf; sl.range = sc.range;
        if (sc.ricoh) { sl.source = 1; sl.depth_var = sc.depth_var; }
        p.slices.push_back(sl);
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

    // ── 3c. RICOH (360) door slices (task Q) ────────────────────────────────────────────────────────────────────
    // The door sits IN the wall y = 5 (the FakeWorld room is +-5). The robot circles at r = 1.5 so each frame is a new
    // viewpoint (frames from one viewpoint are ONE observation); ricoh depth_var 0.02 m^2, range ~5 m.
    const Vec2f wall_door(3.0f, 5.0f);
    const auto ricoh_scene = [&](Vec2f xy, int k, float conf, float dvar)
    {
        Scene sc{{{xy, 2.0f}}}; sc.ricoh = true; sc.conf = conf; sc.depth_var = dvar;
        sc.cam = Vec2f(1.5f * std::cos(0.45f * k), 1.5f * std::sin(0.45f * k));
        sc.range = (xy - sc.cam).norm();
        return sc;
    };
    {   // (a) a door seen ONLY by the ricoh, from moving viewpoints: born later than a ZED one, once, at the pose
        FakeWorld w;
        DoorCore core(cfg);
        core.connect(&w);
        int birth_cycle = -1, removes = 0;
        std::vector<rc::door::DoorDecision> births;
        for (int k = 1; k <= 400; ++k)
        {
            const auto out = core.step(cycle_input(k, k, ricoh_scene(wall_door, k, 0.9f, 0.02f)));
            const auto b = of_kind(out.decisions, K::Birth);
            if (not b.empty()) { births.insert(births.end(), b.begin(), b.end()); if (birth_cycle < 0) birth_cycle = k; }
            removes += static_cast<int>(of_kind(out.decisions, K::Remove).size());
        }
        // SYMMETRY: absence evidence is only ever the ZED silhouette's P(detect)-weighted absence; the ricoh not detecting
        // a door is weight 0 (it has no absence channel), so 300 frames with NO ricoh slice remove nothing.
        for (int k = 401; k <= 700; ++k)
            removes += static_cast<int>(of_kind(core.step(cycle_input(k, k, Scene{})).decisions, K::Remove).size());
        std::print("ricoh-only birth after {} frames; removes {}\n", birth_cycle, removes);
        CHECK(births.size() == 1 and w.created == 1);           // exactly once over 400 frames (no remove/rebirth churn)
        CHECK(removes == 0 and w.live.size() == 1);
        CHECK(birth_cycle > 7 and birth_cycle <= 150);
        if (not births.empty())
            CHECK(std::abs(births[0].a - wall_door.x()) < 0.05 and std::abs(births[0].b - wall_door.y()) < 0.05);
    }
    {   // (b)+(d) CHURN: a long run of scattered low-confidence, high-variance ricoh slices ⇒ 0 births, 0 removals
        FakeWorld w;
        DoorCore core(cfg);
        core.connect(&w);
        std::uint32_t rng = 12345u;
        const auto u = [&] { rng = rng * 1664525u + 1013904223u; return static_cast<float>(rng >> 8) / 16777216.0f; };
        int removes = 0;
        for (int k = 1; k <= 600; ++k)
        {
            const Vec2f xy = (k % 2) ? Vec2f(-4.0f + 8.0f * u(), -4.0f + 8.0f * u())          // anywhere in the room
                                     : Vec2f(-4.0f + 8.0f * u(), 5.0f);                       // anywhere along a wall
            const auto out = core.step(cycle_input(k, k, ricoh_scene(xy, k, 0.3f, 0.5f)));
            removes += static_cast<int>(of_kind(out.decisions, K::Remove).size());
        }
        CHECK(w.created == 0 and removes == 0);
    }
    {   // a PARKED robot: 150 frames of one viewpoint are one observation ⇒ a stationary robot cannot accumulate a phantom
        FakeWorld w;
        DoorCore core(cfg);
        core.connect(&w);
        for (int k = 1; k <= 150; ++k)
        {
            auto sc = ricoh_scene(wall_door, 0, 0.9f, 0.02f);
            core.step(cycle_input(k, k, sc));
        }
        CHECK(w.created == 0);
    }
    {   // a slice whose LiDAR range disagrees with the wall its bearing meets (depth landed on something else) births nothing
        FakeWorld w;
        DoorCore core(cfg);
        core.connect(&w);
        for (int k = 1; k <= 300; ++k)
            core.step(cycle_input(k, k, ricoh_scene(Vec2f(2.0f, 1.0f), k, 0.9f, 0.02f)));   // mid-room, not in any wall
        CHECK(w.created == 0);
    }
    {   // a ricoh slice CONFIRMS/associates to an existing door instead of being dropped
        FakeWorld w;
        DoorCore core(cfg);
        core.connect(&w);
        w.adopt_birth(42, -1, Vec3f(wall_door.x(), wall_door.y(), 1.0f));
        core.step(cycle_input(1, 1, ricoh_scene(wall_door, 1, 0.9f, 0.02f)));
        core.step(cycle_input(2, 2, ricoh_scene(wall_door, 2, 0.9f, 0.02f)));
        CHECK(core.tracker_info().class_dets == 1 and core.tracker_info().assigned == 1);
    }

    {   // (I4a) a PERSISTENT low-quality ricoh 'door' at a fixed wall spot ⇒ no birth; the SAME geometry at good quality ⇒ birth
        int created[2] = {0, 0};
        const float quality[2][2] = {{0.3f, 0.5f}, {0.9f, 0.02f}};   // (conf, depth_var): poor, then good
        for (int q = 0; q < 2; ++q)
        {
            FakeWorld w;
            DoorCore core(cfg);
            core.connect(&w);
            for (int k = 1; k <= 600; ++k)
                core.step(cycle_input(k, k, ricoh_scene(wall_door, k, quality[q][0], quality[q][1])));
            created[q] = w.created;
        }
        std::print("persistent ricoh door: poor quality births {}, good quality births {}\n", created[0], created[1]);
        CHECK(created[0] == 0);
        CHECK(created[1] == 1);
    }
    {   // (I3) ZED and ricoh both see ONE door (ricoh offset 0.25 m ≈ 2 sigma along the wall): exactly one instance, and the two
        // sensors' evidence accrues to ONE candidate, so it is born SOONER than from the ZED alone (the old per-frame dup rule
        // dropped the second sensor's evidence), whichever sensor is listed first.
        const auto run_mixed = [&](int mode, int& births_out, std::size_t& live_out)   // 0 zed alone, 1 zed+ricoh, 2 ricoh+zed
        {
            FakeWorld w;
            DoorCore core(cfg);
            core.connect(&w);
            int first = -1; births_out = 0;
            for (int k = 1; k <= 150; ++k)
            {
                auto z = ricoh_scene(wall_door, k, 1.0f, 0.0f); z.ricoh = false;       // the ZED slice, same viewpoint
                auto r = ricoh_scene(wall_door + Vec2f(0.25f, 0.0f), k, 0.9f, 0.02f);
                DoorCycleInput in = cycle_input(k, k, mode == 2 ? r : z);
                if (mode != 0)
                {
                    auto other = cycle_input(k, k, mode == 2 ? z : r);
                    const std::uint64_t b = in.masks->support_points.size();
                    for (const auto& pt : other.masks->support_points) in.masks->support_points.push_back(pt);
                    auto sl = other.masks->slices[0];
                    sl.support_begin += b; sl.support_end += b;
                    in.masks->slices.push_back(sl);
                }
                const auto nb = of_kind(core.step(in).decisions, K::Birth).size();
                births_out += static_cast<int>(nb);
                if (nb > 0 and first < 0) first = k;
            }
            live_out = w.live.size();
            return first;
        };
        int nb[3]; std::size_t nl[3];
        const int t0 = run_mixed(0, nb[0], nl[0]), t1 = run_mixed(1, nb[1], nl[1]), t2 = run_mixed(2, nb[2], nl[2]);
        std::print("birth frame: zed alone {}, zed+ricoh {}, ricoh+zed {}; instances {} {} {}\n", t0, t1, t2, nl[0], nl[1], nl[2]);
        CHECK(t0 > 0 and t1 > 0 and t2 > 0);
        CHECK(nb[1] == 1 and nl[1] == 1 and nb[2] == 1 and nl[2] == 1);
        CHECK(t1 < t0 and t2 < t0);
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
