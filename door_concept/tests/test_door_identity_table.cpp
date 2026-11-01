// Offline harness for door_identity_table.h: the room-keyed identity table (etc/door_identities.csv) round trip,
// including the old 4-column rows, and the RESTORE decision — which remembered doors come back when a room is
// re-entered from memory (tourP2: room_1 re-entered, door_1 at (-0.54,4.61) was gone until re-seen).
// Runs under the agent's locale (es_ES, comma decimals) on purpose: CLAUDE.md, "reproduce it before trusting a test".
// Build + run:
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_door_identity_table.cpp -o /tmp/test_door_identity_table && /tmp/test_door_identity_table
#include <clocale>
#include <cstdint>
#include <cmath>
#include <print>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include "../src/door_identity_table.h"
#include "../src/door_leaf_channel.h"   // the borrowed hazard's source
static_assert(rc::door::kRememberedExistenceHazardPerS == rc::door::kLeafMoveHyperShape / rc::door::kLeafMoveHyperTimeS,
              "the remembered-existence hazard is the leaf channel's move-hazard hyperprior (door_identity_table.h)");
#include <chrono>
#include <optional>

struct State { float cx = 0, cy = 0, cz = 0, yaw = 0, w = 0.7f, h = 2.f, thickness = 0.05f, phi = 0, ap_cx = 0, ap_cy = 0, ap_yaw = 0; };

using namespace rc::door;
static int fails = 0;
#define CHECK(c) do { if (not (c)) { ++fails; std::print("FAIL {}:{}  {}\n", __FILE__, __LINE__, #c); } } while (0)
static bool near(float a, float b, float tol = 1e-5f) { return std::abs(a - b) <= tol; }

static constexpr std::int64_t kNowMs = 1'792'612'380'153;   // a real wall-clock stamp (rtD1i): 13 digits, not a float
static IdentityRow full_row(std::string room, std::string name, float x, float y)
{
    IdentityRow r;
    r.room = std::move(room); r.name = std::move(name); r.xy = {x, y};
    r.pose = RestorePose{3.10f, 0.95f, 2.13f, 0.05f};
    r.exist_logodds = 4.0f;
    r.leaf = LeafMemory{{-0.09f, 4.61f}, {0.0f, -1.0f}, 19.7f, 9.3f, 0.031f, 0.12f};
    r.seen_wall_ms = kNowMs;   // seen just now
    return r;
}

int main()
{
    std::setlocale(LC_ALL, "");   // the agent runs es_ES under Qt; from_chars must not care

    // ── 1. Round trip: full rows, an identity-only row, and an exist-less row survive write → read exactly ──
    {
        std::vector<IdentityRow> rows{full_row("room_1", "door_1", -0.5432f, 4.6101f)};
        IdentityRow birth;  birth.room = "room_1"; birth.name = "door_2"; birth.xy = {0.59f, 2.54f};   // birth: no pose
        rows.push_back(birth);
        IdentityRow noexist = full_row("room_2", "door_3", 3.6f, 2.18f);
        noexist.exist_logodds.reset(); noexist.leaf.reset();
        rows.push_back(noexist);
        std::stringstream ss;
        write_identity_table(ss, rows);
        const std::string text = ss.str();
        CHECK(text.find(',', 0) != std::string::npos);
        CHECK(text.starts_with(std::string(kIdentityHeader) + "\n"));
        CHECK(text.find("-0.5432") != std::string::npos);    // points, never comma decimals
        const auto p = parse_identity_table(ss);
        CHECK(not p.unkeyed_format);
        CHECK(p.malformed == 0);
        CHECK(p.rows.size() == 3);
        if (p.rows.size() == 3)
        {
            const auto& a = p.rows[0];
            CHECK(a.room == "room_1" and a.name == "door_1");
            CHECK(near(a.xy.x(), -0.5432f) and near(a.xy.y(), 4.6101f));
            CHECK(a.pose.has_value() and near(a.pose->yaw, 3.10f) and near(a.pose->width, 0.95f)
                  and near(a.pose->height, 2.13f) and near(a.pose->depth, 0.05f));
            CHECK(a.exist_logodds.has_value() and near(*a.exist_logodds, 4.0f));
            CHECK(a.seen_wall_ms.has_value() and *a.seen_wall_ms == kNowMs);   // Task D-C: the age, exactly (int64)
            CHECK(a.leaf.has_value() and near(a.leaf->hinge_xy.x(), -0.09f) and near(a.leaf->open_dir.y(), -1.0f)
                  and near(a.leaf->hinge_llr, 19.7f) and near(a.leaf->swing_llr, 9.3f)
                  and near(a.leaf->phi, 0.031f) and near(a.leaf->phi_sigma, 0.12f));
            CHECK(p.rows[1].name == "door_2" and not p.rows[1].pose.has_value() and not p.rows[1].exist_logodds
                  and not p.rows[1].leaf);
            CHECK(p.rows[2].pose.has_value() and not p.rows[2].exist_logodds and not p.rows[2].leaf);
        }
    }

    // ── 2. Backward compatibility: the 2026-10-13 four-column table reads as identity-only rows ──
    {
        std::stringstream ss("room,name,ap_x,ap_y\n"
                             "room_2,door_3,3.60,2.18\n"
                             "room_1,door_1,-0.58,4.61\n"
                             "room_1,door_9,3.7e13,5.2e12\n"     // a transform read mid-birth: not a place
                             "room_1,window_1,1.0,1.0\n"          // not a door
                             "room_1,door_4,abc,1.0\n");          // unparsable
        const auto p = parse_identity_table(ss);
        CHECK(not p.unkeyed_format);
        CHECK(p.rows.size() == 2);
        CHECK(p.malformed == 3);
        for (const auto& r : p.rows) CHECK(not r.pose.has_value());
        if (p.rows.size() == 2) CHECK(p.rows[1].name == "door_1" and near(p.rows[1].xy.x(), -0.58f));
    }
    // …and the pre-2026-10-13 un-keyed table is still refused whole.
    {
        std::stringstream ss("name,ap_x,ap_y\ndoor_1,1.0,2.0\n");
        const auto p = parse_identity_table(ss);
        CHECK(p.unkeyed_format and p.rows.empty());
    }
    // A comma-locale string must NOT parse as a number (from_chars is locale-free both ways).
    {
        std::stringstream ss("room,name,ap_x,ap_y\nroom_1,door_1,-0,58,4,61\n");
        const auto p = parse_identity_table(ss);
        CHECK(p.rows.empty() and p.malformed == 1);
    }

    // ── 3. The restore decision ──
    const float radius = 0.5f;
    std::vector<IdentityRow> table{
        full_row("room_1", "door_1", -0.54f, 4.61f),   // room_1's OTHER door — the one lost in tourP2
        full_row("room_1", "door_2",  0.59f, 2.54f),   // the doorway just crossed (carried across separately)
        full_row("room_2", "door_3",  3.60f, 2.18f),   // another room's door: never restored into room_1
    };
    IdentityRow old4; old4.room = "room_1"; old4.name = "door_5"; old4.xy = {2.0f, -3.0f};   // 4-column row
    table.push_back(old4);

    // 3a. Recalled room_1, the crossed door_2 already carried (live, ON ITS HOLE): restore door_1 only. The crossed door is
    //     recognised by PLACE — since 2026-10-29 a live NAME blocks nothing (names are opaque).
    {
        const std::vector<LiveDoor> live{{"door_2", Eigen::Vector2f(0.60f, 2.52f)}};
        const auto d = plan_restore(table, "room_1", true, live, radius);
        CHECK(d.restore.size() == 1);
        if (d.restore.size() == 1) CHECK(d.restore[0].name == "door_1");
        bool skipped_old = false, skipped_crossed = false;
        for (const auto& s : d.skipped)
        {
            skipped_old     = skipped_old     or (s.name == "door_5" and s.reason == SkipReason::NoPose);
            skipped_crossed = skipped_crossed or (s.name == "door_2" and s.reason == SkipReason::PlaceLive);
            CHECK(s.reason != SkipReason::NameLive);   // retired: never produced
        }
        CHECK(skipped_old);
        CHECK(skipped_crossed);
    }
    // 3b. A fresh room that shares the name (not a recall): nothing comes back.
    {
        const auto d = plan_restore(table, "room_1", false, {}, radius);
        CHECK(d.restore.empty());
        CHECK(d.not_recalled);
    }
    // 3c. Crossed door carried under ANOTHER name, or not resolvable by name at all: the place still matches it.
    {
        const std::vector<LiveDoor> live{{"door_7", Eigen::Vector2f(0.40f, 2.60f)}};
        const auto d = plan_restore(table, "room_1", true, live, radius);
        CHECK(d.restore.size() == 1 and d.restore[0].name == "door_1");
        bool by_place = false;
        for (const auto& s : d.skipped) by_place = by_place or (s.name == "door_2" and s.reason == SkipReason::PlaceLive);
        CHECK(by_place);
    }
    // 3d. (b) A remembered door whose NAME a live node wears but whose HOLE is free (2026-10-29: names are opaque) is
    //     restored — the allocator then gives it a FRESH name; the live node is not renamed, not touched.
    {
        const std::vector<LiveDoor> live{{"door_2", std::nullopt}};
        const auto d = plan_restore(table, "room_1", true, live, radius);
        CHECK(d.restore.size() == 2);
        bool has1 = false, has2 = false;
        for (const auto& r : d.restore) { has1 = has1 or r.name == "door_1"; has2 = has2 or r.name == "door_2"; }
        CHECK(has1 and has2);
        std::vector<std::string> graph{"door_2"};
        std::uint64_t next = 5;
        const auto nm = allocate_door_name("door_2", graph, next);   // the restore asks for its remembered name
        CHECK(nm == "door_5" and next == 6);                           // live ⇒ fresh counter name
        CHECK(graph.size() == 1 and graph[0] == "door_2");             // the live one keeps its name
        const auto nm1 = allocate_door_name("door_1", graph, next);   // remembered name free ⇒ taken
        CHECK(nm1 == "door_1" and next == 6);
    }
    // 3e. Same-hole test uses the remembered width: a live door 1.2 m away from a 0.95 m-wide doorway with r=0.5
    //     (radius + width = 1.45 m) is the same hole.
    {
        const std::vector<LiveDoor> live{{"door_8", Eigen::Vector2f(-0.54f + 1.2f, 4.61f)},
                                         {"door_9", Eigen::Vector2f(0.60f, 2.52f)}};   // door_2's hole, under another name
        const auto d = plan_restore(table, "room_1", true, live, radius);
        CHECK(d.restore.empty());
    }
    // 3f. Existence prior: remembered L when written, the stated default otherwise.
    {
        CHECK(near(restore_existence_prior(table[0], 1.0f, kNowMs), 4.0f, 1e-3f));   // seen just now: the memory itself
        IdentityRow r = table[0]; r.exist_logodds.reset();
        CHECK(near(restore_existence_prior(r, 1.0f, kNowMs), 1.0f));
        r.exist_logodds = std::numeric_limits<float>::quiet_NaN();
        CHECK(near(restore_existence_prior(r, 1.0f, kNowMs), 1.0f));
    }
    // 3g. Task D-C (rtD1i door_5): a remembered belief DECAYS toward the birth prior with the time since the door was last
    //     seen, at the leaf channel's change hazard h = 1/600 s (P = e^{-hΔt}·P_mem + (1 − e^{-hΔt})·P_birth). door_5 left
    //     at L = −4 and was restored 132 s later at −4, below the removal bound logit(0.12) = −1.99; now it enters above it.
    {
        IdentityRow r = table[0];
        r.exist_logodds = -4.0f;
        r.seen_wall_ms = kNowMs - 132'000;
        const float L132 = restore_existence_prior(r, 1.0f, kNowMs);
        const double s = std::exp(-132.0 / 600.0), pm = 1.0 / (1.0 + std::exp(4.0)), pb = 1.0 / (1.0 + std::exp(-1.0));
        const double p = s * pm + (1.0 - s) * pb;
        std::print("remembered L -4 seen 132 s ago -> prior L {:.3f} (closed form {:.3f}); removal bound {:.3f}\n",
                   L132, std::log(p / (1.0 - p)), std::log(0.12 / 0.88));
        CHECK(near(L132, static_cast<float>(std::log(p / (1.0 - p))), 1e-4f));
        CHECK(L132 > std::log(0.12f / 0.88f));
        // monotone in the age, between the memory (Δt = 0) and the birth prior (Δt → ∞)
        float prev = restore_existence_prior(r, 1.0f, kNowMs - 132'000);
        CHECK(near(prev, -4.0f, 1e-3f));
        for (const std::int64_t age_s : {10, 60, 300, 1200, 7200})
        {
            const float L = restore_existence_prior(r, 1.0f, kNowMs - 132'000 + age_s * 1000);
            CHECK(L > prev);
            prev = L;
        }
        CHECK(near(prev, 1.0f, 2e-3f));
        // a clock that went backwards (a stamp from the future) is age 0, never a negative age
        CHECK(near(restore_existence_prior(r, 1.0f, kNowMs - 200'000), -4.0f, 1e-3f));
        // no stamp (a 17-column row written before the column existed) ⇒ unknown age ⇒ the birth prior
        r.seen_wall_ms.reset();
        CHECK(near(restore_existence_prior(r, 1.0f, kNowMs), 1.0f));
        // a 17-column row parses (no seen_ms) and an 18-column one carries it; upsert carries the stamp with the belief
        std::stringstream old17;
        old17 << "room,name,ap_x,ap_y,yaw,width,height,depth,exist_logodds,hinge_x,hinge_y,open_x,open_y,hinge_llr,swing_llr,phi,phi_sigma\n"
              << "room_1,door_5,0.88,2.65,-3.2,1.0,2.1,0.1,-4,,,,,,,,\n";
        const auto p17 = parse_identity_table(old17);
        CHECK(p17.malformed == 0 and p17.rows.size() == 1);
        if (p17.rows.size() == 1)
            CHECK(p17.rows[0].exist_logodds.has_value() and not p17.rows[0].seen_wall_ms.has_value());
        std::vector<IdentityRow> t{full_row("room_1", "door_5", 0.88f, 2.65f)};
        auto w = full_row("", "door_5", 0.88f, 2.65f);
        w.exist_logodds = -4.0f; w.seen_wall_ms = kNowMs + 5000;
        upsert_row(t, "room_1", w, 1.45f, 0.02f);
        CHECK(t.size() == 1 and t[0].seen_wall_ms == kNowMs + 5000 and near(*t[0].exist_logodds, -4.0f));
        // Task D-C round 2: the 19th column, place_sd — round trip; an 18-column row parses without it; upsert mixes it;
        // a fresh entry strips it with the rest of the restore data
        std::vector<IdentityRow> rt{full_row("room_1", "door_2", 0.55f, 2.54f)};
        rt[0].place_sd = 0.037f;
        std::stringstream io;
        write_identity_table(io, rt);
        const auto back = parse_identity_table(io);
        CHECK(back.malformed == 0 and back.rows.size() == 1);
        if (back.rows.size() == 1) CHECK(back.rows[0].place_sd.has_value() and near(*back.rows[0].place_sd, 0.037f));
        std::stringstream old18;
        old18 << "room,name,ap_x,ap_y,yaw,width,height,depth,exist_logodds,hinge_x,hinge_y,open_x,open_y,hinge_llr,swing_llr,phi,phi_sigma,seen_ms\n"
              << "room_1,door_2,0.55,2.54,-3.14,1.0,2.1,0.1,1,,,,,,,,,1792632310409\n";
        const auto p18 = parse_identity_table(old18);
        CHECK(p18.malformed == 0 and p18.rows.size() == 1);
        if (p18.rows.size() == 1) CHECK(p18.rows[0].seen_wall_ms.has_value() and not p18.rows[0].place_sd.has_value());
        auto w2 = full_row("", "door_2", 0.55f, 2.54f);
        w2.place_sd = 0.137f;
        upsert_row(rt, "room_1", w2, 1.45f, 0.5f);
        CHECK(rt.size() == 1 and rt[0].place_sd.has_value() and near(*rt[0].place_sd, 0.087f));
        CHECK(strip_restorable(rt, "room_1") == 1 and not rt[0].place_sd.has_value());
    }

    // 3h. Task D-C review I-1: the row remembers the EVIDENCE (L − the band prior's current term); the restored instance
    //     starts with that term at 0 and the core adds it once — the full L would count the prior twice per round trip.
    {
        struct E { float L; float logodds() const { return L; } };
        struct I { E existence; float room_prior_term; };
        CHECK(near(remembered_logodds(I{{-1.5f}, -2.0f}), 0.5f));   // an off-band door: the prior's −2 is not remembered
        CHECK(near(remembered_logodds(I{{3.0f}, 0.0f}), 3.0f));     // on the band: unchanged
    }

    // ── 4. Stale rows: a FRESH entry of room_1 strips its restore data; a later recall restores only doors seen since ──
    {
        std::vector<IdentityRow> t{full_row("room_1", "door_1", -0.54f, 4.61f),    // an OLDER run's door_1
                                   full_row("room_1", "door_6", 2.00f, -3.00f),    // an older run's phantom
                                   full_row("room_2", "door_3", 3.60f, 2.18f)};
        CHECK(strip_restorable(t, "room_1") == 2);
        CHECK(not t[0].pose and not t[0].leaf and not t[0].exist_logodds and t[0].name == "door_1");   // name kept
        CHECK(not t[1].pose and t[2].pose.has_value());                                                // other rooms untouched
        // During this visit door_1 is seen and fitted: its row is rewritten with a pose (upsert by name).
        std::erase_if(t, [](const IdentityRow& r) { return r.room == "room_1" and r.name == "door_1"; });
        t.push_back(full_row("room_1", "door_1", -0.55f, 4.60f));
        const auto d = plan_restore(t, "room_1", true, {}, radius);
        CHECK(d.restore.size() == 1 and d.restore[0].name == "door_1");
        bool phantom_skipped = false;
        for (const auto& s : d.skipped) phantom_skipped = phantom_skipped or (s.name == "door_6" and s.reason == SkipReason::NoPose);
        CHECK(phantom_skipped);
        CHECK(strip_restorable(t, "room_9") == 0);
    }

    // ── 5. tourP4 sequence: fresh entry → birth by identity → fit → (ltsm evicts the room: doors deleted BEFORE the
    //      switch) → release → recall → restore. The pose row must be written when the evicted door is deleted, because
    //      release then sees no live instance. Same-hole = radius + width, as the agent uses.
    {
        const auto same_hole = [&](float w) { return radius + w; };
        std::vector<IdentityRow> t{full_row("room_1", "door_1", -0.43f, 4.61f), full_row("room_1", "door_2", 0.64f, 2.54f)};
        CHECK(strip_restorable(t, "room_1") == 2);                                   // fresh entry (first room)
        for (auto [n, x, y] : {std::tuple{"door_1", -0.55f, 4.61f}, std::tuple{"door_2", 0.72f, 2.45f}})
        {
            IdentityRow birth; birth.name = n; birth.xy = {x, y};
            CHECK(upsert_row(t, "room_1", birth, same_hole(0.f)));               // birth by identity: id-only
        }
        auto replay = t;                                                             // the bug: nothing written after this
        {
            const std::vector<IdentityRow> live_at_release{};                        // eviction emptied the instance map
            for (const auto& r : live_at_release) upsert_row(replay, "room_1", r, same_hole(r.pose->width));
            const auto d = plan_restore(replay, "room_1", true, {}, radius);
            CHECK(d.restore.empty());                                                // reproduces "no remembered pose"
        }
        // The fix: the externally deleted, fitted doors write their full rows at deletion.
        for (auto r : {full_row("", "door_1", -0.545f, 4.606f), full_row("", "door_2", 0.642f, 2.536f)})
            CHECK(upsert_row(t, "room_1", r, same_hole(r.pose->width)));
        CHECK(t.size() == 2);                                                        // replaced, not duplicated
        // The next room's fresh entry must not touch room_1's rows.
        CHECK(strip_restorable(t, "room_2") == 0);
        // Round-trip through the file as the agent does between visits.
        std::stringstream ss; write_identity_table(ss, t); const auto p = parse_identity_table(ss);
        const std::vector<LiveDoor> live{{"door_2", Eigen::Vector2f(0.64f, 2.54f)}};  // the carried door
        const auto d = plan_restore(p.rows, "room_1", true, live, radius);
        CHECK(d.restore.size() == 1 and d.restore[0].name == "door_1" and d.restore[0].pose.has_value());
    }

    // ── 6. tourP6: the restored instance was built at node insert, BEFORE its RT edge (place read as (0,0)), and the
    //      seeding was skipped. seed_restored_instance must state the pose and make it fitted with no fresh mask, so
    //      process_door_node runs run_inference and the pragmatic affordances (which need a fitted aperture) appear.
    {
        struct FakeModel { State s; const State& state() const { return s; } void set_state(const State& x) { s = x; } };
        struct FakeExist { float L = 0.f, mx = 4.f; void set_max(float m) { mx = m; } void set(float l) { L = std::clamp(l, -mx, mx); }
                           float logodds() const { return L; } };
        struct FakeInst
        {
            FakeModel model; FakeExist existence; bool existence_seeded = false; int matched_frames = 0;
            bool seed_wh_from_box = false;
            std::optional<std::pair<Eigen::Vector2f, Eigen::Vector2f>> leaf_side_hint;
            float phi_est = 0.f, phi_sigma = 0.f, hinge_llr = 0.f, swing_llr = 0.f;
            std::chrono::steady_clock::time_point phi_last_t{}; bool phi_from_command = true;
            float existence_entry_L = 1.f, existence_since_entry = 7.f; std::int64_t last_detection_wall_ms = 0;
            std::optional<float> remembered_place_sd;
        } inst;   // as the node-added path left it: (0,0), unseeded, 0 matched frames
        CHECK(not fitted_this_cycle(false, inst.matched_frames));           // the bug: never fitted ⇒ no approach
        const auto r = full_row("room_1", "door_1", -0.57f, 4.61f);
        seed_restored_instance(inst, r, restore_existence_prior(r, 1.0f, kNowMs), 4.0f, std::chrono::steady_clock::now());
        CHECK(near(inst.model.s.ap_cx, -0.57f) and near(inst.model.s.ap_cy, 4.61f) and near(inst.model.s.cx, -0.57f));
        CHECK(near(inst.model.s.ap_yaw, 3.10f) and near(inst.model.s.w, 0.95f) and near(inst.model.s.h, 2.13f));
        CHECK(inst.existence_seeded and near(inst.existence.L, 4.0f, 1e-3f));
        CHECK(near(inst.existence_entry_L, inst.existence.L) and inst.last_detection_wall_ms == kNowMs);   // Task D-C
        CHECK(inst.existence_since_entry == 0.f);   // round 2: the evidence since entry starts over at the restore
        CHECK(fitted_this_cycle(false, inst.matched_frames));               // fitted with no fresh mask
        CHECK(inst.seed_wh_from_box and inst.leaf_side_hint.has_value() and not inst.phi_from_command);
        // An identity-only row (no pose) still gets its place; size stays the instance's (template) one.
        IdentityRow id_only; id_only.room = "room_1"; id_only.name = "door_5"; id_only.xy = {2.f, -3.f};
        FakeInst j;
        seed_restored_instance(j, id_only, restore_existence_prior(id_only, 1.0f, kNowMs), 4.0f, std::chrono::steady_clock::now());
        CHECK(near(j.model.s.ap_cx, 2.f) and near(j.existence.L, 1.0f) and not j.leaf_side_hint.has_value());
    }

    // ── 7. tourP9 NAME DRIFT: a duplicate (door_4) on door_1's hole does not rename it; one row per hole ──
    {
        std::vector<IdentityRow> t{full_row("room_1", "door_1", -0.58f, 4.61f), full_row("room_1", "door_2", 0.56f, 2.54f)};
        auto dup = full_row("", "door_4", -0.54f, 4.61f);
        CHECK(upsert_row(t, "room_1", dup, radius + 0.95f, 1.0f));
        CHECK(t.size() == 2);
        CHECK(t[0].name == "door_1" and near(t[0].xy.x(), -0.54f));               // the oldest name, the new place
        // A table that already holds BOTH (door_1 oldest, then door_4 on the same hole) collapses to door_1 on the next write.
        std::vector<IdentityRow> u{full_row("room_1", "door_1", -0.58f, 4.61f), full_row("room_1", "door_4", -0.54f, 4.61f)};
        CHECK(upsert_row(u, "room_1", full_row("", "door_4", -0.55f, 4.61f), radius + 0.95f, 1.0f));
        CHECK(u.size() == 1 and u[0].name == "door_1");
        // Birth by identity (no pose) on a hole keeps the row's pose and name.
        IdentityRow birth; birth.name = "door_1"; birth.xy = {-0.56f, 4.61f};
        CHECK(upsert_row(u, "room_1", birth, radius, 0.73f));
        CHECK(u.size() == 1 and u[0].pose.has_value());
    }
    // ── 8. tourP8→P9 PLACE DRIFT: a dying door's degraded estimate barely moves the remembered place ──
    {
        IdentityRow id2; id2.room = "room_1"; id2.name = "door_2"; id2.xy = {0.56f, 2.54f};    // stripped, id-only
        std::vector<IdentityRow> t{id2};
        auto drifted = full_row("", "door_2", 0.35f, 2.54f);                                // fit at shutdown, P(exists) 0.02
        CHECK(upsert_row(t, "room_1", drifted, radius + 0.84f, 0.02f));
        CHECK(t.size() == 1 and std::abs(t[0].xy.x() - 0.556f) < 0.01f);
        // A credible door (P ≈ 1) moves it fully.
        CHECK(upsert_row(t, "room_1", full_row("", "door_2", 0.60f, 2.54f), radius + 0.84f, 0.98f));
        CHECK(std::abs(t[0].xy.x() - 0.599f) < 0.01f);
    }

    // ── 9. (a) OPAQUE NAMES FROM ONE MONOTONIC COUNTER: fresh names strictly increase across births, deaths and restores,
    //        a name is never handed to a second door, and the counter survives a save/load of the table ──
    {
        std::uint64_t next = 1;
        std::vector<std::string> live;
        std::vector<std::uint64_t> fresh;   // the index of every FRESH name issued, in order
        const auto issue = [&](std::string_view preferred)
        {
            const auto before = next;
            const auto nm = allocate_door_name(preferred, live, next);
            if (next != before) fresh.push_back(*door_name_index(nm));
            live.push_back(nm);
            return nm;
        };
        const auto kill = [&](const std::string& nm) { std::erase(live, nm); };
        const auto a = issue({}), b = issue({}), c = issue({});      // three births
        CHECK(a == "door_1" and b == "door_2" and c == "door_3");
        kill(a);                                                     // door_1 dies...
        const auto d = issue({});                                    // ...its number is NOT recycled (was: lowest free)
        CHECK(d == "door_4");
        const auto r1 = issue("door_1");                             // a restore of door_1 (free): its remembered name
        CHECK(r1 == "door_1");
        const auto r2 = issue("door_2");                             // a restore whose name is live: fresh
        CHECK(r2 == "door_5");
        kill(b); kill(c); kill(d);
        const auto e = issue({});
        CHECK(e == "door_6");
        // a live node of another writer / an earlier run with a higher number raises the counter past it
        live.push_back("door_40");
        CHECK(issue({}) == "door_41");
        bool increasing = true;
        for (std::size_t i = 1; i < fresh.size(); ++i) increasing = increasing and fresh[i] > fresh[i - 1];
        CHECK(increasing and fresh.size() == 7);   // 1,2,3,4,5,6 and 41 — the restore of door_1 issued nothing fresh
        // save/load: the counter line round-trips, an old reader-compatible row set is unchanged, and a fresh run's
        // first name continues from the counter even with every door forgotten (no row holds door_41).
        std::vector<IdentityRow> rows{full_row("room_1", "door_1", -0.54f, 4.61f)};
        std::stringstream ss;
        write_identity_table(ss, rows, next);
        CHECK(ss.str().find("\n#next_door,42\n") != std::string::npos);
        const auto p = parse_identity_table(ss);
        CHECK(p.rows.size() == 1 and p.malformed == 0 and p.next_door_index == 42);
        std::uint64_t next2 = next_door_index_floor(p);
        CHECK(next2 == 42);
        const std::vector<std::string> none;
        CHECK(allocate_door_name({}, none, next2) == "door_42");
        // a table from before the counter: the counter starts past every remembered name
        std::stringstream old(std::string(kIdentityHeader) + "\nroom_1,door_7,1.0,2.0\nroom_2,door_3,0.5,0.5\n");
        const auto po = parse_identity_table(old);
        CHECK(po.next_door_index == 0 and next_door_index_floor(po) == 8);
        // a stored counter BELOW a remembered name (hand-edited) cannot hand that name out
        std::stringstream low("room,name,ap_x,ap_y\nroom_1,door_9,1.0,2.0\n#next_door,3\n");
        CHECK(next_door_index_floor(parse_identity_table(low)) == 10);
        // garbage counter: counted malformed, rows intact
        std::stringstream bad("room,name,ap_x,ap_y\nroom_1,door_9,1.0,2.0\n#next_door,x\n");
        const auto pb = parse_identity_table(bad);
        CHECK(pb.rows.size() == 1 and pb.malformed == 1 and pb.next_door_index == 0);
        // names that are not door_<N≥1> are not numbers
        CHECK(not door_name_index("door_").has_value() and not door_name_index("door_0").has_value()
              and not door_name_index("door_3a").has_value() and not door_name_index("passage_3").has_value()
              and door_name_index("door_12") == std::optional<std::uint64_t>(12));
    }
    // ── 10. (c) PLACE RE-IDENTIFICATION UNCHANGED: a re-detected/restored hole keeps its remembered (oldest) row name by
    //         place, whatever name its live node got; a different hole is a different row ──
    {
        std::vector<IdentityRow> t{full_row("room_1", "door_1", -0.54f, 4.61f)};
        // the hole's node now wears a fresh name (door_9, since door_1 was live elsewhere): the row stays door_1
        CHECK(upsert_row(t, "room_1", full_row("", "door_9", -0.50f, 4.60f), radius + 0.95f, 0.9f));
        CHECK(t.size() == 1 and t[0].name == "door_1");
        // a door elsewhere is its own row
        CHECK(upsert_row(t, "room_1", full_row("", "door_10", 2.5f, 0.0f), radius + 0.95f, 0.9f));
        CHECK(t.size() == 2 and t[1].name == "door_10");
        // restore by place: a live node on door_1's hole (any name) blocks it; door_10's free hole is restored
        const std::vector<LiveDoor> live{{"door_9", Eigen::Vector2f(-0.52f, 4.60f)}};
        const auto d = plan_restore(t, "room_1", true, live, radius);
        CHECK(d.restore.size() == 1 and d.restore[0].name == "door_10");
        CHECK(d.skipped.size() == 1 and d.skipped[0].name == "door_1" and d.skipped[0].reason == SkipReason::PlaceLive);
    }

    // ── WARM START (2026-11-01): a START-UP recall announces the FIRST room too; without it the first room stays fresh ──
    {
        CHECK(plan_room_entry(true, true, false, "", "room_1", true).recalled);         // start-up recall: restore room_1's doors
        CHECK(not plan_room_entry(true, true, false, "", "room_1", false).recalled);    // today's rule: the first room is fresh
        CHECK(plan_room_entry(false, true, false, "room_2", "room_1").recalled);        // a crossing recall, as before
        CHECK(not plan_room_entry(true, false, false, "", "room_1", true).recalled);    // no announcement, nothing restored
    }

    if (fails == 0) std::print("test_door_identity_table: ALL PASS\n");
    return fails == 0 ? 0 : 1;
}
