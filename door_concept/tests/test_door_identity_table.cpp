// Offline harness for door_identity_table.h: the room-keyed identity table (etc/door_identities.csv) round trip,
// including the old 4-column rows, and the RESTORE decision — which remembered doors come back when a room is
// re-entered from memory (tourP2: room_1 re-entered, door_1 at (-0.54,4.61) was gone until re-seen).
// Runs under the agent's locale (es_ES, comma decimals) on purpose: CLAUDE.md, "reproduce it before trusting a test".
// Build + run:
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_door_identity_table.cpp -o /tmp/test_door_identity_table && /tmp/test_door_identity_table
#include <clocale>
#include <cmath>
#include <print>
#include <sstream>
#include <string>
#include <vector>
#include "../src/door_identity_table.h"

using namespace rc::door;
static int fails = 0;
#define CHECK(c) do { if (not (c)) { ++fails; std::print("FAIL {}:{}  {}\n", __FILE__, __LINE__, #c); } } while (0)
static bool near(float a, float b, float tol = 1e-5f) { return std::abs(a - b) <= tol; }

static IdentityRow full_row(std::string room, std::string name, float x, float y)
{
    IdentityRow r;
    r.room = std::move(room); r.name = std::move(name); r.xy = {x, y};
    r.pose = RestorePose{3.10f, 0.95f, 2.13f, 0.05f};
    r.exist_logodds = 4.0f;
    r.leaf = LeafMemory{{-0.09f, 4.61f}, {0.0f, -1.0f}, 19.7f, 9.3f, 0.031f, 0.12f};
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

    // 3a. Recalled room_1, the crossed door_2 already carried (live, by name): restore door_1 only.
    {
        const std::vector<LiveDoor> live{{"door_2", Eigen::Vector2f(0.60f, 2.52f)}};
        const auto d = plan_restore(table, "room_1", true, live, radius);
        CHECK(d.restore.size() == 1);
        if (d.restore.size() == 1) CHECK(d.restore[0].name == "door_1");
        bool skipped_old = false, skipped_crossed = false;
        for (const auto& s : d.skipped)
        {
            skipped_old     = skipped_old     or (s.name == "door_5" and s.reason == SkipReason::NoPose);
            skipped_crossed = skipped_crossed or (s.name == "door_2" and s.reason == SkipReason::NameLive);
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
    // 3d. The crossed door still HELD in the room left behind (late carry): its name is live in the graph even though
    //     its place in the new room cannot be resolved — skipped by name, no duplicate.
    {
        const std::vector<LiveDoor> live{{"door_2", std::nullopt}};
        const auto d = plan_restore(table, "room_1", true, live, radius);
        CHECK(d.restore.size() == 1 and d.restore[0].name == "door_1");
    }
    // 3e. Same-hole test uses the remembered width: a live door 1.2 m away from a 0.95 m-wide doorway with r=0.5
    //     (radius + width = 1.45 m) is the same hole.
    {
        const std::vector<LiveDoor> live{{"door_8", Eigen::Vector2f(-0.54f + 1.2f, 4.61f)}, {"door_2", std::nullopt}};
        const auto d = plan_restore(table, "room_1", true, live, radius);
        CHECK(d.restore.empty());
    }
    // 3f. Existence prior: remembered L when written, the stated default otherwise.
    {
        CHECK(near(restore_existence_prior(table[0], 1.0f), 4.0f));
        IdentityRow r = table[0]; r.exist_logodds.reset();
        CHECK(near(restore_existence_prior(r, 1.0f), 1.0f));
        r.exist_logodds = std::numeric_limits<float>::quiet_NaN();
        CHECK(near(restore_existence_prior(r, 1.0f), 1.0f));
    }

    if (fails == 0) std::print("test_door_identity_table: ALL PASS\n");
    return fails == 0 ? 0 : 1;
}
