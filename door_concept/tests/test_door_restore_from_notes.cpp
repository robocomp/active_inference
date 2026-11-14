// Door restore from LTSM notes (2026-11-13, door memory through ltsm, Task 4).
//  - the pure part: a door through a CSV row and through a note restores to the same belief (L, physical, ledger, aged phi);
//  - plan_restore on note rows: a live door at the place skips the note (reported, so the agent adopts it); late notes restore;
//  - the DSR part: read_notes on a standalone DSRGraph (unadopted, RT edge in the room frame, phi sentinel), mark_adopted.
// Build/run: see tests/CMakeLists.txt (target test_door_restore_from_notes).
#include "door_note_reader.h"
#include "core/place_table.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace
{
int failures = 0;
void check(bool ok, int line, const char* what) { if (not ok) { ++failures; std::printf("FAIL line %d: %s\n", line, what); } }
#define CHECK(x) check((x), __LINE__, #x)

using namespace rc::door;
constexpr std::int64_t kNow = 1'700'000'000'000;

IdentityRow csv_row(const char* name, float x, float y)
{
    IdentityRow r;
    r.room = "room_1";  r.name = name;
    r.xy = {x, y};
    r.pose = RestorePose{0.3f, 0.9f, 2.05f, 0.12f};
    r.exist_logodds = 2.2f;  r.exist_physical = 0.4f;  r.place_sd = 0.05f;
    r.seen_wall_ms = kNow - 600'000;  r.sighted_ms = kNow - 600'000;  r.visit = 3;  r.crossed = false;
    LeafMemory L;
    L.hinge_xy = r.xy + Eigen::Vector2f(std::cos(0.3f), std::sin(0.3f)) * 0.45f;
    L.open_dir = Eigen::Vector2f(-std::sin(0.3f), std::cos(0.3f));
    L.hinge_llr = 1.5f;  L.swing_llr = 2.0f;  L.phi = 0.8f;  L.phi_sigma = 0.3f;
    r.leaf = L;
    return r;
}
}

int main()
{
    const core::PlaceTable table(true);
    const auto belief_of = [&](const IdentityRow& r)
    { return table.restored_belief(core::PlaceTable::row_of(r, 0.05f, true), kNow); };

    {   // same door via CSV row and via note -> identical belief and aged phi
        const IdentityRow csv = csv_row("door_5", 1.5f, -2.0f);
        const memnote::NoteAttrs a = memnote::note_of(csv, 0.03f);
        const IdentityRow viaNote = memnote::row_of(a, "room_1", memnote::face_live_name("remembered_door_5"));
        CHECK(viaNote.name == "door_5");
        const auto b0 = belief_of(csv), b1 = belief_of(viaNote);
        CHECK(std::fabs(b0.L_evidence() - b1.L_evidence()) < 1e-5f);
        CHECK(b0.ledger.views().size() == b1.ledger.views().size());
        CHECK(csv.leaf.has_value() and viaNote.leaf.has_value());
        const auto p0 = restore_leaf_phi(csv.leaf->phi, csv.leaf->phi_sigma, csv.seen_wall_ms, kNow, 1.57f);
        const auto p1 = restore_leaf_phi(viaNote.leaf->phi, viaNote.leaf->phi_sigma, viaNote.seen_wall_ms, kNow, 1.57f);
        CHECK(std::fabs(p0.first - p1.first) < 1e-5f and std::fabs(p0.second - p1.second) < 1e-5f);
    }
    {   // a live door at the note's place -> the note is skipped as PlaceLive (the agent adopts it); the other restores
        const std::vector<memnote::NoteInput> notes = {
            {11, "remembered_door_5", memnote::note_of(csv_row("door_5", 1.5f, -2.0f), 0.03f)},
            {12, "remembered_door_6", memnote::note_of(csv_row("door_6", -3.0f, 1.0f), 0.03f)}};
        const auto rows = memnote::rows_of_notes(notes, "room_1");
        const std::vector<LiveDoor> live = {{"door_77", Eigen::Vector2f(1.52f, -2.01f)}};
        const auto d = plan_restore(rows, "room_1", true, live, 0.3f);
        CHECK(d.restore.size() == 1 and d.restore[0].name == "door_6");
        CHECK(d.skipped.size() == 1 and d.skipped[0].name == "door_5" and d.skipped[0].reason == SkipReason::PlaceLive);
    }
    {   // outline ready, 0 notes -> nothing; then 2 notes -> 2 restored
        const std::vector<LiveDoor> live;
        const auto none = plan_restore(memnote::rows_of_notes({}, "room_1"), "room_1", true, live, 0.3f);
        CHECK(none.restore.empty() and none.skipped.empty());
        const std::vector<memnote::NoteInput> notes = {
            {11, "remembered_door_5", memnote::note_of(csv_row("door_5", 1.5f, -2.0f), 0.03f)},
            {12, "remembered_door_6", memnote::note_of(csv_row("door_6", -3.0f, 1.0f), 0.03f)}};
        const auto two = plan_restore(memnote::rows_of_notes(notes, "room_1"), "room_1", true, live, 0.3f);
        CHECK(two.restore.size() == 2 and two.skipped.empty());
    }

    {   // C2: a note at a refuted place is dropped (and reported, so it is adopted); a note elsewhere in the room is kept
        const std::vector<memnote::NoteInput> notes = {
            {11, "remembered_door_5", memnote::note_of(csv_row("door_5", 1.5f, -2.0f), 0.03f)},
            {12, "remembered_door_6", memnote::note_of(csv_row("door_6", -3.0f, 1.0f), 0.03f)}};
        const auto rows = memnote::rows_of_notes(notes, "room_1");
        const std::vector<Eigen::Vector2f> refuted = {Eigen::Vector2f(1.6f, -2.2f)};   // a removed PlaceRow near door_5's place
        const auto sp = drop_refuted_places(rows, refuted, 0.3f);
        CHECK(sp.dropped.size() == 1 and sp.dropped[0] == "door_5");
        CHECK(sp.kept.size() == 1 and sp.kept[0].name == "door_6");
        // the gate is plan_restore's own expression: a refuted place just outside radius + width keeps the note
        const float gate = restore_same_hole_m(0.3f, 0.9f);
        const std::vector<Eigen::Vector2f> far = {Eigen::Vector2f(1.5f + gate + 0.01f, -2.0f)};
        CHECK(drop_refuted_places(rows, far, 0.3f).dropped.empty());
        const std::vector<Eigen::Vector2f> near = {Eigen::Vector2f(1.5f + gate - 0.01f, -2.0f)};
        CHECK(drop_refuted_places(rows, near, 0.3f).dropped.size() == 1);
        // what the kept rows then do in plan_restore: only door_6 is restored
        const auto d = plan_restore(sp.kept, "room_1", true, std::vector<LiveDoor>{}, 0.3f);
        CHECK(d.restore.size() == 1 and d.restore[0].name == "door_6");
    }
    {   // I2b: only door_<digits> asks for its own name back; I1: the gate
        CHECK(memnote::is_plain_door_name("door_5") and memnote::is_plain_door_name("door_123"));
        CHECK(not memnote::is_plain_door_name("r2_door_5_h1") and not memnote::is_plain_door_name("door_5_h1")
              and not memnote::is_plain_door_name("door_5_entry") and not memnote::is_plain_door_name("door_")
              and not memnote::is_plain_door_name(""));
        std::uint64_t next = 40;
        const std::vector<std::string> graph = {"door_5"};
        CHECK(allocate_door_name("", graph, next) == "door_40");   // the empty preferred name -> a fresh counter name
        CHECK(memnote::restore_gate_open(7, 7) and not memnote::restore_gate_open(0, 7) and not memnote::restore_gate_open(6, 7)
              and not memnote::restore_gate_open(0, 0));
    }

    // ── the DSR part ──
    DSR::DSRGraph G("door_restore_notes_test", 994, std::string(SHADOW_JSON), true, static_cast<int8_t>(86), DSR::Queue);
    DSR::Node room = DSR::Node::create<object_node_type>("room_t");
    const auto room_id = G.insert_node(room).value();
    const auto make_note = [&](const char* name, std::uint64_t parent, bool with_rt, float phi, bool adopted)
    {
        DSR::Node n = DSR::Node::create<metaconcept_node_type>(name);
        G.add_or_modify_attrib_local<object_subtype_att>(n, std::string("remembered_door"));
        G.add_or_modify_attrib_local<parent_att>(n, parent);
        G.add_or_modify_attrib_local<width_m_att>(n, 0.9f);
        G.add_or_modify_attrib_local<height_m_att>(n, 2.05f);
        G.add_or_modify_attrib_local<depth_m_att>(n, 0.12f);
        G.add_or_modify_attrib_local<door_exist_evidence_att>(n, 2.2f);
        G.add_or_modify_attrib_local<door_phi_rad_att>(n, phi);
        G.add_or_modify_attrib_local<door_phi_sd_att>(n, 0.3f);
        G.add_or_modify_attrib_local<door_leaf_into_room_prob_att>(n, 0.9f);
        G.add_or_modify_attrib_local<door_leaf_hinge_plus_prob_att>(n, 0.8f);
        G.add_or_modify_attrib_local<door_view_ledger_att>(n, std::string(""));
        if (adopted) G.add_or_modify_attrib_local<door_adopted_att>(n, true);
        const auto id = G.insert_node(n).value();
        if (with_rt)
        {
            auto e = DSR::Edge::create<RT_edge_type>(room_id, id);
            G.add_or_modify_attrib_local<rt_translation_att>(e, std::vector<float>{1.5f, -2.0f, 0.0f});
            G.add_or_modify_attrib_local<rt_rotation_euler_xyz_att>(e, std::vector<float>{0.0f, 0.0f, 0.3f});
            G.insert_or_assign_edge(e);
        }
        return id;
    };
    const auto n1 = make_note("remembered_door_5", room_id, true, 0.8f, false);
    make_note("remembered_door_6", room_id, true, -1.0f, true);     // adopted: skipped
    make_note("remembered_door_7", room_id, false, 0.8f, false);    // no RT edge yet: not readable, not adopted
    make_note("remembered_door_8", room_id + 999, true, 0.8f, false);   // another room's note
    const auto notes = memnote::read_notes(G, room_id);
    CHECK(notes.size() == 1);
    if (notes.size() == 1)
    {
        const auto& a = notes[0].attrs;
        CHECK(notes[0].id == n1 and notes[0].name == "remembered_door_5");
        CHECK(std::fabs(a.xy.x() - 1.5f) < 1e-6f and std::fabs(a.xy.y() + 2.0f) < 1e-6f and std::fabs(a.yaw - 0.3f) < 1e-6f);
        CHECK(a.width == 0.9f and a.height == 2.05f and a.depth == 0.12f);
        CHECK(a.exist_evidence.value_or(0) == 2.2f and a.phi.value_or(0) == 0.8f and a.ledger.has_value());
        CHECK(not a.exist_physical.has_value() and not a.crossed.has_value());   // absent stays absent
        CHECK(memnote::rows_of_notes(notes, "room_t")[0].leaf.has_value());
        CHECK(memnote::mark_adopted(G, n1));
        CHECK(memnote::read_notes(G, room_id).empty());
        CHECK(G.get_node(n1).has_value());   // never deleted by door_concept
    }
    {   // C1 reader side: Eigen's euler split of a negative yaw is (pi, -pi, yaw+pi); the decoded yaw is still the composed yaw
        for (const float yaw : {-1.0f, -2.8f, 0.7f})
        {
            DSR::Node rn = DSR::Node::create<object_node_type>("room_neg");
            const auto rid = G.insert_node(rn).value();
            DSR::Node n = DSR::Node::create<metaconcept_node_type>("remembered_door_9");
            G.add_or_modify_attrib_local<object_subtype_att>(n, std::string("remembered_door"));
            G.add_or_modify_attrib_local<parent_att>(n, rid);
            G.add_or_modify_attrib_local<width_m_att>(n, 0.9f);
            G.add_or_modify_attrib_local<height_m_att>(n, 2.05f);
            G.add_or_modify_attrib_local<depth_m_att>(n, 0.12f);
            const auto id = G.insert_node(n).value();
            auto e = DSR::Edge::create<RT_edge_type>(rid, id);
            const Eigen::Matrix3d R = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
            const Eigen::Vector3d rpy = R.eulerAngles(2, 1, 0).reverse();   // what the old writer stored
            G.add_or_modify_attrib_local<rt_translation_att>(e, std::vector<float>{1.0f, 2.0f, 0.0f});
            G.add_or_modify_attrib_local<rt_rotation_euler_xyz_att>(e, std::vector<float>{(float)rpy.x(), (float)rpy.y(), (float)rpy.z()});
            G.insert_or_assign_edge(e);
            const auto got = memnote::read_notes(G, rid);
            CHECK(got.size() == 1);
            if (got.size() == 1) CHECK(std::fabs(std::remainder(got[0].attrs.yaw - yaw, 2.0f * 3.14159265f)) < 1e-5f);
            G.delete_node(id);
            G.delete_node(rid);
        }
    }
    std::printf(failures == 0 ? "OK\n" : "%d FAILURES\n", failures);
    return failures == 0 ? 0 : 1;
}
