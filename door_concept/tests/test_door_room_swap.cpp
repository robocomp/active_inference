// Offline harness for the room-recognition swap in door_identity_table.h: a LATE recall (ltsm's recall_<K>, recall_late)
// for the room being entered, right after a PROVISIONAL room of another name was left, is a recalled entry (restore of
// K's doors), never "entered FRESH", and the provisional name's identity rows are DROPPED (room names are never reused).
// Build + run:
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_door_room_swap.cpp -o /tmp/test_door_room_swap && /tmp/test_door_room_swap
#include <clocale>
#include <cmath>
#include <print>
#include <set>
#include <unordered_map>
#include <string>
#include <vector>
#include "../src/door_identity_table.h"
#include "../src/door_rt_publish.h"

using namespace rc::door;
static int fails = 0;
static const std::vector<std::string> O4{"room_4"}, NONE{};
#define CHECK(c) do { if (not (c)) { ++fails; std::print("FAIL {}:{}  {}\n", __FILE__, __LINE__, #c); } } while (0)

static IdentityRow row(std::string room, std::string name, float x, float y, bool full)
{
    IdentityRow r;  r.room = std::move(room); r.name = std::move(name); r.xy = {x, y};
    if (full) { r.pose = RestorePose{3.1f, 0.95f, 2.13f, 0.05f}; r.exist_logodds = 3.0f; }
    return r;
}

int main()
{
    std::setlocale(LC_ALL, "");
    // room_3 remembered (full rows); provisional room_4 earned two doors while it was believed new.
    std::vector<IdentityRow> rows{row("room_3", "door_7", 1, 2, true), row("room_4", "door_9", 0.5f, 1, true),
                                  row("room_4", "door_10", -2, 3, false), row("room_1", "door_1", 4, 4, true)};

    // swap: room_4 left, room_3 entered, recall_room_3 announced LATE
    const auto p = plan_room_entry(false, true, true, "room_4", "room_3");
    std::print("door_concept: [restore] '{}' {}\n", "room_3", p.recalled ? "recalled" : "entered FRESH");
    CHECK(p.recalled);
    CHECK(p.drop_left);
    const auto dropped = drop_room(rows, "room_4");
    CHECK(dropped.size() == 2);
    for (const auto& r : rows) CHECK(r.room != "room_4");
    CHECK(rows.size() == 2);                                  // room_3 and room_1 untouched
    CHECK(rows[0].pose.has_value());

    // an ordinary (early) recall never drops what it leaves behind
    CHECK(not plan_room_entry(false, true, false, "room_2", "room_3").drop_left);
    CHECK(plan_room_entry(false, true, false, "room_2", "room_3").recalled);
    // no recall: FRESH, nothing dropped, even if "late" were stale
    const auto f = plan_room_entry(false, false, true, "room_4", "room_5");
    CHECK(not f.recalled and not f.drop_left);
    // first resolution is never a recall, no left room
    CHECK(not plan_room_entry(true, true, true, "", "room_1").recalled);
    // Undo (turn-around): back to the provisional name, announced by nobody: FRESH, and the dropped rows stay dropped
    const auto u = plan_room_entry(false, false, false, "room_3", "room_4");
    CHECK(not u.recalled and not u.drop_left);
    CHECK(drop_room(rows, "room_4").empty());
    // same-name late recall (no swap) drops nothing
    CHECK(not plan_room_entry(false, true, true, "room_3", "room_3").drop_left);

    // C1.4: an announced room restores its remembered rows, the crossed door's face included, and gets no mirror
    {
        std::vector<IdentityRow> rr{row("room_1", "door_1", 2, 0, true), row("room_1", "door_2", -3, 1, true)};   // door_1 = crossed face
        CHECK(plan_room_entry(false, true, false, "room_2", "room_1").recalled);
        CHECK(not entry_mirror_wanted(true, true));            // announced proto room: no mirror
        const auto d = plan_restore(rr, "room_1", true, {}, 0.3f);
        CHECK(d.restore.size() == 2 and d.restore[0].name == "door_1");
        // a face already live at its place is not duplicated (the recreated/carried crossed door)
        std::vector<LiveDoor> live{LiveDoor{"door_9", Eigen::Vector2f(2.05f, 0.f)}};
        const auto d2 = plan_restore(rr, "room_1", true, live, 0.3f);
        CHECK(d2.restore.size() == 1 and d2.restore[0].name == "door_2");
    }
    // ORDER (a): recall live at the room's first sight, retired BEFORE the switch (and the name latch cleared): still announced
    {
        AnnouncedRooms ar;
        ar.observe(7, true);
        ar.observe(7, false);                                  // recall retired: later sightings do not change the verdict
        CHECK(ar.announced(7) and not entry_mirror_wanted(true, ar.announced(7)));
        // (b) a stale announcement of another room (id 5, never entered) does not touch an unannounced same-name room (id 9)
        ar.observe(5, true);
        ar.observe(9, false);
        CHECK(not ar.announced(9) and entry_mirror_wanted(true, ar.announced(9)));
        ar.prune([](std::uint64_t id) { return id != 5; });     // room 5 gone
        CHECK(not ar.seen(5) and ar.seen(9) and ar.size() == 2);
    }
    CHECK(entry_mirror_wanted(true, false));                   // fresh_room_still_mirrors
    CHECK(not entry_mirror_wanted(false, false));              // not proto: never

    // C2.8 accepted_proposal_restores_and_merges: P (room 7, "room_4") renamed to K "room_3" in place; K has remembered doors,
    // one at the place of a live door of P (the same door: merged, not duplicated), one unseen (restored); the crossed DOOR_2
    // face (live, no remembered face in K) stays a live door of K.
    {
        std::vector<ProposalView> props{{"proposal_room_3", 7, "accepted"}};
        const auto a = plan_adoption(props, 7, "room_3", O4, false);
        CHECK(a.restore and a.drop_old);
        CHECK(not plan_adoption(props, 7, "room_3", O4, true).restore);              // once
        CHECK(not plan_adoption(props, 8, "room_3", O4, false).restore);             // another room's id
        std::vector<IdentityRow> t{row("room_3", "door_7", 1, 2, true), row("room_3", "door_8", -3, 0, true),
                                   row("room_4", "door_9", 0.5f, 1, true)};
        drop_room(t, "room_4");
        CHECK(t.size() == 2);
        std::vector<LiveDoor> live{LiveDoor{"door_1", Eigen::Vector2f(1.1f, 2.05f)},      // at door_7's place
                                   LiveDoor{"door_2", Eigen::Vector2f(5.f, 5.f)}};        // the crossed face, no row in K
        const auto d = plan_restore(t, "room_3", true, live, 0.3f);
        CHECK(d.restore.size() == 1 and d.restore[0].name == "door_8");
        CHECK(d.skipped.size() == 1 and d.skipped[0].name == "door_7" and d.skipped[0].reason == SkipReason::PlaceLive);
    }
    // C2.8 rejected_proposal_keeps_fresh (and pending too): no restore, no drop
    {
        for (const char* st : {"rejected", "pending"})
        {
            std::vector<ProposalView> props{{"proposal_room_3", 7, st}};
            const auto a = plan_adoption(props, 7, "room_3", NONE, false);
            CHECK(not a.restore and not a.drop_old);
        }
        // the room kept P's name: the proposal for K does not match it
        std::vector<ProposalView> props{{"proposal_room_3", 7, "accepted"}};
        CHECK(not plan_adoption(props, 7, "room_4", NONE, false).restore);
    }
    // C2.8 rename_is_not_a_room_change: same node id under a new name is Same; only a different id releases
    CHECK(room_follow_kind(7, 7) == FollowKind::Same);
    CHECK(room_follow_kind(7, 9) == FollowKind::Change);
    CHECK(room_follow_kind(0, 9) == FollowKind::First);
    CHECK(room_follow_kind(7, std::nullopt) == FollowKind::NoInformation);

    // C2.8 reseat: a door hung from P at a known P-frame pose; T(P<-K) = shift (2,-1), 90 deg. After accept its K-frame pose is
    // T(K<-P).old, and the inverse round-trips.
    {
        const Rigid2 P_from_K{2.f, -1.f, 1.5707963f};
        const auto K_from_P = rigid_inverse(P_from_K);
        const PublishedPose old{3.f, 1.f, 0.3f};                     // in P
        const auto nw = rigid_apply(K_from_P, old);
        // K point (a,b) = R^T (p - t): p-t = (1,2) -> (2,-1)
        CHECK(std::abs(nw.cx - 2.f) < 1e-4f and std::abs(nw.cy + 1.f) < 1e-4f);
        CHECK(std::abs(nw.yaw - (0.3f - 1.5707963f)) < 1e-4f);
        const auto back = rigid_apply(P_from_K, nw);
        CHECK(std::abs(back.cx - old.cx) < 1e-4f and std::abs(back.cy - old.cy) < 1e-4f and std::abs(back.yaw - old.yaw) < 1e-4f);
    }
    // C2.8 review fixes: the reseat filter walks door -> wall -> floor -> room; no reseat unless the rename was observed
    {
        std::unordered_map<std::uint64_t, std::uint64_t> par{{10, 2}, {2, 3}, {3, 7}, {11, 7}, {12, 99}, {99, 0}};   // 10: door->wall->floor->room 7
        const auto parent_of = [&](std::uint64_t x) { const auto it = par.find(x); return it == par.end() ? 0ull : it->second; };
        CHECK(descends_from(10, 7, parent_of));                  // wall-hung door via the floor
        CHECK(descends_from(11, 7, parent_of));                  // hung directly from the room
        CHECK(not descends_from(12, 7, parent_of));              // another room's door
        std::vector<ProposalView> props{{"proposal_room_3", 7, "accepted"}};
        CHECK(plan_adoption(props, 7, "room_3", O4, false).reseat);
        const auto same = plan_adoption(props, 7, "room_3", NONE, false);
        CHECK(same.restore and not same.reseat);                 // accepted, previous == name: beliefs already in K's frame
        // ordering: the merge by place reads the RE-SEATED live position
        const Rigid2 K_from_P = rigid_inverse(Rigid2{2.f, -1.f, 1.5707963f});
        std::vector<IdentityRow> t{row("room_3", "door_7", 2, -1, true)};
        const auto moved = rigid_apply(K_from_P, PublishedPose{3.f, 1.f, 0.f});          // live door, P-frame (3,1) -> K (2,-1)
        CHECK(plan_restore(t, "room_3", true, std::vector<LiveDoor>{LiveDoor{"d", Eigen::Vector2f(moved.cx, moved.cy)}}, 0.3f).restore.empty());
        CHECK(plan_restore(t, "room_3", true, std::vector<LiveDoor>{LiveDoor{"d", Eigen::Vector2f(3.f, 1.f)}}, 0.3f).restore.size() == 1);   // un-reseated would duplicate
    }
    // C2.8 round 2: the reseat keys on the names this agent held beliefs under, not on which of rename/accepted came first
    {
        struct Sim
        {   RoomNameHistory h; std::set<std::pair<std::uint64_t, std::string>> handled; int reseats = 0;
            void cycle(std::uint64_t id, const std::string& name, const char* state)
            {
                h.observe(id, name);
                if (handled.contains({id, name})) return;
                std::vector<ProposalView> pv{{"proposal_room_3", id, state}};
                const auto o = h.others(id, name);
                const auto a = plan_adoption(pv, id, name, o, false);
                if (not a.restore) return;
                handled.insert({id, name});
                if (a.reseat) ++reseats;
            } };
        { Sim s; s.cycle(7, "room_4", "pending"); s.cycle(7, "room_3", "pending"); s.cycle(7, "room_3", "accepted");
          s.cycle(7, "room_3", "accepted"); CHECK(s.reseats == 1); }                       // (a) rename a cycle before accepted
        { Sim s; s.cycle(7, "room_4", "pending"); s.cycle(7, "room_3", "accepted"); CHECK(s.reseats == 1); }   // (b) same cycle
        { Sim s; s.cycle(7, "room_3", "accepted"); s.cycle(7, "room_3", "accepted"); CHECK(s.reseats == 0); }  // (c) first sight as K
        { Sim s; s.cycle(7, "room_4", "pending");
          for (int i = 0; i < 10; ++i) s.cycle(7, "room_3", "accepted"); CHECK(s.reseats == 1); }              // (d) many cycles
        // descends_from stops at a cycle
        std::unordered_map<std::uint64_t, std::uint64_t> par{{1, 2}, {2, 1}};
        CHECK(not descends_from(1, 9, [&](std::uint64_t x) { return par.at(x); }));
    }
    // Task 19 (recog_proto6): T(K<-P) is read from the ROBOT across the frame switch, not from ltsm's proposal.
    {
        const auto near = [](const Rigid2& a, const Rigid2& b, float tol)
        { return std::hypot(a.tx - b.tx, a.ty - b.ty) < tol and std::abs(std::remainder(a.yaw - b.yaw, 6.2831853f)) < tol; };
        const auto near_pose = [](const PublishedPose& a, const PublishedPose& b, float tol)
        { return std::hypot(a.cx - b.cx, a.cy - b.cy) < tol and std::abs(std::remainder(a.yaw - b.yaw, 6.2831853f)) < tol; };
        // SE(2) power: Δ^1 = Δ, Δ^0 = I, Δ^2 = Δ∘Δ
        const Rigid2 D{0.03f, 0.04f, 0.05f};
        CHECK(near(rigid_power(D, 1.f), D, 1e-5f));
        CHECK(near(rigid_power(D, 0.f), Rigid2{}, 1e-6f));
        CHECK(near(rigid_power(D, 2.f), rigid_compose(D, D), 1e-5f));

        // The live geometry. TRUE frame change: room_4 (P) is room_2 (K) turned +90 deg about DOOR_2's hole — P's entry
        // mirror door_43 at (0,0.01) yaw 3.138 lands on room_2's door_20 hole (2.95,1.55). ltsm's proposal was the 180-deg
        // twin: the true one followed by a half turn about (0.21,-0.19) in K (room_concept's fit then moved the robot 179 deg).
        const Rigid2 K_from_P_true{2.95f, 1.55f, 1.5707963f};
        const Rigid2 half_turn{0.42f, -0.38f, 3.1415927f};                        // 180 deg about (0.21,-0.19)
        const Rigid2 P_from_K_proposed = rigid_inverse(rigid_compose(half_turn, K_from_P_true));
        const PublishedPose door_43_P{0.f, 0.01f, 3.138f};
        const auto by_proposal = rigid_apply(rigid_inverse(P_from_K_proposed), door_43_P);
        CHECK(std::abs(by_proposal.cx + 2.53f) < 0.02f and std::abs(by_proposal.cy + 1.93f) < 0.02f);   // the logged (-2.53,-1.93)
        CHECK(std::hypot(by_proposal.cx - 2.95f, by_proposal.cy - 1.55f) > 6.4f);                       // 6.5 m off the hole

        // The robot drives a constant body twist (0.3 m/s forward, 0.4 rad/s) through the switch; its K pose at t ms.
        const Rigid2 robot_K0{-0.4f, 2.6f, -1.2f}, twist_per_s{0.f, 0.3f, 0.4f};
        const auto robot_K = [&](std::uint64_t t) { return rigid_compose(robot_K0, rigid_power(twist_per_s, 1e-3f * float(t))); };
        const auto robot_P = [&](std::uint64_t t) { return rigid_compose(rigid_inverse(K_from_P_true), robot_K(t)); };
        const auto fixP = [&](std::uint64_t t) { return std::optional<RobotFix>(RobotFix{robot_P(t), t}); };
        const auto fixK = [&](std::uint64_t t) { return std::optional<RobotFix>(RobotFix{robot_K(t), t}); };

        // (a) the rename cycle still reads P's LAST block (published before the adopting cycle): hold, then K's first block
        {
            AdoptionFramePairing pr;
            pr.observe(7, "room_4", fixP(1000)); pr.observe(7, "room_4", fixP(1100));
            pr.observe(7, "room_2", fixP(1150));
            CHECK(not pr.K_from_P(fixP(1150)).has_value() and pr.has_previous());     // may still be P's: hold
            pr.observe(7, "room_2", fixK(1250));
            const auto T = pr.K_from_P(fixK(1250));
            CHECK(T.has_value() and near(*T, K_from_P_true, 2e-3f));
            if (T.has_value())
            {
                const auto reseated = rigid_apply(*T, door_43_P);
                CHECK(near_pose(reseated, rigid_apply(K_from_P_true, door_43_P), 0.01f));   // = T(K<-P)·its P-frame pose
                CHECK(std::hypot(reseated.cx - 2.95f, reseated.cy - 1.55f) < 0.02f);       // at DOOR_2's hole (door_20)
            }
            // applied ONCE: after the re-seat the pairing is rebased; later K blocks give no second frame change, so a door
            // whose numbers are already K's is never moved again
            pr.rebase("room_2");
            pr.observe(7, "room_2", fixK(1350));
            CHECK(not pr.K_from_P(fixK(1350)).has_value());
        }
        // (b) the rename cycle already reads a K block: it is still not trusted (it might have been P's); the next one is
        {
            AdoptionFramePairing pr;
            pr.observe(7, "room_4", fixP(1000)); pr.observe(7, "room_4", fixP(1100));
            pr.observe(7, "room_2", fixK(1200));
            CHECK(not pr.K_from_P(fixK(1200)).has_value());
            pr.observe(7, "room_2", fixK(1300));
            const auto T = pr.K_from_P(fixK(1300));
            CHECK(T.has_value() and near(*T, K_from_P_true, 2e-3f));
        }
        // (c) the same block seen twice is not "newer": no answer until a new stamp
        {
            AdoptionFramePairing pr;
            pr.observe(7, "room_4", fixP(1000));
            pr.observe(7, "room_2", fixP(1000));
            CHECK(not pr.K_from_P(fixP(1000)).has_value());
            pr.observe(7, "room_2", fixK(1100));
            // one P block only: zero-velocity carry, off by 100 ms of motion (0.04 rad at ~3.5 m of lever + 3 cm forward)
            const auto T = pr.K_from_P(fixK(1100));
            CHECK(T.has_value() and std::hypot(T->tx - K_from_P_true.tx, T->ty - K_from_P_true.ty) < 0.2f
                  and std::abs(T->yaw - K_from_P_true.yaw) < 0.05f);
        }
        // (d) nothing read under P (no robot edge then): the pairing cannot answer and says so (the caller falls back)
        {
            AdoptionFramePairing pr;
            pr.observe(7, "room_4", std::nullopt);
            pr.observe(7, "room_2", fixK(1200)); pr.observe(7, "room_2", fixK(1300));
            CHECK(not pr.has_previous() and not pr.K_from_P(fixK(1300)).has_value());
        }
        // (e) another room id restarts the pairing (a crossing is not an adoption)
        {
            AdoptionFramePairing pr;
            pr.observe(7, "room_4", fixP(1000));
            pr.observe(9, "room_5", fixK(1100));
            CHECK(not pr.K_from_P(fixK(1100)).has_value());
        }
    }
    // Task 20: the re-seat reads the ACCEPTED T(P<-K) room_concept wrote on the proposal edge before "accepted"; the
    // robot-paired T is a cross-check only, and nothing waits for K's first robot block.
    {
        const Rigid2 K_from_P_true{2.95f, 1.55f, 1.5707963f};                     // live: room_4 (P) -> room_2 (K)
        const Rigid2 half_turn{0.42f, -0.38f, 3.1415927f};                        // 180 deg about (0.21,-0.19) in K
        const Rigid2 P_from_K_prior = rigid_inverse(rigid_compose(half_turn, K_from_P_true));   // ltsm's (wrong) prior
        const Rigid2 P_from_K_accepted = rigid_inverse(K_from_P_true);           // room_concept's fit, written on the edge
        const PublishedPose door_43_P{-0.00f, 0.01f, 3.138f};                     // the entry mirror in P
        // the verdict is read from the NODE (one DDS sample with the state); the edge may still hold the prior (another topic,
        // unordered): it is not an input. Node absent / short ⇒ malformed, nothing.
        {
            const std::vector<float> node_tr{P_from_K_accepted.tx, P_from_K_accepted.ty, 0.f};
            const std::vector<float> node_rot{0.f, 0.f, P_from_K_accepted.yaw};
            const auto fromnode = node_P_from_K(&node_tr, &node_rot);
            CHECK(fromnode.has_value() and std::hypot(fromnode->tx - P_from_K_accepted.tx, fromnode->ty - P_from_K_accepted.ty) < 1e-6f);
            const auto T_used = accepted_K_from_P("accepted", fromnode);
            CHECK(T_used.has_value() and std::hypot(rigid_apply(*T_used, door_43_P).cx - 2.95f, rigid_apply(*T_used, door_43_P).cy - 1.55f) < 0.02f);
            const std::vector<float> shortv{1.f};
            CHECK(not node_P_from_K(nullptr, &node_rot).has_value() and not node_P_from_K(&node_tr, nullptr).has_value()
                  and not node_P_from_K(&shortv, &node_rot).has_value());
            CHECK(not accepted_K_from_P("accepted", node_P_from_K(nullptr, nullptr)).has_value());   // malformed: no re-seat
        }
        // pending: the edge holds ltsm's prior, and nothing is re-seated by it
        CHECK(not accepted_K_from_P("pending", P_from_K_prior).has_value());
        CHECK(not accepted_K_from_P("accepted", std::nullopt).has_value());
        // accepted: the edge now holds the accepted T (written before the state) — the door lands on DOOR_2's hole
        const auto T = accepted_K_from_P("accepted", P_from_K_accepted);
        CHECK(T.has_value());
        if (T.has_value())
        {
            const auto reseated = rigid_apply(*T, door_43_P);
            std::print("  Task 20: door_43 re-seated to ({:.2f},{:.2f}) yaw {:.3f}; door_20 at (2.95,1.55)\n", reseated.cx, reseated.cy, reseated.yaw);
            CHECK(std::hypot(reseated.cx - 2.95f, reseated.cy - 1.55f) < 0.02f);   // near door_20, at the hole
            const auto by_prior = rigid_apply(rigid_inverse(P_from_K_prior), door_43_P);
            CHECK(std::hypot(by_prior.cx - reseated.cx, by_prior.cy - reseated.cy) > 6.4f);   // the prior did NOT leak
        }
        // no pause: in the rename cycle the robot cannot answer yet (its block may still be P's), the accepted T can
        const Rigid2 robot_K0{-0.4f, 2.6f, -1.2f}, twist_per_s{0.f, 0.3f, 0.4f};
        const auto robot_K = [&](std::uint64_t t) { return rigid_compose(robot_K0, rigid_power(twist_per_s, 1e-3f * float(t))); };
        const auto robot_P = [&](std::uint64_t t) { return rigid_compose(rigid_inverse(K_from_P_true), robot_K(t)); };
        AdoptionFramePairing pr;
        pr.observe(7, "room_4", RobotFix{robot_P(1000), 1000}); pr.observe(7, "room_4", RobotFix{robot_P(1100), 1100});
        pr.observe(7, "room_2", RobotFix{robot_P(1150), 1150});
        CHECK(not pr.K_from_P(RobotFix{robot_P(1150), 1150}).has_value() and T.has_value());
        // the cross-check, once K's first block is read: the robot agrees with the accepted T, and disagrees with the prior
        pr.observe(7, "room_2", RobotFix{robot_K(1250), 1250});
        const auto R = pr.K_from_P(RobotFix{robot_K(1250), 1250});
        CHECK(R.has_value() and T.has_value());
        if (R.has_value() and T.has_value())
        {
            const auto ok = frame_discrepancy(*T, *R), bad = frame_discrepancy(rigid_inverse(P_from_K_prior), *R);
            CHECK(ok.d_xy < 0.01f and ok.d_yaw_deg < 0.2f);
            CHECK(bad.d_xy > 1.0f and bad.d_yaw_deg > 179.0f);
        }
    }
    std::print("{}\n", fails == 0 ? "ALL PASS" : "FAILED");
    return fails != 0;
}
