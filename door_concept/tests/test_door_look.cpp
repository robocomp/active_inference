// Offline harness for door_look.h: the EPISTEMIC "look here" affordance for a REMEMBERED door that is not
// instantiated (tourP5 leg 1: door_2 remembered in room_1's identity table, never looked at, so `cross door_2` had no
// door to cross). Checks: the standpoint sees the remembered aperture and its value is the expected information gain
// about the door's existence; which rows are offered (absent ones), withdrawn (re-born by name or place) and stop being
// offered once absence evidence rules them out; and that a detection at the place confirms instead of refuting.
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -w -I/usr/include/eigen3 -I/usr/include/opencv4 -I../common tests/test_door_look.cpp -o /tmp/test_door_look && /tmp/test_door_look
#include <clocale>
#include <cmath>
#include <numbers>
#include <print>
#include <vector>
#include <functional>
#include "../src/door_look.h"

using namespace rc::door;
static int fails = 0;
#define CHECK(c) do { if (not (c)) { ++fails; std::print("FAIL {}:{}  {}\n", __FILE__, __LINE__, #c); } } while (0)

int main()
{
    std::setlocale(LC_ALL, "");
    // A 5 x 4 m room, CCW. door_2 remembered in the top wall at (1.5, 4.0); door_1 in the right wall at (5.0, 2.0).
    const std::vector<Eigen::Vector2f> room{{0.f, 0.f}, {5.f, 0.f}, {5.f, 4.f}, {0.f, 4.f}};
    rc::nbv::Sensor zed;
    zed.hfov_rad = 1.22f; zed.vfov_rad = 0.96f; zed.height_m = 1.20f;
    const look::Params prm{.robot_radius_m = 0.30f, .clutter_prob = 0.05f, .detection_prob = 0.85f,
                           .prior_w_m = 0.90f, .prior_h_m = 2.0f, .thickness_m = 0.05f};

    IdentityRow d2; d2.room = "room_1"; d2.name = "door_2"; d2.xy = {1.5f, 4.0f};                 // identity-only row
    IdentityRow d1; d1.room = "room_1"; d1.name = "door_1"; d1.xy = {5.0f, 2.0f};
    d1.pose = RestorePose{static_cast<float>(std::numbers::pi / 2), 0.95f, 2.1f, 0.05f};        // full row
    IdentityRow other; other.room = "room_2"; other.name = "door_3"; other.xy = {1.f, 1.f};
    const std::vector<IdentityRow> rows{d2, d1, other};

    // ── 0. A look must add a view: a refused standpoint is not chosen, and with every standpoint refused there is no look ──
    {
        const auto free_plan = look::plan(d2, 0.5f, room, zed, {}, {}, prm);
        CHECK(free_plan.valid);
        const std::function<bool(const Eigen::Vector2f&)> not_that = [&](const Eigen::Vector2f& q) { return (q - free_plan.pos).norm() > 1e-3f; };
        const auto other = look::plan(d2, 0.5f, room, zed, {}, {}, prm, &not_that);
        CHECK(other.valid and (other.pos - free_plan.pos).norm() > 1e-3f);
        const std::function<bool(const Eigen::Vector2f&)> none = [](const Eigen::Vector2f&) { return false; };
        CHECK(not look::plan(d2, 0.5f, room, zed, {}, {}, prm, &none).valid);
    }

    // ── 1. The plan: a standpoint inside the room that SEES the remembered aperture, valued by EIG(existence) ──
    {
        const auto p = look::plan(d2, 0.73f, room, zed, {}, {}, prm);
        CHECK(p.valid);
        CHECK(rc::door::polygon_contains(p.pos, room));
        CHECK(p.pos.y() < 4.0f);                                         // the room side of the top wall
        CHECK(p.visible_frac > 0.5f);                                    // it sees the aperture
        CHECK(p.p_detect > 0.3f);
        const Eigen::Vector2f to_door = d2.xy - p.pos;
        CHECK(std::abs(std::remainder(p.yaw - std::atan2(to_door.y(), to_door.x()), 2.0 * std::numbers::pi)) < 0.05);
        CHECK(p.eig > 0.0f and p.eig <= std::log(2.0f) + 1e-6f);
        std::print("door_2 look: ({:.2f},{:.2f}) yaw {:.2f} p_detect {:.2f} vis {:.2f} EIG {:.3f} nats\n",
                   p.pos.x(), p.pos.y(), p.yaw, p.p_detect, p.visible_frac, p.eig);
        // The full row uses its remembered yaw; standpoint west of the right wall.
        const auto q = look::plan(d1, 0.98f, room, zed, {}, {}, prm);
        CHECK(q.valid and q.pos.x() < 5.0f and q.visible_frac > 0.5f);
        // A confident prior is worth less looking than an uncertain one (EIG is maximal at P = 0.5).
        CHECK(q.eig < look::plan(d1, 0.5f, room, zed, {}, {}, prm).eig);
        // An incomplete camera model REFUSES rather than guesses (as plan_faces does).
        rc::nbv::Sensor bad; bad.vfov_rad = 0.f;
        CHECK(not look::plan(d2, 0.73f, room, bad, {}, {}, prm).valid);
    }
    // EIG of a binary existence observation: zero when certain or when the look cannot detect.
    CHECK(look::existence_eig(0.5f, 0.05f, 0.05f) < 1e-6f);   // a detector that cannot tell the two apart
    CHECK(look::existence_eig(1.0f, 0.9f, 0.05f) < 1e-6f);
    CHECK(look::existence_eig(0.5f, 0.9f, 0.05f) > look::existence_eig(0.5f, 0.3f, 0.05f));

    // ── 2. Which rows are offered: this room's rows whose door is not live and not ruled out ──
    {
        const auto c = look::offer_rows(rows, "room_1", {}, {}, 0.5f, true);
        CHECK(c.size() == 2);                                            // door_2 and door_1, not room_2's door_3
        // Re-born by identity (same name) → withdrawn.
        const std::vector<LiveDoor> live{{"door_2", Eigen::Vector2f(1.52f, 4.0f)}};
        const auto c2 = look::offer_rows(rows, "room_1", live, {}, 0.5f, true);
        CHECK(c2.size() == 1 and c2[0].name == "door_1");
        // Re-born under ANOTHER name at the same hole → withdrawn too.
        const std::vector<LiveDoor> live2{{"door_9", Eigen::Vector2f(1.70f, 4.0f)}};
        CHECK(look::offer_rows(rows, "room_1", live2, {}, 0.5f, true).size() == 1);
        // Ruled out this visit → not offered again.
        const std::vector<std::string> ruled{"door_1"};
        const auto c3 = look::offer_rows(rows, "room_1", {}, ruled, 0.5f, true);
        CHECK(c3.size() == 1 and c3[0].name == "door_2");
        // A room entered FRESH (no ltsm recall, no accepted proposal) is not the remembered room: its frame is unrelated to
        // the rows' places, so nothing is offered (recog_proto10: a stale room_2:door_48 look exhausted the explore budget).
        CHECK(look::offer_rows(rows, "room_1", {}, {}, 0.5f, false).empty());
    }

    // ── 3. The look's belief is a live door's belief (spec 3, Task 10): L = the remembered evidence + the view ledger, the same
    //      Bernoulli trials (detector_llr.h), the same costed deletion line (door_decision.h). Frames of one viewpoint are ONE trial
    //      (tourP9: door_2 ruled out from ONE spot 2.8 m away, p_detect 0.41, by many frames of non-detection); new viewpoints add.
    {
        const auto zed = rc::door::core::default_detectors().zed;
        const rc::door::core::Costs costs{30.f, 30.f, 330.f};          // no samples yet: C_keep = C_delete, the delete line is L = 0
        const float L0 = std::log(0.73f / 0.27f);                      // what memory says: P(exists) 0.73
        const Eigen::Vector2f cam_a(0.35f, -0.28f);                    // tourP9's look standpoint for door_2
        IdentityRow r2 = d2; r2.xy = {0.56f, 2.54f};
        const auto obs = [&](look::Evidence& e, const Eigen::Vector2f& cam, float pd, bool det)
        { return e.observe(r2, room, cam, pd, det, zed, costs, prm.prior_w_m, prm.prior_h_m); };
        look::Evidence one(L0);
        bool out = false;
        for (int k = 0; k < 2000 and not out; ++k)
            out = obs(one, cam_a + Eigen::Vector2f(0.002f * (k % 5), 0.f), 0.41f, false);
        CHECK(not out);
        CHECK(one.viewpoints() == 1);
        std::print("one viewpoint, 2000 absent frames at p_detect 0.41: P(exists) {:.3f}, L {:.3f} (not ruled out)\n", one.p_exists(), one.logodds());
        CHECK(one.p_exists() > 0.4f);
        // The one view is worth exactly one look: 2000 frames == 1 frame.
        look::Evidence once(L0);
        obs(once, cam_a, 0.41f, false);
        CHECK(std::abs(once.logodds() - one.logodds()) < 1e-3f);
        // Several DISTINCT viewpoints (an arc in front of the door): each adds one look's evidence => past the delete line.
        look::Evidence many(L0);
        int views = 0;
        out = false;
        for (int v = 0; v < 12 and not out; ++v)
        {
            const float a = (-50.f + 10.f * v) * std::numbers::pi_v<float> / 180.f;
            const Eigen::Vector2f cam = r2.xy + 2.0f * Eigen::Vector2f(std::sin(a), -std::cos(a));
            for (int k = 0; k < 30 and not out; ++k)
                out = obs(many, cam, 0.6f, false);
            ++views;
        }
        CHECK(out);
        std::print("ruled out after {} arc steps, {} ledger views (P now {:.3f}, L {:.3f})\n", views, many.viewpoints(), many.p_exists(), many.logodds());
        CHECK(many.viewpoints() >= 2);
        // Occluded / out of view (p_detect 0) is NOT evidence: HOLD for ever.
        look::Evidence held(L0);
        for (int k = 0; k < 400; ++k) CHECK(not obs(held, cam_a, 0.0f, false));
        CHECK(std::abs(held.logodds() - L0) < 1e-6f and held.viewpoints() == 0);
        // A detection at the place confirms.
        look::Evidence seen(L0);
        for (int k = 0; k < 20; ++k) CHECK(not obs(seen, cam_a, 0.8f, true));
        CHECK(seen.p_exists() > 0.9f);
        // A costlier deletion moves the line: the same absence no longer rules a place out when losing it is dear (C_delete >> C_keep).
        const rc::door::core::Costs dear{30.f, 3000.f, 3300.f};
        look::Evidence spared(L0);
        bool gone = false;
        for (int v = 0; v < 12 and not gone; ++v)
        {
            const float a = (-50.f + 10.f * v) * std::numbers::pi_v<float> / 180.f;
            const Eigen::Vector2f cam = r2.xy + 2.0f * Eigen::Vector2f(std::sin(a), -std::cos(a));
            gone = spared.observe(r2, room, cam, 0.6f, false, zed, dear, prm.prior_w_m, prm.prior_h_m);
        }
        CHECK(not gone);
        // A removed row is not offered a look: "no door here" was decided already.
        IdentityRow gone_row = d2; gone_row.removed_ms = 1;
        CHECK(look::offer_rows(std::vector<IdentityRow>{gone_row}, "room_1", {}, {}, 0.5f, true).empty());
    }

    if (fails == 0) std::print("test_door_look: ALL PASS\n");
    return fails == 0 ? 0 : 1;
}
