// test_door_look_site — D4: the epistemic look's viewpoint after the controller hands a site back (gold13: the look was offered
// inside Bed(1), the controller found no route and spun). A hand-back is DECAYING evidence that its site is blocked (the
// approach's ApproachMemory, in the door's frame); each face's gain is weighted by P(standable) = 1 − P(refused).
//   1. no memory: the best-gain face wins at its full gain; not unplaceable;
//   2. its site handed back: another face that still sees the door is offered; the refused site is never re-offered;
//   3. continuity: a hand-back 0.3 m from a site lowers its value less than one on it (no gate);
//   4. every usable site handed back ⇒ UNPLACEABLE (the look is withdrawn) — and it returns, continuously, as the memory decays;
//   5. unusable faces (inside furniture / outside the room) are not candidates; none usable ⇒ face −1, not "unplaceable".
//
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_door_look_site.cpp -o /tmp/test_door_look_site && /tmp/test_door_look_site
#include <cmath>
#include <print>

#include "../src/door_look_site.h"
#include "../src/door_approach_pose.h"

namespace
{
int g_fail = 0;
void check(bool ok, const char* what)
{
    if (not ok) { ++g_fail; std::println("  FAIL: {}", what); }
}
}  // namespace

int main()
{
    rc::door::Aperture ap;
    ap.wall_O = {0.f, 0.f}; ap.wall_u = {1.f, 0.f}; ap.s = 0.f; ap.w = 1.f;   // centre (0.5, 0); across_u = +y
    const std::array<Eigen::Vector2f, 4> pos{{{2.3f, 0.f}, {-1.3f, 0.f}, {0.5f, 1.75f}, {0.5f, -1.75f}}};  // +x -x +y -y
    const std::array<float, 4> gain{0.2f, 0.1f, 0.9f, 0.0f};
    const std::array<bool, 4> usable{true, true, true, false};
    constexpr std::int64_t kTau = 60000;
    constexpr float tau_s = 60.f, r = 0.45f;
    const std::string room = "room_2";
    rc::door::ApproachMemory mem;
    const auto side_of = [&](const Eigen::Vector2f& q) { return (q - ap.centre_xy()).dot(ap.across_u()) < 0.f ? -1.f : 1.f; };
    const auto choose = [&](std::int64_t now)
    {
        return rc::door::choose_look_site(gain, usable, pos, [&](const Eigen::Vector2f& q)
                                          { return mem.p_refused(ap, side_of(q), room, q, now, tau_s, r); });
    };

    std::println("1. no memory");
    const auto s0 = choose(0);
    check(s0.face == 2 and s0.value == gain[2] and s0.share == 1.f, "the best face is not offered at its full gain");

    std::println("2. the +y site handed back (the one inside the bed)");
    mem.refuse(ap, side_of(pos[2]), room, pos[2], 0);
    const auto s1 = choose(0);
    std::println("    face {} value {:.3f}", s1.face, s1.value);
    check(s1.face == 0 and s1.value > 0.f and s1.share > 0.2f, "another viewpoint that sees the door was not offered");
    check(s1.face != 2, "the handed-back site was offered again");

    std::println("3. continuity");
    rc::door::ApproachMemory near_mem;
    near_mem.refuse(ap, side_of(pos[2]), room, pos[2] + Eigen::Vector2f(0.3f, 0.f), 0);
    const float p_near = near_mem.p_refused(ap, 1.f, room, pos[2], 0, tau_s, r), p_on = mem.p_refused(ap, 1.f, room, pos[2], 0, tau_s, r);
    check(p_near > 0.f and p_near < p_on and p_on == 1.f, "a hand-back beside the site must weigh less than one on it");

    std::println("4. realistic: hand-backs at age > 0 and displaced from the face centres (the claimed pose after repairs)");
    // hysteresis as step_epistemic applies it (cfg offer 0.60 / withdraw 0.35)
    const auto step_on = [](bool on, float share) { return on ? share >= 0.35f : share > 0.60f; };
    rc::door::ApproachMemory real;
    real.refuse(ap, side_of(pos[2]), room, pos[2] + Eigen::Vector2f(0.1f, -0.15f), 0);
    real.refuse(ap, side_of(pos[0]), room, pos[0] + Eigen::Vector2f(-0.12f, 0.1f), 0);
    real.refuse(ap, side_of(pos[1]), room, pos[1] + Eigen::Vector2f(0.1f, 0.1f), 0);
    const auto chooseR = [&](std::int64_t now)
    {
        return rc::door::choose_look_site(gain, usable, pos, [&](const Eigen::Vector2f& q)
                                          { return real.p_refused(ap, side_of(q), room, q, now, tau_s, r); });
    };
    bool on = true, withdrawn_early = false, back = false; float prev_share = -1.f; bool mono = true;
    for (int k = 0; k <= 400; ++k)
    {
        const std::int64_t now = k * 500 + 20000;   // from 20 s old (age > 0 at every sample)
        const auto s = chooseR(now);
        on = step_on(on, s.share);
        if (k == 0 and not on) withdrawn_early = true;
        if (k > 0 and s.share < prev_share - 1e-6f) mono = false;
        prev_share = s.share;
        if (withdrawn_early and on) { back = true; std::println("    offered again at age {:.0f} s (share {:.2f})", now * 1e-3, s.share); break; }
    }
    check(withdrawn_early, "every site remembered at age 20 s must be withdrawn by the hysteresis (share below withdraw_prob)");
    check(back, "the look did not come back as the memory decayed");
    check(mono, "the share must rise monotonically as the memory decays");
    check(chooseR(100 * kTau).share > 0.99f, "after the memory is gone the full worth returns");

    std::println("5. nothing usable");
    const auto s3 = rc::door::choose_look_site(gain, std::array<bool, 4>{false, false, false, false}, pos, [](const Eigen::Vector2f&) { return 0.f; });
    check(s3.face == -1 and s3.share == 1.f, "no usable face is not a hand-back");

    if (g_fail == 0) std::println("ALL PASS"); else std::println("FAILED ({})", g_fail);
    return g_fail == 0 ? 0 : 1;
}
