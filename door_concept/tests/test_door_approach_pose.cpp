// test_door_approach_pose — D-A: the approach standpoint after the controller hands it back (Unreachable / Infeasible).
// The controller no longer re-sites an object's standpoint; it hands the affordance back. door_concept used to re-offer
// the IDENTICAL pose (the band midpoint on the normal, deterministic) ⇒ a hand-back / re-offer loop. Now a refusal is
// DECAYING evidence that its standpoint is blocked, and the pose is re-solved over door_concept's own feasible set
// (standoff ∈ [d_min, d_max] × lateral offset along the aperture), maximising P(reach)·P(not refused).
//   1. no refusal ⇒ exactly the old pose (band midpoint, on the normal) — nothing changes for a run without hand-backs;
//   2. a refusal ⇒ the next offer differs, stays inside the band and still faces the opening;
//   3. the evidence decays: long after, the original pose returns;
//   4. a two-agent mock (a controller that refuses one blocked footprint): the loop ends — the producer converges to a
//      pose the consumer accepts within a few hand-backs, and never re-offers the refused one while the memory is fresh.
//
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_door_approach_pose.cpp -o /tmp/test_door_approach_pose && /tmp/test_door_approach_pose
#include <cmath>
#include <print>

#include "../src/door_approach_pose.h"

namespace
{
int g_fail = 0, g_checks = 0;
void check(bool ok, const char* what)
{
    ++g_checks;
    if (not ok) { ++g_fail; std::println("  FAIL: {}", what); }
}
rc::door::Aperture door()
{
    rc::door::Aperture a;
    a.wall_O = {0.f, 0.f}; a.wall_u = {1.f, 0.f}; a.s = 1.f; a.w = 1.f; a.h = 2.1f;
    return a;
}
rc::door::ApproachProblem problem()
{
    rc::door::ApproachProblem p;
    p.side = +1.f; p.d_min = 1.6f; p.d_max = 2.0f; p.sigma = 0.10f; p.needed_w = 0.70f;
    p.memory_s = 60.f; p.footprint_r = 0.35f;
    return p;
}
float dist(const rc::door::Pose2& a, const rc::door::Pose2& b) { return std::hypot(a.x - b.x, a.y - b.y); }
float range_of(const rc::door::Pose2& q) { const auto c = door().centre_xy(); return std::hypot(q.x - c.x(), q.y - c.y()); }
}  // namespace

int main()
{
    const auto ap = door();
    const auto pb = problem();
    rc::door::ApproachMemory mem;

    std::println("1. no refusal: the old pose");
    const auto c0 = rc::door::choose_approach_pose(ap, pb, mem, 0);
    const auto old = rc::door::front_pose(ap, pb.side, 0.5f * (pb.d_min + pb.d_max));
    std::println("    ({:.3f},{:.3f}) yaw {:+.3f} vs front_pose ({:.3f},{:.3f}) yaw {:+.3f}", c0.pose.x, c0.pose.y, c0.pose.yaw,
                 old.x, old.y, old.yaw);
    check(c0.pose.x == old.x and c0.pose.y == old.y and c0.pose.yaw == old.yaw, "without a refusal the pose changed");

    std::println("2. one refusal at that pose (t = 0)");
    mem.refuse({c0.pose.x, c0.pose.y}, 0);
    const auto c1 = rc::door::choose_approach_pose(ap, pb, mem, 1000);
    std::println("    -> ({:.3f},{:.3f}) standoff {:.2f} lateral {:+.2f} P(reach) {:.2f} P(refused) {:.2f}, {:.2f} m from the refused one",
                 c1.pose.x, c1.pose.y, c1.standoff, c1.lateral, c1.p_reach, c1.p_refused, dist(c1.pose, c0.pose));
    check(dist(c1.pose, c0.pose) > 0.25f, "the refused pose was re-offered");
    check(c1.standoff >= pb.d_min - 1e-4f and c1.standoff <= pb.d_max + 1e-4f, "the new pose left the band");
    check(range_of(c1.pose) <= pb.d_max + 0.5f * ap.w, "the new pose is beyond the band and the aperture's half-width");
    check(c1.p_reach > 0.3f, "the new pose does not face the opening");
    {   // the heading faces the aperture centre
        const auto c = ap.centre_xy();
        const float want = std::atan2(c.y() - c1.pose.y, c.x() - c1.pose.x);
        check(std::abs(std::remainder(c1.pose.yaw - want, 6.2831853f)) < 1e-3f, "the new pose does not face the aperture");
    }

    std::println("3. the evidence decays");
    const auto c_late = rc::door::choose_approach_pose(ap, pb, mem, 1000 * 600);   // 10 memory constants later
    std::println("    after 600 s: ({:.3f},{:.3f}), P(refused) there {:.3f}", c_late.pose.x, c_late.pose.y, c_late.p_refused);
    check(c_late.pose.x == old.x and c_late.pose.y == old.y, "the original pose never comes back");

    std::println("4. two agents: a consumer that refuses a blocked footprint");
    {
        rc::door::ApproachMemory m2;
        // the consumer: anything whose footprint touches a box standing on the normal at the band midpoint is Unreachable
        const auto blocked = [&](const rc::door::Pose2& q) { return std::hypot(q.x - old.x, q.y - old.y) < 0.45f; };
        int handbacks = 0, repeats = 0;
        rc::door::Pose2 last{1e9f, 1e9f, 0.f};
        bool accepted = false;
        for (int k = 0; k < 20 and not accepted; ++k)
        {
            const auto c = rc::door::choose_approach_pose(ap, pb, m2, 5000 * k);
            if (dist(c.pose, last) < 1e-4f) ++repeats;
            last = c.pose;
            if (blocked(c.pose)) { m2.refuse({c.pose.x, c.pose.y}, 5000 * k); ++handbacks; }
            else accepted = true;
        }
        std::println("    accepted after {} hand-back(s), {} identical re-offer(s), at ({:.2f},{:.2f})", handbacks, repeats, last.x, last.y);
        check(accepted, "the producer never found a pose the consumer accepts (loop)");
        check(repeats == 0, "a refused pose was re-offered identically");
        check(handbacks <= 4, "more than 4 hand-backs before converging");
    }
    std::println("{} / {} checks passed", g_checks - g_fail, g_checks);
    return g_fail == 0 ? 0 : 1;
}
