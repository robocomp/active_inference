// test_door_approach_pose — D-A: the approach standpoint after the controller hands it back (Unreachable / Infeasible /
// OutsideRoom). The controller no longer re-sites an object's standpoint; it hands the affordance back. door_concept used
// to re-offer the IDENTICAL pose ⇒ a hand-back / re-offer loop. Now a refusal is DECAYING evidence that its standpoint is
// blocked, the pose is re-solved over door_concept's own feasible set maximising P(complete)·P(inside)·P(not refused), and
// that score IS the offer's probability, so an exhausted band withdraws the affordance through the offer policy.
// Live parameters (the review's probe): aperture 1.0 m, d_min = w + body/2 + swing clearance = 1.5, d_max = 2.0, σ 0.10,
// needed 0.70, τ 60 s, footprint 0.4 m, OfferProb 0.60 / WithdrawProb 0.35.
//   1. no refusal ⇒ the pose on the aperture's normal, in the band, offered with P ≥ OfferProb. ★Not the band's midpoint
//      any more: the objective is P(the contract completes), and the arrival scatter adds range off the normal, so its
//      argmax sits a little nearer than the midpoint (1.67 m against 1.75 here) — a deliberate change of the no-hand-back pose;
//   2. a refusal ⇒ the next offer differs, stays in the band, faces the opening, and can still complete (P(reach) ≥ 0.90);
//   3. the evidence decays: the original pose returns;
//   4. a two-agent mock (a consumer refusing one blocked footprint): converges, no identical re-offer;
//   5. EXHAUSTION (I-1): a consumer refusing EVERY pose, one hand-back every 2 s: the offer probability (approach_offer_prob) falls below
//      WithdrawProb within a few hand-backs, and no pose is offered twice while the offer is still above WithdrawProb;
//   6. the exile time (M-1): after the last hand-back of an exhausted band, how long until the offer clears OfferProb again
//      (measured 55 s = τ·ln(1/(1−OfferProb)) at τ 60 s: the most recent refusal at the best pose sets it);
//   7. the room (M-4): candidates outside the polygon are not offered — a band cut by a wall moves the pose inside;
//   8. a re-expression (M-5): the aperture re-expressed in another frame (another origin, wall_u/across_u flipped, the
//      approach side flipped with them) — the refusal lands on the same physical standpoint; another room's refusal is not
//      evidence here.
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
constexpr float kOffer = 0.60f, kWithdraw = 0.35f;
rc::door::Aperture door()
{
    rc::door::Aperture a;
    a.wall_O = {0.f, 0.f}; a.wall_u = {1.f, 0.f}; a.s = 1.f; a.w = 1.f; a.h = 2.1f;
    return a;
}
rc::door::ApproachProblem problem()
{
    rc::door::ApproachProblem p;
    p.side = +1.f; p.d_min = 1.0f + 0.3f + 0.2f; p.d_max = std::max(2.0f, p.d_min + 0.3f); p.sigma = 0.10f; p.needed_w = 0.70f;
    p.memory_s = 60.f; p.footprint_r = 0.40f; p.room = "room_1"; p.sigma_arrival = 0.10f;
    return p;
}
float dist(const rc::door::Pose2& a, const rc::door::Pose2& b) { return std::hypot(a.x - b.x, a.y - b.y); }
}  // namespace

int main()
{
    const auto ap = door();
    const auto pb = problem();
    rc::door::ApproachMemory mem;

    std::println("1. no refusal: the old pose");
    const auto c0 = rc::door::choose_approach_pose(ap, pb, mem, 0);
    std::println("    ({:.3f},{:.3f}) P(reach) {:.3f} P(complete) {:.3f} score {:.3f}", c0.pose.x, c0.pose.y, c0.p_reach,
                 c0.p_complete, c0.score);
    check(c0.lateral == 0.0f and c0.standoff >= pb.d_min and c0.standoff <= pb.d_max, "without a refusal the pose left the normal / the band");
    {
        const auto f = rc::door::front_pose(ap, pb.side, c0.standoff);
        check(c0.pose.x == f.x and c0.pose.y == f.y and c0.pose.yaw == f.yaw, "the pose on the normal is not front_pose's");
    }
    check(rc::door::approach_offer_prob(c0, c0) == 1.0f and c0.score > 0.5f, "a free band is not offered with P 1 / cannot complete");

    std::println("2. one refusal at that pose (t = 0)");
    mem.refuse(ap, pb.side, pb.room, {c0.pose.x, c0.pose.y}, 0);
    const auto c1 = rc::door::choose_approach_pose(ap, pb, mem, 1000);
    std::println("    -> ({:.3f},{:.3f}) standoff {:.2f} lateral {:+.2f} P(reach) {:.2f} P(complete) {:.2f} P(refused) {:.2f} "
                 "score {:.2f}, {:.2f} m away", c1.pose.x, c1.pose.y, c1.standoff, c1.lateral, c1.p_reach, c1.p_complete,
                 c1.p_refused, c1.score, dist(c1.pose, c0.pose));
    check(dist(c1.pose, c0.pose) > 0.2f, "the refused pose was re-offered");
    check(c1.standoff >= pb.d_min - 1e-4f and c1.standoff <= pb.d_max + 1e-4f, "the new pose left the band");
    check(c1.p_reach >= rc::door::kApproachCompleteProb, "the new pose cannot satisfy the approach contract");
    {
        const auto c = ap.centre_xy();
        const float want = std::atan2(c.y() - c1.pose.y, c.x() - c1.pose.x);
        check(std::abs(std::remainder(c1.pose.yaw - want, 6.2831853f)) < 1e-3f, "the new pose does not face the aperture");
    }

    std::println("3. the evidence decays");
    const auto c_late = rc::door::choose_approach_pose(ap, pb, mem, 1000 * 600);
    // (the decayed refusal still weighs e^-10 on the original pose, so an EQUAL-scoring neighbour may win the tie: the claim is
    // that the offer is back at the free band's value, not which of two equal poses is printed)
    std::println("    after 600 s: score {:.4f} (free band {:.4f}), P(refused) {:.5f}", c_late.score, c0.score, c_late.p_refused);
    check(c_late.score >= c0.score - 1e-3f and c_late.p_refused < 1e-3f, "the original value never comes back");

    std::println("4. two agents: a consumer refusing one blocked footprint");
    {
        rc::door::ApproachMemory m2;
        const auto blocked = [&](const rc::door::Pose2& q) { return std::hypot(q.x - c0.pose.x, q.y - c0.pose.y) < 0.45f; };
        int handbacks = 0, repeats = 0;
        rc::door::Pose2 last{1e9f, 1e9f, 0.f};
        bool accepted = false;
        for (int k = 0; k < 20 and not accepted; ++k)
        {
            const auto c = rc::door::choose_approach_pose(ap, pb, m2, 5000 * k);
            if (dist(c.pose, last) < 1e-4f) ++repeats;
            last = c.pose;
            if (blocked(c.pose)) { m2.refuse(ap, pb.side, pb.room, {c.pose.x, c.pose.y}, 5000 * k); ++handbacks; }
            else accepted = true;
        }
        std::println("    accepted after {} hand-back(s), {} identical re-offer(s)", handbacks, repeats);
        check(accepted and repeats == 0 and handbacks <= 4, "the mock did not converge without an identical re-offer");
    }

    std::println("5. exhaustion: every pose refused, one hand-back every 2 s");
    rc::door::ApproachMemory m5;
    std::int64_t t_last = 0;
    {
        int k_withdrawn = -1, repeats_on_wire = 0;
        std::vector<rc::door::Pose2> offered;
        for (int k = 0; k < 25; ++k)
        {
            const std::int64_t t = 2000LL * k;
            const auto c = rc::door::choose_approach_pose(ap, pb, m5, t);
            const float p_offer = rc::door::approach_offer_prob(c, rc::door::choose_approach_pose(ap, pb, {}, t));
            if (p_offer < kWithdraw) { k_withdrawn = k; break; }   // the offer policy withdraws it: no more hand-backs
            for (const auto& o : offered) if (dist(o, c.pose) < 1e-4f) ++repeats_on_wire;
            offered.push_back(c.pose);
            std::println("    k={:2} so {:.2f} lat {:+.2f} P(reach) {:.2f} P(complete) {:.2f} P(refused) {:.2f} score {:.2f} P(offer) {:.2f}",
                         k, c.standoff, c.lateral, c.p_reach, c.p_complete, c.p_refused, c.score, p_offer);
            m5.refuse(ap, pb.side, pb.room, {c.pose.x, c.pose.y}, t);
            t_last = t;
        }
        std::println("    withdrawn at hand-back {} ; identical poses re-offered while on the wire: {}", k_withdrawn, repeats_on_wire);
        check(k_withdrawn > 0 and k_withdrawn <= 10, "an exhausted band was not withdrawn within 10 hand-backs");
        check(repeats_on_wire == 0, "a refused pose was offered again while the affordance was still on the wire");
    }

    std::println("6. the exile time");
    {
        float t_back = -1.f;
        for (int s = 1; s <= 3600; ++s)
        {
            const std::int64_t t = t_last + 1000LL * s;
            if (rc::door::approach_offer_prob(rc::door::choose_approach_pose(ap, pb, m5, t), rc::door::choose_approach_pose(ap, pb, {}, t)) >= kOffer)
            { t_back = static_cast<float>(s); break; }
        }
        std::println("    re-offered {:.0f} s after the last hand-back (τ = {:.0f} s; one refusal alone: τ·ln(1/(1−P_offer)) = {:.0f} s)",
                     t_back, pb.memory_s, pb.memory_s * std::log(1.f / (1.f - kOffer)));
        check(t_back > 0.f and t_back <= 10.f * pb.memory_s, "an exhausted band never comes back");
    }

    std::println("7. a wall cutting the band");
    {
        auto p7 = pb;
        // the room ends 1.6 m in front of the door (y < 1.6 is inside: the room is the band's near half only)
        p7.polygon = {{-5.f, 0.f}, {5.f, 0.f}, {5.f, 1.6f}, {-5.f, 1.6f}};
        const auto c = rc::door::choose_approach_pose(ap, p7, rc::door::ApproachMemory{}, 0);
        std::println("    -> standoff {:.2f} (band [{:.2f},{:.2f}], room to 1.60) P(inside) {:.2f} score {:.2f}", c.standoff,
                     p7.d_min, p7.d_max, c.p_inside, c.score);
        check(c.pose.y < 1.6f, "a standpoint outside the room was offered");
    }

    std::println("8. a re-expression of the aperture (room switch)");
    {
        rc::door::ApproachMemory m8;
        const Eigen::Vector2f p_ref(1.3f, 1.75f);
        m8.refuse(ap, +1.f, "room_1", p_ref, 0);
        // the same physical door expressed from another association: wall_u flipped (so across_u flips), origin at the far
        // corner, and the robot's side flips with across_u — the same physical side.
        rc::door::Aperture b = ap;
        b.wall_u = {-1.f, 0.f}; b.wall_O = {3.f, 0.f}; b.s = 3.f - (ap.s + ap.w);
        const float pa = m8.p_refused(ap, +1.f, "room_1", p_ref, 0, 60.f, 0.4f);
        const float pb8 = m8.p_refused(b, -1.f, "room_1", p_ref, 0, 60.f, 0.4f);
        const float pc = m8.p_refused(b, -1.f, "room_2", p_ref, 0, 60.f, 0.4f);
        std::println("    centres ({:.2f},{:.2f}) vs ({:.2f},{:.2f}); P(refused) same frame {:.3f}, re-expressed {:.3f}, other room {:.3f}",
                     ap.centre_xy().x(), ap.centre_xy().y(), b.centre_xy().x(), b.centre_xy().y(), pa, pb8, pc);
        check(std::abs(pa - pb8) < 1e-5f and pa > 0.99f, "the refusal did not survive the re-expression");
        check(pc == 0.f, "another room's refusal was taken as evidence here");
    }
    // ── fix round 2: σ at ARRIVAL, smooth P(complete), σ lowers the VALUE never the availability ──────────────────────────
    const auto offer_of = [&](float sigma_arr, rc::door::ApproachChoice* out = nullptr)
    {
        auto p = pb;
        p.sigma_arrival = sigma_arr;
        const auto c = rc::door::choose_approach_pose(ap, p, rc::door::ApproachMemory{}, 0);
        if (out) *out = c;
        return std::pair{rc::door::approach_offer_prob(c, c), c.p_complete_live};
    };
    std::println("9. a far door: 5 m away, yaw σ 2°, translation σ 5 cm — the σ AT THE STANDPOINT, not yaw × 5 m");
    {
        Eigen::Matrix3f P = Eigen::Matrix3f::Zero();
        P(0, 0) = P(1, 1) = 0.05f * 0.05f;
        P(2, 2) = std::pow(2.0f * 3.14159265f / 180.0f, 2.0f);
        const float sa = rc::door::arrival_sigma(P, c0.standoff, 0.10f);
        const auto [po, val] = offer_of(sa);
        std::println("    arrival σ {:.3f} m (the far-range chain would read {:.3f}); P(offer) {:.2f}, P(complete) {:.2f}", sa,
                     std::sqrt(0.05f * 0.05f + std::pow(5.0f * 2.0f * 3.14159265f / 180.0f, 2.0f)), po, val);
        check(po == 1.0f and val > 0.6f, "a far door with healthy localisation is not offered with a useful value");
    }
    std::println("10. after a HOLD: σ 2 m");
    {
        const auto [po, val] = offer_of(2.0f);
        std::println("    P(offer) {:.2f}, P(complete) {:.3f}", po, val);
        check(po == 1.0f, "a lost robot's approach was WITHDRAWN (it must only lose value)");
        check(val < 0.1f, "a lost robot's approach kept a high value");
    }
    std::println("11. no cliff across σ 0.05 – 0.5");
    {
        float prev = -1.f, worst_jump = 0.f;
        bool avail = true, monotone = true, same_pose = true;
        rc::door::ApproachChoice first;
        (void)offer_of(0.05f, &first);
        for (int i = 0; i <= 450; ++i)
        {
            const float sg = 0.05f + 0.001f * static_cast<float>(i);
            rc::door::ApproachChoice c;
            const auto [po, val] = offer_of(sg, &c);
            avail = avail and po == 1.0f;
            if (prev >= 0.f) { worst_jump = std::max(worst_jump, std::abs(val - prev)); monotone = monotone and val <= prev + 1e-4f; }
            same_pose = same_pose and c.pose.x == first.pose.x and c.pose.y == first.pose.y;
            prev = val;
        }
        std::println("    availability constant: {}; value monotone: {}; largest step per mm of σ: {:.3f}; pose fixed: {}",
                     avail, monotone, worst_jump, same_pose);
        check(avail, "the availability depends on σ");
        check(worst_jump < 0.05f, "the value jumps (a cliff) somewhere in σ 0.05-0.5");
        check(monotone, "the value is not monotone in σ");
        check(same_pose, "the chosen pose changes with σ (epoch churn while unclaimed)");
    }
    std::println("{} / {} checks passed", g_checks - g_fail, g_checks);
    return g_fail == 0 ? 0 : 1;
}
