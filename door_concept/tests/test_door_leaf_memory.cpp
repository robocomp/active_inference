// door_leaf_memory.h: the leaf's side as a published / remembered probability (2026-11-02, chained-actions spec).
//   g++ -std=c++23 -O1 tests/test_door_leaf_memory.cpp -o /tmp/tdlm && /tmp/tdlm
#include "../src/door_leaf_memory.h"
#include "../src/door_close_clear.h"

#include <cmath>
#include <print>

namespace
{
int failures = 0, checks = 0;
void check(bool ok, const char* what)
{
    ++checks;
    if (not ok) { ++failures; std::println("FAIL: {}", what); }
}
bool near(float a, float b, float tol = 1e-4f) { return std::abs(a - b) <= tol; }
}  // namespace

int main()
{
    namespace lm = rc::door::leafmem;

    // (a) sign mapping. swing +1 = into the room (across_u points INTO it); llr is the log-odds of the MAP side.
    check(near(lm::p_into_room(+1.0f, 0.0f), 0.5f), "unknown swing -> 0.5");
    check(lm::p_into_room(+1.0f, 4.0f) > 0.98f, "MAP into, confident -> p_into ~1");
    check(lm::p_into_room(-1.0f, 4.0f) < 0.02f, "MAP out of, confident -> p_into ~0");
    check(near(lm::p_into_room(-1.0f, 2.0f), 1.0f - lm::p_into_room(+1.0f, 2.0f)), "swing sign mirrors p_into about 0.5");
    // the other face of the same hole: the room on the other side
    check(near(lm::other_face_into_room(lm::p_into_room(+1.0f, 3.0f)), lm::p_into_room(-1.0f, 3.0f)), "other face = complement");
    check(near(lm::other_face_into_room(lm::other_face_into_room(0.83f)), 0.83f), "complement is an involution");
    // hinge: the MAP jamb at the node's +x end
    check(lm::p_hinge_plus(true, 3.0f) > 0.9f and lm::p_hinge_plus(false, 3.0f) < 0.1f, "hinge plus follows the MAP jamb's x end");
    check(near(lm::p_hinge_plus(true, 0.0f), 0.5f), "unknown hinge -> 0.5");
    check(near(lm::other_face_hinge_plus(0.9f, true), 0.1f) and near(lm::other_face_hinge_plus(0.9f, false), 0.9f),
          "hinge flips with the frame across faces");

    // (b) seeding from a remembered probability: logit and the side, both ways.
    for (const float p : {0.01f, 0.2f, 0.5f, 0.77f, 0.99f})
    {
        const auto sw = lm::swing_from_into_room(p);
        check(near(lm::p_into_room(sw.swing, sw.llr), p, 1e-3f), "swing seed round-trips to the stored probability");
        for (const bool plus_is_far : {true, false})
        {
            const auto hg = lm::hinge_from_plus(p, plus_is_far);
            // the node's +x end is the Far jamb iff plus_is_far; the MAP jamb is at +x iff hg.far == plus_is_far
            check(near(lm::p_hinge_plus(hg.far == plus_is_far, hg.llr), p, 1e-3f), "hinge seed round-trips");
        }
    }
    check(near(lm::swing_from_into_room(0.99f).llr, std::log(99.0f), 1e-3f), "0.99 -> logit ln 99");
    check(lm::swing_from_into_room(0.01f).swing < 0.0f and lm::swing_from_into_room(0.99f).swing > 0.0f, "seed MAP side");
    check(std::isfinite(lm::swing_from_into_room(0.0f).llr) and std::isfinite(lm::swing_from_into_room(1.0f).llr), "saturated p stays finite");
    check(lm::hinge_from_plus(0.01f, true).far == false and lm::hinge_from_plus(0.01f, false).far == true,
          "hinge at -x is Near when +x is Far, Far when the node faces the other way");

    // (c) the standoff: away < toward; toward at certainty = the full arc.
    const float w = 1.0f, body_half = 0.3f, clr = 0.2f, sigma = 0.1f;
    const float arc = w + body_half + clr;
    const float risk = rc::door::kCloseContactRiskMax;
    const float p_toward_in  = lm::p_leaf_toward(0.99f, 1.0f);   // robot on the room side, leaf swings into the room
    const float p_toward_out = lm::p_leaf_toward(0.01f, 1.0f);   // robot on the room side, leaf swings out
    const float toward = lm::swing_weighted_arc(arc, body_half, sigma, p_toward_in, risk);
    const float away   = lm::swing_weighted_arc(arc, body_half, sigma, p_toward_out, risk);
    std::println("arc {:.3f}  toward(p={:.2f}) {:.3f}  away(p={:.2f}) {:.3f}", arc, p_toward_in, toward, p_toward_out, away);
    check(away < toward, "standoff for a leaf swinging away < toward");
    check(near(away, body_half), "away needs only the body");
    check(toward > 0.95f * arc, "toward needs ~ the full arc");
    check(near(lm::swing_weighted_arc(arc, body_half, sigma, 1.0f, risk), arc), "certain toward = the full arc exactly");
    check(lm::swing_weighted_arc(arc, body_half, sigma, 0.5f, risk) > away, "unknown side stays conservative (> away)");
    float prev = -1.0f; bool mono = true;
    for (int i = 0; i <= 100; ++i)
    {
        const float a = lm::swing_weighted_arc(arc, body_half, sigma, static_cast<float>(i) / 100.0f, risk);
        mono = mono and a >= prev - 1e-6f; prev = a;
    }
    check(mono, "arc is monotone in P(toward)");
    // the robot on the OTHER side of the wall from the same leaf flips toward/away
    check(near(lm::p_leaf_toward(0.99f, 0.0f), 0.01f), "robot beyond the wall: an into-room leaf swings away");
    check(lm::swing_weighted_arc(arc, body_half, sigma, lm::p_leaf_toward(0.99f, 0.0f), risk)
              < lm::swing_weighted_arc(arc, body_half, sigma, lm::p_leaf_toward(0.99f, 1.0f), risk), "side-dependent standoff");
    // the close goal inherits it
    check(rc::door::close_standoff(0.1f, away, sigma) < rc::door::close_standoff(0.1f, toward, sigma), "close goal: away < toward");
    check(near(lm::normal_quantile(0.975f), 1.95996f, 1e-3f) and near(lm::normal_quantile(0.5f), 0.0f), "Phi^-1");

    std::println("{} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
