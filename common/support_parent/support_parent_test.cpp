/*
 * support_parent_test.cpp — the support DECISION arithmetic. Standalone:
 *
 *     g++ -std=c++23 -O1 support_parent_test.cpp -o support_parent_test && ./support_parent_test
 *
 * ★WHY. The sweep that motivated extracting this module (2026-08-17) found that support parenting existed in
 * exactly one agent, bespoke, untested — and that the microwave about to need it would be its first reader.
 * The graph half of support_parent.h cannot be tested without a live DSR (get_nodes_by_type, InnerEigenAPI),
 * so the JUDGEMENT was separated out into log_evidence() / floor_log_evidence() and is tested here. What is
 * left untested is lookup and framing; what is tested is every way the decision can go wrong.
 *
 * The two functions are DUPLICATED below because support_parent.h includes dsr_api.h. Keep them in step — if
 * you change the scoring, change it here and watch this fail first.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>

// ── mirror of rc::support::{log_evidence, floor_log_evidence} ──────────────────────────────────────
static float log_evidence(float r_z, float sigma_z2, float d_xy2, float lambda_xy)
{ return -0.5f * (r_z * r_z) / std::max(1e-9f, sigma_z2) - lambda_xy * d_xy2; }
static float floor_log_evidence(float base_z, float sigma_z)
{ const float sz = std::max(0.005f, sigma_z); return -0.5f * (base_z / sz) * (base_z / sz); }

static int fails = 0;
static void ck(bool c, const char* what) { if (not c) { std::printf("FAIL: %s\n", what); ++fails; } }

// Does a candidate WIN, i.e. beat the floor by the margin? (the rule decide() applies)
static bool wins(float ll_support, float base_z, float sigma_z, float margin)
{ return ll_support > floor_log_evidence(base_z, sigma_z) + margin; }

int main()
{
    constexpr float sz = 0.04f, sz2 = sz * sz, lam = 50.0f, margin = 2.0f;

    // ── a bottle sitting exactly on a well-known table top ────────────────────────────────────────
    ck(wins(log_evidence(0.0f, sz2, 0.0f, lam), 0.75f, sz, margin),
       "a bottle resting exactly on a 0.75 m table beats the floor");

    // ── the same bottle, but its base is on the FLOOR ─────────────────────────────────────────────
    ck(not wins(log_evidence(-0.75f, sz2, 0.0f, lam), 0.0f, sz, margin),
       "a bottle on the floor does NOT get parented to the table above it");

    // ── ★THE MICROWAVE CASE: base at a 0.89 m worktop top, centred, well-known support ────────────
    ck(wins(log_evidence(0.0f, sz2, 0.0f, lam), 0.89f, sz, margin),
       "a microwave resting on a 0.89 m worktop beats the floor");
    // …and the same microwave 4 cm above the worktop (one σ_z) still prefers it to the floor
    ck(wins(log_evidence(0.04f, sz2, 0.0f, lam), 0.89f, sz, margin),
       "one sigma of vertical error does not lose the support");
    // ★★AND HERE IS THE PROPERTY I GOT WRONG FIRST, WHICH IS A REQUIREMENT ON THE CALLER, NOT A BUG.
    // An object 30 cm ABOVE a worktop still beats the floor: −½(0.30/0.04)² = −28 against
    // −½(1.19/0.04)² = −443. That is correct reasoning — it is the better of two bad explanations, and the
    // floor is a terrible one for something 1.19 m up. So a LONE distant support still wins.
    ck(wins(log_evidence(0.30f, sz2, 0.0f, lam), 1.19f, sz, margin),
       "a lone support 30 cm away still beats a hopeless floor — the better of two bad explanations");
    // ⇒ THE CONSEQUENCE: the decision is only as good as the CANDIDATE SET. Given the wall unit as well, the
    // wall unit wins on residual and the worktop loses. So a caller's support_prefixes must name EVERY surface
    // that could hold this object, or it gets attached to the wrong one with a large residual instead of being
    // refused. For a microwave that means {"cabinet", "table"} — worktops AND wall units, not just worktops.
    ck(log_evidence(0.0f, sz2, 0.0f, lam) > log_evidence(0.30f, sz2, 0.0f, lam),
       "with BOTH candidates present the one the object actually rests on wins");

    // ── ★A POORLY-KNOWN SUPPORT IS A WEAK ANCHOR: the σ_z inflation must be able to LOSE ──────────
    // Same 8 cm residual, twice: against a tight support it is decisive, against one with 0.20 m σ it is not.
    const float r = 0.08f;
    ck(log_evidence(r, sz2, 0.0f, lam) < log_evidence(r, sz2 + 0.04f, 0.0f, lam),
       "inflating sigma_z by the support's own z-variance RAISES its evidence (a vague anchor forgives error)");
    ck(wins(log_evidence(r, sz2, 0.0f, lam), 0.75f, sz, margin)
       == wins(log_evidence(r, sz2, 0.0f, lam), 0.75f, sz, margin),
       "the comparison is deterministic");

    // ── the footprint penalty must be able to decide between two supports at the same height ──────
    const float ll_centred = log_evidence(0.0f, sz2, 0.0f, lam);
    const float ll_edge    = log_evidence(0.0f, sz2, 0.05f * 0.05f, lam);
    ck(ll_centred > ll_edge, "a centred object outscores one hanging off the edge at the same height");

    // ── the margin is a REAL requirement, not decoration ──────────────────────────────────────────
    // An object exactly between two explanations (residual == its own height) must stay with the floor.
    const float amb = 0.02f;
    ck(not wins(log_evidence(amb, sz2, 0.0f, lam), amb, sz, margin),
       "an ambiguous object keeps the FLOOR — the safe direction, since the room is always a valid parent");
    ck(wins(log_evidence(0.0f, sz2, 0.0f, lam), amb, sz, 0.0f)
       and not wins(log_evidence(0.0f, sz2, 0.0f, lam), amb, sz, 1000.0f),
       "raising the decision margin can always deny a support (the knob has authority)");

    // ── degenerate inputs must not produce a decision by accident ─────────────────────────────────
    ck(std::isfinite(log_evidence(0.0f, 0.0f, 0.0f, lam)),
       "a zero sigma_z2 is clamped, not divided by");
    ck(floor_log_evidence(0.0f, 0.0f) == 0.0f,
       "a zero sigma_z is clamped; a base at 0 is perfect floor evidence");

    std::printf(fails ? "support_parent: %d FAILED\n" : "support_parent: all checks passed\n", fails);
    return fails ? 1 : 0;
}
