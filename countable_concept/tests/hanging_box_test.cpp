// hanging_box_test — K1: a countable class can HANG (the hood: manifest support = "hangs", body z ∈ [H − e, H]).
//
// What is checked, every bound derived before running (geometry or exact arithmetic, nothing fitted):
//   1. floor-anchored (extent ≤ 0) is unchanged: VerticalSpan gives 0, H/2, H/2 bit-for-bit as the old expressions;
//   2. the hanging SDF: 0 on the underside and on the top, the distance to the underside for a point under the body (the
//      hob/worktop), and the SDF does not depend on where the floor is;
//   3. the beam part (the beam model's convex part): centre / half from the span, and its J column ∂(cz, hz)/∂H equal to a
//      central finite difference of beam_part itself (linear in H ⇒ exact up to rounding, tol 1e-9);
//   4. the birth height evidence of a hanging class scores the cloud's TOP against the H prior: a canopy cloud at
//      [1.55, 2.05] under H ~ N(2.05, 0.30) is evidence FOR the class (p > π_cls), whereas read as floor-anchored its 0.5 m
//      z-extent is evidence AGAINST it (p < π_cls) — the sign is the claim, so no magnitude bound is needed;
//   5. the LiDAR select box: z band [z_base − margin, top + margin].
// Run: ./hanging_box_test
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "countable_belief.h"
#include "countable_beam_staging.h"

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)
static bool near(double a, double b, double tol = 1e-5) { return std::fabs(a - b) < tol; }

int main()
{
    using namespace rc;
    // 1. floor-anchored is bit-identical
    {
        const rc::countable::VerticalSpan f{};
        for (const float H : {0.65f, 0.62f, 1.9f, 0.7312345f})
        {
            CHECK(f.z0(H) == 0.0f and f.zc(H) == 0.5f * H and f.half(H) == 0.5f * H and f.height(H) == H);
            const double Hd = H;
            CHECK(f.zc(Hd) == 0.5 * Hd and f.half(Hd) == 0.5 * Hd);
        }
        CHECK(f.dzc_dH() == 0.5 and f.dhalf_dH() == 0.5 and not f.hangs());
    }
    CountableBeliefParams p;
    p.vertical_extent_m = 0.50f;
    const CountableBeliefState s{1.0f, 2.0f, 2.05f, 0.90f, 0.50f, 0.3f};
    const CountableBelief b(s, p);
    // 2. the hanging SDF
    {
        CHECK(near(b.sdf_box({1.0f, 2.0f, 1.55f}, s), 0.0f));          // the underside, centre of the footprint
        CHECK(near(b.sdf_box({1.0f, 2.0f, 2.05f}, s), 0.0f));          // the top
        CHECK(near(b.sdf_box({1.0f, 2.0f, 0.90f}, s), 0.65f));         // the worktop under it: 1.55 − 0.90
        CHECK(near(b.sdf_box({1.0f, 2.0f, 1.80f}, s), -0.25f));        // the body's centre: −min(half) = −0.25 (z, h both)
        const CountableBelief floor_b(s, CountableBeliefParams{});
        CHECK(near(floor_b.sdf_box({1.0f, 2.0f, 0.90f}, s), -0.25f));  // floor-anchored the same point is inside the box
    }
    // 3. the beam part and its H column
    {
        const auto parts = b.beam_parts_for_test(s);
        CHECK(parts.size() == 1);
        const auto& P = parts[0];
        CHECK(near(P.c.z(), 1.80) and near(P.half.z(), 0.25));
        const double e = 1e-3;
        CountableBeliefState sp = s, sm = s; sp.H += float(e); sm.H -= float(e);
        const auto Pp = b.beam_parts_for_test(sp)[0], Pm = b.beam_parts_for_test(sm)[0];
        const double h = double(sp.H) - double(sm.H);
        CHECK(near(P.J(2, 2), (Pp.c.z() - Pm.c.z()) / h, 1e-9) and near(P.J(2, 2), 1.0, 1e-12));
        CHECK(near(P.J(6, 2), (Pp.half.z() - Pm.half.z()) / h, 1e-9) and near(P.J(6, 2), 0.0, 1e-12));
        const CountableBelief floor_b(s, CountableBeliefParams{});
        const auto F = floor_b.beam_parts_for_test(s)[0];
        CHECK(F.c.z() == 0.5 * double(s.H) and F.half.z() == 0.5 * double(s.H) and F.J(2, 2) == 0.5 and F.J(6, 2) == 0.5);
    }
    // 4. the birth height evidence
    {
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> U(-0.4f, 0.4f), Z(1.55f, 2.05f);
        std::vector<Eigen::Vector3f> canopy;
        for (int i = 0; i < 200; ++i) canopy.emplace_back(1.0f + U(rng), 2.0f + 0.5f * U(rng), Z(rng));
        CountableBeliefParams hp = p;
        hp.prior_height_m = 2.05f; hp.prior_height_std = 0.30f;
        hp.class_p_label_true = 0.8f;
        CountableBeliefParams fp = hp; fp.vertical_extent_m = -1.0f;
        const float p_hang = CountableBelief::candidate_plausibility(canopy, hp);
        const float p_floor = CountableBelief::candidate_plausibility(canopy, fp);
        std::printf("birth: hanging %.4f, read as floor-anchored %.4f, pi_cls %.2f\n", p_hang, p_floor, hp.class_p_label_true);
        CHECK(p_hang > hp.class_p_label_true and p_floor < hp.class_p_label_true);
    }
    // 5. the select box
    {
        const auto fb = countable_select_box(1.0f, 0.6f, 0.75f, 0.10f);
        CHECK(fb.z_lo == -0.10f and fb.z_hi == 0.75f + 0.10f);
        const auto hb = countable_select_box(1.0f, 0.6f, 2.05f, 0.10f, 1.55f);
        CHECK(near(hb.z_lo, 1.45f) and near(hb.z_hi, 2.15f) and hb.radius_xy == fb.radius_xy);
    }
    std::printf("hanging_box_test: %s\n", bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
}
