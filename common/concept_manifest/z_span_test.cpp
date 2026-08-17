/*
 * z_span_test.cpp — the vertical-span rule, tested. Standalone:
 *
 *     g++ -std=c++23 -O1 z_span_test.cpp -o z_span_test && ./z_span_test
 *
 * ★WHY THIS EXISTS (2026-08-17). z_span() decides the ONE band the SDF, the point-admission gate, the NBV
 * Target and the LiDAR carve all derive from — and it had NO test anywhere in the tree. Worse, `counter_top`
 * (the support a MICROWAVE needs; the enum's own comment says "microwave, kettle") was declared, named, and
 * used by no manifest and exercised by no code. The first agent to need it would have been its first test.
 *
 * It caught a real bug the same hour: rc::manifest::adopt_span passed z_base_m = 0 unconditionally, which is
 * invisible for floor_anchored and hangs — every support in use today — and puts a counter_top object ON THE
 * FLOOR. A path with no user is a path with no test, and the first user pays for both.
 *
 * The struct is DUPLICATED here on purpose: concept_manifest.h includes genericworker.h for ConfigLoader,
 * which drags in the whole RoboComp chain and makes a standalone test impossible. Keep the two in step — if
 * you change z_span(), change it here and watch this fail first.
 */
#include <cstdio>
#include <cmath>
#include <utility>
enum class Support { unknown, floor_anchored, leg_supported, hangs, counter_top, resolved };
struct Geometry {
    Support support = Support::unknown;
    float z_top_m = 0, z_base_m = 0, extent_m = 0;
    void z_span(float& z0, float& z1) const {
        switch (support) {
            case Support::floor_anchored: z0 = 0.0f;               z1 = z_top_m;              break;
            case Support::leg_supported:
            case Support::hangs:          z0 = z_top_m - extent_m; z1 = z_top_m;              break;
            case Support::counter_top:    z0 = z_base_m;           z1 = z_base_m + extent_m;  break;
            default:                      z0 = 0.0f;               z1 = 0.0f;                 break;
        }
        if (z1 < z0) std::swap(z0, z1);
    }
};
int fails = 0;
void ck(bool c, const char* w) { if (!c) { std::printf("FAIL: %s\n", w); ++fails; } }
void span(Support s, float top, float base, float ext, float& a, float& b)
{ Geometry g; g.support=s; g.z_top_m=top; g.z_base_m=base; g.extent_m=ext; g.z_span(a,b); }
int main()
{
    float a, b;
    span(Support::floor_anchored, 1.90f, 0, 1.90f, a, b);
    ck(a==0.0f && std::abs(b-1.90f)<1e-6f, "floor_anchored fridge spans [0,1.90]");
    span(Support::hangs, 2.05f, 0, 0.50f, a, b);
    ck(std::abs(a-1.55f)<1e-6f && std::abs(b-2.05f)<1e-6f, "hanging hood spans [1.55,2.05]");
    // ★THE MICROWAVE CASE — first user of counter_top, never exercised before.
    span(Support::counter_top, 0, 0.89f, 0.30f, a, b);
    ck(std::abs(a-0.89f)<1e-6f && std::abs(b-1.19f)<1e-6f, "counter_top microwave spans [0.89,1.19] from z_base");
    span(Support::counter_top, 1.90f, 0.89f, 0.30f, a, b);
    ck(std::abs(a-0.89f)<1e-6f, "counter_top IGNORES z_top — a stale height cannot move it");
    span(Support::counter_top, 0, 0.0f, 0.30f, a, b);
    ck(a==0.0f && std::abs(b-0.30f)<1e-6f, "counter_top with NO z_base collapses to the FLOOR (the trap)");
    span(Support::resolved, 1.0f, 0.5f, 0.3f, a, b);
    ck(a==0.0f && b==0.0f, "resolved yields an EMPTY band, never a plausible guess");
    span(Support::unknown, 1.0f, 0.5f, 0.3f, a, b);
    ck(a==0.0f && b==0.0f, "unknown yields an EMPTY band");
    span(Support::hangs, 0.40f, 0, 0.50f, a, b);
    ck(a<=b, "an extent exceeding the top still yields an ordered span");
    std::printf(fails ? "z_span: %d FAILED\n" : "z_span: all checks passed\n", fails);
    return fails ? 1 : 0;
}
