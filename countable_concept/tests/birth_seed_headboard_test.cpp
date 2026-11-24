// birth_seed_headboard_test — the birth seed's H (F1 round 6.2; docs/superpowers/results/2026-11-25-bed-depth-diagnosis.md).
// A bed cloud: the mattress/box top at 0.65 (= the class prior mean) and its front face, plus 8–30 % of the points on the
// headboard panel (0.10 m thick at the back, up to 1.25 m). The p95 of the whole cloud is the headboard (the 1.02 that collapsed
// engine1's bed); CountableBelief::seed_height weights each point by its box-vs-headboard share and takes the class-prior MAP
// with the box's own top. Asserted: |H − 0.65| ≤ σ_H = 0.10 (within 1σ of the class prior) at every panel fraction; and with no
// panel at all (a nightstand-like class, every share 1) H equals the closed-form MAP of the prior and the cloud's p95.
// Run: ./birth_seed_headboard_test
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
#include "countable_belief.h"

using namespace rc;
static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

static constexpr float kW = 1.75f, kD = 2.10f, kTop = 0.65f, kHbH = 1.25f, kHbT = 0.10f, kSigTop = 0.0289f;

static CountableBeliefParams params(bool panel)
{
    CountableBeliefParams p;
    p.prior_height_m = 0.65f; p.prior_height_std = 0.10f;
    p.prior_footprint_m = kW; p.prior_depth_m = kD;
    p.persistent = true;
    if (panel) { p.headboard_h_m = kHbH; p.headboard_t_m = kHbT; }
    return p;
}

// wall y = 0, room at y < 0; back (local +y) on the wall: centre (0, −kD/2), yaw 0
static std::vector<Eigen::Vector3f> cloud(float panel_frac, bool panel)
{
    std::vector<Eigen::Vector3f> v;
    for (int i = 0; i < 40; ++i) for (int j = 0; j < 30; ++j)          // the box top, its front half (what a front view sees)
        v.emplace_back(-0.5f * kW + kW * (i + 0.5f) / 40.0f, -kD + 0.5f * kD * (j + 0.5f) / 30.0f, kTop);
    for (int i = 0; i < 40; ++i) for (int j = 0; j < 10; ++j)          // the front face
        v.emplace_back(-0.5f * kW + kW * (i + 0.5f) / 40.0f, -kD, kTop * (j + 0.5f) / 10.0f);
    if (panel)
    {
        const int n_box = int(v.size()), n_hb = int(std::lround(panel_frac / (1.0f - panel_frac) * n_box));
        for (int k = 0; k < n_hb; ++k)                                  // the panel's front face, above the box top
            v.emplace_back(-0.5f * kW + kW * ((k % 40) + 0.5f) / 40.0f, -kHbT, kTop + (kHbH - kTop) * ((k / 40) % 20 + 0.5f) / 20.0f);
    }
    return v;
}

int main()
{
    std::printf("birth_seed_headboard_test (F1 round 6.2)\n");
    const CountableBeliefState s0{0.0f, -0.5f * kD, 1.0f, kW, kD, 0.0f};   // the born pose (birth_on_priors); H is re-derived
    for (const float frac : {0.08f, 0.15f, 0.30f})
    {
        const auto pts = cloud(frac, true);
        std::vector<float> z; for (const auto& q : pts) z.push_back(q.z());
        const std::size_t k = std::size_t(0.95f * (z.size() - 1));
        std::nth_element(z.begin(), z.begin() + std::ptrdiff_t(k), z.end());
        const CountableBelief b(s0, params(true));
        const float H = b.seed_height(pts, kSigTop);
        std::printf("  panel %.0f %%: p95 of the cloud %.3f ; seed H %.4f (class prior 0.65 ± 0.10)\n", 100.0f * frac, z[k], H);
        CHECK(std::abs(H - 0.65f) <= 0.10f);
    }
    {   // no panel: every share 1 ⇒ the closed-form MAP of N(0.65, 0.10²) and the p95 with σ_top
        const auto pts = cloud(0.0f, false);
        std::vector<float> z; for (const auto& q : pts) z.push_back(q.z());
        std::ranges::sort(z);
        float top = z.back(); { double acc = 0; for (float zz : z) { acc += 1.0; if (acc >= 0.95 * double(z.size())) { top = zz; break; } } }
        const float a = 1.0f / (0.10f * 0.10f), c = 1.0f / (kSigTop * kSigTop), want = (a * 0.65f + c * top) / (a + c);
        const CountableBelief b(s0, params(false));
        const float H = b.seed_height(pts, kSigTop);
        std::printf("  no panel: weighted p95 %.4f ; seed H %.5f (closed form %.5f)\n", top, H, want);
        CHECK(std::abs(H - want) <= 1e-6f);
    }
    std::printf(bad ? "birth_seed_headboard_test: %d FAILED\n" : "birth_seed_headboard_test: OK\n", bad);
    return bad != 0;
}
