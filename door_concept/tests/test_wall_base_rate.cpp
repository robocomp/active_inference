// Offline harness for src/core/wall_base_rate.h (the per-wall base rate as a function of the OTHER doors' beliefs).
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_wall_base_rate.cpp -o /tmp/test_wall_base_rate && /tmp/test_wall_base_rate
#include "../src/core/wall_base_rate.h"
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <vector>

using namespace rc::door::core;
static int g_fail = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static float logit(double p) { return static_cast<float>(std::log(p / (1.0 - p))); }

int main()
{
    const WallRates seed{};
    // seed_means
    CHECK(std::abs(seed.first.mean() - 0.4) < 1e-9);
    CHECK(std::abs(seed.another.mean() - 0.01) < 1e-9);
    // no_others_is_first
    CHECK(std::abs(wall_base_rate(seed, {}) - std::log(0.4f / 0.6f)) < 1e-5f);
    // certain_other_is_another
    {
        const float r[] = {1.0f};
        CHECK(std::abs(wall_base_rate(seed, r) - std::log(0.01f / 0.99f)) < 1e-5f);
    }
    // uncertain_other_pays_part
    {
        const float r[] = {0.5f};
        const float v = wall_base_rate(seed, r), f = logit(0.4), a = logit(0.01);
        CHECK(v > a and v < f);
        CHECK(std::abs(v - (0.5f * f + 0.5f * a)) < 1e-5f);
    }
    // order_independent
    {
        const float r1[] = {0.2f, 0.7f}, r2[] = {0.7f, 0.2f};
        CHECK(wall_base_rate(seed, r1) == wall_base_rate(seed, r2));
    }
    // wall_count_base_rates (final review 1): the exact marginal of each member under the wall's configuration prior, by brute force
    {
        const auto brute = [&](const std::vector<double>& lo, const std::vector<int>& slot)   // returns base-rate log-odds per member
        {
            const int S = 1 + *std::max_element(slot.begin(), slot.end());
            std::vector<double> lso(S, -1e300);
            for (std::size_t i = 0; i < lo.size(); ++i) lso[slot[i]] = std::log(std::exp(lso[slot[i]]) + std::exp(lo[i]));
            std::vector<double> so(S);
            for (int s = 0; s < S; ++s) so[s] = std::exp(lso[s]);
            const double w1 = 0.4 / 0.6, w2 = 0.01 / 0.99;
            std::vector<double> marg(S, 0.0);
            double Z = 0.0;
            for (unsigned m = 0; m < (1u << S); ++m)
            {
                const int n = __builtin_popcount(m);
                double w = n == 0 ? 1.0 : w1 * std::pow(w2, n - 1);
                for (int s = 0; s < S; ++s) if (m >> s & 1u) w *= so[s];
                Z += w;
                for (int s = 0; s < S; ++s) if (m >> s & 1u) marg[s] += w;
            }
            std::vector<double> out(lo.size());
            for (std::size_t i = 0; i < lo.size(); ++i)
            {
                const double mi = marg[slot[i]] / Z * std::exp(lo[i]) / so[slot[i]];
                out[i] = std::log(mi / (1.0 - mi)) - lo[i];
            }
            return out;
        };
        const WallRates seed2{};
        // a lone member gets exactly the first-door rate, whatever its evidence
        for (const double lo : {-3.0, 0.0, 4.0})
        {
            const std::vector<double> l{lo}; const std::vector<int> sl{0};
            CHECK(std::abs(wall_count_base_rates(seed2, l, sl)[0] - std::log(0.4 / 0.6)) < 1e-9);
        }
        // four evidence-free members: each equal, the brute-force value, order-free
        {
            const std::vector<double> l{0.0, 0.0, 0.0, 0.0}; const std::vector<int> sl{0, 1, 2, 3};
            const auto b = wall_count_base_rates(seed2, l, sl), ref = brute(l, sl);
            for (std::size_t i = 0; i < 4; ++i) CHECK(std::abs(b[i] - ref[i]) < 1e-9 and std::abs(b[i] - b[0]) < 1e-12);
            std::printf("four evidence-free members: base %.4f (marginal p %.4f)\n", b[0], 1.0 / (1.0 + std::exp(-b[0])));
        }
        // mixed evidence on 5 members, two of them one hole: matches the brute force
        {
            const std::vector<double> l{3.0, -0.5, 1.2, 0.3, -2.0}; const std::vector<int> sl{0, 1, 2, 2, 3};
            const auto b = wall_count_base_rates(seed2, l, sl), ref = brute(l, sl);
            for (std::size_t i = 0; i < l.size(); ++i) CHECK(std::abs(b[i] - ref[i]) < 1e-5);   // 1e-5: room_door_posterior holds the slot log-odds as float
            // the same members in another order: the same answers
            const std::vector<double> l2{-2.0, 0.3, 1.2, -0.5, 3.0}; const std::vector<int> s2{3, 2, 2, 1, 0};
            const auto b2 = wall_count_base_rates(seed2, l2, s2);
            for (std::size_t i = 0; i < l.size(); ++i) CHECK(std::abs(b[i] - b2[l.size() - 1 - i]) < 1e-9);
        }
        // two labels on ONE hole are alternatives (a hole holds one door): their marginals sum to at most 1 however strong the evidence;
        // as two places on one wall with the same evidence the pair can both hold (each pays the "another door" cost once, not twice)
        {
            const std::vector<double> l{8.0, 8.0}; const std::vector<int> one_slot{0, 0}, two_slots{0, 1};
            const auto a = wall_count_base_rates(seed2, l, one_slot), d = wall_count_base_rates(seed2, l, two_slots);
            const auto p = [&](double b, double lo) { return 1.0 / (1.0 + std::exp(-(b + lo))); };
            CHECK(p(a[0], 8.0) + p(a[1], 8.0) < 1.0 + 1e-9);
            CHECK(p(d[0], 8.0) + p(d[1], 8.0) > 1.5);
        }
    }
    std::printf(g_fail ? "FAILED\n" : "ALL PASS\n");
    return g_fail != 0;
}
