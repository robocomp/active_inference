// Offline harness for src/core/bernoulli_filter.h (the labelled-Bernoulli update, mask responsibilities, k-best joint
// assignments, discrete EIG). Eigen and std only.
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_bernoulli_filter.cpp -o /tmp/test_bernoulli_filter && /tmp/test_bernoulli_filter
#include "../src/core/bernoulli_filter.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>

using namespace rc::door::core::bf;
static int g_fail = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static double logit(double p) { return std::log(p / (1.0 - p)); }
static double logistic(double l) { return 1.0 / (1.0 + std::exp(-l)); }

int main()
{
    // miss_with_pd_zero_is_noop
    CHECK(update_logodds(1.3, 0.0, 0.0) == 1.3);
    // miss_lowers
    CHECK(std::abs(update_logodds(0.0, 0.9, 0.0) - std::log(0.1)) < 1e-12);
    // matches_probability_form
    {
        const double r = 0.3, pd = 0.8, S = 5.0;
        const double got = logistic(update_logodds(logit(r), pd, S));
        const double want = r * (1 - pd + pd * S) / (1 - r * pd + r * pd * S);
        CHECK(std::abs(got - want) < 1e-9);
    }
    // responsibilities_sum_to_one
    {
        const std::vector<double> lw{-3.0, -1.0, -2.5, -0.2};
        const auto w = responsibilities(lw);
        CHECK(std::abs(std::accumulate(w.begin(), w.end(), 0.0) - 1.0) < 1e-12);
        CHECK(w[3] > w[1] and w[1] > w[2] and w[2] > w[0]);
    }
    // responsibilities_stable
    {
        const std::vector<double> lw{1000.0, 1000.0};
        const auto w = responsibilities(lw);
        CHECK(std::abs(w[0] - 0.5) < 1e-12 and std::abs(w[1] - 0.5) < 1e-12);
        const std::vector<double> lo{-1e300, -std::numeric_limits<double>::infinity()};
        const auto v = responsibilities(lo);
        CHECK(v[0] == 1.0 and v[1] == 0.0);
    }
    // k_best_matches_brute_force
    {
        std::mt19937 rng(7);
        std::normal_distribution<double> nd(0.0, 2.0);
        const int M = 4, C = 5;
        Eigen::MatrixXd lw(M, C);
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < C; ++j)
                lw(i, j) = nd(rng);
        lw(2, 1) = -std::numeric_limits<double>::infinity();   // an impossible pairing
        const bool ex[C] = {true, true, true, false, false};
        // brute force
        std::vector<std::pair<double, std::vector<int>>> all;
        std::vector<int> a(M, 0);
        for (long code = 0; code < 625; ++code)
        {
            long c = code;
            for (int i = 0; i < M; ++i) { a[i] = static_cast<int>(c % C); c /= C; }
            bool ok = true;
            for (int c1 = 0; c1 < 3; ++c1)
                ok = ok and std::count(a.begin(), a.end(), c1) <= 1;
            double s = 0;
            for (int i = 0; i < M; ++i) s += lw(i, a[i]);
            if (ok and std::isfinite(s)) all.push_back({s, a});
        }
        std::ranges::sort(all, [](const auto& x, const auto& y) { return x.first > y.first; });
        const auto got = k_best(lw, ex, 5);
        CHECK(got.size() == 5);
        for (std::size_t i = 0; i < got.size() and i < 5; ++i)
        {
            CHECK(std::abs(got[i].log_w - all[i].first) < 1e-12);
            CHECK(got[i].cause_of_mask == all[i].second);
        }
        // sorted descending
        for (std::size_t i = 1; i < got.size(); ++i) CHECK(got[i - 1].log_w >= got[i].log_w);
    }
    // eig_bounds
    {
        const std::vector<double> w{0.5, 0.5};
        CHECK(std::abs(expected_information_gain(w, {{0.3, 0.7}, {0.3, 0.7}})) < 1e-12);
        CHECK(std::abs(expected_information_gain(w, {{1.0, 0.0}, {0.0, 1.0}}) - std::log(2.0)) < 1e-12);
        const std::vector<double> sk{0.9, 0.1};
        const double H = -(0.9 * std::log(0.9) + 0.1 * std::log(0.1));
        CHECK(std::abs(expected_information_gain(sk, {{1.0, 0.0}, {0.0, 1.0}}) - H) < 1e-12);
    }
    std::printf(g_fail ? "FAILED\n" : "ALL PASS\n");
    return g_fail != 0;
}
