// test_room_door_count.cpp — Part A of spec rt-build docs/superpowers/specs/2026-11-10-room-door-belief-design.md.
// Standalone (from door_concept/):
//   g++ -std=c++23 -O1 tests/test_room_door_count.cpp -o /tmp/trdc && /tmp/trdc
#include "../src/core/room_door_count.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace rc::door::core;

static int g_fail = 0;
#define CHECK(c) do { if (not (c)) { std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

// brute force over 2^K configurations, each door's unfound flag summed exactly
static DoorCountPosterior brute(const std::vector<DoorCountPlace>& pl, const std::vector<double>& prior)
{
    const std::size_t K = pl.size(), n_max = prior.size() - 1;
    DoorCountPosterior out; out.marg.assign(K, 0.0); out.p_count.assign(prior.size(), 0.0);
    double z = 0, more = 0;
    for (std::uint32_t m = 0; m < (1u << K); ++m)
    {
        std::size_t n = 0; double w = 1.0, p_all_found = 1.0;
        for (std::size_t i = 0; i < K; ++i)
            if (m & (1u << i)) { ++n; w *= std::exp(double(pl[i].log_odds)); p_all_found *= pl[i].found; }
        if (n > n_max) continue;
        w *= prior[n] / std::tgamma(double(K) + 1) * std::tgamma(double(n) + 1) * std::tgamma(double(K - n) + 1);
        z += w; out.p_count[n] += w;
        more += w * (1.0 - p_all_found);
        for (std::size_t i = 0; i < K; ++i) if (m & (1u << i)) out.marg[i] += w;
    }
    out.p_more = more / z;
    for (auto& x : out.marg) x /= z;
    for (auto& x : out.p_count) x /= z;
    return out;
}

static void agrees_with_brute_force()
{
    std::mt19937 rng(3);
    std::normal_distribution<float> lo(-1.5f, 2.0f);
    std::uniform_real_distribution<float> uf(0.f, 1.f);
    for (int trial = 0; trial < 20; ++trial)
    {
        std::vector<DoorCountPlace> pl(10);
        for (auto& p : pl) { p.log_odds = lo(rng); p.found = uf(rng) < 0.3f ? (uf(rng) < 0.5f ? 1.f : uf(rng)) : 0.f; }
        const auto prior = geometric_count_prior(2.0, 6);
        const auto a = room_door_posterior(pl, prior), b = brute(pl, prior);
        CHECK(std::abs(a.p_more - b.p_more) < 1e-9);
        for (std::size_t i = 0; i < pl.size(); ++i) CHECK(std::abs(a.marg[i] - b.marg[i]) < 1e-9);
        for (std::size_t n = 0; n < prior.size(); ++n) CHECK(std::abs(a.p_count[n] - b.p_count[n]) < 1e-9);
    }
}

static void no_evidence_is_the_prior()
{   // with no evidence the posterior count IS the prior and every place carries E[N] / K
    const auto prior = geometric_count_prior(2.0, 6);
    const std::vector<DoorCountPlace> pl(20, DoorCountPlace{0.f, 0.f});
    const auto post = room_door_posterior(pl, prior);
    double en = 0; for (std::size_t n = 0; n < prior.size(); ++n) { CHECK(std::abs(post.p_count[n] - prior[n]) < 1e-9); en += double(n) * prior[n]; }
    for (const double m : post.marg) CHECK(std::abs(m - en / 20.0) < 1e-9);
}

static void a_room_has_a_door()
{
    const auto prior = geometric_count_prior(2.0, 6);
    CHECK(prior[0] == 0.0);
    std::vector<DoorCountPlace> pl(20, DoorCountPlace{-4.f, 0.f});   // every place strongly refuted
    const auto post = room_door_posterior(pl, prior);
    CHECK(post.p_count[0] == 0.0);
    CHECK(post.p_more > 0.999);   // N >= 1 and nothing found: a door is somewhere
}

static void start_room_phantom_refuted()
{   // clean34: 20 wall places with no evidence (log-odds 0 once the base rate is out), one phantom candidate refuted step by step
    const auto prior = geometric_count_prior(2.0, 6);
    std::vector<DoorCountPlace> pl(21, DoorCountPlace{0.f, 0.f});
    double prev_wall = 0.0;
    for (const float phantom : {2.0f, 0.0f, -2.0f, -5.0f})
    {
        pl[0].log_odds = phantom;
        const auto post = room_door_posterior(pl, prior);
        std::printf("phantom lo %+.1f: p_more %.3f phantom %.3f wall %.3f\n", phantom, post.p_more, post.marg[0], post.marg[1]);
        CHECK(post.p_more > 0.999);
        if (prev_wall > 0.0) CHECK(post.marg[1] >= prev_wall - 1e-12);   // refuting the phantom moves mass onto the walls
        prev_wall = post.marg[1];
    }
}

static void found_door_carries_the_room()
{   // a non-start room: the entry door is found (f = 1, certain); the count prior then says how likely a second door is
    const auto prior = geometric_count_prior(2.0, 6);
    std::vector<DoorCountPlace> pl(15, DoorCountPlace{0.f, 0.f});
    pl[0] = {20.f, 1.f};
    const auto post = room_door_posterior(pl, prior);
    std::printf("entry found: p_more %.3f  P(N=1) %.3f\n", post.p_more, post.p_count[1]);
    CHECK(std::abs(post.p_more - (1.0 - post.p_count[1])) < 1e-6);   // with no evidence elsewhere, "more" = "more than the entry"
    // knowing ONE place is a door size-biases the count: P(N=n | that) ∝ n P(N=n)  (prior[1] / E[N] = 0.267 here)
    double en = 0; for (std::size_t n = 0; n < prior.size(); ++n) en += double(n) * prior[n];
    CHECK(std::abs(post.p_count[1] - prior[1] / en) < 1e-6);
    // and every other place refuted: the room has just its entry door
    for (std::size_t i = 1; i < pl.size(); ++i) pl[i].log_odds = -6.f;
    CHECK(room_door_posterior(pl, prior).p_more < 0.05);
}

static void a_deleted_door_reopens_the_question()
{   // clean31/35: the only real door's place loses its node (f 1 → 0) and its log-odds drops below the delete line;
    // the room still has a door: p_more jumps, and that place keeps the largest marginal
    const auto prior = geometric_count_prior(2.0, 6);
    std::vector<DoorCountPlace> pl(20, DoorCountPlace{-1.f, 0.f});
    pl[3] = {6.f, 1.f};
    const double before = room_door_posterior(pl, prior).p_more;
    pl[3] = {-1.15f + 0.405f, 0.f};   // L = −1.15 with the base −0.405 taken out
    const auto after = room_door_posterior(pl, prior);
    std::printf("deleted door: p_more %.3f -> %.3f\n", before, after.p_more);
    CHECK(after.p_more > 0.999);
    CHECK(after.p_more > before);
}

static void crossed_door_is_certain()
{
    const auto prior = geometric_count_prior(2.0, 6);
    std::vector<DoorCountPlace> pl(5, DoorCountPlace{-2.f, 0.f});
    pl[2].log_odds = std::numeric_limits<float>::infinity();
    const auto post = room_door_posterior(pl, prior);
    CHECK(post.marg[2] > 1.0 - 1e-9);
}

static void look_value()
{
    CHECK(look_value_s(300.0, 0.5, 0.0, 0.9, 0.6) == 300.0 * 0.5 * 0.9 * 0.6);
    CHECK(look_value_s(300.0, 0.9, 1.0, 0.9, 0.6) == 0.0);   // a found door: nothing left to find there
}

static void count_prior_from_memory()
{
    const std::vector<int> counts = {1, 2, 2};
    const auto p = count_prior_from_rooms(counts, 2.0, 6);
    double s = 0; for (const double x : p) s += x;
    CHECK(std::abs(s - 1.0) < 1e-12);
    CHECK(p[0] == 0.0);
    CHECK(p[2] > p[3]);
}

int main()
{
    agrees_with_brute_force();
    no_evidence_is_the_prior();
    a_room_has_a_door();
    start_room_phantom_refuted();
    found_door_carries_the_room();
    a_deleted_door_reopens_the_question();
    crossed_door_is_certain();
    look_value();
    count_prior_from_memory();
    std::printf(g_fail == 0 ? "test_room_door_count: ALL PASSED\n" : "test_room_door_count: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
