// Standalone test for recall_match.h (common/run_tests.sh).
#include "recall_match.h"

#include <cmath>
#include <print>

using namespace rc::recall;

static int failures = 0;
#define CHECK(c) do { if (not (c)) { std::println(stderr, "FAIL {}:{}: {}", __FILE__, __LINE__, #c); ++failures; } } while (0)

static Candidate cand(std::uint64_t uid, float x, float y, float sigma = 0.1f)
{
    return {uid, Eigen::Vector2f(x, y), Eigen::Matrix2f::Identity() * sigma * sigma};
}
static const Eigen::Matrix2f kMeas = Eigen::Matrix2f::Identity() * 0.01f;

static void test_close_candidate_wins()
{
    const std::vector<Candidate> c{cand(7, 1.f, 1.f)};
    const auto d = decide({1.1f, 1.f}, kMeas, c, Params{});
    CHECK(d.index.has_value() and *d.index == 0);
}
static void test_far_candidate_loses()
{
    const std::vector<Candidate> c{cand(7, 1.f, 1.f)};
    const auto d = decide({4.f, 1.f}, kMeas, c, Params{});
    CHECK(not d.index.has_value());
}
static void test_nearest_of_two()
{
    const std::vector<Candidate> c{cand(1, 0.f, 0.f), cand(2, 0.5f, 0.f)};
    const auto d = decide({0.45f, 0.f}, kMeas, c, Params{});
    CHECK(d.index.has_value() and *d.index == 1);
    const auto e = decide({0.05f, 0.f}, kMeas, c, Params{});
    CHECK(e.index.has_value() and *e.index == 0);
}
static void test_no_candidates_is_new()
{
    const auto d = decide({0.f, 0.f}, kMeas, {}, Params{});
    CHECK(not d.index.has_value());
    CHECK(d.posterior.size() == 1 and std::abs(d.posterior[0] - 1.f) < 1e-6f);
}
static void test_pi_move_one_never_resurrects()
{
    const std::vector<Candidate> c{cand(1, 0.f, 0.f), cand(2, 3.f, 0.f)};
    Params p; p.pi_move = 1.f; p.log_prior_new = 0.f;
    const auto d = decide({0.f, 0.f}, kMeas, c, p);
    CHECK(not d.index.has_value());
    CHECK(d.posterior.size() == 3);
    CHECK(d.posterior[0] == d.posterior[2]);
    CHECK(d.posterior[1] == d.posterior[2]);
}
static void test_pi_move_one_never_resurrects_any_area()
{
    const std::vector<Candidate> c{cand(1, 0.f, 0.f)};
    for (int i = 10; i <= 2000; ++i)
    {
        Params p; p.pi_move = 1.f; p.log_prior_new = 0.f; p.room_area_m2 = float(i) * 0.1f;
        const auto d = decide({0.f, 0.f}, kMeas, c, p);
        CHECK(not d.index.has_value());
        if (d.index.has_value()) { std::println(stderr, "area {}", p.room_area_m2); break; }
    }
}
static void test_posterior_sums_to_one()
{
    const std::vector<Candidate> c{cand(1, 0.f, 0.f), cand(2, 0.5f, 0.f), cand(3, 2.f, 2.f)};
    const auto d = decide({0.3f, 0.1f}, kMeas, c, Params{});
    float s = 0.f;
    for (float v : d.posterior) s += v;
    CHECK(d.posterior.size() == 4 and std::abs(s - 1.f) < 1e-5f);
}

static void test_choose_recall_takes_once()
{
    std::vector<rc::RecallCand> cands(2);
    cands[0].n = 1; cands[0].cand = cand(11, 1.f, 1.f, std::sqrt(0.02f));
    cands[1].n = 2; cands[1].cand = cand(22, 4.f, 4.f, std::sqrt(0.02f));
    Params p{0.5f, 0.0f}; p.room_area_m2 = 20.f;
    const auto a = rc::choose_recall({1.f, 1.f}, 0.35f, cands, p);
    CHECK(a.has_value() and a->who.uid == 11 and a->who.n == 1 and a->p > 0.5f);
    CHECK(cands.size() == 1);
    const auto b = rc::choose_recall({1.f, 1.f}, 0.35f, cands, p);
    CHECK(not b.has_value() or b->who.uid != 11);
}
static void test_choose_recall_empty()
{
    std::vector<rc::RecallCand> cands;
    CHECK(not rc::choose_recall({0.f, 0.f}, 0.35f, cands, Params{}).has_value());
}

int main()
{
    test_close_candidate_wins();
    test_far_candidate_loses();
    test_nearest_of_two();
    test_no_candidates_is_new();
    test_pi_move_one_never_resurrects();
    test_pi_move_one_never_resurrects_any_area();
    test_posterior_sums_to_one();
    test_choose_recall_takes_once();
    test_choose_recall_empty();
    if (failures) return 1;
    std::println("recall_match_test: ok");
    return 0;
}
