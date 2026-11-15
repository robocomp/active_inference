// Standalone test for identity_link.h (common/run_tests.sh).
#include "identity_link.h"

#include <cmath>
#include <print>

using namespace rc::recall;

static int failures = 0;
#define CHECK(c) do { if (not (c)) { std::println(stderr, "FAIL {}:{}: {}", __FILE__, __LINE__, #c); ++failures; } } while (0)

// A 12-term colour posterior: `p` on `name`, the rest shared equally.
static std::array<float, 12> colour(std::string_view name, float p)
{
    std::array<float, 12> q{};
    for (std::size_t i = 0; i < q.size(); ++i)
        q[i] = rc::appearance::kPalette[i].name == name ? p : (1.f - p) / 11.f;
    return q;
}
static LinkCand cand(std::uint64_t uid, std::string name, float prob, float llr, float exist = 1.f)
{
    return {uid, std::log(exist), std::move(name), prob, llr};
}

static void moved_magenta_old_spot_empty()
{
    const std::vector<LinkCand> c{cand(50, "magenta", 0.74f, -3.f)};
    const auto l = link_posterior(c, colour("magenta", 0.74f), LinkParams{});
    CHECK(l.index.has_value() and *l.index == 0);
    CHECK(l.posterior[0] > 0.95f);
}
static void moved_old_spot_unseen_still_links()
{
    const std::vector<LinkCand> seen{cand(50, "magenta", 0.74f, -3.f)};
    const std::vector<LinkCand> unseen{cand(50, "magenta", 0.74f, 0.f)};
    const auto ls = link_posterior(seen, colour("magenta", 0.74f), LinkParams{});
    const auto lu = link_posterior(unseen, colour("magenta", 0.74f), LinkParams{});
    CHECK(lu.index.has_value() and *lu.index == 0);
    CHECK(lu.posterior[0] < ls.posterior[0]);
}
static void colour_mismatch_is_new()
{
    // Confident colours: a 0.74 memory is weak colour evidence by its own account and cannot overrule the
    // count + empty-spot prior (ledger ruling, Task 1).
    const std::vector<LinkCand> c{cand(50, "magenta", 0.95f, -3.f)};
    const auto l = link_posterior(c, colour("green", 0.95f), LinkParams{});
    CHECK(not l.index.has_value());
}
static void colourless_now_is_no_evidence()
{
    CHECK(colour_log_ratio(std::nullopt, "magenta", 0.74f) == 0.f);
    CHECK(colour_log_ratio(colour("magenta", 0.74f), "", 0.f) == 0.f);
}
static void link_prefers_coloured_believed_recall()
{
    const std::vector<LinkCand> c{cand(50, "magenta", 0.74f, -3.f, 0.9f), cand(44, "", 0.f, -3.f, 0.06f)};
    const auto l = link_posterior(c, colour("magenta", 0.74f), LinkParams{});
    CHECK(l.index.has_value() and *l.index == 0);
    CHECK(l.posterior[1] < 0.05f);
}
static void two_identical_candidates_split()
{
    const std::vector<LinkCand> c{cand(50, "magenta", 0.74f, -3.f), cand(51, "magenta", 0.74f, -3.f)};
    const auto l = link_posterior(c, colour("magenta", 0.74f), LinkParams{});
    CHECK(std::abs(l.posterior[0] - l.posterior[1]) < 1e-6f);
}
static void no_candidates()
{
    const auto l = link_posterior({}, colour("magenta", 0.74f), LinkParams{});
    CHECK(not l.index.has_value());
    CHECK(l.posterior.size() == 1 and l.posterior[0] == 1.f);
}

// Final-review fixes F4 / F1: the old spot's emptiness is THIS visit's evidence only (an even prior), and a chair
// seen standing at the spot confirms the ghost, it does not merely hold it.
static void left_spot_is_this_visits_evidence()
{
    CHECK(left_spot_prob(-2.75f, -2.75f) == 0.5f);          // a memory that died at P 0.06, nothing seen yet: undecided
    CHECK(left_spot_prob(2.2f - 3.f, 2.2f) > 0.9f);         // looked, nothing there
    CHECK(left_spot_prob(2.2f + 1.f, 2.2f) < 0.5f);         // looked, something there
}
static void ghost_evidence_confirms_when_seen()
{
    const auto absent = ghost_evidence(0.8f, false, 0.15f);
    CHECK(absent.p_vis == 0.8f and absent.ratio == -0.15f);
    const auto there = ghost_evidence(0.8f, true, 0.15f);
    CHECK(there.p_vis == 0.8f and there.ratio == 0.15f);
    const auto unseen = ghost_evidence(0.f, true, 0.15f);
    CHECK(unseen.p_vis == 0.f);                              // out of view: HOLD
}

int main()
{
    left_spot_is_this_visits_evidence();
    ghost_evidence_confirms_when_seen();
    moved_magenta_old_spot_empty();
    moved_old_spot_unseen_still_links();
    colour_mismatch_is_new();
    colourless_now_is_no_evidence();
    link_prefers_coloured_believed_recall();
    two_identical_candidates_split();
    no_candidates();
    if (failures == 0) std::println("identity_link_test: all passed");
    return failures == 0 ? 0 : 1;
}
