// Stability belief: flapping-but-mostly-stable births in bounded frames; genuinely unstable never does.
#include "stability_belief.h"
#include <cstdio>
#include <vector>
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)
// first frame (1-based) at which the sequence turns decisive, or -1
static int first_birth(rc::StabilityBelief b, const std::vector<bool>& s)
{
    for (std::size_t i = 0; i < s.size(); ++i) if (b.update(s[i])) return static_cast<int>(i) + 1;
    return -1;
}
int main()
{
    const rc::StabilityBelief proto(30);
    // 1. N clean frames births exactly at N (the old behaviour is the limiting case).
    CHECK(first_birth(proto, std::vector<bool>(30, true)) == 30);
    CHECK(first_birth(proto, std::vector<bool>(29, true)) == -1);
    // 2. gold10-like: runs of 6 stable broken by a single unstable frame. The old counter never passed 6.
    std::vector<bool> flap;
    for (int i = 0; i < 400; ++i) flap.push_back(i % 7 != 6);
    const int t = first_birth(proto, flap);
    CHECK(t > 0 and t <= 120);
    std::printf("flap(6 on,1 off) births at frame %d\n", t);
    // 3. gold7-like: 10 stable, 1 bad, 10 stable ... still births.
    std::vector<bool> g7;
    for (int i = 0; i < 600; ++i) g7.push_back(i % 11 != 10);
    CHECK(first_birth(proto, g7) > 0);
    // 4. Genuinely unstable: stable only half the time (and 1/3 of the time) never births.
    std::vector<bool> half, third;
    for (int i = 0; i < 5000; ++i) { half.push_back(i % 2 == 0); third.push_back(i % 3 == 0); }
    CHECK(first_birth(proto, half) == -1);
    CHECK(first_birth(proto, third) == -1);
    // 5. One flap does not erase accumulated evidence.
    rc::StabilityBelief b(30);
    for (int i = 0; i < 20; ++i) b.update(true);
    const double before = b.log_odds();
    b.update(false);
    CHECK(b.log_odds() > 0.5 * before);
    // 6. Decays toward the prior with no further evidence is not modelled; all-unstable goes negative.
    for (int i = 0; i < 100; ++i) b.update(false);
    CHECK(b.posterior() < 0.5);
    std::printf(fails ? "FAILED\n" : "OK\n");
    return fails;
}
