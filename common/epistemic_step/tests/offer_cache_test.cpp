// offer_cache_test.cpp — the offer is computed once per belief update serial (and standing-world version), never during a
// cooldown, and the held proposal is what epistemic::step publishes meanwhile (engine1 diagnosis item 3; plan rev 2 F1.1).
// Run: ./offer_cache_test
#include <cstdio>
#include <cstdint>
#include <optional>
#include <string>

#include "../epistemic_step.h"
#include "../offer_cache.h"

static int fails = 0;
#define CHECK(cond, msg) do { const bool ok_ = (cond); std::printf("  %-70s %s\n", msg, ok_ ? "ok" : "FAIL"); if (not ok_) ++fails; } while (0)

namespace
{
struct FakeAffordance
{
    rc::AffordanceTarget last; int updates = 0, holds = 0;
    std::uint64_t node_id() const { return 0; }
    void hold_offered() { ++holds; }
    void update(const rc::AffordanceTarget& t, bool) { last = t; ++updates; }
};
struct FakeGraph
{
    std::optional<int> get_node(std::uint64_t) { return std::nullopt; }
    template <class Att> std::optional<bool> get_attrib_by_name(const int&) { return std::nullopt; }
};
struct FakeInst { int epistemic_cooldown = 0; bool epistemic_pending = false; std::string node_name = "fake"; FakeAffordance affordance; };
struct Prop { float epistemic_target_x_m = 1, epistemic_target_y_m = 2, epistemic_target_yaw_rad = 0.5f, epistemic_gain = 0.0f;
              bool valid = true; };
}

int main()
{
    std::printf("offer_cache_test\n");
    // The cache alone.
    {
        rc::epistemic::OfferCache<Prop> c;
        float next = 1.0f;
        const auto compute = [&]() -> std::optional<Prop> { Prop p; p.epistemic_gain = next; next += 1.0f; return p; };
        auto a = c.get(1, 7, false, compute);
        auto b = c.get(1, 7, false, compute);
        CHECK(c.computes == 1 and a->epistemic_gain == 1.0f and b->epistemic_gain == 1.0f, "same serial, same world: computed once, held");
        auto d = c.get(2, 7, false, compute);
        CHECK(c.computes == 2 and d->epistemic_gain == 2.0f, "a new update serial: recomputed");
        auto e = c.get(2, 8, false, compute);
        CHECK(c.computes == 3 and e->epistemic_gain == 3.0f, "a new standing-world version: recomputed");
        auto f = c.get(9, 9, true, compute);
        CHECK(c.computes == 3 and f->epistemic_gain == 3.0f, "cooldown: held whatever changed, no compute");
        auto g = c.get(9, 9, false, compute);
        CHECK(c.computes == 4 and g->epistemic_gain == 4.0f, "cooldown over and the key moved: recomputed");
    }
    {
        rc::epistemic::OfferCache<Prop> c;
        int n = 0;
        const auto none = [&]() -> std::optional<Prop> { ++n; return std::nullopt; };
        c.get(1, 1, false, none);
        c.get(1, 1, false, none);
        CHECK(n == 2, "no proposal is not held: asked again on the same key");
        c.get(1, 1, true, none);
        CHECK(n == 3, "cooldown with nothing held: computes (nothing to hold)");
    }
    // Through the real step(): 10 ticks, the belief updates at ticks 0 and 6, a cooldown is running over ticks 2..5.
    {
        FakeInst inst; FakeGraph graph;
        rc::epistemic::OfferCache<Prop> c;
        rc::epistemic::StepInputs in;
        long serial = 1; int computes = 0; float published_in_cooldown = -1.0f;
        rc::epistemic::StepHooks<Prop> hooks;
        hooks.compute = [&]() -> std::optional<Prop>
        { return c.get(serial, 0, inst.epistemic_cooldown > 0, [&]() -> std::optional<Prop> { ++computes; Prop p; p.epistemic_gain = 0.7f; return p; }); };
        for (int t = 0; t < 10; ++t)
        {
            if (t == 6) ++serial;
            if (t == 2) inst.epistemic_cooldown = 5;           // decremented to 4 by step() before compute
            if (t == 4) ++serial;                               // an update DURING the cooldown: still no compute
            rc::epistemic::step(inst, graph, in, hooks);
            if (t == 3) published_in_cooldown = inst.affordance.last.gain;
        }
        CHECK(computes == 2, "10 ticks, updates at 4 (in cooldown) and 6: computed at ticks 0 and 6 only");
        CHECK(inst.affordance.updates == 10, "the affordance is refreshed every tick from the held proposal");
        CHECK(published_in_cooldown == 0.0f, "during the cooldown the held proposal is published at gain 0");
        CHECK(inst.affordance.last.gain == 0.7f, "after it, at the planner's gain");
    }
    std::printf("%s (%d failed)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
