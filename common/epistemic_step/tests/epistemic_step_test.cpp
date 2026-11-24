// epistemic_step_test.cpp — G6 through rc::epistemic::step: a blocked standpoint (p_stand = 0) is published at gain 0 even under a
// verification floor, an unblocked one is published at the floored gain, and a proposal with no p_stand is untouched.
// The step is a template over the instance and the graph, so a stand-in instance with a recording affordance and a stand-in graph run
// the REAL step(); the affordance node id stays 0, so the graph is never asked anything.
#include <cstdio>
#include <cstdint>

#include "../epistemic_step.h"

static int fails = 0;
#define CHECK(cond, msg) do { const bool ok_ = (cond); std::printf("  %-66s %s\n", msg, ok_ ? "ok" : "FAIL"); if (not ok_) ++fails; } while (0)

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
struct ProposalWith { float epistemic_target_x_m = 1, epistemic_target_y_m = 2, epistemic_target_yaw_rad = 0.5f, epistemic_gain = 0.0f;
                      bool valid = true; float p_stand = 1.0f; };
struct ProposalWithout { float epistemic_target_x_m = 1, epistemic_target_y_m = 2, epistemic_target_yaw_rad = 0.5f, epistemic_gain = 0.0f;
                         bool valid = true; };

template <class P>
float run(float gain, float floor_gain, float p, FakeInst& inst)
{
    P prop; prop.epistemic_gain = gain;
    if constexpr (requires { prop.p_stand; }) prop.p_stand = p;
    rc::epistemic::StepInputs in; in.verify_gain_floor = floor_gain;
    rc::epistemic::StepHooks<P> hooks; hooks.compute = [&]() -> std::optional<P> { return prop; };
    FakeGraph graph;
    rc::epistemic::step(inst, graph, in, hooks);
    return inst.affordance.last.gain;
}
}

int main()
{
    std::printf("epistemic_step_test\n");
    { FakeInst i; CHECK(run<ProposalWith>(0.0f, 5.0f, 0.0f, i) == 0.0f and i.affordance.updates == 1,
                        "blocked standpoint under verify floor 5: published at gain 0 (not skipped)"); }
    { FakeInst i; CHECK(run<ProposalWith>(0.3f, 5.0f, 0.0f, i) == 0.0f, "blocked, raw gain 0.3, floor 5: still 0"); }
    { FakeInst i; CHECK(run<ProposalWith>(0.3f, 5.0f, 1.0f, i) == 5.0f, "free standpoint: the floor applies, published 5"); }
    { FakeInst i; CHECK(run<ProposalWith>(0.3f, 0.0f, 1.0f, i) == 0.3f, "free, no floor: the planner's gain"); }
    { FakeInst i; CHECK(run<ProposalWithout>(0.3f, 5.0f, 1.0f, i) == 5.0f, "a proposal without p_stand is untouched"); }
    std::printf("%s (%d failed)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
