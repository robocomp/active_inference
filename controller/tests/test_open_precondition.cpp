// Task O fix round 2: `open`'s in-band clause is a PRECONDITION (where the robot stands), not part of the completion predicate.
//   g++ -std=c++23 -O0 -fPIC -I/usr/include/eigen3 -I../../common -I../../door_concept/src $(pkg-config --cflags Qt6Core) \
//       -o /tmp/top test_open_precondition.cpp -ldsr_api -ldsr_core -lfastcdr -lfastdds $(pkg-config --libs Qt6Core) && /tmp/top
#include "affordance_protocol/affordance_protocol.h"
#include "door_pragmatics.h"
#include <cstdio>
static int fails = 0;
#define CHECK(name, cond) do { const bool ok_ = (cond); std::printf("%s %s\n", ok_ ? "PASS" : "FAIL", name); if (not ok_) ++fails; } while (0)
static void put(DSR::Node& n, const char* k, float v) { n.attrs()[k] = DSR::Attribute(DSR::Value(v), 0, 1); }
int main()
{
    using namespace rc::affordance;
    const auto c = rc::door::open_contract(rc::DoorPragmaticCfg{});
    CHECK("open: the in-band clause is a precondition, not a goal clause",
          c.pre.size() == 1 and c.pre[0].attr == "door_reach_prob" and std::ranges::none_of(c.goal, [](auto& g) { return g.attr == "door_reach_prob"; }));
    DSR::Node fb;
    put(fb, "door_open_prob", 0.9f);
    put(fb, "door_reach_prob", -1.f);   // kUnmeasured sentinel
    CHECK("open: goal_met-style evaluate_goal ignores an UNMEASURED door_reach_prob", evaluate_goal(fb, c.goal));
    put(fb, "door_open_prob", 0.2f);
    CHECK("open: still not done while the leaf is shut", not evaluate_goal(fb, c.goal));
    // wire: the precondition rides the goal arrays prefixed and is split out by read_contract
    DSR::Node aff;
    aff.attrs()["aff_goal_attrs"] = DSR::Attribute(DSR::Value(std::string("pre:door_reach_prob|door_open_prob")), 0, 1);
    aff.attrs()["aff_goal_ops"] = DSR::Attribute(DSR::Value(std::string("ge|ge")), 0, 1);
    aff.attrs()["aff_goal_values"] = DSR::Attribute(DSR::Value(std::vector<float>{0.9f, 0.7f}), 0, 1);
    const auto r = read_contract(aff);
    CHECK("wire: read_contract splits pre: out of the goal", r.pre.size() == 1 and r.pre[0].attr == "door_reach_prob" and r.goal.size() == 1 and r.goal[0].attr == "door_open_prob");
    std::printf("test_open_precondition: %d failures\n", fails);
    return fails != 0;
}
