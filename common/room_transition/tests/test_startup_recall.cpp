// startup_recall.h: the warm start's start-up recall (docs/superpowers/specs/2026-11-01-ltsm-warm-start-design.md) — one
// definition for ltsm (posts it), room_concept (adopts its outline) and door_concept (lets it announce the first room).
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 -o /tmp/tsr test_startup_recall.cpp && /tmp/tsr
#include "../startup_recall.h"

#include <cmath>
#include <print>
#include <vector>

namespace
{
int failures = 0, checks = 0;
void check(bool ok, const char* what)
{
    ++checks;
    if (not ok) { ++failures; std::println("FAIL: {}", what); }
}
}  // namespace

int main()
{
    using rc::room_transition::is_startup_recall;
    using rc::room_transition::recall_outline;
    check(is_startup_recall("room_recall", "robot", false), "a room recall hung from the robot, not late, is a start-up recall");
    check(not is_startup_recall("room_recall", "robot", true), "a late recall is not a start-up recall");
    check(not is_startup_recall("room_recall", "object", false), "a crossing recall (hung from a door) is not a start-up recall");
    check(not is_startup_recall("passage", "robot", false), "another subtype is not a recall");
    check(rc::room_transition::kRecallSubtype == "room_recall", "the subtype ltsm writes");

    const std::vector<float> xs{0.f, 4.f, 4.f}, ys{0.f, 0.f, 3.f};
    const auto o = recall_outline(xs, ys);
    check(o.has_value() and o->size() == 3, "three vertices make an outline");
    check(o.has_value() and o->at(1).isApprox(Eigen::Vector2f(4.f, 0.f)), "vertices keep their order");
    check(not recall_outline(std::vector<float>{0.f, 4.f}, std::vector<float>{0.f, 0.f}).has_value(), "two vertices are no outline");
    check(not recall_outline(xs, std::vector<float>{0.f, 0.f}).has_value(), "x and y of different sizes are no outline");
    check(not recall_outline(std::vector<float>{0.f, NAN, 4.f}, ys).has_value(), "a non-finite vertex is no outline");

    std::println("{} / {} checks passed", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
