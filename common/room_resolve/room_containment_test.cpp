// g++ -std=c++23 -O2 -I/usr/include/eigen3 common/room_resolve/room_containment_test.cpp -o /tmp/room_containment_test && /tmp/room_containment_test
//
// The shared birth rule for countable concept agents (room_containment.h), pinned without a graph:
// inside the outline -> Admit; within the object's physical half-depth of it -> Admit (wall-mounted things);
// outside -> Suppress; no usable outline -> Defer (WAIT, never guess). Plus the dead-parent case of left_behind.
#include "room_containment.h"

#include <cstdio>
#include <optional>
#include <vector>

namespace
{
    int fails = 0;
    void check(const char* what, bool ok)
    {
        fails += ok ? 0 : 1;
        std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    }
}   // namespace

int main()
{
    using rc::room::BirthVerdict;
    using V = Eigen::Vector2f;
    const std::vector<V> square{{0, 0}, {1, 0}, {1, 1}, {0, 1}};

    check("birth_admits_inside", rc::room::birth_verdict(square, {0.5f, 0.5f}, 0.f) == BirthVerdict::Admit);
    check("birth_suppresses_outside", rc::room::birth_verdict(square, {2.f, 0.5f}, 0.1f) == BirthVerdict::Suppress);
    check("birth_wall_object_admitted", rc::room::birth_verdict(square, {0.5f, 1.03f}, 0.05f) == BirthVerdict::Admit);
    check("birth_defers_without_outline", rc::room::birth_verdict({}, {0.5f, 0.5f}, 0.1f) == BirthVerdict::Defer);
    const std::vector<V> two{{0, 0}, {1, 0}};
    check("birth_defers_on_degenerate_outline", rc::room::birth_verdict(two, {0.5f, 0.5f}, 0.1f) == BirthVerdict::Defer);
    check("left_behind_dead_parent", rc::room::left_behind(std::optional<std::uint64_t>{77}, 5));
    check("out_of_room_term", rc::room::out_of_room_logodds(true, 1.5f) == 0.f
                              and rc::room::out_of_room_logodds(false, 1.5f) == -1.5f);

    std::printf(fails ? "room_containment_test: %d FAILED\n" : "room_containment_test: all passed\n", fails);
    return fails ? 1 : 0;
}
