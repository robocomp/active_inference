// Build (from room_concept/):
//   g++ -std=c++23 -O2 -Isrc -I/usr/include/eigen3 tools/exclusion_sliver_test.cpp -o /tmp/est && /tmp/est
//
// explore_tour15: room_1's refined outline carried a 0.3 m x 2.6 m spike through DOOR_0 into the bathroom (an 'extend'
// into free cells seen through a doorway believed closed). Excluded from room_2's free-space map as "another room", the
// spike stopped every beam that crossed it: room_2 saw only the 1.6 m2 pocket on the robot's side, for 18 minutes.
// A part of an outline narrower than the body the ring is eroded by is not room the robot can be in: the exclusion is
// the outline OPENED by that same radius. The left room's real floor must still stop the beams that look into it.
#include "free_space_polygon.h"

#include <clocale>
#include <print>

namespace
{
int failures = 0, checks = 0;
void check(bool ok, int line, const char* what)
{
    ++checks;
    if (not ok) { ++failures; std::println("FAIL line {}: {}", line, what); }
}
#define CHECK(x) check((x), __LINE__, #x)
}


void outline_order_excludes_just_left_only_when_robot_outside()
{
    using Poly = std::vector<Eigen::Vector2f>;
    const Poly L{{-3.f, -3.f}, {3.f, -3.f}, {3.f, 3.f}, {-3.f, 3.f}}, other{{5.f, -3.f}, {9.f, -3.f}, {9.f, 3.f}, {5.f, 3.f}};
    const Eigen::Matrix2f S = Eigen::Matrix2f::Identity() * 0.0064f;   // sigma 8 cm, the pose covariance rearm_after_crossing uses
    {   // clean16: the robot 0.5 m INSIDE the outline just left: it is the room the robot is still in, not "another room"
        std::vector<Poly> ex = {other};
        const auto n = rc::freespace::exclude_left_room(ex, L, Eigen::Vector2f(2.5f, 0.f), S);
        check(n == 1 and ex.size() == 1 and ex[0] == other, __LINE__, "robot inside the room just left: its outline is not excluded");
    }
    {   // a normal crossing: the robot 0.5 m OUTSIDE the outline just left: excluded as ever
        std::vector<Poly> ex = {other};
        const auto n = rc::freespace::exclude_left_room(ex, L, Eigen::Vector2f(3.5f, 0.f), S);
        check(n == 0 and ex.size() == 2 and ex[1] == L, __LINE__, "robot outside the room just left: its outline is excluded");
    }
    {   // the earlier-outline rule is unchanged: an earlier outline holding the robot is dropped, the just-left stays out
        std::vector<Poly> ex = {other};
        const auto n = rc::freespace::exclude_left_room(ex, L, Eigen::Vector2f(7.f, 0.f), S);
        check(n == 1 and ex.size() == 1 and ex[0] == L, __LINE__, "an earlier outline holding the robot is dropped as before");
    }
}

int main()
{
    outline_order_excludes_just_left_only_when_robot_outside();
    std::setlocale(LC_ALL, "");
    using V = Eigen::Vector2f;
    // the room left (x < 0) with a spike poking 2.6 m into the new room (x > 0) along y = 0, 0.3 m wide
    const std::vector<V> left{{-5, -3}, {0, -3}, {0, -0.15f}, {2.6f, -0.15f}, {2.6f, 0.15f}, {0, 0.15f}, {0, 3}, {-5, 3}};
    rc::freespace::FreeSpacePolygon fs;
    fs.set_exclusion({left});
    const V origin(1.3f, -1.f);   // in the new room, beside the spike
    fs.add_scan(origin, {V(1.3f, 1.8f),     // across the spike, to the far side of the new room
                         V(-3.f, -1.f)});   // into the room left, through its real boundary at x = 0
    CHECK(fs.state_at(V(1.3f, 1.0f)) == 1);    // beyond the spike: the new room's floor, seen
    CHECK(fs.state_at(V(1.3f, 0.0f)) == 1);    // the spike itself is no other room's floor
    CHECK(fs.state_at(V(1.3f, 1.8f)) == 2);    // the beam's endpoint is matter
    CHECK(fs.state_at(V(-1.f, -1.f)) != 1);    // the room left still stops the beam that enters it
    CHECK(fs.state_at(V(-3.f, -1.f)) != 1);
    CHECK(fs.state_at(V(0.5f, -1.f)) == 1);    // ...after the new room's own floor up to it
    std::println("exclusion_sliver_test: {}/{} checks passed", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
