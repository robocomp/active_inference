// The room stops at its doors (FreeSpacePolygon::set_openings): a beam from inside the room through the door's aperture
// records free space up to the aperture and nothing beyond; a beam from beyond the plane is not cut; a shut door (the beam
// ends before the plane) is unaffected.
//   g++ -std=c++23 -O2 -Isrc -I/usr/include/eigen3 tools/free_space_door_cut_test.cpp -o /tmp/fsdc && /tmp/fsdc
#include "free_space_polygon.h"

#include <cstdio>

int main()
{
    using rc::freespace::FreeSpacePolygon;
    int fail = 0;
    const auto check = [&](bool ok, const char* what) { if (not ok) { ++fail; std::printf("FAIL: %s\n", what); } };
    // room y < 2, door aperture x in [-0.5, 0.5] on y = 2, n_out = +y
    {
        FreeSpacePolygon fs;
        fs.set_openings({{Eigen::Vector2f(-0.5f, 2.f), Eigen::Vector2f(0.5f, 2.f), Eigen::Vector2f(0.f, 1.f)}});
        fs.add_scan(Eigen::Vector2f(0.f, 0.f), {Eigen::Vector2f(0.f, 5.f)});   // straight through the door to a wall at y 5
        check(fs.state_at(Eigen::Vector2f(0.f, 1.5f)) == 1, "free before the door");
        check(fs.state_at(Eigen::Vector2f(0.f, 3.f)) == 0, "unknown beyond the door (not this room's floor)");
        check(fs.state_at(Eigen::Vector2f(0.f, 5.f)) == 0, "no wall marked in the next room");
        fs.add_scan(Eigen::Vector2f(0.f, 0.f), {Eigen::Vector2f(1.5f, 2.f)});   // to the wall beside the door: not cut
        check(fs.state_at(Eigen::Vector2f(1.5f, 2.f)) == 2, "the wall beside the door is matter");
    }
    {   // a beam that starts beyond the plane (the robot in the doorway / the next room) is not cut by this door
        FreeSpacePolygon fs;
        fs.set_openings({{Eigen::Vector2f(-0.5f, 2.f), Eigen::Vector2f(0.5f, 2.f), Eigen::Vector2f(0.f, 1.f)}});
        fs.add_scan(Eigen::Vector2f(0.f, 2.3f), {Eigen::Vector2f(0.f, 5.f)});
        check(fs.state_at(Eigen::Vector2f(0.f, 3.5f)) == 1, "a beam from beyond the plane records its free space");
    }
    {   // no opening known (n_out zero): not cut
        FreeSpacePolygon fs;
        fs.set_openings({{Eigen::Vector2f(-0.5f, 2.f), Eigen::Vector2f(0.5f, 2.f), Eigen::Vector2f(0.f, 0.f)}});
        fs.add_scan(Eigen::Vector2f(0.f, 0.f), {Eigen::Vector2f(0.f, 5.f)});
        check(fs.state_at(Eigen::Vector2f(0.f, 3.f)) == 1, "an unseated aperture does not cut");
    }
    std::printf("free_space_door_cut_test: %s\n", fail == 0 ? "ALL PASS" : "FAILURES");
    return fail == 0 ? 0 : 1;
}
