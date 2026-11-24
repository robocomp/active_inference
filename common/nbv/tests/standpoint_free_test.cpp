// standpoint_free_test.cpp — can the body stand at a standpoint (common/nbv/standpoint_free.h): the controller's own
// pose_free / can_turn_here test, mirrored on residual's latched cells + room polygon + obstacles. Standalone.
#include <cmath>
#include <cstdio>
#include <vector>

#include "../standpoint_free.h"

static int fails = 0;
#define CHECK(cond, msg) do { const bool ok_ = (cond); std::printf("  %-70s %s\n", msg, ok_ ? "ok" : "FAIL"); if (not ok_) ++fails; } while (0)

int main()
{
    using namespace rc::nbv;
    constexpr float kPi2 = 1.57079633f, cell = 0.05f;
    const rc::RobotFootprint shadow = rc::RobotFootprint::shadow();   // 0.284 m lateral (x), 0.230 m fore/aft (y) half-extents
    const StandBody body = StandBody::of(shadow, cell);
    const std::vector<Eigen::Vector2f> room{{-3, -3}, {3, -3}, {3, 3}, {-3, 3}};
    const auto cells_at = [&](std::vector<float> xy) {   // xy pairs -> [x,y,z] triples
        std::vector<float> f; for (std::size_t i = 0; i + 1 < xy.size(); i += 2) { f.push_back(xy[i]); f.push_back(xy[i + 1]); f.push_back(0.02f); }
        return OccupiedCells::from_xyz(f, cell); };
    std::printf("standpoint_free_test\n");

    {   // an empty world stands anywhere inside the room, in every mode
        const OccupiedCells none;
        const StandWorld w{&body, &none, room, {}};
        CHECK(p_stand(w, {0, 0}, 0.0f, false) == 1.0 and p_stand(w, {0, 0}, 0.3f, true) == 1.0, "empty world: free, facing and turning");
    }
    {   // a latched cell under the body blocks; the same cell 1 m away does not
        const OccupiedCells under = cells_at({0.0f, 0.0f});
        const StandWorld w{&body, &under, room, {}};
        CHECK(p_stand(w, {0, 0}, 0.0f, false) == 0.0, "a latched cell under the body: 0");
        CHECK(p_stand(w, {1.0f, 0}, 0.0f, false) == 1.0 and p_stand(w, {1.0f, 0}, 0.0f, true) == 1.0, "the same cell 1 m away: 1");
    }
    {   // a cell at x=0.25 (spans 0.275..0.325): the body reaches 0.284 along its lateral axis and 0.23 along forward. The first cell
        // fixes the lattice origin, so a far one is listed first to put the pose (0,0) on a cell centre.
        const OccupiedCells side = cells_at({2.5f, 2.5f, 0.25f, 0.0f});
        const StandWorld w{&body, &side, room, {}};
        CHECK(p_stand(w, {0, 0}, 0.0f, false) == 1.0, "yaw 0 (forward +x, 0.23 reach): the cell at x=0.25 is clear");
        CHECK(p_stand(w, {0, 0}, kPi2, false) == 0.0, "yaw pi/2 (lateral axis on room x, 0.284 reach): the same cell blocks");
        CHECK(p_stand(w, {0, 0}, 0.0f, true) == 0.0, "turning in place there: blocked at some heading (can_turn_here)");
        CHECK(p_stand(w, {0, 0}, 2.0f * 3.14159265f, false) == 1.0, "yaw 2pi buckets to heading 0");
    }
    {   // the room polygon: a body half out of the room is blocked
        const OccupiedCells none;
        const StandWorld w{&body, &none, room, {}};
        CHECK(p_stand(w, {2.95f, 0}, 0.0f, false) == 0.0, "body across the room edge: 0");
        CHECK(p_stand(w, {2.6f, 0}, 0.0f, true) == 1.0, "body well inside the room, turning: 1");
    }
    {   // an obstacle polygon (a 0.4 x 0.4 box at x=0.35) blocks yaw pi/2 / turning but not yaw 0 facing it? it is reached by the body at 0.15
        const OccupiedCells none;
        const std::vector<Obstacle> box{Obstacle{0.55f, 0.0f, 0.3f, 0.3f, 0.0f}};   // spans x 0.40..0.70
        const StandWorld w{&body, &none, room, box};
        CHECK(p_stand(w, {0, 0}, 0.0f, false) == 1.0, "obstacle 0.40 m away: the body (0.23 reach) clears it");
        CHECK(p_stand(w, {0.3f, 0}, 0.0f, false) == 0.0, "obstacle under the body: 0");
    }
    {   // no body / no cell size: no belief, no veto
        const StandWorld w{};
        CHECK(p_stand(w, {0, 0}, 0.0f, true) == 1.0, "no world: 1");
    }
    std::printf("%s (%d failed)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
