// Offline harness for src/door_hypothesis_look.h (the hypothesis look: the pose that most reduces the uncertainty there is now).
// Eigen, std and common/nbv only.
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_door_hypothesis_look.cpp -o /tmp/test_door_hypothesis_look && /tmp/test_door_hypothesis_look
#include "../src/door_hypothesis_look.h"
#include <chrono>
#include <cmath>
#include <cstdio>

using namespace rc::door;
using namespace rc::door::hlook;
static int g_fail = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static const std::vector<Eigen::Vector2f> kRoom{{0, 0}, {5, 0}, {5, 4}, {0, 4}};

static Model model_of(std::span<const rc::nbv::Obstacle> furniture, bool decisive)
{
    Model m;
    m.camera.hfov_rad = 1.92f; m.camera.vfov_rad = 1.22f; m.camera.height_m = 1.0f;
    m.camera.env = core::default_detectors().zed.env;
    m.zed = core::default_detectors().zed;
    m.zed.hit_rate_open = 0.7f;
    m.lidar = core::LidarThroughModel{};
    if (decisive) { m.lidar.through_open = 0.99f; m.lidar.through_wall = 0.001f; m.zed.hit_rate = 0.99f; m.zed.hit_rate_open = 0.99f; }
    m.polygon = kRoom;
    m.furniture = furniture;
    return m;
}

/// An aperture in the bottom wall (y = 0), along +x.
static Aperture bottom(float x0, float w = 0.9f)
{
    Aperture a; a.wall_O = {0.f, 0.f}; a.wall_u = {1.f, 0.f}; a.s = x0; a.w = w; a.h = 2.05f; a.thickness = 0.05f;
    return a;
}
/// An aperture in the right wall (x = 5), along +y (the room is on its -x side).
static Aperture right(float y0, float w = 0.9f)
{
    Aperture a; a.wall_O = {5.f, 0.f}; a.wall_u = {0.f, 1.f}; a.s = y0; a.w = w; a.h = 2.05f; a.thickness = 0.05f;
    return a;
}
/// An aperture in the top wall (y = 4), along -x (the room is on its -y side).
static Aperture top(float x0, float w = 0.9f)
{
    Aperture a; a.wall_O = {5.f, 4.f}; a.wall_u = {-1.f, 0.f}; a.s = x0; a.w = w; a.h = 2.05f; a.thickness = 0.05f;
    return a;
}
static float wrap(float a) { while (a > std::numbers::pi_v<float>) a -= 2 * std::numbers::pi_v<float>; while (a < -std::numbers::pi_v<float>) a += 2 * std::numbers::pi_v<float>; return a; }

int main()
{
    const rc::nbv::Obstacle none_o{};
    (void)none_o;
    const std::vector<rc::nbv::Obstacle> no_furniture;
    const float R = 0.30f;

    // room_poses_inside: every pose lies inside the polygon and off the obstacles
    {
        const std::vector<rc::nbv::Obstacle> fur{{2.5f, 2.0f, 1.0f, 0.6f, 0.3f}};
        const auto poses = room_poses(kRoom, fur, R);
        CHECK(not poses.empty());
        bool ok = true;
        for (const auto& p : poses)
            ok = ok and rc::nbv::is_reachable(p.xy, kRoom, R) and not rc::nbv::stands_inside(p.xy, fur, R);
        CHECK(ok);
        CHECK(poses.size() % 8 == 0);   // 8 headings per position
        // 5 x 4 m, 0.6 m cells: the grid is the robot's diameter
        std::printf("room_poses: %zu poses (%zu positions)\n", poses.size(), poses.size() / 8);
    }

    // eig_zero_when_blind: facing away and the LiDAR's sight lines to the span blocked -> nothing to learn
    {
        const std::vector<rc::nbv::Obstacle> fur{{2.5f, 1.0f, 4.0f, 0.2f, 0.f}};   // a long barrier between the pose and the bottom wall
        const Model m = model_of(fur, false);
        Question q{"blind", {}};
        q.hyps.push_back({{bottom(2.0f)}, {LeafState{1.2f, HingeSide::Near, +1.f}}, 0.5});
        q.hyps.push_back({{}, {}, 0.5});
        const Pose away{{2.5f, 3.0f}, std::numbers::pi_v<float> / 2.f};   // facing +y, the aperture is behind
        const auto t = outcome_table(q, away, m);
        const double e = core::bf::expected_information_gain(weights_of(q), t);
        CHECK(e < 1e-6);
    }

    // eig_full_when_decisive: A's opening is read through by the LiDAR from the pose, B's is not in sight
    {
        const std::vector<rc::nbv::Obstacle> fur{{2.5f, 3.0f, 1.4f, 0.2f, 0.f}};   // blocks the sight line to the top wall's span
        const Model m = model_of(fur, true);
        Question q{"decisive", {}};
        q.hyps.push_back({{bottom(1.0f)}, {LeafState{std::numbers::pi_v<float> / 2.f, HingeSide::Near, +1.f}}, 0.5});
        q.hyps.push_back({{top(1.0f)}, {LeafState{std::numbers::pi_v<float> / 2.f, HingeSide::Near, +1.f}}, 0.5});   // spans x = 4.0 .. 3.1
        const Pose at{{1.45f, 2.0f}, -std::numbers::pi_v<float> / 2.f};   // in front of A, facing it
        const Scene sc = make_scene(q, m);
        CHECK(sc.holes.size() == 2);
        // the pose sees A's span and not B's under B (the barrier)
        CHECK(lidar_los(at.xy, q.hyps[0].apertures[0], sc.occ[0][0]) > 0.99f);
        CHECK(lidar_los(at.xy, q.hyps[1].apertures[0], sc.occ[1][1]) < 0.01f);
        const double e = core::bf::expected_information_gain(weights_of(q), outcome_table(q, at, m));
        std::printf("decisive: eig %.4f nats (ln 2 = %.4f)\n", e, std::log(2.0));
        CHECK(e > 0.9 * std::log(2.0));
        const auto poses = room_poses(kRoom, fur, R);
        const Best b = best_look(q, poses, m);
        CHECK(b.valid and b.eig_nats >= e - 1e-9);   // the best over the room is at least as good as that pose
    }

    // corner_look_sees_one_opening: door A (bottom wall) whose open leaf hides door B's aperture (right wall) from the robot's side of
    // the room; neither opening is in the robot's line of sight, and the best look is a pose from which at least one is
    {
        const std::vector<rc::nbv::Obstacle> fur{{2.5f, 1.8f, 4.6f, 0.3f, 0.f}};   // the robot's half of the room is cut off from the wall
        const Model m = model_of(fur, false);
        const Aperture A = bottom(3.5f), B = right(0.5f);
        Question q{"corner", {}};
        q.hyps.push_back({{A}, {LeafState{std::numbers::pi_v<float> / 2.f, HingeSide::Near, +1.f}}, 0.45});
        q.hyps.push_back({{B}, {LeafState{1.2f, HingeSide::Near, +1.f}}, 0.45});
        q.hyps.push_back({{}, {}, 0.10});
        const Scene sc = make_scene(q, m);
        const Eigen::Vector2f robot(2.5f, 3.0f);
        CHECK(lidar_los(robot, A, sc.occ[0][0]) == 0.f and lidar_los(robot, B, sc.occ[1][1]) == 0.f);   // neither visible from the robot
        const auto poses = room_poses(kRoom, fur, R);
        const auto t0 = std::chrono::steady_clock::now();
        const Best b = best_look(q, poses, m);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("corner: best pose (%.2f, %.2f) yaw %.2f eig %.4f nats  [%zu poses, %.1f ms]\n", b.pose.xy.x(), b.pose.xy.y(), b.pose.yaw, b.eig_nats, poses.size(), ms);
        CHECK(b.valid and b.eig_nats > 0.0);
        CHECK(lidar_los(b.pose.xy, A, sc.occ[0][0]) > 0.f or lidar_los(b.pose.xy, B, sc.occ[1][1]) > 0.f);
    }

    // single_place_question: {exists 0.5, not 0.5}: the best pose faces the place
    {
        const Model m = model_of(no_furniture, false);
        Member mem; mem.name = "place_1"; mem.aperture = bottom(2.0f); mem.leaf = LeafState{};
        const Question q = place_question("place_1", mem, 0.5);
        const auto poses = room_poses(kRoom, no_furniture, R);
        const Best b = best_look(q, poses, m);
        CHECK(b.valid and b.eig_nats > 0.0);
        const Eigen::Vector2f to = mem.aperture.centre_xy() - b.pose.xy;
        const float off = std::abs(wrap(std::atan2(to.y(), to.x()) - b.pose.yaw));
        std::printf("single place: best pose (%.2f, %.2f) yaw %.2f, %.1f deg off the place, eig %.4f nats\n", b.pose.xy.x(), b.pose.xy.y(), b.pose.yaw, off * 57.2958f, b.eig_nats);
        CHECK(off < 0.5f * m.camera.hfov_rad);
    }

    // settled: a question with one live hypothesis is settled
    {
        Question q{"s", {}};
        q.hyps.push_back({{bottom(1.0f)}, {LeafState{}}, 1.0});
        q.hyps.push_back({{}, {}, 1e-5});
        CHECK(settled(q));
        q.hyps[1].w = 0.5;
        CHECK(not settled(q));
    }

    // cluster_from_overlapping_leaves: a side-wall leaf hypothesis and the front wall's door explain the same panel: ONE question, null + both
    {
        // the leaf stands along x = 4.4 from the corner region; door B's hinge is at its corner, door A's at the other end
        Member A; A.name = "place_7"; A.aperture = right(2.0f); A.leaf = LeafState{1.2f, HingeSide::Near, +1.f}; A.L = -0.9; A.leaf_born = true;
        Member B; B.name = "door_0"; B.aperture = bottom(3.9f); B.leaf = LeafState{std::numbers::pi_v<float> / 2.f, HingeSide::Far, +1.f}; B.L = 1.0;
        Member C; C.name = "door_9"; C.aperture = top(1.0f); C.leaf = LeafState{}; C.L = 0.2;   // somewhere else: its own question
        const std::array<Member, 3> ms{A, B, C};
        const auto segA = leaf_segment(A.aperture, A.leaf), segB = leaf_segment(B.aperture, B.leaf);
        std::printf("leaf segments: A (%.2f,%.2f)-(%.2f,%.2f)  B (%.2f,%.2f)-(%.2f,%.2f)  distance %.3f\n", segA[0].x(), segA[0].y(), segA[1].x(), segA[1].y(),
                    segB[0].x(), segB[0].y(), segB[1].x(), segB[1].y(), segment_distance(segA, segB));
        const auto qs = questions_from(ms, 5.0f, [](const Member& m) { return m.name == "door_9"; });   // reach 5 m: all three close
        CHECK(qs.size() == 1 and qs[0].hyps.size() == 4);   // null + the three
        const auto qs2 = questions_from(ms, 0.01f, [](const Member& m) { return m.name == "door_9"; });
        CHECK(std::ranges::any_of(qs2, [](const Question& q) { return q.key == "q_door_9" and q.hyps.size() == 2; }));
    }

    // timing (reported, not asserted): a 20 m2 room whose outline has 24 vertices (live rooms have ~30 edges), 4 hypotheses over 3 apertures
    {
        std::vector<Eigen::Vector2f> poly;
        for (int i = 0; i < 6; ++i) poly.push_back({5.f * i / 6.f, 0.f});
        for (int i = 0; i < 6; ++i) poly.push_back({5.f, 4.f * i / 6.f});
        for (int i = 0; i < 6; ++i) poly.push_back({5.f - 5.f * i / 6.f, 4.f});
        for (int i = 0; i < 6; ++i) poly.push_back({0.f, 4.f - 4.f * i / 6.f});
        const std::vector<rc::nbv::Obstacle> fur{{2.5f, 2.0f, 0.8f, 0.5f, 0.2f}, {1.0f, 3.0f, 0.5f, 0.5f, 0.f}};
        Model m = model_of(fur, false);
        m.polygon = poly;
        Question q{"bench", {}};
        const LeafState open{1.2f, HingeSide::Near, +1.f};
        q.hyps.push_back({{}, {}, 0.4});
        q.hyps.push_back({{bottom(1.0f)}, {open}, 0.3});
        q.hyps.push_back({{right(1.0f)}, {open}, 0.2});
        q.hyps.push_back({{top(1.0f)}, {open}, 0.1});
        const auto poses = room_poses(poly, fur, R);
        LookCache cache;
        const auto t0 = std::chrono::steady_clock::now();
        const Best b0 = best_look(q, poses, m, &cache);
        const auto t1 = std::chrono::steady_clock::now();
        const Best b1 = best_look(q, poses, m, &cache);
        const auto t2 = std::chrono::steady_clock::now();
        const double cold = std::chrono::duration<double, std::milli>(t1 - t0).count(), warm = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::printf("bench 20 m2 / 4 hypotheses / %zu poses: cold %.2f ms, warm (tables reused) %.2f ms, same best %d eig %.4f\n", poses.size(), cold, warm,
                    int(b0.pose.xy == b1.pose.xy and cache.hit), b0.eig_nats);
        CHECK(cache.hit and b0.eig_nats == b1.eig_nats);
    }

    if (g_fail == 0) std::printf("ALL PASS\n");
    return g_fail == 0 ? 0 : 1;
}
