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

    // cluster_from_competition: members that competed for a mask form ONE question (null + each); an unrelated open member is its own
    {
        Member A; A.name = "place_7"; A.aperture = right(2.0f); A.leaf = LeafState{1.2f, HingeSide::Near, +1.f}; A.L = -0.9; A.leaf_born = true;
        Member B; B.name = "door_0"; B.aperture = bottom(3.9f); B.leaf = LeafState{std::numbers::pi_v<float> / 2.f, HingeSide::Far, +1.f}; B.L = 1.0;
        Member C; C.name = "door_9"; C.aperture = top(1.0f); C.leaf = LeafState{}; C.L = 0.2;
        const std::array<Member, 3> ms{A, B, C};
        const auto open9 = [](const Member& m) { return m.name == "door_9"; };
        const std::array<std::pair<std::string, std::string>, 1> edge{{{"place_7", "door_0"}}};
        const auto qs = questions_from(ms, edge, open9);
        CHECK(qs.size() == 2);
        CHECK(std::ranges::any_of(qs, [](const Question& q) { return q.key == "c_door_0" and q.hyps.size() == 3; }));   // null + the two
        CHECK(std::ranges::any_of(qs, [](const Question& q) { return q.key == "q_door_9" and q.hyps.size() == 2; }));
        const auto none = questions_from(ms, {}, open9);   // no mask ever tied them: nothing but the open one
        CHECK(none.size() == 1 and none[0].key == "q_door_9");
    }

    // cluster_splits_after_decision: two members linked by one shared mask are one question; a DECISION on one of them (its verdict moves, or it is
    // born/removed) drops the pair, so the next cycle's questions have them apart, until a fresh mask links them again
    {
        Member A; A.name = "place_7"; A.aperture = right(2.0f); A.leaf = LeafState{1.2f, HingeSide::Near, +1.f}; A.L = -0.9;
        Member B; B.name = "door_0"; B.aperture = bottom(3.9f); B.leaf = LeafState{1.2f, HingeSide::Far, +1.f}; B.L = 1.0;
        const std::array<Member, 2> ms{A, B};
        const auto open_all = [](const Member&) { return true; };
        using N = Competition::NodeRef;
        const N na{'c', 7}, nb{'d', 0};
        const auto names = [&](const Competition& c)
        {
            std::vector<std::pair<std::string, std::string>> out;
            for (const auto& [x, y] : c.pairs) out.emplace_back(x == na ? "place_7" : "door_0", y == nb ? "door_0" : "place_7");
            return out;
        };
        Competition comp;
        comp.refresh({{na, 1}, {nb, 1}});
        comp.note({na, nb});                                            // one mask both explain
        comp.refresh({{na, 1}, {nb, 1}});                                // nothing decided: still linked
        const auto e1 = names(comp);
        CHECK(questions_from(ms, e1, open_all).size() == 1);
        comp.refresh({{na, 1}, {nb, 3}});                                // door_0's verdict moved (removed): its pairs go
        const auto e2 = names(comp);
        CHECK(e2.empty() and questions_from(ms, e2, open_all).size() == 2);
        comp.note({na, nb});                                            // a fresh mask links them again
        comp.refresh({{na, 1}, {nb, 3}});
        CHECK(questions_from(ms, names(comp), open_all).size() == 1);
        comp.refresh({{nb, 3}});                                         // the candidate was born as a door (its key is gone): the pair is dead
        CHECK(comp.pairs.empty());
    }

    // sampled_beyond_budget: S = 7 apertures, 8 hypotheses: 4^7 columns is over the budget for the room's poses; the EIG is sampled, nothing cached,
    // and within 0.05 nats of the exact one (computed once here)
    {
        const std::vector<rc::nbv::Obstacle> fur{{2.5f, 2.0f, 0.8f, 0.5f, 0.2f}};
        const Model m = model_of(fur, false);
        Question q{"seven", {}};
        const LeafState open{1.2f, HingeSide::Near, +1.f};
        q.hyps.push_back({{}, {}, 0.3});
        for (int k = 0; k < 7; ++k) q.hyps.push_back({{bottom(0.1f + 0.68f * k, 0.5f)}, {open}, 0.1});
        const auto poses = room_poses(kRoom, fur, R);
        LookCache cache;
        const auto t0 = std::chrono::steady_clock::now();
        const Best b = best_look(q, poses, m, &cache);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        CHECK(cache.sampled and cache.tab.empty() and b.valid);
        double worst = 0.0;
        for (const Pose& p : {b.pose, poses[poses.size() / 3], poses[poses.size() / 2]})
            worst = std::max(worst, std::abs(eig_at(q, p, m, true) - eig_at(q, p, m, false)));
        std::printf("seven slots: %zu poses over budget, sampled best_look %.1f ms, eig %.4f; worst |sampled - exact| over 3 poses %.4f nats\n", poses.size(), ms, b.eig_nats, worst);
        CHECK(worst < 0.05);
    }

    // cache_signature_covers_every_pose: two admissible lists of equal size and endpoints but a different middle pose rebuild the tables
    {
        const Model m = model_of(no_furniture, false);
        Member mem; mem.name = "place_1"; mem.aperture = bottom(2.0f); mem.leaf = LeafState{};
        const Question q = place_question("place_1", mem, 0.5);
        std::vector<Pose> l1{{{1.0f, 2.0f}, 0.f}, {{2.5f, 2.0f}, -1.57f}, {{4.0f, 2.0f}, 3.0f}};
        std::vector<Pose> l2 = l1;
        l2[1] = {{2.5f, 1.0f}, -1.57f};
        LookCache cache;
        const Best b1 = best_look(q, l1, m, &cache);
        CHECK(not cache.hit);
        const Best b1b = best_look(q, l1, m, &cache);
        CHECK(cache.hit and b1b.eig_nats == b1.eig_nats);
        const Best b2 = best_look(q, l2, m, &cache);
        CHECK(not cache.hit);
        const Best b2_fresh = best_look(q, l2, m);
        CHECK(b2.eig_nats == b2_fresh.eig_nats and b2.pose.xy == b2_fresh.pose.xy);
    }

    // standing_pose_is_held: two poses that tie as argmax (alternating with the list order) do not alternate the published target
    {
        const std::vector<rc::nbv::Obstacle> fur{{2.5f, 1.0f, 4.0f, 0.2f, 0.f}};
        const Model m = model_of(fur, false);
        Question q{"hold", {}};
        q.hyps.push_back({{bottom(2.0f)}, {LeafState{1.2f, HingeSide::Near, +1.f}}, 0.5});
        q.hyps.push_back({{}, {}, 0.5});
        const Pose P{{2.5f, 3.0f}, std::numbers::pi_v<float> / 2.f}, Q{{2.0f, 3.0f}, std::numbers::pi_v<float> / 2.f};   // both blind: EIG 0 each
        const std::array<Pose, 2> pq{P, Q}, qp{Q, P};
        Pose standing = P;
        bool alternated_without_hold = best_look(q, pq, m).pose.xy != best_look(q, qp, m).pose.xy;
        CHECK(alternated_without_hold);   // the argmax itself flips with the list order
        bool moved = false;
        for (int k = 0; k < 6; ++k)
        {
            const Best b = best_look(q, k % 2 ? std::span<const Pose>(qp) : std::span<const Pose>(pq), m, nullptr, &standing);
            moved = moved or b.pose.xy != standing.xy;
        }
        CHECK(not moved);
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
