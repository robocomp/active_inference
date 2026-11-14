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
        // the cluster's identity is its reference aperture (the member least along x, rounded to half its width), not a member's name; its hypotheses are the
        // SUBSETS of the members at distinct holes: none, A, B, A and B (final review 7)
        CHECK(std::ranges::any_of(qs, [](const Question& q) { return q.key.starts_with("c_") and q.key.find("door") == std::string::npos and q.hyps.size() == 4; }));
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

    // final review 7: two DECIDED doors that share a mask are independent doors: "both exist" is a hypothesis, so the question settles as their evidence grows
    // (the earlier "at most one of them" kept a ln 2 question standing for the rest of the visit, a hypothesis no look could confirm)
    {
        const auto pair_of = [](double L)
        {
            Member A; A.name = "door_0"; A.aperture = bottom(1.0f); A.leaf = LeafState{}; A.L = L;
            Member B; B.name = "door_1"; B.aperture = bottom(3.0f); B.leaf = LeafState{}; B.L = L;
            return std::vector<Member>{A, B};
        };
        const std::array<std::pair<std::string, std::string>, 1> edge{{{"door_0", "door_1"}}};
        const auto all = [](const Member&) { return true; };
        const auto m5 = pair_of(5.0), m8 = pair_of(8.0);
        const auto q5 = questions_from(m5, edge, all), q8 = questions_from(m8, edge, all);
        CHECK(q5.size() == 1 and q8.size() == 1);
        if (q5.size() == 1 and q8.size() == 1)
        {
            const auto w5 = weights_of(q5[0]);
            const auto both = std::ranges::max_element(w5);
            CHECK(q5[0].hyps[static_cast<std::size_t>(both - w5.begin())].apertures.size() == 2);   // the heaviest hypothesis is "both exist"
            const Model m = model_of(no_furniture, true);
            const auto poses = room_poses(kRoom, no_furniture, R);
            const Best b5 = best_look(q5[0], poses, m), b8 = best_look(q8[0], poses, m);
            std::printf("two decided doors sharing a mask: L=5 -> %zu live hypotheses, best EIG %.4f nats (settled %d); L=8 -> settled %d, best EIG %.5f nats\n",
                        live_part(q5[0]).hyps.size(), b5.eig_nats, int(settled(q5[0])), int(settled(q8[0])), b8.eig_nats);
            CHECK(b5.eig_nats < 0.1);          // far below ln 2 = 0.69, the standing question of the exclusive reading
            CHECK(settled(q8[0]) and b8.eig_nats < 0.01);
        }
        // two labels at ONE hole stay exclusive (a hole holds one door): no hypothesis puts both in
        Member A; A.name = "door_0"; A.aperture = bottom(1.0f); A.leaf = LeafState{}; A.L = 1.0;
        Member B; B.name = "place_3"; B.aperture = bottom(1.05f); B.leaf = LeafState{}; B.L = 1.0;
        const std::array<Member, 2> hole{A, B};
        const std::array<std::pair<std::string, std::string>, 1> e2{{{"door_0", "place_3"}}};
        const auto qh = questions_from(hole, e2, all);
        CHECK(qh.size() == 1 and std::ranges::none_of(qh[0].hyps, [](const Hyp& h) { return h.apertures.size() == 2; }) and qh[0].hyps.size() == 3);
    }
    // final review 8: question identity. A candidate born as a door keeps its key; a cluster is keyed by where it is, not who is in it; entries follow questions
    {
        Member c; c.name = "place_5"; c.aperture = bottom(1.0f); c.leaf = LeafState{}; c.L = 0.2;
        Member d = c; d.name = "door_7"; d.key_name = "place_5";   // the candidate became a door
        const auto open = [](const Member&) { return true; };
        const auto k0 = questions_from(std::array<Member, 1>{c}, {}, open), k1 = questions_from(std::array<Member, 1>{d}, {}, open);
        CHECK(k0.size() == 1 and k1.size() == 1 and k0[0].key == k1[0].key and k0[0].key == "q_place_5");
        Member a; a.name = "door_2"; a.aperture = bottom(1.0f); a.leaf = LeafState{}; a.L = 0.2;
        Member b; b.name = "door_5"; b.aperture = bottom(3.0f); b.leaf = LeafState{}; b.L = 0.4;
        Member lighter; lighter.name = "door_1"; lighter.aperture = bottom(3.0f); lighter.leaf = LeafState{}; lighter.L = 0.1;   // a lighter-named member joins (same hole as b)
        const std::array<std::pair<std::string, std::string>, 1> eab{{{"door_2", "door_5"}}};
        const std::array<std::pair<std::string, std::string>, 2> eabl{{{"door_2", "door_5"}, {"door_5", "door_1"}}};
        const auto q1 = questions_from(std::array<Member, 2>{a, b}, eab, open), q2 = questions_from(std::array<Member, 3>{a, b, lighter}, eabl, open);
        CHECK(q1.size() == 1 and q2.size() == 1 and q1[0].key == q2[0].key);
        // entries follow questions: an exact key first, else the nearest reference within half the narrower width, each entry once
        const std::vector<std::pair<std::string, Aperture>> existing{{"q_place_5", bottom(1.0f)}, {"c_x", bottom(3.0f)}};
        const std::vector<std::pair<std::string, Aperture>> asked{{"q_door_7", bottom(1.1f)}, {"c_y", bottom(3.05f)}, {"q_new", bottom(2.0f)}};
        const auto ek = resolve_keys(existing, asked);
        CHECK(ek.size() == 3 and ek[0] == "q_place_5" and ek[1] == "c_x" and ek[2] == "q_new");
        // never remove a node the controller is driving to: held while the claim runs, withdrawn after
        const std::vector<std::string> keys{"q_place_5", "c_x", "gone"}, continued{"q_place_5"};
        const auto w1 = to_withdraw(keys, continued, [](const std::string& k) { return k == "c_x"; });   // c_x is executing
        const auto w2 = to_withdraw(keys, continued, [](const std::string&) { return false; });          // the claim ended
        CHECK(w1.size() == 1 and w1[0] == "gone" and w2.size() == 2 and w2[0] == "c_x");
    }
    // minor 21: ONE table budget per cycle across questions: a second question that no longer fits is sampled, nothing held
    {
        const Model m = model_of(no_furniture, true);
        const auto poses = room_poses(kRoom, no_furniture, R);
        Question q{"b1", {}};
        q.hyps.push_back({{}, {}, 0.5}); q.hyps.push_back({{bottom(1.0f)}, {LeafState{}}, 0.5});
        Question q2 = q; q2.key = "b2";
        double budget = static_cast<double>(poses.size()) * 2.0 * 4.0 * 1.5;   // room for ONE of the two questions' exact tables
        LookCache c1, c2;
        (void)best_look(q, poses, m, &c1, nullptr, &budget);
        (void)best_look(q2, poses, m, &c2, nullptr, &budget);
        std::printf("one budget across questions: first sampled %d, second sampled %d\n", int(c1.sampled), int(c2.sampled));
        CHECK(not c1.sampled and c2.sampled and c2.tab.empty());
    }

    if (g_fail == 0) std::printf("ALL PASS\n");
    return g_fail == 0 ? 0 : 1;
}
