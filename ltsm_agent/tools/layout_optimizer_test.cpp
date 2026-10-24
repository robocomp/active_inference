// The memory layout as a MAP estimate (layout_optimizer.h) against ground truth, on synthetic apartments.
//   make -C build -j8 layout_optimizer_test && bin/layout_optimizer_test
#include "layout_optimizer.h"
#include <cstdio>
#include <numbers>
#include <random>
using namespace ltsm::layout;
static int failures = 0;
static void check(bool ok, const char *what) { if (not ok) ++failures; std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what); }

static double wrap(double a) { return std::atan2(std::sin(a), std::cos(a)); }
static Eigen::Matrix2d rot(double a) { Eigen::Matrix2d R; R << std::cos(a), -std::sin(a), std::sin(a), std::cos(a); return R; }
static Vec3 between(const Vec3 &a, const Vec3 &b)
{
    const Vec2 t = rot(-a.z()) * (b.head<2>() - a.head<2>());
    return {t.x(), t.y(), wrap(b.z() - a.z())};
}
static Vec3 compose(const Vec3 &a, const Vec3 &b)
{
    const Vec2 t = a.head<2>() + rot(a.z()) * b.head<2>();
    return {t.x(), t.y(), wrap(a.z() + b.z())};
}

/// An axis-aligned world rectangle seen from its room's own frame `pose` (the frame is arbitrary, as room_concept's is).
static Polygon local_rect(const Vec3 &pose, double x0, double y0, double x1, double y1)
{
    Polygon out;
    for (const Vec2 &w : {Vec2(x0, y0), Vec2(x1, y0), Vec2(x1, y1), Vec2(x0, y1)})
        out.push_back(rot(-pose.z()) * (w - pose.head<2>()));
    return out;
}
static Vec2 to_local(const Vec3 &pose, const Vec2 &w) { return rot(-pose.z()) * (w - pose.head<2>()); }
/// A 0.9 m aperture centred at world point `c` along world direction `dir`, seen from the room at `pose`.
static std::array<Vec2, 2> aperture(const Vec3 &pose, const Vec2 &c, const Vec2 &dir)
{
    return {to_local(pose, c - 0.45 * dir), to_local(pose, c + 0.45 * dir)};
}

static double pos_err(const std::vector<Vec3> &est, const std::vector<Vec3> &truth)
{
    double e = 0;
    for (std::size_t i = 0; i < est.size(); ++i) e += (est[i].head<2>() - truth[i].head<2>()).norm();
    return e / static_cast<double>(est.size());
}

int main()
{
    std::mt19937 rng(7);
    std::normal_distribution<double> N(0.0, 1.0);
    std::uniform_real_distribution<double> U(-std::numbers::pi, std::numbers::pi);

    // ── 0. Geometry primitives ────────────────────────────────────────────────────────────────
    std::printf("primitives\n");
    {
        const Polygon sq{{0, 0}, {2, 0}, {2, 2}, {0, 2}};
        check(std::abs(signed_distance(sq, {1, 1}) + 1.0) < 1e-12, "centre of a 2 m square is 1 m inside");
        check(std::abs(signed_distance(sq, {3, 1}) - 1.0) < 1e-12, "1 m right of it is 1 m outside");
        Polygon r30;
        for (const auto &p : sq) r30.push_back(rot(0.3) * p);
        const auto ax = dominant_axis(r30);
        check(ax.has_value() and std::abs(wrap(4 * (ax->phi - 0.3))) < 1e-9, "a square rotated 0.3 rad votes axis 0.3");
        check(ax.has_value() and ax->se4 < 1e-6, "and votes it with ~no dispersion");
    }

    // ── 1. A ring of four rooms: 2×2 grid, 0.12 m walls, four passages (one loop) ───────────────
    //    Noisy seams (5 cm, 2°). The BFS spanning tree reaches the far room through ONE path; the MAP uses both.
    std::printf("ring of four rooms, 300 trials\n");
    {
        const double t = 0.12, W = 4.0, H = 3.0;
        const std::array<std::array<double, 4>, 4> box{{{0, 0, W, H}, {W + t, 0, 2 * W + t, H},
                                                        {0, H + t, W, 2 * H + t}, {W + t, H + t, 2 * W + t, 2 * H + t}}};
        // passages: 0-1 (vertical wall x=W), 0-2 (horizontal wall y=H), 1-3, 2-3; door centres on those walls
        struct P { std::size_t a, b; Vec2 door_a, door_b; };
        const std::vector<P> pass{{0, 1, {W, 1.0}, {W + t, 1.0}}, {0, 2, {1.5, H}, {1.5, H + t}},
                                  {1, 3, {W + t + 2.5, H}, {W + t + 2.5, H + t}}, {2, 3, {W, H + t + 2}, {W + t, H + t + 2}}};
        const double sxy = 0.05, syaw = 2.0 * std::numbers::pi / 180.0;
        Mat3 C = Mat3::Zero();
        C.diagonal() << sxy * sxy, sxy * sxy, syaw * syaw;
        double bfs_sum = 0, map_sum = 0, yaw_bfs = 0, yaw_map = 0;
        int map_wins = 0, converged = 0;
        const int trials = 300;
        for (int k = 0; k < trials; ++k)
        {
            std::vector<Vec3> truth(4);
            for (auto &p : truth) p = Vec3(N(rng), N(rng), U(rng));   // arbitrary room frames
            truth[0] = Vec3::Zero();                                    // the anchor's frame is the map's
            std::vector<Room> rooms(4);
            for (std::size_t i = 0; i < 4; ++i)
                rooms[i] = {"r" + std::to_string(i), local_rect(truth[i], box[i][0], box[i][1], box[i][2], box[i][3])};
            rooms[0].fixed = true;
            std::vector<Seam> seams;
            for (const auto &p : pass)
            {
                Vec3 z = between(truth[p.a], truth[p.b]);
                z += Vec3(sxy * N(rng), sxy * N(rng), syaw * N(rng));
                const Vec2 dir = std::abs(p.door_a.x() - p.door_b.x()) > 1e-9 ? Vec2(0, 1) : Vec2(1, 0);   // along the wall
                seams.push_back({"p", p.a, p.b, z, C, aperture(truth[p.a], p.door_a, dir), aperture(truth[p.b], p.door_b, dir)});
            }
            // BFS from 0: 1 and 2 directly, 3 through 1 (the first passage that reaches it).
            std::vector<Vec3> bfs(4);
            bfs[0] = Vec3::Zero();
            bfs[1] = compose(bfs[0], seams[0].z);
            bfs[2] = compose(bfs[0], seams[1].z);
            bfs[3] = compose(bfs[1], seams[2].z);
            for (std::size_t i = 0; i < 4; ++i) rooms[i].init = bfs[i];
            const auto r = optimise(rooms, seams);
            converged += r.converged;
            const double eb = pos_err(bfs, truth), em = pos_err(r.pose, truth);
            bfs_sum += eb;
            map_sum += em;
            map_wins += em < eb;
            for (std::size_t i = 1; i < 4; ++i)
            {
                yaw_bfs += std::abs(wrap(bfs[i].z() - truth[i].z()));
                yaw_map += std::abs(wrap(r.pose[i].z() - truth[i].z()));
            }
        }
        std::printf("    mean position error  BFS %.4f m   MAP %.4f m   (MAP better in %d/%d)\n",
                    bfs_sum / trials, map_sum / trials, map_wins, trials);
        std::printf("    mean yaw error       BFS %.4f°    MAP %.4f°\n", yaw_bfs / (3 * trials) * 180 / std::numbers::pi,
                    yaw_map / (3 * trials) * 180 / std::numbers::pi);
        check(converged == trials, "every solve converged");
        check(map_sum < 0.7 * bfs_sum, "MAP position error at least 30% below the spanning tree's");
        check(yaw_map < 0.5 * yaw_bfs, "MAP yaw error at least halved (Manhattan + loop)");
    }

    // ── 2. A room at 30° to the building is NOT snapped square ─────────────────────────────────
    std::printf("non-Manhattan room\n");
    {
        const double a30 = 30.0 * std::numbers::pi / 180.0;
        std::vector<Room> rooms(3);
        rooms[0] = {"a", local_rect(Vec3::Zero(), 0, 0, 4, 3), Vec3::Zero(), true};
        rooms[1] = {"b", local_rect(Vec3(4.12, 0, 0), 4.12, 0, 8.12, 3)};
        // c: a 3×2 rectangle in its own frame, placed at 30° beyond a's top wall
        const Vec3 tc(1.0, 3.5, a30);
        rooms[2] = {"c", Polygon{{0, 0}, {3, 0}, {3, 2}, {0, 2}}};
        const std::vector<Vec3> truth{Vec3::Zero(), Vec3(4.12, 0, 0), tc};
        const double sxy = 0.05, syaw = 2.0 * std::numbers::pi / 180.0;
        Mat3 C = Mat3::Zero();
        C.diagonal() << sxy * sxy, sxy * sxy, syaw * syaw;
        std::vector<Seam> seams{{"ab", 0, 1, between(truth[0], truth[1]), C, std::nullopt, std::nullopt},
                                {"ac", 0, 2, between(truth[0], truth[2]), C, std::nullopt, std::nullopt}};
        for (std::size_t i = 0; i < 3; ++i) rooms[i].init = truth[i];
        const auto r = optimise(rooms, seams);
        std::printf("    c yaw %.2f° (truth 30°), P(Manhattan) a %.3f b %.3f c %.3f\n", r.pose[2].z() * 180 / std::numbers::pi,
                    r.p_manhattan[0], r.p_manhattan[1], r.p_manhattan[2]);
        check(std::abs(wrap(r.pose[2].z() - a30)) < 0.2 * std::numbers::pi / 180.0, "c keeps its measured 30° (within 0.2°)");
        check(r.p_manhattan[2] < 0.01, "c is explained as NOT square to the building");
        check(r.p_manhattan[1] > 0.9, "b is square to it (odds ≈ √(2πκ) at κ ≈ 200: ~0.97)");
    }

    // ── 3. A seam that drives one room into another is pushed out by OVERLAP ───────────────────
    std::printf("overlap\n");
    {
        std::vector<Room> rooms(2);
        rooms[0] = {"a", local_rect(Vec3::Zero(), 0, 0, 4, 3), Vec3::Zero(), true};
        rooms[1] = {"b", Polygon{{0, 0}, {4, 0}, {4, 3}, {0, 3}}};
        const Vec3 z(3.6, 0, 0);   // b's frame 0.4 m INSIDE a: 0.52 m short of where its wall would be
        Mat3 C = Mat3::Zero();
        C.diagonal() << 0.3 * 0.3, 0.05 * 0.05, 0.01 * 0.01;   // weak along x, as a recess-uncertain seam is
        rooms[1].init = z;
        Model m;
        m.walls = false;
        const auto r = optimise(rooms, {{"ab", 0, 1, z, C, std::nullopt, std::nullopt}}, m);
        std::printf("    b.x %.3f -> %.3f   overlap energy %.2f -> %.4f\n", z.x(), r.pose[1].x(), r.before.overlap, r.after.overlap);
        check(r.pose[1].x() > 3.95, "b pushed out to (about) the shared wall");
        check(r.after.overlap < 0.05 * r.before.overlap, "overlap energy removed");
        check(r.cov[1](0, 0) < 0.3 * 0.3, "and the posterior along x is tighter than the seam alone");
    }

    // ── 4. WALL pins the across-wall gap a weak seam leaves loose ───────────────────────────────
    std::printf("shared wall\n");
    {
        std::vector<Room> rooms(2);
        rooms[0] = {"a", local_rect(Vec3::Zero(), 0, 0, 4, 3), Vec3::Zero(), true};
        rooms[1] = {"b", Polygon{{0, 0}, {4, 0}, {4, 3}, {0, 3}}};
        const Vec3 z(4.6, 0, 0);   // 0.6 m gap; truth 0.12
        Mat3 C = Mat3::Zero();
        C.diagonal() << 0.3 * 0.3, 0.05 * 0.05, 0.01 * 0.01;
        rooms[1].init = z;
        const auto r = optimise(rooms, {{"ab", 0, 1, z, C, aperture(Vec3::Zero(), Vec2(4.0, 1.5), Vec2(0, 1)),
                                         aperture(Vec3::Zero(), Vec2(0.0, 1.5), Vec2(0, 1))}});
        std::printf("    b.x %.3f -> %.3f (wall prior 4.12)   seam d² %.2f\n", z.x(), r.pose[1].x(), r.seam_d2[0]);
        check(std::abs(r.pose[1].x() - 4.12) < 0.1, "gap pulled to about the wall thickness");
    }

    // ── 5. A gross OUTLIER seam (gold2: two passages between one pair of rooms disagreeing by metres) ───────────────
    //    Given poses come from the BAD seam (as a spanning tree that walked it first would). The mixture must disown it
    //    and the multi-start must find the mode that trusts the good one.
    std::printf("outlier seam\n");
    {
        std::vector<Room> rooms(3);
        rooms[0] = {"a", local_rect(Vec3::Zero(), 0, 0, 4, 3), Vec3::Zero(), true};
        const Vec3 tb(4.12, 0.3, 0.02), tc(0.2, 3.12, -0.01);
        rooms[1] = {"b", local_rect(tb, 4.12, 0, 8.12, 3)};
        rooms[2] = {"c", local_rect(tc, 0, 3.12, 4, 6.12)};
        const std::vector<Vec3> truth{Vec3::Zero(), tb, tc};
        const double sxy = 0.1, syaw = 0.03;
        Mat3 C = Mat3::Zero();
        C.diagonal() << sxy * sxy, sxy * sxy, syaw * syaw;
        const Vec3 good = between(truth[0], truth[1]);
        const Vec3 bad = good + Vec3(0.4, 3.0, 0.5);
        const Vec2 up(0, 1), right(1, 0);
        std::vector<Seam> seams{{"bad", 0, 1, bad, C, aperture(Vec3::Zero(), Vec2(4.0, 1.0), up), aperture(tb, Vec2(4.12, 1.0), up), 10.0},
                                {"good", 0, 1, good, C, aperture(Vec3::Zero(), Vec2(4.0, 2.0), up), aperture(tb, Vec2(4.12, 2.0), up), 10.0},
                                {"ac", 0, 2, between(truth[0], truth[2]), C, aperture(Vec3::Zero(), Vec2(2.0, 3.0), right),
                                 aperture(tc, Vec2(2.0, 3.12), right), 10.0}};
        rooms[1].init = bad;   // the tree walked the bad seam
        rooms[2].init = truth[2];
        const auto r = optimise(rooms, seams);
        const double eb = (r.pose[1].head<2>() - tb.head<2>()).norm();
        std::printf("    b error %.3f m, P(inlier) bad %.4f good %.4f ac %.4f, %d starts, winner %d\n", eb,
                    r.seam_p_inlier[0], r.seam_p_inlier[1], r.seam_p_inlier[2], r.starts, r.winning_start);
        check(eb < 0.1, "b placed by the good seam (within 0.1 m)");
        check(r.seam_p_inlier[0] < 0.01, "the bad seam is disowned (P(inlier) < 0.01)");
        check(r.seam_p_inlier[1] > 0.9 and r.seam_p_inlier[2] > 0.9, "the good ones are kept");
        check(r.winning_start != 0, "found from a start other than the bad given poses");
    }

    // ── 6. EVERY seam into a room wrong: ADJACENCY keeps it near its door instead of letting it drift ─────────────
    std::printf("all seams of a room wrong\n");
    {
        std::vector<Room> rooms(2);
        rooms[0] = {"a", local_rect(Vec3::Zero(), 0, 0, 4, 3), Vec3::Zero(), true};
        const Vec3 tb(4.12, 0.0, 0.0);
        rooms[1] = {"b", local_rect(tb, 4.12, 0, 8.12, 3)};
        Mat3 C = Mat3::Zero();
        C.diagonal() << 0.01, 0.01, 0.0009;
        const Vec2 up(0, 1);
        const Vec3 good = between(Vec3::Zero(), tb);
        std::vector<Seam> seams{{"w1", 0, 1, good + Vec3(5, -4, 1.0), C, aperture(Vec3::Zero(), Vec2(4.0, 1.0), up), aperture(tb, Vec2(4.12, 1.0), up), 10.0},
                                {"w2", 0, 1, good + Vec3(-6, 3, -2.0), C, aperture(Vec3::Zero(), Vec2(4.0, 2.0), up), aperture(tb, Vec2(4.12, 2.0), up), 10.0}};
        rooms[1].init = seams[0].z;
        const auto r = optimise(rooms, seams);
        const double dist = (r.pose[1].head<2>() - Vec2(4.0, 1.5)).norm();
        std::printf("    b origin %.2f m from a's doorway (b reaches %.2f m), P(inlier) %.4f %.4f\n", dist, 5.0, r.seam_p_inlier[0], r.seam_p_inlier[1]);
        check(dist < 2 * 5.0, "b stays within twice its own reach of the doorway (was 288 m in a real memory)");
    }

    // ── 7. Two rooms STACKED exactly on top of each other are not a free lunch (boundary-only sampling made them one) ──
    std::printf("stacked rooms\n");
    {
        std::vector<Room> rooms(2);
        rooms[0] = {"a", local_rect(Vec3::Zero(), 0, 0, 4, 3), Vec3::Zero(), true};
        rooms[1] = {"b", Polygon{{0, 0}, {4, 0}, {4, 3}, {0, 3}}};
        Mat3 C = Mat3::Zero();
        C.diagonal() << 0.01, 0.01, 0.0009;
        const Vec3 good(4.12, 0, 0);
        const std::vector<Seam> seams{{"ab", 0, 1, good, C, std::nullopt, std::nullopt, 10.0}};
        rooms[1].init = Vec3::Zero();   // stacked
        Model m;
        m.max_iterations = 0;   // score the stacked pose itself: no seams ⇒ no other start, no iterations ⇒ no move
        Model m2;
        const auto stacked = optimise(rooms, {}, m);
        std::printf("    overlap energy stacked %.1f\n", stacked.before.overlap);
        check(stacked.before.overlap > 100.0, "stacked rooms carry a large overlap energy");
        const auto r = optimise(rooms, seams, m2);
        std::printf("    with the seam: b.x %.3f (truth 4.12), P(inlier) %.3f\n", r.pose[1].x(), r.seam_p_inlier[0]);
        check(std::abs(r.pose[1].x() - 4.12) < 0.1 and r.seam_p_inlier[0] > 0.9, "the seam's placement wins over stacking");
    }

    std::printf(failures ? "\n%d FAILURE(S)\n" : "\nall passed\n", failures);
    return failures ? 1 : 0;
}
