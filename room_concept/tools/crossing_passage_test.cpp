// Task 16 (2026-10-29): the robot passing THROUGH an aperture is evidence the door is open, and "through" is a
// property of the PATH, not of the current pose. recog_proto5 leg 5: the robot crossed DOOR_2 and walked on into the
// next room; door_39 (a fresh re-birth with a near-shut prior, p_open 1.28e-4) read p_past 1, p_span 1.6e-6 (the robot
// was by then 1.4 m sideways of the mouth) ⇒ p_cross 1.7e-9, no birth. This drives the agent's own PassagePath +
// evaluate() along synthetic paths, exactly as RoomSceneGraph::step_proto_room_impl does (evaluate, then observe).
//   g++ -std=c++23 -I/usr/include/eigen3 -I../src crossing_passage_test.cpp -o crossing_passage_test
#include "door_crossing.h"
#include <cstdio>
#include <random>

using Eigen::Vector2f;
namespace cr = rc::crossing;
static int failures = 0;
static void check(bool ok, const char *what)
{ if (not ok) ++failures; std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what); }

// Room: 6 x 5 m, the door in its +y wall at y = 3, x in [-0.5, 0.5] (a 1 m opening, as DOOR_2).
static const std::vector<Vector2f> kRoom = {{-3, -2}, {3, -2}, {3, 3}, {-3, 3}};
constexpr float kBirth = 0.95f;           // PROTO_ROOM_BIRTH_PROB (pre-existing decision level)
constexpr float kSigPose = 0.01f;         // live localiser sigma (recog_proto5: sig_x ~ 0.0105)
constexpr float kSigLayout = 0.06f;       // publish_corner_sigma, as the agent passes it
constexpr float kRBody = 0.25f;

static rc::DoorAperture door(float p_open, Vector2f a = {-0.5f, 3}, Vector2f b = {0.5f, 3})
{
    rc::DoorAperture ap; ap.a = a; ap.b = b; ap.p_open = p_open; ap.sigma_along = 0.06f; ap.name = "door_test";
    return ap;
}

struct Run { float max_p = 0.f, last_p = 0.f, last_open_post = 0.f; cr::Crossing last; };

// Drive a path (with optional pose jitter, the localiser's own sigma) through the agent's evaluate-then-observe loop.
static Run drive(const rc::DoorAperture& ap, const std::vector<Vector2f>& path, cr::PassagePath& mem, bool use_path = true,
                 unsigned seed = 7)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> n(0.f, kSigPose);
    const Eigen::Matrix2f cov = Eigen::Matrix2f::Identity() * kSigPose * kSigPose;
    Run r;
    for (const auto &p : path)
    {
        const Vector2f xy = p + Vector2f(n(rng), n(rng));
        r.last = cr::evaluate(ap, kRoom, xy, cov, kRBody, kSigLayout,
                              use_path ? mem.samples() : std::span<const cr::PoseSample>{});
        mem.observe(xy, cov);
        r.max_p = std::max(r.max_p, r.last.p_cross);
        r.last_p = r.last.p_cross;
        r.last_open_post = r.last.p_open_post;
    }
    return r;
}

static void line(std::vector<Vector2f>& path, Vector2f from, Vector2f to, float step = 0.02f)
{
    const int n = std::max(1, static_cast<int>((to - from).norm() / step));
    for (int i = 0; i <= n; ++i) path.push_back(from + (to - from) * (static_cast<float>(i) / n));
}
static void park(std::vector<Vector2f>& path, Vector2f at, int cycles) { for (int i = 0; i < cycles; ++i) path.push_back(at); }

int main()
{
    std::printf("(a) THE RECORDED CASE: door prior p_open 1.28e-4, path through the aperture to 0.85 m past\n");
    {
        std::vector<Vector2f> path;
        line(path, {0.f, 1.5f}, {0.f, 3.85f});
        park(path, {0.f, 3.85f}, 40);
        cr::PassagePath mem;
        const auto r = drive(door(1.28e-4f), path, mem);
        std::printf("  at 0.85 m past: p_past %.4f p_through %.6f p_cross %.6f p_open_post %.4f\n",
                    r.last.p_past, r.last.p_through, r.last_p, r.last_open_post);
        check(r.last_p >= kBirth, "p_cross >= 0.95 (birth) despite the near-shut prior");
        check(r.last_open_post > 1.28e-4f * 10.f, "P(open | path) rose well above the prior");

        // …then on into the new room, sideways along it (leg 5: s 2.73, u -1.4)
        std::vector<Vector2f> on;
        line(on, {0.f, 3.85f}, {0.f, 5.7f});
        line(on, {0.f, 5.7f}, {-1.4f, 5.7f});
        const auto r2 = drive(door(1.28e-4f), on, mem, true, 11);
        std::printf("  2.7 m past, 1.4 m sideways: p_span(now) %.3g p_through %.6f p_cross %.6f\n",
                    r2.last.p_span, r2.last.p_through, r2.last_p);
        check(r2.last.p_span < 1e-6f, "the CURRENT centre is far outside the jambs (the live p_span)");
        check(r2.last_p >= kBirth, "the path still says it went through: p_cross >= 0.95");

        cr::PassagePath none;
        std::vector<Vector2f> here{{-1.4f, 5.7f}};
        const auto r3 = drive(door(1.28e-4f), here, none, false);
        std::printf("  same pose, no path (the pre-Task-16 reading): p_cross %.3g\n", r3.last_p);
        check(r3.last_p < kBirth, "(without the path the same pose would not birth: the defect)");

        // the door is re-estimated after the crossing (door_39 moved ~2 m mid-leg): the path is re-read against the
        // CURRENT aperture, so a door that moved but still lies across the path keeps the passage
        const auto moved = door(1.28e-4f, {-0.3f, 3.2f}, {0.7f, 3.2f});
        cr::PassagePath mem2;
        std::vector<Vector2f> all;
        line(all, {0.f, 1.5f}, {0.f, 5.7f});
        line(all, {0.f, 5.7f}, {-1.4f, 5.7f});
        const auto r4 = drive(moved, all, mem2);
        std::printf("  aperture re-estimated 0.2 m out, 0.2 m along: p_cross %.6f\n", r4.last_p);
        check(r4.last_p >= kBirth, "re-read against the moved aperture: still a birth");
    }

    std::printf("(b) STOPS AT THE VANTAGE in front of a door, parked with pose jitter\n");
    for (const float po : {1.28e-4f, 0.5f, 0.99f})
    {
        std::vector<Vector2f> path;
        line(path, {0.f, 1.0f}, {0.f, 2.2f});
        park(path, {0.f, 2.2f}, 600);
        cr::PassagePath mem;
        const auto r = drive(door(po), path, mem);
        std::printf("  p_open %.4g: max p_cross %.3g\n", po, r.max_p);
        check(r.max_p < kBirth, "no birth");
    }
    std::printf("    …and with the robot parked right AT the mouth of a shut door (centre 2 cm short of the line)\n");
    {
        std::vector<Vector2f> path;
        line(path, {0.f, 1.0f}, {0.f, 2.98f});
        park(path, {0.f, 2.98f}, 600);
        cr::PassagePath mem;
        const auto r = drive(door(1.28e-4f), path, mem);
        std::printf("  max p_cross %.3g\n", r.max_p);
        check(r.max_p < kBirth, "no birth from jitter across the plane of a shut leaf");
    }

    std::printf("(c) DRIVES PAST alongside the wall, across the front of the door\n");
    for (const float po : {1.28e-4f, 0.99f})
    {
        std::vector<Vector2f> path;
        line(path, {-2.5f, 2.6f}, {2.5f, 2.6f});
        cr::PassagePath mem;
        const auto r = drive(door(po), path, mem);
        std::printf("  p_open %.4g, 0.4 m from the wall: max p_cross %.3g\n", po, r.max_p);
        check(r.max_p < kBirth, "no birth");
    }
    std::printf("    …a pose ERROR that puts the robot past the wall OUTSIDE the jambs, then slides it along\n");
    for (const float po : {1.28e-4f, 0.99f})
    {
        std::vector<Vector2f> path;
        line(path, {-2.5f, 2.6f}, {-2.5f, 3.4f});   // through the wall, 2.5 m from the door
        line(path, {-2.5f, 3.4f}, {0.f, 3.4f});     // then "past" in front of the opening
        cr::PassagePath mem;
        const auto r = drive(door(po), path, mem);
        std::printf("  p_open %.4g: final p_through %.3g p_cross %.3g (old pose-only reading would say %.3g)\n",
                    po, r.last.p_through, r.last_p,
                    cr::evaluate(door(po), kRoom, {0.f, 3.4f}, Eigen::Matrix2f::Identity() * kSigPose * kSigPose,
                                 kRBody, kSigLayout).p_cross);
        check(r.max_p < kBirth, "no birth: the path entered through the wall, not the door");
    }

    std::printf("(d) frame move: PassagePath::apply keeps the passage when the room frame is re-anchored\n");
    {
        std::vector<Vector2f> path;
        line(path, {0.f, 1.5f}, {0.f, 3.85f});
        cr::PassagePath mem;
        (void)drive(door(1.28e-4f), path, mem);
        Eigen::Affine2f T = Eigen::Affine2f::Identity();
        T.rotate(0.4f); T.pretranslate(Vector2f(1.f, -2.f));
        mem.apply(T);
        std::vector<Vector2f> room2; for (const auto &p : kRoom) room2.push_back(T * p);
        rc::DoorAperture ap2 = door(1.28e-4f, T * Vector2f(-0.5f, 3), T * Vector2f(0.5f, 3));
        const Eigen::Matrix2f cov = Eigen::Matrix2f::Identity() * kSigPose * kSigPose;
        const auto c = cr::evaluate(ap2, room2, T * Vector2f(-1.4f, 5.7f), cov, kRBody, kSigLayout, mem.samples());
        std::printf("  after the move: p_cross %.6f\n", c.p_cross);
        check(c.p_cross >= kBirth, "still a birth in the moved frame");
    }

    std::printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
