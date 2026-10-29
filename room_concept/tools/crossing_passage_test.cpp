// Task 16 (2026-10-29): the robot passing THROUGH an aperture is evidence the door is open, and "through" is a
// property of the PATH, not of the current pose. recog_proto5 leg 5: the robot crossed DOOR_2 and walked on into the
// next room; door_39 (a fresh re-birth with a near-shut prior, p_open 1.28e-4) read p_past 1, p_span 1.6e-6 (the robot
// was by then 1.4 m sideways of the mouth) ⇒ p_cross 1.7e-9, no birth. This drives the agent's own PassageMemory
// along synthetic paths, exactly as RoomSceneGraph::step_proto_room_impl does (begin_frame, then evaluate per door).
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
constexpr float kHalfWidth = 0.23f;       // ROBOT_WIDTH 0.46 / 2

static rc::DoorAperture door(float p_open, Vector2f a = {-0.5f, 3}, Vector2f b = {0.5f, 3})
{
    rc::DoorAperture ap; ap.a = a; ap.b = b; ap.p_open = p_open; ap.sigma_along = 0.06f; ap.name = "door_test";
    return ap;
}

struct Run { float max_p = 0.f, last_p = 0.f, last_open_post = 0.f; cr::Crossing last; };

// One frame of the agent's loop (RoomSceneGraph::step_proto_room_impl): begin_frame with the odometry prediction, then
// evaluate. `est` = the localiser's estimate, `pred` = the odometry prediction of this frame.
static cr::Crossing frame(cr::PassageMemory& mem, const rc::DoorAperture& ap, Vector2f est, Vector2f pred,
                          bool jump_committed = false, const Eigen::Matrix2f& pred_cov = Eigen::Matrix2f::Zero())
{
    const Eigen::Matrix2f cov = Eigen::Matrix2f::Identity() * kSigPose * kSigPose;
    mem.begin_frame(est, cov, pred, cr::polygon_area(kRoom), jump_committed, pred_cov);
    return mem.evaluate(ap, kRoom, est, cov, kRBody, kSigLayout);
}

// Drive a TRUE path; the estimate is truth + the localiser's own jitter, the prediction is the previous estimate moved
// by the true step (perfect odometry). `door_at(k)` lets the door estimate change mid-run.
template <class DoorAt>
static Run drive_fn(DoorAt door_at, const std::vector<Vector2f>& path, cr::PassageMemory& mem, unsigned seed = 7)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> n(0.f, kSigPose);
    Run r;
    Vector2f prev_est = path.empty() ? Vector2f::Zero() : path.front(), prev_true = prev_est;
    for (std::size_t k = 0; k < path.size(); ++k)
    {
        const Vector2f est = path[k] + Vector2f(n(rng), n(rng));
        const Vector2f pred = prev_est + (path[k] - prev_true);
        r.last = frame(mem, door_at(k), est, pred);
        prev_est = est; prev_true = path[k];
        r.max_p = std::max(r.max_p, r.last.p_cross);
        r.last_p = r.last.p_cross;
        r.last_open_post = r.last.p_open_post;
    }
    return r;
}
static Run drive(const rc::DoorAperture& ap, const std::vector<Vector2f>& path, cr::PassageMemory& mem, unsigned seed = 7)
{ return drive_fn([&](std::size_t) { return ap; }, path, mem, seed); }

static void line(std::vector<Vector2f>& path, Vector2f from, Vector2f to, float step = 0.02f)
{
    const int n = std::max(1, static_cast<int>((to - from).norm() / step));
    for (int i = 0; i <= n; ++i) path.push_back(from + (to - from) * (static_cast<float>(i) / n));
}
static void park(std::vector<Vector2f>& path, Vector2f at, int cycles) { for (int i = 0; i < cycles; ++i) path.push_back(at); }

static cr::PassageMemory mem_new() { return cr::PassageMemory(kHalfWidth); }

int main()
{
    std::printf("(a) THE RECORDED CASE: door prior p_open 1.28e-4, path through the aperture to 0.85 m past\n");
    {
        std::vector<Vector2f> path;
        line(path, {0.f, 1.5f}, {0.f, 3.85f});
        park(path, {0.f, 3.85f}, 40);
        auto mem = mem_new();
        const auto r = drive(door(1.28e-4f), path, mem);
        std::printf("  at 0.85 m past: p_past %.4f p_through %.6f p_cross %.6f p_open_post %.4f\n",
                    r.last.p_past, r.last.p_through, r.last_p, r.last_open_post);
        check(r.last_p >= kBirth, "p_cross >= 0.95 (birth) despite the near-shut prior");
        check(r.last_open_post > 1.28e-4f * 10.f, "P(open | path) rose well above the prior");

        // …then on into the new room, sideways along it (leg 5: s 2.73, u -1.4)
        std::vector<Vector2f> on;
        line(on, {0.f, 3.85f}, {0.f, 5.7f});
        line(on, {0.f, 5.7f}, {-1.4f, 5.7f});
        const auto r2 = drive(door(1.28e-4f), on, mem, 11);
        std::printf("  2.7 m past, 1.4 m sideways: p_span(now) %.3g p_through %.6f p_cross %.6f\n",
                    r2.last.p_span, r2.last.p_through, r2.last_p);
        check(r2.last.p_span < 1e-6f, "the CURRENT centre is far outside the jambs (the live p_span)");
        check(r2.last_p >= kBirth, "the passage is remembered: p_cross >= 0.95");

        const auto r3 = cr::evaluate(door(1.28e-4f), kRoom, {-1.4f, 5.7f}, Eigen::Matrix2f::Identity() * kSigPose * kSigPose,
                                     kRBody, kSigLayout);
        std::printf("  same pose, no memory (the pre-Task-16 reading): p_cross %.3g\n", r3.p_cross);
        check(r3.p_cross < kBirth, "(without the memory the same pose would not birth: the defect)");

        // the door is re-estimated AFTER the crossing by a small amount (0.1 m out, 0.15 m along): the robot is still
        // past the new read, so the passage survives
        auto mem2 = mem_new();
        std::vector<Vector2f> all;
        line(all, {0.f, 1.5f}, {0.f, 5.7f});
        line(all, {0.f, 5.7f}, {-1.4f, 5.7f});
        const std::size_t half_way = all.size() / 2;
        const auto r4 = drive_fn([&](std::size_t k) { return k < half_way ? door(1.28e-4f)
                                                                         : door(1.28e-4f, {-0.35f, 3.1f}, {0.65f, 3.1f}); },
                                 all, mem2);
        std::printf("  aperture re-estimated after the passage: p_cross %.6f\n", r4.last_p);
        check(r4.last_p >= kBirth, "a re-estimate after the passage keeps it");
    }

    std::printf("(b) STOPS AT THE VANTAGE in front of a door, parked with pose jitter\n");
    for (const float po : {1.28e-4f, 0.5f, 0.99f})
    {
        std::vector<Vector2f> path;
        line(path, {0.f, 1.0f}, {0.f, 2.2f});
        park(path, {0.f, 2.2f}, 600);
        auto mem = mem_new();
        const auto r = drive(door(po), path, mem);
        std::printf("  p_open %.4g: max p_cross %.3g\n", po, r.max_p);
        check(r.max_p < kBirth, "no birth");
    }
    std::printf("    …and parked right AT the mouth of a shut door (centre 2 cm short of the line)\n");
    {
        std::vector<Vector2f> path;
        line(path, {0.f, 1.0f}, {0.f, 2.98f});
        park(path, {0.f, 2.98f}, 600);
        auto mem = mem_new();
        const auto r = drive(door(1.28e-4f), path, mem);
        std::printf("  max p_cross %.3g\n", r.max_p);
        check(r.max_p < kBirth, "no birth from jitter across the plane of a shut leaf");
    }

    std::printf("(c) DRIVES PAST alongside the wall, across the front of the door\n");
    for (const float po : {1.28e-4f, 0.99f})
    {
        std::vector<Vector2f> path;
        line(path, {-2.5f, 2.6f}, {2.5f, 2.6f});
        auto mem = mem_new();
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
        auto mem = mem_new();
        const auto r = drive(door(po), path, mem);
        std::printf("  p_open %.4g: final p_through %.3g p_cross %.3g\n", po, r.last.p_through, r.last_p);
        check(r.max_p < kBirth, "no birth: the path entered through the wall, not the door");
    }

    std::printf("(e) REVIEW I-1: the DOOR ESTIMATE jumps across the robot's in-room path\n");
    for (const float po : {1.28e-4f, 0.99f})
    {
        // robot drives inside the room toward the door and parks 0.9 m short; then the door estimate jumps from
        // y = 3 to y = 1.2 (into the room, behind the parked robot), so its in-room path now lies past it, in its jambs
        std::vector<Vector2f> path;
        line(path, {0.f, 0.5f}, {0.f, 2.1f});
        park(path, {0.f, 2.1f}, 200);
        const std::size_t jump_at = path.size() / 3;   // the robot is already parked at y 2.1
        auto mem = mem_new();
        const auto r = drive_fn([&](std::size_t k) { return k < jump_at ? door(po) : door(po, {-0.5f, 1.2f}, {0.5f, 1.2f}); },
                                path, mem);
        std::printf("  p_open %.4g: final p_past %.3f p_through %.3g max p_cross %.3g\n",
                    po, r.last.p_past, r.last.p_through, r.max_p);
        check(r.last.p_past > 0.99f, "(the robot IS past the jumped read — the setup)");
        check(r.max_p < kBirth, "no birth: the door moved, the robot did not pass it");
    }
    {
        // the harsher variant: the door jumps behind the robot in ONE frame while it stands still
        std::vector<Vector2f> path;
        line(path, {0.f, 0.5f}, {0.f, 2.1f});
        park(path, {0.f, 2.1f}, 200);
        const std::size_t jump_at = path.size() - 100;   // robot parked at y 2.1, door jumps 3 → 1.6
        auto mem = mem_new();
        const auto r = drive_fn([&](std::size_t k) { return k < jump_at ? door(0.99f) : door(0.99f, {-0.5f, 1.6f}, {0.5f, 1.6f}); },
                                path, mem);
        std::printf("  door jumps behind a parked robot, p_open 0.99: max p_cross %.3g\n", r.max_p);
        check(r.max_p < kBirth, "no birth");
    }

    std::printf("(f) REVIEW I-2: a RELOCALISATION jump across a door's line (odometry says the robot did not move)\n");
    for (const float po : {1.28e-4f, 0.99f})
        for (const bool epoch : {false, true})
        {
            auto mem = mem_new();
            std::vector<Vector2f> path;
            line(path, {0.f, 1.5f}, {0.f, 2.5f});
            const auto r0 = drive(door(po), path, mem);
            (void)r0;
            // the estimate jumps from (0, 2.5) to (0.1, 3.9); the robot is parked (pred = the last estimate)
            cr::Crossing c;
            float maxp = 0.f;
            Vector2f last = {0.f, 2.5f};
            c = frame(mem, door(po), {0.1f, 3.9f}, last, epoch);
            const double w_jump = mem.last_continuity();
            maxp = std::max(maxp, c.p_cross);
            for (int k = 0; k < 100; ++k)
            { c = frame(mem, door(po), {0.1f, 3.9f}, {0.1f, 3.9f}); maxp = std::max(maxp, c.p_cross); }
            std::printf("  p_open %.4g, epoch %s: continuity of the jump %.3g, p_through %.3g, max p_cross %.3g\n",
                        po, epoch ? "advanced" : "unchanged", w_jump, c.p_through, maxp);
            check(maxp < kBirth, "no birth: a teleport carries no passage");
        }
    {
        // the same jump size spread over ONE frame but with odometry that agrees (a genuinely fast robot can't do 1.4 m
        // in a frame, but a dropped stretch of frames can): pred agrees ⇒ continuity 1 ⇒ the passage counts
        auto mem = mem_new();
        std::vector<Vector2f> path;
        line(path, {0.f, 1.5f}, {0.f, 2.5f});
        (void)drive(door(0.99f), path, mem);
        const auto c = frame(mem, door(0.99f), {0.f, 3.9f}, {0.f, 3.9f});
        std::printf("  same displacement, odometry agrees (dropped frames): continuity %.3g, p_cross %.3g\n",
                    mem.last_continuity(), c.p_cross);
        check(c.p_cross >= kBirth, "a step the odometry explains is a traversal");
    }

    std::printf("(g) REVIEW MISSING: the BODY must clear the jambs\n");
    {
        auto mem = mem_new();
        std::vector<Vector2f> path;
        line(path, {0.f, 1.5f}, {0.f, 3.85f});
        const auto r = drive(door(0.99f, {-0.15f, 3}, {0.15f, 3}), path, mem);
        std::printf("  0.3 m aperture, 0.46 m body, p_open 0.99: p_through %.3g p_cross %.3g\n", r.last.p_through, r.last_p);
        check(r.last_p < kBirth, "a body wider than the aperture did not pass through it");
        auto mem2 = mem_new();
        const auto r2 = drive(door(0.99f), path, mem2);
        std::printf("  1 m aperture, same path: p_cross %.6f\n", r2.last_p);
        check(r2.last_p >= kBirth, "the 1 m aperture births");
        for (const float u : {0.1f, 0.2f})
        {
            auto m3 = mem_new();
            std::vector<Vector2f> p3;
            line(p3, {u, 1.5f}, {u, 3.85f});
            const auto a = drive(door(0.99f), p3, m3);
            auto m4 = mem_new();
            const auto b = drive(door(1.28e-4f), p3, m4);
            std::printf("  off-centre u %.2f (clearance %.2f m): p_cross open-prior %.4f, near-shut prior %.4f\n",
                        u, 0.5f - kHalfWidth - u, a.last_p, b.last_p);
        }
    }

    std::printf("(h) ROUND 2 I-1: the door estimate WOBBLES within its sigma while the robot crosses slowly\n");
    for (const float sig_door : {0.03f, 0.06f})
    {
        // robot at 0.1 m/s (5 mm per 20 Hz frame) from 1 m in front to 0.85 m past; every frame door_concept's read
        // of the opening moves by a zero-mean random offset of sig_door along AND across (a re-read, not a jump)
        std::mt19937 rng(3);
        std::normal_distribution<float> dn(0.f, sig_door);
        std::vector<Vector2f> path;
        line(path, {0.f, 2.0f}, {0.f, 3.85f}, 0.005f);
        park(path, {0.f, 3.85f}, 40);
        std::vector<rc::DoorAperture> reads;
        for (std::size_t k = 0; k < path.size(); ++k)
        {
            const Vector2f o(dn(rng), dn(rng));
            reads.push_back(door(1.28e-4f, Vector2f(-0.5f, 3) + o, Vector2f(0.5f, 3) + o));
        }
        auto mem = mem_new();
        const auto r = drive_fn([&](std::size_t k) { return reads[k]; }, path, mem);
        std::printf("  door wobble %.2f m, p_open 1.28e-4: p_past %.4f p_through %.6f p_cross %.6f\n",
                    sig_door, r.last.p_past, r.last.p_through, r.last_p);
        check(r.last_p >= kBirth, "still a birth: a re-drawn line within its sigma is the same opening");
    }

    std::printf("(j) ROUND 3: the door read DRIFTS/FLIPS 30 cm inward across a robot parked 15 cm short of the line\n");
    for (const int frames : {1, 10, 60})
    {
        std::vector<Vector2f> path;
        line(path, {0.f, 1.5f}, {0.f, 2.85f});
        const std::size_t t0 = path.size() + 20;
        park(path, {0.f, 2.85f}, 220);
        auto mem = mem_new();
        const auto r = drive_fn([&](std::size_t k)
        {
            const float f = k < t0 ? 0.f : std::min(1.f, static_cast<float>(k - t0 + 1) / frames);
            const float y = 3.f - 0.30f * f;   // the line moves 3.00 → 2.70, the robot sits at 2.85
            return door(0.99f, {-0.5f, y}, {0.5f, y});
        }, path, mem);
        std::printf("  over %d frame(s), p_open 0.99: final p_past %.3f p_through %.3g max p_cross %.3g\n",
                    frames, r.last.p_past, r.last.p_through, r.max_p);
        check(r.last.p_past > 0.95f, "(the robot IS past the moved read — the setup)");
        check(r.max_p < kBirth, "no birth: the door was re-read, the robot did not move through it");
    }
    {
        // and the door drifting metres in 2 cm steps across a parked robot
        std::vector<Vector2f> path;
        line(path, {0.f, 1.5f}, {0.f, 2.85f});
        const std::size_t t0 = path.size() + 20;
        park(path, {0.f, 2.85f}, 200);
        auto mem = mem_new();
        const auto r = drive_fn([&](std::size_t k)
        {
            const float y = k < t0 ? 3.f : std::max(0.8f, 3.f - 0.02f * static_cast<float>(k - t0 + 1));
            return door(0.99f, {-0.5f, y}, {0.5f, y});
        }, path, mem);
        std::printf("  door drifts 2.2 m in 2 cm steps, p_open 0.99: max p_cross %.3g\n", r.max_p);
        check(r.max_p < kBirth, "no birth");
    }

    std::printf("(i) ROUND 2 I-2: a DOORWAY POSE SNAP of 10 cm that the prediction's own noise explains\n");
    {
        // the localiser's posterior sigma is 1 cm (known to run ~3x tight); the prediction P_pred (with Q) says 5 cm.
        // In the mouth the view snaps the pose 10 cm forward across the line — the odometry predicted the old place.
        const Eigen::Matrix2f Ppred = Eigen::Matrix2f::Identity() * 0.05f * 0.05f;
        for (const bool with_pred_cov : {true, false})
        {
            auto mem = mem_new();
            std::vector<Vector2f> path;
            line(path, {0.f, 1.5f}, {0.f, 2.95f});
            (void)drive(door(1.28e-4f), path, mem);
            cr::Crossing c = frame(mem, door(1.28e-4f), {0.f, 3.05f}, {0.f, 2.95f}, false,
                                   with_pred_cov ? Ppred : Eigen::Matrix2f::Zero());
            const double w = mem.last_continuity();
            for (float y = 3.07f; y < 3.85f; y += 0.02f)
                c = frame(mem, door(1.28e-4f), {0.f, y}, {0.f, y}, false, with_pred_cov ? Ppred : Eigen::Matrix2f::Zero());
            std::printf("  %s: continuity of the snap %.3g, p_cross %.6f\n",
                        with_pred_cov ? "S = P_pred + Sigma (round 2)" : "S = Sigma_now + Sigma_prev (round 1)", w, c.p_cross);
            if (with_pred_cov)
            {
                check(w > 0.99, "the snap step itself counts as a traversal");
                check(c.p_cross >= kBirth, "birth");
            }
            else
                check(w < 0.01, "(round-1 covariance read the snap step as a jump — the defect: a shift carries no passage "
                                "and onward motion crosses nothing, so this branch does not birth; P_pred + Sigma is the cure)");
        }
    }

    std::printf("(k) ROUND 4: a RELOCALISATION jump lands a PARKED robot past the door, then 100 frames of 1 cm jitter with\n"
                "    agreeing (zero) odometry — the estimate dithers, the body never moved\n");
    for (const float land : {3.3f, 3.5f, 3.7f, 3.9f})
        for (const float po : {1.28e-4f, 0.5f, 0.99f})
        {
            auto mem = mem_new();
            std::vector<Vector2f> path;
            line(path, {0.f, 1.5f}, {0.f, 2.5f});
            (void)drive(door(po), path, mem);
            cr::Crossing c = frame(mem, door(po), {0.f, land}, {0.f, 2.5f}, true);   // the committed jump
            float maxp = c.p_cross;
            std::mt19937 rng(3);
            std::normal_distribution<float> n(0.f, kSigPose);
            Vector2f prev = {0.f, land};
            for (int k = 0; k < 100; ++k)
            {
                const Vector2f est = Vector2f(0.f, land) + Vector2f(n(rng), n(rng));
                c = frame(mem, door(po), est, prev);   // pred = the previous estimate: the odometry says "parked"
                prev = est; maxp = std::max(maxp, c.p_cross);
            }
            std::printf("  land %.1f m, p_open %.4g: p_through %.3g p_reread %.3g max p_cross %.3g\n",
                        land, po, c.p_through, c.p_reread, maxp);
            check(c.p_through < 1e-9f, "no passage mass beyond the approach's own tail: the odometric path never crossed");
            check(maxp < kBirth, "no birth from jitter after a teleport");
        }

    std::printf("(l) ROUND 5: a relocalisation jump past the door is NOT a passage, whatever the robot does next —\n"
                "    unless it drives back and genuinely crosses\n");
    for (const float po : {1.28e-4f, 0.99f})
    {
        struct Leg { const char* what; std::vector<Vector2f> steps; bool births; };
        const auto fwd = [](int n, Vector2f st) { return std::vector<Vector2f>(n, st); };
        auto back_and_through = fwd(200, {0.f, -0.005f});                       // 3.3 → 2.3: back through the door
        { auto on = fwd(300, {0.f, 0.005f}); back_and_through.insert(back_and_through.end(), on.begin(), on.end()); }   // → 3.8
        for (const Leg& leg : {Leg{"drives on 1.0 m (the jump put it past; onward motion crosses nothing)", fwd(200, {0.f, 0.005f}), false},
                               Leg{"drives SIDEWAYS 1.0 m", fwd(200, {0.005f, 0.f}), false},
                               Leg{"drives BACK 1.0 m through the door, then forward through it again", back_and_through, true}})
        {
            auto mem = mem_new();
            std::vector<Vector2f> path;
            line(path, {0.f, 1.5f}, {0.f, 2.5f});
            (void)drive(door(po), path, mem);
            cr::Crossing c = frame(mem, door(po), {0.f, 3.3f}, {0.f, 2.5f}, true);   // the committed jump
            std::mt19937 rng(5);
            std::normal_distribution<float> n(0.f, kSigPose);
            Vector2f prev = {0.f, 3.3f}, truth = {0.f, 3.3f};
            float maxp = c.p_cross;
            for (const Vector2f& st : leg.steps)
            {
                truth += st;
                const Vector2f est = truth + Vector2f(n(rng), n(rng));
                c = frame(mem, door(po), est, prev + st);
                prev = est; maxp = std::max(maxp, c.p_cross);
            }
            std::printf("  p_open %.4g, %s: at (%.2f, %.2f) p_through %.3g p_reread %.3g max p_cross %.3g\n", po, leg.what,
                        truth.x(), truth.y(), c.p_through, c.p_reread, maxp);
            check((maxp >= kBirth) == leg.births, leg.births ? "birth: the odometry carried the body through the jambs"
                                                             : "no birth: the jump is not a passage");
        }
    }

    std::printf("(m) ROUND 5: a 0.5 m jump ALONG the doorway direction mid-room, then a real crossing of a SECOND door\n");
    for (const float po : {1.28e-4f, 0.99f})
    {
        // door A at x ∈ [-0.5, 0.5] (as everywhere above), door B at x ∈ [1.5, 2.5], both in the +y wall
        const auto doorB = [&] { auto d = door(po, {1.5f, 3}, {2.5f, 3}); d.name = "door_B"; d.id = 2; return d; };
        const auto doorA = [&] { auto d = door(po); d.id = 1; return d; };
        auto mem = mem_new();
        const Eigen::Matrix2f cov = Eigen::Matrix2f::Identity() * kSigPose * kSigPose;
        const auto both = [&](Vector2f est, Vector2f pred, bool jump) -> std::pair<cr::Crossing, cr::Crossing>
        {
            mem.begin_frame(est, cov, pred, cr::polygon_area(kRoom), jump);
            return {mem.evaluate(doorA(), kRoom, est, cov, kRBody, kSigLayout),
                    mem.evaluate(doorB(), kRoom, est, cov, kRBody, kSigLayout)};
        };
        std::mt19937 rng(9);
        std::normal_distribution<float> n(0.f, kSigPose);
        std::vector<Vector2f> path;
        line(path, {0.f, 0.5f}, {0.f, 1.5f});                       // in front of door A, 1.5 m short
        Vector2f prev = path.front(), truth = path.front();
        float maxA = 0.f, maxB = 0.f;
        const auto run = [&](const std::vector<Vector2f>& pts)
        {
            for (const Vector2f& p : pts)
            {
                const Vector2f est = p + Vector2f(n(rng), n(rng));
                const auto [ca, cb] = both(est, prev + (p - truth), false);
                prev = est; truth = p; maxA = std::max(maxA, ca.p_cross); maxB = std::max(maxB, cb.p_cross);
            }
        };
        run(path);
        // the committed jump: 0.5 m along x (the doorway direction), odometry says parked
        truth = {0.5f, 1.5f};
        { const auto [ca, cb] = both(truth, prev, true); prev = truth; maxA = std::max(maxA, ca.p_cross); maxB = std::max(maxB, cb.p_cross); }
        std::vector<Vector2f> on;
        line(on, {0.5f, 1.5f}, {2.0f, 1.5f});                       // over to door B's axis
        line(on, {2.0f, 1.5f}, {2.0f, 3.85f});                      // through door B
        run(on);
        std::printf("  p_open %.4g: max p_cross door A %.3g, door B %.6f\n", po, maxA, maxB);
        check(maxA < kBirth, "door A: the jump along the wall is no passage");
        check(maxB >= kBirth, "door B: the real crossing after the jump births (the estimate is trusted after a reloc)");
    }

    std::printf("(d) frame move: PassageMemory::apply keeps the passage when the room frame is re-anchored\n");
    {
        std::vector<Vector2f> path;
        line(path, {0.f, 1.5f}, {0.f, 3.85f});
        auto mem = mem_new();
        (void)drive(door(1.28e-4f), path, mem);
        Eigen::Affine2f T = Eigen::Affine2f::Identity();
        T.rotate(0.4f); T.pretranslate(Vector2f(1.f, -2.f));
        mem.apply(T);
        std::vector<Vector2f> room2; for (const auto &p : kRoom) room2.push_back(T * p);
        rc::DoorAperture ap2 = door(1.28e-4f, T * Vector2f(-0.5f, 3), T * Vector2f(0.5f, 3));
        const Eigen::Matrix2f cov = Eigen::Matrix2f::Identity() * kSigPose * kSigPose;
        const Vector2f at = T * Vector2f(0.f, 3.9f);
        mem.begin_frame(at, cov, at, cr::polygon_area(room2), false);
        const auto c = mem.evaluate(ap2, room2, at, cov, kRBody, kSigLayout);
        std::printf("  after the move: p_cross %.6f\n", c.p_cross);
        check(c.p_cross >= kBirth, "still a birth in the moved frame");
    }

    std::printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
