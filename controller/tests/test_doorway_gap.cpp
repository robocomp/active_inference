// doorway_gap.h (Task 18): the doorway measured as the free gap between the jambs in the robot's own LiDAR returns, and the
// cross servo (controller_anchor_servo.h) steered on it, with the returns net (lidar_body_guard.h) in the chain.
// Build+run (from controller/; relative_pose.h pulls rt_twist.h → dsr_api.h, hence Qt/fastdds on the line):
//   Q=/usr/include/x86_64-linux-gnu/qt5; g++ -std=c++23 -O2 -Wall -Wextra -I/usr/include/eigen3 -I../common -Isrc
//     -I$Q -I$Q/QtCore -I$Q/QtGui -I$Q/QtWidgets -fPIC tests/test_doorway_gap.cpp -o /tmp/tdg
//     -lfastdds -lfastcdr -lQt5Core && /tmp/tdg        (one line)
#include "../src/doorway_gap.h"
#include "../src/controller_anchor_servo.h"
#include "../src/lidar_body_guard.h"

#include <chrono>
#include <cstdlib>
#include <cmath>
#include <print>
#include <random>
#include <string_view>
#include <vector>

namespace
{
int failures = 0, checks = 0;
void check(bool ok, int line, std::string_view what)
{
    ++checks;
    if (not ok) { ++failures; std::println("FAIL line {}: {}", line, what); }
}
#define CHECK(x) check((x), __LINE__, #x)

// ── A synthetic world: axis-aligned boxes in the WORLD frame (door wall along x at y = 0; room_1 is y < 0) ─────────────────
struct Box { float x0, y0, x1, y1; };
struct World { std::vector<Box> boxes; };

// A wall of thickness t along y = 0 from x = −L to +L with openings [o0, o1] (world x), and a closed room box around it.
World wall_with_openings(std::vector<std::pair<float, float>> openings, float t, bool room_box = true)
{
    World w;
    const float L = 4.f;
    std::ranges::sort(openings);
    float x = -L;
    for (const auto& [o0, o1] : openings) { if (o0 > x) w.boxes.push_back({x, -0.5f * t, o0, 0.5f * t}); x = o1; }
    if (x < L) w.boxes.push_back({x, -0.5f * t, L, 0.5f * t});
    if (room_box)
    {
        w.boxes.push_back({-L, -4.05f, L, -3.95f});   // room_1 back wall
        w.boxes.push_back({-L, 3.95f, L, 4.05f});     // room_2 far wall
        w.boxes.push_back({-L - 0.1f, -4.05f, -L, 4.05f});
        w.boxes.push_back({L, -4.05f, L + 0.1f, 4.05f});
    }
    return w;
}

struct Pose { float x, y, th; };   // world; robot frame x right, y forward; forward = R(th)·(0,1)

Eigen::Vector2f to_world(const Pose& p, const Eigen::Vector2f& q)
{
    const float c = std::cos(p.th), s = std::sin(p.th);
    return {p.x + c * q.x() - s * q.y(), p.y + s * q.x() + c * q.y()};
}
Eigen::Vector2f to_robot(const Pose& p, const Eigen::Vector2f& w)
{
    const float c = std::cos(p.th), s = std::sin(p.th);
    const Eigen::Vector2f d(w.x() - p.x, w.y() - p.y);
    return {c * d.x() + s * d.y(), -s * d.x() + c * d.y()};
}

// A 2-D LiDAR at the robot centre: `beams` rays, range noise sigma, max range. Returns robot-frame points.
std::vector<Eigen::Vector2f> scan(const World& w, const Pose& p, std::mt19937& rng, float noise = 0.01f, int beams = 1440,
                                  float max_range = 8.f)
{
    std::normal_distribution<float> nd(0.f, noise);
    std::vector<Eigen::Vector2f> out;
    const Eigen::Vector2f o(p.x, p.y);
    for (int k = 0; k < beams; ++k)
    {
        const float b = 2.f * std::numbers::pi_v<float> * float(k) / float(beams);
        const Eigen::Vector2f dr(std::sin(b), std::cos(b));               // robot frame, bearing 0 = forward, + right
        const Eigen::Vector2f d = to_world(p, dr) - o;                    // world direction
        float best = max_range;
        for (const auto& bx : w.boxes)
        {
            // slab method
            float t0 = 0.f, t1 = best;
            bool hit = true;
            for (int ax = 0; ax < 2 and hit; ++ax)
            {
                const float oo = ax == 0 ? o.x() : o.y(), dd = ax == 0 ? d.x() : d.y();
                const float lo = ax == 0 ? bx.x0 : bx.y0, hi = ax == 0 ? bx.x1 : bx.y1;
                if (std::abs(dd) < 1e-9f) { if (oo < lo or oo > hi) hit = false; continue; }
                float ta = (lo - oo) / dd, tb = (hi - oo) / dd;
                if (ta > tb) std::swap(ta, tb);
                t0 = std::max(t0, ta); t1 = std::min(t1, tb);
                if (t0 > t1) hit = false;
            }
            if (hit and t0 > 1e-4f and t0 < best) best = t0;
        }
        if (best < max_range) out.push_back(dr * (best + nd(rng)));
    }
    return out;
}

// The doorway prior as the anchor would give it (robot frame), from a WORLD door centre/normal, offset `lat_err` along the
// wall, with position sigma `s_xy`.
rc::DoorwayGapPrior prior_of(const Pose& p, Eigen::Vector2f door_world, float width, float s_xy, float lat_err = 0.f)
{
    rc::DoorwayGapPrior pr;
    door_world.x() += lat_err;
    pr.centre = to_robot(p, door_world);
    const float c = std::cos(p.th), s = std::sin(p.th);
    pr.normal = Eigen::Vector2f(s * 1.f, c * 1.f);   // world +y in the robot frame: Rᵀ(0,1) = (sin th, cos th)
    pr.P = Eigen::Matrix3f::Zero();
    pr.P(0, 0) = pr.P(1, 1) = s_xy * s_xy;
    pr.P(2, 2) = 0.02f * 0.02f;
    pr.width = width;
    return pr;
}

// Shadow-like body (mesh extents), CCW, robot frame.
const std::vector<Eigen::Vector2f> kBody{{0.242f, -0.2306f}, {0.242f, 0.2471f}, {-0.242f, 0.2471f}, {-0.242f, -0.2306f}};
constexpr float kHalfW = 0.242f, kDepth = 0.4777f, kSigmaR = 0.03f;
}  // namespace

int main()
{
    std::mt19937 rng(7);
    const Eigen::Vector2f door(0.f, 0.f);

    // (a) A 0.90 m opening in a thin wall, the robot 1.5 m out and 0.3 m off-centre: centre within 2 cm, width within 3 cm.
    {
        const auto w = wall_with_openings({{-0.45f, 0.45f}}, 0.02f);
        const Pose p{0.3f, -1.5f, 0.f};
        const auto pts = scan(w, p, rng);
        const auto g = rc::measure_doorway_gap(pts, prior_of(p, door, 0.9f, 0.05f), kHalfW, kSigmaR, kDepth);
        const Eigen::Vector2f truth = to_robot(p, door);
        std::println("(a) seen {} centre ({:+.3f},{:+.3f}) truth ({:+.3f},{:+.3f}) width {:.3f} normal ({:+.3f},{:+.3f}) score {:.3f}, {} candidates",
                     g.seen, g.centre.x(), g.centre.y(), truth.x(), truth.y(), g.width, g.normal.x(), g.normal.y(), g.score, g.candidates);
        CHECK(g.seen);
        CHECK((g.centre - truth).norm() < 0.02f);
        CHECK(std::abs(g.width - 0.90f) < 0.03f);
        CHECK(g.normal.y() > 0.98f);
        // No prior: nothing to say where along which wall to look — not measured.
        CHECK(not rc::measure_doorway_gap(pts, std::nullopt, kHalfW, kSigmaR, kDepth).seen);
        // A thick (0.10 m) wall: the along-wall centre and the width hold; the depth sits inside the wall's thickness.
        const auto wt = wall_with_openings({{-0.45f, 0.45f}}, 0.10f);
        const auto gt = rc::measure_doorway_gap(scan(wt, p, rng), prior_of(p, door, 0.9f, 0.05f), kHalfW, kSigmaR, kDepth);
        std::println("    thick wall: centre ({:+.3f},{:+.3f}) width {:.3f} normal ({:+.3f},{:+.3f})", gt.centre.x(), gt.centre.y(),
                     gt.width, gt.normal.x(), gt.normal.y());
        CHECK(gt.seen and std::abs(gt.centre.x() - truth.x()) < 0.02f and std::abs(gt.centre.y() - truth.y()) < 0.06f);
        CHECK(std::abs(gt.width - 0.90f) < 0.03f);
    }

    // (b) A second, 2 m opening further along the same wall; the prior points at the first: the one in the expected wall
    //     segment is picked. Even a prior
    //     0.2 m off at a claimed sigma of 2 cm (try74: 12 sigma) still picks the first — the gross-error branch.
    {
        const auto w = wall_with_openings({{-0.45f, 0.45f}, {0.9f, 2.9f}}, 0.10f);
        const Pose p{0.3f, -1.5f, 0.f};
        const auto pts = scan(w, p, rng);
        const Eigen::Vector2f truth = to_robot(p, door);
        const auto g = rc::measure_doorway_gap(pts, prior_of(p, door, 0.9f, 0.05f), kHalfW, kSigmaR, kDepth);
        std::println("(b) centre ({:+.3f},{:+.3f}) width {:.3f} score {:.3f}, {} candidates", g.centre.x(), g.centre.y(), g.width, g.score,
                     g.candidates);
        CHECK(g.seen and std::abs(g.centre.x() - truth.x()) < 0.03f and std::abs(g.width - 0.9f) < 0.03f);
        const auto gw = rc::measure_doorway_gap(pts, prior_of(p, door, 0.9f, 0.02f, 0.2f), kHalfW, kSigmaR, kDepth);
        std::println("    wrong prior (+0.2 m at 2 cm): centre ({:+.3f},{:+.3f}) score {:.3f}", gw.centre.x(), gw.centre.y(), gw.score);
        CHECK(gw.seen and std::abs(gw.centre.x() - truth.x()) < 0.03f);
        // The prior pointing at the SECOND (its width 2 m): the second is picked — the association follows the prior.
        const auto g2 = rc::measure_doorway_gap(pts, prior_of(p, Eigen::Vector2f(1.9f, 0.f), 2.0f, 0.05f), kHalfW, kSigmaR, kDepth);
        std::println("    prior at the 2 m opening: centre ({:+.3f},{:+.3f}) width {:.3f}", g2.centre.x(), g2.centre.y(), g2.width);
        CHECK(g2.seen and std::abs(g2.centre.x() - to_robot(p, Eigen::Vector2f(1.9f, 0.f)).x()) < 0.05f and g2.width > 1.9f);
    }

    // Edges are the INNERMOST return on each side (the one the returns net brakes on): with range noise sigma_n a face seen
    // along the ray is read up to ~2.5 sigma_n inward — conservative (the gap reads narrower), hence 3 cm on single edges.
    // (c) A 90° leaf hung from the WEST jamb into the robot's side (0.04 m thick, 0.9 m long): the clear gap is measured
    //     from the leaf face to the far jamb — 0.86 m, its left edge at the leaf face.
    {
        auto w = wall_with_openings({{-0.45f, 0.45f}}, 0.10f);
        w.boxes.push_back({-0.45f, -0.95f, -0.41f, -0.05f});
        const Pose p{0.3f, -1.5f, 0.f};
        const auto g = rc::measure_doorway_gap(scan(w, p, rng), prior_of(p, door, 0.9f, 0.05f), kHalfW, kSigmaR, kDepth);
        std::println("(c) left ({:+.3f},{:+.3f}) right ({:+.3f},{:+.3f}) width {:.3f}", g.left.x(), g.left.y(), g.right.x(), g.right.y(), g.width);
        CHECK(g.seen);
        CHECK(std::abs(g.width - 0.86f) < 0.03f);
        CHECK(std::abs(g.left.x() - to_robot(p, Eigen::Vector2f(-0.41f, 0.f)).x()) < 0.03f);
        CHECK(std::abs(g.right.x() - to_robot(p, Eigen::Vector2f(0.45f, 0.f)).x()) < 0.03f);
    }

    // (d) The robot IN the doorway (its centre on the wall plane, jambs beside the body): still found, edges at ±w/2.
    {
        const auto w = wall_with_openings({{-0.45f, 0.45f}}, 0.10f);
        for (const Pose p : {Pose{0.f, 0.f, 0.f}, Pose{0.05f, 0.1f, 0.1f}})
        {
            const auto g = rc::measure_doorway_gap(scan(w, p, rng), prior_of(p, door, 0.9f, 0.05f), kHalfW, kSigmaR, kDepth);
            const Eigen::Vector2f wl = to_world(p, g.left), wr = to_world(p, g.right);
            std::println("(d) pose ({:+.2f},{:+.2f},{:+.2f}): seen {} left world ({:+.3f},{:+.3f}) right ({:+.3f},{:+.3f}) width {:.3f}",
                         p.x, p.y, p.th, g.seen, wl.x(), wl.y(), wr.x(), wr.y(), g.width);
            CHECK(g.seen);
            CHECK(std::abs(wl.x() + 0.45f) < 0.03f and std::abs(wr.x() - 0.45f) < 0.03f);
            CHECK(std::abs(0.5f * (wl.x() + wr.x()) - p.x * 0.f) < 0.02f);
        }
    }

    // (e) No opening (a solid wall): not seen — with the anchor's prior at where a door would be, and without a prior.
    {
        const auto w = wall_with_openings({}, 0.10f);
        const Pose p{0.3f, -1.5f, 0.f};
        const auto pts = scan(w, p, rng);
        const auto g = rc::measure_doorway_gap(pts, prior_of(p, door, 0.9f, 0.05f), kHalfW, kSigmaR, kDepth);
        std::println("(e) seen {} score {:.4f} ({} candidates)", g.seen, g.score, g.candidates);
        CHECK(not g.seen);
        // A CLOSED door (the leaf in the opening, 4 cm thick, flush with the room_1 face): no ray through — not seen.
        auto wc = wall_with_openings({{-0.45f, 0.45f}}, 0.10f);
        wc.boxes.push_back({-0.45f, -0.05f, 0.45f, -0.01f});
        const auto gc = rc::measure_doorway_gap(scan(wc, p, rng), prior_of(p, door, 0.9f, 0.05f), kHalfW, kSigmaR, kDepth);
        CHECK(not gc.seen);
    }

    // (f) The door AJAR at 45° into the robot's side, hinged on the west jamb (0.9 m leaf, 0.04 m thick): no ray passes the
    //     hinge-side slit; the through-span runs from the leaf's FREE EDGE to the far jamb, and its width along the wall is
    //     the leaf-edge-to-far-jamb distance. The servo's P(body fits) then refuses it (0.26 m < the body).
    {
        auto w = wall_with_openings({{-0.45f, 0.45f}}, 0.10f);
        const float c45 = std::cos(0.785398f), L = 0.9f;
        // The leaf as small boxes along its centre line from the hinge (−0.43, −0.07) at −45° (into room_1, toward +x).
        const Eigen::Vector2f hinge(-0.43f, -0.07f), dir(c45, -c45);
        float edge_x = -1.f;   // the leaf's free edge: its largest x
        for (int i = 0; i <= 90; ++i)
        {
            const Eigen::Vector2f q = hinge + (L * float(i) / 90.f) * dir;
            w.boxes.push_back({q.x() - 0.02f, q.y() - 0.02f, q.x() + 0.02f, q.y() + 0.02f});
            edge_x = std::max(edge_x, q.x() + 0.02f);
        }
        const float expect = 0.45f - edge_x;
        // Viewpoints from which the slit beside the free edge is in view (straight in front of the leaf, at x 0, the free
        // edge hides all but ~0.5° of it — nothing to measure there, and the gap is then carried).
        for (const Pose p : {Pose{0.3f, -1.5f, 0.f}, Pose{0.6f, -1.5f, 0.f}, Pose{0.4f, -1.2f, 0.2f}})
        {
            const auto g = rc::measure_doorway_gap(scan(w, p, rng), prior_of(p, door, 0.9f, 0.05f), kHalfW, kSigmaR, kDepth);
            const Eigen::Vector2f wl = to_world(p, g.left), wr = to_world(p, g.right);
            std::println("(f) pose ({:+.2f},{:+.2f}): seen {} left world x {:+.3f} right x {:+.3f} width {:.3f} (leaf edge to jamb {:.3f})",
                         p.x, p.y, g.seen, wl.x(), wr.x(), g.width, expect);
            CHECK(g.seen);
            CHECK(std::abs(g.width - expect) < 0.03f);
            CHECK(std::abs(wr.x() - 0.45f) < 0.03f);
            const auto c = rc::anchor_cross_command(rc::doorway_gap_pose(g, kSigmaR), g.width, kHalfW + 0.05f, 2.0f, 0.35f, 1.0f, false);
            CHECK(c.adv < 0.01f);   // P(body fits) ≈ 0: no advance into a slit
        }
    }

    // Cost: one measurement on a full 1440-beam scan.
    {
        const auto w = wall_with_openings({{-0.45f, 0.45f}, {1.2f, 3.2f}}, 0.10f);
        const Pose p{0.3f, -1.5f, 0.f};
        const auto pts = scan(w, p, rng);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 20; ++i) (void)rc::measure_doorway_gap(pts, prior_of(p, door, 0.9f, 0.05f), kHalfW, kSigmaR, kDepth);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 20.0;
        std::println("cost: {:.2f} ms per measurement ({} returns)", ms, pts.size());
    }

    // SERVO INTEGRATION. A differential (unicycle) base 1.5 m out, 0.3 m off-centre, 10° heading error; a 0.90 m door with
    // a 90° leaf on the robot's side (west jamb). The anchor the cross starts from is 0.2 m off at a claimed 2 cm (try74).
    // Each cycle: scan → gap (prior = the anchor until the gap is held, then the carried gap) → anchor_cross_command on the
    // GAP → the jamb guard on the gap → the returns net. Lost gap ⇒ carried by the (true) motion. Ends `through` with no
    // contact of the body against any box.
    const auto integrate = [&](float noise, bool leaf_west, float x0, float th0_deg, float anchor_err, bool blind = false)
    {
        auto w = wall_with_openings({{-0.45f, 0.45f}}, 0.10f);
        if (leaf_west) w.boxes.push_back({-0.45f, -0.95f, -0.41f, -0.05f});
        else           w.boxes.push_back({0.41f, -0.95f, 0.45f, -0.05f});
        // Box samples every 5 mm for the contact test.
        std::vector<Eigen::Vector2f> solid;
        for (const auto& b : w.boxes)
            for (float x = b.x0; x <= b.x1 + 1e-6f; x += 0.005f)
                for (float y = b.y0; y <= b.y1 + 1e-6f; y += 0.005f)
                    if (std::abs(x) < 1.5f and std::abs(y) < 1.5f) solid.emplace_back(x, y);
        Pose p{x0, -1.5f, th0_deg * std::numbers::pi_v<float> / 180.f};
        const float dt = 0.05f, v_max = 0.35f, w_max = 1.0f, standoff = 2.0f, margin = 0.05f, a_dec = 1.0f;
        const float half_w_servo = kHalfW + margin;
        std::optional<rc::interaction::RelPose> line;
        float line_width = 0.f;
        bool through = false, contact = false;
        int seen_cycles = 0, lost_cycles = 0, k = 0;
        float min_clear = 1e9f;
        for (; k < int(90.f / dt) and not through; ++k)
        {
            const auto pts = scan(w, p, rng, noise);
            std::optional<rc::DoorwayGapPrior> pr;
            if (line.has_value())
            {
                pr = rc::DoorwayGapPrior{};
                const float psi = line->x.z();
                pr->centre = line->x.head<2>();
                pr->normal = Eigen::Vector2f(-std::sin(psi), std::cos(psi));
                pr->P = line->P;
                pr->width = 0.9f;   // the door's full width: the plausibility bound, not the last measurement
            }
            else
                pr = prior_of(p, door, 0.9f, 0.02f, anchor_err);
            // `blind`: the gap measurement sees NOTHING while the body's centre is within 0.5 m of the wall plane (the jambs beside
            // the body, out of the scan pattern) — the line must be carried by odometry; the returns net keeps its returns.
            const std::vector<Eigen::Vector2f> none;
            const bool blind_now = blind and std::abs(p.y) < 0.5f;
            const auto g = rc::measure_doorway_gap(blind_now ? std::span<const Eigen::Vector2f>(none) : std::span<const Eigen::Vector2f>(pts),
                                                   pr, kHalfW, kSigmaR, kDepth);
            if (g.seen) { line = rc::doorway_gap_pose(g, kSigmaR); line_width = g.width; ++seen_cycles; }
            else { ++lost_cycles; if (std::getenv("GAPDBG")) std::println("  lost at ({:+.2f},{:+.2f},{:+.0f}) cands {} best score {:.4f} w {:.3f} c ({:+.2f},{:+.2f})", p.x, p.y, p.th*57.3f, g.candidates, g.score, g.width, g.centre.x(), g.centre.y()); }
            CHECK(line.has_value() or k > 0);
            if (not line.has_value()) break;
            auto cmd = rc::anchor_cross_command(*line, line_width, half_w_servo, standoff, v_max, w_max, /*holonomic=*/false);
            if (cmd.through) { through = true; break; }
            const auto guard = rc::anchor_jamb_guard(*line, line_width, cmd, false, kBody, margin, a_dec, v_max, w_max);
            float adv = guard.cmd.adv, side = 0.f, rot = guard.cmd.rot;
            const auto ret = rc::lidar_guard_returns(pts, kBody, kSigmaR, 1.0f);
            (void)rc::lidar_cap_command(ret.obstacles, kBody, margin, a_dec, v_max, w_max, adv, side, rot);
            // Integrate the unicycle (forward = R(th)·(0,1)), and carry the gap line by the same motion.
            const float dth = rot * dt, ds = adv * dt;
            p.x += -std::sin(p.th + 0.5f * dth) * ds;
            p.y += std::cos(p.th + 0.5f * dth) * ds;
            p.th += dth;
            // The motion in the PREVIOUS robot frame: forward ds along a chord turned by dth/2.
            line = rc::doorway_gap_carry(*line, -std::sin(0.5f * dth) * ds, std::cos(0.5f * dth) * ds, dth);
            for (const auto& q : solid)
            {
                const Eigen::Vector2f qr = to_robot(p, q);
                if (qr.squaredNorm() > 1.0f) continue;
                const float sd = rc::anchor_signed_distance(kBody, qr);
                min_clear = std::min(min_clear, sd);
                if (sd < 0.f) contact = true;
            }
        }
        std::println("integration (noise {:.2f}, leaf {}, x0 {:+.2f}, heading {:+.0f} deg, anchor err {:+.2f}): through {} after {:.1f} s at world ({:+.3f},{:+.3f}) heading {:+.1f} deg; gap seen {} / lost {} cycles; "
                     "min body clearance {:.3f} m; contact {}", noise, leaf_west ? "west" : "east", x0, th0_deg, anchor_err, through, k * dt, p.x, p.y, p.th * 57.2958f, seen_cycles, lost_cycles,
                     min_clear, contact);
        CHECK(through);
        CHECK(not contact);
        CHECK(p.y > 1.5f);
    };
    integrate(0.01f, true, 0.3f, 10.f, 0.2f);     // the brief's case, with a try74-like wrong anchor
    integrate(0.02f, true, 0.3f, 10.f, 0.0f);     // noisier returns, a right anchor
    integrate(0.02f, false, -0.3f, -10.f, -0.5f); // leaf on the east jamb, mirrored start, a 0.5 m wrong anchor
    integrate(0.02f, true, 0.3f, 10.f, 0.2f, /*blind=*/true);   // the gap lost for the 1 m around the plane: carried
    // A random sweep: start offset ±0.35 m, heading ±15°, anchor error ±0.5 m, either leaf side, blind or not.
    {
        std::mt19937 r2(11);
        std::uniform_real_distribution<float> ux(-0.35f, 0.35f), uh(-15.f, 15.f), ue(-0.5f, 0.5f);
        for (int i = 0; i < 12; ++i)
            integrate(0.02f, (i % 2) == 0, ux(r2), uh(r2), ue(r2), (i % 3) == 0);
    }

    std::println("{} / {} checks passed", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
