// containment_test — CW1: the persistent path keeps every footprint corner inside the room (countable_containment.h), and the
// attachment (the class stand-off on the BACK face) is unchanged. Each case runs the SAME scene twice on this build, containment
// off (= BASE: the switch removes the only new term, so the engine is BASE's bit for bit) and on.
//
// Scene: a chair-like box (w 0.45 along the local x axis, depth h 0.50 along local y, H 0.90), the chair manifest's attachment
// (stand-off 0.05 ± 0.05 on the back face, placement p_against_wall 0.5), seen by the camera only from inside the room (no LiDAR):
// the faces turned toward the camera carry points, the faces toward the wall carry none, so the extent INTO the wall is held only
// by the class size prior (and by the stand-off, for a back face); the camera stands at (−2.4, −1.9), so two faces are in view. The room is the square [−3, 0]², walls x = 0 and y = 0 at the
// corner (0, 0). 40 parked updates per case.
//
// What each case asserts (stated before the run, from the model; the measured values are printed):
//  (a) side-on chair 5 cm from the y = 0 wall (yaw −π/2: its back faces along the wall, its SIDE faces the wall), class size = truth:
//      ON: no corner past the TRUE wall (the staged line moved by the wall latent's mean δ̂, as the engine evaluates the term) by
//      more than σ_s (the containment's own scale, Model.ModelSigmaM: the box surface's std about the real one); the side is not pulled flush: the side's gap ≥ 0.05 − 2σ_s (the points do not see it; the class width puts
//      it at 0.05); the centre stays at the points: no corner reaches the wall, so ON is OFF bit for bit.
//  (a2) the chair of (a) with its class size 0.10 m LARGER than the truth in both extents, and the 0.20 m of its side next to the
//      wall unseen (points only at y ≤ −0.25): the far extent is held only by the size prior, which puts it through the wall in
//      BASE. (The fit relabels it to a back-on mode, yaw ≈ 0, in BASE and AFTER alike: the data's quarter-turn choice.)
//      ON: no corner past the TRUE wall by more than σ_s, and the seen face does not move: |near face AFTER − BASE| ≤ 2σ_s (the
//      containment moves the unseen extent and the wall latent, not the face the points hold).
//  (b) back-on chair at its stand-off (back 5 cm off the y = 0 wall): every corner is 5 cm inside, so the containment term is 0
//      at every iterate the GN visits near the mode ⇒ ON equals OFF within float roundoff (|Δθ| ≤ 1e-5 m / rad, σ ratio 1 ± 1e-4).
//  (c) chair in the room corner whose class size is 0.10 m LARGER than the truth (w, h prior 0.55 / 0.60, truth 0.45 / 0.50),
//      side-on to x = 0 and back-on to y = 0, 5 cm off both: the unobserved extents are pulled toward the walls by the size
//      prior. With the polygon ON: no corner past EITHER (true) wall by more than σ_s. With only the staged wall's half-plane (no
//      polygon in the frame) the other wall is not in the region: printed, the gap to it is not asserted (it shows why the
//      polygon is staged).
//  (d) mid-room chair (1.5 m from every wall): ON and OFF bit-identical (max |Δ| = 0 on θ and on Σ).
// Run: ./containment_test
#include <cmath>
#include <cstdio>
#include <vector>
#include "countable_belief.h"
#include "countable_containment.h"

using namespace rc;
static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

namespace
{
constexpr float kW = 0.45f, kD = 0.50f, kH = 0.90f, kSigmaC = 0.0289f;
const std::vector<Eigen::Vector2f> kRoom{{-3.0f, -3.0f}, {0.0f, -3.0f}, {0.0f, 0.0f}, {-3.0f, 0.0f}};
const Eigen::Vector2f kInterior(-1.5f, -1.5f);

CountableBeliefParams params(float prior_w, float prior_d)
{
    CountableBeliefParams p;
    p.prior_footprint_m = prior_w; p.prior_depth_m = prior_d; p.prior_footprint_std = 0.05f;
    p.prior_height_m = kH; p.prior_height_std = 0.10f;
    p.plaus_height_min = 0.40f; p.plaus_height_soft = 0.08f;
    p.depth_unobs_precision = 0.0f;
    p.room_axis_std = 3.0f * 0.017453293f;
    p.volatility_infer = false;
    p.persistent = true;
    p.wall_standoff_m = 0.05f; p.wall_standoff_std_m = 0.05f;   // chair.concept.toml [prior.attachment]
    p.placement_p_against_wall = 0.5f;                           // chair.concept.toml [prior.placement]
    return p;
}

struct Truth
{
    float cx, cy, w, h, yaw;
    Eigen::Vector2f cam = Eigen::Vector2f(-2.4f, -1.9f);   // two faces in view
    float clip_y = 1e30f;                                   // points with y > clip_y are not seen (the part next to the wall hidden)
};

// Points on the faces of the truth box turned toward the camera (outward normal · (camera − face) > 0), a 10 × 9 grid per face.
std::vector<Eigen::Vector3f> visible_points(const Truth& t, const Eigen::Vector2f& cam)
{
    std::vector<Eigen::Vector3f> pts;
    const Eigen::Vector2f c(t.cx, t.cy), ux(std::cos(t.yaw), std::sin(t.yaw)), uy(-std::sin(t.yaw), std::cos(t.yaw));
    const struct { Eigen::Vector2f n, along; float off, len; } faces[4] = {
        { ux, uy, 0.5f * t.w, t.h}, {-ux, uy, 0.5f * t.w, t.h}, { uy, ux, 0.5f * t.h, t.w}, {-uy, ux, 0.5f * t.h, t.w}};
    for (const auto& f : faces)
    {
        const Eigen::Vector2f fc = c + f.off * f.n;
        if (not (f.n.dot(cam - fc) > 0.0f)) continue;
        for (int i = 0; i < 10; ++i)
            for (int j = 0; j < 9; ++j)
            {
                const Eigen::Vector2f q = fc + f.along * (f.len * ((i + 0.5f) / 10.0f - 0.5f));
                if (q.y() <= t.clip_y) pts.emplace_back(q.x(), q.y(), kH * (j + 0.5f) / 9.0f);
            }
    }
    return pts;
}

// The nearest polygon edge to q as the fitter stages it (CountableFitter::nearest_wall), edge index = its identity.
WallRef nearest_wall(const Eigen::Vector2f& q)
{
    WallRef w;
    float best = 1e30f;
    for (std::size_t i = 0; i < kRoom.size(); ++i)
    {
        const Eigen::Vector2f a = kRoom[i], ab = kRoom[(i + 1) % kRoom.size()] - a;
        const float t = std::clamp((q - a).dot(ab) / ab.squaredNorm(), 0.0f, 1.0f);
        const Eigen::Vector2f foot = a + t * ab;
        if ((q - foot).squaredNorm() < best)
        {
            best = (q - foot).squaredNorm();
            Eigen::Vector2f n(-ab.y(), ab.x());
            n.normalize();
            if (n.dot(kInterior - foot) < 0.0f) n = -n;
            w.ok = true; w.p = foot; w.n = n; w.sigma_m = 0.02f; w.length_m = ab.norm(); w.room = 1; w.edge = int(i);
        }
    }
    return w;
}

CountableFrame frame(const Truth& t, const CountableBelief& b, bool with_polygon)
{
    CountableFrame f;
    const Eigen::Vector2f cam = t.cam;
    f.points = visible_points(t, cam);
    f.R.assign(f.points.size(), b.params().sigma_base_m * b.params().sigma_base_m);
    f.z_top_obs = kH;
    f.wall = nearest_wall(b.back_centre(b.state()));
    if (with_polygon) f.room_polygon = kRoom;
    f.room_axis_ok = true; f.room_axis = 0.0f;
    const float c2 = kSigmaC * kSigmaC;
    f.sig2_cam = {c2, c2, c2, c2, c2, 0.03f * 0.03f};
    f.sig2_L = f.sig2_cam;
    f.sigma_pose = Eigen::Vector3d(1e-8, 1e-8, 0.0).asDiagonal();
    f.rho_cam.fill(1.0f); f.rho_L.fill(1.0f); f.rho_chain = 1.0f;
    f.robot_xy = cam;
    f.camera_xy = cam;
    return f;
}

struct Result
{
    CountableBeliefState s;
    Eigen::Matrix<float, 6, 6> S;
    float min_corner = 0.0f;                   // over every update: the smallest corner signed distance to the STAGED room polygon
    float min_true = 0.0f;                     // the same to the TRUE room: the polygon moved by the wall latent's mean δ̂ along n
};
Result run(const Truth& t, float prior_w, float prior_d, bool containment, bool with_polygon)
{
    // born near the truth (a birth seed is never exact): 2 cm and 2 degrees off
    CountableBelief b(CountableBeliefState{t.cx + 0.02f, t.cy - 0.02f, kH, prior_w, prior_d, t.yaw + 0.035f}, params(prior_w, prior_d));
    b.set_room_interior(kInterior);
    b.set_containment_for_test(containment);
    Result r;
    r.min_corner = r.min_true = 1e30f;
    for (int k = 0; k < 40; ++k)
    {
        const CountableFrame f = frame(t, b, with_polygon);
        b.update(f);
        const auto& s = b.state();
        r.min_corner = std::min(r.min_corner, containment::min_corner_distance(s.cx, s.cy, s.w, s.h, s.yaw, kRoom, false, {}, {}));
        // the true wall is the staged line + δ·n: the object relative to it is c − δ̂·n (the engine's θ + B_w b)
        const Eigen::Vector2f c = Eigen::Vector2f(s.cx, s.cy) - b.wall_offset_mean(f.wall) * f.wall.n;
        r.min_true = std::min(r.min_true, containment::min_corner_distance(c.x(), c.y(), s.w, s.h, s.yaw, kRoom, false, {}, {}));
    }
    r.s = b.state();
    r.S = b.covariance();
    return r;
}
float max_dtheta(const Result& a, const Result& b)
{ return (a.s.vec() - b.s.vec()).cwiseAbs().maxCoeff(); }
float max_dsigma_ratio(const Result& a, const Result& b)
{
    float m = 0.0f;
    for (int i = 0; i < 6; ++i) m = std::max(m, std::abs(std::sqrt(a.S(i, i) / b.S(i, i)) - 1.0f));
    return m;
}
void print(const char* name, const Result& r)
{
    std::printf("  %-30s cx %+.4f cy %+.4f w %.4f h %.4f yaw %+.4f  sd(cx,cy) %.4f %.4f  min corner-to-room staged %+.4f true %+.4f\n",
                name, r.s.cx, r.s.cy, r.s.w, r.s.h, r.s.yaw, std::sqrt(r.S(0, 0)), std::sqrt(r.S(1, 1)), r.min_corner, r.min_true);
}
}   // namespace

int main()
{
    std::printf("containment_test (CW1)\n");
    const float sig_s = CountableBeliefParams{}.model_sigma_m;
    constexpr float kHalfPi = 1.5707963f;

    // (a) side-on, 5 cm off y = 0: yaw π/2 puts the local x (width w) along the wall normal, the depth (back) along the wall
    {
        const Truth t{-1.5f, -(0.05f + 0.5f * kW), kW, kD, -kHalfPi};
        const Result off = run(t, kW, kD, false, true), on = run(t, kW, kD, true, true);
        std::printf(" (a) side-on chair, side 5 cm off the wall y = 0 (truth cy %+.4f)\n", t.cy);
        print("BASE (containment off)", off);
        print("AFTER", on);
        const float side_gap = -(on.s.cy + 0.5f * on.s.w * std::abs(std::sin(on.s.yaw)) + 0.5f * on.s.h * std::abs(std::cos(on.s.yaw)));
        std::printf("      AFTER: side gap to the wall %.4f (truth 0.05), |cy - truth| %.4f\n", side_gap, std::abs(on.s.cy - t.cy));
        CHECK(on.min_true >= -sig_s);
        CHECK(side_gap >= 0.05f - 2.0f * sig_s);
        // no corner ever reaches the wall, so the term is 0 at every iterate: the centre is where the points (and BASE) put it
        CHECK(on.s.vec() == off.s.vec() and on.S == off.S);
    }
    // (a2) see the header
    {
        Truth t{-1.5f, -(0.05f + 0.5f * kW), kW, kD, -kHalfPi};
        t.clip_y = -0.25f;
        const Result off = run(t, kW + 0.10f, kD + 0.10f, false, true), on = run(t, kW + 0.10f, kD + 0.10f, true, true);
        const auto near_face = [](const Result& r)   // the y of the face toward the camera (the box's extent along y, room side)
        { return r.s.cy - 0.5f * r.s.w * std::abs(std::sin(r.s.yaw)) - 0.5f * r.s.h * std::abs(std::cos(r.s.yaw)); };
        std::printf(" (a2) chair 5 cm off y = 0, class size +0.10 m over the truth, its part next to the wall unseen\n");
        print("BASE (containment off)", off);
        print("AFTER", on);
        std::printf("      near (seen) face y: BASE %+.4f AFTER %+.4f (truth %+.4f)\n", near_face(off), near_face(on), t.cy - 0.5f * kW);
        CHECK(on.min_true >= -sig_s);
        CHECK(std::abs(near_face(on) - near_face(off)) <= 2.0f * sig_s);
    }
    // (b) back-on at the stand-off: yaw 0, depth axis +y = toward the wall, back face 5 cm off
    {
        const Truth t{-1.5f, -(0.05f + 0.5f * kD), kW, kD, 0.0f};
        const Result off = run(t, kW, kD, false, true), on = run(t, kW, kD, true, true);
        std::printf(" (b) back-on chair at its stand-off (back 5 cm off y = 0)\n");
        print("BASE (containment off)", off);
        print("AFTER", on);
        std::printf("      max |dtheta| %.3g, max |sigma ratio - 1| %.3g\n", max_dtheta(on, off), max_dsigma_ratio(on, off));
        CHECK(max_dtheta(on, off) <= 1e-5f);
        CHECK(max_dsigma_ratio(on, off) <= 1e-4f);
    }
    // (c) room corner, class size 0.10 m larger than the truth: side-on to x = 0 (local x along +x) is yaw 0 with the width along x;
    //     back-on to y = 0 (depth axis +y). 5 cm off both walls.
    {
        const Truth t{-(0.05f + 0.5f * kW), -(0.05f + 0.5f * kD), kW, kD, 0.0f};
        const float pw = kW + 0.10f, pd = kD + 0.10f;
        const Result off = run(t, pw, pd, false, true), on = run(t, pw, pd, true, true), line = run(t, pw, pd, true, false);
        std::printf(" (c) chair in the room corner (0, 0), class size +0.10 m over the truth, 5 cm off both walls\n");
        print("BASE (containment off)", off);
        print("AFTER (room polygon)", on);
        print("AFTER (staged wall line only)", line);
        CHECK(on.min_true >= -sig_s);
    }
    // (d) mid-room
    {
        const Truth t{-1.5f, -1.5f, kW, kD, 0.3f, Eigen::Vector2f(-2.7f, -2.6f)};
        const Result off = run(t, kW, kD, false, true), on = run(t, kW, kD, true, true);
        std::printf(" (d) mid-room chair\n");
        print("BASE (containment off)", off);
        print("AFTER", on);
        const bool same = on.s.vec() == off.s.vec() and on.S == off.S;
        std::printf("      bit-identical: %s\n", same ? "yes" : "NO");
        CHECK(same);
    }
    std::printf(bad ? "containment_test: %d FAILED\n" : "containment_test: OK\n", bad);
    return bad != 0;
}
