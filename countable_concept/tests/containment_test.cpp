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
//      ON: no corner past the TRUE wall (the staged line moved by δ̂) by more than σ_c (the term's scale); the one-sided term never
//      pulls the side toward the wall (AFTER's side gap ≥ BASE's). Since fix round 2 (N2) the attachment's stand-off enters here and
//      pulls the side-on body's side toward the wall through back_centre — the attachment's behaviour, review I7 (parked), printed.
//  (a2) the chair of (a) with its class size 0.10 m LARGER than the truth in both extents, and the 0.20 m of its side next to the
//      wall unseen (points only at y ≤ −0.25): the far extent is held only by the size prior, which puts it through the wall in
//      BASE. (The fit relabels it to a back-on mode, yaw ≈ 0, in BASE and AFTER alike: the data's quarter-turn choice.)
//      ON: no corner past the TRUE wall by more than σ_s (asserted as written in round 0), and the seen face does not move: |near face AFTER − BASE| ≤ 2σ_s (the
//      containment moves the unseen extent and the wall latent, not the face the points hold).
//  (b) back-on chair at its stand-off (back 5 cm off the y = 0 wall): every corner is 5 cm inside, so the containment term is 0
//      at every iterate the GN visits near the mode ⇒ ON equals OFF within float roundoff (|Δθ| ≤ 1e-5 m / rad, σ ratio 1 ± 1e-4).
//  (c) chair in the room corner whose class size is 0.10 m LARGER than the truth (w, h prior 0.55 / 0.60, truth 0.45 / 0.50),
//      side-on to x = 0 and back-on to y = 0, 5 cm off both: the unobserved extents are pulled toward the walls by the size
//      prior. With the polygon ON: no corner past EITHER (true) wall by more than σ_s. With only the staged wall's half-plane (no
//      polygon in the frame) the other wall is not in the region: printed, the gap to it is not asserted (it shows why the
//      polygon is staged).
//  (d) mid-room chair (1.5 m from every wall): ON and OFF bit-identical (max |Δ| = 0 on θ and on Σ).
//  (e) fix round 1 (I5): a side-on chair HELD side-on (yaw prior 0.05), its extent toward the wall prior +0.10 m and the part next to
//      the wall unseen: the term is active. No corner past the true wall beyond σ_c; still side-on; the seen face within 2σ_s; the
//      centre within its σ of the truth.
//  (f) fix round 1 (I1/I2): a side-on chair whose points push it 3 cm past the staged wall, parked 200 updates, the move statistic on:
//      δ̂ does not travel toward D after the attachment's entry, the move statistic gives no evidence of moving (max jump_p ≤ the
//      prior P(moved)), the term holds it. Fix round 2: run with the parallel term off (M5). With it ON, update 1 caps (the
//      parallel residual at its maximum on a side-on body, with the active term: a concave energy the Gauss–Newton model cannot
//      follow) and, since N2, the attachment's entry is committed on update 2 instead of being lost.
//  (g) finite differences: the corner Jacobian, the polygon gradient, and (fix round 2, M-a) the FULL term E(θ) against −bd at a
//      convex room vertex where λ varies with θ.
//  (h) fix round 2 (N1, the reviewer's case): back-on flush objects whose points put the back D past the staged wall, the
//      attachment's entry converged on update 1 (the θ–δ correlation live), 200 updates, δ̂ at 1e-6: the support's own step does not
//      move δ (below 10 % of the review's leak κ·push, plus the solver's resolution), and δ̂ converges (no ratchet).
//  (m5) printed only: review M5 (parked) — the parallel term turns a side-on chair a quarter turn once the attachment's entry is
//      absorbed. (a), (e) and (f) run with the parallel term off so that they test CW1 and not M5.
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
// The STAGED room: the true room with its y = 0 wall published `shift` m inward (an outline whose edge lies inside the real wall).
std::vector<Eigen::Vector2f> staged_room(float shift)
{ return {{-3.0f, -3.0f}, {0.0f, -3.0f}, {0.0f, -shift}, {-3.0f, -shift}}; }

// Options of a run (defaults = the original cases).
struct Opts
{
    int   n = 40;              // parked updates
    float yaw_std = 0.60f;     // the class yaw prior (prior_yaw_std); a tight one holds a side-on body side-on
    bool  volatility = false;  // infer Q (the chair's [prior.permanence]) so the move statistic jump_p is computed
    float shift = 0.0f;        // the staged y = 0 wall lies this far inside the true one
    bool  trace = false;       // print δ̂, the corner gap and jump_p at a few updates
    float standoff_m = 0.05f, standoff_std_m = 0.05f;   // [prior.attachment] (chair by default)
    float p_against_wall = 0.5f;                         // [prior.placement] (chair; −1 = the class declares none)
    bool  parallel = true;     // the attachment's parallel term (wall_parallel_precision); off isolates CW1 from review M5 (parked)
};
const Eigen::Vector2f kInterior(-1.5f, -1.5f);

CountableBeliefParams params(float prior_w, float prior_d, const Opts& o = {})
{
    CountableBeliefParams p;
    p.prior_yaw_std = o.yaw_std;
    if (o.volatility)
    {   // chair.concept.toml [prior.permanence]
        p.move_rate_per_day = 3.0f; p.move_scale_m = 0.3f; p.move_scale_rad = 0.5f;
    }
    p.prior_footprint_m = prior_w; p.prior_depth_m = prior_d; p.prior_footprint_std = 0.05f;
    p.prior_height_m = kH; p.prior_height_std = 0.10f;
    p.plaus_height_min = 0.40f; p.plaus_height_soft = 0.08f;
    p.depth_unobs_precision = 0.0f;
    p.room_axis_std = 3.0f * 0.017453293f;
    p.volatility_infer = o.volatility;
    p.persistent = true;
    p.wall_standoff_m = o.standoff_m; p.wall_standoff_std_m = o.standoff_std_m;   // [prior.attachment] (chair by default)
    p.placement_p_against_wall = o.p_against_wall;                                // [prior.placement] (chair by default)
    if (not o.parallel) p.wall_parallel_precision = 0.0f;
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
WallRef nearest_wall(const Eigen::Vector2f& q, const std::vector<Eigen::Vector2f>& room)
{
    WallRef w;
    float best = 1e30f;
    for (std::size_t i = 0; i < room.size(); ++i)
    {
        const Eigen::Vector2f a = room[i], ab = room[(i + 1) % room.size()] - a;
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

CountableFrame frame(const Truth& t, const CountableBelief& b, bool with_polygon, float shift = 0.0f)
{
    const std::vector<Eigen::Vector2f> room = staged_room(shift);
    CountableFrame f;
    const Eigen::Vector2f cam = t.cam;
    f.points = visible_points(t, cam);
    f.R.assign(f.points.size(), b.params().sigma_base_m * b.params().sigma_base_m);
    f.z_top_obs = kH;
    f.wall = nearest_wall(b.back_centre(b.state()), room);
    if (with_polygon) f.room_polygon = room;
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
    std::vector<float> delta, gap;             // per update: δ̂ and the staged corner gap
    std::vector<double> jump;                  // per update: P(moved | data) (NaN: none computed)
    int gn_conv = 0;                           // updates whose GN converged
    bool conv1 = false;                        // the FIRST update's GN converged
    float sd1 = 0.0f;                          // the posterior σ_cy after the first update
};
Result run(const Truth& t, float prior_w, float prior_d, bool containment, bool with_polygon, const Opts& o = {})
{
    // born near the truth (a birth seed is never exact): 2 cm and 2 degrees off
    CountableBelief b(CountableBeliefState{t.cx + 0.02f, t.cy - 0.02f, kH, prior_w, prior_d, t.yaw + 0.035f}, params(prior_w, prior_d, o));
    b.set_room_interior(kInterior);
    b.set_containment_for_test(containment);
    Result r;
    r.min_corner = r.min_true = 1e30f;
    const std::vector<Eigen::Vector2f> room = staged_room(o.shift);
    for (int k = 0; k < o.n; ++k)
    {
        const CountableFrame f = frame(t, b, with_polygon, o.shift);
        b.update(f);
        const auto& s = b.state();
        const float staged = containment::min_corner_distance(s.cx, s.cy, s.w, s.h, s.yaw, room, false, {}, {});
        r.min_corner = std::min(r.min_corner, staged);
        // the wall the model believes in is the staged line + δ̂·n: the object relative to it is c − δ̂·n
        const float dl = b.wall_offset_mean(f.wall);
        const Eigen::Vector2f c = Eigen::Vector2f(s.cx, s.cy) - dl * f.wall.n;
        r.min_true = std::min(r.min_true, containment::min_corner_distance(c.x(), c.y(), s.w, s.h, s.yaw, room, false, {}, {}));
        r.delta.push_back(dl);
        r.jump.push_back(b.last_jump_p());
        r.gap.push_back(staged);
        r.gn_conv += b.last_gn_converged() ? 1 : 0;
        if (k == 0) { r.conv1 = b.last_gn_converged(); r.sd1 = std::sqrt(b.covariance()(1, 1)); }
        if (o.trace and (k == 0 or k == 1 or k == 2 or k == 9 or k == 49 or k == 99 or k == o.n - 1))
            std::printf("      update %3d: delta_hat %+.6f  staged corner gap %+.5f  jump_p %.4g  gn %d conv %d  sd_cy %.4f\n", k + 1, dl,
                        staged, b.last_jump_p(), b.last_gn_iters(), int(b.last_gn_converged()), std::sqrt(b.covariance()(1, 1)));
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
    const float sig_c = std::sqrt(sig_s * sig_s + 0.02f * 0.02f);   // the term's scale: σ_s² + the staged wall's σ² (the stand-in for every edge)
    constexpr float kQuarter = 0.7853982f;
    constexpr float kHalfPi = 1.5707963f;

    // (a) side-on, 5 cm off y = 0: yaw −π/2 puts the local x (width w) along the wall normal, the depth (back) along the wall
    {
        const Truth t{-1.5f, -(0.05f + 0.5f * kW), kW, kD, -kHalfPi};
        Opts oa; oa.parallel = false;   // M5 isolated: see (m5) below
        const Result off = run(t, kW, kD, false, true, oa), on = run(t, kW, kD, true, true, oa);
        std::printf(" (a) side-on chair, side 5 cm off the wall y = 0 (truth cy %+.4f)\n", t.cy);
        print("BASE (containment off)", off);
        print("AFTER", on);
        const auto side_gap = [](const Result& r)
        { return -(r.s.cy + 0.5f * r.s.w * std::abs(std::sin(r.s.yaw)) + 0.5f * r.s.h * std::abs(std::cos(r.s.yaw))); };
        std::printf("      side gap to the wall: BASE %+.4f AFTER %+.4f (truth 0.05); |cy - truth| AFTER %.4f\n", side_gap(off), side_gap(on),
                    std::abs(on.s.cy - t.cy));
        // Fix round 2: since N2 the attachment's entry is absorbed here (it was lost before, update 1 capping on the parallel term).
        // The stand-off then acts through back_centre of a side-on body (its centre's gap) and pulls the SIDE toward the wall — in
        // BASE 5.4 cm through it. That pull is the attachment's (review I7, parked: π × f / a back-face share), not the containment's.
        // What CW1 must guarantee, and is asserted: no corner past the true wall beyond the term's scale σ_c, and the one-sided term
        // never pulls toward the wall — AFTER's side gap is at least BASE's.
        CHECK(on.min_true >= -sig_c);
        CHECK(side_gap(on) >= side_gap(off));
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
    // (e) see the header: side-on, held side-on, class extent toward the wall +0.10, the part next to the wall unseen
    {
        Truth t{-1.5f, -(0.05f + 0.5f * kW), kW, kD, -kHalfPi};
        t.clip_y = -0.25f;
        Opts o; o.yaw_std = 0.05f; o.parallel = false;   // M5 isolated: see (m5)
        const Result off = run(t, kW + 0.10f, kD, false, true, o), on = run(t, kW + 0.10f, kD, true, true, o);
        const auto near_face = [](const Result& r)
        { return r.s.cy - 0.5f * r.s.w * std::abs(std::sin(r.s.yaw)) - 0.5f * r.s.h * std::abs(std::cos(r.s.yaw)); };
        std::printf(" (e) side-on chair HELD side-on (yaw prior 0.05 rad), class extent toward the wall +0.10 m, the part next to the wall unseen\n");
        print("BASE (containment off)", off);
        print("AFTER", on);
        const float sd_cy = std::sqrt(on.S(1, 1));
        std::printf("      near (seen) face y: BASE %+.4f AFTER %+.4f (truth %+.4f); |sin yaw| %.4f; |cy - truth| %.4f (sd_cy %.4f)\n",
                    near_face(off), near_face(on), t.cy - 0.5f * kW, std::abs(std::sin(on.s.yaw)), std::abs(on.s.cy - t.cy), sd_cy);
        CHECK(std::abs(std::sin(on.s.yaw)) > std::cos(kQuarter));        // still side-on: the side, not the back, faces the wall
        CHECK(off.min_true < -sig_c);                                     // BASE: the size prior puts it through the wall
        CHECK(on.min_true >= -sig_c);
        CHECK(std::abs(near_face(on) - (t.cy - 0.5f * kW)) <= 2.0f * sig_s);
        CHECK(std::abs(on.s.cy - t.cy) <= sd_cy);
    }
    // (f) see the header: side-on, its side ON the true wall, the staged wall 3 cm inside it, parked 200 updates with the move
    //     statistic on: the points push the corners 3 cm past the staged wall
    {
        constexpr float kD3 = 0.03f;
        const Truth t{-1.5f, -0.5f * kW, kW, kD, -kHalfPi};
        Opts o; o.yaw_std = 0.05f; o.volatility = true; o.shift = kD3; o.n = 200; o.trace = true; o.parallel = false;   // M5 isolated
        std::printf(" (f) side-on chair pushed %.2f m past the staged wall by its points, parked %d updates (move statistic on)\n", kD3, o.n);
        std::printf("    BASE (containment off):\n");
        const Result off = run(t, kW, kD, false, true, o);
        std::printf("    AFTER:\n");
        const Result on = run(t, kW, kD, true, true, o);
        print("BASE (containment off)", off);
        print("AFTER", on);
        // δ̂ after the attachment's one entry (update 1): the drift over updates 2..200 (the ratchet of review I1 would carry it
        // toward kD3 = 3 cm, where the data's corner is legal)
        const auto drift = [](const Result& r)
        { float m = 0.0f; for (std::size_t k = 1; k < r.delta.size(); ++k) m = std::max(m, std::abs(r.delta[k] - r.delta[0])); return m; };
        const auto max_jump = [](const Result& r)
        { double m = 0.0; for (std::size_t k = 1; k < r.jump.size(); ++k) if (std::isfinite(r.jump[k])) m = std::max(m, r.jump[k]); return m; };
        // the prior P(moved) per update: the class rate over one predict period (move_rate_per_day·dt/86400, chair 3/day, dt 0.1 s)
        const double p_move = 3.0 * double(CountableBeliefParams{}.predict_period_s) / 86400.0;
        std::printf("      delta_hat drift over updates 2..%d: BASE %.3g m AFTER %.3g m (delta_hat(1) BASE %+.5f AFTER %+.5f)\n", o.n, drift(off),
                    drift(on), off.delta.front(), on.delta.front());
        std::printf("      max jump_p over updates 2..%d: BASE %.3g AFTER %.3g (the prior P(moved) per update %.3g); GN converged BASE %d AFTER %d of %d\n",
                    o.n, max_jump(off), max_jump(on), p_move, off.gn_conv, on.gn_conv, o.n);
        std::printf("      staged corner gap at update 200: BASE %+.4f AFTER %+.4f; sd_cy BASE %.4f AFTER %.4f\n", off.gap.back(), on.gap.back(),
                    std::sqrt(off.S(1, 1)), std::sqrt(on.S(1, 1)));
        CHECK(on.gap.back() > off.gap.back());                            // the term holds it: less penetration than BASE
        CHECK(on.gap.back() >= -kD3);                                     // never further out than the points put it
        CHECK(drift(on) < 0.1f * kD3);                                    // δ̂ does not travel toward the data (no ratchet)
        CHECK(max_jump(on) <= p_move);                                    // the pushed parked object gives no evidence of moving
    }
    // (g) finite-difference checks of the corner Jacobian (∂p/∂θ, the yaw column included) and of the polygon gradient ∂d/∂q (an
    //     edge's interior and a convex room vertex)
    {
        float worst_j = 0.0f, worst_g = 0.0f;
        const float h = 1e-3f;
        for (float yaw : {-2.9f, -1.5708f, -0.4f, 0.0f, 0.7f, 2.2f})
        {
            const float cx = -1.2f, cy = -0.8f, w = 0.45f, hh = 0.5f;
            const auto c0 = containment::corners(cx, cy, w, hh, yaw);
            for (int i = 0; i < 6; ++i)
            {
                float v[6] = {cx, cy, kH, w, hh, yaw}, u[6] = {cx, cy, kH, w, hh, yaw};
                v[i] += h; u[i] -= h;
                const auto cp = containment::corners(v[0], v[1], v[3], v[4], v[5]), cm = containment::corners(u[0], u[1], u[3], u[4], u[5]);
                for (int k = 0; k < 4; ++k)
                {
                    const Eigen::Vector2f fd = (cp[k].p - cm[k].p) / (2.0f * h);
                    const Eigen::Vector2f an = containment::corner_jacobian(c0[k], w, hh, yaw).col(i);
                    worst_j = std::max(worst_j, (fd - an).norm());
                }
            }
        }
        for (const Eigen::Vector2f q : {Eigen::Vector2f(-1.0f, 0.03f), Eigen::Vector2f(0.04f, -1.7f), Eigen::Vector2f(0.03f, 0.05f),
                                        Eigen::Vector2f(-3.05f, -3.02f)})
        {
            const auto sd = containment::signed_distance(q, kRoom, false, {}, {});
            Eigen::Vector2f fd;
            for (int i = 0; i < 2; ++i)
            {
                Eigen::Vector2f a = q, b = q;
                a(i) += h; b(i) -= h;
                fd(i) = (containment::signed_distance(a, kRoom, false, {}, {}).d - containment::signed_distance(b, kRoom, false, {}, {}).d) / (2.0f * h);
            }
            worst_g = std::max(worst_g, (fd - sd.grad).norm());
        }
        std::printf(" (g) finite differences (h = 1e-3): corner Jacobian worst |fd - analytic| %.3g; polygon gradient worst %.3g\n", worst_j, worst_g);
        // both maps are smooth there (a rotation, and a distance away from the medial axis), so the central difference's error is
        // O(h²·curvature): the corner's yaw curvature is ≤ the half-diagonal 0.34 m ⇒ ≤ 0.34·h²/6 ≈ 6e-8, the distance's ≤ h²/d ≈ 3e-5;
        // the bound is float's resolution at these magnitudes (≈ 1e-7·|p|/h ≈ 3e-4)
        CHECK(worst_j <= 1e-3f);
        CHECK(worst_g <= 1e-3f);
        // the FULL term E(θ) against its gradient −bd (review M-a): a footprint whose corner is past the room's convex vertex (0, 0),
        // the staged edge (y = 0) carrying a wall-latent variance v = 1e-4 that differs from the stand-in 0.02² of the x = 0 edge, so
        // λ varies with θ there; and one past an edge's interior. Central differences in double over θ (h = 1e-4 m / rad).
        CountableBelief bel(CountableBeliefState{-0.3f, -0.3f, kH, kW, kD, 0.0f}, params(kW, kD));
        CountableEngineModel mdl; mdl.b = &bel;
        CountableFrame fr;
        fr.room_polygon = kRoom;
        fr.wall = nearest_wall(Eigen::Vector2f(-0.3f, 0.0f), kRoom);   // the y = 0 edge
        const rc::ai::engine::WallOffset wo{0.004, 1e-4};
        const auto energy = [&](const CountableBeliefState& st)
        {
            CountableEngineModel::Mat6 I = CountableEngineModel::Mat6::Zero(); CountableEngineModel::Vec6 g = CountableEngineModel::Vec6::Zero();
            double E = 0.0;
            mdl.accumulate_containment(st, fr, wo, I, g, &E);
            return E;
        };
        float worst_e = 0.0f, scale = 0.0f;
        for (const CountableBeliefState st : {CountableBeliefState{-0.20f, -0.22f, kH, kW, kD, 0.3f},      // a corner past the vertex
                                             CountableBeliefState{-1.20f, -0.20f, kH, kW, kD, 0.1f}})     // past the edge interior
        {
            CountableEngineModel::Mat6 I = CountableEngineModel::Mat6::Zero(); CountableEngineModel::Vec6 g = CountableEngineModel::Vec6::Zero();
            mdl.accumulate_containment(st, fr, wo, I, g, nullptr);
            for (int i = 0; i < 6; ++i)
            {
                const double h2 = 1e-4;
                auto vp = st.vec(), vm = st.vec();
                vp(i) += float(h2); vm(i) -= float(h2);
                const double fd = (energy(CountableBeliefState::from_vec(vp)) - energy(CountableBeliefState::from_vec(vm))) / (double(vp(i)) - double(vm(i)));
                worst_e = std::max(worst_e, float(std::abs(fd - (-double(g(i))))));
                scale = std::max(scale, std::abs(g(i)));
            }
        }
        std::printf("     full term: worst |dE/dtheta (fd) - (-bd)| %.3g against max |bd| %.3g (relative %.2g)\n", worst_e, scale, worst_e / scale);
        // the relative bound: float states (θ rounded to ~1e-7 of the 1 m scale) differenced over 2e-4 give ~1e-3 of a unit-order
        // gradient; a missing ½d²∂λ/∂q term is O(½λΔv) ≈ 0.4 relative here (the review's estimate), two orders above it.
        CHECK(worst_e <= 1e-2f * scale);
    }
    // (h) see the header: the reviewer's N1 case — a back-on object whose back is on the true wall and whose staged wall lies D inside
    //     it, so its points put the back D past the staged wall; the attachment's entry CONVERGES on update 1, so the prior carries a
    //     θ–δ correlation from then on. 200 parked updates, δ̂ printed at 1e-6. Two classes: the nightstand (stand-off 0.03 ± 0.02, D
    //     = 2.5 and 6 cm) and the chair (0.05 ± 0.05, D = 6 cm). Where the support is active at the end (true corner gap ≤ 0), the
    //     support must add no δ̂ drift beyond BASE's own (the attachment's and the data's coupling, which CW1 does not touch).
    {
        struct Case { const char* name; float w, d, so, so_std, pi, D; };
        for (const Case c : {Case{"nightstand", 0.50f, 0.40f, 0.03f, 0.02f, -1.0f, 0.025f}, Case{"nightstand", 0.50f, 0.40f, 0.03f, 0.02f, -1.0f, 0.06f},
                             Case{"chair", 0.50f, 0.40f, 0.05f, 0.05f, 0.5f, 0.06f}})
        {
            const Truth t{-1.5f, -0.5f * c.d, c.w, c.d, 0.0f};
            Opts o; o.yaw_std = 0.05f; o.volatility = true; o.shift = c.D; o.n = 200; o.trace = true;
            o.standoff_m = c.so; o.standoff_std_m = c.so_std; o.p_against_wall = c.pi;
            std::printf(" (h) back-on flush %s (stand-off %.2f +- %.2f), back %.3f m past the staged wall by its points, %d updates\n", c.name, c.so,
                        c.so_std, c.D, o.n);
            std::printf("    BASE (containment off):\n");
            const Result off = run(t, c.w, c.d, false, true, o);
            std::printf("    AFTER:\n");
            const Result on = run(t, c.w, c.d, true, true, o);
            print("BASE (containment off)", off);
            print("AFTER", on);
            const auto drift = [](const Result& r)
            { float m = 0.0f; for (std::size_t k = 1; k < r.delta.size(); ++k) m = std::max(m, std::abs(r.delta[k] - r.delta[0])); return m; };
            std::printf("      delta_hat(1) BASE %+.6f AFTER %+.6f; drift over updates 2..%d: BASE %.6f AFTER %.6f m; update-1 GN converged BASE %d AFTER %d\n",
                        off.delta.front(), on.delta.front(), o.n, drift(off), drift(on), int(off.conv1), int(on.conv1));
            // (1) the support's own step leaves δ alone. The review's leak is κ·Δg with κ = σ_w²/(σ_w² + σ_so²) (the attachment's
            //     correlation slope along n) and Δg the support's push on the back at update 1 (AFTER − BASE staged corner gap); the
            //     decoupling removes it exactly per linearisation, so what remains is the GN's nonlinearity, asserted below 10 % of
            //     it, plus the solver's own resolution: the stop rule accepts θ within tol_sigma·σ (0.1·σ_cy), and a θ difference
            //     that small moves δ̂ by κ times it through the same correlation in either run.
            // (2) no ratchet: δ̂ converges — the change over updates 100..200 is below 1 % of the change over 1..100 (a ratchet is
            //     linear in the update count, ratio ≈ 1).
            const float sw2 = 0.02f * 0.02f, kap = sw2 / (sw2 + c.so_std * c.so_std);
            const float push = std::abs(on.gap.front() - off.gap.front()), leak1 = std::abs(on.delta.front() - off.delta.front());
            const float late = std::abs(on.delta[199] - on.delta[99]), early = std::abs(on.delta[99] - on.delta[0]);
            std::printf("      update-1 push %.5f m, review's leak k*push %.3g m, measured |delta_hat AFTER - BASE| %.3g m; delta_hat change 100..200 %.3g vs 1..100 %.3g m\n",
                        push, kap * push, leak1, late, early);
            CHECK(on.conv1);                                              // the entry converged: the θ–δ correlation is live
            const float res = kap * float(CountableBeliefParams{}.gn_tol_sigma) * on.sd1;
            std::printf("      solver resolution on delta_hat k*tol_sigma*sd_cy %.3g m\n", res);
            CHECK(leak1 <= 0.1f * kap * push + res);
            CHECK(late <= 0.01f * early + 1e-6f);
            CHECK(on.gap.back() >= -c.D);
        }
    }
    // (m5) REVIEW M5, parked for the owner, printed only: the chair of (a) with the attachment's parallel term ON. Since N2 the
    //      attachment's entry is no longer lost when update 1 caps, so the parallel residual par = ux·n, at its MAXIMUM for a
    //      side-on body (|par| = 1, energy ≈ ½·f·(L/σ_w)² ≈ 7760 nats, zero gradient exactly there), turns the chair a quarter turn
    //      to back-on as soon as the yaw is off the exact maximum. BASE (no CW1) does the same: this is the attachment, not the
    //      containment. Before N2 the cap had hidden it here.
    {
        const Truth t{-1.5f, -(0.05f + 0.5f * kW), kW, kD, -kHalfPi};
        const Result off = run(t, kW, kD, false, true), on = run(t, kW, kD, true, true);
        std::printf(" (m5) the side-on chair of (a) with the parallel term ON (printed only; review M5, parked)\n");
        print("BASE (containment off)", off);
        print("AFTER", on);
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
