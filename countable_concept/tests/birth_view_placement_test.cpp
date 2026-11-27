// birth_view_placement_test — CB1: the chair's birth placement and its yaw mode.
//
// (a) render_middle: the PREDICTED cloud middle of a wall chair, by range and bearing (printed), from the class solid along the
//     view's rays; a front view's middle sits in front of the body centre (the backrest face and the seat front are what it sees),
//     a side view's at it (both ends of the depth are seen). A class whose solid is the box (the nightstand) renders as its box.
// (b) the placement posterior on the G1x first views RECONSTRUCTED from the logs and dumps (camera position relative to the wall,
//     cloud middle, robot pose covariance, room span; the numbers are quoted per case): a synthetic class chair, back to the wall,
//     whose rendered cloud has the logged middle, seen from the logged camera. The five FREE-born outliers are placed AGAINST THE
//     WALL with the view (S12 chair_71 is the exception, stated), and the correct wall births of S01/S02 stay on the wall; mid-room
//     chairs stay free. BASE's comparison (no view) on the same clouds is printed: it put the outliers FREE.
// (c) the mode prior: a chair held 180° and 90° wrong at the wall is turned to back-on-the-wall by the update's adoption
//     (resolve_front_geometric); the posterior goes to uniform as the flush weight goes to 0 (chair moved off the wall); a chair the
//     birth found free (its placement posterior, handed to the belief, is 0) gets no mode prior and is never turned; with no
//     declared placement or with π = 1/2 the clearance prior is BASE's number for a body held correctly.
// Run: LANG=es_ES.UTF-8 ./birth_view_placement_test
#include <clocale>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include <genericworker.h>   // the test stub: the real ConfigLoader (manifest_stub/)
#include "countable_birth_seed.h"
#include "countable_config.h"

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

static rc::CountableConfig load(const std::string& dir, const std::string& file)
{
    std::filesystem::current_path(std::filesystem::path(REPO_DIR) / dir);
    ConfigLoader cl;
    cl.load(file);
    return rc::load_countable_config(cl);
}

namespace
{
constexpr float kZed = 0.0325f + 0.945f;   // ROBOT_GEOMETRY.md: the ZED above the floor
// The wall: y = 0, normal +y into the room; along it is −x (t = (−n.y, n.x)).
rc::WallRef the_wall()
{
    rc::WallRef w; w.ok = true; w.p = {0.f, 0.f}; w.n = {0.f, 1.f}; w.length_m = 6.f; w.sigma_m = 0.02f; return w;
}
float back_to_wall_yaw(const rc::WallRef& w) { return std::atan2(w.n.x(), -w.n.y()); }   // as rc::birth_footprint

// A synthetic first view: the hits of rays from `cam` through a grid over the body's bounding box (the mask's silhouette) on the
// class solid at θ — the visible surface, no noise.
std::vector<Eigen::Vector3f> view_cloud(const Eigen::Vector3f& cam, const rc::CountableBeliefState& th, const rc::CountableConfig& cfg)
{
    rc::BirthSolid so;
    for (const auto& pr : cfg.shape_proportions) so.q.push_back(0.5 * (double(pr.lo) + double(pr.hi)));
    const auto T = so.table(cfg);
    const rc::countable::VerticalSpan span{cfg.vertical_extent_m};
    std::vector<Eigen::Vector3f> out;
    const float R = 0.5f * std::sqrt(th.w * th.w + th.h * th.h);
    for (int i = 0; i <= 40; ++i) for (int j = 0; j <= 40; ++j) for (int k = 0; k <= 30; ++k)
    {
        const Eigen::Vector3f tgt(th.cx - R + 2.f * R * float(i) / 40.f, th.cy - R + 2.f * R * float(j) / 40.f, th.H * float(k) / 30.f);
        Eigen::Vector3f d = (tgt - cam).normalized();
        float t = 0.f;
        for (int it = 0; it < 200 and t < 30.f; ++it)
        {
            const Eigen::Vector3f x = cam + t * d;
            const float s = rc::ai::parts::sdf(x, th.cx, th.cy, th.H, th.w, th.h, th.yaw, T, span);
            if (s < 1e-4f) { out.push_back(x); break; }
            t += s;
        }
    }
    return out;
}

float cloud_mid(const std::vector<Eigen::Vector3f>& pts, const rc::WallRef& w, float z_lo, float z_hi)
{
    std::vector<float> v;
    for (const auto& p : pts) if (p.z() >= z_lo and p.z() <= z_hi) v.push_back((p.head<2>() - w.p).dot(w.n));
    std::sort(v.begin(), v.end());
    return 0.5f * (v[std::size_t(0.05f * float(v.size() - 1))] + v[std::size_t(0.95f * float(v.size() - 1))]);
}

// The cold start, as CountableFitter::run_inference builds it.
struct Cold { rc::CountableBeliefState s0; float z_lo, z_hi; };
Cold cold_start(const std::vector<Eigen::Vector3f>& pts, const rc::CountableConfig& cfg)
{
    Cold c{};
    std::vector<float> zs;
    for (const auto& p : pts) zs.push_back(p.z());
    const std::size_t k = std::size_t(0.95f * float(zs.size() - 1));
    std::nth_element(zs.begin(), zs.begin() + std::ptrdiff_t(k), zs.end());
    c.s0.H = zs[k];
    c.z_hi = c.s0.H + 3.f * cfg.ai2_sigma_base_m;
    c.z_lo = rc::countable::VerticalSpan{cfg.vertical_extent_m}.z0(c.s0.H) + 3.f * cfg.ai2_sigma_base_m;
    const auto mom = rc::CountableBelief::footprint_moment(pts, c.z_lo, c.z_hi);
    c.s0.cx = mom.cx; c.s0.cy = mom.cy; c.s0.w = std::max(0.1f, mom.ext_major);
    c.s0.h = mom.ext_minor > cfg.ai2_depth_obs_band_m ? mom.ext_minor : cfg.ai2_prior_depth_m;
    c.s0.yaw = mom.phi;
    return c;
}

struct Case
{
    const char* name;
    float cam_across, cam_range, mid;      // the log's "cloud middle", the dump's camera_xy against the staged wall
    float sx, sy, spsi;                    // the dump's robot pose std (x, y, ψ) at the capture stamp
    float span;                            // the room's span along the normal (2·(interior − wall)·n, the dump)
    bool expect_wall;
};
}  // namespace

int main()
{
    std::setlocale(LC_ALL, "");
    std::printf("birth_view_placement_test\n");
    const rc::CountableConfig cc = load("chair_concept", "etc/countable.toml");
    const rc::WallRef wall = the_wall();
    const float d_bar = cc.ai2_prior_depth_m, w_bar = cc.ai2_prior_footprint_m, H_bar = cc.ai2_prior_height_m;
    const float yaw_w = back_to_wall_yaw(wall);
    const auto wall_chair = [&](float across) { return rc::CountableBeliefState{0.f, across, H_bar, w_bar, d_bar, yaw_w}; };

    // ── (a) the rendered middle of a wall chair (back at the stand-off) by range, front and side views ──────────────────────────
    {
        const auto th = wall_chair(cc.wall_standoff_m + 0.5f * d_bar);
        const float centre = th.cy;
        rc::BirthSolid so;
        for (const auto& pr : cc.shape_proportions) so.q.push_back(0.5 * (double(pr.lo) + double(pr.hi)));
        std::printf("  (a) wall chair, centre %.3f m off the wall; predicted cloud middle (p5+p95)/2 by range:\n", centre);
        std::printf("      range  front(0 deg)  oblique(45)  side(75)\n");
        bool front_ahead = true, side_at_centre = true;
        for (const float r : {1.5f, 2.5f, 3.0f, 3.5f, 4.0f, 5.0f})
        {
            float m[3];
            const float bearings[3] = {0.f, 45.f, 75.f};
            for (int b = 0; b < 3; ++b)
            {
                const float a = bearings[b] * 0.017453293f;
                const Eigen::Vector3f cam(-r * std::sin(a), centre + r * std::cos(a), kZed);
                const auto pts = view_cloud(cam, th, cc);
                const auto rm = rc::render_middle(pts, cam, th, so.table(cc), rc::countable::VerticalSpan{cc.vertical_extent_m}, wall,
                                                  3.f * cc.ai2_sigma_base_m, th.H + 3.f * cc.ai2_sigma_base_m, cc.ai2_sigma_base_m,
                                                  int(pts.size()));
                m[b] = rm.ok ? float(rm.mid) : NAN;
            }
            std::printf("      %4.1f   %.3f         %.3f        %.3f\n", r, m[0], m[1], m[2]);
            front_ahead = front_ahead and m[0] > centre;
            side_at_centre = side_at_centre and std::abs(m[2] - centre) < std::abs(m[0] - centre);
        }
        CHECK(front_ahead);       // the front view sees the backrest face and the seat front: its middle is in front of the centre
        CHECK(side_at_centre);    // a side view sees both ends of the depth: its middle is nearer the centre than the front view's
        // a class without a parts model renders as its box (parts::sdf's envelope branch): the nightstand
        const rc::CountableConfig nc = load("nightstand_concept", "etc/config.toml");
        const rc::CountableBeliefState nb{0.f, 0.3f, 0.55f, 0.5f, 0.4f, yaw_w};
        const Eigen::Vector3f cam(0.f, 3.3f, kZed);
        const auto pts = view_cloud(cam, nb, nc);
        rc::BirthSolid none;
        const auto rm = rc::render_middle(pts, cam, nb, none.table(nc), rc::countable::VerticalSpan{nc.vertical_extent_m}, wall, 0.f,
                                          1.f, nc.ai2_sigma_base_m, int(pts.size()));
        std::printf("  (a) nightstand (solid %s): %d hits, rendered middle %.3f (box front face at %.3f)\n", nc.solid_parts ? "parts" : "box",
                    rm.n_hit, rm.mid, 0.3f + 0.2f);
        CHECK(not nc.solid_parts and rm.ok and rm.n_hit > 0);
    }

    // ── (b) the reconstructed G1x first views ──────────────────────────────────────────────────────────────────────────────────
    // cam_across / cam_range / mid: the dump's first frame against its staged wall (tools/replay/birth_placement); pose std: the
    // frame's sigma_pose diagonal; span: 2·(room interior − wall)·n from the dump's RoomInteriorEv.
    const Case cases[] = {
        {"S03 chair_18 (chair(5), FREE-born, 180 deg)", 0.991f, 3.777f, 0.403f, 0.0293f, 0.0210f, 0.0107f, 5.51f, true},
        {"S11 chair_64 (chair(5), FREE-born, 180 deg)", 1.407f, 3.806f, 0.422f, 0.0502f, 0.0298f, 0.0144f, 5.51f, true},
        {"S12 chair_72 (chair(3), FREE-born, 180 deg)", 5.371f, 5.197f, 0.423f, 0.0659f, 0.0652f, 0.0143f, 5.46f, true},
        {"S12 chair_73 (chair(4), FREE-born, 180 deg)", 1.913f, 2.211f, 0.545f, 0.0834f, 0.0775f, 0.0288f, 7.68f, true},
        {"S01 chair_2  (chair(5), wall-born, correct)", 3.484f, 3.352f, 0.352f, 0.2279f, 0.0552f, 0.0975f, 5.51f, true},
        {"S02 chair_13 (chair(5), wall-born, correct)", 3.212f, 3.060f, 0.351f, 0.0800f, 0.0400f, 0.0324f, 5.51f, true},
        {"S01 chair_4  (mid-room)",                     6.419f, 5.208f, 1.325f, 0.0349f, 0.0283f, 0.0102f, 5.27f, false},
        {"synthetic mid-room chair, 1.5 m",             4.0f,   3.0f,   1.500f, 0.0300f, 0.0200f, 0.0100f, 5.50f, false},
        {"synthetic chair 0.9 m off the wall",          4.0f,   3.0f,   0.900f, 0.0300f, 0.0200f, 0.0100f, 5.50f, false},
    };
    // S12 chair_71 (chair(5), 0.443 m, the camera 1.27 m off the wall at 3.82 m, pose std 0.029/0.021/0.011) stays FREE with the
    // view: its middle is 3.0 σ from the wall render (σ 0.049, the pose σ across only 0.032). It is printed, not asserted; its
    // yaw mode is corrected by (c)'s adoption in the replay (see the report).
    const Case c71{"S12 chair_71 (chair(5), FREE-born, 180 deg)", 1.265f, 3.822f, 0.443f, 0.0293f, 0.0205f, 0.0105f, 5.51f, false};
    std::vector<Case> all(std::begin(cases), std::end(cases));
    all.push_back(c71);
    int base_wall_outliers = 0;
    float mid_room_p = 0.f;
    for (const auto& k : all)
    {
        // the camera: its across coordinate from the log, along so that its range to the cloud is the log's
        const float da = k.cam_across - k.mid;
        const float along = std::sqrt(std::max(0.f, k.cam_range * k.cam_range - da * da));
        const Eigen::Vector3f cam(-along, k.cam_across, kZed);
        // the chair's centre across, solved so its rendered first-view cloud has the logged middle (bisection)
        float lo = 0.5f * d_bar, hi = k.mid + 1.0f;
        std::vector<Eigen::Vector3f> pts;
        for (int it = 0; it < 30; ++it)
        {
            const float x = 0.5f * (lo + hi);
            pts = view_cloud(cam, wall_chair(x), cc);
            const Cold c = cold_start(pts, cc);
            (cloud_mid(pts, wall, c.z_lo, c.z_hi) < k.mid ? lo : hi) = x;
        }
        pts = view_cloud(cam, wall_chair(0.5f * (lo + hi)), cc);
        Cold c = cold_start(pts, cc);
        rc::BirthView view;
        view.have_cam = true; view.cam = cam;
        view.have_pose = true; view.robot_xy = cam.head<2>();
        view.sigma_pose = Eigen::Vector3d(k.sx * k.sx, k.sy * k.sy, k.spsi * k.spsi).asDiagonal();
        const float span = k.span;
        view.room_span = [span](const rc::WallRef&) { return span; };
        const rc::NearestWall nw = [&](const Eigen::Vector2f&) { return wall; };
        rc::CountableBeliefState sb = c.s0, sa = c.s0;
        const bool base = rc::seed_birth(sb, pts, c.z_lo, c.z_hi, cc, true, nw, std::string(k.name) + " BASE").has_value();
        float p_wall = -1.f;
        const bool cb1 = rc::seed_birth(sa, pts, c.z_lo, c.z_hi, cc, true, nw, std::string(k.name) + " CB1", &view, &p_wall).has_value();
        std::printf("  (b) %-46s middle %.3f (centre %.3f): BASE %s, CB1 %s (expected %s), P(wall | view) %.3g\n", k.name,
                    cloud_mid(pts, wall, c.z_lo, c.z_hi), 0.5f * (lo + hi), base ? "WALL" : "FREE", cb1 ? "WALL" : "FREE",
                    &k == &all.back() ? "printed" : k.expect_wall ? "WALL" : "FREE", p_wall);
        if (&k != &all.back()) CHECK(cb1 == k.expect_wall);
        CHECK(p_wall >= 0.f and p_wall <= 1.f and (p_wall > 0.5f) == cb1);
        // a chair the birth finds well off the wall hands the belief P(wall) = 0 (the exponential's underflow, not a cutoff): its
        // yaw-mode prior has weight 0 and the update's adoption cannot turn it (c)
        if (not k.expect_wall and k.mid > 1.2f) { CHECK(p_wall < 1e-30f); mid_room_p = std::max(mid_room_p, p_wall); }
        if (k.expect_wall and std::string(k.name).find("FREE-born") != std::string::npos and base) ++base_wall_outliers;
    }
    std::printf("  (b) BASE placed %d of the 4 asserted FREE-born outliers on the wall\n", base_wall_outliers);

    // ── (c) the mode prior and its adoption ─────────────────────────────────────────────────────────────────────────────────
    {
        rc::CountableBeliefParams p;
        p.placement_p_against_wall = 0.5f;
        p.wall_reach_m = cc.ai2_wall_reach_m;
        p.door_clearance_gain = cc.ai2_door_clearance_gain;
        const float gap0 = cc.wall_standoff_m;
        const auto at = [&](float back_gap, float yaw_off, bool swap)
        {
            rc::CountableBeliefState s{0.f, back_gap + 0.5f * d_bar, H_bar, swap ? d_bar : w_bar, swap ? w_bar : d_bar,
                                       std::remainder(yaw_w + yaw_off, float(2.0 * M_PI))};
            rc::CountableBelief b(s, p);
            b.set_room_interior({0.f, 3.f});
            b.set_frame_wall_for_test(wall);
            return b;
        };
        const auto back_dot = [&](const rc::CountableBelief& b)
        { return Eigen::Vector2f(-std::sin(b.state().yaw), std::cos(b.state().yaw)).dot(-wall.n); };
        {   // held 180° wrong at the wall
            auto b = at(gap0, float(M_PI), false);
            const auto fp = b.front_posterior();
            std::printf("  (c) held 180 deg wrong at the wall: posterior [%.3f %.3f %.3f %.3f], back.(-n) %.2f", fp[0], fp[1], fp[2], fp[3], back_dot(b));
            CHECK(fp[2] > fp[0] and fp[2] > fp[1] and fp[2] > fp[3]);
            CHECK(b.resolve_front_geometric());
            std::printf(" -> after adoption %.2f\n", back_dot(b));
            CHECK(back_dot(b) > 0.99f);
        }
        {   // held a quarter turn off at the wall (w ↔ h as the mode relabels): the wall-side face carries the weight (CB1)
            auto b = at(gap0, 0.5f * float(M_PI), true);
            const auto fp = b.front_posterior();
            std::printf("  (c) held 90 deg off at the wall: posterior [%.3f %.3f %.3f %.3f], back.(-n) %.2f", fp[0], fp[1], fp[2], fp[3], back_dot(b));
            CHECK(b.resolve_front_geometric());
            std::printf(" -> after adoption %.2f\n", back_dot(b));
            CHECK(back_dot(b) > 0.99f);
        }
        {   // a chair the birth found free, 1.27 m off the wall and held so its back faces the room: weight 0, no adoption
            rc::CountableBeliefParams q = p;
            q.placement_p_against_wall = mid_room_p;
            rc::CountableBeliefState s{0.f, 1.5f, H_bar, w_bar, d_bar, std::remainder(yaw_w + float(M_PI), float(2.0 * M_PI))};
            rc::CountableBelief b(s, q);
            b.set_room_interior({0.f, 3.f});
            b.set_frame_wall_for_test(wall);
            const auto lp = b.door_clearance_logprior();
            std::printf("  (c) birth-free chair 1.27 m off (P(wall) %.3g): lp [%.3g %.3g %.3g %.3g]\n", mid_room_p, lp[0], lp[1], lp[2], lp[3]);
            CHECK(lp[0] == 0.f and lp[1] == 0.f and lp[2] == 0.f and lp[3] == 0.f);
            CHECK(not b.resolve_front_geometric());
        }
        {   // held correctly: the adoption is a no-op
            auto b = at(gap0, 0.f, false);
            CHECK(not b.resolve_front_geometric());
        }
        {   // off the wall: the posterior → uniform as the flush weight → 0 (monotone spread), held 180° wrong
            std::printf("  (c) the posterior's spread (max − min) as the chair moves off the wall:");
            float prev = 2.f;
            bool mono = true;
            float last = 1.f;
            for (const float g : {0.05f, 0.15f, 0.3f, 0.5f, 0.8f, 1.2f, 2.0f})
            {
                const auto fp = at(g, float(M_PI), false).front_posterior();
                const float spread = *std::max_element(fp.begin(), fp.end()) - *std::min_element(fp.begin(), fp.end());
                std::printf(" %.2f m %.4f;", g, spread);
                mono = mono and spread <= prev;
                prev = spread; last = spread;
            }
            std::printf("\n");
            CHECK(mono);
            CHECK(last < 1e-6f);
        }
        {   // BASE equivalence for a body held correctly: no placement declared (−1) and π = 1/2 both give gain·f·cos
            for (const float pi : {-1.0f, 0.5f})
            {
                rc::CountableBeliefParams q = p;
                q.placement_p_against_wall = pi;
                const float g = 0.12f;
                rc::CountableBeliefState s{0.f, g + 0.5f * d_bar, H_bar, w_bar, d_bar, yaw_w};
                rc::CountableBelief b(s, q);
                b.set_room_interior({0.f, 3.f});
                b.set_frame_wall_for_test(wall);
                const auto lp = b.door_clearance_logprior();
                const float rel = g / q.wall_reach_m, f = std::exp(-rel * rel);
                float worst = 0.f;
                for (int k = 0; k < 4; ++k)
                {
                    const float psi = std::remainder(s.yaw + float(k) * float(M_PI_2), float(2.0 * M_PI));
                    const float base = q.door_clearance_gain * f * Eigen::Vector2f(std::sin(psi), -std::cos(psi)).dot(wall.n);
                    worst = std::max(worst, std::abs(lp[std::size_t(k)] - base));
                }
                std::printf("  (c) held correctly, pi %.1f: |lp - BASE| max %.2e\n", pi, worst);
                CHECK(worst <= 1e-6f);
            }
            // π = 1: weight 1 at any distance; π = 0: weight 0
            for (const float pi : {1.0f, 0.0f})
            {
                rc::CountableBeliefParams q = p;
                q.placement_p_against_wall = pi;
                rc::CountableBeliefState s{0.f, 1.5f, H_bar, w_bar, d_bar, yaw_w};
                rc::CountableBelief b(s, q);
                b.set_room_interior({0.f, 3.f});
                b.set_frame_wall_for_test(wall);
                const auto lp = b.door_clearance_logprior();
                std::printf("  (c) pi %.0f, 1.27 m off: lp[0] %.3f\n", pi, lp[0]);
                CHECK(std::abs(lp[0] - (pi > 0.f ? q.door_clearance_gain : 0.f)) <= 1e-5f);
            }
        }
    }
    std::printf(bad ? "birth_view_placement_test: %d FAILED\n" : "birth_view_placement_test: OK\n", bad);
    return bad != 0;
}
