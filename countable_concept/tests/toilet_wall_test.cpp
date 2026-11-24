// toilet_wall_test — the wall enters ONCE (plan rev 2 F1.4; engine1 diagnosis item 2). A toilet stands 17 cm off its wall (the
// Webots world's toilet: back 0.17 m in front of the wall), born with its back ON the wall (birth on priors), and the robot is
// parked: the same front-face view, 128 updates. engine1's toilet ended at σ 1.4 cm, pinned by a hinge re-entered every update.
//
// PREDICTION, stated before the run (a linear-Gaussian count of what is entered once; scratch: the same 4 facts in numpy):
// unknowns (c, h) = the centre along the wall normal and the depth. Facts:
//   1. the camera's front face, c − h/2, through the persistent camera latents (ρ = 1 parked, σ_c = 0.0289 on c and on h):
//      saturates at var σ_c² + σ_c²/4 = 1.044e-3 whatever the number of frames;
//   2. the class depth prior, h ~ N(0.70, 0.10²) (once, the class prior at the class mean);
//   3. the wall, ONCE: back − δ ~ N(standoff 0.03, 0.02²), δ ~ N(0, σ_wall² = 0.02²) ⇒ var 8e-4 on c + h/2 (flush weight 1:
//      one face seen ⇒ the class π_flush, set to 1 here);
//   4. the class position prior on c, σ_pos 0.30 (once).
//   ⇒ σ_n = 0.0214 m, mean pulled toward the wall: c error +0.072 m (the honest stand-off does not cover 17 cm), h 0.818.
// Asserted: σ_n after 128 parked updates within 15 % of 0.0214 m (the GN's last-step term and e add a little), and σ_n(128) ≥
// 0.99·σ_n(1): it does NOT shrink with the number of updates. Why 0.99 and not 1: the camera latents saturate on the FIRST
// frame already (120 points at σ_base² = 9e-4 give the front 7.5e-6, against its saturated latent variance 1.04e-3: < 1 %), so
// later frames can remove at most that share, < 0.4 % of σ — the bound leaves 2.5× room for it, and the hinge re-entered every
// update (what F1.4 removes) takes 16 % off by update 128 in the second case below. The error is printed, not asserted: it is the class/world
// mismatch test (a) will report. The depth_unobs footprint share is off here (it would pin h to the class mean through b_cam,
// another fact) so the count above is the whole model.
// Second case, the engine1 pin itself: the staged wall is 5 mm off so the data put the back 5 mm PAST it (the no-cross term
// active). Before F1.4 the fixed-precision hinge re-entered every update and σ_n fell with the update count (measured on the code
// before F1.4: 0.0184 → 0.0154 m); now there is no no-cross term at all (the stand-off prior, entered once, says "at or off
// the wall"; coordinator ruling 11-24). Asserted: σ_n(128) ≥ 0.99·σ_n(1) (no prediction of its value: it
// depends on where the first GN leaves the gap).
// Run: ./toilet_wall_test
#include <cmath>
#include <cstdio>
#include "countable_belief.h"

using namespace rc;
static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

static constexpr float kW = 0.45f, kD = 0.70f, kH = 0.75f;      // the toilet's class footprint and height (= the truth here)
static constexpr float kOff = 0.17f;                             // the truth's back stands this far off the wall (y = 0)
static constexpr float kTruthCy = -(kOff + 0.5f * kD);           // the room is at y < 0; the wall normal into it is (0, −1)
static constexpr float kSigmaC = 0.0289f;                        // the camera latents' marginal (F1.6)

static CountableBeliefParams params()
{
    CountableBeliefParams p;
    p.prior_footprint_m = kW; p.prior_depth_m = kD; p.prior_footprint_std = 0.10f;
    p.prior_height_m = kH; p.prior_height_std = 0.15f;
    p.plaus_height_min = 0.40f; p.plaus_height_soft = 0.08f;
    p.wall_flush_prior = 1.0f;                                   // one face seen ⇒ the flush weight is π_flush: 1
    p.depth_unobs_precision = 0.0f;                              // see the header
    p.room_axis_std = 3.0f * 0.017453293f;
    p.volatility_infer = false;                                  // Q = the class q₀, 0 here (no [prior.permanence] set): the count above has no process noise
    p.persistent = true;
    // (generic lambdas so the same file also builds against the code before F1.4, for the before/after numbers)
    [](auto& q) { if constexpr (requires { q.wall_standoff_m; }) { q.wall_standoff_m = 0.03f; q.wall_standoff_std_m = 0.02f; } }(p);
    return p;
}

static CountableFrame front_frame(float truth_cy = kTruthCy)
{
    CountableFrame f;
    const float yf = truth_cy - 0.5f * kD;                       // the front face
    for (int i = 0; i < 12; ++i) for (int j = 0; j < 10; ++j)
        f.points.emplace_back(-0.5f * kW + kW * (i + 0.5f) / 12.0f, yf, kH * (j + 0.5f) / 10.0f);
    f.R.assign(f.points.size(), params().sigma_base_m * params().sigma_base_m);
    f.z_top_obs = kH;
    f.wall.ok = true; f.wall.p = Eigen::Vector2f(0, 0); f.wall.n = Eigen::Vector2f(0, -1);
    f.wall.sigma_m = 0.02f; f.wall.length_m = 3.0f;
    [](auto& w) { if constexpr (requires { w.edge; }) { w.room = 1; w.edge = 0; } }(f.wall);
    f.room_axis_ok = true; f.room_axis = 0.0f;
    const float c2 = kSigmaC * kSigmaC;
    f.sig2_cam = {c2, c2, c2, c2, c2, 0.03f * 0.03f};
    f.sig2_L = f.sig2_cam;
    f.sigma_pose = Eigen::Vector3d(1e-8, 1e-8, 0.0).asDiagonal();
    f.rho_cam.fill(1.0f); f.rho_L.fill(1.0f); f.rho_chain = 1.0f;
    f.robot_xy = Eigen::Vector2f(0.0f, -2.5f);
    return f;
}

struct Run { double s1, s10, s128, err; };
static Run parked(float truth_cy, const char* name)
{
    CountableBelief b(CountableBeliefState{0.0f, -0.5f * kD, kH, kW, kD, 0.0f}, params());   // born with its back ON the wall
    b.set_room_interior(Eigen::Vector2f(0.0f, -3.0f));
    const CountableFrame f = front_frame(truth_cy);
    Run r{};
    for (int k = 1; k <= 128; ++k)
    {
        b.update(f);
        const double sn = std::sqrt(double(b.covariance()(1, 1)));
        if (k == 1 or k == 10 or k == 32 or k == 128)
            std::printf("  %s update %3d: c %.4f (truth %.4f, error %+.4f m) h %.4f  sigma_n %.4f m\n", name, k, b.state().cy,
                        truth_cy, b.state().cy - truth_cy, b.state().h, sn);
        if (k == 1) r.s1 = sn;
        if (k == 10) r.s10 = sn;
        if (k == 128) { r.s128 = sn; r.err = b.state().cy - truth_cy; }
    }
    return r;
}

int main()
{
    std::printf("toilet_wall_test (F1.4)\n");
    constexpr double kPredSigma = 0.0214, kPredErr = 0.072;
    const Run a = parked(kTruthCy, "17 cm off");
    std::printf("  predicted sigma_n %.4f m, error %+.3f m ; got sigma_n %.4f m (ratio to update 10: %.3f), error %+.4f m\n",
                kPredSigma, kPredErr, a.s128, a.s128 / a.s10, a.err);
    CHECK(std::abs(a.s128 - kPredSigma) <= 0.15 * kPredSigma);
    CHECK(a.s128 >= 0.99 * a.s1);
    const Run p = parked(0.005f - 0.5f * kD, "5 mm past");
    std::printf("  5 mm past the staged wall: sigma_n %.4f (update 1) %.4f (10) %.4f (128), ratio 128/1 %.3f\n", p.s1, p.s10,
                p.s128, p.s128 / p.s1);
    CHECK(p.s128 >= 0.99 * p.s1);
    std::printf(bad ? "toilet_wall_test: %d FAILED\n" : "toilet_wall_test: OK\n", bad);
    return bad != 0;
}
