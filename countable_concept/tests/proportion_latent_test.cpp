// proportion_latent_test — K2a-L: the class shape PROPORTION as a ρ = 1 latent of the engine (owner-approved model-level fix).
//
// What is checked, every bound derived before running:
//   1. the toilet manifest declares bowl_top ∈ [0.447, 0.675] (cited class knowledge) and both rows name it (bowl z1, tank z0);
//   2. the latent's prior: born on the first measured update at mean 0 (q = the midpoint) and variance Δ²/12 exactly (Δ = hi − lo,
//      in double from the manifest's floats: the engine computes the same expression, so ==), and a class declaring none runs
//      on the engine without the slot (K = 3 + 6N + W, joint P (N + K)², the pre-K2a-L layout);
//   3. the beam parts' Jq column against a central difference of parts_geometry's beam_parts in q, at q = 0.5 ± 2⁻⁸ (both exact
//      in float, so the parts are computed in double from exact inputs and the dependence is linear: tolerance 1e-12, rounding);
//      and the camera's ∂sdf/∂q on the bowl's top face (sdf = z − q·H there, so ∂sdf/∂q = −H) at the model's FD step 1e-3, whose
//      error is the float SDF's rounding: two evaluations of ≤ 1 m at ε_f = 2⁻²⁴ ⇒ 2·6e-8 / 2e-3 = 6e-5, tolerance 1e-4;
//   4. THE CLIFF IS GONE: a synthetic stepped object (bowl + tank, true bowl top 0.46·H), camera points on its visible faces and a
//      ray-cast Helios sweep, fitted with the proportion's prior MEAN swept across [lo, hi] (Δ kept) in 12 equal steps Δμ. Under
//      the Laplace posterior the final mean is affine in the prior mean of q with gain ∂θ̂ᵢ/∂μ = P_{θᵢ q} / σ²_q,prior, and
//      |P_{θᵢ q}| ≤ σ_θᵢ σ_q ≤ σ_θᵢ σ_q,prior (Cauchy–Schwarz; the posterior σ_q ≤ the prior's), so one step moves θᵢ by at most
//      σ_θᵢ · Δμ / σ_q,prior = σ_θᵢ · (Δ/12)/(Δ/√12) = σ_θᵢ/√12 — bound: |Δθ̂ᵢ| ≤ max(σ_θᵢ at both ends)/√12 for i = cy (the
//      depth centre), H, h. A mode switch moves h by the step between the modes (≫ σ). The same sweep with the fraction FIXED at μ
//      (the K2a model) is printed beside it (not checked: it is the model the latent replaces).
//   5. the proportion's information is capped by each channel's per-view common mode on q (fix round 1, I1): a channel measures
//      q + b_q,c, b_q,c born at σ²_q,cm = σ²_size,cm/(H − z0)² (H the seed's at the first predict, z0 = 0) and carried with the
//      view's ρ. Identical frames from ONE view (ρ = 1) at best reveal q + b_q,cam and q + b_q,L exactly, so σ²_q ≥
//      (1/σ²_q,prior + 1/σ²_q,cm,cam + 1/σ²_q,cm,L)⁻¹ — the floor, checked after 40 such frames (relative tolerance 1e-9, rounding);
//      a SECOND view (ρ = 0: both common modes re-drawn) then lowers σ_q below the one-view value.
// Run: ./proportion_latent_test MANIFEST_DIR
#include <clocale>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include "countable_belief.h"
#include "concept_manifest/shape_parts.h"
#include "beam_fixture.h"

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)
static bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

using rc::CountableBelief;
using rc::CountableBeliefParams;
using rc::CountableBeliefState;
using rc::CountableFrame;
namespace bm = rc::ai::beam;
namespace ts = rc::test;

// The synthetic toilet: (0, 0), yaw 0 (back +y, front −y), w 0.58, h 0.74, H 1.0, bowl 0.76 w, tank 0.29 h, bowl top 0.46·H.
static constexpr float kW = 0.58f, kH = 0.74f, kTop = 1.0f, kQtrue = 0.46f;
static std::vector<ts::SceneBox> truth_boxes()
{
    const float yt = 0.5f * kH - 0.5f * 0.29f * kH;
    return {{Eigen::Vector2f(0.0f, 0.0f), 0.0f, 0.76f * kW, kH, kQtrue * kTop, 0.0f},
            {Eigen::Vector2f(0.0f, yt), 0.0f, kW, 0.29f * kH, kTop, kQtrue * kTop}};
}
static CountableBeliefParams toilet_params(const std::string& dir)
{
    CountableBeliefParams p;
    p.prior_footprint_m = 0.58f; p.prior_footprint_std = 0.08f; p.prior_depth_m = 0.73f;
    p.prior_height_m = 0.90f; p.prior_height_std = 0.15f;
    p.plaus_height_min = 0.50f; p.plaus_height_soft = 0.08f;
    p.room_axis_std = 3.0f * 0.017453293f;
    p.persistent = true;
    p.beam = ts::test_beam_config();
    const std::string mf = dir + "/toilet.concept.toml";
    p.shape_parts = rc::manifest::load_shape_parts(mf);
    p.solid_parts = rc::manifest::load_solid_is_parts(mf);
    p.shape_proportions = rc::manifest::load_shape_proportions(mf);
    return p;
}
static const Eigen::Vector3f kHelios(0.0f, -0.37f - 1.5f, 1.1075f);
static void set_latents(CountableFrame& f, const CountableBeliefParams& p)
{
    const float pp = p.common_mode_pos_std * p.common_mode_pos_std, ss = p.common_mode_size_std * p.common_mode_size_std,
                yy = p.common_mode_yaw_std * p.common_mode_yaw_std;
    f.sig2_cam = {pp, pp, ss, ss, ss, yy};
    f.sig2_L   = {pp, pp, ss, ss, ss, yy};
    f.sigma_pose = Eigen::Vector3d(1e-4, 1e-4, 0.0).asDiagonal();
    f.rho_cam.fill(1.0f); f.rho_L.fill(1.0f); f.rho_chain = 1.0f;
    f.robot_xy = kHelios.head<2>();
}
// Camera points on the faces a camera in front sees: the bowl front, the bowl top in front of the tank, the tank front.
static CountableFrame frame(const CountableBeliefParams& p)
{
    CountableFrame f;
    const float bw = 0.76f * kW, zt = kQtrue * kTop, yf = -0.5f * kH, ytf = 0.5f * kH - 0.29f * kH;
    for (int i = 0; i < 10; ++i) for (int j = 0; j < 8; ++j)
        f.points.emplace_back(-0.5f * bw + bw * (i + 0.5f) / 10.0f, yf, zt * (j + 0.5f) / 8.0f);
    for (int i = 0; i < 10; ++i) for (int j = 0; j < 6; ++j)
        f.points.emplace_back(-0.5f * bw + bw * (i + 0.5f) / 10.0f, yf + (ytf - yf) * (j + 0.5f) / 6.0f, zt);
    for (int i = 0; i < 10; ++i) for (int j = 0; j < 8; ++j)
        f.points.emplace_back(-0.5f * kW + kW * (i + 0.5f) / 10.0f, ytf, zt + (kTop - zt) * (j + 0.5f) / 8.0f);
    f.R.assign(f.points.size(), p.sigma_base_m * p.sigma_base_m);
    f.z_top_obs = kTop;
    f.room_axis_ok = true; f.room_axis = 0.0f;
    const auto boxes = truth_boxes();
    const ts::SceneBox sel{Eigen::Vector2f(0.0f, 0.0f), 0.0f, kW, kH, kTop};
    f.beam = ts::helios_beam(kHelios, boxes, {}, sel, 0.10f, ts::hand_labels(boxes));
    set_latents(f, p);
    return f;
}
struct Fit { CountableBeliefState s; Eigen::Matrix<float, 6, 1> sd; double q = 0.0; };
static Fit fit(const CountableBeliefParams& p)
{
    CountableBelief b(CountableBeliefState{0.0f, 0.10f, 0.90f, 0.58f, 0.73f, 0.0f}, p);   // seeded 10 cm toward the back
    b.set_room_interior(Eigen::Vector2f(0.0f, -3.0f));
    const CountableFrame f = frame(p);
    for (int k = 0; k < 6; ++k) b.update(f);
    Fit o;
    o.s = b.state();
    for (int i = 0; i < 6; ++i) o.sd(i) = std::sqrt(std::max(0.0f, b.covariance()(i, i)));
    o.q = b.proportion_value(0);
    return o;
}

int main(int argc, char** argv)
{
    std::setlocale(LC_ALL, "");
    if (argc < 2) { std::printf("usage: %s MANIFEST_DIR\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    // 1. the manifest
    const CountableBeliefParams tp = toilet_params(dir);
    CHECK(tp.shape_proportions.size() == 1);
    if (tp.shape_proportions.size() != 1 or tp.shape_parts.size() != 2) { std::printf("FAILED (%d)\n", bad); return 1; }
    const auto& pr = tp.shape_proportions[0];
    CHECK(pr.name == "bowl_top" and pr.lo == 0.447f and pr.hi == 0.675f and not pr.source.empty());
    CHECK(tp.shape_parts[0].z1_ref == 0 and tp.shape_parts[1].z0_ref == 0 and tp.shape_parts[0].z0_ref == -1);
    const double lo = pr.lo, hi = pr.hi, D = hi - lo;

    // 2. the prior
    {
        CountableBelief b(CountableBeliefState{0.0f, 0.0f, 0.9f, 0.58f, 0.73f, 0.0f}, tp);
        CHECK(b.uses_q());
        CHECK(near(b.proportion_value(0), 0.5 * (lo + hi), 0.0));
        CountableFrame f = frame(tp);
        b.predict_latents_for_test(f);
        const Eigen::MatrixXd P = b.joint_covariance();
        constexpr int j = CountableBelief::N + CountableBelief::EngineQ::kProp;
        CHECK(P.rows() == CountableBelief::N + CountableBelief::EngineQ::K and CountableBelief::EngineQ::K == CountableBelief::K + 3);   // q, and its camera and LiDAR common modes (I1)
        CHECK(P(j, j) == D * D / 12.0);
        CHECK(b.latent_mean()(CountableBelief::EngineQ::kProp) == 0.0);
        CHECK(near(b.proportion_std(0), D / std::sqrt(12.0), 1e-15));
        // a class with no proportion: the engine without the slot
        CountableBeliefParams np = tp;
        np.shape_proportions.clear();
        for (auto& sp : np.shape_parts) sp.z0_ref = sp.z1_ref = -1;
        CountableBelief n(CountableBeliefState{0.0f, 0.0f, 0.9f, 0.58f, 0.73f, 0.0f}, np);
        n.predict_latents_for_test(f);
        CHECK(not n.uses_q() and n.joint_covariance().rows() == CountableBelief::N + CountableBelief::K);
    }
    // 3. the J column
    {
        const rc::countable::VerticalSpan span{};
        const double q0 = 0.5, e = 1.0 / 256.0;
        std::vector<double> qv{q0};
        rc::ai::parts::Table T{std::span<const rc::manifest::ShapePart>(tp.shape_parts), true, -1.0f, 0.10f};
        T.n_q = 1;
        const auto at = [&](double q) { qv[0] = q; T.q = std::span<const double>(qv); return rc::ai::parts::beam_parts(0.3, -0.2, 0.95, 0.56, 0.72, 0.4, T, span); };
        const auto Pp = at(q0 + e), Pm = at(q0 - e), P0 = at(q0);
        double worst = 0.0;
        for (std::size_t k = 0; k < P0.size(); ++k)
        {
            CHECK(P0[k].Jq.cols() == 1);
            Eigen::Matrix<double, 7, 1> a, b;
            a << Pp[k].c, Pp[k].yaw, Pp[k].half;
            b << Pm[k].c, Pm[k].yaw, Pm[k].half;
            worst = std::max(worst, ((a - b) / (2.0 * e) - P0[k].Jq.col(0)).cwiseAbs().maxCoeff());
        }
        CHECK(worst <= 1e-12);
        if (worst > 1e-12) std::printf("  Jq worst %.3g\n", worst);
        // the camera: ∂sdf/∂q on the bowl's top face, in front of the tank (the bowl top at the midpoint q·H)
        CountableBelief b(CountableBeliefState{0.0f, 0.0f, 1.0f, 0.58f, 0.74f, 0.0f}, tp);
        const float zq = float(b.proportion_value(0));
        const float g = b.sdf_dq_for_test(Eigen::Vector3f(0.0f, -0.2f, zq), b.state());
        CHECK(near(g, -1.0, 1e-4));
        if (not near(g, -1.0, 1e-4)) std::printf("  dsdf/dq %.6f (want -1)\n", g);
    }
    // 3b. K2b: the chair's rows, whose seat underside and leg tops are seat_top + a fixed offset (−0.066): the same Jq against
    // a central difference. q + offset is rounded to float in z0_of / z1_of (the toilet's q ± 2⁻⁸ was exact), so the derived
    // tolerance is: each boundary off by ≤ ½ ulp of a value < 1 (2⁻²⁵), times dz ≤ H, two boundaries per row, over 2e:
    // 2 · 2⁻²⁵ · H / (2e) = H · 2⁻²⁵ / e.
    {
        const auto cp = rc::manifest::load_shape_parts(dir + "/chair.concept.toml");
        CHECK(cp.size() == 6);
        const rc::countable::VerticalSpan span{};
        const double q0 = 0.5, e = 1.0 / 256.0, Hc = 0.95;
        std::vector<double> qv{q0};
        rc::ai::parts::Table T{std::span<const rc::manifest::ShapePart>(cp), true, -1.0f, 0.10f};
        T.n_q = 1;
        const auto at = [&](double q) { qv[0] = q; T.q = std::span<const double>(qv); return rc::ai::parts::beam_parts(0.3, -0.2, Hc, 0.50, 0.47, 0.4, T, span); };
        const auto Pp = at(q0 + e), Pm = at(q0 - e), P0 = at(q0);
        double worst = 0.0;
        for (std::size_t k = 0; k < P0.size(); ++k)
        {
            CHECK(P0[k].Jq.cols() == 1);
            Eigen::Matrix<double, 7, 1> a, b;
            a << Pp[k].c, Pp[k].yaw, Pp[k].half;
            b << Pm[k].c, Pm[k].yaw, Pm[k].half;
            worst = std::max(worst, ((a - b) / (2.0 * e) - P0[k].Jq.col(0)).cwiseAbs().maxCoeff());
        }
        const double tol = Hc * std::ldexp(1.0, -25) / e;
        CHECK(worst <= tol);
        std::printf("  chair Jq worst %.3g (tolerance %.3g)\n", worst, tol);
        // the seat slab moves with q (∂zc/∂q = H) and keeps its thickness (∂hz/∂q = 0, 2·hz = 0.066·H up to float rounding)
        if (P0.size() == 6)
        {
            CHECK(near(P0[0].Jq(2, 0), Hc, 1e-12) and P0[0].Jq(6, 0) == 0.0);
            CHECK(near(2.0 * P0[0].half.z(), 0.066 * Hc, 1e-6));
        }
    }
    // 4. the cliff
    {
        const int n = 12;
        const double dmu = D / n, ratio = dmu / (D / std::sqrt(12.0));   // = 1/√12
        std::vector<Fit> L, F;
        for (int k = 0; k <= n; ++k)
        {
            const double mu = lo + k * dmu;
            CountableBeliefParams pl = tp;
            pl.shape_proportions[0].lo = float(mu - 0.5 * D);
            pl.shape_proportions[0].hi = float(mu + 0.5 * D);
            L.push_back(fit(pl));
            CountableBeliefParams pf = tp;                                // the K2a model: the fraction fixed at μ
            pf.shape_proportions.clear();
            pf.shape_parts[0].z1_ref = -1; pf.shape_parts[0].z1_f = float(mu);
            pf.shape_parts[1].z0_ref = -1; pf.shape_parts[1].z0_f = float(mu);
            F.push_back(fit(pf));
        }
        double worst = 0.0;
        for (int k = 0; k <= n; ++k)
            std::printf("  mu %.4f | latent: q %.4f cy %+.4f H %.4f h %.4f (sd cy %.4f H %.4f h %.4f) | fixed: cy %+.4f H %.4f h %.4f\n",
                        lo + k * dmu, L[k].q, L[k].s.cy, L[k].s.H, L[k].s.h, L[k].sd(1), L[k].sd(2), L[k].sd(4), F[k].s.cy, F[k].s.H, F[k].s.h);
        for (int k = 0; k < n; ++k)
            for (const int i : {1, 2, 4})
            {
                const double step = std::abs(double(L[k + 1].s.vec()(i)) - double(L[k].s.vec()(i)));
                const double bound = ratio * double(std::max(L[k].sd(i), L[k + 1].sd(i)));
                worst = std::max(worst, step / bound);
                if (step > bound) { std::printf("FAIL step %d dof %d: |dtheta| %.5f > bound %.5f\n", k, i, step, bound); ++bad; }
            }
        double worst_f = 0.0;
        for (int k = 0; k < n; ++k)
            for (const int i : {1, 2, 4})
                worst_f = std::max(worst_f, std::abs(double(F[k + 1].s.vec()(i)) - double(F[k].s.vec()(i)))
                                                / (ratio * double(std::max(F[k].sd(i), F[k + 1].sd(i)))));
        std::printf("  smoothness: latent worst step / bound %.3f (pass <= 1); fixed fraction (K2a, not checked) %.3f\n", worst, worst_f);
    }
    // 5. the one-view floor and the second view
    {
        CountableBelief b(CountableBeliefState{0.0f, 0.10f, 0.90f, 0.58f, 0.73f, 0.0f}, tp);
        b.set_room_interior(Eigen::Vector2f(0.0f, -3.0f));
        CountableFrame f = frame(tp);
        for (int k = 0; k < 40; ++k) b.update(f);
        const double sq1 = b.proportion_std(0);
        const double cm2 = double(tp.common_mode_size_std) * double(tp.common_mode_size_std) / (0.90 * 0.90);
        const double floor2 = 1.0 / (12.0 / (D * D) + 2.0 / cm2);
        CHECK(sq1 >= std::sqrt(floor2) * (1.0 - 1e-9));
        f.rho_cam.fill(0.0f); f.rho_L.fill(0.0f); f.rho_chain = 0.0f;   // a new view: every common mode re-drawn
        b.update(f);
        f.rho_cam.fill(1.0f); f.rho_L.fill(1.0f); f.rho_chain = 1.0f;
        for (int k = 0; k < 40; ++k) b.update(f);
        const double sq2 = b.proportion_std(0);
        CHECK(sq2 < sq1);
        std::printf("  sigma_q: one view (41 frames) %.5f >= floor %.5f; after a second view %.5f\n", sq1, std::sqrt(floor2), sq2);
    }
    std::printf(bad ? "FAILED (%d)\n" : "proportion_latent_test OK\n", bad);
    return bad ? 1 : 0;
}
