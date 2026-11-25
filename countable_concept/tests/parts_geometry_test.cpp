// parts_geometry_test — K2a: ONE parts table (common/ai_belief/parts_geometry.h) gives the camera SDF, the beam parts and the
// existence carve; the toilet's solid is a bowl and a tank.
//
// What is checked, every bound derived before running (exact geometry or rounding, nothing fitted):
//   1. the toilet manifest declares solid = "parts" with two rows, bowl and tank (class knowledge, cited in the manifest);
//   2. the toilet's camera SDF is the STEPPED solid: on θ = (0, 0, H 1.0, w 0.58, h 0.74, yaw 0) the bowl front (y = −0.37) is a
//      surface at z 0.30 (SDF 0); a point at that front at z 0.80 is OUTSIDE, 0.239 above the bowl top 0.561 (K2a-L: the proportion bowl_top at its midpoint, no q set) (the tank front is
//      0.5254 behind it, the nearer face wins); the tank front y = 0.37 − 0.29·0.74 = 0.1554 at z 0.80 is a surface; a point
//      beside the bowl under the tank (x 0.27, y 0.30, z 0.30) is 0.27 − 0.76·0.29 = 0.0496 outside. Tolerance 1e-5 (float
//      rounding of terms ≤ 1 m). One box would hold the two outside points INSIDE (negative SDF);
//   3. the beam parts are the same solid: for 2000 random points min over the parts' box SDFs (the test's own, double) equals
//      sdf_compound to 1e-5 (float rounding), the convex flag is off (2 parts);
//   4. each part's analytic J (rows ∂(cx, cy, cz, yaw, hx, hy, hz)/∂θ) equals a central finite difference of beam_parts itself,
//      step e = 1e-4 taken on the FLOAT state (the divisor is the step the floats actually hold, so it is exact). Tolerance 2e-7:
//      the parts are computed in double from the float θ, so a component linear in (cx, cy, H, w, h) differences exactly (double
//      rounding ~1e-16 / 2e-4 ≈ 1e-12); in yaw (c, the lever terms) the error is the truncation e²/6·|third derivative| ≤
//      (1e-4)²/6·0.6 ≈ 1e-9 plus the secant's midpoint offset from θ (each stepped float rounds by ≤ ½ ulp(2.05) = 1.2e-7, so
//      the midpoint moves ≤ 1.2e-7) times |second derivative| ≤ |lever| ≤ 0.6, ≈ 7e-8: total ≤ 1e-7, tolerance 2e-7. Also on a
//      HANGING span (∂z0/∂H = 1), which no manifest declares yet;
//   5. the carve reads the same rows: carve_parts' boxes have the beam parts' local centres and sizes (tol 1e-6);
//   6. a solid = "box" class (the nightstand: 5 carve rows) keeps the pre-K2a solid BIT-identically: the SDF is the whole box
//      (the old expression, copied here, compared with ==), one beam part equal (==) to the old envelope part, and the bed's
//      headboard panel likewise. §6 CANNOT fail before the change by construction (it compares against the pre-K2a code itself);
//      it is a regression guard. The "fails before" evidence is checks 1–5 run against the BASE 93f3b35b manifests (11 FAIL).
// Run: ./parts_geometry_test MANIFEST_DIR   (pass a copy of the BASE 93f3b35b manifests to see checks 1–5 fail there)
#include <clocale>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>
#include "countable_belief.h"
#include "countable_shape_parts.h"
#include "concept_manifest/shape_parts.h"
#include "concept_manifest/back_panel.h"

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)
static bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

using rc::CountableBelief;
using rc::CountableBeliefParams;
using rc::CountableBeliefState;
namespace bm = rc::ai::beam;

static double part_box_sdf(const bm::BeamPart& p, const Eigen::Vector3d& x)
{
    const double c = std::cos(p.yaw), s = std::sin(p.yaw), dx = x.x() - p.c.x(), dy = x.y() - p.c.y();
    const Eigen::Vector3d q(c * dx + s * dy, -s * dx + c * dy, x.z() - p.c.z());
    const Eigen::Vector3d a = q.cwiseAbs() - p.half;
    return a.cwiseMax(0.0).norm() + std::min(a.maxCoeff(), 0.0);
}
static Eigen::Matrix<double, 7, 1> pack(const bm::BeamPart& p)
{
    Eigen::Matrix<double, 7, 1> v;
    v << p.c.x(), p.c.y(), p.c.z(), p.yaw, p.half.x(), p.half.y(), p.half.z();
    return v;
}
static void check_jacobian(const CountableBelief& b, const CountableBeliefState& s)
{
    const auto P0 = b.beam_parts_for_test(s);
    const double e = 1e-4;
    for (int j = 0; j < 6; ++j)
    {
        Eigen::Matrix<double, 6, 1> vp = s.vec().cast<double>(), vm = vp;
        vp(j) += e; vm(j) -= e;
        // the state is float: build it from double and step on the float grid (the FD step is what the float holds)
        const auto sp = CountableBeliefState::from_vec(vp.cast<float>()), sm = CountableBeliefState::from_vec(vm.cast<float>());
        const double step = double(sp.vec()(j)) - double(sm.vec()(j));
        const auto Pp = b.beam_parts_for_test(sp), Pm = b.beam_parts_for_test(sm);
        for (std::size_t k = 0; k < P0.size(); ++k)
        {
            const Eigen::Matrix<double, 7, 1> fd = (pack(Pp[k]) - pack(Pm[k])) / step;
            for (int r = 0; r < 7; ++r)
                if (not near(fd(r), P0[k].J(r, j), 2e-7))   // derived in the header, check 4
                { std::printf("FAIL J part %zu row %d col %d: fd %.9f analytic %.9f\n", k, r, j, fd(r), P0[k].J(r, j)); ++bad; }
        }
    }
}

int main(int argc, char** argv)
{
    std::setlocale(LC_ALL, "");
    if (argc < 2) { std::printf("usage: %s MANIFEST_DIR\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    const auto mf = [&](const char* c) { return dir + "/" + c + ".concept.toml"; };

    // 1. the toilet's table
    const auto tparts = rc::manifest::load_shape_parts(mf("toilet"));
    const bool tsolid = rc::manifest::load_solid_is_parts(mf("toilet"));
    CHECK(tsolid);
    CHECK(tparts.size() == 2);
    if (tparts.size() == 2) { CHECK(tparts[0].name == "bowl" and tparts[1].name == "tank"); }

    CountableBeliefParams tp;
    tp.shape_parts = tparts;
    tp.solid_parts = tsolid;
    const CountableBeliefState ts{0.0f, 0.0f, 1.0f, 0.58f, 0.74f, 0.0f};
    const CountableBelief tb(ts, tp);
    // 2. the stepped SDF
    {
        CHECK(near(tb.sdf_compound({0.0f, -0.37f, 0.30f}, ts), 0.0, 1e-5));      // bowl front
        CHECK(near(tb.sdf_compound({0.0f, -0.37f, 0.80f}, ts), 0.239, 1e-5));     // in front of the tank, above the bowl
        CHECK(near(tb.sdf_compound({0.0f, 0.1554f, 0.80f}, ts), 0.0, 1e-5));     // tank front
        CHECK(near(tb.sdf_compound({0.27f, 0.30f, 0.30f}, ts), 0.0496, 1e-5));   // beside the bowl, under the tank
        // 7 cm behind the bowl front at z 0.80: inside the envelope (sdf_box −0.07), 0.239 above the bowl in the parts solid
        CHECK(near(tb.sdf_box({0.0f, -0.30f, 0.80f}, ts), -0.07, 1e-5) and near(tb.sdf_compound({0.0f, -0.30f, 0.80f}, ts), 0.239, 1e-5));
    }
    // 3. the beam parts are the SDF's solid
    {
        const CountableBeliefState s2{1.2f, -0.4f, 0.93f, 0.55f, 0.70f, 0.7f};
        const auto P = tb.beam_parts_for_test(s2);
        CHECK(P.size() == 2 and tb.n_beam_parts() == 2);
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> U(-0.8f, 0.8f), Z(-0.1f, 1.2f);
        double worst = 0.0;
        for (int i = 0; i < 2000; ++i)
        {
            const Eigen::Vector3f x(s2.cx + U(rng), s2.cy + U(rng), Z(rng));
            double m = 1e9;
            for (const auto& q : P) m = std::min(m, part_box_sdf(q, x.cast<double>()));
            worst = std::max(worst, std::fabs(m - double(tb.sdf_compound(x, s2))));
        }
        CHECK(worst <= 1e-5);
        if (worst > 1e-5) std::printf("  beam-vs-SDF worst %.3g\n", worst);
    }
    // 4. the analytic J, floor-anchored and hanging
    {
        check_jacobian(tb, CountableBeliefState{1.2f, -0.4f, 0.93f, 0.55f, 0.70f, 0.7f});
        CountableBeliefParams hp = tp;
        hp.vertical_extent_m = 0.5f;
        const CountableBelief hb(ts, hp);
        check_jacobian(hb, CountableBeliefState{0.3f, 2.0f, 2.05f, 0.90f, 0.50f, -1.1f});
    }
    // 5. the carve reads the same rows
    {
        const auto C = rc::countable::carve_parts_for(ts.w, ts.h, ts.H, tparts, std::nullopt);
        const auto P = tb.beam_parts_for_test(ts);   // yaw 0, centre 0: world = local
        CHECK(C.size() == P.size());
        for (std::size_t k = 0; k < std::min(C.size(), P.size()); ++k)
        {
            CHECK(near(C[k].cx, P[k].c.x(), 1e-6) and near(C[k].cy, P[k].c.y(), 1e-6));
            CHECK(near(C[k].sx, 2.0 * P[k].half.x(), 1e-6) and near(C[k].sy, 2.0 * P[k].half.y(), 1e-6));
            CHECK(near(C[k].z0, P[k].c.z() - P[k].half.z(), 1e-6) and near(C[k].z1, P[k].c.z() + P[k].half.z(), 1e-6));
        }
    }
    // 6. a solid = "box" class is bit-identical (the pre-K2a expressions, verbatim)
    {
        CountableBeliefParams np;
        np.shape_parts = rc::manifest::load_shape_parts(mf("nightstand"));
        np.solid_parts = rc::manifest::load_solid_is_parts(mf("nightstand"));
        CHECK(not np.solid_parts and np.shape_parts.size() == 5);
        const auto bp = rc::manifest::load_back_panel(mf("bed"));
        CHECK(bp.has_value());
        CountableBeliefParams bedp;
        bedp.shape_parts = rc::manifest::load_shape_parts(mf("bed"));
        if (bp) { bedp.headboard_h_m = bp->height_m; bedp.headboard_t_m = bp->thickness_m; }
        const auto old_box = [](float dx, float dy, float dz)
        {
            const float ox = std::max(dx, 0.0f), oy = std::max(dy, 0.0f), oz = std::max(dz, 0.0f);
            return std::sqrt(ox * ox + oy * oy + oz * oz) + std::min(std::max(dx, std::max(dy, dz)), 0.0f);
        };
        std::mt19937 rng(11);
        std::uniform_real_distribution<float> U(-1.5f, 1.5f), Z(-0.2f, 1.6f);
        for (const CountableBeliefParams* pp : {&np, &bedp})
        {
            const CountableBelief b(ts, *pp);
            const CountableBeliefState s{0.4f, -0.2f, 0.62f, 0.50f, 0.60f, 0.37f};
            CHECK(b.n_beam_parts() == (pp->headboard_h_m > 0.0f ? 2 : 1));
            int diff = 0;
            for (int i = 0; i < 3000; ++i)
            {
                const Eigen::Vector3f p(s.cx + U(rng), s.cy + U(rng), Z(rng));
                const float c = std::cos(-s.yaw), sn = std::sin(-s.yaw);
                const float px = p.x() - s.cx, py = p.y() - s.cy;
                const float lx = px * c - py * sn, ly = px * sn + py * c;
                const float d_box = old_box(std::abs(lx) - 0.5f * s.w, std::abs(ly) - 0.5f * s.h, std::abs(p.z() - 0.5f * s.H) - 0.5f * s.H);
                float d = d_box;
                if (pp->headboard_h_m > 0.0f)
                {
                    const float t = std::min(pp->headboard_t_m, s.h), yc = 0.5f * s.h - 0.5f * t, hz = 0.5f * pp->headboard_h_m;
                    d = std::min(d_box, old_box(std::abs(lx) - 0.5f * s.w, std::abs(ly - yc) - 0.5f * t, std::abs(p.z() - hz) - hz));
                }
                diff += b.sdf_compound(p, s) != d;
            }
            CHECK(diff == 0);
            const auto P = b.beam_parts_for_test(s);
            CHECK(not P.empty());
            if (not P.empty())
            {
                const double cx = s.cx, cy = s.cy, H = s.H, w = s.w, h = s.h;
                CHECK(P[0].c == Eigen::Vector3d(cx, cy, 0.5 * H) and P[0].half == Eigen::Vector3d(0.5 * w, 0.5 * h, 0.5 * H));
                CHECK(P[0].yaw == double(s.yaw) and P[0].J(2, 2) == 0.5 and P[0].J(6, 2) == 0.5 and P[0].J(4, 3) == 0.5);
            }
        }
    }
    std::printf(bad ? "FAILED (%d)\n" : "parts_geometry_test OK\n", bad);
    return bad ? 1 : 0;
}
