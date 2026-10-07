// Offline check: rc::mount::Accum::marginal_information() must reproduce Accum::solve() exactly, in
// physical units, with the prior and the data kept apart (plan 2026-10-05 Task 2).
//
//   g++ -std=c++23 -O2 -I/usr/include/eigen3 tools/mount_info_selftest.cpp -o /tmp/mount_info_selftest && /tmp/mount_info_selftest
#include "../src/mount_lidar_pair.h"
#include <cstdio>
#include <random>

using rc::mount::Accum;
static int failures = 0;
static void check(bool ok, const char *what) { std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what); failures += not ok; }

namespace
{
    std::mt19937 rng(20261005);
    double gauss(double s) { return std::normal_distribution<double>(0.0, s)(rng); }

    // Synthetic camera with a known mount error. PairObs::J is d(uv)/d(nuisance) in PRIOR-SIGMA units
    // (make_pair_from scales each column by its sigma). A mount error e makes the PREDICTION wrong by
    // J e while the image is right, so r = -J e and Accum::solve() returns p = -x = e (mount_lidar_pair.h
    // Solution note). So a planted physical error p_phys gives r = -J (p_phys / unit), plus a
    // per-vertex constant offset N(0, 5.3 px) (the nuisance the solve marginalises) and N(0, 1 px).
    // The dt column stays zero, as in the live make_pair_from().
    void fill_synthetic(Accum &acc, bool attributed, const Eigen::Vector4d &unit, const Eigen::Vector4d &planted)
    {
        const Eigen::Vector4d p_sig = planted.cwiseQuotient(unit);
        std::vector<Eigen::Vector2d> offset(12);
        for (auto &o : offset) o = Eigen::Vector2d(gauss(5.3), gauss(5.3));
        std::uniform_real_distribution<double> U(-1.0, 1.0);
        for (int i = 0; i < 2000; ++i)
        {
            const int v = i % 12;
            rc::mount::PairObs o;
            o.vertex = attributed ? v : -1;
            o.J.setZero();
            // pitch / height / yaw columns with range-like diversity so the three separate.
            const double range = 1.0 + 7.0 * (0.5 + 0.5 * U(rng));
            o.J(0, 0) = static_cast<float>(1.0 * U(rng));          o.J(1, 0) = static_cast<float>(3.0 + 0.5 * U(rng));
            o.J(0, 1) = 0.f;                                          o.J(1, 1) = static_cast<float>(20.0 / range);
            o.J(0, 2) = static_cast<float>(4.0 + 0.3 * U(rng));     o.J(1, 2) = static_cast<float>(0.2 * U(rng));
            const Eigen::Matrix<double, 2, 4> J = o.J.leftCols<4>().cast<double>();
            const Eigen::Vector2d r = -J * p_sig + offset[static_cast<std::size_t>(v)]
                                    + Eigen::Vector2d(gauss(1.0), gauss(1.0));
            o.r = r.cast<float>();
            o.cov = Eigen::Matrix2f::Identity();
            o.assoc_prob = 1.f;
            o.ok = true;
            acc.add(o);
        }
    }
}

int main()
{
    const Eigen::Vector4d unit(0.0035, 0.010, 0.0035, 1.0);   // etc/config.toml mount*Sigma
    const Eigen::Vector4d planted(0.002, -0.006, 0.003, 0.0);  // pitch rad, height m, yaw rad, dt

    // A: attributed evidence (every row carries a vertex id) -> marginalisable.
    Accum acc; acc.offset_sigma_px = 5.3;
    fill_synthetic(acc, /*attributed=*/true, unit, planted);
    acc.applied = Eigen::Vector4d(0.3, -0.2, 0.1, 0.0);        // a non-zero prior anchor, as with mountApply on
    const auto s  = acc.solve();
    const auto mi = acc.marginal_information(unit);
    check(s.ok and mi.ok and s.marginalised and mi.marginalised, "A: both solve, marginalised");
    const Eigen::Vector4d p = (mi.H_data + mi.H_prior).ldlt().solve(mi.b_data + mi.b_prior);
    const Eigen::Vector4d p_phys = s.p.cwiseProduct(unit);
    std::printf("  A: p (info) %+.5f %+.5f %+.5f | p (solve) %+.5f %+.5f %+.5f | planted %+.5f %+.5f %+.5f\n",
                p[0], p[1], p[2], p_phys[0], p_phys[1], p_phys[2], planted[0], planted[1], planted[2]);
    check((p - p_phys).cwiseAbs().maxCoeff() < 1e-9 * (1.0 + p_phys.cwiseAbs().maxCoeff()),
          "A: (H_data + H_prior)^-1 (b_data + b_prior) == Solution::p in physical units");
    // solve() multiplies its sigma by the Birge ratio sqrt(max(1, chi2/dof)); marginal_information()
    // reports that factor rather than folding it into H (see MarginalInfo::sigma_inflation).
    const Eigen::Vector4d sig = (mi.H_data + mi.H_prior).inverse().diagonal().cwiseSqrt() * mi.sigma_inflation;
    check((sig - s.sigma.cwiseProduct(unit)).cwiseAbs().maxCoeff() < 1e-9, "A: sigmas agree (with solve()'s Birge inflation)");
    check(mi.H_prior.isApprox(Eigen::Matrix4d(unit.cwiseInverse().cwiseAbs2().asDiagonal())), "A: prior information = 1/unit^2");

    // A2: RE-ANCHORING TO A ROTATED LIDAR (plan Task 5 step 3). When the LiDAR points are rotated by
    // dpsi, a new pair measures p_yaw + dpsi (y = psi_c - (psi_h - psi_app)), so the OLD evidence is
    // re-referenced with Accum::rebase(+dpsi / yaw_sigma) -- DATA only, the camera did not move, so the
    // prior anchor `applied` stays. The data-only solution must shift by exactly +dpsi; the prior not at all.
    // ⚠ DEVIATION: the plan says apply_correction(-dpsi/sigma), which shifts p by -dpsi (the wrong way
    // relative to what new pairs measure) and moves the camera's prior anchor (which mountApply then
    // pushes into the camera projection).
    {
        const double dpsi = 0.004;
        Accum moved = acc;
        moved.rebase(Eigen::Vector4d(0.0, 0.0, dpsi / unit[2], 0.0));
        const auto m0 = acc.marginal_information(unit), m1 = moved.marginal_information(unit);
        const Eigen::Vector4d d0 = m0.H_data.ldlt().solve(m0.b_data), d1 = m1.H_data.ldlt().solve(m1.b_data);
        std::printf("  A2: data-only yaw %+.6f -> %+.6f (shift %+.6f, expect %+.6f)\n", d0[2], d1[2], d1[2] - d0[2], dpsi);
        check(std::abs((d1 - d0)[2] - dpsi) < 1e-9 and std::abs((d1 - d0)[0]) < 1e-9 and std::abs((d1 - d0)[1]) < 1e-9,
              "A2: rebase(+dpsi/sigma) shifts the DATA solution by exactly +dpsi on yaw only");
        check((m1.b_prior - m0.b_prior).cwiseAbs().maxCoeff() == 0.0, "A2: the prior anchor does not move");
    }

    // B: unattributed evidence with the nuisance ON. ⚠ solve() does NOT refuse here -- it silently
    // falls back to the unmarginalised model (marginalised=false), which mountApply then refuses to
    // write. marginal_information() must refuse, so the joint never takes it as marginalised.
    Accum bad; bad.offset_sigma_px = 5.3;
    fill_synthetic(bad, /*attributed=*/false, unit, planted);
    check(not bad.solve().marginalised and not bad.marginal_information(unit).ok,
          "B: unmarginalisable evidence (nuisance requested) is refused");

    // C: too few pairs -> refuses exactly when solve() refuses.
    Accum few; few.offset_sigma_px = 5.3;
    check(not few.solve().ok and not few.marginal_information(unit).ok, "C: refuses exactly when solve() refuses (n < min_n)");

    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
