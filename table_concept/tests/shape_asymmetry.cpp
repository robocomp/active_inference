/*
 * shape_asymmetry.cpp  —  why a SQUARE table was classified ROUND (live 2026-08-17, table_3).
 *
 * TableFitter::evaluate_shape scores the two shape hypotheses with DIFFERENT procedures:
 *   round  : a FRESH RoundTableBelief fitted (40 iterations) to the accumulated support-bank cloud.
 *   square : the LIVE tracking belief's current state, which was fitted to the mask stream under priors,
 *            process noise and common-mode inflation — NOT to that cloud.
 * So the round hypothesis is optimised on the very data it is scored on and the square one is not. Any
 * transient error the tracker carries (a still-converging w/h, an ego-motion-inflated size) is charged to
 * the SQUARE hypothesis alone and reads as evidence for ROUND. The validated offline harness
 * (compare_models.cpp) refits BOTH models — the live port dropped that.
 *
 * This harness measures the size of that unfairness on a synthetic cloud with the live table's geometry
 * (1.26 x 0.686 m, H 0.74), and checks that the SYMMETRIC comparison gets both shapes right.
 *
 * Build: cmake -S tests -B tests/build && make -C tests/build && ./tests/build/shape_asymmetry
 */

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <limits>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <random>
#include <tuple>
#include <vector>

#include <Eigen/Dense>

#include "round_table_belief.h"
#include "table_belief.h"

using rc::RoundBase;         using rc::RoundTableBelief;  using rc::RoundTableParams;
using rc::RoundTableState;   using rc::TableBelief;       using rc::TableBeliefParams;
using rc::TableBeliefState;  using rc::TableFrame;

namespace
{
constexpr float SIGMA_BASE = 0.03f;                 // same R for both models -> the normaliser cancels
constexpr float R_VAR      = SIGMA_BASE * SIGMA_BASE;
constexpr int   FIT_ITERS  = 40;                    // what evaluate_shape gives the round model

// The live table_3 geometry, as fitted (ai2_log.csv cycle 2290).
constexpr float TRUE_W = 1.263f, TRUE_H_EXT = 0.686f, TRUE_TOP_Z = 0.740f;

// Rectangular table cloud: top slab + 4 legs + a little floor clutter. `visible_frac` keeps only the points
// with local y above a cut, emulating the partial coverage a support bank accumulates from one side.
std::vector<Eigen::Vector3f> make_rect_cloud(std::mt19937& rng, float w, float h, float top_z,
                                             float visible_frac)
{
    std::normal_distribution<float>       noise(0.0f, 0.01f);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f), U01(0.0f, 1.0f);
    const float cx = 0.20f, cy = -0.30f, yaw = 0.30f, top_t = 0.03f, lr = 0.025f;
    const float c = std::cos(yaw), sn = std::sin(yaw);
    const auto  to_world = [&](float lx, float ly, float lz) -> Eigen::Vector3f
    { return {cx + c * lx - sn * ly, cy + sn * lx + c * ly, lz}; };
    const float y_cut = 0.5f * h * (1.0f - 2.0f * visible_frac);   // keep ly > y_cut

    std::vector<Eigen::Vector3f> pts;
    for (int i = 0; i < 2400; ++i)
    {
        const float lx = U(rng) * 0.5f * w, ly = U(rng) * 0.5f * h;
        if (ly > y_cut) pts.push_back(to_world(lx, ly, top_z + noise(rng)));
    }
    for (int k = 0; k < 4; ++k)
    {
        const float sx = (k == 0 or k == 3) ? 1.f : -1.f, sy = (k < 2) ? 1.f : -1.f;
        const float lx0 = sx * (0.5f * w - lr), ly0 = sy * (0.5f * h - lr);
        if (ly0 <= y_cut) continue;
        for (int i = 0; i < 200; ++i)
        {
            const float ang = U(rng) * static_cast<float>(M_PI), z = U01(rng) * (top_z - top_t);
            pts.push_back(to_world(lx0 + lr * std::cos(ang) + noise(rng),
                                   ly0 + lr * std::sin(ang) + noise(rng), z));
        }
    }
    for (int i = 0; i < 150; ++i) pts.push_back({cx + U(rng) * 1.5f, cy + U(rng) * 1.5f, U01(rng) * 0.05f});
    return pts;
}

// Round (disc-top, pedestal) table cloud — the known-good case the fix must NOT invert.
std::vector<Eigen::Vector3f> make_round_cloud(std::mt19937& rng, float radius, float top_z)
{
    std::normal_distribution<float>       noise(0.0f, 0.01f);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f), U01(0.0f, 1.0f),
        Uang(0.0f, 2.0f * static_cast<float>(M_PI));
    const float cx = 0.20f, cy = -0.30f, top_t = 0.03f, ped_r = 0.06f;
    std::vector<Eigen::Vector3f> pts;
    for (int i = 0; i < 1400; ++i)   // area-uniform disc top, plus a denser rim band
    {
        const float rho = radius * std::sqrt(U01(rng)), phi = Uang(rng);
        pts.push_back({cx + rho * std::cos(phi), cy + rho * std::sin(phi), top_z + noise(rng)});
    }
    for (int i = 0; i < 400; ++i)
    {
        const float rho = radius * (0.9f + 0.1f * U01(rng)), phi = Uang(rng);
        pts.push_back({cx + rho * std::cos(phi), cy + rho * std::sin(phi), top_z + noise(rng)});
    }
    for (int i = 0; i < 200; ++i)
    {
        const float ang = Uang(rng), z = U01(rng) * (top_z - top_t);
        pts.push_back({cx + ped_r * std::cos(ang) + noise(rng), cy + ped_r * std::sin(ang) + noise(rng), z});
    }
    for (int i = 0; i < 150; ++i) pts.push_back({cx + U(rng) * 1.5f, cy + U(rng) * 1.5f, U01(rng) * 0.05f});
    return pts;
}

// The plain SDF-mixture parameters BOTH hypotheses share (evaluate_shape's round params, mirrored).
TableBeliefParams square_params()
{
    TableBeliefParams p;
    p.sigma_base_m = SIGMA_BASE;
    p.top_thickness = rc::TableBeliefParams{}.top_thickness;
    return p;
}

float round_energy(const std::vector<Eigen::Vector3f>& cloud, const TableBeliefState& seed, int iters,
                   float* out_radius = nullptr)
{
    RoundTableParams rp; rp.sigma_base_m = SIGMA_BASE;
    RoundTableBelief b({seed.cx, seed.cy, seed.H, 0.25f * (seed.w + seed.h)}, rp, RoundBase::Ring);
    TableFrame f; f.points = cloud;
    for (int it = 0; it < iters; ++it) b.update(f);
    if (out_radius) *out_radius = b.state().radius;
    return b.mean_energy(cloud, b.state(), R_VAR);
}

// Square energy WITHOUT a refit (what evaluate_shape does today): score the state as handed in.
float square_energy_as_is(const std::vector<Eigen::Vector3f>& cloud, const TableBeliefState& s)
{
    TableBelief b(s, square_params());
    return b.mean_energy(cloud, s, R_VAR);
}

// Square energy WITH the same refit budget the round model gets (the fix). `resolve` mirrors the offline
// harness, which re-runs the w<->h orientation mode comparison at every iteration.
float square_energy_refit(const std::vector<Eigen::Vector3f>& cloud, const TableBeliefState& seed, int iters,
                          bool resolve, TableBeliefState* out = nullptr)
{
    TableBelief b(seed, square_params());
    TableFrame f; f.points = cloud;
    for (int it = 0; it < iters; ++it)
    {
        b.update(f);
        if (resolve) b.resolve_orientation(cloud, R_VAR);
    }
    if (out) *out = b.state();
    return b.mean_energy(cloud, b.state(), R_VAR);
}

// Cloud-derived seed (compare_models' init_from_cloud): xy centroid, 90th-percentile z as the top, and the
// mean top-band radius as a size hint. Depends on the CLOUD only — so neither hypothesis inherits the
// tracker's transient error, and the comparison is seed-symmetric as well as budget-symmetric.
TableBeliefState seed_from_cloud(const std::vector<Eigen::Vector3f>& pts)
{
    double sx = 0, sy = 0;
    std::vector<float> zs; zs.reserve(pts.size());
    for (const auto& p : pts) { sx += p.x(); sy += p.y(); zs.push_back(p.z()); }
    const float cx = static_cast<float>(sx / pts.size()), cy = static_cast<float>(sy / pts.size());
    std::sort(zs.begin(), zs.end());
    const float H = zs[static_cast<std::size_t>(0.90 * (zs.size() - 1))];
    float rad = 0.0f; int n = 0;
    for (const auto& p : pts)
        if (std::abs(p.z() - H) < 0.08f) { rad += std::hypot(p.x() - cx, p.y() - cy); ++n; }
    rad = n > 0 ? rad / n * 1.3f : 0.5f;
    rad = std::clamp(rad, 0.2f, 1.5f);
    return {cx, cy, H, 1.2f * rad, 0.8f * rad, 0.0f};
}

// PCA of the top-band points -> the footprint's actual major/minor extent and its axis. The mean-radius hint
// a mean over a FILLED top is ~0.5 of the half-diagonal, so it seeds the box far too small; on a real bank
// the box then settles into a local minimum where the outlying points are written off as CLUTTER and never
// pull it open, while the one-parameter disc escapes. That is optimiser luck deciding a shape verdict.
TableBeliefState seed_from_extent(const std::vector<Eigen::Vector3f>& pts)
{
    const TableBeliefState c = seed_from_cloud(pts);
    double sxx = 0, syy = 0, sxy = 0; int n = 0;
    double mx = 0, my = 0;
    for (const auto& p : pts)
        if (std::abs(p.z() - c.H) < 0.08f) { mx += p.x(); my += p.y(); ++n; }
    if (n < 8) return c;
    mx /= n; my /= n;
    for (const auto& p : pts)
        if (std::abs(p.z() - c.H) < 0.08f)
        {
            const double dx = p.x() - mx, dy = p.y() - my;
            sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
        }
    sxx /= n; syy /= n; sxy /= n;
    const double tr = sxx + syy, det = sxx * syy - sxy * sxy;
    const double l1 = 0.5 * tr + std::sqrt(std::max(0.0, 0.25 * tr * tr - det));
    const double l2 = 0.5 * tr - std::sqrt(std::max(0.0, 0.25 * tr * tr - det));
    const float  phi = static_cast<float>(0.5 * std::atan2(2.0 * sxy, sxx - syy));
    // For a uniformly filled rectangle of side L the variance along that side is L^2/12 -> L = sqrt(12*var).
    const float  w = std::clamp(static_cast<float>(std::sqrt(12.0 * std::max(1e-6, l1))), 0.2f, 3.0f);
    const float  h = std::clamp(static_cast<float>(std::sqrt(12.0 * std::max(1e-6, l2))), 0.2f, 3.0f);
    return {static_cast<float>(mx), static_cast<float>(my), c.H, w, h, phi};
}

std::vector<Eigen::Vector3f> load_xyz(const char* path)
{
    std::vector<Eigen::Vector3f> pts;
    std::ifstream fin(path);
    std::string   line;
    while (std::getline(fin, line))
    {
        for (char& ch : line) if (ch == ',') ch = ' ';
        std::istringstream ss(line);
        float x, y, z;
        if (ss >> x >> y >> z) pts.push_back({x, y, z});
    }
    return pts;
}

// MULTI-START: each hypothesis is fitted from EVERY seed and judged at its BEST (lowest-energy) fit. A model
// comparison is meant to compare the two models' best accounts of the data; if each side is instead scored at
// whatever local optimum one arbitrary seed happened to reach, the verdict measures optimiser luck. That is
// not hypothetical here — on the real 401-point bank the box reaches 0.61x0.68 (e=5.156, loses) from the
// mean-radius seed and 1.12x0.10 (e=4.533, wins) from the PCA-extent one, on the same points.
float shape_lbf_multistart(const std::vector<Eigen::Vector3f>& cloud, int iters, TableBeliefState* sq_out,
                           float* rd_out)
{
    const TableBeliefState seeds[] = {seed_from_cloud(cloud), seed_from_extent(cloud)};
    float e_sq = std::numeric_limits<float>::max(), e_rd = std::numeric_limits<float>::max();
    for (const auto& s : seeds)
    {
        TableBeliefState fit{};
        if (const float e = square_energy_refit(cloud, s, iters, true, &fit); e < e_sq)
        { e_sq = e; if (sq_out) *sq_out = fit; }
        float rad = 0.0f;
        if (const float e = round_energy(cloud, s, iters, &rad); e < e_rd)
        { e_rd = e; if (rd_out) *rd_out = rad; }
    }
    return e_sq - e_rd;
}

// One cloud, one seed: print the old (as-is) verdict and every refit variant. Returns the verdict of the
// SHIPPED rule — cloud-derived seed, equal budgets, resolve_orientation. lbf > 0 means "round".
float report(const char* tag, const std::vector<Eigen::Vector3f>& cloud, const TableBeliefState& seed,
             bool expect_round)
{
    std::printf("  [%s] seed w=%.2f h=%.2f yaw=%.0f deg\n", tag, seed.w, seed.h,
                seed.yaw * 180.0f / static_cast<float>(M_PI));
    const TableBeliefState cseed = seed_from_cloud(cloud);
    float ship_lbf = 0.0f;
    for (const auto& [label, iters, resolve, as_is, cloud_seed] :
         std::initializer_list<std::tuple<const char*, int, bool, bool, bool>>{
             {"OLD as-is       ", 0, false, true, false},
             {"refit40 trkseed ", 40, false, false, false},
             {"refit60 trkseed ", 60, true, false, false},
             {"1-start cloudsd ", 20, true, false, true}})
    {
        const TableBeliefState& s = cloud_seed ? cseed : seed;
        float radius = 0.0f;
        const float e_rd = round_energy(cloud, s, iters > 0 ? iters : FIT_ITERS, &radius);
        TableBeliefState fit = s;
        const float e_sq = as_is ? square_energy_as_is(cloud, s)
                                 : square_energy_refit(cloud, s, iters, resolve, &fit);
        const float lbf  = e_sq - e_rd;
        const bool  says_round = lbf > 0.0f;
        std::printf("    %s e_sq=%.4f e_rd=%.4f lbf=%+.4f -> %-6s %s   (square %.2fx%.2f, disc r=%.2f)\n",
                    label, e_sq, e_rd, lbf, says_round ? "round" : "square",
                    says_round == expect_round ? "  ok" : "WRONG", fit.w, fit.h, radius);
        (void) cloud_seed;
    }
    TableBeliefState fit{}; float rad = 0.0f;
    ship_lbf = shape_lbf_multistart(cloud, 20, &fit, &rad);   // ← the SHIPPED rule
    std::printf("    SHIPPED multistart                                     lbf=%+.4f -> %-6s %s   "
                "(square %.2fx%.2f, disc r=%.2f)\n",
                ship_lbf, ship_lbf > 0.0f ? "round" : "square",
                (ship_lbf > 0.0f) == expect_round ? "  ok" : "WRONG", fit.w, fit.h, rad);
    return ship_lbf;
}
}   // namespace

// Real-cloud mode: the seed study on a bank dumped by the live agent (TableConcept.DumpCloudPath).
int real_cloud(const char* path)
{
    const auto cloud = load_xyz(path);
    if (cloud.size() < 50) { std::printf("too few points in %s\n", path); return 2; }
    std::printf("=== %s : %zu pts ===\n", path, cloud.size());
    for (const auto& [label, seed] : std::initializer_list<std::pair<const char*, TableBeliefState>>{
             {"mean-radius seed", seed_from_cloud(cloud)},
             {"PCA-extent seed ", seed_from_extent(cloud)}})
    {
        TableBeliefState fit{};
        const float e_sq = square_energy_refit(cloud, seed, 20, true, &fit);
        float       rad  = 0.0f;
        const float e_rd = round_energy(cloud, seed, 20, &rad);
        std::printf("  %s : seed %.2fx%.2f -> square %.2fx%.2f e=%.4f | disc r=%.2f e=%.4f | lbf=%+.4f -> %s\n",
                    label, seed.w, seed.h, fit.w, fit.h, e_sq, rad, e_rd, e_sq - e_rd,
                    e_sq - e_rd > 0.0f ? "ROUND" : "square");
    }
    TableBeliefState fit{}; float rad = 0.0f;
    const float lbf = shape_lbf_multistart(cloud, 20, &fit, &rad);
    std::printf("  MULTI-START     : square %.2fx%.2f | disc r=%.2f | lbf=%+.4f -> %s\n", fit.w, fit.h, rad,
                lbf, lbf > 0.0f ? "ROUND" : "square");
    return 0;
}

int main(int argc, char** argv)
{
    if (argc > 1) return real_cloud(argv[1]);

    std::mt19937 rng(12345);
    int failures = 0;

    // ── 1. A RECTANGULAR table (the live table_3 geometry), seeded from a drifting tracker state ─────
    // lbf = e_square - e_round; > 0 means the agent says ROUND.
    const auto rect = make_rect_cloud(rng, TRUE_W, TRUE_H_EXT, TRUE_TOP_Z, 0.75f);
    const TableBeliefState truth{0.20f, -0.30f, TRUE_TOP_Z, TRUE_W, TRUE_H_EXT, 0.30f};
    std::printf("=== RECTANGULAR cloud %.2f x %.2f m, %zu pts (expect SQUARE) ===\n", TRUE_W, TRUE_H_EXT,
                rect.size());
    for (const float d : {0.00f, 0.10f, 0.25f})
    {
        // A tracker mid-convergence: extent too long/narrow, centre and yaw still off.
        TableBeliefState s = truth;
        s.w += d; s.h -= 0.5f * d; s.cx += 0.5f * d; s.yaw += d;
        if (report(d == 0.0f ? "converged tracker" : "tracker mid-convergence", rect, s, false) > 0.0f)
            ++failures;
    }

    // ── 2. A genuinely ROUND table — the case the fix must not invert ────────────────────────────────
    const auto rcloud = make_round_cloud(rng, 0.55f, TRUE_TOP_Z);
    std::printf("\n=== ROUND cloud r=0.55 m, %zu pts (expect ROUND) ===\n", rcloud.size());
    for (const float d : {0.00f, 0.25f})
    {
        TableBeliefState s{0.20f, -0.30f, TRUE_TOP_Z, 1.10f + d, 1.10f - 0.5f * d, d};
        if (report(d == 0.0f ? "converged tracker" : "tracker mid-convergence", rcloud, s, true) <= 0.0f)
            ++failures;
    }

    // ── 3. Cost. evaluate_shape runs on the agent's main thread every ShapeEvalPeriod cycles, so the refit
    // budget is paid as a HITCH in the belief loop, not amortised. Measure both hypotheses at bank size.
    {
        const auto  bank = make_rect_cloud(rng, TRUE_W, TRUE_H_EXT, TRUE_TOP_Z, 1.0f);
        const auto  s    = seed_from_cloud(bank);
        square_energy_refit(bank, s, 5, true);   // warm the caches so the first row is not the outlier
        std::printf("\n=== cost AND convergence per evaluation, %zu pts (rect table -> expect lbf < 0) ===\n",
                    bank.size());
        const auto& rnd_bank = rcloud;            // the weak direction: expect lbf > 0
        const auto  rnd_seed = seed_from_cloud(rnd_bank);
        // Stride-subsample: both hypotheses see the SAME points, so the comparison is PAIRED and the
        // variance of the DIFFERENCE is far smaller than that of either mean. Cost is linear in the count.
        const auto stride_to = [](const std::vector<Eigen::Vector3f>& c, std::size_t cap)
        {
            if (c.size() <= cap) return c;
            std::vector<Eigen::Vector3f> out; out.reserve(cap);
            const double step = static_cast<double>(c.size()) / static_cast<double>(cap);
            for (std::size_t i = 0; i < cap; ++i) out.push_back(c[static_cast<std::size_t>(i * step)]);
            return out;
        };
        for (const std::size_t cap : {std::size_t{1000}, std::size_t{100000}})
            for (const int iters : {20})
            {
                const auto ms = [](auto a, auto b)
                { return std::chrono::duration<double, std::milli>(b - a).count(); };
                const auto rect_c = stride_to(bank, cap), rnd_c = stride_to(rnd_bank, cap);
                TableBeliefState fit{}; float rad = 0.0f;
                auto        t0    = std::chrono::steady_clock::now();
                const float lbf   = shape_lbf_multistart(rect_c, iters, &fit, &rad);   // SHIPPED: 2 seeds
                auto        t1    = std::chrono::steady_clock::now();
                const float lbf_r = shape_lbf_multistart(rnd_c, iters, nullptr, nullptr);
                std::printf("  n=%5zu %2d iters: %6.1f ms/eval | rect lbf=%+.4f (%.2fx%.2f) %-5s | round lbf=%+.4f %s\n",
                            rect_c.size(), iters, ms(t0, t1), lbf, fit.w, fit.h,
                            lbf < 0.0f ? "ok" : "WRONG", lbf_r, lbf_r > 0.0f ? "ok" : "WRONG");
            }
    }

    std::printf("\n%s\n", failures == 0
                              ? "PASS: the shipped rule gets both shapes right, from EVERY tracker state"
                              : "FAIL: the shipped rule still misclassifies");
    return failures == 0 ? 0 : 1;
}
