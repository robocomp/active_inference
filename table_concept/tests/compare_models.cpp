/*
 * compare_models.cpp  —  square-vs-round table model selection by free energy / model evidence.
 *
 * Fits the production square TableBelief AND the round RoundTableBelief to the SAME mask point cloud and
 * compares their evidence. Pure Eigen, no DSR/torch. Two modes:
 *   ./compare_models              synthetic A/B self-validation (asserts the FE comparison discriminates shape)
 *   ./compare_models cloud.xyz    fit both to a real captured cloud and print the decision table
 *
 * Metric: TableBelief::mean_energy / RoundTableBelief::mean_energy — the clutter-inclusive mean per-point
 * −log mixture marginal likelihood. Both models fit the SAME points with the SAME R (sigma_base=0.03), so
 * the dropped (2πR)^-3/2 normaliser cancels. Reports the data-fit total NLL, the Occam-corrected Laplace
 * log-evidence (½log|Σ_post| complexity term) and BIC, and the log-Bayes-factor.
 *
 * The ROUND `Ring` variant (5 prims, matched cardinality with the square's top+4legs) is the clean
 * discriminator and the one the synthetic asserts use; the `Pedestal` variant (2 prims, faithful to a real
 * pedestal table) is reported as a physical-realism cross-check but carries a mixture-weight baseline offset.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "table_belief.h"
#include "round_table_belief.h"

using rc::TableBelief;  using rc::TableBeliefState;  using rc::TableBeliefParams;  using rc::TableFrame;
using rc::RoundTableBelief;  using rc::RoundTableState;  using rc::RoundTableParams;  using rc::RoundBase;

namespace
{
constexpr float R_SIGMA_BASE = 0.03f;   // forced identical for BOTH models so R matches (normaliser cancels)
constexpr int   FIT_ITERS    = 60;      // both models run to convergence

struct Score
{
    std::string name;
    int         dof   = 0;
    long        npts  = 0;
    double      meanE = 0.0;   // mean per-point −log mixture (clutter-inclusive)
    double      nll   = 0.0;   // total data energy = npts·meanE  (lower = better fit)
    double      half_logdet_Pinfo = 0.0;   // ½ log|P_post| (Occam complexity; higher = more finely tuned)
    double      logZ  = 0.0;   // Laplace log-evidence up to shared constants (higher = better)
    double      bic   = 0.0;   // 2·nll + dof·ln(npts)  (lower = better)
};

// log det of an SPD matrix via LDLT (stable; sum of log of the D diagonal).
double log_det_spd(const Eigen::MatrixXd& S)
{
    Eigen::LDLT<Eigen::MatrixXd> ldlt(S);
    const Eigen::VectorXd D = ldlt.vectorD();
    double s = 0.0;
    for (int i = 0; i < D.size(); ++i) s += std::log(std::max(1e-30, D(i)));
    return s;
}

// Assemble a Score from the converged fit. `cov` = posterior covariance Σ_post (P_info = Σ_post⁻¹).
Score make_score(const std::string& name, int dof, long npts, double meanE, const Eigen::MatrixXd& cov)
{
    Score s;
    s.name = name; s.dof = dof; s.npts = npts; s.meanE = meanE;
    s.nll  = static_cast<double>(npts) * meanE;
    const double logdet_cov = log_det_spd(cov);          // log|Σ_post|
    s.half_logdet_Pinfo = -0.5 * logdet_cov;             // ½ log|P_post| = −½ log|Σ_post|
    // Laplace evidence: log p(D) ≈ −nll + (d/2)ln2π + ½log|Σ_post|  (shared −(3/2)Nlog2πR + param-prior const dropped)
    s.logZ = -s.nll + 0.5 * dof * std::log(2.0 * M_PI) + 0.5 * logdet_cov;
    s.bic  = 2.0 * s.nll + dof * std::log(std::max(1.0, static_cast<double>(npts)));
    return s;
}

// Robust init from the cloud: xy centroid, a high-quantile top height, and a top-band radial extent hint.
struct InitHint { float cx, cy, H, rad; };
InitHint init_from_cloud(const std::vector<Eigen::Vector3f>& pts)
{
    double sx = 0, sy = 0;
    std::vector<float> zs; zs.reserve(pts.size());
    for (const auto& p : pts) { sx += p.x(); sy += p.y(); zs.push_back(p.z()); }
    const float cx = static_cast<float>(sx / pts.size());
    const float cy = static_cast<float>(sy / pts.size());
    std::sort(zs.begin(), zs.end());
    const float H = zs[static_cast<std::size_t>(0.90 * (zs.size() - 1))];   // 90th-percentile z ≈ tabletop
    // radial extent of top-band points (z within 8 cm of H)
    float rad = 0.0f; int n = 0;
    for (const auto& p : pts)
        if (std::abs(p.z() - H) < 0.08f)
        { rad += std::hypot(p.x() - cx, p.y() - cy); ++n; }
    rad = n > 0 ? rad / n * 1.3f : 0.5f;   // mean radius → a size hint (×1.3 toward the rim)
    return {cx, cy, H, std::clamp(rad, 0.2f, 1.5f)};
}

float rim_fraction(const std::vector<Eigen::Vector3f>& pts, const InitHint& h)
{
    // fraction of top-band points in the outer 20% radial annulus — where disc-vs-box discriminates
    int band = 0, rim = 0;
    for (const auto& p : pts)
        if (std::abs(p.z() - h.H) < 0.08f)
        {
            ++band;
            if (std::hypot(p.x() - h.cx, p.y() - h.cy) > 0.8f * h.rad) ++rim;
        }
    return band > 0 ? static_cast<float>(rim) / band : 0.0f;
}

// ── Fits ────────────────────────────────────────────────────────────────────────────────────────────
Score fit_square(const std::vector<Eigen::Vector3f>& cloud, bool verbose)
{
    TableBeliefState::use_quotient = false;                 // natural [cx,cy,H,w,h,yaw,t] basis for the log-det
    TableBeliefParams P;                                    // defaults: all extra factors OFF, sigma_base 0.03
    P.sigma_base_m = R_SIGMA_BASE;
    const auto h = init_from_cloud(cloud);
    TableBelief b(TableBeliefState{h.cx, h.cy, h.H, 1.2f * h.rad, 0.8f * h.rad, 0.0f}, P);
    TableFrame f; f.points = cloud;
    const float R = R_SIGMA_BASE * R_SIGMA_BASE;
    for (int it = 0; it < FIT_ITERS; ++it) { b.update(f); b.resolve_orientation(cloud, R); }
    const auto& s = b.state();
    const float E = b.mean_energy(cloud, s, R);
    if (verbose)
        std::printf("  square : cx=%.3f cy=%.3f H=%.3f w=%.3f h=%.3f yaw=%.1f deg | mode_p=%.2f sigma_yaw=%.1f deg\n",
                    s.cx, s.cy, s.H, s.w, s.h, s.yaw * 180.0f / static_cast<float>(M_PI),
                    b.mode_posterior(), std::sqrt(b.yaw_marginal_var()) * 180.0f / static_cast<float>(M_PI));
    return make_score("square(box)", TableBelief::N, static_cast<long>(cloud.size()), E,
                      b.covariance().cast<double>());
}

Score fit_round(const std::vector<Eigen::Vector3f>& cloud, RoundBase base, const char* label, bool verbose)
{
    RoundTableParams P; P.sigma_base_m = R_SIGMA_BASE;
    const auto h = init_from_cloud(cloud);
    RoundTableBelief b(RoundTableState{h.cx, h.cy, h.H, h.rad}, P, base);
    TableFrame f; f.points = cloud;
    const float R = R_SIGMA_BASE * R_SIGMA_BASE;
    for (int it = 0; it < FIT_ITERS; ++it) b.update(f);
    const auto& s = b.state();
    const float E = b.mean_energy(cloud, s, R);
    if (verbose)
        std::printf("  %-7s: cx=%.3f cy=%.3f H=%.3f radius=%.3f\n", label, s.cx, s.cy, s.H, s.radius);
    return make_score(std::string("round(") + label + ")", RoundTableBelief::N,
                      static_cast<long>(cloud.size()), E, b.covariance().cast<double>());
}

void print_table(const std::vector<Score>& scores)
{
    std::printf("  %-14s %5s %6s %10s %10s %13s %11s %11s\n",
                "model", "dof", "npts", "meanE", "totalNLL", "half_logdetP", "logZ", "BIC");
    for (const auto& s : scores)
        std::printf("  %-14s %5d %6ld %10.4f %10.1f %13.2f %11.1f %11.1f\n",
                    s.name.c_str(), s.dof, s.npts, s.meanE, s.nll, s.half_logdet_Pinfo, s.logZ, s.bic);
}

// ── Synthetic clouds ──────────────────────────────────────────────────────────────────────────────────
std::vector<Eigen::Vector3f> make_square_cloud(std::mt19937& rng)
{
    std::normal_distribution<float> noise(0.0f, 0.01f);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f), U01(0.0f, 1.0f);
    const float cx = 0.20f, cy = -0.30f, H = 0.74f, w = 1.50f, hgt = 1.00f, yaw = 0.30f, top_t = 0.03f, lr = 0.025f;
    const float c = std::cos(yaw), sn = std::sin(yaw);
    const auto to_world = [&](float lx, float ly, float lz) -> Eigen::Vector3f
    { return {cx + c * lx - sn * ly, cy + sn * lx + c * ly, lz}; };
    std::vector<Eigen::Vector3f> pts;
    for (int i = 0; i < 1200; ++i)
        pts.push_back(to_world(U(rng) * 0.5f * w, U(rng) * 0.5f * hgt, H + noise(rng)));
    for (int k = 0; k < 4; ++k)
    {
        const float sx = (k == 0 or k == 3) ? 1.f : -1.f, sy = (k < 2) ? 1.f : -1.f;
        const float cxk = sx * (0.5f * w - lr), cyk = sy * (0.5f * hgt - lr);
        for (int i = 0; i < 200; ++i)
        {
            const float ang = U(rng) * static_cast<float>(M_PI);
            const float z   = U01(rng) * (H - top_t);
            pts.push_back(to_world(cxk + lr * std::cos(ang) + noise(rng), cyk + lr * std::sin(ang) + noise(rng), z));
        }
    }
    for (int i = 0; i < 150; ++i) pts.push_back({cx + U(rng) * 1.5f, cy + U(rng) * 1.5f, U01(rng) * 0.05f});
    return pts;
}

std::vector<Eigen::Vector3f> make_round_cloud(std::mt19937& rng)
{
    std::normal_distribution<float> noise(0.0f, 0.01f);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f), U01(0.0f, 1.0f), Uang(0.0f, 2.0f * static_cast<float>(M_PI));
    const float cx = 0.20f, cy = -0.30f, H = 0.74f, radius = 0.55f, top_t = 0.03f, ped_r = 0.06f;
    std::vector<Eigen::Vector3f> pts;
    // area-uniform disc top (ρ = radius·√U)
    for (int i = 0; i < 1000; ++i)
    {
        const float rho = radius * std::sqrt(U01(rng)), phi = Uang(rng);
        pts.push_back({cx + rho * std::cos(phi), cy + rho * std::sin(phi), H + noise(rng)});
    }
    // dense RIM band ρ∈[0.9r, r] — the discriminating signal (box corners/edges can't fit this)
    for (int i = 0; i < 400; ++i)
    {
        const float rho = radius * (0.9f + 0.1f * U01(rng)), phi = Uang(rng);
        pts.push_back({cx + rho * std::cos(phi), cy + rho * std::sin(phi), H + noise(rng)});
    }
    // central pedestal
    for (int i = 0; i < 200; ++i)
    {
        const float ang = Uang(rng), z = U01(rng) * (H - top_t);
        pts.push_back({cx + ped_r * std::cos(ang) + noise(rng), cy + ped_r * std::sin(ang) + noise(rng), z});
    }
    for (int i = 0; i < 150; ++i) pts.push_back({cx + U(rng) * 1.5f, cy + U(rng) * 1.5f, U01(rng) * 0.05f});
    return pts;
}

std::vector<Eigen::Vector3f> load_xyz(const std::string& path)
{
    std::vector<Eigen::Vector3f> pts;
    std::ifstream f(path);
    if (not f) { std::fprintf(stderr, "ERROR: cannot open %s\n", path.c_str()); return pts; }
    std::string line;
    while (std::getline(f, line))
    {
        for (char& ch : line) if (ch == ',') ch = ' ';   // accept CSV or whitespace
        std::istringstream ss(line);
        float x, y, z;
        if (ss >> x >> y >> z) pts.push_back({x, y, z});
    }
    return pts;
}

int decide(const std::vector<Eigen::Vector3f>& cloud, const char* tag)
{
    const auto hint = init_from_cloud(cloud);
    std::printf("[%s]  npts=%zu  centroid=(%.2f,%.2f)  H~%.2f  rad~%.2f  rim_frac=%.2f\n",
                tag, cloud.size(), hint.cx, hint.cy, hint.H, hint.rad, rim_fraction(cloud, hint));
    if (rim_fraction(cloud, hint) < 0.08f)
        std::printf("  ** WARN: few rim points — disc-vs-box barely discriminates on this cloud **\n");
    const Score sq   = fit_square(cloud, true);
    const Score rrng = fit_round(cloud, RoundBase::Ring, "ring", true);
    const Score rped = fit_round(cloud, RoundBase::Pedestal, "pedestal", true);
    print_table({sq, rrng, rped});
    std::printf("  log Bayes factor (round[ring] : square) = %.1f   (>0 ⇒ ROUND wins the clean A/B)\n",
                rrng.logZ - sq.logZ);
    const bool round_wins = rrng.nll < sq.nll and rrng.logZ > sq.logZ;
    std::printf("  VERDICT: %s\n", round_wins ? "ROUND better explains this cloud"
                                              : "SQUARE better explains this cloud");
    return round_wins ? 1 : 0;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc > 1)   // real-cloud mode
    {
        const auto cloud = load_xyz(argv[1]);
        if (cloud.size() < 50) { std::fprintf(stderr, "too few points (%zu)\n", cloud.size()); return 2; }
        decide(cloud, argv[1]);
        return 0;
    }

    // Synthetic A/B self-validation — the CLEAN discriminator is round(ring) (matched cardinality).
    std::mt19937 rng(12345);
    std::printf("=== synthetic ROUND cloud (expect round[ring] to win) ===\n");
    const auto round_cloud = make_round_cloud(rng);
    const bool round_ok = [&] {
        const Score sq = fit_square(round_cloud, true);
        const Score rr = fit_round(round_cloud, RoundBase::Ring, "ring", true);
        const Score rp = fit_round(round_cloud, RoundBase::Pedestal, "pedestal", true);
        print_table({sq, rr, rp});
        return rr.nll < sq.nll and rr.logZ > sq.logZ;
    }();

    std::printf("\n=== synthetic SQUARE cloud (expect square to win) ===\n");
    const auto square_cloud = make_square_cloud(rng);
    const bool square_ok = [&] {
        const Score sq = fit_square(square_cloud, true);
        const Score rr = fit_round(square_cloud, RoundBase::Ring, "ring", true);
        const Score rp = fit_round(square_cloud, RoundBase::Pedestal, "pedestal", true);
        print_table({sq, rr, rp});
        return sq.nll < rr.nll and sq.logZ > rr.logZ;
    }();

    std::printf("\n=== self-validation: round_cloud→round %s | square_cloud→square %s ===\n",
                round_ok ? "PASS" : "FAIL", square_ok ? "PASS" : "FAIL");
    return (round_ok and square_ok) ? 0 : 1;
}
