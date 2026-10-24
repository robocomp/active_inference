/*
 * layout_optimizer.cpp — see layout_optimizer.h.
 */
#include "layout_optimizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <functional>
#include <limits>
#include <numbers>
#include <ranges>

namespace ltsm::layout
{

namespace
{
constexpr double kPi = std::numbers::pi;

double wrap(double a) { return std::atan2(std::sin(a), std::cos(a)); }

Eigen::Matrix2d rot(double a)
{
    Eigen::Matrix2d R;
    R << std::cos(a), -std::sin(a), std::sin(a), std::cos(a);
    return R;
}

Vec2 apply(const Vec3 &T, const Vec2 &p) { return rot(T.z()) * p + T.head<2>(); }

/// T(a <- b) from the two map poses.
Vec3 between(const Vec3 &a, const Vec3 &b)
{
    const Vec2 t = rot(-a.z()) * (b.head<2>() - a.head<2>());
    return {t.x(), t.y(), wrap(b.z() - a.z())};
}

/// log Φ(z), the standard normal CDF, without underflow in the far left tail (asymptotic form there).
double log_ndtr(double z)
{
    if (z > -30.0) return std::log(0.5 * std::erfc(-z / std::numbers::sqrt2));
    return -0.5 * z * z - std::log(-z) - 0.5 * std::log(2.0 * kPi);
}

/// log(I0(κ)·e^{−κ}): the von Mises normaliser, computed where I0 itself would overflow by its asymptote.
double log_i0e(double kappa)
{
    if (kappa < 600.0) return std::log(std::cyl_bessel_i(0.0, kappa)) - kappa;
    return -0.5 * std::log(2.0 * kPi * kappa);
}

double signed_area2(const Polygon &poly)
{
    double a2 = 0;
    for (std::size_t i = 0; i < poly.size(); ++i)
    {
        const auto &p = poly[i];
        const auto &q = poly[(i + 1) % poly.size()];
        a2 += p.x() * q.y() - q.x() * p.y();
    }
    return a2;
}

struct Sample { Vec2 p; double w; };
}   // namespace
double signed_distance(const Polygon &poly, const Vec2 &p);
namespace
{

std::vector<Sample> sample_outline(const Polygon &poly, const Model &m)
{
    std::vector<Sample> out;
    for (std::size_t i = 0; i < poly.size(); ++i)
    {
        const Vec2 p = poly[i], q = poly[(i + 1) % poly.size()];
        const double L = (q - p).norm();
        if (L <= 0) continue;
        const int n = std::max(1, static_cast<int>(std::ceil(L / m.sample_step)));
        for (int k = 0; k < n; ++k)
            out.push_back({p + (k + 0.5) / n * (q - p), (L / n) / m.corr_length});
    }
    // The INTERIOR too: "rooms are disjoint" is a statement about area. With the boundary alone two rooms laid exactly
    // on top of each other cost nothing — their outlines coincide — and a real solve found that mode. A grid of cell
    // centres strictly inside, each standing for its cell's area over corr_length².
    if (poly.size() >= 3)
    {
        Vec2 lo = poly.front(), hi = poly.front();
        for (const auto &p : poly) { lo = lo.cwiseMin(p); hi = hi.cwiseMax(p); }
        const double step = m.interior_step;
        for (double x = lo.x() + 0.5 * step; x < hi.x(); x += step)
            for (double y = lo.y() + 0.5 * step; y < hi.y(); y += step)
                if (signed_distance(poly, Vec2(x, y)) < -0.5 * step)   // the cell lies inside; edge cells are the boundary's
                    out.push_back({Vec2(x, y), step * step / (m.corr_length * m.corr_length)});
    }
    return out;
}

/// An outline edge (endpoints in the room's frame), its outward unit normal and length.
struct Edge { Vec2 p, q, n; double L; };

/// The outline edge that is the door's WALL: the one whose LINE best contains both aperture ends and whose segment is
/// nearest the aperture's centre (score = perp(end₀)² + perp(end₁)² + dist(centre, segment)², least wins). Nearest-edge
/// alone picks a door jamb on a notched outline — perpendicular to the wall, which then reads as a 90° misfit.
std::optional<Edge> wall_edge(const Polygon &poly, const std::array<Vec2, 2> &ends)
{
    if (poly.size() < 3) return std::nullopt;
    const double orient = signed_area2(poly) >= 0 ? 1.0 : -1.0;   // CCW ⇒ outward normal is the edge turned clockwise
    const Vec2 mid = 0.5 * (ends[0] + ends[1]);
    std::optional<Edge> best;
    double best_score = std::numeric_limits<double>::max();
    for (std::size_t i = 0; i < poly.size(); ++i)
    {
        const Vec2 p = poly[i], q = poly[(i + 1) % poly.size()];
        const Vec2 e = q - p;
        const double L = e.norm();
        if (L <= 0) continue;
        const Vec2 u = e / L, nrm(u.y(), -u.x());
        const double s = std::clamp((mid - p).dot(u), 0.0, L);
        const double score = std::pow((ends[0] - p).dot(nrm), 2) + std::pow((ends[1] - p).dot(nrm), 2)
                           + (mid - (p + s * u)).squaredNorm();
        if (score < best_score)
        {
            best_score = score;
            best = Edge{p, q, orient * nrm, L};
        }
    }
    return best;
}

/// One factor: which rooms (and whether ψ) it reads, how many residuals it writes, and how.
struct Factor
{
    enum Kind { kPassage, kManhattan, kOverlap, kAdjacency } kind;
    std::array<int, 2> room{-1, -1};
    bool psi = false;
    int rows = 0;
    std::function<void(const std::vector<Vec3> &, double, double *)> eval;
    int seam_rows = 0;   ///< kPassage: the first seam_rows rows are the seam, the rest the wall (for the energy split)
};
}   // namespace

std::optional<Axis> dominant_axis(const Polygon &outline)
{
    // Quadrupled angle: walls at α and α + kπ/2 vote the same. Weighted by length (a long wall is more evidence).
    std::complex<double> z1 = 0, z2 = 0;
    double sum_l = 0, sum_l2 = 0;
    for (std::size_t i = 0; i < outline.size(); ++i)
    {
        const Vec2 e = outline[(i + 1) % outline.size()] - outline[i];
        const double L = e.norm();
        if (L <= 0) continue;
        const double a = std::atan2(e.y(), e.x());
        z1 += L * std::polar(1.0, 4.0 * a);
        z2 += L * std::polar(1.0, 8.0 * a);
        sum_l += L;
        sum_l2 += L * L;
    }
    if (sum_l <= 0) return std::nullopt;
    z1 /= sum_l;
    z2 /= sum_l;
    const double R1 = std::abs(z1);
    // Standard error of a circular mean (Fisher 1993, §4.2.1) with Kish's effective count for the length weights.
    // A room with no dominant direction has R1 → 0 and an infinite error: it then pulls on nothing.
    const double n_eff = sum_l * sum_l / sum_l2;
    const double se4 = R1 > 0 ? std::sqrt(std::max(0.0, 1.0 - std::abs(z2)) / (2.0 * n_eff * R1 * R1))
                              : std::numeric_limits<double>::infinity();
    return Axis{0.25 * std::arg(z1), se4};
}

double signed_distance(const Polygon &poly, const Vec2 &p)
{
    double d = std::numeric_limits<double>::max();
    bool inside = false;
    for (std::size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++)
    {
        const Vec2 a = poly[j], b = poly[i];
        const Vec2 e = b - a;
        const double L2 = e.squaredNorm();
        const double s = L2 > 0 ? std::clamp((p - a).dot(e) / L2, 0.0, 1.0) : 0.0;
        d = std::min(d, (p - (a + s * e)).norm());
        if ((a.y() > p.y()) != (b.y() > p.y()) and p.x() < a.x() + (p.y() - a.y()) * e.x() / e.y()) inside = not inside;
    }
    return inside ? -d : d;
}

namespace
{
Result solve_from(const std::vector<Room> &rooms, const std::vector<Seam> &seams, const Model &m)
{
    Result res;
    const auto n = rooms.size();
    res.pose.resize(n);
    res.cov.assign(n, Mat3::Zero());
    res.p_manhattan.assign(n, -1.0);
    res.seam_d2.assign(seams.size(), 0.0);
    res.seam_p_inlier.assign(seams.size(), 1.0);
    for (std::size_t i = 0; i < n; ++i) res.pose[i] = rooms[i].init;
    if (n == 0) return res;

    // ── Parameters: (x, y, yaw) per free room, then ψ ─────────────────────────────────────────
    const bool any_fixed = std::ranges::any_of(rooms, &Room::fixed);
    std::vector<int> col(n, -1);
    int ncol = 0;
    for (std::size_t i = 0; i < n; ++i)
        if (not (rooms[i].fixed or (not any_fixed and i == 0))) { col[i] = ncol; ncol += 3; }

    // Per-room axis and the precision of "this room is square to ψ": the outline's own vote error, the vertex noise
    // propagated to the mean direction (16·2σ²n/(ΣL)² in 4φ, folded into se4 by the caller of dominant_axis), and how
    // square buildings are.
    std::vector<std::optional<Axis>> axis(n);
    std::vector<double> kappa(n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
    {
        if (not m.manhattan or rooms[i].outline.size() < 3) continue;
        axis[i] = dominant_axis(rooms[i].outline);
        if (not axis[i].has_value() or not std::isfinite(axis[i]->se4)) continue;
        double perim = 0;
        for (std::size_t k = 0; k < rooms[i].outline.size(); ++k)
            perim += (rooms[i].outline[(k + 1) % rooms[i].outline.size()] - rooms[i].outline[k]).norm();
        const double var_noise = 16.0 * 2.0 * m.sigma_outline * m.sigma_outline * rooms[i].outline.size() / (perim * perim);
        const double s4 = 4.0 * m.sigma_square;
        kappa[i] = 1.0 / (axis[i]->se4 * axis[i]->se4 + var_noise + s4 * s4);
    }
    const bool use_psi = std::ranges::any_of(kappa, [](double k) { return k > 0; });
    double psi = 0;
    int psi_col = -1;
    // E(u) = −log[ p·vM(u; κ) + (1−p)/2π ],  u = 4(θ + φ − ψ) wrapped: the MANHATTAN energy of one room.
    const auto manhattan_energy = [pm = m.p_manhattan](double k, double u)
    {
        const double l_in = std::log(pm) + k * (std::cos(u) - 1.0) - std::log(2.0 * kPi) - log_i0e(k);
        const double l_out = std::log(1.0 - pm) - std::log(2.0 * kPi);
        const double mx = std::max(l_in, l_out);
        return -(mx + std::log(std::exp(l_in - mx) + std::exp(l_out - mx)));
    };
    if (use_psi)
    {
        // Start ψ at the MODE, not the mean: each room's own world axis is a candidate and the one with the lowest total
        // energy wins. A mean is pulled off by any room that is not square (a 30° room drags it 10°), and a mixture with
        // κ ~ 200 is flat a few degrees from its peak, so LM started there never finds it.
        double best = std::numeric_limits<double>::max();
        for (std::size_t c = 0; c < n; ++c)
        {
            if (kappa[c] <= 0) continue;
            const double cand = rooms[c].init.z() + axis[c]->phi;
            double e = 0;
            for (std::size_t i = 0; i < n; ++i)
                if (kappa[i] > 0) e += manhattan_energy(kappa[i], wrap(4.0 * (rooms[i].init.z() + axis[i]->phi - cand)));
            if (e < best) { best = e; psi = cand; }
        }
        psi_col = ncol++;
    }

    // ── Factors ───────────────────────────────────────────────────────────────────────────────
    std::vector<Factor> factors;

    // How far each room reaches from its own origin: the support of "b is somewhere near" (OUTLIER, ADJACENCY).
    std::vector<double> reach(n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (const auto &p : rooms[i].outline) reach[i] = std::max(reach[i], p.norm());

    // PASSAGE = SEAM (+ WALL when both sides' walls are known), one inlier/outlier mixture over both:
    //   E(q) = −log[ π·N(q) + (1−π)·U ],  q = |whitened seam|² + |whitened wall|².
    // N's normaliser is the seam's 3-D Gaussian times the wall's 2-D one; U is uniform over b's pose near a (a disc of
    // radius `extent`, any yaw) times uniform over the wall's angle (lines: π) and separation (±extent).
    struct PassageModel { double log_n0 = 0, log_u = 0; bool mixture = false; };
    const auto passage_energy = [pi = m.p_seam_inlier](const PassageModel &pm, double q)
    {
        if (not pm.mixture) return 0.5 * q;
        const double l_in = std::log(pi) + pm.log_n0 - 0.5 * q, l_out = std::log(1.0 - pi) + pm.log_u;
        const double mx = std::max(l_in, l_out);
        return -(mx + std::log(std::exp(l_in - mx) + std::exp(l_out - mx)));
    };
    std::vector<PassageModel> passage_model;
    std::vector<std::function<double(const std::vector<Vec3> &)>> passage_q;   ///< per seam: q at a pose set
    if (m.seams)
        for (const auto &s : seams)
        {
            // Whitening W with WᵀW = cov⁻¹ (W = L⁻¹ for cov = LLᵀ).
            Eigen::LLT<Mat3> llt(s.cov);
            const Mat3 W = llt.info() == Eigen::Success ? Mat3(llt.matrixL().solve(Mat3::Identity()))
                                                        : Mat3(Mat3::Identity() / 0.1);   // see topology_layout: never reached with a stated or defaulted cov
            PassageModel pm;
            if (s.extent > 0)
            {
                pm.mixture = true;
                pm.log_n0 = -1.5 * std::log(2.0 * kPi) - 0.5 * std::log(s.cov.determinant());
                pm.log_u = -std::log(kPi * s.extent * s.extent * 2.0 * kPi);
            }
            // WALL: the door's wall on each side, parallel and one wall thickness apart.
            std::optional<Edge> ea, eb;
            if (m.walls and s.aperture_a.has_value() and s.aperture_b.has_value())
            {
                ea = wall_edge(rooms[s.a].outline, s.aperture_a.value());
                eb = wall_edge(rooms[s.b].outline, s.aperture_b.value());
            }
            const bool wall = ea.has_value() and eb.has_value();
            double sig_par = 1, sig_sep = 1;
            if (wall)
            {
                const double so2 = m.sigma_outline * m.sigma_outline;
                sig_par = std::sqrt(2.0 * so2 * (1.0 / (ea->L * ea->L) + 1.0 / (eb->L * eb->L)));
                sig_sep = std::sqrt(m.wall_t_sigma * m.wall_t_sigma + so2);
                if (pm.mixture)
                {
                    pm.log_n0 += -std::log(2.0 * kPi) - std::log(sig_par * sig_sep);
                    pm.log_u += -std::log(kPi) - std::log(2.0 * s.extent);
                }
            }
            passage_model.push_back(pm);
            // Residual = the whitened differences rescaled so that ½|r|² is the mixture energy above its minimum: the
            // direction is the Gaussian's, the length saturates once the outlier component explains the passage better.
            const double e0 = passage_energy(pm, 0.0);
            const double slope0 = pm.mixture ? 2.0 * (passage_energy(pm, 1e-9) - e0) / 1e-9 : 1.0;   // = P(inlier | q=0)
            const int rows = wall ? 5 : 3;
            // The UNSCALED whitened residual (seam, then wall): what the mixture scores, and what the report reads back.
            const auto raw = [a = s.a, b = s.b, z = s.z, W, wall, ea, eb, sig_par, sig_sep, t = m.wall_t_mean]
                             (const std::vector<Vec3> &x, double *out)
            {
                Vec3 d = between(x[a], x[b]) - z;
                d.z() = wrap(d.z());
                Eigen::Map<Vec3>{out} = W * d;
                if (not wall) return;
                const Vec2 da = ea->q - ea->p, db = eb->q - eb->p;
                // Parallel as LINES: the doubled angle makes the two walls' opposite orientations agree.
                const double aa = x[a].z() + std::atan2(da.y(), da.x()), ab = x[b].z() + std::atan2(db.y(), db.x());
                out[3] = 0.5 * wrap(2.0 * (aa - ab)) / sig_par;
                // Separation along each wall's outward normal, read both ways and averaged.
                const Vec2 pa = apply(x[a], ea->p), pb = apply(x[b], eb->p);
                const Vec2 ma = apply(x[a], 0.5 * (ea->p + ea->q)), mb = apply(x[b], 0.5 * (eb->p + eb->q));
                const Vec2 na = rot(x[a].z()) * ea->n, nb = rot(x[b].z()) * eb->n;
                out[4] = (0.5 * ((mb - pa).dot(na) + (ma - pb).dot(nb)) - t) / sig_sep;
            };
            passage_q.push_back([raw, rows](const std::vector<Vec3> &x)
            {
                std::array<double, 5> v{};
                raw(x, v.data());
                return Eigen::Map<Eigen::VectorXd>(v.data(), rows).squaredNorm();
            });
            factors.push_back({Factor::kPassage, {static_cast<int>(s.a), static_cast<int>(s.b)}, false, rows,
                [raw, pm, e0, slope0, passage_energy, rows](const std::vector<Vec3> &x, double, double *out)
                {
                    raw(x, out);
                    Eigen::Map<Eigen::VectorXd> v(out, rows);
                    const double q = v.squaredNorm();
                    v *= q > 1e-12 ? std::sqrt(std::max(0.0, 2.0 * (passage_energy(pm, q) - e0)) / q) : std::sqrt(slope0);
                }, 3});

            // ADJACENCY: b's origin within b's reach of the doorway in a (a's origin when the aperture is unknown).
            const double sig_adj = reach[s.b] > 0 ? reach[s.b] : s.extent;
            if (m.adjacency and sig_adj > 0)
            {
                const Vec2 c = s.aperture_a.has_value() ? Vec2(0.5 * ((*s.aperture_a)[0] + (*s.aperture_a)[1])) : Vec2::Zero();
                factors.push_back({Factor::kAdjacency, {static_cast<int>(s.a), static_cast<int>(s.b)}, false, 2,
                    [a = s.a, b = s.b, c, sig_adj](const std::vector<Vec3> &x, double, double *out)
                    {
                        Eigen::Map<Vec2>{out} = (between(x[a], x[b]).head<2>() - c) / sig_adj;
                    }});
            }
        }

    if (use_psi)
        for (std::size_t i = 0; i < n; ++i)
        {
            if (kappa[i] <= 0) continue;
            const double k = kappa[i], phi = axis[i]->phi;
            // Residual ±√(2(E(u) − E(0))): exact energy, smooth through u = 0.
            const auto energy = [k, manhattan_energy](double u) { return manhattan_energy(k, u); };
            const double e0 = energy(0.0);
            factors.push_back({Factor::kManhattan, {static_cast<int>(i), -1}, true, 1,
                [i, phi, energy, e0](const std::vector<Vec3> &x, double ps, double *out)
                {
                    const double u = wrap(4.0 * (x[i].z() + phi - ps));
                    out[0] = std::copysign(std::sqrt(std::max(0.0, 2.0 * (energy(u) - e0))), u);
                }});
        }

    // Outlines sampled for OVERLAP; the factors read these by reference, so they live as long as the solve.
    std::vector<std::vector<Sample>> samples(n);
    std::vector<Vec2> centre(n, Vec2::Zero());
    std::vector<double> radius(n, 0.0);
    if (m.overlap)
    {
        const double sigma = std::numbers::sqrt2 * m.sigma_outline;   // both outlines are uncertain
        for (std::size_t i = 0; i < n; ++i)
        {
            if (rooms[i].outline.size() < 3) continue;
            samples[i] = sample_outline(rooms[i].outline, m);
            for (const auto &p : rooms[i].outline) centre[i] += p;
            centre[i] /= static_cast<double>(rooms[i].outline.size());
            for (const auto &p : rooms[i].outline) radius[i] = std::max(radius[i], (p - centre[i]).norm());
        }
        for (std::size_t a = 0; a < n; ++a)
            for (std::size_t b = a + 1; b < n; ++b)
            {
                if (samples[a].empty() or samples[b].empty()) continue;
                const int rows = static_cast<int>(samples[a].size() + samples[b].size());
                factors.push_back({Factor::kOverlap, {static_cast<int>(a), static_cast<int>(b)}, false, rows,
                    [&, a, b, rows, sigma](const std::vector<Vec3> &x, double, double *out)
                    {
                        std::fill_n(out, rows, 0.0);
                        // Rooms whose bounding circles are many σ apart cannot overlap: every term is ~0, skip the
                        // distance work (a cost shortcut — the residual it stands in for is below 1e-4).
                        const Vec2 ca = apply(x[a], centre[a]), cb = apply(x[b], centre[b]);
                        if ((ca - cb).norm() > radius[a] + radius[b] + 8.0 * sigma) return;
                        int r = 0;
                        const auto side = [&](std::size_t into, std::size_t from)
                        {
                            const Vec3 T = between(x[into], x[from]);   // T(into <- from)
                            for (const auto &s : samples[from])
                            {
                                const double d = signed_distance(rooms[into].outline, apply(T, s.p));
                                out[r++] = std::sqrt(std::max(0.0, -2.0 * s.w * log_ndtr(d / sigma)));
                            }
                        };
                        side(a, b);
                        side(b, a);
                    }});
            }
    }

    int nrow = 0;
    for (const auto &f : factors) nrow += f.rows;

    // ── Levenberg–Marquardt on ½Σr² ───────────────────────────────────────────────────────────
    const auto residuals = [&](const std::vector<Vec3> &x, double ps, Eigen::VectorXd &r, Terms *terms)
    {
        r.resize(nrow);
        int at = 0;
        for (const auto &f : factors)
        {
            f.eval(x, ps, r.data() + at);
            if (terms)
            {
                const double e = 0.5 * r.segment(at, f.rows).squaredNorm();
                switch (f.kind)
                {
                    case Factor::kPassage:
                    {
                        const double es = 0.5 * r.segment(at, f.seam_rows).squaredNorm();
                        terms->seam += es;
                        terms->wall += e - es;
                        break;
                    }
                    case Factor::kManhattan: terms->manhattan += e; break;
                    case Factor::kOverlap: terms->overlap += e; break;
                    case Factor::kAdjacency: terms->adjacency += e; break;
                }
            }
            at += f.rows;
        }
    };
    // Numeric Jacobian, per factor (each touches ≤ 2 rooms and ψ). ONE-SIDED, the smaller of forward and backward per
    // entry: residuals carrying an angle (seam yaw, the walls' doubled angle, the Manhattan 4θ) flip sign where the
    // angle wraps at ±π while their energy ½r² stays continuous. A central difference straddling that flip reads it
    // as a slope of ~1/h (it put a 7.8e12 eigenvalue in a real solve, i.e. a 0.00° σ); only one of the two one-sided
    // differences can straddle it, and away from a flip they agree to O(h).
    const auto jacobian = [&](const std::vector<Vec3> &x, double ps, Eigen::MatrixXd &J)
    {
        J.setZero(nrow, ncol);
        constexpr double h = 1e-6;
        int at = 0;
        Eigen::VectorXd r0, rp, rm;
        const auto column = [&](int c, int rows)
        {
            for (int k = 0; k < rows; ++k)
            {
                const double df = (rp(k) - r0(k)) / h, db = (r0(k) - rm(k)) / h;
                J(at + k, c) = std::abs(df) <= std::abs(db) ? df : db;
            }
        };
        for (const auto &f : factors)
        {
            r0.resize(f.rows);
            rp.resize(f.rows);
            rm.resize(f.rows);
            f.eval(x, ps, r0.data());
            for (const int ri : f.room)
            {
                if (ri < 0 or col[ri] < 0) continue;
                for (int k = 0; k < 3; ++k)
                {
                    auto xp = x, xm = x;
                    xp[ri](k) += h;
                    xm[ri](k) -= h;
                    f.eval(xp, ps, rp.data());
                    f.eval(xm, ps, rm.data());
                    column(col[ri] + k, f.rows);
                }
            }
            if (f.psi and psi_col >= 0)
            {
                f.eval(x, ps + h, rp.data());
                f.eval(x, ps - h, rm.data());
                column(psi_col, f.rows);
            }
            at += f.rows;
        }
    };
    const auto step = [&](const std::vector<Vec3> &x, double ps, const Eigen::VectorXd &dx)
    {
        auto y = x;
        for (std::size_t i = 0; i < n; ++i)
            if (col[i] >= 0)
            {
                y[i] += dx.segment<3>(col[i]);
                y[i].z() = wrap(y[i].z());
            }
        return std::pair{y, psi_col >= 0 ? ps + dx(psi_col) : ps};
    };

    auto x = res.pose;
    Eigen::VectorXd r;
    residuals(x, psi, r, &res.before);
    double F = 0.5 * r.squaredNorm();
    Eigen::MatrixXd J;
    double lambda = 1e-3;
    if (ncol > 0)
        for (res.iterations = 0; res.iterations < m.max_iterations; ++res.iterations)
        {
            jacobian(x, psi, J);
            const Eigen::MatrixXd H = J.transpose() * J;
            const Eigen::VectorXd g = J.transpose() * r;
            bool accepted = false;
            double F_new = F;
            while (lambda < 1e12)
            {
                Eigen::MatrixXd A = H;
                A.diagonal() += lambda * (H.diagonal().array() + 1e-9).matrix();
                const Eigen::VectorXd dx = A.ldlt().solve(-g);
                auto [y, ps] = step(x, psi, dx);
                Eigen::VectorXd ry;
                residuals(y, ps, ry, nullptr);
                F_new = 0.5 * ry.squaredNorm();
                if (F_new < F)
                {
                    x = std::move(y);
                    psi = ps;
                    r = std::move(ry);
                    lambda = std::max(lambda / 10.0, 1e-12);
                    accepted = true;
                    break;
                }
                lambda *= 10.0;
            }
            // Stop when no step lowers F, or when it lowers it by less than 1e-6 nats: F is a negative log-probability,
            // and a millionth of a nat changes no posterior anyone could act on (a relative 1e-10 let one start creep
            // for 100 iterations at 1e-6 per step).
            if (not accepted or F - F_new <= 1e-6)
            {
                if (accepted) F = F_new;
                res.converged = true;
                break;
            }
            if (m.verbose)
                std::fprintf(stderr, "    it %3d  F %.6f -> %.6f  lambda %.1e\n", res.iterations, F, F_new, lambda);
            F = F_new;
        }
    else
        res.converged = true;

    residuals(x, psi, r, &res.after);
    res.pose = x;

    // ── Laplace covariance: (JᵀJ)⁻¹ at the optimum, gauge excluded (the fixed rooms carry none) ──
    if (ncol > 0)
    {
        jacobian(x, psi, J);
        const Eigen::MatrixXd H = J.transpose() * J;
        if (m.verbose)
        {
            int at2 = 0;
            for (const auto &f : factors)
            {
                const double nrm = J.middleRows(at2, f.rows).squaredNorm();
                if (nrm > 1e6)
                    std::fprintf(stderr, "    factor kind %d rooms (%d,%d) rows %d |J|^2 %.3g\n", static_cast<int>(f.kind),
                                 f.room[0], f.room[1], f.rows, nrm);
                at2 += f.rows;
            }
            const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
            std::fprintf(stderr, "    H eigenvalues:");
            for (int i = 0; i < es.eigenvalues().size(); ++i) std::fprintf(stderr, " %.3g", es.eigenvalues()(i));
            std::fprintf(stderr, "\n");
        }
        const Eigen::MatrixXd S = H.completeOrthogonalDecomposition().pseudoInverse();
        for (std::size_t i = 0; i < n; ++i)
            if (col[i] >= 0) res.cov[i] = S.block<3, 3>(col[i], col[i]);
    }

    // ── Reports ───────────────────────────────────────────────────────────────────────────────
    int at = 0;
    std::size_t seam_k = 0;
    for (const auto &f : factors)
    {
        if (f.kind == Factor::kPassage)
        {
            const auto &s = seams[seam_k];
            Vec3 d = between(x[s.a], x[s.b]) - s.z;
            d.z() = wrap(d.z());
            res.seam_d2[seam_k] = d.dot(s.cov.ldlt().solve(d));
            if (const auto &pm = passage_model[seam_k]; pm.mixture)
            {
                const double q = passage_q[seam_k](x);
                const double l_in = std::log(m.p_seam_inlier) + pm.log_n0 - 0.5 * q;
                const double l_out = std::log(1.0 - m.p_seam_inlier) + pm.log_u;
                res.seam_p_inlier[seam_k] = 1.0 / (1.0 + std::exp(l_out - l_in));
            }
            ++seam_k;
        }
        if (f.kind == Factor::kManhattan)
        {
            const auto i = static_cast<std::size_t>(f.room[0]);
            const double u = wrap(4.0 * (x[i].z() + axis[i]->phi - psi));
            const double l_in = std::log(m.p_manhattan) + kappa[i] * (std::cos(u) - 1.0) - log_i0e(kappa[i]);
            const double l_out = std::log(1.0 - m.p_manhattan);   // both share the 1/2π
            res.p_manhattan[i] = 1.0 / (1.0 + std::exp(l_out - l_in));
        }
        at += f.rows;
    }
    if (use_psi) res.building_axis = 0.25 * wrap(4.0 * psi);
    return res;
}

}   // namespace

Result optimise(const std::vector<Room> &rooms, const std::vector<Seam> &seams, const Model &m)
{
    // ── Starts: the given poses, and one spanning tree per seam with THAT seam walked first. With an outlier among the
    //    seams the energy has one mode per way of explaining it, and LM only finds the one it starts in; the tree that
    //    does not use the bad seam starts in the right one. The lowest free energy wins — model selection among
    //    modes, not a threshold. ─────────────────────────────────────────────────────────────────────────────
    std::vector<std::vector<Vec3>> starts;
    starts.emplace_back();
    for (const auto &r : rooms) starts.back().push_back(r.init);
    const auto gauge = std::ranges::find_if(rooms, &Room::fixed);
    const std::size_t root = gauge != rooms.end() ? static_cast<std::size_t>(std::distance(rooms.begin(), gauge)) : 0;
    for (std::size_t first = 0; first < seams.size() and not rooms.empty(); ++first)
    {
        std::vector<std::size_t> order{first};
        for (std::size_t k = 0; k < seams.size(); ++k)
            if (k != first) order.push_back(k);
        std::vector<Vec3> x = starts.front();
        std::vector<bool> placed(rooms.size(), false);
        placed[root] = true;
        for (std::size_t i = 0; i < rooms.size(); ++i) placed[i] = placed[i] or rooms[i].fixed;
        for (bool grew = true; grew;)
        {
            grew = false;
            for (const auto k : order)
            {
                const auto &s = seams[k];
                if (placed[s.a] == placed[s.b]) continue;
                const Vec3 inv = [&] { const Vec2 t = -rot(-s.z.z()) * s.z.head<2>(); return Vec3(t.x(), t.y(), -s.z.z()); }();
                const auto compose = [](const Vec3 &p, const Vec3 &q)
                { const Vec2 t = p.head<2>() + rot(p.z()) * q.head<2>(); return Vec3(t.x(), t.y(), wrap(p.z() + q.z())); };
                if (placed[s.a]) x[s.b] = compose(x[s.a], s.z);
                else x[s.a] = compose(x[s.b], inv);
                placed[s.a] = placed[s.b] = true;
                grew = true;
                break;   // re-scan from the front: `first` keeps priority wherever it becomes reachable
            }
        }
        starts.push_back(std::move(x));
    }

    std::optional<Result> best;
    int best_start = 0;
    for (const auto &[i, x] : std::views::enumerate(starts))
    {
        auto rs = rooms;
        for (std::size_t r = 0; r < rs.size(); ++r) rs[r].init = x[r];
        auto res = solve_from(rs, seams, m);
        if (not best.has_value() or res.after.total() < best->after.total())
        {
            best = std::move(res);
            best_start = static_cast<int>(i);
        }
    }
    best->starts = static_cast<int>(starts.size());
    best->winning_start = best_start;
    return best.value();
}

}   // namespace ltsm::layout
