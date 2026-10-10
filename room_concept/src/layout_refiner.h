#pragma once
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
// LAYOUT REFINEMENT AFTER PROMOTION (2026-10-10) — model comparison on the shadow-SLAM evidence.
//
// After the start-up proto is promoted the room is GIVEN and the pose is solved against its SDF. Plain SLAM keeps
// running as a SHADOW on its own map; that map is the model-independent evidence this class scores layouts against:
//   evidence   one observation per 5 cm cell, never per return. surface = a cell where beams STOPPED at least as often
//              as they PASSED (hits >= passes, >= 2 hits: the MAP decision of an occupancy log-odds with symmetric
//              confusion); free = passes > hits. (The bare "hits >= 2" rule let ceiling arcs and furniture ghosts
//              through: 2512 of 3692 such cells on the live apartment were refuted by their own passes.)
//   likelihood surface cell at distance d from the polygon: (1-eps) N(d; 0, sigma) + eps / R_eff, R_eff = A / P
//              (the distance-to-boundary density of a uniform point in the room — no range constant);
//              free cell d_out beyond the boundary: (1-eps) Phi(-d_out / sigma) + eps  (a beam through a modelled
//              wall IS an outlier return: one eps, not two). sigma and eps are fitted by EM under the CURRENT polygon;
//              each cell is weighted w = cell / ell, ell the residual correlation length along the walls (fitted
//              from the residual autocorrelation) — adjacent cells of one wall are not independent observations.
//   criterion  adopt edit e iff   dNLL + kappa * d|V| + ln(n_edges) < 0,   kappa = ln(L / sigma):
//              two-part MDL — the data under the new polygon, the polygon as a vertex list (one free coordinate per
//              Manhattan vertex, resolved to sigma over the span L), and the address of the edited feature. Nothing
//              here is a threshold: a vertex that explains less than kappa never appears, a move that explains less
//              than its own address is never uploaded, and the margin grows with evidence in nats on its own.
//   candidates generated from the evidence, never enumerated: per edge the weighted-median offset of the cells it
//              owns (slide = vertex surgery, strip = union/difference of the strip), a change-point jog along it,
//              and the K largest rectangles of the two contested masks — swept-free OUTSIDE the polygon (extend) and
//              never-swept INSIDE it (notch: a face and whatever stands behind it). K is a CPU budget, flagged.
//   weights    each cell's term is multiplied by its occupancy posterior for its class (FreeSpacePolygon::evidence_cells
//              `conf`): a cell seen twice carries little, one confirmed forty times carries ~1, so an edit's margin
//              GROWS with repeated looks — that is the hysteresis, with no counter and no delay constant.
//   surface position = the MEAN of the returns that ended in the cell, not the cell centre (a centre is biased by up
//              to cell/2 from the face; offline that bias moved every refined wall and cost 2.3 cm of pose on walhit).
// Offline (scratchpad/refine/refine2.py + refine3.py, same algorithm; logs run3_*.log, seq_*.log): IoU .893→.95
// (sdfloc, 28 verts: the promoted layout had cut the kitchen wing 60 cm short), .932→.96 (slam1), .954→.97 (walhit).
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
#include "free_space_polygon.h"
#include <Eigen/Dense>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/multi_polygon.hpp>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

namespace rc::refine
{
struct Params
{
    float cell       = 0.05f;   ///< m — the evidence raster (= the SLAM map's cell)
    float reach      = 0.40f;   ///< m — how far an edge looks for its wall (= manhattan_layout's snap reach)
    int   k_rect     = 8;       ///< ★CPU budget, flagged: largest rectangles tried per contested mask per round
    int   max_rounds = 12;      ///< ★CPU budget, flagged: greedy rounds per proposal
    float sigma0     = 0.03f;   ///< m — EM start (scan_slam sigma_lidar); the fitted value is what is used
};

struct Edit
{
    std::string kind;   ///< slide | strip | jog | notch | extend
    int   edge = -1;
    float value = 0.f;  ///< offset (m) for slide/strip/jog; cell count for notch/extend
    float gain = 0.f;   ///< dNLL + kappa d|V| + ln n  (negative = adopted)
};

struct Result
{
    std::vector<Eigen::Vector2f> polygon;   ///< map frame, CCW
    std::vector<Edit> edits;
    float sigma = 0.f, eps = 0.f, ell = 0.f, kappa = 0.f, w = 1.f;
    float nll_before = 0.f, nll_after = 0.f;
    /// Contested cells (room axes → map frame already applied): the cells whose explanation would change under the
    /// best REJECTED candidates — where a look would decide. The epistemic planner weights standpoints by how many
    /// of these are visible (the "testing behaviour").
    std::vector<Eigen::Vector2f> contested;
};

class LayoutRefiner
{
public:
    explicit LayoutRefiner(Params p = {}) : p_(p) {}

    /// surf/free: evidence cells in the MAP frame (FreeSpacePolygon::evidence_cells), with their class posteriors
    /// (empty = all 1). poly: the given room, map frame. The result is kept (last()) for the explorer's gain.
    [[nodiscard]] Result propose(const std::vector<Eigen::Vector2f>& poly,
                                 const std::vector<Eigen::Vector2f>& surf,
                                 const std::vector<Eigen::Vector2f>& free,
                                 const std::vector<float>& conf_surf = {},
                                 const std::vector<float>& conf_free = {})
    {
        last_ = propose_impl(poly, surf, free, conf_surf, conf_free);
        return last_;
    }
    [[nodiscard]] const Result& last() const { return last_; }
    [[nodiscard]] const Params& params() const { return p_; }

private:
    Params p_;
    Result last_;

    [[nodiscard]] Result propose_impl(const std::vector<Eigen::Vector2f>& poly,
                                      const std::vector<Eigen::Vector2f>& surf,
                                      const std::vector<Eigen::Vector2f>& free,
                                      const std::vector<float>& conf_surf,
                                      const std::vector<float>& conf_free) const
    {
        Result R;
        if (poly.size() < 4 or surf.size() < 20) { R.polygon = poly; return R; }
        const float th = dominant_theta(poly);
        auto V = rot(poly, -th); if (freespace::signed_area(V) < 0.f) std::ranges::reverse(V);
        V = snap_axes(V);
        const auto S = rot(surf, -th), Fr = rot(free, -th);
        Scorer sc(S, Fr, p_, conf_surf.size() == S.size() ? conf_surf : std::vector<float>(S.size(), 1.f),
                  conf_free.size() == Fr.size() ? conf_free : std::vector<float>(Fr.size(), 1.f));
        sc.fit(V);
        R.sigma = sc.sigma; R.eps = sc.eps; R.ell = sc.ell; R.kappa = sc.kappa; R.w = sc.w;
        R.nll_before = sc.nll(V);
        float F0 = R.nll_before;
        std::vector<Eigen::Vector2f> best_rejected_cells;
        for (int round = 0; round < p_.max_rounds; ++round)
        {
            auto cands = candidates(V, sc);
            if (cands.empty()) break;
            std::optional<std::size_t> best; float best_gain = 0.f;
            std::vector<float> gains(cands.size());
            for (std::size_t i = 0; i < cands.size(); ++i)
            {
                const auto& [e, Vq] = cands[i];
                gains[i] = sc.nll(Vq) - F0 + sc.kappa * float(int(Vq.size()) - int(V.size())) + std::log(float(V.size()));
                if (gains[i] < best_gain) { best_gain = gains[i]; best = i; }
            }
            if (not best)
            {   // nothing pays: the contested cells are those of the near-miss candidates (gain within one vertex)
                for (std::size_t i = 0; i < cands.size(); ++i)
                    if (gains[i] < sc.kappa) sc.changed_cells(V, cands[i].second, best_rejected_cells);
                break;
            }
            auto [e, Vq] = cands[*best];
            e.gain = best_gain; R.edits.push_back(e);
            V = std::move(Vq); F0 = sc.nll(V);
        }
        R.nll_after = F0;
        R.polygon = rot(V, th);
        R.contested = rot(best_rejected_cells, th);
        return R;
    }

    // ── geometry ───────────────────────────────────────────────────────────────────────────────────────────
    struct Edge { int k; bool h; float v, lo, hi, sgn; };   // horizontal?, coordinate, extent, outward sign

    static std::vector<Eigen::Vector2f> rot(const std::vector<Eigen::Vector2f>& P, float th)
    {
        const float c = std::cos(th), s = std::sin(th);
        std::vector<Eigen::Vector2f> out; out.reserve(P.size());
        for (const auto& q : P) out.emplace_back(c * q.x() - s * q.y(), s * q.x() + c * q.y());
        return out;
    }
    static float dominant_theta(const std::vector<Eigen::Vector2f>& V)
    {
        double zr = 0, zi = 0;
        for (std::size_t k = 0; k < V.size(); ++k)
        {
            const Eigen::Vector2f d = V[(k + 1) % V.size()] - V[k]; const double L = d.norm();
            if (L <= 0) continue;
            const double a = 4.0 * std::atan2(d.y(), d.x()); zr += L * std::cos(a); zi += L * std::sin(a);
        }
        return static_cast<float>(0.25 * std::atan2(zi, zr));
    }
    /// Make an almost-Manhattan ring exactly axis-aligned: each edge takes its class, consecutive vertices share it.
    static std::vector<Eigen::Vector2f> snap_axes(std::vector<Eigen::Vector2f> V)
    {
        const std::size_t n = V.size();
        for (std::size_t k = 0; k < n; ++k)
        {
            auto& a = V[k]; auto& b = V[(k + 1) % n]; const Eigen::Vector2f d = b - a;
            if (std::abs(d.x()) >= std::abs(d.y())) { const float m = 0.5f * (a.y() + b.y()); a.y() = b.y() = m; }
            else                                     { const float m = 0.5f * (a.x() + b.x()); a.x() = b.x() = m; }
        }
        return V;
    }
    static std::vector<Edge> edges(const std::vector<Eigen::Vector2f>& V)
    {
        std::vector<Edge> E; const int n = static_cast<int>(V.size());
        for (int k = 0; k < n; ++k)
        {
            const auto& a = V[k]; const auto& b = V[(k + 1) % n]; const Eigen::Vector2f d = b - a;
            const bool h = std::abs(d.x()) >= std::abs(d.y());
            // CCW ring: outward normal = (dy, -dx)
            const float sgn = h ? (-d.x() >= 0.f ? 1.f : -1.f) : (d.y() >= 0.f ? 1.f : -1.f);
            if (h) E.push_back({k, true, 0.5f * (a.y() + b.y()), std::min(a.x(), b.x()), std::max(a.x(), b.x()), sgn});
            else   E.push_back({k, false, 0.5f * (a.x() + b.x()), std::min(a.y(), b.y()), std::max(a.y(), b.y()), sgn});
        }
        return E;
    }
    struct Near { float d; int edge; bool along; float across; };   // nearest edge of a point, signed outward offset
    static Near nearest(const Eigen::Vector2f& q, const std::vector<Edge>& E)
    {
        Near best{std::numeric_limits<float>::infinity(), -1, false, 0.f};
        for (std::size_t m = 0; m < E.size(); ++m)
        {
            const auto& e = E[m];
            const float along = e.h ? q.x() : q.y(), across = e.h ? q.y() : q.x();
            const float da = std::max({lo_gap(e.lo, along), along - e.hi, 0.f});
            const float d = std::hypot(da, across - e.v);
            if (d < best.d) best = {d, static_cast<int>(m), da == 0.f, e.sgn * (across - e.v)};
        }
        return best;
    }
    static float lo_gap(float lo, float x) { return lo - x; }

    // ── the model ──────────────────────────────────────────────────────────────────────────────────────────
    struct Scorer
    {
        const std::vector<Eigen::Vector2f>& S; const std::vector<Eigen::Vector2f>& Fr; Params p;
        std::vector<float> CS, CF;   ///< per-cell class posteriors (the accumulated-evidence weights)
        float sigma, eps = 0.3f, ell, w = 1.f, kappa = 0.f, L = 1.f;
        Scorer(const std::vector<Eigen::Vector2f>& s, const std::vector<Eigen::Vector2f>& f, Params pp,
               std::vector<float> cs, std::vector<float> cf)
            : S(s), Fr(f), p(pp), CS(std::move(cs)), CF(std::move(cf)), sigma(pp.sigma0), ell(pp.cell) {}

        static float log_norm(float d, float sig) { return -0.5f * (d / sig) * (d / sig) - std::log(sig * 2.5066283f); }
        static float log_phi_tail(float x)   // log Phi(-x), x >= 0
        { return x < 5.f ? std::log(0.5f * std::erfc(x * 0.70710678f)) : -0.5f * x * x - std::log(x * 2.5066283f); }
        static float lse(float a, float b) { const float m = std::max(a, b); return m + std::log(std::exp(a - m) + std::exp(b - m)); }

        void fit(const std::vector<Eigen::Vector2f>& V)
        {
            const auto E = edges(V); const auto poly_ap = area_perim(V);
            const float lo = -std::log(poly_ap.first / poly_ap.second);
            std::vector<Near> nr(S.size()); for (std::size_t i = 0; i < S.size(); ++i) nr[i] = nearest(S[i], E);
            float sig = p.sigma0, e = 0.3f;
            for (int it = 0; it < 50; ++it)
            {
                double sr = 0, sd2 = 0;
                for (const auto& n : nr)
                {
                    const float a = std::log(1 - e) + log_norm(n.d, sig), b = std::log(e) + lo;
                    const float r = 1.f / (1.f + std::exp(std::clamp(b - a, -50.f, 50.f)));   // inlier responsibility
                    sr += r; sd2 += r * n.d * n.d;
                }
                e = static_cast<float>(1.0 - sr / double(nr.size())); sig = static_cast<float>(std::sqrt(sd2 / std::max(sr, 1e-9)));
            }
            sigma = sig; eps = std::clamp(e, 1e-3f, 0.95f);
            // residual correlation length along the walls: mean offset per along-cell bin, lag autocorrelation
            std::vector<double> acs(12, 0.0); std::vector<int> acn(12, 0);
            for (std::size_t m = 0; m < E.size(); ++m)
            {
                const auto& ed = E[m]; const int nb = static_cast<int>((ed.hi - ed.lo) / p.cell) + 1;
                if (nb < 30) continue;
                std::vector<double> sum(nb, 0.0); std::vector<int> cnt(nb, 0);
                for (std::size_t i = 0; i < S.size(); ++i)
                    if (nr[i].edge == static_cast<int>(m) and nr[i].along and nr[i].d < 3 * sigma)
                    {
                        const float along = ed.h ? S[i].x() : S[i].y();
                        const int b = std::clamp(static_cast<int>((along - ed.lo) / p.cell), 0, nb - 1);
                        sum[b] += nr[i].across; ++cnt[b];
                    }
                std::vector<double> x(nb, 0.0); std::vector<bool> ok(nb, false); double mean = 0; int nok = 0;
                for (int b = 0; b < nb; ++b) if (cnt[b]) { x[b] = sum[b] / cnt[b]; ok[b] = true; mean += x[b]; ++nok; }
                if (nok < 30) continue;
                mean /= nok; double var = 0;
                for (int b = 0; b < nb; ++b) if (ok[b]) { x[b] -= mean; var += x[b] * x[b]; }
                var /= nok; if (var <= 0) continue;
                for (int lag = 0; lag < 12; ++lag)
                {
                    double acc = 0; int nn = 0;
                    for (int b = 0; b + lag < nb; ++b) if (ok[b] and ok[b + lag]) { acc += x[b] * x[b + lag]; ++nn; }
                    if (nn >= 10) { acs[lag] += acc / nn / var; ++acn[lag]; }
                }
            }
            double tau = 0; for (int lag = 1; lag < 12; ++lag) if (acn[lag]) tau += std::max(0.0, acs[lag] / acn[lag]);
            ell = p.cell * static_cast<float>(1 + 2 * tau); w = p.cell / ell;
            float x0 = 1e9f, x1 = -1e9f, y0 = 1e9f, y1 = -1e9f;
            for (const auto& v : V) { x0 = std::min(x0, v.x()); x1 = std::max(x1, v.x()); y0 = std::min(y0, v.y()); y1 = std::max(y1, v.y()); }
            L = std::max(x1 - x0, y1 - y0); kappa = std::log(L / sigma);
        }
        static std::pair<float, float> area_perim(const std::vector<Eigen::Vector2f>& V)
        {
            float per = 0.f; for (std::size_t k = 0; k < V.size(); ++k) per += (V[(k + 1) % V.size()] - V[k]).norm();
            return {std::abs(freespace::signed_area(V)), per};
        }
        /// -log p(evidence | V), weighted. (No complexity term: that is charged on the EDIT in propose().)
        float nll(const std::vector<Eigen::Vector2f>& V) const
        {
            const auto E = edges(V); const auto ap = area_perim(V); const float lo = -std::log(ap.first / ap.second);
            double acc = 0;
            for (std::size_t i = 0; i < S.size(); ++i)
            {
                const auto n = nearest(S[i], E);
                acc -= CS[i] * lse(std::log(1 - eps) + log_norm(n.d, sigma), std::log(eps) + lo);
            }
            for (std::size_t i = 0; i < Fr.size(); ++i)
            {
                if (freespace::point_in_polygon(Fr[i], V)) continue;
                const auto n = nearest(Fr[i], E);
                acc -= CF[i] * lse(std::log(1 - eps) + log_phi_tail(n.d / sigma), std::log(eps));
            }
            return static_cast<float>(w * acc);
        }
        /// Cells whose status (explained / inside / outside) differs between V and Vq — what a look would decide.
        void changed_cells(const std::vector<Eigen::Vector2f>& V, const std::vector<Eigen::Vector2f>& Vq,
                           std::vector<Eigen::Vector2f>& out) const
        {
            const auto E = edges(V), Eq = edges(Vq);
            for (const auto& q : S)
                if ((nearest(q, E).d < 2 * sigma) != (nearest(q, Eq).d < 2 * sigma)) out.push_back(q);
            for (const auto& q : Fr)
                if (freespace::point_in_polygon(q, V) != freespace::point_in_polygon(q, Vq)) out.push_back(q);
        }
    };

    // ── raster: only for the contested masks (which cells are swept-free outside / never-swept inside) ────────────
    struct Raster { float u0, v0; int W, H; std::vector<std::uint8_t> m; };
    Raster rasterize(const std::vector<Eigen::Vector2f>& V, float pad) const
    {
        float x0 = 1e9f, x1 = -1e9f, y0 = 1e9f, y1 = -1e9f;
        for (const auto& v : V) { x0 = std::min(x0, v.x()); x1 = std::max(x1, v.x()); y0 = std::min(y0, v.y()); y1 = std::max(y1, v.y()); }
        Raster r; r.u0 = x0 - pad; r.v0 = y0 - pad;
        r.W = static_cast<int>((x1 - x0 + 2 * pad) / p_.cell) + 2; r.H = static_cast<int>((y1 - y0 + 2 * pad) / p_.cell) + 2;
        r.m.assign(static_cast<std::size_t>(r.W) * r.H, 0);
        for (int j = 0; j < r.H; ++j) for (int i = 0; i < r.W; ++i)
            r.m[static_cast<std::size_t>(j) * r.W + i] =
                freespace::point_in_polygon(Eigen::Vector2f(r.u0 + (i + 0.5f) * p_.cell, r.v0 + (j + 0.5f) * p_.cell), V) ? 1 : 0;
        return r;
    }
    // ── EXACT box edits (2026-10-10). A candidate is the polygon ∪/∖ ONE axis-aligned box, computed on the vector
    // polygon. It used to go polygon → 5 cm raster → paint → re-trace, which edited the WHOLE ring as a side effect:
    // every vertex snapped to the grid and any feature thinner than a cell vanished. Live (2026-10-10 20:37) the promoted
    // layout's fin (a wall stub drawn 2 mm thick) was erased by a "strip" on an unrelated edge, and the MDL term paid
    // ~4 kappa for the dropped vertices — a deletion nobody proposed and nobody scored. Now an edit changes the room
    // only inside its own box; removing a fin needs an edit that is about the fin and pays for it in the data.
    using BPt   = boost::geometry::model::d2::point_xy<double>;
    using BPoly = boost::geometry::model::polygon<BPt, false>;   // CCW, closed
    static std::optional<std::vector<Eigen::Vector2f>> box_edit(const std::vector<Eigen::Vector2f>& V, float a0, float b0, float a1, float b1, bool add)
    {
        namespace bg = boost::geometry;
        if (not (a1 > a0 and b1 > b0)) return std::nullopt;
        BPoly P, B;
        for (const auto& v : V) bg::append(P.outer(), BPt(v.x(), v.y()));
        bg::append(P.outer(), BPt(V.front().x(), V.front().y()));
        bg::correct(P);
        for (const auto& [x, y] : std::array<std::array<float, 2>, 5>{{{a0, b0}, {a1, b0}, {a1, b1}, {a0, b1}, {a0, b0}}})
            bg::append(B.outer(), BPt(x, y));
        bg::correct(B);
        bg::model::multi_polygon<BPoly> out;
        if (add) bg::union_(P, B, out); else bg::difference(P, B, out);
        if (out.size() != 1 or not out.front().inners().empty()) return std::nullopt;   // split or holed: not a room
        std::vector<Eigen::Vector2f> ring;
        const auto& o = out.front().outer();
        for (std::size_t k = 0; k + 1 < o.size(); ++k)   // drop the closing point
        {
            const Eigen::Vector2f q(static_cast<float>(o[k].x()), static_cast<float>(o[k].y()));
            if (ring.empty() or (q - ring.back()).squaredNorm() > 0.f) ring.push_back(q);
        }
        while (ring.size() > 1 and (ring.front() - ring.back()).squaredNorm() == 0.f) ring.pop_back();
        ring = freespace::FreeSpacePolygon::merge_collinear(ring);
        if (ring.size() < 4) return std::nullopt;
        if (freespace::signed_area(ring) < 0.f) std::ranges::reverse(ring);
        if (not freespace::is_simple(ring)) return std::nullopt;
        return ring;
    }
    /// A contested rectangle comes from the 5 cm mask and stops a cell short of the walls (cells within sigma of an
    /// edge are excluded by construction). Close that representation gap: a box side within sigma + one cell of a
    /// parallel edge that overlaps it moves onto that edge, so the box meets the room instead of floating beside it.
    static std::array<float, 4> snap_box(std::array<float, 4> b, const std::vector<Edge>& E, float reach)
    {
        auto snap = [&](float& side, bool vertical_side, float lo, float hi)
        {
            float best = reach, to = side;
            for (const auto& e : E)
                if (e.h != vertical_side and e.hi > lo and e.lo < hi)
                    if (const float d = std::abs(e.v - side); d < best) { best = d; to = e.v; }
            side = to;
        };
        snap(b[0], true, b[1], b[3]); snap(b[2], true, b[1], b[3]);
        snap(b[1], false, b[0], b[2]); snap(b[3], false, b[0], b[2]);
        return b;
    }
    /// The box a strip of edge `e` over [a0,a1] moved by dv paints: from the edge's line to dv along its outward normal.
    static std::array<float, 4> strip_box(const Edge& e, float a0, float a1, float dv)
    {
        const float lo = std::min(e.v, e.v + e.sgn * dv), hi = std::max(e.v, e.v + e.sgn * dv);
        return e.h ? std::array<float, 4>{a0, lo, a1, hi} : std::array<float, 4>{lo, a0, hi, a1};
    }

    /// Greedy maximal rectangles of a mask (histogram method), cell units.
    static std::vector<std::array<int, 4>> largest_rectangles(std::vector<std::uint8_t> M, int W, int H, int K)
    {
        std::vector<std::array<int, 4>> out;
        for (int t = 0; t < K; ++t)
        {
            std::vector<int> h(W, 0); int best_area = 0; std::array<int, 4> best{};
            for (int j = 0; j < H; ++j)
            {
                for (int i = 0; i < W; ++i) h[i] = M[static_cast<std::size_t>(j) * W + i] ? h[i] + 1 : 0;
                std::vector<std::pair<int, int>> st;
                for (int i = 0; i <= W; ++i)
                {
                    const int cur = i < W ? h[i] : 0; int start = i;
                    while (not st.empty() and st.back().second >= cur)
                    {
                        const auto [si, sh] = st.back(); st.pop_back();
                        if (const int area = sh * (i - si); area > best_area) { best_area = area; best = {si, j - sh + 1, i, j + 1}; }
                        start = si;
                    }
                    st.emplace_back(start, cur);
                }
            }
            if (best_area < 2) break;
            out.push_back(best);
            for (int j = best[1]; j < best[3]; ++j) for (int i = best[0]; i < best[2]; ++i) M[static_cast<std::size_t>(j) * W + i] = 0;
        }
        return out;
    }

    // ── candidates from the evidence ───────────────────────────────────────────────────────────────────────
    std::vector<std::pair<Edit, std::vector<Eigen::Vector2f>>> candidates(const std::vector<Eigen::Vector2f>& V, const Scorer& sc) const
    {
        std::vector<std::pair<Edit, std::vector<Eigen::Vector2f>>> out;
        const auto E = edges(V); const int n = static_cast<int>(V.size());
        std::vector<Near> nr(sc.S.size()); for (std::size_t i = 0; i < sc.S.size(); ++i) nr[i] = nearest(sc.S[i], E);
        for (int m = 0; m < n; ++m)
        {
            const auto& e = E[m];
            std::vector<std::pair<float, float>> own;   // (along, outward offset)
            for (std::size_t i = 0; i < sc.S.size(); ++i)
                if (nr[i].edge == m and nr[i].along and nr[i].d < p_.reach) own.emplace_back(e.h ? sc.S[i].x() : sc.S[i].y(), nr[i].across);
            if (own.size() < 4) continue;
            std::ranges::sort(own);
            auto median = [](std::vector<float> v) { std::ranges::nth_element(v, v.begin() + static_cast<long>(v.size() / 2)); return v[v.size() / 2]; };
            std::vector<float> offs; for (const auto& q : own) offs.push_back(q.second);
            const float med = median(offs);
            if (std::abs(med) >= 0.25f * p_.cell)
            {
                // slide: vertex surgery, count-preserving
                auto Vs = V; const int ax = e.h ? 1 : 0;
                Vs[m][ax] += e.sgn * med; Vs[(m + 1) % n][ax] += e.sgn * med;
                if (freespace::is_simple(Vs) and freespace::signed_area(Vs) > 0.f)
                    out.push_back({{"slide", m, med, 0.f}, freespace::FreeSpacePolygon::merge_collinear(Vs)});
                // strip: union/difference of the strip over the edge's extent
                if (auto q = strip(V, e, e.lo, e.hi, med)) out.push_back({{"strip", m, med, 0.f}, *q});
            }
            if (own.size() >= 12)
            {   // jog: the best single change-point (two medians, L1 cost)
                std::size_t bc = 0; float bcost = std::numeric_limits<float>::infinity(), m1 = 0, m2 = 0;
                for (std::size_t c = 4; c + 4 < own.size(); ++c)
                {
                    if (own[c].first - own[c - 1].first < 1e-6f) continue;
                    std::vector<float> a, b; for (std::size_t i = 0; i < own.size(); ++i) (i < c ? a : b).push_back(own[i].second);
                    const float ma = median(a), mb = median(b); float cost = 0.f;
                    for (float x : a) cost += std::abs(x - ma); for (float x : b) cost += std::abs(x - mb);
                    if (cost < bcost) { bcost = cost; bc = c; m1 = ma; m2 = mb; }
                }
                if (bc > 0 and std::abs(m1 - m2) >= 0.5f * p_.cell)
                {
                    const float cp = 0.5f * (own[bc - 1].first + own[bc].first);
                    std::optional<std::vector<Eigen::Vector2f>> q = V;
                    for (const auto& [a0, a1, dv] : {std::tuple{e.lo, cp, m1}, std::tuple{cp, e.hi, m2}})
                        if (q and std::abs(dv) >= 0.25f * p_.cell)
                        {   // both halves are boxes on the ORIGINAL edge line; the second applies to the first's result
                            const auto b = strip_box(e, a0, a1, dv);
                            q = box_edit(*q, b[0], b[1], b[2], b[3], dv > 0.f);
                        }
                    if (q) out.push_back({{"jog", m, m1 - m2, 0.f}, *q});
                }
            }
        }
        // contested masks in the room axes: EXT = free & outside & beyond sigma; UNS = inside & not free & beyond sigma
        {
            auto r = rasterize(V, p_.reach + 2 * p_.cell);
            std::vector<std::uint8_t> freem(r.m.size(), 0), ext(r.m.size(), 0), uns(r.m.size(), 0);
            auto at = [&](const Eigen::Vector2f& q, int& i, int& j)
            { i = static_cast<int>(std::floor((q.x() - r.u0) / p_.cell)); j = static_cast<int>(std::floor((q.y() - r.v0) / p_.cell)); return i >= 0 and j >= 0 and i < r.W and j < r.H; };
            for (const auto& q : sc.Fr) if (int i, j; at(q, i, j)) freem[static_cast<std::size_t>(j) * r.W + i] = 1;
            for (int j = 0; j < r.H; ++j) for (int i = 0; i < r.W; ++i)
            {
                const std::size_t k = static_cast<std::size_t>(j) * r.W + i;
                const Eigen::Vector2f c(r.u0 + (i + 0.5f) * p_.cell, r.v0 + (j + 0.5f) * p_.cell);
                const bool near = nearest(c, E).d <= sc.sigma;
                ext[k] = (freem[k] and not r.m[k] and not near) ? 1 : 0;
                uns[k] = (r.m[k] and not freem[k] and not near) ? 1 : 0;
            }
            for (const auto& [kind, M, add] : {std::tuple{"extend", &ext, true}, std::tuple{"notch", &uns, false}})
                for (const auto& b : largest_rectangles(*M, r.W, r.H, p_.k_rect))
                {
                    const auto bx = snap_box({r.u0 + b[0] * p_.cell, r.v0 + b[1] * p_.cell, r.u0 + b[2] * p_.cell, r.v0 + b[3] * p_.cell},
                                             E, sc.sigma + p_.cell);
                    if (auto q = box_edit(V, bx[0], bx[1], bx[2], bx[3], add)) out.push_back({{kind, -1, float((b[2] - b[0]) * (b[3] - b[1])), 0.f}, *q});
                }
        }
        return out;
    }
    std::optional<std::vector<Eigen::Vector2f>> strip(const std::vector<Eigen::Vector2f>& V, const Edge& e, float a0, float a1, float dv) const
    {
        const auto b = strip_box(e, a0, a1, dv);
        return box_edit(V, b[0], b[1], b[2], b[3], dv > 0.f);
    }
};
} // namespace rc::refine
