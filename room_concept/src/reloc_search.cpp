/*
 *  reloc_search.cpp — see reloc_search.h for the model and why it replaces the lattice.
 */
#include "reloc_search.h"
#include "wall_segmenter.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numbers>
#include <random>
#include <ranges>

namespace rc::reloc
{
    namespace
    {
        constexpr float kPi  = std::numbers::pi_v<float>;
        constexpr float kInf = std::numeric_limits<float>::infinity();

        float wrap(float a) { return std::remainder(a, 2.f * kPi); }

        /// log[(1−ε)·2·N(d; 0, σ²) + ε/L] — the per-point term of ℓ, and the inlier responsibility.
        struct PointTerm
        {
            float log_in_norm;   // log((1−ε)·2/(√(2π)σ))
            float log_out;       // log(ε/L)
            float inv_2s2;
            PointTerm(const Params& p, float extent)
                : log_in_norm(std::log((1.f - p.outlier_frac) * 2.f / (std::sqrt(2.f * kPi) * p.sigma_sdf))),
                  log_out(std::log(p.outlier_frac / std::max(extent, 1e-3f))),
                  inv_2s2(1.f / (2.f * p.sigma_sdf * p.sigma_sdf)) {}
            float ll(float d, float* gamma = nullptr) const
            {
                const float li = log_in_norm - d * d * inv_2s2;
                const float hi = std::max(li, log_out), lo = std::min(li, log_out);
                const float out = hi + std::log1p(std::exp(lo - hi));
                if (gamma != nullptr) *gamma = std::exp(li - out);
                return out;
            }
        };

        float polygon_extent(const std::vector<Eigen::Vector2f>& poly)
        {
            Eigen::Vector2f lo = poly.front(), hi = poly.front();
            for (const auto& v : poly) { lo = lo.cwiseMin(v); hi = hi.cwiseMax(v); }
            return (hi - lo).norm();
        }

        /// Closest boundary point of the polygon to q (and, optionally, which edge it lies on).
        Eigen::Vector2f closest_on_polygon(const std::vector<Eigen::Vector2f>& poly, const Eigen::Vector2f& q,
                                           int* edge = nullptr)
        {
            float best = kInf;
            Eigen::Vector2f best_c = poly.front();
            const auto n = poly.size();
            for (std::size_t i = 0; i < n; ++i)
            {
                const Eigen::Vector2f& a = poly[i];
                const Eigen::Vector2f ab = poly[(i + 1) % n] - a;
                const float len2 = ab.squaredNorm();
                const float t = len2 > 1e-12f ? std::clamp((q - a).dot(ab) / len2, 0.f, 1.f) : 0.f;
                const Eigen::Vector2f c = a + t * ab;
                const float d2 = (q - c).squaredNorm();
                if (d2 < best) { best = d2; best_c = c; if (edge != nullptr) *edge = static_cast<int>(i); }
            }
            return best_c;
        }

        float log_det3(const Eigen::Matrix3f& m)
        {
            const Eigen::LLT<Eigen::Matrix3d> llt(m.cast<double>());
            if (llt.info() != Eigen::Success) return 0.f;
            const auto& L = llt.matrixL();
            return static_cast<float>(2.0 * (std::log(L(0, 0)) + std::log(L(1, 1)) + std::log(L(2, 2))));
        }

        std::vector<Eigen::Vector2f> subsample(const std::vector<Eigen::Vector2f>& pts, int max_pts)
        {
            if (max_pts <= 0 or static_cast<int>(pts.size()) <= max_pts) return pts;
            std::vector<Eigen::Vector2f> out;
            out.reserve(static_cast<std::size_t>(max_pts));
            const double stride = static_cast<double>(pts.size()) / max_pts;
            for (int i = 0; i < max_pts; ++i)
                out.push_back(pts[static_cast<std::size_t>(i * stride)]);
            return out;
        }
    } // namespace

    // ── geometry ────────────────────────────────────────────────────────────────────────────────────
    float polygon_distance(const std::vector<Eigen::Vector2f>& polygon, const Eigen::Vector2f& q)
    {
        return (q - closest_on_polygon(polygon, q)).norm();
    }

    bool inside_polygon(const std::vector<Eigen::Vector2f>& poly, const Eigen::Vector2f& q)
    {
        bool in = false;
        const auto n = poly.size();
        for (std::size_t i = 0, j = n - 1; i < n; j = i++)
        {
            const auto& a = poly[i];
            const auto& b = poly[j];
            if (((a.y() > q.y()) != (b.y() > q.y()))
                and (q.x() < (b.x() - a.x()) * (q.y() - a.y()) / (b.y() - a.y()) + a.x()))
                in = not in;
        }
        return in;
    }

    // ── raster ──────────────────────────────────────────────────────────────────────────────────────
    void LikelihoodRaster::build(const std::vector<Eigen::Vector2f>& polygon, const Params& p)
    {
        nx_ = ny_ = 0;
        ll_.clear(); dist_.clear();
        if (polygon.size() < 3) return;
        extent_ = polygon_extent(polygon);
        if (not (extent_ > 0.f)) return;

        // Pad by 4σ: beyond that the wall term is exp(−8) of its peak and a point is pure clutter, so
        // off-raster lookups lose nothing by returning the clutter term.
        const float pad = 4.f * p.sigma_sdf;
        Eigen::Vector2f lo = polygon.front(), hi = polygon.front();
        for (const auto& v : polygon) { lo = lo.cwiseMin(v); hi = hi.cwiseMax(v); }
        res_ = p.raster_res;
        x0_ = lo.x() - pad;
        y0_ = lo.y() - pad;
        nx_ = static_cast<int>(std::ceil((hi.x() - lo.x() + 2.f * pad) / res_)) + 1;
        ny_ = static_cast<int>(std::ceil((hi.y() - lo.y() + 2.f * pad) / res_)) + 1;

        const PointTerm term(p, extent_);
        ll_out_ = term.log_out;
        ll_.resize(static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_));
        dist_.resize(ll_.size());
        for (int j = 0; j < ny_; ++j)
            for (int i = 0; i < nx_; ++i)
            {
                // Cell CENTRE, so nearest-cell lookup (floor) is unbiased.
                const Eigen::Vector2f q(x0_ + (i + 0.5f) * res_, y0_ + (j + 0.5f) * res_);
                const float d = polygon_distance(polygon, q);
                const auto k = static_cast<std::size_t>(j) * static_cast<std::size_t>(nx_) + static_cast<std::size_t>(i);
                dist_[k] = d;
                ll_[k] = term.ll(d);
            }
    }

    float LikelihoodRaster::ll_at(float x, float y) const
    {
        const int i = static_cast<int>(std::floor((x - x0_) / res_));
        const int j = static_cast<int>(std::floor((y - y0_) / res_));
        if (i < 0 or j < 0 or i >= nx_ or j >= ny_) return ll_out_;
        return ll_[static_cast<std::size_t>(j) * static_cast<std::size_t>(nx_) + static_cast<std::size_t>(i)];
    }

    // ── exact likelihood ────────────────────────────────────────────────────────────────────────────
    float log_likelihood(const std::vector<Eigen::Vector2f>& polygon,
                         const std::vector<Eigen::Vector2f>& pts_robot,
                         const Eigen::Vector3f& pose, const Params& p, float extent, float* median_abs)
    {
        const PointTerm term(p, extent);
        const float c = std::cos(pose.z()), s = std::sin(pose.z());
        float ll = 0.f;
        std::vector<float> ds;
        if (median_abs != nullptr) ds.reserve(pts_robot.size());
        for (const auto& pr : pts_robot)
        {
            const Eigen::Vector2f q(c * pr.x() - s * pr.y() + pose.x(), s * pr.x() + c * pr.y() + pose.y());
            const float d = polygon_distance(polygon, q);
            ll += term.ll(d);
            if (median_abs != nullptr) ds.push_back(d);
        }
        if (median_abs != nullptr)
        {
            if (ds.empty()) *median_abs = std::numeric_limits<float>::quiet_NaN();
            else
            {
                const auto mid = ds.begin() + static_cast<std::ptrdiff_t>(ds.size() / 2);
                std::ranges::nth_element(ds, mid);
                *median_abs = *mid;
            }
        }
        return ll;
    }

    // ── yaw hypotheses from structure ───────────────────────────────────────────────────────────────
    std::vector<float> structural_yaws(const std::vector<Eigen::Vector2f>& polygon,
                                       const std::vector<Eigen::Vector2f>& pts_robot,
                                       const Params& p, std::uint32_t seed, int* n_segments)
    {
        if (n_segments != nullptr) *n_segments = 0;
        if (polygon.size() < 3 or pts_robot.size() < 10) return {};

        std::mt19937 rng(seed);
        const wallseg::Params sp;
        const auto seg = wallseg::segment(pts_robot, sp, rng);
        if (n_segments != nullptr) *n_segments = static_cast<int>(seg.segments.size());
        if (seg.segments.empty()) return {};

        // Inward normals of the polygon edges. The sign of the area fixes which side is inside.
        float area2 = 0.f;
        const auto n = polygon.size();
        for (std::size_t i = 0; i < n; ++i)
        {
            const auto& a = polygon[i];
            const auto& b = polygon[(i + 1) % n];
            area2 += a.x() * b.y() - b.x() * a.y();
        }
        const float ccw = area2 > 0.f ? 1.f : -1.f;
        struct Edge { float psi; float len; };
        std::vector<Edge> edges;
        for (std::size_t i = 0; i < n; ++i)
        {
            const Eigen::Vector2f ab = polygon[(i + 1) % n] - polygon[i];
            const float len = ab.norm();
            if (len < 1e-4f) continue;
            // CCW polygon: interior on the LEFT of a→b, so the inward normal is (−ab.y, ab.x).
            edges.push_back({std::atan2(ccw * ab.x(), -ccw * ab.y()), len});
        }

        // Kernel density on the circle of the implied yaw θ = ψ − φ. Each kernel is as wide as the
        // segment's own σ_φ (from its information matrix), weighted by the points it explains and the
        // length of the edge it is paired with (a long wall is more likely to be the one in view).
        constexpr int kBins = 720;   // 0.5°
        const float bin = 2.f * kPi / kBins;
        std::vector<double> dens(kBins, 0.0);
        for (const auto& sgm : seg.segments)
        {
            const Eigen::Matrix2f info = sgm.info_phi_d;
            float var_phi = 1e-2f;
            if (std::abs(info.determinant()) > 1e-12f)
                var_phi = std::max(info.inverse()(0, 0), 0.f);
            // A kernel narrower than a bin would alias; widen to the bin, which is the density's resolution.
            const float sig = std::sqrt(std::max(var_phi, bin * bin));
            const int half = std::min(kBins / 2, static_cast<int>(std::ceil(3.f * sig / bin)));
            for (const auto& e : edges)
            {
                const float theta = wrap(e.psi - sgm.phi);
                const int c = static_cast<int>(std::lround((theta + kPi) / bin)) % kBins;
                const double w = static_cast<double>(sgm.npts) * e.len;
                for (int k = -half; k <= half; ++k)
                {
                    const float dth = k * bin;
                    dens[static_cast<std::size_t>((c + k + kBins) % kBins)] +=
                        w * std::exp(-0.5 * dth * dth / (sig * sig));
                }
            }
        }

        struct Peak { double v; float theta; };
        std::vector<Peak> peaks;
        for (int i = 0; i < kBins; ++i)
        {
            const double v = dens[static_cast<std::size_t>(i)];
            if (v <= 0.0) continue;
            if (v >= dens[static_cast<std::size_t>((i + 1) % kBins)]
                and v > dens[static_cast<std::size_t>((i - 1 + kBins) % kBins)])
                peaks.push_back({v, -kPi + i * bin});
        }
        std::ranges::sort(peaks, std::ranges::greater{}, &Peak::v);
        std::vector<float> yaws;
        for (const auto& pk : peaks | std::views::take(p.max_yaws))
            yaws.push_back(pk.theta);
        return yaws;
    }

    // ── polish ──────────────────────────────────────────────────────────────────────────────────────
    Mode polish(const std::vector<Eigen::Vector2f>& polygon,
                const std::vector<Eigen::Vector2f>& pts_robot,
                const Eigen::Vector3f& start, const Params& p, float extent)
    {
        // σ is ESTIMATED here, by EM on the inlier responsibilities (σ² ← Σγd²/Σγ), annealing down from the wide
        // σ_sdf the lattice needs for its basins. A polish at the fixed σ_sdf = 0.15 m — the agent's model-
        // mismatch noise, ~7x a scan's real spread — left the likelihood flat over ±10 cm and let one-sided
        // clutter drag the mode: measured in reloc_selftest, right-basin modes polished to median |d| 0.056 m
        // where the truth scored 0.019, and the NEES tail. Never raised above σ_sdf: annealing only.
        Params pl = p;
        PointTerm term(pl, extent);
        float inv_s2 = 1.f / (pl.sigma_sdf * pl.sigma_sdf);
        double em_wd2 = 0.0, em_w = 0.0;
        // The search's own prior: uniform over the room and the circle. Its information (1/extent² on
        // x,y, 1/π² on θ) keeps H invertible along a direction the scan does not constrain — a corridor
        // then reports a covariance as large as the room, which is the honest answer, not a NaN.
        Eigen::Matrix3f prior = Eigen::Matrix3f::Zero();
        prior(0, 0) = prior(1, 1) = 1.f / std::max(extent * extent, 1e-3f);
        prior(2, 2) = 1.f / (kPi * kPi);

        Eigen::Vector3f x = start;
        Eigen::Matrix3f H = prior;
        float ll_cur = -kInf;
        float lambda = 1e-3f;
        // Per-edge sums for the common-mode marginalisation below: Σw and Σw·J over the points on each edge.
        struct EdgeSum { float sw = 0.f; Eigen::Vector3f swj = Eigen::Vector3f::Zero(); };
        std::vector<EdgeSum> edge_sums;
        auto evaluate = [&](const Eigen::Vector3f& pose, Eigen::Matrix3f* Hout, Eigen::Vector3f* gout) -> float
        {
            const float c = std::cos(pose.z()), s = std::sin(pose.z());
            float ll = 0.f;
            Eigen::Matrix3f Hs = prior;
            Eigen::Vector3f g = Eigen::Vector3f::Zero();
            if (Hout != nullptr) { edge_sums.assign(polygon.size(), EdgeSum{}); em_wd2 = 0.0; em_w = 0.0; }
            for (const auto& pr : pts_robot)
            {
                const Eigen::Vector2f rp(c * pr.x() - s * pr.y(), s * pr.x() + c * pr.y());
                const Eigen::Vector2f q = rp + pose.head<2>();
                int edge = 0;
                const Eigen::Vector2f cl = closest_on_polygon(polygon, q, &edge);
                const Eigen::Vector2f diff = q - cl;
                const float d = diff.norm();
                float gamma = 0.f;
                ll += term.ll(d, &gamma);
                if (Hout == nullptr) continue;
                em_wd2 += static_cast<double>(gamma) * d * d;
                em_w   += gamma;
                if (d < 1e-6f) continue;
                const Eigen::Vector2f u = diff / d;
                // ∂q/∂θ = R'·p = (−rp.y, rp.x)
                const Eigen::Vector3f J(u.x(), u.y(), -u.x() * rp.y() + u.y() * rp.x());
                const float w = gamma * inv_s2;
                Hs += w * J * J.transpose();
                g  += (w * d) * J;     // ∂(−ℓ)/∂x
                auto& es = edge_sums[static_cast<std::size_t>(edge)];
                es.sw += w;
                es.swj += w * J;
            }
            if (Hout != nullptr) *Hout = Hs;
            if (gout != nullptr) *gout = g;
            return ll;
        };

        Eigen::Vector3f g;
        ll_cur = evaluate(x, &H, &g);
        for (int it = 0; it < p.polish_iters; ++it)
        {
            Eigen::Matrix3f A = H;
            A.diagonal() *= (1.f + lambda);
            const Eigen::Vector3f step = A.ldlt().solve(-g);
            if (not step.allFinite()) break;
            Eigen::Vector3f xn = x + step;
            xn.z() = wrap(xn.z());
            Eigen::Matrix3f Hn;
            Eigen::Vector3f gn;
            const float ll_new = evaluate(xn, &Hn, &gn);
            if (ll_new >= ll_cur)
            {
                x = xn; H = Hn; g = gn; ll_cur = ll_new;
                lambda = std::max(lambda * 0.3f, 1e-6f);
                // EM step on σ at the accepted pose (the sums evaluate() just filled belong to it).
                bool sigma_moved = false;
                if (em_w > 3.0)
                {
                    const float sig_em = static_cast<float>(std::sqrt(em_wd2 / em_w));
                    if (std::isfinite(sig_em) and sig_em > 0.f and sig_em < 0.99f * pl.sigma_sdf)
                    {
                        pl.sigma_sdf = sig_em;
                        term = PointTerm(pl, extent);
                        inv_s2 = 1.f / (pl.sigma_sdf * pl.sigma_sdf);
                        ll_cur = evaluate(x, &H, &g);   // the objective changed with σ: re-linearise
                        sigma_moved = true;
                    }
                }
                if (sigma_moved) continue;
                // Converged when the UNDAMPED Newton step is negligible against the pose's own uncertainty
                // (Mahalanobis ≪ 1). Testing the change in ℓ instead stopped on a tiny step taken under
                // heavy damping — measured in reloc_selftest: modes left 5–12 cm short with ℓ 14–50 nats
                // below the truth's.
                const Eigen::Vector3f newton = H.ldlt().solve(-g);
                if (newton.allFinite() and newton.dot(H * newton) < 1e-6f) break;
            }
            else
                lambda *= 10.f;
        }

        // COVARIANCE WITH EACH WALL'S COMMON-MODE OFFSET MARGINALISED OUT.
        // The points on one wall do not err independently: clutter, a mis-placed map wall or an
        // unmodelled bias shifts them TOGETHER, and that shared part does not average down with more
        // points. Model it as an offset δ_e ~ N(0, σ_c²) added to every residual on edge e and integrate
        // it out (Woodbury, exact for the linearised model):
        //     H_e  =  Σ w J Jᵀ  −  (Σ w J)(Σ w J)ᵀ / (1/σ_c² + Σ w)
        // A wall seen by n points then contributes at most ~1/σ_c² along its normal instead of n/σ².
        // Measured in reloc_selftest before this term: NEES 4–6 (covariance 2–2.5x too tight in σ),
        // growing as points were ADDED — the signature of an error that does not average down.
        Eigen::Matrix3f H_marg = H;
        if (p.wall_offset_sigma > 0.f)
        {
            // Recompute the per-edge sums at the final pose (evaluate() filled them for the last accepted pose
            // only if that was the last call; make it explicit).
            evaluate(x, &H, nullptr);
            H_marg = H;
            const float inv_c2 = 1.f / (p.wall_offset_sigma * p.wall_offset_sigma);
            for (const auto& es : edge_sums)
                if (es.sw > 0.f)
                    H_marg -= (es.swj * es.swj.transpose()) / (inv_c2 + es.sw);
        }

        Mode m;
        m.pose = x;
        m.pose.z() = wrap(m.pose.z());
        m.cov = H_marg.inverse();
        float med = 0.f;
        m.log_lik = log_likelihood(polygon, pts_robot, m.pose, p, extent, &med);
        m.median_abs = med;
        return m;
    }

    // ── search ──────────────────────────────────────────────────────────────────────────────────────
    Result search(const LikelihoodRaster& raster,
                  const std::vector<Eigen::Vector2f>& polygon,
                  const std::vector<Eigen::Vector2f>& pts_all,
                  const std::vector<Eigen::Vector3f>& seed_poses,
                  const Params& p, std::uint32_t seed)
    {
        const auto t0 = std::chrono::steady_clock::now();
        Result res;
        if (not raster.valid() or polygon.size() < 3 or pts_all.empty()) return res;

        const float extent = raster.extent();
        const auto pts = subsample(pts_all, p.max_points);

        // 1. yaws: structure, else a uniform sweep; seed yaws and their half-turn always included.
        std::vector<float> yaws = structural_yaws(polygon, subsample(pts_all, 4 * p.max_points), p, seed,
                                                  &res.n_segments);
        if (yaws.empty())
            for (int k = 0; k < p.fallback_yaws; ++k)
                yaws.push_back(-kPi + 2.f * kPi * k / p.fallback_yaws);
        res.n_yaws = static_cast<int>(yaws.size());

        // 2. translation lattice, aligned to the raster so a lattice offset is an integer cell offset.
        const int m = std::max(1, static_cast<int>(std::lround(p.lattice_step / raster.res())));
        const float step = m * raster.res();
        const int lx = raster.nx() / m, ly = raster.ny() / m;
        // Valid centres: inside the polygon and clear of every wall by the body radius.
        std::vector<char> valid(static_cast<std::size_t>(lx) * static_cast<std::size_t>(ly), 0);
        for (int j = 0; j < ly; ++j)
            for (int i = 0; i < lx; ++i)
            {
                const Eigen::Vector2f t(raster.x0() + i * step, raster.y0() + j * step);
                const int ci = i * m, cj = j * m;
                const float d = raster.dist()[static_cast<std::size_t>(cj) * static_cast<std::size_t>(raster.nx())
                                              + static_cast<std::size_t>(ci)];
                valid[static_cast<std::size_t>(j) * static_cast<std::size_t>(lx) + static_cast<std::size_t>(i)] =
                    (d >= p.body_clearance and inside_polygon(polygon, t)) ? 1 : 0;
            }

        struct Cand { float ll; Eigen::Vector3f pose; };
        std::vector<Cand> maxima;
        std::vector<float> score(valid.size());
        std::vector<int> ci(pts.size()), cj(pts.size());
        const auto& ll = raster.ll();
        const int nx = raster.nx(), ny = raster.ny();
        for (const float yaw : yaws)
        {
            const float c = std::cos(yaw), s = std::sin(yaw);
            // Lattice point (i,j) sits at x0 + i·step; a point rotated to rp lands in cell
            // floor((rp.x + i·step)/res) = floor(rp.x/res + i·m) — the per-point part is precomputed.
            for (std::size_t k = 0; k < pts.size(); ++k)
            {
                ci[k] = static_cast<int>(std::floor((c * pts[k].x() - s * pts[k].y()) / raster.res()));
                cj[k] = static_cast<int>(std::floor((s * pts[k].x() + c * pts[k].y()) / raster.res()));
            }
            for (int j = 0; j < ly; ++j)
                for (int i = 0; i < lx; ++i)
                {
                    const auto idx = static_cast<std::size_t>(j) * static_cast<std::size_t>(lx) + static_cast<std::size_t>(i);
                    if (not valid[idx]) { score[idx] = -kInf; continue; }
                    float acc = 0.f;
                    const int oi = i * m, oj = j * m;
                    for (std::size_t k = 0; k < pts.size(); ++k)
                    {
                        const int a = ci[k] + oi, b = cj[k] + oj;
                        acc += (a < 0 or b < 0 or a >= nx or b >= ny)
                                   ? raster.ll_outside()
                                   : ll[static_cast<std::size_t>(b) * static_cast<std::size_t>(nx) + static_cast<std::size_t>(a)];
                    }
                    score[idx] = acc;
                    ++res.n_evals;
                }
            // Local maxima over the 8-neighbourhood (ties broken towards the lower index).
            for (int j = 0; j < ly; ++j)
                for (int i = 0; i < lx; ++i)
                {
                    const auto idx = static_cast<std::size_t>(j) * static_cast<std::size_t>(lx) + static_cast<std::size_t>(i);
                    const float v = score[idx];
                    if (v == -kInf) continue;
                    bool is_max = true;
                    for (int dj = -1; dj <= 1 and is_max; ++dj)
                        for (int di = -1; di <= 1; ++di)
                        {
                            if (di == 0 and dj == 0) continue;
                            const int a = i + di, b = j + dj;
                            if (a < 0 or b < 0 or a >= lx or b >= ly) continue;
                            const float w = score[static_cast<std::size_t>(b) * static_cast<std::size_t>(lx) + static_cast<std::size_t>(a)];
                            if (w > v or (w == v and (dj < 0 or (dj == 0 and di < 0)))) { is_max = false; break; }
                        }
                    if (is_max)
                        maxima.push_back({v, Eigen::Vector3f(raster.x0() + i * step, raster.y0() + j * step, yaw)});
                }
        }
        std::ranges::sort(maxima, std::ranges::greater{}, &Cand::ll);
        if (static_cast<int>(maxima.size()) > p.max_seeds) maxima.resize(static_cast<std::size_t>(p.max_seeds));

        // 3. polish every seed (lattice maxima + caller's seeds), then merge seeds that converged together.
        std::vector<Eigen::Vector3f> starts;
        for (const auto& c : maxima) starts.push_back(c.pose);
        for (const auto& sp : seed_poses) starts.push_back(sp);

        // The lattice ran on the subsample for speed; the polish is per seed and cheap, so it uses the
        // denser set — mode PRECISION comes from here, and a 150-point polish left modes ~10 cm off.
        const auto pts_polish = subsample(pts_all, 4 * p.max_points);
        std::vector<Mode> modes;
        for (const auto& st : starts)
        {
            Mode md = polish(polygon, pts_polish, st, p, extent);
            if (not md.pose.allFinite() or not std::isfinite(md.log_lik)) continue;
            bool merged = false;
            for (auto& other : modes)
            {
                Eigen::Vector3f dlt = md.pose - other.pose;
                dlt.z() = wrap(dlt.z());
                const Eigen::Matrix3f S = md.cov + other.cov;
                const float chi2 = dlt.dot(S.ldlt().solve(dlt));
                if (std::isfinite(chi2) and chi2 < p.chi2_merge)
                {
                    if (md.log_lik > other.log_lik) other = md;
                    merged = true;
                    break;
                }
            }
            if (not merged) modes.push_back(md);
        }

        // 4. Laplace evidence  ℓ* + ½log|Σ|, normalised by log-sum-exp.
        for (auto& md : modes) md.log_weight = md.log_lik + 0.5f * log_det3(md.cov);
        std::ranges::sort(modes, std::ranges::greater{}, &Mode::log_weight);
        if (static_cast<int>(modes.size()) > p.max_modes) modes.resize(static_cast<std::size_t>(p.max_modes));
        if (not modes.empty())
        {
            const float top = modes.front().log_weight;
            double z = 0.0;
            for (const auto& md : modes) z += std::exp(static_cast<double>(md.log_weight - top));
            const float logz = top + static_cast<float>(std::log(z));
            double w2 = 0.0;
            for (auto& md : modes)
            {
                md.log_weight -= logz;
                md.weight = std::exp(md.log_weight);
                w2 += static_cast<double>(md.weight) * md.weight;
                if (md.weight > 0.f) res.entropy -= md.weight * md.log_weight;
            }
            res.ess = w2 > 0.0 ? static_cast<float>(1.0 / w2) : 1.f;
        }
        res.modes = std::move(modes);
        res.duration_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return res;
    }
} // namespace rc::reloc
