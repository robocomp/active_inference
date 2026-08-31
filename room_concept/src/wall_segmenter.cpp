#include "wall_segmenter.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace rc::wallseg
{
    namespace
    {
        constexpr float kPi = static_cast<float>(M_PI);

        struct Hypothesis
        {
            Eigen::Vector2f n;
            float d = 0.f;
            int support = 0;
        };

        /// Line through two points, in Hesse form. nullopt when the points coincide.
        std::optional<Hypothesis> two_point_line(const Eigen::Vector2f& a, const Eigen::Vector2f& b)
        {
            const Eigen::Vector2f e = b - a;
            const float len = e.norm();
            if (len < 1e-6f) return std::nullopt;
            Hypothesis h;
            h.n = Eigen::Vector2f(-e.y(), e.x()) / len;
            h.d = h.n.dot(a);
            return h;
        }

        /// Endpoint variance along the wall: the true end lies somewhere within one inter-point gap of
        /// the last inlier (uniform ⇒ gap²/12), plus the perpendicular noise projected along a
        /// slightly-wrong direction (σ²). No metre constant: gap is MEASURED per segment.
        float endpoint_var(float gap, float sigma2)
        {
            return gap * gap / 12.f + sigma2;
        }
    } // namespace

    float WallSegment::sigma2() const
    {
        return resid_var;   // already σ_sensor² + scatter, set by the extractor (see refit below)
    }

    float point_gain_nats(float r, float sigma2, float clutter_area, float extent)
    {
        const float sigma = std::sqrt(std::max(sigma2, 1e-12f));
        const float L = std::max(extent, sigma);          // a line shorter than its noise has no length
        const float ratio = std::max(clutter_area, 1e-6f) / (L * std::sqrt(2.f * kPi) * sigma);
        return std::log(ratio) - 0.5f * r * r / std::max(sigma2, 1e-12f);
    }

    void orient_towards_origin(Eigen::Vector2f& normal, float& d)
    {
        // n·0 − d > 0 ⇔ d < 0: the origin is on the side the normal points to.
        if (d > 0.f) { normal = -normal; d = -d; }
    }

    Result segment(const std::vector<Eigen::Vector2f>& pts, const Params& p, std::mt19937& rng)
    {
        Result out;
        const int N = static_cast<int>(pts.size());
        if (N < p.min_points) { out.unexplained.resize(N); std::iota(out.unexplained.begin(), out.unexplained.end(), 0); return out; }

        std::vector<int> remaining(N);
        std::iota(remaining.begin(), remaining.end(), 0);

        const float sigma2_sensor = p.sensor_sigma * p.sensor_sigma;
        // Clutter model from the scan itself: bbox area and diagonal.
        {
            Eigen::Vector2f lo = pts.front(), hi = pts.front();
            for (const auto& q : pts) { lo = lo.cwiseMin(q); hi = hi.cwiseMax(q); }
            const Eigen::Vector2f ext = hi - lo;
            out.clutter_area = ext.x() * ext.y();
            const float diag = ext.norm();
            const float ratio = out.clutter_area / std::max(1e-6f, diag * std::sqrt(2.f * kPi) * p.sensor_sigma);
            out.band = (ratio > 1.f) ? p.sensor_sigma * std::sqrt(2.f * std::log(ratio))
                                     : p.sensor_sigma * std::sqrt(p.chi2_inlier);
        }
        const float inlier_band2  = out.band * out.band;              // r² below this ⇒ inlier

        while (static_cast<int>(remaining.size()) >= std::max(p.min_points, p.min_remaining)
               and static_cast<int>(out.segments.size()) < p.max_segments)
        {
            const int M = static_cast<int>(remaining.size());
            std::uniform_int_distribution<int> pick(0, M - 1);

            Hypothesis best;
            best.support = 0;
            // Adaptive iteration count: k = log(1−p) / log(1 − w²), w = best inlier fraction so far.
            int k_needed = p.ransac_max_iters;
            int iters = 0;
            while (iters < k_needed and iters < p.ransac_max_iters)
            {
                ++iters;
                const int i = pick(rng);
                int j = pick(rng);
                if (j == i) continue;
                auto h = two_point_line(pts[remaining[i]], pts[remaining[j]]);
                if (not h) continue;
                int support = 0;
                for (int idx : remaining)
                {
                    const float r = h->n.dot(pts[idx]) - h->d;
                    if (r * r <= inlier_band2) ++support;
                }
                if (support > best.support)
                {
                    best = *h;
                    best.support = support;
                    const float w = static_cast<float>(support) / static_cast<float>(M);
                    const float denom = std::log(std::max(1e-9f, 1.f - w * w));
                    // denom < 0 always; a perfect w=1 gives log(1e-9) ⇒ k_needed = 1.
                    k_needed = std::min(p.ransac_max_iters,
                                        std::max(1, static_cast<int>(std::ceil(std::log(1.f - p.ransac_confidence) / denom))));
                }
            }
            out.ransac_iters += iters;
            if (best.support < p.min_points) break;

            // Gather the inliers of the best hypothesis and refit by PCA (the two-point line is only a
            // proposal; the least-squares line is what the points actually say).
            std::vector<int> inl;
            std::vector<Eigen::Vector2f> ipts;
            for (int idx : remaining)
            {
                const float r = best.n.dot(pts[idx]) - best.d;
                if (r * r <= inlier_band2) { inl.push_back(idx); ipts.push_back(pts[idx]); }
            }
            auto line = linefit::fit_line_pca(ipts, p.min_points);
            if (not line) break;

            // Second pass on the REFIT line: the PCA line can admit points the proposal excluded and
            // vice versa; one re-gather keeps the segment honest about its own support.
            inl.clear(); ipts.clear();
            for (int idx : remaining)
            {
                const float r = line->normal.dot(pts[idx]) - line->d;
                if (r * r <= inlier_band2) { inl.push_back(idx); ipts.push_back(pts[idx]); }
            }
            if (static_cast<int>(inl.size()) < p.min_points) break;
            line = linefit::fit_line_pca(ipts, p.min_points);
            if (not line) break;

            WallSegment seg;
            seg.normal = line->normal;
            seg.d      = line->d;
            orient_towards_origin(seg.normal, seg.d);
            seg.phi    = std::atan2(seg.normal.y(), seg.normal.x());
            seg.npts   = static_cast<int>(inl.size());
            // Per-point noise this segment implies: the sensor's own plus whatever scatter the fit
            // could not explain (two physical noise sources; no floor).
            seg.resid_var = sigma2_sensor + line->resid_var;

            // Extent along the tangent, and the outermost spacings for the endpoint uncertainty.
            const Eigen::Vector2f t = linefit::tangent_of(seg.phi);
            std::vector<float> s(ipts.size());
            for (size_t i = 0; i < ipts.size(); ++i) s[i] = t.dot(ipts[i]);
            std::sort(s.begin(), s.end());
            seg.s_min = s.front();
            seg.s_max = s.back();
            seg.gap0  = (s.size() >= 2) ? s[1] - s[0] : 0.f;
            seg.gap1  = (s.size() >= 2) ? s[s.size() - 1] - s[s.size() - 2] : 0.f;
            seg.p0 = seg.normal * seg.d + t * seg.s_min;
            seg.p1 = seg.normal * seg.d + t * seg.s_max;
            {
                linefit::Line2D l; l.normal = seg.normal; l.d = seg.d;
                seg.info_phi_d = linefit::info_phi_d(ipts, l, seg.resid_var);
            }
            seg.inliers = inl;
            out.segments.push_back(std::move(seg));

            // Remove the claimed points.
            std::vector<int> next;
            next.reserve(remaining.size() - inl.size());
            std::sort(inl.begin(), inl.end());
            std::set_difference(remaining.begin(), remaining.end(), inl.begin(), inl.end(),
                                std::back_inserter(next));
            remaining.swap(next);
        }

        // ── Collinear merge ─────────────────────────────────────────────────────────────────────
        // Two segments are the SAME wall when their (φ, d) difference is not resolvable given their own
        // uncertainties: Δᵀ (Λ_a⁻¹ + Λ_b⁻¹)⁻¹ Δ inside a χ²₂ region. A doorway splits the points of one
        // wall into two RANSAC segments; this puts them back together, information-weighted.
        bool merged_any = true;
        while (merged_any and p.chi2_merge > 0.f)
        {
            merged_any = false;
            for (size_t a = 0; a < out.segments.size() and not merged_any; ++a)
                for (size_t b = a + 1; b < out.segments.size() and not merged_any; ++b)
                {
                    const auto& A = out.segments[a];
                    const auto& B = out.segments[b];
                    Eigen::Vector2f delta(linefit::wrap_pi(A.phi - B.phi), A.d - B.d);
                    const Eigen::Matrix2f Sa = A.info_phi_d.inverse();
                    const Eigen::Matrix2f Sb = B.info_phi_d.inverse();
                    if (not Sa.allFinite() or not Sb.allFinite()) continue;
                    const Eigen::Matrix2f S = Sa + Sb;
                    const float chi2 = delta.dot(S.inverse() * delta);
                    if (not std::isfinite(chi2) or chi2 > p.chi2_merge) continue;

                    // Fuse: refit on the union of inliers so the merged segment is a real fit, not an average.
                    std::vector<int> inl = A.inliers;
                    inl.insert(inl.end(), B.inliers.begin(), B.inliers.end());
                    std::vector<Eigen::Vector2f> ipts;
                    ipts.reserve(inl.size());
                    for (int idx : inl) ipts.push_back(pts[idx]);
                    auto line = linefit::fit_line_pca(ipts, p.min_points);
                    if (not line) continue;
                    WallSegment seg;
                    seg.normal = line->normal; seg.d = line->d;
                    orient_towards_origin(seg.normal, seg.d);
                    seg.phi = std::atan2(seg.normal.y(), seg.normal.x());
                    seg.npts = static_cast<int>(inl.size());
                    seg.resid_var = sigma2_sensor + line->resid_var;
                    const Eigen::Vector2f t = linefit::tangent_of(seg.phi);
                    std::vector<float> s(ipts.size());
                    for (size_t i = 0; i < ipts.size(); ++i) s[i] = t.dot(ipts[i]);
                    std::sort(s.begin(), s.end());
                    seg.s_min = s.front(); seg.s_max = s.back();
                    seg.gap0 = (s.size() >= 2) ? s[1] - s[0] : 0.f;
                    seg.gap1 = (s.size() >= 2) ? s[s.size() - 1] - s[s.size() - 2] : 0.f;
                    seg.p0 = seg.normal * seg.d + t * seg.s_min;
                    seg.p1 = seg.normal * seg.d + t * seg.s_max;
                    { linefit::Line2D l; l.normal = seg.normal; l.d = seg.d; seg.info_phi_d = linefit::info_phi_d(ipts, l, seg.resid_var); }
                    seg.inliers = std::move(inl);
                    out.segments[a] = std::move(seg);
                    out.segments.erase(out.segments.begin() + static_cast<long>(b));
                    merged_any = true;
                }
        }

        // ── Observed corners: which segments MEET ───────────────────────────────────────────────
        for (size_t a = 0; a < out.segments.size(); ++a)
            for (size_t b = a + 1; b < out.segments.size(); ++b)
                if (auto c = corner_of(out.segments[a], out.segments[b], static_cast<int>(a), static_cast<int>(b), p))
                    out.corners.push_back(*c);

        out.unexplained = std::move(remaining);
        return out;
    }

    std::optional<ObservedCorner> corner_of(const WallSegment& a, const WallSegment& b,
                                            int ia, int ib, const Params& p)
    {
        // Parallel lines never meet. The test is on the sine of the angle between the normals against
        // the angular uncertainty of the two fits — two nearly-parallel lines with sharp angles do
        // intersect somewhere, but that "corner" is a grazing artefact, and its endpoint test below
        // is what actually decides. Here only the numerically degenerate case is excluded.
        const float sin_ab = std::abs(a.normal.x() * b.normal.y() - a.normal.y() * b.normal.x());
        if (sin_ab < 1e-4f) return std::nullopt;

        const auto c = linefit::intersect(a.line(), b.line());
        if (not c) return std::nullopt;

        // The intersection must lie at an END of each segment, within that end's uncertainty. A
        // corner in the MIDDLE of a segment is a T-junction (a partition meeting a wall), not a
        // polygon vertex; a corner far beyond both extents is two walls that do not meet (a doorway
        // between them, or a corner the robot cannot see). Mahalanobis on the along-wall offset from
        // the nearest end, with the end's own variance.
        const auto end_chi2 = [&](const WallSegment& s) -> float
        {
            const Eigen::Vector2f t = linefit::tangent_of(s.phi);
            const float sc = t.dot(*c);
            const float d0 = sc - s.s_min, d1 = sc - s.s_max;
            const float v0 = endpoint_var(s.gap0, s.sigma2());
            const float v1 = endpoint_var(s.gap1, s.sigma2());
            // Inside the extent (s_min < sc < s_max): distance to the nearer end still applies — a
            // corner a few cm inside the last inlier is the same physical corner.
            return std::min(d0 * d0 / v0, d1 * d1 / v1);
        };
        const float ca = end_chi2(a), cb = end_chi2(b);
        // ⚠ χ²₁ level (the same as the inlier test) — a discrete "do these meet" decision.
        if (ca > p.chi2_inlier or cb > p.chi2_inlier) return std::nullopt;

        ObservedCorner oc;
        oc.seg_a = ia; oc.seg_b = ib;
        oc.point = *c;
        oc.information = a.normal * a.normal.transpose() / a.sigma2()
                       + b.normal * b.normal.transpose() / b.sigma2();
        oc.chi2_end = std::max(ca, cb);
        return oc;
    }
} // namespace rc::wallseg
