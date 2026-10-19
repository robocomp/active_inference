#pragma once
/*
 * doorway_gap.h — the doorway as the robot's own LiDAR measures it: the span of the wall the rays pass THROUGH (Task 18,
 * 2026-10-19).
 *
 * WHY. tries 70–85 crossed whenever the interaction anchor was right, and failed when it was confidently wrong (0.2–1 m at a
 * claimed σ of 2 cm), when the robot's own body read as an obstacle, or when layers of estimators interacted. The last
 * stretch through the doorway is steered on what the LiDAR SEES: a direct measurement, no estimator in between to be wrong.
 * The anchor keeps two jobs: the approach, and saying WHERE ALONG WHICH WALL to look.
 *
 * THE OPENING IS WHERE RAYS PASS THROUGH THE WALL (the user's definition, 10-19). Once the door is open there is no
 * ambiguity among gaps: the opening is the contiguous span, along the wall line, where the returns land BEYOND the wall (or
 * there is none at all). A door only partly open leaves a narrow wall-to-leaf slit at the hinge that lets no ray through, so
 * the through-span IS the clear passable opening, from the leaf's free edge to the far jamb. Its edges are the last returns
 * ON the wall, the jamb or the leaf on each side.
 *
 * INPUT. Robot-frame returns (x right, y forward), already self-filtered by the body footprint at capture (the round-5 path);
 * the body's half-width; and the PRIOR (required): a point on the door's wall line and its normal (the direction of travel),
 * the covariance of that point (the anchor's own, or the carried gap's), and the door's full opening width.
 *
 * SCAN. Nearest return per bearing bin, at ANY range (a far room counts as "through") — ⚑ kGapScanBins is a discretisation
 * (0.25°: 6.5 mm at 1.5 m, below the sensor noise), not a model level. An empty bin is UNKNOWN: it neither opens nor closes
 * the span (a sparse sensor leaves empty bins on a solid wall).
 *
 * MEASUREMENT, the robot clear of the wall (|s| ≥ depth, s = the robot's offset along the normal from the line):
 *   every bin whose ray crosses the line beyond the robot is THROUGH if its return lies more than `depth` beyond the line
 *   (the slab the body occupies while passing — a jamb face, a leaf swung to the far side, anything inside it blocks), and
 *   BLOCKED otherwise (in front of the line: a leaf on the robot's side; on it: the wall). Runs of THROUGH bins (unknown bins
 *   bridged) bounded by a BLOCKED bin on both sides are the candidate openings; their edges are those two blocked returns,
 *   measured ALONG the wall (a = the line's direction). The span is then narrowed by any return inside the slab (|depth| <
 *   `depth`) between the edges — a leaf face on the robot's side (test c).
 *   The WALL LINE is then refitted from the blocked returns 0.1–1.0 m outside each edge: the prior's line and a total-least-
 *   squares line per side compete on residual (range-noise units) + direction under the prior's yaw (gross-widened) — an
 *   ajar leaf makes its side two lines, a leaf at 90° is a line perpendicular to the wall; both lose — and the measurement is
 *   repeated once on the winner, so an anchor's yaw error does not tilt the opening.
 * IN THE DOORWAY (|s| < depth: the jambs beside the body, no ray crosses the line usefully): the line is the prior's (the last
 *   fit, carried); the edges are the nearest returns inside the slab on each side of the robot's own projection on it.
 *
 * WHICH OPENING (a second opening further along the wall): the one whose centre is likeliest under the prior —
 *   s_loc = (1−ε)·exp(−½ dᵀS⁻¹d) + ε·exp(−½ dᵀ(S+G)⁻¹d),  d = c − μ,  S = P_xy + R_meas
 *   (a robust mixture: the anchor is right, or GROSSLY wrong — without the second term a confidently wrong anchor, try74's
 *   12σ, would make the true door "unseen"), times a one-sided door-width plausibility: an opening up to the door's width is
 *   plausible (a half-open door is narrower — the servo's P(body fits) slows or refuses on it), a wider one is not a door:
 *   s_w = exp(−½((w − w_door)₊/σ_w)²). The width is fed to the servo AS MEASURED.
 *
 * ⚑ DECISION LEVELS AND MODEL CONSTANTS, flagged (CLAUDE.md):
 *   kGapScoreFloor   = exp(−4.5): "no gap seen" below a 3σ match on one factor. A decision level — the controller either
 *                      steers on a measured gap or does not (it then carries the last one); there is no half-steering.
 *   kGapPriorGross   = 0.2: P(the anchor is grossly wrong) — tries 73/74/78 (3 of ~15 crossings) had a 0.2–1 m anchor error.
 *   kGapGrossXYM     = 1.0 m, kGapGrossYawRad = 0.4 rad: the scale of those errors (try78 1.0 m, try73 1.0 m / 21°).
 *   kGapWidthSigma   = 0.10 m: how far beyond the door's width an opening is still read as the door (its width estimate).
 *   kGapWallFitNearM = 0.10 m / kGapWallFitFarM = 1.0 m: the wall stretch the line is refitted on, outside each edge —
 *                      beyond a jamb face and a leaf's thickness, within one wall. Geometry of a doorway, not tuned.
 *   kGapWallFitMinPts = 8: fewer wall returns than this and the prior's line stands (a line from a handful of points is
 *                      not better than the anchor's).
 *
 * Pure Eigen + std. Tested in tests/test_doorway_gap.cpp.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <vector>

#include <Eigen/Dense>

#include "../../common/interaction_anchor/relative_pose.h"

namespace rc
{

inline constexpr int   kGapScanBins      = 1440;
inline constexpr float kGapScoreFloor    = 0.011108996f;   // exp(−4.5)
inline constexpr float kGapPriorGross    = 0.2f;
inline constexpr float kGapGrossXYM      = 1.0f;
inline constexpr float kGapGrossYawRad   = 0.4f;
inline constexpr float kGapWidthSigma    = 0.10f;
inline constexpr float kGapWallFitNearM  = 0.10f;
inline constexpr float kGapWallFitFarM   = 1.0f;
inline constexpr int   kGapWallFitMinPts = 8;

struct DoorwayGap
{
    bool seen = false;
    Eigen::Vector2f left = Eigen::Vector2f::Zero(), right = Eigen::Vector2f::Zero(), centre = Eigen::Vector2f::Zero();
    Eigen::Vector2f normal{0.f, 1.f};   // unit, the direction of travel through the gap (the wall's normal)
    float width = 0.f;                  // the clear width, along the wall
    float score = 0.f;
    int candidates = 0;                 // openings found (runs of through-rays bounded on both sides)
    bool in_doorway = false;            // measured from inside the slab (nearest returns beside the body)
    bool wall_fitted = false;           // the line was refitted from the wall's own returns
};

struct DoorwayGapPrior
{
    Eigen::Vector2f centre = Eigen::Vector2f::Zero();   // a point on the wall line: the expected opening centre, robot frame
    Eigen::Vector2f normal{0.f, 1.f};                   // the wall normal, oriented along the crossing (unit)
    Eigen::Matrix3f P = Eigen::Matrix3f::Identity();    // covariance of (x, y, yaw) — the anchor's, or the carried gap's
    float width = 0.9f;                                 // the door's full opening width
};

// The gap's yaw in the anchor convention (controller_anchor_servo.h): n = (−sin ψ, cos ψ), a = (cos ψ, sin ψ) = (n.y, −n.x).
[[nodiscard]] inline float doorway_gap_yaw(const Eigen::Vector2f& n) { return std::atan2(-n.x(), n.y()); }

// The measurement's own covariance on (x, y, yaw): each edge carries σ_r, the centre is their mean, the yaw their difference
// over the width.
[[nodiscard]] inline Eigen::Matrix3f doorway_gap_meas_cov(float width, float sigma_r)
{
    const float s = std::max(sigma_r, 1e-3f);
    Eigen::Matrix3f R = Eigen::Matrix3f::Zero();
    R(0, 0) = R(1, 1) = 0.5f * s * s;
    const float sy = std::sqrt(2.f) * s / std::max(width, 0.1f);
    R(2, 2) = sy * sy;
    return R;
}

// The gap as an anchor-style robot-frame pose (origin = the clear gap's centre, +y = its normal), so the anchor servo and
// guards run on it unchanged.
[[nodiscard]] inline rc::interaction::RelPose doorway_gap_pose(const DoorwayGap& g, float sigma_r)
{
    rc::interaction::RelPose r;
    r.x = {g.centre.x(), g.centre.y(), doorway_gap_yaw(g.normal)};
    r.P = doorway_gap_meas_cov(g.width, sigma_r);
    return r;
}

// The nearest return per bearing bin (bearing = atan2(x, y), 0 forward, + right), any range; nullopt = no return.
[[nodiscard]] inline std::vector<std::optional<Eigen::Vector2f>> doorway_scan(std::span<const Eigen::Vector2f> pts)
{
    std::vector<std::optional<Eigen::Vector2f>> bin(kGapScanBins);
    const float two_pi = 2.f * std::numbers::pi_v<float>;
    for (const auto& q : pts)
    {
        if (not q.allFinite()) continue;
        const float r = q.norm();
        if (r < 1e-4f) continue;
        float b = std::atan2(q.x(), q.y());
        if (b < 0.f) b += two_pi;
        const auto k = std::size_t(std::clamp(int(b / two_pi * float(kGapScanBins)), 0, kGapScanBins - 1));
        if (not bin[k].has_value() or r < bin[k]->norm()) bin[k] = q;
    }
    return bin;
}
[[nodiscard]] inline Eigen::Vector2f doorway_bin_dir(std::size_t k)
{
    const float b = 2.f * std::numbers::pi_v<float> * (float(k) + 0.5f) / float(kGapScanBins);
    return {std::sin(b), std::cos(b)};
}

namespace detail
{
struct GapLine { Eigen::Vector2f m, n, a; };   // a point on the line, its unit normal (crossing direction), a = (n.y, −n.x)
inline GapLine gap_line(const Eigen::Vector2f& m, const Eigen::Vector2f& n)
{
    const Eigen::Vector2f nn = n.normalized();
    return {m, nn, Eigen::Vector2f(nn.y(), -nn.x())};
}

struct GapSpan { float tl, tr; };   // along-wall coordinates (t = (q − m)·a) of the two edges

// Openings by through-rays, the robot clear of the slab. `s` = the robot's offset along n (< 0 before the line).
inline std::vector<GapSpan> gap_through_spans(const std::vector<std::optional<Eigen::Vector2f>>& scan,
                                              std::span<const Eigen::Vector2f> slab_pts, const GapLine& L, float s, float depth)
{
    std::vector<GapSpan> out;
    const Eigen::Vector2f beyond = s < 0.f ? L.n : Eigen::Vector2f(-L.n);
    enum class St : unsigned char { X, U, T, B };   // not crossing, unknown, through, blocked
    const std::size_t N = scan.size();
    std::vector<St> st(N, St::X);
    std::size_t start = N;
    for (std::size_t k = 0; k < N; ++k)
    {
        const Eigen::Vector2f u = doorway_bin_dir(k);
        if (u.dot(beyond) <= 0.f) { if (start == N) start = k; continue; }
        if (not scan[k].has_value()) { st[k] = St::U; continue; }
        st[k] = ((*scan[k] - L.m).dot(beyond) > depth) ? St::T : St::B;
    }
    if (start == N) return out;   // every bin crosses (cannot happen off the line) — no ordering anchor
    // Walk the crossing interval once, in bearing order, from a non-crossing bin.
    std::optional<std::size_t> last_b;
    bool in_run = false, run_open = false;   // run_open: the run started with no blocked bin before it (unbounded)
    std::optional<std::size_t> run_lo;
    for (std::size_t i = 1; i <= N; ++i)
    {
        const std::size_t k = (start + i) % N;
        switch (st[k])
        {
        case St::X:
            in_run = false; last_b.reset(); break;
        case St::U:
            break;
        case St::T:
            if (not in_run) { in_run = true; run_open = not last_b.has_value(); run_lo = last_b; }
            break;
        case St::B:
            if (in_run and not run_open and run_lo.has_value())
            {
                float tl = (*scan[*run_lo] - L.m).dot(L.a), tr = (*scan[k] - L.m).dot(L.a);
                if (tl > tr) std::swap(tl, tr);
                out.push_back({tl, tr});
            }
            in_run = false;
            last_b = k;
            break;
        }
    }
    // Narrow each span by any return inside the slab between its edges (a leaf face on the robot's side).
    for (auto& sp : out)
    {
        const float tc = 0.5f * (sp.tl + sp.tr);
        for (const auto& q : slab_pts)
        {
            const Eigen::Vector2f d = q - L.m;
            if (std::abs(d.dot(L.n)) >= depth) continue;
            const float t = d.dot(L.a);
            if (t <= sp.tl or t >= sp.tr) continue;
            if (t < tc) sp.tl = t; else sp.tr = t;
        }
    }
    return out;
}

// Inside the slab: the nearest returns in it on each side of the robot's projection on the line.
inline std::optional<GapSpan> gap_in_doorway_span(std::span<const Eigen::Vector2f> pts, const GapLine& L, float depth)
{
    const float t0 = (Eigen::Vector2f::Zero() - L.m).dot(L.a);
    float tl = -std::numeric_limits<float>::infinity(), tr = std::numeric_limits<float>::infinity();
    for (const auto& q : pts)
    {
        const Eigen::Vector2f d = q - L.m;
        if (std::abs(d.dot(L.n)) >= depth) continue;
        const float t = d.dot(L.a);
        if (t < t0) tl = std::max(tl, t); else tr = std::min(tr, t);
    }
    if (not std::isfinite(tl) or not std::isfinite(tr)) return std::nullopt;
    return GapSpan{tl, tr};
}

// The wall line refitted from the returns ON it, outside the opening's edges. Three explanations of those returns compete:
// the prior's own line, and a total-least-squares line through each side's stretch fitted on its own (an ajar leaf sweeps
// across its hinge side: two lines there, a large residual; a leaf at 90° is a straight line of its own, PERPENDICULAR to the
// wall). Each is charged its residual in range-noise units plus how unlikely its direction is under the prior (the prior's
// yaw σ, widened by the anchor's gross-error scale kGapGrossYawRad); the cheapest is kept. nullopt = the prior's line wins
// or too few returns on either side.
inline std::optional<GapLine> gap_fit_wall(std::span<const Eigen::Vector2f> pts, const GapLine& L, const GapSpan& sp, float depth,
                                           float sigma_r, float var_yaw)
{
    std::array<std::vector<Eigen::Vector2f>, 2> side;
    for (const auto& q : pts)
    {
        const Eigen::Vector2f d = q - L.m;
        if (std::abs(d.dot(L.n)) >= depth) continue;
        const float t = d.dot(L.a);
        if (t < sp.tl - kGapWallFitNearM and t > sp.tl - kGapWallFitFarM) side[0].push_back(q);
        else if (t > sp.tr + kGapWallFitNearM and t < sp.tr + kGapWallFitFarM) side[1].push_back(q);
    }
    const float s2 = std::max(sigma_r * sigma_r, 1e-6f);
    const float v_ang = std::max(0.f, var_yaw) + kGapGrossYawRad * kGapGrossYawRad;
    // The prior's line: its mean squared residual over every wall return.
    float best_cost = std::numeric_limits<float>::infinity();
    {
        float ss = 0.f; int cnt = 0;
        for (const auto& w : side) for (const auto& q : w) { const float r = (q - L.m).dot(L.n); ss += r * r; ++cnt; }
        if (cnt >= kGapWallFitMinPts) best_cost = ss / float(cnt) / s2;
    }
    std::optional<GapLine> best;
    for (const auto& w : side)
    {
        if (int(w.size()) < kGapWallFitMinPts) continue;
        Eigen::Vector2f mean = Eigen::Vector2f::Zero();
        for (const auto& q : w) mean += q;
        mean /= float(w.size());
        Eigen::Matrix2f C = Eigen::Matrix2f::Zero();
        for (const auto& q : w) C += (q - mean) * (q - mean).transpose();
        const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> es(C);
        if (es.info() != Eigen::Success) continue;
        Eigen::Vector2f n = es.eigenvectors().col(0);   // the smallest eigenvalue's vector: the normal
        if (n.dot(L.n) < 0.f) n = -n;
        const float dpsi = std::atan2(L.n.x() * n.y() - L.n.y() * n.x(), L.n.dot(n));
        const float cost = std::max(0.f, es.eigenvalues()(0)) / float(w.size()) / s2 + dpsi * dpsi / v_ang;
        if (cost >= best_cost) continue;
        best_cost = cost;
        // Keep the opening's along-wall origin: the new line's point nearest the old one.
        best = gap_line(L.m - n * (L.m - mean).dot(n), n);
    }
    return best;
}
}  // namespace detail

[[nodiscard]] inline DoorwayGap measure_doorway_gap(std::span<const Eigen::Vector2f> returns,
                                                    const std::optional<DoorwayGapPrior>& prior, float body_half_width,
                                                    float sigma_r, float depth)
{
    DoorwayGap best;
    if (not prior.has_value() or not prior->centre.allFinite() or not prior->normal.allFinite()
        or not (prior->normal.squaredNorm() > 0.f) or not (depth > 0.f) or not (body_half_width > 0.f))
        return best;
    const auto scan = doorway_scan(returns);
    std::vector<Eigen::Vector2f> pts;
    pts.reserve(returns.size());
    for (const auto& b : scan) if (b.has_value()) pts.push_back(*b);
    if (pts.empty()) return best;

    const float sw = std::sqrt(kGapWidthSigma * kGapWidthSigma + 2.f * sigma_r * sigma_r);
    const Eigen::Matrix2f Pxy = prior->P.topLeftCorner<2, 2>();
    const auto score_of = [&](const Eigen::Vector2f& c, float w)
    {
        const Eigen::Vector2f d = c - prior->centre;
        const Eigen::Matrix2f S = Pxy + doorway_gap_meas_cov(w, sigma_r).topLeftCorner<2, 2>();
        const auto m2 = [&](const Eigen::Matrix2f& C)
        {
            const Eigen::LDLT<Eigen::Matrix2f> ldlt(C);
            return ldlt.info() == Eigen::Success ? d.dot(ldlt.solve(d)) : std::numeric_limits<float>::infinity();
        };
        const float s_loc = (1.f - kGapPriorGross) * std::exp(-0.5f * m2(S))
                          + kGapPriorGross * std::exp(-0.5f * m2(S + kGapGrossXYM * kGapGrossXYM * Eigen::Matrix2f::Identity()));
        const float over = std::max(0.f, w - prior->width) / sw;
        return s_loc * std::exp(-0.5f * over * over);
    };
    const auto pick = [&](const detail::GapLine& L, std::span<const detail::GapSpan> spans, bool in_doorway) -> DoorwayGap
    {
        DoorwayGap g;
        g.candidates = int(spans.size());
        for (const auto& sp : spans)
        {
            const float w = sp.tr - sp.tl;
            if (not (w > 0.f)) continue;
            const Eigen::Vector2f c = L.m + L.a * (0.5f * (sp.tl + sp.tr));
            if (const float sc = score_of(c, w); sc > g.score)
            {
                g.score = sc;
                g.width = w;
                g.centre = c;
                g.left = L.m + L.a * sp.tl;
                g.right = L.m + L.a * sp.tr;
                g.normal = L.n;
            }
        }
        g.in_doorway = in_doorway;
        g.seen = g.score >= kGapScoreFloor;
        return g;
    };

    detail::GapLine L = detail::gap_line(prior->centre, prior->normal);
    for (int pass = 0; pass < 2; ++pass)
    {
        const float s = (Eigen::Vector2f::Zero() - L.m).dot(L.n);
        if (std::abs(s) < depth)
        {
            // In the doorway: the line as given (the last fit, carried), the nearest returns beside the body.
            const auto sp = detail::gap_in_doorway_span(pts, L, depth);
            if (not sp.has_value()) { best.in_doorway = true; return best; }
            const std::array<detail::GapSpan, 1> one{*sp};
            auto g = pick(L, one, true);
            g.wall_fitted = best.wall_fitted;
            return g;
        }
        const auto spans = detail::gap_through_spans(scan, pts, L, s, depth);
        auto g = pick(L, spans, false);
        g.wall_fitted = best.wall_fitted;
        best = g;
        if (pass == 1 or not g.seen) break;
        // Refit the wall from its own returns outside the chosen opening, and measure once more on that line.
        const detail::GapSpan chosen{(g.left - L.m).dot(L.a), (g.right - L.m).dot(L.a)};
        const auto fitted = detail::gap_fit_wall(pts, L, chosen, depth, sigma_r, prior->P(2, 2));
        if (not fitted.has_value()) break;
        L = *fitted;
        best.wall_fitted = true;
    }
    if (best.wall_fitted and not best.seen) best.wall_fitted = false;
    return best;
}

// Carry a gap line (robot-frame pose) by the robot's own motion: the robot moved (side, adv) in its previous frame and
// turned dth. The covariance grows with the motion: the turn by Shadow's measured rotation-scale error
// (relative_pose.h kRotScaleSigma), the translation by ⚑ kGapCarryTransSigma of the distance (an assumption — the wheel
// scale error is not measured; it is the same order as the rotation's).
inline constexpr float kGapCarryTransSigma = 0.05f;
[[nodiscard]] inline rc::interaction::RelPose doorway_gap_carry(const rc::interaction::RelPose& r, float side, float adv, float dth)
{
    rc::interaction::RelPose o = r;
    o.x = rc::interaction::move_mean(r.x, side, adv, dth);
    Eigen::Matrix3f J, Gm;
    rc::interaction::predict_jacobians(r.x, side, adv, dth, J, Gm);
    const float st = kGapCarryTransSigma * std::hypot(side, adv), sr = rc::interaction::kRotScaleSigma * std::abs(dth);
    const Eigen::Vector3f var(st * st, st * st, sr * sr);   // columns of G: adv, side, rot
    o.P = J * r.P * J.transpose() + Gm * var.asDiagonal() * Gm.transpose();
    return o;
}

}  // namespace rc
