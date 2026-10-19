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
 * SIGN. At acquisition (an anchor prior), with the robot clear of the slab, the normal is oriented away from the robot, so the
 * robot is on the near side — the prior's own sign is not trusted (try89). A measured prior keeps its sign; so does an
 * acquisition from inside the doorway, where geometry cannot say which way is away.
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
 *   bridged) bounded by a BLOCKED bin on both sides are the candidate openings; each edge lies between that blocked return
 *   and where the first ray that passed crosses the return's depth, and is taken at their midpoint (try86 fix: with sparse
 *   beams the blocked return alone can sit several cm outside the jamb), measured ALONG the wall (a = the line's direction). The span is then narrowed by any return between the edges on the
 *   body's way through — from the robot's own depth to the slab beyond the line (or a leaf's reach, the door's width): a leaf
 *   face on the robot's side (test c), an ajar leaf's free edge (tests f, f2).
 *   The WALL LINE is then refitted from the blocked returns 0.1–1.0 m outside each edge: the prior's line and a total-least-
 *   squares line per side compete on SUPPORT (returns within 2σ_r) × the likelihood of their direction under the prior's yaw
 *   (an anchor's widened by the gross scale; a measured gap's not) — a leaf at 90° is a line perpendicular to the wall, an
 *   ajar leaf makes its side two lines; both lose — and the measurement is repeated once on the winner.
 * IN THE DOORWAY (|s| < depth: the jambs beside the body, no ray crosses the line usefully): the line is the prior's (the last
 *   fit, carried); the edges are the nearest returns inside the slab on each side of the robot's own projection on it.
 *
 * WHICH OPENING (a second opening further along the wall): the one whose centre is likeliest under the prior —
 *   s_loc = (1−ε)·exp(−½ dᵀS⁻¹d) + ε·exp(−½ dᵀ(S+G)⁻¹d),  d = c − μ,  S = P_xy + R_meas
 *   (a robust mixture: the anchor is right, or GROSSLY wrong — without the second term a confidently wrong anchor, try74's
 *   12σ, would make the true door "unseen"; a MEASURED prior — a gap this claim already holds — has no gross branch), times
 *   the width: wider than the door is not the door, s_w = exp(−½((w − w_door)₊/σ_w)²), and an opening is a doorway to steer
 *   on only as far as the body fits it, P_fit = Φ((w − 2·half_w)/σ_w) (try86: 1–5 cm slivers scored 0.98 under the one-sided
 *   term). A half-open door wider than the body is seen and the servo slows on its measured width; a slit is not seen.
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
    // true = the prior is a gap this claim already MEASURED (carried): no gross-error branch — neither its place nor its
    // line may jump to another opening or another wall (try86: chained re-associations walked the line 70° round the room).
    // false = the anchor: its place may be grossly wrong (the gross branch), its wall's yaw too (refit widened).
    bool measured = false;
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
                                              std::span<const Eigen::Vector2f> slab_pts, const GapLine& L, float s, float depth,
                                              float door_width)
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
    std::size_t first_t = 0, last_t = 0;
    // Where a through ray passes the depth (along `beyond`) of a blocked return, along the wall: the obstacle's edge is
    // between that return and this point, at the same depth (on a thick wall seen obliquely, the line's own crossing would
    // sit inward of the jamb's far corner).
    const auto t_pass = [&](std::size_t k_through, std::size_t k_blocked)
    {
        const Eigen::Vector2f u = doorway_bin_dir(k_through);
        const float dd = (*scan[k_blocked] - L.m).dot(beyond);
        const float lambda = (dd + L.m.dot(beyond)) / u.dot(beyond);
        return (lambda * u - L.m).dot(L.a);
    };
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
            if (not in_run) { in_run = true; run_open = not last_b.has_value(); run_lo = last_b; first_t = k; }
            last_t = k;
            break;
        case St::B:
            if (in_run and not run_open and run_lo.has_value())
            {
                // Each edge lies between the last blocked return and the first ray that passed (try86 fix: with sparse beams
                // and dropped returns the blocked return can sit several cm OUTSIDE the jamb) — its midpoint, along the wall.
                float t0 = 0.5f * ((*scan[*run_lo] - L.m).dot(L.a) + t_pass(first_t, *run_lo));
                float t1 = 0.5f * ((*scan[k] - L.m).dot(L.a) + t_pass(last_t, k));
                if (t0 > t1) std::swap(t0, t1);
                out.push_back({t0, t1});
            }
            in_run = false;
            last_b = k;
            break;
        }
    }
    // Narrow each span by any return between its edges on the body's way through: from the robot's own depth (it drives
    // the whole approach along the normal — an ajar leaf reaching 0.9 m toward it narrows the passage over all of that) to
    // the slab beyond the line, or a leaf's reach (the door's width) if longer.
    for (auto& sp : out)
    {
        const float tc = 0.5f * (sp.tl + sp.tr);
        for (const auto& q : slab_pts)
        {
            const Eigen::Vector2f d = q - L.m;
            const float dd = d.dot(beyond);
            if (dd <= -std::abs(s) or dd >= std::max(depth, door_width)) continue;
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

// The wall line refitted from the returns ON it, outside the opening's edges (0.1–1.0 m beyond each). Candidate lines: the
// prior's own, and a total-least-squares line through each side's stretch (refitted once on its own inliers). Each is scored
// by its SUPPORT — how many of those returns lie within 2σ_r of it — times how likely its direction is under the prior's yaw
// (variance `var_yaw`): support × exp(−½Δψ²/var_yaw). A leaf at 90° is a straight line of its own but perpendicular to the
// wall (try86: the refit took the 81° leaf, and every later cycle drifted further); an ajar leaf makes its side two lines and
// little support. The best is kept; nullopt = the prior's line wins or too few returns.
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
    const float band = 2.f * std::max(sigma_r, 1e-3f);
    const float v_ang = std::max(var_yaw, 1e-6f);
    const auto support = [&](const Eigen::Vector2f& m, const Eigen::Vector2f& n)
    {
        int c = 0;
        for (const auto& w : side) for (const auto& q : w) if (std::abs((q - m).dot(n)) < band) ++c;
        return c;
    };
    const auto tls = [](const std::vector<Eigen::Vector2f>& w, Eigen::Vector2f& mean, Eigen::Vector2f& n)
    {
        if (w.size() < 2) return false;
        mean.setZero();
        for (const auto& q : w) mean += q;
        mean /= float(w.size());
        Eigen::Matrix2f C = Eigen::Matrix2f::Zero();
        for (const auto& q : w) C += (q - mean) * (q - mean).transpose();
        const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> es(C);
        if (es.info() != Eigen::Success) return false;
        n = es.eigenvectors().col(0);   // the smallest eigenvalue's vector: the normal
        return true;
    };
    float best_score = float(support(L.m, L.n));   // the prior's line: Δψ = 0
    std::optional<GapLine> best;
    for (const auto& w : side)
    {
        if (int(w.size()) < kGapWallFitMinPts) continue;
        Eigen::Vector2f mean, n;
        if (not tls(w, mean, n)) continue;
        std::vector<Eigen::Vector2f> in;   // one refit on its own inliers
        for (const auto& q : w) if (std::abs((q - mean).dot(n)) < band) in.push_back(q);
        if (int(in.size()) < kGapWallFitMinPts or not tls(in, mean, n)) continue;
        if (n.dot(L.n) < 0.f) n = -n;
        const float dpsi = std::atan2(L.n.x() * n.y() - L.n.y() * n.x(), L.n.dot(n));
        const float score = float(support(mean, n)) * std::exp(-0.5f * dpsi * dpsi / v_ang);
        if (score <= best_score or support(mean, n) < kGapWallFitMinPts) continue;
        best_score = score;
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
        const float gross = prior->measured ? 0.f : kGapPriorGross;
        const float s_loc = (1.f - gross) * std::exp(-0.5f * m2(S))
                          + gross * std::exp(-0.5f * m2(S + kGapGrossXYM * kGapGrossXYM * Eigen::Matrix2f::Identity()));
        // Width, two-sided (try86 fix): wider than the door is not the door; and an opening is a DOORWAY to steer on only as
        // far as the body fits through it — P(fit) = Φ((w − body width)/σ_w), the servo's own P_fit form. A 4-cm sliver
        // between two beams scores Φ(−4.4) ≈ 5e-6; a half-open door wider than the body is seen, and the servo slows on it.
        const float over = std::max(0.f, w - prior->width) / sw;
        const float p_fit = 0.5f * std::erfc(-((w - 2.f * body_half_width) / sw) * 0.70710678f);
        return s_loc * std::exp(-0.5f * over * over) * p_fit;
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
    // try89 fix: at ACQUISITION the crossing direction is AWAY from the robot — the robot is on the near side (s < 0),
    // whatever sign the prior's normal carries (a leftover anchor from the previous leg had it flipped: s +1.78, 176°, and
    // the cross completed "through" without crossing). The prior says where to look and which gap, never the sign. A held
    // (measured) gap keeps its sign: the robot legitimately ends on its far side.
    // Only with the robot CLEAR of the slab (|s| ≥ depth): standing in the doorway, geometry does not say which way is
    // "away", and flipping on a few cm of s would send a retry back the way it came — the prior's sign stands there (and a
    // claim that starts there cannot complete as through: CrossThroughGate).
    if (const float s0 = (Eigen::Vector2f::Zero() - L.m).dot(L.n); not prior->measured and s0 >= depth)
        L = detail::gap_line(L.m, -L.n);
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
        const auto spans = detail::gap_through_spans(scan, pts, L, s, depth, prior->width);
        auto g = pick(L, spans, false);
        g.wall_fitted = best.wall_fitted;
        best = g;
        if (pass == 1 or not g.seen) break;
        // Refit the wall from its own returns outside the chosen opening, and measure once more on that line.
        const detail::GapSpan chosen{(g.left - L.m).dot(L.a), (g.right - L.m).dot(L.a)};
        // The yaw variance the refit may move within: the prior's own; an anchor's widened by its gross-error scale.
        const float var_yaw = std::max(0.f, prior->P(2, 2)) + doorway_gap_meas_cov(g.width, sigma_r)(2, 2)
                            + (prior->measured ? 0.f : kGapGrossYawRad * kGapGrossYawRad);
        const auto fitted = detail::gap_fit_wall(pts, L, chosen, depth, sigma_r, var_yaw);
        if (not fitted.has_value()) break;
        L = *fitted;
        best.wall_fitted = true;
    }
    if (best.wall_fitted and not best.seen) best.wall_fitted = false;
    return best;
}

// ── THE WALL'S DIRECTION FROM THE ROOM FRAME (try105 leg 5) ─────────────────────────────────────────────────────────────
// WHY. In room_1's corridor stub below door_1 (the open leaf runs 0.95 m along the corridor's west side; the corridor's east
// wall meets the door wall 7.5 cm past the east jamb) no stretch of the door's WALL is visible on the approach: the leaf hides
// the west side, and the corridor wall is perpendicular to it. gap_fit_wall then has nothing to refit on and the line keeps the
// PRIOR's yaw — the anchor's, which door_concept registered 13–20° off there (try105 leg 5: normal at +47.7° from the robot's
// forward against +61° true; −27.8° against −8°) — and once held, that yaw is a measurement (no gross branch) that nothing
// corrects. The servo drove the skewed normal: 0.15 m east of the opening at the mouth, its front-right corner on the east
// casing, the gap "0.6 m behind" (the line also sat 0.8 m short, at the leaf's free edge), three cross attempts lost.
// The ROOM frame knows the wall's DIRECTION well — a polygon edge, seen through the robot's heading (the localiser's σθ,
// ~1°) — though it may be decimetres off in POSITION, which is why the anchor exists. So the room gives the yaw, only the yaw.
// MODEL. The opening lies on one edge k of the room polygon (robot frame). Which one: the edge likeliest under the prior —
// its distance from the prior's centre under the prior's position σ widened by the gap's gross scale, times its direction
// under the prior's yaw σ widened by the gross yaw scale (a corridor's side wall, perpendicular to the door's, scores ~0).
// Its yaw ψ_k (n_k oriented like the prior's normal) measures the wall's yaw with variance σθ² + (√2·σ_c/L_k)² (the robot's
// heading, and the edge's own: two corners σ_c apart over its length L_k). It is fused with the prior through the gap's own
// robust mixture — the anchor right (1 − kGapPriorGross, its claimed P) or grossly wrong (kGapPriorGross, P + G²): a 15°
// disagreement with a claimed 1° is the gross branch, and the posterior follows the room. A MEASURED prior (a held gap) is
// not fused again: its yaw already carries the room's, and the room's error is one error, not fresh noise every cycle.
//   ⚑ kRoomCornerSigmaM = 0.02 m: the room polygon's corner σ (room_concept's published corner floor) — a model constant.
inline constexpr float kRoomCornerSigmaM = 0.02f;
struct DoorwayWallYaw
{
    float psi = 0.f;   // the wall normal's yaw in the anchor convention (doorway_gap_yaw), oriented like the prior's normal
    float var = 0.f;   // its variance (rad²)
    int edge = -1;     // the polygon edge it came from
};
// `polygon_robot`: the room polygon in the ROBOT frame (x right, y forward); `var_heading`: the localiser's heading variance.
[[nodiscard]] inline std::optional<DoorwayWallYaw> doorway_room_wall_yaw(const DoorwayGapPrior& prior,
                                                                         std::span<const Eigen::Vector2f> polygon_robot,
                                                                         float var_heading)
{
    if (polygon_robot.size() < 3 or not prior.centre.allFinite() or not prior.normal.allFinite()
        or not (prior.normal.squaredNorm() > 0.f) or not std::isfinite(var_heading) or var_heading < 0.f)
        return std::nullopt;
    const Eigen::Vector2f n0 = prior.normal.normalized();
    const float psi0 = doorway_gap_yaw(n0);
    const Eigen::Matrix2f Pxy = prior.P.topLeftCorner<2, 2>();
    const float var_yaw = std::max(0.f, prior.P(2, 2)) + kGapGrossYawRad * kGapGrossYawRad;
    std::optional<DoorwayWallYaw> best;
    float best_w = 0.f;
    for (std::size_t k = 0; k < polygon_robot.size(); ++k)
    {
        const Eigen::Vector2f a = polygon_robot[k], b = polygon_robot[(k + 1) % polygon_robot.size()];
        const Eigen::Vector2f ab = b - a;
        const float len = ab.norm();
        if (not (len > 1e-3f) or not ab.allFinite()) continue;
        const Eigen::Vector2f u = ab / len;
        Eigen::Vector2f nk(-u.y(), u.x());
        if (nk.dot(n0) < 0.f) nk = -nk;
        const float t = std::clamp((prior.centre - a).dot(u), 0.f, len);
        const float dist = (prior.centre - (a + t * u)).norm();
        const float var_loc = std::max(0.f, nk.dot(Pxy * nk)) + kGapGrossXYM * kGapGrossXYM;
        const float psi_k = doorway_gap_yaw(nk);
        const float dpsi = rc::interaction::wrap_pi(psi_k - psi0);
        const float w = std::exp(-0.5f * dist * dist / var_loc) * std::exp(-0.5f * dpsi * dpsi / var_yaw);
        if (w > best_w)
        {
            best_w = w;
            const float se = std::sqrt(2.f) * kRoomCornerSigmaM / len;
            best = DoorwayWallYaw{psi_k, var_heading + se * se, int(k)};
        }
    }
    return best;
}
// The prior's yaw fused with the room's (the robust mixture above); its centre stays where it was.
[[nodiscard]] inline DoorwayGapPrior doorway_gap_fuse_wall_yaw(DoorwayGapPrior prior, const DoorwayWallYaw& w)
{
    if (prior.measured or not std::isfinite(w.psi) or not (w.var > 0.f)) return prior;
    const float psi0 = doorway_gap_yaw(prior.normal.normalized());
    const float d = rc::interaction::wrap_pi(w.psi - psi0);
    const float v_right = std::max(0.f, prior.P(2, 2));
    const std::array<std::pair<float, float>, 2> comps{std::pair{1.f - kGapPriorGross, v_right},
                                                       std::pair{kGapPriorGross, v_right + kGapGrossYawRad * kGapGrossYawRad}};
    std::array<float, 2> lw{}, mu{}, var{};
    float lmax = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < 2; ++i)
    {
        const float S = comps[i].second + w.var;
        lw[i] = std::log(comps[i].first) - 0.5f * d * d / S - 0.5f * std::log(S);
        const float K = comps[i].second / S;
        mu[i] = K * d;
        var[i] = comps[i].second * (1.f - K);
        lmax = std::max(lmax, lw[i]);
    }
    float z = 0.f, mean = 0.f, m2 = 0.f;
    for (std::size_t i = 0; i < 2; ++i) z += std::exp(lw[i] - lmax);
    for (std::size_t i = 0; i < 2; ++i)
    {
        const float p = std::exp(lw[i] - lmax) / z;
        mean += p * mu[i];
        m2 += p * (var[i] + mu[i] * mu[i]);
    }
    const float psi = psi0 + mean;
    prior.normal = Eigen::Vector2f(-std::sin(psi), std::cos(psi));
    prior.P(2, 2) = std::max(0.f, m2 - mean * mean);
    prior.P(0, 2) = prior.P(2, 0) = prior.P(1, 2) = prior.P(2, 1) = 0.f;
    return prior;
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


// The carry the session runs, from two consecutive ROOM-frame robot poses (pos, θ; robot frame x right, y forward, room =
// R(θ)·q + pos): the motion in the previous robot frame is d = R(θ_prev)ᵀ(pos − pos_prev), dθ = θ − θ_prev. Only within ONE
// room frame: a room switch re-expresses the pose (try90: room_1 → room_2 at s +0.3 m), and that cycle's "motion" is a frame
// change, not a displacement — dropped (same_room = false). The pure form of ControllerSession::update_doorway_gap's carry.
[[nodiscard]] inline rc::interaction::RelPose doorway_gap_carry_poses(const rc::interaction::RelPose& r,
                                                                     const Eigen::Vector2f& pos_prev, float th_prev,
                                                                     const Eigen::Vector2f& pos, float th, bool same_room)
{
    if (not same_room) return r;
    const Eigen::Vector2f d = Eigen::Rotation2Df(th_prev).toRotationMatrix().transpose() * (pos - pos_prev);
    const float dth = rc::interaction::wrap_pi(th - th_prev);
    if (not d.allFinite() or not std::isfinite(dth)) return r;
    return doorway_gap_carry(r, d.x(), d.y(), dth);
}

}  // namespace rc
