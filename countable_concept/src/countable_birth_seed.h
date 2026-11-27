/*
 * countable_birth_seed.h — the birth's seed placement (F1 round 9 a): the cold-start footprint, or the back on the nearest wall
 * for a class that declares a back-face attachment (rc::birth_on_wall). Pure (no DSR): the fitter calls rc::seed_birth with its
 * nearest-wall query, and birth_seed_rule_test calls it on synthetic clouds.
 */
#pragma once

#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <algorithm>
#include <print>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "countable_belief.h"
#include "countable_birth_footprint.h"
#include "countable_config.h"
#include "countable_vertical_span.h"
#include "../../common/ai_belief/parts_geometry.h"

namespace rc {

using NearestWall = std::function<WallRef(const Eigen::Vector2f&)>;

// BIRTH ON THE PRIORS. A wall-standing object's first view is its FRONT; the principal-axis yaw of that view is
// noise on a near-square footprint and the depth is unseen. So the birth takes what the priors already know
// (spec 7c.8 rev 3.3.14, rc::birth_footprint):
//  · the wall: the one nearest the cloud (the class prior says it stands against one — WallFlushPrior);
//  · yaw: back face onto that wall, front into the room (local +Y = back = −n);
//  · width/depth: the CLASS MEANS. The cloud's extent is not the object's: a mask that took in a neighbour
//    (live14fixA: 2.46 m for a 0.51 m nightstand) seeded a 2.13 m-deep box under the old depth-band rule;
//  · centre: the cloud's middle along the wall, half the class depth in from it;
//  · the cloud's extent widens the first update's POSITION covariance instead (F1.3): along the wall (w̄ − E)²/12 for
//    either sign (a cloud wider than the class, or one clipped/partial and shorter); across it max(0, E − h̄)²/12 (the
//    centre across is set by the wall, so only a cloud reaching deeper than the class says the object may stand off);
//  · H: NOT here — re-derived after the belief is built (CountableBelief::seed_height, F1 round 6.2: the class-prior MAP
//    with the top of the points the box explains, box-vs-headboard shares at this born pose).
// Not a gate on the fit: the belief starts here and the data then moves and sizes it.
// The wall seed's ingredients: the banded cloud's moment centre, the wall nearest it, and the footprint on that wall.
struct WallSeed { rc::BirthFootprint bf; WallRef w; Eigen::Vector2f c = Eigen::Vector2f::Zero(); };
inline std::optional<WallSeed> wall_seed(const std::vector<Eigen::Vector3f>& pts, float z_lo, float z_hi, const CountableConfig& cfg,
                                         const NearestWall& nearest_wall)
{
    std::vector<Eigen::Vector2f> xy;
    for (const auto& p : pts) if (p.z() >= z_lo and p.z() <= z_hi) xy.emplace_back(p.x(), p.y());
    if (xy.size() < 10) return std::nullopt;
    Eigen::Vector2f c = Eigen::Vector2f::Zero();
    for (const auto& q : xy) c += q;
    c /= static_cast<float>(xy.size());
    const WallRef w = nearest_wall(c);
    if (not w.ok) return std::nullopt;
    const float prior_d = cfg.ai2_prior_depth_m > 0.0f ? cfg.ai2_prior_depth_m : cfg.ai2_prior_footprint_m;
    const rc::BirthFootprint bf = rc::birth_footprint(xy, w.p, w.n, cfg.ai2_prior_footprint_m, prior_d, 0.0f);   // the EXCESS; the belief adds its σ_pos²
    if (not bf.ok) return std::nullopt;
    return WallSeed{bf, w, c};
}

inline void apply_wall_seed(CountableBeliefState& s0, const WallSeed& ws, const std::string& name)
{
    const auto& bf = ws.bf;
    s0.w = bf.w; s0.h = bf.h; s0.yaw = bf.yaw;
    s0.cx = bf.centre.x(); s0.cy = bf.centre.y();
    std::print("[{}] birth on priors: wall n=({:.2f},{:.2f}) → yaw={:.1f}° w={:.2f} h={:.2f} (class means) centre ({:.2f},{:.2f}) "
               "[cloud centre ({:.2f},{:.2f}), extent along {:.2f} across {:.2f} → extra pos std along {:.3f} across {:.3f}]\n",
               name, ws.w.n.x(), ws.w.n.y(), s0.yaw * 57.29578f, s0.w, s0.h, s0.cx, s0.cy, ws.c.x(), ws.c.y(), bf.extent_t, bf.extent_n,
               std::sqrt(bf.var_t), std::sqrt(bf.var_n));
}

inline std::optional<Eigen::Matrix2f> birth_on_priors(CountableBeliefState& s0, const std::vector<Eigen::Vector3f>& pts, float z_lo,
                                                      float z_hi, const CountableConfig& cfg, const NearestWall& nearest_wall,
                                                      const std::string& name)
{
    const auto ws = wall_seed(pts, z_lo, z_hi, cfg, nearest_wall);
    if (not ws) return std::nullopt;
    apply_wall_seed(s0, *ws, name);
    return ws->bf.pos_cov;
}

// log N(r; 0, v). v = 0 is a point mass (+inf at r = 0, −inf elsewhere): the arithmetic of a degenerate variance, not a cutoff.
inline double log_normal_1d(double r, double v)
{
    if (not (v > 0.0)) return r == 0.0 ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity();
    return -0.5 * r * r / v - 0.5 * std::log(2.0 * M_PI * v);
}

// ── The first view (CB1): what the birth knows about HOW the cloud was seen ───────────────────────────────────────────────────
// The camera origin at the mask's capture stamp (room frame), the robot's pose covariance there (x, y, ψ: the robot→room edge's
// rt_covariance, CountableFitter::pose_at_stamp) and the room's extent along the wall normal through the cloud (wall to the
// opposite boundary of the room polygon). have_cam = false ⇒ nothing to render: the comparison is BASE's (F2), bit for bit.
struct BirthView
{
    bool            have_cam = false;
    Eigen::Vector3f cam = Eigen::Vector3f::Zero();
    bool            have_pose = false;
    Eigen::Matrix3d sigma_pose = Eigen::Matrix3d::Zero();
    Eigen::Vector2f robot_xy = Eigen::Vector2f::Zero();
    // the room's extent along a wall's inward normal (the fitter's room polygon); empty or ≤ 0 ⇒ unknown (BASE's free term)
    std::function<float(const WallRef&)> room_span;
};

// The class's MODEL solid (the manifest's [[shape.part]] union for solid = "parts", else the whole box, ∪ the back panel), at the
// class proportions' means — the same table the belief builds (CountableBeliefParams::parts_table), so a class with no parts
// model renders as its box (parts::sdf's envelope branch), which is the fallback the brief asks for.
struct BirthSolid
{
    std::vector<double> q;   // the class proportions' values (mean, or ±σ for the propagation)
    rc::ai::parts::Table table(const CountableConfig& cfg) const
    {
        rc::ai::parts::Table T{std::span<const rc::manifest::ShapePart>(cfg.shape_parts), cfg.solid_parts, cfg.ai2_headboard_h_m,
                               cfg.ai2_headboard_t_m};
        T.q = std::span<const double>(q);
        T.n_q = static_cast<int>(q.size());
        return T;
    }
};

// The observed statistic, PREDICTED: the middle (p5 + p95)/2 across the wall of the cloud a body θ would return along the first
// view's rays. The rays are the observed points' own directions from the camera (the mask's silhouette, the observed depths
// unused), each marched to the body's surface (sphere tracing on parts::sdf, exact for a union of boxes); a ray that misses the
// body returns nothing (the visible surface only). Each hit carries the camera's isotropic on-surface noise σ_n
// (Model.AI2SigmaBaseM, the engine's own per-point σ — no range term), so the predicted percentiles are those of the mixture
// Σ N(b_i, σ_n²); their sampling variance q(1−q)/(N f(x_q)²) (N = the observed points in the band, f the mixture density) is the
// spread of the OBSERVED middle about its expectation.
struct RenderedMiddle { bool ok = false; double mid = 0.0, var_stat = 0.0; int n_hit = 0; };
inline RenderedMiddle render_middle(const std::vector<Eigen::Vector3f>& pts, const Eigen::Vector3f& cam, const CountableBeliefState& th,
                                    const rc::ai::parts::Table& T, const rc::countable::VerticalSpan& span, const WallRef& w,
                                    float z_lo, float z_hi, double sigma_n, int n_obs)
{
    RenderedMiddle out;
    const Eigen::Vector3f c(th.cx, th.cy, span.zc(th.H));
    const float rad = 0.5f * std::sqrt(th.w * th.w + th.h * th.h + span.height(th.H) * span.height(th.H));
    const float rad_t = rad + 0.5f * std::max(0.0f, T.panel_h_m);   // the panel stands on the floor, it can reach above a hanging body
    std::vector<double> b;
    b.reserve(pts.size());
    for (const auto& p : pts)
    {
        Eigen::Vector3f d = p - cam;
        const float len = d.norm();
        if (not (len > 0.0f)) continue;
        d /= len;
        // the ray against the body's bounding sphere: [t0, t1] is where a hit can be
        const Eigen::Vector3f oc = cam - c;
        const float bq = oc.dot(d), cq = oc.squaredNorm() - rad_t * rad_t, disc = bq * bq - cq;
        if (disc < 0.0f) continue;
        const float sq = std::sqrt(disc);
        float t = std::max(0.0f, -bq - sq);
        const float t1 = -bq + sq;
        bool hit = false;
        for (int it = 0; it < 128 and t <= t1; ++it)
        {
            const Eigen::Vector3f x = cam + t * d;
            const float s = rc::ai::parts::sdf(x, th.cx, th.cy, th.H, th.w, th.h, th.yaw, T, span);
            if (s < 1e-4f) { hit = true; break; }   // the root finder's tolerance (0.1 mm), not a model cutoff
            t += s;
        }
        if (not hit) continue;
        const Eigen::Vector3f x = cam + t * d;
        if (x.z() < z_lo or x.z() > z_hi) continue;   // the same band as the observed statistic (rc::wall_seed)
        b.push_back(static_cast<double>((x.head<2>() - w.p).dot(w.n)));
    }
    out.n_hit = static_cast<int>(b.size());
    if (b.empty() or not (sigma_n > 0.0)) return out;
    const double inv = 1.0 / sigma_n, nb = static_cast<double>(b.size());
    const auto cdf = [&](double x) { double a = 0.0; for (const double v : b) a += 0.5 * std::erfc(-(x - v) * inv * M_SQRT1_2); return a / nb; };
    const auto pdf = [&](double x)
    { double a = 0.0; for (const double v : b) { const double z = (x - v) * inv; a += std::exp(-0.5 * z * z); } return a * inv / (nb * std::sqrt(2.0 * M_PI)); };
    const auto [lo_it, hi_it] = std::minmax_element(b.begin(), b.end());
    const auto quantile = [&](double q)
    {
        double lo = *lo_it - 8.0 * sigma_n, hi = *hi_it + 8.0 * sigma_n;
        for (int it = 0; it < 60; ++it) { const double m = 0.5 * (lo + hi); (cdf(m) < q ? lo : hi) = m; }
        return 0.5 * (lo + hi);
    };
    const double x5 = quantile(0.05), x95 = quantile(0.95);
    out.mid = 0.5 * (x5 + x95);
    const double n = static_cast<double>(std::max(1, n_obs)), f5 = pdf(x5), f95 = pdf(x95);
    out.var_stat = 0.25 * (0.05 * 0.95 / (n * f5 * f5) + 0.05 * 0.95 / (n * f95 * f95));
    out.ok = true;
    return out;
}

// log of the standard normal's mass over [a, b] (a < b), in the tail-safe form.
inline double log_mass(double a, double b)
{
    const double m = 0.5 * (std::erfc(-b * M_SQRT1_2) - std::erfc(-a * M_SQRT1_2));
    return m > 0.0 ? std::log(m) : -std::numeric_limits<double>::infinity();
}

// The placement mixture's MAP seed (final fix wave F2; view-conditional in CB1). A class whose manifest declares [prior.placement]
// (the chair: p_free 0.5, p_against_wall 0.5) is EITHER against its nearest wall OR free in the room. The seed is the hypothesis
// with the larger
//     p · L(cloud | wall)   vs   (1 − p) · L(cloud | free),
// both in the ONE coordinate the placement speaks about: across the wall (along it both seeds take the cloud's middle). The
// evidence is the cloud's middle m = (p5 + p95)/2 from the wall (rc::birth_footprint's mid_n).
//
// CB1 — WITH A FIRST VIEW (view.have_cam), the hypotheses PREDICT m by rendering the class solid along the first view's rays
// (render_middle), instead of comparing it with a fixed body centre:
//  · Wall: the class-mean body with its back at the declared stand-off s ([prior.attachment]), back face onto the wall, centred
//    along the wall on the cloud. m ~ N(m̂_w, V_w),
//      V_w = V_stat + Σ_j (Δ_j m̂_w)² + σ_wall² + gᵀ Σ_pose g,
//    Δ_j m̂_w = (m̂_w(+σ_j) − m̂_w(−σ_j))/2 the render's response to each class prior over ±1σ: the stand-off (s, σ_s:
//    [prior.attachment] standoff_m / standoff_std_m), the depth (h̄, Model.AI2PriorDepthStd), the width (w̄,
//    Model.AI2PriorFootprintStd), the height (H̄, Model.AI2PriorHeightStd) and each [[shape.proportion]] ((hi − lo)/√12); σ_wall the
//    room wall's own std (WallRef::sigma_m); g = (n, n·ẑ×(c − r)) the across-wall shift of the cloud per unit robot pose error
//    (x, y, ψ), c the cloud's centre, r the robot (the cloud is measured from the robot, the wall is the room's).
//  · Free: the class-mean body at the cloud seed renders the view's offset δ = m̂_f − c_f of the middle from the body centre; the
//    centre itself is uniform over the room's span along n, [h̄/2, D − h̄/2] — the free hypothesis's own placement prior. (BASE
//    centred that prior ON the cloud, x_f ± σ_pos, so the free hypothesis was scored with the data it was meant to predict and
//    always sat at its peak.) Marginalising the centre: L_free = [Φ((b − ĉ)/σ_f) − Φ((a − ĉ)/σ_f)]/(b − a), ĉ = m − δ,
//    σ_f² = V_stat + Σ_j (Δ_j m̂_f)² (depth, width, height, proportions) + gᵀ Σ_pose g. D unknown ⇒ BASE's free term.
// No constant is new: every σ is a declared class prior, the engine's camera σ, the wall's σ or the localiser's covariance.
// WITHOUT a view (no camera pose): BASE's comparison, unchanged — a fixed wall centre x_w = s + h̄/2 and the cloud-seeded free term.
inline bool placement_map_on_wall(const WallSeed& ws, const CountableBeliefState& free_seed, const CountableConfig& cfg, float p,
                                  const std::string& name, const std::vector<Eigen::Vector3f>* pts = nullptr, float z_lo = 0.0f,
                                  float z_hi = 0.0f, const BirthView* view = nullptr, float* p_wall = nullptr)
{
    // the placement POSTERIOR P(against the wall | first view) = 1 / (1 + e^(lf − lw)), handed to the belief as this instance's
    // P(against a wall) (CountableBeliefParams::placement_p_against_wall: the yaw-mode prior's weight)
    const auto post = [p_wall](double lw, double lf) { if (p_wall) *p_wall = static_cast<float>(1.0 / (1.0 + std::exp(lf - lw))); };
    const float h_bar = ws.bf.h;
    const double m = ws.bf.mid_n;
    if (view and view->have_cam and pts)
    {
        const rc::countable::VerticalSpan span{cfg.vertical_extent_m};
        const Eigen::Vector2f n = ws.w.n, t(-n.y(), n.x());
        const float w_bar = ws.bf.w;
        const float H_bar = cfg.ai2_prior_height_m;
        const float sd_d = cfg.ai2_prior_depth_std > 0.0f ? cfg.ai2_prior_depth_std : cfg.ai2_prior_footprint_std;
        BirthSolid solid;
        std::vector<double> q_sd;
        for (const auto& pr : cfg.shape_proportions)
        { solid.q.push_back(0.5 * (double(pr.lo) + double(pr.hi))); q_sd.push_back((double(pr.hi) - double(pr.lo)) / std::sqrt(12.0)); }
        int n_obs = 0;
        for (const auto& q : *pts) if (q.z() >= z_lo and q.z() <= z_hi) ++n_obs;
        const double sig_n = cfg.ai2_sigma_base_m;
        const double along = (ws.bf.centre - ws.w.p).dot(t);   // the cloud's middle along the wall (both hypotheses)
        // the bodies: (s, w, h, H) for the wall one (back at s, centre s + h/2 in), (w, h, H) at the cloud seed for the free one
        const auto wall_body = [&](float s, float w_, float h_, float H_)
        {
            const Eigen::Vector2f cc = ws.w.p + t * float(along) + n * (s + 0.5f * h_);
            return CountableBeliefState{cc.x(), cc.y(), H_, w_, h_, ws.bf.yaw};
        };
        const auto free_body = [&](float w_, float h_, float H_)
        { return CountableBeliefState{free_seed.cx, free_seed.cy, H_, w_, h_, free_seed.yaw}; };
        const auto render = [&](const CountableBeliefState& th, const BirthSolid& so)
        { return render_middle(*pts, view->cam, th, so.table(cfg), span, ws.w, z_lo, z_hi, sig_n, n_obs); };
        // Δ over ±1σ of one prior: the half-difference of the two renders (0 if either side sees nothing)
        const auto half_diff = [](const RenderedMiddle& a, const RenderedMiddle& b) { return (a.ok and b.ok) ? 0.5 * (a.mid - b.mid) : 0.0; };
        double var_pose = 0.0;
        if (view->have_pose)
        {
            const Eigen::Vector2f lever = ws.c - view->robot_xy;
            const Eigen::Vector3d g(n.x(), n.y(), double(n.dot(Eigen::Vector2f(-lever.y(), lever.x()))));
            var_pose = std::max(0.0, g.dot(view->sigma_pose * g));
        }
        const float s0 = cfg.wall_standoff_m, ss = cfg.wall_standoff_std_m;
        const RenderedMiddle rw = render(wall_body(s0, w_bar, h_bar, H_bar), solid);
        const RenderedMiddle rf = render(free_body(w_bar, h_bar, H_bar), solid);
        if (rw.ok and rf.ok)
        {
            double vw = rw.var_stat + double(ws.w.sigma_m) * double(ws.w.sigma_m) + var_pose;
            double vf = rf.var_stat + var_pose;
            const auto add = [](double& v, double d) { v += d * d; };
            add(vw, half_diff(render(wall_body(s0 + ss, w_bar, h_bar, H_bar), solid), render(wall_body(s0 - ss, w_bar, h_bar, H_bar), solid)));
            add(vw, half_diff(render(wall_body(s0, w_bar, h_bar + sd_d, H_bar), solid), render(wall_body(s0, w_bar, h_bar - sd_d, H_bar), solid)));
            add(vf, half_diff(render(free_body(w_bar, h_bar + sd_d, H_bar), solid), render(free_body(w_bar, h_bar - sd_d, H_bar), solid)));
            const float sw = cfg.ai2_prior_footprint_std, sH = cfg.ai2_prior_height_std;
            add(vw, half_diff(render(wall_body(s0, w_bar + sw, h_bar, H_bar), solid), render(wall_body(s0, w_bar - sw, h_bar, H_bar), solid)));
            add(vf, half_diff(render(free_body(w_bar + sw, h_bar, H_bar), solid), render(free_body(w_bar - sw, h_bar, H_bar), solid)));
            add(vw, half_diff(render(wall_body(s0, w_bar, h_bar, H_bar + sH), solid), render(wall_body(s0, w_bar, h_bar, H_bar - sH), solid)));
            add(vf, half_diff(render(free_body(w_bar, h_bar, H_bar + sH), solid), render(free_body(w_bar, h_bar, H_bar - sH), solid)));
            for (std::size_t k = 0; k < solid.q.size(); ++k)
            {
                BirthSolid up = solid, dn = solid;
                up.q[k] += q_sd[k]; dn.q[k] -= q_sd[k];
                add(vw, half_diff(render(wall_body(s0, w_bar, h_bar, H_bar), up), render(wall_body(s0, w_bar, h_bar, H_bar), dn)));
                add(vf, half_diff(render(free_body(w_bar, h_bar, H_bar), up), render(free_body(w_bar, h_bar, H_bar), dn)));
            }
            const double lw = std::log(static_cast<double>(p)) + log_normal_1d(m - rw.mid, vw);
            const double cf = (Eigen::Vector2f(free_seed.cx, free_seed.cy) - ws.w.p).dot(n);
            const double delta = rf.mid - cf, c_hat = m - delta, sf = std::sqrt(vf);
            const double D = view->room_span ? double(view->room_span(ws.w)) : 0.0, a = 0.5 * h_bar, b = D - 0.5 * h_bar;
            double lf;
            const char* free_form;
            if (b > a and sf > 0.0)
            {
                lf = std::log(1.0 - static_cast<double>(p)) + log_mass((a - c_hat) / sf, (b - c_hat) / sf) - std::log(b - a);
                free_form = "uniform over the room span";
            }
            else
            {   // the room's span is unknown: BASE's free term
                const double e = static_cast<double>(h_bar) - ws.bf.extent_n, sp = CountableBeliefParams{}.prior_pos_std;
                lf = std::log(1.0 - static_cast<double>(p)) + log_normal_1d(m - cf, e * e / 12.0 + sp * sp);
                free_form = "BASE (room span unknown)";
            }
            const bool wall = lw > lf;
            post(lw, lf);
            std::print("[{}] birth placement MAP (view): p(wall) {:.2f}; cloud middle {:.3f} m off the wall ({} pts, cam {:.2f} m off "
                       "the wall, {:.2f} m away); wall render {:.3f} ± {:.3f} ({} hits) → log {:.2f}; free render offset {:+.3f}, centre "
                       "{:.3f} ± {:.3f} over [{:.2f}, {:.2f}] ({}) → log {:.2f}; pose σ across {:.3f} → {}\n",
                       name, p, m, n_obs, (view->cam.head<2>() - ws.w.p).dot(n), (view->cam.head<2>() - ws.c).norm(), rw.mid,
                       std::sqrt(vw), rw.n_hit, lw, delta, c_hat, sf, a, b, free_form, lf, std::sqrt(var_pose),
                       wall ? "AGAINST THE WALL" : "FREE (cloud seed)");
            return wall;
        }
        std::print("[{}] birth placement MAP (view): a hypothesis rendered no hit (wall {}, free {}) → BASE's comparison\n", name,
                   rw.n_hit, rf.n_hit);
    }
    const double e = static_cast<double>(h_bar) - ws.bf.extent_n;
    const double v_obs = e * e / 12.0;
    const double x_w = cfg.wall_standoff_m + 0.5 * h_bar;
    const double x_f = (Eigen::Vector2f(free_seed.cx, free_seed.cy) - ws.w.p).dot(ws.w.n);
    const double sp = CountableBeliefParams{}.prior_pos_std;
    const double ss = cfg.wall_standoff_std_m;
    const double lw = std::log(static_cast<double>(p))        + log_normal_1d(m - x_w, v_obs + ss * ss);
    const double lf = std::log(1.0 - static_cast<double>(p))  + log_normal_1d(m - x_f, v_obs + sp * sp);
    const bool wall = lw > lf;
    post(lw, lf);
    std::print("[{}] birth placement MAP: p(wall) {:.2f}; cloud middle {:.3f} m off the wall (extent {:.3f}, class depth {:.3f}); "
               "wall seed {:.3f} ± {:.3f} → log {:.2f}; free seed {:.3f} ± {:.3f} → log {:.2f} → {}\n",
               name, p, m, ws.bf.extent_n, h_bar, x_w, std::sqrt(v_obs + ss * ss), lw, x_f, std::sqrt(v_obs + sp * sp), lf,
               wall ? "AGAINST THE WALL" : "FREE (cloud seed)");
    return wall;
}

// The birth's seed: s0 arrives as the cold-start footprint (the cloud's moment centre and extents, the fitter's); for a class
// that declares a back-face attachment, and when room geometry is known, the back goes on the nearest wall (birth_on_priors) —
// with p = 1 implied unless the manifest declares a placement mixture ([prior.placement], the chair), in which case the seed is
// the mixture's MAP (placement_map_on_wall). p = 1 classes: nightstand, bed, toilet, hood (flush prior + stand-off) and the
// refrigerator (stand-off, flush 0): their path is today's, bit for bit. Never a c2v class (rc::birth_on_wall).
// Returns the cloud's EXCESS position covariance when the wall placed the seed, nullopt when s0 is left as it came. p_wall (CB1),
// when given, receives the placement mixture's POSTERIOR P(against the wall | first view); it is left untouched when no mixture
// decided (a p = 1 class, no room geometry, too few points).
inline std::optional<Eigen::Matrix2f> seed_birth(CountableBeliefState& s0, const std::vector<Eigen::Vector3f>& pts, float z_lo,
                                                 float z_hi, const CountableConfig& cfg, bool have_room_geometry,
                                                 const NearestWall& nearest_wall, const std::string& name,
                                                 const BirthView* view = nullptr, float* p_wall = nullptr)
{
    if (not (have_room_geometry and birth_on_wall(cfg))) return std::nullopt;
    if (not cfg.placement_p_against_wall) return birth_on_priors(s0, pts, z_lo, z_hi, cfg, nearest_wall, name);
    const auto ws = wall_seed(pts, z_lo, z_hi, cfg, nearest_wall);
    if (not ws) return std::nullopt;
    if (not placement_map_on_wall(*ws, s0, cfg, *cfg.placement_p_against_wall, name, &pts, z_lo, z_hi, view, p_wall)) return std::nullopt;
    apply_wall_seed(s0, *ws, name);
    return ws->bf.pos_cov;
}

}  // namespace rc
