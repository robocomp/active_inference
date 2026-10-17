#include "door_frame_evidence.h"

#include <algorithm>
#include <cmath>

namespace rc::door
{

namespace
{

// Φ(x), the same form contour_depth_check.cpp uses.
[[nodiscard]] inline float phi_cdf(float x) { return 0.5f * std::erfc(-x * 0.70710678f); }

[[nodiscard]] inline bool in_image(const cv::Point2f& p, int w, int h)
{
    return p.x >= 0.0f and p.y >= 0.0f and p.x < static_cast<float>(w) and p.y < static_cast<float>(h);
}

// Project P and a short step along the edge tangent and into the aperture; return the pixel, the unit image
// normal of the edge, signed to point into the aperture. nullopt if any of the three does not project.
// ε is a finite-difference step for the local image direction, not a tolerance.
struct Local { cv::Point2f px; cv::Point2f n_in; float depth_m; float ddepth_dpx; };
[[nodiscard]] std::optional<Local> project_local(const Eigen::Vector3d& P, const Eigen::Vector3d& tangent,
                                                 const Eigen::Vector3d& into, const rc::edges::PointProjector& project)
{
    constexpr double eps = 0.02;
    const auto p0 = project(P);
    const auto pt = project(P + eps * tangent);
    const auto pi = project(P + eps * into);
    if (not p0 or not pt or not pi)
        return std::nullopt;
    const cv::Point2f t = pt->px - p0->px;
    const float tl = std::hypot(t.x, t.y);
    if (not std::isfinite(tl) or tl < 1e-6f)
        return std::nullopt;
    cv::Point2f n(-t.y / tl, t.x / tl);
    const cv::Point2f d = pi->px - p0->px;
    float dn = n.x * d.x + n.y * d.y;
    if (dn < 0.0f) { n = -n; dn = -dn; }
    // The wall plane's depth gradient along n: P + ε·into stays IN the wall plane (into is ±u for a jamb,
    // −up for the lintel), so this is the plane's own change in forward depth per pixel across the edge.
    const float g = (dn > 1e-4f) ? (pi->depth_m - p0->depth_m) / dn : 0.0f;
    return Local{p0->px, n, p0->depth_m, std::isfinite(g) ? g : 0.0f};
}

}   // namespace

bool leaf_blocks_sightline(const LeafPose& L, const Eigen::Vector3f& O, const Eigen::Vector3f& P)
{
    // 2-D: sightline O→P against the leaf's mid-plane segment hinge → free edge.
    const Eigen::Vector2f o(O.x(), O.y()), p(P.x(), P.y());
    const Eigen::Vector2f a = L.hinge_xy;
    const Eigen::Vector2f b = L.hinge_xy + 2.0f * L.half_w * L.ex;
    const Eigen::Vector2f d = p - o, ab = b - a;
    const auto cross = [](const Eigen::Vector2f& x, const Eigen::Vector2f& y) { return x.x() * y.y() - x.y() * y.x(); };
    const float rxs = cross(d, ab);
    if (std::abs(rxs) < 1e-9f)
        return false;                                   // parallel: a sightline grazing the leaf's plane
    const float t = cross(a - o, ab) / rxs;             // along the sightline
    const float u = cross(a - o, d) / rxs;              // along the leaf
    if (t <= 0.0f or t >= 1.0f or u < 0.0f or u > 1.0f)
        return false;
    // P lying on the leaf itself (the hinge jamb touches the hinge edge) is not the leaf standing in front.
    if ((1.0f - t) * d.norm() < 2.0f * L.half_t)
        return false;
    const float z = O.z() + t * (P.z() - O.z());
    return z >= L.centre_z - L.half_h and z <= L.centre_z + L.half_h;
}

FrameProbe build_frame_probe(const Aperture& ap, const rc::edges::PointProjector& project,
                             const PointVisibility& visible, const PointVisibility& aperture_side_clear,
                             int img_w, int img_h, float casing_m, float registration_sigma_m)
{
    FrameProbe out;
    if (not project or img_w <= 0 or img_h <= 0 or ap.w <= 1e-3f or ap.h <= 1e-3f)
        return out;

    const Eigen::Vector2f u2 = ap.wall_u.normalized();
    const Eigen::Vector3d u(u2.x(), u2.y(), 0.0);
    const Eigen::Vector3d up(0.0, 0.0, 1.0);
    const auto wall_point = [&](float along, float z) -> Eigen::Vector3d
    {
        const Eigen::Vector2f xy = ap.wall_O + along * u2;
        return {xy.x(), xy.y(), z};
    };
    // A point just inside the hole next to P, for the aperture-side depth read's own-leaf test: the leaf's
    // thickness into the aperture — the scale at which "inside the hole" is distinguishable from "on the jamb".
    const double inset = std::max(0.02f, ap.thickness);
    // The depth half's read window, in metres along the wall: past the casing and the belief's registration
    // error, never more than half the opening (it must not reach the opposite jamb).
    const float side_window_m = std::min(std::max(0.0f, casing_m) + std::max(0.0f, registration_sigma_m), 0.5f * ap.w);

    // `outward` (jambs): the null slides AWAY from the hole only, by kFrameJambOutward — see the header.
    // nullptr (lintel): the symmetric leaf-contour set kControlOffsets.
    const auto add = [&](const Eigen::Vector3d& P, const Eigen::Vector3d& tangent, const Eigen::Vector3d& into,
                         const Eigen::Vector3d* outward)
    {
        ++out.n_total;
        const auto loc = project_local(P, tangent, into, project);
        if (not loc or not in_image(loc->px, img_w, img_h))
            return;                                          // outside the frustum ⇒ not visible
        const float v = visible ? std::clamp(visible(P.cast<float>()), 0.0f, 1.0f) : 1.0f;
        if (not (v > 0.0f))
            return;
        FrameSample s;
        s.px = loc->px; s.n_in = loc->n_in; s.depth_m = loc->depth_m; s.w_vis = v;
        s.plane_ddepth_dpx = loc->ddepth_dpx;
        if (side_window_m > 0.0f)
            if (const auto pc = project(P + static_cast<double>(side_window_m) * into); pc)
            {
                const cv::Point2f dc = pc->px - loc->px;
                s.side_window_px = std::max(0.0f, dc.x * loc->n_in.x + dc.y * loc->n_in.y);
            }
        s.w_depth = aperture_side_clear
                  ? std::clamp(aperture_side_clear((P + inset * into).cast<float>()), 0.0f, 1.0f) : 1.0f;
        // The null: the same sample slid along the WALL. Valid at any phi — the frame lies in the wall.
        for (std::size_t k = 0; k < rc::edges::kControlOffsets.size(); ++k)
        {
            const Eigen::Vector3d Pk = outward != nullptr
                ? Eigen::Vector3d(P + static_cast<double>(kFrameJambOutward[k] * ap.w) * (*outward))
                : Eigen::Vector3d(P + static_cast<double>(rc::edges::kControlOffsets[k] * ap.w) * u);
            const auto lk = project_local(Pk, tangent, into, project);
            if (not lk or not in_image(lk->px, img_w, img_h))
                continue;
            const float wk = visible ? std::clamp(visible(Pk.cast<float>()), 0.0f, 1.0f) : 1.0f;
            s.ctl[k] = FrameControl{lk->px, lk->n_in, wk};
        }
        out.vis_sum += v;
        out.samples.push_back(s);
    };

    const float z0 = ap.floor_z, z1 = ap.floor_z + ap.h;
    for (int i = 0; i < kFrameJambSamples; ++i)
    {
        const float z = z0 + ap.h * (static_cast<float>(i) + 0.5f) / kFrameJambSamples;
        const Eigen::Vector3d out_near = -u, out_far = u;
        add(wall_point(ap.s, z),        up,  u, &out_near); // near jamb: the hole is +u of it, the wall −u
        add(wall_point(ap.s + ap.w, z), up, -u, &out_far);  // far jamb:  the hole is −u of it, the wall +u
    }
    for (int i = 0; i < kFrameLintelSamples; ++i)
    {
        const float a = ap.s + ap.w * (static_cast<float>(i) + 0.5f) / kFrameLintelSamples;
        add(wall_point(a, z1), u, -up, nullptr);            // lintel: the hole is below it
    }
    return out;
}

FrameScore score_frame(const FrameProbe& probe, const rc::edges::PreparedFrame* prep, const cv::Mat* depth,
                       const rc::edges::ContourDepthParams& dp)
{
    FrameScore out;
    out.vis_frac = probe.vis_frac();

    // ── RGB half: visibility-weighted mean across-boundary gradient, against the wall-slid null ───────────
    if (prep != nullptr and prep->valid())
    {
        double t_acc = 0.0, t_w = 0.0, c_acc = 0.0, c_w = 0.0;
        for (const auto& s : probe.samples)
        {
            const float g = rc::edges::across_boundary_gradient_at(
                prep->gx, prep->gy, static_cast<int>(std::lround(s.px.x)), static_cast<int>(std::lround(s.px.y)),
                s.n_in.x, s.n_in.y);
            if (g < 0.0f)
                continue;                                   // outside the usable frame: not measured
            t_acc += static_cast<double>(s.w_vis) * g;
            t_w   += s.w_vis;
            ++out.n_edge;
            for (const auto& c : s.ctl)
            {
                if (not (c.w > 0.0f))
                    continue;
                const float gc = rc::edges::across_boundary_gradient_at(
                    prep->gx, prep->gy, static_cast<int>(std::lround(c.px.x)), static_cast<int>(std::lround(c.px.y)),
                    c.n.x, c.n.y);
                if (gc < 0.0f)
                    continue;
                const double w = static_cast<double>(s.w_vis) * c.w;
                c_acc += w * gc;
                c_w   += w;
                ++out.n_ctl;
            }
        }
        if (t_w > 0.0)
        {
            out.s_true = static_cast<float>(t_acc / t_w);
            if (c_w > 0.0)
            {
                out.s_control = static_cast<float>(c_acc / c_w);
                out.excess = (out.s_true - out.s_control) / prep->frame_ref;
            }
        }
    }

    // ── depth half: does the hole recede behind the wall on the other side of the edge? ──────────────────
    // ★THE GENERATIVE MODEL OF ONE EDGE, and why the read changed (2026-10-13). Across a predicted jamb the
    // depth profile of a REAL doorway is not a step: wall → CASING (frame boards standing PROUD of the wall,
    // i.e. NEARER) → hole (deeper). Read at a single 3-6 px offset, the "hole" side lands on the casing
    // whenever the casing is wider than the offset or the belief's edge is a few cm off — and then the step
    // says "the hole is in FRONT of the wall": live try34, from inside room_2, dep −0.45 / −0.32 on a real
    // open doorway (−0.6..−0.9 on door_24). Reproduced in tests/test_door_frame_evidence.cpp (g) with
    // Webots' Door.proto geometry: −0.46 ignoring the casing.
    // So each side is read over a WINDOW long enough to reach past the casing and the registration error
    // (side_window_px, from FrameCasingM + BoundarySigmaM, at most half the aperture), and summarised by its
    // upper quartile — "how deep does it get just across this edge". A real hole makes the hole side deep
    // somewhere in its window whatever stands in front of part of it; a blank wall does not.
    // ★SYMMETRIC, AND MEASURED AGAINST THE WALL PLANE. Both sides use the same window and the same quantile,
    // so on a blank wall their noise statistics are identical and cancel (a max on one side only would be a
    // ratchet). On a wall seen obliquely the plane itself changes depth across the window: for ANY quantile
    // of a monotone profile the expected difference between the two sides is g·(off_min + K), g the model's
    // own depth-per-pixel of the wall plane along n_in, K the window's far end — so that is the null, not 0.
    // Relative throughout: a common-mode ZED bias cancels in the reads and in the prediction's difference.
    if (depth != nullptr and not depth->empty() and depth->type() == CV_32F)
    {
        constexpr float kQuantile = 0.75f;   // a summary statistic of the window (upper quartile), not a gate
        const auto side_quantile = [&](float px, float py, float nx, float ny, int k0, int k1) -> float
        {
            std::vector<float> vals;
            vals.reserve(static_cast<std::size_t>(std::max(1, k1 - k0 + 1)));
            for (int k = k0; k <= k1; ++k)
            {
                const int qx = static_cast<int>(std::lround((px + k * nx) * dp.depth_scale_x));
                const int qy = static_cast<int>(std::lround((py + k * ny) * dp.depth_scale_y));
                if (qx < 0 or qy < 0 or qx >= depth->cols or qy >= depth->rows)
                    continue;
                const float d = depth->at<float>(qy, qx);
                if (std::isfinite(d) and d > 0.0f)
                    vals.push_back(d);
            }
            if (vals.empty())
                return -1.0f;
            const auto q = static_cast<std::ptrdiff_t>(std::lround(kQuantile * static_cast<float>(vals.size() - 1)));
            std::nth_element(vals.begin(), vals.begin() + q, vals.end());
            return vals[static_cast<std::size_t>(q)];
        };
        double v_acc = 0.0, v_w = 0.0;
        for (const auto& s : probe.samples)
        {
            const double w = static_cast<double>(s.w_vis) * s.w_depth;
            if (not (w > 0.0))
                continue;
            const int k0 = dp.offset_px_min;
            const int k1 = std::max(dp.offset_px_max, k0 + static_cast<int>(std::lround(s.side_window_px)));
            const float d_wall = side_quantile(s.px.x, s.px.y, -s.n_in.x, -s.n_in.y, k0, k1);
            if (not (d_wall > 0.0f))
                continue;                                   // no read on the wall side: NOT measured
            const float d_ap = side_quantile(s.px.x, s.px.y, s.n_in.x, s.n_in.y, k0, k1);
            // No return through the hole is NOT recession (a dropout looks the same) — it says nothing.
            const float plane_step = s.plane_ddepth_dpx * static_cast<float>(k0 + k1);
            const float R = (d_ap > 0.0f)
                          ? phi_cdf((d_ap - d_wall - plane_step) / std::max(dp.sigma_step_m, 1e-3f)) : 0.5f;
            v_acc += w * (2.0 * R - 1.0);
            v_w   += w;
            ++out.n_depth;
        }
        if (v_w > 0.0)
            out.depth_verdict = static_cast<float>(v_acc / v_w);
    }
    return out;
}

}   // namespace rc::door
