/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify it under
 *    the terms of the GNU General Public License as published by the Free
 *    Software Foundation, either version 3 of the License, or (at your option)
 *    any later version. See <http://www.gnu.org/licenses/>.
 */

#pragma once

/*
 *  image_edge_ops.h — the pixel-level primitives of the RGB edge measurement. Header-only,
 *  Eigen + std only. NO OpenCV, and that is a decision, not an omission:
 *
 *    - What we need is dI/dn_hat at a few thousand SCATTERED points on arbitrary-angle contours.
 *      That is a central difference of two bilinear taps along n_hat: 4 taps per site. A full-image
 *      cv::Sobel over 921600 pixels is ~100x the work AND the wrong operator — hood_projection.cpp's
 *      door_ness() is documented as BLIND to horizontal seams for exactly this reason (an axis-aligned
 *      operator applied to a non-axis-aligned structure). Rotating an axis-aligned result afterwards
 *      is not the same thing as differentiating along the normal.
 *    - It keeps the thread-boundary payload a std::vector<uint8_t> with value semantics, so the
 *      cv::Mat refcounted-shallow-handle hazard (CLAUDE.md: the real cause of the 2026-07 cores)
 *      cannot arise here at all.
 *    - common/ is deliberately OpenCV-free, and room_concept does not link it either.
 *
 *  The one genuine risk of hand-rolling this — an off-by-one in the bilinear sampler — is answered
 *  directly by gn_selftest, which checks recovery of a synthetic step edge whose true sub-pixel
 *  position is known analytically, at 8 orientations including 0/45/90 degrees.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <Eigen/Dense>

#include "image_edge_types.h"

namespace rc::img
{
    // ── Projection, from the reduced plain-data model ────────────────────────────────────────────
    /// Reproduces DSR::CameraAPI::project() from a CameraModel, with NO DSR dependency, so the torch
    /// mirror and the selftest can project without a graph. calibrate_camera_model() verifies this
    /// against the real project() at bring-up and refuses the model if they disagree.
    /// Camera frame: x-right, y-DEPTH, z-up.
    inline bool project_with_model(const CameraModel& m, const Eigen::Vector3d& p, Eigen::Vector2d& uv)
    {
        if (m.kind == CameraModel::Kind::Pinhole)
        {
            if (p.y() <= 1e-6) return false;
            uv.x() = m.fx * p.x() / p.y() + m.cx;
            uv.y() = m.cy - m.fy * p.z() / p.y();
            return std::isfinite(uv.x()) and std::isfinite(uv.y());
        }
        const double rho2 = p.x() * p.x() + p.y() * p.y();
        const double r    = p.norm();
        if (rho2 < 1e-18 or r < 1e-9) return false;
        const double az = m.azimuth_sign * std::atan2(p.x(), p.y()) + m.azimuth_offset;
        double u = (az / (2.0 * M_PI) + 0.5) * m.width;
        u = std::fmod(u, static_cast<double>(m.width));
        if (u < 0.0) u += m.width;
        uv.x() = u;
        if (m.kind == CameraModel::Kind::Equirect)
            uv.y() = (std::asin(std::clamp(-p.z() / r, -1.0, 1.0)) / M_PI + 0.5) * m.height;
        else    // Cylindrical: elevation is PLANAR (tan), not the angle — limited vertical FoV
            uv.y() = 0.5 * m.height - (m.width / (2.0 * M_PI)) * (p.z() / std::sqrt(rho2));
        return std::isfinite(uv.x()) and std::isfinite(uv.y());
    }

    /// P = d(u,v)/dp_cam from the reduced model. ANALYTIC for pinhole; central difference (in
    /// camera space, on a smooth analytic map — see camera_jacobian.h for why that is safe here)
    /// for the 360 models, whose panorama convention lives in azimuth_sign/offset.
    inline bool project_jacobian_model(const CameraModel& m, const Eigen::Vector3d& p,
                                       Eigen::Matrix<double, 2, 3>& P)
    {
        if (m.kind == CameraModel::Kind::Pinhole)
        {
            const double Y = p.y();
            if (Y <= 1e-6) return false;
            const double iY = 1.0 / Y, iY2 = iY * iY;
            P(0, 0) = m.fx * iY;  P(0, 1) = -m.fx * p.x() * iY2;  P(0, 2) = 0.0;
            P(1, 0) = 0.0;        P(1, 1) =  m.fy * p.z() * iY2;  P(1, 2) = -m.fy * iY;
            return P.allFinite();
        }
        const double r = p.norm();
        if (r < 1e-6) return false;
        const double h = 1e-6 * std::max(1.0, r);
        for (int j = 0; j < 3; ++j)
        {
            Eigen::Vector3d pp = p, pm = p;
            pp[j] += h; pm[j] -= h;
            Eigen::Vector2d a, b;
            if (not project_with_model(m, pp, a) or not project_with_model(m, pm, b)) return false;
            double du = a.x() - b.x();
            while (du >  0.5 * m.width) du -= m.width;
            while (du <= -0.5 * m.width) du += m.width;
            P(0, j) = du / (2.0 * h);
            P(1, j) = (a.y() - b.y()) / (2.0 * h);
        }
        return P.allFinite();
    }

    // ── Colour -> grey ───────────────────────────────────────────────────────────────────────────
    /// Rec.601 luma. `src` is w*h*3; `dst` is resized to w*h. `swap_rb` for BGR8 input.
    inline void gray_from_rgb8(const std::uint8_t* src, int w, int h, bool swap_rb,
                               std::vector<std::uint8_t>& dst)
    {
        const std::size_t n = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
        dst.resize(n);
        const int ri = swap_rb ? 2 : 0, bi = swap_rb ? 0 : 2;
        for (std::size_t i = 0; i < n; ++i)
        {
            const std::uint8_t* p = src + 3 * i;
            // Integer luma: (77 R + 150 G + 29 B) >> 8. Exact, branch-free, no float round-trip.
            dst[i] = static_cast<std::uint8_t>((77 * p[ri] + 150 * p[1] + 29 * p[bi]) >> 8);
        }
    }

    // ── Sampling ─────────────────────────────────────────────────────────────────────────────────
    /// Bilinear intensity at continuous (u, v), pixel CENTRES at integer coordinates.
    /// Returns false when the 2x2 support is not fully inside the image — the caller must treat an
    /// out-of-bounds tap as "no measurement", never as zero (a clamped border is a synthetic edge).
    ///
    /// `wrap_u` makes the COLUMN axis cyclic, which is what a 360 panorama actually is: column w-1
    /// and column 0 are neighbouring pixels on the same continuous image. Pass 0 for a pinhole,
    /// where they are opposite edges of the sensor and joining them would fabricate an edge.
    ///
    /// ★ THIS IS NOT COSMETIC ON THE RICOH. A wall corner is near-vertical, so its search normal is
    ///   near-HORIZONTAL and the whole normal search runs along u. Without wrapping, a corner within
    ///   L px of the seam has half its window return false, so the peak is picked from whichever
    ///   half stayed in bounds — and if the true edge was on the wrapped side, that peak is NOISE
    ///   reported with a finite sigma. It is a biased match, not a missing sample, and it would show
    ///   up in the CSV as a fat residual tail at exactly two azimuths.
    inline bool bilinear(const std::uint8_t* g, int w, int h, float u, float v, float& out,
                         int wrap_u = 0)
    {
        if (wrap_u > 0)
        {
            u = std::fmod(u, static_cast<float>(wrap_u));
            if (u < 0.f) u += static_cast<float>(wrap_u);
        }
        const float fu = std::floor(u), fv = std::floor(v);
        const int   y0 = static_cast<int>(fv);
        int         x0 = static_cast<int>(fu);
        if (y0 < 0 or y0 + 1 >= h) return false;
        int x1 = x0 + 1;
        if (wrap_u > 0)
        {
            if (x0 < 0 or x0 >= w) return false;      // fmod already folded it; this catches NaN/inf
            if (x1 >= w) x1 -= w;                     // ... and here the seam simply closes
        }
        else if (x0 < 0 or x1 >= w) return false;
        const float au = u - fu, av = v - fv;
        const std::uint8_t* r0 = g + static_cast<std::size_t>(y0) * w;
        const std::uint8_t* r1 = r0 + w;
        const float top = static_cast<float>(r0[x0]) * (1.f - au) + static_cast<float>(r0[x1]) * au;
        const float bot = static_cast<float>(r1[x0]) * (1.f - au) + static_cast<float>(r1[x1]) * au;
        out = top * (1.f - av) + bot * av;
        return true;
    }

    /// Directional derivative dI/ds along the UNIT vector n_hat, at (u,v), by central difference of
    /// two bilinear taps one pixel apart. Units: grey levels per pixel.
    /// `wrap_u`: see bilinear — pass the image width on a panorama, 0 on a pinhole.
    inline bool dir_derivative(const std::uint8_t* g, int w, int h,
                               float u, float v, const Eigen::Vector2f& n_hat, float& out,
                               int wrap_u = 0)
    {
        float a = 0.f, b = 0.f;
        if (not bilinear(g, w, h, u + n_hat.x(), v + n_hat.y(), a, wrap_u)) return false;
        if (not bilinear(g, w, h, u - n_hat.x(), v - n_hat.y(), b, wrap_u)) return false;
        out = 0.5f * (a - b);
        return true;
    }

    // ── Per-frame sensor noise, MEASURED ─────────────────────────────────────────────────────────
    /// Immerkaer's estimator: sigma = 1.4826 * median|I (*) L| / sqrt(6), L the 3x3 Laplacian
    /// [[1,-2,1],[-2,4,-2],[1,-2,1]]. The kernel annihilates any locally-planar intensity ramp, so a
    /// textured but noiseless image still reads ~0 — which is the property we need, because this
    /// number is the DENOMINATOR of every precision the subsystem reports.
    ///
    /// Deliberately measured and not configured: it moves with auto-exposure and gain, and when it
    /// moves the whole frame's precision must move with it. `step` subsamples (16 => ~3600 taps on
    /// 1280x720, ~10 us). Returns grey levels; floored at a small positive value so 1/sigma^2 is finite.
    inline float estimate_noise_sigma_immerkaer(const std::uint8_t* g, int w, int h, int step = 16)
    {
        if (w < 3 or h < 3) return 1.f;
        std::vector<float> resp;
        resp.reserve(static_cast<std::size_t>((w / step + 1)) * static_cast<std::size_t>((h / step + 1)));
        for (int y = 1; y < h - 1; y += step)
            for (int x = 1; x < w - 1; x += step)
            {
                const std::uint8_t* c = g + static_cast<std::size_t>(y) * w + x;
                const float l =  4.f * c[0]
                              - 2.f * (c[-1] + c[1] + c[-w] + c[w])
                              +       (c[-w - 1] + c[-w + 1] + c[w - 1] + c[w + 1]);
                resp.push_back(std::fabs(l));
            }
        if (resp.empty()) return 1.f;
        const auto mid = resp.begin() + static_cast<std::ptrdiff_t>(resp.size() / 2);
        std::nth_element(resp.begin(), mid, resp.end());
        // The 3x3 Laplacian's noise gain is sqrt(sum of squared taps) = sqrt(4*1 + 4*4 + 16) = 6
        // EXACTLY, so sigma_L = 6 * sigma_I. Converting a median of |.| to a sigma costs the usual
        // 1.4826. ★ Not sqrt(6): that overstated sigma_I by 2.449x, i.e. it would have inflated every
        // per-sample variance the subsystem reports by ~6x and quietly muted the whole term.
        const float sigma = 1.4826f * (*mid) / 6.0f;
        return std::max(0.25f, sigma);   // a floor of a quarter grey level: quantisation, not tuning
    }

    // ── Precision, from the Cramer-Rao bound ─────────────────────────────────────────────────────
    /// Variance of the estimated edge location along the search direction.
    ///
    /// Model I(s) = f(s - s*) + eta, eta ~ N(0, sigma_i^2) i.i.d. Since g(s) = dI/dn = f'(s), the
    /// Fisher information about s* is sigma_i^-2 * sum_s g(s)^2, hence
    ///
    ///     sigma_s^2 = sigma_i^2 / sum_s g(s)^2      [ intensity^2 / (intensity/px)^2 = px^2 ]
    ///
    /// Why this is the Active-Inference answer and a contrast score is not:
    ///   - sharp isolated edge  -> sum g^2 large -> sigma << 1 px
    ///   - soft / low contrast  -> sigma grows CONTINUOUSLY
    ///   - flat wall            -> sum g^2 -> 0 -> sigma -> inf -> weight 1/sigma^2 -> 0, so the
    ///                             sample SELF-MUTES and no visibility threshold is needed anywhere.
    /// It also sidesteps both failure modes hood_projection.cpp paid for: mean|Sobel| has no absolute
    /// scale (here sum g^2 and sigma_i come from the same sensor, so the ratio is physical), and
    /// peak/median explodes on a smooth panel (here a smooth panel correctly returns sigma -> inf).
    ///
    /// NOTE the bound is LOCAL: it understates sigma when the profile is multimodal (skirting board
    /// vs floor junction). That case is handled by the inlier/outlier mixture in the factor, NOT by
    /// patching this bound.
    inline float crb_sigma_px(float sum_g2, float sigma_i)
    {
        if (not (sum_g2 > 0.f) or not (sigma_i > 0.f))
            return std::numeric_limits<float>::infinity();
        return sigma_i / std::sqrt(sum_g2);
    }

    // ── Sub-pixel peak ───────────────────────────────────────────────────────────────────────────
    /// Parabola vertex through three equally spaced samples (ym, y0, yp) at (-1, 0, +1), y0 the max.
    /// Returns the offset in (-0.5, 0.5); 0 when the three are degenerate.
    inline float parabolic_vertex(float ym, float y0, float yp)
    {
        const float den = ym - 2.f * y0 + yp;
        if (std::fabs(den) < 1e-12f) return 0.f;
        return std::clamp(0.5f * (ym - yp) / den, -0.5f, 0.5f);
    }


    /// Camera-frame XYZ for a pixel with a measured depth value.
    ///
    /// Matches CameraAPI::get_xyz_from_rgbd_points under DepthConvention::Forward (X right,
    /// Y FORWARD, Z up, the depth value being Y itself) for a pinhole camera, and DECLINES rather
    /// than guesses on a panoramic one. That cortex method used to apply the pinhole intrinsics
    /// unconditionally on every model; it was fixed on 2026-08-28 (cortex a8db664) to dispatch
    /// through ray_from_pixel like project() does, and it now also takes an explicit
    /// DepthConvention, because whether a depth image carries forward distance or radial range is
    /// not something the image itself states.
    ///
    /// ★ `depth` IS ASSUMED TO BE THE FORWARD COORDINATE (perpendicular distance to the image
    ///   plane), not the range along the ray. That is the ZED SDK's convention and the one the
    ///   cortex method encodes, but this data reaches us through the Webots bridge and the
    ///   assumption is NOT verified here. It is checkable from the logged columns: if depth were
    ///   range-along-ray it would exceed the forward distance by 1/cos(angle off axis), a several-
    ///   percent excess that GROWS toward the image edge. Which is why depth_raw, the predicted
    ///   forward distance and the predicted range are all recorded per triple point.
    inline bool xyz_from_pixel_depth(const CameraModel& m, double u, double v, double depth,
                                     Eigen::Vector3d& xyz)
    {
        if (not m.valid or not std::isfinite(depth) or not (depth > 0.0)) return false;
        if (m.kind == CameraModel::Kind::Pinhole)
        {
            if (not (m.fx > 0.f) or not (m.fy > 0.f)) return false;
            xyz = Eigen::Vector3d((u - static_cast<double>(m.cx)) * depth / static_cast<double>(m.fx),
                                  depth,
                                  (static_cast<double>(m.cy) - v) * depth / static_cast<double>(m.fy));
            return true;
        }
        // Panoramic: the depth value can only be RADIAL — there is no single forward axis for a
        // sensor spanning 360 degrees, and dividing by the ray's Y component would diverge at the
        // two azimuths where it crosses zero. Needs the model's own inverse mapping, which this
        // reduced CameraModel does not carry (it was calibrated to reproduce project(), not to
        // invert it), so it declines rather than guessing. CameraAPI::ray_from_pixel does invert
        // all three models if a caller here ever needs it — at the cost of a graph-bound call in
        // the measurement path, which camera_ingestor.h exists to avoid.
        return false;
    }

    /// Column difference on a possibly-CYCLIC image axis.
    ///
    /// ★ A panorama's column axis wraps: column w-1 and column 0 are neighbours. Subtracting two u
    ///   coordinates across that seam gives a residual of nearly a full image WIDTH for two points a
    ///   pixel apart. On a pinhole this is the identity, so calling it unconditionally costs nothing
    ///   and removes the class of bug where a residual is wrap-corrected in one place and not in
    ///   another — the sample search and the factor both handle the seam; the triple-point residuals
    ///   did not, which would have been invisible until a corner happened to sit on it.
    inline double du_wrapped(double du, const CameraModel& m)
    {
        if (m.kind == CameraModel::Kind::Pinhole or not (m.width > 0.f)) return du;
        const double w = static_cast<double>(m.width);
        return du - std::round(du / w) * w;
    }
}  // namespace rc::img
