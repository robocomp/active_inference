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
 *  camera_jacobian.h — d(u,v)/dp_cam for whichever projection model the camera node declares.
 *
 *  DSR::CameraAPI::project() returns the VALUE only. We must not add a derivative to cortex: that
 *  header is root-owned under /usr/local/include/dsr and needs a full user-side reinstall
 *  (CLAUDE.md), which would block this feature on an unrelated deploy. So the derivative lives here.
 *
 *  TWO PATHS, and the split is deliberate:
 *
 *   - PINHOLE -> ANALYTIC. Every parameter is publicly readable (get_focal_x/y/width/height, and
 *     centre = size/2 exactly as CameraAPI's ctor computes it). Cheap and exact.
 *
 *   - EQUIRECTANGULAR / CYLINDRICAL -> CENTRAL DIFFERENCE IN CAMERA SPACE. These are closed-form
 *     too, but `azimuth_sign` and `azimuth_offset` are PRIVATE with no getters, so the analytic
 *     du/dp cannot be written without guessing the panorama column convention — and guessing the
 *     sign wrong flips the yaw Jacobian, producing a solver that converges on the ZED and diverges
 *     only on the Ricoh. A finite difference asks project() itself and cannot disagree with it.
 *
 *  ★ WHY A FINITE DIFFERENCE IS SAFE HERE, when room_concept.cpp:3733 warns against one.
 *    That warning is about differencing in STATE space, where a +-1e-3 probe re-assigns lidar points
 *    to a different polygon segment and the SDF is piecewise — the probe is not a fair reference.
 *    Here we difference a SMOOTH, ANALYTIC, well-conditioned 3->2 map in CAMERA space, with no state
 *    and no data association involved. Cost: 6 project() calls per sample per LM iteration.
 *
 *  gn_selftest asserts the analytic pinhole path and the central-difference path agree to 1e-5, so
 *  neither can rot silently and the fallback validates the fast path.
 */

#include <cmath>

#include <Eigen/Dense>
#include <dsr/api/dsr_camera_api.h>

#include "image_edge_types.h"
#include "image_edge_ops.h"

namespace rc::img
{
    /// P = d(u,v)/dp_cam, 2x3, in pixels per metre. Returns false if the point is not projectable
    /// (behind the pinhole image plane, or degenerate for a 360 model).
    ///
    /// Camera frame convention throughout this codebase: x-right, y-DEPTH(optical axis), z-up —
    /// NOT the optical z-forward convention. See zed-camera-frame-convention.
    inline bool project_jacobian(const DSR::CameraAPI& cam, const Eigen::Vector3d& p,
                                 Eigen::Matrix<double, 2, 3>& P)
    {
        using PM = DSR::CameraAPI::ProjectionModel;

        if (cam.get_projection_model() == PM::Pinhole)
        {
            // u = fx*X/Y + cx        v = cy - fy*Z/Y
            const double Y = p.y();
            if (Y <= 1e-6) return false;              // at or behind the image plane
            const double fx = cam.get_focal_x(), fy = cam.get_focal_y();
            const double iY = 1.0 / Y, iY2 = iY * iY;
            P(0, 0) =  fx * iY;   P(0, 1) = -fx * p.x() * iY2;   P(0, 2) = 0.0;
            P(1, 0) =  0.0;       P(1, 1) =  fy * p.z() * iY2;   P(1, 2) = -fy * iY;
            return P.allFinite();
        }

        // ── 360 models: seam-aware central difference on project() itself ────────────────────────
        const double r = p.norm();
        if (r < 1e-6) return false;
        const double W = static_cast<double>(cam.get_width());
        const double hstep = 1e-4 * std::max(1.0, r);          // metres

        for (int j = 0; j < 3; ++j)
        {
            Eigen::Vector3d pp = p, pm = p;
            pp[j] += hstep;
            pm[j] -= hstep;
            const Eigen::Vector2d up = cam.project(pp);
            const Eigen::Vector2d um = cam.project(pm);
            if (not up.allFinite() or not um.allFinite()) return false;

            // The panorama wraps: two columns straddling the seam differ by ~W but are ADJACENT.
            // Fold du into (-W/2, W/2] before dividing, exactly as ricoh_projection_overlay.cpp:99
            // does when it splits an edge at the seam. Without this the yaw column of P is ~W/2h.
            double du = up.x() - um.x();
            if (W > 0.0)
            {
                while (du >  0.5 * W) du -= W;
                while (du <= -0.5 * W) du += W;
            }
            P(0, j) = du               / (2.0 * hstep);
            P(1, j) = (up.y() - um.y()) / (2.0 * hstep);
        }
        return P.allFinite();
    }

    /// Same, by central difference, for EVERY model. Used by gn_selftest to validate the analytic
    /// pinhole branch against project() itself. Not on the hot path.
    inline bool project_jacobian_numeric(const DSR::CameraAPI& cam, const Eigen::Vector3d& p,
                                         Eigen::Matrix<double, 2, 3>& P)
    {
        const double r = p.norm();
        if (r < 1e-6) return false;
        const double W = static_cast<double>(cam.get_width());
        const double hstep = 1e-4 * std::max(1.0, r);
        for (int j = 0; j < 3; ++j)
        {
            Eigen::Vector3d pp = p, pm = p;
            pp[j] += hstep;
            pm[j] -= hstep;
            const Eigen::Vector2d up = cam.project(pp);
            const Eigen::Vector2d um = cam.project(pm);
            if (not up.allFinite() or not um.allFinite()) return false;
            double du = up.x() - um.x();
            if (cam.get_projection_model() != DSR::CameraAPI::ProjectionModel::Pinhole and W > 0.0)
            {
                while (du >  0.5 * W) du -= W;
                while (du <= -0.5 * W) du += W;
            }
            P(0, j) = du                / (2.0 * hstep);
            P(1, j) = (up.y() - um.y()) / (2.0 * hstep);
        }
        return P.allFinite();
    }

    // ── Reduce a CameraAPI to plain numbers, by asking it, not by assuming ───────────────────────
    /// Builds a CameraModel the torch mirror can use without DSR.
    ///
    /// For the 360 models the two panorama-convention parameters are PRIVATE in CameraAPI, so we
    /// recover them from project() itself. With  u = ((sign*atan2(x,y) + offset)/2pi + 0.5)*W :
    ///     probe (0,1,0) -> atan2 = 0      => u0 = (offset/2pi + 0.5)*W        => offset
    ///     probe (1,0,0) -> atan2 = pi/2   => u1 - u0 = sign*W/4  (mod W)      => sign
    /// Then the result is VERIFIED against project() on a spread of directions; `valid` is false if
    /// the round-trip disagrees, so a future change to the panorama convention fails loudly at
    /// bring-up rather than silently biasing yaw.
    inline CameraModel calibrate_camera_model(const DSR::CameraAPI& cam, float* max_err_px = nullptr)
    {
        using PM = DSR::CameraAPI::ProjectionModel;
        CameraModel m;
        m.width  = static_cast<float>(cam.get_width());
        m.height = static_cast<float>(cam.get_height());
        m.fx = cam.get_focal_x();
        m.fy = cam.get_focal_y();
        m.cx = 0.5f * m.width;      // CameraAPI's ctor sets centre = size/2; there is no cx/cy attribute
        m.cy = 0.5f * m.height;
        if (m.width <= 0.f or m.height <= 0.f) return m;

        switch (cam.get_projection_model())
        {
            case PM::Pinhole:     m.kind = CameraModel::Kind::Pinhole;     break;
            case PM::Equirectangular: m.kind = CameraModel::Kind::Equirect; break;
            default:              m.kind = CameraModel::Kind::Cylindrical; break;
        }

        if (m.kind != CameraModel::Kind::Pinhole)
        {
            const double W = m.width;
            const double u0 = cam.project(Eigen::Vector3d(0.0, 1.0, 0.0)).x();   // azimuth 0
            const double u1 = cam.project(Eigen::Vector3d(1.0, 0.0, 0.0)).x();   // azimuth +pi/2
            m.azimuth_offset = static_cast<float>((u0 / W - 0.5) * 2.0 * M_PI);
            double du = u1 - u0;
            while (du >  0.5 * W) du -= W;
            while (du <= -0.5 * W) du += W;
            m.azimuth_sign = (du >= 0.0) ? 1.f : -1.f;
        }

        // Verify the reduced model reproduces project() on a spread of directions.
        float worst = 0.f;
        const double dirs[][3] = {{0,1,0},{1,1,0},{-1,1,0},{0,1,1},{0,1,-1},{1,2,1},{-2,1,-1},{0.3,1.0,0.6}};
        for (const auto& d : dirs)
        {
            const Eigen::Vector3d p(d[0], d[1], d[2]);
            if (m.kind == CameraModel::Kind::Pinhole and p.y() <= 1e-6) continue;
            const Eigen::Vector2d ref = cam.project(p);
            if (not ref.allFinite()) continue;
            Eigen::Vector2d mine;
            if (not project_with_model(m, p, mine)) continue;
            double du = mine.x() - ref.x();
            if (m.kind != CameraModel::Kind::Pinhole)
            {
                while (du >  0.5 * m.width) du -= m.width;
                while (du <= -0.5 * m.width) du += m.width;
            }
            worst = std::max(worst, static_cast<float>(std::max(std::fabs(du), std::fabs(mine.y() - ref.y()))));
        }
        if (max_err_px) *max_err_px = worst;
        m.valid = (worst < 1e-2f);      // sub-hundredth of a pixel: this is a reproduction, not a fit
        return m;
    }

}  // namespace rc::img
