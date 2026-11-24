/*
 * common/nbv/standpoint_free.h — P(the robot's footprint is free of obstacles) at an offered standpoint (G6).
 *
 * WHY. The controller refuses an epistemic claim when a wall, object or unbelieved residual cell lies under the
 * body in the turn the claim ends with (Step 0 census: 44 of 55 refusals). The object agent used to offer such a
 * standpoint at full value. This is the occupancy belief the controller will test, read BEFORE the offer is made:
 *
 *     p_free = Π over lattice cells whose centre lies under the footprint of (1 − p_occ(cell))
 *
 * A cell outside the grid has no evidence and uses the grid's prior. The factor is continuous (1 on free ground,
 * 0 with a certainly-occupied cell under the body), so the offer's value is simply multiplied by it: no gate.
 *
 * Pure Eigen/std, DSR-free. Pinned by tests/standpoint_free_test.cpp.
 */
#pragma once

#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include <Eigen/Core>

namespace rc::nbv
{

// A row-major occupancy-probability grid: cell (ix, iy) is p_occ[iy*nx + ix], centred at
// (origin_x + (ix+½)·res_m, origin_y + (iy+½)·res_m). Cells beyond [0,nx)×[0,ny) are unobserved: p_prior.
struct GridView
{
    float origin_x = 0.0f, origin_y = 0.0f, res_m = 0.0f;
    int   nx = 0, ny = 0;
    std::span<const float> p_occ;
    float p_prior = 0.0f;

    bool valid() const
    { return res_m > 0.0f and nx > 0 and ny > 0 and p_occ.size() >= static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny); }
};

namespace detail
{
inline bool inside_polygon(const std::vector<Eigen::Vector2f>& poly, float x, float y)
{
    bool in = false;
    for (std::size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++)
        if ((poly[i].y() > y) != (poly[j].y() > y) and
            x < (poly[j].x() - poly[i].x()) * (y - poly[i].y()) / (poly[j].y() - poly[i].y()) + poly[i].x())
            in = not in;
    return in;
}
}   // namespace detail

// `footprint_poly` is the robot-frame silhouette (RobotFootprint::polygon(): x right, y FORWARD). It is placed at `xy`
// with FORWARD heading `yaw` (forward = (cos yaw, sin yaw)), the same quarter turn RobotFootprint::at_pose_yaw applies.
// An unusable grid (res 0 / short buffer) carries no belief, so it cannot veto anything: 1.
inline double p_footprint_free(const GridView& g, const Eigen::Vector2f& xy, float yaw,
                               std::span<const Eigen::Vector2f> footprint_poly)
{
    if (footprint_poly.size() < 3 or not g.valid())
        return 1.0;
    const float th = yaw - 1.57079633f, c = std::cos(th), s = std::sin(th);
    std::vector<Eigen::Vector2f> w;
    w.reserve(footprint_poly.size());
    float xmn = 1e30f, xmx = -1e30f, ymn = 1e30f, ymx = -1e30f;
    for (const auto& p : footprint_poly)
    {
        const Eigen::Vector2f q{xy.x() + c * p.x() - s * p.y(), xy.y() + s * p.x() + c * p.y()};
        w.push_back(q);
        xmn = std::min(xmn, q.x()); xmx = std::max(xmx, q.x());
        ymn = std::min(ymn, q.y()); ymx = std::max(ymx, q.y());
    }
    const int ix0 = static_cast<int>(std::floor((xmn - g.origin_x) / g.res_m - 0.5f));
    const int ix1 = static_cast<int>(std::ceil ((xmx - g.origin_x) / g.res_m - 0.5f));
    const int iy0 = static_cast<int>(std::floor((ymn - g.origin_y) / g.res_m - 0.5f));
    const int iy1 = static_cast<int>(std::ceil ((ymx - g.origin_y) / g.res_m - 0.5f));
    double p_free = 1.0;
    for (int iy = iy0; iy <= iy1; ++iy)
        for (int ix = ix0; ix <= ix1; ++ix)
        {
            const float cx = g.origin_x + (static_cast<float>(ix) + 0.5f) * g.res_m;
            const float cy = g.origin_y + (static_cast<float>(iy) + 0.5f) * g.res_m;
            if (not detail::inside_polygon(w, cx, cy))
                continue;
            const bool in_grid = ix >= 0 and iy >= 0 and ix < g.nx and iy < g.ny;
            const double p = in_grid ? static_cast<double>(g.p_occ[static_cast<std::size_t>(iy) * static_cast<std::size_t>(g.nx) +
                                                                   static_cast<std::size_t>(ix)])
                                     : static_cast<double>(g.p_prior);
            p_free *= 1.0 - p;
        }
    return p_free;
}

}   // namespace rc::nbv
