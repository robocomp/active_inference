#pragma once
// doorway_necks.h — an unrecognised doorway is a hypothesis, not an extension of the room
// (spec docs/superpowers/specs/2026-11-06-unrecognised-doorway-hypothesis-design.md; live case clean26, 2026-11-09: the start room
// took in a strip of the next room's floor seen through a door not yet found, and that strip then cut the next room in two).
//
// Each neck of an outline (a chord between two vertices along the room's axes, cutting off a part B) is judged twice, in nats, on the
// refiner's own description length and on the cells outside B (B's cells are floor under both explanations and cancel):
//   F_room = DL(outline)            the room includes B, and pays for B's vertices
//   F_door = DL(outline without B) - ln P(door on the wall) + (w - w0)^2 / 2(s0^2 + s_w^2)
// F_room > F_door: B is the next room seen through a door. The width w is read off the LiDAR raster, so its noise s_w is the raster's
// cell (not the camera mask's noise: at 0.20 m a real 2.7 m neck would be cut). A wide mouth pays the door's size term and stays the room.
// The best neck is cut, and the outline judged again until none pays. Pure (Eigen + the refiner).
// Flagged: door_width_m / door_width_sd are the typical door of door_concept (Door.PriorWidthM / PriorWidthStd) and fit the simulated world.
#include "layout_refiner.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

namespace rc::refine
{
struct NeckParams
{
    float door_width_m  = 1.00f;   ///< door_concept's Door.PriorWidthM
    float door_width_sd = 0.06f;   ///< door_concept's Door.PriorWidthStd
    float p_door        = 0.40f;   ///< P(a door on this wall): door_concept's first-door base rate
};

struct Opening
{
    Eigen::Vector2f a{0.f, 0.f}, b{0.f, 0.f};
    float cut_area = 0.f;
    float score = 0.f;   ///< F_room - F_door (nats, > 0)
};

struct NeckResult
{
    std::vector<Eigen::Vector2f> polygon;
    std::vector<Opening> openings;
};

namespace detail
{
using Poly = std::vector<Eigen::Vector2f>;

[[nodiscard]] inline float dist_to_boundary(const Poly& P, const Eigen::Vector2f& q)
{
    float best = std::numeric_limits<float>::max();
    for (std::size_t i = 0, j = P.size() - 1; i < P.size(); j = i++)
    {
        const Eigen::Vector2f e = P[i] - P[j];
        const float t = std::clamp((q - P[j]).dot(e) / std::max(1e-9f, e.squaredNorm()), 0.f, 1.f);
        best = std::min(best, (q - (P[j] + t * e)).norm());
    }
    return best;
}

[[nodiscard]] inline Poly drop_collinear(Poly P)
{
    for (bool changed = true; changed and P.size() > 4;)
    {
        changed = false;
        for (std::size_t i = 0; i < P.size() and not changed; ++i)
        {
            const Eigen::Vector2f u = P[i] - P[(i + P.size() - 1) % P.size()], v = P[(i + 1) % P.size()] - P[i];
            if (std::abs(u.x() * v.y() - u.y() * v.x()) < 0.02f * u.norm() * v.norm() and u.dot(v) > 0.f)
            { P.erase(P.begin() + static_cast<std::ptrdiff_t>(i)); changed = true; }
        }
    }
    return P;
}

/// The direction (mod 90 deg) that the most wall length agrees with.
[[nodiscard]] inline float room_axis(const Poly& P)
{
    float best = 0.f, best_w = -1.f;
    for (float th = 0.f; th < 1.5708f; th += 0.01f)
    {
        float w = 0.f;
        for (std::size_t i = 0, j = P.size() - 1; i < P.size(); j = i++)
        {
            const Eigen::Vector2f e = P[i] - P[j];
            const float a = std::remainder(std::atan2(e.y(), e.x()) - th, 1.5708f);
            w += e.norm() * std::exp(-0.5f * a * a / 0.0025f);
        }
        if (w > best_w) { best_w = w; best = th; }
    }
    return best;
}
}  // namespace detail

/// Cut every neck the door explains better than the room does.
[[nodiscard]] inline NeckResult explain_openings(const LayoutRefiner& refiner, const std::vector<Eigen::Vector2f>& outline,
                                                 const std::vector<Eigen::Vector2f>& surf, const std::vector<Eigen::Vector2f>& free,
                                                 const std::vector<float>& conf_surf, const std::vector<float>& conf_free,
                                                 const NeckParams& np = {},
                                                 const std::vector<Eigen::Vector2f>& origin_free = {})
{
    using detail::Poly;
    NeckResult out{.polygon = outline, .openings = {}};
    if (surf.size() < 20) return out;
    const float cell = refiner.params().cell;
    const float cell_margin = 3.f * cell;   // the cells of B's own walls lie just outside B
    const auto in = [](const Poly& P, const Eigen::Vector2f& q) { return freespace::point_in_polygon(q, P); };
    while (out.polygon.size() > 4)
    {
        const Poly& P = out.polygon;
        const int n = static_cast<int>(P.size());
        const float th = detail::room_axis(P);
        std::optional<Opening> best;
        Poly best_kept;
        for (int i = 0; i < n; ++i)
            for (int j = i + 2; j < n; ++j)
            {
                if (i == 0 and j == n - 1) continue;
                const Eigen::Vector2f A = P[i], B = P[j], d = B - A;
                const float w = d.norm();
                // ★CPU budget, flagged: necks along the room's axes only (a door lies in a wall), within 8 deg
                if (w < 2.f * cell or std::abs(std::remainder(std::atan2(d.y(), d.x()) - th, 1.5708f)) > 0.14f) continue;
                bool inside = in(P, A + 0.5f * d);
                for (float t = 0.02f; t < w - 0.02f and inside; t += cell)
                    inside = in(P, A + d * (t / w)) or detail::dist_to_boundary(P, A + d * (t / w)) < 0.2f * cell;
                if (not inside) continue;
                Poly p1(P.begin() + i, P.begin() + j + 1), p2(P.begin() + j, P.end());
                p2.insert(p2.end(), P.begin(), P.begin() + i + 1);
                const bool keep1 = std::abs(freespace::signed_area(p1)) >= std::abs(freespace::signed_area(p2));
                const Poly kept = detail::drop_collinear(keep1 ? p1 : p2), &cut = keep1 ? p2 : p1;
                const float cut_area = std::abs(freespace::signed_area(cut));
                if (kept.size() < 4 or cut_area <= cell * cell) continue;
                // The door hypothesis says floor was SEEN beyond the opening; a kink in a wall has none (only cells of the wall's own band).
                const Eigen::Vector2f u = d / w, nrm(-u.y(), u.x());
                if (std::ranges::none_of(free, [&](const Eigen::Vector2f& q) { return in(cut, q) and std::abs((q - A).dot(nrm)) > cell_margin; })) continue;
                const auto beyond = [&](const Eigen::Vector2f& q)
                {
                    if (in(cut, q)) return true;
                    return detail::dist_to_boundary(cut, q) < cell_margin and not in(kept, q) and detail::dist_to_boundary(kept, q) > cell_margin;
                };
                Poly S, F, FO; std::vector<float> cs, cf;
                for (std::size_t k = 0; k < surf.size(); ++k)
                    if (not beyond(surf[k])) { S.push_back(surf[k]); if (conf_surf.size() == surf.size()) cs.push_back(conf_surf[k]); }
                for (std::size_t k = 0; k < free.size(); ++k)
                    if (not beyond(free[k]))
                    {
                        F.push_back(free[k]);
                        if (conf_free.size() == free.size()) cf.push_back(conf_free[k]);
                        if (origin_free.size() == free.size()) FO.push_back(origin_free[k]);
                    }
                const auto dl = refiner.description_lengths({P, kept}, kept, S, F, cs, cf, FO);
                const float dw = w - np.door_width_m;
                const float score = dl[0] - dl[1] + std::log(np.p_door)
                                    - 0.5f * dw * dw / (np.door_width_sd * np.door_width_sd + cell * cell);
                if (std::isfinite(score) and score > 0.f and (not best.has_value() or score > best->score))
                { best = Opening{.a = A, .b = B, .cut_area = cut_area, .score = score}; best_kept = kept; }
            }
        if (not best.has_value()) break;
        out.openings.push_back(*best);
        out.polygon = std::move(best_kept);
    }
    return out;
}
}  // namespace rc::refine
