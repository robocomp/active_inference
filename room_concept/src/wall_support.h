#pragma once
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
// P(THIS OUTLINE EDGE IS A REAL WALL | the free-space raster) (2026-11-12, clean55/56 false recognitions).
//
// Published as obj_exist_prob on each wall node; ltsm_agent's per-edge extent (room_chain.h) reads it as the probability
// that the remembered room really ENDS at that edge. It replaces "the fraction of the wall seen", which made a real wall
// with furniture in front of it read 0.7 (so ~25% of an unseen extension leaked through it at metre scale) and an edge
// nobody could see (viewed edge-on) read ~0, i.e. "the room surely continues".
//
// Two hypotheses per edge, compared by a likelihood ratio, with a prior:
//   R  a wall stands on the edge (within the refiner's fitted wall noise sigma);
//   N  the outline closed the edge across space that holds no wall there.
// The observation, per stretch of one raster cell along the edge: walking along the edge's normal from the room side, the
// contiguous run of FREE cells that reaches the edge, and where it ENDS (signed outward distance d of the first surface
// cell's measured position, else the far face of the run's last free cell). A run that ends is where the beams that
// swept it ended, i.e. a return. Per stretch
//     p(d | R) = Phi(-d/sigma) [an occluder in front, same as under N] + (1 - eps) R_eff N(d; 0, sigma) [the wall]
//                + eps Phi(d/sigma) [a beam through a wall is an outlier return: the refiner's one eps]
//     p(d | N) = 1 (in units of 1/R_eff: the run ends at whatever stands beyond, R_eff = A / P the room's own scale for
//                where a return lands, the refiner's surface-outlier density)
// so a run that ends ON the edge says R (by ~ R_eff N(0)), one that runs on beyond it says N (by eps), one stopped well in
// front of it says nothing (ratio 1: OCCLUSION IS NOT ABSENCE), and a stretch no run reaches at all is never observed and
// contributes nothing. sigma, eps are the refiner's EM fits under the current outline, R_eff its own; no new constant.
//
// EFFECTIVE COUNT. Adjacent cells along an edge are not independent looks: they come from the same scans and share their
// pose error, which moves a whole wall at once (the common-mode lesson of the Woodbury notes in CLAUDE.md). The refiner
// already measured how far that correlation runs: ell, the residual autocorrelation length along walls, and weights each
// cell by cell / ell. The same weight is used here, for runs that end on the edge and for runs that pass it alike: an
// edge of length L carries ~L / ell independent observations, never L / cell. (The per-resolvable-beam counting of the
// refiner's free term solves a different problem, the count of cells BEHIND one crossing; here each stretch contributes
// one observation whatever lies behind it, so that count cannot grow with depth.)
//
//     logit P(R | raster) = logit(prior) + (cell / ell) * sum_stretches conf * log(p(d | R) / p(d | N))
// conf = the deciding cell's occupancy posterior (FreeSpacePolygon::evidence_cells), the refiner's per-cell weight.
// FLAGGED: prior = 1/2 — no evidence of how often an outline edge nobody observed is a wall; an unobserved edge reads 1/2.
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <numbers>
#include <unordered_map>
#include <vector>

namespace rc::walls
{
struct SupportParams
{
    float cell  = 0.05f;   ///< m, the evidence raster
    float sigma = 0.03f;   ///< m, the refiner's fitted wall noise
    float eps   = 0.1f;    ///< the refiner's fitted outlier rate
    float ell   = 0.5f;    ///< m, the refiner's fitted correlation length along walls
    float r_eff = 1.f;     ///< m, A / P of the outline
    float prior = 0.5f;    ///< FLAGGED, see the header
};

/// The evidence raster as a cell lookup: surface cells keep their measured (mean-return) position, free cells their centre.
class EvidenceRaster
{
public:
    EvidenceRaster(const std::vector<Eigen::Vector2f>& surf, const std::vector<float>& conf_surf,
                   const std::vector<Eigen::Vector2f>& free, const std::vector<float>& conf_free, float cell)
        : cell_(cell)
    {
        // the grid's offset from the free cells' centres (x0 + (i + 1/2) cell): surface positions are means inside a cell
        if (not free.empty())
        {
            off_ = {std::remainder(free.front().x() - 0.5f * cell, cell), std::remainder(free.front().y() - 0.5f * cell, cell)};
        }
        for (std::size_t i = 0; i < free.size(); ++i) cells_[key(free[i])] = {false, i < conf_free.size() ? conf_free[i] : 1.f, free[i]};
        for (std::size_t i = 0; i < surf.size(); ++i) cells_[key(surf[i])] = {true, i < conf_surf.size() ? conf_surf[i] : 1.f, surf[i]};
    }
    struct Cell { bool surface = false; float conf = 1.f; Eigen::Vector2f pos = Eigen::Vector2f::Zero(); };
    [[nodiscard]] const Cell* at(const Eigen::Vector2f& p) const
    {
        const auto it = cells_.find(key(p));
        return it == cells_.end() ? nullptr : &it->second;
    }
    [[nodiscard]] float cell() const { return cell_; }

private:
    [[nodiscard]] std::int64_t key(const Eigen::Vector2f& p) const
    {
        const auto i = static_cast<std::int64_t>(std::floor((p.x() - off_.x()) / cell_));
        const auto j = static_cast<std::int64_t>(std::floor((p.y() - off_.y()) / cell_));
        return (i << 32) ^ (j & 0xffffffffLL);
    }
    float cell_;
    Eigen::Vector2f off_ = Eigen::Vector2f::Zero();
    std::unordered_map<std::int64_t, Cell> cells_;
};

namespace detail
{
inline double phi(double x) { return 0.5 * std::erfc(-x / std::numbers::sqrt2); }

/// log p(d|R)/p(d|N) for one stretch, times the deciding cell's confidence; 0 when no run reaches the stretch.
inline double stretch_llr(const EvidenceRaster& R, const Eigen::Vector2f& q, const Eigen::Vector2f& out, const SupportParams& P)
{
    const float step = 0.5f * R.cell();
    // CPU bounds, not model terms: a run that starts > 1 m inside, or a ratio evaluated > 3 m beyond, has already
    // saturated (occluded: ratio 1; passed: ratio eps).
    constexpr float kMaxIn = 1.f, kMaxOut = 3.f;
    float t = 0.f;
    const EvidenceRaster::Cell* c = nullptr;
    for (; t <= kMaxIn; t += step)
        if ((c = R.at(q - out * t)) != nullptr) break;
    if (c == nullptr) return 0.0;                        // never observed: no evidence either way
    double d = 0.0, w = 1.0;
    if (c->surface) { d = (c->pos - q).dot(out); w = c->conf; }
    else
    {
        float s = -t, last = -t;
        double wl = c->conf;
        const EvidenceRaster::Cell* e = c;
        while (e != nullptr and not e->surface and s <= kMaxOut)
        {
            last = s; wl = e->conf;
            s += step;
            e = R.at(q + out * s);
        }
        if (e != nullptr and e->surface) { d = (e->pos - q).dot(out); w = e->conf; }
        else { d = last + 0.5 * R.cell(); w = wl; }    // far face of the run's last free cell
    }
    const double sg = P.sigma, z = d / sg;
    const double n = std::exp(-0.5 * z * z) / (sg * std::sqrt(2.0 * std::numbers::pi));
    const double lr = phi(-z) + (1.0 - P.eps) * P.r_eff * n + P.eps * phi(z);
    return w * std::log(lr);
}
}   // namespace detail

/// P(R | raster) for the edge a -> b; `out` = its unit OUTWARD normal. `llr_out` (optional): the summed log ratio.
inline float wall_real_prob(const Eigen::Vector2f& a, const Eigen::Vector2f& b, const Eigen::Vector2f& out,
                            const EvidenceRaster& R, const SupportParams& P, double* llr_out = nullptr)
{
    const float len = (b - a).norm();
    const double logit0 = std::log(P.prior / (1.0 - P.prior));
    if (not (len > 0.f) or not (P.ell > 0.f))
    {
        if (llr_out) *llr_out = 0.0;
        return P.prior;
    }
    const Eigen::Vector2f u = (b - a) / len;
    const int n = std::max(1, static_cast<int>(std::ceil(len / R.cell())));
    double sum = 0.0;
    for (int j = 0; j < n; ++j)
        sum += detail::stretch_llr(R, a + u * ((static_cast<float>(j) + 0.5f) * len / static_cast<float>(n)), out, P);
    const double llr = double(R.cell()) / double(P.ell) * sum;
    if (llr_out) *llr_out = llr;
    return static_cast<float>(1.0 / (1.0 + std::exp(-(logit0 + llr))));
}
}   // namespace rc::walls
