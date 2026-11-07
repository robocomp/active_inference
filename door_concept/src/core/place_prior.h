// place_prior.h — the STRUCTURAL PRIOR of a door place: the wall band, the per-wall base rate and the size term.
// Entered ONCE on the place's belief and REPLACED when it changes, never accumulated per frame (a prior integrated per
// frame is a fake sensor). Header-only, DSR-free.
//
// Build + run the test (from door_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_place_prior.cpp -o /tmp/test_place_prior && /tmp/test_place_prior
//
// ★The size term is the NORMALISED LOG PREDICTIVE DENSITY of the fitted size under the door prior, per axis (ruling R13),
//   NOT a KL to the prior. With w ~ N(ŵ, width_sd²) measured and the door prior N(μ_w, σ_w²), the predictive variance is
//   v = σ_w² + width_sd² and the term is  ln N(ŵ; μ_w, v) − ln N(μ_w; μ_w, v) = −½ (ŵ − μ_w)² / v  (same for height). A KL grows
//   like ln(σ_w / sd) as the fit gets precise, so a WELL-MEASURED TYPICAL door would pay without bound — a prior that punishes good
//   measurement. The predictive pays exactly 0 at the typical size and more the more atypical the size is, however precisely it
//   is measured.
// ★ONE SIZE PRIOR FOR ONE QUANTITY (Task 10 fix round 4, ruling): the door prior (μ_w, σ_w, μ_h, σ_h) is the door FIT's own
//   prior, Door.PriorWidthM ± Door.PriorWidthStd and Door.PriorHeightM ± Door.PriorHeightStd, passed in as SizePrior. There is
//   no second, existence-only size constant: two priors for one door size made every real door pay ~−0.75 nats for being the
//   size the fit holds it at (and scored a ŵ already pulled by the other prior). The spec's typical-real-door value
//   (0.85 ± 0.12 / 2.05 ± 0.06, spec §3.2) is the recommended Door.* value for the real robot (etc/config.toml).
//
// ★CONSTANTS, each flagged (global-constraints): 0.4 / 0.01 are the per-wall base rates (§6).
// ★THE ONE GATE, flagged: wall_groups joins edges whose endpoints lie within 2σ of the wall's line. That 2σ is a χ² gate:
//   two edges are one wall exactly when their offsets are indistinguishable under the face uncertainty σ the band already
//   carries. It is not a free tuning knob: σ is the band's own.
#pragma once
#include "../door_room_band.h"
#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace rc::door::core
{
using Vec2f = Eigen::Vector2f;

/// The door's size prior: the door FIT's own (Door.PriorWidthM/Std, Door.PriorHeightM/Std) - see the header.
struct SizePrior
{
    float w_m, w_sd, h_m, h_sd;
};

/// polygon is a reference (the room's outline outlives the call; signed_boundary_distance takes a vector).
struct PlacePriorInput
{
    const std::vector<Vec2f>& polygon;
    bool proto;                         ///< proto-room: outline is the measured floor. ONE-SIDED frontier: see place_prior()
    rc::door::RoomBand band;
    Vec2f aperture_xy;
    Vec2f wall_dir;                     ///< unit direction of the wall at the aperture (reserved; width is along it)
    float width, height, width_sd, height_sd;
    bool size_resolved;
    int doors_on_wall_before;
    SizePrior size_prior;               ///< the fit's width/height prior (one prior for one quantity)
    float edge_sigma_m = 0.0f;          ///< the proto outline edge's own published uncertainty (m); room_concept publishes none on the room node, so 0
};

struct PlacePrior
{
    float band = 0, base_rate = 0, size = 0;
    [[nodiscard]] float total() const { return band + base_rate + size; }
};

namespace detail
{
// ln N(x; mu, v) − ln N(mu; mu, v)
[[nodiscard]] inline float normalised_log_predictive(float x, float mu, float prior_sd, float meas_sd)
{
    const float v = prior_sd * prior_sd + meas_sd * meas_sd;
    const float d = x - mu;
    return -0.5f * d * d / v;
}
}   // namespace detail

[[nodiscard]] inline PlacePrior place_prior(const PlacePriorInput& in)
{
    PlacePrior p;
    // 0.4 first door on a wall, 0.01 each further one (§6, flagged constants).
    p.base_rate = (in.doors_on_wall_before == 0) ? std::log(0.4f / 0.6f) : std::log(0.01f / 0.99f);

    const float delta = rc::door::signed_boundary_distance(in.polygon, in.aperture_xy);   // NaN: no polygon ⇒ no prior
    if (std::isfinite(delta))
    {
        // ONE-SIDED PROTO FRONTIER, no cutoff. delta < 0 (inside the measured floor) pays the band. delta >= 0 pays band_log_prior(delta)
        // with sigma widened by the outline edge's own uncertainty (hypot; 0 when none is published - the graph carries only
        // delimiting_polygon_x/y): a door AT the frontier (within the outline's error of the edge) pays about 0, one metres beyond the
        // measured floor pays as off-wall. For a promoted room the same formula applies with edge_sigma_m = 0.
        rc::door::RoomBand b = in.band;
        if (in.proto and delta >= 0.0f)
            b.sigma_m = std::hypot(b.sigma_m, in.edge_sigma_m);
        p.band = rc::door::band_log_prior(delta, b);
    }
    if (in.size_resolved)
        p.size = detail::normalised_log_predictive(in.width, in.size_prior.w_m, in.size_prior.w_sd, in.width_sd)
               + detail::normalised_log_predictive(in.height, in.size_prior.h_m, in.size_prior.h_sd, in.height_sd);
    return p;
}

/// edge i (v[i] → v[i+1], wrapping) → wall id, ids consecutive from 0 in order of first appearance. An edge joins the
/// current wall when BOTH its endpoints lie within 2σ of that wall's line (2σ: χ² gate, see the header). This one test
/// covers a collinear continuation, a small angular wobble and a short jog (a 5 cm step is within the face uncertainty);
/// a real corner puts the far endpoint metres off the line, so it starts a new wall. The last edge also tests against
/// the first wall, so a wall split by the polygon's start vertex is one group.
[[nodiscard]] inline std::vector<int> wall_groups(const std::vector<Vec2f>& polygon, float sigma_m)
{
    const std::size_t n = polygon.size();
    std::vector<int> id(n, 0);
    if (n < 3)
        return id;
    const float gate = 2.0f * sigma_m;   // flagged χ² gate
    auto edge = [&](std::size_t i) { return std::pair{polygon[i], polygon[(i + 1) % n]}; };
    auto near_line = [&](std::size_t line_edge, std::size_t e) {
        const auto [a, b] = edge(line_edge);
        const Vec2f d = b - a;
        const float len = d.norm();
        if (len < 1e-9f)
            return false;
        const Vec2f nrm(-d.y() / len, d.x() / len);
        const auto [p, q] = edge(e);
        return std::abs(nrm.dot(p - a)) < gate and std::abs(nrm.dot(q - a)) < gate;
    };
    std::size_t line_edge = 0;
    int next = 1;
    for (std::size_t i = 1; i < n; ++i)
    {
        if (near_line(line_edge, i))
            id[i] = id[line_edge];
        else
        {
            id[i] = next++;
            line_edge = i;
        }
    }
    if (id[n - 1] != 0 and near_line(0, n - 1))
    {
        const int old = id[n - 1];
        for (auto& g : id)
            if (g == old)
                g = 0;
        for (auto& g : id)
            if (g > old)
                --g;
    }
    return id;
}

}   // namespace rc::door::core
