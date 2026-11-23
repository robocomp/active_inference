#pragma once
// ── THE ROOM'S COVERAGE OFFERS IN NATS (2026-11-23, AuRo journal item A) ────────────────────────────────────
//
// While a room is being built its afford_room gain came from plan_coverage in SQUARE METRES (unknown floor a view
// would reveal) or, in the camera phase, in METRES of boundary wall a look would put before the camera — and was
// published to the controller as if it were nats, next to object and door offers that really are nats. This header
// turns those quantities into the expected information of the view, on the shared currency:
//
//   · UNKNOWN floor is floor whose occupancy has not been observed. Occupancy is correlated over a length ℓ, so an
//     unknown area A holds A/ℓ² independent regions, each carrying h(π_occ) nats under the prior π_occ = P(occupied),
//     all of which one observation resolves. A view that reaches A m² of unknown floor offers (A/ℓ²)·h(π_occ).
//   · a boundary SEGMENT one door width long (the camera phase's own unit: camera_coverage.h price = w_m·p) is either
//     a door or wall; under the prior π_door it carries h(π_door). A look that covers L metres offers (L/w)·h(π_door).
//
// Both priors are Beta, learned online from the rooms the robot has finished (occupied vs free cells; doors found vs
// door-width segments of wall). Their starting values are stated priors, not tuned constants.
//
// The ranking inside plan_coverage is unchanged (one positive factor), and so is every explored/stop belief
// (room_explored.h p_stop is a ratio of gain to price, both converted alike). ℓ, π_occ and π_door are priors learnt
// from the rooms the robot finishes.
#include <algorithm>
#include <cmath>

namespace rc::coverage
{
/// Binary entropy in nats; 0 at p ∈ {0, 1}, ln 2 at ½.
[[nodiscard]] inline double binary_entropy(double p)
{
    p = std::clamp(p, 0.0, 1.0);
    const auto t = [](double x) { return x > 0.0 ? -x * std::log(x) : 0.0; };
    return t(p) + t(1.0 - p);
}

/// Beta(a, b) prior over a Bernoulli rate, learned by counts. The predictive P(success) is its mean.
struct BetaPrior
{
    double a = 1.0, b = 1.0;
    [[nodiscard]] double mean() const { return a / (a + b); }
    /// Information of one unobserved Bernoulli under this prior (predictive entropy).
    [[nodiscard]] double nats_per_unit() const { return binary_entropy(mean()); }
    void add(double successes, double failures)
    {
        a += std::max(0.0, successes);
        b += std::max(0.0, failures);
    }
};

/// The occupancy CORRELATION LENGTH ℓ (m): floor occupancy is not independent cell by cell — furniture and walls
/// occupy patches — so the information of an unknown area is that of area/ℓ² independent regions, not of its raster
/// cells (owner 2026-11-23: per-cell counting made 1 m² worth 277 nats at 5 cm). A log-normal-style running belief:
/// the prior ℓ₀ counts as `n` observations, each finished room adds its measured ℓ (FreeSpacePolygon autocorrelation).
struct LengthPrior
{
    double log_mean = std::log(0.3), n = 1.0;
    [[nodiscard]] double value() const { return std::exp(log_mean); }
    void add(double ell_m, double weight = 1.0)
    {
        if (not (ell_m > 0.0) or not (weight > 0.0)) return;
        log_mean = (n * log_mean + weight * std::log(ell_m)) / (n + weight);
        n += weight;
    }
};

/// m² of unknown floor → nats: area / max(ℓ, cell)² independent regions × h(P(occupied)).
[[nodiscard]] inline float area_to_nats(float area_m2, float cell_m, const BetaPrior& occ, const LengthPrior& ell = {})
{
    if (not (cell_m > 0.f)) return 0.f;
    const double l = std::max(static_cast<double>(cell_m), ell.value());
    return static_cast<float>(static_cast<double>(area_m2) / (l * l) * occ.nats_per_unit());
}

/// m of boundary wall a look covers → nats, given the door width (the camera phase's unit) and the door prior.
[[nodiscard]] inline float wall_to_nats(float length_m, float door_w_m, const BetaPrior& door)
{
    if (not (door_w_m > 0.f)) return 0.f;
    return static_cast<float>(static_cast<double>(length_m) / door_w_m * door.nats_per_unit());
}
}   // namespace rc::coverage
