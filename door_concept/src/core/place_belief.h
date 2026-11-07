// place_belief.h — Task 8 of the door-existence model: THE ONE BELIEF of a door place (spec §3.1).
//
//   L(θ) = structural prior + carried evidence + crossing evidence + Σ over views ℓ_v
//
// `prior` is entered once and REPLACED when it changes (place_prior.h), never accumulated. `carried` is the evidence a
// door brings from where it was last believed in (a matched memory row's aged L_evidence; Task 9: the crossed face's
// L_evidence). `crossing` is rc::door::crossing_logodds of a crossing observed through this aperture. The ledger holds one
// record per viewpoint for every evidence channel (view_ledger.h). A memory row stores L_evidence = L() − prior.total().
// P(door) = σ(L). Header-only, DSR-free; door_fit_state.h embeds one per door, the DoorCore one per pending candidate.
#pragma once
#include <cmath>
#include <limits>
#include "place_prior.h"
#include "view_ledger.h"

namespace rc::door::core
{
struct PlaceBelief
{
    PlacePrior prior;
    float carried = 0.f, crossing = 0.f;
    ViewLedger ledger;
    /// The log-odds of "a door is here" against "wall or clutter here".
    [[nodiscard]] float L() const { return prior.total() + carried + crossing + ledger.sum(); }
    /// What a memory row stores: everything but the structural prior (re-entered from the room it is restored into).
    [[nodiscard]] float L_evidence() const { return carried + crossing + ledger.sum(); }
    [[nodiscard]] float p() const { return 1.0f / (1.0f + std::exp(-L())); }
};

/// What one existence update did to one place, per channel (spec §3.8). NaN = NOT COMPUTED this cycle (printed `nan` in the
/// CSV and `—` on stdout, never a stale value). A channel's dl is the change it made to the ledger sum this cycle.
struct ExistenceTrace
{
    static constexpr float kNan = std::numeric_limits<float>::quiet_NaN();
    float zed_dl = kNan, ric_dl = kNan, geo_dl = kNan, act_dl = kNan;
    float zed_q = kNan, ric_q = kNan;          // the camera's detect_prob this frame
    float zed_hit = kNan, ric_hit = kNan;      // the association weight of this frame's detection (0 = a miss)
    float kappa = kNan;
    float geo_x = kNan, geo_mu0 = kNan, geo_mu1 = kNan, geo_s = kNan;
    int zed_view = -1, ric_view = -1;          // ledger index of the view this frame joined (-1 = none)
    float delete_line = kNan;
};
}  // namespace rc::door::core
