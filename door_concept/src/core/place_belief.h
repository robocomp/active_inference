// place_belief.h — Task 8 of the door-existence model: THE ONE BELIEF of a door place (spec §3.1, §3.1b).
//
// ★THREE HYPOTHESES (2026-11-08): D a door, N nothing door-like, K something door-shaped that is not a door. The detectors do not
//  separate D from K, only a physical test does. Marginalising K out of "not a door", with lambda = P(K | not a door, the detectors would
//  fire), gives the one formula (Pablo 2026-11-08, spec §3.1b), computed in ONE place (L_evidence below, candidates and born doors alike):
//
//     L(θ) = structural prior + carried + Sc + Sp − ln((1 − lambda) + lambda·exp(Sc))
//
//  Sc = the perceptual evidence (zed, ricoh, geometry, support over the ledger's views, plus carried_sc); Sp = the physical evidence (actuation,
//  crossing, carried). For Sc ≫ 0 vision saturates at −ln(lambda) above the priors: 25 agreeing views of one object are one object. For Sc < 0 the
//  term is ≈ −ln(1−lambda): absence evidence behaves as before.
//
// `prior` is entered once and REPLACED when it changes (place_prior.h), never accumulated. `crossing` is rc::door::crossing_logodds of a crossing
// observed through this aperture. The ledger holds one record per viewpoint for every evidence channel (view_ledger.h).
// A memory row stores L_evidence = L() − prior.total() AND the physical part Sp (PlaceRow::physical).
//
// ★WHAT IS REMEMBERED (remember() below). A remembered L_evidence has vision already saturated inside it; adding fresh Sc on top would saturate
//  a SECOND time, and a door restored from memory could climb to prior + 2·(−ln lambda) from vision alone (double counting). So what is carried splits:
//    `carried`      physical evidence, outside the saturation (it survives a revisit as it is);
//    `carried_sc`   the perceptual evidence, back on the Sc scale (inverse of the saturation), INSIDE the saturation: new views add to it there.
//  Aging (hazard) reduces both parts alike; the inversion makes the aged part of vision re-climb to the same bound with new views.
// P(door) = σ(L). Header-only, DSR-free; door_fit_state.h embeds one per door, the DoorCore one per pending candidate.
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include "place_prior.h"
#include "view_ledger.h"

namespace rc::door::core
{
/// ⚑MODEL CONSTANT, flagged (CLAUDE.md): P(a door-shaped non-door | not a door, the detectors would fire). A sensor-model constant, to be
/// learned offline like the detector rates (tools/fit_detector_envelopes.py does not fit it yet). Existence.LookalikeProb.
inline constexpr float kLookalikeProbDefault = 0.05f;

/// Sc − ln((1−lambda) + lambda·exp(Sc)) = −ln((1−lambda)·exp(−Sc) + lambda), as a log-sum: finite for any finite Sc, no exp(Sc) to overflow.
[[nodiscard]] inline double saturated_vision(double Sc, double lambda)
{
    const double a = std::log1p(-lambda) - Sc, b = std::log(lambda);
    const double hi = std::max(a, b), lo = std::min(a, b);
    return -(hi + std::log1p(std::exp(lo - hi)));
}
/// The inverse of saturated_vision on V < −ln(lambda), the Sc that saturates to V. Stable at V ≪ 0 (no exp(−V)).
[[nodiscard]] inline double vision_of_saturated(double V, double lambda)
{
    return std::log1p(-lambda) + V - std::log1p(-lambda * std::exp(V));
}

struct PlaceBelief
{
    PlacePrior prior;
    float carried = 0.f, crossing = 0.f;
    float carried_sc = 0.f;            // perceptual evidence remembered from views no longer in the ledger, on the Sc scale (remember())
    ViewLedger ledger;
    float lookalike = kLookalikeProbDefault;   // lambda (Existence.LookalikeProb): stamped by observe_place from the config every frame

    /// Sc, the perceptual evidence: door-shaped-thing evidence, carried + the ledger's views.
    [[nodiscard]] double Sc() const { return double(carried_sc) + ledger.perceptual_sum(); }
    /// Sp, the physical evidence: actuation (ledger), crossing and what is carried outside vision.
    [[nodiscard]] double Sp() const { return double(carried) + crossing + ledger.physical_sum(); }
    /// ln((1−lambda) + lambda·exp(Sc)), the saturation term the belief subtracts (logged; L_evidence uses the same computation).
    [[nodiscard]] double saturation_term() const { return Sc() - saturated_vision(Sc(), lookalike); }
    /// What a memory row stores: everything but the structural prior (re-entered from the room it is restored into).
    [[nodiscard]] float L_evidence() const { return float(Sp() + saturated_vision(Sc(), lookalike)); }
    /// The log-odds of "a door is here" against "nothing door-like or a look-alike here" (the header's formula).
    [[nodiscard]] float L() const { return prior.total() + L_evidence(); }
    [[nodiscard]] float p() const { return 1.0f / (1.0f + std::exp(-L())); }

    /// Remember evidence: the (carried, carried_sc) that make L_evidence() == `evidence` with the ledger as it is now and `physical` of it being Sp.
    /// (Set `ledger` and `lookalike` first.) ⚑DOMAIN GUARD, not a model threshold: the perceptual part V = evidence − physical may only be inverted
    /// below −ln(lambda); a V at or above that bound (float rounding; the row written under another lambda) keeps its surplus outside the saturation,
    /// so the remembered belief is reproduced exactly and vision still cannot add to it.
    void remember(float evidence, float physical)
    {
        const double lam = lookalike;
        const double v = double(evidence) - physical;
        const double held = std::min(v, -std::log(lam) - double(std::numeric_limits<float>::epsilon()));
        carried = float(double(physical) + (v - held)) - ledger.physical_sum();
        carried_sc = float(vision_of_saturated(held, lam)) - ledger.perceptual_sum();
    }
};

/// What one existence update did to one place, per channel (spec §3.8). NaN = NOT COMPUTED this cycle (printed `nan` in the
/// CSV and `—` on stdout, never a stale value). A channel's dl is the change it made to the ledger sum this cycle.
struct FitSupport { float x = 0.f; int sensor = 0; };   // the fit's clutter share and the camera (0 ZED, else ricoh) it came from

struct ExistenceTrace
{
    static constexpr float kNan = std::numeric_limits<float>::quiet_NaN();
    float zed_dl = kNan, ric_dl = kNan, geo_dl = kNan, act_dl = kNan;
    float zed_q = kNan, ric_q = kNan;          // the camera's detect_prob this frame
    float zed_hit = kNan, ric_hit = kNan;      // the association weight of this frame's detection (0 = a miss)
    float kappa = kNan;
    float geo_x = kNan, geo_mu0 = kNan, geo_mu1 = kNan, geo_s = kNan;
    float sup_x = kNan, sup_dl = kNan;         // the fit's clutter share this cycle and what ℓ_sup changed the ledger by (NaN = no fit)
    int zed_view = -1, ric_view = -1;          // ledger index of the view this frame joined (-1 = none)
    float delete_line = kNan;
};
}  // namespace rc::door::core
