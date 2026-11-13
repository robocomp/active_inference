// place_belief.h — Task 8 of the door-existence model: THE ONE BELIEF of a door place (spec §3.1, §3.1b).
//
// ★THREE HYPOTHESES (2026-11-08): D a door, N nothing door-like, K something door-shaped that is not a door. Each has its own prior on the place:
//  pi_D the per-wall door base rate (place_prior.h: 0.4 first door, 0.01 further), pi_K the per-wall look-alike rate (Existence.LookalikeProb, default 0.002),
//  pi_N = 1 − pi_D − pi_K. The detectors do not separate D from K; only a physical test does. The one formula (computed in L_evidence below, candidates and born doors alike):
//
//     L(θ) = ln(pi_D) − ln(pi_N + pi_K·exp(Sc)) + Sc + Sp + band + size + carried
//
//  which is prior.total() + Sp + saturated_vision(Sc, lambda) with lambda = pi_K / (1 − pi_D) (the base-rate term prior.base_rate = ln(pi_D / (1 − pi_D)) is
//  counted once, in prior.total()). Sc = the perceptual evidence (zed, ricoh, geometry, support over the ledger's views, plus carried_sc); Sp = the physical evidence (actuation,
//  crossing, carried). Vision saturates near ln(pi_D / pi_K): +5.3 for a first door, +1.6 for a further door, so a further door can still be born from vision. 25 agreeing views of one object are one object.
//  For Sc ≪ 0 absence evidence behaves as the old sum.
//  ★A CROSSING IS PROOF (Pablo 2026-11-08): P(crossed | not a door) = 0. `crossed` is that fact, latched, and carried across a swap and a restore: a crossed place gets no delete line and no delete verdict, whatever its band, size or views.
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
/// ⚑MODEL CONSTANT, flagged (CLAUDE.md): pi_K, the per-wall prior of a door-shaped thing that is not a door (Existence.LookalikeProb). A sensor-model constant, to be
/// learned offline like the detector rates (tools/fit_detector_envelopes.py does not fit it yet).
inline constexpr float kLookalikeProbDefault = 0.002f;

/// lambda = pi_K / (1 − pi_D), with pi_D from the base-rate log-odds (0 = no prior known yet, pi_D = 1/2): the share of "not a door" that is a look-alike.
/// ⚑DOMAIN GUARD, not a model threshold: pi_N must stay positive; lambda ≥ 1 would mean pi_K + pi_D ≥ 1, an impossible config.
[[nodiscard]] inline double lookalike_lambda(double pi_K, double base_rate_logodds)
{
    return std::min(pi_K * (1.0 + std::exp(base_rate_logodds)), 1.0 - 1e-6);
}

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
    float lookalike = kLookalikeProbDefault;   // pi_K (Existence.LookalikeProb): stamped by observe_place from the config every frame
    bool crossed = false;              // proof: the robot's body went through this place AND the crossing STOOD (the header): no delete line, ever
    // ★THE PROOF WAITS FOR THE CONFIRMATION (clean16, 2026-11-09): a phantom door whose plane the body crossed latched `crossed` and could never be deleted.
    //  The body going through is `crossing_pending`: the door keeps its protection from deletion while the crossing is open, and the proof is final when the
    //  room switch stood (confirm_crossing, door_concept's worker: the entered room established, or left by another crossing). A crossing recognition
    //  WITHDRAWS (the robot never left) is the opposite proof (refute_crossing).
    bool crossing_pending = false;

    /// The body went through this place: pending, unless the proof already stands.
    void note_crossing() { if (not crossed) crossing_pending = true; }
    /// The room switch stood: the pending crossing is proof. No-op without a pending crossing.
    void confirm_crossing() { if (crossing_pending) crossed = true; crossing_pending = false; }
    /// Deleting this door would strand a crossing: pending or proven.
    [[nodiscard]] bool crossing_protected() const { return crossed or crossing_pending; }
    /// THE CROSSING DID NOT HAPPEN: the body went through this plane and found no doorway, P(that | a door here) = 0, the mirror of "a crossing is proof".
    /// The crossing evidence goes, and the place is held to the structural prior: the physical term cancels anything vision can add (saturated_vision < −ln λ), so
    /// no fresh look at the look-alike can rebuild the door. λ is the model's own look-alike share: nothing else is tuned.
    void refute_crossing()
    {
        crossed = false;
        crossing_pending = false;
        crossing = float(std::log(lambda()));
    }

    [[nodiscard]] double lambda() const { return lookalike_lambda(lookalike, prior.base_rate); }

    /// Sc, the perceptual evidence: door-shaped-thing evidence, carried + the ledger's views.
    [[nodiscard]] double Sc() const { return double(carried_sc) + ledger.perceptual_sum(); }
    /// Sp, the physical evidence: actuation (ledger), crossing and what is carried outside vision.
    [[nodiscard]] double Sp() const { return double(carried) + crossing + ledger.physical_sum(); }
    /// ln((1−lambda) + lambda·exp(Sc)), the saturation term the belief subtracts (logged; L_evidence uses the same computation).
    [[nodiscard]] double saturation_term() const { return Sc() - saturated_vision(Sc(), lambda()); }
    /// What a memory row stores: everything but the structural prior (re-entered from the room it is restored into).
    [[nodiscard]] float L_evidence() const { return float(Sp() + saturated_vision(Sc(), lambda())); }
    /// The log-odds of "a door is here" against "nothing door-like or a look-alike here" (the header's formula).
    [[nodiscard]] float L() const { return prior.total() + L_evidence(); }
    [[nodiscard]] float p() const { return 1.0f / (1.0f + std::exp(-L())); }

    /// Remember evidence: the (carried, carried_sc) that make L_evidence() == `evidence` with the ledger as it is now and `physical` of it being Sp.
    /// (Set `ledger` and `lookalike` first.) ⚑DOMAIN GUARD, not a model threshold: the perceptual part V = evidence − physical may only be inverted
    /// below −ln(lambda); a V at or above that bound (float rounding; the row written under another lambda) keeps its surplus outside the saturation,
    /// so the remembered belief is reproduced exactly and vision still cannot add to it.
    void remember(float evidence, float physical)
    {
        const double lam = lambda();
        const double v = double(evidence) - physical;
        const double held = std::min(v, -std::log(lam) - double(std::numeric_limits<float>::epsilon()));
        carried = float(double(physical) + (v - held)) - ledger.physical_sum();
        carried_sc = float(vision_of_saturated(held, lam)) - ledger.perceptual_sum();
    }
};

/// What one existence update did to one place, per channel (spec §3.8). NaN = NOT COMPUTED this cycle (printed `nan` in the
/// CSV and `—` on stdout, never a stale value). A channel's dl is the change it made to the ledger sum this cycle.
struct FitSupport { float x = 0.f; int sensor = 0; float coverage = 1.f; float overlap = 1.f; };   // the fit's clutter share, the camera (0 ZED, else ricoh) it came from,
                                                                              // and the share of the opening the mask samples (rc::door::aperture_coverage)

struct ExistenceTrace
{
    static constexpr float kNan = std::numeric_limits<float>::quiet_NaN();
    float zed_dl = kNan, ric_dl = kNan, geo_dl = kNan, act_dl = kNan;
    float zed_q = kNan, ric_q = kNan;          // the camera's detect_prob this frame
    float zed_hit = kNan, ric_hit = kNan;      // the association weight of this frame's detection (0 = a miss)
    float kappa = kNan;
    float geo_x = kNan, geo_mu0 = kNan, geo_mu1 = kNan, geo_s = kNan;
    float sup_x = kNan, sup_dl = kNan, sup_c = kNan, sup_q = kNan;   // sup_c: the mask's coverage of the opening (the trial's weight with κ); sup_q: P(the window was on the opening)
            // the fit's clutter share this cycle and what ℓ_sup changed the ledger by (NaN = no fit)
    int zed_view = -1, ric_view = -1;          // ledger index of the view this frame joined (-1 = none)
    float delete_line = kNan;
};
}  // namespace rc::door::core
