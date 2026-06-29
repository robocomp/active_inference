/*
 * belief_stabilizer.h
 *
 * Generic per-DOF belief stabiliser shared by the CORTEX concept agents (table_concept,
 * bottle_concept, …). Extracted from table_concept's TableFitter so the stabilisation logic lives
 * in ONE place instead of being re-derived per agent.
 *
 * It is object-AGNOSTIC: the per-frame observation Fisher information (the curvature of the model's
 * SDF data-likelihood, ∂SDF/∂θ) is computed by each agent's own model and handed in. The stabiliser
 * then runs, per DOF:
 *
 *   1. CONFIDENCE WEIGHTING — scale this frame's observation precision by w(YOLO score) so a
 *      low-score mask barely affects the belief.                          [weight_observation]
 *   2. KALMAN ACCEPTANCE GAIN — K = obs/(Y_pred+obs) from a Q-bleed information filter, optionally
 *      maturity-stiffened by the accumulated "equivalent views" so a well-seen DOF barely moves on
 *      any one frame.                                                       [compute_acceptance]
 *   3. CUSUM / SPRT GATE — a sequential change-detector on top of the gain: an isolated surprising
 *      frame is rejected (gain→0), only a CONFIDENT, COHERENT run unlocks a DOF and lets it ease
 *      onto the new evidence (deflate, never snap). Position vs size use separate barriers.  [↑]
 *   4. ACCUMULATION — fold the (weighted) observation into the filter (predict + update +
 *      normalised "views"), so precision converges to a finite steady state.   [accumulate]
 *
 * State (StabilizerState<N>) is per-belief-instance; the algorithm + parameters (BeliefStabilizer<N>)
 * are shared across instances. Header-only template so it plugs directly into each agent's existing
 * std::array<float,N> acceptance (table N=8, bottle N=5) with no size adapter.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace rc {

// ── Tunables (mapped 1:1 from each agent's config) ─────────────────────────────────────────────
struct StabilizerParams
{
    bool  fisher_filter_enabled = true;   // master: use the real per-DOF Fisher info (vs a legacy proxy)
    bool  kalman_stiffness      = false;  // drive the acceptance lerp from the Kalman gain (vs a heuristic)
    bool  maturity_stiffness    = true;   // scale the gain by views/(views+accumulated) → mature DOFs stiffen
    float info_decay            = 1.0f;   // normalised-accumulator fading memory (<1 forgets, 1 = pure accumulation)
    float views_half            = 4.0f;   // equivalent-views at which the maturity stiffener halves the gain
    // Quality-aware maturity: the "views" accumulator is normalised by a running quality bar (best
    // single-view info). When a MUCH more informative view arrives (a close approach after far
    // glimpses), it ratchets the bar up and rescales the accumulated views DOWN — so the belief
    // re-opens to the better evidence instead of staying locked at the far-view fit. Off → plain
    // all-time-max normalisation (legacy: every view counts ~1, far hardens as fast as close).
    bool  info_quality_rescale  = true;
    float info_peak_decay       = 1.0f;   // slow forgetting of the quality bar (1 = non-decaying; <1 lets it drift down)
    // The quality bar is an EMA of the high observations (info_peak_ema smooths single-frame obs noise),
    // and it only RATCHETS UP / rescales when a view exceeds the bar by ≥ info_peak_ratchet×. Without
    // this the per-frame obs (which swings orders of magnitude) re-ratchets the bar constantly →
    // accumulator never matures → the DOF stays plastic and wanders. ratchet>1 makes only a GENUINELY
    // better view (a real close approach, ~10–100×) re-open the fit; ordinary flutter is absorbed.
    float info_peak_ratchet     = 2.5f;   // min ratio over the bar to ratchet+rescale (1 = legacy: every spike)
    float info_peak_ema         = 0.30f;  // EMA weight folding a new high obs into the bar (0 = frozen bar, 1 = raw max)
    float process_std_len       = 0.005f; // info-filter predict process-noise std per fresh frame, length DOFs (m)
    float process_std_ang       = 0.01f;  // …for the periodic (yaw) DOF (rad)
    float process_std_pos       = -1.0f;  // …for the POSITION DOFs (is_position); <0 ⇒ use process_std_len.
                                          // A static object's centre barely moves frame-to-frame, so a small
                                          // value lets position precision accumulate → its gain → 0 (locks).

    // Confidence (YOLO score) weighting: w = clamp01((conf−floor)/(ref−floor))^power, applied to obs info.
    bool  mask_conf_weight = true;
    float mask_conf_floor  = 0.2f;
    float mask_conf_ref    = 0.5f;
    float mask_conf_power  = 2.0f;

    // CUSUM / SPRT change-detector gate.
    bool  ce_gate            = false;
    float ce_band_len        = 0.02f;     // length-DOF surprise deadband (m): innovations within this are "coherent"
    float ce_band_ang        = 0.05f;     // yaw surprise deadband (rad)
    float ce_base_pos        = 6.0f;      // position (cx,cy) barrier floor (excess-bands)
    float ce_lambda_pos      = 1.0f;      // …per accumulated equivalent-view
    float ce_base_size       = 3.0f;      // size/shape barrier floor
    float ce_lambda_size     = 0.5f;      // …per accumulated equivalent-view
    // Yaw gets its OWN barrier (default = size, so a layout with no yaw_index is unaffected). A rigid
    // object's orientation, once seen, is its most stable property and the cheapest to confuse from a
    // partial/ambiguous view — set these HIGH to make yaw the hardest DOF to unlock.
    float ce_base_yaw        = 3.0f;      // yaw barrier floor
    float ce_lambda_yaw      = 0.5f;      // …per accumulated equivalent-view
    float ce_decay           = 0.8f;      // per-coherent-frame pay-down of the accumulator
    float ce_step_cap        = 1.0f;      // cap the per-frame CUSUM increment → unlock needs a RUN, not one outlier
    float ce_unlock_deflate  = 0.3f;      // on unlock, multiply that DOF's accumulated info by this (concede old estimate)
};

// ── DOF descriptor: which index is periodic (yaw, −1 = none) and which are the "position" group ──
template <int N>
struct StabilizerLayout
{
    int yaw_index = -1;
    std::array<bool, N> is_position{};
};

// ── Per-instance state (the loose arrays that used to live on each *Instance, grouped) ──────────
template <int N>
struct StabilizerState
{
    std::array<float, N> fisher_info{};          // normalised "equivalent views" accumulator (drives stiffness)
    std::array<float, N> fisher_info_raw{};      // un-normalised accumulated precision Σ⁻¹ (posterior std / RT cov)
    std::array<float, N> fisher_info_peak{};     // per-DOF adaptive normaliser (best single-view info seen)
    std::array<float, N> last_obs_info{};        // this frame's observation Fisher diagonal, AFTER confidence weighting
    std::array<float, N> last_kalman_gain{};     // per-DOF acceptance lerp gain (the calibrated stiffness)
    std::array<float, N> counter_evidence{};     // signed CUSUM of one-sided surprise vs the committed belief
    std::array<float, N> last_ce_gate{};         // gate decision: −1 reject/locked, 0 passthrough, +1 unlock
    float last_mask_conf_weight = 1.0f;          // w(YOLO score) applied this frame (1 = full trust)
    // Model evidence ρ ∈ (0,1] for THIS frame (mean inlier responsibility): how well the current belief
    // explains the fresh observation. The predict step discounts the carried precision by ρ, so a
    // persistently UN-explained observation (the model is wrong) loosens the belief and lets it yield to
    // the data — the falsifiability that makes a CUSUM/freeze/re-open unnecessary. 1 = perfectly explained
    // (precision fully persists). Set by the agent each fresh frame before compute_acceptance/accumulate.
    float model_evidence = 1.0f;
};

template <int N>
class BeliefStabilizer
{
public:
    using Vec = std::array<float, N>;

    BeliefStabilizer() = default;
    BeliefStabilizer(StabilizerLayout<N> layout, StabilizerParams params)
        : layout_(layout), params_(params) {}

    void set_layout(const StabilizerLayout<N>& l) { layout_ = l; }
    void set_params(const StabilizerParams& p)    { params_ = p; }
    const StabilizerParams& params() const        { return params_; }

    // The acceptance lerp should be driven by the Kalman gain only when both switches are on.
    bool kalman_active() const { return params_.fisher_filter_enabled and params_.kalman_stiffness; }

    // (1) Weight this frame's observation Fisher diagonal by the mask confidence and store it on the
    // state. Returns the applied weight w∈[0,1]. Call once per fresh frame, BEFORE compute_acceptance.
    float weight_observation(StabilizerState<N>& s, const Vec& obs_info, float mask_confidence) const
    {
        float w = 1.0f;
        if (params_.mask_conf_weight)
        {
            const float c0 = std::clamp(params_.mask_conf_floor, 0.0f, 0.99f);
            const float cr = std::max(c0 + 1e-3f, params_.mask_conf_ref);
            w = std::pow(std::clamp((mask_confidence - c0) / (cr - c0), 0.0f, 1.0f),
                         std::max(0.1f, params_.mask_conf_power));
        }
        s.last_mask_conf_weight = w;
        for (int j = 0; j < N; ++j)
            s.last_obs_info[j] = obs_info[j] * w;
        return w;
    }

    // (2) Per-DOF acceptance gains (maturity-stiffened Kalman) + the CUSUM/SPRT gate, given the raw
    // fit and the previous accepted state (for the innovation). Reads the PRIOR precision; deflates
    // the accumulators on an unlock. Afterwards last_kalman_gain[j] is the lerp weight to use.
    void compute_acceptance(StabilizerState<N>& s, const Vec& raw, const Vec& prev) const
    {
        const float vh = std::max(1e-3f, params_.views_half);
        const float rho = std::clamp(s.model_evidence, 1e-3f, 1.0f);
        for (int j = 0; j < N; ++j)
        {
            // Evidence-scaled predict: a poorly-explained belief (low ρ) is discounted so its Kalman gain
            // rises and it yields to the data — the model is falsifiable, no gate needed.
            const float Yprev = s.fisher_info_raw[j] * rho;
            const float Ypred = 1.0f / (1.0f / std::max(Yprev, 1e-9f) + process_q(j));
            const float oi    = s.last_obs_info[j];
            float k = (oi > 0.0f) ? oi / (Ypred + oi) : 0.0f;
            if (params_.maturity_stiffness)
                k *= vh / (vh + s.fisher_info[j]);   // mature DOF barely moves on one frame
            s.last_kalman_gain[j] = k;
        }

        if (not params_.ce_gate)
            return;

        for (int j = 0; j < N; ++j)
        {
            const bool  is_ang = (j == layout_.yaw_index);
            const float band   = is_ang ? std::max(1e-4f, params_.ce_band_ang)
                                        : std::max(1e-4f, params_.ce_band_len);
            float d = raw[j] - prev[j];
            if (is_ang) d = wrap_angle(d);            // periodic innovation
            const float e = std::abs(d) / band;       // surprise, in deadbands
            if (e <= 1.0f)
            {
                s.counter_evidence[j] *= params_.ce_decay;   // coherent → pay down suspicion
                s.last_ce_gate[j] = 0.0f;                    // passthrough (keep Kalman gain)
                continue;
            }
            const float dir = (d >= 0.0f) ? 1.0f : -1.0f;
            if (s.counter_evidence[j] * dir < 0.0f)
                s.counter_evidence[j] = 0.0f;                // direction flip → restart the run
            // WEIGHTED by mask confidence and capped per frame: a low-score / lone glitch can barely
            // move S → it gets rejected (locks the model); only a confident, coherent RUN can unlock.
            s.counter_evidence[j] += dir * s.last_mask_conf_weight *
                                     std::min(e - 1.0f, params_.ce_step_cap);
            if (std::abs(s.counter_evidence[j]) > barrier_for(s, j))
            {
                // Unlock: allow re-adaptation GRADUALLY. Keep the calibrated (small) Kalman gain and
                // deflate the accumulated precision so the gain rises over the next frames and the
                // belief eases onto the new evidence — never a one-frame snap.
                s.counter_evidence[j]  = 0.0f;
                s.fisher_info[j]      *= params_.ce_unlock_deflate;
                s.fisher_info_raw[j]  *= params_.ce_unlock_deflate;
                s.last_ce_gate[j]      = 1.0f;
            }
            else
            {
                s.last_kalman_gain[j] = 0.0f;                // unconfirmed surprise → reject (lock)
                s.last_ce_gate[j]     = -1.0f;
            }
        }
    }

    // (4) Fold the (weighted) observation into the accumulators. Call on FRESH frames only, AFTER
    // compute_acceptance (which reads the prior precision). Predict (Q-bleed) + update; the normalised
    // accumulator counts "equivalent views" (each DOF's curvature normalised by its best single view).
    void accumulate(StabilizerState<N>& s) const
    {
        const float decay      = std::clamp(params_.info_decay, 0.0f, 1.0f);
        const float peak_decay = std::clamp(params_.info_peak_decay, 0.5f, 1.0f);
        const float rho        = std::clamp(s.model_evidence, 1e-3f, 1.0f);   // evidence-scaled forgetting
        for (int j = 0; j < N; ++j)
        {
            const float oi = s.last_obs_info[j];

            // Quality bar = a smoothed, ratcheting reference for single-view info. When THIS view beats
            // the bar by ≥ ratchet×, it's a GENUINELY better look (a real close approach): raise the bar
            // (EMA of the high obs, not the raw spike) and rescale the accumulated "views" — counted at
            // the OLD, lower bar — down, so the belief re-opens to the better evidence. Ordinary per-frame
            // obs flutter (orders of magnitude) is BELOW the ratchet → the bar holds and the DOF matures.
            // Off (legacy) → plain decaying all-time max, every spike re-ratchets.
            float bar = s.fisher_info_peak[j] * peak_decay;   // slowly forget the old bar
            if (params_.info_quality_rescale)
            {
                const float ratchet = std::max(1.0f, params_.info_peak_ratchet);
                if (oi > bar * ratchet)
                {
                    // EMA the new high in (don't jump to the raw spike), then rescale stale views to it.
                    const float bar_new = bar > 1e-6f
                        ? bar + std::clamp(params_.info_peak_ema, 0.0f, 1.0f) * (oi - bar) : oi;
                    if (bar > 1e-6f) s.fisher_info[j] *= bar / bar_new;
                    bar = bar_new;
                }
            }
            else
                bar = std::max(oi, bar);   // legacy all-time(decaying) max
            s.fisher_info_peak[j] = bar;

            const float incr = bar > 1e-6f ? std::clamp(oi / bar, 0.0f, 1.0f) : 0.0f;
            s.fisher_info[j] = s.fisher_info[j] * decay * rho + incr;   // bad fit (low ρ) → forget faster

            const float Yprev = s.fisher_info_raw[j] * rho;             // evidence-scaled carried precision
            const float Ypred = 1.0f / (1.0f / std::max(Yprev, 1e-9f) + process_q(j));
            s.fisher_info_raw[j] = Ypred + oi;
        }
    }

    // Swap all per-DOF accumulators of two DOFs. Used when the owning model canonicalises its state by
    // swapping two interchangeable dimensions (e.g. a box's w↔h with a 90° yaw turn): the precision /
    // CUSUM / view history must follow the dimension it belongs to, or the swapped DOF inherits the
    // wrong confidence. Rare (only on an actual canonical flip), so the cost is irrelevant.
    static void swap_dofs(StabilizerState<N>& s, int a, int b)
    {
        if (a == b or a < 0 or b < 0 or a >= N or b >= N) return;
        std::swap(s.fisher_info[a],      s.fisher_info[b]);
        std::swap(s.fisher_info_raw[a],  s.fisher_info_raw[b]);
        std::swap(s.fisher_info_peak[a], s.fisher_info_peak[b]);
        std::swap(s.last_obs_info[a],    s.last_obs_info[b]);
        std::swap(s.last_kalman_gain[a], s.last_kalman_gain[b]);
        std::swap(s.counter_evidence[a], s.counter_evidence[b]);
        std::swap(s.last_ce_gate[a],     s.last_ce_gate[b]);
    }

    // Data-only posterior std ×1000 (mm for length DOFs, mrad for the yaw DOF); −1 until first observed.
    static float posterior_std_milli(const StabilizerState<N>& s, int j)
    {
        const float prec = s.fisher_info_raw[j];
        return prec > 1e-6f ? 1000.0f / std::sqrt(prec) : -1.0f;
    }

private:
    float process_q(int j) const
    {
        float std;
        if (j == layout_.yaw_index)
            std = params_.process_std_ang;
        else if (j >= 0 and j < N and layout_.is_position[j] and params_.process_std_pos >= 0.0f)
            std = params_.process_std_pos;   // static-centre: small Q ⇒ precision accumulates ⇒ position locks
        else
            std = params_.process_std_len;
        return std * std;
    }

    float barrier_for(const StabilizerState<N>& s, int j) const
    {
        float base, lambda;
        if (j == layout_.yaw_index)                         { base = params_.ce_base_yaw;  lambda = params_.ce_lambda_yaw; }
        else if (j >= 0 and j < N and layout_.is_position[j]) { base = params_.ce_base_pos;  lambda = params_.ce_lambda_pos; }
        else                                                 { base = params_.ce_base_size; lambda = params_.ce_lambda_size; }
        return base + lambda * s.fisher_info[j];
    }

    static float wrap_angle(float a)
    {
        constexpr float kPi = 3.14159265358979323846f;
        return std::remainder(a, 2.0f * kPi);
    }

    StabilizerLayout<N> layout_{};
    StabilizerParams    params_{};
};

}  // namespace rc
