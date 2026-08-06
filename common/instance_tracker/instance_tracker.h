/*
 * instance_tracker.h
 *
 * Shared multi-object BIRTH / ASSOCIATE / DEATH layer for the CORTEX concept agents
 * (table_concept, bottle_concept, …). It turns the per-cycle set of detections (mask slices of the
 * agent's label) + the current set of tracks (the agent's live instances) into:
 *   - an ASSOCIATION  (which detection feeds which instance) — gated, global 1-to-1, not greedy,
 *   - a list of BIRTHS (detections unexplained by any instance for a sustained run → spawn a new one),
 *   - a list of DEATHS (instances unsupported for too long → retire).
 *
 * Object-AGNOSTIC: it reasons only about 2-D centres + (optional) XY covariances. The agent fills a
 * TrackView per instance (centre from the fit, cov from the stabiliser posterior / P) and a
 * DetectionView per mask slice, calls update() once per cycle, and acts on the result (delete dead
 * nodes, scaffold born ones from the detection, set inst.assigned_mask so observe() uses the RIGHT
 * mask instead of a greedy nearest-neighbour). Header-only; depends only on Eigen.
 *
 * Association = covariance-gated cost (squared Mahalanobis when the track has a cov, else squared XY
 * distance vs a metric gate), solved by greedy-lowest-cost over gated pairs — exact 1-to-1 for the
 * small instance counts these agents see, without a full Hungarian. The create-vs-assign decision is
 * the model-selection the agent already uses for the support surface, lifted to instances: a detection
 * that NO instance explains (outside every gate), held birth_frames consecutive frames and far enough
 * from every track, is better explained by a NEW object than by stretching an existing fit.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

namespace rc {

// Tunables for the birth/associate/death policy (gates, costs, birth/death debounce). Physical, not tuned.
struct TrackerParams
{
    float gate_mahalanobis = 9.0f;    // χ²₂ gate (~3σ) for a det↔track match when the track has a cov
    float gate_fallback_m  = 0.40f;   // metric XY gate (m) when the track has no usable cov yet
    // Detection-noise std R (m) added to the track cov P to form the INNOVATION covariance S = P + R²I
    // used by the Mahalanobis gate. Without it the gate uses P alone, so an overconfident fit (σ ~mm)
    // rejects every real detection (the raw mask centroid wanders cm-scale vs the de-projected fit
    // centre) → tracks starve and die. R is the measurement spread: centroid jitter + centroid-vs-axis
    // offset. Must be ≥ that offset; default covers a ~6 cm front-arc/de-projection gap.
    float detection_noise_m = 0.05f;
    int   birth_frames     = 5;       // consecutive frames a detection must stay unassigned to spawn a track
    int   death_frames     = 60;      // consecutive frames a track may go unsupported before retirement
    float birth_min_sep_m  = 0.25f;   // a birth candidate must be ≥ this from every existing track AND from
                                      // every other pending candidate (anti-duplicate)
    float birth_match_m    = 0.20f;   // a pending candidate persists across frames if a fresh unassigned det
                                      // lands within this of it
    // Multi-detection fusion: when true, ONE track may absorb SEVERAL detections in a cycle (each detection
    // still goes to at most one track). Use it when the same object is seen by independent sensors in the same
    // frame (e.g. a table in both the ZED and the ricoh-360 masks) and the agent wants to fuse ALL of them —
    // it runs one belief update per assigned slice, so each sensor keeps its own R and common-mode error.
    // false (default) = classic 1-to-1 (a track takes only its single best detection). bottle/chair unchanged.
    bool  multi_det_per_track = false;
    // Assignment COST for cov-bearing pairs (the GATE is always squared Mahalanobis ≤ gate_mahalanobis).
    // false → cost = m² (raw squared Mahalanobis). true → cost = ½(m² + ln|S|), the Gaussian negative
    // log-likelihood (additive constant dropped, argmin-invariant). The ln|S| term makes tracks with
    // DIFFERENT cov sizes compete by likelihood, not raw distance — a wide freshly-born track no longer
    // out-bids a tight mature one merely by being nearer. Note: NLL costs (cov pairs) are on a different
    // scale than the metric-fallback r² (no-cov pairs); when both kinds coexist in one frame the cov
    // pairs sort first — usually desirable (an established track wins), but be aware when enabling.
    bool  nll_cost = false;
    // Optional VERTICAL separation gate (m). 0 (default) = disabled → purely 2-D association (bottle/
    // chair/table unchanged). When > 0, a det↔track pair is infeasible if |det.z − track.z| exceeds it,
    // and a detection stacked vertically above/below a track (beyond this) does NOT count as "on top of"
    // it for the birth-separation test. cabinet_concept uses it so a WALL-unit mask (z≈1.7) and a BASE
    // run (z≈0.35) sharing an XY footprint become DIFFERENT tracks instead of one flip-flopping instance.
    float z_gate_m = 0.0f;
};

// One live instance the tracker reasons about (filled by the agent from its instances each cycle).
struct TrackView
{
    std::uint64_t   id  = 0;
    Eigen::Vector2f xy  = Eigen::Vector2f::Zero();   // current/predicted centre (room frame)
    Eigen::Matrix2f cov = Eigen::Matrix2f::Identity();
    bool            has_cov = false;                 // false → use the metric fallback gate
    float           z   = 0.0f;                      // vertical coord for the optional z_gate (see TrackerParams)
    // Negative-information gate for DEATH: only accrue an unsupported "miss" when the object SHOULD be
    // visible (inside the camera frustum, not behind it) yet wasn't detected — that is real evidence it
    // was removed. Out-of-FoV absence is NOT evidence: with expected_visible=false the miss timer is
    // HELD, so the instance persists. Default true = legacy (every unsupported frame counts).
    bool            expected_visible = true;
};

// One incoming detection (the agent's mask slice). slice_index maps the assignment back to the packet.
struct DetectionView
{
    Eigen::Vector2f xy = Eigen::Vector2f::Zero();
    int             slice_index = -1;
    float           z  = 0.0f;                       // vertical coord for the optional z_gate (see TrackerParams)
    // A detection may ASSOCIATE to (refine) an existing track regardless, but only a birthable one may
    // SPAWN a new instance. The agent sets this false for evidence it judges too weak to create an object on
    // its own this frame — e.g. a low-confidence or high-range-variance 360-RGB/LiDAR peripheral mask. Such a
    // detection still refines a track it gates to; it just can't birth phantoms. Default true.
    bool            birthable = true;
    // Per-frame BIRTH EVIDENCE this detection contributes to its pending candidate's maturation. Default 1.0 ⇒
    // the classic "birth_frames consecutive frames" behaviour (streak += 1 each frame). An agent that has an
    // INDEPENDENT corroborating cue this frame (e.g. table_concept: residual-grid surprise mass under the
    // detection — the geometry already agrees a real object is here) raises it >1 so a corroborated detection
    // crosses the birth boundary in fewer frames, while an un-corroborated/phantom detection (evidence 1.0)
    // still serves the full debounce. Birth becomes evidence-accumulation-to-a-boundary, not a fixed counter.
    float           birth_evidence = 1.0f;
};

// One cycle's decision: det→track assignment + which detections birth + which tracks die.
struct TrackerResult
{
    std::vector<int>           assignment;   // per detection: index into `tracks` (-1 = unassigned)
    std::vector<int>           births;       // detection indices that should spawn a new instance
    std::vector<std::uint64_t> deaths;       // track ids to retire

    // ── Candidate identity: lets an agent ACCUMULATE the probation burst (common/birth_fragment) ──
    // A pending candidate matures over birth_frames observations, but the tracker only ever kept a
    // streak counter and the LATEST xy — every mask cloud seen while maturing was discarded, so an
    // instance was born from ONE frame's centroid despite several frames of evidence having been
    // examined. Exposing a stable id per candidate lets the agent bank that evidence next to the
    // candidate and hand the whole burst to the fitter at promotion (deferred commitment: choose the
    // representation, and whether to commit at all, with all the data in hand). Agents that don't
    // care simply ignore these three fields.
    std::vector<std::uint64_t> cand_of_det;        // per detection: candidate id it fed (0 = none)
    std::vector<std::uint64_t> expired_candidates; // candidate ids dropped this cycle (free their burst)
    std::vector<std::uint64_t> birth_cand_ids;     // parallel to `births`: the promoted candidate's id
};

// Stateful multi-object tracker: carries the miss-counters and pending birth candidates across cycles.
class InstanceTracker
{
public:
    void set_params(const TrackerParams& p) { params_ = p; }
    const TrackerParams& params() const { return params_; }

    // Run one cycle: gate → greedy 1-to-1 associate → negative-information death → persisted-candidate birth.
    TrackerResult update(const std::vector<TrackView>& tracks,
                         const std::vector<DetectionView>& dets)
    {
        TrackerResult out;
        out.assignment.assign(dets.size(), -1);
        out.cand_of_det.assign(dets.size(), 0);

        // ── 1. Gated cost for every (det, track) pair ────────────────────────────────────────────
        struct Pair { float cost; int det; int trk; };
        std::vector<Pair> pairs;
        for (int d = 0; d < static_cast<int>(dets.size()); ++d)
            for (int t = 0; t < static_cast<int>(tracks.size()); ++t)
            {
                // Vertical separation gate: an upper-tier det never associates to a base-tier track.
                if (params_.z_gate_m > 0.0f and std::abs(dets[d].z - tracks[t].z) > params_.z_gate_m)
                    continue;
                const Eigen::Vector2f e = dets[d].xy - tracks[t].xy;
                float cost; bool ok;
                if (tracks[t].has_cov)
                {
                    // Innovation covariance S = P + R²I (fit cov + detection noise). Gating on P alone
                    // makes an overconfident fit reject every real detection — see detection_noise_m.
                    const float r2n = params_.detection_noise_m * params_.detection_noise_m;
                    const Eigen::Matrix2f S = tracks[t].cov + r2n * Eigen::Matrix2f::Identity();
                    const float m2 = e.dot(S.inverse() * e);   // squared Mahalanobis vs innovation cov
                    ok = std::isfinite(m2) and m2 <= params_.gate_mahalanobis;   // gate is ALWAYS m²
                    if (params_.nll_cost)
                    {
                        // Gaussian NLL cost ½(m² + ln|S|); +½d·ln(2π) constant dropped (argmin-neutral).
                        const float detS = S.determinant();
                        cost = 0.5f * (m2 + std::log(std::max(detS, 1e-12f)));
                    }
                    else
                        cost = m2;
                }
                else
                {
                    const float r2 = e.squaredNorm();
                    ok = r2 <= params_.gate_fallback_m * params_.gate_fallback_m;
                    cost = r2;
                }
                if (ok) pairs.push_back({cost, d, t});
            }

        // ── 2. Greedy global 1-to-1: lowest cost first, each det+track used once ──────────────────
        std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b){ return a.cost < b.cost; });
        std::vector<bool> det_used(dets.size(), false), trk_used(tracks.size(), false);
        for (const auto& p : pairs)
        {
            // Each detection is used at most once. A track is used at most once too UNLESS multi-detection
            // fusion is on, in which case a track may absorb several detections (trk_used then only records
            // "this track was supported" for the death timer below, and never blocks a further match).
            if (det_used[p.det]) continue;
            if (trk_used[p.trk] and not params_.multi_det_per_track) continue;
            out.assignment[p.det] = p.trk;
            det_used[p.det] = true;
            trk_used[p.trk] = true;
        }

        // ── 3. DEATH: a track unsupported WHILE it should be visible accrues misses; retire past
        //        death_frames. If it is NOT expected_visible (out of the camera frustum), the miss
        //        timer is HELD — out-of-FoV absence is not evidence of removal, so it persists.
        std::unordered_map<std::uint64_t, int> next_miss;
        for (int t = 0; t < static_cast<int>(tracks.size()); ++t)
        {
            const std::uint64_t id = tracks[t].id;
            if (trk_used[t]) { next_miss[id] = 0; continue; }
            if (not tracks[t].expected_visible) { next_miss[id] = miss_count_[id]; continue; }   // hold
            const int miss = miss_count_[id] + 1;
            if (miss >= params_.death_frames) out.deaths.push_back(id);
            else                              next_miss[id] = miss;
        }
        miss_count_.swap(next_miss);   // drops counters for tracks that vanished (already deleted)

        // ── 4. BIRTH: unassigned detections, persisted across frames, far from tracks + each other ──
        std::vector<Candidate> next_cand;
        const auto far_from_tracks = [&](const Eigen::Vector2f& xy, float z)
        {
            for (const auto& t : tracks)
            {
                // A track stacked far above/below (a different tier) does not block a birth here.
                if (params_.z_gate_m > 0.0f and std::abs(t.z - z) > params_.z_gate_m) continue;
                if ((t.xy - xy).norm() < params_.birth_min_sep_m) return false;
            }
            return true;
        };
        for (int d = 0; d < static_cast<int>(dets.size()); ++d)
        {
            if (out.assignment[d] != -1) continue;            // explained by an existing instance
            if (not dets[d].birthable) continue;              // low-trust evidence may refine, never spawn
            const Eigen::Vector2f& xy = dets[d].xy;
            if (not far_from_tracks(xy, dets[d].z)) continue; // sits on top of a same-tier track → not new

            // Match to a pending candidate (carried from prior frames) within birth_match_m.
            int best = -1; float best_r = params_.birth_match_m;
            for (int c = 0; c < static_cast<int>(candidates_.size()); ++c)
            {
                if (candidates_[c].claimed) continue;
                const float r = (candidates_[c].xy - xy).norm();
                if (r < best_r) { best_r = r; best = c; }
            }
            Candidate cand;
            if (best >= 0) { cand = candidates_[best]; candidates_[best].claimed = true; }
            else           { cand.id = ++next_cand_id_; }      // fresh blob → new identity for its burst
            cand.xy     = xy;                                 // track the blob's latest position
            cand.streak = (best >= 0 ? cand.streak : 0.0f) + dets[d].birth_evidence;   // accumulate birth evidence

            // Don't let two fresh dets seed the same birth this frame.
            bool dup = false;
            for (const auto& nc : next_cand)
                if ((nc.xy - xy).norm() < params_.birth_min_sep_m) { dup = true; break; }
            if (dup) continue;

            // This detection's evidence belongs to `cand` — the agent banks its mask cloud under this id.
            out.cand_of_det[d] = cand.id;

            if (cand.streak >= params_.birth_frames)
            {
                out.births.push_back(d);                      // promote: spawn an instance from this det
                out.birth_cand_ids.push_back(cand.id);        // …and hand over its accumulated burst
            }
            else
                next_cand.push_back(cand);                    // keep maturing
        }

        // EXPIRY: every id carried into this cycle that is neither still maturing nor promoted is gone
        // (blob vanished, or lost the anti-duplicate test) — tell the agent so it can free the burst.
        // Computed against the survivor set rather than the `claimed` flag so the `dup`-drop path, which
        // claims a candidate and then abandons it, cannot leak a bank entry.
        for (const auto& c : candidates_)
        {
            if (c.id == 0) continue;                          // pre-existing candidate from an older build
            bool survives = false;
            for (const auto& nc : next_cand)      if (nc.id == c.id) { survives = true; break; }
            if (not survives)
                for (const auto id : out.birth_cand_ids) if (id == c.id) { survives = true; break; }
            if (not survives) out.expired_candidates.push_back(c.id);
        }
        candidates_.swap(next_cand);                          // unmatched candidates expire (blob gone)

        return out;
    }

    void reset() { miss_count_.clear(); candidates_.clear(); }

private:
    // `id` is a tracker-local monotonic identity for a pending birth, stable across the frames it
    // matures over. It is the key an agent banks the candidate's mask clouds under (see TrackerResult).
    struct Candidate { std::uint64_t id = 0; Eigen::Vector2f xy = Eigen::Vector2f::Zero();
                       float streak = 0.0f; bool claimed = false; };

    TrackerParams                          params_;
    std::unordered_map<std::uint64_t, int> miss_count_;   // track id → consecutive unsupported frames
    std::vector<Candidate>                 candidates_;   // pending births (not yet promoted)
    std::uint64_t                          next_cand_id_ = 0;   // never reused; 0 means "no candidate"
};

}  // namespace rc
