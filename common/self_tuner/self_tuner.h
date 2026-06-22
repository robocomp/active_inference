// RoboComp active_inference - self_tuner (online behavioural-parameter adaptation)
//
// A generic, dependency-light engine that lets an agent adapt its OWN behavioural
// constants online. The agent exposes a curated set of knobs as a [Tune] block of
// {name, min, max, initial}; each episode (or window) it asks the tuner for a
// candidate value to USE, runs, and reports the measured objective. The tuner does
// gradient-free descent on that objective over the parameter box.
//
// FRAMING. The objective is the agent's free energy / model evidence surrogate
// (Goal::Minimize by default — lower F is better). The controller's task cost
// (−success + λ·time) and a perception agent's prediction residual / NEES are both
// instances. The exploration policy is the acquisition: it spends evaluations to
// resolve which θ is best (epistemic) while improving the objective (pragmatic).
//
// DESIGN.
//   - DEP-FREE: only the C++ stdlib. No DSR/Qt/Eigen/ConfigLoader, so ANY agent
//     (or a standalone test) can link it. Coordination with the shared graph (the
//     "who may explore now" token) is layered ON TOP via set_exploration_allowed().
//   - TIMESCALE-AGNOSTIC: the agent drives a propose()/report() cycle at whatever
//     boundary it has (a pick-place rep; a perception window). The tuner never
//     assumes a clock.
//   - SAFE BY CONSTRUCTION: candidates are always clipped to [min,max] — the range
//     IS the safety envelope. A non-finite objective (a crashed/aborted episode) is
//     treated as worst and cannot poison the incumbent.
//   - NOISE-AWARE: repeats>1 averages several episodes per candidate before deciding,
//     for the noisy metrics these agents have (stochastic grasps, scene variance).
//
// USAGE (agent side):
//   rc::tune::SelfTuner tuner(params, cfg);
//   // each rep/window:
//   tuner.propose();                                  // roll a candidate
//   double d = tuner.value("palm_grasp_dist");        // USE this instead of the const
//   ... run the episode, measure the objective J ...
//   tuner.report(J);                                  // update incumbent + trust region

#pragma once

#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace rc::tune
{

// One tunable knob and its safe range. `initial` seeds the incumbent (the
// hand-tuned value); exploration starts in a trust region around it.
struct Parameter
{
    std::string name;
    double      min     = 0.0;
    double      max     = 1.0;
    double      initial = 0.0;
};

// Objective convention. Default Minimize: the reported scalar is a COST / free
// energy (lower is better). An agent maximizing a reward sets Goal::Maximize (or
// reports −reward and keeps Minimize).
enum class Goal { Minimize, Maximize };

struct TunerConfig
{
    Goal   goal          = Goal::Minimize;
    // Trust region = a Gaussian step whose σ for each knob is `radius` × that knob's
    // (max−min). Grows on improvement, shrinks on a miss — so steps stay "close" and
    // contract as the incumbent settles (precision over θ rising = the skill curve).
    double init_radius   = 0.15;
    double radius_grow   = 1.3;
    double radius_shrink = 0.7;
    double min_radius    = 0.01;
    double max_radius    = 0.50;
    int    repeats       = 1;    // episodes averaged per candidate before deciding (noise control)
    std::uint64_t seed   = 0;    // RNG seed; 0 ⇒ seed from random_device
    std::string state_path;      // persistence file ("" ⇒ in-memory only)
    bool   explore       = true; // master switch; false ⇒ always serve the incumbent (exploit only)
};

// Default exploration policy: trust-region Gaussian random search ((1+1)-ES style).
// Robust, gradient-free, small-step, noise-tolerant with repeats. The policy is
// isolated behind propose_candidate()/on_result() so SPSA / Bayesian-opt / CMA-ES
// can replace it later without touching the agent-facing API.
class SelfTuner
{
public:
    SelfTuner(std::vector<Parameter> params, TunerConfig cfg = {});

    // ── Agent-facing propose/report cycle ────────────────────────────────────
    // Begin one evaluation: roll the candidate the agent will run with. While a
    // candidate is mid-repeats it is kept; exploration disabled ⇒ serves incumbent.
    void propose();

    // The value to USE this episode for `name` — the candidate during exploration,
    // the incumbent otherwise. Unknown name ⇒ `fallback` (logged once).
    [[nodiscard]] double value(const std::string& name, double fallback = 0.0) const;
    [[nodiscard]] double value(std::size_t idx) const;

    // Report the measured objective for the in-play candidate. Averages over
    // `repeats` reports, then accepts/rejects vs the incumbent and adapts the
    // radius. Non-finite ⇒ counted as worst (reject). Persists if state_path set.
    void report(double objective);

    // ── Coordination (graph exploration token) ───────────────────────────────
    // When false, propose() serves the incumbent unchanged: the agent keeps running
    // but stops perturbing — so an upstream agent can explore without this one's
    // outputs (its inputs) drifting. The "who may explore now" token lives in the
    // graph; this is the local gate it drives.
    void set_exploration_allowed(bool allowed);
    [[nodiscard]] bool exploration_allowed() const { return allowed_; }

    // ── Introspection / logging ──────────────────────────────────────────────
    [[nodiscard]] const std::vector<double>& incumbent() const { return best_x_; }
    [[nodiscard]] std::vector<std::pair<std::string, double>> incumbent_named() const;
    [[nodiscard]] double      incumbent_objective() const { return best_obj_; }
    [[nodiscard]] std::size_t evaluations() const { return evals_; }
    [[nodiscard]] double      radius() const { return radius_; }
    [[nodiscard]] std::size_t size() const { return params_.size(); }

    // ── Persistence (also auto-saved on each accepted report) ─────────────────
    bool save() const;
    bool load();   // matches by name; unknown/missing knobs keep their initial

private:
    [[nodiscard]] bool   better(double a, double b) const;   // a strictly better than b under goal
    [[nodiscard]] double clip(std::size_t i, double v) const;
    void roll_candidate();                                    // fill cand_x_ within the trust region
    [[nodiscard]] std::optional<std::size_t> index_of(const std::string& name) const;

    std::vector<Parameter> params_;
    TunerConfig            cfg_;

    std::vector<double> best_x_;   // incumbent θ*
    std::vector<double> cand_x_;   // candidate currently being evaluated
    double              best_obj_; // incumbent objective (+inf/−inf seed per goal)
    double              radius_;   // current trust-region fraction

    // Repeat accumulation for the in-play candidate.
    double obj_sum_       = 0.0;
    int    obj_count_     = 0;
    bool   candidate_live_ = false;

    std::size_t evals_   = 0;      // total episodes reported
    bool        allowed_ = true;   // exploration gate (coordination token)
    mutable std::mt19937_64 rng_;
};

// Convenience: parse a Parameter from a name + a "min max [initial]" spec string
// (e.g. an agent's TOML `palm_grasp_dist = "0.06 0.10 0.08"`). initial defaults to
// the midpoint. Returns nullopt on a malformed spec.
[[nodiscard]] std::optional<Parameter> parse_parameter(std::string_view name, std::string_view spec);

}  // namespace rc::tune
