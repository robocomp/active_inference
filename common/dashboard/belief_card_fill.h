/*
 * common/dashboard/belief_card_fill.h — the three parts of a BeliefCard that are the same everywhere. SHARED.
 *
 * refresh_belief_inspector() is genuinely per-agent where it names the object's DOFs and its discrete modes
 * (a fridge's door-facing quartet, a table's w<->h swap + round/square, a cabinet's tiers). But three blocks
 * inside it never varied across the four agents, and one of them carries a trap worth stating once:
 *
 *   fill_dofs    — one row per Sigma index: value, sqrt of the diagonal, and the consumer's sigma* demand.
 *   fill_cov     — the ROW-MAJOR copy. ★Eigen stores COLUMN-major. Sigma is symmetric today, so an implicit
 *                  .data() copy happens to work — and would silently transpose the heatmap the day some
 *                  belief reports a non-symmetric matrix. Written out explicitly for that reason, once.
 *   fill_scalars — the gauge set the card footer shows. Four copies listed the same nine fields; adding a
 *                  gauge meant remembering four files, which is how a set like this drifts.
 *
 * Templated on the belief/instance rather than typed, because every agent's Instance already exposes these
 * field names — the duck typing IS the shared contract, and a rename breaks the build rather than a panel.
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <span>
#include <vector>

#include "belief_inspector.h"          // rc::BeliefCard
#include "../ai_belief/dof_spec.h"     // rc::DofSpec

namespace rc::dashboard
{

// One row per Sigma index, in Sigma order. `values` and `dofs` must both be N long.
template <class Matrix>
inline void fill_dofs(BeliefCard& c, std::span<const rc::DofSpec> dofs,
                      std::span<const float> values, const Matrix& S)
{
    const std::size_t n = std::min(dofs.size(), values.size());
    c.dofs.reserve(n);
    for (std::size_t j = 0; j < n; ++j)
        c.dofs.push_back({dofs[j].name, dofs[j].unit, values[j],
                          std::sqrt(std::max(0.0f, static_cast<float>(S(j, j)))), dofs[j].sigma_star});
}

// ★ROW-MAJOR, filled explicitly. Eigen is column-major; Sigma is symmetric today, so a .data() copy would
// work by luck and transpose the heatmap the day it stops being.
template <class Matrix>
inline void fill_cov(BeliefCard& c, const Matrix& S, int N)
{
    c.cov.resize(static_cast<std::size_t>(N) * static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            c.cov[static_cast<std::size_t>(i) * N + j] = static_cast<float>(S(i, j));
}

// The card footer's gauges. `now` is passed in so every card in one refresh shares a single clock read —
// otherwise the ages in one pass disagree with each other by the time the loop takes.
template <class Instance>
inline void fill_scalars(BeliefCard& c, const Instance& inst,
                         std::chrono::steady_clock::time_point now)
{
    c.s.fe            = inst.dbg_energy;
    c.s.fe_baseline   = inst.fe_baseline;
    c.s.fe_surprise   = inst.fe_surprise;
    c.s.logodds       = inst.existence.logodds();
    c.s.p_exists      = inst.existence.p_exists();
    // A belief never touched has no age — reported as -1 rather than as "now", which would read as fresh.
    c.s.age_s         = inst.last_belief_touch.time_since_epoch().count() == 0
                      ? -1.0f
                      : std::chrono::duration<float>(now - inst.last_belief_touch).count();
    // The streak is fractional (it counts resolving LOOKS, not cycles); the card shows whole looks accrued.
    c.s.remove_streak = static_cast<int>(inst.existence_debounce.streak);
    c.s.since_det     = inst.frames_since_detection;
    c.s.initialized   = inst.ai2_initialized;
}

}  // namespace rc::dashboard
