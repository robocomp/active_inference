/*
 * common/dashboard/belief_series.h — the four per-instance dashboard traces, written once. SHARED.
 *
 * Every concept agent feeds the same four plots for each believed instance: free energy (with its baseline
 * drawn on the same axes), the FE SURPRISE on its own panel because it lives on a much smaller scale, the
 * belief uncertainty U(Σ), and the residual point count. The code that did it was ~35 lines per agent and
 * byte-identical across refrigerator, table, hood and cabinet.
 *
 * ★THE SEAM IS A SAMPLE, NOT AN INSTANCE. Each agent's Instance type is its own, and `belief_uncertainty()`
 * is per-agent (rc::geom:: for most, a local for chair, an inline σ sum over the bottle's own DOFs) — so the
 * caller reduces its instance to the six numbers a plot can actually draw and hands those over. That also
 * keeps this file free of every agent's headers.
 *
 * ★★AND IT CARRIES A FIX THAT HAD REACHED FOUR AGENTS OF SIX. The residual trace must plot a HELD value —
 * the same number the EvidenceMonitor counter shows, which keeps its last reading between mask frames. Chair
 * and door still plotted `observation.residual_pts.size()` gated on `has_fresh_data`, so on any non-fresh
 * publish they added no point at all and the line simply vanished, disagreeing with the counter beside it.
 * There is one residual input here and it is the held one; the fresh-only variant is not expressible.
 */

#pragma once

#include <string>

#include <QColor>

#include "timeseries_plot.h"

namespace rc::dash
{

// The four plots an agent owns. Any of them may be null (headless, or a dashboard built without that panel).
struct BeliefPlots
{
    rc::TimeSeriesPlot* fe       = nullptr;   // free energy + its baseline
    rc::TimeSeriesPlot* surprise = nullptr;   // FE surprise — own panel, much smaller scale
    rc::TimeSeriesPlot* cov      = nullptr;   // belief uncertainty U(Σ)
    rc::TimeSeriesPlot* res      = nullptr;   // residual point count
};

// One instance reduced to what a trace can draw.
struct BeliefSample
{
    std::string node;                  // series key — the instance's node name
    float free_energy  = 0.0f;
    float fe_baseline  = -1.0f;        // <0 ⇒ uninitialised (before the first fit) ⇒ baseline+surprise skipped
    float fe_surprise  = 0.0f;
    float uncertainty  = 0.0f;         // U(Σ) = Σ position+size posterior std (m)
    float residual_pts = 0.0f;         // ★the HELD count, not a fresh-frames-only one — see the header note
};

// Register the series lazily and idempotently, then add this cycle's points.
//
// ★REGISTRATION HAPPENS HERE, on the thread that adds the points, and every cycle. An instance is often
// created down the graph-signal path BEFORE the plots exist, so a one-shot `if (created)` registration at
// birth never fires for it and every later point is dropped on the floor.
inline void publish_belief_series(const BeliefPlots& p, const BeliefSample& s)
{
    if (p.fe == nullptr)
        return;                                   // no dashboard this run — nothing downstream to feed

    p.fe->add_series(s.node + "_fe", QColor(255, 170, 0), 1.1f);
    p.fe->add_point (s.node + "_fe", s.free_energy);
    // The baseline shares the FE axes deliberately: the FE line lifting ABOVE the grey baseline IS the
    // surprise, shown visually. The smoothed gap gets its own panel because of the scale difference.
    p.fe->add_series(s.node + "_base", QColor(140, 140, 140), 0.9f);
    if (p.surprise) p.surprise->add_series(s.node + "_surprise", QColor(255, 60, 60), 1.3f);
    if (s.fe_baseline >= 0.0f)
    {
        p.fe->add_point(s.node + "_base", s.fe_baseline);
        if (p.surprise) p.surprise->add_point(s.node + "_surprise", s.fe_surprise);
    }
    if (p.cov)
    {
        p.cov->add_series(s.node + "_cov", QColor(0, 190, 255), 1.1f);
        p.cov->add_point (s.node + "_cov", s.uncertainty);
    }
    if (p.res)
    {
        p.res->add_series(s.node + "_res", QColor(170, 80, 255), 1.1f);
        p.res->add_point (s.node + "_res", s.residual_pts);
    }
}

}  // namespace rc::dash
