/*
 * specificworker_dashboard.cpp — SpecificWorker's standalone Qt dashboard + evidence-monitor methods.
 *
 * Split from specificworker.cpp (same class, separate translation unit), mirroring table_concept. Owns the
 * window construction (build_dashboard), the evidence-monitor snapshot push (refresh_evidence_monitor), the
 * belief inspector and strip feeds, and the window-geometry persistence. The windows, their layout, the
 * details toggle and the geometry are SHARED (common/dashboard/); what stays here is what names this concept
 * and how a chair's belief maps onto a card. The time-series plots are fed from compute().
 */

#include "specificworker.h"

#include "../../common/dashboard/belief_certainty.h"   // rc::dash::fill_certainty (SHARED)
#include "../../common/dashboard/window_geometry.h"    // rc::dashboard:: (SHARED)
#include "../../common/dashboard/concept_dashboard.h"  // rc::dashboard::build (SHARED)
#include "../../common/dashboard/belief_card_fill.h"   // rc::dashboard::fill_* (SHARED)
#include "chair_dof.h"   // rc::kChairDofs — names/units/σ* for the BeliefInspector rows

#include <QColor>

#include <array>
#include <chrono>
#include <cmath>
#include <vector>

// ─── Evidence monitor ────────────────────────────────────────────────────────────────────────────

// Section 1: push this cycle's evidence-pipeline counters, then (same tick, same data) the belief
// inspector — so the two sections can never show different cycles. Throttled to ~5 Hz. Main-thread.
void SpecificWorker::refresh_evidence_monitor()
{
    if (not evidence_monitor_)
        return;
    const auto now = std::chrono::steady_clock::now();
    if (last_monitor_tp_.time_since_epoch().count() != 0
        and std::chrono::duration<float>(now - last_monitor_tp_).count() < 0.2f)
        return;
    last_monitor_tp_ = now;

    evidence_monitor_->update_view(ev_g_);
    refresh_belief_inspector();
    refresh_belief_strip();   // same tick, same instance pass — the views can never disagree
}

// ─── Compact belief strip ────────────────────────────────────────────────────────────────────────

// One row per chair: the adequacy gap (nats still missing before Σ meets the consumer's σ*), p(exists),
// the FE surprise and the node's birth stamp. The widget owns the history — this pushes the current
// instant only, on the same throttled tick as the two panels above.
void SpecificWorker::refresh_belief_strip()
{
    // The loop, the REPORTED covariance, the creation-stamp birth time and update_view() are SHARED
    // (common/dashboard/belief_strip.h). What stays is what only this agent knows: where p_exists
    // comes from and which DOF table describes the belief.
    rc::dash::fill_strip(belief_strip_, fitter_->instances(), *G,
        [](rc::BeliefStripRow& r, const auto& inst)
        {
            // exist_logodds is NaN until the existence channel initialises — leave p_exists NaN in
            // that case (the widget reads NaN as "no existence belief yet") rather than inventing 0.5.
            if (std::isfinite(inst.exist_logodds))
                r.p_exists = 1.0f / (1.0f + std::exp(-inst.exist_logodds));
            rc::dash::fill_certainty(r, inst.ai2_belief.covariance_reported(), rc::kChairDofs);
        });
}

// ─── Belief inspector ────────────────────────────────────────────────────────────────────────────

// Build the per-instance BELIEF snapshot (pose state, Σ, the 4-mode yaw posterior, scalar gauges) and push
// it to the bottom panel. Called from refresh_evidence_monitor(), so it inherits that method's ~5 Hz gate
// and its single pass over the live instances. Main-thread.
void SpecificWorker::refresh_belief_inspector()
{
    if (not belief_inspector_)
        return;
    const auto now = std::chrono::steady_clock::now();

    std::vector<rc::BeliefCard> cards;
    cards.reserve(fitter_->instances().size());
    for (const auto& [id, inst] : fitter_->instances())
    {
        rc::BeliefCard c;
        c.node = inst.node_name;

        // REPORTED covariance: σ_yaw carries the 4-mode orientation entropy, so an unresolved chair shows
        // the honest orientation uncertainty rather than the within-mode one.
        const auto  S = inst.ai2_belief.covariance_reported();
        const auto& s = inst.ai2_belief.state();
        const std::array<float, rc::ChairBelief::N> v = {s.cx, s.cy, s.yaw};
        rc::dashboard::fill_dofs(c, rc::kChairDofs, v, S);
        rc::dashboard::fill_cov(c, S, rc::ChairBelief::N);

        // The chair's discrete ambiguity is its 4-fold yaw symmetry (front/back/side of a squarish seat).
        const auto pm = inst.ai2_belief.mode_posterior();
        static const char* kYawLabels[4] = {"m0", "m0+90", "m0+180", "m0-90"};
        for (int k = 0; k < 4; ++k)
            c.modes.push_back({"yaw", kYawLabels[k], pm[k]});

        // SHARED gauges. exist_logodds is NaN until the existence channel is seeded; fill_scalars detects
        // that mirror and leaves the existence gauges at "-" rather than a fake 0.5.
        rc::dashboard::fill_scalars(c, inst, now);
        cards.push_back(std::move(c));
    }
    belief_inspector_->update_view(cards);
}

// ─── Dashboard construction + geometry ───────────────────────────────────────────────────────────

// No max size here: the full dashboard is one the user may legitimately maximise, so its saved state is
// honoured as-is. The strip passes one — see window_geometry.h.
void SpecificWorker::restore_dashboard_geometry()
{
    rc::dashboard::restore_window_geometry(dashboard_window_, QStringLiteral("chair_concept"), QStringLiteral("DashboardWindow"),
                                           QSize(1180, 900));
}

void SpecificWorker::save_dashboard_geometry() const
{
    rc::dashboard::save_window_geometry(dashboard_window_, QStringLiteral("chair_concept"), QStringLiteral("DashboardWindow"));
}

void SpecificWorker::restore_strip_geometry()
{
    rc::dashboard::restore_window_geometry(strip_window_, QStringLiteral("chair_concept"), QStringLiteral("BeliefStripWindow"),
                                           QSize(520, 210), QSize(900, 420));
}

void SpecificWorker::save_strip_geometry() const
{
    rc::dashboard::save_window_geometry(strip_window_, QStringLiteral("chair_concept"), QStringLiteral("BeliefStripWindow"));
}

// Build the belief strip (small, always open) and the full dashboard behind its "details" button. Both
// windows, their layout, the toggle and the geometry are SHARED (common/dashboard/concept_dashboard.h);
// what stays here is the four strings that name this concept and where the widget pointers land.
//
// Main thread, from initialize(). Chair has no headless switch, so it always builds (enabled = true).
// Neither window is WA_DeleteOnClose: the compute() feed holds the raw plot pointers.
void SpecificWorker::build_dashboard()
{
    const rc::dashboard::BuildSpec spec{
        .agent        = QStringLiteral("chair_concept"),
        .series_title = QStringLiteral("Chair — Free Energy, Surprise, Belief Uncertainty & Residuals"),
        .strip_title  = QStringLiteral("chair_concept — beliefs"),
        .strip_label  = QStringLiteral("chairs"),
        .dashboard_size = QSize(1180, 900),
    };
    const auto w = rc::dashboard::build(spec, true);

    custom_widget_    = w.custom;
    ts_plot_          = w.fe;
    ts_surprise_plot_ = w.surprise;
    ts_cov_plot_      = w.cov;
    ts_res_plot_      = w.res;
    belief_inspector_ = w.inspector;
    evidence_monitor_ = w.evidence;
    dashboard_window_ = w.dashboard_window;
    strip_window_     = w.strip_window;
    belief_strip_     = w.strip;

    // GenericWorker::initialize() may have already started compute(), so some instances can exist before
    // the plots are constructed.
    for (const auto& [_, inst] : fitter_->instances())
    {
        ts_plot_->add_series(inst.node_name + "_fe", QColor(255, 170, 0), 1.1f);
        ts_cov_plot_->add_series(inst.node_name + "_cov", QColor(0, 190, 255), 1.1f);
        ts_res_plot_->add_series(inst.node_name + "_res", QColor(170, 80, 255), 1.1f);
    }
}
