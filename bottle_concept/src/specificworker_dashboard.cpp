/*
 * specificworker_dashboard.cpp — SpecificWorker's standalone Qt dashboard + evidence-monitor methods.
 *
 * Split from specificworker.cpp (same class, separate translation unit), mirroring table_concept. Owns the
 * window construction (build_dashboard), the evidence-monitor snapshot push (refresh_evidence_monitor), the
 * belief inspector and strip feeds, and the window-geometry persistence. The windows, their layout, the
 * details toggle and the geometry are SHARED (common/dashboard/); what stays here is what names this concept
 * and how a bottle's belief maps onto a card. The time-series plots are fed from publish_bottle_diagnostics.
 */

#include "specificworker.h"

#include "../../common/dashboard/belief_certainty.h"   // rc::dash::fill_certainty (SHARED)
#include "../../common/dashboard/window_geometry.h"    // rc::dashboard:: (SHARED)
#include "../../common/dashboard/concept_dashboard.h"  // rc::dashboard::build (SHARED)
#include "../../common/dashboard/belief_card_fill.h"   // rc::dashboard::fill_* (SHARED)
#include "bottle_dof.h"   // rc::kBottleDofs — names/units for the BeliefInspector rows

#include <array>
#include <chrono>
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
    refresh_belief_strip();   // same tick, same instance pass
}

// ─── Compact belief strip ────────────────────────────────────────────────────────────────────────

// One row per bottle. gap_nats is the -1 sentinel because kBottleDofs declares no σ* (see the note there),
// so the widget falls back to ½·ln det Σ — which is why the number reads NEGATIVE and is a log-volume, not a
// remaining-work figure. p(exists) comes from the shared rc::exist channel (bottle_existence.cpp).
//
// ★This comment used to say "p(exists) is unavailable (bottle has no existence channel)" and the field was
// left NaN. That became false the moment the channel landed, and the strip renders NaN as an EMPTY GREY BAR
// with a dash — indistinguishable from "this agent has no existence belief". So the one display whose entire
// job is "is it still there?" showed nothing while L was live, and stayed silent through both the NaN poison
// and the LiDAR outvoting a live detection. Two widgets, two feeds: fixing the inspector card did not fix
// this one.
void SpecificWorker::refresh_belief_strip()
{
    // The loop, the REPORTED covariance, the creation-stamp birth time and update_view() are SHARED
    // (common/dashboard/belief_strip.h). What stays is what only this agent knows: where p_exists
    // comes from and which DOF table describes the belief.
    rc::dash::fill_strip(belief_strip_, fitter_->instances(), *G,
        [](rc::BeliefStripRow& r, const auto& inst)
        {
            r.p_exists = inst.existence.p_exists();
            rc::dash::fill_certainty(r, inst.ai2_belief.covariance(), rc::kBottleDofs);   // bottle has no covariance_reported()
        });
}

// ─── Belief inspector ────────────────────────────────────────────────────────────────────────────

// Build the per-instance BELIEF snapshot (state, Σ, scalar gauges) and push it to the bottom panel. Called
// from refresh_evidence_monitor(), so it inherits that method's ~5 Hz gate and its single pass over the live
// instances. Deliberately NOT hung off publish_bottle_diagnostics, which runs per-node. Main-thread only.
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

        // A bottle is a yaw-symmetric cylinder: no orientation DOF and no discrete modes, so Σ is the plain
        // posterior (there is no covariance_reported() to fold a mode entropy into).
        const auto& S = inst.ai2_belief.covariance();
        const auto& s = inst.ai2_belief.state();
        const std::array<float, rc::BottleBelief::N> v = {s.cx, s.cy, s.cz, s.radius, s.height};
        rc::dashboard::fill_dofs(c, rc::kBottleDofs, v, S);
        rc::dashboard::fill_cov(c, S, rc::BottleBelief::N);

        // SHARED gauges. Bottle's two particulars are handled there by member detection: prev_free_energy is
        // seeded to FLT_MAX (reads "-" until the first fit), and existence shows only once existence_seeded.
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
    rc::dashboard::restore_window_geometry(dashboard_window_, QStringLiteral("bottle_concept"), QStringLiteral("DashboardWindow"),
                                           QSize(1180, 900));
}

void SpecificWorker::save_dashboard_geometry() const
{
    rc::dashboard::save_window_geometry(dashboard_window_, QStringLiteral("bottle_concept"), QStringLiteral("DashboardWindow"));
}

void SpecificWorker::restore_strip_geometry()
{
    rc::dashboard::restore_window_geometry(strip_window_, QStringLiteral("bottle_concept"), QStringLiteral("BeliefStripWindow"),
                                           QSize(520, 210), QSize(900, 420));
}

void SpecificWorker::save_strip_geometry() const
{
    rc::dashboard::save_window_geometry(strip_window_, QStringLiteral("bottle_concept"), QStringLiteral("BeliefStripWindow"));
}

// Build the belief strip (small, always open) and the full dashboard behind its "details" button. Both
// windows, their layout, the toggle and the geometry are SHARED (common/dashboard/concept_dashboard.h);
// what stays here is the four strings that name this concept and where the widget pointers land.
//
// Main thread, from initialize(). Bottle has no headless switch, so it always builds (enabled = true).
// Neither window is WA_DeleteOnClose: publish_bottle_diagnostics holds the raw plot pointers.
void SpecificWorker::build_dashboard()
{
    const rc::dashboard::BuildSpec spec{
        .agent        = QStringLiteral("bottle_concept"),
        .series_title = QStringLiteral("Bottle — Free Energy, Surprise, Belief Uncertainty & Residuals"),
        .strip_title  = QStringLiteral("bottle_concept — beliefs"),
        .strip_label  = QStringLiteral("bottles"),
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

    // GenericWorker::initialize() may have started compute() already, so some instances can exist.
    for (auto& [_, inst] : fitter_->instances())
        publish_bottle_diagnostics(inst, inst.prev_free_energy);
}
