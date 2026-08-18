/*
 * specificworker_dashboard.cpp — SpecificWorker's standalone Qt dashboard + evidence-monitor methods.
 *
 * Split from specificworker.cpp (same class, separate translation unit). Owns the evidence-monitor snapshot
 * push (refresh_evidence_monitor), the belief-timeseries pruning (prune_dead_series), and the dashboard
 * window-geometry persistence (restore/save_dashboard_geometry). The plots themselves are members of
 * SpecificWorker, fed from compute(); these methods manage their lifecycle and the monitor view.
 */

#include "specificworker.h"

#include "../../common/dashboard/belief_certainty.h"   // rc::dash::fill_certainty (SHARED)

#include "../../common/dashboard/window_geometry.h"    // rc::dashboard:: (SHARED)
#include "../../common/dashboard/concept_dashboard.h"  // rc::dashboard::build (SHARED)
#include "../../common/dashboard/belief_card_fill.h"   // rc::dashboard::fill_* (SHARED)
#include "../../common/dashboard/series_pruner.h"     // rc::dashboard:: (SHARED)
#include "cabinet_dof.h"   // rc::kCabinetDofs / kWallRunDofs — names/units/σ* for the inspector rows
#include "cabinet_geometry.h"   // rc::geom::belief_uncertainty

#include <QByteArray>
#include <QSettings>
#include <QPushButton>
#include <QHBoxLayout>
#include <QFont>
#include <QFontMetrics>
#include <QVBoxLayout>
#include <QWidget>
#include <QColor>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <string>
#include <unordered_set>
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
    refresh_belief_inspector();   // same tick, same instance pass — the two views can never disagree
    refresh_belief_strip();       // …and the compact strip reads the same cycle as both
}

// ─── Belief inspector ────────────────────────────────────────────────────────────────────────────

// Build the per-instance BELIEF snapshot and push it to the bottom panel. Called from
// refresh_evidence_monitor(), so it inherits that method's ~5 Hz gate. All reads are main-thread.
//
// Two models, one card format — which is exactly what the agent-agnostic BeliefInspector API buys:
//   • KITCHEN model: identity IS the (wall, tier) cell, so each ACTIVE cell owns a 5-DOF WallRunBelief
//     [t0,t1,d,z0,z1] in its wall chart (yaw and lateral placement are unrepresentable by construction).
//   • classic model: free 7-DOF boxes [cx,cy,yaw,L,d,z0,z1] tracked as instances, with a tier mode.
void SpecificWorker::refresh_belief_inspector()
{
    if (not belief_inspector_)
        return;
    const auto now = std::chrono::steady_clock::now();
    std::vector<rc::BeliefCard> cards;

    // DOF rows + the row-major Σ copy are SHARED (common/dashboard/belief_card_fill.h) — this agent had
    // factored them into a local lambda, which was the same code a fifth time. The Eigen column-major
    // transpose hazard is documented there, once.
    const auto fill = [](rc::BeliefCard& c, const auto& S, const auto& v, const auto& specs)
    {
        rc::dashboard::fill_dofs(c, specs, v, S);
        rc::dashboard::fill_cov(c, S, static_cast<int>(specs.size()));
    };

    if (cfg_.kitchen_model)
    {
        for (const auto& cell : kitchen_mgr_.cells())
        {
            if (not cell.active())
                continue;   // a candidate cell has no belief yet — nothing to inspect
            rc::BeliefCard c;
            c.node  = cell.geom.id;
            c.model = cell.geom.tier == 0 ? "wall-run base" : "wall-run upper";
            const auto& S = cell.belief->covariance();
            const auto& s = cell.belief->state();
            const std::array<float, rc::WallRunBelief::N> v = {s.t0, s.t1, s.d, s.z0, s.z1};
            fill(c, S, v, rc::kWallRunDofs);
            c.s.fe          = cell.fe;
            c.s.logodds     = cell.existence;
            c.s.p_exists    = 1.0f / (1.0f + std::exp(-cell.existence));
            c.s.initialized = true;   // an active cell is by definition a live belief
            cards.push_back(std::move(c));
        }
        if (const auto* pen = kitchen_mgr_.peninsula())   // the peninsula has no cell
        {
            rc::BeliefCard c;
            c.node  = "peninsula";
            c.model = "wall-run peninsula";
            const auto& S = pen->covariance();
            const auto& s = pen->state();
            const std::array<float, rc::WallRunBelief::N> v = {s.t0, s.t1, s.d, s.z0, s.z1};
            fill(c, S, v, rc::kWallRunDofs);
            c.s.logodds     = kitchen_mgr_.peninsula_existence();
            c.s.p_exists    = 1.0f / (1.0f + std::exp(-kitchen_mgr_.peninsula_existence()));
            c.s.initialized = true;
            cards.push_back(std::move(c));
        }
        belief_inspector_->update_view(cards);
        return;
    }

    cards.reserve(fitter_->instances().size());
    for (const auto& [id, inst] : fitter_->instances())
    {
        rc::BeliefCard c;
        c.node = inst.node_name;

        // REPORTED covariance: the (d,z0,z1) diagonals carry the discrete-tier entropy, so a cabinet whose
        // tier is unresolved shows the honest depth/height uncertainty. Same matrix the NBV planner scores.
        const auto  S = inst.ai2_belief.covariance_reported();
        const auto& s = inst.ai2_belief.state();
        const std::array<float, rc::CabinetBelief::N> v = {s.cx, s.cy, s.yaw, s.L, s.d, s.z0, s.z1};
        fill(c, S, v, rc::kCabinetDofs);

        const bool  base  = inst.ai2_belief.tier() == rc::CabinetTier::Base;
        const float p_alt = inst.ai2_belief.tier_posterior();   // p(the OTHER tier)
        c.modes.push_back({"tier", base ? "base" : "wall", 1.0f - p_alt});
        c.modes.push_back({"tier", base ? "wall" : "base", p_alt});

        rc::dashboard::fill_scalars(c, inst, now);
        cards.push_back(std::move(c));
    }
    belief_inspector_->update_view(cards);
}

// ─── Dashboard construction + geometry ────────────────────────────────────────────────────────────


// No max size here: the full dashboard is one the user may legitimately maximise, so its saved state is
// honoured as-is. That asymmetry is the whole reason max_restored is a parameter and not a constant.
void SpecificWorker::restore_dashboard_geometry()
{
    rc::dashboard::restore_window_geometry(dashboard_window_, QStringLiteral("cabinet_concept"), QStringLiteral("DashboardWindow"),
                                           QSize(560, 900));
}

void SpecificWorker::save_dashboard_geometry() const
{
    rc::dashboard::save_window_geometry(dashboard_window_, QStringLiteral("cabinet_concept"), QStringLiteral("DashboardWindow"));
}


// Remove the timeseries of instances no longer in the graph, so a removed one's lines don't persist (and one
// reborn under the same name gets fresh series via the idempotent add_series in the feed). SHARED — the
// agent supplies only the live names and its own remembered set.
void SpecificWorker::prune_dead_series()
{
    std::vector<std::string> live;
    live.reserve(fitter_->instances().size());
    for (const auto& [id, inst] : fitter_->instances()) live.push_back(inst.node_name);
    rc::dashboard::prune_dead_series({ts_plot_, ts_surprise_plot_, ts_cov_plot_, ts_res_plot_},
                                     live, ts_known_cabinets_);
}

// Build the belief strip (small, always open) and the full dashboard behind its "details" button. Both
// windows, their layout, the toggle and the geometry are SHARED (common/dashboard/concept_dashboard.h);
// what stays here is the four strings that name this concept and where the widget pointers land.
//
// Main thread, from initialize(). cfg_.show_dashboard = false builds NOTHING: every pointer stays null and
// the compute feed already no-ops on null.
void SpecificWorker::build_dashboard()
{
    const rc::dashboard::BuildSpec spec{
        .agent        = QStringLiteral("cabinet_concept"),
        .series_title = QStringLiteral("Cabinet — Free Energy, Surprise, Belief Uncertainty & Residuals"),
        .strip_title  = QStringLiteral("cabinet_concept — beliefs"),
        .strip_label  = QStringLiteral("cabinets"),
        .dashboard_size = QSize(560, 900),
    };
    const auto w = rc::dashboard::build(spec, cfg_.show_dashboard);

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
}

// One row per cabinet: the adequacy gap (nats still missing before Σ meets the consumer's σ*), the
// existence probability, the FE surprise, and the node's birth stamp. The widget owns the history — this
// only pushes the current instant, on the same throttled tick as the panels above.
void SpecificWorker::refresh_belief_strip()
{
    if (not belief_strip_)
        return;   // headless: nothing was built

    std::vector<rc::BeliefStripRow> rows;

    // ★THE DISPLAY UNIT FOLLOWS THE MODEL, exactly as refresh_belief_inspector() does. Under the
    // KITCHEN-OF-RUNS model the beliefs are owned by the (wall, tier) CELLS — `fitter_->instances()` is not
    // the thing on screen, and iterating it here is why the strip came up empty. Keep the two views reading
    // the same container or they will disagree about what exists.
    if (cfg_.kitchen_model)
    {
        for (const auto& cell : kitchen_mgr_.cells())
        {
            if (not cell.active())
                continue;   // a candidate cell has no belief yet — nothing to trace
            rc::BeliefStripRow r;
            r.node        = cell.geom.id;
            r.model       = cell.geom.tier == 0 ? "wall-run base" : "wall-run upper";
            r.p_exists    = 1.0f / (1.0f + std::exp(-cell.existence));
            r.initialized = true;   // an active cell is by definition a live belief

            const auto& S = cell.belief->covariance();
            rc::dash::fill_certainty(r, S, rc::kWallRunDofs);
            rows.push_back(std::move(r));
        }
        if (const auto* pen = kitchen_mgr_.peninsula())   // the peninsula has no cell
        {
            rc::BeliefStripRow r;
            r.node        = "peninsula";
            r.model       = "wall-run peninsula";
            r.p_exists    = 1.0f / (1.0f + std::exp(-kitchen_mgr_.peninsula_existence()));
            r.initialized = true;
            const auto& S = pen->covariance();
            rc::dash::fill_certainty(r, S, rc::kWallRunDofs);
            rows.push_back(std::move(r));
        }
        belief_strip_->update_view(rows);
        return;
    }

    // Per-instance model (kitchen_model off): one row per cabinet instance. THIS branch is the INSTANCE
    // family and uses the shared filler; the kitchen branch above cannot, because its unit is a CELL — no
    // node, no node_name, no fe_surprise, no creation stamp to read an age from. That is the family boundary,
    // not an omission.
    rc::dash::fill_strip(belief_strip_, fitter_->instances(), *G,
        [](rc::BeliefStripRow& r, const auto& inst)
        {
            r.p_exists = inst.existence.p_exists();
            rc::dash::fill_certainty(r, inst.ai2_belief.covariance_reported(), rc::kCabinetDofs);
        });
}

// Geometry persistence is SHARED (common/dashboard/window_geometry.h). The strip passes a max size: a
// restored geometry carries the window STATE, so a strip that was ever maximised comes back filling the
// screen — indistinguishable, to the user, from the big dashboard having opened itself.
void SpecificWorker::restore_strip_geometry()
{
    rc::dashboard::restore_window_geometry(strip_window_, QStringLiteral("cabinet_concept"), QStringLiteral("BeliefStripWindow"),
                                           QSize(520, 210), QSize(900, 420));
}

void SpecificWorker::save_strip_geometry() const
{
    rc::dashboard::save_window_geometry(strip_window_, QStringLiteral("cabinet_concept"), QStringLiteral("BeliefStripWindow"));
}
