/*
 * specificworker_dashboard.cpp — SpecificWorker's standalone Qt dashboard + evidence-monitor methods.
 *
 * Split from specificworker.cpp (same class, separate translation unit). Owns the evidence-monitor snapshot
 * push (refresh_evidence_monitor), the belief-timeseries pruning (prune_dead_series), and the dashboard
 * window-geometry persistence (restore/save_dashboard_geometry). The plots themselves are members of
 * SpecificWorker, fed from compute(); these methods manage their lifecycle and the monitor view.
 */

#include "specificworker.h"
#include "cabinet_dof.h"   // rc::kCabinetDofs / kWallRunDofs — names/units/σ* for the inspector rows
#include "cabinet_geometry.h"   // rc::geom::belief_uncertainty

#include <QByteArray>
#include <QSettings>
#include <QSplitter>
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

// Build the per-instance snapshot from live instance state and push it to the monitor at ~5 Hz (a full
// QTableWidget rebuild every compute cycle would waste the GUI thread). All reads are main-thread.
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

    // Fill one card's DOF rows + row-major Σ from any belief with a state vector `v` and covariance S.
    // Eigen stores column-major, so the copy is an explicit double loop rather than a .data() memcpy.
    const auto fill = [](rc::BeliefCard& c, const auto& S, const auto& v, const auto& specs)
    {
        constexpr int N = static_cast<int>(specs.size());
        for (int j = 0; j < N; ++j)
            c.dofs.push_back({specs[j].name, specs[j].unit, v[j],
                              std::sqrt(std::max(0.0f, S(j, j))), specs[j].sigma_star});
        c.cov.resize(N * N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                c.cov[i * N + j] = S(i, j);
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
        if (const auto* isl = kitchen_mgr_.island())   // the free-standing peninsula has no cell
        {
            rc::BeliefCard c;
            c.node  = "island";
            c.model = "wall-run island";
            const auto& S = isl->covariance();
            const auto& s = isl->state();
            const std::array<float, rc::WallRunBelief::N> v = {s.t0, s.t1, s.d, s.z0, s.z1};
            fill(c, S, v, rc::kWallRunDofs);
            c.s.logodds     = kitchen_mgr_.island_existence();
            c.s.p_exists    = 1.0f / (1.0f + std::exp(-kitchen_mgr_.island_existence()));
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

        c.s.fe            = inst.dbg_energy;
        c.s.fe_baseline   = inst.fe_baseline;
        c.s.fe_surprise   = inst.fe_surprise;
        c.s.logodds       = inst.existence.logodds();
        c.s.p_exists      = inst.existence.p_exists();
        c.s.age_s         = inst.last_belief_touch.time_since_epoch().count() == 0
                          ? -1.0f
                          : std::chrono::duration<float>(now - inst.last_belief_touch).count();
        c.s.remove_streak = inst.existence_remove_streak;
        c.s.since_det     = inst.frames_since_detection;
        c.s.initialized   = inst.ai2_initialized;
        cards.push_back(std::move(c));
    }
    belief_inspector_->update_view(cards);
}

// ─── Dashboard construction + geometry ────────────────────────────────────────────────────────────

// Persist/restore the standalone dashboard window's geometry. The generated save_window_settings()
// only covers the QMainWindow(s) in `windows`; our extracted top-level widget is separate, so we
// carry its own QSettings entry (mirrors room_concept's RoomViewer).
void SpecificWorker::restore_dashboard_geometry()
{
    if (not dashboard_window_)
        return;
    QSettings settings(QStringLiteral("RoboComp"), QStringLiteral("cabinet_concept"));
    const QByteArray geom = settings.value(QStringLiteral("DashboardWindow_geometry")).toByteArray();
    if (not geom.isEmpty())
        dashboard_window_->restoreGeometry(geom);
    else
        dashboard_window_->resize(560, 900);
}

void SpecificWorker::save_dashboard_geometry() const
{
    if (not dashboard_window_)
        return;
    QSettings settings(QStringLiteral("RoboComp"), QStringLiteral("cabinet_concept"));
    settings.setValue(QStringLiteral("DashboardWindow_geometry"), dashboard_window_->saveGeometry());
    settings.sync();
}

// Remove timeseries series for cabinets no longer present in the graph, so a removed cabinet's lines don't persist
// (and a cabinet reborn under the same name gets fresh series via the idempotent add_series in the feed). Runs
// every compute cycle; cheap (a set diff over the few live instances).
void SpecificWorker::prune_dead_series()
{
    if (not ts_plot_) return;
    std::unordered_set<std::string> live;
    for (const auto& [id, inst] : fitter_->instances()) live.insert(inst.node_name);
    for (auto it = ts_known_cabinets_.begin(); it != ts_known_cabinets_.end();)
    {
        if (live.contains(*it)) { ++it; continue; }
        const std::string& n = *it;
        ts_plot_->remove_series(n + "_fe");   ts_plot_->remove_series(n + "_base");
        if (ts_surprise_plot_) ts_surprise_plot_->remove_series(n + "_surprise");
        if (ts_cov_plot_)      ts_cov_plot_->remove_series(n + "_cov");
        if (ts_res_plot_)      ts_res_plot_->remove_series(n + "_res");
        it = ts_known_cabinets_.erase(it);
    }
    for (const auto& [id, inst] : fitter_->instances()) ts_known_cabinets_.insert(inst.node_name);
}
// Build the two standalone top-level windows (belief timeseries dashboard + evidence monitor) on the main
// thread from initialize(). Plain QWidgets (no QOpenGL), only HIDDEN on close (compute() keeps raw pointers).
void SpecificWorker::build_dashboard()
{
    // ── Time-series dashboard — its OWN top-level window ──────────────────────
    // Extracted from the DSR graph dock (add_custom_widget_to_dock) into a standalone window, so the
    // dashboard shows even with Agent.graph=false (no DSRViewer created). Mirrors room_concept and
    // kinova_controller. The TimeSeriesPlot is a plain QWidget (no QOpenGL backing store), safe as a
    // top-level. NOT WA_DeleteOnClose: closing must only HIDE it, or the ts_*_plot_ pointers the
    // compute() feed uses would dangle. A QApplication always exists (generated/main.cpp).
    {
        custom_widget_ = new Custom_widget("Cabinet — Free Energy, Surprise, Belief Uncertainty & Residuals");

        // Create plot inside frame_series
        auto* series_layout = new QVBoxLayout(custom_widget_->frame_series);
        series_layout->setContentsMargins(0, 0, 0, 0);
        custom_widget_->frame_series->setLayout(series_layout);

        ts_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
        ts_plot_->set_visible_window(60.f);
        series_layout->addWidget(ts_plot_);

        // FE SURPRISE (attention signal) on its own panel — it lives on a much smaller scale (~0–1) than the FE
        // (~2–8), so it needs the full panel height to be readable. Spikes when a cabinet moves, decays as the fit
        // re-converges. See the belief/fitter plumbing (inst.fe_surprise).
        ts_surprise_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
        ts_surprise_plot_->set_visible_window(60.f);
        series_layout->addWidget(ts_surprise_plot_);

        ts_cov_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
        ts_cov_plot_->set_visible_window(60.f);
        series_layout->addWidget(ts_cov_plot_);

        ts_res_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
        ts_res_plot_->set_visible_window(60.f);
        series_layout->addWidget(ts_res_plot_);


        // (D) BELIEF INSPECTOR — the panel that replaced the σ_w/σ_h trace. A time-series of two variances
        // showed a slice of the belief; this shows ALL of it: every state DOF (value, σ, the consumer's
        // demand σ*, the remaining adequacy gap in nats), Σ as a correlation heatmap, and the discrete-tier
        // posterior. Serves both cabinet models. Stretch 2: it is the panel you actually read.
        belief_inspector_ = new rc::BeliefInspector(QStringLiteral("belief inspector"),
                                                    custom_widget_->frame_series);
        series_layout->addWidget(belief_inspector_, 2);

        // Per-instance series are registered idempotently by the diagnostics feed (publish_cabinet_diagnostics)
        // every cycle, so pre-existing instances need no separate registration here.
    }

    // ── Evidence-consuming monitor (goes into the UPPER splitter pane) — per-instance snapshot + counters ──
    // Same lifecycle discipline as the dashboard above: plain QWidget (no QOpenGL), built on the main thread,
    // updated from compute() (which is the GUI thread here). A CHILD of the combined window now.
    evidence_monitor_ = new rc::EvidenceMonitor(QStringLiteral("cabinet_concept — evidence monitor"));

    // ── Combined window: evidence monitor (top) over the belief plots (bottom) in a resizable splitter ──
    // (mirrors table_concept). Only HIDDEN on close, never deleted (compute() keeps the raw child pointers).
    dashboard_window_ = new QWidget;
    dashboard_window_->setWindowTitle(QStringLiteral("cabinet_concept — dashboard"));
    auto* outer = new QVBoxLayout(dashboard_window_);
    outer->setContentsMargins(0, 0, 0, 0);
    auto* split = new QSplitter(Qt::Vertical, dashboard_window_);
    split->addWidget(evidence_monitor_);   // reparents into the splitter
    split->addWidget(custom_widget_);
    split->setStretchFactor(0, 0);          // counter strip keeps its (small) size
    split->setStretchFactor(1, 1);
    split->setSizes({64, 836});
    outer->addWidget(split);

    restore_dashboard_geometry();
    dashboard_window_->show();
}
