/*
 * specificworker_dashboard.cpp — SpecificWorker's standalone Qt dashboard + evidence-monitor methods.
 *
 * Split from specificworker.cpp (same class, separate translation unit). Owns the evidence-monitor snapshot
 * push (refresh_evidence_monitor), the belief-timeseries pruning (prune_dead_series), and the dashboard
 * window-geometry persistence (restore/save_dashboard_geometry). The plots themselves are members of
 * SpecificWorker, fed from compute(); these methods manage their lifecycle and the monitor view.
 */

#include "specificworker.h"
#include "table_geometry.h"   // rc::geom::belief_uncertainty

#include <QByteArray>
#include <QSettings>
#include <QSplitter>
#include <QVBoxLayout>
#include <QColor>

#include <algorithm>
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

    std::vector<rc::EvidenceRow> rows;
    rows.reserve(fitter_->instances().size());
    for (const auto& [id, inst] : fitter_->instances())
    {
        rc::EvidenceRow x;
        x.node       = inst.node_name;
        x.conf       = inst.last_mask_confidence;
        x.since_det  = inst.frames_since_detection;
        x.n_zed      = inst.dbg_n_zed_slices;
        x.n_ricoh    = 0;   // ricoh is bearing-only now (never fused); column kept for the monitor layout
        x.cand       = inst.dbg_cand_pts;
        x.resid      = inst.dbg_resid_pts;
        x.gated      = inst.dbg_gated;
        x.energy     = inst.dbg_energy;
        x.R          = inst.dbg_R;
        x.lidar_rays = inst.dbg_lidar_rays;
        x.lidar_raw  = inst.dbg_lidar_raw;
        x.lidar_resid= inst.dbg_lidar_resid_m;
        x.cov_ang    = inst.dbg_lidar_cov_ang;
        x.vacate     = inst.ai2_belief.last_vacate_beams();
        x.coverage   = inst.ai2_belief.last_coverage_pts();
        x.L          = inst.existence.logodds();
        x.p          = inst.existence.p_exists();
        x.ex_locc    = inst.dbg_ex_lidar_occ;  x.ex_lfree = inst.dbg_ex_lidar_free;  x.ex_ln    = inst.dbg_ex_lidar_n;
        x.ex_lfree_eff = inst.dbg_ex_lidar_free_eff;
        x.ex_socc    = inst.dbg_ex_sil_occ;    x.ex_sfree = inst.dbg_ex_sil_free;    x.ex_sndet = inst.dbg_ex_sil_ndet;
        x.ex_sfree_eff = inst.dbg_ex_sil_free_eff;
        x.streak     = inst.existence_remove_streak;
        const auto& s = inst.ai2_belief.state();
        x.w = s.w; x.h = s.h; x.H = s.H;
        x.subtype = inst.subtype;   // round/square from the free-energy shape model-selection
        rows.push_back(std::move(x));
    }
    evidence_monitor_->update_view(ev_g_, rows);
}

// ─── Dashboard construction + geometry ────────────────────────────────────────────────────────────

// Persist/restore the standalone dashboard window's geometry. The generated save_window_settings()
// only covers the QMainWindow(s) in `windows`; our extracted top-level widget is separate, so we
// carry its own QSettings entry (mirrors room_concept's RoomViewer).
void SpecificWorker::restore_dashboard_geometry()
{
    if (not dashboard_window_)
        return;
    QSettings settings(QStringLiteral("RoboComp"), QStringLiteral("table_concept"));
    const QByteArray geom = settings.value(QStringLiteral("DashboardWindow_geometry")).toByteArray();
    if (not geom.isEmpty())
        dashboard_window_->restoreGeometry(geom);
    else
        dashboard_window_->resize(1180, 900);
}

void SpecificWorker::save_dashboard_geometry() const
{
    if (not dashboard_window_)
        return;
    QSettings settings(QStringLiteral("RoboComp"), QStringLiteral("table_concept"));
    settings.setValue(QStringLiteral("DashboardWindow_geometry"), dashboard_window_->saveGeometry());
    settings.sync();
}

// Remove timeseries series for tables no longer present in the graph, so a removed table's lines don't persist
// (and a table reborn under the same name gets fresh series via the idempotent add_series in the feed). Runs
// every compute cycle; cheap (a set diff over the few live instances).
void SpecificWorker::prune_dead_series()
{
    if (not ts_plot_) return;
    std::unordered_set<std::string> live;
    for (const auto& [id, inst] : fitter_->instances()) live.insert(inst.node_name);
    for (auto it = ts_known_tables_.begin(); it != ts_known_tables_.end();)
    {
        if (live.contains(*it)) { ++it; continue; }
        const std::string& n = *it;
        ts_plot_->remove_series(n + "_fe");   ts_plot_->remove_series(n + "_base");
        if (ts_surprise_plot_) ts_surprise_plot_->remove_series(n + "_surprise");
        if (ts_cov_plot_)      ts_cov_plot_->remove_series(n + "_cov");
        if (ts_res_plot_)      ts_res_plot_->remove_series(n + "_res");
        if (ts_state_plot_)  { ts_state_plot_->remove_series(n + "_w");  ts_state_plot_->remove_series(n + "_h"); }
        if (ts_ce_plot_)     { ts_ce_plot_->remove_series(n + "_sW");    ts_ce_plot_->remove_series(n + "_sH"); }
        it = ts_known_tables_.erase(it);
    }
    for (const auto& [id, inst] : fitter_->instances()) ts_known_tables_.insert(inst.node_name);
}
// Build ONE combined top-level window — the belief timeseries dashboard + the evidence monitor stacked in a
// vertical splitter — on the main thread from initialize(). cfg_.show_dashboard=false builds NOTHING (headless):
// the widget pointers stay null and the compute feed (publish_table_diagnostics / refresh_evidence_monitor /
// prune_dead_series) already no-ops on null. Plain QWidgets (no QOpenGL); only HIDDEN on close (compute() keeps
// raw pointers, so never WA_DeleteOnClose). A QApplication always exists (generated/main.cpp).
void SpecificWorker::build_dashboard()
{
    if (not cfg_.show_dashboard)
        return;   // headless: no GUI windows built; the compute feed no-ops on the null widgets

    // ── Belief timeseries dashboard (goes into the LOWER splitter pane) ───────
    // The TimeSeriesPlot is a plain QWidget; here it is a CHILD of the combined window rather than its own
    // top-level. Shows even with Agent.graph=false (no DSRViewer).
    {
        custom_widget_ = new Custom_widget("Table Model — Free Energy, Belief Uncertainty, Residuals & Dimensions (w,h)");

        // Create plot inside frame_series
        auto* series_layout = new QVBoxLayout(custom_widget_->frame_series);
        series_layout->setContentsMargins(0, 0, 0, 0);
        custom_widget_->frame_series->setLayout(series_layout);

        ts_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
        ts_plot_->set_visible_window(60.f);
        series_layout->addWidget(ts_plot_);

        // FE SURPRISE (attention signal) on its own panel — it lives on a much smaller scale (~0–1) than the FE
        // (~2–8), so it needs the full panel height to be readable. Spikes when a table moves, decays as the fit
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

        // Inferred table dimensions (w, h) — the size DOFs the stabiliser targets. Watch these to
        // confirm the belief has stopped jittering between fresh masks (flat = stable).
        ts_state_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
        ts_state_plot_->set_visible_window(60.f);
        series_layout->addWidget(ts_state_plot_);

        // (D) Belief SIZE POSTERIOR STD σ_w/σ_h, in mm (fed from specificworker.cpp; see SpecificWorker's member
        // doc). This panel used to show a counter-evidence CUSUM accumulator; that quantity and its gate no
        // longer exist. Read it as convergence: σ falls as the extent is resolved, and RISES again when the
        // evidence goes stale (the age-inflation path) or a surprise widens the belief.
        {
            ts_ce_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
            ts_ce_plot_->set_visible_window(60.f);
            series_layout->addWidget(ts_ce_plot_);
        }

        // Per-instance series are registered idempotently by the diagnostics feed (publish_table_diagnostics)
        // every cycle, so pre-existing instances need no separate registration here.
    }

    // ── Evidence-consuming monitor (goes into the UPPER splitter pane) — per-instance snapshot + counters ──
    // Same lifecycle: plain QWidget, main-thread, updated from compute(). A CHILD of the combined window now.
    evidence_monitor_ = new rc::EvidenceMonitor(QStringLiteral("table_concept — evidence monitor"));

    // ── Combined window: evidence monitor (top) over the belief plots (bottom) in a resizable splitter ──
    dashboard_window_ = new QWidget;
    dashboard_window_->setWindowTitle(QStringLiteral("table_concept — dashboard"));
    auto* outer = new QVBoxLayout(dashboard_window_);
    outer->setContentsMargins(0, 0, 0, 0);
    auto* split = new QSplitter(Qt::Vertical, dashboard_window_);
    split->addWidget(evidence_monitor_);   // reparents into the splitter
    split->addWidget(custom_widget_);
    split->setStretchFactor(0, 0);          // monitor keeps its size; plots take the extra space
    split->setStretchFactor(1, 1);
    split->setSizes({320, 580});
    outer->addWidget(split);

    restore_dashboard_geometry();
    dashboard_window_->show();
}
