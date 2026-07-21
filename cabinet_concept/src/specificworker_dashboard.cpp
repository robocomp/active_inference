/*
 * specificworker_dashboard.cpp — SpecificWorker's standalone Qt dashboard + evidence-monitor methods.
 *
 * Split from specificworker.cpp (same class, separate translation unit). Owns the evidence-monitor snapshot
 * push (refresh_evidence_monitor), the belief-timeseries pruning (prune_dead_series), and the dashboard
 * window-geometry persistence (restore/save_dashboard_geometry). The plots themselves are members of
 * SpecificWorker, fed from compute(); these methods manage their lifecycle and the monitor view.
 */

#include "specificworker.h"
#include "cabinet_geometry.h"   // rc::geom::belief_uncertainty

#include <QByteArray>
#include <QSettings>
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
        x.vacate     = inst.ai2_belief.last_span_pts();
        x.coverage   = inst.ai2_belief.last_lidar_rays();
        x.L          = inst.existence.logodds();
        x.p          = inst.existence.p_exists();
        x.ex_locc    = inst.dbg_ex_lidar_occ;  x.ex_lfree = inst.dbg_ex_lidar_free;  x.ex_ln    = inst.dbg_ex_lidar_n;
        x.ex_lfree_eff = inst.dbg_ex_lidar_free_eff;
        x.ex_socc    = inst.dbg_ex_sil_occ;    x.ex_sfree = inst.dbg_ex_sil_free;    x.ex_sndet = inst.dbg_ex_sil_ndet;
        x.ex_sfree_eff = inst.dbg_ex_sil_free_eff;
        x.streak     = inst.existence_remove_streak;
        const auto& s = inst.ai2_belief.state();
        x.w = s.L; x.h = s.d; x.H = s.z1;
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
    if (not custom_widget_)
        return;
    QSettings settings(QStringLiteral("RoboComp"), QStringLiteral("cabinet_concept"));
    const QByteArray geom = settings.value(QStringLiteral("DashboardWindow_geometry")).toByteArray();
    if (not geom.isEmpty())
        custom_widget_->restoreGeometry(geom);
    else
        custom_widget_->resize(560, 620);
}

void SpecificWorker::save_dashboard_geometry() const
{
    if (not custom_widget_)
        return;
    QSettings settings(QStringLiteral("RoboComp"), QStringLiteral("cabinet_concept"));
    settings.setValue(QStringLiteral("DashboardWindow_geometry"), custom_widget_->saveGeometry());
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
        if (ts_state_plot_)  { ts_state_plot_->remove_series(n + "_w");  ts_state_plot_->remove_series(n + "_h"); }
        if (ts_ce_plot_)     { ts_ce_plot_->remove_series(n + "_sW");    ts_ce_plot_->remove_series(n + "_sH"); }
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
        custom_widget_ = new Custom_widget("Cabinet Model — Free Energy, Coverage Deficit, Residuals & Dimensions (w,h)");
        custom_widget_->setWindowTitle(QStringLiteral("cabinet_concept — belief dashboard"));
        restore_dashboard_geometry();
        custom_widget_->show();

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

        // Inferred cabinet dimensions (w, h) — the size DOFs the stabiliser targets. Watch these to
        // confirm the belief has stopped jittering between fresh masks (flat = stable).
        ts_state_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
        ts_state_plot_->set_visible_window(60.f);
        series_layout->addWidget(ts_state_plot_);

        // (D) Counter-evidence accumulator S_w/S_h (the CUSUM gate is always on). Watch S charge on a
        // surprise and decay back (glitch absorbed) vs climb to the barrier (a real change unlocks).
        {
            ts_ce_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
            ts_ce_plot_->set_visible_window(60.f);
            series_layout->addWidget(ts_ce_plot_);
        }

        // Per-instance series are registered idempotently by the diagnostics feed (publish_cabinet_diagnostics)
        // every cycle, so pre-existing instances need no separate registration here.
    }

    // ── Evidence-consuming monitor — its OWN top-level window (per-instance snapshot + global counters) ──
    // Same lifecycle discipline as the dashboard above: plain QWidget (no QOpenGL), built on the main thread,
    // updated from compute() (which is the GUI thread here). Only HIDDEN on close, never deleted (compute()
    // keeps a raw pointer). Shows even with Agent.graph=false.
    evidence_monitor_ = new rc::EvidenceMonitor(QStringLiteral("cabinet_concept — evidence monitor"));
    evidence_monitor_->show();
}
