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
        c.s.remove_streak = static_cast<int>(inst.existence_debounce.streak);
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
        series_layout->addWidget(ts_plot_, 1);

        // FE SURPRISE (attention signal) on its own panel — it lives on a much smaller scale (~0–1) than the FE
        // (~2–8), so it needs the full panel height to be readable. Spikes when a cabinet moves, decays as the fit
        // re-converges. See the belief/fitter plumbing (inst.fe_surprise).
        ts_surprise_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
        ts_surprise_plot_->set_visible_window(60.f);
        series_layout->addWidget(ts_surprise_plot_, 1);

        ts_cov_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
        ts_cov_plot_->set_visible_window(60.f);
        series_layout->addWidget(ts_cov_plot_, 1);

        ts_res_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
        ts_res_plot_->set_visible_window(60.f);
        series_layout->addWidget(ts_res_plot_, 1);


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
    // No splitter: the counter strip is two lines of text with nothing to resize, so it simply takes
    // its natural height and the plots + inspector get everything else.
    outer->addWidget(evidence_monitor_, 0);   // section 1 — natural height
    outer->addWidget(custom_widget_, 1);      // sections 2 + 3 — all remaining space

    // NOT shown at startup: the compact strip below is the standing display, and this window is the
    // drill-down you ask for with its "details ▸" button. Geometry is still restored here, so the first
    // click puts it back exactly where you last left it. (Want it up from the start again? add ->show().)
    restore_dashboard_geometry();
    // EXPLICITLY hidden: restoreGeometry() also restores the window STATE, so relying on
    // "we never called show()" is fragile. The drill-down must start down.
    if (dashboard_window_) dashboard_window_->hide();

    // ── Compact belief strip — a SEPARATE, small top-level window ──────────────────────────────────
    // One row per cabinet, each row a 60 s trace of the adequacy gap + p(exists) + FE surprise. Small on
    // purpose: it is meant to sit in a corner and stay open, with the big dashboard above as the
    // drill-down. Mirrors table_concept.
    strip_window_ = new QWidget;
    strip_window_->setWindowTitle(QStringLiteral("cabinet_concept — beliefs"));
    auto* strip_layout = new QVBoxLayout(strip_window_);
    strip_layout->setContentsMargins(0, 0, 0, 0);
    belief_strip_ = new rc::BeliefStrip(QStringLiteral("cabinets"), strip_window_);
    belief_strip_->set_visible_window(60.f);
    strip_layout->addWidget(belief_strip_, 1);

    // "details ▸" — reveal the big dashboard on demand. A lambda connect needs no Q_OBJECT on either side,
    // so the moc-free widget pattern is preserved; strip_window_ is the context object, so the connection
    // dies with the window rather than outliving it. show() alone is not enough: a window the user
    // minimised or buried needs the un-minimise + raise + activate as well.
    {
        auto* bar = new QHBoxLayout;
        bar->setContentsMargins(4, 0, 4, 3);
        bar->addStretch(1);
        auto* details = new QPushButton(QStringLiteral("details \u25B8"), strip_window_);
        details->setToolTip(QStringLiteral("show / hide the full dashboard: evidence counters, FE/surprise/\u03A3 "
                                           "time series, and the per-DOF belief inspector"));
        QFont bf = details->font(); bf.setPointSizeF(bf.pointSizeF() - 1.0); details->setFont(bf);
        details->setFixedHeight(QFontMetrics(bf).height() + 8);
        QObject::connect(details, &QPushButton::clicked, strip_window_, [this, details]()
        {
            if (not dashboard_window_) return;
            // TOGGLE: the button the dashboard came out of is the button it goes back into. A drill-down
            // that can only be OPENED is one that stays open — the 1180x900 window sits on top of
            // everything and the only way back to a clear screen is to hunt it down in the window list
            // and minimise it by hand.
            // ★A MINIMISED window counts as PUT AWAY, not as up: otherwise the first click after
            // minimising would "hide" an invisible window and it would take two clicks to see it again.
            const bool up = dashboard_window_->isVisible() and not dashboard_window_->isMinimized();
            if (up)
                dashboard_window_->hide();
            else
            {
                dashboard_window_->show();
                dashboard_window_->setWindowState((dashboard_window_->windowState() & ~Qt::WindowMinimized)
                                                  | Qt::WindowActive);
                dashboard_window_->raise();
                dashboard_window_->activateWindow();
            }
            // The label says what the NEXT click does. It can go stale if the window is closed from its
            // own title bar; the state is re-read above on every click, so the behaviour never is.
            details->setText(up ? QStringLiteral("details ▸") : QStringLiteral("◂ hide"));
        });
        bar->addWidget(details, 0);
        strip_layout->addLayout(bar, 0);
    }

    restore_strip_geometry();
    strip_window_->show();
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
            r.gap_nats = rc::any_sigma_star(rc::kWallRunDofs)
                       ? rc::adequacy_gap_nats(rc::kWallRunDofs, [&](std::size_t j) { return S(j, j); })
                       : -1.0f;
            const auto llt = S.llt();
            if (llt.info() == Eigen::Success)
                r.logdet_nats = llt.matrixL().toDenseMatrix().diagonal().array().log().sum();
            rows.push_back(std::move(r));
        }
        if (const auto* isl = kitchen_mgr_.island())   // the free-standing peninsula has no cell
        {
            rc::BeliefStripRow r;
            r.node        = "island";
            r.model       = "wall-run island";
            r.p_exists    = 1.0f / (1.0f + std::exp(-kitchen_mgr_.island_existence()));
            r.initialized = true;
            const auto& S = isl->covariance();
            r.gap_nats = rc::any_sigma_star(rc::kWallRunDofs)
                       ? rc::adequacy_gap_nats(rc::kWallRunDofs, [&](std::size_t j) { return S(j, j); })
                       : -1.0f;
            const auto llt = S.llt();
            if (llt.info() == Eigen::Success)
                r.logdet_nats = llt.matrixL().toDenseMatrix().diagonal().array().log().sum();
            rows.push_back(std::move(r));
        }
        belief_strip_->update_view(rows);
        return;
    }

    // Per-instance model (kitchen_model off): one row per cabinet instance.
    rows.reserve(fitter_->instances().size());
    for (const auto& [id, inst] : fitter_->instances())
    {
        rc::BeliefStripRow r;
        r.node        = inst.node_name;
        r.p_exists    = inst.existence.p_exists();
        r.surprise    = inst.fe_surprise;
        r.initialized = inst.ai2_initialized;

        // The same REPORTED covariance the inspector and the NBV planner use, so the strip cannot
        // disagree with either.
        const auto S = inst.ai2_belief.covariance_reported();
        r.gap_nats = rc::any_sigma_star(rc::kCabinetDofs)
                   ? rc::adequacy_gap_nats(rc::kCabinetDofs, [&](std::size_t j) { return S(j, j); })
                   : -1.0f;

        // Fallback channel: ½·ln det Σ via the Cholesky (Σ log L_ii), not log(det()) — a covariance with
        // centimetre σ has a determinant where a direct determinant is numerical noise.
        const auto llt = S.llt();
        if (llt.info() == Eigen::Success)
            r.logdet_nats = llt.matrixL().toDenseMatrix().diagonal().array().log().sum();

        if (const auto n = G->get_node(inst.node_id); n.has_value())
            r.birth_ms = G->get_attrib_by_name<timestamp_creation_att>(n.value()).value_or(0);

        rows.push_back(std::move(r));
    }
    belief_strip_->update_view(rows);
}

void SpecificWorker::restore_strip_geometry()
{
    if (not strip_window_)
        return;
    QSettings settings(QStringLiteral("RoboComp"), QStringLiteral("cabinet_concept"));
    const QByteArray geom = settings.value(QStringLiteral("BeliefStripWindow_geometry")).toByteArray();
    if (not geom.isEmpty())
    {
        strip_window_->restoreGeometry(geom);
        // ★A restored geometry carries the window STATE too. If the strip was ever maximised its saved
        // state brings it back filling the screen — indistinguishable, to the user, from the big dashboard
        // having opened itself. The strip is meant to sit in a corner, so refuse those states and cap the
        // size; the position is still honoured.
        strip_window_->setWindowState(strip_window_->windowState()
                                      & ~(Qt::WindowMaximized | Qt::WindowFullScreen));
        const QSize sz = strip_window_->size();
        strip_window_->resize(std::min(sz.width(), 900), std::min(sz.height(), 420));
    }
    else
        strip_window_->resize(520, 210);   // small ON PURPOSE — it sits in a corner, always open
}

void SpecificWorker::save_strip_geometry() const
{
    if (not strip_window_)
        return;
    QSettings settings(QStringLiteral("RoboComp"), QStringLiteral("cabinet_concept"));
    settings.setValue(QStringLiteral("BeliefStripWindow_geometry"), strip_window_->saveGeometry());
    settings.sync();
}
