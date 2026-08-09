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
#include "table_dof.h"        // rc::kTableDofs — names/units/σ* for the BeliefInspector rows

#include <QByteArray>
#include <QFont>
#include <QFontMetrics>
#include <QHBoxLayout>
#include <QPushButton>
#include <QSettings>
#include <QVBoxLayout>
#include <QColor>

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_set>
#include <vector>
#include <array>
#include <chrono>

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

// ─── Compact belief strip ────────────────────────────────────────────────────────────────────────

// One row per table: the adequacy gap (nats still missing before Σ meets the consumer's σ*), the
// existence probability, the FE surprise, and the node's birth stamp. The widget owns the history — this
// only pushes the current instant, on the same ~5 Hz tick as the two panels above.
void SpecificWorker::refresh_belief_strip()
{
    if (not belief_strip_)
        return;   // headless (cfg_.show_dashboard=false): nothing was built

    std::vector<rc::BeliefStripRow> rows;
    rows.reserve(fitter_->instances().size());
    for (const auto& [id, inst] : fitter_->instances())
    {
        rc::BeliefStripRow r;
        r.node        = inst.node_name;
        r.model       = inst.subtype;
        r.p_exists    = inst.existence.p_exists();
        r.surprise    = inst.fe_surprise;
        r.initialized = inst.ai2_initialized;

        // Same REPORTED covariance the inspector and the NBV planner use (its yaw entry carries the
        // discrete-mode entropy), so the strip cannot disagree with either.
        const auto S = inst.ai2_belief.covariance_reported();
        // `adequacy_gap_nats` returns 0 for a DOF table with no σ* anywhere — an empty sum — and 0 is
        // exactly the value that means "adequate". Ask first, and carry "no demand" as the -1 sentinel.
        r.gap_nats   = rc::any_sigma_star(rc::kTableDofs)
                     ? rc::adequacy_gap_nats(rc::kTableDofs, [&](std::size_t j) { return S(j, j); })
                     : -1.0f;

        // Fallback channel, computed even though the table publishes σ*: ½·ln det Σ via the Cholesky
        // (Σ log L_ii), not log(det()) — a 7-DOF covariance with centimetre σ has a determinant around
        // 1e-20, where a direct determinant is numerical noise.
        const auto llt = S.llt();
        if (llt.info() == Eigen::Success)
            r.logdet_nats = llt.matrixL().toDenseMatrix().diagonal().array().log().sum();

        // Birth from the node's own creation stamp, so `age` survives a dashboard opened long after the
        // table was born. Absent (an older node, or one this agent adopted) ⇒ 0 ⇒ the widget falls back
        // to when it first saw the row and says so by measuring from there.
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
    QSettings settings(QStringLiteral("RoboComp"), QStringLiteral("table_concept"));
    const QByteArray geom = settings.value(QStringLiteral("BeliefStripWindow_geometry")).toByteArray();
    if (not geom.isEmpty())
        strip_window_->restoreGeometry(geom);
    else
        strip_window_->resize(520, 210);   // small ON PURPOSE — it is meant to sit in a corner, always open
}

void SpecificWorker::save_strip_geometry() const
{
    if (not strip_window_)
        return;
    QSettings settings(QStringLiteral("RoboComp"), QStringLiteral("table_concept"));
    settings.setValue(QStringLiteral("BeliefStripWindow_geometry"), strip_window_->saveGeometry());
    settings.sync();
}

// ─── Belief inspector ────────────────────────────────────────────────────────────────────────────

// Build the per-instance BELIEF snapshot (state, Σ, discrete modes, scalar gauges) and push it to the
// bottom panel. Called from refresh_evidence_monitor(), so it inherits that method's ~5 Hz gate and its
// single pass over the live instances. All reads are main-thread.
void SpecificWorker::refresh_belief_inspector()
{
    if (not belief_inspector_)
        return;   // headless (cfg_.show_dashboard=false): nothing was built
    const auto now = std::chrono::steady_clock::now();

    std::vector<rc::BeliefCard> cards;
    cards.reserve(fitter_->instances().size());
    for (const auto& [id, inst] : fitter_->instances())
    {
        rc::BeliefCard c;
        c.node  = inst.node_name;
        c.model = inst.subtype;   // "square" | "round" — the free-energy shape model-selection winner

        // REPORTED covariance: the yaw entry carries the discrete-mode entropy, so a mode-ambiguous table
        // shows the honest σ_yaw rather than the within-mode one. Same matrix the NBV planner scores.
        const auto  S = inst.ai2_belief.covariance_reported();
        const auto& s = inst.ai2_belief.state();
        const std::array<float, rc::TableBelief::N> v = {s.cx, s.cy, s.H, s.w, s.h, s.yaw, s.t};
        for (int j = 0; j < rc::TableBelief::N; ++j)
            c.dofs.push_back({rc::kTableDofs[j].name, rc::kTableDofs[j].unit, v[j],
                              std::sqrt(std::max(0.0f, S(j, j))), rc::kTableDofs[j].sigma_star});

        // Row-major copy, filled explicitly: Eigen stores column-major, and while Σ is symmetric today an
        // implicit .data() copy would silently transpose if that ever stopped being true.
        c.cov.resize(rc::TableBelief::N * rc::TableBelief::N);
        for (int i = 0; i < rc::TableBelief::N; ++i)
            for (int j = 0; j < rc::TableBelief::N; ++j)
                c.cov[i * rc::TableBelief::N + j] = S(i, j);
        // Under the C2v symmetry quotient only Σ's DIAGONAL is mapped back to (w,h,yaw); the off-diagonals
        // stay in the (s,a1,a2) chart. Label the heatmap axes for what they are rather than mislabel them.
        if (rc::TableBeliefState::use_quotient)
            c.cov_labels = {"cx", "cy", "H", "s", "a1", "a2", "t"};

        const float p_swap  = inst.ai2_belief.mode_posterior();          // p(the w↔h swapped mode)
        c.modes.push_back({"w<->h", "keep", 1.0f - p_swap});
        c.modes.push_back({"w<->h", "swap", p_swap});
        const float p_round = 1.0f / (1.0f + std::exp(-inst.shape_evidence));   // accumulated log-Bayes factor
        c.modes.push_back({"shape", "round",  p_round});
        c.modes.push_back({"shape", "square", 1.0f - p_round});

        c.s.fe            = inst.dbg_energy;
        c.s.fe_baseline   = inst.fe_baseline;
        c.s.fe_surprise   = inst.fe_surprise;
        c.s.logodds       = inst.existence.logodds();
        c.s.p_exists      = inst.existence.p_exists();
        c.s.age_s         = inst.last_belief_touch.time_since_epoch().count() == 0
                          ? -1.0f
                          : std::chrono::duration<float>(now - inst.last_belief_touch).count();
        // The streak is fractional now (resolving LOOKS, not cycles); the card shows whole looks accrued.
        c.s.remove_streak = static_cast<int>(inst.existence_remove_streak);
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
        custom_widget_ = new Custom_widget("Table — Free Energy, Surprise, Belief Uncertainty & Residuals");

        // Create plot inside frame_series
        auto* series_layout = new QVBoxLayout(custom_widget_->frame_series);
        series_layout->setContentsMargins(0, 0, 0, 0);
        custom_widget_->frame_series->setLayout(series_layout);

        ts_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
        ts_plot_->set_visible_window(60.f);
        series_layout->addWidget(ts_plot_, 1);

        // FE SURPRISE (attention signal) on its own panel — it lives on a much smaller scale (~0–1) than the FE
        // (~2–8), so it needs the full panel height to be readable. Spikes when a table moves, decays as the fit
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
        // demand σ*, the remaining adequacy gap in nats), Σ as a correlation heatmap — which is where the
        // structure lives, e.g. w↔yaw going jointly unresolved on a grazing view — and the discrete-mode
        // posteriors (shape, w↔h flip). Stretch 2: it is the panel you actually read.
        belief_inspector_ = new rc::BeliefInspector(QStringLiteral("belief inspector"),
                                                    custom_widget_->frame_series);
        series_layout->addWidget(belief_inspector_, 2);

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
    // No splitter: the counter strip is two lines of text with nothing to resize, so it simply takes
    // its natural height and the plots + inspector get everything else.
    outer->addWidget(evidence_monitor_, 0);   // section 1 — natural height
    outer->addWidget(custom_widget_, 1);      // sections 2 + 3 — all remaining space

    // NOT shown at startup: the strip below is the standing display, and this one is the drill-down you
    // ask for with its "details ▸" button. Its geometry is still restored here, so the first click puts
    // it back exactly where you last left it. (Want it up from the start again? add `->show()`.)
    restore_dashboard_geometry();

    // ── Compact belief strip — a SEPARATE, small top-level window ──────────────────────────────────
    // Deliberately not another panel inside the big window: the point is a display small enough to keep
    // in a corner of the screen while the 1180×900 dashboard stays closed until something looks wrong.
    // One row per table, each row a 60 s trace of the adequacy gap + p(exists) + FE surprise.
    strip_window_ = new QWidget;
    strip_window_->setWindowTitle(QStringLiteral("table_concept — beliefs"));
    auto* strip_layout = new QVBoxLayout(strip_window_);
    strip_layout->setContentsMargins(0, 0, 0, 0);
    belief_strip_ = new rc::BeliefStrip(QStringLiteral("tables"), strip_window_);
    belief_strip_->set_visible_window(60.f);
    strip_layout->addWidget(belief_strip_, 1);

    // "details ▸" — reveal the big dashboard on demand. A lambda connect needs no Q_OBJECT on either
    // side, so the moc-free widget pattern is preserved; strip_window_ is the context object, so the
    // connection dies with the window rather than outliving it. show() is not enough on its own: a
    // window the user minimised or buried needs the un-minimise + raise + activate as well.
    {
        auto* bar = new QHBoxLayout;
        bar->setContentsMargins(4, 0, 4, 3);
        bar->addStretch(1);
        auto* details = new QPushButton(QStringLiteral("details ▸"), strip_window_);
        details->setToolTip(QStringLiteral("open the full dashboard: evidence counters, FE/surprise/Σ "
                                           "time series, and the per-DOF belief inspector"));
        QFont bf = details->font(); bf.setPointSizeF(bf.pointSizeF() - 1.0); details->setFont(bf);
        details->setFixedHeight(QFontMetrics(bf).height() + 8);
        QObject::connect(details, &QPushButton::clicked, strip_window_, [this]()
        {
            if (not dashboard_window_) return;
            dashboard_window_->show();
            dashboard_window_->setWindowState((dashboard_window_->windowState() & ~Qt::WindowMinimized)
                                              | Qt::WindowActive);
            dashboard_window_->raise();
            dashboard_window_->activateWindow();
        });
        bar->addWidget(details, 0);
        strip_layout->addLayout(bar, 0);
    }

    restore_strip_geometry();
    strip_window_->show();
}
