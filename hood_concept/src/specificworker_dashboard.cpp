/*
 * specificworker_dashboard.cpp — SpecificWorker's standalone Qt dashboard + evidence-monitor methods.
 *
 * Split from specificworker.cpp (same class, separate translation unit). Owns the evidence-monitor snapshot
 * push (refresh_evidence_monitor), the belief-timeseries pruning (prune_dead_series), and the dashboard
 * window-geometry persistence (restore/save_dashboard_geometry). The plots themselves are members of
 * SpecificWorker, fed from compute(); these methods manage their lifecycle and the monitor view.
 */

#include "specificworker.h"

#include "../../common/dashboard/window_geometry.h"   // rc::dashboard:: (SHARED)
#include "../../common/dashboard/series_pruner.h"     // rc::dashboard:: (SHARED)
#include "hood_geometry.h"   // rc::geom::belief_uncertainty
#include "hood_dof.h"        // rc::kHoodDofs — names/units/σ* for the inspector rows

#include <QByteArray>
#include <QFont>
#include <QFontMetrics>
#include <QHBoxLayout>
#include <QPushButton>
#include <QSettings>
#include <QVBoxLayout>
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
// ─── Compact belief strip ────────────────────────────────────────────────────────────────────────

// One row per instance: the certainty channel (adequacy gap in nats, or ½·ln|Σ| when this agent
// publishes no σ*), p(existence), the FE surprise, and the node's birth stamp. The widget owns the
// history — this only pushes the current instant, on the same ~5 Hz tick as the panels above.
void SpecificWorker::refresh_belief_strip()
{
    if (not belief_strip_)
        return;   // headless (no dashboard built)

    std::vector<rc::BeliefStripRow> rows;
    rows.reserve(fitter_->instances().size());
    for (const auto& [id, inst] : fitter_->instances())
    {
        rc::BeliefStripRow r;
        r.node        = inst.node_name;
        r.surprise    = inst.fe_surprise;
        r.initialized = inst.ai2_initialized;
        r.p_exists    = inst.existence.p_exists();
        // Same REPORTED covariance the inspector and the NBV planner use, so the views cannot disagree.
        const auto S = inst.ai2_belief.covariance_reported();
        // `adequacy_gap_nats` returns 0 for a DOF table with no σ* anywhere — an empty sum — and 0 is
        // exactly the value that means "adequate". Ask first, and carry "no demand" as the -1 sentinel;
        // the strip then falls back to ½·ln|Σ| and says so in its heading.
        r.gap_nats = rc::any_sigma_star(rc::kHoodDofs)
                   ? rc::adequacy_gap_nats(rc::kHoodDofs, [&](std::size_t j) { return S(j, j); })
                   : -1.0f;

        // Fallback certainty channel: ½·ln det Σ via the Cholesky (Σ log L_ii), not log(det()) — a
        // covariance with centimetre σ has a determinant near the floor of float, where a direct
        // determinant is numerical noise.
        const auto llt = S.llt();
        if (llt.info() == Eigen::Success)
            r.logdet_nats = llt.matrixL().toDenseMatrix().diagonal().array().log().sum();

        // Birth from the node's own creation stamp, so `age` survives a dashboard opened long after the
        // instance was born. Absent ⇒ 0 ⇒ the widget falls back to when it first saw the row.
        if (const auto n = G->get_node(inst.node_id); n.has_value())
            r.birth_ms = G->get_attrib_by_name<timestamp_creation_att>(n.value()).value_or(0);

        rows.push_back(std::move(r));
    }
    belief_strip_->update_view(rows);
}

// Geometry persistence is SHARED (common/dashboard/window_geometry.h). The strip passes a max size: a
// restored geometry carries the window STATE, so a strip that was ever maximised comes back filling the
// screen — indistinguishable, to the user, from the big dashboard having opened itself.
void SpecificWorker::restore_strip_geometry()
{
    rc::dashboard::restore_window_geometry(strip_window_, QStringLiteral("hood_concept"), QStringLiteral("BeliefStripWindow"),
                                           QSize(520, 210), QSize(900, 420));
}

void SpecificWorker::save_strip_geometry() const
{
    rc::dashboard::save_window_geometry(strip_window_, QStringLiteral("hood_concept"), QStringLiteral("BeliefStripWindow"));
}


// ─── Belief inspector ────────────────────────────────────────────────────────────────────────────

// Build the per-instance BELIEF snapshot (state, Σ, door-facing modes, scalar gauges) and push it to the
// bottom panel. Called from refresh_evidence_monitor(), so it inherits that method's ~5 Hz gate and its
// single pass over the live instances. All reads are main-thread.
void SpecificWorker::refresh_belief_inspector()
{
    if (not belief_inspector_)
        return;   // headless: nothing was built
    const auto now = std::chrono::steady_clock::now();

    std::vector<rc::BeliefCard> cards;
    cards.reserve(fitter_->instances().size());
    for (const auto& [id, inst] : fitter_->instances())
    {
        rc::BeliefCard c;
        c.node = inst.node_name;

        // REPORTED covariance: σ_yaw carries the residual door-facing entropy, so an unresolved front shows
        // the honest orientation uncertainty. Same matrix the NBV planner scores.
        const auto  S = inst.ai2_belief.covariance_reported();
        const auto& s = inst.ai2_belief.state();
        const std::array<float, rc::HoodBelief::N> v = {s.cx, s.cy, s.H, s.w, s.h, s.yaw};
        for (int j = 0; j < rc::HoodBelief::N; ++j)
            c.dofs.push_back({rc::kHoodDofs[j].name, rc::kHoodDofs[j].unit, v[j],
                              std::sqrt(std::max(0.0f, S(j, j))), rc::kHoodDofs[j].sigma_star});

        // Row-major copy, filled explicitly: Eigen stores column-major, and while Σ is symmetric today an
        // implicit .data() copy would silently transpose if that ever stopped being true.
        constexpr int N = rc::HoodBelief::N;
        c.cov.resize(N * N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                c.cov[i * N + j] = S(i, j);

        // The genuine discrete ambiguity here is WHICH WAY THE DOOR FACES (resolved by appearance, not
        // geometry) — mode_posterior() is an inert stub for this agent, so it is deliberately not shown.
        const auto fp = inst.ai2_belief.front_posterior();
        static const char* kFrontLabels[4] = {"cur", "+90", "180", "-90"};
        for (int k = 0; k < 4; ++k)
            c.modes.push_back({"front", kFrontLabels[k], fp[k]});

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


// No max size here: the full dashboard is one the user may legitimately maximise, so its saved state is
// honoured as-is. That asymmetry is the whole reason max_restored is a parameter and not a constant.
void SpecificWorker::restore_dashboard_geometry()
{
    rc::dashboard::restore_window_geometry(dashboard_window_, QStringLiteral("hood_concept"), QStringLiteral("DashboardWindow"),
                                           QSize(1180, 900));
}

void SpecificWorker::save_dashboard_geometry() const
{
    rc::dashboard::save_window_geometry(dashboard_window_, QStringLiteral("hood_concept"), QStringLiteral("DashboardWindow"));
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
                                     live, ts_known_hoods_);
}
// Build ONE combined top-level window — the belief timeseries dashboard + the evidence monitor stacked in a
// vertical splitter — on the main thread from initialize(). cfg_.show_dashboard=false builds NOTHING (headless):
// the widget pointers stay null and the compute feed (publish_hood_diagnostics / refresh_evidence_monitor /
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
        custom_widget_ = new Custom_widget("Hood — Free Energy, Surprise, Belief Uncertainty & Residuals");

        // Create plot inside frame_series
        auto* series_layout = new QVBoxLayout(custom_widget_->frame_series);
        series_layout->setContentsMargins(0, 0, 0, 0);
        custom_widget_->frame_series->setLayout(series_layout);

        ts_plot_ = new rc::TimeSeriesPlot(custom_widget_->frame_series);
        ts_plot_->set_visible_window(60.f);
        series_layout->addWidget(ts_plot_, 1);

        // FE SURPRISE (attention signal) on its own panel — it lives on a much smaller scale (~0–1) than the FE
        // (~2–8), so it needs the full panel height to be readable. Spikes when a hood moves, decays as the fit
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
        // structure lives — and the door-facing mode posterior. Stretch 2: it is the panel you read.
        belief_inspector_ = new rc::BeliefInspector(QStringLiteral("belief inspector"),
                                                    custom_widget_->frame_series);
        series_layout->addWidget(belief_inspector_, 2);

        // Per-instance series are registered idempotently by the diagnostics feed (publish_hood_diagnostics)
        // every cycle, so pre-existing instances need no separate registration here.
    }

    // ── Evidence-consuming monitor (goes into the UPPER splitter pane) — per-instance snapshot + counters ──
    // Same lifecycle: plain QWidget, main-thread, updated from compute(). A CHILD of the combined window now.
    evidence_monitor_ = new rc::EvidenceMonitor(QStringLiteral("hood_concept — evidence monitor"));

    // ── Combined window: evidence monitor (top) over the belief plots (bottom) in a resizable splitter ──
    dashboard_window_ = new QWidget;
    dashboard_window_->setWindowTitle(QStringLiteral("hood_concept — dashboard"));
    auto* outer = new QVBoxLayout(dashboard_window_);
    outer->setContentsMargins(0, 0, 0, 0);
    // No splitter: the counter strip is two lines of text with nothing to resize, so it simply takes
    // its natural height and the plots + inspector get everything else.
    outer->addWidget(evidence_monitor_, 0);   // section 1 — natural height
    outer->addWidget(custom_widget_, 1);      // sections 2 + 3 — all remaining space

    // NOT shown at startup: the strip below is the standing display and this is its drill-down,
    // opened by the strip's "details ▸" button. Geometry is still restored, so the first click
    // puts it back where you left it. (Want it up from the start? add `->show()`.)
    restore_dashboard_geometry();

    // ── Compact belief strip — a SEPARATE, small top-level window ──────────────────────────────────
    // Not another panel inside the big window: the point is a display small enough to keep in a corner
    // while the 1180×900 dashboard stays closed until something looks wrong. One row per instance, each
    // row a 60 s trace of the certainty channel + p(existence) + FE surprise.
    strip_window_ = new QWidget;
    strip_window_->setWindowTitle(QStringLiteral("hoods — beliefs"));
    auto* strip_layout = new QVBoxLayout(strip_window_);
    strip_layout->setContentsMargins(0, 0, 0, 0);
    belief_strip_ = new rc::BeliefStrip(QStringLiteral("hoods"), strip_window_);
    belief_strip_->set_visible_window(60.f);
    strip_layout->addWidget(belief_strip_, 1);

    // "details ▸" — reveal the big dashboard on demand. A lambda connect needs no Q_OBJECT on either
    // side, so the moc-free widget pattern is preserved; strip_window_ is the context object, so the
    // connection dies with the window. show() alone is not enough for a minimised or buried window.
    {
        auto* bar = new QHBoxLayout;
        bar->setContentsMargins(4, 0, 4, 3);
        bar->addStretch(1);
        auto* details = new QPushButton(QStringLiteral("details ▸"), strip_window_);
        details->setToolTip(QStringLiteral("show / hide the full dashboard: evidence counters, FE/surprise/Σ "
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
