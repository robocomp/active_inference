/*
 * common/dashboard/concept_dashboard.h — the concept agent's two windows, built once. SHARED, header-only.
 *
 * Every object-concept agent shows the same thing in the same shape: a small BELIEF STRIP that stays open in
 * a corner (one row per belief unit, 60 s traces), and behind a "details ▸" button a full DASHBOARD — the
 * evidence-counter strip over four time-series panels and the per-DOF belief inspector. That was ~110 lines
 * of Qt written four times. refrigerator's and hood's were BYTE-IDENTICAL after normalising the object noun.
 *
 * ★TWO REAL DEFECTS THE COPIES HAD ALREADY SPLIT ON, both now fixed for everyone:
 *
 *   1. THE DRILL-DOWN COULD COME UP BY ITSELF. restoreGeometry() restores the window STATE as well as the
 *      rectangle, so "we simply never called show()" is not enough to keep the dashboard down — a session
 *      that ended with it visible brings it back at startup. table and cabinet call hide() explicitly;
 *      refrigerator and hood do not. Same failure as the strip's maximised-state clamp in window_geometry.h,
 *      found in the same file a day apart.
 *   2. CABINET HAS NO HEADLESS MODE. Its three siblings return early on `show_dashboard = false` and leave
 *      every widget pointer null (the compute feed already no-ops on null). cabinet has no such config key
 *      at all and builds GUI windows unconditionally. Here the guard is the caller's `enabled` flag, so an
 *      agent that forgets it gets a compile-time argument rather than a silent window.
 *
 * HEADER-ONLY ON PURPOSE. custom_widget.h / belief_inspector.h / belief_strip.h / evidence_monitor.h are
 * header-only Q_OBJECT classes mocced by the CONSUMER's AUTOMOC (see common/CMakeLists.txt: giving them a
 * second moc TU produces duplicate staticMetaObject symbols at link). Compiling this into ai_common_ui would
 * do exactly that, so it stays a header the agent includes.
 *
 * Qt only. No DSR, no agent types, no belief types — what goes IN the panels is fed separately.
 */

#pragma once

#include <QFont>
#include <QFontMetrics>
#include <QHBoxLayout>
#include <QPushButton>
#include <QSize>
#include <QString>
#include <QVBoxLayout>
#include <QWidget>
#include <Qt>

#include "belief_inspector.h"
#include "belief_strip.h"
#include "custom_widget.h"
#include "evidence_monitor.h"
#include "timeseries_plot.h"
#include "window_geometry.h"

namespace rc::dashboard
{

// What is genuinely per-agent: four strings and one size. Everything else is the same display.
struct BuildSpec
{
    QString agent;            // "refrigerator_concept" — QSettings scope and window titles
    QString series_title;     // headline over the time-series panels
    QString strip_title;      // belief-strip window title
    QString strip_label;      // belief-strip heading (usually the plural noun)
    QSize   dashboard_size{1180, 900};   // fallback when no geometry was ever saved
    float   visible_window_s = 60.0f;    // trace length on every panel
};

// The widgets the agent keeps raw pointers to. All null when built with enabled = false.
struct Widgets
{
    Custom_widget*       custom    = nullptr;
    rc::TimeSeriesPlot*  fe        = nullptr;   // free energy + baseline
    rc::TimeSeriesPlot*  surprise  = nullptr;   // FE surprise — its own panel, it lives on a much smaller scale
    rc::TimeSeriesPlot*  cov       = nullptr;
    rc::TimeSeriesPlot*  res       = nullptr;
    rc::BeliefInspector* inspector = nullptr;
    rc::EvidenceMonitor* evidence  = nullptr;
    QWidget*             dashboard_window = nullptr;
    QWidget*             strip_window     = nullptr;
    rc::BeliefStrip*     strip            = nullptr;
};

// Build both windows on the MAIN thread. `enabled = false` builds nothing and returns all-null (headless).
//
// The strip is shown; the dashboard is restored but explicitly hidden — it is a drill-down, and it must
// start down regardless of what state the last session saved. Neither window is WA_DeleteOnClose: compute()
// holds these raw pointers, so closing must only hide.
inline Widgets build(const BuildSpec& spec, bool enabled)
{
    Widgets w;
    if (not enabled)
        return w;   // headless: no GUI windows; the compute feed no-ops on the null pointers

    // ── Time-series panels + belief inspector, inside the combined window's lower area ──────────────
    w.custom = new Custom_widget(spec.series_title);
    auto* series_layout = new QVBoxLayout(w.custom->frame_series);
    series_layout->setContentsMargins(0, 0, 0, 0);
    w.custom->frame_series->setLayout(series_layout);

    const auto add_plot = [&](rc::TimeSeriesPlot*& p)
    {
        p = new rc::TimeSeriesPlot(w.custom->frame_series);
        p->set_visible_window(spec.visible_window_s);
        series_layout->addWidget(p, 1);
    };
    add_plot(w.fe);
    add_plot(w.surprise);   // separate panel: FE surprise sits at ~0-1 against an FE of ~2-8 and would be flat
    add_plot(w.cov);
    add_plot(w.res);

    // The panel that replaced the per-DOF sigma traces: every state DOF (value, sigma, the consumer's demand
    // sigma*, the remaining adequacy gap in nats), Sigma as a correlation heatmap — which is where the
    // structure lives — and the discrete-mode posteriors. Stretch 2: it is the panel you read.
    w.inspector = new rc::BeliefInspector(QStringLiteral("belief inspector"), w.custom->frame_series);
    series_layout->addWidget(w.inspector, 2);

    // ── Evidence-consuming monitor: per-instance snapshot + counters ────────────────────────────────
    w.evidence = new rc::EvidenceMonitor(spec.agent + QStringLiteral(" — evidence monitor"));

    // ── Combined window: counters on top, plots + inspector below ───────────────────────────────────
    // No splitter: the counter strip is two lines of text with nothing to resize, so it takes its natural
    // height and everything else goes to the plots.
    w.dashboard_window = new QWidget;
    w.dashboard_window->setWindowTitle(spec.agent + QStringLiteral(" — dashboard"));
    auto* outer = new QVBoxLayout(w.dashboard_window);
    outer->setContentsMargins(0, 0, 0, 0);
    outer->addWidget(w.evidence, 0);
    outer->addWidget(w.custom, 1);

    restore_window_geometry(w.dashboard_window, spec.agent, QStringLiteral("DashboardWindow"),
                            spec.dashboard_size);
    // ★EXPLICITLY hidden. restoreGeometry() restores the window STATE too, so "we never called show()" does
    // not keep it down — a session that ended with it up brings it back. The drill-down must start down.
    w.dashboard_window->hide();

    // ── Compact belief strip — a SEPARATE, small top-level window ───────────────────────────────────
    // Not another panel inside the big one: the point is a display small enough to keep in a corner while
    // the dashboard stays closed until something looks wrong.
    w.strip_window = new QWidget;
    w.strip_window->setWindowTitle(spec.strip_title);
    auto* strip_layout = new QVBoxLayout(w.strip_window);
    strip_layout->setContentsMargins(0, 0, 0, 0);
    w.strip = new rc::BeliefStrip(spec.strip_label, w.strip_window);
    w.strip->set_visible_window(spec.visible_window_s);
    strip_layout->addWidget(w.strip, 1);

    // "details ▸" — reveal the dashboard on demand. A lambda connect needs no Q_OBJECT on either side, so
    // the moc-free widget pattern is preserved; strip_window is the context object, so the connection dies
    // with the window rather than outliving it.
    {
        auto* bar = new QHBoxLayout;
        bar->setContentsMargins(4, 0, 4, 3);
        bar->addStretch(1);
        auto* details = new QPushButton(QStringLiteral("details ▸"), w.strip_window);
        details->setToolTip(QStringLiteral("show / hide the full dashboard: evidence counters, FE/surprise/Σ "
                                           "time series, and the per-DOF belief inspector"));
        QFont bf = details->font(); bf.setPointSizeF(bf.pointSizeF() - 1.0); details->setFont(bf);
        details->setFixedHeight(QFontMetrics(bf).height() + 8);
        QWidget* dash = w.dashboard_window;
        QObject::connect(details, &QPushButton::clicked, w.strip_window, [dash, details]()
        {
            if (not dash) return;
            // TOGGLE: the button the dashboard came out of is the button it goes back into. A drill-down
            // that can only be OPENED is one that stays open — it sits on top of everything and the only way
            // back to a clear screen is to hunt it down in the window list and minimise it by hand.
            // ★A MINIMISED window counts as PUT AWAY, not as up: otherwise the first click after minimising
            // would "hide" an invisible window and it would take two clicks to see it again.
            const bool up = dash->isVisible() and not dash->isMinimized();
            if (up)
                dash->hide();
            else
            {
                dash->show();
                dash->setWindowState((dash->windowState() & ~Qt::WindowMinimized) | Qt::WindowActive);
                dash->raise();
                dash->activateWindow();
            }
            // The label says what the NEXT click does. It can go stale if the window is closed from its own
            // title bar; the state is re-read above on every click, so the behaviour never is.
            details->setText(up ? QStringLiteral("details ▸") : QStringLiteral("◂ hide"));
        });
        bar->addWidget(details, 0);
        strip_layout->addLayout(bar, 0);
    }

    // The strip is small ON PURPOSE and its restore is capped — see window_geometry.h.
    restore_window_geometry(w.strip_window, spec.agent, QStringLiteral("BeliefStripWindow"),
                            QSize(520, 210), QSize(900, 420));
    w.strip_window->show();
    return w;
}

}  // namespace rc::dashboard
