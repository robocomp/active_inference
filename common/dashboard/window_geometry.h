/*
 * common/dashboard/window_geometry.h — persist/restore a standalone dashboard window. SHARED, header-only.
 *
 * The generated save_window_settings() only covers the QMainWindow(s) in `windows`; every concept agent's
 * dashboard and belief strip are separate top-level widgets, so each carries its own QSettings entry. That
 * was four copies of two function pairs, differing in the agent name string and the fallback size.
 *
 * ★WHAT THE COPIES HAD ALREADY DRIFTED ON. cabinet and table clamp a RESTORED strip geometry; hood and
 * refrigerator do not. A restored geometry carries the window STATE, so a strip that was ever maximised
 * comes back filling the screen — indistinguishable, to the user, from the big dashboard having opened
 * itself. Two agents fixed that and two kept the bug. Per the UNION rule the clamp is applied for
 * everyone, expressed as an optional max size: pass one and the restore is capped, omit it and the saved
 * geometry is honoured as-is (what a full dashboard wants).
 *
 * Qt only. No DSR, no agent types.
 */

#pragma once

#include <QByteArray>
#include <QSettings>
#include <QSize>
#include <QString>
#include <QWidget>
#include <Qt>

namespace rc::dashboard
{

// Restore `w` from QSettings("RoboComp", <agent>)/<key>_geometry.
//
// `fallback` is the size to use when nothing was ever saved. `max_restored` (if valid) caps a RESTORED
// geometry and strips the maximised/fullscreen state — the position is still honoured. Use it for a window
// that is meant to stay small (the belief strip); leave it default for one the user may legitimately
// maximise (the dashboard).
inline void restore_window_geometry(QWidget* w, const QString& agent, const QString& key,
                                    QSize fallback, QSize max_restored = QSize())
{
    if (not w)
        return;
    QSettings settings(QStringLiteral("RoboComp"), agent);
    const QByteArray geom = settings.value(key + QStringLiteral("_geometry")).toByteArray();
    if (geom.isEmpty())
    {
        w->resize(fallback);
        return;
    }
    w->restoreGeometry(geom);
    if (not max_restored.isValid())
        return;
    w->setWindowState(w->windowState() & ~(Qt::WindowMaximized | Qt::WindowFullScreen));
    const QSize sz = w->size();
    w->resize(qMin(sz.width(), max_restored.width()), qMin(sz.height(), max_restored.height()));
}

inline void save_window_geometry(const QWidget* w, const QString& agent, const QString& key)
{
    if (not w)
        return;
    QSettings settings(QStringLiteral("RoboComp"), agent);
    settings.setValue(key + QStringLiteral("_geometry"), w->saveGeometry());
    settings.sync();
}

}  // namespace rc::dashboard
