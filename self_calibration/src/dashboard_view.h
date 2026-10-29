/*
 * dashboard_view.h -- the Qt face of the self_calibration dashboard: one QPainter widget over a DashboardModel.
 * DISPLAY ONLY, main thread. No Q_OBJECT (no signals/slots), so no moc needed. Colours come from the palette so the
 * window follows the desktop theme; the two fragment labels use a fixed blue / orange pair (distinguishable in both).
 *
 * Information bars are on a LOG10 axis [10^-3, 10^4] because accumulated information spans orders of magnitude
 * (data precision / prior precision grows from 0 with every straight or turn); the tick at 1 marks "data worth
 * exactly the prior". A value off the axis is drawn at the edge; no colour switches on any value.
 */
#pragma once

#include "dashboard_model.h"

#include <QFontMetrics>
#include <QPainter>
#include <QBrush>
#include <QPalette>
#include <QString>
#include <QWidget>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>


class DashboardView : public QWidget
{
public:
    explicit DashboardView(const sc::dash::DashboardModel& m, QWidget* parent = nullptr) : QWidget(parent), model_(m)
    {
        setMinimumSize(640, 430);
        setAutoFillBackground(true);
    }
    void set_streams(std::vector<sc::dash::StreamRow> s) { streams_ = std::move(s); }

protected:
    void paintEvent(QPaintEvent*) override
    {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        const QPalette& pal = palette();
        const QColor fg = pal.color(QPalette::WindowText), mid = pal.color(QPalette::Mid), base = pal.color(QPalette::Base);
        const QColor c_yaw(0x3b, 0x82, 0xc4), c_lever(0xe0, 0x82, 0x1e), c_stall(220, 60, 60, 60);
        const QFontMetrics fm(font());
        const int W = width(), M = 14, lh = fm.height() + 4;
        int y = M;

        auto heading = [&](const QString& t) { QFont f = font(); f.setBold(true); p.setFont(f); p.setPen(fg); p.drawText(M, y + fm.ascent(), t); p.setFont(font()); y += lh + 2; };

        // 1. counters
        heading(QStringLiteral("Fragments (selected pairs)"));
        using sc::dash::Label;
        p.setPen(fg);
        p.drawText(M, y + fm.ascent(), QStringLiteral("keyframes captured %1    dropped by stall %2").arg(model_.keyframes()).arg(model_.dropped_stalled()));
        y += lh;
        p.drawText(M, y + fm.ascent(), QStringLiteral("last 60 s %1  (yaw %2, lever %3)    total %4  (yaw %5, lever %6)")
                   .arg(model_.pairs_recent()).arg(model_.pairs_recent(Label::yaw)).arg(model_.pairs_recent(Label::lever))
                   .arg(model_.pairs_total()).arg(model_.pairs_total(Label::yaw)).arg(model_.pairs_total(Label::lever)));
        y += lh + 8;

        // 2. information per DoF
        heading(QStringLiteral("Information per DoF: data precision / prior precision (log axis, tick = 1)"));
        const int lab_w = fm.horizontalAdvance(QStringLiteral("Helios psi ")) + 4;
        const int bx = M + lab_w, bw = std::max(50, W - bx - M - 70), bh = lh;
        constexpr double lo = -3.0, hi = 4.0;
        auto xof = [&](double ratio) { const double l = ratio > 0.0 ? std::log10(ratio) : lo; return bx + bw * (std::clamp(l, lo, hi) - lo) / (hi - lo); };
        const auto r = model_.ratios();
        static const char* names[3] = {"Helios dx", "Helios dy", "Helios psi"};
        for (int i = 0; i < 3; ++i)
        {
            p.setPen(fg);
            p.drawText(M, y + fm.ascent(), names[i]);
            p.fillRect(QRectF(bx, y, bw, bh), base);
            p.setPen(mid); p.drawRect(QRectF(bx, y, bw, bh));
            p.fillRect(QRectF(bx, y + 2, xof(r[i]) - bx, bh - 4), i == 2 ? c_yaw : c_lever);
            const double x1 = xof(1.0);
            p.setPen(QPen(fg, 2)); p.drawLine(QPointF(x1, y - 2), QPointF(x1, y + bh + 2));
            p.setPen(fg);
            p.drawText(bx + bw + 6, y + fm.ascent() + 2, QString::number(r[i], 'g', 3));
            y += bh + 6;
        }
        y += 6;

        // 3. timeline
        heading(QStringLiteral("Fragment timeline (last 5 min); red = a stream stalled"));
        const int tx = M, tw = std::max(50, W - 2 * M), th = 3 * lh;
        const std::int64_t tend = model_.now_ms(), t0 = tend - model_.window_ms();
        auto tx_of = [&](std::int64_t t) { return tx + tw * double(std::clamp<std::int64_t>(t - t0, 0, model_.window_ms())) / double(model_.window_ms()); };
        p.fillRect(QRectF(tx, y, tw, th), base);
        for (const auto& s : model_.stalls())
            p.fillRect(QRectF(tx_of(s.t0_ms), y, std::max(2.0, tx_of(s.t1_ms) - tx_of(s.t0_ms)), th), c_stall);
        for (const auto& f : model_.fragments())
        {
            const bool yaw = f.label == Label::yaw;
            const double a = tx_of(f.t0_ms), b = tx_of(f.t1_ms);
            p.fillRect(QRectF(a, y + (yaw ? 4 : th / 2 + 2), std::max(2.0, b - a), th / 2 - 6), yaw ? c_yaw : c_lever);
        }
        p.setPen(mid); p.drawRect(QRectF(tx, y, tw, th));
        p.setPen(fg);
        p.drawText(tx + 4, y + fm.ascent() + 1, QStringLiteral("yaw"));
        p.drawText(tx + 4, y + th / 2 + fm.ascent() + 1, QStringLiteral("lever"));
        y += th + 4;
        p.drawText(tx, y + fm.ascent(), QStringLiteral("-5 min"));
        p.drawText(tx + tw - fm.horizontalAdvance(QStringLiteral("now")), y + fm.ascent(), QStringLiteral("now"));
        y += lh + 8;

        // 4. stream health
        heading(QStringLiteral("Streams"));
        if (streams_.empty())
        {
            p.setPen(fg); p.drawText(M, y + fm.ascent(), QStringLiteral("no stream up yet"));
        }
        int x = M;
        for (const auto& s : streams_)
        {
            using St = sc::dash::StreamStatus;
            const QString key = QString::fromStdString(s.key);
            const QString t = s.status == St::waiting ? key + QStringLiteral(" waiting")
                            : (s.show_rate ? QStringLiteral("%1 %2 Hz%3").arg(key).arg(s.rate_hz, 0, 'f', 1).arg(s.status == St::stalled ? " STALLED" : "")
                                           : QStringLiteral("%1%2").arg(key).arg(s.status == St::stalled ? " STALLED" : " ok"));
            const int w = fm.horizontalAdvance(t) + 16;
            if (x + w > W - M) { x = M; y += lh + 4; }
            p.setPen(mid);
            p.setBrush(s.status == St::stalled ? QBrush(QColor(220, 60, 60, 90)) : s.status == St::waiting ? QBrush(QColor(128, 128, 128, 70)) : QBrush(Qt::NoBrush));
            p.drawRoundedRect(QRectF(x, y, w - 6, lh), 4, 4);
            p.setPen(s.status == St::waiting ? mid : fg); p.drawText(x + 5, y + fm.ascent() + 2, t);
            x += w;
        }
    }

private:
    const sc::dash::DashboardModel& model_;
    std::vector<sc::dash::StreamRow> streams_;
};
