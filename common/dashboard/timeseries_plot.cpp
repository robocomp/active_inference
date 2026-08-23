/*
 * common/dashboard/timeseries_plot.cpp  —  shared scrolling time-series plot (implementation)
 *
 * SHARED dashboard widget across the concept agents: series bookkeeping (add/remove + running-average
 * companions), thread-safe sampling, and the auto-scaling paint (axes, stroked polylines, legend).
 */

#include "timeseries_plot.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <cstdio>

namespace rc {

// ─── Legend labels / fixed value ranges (anonymous helpers) ──────────────────────────────────────

namespace
{
// Short legend label from a series name (…_cov → "cov", …_fe → "FE", …_res → "res", else the raw name).
QString legend_label_for_series(const std::string& name)
{
    if (name == "cov_det_scaled" || (name.size() >= 4 && name.substr(name.size() - 4) == "_cov"))
        return "cov";
    if (name == "free_energy" || (name.size() >= 3 && name.substr(name.size() - 3) == "_fe"))
        return "FE";
    if (name.size() >= 4 && name.substr(name.size() - 4) == "_res")
        return "res";
    return QString::fromStdString(name);
}

std::optional<std::pair<float, float>> fixed_value_range_for_series(const std::string& /*name*/)
{
    // AUTOSCALE all panels. The old fixed ranges (FE 0–1.5, cov 0–100, res 0–20) were calibrated to stale scales:
    // the corrected free energy now sits at ~2–8 (pegged off the top of 1.5) and the covariance is < 1 (flat at
    // the bottom of 100). Auto-scaling to the visible data (with a centred fallback for flat series, below) shows
    // the real dynamics — including the FE baseline + surprise now on the FE panel.
    return std::nullopt;
}

}

// ─── Construction ────────────────────────────────────────────────────────────────────────────────

TimeSeriesPlot::TimeSeriesPlot(QWidget* parent)
    : QWidget(parent)
{
    setMinimumHeight(55);
    setAutoFillBackground(true);
    QPalette pal = palette();
    pal.setColor(QPalette::Window, QColor(255, 255, 255));
    setPalette(pal);

    clock_.start();
    startTimer(100);  // repaint at ~10 Hz
}

// ─── Series management ───────────────────────────────────────────────────────────────────────────


namespace
{
    /// A series whose last sample is NaN is GAPPED, not broken. Several traces here use NaN
    /// deliberately to mean "this channel has nothing to say about the current stretch" -- the
    /// controller's adv_meas/rot_meas while it is commanding, sigma when the limiter is out of the
    /// loop -- so the line breaks instead of holding a stale value across a hand-over. Printing that
    /// as "nan" in the legend makes a deliberate gap look like a fault, which is exactly the wrong
    /// reading: the value is absent, not wrong.
    inline QString legend_value_text(float v)
    { return std::isfinite(v) ? QString("%1").arg(v, 0, 'f', 4) : QString("--"); }
}
void TimeSeriesPlot::add_series(const std::string& name, QColor colour,
                                float line_width, int avg_window)
{
    std::lock_guard lk(mu_);
    if (series_.find(name) != series_.end())
        return;   // idempotent: keep the existing series (and its samples) on repeat calls
    auto& s = series_[name];
    s.name = name;
    s.colour = colour;
    s.line_width = line_width;
    s.avg_window = std::max(0, avg_window);

    if (s.avg_window > 0)
    {
        // Create a lighter, thinner companion series for the running average
        const std::string avg_name = name + "_avg";
        s.avg_companion = avg_name;
        auto& a = series_[avg_name];
        a.name = avg_name;
        a.colour = colour.lighter(160);
        a.line_width = line_width + 0.5f;
    }
}

void TimeSeriesPlot::remove_series(const std::string& name)
{
    std::lock_guard lk(mu_);
    if (auto it = series_.find(name); it != series_.end())
    {
        if (!it->second.avg_companion.empty())
            series_.erase(it->second.avg_companion);
        series_.erase(it);
    }
}

// ─── Sampling ────────────────────────────────────────────────────────────────────────────────────

void TimeSeriesPlot::add_point(const std::string& name, float value)
{
    std::lock_guard lk(mu_);
    auto it = series_.find(name);
    if (it == series_.end()) return;

    const float t = clock_.elapsed() / 1000.f;
    it->second.samples.push_back({t, value});

    // Trim old samples beyond 2× visible window
    const float cutoff = t - 2.f * window_sec_;
    auto& q = it->second.samples;
    while (!q.empty() && q.front().t < cutoff)
        q.pop_front();

    // Update running average companion if configured. A GAP marker is not a value: folding it into the
    // ring would poison the average with NaN for the whole window.
    auto& s = it->second;
    if (std::isfinite(value) && s.avg_window > 0 && !s.avg_companion.empty())
    {
        s.avg_ring.push_back(value);
        s.avg_sum += value;
        while (static_cast<int>(s.avg_ring.size()) > s.avg_window)
        {
            s.avg_sum -= s.avg_ring.front();
            s.avg_ring.pop_front();
        }
        const float avg = s.avg_sum / static_cast<float>(s.avg_ring.size());

        auto avg_it = series_.find(s.avg_companion);
        if (avg_it != series_.end())
        {
            avg_it->second.samples.push_back({t, avg});
            auto& aq = avg_it->second.samples;
            while (!aq.empty() && aq.front().t < cutoff)
                aq.pop_front();
        }
    }
}

void TimeSeriesPlot::set_visible_window(float seconds)
{
    window_sec_ = std::max(1.f, seconds);
}

// ─── Painting ────────────────────────────────────────────────────────────────────────────────────

void TimeSeriesPlot::timerEvent(QTimerEvent*)
{
    update();  // schedule repaint
}

void TimeSeriesPlot::paintEvent(QPaintEvent*)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    std::lock_guard lk(mu_);

    // Determine time range
    const float t_now = clock_.elapsed() / 1000.f;
    const float t_min = t_now - window_sec_;
    const float t_max = t_now;

    // Determine value range across all visible series.
    // Dedicated FE/cov/res panels get a stable range so flat or sparse data
    // still remain visible.
    std::optional<std::pair<float, float>> fixed_range;
    for (const auto& [key, s] : series_)
    {
        if (s.samples.empty()) continue;
        if (key.size() > 4 && key.substr(key.size() - 4) == "_avg") continue;
        if (const auto hint = fixed_value_range_for_series(s.name); hint.has_value())
        {
            fixed_range = hint;
            break;
        }
    }

    float v_min = std::numeric_limits<float>::max();
    float v_max = std::numeric_limits<float>::lowest();
    if (fixed_range.has_value())
    {
        v_min = fixed_range->first;
        v_max = fixed_range->second;
    }
    else
    {
        for (const auto& [key, s] : series_)
        {
            if (key.size() > 4 && key.substr(key.size() - 4) == "_avg") continue;
            for (const auto& pt : s.samples)
            {
                if (pt.t < t_min) continue;
                if (!std::isfinite(pt.v)) continue;   // a GAP marker carries no value — see add_point
                v_min = std::min(v_min, pt.v);
                v_max = std::max(v_max, pt.v);
            }
        }
        if (v_min == std::numeric_limits<float>::max())      // no visible data
        { v_min = 0.f; v_max = 1.f; }
        else if (v_min >= v_max)                             // flat / single value → centre it with a margin
        { const float c = v_min, m = std::max(0.5f, 0.1f * std::abs(c)); v_min = c - m; v_max = c + m; }
    }

    // Add 5% padding
    if (!fixed_range.has_value())
    {
        const float pad = (v_max - v_min) * 0.05f;
        v_min -= pad;
        v_max += pad;
    }

    draw_axes(p, t_min, t_max, v_min, v_max);

    // Plot area
    const float pw = static_cast<float>(width()  - kLeft - kRight);
    const float ph = static_cast<float>(height() - kTop  - kBottom);

    auto map_x = [&](float t) -> float { return kLeft + (t - t_min) / (t_max - t_min) * pw; };
    auto map_y = [&](float v) -> float { return kTop  + (1.f - (v - v_min) / (v_max - v_min)) * ph; };

    // Draw each series
    for (const auto& [_, s] : series_)
    {
        if (s.samples.empty()) continue;

        QPainterPath path;
        bool started = false;
        for (const auto& pt : s.samples)
        {
            if (pt.t < t_min) continue;
            // ★NON-FINITE = A GAP, NOT A VALUE. A series can be legitimately UNDEFINED for a stretch —
            // a quantity that only exists while some condition holds — and joining across that stretch
            // would draw a line through time when the thing was not there. Breaking the path says
            // "absent" where a straight segment would have said "unchanged".
            if (!std::isfinite(pt.v)) { started = false; continue; }
            const float sx = map_x(pt.t);
            const float sy = map_y(pt.v);
            if (!started) { path.moveTo(sx, sy); started = true; }
            else          { path.lineTo(sx, sy); }
        }
        p.setPen(QPen(s.colour, s.line_width));
        p.setBrush(Qt::NoBrush);   // stroke only — else the latest-sample dot's brush leaks in and
                                   // drawPath FILLS the (implicitly closed) polyline, filling peaks
        p.drawPath(path);

        // Mark the latest sample without drawing a full-width guide line. The latest FINITE one: a dot
        // at a gap marker has no y to sit at, and drawing it at the last known value would assert a
        // current reading for a series that currently has none.
        const Sample* last_finite = nullptr;
        for (auto it = s.samples.rbegin(); it != s.samples.rend(); ++it)
            if (std::isfinite(it->v) && it->t >= t_min) { last_finite = &*it; break; }
        if (last_finite == nullptr) continue;
        const auto& last = *last_finite;
        const float last_x = map_x(last.t);
        const float last_y = map_y(last.v);
        p.setPen(QPen(s.colour, 1.0f));
        p.setBrush(s.colour);
        p.drawEllipse(QPointF(last_x, last_y), 2.5, 2.5);
    }

    draw_legend(p);
}

void TimeSeriesPlot::draw_axes(QPainter& p, float t_min, float t_max,
                                float v_min, float v_max) const
{
    const float pw = static_cast<float>(width()  - kLeft - kRight);
    const float ph = static_cast<float>(height() - kTop  - kBottom);

    p.setPen(QPen(QColor(80, 80, 80), 1));
    // Y axis
    p.drawLine(kLeft, kTop, kLeft, kTop + static_cast<int>(ph));
    // X axis
    p.drawLine(kLeft, kTop + static_cast<int>(ph),
               kLeft + static_cast<int>(pw), kTop + static_cast<int>(ph));

    p.setFont(QFont("Monospace", 7));
    p.setPen(QColor(20, 20, 20));

    // Y tick labels (5 ticks)
    for (int i = 0; i <= 4; ++i)
    {
        const float frac = static_cast<float>(i) / 4.f;
        const float val = v_max - frac * (v_max - v_min);
        const int y = kTop + static_cast<int>(frac * ph);
        p.drawText(QRect(0, y - 7, kLeft - 4, 14), Qt::AlignRight | Qt::AlignVCenter,
                   QString::number(val, 'f', 3));
        // grid line
        p.setPen(QPen(QColor(200, 200, 200), 1, Qt::DotLine));
        p.drawLine(kLeft, y, kLeft + static_cast<int>(pw), y);
        p.setPen(QColor(20, 20, 20));
    }

    // X tick labels — show relative seconds
    for (int i = 0; i <= 4; ++i)
    {
        const float frac = static_cast<float>(i) / 4.f;
        const float t = t_min + frac * (t_max - t_min);
        const int x = kLeft + static_cast<int>(frac * pw);
        p.drawText(QRect(x - 20, kTop + static_cast<int>(ph) + 2, 40, 16),
                   Qt::AlignHCenter | Qt::AlignTop,
                   QString::number(t, 'f', 0) + "s");
    }
}

void TimeSeriesPlot::draw_legend(QPainter& p) const
{
    // Collect primary series (skip auto-generated "_avg" companions)
    struct Entry { std::string label; QColor colour; float last_val; };
    std::vector<Entry> entries;
    for (const auto& [key, s] : series_)
    {
        if (key.size() > 4 && key.substr(key.size() - 4) == "_avg") continue;
        float val = s.samples.empty() ? 0.f : s.samples.back().v;
        entries.push_back({s.name, s.colour, val});
    }
    if (entries.empty()) return;

    const QFont font("Monospace", 7);
    p.setFont(font);
    const QFontMetrics fm(font);

    constexpr int swatch = 10;
    constexpr int pad = 3;
    constexpr int row_h = 13;

    // Compute box width from longest label
    int max_text_w = 0;
    for (const auto& e : entries)
    {
        QString txt = legend_label_for_series(e.label) + QString(": %1").arg(legend_value_text(e.last_val));
        max_text_w = std::max(max_text_w, fm.horizontalAdvance(txt));
    }
    const int box_w = swatch + pad * 3 + max_text_w;
    const int box_h = pad * 2 + static_cast<int>(entries.size()) * row_h;

    const int bx = kLeft + 6;
    const int by = kTop + 2;

    // Semi-transparent background
    p.setPen(Qt::NoPen);
    p.setBrush(QColor(255, 255, 255, 220));
    p.drawRoundedRect(bx, by, box_w, box_h, 3, 3);

    p.setPen(QPen(QColor(180, 180, 180), 1));
    p.setBrush(Qt::NoBrush);
    p.drawRoundedRect(bx, by, box_w, box_h, 3, 3);

    int y = by + pad;
    for (const auto& e : entries)
    {
        // Colour swatch
        p.setPen(Qt::NoPen);
        p.setBrush(e.colour);
        p.drawRect(bx + pad, y + 2, swatch, swatch);

        // Label + value
        p.setPen(QColor(20, 20, 20));
        QString txt = legend_label_for_series(e.label) + QString(": %1").arg(legend_value_text(e.last_val));
        p.drawText(bx + pad + swatch + pad, y, max_text_w, row_h,
                   Qt::AlignLeft | Qt::AlignVCenter, txt);
        y += row_h;
    }
}

} // namespace rc
