#pragma once

/*
 * common/dashboard/timeseries_plot.h  —  shared scrolling time-series plot widget
 *
 * SHARED dashboard widget for every concept agent's diagnostics dock (FE / dimensions / posterior σ panels).
 * Q_OBJECT QWidget; thread-safe add_point() (mutex), wall-clock X axis, auto-scaling Y, N named series.
 */

#include <QWidget>
#include <QPainter>
#include <QPainterPath>
#include <QElapsedTimer>
#include <deque>
#include <unordered_map>
#include <string>
#include <mutex>
#include <cmath>

namespace rc {

// Lightweight scrolling time-series plot: X = wall-clock seconds since the first point, Y auto-scales to the
// visible data, and any number of named series share the same axes.
//
//   plot->add_series("sdf_mse", Qt::red);
//   plot->add_point("sdf_mse", value);        // in the compute loop
class TimeSeriesPlot : public QWidget
{
    Q_OBJECT

public:
    explicit TimeSeriesPlot(QWidget* parent = nullptr);

    // Register a new series with the given display colour.
    // If avg_window > 0 a companion "<name>_avg" series is auto-created with a lighter colour and its running
    // average is updated on every add_point().
    void add_series(const std::string& name, QColor colour,
                    float line_width = 1.5f, int avg_window = 0);

    // Append a sample to the named series (call from any thread); updates the running-average companion if any.
    // A NON-FINITE value (NaN) is a GAP MARKER, not a reading: the line breaks there, autoscale ignores
    // it, the latest-sample dot skips it and the running average does not see it. Use it for a series
    // that is genuinely UNDEFINED for a stretch — joining across that stretch would draw a line through
    // time when the quantity did not exist, which reads as "unchanged" rather than "absent".
    void add_point(const std::string& name, float value);

    // Drop a series (and its running-average companion). No-op if absent.
    void remove_series(const std::string& name);

    // How many seconds of history to display (default 30).
    void set_visible_window(float seconds);

protected:
    void paintEvent(QPaintEvent* event) override;
    void timerEvent(QTimerEvent* event) override;

private:
    struct Sample { float t; float v; };  // elapsed seconds, value

    struct Series
    {
        std::string name;
        QColor colour{Qt::white};
        float line_width = 1.5f;
        std::deque<Sample> samples;

        // Running-average state (0 = disabled)
        int avg_window = 0;
        std::string avg_companion;        // key of the companion avg series
        std::deque<float> avg_ring;       // last N raw values
        float avg_sum = 0.f;
    };

    std::mutex mu_;
    std::unordered_map<std::string, Series> series_;
    QElapsedTimer clock_;
    float window_sec_ = 30.f;

    // Margins (pixels)
    static constexpr int kLeft = 55;
    static constexpr int kRight = 10;
    static constexpr int kTop = 20;
    static constexpr int kBottom = 22;

    void draw_axes(QPainter& p, float t_min, float t_max, float v_min, float v_max) const;
    void draw_legend(QPainter& p) const;
};

} // namespace rc
