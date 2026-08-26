/*  calibration_viewer.h — a window showing how the self-calibration is going.
 *
 *  One row per parameter: its current value with 1-sigma, whether this window actually TAUGHT it,
 *  and a trace of how it has moved.
 *
 *  ★ THE "informed" LAMP IS THE POINT, not decoration. A parameter's VALUE cannot distinguish
 *  "converged" from "never asked": both look like a number that has stopped moving. Only the
 *  precision separates them, and both cases have been seen live — eps_yaw sat near zero with a
 *  WIDENING sigma through a turn-heavy run, and k_omega did the same through a straight-heavy one.
 *  Showing the value alone would have made each look like a settled measurement.
 */
#pragma once

#include "calibration_estimator.h"
#include "timeseries_plot.h"

#include <QDialog>
#include <QLabel>
#include <array>
#include <functional>

namespace rc
{
    class CalibrationViewer : public QDialog
    {
        Q_OBJECT
    public:
        explicit CalibrationViewer(QWidget* parent = nullptr);

        /// Called once per localiser cycle. Cheap when the window is hidden: the traces are only
        /// appended while visible, so an unopened viewer costs nothing but the call.
        /// What "Reset" does. Set by the owner; the window itself knows nothing about the estimator.
        void set_reset_handler(std::function<void()> h) { on_reset_ = std::move(h); }

        void update_values(const Eigen::Matrix<float, rc::calib::P_COUNT, 1>& value,
                           const Eigen::Matrix<float, rc::calib::P_COUNT, 1>& sigma,
                           int informed_mask, float condition, int episodes);

    private:
        struct Row
        {
            QLabel* name = nullptr;
            QLabel* value = nullptr;
            QLabel* lamp = nullptr;
            TimeSeriesPlot* plot = nullptr;
            float scale = 1.f;          // display scaling (rad -> deg, fraction -> %)
            const char* unit = "";
            // What this parameter is and what motion identifies it — shown on hover over the name,
            // the value and the lamp. Static string literals from the spec table; not owned.
            const char* why = "";
        };
        std::array<Row, rc::calib::P_COUNT> rows_{};
        QLabel* summary_ = nullptr;
        std::function<void()> on_reset_;
    };
}
