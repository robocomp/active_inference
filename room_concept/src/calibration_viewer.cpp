#include "calibration_viewer.h"

#include <QHBoxLayout>
#include <QSizePolicy>
#include <QVBoxLayout>
#include <cmath>

namespace rc
{
    CalibrationViewer::CalibrationViewer(QWidget* parent) : QDialog(parent)
    {
        setWindowTitle("Self-calibration");
        resize(900, 620);   // tall enough that four full-width traces are each readable
        auto* outer = new QVBoxLayout(this);

        summary_ = new QLabel(this);
        summary_->setStyleSheet("font-family: monospace; font-size: 11px; color: #cfd3d6;");
        outer->addWidget(summary_);

        // ── LAYOUT: THE TRACE GETS THE WHOLE WIDTH ────────────────────────────────────────────────
        // A grid with the labels in their own columns leaves the plot only whatever is left over on
        // the right, which is the narrowest part of the window and the part carrying the actual
        // information. So each parameter is a SECTION instead: a thin header row of labels, then a
        // full-width trace under it that takes all the vertical stretch. Four traces reading edge to
        // edge is the whole reason this window exists rather than another strip in the side panel.
        const struct { const char* name; float scale; const char* unit; QColor col; }
        spec[rc::calib::P_COUNT] = {
            { "translation scale", 100.f,               "%",     QColor(41, 128, 185) },
            { "mount yaw",         float(180.0 / M_PI), "deg",   QColor(241, 196, 15) },
            { "gyro scale",        100.f,               "%",     QColor(192, 57, 43)  },
            { "gyro bias",         float(180.0 / M_PI), "deg/s", QColor(39, 174, 96)  },
        };

        for (int i = 0; i < rc::calib::P_COUNT; ++i)
        {
            auto& r = rows_[i];
            r.scale = spec[i].scale;
            r.unit  = spec[i].unit;

            auto* header = new QHBoxLayout();
            r.name = new QLabel(spec[i].name, this);
            r.name->setStyleSheet(QString("font-size: 12px; font-weight: bold; color: %1;")
                                      .arg(spec[i].col.name()));
            r.value = new QLabel("--", this);
            r.value->setStyleSheet("font-family: monospace; font-size: 12px; color: #e6e9ea;");
            r.lamp = new QLabel(this);
            header->addWidget(r.name);
            header->addSpacing(12);
            header->addWidget(r.value);
            header->addStretch(1);          // pushes the lamp to the right edge, labels stay left
            header->addWidget(r.lamp);
            outer->addLayout(header);

            r.plot = new TimeSeriesPlot(this);
            r.plot->set_visible_window(600.f);   // ten minutes: this moves slowly on purpose
            r.plot->add_series(spec[i].name, spec[i].col, 1.8f, 0);
            r.plot->set_reference_line(0.f, QColor(120, 120, 120), "");
            r.plot->setMinimumHeight(90);
            r.plot->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
            outer->addWidget(r.plot, 1);    // stretch 1: the traces absorb the window's height
        }
    }

    void CalibrationViewer::update_values(const Eigen::Matrix<float, rc::calib::P_COUNT, 1>& value,
                                          const Eigen::Matrix<float, rc::calib::P_COUNT, 1>& sigma,
                                          int informed_mask, float condition, int episodes)
    {
        if (not isVisible())
            return;      // traces only advance while someone is looking; hidden costs nothing

        for (int i = 0; i < rc::calib::P_COUNT; ++i)
        {
            auto& r = rows_[i];
            const float v = value[i] * r.scale;
            const float s = sigma[i] * r.scale;
            // A zero sigma is "not solved yet", never "exact". Printing it as +/- 0.000 would claim
            // certainty the estimator has not got, which is the same failure this whole window
            // exists to prevent -- so it is shown as unknown instead.
            r.value->setText(s > 0.f
                ? QString("%1 ± %2 %3").arg(v, 8, 'f', 3).arg(s, 6, 'f', 3).arg(r.unit)
                : QString("%1 ± ?      %2").arg(v, 8, 'f', 3).arg(r.unit));
            const bool informed = (informed_mask >> i) & 1;
            // "learning" = this window shrank the posterior. "not asked" = the driving contained
            // none of the covariate this parameter needs, so the value is the previous one held,
            // NOT a measurement. Naming it that way is the whole reason the lamp exists.
            r.lamp->setText(informed ? "learning" : "not asked");
            r.lamp->setStyleSheet(informed
                ? "font-size: 11px; color: #2ecc71; font-weight: bold;"
                : "font-size: 11px; color: #7f8c8d;");
            r.plot->add_point(rows_[i].name->text().toStdString(), v);
        }

        // Condition number of the CORRELATION-normalised information matrix: how nearly two
        // parameters have become indistinguishable in the data seen so far. ~1 is healthy; large
        // means some direction is unobserved, e.g. turning at a constant rate makes the gyro's scale
        // and its bias collinear and no estimator can separate them.
        summary_->setText(QString("episodes %1     conditioning %2%3")
                              .arg(episodes, 4)
                              .arg(condition, 7, 'f', 1)
                              .arg(condition > 50.f ? "   <- two parameters are becoming collinear"
                                                    : ""));
    }
}
