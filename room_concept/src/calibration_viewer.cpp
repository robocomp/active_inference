#include "calibration_viewer.h"

#include <QFrame>
#include <QHBoxLayout>
#include <algorithm>
#include <iterator>
#include <QSizePolicy>
#include <QVBoxLayout>
#include <QMessageBox>
#include <QPushButton>
#include <cmath>

namespace rc
{
    CalibrationViewer::CalibrationViewer(QWidget* parent) : QDialog(parent)
    {
        setWindowTitle("Self-calibration");
        resize(1180, 800);  // wide enough for two camera columns side by side, and tall enough
                            // that each full-width motion trace is still readable
        auto* outer = new QVBoxLayout(this);

        summary_ = new QLabel(this);
        summary_->setStyleSheet("font-family: monospace; font-size: 11px; color: #cfd3d6;");
        outer->addWidget(summary_);

        // ── Reset to the priors ──────────────────────────────────────────────────────────────────
        // Deletes the saved window as well as clearing it in memory. A reset that left the file
        // behind would be undone by the next restart, which is not what anyone pressing this means.
        // Confirmed, because the thing it discards can be hours of driving and minutes of pivoting
        // that ordinary motion does not reproduce.
        auto* reset = new QPushButton("Reset to priors", this);
        reset->setToolTip("Forget every measurement and delete the saved evidence, returning the seven\n"
                          "motion parameters and the DRIVING camera's four mount parameters to their\n"
                          "priors. Use it after changing something physical about the robot: a wheel,\n"
                          "a mount, the base kinematics. Measurements taken before such a change\n"
                          "describe a different machine.\n\n"
                          "★ It does NOT clear the other camera columns: each extra calibration\n"
                          "channel keeps its own evidence file, and nothing here deletes it. Say so\n"
                          "rather than let a column survive a reset that claimed to clear it.");
        connect(reset, &QPushButton::clicked, this, [this]
        {
            const auto answer = QMessageBox::question(
                this, "Reset calibration",
                "Forget every measurement and delete the saved window?\n\n"
                "This discards the driving episodes, any closed pivots, AND the camera-mount "
                "evidence OF THE DRIVING CAMERA. A closed pivot is several minutes of the robot "
                "turning in place and cannot be reproduced by ordinary driving.\n\n"
                "Those eleven parameters return to their priors and report NOT informed. The other "
                "camera columns keep their own evidence files and are NOT cleared.",
                QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
            if (answer == QMessageBox::Yes and on_reset_) on_reset_();
        });
        outer->addWidget(reset);

        // ── LAYOUT: THE TRACE GETS THE WHOLE WIDTH ────────────────────────────────────────────────
        // A grid with the labels in their own columns leaves the plot only whatever is left over on
        // the right, which is the narrowest part of the window and the part carrying the actual
        // information. So each parameter is a SECTION instead: a thin header row of labels, then a
        // full-width trace under it that takes all the vertical stretch. Four traces reading edge to
        // edge is the whole reason this window exists rather than another strip in the side panel.
        // ★ SIZE IS DEDUCED AND THEN CHECKED, never declared as [P_COUNT]. Declaring the size lets
        // too-few initialisers compile: the rest are value-initialised, so `name` becomes a null
        // char* and the loop below constructs a std::string from it. That is not hypothetical --
        // P_COUNT went 4 -> 6 and this array was updated one build later, and the binary in between
        // segfaulted with "basic_string: construction from null". The static_assert turns that into
        // a compile error the next time a parameter is added.
        // ★EACH PARAMETER CARRIES ITS OWN EXPLANATION. A label reading "wheel mismatch" and a trace in
        // mrad/m tell you a number is moving; they do not tell you what the robot is doing wrong when
        // it moves, what motion would pin it down, or why it may sit at its prior for ever. That last
        // one matters most: five of these six are UNOBSERVABLE on a diet the robot happens not to be
        // eating, and a flat trace then means "never asked", not "measured to be zero". The `why`
        // text says which covariate each one loads on, because that is the whole of the answer to
        // "why is this one not learning?".
        const struct { const char* name; float scale; const char* unit; QColor col; const char* why; }
        spec[] = {
            { "translation scale", 100.f,               "%",     QColor(41, 128, 185),
              "<b>k_v — translation odometry scale</b><br>"
              "How much further (or less far) the wheels say the robot went than it really did, as a "
              "fraction. +1% means a commanded 10 m reads as 10.1 m.<br><br>"
              "<b>Learns from:</b> forward travel — it loads on <i>d_forward</i> in the along-track "
              "residual. Every traversal to an affordance feeds it, so it is usually the best-known "
              "of the seven.<br>"
              "<b>Stays at its prior when:</b> the robot only turns." },
            { "mount yaw",         float(180.0 / M_PI), "deg",   QColor(241, 196, 15),
              "<b>eps_yaw — body/sensor mount yaw offset</b><br>"
              "A fixed angular error between the frame the odometry reports in and the frame the "
              "robot actually drives in. It makes straight-line driving curve away sideways, "
              "steadily, without any turning being commanded.<br><br>"
              "<b>Learns from:</b> forward travel seen in the CROSS-track residual — it loads on "
              "<i>-d_forward</i> there. Driving straight is what reveals it; a pivot cannot.<br>"
              "<b>Distinguished from lateral scale by:</b> which component of the motion it rides on "
              "— this one on forward travel, that one on sideways travel." },
            { "gyro scale",        100.f,               "%",     QColor(192, 57, 43),
              "<b>k_omega — GYRO rotation scale</b><br>"
              "How much more (or less) rotation the gyro reports than actually happened, as a "
              "fraction. Acts on the gyro's heading factor only; the wheels have their own "
              "(wheel rot. scale). This is the parameter the calibration pivot exists to measure, and "
              "the only one a closure can check without a map.<br><br>"
              "<b>Learns from:</b> turning that the GYRO carried into the prediction — it loads on "
              "the gyro-weighted rotation <i>th_gyro</i>, scaled by P(moving), so rotation the rest "
              "filter withheld teaches it nothing.<br>"
              "<b>★Confounded with gyro bias</b> whenever the robot turns at a steady rate: at fixed "
              "omega, d_theta and elapsed time are proportional, so the two columns are collinear and "
              "no estimator can separate them. Turning at DIFFERENT rates is what breaks the tie." },
            { "gyro bias",         float(180.0 / M_PI), "deg/s", QColor(39, 174, 96),
              "<b>b_omega — gyro bias</b><br>"
              "A constant phantom rotation rate the gyro reports, whether or not the robot turns. "
              "It integrates with TIME rather than with motion. While parked the rest filter keeps it "
              "out of the prediction, so it shows up as heading drift DURING motion.<br><br>"
              "<b>Learns from:</b> gyro-weighted moving time in the heading residual — it loads on "
              "<i>-t_gyro</i> scaled by P(moving). Since 2026-10-04: parked seconds used to be counted "
              "here and diluted the estimate (arm 7).<br>"
              "<b>★Confounded with gyro scale</b> at any single rotation rate (see above). It is "
              "separable only across motion at different angular speeds, which is why a "
              "constant-rate pivot buys this parameter nothing." },
            { "lateral scale",     100.f,               "%",     QColor(155, 89, 182),
              "<b>k_lat — lateral (sideways) odometry scale</b><br>"
              "The same error as translation scale, but on sideways motion. Only meaningful on a base "
              "that can actually move sideways — on a differential-drive robot there is no such "
              "motion and this can never be excited.<br><br>"
              "<b>Learns from:</b> sideways travel — it loads on <i>d_lateral</i> in the cross-track "
              "residual.<br>"
              "<b>Stays at its prior when:</b> the base never strafes, which for most of this robot's "
              "day it does not." },
            { "wheel mismatch",    1000.f,              "mrad/m",QColor(230, 126, 34),
              "<b>dk_wheel — per-wheel effective-radius mismatch</b><br>"
              "Unequal wheels, so driving straight quietly turns the robot. Measured as radians of "
              "unintended heading change per metre travelled — a physical asymmetry (tyre wear, "
              "pressure, load), not a sensing error.<br><br>"
              "<b>Learns from:</b> forward travel seen in the HEADING residual — it loads on "
              "<i>d_forward</i> there.<br>"
              "<b>Distinguished from gyro scale by:</b> gyro scale rides on rotation, this rides on "
              "distance. Driving straight separates them; turning on the spot does not." },
            { "wheel rot. scale",  100.f,               "%",     QColor(52, 152, 219),
              "<b>k_omega_w — WHEEL rotation scale</b><br>"
              "How much more rotation the WHEELS report than happened: wheel radius over track width, "
              "plus the scrubbing a skid-steered turn adds (5-8% long on the simulated Shadow; arm 7 "
              "learnt +9%). A different error "
              "from the gyro's, so it gets its own number.<br><br>"
              "<b>Learns from:</b> rotation on segments where the wheels carry weight in the heading — "
              "slow turns and the edges of stops. On fast turns the gyro takes the heading and this "
              "is barely excited, which is honest: the wheels said little there.<br>"
              "<b>Distinguished from gyro scale by:</b> which factor the rotation was weighted into." },
            { "lever x (lat.)",    1000.f,              "mm",    QColor(127, 140, 141),
              "<b>lever_x — helios LiDAR offset from the axle midpoint, LATERAL</b><br>"
              "How far the LiDAR really sits sideways from where the robot description says. A sensor "
              "off the spin axis traces a circle when the robot turns, so the localiser reports a "
              "translation the wheels never made.<br><br>"
              "<b>Learns from:</b> ROTATION only — per correction, (I - R(dθ)ᵀ)·lever lands in the "
              "translation residual. Straight driving cannot see it.<br>"
              "<b>Acts only</b> with LidarMountApply (on the LiDAR points). Plan 2026-10-05." },
            { "lever y (fwd.)",    1000.f,              "mm",    QColor(149, 165, 166),
              "<b>lever_y — helios LiDAR offset from the axle midpoint, FORWARD</b><br>"
              "As lever x, along the driving direction.<br><br>"
              "<b>Learns from:</b> ROTATION only. Separated from the mount yaw (which rides on forward "
              "travel) by covariate.<br>"
              "<b>Acts only</b> with LidarMountApply (on the LiDAR points). Plan 2026-10-05." },
        };
        static_assert(std::size(spec) == static_cast<std::size_t>(rc::calib::P_COUNT),
                      "add a row here whenever rc::calib::Param gains a parameter");

        for (int i = 0; i < rc::calib::P_COUNT; ++i)
        {
            auto& r = rows_[i];
            r.scale = spec[i].scale;
            r.unit  = spec[i].unit;

            auto* header = new QHBoxLayout();
            r.name = new QLabel(spec[i].name, this);
            r.name->setStyleSheet(QString("font-size: 12px; font-weight: bold; color: %1;")
                                      .arg(spec[i].col.name()));
            // Rich text so the tooltip can carry structure; a fixed width so a paragraph does not
            // become one unreadable line across the desktop.
            r.name->setToolTip(QString("<div style='width: 380px'>%1</div>").arg(spec[i].why));
            // The lamp beside it says LEARNING or NOT ASKED, and the tooltip is where "not asked
            // about what?" is answered — so give it the same explanation rather than a bare word.
            r.why = spec[i].why;
            r.value = new QLabel("--", this);
            r.value->setStyleSheet("font-family: monospace; font-size: 12px; color: #e6e9ea;");
            r.lamp = new QLabel(this);
            r.value->setToolTip(QString("<div style='width: 380px'>%1</div>").arg(spec[i].why));
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
            r.plot->setMinimumHeight(64);   // six rows now; still readable, still full width
            r.plot->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
            outer->addWidget(r.plot, 1);    // stretch 1: the traces absorb the window's height
        }

        // The camera mount columns and the sensor-triangle line were removed 2026-10-29 with the camera
        // mount calibrator (camera extrinsics: robot_concept's graph + overlay, estimated by sensor_calibration).
    }

    void CalibrationViewer::update_values(const Eigen::Matrix<float, rc::calib::P_COUNT, 1>& value,
                                          const Eigen::Matrix<float, rc::calib::P_COUNT, 1>& sigma,
                                          int informed_mask, int applied_mask, float condition,
                                          int episodes)
    {
        // ★ NO VISIBILITY GATE. It used to return early when hidden, so the traces began at the
        // moment the window was opened and every parameter appeared to start learning right then —
        // the estimator has been running since startup, but nothing on screen said so, and the window
        // showed a flat line from t=0 that was really t=opened. A viewer that misrepresents WHEN a
        // thing was learnt is worse than no viewer, because the plot looks like evidence.
        // The cost of not gating is six labels and twelve deque pushes per cycle; Qt does not paint a
        // hidden widget, which was the only expensive part.

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
            const bool applied  = (applied_mask >> i) & 1;
            // "learning" = this window shrank the posterior. "not asked" = the driving contained
            // none of the covariate this parameter needs, so the value is the previous one held,
            // NOT a measurement. Naming it that way is the whole reason the lamp exists.
            // ★ AND WHETHER IT ACTS. Informed is not the same as applied: a parameter masked out by
            //   MotionCalibApplyMask (or with MotionCalibApply off) learns but never touches the
            //   odometry, and the old lamp said "learning" for both.
            r.lamp->setText(not informed ? "not asked" : applied ? "learning · APPLIED" : "learning · not applied");
            // ★"NOT ASKED" IS A QUESTION, SO ANSWER IT WHERE IT IS ASKED. The lamp says the data has
            // not outweighed the prior; the useful next thing to know is what motion WOULD, and that
            // is exactly what the parameter's own note says. Prepending the verdict keeps the two
            // together instead of leaving the reader to pair them up.
            r.lamp->setToolTip(QString("<div style='width: 380px'>%1%2</div>")
                                   .arg(informed
                                            ? QStringLiteral("<b>Learning:</b> the motion has "
                                                             "outweighed the prior. APPLIED = it is "
                                                             "correcting the odometry now; not applied "
                                                             "= estimated only (apply mask).<br><br>")
                                            : QStringLiteral("<b>Not asked:</b> the robot has not "
                                                             "made the motion that identifies this, "
                                                             "so the value shown is still the prior "
                                                             "— not a measurement of zero.<br><br>"))
                                   .arg(QString::fromUtf8(r.why)));
            r.lamp->setStyleSheet(not informed ? "font-size: 11px; color: #7f8c8d;"
                                  : applied ? "font-size: 11px; color: #2ecc71; font-weight: bold;"
                                            : "font-size: 11px; color: #e67e22; font-weight: bold;");
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
