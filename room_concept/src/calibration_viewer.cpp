#include "calibration_viewer.h"

#include <QHBoxLayout>
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
        resize(900, 760);   // tall enough that four full-width traces are each readable
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
        reset->setToolTip("Forget every measurement and delete the saved evidence, returning BOTH\n"
                          "blocks — the six motion parameters and the four camera-mount ones — to\n"
                          "their priors. Use it after changing something physical about the robot:\n"
                          "a wheel, a mount, the base kinematics. Measurements taken before such a\n"
                          "change describe a different machine.");
        connect(reset, &QPushButton::clicked, this, [this]
        {
            const auto answer = QMessageBox::question(
                this, "Reset calibration",
                "Forget every measurement and delete the saved window?\n\n"
                "This discards the driving episodes, any closed pivots, AND the camera-mount "
                "evidence. A closed pivot is several minutes of the robot turning in place and "
                "cannot be reproduced by ordinary driving.\n\n"
                "All ten parameters return to their priors and report NOT informed.",
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
              "of the six.<br>"
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
              "<b>k_omega — gyro / rotation scale</b><br>"
              "How much more (or less) rotation the odometry reports than actually happened, as a "
              "fraction. This is the parameter the calibration pivot exists to measure, and the only "
              "one a closure can check without a map.<br><br>"
              "<b>Learns from:</b> turning — it loads on <i>d_theta</i>.<br>"
              "<b>★Confounded with gyro bias</b> whenever the robot turns at a steady rate: at fixed "
              "omega, d_theta and elapsed time are proportional, so the two columns are collinear and "
              "no estimator can separate them. Turning at DIFFERENT rates is what breaks the tie." },
            { "gyro bias",         float(180.0 / M_PI), "deg/s", QColor(39, 174, 96),
              "<b>b_omega — gyro bias</b><br>"
              "A constant phantom rotation rate the gyro reports while the robot is perfectly still. "
              "It integrates with TIME rather than with motion, so it is what makes a parked robot's "
              "heading drift.<br><br>"
              "<b>Learns from:</b> elapsed time in the heading residual — it loads on "
              "<i>duration</i>.<br>"
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

        // ── The CAMERA block ─────────────────────────────────────────────────────────────────────
        // A separate estimator, so a separate section with its own conditioning line. Not merged
        // into the six above: different stream, no shared covariate, and its count is corner PAIRS
        // rather than driving episodes — two numbers both labelled "episodes" would be a quiet lie.
        // The `why` text comes from rc::camcal::param_why() rather than being duplicated here, so
        // the tooltip and the estimator can never drift apart.
        {
            auto* hdr = new QLabel("camera mount  —  RGB corners against LiDAR corners, pose-free", this);
            hdr->setStyleSheet("font-size: 12px; font-weight: bold; color: #b8c4cc; "
                               "padding-top: 10px; border-top: 1px solid #3a4045;");
            hdr->setToolTip("<div style='width: 380px'>Estimated from the SAME physical corner seen "
                            "by both sensors, with the localiser's pose absent from the residual "
                            "entirely — so these cannot absorb a pose error, and a heading error "
                            "cannot masquerade as a boresight.</div>");
            outer->addWidget(hdr);
            cam_which_ = new QLabel("camera: —", this);
            cam_which_->setStyleSheet("font-family: monospace; font-size: 11px; color: #7f8c8d;");
            cam_which_->setToolTip("<div style='width: 400px'>Which camera these four parameters "
                                   "describe. They are a property of ONE mount — body&rarr;zed sits at "
                                   "1.08 m and body&rarr;ricoh at 1.42 m — so a value learned for one "
                                   "says nothing about the other. The evidence is kept in a separate "
                                   "file per camera for that reason, and a file from the wrong camera "
                                   "is refused rather than merged.</div>");
            outer->addWidget(cam_which_);
            cam_summary_ = new QLabel("pairs 0", this);
            cam_summary_->setStyleSheet("font-family: monospace; font-size: 11px; color: #95a5a6;");
            outer->addWidget(cam_summary_);

            const struct { const char* name; float scale; const char* unit; QColor col; }
            cspec[] = {
                { "cam pitch",  180.f / static_cast<float>(M_PI), "deg", QColor(230, 126, 34) },
                { "cam height", 1000.f,                           "mm",  QColor(155, 89, 182) },
                { "cam yaw",    180.f / static_cast<float>(M_PI), "deg", QColor(26, 188, 156) },
                { "cam dt",     1.f,                              "x",   QColor(149, 165, 166) },
            };
            static_assert(std::size(cspec) == static_cast<std::size_t>(rc::camcal::P_COUNT),
                          "add a row here whenever rc::camcal::Param gains a parameter");

            for (int i = 0; i < rc::camcal::P_COUNT; ++i)
            {
                auto& r = cam_rows_[i];
                r.scale = cspec[i].scale;
                r.unit  = cspec[i].unit;
                r.why   = rc::camcal::param_why(i).data();

                auto* header = new QHBoxLayout();
                r.name = new QLabel(cspec[i].name, this);
                r.name->setStyleSheet(QString("font-size: 12px; font-weight: bold; color: %1;")
                                          .arg(cspec[i].col.name()));
                r.name->setToolTip(QString("<div style='width: 380px'>%1</div>").arg(r.why));
                r.value = new QLabel("--", this);
                r.value->setStyleSheet("font-family: monospace; font-size: 12px; color: #e6e9ea;");
                r.value->setToolTip(QString("<div style='width: 380px'>%1</div>").arg(r.why));
                r.lamp = new QLabel(this);
                header->addWidget(r.name);
                header->addSpacing(12);
                header->addWidget(r.value);
                header->addStretch(1);
                header->addWidget(r.lamp);
                outer->addLayout(header);

                r.plot = new TimeSeriesPlot(this);
                r.plot->set_visible_window(600.f);
                r.plot->add_series(cspec[i].name, cspec[i].col, 1.8f, 0);
                r.plot->set_reference_line(0.f, QColor(120, 120, 120), "");
                r.plot->setMinimumHeight(56);
                r.plot->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
                outer->addWidget(r.plot, 1);
            }

            loop_label_ = new QLabel("sensor triangle: no corner seen by two cameras yet", this);
            loop_label_->setStyleSheet("font-family: monospace; font-size: 11px; color: #7f8c8d;");
            loop_label_->setToolTip(
                "<div style='width: 420px'><b>The sensor triangle.</b><br><br>"
                "Each camera's residual against the LiDAR is <i>(camera error) + (LiDAR corner "
                "error)</i>, and with ONE camera those are inseparable — a systematic in the LiDAR's "
                "corner detector is indistinguishable from a camera mount error and would be "
                "corrected into the mount, silently.<br><br>"
                "Differencing two cameras against the <b>same</b> corner cancels the LiDAR term:<br>"
                "&nbsp;&nbsp;• large individual residuals, <b>small</b> difference &rarr; the fault "
                "is the LiDAR<br>"
                "&nbsp;&nbsp;• <b>large</b> difference &rarr; the two cameras disagree<br><br>"
                "Three sensors let you <i>attribute</i> error; two only let you measure "
                "disagreement. In degrees, not pixels — a pixel is 0.128° on the zed and 0.188° on "
                "the ricoh, so a pixel residual is not comparable across cameras.<br><br>"
                "This is the only reading in this window that needs no ground truth, so it is the "
                "one that still works on the real robot.</div>");
            outer->addWidget(loop_label_);
        }
    }

    void CalibrationViewer::update_loop_closure(double du_deg, double dv_deg,
                                                double sd_du, double sd_dv, long n)
    {
        if (loop_label_ == nullptr) return;
        if (n <= 0)
        {
            loop_label_->setText("sensor triangle: no corner seen by two cameras yet");
            loop_label_->setStyleSheet("font-family: monospace; font-size: 11px; color: #7f8c8d;");
            return;
        }
        loop_label_->setText(QString("sensor triangle: %1 shared corners   du %2 ± %3°   dv %4 ± %5°")
                                 .arg(n).arg(du_deg, 0, 'f', 4).arg(sd_du, 0, 'f', 4)
                                 .arg(dv_deg, 0, 'f', 4).arg(sd_dv, 0, 'f', 4));
        const bool tight = std::abs(du_deg) < 0.05 and std::abs(dv_deg) < 0.05;
        loop_label_->setStyleSheet(tight
            ? "font-family: monospace; font-size: 11px; color: #2ecc71;"
            : "font-family: monospace; font-size: 11px; color: #e67e22;");
    }

    void CalibrationViewer::update_camera(const Eigen::Matrix<float, rc::camcal::P_COUNT, 1>& value,
                                          const Eigen::Matrix<float, rc::camcal::P_COUNT, 1>& sigma,
                                          int informed_mask, float condition, long pairs,
                                          const std::string& camera)
    {
        if (cam_which_ != nullptr and not camera.empty())
            cam_which_->setText(QString("camera: %1").arg(QString::fromStdString(camera)));
        for (int i = 0; i < rc::camcal::P_COUNT; ++i)
        {
            auto& r = cam_rows_[i];
            if (r.value == nullptr) continue;
            const float v = value[i] * r.scale;
            const float sg = sigma[i] * r.scale;
            // Same rule as the motion block: a zero sigma is "not solved yet", never "exact".
            r.value->setText(sg > 0.f
                ? QString("%1 ± %2 %3").arg(v, 8, 'f', 3).arg(sg, 6, 'f', 3).arg(r.unit)
                : QString("%1 ± ?      %2").arg(v, 8, 'f', 3).arg(r.unit));
            const bool informed = (informed_mask >> i) & 1;
            r.lamp->setText(informed ? "learning" : "not asked");
            r.lamp->setToolTip(QString("<div style='width: 380px'>%1%2</div>")
                                   .arg(informed
                                            ? QStringLiteral("<b>Learning:</b> the views seen have "
                                                             "outweighed the prior.<br><br>")
                                            : QStringLiteral("<b>Not asked:</b> the robot has not "
                                                             "seen the corners that identify this, "
                                                             "so the value shown is still the prior "
                                                             "— not a measurement of zero.<br><br>"))
                                   .arg(r.why));
            r.lamp->setStyleSheet(informed
                ? "font-size: 11px; color: #2ecc71; font-weight: bold;"
                : "font-size: 11px; color: #7f8c8d;");
            r.plot->add_point(r.name->text().toStdString(), v);
        }
        if (cam_summary_ != nullptr)
            cam_summary_->setText(
                QString("pairs %1     conditioning %2%3")
                    .arg(pairs, 6).arg(condition, 7, 'f', 1)
                    // Pitch and height are KNOWN to be collinear here and near-wall driving did not
                    // break it, so name the pair rather than leaving the reader to guess which two.
                    .arg(condition > 50.f ? "   <- collinear; pitch/height is the usual pair"
                                          : ""));
    }

    void CalibrationViewer::update_values(const Eigen::Matrix<float, rc::calib::P_COUNT, 1>& value,
                                          const Eigen::Matrix<float, rc::calib::P_COUNT, 1>& sigma,
                                          int informed_mask, float condition, int episodes)
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
            // "learning" = this window shrank the posterior. "not asked" = the driving contained
            // none of the covariate this parameter needs, so the value is the previous one held,
            // NOT a measurement. Naming it that way is the whole reason the lamp exists.
            r.lamp->setText(informed ? "learning" : "not asked");
            // ★"NOT ASKED" IS A QUESTION, SO ANSWER IT WHERE IT IS ASKED. The lamp says the data has
            // not outweighed the prior; the useful next thing to know is what motion WOULD, and that
            // is exactly what the parameter's own note says. Prepending the verdict keeps the two
            // together instead of leaving the reader to pair them up.
            r.lamp->setToolTip(QString("<div style='width: 380px'>%1%2</div>")
                                   .arg(informed
                                            ? QStringLiteral("<b>Learning:</b> the motion has "
                                                             "outweighed the prior.<br><br>")
                                            : QStringLiteral("<b>Not asked:</b> the robot has not "
                                                             "made the motion that identifies this, "
                                                             "so the value shown is still the prior "
                                                             "— not a measurement of zero.<br><br>"))
                                   .arg(QString::fromUtf8(r.why)));
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
