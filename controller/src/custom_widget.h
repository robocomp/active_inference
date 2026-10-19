/*
 *    Copyright (C) 2020 by YOUR NAME HERE
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    RoboComp is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
	\brief
	@author authorname
*/



#ifndef CUSTOMWIDGET_H
#define CUSTOMWIDGET_H

#if Qt5_FOUND
	#include <QtWidgets>
#else
	#include <QtGui>
	#include <QtWidgets>
#endif

#include <cmath>
#include <limits>

#include "../../common/dashboard/timeseries_plot.h"
#include "motion_verdict.h"
#include "controller_mission_panel.h"


class Custom_widget : public QWidget
{
public:
    explicit Custom_widget(QWidget *parent = nullptr) : QWidget(parent)
    {
        auto *main_layout = new QVBoxLayout(this);
        // The window's size is the USER's (it is saved and restored), never the content's. With the default
        // constraint a long status text in any row raises the layout's minimum width and Qt GROWS the
        // window to fit it; that grown size was then saved and every restart came back wider (916 -> 1142 px
        // measured). Without a constraint an over-long row is clipped instead; the user can still widen it.
        main_layout->setSizeConstraint(QLayout::SetNoConstraint);
        main_layout->setContentsMargins(8, 8, 8, 8);
        main_layout->setSpacing(8);

        // ── Row 1: what the base is being told to do, and what arrival is waiting on ──
        // QLCDNumber rather than a formatted label: these are the numbers you WATCH while the robot
        // moves, and a proportional label re-flows as digits change, which makes a jittering value hard
        // to read precisely when it matters. Fixed-width segments do not move.
        auto *pose_panel = new QFrame(this);
        pose_panel->setFrameShape(QFrame::StyledPanel);
        pose_panel->setFrameShadow(QFrame::Raised);

        // ★TWO LINES, NOT ONE. A single line of nine LCDs set the window's minimum width; the command
        // readings (what the base is told) go on the first line, session + arrival on the second.
        auto *pose_lines = new QVBoxLayout(pose_panel);
        pose_lines->setContentsMargins(4, 3, 4, 3);
        pose_lines->setSpacing(2);
        auto *pose_row = new QHBoxLayout;
        pose_row->setSpacing(3);
        pose_lines->addLayout(pose_row);

        const auto make_lcd = [&](QWidget *parent, int digits, const QString &tip)
        {
            auto *lcd = new QLCDNumber(digits, parent);
            lcd->setSegmentStyle(QLCDNumber::Flat);
            lcd->setFrameShape(QFrame::NoFrame);
            // ★WIDTH FOLLOWS THE DIGIT COUNT. It was a flat 58 px for every LCD, so a 5-digit reading
            // reserved as much room as an 8-digit one and the row's MINIMUM width — which is what the
            // window can never be dragged below — carried that slack seven times over.
            lcd->setMinimumSize(9 * digits + 4, 24);
            // ...and it must not grow past what it needs either: the default policy is Expanding, so a
            // wide window handed the surplus to the LCDs rather than to the plots below, which is the
            // other half of why this row looked spread out.
            lcd->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
            lcd->setToolTip(tip);
            lcd->display(QStringLiteral("---"));
            return lcd;
        };

        // ── LIDAR RATE, FIRST THING IN THE ROW ───────────────────────────────────────────────
        // The rate the CONTROLLER actually processes scans at, in wall time — not the sensor's
        // nominal rate. It sits first because when the overlay lags, this is the number that says
        // whether the cause is upstream (frames are not arriving) or downstream (they are arriving
        // and something else is late), and that is the first fork in every such diagnosis.
        pose_row->addWidget(new QLabel("lidar", pose_panel));
        lidar_hz_lcd_ = make_lcd(pose_panel, 4,
            QStringLiteral("LiDAR scans per second ACCEPTED and processed by this controller, in wall "
                           "time over a ~2 s window.\n"
                           "NOT the sensor's nominal rate, and not the stamp-spacing estimate — those "
                           "are blind to dropped frames.\n"
                           "The overlay is registered at the newest scan's own stamp, so this rate "
                           "bounds how stale the drawn cloud can be: 20 Hz = 50 ms, 10 Hz = 100 ms.\n"
                           "Decays toward 0 during a stall rather than freezing at the last good value."));
        pose_row->addWidget(lidar_hz_lcd_);
        pose_row->addWidget(new QLabel("Hz", pose_panel));

        pose_row->addWidget(new QLabel("adv", pose_panel));
        cmd_adv_lcd_ = make_lcd(pose_panel, 5, QStringLiteral("Commanded forward speed (mm/s)."));
        pose_row->addWidget(cmd_adv_lcd_);
        pose_row->addWidget(new QLabel("side", pose_panel));
        cmd_side_lcd_ = make_lcd(pose_panel, 5, QStringLiteral("Commanded lateral speed (mm/s)."));
        pose_row->addWidget(cmd_side_lcd_);
        pose_row->addWidget(new QLabel("rot", pose_panel));
        cmd_rot_lcd_ = make_lcd(pose_panel, 5, QStringLiteral("Commanded rotation (rad/s)."));
        pose_row->addWidget(cmd_rot_lcd_);

        // Alerts (LiDAR stall, recovery). Hidden while nothing is wrong, so the row stays numeric —
        // but a stalled sensor must never be silent just because the row is now made of digits.
        alert_label_ = new QLabel(QString(), pose_panel);
        alert_label_->setVisible(false);
        alert_label_->setStyleSheet("QLabel { background-color: #c0392b; color: white; font-weight: bold;"
                                    " border-radius: 4px; padding: 2px 8px; }");
        pose_row->addWidget(alert_label_);
        pose_row->addStretch();

        // Second line: session totals, then what the arrival test is waiting on.
        pose_row = new QHBoxLayout;
        pose_row->setSpacing(3);
        pose_lines->addLayout(pose_row);

        // ★A FIXED GAP, NOT A STRETCH. Two addStretch() calls used to float the session pair mid-row and
        // push the goal pair to the far right, which reads as three widely separated islands and gives
        // the window no reason to be narrow. A fixed gap keeps the groups legibly apart while letting
        // the row pack to its natural width, and the single trailing stretch below takes up whatever
        // slack is left rather than distributing it between the numbers.
        // ── SESSION TOTALS ──────────────────────────────────────────────────────────────────────
        // Distance driven and time elapsed since the agent started — every mission, click target and
        // affordance, NOT per run. MissionRunner integrates both, but only while a mission is RUNNING,
        // so nothing ever counted a target or the drive back to a start point. LCDs like the rest of
        // this row: these tick continuously, and a proportional label re-flows as digits change.
        // ★UNITS TRAIL THE NUMBER, and there is no group label. "run" named the pair but left both
        // readings unitless, so the odometer and the clock were two anonymous numbers side by side;
        // "12.4 m." and "07:31 mins." each say what they are without a caption to bind them to.
        // A stretch on BOTH sides floats the pair mid-row between the command LCDs and the goal
        // readout, instead of leaving it crowded against the right end.
        session_dist_lcd_ = make_lcd(pose_panel, 6,
            QStringLiteral("Distance driven this session (m, or km past 1000).\n"
                           "All missions, targets and affordances since the agent started."));
        pose_row->addWidget(session_dist_lcd_);
        pose_row->addWidget(new QLabel("m.", pose_panel));
        session_time_lcd_ = make_lcd(pose_panel, 8,
            QStringLiteral("Wall time since the agent started (mm:ss, or h:mm:ss past an hour)."));
        pose_row->addWidget(session_time_lcd_);
        pose_row->addWidget(new QLabel("mins.", pose_panel));

        pose_row->addSpacing(14);

        // Right end: what the ARRIVAL test is waiting on — remaining distance and heading error.
        pose_row->addWidget(new QLabel("d", pose_panel));
        goal_dist_lcd_ = make_lcd(pose_panel, 5,
            QStringLiteral("Remaining linear distance to the end of the planned path (m)."));
        pose_row->addWidget(goal_dist_lcd_);
        pose_row->addWidget(new QLabel("\u03b8", pose_panel));
        goal_theta_lcd_ = make_lcd(pose_panel, 5,
            QStringLiteral("Angular error to the commanded facing yaw (deg).\n"
                           "Blank when the target carries none. Amber = rotating in place to align."));
        pose_row->addWidget(goal_theta_lcd_);
        // The ONE stretch, at the end: slack goes here rather than between the readings.
        pose_row->addStretch();

        main_layout->addWidget(pose_panel);

        // ---- toolbar row ----
        auto *toolbar = new QFrame(this);
        toolbar->setFrameShape(QFrame::StyledPanel);
        auto *toolbar_layout = new QHBoxLayout(toolbar);
        toolbar_layout->setContentsMargins(4, 2, 4, 2);
        toolbar_layout->setSpacing(6);
        lidar_toggle_btn = new QPushButton("Lidar points", toolbar);
        lidar_toggle_btn->setCheckable(true);
        lidar_toggle_btn->setChecked(false);
        toolbar_layout->addWidget(lidar_toggle_btn);
        toolbar_layout_ = toolbar_layout;


        // ── WHY IS THE ROBOT NOT MOVING (motion_verdict.h) — permanent, replaces the old STUCK chip ──────────────
        // One chip, always visible, taking the toolbar's slack: the binding cause, its detail and since-when, coloured
        // by who has to act (driving / deliberate / operator / waiting on another agent / blocked by the world / fault).
        // The escape manoeuvre the STUCK chip used to show is one of its causes (ESCAPING). Full text in the tooltip;
        // the label is elided to its width so a long detail never sets the window's minimum width.
        why_label_ = new QLabel(QStringLiteral(" no verdict yet "), toolbar);
        why_label_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
        why_label_->setMinimumWidth(120);
        why_label_->setTextFormat(Qt::PlainText);
        why_label_->setStyleSheet(why_style("#7f8c8d", "white"));
        toolbar_layout->addWidget(why_label_, /*stretch=*/1);

        // Which affordance the epistemic planner picked. It belongs beside the drive controls — it is
        // the answer to "why is it going there", not a pose readout.
        toolbar_layout->addWidget(new QLabel("affordance", toolbar));
        selected_affordance_value_ = new QLabel(toolbar);
        // CLICKABLE: it opens the affordance program window. An affordance is a small program with
        // steps, and the name alone says only WHICH one is running, never how far in it is or what is
        // holding it up.
        selected_affordance_value_->setCursor(Qt::PointingHandCursor);
        selected_affordance_value_->setToolTip(QStringLiteral("Click to open the affordance program "
                                                              "— steps, progress, and what is blocking."));
        selected_affordance_value_->installEventFilter(this);
        selected_affordance_value_->setTextFormat(Qt::RichText);
        selected_affordance_value_->setTextInteractionFlags(Qt::TextSelectableByMouse);
        // Bounded: affordance names come from graph nodes and can be long, and an unbounded label at the
        // end of the row sets the window's minimum width. The full text stays in the tooltip.
        selected_affordance_value_->setMaximumWidth(240);
        toolbar_layout->addWidget(selected_affordance_value_);

        main_layout->addWidget(toolbar);

        // ---- mission row ----
        // All of it lives in MissionPanel; this widget only decides WHERE it goes.
        // Created later by attach_mission_panel(), because it needs the worker's callbacks.
        mission_row_index = main_layout->count();

        frame = new QFrame(this);
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Sunken);
        frame->setMinimumSize(320, 320);
        main_layout->addWidget(frame, 3);

        // Affordance EFE time-series panel, below the 2D view.
        auto *efe_panel = new QFrame(this);
        efe_panel->setFrameShape(QFrame::StyledPanel);
        efe_panel->setFrameShadow(QFrame::Sunken);
        auto *efe_layout = new QVBoxLayout(efe_panel);
        efe_layout->setContentsMargins(4, 2, 4, 2);
        efe_layout->setSpacing(2);
        efe_layout->addWidget(new QLabel("Affordance EFE score (gain − λ·dist; higher = selected)", efe_panel));
        affordance_efe_plot = new rc::TimeSeriesPlot(efe_panel);
        affordance_efe_plot->setMinimumHeight(80);
        affordance_efe_plot->set_visible_window(60.f);
        efe_layout->addWidget(affordance_efe_plot, 1);
        main_layout->addWidget(efe_panel, 1);

        // ── RUNNING J, directly below the EFE score ──────────────────────────────────────────────
        // ── WHAT THE WHEELS ARE ACTUALLY ASKED TO DO ────────────────────────────────────────────
        // The commanded advance and rotation, sampled at the OUTPUT rate (every command the motion
        // commander emits, ~40 Hz) rather than at the GUI rate — smoothness is a property of the signal
        // the base receives, and sampling it at the redraw rate would alias exactly the jitter worth
        // seeing. Both live in roughly 0..0.8 (m/s and rad/s), so they share an axis legibly.
        // ★This replaced the L-adaptation plot (objective cross_rms vs constraint rot_per_m). That pair
        // was fed from MissionRunner::summary() and therefore read zero unless a MISSION was running —
        // through every clicked target and every affordance drive, which is most of what the robot does.
        auto *j_panel = new QFrame(this);
        j_panel->setFrameShape(QFrame::StyledPanel);
        j_panel->setFrameShadow(QFrame::Sunken);
        auto *j_layout = new QVBoxLayout(j_panel);
        j_layout->setContentsMargins(4, 2, 4, 2);
        j_layout->setSpacing(2);
        // ★THE CAPTION IS A NAME, NOT A LEGEND. It read out every series, its units, what the top of the
        // sigma line meant and what a gap meant — ~150 characters, which set a floor under the whole
        // window's width for text nobody needs on the tenth glance. The panel is named here and
        // EXPLAINED on hover, so none of it is lost and none of it costs width.
        auto *j_caption = new QLabel("Commanded velocity  |  σxy", j_panel);
        j_caption->setToolTip(
            QStringLiteral("adv (m/s) and rot (rad/s), from whoever is driving — this controller's own\n"
                           "output loop when it is commanding, otherwise the shared graph's robot_ref_*\n"
                           "(a joystick, say). A GAP means nothing is commanding the base.\n\n"
                           "σxy is the localisation sigma the speed limiter is reading, mapped onto the\n"
                           "same axis: the TOP of the velocity band is PoseXYStdStop, i.e. the throttle\n"
                           "is floored. It breaks when the limiter is not in the loop.\n\n"
                           "adv_meas / rot_meas (orange / teal) are MEASURED base motion, drawn only when\n"
                           "nothing publishes a command at all — a joystick drive, where robot_ref_*\n"
                           "stays at zero. They are what the robot DID, not what it was told."));
        j_layout->addWidget(j_caption);
        mission_j_plot = new rc::TimeSeriesPlot(j_panel);
        mission_j_plot->setMinimumHeight(80);
        // ~30 s: long enough to hold a whole approach and its arrival, short enough that individual
        // command steps are still distinguishable. A 120 s window compressed them into a smear.
        mission_j_plot->set_visible_window(30.f);
        // Registered HERE, on the GUI thread at construction, so the output thread only ever calls
        // add_point() — which is mutex-guarded and documented safe from any thread. Registering lazily
        // from that thread would be the one unguarded moment.
        mission_j_plot->add_series("adv", QColor(55, 55, 55), 2.0f);
        mission_j_plot->add_series("rot", QColor(70, 130, 200), 2.0f);
        // ── THE LOCALISATION UNCERTAINTY THAT IS THROTTLING THE BASE, ON THE SAME AXES ───────────
        // sigma_xy is metres, so it is mapped onto the m/s axis before it gets here (stop knee -> max
        // adv, clamped) — see ControllerDisplay::set_uncertainty_trace_value. Without that mapping it
        // would either sit invisibly along the bottom or, once past the knee, blow the auto-scale and
        // squash the two traces it exists to be read against.
        // Thin and violet: it is a REFERENCE line, not a command, and neither the near-black adv nor the
        // blue rot can be confused with it. The question it answers at a glance is the one that took a
        // whole session to answer offline — when adv sags, is this line high?
        mission_j_plot->add_series("sigma", QColor(170, 60, 175), 1.4f);
        // ── WHAT THE BASE ACTUALLY DID, WHEN NOBODY PUBLISHES WHAT IT WAS TOLD ───────────────────
        // Only ever drawn while no command is available at all — a joystick drive, where robot_ref_*
        // stays at zero (measured: 0.640 m/s of real motion against a 0.000 reference). Gaps the rest
        // of the time, so they cost nothing whenever a command exists. Separate series because these
        // are MEASURED: folding them into adv/rot would change what the panel means without saying so.
        // Warm pair, clear of the near-black adv, the blue rot and the violet sigma.
        mission_j_plot->add_series("adv_meas", QColor(200, 110, 20), 2.0f);
        mission_j_plot->add_series("rot_meas", QColor(20, 150, 140), 2.0f);
        j_layout->addWidget(mission_j_plot, 1);
        main_layout->addWidget(j_panel, 1);
    }
	~Custom_widget()
    {

    }

    // Session totals. Dedups: called every cycle, and re-rendering an unchanged string churns Qt for
    // nothing (the same reasoning as set_cmd_vel below). 0.1 m and 1 s are the displayed resolutions.
    void set_session(float metres, float seconds)
    {
        if (session_dist_lcd_ != nullptr
            and (not std::isfinite(session_metres_shown_) or std::abs(metres - session_metres_shown_) >= 0.1f))
        {
            session_metres_shown_ = metres;
            session_dist_lcd_->display(metres >= 1000.f
                ? QString::number(static_cast<double>(metres) / 1000.0, 'f', 2) + "k"
                : QString::number(static_cast<double>(metres), 'f', 1));
        }
        const int secs = static_cast<int>(seconds);
        if (session_time_lcd_ != nullptr and secs != session_secs_shown_)
        {
            session_secs_shown_ = secs;
            const int h = secs / 3600, m = (secs / 60) % 60, sec = secs % 60;
            session_time_lcd_->display(h > 0
                ? QStringLiteral("%1:%2:%3").arg(h).arg(m, 2, 10, QLatin1Char('0'))
                                            .arg(sec, 2, 10, QLatin1Char('0'))
                : QStringLiteral("%1:%2").arg(m, 2, 10, QLatin1Char('0'))
                                         .arg(sec, 2, 10, QLatin1Char('0')));
        }
    }

    // Scans/second the controller ACCEPTED, in wall time. Dedups like set_cmd_vel: it is pushed every
    // cycle and re-displaying an unchanged reading churns Qt for nothing.
    // ★"---" for "not measured yet" is NOT the same as 0.0, which means the frames STOPPED. A display
    // that showed the same thing for both would hide a stall behind a blank.
    void set_lidar_hz(float hz)
    {
        if (lidar_hz_lcd_ == nullptr) return;
        if (hz < 0.f)
        {
            if (not std::isfinite(lidar_hz_shown_)) return;
            lidar_hz_shown_ = std::numeric_limits<float>::quiet_NaN();
            lidar_hz_lcd_->display(QStringLiteral("---"));
            return;
        }
        if (std::isfinite(lidar_hz_shown_) and std::abs(hz - lidar_hz_shown_) < 0.05f) return;
        lidar_hz_shown_ = hz;
        lidar_hz_lcd_->display(QString::number(static_cast<double>(hz), 'f', 1));
    }

    // The commanded base velocity, as numbers rather than a sentence. Dedups: this is called on every
    // command (~20 Hz) and re-displaying an unchanged value churns Qt for nothing.
    void set_cmd_vel(float adv_mm_s, float side_mm_s, float rot_rps)
    {
        const auto put = [](QLCDNumber *lcd, float v, int decimals, float &shown)
        {
            if (lcd == nullptr or (std::isfinite(shown) and std::abs(v - shown) < 1e-4f)) return;
            shown = v;
            lcd->display(QString::number(static_cast<double>(v), 'f', decimals));
        };
        put(cmd_adv_lcd_, adv_mm_s, 0, cmd_adv_shown_);
        put(cmd_side_lcd_, side_mm_s, 0, cmd_side_shown_);
        put(cmd_rot_lcd_, rot_rps, 2, cmd_rot_shown_);
    }

    // Alerts that are not numbers (LiDAR stall, recovery). Empty hides the badge.
    void set_cmd_vel_text(const QString &text)
    {
        if (alert_label_ == nullptr or text == alert_shown_) return;
        alert_shown_ = text;
        alert_label_->setText(text);
        alert_label_->setVisible(not text.isEmpty());
    }

    // The CURRENT affordance is the thing you are looking for on this row, so it is coloured and bold;
    // the previous one is context and is dimmed. Formatting lives here rather than in the worker: the
    // worker knows which affordance is selected, not what colour "selected" should be.
    // "none" is deliberately styled as absence (grey, italic) — a selection and the lack of one must not
    // read the same at a glance.
    void set_affordance_clicked(std::function<void()> cb) { affordance_clicked_ = std::move(cb); }

    void set_selected_affordance(const QString &current, const QString &previous)
    {
        if (selected_affordance_value_ == nullptr) return;
        const bool have = not current.isEmpty() and current != QStringLiteral("none");
        const QString html =
            (have ? QStringLiteral("<span style='color:#2e86de; font-weight:bold;'>%1</span>")
                  : QStringLiteral("<span style='color:#7f8c8d; font-style:italic;'>%1</span>"))
                .arg(current.toHtmlEscaped())
            + QStringLiteral("<span style='color:#95a5a6;'> &nbsp;(prev: %1)</span>")
                  .arg(previous.toHtmlEscaped());
        if (html == affordance_shown_) return;   // called every cycle; don't churn Qt
        affordance_shown_ = html;
        selected_affordance_value_->setText(html);
        selected_affordance_value_->setToolTip(current + QStringLiteral("   (prev: ") + previous + ")");
    }

    // The escape manoeuvre is shown by the WHY chip now (cause ESCAPING); kept so callers need not change.
    void set_stuck_active(bool active) { stuck_active_shown_ = active; }

    // The WHY chip (motion_verdict.h). Called every frame from present(); dedups text, tooltip and colour.
    void set_motion_verdict(const rc::why::View &v)
    {
        if (why_label_ == nullptr) return;
        if (not v.valid)
            return;   // keeps "no verdict yet" until the first control pass selects one
        const char *bg = "#7f8c8d", *fg = "white";
        switch (v.klass)
        {
            case rc::why::Klass::Driving:        bg = "#27ae60"; break;
            case rc::why::Klass::Deliberate:     bg = "#2980b9"; break;
            case rc::why::Klass::Operator:       bg = "#7f8c8d"; break;
            case rc::why::Klass::WaitingOnAgent: bg = "#f1c40f"; fg = "black"; break;
            case rc::why::Klass::BlockedByWorld: bg = "#e67e22"; break;
            case rc::why::Klass::Fault:          bg = "#c0392b"; break;
        }
        const QString head = QString::fromStdString(std::format("{} {} · {}", v.driving ? "▶ MOVING" : "■ STOPPED",
                                                                rc::why::cause_name(v.cause),
                                                                rc::why::format_duration(v.duration_ms)));
        const QString full = head + QStringLiteral(" — ") + QString::fromStdString(v.detail);
        const QString text = why_label_->fontMetrics().elidedText(full, Qt::ElideRight,
                                                                 std::max(80, why_label_->width() - 20));
        if (text != why_text_shown_) { why_text_shown_ = text; why_label_->setText(text); }
        const QString tip = full + QStringLiteral("\n\nclass: ") + QString::fromUtf8(rc::why::klass_name(v.klass).data(),
                                                                                   int(rc::why::klass_name(v.klass).size()))
                          + QStringLiteral("\nLogged as [why] on change; every change in why_log.jsonl.");
        if (tip != why_tip_shown_) { why_tip_shown_ = tip; why_label_->setToolTip(tip); }
        const QString style = why_style(bg, fg);
        if (style != why_style_shown_) { why_style_shown_ = style; why_label_->setStyleSheet(style); }
    }
    static QString why_style(const char *bg, const char *fg)
    {
        return QString("QLabel { background-color: %1; color: %2; font-weight: bold; border-radius: 4px; padding: 2px 8px; }")
            .arg(bg, fg);
    }

    // What the ARRIVAL test is waiting on, at the right end of the pose row. `yaw_err_rad` is nullopt
    // when the target carries no commanded facing yaw (a plain mouse target), in which case theta reads
    // "---" rather than 0 — an absent constraint and a satisfied one must not look alike.
    void set_goal_distance(std::optional<float> dist_m, std::optional<float> yaw_err_rad, bool aligning)
    {
        if (goal_dist_lcd_ != nullptr)
        {
            const QString d = dist_m.has_value()
                ? QString::number(static_cast<double>(*dist_m), 'f', 2) : QStringLiteral("---");
            if (d != goal_dist_shown_) { goal_dist_shown_ = d; goal_dist_lcd_->display(d); }
        }
        if (goal_theta_lcd_ != nullptr)
        {
            const QString th = yaw_err_rad.has_value()
                ? QString::number(static_cast<double>(*yaw_err_rad) * 180.0 / M_PI, 'f', 1)
                : QStringLiteral("---");
            if (th != goal_theta_shown_) { goal_theta_shown_ = th; goal_theta_lcd_->display(th); }
            if (aligning != goal_aligning_shown_)
            {
                goal_aligning_shown_ = aligning;
                // Amber while rotating in place, so an alignment that hunts instead of converging is
                // visible immediately rather than inferred from the robot's behaviour.
                goal_theta_lcd_->setStyleSheet(aligning ? "QLCDNumber { background-color: #e67e22; }" : "");
            }
        }
    }


public:
    QFrame *frame = nullptr;
    rc::MissionPanel *mission_panel = nullptr;
    int mission_row_index = 0;   // where attach_mission_panel() inserts the panel
    QHBoxLayout *toolbar_layout_ = nullptr;   // top line; hosts the drive button

    void attach_mission_panel(rc::MissionPanel *panel)
    {
        mission_panel = panel;
        if (panel == nullptr)
            return;
        if (auto *lay = qobject_cast<QVBoxLayout *>(layout()); lay != nullptr)
            lay->insertWidget(mission_row_index, panel);
        // The drive control is the primary action, so it goes FIRST on the top toolbar rather than at the
        // end of the mission row. The panel still owns it; this only decides where it is shown.
        if (toolbar_layout_ != nullptr and panel->drive_button() != nullptr)
            toolbar_layout_->insertWidget(0, panel->drive_button());
        if (toolbar_layout_ != nullptr and panel->pause_button() != nullptr)
            toolbar_layout_->insertWidget(1, panel->pause_button());
    }

    // The status line (controller_status_line.h), directly under the mission row (attach_mission_panel must run
    // first). It took the place of the door-passage row.
    void attach_status_line(QWidget *line)
    {
        if (line == nullptr) return;
        if (auto *lay = qobject_cast<QVBoxLayout *>(layout()); lay != nullptr)
            lay->insertWidget(mission_row_index + (mission_panel != nullptr ? 1 : 0), line);
    }

    rc::TimeSeriesPlot *affordance_efe_plot = nullptr;
    rc::TimeSeriesPlot *mission_j_plot = nullptr;   // commanded adv/rot trace, below the EFE panel
    QPushButton *lidar_toggle_btn = nullptr;

protected:
    // QLabel carries no clicked() signal and this widget has no Q_OBJECT (the fleet dashboard
    // convention), so the affordance name's click is hit-tested here instead of pulling in a subclass
    // just to get one signal. Mapped through the label's own parent, because it lives in the toolbar
    // frame and its geometry() is in THAT frame's coordinates, not this widget's.
    // ★AN EVENT FILTER, NOT mousePressEvent. The label sets TextSelectableByMouse (so a long node name
    // can be copied), and that makes QLabel CONSUME the press — it never reaches this widget's handler,
    // so hit-testing there would silently never fire. A filter sees the event before the label does.
    // Returns false on purpose: the label still gets it, so selecting the text keeps working.
    bool eventFilter(QObject *watched, QEvent *event) override
    {
        if (watched == selected_affordance_value_ and event->type() == QEvent::MouseButtonRelease
            and affordance_clicked_)
            if (const auto *me = static_cast<QMouseEvent *>(event); me->button() == Qt::LeftButton)
                affordance_clicked_();
        return QWidget::eventFilter(watched, event);
    }

private:
        QLCDNumber *lidar_hz_lcd_ = nullptr;
    float lidar_hz_shown_ = std::numeric_limits<float>::quiet_NaN();
    QLCDNumber *cmd_adv_lcd_ = nullptr;
    QLCDNumber *cmd_side_lcd_ = nullptr;
    QLCDNumber *cmd_rot_lcd_ = nullptr;
    QLCDNumber *goal_dist_lcd_ = nullptr;
    QLCDNumber *goal_theta_lcd_ = nullptr;
    QLabel *alert_label_ = nullptr;
    QString alert_shown_;
    float cmd_adv_shown_ = std::numeric_limits<float>::quiet_NaN();
    float cmd_side_shown_ = std::numeric_limits<float>::quiet_NaN();
    float cmd_rot_shown_ = std::numeric_limits<float>::quiet_NaN();
    QString goal_dist_shown_, goal_theta_shown_;
    QLabel *selected_affordance_value_ = nullptr;
    QString affordance_shown_;
    QLabel *why_label_ = nullptr;   // the WHY chip (set_motion_verdict)
    QString why_text_shown_, why_tip_shown_, why_style_shown_;

    std::function<void()> affordance_clicked_;   // opens the affordance program window
    QLCDNumber *session_dist_lcd_ = nullptr;   // session totals, top row
    QLCDNumber *session_time_lcd_ = nullptr;

    float session_metres_shown_ = std::numeric_limits<float>::quiet_NaN();
    int session_secs_shown_ = -1;
    bool stuck_active_shown_ = false;
    bool goal_aligning_shown_ = false;
};
#endif
